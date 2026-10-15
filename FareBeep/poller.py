"""Telegram long-polling transport - the tunnel-free fallback.

Cloudflared quick tunnels are unreliable on some ISPs (measured: repeated
"control stream encountered a failure while serving"). Polling needs NO
public URL: the bot pulls updates over an outbound connection to
api.telegram.org and feeds them into the EXACT same conversational pipeline
(_handle_incoming_message) as the webhook.

Run:  python -m FareBeep.poller     (a separate process, alongside uvicorn)

Telegram only delivers to ONE endpoint, so the poller deletes the webhook on
start. When you're back on a stable tunnel, stop the poller and re-run:
    python FareBeep/set_telegram_webhook.py <tunnel-url>
"""
import logging
import time

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("farebeep.poller")

from FareBeep.config import TELEGRAM_BOT_TOKEN, TELEGRAM_POLL_TIMEOUT  # noqa: E402

# Single-poller guard: a different advisory-lock key from serve_all's
# loops lock. Two pollers (a restarted double, a laptop + Railway)
# would EACH fetch every update -> duplicate replies, split context,
# double bookings. The second starter exits loudly instead.
POLLER_LOCK_KEY = 8391091

# Held lock state: keep BOTH referenced for the process lifetime. The
# advisory lock dies with the connection, and the connection dies with
# the session - dropping either reference lets GC silently release the
# lock and a second poller starts double-answering users.
_HELD_LOCK = None


def _acquire_poller_lock():
    """Exit unless this process owns the Telegram-poll lock.

    Returns the held connection (kept in _HELD_LOCK), or None on the
    SQLite fallback (dev/tests: no advisory locks there).
    """
    global _HELD_LOCK
    from FareBeep.database import DATABASE_PROVIDER, SessionLocal
    if DATABASE_PROVIDER == "SQLite (fallback)":
        return None
    from sqlalchemy import text
    session = SessionLocal()
    conn = session.connection()
    try:
        held = conn.execute(
            text("select pg_try_advisory_lock(:k)"),
            {"k": POLLER_LOCK_KEY}).scalar()
    except Exception:
        logger.exception("Poller lock check failed")
        held = False
    if held:
        _HELD_LOCK = (session, conn)
        return conn
    try:
        session.close()
    except Exception:
        pass
    logger.error("Another Telegram poller already holds the lock - "
                 "exiting now (no double replies). Stop the other poller "
                 "first if this one should take over.")
    raise SystemExit(0)


def _api(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"


def _handle(chat_id: str, text: str) -> None:
    from FareBeep.main import _handle_incoming_message
    from FareBeep.notifier import TelegramBot
    try:
        # Typing indicator first (best-effort): the chat feels alive
        # while the brain works. Never blocks the reply.
        TelegramBot().send_action(chat_id)
        _handle_incoming_message(chat_id, text)
    except Exception as e:
        logger.error("Polled message failed (%s): %s", chat_id, e)


def poll_once(client: httpx.Client, offset: int) -> int:
    """Fetch one batch of updates, route each message, return next offset.

    message + callback_query: fare-card button taps ride the same
    _dispatch_tap gates as the Meta channel (see interactive._telegram_callback)."""
    resp = client.get(
        _api("getUpdates"),
        params={"timeout": TELEGRAM_POLL_TIMEOUT, "offset": offset,
                "allowed_updates": ["message", "callback_query"]})
    resp.raise_for_status()
    for update in resp.json().get("result") or []:
        offset = max(offset, int(update["update_id"]) + 1)
        cbq = update.get("callback_query")
        if cbq:
            from FareBeep.interactive import _telegram_callback
            _telegram_callback(str(cbq.get("id") or ""),
                               str(((cbq.get("message") or {})
                                    .get("chat") or {}).get("id") or ""),
                               str(cbq.get("data") or ""))
            continue
        msg = update.get("message") or {}
        chat_id = str((msg.get("chat") or {}).get("id") or "")
        text = str(msg.get("text") or "")
        if text and chat_id:
            _handle(chat_id, text)
    return offset


def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is empty - set it in FareBeep/.env")
        return
    # Own the single-poller lock BEFORE touching Telegram (a second
    # poller would double-answer every user).
    _acquire_poller_lock()
    # Ledger tables + additive migrations + drift alarm: the poller used
    # to skip init_db, so model changes (e.g. chat_state.agent_history)
    # broke it silently while web/worker were fine.
    from FareBeep.database import init_db
    from FareBeep.models import Base
    init_db(Base)
    client = httpx.Client(timeout=TELEGRAM_POLL_TIMEOUT + 15)
    client.get(_api("deleteWebhook")).raise_for_status()
    logger.info("Telegram polling started (no tunnel needed) - webhook deleted")
    offset = 0
    while True:
        try:
            offset = poll_once(client, offset)
        except Exception as e:
            logger.warning("Poll failed: %s", e)
            time.sleep(3)


if __name__ == "__main__":
    main()
