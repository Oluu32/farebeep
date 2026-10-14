"""Human support relay and the signed "my tickets" link.

Support threads live in the user's ChatState row. An explicit ask, a
money-trouble phrase, or two agent failures opens a thread and pings
ADMIN_ALERT_PHONE. The founder replies with "R <phone> <text>", lists
threads with /open, and closes one with /done [phone].

The signed "my tickets" link (_ticket_link_token / _ticket_link_phone)
lives here too: it is the same HMAC session-token family, and booking.py
verifies it.

Helpers that main owns (_say, _notify_admin, _looks_negated) and the
SessionLocal factory are resolved through FareBeep.main at call time, not
imported, so tests that monkeypatch them on main still take effect. This
module must never import main at load time (main imports it).
"""
import logging
import re
import time
import uuid

from FareBeep import chatstate
from FareBeep.config import META_APP_SECRET, SUPPORT_TICKET_TTL_HOURS
from FareBeep.models import ChatState, Subscription, User, utcnow

logger = logging.getLogger("farebeep.main")


def _main():
    """The FareBeep.main module, looked up at call time (see docstring)."""
    from FareBeep import main
    return main


_TICKET_LINK_TTL = 900            # 15 minutes


# ---------------------------------------------------------------------------
# Human support relay (Phase 1) - the founder IS the support desk.
# An explicit human ask (SUPPORT / "speak to a human"), a refund or
# payment-failure phrase, or two agent failures in a row opens a thread
# in the user's ChatState row and pings ADMIN_ALERT_PHONE with context.
# The founder answers from their own chat: "R <phone> <text>" relays,
# /open lists threads, /done [phone] closes one. No helpdesk vendor, no
# queue UI - capture, relay, close. (A SupportTicket table earns its
# keep only when volume makes history worth querying.)
# ---------------------------------------------------------------------------
_SUPPORT_PATTERNS = (
    r"\bsupport\b",
    r"\bspeak to (a |an )?(human|agent|person)\b",
    r"\btalk to (a |an )?(human|agent|person)\b",
    r"\bhuman agent\b",
    r"\bcustomer (care|service|support)\b",
    r"\brefund\b", r"\bcharged twice\b", r"\bdouble.?charg",
    r"\bpayment (failed|fail|issue|problem)\b", r"\bfailed payment\b",
    r"\bdidn'?t (get|receive) (my |the )?(ticket|booking)\b",
    r"\bmoney back\b", r"\bscam\b",
)
_support_res = [re.compile(p, re.IGNORECASE) for p in _SUPPORT_PATTERNS]
_SUPPORT_CLOSE_WORDS = frozenset({
    "cancel support", "end support", "resolved", "solved",
    "it works now", "all good now", "no longer needed",
})
_SUPPORT_RELAY_RE = re.compile(r"^R\s+(\+?\d[\d\s\-]{6,})\s+(.+)$",
                               re.IGNORECASE | re.DOTALL)
_support_frustration: dict = {}   # phone -> consecutive agent failures


def _admin_phone() -> str:
    """Lazily read (tests flip the env per case)."""
    from FareBeep.config import ADMIN_ALERT_PHONE
    return ADMIN_ALERT_PHONE or ""


def _norm_phone(s) -> str:
    return re.sub(r"[\s\-\(\)]", "", str(s or "")).strip()


def _is_support_ask(text: str) -> bool:
    """Explicit human ask or money-trouble phrase. Negated phrasing
    ('don't refund me') never triggers - same guard as STOP."""
    flat = (text or "").strip().lower()
    if not flat or _main()._looks_negated(flat):
        return False
    return any(rx.search(flat) for rx in _support_res)


def _ticket_is_open(ticket) -> bool:
    return bool(ticket) and ticket.get("status") == "open"


def _ticket_age_hours(ticket) -> float:
    from datetime import datetime
    try:
        opened = datetime.fromisoformat(str(ticket.get("opened_at")))
        return (utcnow() - opened).total_seconds() / 3600.0
    except Exception:
        return 0.0


def _support_append(db, phone: str, side: str, text: str) -> dict:
    """Append one turn to the thread (capped) and persist."""
    ticket = chatstate.get_support_ticket(db, phone) or {}
    msgs = ticket.setdefault("messages", [])
    msgs.append({"side": side, "at": utcnow().isoformat(),
                 "text": (text or "")[:400]})
    ticket["messages"] = msgs[-12:]
    chatstate.set_support_ticket(db, phone, ticket)
    return ticket


def _support_alert(db, user: User, ticket: dict) -> str:
    """The ops alert: who, why, context, and the exact reply syntax."""
    fare = chatstate.get_last_fare(db, user.phone) or {}
    fare_line = "none this chat"
    if fare:
        try:
            price = f"\u20a6{float(fare.get('price') or 0):,.0f}"
        except Exception:
            price = str(fare.get("price"))
        fare_line = (f"{fare.get('origin_iata', '?')}->"
                     f"{fare.get('destination_iata', '?')} "
                     f"{fare.get('flight_date', '')} {price}")
    try:
        n_beeps = db.query(Subscription).filter(
            Subscription.user_id == user.user_id,
            Subscription.paused.is_(False)).count()
    except Exception:
        n_beeps = 0
    channel = "WhatsApp" if str(user.phone).startswith("+") else "Telegram"
    lines = [f"\U0001f3a7 SUPPORT {ticket.get('id')} - "
             f"{user.name or 'Unknown'} ({user.phone}, {channel})",
             f"Trigger: {ticket.get('trigger', 'explicit')}",
             f"Fare context: {fare_line} | Active beeps: {n_beeps}"]
    msgs = ticket.get("messages") or []
    last_text = (msgs[-1].get("text", "") if msgs else "").strip()
    if last_text:
        lines.append(f"Says: \"{last_text[:400]}\"")
    lines.append(f"Reply: R {user.phone} <message> | /open | "
                 f"/done {user.phone}")
    return "\n".join(lines)


def _open_support_ticket(db, user: User, text: str, trigger: str) -> dict:
    ticket = {
        "id": "SUP-" + uuid.uuid4().hex[:6].upper(),
        "opened_at": utcnow().isoformat(),
        "trigger": trigger,
        "status": "open",
        "messages": [],
    }
    if text:
        ticket["messages"].append({"side": "user",
                                   "at": utcnow().isoformat(),
                                   "text": text[:400]})
    chatstate.set_support_ticket(db, user.phone, ticket)
    _main()._say(user.phone,
         "You've reached a human \U0001f3a7 - the FareBeep founder replies "
         "right here in this chat, usually within a few hours. Your "
         "message and trip context are attached, so no need to repeat "
         "yourself. You can still search fares meanwhile - reply "
         "CANCEL SUPPORT to hand back to the bot.",
         user.name, humanized=True)
    _main()._notify_admin(_support_alert(db, user, ticket))
    return ticket


def _trigger_label(text: str) -> str:
    flat = (text or "").lower()
    if any(w in flat for w in ("refund", "charged", "double", "scam",
                               "payment", "money", "debited", "deducted")):
        return "auto: payment/refund"
    return "explicit ask"


def _handle_support(db, user: User, text: str) -> bool:
    """The support gate. True = turn consumed. A fresh ask opens a
    thread; while one is open the user's words relay to the founder
    (the bot must not talk over a live human conversation). Stale
    threads (>SUPPORT_TICKET_TTL_HOURS) auto-close on touch."""
    ticket = chatstate.get_support_ticket(db, user.phone)
    if _ticket_is_open(ticket) and \
            _ticket_age_hours(ticket) > SUPPORT_TICKET_TTL_HOURS:
        ticket["status"] = "stale"
        chatstate.set_support_ticket(db, user.phone, ticket)
        _main()._say(user.phone,
             "(The earlier support thread was closed automatically "
             "after 48h - reply SUPPORT anytime to open a fresh one.)",
             user.name, humanized=True)
        ticket = chatstate.get_support_ticket(db, user.phone)
    flat = (text or "").strip().lower().rstrip("!.,?").strip()
    close = flat in _SUPPORT_CLOSE_WORDS or flat.startswith(
        ("solved", "resolved", "cancel support", "end support"))
    if close:
        if not _ticket_is_open(ticket):
            return False          # no thread: falls through to the bot
        ticket["status"] = "resolved"
        ticket["closed_at"] = utcnow().isoformat()
        chatstate.set_support_ticket(db, user.phone, ticket)
        _main()._notify_admin(f"\u2705 {ticket.get('id')} closed BY USER "
                      f"({user.phone}).")
        _main()._say(user.phone,
             "Support thread closed - glad we could sort it. Reply "
             "SUPPORT anytime if anything else comes up. \U0001f41d",
             user.name, humanized=True)
        return True
    if _is_support_ask(text):
        if _ticket_is_open(ticket):
            _support_append(db, user.phone, "user", text)
            _main()._notify_admin(f"\U0001f4e8 {ticket.get('id')} ({user.phone}) "
                          f"added: \"{(text or '')[:200]}\"")
        else:
            _open_support_ticket(db, user, text, trigger=_trigger_label(text))
        return True
    if _ticket_is_open(ticket):
        # Human owns this thread: everything (the commands that could
        # still run ran above) reaches the founder's chat.
        _support_append(db, user.phone, "user", text)
        _main()._notify_admin(f"\U0001f4e8 {ticket.get('id')} ({user.phone}) says: "
                      f"\"{(text or '')[:200]}\"")
        return True
    return False


def _has_open_support(phone: str) -> bool:
    """Throttle-exemption probe - only called on the burst path. Best
    effort: burst control is process-local and must never depend on the
    DB being up, so a lookup failure just means the throttle applies."""
    try:
        db = _main().SessionLocal()
        try:
            return _ticket_is_open(chatstate.get_support_ticket(db, phone))
        finally:
            db.close()
    except Exception as e:
        logger.warning("support throttle-exempt lookup failed: %s", e)
        return False


def _resolve_target(db, token: str):
    """Admin typed a number possibly without the leading '+': exact
    User.phone first, then the '+' variant. Telegram chat ids match
    exactly. Unknown tokens are refused - never blast a guess."""
    tok = _norm_phone(token)
    if not tok:
        return None
    row = db.query(User).filter(User.phone == tok).first()
    if row is None and tok.isdigit():
        row = db.query(User).filter(User.phone == "+" + tok).first()
    return row.phone if row else None


def _relay_admin_reply(admin_phone: str, token: str, body: str) -> bool:
    db = _main().SessionLocal()
    try:
        target = _resolve_target(db, token)
        if target is None:
            _main()._say(admin_phone, f"Unknown user \"{token}\" - no chat "
                 f"found. /open lists threads.", humanized=True)
            return True
        ticket = _support_append(db, target, "admin", body)
    finally:
        db.close()
    _main()._say(target, f"\U0001f3a7 FareBeep support: {body}", humanized=True)
    status = ticket.get("status")
    hint = "" if status == "open" else \
        f" (note: thread was {status}; /done {target} closes it)"
    _main()._say(admin_phone, f"\u2192 sent to {target}{hint}", humanized=True)
    return True


def _admin_list_tickets(admin_phone: str) -> bool:
    db = _main().SessionLocal()
    try:
        rows = db.query(ChatState).filter(
            ChatState.support_ticket.isnot(None)).all()
        open_rows = [(r.phone, r.support_ticket) for r in rows
                     if _ticket_is_open(r.support_ticket)]
    finally:
        db.close()
    if not open_rows:
        _main()._say(admin_phone, "No open support threads. \U0001f389",
             humanized=True)
        return True
    lines = []
    for ph, tk in open_rows:
        msgs = tk.get("messages") or []
        last = msgs[-1].get("text", "") if msgs else ""
        lines.append(f"{tk.get('id')} | {ph} | {tk.get('trigger')} | "
                     f"{_ticket_age_hours(tk):.0f}h | \"{last[:60]}\"")
    _main()._say(admin_phone, "Open support threads:\n" + "\n".join(lines) +
         "\n\nRelay: R <phone> <message> | Close: /done <phone>",
         humanized=True)
    return True


def _admin_close_ticket(admin_phone: str, token: str) -> bool:
    db = _main().SessionLocal()
    try:
        if token:
            target = _resolve_target(db, token)
            if target is None:
                _main()._say(admin_phone, f"Unknown user \"{token}\".",
                     humanized=True)
                return True
            ticket = chatstate.get_support_ticket(db, target)
            if not _ticket_is_open(ticket):
                _main()._say(admin_phone, f"No open thread for {target}.",
                     humanized=True)
                return True
        else:
            rows = db.query(ChatState).filter(
                ChatState.support_ticket.isnot(None)).all()
            open_rows = [(r.phone, r.support_ticket) for r in rows
                         if _ticket_is_open(r.support_ticket)]
            if not open_rows:
                _main()._say(admin_phone, "No open support threads.",
                     humanized=True)
                return True
            if len(open_rows) > 1:
                _main()._say(admin_phone,
                     f"{len(open_rows)} threads open - say "
                     f"/done <phone>. /open lists them.", humanized=True)
                return True
            target, ticket = open_rows[0]
        ticket["status"] = "resolved"
        ticket["closed_at"] = utcnow().isoformat()
        chatstate.set_support_ticket(db, target, ticket)
        tid = ticket.get("id")
    finally:
        db.close()
    _main()._say(target, "Support thread closed - glad we could sort it. Reply "
                 "SUPPORT anytime if anything else comes up. \U0001f41d",
         humanized=True)
    _main()._say(admin_phone, f"Closed {tid} ({target}).", humanized=True)
    return True


def _handle_admin_console(phone: str, text: str) -> bool:
    """Founder powers, riding the same webhook as everyone else.
    Only ADMIN_ALERT_PHONE holds them; anything else from that phone
    falls through so the founder can still use the bot normally."""
    admin = _admin_phone()
    if not admin or _norm_phone(phone) != _norm_phone(admin):
        return False
    t = (text or "").strip()
    m = _SUPPORT_RELAY_RE.match(t)
    if m:
        return _relay_admin_reply(phone, m.group(1), m.group(2).strip())
    up = t.upper()
    if up == "/OPEN" or up.startswith("/OPEN "):
        return _admin_list_tickets(phone)
    if up == "/DONE" or up.startswith("/DONE "):
        return _admin_close_ticket(phone, t[5:].strip())
    return False


def _note_agent_outcome(phone: str, ok: bool) -> None:
    """Two consecutive agent failures on one phone = the user got no
    answer twice - open a support thread automatically (the bot must
    not fail silently into a void)."""
    if ok:
        _support_frustration.pop(phone, None)
        return
    n = _support_frustration.get(phone, 0) + 1
    _support_frustration[phone] = n
    if n < 2:
        return
    _support_frustration.pop(phone, None)
    try:
        db = _main().SessionLocal()
        try:
            user = db.query(User).filter(User.phone == phone).first()
            if user is None or _ticket_is_open(
                    chatstate.get_support_ticket(db, phone)):
                return
            _open_support_ticket(db, user, "",
                                 trigger="auto: bot failed twice")
        finally:
            db.close()
    except Exception as e:
        logger.warning("auto support-open after agent failures failed: %s", e)


def _ops_support_rows(db) -> list:
    """Open/stale threads for the ops cockpit: identity, trigger, and the
    transcript tail so the human can read context before replying.
    (Small table: filter in Python - dialect-proof.)"""
    out = []
    for r in db.query(ChatState).filter(
            ChatState.support_ticket.isnot(None)).all():
        tk = r.support_ticket or {}
        if tk.get("status") in ("open", "stale"):
            out.append({
                "id": tk.get("id"), "phone": r.phone,
                "trigger": tk.get("trigger"),
                "status": tk.get("status"),
                "age_hours": round(_ticket_age_hours(tk), 1),
                "messages_count": len(tk.get("messages") or []),
                "messages": [{"side": m.get("side"), "text": m.get("text"),
                              "at": m.get("at")}
                             for m in (tk.get("messages") or [])[-6:]],
                "relay": f"R {r.phone} ",
            })
    return out


def _ticket_link_token(phone: str) -> str:
    """base64url(phone.expiry.hmac) - signature keyed on META_APP_SECRET."""
    import base64
    import hashlib
    import hmac
    exp = int(time.time()) + _TICKET_LINK_TTL
    payload = f"{phone}.{exp}"
    sig = hmac.new((META_APP_SECRET or "farebeep-dev").encode(),
                   payload.encode(), hashlib.sha256).hexdigest()[:32]
    return base64.urlsafe_b64encode(
        f"{payload}.{sig}".encode()).decode().rstrip("=")


def _ticket_link_phone(token: str) -> "str | None":
    """Inverse of _ticket_link_token. None when the signature, expiry or
    shape is wrong - the caller answers with a generic expired page."""
    import base64
    import hashlib
    import hmac
    try:
        padded = token + "=" * (-len(token) % 4)
        phone, exp, sig = base64.urlsafe_b64decode(
            padded.encode()).decode().rsplit(".", 2)
        expect = hmac.new((META_APP_SECRET or "farebeep-dev").encode(),
                          f"{phone}.{exp}".encode(),
                          hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(sig, expect) or int(exp) < time.time():
            return None
        return phone
    except Exception:
        return None
