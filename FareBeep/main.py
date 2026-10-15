"""FareBeep API - WhatsApp conversational layer (Meta Cloud + Twilio test).

Endpoints:
  GET  /webhook/meta    - Meta handshake (hub.challenge echo after
                           hub.verify_token check)
  POST /webhook/meta    - Meta message receiver. X-Hub-Signature-256 is
                           verified with HMAC-SHA256 over the RAW body using
                           META_APP_SECRET (the requirement: `use hmac to
                           verify X-Hub-Signature-256`).
  POST /webhook/twilio  - Twilio WhatsApp Sandbox receiver (test channel).
                           X-Twilio-Signature is verified with the account
                           auth token; the reply goes out via the REST API
                           in a background task.
  GET/POST /tools/search - ElevenLabs server tool: ledger-first fare lookup
                           (ledger hit wins; Travels247 live on miss, UPSERTed).
  POST /tools/reserve   - ElevenLabs server tool: live re-verify + 10-minute
                           booking_session + Paystack link (payment FIRST -
                           the Travels247 PNR is only issued in /webhook/paystack
                           after charge.success).
  POST /webhook/paystack- Paystack transaction events (settles the
                           10-minute booking loop; issues the Travels247 PNR on
                           success when a booking_token + travellers exist).
  GET  /health          - liveness.
  GET  /admin/ops       - ops snapshot (X-Admin-Token; closed without
                          ADMIN_TOKEN): inbound dispatch health, dead
                          letters, receipts, beeps, bookings, agent stats
  POST /admin/ops/dead-letters/{id}/replay
                        - re-dispatch one dead letter from its payload

Run:  uvicorn FareBeep.main:app --port 8000
"""
import hashlib
import hmac
import json
import logging
import os
import re
import time
from typing import Optional

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse)
from fastapi.staticfiles import StaticFiles

from FareBeep import brain, cards, ticket_delivery
from FareBeep.config import (ADMIN_TOKEN, APP_BASE_URL, CONSENT_VERSION,
                             MESSAGING_PROVIDER, META_APP_SECRET,
                             META_VERIFY_TOKEN,
                             REQUOTE_TOLERANCE_NGN, TELEGRAM_BOT_TOKEN,
                             ELEVENLABS_TOOL_SECRET,
                             GROQ_API_KEY, GUIDED_MODE)
from FareBeep.database import SessionLocal, init_db
from FareBeep.iata import city_name, resolve_iata
from FareBeep.models import BookingSession, User, utcnow
from FareBeep.notifier import MetaWhatsapp, get_notifier
from FareBeep.templating import render_html
from FareBeep.payments import verify_paystack_signature
from FareBeep.search import LedgerOnlyEngine, LedgerSearch
from FareBeep.suppliers import get_inventory_client, pick_cheapest
from FareBeep.travels247 import Travels247Error  # supplier error shape (shared contract)
from FareBeep.transactions import BookingService, PaystackError
from FareBeep.inbound import (MAX_INBOUND_ATTEMPTS,
                              _arrival_key, _claim_inbound_messages,
                              _classify_all_entries, _dispatch_single,
                              _lease_owner, _mark_processed, _phone_lock,
                              _restore_message, acquire_queued_messages,
                              recover_orphaned_inbound)
from FareBeep.support import (_handle_admin_console, _handle_support,
                              _has_open_support, _note_agent_outcome,
                              _ticket_link_token)
from FareBeep.subscriptions import (_flow_target_price, _handle_manage,
                                    _handle_stop, _is_stop_request,
                                    _looks_negated, _reply_subscribe,
                                    _reply_unsubscribe, _send_beep_flow,
                                    _tap_alert_by_phone, _tap_beep_by_phone)
from FareBeep.dialog import (
    _is_session_greeting, _ops_ledger, _handle_my_bookings, _RATE_LIMIT,
    _rate_hits, _rate_cooled, _rate_allow, _handle_incoming_message, _say,
    _get_or_create_user, _try_pick, _reply_fare,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("farebeep.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: run the startup sequence (schema init, connection
    check, orphaned-inbound recovery) on boot. No shutdown work needed."""
    _startup()
    yield


app = FastAPI(title="FareBeep - Transactional Utility", lifespan=lifespan)

# WhatsApp Flows data-exchange endpoint (the "Set a beep" screens).
from FareBeep.whatsapp.flows import router as flows_router  # noqa: E402
app.include_router(flows_router)

notifier = get_notifier()

# Which brain answered, this process. Ops-visible via GET /admin/ops so a
# silently-degrading Groq key (fallback storms) shows up instead of hiding
# in per-request logs. Per-process counters reset on restart - direction,
# not billing.
_agent_stats = {"groq_ok": 0, "groq_fail": 0, "guided_turns": 0}

# Per-chat conversational memory lives in the `chat_state` table (see
# FareBeep/chatstate.py) - NOT in RAM, so it survives deploys/restarts and
# works from any replica. Last quoted fare, last ranked list, and any
# pending follow-up question are read/written there on every turn.


def _startup():
    """Create the schema on first run (safe in both SQLite + Supabase modes)
    then verify the connection - prints the mission banner:
    '✅ Connected to Supabase Shared Ledger'. Finally, recover any inbound
    messages orphaned by a crash-after-ack (claimed, 200 sent, never
    processed) - normally zero, so boot stays fast."""
    from FareBeep.database import verify_connection
    from FareBeep.models import Base
    init_db(Base)
    verify_connection()
    try:
        recovered = recover_orphaned_inbound()
        if recovered:
            logger.warning("Startup recovered %d orphaned message(s)",
                           recovered)
    except Exception as e:
        logger.error("Startup recovery failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Meta Cloud API Webhook - RECEIVER
# ---------------------------------------------------------------------------
def _verify_meta_signature(raw_body: bytes, signature_header: str) -> bool:
    """X-Hub-Signature-256 = "sha256=" + HMAC-SHA256(META_APP_SECRET, raw_body).

    The HMAC is computed over the RAW request body - never over a parsed /
    re-serialized version.
    """
    if not signature_header or not META_APP_SECRET:
        return False
    expected = "sha256=" + hmac.new(
        META_APP_SECRET.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


# ---------------------------------------------------------------------------
# Meta inbound bridge - classify ALL messages, save-before-ack, FIFO per phone.
# a. HMAC verified above (unchanged).
# b. Every entry/change/message in the payload is classified (not just [0]).
# c. New wamids are inserted into processed_messages BEFORE the 200 ack
#    (ON CONFLICT DO NOTHING semantics via IntegrityError catch - portable
#    across Supabase Postgres and SQLite). Duplicates are dropped here.
# d. The 200 ack goes out immediately - no typing bubble, no brain, no
#    network calls in the request thread (Meta 20s deadline).
# e. Newly-claimed messages are processed in ONE background task, grouped
#    by phone in timestamp (FIFO) order, through the EXISTING concierge
#    pipeline (_handle_incoming_message / _tap_*). Booking/payment paths
#    are untouched - this bridge only changes HOW messages reach them.
# f. FAILED rows are reclaimable: when Meta redelivers the same wamid
#    (its standard retry), a row stuck at failed with attempts left is
#    re-queued instead of dup-dropped - bounded by MAX_INBOUND_ATTEMPTS.
#    done/queued/exhausted rows stay dropped (never double-book).
# g. Outbound status callbacks (sent/delivered/read/failed) are UPSERTed
#    into delivery_receipts for ops visibility. Failed receipts NEVER
#    trigger an automatic resend here - booking/payment messages must
#    not be blind-retried.
# ---------------------------------------------------------------------------

def _process_inbound_batch(messages: list) -> None:
    """Background worker: SAVED order per phone, then mark each row
    done/failed. The messages list is only the trigger - the work order
    comes from the DB (queued rows for the phone, see _arrival_key), so
    separate webhook POSTs for one customer still run deterministically,
    never in thread-scheduling order. In-memory copies of already-stored
    rows are skipped (a racing batch may have taken them); id-less
    messages run after the recorded ones. One bad message never kills
    the batch (booking safety: exceptions are contained per message).

    Crash-mid-turn honesty: a turn that dies AFTER writing a booking
    but BEFORE done WILL re-run on recovery/redelivery and can leave a
    second PENDING session (new payment_ref). Money still moves only via
    Paystack per unique ref, settle is idempotent, and the worker expiry
    sweep clears stale pendings - but exactly-once booking is NOT
    claimed here (see test_booking_crash_mid_turn_documents_duplicates).
    """
    grouped: dict = {}
    order: list = []
    for m in messages:
        phone = m.from_number or ""
        if not phone:
            continue
        if phone not in grouped:
            grouped[phone] = []
            order.append(phone)
        grouped[phone].append(m)
    for phone in order:
        # Per-phone lock (this process) + atomic lease (every process):
        # two batches for one customer can neither interleave nor take
        # each other's rows. Different customers still run parallel.
        with _phone_lock(phone):
            taken = acquire_queued_messages(_lease_owner("web"), phone=phone)
            work = []
            for t in sorted(taken,
                            key=lambda t: _arrival_key(t["payload"],
                                                       t["created_at"])):
                snap = t["payload"] or {}
                if snap.get("message_id"):
                    work.append(_restore_message(snap))
                # Payload-less rows are the sweep's job (parked, not
                # replayed) - never dispatched from here.
            for m in grouped[phone]:
                if not (m.message_id or "").strip():
                    work.append(m)  # id-less: no arrival record, run as-is
            for m in work:
                try:
                    _dispatch_single(m)
                    _mark_processed(m.message_id, "done")
                except Exception as e:
                    logger.error("Inbound handling failed (%s): %s",
                                 phone, e)
                    _mark_processed(m.message_id, "failed", error=str(e))


def _record_statuses(payload: dict) -> int:
    """Persist Meta outbound status callbacks (sent/delivered/read/failed)
    into delivery_receipts (upsert per wamid). Returns the count recorded.
    A failed receipt is ops-visible and loudly logged - but NEVER resent
    from here: booking/payment texts must not be blind-retried."""
    from FareBeep.models import DeliveryReceipt
    recorded = 0
    db = SessionLocal()
    try:
        for entry in payload.get("entry") or []:
            for change in entry.get("changes", []):
                value = change.get("value", {})
                for st in value.get("statuses", []):
                    if not isinstance(st, dict):
                        continue
                    mid = st.get("id", "")
                    if not mid:
                        continue
                    status = st.get("status", "")
                    phone = st.get("recipient_id")
                    try:
                        row = db.query(DeliveryReceipt).filter_by(
                            message_id=mid).first()
                        if row is None:
                            db.add(DeliveryReceipt(
                                message_id=mid, phone=phone, status=status))
                        else:
                            row.status = status
                            if phone:
                                row.phone = phone
                        db.commit()
                        recorded += 1
                        if status == "failed":
                            logger.warning(
                                "Outbound %s to %s FAILED "
                                "(see delivery_receipts - no auto-resend)",
                                mid, phone)
                    except Exception as e:
                        db.rollback()
                        logger.warning("Receipt upsert failed for %s: %s",
                                       mid, e)
    finally:
        db.close()
    return recorded




# Standalone greetings: the WHOLE message is the greeting, nothing else.
# (A route question containing "hi" - "hi, lagos to abuja" - is not this.)


def _send_fare_cards(user: User, fares: list, origin_iata: str,
                     destination_iata: str) -> None:
    """Follow a fare text with tappable flight cards (Meta channel only -
    other channels keep the text they already got). One card per fare,
    logo on top, price under, Book + Set-alert buttons. idx is 1-based
    so taps reuse the "reply 1, 2, 3" pick gate; a lone fare uses the
    bare BOOK / last_fare context instead."""
    if not hasattr(notifier, "send_interactive_card"):
        return  # non-Meta channel (Telegram/Twilio) - keep the text reply
    single = len(fares) == 1
    for i, fare in enumerate(fares[:3], start=1):
        idx = 0 if single else i
        notifier.send_interactive_card(
            user.phone,
            body=cards.card_body(fare, origin_iata, destination_iata),
            buttons=cards.card_buttons(fare, idx),
            image_url=cards.logo_for(fare.get("airline")),
            footer="FareBeep \u2022 verified just now")


def _help_text() -> str:
    return ("Beep! 🎫 FareBeep is your fast Nigerian flight concierge. "
            "Just tell me where and when, e.g. 'Lagos to Abuja tomorrow'\n"
            "      'BOOK Lagos to Abuja' (reserve in 10 mins)\n"
            "      'TRACK Lagos to Abuja below 80k' (price alert)\n"
            "      'Track P47123' (flight status)")


# ---------------------------------------------------------------------------
# Twilio WhatsApp Sandbox Webhook - TEST CHANNEL
# ---------------------------------------------------------------------------
def _verify_twilio_signature(raw_url: str, form: dict, signature: str) -> bool:
    """Compare X-Twilio-Signature (base64 HMAC-SHA1 over the URL + POST
    params, keyed by the account auth token) the way Twilio ships it."""
    if not signature or not MESSAGING_PROVIDER.lower() == "twilio":
        return False
    from twilio.request_validator import RequestValidator
    try:
        validator = RequestValidator(os.getenv("TWILIO_AUTH_TOKEN") or "")
        return validator.validate(raw_url, form, signature)
    except Exception as e:
        logger.warning("Twilio signature validation error: %s", e)
        return False


@app.post("/webhook/twilio")
async def twilio_webhook(request: Request, background: BackgroundTasks):
    """Twilio sandbox receiver (form-encoded, x-twilio-signature verified).

    Replies are sent by the background task THROUGH the REST API, so this
    handler always answers Twilio with an empty TwiML ack instantly.
    """
    form = dict(await request.form())
    sig = request.headers.get("X-Twilio-Signature", "")
    if not _verify_twilio_signature(str(request.url), form, sig):
        logger.warning("Twilio webhook REJECTED: bad X-Twilio-Signature")
        return Response(status_code=403)

    phone = str(form.get("From", "")).removeprefix("whatsapp:")
    text = str(form.get("Body", ""))
    if text and phone:
        background.add_task(_handle_incoming_message, phone, text)
    return Response(content="<?xml version='1.0' encoding='UTF-8'?><Response></Response>",
                    media_type="application/xml")


@app.get("/webhook/twilio")
async def twilio_verify(request: Request):
    """Twilio sandbox webhook settings may fire a GET before the POST."""
    return PlainTextResponse("FareBeep Twilio webhook is live")


# ---------------------------------------------------------------------------
# Telegram Bot API Webhook - FASTEST TEST CHANNEL
# ---------------------------------------------------------------------------
def _verify_telegram_secret(secret: str) -> bool:
    """X-Telegram-Bot-Api-Secret-Token set via setWebhook(secret_token)."""
    expected = os.getenv("TELEGRAM_WEBHOOK_SECRET") or ""
    return bool(expected) and hmac.compare_digest(expected, secret)




def _telegram_typing(chat_id: str) -> None:
    from FareBeep.notifier import TelegramBot
    TelegramBot().send_action(chat_id)  # typing… (best-effort)




# ---------------------------------------------------------------------------
# Telegram Mini App - the "Set a beep" form (the WhatsApp-Flow equivalent)
# ---------------------------------------------------------------------------
def _verify_telegram_init_data(init_data: str) -> Optional[str]:
    """Validate Telegram WebApp initData and return the chat_id (str).

    Telegram signs the payload with HMAC-SHA256 whose key is
    HMAC-SHA256(key='WebAppData', msg=<bot token>); the signed material
    is every field EXCEPT hash itself, joined by newlines, sorted.
    Returns None on any failure - an unverified page call never touches
    the ledger.
    """
    import hashlib
    if not init_data:
        return None
    try:
        from urllib.parse import parse_qsl, quote
        # parse_qsl URL-decodes values; Telegram signs the ENCODED wire
        # form, so rebuild the check string field by field with quote().
        pairs_list = parse_qsl(init_data, keep_blank_values=True)
        pairs = dict(pairs_list)
        received = pairs.get("hash", "")
        if not received:
            return None
        check = "\n".join(
            f"{k}={quote(str(v), safe='')}" for k, v in sorted(
                (k, v) for k, v in pairs_list if k != "hash"))
        # The WebApp scheme (NOT the login-widget SHA256): the HMAC key
        # itself derives from the bot token keyed by 'WebAppData'.
        secret = hmac.new(b"WebAppData", (TELEGRAM_BOT_TOKEN or "").encode(),
                          hashlib.sha256).digest()
        calc = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(calc, received):
            logger.warning("Mini App initData REJECTED (bad hash)")
            return None
        user = json.loads(pairs.get("user") or "{}")
        chat_id = str(user.get("id") or "")
        return chat_id or None
    except Exception as e:
        logger.warning("Mini App initData parse failed: %s", e)
        return None


@app.get("/mini/beep")
async def mini_beep_page():
    """The Telegram Mini App form (the 'Set a beep' equivalent of the
    WhatsApp Flow). Served as a file; identity arrives signed in initData."""
    return FileResponse(os.path.join(WEB_DIR, "mini-beep.html"))


@app.post("/mini/beep/submit")
async def mini_beep_submit(request: Request):
    """Mini App form completion -> the SAME SubscriptionMonitor.subscribe
    the WhatsApp Flow completion uses (idempotent per user+route).
    Identity comes ONLY from verified initData - never from the body."""
    body = await request.json()
    chat_id = _verify_telegram_init_data(str(body.get("initData") or ""))
    if not chat_id:
        return JSONResponse(status_code=401, content={
            "ok": False, "error": "Could not verify this Telegram session - "
                                  "reopen the form from the chat."})
    origin = resolve_iata(str(body.get("origin") or ""))
    destination = resolve_iata(str(body.get("destination") or ""))
    if not origin or not destination or origin == destination:
        return JSONResponse(status_code=422, content={
            "ok": False, "error": "Pick two different cities."})
    target = _flow_target_price(body)
    if target is None and not body.get("dropwatch"):
        return JSONResponse(status_code=422, content={
            "ok": False, "error": "Give a target price or keep drop-watch on."})
    db = SessionLocal()
    try:
        user = _get_or_create_user(db, chat_id)
        from FareBeep.alerts import SubscriptionMonitor, target_realism_note
        SubscriptionMonitor(db).subscribe(
            user.user_id, origin, destination,
            target_price=target,
            target_date=str(body.get("departure_date") or "") or None)
        date_label = str(body.get("departure_date") or "the coming weeks")
        summary = (f"Watching {city_name(origin)} -> "
                   f"{city_name(destination)} ({date_label}) - beep at "
                   f"NGN {target:,.0f} or lower." if target else
                   f"Watching {city_name(origin)} -> "
                   f"{city_name(destination)} ({date_label}) - beep on any "
                   f"genuine drop.")
        return {"ok": True, "summary": summary,
                "realism_warning": target_realism_note(db, origin,
                                                       destination, target)}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Admin ops - dead-letter visibility + replay (X-Admin-Token guarded)
# ---------------------------------------------------------------------------
def _admin_ok(request: Request) -> bool:
    """Shared-secret gate. ADMIN_TOKEN unset = the admin surface stays
    closed; 404 (not 403) so the endpoint's existence isn't advertised."""
    expected = ADMIN_TOKEN or ""
    if not expected:
        return False
    provided = request.headers.get("X-Admin-Token", "")
    return bool(provided) and hmac.compare_digest(expected, provided)








# ---------------------------------------------------------------------------
# ElevenLabs Server Tools - the conversational agent's fare API
# ---------------------------------------------------------------------------
async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_dict(value) -> dict:
    """Accept a dict or a JSON string (ElevenLabs declares objects to its
    model as type string, so callers send '{"adults":1,...}' as text).
    Anything else -> {} (callers apply their own defaults)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _verify_tool_auth(request: Request) -> bool:
    """Shared-secret check for ElevenLabs webhook-tool calls.

    In the ElevenLabs dashboard each tool gets a custom auth header
    (X-FareBeep-Tool-Secret: <ELEVENLABS_TOOL_SECRET>). When the secret
    is unset the tools stay open (local dev only - production must set
    it, otherwise anyone on the internet can trigger bookings).
    """
    if not ELEVENLABS_TOOL_SECRET:
        logger.warning("ELEVENLABS_TOOL_SECRET unset - /tools/* open")
        return True
    return hmac.compare_digest(
        request.headers.get("X-FareBeep-Tool-Secret", ""),
        ELEVENLABS_TOOL_SECRET)


def _tool_fare_summary(fare: dict, source: str, origin: str,
                       destination: str, flight_date: str) -> dict:
    """Human-readable fare summary for the AI to present (Airline, Time,
    Price), plus the machine fields it needs to book (booking_token)."""
    airline = fare.get("airline_name") or fare.get("airline") or "Airline"
    legs = ""
    if fare.get("departure_time") or fare.get("arrival_time"):
        legs = (f", {fare.get('departure_code') or origin} "
                f"{fare.get('departure_time') or ''}->"
                f"{fare.get('arrival_code') or destination} "
                f"{fare.get('arrival_time') or ''}")
    flight = f" {fare['flight_no']}" if fare.get("flight_no") else ""
    price = fare["price"]
    return {
        "found": True,
        "source": source,
        "summary": (f"{airline}{flight}, {origin}->{destination} on "
                    f"{flight_date}{legs}, NGN {price:,.0f}"),
        "airline": airline,
        "flight_no": fare.get("flight_no"),
        "departs_at": fare.get("departure_time"),
        "arrives_at": fare.get("arrival_time"),
        "price_ngn": price,
        "currency": fare.get("currency") or "NGN",
        "booking_token": (fare.get("booking_token")
                          if source == "247travels" else None),
        "origin": origin,
        "destination": destination,
        "flight_date": flight_date,
        "note": ("Prices are unusually high right now."
                 if fare.get("above_guardrail") else None),
    }


@app.api_route("/tools/search", methods=["GET", "POST"])
async def tools_search(request: Request):
    """ElevenLabs server tool: ledger-first fare lookup.

    Args (query or JSON): origin, destination (city or IATA), flight_date
    (YYYY-MM-DD), adults (default 1), phone (optional, log attribution).
    A ledger hit (<500ms, free) wins;
    on a miss Travels247 is queried live (search_mode=external) and the
    cheapest offer is UPSERTed into the ledger for the community.
    """
    if not _verify_tool_auth(request):
        logger.warning("/tools/search REJECTED: bad tool secret")
        return JSONResponse(status_code=403, content={
            "found": False, "error": "forbidden"})
    args = dict(request.query_params) if request.method == "GET" \
        else await _json_body(request)
    origin = resolve_iata(args.get("origin") or "")
    destination = resolve_iata(args.get("destination") or "")
    flight_date = str(args.get("flight_date") or "")[:10]
    adults = _as_int(args.get("adults"), 1)
    adults = max(1, adults)  # supplier rejects 0/negative passenger counts
    caller = str(args.get("phone") or "")
    if not origin or not destination or not flight_date:
        return JSONResponse(status_code=422, content={
            "found": False,
            "error": "origin, destination and flight_date (YYYY-MM-DD) "
                     "are required"})
    db = SessionLocal()
    try:
        ledger = LedgerSearch(db, live=LedgerOnlyEngine())
        hit = ledger.search(origin, destination, flight_date, verify=False)
        if hit is not None:
            logger.info("tools/search ledger hit %s->%s %s (caller=%s)",
                        origin, destination, flight_date, caller)
            return _tool_fare_summary(hit, "ledger", origin, destination,
                                      flight_date)
        sky = get_inventory_client()
        try:
            offers = await sky.search_offers(origin, destination,
                                             flight_date, adults=adults)
        finally:
            await sky.close()
        best = pick_cheapest(offers)
        if best is None:
            return {"found": False, "source": "247travels",
                    "error": f"no live offers for {origin}->{destination} "
                             f"on {flight_date}"}
        # The ledger keeps the price (no public URL exists for a Travels247
        # offer, so verify_link stays empty; the booking_token travels in
        # this response, and /tools/reserve mints a fresh one anyway).
        # Surge-guarded, no 90s hold on the voice path (reserve re-verifies).
        ledger._verify_and_upsert(origin, destination, flight_date, {
            "price": best["price"], "currency": best["currency"],
            "airline": best["airline_name"], "verify_link": None},
            verify=False)
        return _tool_fare_summary(best, "247travels", origin, destination,
                                  flight_date)
    finally:
        db.close()


@app.post("/tools/reserve")
async def tools_reserve(request: Request):
    """ElevenLabs server tool: price-lock + Paystack link (payment FIRST).

    Travels247 creates a LIVE PNR on every /reserve call, so FareBeep NEVER
    calls Travels247 reserve here - it re-verifies the price live, opens the
    10-minute booking_session (13 for bank transfer), and returns the
    Paystack link. The PNR is issued in /webhook/paystack after
    charge.success. Body: {phone, booking_token? | origin+destination+
    flight_date, passengers?, travellers?, payment_method?}.
    """
    if not _verify_tool_auth(request):
        logger.warning("/tools/reserve REJECTED: bad tool secret")
        return JSONResponse(status_code=403, content={
            "locked": False, "error": "forbidden"})
    body = await _json_body(request)
    phone = str(body.get("phone") or "").strip()
    if not phone:
        return JSONResponse(status_code=422, content={
            "locked": False, "error": "phone is required"})
    pax = _as_dict(body.get("passengers"))
    adults = _as_int(pax.get("adults"), 1)
    adults = max(1, adults)  # supplier rejects 0/negative passenger counts
    children = _as_int(pax.get("children"), 0)
    infants = _as_int(pax.get("infants"), 0)
    db = SessionLocal()
    try:
        user = _get_or_create_user(db, phone)
        sky = get_inventory_client()
        try:
            if body.get("booking_token"):
                origin = resolve_iata(body.get("origin") or "")
                destination = resolve_iata(body.get("destination") or "")
                flight_date = str(body.get("flight_date") or "")[:10]
                airline = body.get("airline")
                if not origin or not destination or not flight_date:
                    return JSONResponse(status_code=422, content={
                        "locked": False,
                        "error": "token bookings still need origin, "
                                 "destination and flight_date for the lock"})
                priced = await sky.verify_price(
                    body["booking_token"], adults=adults,
                    children=children, infants=infants)
            else:
                origin = resolve_iata(body.get("origin") or "")
                destination = resolve_iata(body.get("destination") or "")
                flight_date = str(body.get("flight_date") or "")[:10]
                if not origin or not destination or not flight_date:
                    return JSONResponse(status_code=422, content={
                        "locked": False,
                        "error": "booking_token or origin+destination+"
                                 "flight_date is required"})
                offers = await sky.search_offers(
                    origin, destination, flight_date, adults=adults,
                    children=children, infants=infants)
                best = pick_cheapest(offers)
                if best is None:
                    return {"locked": False, "source": "247travels",
                            "error": f"no live offers for "
                                     f"{origin}->{destination} on {flight_date}"}
                priced = await sky.verify_price(
                    best["booking_token"], adults=adults,
                    children=children, infants=infants)
                airline = best["airline_name"]
        except Travels247Error as e:
            logger.warning("tools/reserve Travels247 failed: %s", e)
            return JSONResponse(status_code=502, content={
                "locked": False, "error": f"live price check failed: {e}"})
        finally:
            await sky.close()
        bookings = BookingService(db)
        try:
            result = bookings.create_booking(
                user.user_id, origin, destination, flight_date,
                priced["verified_price"], airline=airline, source="247travels",
                payment_method=body.get("payment_method"))
        except PaystackError as e:
            logger.warning("tools/reserve Paystack link failed: %s", e)
            return JSONResponse(status_code=502, content={
                "locked": False, "error": f"payment link failed: {e}"})
        session = result["session"]
        details = dict(session.flight_details or {})
        details.update({
            "booking_token": priced["booking_token"],
            "passengers": {"adults": adults, "children": children,
                           "infants": infants},
            "travellers": _as_dict(body.get("travellers")),
        })
        session.flight_details = details
        db.commit()
        total = result["total_amount"]
        link = result["payment_link"]
        return {
            "locked": True,
            "payment_link": link,
            "total_amount": total,
            "payment_ref": session.payment_ref,
            "expires_at": result["expires_at"].isoformat(),
            "summary": (f"Locked {airline or 'flight'} {origin}->"
                        f"{destination} {flight_date} at NGN {total:,.0f}. "
                        f"Pay within 10 minutes: {link}"),
        }
    finally:
        db.close()


async def _maybe_issue_247travels_ticket(session,
                                      fallback_pnr: str) -> tuple:
    """Deprecated shim: ticketing now lives in FareBeep.ticketing.

    Kept so older call sites keep working; the Paystack webhook calls
    ticketing.issue_pnr directly.
    """
    from FareBeep import ticketing
    return tuple(await ticketing.issue_pnr(session, fallback_pnr))


# ---------------------------------------------------------------------------
# Paystack webhook - the other half of the settlement loop
# ---------------------------------------------------------------------------
def _notify_admin(text: str) -> None:
    """Best-effort admin alert (Refund Required etc.). Goes to
    ADMIN_ALERT_PHONE if configured, else the log (a real ops channel can
    be wired without touching the loop)."""
    from FareBeep.config import ADMIN_ALERT_PHONE
    if ADMIN_ALERT_PHONE:
        try:
            notifier.send_text(ADMIN_ALERT_PHONE, text)
            logger.info("Admin alert sent to %s", ADMIN_ALERT_PHONE)
            return
        except Exception as e:
            logger.error("Admin alert send failed: %s", e)
    logger.warning("ADMIN ALERT (no recipient configured): %s", text)


def _notify_session_user(session, text: str) -> None:
    """Send an outbound message to the session's owner (WhatsApp number or
    Telegram chat_id - whatever the user's phone column holds)."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.user_id == session.user_id).first()
    finally:
        db.close()
    if user is None:
        logger.warning("Session %s has no user row - message not sent: %s",
                       session.payment_ref, text)
        return
    notifier.send_text(user.phone, text)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    # Vendor contract drift probe: when the fare source's payload changes a
    # pinned field name, `providers.probe_contract` reports it here so churn
    # shows up as a health signal before users see a broken quote.
    from FareBeep import providers
    probe = providers.tiqwa_probe()
    if probe is None:
        return {"status": "ok", "service": "FareBeep"}
    return {"status": "ok" if probe["ok"] else "degraded",
            "service": "FareBeep", "fare_provider_probe": probe}


# ---------------------------------------------------------------------------
# Booking confirmation page - the ONLY place a user proceeds to payment.
# The WhatsApp bot sends a link here instead of a raw Paystack URL. The page
# reconfirms the quoted price (breakdown) and captures NDPA consent (version
# + timestamp + phone) BEFORE the Paystack redirect. All bookings flow
# through this page, so consent is always captured - no text-parsing needed.
# ---------------------------------------------------------------------------
_CONSENT_TEXT = (
    "By proceeding, you agree that FareBeep may collect and process your "
    "details (and, for ticket purchases, passenger/travel document data) "
    "for the purpose of booking, pricing and confirming your flight, "
    "sending you fare alerts, and contacting you about your booking. "
    "We do not share your data with third parties for their own marketing. "
    "You can stop alerts anytime by replying STOP. "
    "This page confirms your INTENT to purchase the flight shown at the "
    "quoted total - the price is locked for 10 minutes, and your ticket is "
    "issued after payment is confirmed by our payment partner. "
    "This is our current data notice (version {version}).")


def _as_naive(dt):
    """Normalise a datetime for safe comparison.

    Postgres returns timestamptz columns as offset-NAIVE datetimes, while
    models.utcnow() is offset-aware - comparing the two raises TypeError.
    Strip tzinfo from both sides before comparing (all values are UTC).
    """
    return dt.replace(tzinfo=None) if dt is not None else None


def _is_expired(session: BookingSession) -> bool:
    return _as_naive(session.expires_at) < _as_naive(utcnow())


def _booking_not_found() -> HTMLResponse:
    return HTMLResponse("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FareBeep - Booking</title></head>
<body style="font-family:system-ui,sans-serif;background:#0b1220;color:#e8eef7;
display:grid;place-items:center;min-height:100vh;margin:0">
<div style="background:#131c2e;border:1px solid #26324a;border-radius:16px;
padding:2.5rem;max-width:26rem;text-align:center">
<h1>Booking not found</h1>
<p style="color:#9fb0c9">This link is invalid or already used. Ask the bot
for a fresh fare and booking in WhatsApp.</p>
</div></body></html>""")


def _booking_closed(origin: str = "", destination: str = "") -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FareBeep - Window closed</title></head>
<body style="font-family:system-ui,sans-serif;background:#0b1220;color:#e8eef7;
display:grid;place-items:center;min-height:100vh;margin:0">
<div style="background:#131c2e;border:1px solid #26324a;border-radius:16px;
padding:2.5rem;max-width:26rem;text-align:center">
<h1>Price window closed</h1>
<p style="color:#9fb0c9">The 10-minute price lock for
{origin} &rarr; {destination} has expired. Send a new message in WhatsApp
(e.g. &ldquo;BOOK Lagos to Abuja tomorrow&rdquo;) and I&rsquo;ll re-check
the latest fare for you.</p>
</div></body></html>""")


def _booking_page(session: BookingSession, user_email: str = None) -> HTMLResponse:
    return render_html(
        "booking.html",
        origin=city_name(session.origin),
        destination=city_name(session.destination),
        flight_date=session.flight_date,
        airline=(session.flight_details or {}).get("airline") or "—",
        expires=session.expires_at.strftime("%H:%M"),
        session_id=session.id,
        airline_price=f"{session.airline_price:,.0f}",
        fees=f"{session.markup + session.processing_fee:,.0f}",
        total=f"{session.total_price:,.0f}",
        email_value=user_email or session.contact_email or "",
        consent_text=_CONSENT_TEXT.format(version=CONSENT_VERSION),
    )




_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")










# ---------------------------------------------------------------------------
# Landing website - the FareBeep home page (served from FareBeep/web/).
# Assets are static files; "/" hands out the single-page site.
# ---------------------------------------------------------------------------
WEB_DIR = Path(__file__).resolve().parent / "web"

import mimetypes

mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("image/avif", ".avif")

app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/og.png", include_in_schema=False)
def og_image():
    """The social-share card. Meta/WhatsApp/Twitter crawlers ignore SVG,
    so the SVG source is rendered to PNG once and cached in memory."""
    png_path = WEB_DIR / "og.png"
    if not png_path.exists():
        png_path = WEB_DIR / "og.svg"
    return FileResponse(png_path, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/", include_in_schema=False)
def landing():
    # no-cache (revalidate, not no-store): HTML must never go stale -
    # assets carry their own versioned URLs and cache hard instead.
    return FileResponse(WEB_DIR / "index.html",
                        headers={"Cache-Control": "no-cache"})



# Webhook endpoints (Meta, Telegram, Paystack) live in routers/webhooks.py.
# Mounted last: that module reads helpers off this module at request time.
from FareBeep.routers.webhooks import router as webhooks_router  # noqa: E402
app.include_router(webhooks_router)

# Booking/checkout/ticket and admin endpoints live in routers/.
from FareBeep.routers.booking import router as booking_router  # noqa: E402
from FareBeep.routers.admin import router as admin_router  # noqa: E402
app.include_router(booking_router)
app.include_router(admin_router)
