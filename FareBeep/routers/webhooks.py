"""Inbound webhook endpoints: Meta WhatsApp, Telegram, Paystack.

Endpoints only. Shared logic (inbound queue, dispatch, settlement,
notifications) stays in FareBeep.main and is looked up at request time
through `_m`, so monkeypatching main's globals keeps working. main.py
mounts this router after all of its helpers are defined.
"""
import logging

from fastapi import APIRouter, BackgroundTasks, Request, Response
from fastapi.responses import PlainTextResponse

from FareBeep import main as _m
from FareBeep import ticket_delivery, ticketing

logger = logging.getLogger("farebeep.main")
router = APIRouter()


# ---------------------------------------------------------------------------
# Meta Cloud API Webhook - HANDSHAKE
# ---------------------------------------------------------------------------
@router.get("/webhook/meta")
async def meta_verify(request: Request):
    """GET verification: echo hub.challenge when hub.verify_token matches."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode == "subscribe" and token == _m.META_VERIFY_TOKEN:
        logger.info("Meta webhook subscribed (handshake OK)")
        return PlainTextResponse(challenge, media_type="text/plain")
    logger.warning("Meta handshake rejected (mode=%r token=%r)", mode, token)
    return Response(status_code=403)


@router.post("/webhook/meta")
async def meta_webhook(request: Request, background: BackgroundTasks):
    raw = await request.body()

    if not _m._verify_meta_signature(raw, request.headers.get("X-Hub-Signature-256", "")):
        logger.warning("Meta webhook REJECTED: bad X-Hub-Signature-256")
        return Response(status_code=403)

    payload = await request.json()
    # Status callbacks ride the same webhook - record first so a
    # statuses-only payload still tracks delivery before the 200 ack.
    _m._record_statuses(payload)
    claimed = _m._claim_inbound_messages(_m._classify_all_entries(payload))

    # Ack Meta immediately (20s deadline); the batch runs off-thread.
    # NOTE: plain "200 OK" (not {"status":"ok"}) keeps the existing
    # webhook contract + tests green - the status code is the ack.
    if claimed:
        background.add_task(_m._process_inbound_batch, claimed)
    return Response(content="200 OK", media_type="text/plain")


@router.post("/webhook/telegram")
async def telegram_webhook(request: Request, background: BackgroundTasks):
    """Telegram Bot API receiver (JSON, secret-token header verified).

    The chat_id IS the user identity (stored where a phone number would
    live on the WhatsApp channels), so the whole conversational pipeline
    is reused unchanged.
    """
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not _m._verify_telegram_secret(secret):
        logger.warning("Telegram webhook REJECTED: bad secret token")
        return Response(status_code=403)

    update = await request.json()
    # Button taps arrive as callback_query updates (inline keyboards).
    # Route them through the SAME gates as Meta button_reply taps.
    cbq = update.get("callback_query") or {}
    if cbq:
        cb_id = str(cbq.get("id") or "")
        data = str(cbq.get("data") or "")
        cb_chat = str(((cbq.get("message") or {}).get("chat") or {})
                      .get("id") or "")
        if cb_id:
            background.add_task(_m._telegram_callback, cb_id, cb_chat, data)
        return {"ok": True}
    message = update.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = str(chat.get("id") or "")
    text = str(message.get("text") or "")
    if text and chat_id:
        # typing bubble runs as a background task (FIFO, before the reply
        # task): a dead Telegram API must never delay the 200 ack
        background.add_task(_m._telegram_typing, chat_id)
        background.add_task(_m._handle_incoming_message, chat_id, text)
    return {"ok": True}


@router.get("/webhook/telegram")
async def telegram_verify(request: Request):
    return PlainTextResponse("FareBeep Telegram webhook is live")


@router.post("/webhook/paystack")
async def paystack_webhook(request: Request):
    """Paystack event receiver - X-Paystack-Signature (HMAC-SHA512) verified
    over the RAW body. On charge.success the booking is gated by expires_at:

      now() <= expires_at  -> status = paid, mock Ticket Issued (PNR FB-XXXX)
      now() >  expires_at  -> status = expired, Refund Required admin alert
                              + the user is told a refund/price-match is coming
    """
    raw = await request.body()
    if not _m.verify_paystack_signature(
            raw, request.headers.get("x-paystack-signature", "")):
        logger.warning("Paystack webhook REJECTED: bad signature")
        return Response(status_code=403)

    payload = await request.json()
    event = payload.get("event", "")
    data = payload.get("data") or {}
    reference = str(data.get("reference", ""))
    status = str(data.get("status", ""))

    db = _m.SessionLocal()
    try:
        bookings = _m.BookingService(db)
        outcome = bookings.settle_payment(reference, status)
        logger.info("Paystack %s -> %s", reference, outcome["outcome"])

        session = outcome.get("session")
        if outcome["outcome"] == "paid":
            pnr, ticket_note, provisional = await ticketing.issue_pnr(
                session, outcome.get("pnr") or "FB-????")
            _m._notify_session_user(
                session,
                f"✅ BOOKING CONFIRMED - payment received!\n"
                f"PNR: {pnr}{provisional}\n"
                f"{_m.city_name(session.origin)} -> {_m.city_name(session.destination)} "
                f"{session.flight_date}\n"
                f"Paid: ₦{session.total_price:,.0f}\n"
                f"{ticket_note} Safe travels!")
            # Voucher when ticket-holder details are on file; otherwise
            # the /book form was skipped - collect them in chat first.
            ticket_delivery.dispatch_ticket(db, session, pnr, _m.notifier)
        elif outcome["outcome"] == "refund_required":
            _m._notify_session_user(
                session,
                "Payment was successful, but the 10-minute window closed. "
                "Your refund is being processed automatically - no "
                "questions.")
            # THE REFUND PROMISE: the airline was never ticketed, so the
            # money goes back via the Paystack API - not just an alert.
            refunded = False
            try:
                from FareBeep.payments import refund_paystack_transaction
                refund_paystack_transaction(reference)
                refunded = True
            except Exception as e:
                logger.error("Auto-refund failed for %s: %s", reference, e)
            if refunded:
                tail = "No action needed - Paystack refund issued."
            else:
                tail = ("MANUAL REFUND REQUIRED - the Paystack refund call "
                        "failed.")
            _m._notify_admin(
                f"{'AUTO-REFUNDED' if refunded else 'REFUND FAILED'}: "
                f"{reference} was paid after its 10-minute lock expired "
                f"({session.expires_at}); the airline API was NOT called. "
                f"{tail}")
        return {"ok": True, "outcome": outcome["outcome"]}
    finally:
        db.close()
