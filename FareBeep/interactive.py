"""Interactive replies: fare-card button taps (Meta and Telegram),
Telegram callback queries, WhatsApp Flow completions, and boarding-pass
forwards.

Every handler here is a thin entry point; the conversation turn itself
(_handle_incoming_message) and the say/notifier plumbing stay in
FareBeep.main. Those names are resolved through main at call time, so
tests that monkeypatch them on main still take effect. This module must
never import main at load time (main and the routers import it).
"""
import logging

from FareBeep import cards, ticket_delivery
from FareBeep.iata import city_name
from FareBeep.models import User
from FareBeep.subscriptions import _flow_target_price

logger = logging.getLogger("farebeep.main")


def _main():
    """The FareBeep.main module, looked up at call time (see docstring)."""
    from FareBeep import main
    return main


_MAX_PASS_BYTES = 8 * 1024 * 1024   # boarding passes are small; be strict


def _download_whatsapp_media(media_id: str) -> "tuple[bytes, str] | None":
    """Fetch media bytes from Meta's Graph API. Returns (bytes, mime)."""
    import httpx
    from FareBeep.config import META_API_VERSION, META_ACCESS_TOKEN
    try:
        with httpx.Client(timeout=30) as client:
            meta = client.get(
                f"https://graph.facebook.com/{META_API_VERSION}/{media_id}",
                headers={"Authorization": f"Bearer {META_ACCESS_TOKEN}"},
            )
            meta.raise_for_status()
            url = meta.json().get("url")
            if not url:
                return None
            blob = client.get(
                url, headers={"Authorization": f"Bearer {META_ACCESS_TOKEN}"})
            blob.raise_for_status()
            return blob.content, meta.json().get("mime_type", "")
    except Exception as e:
        logger.warning("Media download failed for %s: %s", media_id, e)
        return None


def _handle_boarding_pass(phone: str, msg) -> None:
    """Store an airline-issued boarding pass (PDF or screenshot) the user
    forwarded in chat, attached to their most recent PAID booking."""
    from FareBeep.models import BookingSession
    media_id = msg.media_id or ""
    if not media_id:
        return
    got = _download_whatsapp_media(media_id)
    if got is None:
        _main()._say(phone, "I couldn't download that file - please try sending "
                    "it again.", None)
        return
    blob, mime = got
    if len(blob) > _MAX_PASS_BYTES:
        _main()._say(phone, "That file is too large to store. Send the PDF "
                    "boarding pass (or a screenshot) instead.", None)
        return
    db = _main().SessionLocal()
    try:
        user = db.query(User).filter(User.phone == phone).first()
        booking = None
        if user:
            booking = (db.query(BookingSession)
                       .filter(BookingSession.user_id == user.user_id,
                               BookingSession.status == "paid")
                       .order_by(BookingSession.created_at.desc())
                       .first())
        if not booking:
            _main()._say(phone, "I don't have a confirmed ticket to attach this "
                        "to yet. Book a flight first, then forward the "
                        "boarding pass here.", None)
            return
        booking.boarding_pass_blob = blob
        booking.boarding_pass_name = msg.media_filename or "boarding-pass"
        booking.boarding_pass_mime = (msg.media_mime or mime or
                                      "application/pdf")
        db.commit()
        _main()._say(phone,
             f"🎫 Boarding pass saved to your {booking.origin} → "
             f"{booking.destination} booking. View it anytime - "
             f"just say 'my bookings'.",
             user.name if user else None)
        ticket_delivery.email_boarding_pass(booking)
    finally:
        db.close()


def _dispatch_tap(phone: str, tap_id: str, source: str = "meta") -> None:
    """Route one fare-card button tap to its handler.

    Shared by the Meta webhook AND Telegram callback_query: both channels
    emit the SAME ids (pick:N / alert:N / beep:N / set_beep / book), so
    cards.translate_tap stays the single parser and every channel rides
    the same tested gates (pick gate, _tap_alert, _tap_beep, the guided
    TRACK fallback)."""
    tap = cards.translate_tap(tap_id)
    if tap is None:
        logger.warning("%s tap ignored: unknown button id %r", source, tap_id)
        return
    if tap[0] == "alert":
        _main()._tap_alert_by_phone(phone, tap[1])
        return
    if tap[0] == "beep":
        _main()._tap_beep_by_phone(phone, tap[1])
        return
    if tap[0] == "set_beep":
        _main()._send_beep_flow(phone)
        return
    _main()._handle_incoming_message(
        phone, str(tap[1]) if tap[0] == "pick" else "BOOK")


def _telegram_callback(cb_id: str, chat_id: str, data: str) -> None:
    """A fare-card button tap from Telegram: stop the button spinner,
    then run the SAME tap gates the Meta cards use. Own background-task
    session discipline as every _handle_incoming_message turn."""
    if cb_id:
        _main().notifier.answer_callback(cb_id)
    if data and chat_id:
        _dispatch_tap(chat_id, data, source="telegram")


def _handle_flow_completion(phone: str, token: str, data: dict) -> None:
    """A finished WhatsApp Flow (nfm_reply in chat). The /flow endpoint
    already validated and created the watch; this turn is the
    user-facing confirmation AND the crash-safe fallback (endpoint calls
    are not durable; webhook turns are save-before-ack'd and leased).
    Creation is idempotent per (user, route), so both paths firing is
    safe."""
    d = (data or {}).get("beep_data") or (data or {})
    origin = str(d.get("origin") or "").upper()
    destination = str(d.get("destination") or "").upper()
    if not origin or not destination or origin == destination:
        _main()._say(phone, "Something went wrong with the beep form 😅 - try it "
                    "again, or just tell me the route here in chat.")
        return
    db = _main().SessionLocal()
    try:
        user = _main()._get_or_create_user(db, phone)
        from FareBeep.alerts import SubscriptionMonitor
        SubscriptionMonitor(db).subscribe(
            user.user_id, origin, destination,
            target_price=_flow_target_price(d),
            target_date=str(d.get("departure_date") or "") or None)
    finally:
        db.close()
    date_label = d.get("departure_date") or "your dates"
    target = _flow_target_price(d)
    target_label = (f"₦{target:,.0f}" if target
                    else "any fare drop (we alert from a 10% drop)")
    _main()._say(phone,
         "✅ *Your beep is on!* Here's everything you set:\n"
         f"📍 Route: {city_name(origin)} ({origin}) → "
         f"{city_name(destination)} ({destination})\n"
         f"📅 Date: {date_label}\n"
         f"🎯 Watching for: {target_label}\n"
         "We'll message you here the moment the fare meets your target. "
         "Reply *my beeps* anytime to pause, change or drop it.")
