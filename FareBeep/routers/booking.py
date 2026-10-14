"""Booking, checkout, and ticket pages: /book, /tickets, /payment/status.

Endpoints only. Shared logic stays in FareBeep.main and is looked up at
request time through `_m`, so monkeypatching main's globals keeps working.
main.py mounts this router after all of its helpers are defined.
"""
import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from FareBeep import main as _m
from FareBeep import support
from FareBeep.models import BookingSession, User, utcnow
from FareBeep.templating import render_html

logger = logging.getLogger("farebeep.main")
router = APIRouter()


@router.get("/book/{session_id}")
def booking_page(session_id: uuid.UUID):
    db = _m.SessionLocal()
    try:
        session = db.get(BookingSession, session_id)
        if session is None:
            return _m._booking_not_found()
        if _m._is_expired(session):
            return _m._booking_closed(session.origin, session.destination)
        user = db.get(User, session.user_id)
        return _m._booking_page(session, user_email=user.email if user else None)
    finally:
        db.close()


@router.post("/book/{session_id}/confirm")
async def booking_confirm(session_id: uuid.UUID, request: Request):
    """Capture ticket-holder details + NDPA consent, then Paystack."""
    form = dict(await request.form())
    db = _m.SessionLocal()
    try:
        session = db.get(BookingSession, session_id)
        if session is None:
            return _m._booking_not_found()
        if _m._is_expired(session):
            return _m._booking_closed(session.origin, session.destination)
        user = db.get(User, session.user_id)
        name = str(form.get("passenger_name") or "").strip()[:80]
        email = str(form.get("contact_email") or "").strip().lower()[:120]
        if not name or not _m._EMAIL_RE.match(email):
            return HTMLResponse(status_code=422, content=(
                "<!doctype html><meta charset='utf-8'>"
                "<body style='font-family:system-ui;background:#0b1220;"
                "color:#e8eef7;padding:2rem'>Please enter a valid "
                "passenger name and email address. "
                "<a style='color:#6ee7a0' href='javascript:history.back()'>"
                "Go back</a>.</body>"))
        session.passenger_name = name
        session.contact_email = email
        if user is not None:
            user.consent_at = utcnow()
            user.consent_text_version = _m.CONSENT_VERSION
            if not user.email:
                user.email = email
            logger.info("Consent recorded v%s for user %s (booking %s)",
                        _m.CONSENT_VERSION, session.user_id, session.payment_ref)
        db.commit()
        return RedirectResponse(session.callback_url or "/payment/status",
                                status_code=303)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# "My tickets" page - the read-only browser view behind the signed chat
# link. Mobile-first (it opens in WhatsApp's in-app webview): no cookies,
# no JS, no lateral links - the token is the session.
# ---------------------------------------------------------------------------
# Airline online check-in portals (verified official URLs). Airlines
# expose no check-in API, so FareBeep deep-links the user to the
# carrier's own portal - the honest version of "check in with us".
_CHECKIN_URLS = {
    "air peace": "https://book-airpeace.crane.aero/ibe/checkin/search",
    "ibom air": "https://www.ibomair.com/check-in/",
    "green africa": "https://greenafrica.com/check-in/login",
    "arik air": "https://arikair.crane.aero/ibe/checkin/search",
}


def _checkin_target(booking) -> "tuple[str, str] | None":
    """(portal_url, airline) for a booking's carrier, or None if unknown."""
    import json as _json
    import urllib.parse
    try:
        details = _json.loads(booking.flight_details) \
            if booking.flight_details else {}
    except Exception:
        details = {}
    airline = str(details.get("airline") or "").strip()
    if not airline:
        return None
    url = _CHECKIN_URLS.get(airline.lower())
    if not url:
        url = ("https://www.google.com/search?q="
               + urllib.parse.quote(f"{airline} online check-in"))
    return url, airline


@router.get("/tickets", include_in_schema=False)
def tickets_page(t: str = ""):
    phone = support._ticket_link_phone(t)
    if not phone:
        return render_html("tickets.html", status_code=403, expired=True)

    from FareBeep.transactions import pnr_from_ref
    from FareBeep.models import Subscription
    db = _m.SessionLocal()
    try:
        user = db.query(User).filter(User.phone == phone).first()
        if not user:
            return render_html("tickets.html", phone=phone, no_account=True,
                               bookings=[], beeps=[])
        bookings = (db.query(BookingSession)
                    .filter(BookingSession.user_id == user.user_id)
                    .order_by(BookingSession.created_at.desc())
                    .limit(20).all())
        beeps = (db.query(Subscription)
                 .filter(Subscription.user_id == user.user_id)
                 .all())
        booking_rows = []
        for b in bookings:
            paid = b.status == "paid"
            target = _checkin_target(b) if paid else None
            booking_rows.append({
                "origin": b.origin,
                "destination": b.destination,
                "flight_date": b.flight_date,
                "total": f"{b.total_price:,.0f}",
                "status": b.status,
                "pnr": pnr_from_ref(b.payment_ref) if paid else None,
                "payment_ref": b.payment_ref,
                "checkin_url": target[0] if target else None,
                "airline": target[1] if target else None,
                "pass_url": (f"/tickets/pass?t={t}&b={b.id}"
                             if b.boarding_pass_blob else None),
                "boarding_pass_name": b.boarding_pass_name,
            })
        beep_rows = [{
            "origin": s.origin,
            "destination": s.destination,
            "state": "paused" if s.paused else "watching",
            "target": f"{s.target_price:,.0f}" if s.target_price else None,
            "target_date": s.target_date,
        } for s in beeps]
    finally:
        db.close()

    return render_html("tickets.html", phone=phone, no_account=False,
                       bookings=booking_rows, beeps=beep_rows, expired=False)


# ---------------------------------------------------------------------------
# Boarding-pass download - same signed-link session as /tickets, scoped to
# one booking; the link token's phone must own the booking.
# ---------------------------------------------------------------------------
@router.get("/tickets/pass", include_in_schema=False)
def boarding_pass(t: str = "", b: str = ""):
    phone = support._ticket_link_phone(t)
    if not phone:
        return HTMLResponse("Link expired - ask the bot for a fresh one.",
                            status_code=403)
    import uuid as _uuid
    from FareBeep.models import BookingSession
    db = _m.SessionLocal()
    try:
        user = db.query(User).filter(User.phone == phone).first()
        booking = None
        if user:
            try:
                bid = _uuid.UUID(b)
            except ValueError:
                bid = None
            if bid:
                booking = (db.query(BookingSession)
                           .filter(BookingSession.id == bid,
                                   BookingSession.user_id == user.user_id)
                           .first())
        if not booking or not booking.boarding_pass_blob:
            return HTMLResponse("No boarding pass on file for this booking.",
                                status_code=404)
        from fastapi.responses import Response as FastResponse
        return FastResponse(
            content=booking.boarding_pass_blob,
            media_type=booking.boarding_pass_mime or "application/pdf",
            headers={"Content-Disposition":
                     f'inline; filename="{booking.boarding_pass_name}"'})
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Payment status page - where Paystack sends the user after checkout
# (PAYSTACK_CALLBACK_URL). The actual settlement happens in the webhook;
# this page is just a friendly confirmation screen.
# ---------------------------------------------------------------------------
@router.get("/payment/status")
def payment_status(reference: str = ""):
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FareBeep - Payment</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif;
         background: #0b1220; color: #e8eef7; display: grid; place-items: center;
         min-height: 100vh; margin: 0; }}
  .card {{ background: #131c2e; border: 1px solid #26324a; border-radius: 16px;
          padding: 2.5rem; max-width: 26rem; text-align: center; }}
  .check {{ font-size: 3rem; }}
  h1 {{ font-size: 1.25rem; margin: 0.75rem 0 0.5rem; }}
  p {{ color: #9fb0c9; margin: 0.25rem 0; line-height: 1.5; }}
  .ref {{ color: #64748b; font-size: 0.85rem; margin-top: 1rem; }}
</style>
</head>
<body>
<div class="card">
  <div class="check">✅</div>
  <h1>Payment received</h1>
  <p>Your booking is being confirmed.<br>We&#39;ll send your ticket and PNR to your WhatsApp shortly.</p>
  <p class="ref">Reference: {reference or "—"}</p>
</div>
</body>
</html>"""
    return HTMLResponse(html)
