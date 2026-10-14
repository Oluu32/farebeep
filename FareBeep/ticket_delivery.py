"""Ticket delivery - getting a confirmed booking into the user's hands.

Runs after Paystack confirms payment and ticketing.issue_pnr has produced
a PNR. Covers the voucher PDF (chat document + email), the emailed copy of
a forwarded boarding pass, and the chat fallback that collects
ticket-holder details when the /book form was skipped.

Every send is best-effort: a delivery failure is logged and never masks
the paid confirmation. The notifier is passed in explicitly so the caller
picks the channel and tests can substitute a fake.
"""
import logging

from sqlalchemy.orm import object_session

from FareBeep import chatstate, emailer
from FareBeep.database import SessionLocal
from FareBeep.iata import city_name
from FareBeep.models import User, utcnow

logger = logging.getLogger("farebeep.ticket_delivery")


def _user_phone(session) -> str | None:
    """Phone/chat id of the booking's owner. Reads through the session the
    booking is attached to; opens a short-lived one only when detached."""
    db = object_session(session)
    own = db is None
    if own:
        db = SessionLocal()
    try:
        user = db.get(User, session.user_id)
        return user.phone if user else None
    finally:
        if own:
            db.close()


def email_voucher(session, pnr: str, pdf: bytes) -> None:
    """Email the voucher PDF with a compiled booking summary as the body.
    PDFs only (per product decision); best-effort - never raises."""
    try:
        if not emailer.is_configured() or not session.contact_email:
            return
        details = session.flight_details or {}
        body = (
            f"Booking confirmed - here's everything on record.\n\n"
            f"PNR: {pnr}\n"
            f"Passenger: {session.passenger_name or '—'}\n"
            f"Route: {city_name(session.origin)} ({session.origin}) -> "
            f"{city_name(session.destination)} ({session.destination})\n"
            f"Date: {session.flight_date}\n"
            f"Airline: {details.get('airline') or '—'}\n"
            f"Airline price: ₦{session.airline_price:,.0f}\n"
            f"Markup + fees: ₦{(session.markup or 0) + (session.processing_fee or 0):,.0f}\n"
            f"TOTAL PAID: ₦{session.total_price:,.0f}\n"
            f"Paid at: {(session.paid_at or utcnow()).strftime('%Y-%m-%d %H:%M')} UTC\n\n"
            f"The voucher is attached as a PDF. Safe travels!\n"
            f"- FareBeep")
        sent = emailer.send_email(
            session.contact_email, f"Your FareBeep booking - PNR {pnr}",
            body, attachment={"filename": f"FareBeep-{pnr}.pdf",
                              "content_bytes": pdf,
                              "mime": "application/pdf"})
        if not sent:
            logger.info("Voucher email not sent for %s (unconfigured or "
                        "rejected)", session.payment_ref)
    except Exception as e:  # noqa: BLE001
        logger.error("Voucher email failed for %s: %s", session.payment_ref, e)


def send_ticket_pdf(session, pnr: str, notifier) -> None:
    """Render the booking voucher, email it and push it as a document to
    the user's channel. Channels without document support skip the push
    (the email still goes out). Best-effort: never raises."""
    try:
        from FareBeep.ticket_pdf import render_ticket_pdf
        data = render_ticket_pdf(session, pnr, city_name=city_name)
        email_voucher(session, pnr, data)
        if not hasattr(notifier, "send_document"):
            logger.info("Channel has no document support - ticket PDF "
                        "skipped for %s", session.payment_ref)
            return
        phone = _user_phone(session)
        if phone is None:
            return
        notifier.send_document(
            phone, data, filename=f"FareBeep-{pnr}.pdf",
            caption=f"Your FareBeep booking - PNR {pnr}. "
                    f"Present this at check-in.")
    except Exception as e:  # noqa: BLE001
        logger.error("Ticket PDF send failed for %s: %s",
                     session.payment_ref, e)


def email_boarding_pass(booking) -> None:
    """Email a forwarded boarding pass back so the user keeps a copy
    outside the chat. PDFs/screenshot bytes only; best-effort."""
    try:
        if not (emailer.is_configured() and booking.contact_email
                and booking.boarding_pass_blob):
            return
        emailer.send_email(
            booking.contact_email,
            "Your FareBeep boarding pass - "
            f"{booking.origin} -> {booking.destination}",
            "Boarding pass attached (a copy of what you forwarded in "
            "chat). Safe travels!\n- FareBeep",
            attachment={"filename": booking.boarding_pass_name
                        or "boarding-pass",
                        "content_bytes": booking.boarding_pass_blob,
                        "mime": booking.boarding_pass_mime
                        or "application/pdf"})
    except Exception as e:  # noqa: BLE001
        logger.error("Boarding pass email failed for %s: %s",
                     booking.payment_ref, e)


def ask_ticket_details(db, session, pnr: str, notifier) -> None:
    """Paid booking whose ticket-holder details never arrived (the /book
    form was skipped). Park the booking in a two-step chat ask; the voucher
    goes out once both answers land."""
    user = db.get(User, session.user_id)
    if user is None:
        logger.warning("No user for paid session %s - cannot ask for "
                       "ticket details", session.payment_ref)
        return
    chatstate.set_pending_ticket_details(db, user.phone, {
        "booking_id": str(session.id), "stage": "name", "pnr": pnr})
    notifier.send_text(user.phone,
                       "Almost done with your ticket 🎫 - one detail "
                       "missing: what's the traveller's full name as it "
                       "appears on their ID?")


def dispatch_ticket(db, session, pnr: str, notifier) -> None:
    """Post-payment hand-off: send the voucher when ticket-holder details
    are on file, otherwise ask for them in chat first."""
    if session.passenger_name and session.contact_email:
        send_ticket_pdf(session, pnr, notifier)
    else:
        ask_ticket_details(db, session, pnr, notifier)


__all__ = ["email_voucher", "send_ticket_pdf", "email_boarding_pass",
           "ask_ticket_details", "dispatch_ticket"]
