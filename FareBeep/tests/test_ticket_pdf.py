"""TICKET PDF - voucher rendering + document push after Paystack settle."""
import pytest

from FareBeep import main, ticket_delivery
from FareBeep.ticket_pdf import render_ticket_pdf


class _FakeSession:
    """Duck-typed BookingSession for the renderer."""
    origin = "LOS"
    destination = "ABV"
    flight_date = "2026-10-01"
    flight_iata = "P47123"
    airline_price = 90000.0
    total_price = 98500.0
    payment_ref = "FB-ABCD1234"


def test_render_produces_valid_pdf():
    data = render_ticket_pdf(_FakeSession(), "FB-1234")
    assert data.startswith(b"%PDF-")
    assert len(data) > 1000


def test_render_with_city_names():
    data = render_ticket_pdf(_FakeSession(), "FB-1234",
                             city_name=lambda c: {"LOS": "Lagos"}.get(c, c))
    assert data.startswith(b"%PDF-")


def test_paystack_paid_sends_document(monkeypatch, db):
    """The full settle branch: paid -> text note + ticket PDF document."""
    from sqlalchemy.orm import sessionmaker
    from FareBeep.models import User, BookingSession
    sent_docs = []
    sent_texts = []

    class FakeNotifier:
        def send_text(self, to, body):
            sent_texts.append((to, body))

        def send_document(self, to, data, filename, caption=None,
                          mime="application/pdf"):
            sent_docs.append((to, data, filename))

    fake = FakeNotifier()
    monkeypatch.setattr(main, "notifier", fake)
    monkeypatch.setattr(main, "SessionLocal",
                        sessionmaker(bind=db.get_bind()))

    u = User(phone="2348144180146", name="Ada")
    db.add(u)
    db.commit()
    db.refresh(u)
    b = BookingSession(
        user_id=u.user_id, origin="LOS", destination="ABV",
        flight_date="2026-10-01", payment_ref="FB-ABCD1234",
        total_price=98500.0, airline_price=90000.0, processing_fee=500.0,
        status="paid")
    db.add(b)
    db.commit()
    db.refresh(b)

    ticket_delivery.send_ticket_pdf(b, "FB-1234", fake)
    assert len(sent_docs) == 1
    to, data, filename = sent_docs[0]
    assert to == "2348144180146"
    assert data.startswith(b"%PDF-")
    assert filename == "FareBeep-FB-1234.pdf"


def test_pdf_failure_never_raises(monkeypatch, db):
    """Render blowing up must be swallowed - the text confirmation already
    went out; the PDF is a bonus."""
    from FareBeep.models import BookingSession
    monkeypatch.setattr(
        "FareBeep.ticket_pdf.render_ticket_pdf",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    b = BookingSession(payment_ref="FB-X", user_id=None)
    ticket_delivery.send_ticket_pdf(b, "FB-9999", None)  # must not raise
