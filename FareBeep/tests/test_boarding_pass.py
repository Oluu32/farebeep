"""BOARDING PASS - forward-to-bot capture, storage, and secure download."""
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from FareBeep import interactive, main
from FareBeep.models import Base, BookingSession, User
from FareBeep.whatsapp.router import InboundMessage, MessageType


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session


@pytest.fixture
def client(monkeypatch, session_factory):
    monkeypatch.setattr(main, "META_VERIFY_TOKEN", "test-verify-token")
    monkeypatch.setattr(main, "META_APP_SECRET", "test-app-secret")
    monkeypatch.setattr(main, "SessionLocal", session_factory)
    monkeypatch.setattr(main.brain, "GEMINI_API_KEY", None)
    monkeypatch.setattr(main, "GROQ_API_KEY", None)
    return TestClient(main.app)


def _user_with_paid_booking(session_factory, phone="2348144180146"):
    s = session_factory()
    u = User(phone=phone, name="Ada")
    s.add(u)
    s.commit()
    s.refresh(u)
    b = BookingSession(
        user_id=u.user_id, origin="LOS", destination="ABV",
        flight_date="2026-10-01", payment_ref="FB-ABCD1234",
        total_price=98500.0, airline_price=90000.0, processing_fee=500.0,
        status="paid", flight_details='{"airline": "Air Peace"}')
    s.add(b)
    s.commit()
    s.refresh(b)
    bid, uname = b.id, u.phone
    s.close()
    return uname, bid


def _doc_msg(phone="2348144180146", media_id="MEDIA1"):
    return InboundMessage(
        message_id="wamid.doc1", from_number=phone,
        message_type=MessageType.DOCUMENT, timestamp="0", raw={},
        media_id=media_id, media_filename="boarding-pass.pdf",
        media_mime="application/pdf")


def test_router_classifies_document_and_image():
    from FareBeep.whatsapp.router import _parse
    d = _parse({"type": "document",
                "document": {"id": "M1", "filename": "pass.pdf",
                             "mime_type": "application/pdf"}})
    assert d.message_type == MessageType.DOCUMENT
    assert d.media_id == "M1" and d.media_filename == "pass.pdf"
    i = _parse({"type": "image", "image": {"id": "M2",
                                           "mime_type": "image/jpeg"}})
    assert i.message_type == MessageType.DOCUMENT
    assert i.media_id == "M2"


def test_forward_pass_attaches_to_latest_paid_booking(
        monkeypatch, session_factory):
    sent = []
    monkeypatch.setattr(main, "SessionLocal", session_factory)
    monkeypatch.setattr(main, "_say",
                        lambda phone, text, name=None, **kw: sent.append(text))
    monkeypatch.setattr(interactive, "_download_whatsapp_media",
                        lambda mid: (b"%PDF-1.4 fake", "application/pdf"))
    user, bid = _user_with_paid_booking(session_factory)
    interactive._handle_boarding_pass(user, _doc_msg(phone=user))
    assert "Boarding pass saved" in sent[0]
    s = session_factory()
    b = s.get(BookingSession, bid)
    assert b.boarding_pass_blob == b"%PDF-1.4 fake"
    assert b.boarding_pass_name == "boarding-pass.pdf"
    s.close()


def test_forward_without_booking_gets_polite_reply(
        monkeypatch, session_factory):
    sent = []
    monkeypatch.setattr(main, "SessionLocal", session_factory)
    monkeypatch.setattr(main, "_say",
                        lambda phone, text, name=None, **kw: sent.append(text))
    monkeypatch.setattr(interactive, "_download_whatsapp_media",
                        lambda mid: (b"x", "application/pdf"))
    interactive._handle_boarding_pass("234999999001", _doc_msg(phone="234999999001"))
    assert "confirmed ticket" in sent[0].lower()


def test_pass_download_roundtrip(client, session_factory):
    user, bid = _user_with_paid_booking(session_factory)
    s = session_factory()
    b = s.get(BookingSession, bid)
    b.boarding_pass_blob = b"%PDF-1.4 pass"
    b.boarding_pass_name = "pass.pdf"
    b.boarding_pass_mime = "application/pdf"
    s.commit()
    s.close()

    tok = main._ticket_link_token(user)
    r = client.get("/tickets/pass", params={"t": tok, "b": str(bid)})
    assert r.status_code == 200
    assert r.content == b"%PDF-1.4 pass"
    assert "application/pdf" in r.headers["content-type"]
    # the /tickets page links to it
    page = client.get("/tickets", params={"t": tok})
    assert "/tickets/pass?" in page.text
    assert "Boarding pass" in page.text


def test_pass_download_blocks_cross_user(client, session_factory):
    # a valid link for another phone must NOT serve someone else's pass
    _user_with_paid_booking(session_factory, phone="2348144180146")
    stranger = main._ticket_link_token("2348999999999")
    s = session_factory()
    bid = s.query(BookingSession).first().id
    s.close()
    r = client.get("/tickets/pass", params={"t": stranger, "b": str(bid)})
    assert r.status_code in (403, 404)
