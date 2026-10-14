"""MY TICKETS - signed-link ticket page + deterministic chat trigger."""
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from FareBeep import main, support
from FareBeep.models import Base, BookingSession, User


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


def _user(session_factory, phone="2348144180146"):
    s = session_factory()
    u = User(phone=phone, name="Ada")
    s.add(u)
    s.commit()
    s.refresh(u)
    s.close()
    return u


def _booking(session_factory, user, status="paid", ref="FB-ABCD1234",
             flight_details='{"airline": "Air Peace"}'):
    s = session_factory()
    b = BookingSession(
        user_id=user.user_id, origin="LOS", destination="ABV",
        flight_date="2026-10-01", payment_ref=ref, total_price=98500.0,
        airline_price=90000.0, processing_fee=500.0, status=status,
        flight_details=flight_details)
    s.add(b)
    s.commit()
    s.close()


def test_token_roundtrip():
    tok = main._ticket_link_token("2348144180146")
    assert support._ticket_link_phone(tok) == "2348144180146"


def test_token_rejects_tamper_and_expiry():
    tok = main._ticket_link_token("2348144180146")
    assert support._ticket_link_phone(tok[:-2] + "xx") is None
    # expired: craft a token with an old expiry using the same secret
    import base64
    import hashlib
    import hmac
    payload = "2348144180146.%d" % (int(time.time()) - 10)
    sig = hmac.new(b"test-app-secret", payload.encode(),
                   hashlib.sha256).hexdigest()[:32]
    stale = base64.urlsafe_b64encode(
        f"{payload}.{sig}".encode()).decode().rstrip("=")
    assert support._ticket_link_phone(stale) is None


def test_page_renders_paid_ticket(client, session_factory):
    u = _user(session_factory)
    _booking(session_factory, u, status="paid")
    tok = main._ticket_link_token(u.phone)
    r = client.get("/tickets", params={"t": tok})
    assert r.status_code == 200
    assert "LOS → ABV" in r.text
    assert "FB-1234" in r.text                 # PNR = last 4 of ref
    assert "Ticket confirmed" in r.text
    assert "Check in online · Air Peace" in r.text
    assert "book-airpeace.crane.aero" in r.text


def test_page_checkin_falls_back_for_unknown_airline(client, session_factory):
    u = _user(session_factory)
    _booking(session_factory, u, status="paid",
             flight_details='{"airline": "Mystery Air"}')
    tok = main._ticket_link_token(u.phone)
    r = client.get("/tickets", params={"t": tok})
    assert "Mystery%20Air%20online%20check-in" in r.text


def test_page_rejects_bad_token(client):
    r = client.get("/tickets", params={"t": "garbage"})
    assert r.status_code == 403
    assert "expired" in r.text.lower()


def test_chat_trigger_sends_signed_link(client, monkeypatch, session_factory):
    sent = []
    monkeypatch.setattr(main, "_say",
                        lambda phone, text, name=None, **kw: sent.append(text))
    s = session_factory()
    u = User(phone="2348144180146", name="Ada")
    s.add(u)
    s.commit()
    s.refresh(u)
    s.close()
    db = session_factory()
    user = db.query(User).filter(User.phone == u.phone).first()
    assert main._handle_my_bookings(db, user, "My bookings?") is True
    assert main._handle_my_bookings(db, user, "fare LOS ABV") is False
    db.close()
    assert len(sent) == 1
    assert "/tickets?t=" in sent[0]
    assert "15 minutes" in sent[0]
