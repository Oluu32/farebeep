"""ELEVENLABS SERVER TOOLS - /tools/search + /tools/reserve + PNR issuance."""
import hashlib
import hmac
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from FareBeep import main, ticketing, transactions
from FareBeep.models import (Base, BookingSession, FareLedger, User, utcnow)
from FareBeep.transactions import BookingService


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
    monkeypatch.setattr(main, "SessionLocal", session_factory)
    monkeypatch.setattr(main.brain, "GEMINI_API_KEY", None)
    monkeypatch.setattr(main, "GROQ_API_KEY", None)  # deterministic path
    monkeypatch.setattr(main, "ELEVENLABS_TOOL_SECRET", None)
    monkeypatch.setattr(
        transactions, "initialize_paystack_payment",
        lambda ref, total, email: {
            "access_code": f"AC_{ref}",
            "authorization_url": f"https://paystack.com/pay/{ref}"})
    return TestClient(main.app)


@pytest.fixture
def db(session_factory):
    session = session_factory()
    yield session
    session.close()


OFFER = {"flight_no": "P47123", "airline": "P4",
         "airline_name": "Air Peace",
         "departure_code": "LOS", "arrival_code": "ABV",
         "departure_time": "08:30", "arrival_time": "09:25",
         "duration": "0h 55m", "baggage": "20kg", "cabin": "Economy",
         "price": 98000.0, "currency": "NGN", "booking_token": "btk_aaa"}


class FakeTravels247:
    """Scripted async Travels247: offers list + verified price + PNR."""

    def __init__(self, offers=None, verified_price=None, pnr="ABC123"):
        self.offers = offers if offers is not None else [dict(OFFER)]
        self.verified_price = (verified_price if verified_price is not None
                               else (self.offers[0]["price"]
                                     if self.offers else 0.0))
        self.pnr = pnr
        self.closed = False

    async def search_offers(self, *a, **k):
        return [dict(o) for o in self.offers]

    async def verify_price(self, booking_token, **k):
        return {"verified": True, "price_changed": False,
                "original_price": self.verified_price,
                "verified_price": self.verified_price,
                "currency": "NGN", "booking_token": "btk_bbb",
                "expires_at": None}

    async def reserve(self, booking_token, travellers, **k):
        return {"pnr": self.pnr, "booking_reference": self.pnr,
                "carrier": "P4", "status": "confirmed",
                "ticket_deadline": "2026-05-19 14:30:00"}

    async def close(self):
        self.closed = True


def _seed_ledger(db):
    db.add(FareLedger(origin="LOS", destination="ABV",
                      flight_date="2026-08-20", price=98000.0,
                      currency="NGN", airline="Air Peace",
                      verify_link=None, last_updated=utcnow()))
    db.commit()


def test_tools_search_serves_ledger_hit(client, db, monkeypatch):
    _seed_ledger(db)
    fake = FakeTravels247(offers=[])
    monkeypatch.setattr(main, "get_inventory_client", lambda *a, **k: fake)

    r = client.post("/tools/search",
                    json={"origin": "Lagos", "destination": "Abuja",
                          "flight_date": "2026-08-20"})

    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert body["source"] == "ledger"
    assert body["booking_token"] is None
    assert "Air Peace" in body["summary"]
    assert "98,000" in body["summary"]


def test_tools_search_get_method_serves_ledger_hit(client, db,
                                                      monkeypatch):
    _seed_ledger(db)
    monkeypatch.setattr(main, "get_inventory_client",
                        lambda *a, **k: FakeTravels247(offers=[]))

    r = client.get("/tools/search",
                   params={"origin": "LOS", "destination": "ABV",
                           "flight_date": "2026-08-20"})

    assert r.status_code == 200
    assert r.json()["source"] == "ledger"


def test_tools_search_miss_queries_247travels_and_upserts(
        client, db, monkeypatch):
    fake = FakeTravels247()
    monkeypatch.setattr(main, "get_inventory_client", lambda *a, **k: fake)

    r = client.post("/tools/search",
                    json={"origin": "LOS", "destination": "ABV",
                          "flight_date": "2026-08-21"})

    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert body["source"] == "247travels"
    assert body["booking_token"] == "btk_aaa"
    assert "P47123" in body["summary"]
    assert fake.closed is True
    row = db.query(FareLedger).one()
    assert row.price == 98000.0
    assert (row.origin, row.destination) == ("LOS", "ABV")


def test_tools_search_no_live_offers(client, monkeypatch):
    monkeypatch.setattr(main, "get_inventory_client",
                        lambda *a, **k: FakeTravels247(offers=[]))

    r = client.post("/tools/search",
                    json={"origin": "LOS", "destination": "ABV",
                          "flight_date": "2026-08-22"})

    assert r.status_code == 200
    assert r.json()["found"] is False


def test_tools_search_rejects_bad_secret(client, monkeypatch):
    monkeypatch.setattr(main, "ELEVENLABS_TOOL_SECRET", "s3cr3t")

    r = client.post("/tools/search",
                    json={"origin": "LOS", "destination": "ABV",
                          "flight_date": "2026-08-20"})
    assert r.status_code == 403

    r = client.post("/tools/search", headers={"X-FareBeep-Tool-Secret": "nope"},
                    json={"origin": "LOS", "destination": "ABV",
                          "flight_date": "2026-08-20"})
    assert r.status_code == 403


def test_tools_search_accepts_good_secret(client, db, monkeypatch):
    monkeypatch.setattr(main, "ELEVENLABS_TOOL_SECRET", "s3cr3t")
    monkeypatch.setattr(main, "get_inventory_client",
                        lambda *a, **k: FakeTravels247(offers=[]))
    _seed_ledger(db)

    r = client.post("/tools/search", headers={"X-FareBeep-Tool-Secret": "s3cr3t"},
                    json={"origin": "LOS", "destination": "ABV",
                          "flight_date": "2026-08-20",
                          "phone": "+2348012345678"})
    assert r.status_code == 200
    assert r.json()["source"] == "ledger"


def test_tools_reserve_rejects_bad_secret(client, monkeypatch):
    monkeypatch.setattr(main, "ELEVENLABS_TOOL_SECRET", "s3cr3t")

    r = client.post("/tools/reserve",
                    json={"phone": "+2348012345678",
                          "booking_token": "btk_aaa",
                          "origin": "LOS", "destination": "ABV",
                          "flight_date": "2026-08-21"})
    assert r.status_code == 403
    assert r.json()["locked"] is False


def test_tools_search_rejects_bad_input(client):
    r = client.post("/tools/search",
                    json={"origin": "Xanadu", "destination": "Atlantis",
                          "flight_date": "2026-08-20"})
    assert r.status_code == 422
    assert r.json()["found"] is False


def test_tools_reserve_with_token_locks_and_links(client, db, monkeypatch):
    fake = FakeTravels247()
    monkeypatch.setattr(main, "get_inventory_client", lambda *a, **k: fake)

    r = client.post("/tools/reserve",
                    json={"phone": "+2348012345678",
                          "booking_token": "btk_aaa",
                          "origin": "LOS", "destination": "ABV",
                          "flight_date": "2026-08-21",
                          "airline": "Air Peace",
                          "passengers": {"adults": 1},
                          "travellers": {"primary_guest": {"first_name": "Ada"}}})

    assert r.status_code == 200
    body = r.json()
    assert body["locked"] is True
    assert body["payment_link"].startswith("https://paystack.com/pay/FB-")
    # total = (98000 + 5000 + 100) / 0.985
    assert body["total_amount"] == pytest.approx(104670.05, abs=0.01)
    assert "Pay within 10 minutes" in body["summary"]
    session = db.query(BookingSession).one()
    assert session.status == "pending"
    assert session.flight_details["booking_token"] == "btk_bbb"
    assert session.flight_details["travellers"] == {
        "primary_guest": {"first_name": "Ada"}}
    assert (session.expires_at - session.created_at
            <= timedelta(minutes=13))   # unknown method -> bank buffer
    assert (session.expires_at - session.created_at
            > timedelta(minutes=12))


def test_tools_reserve_accepts_string_passengers_and_travellers(
        client, db, monkeypatch):
    """ElevenLabs declares objects as type string - JSON text must parse,
    not 500 (and string travellers must not poison the PNR step)."""
    fake = FakeTravels247()
    monkeypatch.setattr(main, "get_inventory_client", lambda *a, **k: fake)

    r = client.post("/tools/reserve",
                    json={"phone": "+2348012345678",
                          "booking_token": "btk_aaa",
                          "origin": "LOS", "destination": "ABV",
                          "flight_date": "2026-08-21",
                          "airline": "Air Peace",
                          "passengers": '{"adults": 2}',
                          "travellers": ('{"primary_guest": '
                                         '{"first_name": "Ada"}}')})

    assert r.status_code == 200
    assert r.json()["locked"] is True
    session = db.query(BookingSession).one()
    assert session.flight_details["passengers"] == {
        "adults": 2, "children": 0, "infants": 0}
    assert session.flight_details["travellers"] == {
        "primary_guest": {"first_name": "Ada"}}


def test_tools_reserve_without_token_searches_first(client, monkeypatch):
    fake = FakeTravels247()
    monkeypatch.setattr(main, "get_inventory_client", lambda *a, **k: fake)

    r = client.post("/tools/reserve",
                    json={"phone": "+2348012345678",
                          "origin": "LOS", "destination": "ABV",
                          "flight_date": "2026-08-21"})

    assert r.status_code == 200
    assert r.json()["locked"] is True


def test_tools_reserve_247travels_down_returns_502(client, monkeypatch):
    from FareBeep.travels247 import Travels247Error

    class _Down(FakeTravels247):
        async def verify_price(self, *a, **k):
            raise Travels247Error("supplier 502")

    monkeypatch.setattr(main, "get_inventory_client", lambda *a, **k: _Down())

    r = client.post("/tools/reserve",
                    json={"phone": "+2348012345678",
                          "booking_token": "btk_aaa",
                          "origin": "LOS", "destination": "ABV",
                          "flight_date": "2026-08-21"})

    assert r.status_code == 502
    assert r.json()["locked"] is False


def _paystack_post(client, ref, monkeypatch):
    monkeypatch.setattr("FareBeep.payments.PAYSTACK_SECRET_KEY",
                        "test-paystack-key")
    body = ('{"event":"charge.success","data":'
            f'{{"reference":"{ref}","status":"success"}}}}').encode()
    sig = hmac.new(b"test-paystack-key", body, hashlib.sha512).hexdigest()
    return client.post("/webhook/paystack", content=body,
                       headers={"x-paystack-signature": sig})


def _paid_session(db):
    """A booking_session as /tools/reserve leaves it (token + travellers)."""
    user = User(phone="+2348012345678")
    db.add(user)
    db.commit()
    db.refresh(user)
    svc = BookingService(db)
    created = svc.create_booking(
        user.user_id, "LOS", "ABV", "2026-08-21", 98000.0,
        airline="Air Peace", source="247travels")
    session = created["session"]
    details = dict(session.flight_details or {})
    details.update({
        "booking_token": "btk_bbb",
        "passengers": {"adults": 1, "children": 0, "infants": 0},
        "travellers": {"primary_guest": {"first_name": "Ada"}}})
    session.flight_details = details
    db.commit()
    return session.payment_ref


def test_paystack_webhook_issues_real_pnr(client, db,
                                          monkeypatch):
    """Paid + Travels247 creds + stored token/travellers -> real PNR in the beep."""
    sky = FakeTravels247(pnr="ABC123")
    sky.email, sky.password = "p@x.com", "pw"
    monkeypatch.setattr(ticketing, "get_inventory_client",
                        lambda *a, **k: sky)
    sent = []
    monkeypatch.setattr(main.notifier, "send_text",
                        lambda to, body: sent.append((to, body)) or True)
    ref = _paid_session(db)

    r = _paystack_post(client, ref, monkeypatch)

    assert r.status_code == 200
    assert r.json()["outcome"] == "paid"
    assert sent and "ABC123" in sent[0][1]
    assert "(provisional)" not in sent[0][1]
    assert db.query(BookingSession).one().status == "paid"


def test_paystack_webhook_keeps_provisional_pnr_without_247travels(
        client, db, monkeypatch):
    """No Travels247 creds configured -> the mock-PNR path is preserved."""
    monkeypatch.setattr(ticketing, "get_inventory_client",
                        lambda *a, **k: FakeTravels247())
    sent = []
    monkeypatch.setattr(main.notifier, "send_text",
                        lambda to, body: sent.append((to, body)) or True)
    ref = _paid_session(db)

    r = _paystack_post(client, ref, monkeypatch)

    assert r.status_code == 200
    assert r.json()["outcome"] == "paid"
    assert sent and "(provisional)" in sent[0][1]

