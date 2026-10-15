"""INTERACTIVE - fare-card tap routing (Meta + Telegram), Telegram callback
queries, and WhatsApp Flow completions. Covers interactive.py and checks
that main still exposes the same handlers through its re-exports."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from FareBeep import interactive, main
from FareBeep.models import Base, Subscription, User


@pytest.fixture
def session_factory(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(main, "SessionLocal", factory)
    return factory


@pytest.fixture
def routed(monkeypatch):
    """Record every entry point a tap can reach, resolved through main."""
    calls = []

    def recorder(name):
        return lambda *args, **kwargs: calls.append((name, args))

    for name in ("_tap_alert_by_phone", "_tap_beep_by_phone",
                 "_send_beep_flow", "_handle_incoming_message"):
        monkeypatch.setattr(main, name, recorder(name))
    return calls


# --- tap routing -----------------------------------------------------------
def test_alert_tap_routes_to_tap_alert_by_phone(routed):
    interactive._dispatch_tap("+2348011111111", "alert:2")
    assert routed == [("_tap_alert_by_phone", ("+2348011111111", 2))]


def test_beep_tap_routes_to_tap_beep_by_phone(routed):
    interactive._dispatch_tap("+2348011111111", "beep:424")
    assert routed == [("_tap_beep_by_phone", ("+2348011111111", 424))]


def test_set_beep_tap_opens_the_beep_flow(routed):
    interactive._dispatch_tap("+2348011111111", "set_beep")
    assert routed == [("_send_beep_flow", ("+2348011111111",))]


def test_pick_tap_replays_the_number_as_a_chat_turn(routed):
    interactive._dispatch_tap("+2348011111111", "pick:3")
    assert routed == [("_handle_incoming_message", ("+2348011111111", "3"))]


def test_book_tap_becomes_a_book_turn(routed):
    interactive._dispatch_tap("+2348011111111", "book")
    assert routed == [("_handle_incoming_message", ("+2348011111111", "BOOK"))]


def test_unknown_tap_is_ignored(routed):
    interactive._dispatch_tap("+2348011111111", "not-a-button")
    interactive._dispatch_tap("+2348011111111", "dismiss")
    assert routed == []


def test_telegram_source_uses_the_same_routing(routed):
    interactive._dispatch_tap("555001", "alert:1", source="telegram")
    assert routed == [("_tap_alert_by_phone", ("555001", 1))]


# --- Telegram callback queries ----------------------------------------------
class _FakeNotifier:
    def __init__(self):
        self.answered = []

    def answer_callback(self, cb_id, text=None):
        self.answered.append(cb_id)
        return True


def test_callback_stops_spinner_then_routes_the_tap(monkeypatch, routed):
    fake = _FakeNotifier()
    monkeypatch.setattr(main, "notifier", fake)
    interactive._telegram_callback("CB42", "555001", "pick:2")
    assert fake.answered == ["CB42"]
    assert routed == [("_handle_incoming_message", ("555001", "2"))]


def test_callback_without_id_still_routes_but_does_not_answer(
        monkeypatch, routed):
    fake = _FakeNotifier()
    monkeypatch.setattr(main, "notifier", fake)
    interactive._telegram_callback("", "555001", "book")
    assert fake.answered == []
    assert routed == [("_handle_incoming_message", ("555001", "BOOK"))]


def test_callback_without_data_or_chat_is_a_no_op(monkeypatch, routed):
    fake = _FakeNotifier()
    monkeypatch.setattr(main, "notifier", fake)
    interactive._telegram_callback("CB9", "", "book")
    interactive._telegram_callback("CB10", "555001", "")
    assert fake.answered == ["CB9", "CB10"]
    assert routed == []


# --- WhatsApp Flow completion -----------------------------------------------
@pytest.fixture
def replies(monkeypatch):
    out = []
    monkeypatch.setattr(main, "_say",
                        lambda phone, msg, *a, **k: out.append((phone, msg)))
    return out


def test_flow_completion_creates_the_watch_and_confirms(
        session_factory, replies):
    phone = "+2348022222222"
    interactive._handle_flow_completion(phone, "tok", {"beep_data": {
        "origin": "los", "destination": "abv",
        "departure_date": "2026-12-01", "target_price": "70000"}})

    s = session_factory()
    try:
        user = s.query(User).filter(User.phone == phone).one()
        sub = s.query(Subscription).filter_by(user_id=user.user_id).one()
        assert (sub.origin, sub.destination) == ("LOS", "ABV")
        assert sub.target_price == 70000.0
    finally:
        s.close()

    assert len(replies) == 1
    sent_to, text = replies[0]
    assert sent_to == phone
    assert "Your beep is on" in text
    assert "₦70,000" in text


def test_flow_completion_without_target_watches_for_any_drop(
        session_factory, replies):
    phone = "+2348022222223"
    interactive._handle_flow_completion(phone, "tok", {
        "origin": "LOS", "destination": "ABV"})

    s = session_factory()
    try:
        user = s.query(User).filter(User.phone == phone).one()
        sub = s.query(Subscription).filter_by(user_id=user.user_id).one()
        assert sub.target_price is None
    finally:
        s.close()
    assert "any fare drop" in replies[0][1]


@pytest.mark.parametrize("data", [
    {},
    {"origin": "LOS"},
    {"origin": "LOS", "destination": "LOS"},
])
def test_flow_completion_rejects_incomplete_or_same_route(
        session_factory, replies, data):
    phone = "+2348022222224"
    interactive._handle_flow_completion(phone, "tok", data)

    s = session_factory()
    try:
        assert s.query(Subscription).count() == 0
    finally:
        s.close()
    assert len(replies) == 1
    assert "Something went wrong" in replies[0][1]


def test_flow_completion_is_idempotent_per_route(session_factory, replies):
    # The /flow endpoint and the webhook fallback can both fire for one
    # form; creation is per (user, route) so the second must not duplicate.
    phone = "+2348022222225"
    data = {"origin": "LOS", "destination": "ABV",
            "departure_date": "2026-12-01"}
    interactive._handle_flow_completion(phone, "tok", data)
    interactive._handle_flow_completion(phone, "tok", data)

    s = session_factory()
    try:
        assert s.query(Subscription).count() == 1
    finally:
        s.close()


# --- re-export contract -----------------------------------------------------
def test_main_no_longer_owns_the_moved_handlers():
    for name in ("_handle_boarding_pass", "_dispatch_tap",
                 "_telegram_callback", "_handle_flow_completion",
                 "_download_whatsapp_media"):
        assert not hasattr(main, name), name
        assert hasattr(interactive, name), name
