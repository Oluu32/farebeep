"""FLIGHT STATUS - agent tools, Aviationstack lookups, and alert formatting."""
import json
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from FareBeep import agent as agent_mod
from FareBeep.agent import build_tools
from FareBeep.config import META_TEMPLATE_FLIGHT_STATUS
from FareBeep.models import StatusWatch, User
from FareBeep.notifier import (TelegramBot, TwilioWhatsapp,
                               format_flight_status)
from FareBeep.status import (AviationstackClient, StatusLookupError,
                             normalize_flight_iata, parse_departure)

PHONE = "+2348012345678"
UTC = timezone.utc


@pytest.fixture
def user(db):
    u = User(phone=PHONE)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture
def frozen(monkeypatch):
    """Pin the agent's clock and Lagos 'today' well before the fixture dates."""
    now = datetime(2030, 3, 1, tzinfo=UTC)
    monkeypatch.setattr(agent_mod, "utcnow", lambda: now)
    monkeypatch.setattr("FareBeep.dates.lagos_today", lambda: date(2030, 3, 1))
    return now


def _tools(db):
    return {t.name: t for t in build_tools(db, PHONE)}


class FakeAviation:
    """Stands in for AviationstackClient inside the agent tool."""
    detail = None
    error = None
    api_key = "test-key"
    calls = []

    def __init__(self, *a, **k):
        self.api_key = FakeAviation.api_key

    def flight_detail(self, flight_iata, flight_date):
        FakeAviation.calls.append((flight_iata, flight_date))
        if FakeAviation.error:
            raise FakeAviation.error
        return FakeAviation.detail


@pytest.fixture
def fake_aviation(monkeypatch):
    FakeAviation.detail = None
    FakeAviation.error = None
    FakeAviation.api_key = "test-key"
    FakeAviation.calls = []
    monkeypatch.setattr("FareBeep.status.AviationstackClient", FakeAviation)
    return FakeAviation


DELAYED = {
    "flight_iata": "P47123", "airline": "Air Peace", "status": "delayed",
    "dep_scheduled": "2030-03-10T08:30:00+01:00",
    "dep_estimated": "2030-03-10T09:35:00+01:00",
    "dep_delay_min": 65, "dep_terminal": "2", "dep_gate": "B4",
    "arr_scheduled": "2030-03-10T09:25:00+01:00",
    "arr_estimated": "2030-03-10T10:30:00+01:00",
    "arr_delay_min": 65, "arr_terminal": "1",
    "arr_gate": None,
}


# ---------------------------------------------------------------------------
# check_flight_status tool
# ---------------------------------------------------------------------------
def test_check_flight_status_returns_summary_and_fields(db, fake_aviation, frozen):
    fake_aviation.detail = DELAYED
    out = json.loads(_tools(db)["check_flight_status"].invoke(
        {"flight_iata": "p4 7123", "flight_date": "2030-03-10"}))

    assert out["found"] is True
    assert out["flight_date"] == "2030-03-10"
    assert out["status"] == "delayed"
    assert out["dep_delay_min"] == 65
    assert "DELAYED by 1h 05m" in out["summary"]
    assert "Gate B4" in out["summary"]
    assert fake_aviation.calls == [("P47123", "2030-03-10")]


def test_check_flight_status_defaults_to_today(db, fake_aviation, frozen):
    fake_aviation.detail = DELAYED
    _tools(db)["check_flight_status"].invoke({"flight_iata": "P47123"})
    assert fake_aviation.calls == [("P47123", "2030-03-01")]


def test_check_flight_status_rejects_bad_flight_number_without_calling_api(
        db, fake_aviation, frozen):
    out = json.loads(_tools(db)["check_flight_status"].invoke(
        {"flight_iata": "hello there"}))
    assert out["found"] is False
    assert "flight number" in out["error"]
    assert fake_aviation.calls == []


def test_check_flight_status_not_configured_is_graceful(db, fake_aviation, frozen):
    fake_aviation.api_key = None
    out = json.loads(_tools(db)["check_flight_status"].invoke(
        {"flight_iata": "P47123", "flight_date": "2030-03-10"}))
    assert out["found"] is False
    assert "isn't configured" in out["error"]
    assert fake_aviation.calls == []


def test_check_flight_status_api_failure_is_graceful(db, fake_aviation, frozen):
    fake_aviation.error = StatusLookupError("boom")
    out = json.loads(_tools(db)["check_flight_status"].invoke(
        {"flight_iata": "P47123", "flight_date": "2030-03-10"}))
    assert out["found"] is False
    assert "unreachable" in out["error"]


def test_check_flight_status_no_record(db, fake_aviation, frozen):
    fake_aviation.detail = None
    out = json.loads(_tools(db)["check_flight_status"].invoke(
        {"flight_iata": "P47123", "flight_date": "2030-03-10"}))
    assert out["found"] is False
    assert "no status record for P47123 on 2030-03-10" in out["error"]


def test_check_flight_status_bad_date_is_graceful(db, fake_aviation, frozen):
    out = json.loads(_tools(db)["check_flight_status"].invoke(
        {"flight_iata": "P47123", "flight_date": "not-a-date"}))
    assert out["found"] is False
    assert out["error"].startswith("bad date")
    assert fake_aviation.calls == []


# ---------------------------------------------------------------------------
# AviationstackClient.flight_detail (HTTP level)
# ---------------------------------------------------------------------------
def _client(handler):
    return AviationstackClient(
        api_key="k", http_client=httpx.Client(
            transport=httpx.MockTransport(handler)))


def _row(flight_date):
    return {"flight_date": flight_date, "flight_status": "Delayed",
            "flight": {"iata": "P47123"}, "airline": {"name": "Air Peace"},
            "departure": {"scheduled": "2030-03-10T08:30:00+01:00",
                          "delay": 30, "terminal": "2", "gate": "B4"},
            "arrival": {"scheduled": "2030-03-10T09:25:00+01:00"}}


def test_flight_detail_maps_aviationstack_row():
    client = _client(lambda req: httpx.Response(
        200, json={"data": [_row("2030-03-10")]}))
    detail = client.flight_detail("P47123", "2030-03-10")
    assert detail["status"] == "delayed"
    assert detail["airline"] == "Air Peace"
    assert detail["dep_delay_min"] == 30
    assert detail["dep_gate"] == "B4"
    assert detail["arr_scheduled"] == "2030-03-10T09:25:00+01:00"


def test_flight_detail_ignores_record_for_another_day():
    client = _client(lambda req: httpx.Response(
        200, json={"data": [_row("2030-03-09")]}))
    assert client.flight_detail("P47123", "2030-03-10") is None


def test_flight_detail_raises_lookup_error_on_http_failure():
    client = _client(lambda req: httpx.Response(500))
    with pytest.raises(StatusLookupError):
        client.flight_detail("P47123", "2030-03-10")


def test_flight_status_still_returns_none_on_failure():
    client = _client(lambda req: httpx.Response(500))
    assert client.flight_status("P47123", "2030-03-10") is None


# ---------------------------------------------------------------------------
# track_flight_status tool - the 3-hour watch window
# ---------------------------------------------------------------------------
def test_track_creates_watch_three_hours_before_departure(db, user, frozen):
    out = json.loads(_tools(db)["track_flight_status"].invoke({
        "flight_iata": "P47123", "flight_date": "2030-03-10",
        "scheduled_departure": "2030-03-10 08:30"}))

    assert out["watching"] is True
    watch = db.query(StatusWatch).one()
    assert watch.user_id == user.user_id
    assert watch.flight_iata == "P47123"
    assert watch.flight_date == "2030-03-10"
    # 08:30 Lagos is 07:30 UTC; the window opens at 04:30 UTC (05:30 Lagos).
    dep = watch.scheduled_departure.replace(tzinfo=UTC)
    start = watch.watch_starts_at.replace(tzinfo=UTC)
    assert dep == datetime(2030, 3, 10, 7, 30, tzinfo=UTC)
    assert start == dep - timedelta(hours=3)
    assert "I'll start checking at 05:30 on 10 Mar" in out["summary"]


def test_track_bare_time_reads_on_flight_date(db, user, frozen):
    out = json.loads(_tools(db)["track_flight_status"].invoke({
        "flight_iata": "P47123", "flight_date": "2030-03-10",
        "scheduled_departure": "08:30"}))
    assert out["watching"] is True
    watch = db.query(StatusWatch).one()
    assert watch.scheduled_departure.replace(tzinfo=UTC) == \
        datetime(2030, 3, 10, 7, 30, tzinfo=UTC)


def test_track_inside_window_says_watching_now(db, user, frozen, monkeypatch):
    # Clock 1h before departure: the 3-hour window is already open.
    monkeypatch.setattr(agent_mod, "utcnow",
                        lambda: datetime(2030, 3, 10, 6, 30, tzinfo=UTC))
    out = json.loads(_tools(db)["track_flight_status"].invoke({
        "flight_iata": "P47123", "flight_date": "2030-03-10",
        "scheduled_departure": "2030-03-10 08:30"}))
    assert out["watching"] is True
    assert "I'm watching it now" in out["summary"]


def test_track_is_idempotent_for_same_flight(db, user, frozen):
    args = {"flight_iata": "P47123", "flight_date": "2030-03-10",
            "scheduled_departure": "2030-03-10 08:30"}
    tools = _tools(db)
    first = json.loads(tools["track_flight_status"].invoke(args))
    second = json.loads(tools["track_flight_status"].invoke(args))
    assert first["watch_starts_at"] == second["watch_starts_at"]
    assert db.query(StatusWatch).count() == 1


def test_track_rejects_departure_already_passed(db, user, frozen):
    out = json.loads(_tools(db)["track_flight_status"].invoke({
        "flight_iata": "P47123", "flight_date": "2030-03-01",
        "scheduled_departure": "2030-03-01 00:30"}))
    assert out["watching"] is False
    assert "already passed" in out["error"]
    assert db.query(StatusWatch).count() == 0


def test_track_rejects_unparseable_departure(db, user, frozen):
    out = json.loads(_tools(db)["track_flight_status"].invoke({
        "flight_iata": "P47123", "flight_date": "2030-03-10",
        "scheduled_departure": "tomorrow morning"}))
    assert out["watching"] is False
    assert "departure time" in out["error"]
    assert db.query(StatusWatch).count() == 0


def test_track_unknown_user_is_graceful(db, frozen):
    out = json.loads(_tools(db)["track_flight_status"].invoke({
        "flight_iata": "P47123", "flight_date": "2030-03-10",
        "scheduled_departure": "2030-03-10 08:30"}))
    assert out["watching"] is False
    assert "unknown user" in out["error"]


def test_both_status_tools_are_registered(db):
    names = set(_tools(db))
    assert {"check_flight_status", "track_flight_status"} <= names


def test_parse_departure_and_flight_number_helpers():
    assert normalize_flight_iata(" p4 7123 ") == "P47123"
    assert normalize_flight_iata("nope!!") is None
    assert parse_departure("2030-03-10 08:30") == \
        datetime(2030, 3, 10, 7, 30, tzinfo=UTC)
    with pytest.raises(ValueError):
        parse_departure("08:30")  # bare time without a flight date


# ---------------------------------------------------------------------------
# Notifier formatting
# ---------------------------------------------------------------------------
def test_format_delayed_flight_is_human_readable():
    text = format_flight_status(DELAYED)
    lines = text.splitlines()
    assert lines[0] == "✈️ Air Peace P47123 - DELAYED by 1h 05m"
    assert lines[1] == \
        "Departs 08:30 (now 09:35) · Terminal 2 · Gate B4"
    assert lines[2] == "Arrives 09:25 (now 10:30) · Terminal 1"
    assert "Check the departure board" in lines[3]
    assert "{" not in text and "None" not in text


def test_format_cancelled_gives_rebook_guidance():
    text = format_flight_status({"flight_iata": "P47123",
                                 "status": "cancelled"})
    assert text.splitlines()[0] == "✈️ P47123 - CANCELLED"
    assert "rebook" in text


def test_format_on_time_flight_omits_delay():
    text = format_flight_status({"flight_iata": "P47123", "airline": "Ibom Air",
                                 "status": "scheduled",
                                 "dep_scheduled": "2030-03-10T08:30:00+01:00",
                                 "dep_delay_min": 0})
    assert text.splitlines()[0] == "✈️ Ibom Air P47123 - ON TIME"
    assert "(now" not in text and "late" not in text


def test_format_unknown_status_still_renders():
    text = format_flight_status({"flight_iata": "P47123", "status": None})
    assert text.splitlines()[0] == "✈️ P47123 - STATUS UNKNOWN"


def test_telegram_flight_status_template_renders_as_sentence():
    class FakeHTTP:
        def post(self, url, json):
            self.payload = json
            return FakeResp()

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    http = FakeHTTP()
    bot = TelegramBot(token="123:abc", http_client=http)
    ok = bot.send_template("987654321", META_TEMPLATE_FLIGHT_STATUS,
                           body_parameters=["P47123", "DELAYED"])
    assert ok is True
    assert http.payload["text"].startswith("✈️ P47123 - DELAYED")
    assert "farebeep_flight_status:" not in http.payload["text"]


def test_twilio_flight_status_template_renders_as_sentence():
    class FakeMessages:
        def create(self, **kwargs):
            self.kwargs = kwargs

    class FakeClient:
        messages = FakeMessages()

    tw = TwilioWhatsapp(account_sid="AC", auth_token="t",
                        from_whatsapp="whatsapp:+14155238886",
                        client=FakeClient())
    ok = tw.send_template("+2348012345678", META_TEMPLATE_FLIGHT_STATUS,
                          body_parameters=["P47123", "CANCELLED"])
    assert ok is True
    body = FakeClient.messages.kwargs["body"]
    assert body.startswith("✈️ P47123 - CANCELLED")
    assert "rebook" in body
