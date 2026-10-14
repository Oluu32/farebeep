"""SUPPLIER TICKETING - provider-agnostic PNR issuance and fallbacks."""
import asyncio
from types import SimpleNamespace

import pytest

from FareBeep import main, ticketing
from FareBeep.quickair import QuickAirClient, QuickAirError
from FareBeep.travels247 import Travels247Client, Travels247Error

TRAVELLERS = {"primary_guest": {"first_name": "Ada", "last_name": "Obi"}}


def _session(**details):
    base = {"booking_token": "btk_aaa", "travellers": TRAVELLERS,
            "passengers": {"adults": 2, "children": 0, "infants": 0}}
    base.update(details)
    return SimpleNamespace(payment_ref="FB-TEST1234", flight_details=base)


def _issue(session, fallback="FB-1234"):
    return asyncio.run(ticketing.issue_pnr(session, fallback))


@pytest.fixture
def calls(monkeypatch):
    """Record supplier calls by patching the real client classes in place."""
    log = {"verify": [], "reserve": [], "close": []}

    def install(cls, pnr, deadline=None, error=None):
        async def verify_price(self, booking_token, **kw):
            log["verify"].append((type(self).__name__, booking_token, kw))
            if error:
                raise error
            return {"booking_token": "btk_verified", "verified_price": 98000.0}

        async def reserve(self, booking_token, travellers, **kw):
            log["reserve"].append((type(self).__name__, booking_token,
                                   travellers, kw))
            if error:
                raise error
            return {"pnr": pnr, "booking_reference": pnr,
                    "ticket_deadline": deadline}

        async def close(self):
            log["close"].append(type(self).__name__)

        monkeypatch.setattr(cls, "verify_price", verify_price)
        monkeypatch.setattr(cls, "reserve", reserve)
        monkeypatch.setattr(cls, "close", close)

    log["install"] = install
    return log


@pytest.fixture
def travels247_configured(monkeypatch):
    monkeypatch.setattr("FareBeep.config.INVENTORY_PROVIDER", "travels247")
    monkeypatch.setattr("FareBeep.travels247.TRAVELS247_EMAIL", "p@x.com")
    monkeypatch.setattr("FareBeep.travels247.TRAVELS247_PASSWORD", "pw")


@pytest.fixture
def quickair_configured(monkeypatch):
    monkeypatch.setattr("FareBeep.config.INVENTORY_PROVIDER", "quickair")
    monkeypatch.setattr("FareBeep.quickair.QUICKAIR_EMAIL", "q@x.com")
    monkeypatch.setattr("FareBeep.quickair.QUICKAIR_PASSWORD", "qpw")


# ---------------------------------------------------------------------------
# Travels247 path
# ---------------------------------------------------------------------------
def test_travels247_issues_confirmed_pnr(travels247_configured, calls):
    calls["install"](Travels247Client, "TRV777",
                     deadline="2026-10-15 12:00:00")

    result = _issue(_session())

    assert result.pnr == "TRV777"
    assert result.provisional == ""
    assert result.note.startswith("E-ticket issued.")
    assert "2026-10-15 12:00:00" in result.note
    # verify_price -> reserve, with the passenger counts forwarded
    assert calls["verify"] == [("Travels247Client", "btk_aaa",
                                {"adults": 2, "children": 0, "infants": 0})]
    name, token, travellers, kw = calls["reserve"][0]
    assert (name, token) == ("Travels247Client", "btk_verified")
    assert travellers == TRAVELLERS
    assert kw == {"adults": 2, "children": 0, "infants": 0}
    assert calls["close"] == ["Travels247Client"]


def test_travels247_error_keeps_provisional_pnr(travels247_configured, calls):
    calls["install"](Travels247Client, "unused",
                     error=Travels247Error("supplier 502"))

    result = _issue(_session())

    assert result.pnr == "FB-1234"
    assert result.provisional == " (provisional)"
    assert result.note == ticketing.PENDING_NOTE
    assert calls["close"] == ["Travels247Client"]


# ---------------------------------------------------------------------------
# QuickAir path
# ---------------------------------------------------------------------------
def test_quickair_issues_confirmed_pnr(quickair_configured, calls, monkeypatch):
    calls["install"](QuickAirClient, "QA999")
    # Selecting QuickAir must never touch Travels247.
    monkeypatch.setattr(Travels247Client, "reserve",
                        lambda *a, **k: pytest.fail("Travels247 used"))

    result = _issue(_session())

    assert result.pnr == "QA999"
    assert result.provisional == ""
    assert result.note.startswith("E-ticket issued.")
    assert calls["reserve"][0][0] == "QuickAirClient"
    assert calls["close"] == ["QuickAirClient"]


def test_quickair_error_keeps_provisional_pnr(quickair_configured, calls):
    calls["install"](QuickAirClient, "unused",
                     error=QuickAirError("reserve rejected"))

    result = _issue(_session())

    assert result.pnr == "FB-1234"
    assert result.provisional == " (provisional)"
    assert result.note == ticketing.PENDING_NOTE


# ---------------------------------------------------------------------------
# Fallback paths (mock PNR + provisional notice)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("provider,cls", [("travels247", Travels247Client),
                                          ("quickair", QuickAirClient)])
def test_missing_credentials_fall_back_without_network(
        provider, cls, monkeypatch, calls):
    monkeypatch.setattr("FareBeep.config.INVENTORY_PROVIDER", provider)
    monkeypatch.setattr("FareBeep.travels247.TRAVELS247_EMAIL", None)
    monkeypatch.setattr("FareBeep.travels247.TRAVELS247_PASSWORD", None)
    monkeypatch.setattr("FareBeep.quickair.QUICKAIR_EMAIL", None)
    monkeypatch.setattr("FareBeep.quickair.QUICKAIR_PASSWORD", None)
    calls["install"](cls, "should-not-happen")

    result = _issue(_session())

    assert result == ticketing.TicketResult(
        "FB-1234", ticketing.PENDING_NOTE, " (provisional)")
    assert calls["verify"] == [] and calls["reserve"] == []
    assert calls["close"] == [cls.__name__]


def test_missing_token_falls_back_even_when_configured(
        travels247_configured, calls):
    calls["install"](Travels247Client, "should-not-happen")

    result = _issue(_session(booking_token=None))

    assert result.pnr == "FB-1234"
    assert result.provisional == " (provisional)"
    assert calls["reserve"] == []


def test_missing_travellers_falls_back_even_when_configured(
        quickair_configured, calls):
    calls["install"](QuickAirClient, "should-not-happen")

    result = _issue(_session(travellers=None))

    assert result.pnr == "FB-1234"
    assert calls["reserve"] == []


def test_supplier_configured_reads_credentials():
    assert ticketing.supplier_configured(
        SimpleNamespace(email="a", password="b"))
    assert not ticketing.supplier_configured(
        SimpleNamespace(email="a", password=None))
    assert not ticketing.supplier_configured(object())


def test_legacy_main_helper_delegates_to_ticketing(monkeypatch):
    """main._maybe_issue_247travels_ticket is now a shim over ticketing."""
    monkeypatch.setattr("FareBeep.config.INVENTORY_PROVIDER", "travels247")
    monkeypatch.setattr("FareBeep.travels247.TRAVELS247_EMAIL", None)
    monkeypatch.setattr("FareBeep.travels247.TRAVELS247_PASSWORD", None)

    out = asyncio.run(main._maybe_issue_247travels_ticket(_session(), "FB-9"))

    assert out == ("FB-9", ticketing.PENDING_NOTE, " (provisional)")
