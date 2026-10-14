"""BEEP MANAGEMENT - list/pause/resume/cancel/edit, edit-price parsing, and
the stop/unsubscribe/erase flows. Covers subscriptions.py and checks that
main still routes the same messages through it."""
import pytest

from FareBeep import main, subscriptions
from FareBeep.alerts import SubscriptionMonitor
from FareBeep.models import Subscription, User

PHONE = "+2348012345678"


@pytest.fixture
def user(db):
    u = User(phone=PHONE, name="Ada")
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture
def sent(monkeypatch):
    out = []
    monkeypatch.setattr(main, "_say",
                        lambda phone, msg, *a, **k: out.append(msg))
    return out


def _watch(db, user, dest="ABV", target=None):
    SubscriptionMonitor(db).subscribe(user.user_id, "LOS", dest,
                                      target_price=target,
                                      target_date="2026-12-01")
    db.commit()


def _subs(db, user):
    return db.query(Subscription).filter_by(user_id=user.user_id).all()


# --- edit-price parsing -----------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("70000", 70000.0),
    ("70k", 70000.0),
    ("70,000", 70000.0),
    ("1 70k", 70000.0),          # last number wins
    ("60k 75k", 75000.0),
    ("DROP", "drop"),
    ("drop 70k", "drop"),        # the word DROP beats a number
    ("", None),
    ("cheap", None),
])
def test_parse_edit_price(raw, expected):
    assert subscriptions._parse_edit_price(raw) == expected


# --- alert management -------------------------------------------------------
def test_pause_and_resume_single_beep_without_number(db, user, sent):
    _watch(db, user)

    assert subscriptions._handle_manage(db, user, "PAUSE") is True
    assert _subs(db, user)[0].paused is True
    assert any("Paused" in m for m in sent)

    assert subscriptions._handle_manage(db, user, "RESUME") is True
    assert _subs(db, user)[0].paused is False
    assert any("Resumed" in m for m in sent)


def test_bare_pause_with_several_beeps_lists_instead_of_guessing(
        db, user, sent):
    _watch(db, user, dest="ABV")
    _watch(db, user, dest="PHC")

    assert subscriptions._handle_manage(db, user, "PAUSE") is True
    assert all(s.paused is False for s in _subs(db, user))
    assert any("Your beeps (2)" in m for m in sent)


def test_cancel_by_number_deletes_only_that_beep(db, user, sent):
    _watch(db, user, dest="ABV")
    _watch(db, user, dest="PHC")

    assert subscriptions._handle_manage(db, user, "CANCEL 2") is True
    assert len(_subs(db, user)) == 1
    assert any("Cancelled" in m for m in sent)


def test_edit_sets_new_target_price(db, user, sent):
    _watch(db, user)

    subscriptions._handle_manage(db, user, "EDIT 1 70k")

    assert _subs(db, user)[0].target_price == 70000.0
    assert any("₦70,000 or lower" in m for m in sent)


def test_edit_drop_clears_target_price(db, user, sent):
    _watch(db, user, target=80000.0)

    subscriptions._handle_manage(db, user, "EDIT 1 DROP")

    assert _subs(db, user)[0].target_price is None
    assert any("any genuine drop" in m for m in sent)


def test_edit_without_price_asks_for_one_and_changes_nothing(db, user, sent):
    _watch(db, user, target=80000.0)

    subscriptions._handle_manage(db, user, "EDIT 1")

    assert _subs(db, user)[0].target_price == 80000.0
    assert any("reply like: EDIT" in m for m in sent)


def test_manage_verb_with_no_beeps_says_so(db, user, sent):
    assert subscriptions._handle_manage(db, user, "PAUSE") is True
    assert any("no price beeps to change" in m for m in sent)


def test_plain_chat_is_not_a_manage_command(db, user, sent):
    assert subscriptions._handle_manage(db, user, "hello there") is False
    assert sent == []


# --- stop / unsubscribe / erase --------------------------------------------
@pytest.mark.parametrize("text", [
    "stop",
    "STOP alerts",
    "please stop messaging me",
    "unsubscribe",
    "delete my data",
    "cancel my alerts",
])
def test_explicit_stop_requests_are_detected(text):
    assert subscriptions._is_stop_request(text) is True


@pytest.mark.parametrize("text", [
    "stopover in Abuja",
    "nonstop flight to Abuja",
    "don't cancel my alerts",
    "don't stop the beep",
    "lagos to abuja tomorrow",
])
def test_non_stop_text_is_not_a_stop(text):
    assert subscriptions._is_stop_request(text) is False


def test_handle_stop_removes_every_beep_and_confirms(db, user, sent):
    _watch(db, user, dest="ABV")
    _watch(db, user, dest="PHC")

    subscriptions._handle_stop(db, user)

    assert _subs(db, user) == []
    assert any("all 2 price alert(s) removed" in m for m in sent)


def test_handle_stop_with_no_beeps_still_confirms(db, user, sent):
    subscriptions._handle_stop(db, user)
    assert any("nothing to stop" in m for m in sent)


def test_reply_unsubscribe_removes_all_beeps(db, user, sent):
    _watch(db, user, dest="ABV")
    _watch(db, user, dest="PHC")

    subscriptions._reply_unsubscribe(db, user)

    assert _subs(db, user) == []
    assert any("2 price alert(s) removed" in m for m in sent)


def test_stop_message_routes_through_main_and_clears_beeps(
        db, user, sent, monkeypatch):
    _watch(db, user)
    monkeypatch.setattr(main, "SessionLocal", lambda: db)

    main._handle_incoming_message(PHONE, "stop")

    assert _subs(db, user) == []
    assert any("Stopped" in m for m in sent)


def test_main_reexports_the_subscriptions_handlers():
    assert main._handle_stop is subscriptions._handle_stop
    assert main._is_stop_request is subscriptions._is_stop_request
    assert main._tap_beep_by_phone is subscriptions._tap_beep_by_phone
