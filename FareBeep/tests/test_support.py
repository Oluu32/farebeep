"""SUPPORT RELAY - human escalation without a helpdesk.

Answers, with tests:
  Q1 an explicit ask (SUPPORT / "speak to a human") opens a thread,
     acks the user, and pings the admin with context + reply syntax.
  Q2 money-trouble phrases auto-escalate ("charged twice", "refund");
     negated phrasing ("don't refund me") never triggers.
  Q3 while a thread is open the user's words relay to the admin (the
     agent never talks over a human); MY BOOKINGS still works.
  Q4 the founder's "R <phone> <text>" reaches the user verbatim and
     lands in the transcript; unknown targets are refused.
  Q5 /done closes (user told, admin told); user close-words too; the
     admin's reply to a closed thread is delivered but flagged.
  Q6 an open thread bypasses the rate limit; STOP still clears it.
  Q7 two consecutive agent failures auto-open a thread.
  Q8 threads older than the TTL auto-close on touch.
  Q9 /admin/ops shows open threads.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from FareBeep import chatstate, main, support
from FareBeep.models import Base, User

ADMIN = "+2340000000001"
USER = "+2348012345678"


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, session_factory):
    monkeypatch.setattr(main, "SessionLocal", session_factory)
    monkeypatch.setattr("FareBeep.config.ADMIN_ALERT_PHONE", ADMIN)
    monkeypatch.setattr(main, "GROQ_API_KEY", None)
    monkeypatch.setattr(main.brain, "GEMINI_API_KEY", None)
    main._rate_hits.clear()
    main._rate_cooled.clear()
    support._support_frustration.clear()
    yield
    main._rate_hits.clear()
    main._rate_cooled.clear()
    support._support_frustration.clear()


@pytest.fixture
def sent(monkeypatch):
    out = []
    monkeypatch.setattr(main.notifier, "send_text",
                        lambda to, body: out.append((to, body)))
    return out


@pytest.fixture
def user(session_factory):
    s = session_factory()
    u = User(phone=USER, name="Ada")
    s.add(u)
    s.commit()
    s.close()
    return u


def _ticket(factory, phone=USER):
    s = factory()
    try:
        return chatstate.get_support_ticket(s, phone)
    finally:
        s.close()


def _to_admin(sent):
    return [b for to, b in sent if to == ADMIN]


def _to_user(sent):
    return [b for to, b in sent if to == USER]


# --- Q1: explicit ask opens a thread ---------------------------------------
def test_explicit_ask_opens_thread_and_alerts_admin(session_factory, user,
                                                    sent):
    main._handle_incoming_message(USER, "speak to a human")

    tk = _ticket(session_factory)
    assert tk and tk["status"] == "open"
    assert tk["messages"][0]["text"] == "speak to a human"

    ack = "\n".join(_to_user(sent))
    assert "founder" in ack.lower() and "CANCEL SUPPORT" in ack

    alert = "\n".join(_to_admin(sent))
    assert "SUPPORT SUP-" in alert
    assert USER in alert and "R " + USER in alert, \
        "alert must carry the exact relay syntax"


def test_bare_support_keyword_opens_thread(session_factory, user, sent):
    main._handle_incoming_message(USER, "SUPPORT")
    assert _ticket(session_factory)


# --- Q2: money trouble auto-escalates, negation never does -----------------
def test_payment_trouble_auto_escalates(session_factory, user, sent):
    main._handle_incoming_message(
        USER, "I was charged twice and want a refund")

    tk = _ticket(session_factory)
    assert tk and tk["trigger"] == "auto: payment/refund"
    assert "charged twice" in "\n".join(_to_admin(sent))


def test_negated_refund_never_triggers(session_factory, user, sent):
    main._handle_incoming_message(USER, "please don't refund me, all good")

    assert _ticket(session_factory) is None
    assert not any("SUPPORT SUP-" in b for b in _to_admin(sent))


# --- Q3: open thread routes words to the human, commands still work --------
def test_open_thread_relays_and_blocks_agent(session_factory, user, sent):
    main._handle_incoming_message(USER, "SUPPORT")
    sent.clear()

    main._handle_incoming_message(USER, "any news?")
    assert not _to_user(sent), "bot must stay silent on a live thread"
    admin = "\n".join(_to_admin(sent))
    assert "says:" in admin and "any news" in admin
    assert _ticket(session_factory)["messages"][-1]["side"] == "user"

    # benign commands still work while the thread is open
    sent.clear()
    main._handle_incoming_message(USER, "my bookings")
    assert any("/tickets?t=" in b for b in _to_user(sent))


# --- Q4: the relay ----------------------------------------------------------
def test_relay_reaches_user_verbatim(session_factory, user, sent):
    main._handle_incoming_message(USER, "SUPPORT")
    sent.clear()

    main._handle_incoming_message(ADMIN, f"R {USER} hello Ada, checking now")

    out = "\n".join(_to_user(sent))
    assert "\U0001f3a7 FareBeep support: hello Ada, checking now" in out
    assert "checking now" in out and "compose" not in out
    tk = _ticket(session_factory)
    assert tk["messages"][-1] == {"side": "admin", "text":
                                  "hello Ada, checking now",
                                  "at": tk["messages"][-1]["at"]}
    assert any("\u2192 sent to" in b for b in _to_admin(sent))


def test_relay_accepts_number_without_plus(session_factory, user, sent):
    main._handle_incoming_message(USER, "SUPPORT")
    sent.clear()
    main._handle_incoming_message(ADMIN, f"R {USER.lstrip('+')} on it")

    assert any("on it" in b for b in _to_user(sent))


def test_relay_refuses_unknown_target(session_factory, sent):
    main._handle_incoming_message(ADMIN, "R +2349999999990 who is this")
    assert any("Unknown user" in b for b in _to_admin(sent))
    assert not _to_user(sent)


def test_relay_multiline_body(session_factory, user, sent):
    main._handle_incoming_message(USER, "SUPPORT")
    sent.clear()
    main._handle_incoming_message(ADMIN, f"R {USER} first line\nsecond line")

    assert "first line\nsecond line" in "\n".join(_to_user(sent))


# --- Q5: closing ------------------------------------------------------------
def test_admin_done_closes_and_notifies(session_factory, user, sent):
    main._handle_incoming_message(USER, "SUPPORT")
    sent.clear()

    main._handle_incoming_message(ADMIN, f"/done {USER}")

    assert "closed" in "\n".join(_to_user(sent)).lower()
    assert any("Closed SUP-" in b for b in _to_admin(sent))
    assert _ticket(session_factory)["status"] == "resolved"


def test_done_with_two_open_threads_asks_for_phone(session_factory, user,
                                                   sent):
    s = session_factory()
    s.add(User(phone="+2348022222222", name="Second"))
    s.commit()
    s.close()
    main._handle_incoming_message(USER, "SUPPORT")
    main._handle_incoming_message("+2348022222222", "SUPPORT")
    sent.clear()

    main._handle_incoming_message(ADMIN, "/done")

    assert any("/done <phone>" in b for b in _to_admin(sent))
    assert _ticket(session_factory)["status"] == "open"


def test_user_close_word_closes_thread(session_factory, user, sent):
    main._handle_incoming_message(USER, "SUPPORT")
    sent.clear()

    main._handle_incoming_message(USER, "solved, thank you!")

    assert _ticket(session_factory)["status"] == "resolved"
    assert any("BY USER" in b for b in _to_admin(sent))


def test_close_word_with_no_thread_falls_through(session_factory, user,
                                                 sent):
    main._handle_incoming_message(USER, "solved")
    assert _ticket(session_factory) is None


def test_relay_to_closed_thread_delivers_but_flags(session_factory, user,
                                                   sent):
    main._handle_incoming_message(USER, "SUPPORT")
    main._handle_incoming_message(ADMIN, f"/done {USER}")
    sent.clear()

    main._handle_incoming_message(ADMIN, f"R {USER} late reply")

    assert any("late reply" in b for b in _to_user(sent))
    assert any("thread was resolved" in b for b in _to_admin(sent))


# --- Q6: throttle exemption + STOP ------------------------------------------
def test_open_thread_bypasses_rate_limit(session_factory, user, sent,
                                         monkeypatch):
    monkeypatch.setattr(main, "_RATE_LIMIT", 1)
    main._handle_incoming_message(USER, "SUPPORT")
    sent.clear()

    main._handle_incoming_message(USER, "message two")
    main._handle_incoming_message(USER, "message three")

    admin = "\n".join(_to_admin(sent))
    assert "message two" in admin and "message three" in admin
    assert not any("Easy o!" in b for b in _to_user(sent))


def test_stop_clears_thread_and_restores_throttle(session_factory, user,
                                                  sent, monkeypatch):
    monkeypatch.setattr(main, "_RATE_LIMIT", 1)
    main._handle_incoming_message(USER, "SUPPORT")

    main._handle_incoming_message(USER, "STOP")
    assert _ticket(session_factory) is None

    sent.clear()
    main._handle_incoming_message(USER, "one")
    main._handle_incoming_message(USER, "two")
    assert any("Easy o!" in b for b in _to_user(sent)), \
        "after STOP the normal throttle applies again"


# --- Q7: agent-failure auto-escalation --------------------------------------
def test_two_agent_failures_open_thread(session_factory, user, sent,
                                        monkeypatch):
    monkeypatch.setattr(main, "GROQ_API_KEY", "test-key")

    def _boom(*a, **k):
        raise RuntimeError("groq quota")

    monkeypatch.setattr("FareBeep.agent.agent_reply", _boom)

    main._handle_incoming_message(USER, "lagos to abuja friday")
    assert _ticket(session_factory) is None, "one failure is not a pattern"

    main._handle_incoming_message(USER, "lagos to abuja friday")
    tk = _ticket(session_factory)
    assert tk and tk["trigger"] == "auto: bot failed twice"
    assert any("SUPPORT SUP-" in b for b in _to_admin(sent))

    # a successful turn resets the counter (fresh user, one failure, no
    # ticket): use a second phone
    s = session_factory()
    s.add(User(phone="+2348033333333", name="Chike"))
    s.commit()
    s.close()
    monkeypatch.setattr("FareBeep.agent.agent_reply",
                        lambda *a, **k: "all good")
    main._handle_incoming_message("+2348033333333", "hi there")
    main._handle_incoming_message("+2348033333333", "lagos to abuja")
    monkeypatch.setattr("FareBeep.agent.agent_reply", _boom)
    main._handle_incoming_message("+2348033333333", "and now?")
    assert _ticket(session_factory, "+2348033333333") is None


# --- Q8: TTL auto-close ------------------------------------------------------
def test_stale_thread_auto_closes_on_touch(session_factory, user, sent):
    from datetime import timedelta
    chatstate.set_support_ticket(
        session_factory(), USER,
        {"id": "SUP-OLD1", "opened_at":
         (main.utcnow() - timedelta(hours=72)).isoformat(),
         "trigger": "explicit ask", "status": "open",
         "messages": [{"side": "user", "at": "", "text": "old"}]})

    main._handle_incoming_message(USER, "hello again")

    tk = _ticket(session_factory)
    assert tk["status"] == "stale"
    assert any("closed automatically" in b for b in _to_user(sent))


# --- Q9: ops visibility ------------------------------------------------------
def test_ops_panel_lists_open_threads(session_factory, user, monkeypatch):
    monkeypatch.setattr(main, "ADMIN_TOKEN", "tok")
    main._handle_incoming_message(USER, "SUPPORT")

    client = TestClient(main.app)
    r = client.get("/admin/ops", headers={"X-Admin-Token": "tok"})
    assert r.status_code == 200
    threads = r.json()["support"]["open_threads"]
    assert len(threads) == 1
    assert threads[0]["phone"] == USER
    assert threads[0]["status"] == "open"


# --- module boundary: support.py owns these helpers ------------------------
def test_support_module_owns_admin_console_and_ops_rows(session_factory,
                                                        user, sent):
    main._handle_incoming_message(USER, "SUPPORT")
    sent.clear()

    assert support._handle_admin_console(ADMIN, "/open") is True
    assert any(USER in b for b in _to_admin(sent))

    s = session_factory()
    try:
        rows = support._ops_support_rows(s)
    finally:
        s.close()
    assert [r["phone"] for r in rows] == [USER]
    assert rows[0]["relay"] == f"R {USER} "


def test_ticket_link_token_roundtrip_lives_in_support():
    tok = support._ticket_link_token(USER)
    assert support._ticket_link_phone(tok) == USER
    assert main._ticket_link_token is support._ticket_link_token
