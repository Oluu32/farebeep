import os

# Hermetic tests, part 1: neutralize every deliberate backoff sleep (HTTP
# retry backoff, LLM 429 backoff, low-price anomaly re-check) BEFORE
# FareBeep.config is imported anywhere. Retry loops keep their semantics -
# they just don't really wait. Production defaults to SLEEP_SCALE=1.0
# (see config.py).
os.environ.setdefault("SLEEP_SCALE", "0")

# Hermetic tests, part 2: the shared database singleton must never point
# at the real Supabase Postgres during tests. The .env carries a live
# SUPABASE_DB_URL, so any code path that reaches the module-global
# SessionLocal without a per-test monkeypatch (e.g. the rate-limit
# support-thread probe) would otherwise do a network round-trip per query
# - slow, weather-dependent, and reading/writing production rows.
#
# Belt: fall back to local SQLite if anything ever rebuilds the engine.
# Braces: build the lazy singleton NOW, bound to in-memory SQLite with
# StaticPool (one connection shared across threads - same pattern as the
# `db` fixture below, needed because TestClient runs the app on a worker
# thread).
os.environ.setdefault("FALLBACK_TO_SQLITE", "1")
from FareBeep import database  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

database._engine = database.make_engine("sqlite://", "SQLite (fallback)")

import pytest  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from FareBeep.models import Base  # noqa: E402

_shared_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
database._engine = _shared_engine
database._sessionmaker = sessionmaker(bind=_shared_engine,
                                      autocommit=False, autoflush=False)


@pytest.fixture(autouse=True)
def _hermetic_network(monkeypatch):
    """No test may dial the real internet. The live-data seam that leaked
    out of otherwise-offline tests:
    - the live supplier engine (search.SupplierLiveEngine): _offers is
      stubbed to return [] so any ledger miss is a clean 'no fare' instead
      of a network call.
    """
    from FareBeep import search as search_mod
    monkeypatch.setattr(search_mod.SupplierLiveEngine, "_offers",
                        lambda self, origin, destination, flight_date: [])

    # Telegram Bot API: the webhook's best-effort typing bubble constructs a
    # fresh TelegramBot() (module-local import), and main.notifier is a
    # TelegramBot singleton built from the real .env token at import time.
    # Without this stand-in every webhook post makes a real api.telegram.org
    # round-trip. Tests that exercise the TelegramBot class itself construct
    # it directly (module-level import, unaffected) with their own fakes.
    from FareBeep import main as main_mod
    from FareBeep import notifier as notifier_mod

    class _NullTelegram:
        """Network-free TelegramBot stand-in."""

        def __init__(self, *a, **k):
            pass

        def send_text(self, to, body, **k):
            return True

        def send_action(self, to, action="typing"):
            return True

        def send_interactive_card(self, to, body, buttons, **k):
            return True

        def answer_callback(self, callback_id, text=None):
            return True

    monkeypatch.setattr(notifier_mod, "TelegramBot", _NullTelegram)

    # Meta Cloud API: the webhook receiver fires a best-effort typing
    # indicator per inbound message (main calls MetaWhatsapp() directly,
    # name resolved in main's namespace) - a real graph.facebook.com
    # round-trip per tap without this stand-in.
    class _NullMeta(_NullTelegram):

        def send_typing_indicator(self, to):
            return True

        def _send(self, to, payload, what=""):
            return True

    monkeypatch.setattr(main_mod, "MetaWhatsapp", _NullMeta)
    if hasattr(main_mod, "notifier"):
        monkeypatch.setattr(main_mod.notifier, "send_text",
                            lambda to, body, **k: True, raising=False)
        monkeypatch.setattr(main_mod.notifier, "send_action",
                            lambda to, action="typing": True, raising=False)
        monkeypatch.setattr(main_mod.notifier, "send_document",
                            lambda *a, **k: True, raising=False)
    yield


@pytest.fixture(autouse=True)
def _hermetic_db():
    """Fresh shared schema per test: anything that slipped through to the
    module-global SessionLocal sees clean tables and leaves nothing behind."""
    Base.metadata.drop_all(_shared_engine)
    Base.metadata.create_all(_shared_engine)
    yield
    Base.metadata.drop_all(_shared_engine)


@pytest.fixture(autouse=True)
def _reset_rate_limit_buckets():
    """The per-phone throttle lives in module-global dicts; without this,
    a message-heavy test file (concierge pipeline) throttles later tests
    for the same phone."""
    from FareBeep import main
    main._rate_hits.clear()
    main._rate_cooled.clear()
    yield
    main._rate_hits.clear()
    main._rate_cooled.clear()


@pytest.fixture
def db():
    """In-memory SQLite session - models are portable (Uuid, no PG-only types)."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    Base.metadata.drop_all(engine)
