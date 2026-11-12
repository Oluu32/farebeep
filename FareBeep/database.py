"""Database layer - SQLAlchemy over the Supabase Postgres Shared Ledger.

Point SUPABASE_DB_URL at the Supabase connection string (see .env.example)
and every table in models.py maps to schema.sql (run it once in the
Supabase SQL Editor). The shared ledger (fare_ledger) is what makes FareBeep
a *community* utility: every search result is UPSERTed (Postgres
ON CONFLICT - see search._upsert_fare) so the next user asking the same
(origin, destination, date) gets a <500ms ledger hit instead of a paid
live supplier call.

Connection strategy (cloud latency):
  - psycopg2 driver, pool_pre_ping so a dead Supabase connection is
    transparently replaced before use
  - a small bounded pool (pool_size=5, max_overflow=10) with pool_recycle
    to keep connections fresh across Supabase's idle-kill timeouts
  - connect_timeout=10 so a slow network fails fast instead of hanging

The engine is created LAZILY (first session), so importing this module
never fails - even when the driver isn't installed yet.

SQLITE FALLBACK (kept, commented, dev-only): when SUPABASE_DB_URL is not
set, the app can run against a local file with FALLBACK_TO_SQLITE=1 - the
same schema semantics, for tests / laptop demos. Never ship that flag.
"""
import logging
import os
import sys
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from FareBeep.config import BASE_DIR, SUPABASE_DB_URL

logger = logging.getLogger("farebeep.database")

# -- Cloud first: Supabase Postgres (the production Shared Ledger) ----------
if SUPABASE_DB_URL:
    DATABASE_URL = SUPABASE_DB_URL
    DATABASE_PROVIDER = "Supabase"
elif os.getenv("FALLBACK_TO_SQLITE", "0") == "1":
    # ------------------------------------------------------------------
    # SQLite fallback (DEV/TESTS ONLY - commented out for production).
    # The pre-cloud prototype used sqlite:///farebeep_local.db; it is
    # preserved behind an explicit flag so tests and laptop demos keep
    # working with ZERO cloud setup. Do NOT set FALLBACK_TO_SQLITE in
    # production - the app must run on the Supabase Shared Ledger.
    # ------------------------------------------------------------------
    DATABASE_URL = f"sqlite:///{(BASE_DIR / 'farebeep_local.db').as_posix()}"
    DATABASE_PROVIDER = "SQLite (fallback)"
else:
    raise RuntimeError(
        "SUPABASE_DB_URL is not set. Add the Supabase Postgres connection "
        "string to FareBeep/.env (or set FALLBACK_TO_SQLITE=1 for local "
        "development only).")

_engine: Engine = None
_sessionmaker = None

# Cloud-friendly pool settings (Supabase Postgres, shared-compute limits)
_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
_POOL_MAX_OVERFLOW = int(os.getenv("DB_POOL_MAX_OVERFLOW", "10"))
_POOL_RECYCLE_SECONDS = int(os.getenv("DB_POOL_RECYCLE_SECONDS", "1800"))


def _normalize_postgres_url(url: str) -> str:
    """Force the psycopg2 driver so the Postgres URL is always
    `postgresql+psycopg2://...` regardless of the scheme Supabase gave us
    (postgres://, postgresql://, postgresql+psycopg2:// all work)."""
    for prefix in ("postgresql+psycopg2://", "postgresql+psycopg2+asyncpg://",
                   "postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg2://" + url.split("://", 1)[1]
    return url


def make_engine(url: str = DATABASE_URL, provider: str = DATABASE_PROVIDER):
    """Build the engine for `url`.

    - Supabase/Postgres: psycopg2 + the cloud pool settings above.
    - SQLite fallback (dev/tests): check_same_thread=False, plain pool.
    """
    if provider != "SQLite (fallback)" or not url.startswith("sqlite"):
        url = _normalize_postgres_url(url)
        return create_engine(
            url,
            pool_pre_ping=True,                    # replace dead Supabase conns
            pool_size=_POOL_SIZE,
            max_overflow=_POOL_MAX_OVERFLOW,
            pool_recycle=_POOL_RECYCLE_SECONDS,    # survive idle-kill timeouts
            connect_args={"connect_timeout": 10},  # fail fast, don't hang
        )
    # --- sqlite fallback (dev/tests only) ---
    return create_engine(url, pool_pre_ping=True,
                         connect_args={"check_same_thread": False})


def get_engine():
    """Lazy engine singleton - the DB driver is only required on first use."""
    global _engine
    if _engine is None:
        try:
            _engine = make_engine(DATABASE_URL, DATABASE_PROVIDER)
            logger.info("DB engine ready: %s (%s)",
                        DATABASE_PROVIDER,
                        DATABASE_URL.split("@")[-1].split(":")[0])
        except ImportError as e:  # e.g. psycopg2 missing
            raise RuntimeError(
                f"Cannot create engine for {DATABASE_URL.split('@')[-1]}: {e}. "
                "Install requirements.txt (psycopg2-binary).") from e
    return _engine


def SessionLocal() -> Session:
    """Create a new scoped session (binds the lazy engine on first call)."""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = sessionmaker(bind=get_engine(),
                                     autocommit=False, autoflush=False)
    return _sessionmaker()


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_connection(db: Session) -> bool:
    """Liveness helper: run `select 1` through the session's bind."""
    try:
        db.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def _banner(msg: str) -> None:
    """Console banner that survives Windows' cp1252 redirected output:
    reconfigure the streams to UTF-8 (replacing unencodable chars) so the
    ✅/❌ mission banner prints on any platform."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print(msg)


def verify_connection() -> bool:
    """Startup check - the mission banner.

    Prints "✅ Connected to Supabase Shared Ledger" to the console when the
    production database answers `select 1` (also logged so Railway shows it).
    """
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
        if DATABASE_PROVIDER == "Supabase":
            _banner("✅ Connected to Supabase Shared Ledger")
        else:
            _banner("✅ Connected to local SQLite (dev fallback - "
                    "set SUPABASE_DB_URL for production)")
        logger.info("Database connection verified: %s", DATABASE_PROVIDER)
        return True
    except Exception as e:
        _banner(f"❌ Database connection FAILED: {e}")
        logger.error("Database connection FAILED: %s", e)
        return False


def init_db(base) -> None:
    """Create tables if missing (runtime, not provider-dependent).

    - SQLite fallback: creates farebeep_local.db on first run.
    - Supabase: tables are normally created via schema.sql; this is a safe
      auto-idempotent fallback for development.
    Afterwards: additive migrations, then the drift alarm (anything the
    models expect but the live DB lacks - the bot breaks silently on
    those, so they scream here instead).
    """
    base.metadata.create_all(bind=get_engine())
    _apply_additive_migrations()
    drifted = check_schema_drift(base)
    if drifted:
        _banner(f"❌ SCHEMA DRIFT: {len(drifted)} column(s) missing from "
                f"the live DB: {', '.join(drifted)} - see log")


def check_schema_drift(base) -> list:
    """Model-vs-database column diff. Returns ['table.column', ...]
    missing from the live DB (tables entirely absent as 'table.*').

    Runs AFTER _apply_additive_migrations, so anything listed here is
    NOT self-healable: add it to the migrations list or schema.sql.
    Every gap is logger.error - a missing column fails requests with
    raw SQL errors and no user-facing explanation.
    """
    from sqlalchemy import inspect
    try:
        inspector = inspect(get_engine())
        live_tables = set(inspector.get_table_names())
    except Exception as e:
        logger.error("Schema drift check failed (cannot inspect DB): %s", e)
        return ["<check-failed>"]
    missing = []
    for table in base.metadata.tables.values():
        if table.name not in live_tables:
            missing.append(f"{table.name}.* (table)")
            continue
        try:
            live_cols = {c["name"]
                         for c in inspector.get_columns(table.name)}
        except Exception as e:
            logger.error("Schema drift check failed on %s: %s",
                         table.name, e)
            continue
        for col in table.columns:
            if col.name not in live_cols:
                missing.append(f"{table.name}.{col.name}")
    for gap in missing:
        logger.error("SCHEMA DRIFT: live DB is missing %s - add it to "
                     "_apply_additive_migrations() or schema.sql", gap)
    if not missing:
        logger.info("Schema drift check: models match the live DB")
    return missing


def _apply_additive_migrations() -> None:
    """Additive column migrations: create_all only creates missing TABLES,
    never new columns on tables that already shipped. Each entry is an
    idempotent ALTER ... ADD COLUMN IF NOT EXISTS (Postgres-only - SQLite
    lacks IF NOT EXISTS here, and the test suite builds fresh schemas)."""
    if DATABASE_PROVIDER != "Supabase":
        return
    additive = [
        ("chat_state", "pending_requote", "JSON"),
        ("chat_state", "agent_history", "JSON"),
        ("subscriptions", "paused", "BOOLEAN DEFAULT FALSE"),
        ("processed_messages", "attempts", "INTEGER DEFAULT 0"),
        ("processed_messages", "last_error", "TEXT"),
        ("processed_messages", "payload", "JSONB"),
        ("processed_messages", "lease_owner", "TEXT"),
        ("processed_messages", "lease_expires_at", "TIMESTAMPTZ"),
        ("booking_sessions", "boarding_pass_blob", "BYTEA"),
        ("booking_sessions", "boarding_pass_name", "TEXT"),
        ("booking_sessions", "boarding_pass_mime", "TEXT"),
        ("booking_sessions", "passenger_name", "TEXT"),
        ("booking_sessions", "contact_email", "TEXT"),
        ("chat_state", "pending_ticket_details", "JSONB"),
        ("chat_state", "support_ticket", "JSONB"),
        ("chat_state", "pending_booking", "JSONB"),
        ("chat_state", "pending_voice", "JSONB"),
    ]
    engine = get_engine()
    with engine.begin() as conn:
        for table, column, ddl_type in additive:
            try:
                conn.execute(text(
                    f"ALTER TABLE {table} "
                    f"ADD COLUMN IF NOT EXISTS {column} {ddl_type}"))
                logger.info("Additive migration OK: %s.%s %s",
                            table, column, ddl_type)
            except Exception as e:
                logger.error("Additive migration failed %s.%s: %s",
                             table, column, e)
