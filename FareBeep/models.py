"""SQLAlchemy models for FareBeep - a Transactional Utility (NOT a SaaS).

Migrated FROM naijafly/app/models/models.py:
  - `SeenUser`          -> `User`          (phone-number identity)
  - `UserSubscription`  -> `Subscription`  (flattened: IATA codes, not FK'd
                           to a routes table; a utility stores the route
                           directly on the row)

Dropped (per the reconstruction brief):
  - Twilio-only model columns (Twilio SDK usage removed)
  - Rule-based parser models (n/a - brain.py uses Gemini)
  - SaaS-style Route/Fare satellite tables (the fare_ledger *is* the cargo)

Added for FareBeep core flows:
  - FareLedger      : the shared, community search cache (15-min TTL)
  - BookingSession  : the 10-minute transactional loop state machine
  - StatusWatch     : per-booking 3-hour pre-departure watch window
  - StatusEvent     : status-change dedupe log (template message sent once)
"""
import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        LargeBinary,
                        JSON, String, Text, Uuid, UniqueConstraint)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utcnow() -> datetime:
    """Naive-UTC timestamp for storage (matches Postgres timestamptz when
    the server session timezone is UTC, and survives SQLite round-trips)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_uuid():
    return uuid.uuid4()


# -------------------------------------------------------------------------
# users - migrated from naijafly SeenUser (Meta/WhatsApp phone identity)
# -------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    user_id = Column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    phone = Column(String, unique=True, index=True)   # "+2348012345678"
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    preferred_currency = Column(String, default="NGN")
    first_seen_at = Column(DateTime(timezone=True), default=utcnow)
    # NDPA consent record: timestamp + version of the consent text accepted
    # on the booking confirmation page. NULL until the user agrees.
    consent_at = Column(DateTime(timezone=True), nullable=True)
    consent_text_version = Column(String, nullable=True)

    subscriptions = relationship("Subscription", back_populates="user",
                                 cascade="all, delete-orphan")
    booking_sessions = relationship("BookingSession", back_populates="user",
                                    cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# subscriptions (migrated from naijafly UserSubscription)
# ---------------------------------------------------------------------------
class Subscription(Base):
    __tablename__ = "subscriptions"
    __table_args__ = (
        UniqueConstraint("user_id", "origin", "destination",
                         name="uq_subscription_user_route"),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(ForeignKey("users.user_id"), index=True)
    origin = Column(String)          # IATA code, e.g. "LOS"
    destination = Column(String)     # IATA code, e.g. "ABV"
    target_price = Column(Float)     # None = alert on any >10% drop
    target_date = Column(DateTime(timezone=True), nullable=True)  # NULL = rolling window
    last_price = Column(Float, nullable=True)          # last observed fare (baseline)
    last_alerted_price = Column(Float, nullable=True)  # dedupe: never re-alert same price
    paused = Column(Boolean, default=False)            # user-paused: cycle skips, row kept
    created_at = Column(DateTime(timezone=True), default=utcnow)

    user = relationship("User", back_populates="subscriptions")


# ---------------------------------------------------------------------------
# chat_state - per-chat conversational memory (DB-backed, survives deploys)
# One row per phone; the last quoted fare, last ranked list and any pending
# follow-up question. JSON columns so the schema never changes with features.
# ---------------------------------------------------------------------------
class ChatState(Base):
    __tablename__ = "chat_state"

    id = Column(Integer, primary_key=True)
    phone = Column(String, unique=True, index=True)
    last_fare = Column(JSON, nullable=True)      # {origin_iata, destination_iata,
                                                 #  flight_date, price, airline}
    last_fares = Column(JSON, nullable=True)     # {origin_iata, destination_iata,
                                                 #  flight_date, fares: [...]}
    pending_fare = Column(JSON, nullable=True)   # {origin_iata, destination_iata, date}
    pending_requote = Column(JSON, nullable=True)  # {origin_iata, destination_iata,
                                                    #  flight_date, fare} - waiting for the
                                                    #  user's "yes" after a price move
    pending_ticket_details = Column(JSON, nullable=True)  # {booking_id, stage:
                                                    #  "name"|"email", pnr} - chat
                                                    #  fallback collecting ticket-holder
                                                    #  details the /book page missed
    agent_history = Column(JSON, nullable=True)    # [{role, content}...] - the
                                                 # Groq agent's rolling chat
                                                 # memory (last turns only)
    pending_booking = Column(JSON, nullable=True)  # {origin_iata, destination_iata,
                                                 #  flight_date, price, airline,
                                                 #  travellers, stage: "name"} -
                                                 # deterministic chat-booking
                                                 # collection (main._try_booking_answer)
    support_ticket = Column(JSON, nullable=True)   # {id, opened_at, trigger,
                                                 # status, messages: [...]} -
                                                 # the human-support relay
                                                 # thread (Phase 1: no table)
    pending_voice = Column(JSON, nullable=True)    # voice.VoiceDraft.to_state():
                                                 # a transcribed voice note
                                                 # waiting for YES / the
                                                 # missing piece (voice.py)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


# ---------------------------------------------------------------------------
# fare_ledger - THE SHARED LEDGER (community cache, 8-15 min TTL)
# One row per (origin, destination, flight_date). Upsert target for search.py.
# ---------------------------------------------------------------------------
class FareLedger(Base):
    __tablename__ = "fare_ledger"
    __table_args__ = (
        UniqueConstraint("origin", "destination", "flight_date",
                         name="uq_fare_ledger_route_date"),
    )

    id = Column(Integer, primary_key=True)
    origin = Column(String, index=True)
    destination = Column(String, index=True)
    flight_date = Column(String, index=True)       # "YYYY-MM-DD"
    price = Column(Float)
    currency = Column(String, default="NGN")
    airline = Column(String, nullable=True)
    verify_link = Column(Text, nullable=True)
    last_updated = Column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# fx_rates - USD->NGN snapshots (price tracking, one row per daily fetch)
# The worker records a snapshot every FX_RATE_TTL_HOURS so the founder can
# see the naira trend (quoting itself never converts: suppliers price in NGN).
# ---------------------------------------------------------------------------
class FxRate(Base):
    __tablename__ = "fx_rates"

    id = Column(Integer, primary_key=True)
    usd_ngn = Column(Float)                    # NGN per 1 USD
    source = Column(String, nullable=True)     # e.g. "open.er-api.com"
    fetched_at = Column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# booking_sessions - THE SETTLEMENT ENGINE'S 10-MINUTE TRANSACTIONAL LOOP
# status: pending -> paid | expired | failed
# ---------------------------------------------------------------------------
class SessionStatus(str, enum.Enum):
    PENDING = "pending"
    PAID = "paid"
    EXPIRED = "expired"
    FAILED = "failed"


class BookingSession(Base):
    """One row per BOOK -> Paystack settlement attempt.

    Mission-aligned columns (see schema.sql):
      payment_ref    - unique Paystack reference (FB-<hex>)
      flight_details - JSONB snapshot of {airline, route, net_price, source}
      total_price    - what the user paid (fare + ARHA markup + fee)
      status         - pending | paid | expired | failed
      expires_at     - card: created_at + 10m; otherwise + 13m (bank buffer)
    """

    __tablename__ = "booking_sessions"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    user_id = Column(ForeignKey("users.user_id"), index=True)
    origin = Column(String)
    destination = Column(String)
    flight_date = Column(String)             # "YYYY-MM-DD"
    flight_iata = Column(String, nullable=True)           # e.g. "P47123"
    scheduled_departure = Column(DateTime(timezone=True), nullable=True)
    airline_price = Column(Float)            # net fare from the LIVE supplier hit
    markup = Column(Float, default=5000.0)   # ARHA_MARKUP_NGN flat margin
    processing_fee = Column(Float)           # Paystack fee (user-funded)
    total_price = Column(Float)              # airline_price + markup + fee
    flight_details = Column(JSON, nullable=True)  # {airline, route, net_price, source}
    # Ticket-holder details: captured on the /book confirmation page
    # (chat fallback asks if the page was skipped). contact_email doubles
    # as the Paystack customer email and the ticket-voucher email address.
    passenger_name = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    currency = Column(String, default="NGN")
    status = Column(String, default=SessionStatus.PENDING.value)
    expires_at = Column(DateTime(timezone=True))  # card +10m / otherwise +13m
    payment_ref = Column(String, unique=True, index=True)
    paystack_access_code = Column(String, nullable=True)
    callback_url = Column(String, nullable=True)
    paid_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    # Boarding-pass capture: the user forwards the airline-issued pass
    # (PDF or screenshot) to the bot; bytes live here (small files only).
    boarding_pass_blob = Column(LargeBinary, nullable=True)
    boarding_pass_name = Column(String, nullable=True)
    boarding_pass_mime = Column(String, nullable=True)

    user = relationship("User", back_populates="booking_sessions")


# ---------------------------------------------------------------------------
# status_watches - 3-hour pre-departure watch window (status.py)
# ---------------------------------------------------------------------------
class StatusWatch(Base):
    __tablename__ = "status_watches"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    booking_id = Column(ForeignKey("booking_sessions.id"))
    user_id = Column(ForeignKey("users.user_id"), index=True)
    flight_iata = Column(String)             # e.g. "P47123"
    flight_date = Column(String)             # "YYYY-MM-DD"
    scheduled_departure = Column(DateTime(timezone=True))
    watch_starts_at = Column(DateTime(timezone=True))  # departure - 3h
    last_status = Column(String, nullable=True)
    initiated = Column(Boolean, default=False)
    last_checked_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# status_events - status-change log + template-message dedupe (status.py)
# ---------------------------------------------------------------------------
class StatusEvent(Base):
    __tablename__ = "status_events"

    id = Column(Integer, primary_key=True)
    watch_id = Column(ForeignKey("status_watches.id"), index=True)
    status = Column(String)          # "delayed" | "cancelled" | ...
    previous = Column(String, nullable=True)
    detail = Column(Text, nullable=True)
    template_sent = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# processed_messages - WhatsApp inbound dedupe log (main.meta_webhook)
# One row per Meta wamid (message_id). Inserted BEFORE the 200 ack
# (save-before-ack), so a Meta retry or a worker restart never double-books.
# Portable: String PK + DateTime work on both Supabase Postgres and SQLite.
# status: queued -> done | failed (failed keeps the row so retries stay
# idempotent - the next attempt re-queues by flipping back to queued).
# ---------------------------------------------------------------------------
class ProcessedMessage(Base):
    __tablename__ = "processed_messages"

    message_id = Column(String, primary_key=True)  # Meta wamid
    phone = Column(String, index=True, nullable=True)
    message_type = Column(String, nullable=True)
    status = Column(String, default="queued")       # queued | done | failed
    attempts = Column(Integer, default=0)           # background tries so far
    last_error = Column(Text, nullable=True)        # last failure, truncated
    payload = Column(JSON, nullable=True)           # inbound snapshot for
                                                    # crash recovery (see
                                                    # recover_orphaned_inbound)
    # Ownership lease: WHO may dispatch this row and UNTIL WHEN. A worker
    # dispatches only rows it atomically leased (single UPDATE ... WHERE
    # lease free-or-expired ... RETURNING - see acquire_queued_messages),
    # so web batches, the startup sweep and the periodic sweep can never
    # take the same row concurrently - across threads AND processes.
    lease_owner = Column(String, nullable=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# delivery_receipts - Meta outbound status callbacks (main.meta_webhook)
# One row per OUTBOUND wamid (our sent message id), UPSERTed on every
# sent -> delivered -> read transition. A "failed" status stays visible
# for ops instead of vanishing into the log - but nothing here auto-resends:
# resending is a product decision per message kind (never blind - booking
# and payment messages are NEVER auto-retried from this table).
# ---------------------------------------------------------------------------
class DeliveryReceipt(Base):
    __tablename__ = "delivery_receipts"

    message_id = Column(String, primary_key=True)  # our outbound wamid
    phone = Column(String, index=True, nullable=True)  # recipient_id
    status = Column(String, default="sent")  # sent|delivered|read|failed
    updated_at = Column(DateTime(timezone=True), default=utcnow,
                        onupdate=utcnow)
    created_at = Column(DateTime(timezone=True), default=utcnow)


__all__ = [
    "Base", "utcnow",
    "User", "Subscription", "ChatState", "FareLedger", "FxRate",
    "BookingSession", "SessionStatus", "StatusWatch", "StatusEvent",
    "ProcessedMessage", "DeliveryReceipt",
]
