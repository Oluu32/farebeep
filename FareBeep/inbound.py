"""Inbound message queue - save-before-ack, leases, idempotent dispatch.

Meta/Telegram/Twilio inbound messages go through this module:

  1. `_claim_inbound_messages` inserts each wamid as `queued` BEFORE the
     webhook acks (save-before-ack). Repeats are dropped; failed rows with
     attempts left are re-queued (bounded by MAX_INBOUND_ATTEMPTS).
  2. `acquire_queued_messages` takes an ownership LEASE on rows with one
     atomic UPDATE ... RETURNING, so two workers never hold the same row.
  3. `_dispatch_single` runs one message through the concierge pipeline;
     `_mark_processed` records the outcome and releases the lease.
  4. `recover_orphaned_inbound` replays rows orphaned by a crash-after-ack.

Dependencies that tests monkeypatch on `FareBeep.main` (SessionLocal,
MetaWhatsapp, the _handle_* turn handlers, _dispatch_single) are resolved
through `_main()` at call time, not imported, so those patches still apply.
This module must never import main at load time (main imports it).
"""
import logging
import os
import threading
import uuid
from datetime import timedelta

from sqlalchemy import and_, or_, select, update
from sqlalchemy.exc import IntegrityError

from FareBeep.models import ProcessedMessage, utcnow

# Logger name kept as before so existing log routing is unchanged.
logger = logging.getLogger("farebeep.main")


def _main():
    """The FareBeep.main module, looked up at call time (see docstring)."""
    from FareBeep import main
    return main


MAX_INBOUND_ATTEMPTS = 3

# Ownership lease: a worker dispatches ONLY rows it holds a live lease
# on. Web batches, the startup sweep and the periodic sweep all take
# rows through acquire_queued_messages (one atomic UPDATE ... WHERE the
# lease is free-or-expired ... RETURNING), so two workers - threads in
# this process or separate processes/replicas - can never hold the same
# row at once. The sweep additionally requires row AGE, the lease
# requires OWNERSHIP: a slow-but-live turn holds its lease, so recovery
# can never steal it no matter how long it runs.
LEASE_SECONDS = 600


def _lease_owner(prefix: str) -> str:
    """Ops-visible lease holder id: role + pid + thread + random."""
    return (f"{prefix}:{os.getpid()}:{threading.get_ident()}:"
            f"{uuid.uuid4().hex[:8]}")


def acquire_queued_messages(owner: str, phone: str = None,
                            stale_minutes: int = 0,
                            lease_seconds: int = LEASE_SECONDS,
                            limit: int = None) -> list:
    """Atomically lease queued rows. ONE UPDATE statement takes every
    matching row whose lease is free (NULL) or expired and stamps it
    with (owner, deadline) - the database serializes concurrent
    acquirers, so the returned sets are always disjoint. Returns
    [{message_id, payload, created_at, phone}]. Never raises (empty on
    DB error - a missed sweep is safer than a crashed webhook)."""
    now = utcnow()
    conds = [
        ProcessedMessage.status == "queued",
        or_(ProcessedMessage.lease_expires_at.is_(None),
            ProcessedMessage.lease_expires_at < now),
    ]
    if phone is not None:
        conds.append(ProcessedMessage.phone == phone)
    if stale_minutes:
        conds.append(ProcessedMessage.created_at
                     < now - timedelta(minutes=stale_minutes))
    if limit:
        # Oldest claims first (bounds one batch without starving old
        # rows). Portable form: Postgres has no UPDATE..ORDER BY..LIMIT,
        # so cap via a subselect both engines accept.
        conds.append(ProcessedMessage.message_id.in_(
            select(ProcessedMessage.message_id)
            .where(and_(*conds))
            .order_by(ProcessedMessage.created_at)
            .limit(limit)
            .scalar_subquery()))
    stmt = (update(ProcessedMessage).where(and_(*conds))
            .values(lease_owner=owner,
                    lease_expires_at=now + timedelta(seconds=lease_seconds))
            .returning(ProcessedMessage.message_id,
                       ProcessedMessage.payload,
                       ProcessedMessage.created_at,
                       ProcessedMessage.phone))
    db = _main().SessionLocal()
    try:
        rows = db.execute(stmt).all()
        db.commit()
        return [{"message_id": r[0], "payload": r[1],
                 "created_at": r[2], "phone": r[3]} for r in rows]
    except Exception as e:
        logger.warning("Lease acquire failed (%s): %s", owner, e)
        try:
            db.rollback()
        except Exception:
            pass
        return []
    finally:
        db.close()


def _snapshot_message(m) -> dict:
    """JSON-safe snapshot of a classified message, stored at claim time so
    a crash-after-ack is recoverable: the payload carries everything
    _dispatch_single needs (no re-fetch, no Meta redelivery required)."""
    mtype = (m.message_type.value
             if hasattr(m.message_type, "value") else str(m.message_type))
    return {
        "message_id": m.message_id or "",
        "from_number": m.from_number or "",
        "message_type": mtype,
        "timestamp": m.timestamp or "",
        "text": m.text,
        "button_id": m.button_id,
        "button_title": m.button_title,
        "list_id": m.list_id,
        "list_title": m.list_title,
        "flow_token": m.flow_token,
        "flow_data": m.flow_data,
        "raw": m.raw if isinstance(m.raw, dict) else {},
    }


def _restore_message(snap: dict):
    """Rebuild an InboundMessage from a stored snapshot (startup recovery)."""
    from FareBeep.whatsapp.router import InboundMessage, MessageType
    try:
        mtype = MessageType((snap or {}).get("message_type", "unknown"))
    except ValueError:
        mtype = MessageType.UNKNOWN
    return InboundMessage(
        message_id=(snap or {}).get("message_id", ""),
        from_number=(snap or {}).get("from_number", ""),
        message_type=mtype,
        timestamp=(snap or {}).get("timestamp", ""),
        raw=(snap or {}).get("raw", {}) or {},
        text=(snap or {}).get("text"),
        button_id=(snap or {}).get("button_id"),
        button_title=(snap or {}).get("button_title"),
        list_id=(snap or {}).get("list_id"),
        list_title=(snap or {}).get("list_title"),
        flow_token=(snap or {}).get("flow_token"),
        flow_data=(snap or {}).get("flow_data"),
    )


def recover_orphaned_inbound(db=None, stale_minutes: int = 0) -> int:
    """Re-dispatch rows stuck at queued, rebuilt from stored payloads.

    Two callers:
      - startup (_startup, stale_minutes=0): at boot no worker is alive,
        so EVERY queued row is a crash-after-ack orphan - replay all.
      - worker sweep (run_cycles, stale_minutes=5): periodic safety net
        that needs NO restart and NO Meta redelivery. Only rows queued
        longer than the cutoff are orphans - fresh rows may still have a
        live background task working on them, so they are left alone.

    ORDER RULE (shared with _process_inbound_batch): sender timestamp
    first (user-intent order - one payload can arrive wire-shuffled),
    claim time (created_at) breaks ties and orders separate requests.
    See _arrival_key.

    Attempts accounting continues, so a poison message still
    dead-letters instead of looping forever. Rows with NO stored
    payload (claimed by the pre-recovery build) can never be replayed:
    stale ones are parked as exhausted/failed with an explicit error
    instead of cycling. Returns the count re-dispatched. Never raises.
    """
    own = db is None
    db = db or _main().SessionLocal()
    try:
        # Lease, don't just read: rows whose lease is live belong to a
        # running worker (possibly on another thread/process) and are
        # NOT orphans, however old they look. Only free-or-expired rows
        # are taken - atomically, so a concurrent batch and sweep split
        # the work instead of doubling it.
        taken = acquire_queued_messages(
            _lease_owner("sweep" if stale_minutes else "startup"),
            stale_minutes=stale_minutes)
        legacy = [t for t in taken
                  if not (t["payload"] or {}).get("message_id")]
        if legacy:
            db.query(ProcessedMessage).filter(
                ProcessedMessage.message_id.in_(
                    [t["message_id"] for t in legacy])).update(
                {"status": "failed",
                 "attempts": MAX_INBOUND_ATTEMPTS,
                 "last_error": ("no stored payload (claimed before crash "
                                "recovery shipped) - ask the customer to "
                                "resend"),
                 "processed_at": utcnow(),
                 "lease_owner": None,
                 "lease_expires_at": None},
                synchronize_session=False)
            db.commit()
        replayable = [t for t in taken
                      if (t["payload"] or {}).get("message_id")]
        if not replayable:
            return 0
        logger.warning("Recovery: re-dispatching %d orphaned inbound "
                       "message(s)", len(replayable))
        grouped: dict = {}
        order: list = []
        for t in replayable:
            snap = t["payload"] or {}
            phone = snap.get("from_number") or t["phone"] or ""
            if not phone:
                continue
            if phone not in grouped:
                grouped[phone] = []
                order.append(phone)
            grouped[phone].append(t)
        for phone in order:
            with _phone_lock(phone):
                batch = sorted(
                    grouped[phone],
                    key=lambda t: _arrival_key(t["payload"], t["created_at"]))
                for t in batch:
                    m = _restore_message(t["payload"])
                    try:
                        # Resolved on main so a patched _dispatch_single applies.
                        _main()._dispatch_single(m)
                        _mark_processed(m.message_id, "done")
                    except Exception as e:
                        logger.error("Recovery handling failed (%s): %s",
                                     phone, e)
                        _mark_processed(m.message_id, "failed",
                                        error=str(e))
        return len(replayable)
    except Exception as e:
        logger.error("Recovery sweep failed (orphans stay queued): %s", e)
        return 0
    finally:
        if own:
            try:
                db.close()
            except Exception:
                pass


# Per-phone batch locks: two webhook POSTs for the same customer (or a
# startup recovery racing a live request) must never interleave turns -
# each phone's batch runs alone. Single-process scope: it serializes the
# threadpool BackgroundTasks on this replica. Cross-replica races need
# DB-level serialization (Railway runs one replica by default - flag it
# before scaling web replicas).
_PHONE_LOCKS: dict = {}
_PHONE_LOCKS_GUARD = threading.Lock()


def _phone_lock(phone: str) -> threading.Lock:
    with _PHONE_LOCKS_GUARD:
        lock = _PHONE_LOCKS.get(phone)
        if lock is None:
            lock = threading.Lock()
            _PHONE_LOCKS[phone] = lock
        return lock


def _arrival_key(payload: dict, created) -> tuple:
    """Sort key for one customer's queued rows: sender timestamp first,
    claim time second. Timestamp = the order the human typed them (Meta
    can deliver one payload wire-shuffled); created_at = our arrival
    record, which orders separate requests and breaks timestamp ties.
    NOTE the limit this implies: only ARRIVED (claimed) messages can be
    ordered - a message Meta hasn't delivered yet cannot be sequenced,
    so a late arrival with an early timestamp still runs late."""
    try:
        ts = int(str((payload or {}).get("timestamp") or ""))
        has_ts = 0
    except (ValueError, TypeError):
        ts, has_ts = 0, 1
    created_key = (created.isoformat() if hasattr(created, "isoformat")
                   else str(created or ""))
    return (has_ts, ts, created_key)


def _classify_all_entries(payload: dict) -> list:
    """Classify every message in every entry. Legacy template quick-reply
    taps (messages[].button.payload - no 'interactive' wrapper) bypass the
    router, so they are re-attached here as BUTTON_REPLY (same tap ids the
    old single-message path fed to cards.translate_tap)."""
    from FareBeep.whatsapp.router import (InboundMessage, MessageType,
                                          classify_message)
    classified: list = []
    seen_ids: set = set()
    for entry in payload.get("entry") or []:
        try:
            batch = classify_message(entry)
        except Exception as e:
            logger.warning("classify_message failed on entry: %s", e)
            batch = []
        for m in batch:
            classified.append(m)
            if m.message_id:
                seen_ids.add(m.message_id)
    # Legacy sweep: template button payloads the router drops (it returns
    # None for unknown shapes). Preserves the old BOOK/pick/alert/beep tap
    # behaviour for those messages.
    for entry in payload.get("entry") or []:
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []):
                if not isinstance(msg, dict):
                    continue
                mid = msg.get("id", "")
                if mid and mid in seen_ids:
                    continue
                tap_id = (msg.get("button") or {}).get("payload") or ""
                if tap_id and msg.get("from"):
                    classified.append(InboundMessage(
                        message_id=mid,
                        from_number=msg.get("from", ""),
                        message_type=MessageType.BUTTON_REPLY,
                        timestamp=msg.get("timestamp", ""),
                        raw=msg,
                        button_id=tap_id,
                        button_title=tap_id,
                    ))
                    if mid:
                        seen_ids.add(mid)
    return classified


def _claim_inbound_messages(messages: list) -> list:
    """Save-before-ack: insert new wamids as queued. Duplicates (PK clash)
    are dropped - EXCEPT failed rows with attempts left, which Meta's own
    redelivery reclaims (status back to queued, attempts preserved). That
    is the retry transport: durable across restarts, bounded by
    MAX_INBOUND_ATTEMPTS, and it never re-runs a done turn (no double-book).
    Messages WITHOUT an id (old tests, malformed retries) cannot dedupe -
    they always pass through, preserving legacy behaviour."""
    claimed = []
    db = _main().SessionLocal()
    try:
        for m in messages:
            mid = (m.message_id or "").strip()
            if not mid:
                claimed.append(m)
                continue
            try:
                db.add(ProcessedMessage(
                    message_id=mid,
                    phone=m.from_number or None,
                    message_type=(m.message_type.value
                                  if hasattr(m.message_type, "value")
                                  else str(m.message_type)),
                    status="queued",
                    payload=_snapshot_message(m),
                ))
                db.commit()
                claimed.append(m)
            except IntegrityError:
                db.rollback()
                row = db.query(ProcessedMessage).filter_by(
                    message_id=mid).first()
                if (row is not None and row.status == "failed"
                        and (row.attempts or 0) < MAX_INBOUND_ATTEMPTS):
                    row.status = "queued"
                    row.lease_owner = None     # next acquirer leases it
                    row.lease_expires_at = None
                    db.commit()
                    claimed.append(m)
                    logger.info("Reclaim: retrying failed wamid %s "
                                "(attempt %s)", mid, (row.attempts or 0) + 1)
                else:
                    logger.info("Dedup: dropping repeat wamid %s", mid)
            except Exception as e:
                db.rollback()
                logger.warning("Claim failed for %s: %s", mid, e)
    finally:
        db.close()
    return claimed


def _mark_processed(message_id: str, status: str, error: str = None) -> None:
    """Flip a claimed row to done/failed. Failed bumps attempts and keeps
    the truncated error - the row stays reclaimable until attempts run
    out, then it is permanently dropped (ops-visible via last_error)."""
    if not (message_id or "").strip():
        return
    db = _main().SessionLocal()
    try:
        row = db.query(ProcessedMessage).filter_by(
            message_id=message_id).first()
        if row is not None:
            row.status = status
            row.lease_owner = None       # release ownership with the outcome
            row.lease_expires_at = None
            if status == "failed":
                row.attempts = (row.attempts or 0) + 1
                if error:
                    row.last_error = str(error)[:500]
                if row.attempts >= MAX_INBOUND_ATTEMPTS:
                    logger.error(
                        "Dead-letter: wamid %s from %s failed %d times - "
                        "retained in processed_messages, needs a human "
                        "(last error: %s)",
                        message_id, row.phone, row.attempts,
                        (row.last_error or "")[:200])
            if status in ("done", "failed"):
                row.processed_at = utcnow()
            db.commit()
    except Exception as e:
        logger.warning("Mark %s=%s failed: %s", message_id, status, e)
    finally:
        db.close()


def _dispatch_single(msg) -> None:
    """One classified message through the EXISTING pipeline. Typing bubble
    lives here (background) so the webhook thread never blocks on network.
    Turn handlers are resolved on main at call time so patches apply."""
    from FareBeep.whatsapp.router import MessageType
    from FareBeep import interactive as ix
    m = _main()
    phone = msg.from_number or ""
    if not phone:
        return
    try:
        m.MetaWhatsapp().send_typing_indicator(phone)
    except Exception as e:
        logger.warning("Typing bubble failed (%s): %s", phone, e)
    if msg.message_type == MessageType.TEXT:
        if (msg.text or "").strip():
            m._handle_incoming_message(phone, msg.text)
        return
    if msg.message_type == MessageType.UNKNOWN:
        # Legacy tolerance: typeless payloads (old tests, early Meta
        # shape) carry text.body with no "type" field. The old
        # single-message path read that body directly - do the same
        # rather than dropping a real customer message.
        legacy = ((msg.raw or {}).get("text") or {}).get("body", "")
        if isinstance(legacy, str) and legacy.strip():
            m._handle_incoming_message(phone, legacy)
        return
    if msg.message_type == MessageType.FLOW_RESPONSE:
        ix._handle_flow_completion(phone, msg.flow_token, msg.flow_data)
        return
    if msg.message_type in (MessageType.BUTTON_REPLY,
                            MessageType.LIST_REPLY):
        tap_id = msg.button_id or msg.list_id or ""
        if not tap_id:
            return
        ix._dispatch_tap(phone, tap_id, source="meta")
        return
    if msg.message_type == MessageType.DOCUMENT:
        ix._handle_boarding_pass(phone, msg)
        return
    logger.info("Inbound %s from %s needs no concierge turn - ignored",
                msg.message_type, phone)


__all__ = [
    "MAX_INBOUND_ATTEMPTS", "LEASE_SECONDS", "acquire_queued_messages",
    "recover_orphaned_inbound", "_dispatch_single", "_claim_inbound_messages",
    "_classify_all_entries", "_mark_processed", "_phone_lock",
    "_lease_owner", "_restore_message", "_snapshot_message", "_arrival_key",
]
