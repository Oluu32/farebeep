"""Admin surface: ops cockpit, dead-letter replay, ledger warm, admin console.

Endpoints only. Shared logic (the admin token gate, dispatch, ops queries)
stays in FareBeep.main and is looked up at request time through `_m`.
main.py mounts this router after all of its helpers are defined.
"""
import logging
from sqlalchemy import func

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from FareBeep import main as _m
from FareBeep.models import BookingSession, DeliveryReceipt, ProcessedMessage, Subscription, utcnow

logger = logging.getLogger("farebeep.main")
router = APIRouter()


@router.get("/admin/ops")
def admin_ops(request: Request):
    """One ops snapshot: inbound dispatch health, dead letters, outbound
    receipts, beeps, bookings, and which brain answered. Read-only."""
    if not _m._admin_ok(request):
        return Response(status_code=404)
    db = _m.SessionLocal()
    try:
        inbound = dict(db.query(
            ProcessedMessage.status, func.count(ProcessedMessage.message_id)
        ).group_by(ProcessedMessage.status).all())
        from datetime import timedelta
        stale_queued = db.query(ProcessedMessage).filter(
            ProcessedMessage.status == "queued",
            ProcessedMessage.created_at < utcnow() - timedelta(minutes=5)
        ).count()
        dead_rows = db.query(ProcessedMessage).filter(
            ProcessedMessage.status == "failed"
        ).order_by(ProcessedMessage.created_at.desc()).limit(20).all()
        dead = [{
            "message_id": r.message_id,
            "phone": r.phone,
            "attempts": r.attempts,
            "last_error": (r.last_error or "")[:300],
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "replayable": bool((r.payload or {}).get("message_id")),
        } for r in dead_rows]
        receipts = dict(db.query(
            DeliveryReceipt.status, func.count(DeliveryReceipt.message_id)
        ).group_by(DeliveryReceipt.status).all())
        subs = db.query(Subscription).all()
        bookings = db.query(BookingSession.status,
                            func.count(BookingSession.id)
                            ).group_by(BookingSession.status).all()
        return {
            "inbound": {"by_status": inbound, "stale_queued_5m": stale_queued},
            "dead_letters_recent": dead,
            "delivery_receipts": {"by_status": receipts},
            "beeps": {
                "total": len(subs),
                "active": sum(1 for s in subs if not s.paused),
                "paused": sum(1 for s in subs if s.paused),
            },
            "bookings": {str(getattr(k, "value", k)): v for k, v in bookings},
            "support": {"open_threads": _m._ops_support_rows(db)},
            "ledger": _m._ops_ledger(db),
            "agent_this_process": dict(_m._agent_stats),
        }
    finally:
        db.close()


@router.post("/admin/ops/dead-letters/{message_id}/replay")
def admin_replay_dead_letter(message_id: str, request: Request):
    """Re-dispatch one dead-lettered inbound from its stored payload,
    through the exact recovery path (per-phone lock, single turn).
    Synchronous on purpose: the caller sees the new terminal state."""
    if not _m._admin_ok(request):
        return Response(status_code=404)
    db = _m.SessionLocal()
    try:
        row = db.query(ProcessedMessage).filter(
            ProcessedMessage.message_id == message_id).first()
        if row is None or row.status != "failed":
            return JSONResponse({"replayed": False,
                                 "reason": "not a dead letter"}, status_code=404)
        payload = row.payload or {}
        if not payload.get("message_id"):
            return JSONResponse({"replayed": False,
                                 "reason": "no stored payload (pre-recovery row)"})
        m = _m._restore_message(payload)
        phone = m.from_number or row.phone or ""
        if not phone:
            return JSONResponse({"replayed": False,
                                 "reason": "no phone on payload"})
        with _m._phone_lock(phone):
            try:
                _m._dispatch_single(m)
                _m._mark_processed(m.message_id, "done")
                return {"replayed": True, "message_id": m.message_id}
            except Exception as e:
                _m._mark_processed(m.message_id, "failed", error=f"replay: {e}")
                return JSONResponse({"replayed": False,
                                     "reason": str(e)[:300]})
    finally:
        db.close()


@router.post("/admin/ops/ledger/warm-now")
def admin_warm_ledger_now(request: Request):
    """Trigger a ledger warm cycle immediately (bypasses the 15-minute
    interval gate - the worker's scheduled cycle still runs as usual).
    Synchronous like replay: the caller sees the real counts. 247travels
    login failure surfaces as warmed=0 + error, not a 500."""
    if not _m._admin_ok(request):
        return Response(status_code=404)
    import FareBeep.warmer as warmer
    try:
        stats = warmer.warm_ledger()
        if stats.get("skipped_test_mode"):
            return JSONResponse({"ok": False,
                                 "reason": "TRAVELS247_MODE=test: demo "
                                 "credentials must not cache fares",
                                 **stats}, status_code=409)
        stats["ok"] = stats.get("errors", 0) < stats.get("routes", 1) * len(stats.get("dates", []))
        return stats
    except Exception as e:
        logger.exception("Warm-now failed")
        return JSONResponse({"ok": False, "error": str(e)[:300]},
                            status_code=500)


@router.get("/admin", include_in_schema=False)
def admin_cockpit():
    """The ops cockpit - a browser UI over /admin/ops + dead-letter
    replay. Only the SHELL is served here (when ADMIN_TOKEN is set);
    the data still demands the X-Admin-Token header via fetch, so the
    token never rides a URL or a log. Unset token = closed surface
    (404, like every admin route)."""
    if not _m.ADMIN_TOKEN:
        return Response(status_code=404)
    return FileResponse(_m.WEB_DIR / "admin.html",
                        headers={"Cache-Control": "no-store"})
