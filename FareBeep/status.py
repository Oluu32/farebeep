"""THE STATUS MONITOR ("Status Beep") - Aviationstack + 3-hour watch window.

Per the reconstruction brief:

  1. Do NOT track flights 24/7.
  2. Only initiate a "Status Watch" 3 hours BEFORE the departure stored in a
     booking (watch_starts_at = scheduled_departure - 3h).
  3. If the flight status changes (e.g. -> "Delayed"), send a proactive
     WhatsApp TEMPLATE message to the user (notifier.send_template).

The worker (FareBeep/worker.py) calls `run_watch_cycle(db)` every
STATUS_POLL_SECONDS. Each cycle only queries flight_status for watches whose
window has opened - nothing else spends aviationstack credits.

The conversational agent reuses the same client for on-demand lookups
(`AviationstackClient.flight_detail`) and registers watches via
`StatusService.create_watch` / `find_watch`.
"""
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import httpx
from sqlalchemy.orm import Session

from FareBeep.config import AVIATIONSTACK_API_KEY, STATUS_WATCH_LEAD_HOURS
from FareBeep.models import StatusEvent, StatusWatch, User, utcnow

logger = logging.getLogger("farebeep.status")

# Statuses that actually matter to a passenger (per aviationstack).
NOTIFIABLE = {"delayed", "cancelled", "diverted", "landed"}

# Lagos (WAT) has no DST, so a fixed +01:00 offset is exact for domestic flights.
LAGOS_TZ = timezone(timedelta(hours=1))

# Airline IATA code (2 chars, may contain a digit, e.g. "6E") + 1-4 digits + optional suffix.
_FLIGHT_RE = re.compile(r"[A-Z0-9]{2}\d{1,4}[A-Z]?")
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")


class StatusLookupError(Exception):
    """The Aviationstack call failed (network, HTTP error, bad payload)."""


def normalize_flight_iata(value) -> Optional[str]:
    """'p4 7123' -> 'P47123'. Returns None when it is not a flight number."""
    code = re.sub(r"\s+", "", str(value or "")).upper()
    return code if _FLIGHT_RE.fullmatch(code) else None


def as_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Treat naive datetimes (SQLite drops tzinfo) as UTC and return aware UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_departure(value: str, flight_date: str = None) -> datetime:
    """Parse a scheduled departure into aware UTC.

    Accepts 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DDTHH:MM[:SS][+hh:mm]' (naive
    values are Lagos time), or a bare 'HH:MM' which is read on flight_date.
    Raises ValueError on anything else.
    """
    raw = str(value or "").strip()
    time_only = _TIME_RE.fullmatch(raw)
    if time_only:
        if not flight_date:
            raise ValueError("a bare time needs flight_date")
        hh, mm = time_only.groups()
        raw = f"{flight_date}T{int(hh):02d}:{mm}"
    raw = raw.replace(" ", "T", 1)
    dt = datetime.fromisoformat(raw)  # ValueError on junk
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LAGOS_TZ)
    return dt.astimezone(timezone.utc)


class AviationstackClient:
    """Aviationstack status client (free-plan aware).

    Finding during live verification with an actual free key:
      - `/v1/flights` with `flight_iata`      -> works
      - `flight_date` filter param            -> 403 (paid-plan only)
      - `/v1/flights_status`                  -> not on the free plan
    So we never send `flight_date`; instead each row carries its own date
    and we cross-check it against the requested date before trusting it.
    """

    def __init__(self, api_key: str = None, http_client: httpx.Client = None):
        self.api_key = api_key or AVIATIONSTACK_API_KEY
        self._http = http_client or httpx.Client(timeout=12.0)

    def _rows(self, flight_iata: str) -> list:
        """Raw `data` rows for a flight number. Raises StatusLookupError."""
        try:
            resp = self._http.get(
                "https://api.aviationstack.com/v1/flights",
                params={"access_key": self.api_key,
                        "flight_iata": flight_iata,
                        "limit": 1})
            resp.raise_for_status()
            return resp.json().get("data") or []
        except Exception as e:
            raise StatusLookupError(
                f"Aviationstack call failed for {flight_iata}: {e}") from e

    @staticmethod
    def _matching_row(rows: list, flight_iata: str, flight_date: str):
        """Free plan ignores our date filter - only trust a record dated the
        flight's departure day."""
        if not rows:
            return None
        row = rows[0]
        if (row.get("flight_date") or "") != flight_date:
            logger.info("Aviationstack: %s record is for %s, not %s - skipped",
                        flight_iata, row.get("flight_date"), flight_date)
            return None
        return row

    def flight_status(self, flight_iata: str, flight_date: str) -> Optional[str]:
        """Return the raw flight_status string (or None on failure / no match).

        Example: flight_iata="P47123", flight_date="2026-08-07"
          -> "delayed" | "scheduled" | "active" | "landed" | "cancelled" ...
        """
        if not self.api_key:
            raise RuntimeError("AVIATIONSTACK_API_KEY not set.")
        try:
            rows = self._rows(flight_iata)
        except StatusLookupError as e:
            logger.warning("%s", e)
            return None
        row = self._matching_row(rows, flight_iata, flight_date)
        if row is None:
            return None
        status = (row.get("flight_status") or "").strip().lower()
        return status or None

    def flight_detail(self, flight_iata: str, flight_date: str) -> Optional[dict]:
        """Full status for one flight on one date, or None when there is no
        matching record. Raises StatusLookupError when the call itself fails.

        Keys: flight_iata, airline, status, dep_scheduled, dep_estimated,
        dep_delay_min, dep_terminal, dep_gate, arr_scheduled, arr_estimated,
        arr_delay_min, arr_terminal, arr_gate.
        """
        if not self.api_key:
            raise RuntimeError("AVIATIONSTACK_API_KEY not set.")
        row = self._matching_row(self._rows(flight_iata), flight_iata,
                                 flight_date)
        if row is None:
            return None
        flight = row.get("flight") or {}
        airline = row.get("airline") or {}
        dep = row.get("departure") or {}
        arr = row.get("arrival") or {}
        return {
            "flight_iata": flight.get("iata") or flight_iata,
            "airline": airline.get("name"),
            "status": (row.get("flight_status") or "").strip().lower() or None,
            "dep_scheduled": dep.get("scheduled"),
            "dep_estimated": dep.get("estimated"),
            "dep_delay_min": dep.get("delay"),
            "dep_terminal": dep.get("terminal"),
            "dep_gate": dep.get("gate"),
            "arr_scheduled": arr.get("scheduled"),
            "arr_estimated": arr.get("estimated"),
            "arr_delay_min": arr.get("delay"),
            "arr_terminal": arr.get("terminal"),
            "arr_gate": arr.get("gate"),
        }


class StatusService:
    """Owns the status_watch lifecycle and the push-on-change logic."""

    def __init__(self, db: Session, api: AviationstackClient = None,
                 notifier=None, watch_lead_hours: int = None,
                 template_name: str = None, clock: Callable = None):
        self.db = db
        self.api = api or AviationstackClient()
        self.notifier = notifier
        self.watch_lead_hours = watch_lead_hours or STATUS_WATCH_LEAD_HOURS
        self.template_name = template_name
        self.clock = clock or utcnow

    # -- create -----------------------------------------------------------
    def create_watch(self, booking_id, user_id, flight_iata: str,
                     flight_date: str, scheduled_departure) -> StatusWatch:
        """Register a watch. Watch window does NOT start until 3h before."""
        dep = scheduled_departure
        if isinstance(dep, str):
            dep = datetime.fromisoformat(dep)
        watch = StatusWatch(
            booking_id=booking_id,
            user_id=user_id,
            flight_iata=flight_iata,
            flight_date=flight_date,
            scheduled_departure=dep,
            watch_starts_at=dep - timedelta(hours=self.watch_lead_hours),
        )
        self.db.add(watch)
        self.db.commit()
        return watch

    def find_watch(self, user_id, flight_iata: str, flight_date: str,
                   scheduled_departure: datetime) -> Optional[StatusWatch]:
        """An existing watch for the same user, flight, date and departure."""
        dep = as_utc(scheduled_departure)
        candidates = (self.db.query(StatusWatch)
                      .filter(StatusWatch.user_id == user_id,
                              StatusWatch.flight_iata == flight_iata,
                              StatusWatch.flight_date == flight_date)
                      .all())
        for watch in candidates:
            if as_utc(watch.scheduled_departure) == dep:
                return watch
        return None

    # -- the cycle ---------------------------------------------------------
    def run_watch_cycle(self) -> int:
        """Check every watch whose 3-hour window is OPEN right now.

        Persist: a watch's window opened but the flight already departed
        (scheduled_departure < now) is considered complete and skipped.
        Returns the number of proactive template messages sent.
        """
        now = self.clock()
        due = (self.db.query(StatusWatch)
               .filter(StatusWatch.watch_starts_at <= now)
               .filter(StatusWatch.scheduled_departure > now)
               .all())

        messages_sent = 0
        for watch in due:
            watch.initiated = True
            watch.last_checked_at = now

            status = self.api.flight_status(watch.flight_iata,
                                            watch.flight_date)
            if not status:
                continue

            previous = watch.last_status
            changed = (
                previous is None and status != "scheduled"   # first sighting is not the baseline
            ) or (previous is not None and status != previous)
            if changed:
                messages_sent += self._on_status_change(watch, status, previous)
            watch.last_status = status
        self.db.commit()
        if messages_sent:
            logger.info("Status cycle: %d proactive template(s) sent",
                        messages_sent)
        return messages_sent

    def _on_status_change(self, watch: StatusWatch, status: str,
                          previous: str) -> int:
        """Persist a status_event and push a proactive template message."""
        event = StatusEvent(watch_id=watch.id, status=status, previous=previous)
        self.db.add(event)
        self.db.flush()

        sent = 0
        # Only spend template messages on statuses the passenger must act on.
        if status in NOTIFIABLE and self.notifier is not None \
           and self.template_name:
            user = (self.db.query(User).filter(User.user_id == watch.user_id)
                    .first())
            if user is not None:
                ok = self.notifier.send_template(
                    user.phone, self.template_name,
                    body_parameters=[watch.flight_iata, status.upper()])
                sent = 1 if ok else 0
                event.template_sent = bool(ok)
        return sent
