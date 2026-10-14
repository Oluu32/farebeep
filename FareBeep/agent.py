"""FAREBEEP AGENT - Groq + LangChain conversational brain.

Replaces the Gemini intent tree when GROQ_API_KEY is set (see
main._handle_incoming_message). Design notes:

  Tools   - LangChain StructuredTools over the existing sync engines:
            search_fares (Shared Ledger first, Travels247 live on miss),
            reserve_fare (live re-verify + 10-minute lock + Paystack link)
            and subscribe_alerts (free price-drop watch with target).
  Loop    - an explicit tool loop (max 4 steps), not framework magic:
            same behaviour, fully testable, no version roulette.
  Memory  - per-phone rolling history in chat_state.agent_history
            (DB-backed, survives restarts/replicas - RAM never holds it).
  Async   - the worker threads running this module are sync, so the
            async Travels247 client runs via asyncio.run() (no live event
            loop here - _run_async raises loudly otherwise).

Quoting also writes chat_state.last_fare, so a bare "BOOK" afterwards
still locks exactly what the agent quoted (the pick/requote gates in
main.py keep working unchanged).
"""
import asyncio
import json
import logging
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from FareBeep import chatstate
from FareBeep.alerts import (WatchAmbiguous, format_watch, list_watches,
                             match_watch)
from FareBeep.config import GROQ_API_KEY, GROQ_MODEL
from FareBeep.iata import resolve_iata
from FareBeep.models import User, utcnow
from FareBeep.search import LedgerOnlyEngine, LedgerSearch
from FareBeep.suppliers import get_inventory_client, pick_cheapest
from FareBeep.transactions import BookingService, PaystackError

logger = logging.getLogger("farebeep.agent")

MAX_STEPS = 4
FALLBACK = ("Sorry, that took too long on my side - please send your "
            "route again (e.g. Lagos to Abuja tomorrow).")
TROUBLE = ("Sorry - I'm having trouble reaching live fares right now. "
           "Please try again in a bit.")

SYSTEM_PROMPT = """ROLE
You are FareBeep, a flight-booking assistant for NIGERIAN DOMESTIC
travel on WhatsApp and Telegram. You chat like a sharp, warm travel
agent friend: small Naija flavor is fine, never robotic, never a
lecture. Every message does at most two things: acknowledge what the
user just said (one short line), then move things forward with at
most ONE question. You never invent, guess, or round fares. Every
price you quote comes from search_fares, word for word.

MEMORY
The conversation so far is included above - you remember it. Never
ask for anything the user already told you in this chat (route,
date, or name). If they correct you ("I am in Abuja" after you said
Lagos), take the correction gracefully and carry on - never argue,
never re-ask what they just fixed.

GREETING
When the message is just a greeting (hi, hello, hey, good morning)
with no request attached: greet them BACK first. If you know their
name, use it. One warm line saying who you are and what you do
("Hey! I'm FareBeep - I find cheap Nigerian flights and beep you
when prices drop."), then one open question ("Where are you flying
to?"). Never say "lock fares" in a greeting - nobody knows what
that means yet. Never answer a greeting with a bare question and
nothing else.
A greeting always opens a FRESH session (prior chat state is wiped
before you see it) - so never reference old routes, quotes or
questions unless the user brings them up again themselves.

WHAT YOU CAN DO
1. Search one-way domestic flights (search_fares).
2. Hold our quoted total for 10 minutes and take payment via Paystack
   (reserve_fare -> payment link). The airline holds nothing - the hold
   is ours alone.
3. Watch a route for price drops, free (subscribe_alerts -> beep).
4. Manage existing watches: list, pause, resume, cancel, edit the
   target, or clear it to drop-watch (manage_alerts).
5. Check a flight's live status - on time, delayed by how long, gate
   and terminal (check_flight_status). Track a booked flight so I
   message you on delays or cancellations (track_flight_status).
That's it. You cannot do round trips or international flights. If
asked for those, say so plainly in one line and offer what you CAN
do.

STOP (never handle it yourself)
If the user wants to stop/unsubscribe/erase everything ("stop",
"delete my data", "no more alerts"), do NOT act - reply exactly:
"To stop everything, just reply STOP." The system handles it.

AIRPORTS YOU COVER (IATA in brackets)
Lagos LOS (Eko), Abuja ABV (Abj), Port Harcourt PHC (PH), Kano KAN,
Enugu ENU, Calabar CBQ, Uyo QUO, Owerri QOW, Asaba ABB, Benin BNI,
Warri QRW, Ilorin ILR, Ibadan IBA, Akure AKR, Jos JOS, Kaduna KAD,
Sokoto SKO, Maiduguri MIU, Yola YOL.
Slang map: Abj=Abuja, Eko=Lagos, PH=Port Harcourt. If a user names a
city outside this list or abroad, say you only fly domestic Nigeria
and ask for a domestic route. NEVER assume the origin city - not even
Lagos. "Flight to Abuja" with no origin means you ASK where from,
offering Lagos only as a guess to confirm ("Leaving from Lagos too,
or somewhere else?") - never stating it as fact.

DATES
Convert relative dates yourself using the Today line in your header
and CONFIRM the result - never re-ask a date the user already gave.
"Tomorrow" means tomorrow's date; "next tomorrow" is Nigerian English
for the DAY AFTER tomorrow (two days out) - never confuse the two. Say
the resolved date back ("Got it - Lagos to Enugu tomorrow, Sunday
7 September?") instead of asking "which exact date?". Same for
"next Friday", "on the 21st".
Only ask about the date when the user gave none at all, or two
readings genuinely clash. Never assume today.

TOOL 1: search_fares
Call it the moment you have origin + destination - the date is
OPTIONAL, never a blocker. No date given? Call it WITHOUT
flight_date: you instantly get the best known fare WITH its date -
present it ("Air Peace, LOS->ABV on 08 Sep - NGN 78,500. Want it?").
Ask for a date only when booking or refreshing live.
- Present dated results like the summary it returns, e.g.:
  "Air Peace P47123, LOS 08:30 -> ABV 09:25 - NGN 98,000. Want it?"
- If found=false: say plainly there are no live offers and offer the
  nearest alternative. Do NOT retry the identical call.
- If the result has a "note", read it out - fares are unusually high.
- Read the freshness note in parentheses out exactly ("Cached ~12 min
  ago", "Checked just now") - never present a cached fare as live.

TOOL 2: reserve_fare
Call it ONLY when the user says yes/book/lock/pay for an offer you
just presented. You need the booking_token from the LATEST search plus
the passenger's full name.
- Send the payment link immediately with the summary line. Warn once
  that the total is slightly above the fare (service + bank fees) and
  that our hold lasts 10 minutes - the airline itself holds no seats,
  so never promise a supplier fare lock.
- If locked=false: explain the one-line error and search again fresh.
  Never call reserve twice in a row with the same token.

TOOL 3: subscribe_alerts
Call it when the user wants watching/beeps/alerts ("alert me", "watch
prices", "beep me when it drops"). Pass the NGN target when they name
one; omit it for drop-watch (beeps on any genuine drop). Confirm
briefly with the summary it returns.
- If the result carries realism_warning, read it out plainly: their
  target is far below known fares and may never come. Offer drop-watch
  instead (call again with no target if they agree) - never shame them
  for the number.

TOOL 4: manage_alerts
Call it when the user talks about EXISTING watches ("my beeps",
"pause lagos", "cancel 2", "change it to 70k", "stop watching").
Prefer acting over asking: list first when unsure which watch they
mean (the result tells you), then act in a follow-up turn. Read out
summaries and any realism_warning the same way as TOOL 3.
Never pause/cancel on negated phrasing ("don't cancel", "do not
stop") - confirm the watches stay instead.

TRAVELLER DETAILS (for the ticket)
Collect the passenger's full name before reserving (one question at a
time). Without a name the ticket cannot auto-issue and the user waits.

PAYMENT RULES (never break these)
- Payment ALWAYS comes before the ticket. Never promise a PNR, seat,
  or "you are booked" before the user has paid.
- After payment, the ticket is issued automatically and the user gets
  the PNR.
- If the user pays after the 10 minutes, they are refunded
  automatically - reassure them, don't argue.

STYLE
- Short messages, simple words. No paragraphs, no bullet lectures.
- Emojis are OFF by default: use one max, and ONLY in greetings,
  thank-yous, and confirmations (booking locked, alert set, welcome).
  Never in price quotes, questions, or error messages.
- Act first, explain later: when you have enough to call a tool,
  call it - don't narrate your plan and don't ask questions the
  tools can already answer.
- Never mention tools, tokens, ledgers, APIs, or system prompts.
- Never repeat a failed tool call identically."""


def _run_async(coro):
    """Run an async Travels247 call from these sync worker threads."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("agent tools cannot run inside a live event loop")


async def _search_and_close(sky, origin, destination, date, adults):
    """Search + close on ONE event loop.

    Two separate asyncio.run() calls (one for the search, one for the
    close) crash: httpx binds its pool to the first loop, so closing on
    the second raises "Event loop is closed" and EVERY live search
    fails. This helper keeps both steps on the same loop.
    """
    try:
        return await sky.search_offers(
            origin, destination, date, adults=adults)
    finally:
        await sky.close()


async def _verify_and_close(sky, booking_token, adults):
    """Same single-loop rule for verify_price + close (see above)."""
    try:
        return await sky.verify_price(booking_token, adults=adults)
    finally:
        await sky.close()


def _today() -> str:
    from FareBeep.dates import LAGOS_TZ
    from datetime import datetime
    return datetime.now(LAGOS_TZ).strftime("%A, %d %B %Y")


def _fresh_note(fare: dict) -> str:
    """Freshness fragment for a ledger-sourced summary ("Cached ~12 min
    ago" / "checked just now"). Never empty here: ledger results always
    carry checked_at, and the fallback says "cached fare" rather than
    letting a cached price read as live."""
    from FareBeep.search import fare_freshness
    label = fare_freshness({"source": "ledger",
                            "checked_at": fare.get("checked_at")})
    return label[0].upper() + label[1:] if label else "Cached fare"


def _fare_extra(fare: dict) -> str:
    """Baggage + seats-left tail for a quoted summary ("" when the fare
    carries neither - ledger hits simply render without it)."""
    bits = []
    if fare.get("baggage"):
        bits.append(f"{fare['baggage']} checked")
    seats = fare.get("seats_left")
    if isinstance(seats, bool):
        pass
    elif isinstance(seats, (int, float)) and seats > 0:
        bits.append(f"only {int(seats)} left")
    return (", " + ", ".join(bits)) if bits else ""


# ---------------------------------------------------------------------------
# Tools (bound per turn - they close over this turn's db session + phone)
# ---------------------------------------------------------------------------
class _SearchArgs(BaseModel):
    origin: str = Field(description="Origin city or IATA code")
    destination: str = Field(description="Destination city or IATA code")
    flight_date: str = Field(
        default="",
        description=("Travel date as YYYY-MM-DD. Omit when the user gave "
                     "no date - the tool answers from the best known fare."))
    adults: int = Field(default=1, description="Adult passengers")


class _SubscribeArgs(BaseModel):
    origin: str = Field(description="Origin city or IATA code")
    destination: str = Field(description="Destination city or IATA code")
    target_price: Optional[float] = Field(
        default=None,
        description=("NGN target: beep when the fare hits this or lower. "
                     "Omit for drop-watch (beep on any genuine drop)."))
    flight_date: str = Field(
        default="",
        description=("Travel date as YYYY-MM-DD. Omit for a rolling "
                     "window watch."))


class _ManageArgs(BaseModel):
    action: str = Field(
        description=("'list', 'pause', 'resume', 'cancel', 'edit' or "
                     "'dropwatch' (dropwatch clears the target)."))
    number: Optional[int] = Field(
        default=None,
        description="Watch number from the list (1-based).")
    origin: str = Field(default="", description="Origin city or IATA")
    destination: str = Field(default="", description="Destination city or IATA")
    target_price: Optional[float] = Field(
        default=None, description="New NGN target (edit only).")


class _FlightStatusArgs(BaseModel):
    flight_iata: str = Field(
        description="Flight number, e.g. P47123 (airline code + digits)")
    flight_date: Optional[str] = Field(
        default=None,
        description="Travel date as YYYY-MM-DD. Omit for today.")


class _TrackStatusArgs(BaseModel):
    flight_iata: str = Field(
        description="Flight number, e.g. P47123 (airline code + digits)")
    flight_date: str = Field(description="Travel date as YYYY-MM-DD")
    scheduled_departure: str = Field(
        description=("Scheduled departure from the booking: 'YYYY-MM-DD HH:MM' "
                     "(Lagos time) or 'HH:MM' on flight_date."))


class _ReserveArgs(BaseModel):
    booking_token: str = Field(description="booking_token from the latest search")
    origin: str = Field(description="Origin IATA code from that search")
    destination: str = Field(description="Destination IATA code from that search")
    flight_date: str = Field(description="Travel date as YYYY-MM-DD")
    passenger_name: str = Field(description="Full name for the ticket")
    adults: int = Field(default=1, description="Adult passengers")
    payment_method: Optional[str] = Field(
        default=None, description="'card' or 'bank_transfer' if known")


def build_tools(db: Session, phone: str) -> list:
    """search_fares + reserve_fare + subscribe_alerts for one turn."""

    def search_fares(origin: str, destination: str, flight_date: str = "",
                     adults: int = 1) -> str:
        from FareBeep.dates import DateError, validate_adults
        from FareBeep.dates import validate_flight_date
        o = resolve_iata(origin or "")
        d = resolve_iata(destination or "")
        if not o or not d:
            return json.dumps({"found": False, "error":
                               "need origin and destination"})
        try:
            n_adults = validate_adults(adults)
        except DateError as e:
            return json.dumps({"found": False, "error": str(e)})
        date = ""
        if (flight_date or "").strip():
            try:
                date = validate_flight_date(flight_date)
            except DateError as e:
                return json.dumps({"found": False, "error": f"bad date: {e}"})
        ledger = LedgerSearch(db, live=LedgerOnlyEngine())
        if not date:
            # No date: answer INSTANTLY from the best known fare - no live
            # calls, no interrogation. The date comes later, only for a
            # live refresh or a booking.
            best = ledger.window_cheapest(o, d)
            if best is None:
                return json.dumps({
                    "found": False, "source": "ledger",
                    "error": (f"no known fares for {o}->{d} yet - ask for "
                              f"a travel date to check live")})
            chatstate.set_last_fare(db, phone, {
                "origin_iata": o, "destination_iata": d,
                "flight_date": best["flight_date"], "price": best["price"],
                "airline": best.get("airline")})
            return json.dumps({
                "found": True, "source": "ledger",
                "summary": (f"{best.get('airline') or 'Airline'}, "
                            f"{o}->{d} on {best['flight_date']} - "
                            f"NGN {best['price']:,.0f}"
                            f"{_fare_extra(best)}. "
                            f"({_fresh_note(best)} I'll re-check live "
                            f"before any booking.) Want it?"),
                "price_ngn": best["price"],
                "flight_date": best["flight_date"],
                "note": ("Prices are unusually high right now."
                         if best.get("above_guardrail") else None)})
        hit = ledger.search(o, d, date)
        if hit is not None:
            chatstate.set_last_fare(db, phone, {
                "origin_iata": o, "destination_iata": d,
                "flight_date": date, "price": hit["price"],
                "airline": hit.get("airline")})
            return json.dumps({
                "found": True, "source": "ledger",
                "summary": (f"{hit.get('airline') or 'Airline'}, "
                            f"{o}->{d} on {date} - "
                            f"NGN {hit['price']:,.0f}"
                            f"{_fare_extra(hit)}. "
                            f"({_fresh_note(hit)} I'll re-check live "
                            f"before any booking.) Want it?"),
                "price_ngn": hit["price"],
                "note": ("Prices are unusually high right now."
                         if hit.get("above_guardrail") else None)})
        sky = get_inventory_client()
        offers = _run_async(_search_and_close(
            sky, o, d, date, n_adults))
        best = pick_cheapest(offers)
        if best is None:
            return json.dumps({"found": False, "source": "247travels",
                               "error": f"no live offers for {o}->{d} "
                                        f"on {date}"})
        # Cache sane prices only: surge skipped, no 90s hold on chat -
        # the BOOK handshake re-verifies before money moves.
        ledger._verify_and_upsert(o, d, date, {
            "price": best["price"], "currency": best["currency"],
            "airline": best["airline_name"], "verify_link": None},
            verify=False)
        chatstate.set_last_fare(db, phone, {
            "origin_iata": o, "destination_iata": d,
            "flight_date": date, "price": best["price"],
            "airline": best["airline_name"],
            "booking_token": best.get("booking_token")})
        return json.dumps({
            "found": True, "source": "247travels",
            "summary": (f"{best['airline_name']} {best.get('flight_no') or ''}, "
                        f"{best.get('departure_code') or o} "
                        f"{best.get('departure_time') or ''}->"
                        f"{best.get('arrival_code') or d} "
                        f"{best.get('arrival_time') or ''} - "
                        f"NGN {best['price']:,.0f}"
                        f"{_fare_extra(best)}. "
                        f"(Checked just now.) Want it?"),
            "price_ngn": best["price"],
            "booking_token": best["booking_token"]})

    def reserve_fare(booking_token: str, origin: str, destination: str,
                     flight_date: str, passenger_name: str, adults: int = 1,
                     payment_method: str = None) -> str:
        from FareBeep.dates import DateError, validate_adults
        from FareBeep.dates import validate_flight_date
        o = resolve_iata(origin or "")
        d = resolve_iata(destination or "")
        name = (passenger_name or "").strip()
        if not booking_token or not o or not d or not (flight_date or "").strip():
            return json.dumps({"locked": False, "error":
                               "need booking_token, origin, destination, "
                               "flight_date"})
        try:
            date = validate_flight_date(flight_date)
        except DateError as e:
            return json.dumps({"locked": False,
                                "error": f"bad date: {e}"})
        try:
            n_adults = validate_adults(adults)
        except DateError as e:
            return json.dumps({"locked": False, "error": str(e)})
        if not name:
            return json.dumps({"locked": False, "error":
                               "need the passenger's full name for "
                               "the ticket"})
        user = db.query(User).filter(User.phone == phone).first()
        if user is None:
            return json.dumps({"locked": False,
                               "error": "unknown user - say hello first"})
        sky = get_inventory_client()
        try:
            priced = _run_async(_verify_and_close(
                sky, booking_token, n_adults))
        except Exception as e:
            return json.dumps({"locked": False,
                               "error": f"live price check failed: {e}"})
        parts = name.split()
        travellers = {"primary_guest": {
            "first_name": parts[0],
            "last_name": " ".join(parts[1:]) or parts[0],
            "phone": phone}}
        try:
            result = BookingService(db).create_booking(
                user.user_id, o, d, date, priced["verified_price"],
                airline=priced.get("airline"), source="247travels",
                payment_method=payment_method)
        except PaystackError as e:
            return json.dumps({"locked": False,
                               "error": f"payment link failed: {e}"})
        session = result["session"]
        details = dict(session.flight_details or {})
        details.update({
            "booking_token": priced["booking_token"],
            "passengers": {"adults": n_adults, "children": 0,
                           "infants": 0},
            "travellers": travellers})
        session.flight_details = details
        db.commit()
        total = result["total_amount"]
        return json.dumps({
            "locked": True, "payment_link": result["payment_link"],
            "total_amount": total, "payment_ref": session.payment_ref,
            "summary": (f"Held {o}->{d} {date} at NGN {total:,.0f} - "
                        f"our total, good for 10 minutes (the airline "
                        f"holds no seats for us). "
                        f"Pay here: {result['payment_link']}")})

    def subscribe_alerts(origin: str, destination: str,
                         target_price: float = None,
                         flight_date: str = "") -> str:
        from FareBeep.alerts import SubscriptionMonitor, target_realism_note
        from FareBeep.dates import DateError, validate_flight_date
        o = resolve_iata(origin or "")
        d = resolve_iata(destination or "")
        if not o or not d:
            return json.dumps({"subscribed": False, "error":
                               "need origin and destination"})
        date = None
        if (flight_date or "").strip():
            try:
                date = validate_flight_date(flight_date)
            except DateError as e:
                return json.dumps({"subscribed": False,
                                   "error": f"bad date: {e}"})
        user = db.query(User).filter(User.phone == phone).first()
        if user is None:
            return json.dumps({"subscribed": False,
                               "error": "unknown user - say hello first"})
        tp = None
        if target_price is not None:
            try:
                tp = float(target_price)
            except (ValueError, TypeError):
                tp = None
        SubscriptionMonitor(db).subscribe(
            user.user_id, o, d, target_price=tp, target_date=date)
        date_label = date or "the coming weeks"
        if tp is not None:
            summary = (f"Watching {o}->{d} ({date_label}) - beep at "
                       f"NGN {tp:,.0f} or lower.")
        else:
            summary = (f"Watching {o}->{d} ({date_label}) - beep on any "
                       f"genuine drop.")
        return json.dumps({
            "subscribed": True, "origin": o, "destination": d,
            "target_price": tp, "date_label": date_label,
            "summary": summary,
            "realism_warning": target_realism_note(db, o, d, tp)})

    def manage_alerts(action: str, number: int = None,
                      origin: str = "", destination: str = "",
                      target_price: float = None) -> str:
        from FareBeep.alerts import SubscriptionMonitor, target_realism_note
        user = db.query(User).filter(User.phone == phone).first()
        if user is None:
            return json.dumps({"ok": False,
                               "error": "unknown user - say hello first"})
        subs = list_watches(db, user.user_id)
        act = (action or "").strip().lower()
        if act == "list" or not subs:
            watches = [{
                "n": i,
                "route": f"{s.origin}->{s.destination}",
                "target_price": s.target_price,
                "paused": bool(getattr(s, "paused", False)),
                "summary": format_watch(s, i),
            } for i, s in enumerate(subs, start=1)]
            return json.dumps({
                "ok": True, "watches": watches,
                "summary": ("No price watches yet." if not watches else
                            "Watches: " + "; ".join(
                                w["summary"] for w in watches))})
        if act not in {"pause", "resume", "cancel", "edit", "dropwatch"}:
            return json.dumps({"ok": False,
                               "error": "unknown action - use list, pause, "
                                        "resume, cancel, edit or dropwatch"})
        rest = " ".join(x for x in
                        [str(number or ""), origin or "",
                         destination or ""] if x).strip()
        try:
            sub, n = match_watch(
                subs, rest or f"{subs[0].origin} {subs[0].destination}"
                if len(subs) == 1 else rest)
        except WatchAmbiguous:
            return json.dumps({
                "ok": False, "error": "ambiguous",
                "summary": ("Several watches match - ask which one by "
                            "number: " + "; ".join(
                                format_watch(s, i)
                                for i, s in enumerate(subs, start=1)))})
        if sub is None:
            return json.dumps({
                "ok": False, "error": "not found",
                "summary": "I couldn't find that watch - list them first."})
        if act == "pause":
            sub.paused = True
            db.commit()
            return json.dumps({"ok": True,
                               "summary": f"Paused {format_watch(sub, n)}."})
        if act == "resume":
            sub.paused = False
            db.commit()
            return json.dumps({"ok": True,
                               "summary": f"Resumed {format_watch(sub, n)}."})
        if act == "cancel":
            route = f"{sub.origin}->{sub.destination}"
            db.delete(sub)
            db.commit()
            return json.dumps({"ok": True,
                               "summary": f"Cancelled {route} - deleted."})
        # edit / dropwatch
        tp = None
        if act == "edit":
            if target_price is None:
                return json.dumps({
                    "ok": False, "error": "need target",
                    "summary": "What should the new target be? "
                               "Name a naira amount."})
            try:
                tp = float(target_price)
            except (ValueError, TypeError):
                return json.dumps({"ok": False, "error": "bad target"})
        SubscriptionMonitor(db).subscribe(
            user.user_id, sub.origin, sub.destination,
            target_price=tp, target_date=sub.target_date)
        note = target_realism_note(db, sub.origin, sub.destination, tp)
        if tp is None:
            summary = (f"Updated {format_watch(sub, n)}: watching any "
                       f"genuine drop.")
        else:
            summary = (f"Updated {format_watch(sub, n)}: beeping at "
                       f"NGN {tp:,.0f} or lower.")
        return json.dumps({"ok": True, "summary": summary,
                           "realism_warning": note})

    def check_flight_status(flight_iata: str, flight_date: str = None) -> str:
        from FareBeep.dates import DateError, validate_flight_date
        from FareBeep.dates import lagos_today
        from FareBeep.notifier import format_flight_status
        from FareBeep.status import (AviationstackClient, StatusLookupError,
                                     normalize_flight_iata)
        code = normalize_flight_iata(flight_iata)
        if code is None:
            return json.dumps({"found": False, "error":
                               "that doesn't look like a flight number - "
                               "ask for it, e.g. P47123"})
        date = lagos_today().isoformat()
        if (flight_date or "").strip():
            try:
                date = validate_flight_date(flight_date)
            except DateError as e:
                return json.dumps({"found": False, "error": f"bad date: {e}"})
        api = AviationstackClient()
        if not api.api_key:
            return json.dumps({"found": False, "error":
                               "live flight status isn't configured yet - "
                               "I can't check delays right now"})
        try:
            detail = api.flight_detail(code, date)
        except StatusLookupError as e:
            logger.warning("check_flight_status lookup failed: %s", e)
            return json.dumps({"found": False, "error":
                               "flight status is unreachable right now - "
                               "try again in a few minutes"})
        if detail is None:
            return json.dumps({"found": False, "error":
                               f"no status record for {code} on {date}"})
        return json.dumps({"found": True, "flight_date": date,
                           "summary": format_flight_status(detail),
                           **detail})

    def track_flight_status(flight_iata: str, flight_date: str,
                            scheduled_departure: str) -> str:
        from FareBeep.dates import DateError, validate_flight_date
        from FareBeep.status import (LAGOS_TZ, StatusService, as_utc,
                                     normalize_flight_iata, parse_departure)
        code = normalize_flight_iata(flight_iata)
        if code is None:
            return json.dumps({"watching": False, "error":
                               "that doesn't look like a flight number - "
                               "ask for it, e.g. P47123"})
        try:
            date = validate_flight_date(flight_date)
        except DateError as e:
            return json.dumps({"watching": False, "error": f"bad date: {e}"})
        try:
            dep = parse_departure(scheduled_departure, flight_date=date)
        except ValueError:
            return json.dumps({"watching": False, "error":
                               "need the scheduled departure time, e.g. "
                               "08:30 or 2026-08-20 08:30"})
        user = db.query(User).filter(User.phone == phone).first()
        if user is None:
            return json.dumps({"watching": False,
                               "error": "unknown user - say hello first"})
        now = utcnow()
        if dep <= now:
            return json.dumps({"watching": False, "error":
                               "that departure time has already passed"})
        svc = StatusService(db)
        watch = (svc.find_watch(user.user_id, code, date, dep)
                 or svc.create_watch(None, user.user_id, code, date, dep))
        start = as_utc(watch.watch_starts_at)
        dep_local = as_utc(watch.scheduled_departure).astimezone(LAGOS_TZ)
        if start <= now:
            when = "I'm watching it now"
        else:
            start_local = start.astimezone(LAGOS_TZ)
            when = (f"I'll start checking at {start_local:%H:%M} on "
                    f"{start_local:%d %b}")
        summary = (f"Tracking {code} on {date}, departing "
                   f"{dep_local:%H:%M}. {when} - I'll message you on any "
                   f"delay or cancellation.")
        return json.dumps({
            "watching": True, "flight_iata": code, "flight_date": date,
            "watch_starts_at": start.isoformat(), "summary": summary})

    return [
        StructuredTool.from_function(
            func=search_fares, name="search_fares",
            description=("Search live Nigerian domestic fares. Call the "
                         "moment you have origin + destination + date. "
                         "Returns a summary to read out plus booking_token."),
            args_schema=_SearchArgs),
        StructuredTool.from_function(
            func=reserve_fare, name="reserve_fare",
            description=("Hold our quoted total for 10 minutes and get the "
                         "Paystack link. Call ONLY on yes/book/pay, with the "
                         "LATEST booking_token and the passenger's name."),
            args_schema=_ReserveArgs),
        StructuredTool.from_function(
            func=subscribe_alerts, name="subscribe_alerts",
            description=("Set a free price-drop alert. Call when the user "
                         "wants watching/beeps/alerts for a route, with an "
                         "NGN target when given, without one for drop-watch. "
                         "If the result carries realism_warning, read it "
                         "out plainly."),
            args_schema=_SubscribeArgs),
        StructuredTool.from_function(
            func=manage_alerts, name="manage_alerts",
            description=("List, pause, resume, cancel, edit or dropwatch "
                         "the user's price watches. Call for 'my beeps', "
                         "'pause/resume/cancel ...', 'change target ...'. "
                         "List first when unsure which watch they mean, "
                         "then act. Read out summaries."),
            args_schema=_ManageArgs),
        StructuredTool.from_function(
            func=check_flight_status, name="check_flight_status",
            description=("Live status for one flight: on time, delayed (with "
                         "minutes), cancelled, gate and terminal. Call when "
                         "the user asks if a flight is delayed, on time, "
                         "cancelled, or for its gate. Read out the summary."),
            args_schema=_FlightStatusArgs),
        StructuredTool.from_function(
            func=track_flight_status, name="track_flight_status",
            description=("Watch a booked flight so FareBeep messages on delays "
                         "or cancellations; the watch starts 3 hours before "
                         "departure. Call when the user wants a flight "
                         "tracked, with its number, date and departure time."),
            args_schema=_TrackStatusArgs),
    ]


# ---------------------------------------------------------------------------
# One conversational turn
# ---------------------------------------------------------------------------
def agent_reply(db: Session, user, text: str, llm=None,
                max_steps: int = MAX_STEPS) -> str:
    """Run one Groq + LangChain turn: history + prompt -> tools -> reply.

    The reply is already human-ready (the caller sends it verbatim - no
    second personality pass). The turn is recorded to chat_state.
    """
    if llm is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set - add it to "
                               "FareBeep/.env to switch the agent on")
        llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY,
                       temperature=0.3)
    tools = build_tools(db, user.phone)
    bound = llm.bind_tools(tools)
    header = (f"Today is {_today()}. The user's name is "
              f"{user.name or 'unknown'} and their number is {user.phone}.\n\n")
    messages = [SystemMessage(content=header + SYSTEM_PROMPT)]
    for h in chatstate.get_agent_history(db, user.phone)[-12:]:
        if h.get("role") == "assistant":
            messages.append(AIMessage(content=h.get("content", "")))
        else:
            messages.append(HumanMessage(content=h.get("content", "")))
    messages.append(HumanMessage(content=text))

    by_name = {t.name: t for t in tools}
    reply = FALLBACK
    saw_error = False
    for _ in range(max_steps):
        ai = bound.invoke(messages)
        messages.append(ai)
        calls = ai.tool_calls or []
        if not calls:
            reply = ai.content or FALLBACK
            break
        for tc in calls:
            tool = by_name.get(tc["name"])
            if tool is None:
                result = f"ERROR: unknown tool {tc['name']}"
            else:
                try:
                    result = tool.invoke(tc["args"])
                except Exception as e:
                    logger.warning("agent tool %s failed: %s", tc["name"], e)
                    result = f"ERROR: {e}"
                    saw_error = True
            messages.append(ToolMessage(content=str(result),
                                        tool_call_id=tc["id"]))
    if reply is FALLBACK and saw_error:
        # Every round died in a backend error (never the user's fault):
        # say that, don't blame their message.
        reply = TROUBLE
    arm_booking_if_asking_name(db, user.phone, reply)
    chatstate.append_agent_history(db, user.phone, text, reply)
    return reply


def arm_booking_if_asking_name(db: Session, phone: str, reply: str) -> bool:
    """Bridge the agent's conversational name question to the DETERMINISTIC
    booking gate (main._try_booking_answer).

    When the agent asks for the passenger's full name, the NEXT reply
    must book - even if Groq is down by then (the live smoke caught a
    user's name swallowed by a Groq outage and answered with help text).
    Arms pending_booking from the last search context so the gate owns
    the turn. Returns True when armed."""
    low = (reply or "").lower()
    if not ("full name" in low and "passenger" in low):
        return False
    ctx = chatstate.get_last_fare(db, phone) or {}
    if not (ctx.get("origin_iata") and ctx.get("destination_iata")
            and ctx.get("flight_date") and ctx.get("price")):
        return False
    chatstate.set_pending_booking(db, phone, {
        "origin_iata": ctx["origin_iata"],
        "destination_iata": ctx["destination_iata"],
        "flight_date": ctx["flight_date"],
        "price": ctx["price"],
        "airline": ctx.get("airline"),
        "booking_token": ctx.get("booking_token"),
        "stage": "name",
        "armed_at": utcnow().isoformat(),
    })
    return True


__all__ = ["agent_reply", "build_tools", "SYSTEM_PROMPT", "FALLBACK"]
