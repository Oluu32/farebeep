"""Conversational dialog and booking flows for the FareBeep concierge.

Each function here handles one turn: a user message in, zero or more
replies out. Entry points that main still owns (the webhooks, the inbound
pipeline, taps, the support console, admin ops) are read through
FareBeep.main at call time, so tests that monkeypatch names on main still
reach the code they target. This module never imports main at load time
(main imports it).
"""
import logging
import re
import uuid

from sqlalchemy import func

from FareBeep import brain, chatstate
from FareBeep.config import RATE_LIMIT_PER_MIN
from FareBeep.iata import city_name
from FareBeep.models import (BookingSession, FareLedger, SessionStatus,
                             User, utcnow)

logger = logging.getLogger("farebeep.main")


def _main():
    """The FareBeep.main module, looked up at call time (see docstring)."""
    from FareBeep import main
    return main


_SESSION_GREETINGS = frozenset({
    "hi", "hello", "hey", "yo", "good morning", "good afternoon",
    "good evening", "hi there", "hello there", "hey there",
})


def _is_session_greeting(text: str) -> bool:
    """True when the message is just a greeting (punctuation tolerated)."""
    return (text or "").strip().lower().rstrip("!.").strip() \
        in _SESSION_GREETINGS


# Honesty prefix when the smart brain is down and the deterministic one
# answers instead: one line, no jargon, then normal handling continues.
RESTING_NOTE = ("Our smart brain is resting right now, so I'm running on "
                "simple mode - I can still check fares and set beeps if "
                "you send route + date plainly, e.g. 'Lagos to Abuja "
                "tomorrow'.\n\n")


def _ops_ledger(db) -> dict:
    """Ledger warmth for ops: coverage vs the warmer's route list and
    freshness. The flow's review screen answers from these rows, so a
    cold ledger literally shows users 'No cached fare yet'."""
    from FareBeep.warmer import WARM_ROUTES
    from datetime import timedelta
    try:
        total = db.query(FareLedger).count()
        fresh = db.query(FareLedger).filter(
            FareLedger.last_updated >= utcnow() - timedelta(hours=26)
        ).count()
        warmed_pairs = {(o, d) for o, d in WARM_ROUTES}
        covered = 0
        for o, d in db.query(FareLedger.origin, FareLedger.destination
                             ).distinct().all():
            if (str(o).upper(), str(d).upper()) in warmed_pairs:
                covered += 1
        newest = db.query(func.max(FareLedger.last_updated)).scalar()
        return {"rows": total, "fresh_24h": fresh,
                "warm_routes_covered": f"{covered}/{len(warmed_pairs)}",
                "newest_fare_at": newest.isoformat() if newest else None}
    except Exception as e:
        logger.warning("Ops ledger stats failed: %s", e)
        return {"rows": None, "error": str(e)[:120]}


# Watch helpers (list/match/format) live in alerts.py next to the
# Subscription model logic - imported lazily like the rest of main's
# alerts usage. _tap_alert keeps its own index+context resolution.


# ---------------------------------------------------------------------------
# "My tickets" - secure browser ticket lookup. The chat mints a short-lived
# HMAC-signed link; the browser (WhatsApp's in-app webview included) renders
# the user's bookings and price beeps. No cookies needed - the signed token
# IS the session, so the page must stay read-only.
# ---------------------------------------------------------------------------
_MY_TRIPS_PHRASES = frozenset({
    "my bookings", "my booking", "my tickets", "my ticket", "my trips",
    "my trip", "view tickets", "view ticket", "check tickets", "check ticket",
    "view bookings", "check bookings", "my beeps", "my alerts",
})


def _handle_my_bookings(db, user: User, text: str) -> bool:
    """Deterministic 'my tickets' turn: hand out the signed browser link.
    True = turn handled (agent and brain must not run - the agent must
    never improvise a token, the link is the user's session)."""
    flat = (text or "").strip().lower().rstrip("?!.")
    if flat not in _MY_TRIPS_PHRASES:
        return False
    token = _main()._ticket_link_token(user.phone)
    _main()._say(user.phone,
         f"🔐 Your secure bookings page (valid 15 minutes):\n"
         f"{_main().APP_BASE_URL}/tickets?t={token}\n\n"
         f"Anyone holding this link can see your bookings, so keep it "
         f"private. Ask again anytime for a fresh one.",
         user.name)
    return True


# ---------------------------------------------------------------------------
# Per-phone rate limit (anti-abuse): one chatty number must not burn Groq
# tokens and live fare searches for everyone else. Sliding 60s window,
# per process (restarts reset it - that is fine for burst control).
# Over the limit: ONE polite cooldown note per window, then silence -
# replying to every burst message rewards the burst. STOP/unsubscribe is
# NEVER throttled: opting out must always work, even mid-flood.
# ---------------------------------------------------------------------------
_RATE_LIMIT = RATE_LIMIT_PER_MIN      # module global so ops/tests can tune
_RATE_WINDOW = 60.0                   # seconds
_rate_hits: dict = {}                 # phone -> [monotonic hit timestamps]
_rate_cooled: dict = {}               # phone -> monotonic ts of last note
_COOLDOWN_TEXT = ("Easy o! 😅 You dey move fast - abeg wait small, "
                  "then try again. (Reply STOP anytime to opt out.)")


def _rate_allow(phone: str, text: str) -> bool:
    """True = process the message. False = throttled (cooldown already
    handled). Uses time.monotonic so wall-clock jumps never un-throttle
    a flooder."""
    if _main()._is_stop_request(text):
        return True
    now = _main().time.monotonic()
    hits = [t for t in _rate_hits.get(phone, [])
            if now - t < _RATE_WINDOW]
    if len(hits) >= _main()._RATE_LIMIT:
        if _main()._has_open_support(phone):
            # A live support thread bypasses the throttle (like STOP):
            # burst control must never silence a human conversation.
            hits.append(now)
            _rate_hits[phone] = hits
            return True
        _rate_hits[phone] = hits
        if now - _rate_cooled.get(phone, -_RATE_WINDOW) >= _RATE_WINDOW:
            _rate_cooled[phone] = now
            _main()._say(phone, _COOLDOWN_TEXT)
        return False
    hits.append(now)
    _rate_hits[phone] = hits
    return True


def _handle_incoming_message(phone: str, text: str) -> None:
    """PASS 2 - CONCIERGE LOGIC: intent -> ask / search / act -> reply."""
    # ADMIN CONSOLE first: /open, /done and R <phone> <text> relays are
    # consumed here - they never reach the throttle or the concierge,
    # and never create a User row for the admin phone.
    if _main()._handle_admin_console(phone, text):
        return
    if not _rate_allow(phone, text):
        return
    db = _main().SessionLocal()
    try:
        try:
            user = _get_or_create_user(db, phone)

            # GREETING RESET: sessions never end on their own, so a
            # standalone hello starts a whole new one. Without this,
            # stale quotes, ranked picks and pending follow-ups hijack
            # the NEXT route ("hi" -> "Abuja" would answer a dead
            # question from the old thread) - and the agent's rolling
            # memory would resurrect it conversationally anyway ("still
            # on Abuja->PHC?"). So EVERYTHING chat-scoped goes: quotes,
            # picks, pending items AND agent history. The User row (name,
            # identity) survives - only the conversation restarts.
            if _is_session_greeting(text):
                chatstate.clear_pending_fare(db, user.phone)
                chatstate.clear_pending_requote(db, user.phone)
                chatstate.clear_last_fare(db, user.phone)
                chatstate.clear_last_fares(db, user.phone)
                chatstate.clear_agent_history(db, user.phone)

            # PICK GATE (before the brain): a "1", "2" or "3" reply right
            # after a ranked fare list selects and locks THAT fare. Narrow:
            # only when an active list exists AND the message is the number
            # (or "number 2" / "the 2nd one"). Without a list, a bare number
            # falls through to the brain as a DATE - never intercepted.
            pick = _main()._try_pick(db, text, phone)
            if pick is not None:
                if pick == "out_of_range":
                    n = len((chatstate.get_last_fares(db, phone) or {})
                            .get("fares", []))
                    _main()._say(phone, f"I only showed {n} option"
                         + ("s" if n != 1 else "")
                         + ". Reply 1-" + str(n) + ", or ask me for a new search.",
                         user.name)
                elif pick == "unclear":
                    ctx = chatstate.get_last_fares(db, phone) or {}
                    msg = brain.compose_unclear_pick_reply(
                        text, ctx.get("fares") or [], user_name=user.name)
                    _main()._say(user.phone, msg, user.name, humanized=True)
                else:
                    _reply_booking(db, user, brain.Intent(intent="book"),
                                   picked_fare=pick)
                return

            # TICKET-DETAILS GATE (before everything else): a paid booking
            # waiting for passenger name/email - any reply is an answer,
            # never a new search.
            if _try_ticket_details_answer(db, user, text):
                return

            # BOOKING GATE (deterministic chat booking): after a live
            # re-quote was held, the user's reply is the passenger's
            # NAME - never a fresh search, never an agent turn. This is
            # the plain-text equivalent of the button Book path, immune
            # to LLM timeouts (the bug the E2E smoke caught).
            if _try_booking_answer(db, user, text):
                return

            # REQUOTE GATE: after "the price moved to ₦X - still lock it?",
            # a bare yes/book answer proceeds with the already-live-quoted
            # fare; a no/cancel politely aborts. Anything naming a city or
            # date is a fresh request - it falls through to the brain.
            if _try_requote_answer(db, user, text):
                return

            # CONTINUATION: the user answered our follow-up question with a
            # bare city ("abuja" after "where will you be flying from?") -
            # fill the missing piece and search, like a person would.
            if _fill_pending_fare(db, user, text):
                return

            # DATE CONFIRMATION answer ("5 June" after "did you mean 5
            # June or 6 May?"): the pending route is complete, only the
            # date was missing - fill it and search. A city mention or
            # an ambiguous answer falls through for fresh handling.
            if _fill_pending_date(db, user, text):
                return

            # AMBIGUOUS DATE GUARD (before the agent, like STOP/MANAGE):
            # "05/06" reads two ways and must never reach search, booking
            # - or the agent's tools - as a guessed date. Parsed locally
            # (free, deterministic) so online/offline behave identically.
            if _try_date_confirm(db, user, text):
                return

            # GROQ AGENT: when configured (and not in guided mode), the
            # LangChain agent owns the turn end-to-end (tools included)
            # and its reply goes out verbatim. On ANY agent failure
            # (quota, outage, timeout) we fall through to the
            # deterministic brain below instead of going silent - the
            # user always gets an answer.
            #
            # Just before it: STOP / MANAGE whole-message commands own
            # their turns (the agent must never improvise deletions or
            # beep edits - a wrong guess here deletes real user data).
            # They sit AFTER pick/requote/pending so a financial answer
            # ("cancel" to a price question) always wins over commands.
            guided = _main().GUIDED_MODE
            if _main()._is_stop_request(text):
                _main()._handle_stop(db, user)
                return
            if _main()._handle_manage(db, user, text):
                return
            if _handle_my_bookings(db, user, text):
                return

            # SUPPORT RELAY: an explicit human ask, a refund/payment
            # phrase, or a live thread hands the turn to the founder's
            # own chat. AFTER the financial gates (a "cancel" answer to
            # a price question must never become a ticket) and BEFORE
            # the agent - the agent must never talk a refund demand
            # down alone.
            if _main()._handle_support(db, user, text):
                return
            if _main().GROQ_API_KEY and not guided:
                from FareBeep import agent as fare_agent
                try:
                    _main()._say(phone, fare_agent.agent_reply(db, user, text),
                         user.name, humanized=True)
                    _main()._agent_stats["groq_ok"] += 1
                    _main()._note_agent_outcome(phone, ok=True)
                    return
                except Exception as e:
                    _main()._agent_stats["groq_fail"] += 1
                    _main()._note_agent_outcome(phone, ok=False)
                    logger.warning("Groq agent failed (%s) - guided "
                                   "fallback: %s", phone, e)
                    guided = True
            _main()._agent_stats["guided_turns"] += 1

            intent = brain.parse_intent(text, force_local=guided)
            logger.info("Intent=%s payload=%s phone=%s",
                        intent.intent, intent.as_dict(), phone)

            if intent.name and not user.name:
                user.name = intent.name
                db.commit()
                logger.info("Captured name %s for %s", intent.name, phone)

            if intent.intent == "fare":
                if intent.has_route and _needs_date_confirm(intent):
                    _ask_date_confirm(db, user, intent)
                elif intent.has_route:
                    chatstate.clear_pending_fare(db, user.phone)  # superseded by a full route
                    _reply_fare(db, user, intent)
                elif intent.destination_iata and intent.date and not intent.origin_iata:
                    # "Find me a flight to Abj for next tuesday" - destination
                    # + date, no origin: search from the default hub (Lagos).
                    # BUT when the destination IS the hub (Lagos), the hub
                    # cannot also be the origin - that would be Lagos->Lagos.
                    # Ask where they're flying from instead.
                    if intent.destination_iata == "LOS":
                        _ask_missing_info(db, user, intent)
                    else:
                        _reply_fare(db, user, intent)
                elif intent.is_partial:
                    _ask_missing_info(db, user, intent)
                else:
                    _main()._say(user.phone, _main()._help_text(), user.name)
            elif intent.intent == "book":
                # BOOK after a fare quote: the route/date the user discussed
                # live in the per-chat context (chat_state), so a bare "BOOK"
                # books exactly what was quoted - never silently today.
                if intent.has_route or chatstate.get_last_fare(db, user.phone) is not None:
                    if intent.has_route and _needs_date_confirm(intent):
                        _ask_date_confirm(db, user, intent)
                    else:
                        _reply_booking(db, user, intent)
                elif intent.is_partial:
                    _ask_missing_info(db, user, intent)
                else:
                    _main()._say(user.phone,
                         "To book: send your route, e.g. 'BOOK Lagos to "
                         "Abuja tomorrow' - or just say BOOK right after a "
                         "fare quote I sent you.",
                         user.name)
            elif intent.intent == "subscribe":
                if intent.has_route and _needs_date_confirm(intent):
                    _ask_date_confirm(db, user, intent)
                else:
                    _main()._reply_subscribe(db, user, intent)
            elif intent.intent == "unsubscribe":
                if _main()._looks_negated(text):
                    # "don't cancel my alerts" parsed as unsubscribe: the
                    # user is reassuring, not erasing. Confirm, delete nothing.
                    _main()._say(user.phone,
                         "Understood - your price alerts stay exactly as "
                         "they are. Anything else I can do?",
                         user.name)
                else:
                    _main()._reply_unsubscribe(db, user)
            elif intent.intent in ("status", "track"):
                _reply_status_ack(user, intent)
            else:
                msg = _main()._help_text()
                if guided:
                    msg = RESTING_NOTE + msg
                _main()._say(user.phone, msg, user.name)
        except Exception as e:
            # never let a single turn crash the webhook thread
            logger.error("Message handling failed (%s): %s", phone, e)
    finally:
        db.close()


def _say(phone: str, msg: str, user_name: str = None,
         humanized: bool = False) -> None:
    """Send a reply. By default Gemini runs a personality pass first so the
    bot sounds human; on AI failure the template goes out verbatim. When the
    message is ALREADY AI-narrated (the ranked fare list) pass humanized=True
    so it is sent verbatim - no second personality pass to garble the prices."""
    if not msg:
        _main().notifier.send_text(phone, "")
        return
    if humanized:
        _main().notifier.send_text(phone, msg)
        return
    _main().notifier.send_text(phone, brain.compose_reply(msg, user_name=user_name))


_YES_WORDS = {"yes", "yeah", "yep", "y", "sure", "ok", "okay", "alright",
              "book", "lock", "proceed", "confirm", "continue", "go", "ahead"}
_NO_WORDS = {"no", "nope", "nah", "cancel", "forget", "never", "stop"}

# Booking-name collection: words too short/numeric to be a person's name
# (a bare "2" or "yes" in the booking gate is a stray pick/answer, not
# "my name is Ed").
_NAME_TOO_SHORT = re.compile(r"^[^A-Za-z]*$")


def _booking_name_ok(text: str) -> bool:
    """A plausible passenger name for the booking gate: has letters, is
    short enough to be a name, and is not a bare yes/no/digit answer."""
    t = (text or "").strip()
    if not t or len(t) > 80 or not re.search(r"[A-Za-z]", t):
        return False
    if t.lower() in (_YES_WORDS | _NO_WORDS) or _NAME_TOO_SHORT.match(t):
        return False
    return True


def _try_booking_answer(db, user: User, text: str) -> bool:
    """THE DETERMINISTIC BOOKING GATE - plain-text name collection.

    When a live re-quote was held (pending_booking set), the user's next
    reply is the traveller's full name: we build the travellers dict,
    call the SAME create_booking path the button taps use, and send the
    Paystack link. No agent loop, no intent parsing - the exact failure
    mode ('Gideon sha' swallowed by a Groq timeout) is unreachable here.

    Falls back to the BUTTON path (fare cards) when the hold expired or
    the reply isn't a plausible name: a stray digit/yes re-enters the
    normal gates, and a REAL new request (a city name, a route) flows to
    search as usual. Returns True when the message was consumed."""
    pending = chatstate.get_pending_booking(db, user.phone)
    if not pending:
        return False
    text = (text or "").strip()

    # A negation cancels the whole booking attempt - say so plainly.
    if text.lower() in _NO_WORDS or "cancel" in text.lower():
        chatstate.clear_pending_booking(db, user.phone)
        _main()._say(user.phone, "No problem - booking dropped. Nothing was held "
                        "or charged.", user.name)
        return True

    if _booking_name_ok(text):
        # The agent can arm this gate conversationally, so a fresh route
        # request ("Lagos to Abuja tomorrow") must NOT become a passenger
        # name - it falls through to the normal search gates instead.
        if brain.single_city(text) is not None or brain.has_date(text):
            chatstate.clear_pending_booking(db, user.phone)
            logger.info("Booking gate dropped (looks like a new route): %r",
                        text[:40])
            return False
        parts = text.split()
        travellers = {"first_name": parts[0],
                      "last_name": " ".join(parts[1:]) or parts[0],
                      "phone": user.phone}
        chatstate.clear_pending_booking(db, user.phone)
        _create_and_send_booking(
            db, user, pending["origin_iata"], pending["destination_iata"],
            pending["flight_date"],
            {"price": pending["price"], "airline": pending.get("airline")},
            flight_iata=pending.get("flight_iata"),
            extra_details={
                "booking_token": pending.get("booking_token"),
                "passengers": {"adults": 1, "children": 0, "infants": 0},
                "travellers": travellers})
        return True

    # Not a name and not a cancel: the hold has likely gone stale (or the
    # user changed topic). Drop it - the message falls through to the
    # normal gates so a genuine new request still works.
    chatstate.clear_pending_booking(db, user.phone)
    logger.info("Booking gate dropped (not a name): %r from %s",
                text[:40], user.phone)
    return False


def _try_ticket_details_answer(db, user: User, text: str) -> bool:
    """Chat fallback for ticket-holder details: paid bookings that never
    went through the /book page form. Two-step ask (name, then email);
    on completion the voucher PDF + email go out. Returns True when the
    message was consumed as an answer."""
    pending = chatstate.get_pending_ticket_details(db, user.phone)
    if not pending:
        return False
    booking = db.get(BookingSession, uuid.UUID(str(pending.get("booking_id"))))
    if booking is None or booking.status != SessionStatus.PAID.value:
        chatstate.clear_pending_ticket_details(db, user.phone)
        return False
    text = (text or "").strip()
    if pending.get("stage") == "name":
        if not text or len(text) > 80 or not re.search(r"[A-Za-z]", text):
            _main()._say(user.phone,
                 "Sorry - I need the traveller's full name as it appears "
                 "on their ID (letters, e.g. Chinedu Okafor).",
                 user.name)
            return True
        pending = {**pending, "stage": "email", "name": text}
        chatstate.set_pending_ticket_details(db, user.phone, pending)
        _main()._say(user.phone,
             f"Thanks, {text}. And the email address the ticket should "
             f"go to?", user.name)
        return True
    # stage == "email"
    email = text.lower()
    if not _main()._EMAIL_RE.match(email):
        _main()._say(user.phone,
             "That doesn't look like an email address - try again "
             "(e.g. you@example.com).", user.name)
        return True
    booking.passenger_name = pending.get("name") or "Passenger"
    booking.contact_email = email
    if not user.email:
        user.email = email
    db.commit()
    chatstate.clear_pending_ticket_details(db, user.phone)
    pnr = str(pending.get("pnr") or "FB-????")
    _main()._say(user.phone,
         f"✅ All set! Ticket for *{booking.passenger_name}* goes to "
         f"{email}. Here's your voucher:", user.name)
    _main().ticket_delivery.send_ticket_pdf(booking, pnr, _main().notifier)
    return True


def _try_requote_answer(db, user: User, text: str) -> bool:
    """Answer to the price-move question ("the price moved to ₦X - still
    lock it?"). YES/BOOK -> book the already re-quoted fare; NO/CANCEL ->
    politely abort. Returns True when handled. A message naming a city or
    a date is a fresh request, not an answer - it falls through."""
    pending = chatstate.get_pending_requote(db, user.phone)
    if not pending:
        return False
    if brain.single_city(text) is not None or brain.has_date(text):
        return False
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return False
    if all(w in _YES_WORDS for w in words):
        chatstate.clear_pending_requote(db, user.phone)
        _reply_requoted_booking(db, user, pending)
        return True
    if all(w in _NO_WORDS for w in words):
        chatstate.clear_pending_requote(db, user.phone)
        _main()._say(user.phone,
             "No problem - that quote is gone anyway. Ask me for a fresh "
             "search and I'll check the latest price.",
             user.name)
        return True
    return False


def _reply_requoted_booking(db, user: User, pending: dict) -> None:
    """Book the fare that was LIVE re-quoted before the price-move question
    (no re-fetch - it is seconds old). Surge is still refused."""
    fare = pending.get("fare") or {}
    origin_iata = pending.get("origin_iata")
    destination_iata = pending.get("destination_iata")
    flight_date = pending.get("flight_date")
    if not (origin_iata and destination_iata and flight_date and fare):
        _main()._say(user.phone,
             "That quote has expired - ask me for a fresh search and I'll "
             "check the latest price.",
             user.name)
        return
    if fare.get("above_guardrail"):
        _main()._say(user.phone,
             "That fare is at a surge price right now - I can't lock it "
             "safely. Try again in a bit, or TRACK it and I'll Beep you "
             "when it normalises.",
             user.name)
        return
    _create_and_send_booking(db, user, origin_iata, destination_iata,
                             flight_date, fare,
                             flight_iata=fare.get("flight_number"))


def _fill_pending_fare(db, user: User, text: str) -> bool:
    """Continue a partial fare conversation: we asked a follow-up (e.g.
    "where will you be flying from?"), and the user answered with a bare
    city. Fill the missing slot and run the search. Returns True when
    handled. Only fires for a clean single-city answer with no new date -
    anything richer is a fresh request for the brain."""
    pending = chatstate.get_pending_fare(db, user.phone)
    if not pending:
        return False
    iata = brain.single_city(text)
    if iata is None or brain.has_date(text):
        return False
    origin_iata = pending.get("origin_iata")
    destination_iata = pending.get("destination_iata")
    date_ = pending.get("date")
    if not origin_iata and destination_iata:
        origin_iata = iata
    elif not destination_iata and origin_iata:
        destination_iata = iata
    else:
        return False
    if not (origin_iata and destination_iata and date_):
        return False
    chatstate.clear_pending_fare(db, user.phone)
    intent = brain.Intent(intent="fare",
                          origin=city_name(origin_iata),
                          destination=city_name(destination_iata),
                          date=date_)
    logger.info("Filled pending fare: %s -> %s on %s (from '%s')",
                origin_iata, destination_iata, date_, text)
    _reply_fare(db, user, intent)
    return True


def _fill_pending_date(db, user: User, text: str) -> bool:
    """Mirror of _fill_pending_fare for the date slot: we asked 'did you
    mean X or Y?' and the user answered with a plain date. Fill it and
    run the search. Returns True when handled. Declines when the pending
    route is incomplete, when the text names a city (a fresh request for
    the brain), when no date parses, or when the answer is itself
    ambiguous (falls through so the confirm gate re-asks - the loop can
    never book a guessed date)."""
    pending = chatstate.get_pending_fare(db, user.phone)
    if not pending:
        return False
    origin_iata = pending.get("origin_iata")
    destination_iata = pending.get("destination_iata")
    if not (origin_iata and destination_iata) or pending.get("date"):
        return False
    if brain._local_route(text):
        return False
    day = brain._local_date(text)
    if not day:
        return False
    from FareBeep.dates import ambiguous_date_hint
    if ambiguous_date_hint(text) is not None:
        return False
    chatstate.clear_pending_fare(db, user.phone)
    intent = brain.Intent(intent="fare",
                          origin=city_name(origin_iata),
                          destination=city_name(destination_iata),
                          date=day)
    logger.info("Filled pending date: %s -> %s on %s (from '%s')",
                origin_iata, destination_iata, day, text)
    _reply_fare(db, user, intent)
    return True


def _try_date_confirm(db, user: User, text: str) -> bool:
    """Pre-agent ambiguous-date guard: locally parse this turn (no AI
    cost); when it names a complete route with a two-way date, ask which
    reading and stop - the agent's tools never see the guess. Returns
    True when handled."""
    try:
        intent = brain.parse_intent(text, force_local=True)
    except Exception:
        return False
    if intent is None or intent.intent not in ("fare", "book", "subscribe"):
        return False
    if not intent.has_route or not _needs_date_confirm(intent):
        return False
    _ask_date_confirm(db, user, intent)
    return True


def _ask_date_confirm(db, user: User, intent: brain.Intent) -> None:
    """Ambiguous date text on a complete route: remember the route
    (dateless pending) and ask which reading - never search or book a
    guessed date. The plain-date answer completes via _fill_pending_date."""
    chatstate.set_pending_fare(db, user.phone, {
        "origin_iata": intent.origin_iata,
        "destination_iata": intent.destination_iata,
        "date": None,
    })
    _main()._say(user.phone,
         f"{intent.date_hint} Reply with the date plainly, e.g. '5 June'.",
         user.name)


def _needs_date_confirm(intent: brain.Intent) -> bool:
    """True when the turn carries an unresolved date ambiguity: the date
    slot is empty BECAUSE the text read two ways (not because no date
    was given). Money-adjacent branches gate on this before acting."""
    return bool(intent.date_hint and not intent.date)


def _ask_missing_info(db, user: User, intent: brain.Intent) -> None:
    """Pass 2 follow-up: the extraction was partial - ask for the missing
    pieces like a human agent would, never show a blank error. For fare
    partials we remember what we already know (chat_state) so the user's
    next bare-city answer completes the search (_fill_pending_fare)."""
    if intent.intent == "fare":
        chatstate.set_pending_fare(db, user.phone, {
            "origin_iata": intent.origin_iata,
            "destination_iata": intent.destination_iata,
            "date": intent.date,
        })
    dest = city_name(intent.destination_iata) if intent.destination_iata else None
    origin = city_name(intent.origin_iata) if intent.origin_iata else None
    if dest and not origin and not intent.date:
        msg = (f"I'd love to help you get to {dest}! ✈️ Where will you be "
               f"flying from, and what date are you looking at?")
    elif dest and not origin:
        msg = f"I'd love to help you get to {dest}! ✈️ Where will you be flying from?"
    elif origin and not dest:
        msg = f"Great - from {origin}! 🎉 Where are you headed?"
    else:
        msg = ("I'd love to help! ✈️ Tell me where you're going, where you're "
               "flying from, and what date you have in mind.")
    _main()._say(user.phone, msg, user.name)


def _get_or_create_user(db, phone: str) -> User:
    user = db.query(User).filter(User.phone == phone).first()
    if user is None:
        user = User(phone=phone, first_seen_at=utcnow())
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def _try_pick(db, text: str, phone: str):
    """Interpret a reply as a ranked-list pick. Returns the picked fare dict,
    "out_of_range" when the number exceeds the list, "unclear" when the reply
    looks like a pick but no option resolves, or None to fall through to the
    brain (no active list, or not a pick-shaped message)."""
    ctx = chatstate.get_last_fares(db, phone)
    if not ctx or not ctx.get("fares"):
        return None
    m = _PICK_RE.search(text.strip())
    if m:
        n = int(m.group(1))
        if n > len(ctx["fares"]):
            return "out_of_range"
        return ctx["fares"][n - 1]
    # Natural-language pick ("the second one", "Air Peace", "the 7am flight",
    # "the cheapest") resolved by the brain. Gated on pick signals so an
    # unrelated message never costs an AI call.
    if _looks_like_pick(text, ctx["fares"]):
        idx = brain.resolve_pick(text, ctx["fares"])
        if idx is None:
            # A QUESTION about the list ("is that the cheapest?", "are they
            # the same airline?") is NOT a pick - let the brain answer it
            # instead of forcing a 1-2-3 redirect.
            if _QUESTION_RE.search(text):
                return None
            return "unclear"
        return ctx["fares"][idx - 1]
    return None


# A bare "2", "number 2", "option 2", "pick 2", "the 2nd one", "#2".
# Only reached when an active ranked list exists (see _try_pick).
_PICK_RE = re.compile(
    r"^(?:the\s+)?(?:number|option|pick|choose|select)?\s*#?\s*"
    r"([1-9])\s*(?:st|nd|rd|th)?\s*(?:one)?\s*$",
    re.IGNORECASE)

# Cheap gate before the AI pick resolver: only messages that could plausibly
# be picking (ordinals, "one", "cheapest", an am/pm time, ...) spend an AI
# call. Everything else falls through to the brain untouched.
_PICK_SIGNAL_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|last|"
    r"one|option|cheapest|cheaper|cheap|best|deal|pick|choose|select)\b"
    r"|\b\d(?:st|nd|rd|th)\b|\b\d{1,2}\s*(?:am|pm)\b", re.IGNORECASE)

_QUESTION_RE = re.compile(
    r"\?|\b(is|are|was|were|do|does|did|can|could|what|which|how|when|"
    r"where)\b", re.IGNORECASE)


def _looks_like_pick(text: str, fares: list) -> bool:
    """True when the message could be choosing a listed fare - so we try the
    pick resolver; False means it is unrelated and should go to the brain."""
    if _PICK_SIGNAL_RE.search(text):
        return True
    flat = text.lower()
    upper = text.upper()
    for f in fares:
        airline = (f.get("airline") or "").lower()
        if airline and any(t in flat for t in airline.split() if len(t) >= 3):
            return True
        fnum = (f.get("flight_number") or "").upper()
        if fnum and fnum in upper:
            return True
        dep = f.get("departs_at")
        if dep and re.match(r"\d{1,2}:\d{2}", str(dep)) and str(dep)[:5] in flat:
            return True
    return False


def _reply_fare(db, user: User, intent: brain.Intent) -> None:
    # A fresh search supersedes any open price-move question.
    chatstate.clear_pending_requote(db, user.phone)
    # Pass 2: destination-only with a date -> search from the user's default
    # hub (Lagos - Nigeria's main base). The fare reply names both cities,
    # so any wrong assumption is visible and correctable.
    origin_iata = intent.origin_iata or "LOS"
    search = _main().LedgerSearch(db)
    fares, surge_fares = search.search_list(origin_iata,
                                            intent.destination_iata,
                                            intent.date, limit=3)
    if not fares:
        if surge_fares:
            # Everything is above the guardrail - flag it, don't fake "no fares".
            surge_price = min(f["price"] for f in surge_fares)
            _main()._say(user.phone,
                 f"Flights are at a surge rate on "
                 f"{city_name(origin_iata)} -> "
                 f"{city_name(intent.destination_iata)} right now. 🚨 Prices "
                 f"are unusually high (from ₦{surge_price:,.0f}) - I won't "
                 f"sell you that. Try again in a bit, or TRACK and I'll Beep "
                 f"you when it normalises.",
                 user.name)
            return
        _main()._say(user.phone,
             f"No fare found {city_name(origin_iata)} -> "
             f"{city_name(intent.destination_iata)} for "
             f"{intent.date or 'that date'}.",
             user.name)
        return

    if len(fares) == 1:
        # Thin route: one result - keep the classic single-fare reply.
        fare = fares[0]
        chatstate.set_last_fare(db, user.phone, {
            "origin_iata": origin_iata,
            "destination_iata": intent.destination_iata,
            "flight_date": fare["flight_date"],
            "price": fare["price"],
            "airline": fare["airline"],
        })
        _main()._say(user.phone,
             f"Fare {city_name(origin_iata)} -> "
             f"{city_name(intent.destination_iata)} {fare['flight_date']}:\n"
             f"₦{fare['price']:,.0f} via {fare['airline']} "
             f"({_fresh_label(fare)})\n"
             + (f"Verify: {fare['verify_link']}\n" if fare.get("verify_link")
                else "")
             + f"Reply BOOK - I'll re-confirm the live price before you "
             f"pay - or TRACK to get a Beep when it drops.",
             user.name)
        _main()._send_fare_cards(user, [fare], origin_iata, intent.destination_iata)
        return

    # Ranked list - one option per airline, NARRATED by the brain so it reads
    # like a person presenting choices, not a menu. The numbered options stay
    # pickable (reply 1, 2 or 3 - or just say the airline).
    chatstate.set_last_fares(db, user.phone, {
        "origin_iata": origin_iata,
        "destination_iata": intent.destination_iata,
        "flight_date": fares[0]["flight_date"],
        "fares": fares,
    })
    origin = city_name(origin_iata)
    destination = city_name(intent.destination_iata)
    msg = brain.compose_ranked_reply(
        fares, origin, destination, fares[0]["flight_date"],
        user_name=user.name)
    _main()._say(user.phone, msg, user.name, humanized=True)
    _main()._send_fare_cards(user, fares, origin_iata, intent.destination_iata)


def _fresh_label(fare: dict) -> str:
    """'checked just now' for live results, 'cached ~N min ago' for
    ledger rows - so a quoted fare never reads fresher than it is."""
    from FareBeep.search import fare_freshness
    return fare_freshness(fare) or "price shown as found"


def _booking_total(airline_price: float) -> float:
    from FareBeep.payments import calculate_final_price
    return calculate_final_price(airline_price)["total_amount"]


def _reply_booking(db, user: User, intent: brain.Intent,
                   picked_fare: dict = None) -> None:
    """THE LIVE HANDSHAKE - what happens when a user replies BOOK.

    1. FORCE REFRESH: the Shared Ledger is ignored; the inventory supplier
       is queried LIVE so the seat exists at the quoted price right now.
    2. Session: a booking_session row is saved with expires_at = now +
       13m by default (10m once the payment method is card).
    3. The WhatsApp/TG call: the Paystack TEST link + the held-total
    message (our 10-minute hold, never a supplier fare lock), with the
    10-minute expiry stated up front.

    picked_fare: when set, the user picked "1, 2 or 3" from a ranked list -
    the route/date come from the list context and the SELECTED flight is
    re-verified live (by flight number) so the price is fresh, never the
    stale list price.

    Date resolution: the intent's own date wins; otherwise the date of the
    LAST fare quote in this chat (so "BOOK" right after a quote books that
    flight). With neither, the bot ASKS - it never silently books today.
    """
    ctx = chatstate.get_last_fare(db, user.phone) or {}
    list_ctx = chatstate.get_last_fares(db, user.phone) or {}
    if picked_fare is not None:
        origin_iata = list_ctx.get("origin_iata")
        destination_iata = list_ctx.get("destination_iata")
        flight_date = list_ctx.get("flight_date")
        if not (origin_iata and destination_iata and flight_date):
            _main()._say(user.phone,
                 "I've lost that fare list - ask me for the fares again "
                 "and I'll re-run the search.",
                 user.name)
            return
    else:
        origin_iata = intent.origin_iata or ctx.get("origin_iata")
        destination_iata = intent.destination_iata or ctx.get("destination_iata")
        flight_date = intent.date or ctx.get("flight_date")
        if not (origin_iata and destination_iata):
            _main()._say(user.phone,
                 "To book: send your route, e.g. BOOK Lagos to Abuja tomorrow "
                 "(or BOOK right after a fare quote I just sent you).",
                 user.name)
            return
        if not flight_date:
            _main()._say(user.phone,
                 "Which date? E.g. BOOK Lagos to Abuja on the 31st - I won't "
                 "book a flight for today without you saying so.",
                 user.name)
            return

    search = _main().LedgerSearch(db)
    expected_price = None
    if picked_fare is not None:
        # Force-refresh the ranked list live and re-locate the picked flight
        # by its number (prices move - never book the stale list price).
        fares, _ = search.search_list(origin_iata, destination_iata,
                                      flight_date, limit=6)
        picked_no = picked_fare.get("flight_number")
        if picked_no:
            fare = next((f for f in fares
                         if f.get("flight_number") == picked_no), None)
        else:
            fare = next((f for f in fares
                         if f.get("airline") == picked_fare.get("airline")
                         and f.get("departs_at") == picked_fare.get("departs_at")),
                        None)
        if fare is None:
            _main()._say(user.phone,
                 "That option is no longer available at that price - the "
                 "seat or fare may have moved. Reply 1, 2 or 3 again, or "
                 "ask me for a fresh search.",
                 user.name)
            return
        expected_price = picked_fare.get("price")
        if fare.get("above_guardrail"):
            _main()._say(user.phone,
                 "That fare is at a surge price right now - I can't lock it "
                 "safely. Try again in a bit, or TRACK it and I'll Beep you "
                 "when it normalises.",
                 user.name)
            return
    else:
        fare = search.search(origin_iata, destination_iata, flight_date,
                             force_refresh=True)
        if fare is None:
            _main()._say(user.phone,
                 "That seat/route is no longer available right now - ask me for "
                 "the latest fare again and I'll re-check live.",
                 user.name)
            return
        if fare.get("above_guardrail"):
            _main()._say(user.phone,
                 "That fare is at a surge price right now - I can't lock it "
                 "safely. Try again in a bit, or TRACK it and I'll Beep you "
                 "when it normalises.",
                 user.name)
            return
        expected_price = ctx.get("price") if ctx else None

    # THE PRICE-MOVE HANDCHECK: the live re-quote may differ from what the
    # user was shown. Within the tolerance (the natural volatility buffer)
    # book silently; beyond it, ASK first - never take a silent price jump.
    #
    # DETERMINISTIC NAME COLLECTION: a bookable fare (it carries a
    # booking_token) can become a REAL PNR the moment payment lands -
    # but only with the traveller's name. Unknown name + token present:
    # hold the re-quoted fare in pending_booking and ask. The next reply
    # is consumed as the name by _try_booking_answer - no agent loop, no
    # NLU between the price hold and money moving. Fares WITHOUT a token
    # (ledger rows, old-shape quotes) keep the one-step BOOK contract;
    # their tickets go out via the post-payment details flow.
    if not (user.name or "").strip() and fare.get("booking_token"):
        chatstate.clear_last_fares(db, user.phone)   # superseded by the question
        chatstate.set_pending_booking(db, user.phone, {
            "origin_iata": origin_iata,
            "destination_iata": destination_iata,
            "flight_date": fare["flight_date"],
            "price": fare["price"],
            "airline": fare.get("airline"),
            "booking_token": fare.get("booking_token"),
            "flight_iata": fare.get("flight_number") or intent.flight,
            "stage": "name"})
        _main()._say(user.phone,
             f"Great - {city_name(origin_iata)} -> "
             f"{city_name(destination_iata)} on {fare['flight_date']} at "
             f"₦{fare['price']:,.0f} ({fare.get('airline') or 'airline'}), "
             f"re-checked live just now.\n"
             f"What's the passenger's full name (as on their ID)? I'll "
             f"hold the total for 10 minutes straight after.",
             user.name)
        return
    if (expected_price is not None
            and abs(fare["price"] - expected_price) > _main().REQUOTE_TOLERANCE_NGN):
        chatstate.clear_last_fares(db, user.phone)   # superseded by the question
        chatstate.set_pending_requote(db, user.phone, {
            "origin_iata": origin_iata,
            "destination_iata": destination_iata,
            "flight_date": fare["flight_date"],
            "fare": fare,
        })
        _main()._say(user.phone,
             f"The fare moved from ₦{expected_price:,.0f} to "
             f"₦{fare['price']:,.0f} while you were deciding - the airline "
             f"repriced it. Still go ahead at the new price? Reply YES or "
             f"BOOK - or NO to cancel.",
             user.name)
        return

    _create_and_send_booking(db, user, origin_iata, destination_iata,
                             fare["flight_date"], fare,
                             flight_iata=fare.get("flight_number")
                             or intent.flight)

    if picked_fare is not None:
        # A pick is consumed by booking: a stray "2" later must not rebook.
        chatstate.clear_last_fares(db, user.phone)


def _create_and_send_booking(db, user: User, origin_iata: str,
                             destination_iata: str, flight_date: str,
                             fare: dict, flight_iata: str = None,
                             extra_details: dict = None) -> None:
    """Create the 10-minute booking session and send the held-total
    message with the confirmation-page link. Shared by the direct BOOK
    flow, the re-quoted "yes" flow, and the deterministic booking gate.

    extra_details: merged into flight_details after creation (the booking
    gate's travellers/booking_token - stored so the paid webhook can
    ticket via the supplier without another user round-trip)."""
    bookings = _main().BookingService(db)
    try:
        result = bookings.create_booking(
            user.user_id, origin_iata, destination_iata,
            flight_date, fare["price"],
            flight_iata=flight_iata,
            email=user.email or f"{user.phone.replace('+', '')}@farebeep.ng",
            airline=fare.get("airline"), source="live")
    except Exception as e:
        logger.error("Booking creation failed: %s", e)
        _main()._say(user.phone, "Payment link could not be created. Try again in a minute.", user.name)
        return

    if extra_details:
        session = result["session"]
        details = dict(session.flight_details or {})
        details.update(extra_details)
        session.flight_details = details
        db.commit()

    session = result["session"]
    expires = result["expires_at"].strftime("%H:%M")
    origin = city_name(origin_iata)
    destination = city_name(destination_iata)
    # The user goes to the booking confirmation page (NOT the raw Paystack
    # URL): it reconfirms the price breakdown + captures NDPA consent before
    # redirecting to payment. Every booking flows through this page.
    book_url = f"{_main().APP_BASE_URL}/book/{session.id}"
    _main()._say(user.phone,
         f"🔒 Total held for 10 minutes - our promise, not the airline's: "
         f"no supplier holds seats for us, so if the fare moves again "
         f"before you pay, tell us and we'll make it right.\n"
         f"{origin} -> {destination} on {flight_date}\n"
         f"Airline price (re-checked live just now): "
         f"₦{fare['price']:,.0f}\n"
         f"Markup + fees: ₦{session.markup + session.processing_fee:,.0f}\n"
         f"TOTAL:         ₦{result['total_amount']:,.0f}\n\n"
         f"Confirm & pay here (valid until {expires} today):\n"
         f"{book_url}\n\n"
         f"⚠️ PAYSTACK TEST MODE - use the test card, no real money leaves "
         f"your account.\n"
         f"Miss the window? Your payment is auto-refunded - no questions.",
         user.name)


def _reply_status_ack(user: User, intent: brain.Intent) -> None:
    flight = intent.flight or "your flight"
    _main()._say(user.phone,
         f"Status watch for {flight} is attached to a paid booking. "
         f"We'll text you via template 3h before departure if anything changes.",
         user.name)
