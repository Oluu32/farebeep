"""Beep subscriptions: managing watches, stop/erase requests, and the tap
handlers that arm or re-check a beep from a card.

A "beep" is a SubscriptionMonitor row (route + optional target price +
date). This module owns the chat-side flows around it:

  - deterministic management: LIST, PAUSE, RESUME, CANCEL, EDIT (_handle_manage)
  - stop / unsubscribe / erase: _is_stop_request, _handle_stop, _reply_unsubscribe
  - arming a beep: _reply_subscribe (TRACK), _tap_alert (card "set alert"),
    _send_beep_flow (the "set a beep" button)
  - card taps: _tap_beep, _tap_alert (and the *_by_phone background entry points)

Names that main owns (_say, SessionLocal, _get_or_create_user, _reply_fare,
_handle_incoming_message, notifier, MetaWhatsapp, MESSAGING_PROVIDER,
APP_BASE_URL) are resolved through FareBeep.main at call time, not imported,
so tests that monkeypatch them on main still take effect. This module must
never import main at load time (main imports it).
"""
import logging
import time

from FareBeep import brain, chatstate
from FareBeep.iata import city_name
from FareBeep.models import User

logger = logging.getLogger("farebeep.main")


def _main():
    """The FareBeep.main module, looked up at call time (see docstring)."""
    from FareBeep import main
    return main


def _send_beep_flow(phone: str) -> None:
    """Offer the WhatsApp Flow (Meta-hosted structured screens). Without
    BEEP_FLOW_ID - or on a non-Meta channel - degrade to the guided
    TRACK path so the button never dead-ends."""
    if _main().MESSAGING_PROVIDER.lower() == "telegram":
        # The Meta-hosted Flow cannot open in a Telegram chat - the
        # Mini App is its equivalent: the beep form opens embedded in
        # the chat (needs a public https origin). Without one, the
        # guided TRACK path keeps the button from dead-ending.
        if _main().APP_BASE_URL.startswith("https://"):
            sent = _main().notifier.send_beep_app(
                phone, f"{_main().APP_BASE_URL}/mini/beep")
            if sent:
                return
        _main()._handle_incoming_message(phone, "TRACK")
        return
    try:
        sent = _main().MetaWhatsapp().send_flow(
            phone, flow_cta="Set my beep",
            flow_token=f"beep:{phone}:{int(time.time())}",
            body="Set a price-drop alert - we'll message you when fares drop.")
    except Exception as e:
        logger.warning("send_flow failed (%s) - TRACK fallback: %s",
                       phone, e)
        sent = False
    if not sent:
        _main()._handle_incoming_message(phone, "TRACK")


def _flow_target_price(d: dict):
    raw = d.get("target_price")
    if raw in (None, ""):
        return None
    try:
        v = float(raw)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


_MANAGE_LIST = frozenset({
    "beeps", "my beeps", "my alerts", "alerts", "watches", "my watches",
    "show beeps", "show alerts", "list beeps", "list alerts",
})
_MANAGE_VERBS = ("PAUSE", "RESUME", "CANCEL", "EDIT")


def _list_beeps(db, user: User) -> None:
    """Numbered beep list (numbers match PAUSE/RESUME/CANCEL/EDIT)."""
    from FareBeep.alerts import format_watch, list_watches
    subs = list_watches(db, user.user_id)
    if not subs:
        _main()._say(user.phone,
             "You have no price beeps yet. Send a route like 'Lagos to "
             "Abuja tomorrow' and I'll watch it - or TRACK it with a "
             "target price.",
             user.name)
        return
    lines = "\n".join(format_watch(s, i) for i, s in enumerate(subs, start=1))
    _main()._say(user.phone,
         f"🔔 Your beeps ({len(subs)}):\n{lines}\n\n"
         f"Reply PAUSE n, RESUME n, CANCEL n, or EDIT n 70000.",
         user.name)


def _parse_edit_price(rest: str):
    """Last number in the text is the new target ('70k', '70,000');
    the word DROP (and nothing numeric) clears it to drop-watch."""
    import re
    if re.search(r"\bdrop\b", rest or "", re.IGNORECASE):
        return "drop"
    nums = re.findall(r"(\d[\d,]*)\s*(k)?\b", rest or "", re.IGNORECASE)
    if not nums:
        return None
    digits, kilo = nums[-1]
    try:
        value = int(digits.replace(",", ""))
    except ValueError:
        return None
    return float(value * 1000 if kilo else value)


def _handle_manage(db, user: User, text: str) -> bool:
    """Deterministic beep management. True = turn handled (agent and
    brain must not run). Bare PAUSE/RESUME apply to a lone watch;
    bare CANCEL/EDIT always ask first (destructive/ambiguous)."""
    norm = (text or "").strip()
    flat = norm.lower().rstrip("?!.")
    upper = norm.upper().rstrip("?!.")
    if flat in _MANAGE_LIST:
        _list_beeps(db, user)
        return True
    verb = None
    for v in _MANAGE_VERBS:
        if upper == v or upper.startswith(v + " "):
            verb = v
            break
    if verb is None:
        return False
    rest = norm[len(verb):].strip()
    from FareBeep.alerts import (WatchAmbiguous, format_watch, list_watches,
                                 match_watch)
    subs = list_watches(db, user.user_id)
    if not subs:
        _main()._say(user.phone,
             "You have no price beeps to change. Send a route like "
             "'Lagos to Abuja tomorrow' and I'll watch it.",
             user.name)
        return True
    if not rest:
        if verb in ("PAUSE", "RESUME") and len(subs) == 1:
            sub, n = subs[0], 1
        else:
            _list_beeps(db, user)
            return True
    else:
        try:
            found = match_watch(subs, rest)
        except WatchAmbiguous:
            _main()._say(user.phone,
                 "Which one did you mean?\n" +
                 "\n".join(format_watch(s, i)
                            for i, s in enumerate(subs, start=1)) +
                 f"\n\nReply {verb} with its number.",
                 user.name)
            return True
        sub, n = found
        if sub is None:
            if verb == "EDIT" and len(subs) == 1:
                sub, n = subs[0], 1
            else:
                _main()._say(user.phone,
                     "I couldn't find that watch. " +
                     "\n".join(format_watch(s, i)
                                for i, s in enumerate(subs, start=1)) +
                     f"\n\nReply {verb} with its number.",
                     user.name)
                return True
    if verb == "PAUSE":
        sub.paused = True
        db.commit()
        _main()._say(user.phone,
             f"⏸ Paused {format_watch(sub, n)} - settings kept, silent "
             f"until you RESUME it.",
             user.name)
    elif verb == "RESUME":
        sub.paused = False
        db.commit()
        _main()._say(user.phone, f"▶ Resumed {format_watch(sub, n)}.", user.name)
    elif verb == "CANCEL":
        db.delete(sub)
        db.commit()
        _main()._say(user.phone,
             f"❌ Cancelled {city_name(sub.origin)} -> "
             f"{city_name(sub.destination)} - watch deleted.",
             user.name)
    elif verb == "EDIT":
        # The watch number is not a price: "EDIT 1" must ask for one,
        # not set the target to N naira.
        import re
        price_text = re.sub(rf"^{n}\b", "", rest).strip()
        price = _parse_edit_price(price_text)
        if price is None:
            _main()._say(user.phone,
                 f"To change {format_watch(sub, n)}, reply like: EDIT {n} "
                 f"70000 - or EDIT {n} DROP to watch any genuine drop.",
                 user.name)
            return True
        from FareBeep.alerts import SubscriptionMonitor
        new_target = None if price == "drop" else price
        SubscriptionMonitor(db).subscribe(
            user.user_id, sub.origin, sub.destination,
            target_price=new_target, target_date=sub.target_date)
        if new_target is None:
            _main()._say(user.phone,
                 f"✏️ Updated {format_watch(sub, n)}: now watching any "
                 f"genuine drop.",
                 user.name)
        else:
            _main()._say(user.phone,
                 f"✏️ Updated {format_watch(sub, n)}: now beeping at "
                 f"₦{new_target:,.0f} or lower.",
                 user.name)
    return True


# Whole-message stop requests: handled deterministically BEFORE the agent
# (the model must never improvise deletions - NDPA data promise).
_STOP_PHRASES = frozenset({
    "stop", "unsubscribe", "stop all", "stop alerts", "cancel all alerts",
    "cancel my alerts", "remove all alerts", "delete my data",
    "delete everything", "opt out", "optout", "stop beeps",
})

# Phrases that stop everything even buried in a sentence ("please stop
# messaging me"). Each REQUIRES an object word - a bare "stop" inside
# other words ("stopover in Abuja", "nonstop flight") must never match.
_STOP_CONTAINS = (
    r"\bstop\s+(all\s+)?(alerts|beeps|messages|messaging|notifications)\b",
    r"\bunsubscribe\b",
    r"\bcancel\s+(all\s+|my\s+)?(alerts|beeps)\b",
    r"\bdelete\s+(my|all|everything)\b",
    r"\bopt\s?out\b",
)


def _is_stop_request(text: str) -> bool:
    """Stop/unsubscribe/erase request: whole message, or a stop-phrase
    with an explicit object (punctuation tolerated). Negated phrasing
    ("don't cancel my alerts") is NEVER a stop - it falls through to
    the agent/brain for nuanced handling."""
    import re
    flat = (text or "").strip().lower().rstrip("!.").strip()
    if flat in _STOP_PHRASES:
        return True
    if _looks_negated(flat):
        return False
    return any(re.search(pat, flat) for pat in _STOP_CONTAINS)


def _looks_negated(text: str) -> bool:
    """Hedged/negated phrasing that must never trigger deletion."""
    import re
    return bool(re.search(r"\b(don'?t|do not|never mind|not really)\b",
                          text or ""))


def _handle_stop(db, user: User) -> None:
    """Delete every subscription AND all chat-scoped state, then confirm.
    Money records (booking_sessions) are untouched - refunds stay provable."""
    from FareBeep.alerts import SubscriptionMonitor
    removed = SubscriptionMonitor(db).unsubscribe(user.user_id)
    chatstate.clear_pending_fare(db, user.phone)
    chatstate.clear_pending_requote(db, user.phone)
    chatstate.clear_last_fare(db, user.phone)
    chatstate.clear_last_fares(db, user.phone)
    chatstate.clear_agent_history(db, user.phone)
    chatstate.clear_support_ticket(db, user.phone)
    if removed:
        _main()._say(user.phone,
             f"🔕 Stopped - all {removed} price alert(s) removed and your "
             f"chat data deleted.",
             user.name)
    else:
        _main()._say(user.phone,
             "You have no active price alerts - nothing to stop. "
             "Your chat data is cleared anyway.",
             user.name)


def _reply_subscribe(db, user: User, intent: brain.Intent) -> None:
    """Set a fare-drop subscription (target price optional).

    Like BOOK: a bare TRACK right after a fare quote arms the alert on the
    ROUTE + DATE the user was just shown (never a made-up route). The route
    lives in the single-fare context OR the active ranked list - both count
    as "the route we were just discussing"."""
    from FareBeep.alerts import SubscriptionMonitor
    ctx = chatstate.get_last_fare(db, user.phone) or {}
    list_ctx = chatstate.get_last_fares(db, user.phone) or {}
    origin_iata = (intent.origin_iata or ctx.get("origin_iata")
                   or list_ctx.get("origin_iata"))
    destination_iata = (intent.destination_iata or ctx.get("destination_iata")
                        or list_ctx.get("destination_iata"))
    if not (origin_iata and destination_iata):
        _main()._say(user.phone,
             "To set a price alert, send your route, e.g. 'TRACK Lagos to "
             "Abuja below 80k' - or just say TRACK right after a fare quote.",
             user.name)
        return
    target_date = (intent.date or ctx.get("flight_date")
                   or list_ctx.get("flight_date"))

    monitor = SubscriptionMonitor(db)
    monitor.subscribe(user.user_id, origin_iata, destination_iata,
                      target_price=intent.target_price, target_date=target_date)

    origin = city_name(origin_iata)
    destination = city_name(destination_iata)
    if intent.target_price is not None:
        msg = (f"✅ Beep armed: {origin} -> {destination}\n"
               f"We'll text you the moment it hits ₦{intent.target_price:,.0f} or lower.")
        from FareBeep.alerts import target_realism_note
        note = target_realism_note(db, origin_iata, destination_iata,
                                   intent.target_price)
        if note:
            msg += f"\n\n{note}"
    else:
        msg = (f"✅ Beep armed: {origin} -> {destination}\n"
               f"We'll text you when the fare drops by 10% or more.")
    _main()._say(user.phone, msg, user.name)


def _tap_beep_by_phone(phone: str, sub_id: int) -> None:
    """Background task for a beep template's "Book now" tap: re-present
    that subscription's route fresh (never books blind). Own session,
    like _main()._handle_incoming_message - never the request's."""
    db = _main().SessionLocal()
    try:
        try:
            user = _main()._get_or_create_user(db, phone)
            _tap_beep(db, user, sub_id)
        except Exception as e:
            logger.error("Beep-tap handling failed (%s): %s", phone, e)
    finally:
        db.close()


def _tap_beep(db, user: User, sub_id: int) -> None:
    """"Book now" from a price-drop Beep: look up the subscription (it
    must belong to THIS user - foreign ids are ignored silently) and
    run a fresh fare search for its route+date. The user then picks
    and books from live numbers, exactly like a typed search."""
    from FareBeep.models import Subscription
    sub = db.query(Subscription).filter_by(id=sub_id).first()
    if sub is None:
        _main()._say(user.phone,
             "That alert has expired - send me your route and I'll "
             "check fresh fares for you.",
             user.name)
        return
    if sub.user_id != user.user_id:
        logger.warning("Beep tap rejected: sub %s not owned by %s",
                       sub_id, user.phone)
        return
    intent = brain.Intent(intent="fare",
                          origin=city_name(sub.origin),
                          destination=city_name(sub.destination),
                          date=sub.target_date)
    logger.info("Beep tap: fresh search %s->%s for %s",
                sub.origin, sub.destination, user.phone)
    _main()._reply_fare(db, user, intent)


def _tap_alert_by_phone(phone: str, idx: int) -> None:
    """Background task for a card's "Set alert" tap: subscribe from the
    fare the user was just shown (ranked list first, single quote next).
    Own session, like _main()._handle_incoming_message - never the request's."""
    db = _main().SessionLocal()
    try:
        try:
            user = _main()._get_or_create_user(db, phone)
            _tap_alert(db, user, idx)
        except Exception as e:
            logger.error("Alert-tap handling failed (%s): %s", phone, e)
    finally:
        db.close()


def _tap_alert(db, user: User, idx: int) -> None:
    """Subscribe the tapped fare's route+date, target = its price ("beep
    me below this"). idx is 1-based like the cards and the pick gate
    (alert:1 = first fare); 0 = single-fare card backed by last_fare.
    Mirrors _reply_subscribe's confirm wording."""
    from FareBeep.alerts import SubscriptionMonitor
    ctx = chatstate.get_last_fares(db, user.phone) or {}
    fares = ctx.get("fares") or []
    fare = fares[idx - 1] if 1 <= idx <= len(fares) else None
    origin_iata = ctx.get("origin_iata")
    destination_iata = ctx.get("destination_iata")
    if fare is None:
        # Single-fare card (alert:0): the quote lives in last_fare.
        single = chatstate.get_last_fare(db, user.phone) or {}
        if not single.get("price"):
            _main()._say(user.phone,
                 "That fare list has expired - ask me for the fares again "
                 "and I'll re-run the search.",
                 user.name)
            return
        fare, origin_iata, destination_iata = (
            single, single.get("origin_iata"), single.get("destination_iata"))
    if not (origin_iata and destination_iata):
        _main()._say(user.phone,
             "That fare list has expired - ask me for the fares again "
             "and I'll re-run the search.",
             user.name)
        return
    monitor = SubscriptionMonitor(db)
    monitor.subscribe(user.user_id, origin_iata, destination_iata,
                      target_price=fare.get("price"),
                      target_date=fare.get("flight_date"))
    _main()._say(user.phone,
         f"✅ Beep armed: {city_name(origin_iata)} -> "
         f"{city_name(destination_iata)}\n"
         f"We'll text you the moment it hits "
         f"₦{fare.get('price', 0):,.0f} or lower.",
         user.name)


def _reply_unsubscribe(db, user: User) -> None:
    """Remove all subscriptions for the user (NDPA-style data removal)."""
    from FareBeep.alerts import SubscriptionMonitor
    removed = SubscriptionMonitor(db).unsubscribe(user.user_id)
    if removed:
        _main()._say(user.phone,
             f"🔕 {removed} price alert(s) removed. Your route data is deleted.",
             user.name)
    else:
        _main()._say(user.phone, "You have no active price alerts.", user.name)
