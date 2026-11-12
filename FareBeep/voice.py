"""Voice notes -> grounded flight search (WhatsApp, Telegram, Twilio).

Nigerians send voice notes, not neat text: "abeg find me Lagos to Abuja
Friday morning". This module turns one into a search WITHOUT guessing,
using the same pattern as Kwota (an artisan quoting tool built on the
open NaijaVox ASR model):

    audio --ASR--> transcript --parse_intent--> Intent
                                   |
                         GROUNDING (deterministic)
              every field must point at words actually heard
                                   |
          "I heard: ..."  + each field next to its quote  --> YES / fix

1. LISTEN (pluggable ASR, `VOICE_ASR`):
   - "naijavox": Axiveri/NaijaVox-2.0 (Apache-2.0, Whisper-large-v3 LoRA
     fine-tuned on Pidgin, Yoruba, Hausa, Igbo and Nigerian English).
     No serverless provider hosts it, so FareBeep calls YOUR deployment
     (Hugging Face Inference Endpoint or a small GPU box) at NAIJAVOX_URL.
   - "groq": Groq's hosted whisper-large-v3 (OpenAI-compatible
     /audio/transcriptions). Same GROQ_API_KEY the agent already uses.
     Generic Whisper is weaker on Pidgin, but it is zero-ops.
   - "auto" (default): NaijaVox when NAIJAVOX_URL is set, Groq as the
     fallback when NaijaVox is down or unset.
2. UNDERSTAND: the transcript goes through the SAME brain.parse_intent
   as typed text (Gemini, or the local parser in guided mode).
3. DON'T TRUST IT: `ground()` only keeps a city/date when the words that
   produced it can be found in the transcript (exact alias, or a fuzzy
   match for ASR spelling drift like "abuya"). Anything the model
   produced that was never heard is dropped and asked for - never searched.
4. CONFIRM: the user sees the transcript and every field beside its
   quote, then replies YES. Typed corrections replace the voice draft.

Booking-sensitive moments (a passenger name being collected, a re-quote
waiting for yes/no) never accept voice: a misheard name on a ticket costs
real money, so the user is asked to type instead.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from FareBeep import config
from FareBeep.iata import CITY_TO_IATA, city_name

logger = logging.getLogger("farebeep.voice")

MAX_AUDIO_BYTES = 16 * 1024 * 1024     # WhatsApp's own audio cap is 16 MB
DRAFT_TTL_MINUTES = 15                 # a stale "YES" must not search

_YES = {"yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "correct",
        "confirm", "go", "go ahead", "search", "yes search", "na so",
        "na im", "e correct", "oya", "ehen"}
# "stop" is deliberately absent: STOP keeps its unsubscribe meaning.
_NO = {"no", "nope", "nah", "cancel", "wrong", "no be so"}


# ---------------------------------------------------------------------------
# 1. LISTEN - pluggable ASR
# ---------------------------------------------------------------------------
@dataclass
class Transcript:
    text: str
    engine: str                     # "naijavox" | "groq"


class TranscriptionError(RuntimeError):
    pass


def _filename_for(mime: str) -> str:
    mime = (mime or "").split(";")[0].strip().lower()
    return {
        "audio/ogg": "voice.ogg", "audio/opus": "voice.ogg",
        "audio/mpeg": "voice.mp3", "audio/mp3": "voice.mp3",
        "audio/mp4": "voice.m4a", "audio/aac": "voice.m4a",
        "audio/x-m4a": "voice.m4a", "audio/amr": "voice.amr",
        "audio/wav": "voice.wav", "audio/x-wav": "voice.wav",
        "audio/webm": "voice.webm",
    }.get(mime, "voice.ogg")


def _transcribe_naijavox(audio: bytes, mime: str,
                         client: httpx.Client) -> str:
    """POST raw audio to a NaijaVox deployment.

    Contract = the Hugging Face Inference Endpoint ASR contract: raw bytes
    in, {"text": "..."} out (a list of one such dict is tolerated).
    NAIJAVOX_LANGUAGE (pcm/en/yo/ha/ig) is forwarded as a generation
    parameter header for custom handlers that honour it; the stock
    pipeline ignores it and auto-detects.
    """
    url = config.NAIJAVOX_URL
    if not url:
        raise TranscriptionError("NAIJAVOX_URL not set")
    headers = {"Content-Type": (mime or "audio/ogg").split(";")[0]}
    if config.NAIJAVOX_TOKEN:
        headers["Authorization"] = f"Bearer {config.NAIJAVOX_TOKEN}"
    if config.NAIJAVOX_LANGUAGE:
        headers["X-Language"] = config.NAIJAVOX_LANGUAGE
    resp = client.post(url, content=audio, headers=headers,
                       timeout=config.VOICE_ASR_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        data = data[0]
    text = (data or {}).get("text") if isinstance(data, dict) else None
    if not isinstance(text, str):
        raise TranscriptionError(f"NaijaVox returned no text: {str(data)[:120]}")
    return text


def _transcribe_groq(audio: bytes, mime: str, client: httpx.Client) -> str:
    """Groq hosted Whisper (OpenAI-compatible multipart upload)."""
    key = config.GROQ_API_KEY
    if not key:
        raise TranscriptionError("GROQ_API_KEY not set")
    # A short prompt biases Whisper toward the words FareBeep cares about
    # (city names, Pidgin travel phrasing) - spelling, not meaning.
    prompt = ("Nigerian Pidgin and English flight request. Lagos, Abuja, "
              "Port Harcourt, Enugu, Kano, Owerri, Uyo, Calabar, Benin. "
              "Abeg find me flight, tomorrow, next week, Friday morning.")
    resp = client.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {key}"},
        files={"file": (_filename_for(mime), audio,
                        (mime or "audio/ogg").split(";")[0])},
        data={"model": config.GROQ_ASR_MODEL, "prompt": prompt,
              "response_format": "json", "temperature": "0"},
        timeout=config.VOICE_ASR_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.json().get("text")
    if not isinstance(text, str):
        raise TranscriptionError("Groq returned no text")
    return text


_ENGINES = {"naijavox": _transcribe_naijavox, "groq": _transcribe_groq}


def engine_order() -> list[str]:
    """Which engines to try, in order, for the configured VOICE_ASR."""
    mode = (config.VOICE_ASR or "auto").strip().lower()
    if mode == "off":
        return []
    if mode in _ENGINES:
        return [mode]
    order = []
    if config.NAIJAVOX_URL:
        order.append("naijavox")
    if config.GROQ_API_KEY:
        order.append("groq")
    return order


def transcribe(audio: bytes, mime: str = "audio/ogg",
               http_client: Optional[httpx.Client] = None) -> Transcript:
    """Audio bytes -> Transcript. Tries each configured engine in order;
    raises TranscriptionError only when every engine failed."""
    if not audio:
        raise TranscriptionError("empty audio")
    if len(audio) > MAX_AUDIO_BYTES:
        raise TranscriptionError("audio too large")
    order = engine_order()
    if not order:
        raise TranscriptionError("no ASR engine configured "
                                 "(set NAIJAVOX_URL or GROQ_API_KEY)")
    own = http_client is None
    client = http_client or httpx.Client()
    errors = []
    try:
        for name in order:
            try:
                text = " ".join((_ENGINES[name](audio, mime, client) or "")
                                .split())
                logger.info("Voice note transcribed by %s (%d chars)",
                            name, len(text))
                return Transcript(text=text, engine=name)
            except Exception as e:
                logger.warning("ASR %s failed: %s", name, e)
                errors.append(f"{name}: {e}")
    finally:
        if own:
            client.close()
    raise TranscriptionError("; ".join(errors))


# ---------------------------------------------------------------------------
# 3. DON'T TRUST IT - grounding every field in the heard words
# ---------------------------------------------------------------------------
@dataclass
class Grounded:
    value: object = None            # IATA / ISO date / int / str
    quote: Optional[str] = None     # the exact transcript words
    fuzzy: bool = False             # matched through ASR spelling drift
    assumed: bool = False           # default, never said (passengers=1)


@dataclass
class VoiceDraft:
    transcript: str
    origin: Grounded = field(default_factory=Grounded)
    destination: Grounded = field(default_factory=Grounded)
    date: Grounded = field(default_factory=Grounded)
    time_of_day: Grounded = field(default_factory=Grounded)
    passengers: Grounded = field(default_factory=lambda: Grounded(
        value=1, assumed=True))
    dropped: list = field(default_factory=list)   # model output never heard
    intent: str = "fare"

    @property
    def complete(self) -> bool:
        return bool(self.origin.value and self.destination.value
                    and self.date.value)

    def missing(self) -> list[str]:
        out = []
        if not self.origin.value:
            out.append("origin")
        if not self.destination.value:
            out.append("destination")
        if not self.date.value:
            out.append("date")
        return out

    def canonical_text(self) -> str:
        """Plain text the EXISTING pipeline already understands, e.g.
        'Lagos to Abuja 2026-11-20 for 2 adults'. Only grounded fields."""
        parts = []
        if self.intent == "book":
            parts.append("book")
        if self.origin.value and self.destination.value:
            parts.append(f"{city_name(self.origin.value)} to "
                         f"{city_name(self.destination.value)}")
        elif self.destination.value:
            parts.append(f"flight to {city_name(self.destination.value)}")
        elif self.origin.value:
            parts.append(f"flight from {city_name(self.origin.value)}")
        if self.date.value:
            parts.append(str(self.date.value))
        n = int(self.passengers.value or 1)
        if not self.passengers.assumed and n > 1:
            parts.append(f"for {n} adults")
        return " ".join(parts)

    def to_state(self) -> dict:
        def g(x: Grounded) -> dict:
            return {"value": x.value, "quote": x.quote, "fuzzy": x.fuzzy,
                    "assumed": x.assumed}
        return {
            "transcript": self.transcript, "intent": self.intent,
            "origin": g(self.origin), "destination": g(self.destination),
            "date": g(self.date), "time_of_day": g(self.time_of_day),
            "passengers": g(self.passengers),
            "canonical": self.canonical_text(),
            "complete": self.complete,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


_WORD_RE = re.compile(r"[A-Za-z0-9']+(?:[-/][A-Za-z0-9']+)*")
_ROUTE_WORDS = {"from", "to", "go", "for", "reach", "comot", "fly", "into",
                "leave", "leaving", "travel", "commot", "enter"}


def _tokens(text: str) -> list[tuple[str, int, int]]:
    """(lowercased word, start, end) spans over the ORIGINAL transcript."""
    return [(m.group(0).lower(), m.start(), m.end())
            for m in _WORD_RE.finditer(text)]


def _aliases_for(iata: str) -> list[str]:
    # Long, real-word aliases only for fuzzy work; 2-3 letter codes
    # ("ph", "abj") must match exactly or they collide with normal words.
    return sorted((a for a, code in CITY_TO_IATA.items() if code == iata),
                  key=len, reverse=True)


def _with_route_word(text: str, toks, i_start: int, i_end: int) -> str:
    """Quote = the city words, plus the route word right before them
    ("from Lagos", "go Abuja") so the user sees WHICH side it went to."""
    s = toks[i_start][1]
    if i_start > 0 and toks[i_start - 1][0] in _ROUTE_WORDS:
        s = toks[i_start - 1][1]
    return text[s:toks[i_end][2]]


def _find_city(text: str, iata: str, skip_spans=()) -> Optional[Grounded]:
    """Locate words in the transcript that resolve to `iata`."""
    toks = _tokens(text)
    words = [t[0] for t in toks]
    taken = set()
    for a, b in skip_spans:
        taken.update(range(a, b + 1))
    # exact alias (multi-word first)
    for alias in _aliases_for(iata):
        parts = alias.split()
        n = len(parts)
        for i in range(len(words) - n + 1):
            if words[i:i + n] == parts and not taken & set(range(i, i + n)):
                return Grounded(value=iata,
                                quote=_with_route_word(text, toks, i, i + n - 1))
    # fuzzy (ASR spelling drift): single tokens >= 4 letters vs real names
    for alias in _aliases_for(iata):
        if " " in alias or len(alias) < 4:
            continue
        for i, w in enumerate(words):
            if i in taken or len(w) < 4:
                continue
            if difflib.SequenceMatcher(None, w, alias).ratio() >= 0.8:
                return Grounded(value=iata, fuzzy=True,
                                quote=_with_route_word(text, toks, i, i))
    return None


def _span_of(text: str, quote: str) -> tuple[int, int]:
    """Token index span of a quote returned by _find_city."""
    toks = _tokens(text)
    q = _tokens(quote)
    words = [t[0] for t in toks]
    qw = [t[0] for t in q]
    for i in range(len(words) - len(qw) + 1):
        if words[i:i + len(qw)] == qw:
            return (i, i + len(qw) - 1)
    return (-1, -1)


_FROM_WORDS = {"from", "comot", "commot", "leave", "leaving"}
_TO_WORDS = {"to", "go", "reach", "into", "enter"}


def _fuzzy_route(text: str) -> tuple[Optional[str], Optional[str]]:
    """Local route read when the brain didn't get both sides: cities in
    order of appearance (exact aliases, or fuzzy matches on real names for
    ASR drift), sides decided by the word before each city ("from Lagos",
    "go Abuja"); unmarked cities fill the open side in order."""
    toks = _tokens(text)
    names = {}
    for alias, code in CITY_TO_IATA.items():
        if len(alias) >= 4 and " " not in alias:
            names.setdefault(alias, code)
    hits = []                                  # (code, role)
    for i, (w, _, _) in enumerate(toks):
        code = None
        if w in CITY_TO_IATA and len(w) >= 3:
            code = CITY_TO_IATA[w]
        elif len(w) >= 4:
            best = difflib.get_close_matches(w, list(names), n=1, cutoff=0.8)
            if best:
                code = names[best[0]]
        if not code or any(code == h[0] for h in hits):
            continue
        prev = toks[i - 1][0] if i > 0 else ""
        role = ("origin" if prev in _FROM_WORDS
                else "destination" if prev in _TO_WORDS else None)
        hits.append((code, role))
    origin = next((c for c, r in hits if r == "origin"), None)
    dest = next((c for c, r in hits if r == "destination" and c != origin),
                None)
    for c, r in hits:
        if r is None and c not in (origin, dest):
            if len(hits) == 1:
                dest = dest or c          # "flight to X" style, one city
            elif origin is None:
                origin = c
            elif dest is None:
                dest = c
    return origin, dest


def _lead(quote: Optional[str]) -> str:
    toks = _tokens(quote or "")
    return toks[0][0] if len(toks) > 1 else ""


_DATE_LEAD = {"next", "this", "coming", "week", "on", "the", "by"}


def _find_date(text: str, iso: str, today) -> Optional[Grounded]:
    """Shortest, earliest window of 1-4 words that parses to `iso`,
    widened left over date words ("next week Monday", "on the 20th") so
    the user sees the whole phrase that was heard."""
    from FareBeep.dates import parse_expression

    def parses(a: int, b: int) -> bool:
        try:
            return parse_expression(text[toks[a][1]:toks[b][2]],
                                    today=today) == iso
        except Exception:
            return False

    toks = _tokens(text)
    for n in range(1, 5):
        for i in range(len(toks) - n + 1):
            j = i + n - 1
            if parses(i, j):
                while i > 0 and toks[i - 1][0] in _DATE_LEAD \
                        and parses(i - 1, j):
                    i -= 1
                return Grounded(value=iso, quote=text[toks[i][1]:toks[j][2]])
    return None


_TIME_RE = re.compile(
    r"\b(early\s+morning|first\s+flight|last\s+flight|morning|afternoon|"
    r"evening|night|mid-?day|noon)\b", re.IGNORECASE)

_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_PAX_RE = re.compile(
    r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:of\s+us|people|persons|person|adults|adult|passengers|passenger|"
    r"tickets|ticket|seats|seat)\b", re.IGNORECASE)
_PAX_PAIR_RE = re.compile(r"\b(me\s+and\s+my\s+\w+|my\s+wife\s+and\s+i|"
                          r"my\s+husband\s+and\s+i|both\s+of\s+us)\b",
                          re.IGNORECASE)


def _find_passengers(text: str) -> Grounded:
    m = _PAX_RE.search(text)
    if m:
        raw = m.group(1).lower()
        n = int(raw) if raw.isdigit() else _NUM_WORDS[raw]
        if 1 <= n <= 9:
            return Grounded(value=n, quote=m.group(0))
    m = _PAX_PAIR_RE.search(text)
    if m:
        return Grounded(value=2, quote=m.group(0))
    return Grounded(value=1, assumed=True)


def ground(transcript: str, intent=None, today=None) -> VoiceDraft:
    """Build a VoiceDraft where every kept field is backed by heard words.

    `intent` is a brain.Intent from parse_intent(transcript). Its fields
    are KEPT only when grounded; otherwise they go to `dropped` and the
    user is asked. When the brain found nothing, a local fuzzy pass over
    the transcript still tries (ASR spelling drift)."""
    from FareBeep.dates import lagos_today
    today = today or lagos_today()
    draft = VoiceDraft(transcript=transcript)
    if intent is not None and intent.intent in ("book", "fare", "subscribe"):
        draft.intent = "book" if intent.intent == "book" else "fare"

    o_iata = intent.origin_iata if intent is not None else None
    d_iata = intent.destination_iata if intent is not None else None
    if not (o_iata and d_iata):
        # The brain missed a side (often ASR drift: "abuya"). The local
        # read decides sides from route words, so "abuya from lagos" is
        # LOS -> ABV, not the brain's single-city guess.
        fo, fd = _fuzzy_route(transcript)
        if fo and fd:
            o_iata, d_iata = fo, fd
        elif fo:                          # "from X" heard, no "to" side
            o_iata = fo
            d_iata = None if d_iata == fo else d_iata
        elif fd and not o_iata:
            d_iata = fd

    if o_iata:
        g = _find_city(transcript, o_iata)
        if g:
            draft.origin = g
        else:
            draft.dropped.append(("origin", o_iata))
    if d_iata:
        skip = []
        if draft.origin.quote:
            a, b = _span_of(transcript, draft.origin.quote)
            if a >= 0:
                skip.append((a, b))
        g = _find_city(transcript, d_iata, skip_spans=skip)
        if g:
            draft.destination = g
        else:
            draft.dropped.append(("destination", d_iata))
    if draft.origin.value and draft.origin.value == draft.destination.value:
        draft.destination = Grounded()
    # The heard route words outrank word order: "go Kano from Abuja" is
    # ABV -> KAN even when a parser read the cities left to right.
    if _lead(draft.origin.quote) in _TO_WORDS \
            or _lead(draft.destination.quote) in _FROM_WORDS:
        draft.origin, draft.destination = draft.destination, draft.origin

    iso = intent.date if intent is not None else None
    if not iso:
        from FareBeep.dates import parse_expression
        try:
            iso = parse_expression(transcript, today=today)
        except Exception:
            iso = None
    if iso:
        g = _find_date(transcript, iso, today)
        if g:
            draft.date = g
        else:
            draft.dropped.append(("date", iso))

    m = _TIME_RE.search(transcript)
    if m:
        draft.time_of_day = Grounded(value=m.group(1).lower(),
                                     quote=m.group(0))
    draft.passengers = _find_passengers(transcript)
    return draft


# ---------------------------------------------------------------------------
# 4. CONFIRM - what the user sees
# ---------------------------------------------------------------------------
def _pretty_date(iso: str) -> str:
    try:
        d = datetime.strptime(iso, "%Y-%m-%d")
        return d.strftime("%a %d %b").replace(" 0", " ")
    except (TypeError, ValueError):
        return str(iso)


def _line(label: str, shown: str, g: Grounded) -> str:
    if g.assumed:
        return f"• {label}: {shown} (you didn't say, so I assumed)"
    tail = f' - heard "{g.quote}"' if g.quote else ""
    if g.fuzzy:
        tail += " ⚠ check this"
    return f"• {label}: {shown}{tail}"


def _question_for(missing: list[str]) -> str:
    if missing == ["date"]:
        return "What day do you want to fly?"
    if missing == ["origin"]:
        return "Where are you flying from?"
    if missing == ["destination"]:
        return "Where are you flying to?"
    if "origin" in missing and "destination" in missing:
        return ("Where are you flying from and to, and what day? "
                "e.g. 'Lagos to Abuja Friday'")
    words = {"origin": "where you're flying from",
             "destination": "where you're flying to",
             "date": "what day"}
    return "Tell me " + " and ".join(words[m] for m in missing) + "."


def confirmation_text(draft: VoiceDraft) -> str:
    lines = [f'🎙️ I heard: "{draft.transcript}"', ""]
    if draft.origin.value:
        lines.append(_line("From", city_name(draft.origin.value), draft.origin))
    if draft.destination.value:
        lines.append(_line("To", city_name(draft.destination.value),
                           draft.destination))
    if draft.date.value:
        lines.append(_line("Date", _pretty_date(draft.date.value), draft.date))
    if draft.time_of_day.value:
        lines.append(_line("Time", str(draft.time_of_day.value),
                           draft.time_of_day)
                     + " (I'll show every flight that day)")
    if draft.origin.value or draft.destination.value or draft.date.value:
        n = int(draft.passengers.value or 1)
        lines.append(_line("Passengers", str(n), draft.passengers))
    lines.append("")
    if draft.complete:
        lines.append("Reply YES to search, or type what to change.")
    else:
        lines.append(_question_for(draft.missing()))
    return "\n".join(lines)


UNHEARD_TEXT = ("🎙️ I couldn't make out that voice note. Please send it "
                "again or type it, e.g. 'Lagos to Abuja Friday'.")
TYPE_IT_TEXT = ("🎙️ Got your voice note - but for names and payment "
                "answers, please type your reply so nothing gets misheard.")


# ---------------------------------------------------------------------------
# Draft state helpers (chat_state.pending_voice)
# ---------------------------------------------------------------------------
def draft_is_fresh(state: Optional[dict], now=None) -> bool:
    if not state:
        return False
    try:
        created = datetime.fromisoformat(state["created_at"])
    except (KeyError, TypeError, ValueError):
        return False
    now = now or datetime.now(timezone.utc)
    return now - created <= timedelta(minutes=DRAFT_TTL_MINUTES)


def is_yes(text: str) -> bool:
    t = " ".join((text or "").lower().replace("!", " ").replace(".", " ")
                 .split())
    return t in _YES


def is_no(text: str) -> bool:
    t = " ".join((text or "").lower().replace("!", " ").replace(".", " ")
                 .split())
    return t in _NO


def merge_typed_answer(state: dict, text: str) -> Optional[str]:
    """A typed reply to a PARTIAL voice draft ("Abuja", "Friday").

    Returns the combined plain-text request for the normal pipeline, or
    None when the reply is a brand-new request (it names two cities) and
    should be handled on its own. Typed words need no confirmation: the
    user wrote them."""
    from FareBeep.brain import _local_route
    if len(_local_route((text or "").lower()) or []) >= 2:
        return None
    base = state.get("canonical") or ""
    missing = []
    for k in ("origin", "destination"):
        if not (state.get(k) or {}).get("value"):
            missing.append(k)
    t = (text or "").strip()
    one_city = _local_route(t.lower()) or []
    if len(one_city) == 1 and missing:
        city = city_name(one_city[0])
        o = (state.get("origin") or {}).get("value")
        d = (state.get("destination") or {}).get("value")
        rest = re.sub(r"\b(from|to)\b", " ", t, flags=re.IGNORECASE)
        for alias, code in CITY_TO_IATA.items():
            if code == one_city[0]:
                rest = re.sub(r"\b" + re.escape(alias) + r"\b", " ", rest,
                              flags=re.IGNORECASE)
        rest = " ".join(rest.split())
        date = (state.get("date") or {}).get("value") or ""
        if d and not o:
            route = f"{city} to {city_name(d)}"
        elif o and not d:
            route = f"{city_name(o)} to {city}"
        else:
            # neither side known: "from X" fills origin, anything else
            # is where they're going
            route = (f"from {city}" if re.search(r"\bfrom\b", t, re.I)
                     else f"flight to {city}")
        pax = state.get("passengers") or {}
        party = (f"for {pax.get('value')} adults"
                 if not pax.get("assumed") and (pax.get("value") or 1) > 1
                 else "")
        return " ".join(p for p in (route, str(date), rest, party) if p)
    return f"{base} {t}".strip()


# ---------------------------------------------------------------------------
# The turn: one voice note in, one grounded confirmation out
# ---------------------------------------------------------------------------
def _main():
    """FareBeep.main resolved at call time (tests patch its globals)."""
    from FareBeep import main
    return main


def _booking_sensitive(db, phone: str) -> bool:
    """A name, email or yes/no for money is being collected right now."""
    from FareBeep import chatstate
    return bool(chatstate.get_pending_booking(db, phone)
                or chatstate.get_pending_ticket_details(db, phone)
                or chatstate.get_pending_requote(db, phone))


def handle_voice_note(phone: str, audio: Optional[bytes],
                      mime: str = "audio/ogg") -> None:
    """Transcribe, ground, and confirm. Never searches on its own: the
    search runs only after the user's YES (or their typed missing piece),
    through the normal text pipeline."""
    from FareBeep import brain, chatstate
    m = _main()
    if not phone:
        return
    if not engine_order():
        m._say(phone, "🎙️ I can't listen to voice notes yet - please type "
               "it, e.g. 'Lagos to Abuja Friday'.", None, humanized=True)
        return
    if not audio:
        m._say(phone, UNHEARD_TEXT, None, humanized=True)
        return
    if not m._rate_allow(phone, "[voice note]"):
        return
    db = m.SessionLocal()
    try:
        if _booking_sensitive(db, phone):
            m._say(phone, TYPE_IT_TEXT, None, humanized=True)
            return
        try:
            heard = transcribe(audio, mime)
        except TranscriptionError as e:
            logger.warning("Voice note from %s not transcribed: %s", phone, e)
            m._say(phone, UNHEARD_TEXT, None, humanized=True)
            return
        if len(heard.text.strip()) < 2:
            m._say(phone, UNHEARD_TEXT, None, humanized=True)
            return
        intent = brain.parse_intent(heard.text, force_local=m.GUIDED_MODE)
        draft = ground(heard.text, intent)
        logger.info("Voice draft %s engine=%s fields=%s dropped=%s",
                    phone, heard.engine, draft.canonical_text(),
                    draft.dropped)
        if not (draft.origin.value or draft.destination.value):
            # Not a flight search ("my bookings", "talk to a human",
            # a greeting): echo what was heard, then let the normal
            # pipeline take the words like a typed message.
            chatstate.clear_pending_voice(db, phone)
            m._say(phone, f'🎙️ I heard: "{heard.text}"', None, humanized=True)
            db.close()
            db = None
            m._handle_incoming_message(phone, heard.text, _skip_rate=True)
            return
        chatstate.set_pending_voice(db, phone, draft.to_state())
        m._say(phone, confirmation_text(draft), None, humanized=True)
    finally:
        if db is not None:
            db.close()


def try_voice_answer(db, user, text: str) -> bool:
    """dialog gate: the user's typed reply to a pending voice draft.

    YES on a complete draft -> run its canonical text through the normal
    pipeline. NO -> drop it. A typed missing piece ("Abuja", "Friday")
    -> merge and run. A brand-new request -> drop the draft, fall through.
    Returns True when the turn was fully handled here."""
    from FareBeep import chatstate
    state = chatstate.get_pending_voice(db, user.phone)
    if not state:
        return False
    chatstate.clear_pending_voice(db, user.phone)
    if not draft_is_fresh(state):
        return False                     # stale: treat as a fresh message
    m = _main()
    if is_no(text):
        m._say(user.phone, "No wahala - send the voice note again or type "
               "it, e.g. 'Lagos to Abuja Friday'.", user.name, humanized=True)
        return True
    if is_yes(text):
        if state.get("complete") and state.get("canonical"):
            db.commit()
            m._handle_incoming_message(user.phone, state["canonical"],
                                        _skip_rate=True)
            return True
        m._say(user.phone, _question_for(
            [k for k in ("origin", "destination", "date")
             if not (state.get(k) or {}).get("value")]),
            user.name, humanized=True)
        chatstate.set_pending_voice(db, user.phone, state)
        return True
    merged = merge_typed_answer(state, text)
    if merged is None:
        return False                     # new request - normal handling
    m._handle_incoming_message(user.phone, merged, _skip_rate=True)
    return True
