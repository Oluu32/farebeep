"""VOICE NOTES - transcribe, ground every field in the heard words, confirm.

Answers, with tests:
  Q1 grounding keeps only fields backed by transcript words (exact alias,
     fuzzy ASR drift, route words decide sides) and drops anything the
     model produced that was never heard.
  Q2 the confirmation shows the transcript and each field beside its
     quote; complete drafts ask for YES, partial ones ask the missing bit.
  Q3 ASR is pluggable: NaijaVox endpoint first, Groq Whisper fallback,
     "off" disables, and every failure is a TranscriptionError.
  Q4 the turn: a voice note never searches by itself; YES runs the
     canonical text through the normal pipeline once (one rate-limit
     hit); typed missing pieces merge; booking-sensitive moments refuse
     voice.
  Q5 channels: WhatsApp audio classifies as AUDIO and dispatches;
     Telegram voice updates are routed to the voice handler.
"""
import json
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from FareBeep import brain, chatstate, config, main, voice
from FareBeep.models import Base
from FareBeep.whatsapp.router import MessageType, classify_message

TODAY = date(2026, 11, 12)        # a Thursday


def _draft(text):
    intent = brain.parse_intent(text, force_local=True)
    return voice.ground(text, intent, today=TODAY)


# -- Q1: grounding -----------------------------------------------------------
def test_pidgin_request_grounds_every_field():
    d = _draft("Abeg find me Lagos to Abuja Friday morning")
    assert (d.origin.value, d.destination.value) == ("LOS", "ABV")
    assert d.origin.quote == "Lagos"
    assert d.destination.quote == "to Abuja"
    assert d.date.value == "2026-11-13" and d.date.quote == "Friday"
    assert d.time_of_day.value == "morning"
    assert d.passengers.assumed and d.passengers.value == 1
    assert d.complete
    assert d.canonical_text() == "Lagos to Abuja 2026-11-13"


def test_asr_spelling_drift_is_fuzzy_matched_and_flagged():
    d = _draft("I wan go abuya from lagos tomorrow")
    assert (d.origin.value, d.destination.value) == ("LOS", "ABV")
    assert d.destination.fuzzy and d.destination.quote == "go abuya"
    assert not d.origin.fuzzy


def test_route_words_beat_word_order():
    d = _draft("two of us dey go Kano from Abuja on the 20th")
    assert (d.origin.value, d.destination.value) == ("ABV", "KAN")
    assert d.date.quote == "on the 20th"
    assert d.passengers.value == 2 and d.passengers.quote == "two of us"
    assert d.canonical_text() == "Abuja to Kano 2026-11-20 for 2 adults"


def test_model_output_never_heard_is_dropped_not_searched():
    """An LLM that 'helpfully' fills Kano (never said) must not search it."""
    fake = brain.Intent(intent="fare", origin="Kano", destination="Abuja",
                        date="2026-11-20")
    d = voice.ground("abeg I wan go Abuja", fake, today=TODAY)
    assert d.origin.value is None
    assert ("origin", "KAN") in d.dropped
    assert ("date", "2026-11-20") in d.dropped
    assert d.destination.value == "ABV"
    assert not d.complete
    assert set(d.missing()) == {"origin", "date"}


def test_partial_draft_asks_for_the_missing_side():
    d = _draft("me and my wife wan fly Port Harcourt next week Monday")
    assert d.destination.value == "PHC" and d.origin.value is None
    assert d.date.quote == "next week Monday"
    assert d.passengers.value == 2
    text = voice.confirmation_text(d)
    assert "Where are you flying from?" in text
    assert "Reply YES" not in text


# -- Q2: confirmation ----------------------------------------------------------
def test_confirmation_shows_transcript_and_quotes():
    text = voice.confirmation_text(_draft("I wan go abuya from lagos tomorrow"))
    assert text.startswith('🎙️ I heard: "I wan go abuya from lagos tomorrow"')
    assert '• From: Lagos - heard "from lagos"' in text
    assert "⚠ check this" in text
    assert "Reply YES to search" in text


# -- Q3: pluggable ASR ---------------------------------------------------------
def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_naijavox_first_then_groq_fallback(monkeypatch):
    monkeypatch.setattr(config, "VOICE_ASR", "auto")
    monkeypatch.setattr(config, "NAIJAVOX_URL", "https://nv.example/asr")
    monkeypatch.setattr(config, "NAIJAVOX_TOKEN", "hf_test")
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test")
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "nv.example":
            return httpx.Response(503, json={"error": "loading"})
        assert request.headers["authorization"] == "Bearer gsk_test"
        return httpx.Response(200, json={"text": " Lagos  to Abuja "})

    t = voice.transcribe(b"OggS...", "audio/ogg; codecs=opus",
                         http_client=_client(handler))
    assert calls == ["nv.example", "api.groq.com"]
    assert (t.text, t.engine) == ("Lagos to Abuja", "groq")


def test_naijavox_list_response_and_auth(monkeypatch):
    monkeypatch.setattr(config, "VOICE_ASR", "naijavox")
    monkeypatch.setattr(config, "NAIJAVOX_URL", "https://nv.example/asr")
    monkeypatch.setattr(config, "NAIJAVOX_TOKEN", "hf_test")

    def handler(request):
        assert request.headers["authorization"] == "Bearer hf_test"
        assert request.headers["content-type"] == "audio/ogg"
        assert request.content == b"OggS..."
        return httpx.Response(200, json=[{"text": "Abuja tomorrow"}])

    t = voice.transcribe(b"OggS...", "audio/ogg", http_client=_client(handler))
    assert (t.text, t.engine) == ("Abuja tomorrow", "naijavox")


def test_off_and_unconfigured_raise(monkeypatch):
    monkeypatch.setattr(config, "VOICE_ASR", "off")
    assert voice.engine_order() == []
    with pytest.raises(voice.TranscriptionError):
        voice.transcribe(b"x")
    monkeypatch.setattr(config, "VOICE_ASR", "auto")
    monkeypatch.setattr(config, "NAIJAVOX_URL", None)
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    assert voice.engine_order() == []
    with pytest.raises(voice.TranscriptionError):
        voice.transcribe(b"x")


def test_oversized_audio_rejected(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test")
    with pytest.raises(voice.TranscriptionError):
        voice.transcribe(b"0" * (voice.MAX_AUDIO_BYTES + 1))


# -- Q4: the turn --------------------------------------------------------------
@pytest.fixture
def turn(monkeypatch):
    engine = create_engine("sqlite://",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(main, "SessionLocal", Session)
    monkeypatch.setattr(main, "GUIDED_MODE", True)
    monkeypatch.setattr(config, "VOICE_ASR", "groq")
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test")
    said, turns = [], []
    monkeypatch.setattr(main, "_say",
                        lambda phone, msg, name=None, humanized=False:
                        said.append(msg))
    real = main._handle_incoming_message

    def spy(phone, text, _skip_rate=False):
        turns.append((text, _skip_rate))
        if text.lower() in ("yes", "no", "abuja", "friday", "lagos to kano tomorrow"):
            return real(phone, text, _skip_rate=_skip_rate)
    monkeypatch.setattr(main, "_handle_incoming_message", spy)

    def heard(text):
        monkeypatch.setattr(voice, "transcribe",
                            lambda audio, mime="audio/ogg", **k:
                            voice.Transcript(text=text, engine="groq"))
    return type("T", (), {"said": said, "turns": turns, "heard": heard,
                          "Session": Session, "spy": spy})


def test_voice_note_confirms_and_never_searches_alone(turn):
    turn.heard("Abeg find me Lagos to Abuja Friday morning")
    voice.handle_voice_note("2348000000001", b"OggS")
    assert len(turn.said) == 1 and "Reply YES" in turn.said[0]
    assert turn.turns == []                       # no search yet
    db = turn.Session()
    state = chatstate.get_pending_voice(db, "2348000000001")
    db.close()
    assert state["complete"] and state["canonical"].startswith("Lagos to Abuja")


def test_yes_runs_canonical_text_once_without_second_rate_hit(turn):
    turn.heard("Abeg find me Lagos to Abuja Friday morning")
    voice.handle_voice_note("2348000000002", b"OggS")
    turn.spy("2348000000002", "yes")
    assert turn.turns[0] == ("yes", False)
    canon = [t for t in turn.turns[1:]]
    assert len(canon) == 1
    assert canon[0][0].startswith("Lagos to Abuja 20")
    assert canon[0][1] is True                     # already counted
    db = turn.Session()
    assert chatstate.get_pending_voice(db, "2348000000002") is None
    db.close()


def test_no_drops_the_draft(turn):
    turn.heard("Lagos to Abuja tomorrow")
    voice.handle_voice_note("2348000000003", b"OggS")
    turn.spy("2348000000003", "no")
    assert "No wahala" in turn.said[-1]
    assert [t[0] for t in turn.turns] == ["no"]


def test_typed_missing_city_merges_into_the_heard_request(turn):
    turn.heard("I wan fly go Abuja on Friday")
    voice.handle_voice_note("2348000000004", b"OggS")
    assert "Where are you flying from?" in turn.said[-1]
    turn.turns.clear()
    turn.spy("2348000000004", "Lagos to Kano tomorrow")   # brand-new request
    assert [t[0] for t in turn.turns] == ["Lagos to Kano tomorrow"]


def test_typed_origin_completes_partial_draft():
    state = {"canonical": "flight to Abuja 2026-11-13",
             "origin": {"value": None}, "destination": {"value": "ABV"},
             "date": {"value": "2026-11-13"},
             "passengers": {"value": 2, "assumed": False}}
    merged = voice.merge_typed_answer(state, "Lagos")
    assert merged == "Lagos to Abuja 2026-11-13 for 2 adults"
    assert voice.merge_typed_answer(state, "Kano to Enugu") is None


def test_stale_draft_is_ignored():
    old = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    assert not voice.draft_is_fresh({"created_at": old})
    assert voice.draft_is_fresh(
        {"created_at": datetime.now(timezone.utc).isoformat()})


def test_non_search_voice_goes_through_normal_pipeline(turn):
    turn.heard("check my booking")
    voice.handle_voice_note("2348000000005", b"OggS")
    assert turn.said[0] == '🎙️ I heard: "check my booking"'
    assert turn.turns == [("check my booking", True)]


def test_booking_sensitive_state_refuses_voice(turn):
    db = turn.Session()
    chatstate.set_pending_booking(db, "2348000000006",
                                  {"stage": "name", "origin_iata": "LOS"})
    db.close()
    turn.heard("Tunde Bakare")
    voice.handle_voice_note("2348000000006", b"OggS")
    assert turn.said == [voice.TYPE_IT_TEXT]
    assert turn.turns == []


def test_failed_transcription_asks_to_type(turn, monkeypatch):
    def boom(*a, **k):
        raise voice.TranscriptionError("down")
    monkeypatch.setattr(voice, "transcribe", boom)
    voice.handle_voice_note("2348000000007", b"OggS")
    assert turn.said == [voice.UNHEARD_TEXT]


def test_voice_off_replies_politely(turn, monkeypatch):
    monkeypatch.setattr(config, "VOICE_ASR", "off")
    voice.handle_voice_note("2348000000008", b"OggS")
    assert "can't listen to voice notes yet" in turn.said[0]


# -- Q5: channels --------------------------------------------------------------
def test_whatsapp_audio_classifies_as_audio():
    entry = {"changes": [{"value": {"messages": [{
        "id": "wamid.V1", "from": "2348000000009", "timestamp": "1",
        "type": "audio",
        "audio": {"id": "MEDIA1", "mime_type": "audio/ogg; codecs=opus",
                  "voice": True}}]}}]}
    [msg] = classify_message(entry)
    assert msg.message_type == MessageType.AUDIO
    assert msg.media_id == "MEDIA1"
    assert msg.media_mime.startswith("audio/ogg")


def test_whatsapp_audio_dispatches_to_voice(monkeypatch):
    from FareBeep import inbound, interactive
    got = {}
    monkeypatch.setattr(interactive, "_download_whatsapp_media",
                        lambda mid: (b"OggS", "audio/ogg"))
    monkeypatch.setattr(voice, "handle_voice_note",
                        lambda phone, audio, mime="audio/ogg":
                        got.update(phone=phone, audio=audio, mime=mime))
    entry = {"changes": [{"value": {"messages": [{
        "id": "wamid.V2", "from": "2348000000010", "timestamp": "1",
        "type": "audio", "audio": {"id": "MEDIA2", "mime_type": "audio/ogg"}}]}}]}
    [msg] = classify_message(entry)
    inbound._dispatch_single(msg)
    assert got == {"phone": "2348000000010", "audio": b"OggS",
                   "mime": "audio/ogg"}


def test_telegram_voice_update_routes_to_voice_handler(monkeypatch):
    from FareBeep import interactive
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "s3cret")
    seen = []
    monkeypatch.setattr(interactive, "_handle_telegram_voice",
                        lambda chat, fid, mime: seen.append((chat, fid, mime)))
    monkeypatch.setattr(main, "_telegram_typing", lambda chat: None)
    client = TestClient(main.app)
    r = client.post("/webhook/telegram",
                    headers={"X-Telegram-Bot-Api-Secret-Token": "s3cret"},
                    json={"message": {"chat": {"id": 42},
                                      "voice": {"file_id": "F1",
                                                "mime_type": "audio/ogg"}}})
    assert r.status_code == 200
    assert seen == [("42", "F1", "audio/ogg")]
