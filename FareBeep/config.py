"""FareBeep configuration - single source of truth for env values.

Loads .env from the FareBeep directory itself, so the utility is portable:
you can drop the folder anywhere and it reads its own .env.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _get(key: str, default=None):
    return os.getenv(key, default)


def _get_float(key: str, default: float) -> float:
    raw = _get(key)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _get_int(key: str, default: int) -> int:
    raw = _get(key)
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


# --- Supabase ---
SUPABASE_URL = _get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = _get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_DB_URL = _get("SUPABASE_DB_URL", _get("DATABASE_URL"))

# --- Meta WhatsApp Cloud API ---
META_VERIFY_TOKEN = _get("META_VERIFY_TOKEN")
META_ACCESS_TOKEN = _get("META_ACCESS_TOKEN")
META_PHONE_NUMBER_ID = _get("META_PHONE_NUMBER_ID")
META_APP_SECRET = _get("META_APP_SECRET")
META_API_VERSION = _get("META_API_VERSION", "v20.0")
META_TEMPLATE_FLIGHT_STATUS = _get("META_TEMPLATE_FLIGHT_STATUS",
                                   "farebeep_flight_status")
META_TEMPLATE_PRICE_DROP = _get("META_TEMPLATE_PRICE_DROP",
                                "farebeep_price_drop")

# --- Messaging provider switch ---
# "meta" (production path: Meta Cloud API direct, no BSP needed) or
# "twilio" (test path: Twilio WhatsApp Sandbox - one shared number, no
# templates, outbound within the 24h session) or
# "telegram" (fastest test path: plain Bot API - no approval, no sandbox,
# no 24h window; identity is the chat_id). Factory: notifier.get_notifier()
MESSAGING_PROVIDER = _get("MESSAGING_PROVIDER", "meta")
TWILIO_ACCOUNT_SID = _get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = _get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_WHATSAPP = _get("TWILIO_FROM_WHATSAPP")    # "whatsapp:+14155238886"
TELEGRAM_BOT_TOKEN = _get("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_SECRET = _get("TELEGRAM_WEBHOOK_SECRET")
# Long-poll seconds for the tunnel-free poller (FareBeep/poller.py).
# Telegram itself is the transport: no public URL, no cloudflared.
TELEGRAM_POLL_TIMEOUT = int(_get("TELEGRAM_POLL_TIMEOUT", "25"))

# Multiplier applied to every deliberate backoff/reattempt sleep (retry
# backoff, anomaly re-check). 1.0 = production timing; tests set it to 0
# (see conftest.py) so retry loops keep their semantics without real waits.
SLEEP_SCALE = float(_get("SLEEP_SCALE", "1.0"))

# --- Gemini ---
GEMINI_API_KEY = _get("GEMINI_API_KEY")
# GEMINI_MODEL default: "gemini-1.5-flash" is retired (404, confirmed live), and
# "gemini-2.5-flash" is "no longer available to new users" on new keys.
# gemini-flash-latest is the current affordable flash model (verified 200).
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-flash-latest")

# --- FX snapshot (worker.record_fx_rate, the founder's price tracking) ---
# How often the USD->NGN rate is re-fetched (open.er-api.com, free, no key)
# and a snapshot row is recorded for the tracked history. Quoting NEVER
# converts USD: live supplier fares arrive in NGN.
FX_RATE_TTL_HOURS = _get_int("FX_RATE_TTL_HOURS", 12)

# --- Paystack ---
PAYSTACK_SECRET_KEY = _get("PAYSTACK_SECRET_KEY")
PAYSTACK_PUBLIC_KEY = _get("PAYSTACK_PUBLIC_KEY")
PAYSTACK_CALLBACK_URL = _get("PAYSTACK_CALLBACK_URL")

# --- Aviationstack ---
AVIATIONSTACK_API_KEY = _get("AVIATIONSTACK_API_KEY")

# --- HTTP resilience (the defensive integration layer, providers.py) ---
HTTP_TIMEOUT = _get_float("HTTP_TIMEOUT", 20.0)
HTTP_MAX_RETRIES = _get_int("HTTP_MAX_RETRIES", 3)

# --- Fare source provider (providers.get_live_engine) ---
# The USD demo aggregator (SerpApi/Google Flights) is RETIRED. Live fares
# come from the inventory suppliers (Travels247/QuickAir, INVENTORY_PROVIDER)
# through search.SupplierLiveEngine; "ledger_only" answers from the cache.
# "tiqwa"  = the production consolidator engine (FareBeep/tiqwa.py). The
#            client ships once the Tiqwa API token + contract are available.
FARE_PROVIDER = _get("FARE_PROVIDER", "ledger_only")
TIQWA_API_KEY = _get("TIQWA_API_KEY")
TIQWA_ENV = _get("TIQWA_ENV", "sandbox")
TIQWA_BASE_URL = _get(
    "TIQWA_BASE_URL",
    "https://sandbox.tiqwa.com/v1/flight" if TIQWA_ENV == "sandbox"
    else "https://prod.tiqwa.com/v1/flight")

# --- ElevenLabs Conversational AI (the voice/chat brain) ---
# The agent connects to WhatsApp natively (dashboard: Integrations ->
# WhatsApp) and calls /tools/search + /tools/reserve as webhook tools.
# ELEVENLABS_TOOL_SECRET is the shared secret you paste into each tool's
# custom auth headers in the ElevenLabs dashboard (header
# X-FareBeep-Tool-Secret). Unset = tools stay open (dev only).
ELEVENLABS_AGENT_ID = _get("ELEVENLABS_AGENT_ID")
ELEVENLABS_TOOL_SECRET = _get("ELEVENLABS_TOOL_SECRET")

# --- Groq + LangChain (FareBeep's own AI agent - agent.py) ---
# Unset GROQ_API_KEY = the agent stays off and the Gemini brain answers.
GROQ_API_KEY = _get("GROQ_API_KEY")
GROQ_MODEL = _get("GROQ_MODEL", "openai/gpt-oss-120b")
# GUIDED_MODE=1 forces the deterministic brain (no Groq calls at all):
# quota conservation, outage survival, or simply cheaper operation.
# The agent is skipped and parse_intent runs fully offline.
GUIDED_MODE = _get("GUIDED_MODE", "") == "1"

# --- Voice notes (voice.py) ---
# VOICE_ASR: "auto" (NaijaVox when NAIJAVOX_URL is set, Groq Whisper as
# fallback), "naijavox" (only), "groq" (only), or "off" (voice notes get
# a polite "please type it" reply).
VOICE_ASR = _get("VOICE_ASR", "auto")
# Your own NaijaVox-2.0 deployment (Axiveri/NaijaVox-2.0 is not served by
# any serverless provider): a Hugging Face Inference Endpoint URL or any
# server that takes raw audio bytes and returns {"text": "..."}.
NAIJAVOX_URL = _get("NAIJAVOX_URL")
NAIJAVOX_TOKEN = _get("NAIJAVOX_TOKEN")          # HF token for the endpoint
NAIJAVOX_LANGUAGE = _get("NAIJAVOX_LANGUAGE", "")  # pcm|en|yo|ha|ig hint
GROQ_ASR_MODEL = _get("GROQ_ASR_MODEL", "whisper-large-v3")
VOICE_ASR_TIMEOUT = _get_float("VOICE_ASR_TIMEOUT", 45.0)

# --- Primary inventory: 247Travels Travels247 (Xown Solutions) ---
# Search (external) -> Pricing (verify) -> Reserve (PNR). JWT login with
# the partner api-role account; the 900s access token is cached and the
# client re-logs-in before it dies (well under the 10/min/IP limit).
TRAVELS247_BASE_URL = _get("TRAVELS247_BASE_URL", "https://247travels.com/api")
TRAVELS247_EMAIL = _get("TRAVELS247_EMAIL")
TRAVELS247_PASSWORD = _get("TRAVELS247_PASSWORD")
# "live" (default) or "test" - test credentials must never cache fares into
# the Shared Ledger: baselines drive real beeps, and a demo baseline can
# trigger fake alerts. The warmer refuses to run in test mode.
TRAVELS247_MODE = _get("TRAVELS247_MODE", "live")

# --- Inventory supplier (see FareBeep/suppliers.py) ---
# "travels247" (default) or "quickair" - one flip switches the whole
# inventory side; both clients return the same normalized offers.
INVENTORY_PROVIDER = _get("INVENTORY_PROVIDER", "travels247")
QUICKAIR_BASE_URL = _get("QUICKAIR_BASE_URL", "https://quickair.app/api")
QUICKAIR_EMAIL = _get("QUICKAIR_EMAIL")
QUICKAIR_PASSWORD = _get("QUICKAIR_PASSWORD")

# --- Business rules ---
MARKUP_NAIRA = _get_float("MARKUP_NAIRA", 3000.0)
PROCESSING_FEE_RATE = _get_float("PROCESSING_FEE_RATE", 0.015)
# The Settlement Engine's margin: the flat NGN markup charged on every
# booking, over and above the airline fare (Settlement brief: ARHA_MARKUP).
ARHA_MARKUP_NGN = _get_float("ARHA_MARKUP_NGN", 5000.0)
# Real Paystack Nigeria fee: 1.5% + NGN 100, known-fee cap NGN 2,000.
# (pricing rule from http://support.paystack.com/en/articles/2130306)
PAYSTACK_FLAT_FEE_NAIRA = _get_float("PAYSTACK_FLAT_FEE_NAIRA", 100.0)
PAYSTACK_FEE_CAP_NAIRA = _get_float("PAYSTACK_FEE_CAP_NAIRA", 2000.0)
# Who gets the "Refund Required" alert when a payment lands after the
# 10-minute window closed (a phone number or Telegram chat_id as string).
ADMIN_ALERT_PHONE = _get("ADMIN_ALERT_PHONE")
# Ops visibility: /admin/ops + dead-letter replay require the X-Admin-Token
# header to match this shared secret. Unset = the admin surface stays CLOSED
# (404) - flip it on only where you can keep the value secret.
ADMIN_TOKEN = _get("ADMIN_TOKEN")

# --- Anti-abuse -------------------------------------------------------------
# Per-phone inbound throttle: sliding 60-second window. Over the limit the
# user gets ONE polite cooldown note per window, then silence (replying to
# every burst message rewards the burst). STOP/unsubscribe is NEVER
# throttled - opting out must always work, even mid-flood.
RATE_LIMIT_PER_MIN = _get_int("RATE_LIMIT_PER_MIN", 20)

# --- Human support relay ----------------------------------------------------
# Open support threads auto-close after this many hours (marked "stale" in
# ops, not deleted - the transcript survives in the ChatState row).
SUPPORT_TICKET_TTL_HOURS = _get_int("SUPPORT_TICKET_TTL_HOURS", 48)

# --- WhatsApp Flows (structured screens UI) --------------------------------
# The "Set a beep" Flow asset ID from WhatsApp Manager (Flows). Unset =
# the "Set a beep" card button degrades to the plain-text TRACK path.
BEEP_FLOW_ID = _get("BEEP_FLOW_ID")
# "draft" while building in Flow Builder (testable with your own number),
# "published" once the flow is live for everyone.
BEEP_FLOW_MODE = _get("BEEP_FLOW_MODE", "draft")
# RSA private key (PEM) for Flow endpoint encryption. Generate with:
#   openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048
# and upload the PUBLIC half in WhatsApp Manager -> your Flow -> Endpoint.
# Unset = /flow speaks PLAINTEXT (dev only - Meta requires encryption in
# production).
FLOW_PRIVATE_KEY = _get("FLOW_PRIVATE_KEY")
STATUS_WATCH_LEAD_HOURS = _get_int("STATUS_WATCH_LEAD_HOURS", 3)
STATUS_POLL_SECONDS = _get_int("STATUS_POLL_SECONDS", 300)
# APScheduler worker (--scheduled mode): how often the TRACKING checks
# (fare-drop Beeps) run; booking sweep + status watches stay on the fast loop.
TRACKING_POLL_HOURS = _get_int("TRACKING_POLL_HOURS", 4)
# Route warmer: top-route fares refreshed at most this often (minutes).
# Each cycle costs len(routes) x len(days) 247travels searches.
WARM_INTERVAL_MINUTES = _get_int("WARM_INTERVAL_MINUTES", 15)
# Price guardrail: a one-way domestic fare above this (₦NGN) is treated as
# an anomaly - the bot says prices are unusually high instead of quoting it.
FARE_PRICE_GUARDRAIL_NGN = _get_float("FARE_PRICE_GUARDRAIL_NGN", 250000.0)
# The live re-quote tolerance: when BOOK reveals a price that moved by MORE
# than this (₦NGN) vs what the user was shown, the bot re-confirms first
# ("the price moved to ₦X - still lock it?"). Smaller moves are the natural
# volatility buffer - the booking proceeds silently.
REQUOTE_TOLERANCE_NGN = _get_float("REQUOTE_TOLERANCE_NGN", 1000.0)

# --- Public app URL + consent ---
# APP_BASE_URL is the public origin used to build user-facing links (booking
# confirmation page). Set to the Railway URL in production.
APP_BASE_URL = _get("APP_BASE_URL", "http://localhost:8000")
# Version tag of the consent text shown on the booking confirmation page.
# Bump it when the wording changes; users with an older version get asked
# again on their next booking (NDPA re-consent).
CONSENT_VERSION = _get("CONSENT_VERSION", "v2-2026-08-16")
