"""THE CONVERSATIONAL LAYER (outbound) - provider-agnostic WhatsApp sending.

Two channels share one interface (send_text / send_template) so the brain
pipeline in main.py never cares which one is wired:

  meta     - Meta WhatsApp Cloud API direct (the production path; 360dialog /
             Infobip are just resellers of the same API - we need no BSP).
  twilio   - Twilio WhatsApp Sandbox (a test path: one shared sandbox number
             users join with a code; no templates in the sandbox, so
             send_template degrades to a plain text message).
  telegram - Telegram Bot API (the FASTEST test path: no approval, no
             sandbox, no 24-hour window, no templates. Identity = chat_id).

Switch with MESSAGING_PROVIDER in .env, or call get_notifier().
"""
import logging
from datetime import datetime

import httpx

from FareBeep.config import (MESSAGING_PROVIDER, META_ACCESS_TOKEN,
                             META_API_VERSION, META_PHONE_NUMBER_ID,
                             TELEGRAM_BOT_TOKEN, TWILIO_ACCOUNT_SID,
                             TWILIO_AUTH_TOKEN, TWILIO_FROM_WHATSAPP,
                             META_TEMPLATE_FLIGHT_STATUS)

logger = logging.getLogger("farebeep.notifier")

GRAPH_BASE = "https://graph.facebook.com"


_STATUS_LABELS = {
    "scheduled": "ON TIME", "active": "IN THE AIR", "landed": "LANDED",
    "delayed": "DELAYED", "cancelled": "CANCELLED",
    "diverted": "DIVERTED", "incident": "INCIDENT",
}
_STATUS_GUIDANCE = {
    "scheduled": "No changes so far - I'll tell you if that changes.",
    "active": "Your flight is in the air.",
    "landed": "Your flight has landed.",
    "delayed": ("Check the departure board before you leave for the "
                "airport - I'll message you if it moves again."),
    "cancelled": ("Contact your airline to rebook, or tell me your route "
                  "and I'll look for another flight."),
    "diverted": ("Your flight was diverted - check with the airline for "
                 "the new arrival airport."),
    "incident": ("Your flight reported an incident - follow the airline's "
                 "updates before you travel."),
}


def _clock(value):
    """'2026-08-20T08:30:00+01:00' -> '08:30' (None when unparseable)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).strftime("%H:%M")
    except ValueError:
        return None


def _minutes(value):
    """Whole positive delay minutes, or None for on-time / unknown."""
    try:
        m = int(value)
    except (TypeError, ValueError):
        return None
    return m if m > 0 else None


def _duration(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h}h {m:02d}m" if h else f"{m} min"


def format_flight_status(detail: dict) -> str:
    """Human-readable flight status for the passenger (WhatsApp/Telegram).

    `detail` is the dict from AviationstackClient.flight_detail; a bare
    {"flight_iata", "status"} works too (the proactive template only
    carries those two). Missing fields are simply left out.
    """
    d = detail or {}
    code = str(d.get("flight_iata") or "Your flight").upper()
    airline = d.get("airline")
    status = str(d.get("status") or "").strip().lower()
    label = _STATUS_LABELS.get(status, status.upper() or "STATUS UNKNOWN")

    head = f"✈️ {airline + ' ' if airline else ''}{code} - {label}"
    dep_delay = _minutes(d.get("dep_delay_min"))
    if status == "delayed" and dep_delay:
        head += f" by {_duration(dep_delay)}"
    lines = [head]

    dep = _clock(d.get("dep_scheduled"))
    if dep:
        parts = [f"Departs {dep}"]
        dep_now = _clock(d.get("dep_estimated"))
        if dep_now and dep_now != dep:
            parts[0] += f" (now {dep_now})"
        if d.get("dep_terminal"):
            parts.append(f"Terminal {d['dep_terminal']}")
        if d.get("dep_gate"):
            parts.append(f"Gate {d['dep_gate']}")
        lines.append(" · ".join(parts))

    arr = _clock(d.get("arr_scheduled"))
    if arr:
        parts = [f"Arrives {arr}"]
        arr_now = _clock(d.get("arr_estimated"))
        if arr_now and arr_now != arr:
            parts[0] += f" (now {arr_now})"
        if d.get("arr_terminal"):
            parts.append(f"Terminal {d['arr_terminal']}")
        lines.append(" · ".join(parts))

    lines.append(_STATUS_GUIDANCE.get(
        status, "Check with your airline for the latest."))
    return "\n".join(lines)


def _degraded_template_text(template_name: str, parts: list) -> str:
    """Plain-text stand-in for a template on channels without templates.
    Flight-status alerts read as a sentence; anything else keeps the
    generic 'name: params' form."""
    if template_name == META_TEMPLATE_FLIGHT_STATUS and len(parts) >= 2:
        return format_flight_status(
            {"flight_iata": parts[0], "status": parts[1]})
    return f"{template_name}: {' '.join(parts)}"


def get_notifier():
    """Factory - return the transport configured by MESSAGING_PROVIDER."""
    provider = (MESSAGING_PROVIDER or "meta").lower()
    if provider == "twilio":
        return TwilioWhatsapp()
    if provider == "telegram":
        return TelegramBot()
    return MetaWhatsapp()


class MetaWhatsapp:
    """Minimal Meta Cloud API outbound client."""

    def __init__(self, access_token: str = None, phone_number_id: str = None,
                 api_version: str = None, http_client: httpx.Client = None):
        self.access_token = access_token or META_ACCESS_TOKEN
        self.phone_number_id = phone_number_id or META_PHONE_NUMBER_ID
        self.api_version = api_version or META_API_VERSION
        self._http = http_client or httpx.Client(timeout=10.0)

    @property
    def _messages_url(self) -> str:
        return (f"{GRAPH_BASE}/{self.api_version}"
                f"/{self.phone_number_id}/messages")

    def _send(self, to: str, payload: dict, message_type: str) -> bool:
        if not self.access_token or not self.phone_number_id:
            logger.warning(
                "META_ACCESS_TOKEN / META_PHONE_NUMBER_ID not set - "
                "message NOT sent (%s to %s)", message_type, to)
            return False
        payload["messaging_product"] = "whatsapp"
        payload["to"] = to
        try:
            resp = self._http.post(
                self._messages_url,
                headers={"Authorization": f"Bearer {self.access_token}"},
                json=payload)
            resp.raise_for_status()
            logger.info("WhatsApp %s sent to %s", message_type, to)
            return True
        except Exception as e:
            logger.error("WhatsApp %s failed to %s: %s", message_type, to, e)
            return False

    def send_text(self, to: str, body: str) -> bool:
        """Instant message (24-hr window / user-initiated replies)."""
        return self._send(to, {"type": "text",
                               "text": {"body": body}}, "text")

    def send_document(self, to: str, data: bytes, filename: str,
                      caption: str = None,
                      mime: str = "application/pdf") -> bool:
        """Send a file (e.g. the ticket PDF): upload to Meta media first,
        then send the document message referencing the media id."""
        if not self.access_token or not self.phone_number_id:
            logger.warning("META creds not set - document NOT sent to %s", to)
            return False
        try:
            upload = self._http.post(
                f"{GRAPH_BASE}/{self.api_version}"
                f"/{self.phone_number_id}/media",
                headers={"Authorization": f"Bearer {self.access_token}"},
                data={"messaging_product": "whatsapp"},
                files={"file": (filename, data, mime)})
            upload.raise_for_status()
            media_id = upload.json().get("id")
            if not media_id:
                logger.error("Media upload returned no id: %s", upload.text)
                return False
            doc = {"type": "document",
                   "document": {"id": media_id, "filename": filename}}
            if caption:
                doc["document"]["caption"] = caption[:1024]
            return self._send(to, doc, "document")
        except Exception as e:
            logger.error("Document send failed to %s: %s", to, e)
            return False

    def send_typing_indicator(self, to: str) -> bool:
        """Show 'typing...' on the user's WhatsApp (up to ~25s or until
        our reply lands). Best-effort: never raises, never blocks the
        background reply task."""
        return self._send(to, {"type": "typing_indicator",
                               "typing_indicator": {"type": "text"}},
                          "typing_indicator")

    def send_interactive_card(self, to: str, body: str,                                buttons: list,
                                image_url: str = None,
                                footer: str = None) -> bool:
        """Flight card: optional airline-logo image header, price body,
        up to 3 tap buttons. buttons = [(button_id, title), ...] -
        titles are clipped to WhatsApp's 20-char button limit.

        No catalog, no templates, no approvals: works inside the 24h
        user window with live prices. See FareBeep/cards.py.
        """
        interactive = {"type": "button", "body": {"text": body}}
        if image_url:
            interactive["header"] = {"type": "image",
                                     "image": {"link": image_url}}
        if footer:
            interactive["footer"] = {"text": footer}
        interactive["action"] = {"buttons": [
            {"type": "reply",
             "reply": {"id": bid, "title": str(title)[:20]}}
            for bid, title in (buttons or [])[:3]
        ]}
        return self._send(to, {"type": "interactive",
                               "interactive": interactive}, "flight_card")

    def send_flow(self, to: str, flow_cta: str, flow_token: str,
                  flow_id: str = None, flow_mode: str = None,
                  body: str = "Set up your price alert") -> bool:
        """WhatsApp Flow (structured screens). Opens the Meta-hosted
        "Set a beep" form. Requires BEEP_FLOW_ID; returns False (caller
        degrades to text) when unset or the channel is not Meta."""
        from FareBeep.config import BEEP_FLOW_ID, BEEP_FLOW_MODE
        fid = flow_id or BEEP_FLOW_ID
        if not self._ready or not fid:
            return False
        interactive = {
            "type": "flow",
            "body": {"text": body[:1024]},
            "action": {"name": "flow", "parameters": {
                "flow_message_version": "3",
                "flow_token": flow_token,
                "flow_id": fid,
                "flow_cta": flow_cta[:20],
                "flow_action_type": "navigate",
                "flow_action": {"screen": "SET_BEEP_TRIP"},
                "mode": flow_mode or BEEP_FLOW_MODE,
            }},
        }
        return self._send(to, {"type": "interactive",
                               "interactive": interactive}, "flow")

    def send_template(self, to: str, template_name: str,
                      body_parameters: list = None,
                      language: str = "en_US",
                      policy: str = "deterministic",
                      buttons: list = None) -> bool:
        """Proactive WhatsApp Template message (outside the 24h window).

        Used by status.py for status-change pushes (e.g. "Delayed") and
        by alerts.py for price-drop Beeps. The template must already be
        approved in the Meta app, with the exact same name (see
        META_TEMPLATE_* in config).

        buttons: optional per-send quick-reply payloads, positionally
        matched to the template's buttons, e.g. [{"payload": "beep:12"},
        {"payload": "dismiss"}]. Entries may be None to leave a button
        static. Taps return through /webhook/meta as button_reply ids.
        """
        components = []
        if body_parameters:
            components.append({
                "type": "body",
                "parameters": [{"type": "text", "text": p}
                               for p in body_parameters],
            })
        template = {
            "name": template_name,
            "language": {"code": language, "policy": policy},
        }
        if buttons:
            for i, btn in enumerate(buttons):
                if not btn:
                    continue
                components.append({
                    "type": "button",
                    "sub_type": btn.get("sub_type", "quick_reply"),
                    "index": str(i),
                    "parameters": [{"type": "payload",
                                    "payload": btn["payload"]}],
                })
        if components:
            template["components"] = components
        return self._send(to, {"type": "template",
                               "template": template}, "template")


class TwilioWhatsapp:
    """Twilio WhatsApp Sandbox transport - THE TEST-VERSION CHANNEL.

    Sandbox limits accepted for the test build:
      - one shared sandbox number (user must send `join <code>` first)
      - NO approved templates: send_template degrades to send_text
      - outbound only inside the 24-hour window after a user message
    """

    def __init__(self, account_sid: str = None, auth_token: str = None,
                 from_whatsapp: str = None, client=None):
        self.account_sid = account_sid or TWILIO_ACCOUNT_SID
        self.auth_token = auth_token or TWILIO_AUTH_TOKEN
        self.from_whatsapp = from_whatsapp or TWILIO_FROM_WHATSAPP
        self._client = client
        if client is None and self.account_sid and self.auth_token:
            from twilio.rest import Client
            self._client = Client(self.account_sid, self.auth_token)

    @property
    def _ready(self) -> bool:
        return bool(self._client and self.from_whatsapp)

    def send_text(self, to: str, body: str) -> bool:
        """Outbound text via the Twilio REST API (sandbox number)."""
        if not self._ready:
            logger.warning(
                "Twilio not configured (ACCOUNT_SID/AUTH_TOKEN/FROM_WHATSAPP) "
                "- message NOT sent to %s", to)
            return False
        try:
            self._client.messages.create(
                from_=self.from_whatsapp, to=f"whatsapp:{to}", body=body)
            logger.info("Twilio text sent to %s", to)
            return True
        except Exception as e:
            logger.error("Twilio text failed to %s: %s", to, e)
            return False

    def send_template(self, to: str, template_name: str,
                      body_parameters: list = None,
                      language: str = "en_US",
                      policy: str = "deterministic") -> bool:
        """Sandbox has no templates - degrade to a plain-text push."""
        parts = [p if isinstance(p, str) else str(p)
                 for p in (body_parameters or [])]
        body = _degraded_template_text(template_name, parts)
        return self.send_text(to, body)


class TelegramBot:
    """Telegram Bot API transport - THE FASTEST TEST CHANNEL.

    No approval, no sandbox join codes, no 24-hour window, no templates:
    the bot can message anyone who has opened the chat. Identity in the
    FareBeep User model is the chat_id (str), stored where a phone number
    would live on the WhatsApp channels.
    """

    def __init__(self, token: str = None, http_client: httpx.Client = None):
        self.token = token or TELEGRAM_BOT_TOKEN
        self._http = http_client or httpx.Client(timeout=10.0)
        # send_action rides the injected client (tests) or gets its own
        # tight-timeout client on first use
        self._action_http = http_client

    @property
    def _ready(self) -> bool:
        return bool(self.token)

    def _api_url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.token}/{method}"

    def send_text(self, to: str, body: str) -> bool:
        """sendMessage to the chat_id (`to`). Plain mode: no markdown
        escaping needed for fare text with // and currencies."""
        if not self._ready:
            logger.warning("TELEGRAM_BOT_TOKEN not set - message NOT sent to %s",
                           to)
            return False
        try:
            resp = self._http.post(
                self._api_url("sendMessage"),
                json={"chat_id": to, "text": body})
            resp.raise_for_status()
            data = resp.json()
            ok = bool(data.get("ok"))
            logger.info("Telegram message sent to %s (ok=%s)", to, ok)
            return ok
        except Exception as e:
            logger.error("Telegram message failed to %s: %s", to, e)
            return False

    def send_action(self, to: str, action: str = "typing") -> bool:
        """Typing indicator (sendChatAction). Telegram shows it ~5s, so
        call on receipt - the reply lands while it is still visible on
        fast turns. Best-effort: never raises, never blocks a reply.
        Uses a tight timeout (1s connect / 3s total): a dead Telegram
        API must not stall a webhook ack or a poller turn."""
        if not self._ready:
            return False
        if self._action_http is None:
            self._action_http = httpx.Client(
                timeout=httpx.Timeout(3.0, connect=1.0))
        try:
            resp = self._action_http.post(
                self._api_url("sendChatAction"),
                json={"chat_id": to, "action": action})
            resp.raise_for_status()
            return bool(resp.json().get("ok"))
        except Exception:
            return False

    def send_interactive_card(self, to: str, body: str, buttons: list,
                              image_url: str = None,
                              footer: str = None) -> bool:
        """Fare card for Telegram: optional airline-logo photo + body
        text + up to 3 inline keyboard buttons.

        buttons = [(button_id, title), ...] - the SAME tap ids the Meta
        cards emit (pick:N / alert:N / book), so callback routing reuses
        cards.translate_tap and the tested pick/alert gates unchanged.
        Titles are clipped to 64 chars (Telegram's per-button limit).
        The logo rides as sendPhoto's caption (photo albums can't carry
        keyboards); a card without a logo degrades to sendMessage with
        reply_markup. Best-effort like every transport: returns False
        on failure, never raises."""
        if not self._ready:
            logger.warning("TELEGRAM_BOT_TOKEN not set - card NOT sent to %s",
                           to)
            return False
        keyboard = [[{"text": str(title)[:64], "callback_data": str(bid)}]
                    for bid, title in (buttons or [])[:3]]
        markup = {"inline_keyboard": keyboard}
        try:
            if image_url:
                resp = self._http.post(
                    self._api_url("sendPhoto"),
                    json={"chat_id": to, "photo": image_url,
                          "caption": body[:1024],
                          "reply_markup": markup})
            else:
                resp = self._http.post(
                    self._api_url("sendMessage"),
                    json={"chat_id": to, "text": body,
                          "reply_markup": markup})
            resp.raise_for_status()
            ok = bool(resp.json().get("ok"))
            logger.info("Telegram card sent to %s (ok=%s)", to, ok)
            return ok
        except Exception as e:
            logger.error("Telegram card failed to %s: %s", to, e)
            return False

    def answer_callback(self, callback_id: str, text: str = None) -> bool:
        """Acknowledge a callback_query (stops the button spinner) and
        optionally toast a short confirmation. Best-effort."""
        if not self._ready or not callback_id:
            return False
        payload = {"callback_query_id": callback_id}
        if text:
            payload["text"] = text[:200]
        try:
            resp = self._http.post(
                self._api_url("answerCallbackQuery"), json=payload)
            return bool(resp.json().get("ok"))
        except Exception:
            return False

    def send_beep_app(self, to: str, url: str) -> bool:
        """Open the Mini App beep form (the WhatsApp-Flow equivalent): a
        web_app inline button launches the embedded form inside the chat.
        Telegram requires an https URL - callers fall back to the guided
        TRACK path when the app is not on a public origin. Best-effort."""
        if not self._ready or not (url or "").startswith("https://"):
            return False
        try:
            resp = self._http.post(
                self._api_url("sendMessage"),
                json={"chat_id": to,
                      "text": ("🎯 Set a beep - pick the route and we'll "
                               "watch the fares for you."),
                      "reply_markup": {"inline_keyboard": [[
                          {"text": "🎯 Set a beep",
                           "web_app": {"url": url}}]]}})
            resp.raise_for_status()
            ok = bool(resp.json().get("ok"))
            logger.info("Telegram beep-app button sent to %s (ok=%s)", to, ok)
            return ok
        except Exception as e:
            logger.error("Telegram beep-app failed to %s: %s", to, e)
            return False

    def send_template(self, to: str, template_name: str,
                      body_parameters: list = None,
                      language: str = "en_US",
                      policy: str = "deterministic") -> bool:
        """No templates outside WhatsApp - degrade to a plain-text push."""
        parts = [p if isinstance(p, str) else str(p)
                 for p in (body_parameters or [])]
        body = _degraded_template_text(template_name, parts)
        return self.send_text(to, body)