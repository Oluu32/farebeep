"""Classify inbound WhatsApp events and detect commands."""
import json, logging
from enum import Enum
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

class MessageType(str, Enum):
    TEXT = "text"
    BUTTON_REPLY = "button_reply"
    LIST_REPLY = "list_reply"
    FLOW_RESPONSE = "flow_response"
    INTERACTIVE = "interactive"
    DOCUMENT = "document"
    AUDIO = "audio"
    UNSUPPORTED = "unsupported"
    STATUS_UPDATE = "status_update"
    UNKNOWN = "unknown"

@dataclass
class InboundMessage:
    message_id: str
    from_number: str
    message_type: MessageType
    timestamp: str
    raw: dict[str, Any]
    text: str | None = None
    button_id: str | None = None
    button_title: str | None = None
    list_id: str | None = None
    list_title: str | None = None
    flow_token: str | None = None
    flow_data: dict[str, Any] | None = None
    media_id: str | None = None
    media_filename: str | None = None
    media_mime: str | None = None

def classify_message(entry: dict[str, Any]) -> list[InboundMessage]:
    messages = []
    for change in entry.get("changes", []):
        value = change.get("value", {})
        if "statuses" in value and "messages" not in value:
            continue
        for msg in value.get("messages", []):
            parsed = _parse(msg)
            if parsed:
                messages.append(parsed)
    return messages

def _parse(msg: dict) -> InboundMessage | None:
    base = InboundMessage(
        message_id=msg.get("id", ""),
        from_number=msg.get("from", ""),
        message_type=MessageType.UNKNOWN,
        timestamp=msg.get("timestamp", ""),
        raw=msg,
    )
    t = msg.get("type", "") or ""
    if t == "text" or (not t and isinstance(msg.get("text"), dict)
                       and "body" in msg["text"]):
        base.message_type = MessageType.TEXT
        base.text = msg.get("text", {}).get("body", "").strip()
        return base
    inter = msg.get("interactive", {})
    if t == "interactive" or (not t and isinstance(inter, dict) and inter):
        it = inter.get("type", "")
        if it == "button_reply" or (not it and "button_reply" in inter):
            r = inter.get("button_reply", {})
            base.message_type = MessageType.BUTTON_REPLY
            base.button_id = r.get("id", "")
            base.button_title = r.get("title", "")
            return base
        if it == "list_reply" or (not it and "list_reply" in inter):
            r = inter.get("list_reply", {})
            base.message_type = MessageType.LIST_REPLY
            base.list_id = r.get("id", "")
            base.list_title = r.get("title", "")
            return base
        if it == "nfm_reply":
            nfm = inter.get("nfm_reply", {})
            base.message_type = MessageType.FLOW_RESPONSE
            base.flow_token = nfm.get("flow_token", "")
            try:
                base.flow_data = json.loads(nfm.get("response_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                base.flow_data = {}
            return base
        base.message_type = MessageType.INTERACTIVE
        return base
    if t in ("document", "image"):
        # Boarding-pass capture: airlines hand out PDFs (document) and
        # users screenshot them (image) - both are storable media.
        media = msg.get(t, {})
        base.message_type = MessageType.DOCUMENT
        base.media_id = media.get("id", "")
        base.media_filename = (media.get("filename")
                               or media.get("caption") or "boarding-pass")
        base.media_mime = media.get("mime_type", "")
        return base
    if t == "audio":
        # Voice notes (audio.voice=true) and forwarded audio files: both
        # go to the voice pipeline (FareBeep.voice).
        media = msg.get("audio", {}) or {}
        base.message_type = MessageType.AUDIO
        base.media_id = media.get("id", "")
        base.media_mime = media.get("mime_type", "")
        return base
    if t == "video":
        base.message_type = MessageType.UNSUPPORTED
        return base
    return base

COMMANDS = {
    "menu": "menu", "help": "help", "start": "menu",
    "hi": "menu", "hello": "menu", "stop": "stop",
    "unsubscribe": "stop", "my watches": "my_watches",
    "my beeps": "my_watches", "my account": "my_account",
    "my bookings": "my_bookings",
}

def detect_command(text: str) -> str | None:
    return COMMANDS.get(text.strip().lower())
