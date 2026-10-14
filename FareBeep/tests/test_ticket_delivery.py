"""TICKET DELIVERY - voucher push, email copies, chat ticket-detail ask."""
from FareBeep import chatstate, emailer, ticket_delivery
from FareBeep.models import BookingSession, User
from FareBeep.notifier import TelegramBot


class FakeNotifier:
    def __init__(self):
        self.texts = []
        self.docs = []

    def send_text(self, to, body):
        self.texts.append((to, body))
        return True

    def send_document(self, to, data, filename, caption=None,
                      mime="application/pdf"):
        self.docs.append((to, data, filename))
        return True


class TextOnlyNotifier:
    """A channel without document support (no send_document attribute)."""

    def __init__(self):
        self.texts = []

    def send_text(self, to, body):
        self.texts.append((to, body))
        return True


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeHttp:
    def __init__(self):
        self.calls = []

    def post(self, url, data=None, files=None, json=None):
        self.calls.append({"url": url, "data": data, "files": files})
        return FakeResponse({"ok": True})


def _booking(db, phone="2348144180146", contact_email="ada@example.com",
             passenger="Ada Obi"):
    u = User(phone=phone, name="Ada")
    db.add(u)
    db.commit()
    db.refresh(u)
    b = BookingSession(
        user_id=u.user_id, origin="LOS", destination="ABV",
        flight_date="2026-10-01", payment_ref="FB-ABCD1234",
        total_price=98500.0, airline_price=90000.0, processing_fee=500.0,
        status="paid", passenger_name=passenger, contact_email=contact_email)
    db.add(b)
    db.commit()
    db.refresh(b)
    return u, b


def _capture_email(monkeypatch, configured=True):
    sent = []
    monkeypatch.setattr(emailer, "is_configured", lambda: configured)
    monkeypatch.setattr(
        emailer, "send_email",
        lambda to, subject, body, attachment=None:
            sent.append((to, subject, attachment)) or True)
    return sent


def test_send_ticket_pdf_pushes_document_and_emails(db, monkeypatch):
    sent = _capture_email(monkeypatch)
    _, b = _booking(db)
    fake = FakeNotifier()
    ticket_delivery.send_ticket_pdf(b, "FB-1234", fake)
    assert len(fake.docs) == 1
    to, data, filename = fake.docs[0]
    assert to == "2348144180146"
    assert data.startswith(b"%PDF-")
    assert filename == "FareBeep-FB-1234.pdf"
    assert len(sent) == 1
    assert sent[0][0] == "ada@example.com"
    assert sent[0][2]["filename"] == "FareBeep-FB-1234.pdf"


def test_send_ticket_pdf_without_document_support_still_emails(db, monkeypatch):
    sent = _capture_email(monkeypatch)
    _, b = _booking(db)
    ticket_delivery.send_ticket_pdf(b, "FB-1234", TextOnlyNotifier())
    assert len(sent) == 1


def test_telegram_send_document_uses_multipart_send_document():
    http = FakeHttp()
    bot = TelegramBot(token="123:abc", http_client=http)
    ok = bot.send_document("555", b"%PDF-1.4 x", "FareBeep-FB-1.pdf",
                           caption="Present this at check-in.")
    assert ok is True
    call = http.calls[0]
    assert call["url"].endswith("/bot123:abc/sendDocument")
    assert call["data"]["chat_id"] == "555"
    assert call["data"]["caption"] == "Present this at check-in."
    name, body, mime = call["files"]["document"]
    assert name == "FareBeep-FB-1.pdf" and body == b"%PDF-1.4 x"
    assert mime == "application/pdf"


def test_telegram_send_document_without_token_is_noop():
    http = FakeHttp()
    bot = TelegramBot(token="123:abc", http_client=http)
    bot.token = ""  # the constructor falls back to the env token
    assert bot.send_document("555", b"x", "a.pdf") is False
    assert http.calls == []


def test_email_voucher_skips_without_contact_email(db, monkeypatch):
    sent = _capture_email(monkeypatch)
    _, b = _booking(db, contact_email=None)
    ticket_delivery.email_voucher(b, "FB-1234", b"%PDF-1.4 x")
    assert sent == []


def test_email_voucher_skips_when_unconfigured(db, monkeypatch):
    sent = _capture_email(monkeypatch, configured=False)
    _, b = _booking(db)
    ticket_delivery.email_voucher(b, "FB-1234", b"%PDF-1.4 x")
    assert sent == []


def test_email_boarding_pass_sends_copy(db, monkeypatch):
    sent = _capture_email(monkeypatch)
    _, b = _booking(db)
    b.boarding_pass_blob = b"%PDF-1.4 pass"
    b.boarding_pass_name = "pass.pdf"
    b.boarding_pass_mime = "application/pdf"
    db.commit()
    ticket_delivery.email_boarding_pass(b)
    assert len(sent) == 1
    assert sent[0][2]["content_bytes"] == b"%PDF-1.4 pass"
    assert sent[0][2]["filename"] == "pass.pdf"


def test_dispatch_sends_voucher_when_details_on_file(db, monkeypatch):
    _capture_email(monkeypatch)
    _, b = _booking(db)
    fake = FakeNotifier()
    ticket_delivery.dispatch_ticket(db, b, "FB-1234", fake)
    assert len(fake.docs) == 1
    assert fake.texts == []


def test_dispatch_asks_for_details_when_missing(db, monkeypatch):
    _capture_email(monkeypatch)
    u, b = _booking(db, contact_email=None, passenger=None)
    fake = FakeNotifier()
    ticket_delivery.dispatch_ticket(db, b, "FB-1234", fake)
    assert fake.docs == []
    assert len(fake.texts) == 1
    assert fake.texts[0][0] == u.phone
    pending = chatstate.get_pending_ticket_details(db, u.phone)
    assert pending["stage"] == "name"
    assert pending["pnr"] == "FB-1234"
