"""Supplier ticketing - turn a PAID booking into a confirmed PNR.

Called by the Paystack webhook (routers/webhooks.py) after settlement. The
supplier is whichever inventory client INVENTORY_PROVIDER selects
(suppliers.get_inventory_client): Travels247 or QuickAir. Both speak the
same contract (verify_price -> reserve -> close), so this module never
branches on the vendor - only on whether that vendor's credentials are set.

Fallback: when the supplier has no credentials, the booking has no stored
token/travellers, or the supplier call fails, the paid booking keeps the
provisional PNR (FB-XXXX) and the user gets a provisional notice. Payment is
never reversed here; late payments are refunded by the expiry path.
"""
import logging
from typing import NamedTuple

from FareBeep.quickair import QuickAirError
from FareBeep.suppliers import get_inventory_client
from FareBeep.travels247 import Travels247Error

logger = logging.getLogger("farebeep.ticketing")

# Every supplier raises its own error type; both share this fallback path.
SUPPLIER_ERRORS = (Travels247Error, QuickAirError)

PROVISIONAL_SUFFIX = " (provisional)"
PENDING_NOTE = "Your e-ticket is issued once the airline confirms availability."


class TicketResult(NamedTuple):
    pnr: str
    note: str
    provisional: str


def supplier_configured(client) -> bool:
    """True when the supplier client has its login credentials."""
    return bool(getattr(client, "email", None)
                and getattr(client, "password", None))


def _count(value, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _fallback(fallback_pnr: str) -> TicketResult:
    return TicketResult(fallback_pnr, PENDING_NOTE, PROVISIONAL_SUFFIX)


async def issue_pnr(session, fallback_pnr: str) -> TicketResult:
    """Turn a paid session into a confirmed PNR via the configured supplier.

    Never raises for supplier problems: any missing configuration, missing
    booking data, or supplier error returns the fallback (mock PNR plus
    provisional notice) so the paid booking is still delivered.
    """
    details = session.flight_details or {}
    token = details.get("booking_token")
    travellers = details.get("travellers")
    sky = get_inventory_client()
    supplier = type(sky).__name__
    try:
        if not supplier_configured(sky):
            logger.info("%s ticketing skipped for %s: credentials not set",
                        supplier, session.payment_ref)
            return _fallback(fallback_pnr)
        if not token or not travellers:
            logger.info("%s ticketing skipped for %s: no token/travellers",
                        supplier, session.payment_ref)
            return _fallback(fallback_pnr)

        pax = details.get("passengers") or {}
        adults = _count(pax.get("adults"), 1)
        children = _count(pax.get("children"), 0)
        infants = _count(pax.get("infants"), 0)
        priced = await sky.verify_price(
            token, adults=adults, children=children, infants=infants)
        res = await sky.reserve(
            priced["booking_token"], travellers,
            adults=adults, children=children, infants=infants)
        logger.info("%s TICKET %s -> PNR %s",
                    supplier, session.payment_ref, res["pnr"])
        deadline = (f" Ticket deadline: {res['ticket_deadline']}."
                    if res.get("ticket_deadline") else "")
        return TicketResult(res["pnr"], f"E-ticket issued.{deadline}", "")
    except SUPPLIER_ERRORS as e:
        logger.error("%s ticketing failed for %s: %s - keeping provisional PNR",
                     supplier, session.payment_ref, e)
        return _fallback(fallback_pnr)
    finally:
        await sky.close()


__all__ = ["TicketResult", "issue_pnr", "supplier_configured"]
