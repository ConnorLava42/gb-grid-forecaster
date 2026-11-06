"""UK bank holidays for England and Wales, which cover most of GB demand."""

from datetime import date

import holidays

from gridforecast.data.http import get_json


def bank_holidays(years: range) -> set[date]:
    """Bank holidays in England and Wales for the given years.

    The `holidays` package has full history, including one-offs such as the
    2022 Platinum Jubilee and the Queen's funeral, which matter for demand.
    """
    return set(holidays.country_holidays("GB", subdiv="ENG", years=list(years)))


def govuk_bank_holidays() -> set[date]:
    """Official list from gov.uk (recent and upcoming years only), for cross-checking."""
    payload = get_json("https://www.gov.uk/bank-holidays.json", {})
    events = payload["england-and-wales"]["events"]
    return {date.fromisoformat(e["date"]) for e in events}
