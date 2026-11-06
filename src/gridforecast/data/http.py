"""One polite, retrying HTTP helper shared by every API client."""

import time

import requests

USER_AGENT = "gb-grid-forecaster (portfolio project)"


def get_json(url: str, params: dict, retries: int = 5, timeout: int = 60):
    """GET a URL and return the parsed JSON, retrying on rate limits and outages.

    Waits 1, 2, 4, 8... seconds between attempts (exponential back-off), which
    is what both Elexon and Open-Meteo ask clients to do after a 429.
    """
    for attempt in range(retries):
        try:
            r = requests.get(
                url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
        except requests.ConnectionError:
            time.sleep(2**attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2**attempt)
            continue
        r.raise_for_status()  # a 400 means our parameters are wrong: fail loudly
        return r.json()
    raise RuntimeError(f"Request failed after {retries} attempts: {url} {params}")
