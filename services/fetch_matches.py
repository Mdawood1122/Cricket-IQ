"""
fetch_matches.py

Talks to the CricAPI (cricapi.com) live-data API and hands back
clean, ready-to-use data (mostly pandas DataFrames) for the rest
of the app to consume.

Why a class? Bundling the api_key + base_url as instance state
means every method below can just say `self.api_key` instead of
re-reading the .env file on every call - one load, reused everywhere.
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Reads the .env file in the project root and loads any KEY=VALUE
# lines into the process environment (os.environ). Safe to call
# more than once - it's a no-op if variables are already loaded.
load_dotenv()


class CricketFetcher:
    """Fetches live cricket data from CricAPI and returns clean pandas
    DataFrames instead of raw JSON, so the rest of the app never has
    to deal with CricAPI's response shape directly."""

    def __init__(self):
        # os.getenv reads the CRICAPI_KEY that load_dotenv() put into
        # the environment. Never hardcode the key itself here - it
        # lives only in .env (which is gitignored).
        self.api_key = os.getenv("CRICAPI_KEY")
        if not self.api_key:
            raise ValueError(
                "CRICAPI_KEY not found. Make sure it's set in your .env file."
            )
        self.base_url = "https://api.cricapi.com/v1"

    def get_live_matches(self) -> pd.DataFrame:
        """Fetch every live/current match worldwide.

        Returns a DataFrame with columns: match, type, status, date, venue.
        Returns an empty DataFrame (with those columns) if the API call
        fails or there are no live matches, so callers never have to
        special-case "no data".
        """
        url = f"{self.base_url}/currentMatches"
        params = {"apikey": self.api_key, "offset": 0}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # raises if status code is 4xx/5xx
            payload = response.json()
        except requests.RequestException as e:
            print(f"[CricketFetcher] Network/API error: {e}")
            return pd.DataFrame(columns=["match", "type", "status", "date", "venue"])

        matches = payload.get("data", [])

        match_list = []
        for match in matches:
            match_list.append({
                "match_id": match.get("id"),
                "match": match.get("name"),
                "type": match.get("matchType"),
                "status": match.get("status"),
                "date": match.get("date"),
                "venue": match.get("venue", "Unknown"),
            })

        return pd.DataFrame(match_list, columns=["match_id", "match", "type", "status", "date", "venue"])

    def get_by_type(self, match_type: str) -> pd.DataFrame:
        """Filter live matches down to one format.

        match_type: 't20', 'odi', or 'test' (case-insensitive).
        """
        df = self.get_live_matches()
        if df.empty:
            return df
        return df[df["type"].str.lower() == match_type.lower()].reset_index(drop=True)

    def get_scorecard(self, match_id: str) -> dict:
        """Fetch the full ball-by-ball scorecard for a single match.

        A scorecard has nested structure (innings -> batting/bowling
        lists) that doesn't flatten cleanly into a DataFrame, so this
        returns the parsed JSON dict as-is. The API/dashboard layers
        can pick out what they need from it.
        """
        url = f"{self.base_url}/match_scorecard"
        params = {"apikey": self.api_key, "id": match_id}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as e:
            print(f"[CricketFetcher] Network/API error: {e}")
            return {}

        return payload.get("data", {})


if __name__ == "__main__":
    # Quick manual smoke test: run `python services/fetch_matches.py`
    # from the project root (with the venv active) to sanity-check
    # your API key and see real data.
    fetcher = CricketFetcher()

    print("Live matches worldwide:")
    print(fetcher.get_live_matches())

    print("\nT20 matches only:")
    print(fetcher.get_by_type("t20"))
