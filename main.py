"""
main.py

Entry point for Cricket IQ. This file doesn't contain any logic of its
own - it just wires together the pieces defined elsewhere (services,
models, database) and starts the API server. All the actual behavior
lives in:

  services/fetch_matches.py  - CricketFetcher (live data from CricAPI)
  models/win_predictor.py    - WinPredictor
  models/score_predictor.py  - ScorePredictor
  models/player_predictor.py - PlayerPredictor
  database.py                 - SQLAlchemy models + init_db()
  api.py                       - FastAPI app that ties all of the above together

Run with: python main.py   (from the project root, venv active)
Then in another terminal:  streamlit run dashboard.py
"""

import uvicorn

from services.fetch_matches import CricketFetcher
from models.win_predictor import WinPredictor
from models.score_predictor import ScorePredictor
from models.player_predictor import PlayerPredictor
from database import init_db
from api import app  # the actual FastAPI app - see api.py for routes + startup logic


def check_setup():
    """Quick sanity check before starting the server: confirms the API key
    is present and reports which of the 3 models have been trained yet.
    Doesn't fail the startup - api.py already handles missing models
    gracefully (those endpoints just 503 until you train them)."""
    print("Cricket IQ - startup check")

    try:
        CricketFetcher()
        print("  [ok] CRICAPI_KEY found in .env")
    except ValueError as e:
        print(f"  [!!] {e}")

    for name, cls in [("win_model.pkl", WinPredictor), ("score_model.pkl", ScorePredictor),
                       ("player_model.pkl", PlayerPredictor)]:
        try:
            cls.load()
            print(f"  [ok] {name} found")
        except FileNotFoundError:
            print(f"  [--] {name} not trained yet")

    init_db()
    print("  [ok] database ready (cricket_iq.db)")


if __name__ == "__main__":
    check_setup()
    print("\nStarting API server at http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
