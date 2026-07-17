"""
api.py

FastAPI backend for Cricket IQ. Serves live match data (via CricketFetcher)
and ML predictions (win probability, team score, player score) from the
three models trained in models/. Talks to Streamlit over CORS, and pushes
periodic live updates over a WebSocket.

Run with: uvicorn api:app --reload   (from the project root, venv active)
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.fetch_matches import CricketFetcher
from models.win_predictor import WinPredictor
from models.score_predictor import ScorePredictor
from models.player_predictor import PlayerPredictor
from database import SessionLocal, Prediction, init_db

# ---------------- Request schemas ----------------
# These mirror each model's FEATURE_COLUMNS (see models/*.py) so a request
# body maps straight onto what predict_*() expects.

class WinPredictionRequest(BaseModel):
    current_runs: float
    wickets_fallen: float
    overs_bowled: float
    current_run_rate: float
    required_run_rate: float
    wickets_remaining: float
    venue_average: float
    batting_team_form: float
    bowling_team_form: float
    match_type: str = "t20"
    match_id: Optional[str] = None  # only used to tag the saved prediction row


class TeamScoreRequest(BaseModel):
    current_runs: float
    wickets_fallen: float
    overs_bowled: float
    current_run_rate: float
    wickets_remaining: float
    venue_average: float
    batting_depth: float
    death_overs_form: float
    match_type: str = "t20"
    match_id: Optional[str] = None


class PlayerScoreRequest(BaseModel):
    player_form_last5: float
    balls_faced: float
    current_runs: float
    vs_bowler_type: str = "spin_proxy"
    venue: str = ""
    strike_rate: float
    partnership_runs: float
    match_type: str = "t20"


# ---------------- Model loading (once, at startup) ----------------
# Loaded into module-level globals rather than re-loading from disk on
# every request - joblib.load() reads the whole pickled model + its
# lookup tables each time, which is wasteful to do per-request.

fetcher: Optional[CricketFetcher] = None
win_predictor: Optional[WinPredictor] = None
score_predictor: Optional[ScorePredictor] = None
player_predictor: Optional[PlayerPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global fetcher, win_predictor, score_predictor, player_predictor

    init_db()
    fetcher = CricketFetcher()

    try:
        win_predictor = WinPredictor.load()
    except FileNotFoundError:
        print("[api] win_model.pkl not found - run models/win_predictor.py first. "
              "/api/predict/win will 503 until then.")
    try:
        score_predictor = ScorePredictor.load()
    except FileNotFoundError:
        print("[api] score_model.pkl not found - run models/score_predictor.py first. "
              "/api/predict/team-score will 503 until then.")
    try:
        player_predictor = PlayerPredictor.load()
    except FileNotFoundError:
        print("[api] player_model.pkl not found - run models/player_predictor.py first. "
              "/api/predict/player-score will 503 until then.")

    yield  # app runs here

    # nothing to tear down - joblib-loaded models and the fetcher hold no
    # open connections that need closing.


app = FastAPI(title="Cricket IQ API", lifespan=lifespan)

# Streamlit runs on a different port, so it needs CORS to call this API
# from the browser. Wide open here since this is a personal/local project;
# tighten allow_origins before deploying somewhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Live data endpoints ----------------

@app.get("/api/live/matches")
def get_live_matches():
    """All live matches worldwide, across every format."""
    df = fetcher.get_live_matches()
    return {"count": len(df), "matches": df.to_dict(orient="records")}


@app.get("/api/live/scorecard/{match_id}")
def get_scorecard(match_id: str):
    """Full ball-by-ball scorecard for one match."""
    scorecard = fetcher.get_scorecard(match_id)
    if not scorecard:
        raise HTTPException(status_code=404, detail=f"No scorecard found for match_id={match_id}")
    return scorecard


# ---------------- Prediction endpoints ----------------

def _save_prediction(match_id: Optional[str], predicted_score=None, win_pct=None):
    """Every prediction gets logged to the predictions table (actual_score
    and win_actual stay NULL until Match History is reconciled after the
    match ends - see database.py)."""
    if match_id is None:
        return
    db = SessionLocal()
    try:
        db.add(Prediction(
            match_id=match_id,
            predicted_score=predicted_score,
            win_prediction=win_pct,
        ))
        db.commit()
    finally:
        db.close()


@app.post("/api/predict/win")
def predict_win(req: WinPredictionRequest):
    if win_predictor is None:
        raise HTTPException(status_code=503, detail="Win model not loaded - train it first (models/win_predictor.py).")
    state = req.dict(exclude={"match_id"})
    result = win_predictor.predict_win_probability(state)
    _save_prediction(req.match_id, win_pct=result["batting_team_win_pct"])
    return result


@app.post("/api/predict/team-score")
def predict_team_score(req: TeamScoreRequest):
    if score_predictor is None:
        raise HTTPException(status_code=503, detail="Score model not loaded - train it first (models/score_predictor.py).")
    state = req.dict(exclude={"match_id"})
    result = score_predictor.predict_score(state)
    _save_prediction(req.match_id, predicted_score=result["predicted_score"])
    return result


@app.post("/api/predict/player-score")
def predict_player_score(req: PlayerScoreRequest):
    if player_predictor is None:
        raise HTTPException(status_code=503, detail="Player model not loaded - train it first (models/player_predictor.py).")
    return player_predictor.predict_runs(req.dict())


# ---------------- WebSocket: live score + predictions ----------------

@app.websocket("/ws/live/{match_id}")
async def live_match_socket(websocket: WebSocket, match_id: str):
    """Pushes the live scorecard (plus a win-probability prediction when
    there's enough state to compute one) every 10 seconds until the client
    disconnects."""
    await websocket.accept()
    try:
        while True:
            scorecard = fetcher.get_scorecard(match_id)
            payload = {"match_id": match_id, "scorecard": scorecard}

            # Best-effort win-probability push: CricAPI's scorecard shape
            # varies match-to-match, so only attempt this when the fields
            # we need are actually present rather than guessing/crashing.
            try:
                score_entry = scorecard["score"][-1]
                overs = float(score_entry["o"])
                runs = int(score_entry["r"])
                wickets = int(score_entry["w"])
                if win_predictor is not None and overs > 0:
                    state = {
                        "current_runs": runs, "wickets_fallen": wickets,
                        "overs_bowled": overs, "current_run_rate": runs / overs,
                        "required_run_rate": 0, "wickets_remaining": 10 - wickets,
                        "venue_average": 160, "batting_team_form": 0.5,
                        "bowling_team_form": 0.5, "match_type": "t20",
                    }
                    payload["win_prediction"] = win_predictor.predict_win_probability(state)
            except (KeyError, IndexError, TypeError, ValueError):
                pass  # not enough data yet to predict - still send the raw scorecard

            await websocket.send_json(payload)
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        pass
