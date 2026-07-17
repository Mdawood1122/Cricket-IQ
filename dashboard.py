"""
dashboard.py

Streamlit frontend for Cricket IQ. Talks to the FastAPI backend (api.py)
over HTTP/JSON - run both at once:

    uvicorn api:app --reload          (in one terminal)
    streamlit run dashboard.py        (in another)

Five pages, picked from the sidebar: Live Scores, Win Probability,
Score Prediction, Player Predictions, Match History.
"""

import os
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from database import SessionLocal, Prediction

# Resolution order for where the API lives:
#   1. Streamlit Cloud secret (Settings -> Secrets -> API_BASE_URL = "...")
#   2. CRICKET_IQ_API_URL environment variable (handy for Render/Docker)
#   3. localhost, for running everything on one machine
try:
    API_BASE = st.secrets["API_BASE_URL"]
except (KeyError, FileNotFoundError):
    API_BASE = os.getenv("CRICKET_IQ_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Cricket IQ", page_icon="🏏", layout="wide")


# ---------------- Small shared helpers ----------------

def api_get(path, **kwargs):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Couldn't reach the API at {API_BASE}{path} - is `uvicorn api:app` running? ({e})")
        return None


def api_post(path, payload):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Prediction request failed: {e}")
        return None


# ---------------- Page 1: Live Scores ----------------

def page_live_scores():
    st.title("🔴 Live Scores")

    fmt = st.selectbox("Format", ["All", "T20", "ODI", "Test"])
    auto = st.checkbox("Auto-refresh every 30s", value=False)

    data = api_get("/api/live/matches")
    if data is None:
        return

    matches = data["matches"]
    if fmt != "All":
        matches = [m for m in matches if str(m.get("type", "")).lower() == fmt.lower()]

    st.caption(f"{len(matches)} live match(es) worldwide" + (f" ({fmt})" if fmt != "All" else ""))

    if not matches:
        st.info("No live matches right now - check back later, or try a different format filter.")
    else:
        df = pd.DataFrame(matches)
        cols = [c for c in ["match", "type", "status", "venue", "date"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

    if auto:
        # Lightweight polling refresh - blocks the script for 30s then
        # reruns. Fine for a personal dashboard; swap in the
        # streamlit-autorefresh package if you want a non-blocking version.
        time.sleep(30)
        st.rerun()


# ---------------- Page 2: Win Probability ----------------

def _gauge(pct, label):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        title={"text": label},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "darkgreen" if pct >= 50 else "darkred"}},
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def page_win_probability():
    st.title("📊 Win Probability")
    st.caption(
        "Uses the trained WinPredictor model. Since CricAPI's free tier "
        "doesn't expose venue history or team form directly, those two "
        "inputs default to neutral placeholders below - tune them if you "
        "know the ground and teams well."
    )

    data = api_get("/api/live/matches")
    if not data or not data["matches"]:
        st.info("No live matches to pick from right now.")
        return

    options = {m["match"]: m for m in data["matches"] if m.get("match_id")}
    choice = st.selectbox("Select a live match", list(options.keys()))
    match = options[choice]
    match_id = match["match_id"]

    col1, col2 = st.columns(2)
    with col1:
        current_runs = st.number_input("Current runs (chasing team)", 0, 500, 90)
        wickets_fallen = st.number_input("Wickets fallen", 0, 10, 3)
        overs_bowled = st.number_input("Overs bowled", 0.0, 50.0, 12.0)
        target = st.number_input("Target", 1, 600, 180)
    with col2:
        match_type = st.selectbox("Match type", ["t20", "odi"])
        venue_average = st.number_input("Venue average score", 50, 400, 160)
        batting_team_form = st.slider("Batting team form (win rate)", 0.0, 1.0, 0.5)
        bowling_team_form = st.slider("Bowling team form (win rate)", 0.0, 1.0, 0.5)

    total_balls = 120 if match_type == "t20" else 300
    balls_bowled = int(overs_bowled * 6)
    balls_left = max(total_balls - balls_bowled, 1)
    runs_left = max(target - current_runs, 0)

    if st.button("Get win probability"):
        payload = {
            "current_runs": current_runs, "wickets_fallen": wickets_fallen,
            "overs_bowled": overs_bowled,
            "current_run_rate": current_runs / max(overs_bowled, 0.1),
            "required_run_rate": runs_left / (balls_left / 6),
            "wickets_remaining": 10 - wickets_fallen,
            "venue_average": venue_average,
            "batting_team_form": batting_team_form,
            "bowling_team_form": bowling_team_form,
            "match_type": match_type,
            "match_id": match_id,
        }
        result = api_post("/api/predict/win", payload)
        if result:
            c1, c2 = st.columns(2)
            c1.plotly_chart(_gauge(result["batting_team_win_pct"], "Batting (chasing) team"), use_container_width=True)
            c2.plotly_chart(_gauge(result["bowling_team_win_pct"], "Bowling (defending) team"), use_container_width=True)

            # Track over-by-over history in session state so the chart
            # builds up as you fetch predictions across an innings.
            key = f"win_history_{match_id}"
            history = st.session_state.setdefault(key, [])
            history.append({"over": overs_bowled, "batting_team_win_pct": result["batting_team_win_pct"]})
            st.session_state[key] = history

    key = f"win_history_{match_id}"
    if st.session_state.get(key):
        st.subheader("Win probability over time")
        hist_df = pd.DataFrame(st.session_state[key])
        st.line_chart(hist_df.set_index("over"))


# ---------------- Page 3: Score Prediction ----------------

def page_score_prediction():
    st.title("🎯 Score Prediction")

    with st.form("score_form"):
        col1, col2 = st.columns(2)
        with col1:
            current_runs = st.number_input("Current runs", 0, 500, 90)
            wickets_fallen = st.number_input("Wickets fallen", 0, 10, 3)
            overs_bowled = st.number_input("Overs bowled", 0.0, 50.0, 12.0)
            match_type = st.selectbox("Match type", ["t20", "odi"])
        with col2:
            venue_average = st.number_input("Venue average score", 50, 400, 160)
            batting_depth = st.slider("Batting depth (avg distinct batters used)", 5.0, 11.0, 8.0)
            death_overs_form = st.number_input("Death-overs run rate (team historical)", 0.0, 20.0, 9.0)

        submitted = st.form_submit_button("Predict final score")

    if submitted:
        payload = {
            "current_runs": current_runs, "wickets_fallen": wickets_fallen,
            "overs_bowled": overs_bowled,
            "current_run_rate": current_runs / max(overs_bowled, 0.1),
            "wickets_remaining": 10 - wickets_fallen,
            "venue_average": venue_average,
            "batting_depth": batting_depth,
            "death_overs_form": death_overs_form,
            "match_type": match_type,
        }
        result = api_post("/api/predict/team-score", payload)
        if result:
            st.metric("Predicted final score", result["predicted_score"],
                       help=f"Range: {result['min_score']}-{result['max_score']}")
            st.caption(f"Likely range: {result['min_score']} - {result['max_score']}")

            st.subheader("What influenced this prediction")
            st.caption(
                "Feature importances are learned from the training data as a "
                "whole (how much each factor matters on average across all "
                "matches), not recomputed per-prediction - RandomForest "
                "doesn't natively explain individual predictions without an "
                "extra tool like SHAP."
            )
            # Loaded directly from the saved model rather than through a new
            # API endpoint - feature_importances() is a property of the
            # trained forest, not something that varies per request.
            try:
                from models.score_predictor import ScorePredictor
                sp = ScorePredictor.load()
                imp_df = pd.DataFrame(
                    list(sp.feature_importances().items()), columns=["feature", "importance"]
                )
                st.bar_chart(imp_df.set_index("feature"))
            except FileNotFoundError:
                st.info("Train the score model first (models/score_predictor.py) to see feature importances.")


# ---------------- Page 4: Player Predictions ----------------

def page_player_predictions():
    st.title("🏏 Player Predictions")
    st.caption(
        "Pulls the current squad/scorecard for a live match, then runs each "
        "player still batting through the PlayerPredictor model."
    )

    data = api_get("/api/live/matches")
    if not data or not data["matches"]:
        st.info("No live matches to pick from right now.")
        return

    options = {m["match"]: m for m in data["matches"] if m.get("match_id")}
    choice = st.selectbox("Select a live match", list(options.keys()))
    match_id = options[choice]["match_id"]

    scorecard = api_get(f"/api/live/scorecard/{match_id}")
    if not scorecard:
        return

    rows = []
    for inning in scorecard.get("scorecard", []):
        for batter in inning.get("batting", []):
            if not batter.get("dismissal") and int(batter.get("r", 0) or 0) >= 0:
                payload = {
                    "player_form_last5": 30.0,  # no historical form available from CricAPI's free tier
                    "balls_faced": int(batter.get("b", 0) or 0),
                    "current_runs": int(batter.get("r", 0) or 0),
                    "vs_bowler_type": "pace_proxy",
                    "venue": scorecard.get("venue", ""),
                    "strike_rate": float(batter.get("sr", 0) or 0),
                    "partnership_runs": int(batter.get("r", 0) or 0),
                    "match_type": "t20",
                }
                pred = api_post("/api/predict/player-score", payload)
                rows.append({
                    "player": batter.get("batsman", {}).get("name", "Unknown"),
                    "current_score": payload["current_runs"],
                    "predicted_final": pred["predicted_runs"] if pred else None,
                    "confidence_pct": pred["confidence_pct"] if pred else None,
                })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No batters currently in - the scorecard may not have started yet, or the innings has ended.")


# ---------------- Page 5: Match History ----------------

def page_match_history():
    st.title("📈 Match History")

    db = SessionLocal()
    try:
        predictions = db.query(Prediction).order_by(Prediction.timestamp.desc()).all()
    finally:
        db.close()

    if not predictions:
        st.info("No predictions logged yet - use the Win Probability or Score Prediction pages first.")
        return

    df = pd.DataFrame([{
        "id": p.id, "match_id": p.match_id, "predicted_score": p.predicted_score,
        "actual_score": p.actual_score, "win_prediction_pct": p.win_prediction,
        "win_actual": p.win_actual, "timestamp": p.timestamp,
    } for p in predictions])

    st.subheader("Predictions vs actuals")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Model accuracy")
    scored = df.dropna(subset=["predicted_score", "actual_score"])
    if not scored.empty:
        scored = scored.copy()
        scored["error"] = (scored["predicted_score"] - scored["actual_score"]).abs()
        st.metric("Avg score prediction error (runs)", round(scored["error"].mean(), 1))
        st.subheader("Score prediction error over time")
        fig = px.line(scored.sort_values("timestamp"), x="timestamp", y="error", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption(
            "No resolved score predictions yet - actual_score gets filled in "
            "once a match finishes and results are reconciled (not automated "
            "yet; update database.py Prediction rows manually or add a "
            "reconciliation job once you have a reliable 'match ended' signal)."
        )

    win_scored = df.dropna(subset=["win_prediction_pct", "win_actual"])
    if not win_scored.empty:
        win_scored = win_scored.copy()
        win_scored["correct"] = (
            (win_scored["win_prediction_pct"] >= 50) == (win_scored["win_actual"] == 1)
        )
        st.metric("Win prediction accuracy", f"{win_scored['correct'].mean():.1%}")


# ---------------- Sidebar navigation ----------------

PAGES = {
    "Live Scores": page_live_scores,
    "Win Probability": page_win_probability,
    "Score Prediction": page_score_prediction,
    "Player Predictions": page_player_predictions,
    "Match History": page_match_history,
}

st.sidebar.title("🏏 Cricket IQ")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[selection]()
