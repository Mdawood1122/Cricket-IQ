"""
score_predictor.py

ScorePredictor estimates a batting team's final innings total from an
in-progress innings, using Cricsheet ball-by-ball data.

Unlike WinPredictor (which only makes sense for the 2nd innings), this
trains on EVERY innings - there's always a "final score" to predict,
whether you're batting first or chasing.

Algorithm: RandomForestRegressor, tuned with BayesSearchCV.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Integer, Real

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "saved", "score_model.pkl")

FEATURE_COLUMNS = [
    "current_runs", "wickets_fallen", "overs_bowled", "current_run_rate",
    "wickets_remaining", "venue_average", "batting_depth",
    "death_overs_form", "match_type_encoded",
]

OVERS_BY_FORMAT = {"t20s": 20, "odis": 50}

# A rough score range around the point prediction, sized as a fraction of
# the prediction and widened for how early in the innings we are (more
# overs left = more uncertainty). This isn't a proper prediction interval
# (that would need e.g. quantile regression forests) but gives dashboard
# users a sane min/max band without a second model.
def _score_range(predicted, overs_bowled, total_overs):
    overs_left_frac = max(total_overs - overs_bowled, 0) / total_overs
    spread = predicted * (0.06 + 0.18 * overs_left_frac)
    return max(predicted - spread, predicted * 0.5), predicted + spread


class ScorePredictor:
    """Predicts a batting team's final innings total (+ a min/max range)
    from the current state of an in-progress innings."""

    def __init__(self):
        self.model = None
        self.match_type_encoder = LabelEncoder()
        self.venue_avg_lookup = {}
        self.team_depth_lookup = {}  # proxy for "batting depth"
        self.team_death_form_lookup = {}  # proxy for "death overs form"

    # ---------------- Feature engineering ----------------

    def build_training_data(self, competitions=("t20s", "odis"), sample_size=60000, random_state=42):
        """One row per ball of every innings, labelled with that innings'
        eventual final score. sample_size caps rows for training-time
        feasibility (RandomForest scales much better than SVC, so this
        cap is looser than WinPredictor's, but full ball-by-ball data
        across all matches is still tens of millions of rows).
        """
        all_rows = []

        for comp in competitions:
            deliveries = pd.read_csv(os.path.join(DATA_DIR, comp, "deliveries.csv.gz"))
            matches = pd.read_csv(os.path.join(DATA_DIR, comp, "matches.csv.gz"))
            matches = matches.dropna(subset=["team1", "team2"])

            deliveries["total_runs"] = deliveries["runs_off_bat"].fillna(0) + deliveries["extras"].fillna(0)
            deliveries["is_wicket"] = deliveries["player_dismissed"].notna().astype(int)

            # Final score per innings (what we're trying to predict).
            final_scores = deliveries.groupby(["match_id", "innings"])["total_runs"].sum()

            # Venue average = mean final score across all innings at that venue.
            venue_by_match = matches.set_index("match_id")["venue"]
            venue_series = final_scores.reset_index()
            venue_series["venue"] = venue_series["match_id"].map(venue_by_match)
            venue_avg = venue_series.groupby("venue")["total_runs"].mean()
            self.venue_avg_lookup.update(venue_avg.to_dict())

            # "Batting depth" proxy: how many distinct batters a team sends
            # in on average (deeper batting lineups -> higher late-innings
            # scoring potential). "Death overs form" proxy: each team's
            # average run rate in overs 15+ (t20) / 40+ (odi) historically.
            total_overs = OVERS_BY_FORMAT.get(comp, 20)
            death_over_start = total_overs - 5
            death = deliveries[deliveries["ball"] >= death_over_start]
            death_rr = death.groupby("batting_team")["total_runs"].sum() / \
                       (death.groupby("batting_team").size() / 6).replace(0, np.nan)
            self.team_death_form_lookup.update(death_rr.fillna(death_rr.mean()).to_dict())

            depth = deliveries.groupby(["match_id", "innings", "batting_team"])["striker"].nunique()
            depth_by_team = depth.groupby("batting_team").mean()
            self.team_depth_lookup.update(depth_by_team.to_dict())

            merged = deliveries.merge(
                matches[["match_id", "match_type"]], on="match_id", how="inner"
            )
            merged = merged.sort_values(["match_id", "innings", "ball"])
            merged["current_runs"] = merged.groupby(["match_id", "innings"])["total_runs"].cumsum()
            merged["wickets_fallen"] = merged.groupby(["match_id", "innings"])["is_wicket"].cumsum()

            over_part = np.floor(merged["ball"])
            ball_in_over = np.round((merged["ball"] - over_part) * 10).astype(int)
            merged["balls_done"] = (over_part * 6 + ball_in_over).astype(int)

            merged["overs_bowled"] = merged["ball"]
            merged["current_run_rate"] = merged["current_runs"] / merged["overs_bowled"].clip(lower=0.1)
            merged["wickets_remaining"] = 10 - merged["wickets_fallen"]
            merged["venue_average"] = merged["venue"].map(self.venue_avg_lookup)
            merged["batting_depth"] = merged["batting_team"].map(self.team_depth_lookup)
            merged["death_overs_form"] = merged["batting_team"].map(self.team_death_form_lookup)

            final_scores_flat = final_scores.reset_index().rename(columns={"total_runs": "final_score"})
            merged = merged.merge(final_scores_flat, on=["match_id", "innings"], how="left")

            all_rows.append(merged[[
                "current_runs", "wickets_fallen", "overs_bowled", "current_run_rate",
                "wickets_remaining", "venue_average", "batting_depth",
                "death_overs_form", "match_type", "final_score",
            ]])

        df = pd.concat(all_rows, ignore_index=True)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=random_state)

        df["match_type_encoded"] = self.match_type_encoder.fit_transform(df["match_type"])

        X = df[FEATURE_COLUMNS]
        y = df["final_score"]
        return X, y

    # ---------------- Training / evaluation ----------------

    def train(self, X, y, n_iter=100, cv=5, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        search_space = {
            "n_estimators": Integer(50, 400),
            "max_depth": Integer(3, 30),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf": Integer(1, 10),
            "max_features": Real(0.1, 1.0, prior="uniform"),
        }

        opt = BayesSearchCV(
            estimator=RandomForestRegressor(random_state=random_state, n_jobs=-1),
            search_spaces=search_space,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )
        opt.fit(X_train, y_train)
        self.model = opt.best_estimator_

        print(f"Best params: {opt.best_params_}")
        self._X_test, self._y_test = X_test, y_test
        return self

    def evaluate(self):
        if self.model is None:
            raise RuntimeError("Call train() before evaluate().")

        preds = self.model.predict(self._X_test)
        rmse = mean_squared_error(self._y_test, preds) ** 0.5
        mae = mean_absolute_error(self._y_test, preds)
        r2 = r2_score(self._y_test, preds)

        print(f"RMSE: {rmse:.2f} runs")
        print(f"MAE:  {mae:.2f} runs")
        print(f"R2:   {r2:.4f}")
        return {"rmse": rmse, "mae": mae, "r2": r2}

    # ---------------- Inference ----------------

    def predict_score(self, match_state: dict) -> dict:
        """Returns {'predicted_score': int, 'min_score': int, 'max_score': int}."""
        if self.model is None:
            raise RuntimeError("Model not trained/loaded.")

        row = dict(match_state)
        match_type = str(row.pop("match_type", "t20"))
        try:
            row["match_type_encoded"] = self.match_type_encoder.transform([match_type])[0]
        except ValueError:
            row["match_type_encoded"] = 0

        X = pd.DataFrame([row])[FEATURE_COLUMNS]
        predicted = float(self.model.predict(X)[0])

        total_overs = OVERS_BY_FORMAT.get("t20s" if "t20" in match_type.lower() else "odis", 20)
        low, high = _score_range(predicted, match_state.get("overs_bowled", 0), total_overs)

        return {
            "predicted_score": round(predicted),
            "min_score": round(low),
            "max_score": round(high),
        }

    def feature_importances(self) -> dict:
        """Which factors influenced the prediction most, for the dashboard's
        'what drove this prediction' panel."""
        if self.model is None:
            raise RuntimeError("Model not trained/loaded.")
        return dict(sorted(
            zip(FEATURE_COLUMNS, self.model.feature_importances_.tolist()),
            key=lambda kv: kv[1], reverse=True,
        ))

    # ---------------- Persistence ----------------

    def save(self, path=SAVE_PATH):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self, path)
        print(f"Saved ScorePredictor to {path}")

    @staticmethod
    def load(path=SAVE_PATH) -> "ScorePredictor":
        return joblib.load(path)


if __name__ == "__main__":
    # Run with: python -m models.score_predictor  (from the project root, venv active)
    # (see the note in win_predictor.py's __main__ block for why -m matters here)
    sp = ScorePredictor()
    print("Building training data from data/t20s and data/odis ...")
    X, y = sp.build_training_data()
    print(f"Training rows: {len(X)}  (avg final score in sample: {y.mean():.1f})")

    print("Training RandomForestRegressor with BayesSearchCV (this can take a few minutes) ...")
    sp.train(X, y)
    sp.evaluate()
    sp.save()
