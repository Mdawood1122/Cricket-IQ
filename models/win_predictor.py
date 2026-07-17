"""
win_predictor.py

WinPredictor learns to estimate a chasing team's win probability from
an in-progress run chase, using Cricsheet's historical ball-by-ball data.

Why only the 2nd innings? "Required run rate" and "target" only exist
once there's a target to chase - so every training row here is one ball
of a run-chase, labelled with whether the chasing (batting) team went
on to win.

Algorithm: SVC(probability=True), tuned with BayesSearchCV.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "saved", "win_model.pkl")

FEATURE_COLUMNS = [
    "current_runs", "wickets_fallen", "overs_bowled", "current_run_rate",
    "required_run_rate", "wickets_remaining", "venue_average",
    "batting_team_form", "bowling_team_form", "match_type_encoded",
]

# Overs per innings by format - used to work out balls_left for required run rate.
OVERS_BY_FORMAT = {"t20s": 20, "odis": 50}


class WinPredictor:
    """Predicts win probability for both teams at any point in a run chase."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.match_type_encoder = LabelEncoder()
        # Lookups learned from training data, needed again at inference time
        # (a live match state won't come with its own venue history attached).
        self.venue_avg_lookup = {}
        self.team_form_lookup = {}
        self.global_avg_target = 0.0

    # ---------------- Feature engineering ----------------

    def _team_form_lookup_from(self, matches: pd.DataFrame) -> dict:
        """Overall historical win rate per team. A simple stand-in for
        'recent form' - a production system would window this to each
        team's last N matches instead of their whole history."""
        wins = matches["winner"].value_counts()
        games = pd.concat([matches["team1"], matches["team2"]]).value_counts()
        form = (wins.reindex(games.index).fillna(0) / games).fillna(0.5)
        return form.to_dict()

    def build_training_data(self, competitions=("t20s", "odis"), sample_size=40000, random_state=42):
        """Builds one training row per ball of every run-chase in the given
        competitions' Cricsheet data, then downsamples to `sample_size` rows.

        Why downsample: SVC's training cost grows roughly quadratically
        with row count, so running a 100-iteration, 5-fold BayesSearchCV
        over the full multi-million-row dataset would take hours. 40,000
        rows keeps a full search to a few minutes while still giving the
        model a solid, varied sample to learn from - raise this if you
        have the time and want to trade training speed for accuracy.
        """
        all_rows = []

        for comp in competitions:
            deliveries = pd.read_csv(os.path.join(DATA_DIR, comp, "deliveries.csv.gz"))
            matches = pd.read_csv(os.path.join(DATA_DIR, comp, "matches.csv.gz"))
            matches = matches.dropna(subset=["winner", "team1", "team2"])

            self.team_form_lookup.update(self._team_form_lookup_from(matches))

            deliveries["total_runs"] = deliveries["runs_off_bat"].fillna(0) + deliveries["extras"].fillna(0)
            deliveries["is_wicket"] = deliveries["player_dismissed"].notna().astype(int)

            # First-innings total per match = the target the 2nd innings chases.
            inn1 = deliveries[deliveries["innings"] == 1]
            inn1_totals = inn1.groupby("match_id")["total_runs"].sum()

            # Venue average first-innings score, used as a "how hard is this
            # ground to defend on" feature.
            venue_by_match = matches.set_index("match_id")["venue"]
            venue_totals = inn1_totals.groupby(venue_by_match.reindex(inn1_totals.index)).mean()
            self.venue_avg_lookup.update(venue_totals.to_dict())

            # Note: deliveries already has its own 'venue' column, so we
            # deliberately don't pull 'venue' from matches too - merging two
            # frames that both have a 'venue' column would rename them to
            # 'venue_x'/'venue_y' instead of keeping a plain 'venue' column.
            inn2 = deliveries[deliveries["innings"] == 2].merge(
                matches[["match_id", "winner", "team1", "team2", "match_type"]],
                on="match_id", how="inner",
            )
            inn2 = inn2[inn2["match_id"].isin(inn1_totals.index)].copy()
            inn2["target"] = inn2["match_id"].map(inn1_totals) + 1

            # Vectorized running totals within each match (much faster than
            # looping ball-by-ball in Python).
            inn2 = inn2.sort_values(["match_id", "ball"])
            inn2["current_runs"] = inn2.groupby("match_id")["total_runs"].cumsum()
            inn2["wickets_fallen"] = inn2.groupby("match_id")["is_wicket"].cumsum()

            # 'ball' is over.ball_in_over (e.g. 12.3 = over 13, 3rd ball).
            over_part = np.floor(inn2["ball"])
            ball_in_over = np.round((inn2["ball"] - over_part) * 10).astype(int)
            inn2["balls_done"] = (over_part * 6 + ball_in_over).astype(int)

            total_overs = inn2["match_type"].str.lower().map(
                lambda mt: OVERS_BY_FORMAT.get("t20s" if "t20" in mt else "odis", 50)
            )
            total_balls = total_overs * 6
            balls_left = (total_balls - inn2["balls_done"]).clip(lower=1)
            runs_left = (inn2["target"] - inn2["current_runs"]).clip(lower=0)

            inn2["overs_bowled"] = inn2["ball"]
            inn2["current_run_rate"] = inn2["current_runs"] / inn2["overs_bowled"].clip(lower=0.1)
            inn2["required_run_rate"] = runs_left / (balls_left / 6)
            inn2["wickets_remaining"] = 10 - inn2["wickets_fallen"]
            inn2["venue_average"] = inn2["venue"].map(self.venue_avg_lookup).fillna(inn2["target"])
            inn2["batting_team_form"] = inn2["batting_team"].map(self.team_form_lookup).fillna(0.5)
            inn2["bowling_team_form"] = inn2["bowling_team"].map(self.team_form_lookup).fillna(0.5)
            inn2["batting_team_won"] = (inn2["winner"] == inn2["batting_team"]).astype(int)

            all_rows.append(inn2[[
                "current_runs", "wickets_fallen", "overs_bowled", "current_run_rate",
                "required_run_rate", "wickets_remaining", "venue_average",
                "batting_team_form", "bowling_team_form", "match_type", "batting_team_won",
            ]])

        df = pd.concat(all_rows, ignore_index=True)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=random_state)

        self.global_avg_target = df["venue_average"].mean()
        df["match_type_encoded"] = self.match_type_encoder.fit_transform(df["match_type"])

        X = df[FEATURE_COLUMNS]
        y = df["batting_team_won"]
        return X, y

    # ---------------- Training / evaluation ----------------

    def train(self, X, y, n_iter=100, cv=5, random_state=42):
        """Tunes an SVC(probability=True) with BayesSearchCV over a
        100-candidate, 5-fold Bayesian search of C, gamma, and kernel."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        search_space = {
            "C": Real(1e-2, 1e3, prior="log-uniform"),
            "gamma": Real(1e-4, 1e1, prior="log-uniform"),
            "kernel": Categorical(["rbf", "poly"]),
        }

        opt = BayesSearchCV(
            estimator=SVC(probability=True, random_state=random_state),
            search_spaces=search_space,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )
        opt.fit(X_train_scaled, y_train)
        self.model = opt.best_estimator_

        print(f"Best params: {opt.best_params_}")
        self._X_test, self._y_test = X_test_scaled, y_test
        return self

    def evaluate(self):
        """Prints accuracy, confusion matrix, and classification report on
        the held-out test split from the most recent train() call."""
        if self.model is None:
            raise RuntimeError("Call train() before evaluate().")

        preds = self.model.predict(self._X_test)
        acc = accuracy_score(self._y_test, preds)

        print(f"Accuracy: {acc:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(self._y_test, preds))
        print("Classification report:")
        print(classification_report(self._y_test, preds))
        return acc

    # ---------------- Inference ----------------

    def predict_win_probability(self, match_state: dict) -> dict:
        """Given a live match state dict (same fields as FEATURE_COLUMNS,
        minus the _encoded suffix - pass match_type as a string like 't20'),
        returns {'batting_team_win_pct': .., 'bowling_team_win_pct': ..}.
        """
        if self.model is None:
            raise RuntimeError("Model not trained/loaded.")

        row = dict(match_state)
        match_type = str(row.pop("match_type", "t20"))
        try:
            row["match_type_encoded"] = self.match_type_encoder.transform([match_type])[0]
        except ValueError:
            # Unseen match type at inference time - fall back to the most
            # common encoded value rather than crashing the request.
            row["match_type_encoded"] = 0

        X = pd.DataFrame([row])[FEATURE_COLUMNS]
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]  # [P(loss), P(win)] for the batting team

        return {
            "batting_team_win_pct": round(float(proba[1]) * 100, 2),
            "bowling_team_win_pct": round(float(proba[0]) * 100, 2),
        }

    # ---------------- Persistence ----------------

    def save(self, path=SAVE_PATH):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self, path)
        print(f"Saved WinPredictor to {path}")

    @staticmethod
    def load(path=SAVE_PATH) -> "WinPredictor":
        return joblib.load(path)


if __name__ == "__main__":
    # Run with: python -m models.win_predictor  (from the project root, venv active)
    #
    # Important: use `-m models.win_predictor`, NOT `python models\win_predictor.py`.
    # Running it as a bare script makes Python pickle the WinPredictor class
    # under the module name '__main__', and api.py (which imports it as
    # models.win_predictor) then fails to unpickle the saved model with
    # "ModuleNotFoundError: No module named 'win_predictor'". Running it as
    # a module keeps the class's pickled path consistent with how api.py
    # imports it.
    wp = WinPredictor()
    print("Building training data from data/t20s and data/odis ...")
    X, y = wp.build_training_data()
    print(f"Training rows: {len(X)}  (win rate in sample: {y.mean():.2%})")

    print("Training SVC with BayesSearchCV (this can take a few minutes) ...")
    wp.train(X, y)
    wp.evaluate()
    wp.save()
