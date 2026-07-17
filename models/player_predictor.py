"""
player_predictor.py

PlayerPredictor estimates how many runs a batter will finish their
current innings on, given how they've batted so far.

Algorithm: RandomForestRegressor.

Note on 'vs_bowler_type': Cricsheet's ball-by-ball data doesn't include
a bowling-style field (pace/spin/etc.), so this is approximated from
which phase of the innings each bowler is used in most often (powerplay/
death-overs specialists lean pace, middle-overs specialists lean spin).
It's a genuine limitation of the free data source, not a bug - if you
later get access to a bowler-style lookup table (name -> pace/spin),
swap it in to replace `_infer_bowler_type()` for a real signal.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "saved", "player_model.pkl")

FEATURE_COLUMNS = [
    "player_form_last5", "balls_faced", "current_runs", "vs_bowler_type_encoded",
    "venue_encoded", "strike_rate", "partnership_runs", "match_type_encoded",
]

OVERS_BY_FORMAT = {"t20s": 20, "odis": 50}


class PlayerPredictor:
    """Predicts a batter's final runs in their current innings."""

    def __init__(self):
        self.model = None
        self.match_type_encoder = LabelEncoder()
        self.venue_encoder = LabelEncoder()
        self.bowler_type_encoder = LabelEncoder()
        self.bowler_type_lookup = {}  # bowler name -> 'pace_proxy' / 'spin_proxy'
        self.player_form_lookup = {}  # (player) -> most recent rolling-5 average, for live inference

    # ---------------- Feature engineering ----------------

    def _infer_bowler_types(self, deliveries: pd.DataFrame, total_overs: int) -> pd.Series:
        """See module docstring - phase-of-innings usage as a bowling-style proxy."""
        death_start = total_overs - 5
        powerplay_end = 6 if total_overs == 20 else 10

        phase = np.where(deliveries["ball"] < powerplay_end, "new_ball",
                 np.where(deliveries["ball"] >= death_start, "death", "middle"))
        usage = pd.DataFrame({"bowler": deliveries["bowler"], "phase": phase})
        dominant_phase = usage.groupby("bowler")["phase"].agg(lambda s: s.value_counts().idxmax())
        return dominant_phase.map({"new_ball": "pace_proxy", "death": "pace_proxy", "middle": "spin_proxy"})

    def build_training_data(self, competitions=("t20s", "odis"), sample_size=60000, random_state=42):
        all_rows = []

        for comp in competitions:
            deliveries = pd.read_csv(os.path.join(DATA_DIR, comp, "deliveries.csv.gz"))
            matches = pd.read_csv(os.path.join(DATA_DIR, comp, "matches.csv.gz"))
            total_overs = OVERS_BY_FORMAT.get(comp, 20)

            deliveries["total_runs"] = deliveries["runs_off_bat"].fillna(0) + deliveries["extras"].fillna(0)
            deliveries["is_wicket"] = deliveries["player_dismissed"].notna().astype(int)
            deliveries["is_wide"] = deliveries["wides"].notna()

            bowler_types = self._infer_bowler_types(deliveries, total_overs)
            self.bowler_type_lookup.update(bowler_types.to_dict())
            deliveries["vs_bowler_type"] = deliveries["bowler"].map(bowler_types).fillna("spin_proxy")

            merged = deliveries.merge(
                matches[["match_id", "date", "match_type"]], on="match_id", how="inner"
            )
            merged = merged.sort_values(["match_id", "innings", "ball"])

            # --- per-striker running totals within this innings ---
            merged["current_runs"] = merged.groupby(["match_id", "innings", "striker"])["runs_off_bat"].cumsum()
            merged["balls_faced"] = merged[~merged["is_wide"]].groupby(
                ["match_id", "innings", "striker"]).cumcount() + 1
            merged["balls_faced"] = merged["balls_faced"].ffill().fillna(1)
            merged["strike_rate"] = merged["current_runs"] / merged["balls_faced"].clip(lower=1) * 100

            # --- team-level cumulative runs, used for partnership_runs ---
            merged["team_cum_runs"] = merged.groupby(["match_id", "innings"])["total_runs"].cumsum()
            wicket_marker = np.where(merged["is_wicket"] == 1, merged["team_cum_runs"], np.nan)
            merged["_wicket_marker"] = wicket_marker
            merged["partnership_start_runs"] = merged.groupby(["match_id", "innings"])["_wicket_marker"] \
                .transform(lambda s: s.shift(1).ffill()).fillna(0)
            merged["partnership_runs"] = merged["team_cum_runs"] - merged["partnership_start_runs"]

            # --- final runs each player finished their innings on (the label) ---
            player_final = deliveries.groupby(["match_id", "innings", "striker"])["runs_off_bat"] \
                .sum().reset_index().rename(columns={"striker": "player", "runs_off_bat": "final_runs"})

            # --- rolling "form" = average of that player's previous 5 innings ---
            player_dates = player_final.merge(
                matches[["match_id", "date"]], on="match_id", how="left"
            ).sort_values(["player", "date"])
            player_dates["player_form_last5"] = player_dates.groupby("player")["final_runs"] \
                .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            player_dates["player_form_last5"] = player_dates["player_form_last5"].fillna(
                player_dates["final_runs"].mean()
            )
            # remember each player's most recent form value for live inference later
            latest_form = player_dates.sort_values("date").groupby("player")["player_form_last5"].last()
            self.player_form_lookup.update(latest_form.to_dict())

            merged = merged.merge(
                player_dates[["match_id", "innings", "player", "player_form_last5", "final_runs"]],
                left_on=["match_id", "innings", "striker"],
                right_on=["match_id", "innings", "player"],
                how="left",
            )

            all_rows.append(merged[[
                "player_form_last5", "balls_faced", "current_runs", "vs_bowler_type",
                "venue", "strike_rate", "partnership_runs", "match_type", "final_runs",
            ]])

        df = pd.concat(all_rows, ignore_index=True)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=random_state)

        df["vs_bowler_type_encoded"] = self.bowler_type_encoder.fit_transform(df["vs_bowler_type"])
        df["venue_encoded"] = self.venue_encoder.fit_transform(df["venue"])
        df["match_type_encoded"] = self.match_type_encoder.fit_transform(df["match_type"])

        X = df[FEATURE_COLUMNS]
        y = df["final_runs"]
        return X, y

    # ---------------- Training / evaluation ----------------

    def train(self, X, y, n_estimators=300, max_depth=None, random_state=42):
        """A plain RandomForestRegressor fit (no BayesSearchCV here - the
        spec calls for tuning on WinPredictor/ScorePredictor; this model
        uses sensible defaults, but train() accepts overrides if you want
        to tune it yourself)."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
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

    def predict_runs(self, player_state: dict) -> dict:
        """player_state should include: player_form_last5, balls_faced,
        current_runs, vs_bowler_type ('pace_proxy'/'spin_proxy'), venue,
        strike_rate, partnership_runs, match_type. Returns predicted final
        runs plus a rough confidence score (based on tree agreement)."""
        if self.model is None:
            raise RuntimeError("Model not trained/loaded.")

        row = dict(player_state)
        bowler_type = str(row.pop("vs_bowler_type", "spin_proxy"))
        venue = str(row.pop("venue", ""))
        match_type = str(row.pop("match_type", "t20"))

        def safe_encode(encoder, value, default=0):
            try:
                return int(encoder.transform([value])[0])
            except ValueError:
                return default

        row["vs_bowler_type_encoded"] = safe_encode(self.bowler_type_encoder, bowler_type)
        row["venue_encoded"] = safe_encode(self.venue_encoder, venue)
        row["match_type_encoded"] = safe_encode(self.match_type_encoder, match_type)

        X = pd.DataFrame([row])[FEATURE_COLUMNS]

        # Confidence = agreement across the forest's individual trees -
        # tighter spread of tree predictions = higher confidence.
        # (.values strips the DataFrame's column names, which the individual
        # trees weren't fitted with and would otherwise warn about.)
        tree_preds = np.array([t.predict(X.values)[0] for t in self.model.estimators_])
        predicted = float(tree_preds.mean())
        spread = float(tree_preds.std())
        confidence = max(0.0, 100.0 - min(spread * 4, 100.0))

        return {
            "predicted_runs": round(predicted),
            "confidence_pct": round(confidence, 1),
        }

    # ---------------- Persistence ----------------

    def save(self, path=SAVE_PATH):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self, path)
        print(f"Saved PlayerPredictor to {path}")

    @staticmethod
    def load(path=SAVE_PATH) -> "PlayerPredictor":
        return joblib.load(path)


if __name__ == "__main__":
    # Run with: python -m models.player_predictor  (from the project root, venv active)
    # (see the note in win_predictor.py's __main__ block for why -m matters here)
    pp = PlayerPredictor()
    print("Building training data from data/t20s and data/odis ...")
    X, y = pp.build_training_data()
    print(f"Training rows: {len(X)}  (avg final runs in sample: {y.mean():.1f})")

    print("Training RandomForestRegressor ...")
    pp.train(X, y)
    pp.evaluate()
    pp.save()
