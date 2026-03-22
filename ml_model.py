"""
Random Forest Trading Model
-----------------------------
• Trains a binary classifier: "will price rise ≥ threshold% over next N days?"
• Uses time-series cross-validation to avoid look-ahead bias
• Provides feature importances and probability-based signals

Python 3.11 / pandas 2.x notes
-------------------------------
pandas 2.0 introduced Copy-on-Write semantics and raises
ChainedAssignmentError when writing to a slice of a DataFrame.
All mutated DataFrames are explicitly .copy()-ed before assignment.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)

logger = logging.getLogger(__name__)

# All features the model can use (subset present in df is selected automatically)
ALL_FEATURE_COLS = [
    # RSI
    "rsi_14",
    # MACD
    "macd", "macd_signal", "macd_hist",
    # Bollinger Bands
    "bb_width", "bb_pct",
]


class TradingModel:
    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.scaler = StandardScaler()
        self._raw_data: dict[str, pd.DataFrame] = {}   # symbol → indicator df
        self._signal_data: dict[str, pd.DataFrame] = {}  # symbol → df w/ signals
        self._last_params: dict = {}

    # ------------------------------------------------------------------ #
    #  Data management
    # ------------------------------------------------------------------ #

    def set_data(self, symbol: str, df: pd.DataFrame) -> None:
        self._raw_data[symbol] = df
        # Clear stale signals when new data arrives
        self._signal_data.pop(symbol, None)

    def get_signal_data(self, symbol: str) -> pd.DataFrame | None:
        return self._signal_data.get(symbol)

    def list_trained_symbols(self):
        return list(self._signal_data.keys())

    # ------------------------------------------------------------------ #
    #  Feature engineering
    # ------------------------------------------------------------------ #

    def _prepare_features(
        self,
        df: pd.DataFrame,
        forward_days: int,
        threshold: float,
    ):
        df = df.copy()

        # Target: -1 = SHORT (price drops ≥ threshold), 0 = HOLD, 1 = BUY (price rises ≥ threshold)
        future_ret = df["close"].pct_change(forward_days).shift(-forward_days)
        target = pd.Series(0, index=df.index, dtype=int)
        target[future_ret >=  threshold] = 1
        target[future_ret <= -threshold] = -1
        df["_target"] = target

        feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]

        # pandas 2.x: explicit .copy() prevents ChainedAssignmentError when
        # we later write 'signal' and 'signal_prob' columns back into df_clean.
        df_clean = df.dropna(subset=feature_cols + ["_target"]).copy()

        X = df_clean[feature_cols]
        y = df_clean["_target"]
        return X, y, df_clean, feature_cols

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def train(
        self,
        symbol: str,
        forward_days: int = 5,
        threshold: float = 0.02,
        n_estimators: int = 200,
        max_depth: int = 10,
        n_splits: int = 5,
    ) -> dict:
        if symbol not in self._raw_data:
            return {"success": False, "message": f"No data loaded for '{symbol}'."}

        df = self._raw_data[symbol]
        X, y, df_clean, feature_cols = self._prepare_features(df, forward_days, threshold)

        if len(X) < 120:
            return {
                "success": False,
                "message": f"Not enough data ({len(X)} rows after cleaning). Need ≥120.",
            }

        # ── Time-series cross-validation ────────────────────────────
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_accuracies = []
        cv_precisions = []

        for train_idx, val_idx in tscv.split(X):
            X_tr = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            scaler_tmp = StandardScaler()
            X_tr_s = scaler_tmp.fit_transform(X_tr)
            X_val_s = scaler_tmp.transform(X_val)

            rf_tmp = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            rf_tmp.fit(X_tr_s, y_tr)
            preds = rf_tmp.predict(X_val_s)
            cv_accuracies.append(accuracy_score(y_val, preds))
            cv_precisions.append(precision_score(y_val, preds, zero_division=0, average="weighted"))

        # ── Final model on all data ──────────────────────────────────
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y)

        # ── In-sample evaluation (last fold) ─────────────────────────
        last_train, last_test = list(tscv.split(X))[-1]
        X_tr_f = self.scaler.transform(X.iloc[last_train])
        X_te_f = self.scaler.transform(X.iloc[last_test])
        y_te_f = y.iloc[last_test]
        y_pred_f = self.model.predict(X_te_f)

        report = classification_report(y_te_f, y_pred_f, output_dict=True)

        # ── Feature importances ──────────────────────────────────────
        importance = {
            col: round(float(imp), 6)
            for col, imp in sorted(
                zip(feature_cols, self.model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        }

        # ── Generate signals on full dataset ─────────────────────────
        # classes_ is sorted: [-1, 0, 1] — get prob of the predicted class per row
        classes = list(self.model.classes_)
        full_scaled  = self.scaler.transform(X)
        full_preds   = self.model.predict(full_scaled)
        full_probas  = self.model.predict_proba(full_scaled)
        pred_indices = [classes.index(int(p)) for p in full_preds]
        signal_probs = full_probas[np.arange(len(full_preds)), pred_indices]

        df_clean["signal"]      = full_preds
        df_clean["signal_prob"] = signal_probs

        self._signal_data[symbol] = df_clean
        self._last_params = {
            "symbol": symbol,
            "forward_days": forward_days,
            "threshold": threshold,
            "n_estimators": n_estimators,
        }

        # Class distribution
        buy_pct   = float((y ==  1).mean())
        short_pct = float((y == -1).mean())

        return {
            "success": True,
            "symbol": symbol,
            "rows_trained": len(X),
            "features_used": feature_cols,
            "feature_count": len(feature_cols),
            "forward_days": forward_days,
            "threshold": threshold,
            "buy_signal_pct":   round(buy_pct   * 100, 1),
            "short_signal_pct": round(short_pct * 100, 1),
            "cv_accuracy_mean": round(float(np.mean(cv_accuracies)) * 100, 2),
            "cv_accuracy_std":  round(float(np.std(cv_accuracies))  * 100, 2),
            "cv_precision_mean": round(float(np.mean(cv_precisions)) * 100, 2),
            "test_accuracy":  round(accuracy_score(y_te_f, y_pred_f) * 100, 2),
            "test_precision": round(precision_score(y_te_f, y_pred_f, zero_division=0, average="weighted") * 100, 2),
            "test_recall":    round(recall_score(y_te_f, y_pred_f, zero_division=0, average="weighted") * 100, 2),
            "test_f1":        round(f1_score(y_te_f, y_pred_f, zero_division=0, average="weighted") * 100, 2),
            "feature_importance": importance,
            "report": report,
        }

    # ------------------------------------------------------------------ #
    #  Prediction (latest bar)
    # ------------------------------------------------------------------ #

    def predict(self, symbol: str) -> dict:
        if self.model is None:
            return {"success": False, "message": "Model not trained yet."}

        df = self._raw_data.get(symbol)
        if df is None:
            return {"success": False, "message": f"No data for '{symbol}'."}

        feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
        latest = df[feature_cols].dropna().tail(1)

        if latest.empty:
            return {"success": False, "message": "Latest row has NaN features."}

        latest_scaled = self.scaler.transform(latest)
        prediction = int(self.model.predict(latest_scaled)[0])
        classes    = list(self.model.classes_)
        probas     = self.model.predict_proba(latest_scaled)[0]
        probability = float(probas[classes.index(prediction)])

        signal_label = "BUY" if prediction == 1 else ("SHORT" if prediction == -1 else "HOLD")

        idx = df.index[-1]
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)

        return {
            "success": True,
            "symbol": symbol,
            "date": date_str,
            "signal": signal_label,
            "prediction": prediction,
            "probability": round(probability, 4),
            "last_close": round(float(df["close"].iloc[-1]), 4),
        }

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str = "model.joblib") -> bool:
        if self.model is None:
            return False
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        return True

    def load(self, path: str = "model.joblib") -> bool:
        if not os.path.exists(path):
            return False
        obj = joblib.load(path)
        self.model  = obj["model"]
        self.scaler = obj["scaler"]
        return True
