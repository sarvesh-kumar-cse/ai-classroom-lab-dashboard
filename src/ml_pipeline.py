"""ML pipeline reproducing the notebook's Random Forest and XGBoost models.

Includes:
- Exact date feature engineering (month, day_of_month, day_of_year)
- Exact 15 input features (6 categorical + 9 numerical)
- Target leakage audit excluding utilization_pct, actual_occupancy, status
- Chronological 80:20 train-test split by unique sorted dates
- ColumnTransformer (OneHotEncoder + SimpleImputer)
- RandomForestRegressor and XGBRegressor training, evaluation, and live inference
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.data_generator import PROJECT_ROOT, classify_room_utilization, generate_datasets


MODELS_DIR = PROJECT_ROOT / "models"

TARGET = "utilization_pct"

CATEGORICAL_FEATURES = [
    "day_of_week",
    "room_id",
    "room_type",
    "time_slot",
    "course_id",
    "department",
]

NUMERICAL_FEATURES = [
    "capacity",
    "semester",
    "scheduled_students",
    "is_scheduled",
    "is_holiday",
    "is_exam_period",
    "month",
    "day_of_month",
    "day_of_year",
]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

FORBIDDEN_LEAKAGE_FEATURES = [
    "utilization_pct",
    "actual_occupancy",
    "status",
]


def prepare_ml_dataframe(history_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact notebook preprocessing before splitting."""
    ml_df = history_df.copy()
    ml_df["date"] = pd.to_datetime(ml_df["date"])
    ml_df["month"] = ml_df["date"].dt.month
    ml_df["day_of_month"] = ml_df["date"].dt.day
    ml_df["day_of_year"] = ml_df["date"].dt.dayofyear
    ml_df[CATEGORICAL_FEATURES] = ml_df[CATEGORICAL_FEATURES].fillna("Unscheduled")
    ml_df = ml_df.sort_values("date").reset_index(drop=True)
    return ml_df


def build_preprocessor() -> ColumnTransformer:
    """Create the notebook's ColumnTransformer preprocessing pipeline."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", SimpleImputer(strategy="median"), NUMERICAL_FEATURES),
        ]
    )


def run_leakage_audit(feature_columns: list[str]) -> dict[str, Any]:
    """Verify no target leakage features are present in model inputs."""
    checks = []
    has_leakage = False
    for col in FORBIDDEN_LEAKAGE_FEATURES:
        present = col in feature_columns
        if present:
            has_leakage = True
        checks.append({
            "feature": col,
            "present_in_inputs": present,
            "status": "WARNING: Leakage detected" if present else "SAFE: Excluded",
        })
    return {
        "input_features": feature_columns,
        "total_input_features": len(feature_columns),
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "checks": checks,
        "passed": not has_leakage,
    }


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    residuals = y_true - y_pred
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "r2_pct": round(r2 * 100, 2),
        "mean_error": round(float(np.mean(residuals)), 4),
        "std_error": round(float(np.std(residuals)), 4),
        "median_abs_error": round(float(np.median(np.abs(residuals))), 4),
        "p95_abs_error": round(float(np.percentile(np.abs(residuals), 95)), 4),
    }


def _build_scatter_and_error_samples(
    y_true: np.ndarray,
    rf_pred: np.ndarray,
    xgb_pred: np.ndarray,
    test_dates: pd.Series,
    sample_size: int = 800,
) -> dict[str, Any]:
    """Create deterministic sampled scatter points and full-test residual histograms."""
    rng = np.random.default_rng(42)
    n = len(y_true)
    idx = np.sort(rng.choice(n, size=min(sample_size, n), replace=False))

    scatter_points = []
    for i in idx:
        scatter_points.append({
            "actual": round(float(y_true[i]), 2),
            "rf_pred": round(float(rf_pred[i]), 2),
            "xgb_pred": round(float(xgb_pred[i]), 2),
            "rf_error": round(float(y_true[i] - rf_pred[i]), 2),
            "xgb_error": round(float(y_true[i] - xgb_pred[i]), 2),
            "date": test_dates.iloc[i].strftime("%Y-%m-%d"),
        })

    # Compute error distribution bins across ALL test records
    rf_err = y_true - rf_pred
    xgb_err = y_true - xgb_pred
    bins = np.linspace(-30, 30, 25)
    rf_hist, edges = np.histogram(np.clip(rf_err, -30, 30), bins=bins)
    xgb_hist, _ = np.histogram(np.clip(xgb_err, -30, 30), bins=bins)

    error_bins = []
    for i in range(len(rf_hist)):
        error_bins.append({
            "bin_label": f"{edges[i]:.1f} to {edges[i + 1]:.1f}",
            "bin_mid": round(float((edges[i] + edges[i + 1]) / 2), 2),
            "rf_count": int(rf_hist[i]),
            "xgb_count": int(xgb_hist[i]),
        })

    # Scheduled vs Unscheduled error breakdown
    scheduled_mask = y_true > 0
    return {
        "scatter_points": scatter_points,
        "error_histogram": error_bins,
        "test_min": round(float(np.min(y_true)), 2),
        "test_max": round(float(np.max(y_true)), 2),
        "scheduled_test_count": int(np.sum(scheduled_mask)),
        "unscheduled_test_count": int(np.sum(~scheduled_mask)),
        "scheduled_rf_mae": round(float(mean_absolute_error(y_true[scheduled_mask], rf_pred[scheduled_mask])), 4),
        "scheduled_xgb_mae": round(float(mean_absolute_error(y_true[scheduled_mask], xgb_pred[scheduled_mask])), 4),
    }


def train_and_evaluate(history_df: pd.DataFrame | None = None, models_dir: Path | str = MODELS_DIR, force: bool = False) -> dict[str, Any]:
    """Train Random Forest & XGBoost on the chronological 80:20 split and persist artifacts."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    rf_file = models_path / "rf_model.joblib"
    xgb_file = models_path / "xgb_model.joblib"
    eval_file = models_path / "evaluation_artifacts.json"

    if not force and rf_file.exists() and xgb_file.exists() and eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if history_df is None:
        history_df = generate_datasets()["history"]

    ml_df = prepare_ml_dataframe(history_df)

    unique_dates = sorted(ml_df["date"].unique())
    split_index = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_index]

    train_df = ml_df[ml_df["date"] < split_date]
    test_df = ml_df[ml_df["date"] >= split_date]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    rf_model = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    xgb_model = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    y_test_arr = y_test.to_numpy(dtype=float)
    rf_metrics = _evaluate_predictions(y_test_arr, rf_pred)
    xgb_metrics = _evaluate_predictions(y_test_arr, xgb_pred)

    # Determine recommended model strictly from actual metrics
    best_model_key = "xgboost" if (xgb_metrics["rmse"] <= rf_metrics["rmse"] and xgb_metrics["r2"] >= rf_metrics["r2"]) else "random_forest"
    best_model_name = "XGBoost Regressor" if best_model_key == "xgboost" else "Random Forest Regressor"

    # Extract top feature importances mapped back to readable names
    feature_names_out = rf_model.named_steps["preprocessor"].get_feature_names_out()
    rf_importances = rf_model.named_steps["model"].feature_importances_
    xgb_importances = xgb_model.named_steps["model"].feature_importances_

    grouped_rf: dict[str, float] = {f: 0.0 for f in FEATURES}
    grouped_xgb: dict[str, float] = {f: 0.0 for f in FEATURES}
    for raw_name, rf_imp, xgb_imp in zip(feature_names_out, rf_importances, xgb_importances):
        clean = raw_name.split("__", 1)[-1]
        matched = next((f for f in FEATURES if clean == f or clean.startswith(f + "_")), clean)
        grouped_rf[matched] = grouped_rf.get(matched, 0.0) + float(rf_imp)
        grouped_xgb[matched] = grouped_xgb.get(matched, 0.0) + float(xgb_imp)

    feature_importance_list = sorted(
        [
            {
                "feature": feat,
                "rf_importance": round(grouped_rf[feat] * 100, 2),
                "xgb_importance": round(grouped_xgb[feat] * 100, 2),
            }
            for feat in FEATURES
        ],
        key=lambda x: max(x["rf_importance"], x["xgb_importance"]),
        reverse=True,
    )

    artifacts: dict[str, Any] = {
        "data_source_note": "Trained on synthetic university utilization data generated via AI_Classroom_Lab_Utilization_System.ipynb (seed=42).",
        "split_info": {
            "strategy": "Chronological 80:20 Split by Unique Dates",
            "total_records": int(len(ml_df)),
            "unique_dates": int(len(unique_dates)),
            "split_date": pd.Timestamp(split_date).strftime("%Y-%m-%d"),
            "train_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "train_start": pd.Timestamp(train_df["date"].min()).strftime("%Y-%m-%d"),
            "train_end": pd.Timestamp(train_df["date"].max()).strftime("%Y-%m-%d"),
            "test_start": pd.Timestamp(test_df["date"].min()).strftime("%Y-%m-%d"),
            "test_end": pd.Timestamp(test_df["date"].max()).strftime("%Y-%m-%d"),
        },
        "models": {
            "random_forest": {
                "name": "Random Forest Regressor",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "min_samples_leaf": 2,
                    "random_state": 42,
                },
                "metrics": rf_metrics,
            },
            "xgboost": {
                "name": "XGBoost Regressor",
                "params": {
                    "n_estimators": 400,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "reg:squarederror",
                    "random_state": 42,
                },
                "metrics": xgb_metrics,
            },
        },
        "recommendation": {
            "recommended_model_key": best_model_key,
            "recommended_model_name": best_model_name,
            "mae_improvement": round(rf_metrics["mae"] - xgb_metrics["mae"], 4),
            "rmse_improvement": round(rf_metrics["rmse"] - xgb_metrics["rmse"], 4),
            "r2_improvement": round(xgb_metrics["r2"] - rf_metrics["r2"], 4),
        },
        "leakage_audit": run_leakage_audit(list(X_train.columns)),
        "feature_importance": feature_importance_list,
        "visualizations": _build_scatter_and_error_samples(
            y_test_arr, rf_pred, xgb_pred, test_df["date"]
        ),
    }

    joblib.dump(rf_model, rf_file)
    joblib.dump(xgb_model, xgb_file)
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    return artifacts


class ModelRegistry:
    """Lazy-loaded holder for the trained scikit-learn/XGBoost pipelines."""

    def __init__(self, models_dir: Path | str = MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self.rf_model: Pipeline | None = None
        self.xgb_model: Pipeline | None = None
        self.artifacts: dict[str, Any] | None = None

    def ensure_loaded(self, history_df: pd.DataFrame | None = None) -> None:
        if self.rf_model is not None and self.xgb_model is not None and self.artifacts is not None:
            return
        self.artifacts = train_and_evaluate(history_df=history_df, models_dir=self.models_dir)
        self.rf_model = joblib.load(self.models_dir / "rf_model.joblib")
        self.xgb_model = joblib.load(self.models_dir / "xgb_model.joblib")

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Predict utilization percentage using the exact 15-feature notebook pipeline."""
        self.ensure_loaded()
        assert self.rf_model is not None and self.xgb_model is not None and self.artifacts is not None

        date_val = pd.to_datetime(payload["date"])
        is_scheduled = int(payload["is_scheduled"])
        is_holiday = int(payload["is_holiday"])

        course_id = payload.get("course_id") or "Unscheduled"
        department = payload.get("department") or "Unscheduled"
        if is_scheduled == 0:
            course_id = "Unscheduled"
            department = "Unscheduled"

        semester_raw = payload.get("semester")
        semester = np.nan if (is_scheduled == 0 or semester_raw in (None, "", "Unscheduled")) else float(semester_raw)

        scheduled_students = 0 if is_scheduled == 0 else int(payload["scheduled_students"])

        row = pd.DataFrame([{
            "day_of_week": str(payload.get("day_of_week") or date_val.day_name()),
            "room_id": str(payload["room_id"]),
            "room_type": str(payload["room_type"]),
            "time_slot": str(payload["time_slot"]),
            "course_id": str(course_id),
            "department": str(department),
            "capacity": int(payload["capacity"]),
            "semester": semester,
            "scheduled_students": scheduled_students,
            "is_scheduled": is_scheduled,
            "is_holiday": is_holiday,
            "is_exam_period": int(payload["is_exam_period"]),
            "month": int(date_val.month),
            "day_of_month": int(date_val.day),
            "day_of_year": int(date_val.dayofyear),
        }])[FEATURES]

        rf_pred = max(0.0, float(self.rf_model.predict(row)[0]))
        xgb_pred = max(0.0, float(self.xgb_model.predict(row)[0]))

        selected_key = payload.get("model_name", "xgboost")
        if selected_key not in ("random_forest", "xgboost"):
            selected_key = "xgboost"

        selected_pred = xgb_pred if selected_key == "xgboost" else rf_pred
        selected_pred_round = round(selected_pred, 2)

        if is_scheduled == 0 or is_holiday == 1 and is_scheduled == 0:
            slot_status = "Available"
        elif selected_pred_round > 80:
            slot_status = "Overloaded"
        elif selected_pred_round < 30:
            slot_status = "Underutilized"
        else:
            slot_status = "Normal"

        room_band_status = classify_room_utilization(selected_pred_round)

        return {
            "selected_model": selected_key,
            "selected_model_name": self.artifacts["models"][selected_key]["name"],
            "predicted_utilization_pct": selected_pred_round,
            "predicted_resource_status": room_band_status,
            "predicted_slot_status": slot_status,
            "predictions": {
                "random_forest": {
                    "name": "Random Forest Regressor",
                    "predicted_utilization_pct": round(rf_pred, 2),
                    "predicted_status": classify_room_utilization(rf_pred),
                    "metrics": self.artifacts["models"]["random_forest"]["metrics"],
                },
                "xgboost": {
                    "name": "XGBoost Regressor",
                    "predicted_utilization_pct": round(xgb_pred, 2),
                    "predicted_status": classify_room_utilization(xgb_pred),
                    "metrics": self.artifacts["models"]["xgboost"]["metrics"],
                },
            },
            "features_sent": row.fillna("Median-Imputed (Unscheduled)").to_dict(orient="records")[0],
            "synthetic_notice": self.artifacts["data_source_note"],
        }


if __name__ == "__main__":
    results = train_and_evaluate(force=True)
    print("RF Metrics:", results["models"]["random_forest"]["metrics"])
    print("XGB Metrics:", results["models"]["xgboost"]["metrics"])
    print("Recommended:", results["recommendation"])
