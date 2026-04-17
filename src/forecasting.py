import os
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from src.config import CFG
from src.utils import print_header, rmse, make_time_features, add_lag_features, plot_forecast
from src.data_ingestion import load_weather_if_available


def make_synthetic_event_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="D")
    df = pd.DataFrame({"date": dates})
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    df["event_flag"] = 0
    event_prob = np.where(df["day_of_week"].isin([4, 5]), 0.30, 0.08)
    event_prob += np.where(df["month"].isin([9, 10, 11]), 0.08, 0.00)
    event_prob += np.where(df["month"].isin([3, 4]), 0.05, 0.00)

    rng = np.random.default_rng(42)
    df["event_flag"] = (rng.random(len(df)) < event_prob).astype(int)
    df["event_intensity"] = np.where(df["event_flag"] == 1, rng.integers(1, 4, size=len(df)), 0)
    return df[["date", "event_flag", "event_intensity"]]


def enrich_with_context(food_df: pd.DataFrame) -> pd.DataFrame:
    df = food_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = make_time_features(df, "date")

    weather = load_weather_if_available()
    if weather is not None:
        weather_cols = [c for c in weather.columns if c != "date"]
        df = df.merge(weather[["date"] + weather_cols], on="date", how="left")
    else:
        rng = np.random.default_rng(42)
        df["temp"] = 18 + 10 * np.sin(2 * np.pi * df["month"] / 12) + rng.normal(0, 2, len(df))
        df["precip"] = np.clip(rng.gamma(shape=1.2, scale=1.5, size=len(df)) - 1.0, 0, None)

    events = make_synthetic_event_calendar(
        df["date"].min().date().isoformat(),
        df["date"].max().date().isoformat()
    )
    df = df.merge(events, on="date", how="left")

    base_by_location = {
        "dining_hall": 1.6,
        "dorm": 1.2,
        "academic": 1.0,
        "event_venue": 1.8,
    }

    df["location_type"] = df["location_id"].astype(str).apply(
        lambda x: "dining_hall" if "dining" in x.lower() or "canteen" in x.lower() else (
            "dorm" if "dorm" in x.lower() else (
                "event_venue" if "event" in x.lower() or "stadium" in x.lower() else "academic"
            )
        )
    )

    rng = np.random.default_rng(42)
    df["foot_traffic_proxy"] = (
        df["location_type"].map(base_by_location).fillna(1.0)
        * (1.0 + 0.35 * df["event_flag"] + 0.12 * df["is_weekend"])
        * (1.0 + rng.normal(0, 0.05, len(df)))
        * 100
    )

    return df


def prepare_forecasting_table(master_df: pd.DataFrame) -> pd.DataFrame:
    df = enrich_with_context(master_df)

    daily = (
        df.groupby(["date", "location_id", "location_type"], as_index=False)
          .agg({
              "waste_volume": "sum",
              "event_flag": "max",
              "event_intensity": "max",
              "foot_traffic_proxy": "mean",
              "temp": "mean" if "temp" in df.columns else "size",
              "precip": "mean" if "precip" in df.columns else "size",
              "day_of_week": "first",
              "week_of_year": "first",
              "month": "first",
              "is_weekend": "first",
              "semester_week_proxy": "first",
          })
    )

    if "temp" not in daily.columns:
        daily["temp"] = np.nan
    if "precip" not in daily.columns:
        daily["precip"] = np.nan

    daily = add_lag_features(daily, ["location_id"], "waste_volume", [1, 2, 3, 7, 14])
    daily = daily.dropna().reset_index(drop=True)
    return daily


def temporal_train_val_test_split(df: pd.DataFrame, date_col: str = "date", train_frac: float = 0.7, val_frac: float = 0.15):
    df = df.sort_values(date_col).reset_index(drop=True)
    unique_dates = sorted(df[date_col].unique())
    n = len(unique_dates)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    train_df = df[df[date_col].isin(train_dates)].copy()
    val_df = df[df[date_col].isin(val_dates)].copy()
    test_df = df[df[date_col].isin(test_dates)].copy()
    return train_df, val_df, test_df


def fit_xgb_forecaster(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols, target_col: str = "waste_volume"):
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(train_df[feature_cols], train_df[target_col])
    val_pred = model.predict(val_df[feature_cols])
    return model, val_pred


def fit_lgbm_forecaster(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols, target_col: str = "waste_volume"):
    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        random_state=42,
    )
    model.fit(train_df[feature_cols], train_df[target_col])
    val_pred = model.predict(val_df[feature_cols])
    return model, val_pred


def run_forecasting_pipeline(master_df: pd.DataFrame) -> pd.DataFrame:
    print_header("Running forecasting pipeline")
    daily = prepare_forecasting_table(master_df)

    feature_cols = [
        c for c in daily.columns
        if c not in ["date", "location_id", "location_type", "waste_volume"]
        and pd.api.types.is_numeric_dtype(daily[c])
    ]

    train_df, val_df, test_df = temporal_train_val_test_split(daily)

    xgb_model, xgb_val_pred = fit_xgb_forecaster(train_df, val_df, feature_cols)
    lgbm_model, lgbm_val_pred = fit_lgbm_forecaster(train_df, val_df, feature_cols)

    xgb_val_rmse = rmse(val_df["waste_volume"], xgb_val_pred)
    lgbm_val_rmse = rmse(val_df["waste_volume"], lgbm_val_pred)

    best_model_name = "xgboost" if xgb_val_rmse <= lgbm_val_rmse else "lightgbm"
    best_model = xgb_model if best_model_name == "xgboost" else lgbm_model

    print(f"Validation RMSE - XGBoost: {xgb_val_rmse:.4f}")
    print(f"Validation RMSE - LightGBM: {lgbm_val_rmse:.4f}")
    print(f"Selected best model: {best_model_name}")

    test_pred = best_model.predict(test_df[feature_cols])

    results = test_df[["date", "location_id", "waste_volume"]].copy()
    results["y_true"] = results["waste_volume"]
    results["y_pred"] = test_pred
    pred_std = results["y_pred"].std()
    results["y_pred_lower"] = np.clip(results["y_pred"] - pred_std, 0, None)
    results["y_pred_upper"] = results["y_pred"] + pred_std
    results = results.drop(columns=["waste_volume"])

    os.makedirs(CFG.outputs_dir, exist_ok=True)
    os.makedirs(CFG.figures_dir, exist_ok=True)

    results.to_csv(os.path.join(CFG.outputs_dir, "food_waste_forecast.csv"), index=False)

    metrics = pd.DataFrame([{
        "model": best_model_name,
        "mae": mean_absolute_error(results["y_true"], results["y_pred"]),
        "rmse": rmse(results["y_true"], results["y_pred"]),
    }])
    metrics.to_csv(os.path.join(CFG.outputs_dir, "forecast_metrics.csv"), index=False)

    plot_forecast(
        results["y_true"].reset_index(drop=True),
        results["y_pred"].reset_index(drop=True),
        title=f"Food Waste Forecast ({best_model_name})",
        path=os.path.join(CFG.figures_dir, "forecast_vs_actual.png"),
    )

    print(f"Saved forecast outputs to {os.path.join(CFG.outputs_dir, 'food_waste_forecast.csv')}")
    return results
