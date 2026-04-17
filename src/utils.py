import os
import re
import glob
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8", "latin1", "cp1252"]
    last_exc = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_exc = e
    raise last_exc


def find_files(root: str, patterns: List[str]) -> List[str]:
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    return sorted(list(set(files)))


def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    return df


def infer_datetime_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if any(k in c for k in ["date", "time", "timestamp", "day"])]
    for c in candidates:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.5:
                return c
        except Exception:
            continue
    return None


def infer_numeric_target(df: pd.DataFrame) -> Optional[str]:
    candidates = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            score = 0
            if any(k in c for k in ["waste", "kg", "lb", "amount", "volume", "weight"]):
                score += 3
            if df[c].notna().mean() > 0.5:
                score += 1
            candidates.append((c, score))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_time_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    df["day_of_week"] = df[dt_col].dt.dayofweek
    df["week_of_year"] = df[dt_col].dt.isocalendar().week.astype(int)
    df["month"] = df[dt_col].dt.month
    df["quarter"] = df[dt_col].dt.quarter
    df["day_of_month"] = df[dt_col].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["semester_week_proxy"] = ((df[dt_col].dt.dayofyear - 1) // 7 + 1).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, group_cols: List[str], target_col: str, lags: List[int]) -> pd.DataFrame:
    df = df.copy().sort_values(group_cols + ["date"])

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)

    for window in [3, 7, 14]:
        df[f"{target_col}_rollmean_{window}"] = (
            df.groupby(group_cols)[target_col]
              .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    return df


def plot_forecast(actual: pd.Series, pred: pd.Series, title: str, path: str):
    plt.figure(figsize=(12, 5))
    plt.plot(actual.values, label="Actual")
    plt.plot(pred.values, label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Waste Volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def ensure_kaggle_credentials():
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_path):
        print("[WARN] Kaggle API credentials not found at ~/.kaggle/kaggle.json")
        return False
    return True
