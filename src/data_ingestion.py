import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import CFG
from src.utils import (
    print_header,
    safe_read_csv,
    find_files,
    normalize_colnames,
    infer_datetime_column,
    infer_numeric_target,
)


def load_food_waste_sources(raw_root: str) -> Dict[str, pd.DataFrame]:
    print_header("Loading food waste sources")
    files = find_files(raw_root, ["*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls"])
    frames = {}

    for fp in files:
        name = Path(fp).stem.lower()
        try:
            if fp.endswith(".csv"):
                df = safe_read_csv(fp)
            elif fp.endswith(".tsv") or fp.endswith(".txt"):
                df = safe_read_csv(fp, sep=None, engine="python")
            elif fp.endswith(".xlsx") or fp.endswith(".xls"):
                df = pd.read_excel(fp)
            else:
                continue

            df = normalize_colnames(df)
            if len(df) > 0:
                frames[name] = df
                print(f"Loaded: {fp} -> shape={df.shape}")
        except Exception as e:
            print(f"[WARN] Could not load {fp}: {e}")

    return frames


def standardize_food_waste_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = normalize_colnames(df)
    dt_col = infer_datetime_column(df)
    target_col = infer_numeric_target(df)

    if dt_col is None:
        df = df.copy()
        df["date"] = pd.date_range(start=CFG.start_date, periods=len(df), freq="D")
        dt_col = "date"
    else:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df[df[dt_col].notna()].copy()
        df = df.rename(columns={dt_col: "date"})

    if target_col is None:
        raise ValueError(f"No numeric waste-like target found in source={source_name}")

    possible_loc_cols = [
    c for c in df.columns
    if any(k in c for k in ["location", "site", "kitchen", "canteen", "restaurant", "facility"])
    ]
    if possible_loc_cols:
        loc_col = possible_loc_cols[0]
        df["location_id"] = df[loc_col].astype(str).fillna(source_name)
    else:
        df["location_id"] = source_name

    possible_type_cols = [c for c in df.columns if any(k in c for k in ["waste_type", "category", "bin", "material", "class"])]
    if possible_type_cols:
        type_col = possible_type_cols[0]
        df["waste_type"] = df[type_col].astype(str).fillna("food")
    else:
        df["waste_type"] = "food"

    out = df[["date", "location_id", target_col, "waste_type"]].copy()
    out = out.rename(columns={target_col: "waste_volume"})
    out["source"] = source_name

    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.groupby(["date", "location_id", "waste_type", "source"], as_index=False)["waste_volume"].sum()
    out["date"] = pd.to_datetime(out["date"])
    return out


def build_food_waste_master_table(raw_root: str = CFG.raw_dir) -> pd.DataFrame:
    frames = load_food_waste_sources(raw_root)
    standardized = []

    for name, df in frames.items():
        try:
            std = standardize_food_waste_df(df, name)
            standardized.append(std)
        except Exception as e:
            print(f"[WARN] Could not standardize {name}: {e}")

    if not standardized:
        raise RuntimeError("No food waste sources could be standardized. Check data/raw.")

    master = pd.concat(standardized, ignore_index=True)
    master = master.sort_values(["location_id", "date"]).reset_index(drop=True)

    os.makedirs(CFG.processed_dir, exist_ok=True)
    master.to_csv(os.path.join(CFG.processed_dir, "food_waste_master.csv"), index=False)
    print(f"Saved master food waste table with shape {master.shape}")
    return master


def load_weather_if_available(raw_root: str = CFG.raw_dir) -> Optional[pd.DataFrame]:
    fp = os.path.join(raw_root, "weather_visual_crossing.csv")
    if os.path.exists(fp):
        df = safe_read_csv(fp)
        df = normalize_colnames(df)
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        elif "date" not in df.columns:
            dt_col = infer_datetime_column(df)
            if dt_col:
                df = df.rename(columns={dt_col: "date"})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            return df
    return None
