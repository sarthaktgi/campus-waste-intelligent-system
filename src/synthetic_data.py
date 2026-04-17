import os
import numpy as np
import pandas as pd

from src.config import CFG
from src.utils import print_header
from src.forecasting import make_synthetic_event_calendar


CAMPUS_ZONES = {
    "Dining Hall A": {"type": "dining", "waste_multiplier": 1.30, "bins": 4},
    "Dining Hall B": {"type": "dining", "waste_multiplier": 1.15, "bins": 4},
    "Dorm North": {"type": "dorm", "waste_multiplier": 0.80, "bins": 3},
    "Dorm South": {"type": "dorm", "waste_multiplier": 0.75, "bins": 3},
    "Academic Center": {"type": "academic", "waste_multiplier": 0.60, "bins": 3},
    "Student Union": {"type": "academic", "waste_multiplier": 0.90, "bins": 4},
    "Event Arena": {"type": "event", "waste_multiplier": 1.10, "bins": 5},
}


def create_synthetic_campus_table(forecast_df: pd.DataFrame) -> pd.DataFrame:
    print_header("Creating synthetic campus bin-level data")

    forecast_df = forecast_df.copy()
    forecast_df["date"] = pd.to_datetime(forecast_df["date"]).dt.normalize()

    # Optional: compress dates into a cleaner campus study window
    unique_dates = sorted(forecast_df["date"].unique())
    remapped_dates = pd.date_range("2025-01-01", periods=len(unique_dates), freq="D")
    date_map = dict(zip(unique_dates, remapped_dates))
    forecast_df["date"] = forecast_df["date"].map(date_map)

    context = make_synthetic_event_calendar(
        forecast_df["date"].min().date().isoformat(),
        forecast_df["date"].max().date().isoformat()
    )
    forecast_df = forecast_df.merge(context, on="date", how="left")

    rows = []
    rng = np.random.default_rng(42)

    for _, row in forecast_df.iterrows():
        base_pred = row["y_pred"]

        for zone_name, zone_info in CAMPUS_ZONES.items():
            location_type = zone_info["type"]
            zone_pred = base_pred * zone_info["waste_multiplier"] / 6.0  # scaled down to campus zone level
            num_bins = zone_info["bins"]

            for bin_idx in range(num_bins):
                bin_type = ["food", "recycle", "landfill"][bin_idx % 3]

                bin_capacity = {
                    "food": rng.integers(50, 90),
                    "recycle": rng.integers(40, 80),
                    "landfill": rng.integers(45, 85),
                }[bin_type]

                pickup_frequency = rng.choice([1, 2, 3])
                alt_bin_distance = rng.uniform(2, 15)
                signage_strength = rng.choice([0, 1, 2])

                crowdedness = (1 + 0.4 * row["event_flag"] + rng.normal(0, 0.1)) * (
                    1.2 if location_type == "dining" else 1.0
                )

                waste_share = {"food": 0.55, "recycle": 0.25, "landfill": 0.20}[bin_type]
                expected_fill = max(0, zone_pred * waste_share / max(bin_capacity, 1))
                overflow_risk = max(0, expected_fill - 1.0)

                p_contam = (
                    0.08
                    + 0.12 * overflow_risk
                    + 0.10 * row["event_flag"]
                    + 0.08 * (location_type == "event")
                    + 0.06 * (location_type == "dorm")
                    + 0.05 * (alt_bin_distance / 15)
                    + 0.08 * (pickup_frequency == 1)
                    - 0.05 * signage_strength
                    + rng.normal(0, 0.03)
                )
                p_contam = float(np.clip(p_contam, 0.01, 0.95))
                contam_true = int(rng.random() < p_contam)
                contam_pct_true = float(np.clip(p_contam + rng.normal(0, 0.05), 0.0, 1.0))

                rows.append({
                    "timestamp": row["date"],
                    "location_id": zone_name,
                    "bin_id": f"{zone_name.replace(' ', '_')}_bin_{bin_idx+1}",
                    "bin_type": bin_type,
                    "location_type": location_type,
                    "predicted_waste_volume": float(zone_pred),
                    "event_flag": int(row["event_flag"]),
                    "event_intensity": int(row["event_intensity"]),
                    "bin_capacity": int(bin_capacity),
                    "pickup_frequency": int(pickup_frequency),
                    "alt_bin_distance": float(alt_bin_distance),
                    "signage_strength": int(signage_strength),
                    "crowdedness": float(crowdedness),
                    "expected_fill": float(expected_fill),
                    "overflow_risk": float(overflow_risk),
                    "p_contam_true_latent": float(p_contam),
                    "contam_true": int(contam_true),
                    "contam_pct_true": float(contam_pct_true),
                })

    campus = pd.DataFrame(rows)

    os.makedirs(CFG.processed_dir, exist_ok=True)
    campus.to_csv(os.path.join(CFG.processed_dir, "synthetic_campus_bins.csv"), index=False)
    return campus
