import os
import numpy as np
import pandas as pd

from src.config import CFG
from src.utils import print_header
from src.forecasting import make_synthetic_event_calendar


def create_synthetic_campus_table(forecast_df: pd.DataFrame) -> pd.DataFrame:
    print_header("Creating synthetic campus bin-level data")
    forecast_df = forecast_df.copy()
    forecast_df["date"] = pd.to_datetime(forecast_df["date"]).dt.normalize()

    context = make_synthetic_event_calendar(
        forecast_df["date"].min().date().isoformat(),
        forecast_df["date"].max().date().isoformat()
    )
    df = forecast_df.merge(context, on="date", how="left")

    df["location_type"] = df["location_id"].astype(str).apply(
        lambda x: "dining" if "dining" in x.lower() or "canteen" in x.lower() else (
            "dorm" if "dorm" in x.lower() else (
                "event" if "event" in x.lower() or "stadium" in x.lower() else "academic"
            )
        )
    )

    rows = []
    rng = np.random.default_rng(42)

    for _, row in df.iterrows():
        num_bins = 3 if row["location_type"] == "dining" else 2
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
                1.2 if row["location_type"] == "dining" else 1.0
            )

            waste_share = {"food": 0.55, "recycle": 0.25, "landfill": 0.20}[bin_type]
            expected_fill = max(0, row["y_pred"] * waste_share / max(bin_capacity, 1))
            overflow_risk = max(0, expected_fill - 1.0)

            p_contam = (
                0.08
                + 0.12 * overflow_risk
                + 0.10 * row["event_flag"]
                + 0.08 * (row["location_type"] == "event")
                + 0.06 * (row["location_type"] == "dorm")
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
                "location_id": row["location_id"],
                "bin_id": f"{row['location_id']}_bin_{bin_idx+1}",
                "bin_type": bin_type,
                "location_type": row["location_type"],
                "predicted_waste_volume": row["y_pred"],
                "event_flag": row["event_flag"],
                "event_intensity": row["event_intensity"],
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
