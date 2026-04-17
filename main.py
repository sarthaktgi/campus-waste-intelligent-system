import os

from src.config import CFG
from src.data_ingestion import build_food_waste_master_table
from src.forecasting import run_forecasting_pipeline
from src.synthetic_data import create_synthetic_campus_table
from src.contamination_model import train_contamination_model
from src.policy_optimization import run_policy_layer


def ensure_dirs():
    for path in [CFG.raw_dir, CFG.processed_dir, CFG.outputs_dir, CFG.figures_dir, CFG.models_dir]:
        os.makedirs(path, exist_ok=True)


def main():
    ensure_dirs()

    master_df = build_food_waste_master_table(CFG.raw_dir)
    forecast_df = run_forecasting_pipeline(master_df)
    campus_df = create_synthetic_campus_table(forecast_df)
    contamination_preds = train_contamination_model(campus_df)
    policy_summary = run_policy_layer(campus_df, contamination_preds)

    print("\nPipeline completed successfully.")
    print(policy_summary)


if __name__ == "__main__":
    main()
