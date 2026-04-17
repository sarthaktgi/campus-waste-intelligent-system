from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Config:
    base_dir: str = "."
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    outputs_dir: str = "outputs"
    figures_dir: str = "figures"
    models_dir: str = "models"

    kaggle_food_waste: str = "joebeachcapital/food-waste"
    kaggle_waste_classification: str = "techsash/waste-classification-data"
    fikwaste_url: str = "https://osf.io/download/tyaj6/"
    trashnet_zip_url: str = "https://github.com/garythung/trashnet/archive/refs/heads/master.zip"
    realwaste_kaggle: str = "joebeachcapital/realwaste"

    visual_crossing_api_key: Optional[str] = None
    weather_location: str = "Birmingham,AL"
    weather_start_date: str = "2024-01-01"
    weather_end_date: str = "2024-12-31"

    roboflow_api_key: Optional[str] = None
    roboflow_workspace: Optional[str] = None
    roboflow_project: Optional[str] = None
    roboflow_version: Optional[int] = None

    campus_locations: Tuple[str, ...] = ("dining_hall", "dorm", "academic", "event_venue")
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    use_hourly: bool = False

    forecast_target_name: str = "waste_volume"
    contamination_threshold: float = 0.25


CFG = Config()
