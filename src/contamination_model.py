import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.config import CFG
from src.utils import print_header


def train_contamination_model(campus_df: pd.DataFrame) -> pd.DataFrame:
    print_header("Training contamination risk model")
    df = campus_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    feature_cols = [
        "bin_type",
        "location_type",
        "predicted_waste_volume",
        "event_flag",
        "event_intensity",
        "bin_capacity",
        "pickup_frequency",
        "alt_bin_distance",
        "signage_strength",
        "crowdedness",
        "expected_fill",
        "overflow_risk",
        "day_of_week",
        "month",
    ]
    target_col = "contam_true"

    dates = sorted(df["timestamp"].dt.date.unique())
    n = len(dates)
    train_cut = int(n * 0.7)
    val_cut = int(n * 0.85)

    train_dates = dates[:train_cut]
    val_dates = dates[train_cut:val_cut]
    test_dates = dates[val_cut:]

    train_df = df[df["timestamp"].dt.date.isin(train_dates)].copy()
    val_df = df[df["timestamp"].dt.date.isin(val_dates)].copy()
    test_df = df[df["timestamp"].dt.date.isin(test_dates)].copy()

    cat_cols = ["bin_type", "location_type"]
    encoders = {}

    transformed_train = train_df[feature_cols].copy()
    transformed_val = val_df[feature_cols].copy()
    transformed_test = test_df[feature_cols].copy()

    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([
            transformed_train[col].astype(str),
            transformed_val[col].astype(str),
            transformed_test[col].astype(str)
        ], axis=0)
        le.fit(combined)
        transformed_train[col] = le.transform(transformed_train[col].astype(str))
        transformed_val[col] = le.transform(transformed_val[col].astype(str))
        transformed_test[col] = le.transform(transformed_test[col].astype(str))
        encoders[col] = le

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    )
    xgb.fit(transformed_train[feature_cols], train_df[target_col])

    calibrator = CalibratedClassifierCV(xgb, method="isotonic", cv="prefit")
    calibrator.fit(transformed_val[feature_cols], val_df[target_col])

    p_test = calibrator.predict_proba(transformed_test[feature_cols])[:, 1]
    y_test_pred = (p_test >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(test_df[target_col], p_test),
        "f1": f1_score(test_df[target_col], y_test_pred),
        "brier": brier_score_loss(test_df[target_col], p_test),
    }
    print("Contamination model metrics:", metrics)

    results = test_df[["timestamp", "bin_id", "location_id", "contam_true"]].copy()
    results["p_contam"] = p_test
    results["contam_pred"] = y_test_pred

    os.makedirs(CFG.outputs_dir, exist_ok=True)
    os.makedirs(CFG.figures_dir, exist_ok=True)

    results.to_csv(os.path.join(CFG.outputs_dir, "contamination_predictions.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(CFG.outputs_dir, "contamination_metrics.csv"), index=False)

    prob_true, prob_pred = calibration_curve(test_df[target_col], p_test, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Contamination Model Reliability Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.figures_dir, "contamination_reliability_curve.png"))
    plt.close()

    return results
