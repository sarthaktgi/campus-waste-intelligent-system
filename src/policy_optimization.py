import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import CFG
from src.utils import print_header

try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except Exception:
    ORTOOLS_AVAILABLE = False


def merge_for_policy(campus_df: pd.DataFrame, contamination_preds: pd.DataFrame) -> pd.DataFrame:
    merged = campus_df.merge(
        contamination_preds[["timestamp", "bin_id", "p_contam", "contam_pred"]],
        on=["timestamp", "bin_id"],
        how="left",
    )
    merged["p_contam"] = merged["p_contam"].fillna(merged["p_contam_true_latent"])
    merged["contam_pred"] = merged["contam_pred"].fillna((merged["p_contam"] >= 0.5).astype(int))
    return merged


def compute_operational_costs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hauling_cost = 12.0 / df["pickup_frequency"].clip(lower=1)
    contamination_cost = 60.0 * df["p_contam"]
    overflow_penalty = 80.0 * np.clip(df["expected_fill"] - 1.0, 0, None)
    carbon_proxy = 8.0 * df["predicted_waste_volume"] / 100.0

    df["cost_hauling"] = hauling_cost
    df["cost_contamination"] = contamination_cost
    df["cost_overflow"] = overflow_penalty
    df["cost_carbon_proxy"] = carbon_proxy
    df["total_cost"] = df[["cost_hauling", "cost_contamination", "cost_overflow", "cost_carbon_proxy"]].sum(axis=1)
    return df


def simulate_intervention(df: pd.DataFrame, intervention: str) -> pd.DataFrame:
    out = df.copy()

    if intervention == "baseline":
        return compute_operational_costs(out)

    if intervention == "increase_pickup_high_risk":
        mask = (out["p_contam"] > 0.45) | (out["expected_fill"] > 0.9)
        out.loc[mask, "pickup_frequency"] = np.minimum(out.loc[mask, "pickup_frequency"] + 1, 4)
        out.loc[mask, "expected_fill"] = out.loc[mask, "expected_fill"] * 0.85
        out.loc[mask, "p_contam"] = np.clip(out.loc[mask, "p_contam"] * 0.92, 0, 1)

    elif intervention == "improved_signage":
        mask = out["location_type"].isin(["dorm", "event", "academic"])
        out.loc[mask, "signage_strength"] = np.minimum(out.loc[mask, "signage_strength"] + 1, 2)
        out.loc[mask, "p_contam"] = np.clip(out.loc[mask, "p_contam"] * 0.85, 0, 1)

    elif intervention == "add_bins_high_traffic":
        mask = out["expected_fill"] > 0.95
        out.loc[mask, "bin_capacity"] = out.loc[mask, "bin_capacity"] * 1.25
        out.loc[mask, "expected_fill"] = out.loc[mask, "expected_fill"] / 1.25
        out.loc[mask, "p_contam"] = np.clip(out.loc[mask, "p_contam"] * 0.88, 0, 1)

    elif intervention == "paired_bins_strategy":
        mask = out["location_type"].isin(["academic", "event", "dorm"])
        out.loc[mask, "alt_bin_distance"] = out.loc[mask, "alt_bin_distance"] * 0.75
        out.loc[mask, "p_contam"] = np.clip(out.loc[mask, "p_contam"] * 0.83, 0, 1)

    elif intervention == "combined_strategy":
        out = simulate_intervention(out, "increase_pickup_high_risk")
        out = simulate_intervention(out, "improved_signage")
        out = simulate_intervention(out, "add_bins_high_traffic")
        out = simulate_intervention(out, "paired_bins_strategy")
        return compute_operational_costs(out)

    return compute_operational_costs(out)


def greedy_budgeted_actions(df: pd.DataFrame, budget: float = 1500.0) -> pd.DataFrame:
    temp = df.copy()
    temp["benefit_score"] = 65.0 * temp["p_contam"] + 50.0 * np.clip(temp["expected_fill"] - 0.9, 0, None)
    temp["action_cost"] = 45.0 + 15.0 * (temp["location_type"].isin(["event", "dorm"]).astype(int))
    temp = temp.sort_values("benefit_score", ascending=False).copy()
    temp["optimized_action"] = 0

    spent = 0.0
    chosen_idx = []
    for idx, row in temp.iterrows():
        c = float(row["action_cost"])
        if spent + c <= budget:
            spent += c
            chosen_idx.append(idx)

    temp.loc[chosen_idx, "optimized_action"] = 1

    mask = temp["optimized_action"] == 1
    temp.loc[mask, "pickup_frequency"] = np.minimum(temp.loc[mask, "pickup_frequency"] + 1, 4)
    temp.loc[mask, "signage_strength"] = np.minimum(temp.loc[mask, "signage_strength"] + 1, 2)
    temp.loc[mask, "bin_capacity"] = temp.loc[mask, "bin_capacity"] * 1.15
    temp.loc[mask, "expected_fill"] = temp.loc[mask, "expected_fill"] * 0.82
    temp.loc[mask, "p_contam"] = np.clip(temp.loc[mask, "p_contam"] * 0.80, 0, 1)

    return compute_operational_costs(temp)


def optimize_budgeted_actions(df: pd.DataFrame, budget: float = 1500.0) -> pd.DataFrame:
    if not ORTOOLS_AVAILABLE:
        print("[INFO] OR-Tools unavailable; using greedy strategy.")
        return greedy_budgeted_actions(df, budget)

    print_header("Running budgeted optimization")
    temp = df.copy().reset_index(drop=True)
    temp["benefit"] = 65.0 * temp["p_contam"] + 50.0 * np.clip(temp["expected_fill"] - 0.9, 0, None)
    temp["action_cost"] = 45.0 + 15.0 * (temp["location_type"].isin(["event", "dorm"]).astype(int))

    solver = pywraplp.Solver.CreateSolver("SCIP")
    x = {i: solver.IntVar(0, 1, f"x_{i}") for i in temp.index}

    solver.Add(sum(x[i] * float(temp.loc[i, "action_cost"]) for i in temp.index) <= budget)
    solver.Maximize(sum(x[i] * float(temp.loc[i, "benefit"]) for i in temp.index))

    status = solver.Solve()
    if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        print("[WARN] Optimization failed; using greedy fallback.")
        return greedy_budgeted_actions(df, budget)

    chosen = [i for i in temp.index if x[i].solution_value() > 0.5]
    temp["optimized_action"] = 0
    temp.loc[chosen, "optimized_action"] = 1

    mask = temp["optimized_action"] == 1
    temp.loc[mask, "pickup_frequency"] = np.minimum(temp.loc[mask, "pickup_frequency"] + 1, 4)
    temp.loc[mask, "signage_strength"] = np.minimum(temp.loc[mask, "signage_strength"] + 1, 2)
    temp.loc[mask, "bin_capacity"] = temp.loc[mask, "bin_capacity"] * 1.15
    temp.loc[mask, "expected_fill"] = temp.loc[mask, "expected_fill"] * 0.82
    temp.loc[mask, "p_contam"] = np.clip(temp.loc[mask, "p_contam"] * 0.80, 0, 1)

    return compute_operational_costs(temp)


def run_policy_layer(campus_df: pd.DataFrame, contamination_preds: pd.DataFrame) -> pd.DataFrame:
    print_header("Running policy layer")
    merged = merge_for_policy(campus_df, contamination_preds)

    scenarios = {}
    for intervention in [
        "baseline",
        "increase_pickup_high_risk",
        "improved_signage",
        "add_bins_high_traffic",
        "paired_bins_strategy",
        "combined_strategy",
    ]:
        scenarios[intervention] = simulate_intervention(merged, intervention)

    scenarios["budget_optimized"] = optimize_budgeted_actions(merged, budget=1500.0)

    summary_rows = []
    baseline_cost = scenarios["baseline"]["total_cost"].sum()
    baseline_contam = scenarios["baseline"]["p_contam"].mean()

    for name, sdf in scenarios.items():
        total_cost = sdf["total_cost"].sum()
        mean_contam = sdf["p_contam"].mean()
        mean_fill = sdf["expected_fill"].mean()
        summary_rows.append({
            "scenario": name,
            "total_cost": total_cost,
            "cost_savings_vs_baseline": baseline_cost - total_cost,
            "mean_contamination": mean_contam,
            "contamination_reduction_vs_baseline": baseline_contam - mean_contam,
            "mean_expected_fill": mean_fill,
        })

    summary = pd.DataFrame(summary_rows).sort_values("total_cost")

    os.makedirs(CFG.outputs_dir, exist_ok=True)
    os.makedirs(CFG.figures_dir, exist_ok=True)

    summary.to_csv(os.path.join(CFG.outputs_dir, "policy_results.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(summary["scenario"], summary["cost_savings_vs_baseline"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Cost savings vs baseline")
    plt.title("Policy Scenario Cost Savings")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.figures_dir, "policy_cost_savings.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(summary["scenario"], summary["contamination_reduction_vs_baseline"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Contamination reduction vs baseline")
    plt.title("Policy Scenario Impact on Contamination")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.figures_dir, "policy_contamination_reduction.png"))
    plt.close()

    return summary
