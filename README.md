# Campus Waste Intelligent System

This project implements a modular pipeline for modeling and improving waste management in a simulated campus environment.

## Project Components
- Data ingestion and preprocessing
- Waste forecasting (XGBoost, LightGBM)
- Synthetic campus bin-level data generation
- Contamination risk modeling
- Policy optimization

## Repository Structure
- `src/` — core source code modules
- `notebooks/` — notebook version of the workflow
- `data/` — project data files
- `outputs/` — generated CSV outputs
- `figures/` — generated visualizations
- `main.py` — end-to-end pipeline runner

## Key Results
- Forecasting RMSE ~27 (XGBoost best)
- Contamination model AUC ~0.66
- Policy optimization reduces cost and contamination vs baseline

## Figures
- `figures/forecast_vs_actual.png`
- `figures/contamination_reliability_curve.png`
- `figures/policy_cost_savings.png`
- `figures/policy_contamination_reduction.png`

## How to Run

```bash
pip install -r requirements.txt
python main.py
