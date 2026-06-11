"""
src/simulation.py
────────────────────────────────────────────────────────────
Scenario modelling engine for the Hyderabad Metro
Urban Mobility Accessibility & Decision Support Platform.

Purpose
-------
Model the relative accessibility impact of hypothetical interventions:
- feeder routes
- e-rickshaw loops
- bus frequency improvements
- micromobility access
- multimodal integration upgrades

This module DOES NOT modify GTFS and does NOT produce ridership forecasts.
It performs explainable analytical simulations on top of:
- LMCI (Last-Mile Connectivity Index)
- MCLP (Maximum Coverage Location Problem)
- scoring outputs

All outputs are indicative accessibility estimates based on multimodal
connectivity assumptions — not AFC/smartcard or passenger count data.

Core philosophy:
- deterministic
- explainable
- investor/demo-safe
- policy-friendly

Inputs
------
outputs/station_priority_scores.csv
outputs/mclp_selected_stations.csv
outputs/station_lmci_summary.csv
outputs/lmci_station_scores.csv

Exports
-------
outputs/simulation_station_impacts.csv
outputs/simulation_network_summary.csv
outputs/simulation_intervention_ranking.csv
outputs/simulation_scenarios.csv
outputs/simulation_audit_report.csv
assets/simulation_station_impacts.geojson
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    from src.preprocessing import minmax_norm, to_gdf
except ImportError:
    from preprocessing import minmax_norm, to_gdf


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s — %(message)s",
)

logger = logging.getLogger("SIMULATION")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """Configuration for simulation engine."""

    outputs_dir: Path = Path("outputs")
    assets_dir: Path = Path("assets")

    station_priority_path: Path = Path("outputs/station_priority_scores.csv")
    mclp_selected_path: Path = Path("outputs/mclp_selected_stations.csv")
    station_summary_path: Path = Path("outputs/station_lmci_summary.csv")
    lmci_station_scores_path: Path = Path("outputs/lmci_station_scores.csv")

    metric_crs: int = 32644
    wgs84_crs: int = 4326

    # Scenario assumptions
    base_ridership_gain_factor: float = 0.08
    feeder_bonus_factor: float = 0.12
    multimodal_bonus_factor: float = 0.10
    temporal_stability_factor: float = 0.06
    desert_recovery_factor: float = 0.15

    # LMCI simulation assumptions
    lmci_improvement_cap: float = 3.0
    minimum_priority_threshold: float = 35.0

    # Network assumptions
    estimated_conversion_rate: float = 0.18
    estimated_daily_users_per_demand_point: int = 40

    # Scenario sizes
    top_interventions_count: int = 10

    def __post_init__(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)


CFG = SimulationConfig()


# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────

def _load_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning(f"Missing optional simulation input: {path}")
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    logger.info(f"Loaded {path.name}: {len(df):,} rows")
    return df



def load_simulation_inputs(cfg: SimulationConfig = CFG) -> Dict[str, Optional[pd.DataFrame]]:
    """Load all required simulation inputs."""

    station_priority = _load_csv_if_exists(cfg.station_priority_path)

    if station_priority is None:
        raise FileNotFoundError(
            "station_priority_scores.csv not found. Run src/scoring.py first."
        )

    return {
        "station_priority": station_priority,
        "mclp_selected": _load_csv_if_exists(cfg.mclp_selected_path),
        "station_summary": _load_csv_if_exists(cfg.station_summary_path),
        "lmci_station_scores": _load_csv_if_exists(cfg.lmci_station_scores_path),
    }


# ─────────────────────────────────────────────────────────────
# SCENARIO DEFINITIONS
# ─────────────────────────────────────────────────────────────

def define_intervention_scenarios() -> pd.DataFrame:
    """
    Define intervention catalog.

    These are intentionally explainable and policy-oriented.
    """

    scenarios = [
        {
            "scenario_id": "S1",
            "scenario_name": "Feeder Shuttle Integration",
            "intervention_type": "feeder",
            "lmci_boost": 2.2,
            "ridership_multiplier": 1.18,
            "network_effect": 1.12,
            "cost_band": "Medium",
        },
        {
            "scenario_id": "S2",
            "scenario_name": "E-Rickshaw Last-Mile Loop",
            "intervention_type": "micromobility",
            "lmci_boost": 1.4,
            "ridership_multiplier": 1.10,
            "network_effect": 1.05,
            "cost_band": "Low",
        },
        {
            "scenario_id": "S3",
            "scenario_name": "Bus Frequency Enhancement",
            "intervention_type": "frequency",
            "lmci_boost": 1.8,
            "ridership_multiplier": 1.14,
            "network_effect": 1.08,
            "cost_band": "Medium",
        },
        {
            "scenario_id": "S4",
            "scenario_name": "Integrated Multimodal Hub",
            "intervention_type": "multimodal",
            "lmci_boost": 2.8,
            "ridership_multiplier": 1.24,
            "network_effect": 1.16,
            "cost_band": "High",
        },
        {
            "scenario_id": "S5",
            "scenario_name": "Pedestrian Access Upgrade",
            "intervention_type": "walkability",
            "lmci_boost": 1.0,
            "ridership_multiplier": 1.06,
            "network_effect": 1.03,
            "cost_band": "Low",
        },
    ]

    return pd.DataFrame(scenarios)


# ─────────────────────────────────────────────────────────────
# SIGNAL EXTRACTION
# ─────────────────────────────────────────────────────────────

def enrich_station_inputs(
    station_df: pd.DataFrame,
    mclp_selected: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Prepare station-level simulation features."""

    out = station_df.copy()

    numeric_candidates = [
        "final_priority_score",
        "LMCI_old",
        "LMCI_new",
        "LMCI_new2",
        "LMCI_mean",
        "Morning_LMCI",
        "Midday_LMCI",
        "Evening_LMCI",
        "temporal_gap",
        "demand_signal",
        "weighted_demand_if_selected",
        "covered_demand_points_if_selected",
    ]

    for col in numeric_candidates:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    if "mclp_selected" not in out.columns:
        out["mclp_selected"] = False

    if mclp_selected is not None and not mclp_selected.empty:
        selected_ids = set(mclp_selected["stop_id"].astype(str))
        out["mclp_selected"] = out["stop_id"].astype(str).isin(selected_ids)

    if "desert_severity" not in out.columns:
        out["desert_severity"] = "None"

    return out


# ─────────────────────────────────────────────────────────────
# STATION IMPACT MODEL
# ─────────────────────────────────────────────────────────────

def estimate_station_impact(
    station_row: pd.Series,
    scenario_row: pd.Series,
    cfg: SimulationConfig = CFG,
) -> Dict[str, object]:
    """
    Model station-level accessibility impact for a given intervention scenario.

    This is NOT ridership forecasting.
    It is a relative accessibility simulation for planning and comparison.
    Outputs are indicative and based on multimodal connectivity assumptions only.
    """

    priority = float(station_row.get("final_priority_score", 0.0))

    if priority < cfg.minimum_priority_threshold:
        intervention_readiness = 0.25
    elif priority < 55:
        intervention_readiness = 0.50
    elif priority < 75:
        intervention_readiness = 0.75
    else:
        intervention_readiness = 1.0

    demand_signal = float(
        station_row.get(
            "demand_signal",
            station_row.get("weighted_demand_if_selected", 0.0),
        )
    )

    demand_norm = float(minmax_norm(pd.Series([0, demand_signal])).iloc[-1])

    current_lmci = float(
        station_row.get(
            "LMCI_new2",
            station_row.get("LMCI_new", station_row.get("LMCI_old", 0.0)),
        )
    )

    temporal_gap = float(station_row.get("temporal_gap", 0.0))

    is_desert = str(station_row.get("desert_severity", "None")).lower() == "persistent"
    is_mclp = bool(station_row.get("mclp_selected", False))

    scenario_boost = float(scenario_row["lmci_boost"])

    # Equity-sensitive impact scaling
    impact_multiplier = 1.0

    if is_desert:
        impact_multiplier += cfg.desert_recovery_factor

    if is_mclp:
        impact_multiplier += cfg.feeder_bonus_factor

    if temporal_gap > 1.5:
        impact_multiplier += cfg.temporal_stability_factor

    if scenario_row["intervention_type"] == "multimodal":
        impact_multiplier += cfg.multimodal_bonus_factor

    # Simulated LMCI improvement
    lmci_gain = min(
        scenario_boost * intervention_readiness * impact_multiplier,
        cfg.lmci_improvement_cap,
    )

    simulated_lmci = current_lmci + lmci_gain

    # Ridership effect
    base_users = max(
        demand_signal * cfg.estimated_daily_users_per_demand_point,
        100.0,
    )

    ridership_multiplier = float(scenario_row["ridership_multiplier"])

    simulated_ridership_gain = (
        base_users
        * cfg.base_ridership_gain_factor
        * ridership_multiplier
        * intervention_readiness
        * impact_multiplier
    )

    simulated_conversion_gain = (
        simulated_ridership_gain
        * cfg.estimated_conversion_rate
    )

    network_effect = float(scenario_row["network_effect"])

    network_value_score = (
        simulated_lmci
        * network_effect
        * impact_multiplier
    )

    return {
        "stop_id": station_row["stop_id"],
        "stop_name": station_row["stop_name"],
        "scenario_id": scenario_row["scenario_id"],
        "scenario_name": scenario_row["scenario_name"],
        "intervention_type": scenario_row["intervention_type"],
        "cost_band": scenario_row["cost_band"],
        "current_lmci": current_lmci,
        "simulated_lmci": simulated_lmci,
        "lmci_gain": lmci_gain,
        "priority_score": priority,
        "demand_signal": demand_signal,
        "intervention_readiness": intervention_readiness,
        "impact_multiplier": impact_multiplier,
        "simulated_daily_ridership_gain": simulated_ridership_gain,
        "simulated_conversion_gain": simulated_conversion_gain,
        "network_value_score": network_value_score,
        "mclp_selected": is_mclp,
        "desert_severity": station_row.get("desert_severity", "None"),
    }


# ─────────────────────────────────────────────────────────────
# RUN ALL SCENARIOS
# ─────────────────────────────────────────────────────────────

def run_station_scenarios(
    station_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    cfg: SimulationConfig = CFG,
) -> pd.DataFrame:
    """Run all intervention scenarios across all stations."""

    rows: List[Dict[str, object]] = []

    for _, station in station_df.iterrows():
        for _, scenario in scenarios_df.iterrows():
            rows.append(
                estimate_station_impact(
                    station_row=station,
                    scenario_row=scenario,
                    cfg=cfg,
                )
            )

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError("Simulation produced zero rows.")

    out["simulation_priority_score"] = (
        out["lmci_gain"] * 30.0
        + out["simulated_conversion_gain"] * 0.05
        + out["network_value_score"] * 4.0
    )

    out["simulation_priority_score"] = out["simulation_priority_score"].round(2)

    return out


# ─────────────────────────────────────────────────────────────
# BEST SCENARIO SELECTION
# ─────────────────────────────────────────────────────────────

def select_best_interventions(
    simulation_df: pd.DataFrame,
    cfg: SimulationConfig = CFG,
) -> pd.DataFrame:
    """Pick best intervention scenario per station."""

    out = (
        simulation_df
        .sort_values(
            ["stop_name", "simulation_priority_score"],
            ascending=[True, False],
        )
        .groupby("stop_name", as_index=False)
        .first()
    )

    out = out.sort_values(
        ["simulation_priority_score", "simulated_daily_ridership_gain"],
        ascending=False,
    ).reset_index(drop=True)

    out["intervention_rank"] = np.arange(1, len(out) + 1)

    return out


# ─────────────────────────────────────────────────────────────
# NETWORK SUMMARY
# ─────────────────────────────────────────────────────────────

def build_network_summary(
    best_df: pd.DataFrame,
    simulation_df: pd.DataFrame,
    cfg: SimulationConfig = CFG,
) -> pd.DataFrame:
    """Create executive simulation metrics."""

    top_interventions = best_df.head(cfg.top_interventions_count)

    total_daily_gain = float(
        top_interventions["simulated_daily_ridership_gain"].sum()
    )

    total_conversion_gain = float(
        top_interventions["simulated_conversion_gain"].sum()
    )

    mean_lmci_gain = float(
        top_interventions["lmci_gain"].mean()
    )

    persistent_deserts_targeted = int(
        top_interventions["desert_severity"]
        .astype(str)
        .str.lower()
        .eq("persistent")
        .sum()
    )

    multimodal_projects = int(
        top_interventions["intervention_type"]
        .astype(str)
        .str.lower()
        .eq("multimodal")
        .sum()
    )

    feeder_projects = int(
        top_interventions["intervention_type"]
        .astype(str)
        .str.lower()
        .eq("feeder")
        .sum()
    )

    summary = pd.DataFrame([
        {
            "stations_simulated": len(best_df),
            "scenarios_tested": simulation_df["scenario_id"].nunique(),
            "top_interventions_evaluated": len(top_interventions),
            "estimated_daily_ridership_gain": total_daily_gain,   # labelled "Accessibility Impact Index" in UI
            "estimated_daily_conversion_gain": total_conversion_gain,
            "mean_lmci_gain": mean_lmci_gain,
            "persistent_deserts_targeted": persistent_deserts_targeted,
            "multimodal_projects_recommended": multimodal_projects,
            "feeder_projects_recommended": feeder_projects,
        }
    ])

    return summary


# ─────────────────────────────────────────────────────────────
# SCENARIO RANKING
# ─────────────────────────────────────────────────────────────

def rank_intervention_types(
    simulation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate performance by intervention type."""

    grouped = (
        simulation_df
        .groupby(
            [
                "intervention_type",
                "scenario_name",
                "cost_band",
            ],
            as_index=False,
        )
        .agg({
            "lmci_gain": "mean",
            "simulated_daily_ridership_gain": "mean",
            "simulated_conversion_gain": "mean",
            "network_value_score": "mean",
            "simulation_priority_score": "mean",
        })
    )

    grouped = grouped.sort_values(
        "simulation_priority_score",
        ascending=False,
    ).reset_index(drop=True)

    grouped["intervention_type_rank"] = np.arange(1, len(grouped) + 1)

    return grouped


# ─────────────────────────────────────────────────────────────
# AUDIT REPORT
# ─────────────────────────────────────────────────────────────

def build_simulation_audit_report(
    simulation_df: pd.DataFrame,
    best_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create explainability audit report."""

    checks = []

    def add(name: str, passed: bool, detail: str):
        checks.append({
            "check": name,
            "passed": bool(passed),
            "detail": detail,
        })

    add(
        "simulation_rows_created",
        len(simulation_df) > 0,
        f"rows={len(simulation_df)}",
    )

    add(
        "best_interventions_available",
        len(best_df) > 0,
        f"rows={len(best_df)}",
    )

    add(
        "lmci_gain_positive",
        (simulation_df["lmci_gain"] >= 0).all(),
        "all lmci gains are non-negative",
    )

    add(
        "ridership_gain_positive",
        (simulation_df["simulated_daily_ridership_gain"] >= 0).all(),
        "all ridership gains are non-negative",
    )

    add(
        "scenario_diversity",
        simulation_df["scenario_id"].nunique() >= 3,
        f"unique scenarios={simulation_df['scenario_id'].nunique()}",
    )

    return pd.DataFrame(checks)


# ─────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────

def export_simulation_outputs(
    simulation_df: pd.DataFrame,
    best_df: pd.DataFrame,
    network_summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    cfg: SimulationConfig = CFG,
) -> Dict[str, Path]:
    """Export all simulation outputs."""

    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.assets_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    impacts_path = cfg.outputs_dir / "simulation_station_impacts.csv"
    best_df.to_csv(impacts_path, index=False)
    paths["simulation_station_impacts"] = impacts_path

    summary_path = cfg.outputs_dir / "simulation_network_summary.csv"
    network_summary_df.to_csv(summary_path, index=False)
    paths["simulation_network_summary"] = summary_path

    ranking_path = cfg.outputs_dir / "simulation_intervention_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)
    paths["simulation_intervention_ranking"] = ranking_path

    scenarios_path = cfg.outputs_dir / "simulation_scenarios.csv"
    simulation_df.to_csv(scenarios_path, index=False)
    paths["simulation_scenarios"] = scenarios_path

    audit_path = cfg.outputs_dir / "simulation_audit_report.csv"
    audit_df.to_csv(audit_path, index=False)
    paths["simulation_audit_report"] = audit_path

    try:
        geojson_df = best_df.copy()

        if {"stop_lat", "stop_lon"}.issubset(geojson_df.columns):
            gdf = to_gdf(
                geojson_df,
                lat_col="stop_lat",
                lon_col="stop_lon",
                crs_out=cfg.wgs84_crs,
            )

            geojson_path = cfg.assets_dir / "simulation_station_impacts.geojson"
            gdf.to_file(geojson_path, driver="GeoJSON")
            paths["simulation_station_impacts_geojson"] = geojson_path

    except Exception as exc:
        logger.warning(f"Simulation GeoJSON export skipped: {exc}")

    logger.info("Simulation exports complete:")

    for key, path in paths.items():
        logger.info(f"  {key}: {path}")

    return paths


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_simulation_outputs(best_df: pd.DataFrame) -> None:
    """Fail-fast validation for dashboard safety."""

    required = [
        "stop_id",
        "stop_name",
        "scenario_name",
        "simulated_lmci",
        "lmci_gain",
        "simulation_priority_score",
        "simulated_daily_ridership_gain",
    ]

    missing = [c for c in required if c not in best_df.columns]

    if missing:
        raise ValueError(f"Simulation output missing columns: {missing}")

    if best_df["simulation_priority_score"].isna().any():
        raise ValueError("simulation_priority_score contains NaN values")

    logger.info("Simulation validation passed.")


# ─────────────────────────────────────────────────────────────
# END-TO-END PIPELINE
# ─────────────────────────────────────────────────────────────

def run_simulation_pipeline(
    cfg: SimulationConfig = CFG,
) -> Dict[str, object]:
    """Run full simulation pipeline."""

    logger.info("=" * 70)
    logger.info("STARTING SIMULATION PIPELINE")
    logger.info("=" * 70)

    inputs = load_simulation_inputs(cfg)

    station_df = enrich_station_inputs(
        station_df=inputs["station_priority"],
        mclp_selected=inputs["mclp_selected"],
    )

    scenarios_df = define_intervention_scenarios()

    simulation_df = run_station_scenarios(
        station_df=station_df,
        scenarios_df=scenarios_df,
        cfg=cfg,
    )

    best_df = select_best_interventions(
        simulation_df=simulation_df,
        cfg=cfg,
    )

    network_summary_df = build_network_summary(
        best_df=best_df,
        simulation_df=simulation_df,
        cfg=cfg,
    )

    ranking_df = rank_intervention_types(simulation_df)

    audit_df = build_simulation_audit_report(
        simulation_df=simulation_df,
        best_df=best_df,
    )

    validate_simulation_outputs(best_df)

    export_paths = export_simulation_outputs(
        simulation_df=simulation_df,
        best_df=best_df,
        network_summary_df=network_summary_df,
        ranking_df=ranking_df,
        audit_df=audit_df,
        cfg=cfg,
    )

    logger.info("=" * 70)
    logger.info("SIMULATION PIPELINE COMPLETE")
    logger.info(f"Stations simulated: {len(best_df)}")
    logger.info(
        f"Top intervention: {best_df.iloc[0]['stop_name']} → "
        f"{best_df.iloc[0]['scenario_name']}"
    )
    logger.info("=" * 70)

    return {
        "simulation_df": simulation_df,
        "best_df": best_df,
        "network_summary_df": network_summary_df,
        "ranking_df": ranking_df,
        "audit_df": audit_df,
        "export_paths": export_paths,
    }


if __name__ == "__main__":
    run_simulation_pipeline()