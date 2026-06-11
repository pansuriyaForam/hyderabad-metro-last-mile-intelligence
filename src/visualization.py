"""
src/visualization.py
────────────────────────────────────────────────────────────
Visualization engine for the Hyderabad Metro
Last-Mile Connectivity project.

Purpose
-------
Generate clean, dashboard-safe, investor-ready visuals from:
- LMCI outputs
- MCLP outputs
- scoring outputs
- simulation outputs

Outputs
-------
assets/plots/
├── lmci_distribution.png
├── top_priority_stations.png
├── mismatch_distribution.png
├── temporal_gap_analysis.png
├── mclp_coverage_curve.png
├── intervention_ranking.png
├── simulation_impact.png
├── network_summary_dashboard.png
├── top5_conversion_insights.png
└── priority_map.png

Design philosophy
-----------------
- readable
- deterministic
- presentation-safe
- no notebook clutter
- publication friendly
- Streamlit compatible
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s — %(message)s",
)

logger = logging.getLogger("VISUALIZATION")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class VisualizationConfig:
    """Visualization configuration."""

    outputs_dir: Path = Path("outputs")
    assets_dir: Path = Path("assets")
    plots_dir: Path = Path("assets/plots")

    station_priority_path: Path = Path("outputs/station_priority_scores.csv")
    mismatch_path: Path = Path("outputs/demand_service_mismatch.csv")
    mclp_coverage_path: Path = Path("outputs/mclp_coverage_by_k.csv")
    simulation_station_path: Path = Path("outputs/simulation_station_impacts.csv")
    simulation_ranking_path: Path = Path("outputs/simulation_intervention_ranking.csv")
    executive_summary_path: Path = Path("outputs/executive_summary_metrics.csv")
    conversion_insights_path: Path = Path("outputs/conversion_insights_top5.csv")

    figsize_large: tuple = (14, 8)
    figsize_medium: tuple = (10, 6)
    figsize_square: tuple = (10, 10)

    dpi: int = 300

    def __post_init__(self):
        self.plots_dir.mkdir(parents=True, exist_ok=True)


CFG = VisualizationConfig()


# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required visualization input missing: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    logger.info(f"Loaded {path.name}: {len(df):,} rows")
    return df



def load_visualization_inputs(cfg: VisualizationConfig = CFG) -> Dict[str, pd.DataFrame]:
    """Load all required visualization datasets."""

    return {
        "station_priority": _load_csv(cfg.station_priority_path),
        "mismatch": _load_csv(cfg.mismatch_path),
        "mclp_coverage": _load_csv(cfg.mclp_coverage_path),
        "simulation_station": _load_csv(cfg.simulation_station_path),
        "simulation_ranking": _load_csv(cfg.simulation_ranking_path),
        "executive_summary": _load_csv(cfg.executive_summary_path),
        "conversion_insights": _load_csv(cfg.conversion_insights_path),
    }


# ─────────────────────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────────────────────

def save_plot(fig, path: Path, cfg: VisualizationConfig = CFG):
    """Save figure safely."""

    fig.tight_layout()
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved plot: {path}")


# ─────────────────────────────────────────────────────────────
# LMCI DISTRIBUTION
# ─────────────────────────────────────────────────────────────

def plot_lmci_distribution(
    station_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """Plot LMCI score distribution."""

    lmci_col = None
    for c in ["LMCI_new2", "LMCI_new", "LMCI_old", "LMCI_mean", "LMCI"]:
        if c in station_df.columns:
            lmci_col = c
            break

    if lmci_col is None:
        raise ValueError("No LMCI column available for plotting")

    fig, ax = plt.subplots(figsize=cfg.figsize_medium)

    ax.hist(
        station_df[lmci_col],
        bins=12,
        edgecolor="black",
    )

    ax.set_title("LMCI Score Distribution")
    ax.set_xlabel("LMCI Score")
    ax.set_ylabel("Number of Stations")

    path = cfg.plots_dir / "lmci_distribution.png"
    save_plot(fig, path, cfg)

    return path


# ─────────────────────────────────────────────────────────────
# TOP PRIORITY STATIONS
# ─────────────────────────────────────────────────────────────

def plot_top_priority_stations(
    station_df: pd.DataFrame,
    top_n: int = 10,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """Plot top intervention stations."""

    df = station_df.sort_values(
        "final_priority_score",
        ascending=False,
    ).head(top_n)

    fig, ax = plt.subplots(figsize=cfg.figsize_large)

    ax.barh(
        df["stop_name"],
        df["final_priority_score"],
    )

    ax.invert_yaxis()

    ax.set_title(f"Top {top_n} Priority Stations")
    ax.set_xlabel("Priority Score")
    ax.set_ylabel("Station")

    path = cfg.plots_dir / "top_priority_stations.png"
    save_plot(fig, path, cfg)

    return path


# ─────────────────────────────────────────────────────────────
# DEMAND-SERVICE MISMATCH
# ─────────────────────────────────────────────────────────────

def plot_mismatch_distribution(
    mismatch_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """Plot demand-service mismatch classes."""

    counts = mismatch_df["mismatch_class"].value_counts()

    fig, ax = plt.subplots(figsize=cfg.figsize_medium)

    ax.bar(
        counts.index,
        counts.values,
    )

    ax.set_title("Demand-Service Mismatch Distribution")
    ax.set_ylabel("Stations")
    ax.tick_params(axis="x", rotation=20)

    path = cfg.plots_dir / "mismatch_distribution.png"
    save_plot(fig, path, cfg)

    return path


# ─────────────────────────────────────────────────────────────
# TEMPORAL GAP ANALYSIS
# ─────────────────────────────────────────────────────────────

def plot_temporal_gap_analysis(
    station_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """Plot temporal instability across stations."""

    if "temporal_gap" not in station_df.columns:
        raise ValueError("temporal_gap column missing")

    df = station_df.sort_values("temporal_gap", ascending=False)

    fig, ax = plt.subplots(figsize=cfg.figsize_large)

    ax.plot(
        df["stop_name"],
        df["temporal_gap"],
        marker="o",
    )

    ax.set_title("Temporal Accessibility Gap")
    ax.set_ylabel("LMCI Gap")
    ax.tick_params(axis="x", rotation=90)

    path = cfg.plots_dir / "temporal_gap_analysis.png"
    save_plot(fig, path, cfg)

    return path


# ─────────────────────────────────────────────────────────────
# MCLP COVERAGE CURVE
# ─────────────────────────────────────────────────────────────

def plot_mclp_coverage_curve(
    coverage_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """Plot coverage vs number of intervention stations."""

    fig, ax = plt.subplots(figsize=cfg.figsize_medium)

    ax.plot(
        coverage_df["k"],
        coverage_df["coverage_pct"],
        marker="o",
    )

    ax.set_title("MCLP Coverage Curve")
    ax.set_xlabel("Number of Selected Stations (k)")
    ax.set_ylabel("Coverage Percentage")

    path = cfg.plots_dir / "mclp_coverage_curve.png"
    save_plot(fig, path, cfg)

    return path


# ─────────────────────────────────────────────────────────────
# INTERVENTION TYPE RANKING
# ─────────────────────────────────────────────────────────────

def plot_intervention_ranking(
    ranking_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """Plot intervention type effectiveness."""

    fig, ax = plt.subplots(figsize=cfg.figsize_large)

    ax.barh(
        ranking_df["scenario_name"],
        ranking_df["simulation_priority_score"],
    )

    ax.invert_yaxis()

    ax.set_title("Intervention Strategy Ranking")
    ax.set_xlabel("Simulation Priority Score")

    path = cfg.plots_dir / "intervention_ranking.png"
    save_plot(fig, path, cfg)

    return path


# ─────────────────────────────────────────────────────────────
# SIMULATION IMPACT
# ─────────────────────────────────────────────────────────────

def plot_simulation_impact(
    simulation_df: pd.DataFrame,
    top_n: int = 10,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """Plot top simulated interventions."""

    df = simulation_df.sort_values(
        "simulation_priority_score",
        ascending=False,
    ).head(top_n)

    fig, ax = plt.subplots(figsize=cfg.figsize_large)

    ax.barh(
        df["stop_name"],
        df["simulated_daily_ridership_gain"],
    )

    ax.invert_yaxis()

    ax.set_title("Estimated Ridership Gain from Top Interventions")
    ax.set_xlabel("Estimated Daily Ridership Gain")

    path = cfg.plots_dir / "simulation_impact.png"
    save_plot(fig, path, cfg)

    return path


# ─────────────────────────────────────────────────────────────
# NETWORK SUMMARY DASHBOARD
# ─────────────────────────────────────────────────────────────

def plot_network_summary_dashboard(
    summary_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """Create compact executive summary visual."""

    row = summary_df.iloc[0]

    metrics = {
        "Stations": row.get("stations_simulated", 0),
        "Daily Gain": int(row.get("estimated_daily_ridership_gain", 0)),
        "Conversion": int(row.get("estimated_daily_conversion_gain", 0)),
        "LMCI Gain": round(row.get("mean_lmci_gain", 0), 2),
        "Deserts": row.get("persistent_deserts_targeted", 0),
    }

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.axis("off")

    text = "\n".join([
        f"{k}: {v}" for k, v in metrics.items()
    ])

    ax.text(
        0.5,
        0.5,
        text,
        fontsize=18,
        ha="center",
        va="center",
    )

    ax.set_title("Network Simulation Executive Summary")

    path = cfg.plots_dir / "network_summary_dashboard.png"
    save_plot(fig, path, cfg)

    return path


# ─────────────────────────────────────────────────────────────
# TOP 5 CONVERSION INSIGHTS
# ─────────────────────────────────────────────────────────────

def plot_top5_conversion_insights(
    insights_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """Plot top 5 conversion insights."""

    df = insights_df.head(5)

    fig, ax = plt.subplots(figsize=cfg.figsize_large)

    ax.barh(
        df["stop_name"],
        df["final_priority_score"],
    )

    ax.invert_yaxis()

    ax.set_title("Top 5 Conversion Insights")
    ax.set_xlabel("Priority Score")

    path = cfg.plots_dir / "top5_conversion_insights.png"
    save_plot(fig, path, cfg)

    return path


# ─────────────────────────────────────────────────────────────
# PRIORITY MAP
# ─────────────────────────────────────────────────────────────

def plot_priority_map(
    station_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """Plot geospatial priority map."""

    required = {"stop_lat", "stop_lon", "final_priority_score"}

    if not required.issubset(station_df.columns):
        raise ValueError("Missing columns for priority map")

    gdf = gpd.GeoDataFrame(
        station_df.copy(),
        geometry=gpd.points_from_xy(
            station_df["stop_lon"],
            station_df["stop_lat"],
        ),
        crs="EPSG:4326",
    )

    fig, ax = plt.subplots(figsize=cfg.figsize_square)

    gdf.plot(
        column="final_priority_score",
        legend=True,
        markersize=60,
        ax=ax,
    )

    ax.set_title("Station Priority Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    path = cfg.plots_dir / "priority_map.png"
    save_plot(fig, path, cfg)

    return path


# ─────────────────────────────────────────────────────────────
# GENERATE ALL VISUALS
# ─────────────────────────────────────────────────────────────

def generate_all_visualizations(
    cfg: VisualizationConfig = CFG,
) -> Dict[str, Path]:
    """Generate all project visualizations."""

    logger.info("=" * 70)
    logger.info("STARTING VISUALIZATION PIPELINE")
    logger.info("=" * 70)

    data = load_visualization_inputs(cfg)

    paths = {}

    paths["lmci_distribution"] = plot_lmci_distribution(data["station_priority"], cfg)

    paths["top_priority_stations"] = plot_top_priority_stations(data["station_priority"], cfg=cfg)

    paths["mismatch_distribution"] = plot_mismatch_distribution(data["mismatch"], cfg)

    paths["temporal_gap_analysis"] = plot_temporal_gap_analysis(data["station_priority"], cfg)

    paths["mclp_coverage_curve"] = plot_mclp_coverage_curve(data["mclp_coverage"], cfg)

    paths["intervention_ranking"] = plot_intervention_ranking(data["simulation_ranking"], cfg)

    paths["simulation_impact"] = plot_simulation_impact(data["simulation_station"], cfg=cfg)

    paths["network_summary_dashboard"] = plot_network_summary_dashboard(data["executive_summary"], cfg)

    paths["top5_conversion_insights"] = plot_top5_conversion_insights(data["conversion_insights"], cfg)

    paths["priority_map"] = plot_priority_map(data["station_priority"], cfg)

    logger.info("=" * 70)
    logger.info("VISUALIZATION PIPELINE COMPLETE")
    logger.info(f"Generated plots: {len(paths)}")
    logger.info("=" * 70)

    return paths


if __name__ == "__main__":
    generate_all_visualizations()