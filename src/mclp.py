"""
src/mclp.py
────────────────────────────────────────────────────────────
MCLP engine for the Hyderabad Metro Last-Mile Connectivity project.

MCLP = Maximal Covering Location Problem

Purpose
-------
Select the best metro stations / intervention locations that maximize
weighted demand coverage within a service radius.

This module is intentionally deterministic and explainable:
- no solver dependency
- greedy approximation
- dashboard-safe CSV exports
- reproducible ranked recommendations

Inputs expected from previous modules
-------------------------------------
outputs/station_lmci_summary.csv
outputs/lmci_station_scores.csv
outputs/demand_points.csv

Exports
-------
outputs/mclp_selected_stations.csv
outputs/mclp_coverage_by_k.csv
outputs/mclp_demand_assignment.csv
outputs/mclp_candidate_scores.csv
assets/mclp_selected_stations.geojson
assets/mclp_demand_assignment.geojson
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

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

logger = logging.getLogger("MCLP")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class MCLPConfig:
    """Configuration for MCLP optimization."""

    outputs_dir: Path = Path("outputs")
    assets_dir: Path = Path("assets")

    station_scores_path: Path = Path("outputs/station_lmci_summary.csv")
    fallback_station_scores_path: Path = Path("outputs/lmci_station_scores.csv")
    demand_points_path: Path = Path("outputs/demand_points.csv")

    # CRS
    wgs84_crs: int = 4326
    metric_crs: int = 32644

    # Coverage assumptions
    coverage_radius_m: float = 800.0
    detour_factor: float = 1.35

    # Optimization defaults
    max_k: int = 10
    default_k: int = 5

    # Demand weighting
    business_weight: float = 1.00
    school_weight: float = 1.25
    default_demand_weight: float = 1.00

    # Candidate prioritization
    underserved_bonus: float = 1.20
    persistent_desert_bonus: float = 1.30

    # If true, candidate score considers both uncovered demand and LMCI deficit.
    use_equity_weighting: bool = True

    def __post_init__(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)


CFG = MCLPConfig()


# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────

def load_station_scores(cfg: MCLPConfig = CFG) -> pd.DataFrame:
    """Load station-level LMCI outputs."""

    if cfg.station_scores_path.exists():
        path = cfg.station_scores_path
    elif cfg.fallback_station_scores_path.exists():
        path = cfg.fallback_station_scores_path
    else:
        raise FileNotFoundError(
            "No station LMCI file found. Run src/lmci.py first. Expected one of: "
            f"{cfg.station_scores_path}, {cfg.fallback_station_scores_path}"
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Station score file missing columns: {missing}")

    df["stop_lat"] = pd.to_numeric(df["stop_lat"], errors="coerce")
    df["stop_lon"] = pd.to_numeric(df["stop_lon"], errors="coerce")
    df = df.dropna(subset=["stop_lat", "stop_lon"]).reset_index(drop=True)

    logger.info(f"Loaded station scores: {len(df):,} stations from {path}")
    return df



def load_demand_points(cfg: MCLPConfig = CFG) -> pd.DataFrame:
    """Load demand points created by preprocessing.py."""

    if not cfg.demand_points_path.exists():
        raise FileNotFoundError(
            f"Demand points not found: {cfg.demand_points_path}. "
            "Run src/preprocessing.py first."
        )

    df = pd.read_csv(cfg.demand_points_path)
    df.columns = df.columns.str.strip().str.lower()

    required = {"lat", "lon"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Demand file missing required columns: {missing}")

    if "type" not in df.columns:
        df["type"] = "unknown"

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Demand file is empty after coordinate cleaning.")

    df["demand_id"] = np.arange(len(df), dtype=int)

    logger.info(f"Loaded demand points: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────
# DEMAND WEIGHTING
# ─────────────────────────────────────────────────────────────

def assign_demand_weights(
    demand_df: pd.DataFrame,
    cfg: MCLPConfig = CFG,
) -> pd.DataFrame:
    """Assign weights to demand points by type."""

    out = demand_df.copy()

    def weight_for_type(value: str) -> float:
        text = str(value).lower().strip()

        if "school" in text:
            return cfg.school_weight
        if "business" in text or "poi" in text:
            return cfg.business_weight
        return cfg.default_demand_weight

    out["base_weight"] = out["type"].apply(weight_for_type)
    out["demand_weight"] = out["base_weight"].astype(float)

    return out



def enrich_station_candidates(
    stations_df: pd.DataFrame,
    cfg: MCLPConfig = CFG,
) -> pd.DataFrame:
    """
    Add candidate-level equity weights.

    Lower LMCI means higher intervention value.
    Persistent deserts get an additional policy-priority multiplier.
    """

    out = stations_df.copy()

    # Pick best available LMCI column.
    lmci_col = None
    for candidate in ["LMCI_new2", "LMCI_new", "LMCI_old", "LMCI_mean", "LMCI"]:
        if candidate in out.columns:
            lmci_col = candidate
            break

    if lmci_col is None:
        logger.warning("No LMCI column found. Candidate equity weight defaults to 1.")
        out["lmci_for_mclp"] = 0.0
        out["lmci_deficit_norm"] = 1.0
    else:
        out["lmci_for_mclp"] = pd.to_numeric(out[lmci_col], errors="coerce").fillna(0.0)
        out["lmci_deficit"] = out["lmci_for_mclp"].max() - out["lmci_for_mclp"]
        out["lmci_deficit_norm"] = minmax_norm(out["lmci_deficit"].astype(float))

    if "is_persistent_desert" in out.columns:
        out["is_persistent_desert"] = out["is_persistent_desert"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        out["is_persistent_desert"] = False

    if "desert_severity" not in out.columns:
        out["desert_severity"] = np.where(out["is_persistent_desert"], "Persistent", "None")

    if cfg.use_equity_weighting:
        out["candidate_equity_weight"] = 1.0 + out["lmci_deficit_norm"] * (cfg.underserved_bonus - 1.0)
        out.loc[out["is_persistent_desert"], "candidate_equity_weight"] *= cfg.persistent_desert_bonus
    else:
        out["candidate_equity_weight"] = 1.0

    return out


# ─────────────────────────────────────────────────────────────
# COVERAGE MATRIX
# ─────────────────────────────────────────────────────────────

def build_coverage_matrix(
    stations_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    cfg: MCLPConfig = CFG,
) -> Tuple[List[Set[int]], np.ndarray, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Build station-to-demand coverage sets.

    Returns
    -------
    coverage_sets:
        list where coverage_sets[i] contains demand_id values covered by station i

    distance_matrix_m:
        approximate network distances from station i to demand j

    station_gdf:
        projected station GeoDataFrame

    demand_gdf:
        projected demand GeoDataFrame
    """

    station_gdf = to_gdf(
        stations_df,
        lat_col="stop_lat",
        lon_col="stop_lon",
        crs_out=cfg.metric_crs,
    )

    demand_gdf = to_gdf(
        demand_df,
        lat_col="lat",
        lon_col="lon",
        crs_out=cfg.metric_crs,
    )

    station_xy = np.column_stack([station_gdf.geometry.x, station_gdf.geometry.y])
    demand_xy = np.column_stack([demand_gdf.geometry.x, demand_gdf.geometry.y])

    demand_tree = cKDTree(demand_xy)

    euclidean_radius = cfg.coverage_radius_m / max(cfg.detour_factor, 1e-9)

    coverage_sets: List[Set[int]] = []

    for xy in station_xy:
        covered_idx = demand_tree.query_ball_point(xy, r=euclidean_radius)
        covered_ids = set(demand_df.iloc[covered_idx]["demand_id"].astype(int).tolist())
        coverage_sets.append(covered_ids)

    # Full matrix is useful for assignment export. 57 x demand points is fine.
    diff = station_xy[:, None, :] - demand_xy[None, :, :]
    distance_matrix_m = np.sqrt((diff ** 2).sum(axis=2)) * cfg.detour_factor

    logger.info(
        f"Coverage matrix built: stations={len(stations_df)}, demand={len(demand_df)}, "
        f"radius={cfg.coverage_radius_m:.0f}m network-adjusted"
    )

    return coverage_sets, distance_matrix_m, station_gdf, demand_gdf


# ─────────────────────────────────────────────────────────────
# GREEDY MCLP
# ─────────────────────────────────────────────────────────────

def _weighted_coverage_value(
    demand_ids: Set[int],
    demand_weight_map: Dict[int, float],
) -> float:
    return float(sum(demand_weight_map.get(i, 0.0) for i in demand_ids))



def greedy_mclp(
    stations_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    coverage_sets: Sequence[Set[int]],
    k: int,
    cfg: MCLPConfig = CFG,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Greedy MCLP approximation.

    At every step, select the station with the largest marginal gain in
    uncovered weighted demand.

    Tie-breakers:
    1. higher equity-weighted marginal gain
    2. lower LMCI
    3. station name alphabetical
    """

    if k <= 0:
        raise ValueError("k must be positive.")

    if len(stations_df) != len(coverage_sets):
        raise ValueError("stations_df and coverage_sets length mismatch.")

    k = min(k, len(stations_df))

    demand_weight_map = dict(
        zip(demand_df["demand_id"].astype(int), demand_df["demand_weight"].astype(float))
    )

    total_weighted_demand = max(float(demand_df["demand_weight"].sum()), 1e-9)

    selected_indices: List[int] = []
    covered: Set[int] = set()
    rows: List[Dict[str, object]] = []

    station_work = stations_df.reset_index(drop=True).copy()

    for rank in range(1, k + 1):
        best_idx: Optional[int] = None
        best_tuple: Optional[Tuple[float, float, float, str]] = None
        best_new_covered: Set[int] = set()

        for idx, candidate_set in enumerate(coverage_sets):
            if idx in selected_indices:
                continue

            newly_covered = candidate_set - covered
            raw_gain = _weighted_coverage_value(newly_covered, demand_weight_map)

            equity_weight = float(station_work.loc[idx, "candidate_equity_weight"])
            equity_gain = raw_gain * equity_weight

            lmci_for_tiebreak = float(station_work.loc[idx, "lmci_for_mclp"])
            station_name = str(station_work.loc[idx, "stop_name"])

            # Python tuple comparison is ascending, so use positive equity gain,
            # positive raw gain, negative LMCI, reverse sorting logic manually.
            score_tuple = (
                equity_gain,
                raw_gain,
                -lmci_for_tiebreak,
                station_name,
            )

            if best_tuple is None or score_tuple > best_tuple:
                best_tuple = score_tuple
                best_idx = idx
                best_new_covered = newly_covered

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        covered |= best_new_covered

        cumulative_weighted = _weighted_coverage_value(covered, demand_weight_map)
        marginal_weighted = _weighted_coverage_value(best_new_covered, demand_weight_map)

        row = station_work.loc[best_idx].to_dict()
        row.update({
            "selection_rank": rank,
            "k": k,
            "marginal_demand_points": len(best_new_covered),
            "marginal_weighted_demand": marginal_weighted,
            "cumulative_demand_points": len(covered),
            "cumulative_weighted_demand": cumulative_weighted,
            "coverage_pct": 100.0 * cumulative_weighted / total_weighted_demand,
        })

        rows.append(row)

        logger.info(
            f"Rank {rank}: {row['stop_name']} | "
            f"marginal={marginal_weighted:.2f} | "
            f"coverage={row['coverage_pct']:.2f}%"
        )

    selected_df = pd.DataFrame(rows)

    summary = {
        "k": k,
        "selected_count": len(selected_df),
        "covered_demand_points": len(covered),
        "total_demand_points": len(demand_df),
        "covered_weighted_demand": _weighted_coverage_value(covered, demand_weight_map),
        "total_weighted_demand": total_weighted_demand,
        "coverage_pct": 100.0 * _weighted_coverage_value(covered, demand_weight_map) / total_weighted_demand,
        "covered_ids": covered,
        "selected_indices": selected_indices,
    }

    return selected_df, summary


# ─────────────────────────────────────────────────────────────
# COVERAGE CURVE
# ─────────────────────────────────────────────────────────────

def compute_coverage_by_k(
    stations_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    coverage_sets: Sequence[Set[int]],
    max_k: int,
    cfg: MCLPConfig = CFG,
) -> pd.DataFrame:
    """Run greedy MCLP for k = 1...max_k and return coverage curve."""

    rows: List[Dict[str, object]] = []
    max_k = min(max_k, len(stations_df))

    for k in range(1, max_k + 1):
        selected_df, summary = greedy_mclp(
            stations_df=stations_df,
            demand_df=demand_df,
            coverage_sets=coverage_sets,
            k=k,
            cfg=cfg,
        )

        selected_names = [] if selected_df.empty else selected_df["stop_name"].tolist()

        rows.append({
            "k": k,
            "selected_stations": " | ".join(selected_names),
            "covered_demand_points": summary["covered_demand_points"],
            "total_demand_points": summary["total_demand_points"],
            "covered_weighted_demand": summary["covered_weighted_demand"],
            "total_weighted_demand": summary["total_weighted_demand"],
            "coverage_pct": summary["coverage_pct"],
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# ASSIGNMENT EXPORT
# ─────────────────────────────────────────────────────────────

def assign_demand_to_selected_stations(
    stations_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    distance_matrix_m: np.ndarray,
    cfg: MCLPConfig = CFG,
) -> pd.DataFrame:
    """
    Assign covered demand points to nearest selected station.

    This is for dashboard storytelling:
    - which demand point is served by which selected station
    - what distance is assumed
    - what type of demand it represents
    """

    if selected_df.empty:
        return pd.DataFrame()

    station_lookup = stations_df.reset_index(drop=True).copy()
    selected_stop_ids = selected_df["stop_id"].tolist()
    selected_indices = station_lookup.index[station_lookup["stop_id"].isin(selected_stop_ids)].tolist()

    rows: List[Dict[str, object]] = []

    for demand_pos, demand_row in demand_df.reset_index(drop=True).iterrows():
        selected_distances = distance_matrix_m[selected_indices, demand_pos]
        nearest_pos = int(np.argmin(selected_distances))
        nearest_station_idx = selected_indices[nearest_pos]
        nearest_dist = float(selected_distances[nearest_pos])

        if nearest_dist <= cfg.coverage_radius_m:
            station_row = station_lookup.loc[nearest_station_idx]

            rows.append({
                "demand_id": int(demand_row["demand_id"]),
                "lat": float(demand_row["lat"]),
                "lon": float(demand_row["lon"]),
                "type": demand_row.get("type", "unknown"),
                "demand_weight": float(demand_row["demand_weight"]),
                "assigned_station_id": station_row["stop_id"],
                "assigned_station_name": station_row["stop_name"],
                "distance_to_station_m": nearest_dist,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# CANDIDATE SCORE EXPORT
# ─────────────────────────────────────────────────────────────

def build_candidate_scores(
    stations_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    coverage_sets: Sequence[Set[int]],
    cfg: MCLPConfig = CFG,
) -> pd.DataFrame:
    """Create ranked candidate table before greedy selection."""

    demand_weight_map = dict(
        zip(demand_df["demand_id"].astype(int), demand_df["demand_weight"].astype(float))
    )

    rows = []

    for idx, station in stations_df.reset_index(drop=True).iterrows():
        covered_ids = coverage_sets[idx]
        raw_weight = _weighted_coverage_value(covered_ids, demand_weight_map)
        equity_weight = float(station.get("candidate_equity_weight", 1.0))

        row = station.to_dict()
        row.update({
            "covered_demand_points_if_selected": len(covered_ids),
            "weighted_demand_if_selected": raw_weight,
            "equity_weighted_candidate_score": raw_weight * equity_weight,
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["equity_weighted_candidate_score", "weighted_demand_if_selected"],
        ascending=False,
    ).reset_index(drop=True)

    out["candidate_rank"] = np.arange(1, len(out) + 1)

    return out


# ─────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────

def export_mclp_outputs(
    cfg: MCLPConfig,
    selected_df: pd.DataFrame,
    coverage_by_k_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    candidate_scores_df: pd.DataFrame,
) -> Dict[str, Path]:
    """Save all MCLP outputs into outputs/ and assets/."""

    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.assets_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    selected_path = cfg.outputs_dir / "mclp_selected_stations.csv"
    selected_df.to_csv(selected_path, index=False)
    paths["selected_stations"] = selected_path

    coverage_path = cfg.outputs_dir / "mclp_coverage_by_k.csv"
    coverage_by_k_df.to_csv(coverage_path, index=False)
    paths["coverage_by_k"] = coverage_path

    assignment_path = cfg.outputs_dir / "mclp_demand_assignment.csv"
    assignment_df.to_csv(assignment_path, index=False)
    paths["demand_assignment"] = assignment_path

    candidate_path = cfg.outputs_dir / "mclp_candidate_scores.csv"
    candidate_scores_df.to_csv(candidate_path, index=False)
    paths["candidate_scores"] = candidate_path

    # GeoJSON exports for dashboard/map usage
    try:
        if not selected_df.empty:
            selected_gdf = to_gdf(
                selected_df,
                lat_col="stop_lat",
                lon_col="stop_lon",
                crs_out=cfg.wgs84_crs,
            )
            selected_geojson = cfg.assets_dir / "mclp_selected_stations.geojson"
            selected_gdf.to_file(selected_geojson, driver="GeoJSON")
            paths["selected_stations_geojson"] = selected_geojson
    except Exception as exc:
        logger.warning(f"Selected stations GeoJSON export skipped: {exc}")

    try:
        if not assignment_df.empty:
            assignment_gdf = to_gdf(
                assignment_df,
                lat_col="lat",
                lon_col="lon",
                crs_out=cfg.wgs84_crs,
            )
            assignment_geojson = cfg.assets_dir / "mclp_demand_assignment.geojson"
            assignment_gdf.to_file(assignment_geojson, driver="GeoJSON")
            paths["demand_assignment_geojson"] = assignment_geojson
    except Exception as exc:
        logger.warning(f"Demand assignment GeoJSON export skipped: {exc}")

    logger.info("MCLP exports complete:")
    for key, path in paths.items():
        logger.info(f"  {key}: {path}")

    return paths


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_mclp_outputs(
    selected_df: pd.DataFrame,
    coverage_by_k_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    candidate_scores_df: pd.DataFrame,
) -> None:
    """Basic fail-fast validation for dashboard reliability."""

    if selected_df.empty:
        raise ValueError("MCLP selected_df is empty. No stations selected.")

    required_selected = {
        "stop_id",
        "stop_name",
        "selection_rank",
        "coverage_pct",
        "marginal_weighted_demand",
    }
    missing_selected = required_selected - set(selected_df.columns)
    if missing_selected:
        raise ValueError(f"Selected station output missing columns: {missing_selected}")

    if coverage_by_k_df.empty:
        raise ValueError("coverage_by_k_df is empty.")

    if "coverage_pct" not in coverage_by_k_df.columns:
        raise ValueError("coverage_by_k_df missing coverage_pct.")

    if candidate_scores_df.empty:
        raise ValueError("candidate_scores_df is empty.")

    # Assignment can be empty only if selected stations cover zero demand.
    if assignment_df.empty:
        logger.warning("Demand assignment is empty. Check coverage radius and demand coordinates.")

    logger.info("MCLP validation passed.")


# ─────────────────────────────────────────────────────────────
# END-TO-END PIPELINE
# ─────────────────────────────────────────────────────────────

def run_mclp_pipeline(
    cfg: MCLPConfig = CFG,
    k: Optional[int] = None,
) -> Dict[str, object]:
    """
    Run complete MCLP pipeline.

    Parameters
    ----------
    cfg:
        MCLPConfig

    k:
        Number of intervention stations to select.
        Defaults to cfg.default_k.

    Returns
    -------
    dict containing stations, demand, selected, coverage curve,
    assignment table, candidate scores, and export paths.
    """

    logger.info("=" * 70)
    logger.info("STARTING MCLP PIPELINE")
    logger.info("=" * 70)

    if k is None:
        k = cfg.default_k

    stations = load_station_scores(cfg)
    stations = enrich_station_candidates(stations, cfg)

    demand = load_demand_points(cfg)
    demand = assign_demand_weights(demand, cfg)

    coverage_sets, distance_matrix_m, station_gdf, demand_gdf = build_coverage_matrix(
        stations_df=stations,
        demand_df=demand,
        cfg=cfg,
    )

    selected_df, summary = greedy_mclp(
        stations_df=stations,
        demand_df=demand,
        coverage_sets=coverage_sets,
        k=k,
        cfg=cfg,
    )

    coverage_by_k_df = compute_coverage_by_k(
        stations_df=stations,
        demand_df=demand,
        coverage_sets=coverage_sets,
        max_k=cfg.max_k,
        cfg=cfg,
    )

    assignment_df = assign_demand_to_selected_stations(
        stations_df=stations,
        demand_df=demand,
        selected_df=selected_df,
        distance_matrix_m=distance_matrix_m,
        cfg=cfg,
    )

    candidate_scores_df = build_candidate_scores(
        stations_df=stations,
        demand_df=demand,
        coverage_sets=coverage_sets,
        cfg=cfg,
    )

    validate_mclp_outputs(
        selected_df=selected_df,
        coverage_by_k_df=coverage_by_k_df,
        assignment_df=assignment_df,
        candidate_scores_df=candidate_scores_df,
    )

    export_paths = export_mclp_outputs(
        cfg=cfg,
        selected_df=selected_df,
        coverage_by_k_df=coverage_by_k_df,
        assignment_df=assignment_df,
        candidate_scores_df=candidate_scores_df,
    )

    logger.info("=" * 70)
    logger.info("MCLP PIPELINE COMPLETE")
    logger.info(f"Selected stations: {len(selected_df)}")
    logger.info(f"Weighted coverage: {summary['coverage_pct']:.2f}%")
    logger.info("=" * 70)

    return {
        "stations": stations,
        "demand": demand,
        "coverage_sets": coverage_sets,
        "distance_matrix_m": distance_matrix_m,
        "station_gdf": station_gdf,
        "demand_gdf": demand_gdf,
        "selected_df": selected_df,
        "coverage_by_k_df": coverage_by_k_df,
        "assignment_df": assignment_df,
        "candidate_scores_df": candidate_scores_df,
        "summary": summary,
        "export_paths": export_paths,
    }


if __name__ == "__main__":
    run_mclp_pipeline()
