from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------
# Config
# ---------------------------
PARQUET_DIR_DEFAULT = Path("parquet_out")

RESULT_COLS_NICE = [
    "race_id",
    "driver_id",
    "constructor_id",
    "grid",
    "position",
    "position_order",
    "points",
    "laps",
    "status",
    "status_id",
    # enriched columns (if you have fact_results_enriched)
    "season",
    "round",
    "race_name",
    "date",
    "race_datetime_utc",
    "circuit_id",
    "full_name",
    "givenname",
    "familyname",
    "constructor_name",
    "circuit_name",
    "country",
]

ERA_BINS = [
    (1950, 1967, "Early F1"),
    (1968, 1982, "DFV Era"),
    (1983, 1988, "Turbo Era"),
    (1989, 2005, "V10/V8 Era"),
    (2006, 2013, "Pre-Hybrid"),
    (2014, 2021, "Hybrid Era"),
    (2022, 2100, "Ground Effect Hybrid"),
]


@dataclass
class Weights:
    career: float = 0.30
    peak: float = 0.25
    context: float = 0.20
    longevity: float = 0.15
    quali: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return {
            "career": self.career,
            "peak": self.peak,
            "context": self.context,
            "longevity": self.longevity,
            "quali": self.quali,
        }

    def normalize(self) -> "Weights":
        d = self.as_dict()
        s = sum(d.values())
        if s <= 0:
            return self
        return Weights(**{k: v / s for k, v in d.items()})


# ---------------------------
# Loading
# ---------------------------
def load_parquets(parquet_dir: Path | str = PARQUET_DIR_DEFAULT) -> dict[str, pd.DataFrame]:
    p = Path(parquet_dir)

    paths = {
        "results_enriched": p / "fact_results_enriched.parquet",
        "results": p / "fact_results.parquet",
        "qualifying": p / "fact_qualifying.parquet",
        "driver_standings": p / "fact_driver_standings.parquet",
        "constructor_standings": p / "fact_constructor_standings.parquet",
        "dim_drivers": p / "dim_drivers.parquet",
        "dim_constructors": p / "dim_constructors.parquet",
        "dim_races": p / "dim_races.parquet",
        "dim_circuits": p / "dim_circuits.parquet",
    }

    out: dict[str, pd.DataFrame] = {}
    for k, fp in paths.items():
        if fp.exists():
            out[k] = pd.read_parquet(fp)

    if "results_enriched" not in out and "results" not in out:
        raise FileNotFoundError(
            "Need at least fact_results_enriched.parquet OR fact_results.parquet in parquet_out/"
        )

    return out


# ---------------------------
# Utilities
# ---------------------------
def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _safe_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _add_era(df: pd.DataFrame, year_col: str = "season") -> pd.DataFrame:
    df = df.copy()
    if year_col not in df.columns:
        df["era"] = pd.NA
        return df

    def to_era(y: float) -> str | None:
        if pd.isna(y):
            return None
        y = int(y)
        for a, b, name in ERA_BINS:
            if a <= y <= b:
                return name
        return None

    df["era"] = df[year_col].apply(to_era).astype("string")
    return df


def percentile_0_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(pct=True, na_option="bottom") * 100.0


# ---------------------------
# Feature building
# ---------------------------
def build_driver_features(parquet_dir: Path | str = PARQUET_DIR_DEFAULT) -> pd.DataFrame:
    """
    Produces one row per driver with career, peak, longevity, qualifying, context, and CHAMPIONSHIPS.
    """
    t = load_parquets(parquet_dir)

    # --- RESULTS ---
    if "results_enriched" in t:
        res = t["results_enriched"].copy()
    else:
        res = t["results"].copy()
        if "dim_races" in t and "race_id" in res.columns:
            res = res.merge(
                t["dim_races"][["race_id", "season", "round", "circuit_id", "date"]],
                on="race_id",
                how="left",
            )
        if "dim_drivers" in t and "driver_id" in res.columns:
            keep = [
                c
                for c in ["driver_id", "full_name", "givenname", "familyname", "nationality", "dob"]
                if c in t["dim_drivers"].columns
            ]
            res = res.merge(t["dim_drivers"][keep], on="driver_id", how="left")
        if "dim_constructors" in t and "constructor_id" in res.columns:
            keep = [c for c in ["constructor_id", "name"] if c in t["dim_constructors"].columns]
            tmp = t["dim_constructors"][keep].copy()
            if "name" in tmp.columns:
                tmp = tmp.rename(columns={"name": "constructor_name"})
            res = res.merge(tmp, on="constructor_id", how="left")

    res = res[[c for c in RESULT_COLS_NICE if c in res.columns]].copy()

    res = _ensure_numeric(res, ["grid", "position", "position_order", "points", "laps", "season", "round"])
    pos_col = _safe_col(res, ["position_order", "position"])
    if pos_col is None:
        raise ValueError("Results table missing position/position_order columns; can't compute wins/podiums.")
    res["finish_pos"] = pd.to_numeric(res[pos_col], errors="coerce")

    res["start"] = 1
    res["win"] = (res["finish_pos"] == 1).astype(int)
    res["podium"] = (res["finish_pos"].between(1, 3, inclusive="both")).astype(int)
    res["top5"] = (res["finish_pos"].between(1, 5, inclusive="both")).astype(int)
    res["top10"] = (res["finish_pos"].between(1, 10, inclusive="both")).astype(int)

    if "season" in res.columns:
        res = _add_era(res, "season")

    # --- QUALIFYING ---
    quali = None
    if "qualifying" in t:
        quali = t["qualifying"].copy()
        quali = _ensure_numeric(quali, ["position"])
        if "dim_races" in t and "race_id" in quali.columns and "season" not in quali.columns:
            quali = quali.merge(t["dim_races"][["race_id", "season"]], on="race_id", how="left")
        quali = _add_era(quali, "season")

    # --- Constructor standings -> car strength proxy (season-level) ---
    car_strength = None
    if "constructor_standings" in t and "season" in t["constructor_standings"].columns:
        cs = t["constructor_standings"].copy()
        cs = _ensure_numeric(cs, ["season", "round", "points"])
        if "round" in cs.columns:
            last_round = cs.groupby("season")["round"].max().rename("last_round")
            cs = cs.merge(last_round, on="season", how="left")
            cs_last = cs[cs["round"] == cs["last_round"]].copy()
        else:
            cs_last = cs.copy()

        season_total = cs_last.groupby("season")["points"].sum(min_count=1).rename("season_constructor_points_total")
        cs_last = cs_last.merge(season_total, on="season", how="left")

        if "constructor_id" in cs_last.columns:
            cs_last["constructor_points_share_season"] = cs_last["points"] / cs_last["season_constructor_points_total"]
            car_strength = cs_last[["season", "constructor_id", "constructor_points_share_season"]].copy()

    # --- CHAMPIONSHIPS ---
    championships = None
    if "driver_standings" in t:
        ds = t["driver_standings"].copy()
        ds = _ensure_numeric(ds, ["season", "round", "position"])

        needed = {"season", "round", "position", "driver_id"}
        if needed.issubset(ds.columns):
            last_round = ds.groupby("season")["round"].max().rename("last_round")
            ds = ds.merge(last_round, on="season", how="left")
            final_standings = ds[ds["round"] == ds["last_round"]].copy()

            champs = final_standings[final_standings["position"] == 1].copy()
            championships = champs.groupby("driver_id").size().rename("championships").reset_index()

    # ---------------------------
    # CAREER AGGREGATES
    # ---------------------------
    g = res.groupby("driver_id", dropna=False)

    career = pd.DataFrame(
        {
            "starts": g["start"].sum(),
            "wins": g["win"].sum(),
            "podiums": g["podium"].sum(),
            "top5": g["top5"].sum(),
            "top10": g["top10"].sum(),
            "points_total": g["points"].sum(min_count=1) if "points" in res.columns else np.nan,
            "avg_finish_pos": g["finish_pos"].mean(),
        }
    ).reset_index()

    career["win_rate"] = career["wins"] / career["starts"].replace({0: np.nan})
    career["podium_rate"] = career["podiums"] / career["starts"].replace({0: np.nan})
    career["points_per_start"] = career["points_total"] / career["starts"].replace({0: np.nan})

    if "season" in res.columns:
        seasons = res.dropna(subset=["season"]).groupby("driver_id")["season"].nunique().rename("seasons")
        career = career.merge(seasons.reset_index(), on="driver_id", how="left")

        first_last = (
            res.dropna(subset=["season"]).groupby("driver_id")["season"].agg(["min", "max"]).reset_index()
        )
        first_last = first_last.rename(columns={"min": "first_season", "max": "last_season"})
        career = career.merge(first_last, on="driver_id", how="left")
        career["career_span_years"] = career["last_season"] - career["first_season"] + 1
    else:
        career["seasons"] = np.nan
        career["first_season"] = np.nan
        career["last_season"] = np.nan
        career["career_span_years"] = np.nan

    # ---------------------------
    # PEAK (best seasons) - FIXED (no KeyError)
    # ---------------------------
    if "season" in res.columns:
        season_driver = (
            res.groupby(["driver_id", "season"], dropna=False)
            .agg(
                starts=("start", "sum"),
                wins=("win", "sum"),
                podiums=("podium", "sum"),
                points=("points", "sum") if "points" in res.columns else ("start", "sum"),
                avg_finish=("finish_pos", "mean"),
            )
            .reset_index()
        )

        season_driver["win_rate_season"] = season_driver["wins"] / season_driver["starts"].replace({0: np.nan})
        season_driver["podium_rate_season"] = season_driver["podiums"] / season_driver["starts"].replace({0: np.nan})
        season_driver["points_per_start_season"] = season_driver["points"] / season_driver["starts"].replace({0: np.nan})

        # Sort seasons by points_per_start_season (desc) per driver
        season_driver = season_driver.sort_values(["driver_id", "points_per_start_season"], ascending=[True, False])

        # SAFE Peak3/Peak1 that always keeps driver_id as a column
        top3 = season_driver.groupby("driver_id", as_index=False).head(3).copy()
        peak3 = (
            top3.groupby("driver_id", as_index=False)["points_per_start_season"]
            .mean()
            .rename(columns={"points_per_start_season": "peak3_points_per_start"})
        )

        top1 = season_driver.groupby("driver_id", as_index=False).head(1).copy()
        peak1 = (
            top1.groupby("driver_id", as_index=False)["points_per_start_season"]
            .mean()
            .rename(columns={"points_per_start_season": "peak1_points_per_start"})
        )

        best_win = (
            season_driver.groupby("driver_id", as_index=False)["win_rate_season"]
            .max()
            .rename(columns={"win_rate_season": "best_season_win_rate"})
        )
        best_podium = (
            season_driver.groupby("driver_id", as_index=False)["podium_rate_season"]
            .max()
            .rename(columns={"podium_rate_season": "best_season_podium_rate"})
        )
        best_finish = (
            season_driver.groupby("driver_id", as_index=False)["avg_finish"]
            .min()
            .rename(columns={"avg_finish": "best_season_avg_finish"})
        )

        peak = (
            peak3.merge(peak1, on="driver_id", how="outer")
            .merge(best_win, on="driver_id", how="outer")
            .merge(best_podium, on="driver_id", how="outer")
            .merge(best_finish, on="driver_id", how="outer")
        )
    else:
        peak = pd.DataFrame({"driver_id": career["driver_id"]})

    # ---------------------------
    # QUALIFYING FEATURES
    # ---------------------------
    if quali is not None and {"position", "driver_id"}.issubset(quali.columns):
        qg = quali.groupby("driver_id", dropna=False)
        quali_feat = pd.DataFrame(
            {
                "quali_starts": qg["position"].count(),
                "avg_quali_pos": qg["position"].mean(),
                "pole_count": (quali["position"] == 1).groupby(quali["driver_id"]).sum(),
            }
        ).reset_index()
        quali_feat["pole_rate"] = quali_feat["pole_count"] / quali_feat["quali_starts"].replace({0: np.nan})
    else:
        quali_feat = pd.DataFrame({"driver_id": career["driver_id"]})
        quali_feat["quali_starts"] = np.nan
        quali_feat["avg_quali_pos"] = np.nan
        quali_feat["pole_count"] = np.nan
        quali_feat["pole_rate"] = np.nan

    # ---------------------------
    # CONTEXT FEATURES (car dominance proxy)
    # ---------------------------
    if car_strength is not None and {"season", "constructor_id"}.issubset(res.columns):
        res_cs = res.merge(car_strength, on=["season", "constructor_id"], how="left")
        dcs = (
            res_cs.groupby(["driver_id", "season"])
            .agg(
                starts=("start", "sum"),
                points=("points", "sum") if "points" in res_cs.columns else ("start", "sum"),
                car_strength_avg=("constructor_points_share_season", "mean"),
            )
            .reset_index()
        )

        dcs["points_per_start"] = dcs["points"] / dcs["starts"].replace({0: np.nan})
        dcs["overachieve"] = dcs["points_per_start"] / dcs["car_strength_avg"].replace({0: np.nan})

        context = (
            dcs.groupby("driver_id", as_index=False)
            .agg(
                avg_car_strength=("car_strength_avg", "mean"),
                overachieve_peak=("overachieve", "max"),
                overachieve_avg=("overachieve", "mean"),
            )
            .reset_index(drop=True)
        )
    else:
        context = pd.DataFrame({"driver_id": career["driver_id"]})
        context["avg_car_strength"] = np.nan
        context["overachieve_peak"] = np.nan
        context["overachieve_avg"] = np.nan

    # ---------------------------
    # LONGEVITY / CONSISTENCY
    # ---------------------------
    cons = (
        res.groupby("driver_id", as_index=False)
        .agg(
            finish_pos_std=("finish_pos", "std"),
            finish_pos_median=("finish_pos", "median"),
        )
        .copy()
    )

    if "season" in res.columns:
        sd = (
            res.groupby(["driver_id", "season"], as_index=False)
            .agg(
                wins=("win", "sum"),
                podiums=("podium", "sum"),
                starts=("start", "sum"),
            )
            .copy()
        )

        lon = (
            sd.groupby("driver_id", as_index=False)
            .agg(
                seasons_with_win=("wins", lambda x: int((x > 0).sum())),
                seasons_with_podium=("podiums", lambda x: int((x > 0).sum())),
            )
            .copy()
        )
    else:
        lon = pd.DataFrame({"driver_id": career["driver_id"]})
        lon["seasons_with_win"] = np.nan
        lon["seasons_with_podium"] = np.nan

    # ---------------------------
    # Names
    # ---------------------------
    if "dim_drivers" in t and "driver_id" in t["dim_drivers"].columns:
        name_cols = [
            c
            for c in ["driver_id", "full_name", "givenname", "familyname", "nationality", "dob"]
            if c in t["dim_drivers"].columns
        ]
        names = t["dim_drivers"][name_cols].drop_duplicates(subset=["driver_id"])
    else:
        cand = [c for c in ["driver_id", "full_name", "givenname", "familyname"] if c in res.columns]
        names = (
            res[cand].drop_duplicates(subset=["driver_id"])
            if cand
            else pd.DataFrame({"driver_id": career["driver_id"]})
        )

    # ---------------------------
    # Combine all features
    # ---------------------------
    df = names.merge(career, on="driver_id", how="left")
    df = df.merge(peak, on="driver_id", how="left")
    df = df.merge(quali_feat, on="driver_id", how="left")
    df = df.merge(context, on="driver_id", how="left")
    df = df.merge(cons, on="driver_id", how="left")
    df = df.merge(lon, on="driver_id", how="left")

    if championships is not None:
        df = df.merge(championships, on="driver_id", how="left")
    if "championships" not in df.columns:
        df["championships"] = 0
    df["championships"] = pd.to_numeric(df["championships"], errors="coerce").fillna(0).astype(int)

    if "full_name" not in df.columns and {"givenname", "familyname"}.issubset(df.columns):
        df["full_name"] = (df["givenname"].astype(str) + " " + df["familyname"].astype(str)).str.strip()

    return df


# ---------------------------
# Scoring
# ---------------------------
def compute_subscores(features: pd.DataFrame, era_normalize: bool = True) -> pd.DataFrame:
    df = features.copy()

    career_components = []
    if "championships" in df.columns:
        career_components.append(percentile_0_100(df["championships"]))
    if "wins" in df.columns:
        career_components.append(percentile_0_100(df["wins"]))
    if "win_rate" in df.columns:
        career_components.append(percentile_0_100(df["win_rate"]))
    if "podium_rate" in df.columns:
        career_components.append(percentile_0_100(df["podium_rate"]))
    if "points_per_start" in df.columns:
        career_components.append(percentile_0_100(df["points_per_start"]))
    if "avg_finish_pos" in df.columns:
        career_components.append(100 - percentile_0_100(df["avg_finish_pos"]))

    df["career_score"] = np.nan if not career_components else np.nanmean(
        np.vstack([c.to_numpy() for c in career_components]), axis=0
    )

    peak_components = []
    for col in ["peak3_points_per_start", "peak1_points_per_start", "best_season_win_rate", "best_season_podium_rate"]:
        if col in df.columns:
            peak_components.append(percentile_0_100(df[col]))
    if "best_season_avg_finish" in df.columns:
        peak_components.append(100 - percentile_0_100(df["best_season_avg_finish"]))
    df["peak_score"] = np.nan if not peak_components else np.nanmean(
        np.vstack([c.to_numpy() for c in peak_components]), axis=0
    )

    context_components = []
    if "avg_car_strength" in df.columns and df["avg_car_strength"].notna().any():
        context_components.append(100 - percentile_0_100(df["avg_car_strength"]))
    if "overachieve_peak" in df.columns and df["overachieve_peak"].notna().any():
        context_components.append(percentile_0_100(df["overachieve_peak"]))
    if "overachieve_avg" in df.columns and df["overachieve_avg"].notna().any():
        context_components.append(percentile_0_100(df["overachieve_avg"]))
    df["context_score"] = np.nan if not context_components else np.nanmean(
        np.vstack([c.to_numpy() for c in context_components]), axis=0
    )

    lon_components = []
    if "seasons" in df.columns:
        lon_components.append(percentile_0_100(df["seasons"]))
    if "career_span_years" in df.columns:
        lon_components.append(percentile_0_100(df["career_span_years"]))
    if "seasons_with_win" in df.columns:
        lon_components.append(percentile_0_100(df["seasons_with_win"]))
    if "finish_pos_std" in df.columns and df["finish_pos_std"].notna().any():
        lon_components.append(100 - percentile_0_100(df["finish_pos_std"]))
    df["longevity_score"] = np.nan if not lon_components else np.nanmean(
        np.vstack([c.to_numpy() for c in lon_components]), axis=0
    )

    q_components = []
    if "pole_count" in df.columns and df["pole_count"].notna().any():
        q_components.append(percentile_0_100(df["pole_count"]))
    if "pole_rate" in df.columns and df["pole_rate"].notna().any():
        q_components.append(percentile_0_100(df["pole_rate"]))
    if "avg_quali_pos" in df.columns and df["avg_quali_pos"].notna().any():
        q_components.append(100 - percentile_0_100(df["avg_quali_pos"]))
    df["quali_score"] = np.nan if not q_components else np.nanmean(
        np.vstack([c.to_numpy() for c in q_components]), axis=0
    )

    # Era multiplier (optional)
    if era_normalize and "first_season" in df.columns:
        def mult(y):
            if pd.isna(y):
                return 1.0
            y = int(y)
            if y <= 1967: return 0.98
            if y <= 1982: return 1.00
            if y <= 1988: return 1.01
            if y <= 2005: return 1.02
            if y <= 2013: return 1.03
            if y <= 2021: return 1.05
            return 1.06

        df["era_multiplier"] = df["first_season"].apply(mult)
    else:
        df["era_multiplier"] = 1.0

    return df


def compute_goat_ranking(
    features: pd.DataFrame,
    weights: Weights | dict[str, float] | None = None,
    era_normalize: bool = True,
    min_starts: int = 20,
) -> pd.DataFrame:
    if weights is None:
        w = Weights()
    elif isinstance(weights, dict):
        w = Weights(**{k: float(v) for k, v in weights.items()})
    else:
        w = weights
    w = w.normalize()

    df = compute_subscores(features, era_normalize=era_normalize)

    if "starts" in df.columns:
        df = df[df["starts"].fillna(0) >= min_starts].copy()

    df["goat_score_raw"] = (
        w.career * df["career_score"]
        + w.peak * df["peak_score"]
        + w.context * df["context_score"]
        + w.longevity * df["longevity_score"]
        + w.quali * df["quali_score"]
    )

    df["goat_score"] = df["goat_score_raw"] * df["era_multiplier"].fillna(1.0)

    df = df.sort_values("goat_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# ---------------------------
# Build & save features
# ---------------------------
def build_and_save_features(
    parquet_dir: Path | str = PARQUET_DIR_DEFAULT,
    out_path: Path | str | None = None
) -> Path:
    parquet_dir = Path(parquet_dir)
    if out_path is None:
        out_path = parquet_dir / "driver_career_features.parquet"
    out_path = Path(out_path)

    feats = build_driver_features(parquet_dir)
    feats.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    out = build_and_save_features(PARQUET_DIR_DEFAULT)
    print(f"Wrote: {out}")
