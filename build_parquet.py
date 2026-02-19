# build_parquet.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

# -----------------------------
# Paths (robust)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "archive"
OUT_DIR = BASE_DIR / "parquet_out"

CSV_FILES = {
    "circuits": "circuits.csv",
    "constructors": "constructors.csv",
    "constructor_standings": "constructor_standings.csv",
    "drivers": "drivers.csv",
    "driver_standings": "driver_standings.csv",
    "qualifying": "qualifying.csv",
    "races": "races.csv",
    "results": "results.csv",
    "f1_2025_last_race_results": "f1_2025_last_race_results.csv",
}

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_csv_safely(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise last_err

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_").replace("-", "_").lower() for c in df.columns]
    return df

def normalize_key(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Force join keys to STRING consistently (avoids float vs object issues).
    Also normalizes common missing markers.
    """
    df = df.copy()
    if col in df.columns:
        s = df[col].astype("string").str.strip()
        s = s.replace({"\\N": pd.NA, "nan": pd.NA, "None": pd.NA, "": pd.NA})
        # Also handle "1.0" -> "1" if it looks numeric
        s = s.str.replace(r"\.0$", "", regex=True)
        df[col] = s
    return df

def drop_bad_keys_and_dedupe(df: pd.DataFrame, key: str, keep_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Prevent merge explosions:
    - remove rows where key is missing
    - keep only one row per key in dimension-like tables
    """
    df = df.copy()
    if keep_cols is not None:
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]

    if key in df.columns:
        df = df[df[key].notna()]
        df = df.drop_duplicates(subset=[key], keep="first")
    return df

def to_parquet(df: pd.DataFrame, out_path: Path) -> None:
    df.to_parquet(out_path, index=False, engine="pyarrow")

# -----------------------------
# Load
# -----------------------------
def load_all() -> dict[str, pd.DataFrame]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Can't find archive folder at: {DATA_DIR}")

    tables: dict[str, pd.DataFrame] = {}
    for key, fname in CSV_FILES.items():
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        df = read_csv_safely(fpath)
        df = clean_columns(df)
        tables[key] = df
    return tables

# -----------------------------
# Transform
# -----------------------------
def transform(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    t = {k: v.copy() for k, v in tables.items()}

    # Normalize join keys as STRING everywhere
    key_cols = [
        "race_id", "driver_id", "constructor_id", "circuit_id",
        "result_id", "qualify_id",
        "driver_standings_id", "constructor_standings_id",
    ]
    for name, df in t.items():
        for col in key_cols:
            t[name] = normalize_key(t[name], col)

    # Races: parse date and build datetime safely (no warning spam)
    if "races" in t:
        races = t["races"].copy()
        if "date" in races.columns:
            races["date"] = pd.to_datetime(races["date"], errors="coerce", format="%Y-%m-%d")

        if "time" in races.columns and "date" in races.columns:
            time_clean = races["time"].astype("string").str.strip()
            time_clean = time_clean.replace({"\\N": pd.NA, "nan": pd.NA, "None": pd.NA, "": pd.NA})
            time_clean = time_clean.str.replace("Z", "", regex=False).fillna("00:00:00")

            races["race_datetime_utc"] = pd.to_datetime(
                races["date"].dt.strftime("%Y-%m-%d") + " " + time_clean,
                errors="coerce",
                utc=True,
            )

        t["races"] = races

    # Drivers: full_name + dob
    if "drivers" in t:
        drivers = t["drivers"].copy()
        if "dob" in drivers.columns:
            drivers["dob"] = pd.to_datetime(drivers["dob"], errors="coerce", format="%Y-%m-%d")
        if "givenname" in drivers.columns and "familyname" in drivers.columns:
            drivers["full_name"] = (
                drivers["givenname"].astype("string").str.strip()
                + " "
                + drivers["familyname"].astype("string").str.strip()
            ).str.strip()
        t["drivers"] = drivers

    return t

# -----------------------------
# Build outputs
# -----------------------------
def build_outputs(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    # Save normalized raw tables too (helpful debugging)
    for k, df in tables.items():
        out[f"raw_{k}"] = df

    # Dimension tables (dedup + drop missing keys)
    if "drivers" in tables:
        out["dim_drivers"] = drop_bad_keys_and_dedupe(
            tables["drivers"],
            key="driver_id",
            keep_cols=["driver_id", "full_name", "givenname", "familyname", "nationality", "dob"]
        )
    if "constructors" in tables:
        out["dim_constructors"] = drop_bad_keys_and_dedupe(
            tables["constructors"],
            key="constructor_id",
            keep_cols=["constructor_id", "name", "nationality"]
        )
    if "circuits" in tables:
        # circuits sometimes uses lng instead of long
        circuits = tables["circuits"].copy()
        if "lng" in circuits.columns and "long" not in circuits.columns:
            circuits = circuits.rename(columns={"lng": "long"})
        out["dim_circuits"] = drop_bad_keys_and_dedupe(
            circuits,
            key="circuit_id",
            keep_cols=["circuit_id", "name", "country", "locality", "lat", "long"]
        )
    if "races" in tables:
        # races sometimes uses "name" instead of "race_name"
        races = tables["races"].copy()
        if "name" in races.columns and "race_name" not in races.columns:
            races = races.rename(columns={"name": "race_name"})
        out["dim_races"] = drop_bad_keys_and_dedupe(
            races,
            key="race_id",
            keep_cols=["race_id", "season", "round", "race_name", "date", "race_datetime_utc", "circuit_id"]
        )

    # Fact tables (kept as-is, but keys are normalized to string already)
    for k in ("results", "qualifying", "driver_standings", "constructor_standings"):
        if k in tables:
            out[f"fact_{k}"] = tables[k]

    # Enriched results (safe merge: dims are deduped + no missing keys)
    need = {"results", "races", "drivers", "constructors", "circuits"}
    if need.issubset(tables.keys()):
        results = tables["results"].copy()

        dim_races = out["dim_races"]
        dim_drivers = out["dim_drivers"]
        dim_constructors = out["dim_constructors"]
        dim_circuits = out["dim_circuits"]

        # Drop rows with missing join keys on the LEFT too (avoids weird behavior)
        for key in ("race_id", "driver_id", "constructor_id"):
            if key in results.columns:
                results = results[results[key].notna()]

        enriched = (
            results
            .merge(dim_races, on="race_id", how="left")
            .merge(dim_drivers, on="driver_id", how="left")
            .merge(dim_constructors, on="constructor_id", how="left", suffixes=("", "_constructor"))
            .merge(dim_circuits, on="circuit_id", how="left", suffixes=("", "_circuit"))
        )
        out["fact_results_enriched"] = enriched

    # Optional extra dataset
    if "f1_2025_last_race_results" in tables:
        out["raw_f1_2025_last_race_results"] = tables["f1_2025_last_race_results"]

    return out

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ensure_dir(OUT_DIR)

    tables = load_all()
    tables = transform(tables)
    outputs = build_outputs(tables)

    for name, df in outputs.items():
        to_parquet(df, OUT_DIR / f"{name}.parquet")

    manifest = [{"table": n, "rows": int(df.shape[0]), "cols": int(df.shape[1])} for n, df in outputs.items()]
    pd.DataFrame(manifest).sort_values("table").to_csv(OUT_DIR / "manifest.csv", index=False)

    print(f"Done. Wrote {len(outputs)} parquet files to: {OUT_DIR.resolve()}")
    print("Also wrote manifest.csv")

if __name__ == "__main__":
    main()
