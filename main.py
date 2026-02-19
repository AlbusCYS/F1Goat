from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from backend_goat import compute_goat_ranking

PARQUET_DIR = Path("parquet_out")
FEATURES_PATH = PARQUET_DIR / "driver_career_features.parquet"

app = FastAPI(title="F1 GOAT API", version="1.0")

# Allow the Next.js frontend to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load features once on startup (fast)
FEATURES_DF: pd.DataFrame | None = None


class WeightPayload(BaseModel):
    career: float = Field(0.30, ge=0.0)
    peak: float = Field(0.25, ge=0.0)
    context: float = Field(0.20, ge=0.0)
    longevity: float = Field(0.15, ge=0.0)
    quali: float = Field(0.10, ge=0.0)


class RankRequest(BaseModel):
    weights: WeightPayload = WeightPayload()
    era_normalize: bool = True
    min_starts: int = 30
    top_n: int = 50


@app.on_event("startup")
def _startup() -> None:
    global FEATURES_DF
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PATH}. Run:\n"
            "  python backend_goat.py\n"
            "to generate driver_career_features.parquet"
        )
    FEATURES_DF = pd.read_parquet(FEATURES_PATH)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/rank")
def rank(req: RankRequest):
    assert FEATURES_DF is not None

    weights_dict = req.weights.model_dump()
    ranked = compute_goat_ranking(
        FEATURES_DF,
        weights=weights_dict,
        era_normalize=req.era_normalize,
        min_starts=req.min_starts,
    )

    # pick columns to send to frontend
    cols = [
        "rank", "full_name", "goat_score",
        "career_score", "peak_score", "context_score", "longevity_score", "quali_score",
        "championships",
        "starts", "wins", "podiums",
    ]
    cols = [c for c in cols if c in ranked.columns]
    ranked_out = ranked[cols].head(req.top_n).copy()

    # make it JSON-serializable
    ranked_out = ranked_out.where(pd.notna(ranked_out), None)

    return {"rows": ranked_out.to_dict(orient="records")}
