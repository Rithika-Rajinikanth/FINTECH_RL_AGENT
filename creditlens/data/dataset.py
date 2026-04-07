"""
CreditLens — Dataset Loader
Loads loans.parquet into memory once at startup for fast row access.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional
import random

import pandas as pd
from loguru import logger

PARQUET_PATH = Path(__file__).parent / "loans.parquet"


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    if not PARQUET_PATH.exists():
        logger.warning("loans.parquet not found — running data generation pipeline...")
        from creditlens.data.generate import run_pipeline
        run_pipeline()
    df = pd.read_parquet(PARQUET_PATH)
    logger.info(f"Dataset loaded: {len(df)} applicants")
    return df


def sample_applicants(
    n: int,
    fraud_count: int = 0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Sample n applicants from the dataset, ensuring fraud_count fraudsters are included.
    Returns a fresh shuffled DataFrame for each episode.
    """
    if seed is not None:
        random.seed(seed)

    df = load_dataset()
    fraud_df = df[df["is_fraud"] == True]
    clean_df = df[df["is_fraud"] == False]

    fraud_sample = fraud_df.sample(n=min(fraud_count, len(fraud_df)), random_state=seed)
    clean_needed = n - len(fraud_sample)
    clean_sample = clean_df.sample(n=min(clean_needed, len(clean_df)), random_state=seed)

    combined = pd.concat([fraud_sample, clean_sample]).sample(frac=1, random_state=seed).reset_index(drop=True)
    # Re-assign applicant IDs for the episode
    combined = combined.copy()
    combined["applicant_id"] = [f"EP_{i:03d}" for i in range(len(combined))]
    return combined
