"""Similarity engine for archetype-based player search."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

try:
    from src.common import attach_player_key, configure_logging, ensure_parent_dir, load_csv, normalise_filters, validate_non_negative_float, validate_positive_int
    from src.config import DEFAULT_TOP_N, FEATURE_COLUMNS
except ImportError:
    from common import attach_player_key, configure_logging, ensure_parent_dir, load_csv, normalise_filters, validate_non_negative_float, validate_positive_int
    from config import DEFAULT_TOP_N, FEATURE_COLUMNS

VALID_METRICS = ["cosine", "euclidean", "mahalanobis"]


def prepare_feature_matrix(features_df: pd.DataFrame, archetype_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    features = attach_player_key(features_df)
    archetype = attach_player_key(archetype_df)
    combined = pd.concat([features.copy(), archetype.copy()], ignore_index=True)
    combined[FEATURE_COLUMNS] = combined[FEATURE_COLUMNS].fillna(combined[FEATURE_COLUMNS].median())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined[FEATURE_COLUMNS])
    return combined, scaled, scaler


def filter_candidate_pool(
    dataframe: pd.DataFrame,
    leagues: Iterable[str] | None = None,
    clubs: Iterable[str] | None = None,
    min_minutes: float = 0.0,
    season: str | None = None,
    exclude_player_name: str | None = None,
) -> pd.DataFrame:
    filtered = attach_player_key(dataframe)
    leagues = normalise_filters(leagues)
    clubs = normalise_filters(clubs)
    validate_non_negative_float(min_minutes, "min_minutes")

    if leagues:
        filtered = filtered[filtered["league"].isin(leagues)]
    if clubs:
        filtered = filtered[filtered["club"].isin(clubs)]
    if season:
        filtered = filtered[filtered["season"] == season]
    if min_minutes:
        filtered = filtered[filtered["minutes_played"] >= min_minutes]
    if exclude_player_name:
        filtered = filtered[filtered["player_name"].str.casefold() != exclude_player_name.casefold()]

    if filtered.empty:
        raise ValueError("Candidate filters removed all players. Relax the filters and try again.")
    return filtered


def score_similarity(
    features_df: pd.DataFrame,
    archetype_df: pd.DataFrame,
    metric: str = "cosine",
    top_n: int | None = None,
    leagues: Iterable[str] | None = None,
    clubs: Iterable[str] | None = None,
    min_minutes: float = 0.0,
    season: str | None = None,
    exclude_player_name: str | None = None,
) -> pd.DataFrame:
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {VALID_METRICS}; received {metric!r}")
    if top_n is not None:
        validate_positive_int(top_n, "top_n")

    candidates = filter_candidate_pool(
        features_df,
        leagues=leagues,
        clubs=clubs,
        min_minutes=min_minutes,
        season=season,
        exclude_player_name=exclude_player_name,
    )
    combined, scaled, _ = prepare_feature_matrix(candidates, archetype_df)
    target_vec = scaled[-1:].copy()
    player_vecs = scaled[:-1].copy()

    if metric == "cosine":
        distances = cdist(player_vecs, target_vec, metric="cosine").reshape(-1)
        scores = 1 - distances
    elif metric == "euclidean":
        distances = cdist(player_vecs, target_vec, metric="euclidean").reshape(-1)
        scores = 1 / (1 + distances)
    else:
        lw_model = LedoitWolf().fit(player_vecs)
        inverse_covariance = np.linalg.pinv(lw_model.covariance_)
        distances = cdist(player_vecs, target_vec, metric="mahalanobis", VI=inverse_covariance).reshape(-1)
        scores = 1 / (1 + distances)

    result = combined.iloc[:-1][["player_key", "player_name", "club", "league", "age", "season", "minutes_played"]].copy()
    result["similarity_score"] = scores
    result = result.sort_values("similarity_score", ascending=False).reset_index(drop=True)
    result["ranking"] = np.arange(1, len(result) + 1)
    if top_n is not None:
        result = result.head(top_n).reset_index(drop=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run player similarity search against an archetype.")
    parser.add_argument("--features-file", type=Path, default=Path("data/player_season_features.csv"))
    parser.add_argument("--profile-file", type=Path, default=Path("data/peak_kdb_profile.csv"))
    parser.add_argument("--metric", choices=VALID_METRICS, default="cosine")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--output-file", type=Path, default=Path("output/similarity_rankings.csv"))
    parser.add_argument("--league", action="append", default=[], help="Optional league filter; repeat for multiple leagues.")
    parser.add_argument("--club", action="append", default=[], help="Optional club filter; repeat for multiple clubs.")
    parser.add_argument("--season", type=str, default=None, help="Optional exact season filter.")
    parser.add_argument("--min-minutes", type=float, default=0.0, help="Minimum minutes filter for candidate players.")
    parser.add_argument("--exclude-player-name", type=str, default=None, help="Exclude all rows for the source player from results.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    features_df = load_csv(args.features_file, "Features file")
    archetype_df = load_csv(args.profile_file, "Profile file")
    rankings = score_similarity(
        features_df,
        archetype_df,
        metric=args.metric,
        top_n=args.top_n,
        leagues=args.league,
        clubs=args.club,
        min_minutes=args.min_minutes,
        season=args.season,
        exclude_player_name=args.exclude_player_name,
    )
    ensure_parent_dir(args.output_file)
    rankings.to_csv(args.output_file, index=False)
    print(rankings.to_string(index=False))
    print(f"Saved rankings to {args.output_file}")


if __name__ == "__main__":
    main()
