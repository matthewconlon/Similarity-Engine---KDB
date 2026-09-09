"""Similarity engine for archetype-based player search."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from build_player_features import FEATURE_COLUMNS


def prepare_feature_matrix(features_df: pd.DataFrame, archetype_df: pd.DataFrame):
    combined = pd.concat([features_df.copy(), archetype_df.copy()], ignore_index=True)
    combined[FEATURE_COLUMNS] = combined[FEATURE_COLUMNS].fillna(combined[FEATURE_COLUMNS].median())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined[FEATURE_COLUMNS])
    return combined, scaled, scaler


def score_similarity(features_df: pd.DataFrame, archetype_df: pd.DataFrame, metric: str = "cosine") -> pd.DataFrame:
    combined, scaled, _ = prepare_feature_matrix(features_df, archetype_df)
    target_vec = scaled[-1:].copy()
    player_vecs = scaled[:-1].copy()

    if metric == "cosine":
        distances = cdist(player_vecs, target_vec, metric="cosine").reshape(-1)
        scores = 1 - distances
    elif metric == "euclidean":
        distances = cdist(player_vecs, target_vec, metric="euclidean").reshape(-1)
        scores = 1 / (1 + distances)
    elif metric == "mahalanobis":
        lw = LedoitWolf().fit(player_vecs)
        vi = np.linalg.pinv(lw.covariance_)
        distances = cdist(player_vecs, target_vec, metric="mahalanobis", VI=vi).reshape(-1)
        scores = 1 / (1 + distances)
    else:
        raise ValueError("metric must be cosine, euclidean, or mahalanobis")

    result = combined.iloc[:-1][["player_name", "club", "league", "age", "season"]].copy()
    result["similarity_score"] = scores
    result = result.sort_values("similarity_score", ascending=False).reset_index(drop=True)
    result["ranking"] = np.arange(1, len(result) + 1)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run player similarity search against an archetype.")
    parser.add_argument("--features-file", type=Path, default=Path("data/player_season_features.csv"))
    parser.add_argument("--profile-file", type=Path, default=Path("data/peak_kdb_profile.csv"))
    parser.add_argument("--metric", choices=["cosine", "euclidean", "mahalanobis"], default="cosine")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--output-file", type=Path, default=Path("output/similarity_rankings.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_df = pd.read_csv(args.features_file)
    archetype_df = pd.read_csv(args.profile_file)
    rankings = score_similarity(features_df, archetype_df, metric=args.metric)
    top_n = rankings.head(args.top_n)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    top_n.to_csv(args.output_file, index=False)
    print(top_n.to_string(index=False))
    print(f"Saved rankings to {args.output_file}")


if __name__ == "__main__":
    main()
