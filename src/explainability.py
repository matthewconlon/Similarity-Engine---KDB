"""Explainability utilities for similarity recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from build_player_features import FEATURE_COLUMNS


def create_explanations(features_df: pd.DataFrame, archetype_df: pd.DataFrame, rankings_df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    combined = pd.concat([features_df.copy(), archetype_df.copy()], ignore_index=True)
    combined[FEATURE_COLUMNS] = combined[FEATURE_COLUMNS].fillna(combined[FEATURE_COLUMNS].median())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined[FEATURE_COLUMNS])

    scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLUMNS)
    scaled_df["player_name"] = combined["player_name"].values
    target = scaled_df.iloc[-1]

    explanation_rows = []
    top_players = rankings_df.head(top_n)["player_name"].tolist()

    for player in top_players:
        row = scaled_df[scaled_df["player_name"] == player]
        if row.empty:
            continue
        row = row.iloc[0]
        deltas = row[FEATURE_COLUMNS] - target[FEATURE_COLUMNS]
        closest = deltas.abs().sort_values().head(5).index.tolist()
        largest_gaps = deltas.abs().sort_values(ascending=False).head(5).index.tolist()
        explanation_rows.append(
            {
                "player_name": player,
                "most_similar_metrics": ", ".join(closest),
                "largest_gaps_vs_peak_kdb": ", ".join(largest_gaps),
            }
        )

    return pd.DataFrame(explanation_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate explainability report for similarity results.")
    parser.add_argument("--features-file", type=Path, default=Path("data/player_season_features.csv"))
    parser.add_argument("--profile-file", type=Path, default=Path("data/peak_kdb_profile.csv"))
    parser.add_argument("--rankings-file", type=Path, default=Path("output/similarity_rankings.csv"))
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--output-file", type=Path, default=Path("output/player_explanations.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_df = pd.read_csv(args.features_file)
    archetype_df = pd.read_csv(args.profile_file)
    rankings_df = pd.read_csv(args.rankings_file)
    explanations = create_explanations(features_df, archetype_df, rankings_df, args.top_n)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    explanations.to_csv(args.output_file, index=False)
    print(explanations.to_string(index=False))


if __name__ == "__main__":
    main()
