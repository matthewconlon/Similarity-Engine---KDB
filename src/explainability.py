"""Explainability utilities for similarity recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from src.common import attach_player_key, configure_logging, ensure_parent_dir, fill_feature_nans, load_csv, validate_positive_int
    from src.config import DEFAULT_TOP_N, FEATURE_COLUMNS, FEATURE_LABELS
except ImportError:
    from common import attach_player_key, configure_logging, ensure_parent_dir, fill_feature_nans, load_csv, validate_positive_int
    from config import DEFAULT_TOP_N, FEATURE_COLUMNS, FEATURE_LABELS


def create_explanations(
    features_df: pd.DataFrame,
    archetype_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    validate_positive_int(top_n, "top_n")
    feature_rows = attach_player_key(features_df)
    ranking_rows = attach_player_key(rankings_df)
    archetype_rows = attach_player_key(archetype_df)

    combined = pd.concat([feature_rows.copy(), archetype_rows.copy()], ignore_index=True)
    combined = fill_feature_nans(combined, FEATURE_COLUMNS)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined[FEATURE_COLUMNS])

    scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLUMNS)
    scaled_df["player_key"] = combined["player_key"].values
    target = scaled_df.iloc[-1]

    explanation_rows: list[dict[str, str]] = []
    top_players = ranking_rows.head(top_n)

    for _, ranked_player in top_players.iterrows():
        row = scaled_df[scaled_df["player_key"] == ranked_player["player_key"]]
        if row.empty:
            continue

        candidate = row.iloc[0]
        deltas = candidate[FEATURE_COLUMNS] - target[FEATURE_COLUMNS]
        closest = deltas.abs().sort_values().head(5).index.tolist()
        largest_gaps = deltas.abs().sort_values(ascending=False).head(5).index.tolist()
        explanation_rows.append(
            {
                "player_key": ranked_player["player_key"],
                "player_name": ranked_player["player_name"],
                "season": ranked_player["season"],
                "club": ranked_player["club"],
                "most_similar_metrics": ", ".join(FEATURE_LABELS[column] for column in closest),
                "largest_gaps_vs_archetype": ", ".join(FEATURE_LABELS[column] for column in largest_gaps),
            }
        )

    return pd.DataFrame(explanation_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate explainability report for similarity results.")
    parser.add_argument("--features-file", type=Path, default=Path("data/player_season_features.csv"))
    parser.add_argument("--profile-file", type=Path, default=Path("data/peak_kdb_profile.csv"))
    parser.add_argument("--rankings-file", type=Path, default=Path("output/similarity_rankings.csv"))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--output-file", type=Path, default=Path("output/player_explanations.csv"))
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    features_df = load_csv(args.features_file, "Features file")
    archetype_df = load_csv(args.profile_file, "Profile file")
    rankings_df = load_csv(args.rankings_file, "Rankings file")
    explanations = create_explanations(features_df, archetype_df, rankings_df, args.top_n)
    ensure_parent_dir(args.output_file)
    explanations.to_csv(args.output_file, index=False)
    print(explanations.to_string(index=False))


if __name__ == "__main__":
    main()
