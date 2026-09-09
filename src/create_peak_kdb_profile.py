"""Create the Peak Kevin De Bruyne archetype profile."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_player_features import FEATURE_COLUMNS

DEFAULT_KDB_NAME = "Kevin De Bruyne"


def score_season(df: pd.DataFrame) -> pd.Series:
    cols = [
        "xa_p90",
        "key_passes_p90",
        "through_balls_p90",
        "progressive_passes_p90",
        "chances_created_p90",
        "goal_creating_actions_p90",
        "assists_p90",
        "shots_p90",
        "xg_p90",
    ]
    return df[cols].sum(axis=1)


def create_peak_profile(features_file: Path, output_file: Path, mode: str, player_name: str = DEFAULT_KDB_NAME) -> pd.DataFrame:
    df = pd.read_csv(features_file)
    kdb = df[df["player_name"].str.lower() == player_name.lower()].copy()
    if kdb.empty:
        raise ValueError(f"No rows found for player: {player_name}")

    kdb["composite_score"] = score_season(kdb)
    kdb = kdb.sort_values("composite_score", ascending=False)

    if mode == "best_season":
        selected = kdb.head(1)
    elif mode == "best_three_average":
        selected = kdb.head(min(3, len(kdb)))
    else:
        raise ValueError("mode must be 'best_season' or 'best_three_average'")

    vector = selected[FEATURE_COLUMNS].mean().to_frame().T
    vector.insert(0, "player_name", f"Peak {player_name}")
    vector.insert(1, "season", mode)
    vector.insert(2, "club", "Archetype")
    vector.insert(3, "league", "Archetype")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    vector.to_csv(output_file, index=False)
    return vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Peak Kevin De Bruyne profile.")
    parser.add_argument("--features-file", type=Path, default=Path("data/player_season_features.csv"))
    parser.add_argument("--output-file", type=Path, default=Path("data/peak_kdb_profile.csv"))
    parser.add_argument("--mode", choices=["best_season", "best_three_average"], default="best_three_average")
    parser.add_argument("--player-name", type=str, default=DEFAULT_KDB_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = create_peak_profile(args.features_file, args.output_file, args.mode, args.player_name)
    print(profile.T)


if __name__ == "__main__":
    main()
