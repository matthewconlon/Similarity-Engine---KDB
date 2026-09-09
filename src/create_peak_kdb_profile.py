"""Create an archetype profile from a player's peak seasons."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from src.common import attach_player_key, configure_logging, ensure_parent_dir, load_csv
    from src.config import DEFAULT_ARCHETYPE_PLAYER, FEATURE_COLUMNS
except ImportError:
    from common import attach_player_key, configure_logging, ensure_parent_dir, load_csv
    from config import DEFAULT_ARCHETYPE_PLAYER, FEATURE_COLUMNS

PROFILE_MODES = ["best_season", "best_three_average"]


def score_season(dataframe: pd.DataFrame) -> pd.Series:
    scoring_columns = [
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
    return dataframe[scoring_columns].sum(axis=1)


def create_archetype_profile(dataframe: pd.DataFrame, player_name: str, mode: str) -> pd.DataFrame:
    if mode not in PROFILE_MODES:
        raise ValueError(f"mode must be one of {PROFILE_MODES}; received {mode!r}")

    player_rows = dataframe[dataframe["player_name"].str.casefold() == player_name.casefold()].copy()
    if player_rows.empty:
        raise ValueError(f"No rows found for player: {player_name}")

    player_rows["composite_score"] = score_season(player_rows)
    player_rows = player_rows.sort_values("composite_score", ascending=False)

    if mode == "best_season":
        selected = player_rows.head(1)
        profile_label = f"Peak {player_name}"
    else:
        selected = player_rows.head(min(3, len(player_rows)))
        profile_label = f"Peak {player_name}"

    vector = selected[FEATURE_COLUMNS].mean().to_frame().T
    vector.insert(0, "player_name", profile_label)
    vector.insert(1, "season", mode)
    vector.insert(2, "club", "Archetype")
    vector.insert(3, "league", "Archetype")
    vector = attach_player_key(vector)
    return vector


def create_peak_profile(
    features_file: Path,
    output_file: Path,
    mode: str,
    player_name: str = DEFAULT_ARCHETYPE_PLAYER,
    verbose: bool = False,
) -> pd.DataFrame:
    configure_logging(verbose)
    dataframe = load_csv(features_file, "Features file")
    vector = create_archetype_profile(dataframe, player_name=player_name, mode=mode)
    ensure_parent_dir(output_file)
    vector.to_csv(output_file, index=False)
    return vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a peak archetype profile from player features.")
    parser.add_argument("--features-file", type=Path, default=Path("data/player_season_features.csv"))
    parser.add_argument("--output-file", type=Path, default=Path("data/peak_kdb_profile.csv"))
    parser.add_argument("--mode", choices=PROFILE_MODES, default="best_three_average")
    parser.add_argument("--player-name", type=str, default=DEFAULT_ARCHETYPE_PLAYER)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = create_peak_profile(
        args.features_file,
        args.output_file,
        args.mode,
        player_name=args.player_name,
        verbose=args.verbose,
    )
    print(profile.T)


if __name__ == "__main__":
    main()
