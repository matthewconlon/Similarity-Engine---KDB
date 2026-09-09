"""Backward-compatible wrapper around the shared visualisations module."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from src.common import configure_logging, load_csv
    from src.visualisations import build_pca_plot, generate_visualisations
except ImportError:
    from common import configure_logging, load_csv
    from visualisations import build_pca_plot, generate_visualisations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PCA and companion visualisations for player similarity.")
    parser.add_argument("--features-file", type=Path, default=Path("data/player_season_features.csv"))
    parser.add_argument("--profile-file", type=Path, default=Path("data/peak_kdb_profile.csv"))
    parser.add_argument("--rankings-file", type=Path, default=Path("output/similarity_rankings.csv"))
    parser.add_argument("--output-plot", type=Path, default=Path("output/pca_peak_kdb.png"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for radar, cluster, and UMAP plots.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    features_df = load_csv(args.features_file, "Features file")
    archetype_df = load_csv(args.profile_file, "Profile file")
    rankings_df = load_csv(args.rankings_file, "Rankings file")

    build_pca_plot(features_df, archetype_df, rankings_df, args.output_plot)
    print(f"Saved PCA plot to {args.output_plot}")

    output_dir = args.output_dir or args.output_plot.parent
    saved = generate_visualisations(features_df, archetype_df, rankings_df, output_dir)
    for name, path in saved.items():
        if name == "pca":
            continue
        if path is None:
            print(f"Skipped {name} visualisation because umap-learn is unavailable.")
        else:
            print(f"Saved {name} visualisation to {path}")


if __name__ == "__main__":
    main()
