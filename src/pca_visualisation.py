"""PCA and cluster-style visualisation utilities for the similarity engine."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from build_player_features import FEATURE_COLUMNS

sns.set_theme(style="whitegrid")


def build_pca_plot(features_df: pd.DataFrame, archetype_df: pd.DataFrame, rankings_df: pd.DataFrame, output_plot: Path) -> pd.DataFrame:
    combined = pd.concat([features_df.copy(), archetype_df.copy()], ignore_index=True)
    combined[FEATURE_COLUMNS] = combined[FEATURE_COLUMNS].fillna(combined[FEATURE_COLUMNS].median())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined[FEATURE_COLUMNS])

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)

    pca_df = combined[["player_name", "club", "league", "season"]].copy()
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]
    pca_df["highlight"] = "Other Players"

    top_players = set(rankings_df["player_name"].head(25))
    pca_df.loc[pca_df["player_name"].isin(top_players), "highlight"] = "Top 25 Similar"
    pca_df.loc[pca_df["player_name"].str.startswith("Peak "), "highlight"] = "Peak KDB"

    plt.figure(figsize=(14, 9))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="highlight", alpha=0.75, s=70)

    for _, row in pca_df[pca_df["highlight"] != "Other Players"].iterrows():
        plt.text(row["PC1"] + 0.05, row["PC2"] + 0.05, row["player_name"], fontsize=8)

    plt.title("Peak KDB Similarity Map (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    plt.close()
    return pca_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PCA visualisation for player similarity.")
    parser.add_argument("--features-file", type=Path, default=Path("data/player_season_features.csv"))
    parser.add_argument("--profile-file", type=Path, default=Path("data/peak_kdb_profile.csv"))
    parser.add_argument("--rankings-file", type=Path, default=Path("output/similarity_rankings.csv"))
    parser.add_argument("--output-plot", type=Path, default=Path("output/pca_peak_kdb.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_df = pd.read_csv(args.features_file)
    archetype_df = pd.read_csv(args.profile_file)
    rankings_df = pd.read_csv(args.rankings_file)
    build_pca_plot(features_df, archetype_df, rankings_df, args.output_plot)
    print(f"Saved PCA plot to {args.output_plot}")


if __name__ == "__main__":
    main()
