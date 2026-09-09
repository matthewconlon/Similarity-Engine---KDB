"""Reusable plotting utilities for archetype similarity outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    import umap.umap_ as umap
except ImportError:  # pragma: no cover - optional dependency path
    umap = None

try:
    from src.common import attach_player_key, configure_logging, ensure_directory, fill_feature_nans, load_csv, slugify
    from src.config import DEFAULT_TOP_N, FEATURE_COLUMNS, FEATURE_LABELS, RADAR_FEATURE_COLUMNS
except ImportError:
    from common import attach_player_key, configure_logging, ensure_directory, fill_feature_nans, load_csv, slugify
    from config import DEFAULT_TOP_N, FEATURE_COLUMNS, FEATURE_LABELS, RADAR_FEATURE_COLUMNS

sns.set_theme(style="whitegrid")


def prepare_visualisation_frame(
    features_df: pd.DataFrame,
    archetype_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    features = attach_player_key(features_df)
    archetype = attach_player_key(archetype_df)
    rankings = attach_player_key(rankings_df)
    combined = pd.concat([features.copy(), archetype.copy()], ignore_index=True)
    combined = fill_feature_nans(combined, FEATURE_COLUMNS)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined[FEATURE_COLUMNS])

    top_keys = set(rankings.head(top_n)["player_key"])
    plot_df = combined[["player_key", "player_name", "club", "league", "season"]].copy()
    plot_df["highlight"] = "Other Players"
    plot_df.loc[plot_df["player_key"].isin(top_keys), "highlight"] = "Top Similar Players"
    plot_df.loc[plot_df["club"].eq("Archetype"), "highlight"] = "Selected Archetype"
    return combined, scaled, plot_df


def annotate_highlights(axis: plt.Axes, dataframe: pd.DataFrame) -> None:
    for _, row in dataframe[dataframe["highlight"] != "Other Players"].iterrows():
        axis.text(row["x"] + 0.05, row["y"] + 0.05, row["player_name"], fontsize=8)


def save_embedding_plot(
    plot_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    output_path: Path,
    hue: str = "highlight",
) -> Path:
    ensure_directory(output_path.parent)
    figure, axis = plt.subplots(figsize=(14, 9))
    sns.scatterplot(data=plot_df, x=x_column, y=y_column, hue=hue, alpha=0.78, s=70, ax=axis)
    axis.set_title(title)
    axis.set_xlabel(x_column.upper())
    axis.set_ylabel(y_column.upper())
    annotate_highlights(axis, plot_df.rename(columns={x_column: "x", y_column: "y"}))
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def build_pca_plot(
    features_df: pd.DataFrame,
    archetype_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    output_path: Path,
    top_n: int = DEFAULT_TOP_N,
) -> Path:
    combined, scaled, plot_df = prepare_visualisation_frame(features_df, archetype_df, rankings_df, top_n=top_n)
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled)
    plot_df["pc1"] = coordinates[:, 0]
    plot_df["pc2"] = coordinates[:, 1]
    archetype_name = combined.iloc[-1]["player_name"]
    return save_embedding_plot(
        plot_df,
        x_column="pc1",
        y_column="pc2",
        title=f"{archetype_name} Similarity Map (PCA)",
        output_path=output_path,
    )


def build_cluster_plot(
    features_df: pd.DataFrame,
    archetype_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    output_path: Path,
    top_n: int = DEFAULT_TOP_N,
    n_clusters: int = 6,
) -> Path:
    combined, scaled, plot_df = prepare_visualisation_frame(features_df, archetype_df, rankings_df, top_n=top_n)
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled)
    cluster_count = max(2, min(n_clusters, len(plot_df) - 1))
    cluster_labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(scaled)
    plot_df["pc1"] = coordinates[:, 0]
    plot_df["pc2"] = coordinates[:, 1]
    plot_df["cluster"] = cluster_labels.astype(str)
    archetype_name = combined.iloc[-1]["player_name"]
    return save_embedding_plot(
        plot_df,
        x_column="pc1",
        y_column="pc2",
        title=f"{archetype_name} Player Grouping Map",
        output_path=output_path,
        hue="cluster",
    )


def build_umap_plot(
    features_df: pd.DataFrame,
    archetype_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    output_path: Path,
    top_n: int = DEFAULT_TOP_N,
) -> Path | None:
    if umap is None:
        return None

    combined, scaled, plot_df = prepare_visualisation_frame(features_df, archetype_df, rankings_df, top_n=top_n)
    reducer = umap.UMAP(n_components=2, random_state=42)
    coordinates = reducer.fit_transform(scaled)
    plot_df["umap1"] = coordinates[:, 0]
    plot_df["umap2"] = coordinates[:, 1]
    archetype_name = combined.iloc[-1]["player_name"]
    return save_embedding_plot(
        plot_df,
        x_column="umap1",
        y_column="umap2",
        title=f"{archetype_name} Similarity Map (UMAP)",
        output_path=output_path,
    )


def build_radar_chart(
    features_df: pd.DataFrame,
    archetype_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    output_path: Path,
    player_keys: Iterable[str] | None = None,
    max_players: int = 3,
) -> Path:
    features = attach_player_key(features_df)
    archetype = attach_player_key(archetype_df)
    rankings = attach_player_key(rankings_df)

    selected_keys = list(player_keys or rankings.head(max_players)["player_key"].tolist())
    comparison = pd.concat(
        [archetype, features[features["player_key"].isin(selected_keys)]],
        ignore_index=True,
    )
    if comparison.empty:
        raise ValueError("No players available to plot on the radar chart.")

    scaler = MinMaxScaler()
    scaled_pool = pd.concat([features, archetype], ignore_index=True)
    scaled_pool = fill_feature_nans(scaled_pool, RADAR_FEATURE_COLUMNS)
    scaled_values = scaler.fit_transform(scaled_pool[RADAR_FEATURE_COLUMNS])
    scaled_pool = scaled_pool[["player_key", "player_name"] + RADAR_FEATURE_COLUMNS].copy()
    scaled_pool[RADAR_FEATURE_COLUMNS] = scaled_values
    radar_frame = scaled_pool[scaled_pool["player_key"].isin(comparison["player_key"])]

    categories = [FEATURE_LABELS[column] for column in RADAR_FEATURE_COLUMNS]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ensure_directory(output_path.parent)
    figure, axis = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    for _, row in radar_frame.iterrows():
        values = row[RADAR_FEATURE_COLUMNS].tolist()
        values += values[:1]
        axis.plot(angles, values, linewidth=2, label=row["player_name"])
        axis.fill(angles, values, alpha=0.1)

    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(categories)
    axis.set_yticklabels([])
    axis.set_title(f"Archetype Radar: {archetype.iloc[0]['player_name']} vs Similar Profiles")
    axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def generate_visualisations(
    features_df: pd.DataFrame,
    archetype_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Path | None]:
    archetype_name = attach_player_key(archetype_df).iloc[0]["player_name"]
    slug = slugify(archetype_name)
    paths = {
        "pca": build_pca_plot(
            features_df,
            archetype_df,
            rankings_df,
            output_dir / "pca_plots" / f"{slug}-pca.png",
            top_n=top_n,
        ),
        "cluster": build_cluster_plot(
            features_df,
            archetype_df,
            rankings_df,
            output_dir / "cluster_plots" / f"{slug}-clusters.png",
            top_n=top_n,
        ),
        "radar": build_radar_chart(
            features_df,
            archetype_df,
            rankings_df,
            output_dir / "radar_charts" / f"{slug}-radar.png",
        ),
    }
    paths["umap"] = build_umap_plot(
        features_df,
        archetype_df,
        rankings_df,
        output_dir / "umap_plots" / f"{slug}-umap.png",
        top_n=top_n,
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create similarity visualisations for an archetype search.")
    parser.add_argument("--features-file", type=Path, default=Path("data/player_season_features.csv"))
    parser.add_argument("--profile-file", type=Path, default=Path("data/peak_kdb_profile.csv"))
    parser.add_argument("--rankings-file", type=Path, default=Path("output/similarity_rankings.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    features_df = load_csv(args.features_file, "Features file")
    archetype_df = load_csv(args.profile_file, "Profile file")
    rankings_df = load_csv(args.rankings_file, "Rankings file")
    saved_paths = generate_visualisations(features_df, archetype_df, rankings_df, args.output_dir, top_n=args.top_n)
    for name, path in saved_paths.items():
        if path is None:
            print(f"Skipped {name} visualisation because umap-learn is unavailable.")
        else:
            print(f"Saved {name} visualisation to {path}")


if __name__ == "__main__":
    main()
