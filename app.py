from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.common import ensure_directory, slugify
from src.config import DEFAULT_ARCHETYPE_PLAYER, DEFAULT_TOP_N
from src.create_peak_kdb_profile import create_archetype_profile
from src.explainability import create_explanations
from src.similarity_engine import score_similarity
from src.visualisations import generate_visualisations

st.set_page_config(page_title="Football Archetype Similarity Engine", layout="wide")

DEFAULT_FEATURES_PATH = Path("data/player_season_features.csv")
DEFAULT_OUTPUT_DIR = Path("output/streamlit")


@st.cache_data(show_spinner=False)
def load_features(path_as_text: str) -> pd.DataFrame:
    path = Path(path_as_text)
    if not path.exists():
        raise FileNotFoundError(
            f"Features file not found at {path}. Run src/build_player_features.py first or provide another path."
        )
    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError(f"Features file is empty: {path}")
    return dataframe


def filter_options(dataframe: pd.DataFrame, column: str) -> list[str]:
    return sorted(value for value in dataframe[column].dropna().astype(str).unique().tolist() if value)


def save_tables(output_dir: Path, slug: str, rankings: pd.DataFrame, explanations: pd.DataFrame) -> dict[str, Path]:
    ensure_directory(output_dir)
    rankings_path = output_dir / f"{slug}-rankings.csv"
    explanations_path = output_dir / f"{slug}-explanations.csv"
    rankings.to_csv(rankings_path, index=False)
    explanations.to_csv(explanations_path, index=False)
    return {"rankings": rankings_path, "explanations": explanations_path}


def main() -> None:
    st.title("Football Archetype Similarity Engine")
    st.caption("Same mathematics. Different domain. Build an archetype, then find the nearest profiles.")

    with st.sidebar:
        st.header("Inputs")
        features_path = st.text_input("Features CSV", str(DEFAULT_FEATURES_PATH))

    try:
        features_df = load_features(features_path)
    except Exception as exc:  # pragma: no cover - streamlit interaction path
        st.error(str(exc))
        st.stop()

    players = filter_options(features_df, "player_name")
    leagues = filter_options(features_df, "league")
    clubs = filter_options(features_df, "club")
    seasons = filter_options(features_df, "season")

    with st.sidebar:
        default_index = players.index(DEFAULT_ARCHETYPE_PLAYER) if DEFAULT_ARCHETYPE_PLAYER in players else 0
        source_player = st.selectbox("Archetype source player", options=players, index=default_index)
        archetype_mode = st.selectbox(
            "Archetype mode",
            options=["best_season", "best_three_average"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        similarity_metric = st.selectbox("Similarity metric", options=["cosine", "euclidean", "mahalanobis"])
        top_n = st.slider("Number of similar players", min_value=5, max_value=50, value=DEFAULT_TOP_N, step=5)
        min_minutes = st.slider("Minimum minutes", min_value=0, max_value=int(features_df["minutes_played"].max()), value=900, step=90)
        selected_leagues = st.multiselect("Candidate leagues", options=leagues)
        selected_clubs = st.multiselect("Candidate clubs", options=clubs)
        selected_season = st.selectbox("Candidate season", options=["All seasons"] + seasons)
        run_search = st.button("Generate similarity search", type="primary")

    if not run_search:
        st.info("Select an archetype and filters, then click **Generate similarity search**.")
        st.stop()

    archetype_df = create_archetype_profile(features_df, player_name=source_player, mode=archetype_mode)
    season_filter = None if selected_season == "All seasons" else selected_season
    rankings = score_similarity(
        features_df,
        archetype_df,
        metric=similarity_metric,
        top_n=top_n,
        leagues=selected_leagues,
        clubs=selected_clubs,
        min_minutes=float(min_minutes),
        season=season_filter,
        exclude_player_name=source_player,
    )
    explanations = create_explanations(features_df, archetype_df, rankings, top_n=top_n)
    results = rankings.merge(explanations, on=["player_key", "player_name", "season", "club"], how="left")

    output_dir = ensure_directory(DEFAULT_OUTPUT_DIR / slugify(f"{source_player}-{archetype_mode}-{similarity_metric}"))
    saved_tables = save_tables(output_dir, slugify(source_player), rankings, explanations)
    saved_plots = generate_visualisations(features_df, archetype_df, rankings, output_dir, top_n=top_n)

    st.subheader("Archetype summary")
    source_rows = features_df[features_df["player_name"].str.casefold() == source_player.casefold()].copy()
    source_rows = source_rows.assign(archetype_score=lambda frame: frame[[
        "xa_p90",
        "key_passes_p90",
        "through_balls_p90",
        "progressive_passes_p90",
        "chances_created_p90",
        "goal_creating_actions_p90",
        "assists_p90",
        "shots_p90",
        "xg_p90",
    ]].sum(axis=1)).sort_values("archetype_score", ascending=False)
    st.dataframe(source_rows[["player_name", "season", "club", "league", "minutes_played"]].head(3), use_container_width=True)

    st.subheader("Top similar players")
    st.dataframe(results, use_container_width=True)
    st.caption(f"Saved rankings to {saved_tables['rankings']} and explanations to {saved_tables['explanations']}.")

    visual_col1, visual_col2 = st.columns(2)
    with visual_col1:
        st.markdown("### PCA similarity map")
        st.image(str(saved_plots["pca"]))
        st.markdown("### Cluster diagram")
        st.image(str(saved_plots["cluster"]))
    with visual_col2:
        st.markdown("### Radar chart")
        st.image(str(saved_plots["radar"]))
        st.markdown("### Optional UMAP view")
        if saved_plots["umap"] is None:
            st.info("UMAP is unavailable in this environment. Install `umap-learn` to enable this view.")
        else:
            st.image(str(saved_plots["umap"]))

    st.subheader("Explainability view")
    st.dataframe(explanations, use_container_width=True)


if __name__ == "__main__":
    main()
