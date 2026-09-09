# Peak KDB Similarity Engine

A production-hardened Python project for building **archetype-based football similarity searches** from public StatsBomb event data.

> **Core principle:** this project does **not** predict the "best" player.
> It creates an archetype — such as **Peak Kevin De Bruyne**, **Peak Rodri**, **Peak Bernardo Silva**, or **Peak Jude Bellingham** — and finds the most similar player-season profiles in multi-dimensional feature space.

## What is included

- automated StatsBomb Open Data download
- player-season feature engineering from event data
- arbitrary-player archetype creation using:
  - best season
  - best three-season average
- similarity scoring with:
  - cosine similarity
  - euclidean distance
  - mahalanobis distance
- explainability outputs showing closest metrics and biggest gaps
- visual outputs saved into `output/`
  - PCA similarity map
  - radar chart
  - cluster diagram
  - optional UMAP map when `umap-learn` is available
- Streamlit app for interactive archetype search and filtering

## Repository structure

```text
.
├── app.py
├── notebooks/
│   ├── 01_Data_Acquisition.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Peak_KDB_Similarity.ipynb
├── src/
│   ├── __init__.py
│   ├── common.py
│   ├── config.py
│   ├── build_player_features.py
│   ├── create_peak_kdb_profile.py
│   ├── download_statsbomb_data.py
│   ├── explainability.py
│   ├── pca_visualisation.py
│   ├── similarity_engine.py
│   └── visualisations.py
├── data/
├── output/
├── README.md
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Recommended CLI usage

The scripts support both styles below:

```bash
python -m src.download_statsbomb_data --help
python src/download_statsbomb_data.py --help
```

For the most reliable imports, prefer `python -m src.<module>`.

## End-to-end pipeline

### 1) Download StatsBomb Open Data

```bash
python -m src.download_statsbomb_data \
  --output-dir data/raw \
  --limit-matches 250
```

### 2) Build player-season features

```bash
python -m src.build_player_features \
  --raw-dir data/raw \
  --output-file data/player_season_features.csv
```

### 3) Create an archetype profile

Kevin De Bruyne example:

```bash
python -m src.create_peak_kdb_profile \
  --features-file data/player_season_features.csv \
  --output-file data/peak_kdb_profile.csv \
  --player-name "Kevin De Bruyne" \
  --mode best_three_average
```

Other players work the same way:

```bash
python -m src.create_peak_kdb_profile \
  --features-file data/player_season_features.csv \
  --output-file data/peak_rodri_profile.csv \
  --player-name "Rodri" \
  --mode best_season
```

### 4) Run the similarity engine

```bash
python -m src.similarity_engine \
  --features-file data/player_season_features.csv \
  --profile-file data/peak_kdb_profile.csv \
  --metric cosine \
  --top-n 25 \
  --min-minutes 900 \
  --exclude-player-name "Kevin De Bruyne" \
  --output-file output/similarity_rankings.csv
```

Optional filters:

- `--league <league>` (repeatable)
- `--club <club>` (repeatable)
- `--season <exact season label>`
- `--min-minutes <minutes>`
- `--exclude-player-name <player>`

### 5) Generate explainability output

```bash
python -m src.explainability \
  --features-file data/player_season_features.csv \
  --profile-file data/peak_kdb_profile.csv \
  --rankings-file output/similarity_rankings.csv \
  --output-file output/player_explanations.csv
```

### 6) Generate visualisations

```bash
python -m src.visualisations \
  --features-file data/player_season_features.csv \
  --profile-file data/peak_kdb_profile.csv \
  --rankings-file output/similarity_rankings.csv \
  --output-dir output
```

Outputs are created automatically under:

- `output/pca_plots/`
- `output/radar_charts/`
- `output/cluster_plots/`
- `output/umap_plots/` (when UMAP is available)

## Streamlit app

After generating `data/player_season_features.csv`, launch:

```bash
streamlit run app.py
```

The app lets you:

- select any player as the archetype source
- choose archetype mode: best season or best three average
- choose similarity metric: cosine, euclidean, or mahalanobis
- filter candidate players by league, club, minimum minutes, and season
- inspect top similar players and explainability output
- display PCA, radar, cluster, and optional UMAP views
- save outputs into `output/streamlit/`

## Engineered metrics

The project builds a multi-metric profile spanning:

- passing
- creativity
- progression
- attacking
- possession
- defensive work
- availability

Representative examples include:

- progressive passes
- key passes
- through balls
- pass completion %
- passes into final third
- passes into penalty area
- xA proxy
- shot creating actions
- progressive carries
- goals and assists
- ball recoveries
- pressures, tackles, and interceptions
- minutes played, starts, matches played, and age

## Production hardening improvements

Compared with the initial scaffold, the repository now includes:

- shared configuration and utility modules under `src/`
- stronger file and argument validation
- output-directory creation in every writing step
- logging support via `--verbose`
- import paths that work for both package-style CLI usage and notebooks
- exact `player_key` matching to avoid ambiguous explainability and plotting results for duplicated names
- modular plotting helpers reusable by CLI scripts and the Streamlit app
- graceful UMAP degradation when the optional dependency is unavailable

## Notebook/tutorial framing

The notebooks remain aligned to the same hackathon tutorial flow:

1. data acquisition
2. feature engineering
3. archetype creation
4. scaling and similarity scoring
5. explainability
6. visualisation
7. **Same mathematics. Different domain.**

## Notes and assumptions

- StatsBomb Open Data coverage varies by competition and season.
- Open event data does not expose every proprietary scouting metric, so the project uses transparent approximations where needed.
- `xA`, shot-creating actions, and goal-creating actions are public-data approximations based on nearby actions and shot outcomes.
- If `umap-learn` is not installed or importable, the rest of the project continues to run normally.

## Same mathematics. Different domain.

This workflow can be transferred beyond football:

- wells
- equipment
- production assets
- digital use cases

The pattern stays the same:

1. define a target archetype
2. represent entities as vectors
3. normalise the data
4. compute similarity in feature space
5. explain the similarities and gaps

## License

MIT
