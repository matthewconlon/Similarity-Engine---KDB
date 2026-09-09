# Peak KDB Similarity Engine

A production-quality Python project that builds an **archetype-based football similarity engine** inspired by the idea behind Manchester City's multi-metric replacement scouting workflow.

> **Core principle:** this project does **not** try to predict the "best" player.
> It creates an archetype — **Peak Kevin De Bruyne** — and finds players whose statistical profiles are most similar in a multi-dimensional feature space.

## Project goals

- Acquire public football event data from **StatsBomb Open Data**.
- Engineer ~40 football performance metrics at player-season level.
- Build a **Peak Kevin De Bruyne** profile using either:
  - the best single season, or
  - the average of his best 3 seasons.
- Scale all features with `StandardScaler`.
- Compare all players to the archetype using:
  - cosine similarity
  - euclidean distance
  - mahalanobis distance
- Explain **why** players are similar.
- Produce hackathon-friendly visualisations and notebooks.

## Repository structure

```text
.
├── data/
├── notebooks/
│   ├── 01_Data_Acquisition.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Peak_KDB_Similarity.ipynb
├── output/
├── src/
│   ├── build_player_features.py
│   ├── create_peak_kdb_profile.py
│   ├── download_statsbomb_data.py
│   ├── explainability.py
│   ├── pca_visualisation.py
│   └── similarity_engine.py
├── README.md
└── requirements.txt
```

## Data source

Primary source: **StatsBomb Open Data**.

This project uses the public open-data repository structure published by StatsBomb / Hudl and downloads:
- competitions
- matches
- lineups
- events

## Metrics engineered

The pipeline generates event-derived player-season features across:

### Passing
- progressive passes
- key passes
- through balls
- pass completion %
- passes into final third
- passes into penalty area
- crosses
- switches
- total passes attempted/completed
- pass length

### Creativity
- expected assists (`xA`, approximated from shot xG on assisted shots)
- shot creating actions (approximated from pass/carry immediately preceding shot)
- goal creating actions (approximated from pass/carry immediately preceding goal)
- chances created
- big chances created proxy (pass leading to high-xG shot)

### Progression
- progressive carries
- carries into final third
- carries into penalty area
- total carries
- carry distance

### Attacking
- goals
- assists
- shots
- shots on target
- xG

### Possession
- touches
- ball recoveries
- turnovers
- dispossessions
- miscontrols
- fouls won

### Defensive
- pressures
- tackles
- interceptions
- duels won proxy

### Availability
- minutes played
- starts
- matches played
- age

## Important methodological note

Many commercial metrics are not directly available in open event data. Where needed, this project uses **transparent public approximations**:

- **xA**: summed xG of shots that directly follow a player's pass.
- **Shot Creating Actions**: player actions that directly precede a shot in the same possession chain approximation.
- **Goal Creating Actions**: same approach for goals.
- **Pressures**: based on StatsBomb pressure events.
- **Progressive passes/carries**: based on distance-to-goal reduction thresholds.

These approximations are documented in code comments and are intended for **education, prototyping, and hackathon use**.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\\Scripts\\activate   # Windows

pip install -r requirements.txt
```

## End-to-end usage

### 1) Download StatsBomb Open Data

```bash
python src/download_statsbomb_data.py --output-dir data/raw
```

### 2) Build player-season features

```bash
python src/build_player_features.py \
  --raw-dir data/raw \
  --output-file data/player_season_features.csv
```

### 3) Create Peak KDB archetype

```bash
python src/create_peak_kdb_profile.py \
  --features-file data/player_season_features.csv \
  --output-file data/peak_kdb_profile.csv \
  --mode best_three_average
```

Available modes:
- `best_season`
- `best_three_average`

### 4) Run similarity engine

```bash
python src/similarity_engine.py \
  --features-file data/player_season_features.csv \
  --profile-file data/peak_kdb_profile.csv \
  --metric cosine \
  --top-n 25 \
  --output-file output/similarity_rankings.csv
```

Available similarity metrics:
- `cosine`
- `euclidean`
- `mahalanobis`

### 5) Generate PCA visualisation

```bash
python src/pca_visualisation.py \
  --features-file data/player_season_features.csv \
  --profile-file data/peak_kdb_profile.csv \
  --rankings-file output/similarity_rankings.csv \
  --output-plot output/pca_peak_kdb.png
```

## Notebook flow

The notebooks are built as a tutorial for a corporate data science hackathon:

1. **01_Data_Acquisition.ipynb**
   - data source overview
   - automated download
   - raw data inspection

2. **02_Feature_Engineering.ipynb**
   - event cleaning
   - player-season aggregation
   - metric definitions
   - per-90 transformations

3. **03_Peak_KDB_Similarity.ipynb**
   - build Peak KDB archetype
   - scale features
   - compute similarities
   - PCA and explainability
   - final section: **Same mathematics. Different domain.**

## Hackathon framing

This project should be presented with the message:

> **Do not predict the best performer.**
> Instead, create an archetype and find similar entities.

### Same mathematics. Different domain.

The same workflow can be reused outside football:
- wells
- equipment
- production assets
- digital use cases

The pattern is unchanged:
1. define a target archetype
2. represent entities as vectors
3. normalize the data
4. compute similarity in feature space
5. explain the similarities and gaps

## Notes and assumptions

- StatsBomb Open Data does not cover every league and season uniformly.
- The project is designed to be **modular**, so alternative public datasets can be added later.
- Some player availability fields may be partially inferred from lineups/events.
- Age is computed when birth date is available in lineup metadata; otherwise left missing and imputed conservatively.

## Future extensions

- Streamlit app for arbitrary archetypes
- UMAP visualisation
- role-based clustering
- league filtering and minimum-minute thresholds
- comparison against other archetypes such as Rodri, Bernardo Silva, Jude Bellingham

## License

MIT
