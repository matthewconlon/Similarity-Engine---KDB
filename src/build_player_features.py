"""Build player-season features from StatsBomb Open Data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.common import attach_player_key, configure_logging, ensure_parent_dir, validate_existing_path
    from src.config import DEFAULT_FALLBACK_AGE, FEATURE_COLUMNS
except ImportError:
    from common import attach_player_key, configure_logging, ensure_parent_dir, validate_existing_path
    from config import DEFAULT_FALLBACK_AGE, FEATURE_COLUMNS

FINAL_THIRD_X = 80.0
PENALTY_AREA_X = 102.0
PENALTY_AREA_Y_MIN = 18.0
PENALTY_AREA_Y_MAX = 62.0
COUNT_METRICS = [
    "progressive_passes",
    "key_passes",
    "through_balls",
    "passes_into_final_third",
    "passes_into_penalty_area",
    "crosses",
    "switches",
    "passes_attempted",
    "passes_completed",
    "xa",
    "shot_creating_actions",
    "goal_creating_actions",
    "chances_created",
    "big_chances_created",
    "progressive_carries",
    "carries_into_final_third",
    "carries_into_penalty_area",
    "carries",
    "goals",
    "assists",
    "shots",
    "shots_on_target",
    "xg",
    "touches",
    "ball_recoveries",
    "turnovers",
    "dispossessed",
    "miscontrols",
    "fouls_won",
    "pressures",
    "tackles",
    "interceptions",
    "duels_won",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def distance_to_goal(location: list[float] | None) -> float | None:
    if not location or len(location) < 2:
        return None
    x_coord, y_coord = location[:2]
    return float(np.sqrt((120.0 - x_coord) ** 2 + (40.0 - y_coord) ** 2))


def is_progressive(start: list[float] | None, end: list[float] | None) -> bool:
    start_distance = distance_to_goal(start)
    end_distance = distance_to_goal(end)
    if start_distance is None or end_distance is None:
        return False
    return (start_distance - end_distance) >= 10.0


def in_penalty_area(location: list[float] | None) -> bool:
    if not location or len(location) < 2:
        return False
    x_coord, y_coord = location[:2]
    return x_coord >= PENALTY_AREA_X and PENALTY_AREA_Y_MIN <= y_coord <= PENALTY_AREA_Y_MAX


def into_final_third(end_location: list[float] | None) -> bool:
    return bool(end_location and len(end_location) >= 2 and end_location[0] >= FINAL_THIRD_X)


def collect_match_metadata(raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for matches_file in (raw_dir / "matches").rglob("*.json"):
        matches = load_json(matches_file)
        for match in matches:
            competition_name = match.get("competition", {}).get("competition_name", "Unknown")
            season_name = match.get("season", {}).get("season_name", "Unknown")
            rows.append(
                {
                    "match_id": match["match_id"],
                    "competition_name": competition_name,
                    "season_name": season_name,
                    "match_date": match.get("match_date"),
                    "season_key": f"{competition_name} | {season_name}",
                }
            )
    return pd.DataFrame(rows)


def parse_lineups(raw_dir: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[int, set[int]]]:
    player_meta: dict[tuple[int, int], dict[str, Any]] = {}
    starters_by_match: dict[int, set[int]] = defaultdict(set)

    for lineup_file in (raw_dir / "lineups").glob("*.json"):
        match_id = int(lineup_file.stem)
        lineups = load_json(lineup_file)
        for team_block in lineups:
            team_name = team_block.get("team_name", "Unknown")
            for player in team_block.get("lineup", []):
                player_id = player.get("player_id")
                if player_id is None:
                    continue
                player_meta[(match_id, player_id)] = {
                    "player_name": player.get("player_name"),
                    "team_name": team_name,
                    "birth_date": player.get("birth_date"),
                }
                positions = player.get("positions") or []
                if positions:
                    earliest = min(positions, key=lambda position: position.get("from", "99:99"))
                    if earliest.get("start_reason") == "Starting XI":
                        starters_by_match[match_id].add(player_id)

    return player_meta, starters_by_match


def initialise_record(row: dict[str, Any], player_name: str, season_key: str, club: str, league: str) -> None:
    row["player_name"] = player_name
    row["season"] = season_key
    row["club"] = club
    row["league"] = league
    row.setdefault("match_ids", set())
    row.setdefault("counted_start_matches", set())


def update_pass_metrics(row: dict[str, Any], event: dict[str, Any], neighbour_events: list[dict[str, Any]]) -> None:
    pass_event = event.get("pass", {})
    start = event.get("location")
    end = pass_event.get("end_location")
    completed = pass_event.get("outcome") is None

    row["passes_attempted"] += 1
    row["passes_completed"] += int(completed)
    row["pass_length_sum"] += pass_event.get("length", 0.0) or 0.0
    row["key_passes"] += int(bool(pass_event.get("shot_assist")))
    row["assists"] += int(bool(pass_event.get("goal_assist")))
    row["through_balls"] += int(bool(pass_event.get("through_ball")))
    row["crosses"] += int(bool(pass_event.get("cross")))
    row["switches"] += int(bool(pass_event.get("switch")))
    row["passes_into_final_third"] += int(into_final_third(end))
    row["passes_into_penalty_area"] += int(in_penalty_area(end))
    row["progressive_passes"] += int(is_progressive(start, end))

    if not pass_event.get("shot_assist"):
        return

    related_shots = [candidate for candidate in neighbour_events if candidate.get("type", {}).get("name") == "Shot"]
    if not related_shots:
        return

    shot = related_shots[0]
    shot_xg = (shot.get("shot") or {}).get("statsbomb_xg", 0.0) or 0.0
    row["xa"] += shot_xg
    row["chances_created"] += 1
    row["big_chances_created"] += int(shot_xg >= 0.20)
    row["shot_creating_actions"] += 1
    if ((shot.get("shot") or {}).get("outcome") or {}).get("name") == "Goal":
        row["goal_creating_actions"] += 1


def update_carry_metrics(row: dict[str, Any], event: dict[str, Any], neighbour_events: list[dict[str, Any]]) -> None:
    carry = event.get("carry", {})
    start = event.get("location")
    end = carry.get("end_location")

    row["carries"] += 1
    if start and end and len(start) >= 2 and len(end) >= 2:
        row["carry_distance_sum"] += float(np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2))
    row["progressive_carries"] += int(is_progressive(start, end))
    row["carries_into_final_third"] += int(into_final_third(end))
    row["carries_into_penalty_area"] += int(in_penalty_area(end))

    related_shots = [candidate for candidate in neighbour_events if candidate.get("type", {}).get("name") == "Shot"]
    if related_shots:
        row["shot_creating_actions"] += 1
        if ((related_shots[0].get("shot") or {}).get("outcome") or {}).get("name") == "Goal":
            row["goal_creating_actions"] += 1


def finalise_records(records: dict[tuple[str, str, str], dict[str, Any]]) -> pd.DataFrame:
    output_rows: list[dict[str, Any]] = []

    for row in records.values():
        minutes_played = max(float(row.get("minutes_played", 0.0)), 1.0)
        nineties = minutes_played / 90.0
        row["matches_played"] = len(row.get("match_ids", set()))
        row["pass_completion_pct"] = 100 * row.get("passes_completed", 0.0) / max(row.get("passes_attempted", 1.0), 1.0)
        row["avg_pass_length"] = row.get("pass_length_sum", 0.0) / max(row.get("passes_attempted", 1.0), 1.0)
        row["carry_distance_p90"] = row.get("carry_distance_sum", 0.0) / nineties

        for metric in COUNT_METRICS:
            row[f"{metric}_p90"] = row.get(metric, 0.0) / nineties

        output_rows.append({col: row.get(col, np.nan) for col in ["player_name", "season", "club", "league"] + FEATURE_COLUMNS})

    features = pd.DataFrame(output_rows)
    if features.empty:
        return features

    if "age" in features.columns:
        age_series = features["age"].dropna()
        fallback_age = float(age_series.median()) if not age_series.empty else DEFAULT_FALLBACK_AGE
        features["age"] = features["age"].fillna(fallback_age)

    for column in ["starts", "matches_played", "minutes_played"]:
        features[column] = features[column].fillna(0)

    features = attach_player_key(features)
    return features.sort_values(["player_name", "season", "club"]).reset_index(drop=True)


def build_features(raw_dir: Path, verbose: bool = False) -> pd.DataFrame:
    logger = configure_logging(verbose)
    validate_existing_path(raw_dir, "Raw StatsBomb directory")

    events_dir = validate_existing_path(raw_dir / "events", "Events directory")
    validate_existing_path(raw_dir / "matches", "Matches directory")
    validate_existing_path(raw_dir / "lineups", "Lineups directory")

    match_meta = collect_match_metadata(raw_dir)
    if match_meta.empty:
        raise ValueError(f"No match metadata found under {raw_dir}")

    meta_lookup = match_meta.set_index("match_id").to_dict("index")
    player_meta, starters_by_match = parse_lineups(raw_dir)
    records: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: defaultdict(float))

    for events_file in sorted(events_dir.glob("*.json")):
        match_id = int(events_file.stem)
        match_info = meta_lookup.get(match_id)
        if match_info is None:
            logger.warning("Skipping events file without match metadata: %s", events_file)
            continue

        events = load_json(events_file)
        match_minutes_by_player: dict[int, float] = defaultdict(float)
        seen_players: set[int] = set()

        for index, event in enumerate(events):
            player = event.get("player") or {}
            player_id = player.get("id")
            player_name = player.get("name")
            if player_id is None or not player_name:
                continue

            seen_players.add(player_id)
            team_name = (event.get("team") or {}).get("name") or player_meta.get((match_id, player_id), {}).get("team_name") or "Unknown"
            season_key = match_info["season_key"]
            record_key = (player_name, season_key, team_name)
            row = records[record_key]
            initialise_record(row, player_name, season_key, team_name, match_info.get("competition_name") or "Unknown")
            row["match_ids"].add(match_id)

            minute = float(event.get("minute", 0)) + float(event.get("second", 0)) / 60.0
            match_minutes_by_player[player_id] = max(match_minutes_by_player[player_id], minute)

            if match_id not in row["counted_start_matches"]:
                row["counted_start_matches"].add(match_id)
                row["starts"] += int(player_id in starters_by_match.get(match_id, set()))

            birth_date = player_meta.get((match_id, player_id), {}).get("birth_date")
            if birth_date and match_info.get("match_date"):
                try:
                    row["age"] = (pd.to_datetime(match_info["match_date"]) - pd.to_datetime(birth_date)).days / 365.25
                except (TypeError, ValueError):
                    logger.debug("Unable to parse birth date for player=%s match=%s", player_name, match_id)

            event_type = (event.get("type") or {}).get("name")
            row["touches"] += 1
            neighbours = events[index + 1 : index + 4]

            if event_type == "Pass":
                update_pass_metrics(row, event, neighbours)
            elif event_type == "Carry":
                update_carry_metrics(row, event, neighbours[:2])
            elif event_type == "Shot":
                shot = event.get("shot", {})
                outcome = (shot.get("outcome") or {}).get("name")
                row["shots"] += 1
                row["xg"] += shot.get("statsbomb_xg", 0.0) or 0.0
                row["goals"] += int(outcome == "Goal")
                row["shots_on_target"] += int(outcome in {"Goal", "Saved", "Saved To Post"})
            elif event_type == "Pressure":
                row["pressures"] += 1
            elif event_type == "Ball Recovery":
                row["ball_recoveries"] += 1
            elif event_type == "Miscontrol":
                row["miscontrols"] += 1
            elif event_type == "Dispossessed":
                row["dispossessed"] += 1
            elif event_type == "Interception":
                row["interceptions"] += 1
            elif event_type == "Foul Won":
                row["fouls_won"] += 1
            elif event_type == "Duel":
                duel = event.get("duel", {})
                duel_outcome = (duel.get("outcome") or {}).get("name")
                duel_type = (duel.get("type") or {}).get("name")
                row["duels_won"] += int(duel_outcome in {"Won", "Success In Play", "Success Out"})
                row["tackles"] += int(duel_type == "Tackle")
            elif event_type == "Dribble":
                dribble_outcome = ((event.get("dribble") or {}).get("outcome") or {}).get("name")
                row["turnovers"] += int(dribble_outcome == "Incomplete")
            elif event_type == "50/50":
                fifty_outcome = ((event.get("50_50") or {}).get("outcome") or {}).get("name")
                row["duels_won"] += int(fifty_outcome in {"Won", "Success To Team"})

        for player_id in seen_players:
            meta = player_meta.get((match_id, player_id))
            if not meta:
                continue
            record_key = (meta.get("player_name"), match_info["season_key"], meta.get("team_name") or "Unknown")
            records[record_key]["minutes_played"] += max(match_minutes_by_player[player_id], 1.0)

    return finalise_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player-season features from StatsBomb Open Data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-file", type=Path, default=Path("data/player_season_features.csv"))
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = build_features(args.raw_dir, verbose=args.verbose)
    ensure_parent_dir(args.output_file)
    dataframe.to_csv(args.output_file, index=False)
    print(f"Saved player-season features to {args.output_file}")
    print(f"Rows: {len(dataframe):,} | Columns: {len(dataframe.columns):,}")


if __name__ == "__main__":
    main()
