"""Build player-season features from StatsBomb Open Data.

This module aggregates event-level data into player-season metrics suitable for
an archetype-based similarity engine.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FINAL_THIRD_X = 80.0
PENALTY_AREA_X = 102.0
PENALTY_AREA_Y_MIN = 18.0
PENALTY_AREA_Y_MAX = 62.0

FEATURE_COLUMNS = [
    "progressive_passes_p90",
    "key_passes_p90",
    "through_balls_p90",
    "pass_completion_pct",
    "passes_into_final_third_p90",
    "passes_into_penalty_area_p90",
    "crosses_p90",
    "switches_p90",
    "passes_attempted_p90",
    "passes_completed_p90",
    "avg_pass_length",
    "xa_p90",
    "shot_creating_actions_p90",
    "goal_creating_actions_p90",
    "chances_created_p90",
    "big_chances_created_p90",
    "progressive_carries_p90",
    "carries_into_final_third_p90",
    "carries_into_penalty_area_p90",
    "carries_p90",
    "carry_distance_p90",
    "goals_p90",
    "assists_p90",
    "shots_p90",
    "shots_on_target_p90",
    "xg_p90",
    "touches_p90",
    "ball_recoveries_p90",
    "turnovers_p90",
    "dispossessed_p90",
    "miscontrols_p90",
    "fouls_won_p90",
    "pressures_p90",
    "tackles_p90",
    "interceptions_p90",
    "duels_won_p90",
    "minutes_played",
    "starts",
    "matches_played",
    "age",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def distance_to_goal(location: list[float] | None) -> float | None:
    if not location or len(location) < 2:
        return None
    x, y = location[0], location[1]
    return float(np.sqrt((120.0 - x) ** 2 + (40.0 - y) ** 2))


def is_progressive(start: list[float] | None, end: list[float] | None) -> bool:
    start_d = distance_to_goal(start)
    end_d = distance_to_goal(end)
    if start_d is None or end_d is None:
        return False
    return (start_d - end_d) >= 10.0


def in_penalty_area(location: list[float] | None) -> bool:
    if not location or len(location) < 2:
        return False
    x, y = location[:2]
    return x >= PENALTY_AREA_X and PENALTY_AREA_Y_MIN <= y <= PENALTY_AREA_Y_MAX


def into_final_third(end: list[float] | None) -> bool:
    return bool(end and len(end) >= 2 and end[0] >= FINAL_THIRD_X)


def collect_match_metadata(raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for matches_file in (raw_dir / "matches").rglob("*.json"):
        matches = load_json(matches_file)
        for match in matches:
            rows.append(
                {
                    "match_id": match["match_id"],
                    "competition_name": match.get("competition", {}).get("competition_name"),
                    "season_name": match.get("season", {}).get("season_name"),
                    "match_date": match.get("match_date"),
                    "season_key": f"{match.get('competition', {}).get('competition_name', 'Unknown')} | {match.get('season', {}).get('season_name', 'Unknown')}",
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
            team_name = team_block.get("team_name")
            for player in team_block.get("lineup", []):
                player_id = player.get("player_id")
                player_meta[(match_id, player_id)] = {
                    "player_name": player.get("player_name"),
                    "team_name": team_name,
                    "birth_date": player.get("birth_date"),
                }
                if player.get("positions"):
                    earliest = min(player["positions"], key=lambda x: x.get("from", "99:99"))
                    if earliest.get("start_reason") == "Starting XI":
                        starters_by_match[match_id].add(player_id)
    return player_meta, starters_by_match


def build_features(raw_dir: Path) -> pd.DataFrame:
    match_meta = collect_match_metadata(raw_dir)
    meta_lookup = match_meta.set_index("match_id").to_dict("index")
    player_meta, starters_by_match = parse_lineups(raw_dir)
    records: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: defaultdict(float))

    for events_file in sorted((raw_dir / "events").glob("*.json")):
        match_id = int(events_file.stem)
        if match_id not in meta_lookup:
            continue
        match_info = meta_lookup[match_id]
        events = load_json(events_file)

        match_minutes_by_player: dict[int, float] = defaultdict(float)
        seen_players: set[int] = set()

        for idx, event in enumerate(events):
            player = event.get("player") or {}
            player_id = player.get("id")
            player_name = player.get("name")
            if player_id is None or player_name is None:
                continue

            seen_players.add(player_id)
            team_name = (event.get("team") or {}).get("name") or "Unknown"
            record_key = (player_name, match_info["season_key"], team_name)
            row = records[record_key]
            row["player_name"] = player_name
            row["season"] = match_info["season_key"]
            row["club"] = team_name
            row["league"] = match_info.get("competition_name") or "Unknown"
            row.setdefault("match_ids", set()).add(match_id)

            minute = float(event.get("minute", 0)) + float(event.get("second", 0)) / 60.0
            match_minutes_by_player[player_id] = max(match_minutes_by_player[player_id], minute)

            meta = player_meta.get((match_id, player_id), {})
            if match_id not in row.get("counted_start_matches", set()):
                row.setdefault("counted_start_matches", set()).add(match_id)
                row["starts"] += int(player_id in starters_by_match.get(match_id, set()))

            birth_date = meta.get("birth_date")
            if birth_date and match_info.get("match_date"):
                try:
                    row["age"] = (pd.to_datetime(match_info["match_date"]) - pd.to_datetime(birth_date)).days / 365.25
                except Exception:
                    pass

            event_type = (event.get("type") or {}).get("name")
            row["touches"] += 1

            if event_type == "Pass":
                p = event.get("pass", {})
                start = event.get("location")
                end = p.get("end_location")
                completed = p.get("outcome") is None
                row["passes_attempted"] += 1
                row["passes_completed"] += int(completed)
                row["pass_length_sum"] += p.get("length", 0.0) or 0.0
                row["key_passes"] += int(bool(p.get("shot_assist")))
                row["assists"] += int(bool(p.get("goal_assist")))
                row["through_balls"] += int(bool(p.get("through_ball")))
                row["crosses"] += int(bool(p.get("cross")))
                row["switches"] += int(bool(p.get("switch")))
                row["passes_into_final_third"] += int(into_final_third(end))
                row["passes_into_penalty_area"] += int(in_penalty_area(end))
                row["progressive_passes"] += int(is_progressive(start, end))

                if p.get("shot_assist"):
                    related_shots = [e for e in events[idx + 1 : idx + 4] if e.get("type", {}).get("name") == "Shot"]
                    if related_shots:
                        shot = related_shots[0]
                        shot_xg = (shot.get("shot") or {}).get("statsbomb_xg", 0.0) or 0.0
                        row["xa"] += shot_xg
                        row["chances_created"] += 1
                        row["big_chances_created"] += int(shot_xg >= 0.20)
                        row["shot_creating_actions"] += 1
                        if ((shot.get("shot") or {}).get("outcome") or {}).get("name") == "Goal":
                            row["goal_creating_actions"] += 1

            elif event_type == "Carry":
                carry = event.get("carry", {})
                start = event.get("location")
                end = carry.get("end_location")
                row["carries"] += 1
                if start and end and len(start) >= 2 and len(end) >= 2:
                    row["carry_distance_sum"] += np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                row["progressive_carries"] += int(is_progressive(start, end))
                row["carries_into_final_third"] += int(into_final_third(end))
                row["carries_into_penalty_area"] += int(in_penalty_area(end))

                related_shots = [e for e in events[idx + 1 : idx + 3] if e.get("type", {}).get("name") == "Shot"]
                if related_shots:
                    row["shot_creating_actions"] += 1
                    if ((related_shots[0].get("shot") or {}).get("outcome") or {}).get("name") == "Goal":
                        row["goal_creating_actions"] += 1

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
            player_name = meta.get("player_name")
            team_name = meta.get("team_name") or "Unknown"
            record_key = (player_name, match_info["season_key"], team_name)
            records[record_key]["minutes_played"] += max(match_minutes_by_player[player_id], 1.0)

    rows = []
    for row in records.values():
        minutes = max(float(row.get("minutes_played", 0.0)), 1.0)
        nineties = minutes / 90.0
        row["matches_played"] = len(row.get("match_ids", set()))
        row["pass_completion_pct"] = 100 * row.get("passes_completed", 0.0) / max(row.get("passes_attempted", 1.0), 1.0)
        row["avg_pass_length"] = row.get("pass_length_sum", 0.0) / max(row.get("passes_attempted", 1.0), 1.0)
        row["carry_distance_p90"] = row.get("carry_distance_sum", 0.0) / nineties

        count_metrics = [
            "progressive_passes", "key_passes", "through_balls", "passes_into_final_third", "passes_into_penalty_area",
            "crosses", "switches", "passes_attempted", "passes_completed", "xa", "shot_creating_actions",
            "goal_creating_actions", "chances_created", "big_chances_created", "progressive_carries",
            "carries_into_final_third", "carries_into_penalty_area", "carries", "goals", "assists", "shots",
            "shots_on_target", "xg", "touches", "ball_recoveries", "turnovers", "dispossessed", "miscontrols",
            "fouls_won", "pressures", "tackles", "interceptions", "duels_won"
        ]
        for metric in count_metrics:
            row[f"{metric}_p90"] = row.get(metric, 0.0) / nineties

        out = {col: row.get(col, np.nan) for col in ["player_name", "season", "club", "league"] + FEATURE_COLUMNS}
        rows.append(out)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["age"] = df["age"].fillna(df["age"].median())
    df["starts"] = df["starts"].fillna(0)
    df["matches_played"] = df["matches_played"].fillna(0)
    df["minutes_played"] = df["minutes_played"].fillna(0)
    return df.sort_values(["player_name", "season"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player-season features from StatsBomb Open Data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-file", type=Path, default=Path("data/player_season_features.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_features(args.raw_dir)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(f"Saved player-season features to {args.output_file}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")


if __name__ == "__main__":
    main()
