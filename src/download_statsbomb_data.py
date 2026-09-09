"""Download StatsBomb Open Data files locally.

This script mirrors the public JSON structure from the StatsBomb open-data
repository so the rest of the project can run without manual download steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITIONS_URL = f"{BASE_URL}/competitions.json"


def fetch_json(url: str):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def save_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def download_competitions(output_dir: Path) -> list[dict]:
    competitions = fetch_json(COMPETITIONS_URL)
    save_json(competitions, output_dir / "competitions.json")
    return competitions


def iter_match_urls(competitions: Iterable[dict]) -> Iterable[tuple[int, int, str]]:
    for comp in competitions:
        competition_id = comp["competition_id"]
        season_id = comp["season_id"]
        yield competition_id, season_id, f"{BASE_URL}/matches/{competition_id}/{season_id}.json"


def download_all(output_dir: Path, limit_matches: int | None = None) -> None:
    competitions = download_competitions(output_dir)
    total_matches_downloaded = 0

    for competition_id, season_id, matches_url in tqdm(list(iter_match_urls(competitions)), desc="Competitions"):
        try:
            matches = fetch_json(matches_url)
        except requests.HTTPError:
            continue

        save_json(matches, output_dir / "matches" / str(competition_id) / f"{season_id}.json")

        for match in tqdm(matches, desc=f"Matches {competition_id}-{season_id}", leave=False):
            match_id = match["match_id"]

            if limit_matches is not None and total_matches_downloaded >= limit_matches:
                return

            events_url = f"{BASE_URL}/events/{match_id}.json"
            lineups_url = f"{BASE_URL}/lineups/{match_id}.json"

            try:
                events = fetch_json(events_url)
                lineups = fetch_json(lineups_url)
            except requests.HTTPError:
                continue

            save_json(events, output_dir / "events" / f"{match_id}.json")
            save_json(lineups, output_dir / "lineups" / f"{match_id}.json")
            total_matches_downloaded += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download StatsBomb Open Data locally.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--limit-matches",
        type=int,
        default=None,
        help="Optional limit for quick experimentation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_all(args.output_dir, args.limit_matches)
    print(f"Downloaded StatsBomb data into: {args.output_dir}")


if __name__ == "__main__":
    main()
