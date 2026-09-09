"""Download StatsBomb Open Data files locally with basic production safeguards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

try:
    from src.common import configure_logging, ensure_directory, validate_non_negative_float
except ImportError:
    from common import configure_logging, ensure_directory, validate_non_negative_float

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITIONS_URL = f"{BASE_URL}/competitions.json"


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_json(session: requests.Session, url: str) -> Any:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False)


def download_competitions(session: requests.Session, output_dir: Path) -> list[dict[str, Any]]:
    competitions = fetch_json(session, COMPETITIONS_URL)
    save_json(competitions, output_dir / "competitions.json")
    return competitions


def iter_match_urls(competitions: Iterable[dict[str, Any]]) -> Iterable[tuple[int, int, str]]:
    for competition in competitions:
        competition_id = competition["competition_id"]
        season_id = competition["season_id"]
        yield competition_id, season_id, f"{BASE_URL}/matches/{competition_id}/{season_id}.json"


def download_all(output_dir: Path, limit_matches: int | None = None, verbose: bool = False) -> dict[str, int]:
    logger = configure_logging(verbose)
    ensure_directory(output_dir)
    if limit_matches is not None:
        validate_non_negative_float(limit_matches, "limit_matches")

    session = build_session()
    competitions = download_competitions(session, output_dir)
    totals = {"competitions": len(competitions), "matches": 0, "events": 0, "lineups": 0}

    for competition_id, season_id, matches_url in tqdm(list(iter_match_urls(competitions)), desc="Competitions"):
        try:
            matches = fetch_json(session, matches_url)
        except requests.HTTPError as exc:
            logger.warning("Skipping matches for competition=%s season=%s: %s", competition_id, season_id, exc)
            continue

        save_json(matches, output_dir / "matches" / str(competition_id) / f"{season_id}.json")

        for match in tqdm(matches, desc=f"Matches {competition_id}-{season_id}", leave=False):
            if limit_matches is not None and totals["matches"] >= limit_matches:
                logger.info("Reached match download limit of %s", limit_matches)
                return totals

            match_id = match["match_id"]
            events_url = f"{BASE_URL}/events/{match_id}.json"
            lineups_url = f"{BASE_URL}/lineups/{match_id}.json"

            try:
                events = fetch_json(session, events_url)
                lineups = fetch_json(session, lineups_url)
            except (requests.HTTPError, requests.JSONDecodeError) as exc:
                logger.warning("Skipping match_id=%s due to download error: %s", match_id, exc)
                continue

            save_json(events, output_dir / "events" / f"{match_id}.json")
            save_json(lineups, output_dir / "lineups" / f"{match_id}.json")
            totals["matches"] += 1
            totals["events"] += 1
            totals["lineups"] += 1

    logger.info("Download complete: %s", totals)
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download StatsBomb Open Data locally.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--limit-matches",
        type=int,
        default=None,
        help="Optional limit for quick experimentation.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    totals = download_all(args.output_dir, args.limit_matches, verbose=args.verbose)
    print(f"Downloaded StatsBomb data into: {args.output_dir}")
    print(totals)


if __name__ == "__main__":
    main()
