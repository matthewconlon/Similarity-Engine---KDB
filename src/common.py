"""Shared helpers for CLI scripts, notebooks, and the Streamlit app."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER_NAME = "similarity_engine"


def configure_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger(LOGGER_NAME)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def validate_positive_int(value: int, label: str) -> int:
    if value <= 0:
        raise ValueError(f"{label} must be greater than 0; received {value}.")
    return value


def validate_non_negative_float(value: float, label: str) -> float:
    if value < 0:
        raise ValueError(f"{label} must be non-negative; received {value}.")
    return value


def load_csv(path: Path, label: str) -> pd.DataFrame:
    validate_existing_path(path, label)
    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError(f"{label} is empty: {path}")
    return dataframe


def build_player_key(player_name: str, season: str, club: str) -> str:
    return " | ".join(str(value).strip() for value in (player_name, season, club))


def attach_player_key(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    required = {"player_name", "season", "club"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cannot create player_key; missing columns: {sorted(missing)}")
    frame["player_key"] = frame.apply(
        lambda row: build_player_key(row["player_name"], row["season"], row["club"]),
        axis=1,
    )
    return frame


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "output"


def normalise_filters(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    return [value for value in values if str(value).strip()]
