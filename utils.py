"""
utils.py

General-purpose utilities used across the Sustainability Tracker app.

Provided helpers:
- format_emissions(emissions): format a float as a kg CO₂ string.
- today_date(): ISO date string for today.
- percentage_change(old, new): percent change with safe handling of zero baseline.
- normalize_activity_name(name): normalize labels to canonical factor keys.
- friendly_message(emissions): quick status blurb by daily footprint size.
- safe_float(value, default): coerce any input to float with a default fallback.
"""

from __future__ import annotations

import datetime
from typing import Any


def format_emissions(emissions: float) -> str:
    """Format a number of kilograms CO₂ with 2 decimals and unit.

    Example: 12.345 -> "12.35 kg CO₂"
    """
    return f"{emissions:.2f} kg CO₂"


def today_date() -> str:
    """Return today's date as ISO string (YYYY-MM-DD)."""
    return datetime.date.today().isoformat()


def percentage_change(old: float, new: float) -> float:
    """Compute percentage change from old to new.

    If old == 0, returns 0 to avoid division by zero (interpreted as "no baseline").
    """
    if old == 0:
        return 0.0
    return round(((new - old) / old) * 100, 2)


def normalize_activity_name(name: str) -> str:
    """Normalize arbitrary activity labels to canonical factor keys.

    Operations:
    - Trim whitespace
    - Lowercase
    - Replace spaces, dashes, and slashes with underscores
    - Remove parentheses
    - Collapse multiple underscores

    Examples:
    - "Electricity (kWh)" -> "electricity_kwh"
    - "Flight short/km"   -> "flight_short_km"
    """
    s = name.strip().lower()
    # Remove parentheses
    s = s.replace("(", "").replace(")", "")
    # Unify common separators to underscores
    for ch in [" ", "-", "/", "\\"]:
        s = s.replace(ch, "_")
    # Collapse multiple underscores
    while "__" in s:
        s = s.replace("__", "_")
    return s


def friendly_message(emissions: float) -> str:
    """Return a short status message for a daily footprint value.

    Thresholds are intentionally simple and can be tuned later.
    """
    if emissions > 50:
        return "🚨 High footprint today! Try to reduce energy or transport use."
    elif emissions > 20:
        return "🌱 Moderate footprint. Small changes can make a big difference!"
    else:
        return "🌍 Low footprint today, great job!"


def safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort conversion to float with a default fallback.

    Examples:
    - safe_float("3.14") -> 3.14
    - safe_float(None)   -> 0.0 (default)
    - safe_float("abc", default=1.0) -> 1.0
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)