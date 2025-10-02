import math
import datetime as dt
import pandas as pd
import pytest

import app
from co2_engine import CO2_FACTORS


def test_category_map_keys_exist_in_factors():
    """
    Ensures every key in app.CATEGORY_MAP exists in co2_engine.CO2_FACTORS.
    This will fail if there is a mismatch like electricity_kWh vs electricity_kwh.
    """
    missing = []
    for cat, keys in app.CATEGORY_MAP.items():
        for k in keys:
            if k not in CO2_FACTORS:
                missing.append((cat, k))
    assert not missing, f"Missing keys in CO2_FACTORS: {missing}"


def test_compute_category_emissions_aggregates():
    """
    Validates compute_category_emissions() math by comparing totals
    against the sum of amount * factor for one sample in each category.
    """
    # Build a sample aligned to CO2_FACTORS naming (lowercase *_kwh)
    user_data = {
        "electricity_kwh": 10,   # 10 * 0.233 = 2.33
        "bus_km": 15,            # 15 * 0.12  = 1.80
        "meat_kg": 0.2,          # 0.2 * 27.0 = 5.40
    }
    cat = app.compute_category_emissions(user_data)

    # Compute expected by category
    energy_expected = user_data["electricity_kwh"] * CO2_FACTORS["electricity_kwh"]
    transport_expected = user_data["bus_km"] * CO2_FACTORS["bus_km"]
    meals_expected = user_data["meat_kg"] * CO2_FACTORS["meat_kg"]

    # compare with rounding used in function
    assert math.isclose(cat.get("Energy", 0), round(energy_expected, 2))
    assert math.isclose(cat.get("Transport", 0), round(transport_expected, 2))
    assert math.isclose(cat.get("Meals", 0), round(meals_expected, 2))


def test_save_and_load_history(tmp_path, monkeypatch):
    """
    Smoke test the CSV persistence: save one entry and ensure we can read it back and it is sorted.
    """
    tmp_csv = tmp_path / "history.csv"
    monkeypatch.setattr(app, "HISTORY_FILE", str(tmp_csv))

    date_val = dt.date(2025, 1, 2)
    user_data = {
        "electricity_kwh": 5.0,
        "bus_km": 10.0,
        "meat_kg": 0.1,
    }
    total = user_data["electricity_kwh"] * CO2_FACTORS["electricity_kwh"] \
        + user_data["bus_km"] * CO2_FACTORS["bus_km"] \
        + user_data["meat_kg"] * CO2_FACTORS["meat_kg"]
    total = round(total, 2)

    app.save_entry(date_val, user_data, total)

    df = app.load_history()
    assert not df.empty
    assert "total_kg" in df.columns
    assert pd.to_datetime(date_val) in set(df["date"])
    row = df[df["date"].dt.date == date_val]
    assert not row.empty
    assert float(row["total_kg"].iloc[0]) == total


def test_award_badges_logic():
    """
    Basic checks for badges based on streak and total.
    """
    # Build a history df with a 3-day streak ending today
    today = dt.date(2025, 1, 3)
    dates = pd.to_datetime([dt.date(2025, 1, 1), dt.date(2025, 1, 2), today])
    df = pd.DataFrame({"date": dates, "total_kg": [30.0, 25.0, 18.0]})

    streak = app.compute_streak(df, today)
    badges = app.award_badges(today_total=18.0, streak=streak, df=df)

    # Should include consistency & low impact & 3-day streak
    assert any("Consistency" in b for b in badges)
    assert any("Low Impact" in b for b in badges)
    assert any("3-Day Streak" in b for b in badges)