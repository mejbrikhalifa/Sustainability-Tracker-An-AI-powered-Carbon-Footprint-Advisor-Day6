import pytest
from utils import normalize_activity_name, percentage_change, safe_float, format_emissions

def test_normalize_activity_name_various_separators():
    assert normalize_activity_name("Electricity (kWh)") == "electricity_kwh"
    assert normalize_activity_name("Flight short/km") == "flight_short_km"
    assert normalize_activity_name("cold-water") == "cold_water"
    assert normalize_activity_name("  Train  km ") == "train_km"

def test_percentage_change_basic_and_zero():
    assert percentage_change(100, 110) == 10.0
    assert percentage_change(50, 25) == -50.0
    assert percentage_change(0, 100) == 0.0

def test_safe_float_happy_and_errors():
    assert safe_float("3.14") == pytest.approx(3.14)
    assert safe_float(None) == 0.0
    assert safe_float("abc", default=1.0) == 1.0

def test_format_emissions():
    assert format_emissions(12.345) == "12.35 kg CO₂"