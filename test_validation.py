import pytest

import app  # import validation helpers from the Streamlit app module


def test_has_meaningful_input_true_when_any_gt_zero():
    user_data = {
        "electricity_kwh": 0.0,
        "bus_km": 2.0,
        "meat_kg": 0.0,
    }
    assert app.has_meaningful_input(user_data) is True


def test_has_meaningful_input_false_for_all_zero_or_none():
    user_data = {
        "electricity_kwh": 0.0,
        "bus_km": 0.0,
        "meat_kg": None,
    }
    assert app.has_meaningful_input(user_data) is False


def test_find_invalid_fields_negative_values():
    user_data = {
        "electricity_kwh": -1.0,
        "bus_km": 10.0,
        "meat_kg": -0.5,
    }
    invalid = app.find_invalid_fields(user_data)
    assert "electricity_kwh" in invalid
    assert "meat_kg" in invalid
    assert "bus_km" not in invalid


def test_find_invalid_fields_non_numeric_values():
    user_data = {
        "electricity_kwh": "abc",
        "bus_km": 5.0,
    }
    invalid = app.find_invalid_fields(user_data)
    assert "electricity_kwh" in invalid
    assert "bus_km" not in invalid


def test_combined_validations_edge_mix():
    user_data = {
        "electricity_kwh": 0.0,
        "bus_km": "",
        "meat_kg": -0.1,
        "train_km": 0.0,
    }
    assert app.has_meaningful_input(user_data) is False
    invalid = app.find_invalid_fields(user_data)
    assert set(invalid) == {"bus_km", "meat_kg"}
