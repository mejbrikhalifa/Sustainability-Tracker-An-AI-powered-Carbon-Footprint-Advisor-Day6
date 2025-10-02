import math
import pytest

from co2_engine import calculate_co2, calculate_co2_breakdown, CO2_FACTORS

def test_calculate_co2_basic_sum():
    user_data = {
        "electricity_kwh": 10,   # 10 * 0.233 = 2.33
        "bus_km": 15,            # 15 * 0.12  = 1.80
        "meat_kg": 0.2,          # 0.2 * 27.0 = 5.40
    }
    total = calculate_co2(user_data)
    assert math.isclose(total, 2.33 + 1.8 + 5.4, rel_tol=1e-6)

def test_calculate_co2_unknown_and_non_numeric_are_ignored(capfd):
    user_data = {
        "UNKNOWN_ACTIVITY": 5,
        "electricity_kwh": "abc",     # non-numeric -> ignored
        "meat_kg": -1,                # negative -> treated as 0
        "bus_km": 10,
    }
    total = calculate_co2(user_data)
    # Only bus_km should contribute: 10 * 0.12 = 1.2
    assert math.isclose(total, 1.2, rel_tol=1e-6)

    # Ensure we print warnings (not strictly required but good for visibility)
    out, _ = capfd.readouterr()
    assert "not found in CO2_FACTORS" in out
    assert "not numeric" in out or "negative amount" in out

def test_calculate_co2_breakdown_sorted_keys():
    user_data = {"electricity_kwh": 4, "bus_km": 5}
    breakdown = calculate_co2_breakdown(user_data)
    # 4 * 0.233 = 0.932 ; 5 * 0.12 = 0.6
    assert breakdown["electricity_kwh"] == pytest.approx(0.932, rel=1e-6)
    assert breakdown["bus_km"] == pytest.approx(0.6, rel=1e-6)

def test_calculate_co2_breakdown_handles_weird_keys():
    user_data = {"Electricity (kWh)": 2, "Bus (km)": 10}
    breakdown = calculate_co2_breakdown(user_data)
    # Normalization should map to the canonical keys
    assert "electricity_kwh" in {k.lower() for k in breakdown.keys()}
    # The exact key returned is normalized by co2_engine; ensure correct total
    total = calculate_co2(user_data)
    assert total == pytest.approx(round(2 * CO2_FACTORS["electricity_kwh"] + 10 * CO2_FACTORS["bus_km"], 2), rel=1e-6)


# -----------------------------
# Manual runner for python file execution
# -----------------------------
if __name__ == "__main__":
    print("Running CO2 engine tests manually...\n")
    test_calculate_co2_basic_sum()
    print("✅ test_calculate_co2_basic_sum passed")

    test_calculate_co2_unknown_and_non_numeric_are_ignored()
    print("✅ test_calculate_co2_unknown_and_non_numeric_are_ignored passed")

    test_calculate_co2_breakdown_sorted_keys()
    print("✅ test_calculate_co2_breakdown_sorted_keys passed")

    test_calculate_co2_breakdown_handles_weird_keys()
    print("✅ test_calculate_co2_breakdown_handles_weird_keys passed")

    print("\nAll tests passed! 🎉")
