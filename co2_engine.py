"""
co2_engine.py

A small engine to estimate CO₂ emissions from daily activities.

- CO2_FACTORS contains emission factors (kg CO₂ per unit) for supported activities.
- calculate_co2(activity_data) returns the total emissions in kilograms of CO₂.
- calculate_co2_breakdown(activity_data) (optional) returns per-activity emissions
  for deeper insights and debugging.

Notes for readers:
- Keys in activity_data should match the factor keys (e.g., "electricity_kWh").
- We normalize input keys (lowercase, underscores) so "Electricity (kWh)" → "electricity_kWh" still matches.
- Factors are illustrative and can be adapted to local datasets (EPA/IPCC, supplier-specific, etc.).
"""

from typing import Dict, Mapping, Optional
from utils import normalize_activity_name

# Emission factors in kg CO₂ per unit.
# You can adjust these values based on your country/utility factors or published datasets.
CO2_FACTORS: Dict[str, float] = {
    # Energy
    "electricity_kwh": 0.233,        # per kWh
    "natural_gas_m3": 2.03,          # per cubic meter
    "hot_water_liter": 0.25,         # per liter (includes energy for heating water)
    "cold_water_liter": 0.075,       # per liter (pumping/treatment, if desired)
    "district_heating_kwh": 0.15,    # per kWh
    "propane_liter": 1.51,           # per liter
    "fuel_oil_liter": 2.52,          # per liter

    # Transport
    "petrol_liter": 0.235,           # per liter gasoline
    "diesel_liter": 0.268,           # per liter diesel
    "bus_km": 0.12,                  # per km
    "train_km": 0.14,                # per km (very rough average)
    "bicycle_km": 0.0,               # cycling assumed zero direct emissions
    "flight_short_km": 0.275,        # per km (short-haul average)
    "flight_long_km": 0.175,         # per km (long-haul average)

    # Meals (food mass consumed in kg)
    "meat_kg": 27.0,
    "chicken_kg": 6.9,
    "eggs_kg": 4.8,
    "dairy_kg": 13.0,
    "vegetarian_kg": 2.0,
    "vegan_kg": 1.5,
}


def _get_factor(activity_key: str) -> Optional[float]:
    """
    Return the emission factor for an activity, after normalizing its key.

    We accept flexible keys (e.g., "Electricity (kWh)") by normalizing them into
    the canonical format used by CO2_FACTORS.
    """
    normalized = normalize_activity_name(activity_key)
    return CO2_FACTORS.get(normalized)


def calculate_co2(activity_data: Mapping[str, float]) -> float:
    """
    Calculate total CO₂ emissions for a set of activities.

    Parameters
    - activity_data: mapping of activity key to amount used/done for the day.
      Example:
          {"electricity_kWh": 4.2, "bus_km": 12, "meat_kg": 0.15}

    Returns
    - Total emissions (kg CO₂) rounded to 2 decimals.

    Behavior
    - Non-numeric or negative amounts are ignored with a warning.
    - Unknown activity keys are ignored with a warning.
    """
    total_emissions = 0.0

    for activity, amount in activity_data.items():
        factor = _get_factor(activity)
        if factor is None:
            print(f"⚠️ Warning: '{activity}' not found in CO2_FACTORS")
            continue

        # Coerce amount to float and guard against negatives
        try:
            amt_val = float(amount)
        except (TypeError, ValueError):
            print(f"⚠️ Warning: amount for '{activity}' is not numeric; skipping.")
            continue

        if amt_val < 0:
            print(f"⚠️ Warning: negative amount for '{activity}' ({amt_val}); treating as 0.")
            amt_val = 0.0

        total_emissions += factor * amt_val

    return round(total_emissions, 2)


def calculate_co2_breakdown(activity_data: Mapping[str, float]) -> Dict[str, float]:
    """
    Return per-activity emissions (kg CO₂) for insight and debugging.

    Unknown or invalid entries are skipped.
    Keys are returned in their normalized form.
    """
    breakdown: Dict[str, float] = {}

    for activity, amount in activity_data.items():
        normalized = normalize_activity_name(activity)
        factor = CO2_FACTORS.get(normalized)
        if factor is None:
            continue

        try:
            amt_val = float(amount)
        except (TypeError, ValueError):
            continue

        if amt_val < 0:
            amt_val = 0.0

        kg = factor * amt_val
        if kg:
            # more precision here to help users debug contributions
            breakdown[normalized] = round(kg, 4)

    return breakdown