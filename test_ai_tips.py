import os
from types import SimpleNamespace
import pytest

import ai_tips
from openai import OpenAIError
from unittest.mock import patch


def test_no_api_key_falls_back(monkeypatch):
    # Ensure key is not present
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    tip = ai_tips.generate_eco_tip(
        {"electricity_kwh": 5, "bus_km": 10, "meat_kg": 0.2},
        emissions=15.0,
    )
    assert isinstance(tip, str)
    assert len(tip) > 0
    # Should not raise and should be a meaningful local tip


def test_openai_success_returns_gpt_tip(monkeypatch):
    # Provide a fake API key to exercise the GPT branch
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    # Mock the OpenAI client create() call to return a fixed message
    class FakeChoice:
        def __init__(self, content):
            self.message = SimpleNamespace(content=content)

    class FakeResponse:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    def fake_create(**kwargs):
        return FakeResponse("Use a smart power strip to reduce standby energy.")

    monkeypatch.setattr(
        ai_tips.client.chat.completions,
        "create",
        fake_create,
        raising=True,
    )

    tip = ai_tips.generate_eco_tip({"electricity_kwh": 5}, emissions=2.5)
    assert "smart power strip" in tip.lower()


def test_openai_error_falls_back_to_local(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    def fake_create(**kwargs):
        raise OpenAIError("quota exceeded")

    monkeypatch.setattr(
        ai_tips.client.chat.completions,
        "create",
        fake_create,
        raising=True,
    )

    # Make transport dominant to get a transport-focused local tip
    user_data = {"petrol_liter": 8, "electricity_kwh": 1}
    tip = ai_tips.generate_eco_tip(user_data, emissions=30.0)
    assert isinstance(tip, str)
    # local_tip mentions biggest source or a category; "petrol" likely included
    assert "petrol" in tip.lower() or "transport" in tip.lower()


def test_local_tip_picks_largest_emitter():
    # Directly test rules-based behavior
    tip = ai_tips.local_tip(
        {"meat_kg": 1.0, "electricity_kwh": 1.0, "bus_km": 1.0},
        emissions=40.0,
    )
    # Meat should dominate at 27 kg CO2/kg
    assert "meat" in tip.lower()
    # Tiered prefix for moderate/high footprint should appear
    assert ("🚨" in tip) or ("🌱" in tip) or ("🌍" in tip)


def test_local_tip_targets_electricity_when_dominant():
    # Ensure the targeted key mapping is correct for electricity_kwh
    user_data = {"electricity_kwh": 20.0, "bus_km": 1.0, "meat_kg": 0.05}
    tip = ai_tips.local_tip(user_data, emissions=10.0)
    # Should reference electricity explicitly after underscore replace
    assert "electricity kwh" in tip.lower()
    # And include a targeted electricity suggestion
    assert "standby" in tip.lower() or "led" in tip.lower() or "smart" in tip.lower()


def test_local_tip_targets_district_heating_when_dominant():
    # Ensure the targeted key mapping is correct for district_heating_kwh
    user_data = {"district_heating_kwh": 30.0, "electricity_kwh": 1.0, "bus_km": 0.0}
    tip = ai_tips.local_tip(user_data, emissions=20.0)
    assert "district heating kwh" in tip.lower()
    # Should include advice about thermostat/insulation per the targeted mapping
    assert "thermostat" in tip.lower() or "insulation" in tip.lower()


def test_gpt_tip_cached_for_repeated_inputs(monkeypatch):
    # Ensure GPT branch is used
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    # Clear cache to avoid cross-test effects
    ai_tips._generate_eco_tip_cached.cache_clear()

    call_count = {"n": 0}

    class FakeChoice:
        def __init__(self, content):
            self.message = SimpleNamespace(content=content)

    class FakeResponse:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    def fake_create(**kwargs):
        call_count["n"] += 1
        return FakeResponse("Cached eco tip")

    monkeypatch.setattr(
        ai_tips.client.chat.completions,
        "create",
        fake_create,
        raising=True,
    )

    user_data = {"electricity_kwh": 5, "bus_km": 10}
    tip1 = ai_tips.generate_eco_tip(user_data, emissions=12.0)
    tip2 = ai_tips.generate_eco_tip(user_data, emissions=12.0)

    assert tip1 == tip2
    # Because of caching, the OpenAI call should be made only once
    assert call_count["n"] == 1


def test_gpt_retry_then_success(monkeypatch):
    # Ensure GPT branch is used
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    ai_tips._generate_eco_tip_cached.cache_clear()

    attempts = {"n": 0}

    class FakeChoice:
        def __init__(self, content):
            self.message = SimpleNamespace(content=content)

    class FakeResponse:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    def fake_create(**kwargs):
        attempts["n"] += 1
        # Fail first two attempts, succeed on third
        if attempts["n"] < 3:
            raise OpenAIError("rate limit")
        return FakeResponse("Recovered after retries")

    # Speed up the test by skipping actual sleep
    monkeypatch.setattr(ai_tips.time, "sleep", lambda s: None, raising=True)

    monkeypatch.setattr(
        ai_tips.client.chat.completions,
        "create",
        fake_create,
        raising=True,
    )

    user_data = {"electricity_kwh": 2}
    tip = ai_tips.generate_eco_tip(user_data, emissions=1.0)
    assert "recovered" in tip.lower()
    # Should have attempted exactly 3 times before succeeding
    assert attempts["n"] == 3


def test_gpt_retry_then_fallback(monkeypatch):
    # Force GPT branch, but make it fail all retries then fall back to local tip
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    ai_tips._generate_eco_tip_cached.cache_clear()

    attempts = {"n": 0}

    def fake_create(**kwargs):
        attempts["n"] += 1
        # Always fail to trigger final fallback
        raise OpenAIError("rate limit")

    # Speed up: skip actual sleep
    monkeypatch.setattr(ai_tips.time, "sleep", lambda s: None, raising=True)

    monkeypatch.setattr(
        ai_tips.client.chat.completions,
        "create",
        fake_create,
        raising=True,
    )

    # Make transport dominant to assert we get a transport-focused local tip
    user_data = {"petrol_liter": 5.0, "bus_km": 2.0, "electricity_kwh": 0.5}
    tip = ai_tips.generate_eco_tip(user_data, emissions=12.0)

    # After exhausting retries (3), we should fall back to local tip
    assert attempts["n"] == 3
    assert isinstance(tip, str) and len(tip) > 0
    assert "transport" in tip.lower() or "petrol" in tip.lower()


def test_gpt_api_error_fallback(monkeypatch):
    # Force GPT branch and then force an API error to trigger fallback
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    ai_tips._generate_eco_tip_cached.cache_clear()

    user_data = {"electricity_kwh": 3, "bus_km": 2, "meat_kg": 0.5}
    emissions = 15.0

    with patch("ai_tips.client.chat.completions.create", side_effect=OpenAIError("Simulated error")):
        tip = ai_tips.generate_eco_tip(user_data, emissions)

    # Fallback should be used and cleaned
    expected = ai_tips.clean_tip(ai_tips.local_tip(user_data, emissions))
    assert tip == expected


def test_missing_input_edge_case(monkeypatch):
    # No API key present -> local fallback
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    user_data = {"electricity_kwh": 0, "bus_km": 0, "meat_kg": 0}
    emissions = 0.0

    tip = ai_tips.generate_eco_tip(user_data, emissions)
    assert isinstance(tip, str)
    assert len(tip.strip()) > 0