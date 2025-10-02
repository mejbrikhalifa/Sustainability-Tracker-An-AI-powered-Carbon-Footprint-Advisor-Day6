import app


def test_should_generate_tip_true_when_valid_and_meaningful():
    user_data = {
        "electricity_kwh": 1.0,
        "bus_km": 0.0,
        "meat_kg": 0.0,
    }
    assert app.should_generate_tip(user_data) is True


def test_should_generate_tip_false_when_all_zero():
    user_data = {
        "electricity_kwh": 0.0,
        "bus_km": 0.0,
        "meat_kg": 0.0,
    }
    assert app.should_generate_tip(user_data) is False


def test_should_generate_tip_false_when_invalid_numeric():
    user_data = {
        "electricity_kwh": -0.1,
        "bus_km": 2.0,
    }
    assert app.should_generate_tip(user_data) is False


def test_should_generate_tip_false_when_non_numeric():
    user_data = {
        "electricity_kwh": "abc",
        "bus_km": 2.0,
    }
    assert app.should_generate_tip(user_data) is False
