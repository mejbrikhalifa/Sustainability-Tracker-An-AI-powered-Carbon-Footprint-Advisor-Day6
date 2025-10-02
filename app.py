# app.py
import os
import pandas as pd
import datetime as dt
import streamlit as st
import io
from co2_engine import calculate_co2, CO2_FACTORS, calculate_co2_breakdown
from utils import (
    format_emissions as fmt_emissions,
    friendly_message as status_message,
    percentage_change,
)
from ai_tips import (
    generate_tip,
    LAST_TIP_SOURCE,
    generate_ai_summary,
    generate_eco_tip_with_prompt,
    classify_input_type,
    set_extreme_thresholds,
)
import time
import concurrent.futures
import csv
import json
import random
import zipfile
import altair as alt
try:
    from altair_saver import save as alt_save
except Exception:
    alt_save = None
import tempfile

# Set page config first (must be the first Streamlit command)
st.set_page_config(page_title="Sustainability Tracker", page_icon="🌍", layout="wide")

# =========================
# Category Mapping & Storage
# =========================
CATEGORY_MAP = {
    "Energy": [
        "electricity_kwh",
        "natural_gas_m3",
        "hot_water_liter",
        "cold_water_liter",
        "district_heating_kwh",
        "propane_liter",
        "fuel_oil_liter",
    ],
    "Transport": [
        "petrol_liter",
        "diesel_liter",
        "bus_km",
        "train_km",
        "bicycle_km",
        "flight_short_km",
        "flight_long_km",
    ],
    "Meals": [
        "meat_kg",
        "chicken_kg",
        "eggs_kg",
        "dairy_kg",
        "vegetarian_kg",
        "vegan_kg",
    ],
}
ALL_KEYS = [k for keys in CATEGORY_MAP.values() for k in keys]

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.csv")
PREFS_PATH = os.path.join(os.path.dirname(__file__), ".user_prefs.json")


# =========================
# Preferences (persisted across sessions)
# =========================
def load_user_prefs() -> dict:
    try:
        if os.path.exists(PREFS_PATH):
            with open(PREFS_PATH, "r", encoding="utf-8") as f:
                prefs = json.load(f) or {}
                # Surface a small toast/banner once per session to show prefs applied
                if not st.session_state.get("_prefs_loaded_banner_shown"):
                    st.toast("Loaded saved preferences from .user_prefs.json", icon="✅")
                    st.session_state["_prefs_loaded_banner_shown"] = True
                return prefs
    except Exception:
        pass
    return {}


def save_user_prefs(prefs: dict):
    try:
        with open(PREFS_PATH, "w", encoding="utf-8") as f:
            json.dump(prefs or {}, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def get_pref(key: str, default=None):
    prefs = st.session_state.get("user_prefs")
    if prefs is None:
        prefs = load_user_prefs()
        st.session_state["user_prefs"] = prefs
    return prefs.get(key, default)


def set_pref(key: str, value):
    prefs = st.session_state.get("user_prefs")
    if prefs is None:
        prefs = load_user_prefs()
        st.session_state["user_prefs"] = prefs
    prefs[key] = value
    save_user_prefs(prefs)


# =========================
# Helper Functions
# =========================
def compute_category_emissions(activity_data: dict) -> dict:
    result = {}
    for cat, keys in CATEGORY_MAP.items():
        subtotal = 0.0
        for k in keys:
            amt = float(activity_data.get(k, 0) or 0)
            factor = CO2_FACTORS.get(k)
            if factor is not None:
                subtotal += amt * factor
        result[cat] = round(subtotal, 2)
    return result


def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_entry(date_val: dt.date, activity_data: dict, total: float):
    df = load_history()
    row = {"date": pd.to_datetime(date_val)}
    for k in ALL_KEYS:
        row[k] = float(activity_data.get(k, 0) or 0)
    row["total_kg"] = float(total)

    if df.empty:
        df = pd.DataFrame([row])
    else:
        mask = df["date"].dt.date == date_val
        if mask.any():
            # Upsert
            df.loc[mask, list(row.keys())] = list(row.values())
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df = df.sort_values("date")
    df.to_csv(HISTORY_FILE, index=False)


def get_yesterday_total(df: pd.DataFrame, date_val: dt.date) -> float:
    if df.empty:
        return 0.0
    yesterday = pd.to_datetime(date_val) - pd.Timedelta(days=1)
    mask = df["date"].dt.date == yesterday.date()
    if mask.any():
        return float(df.loc[mask, "total_kg"].iloc[0])
    return 0.0


def compute_streak(df: pd.DataFrame, date_val: dt.date) -> int:
    """Compute the current streak of consecutive days up to date_val."""
    if df.empty:
        return 0

    # Ensure all dates are datetime.date
    df_dates = df["date"].dt.date if pd.api.types.is_datetime64_any_dtype(df["date"]) else df["date"]
    dayset = set(df_dates)

    streak = 0
    current = date_val
    while current in dayset:
        streak += 1
        current -= dt.timedelta(days=1)

    return streak


def award_badges(today_total: float, streak: int, df: pd.DataFrame) -> list:
    badges = []
    if not df.empty:
        badges.append("📅 Consistency: Entries logged!")
    if today_total < 20:
        badges.append("🌿 Low Impact Day (< 20 kg)")
    if streak >= 3:
        badges.append("🔥 3-Day Streak")
    if streak >= 7:
        badges.append("🏆 7-Day Streak")
    if not df.empty:
        recent = df.tail(7)
        avg7 = float(recent["total_kg"].mean()) if not recent.empty else 0.0
        if avg7 and today_total < 0.9 * avg7:
            badges.append("📈 10% Better than 7-day avg")
    return badges


# =========================
# Helper formatters
# =========================
def format_summary(user_data: dict) -> str:
    """Return a compact, human-friendly summary of today's inputs.
    Only include fields that are present and > 0 where numeric.
    """
    parts: list[str] = []
    def _num(v):
        try:
            return float(v)
        except Exception:
            return v

    # Transport
    if (val := _num(user_data.get("petrol_liter"))) and isinstance(val, float) and val > 0:
        parts.append(f"🚗 Petrol: {val:.1f} L")
    if (val := _num(user_data.get("diesel_liter"))) and isinstance(val, float) and val > 0:
        parts.append(f"🚙 Diesel: {val:.1f} L")
    if (val := _num(user_data.get("bus_km"))) and isinstance(val, float) and val > 0:
        parts.append(f"🚌 Bus: {val:.0f} km")
    if (val := _num(user_data.get("train_km"))) and isinstance(val, float) and val > 0:
        parts.append(f"🚆 Train: {val:.0f} km")
    if (val := _num(user_data.get("bicycle_km"))) and isinstance(val, float) and val > 0:
        parts.append(f"🚴 Bike: {val:.0f} km")

    # Energy
    if (val := _num(user_data.get("electricity_kwh"))) and isinstance(val, float) and val > 0:
        parts.append(f"⚡ Electricity: {val:.1f} kWh")
    if (val := _num(user_data.get("district_heating_kwh"))) and isinstance(val, float) and val > 0:
        parts.append(f"🔥 District heat: {val:.1f} kWh")
    if (val := _num(user_data.get("natural_gas_m3"))) and isinstance(val, float) and val > 0:
        parts.append(f"🏠 Gas: {val:.1f} m³")
    if (val := _num(user_data.get("hot_water_liter"))) and isinstance(val, float) and val > 0:
        parts.append(f"🚿 Hot water: {val:.0f} L")

    # Meals
    meal_bits = []
    for key, label in [
        ("meat_kg", "🥩 Meat"),
        ("chicken_kg", "🍗 Chicken"),
        ("dairy_kg", "🥛 Dairy"),
        ("eggs_kg", "🥚 Eggs"),
        ("vegetarian_kg", "🥗 Veg"),
        ("vegan_kg", "🌱 Vegan"),
    ]:
        val = _num(user_data.get(key))
        if isinstance(val, float) and val > 0:
            meal_bits.append(f"{label}: {val:.2f} kg")
    if meal_bits:
        parts.append(" | ".join(meal_bits))

    return " | ".join(parts) if parts else "No activities logged yet."


def format_summary_html(user_data: dict) -> str:
    """Return an HTML-formatted summary with colored tags. Safe for st.markdown(..., unsafe_allow_html=True)."""
    # Color groups
    def tag(label: str, color: str) -> str:
        return (
            f"<span style='display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;"
            f"border-radius:12px;background:{color};color:#112;border:1px solid rgba(0,0,0,0.1);font-size:0.92em;'>"
            f"{label}</span>"
        )
    html_parts: list[str] = []
    # Transport (green-ish)
    trans_color = "#e6f4ea"  # light green
    for key, icon, unit, fmt in [
        ("petrol_liter", "🚗 Petrol", "L", "{:.1f}"),
        ("diesel_liter", "🚙 Diesel", "L", "{:.1f}"),
        ("bus_km", "🚌 Bus", "km", "{:.0f}"),
        ("train_km", "🚆 Train", "km", "{:.0f}"),
        ("bicycle_km", "🚴 Bike", "km", "{:.0f}"),
    ]:
        val = user_data.get(key)
        try:
            fv = float(val)
        except Exception:
            fv = 0.0
        if fv > 0:
            html_parts.append(tag(f"{icon}: {fmt.format(fv)} {unit}", trans_color))

    # Energy (blue-ish)
    energy_color = "#e8f0fe"  # light blue
    for key, icon, unit, fmt in [
        ("electricity_kwh", "⚡ Electricity", "kWh", "{:.1f}"),
        ("district_heating_kwh", "🔥 District heat", "kWh", "{:.1f}"),
        ("natural_gas_m3", "🏠 Gas", "m³", "{:.1f}"),
        ("hot_water_liter", "🚿 Hot water", "L", "{:.0f}"),
    ]:
        val = user_data.get(key)
        try:
            fv = float(val)
        except Exception:
            fv = 0.0
        if fv > 0:
            html_parts.append(tag(f"{icon}: {fmt.format(fv)} {unit}", energy_color))

    # Meals (orange-ish)
    meal_color = "#fff4e5"  # light orange
    for key, icon in [
        ("meat_kg", "🥩 Meat"),
        ("chicken_kg", "🍗 Chicken"),
        ("dairy_kg", "🥛 Dairy"),
        ("eggs_kg", "🥚 Eggs"),
        ("vegetarian_kg", "🥗 Veg"),
        ("vegan_kg", "🌱 Vegan"),
    ]:
        val = user_data.get(key)
        try:
            fv = float(val)
        except Exception:
            fv = 0.0
        if fv > 0:
            html_parts.append(tag(f"{icon}: {fv:.2f} kg", meal_color))

    if not html_parts:
        return "<em>No activities logged yet.</em>"
    return "\n".join(html_parts)


def dominant_category_icon(user_data: dict) -> tuple[str, str]:
    """Return (icon, category_label) for the dominant emitting category.
    Defaults to neutral if nothing is logged.
    """
    try:
        cat = compute_category_emissions(user_data)
    except Exception:
        cat = {}
    if not cat:
        return ("💡", "Tip")
    dom = max(cat.items(), key=lambda x: x[1])[0]
    icon_map = {"Energy": "⚡", "Transport": "🚗", "Meals": "🥗"}
    return (icon_map.get(dom, "💡"), dom)


# =========================
# Validation helpers
# =========================
def _coerce_float(v):
    try:
        return float(v)
    except Exception:
        return None

def has_meaningful_input(user_data: dict) -> bool:
    """True if at least one numeric input is > 0."""
    for v in user_data.values():
        fv = _coerce_float(v)
        if fv is not None and fv > 0:
            return True
    return False

def find_invalid_fields(user_data: dict) -> list[str]:
    """Return keys that are negative or non-numeric when a number is expected."""
    bad = []
    for k, v in user_data.items():
        fv = _coerce_float(v)
        if fv is None:
            bad.append(k)
        elif fv < 0:
            bad.append(k)
    return bad

def show_input_warnings(user_data: dict):
    """Render inline warnings grouped by category for any invalid fields.
    This is shown immediately after inputs so users can correct quickly.
    """
    invalid = find_invalid_fields(user_data)
    if not invalid:
        return
    # Group invalid keys by category using CATEGORY_MAP
    grouped = {cat: [] for cat in CATEGORY_MAP.keys()}
    for cat, keys in CATEGORY_MAP.items():
        grouped[cat] = [k for k in invalid if k in keys]
    # Render per-category messages if any
    has_any = any(grouped[cat] for cat in grouped)
    if has_any:
        st.markdown("<div style='color:#b00020;font-weight:600;'>Input issues detected:</div>", unsafe_allow_html=True)
        for cat in ["Energy", "Transport", "Meals"]:
            if grouped.get(cat):
                issues = ", ".join(grouped[cat])
                st.markdown(f"- <span style='color:#b00020;'>[{cat}] Invalid: {issues}</span>", unsafe_allow_html=True)

def should_generate_tip(user_data: dict) -> bool:
    """Pure decision helper: return True if inputs are valid and meaningful.
    This is used by the UI layer and covered by unit tests.
    """
    invalid = find_invalid_fields(user_data)
    if invalid:
        return False
    return has_meaningful_input(user_data)

# =========================
# Streamlit App
# =========================
def main():
    # Density + header
    # Initialize persisted UI density in session state
    if "density" not in st.session_state:
        st.session_state["density"] = "Compact"

    # Read density from URL query params if present (new API)
    try:
        qp_density = st.query_params.get("density")
        if qp_density in ("Compact", "Comfy") and qp_density != st.session_state["density"]:
            st.session_state["_pending_density"] = qp_density
    except Exception:
        pass

    # Density toggle: Compact vs Comfy
    def _apply_pending_density_if_any():
        """Apply queued density (Compact/Comfy) before the density radio is instantiated."""
        pdn = st.session_state.get("_pending_density")
        if isinstance(pdn, str) and pdn:
            st.session_state["density"] = pdn
            st.session_state.pop("_pending_density", None)

    def _apply_pending_demo_toggle_if_any():
        """Apply a queued request to turn demo_mode off before the checkbox exists."""
        if st.session_state.get("_pending_demo_off"):
            # Safely set the widget-bound key before checkbox is created in this run
            st.session_state["demo_mode"] = False
            st.session_state.pop("_pending_demo_off", None)

    _apply_pending_density_if_any()
    _apply_pending_demo_toggle_if_any()

    dens_col1, dens_col2 = st.columns([3, 1])
    with dens_col1:
        st.title("Sustainability Tracker 🌍")
        st.caption("Track daily CO₂ emissions and get actionable tips")
    with dens_col2:
        _apply_pending_density_if_any()
        # Persist density across sessions using user prefs
        default_density = get_pref("ui_density", st.session_state.get("density", "Compact"))
        st.session_state["density"] = default_density
        dens_choice = st.radio(
            "Density",
            ["Compact", "Comfy"],
            index=(0 if default_density == "Compact" else 1),
            horizontal=True,
            key="density",
        )
        if dens_choice != default_density:
            set_pref("ui_density", dens_choice)
        # Quick settings: reset to defaults
        if st.button("↩️ Reset preferences to defaults"):
            try:
                if os.path.exists(PREFS_PATH):
                    os.remove(PREFS_PATH)
                # Clear cached prefs and some session flags
                st.session_state.pop("user_prefs", None)
                # Also clear remembered UI toggles so defaults take effect
                for k in [
                    "prompt_len_chart_type", "prompt_time_bin", "pdf_auto_clear_experiments",
                    "pdf_title", "pdf_primary_color", "pdf_text_color", "pdf_chart_bg",
                    "pdf_side_margin", "pdf_top_margin", "pdf_bottom_margin",
                    "pdf_include_pie", "pdf_include_spark", "pdf_include_ai_summary",
                    "pdf_include_prompt_appendix", "pdf_include_footer", "pdf_footer_text",
                    "prompt_mode", "prompt_category_opt", "density",
                    "exp_ai_summary_open_toggle", "exp_prompt_experiments_open_toggle",
                    "exp_prompt_log_open_toggle", "exp_pdf_options_open_toggle",
                ]:
                    st.session_state.pop(k, None)
                st.success("Preferences reset. Reload the page to see defaults applied.")
            except Exception as e:
                st.error(f"Could not reset preferences: {e}")
        # Help popover with a short FAQ
        # Help popover/expander with a short FAQ
        HelpContainer = st.popover if hasattr(st, "popover") else st.expander
        with HelpContainer("Help"):
            st.markdown(
                """
                - **How are emissions calculated?** Using standard factors per activity (kg CO₂ per unit).
                - **Why is bicycle 0?** Cycling has negligible direct CO₂ emissions in this model.
                - **How do I save/export?** Click "Calculate & Save" then download the CSV in Dashboard.
                - **Tips to reduce CO₂?** See the Eco tip card and focus on your biggest source first.
                
                <br/>
                <a href="#secrets" style="text-decoration:none;">
                  <span style="display:inline-block;padding:2px 8px;border-radius:12px;background:#eef;border:1px solid #ccd;color:#223;">🔐 Secrets (README)</span>
                </a>
                <div style="font-size:0.9em;color:#555;">Configure your OPENAI_API_KEY via <code>.env</code>. See README → Secrets.</div>
                """,
                unsafe_allow_html=True,
            )
        # Demo mode: force Compact, load demo values, auto-generate tip
        demo_mode = st.checkbox(
            "Demo mode",
            value=st.session_state.get("demo_mode", False),
            help="Force Compact density, load demo values, and auto-generate a tip.",
            key="demo_mode",
        )
        # Subtle status line about snapshot (for demo debugging)
        if demo_mode:
            _snap = st.session_state.get("demo_snapshot")
            if isinstance(_snap, dict) and _snap.get("ts"):
                st.caption(f"Demo snapshot captured at {_snap['ts']}")
                with st.popover("View snapshot detail"):
                    st.caption(f"Density before demo: {_snap.get('density', 'Comfy')}")
                    inputs = _snap.get("inputs", {})
                    if inputs:
                        st.json(inputs)
                    else:
                        st.write("No inputs captured in snapshot.")
            else:
                st.caption("Demo snapshot: none yet")
        if demo_mode and not st.session_state.get("demo_mode_applied", False):
            # Snapshot current density and inputs to allow restore on exit
            input_keys = [
                # Energy
                "electricity_kwh",
                "natural_gas_m3",
                "hot_water_liter",
                "cold_water_liter",
                "district_heating_kwh",
                "propane_liter",
                "fuel_oil_liter",
                # Transport
                "bus_km",
                "train_km",
                "bicycle_km",
                "petrol_liter",
                "diesel_liter",
                "flight_short_km",
                "flight_long_km",
                # Meals
                "meat_kg",
                "chicken_kg",
                "eggs_kg",
                "dairy_kg",
                "vegetarian_kg",
                "vegan_kg",
            ]
            st.session_state["demo_snapshot"] = {
                "density": st.session_state.get("density", "Comfy"),
                "inputs": {f"in_{k}": st.session_state.get(f"in_{k}", 0.0) for k in input_keys},
                "ts": dt.datetime.now().isoformat(),
            }
            # Queue density to Compact
            st.session_state["_pending_density"] = "Compact"
            # Load representative demo values
            demo_vals = {
                # Energy
                "electricity_kwh": 6.0,
                "natural_gas_m3": 1.2,
                "hot_water_liter": 60,
                # Transport
                "bus_km": 10,
                "train_km": 0,
                "petrol_liter": 2.5,
                # Meals
                "meat_kg": 0.15,
                "dairy_kg": 0.3,
                "vegetarian_kg": 0.2,
            }
            st.session_state["_pending_values"] = demo_vals
            # Auto-generate in Eco Tips on next run
            st.session_state["tips_autogen"] = True
            st.session_state["demo_mode_applied"] = True
            try:
                st.rerun()
            except Exception:
                pass
        # Exit Demo Mode helper: resets inputs and layout back to defaults
        if demo_mode:
            demo_keys = [
                # Energy
                "electricity_kwh",
                "natural_gas_m3",
                "hot_water_liter",
                "cold_water_liter",
                "district_heating_kwh",
                "propane_liter",
                "fuel_oil_liter",
                # Transport
                "bus_km",
                "train_km",
                "bicycle_km",
                "petrol_liter",
                "diesel_liter",
                "flight_short_km",
                "flight_long_km",
                # Meals
                "meat_kg",
                "eggs_kg",
                "dairy_kg",
                "vegetarian_kg",
                "chicken_kg",
                "vegan_kg",
            ]
            if st.button("Exit Demo Mode"):
                # Restore from snapshot if available; otherwise clear to zeros and comfy
                snap = st.session_state.get("demo_snapshot")
                if snap and isinstance(snap, dict):
                    # Convert stored 'in_*' keys back to canonical field keys
                    restored = {}
                    for key, val in snap.get("inputs", {}).items():
                        if key.startswith("in_"):
                            restored[key[3:]] = val
                    st.session_state["_pending_values"] = restored
                    st.session_state["_pending_density"] = snap.get("density", "Comfy")
                else:
                    st.session_state["_pending_values"] = {k: 0.0 for k in demo_keys}
                    st.session_state["_pending_density"] = "Comfy"
                st.session_state["tips_autogen"] = False
                st.session_state["demo_mode_applied"] = False
                # Do NOT set the widget key directly here; queue an off toggle instead
                st.session_state["_pending_demo_off"] = True
                st.session_state.pop("demo_snapshot", None)
                try:
                    st.rerun()
                except Exception:
                    pass

        # Hidden debug controls
        with st.expander("Debug (performance)", expanded=False):
            default_th = st.session_state.get("spinner_threshold", 0.3)
            th = st.slider("Spinner threshold (seconds)", 0.0, 2.0, float(default_th), 0.05)
            st.session_state["spinner_threshold"] = float(th)
            st.checkbox(
                "Enable performance logging (perf_log.csv)",
                value=st.session_state.get("perf_logging", False),
                key="perf_logging",
                help="Append eco-tip generation timings to perf_log.csv",
            )
            st.markdown(
                """
                <a href="#secrets" style="text-decoration:none;">
                  <span style="display:inline-block;padding:2px 8px;border-radius:12px;background:#eef;border:1px solid #ccd;color:#223;">🔐 Secrets (README)</span>
                </a>
                <div style="font-size:0.9em;color:#555;">Configure your OPENAI_API_KEY via <code>.env</code>. See README → Secrets.</div>
                """,
                unsafe_allow_html=True,
            )
        # Copy shareable link button (copies current URL with density param)
        st.markdown(
            """
            <button id=\"copy-link-btn\" style=\"margin-top:0.25rem;\">Copy shareable link</button>
            <script>
            const btn = document.getElementById('copy-link-btn');
            if (btn) {
              btn.addEventListener('click', async () => {
                try {
                  await navigator.clipboard.writeText(window.location.href);
                  const old = btn.textContent;
                  btn.textContent = 'Copied!';
                  setTimeout(() => { btn.textContent = old; }, 1500);
                } catch (e) {
                  btn.textContent = 'Copy failed';
                  setTimeout(() => { btn.textContent = 'Copy shareable link'; }, 1500);
                }
              });
            }
            </script>
            """,
            unsafe_allow_html=True,
        )
        # Reset layout button: revert to Compact density and update URL
        if st.button("Reset layout", type="secondary"):
            st.session_state["_pending_density"] = "Compact"
            try:
                st.query_params["density"] = "Compact"
            except Exception:
                pass
            st.success("Layout reset to Compact. Collapse expanders for best PDF.")
            try:
                st.rerun()
            except Exception:
                pass
        # Clear inputs button: zero all input fields
        if st.button("Clear inputs", help="Reset all fields to zero for today’s entry."):
            try:
                for _k in ALL_KEYS:
                    _sk = f"in_{_k}"
                    if _sk in st.session_state:
                        st.session_state[_sk] = 0.0
            except Exception:
                pass
            st.success("Inputs cleared.")
        # Demo and preset fillers
        with st.popover("Prefill demos/presets"):
            st.markdown("Pick a scenario to quickly populate inputs for demos.")
            c_demo, c_p1, c_p2 = st.columns(3)
            def _apply_values(vals: dict):
                # Queue values to apply before widgets are instantiated, then rerun
                st.session_state["_pending_values"] = {k: float(v) for k, v in vals.items()}
                try:
                    st.rerun()
                except Exception:
                    pass
            with c_demo:
                if st.button("Demo values"):
                    _apply_values({
                        # Energy
                        "electricity_kwh": 8,
                        "natural_gas_m3": 1.2,
                        "hot_water_liter": 60,
                        # Transport
                        "bus_km": 10,
                        "train_km": 0,
                        "petrol_liter": 2.5,
                        # Meals
                        "meat_kg": 0.15,
                        "dairy_kg": 0.3,
                        "vegetarian_kg": 0.2,
                    })
            with c_p1:
                if st.button("No car day"):
                    _apply_values({
                        "petrol_liter": 0,
                        "diesel_liter": 0,
                        "bus_km": 12,
                        "train_km": 6,
                        "bicycle_km": 5,
                    })
            with c_p2:
                if st.button("Vegetarian day"):
                    _apply_values({
                        "meat_kg": 0,
                        "chicken_kg": 0,
                        "vegetarian_kg": 0.6,
                        "vegan_kg": 0.2,
                        "dairy_kg": 0.25,
                    })
            c_p3, _, _ = st.columns(3)
            with c_p3:
                if st.button("Business trip"):
                    _apply_values({
                        "flight_short_km": 600,
                        "train_km": 20,
                        "electricity_kwh": 6,
                        "meat_kg": 0.25,
                    })

    # IMPORTANT: assign density BEFORE using it below
    density = st.session_state["density"]

    # Update URL query param to reflect current density (new API)
    try:
        st.query_params["density"] = density
    except Exception:
        pass

    # Heights and paddings based on density
    if density == "Compact":
        pad_top, pad_bottom = "1rem", "1rem"
        table_height = 150
        trend_height = 180
        bar_height = 180
        per_activity_height = 260
        expander_default = False
    else:
        pad_top, pad_bottom = "2rem", "2rem"
        table_height = 220
        trend_height = 260
        bar_height = 260
        per_activity_height = 360
        expander_default = True

    # Hide Streamlit default menu, footer, and header for cleaner PDF export
    st.markdown(
        f"""
        <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        .block-container {{padding-top: {pad_top}; padding-bottom: {pad_bottom};}}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top row: date and action area
    top_c1, top_c2 = st.columns([1, 2])
    with top_c1:
        selected_date = st.date_input("Date", value=dt.date.today())
    with top_c2:
        st.write("")

    def _apply_pending_values_if_any():
        """Apply any queued preset/demo values before widgets are created."""
        pending = st.session_state.get("_pending_values")
        if isinstance(pending, dict) and pending:
            for k, v in pending.items():
                st.session_state[f"in_{k}"] = float(v)
            st.session_state.pop("_pending_values", None)

    _apply_pending_values_if_any()

    with st.form("daily_input"):
        # Inputs grouped in compact expanders (density controlled)
        with st.expander("Energy inputs", expanded=expander_default):
            e1, e2, e3 = st.columns(3)
            with e1:
                electricity = st.number_input("Electricity (kWh)", value=0.0, min_value=0.0, step=0.1, key="in_electricity_kwh", help="Enter a number ≥ 0")
                _e = _coerce_float(electricity)
                if _e is None or (_e is not None and _e < 0):
                    st.markdown("<div style='color:#b00020;font-size:0.9em;'>Enter a number ≥ 0</div>", unsafe_allow_html=True)
                natural_gas = st.number_input("Natural Gas (m³)", value=0.0, min_value=0.0, step=0.1, key="in_natural_gas_m3", help="Enter a number ≥ 0")
            with e2:
                hot_water = st.number_input("Hot Water (L)", value=0.0, min_value=0.0, step=1.0, key="in_hot_water_liter", help="Enter a number ≥ 0")
                cold_water = st.number_input("Cold/Chilled Water (L)", value=0.0, min_value=0.0, step=1.0, key="in_cold_water_liter", help="Enter a number ≥ 0")
            with e3:
                district_heating = st.number_input("District Heating (kWh)", value=0.0, min_value=0.0, step=0.1, key="in_district_heating_kwh", help="Enter a number ≥ 0")
                propane = st.number_input("Propane (L)", value=0.0, min_value=0.0, step=0.1, key="in_propane_liter", help="Enter a number ≥ 0")
                fuel_oil = st.number_input("Fuel Oil (L)", value=0.0, min_value=0.0, step=0.1, key="in_fuel_oil_liter", help="Enter a number ≥ 0")

        with st.expander("Transport inputs", expanded=expander_default):
            t1, t2, t3 = st.columns(3)
            with t1:
                petrol = st.number_input("Car Petrol (L)", value=0.0, min_value=0.0, step=0.1, key="in_petrol_liter", help="Enter a number ≥ 0")
                _p = _coerce_float(petrol)
                if _p is None or (_p is not None and _p < 0):
                    st.markdown("<div style='color:#b00020;font-size:0.9em;'>Enter a number ≥ 0</div>", unsafe_allow_html=True)
                diesel = st.number_input("Car Diesel (L)", value=0.0, min_value=0.0, step=0.1, key="in_diesel_liter", help="Enter a number ≥ 0")
            with t2:
                bus = st.number_input("Bus (km)", value=0.0, min_value=0.0, step=1.0, key="in_bus_km", help="Enter a number ≥ 0")
                train = st.number_input("Train (km)", value=0.0, min_value=0.0, step=1.0, key="in_train_km", help="Enter a number ≥ 0")
                bicycle = st.number_input("Bicycle (km)", value=0.0, min_value=0.0, step=1.0, key="in_bicycle_km", help="Enter a number ≥ 0")
            with t3:
                flight_short = st.number_input("Flight Short (km)", value=0.0, min_value=0.0, step=1.0, key="in_flight_short_km", help="Enter a number ≥ 0")
                flight_long = st.number_input("Flight Long (km)", value=0.0, min_value=0.0, step=1.0, key="in_flight_long_km", help="Enter a number ≥ 0")

        with st.expander("Meals inputs", expanded=expander_default):
            m1, m2, m3 = st.columns(3)
            with m1:
                meat = st.number_input("Meat (kg)", value=0.0, min_value=0.0, step=0.1, key="in_meat_kg", help="Enter a number ≥ 0")
                _m = _coerce_float(meat)
                if _m is None or (_m is not None and _m < 0):
                    st.markdown("<div style='color:#b00020;font-size:0.9em;'>Enter a number ≥ 0</div>", unsafe_allow_html=True)
                chicken = st.number_input("Chicken (kg)", value=0.0, min_value=0.0, step=0.1, key="in_chicken_kg", help="Enter a number ≥ 0")
            with m2:
                eggs = st.number_input("Eggs (kg)", value=0.0, min_value=0.0, step=0.1, key="in_eggs_kg", help="Enter a number ≥ 0")
                dairy = st.number_input("Dairy (kg)", value=0.0, min_value=0.0, step=0.1, key="in_dairy_kg", help="Enter a number ≥ 0")
            with m3:
                vegetarian = st.number_input("Vegetarian (kg)", value=0.0, min_value=0.0, step=0.1, key="in_vegetarian_kg", help="Enter a number ≥ 0")
                vegan = st.number_input("Vegan (kg)", value=0.0, min_value=0.0, step=0.1, key="in_vegan_kg", help="Enter a number ≥ 0")

        submitted = st.form_submit_button("Calculate & Save")

    # Gather input into a dict compatible with CO2_FACTORS
    user_data = {
        "electricity_kwh": electricity,
        "natural_gas_m3": natural_gas,
        "hot_water_liter": hot_water,
        "cold_water_liter": cold_water,
        "district_heating_kwh": district_heating,
        "propane_liter": propane,
        "fuel_oil_liter": fuel_oil,
        "petrol_liter": petrol,
        "diesel_liter": diesel,
        "bus_km": bus,
        "train_km": train,
        "bicycle_km": bicycle,
        "flight_short_km": flight_short,
        "flight_long_km": flight_long,
        "meat_kg": meat,
        "chicken_kg": chicken,
        "eggs_kg": eggs,
        "dairy_kg": dairy,
        "vegetarian_kg": vegetarian,
        "vegan_kg": vegan,
    }

    # Global input hint (tooltip-style note)
    st.markdown("<div style='color:#5f6368;font-size:0.9em;'>Hint: All numeric inputs should be <b>≥ 0</b>. Enter whole numbers or decimals as needed.</div>", unsafe_allow_html=True)

    # Inline warnings near inputs (if any invalid fields)
    show_input_warnings(user_data)

    # Calculate total emissions
    emissions = calculate_co2(user_data)
    # Store for cross-tab visibility (Eco Tips tab)
    st.session_state["emissions_today"] = float(emissions)

    # Compute per-activity once for optional breakdown tab
    per_activity = calculate_co2_breakdown(user_data)

    # Load history for KPIs and visuals
    history_df = load_history()
    yesterday_total = get_yesterday_total(history_df, selected_date)
    delta_pct = percentage_change(yesterday_total, emissions)
    streak = compute_streak(history_df, selected_date)

    # KPIs (compact)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", fmt_emissions(emissions))
    c2.metric("Δ vs. Yesterday", f"{delta_pct:.2f}%")
    c3.metric("Streak", f"{streak} day(s)")

    # Tabs for Dashboard and Breakdown
    tab_dashboard, tab_history, tab_breakdown, tab_tips = st.tabs(["📊 Dashboard", "📜 History", "📉 Breakdown", "💡 Eco Tips"])

    with tab_dashboard:
        # Two-column layout for compact one-page UI
        left_col, right_col = st.columns([2, 1])

        with left_col:
            # Category-wise table
            cat_emissions = compute_category_emissions(user_data)
            st.caption("Category totals (kg CO₂)")
            st.dataframe(
                pd.DataFrame.from_dict(cat_emissions, orient="index", columns=["kg CO₂"]),
                use_container_width=True,
                height=table_height,
            )

            st.caption("Today's category breakdown")
            st.bar_chart(pd.Series(cat_emissions, name="kg CO₂"), height=bar_height)

        with right_col:
            # Save after calculation
            if submitted:
                invalid = find_invalid_fields(user_data)
                if invalid:
                    bad_list = ", ".join(invalid)
                    st.warning(f"Some inputs look invalid (negative or non-numeric): {bad_list}. Please correct them before saving.")
                    # Optional logging
                    if st.session_state.get("perf_logging", False):
                        log_path = os.path.join(os.getcwd(), "perf_log.csv")
                        file_exists = os.path.exists(log_path)
                        try:
                            with open(log_path, mode="a", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                if not file_exists:
                                    writer.writerow(["timestamp", "elapsed_s", "emissions_kg"])  # header
                                writer.writerow([dt.datetime.now().isoformat(), "warning:invalid_inputs", f"{emissions:.4f}"])
                        except Exception:
                            pass
                elif not has_meaningful_input(user_data):
                    st.warning("No valid input detected – please log at least one activity before saving.")
                    if st.session_state.get("perf_logging", False):
                        log_path = os.path.join(os.getcwd(), "perf_log.csv")
                        file_exists = os.path.exists(log_path)
                        try:
                            with open(log_path, mode="a", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                if not file_exists:
                                    writer.writerow(["timestamp", "elapsed_s", "emissions_kg"])  # header
                                writer.writerow([dt.datetime.now().isoformat(), "warning:no_inputs", f"{emissions:.4f}"])
                        except Exception:
                            pass
                else:
                    save_entry(selected_date, user_data, emissions)
                    st.success("Saved.")

            # Visualizations (reduced height)
            history_df = load_history()  # reload after potential save
            if not history_df.empty:
                st.caption("Trend (Total kg CO₂)")
                history_df_display = history_df.copy()
                history_df_display["date"] = history_df_display["date"].dt.date
                st.line_chart(history_df_display.set_index("date")["total_kg"], height=trend_height)

                # CSV export button
                csv_buf = io.StringIO()
                history_df.to_csv(csv_buf, index=False)
                st.download_button(
                    label="⬇️ Download history CSV",
                    data=csv_buf.getvalue(),
                    file_name="history.csv",
                    mime="text/csv",
                    key="download_history_csv_dashboard",
                )

            # Eco tip and status (compact)
            st.caption("Eco tip & status")
            start_time = time.time()
            placeholder = st.empty()
            tip = None
            threshold = float(st.session_state.get("spinner_threshold", 0.3))
            # Run tip generation in a background thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(generate_tip, user_data, emissions)
                spinner_shown = False
                # Poll until done; when threshold reached, block within spinner context
                while True:
                    if fut.done():
                        tip = fut.result()
                        break
                    elapsed_loop = time.time() - start_time
                    if not spinner_shown and elapsed_loop > threshold:
                        with placeholder.container():
                            with st.spinner("Generating eco-tip..."):
                                tip = fut.result()  # wait until complete while spinner shows
                        spinner_shown = True
                        break
                    time.sleep(0.05)
            elapsed = time.time() - start_time
            icon, dom_cat = dominant_category_icon(user_data)
            st.session_state["last_tip"] = tip
            st.session_state["last_tip_icon"] = icon
            st.success(f"{icon} {tip}")
            st.caption(f"Tip generated in {elapsed:.2f}s")
            # Optional perf logging
            if st.session_state.get("perf_logging", False):
                log_path = os.path.join(os.getcwd(), "perf_log.csv")
                file_exists = os.path.exists(log_path)
                try:
                    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["timestamp", "elapsed_s", "emissions_kg"])  # header
                        writer.writerow([dt.datetime.now().isoformat(), f"{elapsed:.4f}", f"{emissions:.4f}"])
                except Exception:
                    pass
            st.success(status_message(emissions))

            # Badges (compact list)
            st.caption("Badges")
            badges = award_badges(emissions, streak, history_df)
            if badges:
                for b in badges:
                    st.markdown(f"- {b}")
            else:
                st.write("Log entries to start earning badges!")

        # Second row: mini sparklines by category
        if not history_df.empty:
            st.divider()
            st.caption("Mini trends by category")

            def _category_series(df: pd.DataFrame, keys: list[str]) -> pd.Series | None:
                present = [k for k in keys if k in df.columns]
                if not present:
                    return None
                s = pd.Series(0.0, index=df.index)
                for k in present:
                    factor = CO2_FACTORS.get(k)
                    s = s + df[k].fillna(0).astype(float) * factor
                return s

            def _seven_day_delta(s: pd.Series):
                if s is None or s.empty:
                    return None, None
                s = s.dropna()
                if len(s) < 2:
                    return None, None
                last7 = float(s.iloc[-7:].sum())
                prev7 = float(s.iloc[-14:-7].sum()) if len(s) >= 14 else 0.0
                return last7, percentage_change(prev7, last7)

            df_sorted = history_df.sort_values("date").copy()
            df_sorted_indexed = df_sorted.set_index("date")

            energy_s = _category_series(df_sorted, CATEGORY_MAP["Energy"]) 
            energy_s = energy_s if (energy_s is not None and not energy_s.empty) else pd.Series(dtype=float)
            transport_s = _category_series(df_sorted, CATEGORY_MAP["Transport"]) 
            transport_s = transport_s if (transport_s is not None and not transport_s.empty) else pd.Series(dtype=float)
            meals_s = _category_series(df_sorted, CATEGORY_MAP["Meals"]) 
            meals_s = meals_s if (meals_s is not None and not meals_s.empty) else pd.Series(dtype=float)

            mini_height = 120 if density == "Compact" else 160
            c_en, c_tr, c_me = st.columns(3)
            with c_en:
                st.markdown("**Energy**")
                if not energy_s.empty:
                    st.line_chart(energy_s.set_axis(df_sorted_indexed.index), height=mini_height)
                    en_last7, en_pct = _seven_day_delta(energy_s)
                    if en_last7 is not None:
                        st.metric("7d total", f"{en_last7:.2f} kg", f"{en_pct:.1f}%", delta_color="inverse")
                    else:
                        st.caption("Not enough data yet")
                else:
                    st.write("No data yet")
            with c_tr:
                st.markdown("**Transport**")
                if not transport_s.empty:
                    st.line_chart(transport_s.set_axis(df_sorted_indexed.index), height=mini_height)
                    tr_last7, tr_pct = _seven_day_delta(transport_s)
                    if tr_last7 is not None:
                        st.metric("7d total", f"{tr_last7:.2f} kg", f"{tr_pct:.1f}%", delta_color="inverse")
                    else:
                        st.caption("Not enough data yet")
                else:
                    st.write("No data yet")
            with c_me:
                st.markdown("**Meals**")
                if not meals_s.empty:
                    st.line_chart(meals_s.set_axis(df_sorted_indexed.index), height=mini_height)
                    me_last7, me_pct = _seven_day_delta(meals_s)
                    if me_last7 is not None:
                        st.metric("7d total", f"{me_last7:.2f} kg", f"{me_pct:.1f}%", delta_color="inverse")
                    else:
                        st.caption("Not enough data yet")
                else:
                    st.write("No data yet")

    with tab_history:
        st.header("Saved History")
        history_all = load_history()
        if history_all.empty:
            st.info("No entries yet. Click Calculate & Save on the Dashboard to start your history.")
        else:
            st.caption("All logged entries (most recent shown first)")
            display_df = history_all.copy()
            display_df["date"] = display_df["date"].dt.date
            st.dataframe(display_df.sort_values("date", ascending=False), use_container_width=True, height=per_activity_height)

            # CSV export
            csv_buf = io.StringIO()
            history_all.to_csv(csv_buf, index=False)
            st.download_button(
                label="⬇️ Download history CSV",
                data=csv_buf.getvalue(),
                file_name="history.csv",
                mime="text/csv",
                key="download_history_csv_history_tab",
            )

    with tab_breakdown:
        st.caption("Per-activity emissions (kg CO₂)")
        if per_activity:
            st.dataframe(
                pd.Series(per_activity, name="kg CO₂").sort_values(ascending=False).to_frame(),
                use_container_width=True,
                height=per_activity_height,
            )
        else:
            st.info("No per-activity data to show yet.")

    with tab_tips:
        icon_hdr, dom_hdr = dominant_category_icon(user_data)
        st.subheader(f"{icon_hdr} Personalized Eco Tips")
        st.caption(f"Get a personalized tip based on today’s inputs and total emissions. Dominant today: {dom_hdr}.")

        # API source badge (GPT vs Fallback)
        source = st.session_state.get("last_tip_source") or ("GPT" if LAST_TIP_SOURCE == "gpt" else ("Fallback" if LAST_TIP_SOURCE == "fallback" else "Unknown"))
        badge_color = "#16a34a" if source == "GPT" else ("#6b7280" if source == "Fallback" else "#9ca3af")
        st.markdown(f"<div style='display:inline-block;padding:2px 8px;border-radius:12px;background:{badge_color};color:white;font-size:0.85em;'>AI source: {source}</div>", unsafe_allow_html=True)

        # Compact summary of today's inputs for context
        st.markdown("**Summary of today’s activities**")
        summary_str = format_summary(user_data)
        # Colored tag summary (HTML)
        st.markdown(format_summary_html(user_data), unsafe_allow_html=True)
        # Explicitly show today's total emissions coming from backend/session
        em_today = float(st.session_state.get("emissions_today", emissions))
        st.metric("Today's total (backend)", fmt_emissions(em_today))
        # Copy-ready block using Streamlit's built-in copy icon on code blocks
        st.caption("Copy-ready summary (use the copy icon on the right):")
        st.code(summary_str)

        # Optional AI-powered summary alongside the rule-based one
        ai_open_default = bool(get_pref("exp_ai_summary_open", False))
        with st.expander("AI Summary (beta)", expanded=ai_open_default):
            ai_col1, ai_col2 = st.columns([1, 2])
            with ai_col1:
                default_show_ai = bool(get_pref("default_show_ai_summary", st.session_state.get("use_ai_summary", False)))
                use_ai_summary = st.checkbox(
                    "Show AI summary",
                    value=default_show_ai,
                    key="use_ai_summary",
                )
                pers_toggle = st.checkbox(
                    "Use AI summary by default",
                    value=default_show_ai,
                    key="use_ai_summary_default_toggle",
                    help="Save this as the default for future sessions",
                )
                if pers_toggle != default_show_ai:
                    set_pref("default_show_ai_summary", bool(pers_toggle))
            with ai_col2:
                regen_ai = st.button("🔁 Regenerate AI summary", disabled=not st.session_state.get("use_ai_summary", False))

            ai_open_toggle = st.checkbox("Open this section by default", value=ai_open_default, key="exp_ai_summary_open_toggle")
            if ai_open_toggle != ai_open_default:
                set_pref("exp_ai_summary_open", bool(ai_open_toggle))

        # Prompt experiments (choose structure and optional category)
        exp_open_default = bool(get_pref("exp_prompt_experiments_open", False))
        with st.expander("Prompt experiments", expanded=exp_open_default):
            exp_col1, exp_col2, exp_col3, exp_col4 = st.columns([1.2, 1.2, 1, 1])
            with exp_col1:
                default_mode = get_pref("default_prompt_mode", st.session_state.get("prompt_mode", "Contextualized"))
                prompt_mode = st.selectbox(
                    "Prompt mode",
                    options=["Contextualized", "Directive", "Persona"],
                    index=["Contextualized", "Directive", "Persona"].index(default_mode if default_mode in ["Contextualized", "Directive", "Persona"] else "Contextualized"),
                    key="prompt_mode",
                    help="Switch between different prompt structures",
                )
                if prompt_mode != default_mode:
                    set_pref("default_prompt_mode", prompt_mode)
            with exp_col2:
                default_cat = get_pref("default_prompt_category", st.session_state.get("prompt_category_opt", "Auto/None"))
                category_opt = st.selectbox(
                    "Category (optional)",
                    options=["Auto/None", "Energy", "Transport", "Meals"],
                    index=( ["Auto/None", "Energy", "Transport", "Meals"].index(default_cat) if (isinstance(default_cat, str) and default_cat in ["Auto/None", "Energy", "Transport", "Meals"]) else 0 ),
                    key="prompt_category_opt",
                    help="Adds a focus hint to the prompt if selected",
                )
                if category_opt != default_cat:
                    set_pref("default_prompt_category", category_opt)
            with exp_col3:
                varN = st.number_input("Randomized variations (N)", min_value=1, max_value=10, value=1, step=1, key="d6_varN", help="Run each scenario N times with small random perturbations to stress-test diversity.")
            with exp_col4:
                run_exp = st.button("🧪 Run prompt experiment")

            exp_open_toggle = st.checkbox("Open this section by default", value=exp_open_default, key="exp_prompt_experiments_open_toggle")
            if exp_open_toggle != exp_open_default:
                set_pref("exp_prompt_experiments_open", bool(exp_open_toggle))

        # Prompt log analysis
        log_open_default = bool(get_pref("exp_prompt_log_open", False))
        with st.expander("Prompt log analysis", expanded=log_open_default):
            log_path = os.path.join(os.getcwd(), "prompt_log.csv")
            if not os.path.exists(log_path):
                st.info("No prompt_log.csv found yet. Run a prompt experiment to create it.")
            else:
                try:
                    df_log = pd.read_csv(log_path)
                except Exception as e:
                    st.error(f"Could not read prompt_log.csv: {e}.")
                    df_log = None

                if df_log is not None and not df_log.empty:
                    # Basic hygiene
                    df_log["tip"] = df_log.get("tip", "").astype(str)
                    df_log["prompt"] = df_log.get("prompt", "").astype(str)
                    # Parse timestamp if present
                    if "timestamp" in df_log.columns:
                        with pd.option_context('mode.copy_on_write', False):
                            try:
                                df_log["timestamp"] = pd.to_datetime(df_log["timestamp"], errors="coerce")
                            except Exception:
                                pass

                    # Category filter
                    cats = sorted([c for c in df_log.get("category", pd.Series(dtype=str)).dropna().unique().tolist() if str(c).strip()])
                    chosen_cat = st.selectbox("Filter by category", options=["All"] + cats, index=0)
                    dfv = df_log if chosen_cat == "All" else df_log[df_log["category"].astype(str) == chosen_cat]
                    # Mode filter
                    modes = sorted([m for m in df_log.get("mode", pd.Series(dtype=str)).dropna().unique().tolist() if str(m).strip()])
                    chosen_mode = st.selectbox("Filter by mode", options=["All"] + modes, index=0)
                    if chosen_mode != "All":
                        dfv = dfv[dfv["mode"].astype(str) == chosen_mode]
                    if dfv.empty:
                        st.info("No rows match the current filter.")
                        st.stop()

                    total = len(dfv)
                    tip_words = dfv["tip"].fillna("").apply(lambda s: len(str(s).split()))
                    tip_chars = dfv["tip"].fillna("").apply(lambda s: len(str(s)))
                    avg_words = float(tip_words.mean()) if total else 0.0
                    avg_chars = float(tip_chars.mean()) if total else 0.0
                    uniq_tips = dfv["tip"].fillna("").str.strip().str.lower().nunique()
                    uniq_pct = (uniq_tips / total * 100.0) if total else 0.0
                    fallback_mask = dfv["prompt"].fillna("").str.contains("Fallback used", case=False, na=False)
                    fallback_count = int(fallback_mask.sum())

                    cA, cB, cC, cD = st.columns(4)
                    with cA:
                        st.metric("Samples", f"{total}")
                    with cB:
                        st.metric("Avg tip length (words)", f"{avg_words:.1f}")
                    with cC:
                        st.metric("Unique tips", f"{uniq_tips} ({uniq_pct:.1f}%)")
                    with cD:
                        st.metric("Fallback count", f"{fallback_count}")

                    # Tip length over time chart grouped by mode
                    st.caption("Tip length over time by mode")
                    chart_cols = st.columns([1,1])
                    with chart_cols[0]:
                        # Persist chart type in-session and to disk
                        default_chart_type = get_pref("prompt_len_chart_type", st.session_state.get("prompt_len_chart_type", "Line"))
                        chart_choice = st.radio(
                            "Chart type",
                            ["Line", "Bar"],
                            horizontal=True,
                            index=(0 if default_chart_type == "Line" else 1),
                            key="prompt_len_chart_type",
                        )
                        # Save any change
                        if chart_choice != default_chart_type:
                            set_pref("prompt_len_chart_type", chart_choice)
                    with chart_cols[1]:
                        # Persist binning choice in-session and to disk
                        default_bin = get_pref("prompt_time_bin", st.session_state.get("prompt_time_bin", "Hourly"))
                        bin_choice = st.radio(
                            "Time binning",
                            ["Hourly", "Daily"],
                            horizontal=True,
                            index=(0 if default_bin == "Hourly" else 1),
                            key="prompt_time_bin",
                        )
                        if bin_choice != default_bin:
                            set_pref("prompt_time_bin", bin_choice)
                    bin_freq = "H" if bin_choice == "Hourly" else "D"
                    try:
                        df_plot = dfv.copy()
                        df_plot["words"] = df_plot["tip"].fillna("").apply(lambda s: len(str(s).split()))
                        df_plot["chars"] = df_plot["tip"].fillna("").apply(lambda s: len(str(s)))
                        if "timestamp" in df_plot.columns and df_plot["timestamp"].notna().any() and "mode" in df_plot.columns:
                            piv = (df_plot
                                   .dropna(subset=["timestamp"]) 
                                   .groupby([pd.Grouper(key="timestamp", freq=bin_freq), "mode"])  # time bins
                                   .agg(words=("words", "mean"))
                                   .reset_index())
                            chart_df = piv.pivot(index="timestamp", columns="mode", values="words").sort_index()
                            if chart_choice == "Line":
                                st.line_chart(chart_df, height=220)
                            else:
                                st.bar_chart(chart_df, height=220)
                            # Characters chart
                            piv_c = (df_plot
                                     .dropna(subset=["timestamp"]) 
                                     .groupby([pd.Grouper(key="timestamp", freq=bin_freq), "mode"])  # time bins
                                     .agg(chars=("chars", "mean"))
                                     .reset_index())
                            chart_df_c = piv_c.pivot(index="timestamp", columns="mode", values="chars").sort_index()
                            st.caption("Characters over time by mode")
                            if chart_choice == "Line":
                                st.line_chart(chart_df_c, height=200)
                            else:
                                st.bar_chart(chart_df_c, height=200)
                        else:
                            st.info("Not enough timestamp/mode data to build the chart.")
                    except Exception as e:
                        st.caption(f"Chart unavailable: {e}")

                    # OK-rate over time chart grouped by mode with 95% CI using Altair
                    st.caption("OK-rate over time by mode (with 95% CI)")
                    show_ok_ci = st.checkbox("Show OK-rate with CIs", value=False, key="prompt_okrate_ci_toggle")
                    bootN = st.number_input("Bootstraps", min_value=100, max_value=2000, value=300, step=50, key="prompt_okrate_bootN") if show_ok_ci else 300
                    if show_ok_ci:
                        def _score_flags_row(row):
                            t = str(row.get("tip", "")).lower()
                            words = len(t.split())
                            verbs = ["try", "switch", "take", "bike", "walk", "reduce", "set", "lower", "replace", "unplug", "turn off", "plan", "use", "install", "carpool"]
                            actionable = any(v in t for v in verbs)
                            cat = str(row.get("category", "")).strip()
                            cat_hints = {
                                "Energy": ["electric", "heating", "thermostat", "kwh", "standby", "plug"],
                                "Transport": ["car", "bus", "train", "bike", "walk", "commute", "drive"],
                                "Meals": ["meal", "meat", "plant", "vegan", "vegetarian", "dairy"],
                            }
                            hints = cat_hints.get(cat, [])
                            relevant = True if (cat == "" or cat == "Mixed" or cat == "Ambiguous") else any(h in t for h in hints)
                            simple = (words <= 35) and (t.count('.') <= 2)
                            return bool(relevant and actionable and simple)

                        df_ok = dfv.copy()
                        df_ok["ok"] = df_ok.apply(_score_flags_row, axis=1)
                        agg = (
                            df_ok.dropna(subset=["timestamp"]) 
                                .groupby([pd.Grouper(key="timestamp", freq=bin_freq), "mode"]) 
                                .agg(ok_rate=("ok", lambda s: float(s.mean())*100.0), n=("ok","size"))
                                .reset_index()
                        )
                        # Bootstrap CIs per bin/mode
                        rows_ok = []
                        for (ts, md_), ggrp in agg.groupby(["timestamp", "mode"], dropna=False):
                            ok_rate = float(ggrp["ok_rate"].iloc[0]); n = int(ggrp["n"].iloc[0])
                            samples = []
                            for _ in range(int(bootN)):
                                p = max(0.0, min(1.0, ok_rate/100.0))
                                ones = sum(1 for __ in range(n) if random.random() < p)
                                samples.append((ones/n)*100.0 if n>0 else 0.0)
                            samples.sort()
                            lo = samples[int(0.025*(len(samples)-1))] if samples else 0.0
                            hi = samples[int(0.975*(len(samples)-1))] if samples else 0.0
                            rows_ok.append({"timestamp": ts, "mode": md_, "ok_rate": ok_rate, "ok_ci_low": lo, "ok_ci_high": hi, "n": n})
                        df_ok_ci = pd.DataFrame(rows_ok)
                        try:
                            if not df_ok_ci.empty:
                                ch = alt.Chart(df_ok_ci).encode(
                                x=alt.X('timestamp:T', title='Time'),
                                y=alt.Y('ok_rate:Q', title='OK rate (%)'),
                                color='mode:N',
                                tooltip=['mode:N','timestamp:T','n:Q','ok_rate:Q','ok_ci_low:Q','ok_ci_high:Q']
                            )
                            lines = ch.mark_line()
                            errs = ch.mark_errorband(opacity=0.3).encode(y='ok_ci_low:Q', y2='ok_ci_high:Q')
                            ok_time_chart = errs + lines
                            st.altair_chart(ok_time_chart, use_container_width=True)
                            # Save PNG to session for ZIP export if altair_saver available
                            if alt_save is not None:
                                try:
                                    import tempfile, os
                                    tmpf = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                                    alt_save(ok_time_chart, tmpf.name, fmt='png')
                                    with open(tmpf.name, 'rb') as _f:
                                        st.session_state['ok_rate_time_png'] = _f.read()
                                    os.unlink(tmpf.name)
                                except Exception:
                                    st.session_state['ok_rate_time_png'] = None
                            # Save CSV of over-time data to session for ZIP export
                            try:
                                st.session_state['ok_rate_time_csv'] = df_ok_ci.to_csv(index=False)
                            except Exception:
                                st.session_state['ok_rate_time_csv'] = None
                            # Save chart spec for HTML embedding
                            try:
                                st.session_state['ok_rate_time_spec'] = ok_time_chart.to_dict()
                            except Exception:
                                st.session_state['ok_rate_time_spec'] = None
                        except Exception as e:
                            st.caption(f"Altair error bar chart unavailable: {e}")

                        # Per-mode stats if available
                        if "mode" in dfv.columns:
                            try:
                                grp = dfv.groupby("mode", dropna=False)
                                rows = []
                                for mode_name, g in grp:
                                    tips_norm = g["tip"].fillna("").str.strip().str.lower().tolist()
                                    # Unique tips (case-insensitive)
                                    uniq = len(set(tips_norm))
                                    uniq_ratio = (uniq/len(g)*100.0) if len(g) else 0.0
                                    # Bigram distinctiveness
                                    def bigrams_of(s: str):
                                        toks = [t for t in s.split() if t]
                                        return list(zip(toks, toks[1:])) if len(toks) > 1 else []
                                    all_bigrams = []
                                    for t in tips_norm:
                                        all_bigrams.extend(bigrams_of(t.lower().strip()))
                                    total_bi = len(all_bigrams)
                                    uniq_bi = len(set(all_bigrams))
                                    uniq_bi_ratio = (uniq_bi/total_bi*100.0) if total_bi else 0.0
                                    avg_w = float(g["tip"].fillna("").apply(lambda s: len(str(s).split())).mean()) if len(g)>0 else 0.0
                                    fallbacks = int(g["prompt"].fillna("").str.contains("Fallback used", case=False, na=False).sum())
                                    rows.append({
                                        "mode": mode_name,
                                        "samples": len(g),
                                        "avg_words": avg_w,
                                        "unique_tips": uniq,
                                        "unique_tips_%": uniq_ratio,
                                        "unique_bigrams": uniq_bi,
                                        "unique_bigrams_%": uniq_bi_ratio,
                                        "fallbacks": fallbacks,
                                    })
                                per_mode = pd.DataFrame(rows).sort_values("mode")
                                st.caption("Per‑mode summary")
                                st.dataframe(per_mode, use_container_width=True)
                            except Exception:
                                pass

                            # Optional: boxplot per mode for tip lengths
                            show_box = st.checkbox("Show boxplot per mode (tip length in words)", value=False, key="prompt_boxplot_toggle")
                            if show_box:
                                try:
                                    import matplotlib.pyplot as plt
                                    fig, ax = plt.subplots(figsize=(6, 3))
                                    df_box = dfv.copy()
                                    if "mode" in df_box.columns:
                                        df_box["words"] = df_box["tip"].fillna("").apply(lambda s: len(str(s).split()))
                                        modes_order = sorted(df_box["mode"].dropna().unique().tolist())
                                        data = [df_box[df_box["mode"] == m]["words"].tolist() for m in modes_order]
                                        ax.boxplot(data, labels=modes_order, showfliers=True)
                                        ax.set_ylabel("Words")
                                        ax.set_title("Tip length distribution by mode")
                                        st.pyplot(fig, clear_figure=True)
                                except Exception:
                                    st.caption("Boxplot unavailable (matplotlib not installed)")
                        st.divider()
                        c1, c2 = st.columns([1,1])
                        with c1:
                            st.caption("Last 10 runs")
                            st.dataframe(dfv.tail(10), use_container_width=True, height=260)
                        with c2:
                            try:
                                with open(log_path, "r", encoding="utf-8") as fh:
                                    st.download_button("⬇️ Download prompt_log.csv", data=fh.read(), file_name="prompt_log.csv", mime="text/csv")
                            except Exception:
                                pass
                            if st.button("🗑️ Clear log (prompt_log.csv)"):
                                try:
                                    os.remove(log_path)
                                    st.success("Cleared prompt_log.csv. Refresh or run a new experiment to recreate.")
                                except Exception as e:
                                    st.error(f"Could not delete log: {e}")
                            if st.button("🧹 Clear experiments (session)"):
                                st.session_state["prompt_experiments"] = []
                                st.success("Cleared in-session prompt experiments. New runs will populate the appendix.")

        # Day 6: Prompt testing suite
        with st.expander("Day 6: Prompt testing suite", expanded=False):
            st.caption("Run predefined scenarios across modes (incl. ambiguous) and score tips for relevance, actionability, and simplicity.")
            colA, colB, colC, colD, colE, colF = st.columns([1.2, 1, 1, 1, 1, 1.4])
            with colA:
                default_amb = bool(get_pref("d6_include_ambig", st.session_state.get("d6_include_ambig", True)))
                include_ambiguous = st.checkbox("Include ambiguous cases", value=default_amb, key="d6_include_ambig")
                if include_ambiguous != default_amb:
                    set_pref("d6_include_ambig", bool(include_ambiguous))
            with colB:
                default_modes = get_pref("d6_modes", st.session_state.get("d6_modes", ["Contextualized", "Directive", "Persona"]))
                if not isinstance(default_modes, list) or not default_modes:
                    default_modes = ["Contextualized", "Directive", "Persona"]
                d6_modes = st.multiselect(
                    "Modes to test",
                    options=["Contextualized", "Directive", "Persona"],
                    default=default_modes,
                    key="d6_modes",
                )
                if sorted(d6_modes) != sorted(default_modes):
                    set_pref("d6_modes", d6_modes)
            with colC:
                default_varN = int(get_pref("d6_varN", st.session_state.get("d6_varN", 1)))
                varN = st.number_input("Randomized variations (N)", min_value=1, max_value=10, value=default_varN, step=1, key="d6_varN", help="Run each scenario N times with small random perturbations to stress-test diversity.")
                if int(varN) != default_varN:
                    set_pref("d6_varN", int(varN))
            with colD:
                run_suite = st.button("▶️ Run 10+ test scenarios")
            with colE:
                default_boot = int(get_pref("d6_bootN", st.session_state.get("d6_bootN", 300)))
                bootN = st.number_input("Bootstraps", min_value=100, max_value=2000, value=default_boot, step=50, key="d6_bootN", help="Resamples per mode to compute 95% CIs for OK rate and diversity metrics.")
                if int(bootN) != default_boot:
                    set_pref("d6_bootN", int(bootN))
            with colF:
                default_always = bool(get_pref("d6_always_ok_time", st.session_state.get("d6_always_ok_time", True)))
                always_ok_time = st.checkbox("Always include OK-rate over time", value=default_always, key="d6_always_ok_time", help="Generate the OK-rate over-time chart/spec as part of the Day 6 run so it's guaranteed in the HTML/ZIP.")
                if always_ok_time != default_always:
                    set_pref("d6_always_ok_time", bool(always_ok_time))
            # Quick toggle: Force edge-case visuals in report.html
            force_ec_default = bool(get_pref("d6_force_ec_visuals", st.session_state.get("d6_force_ec_visuals", False)))
            force_ec = st.checkbox("Force edge-case visuals in report.html", value=force_ec_default, help="Embed the edge-case distribution chart in the HTML report even if no edge-case rows are present.", key="d6_force_ec_visuals")
            if bool(force_ec) != bool(force_ec_default):
                set_pref("d6_force_ec_visuals", bool(force_ec))

            # Advanced: Tunable edge-case thresholds
            with st.expander("Advanced: Edge-case thresholds (extreme values)", expanded=False):
                st.caption("Adjust per-activity numeric limits used to classify 'extreme' inputs. Saved across sessions.")
                default_thr = get_pref("d6_thresholds", st.session_state.get("d6_thresholds", {
                    "electricity_kwh": 200.0,
                    "natural_gas_m3": 100.0,
                    "hot_water_liter": 2000.0,
                    "petrol_liter": 100.0,
                    "diesel_liter": 100.0,
                    "bus_km": 500.0,
                    "rail_km": 1000.0,
                    "meat_kg": 10.0,
                    "dairy_kg": 15.0,
                    "vegetarian_kg": 20.0,
                }))
                last_upd = get_pref("d6_thresholds_updated_at", "")
                if last_upd:
                    st.caption(f"Last updated: {last_upd}")
                c1, c2, c3 = st.columns(3)
                with c1:
                    th_elec = st.number_input("electricity_kwh", min_value=1.0, max_value=10000.0, value=float(default_thr.get("electricity_kwh", 200.0)), step=1.0)
                    th_gas = st.number_input("natural_gas_m3", min_value=1.0, max_value=10000.0, value=float(default_thr.get("natural_gas_m3", 100.0)), step=1.0)
                    th_hot = st.number_input("hot_water_liter", min_value=10.0, max_value=100000.0, value=float(default_thr.get("hot_water_liter", 2000.0)), step=10.0)
                with c2:
                    th_pet = st.number_input("petrol_liter", min_value=1.0, max_value=10000.0, value=float(default_thr.get("petrol_liter", 100.0)), step=1.0)
                    th_dsl = st.number_input("diesel_liter", min_value=1.0, max_value=10000.0, value=float(default_thr.get("diesel_liter", 100.0)), step=1.0)
                    th_bus = st.number_input("bus_km", min_value=1.0, max_value=100000.0, value=float(default_thr.get("bus_km", 500.0)), step=5.0)
                with c3:
                    th_rail = st.number_input("rail_km", min_value=1.0, max_value=100000.0, value=float(default_thr.get("rail_km", 1000.0)), step=10.0)
                    th_meat = st.number_input("meat_kg", min_value=0.1, max_value=1000.0, value=float(default_thr.get("meat_kg", 10.0)), step=0.1, format="%0.1f")
                    th_dairy = st.number_input("dairy_kg", min_value=0.1, max_value=1000.0, value=float(default_thr.get("dairy_kg", 15.0)), step=0.1, format="%0.1f")
                    th_veg = st.number_input("vegetarian_kg", min_value=0.1, max_value=1000.0, value=float(default_thr.get("vegetarian_kg", 20.0)), step=0.1, format="%0.1f")
                new_thr = {
                    "electricity_kwh": float(th_elec),
                    "natural_gas_m3": float(th_gas),
                    "hot_water_liter": float(th_hot),
                    "petrol_liter": float(th_pet),
                    "diesel_liter": float(th_dsl),
                    "bus_km": float(th_bus),
                    "rail_km": float(th_rail),
                    "meat_kg": float(th_meat),
                    "dairy_kg": float(th_dairy),
                    "vegetarian_kg": float(th_veg),
                }
                # Apply and persist thresholds
                if new_thr != default_thr:
                    set_pref("d6_thresholds", new_thr)
                    set_pref("d6_thresholds_updated_at", dt.datetime.now().isoformat())
                # Always apply current thresholds to the classifier
                try:
                    set_extreme_thresholds(get_pref("d6_thresholds", new_thr))
                except Exception:
                    pass
                # Controls row: Reset to defaults and Export JSON
                rc1, rc2 = st.columns([1,1])
                with rc1:
                    if st.button("Reset thresholds to defaults", key="btn_reset_thresholds"):
                        defaults = {
                            "electricity_kwh": 200.0,
                            "natural_gas_m3": 100.0,
                            "hot_water_liter": 2000.0,
                            "petrol_liter": 100.0,
                            "diesel_liter": 100.0,
                            "bus_km": 500.0,
                            "rail_km": 1000.0,
                            "meat_kg": 10.0,
                            "dairy_kg": 15.0,
                            "vegetarian_kg": 20.0,
                        }
                        set_pref("d6_thresholds", defaults)
                        set_pref("d6_thresholds_updated_at", dt.datetime.now().isoformat())
                        try:
                            set_extreme_thresholds(defaults)
                        except Exception:
                            pass
                        st.success("Thresholds reset to defaults. Collapse/expand to refresh controls if values don't immediately reflect.")
                with rc2:
                    try:
                        thr_json = json.dumps(get_pref("d6_thresholds", new_thr), indent=2)
                        st.download_button("⬇️ Export thresholds JSON", data=thr_json, file_name="edge_case_thresholds.json", mime="application/json", key="btn_export_thresholds")
                    except Exception:
                        pass

            # Import thresholds JSON
            upl = st.file_uploader("Import thresholds JSON", type=["json"], key="upl_thresholds_json")
            if upl is not None:
                try:
                    data = json.loads(upl.getvalue().decode("utf-8"))
                    if isinstance(data, dict) and data:
                        set_pref("d6_thresholds", data)
                        set_pref("d6_thresholds_updated_at", dt.datetime.now().isoformat())
                        try:
                            set_extreme_thresholds(data)
                        except Exception:
                            pass
                        st.success("Imported thresholds applied.")
                    else:
                        st.warning("Uploaded JSON is not a non-empty object. No changes applied.")
                except Exception as e:
                    st.error(f"Could not parse JSON: {e}")

            def _scenario_inputs():
                # Build at least 10 scenarios touching energy, transport, meals, plus ambiguous
                scenarios = []
                # Energy-focused
                scenarios.append(("Energy", {"electricity_kwh": 9.5}))
                scenarios.append(("Energy", {"natural_gas_m3": 4.0}))
                # Transport-focused
                scenarios.append(("Transport", {"petrol_liter": 6.0}))
                scenarios.append(("Transport", {"diesel_liter": 5.0}))
                scenarios.append(("Transport", {"bus_km": 20.0}))
                # Meals-focused
                scenarios.append(("Meals", {"meat_kg": 0.35}))
                scenarios.append(("Meals", {"dairy_kg": 0.6}))
                scenarios.append(("Meals", {"vegetarian_kg": 0.8}))
                # Mixed day
                scenarios.append(("Mixed", {"electricity_kwh": 3.0, "petrol_liter": 2.0, "meat_kg": 0.2}))
                # Ambiguous / unclear
                if include_ambiguous:
                    scenarios.append(("Ambiguous", {}))
                    scenarios.append(("Ambiguous", {"unknown_key": "help"}))
                    # Richer edge cases
                    scenarios.append(("Ambiguous", {"note": "🚗🍔💡"}))          # emojis
                    scenarios.append(("Ambiguous", {"note": "asdf123"}))         # nonsense
                    scenarios.append(("Ambiguous", {"note": "help"}))            # help request
                    scenarios.append(("Ambiguous", {"electricity_kwh": -5}))     # negative
                    scenarios.append(("Ambiguous", {"electricity_kwh": 50000}))  # extreme
                return scenarios

            def _perturb_inputs(inputs: dict) -> dict:
                # Apply small random multiplicative noise to numeric fields
                if not inputs:
                    return inputs
                out = {}
                for k, v in inputs.items():
                    try:
                        f = float(v)
                        noise = 1.0 + random.uniform(-0.15, 0.15)
                        out[k] = max(0.0, f * noise)
                    except Exception:
                        out[k] = v
                return out

            def _score_tip(tip: str, category: str) -> dict:
                t = (tip or "").lower()
                words = len(t.split())
                # Heuristic action verbs
                verbs = ["try", "switch", "take", "bike", "walk", "reduce", "set", "lower", "replace", "unplug", "turn off", "plan", "use", "install", "carpool"]
                actionable = any(v in t for v in verbs)
                # Relevance by category keyword hints
                cat_hints = {
                    "Energy": ["electric", "heating", "thermostat", "kwh", "standby", "plug"],
                    "Transport": ["car", "bus", "train", "bike", "walk", "commute", "drive"],
                    "Meals": ["meal", "meat", "plant", "vegan", "vegetarian", "dairy"],
                    "Mixed": [],
                    "Ambiguous": [],
                }
                hints = cat_hints.get(category, [])
                relevant = True if category in ("Mixed", "Ambiguous") else any(h in t for h in hints)
                simple = (words <= 35) and (t.count('.') <= 2)
                flags = {
                    "relevant": relevant,
                    "actionable": actionable,
                    "simple": simple,
                }
                flags["ok"] = all(flags.values())
                return flags

            if run_suite:
                scenarios = _scenario_inputs()
                results_rows = []
                for (cat, base_inputs) in scenarios:
                    for _ in range(int(varN)):
                        user_inputs = _perturb_inputs(base_inputs)
                        # Estimate emissions locally (simple sum using CO2_FACTORS)
                        em_est = 0.0
                        for k, v in (user_inputs or {}).items():
                            try:
                                em_est += float(v or 0) * CO2_FACTORS.get(k, 0.0)
                            except Exception:
                                pass
                        for m in d6_modes:
                            tip_i, prompt_i = generate_eco_tip_with_prompt(user_inputs, float(em_est), mode=m, category=(None if cat in ("Ambiguous", "Mixed") else cat))
                            flags = _score_tip(tip_i, cat)
                            results_rows.append({
                                "timestamp": dt.datetime.now().isoformat(),
                                "category": cat,
                                "mode": m,
                                "emissions_est_kg": f"{em_est:.2f}",
                                "tip": tip_i,
                                "relevant": flags["relevant"],
                                "actionable": flags["actionable"],
                                "simple": flags["simple"],
                                "ok": flags["ok"],
                                "prompt": prompt_i,
                                "input_type": classify_input_type(user_inputs),
                                "EC": "⚠️" if classify_input_type(user_inputs) != "valid" else "",
                            })
                # Convert and show table
                try:
                    df_cmp = pd.DataFrame(results_rows)
                    st.markdown("**Prompt Comparison Table (Day 6)**")
                    view_cols = ["timestamp", "category", "mode", "emissions_est_kg", "ok", "relevant", "actionable", "simple", "fallback_used", "input_type", "EC", "tip"]
                    # Edge-case filter UI
                    df_to_show = df_cmp
                    if "input_type" in df_cmp.columns:
                        unique_types = sorted([str(x) for x in df_cmp["input_type"].dropna().unique().tolist()])
                        default_filter = get_pref("d6_filter_input_type", unique_types)
                        # Ensure default matches available set
                        if not isinstance(default_filter, list) or not default_filter:
                            default_filter = unique_types
                        default_filter = [t for t in default_filter if t in unique_types]
                        sel_types = st.multiselect("Filter by input_type (edge-case)", options=unique_types, default=default_filter, key="d6_filter_input_type")
                        if sel_types != default_filter:
                            set_pref("d6_filter_input_type", sel_types)
                        if sel_types and len(sel_types) < len(unique_types):
                            df_to_show = df_cmp[df_cmp["input_type"].astype(str).isin(sel_types)]
                    st.dataframe(df_to_show[view_cols], use_container_width=True, height=380)
                    # Per-mode diversity and OK-rate metrics
                    try:
                        rows = []
                        # Helpers for bootstrap CIs
                        def _quantile(vals, q):
                            if not vals:
                                return 0.0
                            s = sorted(vals)
                            idx = max(0, min(len(s)-1, int(q * (len(s)-1))))
                            return float(s[idx])
                        def _bigrams_of(s: str):
                            toks = [t for t in s.split() if t]
                            return list(zip(toks, toks[1:])) if len(toks) > 1 else []
                        def _bootstrap_metrics(mode_df: pd.DataFrame, n_boot: int = 300):
                            tips = mode_df["tip"].fillna("").astype(str).tolist()
                            oks = mode_df["ok"].astype(bool).tolist()
                            n = len(tips)
                            if n == 0:
                                return {"ok_ci": (0.0, 0.0), "uniq_tips_ci": (0.0, 0.0), "uniq_bi_ci": (0.0, 0.0)}
                            ok_samples = []
                            utip_samples = []
                            ubi_samples = []
                            for _ in range(int(bootN)):
                                idxs = [random.randrange(0, n) for _ in range(n)]
                                tips_s = [tips[i] for i in idxs]
                                oks_s = [oks[i] for i in idxs]
                                # OK rate
                                ok_rate = (sum(oks_s)/len(oks_s))*100.0
                                ok_samples.append(ok_rate)
                                # Unique tips %
                                utip = (len(set(t.strip().lower() for t in tips_s))/len(tips_s))*100.0
                                utip_samples.append(utip)
                                # Unique bigrams %
                                bigs = []
                                for t in tips_s:
                                    bigs.extend(_bigrams_of(t.lower().strip()))
                                total_b = len(bigs)
                                ubi = ((len(set(bigs))/total_b)*100.0) if total_b else 0.0
                                ubi_samples.append(ubi)
                            return {
                                "ok_ci": (_quantile(ok_samples, 0.025), _quantile(ok_samples, 0.975)),
                                "uniq_tips_ci": (_quantile(utip_samples, 0.025), _quantile(utip_samples, 0.975)),
                                "uniq_bi_ci": (_quantile(ubi_samples, 0.025), _quantile(ubi_samples, 0.975)),
                            }
                        for mode_name, g in df_cmp.groupby("mode", dropna=False):
                            tips_norm = g["tip"].fillna("").str.strip().str.lower().tolist()
                            uniq = len(set(tips_norm))
                            uniq_ratio = (uniq/len(g)*100.0) if len(g) else 0.0
                            # Bigram distinctiveness
                            all_bigrams = []
                            for t in tips_norm:
                                all_bigrams.extend(_bigrams_of(t))
                            total_bi = len(all_bigrams)
                            uniq_bi = len(set(all_bigrams))
                            uniq_bi_ratio = (uniq_bi/total_bi*100.0) if total_bi else 0.0
                            ok_rate = float(g["ok"].mean()) * 100.0 if len(g)>0 else 0.0
                            cis = _bootstrap_metrics(g, n_boot=int(bootN))
                            ok_ci_low, ok_ci_high = cis['ok_ci']
                            ut_ci_low, ut_ci_high = cis['uniq_tips_ci']
                            ub_ci_low, ub_ci_high = cis['uniq_bi_ci']
                            rows.append({
                                "mode": mode_name,
                                "samples": len(g),
                                "ok_rate_%": ok_rate,
                                "ok_rate_ci_low": float(ok_ci_low),
                                "ok_rate_ci_high": float(ok_ci_high),
                                "unique_tips": uniq,
                                "unique_tips_%": uniq_ratio,
                                "unique_tips_ci_low": float(ut_ci_low),
                                "unique_tips_ci_high": float(ut_ci_high),
                                "unique_bigrams": uniq_bi,
                                "unique_bigrams_%": uniq_bi_ratio,
                                "unique_bigrams_ci_low": float(ub_ci_low),
                                "unique_bigrams_ci_high": float(ub_ci_high),
                            })
                        per_mode_metrics = pd.DataFrame(rows).sort_values("mode")
                        st.caption("Per‑mode diversity and OK‑rate")
                        st.dataframe(per_mode_metrics, use_container_width=True)
                        # Altair charts with error bars
                        try:
                            base_ok = alt.Chart(per_mode_metrics).encode(x=alt.X('mode:N', title='Mode'))
                            bars_ok = base_ok.mark_bar(color='#4F46E5').encode(
                                y=alt.Y('ok_rate_%:Q', title='OK rate (%)'),
                                tooltip=['mode:N','samples:Q','ok_rate_%:Q','ok_rate_ci_low:Q','ok_rate_ci_high:Q']
                            )
                            error_ok = base_ok.mark_errorband(opacity=0.3).encode(y='ok_rate_ci_low:Q', y2='ok_rate_ci_high:Q')
                            st.altair_chart(bars_ok + error_ok, use_container_width=True)

                            cols = st.columns(2)
                            with cols[0]:
                                st.caption("Unique tips % by mode (with CI)")
                                base_ut = alt.Chart(per_mode_metrics).encode(x=alt.X('mode:N', title='Mode'))
                                bars_ut = base_ut.mark_bar(color='#059669').encode(
                                    y=alt.Y('unique_tips_%:Q', title='Unique tips (%)'),
                                    tooltip=['mode:N','samples:Q','unique_tips_%:Q','unique_tips_ci_low:Q','unique_tips_ci_high:Q']
                                )
                                error_ut = base_ut.mark_errorband().encode(y='unique_tips_ci_low:Q', y2='unique_tips_ci_high:Q')
                                st.altair_chart(bars_ut + error_ut, use_container_width=True)
                            with cols[1]:
                                st.caption("Unique bigrams % by mode (with CI)")
                                base_ub = alt.Chart(per_mode_metrics).encode(x=alt.X('mode:N', title='Mode'))
                                bars_ub = base_ub.mark_bar(color='#2563EB').encode(
                                    y=alt.Y('unique_bigrams_%:Q', title='Unique bigrams (%)'),
                                    tooltip=['mode:N','samples:Q','unique_bigrams_%:Q','unique_bigrams_ci_low:Q','unique_bigrams_ci_high:Q']
                                )
                                error_ub = base_ub.mark_errorband().encode(y='unique_bigrams_ci_low:Q', y2='unique_bigrams_ci_high:Q')
                                st.altair_chart(bars_ub + error_ub, use_container_width=True)
                        except Exception as e:
                            st.caption(f"Altair error bar charts unavailable: {e}")
                    except Exception:
                        df_cmp = None
                    # Findings summary for HTML
                    try:
                        best_row = per_mode_metrics.sort_values('ok_rate_%', ascending=False).iloc[0]
                        findings = [
                            f"Best mode: {best_row['mode']} (OK rate {best_row['ok_rate_%']:.1f}% | 95% CI [{best_row['ok_rate_ci_low']:.1f}, {best_row['ok_rate_ci_high']:.1f}])",
                            f"Diversity (unique tips %): {best_row['unique_tips_%']:.1f}% | 95% CI [{best_row['unique_tips_ci_low']:.1f}, {best_row['unique_tips_ci_high']:.1f}]",
                            f"Diversity (unique bigrams %): {best_row['unique_bigrams_%']:.1f}% | 95% CI [{best_row['unique_bigrams_ci_low']:.1f}, {best_row['unique_bigrams_ci_high']:.1f}]",
                        ]
                    except Exception:
                        findings = []
                    # Top 3 deltas across modes (OK rate) and CI overlap flag
                    try:
                        pm = per_mode_metrics[["mode","ok_rate_%","ok_rate_ci_low","ok_rate_ci_high"]].copy()
                        pairs = []
                        for i in range(len(pm)):
                            for j in range(i+1, len(pm)):
                                a = pm.iloc[i]; b = pm.iloc[j]
                                delta = float(a["ok_rate_%"] - b["ok_rate_%"])  # a minus b
                                # CI overlap: if intervals intersect
                                overlap = not (a["ok_rate_ci_high"] < b["ok_rate_ci_low"] or b["ok_rate_ci_high"] < a["ok_rate_ci_low"])
                                pairs.append({
                                    "pair": f"{a['mode']} vs {b['mode']}",
                                    "delta": delta,
                                    "overlap": overlap,
                                })
                        pairs_sorted = sorted(pairs, key=lambda x: abs(x["delta"]), reverse=True)[:3]
                        top3 = [f"{p['pair']}: ΔOK {p['delta']:+.1f} pts (CIs overlap: {'yes' if p['overlap'] else 'no'})" for p in pairs_sorted]
                    except Exception:
                        top3 = []
                    # Always include OK-rate over time
                    if bool(get_pref("d6_always_ok_time", st.session_state.get("d6_always_ok_time", True))):
                        try:
                            # Build ok over-time from df_cmp directly (hourly)
                            df_tmp = df_cmp.copy()
                            df_tmp['timestamp'] = pd.to_datetime(df_tmp['timestamp'], errors='coerce')
                            df_tmp = df_tmp.dropna(subset=['timestamp'])
                            agg = (
                                df_tmp.groupby([pd.Grouper(key='timestamp', freq='H'), 'mode'])
                                    .agg(ok_rate=('ok', lambda s: float(s.mean())*100.0), n=('ok','size'))
                                    .reset_index()
                            )
                            # Bootstrap
                            rows_ok = []
                            for (ts, md_), ggrp in agg.groupby(['timestamp','mode'], dropna=False):
                                ok_rate = float(ggrp['ok_rate'].iloc[0]); n = int(ggrp['n'].iloc[0])
                                samples = []
                                for _ in range(int(bootN)):
                                    p = max(0.0, min(1.0, ok_rate/100.0))
                                    ones = sum(1 for __ in range(n) if random.random() < p)
                                    samples.append((ones/n)*100.0 if n>0 else 0.0)
                                samples.sort()
                                lo = samples[int(0.025*(len(samples)-1))] if samples else 0.0
                                hi = samples[int(0.975*(len(samples)-1))] if samples else 0.0
                                rows_ok.append({'timestamp': ts, 'mode': md_, 'ok_rate': ok_rate, 'ok_ci_low': lo, 'ok_ci_high': hi, 'n': n})
                            df_ok_ci = pd.DataFrame(rows_ok)
                            if not df_ok_ci.empty:
                                ch = alt.Chart(df_ok_ci).encode(
                                    x=alt.X('timestamp:T', title='Time'), y=alt.Y('ok_rate:Q', title='OK rate (%)'), color='mode:N',
                                    tooltip=['mode:N','timestamp:T','n:Q','ok_rate:Q','ok_ci_low:Q','ok_ci_high:Q']
                                )
                                ok_time_chart = ch.mark_errorband(opacity=0.3).encode(y='ok_ci_low:Q', y2='ok_ci_high:Q') + ch.mark_line()
                                # Save assets into session
                                st.session_state['ok_rate_time_spec'] = ok_time_chart.to_dict()
                                try:
                                    st.session_state['ok_rate_time_csv'] = df_ok_ci.to_csv(index=False)
                                except Exception:
                                    pass
                                if alt_save is not None:
                                    import tempfile, os
                                    tmpf = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                                    alt_save(ok_time_chart, tmpf.name, fmt='png')
                                    with open(tmpf.name, 'rb') as _f:
                                        st.session_state['ok_rate_time_png'] = _f.read()
                                    os.unlink(tmpf.name)
                        except Exception:
                            pass
                    # Markdown export (compact table)
                    def _to_markdown_table(df: pd.DataFrame) -> str:
                        headers = " | ".join(view_cols)
                        sep = " | ".join(["---"] * len(view_cols))
                        lines = [f"| {headers} |", f"| {sep} |"]
                        for _, row in df.iterrows():
                            vals = [str(row[c]) for c in view_cols]
                            lines.append("| " + " | ".join(vals) + " |")
                        return "\n".join(lines)
                    # Summary section (top-line findings)
                    try:
                        best_row = per_mode_metrics.sort_values('ok_rate_%', ascending=False).iloc[0]
                        summary_lines = [
                            "# Prompt Comparison Table (Day 6)",
                            "",
                            "## Summary Findings",
                            f"- **Best mode**: {best_row['mode']} (OK rate: {best_row['ok_rate_%']:.1f}% | 95% CI [{best_row['ok_rate_ci_low']:.1f}, {best_row['ok_rate_ci_high']:.1f}])",
                            f"- **Diversity (unique tips %)**: {best_row['unique_tips_%']:.1f}% | 95% CI [{best_row['unique_tips_ci_low']:.1f}, {best_row['unique_tips_ci_high']:.1f}]",
                            f"- **Diversity (unique bigrams %)**: {best_row['unique_bigrams_%']:.1f}% | 95% CI [{best_row['unique_bigrams_ci_low']:.1f}, {best_row['unique_bigrams_ci_high']:.1f}]",
                            "",
                            "## Full Table",
                        ]
                        md_header = "\n".join(summary_lines)
                    except Exception:
                        md_header = "# Prompt Comparison Table (Day 6)\n\n## Full Table"
                    md_table = _to_markdown_table(df_cmp[view_cols])
                    md = md_header + "\n\n" + md_table
                    st.markdown("**Markdown preview**")
                    st.code(md, language="markdown")
                    st.download_button(
                        label="⬇️ Save comparison as Markdown",
                        data=md,
                        file_name="prompt_compare.md",
                        mime="text/markdown",
                        key="download_prompt_compare_md",
                    )
                    # ZIP export (CSV + MD + per-mode metrics)
                    try:
                        buf = io.BytesIO()
                        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                            # README.md with quick guide
                            readme = []
                            readme.append("# Day 6 Report Package")
                            readme.append("")
                            readme.append("## Files")
                            readme.append("- prompt_compare.csv — All test rows (mode/category/ok flags/tip/prompt)")
                            readme.append("- prompt_compare.md — Summary + full table in Markdown")
                            readme.append("- per_mode_metrics.csv — Per-mode OK rate and diversity with 95% CIs")
                            readme.append("- per_mode_metrics.md — Per-mode metrics in Markdown")
                            readme.append("- ok_rate.png — OK rate bar chart with error bars")
                            readme.append("- unique_tips.png — Unique tips % bar chart with error bars")
                            readme.append("- unique_bigrams.png — Unique bigrams % bar chart with error bars")
                            readme.append("- ok_rate_over_time.png — OK rate over time with 95% CI (if generated)")
                            readme.append("- ok_rate_over_time.csv — Data for over-time chart (if generated)")
                            readme.append("- report.html — Interactive HTML report (if charts available)")
                            readme.append("")
                            readme.append("## How to interpret")
                            readme.append("- OK rate: percent of tips that are Relevant, Actionable, and Simple.")
                            readme.append("- Unique tips %: lexical uniqueness across tips (case-insensitive).")
                            readme.append("- Unique bigrams %: n-gram diversity proxy (higher means more varied phrasing).")
                            readme.append("- 95% CI: uncertainty from bootstrap resampling.")
                            readme.append("")
                            readme.append("## input_type taxonomy (heuristics)")
                            readme.append("- empty: no usable numeric or text inputs provided")
                            readme.append("- help: explicit request for help (e.g., 'help', '?', 'what should I do?')")
                            readme.append("- emoji: text dominated by emojis/symbols (very low alphanumeric ratio)")
                            readme.append("- nonsense: symbol-heavy/gibberish without clear meaning")
                            readme.append("- negative: any numeric input is negative (per field)")
                            readme.append("- extreme: numeric input exceeds per-activity sanity thresholds (heuristics)")
                            readme.append("- valid: inputs pass the above checks")
                            zf.writestr('README.md', "\n".join(readme))
                            # Combined HTML (embed Altair specs)
                            try:
                                html_parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>Day 6 Report</title>"]
                                html_parts.append("<script src='https://cdn.jsdelivr.net/npm//vega@5'></script>")
                                html_parts.append("<script src='https://cdn.jsdelivr.net/npm/vega-lite@5'></script>")
                                html_parts.append("<script src='https://cdn.jsdelivr.net/npm/vega-embed@6'></script>")
                                html_parts.append("</head><body><h1>Day 6 Interactive Report</h1>")
                                def _embed_div(spec, div_id):
                                    return f"<div id='{div_id}'></div><script>vegaEmbed('#{div_id}', {json.dumps(spec)});</script>"
                                # Build specs for the three per-mode charts
                                try:
                                    spec_ok = (bars_ok + error_ok).to_dict()
                                    spec_ut = (bars_ut + error_ut).to_dict()
                                    spec_ub = (bars_ub + error_ub).to_dict()
                                    # Findings block
                                    if findings:
                                        html_parts.append("<h2>Findings</h2><ul>" + "".join([f"<li>{f}</li>" for f in findings]) + "</ul>")
                                    if top3:
                                        html_parts.append("<h3>Top 3 deltas (OK rate)</h3><ul>" + "".join([f"<li>{t}</li>" for t in top3]) + "</ul>")
                                    html_parts.append("<h2>OK rate by mode</h2>")
                                    html_parts.append(_embed_div(spec_ok, "ok_rate"))
                                    html_parts.append("<h2>Unique tips % by mode</h2>")
                                    html_parts.append(_embed_div(spec_ut, "unique_tips"))
                                    html_parts.append("<h2>Unique bigrams % by mode</h2>")
                                    html_parts.append(_embed_div(spec_ub, "unique_bigrams"))
                                except Exception:
                                    pass
                                # Over-time chart if present
                                if st.session_state.get('ok_rate_time_spec'):
                                    html_parts.append("<h2>OK rate over time</h2>")
                                    html_parts.append(_embed_div(st.session_state.get('ok_rate_time_spec'), "ok_rate_time"))
                                # Edge-case distribution chart (optional force)
                                try:
                                    force_ec = bool(get_pref("d6_force_ec_visuals", st.session_state.get("d6_force_ec_visuals", False)))
                                    show_ec = force_ec
                                    spec_ec = None
                                    no_edge_cases = False
                                    if "input_type" in df_cmp.columns:
                                        df_e = df_cmp.copy()
                                        e_counts = df_e.groupby(["mode", "input_type"], dropna=False).size().reset_index(name="count")
                                        import altair as alt
                                        ec_chart = alt.Chart(e_counts).mark_bar().encode(
                                            x=alt.X('mode:N', title='Mode'),
                                            y=alt.Y('count:Q', stack='normalize', title='Edge-case proportion'),
                                            color=alt.Color('input_type:N', title='input_type')
                                        )
                                        spec_ec = ec_chart.to_dict()
                                        show_ec = True
                                        try:
                                            no_edge_cases = df_e["input_type"].astype(str).apply(lambda s: s.strip().lower() == 'valid').all()
                                        except Exception:
                                            no_edge_cases = False
                                    if show_ec and spec_ec is not None:
                                        html_parts.append("<h2>Edge-case distribution by mode</h2>")
                                        html_parts.append(_embed_div(spec_ec, "edge_cases"))
                                        if force_ec and no_edge_cases:
                                            html_parts.append("<div style='font-size:0.9em;color:#64748B;font-style:italic;margin-top:4px;'>⚠️ Edge-case chart displayed due to forced rendering; no edge-case inputs were present in this run.</div>")
                                except Exception:
                                    pass
                                html_parts.append("</body></html>")
                                zf.writestr('report.html', "".join(html_parts))
                            except Exception:
                                pass
                            zf_bytes = buf.getvalue()
                            st.download_button(
                                label="⬇️ Save all (CSV + MD) as ZIP",
                                data=zf_bytes,
                                file_name="day6_results.zip",
                                mime="application/zip",
                                key="download_day6_zip",
                            )
                    except Exception as e:
                        st.caption(f"ZIP export unavailable: {e}")
                except Exception:
                    df_cmp = None
                # Determine AI summary to include in PDF if requested
                ai_summary_for_pdf = None
                try:
                    if bool(st.session_state.get("pdf_include_ai_summary", False)):
                        # Prefer any cached AI summary from the UI
                        ai_summary_for_pdf = st.session_state.get("ai_summary_text")
                        if not ai_summary_for_pdf:
                            # Generate on-demand for PDF context
                            df_hist = load_history()
                            ref_date = selected_date if isinstance(selected_date, (dt.date, dt.datetime)) else dt.date.today()
                            y_total = get_yesterday_total(df_hist, ref_date if isinstance(ref_date, dt.date) else ref_date.date())
                            cmp = percentage_change(y_total, em_today)
                            if y_total > 0:
                                comparison_text = f"{abs(cmp):.1f}% {'lower' if cmp < 0 else 'higher'} than yesterday ({fmt_emissions(y_total)})"
                            else:
                                comparison_text = "No data for yesterday"
                            # Weekly context (simple 7-day average)
                            weekly_context = None
                            try:
                                if not df_hist.empty:
                                    dfx = df_hist.copy()
                                    dfx["date"] = pd.to_datetime(dfx["date"]).dt.date
                                    last7 = dfx.sort_values("date").tail(7)
                                    if not last7.empty:
                                        avg7 = float(last7["total_kg"].mean())
                                        weekly_context = f"7-day average: {avg7:.2f} kg CO₂"
                            except Exception:
                                weekly_context = None
                            ai_summary_for_pdf = generate_ai_summary(
                                user_data=user_data,
                                emissions=em_today,
                                date=(ref_date.isoformat() if isinstance(ref_date, (dt.date, dt.datetime)) else str(ref_date)),
                                comparison_text=comparison_text,
                                streak_days=int(st.session_state.get("streak_days", 0)) if isinstance(st.session_state.get("streak_days", 0), (int, float)) else 0,
                                weekly_context=weekly_context,
                            )
                except Exception:
                    ai_summary_for_pdf = None
                # Prepare 7-day per-category sparkline data from history
                spark = {}
                try:
                    _hist = load_history()
                    if not _hist.empty:
                        dfh = _hist.copy()
                        dfh["date"] = pd.to_datetime(dfh["date"]).dt.date
                        last_dates = sorted(dfh["date"].unique())[-7:]
                        for d in last_dates:
                            row = dfh[dfh["date"] == d]
                            if row.empty:
                                continue
                            rdict = {k: float(row.iloc[-1].get(k, 0) or 0) for k in ALL_KEYS if k in row.columns}
                            cat_vals = compute_category_emissions(rdict)
                            for cat, val in cat_vals.items():
                                spark.setdefault(cat, []).append(float(val))
                except Exception:
                    spark = {}
                # Prefer uploaded logo; fallback to project's logo.png path if present
                logo_bytes = None
                if logo_file is None:
                    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
                    if os.path.exists(logo_path):
                        try:
                            with open(logo_path, "rb") as lf:
                                logo_bytes = lf.read()
                        except Exception:
                            logo_bytes = None
                pdf_bytes, err = build_eco_tips_pdf(
                    summary_str, tip_for_pdf, ai_summary_for_pdf, em_today, date_str, src_label, per_activity, compute_category_emissions(user_data), {
                        "today_total": fmt_emissions(em_today),
                        "yesterday_total": fmt_emissions(yesterday_total) if 'yesterday_total' in locals() else "",
                        "delta_pct": f"{percentage_change(yesterday_total, em_today):.2f}%" if 'yesterday_total' in locals() else "",
                        "streak_days": f"{streak} days" if 'streak' in locals() else "",
                    }, logo_bytes=logo_bytes, title_text=pdf_title, primary_color=pdf_primary_color, include_pie=bool(pdf_include_pie), include_sparklines=bool(pdf_include_spark), spark_data=spark, footer_text=pdf_footer_text if pdf_include_footer else None, margins_cm={"side": float(pdf_side_margin), "top": float(pdf_top_margin), "bottom": float(pdf_bottom_margin)}, text_hex=pdf_text_color, chart_bg_hex=pdf_chart_bg, experiments_appendix=(st.session_state.get("prompt_experiments") if bool(st.session_state.get("pdf_include_prompt_appendix", False)) else None))
                if pdf_bytes:
                    st.download_button(
                        label=" Download Eco Tips PDF",
                        data=pdf_bytes,
                        file_name=f"eco_tips_{date_str}.pdf",
                        mime="application/pdf",
                        key="download_eco_tips_pdf",
                    )
                    # After export, auto-clear experiments if enabled
                    if bool(get_pref("pdf_auto_clear_experiments", st.session_state.get("pdf_auto_clear_experiments", False))):
                        st.session_state["prompt_experiments"] = []
                        st.caption("Auto-cleared in-session prompt experiments after export.")
                else:
                    st.error(err or "PDF generation failed.")

        # Save all (CSV + MD) as ZIP
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            # README.md with quick guide
            readme = []
            readme.append("# Day 6 Report Package")
            readme.append("")
            readme.append("## Files")
            readme.append("- prompt_compare.csv — All test rows (mode/category/ok flags/tip/prompt)")
            readme.append("- prompt_compare.md — Summary + full table in Markdown")
            readme.append("- per_mode_metrics.csv — Per-mode OK rate and diversity with 95% CIs")
            readme.append("- per_mode_metrics.md — Per-mode metrics in Markdown")
            readme.append("- ok_rate.png — OK rate bar chart with error bars")
            readme.append("- unique_tips.png — Unique tips % bar chart with error bars")
            readme.append("- unique_bigrams.png — Unique bigrams % bar chart with error bars")
            readme.append("- ok_rate_over_time.png — OK rate over time with 95% CI (if generated)")
            readme.append("- ok_rate_over_time.csv — Data for over-time chart (if generated)")
            readme.append("- report.html — Interactive HTML report (if charts available)")
            readme.append("")
            readme.append("## How to interpret")
            readme.append("- OK rate: percent of tips that are Relevant, Actionable, and Simple.")
            readme.append("- Unique tips %: lexical uniqueness across tips (case-insensitive).")
            readme.append("- Unique bigrams %: n-gram diversity proxy (higher means more varied phrasing).")
            readme.append("- 95% CI: uncertainty from bootstrap resampling.")
            readme.append("")
            readme.append("## input_type taxonomy (heuristics)")
            readme.append("- empty: no usable numeric or text inputs provided")
            readme.append("- help: explicit request for help (e.g., 'help', '?', 'what should I do?')")
            readme.append("- emoji: text dominated by emojis/symbols (very low alphanumeric ratio)")
            readme.append("- nonsense: symbol-heavy/gibberish without clear meaning")
            readme.append("- negative: any numeric input is negative (per field)")
            readme.append("- extreme: numeric input exceeds per-activity sanity thresholds (heuristics)")
            readme.append("- valid: inputs pass the above checks")
            zf.writestr('README.md', "\n".join(readme))
            # Charts as PNG (if altair_saver available)
            if alt_save is not None:
                try:
                    # Save OK-rate bar with CIs
                    ok_tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    alt_save(bars_ok + error_ok, ok_tmp.name, fmt='png')
                    with open(ok_tmp.name, 'rb') as _f:
                        zf.writestr('ok_rate.png', _f.read())
                    os.unlink(ok_tmp.name)
                except Exception:
                    pass
            # Include active thresholds JSON for provenance
            try:
                thr_json = json.dumps(get_pref("d6_thresholds", {
                    "electricity_kwh": 200.0,
                    "natural_gas_m3": 100.0,
                    "hot_water_liter": 2000.0,
                    "petrol_liter": 100.0,
                    "diesel_liter": 100.0,
                    "bus_km": 500.0,
                    "rail_km": 1000.0,
                    "meat_kg": 10.0,
                    "dairy_kg": 15.0,
                    "vegetarian_kg": 20.0,
                }), indent=2)
                zf.writestr('edge_case_thresholds.json', thr_json)
            except Exception:
                pass
            zf_bytes = buf.getvalue()
            st.download_button(
                label="⬇️ Save all (CSV + MD) as ZIP",
                data=zf_bytes,
                file_name="day6_results.zip",
                mime="application/zip",
                key="download_day6_zip",
            )

if __name__ == "__main__":
    main()