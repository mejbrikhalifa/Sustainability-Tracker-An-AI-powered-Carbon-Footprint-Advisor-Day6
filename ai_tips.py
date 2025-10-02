# ai_tips.py
import os
from dotenv import load_dotenv
import time
from functools import lru_cache
from openai import OpenAI, OpenAIError

# Create client (safe even if key is missing; we guard before calling)
load_dotenv()  # Load variables from .env if present
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Public flag for UI to inspect last tip source: "gpt" | "fallback" | "unknown"
LAST_TIP_SOURCE = "unknown"

# Local factors used only for rules-based fallback logic.
# These mirror typical factors used elsewhere in the app, but are intentionally local
# so this module stays self-contained and never crashes due to imports.
LOCAL_CO2_FACTORS = {
    "electricity_kwh": 0.233,
    "natural_gas_m3": 2.03,
    "hot_water_liter": 0.25,
    "cold_water_liter": 0.075,
    "district_heating_kwh": 0.15,
    "propane_liter": 1.51,
    "fuel_oil_liter": 2.52,
    "petrol_liter": 0.235,
    "diesel_liter": 0.268,
    "bus_km": 0.12,
    "train_km": 0.14,
    "bicycle_km": 0.0,
    "flight_short_km": 0.275,
    "flight_long_km": 0.175,
    "meat_kg": 27.0,
    "chicken_kg": 6.9,
    "eggs_kg": 4.8,
    "dairy_kg": 13.0,
    "vegetarian_kg": 2.0,
    "vegan_kg": 1.5,
}

# Simple mapping of activities to categories (kept local to avoid cross-module deps)
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

# Per-activity numeric sanity thresholds (heuristics). Values above these are considered 'extreme'.
EXTREME_THRESHOLDS = {
    # Energy
    "electricity_kwh": 200.0,        # kWh/day (very high household day)
    "natural_gas_m3": 100.0,         # m^3/day
    "hot_water_liter": 2000.0,       # liters/day
    # Transport
    "petrol_liter": 100.0,           # liters/day
    "diesel_liter": 100.0,
    "bus_km": 500.0,                 # km/day
    "rail_km": 1000.0,
    # Meals (by mass consumed)
    "meat_kg": 10.0,                 # kg/day
    "dairy_kg": 15.0,
    "vegetarian_kg": 20.0,
}

def set_extreme_thresholds(new_thresholds: dict | None):
    """Override the default EXTREME_THRESHOLDS with values from the app.
    Pass a dict of {key: float}. Invalid entries are ignored.
    """
    global EXTREME_THRESHOLDS
    if not isinstance(new_thresholds, dict):
        return
    updated = EXTREME_THRESHOLDS.copy()
    for k, v in new_thresholds.items():
        try:
            updated[k] = float(v)
        except Exception:
            continue
    EXTREME_THRESHOLDS = updated

# -------- Ambiguity & input checks --------
def _has_meaningful_inputs(user_data: dict) -> bool:
    if not isinstance(user_data, dict) or not user_data:
        return False
    for v in user_data.values():
        try:
            if float(v or 0) > 0:
                return True
        except Exception:
            continue
    return False

def _generic_tip_or_clarify() -> str:
    # Provide a concise general tip and a clarifying follow‑up
    return (
        "💡 Turn off unused lights, unplug idle chargers, and take a short walk instead of a 1–2 km drive. "
        "Would you like a tip about Transport, Meals, or Energy?"
    )

def classify_input_type(user_data: dict) -> str:
    """Classify input edge-case type for logging and analysis.
    Returns one of: empty, help, emoji, nonsense, negative, extreme, valid.
    """
    if not isinstance(user_data, dict) or not user_data:
        return "empty"
    texts = []
    nums = []
    # Per-key numeric scan (catch negative or extreme early)
    for k, v in user_data.items():
        if isinstance(v, str):
            texts.append(v.strip())
            continue
        try:
            fv = float(v)
            nums.append(fv)
            if fv < 0:
                return "negative"
            thr = EXTREME_THRESHOLDS.get(k)
            if isinstance(thr, (int, float)) and fv > float(thr):
                return "extreme"
        except Exception:
            # ignore non-numeric
            pass
    # Empty-ish
    if (not texts) and (not nums):
        return "empty"
    # Help or question
    if any(t.lower() in {"help", "?", "what should i do?", "what can i do?"} for t in texts if t):
        return "help"
    # Emoji or symbol-heavy (few alnum)
    import re
    if any(t and (len(re.findall(r"[A-Za-z0-9]", t)) <= max(1, len(t) // 5)) for t in texts):
        # If contains common emoji/symbols
        if any(ch for ch in "🚗🍔💡🔥✨🌱😊👍🏽⚡" if (ch in "".join(texts))):
            return "emoji"
        # Otherwise likely nonsense
        return "nonsense"
    return "valid"

def _compute_breakdowns(user_data: dict, emissions: float):
    """Compute per-activity kg, per-category totals, dominant category and labels.
    Returns a dict with keys: activity_lines, category_lines, dominant_cat, dominant_pct, emission_level.
    """
    # Per-activity
    activity_kg = {}
    for k, v in (user_data or {}).items():
        try:
            amt = float(v or 0)
        except Exception:
            amt = 0.0
        factor = LOCAL_CO2_FACTORS.get(k, 0.0)
        kg = amt * factor
        if kg > 0:
            activity_kg[k] = kg

    # Per-category totals
    cat_totals = {cat: 0.0 for cat in CATEGORY_MAP}
    for cat, keys in CATEGORY_MAP.items():
        for k in keys:
            try:
                amt = float((user_data or {}).get(k, 0) or 0)
            except Exception:
                amt = 0.0
            cat_totals[cat] += amt * LOCAL_CO2_FACTORS.get(k, 0.0)

    # Dominant category
    if cat_totals:
        dominant_cat = max(cat_totals.items(), key=lambda x: x[1])[0]
        dom_val = cat_totals[dominant_cat]
        dominant_pct = (dom_val / emissions * 100.0) if emissions and emissions > 0 else 0.0
    else:
        dominant_cat, dominant_pct = "", 0.0

    # Emission level bucket
    if emissions > 50:
        emission_level = "high"
    elif emissions > 25:
        emission_level = "moderate"
    else:
        emission_level = "low"

    # Human-friendly lines
    activity_lines = []
    for k, kg in sorted(activity_kg.items(), key=lambda x: x[1], reverse=True):
        activity_lines.append(f"- {k.replace('_', ' ')}: {kg:.2f} kg CO₂")

    category_lines = []
    total = max(emissions, 0.0001)
    for cat, val in sorted(cat_totals.items(), key=lambda x: x[1], reverse=True):
        pct = (val / total) * 100.0
        category_lines.append(f"- {cat}: {val:.2f} kg ({pct:.1f}%)")

    return {
        "activity_lines": "\n".join(activity_lines) if activity_lines else "(no impactful activities logged)",
        "category_lines": "\n".join(category_lines) if category_lines else "(no categories)",
        "dominant_cat": dominant_cat,
        "dominant_pct": dominant_pct,
        "emission_level": emission_level,
    }

def generate_eco_tip(user_data: dict, emissions: float) -> str:
    """Public entry point used by the app. Tries GPT with caching and backoff;
    falls back to local rules if key missing or calls fail.
    """
    global LAST_TIP_SOURCE
    # Handle ambiguous/noisy inputs up front
    if not _has_meaningful_inputs(user_data):
        LAST_TIP_SOURCE = "fallback"
        return clean_tip(_generic_tip_or_clarify())
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not set. Using local tip generator.")
        LAST_TIP_SOURCE = "fallback"
        return clean_tip(local_tip(user_data, emissions))

    # Build a deterministic, structured context string for better prompting + caching
    try:
        breakdown = _compute_breakdowns(user_data, float(emissions or 0))
        context_str = (
            f"EMISSION LEVEL: {breakdown['emission_level']}\n"
            f"DOMINANT CATEGORY: {breakdown['dominant_cat']} ({breakdown['dominant_pct']:.1f}%)\n\n"
            f"ACTIVITY BREAKDOWN:\n{breakdown['activity_lines']}\n\n"
            f"CATEGORY BREAKDOWN:\n{breakdown['category_lines']}\n"
        )
    except Exception:
        # Fallback to simple key=value summary for cache key if something goes wrong
        try:
            context_str = ",".join(f"{k}={user_data.get(k, 0)}" for k in sorted(user_data.keys()))
        except Exception:
            context_str = str(sorted(user_data.items()))

    tip = _generate_eco_tip_cached(context_str, float(emissions or 0))
    if tip:
        LAST_TIP_SOURCE = "gpt"
        return clean_tip(tip)
    LAST_TIP_SOURCE = "fallback"
    return clean_tip(local_tip(user_data, emissions))

@lru_cache(maxsize=128)
def _generate_eco_tip_cached(user_data_key: str, emissions: float) -> str:
    """Cached GPT tip generator. Returns empty string on failure to signal fallback."""
    prompt = (
        """
        You are an expert sustainability coach providing personalized, actionable advice.

        CONTEXT:
        - Total CO₂ emissions today: {emissions:.2f} kg
        {structured_context}

        Generate a hyper-personalized eco tip following these rules:
        1) Target the biggest lever first (use the dominant category if present).
        2) Tailor guidance by emission level:
           - >50 kg: focus on highest-impact action today/tomorrow.
           - 25–50 kg: balance impact with feasibility.
           - <25 kg: suggest maintenance/optimization.
        3) Make it specific and doable in 24–48h. Include numbers/timeframes when possible.
        4) Relate directly to the activities listed.
        5) Tone: positive, motivational.
        6) Format: max 2 short sentences OR 1 concise bullet.
        7) Start with an emoji matching the category (⚡ energy, 🚗 transport, 🥗 meals) if relevant.
        """.strip()
    ).format(structured_context=user_data_key, emissions=emissions)

    return _gpt_tip_from_prompt(prompt)

def _gpt_tip_from_prompt(prompt: str) -> str:
    """Helper: call OpenAI with a provided prompt and return text or empty on failure."""
    retries = 3
    base_delay = 1.0
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a sustainability assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=160,
                temperature=0.7,
            )
            return (response.choices[0].message.content or "").strip()
        except OpenAIError as e:
            sleep_s = base_delay * (2 ** attempt)
            print(f"⚠️ GPT call failed (attempt {attempt+1}/{retries}): {e}. Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
        except Exception as e:
            print(f"⚠️ Unexpected GPT error: {e}")
            break
    return ""

# --- Prompt experimentation API ---

def _build_prompt_variant(
    emissions: float,
    structured_context: str,
    mode: str = "Contextualized",
    category: str | None = None,
) -> str:
    """Return a prompt string for the chosen mode. Modes: Directive, Contextualized, Persona.
    Optional category ("Energy", "Transport", "Meals") adds an extra focus cue.
    """
    cat_hint = ""
    if isinstance(category, str) and category in ("Energy", "Transport", "Meals"):
        cat_hint = f"\nFocus category: {category}."

    if mode == "Directive":
        return (
            """
            Generate a concise eco tip (max 2 short sentences) that the user can do within 24–48h.
            Make it specific, positive, and tied to their highest-impact activity today.
            {cat_hint}
            CONTEXT:
            - Total CO₂ today: {emissions:.2f} kg
            {structured_context}
            """.strip()
        ).format(emissions=emissions, structured_context=structured_context, cat_hint=cat_hint)

    if mode == "Persona":
        return (
            """
            You are a friendly, motivational sustainability coach with a practical tone.
            Provide ONE targeted action (max 2 short sentences) the user can take in the next 24–48 hours.
            Use an emoji matching the category (⚡ energy, 🚗 transport, 🥗 meals) if applicable.
            {cat_hint}
            CONTEXT:
            - Total CO₂ today: {emissions:.2f} kg
            {structured_context}
            """.strip()
        ).format(emissions=emissions, structured_context=structured_context, cat_hint=cat_hint)

    # Default: Contextualized
    return (
        """
        You are an expert sustainability assistant.
        Based on the user's day, suggest a concrete action to reduce their largest source.
        Requirements: specific numbers/timeframes if possible; 1–2 short sentences; positive tone.
        {cat_hint}
        CONTEXT:
        - Total CO₂ today: {emissions:.2f} kg
        {structured_context}
        """.strip()
    ).format(emissions=emissions, structured_context=structured_context, cat_hint=cat_hint)

def generate_eco_tip_with_prompt(
    user_data: dict,
    emissions: float,
    mode: str = "Contextualized",
    category: str | None = None,
) -> tuple[str, str]:
    """Generate an eco tip for experimentation, returning (tip_text, prompt_text).
    Respects OPENAI_API_KEY and falls back to local rules; on fallback, returns the constructed prompt with a note.
    """
    # Ambiguity handling: if no meaningful inputs, return general tip + clarification
    if not _has_meaningful_inputs(user_data):
        tip = _generic_tip_or_clarify()
        # Still return a synthetic prompt explaining ambiguity handling for logging
        prompt = (
            "Ambiguous inputs detected (no meaningful activity values). "
            "Returning general fallback + clarification question."
        )
        return clean_tip(tip), prompt
    # Build structured context
    breakdown = _compute_breakdowns(user_data, float(emissions or 0))
    structured_context = (
        f"EMISSION LEVEL: {breakdown['emission_level']}\n"
        f"DOMINANT CATEGORY: {breakdown['dominant_cat']} ({breakdown['dominant_pct']:.1f}%)\n\n"
        f"ACTIVITY BREAKDOWN:\n{breakdown['activity_lines']}\n\n"
        f"CATEGORY BREAKDOWN:\n{breakdown['category_lines']}\n"
    )
    prompt = _build_prompt_variant(float(emissions or 0), structured_context, mode=mode, category=category)

    if not os.getenv("OPENAI_API_KEY"):
        tip = local_tip(user_data, emissions)
        return clean_tip(tip), prompt + "\n\n(Note: Fallback used; no API key)"

    # Call GPT with the exact prompt
    tip_text = _gpt_tip_from_prompt(prompt)
    if not tip_text:
        tip_text = local_tip(user_data, emissions)
    return clean_tip(tip_text), prompt

def local_tip(user_data: dict, emissions: float) -> str:
    """
    Simple rules-based fallback that never crashes and gives helpful, actionable tips.
    - Identifies the largest-emitting activity using LOCAL_CO2_FACTORS
    - Provides a targeted tip for that activity
    - Includes tiered guidance based on total emissions
    """
    # Largest emitter detection
    best_key = None
    best_kg = 0.0
    for k, amt in user_data.items():
        try:
            amt_f = float(amt or 0)
        except Exception:
            amt_f = 0.0
        factor = LOCAL_CO2_FACTORS.get(k)
        if factor is None:
            continue
        kg = amt_f * factor
        if kg > best_kg:
            best_kg = kg
            best_key = k

    # Tiered guidance based on total emissions
    if emissions > 60:
        preface = "🚨 High footprint today."
    elif emissions > 25:
        preface = "🌱 Moderate footprint today."
    else:
        preface = "🌍 Low footprint today—nice work!"

    # Targeted, practical suggestions
    tips_by_key = {
        # Energy
        "electricity_kwh": "Reduce standby power: switch devices fully off, use smart strips, and swap to LED bulbs.",
        "natural_gas_m3": "Lower heating setpoint by 1°C and seal drafts to cut gas use.",
        "hot_water_liter": "Take shorter showers and wash clothes on cold to cut hot water.",
        "cold_water_liter": "Fix leaks and install low‑flow faucets to save water and energy.",
        "district_heating_kwh": "Use a programmable thermostat and improve insulation to reduce heat demand.",
        "propane_liter": "Service your boiler and optimize thermostat schedules to trim propane use.",
        "fuel_oil_liter": "Schedule a boiler tune‑up and improve home insulation to cut oil use.",
        # Transport
        "petrol_liter": "Try car‑pooling or public transport 1–2 days/week; keep tires properly inflated.",
        "diesel_liter": "Combine errands into one trip and ease acceleration to save fuel.",
        "bus_km": "Great choice using the bus—consider a weekly pass to keep it going.",
        "train_km": "Nice! Train is low‑carbon—can you replace a short car trip with train?",
        "bicycle_km": "Awesome cycling—aim to replace one short car errand by bike this week.",
        "flight_short_km": "Consider rail for short trips, or bundle meetings to reduce flight frequency.",
        "flight_long_km": "Plan fewer long‑haul flights; if needed, choose non‑stop routes and economy seats.",
        # Meals
        "meat_kg": "Try a meat‑free day or swap red meat for chicken/plant‑based options.",
        "chicken_kg": "Balance meals with beans, lentils, and seasonal veggies a few times this week.",
        "eggs_kg": "Source from local farms and add plant‑based proteins to diversify.",
        "dairy_kg": "Switch to plant milk for coffee/tea and try dairy‑free snacks.",
        "vegetarian_kg": "Great! Add pulses and whole grains for protein and nutrition.",
        "vegan_kg": "Excellent! Keep variety with legumes, nuts, and B12‑fortified foods.",
    }

    if best_key and best_key in tips_by_key and best_kg > 0:
        return f"{preface} Biggest source: {best_key.replace('_', ' ')}. Tip: {tips_by_key[best_key]}"

    # Otherwise choose a general practical tip based on broad categories
    energy_load = sum((float(user_data.get(k, 0) or 0)) * LOCAL_CO2_FACTORS.get(k, 0) for k in [
        "electricity_kwh", "natural_gas_m3", "district_heating_kwh", "propane_liter", "fuel_oil_liter"
    ])
    transport_load = sum((float(user_data.get(k, 0) or 0)) * LOCAL_CO2_FACTORS.get(k, 0) for k in [
        "petrol_liter", "diesel_liter", "bus_km", "train_km", "flight_short_km", "flight_long_km"
    ])
    meals_load = sum((float(user_data.get(k, 0) or 0)) * LOCAL_CO2_FACTORS.get(k, 0) for k in [
        "meat_kg", "chicken_kg", "dairy_kg", "eggs_kg"
    ])

    if transport_load >= energy_load and transport_load >= meals_load and transport_load > 0:
        return f"{preface} Transport dominates—plan a no‑car day, try car‑pooling, or take the bus/train for one commute."
    if energy_load >= transport_load and energy_load >= meals_load and energy_load > 0:
        return f"{preface} Energy dominates—set heating 1–2°C lower and switch off devices fully at night."
    if meals_load > 0:
        return f"{preface} Diet is a big lever—try a meat‑free day and batch‑cook plant‑based meals this week."

    # Final generic tip
    return f"{preface} Start small: one meat‑free meal, one public‑transport trip, and switch devices fully off tonight."

def clean_tip(tip: str, max_sentences: int = 2) -> str:
    """Trim whitespace and limit the tip to a maximum number of sentences.
    Keeps the content concise for the UI.
    """
    if not isinstance(tip, str):
        return ""
    tip = tip.strip()
    if not tip:
        return tip
    # Split on periods while preserving basic punctuation
    parts = [p.strip() for p in tip.split('.') if p.strip()]
    if len(parts) > max_sentences:
        tip = '. '.join(parts[:max_sentences]).strip() + '.'
    # Nudge to avoid trailing overly long outputs
    if len(tip) > 280:
        tip = tip[:277].rstrip() + '…'
    return tip

def generate_tip(user_data: dict, emissions: float) -> str:
    """Facade used by the UI. Delegates to generate_eco_tip so we keep caching,
    backoff, prompt engineering, and fallback behaviors in one place.
    """
    return generate_eco_tip(user_data, emissions)

# Optional: AI-powered daily summary generator with fallback

def generate_ai_summary(
    user_data: dict,
    emissions: float,
    date: str | None = None,
    comparison_text: str | None = None,
    streak_days: int = 0,
    weekly_context: str | None = None,
) -> str:
    """Generate a concise AI summary. Falls back to a rules-based summary if API is unavailable.
    This function is additive and does not change existing UI unless wired in.
    """
    def _fallback_summary() -> str:
        b = _compute_breakdowns(user_data, float(emissions or 0))
        parts = []
        if date:
            parts.append(f"Date: {date}.")
        parts.append(f"Total: {emissions:.2f} kg CO₂ ({b['emission_level']}).")
        if comparison_text:
            parts.append(comparison_text)
        if streak_days and streak_days > 0:
            parts.append(f"Streak: {streak_days} days.")
        if b["dominant_cat"]:
            parts.append(f"{b['dominant_cat']} led today ({b['dominant_pct']:.1f}%).")
        return " ".join(parts)

    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_summary()

    b = _compute_breakdowns(user_data, float(emissions or 0))
    structured = (
        f"DATE: {date or '(unknown)'}\n"
        f"TOTAL: {emissions:.2f} kg CO₂\n"
        f"COMPARISON: {comparison_text or 'n/a'}\n"
        f"STREAK: {streak_days} days\n"
        f"EMISSION LEVEL: {b['emission_level']}\n"
        f"DOMINANT: {b['dominant_cat']} ({b['dominant_pct']:.1f}%)\n\n"
        f"ACTIVITIES:\n{b['activity_lines']}\n\n"
        f"CATEGORIES:\n{b['category_lines']}\n\n"
        f"WEEKLY CONTEXT:\n{weekly_context or '(none)'}\n"
    )

    prompt = (
        """
        You are a sustainability data analyst providing an insightful daily summary.

        USER DATA (structured):
        {structured}

        Write a concise 3–4 sentence summary that includes:
        - Performance snapshot (low/moderate/high, key achievement/concern, streak acknowledgment if >0)
        - Impact analysis (top 1–2 contributors with rough percentages)
        - One actionable forward-looking suggestion
        - Conversational tone with 1–2 relevant emojis, end motivationally
        """.strip()
    ).format(structured=structured)

    retries = 3
    base_delay = 1.0
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful, data-driven sustainability analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=220,
                temperature=0.5,
            )
            text = (response.choices[0].message.content or "").strip()
            return text or _fallback_summary()
        except OpenAIError as e:
            sleep_s = base_delay * (2 ** attempt)
            print(f"⚠️ GPT summary failed (attempt {attempt+1}/{retries}): {e}. Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
        except Exception as e:
            print(f"⚠️ Unexpected GPT summary error: {e}")
            break
    return _fallback_summary()