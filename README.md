# Sustainability Tracker – Day 6

**Tagline:** Helping users measure, reduce, and act for a greener future.

---

## Overview

Day 6 focused on **final enhancements, debugging, and security hardening** for the Sustainability Tracker. The goal was to ensure a smooth, secure, and reliable user experience when generating eco-tips via the LLM/API integration.

Key work areas included:

- Prompt optimization and handling ambiguous queries
- Edge-case detection and fallback responses
- Performance tuning and caching
- Error handling and debugging
- UX enhancements
- Security and trustworthiness
- Repository deliverables and documentation

---

## Features & Improvements

### 1. Prompt Optimization
- Reviewed all prompts for clarity, relevance, and actionability.
- Tested multiple structures: short, directive, contextualized.
- Added handling for ambiguous user queries (e.g., "What can I do today?").
- Ensured tips are simple, actionable, and sustainability-focused.

### 2. Handling Edge Cases
- Detected incomplete or noisy inputs: empty fields, nonsense, emojis, "help", negative/extreme numbers.
- Implemented fallback responses:
  - Default eco tip
  - Clarification request
- Added logging of test cases and responses.
- Provided edge-case metrics in tables, charts, and `findings.md`.
- Tunable thresholds for numeric extremes, persisted via `.user_prefs.json`.
- Optional force rendering of edge-case chart in reports.

### 3. Performance Tuning
- Measured response times for tip generation.
- Added caching for repeated queries or common tips.
- Optimized token usage by trimming unnecessary prompt text.
- Verified rate limits to avoid quota errors.

### 4. Error Handling & Debugging
- Added user-friendly messages for API failures.
- Fixed formatting issues (long tips, broken newlines).
- Tested limits for very long inputs and rapid multiple requests.
- Ensured "Last Tip" persists when switching tabs.

### 5. Final UX Enhancements
- Loading indicators for tip and summary generation.
- Streamlit buttons for copy/export functionality.
- Improved layout for compact density and organized sections.
- Confirmed PDF/CSV export works smoothly.

### 6. Security & Trustworthiness
- Mitigated prompt injection and sensitive data leakage.
- Used **LLMGuard / PromptShield** for prompt sanitization.
- Added **Guardrails AI** for structured outputs (eco-tip format).
- Restricted role-based tool access for future integrations.
- Documented all applied safeguards.

### 7. Repository & Deliverables
- All code changes pushed to **Day 6 branch**.
- Exported Day 6 ZIP includes:
  - `README.md` (this guide)
  - `prompt_compare.csv` & `.md`
  - `per_mode_metrics.csv` & `.md`
  - Charts: `ok_rate.png`, `unique_tips.png`, `unique_bigrams.png`, `ok_rate_over_time.png`
  - Interactive `report.html` (Altair/Vega-Lite charts embedded)
  - `findings.md` summarizing:
    - Top 3 deltas in OK rate
    - Edge-case counts and fallback rates
    - Thresholds applied
  - `edge_case_thresholds.json` (active thresholds used)
- Prepared demo video showing new features.

---

## Usage

1. Install required packages:

```bash
pip install -r requirements.txt
