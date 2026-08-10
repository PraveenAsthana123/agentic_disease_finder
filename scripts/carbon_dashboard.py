"""
Green AI / Carbon Footprint Dashboard
Real-time power estimation + annual CO2 projection + sustainability scoring
for the NeuroLab AI epilepsy platform.

Reads: psutil system metrics + CarbonTracker compute model + model_comparison DB table
Endpoints: /api/carbon-tracker/overview | breakdown | definitions
"""
import os
import sqlite3
from datetime import datetime

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH  = os.path.join(_BASE_DIR, "data", "clinical.db")

# ---------------------------------------------------------------------------
# Carbon intensity reference (kg CO2/kWh) — IEA 2023 estimates
# ---------------------------------------------------------------------------
CARBON_INTENSITY = {
    "canada":        {"kg_co2_kwh": 0.130, "label": "Canada",       "note": "Hydro-heavy grid"},
    "norway":        {"kg_co2_kwh": 0.017, "label": "Norway",       "note": "Near-100% hydro"},
    "france":        {"kg_co2_kwh": 0.056, "label": "France",       "note": "Nuclear-heavy"},
    "eu_average":    {"kg_co2_kwh": 0.276, "label": "EU Average",   "note": "Mixed grid"},
    "us_average":    {"kg_co2_kwh": 0.417, "label": "US Average",   "note": "Mixed fossil/renewable"},
    "global_average":{"kg_co2_kwh": 0.475, "label": "Global Avg",   "note": "IEA 2023"},
    "india":         {"kg_co2_kwh": 0.708, "label": "India",        "note": "Coal-heavy"},
    "china":         {"kg_co2_kwh": 0.555, "label": "China",        "note": "Coal + growing renewables"},
}

# Region we model for (Canada — where the operator is located)
ACTIVE_REGION = "canada"

# ---------------------------------------------------------------------------
# Hardware power model
# ---------------------------------------------------------------------------
HARDWARE_POWER = {
    "cpu_active_w":       65,
    "gpu_inference_w":    75,
    "gpu_training_w":    250,
    "ram_per_gb_w":        3,
    "ssd_w":               5,
    "network_w":          10,
}

# ---------------------------------------------------------------------------
# Equivalences
# ---------------------------------------------------------------------------
def _equivalences(annual_co2_kg: float, annual_kwh: float) -> dict:
    return {
        "car_driving_km":        round(annual_co2_kg / 0.21),
        "trees_to_offset":       round(annual_co2_kg / 21, 1),
        "homes_powered_fraction":round(annual_kwh / 7500, 4),
        "smartphones_charged":   int(annual_kwh / 0.012),
        "flights_nyc_lon":       round(annual_co2_kg / 1100, 3),
    }

# ---------------------------------------------------------------------------
# Real power estimate via psutil
# ---------------------------------------------------------------------------
def _current_power() -> dict:
    try:
        import psutil
        cpu_pct  = psutil.cpu_percent(interval=0.2)
        mem_gb   = psutil.virtual_memory().used / (1024**3)
        # cpu scales linearly cpu_idle(10W) → cpu_max(125W)
        cpu_w    = round(10 + (125 - 10) * (cpu_pct / 100), 1)
        ram_w    = round(mem_gb * HARDWARE_POWER["ram_per_gb_w"], 1)
        gpu_w    = HARDWARE_POWER["gpu_inference_w"]
        other_w  = HARDWARE_POWER["ssd_w"] + HARDWARE_POWER["network_w"]
        total_w  = round(cpu_w + ram_w + gpu_w + other_w, 1)
        return {
            "cpu_pct":    round(cpu_pct, 1),
            "memory_gb":  round(mem_gb, 2),
            "cpu_w":      cpu_w,
            "ram_w":      ram_w,
            "gpu_w":      gpu_w,
            "other_w":    other_w,
            "total_w":    total_w,
        }
    except Exception:
        return {
            "cpu_pct":   5.0, "memory_gb": 8.0,
            "cpu_w":     16.0, "ram_w":    24.0,
            "gpu_w":     75.0, "other_w":  15.0,
            "total_w":   130.0,
        }

# ---------------------------------------------------------------------------
# Model training history from clinical.db
# ---------------------------------------------------------------------------
def _training_history() -> list:
    """Pull model_comparison rows for disease/accuracy context."""
    try:
        conn  = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows  = conn.execute(
            "SELECT model_type, disease, accuracy, auc, training_time_seconds, "
            "       created_at "
            "FROM model_comparison ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Estimate CO2 for a training run (watts × time → kWh → CO2)
# ---------------------------------------------------------------------------
def _training_co2(training_seconds: float, gpu_active: bool = True) -> dict:
    watts   = HARDWARE_POWER["cpu_active_w"] + (
        HARDWARE_POWER["gpu_training_w"] if gpu_active else 0
    ) + 8 * HARDWARE_POWER["ram_per_gb_w"]   # ~8 GB RAM during training
    kwh     = watts * training_seconds / 3600 / 1000
    co2_kg  = kwh * CARBON_INTENSITY[ACTIVE_REGION]["kg_co2_kwh"]
    return {"watts": watts, "kwh": round(kwh, 6), "co2_kg": round(co2_kg, 6)}

# ---------------------------------------------------------------------------
# Sustainability efficiency score (0-100)
# ---------------------------------------------------------------------------
def _efficiency_score(annual_kwh: float, annual_co2_kg: float) -> dict:
    intensity = CARBON_INTENSITY[ACTIVE_REGION]["kg_co2_kwh"]
    min_i, max_i = 0.017, 0.708
    # Region score: Canada 0.130 → ~83/100 on a 0-50 scale
    region_score = round(50 * (1 - (intensity - min_i) / (max_i - min_i)), 1)
    # Efficiency score: lower kWh per day is better (target 1 kWh/day)
    daily_kwh = annual_kwh / 365
    eff_score = round(30 * min(1, 1.0 / max(daily_kwh, 0.01)), 1)
    # Volume score: lower absolute annual CO2 (target <100 kg) is better
    vol_score = round(20 * min(1, 100 / max(annual_co2_kg, 1)), 1)
    total     = round(region_score + eff_score + vol_score, 1)
    grade     = (
        "A+" if total >= 90 else "A" if total >= 80 else
        "B"  if total >= 70 else "C" if total >= 60 else
        "D"  if total >= 50 else "F"
    )
    return {
        "total":          total,
        "region_score":   region_score,
        "efficiency_score": eff_score,
        "volume_score":   vol_score,
        "grade":          grade,
    }

# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
def _recommendations(annual_co2_kg: float, annual_kwh: float) -> list:
    recs = []
    intensity = CARBON_INTENSITY[ACTIVE_REGION]["kg_co2_kwh"]
    if intensity > 0.3:
        recs.append({
            "priority": "medium",
            "action": "Switch to a renewables-heavy cloud region (e.g., Norway 0.017 kg/kWh) to cut CO2 by up to 87%.",
        })
    if annual_kwh > 500:
        recs.append({
            "priority": "high",
            "action": "Apply INT8 quantization — reduces inference energy 30-50% with <1% accuracy drop.",
        })
    recs.append({
        "priority": "low",
        "action": "Enable model-output caching for repeated patient queries (same EEG segment).",
    })
    if annual_co2_kg > 200:
        recs.append({
            "priority": "medium",
            "action": f"Purchase ~{round(annual_co2_kg/1000*15,2)} USD/year in carbon offsets to neutralize footprint.",
        })
    recs.append({
        "priority": "low",
        "action": "Batch inference requests during off-peak hours (lower grid carbon intensity at night).",
    })
    return recs

# ---------------------------------------------------------------------------
# Public API ── overview()
# ---------------------------------------------------------------------------
def overview() -> dict:
    """KPIs: current power, daily/annual kWh, annual CO2, efficiency score,
    equivalences, region comparison, training cost summary."""
    power = _current_power()

    # Daily energy (24h at current load)
    daily_kwh  = round(power["total_w"] * 24 / 1000, 3)
    annual_kwh = round(daily_kwh * 365, 2)
    intensity  = CARBON_INTENSITY[ACTIVE_REGION]["kg_co2_kwh"]
    annual_co2 = round(annual_kwh * intensity, 2)

    # Per-prediction estimate (1 inference ≈ 50ms GPU, ~18W GPU fraction)
    pred_kwh   = round(75 * 0.05 / 3600 / 1000, 8)
    pred_co2_g = round(pred_kwh * intensity * 1000, 6)

    # Training CO2 — aggregate from model_comparison
    history = _training_history()
    total_training_kwh = 0.0
    total_training_co2 = 0.0
    for row in history:
        secs = row.get("training_time_seconds") or 60
        est  = _training_co2(float(secs), gpu_active=True)
        total_training_kwh += est["kwh"]
        total_training_co2 += est["co2_kg"]

    score = _efficiency_score(annual_kwh, annual_co2)
    equiv = _equivalences(annual_co2, annual_kwh)

    # Offset cost (US$15/ton)
    offset_cost_annual = round(annual_co2 / 1000 * 15, 2)

    return {
        "generated_at":    datetime.utcnow().isoformat() + "Z",
        "region":          ACTIVE_REGION,
        "carbon_intensity_kg_per_kwh": intensity,
        "kpis": {
            "current_power_w":       power["total_w"],
            "daily_kwh":             daily_kwh,
            "annual_kwh":            annual_kwh,
            "annual_co2_kg":         annual_co2,
            "efficiency_score":      score["total"],
            "efficiency_grade":      score["grade"],
            "offset_cost_usd_yr":    offset_cost_annual,
            "total_training_co2_kg": round(total_training_co2, 4),
            "total_training_kwh":    round(total_training_kwh, 6),
            "kwh_per_prediction":    pred_kwh,
            "co2_g_per_prediction":  pred_co2_g,
        },
        "efficiency_breakdown": score,
        "equivalences":         equiv,
        "current_power_detail": power,
        "recommendations":      _recommendations(annual_co2, annual_kwh),
        "training_summary": {
            "total_runs":    len(history),
            "total_kwh":     round(total_training_kwh, 6),
            "total_co2_kg":  round(total_training_co2, 4),
        },
    }

# ---------------------------------------------------------------------------
# Public API ── breakdown()
# ---------------------------------------------------------------------------
def breakdown() -> dict:
    """Region comparison table, hardware power breakdown, training per-model stats,
    carbon offset tiers, per-prediction projection."""
    power    = _current_power()
    history  = _training_history()

    # Region comparison
    region_table = []
    for key, info in CARBON_INTENSITY.items():
        # Assume same power usage, compare CO2 only
        daily_kwh   = power["total_w"] * 24 / 1000
        annual_kwh  = daily_kwh * 365
        annual_co2  = round(annual_kwh * info["kg_co2_kwh"], 2)
        annual_cost = round(annual_co2 / 1000 * 15, 2)
        region_table.append({
            "region":     key,
            "label":      info["label"],
            "note":       info["note"],
            "kg_co2_kwh": info["kg_co2_kwh"],
            "annual_co2_kg":  annual_co2,
            "offset_cost_usd": annual_cost,
            "active":     key == ACTIVE_REGION,
        })
    region_table.sort(key=lambda r: r["kg_co2_kwh"])

    # Hardware breakdown
    hw_breakdown = [
        {"component": "CPU",       "watts": power["cpu_w"],   "pct": round(power["cpu_w"]   / power["total_w"] * 100, 1)},
        {"component": "GPU",       "watts": power["gpu_w"],   "pct": round(power["gpu_w"]   / power["total_w"] * 100, 1)},
        {"component": "RAM",       "watts": power["ram_w"],   "pct": round(power["ram_w"]   / power["total_w"] * 100, 1)},
        {"component": "Other",     "watts": power["other_w"], "pct": round(power["other_w"] / power["total_w"] * 100, 1)},
    ]

    # Training per-model type
    by_model: dict = {}
    for row in history:
        mt = row.get("model_type", "unknown")
        if mt not in by_model:
            by_model[mt] = {"runs": 0, "total_kwh": 0.0, "total_co2_kg": 0.0}
        secs = float(row.get("training_time_seconds") or 60)
        est  = _training_co2(secs)
        by_model[mt]["runs"]          += 1
        by_model[mt]["total_kwh"]     += est["kwh"]
        by_model[mt]["total_co2_kg"]  += est["co2_kg"]

    model_co2_table = []
    for mt, stats in sorted(by_model.items(), key=lambda x: -x[1]["total_co2_kg"]):
        model_co2_table.append({
            "model_type":    mt,
            "runs":          stats["runs"],
            "total_kwh":     round(stats["total_kwh"], 6),
            "total_co2_kg":  round(stats["total_co2_kg"], 6),
            "avg_co2_g_per_run": round(stats["total_co2_kg"] / stats["runs"] * 1000, 4) if stats["runs"] else 0,
        })

    # Prediction volume scenarios
    intensity = CARBON_INTENSITY[ACTIVE_REGION]["kg_co2_kwh"]
    pred_kwh  = 75 * 0.05 / 3600 / 1000  # 50ms × 75W GPU
    scenarios = []
    for n in [100, 500, 1000, 5000, 10000, 50000]:
        ann_kwh  = round(pred_kwh * n * 365, 4)
        ann_co2  = round(ann_kwh * intensity, 4)
        scenarios.append({
            "daily_predictions": n,
            "annual_kwh":       ann_kwh,
            "annual_co2_kg":    ann_co2,
            "offset_cost_usd":  round(ann_co2 / 1000 * 15, 4),
        })

    # Carbon offset tiers (voluntary market)
    offset_tiers = [
        {"provider": "Gold Standard",      "usd_per_ton": 15,  "note": "Forestry + community"},
        {"provider": "Verra VCS",          "usd_per_ton": 8,   "note": "Verified Carbon Standard"},
        {"provider": "Climate Partner",    "usd_per_ton": 22,  "note": "Premium projects"},
        {"provider": "Microsoft Azure",    "usd_per_ton": 12,  "note": "Renewable Energy Credits"},
        {"provider": "Terrapass",          "usd_per_ton": 10,  "note": "US-based projects"},
    ]

    return {
        "region_comparison":    region_table,
        "hardware_breakdown":   hw_breakdown,
        "model_co2_table":      model_co2_table,
        "prediction_scenarios": scenarios,
        "carbon_offset_tiers":  offset_tiers,
    }

# ---------------------------------------------------------------------------
# Public API ── definitions()
# ---------------------------------------------------------------------------
def definitions() -> dict:
    return {
        "glossary": [
            {
                "term":       "kWh (kilowatt-hour)",
                "definition": "Unit of energy equal to 1000 watts consumed for 1 hour. "
                              "1 kWh ≈ average home uses 30 kWh/day.",
                "source":     "IEC 60050",
            },
            {
                "term":       "CO₂e (CO2 equivalent)",
                "definition": "All greenhouse gases converted to equivalent CO2 warming impact. "
                              "Electricity CO2e = kWh × grid carbon intensity.",
                "source":     "IPCC AR6",
            },
            {
                "term":       "Carbon Intensity (kg CO₂/kWh)",
                "definition": "Average CO2 emitted per kWh of electricity generated in a region. "
                              "Ranges from 0.017 (Norway, hydro) to 0.708 (India, coal).",
                "source":     "IEA 2023 Electricity Market Report",
            },
            {
                "term":       "Carbon Offset",
                "definition": "A verified reduction in CO2 emissions (e.g., reforestation, renewable "
                              "energy) purchased to compensate for an organisation's emissions.",
                "source":     "Voluntary Carbon Market (VERRA, Gold Standard)",
            },
            {
                "term":       "PUE (Power Usage Effectiveness)",
                "definition": "Ratio of total data-center power to IT equipment power. "
                              "PUE=1.0 is ideal; cloud providers average 1.1–1.2.",
                "source":     "Green Grid Standard",
            },
            {
                "term":       "Green AI",
                "definition": "AI research and deployment practices designed to minimize energy "
                              "consumption and carbon footprint while maintaining performance.",
                "source":     "Schwartz et al., 2019 — Green AI (ACM)",
            },
            {
                "term":       "Model Quantization",
                "definition": "Compressing model weights from float32 to int8 or float16, "
                              "reducing inference energy 30-50% with minimal accuracy loss.",
                "source":     "Dettmers et al., 2022",
            },
            {
                "term":       "Efficiency Score",
                "definition": "Composite 0–100 score combining region carbon intensity (50 pts), "
                              "energy per prediction (30 pts), and total annual emissions (20 pts).",
                "source":     "NeuroLab AI internal metric",
            },
        ],
        "metrics_reference": {
            "active_region":          ACTIVE_REGION,
            "active_carbon_intensity":CARBON_INTENSITY[ACTIVE_REGION]["kg_co2_kwh"],
            "inference_model_watts":  75,
            "inference_duration_ms":  50,
            "tree_absorption_kg_yr":  21,
            "car_co2_per_km":         0.21,
            "home_annual_kwh":        7500,
            "smartphone_charge_kwh":  0.012,
            "gold_std_offset_usd_ton":15,
        },
        "standards": [
            "IEA 2023 Electricity Market Report",
            "IPCC AR6 — Mitigation of Climate Change",
            "ACM Green AI (Schwartz et al., 2019)",
            "ISO 14064 — Greenhouse Gas Accounting",
            "Science Based Targets Initiative (SBTi)",
        ],
    }
