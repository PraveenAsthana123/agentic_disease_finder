"""
Create SLP (Speech-Language Pathologist) assessment data for epilepsy patients.
Inserts BNT (Boston Naming Test), WAB (Western Aphasia Battery),
Verbal Fluency, and MASA (Modified Mann Assessment of Swallowing Ability)
records into the assessments table in clinical.db.

Uses realistic score distributions for epilepsy patients — language deficits
are common in temporal lobe epilepsy (TLE), especially left-lateralized.
"""

import sqlite3
import os
import json
import random
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clinical.db")


def create_slp_data():
    c = sqlite3.connect(DB_PATH)

    # Get EPAT patients (the main patient cohort)
    patients = c.execute(
        "SELECT patient_id, age, gender FROM patients WHERE patient_id LIKE 'EPAT%' ORDER BY patient_id"
    ).fetchall()

    if not patients:
        print("No EPAT patients found.")
        c.close()
        return

    # Check if SLP data already exists
    existing = c.execute("SELECT COUNT(*) FROM assessments WHERE instrument IN ('BNT','WAB','VERBAL_FLUENCY','MASA')").fetchone()[0]
    if existing > 0:
        print(f"SLP data already exists ({existing} records). Skipping.")
        c.close()
        return

    now = datetime.now()
    records = []

    for pid, age, gender in patients:
        # Seed per patient for reproducibility
        random.seed(hash(pid + "SLP") % (2**31))

        # Simulate lateralization factor: ~40% of TLE patients have left-lateralized
        # focus → more language impact. Use patient index as pseudo-random source.
        idx = int(pid.replace("EPAT", "")) if pid.startswith("EPAT") else 1
        left_lateral = (idx % 5) in (0, 1)  # ~40% left-lateralized
        age_factor = max(0, (age or 35) - 30) / 50  # older → more deficit

        base_deficit = 0.15 if left_lateral else 0.05
        deficit = base_deficit + age_factor * 0.10

        # --- BNT (Boston Naming Test) ---
        # 60 items, normal ≥54/60, mild anomia 45-53, moderate 30-44, severe <30
        bnt_score = max(15, min(60, int(60 * (1 - deficit) - random.gauss(0, 4))))
        if bnt_score >= 54:
            bnt_interp, bnt_level = "Normal naming ability", "normal"
        elif bnt_score >= 45:
            bnt_interp, bnt_level = "Mild anomia", "mild"
        elif bnt_score >= 30:
            bnt_interp, bnt_level = "Moderate anomia — word-finding difficulties likely impact daily communication", "moderate"
        else:
            bnt_interp, bnt_level = "Severe anomia — significant confrontation naming deficit", "severe"

        bnt_items = {}
        correct_items = random.sample(range(1, 61), bnt_score)
        for i in range(1, 61):
            bnt_items[f"item{i}"] = 1 if i in correct_items else 0
        # Add semantic/phonemic cue counts
        missed = 60 - bnt_score
        bnt_items["semantic_cues_given"] = min(missed, random.randint(int(missed * 0.3), int(missed * 0.7) + 1))
        bnt_items["phonemic_cues_given"] = min(missed, random.randint(int(missed * 0.2), int(missed * 0.5) + 1))
        bnt_items["semantic_cue_correct"] = random.randint(int(bnt_items["semantic_cues_given"] * 0.3), bnt_items["semantic_cues_given"] + 1)
        bnt_items["phonemic_cue_correct"] = random.randint(int(bnt_items["phonemic_cues_given"] * 0.4), bnt_items["phonemic_cues_given"] + 1)

        bnt_alert = "Language referral recommended" if bnt_level in ("moderate", "severe") else ""
        ts = (now - timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%dT%H:%M:%S-06:00")

        records.append((pid, "BNT", json.dumps(bnt_items), bnt_score, 60.0, bnt_interp, bnt_level, bnt_alert, "SLP_Auto", ts, ts))

        # --- WAB (Western Aphasia Battery) — Aphasia Quotient ---
        # AQ range 0-100, normal ≥93.8, mild 76-93.7, moderate 51-75, severe ≤50
        wab_aq = max(25, min(100, round(100 * (1 - deficit * 1.2) - random.gauss(0, 5), 1)))
        # Sub-scores: Spontaneous Speech (20), Comprehension (10), Repetition (10), Naming (10) → AQ = (sum/50)*100
        sp_speech = max(0, min(20, round(20 * (wab_aq / 100) + random.gauss(0, 1), 1)))
        comprehension = max(0, min(10, round(10 * (wab_aq / 100) + random.gauss(0, 0.5), 1)))
        repetition = max(0, min(10, round(10 * (wab_aq / 100) + random.gauss(0, 0.8), 1)))
        naming = max(0, min(10, round(10 * (wab_aq / 100) + random.gauss(0, 0.6), 1)))

        if wab_aq >= 93.8:
            wab_interp, wab_level = "No aphasia", "normal"
        elif wab_aq >= 76:
            wab_interp, wab_level = "Mild aphasia — subtle word-finding or sentence formulation difficulty", "mild"
        elif wab_aq >= 51:
            wab_interp, wab_level = "Moderate aphasia — functional communication impaired", "moderate"
        else:
            wab_interp, wab_level = "Severe aphasia — significantly limited verbal output", "severe"

        wab_items = {
            "spontaneous_speech": sp_speech,
            "auditory_comprehension": comprehension,
            "repetition": repetition,
            "naming_word_finding": naming,
            "aphasia_quotient": wab_aq,
            "aphasia_type": _classify_aphasia_type(sp_speech, comprehension, repetition, naming)
        }
        wab_alert = "Aphasia therapy consultation recommended" if wab_level in ("moderate", "severe") else ""

        records.append((pid, "WAB", json.dumps(wab_items), wab_aq, 100.0, wab_interp, wab_level, wab_alert, "SLP_Auto", ts, ts))

        # --- Verbal Fluency (phonemic + semantic) ---
        # Normal: phonemic ≥12 words/min, semantic ≥15 words/min
        phon_score = max(3, int(15 * (1 - deficit) - random.gauss(0, 3)))
        sem_score = max(4, int(20 * (1 - deficit) - random.gauss(0, 3)))
        vf_total = phon_score + sem_score

        if phon_score >= 12 and sem_score >= 15:
            vf_interp, vf_level = "Normal verbal fluency", "normal"
        elif phon_score >= 8 or sem_score >= 10:
            vf_interp, vf_level = "Mildly reduced verbal fluency", "mild"
        else:
            vf_interp, vf_level = "Significantly reduced verbal fluency — frontal/temporal involvement likely", "moderate"

        vf_items = {
            "phonemic_f": random.randint(max(1, phon_score - 3), phon_score),
            "phonemic_a": random.randint(max(1, phon_score - 4), phon_score),
            "phonemic_s": random.randint(max(1, phon_score - 3), phon_score),
            "phonemic_total": phon_score,
            "semantic_animals": random.randint(max(2, sem_score - 4), sem_score + 2),
            "semantic_fruits": random.randint(max(2, sem_score - 5), sem_score),
            "semantic_total": sem_score,
            "perseverations": random.randint(0, 3),
            "intrusions": random.randint(0, 2),
            "clustering_score": round(random.uniform(1.5, 4.0), 1),
            "switching_score": round(random.uniform(5, 18), 1)
        }
        vf_alert = "Executive-language function assessment recommended" if vf_level == "moderate" else ""

        records.append((pid, "VERBAL_FLUENCY", json.dumps(vf_items), vf_total, 40.0, vf_interp, vf_level, vf_alert, "SLP_Auto", ts, ts))

        # --- MASA (Modified Mann Assessment of Swallowing Ability) ---
        # 200 max, ≥170 = normal, 149-169 = mild dysphagia, 130-148 = moderate, <130 = severe
        # Most epilepsy patients have normal swallowing unless post-ictal or on heavy sedation
        masa_base = 185 if not left_lateral else 175
        masa_score = max(100, min(200, int(masa_base - random.gauss(0, 12))))

        if masa_score >= 170:
            masa_interp, masa_level = "Normal swallowing function", "normal"
        elif masa_score >= 149:
            masa_interp, masa_level = "Mild dysphagia risk — diet modification may be needed post-ictally", "mild"
        elif masa_score >= 130:
            masa_interp, masa_level = "Moderate dysphagia — aspiration risk during/after seizures", "moderate"
        else:
            masa_interp, masa_level = "Severe dysphagia — high aspiration risk, modified diet required", "severe"

        masa_items = {
            "alertness": random.randint(8, 10),
            "cooperation": random.randint(8, 10),
            "respiration": random.randint(8, 10),
            "oral_motor": random.randint(max(5, masa_score // 25), 10),
            "tongue_movement": random.randint(max(5, masa_score // 22), 10),
            "gag_reflex": random.randint(7, 10),
            "voluntary_cough": random.randint(max(5, masa_score // 23), 10),
            "palate_movement": random.randint(7, 10),
            "bolus_clearance": random.randint(max(5, masa_score // 22), 10),
            "pharyngeal_response": random.randint(max(5, masa_score // 22), 10),
            "post_ictal_risk_flag": left_lateral or (age or 35) > 50,
            "rescue_med_aspiration_risk": random.choice([True, False]) if left_lateral else False
        }
        masa_alert = "Swallowing safety evaluation recommended" if masa_level in ("moderate", "severe") else ""

        records.append((pid, "MASA", json.dumps(masa_items), masa_score, 200.0, masa_interp, masa_level, masa_alert, "SLP_Auto", ts, ts))

    # Insert all
    c.executemany(
        """INSERT INTO assessments (patient_id, instrument, answers_json, score, max_score,
           interpretation, level, alert, examiner, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        records
    )
    c.commit()
    c.close()
    print(f"Created {len(records)} SLP assessment records for {len(patients)} patients.")


def _classify_aphasia_type(sp, comp, rep, nam):
    """Classify aphasia type based on WAB sub-scores (simplified)."""
    fluent = sp >= 10
    comp_ok = comp >= 6
    rep_ok = rep >= 6

    if fluent and comp_ok and rep_ok:
        return "No aphasia"
    elif fluent and comp_ok and not rep_ok:
        return "Conduction"
    elif fluent and not comp_ok:
        return "Wernicke" if not rep_ok else "Transcortical Sensory"
    elif not fluent and comp_ok:
        return "Broca" if not rep_ok else "Transcortical Motor"
    else:
        return "Global" if not rep_ok else "Mixed Transcortical"


if __name__ == "__main__":
    create_slp_data()
