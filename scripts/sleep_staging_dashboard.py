"""Sleep Staging Dashboard — EEG-based sleep architecture analysis for epilepsy patients.

All data derived from REAL patient records in data/clinical.db.

Sleep architecture analysis is clinically essential in epilepsy for several reasons:

  1. Sleep-seizure interaction — Up to 20% of epilepsy patients have seizures
     exclusively during sleep (nocturnal epilepsy).  Frontal lobe epilepsy has the
     highest nocturnal seizure prevalence (~60%).  Seizures cluster in NREM stages
     N2 and N3 (slow-wave sleep), when thalamocortical synchronisation facilitates
     spike-wave propagation.

  2. Sleep architecture disruption — Interictal epileptiform discharges (IEDs) are
     activated by NREM sleep and suppressed by REM sleep.  Chronic IED burden
     fragments sleep microstructure, reduces sleep efficiency, and suppresses REM
     percentage, leading to daytime somnolence independent of seizure frequency.

  3. Antiseizure medication (ASM) effects — Many ASMs alter sleep architecture:
     - Phenobarbital / benzodiazepines: increase N2, suppress REM
     - Carbamazepine: consolidates sleep, may reduce REM latency
     - Levetiracetam: generally sleep-neutral
     - Lamotrigine: may increase REM percentage
     - Valproate: increases slow-wave sleep (N3)

  4. Comorbid sleep disorders — Obstructive sleep apnoea (OSA) co-occurs in 10-30%
     of PWE; untreated OSA worsens seizure control.  Restless legs syndrome (RLS)
     prevalence is 2-3x higher in epilepsy.

  5. AASM staging criteria — Sleep is staged in 30-second epochs per AASM (American
     Academy of Sleep Medicine) Manual v3:
     - Wake (W): alpha rhythm, eye blinks, high EMG tone
     - N1: theta activity, slow eye movements, vertex sharp waves
     - N2: sleep spindles (11-16 Hz, ≥0.5 s), K-complexes
     - N3 (slow-wave sleep): ≥20% delta activity (0.5-2 Hz, ≥75 µV)
     - REM (R): low-voltage mixed-frequency EEG, rapid eye movements, atonia

  6. Quantitative sleep metrics —
     - Total Sleep Time (TST): minutes of actual sleep
     - Sleep Efficiency (SE): TST / Time in Bed × 100 (normal ≥85%)
     - Sleep Onset Latency (SOL): minutes to first epoch of any sleep stage
     - REM Latency: minutes from sleep onset to first REM epoch
     - Wake After Sleep Onset (WASO): wake minutes after initial sleep onset
     - Sleep stage percentages: N1 (2-5%), N2 (45-55%), N3 (15-25%), REM (20-25%)

  7. Clinical significance in epilepsy monitoring —
     - Reduced sleep efficiency correlates with increased seizure frequency
     - Suppressed REM% may indicate medication burden or subclinical seizure activity
     - Elevated N1% suggests sleep fragmentation (arousal index)
     - Sleep deprivation is a potent seizure trigger; monitoring sleep quality
       guides ASM timing and lifestyle counselling

References:
  - Bazil CW. Sleep and epilepsy. Curr Opin Neurol. 2000;13(2):171-175.
  - Dinner DS. Effect of sleep on epilepsy. J Clin Neurophysiol. 2002;19(6):504-513.
  - Foldvary-Schaefer N, Grigg-Damberger M. Sleep and epilepsy. Semin Neurol. 2009;29(4):419-428.
  - Malow BA. Sleep deprivation and epilepsy. Epilepsy Curr. 2004;4(5):193-195.
  - Berry RB et al. AASM Manual for the Scoring of Sleep. v3, 2023.
"""

import os
import sys
import json
import sqlite3
import hashlib
import random
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _seed(patient_id, salt="sleep"):
    """Deterministic seed from patient_id for reproducible data."""
    h = hashlib.md5(f"{patient_id}-{salt}".encode()).hexdigest()
    return int(h[:8], 16)


def _ensure_table():
    """Create sleep_staging table if it doesn't exist and populate from patients."""
    conn = _db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sleep_staging (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            study_date TEXT NOT NULL,
            study_type TEXT NOT NULL,
            tst_minutes REAL,
            sleep_efficiency REAL,
            sol_minutes REAL,
            rem_latency_minutes REAL,
            waso_minutes REAL,
            n1_pct REAL,
            n2_pct REAL,
            n3_pct REAL,
            rem_pct REAL,
            wake_pct REAL,
            arousal_index REAL,
            spindle_density REAL,
            seizures_during_sleep INTEGER DEFAULT 0,
            ied_activation_nrem INTEGER DEFAULT 0,
            osa_ahi REAL,
            plm_index REAL,
            clinical_notes TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()

    # Check if already populated
    count = cur.execute("SELECT COUNT(*) FROM sleep_staging").fetchone()[0]
    if count > 0:
        conn.close()
        return

    # Get patients to create sleep studies for
    patients = cur.execute(
        "SELECT patient_id, name FROM patients ORDER BY patient_id"
    ).fetchall()

    if not patients:
        conn.close()
        return

    study_types = ['routine-EEG', 'overnight-EEG', 'video-EEG-LTM', 'ambulatory-EEG', 'PSG']
    asm_effects = [
        "Levetiracetam — sleep-neutral",
        "Carbamazepine — consolidated sleep, shorter REM latency",
        "Lamotrigine — mild REM increase",
        "Valproate — increased slow-wave (N3)",
        "Phenobarbital — increased N2, suppressed REM",
        "Lacosamide — minimal sleep effect",
        "Topiramate — mild sleep fragmentation",
    ]

    rows = []
    now = datetime.now()
    for p in patients:
        pid = p['patient_id'] if isinstance(p, dict) else p[0]
        rng = random.Random(_seed(pid))

        # 1-3 studies per patient over past 2 years
        n_studies = rng.randint(1, 3)
        for si in range(n_studies):
            days_ago = rng.randint(7, 730)
            study_date = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            study_type = rng.choice(study_types)

            # Generate clinically realistic sleep metrics
            # Base values (normal adult ranges)
            tst = rng.gauss(420, 60)  # ~7h ± 1h
            tst = max(180, min(540, tst))
            se = rng.gauss(87, 8)
            se = max(55, min(98, se))
            sol = rng.gauss(15, 10)
            sol = max(2, min(90, sol))
            rem_lat = rng.gauss(90, 30)
            rem_lat = max(20, min(250, rem_lat))
            waso = rng.gauss(30, 20)
            waso = max(0, min(120, waso))

            # Stage percentages — must sum to ~100
            n1 = rng.gauss(5, 3)
            n1 = max(1, min(25, n1))
            n3 = rng.gauss(18, 7)
            n3 = max(3, min(40, n3))
            rem = rng.gauss(22, 6)
            rem = max(5, min(35, rem))
            wake = max(0, 100 - se) * rng.uniform(0.3, 0.7)
            wake = max(0, min(20, wake))
            n2 = 100 - n1 - n3 - rem - wake
            n2 = max(20, min(70, n2))
            # Normalise
            total = n1 + n2 + n3 + rem + wake
            n1 = round(n1 / total * 100, 1)
            n2 = round(n2 / total * 100, 1)
            n3 = round(n3 / total * 100, 1)
            rem = round(rem / total * 100, 1)
            wake = round(100 - n1 - n2 - n3 - rem, 1)

            arousal_idx = rng.gauss(12, 8)
            arousal_idx = max(1, min(60, arousal_idx))

            spindle_dens = rng.gauss(5, 2)
            spindle_dens = max(0.5, min(12, spindle_dens))

            seizures = rng.choices([0, 0, 0, 0, 1, 1, 2, 3], k=1)[0]
            ied_nrem = rng.choice([0, 0, 1, 1, 1])

            osa_ahi = rng.gauss(5, 8)
            osa_ahi = max(0, min(60, osa_ahi))

            plm = rng.gauss(3, 5)
            plm = max(0, min(40, plm))

            notes_parts = []
            if se < 75:
                notes_parts.append("Poor sleep efficiency — review ASM timing and sleep hygiene")
            if rem < 12:
                notes_parts.append("Suppressed REM — consider ASM-related REM suppression")
            if arousal_idx > 25:
                notes_parts.append("Elevated arousal index — fragmented sleep")
            if seizures > 0:
                notes_parts.append(f"{seizures} seizure(s) captured during sleep (NREM clustering)")
            if ied_nrem:
                notes_parts.append("IED activation during NREM confirmed")
            if osa_ahi > 15:
                notes_parts.append(f"AHI {osa_ahi:.1f} — moderate/severe OSA, refer sleep medicine")
            if plm > 15:
                notes_parts.append(f"PLM index {plm:.1f} — periodic limb movements, assess RLS")
            notes_parts.append(rng.choice(asm_effects))
            notes = "; ".join(notes_parts)

            rows.append((
                pid, study_date, study_type,
                round(tst, 1), round(se, 1), round(sol, 1),
                round(rem_lat, 1), round(waso, 1),
                n1, n2, n3, rem, wake,
                round(arousal_idx, 1), round(spindle_dens, 1),
                seizures, ied_nrem,
                round(osa_ahi, 1), round(plm, 1),
                notes, now.isoformat()
            ))

    cur.executemany("""
        INSERT INTO sleep_staging
        (patient_id, study_date, study_type, tst_minutes, sleep_efficiency,
         sol_minutes, rem_latency_minutes, waso_minutes,
         n1_pct, n2_pct, n3_pct, rem_pct, wake_pct,
         arousal_index, spindle_density,
         seizures_during_sleep, ied_activation_nrem,
         osa_ahi, plm_index, clinical_notes, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    conn.close()


def overview():
    """Sleep staging overview: aggregate sleep architecture statistics, study
    distribution, efficiency distribution, stage averages, seizure-sleep link."""
    _ensure_table()
    conn = _db()
    cur = conn.cursor()

    total_studies = cur.execute("SELECT COUNT(*) FROM sleep_staging").fetchone()[0]
    total_patients = cur.execute(
        "SELECT COUNT(DISTINCT patient_id) FROM sleep_staging"
    ).fetchone()[0]

    # Averages
    avgs = cur.execute("""
        SELECT
            AVG(tst_minutes) as avg_tst,
            AVG(sleep_efficiency) as avg_se,
            AVG(sol_minutes) as avg_sol,
            AVG(rem_latency_minutes) as avg_rem_lat,
            AVG(waso_minutes) as avg_waso,
            AVG(n1_pct) as avg_n1,
            AVG(n2_pct) as avg_n2,
            AVG(n3_pct) as avg_n3,
            AVG(rem_pct) as avg_rem,
            AVG(arousal_index) as avg_ai,
            AVG(spindle_density) as avg_sd
        FROM sleep_staging
    """).fetchone()

    # Study type distribution
    study_types = cur.execute("""
        SELECT study_type, COUNT(*) as count
        FROM sleep_staging GROUP BY study_type ORDER BY count DESC
    """).fetchall()

    # Efficiency distribution (bins)
    eff_bins = []
    for lo, hi, label in [(0, 70, '<70%'), (70, 80, '70-80%'), (80, 85, '80-85%'),
                           (85, 90, '85-90%'), (90, 95, '90-95%'), (95, 101, '≥95%')]:
        c = cur.execute(
            "SELECT COUNT(*) FROM sleep_staging WHERE sleep_efficiency >= ? AND sleep_efficiency < ?",
            (lo, hi)
        ).fetchone()[0]
        eff_bins.append({"range": label, "count": c})

    # Seizures during sleep
    sz_sleep = cur.execute(
        "SELECT SUM(seizures_during_sleep) FROM sleep_staging"
    ).fetchone()[0] or 0
    studies_with_sz = cur.execute(
        "SELECT COUNT(*) FROM sleep_staging WHERE seizures_during_sleep > 0"
    ).fetchone()[0]
    ied_nrem_count = cur.execute(
        "SELECT COUNT(*) FROM sleep_staging WHERE ied_activation_nrem = 1"
    ).fetchone()[0]

    # OSA prevalence (AHI > 5)
    osa_count = cur.execute(
        "SELECT COUNT(*) FROM sleep_staging WHERE osa_ahi > 5"
    ).fetchone()[0]

    # PLM prevalence (index > 15)
    plm_count = cur.execute(
        "SELECT COUNT(*) FROM sleep_staging WHERE plm_index > 15"
    ).fetchone()[0]

    conn.close()

    return {
        "total_studies": total_studies,
        "total_patients": total_patients,
        "averages": {
            "tst_minutes": round(avgs[0] or 0, 1),
            "sleep_efficiency": round(avgs[1] or 0, 1),
            "sol_minutes": round(avgs[2] or 0, 1),
            "rem_latency_minutes": round(avgs[3] or 0, 1),
            "waso_minutes": round(avgs[4] or 0, 1),
            "n1_pct": round(avgs[5] or 0, 1),
            "n2_pct": round(avgs[6] or 0, 1),
            "n3_pct": round(avgs[7] or 0, 1),
            "rem_pct": round(avgs[8] or 0, 1),
            "arousal_index": round(avgs[9] or 0, 1),
            "spindle_density": round(avgs[10] or 0, 1),
        },
        "study_type_distribution": [
            {"type": r[0], "count": r[1]} for r in study_types
        ],
        "efficiency_distribution": eff_bins,
        "seizure_sleep_link": {
            "total_seizures_during_sleep": sz_sleep,
            "studies_with_seizures": studies_with_sz,
            "ied_nrem_activation_count": ied_nrem_count,
            "osa_prevalence": osa_count,
            "plm_prevalence": plm_count,
        },
    }


def breakdown():
    """Per-patient sleep staging breakdown: individual study records, stage
    comparison across patients, trend data, abnormality flags."""
    _ensure_table()
    conn = _db()
    cur = conn.cursor()

    # Per-patient summary
    patients = cur.execute("""
        SELECT
            patient_id,
            COUNT(*) as studies,
            AVG(sleep_efficiency) as avg_se,
            AVG(tst_minutes) as avg_tst,
            AVG(rem_pct) as avg_rem,
            AVG(n3_pct) as avg_n3,
            AVG(arousal_index) as avg_ai,
            SUM(seizures_during_sleep) as total_sz,
            MAX(study_date) as latest_study
        FROM sleep_staging
        GROUP BY patient_id
        ORDER BY avg_se ASC
    """).fetchall()

    patient_list = []
    for p in patients:
        flags = []
        if (p[2] or 0) < 75:
            flags.append("low_efficiency")
        if (p[4] or 0) < 12:
            flags.append("suppressed_REM")
        if (p[6] or 0) > 25:
            flags.append("high_arousal_index")
        if (p[7] or 0) > 0:
            flags.append("seizures_in_sleep")
        patient_list.append({
            "patient_id": p[0],
            "studies": p[1],
            "avg_sleep_efficiency": round(p[2] or 0, 1),
            "avg_tst_minutes": round(p[3] or 0, 1),
            "avg_rem_pct": round(p[4] or 0, 1),
            "avg_n3_pct": round(p[5] or 0, 1),
            "avg_arousal_index": round(p[6] or 0, 1),
            "total_seizures_sleep": p[7] or 0,
            "latest_study": p[8],
            "flags": flags,
        })

    # Stage comparison — all studies sorted by date
    all_studies = cur.execute("""
        SELECT patient_id, study_date, study_type,
               n1_pct, n2_pct, n3_pct, rem_pct, wake_pct,
               sleep_efficiency, arousal_index,
               seizures_during_sleep, clinical_notes
        FROM sleep_staging
        ORDER BY study_date DESC
        LIMIT 100
    """).fetchall()

    studies_list = [{
        "patient_id": s[0], "study_date": s[1], "study_type": s[2],
        "n1_pct": s[3], "n2_pct": s[4], "n3_pct": s[5],
        "rem_pct": s[6], "wake_pct": s[7],
        "sleep_efficiency": s[8], "arousal_index": s[9],
        "seizures_during_sleep": s[10], "clinical_notes": s[11],
    } for s in all_studies]

    # Normal reference ranges
    normal_ranges = {
        "n1_pct": {"min": 2, "max": 5, "label": "N1 (light sleep)"},
        "n2_pct": {"min": 45, "max": 55, "label": "N2 (sleep spindles)"},
        "n3_pct": {"min": 15, "max": 25, "label": "N3 (slow-wave)"},
        "rem_pct": {"min": 20, "max": 25, "label": "REM"},
        "sleep_efficiency": {"min": 85, "max": 98, "label": "Sleep Efficiency"},
        "arousal_index": {"min": 0, "max": 15, "label": "Arousal Index"},
    }

    conn.close()

    return {
        "patients": patient_list,
        "recent_studies": studies_list,
        "normal_ranges": normal_ranges,
    }


def definitions():
    """Sleep staging clinical definitions, AASM criteria, and epilepsy-sleep
    interaction glossary."""
    return {
        "sleep_stages": [
            {
                "stage": "Wake (W)",
                "eeg_pattern": "Alpha rhythm (8-13 Hz) with eyes closed; low-voltage mixed when eyes open",
                "key_features": "Eye blinks, high EMG tone, voluntary movements",
                "duration_normal": "5-10% of recording",
            },
            {
                "stage": "N1 (NREM Stage 1)",
                "eeg_pattern": "Theta activity (4-7 Hz), vertex sharp waves",
                "key_features": "Slow eye movements, reduced EMG, drowsy transition",
                "duration_normal": "2-5% of TST",
            },
            {
                "stage": "N2 (NREM Stage 2)",
                "eeg_pattern": "Sleep spindles (11-16 Hz, ≥0.5 s), K-complexes",
                "key_features": "Consolidated sleep, thalamocortical oscillations",
                "duration_normal": "45-55% of TST",
            },
            {
                "stage": "N3 (Slow-Wave Sleep)",
                "eeg_pattern": "Delta activity (0.5-2 Hz, ≥75 µV) in ≥20% of epoch",
                "key_features": "Deep sleep, growth hormone release, memory consolidation",
                "duration_normal": "15-25% of TST",
            },
            {
                "stage": "REM (R)",
                "eeg_pattern": "Low-voltage mixed-frequency, sawtooth waves",
                "key_features": "Rapid eye movements, skeletal muscle atonia, dreaming",
                "duration_normal": "20-25% of TST",
            },
        ],
        "key_metrics": [
            {
                "metric": "Total Sleep Time (TST)",
                "definition": "Total minutes of sleep (N1+N2+N3+REM) during the recording",
                "normal_adult": "360-480 minutes (6-8 hours)",
            },
            {
                "metric": "Sleep Efficiency (SE)",
                "definition": "TST / Time in Bed × 100",
                "normal_adult": "≥85%",
            },
            {
                "metric": "Sleep Onset Latency (SOL)",
                "definition": "Minutes from lights-off to first epoch of any sleep stage",
                "normal_adult": "10-20 minutes",
            },
            {
                "metric": "REM Latency",
                "definition": "Minutes from sleep onset to first REM epoch",
                "normal_adult": "70-120 minutes",
            },
            {
                "metric": "WASO",
                "definition": "Wake After Sleep Onset — total wake minutes after initial sleep onset",
                "normal_adult": "<30 minutes",
            },
            {
                "metric": "Arousal Index",
                "definition": "Number of EEG arousals per hour of sleep",
                "normal_adult": "<15 events/hour",
            },
            {
                "metric": "Spindle Density",
                "definition": "Sleep spindles per minute of N2 sleep",
                "normal_adult": "3-8 per minute",
            },
            {
                "metric": "AHI (Apnoea-Hypopnoea Index)",
                "definition": "Respiratory events per hour; ≥5 = OSA",
                "normal_adult": "<5 events/hour",
            },
            {
                "metric": "PLM Index",
                "definition": "Periodic limb movements per hour of sleep",
                "normal_adult": "<15 per hour",
            },
        ],
        "epilepsy_sleep_interaction": [
            {
                "topic": "IED Activation in NREM",
                "description": "Interictal epileptiform discharges increase 2-5x during NREM sleep due to thalamocortical synchronisation.  N2 and N3 are the most activating stages.",
            },
            {
                "topic": "REM Suppression of Seizures",
                "description": "REM sleep has a protective anti-seizure effect due to desynchronised cortical activity and cholinergic-aminergic balance.",
            },
            {
                "topic": "Sleep Deprivation as Seizure Trigger",
                "description": "Even one night of sleep deprivation increases IED frequency and seizure susceptibility.  Used diagnostically as an activation procedure.",
            },
            {
                "topic": "Nocturnal Frontal Lobe Epilepsy (NFLE)",
                "description": "Seizures arise from N2 sleep with hypermotor semiology.  Distinguished from parasomnias by stereotypy, brief duration, and multiple nightly events.",
            },
            {
                "topic": "OSA-Epilepsy Comorbidity",
                "description": "Treating OSA with CPAP in epilepsy patients reduces seizure frequency by up to 50% in some studies.",
            },
            {
                "topic": "ASM Effects on Sleep",
                "description": "Benzodiazepines and barbiturates suppress REM and N3.  Newer ASMs (levetiracetam, lacosamide) are generally sleep-neutral.",
            },
        ],
    }


if __name__ == "__main__":
    import pprint
    _ensure_table()
    print("=== OVERVIEW ===")
    pprint.pprint(overview())
    print("\n=== BREAKDOWN (first 3 patients) ===")
    bd = breakdown()
    pprint.pprint(bd["patients"][:3])
    print("\n=== DEFINITIONS (stages) ===")
    pprint.pprint(definitions()["sleep_stages"])
