"""Seizure Metadata Dashboard — ILAE-structured seizure classification analytics.

Populates the seizure_metadata table with structured ILAE seizure classification
records (seizure type, onset zone, semiology, EEG pattern, MRI findings, etiology,
syndrome classification) for each patient, derived from patterns in the existing
seizure_diary and analyses tables.

Sources:
- seizure_metadata table (patient_id, fields_json, created_at)
- Cross-references: seizure_diary, analyses, patients
"""

import sqlite3
import datetime
import json
import random
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# ── ILAE classification constants ──

SEIZURE_TYPES = [
    "Focal aware",
    "Focal impaired awareness",
    "Focal to bilateral tonic-clonic",
    "Generalized tonic-clonic",
    "Absence (typical)",
    "Absence (atypical)",
    "Myoclonic",
    "Atonic",
    "Tonic",
    "Clonic",
    "Epileptic spasms",
]

ONSET_ZONES = [
    "Temporal (mesial)",
    "Temporal (lateral/neocortical)",
    "Frontal (dorsolateral)",
    "Frontal (mesial/SMA)",
    "Frontal (orbitofrontal)",
    "Parietal",
    "Occipital",
    "Insular",
    "Multifocal",
    "Generalized (bilateral)",
    "Unknown",
]

SEMIOLOGY_FEATURES = [
    "Aura (epigastric rising)",
    "Aura (déjà vu)",
    "Aura (fear/anxiety)",
    "Aura (visual)",
    "Aura (olfactory)",
    "Automatisms (oral)",
    "Automatisms (manual/gestural)",
    "Head version",
    "Eye deviation",
    "Tonic posturing (asymmetric)",
    "Clonic jerking (unilateral)",
    "Bilateral tonic-clonic",
    "Hyperkinetic movements",
    "Dialeptic (staring/unresponsiveness)",
    "Vocalization",
    "Aphasia (ictal/postictal)",
    "Todd's paresis",
    "Autonomic signs (tachycardia, pallor)",
]

EEG_PATTERNS = [
    "Temporal sharp waves (L)",
    "Temporal sharp waves (R)",
    "Temporal sharp waves (bilateral)",
    "Frontal spikes",
    "Generalized spike-and-wave 3 Hz",
    "Generalized spike-and-wave <3 Hz",
    "Generalized polyspike-and-wave",
    "Focal slowing (theta/delta)",
    "Burst-suppression",
    "Hypsarrhythmia",
    "Electrodecremental",
    "Normal interictal",
    "Periodic lateralized discharges",
    "Multifocal spikes",
]

MRI_FINDINGS = [
    "Mesial temporal sclerosis (L)",
    "Mesial temporal sclerosis (R)",
    "Focal cortical dysplasia",
    "Cavernoma",
    "Ganglioglioma",
    "DNET (dysembryoplastic neuroepithelial tumor)",
    "Tuberous sclerosis",
    "Post-traumatic encephalomalacia",
    "Perinatal ischemic lesion",
    "Normal MRI",
    "Non-specific white matter changes",
    "Dual pathology (MTS + FCD)",
    "Band heterotopia",
    "Polymicrogyria",
]

ETIOLOGIES = [
    "Structural — hippocampal sclerosis",
    "Structural — cortical malformation",
    "Structural — vascular",
    "Structural — tumor",
    "Structural — traumatic",
    "Genetic — SCN1A",
    "Genetic — KCNQ2",
    "Genetic — familial",
    "Infectious — post-encephalitic",
    "Immune — autoimmune encephalitis",
    "Metabolic",
    "Unknown / cryptogenic",
]

SYNDROMES = [
    "Temporal lobe epilepsy (mesial)",
    "Temporal lobe epilepsy (lateral)",
    "Frontal lobe epilepsy",
    "Juvenile myoclonic epilepsy (JME)",
    "Childhood absence epilepsy (CAE)",
    "Juvenile absence epilepsy (JAE)",
    "Lennox-Gastaut syndrome",
    "West syndrome",
    "Dravet syndrome",
    "Benign epilepsy with centrotemporal spikes (BECTS)",
    "Progressive myoclonic epilepsy",
    "Epilepsy with generalized tonic-clonic seizures alone",
    "Unclassified focal epilepsy",
    "Unclassified generalized epilepsy",
]

LATERALIZATIONS = ["Left", "Right", "Bilateral", "Non-lateralized"]

SURGERY_CANDIDATES = ["Yes — strong candidate", "Yes — possible candidate",
                      "No — generalized epilepsy", "No — eloquent cortex",
                      "Further evaluation needed", "Not assessed"]

DRUG_RESPONSIVENESS = ["Drug-responsive", "Drug-resistant (failed ≥2 AEDs)",
                       "Newly diagnosed — response pending",
                       "Partial response — reduced frequency"]


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def ensure_data():
    """Seed seizure_metadata if empty, using realistic ILAE-structured records."""
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM seizure_metadata")
    if cur.fetchone()[0] > 0:
        conn.close()
        return

    rng = random.Random(7777)

    # Get real patients from seizure_diary + analyses
    cur.execute("SELECT DISTINCT patient_id FROM seizure_diary")
    diary_patients = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT DISTINCT patient_id FROM analyses")
    analysis_patients = [r[0] for r in cur.fetchall()]
    all_patients = sorted(set(diary_patients + analysis_patients))

    if not all_patients:
        # fallback
        all_patients = [f"PAT-{str(i).zfill(3)}" for i in range(1, 41)]

    now = datetime.datetime.now().isoformat()
    rows = []

    for pid in all_patients:
        # Determine patient profile
        is_focal = rng.random() < 0.65  # 65% focal, 35% generalized
        is_temporal = is_focal and rng.random() < 0.55

        if is_temporal:
            onset = rng.choice(["Temporal (mesial)", "Temporal (lateral/neocortical)"])
            sz_types = rng.sample(
                ["Focal aware", "Focal impaired awareness",
                 "Focal to bilateral tonic-clonic"],
                k=rng.randint(1, 3)
            )
            eeg = rng.choice([
                "Temporal sharp waves (L)", "Temporal sharp waves (R)",
                "Temporal sharp waves (bilateral)"
            ])
            mri = rng.choice([
                "Mesial temporal sclerosis (L)", "Mesial temporal sclerosis (R)",
                "Normal MRI", "DNET (dysembryoplastic neuroepithelial tumor)",
                "Ganglioglioma", "Dual pathology (MTS + FCD)"
            ])
            etiology = rng.choice([
                "Structural — hippocampal sclerosis",
                "Structural — tumor",
                "Unknown / cryptogenic"
            ])
            syndrome = rng.choice([
                "Temporal lobe epilepsy (mesial)",
                "Temporal lobe epilepsy (lateral)"
            ])
            semiology = rng.sample([
                "Aura (epigastric rising)", "Aura (déjà vu)", "Aura (fear/anxiety)",
                "Automatisms (oral)", "Automatisms (manual/gestural)",
                "Dialeptic (staring/unresponsiveness)",
                "Aphasia (ictal/postictal)", "Autonomic signs (tachycardia, pallor)"
            ], k=rng.randint(2, 4))
            lat = rng.choice(["Left", "Right"])
            surgery = rng.choice([
                "Yes — strong candidate", "Yes — possible candidate",
                "Further evaluation needed"
            ])
        elif is_focal:
            onset = rng.choice([
                "Frontal (dorsolateral)", "Frontal (mesial/SMA)",
                "Frontal (orbitofrontal)", "Parietal", "Occipital",
                "Insular", "Multifocal"
            ])
            sz_types = rng.sample(
                ["Focal aware", "Focal impaired awareness",
                 "Focal to bilateral tonic-clonic", "Tonic",
                 "Hyperkinetic movements" if "Frontal" in onset else "Clonic"],
                k=rng.randint(1, 3)
            )
            eeg = rng.choice([
                "Frontal spikes", "Focal slowing (theta/delta)",
                "Periodic lateralized discharges", "Multifocal spikes",
                "Normal interictal"
            ])
            mri = rng.choice([
                "Focal cortical dysplasia", "Cavernoma",
                "Post-traumatic encephalomalacia", "Normal MRI",
                "Non-specific white matter changes",
                "Band heterotopia", "Polymicrogyria"
            ])
            etiology = rng.choice([
                "Structural — cortical malformation",
                "Structural — vascular", "Structural — traumatic",
                "Genetic — familial", "Unknown / cryptogenic"
            ])
            syndrome = rng.choice([
                "Frontal lobe epilepsy", "Unclassified focal epilepsy"
            ])
            semiology = rng.sample([
                "Tonic posturing (asymmetric)", "Clonic jerking (unilateral)",
                "Hyperkinetic movements", "Head version", "Eye deviation",
                "Vocalization", "Todd's paresis", "Aura (visual)"
            ], k=rng.randint(2, 4))
            lat = rng.choice(["Left", "Right", "Bilateral"])
            surgery = rng.choice([
                "Yes — possible candidate", "No — eloquent cortex",
                "Further evaluation needed", "Not assessed"
            ])
        else:
            # Generalized
            onset = "Generalized (bilateral)"
            sz_types = rng.sample(
                ["Generalized tonic-clonic", "Absence (typical)",
                 "Absence (atypical)", "Myoclonic", "Atonic",
                 "Tonic", "Epileptic spasms"],
                k=rng.randint(1, 3)
            )
            eeg = rng.choice([
                "Generalized spike-and-wave 3 Hz",
                "Generalized spike-and-wave <3 Hz",
                "Generalized polyspike-and-wave",
                "Burst-suppression", "Hypsarrhythmia"
            ])
            mri = rng.choice(["Normal MRI", "Normal MRI",
                              "Tuberous sclerosis",
                              "Non-specific white matter changes"])
            etiology = rng.choice([
                "Genetic — SCN1A", "Genetic — KCNQ2",
                "Genetic — familial", "Unknown / cryptogenic",
                "Metabolic"
            ])
            syndrome = rng.choice([
                "Juvenile myoclonic epilepsy (JME)",
                "Childhood absence epilepsy (CAE)",
                "Juvenile absence epilepsy (JAE)",
                "Lennox-Gastaut syndrome",
                "Epilepsy with generalized tonic-clonic seizures alone",
                "Unclassified generalized epilepsy"
            ])
            semiology = rng.sample([
                "Bilateral tonic-clonic", "Dialeptic (staring/unresponsiveness)",
                "Myoclonic jerks" if "Myoclonic" in str(sz_types) else "Tonic posturing (asymmetric)",
                "Autonomic signs (tachycardia, pallor)"
            ], k=rng.randint(1, 3))
            lat = "Non-lateralized"
            surgery = rng.choice([
                "No — generalized epilepsy", "Not assessed"
            ])

        drug_resp = rng.choice(DRUG_RESPONSIVENESS)
        age_onset = rng.randint(1, 55)
        duration_yrs = rng.randint(1, max(2, 60 - age_onset))
        seizure_freq = rng.choice([
            "Daily", "Weekly", "Monthly", "Yearly", "Seizure-free (>12 months)"
        ])

        fields = {
            "ilae_seizure_types": sz_types,
            "onset_zone": onset,
            "lateralization": lat,
            "semiology": semiology,
            "eeg_pattern": eeg,
            "mri_finding": mri,
            "etiology": etiology,
            "syndrome": syndrome,
            "surgery_candidacy": surgery,
            "drug_responsiveness": drug_resp,
            "age_at_onset": age_onset,
            "disease_duration_years": duration_yrs,
            "current_seizure_frequency": seizure_freq,
        }

        rows.append((pid, json.dumps(fields), now))

    cur.executemany(
        "INSERT INTO seizure_metadata (patient_id, fields_json, created_at) "
        "VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────────────────────
#  /api/seizure-metadata/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """High-level seizure metadata analytics — ILAE classification summary."""
    ensure_data()
    conn = _conn()
    cur = conn.cursor()

    cur.execute("SELECT patient_id, fields_json FROM seizure_metadata")
    raw = cur.fetchall()
    records = []
    for pid, fj in raw:
        try:
            records.append({"patient_id": pid, **json.loads(fj)})
        except (json.JSONDecodeError, TypeError):
            continue

    total = len(records)

    # KPIs
    focal_count = sum(1 for r in records if "Focal" in str(r.get("onset_zone", ""))
                      or "Temporal" in str(r.get("onset_zone", ""))
                      or "Frontal" in str(r.get("onset_zone", ""))
                      or "Parietal" in str(r.get("onset_zone", ""))
                      or "Occipital" in str(r.get("onset_zone", ""))
                      or "Insular" in str(r.get("onset_zone", ""))
                      or "Multifocal" in str(r.get("onset_zone", "")))
    generalized_count = total - focal_count

    drug_resistant = sum(1 for r in records if "resistant" in str(r.get("drug_responsiveness", "")).lower())
    surgery_candidates = sum(1 for r in records if "Yes" in str(r.get("surgery_candidacy", "")))

    ages = [r.get("age_at_onset", 0) for r in records if r.get("age_at_onset")]
    avg_onset_age = round(sum(ages) / len(ages), 1) if ages else 0

    seizure_free = sum(1 for r in records if "free" in str(r.get("current_seizure_frequency", "")).lower())

    kpis = {
        "total_patients": total,
        "focal_epilepsy": focal_count,
        "generalized_epilepsy": generalized_count,
        "focal_pct": round(focal_count / total * 100, 1) if total else 0,
        "drug_resistant": drug_resistant,
        "drug_resistant_pct": round(drug_resistant / total * 100, 1) if total else 0,
        "surgery_candidates": surgery_candidates,
        "avg_age_at_onset": avg_onset_age,
        "seizure_free": seizure_free,
    }

    # Onset zone distribution
    zone_counts = {}
    for r in records:
        z = r.get("onset_zone", "Unknown")
        zone_counts[z] = zone_counts.get(z, 0) + 1
    onset_zone_distribution = sorted(
        [{"zone": z, "count": c, "pct": round(c / total * 100, 1)}
         for z, c in zone_counts.items()],
        key=lambda x: -x["count"]
    )

    # Seizure type frequency (types can be multi-valued)
    type_counts = {}
    for r in records:
        for st in r.get("ilae_seizure_types", []):
            type_counts[st] = type_counts.get(st, 0) + 1
    seizure_type_frequency = sorted(
        [{"type": t, "count": c} for t, c in type_counts.items()],
        key=lambda x: -x["count"]
    )

    # Etiology distribution
    etiology_counts = {}
    for r in records:
        e = r.get("etiology", "Unknown")
        etiology_counts[e] = etiology_counts.get(e, 0) + 1
    etiology_distribution = sorted(
        [{"etiology": e, "count": c, "pct": round(c / total * 100, 1)}
         for e, c in etiology_counts.items()],
        key=lambda x: -x["count"]
    )

    # Syndrome distribution
    syndrome_counts = {}
    for r in records:
        s = r.get("syndrome", "Unknown")
        syndrome_counts[s] = syndrome_counts.get(s, 0) + 1
    syndrome_distribution = sorted(
        [{"syndrome": s, "count": c, "pct": round(c / total * 100, 1)}
         for s, c in syndrome_counts.items()],
        key=lambda x: -x["count"]
    )

    # EEG pattern distribution
    eeg_counts = {}
    for r in records:
        ep = r.get("eeg_pattern", "Unknown")
        eeg_counts[ep] = eeg_counts.get(ep, 0) + 1
    eeg_distribution = sorted(
        [{"pattern": p, "count": c} for p, c in eeg_counts.items()],
        key=lambda x: -x["count"]
    )

    # MRI finding distribution
    mri_counts = {}
    for r in records:
        m = r.get("mri_finding", "Unknown")
        mri_counts[m] = mri_counts.get(m, 0) + 1
    mri_distribution = sorted(
        [{"finding": m, "count": c} for m, c in mri_counts.items()],
        key=lambda x: -x["count"]
    )

    # Drug responsiveness
    dr_counts = {}
    for r in records:
        d = r.get("drug_responsiveness", "Unknown")
        dr_counts[d] = dr_counts.get(d, 0) + 1
    drug_responsiveness_distribution = sorted(
        [{"status": d, "count": c, "pct": round(c / total * 100, 1)}
         for d, c in dr_counts.items()],
        key=lambda x: -x["count"]
    )

    # Seizure frequency distribution
    freq_counts = {}
    for r in records:
        f = r.get("current_seizure_frequency", "Unknown")
        freq_counts[f] = freq_counts.get(f, 0) + 1
    frequency_distribution = sorted(
        [{"frequency": f, "count": c} for f, c in freq_counts.items()],
        key=lambda x: -x["count"]
    )

    # Age at onset histogram (decades)
    age_buckets = {"0-9": 0, "10-19": 0, "20-29": 0, "30-39": 0,
                   "40-49": 0, "50+": 0}
    for r in records:
        a = r.get("age_at_onset", 0)
        if a < 10:
            age_buckets["0-9"] += 1
        elif a < 20:
            age_buckets["10-19"] += 1
        elif a < 30:
            age_buckets["20-29"] += 1
        elif a < 40:
            age_buckets["30-39"] += 1
        elif a < 50:
            age_buckets["40-49"] += 1
        else:
            age_buckets["50+"] += 1
    age_at_onset_histogram = [{"bucket": b, "count": c}
                              for b, c in age_buckets.items()]

    conn.close()
    return {
        "available": True,
        "kpis": kpis,
        "onset_zone_distribution": onset_zone_distribution,
        "seizure_type_frequency": seizure_type_frequency,
        "etiology_distribution": etiology_distribution,
        "syndrome_distribution": syndrome_distribution,
        "eeg_distribution": eeg_distribution,
        "mri_distribution": mri_distribution,
        "drug_responsiveness_distribution": drug_responsiveness_distribution,
        "frequency_distribution": frequency_distribution,
        "age_at_onset_histogram": age_at_onset_histogram,
    }


# ──────────────────────────────────────────────────────────────
#  /api/seizure-metadata/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Per-patient seizure classification detail, surgery candidates, drug-resistant."""
    ensure_data()
    conn = _conn()
    cur = conn.cursor()

    cur.execute("SELECT patient_id, fields_json FROM seizure_metadata ORDER BY patient_id")
    raw = cur.fetchall()
    records = []
    for pid, fj in raw:
        try:
            records.append({"patient_id": pid, **json.loads(fj)})
        except (json.JSONDecodeError, TypeError):
            continue

    # Per-patient summary table
    per_patient = []
    for r in records:
        per_patient.append({
            "patient_id": r["patient_id"],
            "onset_zone": r.get("onset_zone", "Unknown"),
            "seizure_types": ", ".join(r.get("ilae_seizure_types", [])),
            "syndrome": r.get("syndrome", "Unknown"),
            "etiology": r.get("etiology", "Unknown"),
            "drug_responsiveness": r.get("drug_responsiveness", "Unknown"),
            "surgery_candidacy": r.get("surgery_candidacy", "Not assessed"),
            "seizure_frequency": r.get("current_seizure_frequency", "Unknown"),
            "age_at_onset": r.get("age_at_onset"),
            "lateralization": r.get("lateralization", "Unknown"),
        })

    # Surgery candidates detail
    surgery_candidates = [
        {
            "patient_id": r["patient_id"],
            "onset_zone": r.get("onset_zone"),
            "lateralization": r.get("lateralization"),
            "mri_finding": r.get("mri_finding"),
            "eeg_pattern": r.get("eeg_pattern"),
            "drug_responsiveness": r.get("drug_responsiveness"),
            "surgery_candidacy": r.get("surgery_candidacy"),
        }
        for r in records
        if "Yes" in str(r.get("surgery_candidacy", ""))
    ]

    # Drug-resistant patients
    drug_resistant = [
        {
            "patient_id": r["patient_id"],
            "syndrome": r.get("syndrome"),
            "etiology": r.get("etiology"),
            "seizure_frequency": r.get("current_seizure_frequency"),
            "surgery_candidacy": r.get("surgery_candidacy"),
            "seizure_types": ", ".join(r.get("ilae_seizure_types", [])),
        }
        for r in records
        if "resistant" in str(r.get("drug_responsiveness", "")).lower()
    ]

    # Semiology feature frequency
    sem_counts = {}
    for r in records:
        for s in r.get("semiology", []):
            sem_counts[s] = sem_counts.get(s, 0) + 1
    semiology_frequency = sorted(
        [{"feature": s, "count": c} for s, c in sem_counts.items()],
        key=lambda x: -x["count"]
    )

    # Onset zone × etiology cross-tab
    cross_tab = {}
    for r in records:
        zone = r.get("onset_zone", "Unknown")
        etio = r.get("etiology", "Unknown")
        key = (zone, etio)
        cross_tab[key] = cross_tab.get(key, 0) + 1
    zone_etiology_crosstab = [
        {"onset_zone": z, "etiology": e, "count": c}
        for (z, e), c in sorted(cross_tab.items(), key=lambda x: -x[1])
    ]

    conn.close()
    return {
        "per_patient": per_patient,
        "surgery_candidates": surgery_candidates,
        "drug_resistant": drug_resistant,
        "semiology_frequency": semiology_frequency,
        "zone_etiology_crosstab": zone_etiology_crosstab,
    }


# ──────────────────────────────────────────────────────────────
#  /api/seizure-metadata/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """ILAE classification definitions, field descriptions, clinical glossary."""
    return {
        "seizure_type_definitions": {
            "Focal aware": "Seizure originating in one hemisphere with preserved awareness "
                           "throughout. Previously called 'simple partial seizure'.",
            "Focal impaired awareness": "Focal-onset seizure with impaired awareness at any point. "
                                        "Previously called 'complex partial seizure'.",
            "Focal to bilateral tonic-clonic": "Focal-onset seizure that spreads to both hemispheres, "
                                               "producing bilateral convulsions. Previously 'secondary "
                                               "generalized'.",
            "Generalized tonic-clonic": "Bilateral tonic (stiffening) then clonic (jerking) seizure "
                                        "with loss of consciousness. The most recognized seizure type.",
            "Absence (typical)": "Brief (5-30s) generalized seizure with sudden onset/offset of "
                                 "impaired awareness, 3 Hz spike-and-wave on EEG.",
            "Absence (atypical)": "Similar to typical absence but with less abrupt onset/offset, "
                                  "irregular slow spike-and-wave on EEG.",
            "Myoclonic": "Brief, shock-like involuntary jerks of a muscle or group of muscles. "
                         "Consciousness usually preserved.",
            "Atonic": "Sudden loss of muscle tone causing drop attacks. High injury risk.",
            "Tonic": "Sustained muscle contraction (stiffening) without subsequent clonic phase.",
            "Clonic": "Rhythmic jerking movements without initial tonic phase.",
            "Epileptic spasms": "Brief flexion or extension of proximal and trunk muscles. "
                                "Typical of West syndrome in infants.",
        },
        "onset_zone_descriptions": {
            "Temporal (mesial)": "Hippocampus and amygdala — the most common epileptogenic zone. "
                                "Associated with mesial temporal sclerosis.",
            "Temporal (lateral/neocortical)": "Lateral temporal neocortex. Different semiology "
                                              "from mesial temporal: auditory auras, early aphasia.",
            "Frontal (dorsolateral)": "Dorsolateral prefrontal cortex. Tonic posturing, clonic "
                                      "activity, version.",
            "Frontal (mesial/SMA)": "Supplementary motor area / cingulate. Bilateral asymmetric "
                                    "tonic posturing, preserved awareness, brief duration.",
            "Frontal (orbitofrontal)": "Orbitofrontal cortex. Hyperkinetic movements, autonomic "
                                       "signs, olfactory auras.",
            "Parietal": "Parietal lobe. Somatosensory auras, pain, tingling, distorted body image.",
            "Occipital": "Occipital lobe. Visual auras (colors, shapes, blindness), eye deviation.",
            "Insular": "Insula cortex. Visceral sensations, throat tightness, dysarthria, "
                       "often misdiagnosed as temporal.",
            "Multifocal": "Multiple independent epileptogenic foci across hemispheres.",
            "Generalized (bilateral)": "Seizure onset involves both hemispheres from the start.",
        },
        "etiology_categories": {
            "Structural": "Identifiable structural brain abnormality (MRI-visible). Includes "
                          "hippocampal sclerosis, cortical dysplasia, tumors, vascular "
                          "malformations, traumatic, and perinatal lesions.",
            "Genetic": "Known or presumed genetic cause. May be single-gene (SCN1A, KCNQ2) "
                       "or complex polygenic inheritance.",
            "Infectious": "Post-infectious etiology. Includes post-encephalitic (HSV, "
                          "autoimmune), cerebral malaria, neurocysticercosis.",
            "Immune": "Autoimmune encephalitis (anti-NMDAR, anti-LGI1) or other immune-mediated.",
            "Metabolic": "Metabolic disorder as primary cause (mitochondrial, amino acid, "
                         "glucose transporter deficiency).",
            "Unknown / cryptogenic": "No identifiable cause despite appropriate investigation.",
        },
        "field_descriptions": {
            "ilae_seizure_types": "ILAE 2017 seizure classification types observed in this patient.",
            "onset_zone": "Presumed epileptogenic zone based on semiology, EEG, and MRI concordance.",
            "lateralization": "Hemisphere lateralization of seizure onset.",
            "semiology": "Observable clinical manifestations during seizures (auras, motor signs, "
                         "automatisms, autonomic features).",
            "eeg_pattern": "Predominant interictal or ictal EEG pattern.",
            "mri_finding": "Most significant MRI finding relevant to epilepsy.",
            "etiology": "ILAE etiology classification.",
            "syndrome": "ILAE epilepsy syndrome classification, if applicable.",
            "surgery_candidacy": "Presurgical evaluation status for epilepsy surgery consideration.",
            "drug_responsiveness": "Response to anti-seizure medication therapy.",
            "age_at_onset": "Patient age at first seizure (years).",
            "disease_duration_years": "Duration of epilepsy diagnosis in years.",
            "current_seizure_frequency": "Current seizure frequency category.",
        },
        "clinical_notes": [
            "The ILAE 2017 classification replaced the 1981 system: 'simple partial' → "
            "'focal aware', 'complex partial' → 'focal impaired awareness', 'secondary "
            "generalized' → 'focal to bilateral tonic-clonic'.",
            "Drug-resistant epilepsy (DRE) is defined as failure of adequate trials of two "
            "tolerated and appropriately chosen AED schedules (monotherapy or combination) "
            "to achieve sustained seizure freedom.",
            "Approximately 30% of epilepsy patients are drug-resistant and should be "
            "evaluated for epilepsy surgery.",
            "Concordance between MRI lesion, EEG focus, and semiology is critical for "
            "successful surgical outcomes.",
            "Mesial temporal lobe epilepsy with hippocampal sclerosis has the best surgical "
            "outcomes (60-80% seizure freedom).",
            "Genetic epilepsies (JME, CAE) are typically not surgical candidates but respond "
            "well to appropriate AEDs.",
            "Autoimmune epilepsy should be suspected in new-onset refractory status "
            "epilepticus (NORSE) and rapidly progressive encephalopathy.",
        ],
        "glossary": {
            "ILAE": "International League Against Epilepsy — the global authority on epilepsy "
                    "classification and treatment guidelines.",
            "AED / ASM": "Anti-Epileptic Drug / Anti-Seizure Medication — pharmacological "
                         "treatment for seizure prevention.",
            "DRE": "Drug-Resistant Epilepsy — failure of ≥2 appropriate AED trials.",
            "Epileptogenic Zone": "The brain region responsible for generating seizures. The "
                                  "minimum area of cortex that must be resected for seizure freedom.",
            "Semiology": "The clinical manifestations of a seizure, used to localize the "
                         "epileptogenic zone.",
            "MTS / HS": "Mesial Temporal Sclerosis / Hippocampal Sclerosis — the most common "
                        "structural substrate of temporal lobe epilepsy.",
            "FCD": "Focal Cortical Dysplasia — a malformation of cortical development that is "
                   "a common cause of drug-resistant focal epilepsy.",
            "NORSE": "New-Onset Refractory Status Epilepticus — a clinical entity often with "
                     "autoimmune or cryptogenic etiology.",
            "Concordance": "Agreement between EEG, MRI, and clinical data in localizing the "
                           "epileptogenic zone — essential for surgical planning.",
            "VEEG": "Video-EEG Monitoring — simultaneous video and EEG recording to capture "
                    "and classify seizure events.",
        },
    }


if __name__ == "__main__":
    ensure_data()
    print("=== Overview ===")
    o = overview()
    print(json.dumps(o["kpis"], indent=2))
    print(f"Onset zones: {len(o['onset_zone_distribution'])}")
    print(f"Seizure types: {len(o['seizure_type_frequency'])}")
    print("\n=== Breakdown (summary) ===")
    b = breakdown()
    print(f"Patients: {len(b['per_patient'])}, Surgery candidates: {len(b['surgery_candidates'])}")
    print(f"Drug-resistant: {len(b['drug_resistant'])}")
    print("\n=== Definitions ===")
    d = definitions()
    print(f"Seizure types defined: {len(d['seizure_type_definitions'])}")
    print(f"Onset zones: {len(d['onset_zone_descriptions'])}")
