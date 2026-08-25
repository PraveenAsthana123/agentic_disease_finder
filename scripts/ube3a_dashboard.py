#!/usr/bin/env python3
"""Angelman Syndrome (UBE3A / E6-AP Ubiquitin Ligase Deficiency) Dashboard.

UBE3A gene encodes E6-AP (E6-Associated Protein), an E3 ubiquitin ligase:
  UBE3A: 865 aa; cytoplasmic/nuclear; HECT-domain E3 ubiquitin ligase
  Locus: 15q11.2-q13; GENOMIC IMPRINTING — maternal allele ONLY expressed in neurons
  OMIM Gene: *601623
  OMIM Disease: #105830 (Angelman Syndrome)
  Prevalence: ~1:12,000–20,000 live births

IMPRINTING MECHANISM — WHY MATERNAL LOF CAUSES DISEASE:
  UBE3A undergoes TISSUE-SPECIFIC IMPRINTING in post-mitotic neurons.
  In neurons ONLY: paternal UBE3A is silenced by SNHG14/UBE3A-ATS antisense lncRNA
    (produced from paternal 15q11-q13 imprinting centre).
  Maternal UBE3A is the SOLE active copy in neurons.
  Maternal UBE3A LOF → NO UBE3A protein in neurons → Angelman Syndrome.
  In non-neuronal cells: BOTH alleles expressed → carriers tolerate heterozygous LOF.

FOUR GENETIC MECHANISMS (by frequency):
  1. Maternal deletion 15q11.2-q13 (65-70%): ~5 Mb deletion; detected by CMA/FISH
     Most severe phenotype; atypical facial features; deletions bp1-3 vs bp2-3
  2. UBE3A point mutation / small indel (10-15%): inherited or de novo maternal
     Detected by UBE3A sequencing; variable phenotype
  3. Paternal uniparental disomy (UPD) — chromosome 15 (3-7%): two paternal copies;
     Mildest phenotype; minimal speech sometimes present; detected by methylation/SNP array
  4. Imprinting defect (IC mutation) (2-3%): mutation in imprinting centre on maternal 15q11;
     Paternal methylation pattern on maternal chromosome; most responsive to ASO therapy

EEG — CHARACTERISTIC PATTERNS (pathognomonic in right clinical context):
  1. HIGH-AMPLITUDE (200-500 µV) RHYTHMIC DELTA — 2-3 Hz
     Often frontally or occipitally dominant; runs of 2-3 s; triggered by eye closure
  2. NOTCHED DELTA — high-amplitude delta with superimposed spikes
  3. LARGE AMPLITUDE SLOW SPIKE-WAVE — 4-6 Hz (theta) dominant
  4. OCCIPITAL DOMINANCE — high-amplitude delta/theta, eye closure-sensitive
  5. RUNS OF DIFFUSE DELTA-THETA at 4-6 Hz in older children/adults

SEIZURE TYPES (multiple; ~85% prevalence):
  Myoclonic: 60% (most common; childhood)
  GTCS: 45%
  Atonic (drop attacks): 30%
  Typical absence: 25%
  Focal (± secondary generalisation): 35%
  West syndrome (infantile spasms): 20% (early; 6-18 months)
  Atypical absence: 20%
  Status epilepticus: 15% (mostly myoclonic; management-refractory)
  NCSE (non-convulsive): 10%

CLINICAL TRIAD (Angelman Angelic Appearance):
  1. SEVERE ID + absent or near-absent speech (but receptive language better)
  2. MOVEMENT/BALANCE: ataxic gait, intention tremor, hand flapping, tremulous limb movements
  3. HAPPY DEMEANOR: frequent smiling/laughter, excitable, fascination with water
  4. Microcephaly (postnatal; 50%)
  5. Hypopigmentation (deletion class; due to OCA2/P gene co-deletion)
  6. EEG abnormality (> 80%)
  7. Epilepsy (~85%)

TREATMENT (11 key treatments):
  1. Valproate (VPA): Level A — first line for myoclonic + GTCS + absence; CAUTION with LTG
  2. Clonazepam: Level A — highly effective for myoclonic; tolerance risk long-term
  3. Levetiracetam (LEV): Level B — broad-spectrum; safe; preferred if VPA avoided
  4. Topiramate (TPB): Level B — focal + generalised; cognitive side-effects concerning
  5. Clobazam: Level B — adjunct for myoclonic; tolerance
  6. Lamotrigine (LTG): Level B — AVOID WITH VPA (Stevens-Johnson risk); useful alone
  7. Fenfluramine (FFA): Level A (FDA-approved 2023 for AS) — serotonin/sigma-1 agonist;
     reduces all seizure types; cardiac monitoring required
  8. Melatonin: Level A — severe sleep disturbance (90%+ of AS); 2-10 mg nocte
  9. Cannabidiol (CBD): Level C — adjunct for drug-resistant seizures
  10. ASO UBE3A-ATS knockdown (investigational): Phase I/II — silences paternal SNHG14/UBE3A-ATS
      to de-repress paternal UBE3A in neurons; promising in mouse models
  11. AVOID: Carbamazepine (CBZ) / Oxcarbazepine (OXC) ABSOLUTE CI — worsens myoclonic/atonic;
      vigabatrin and tiagabine may worsen absence/myoclonic

KEY DIFFERENTIALS:
  Rett syndrome (MECP2 LOF): females; regression; hand stereotypies; breathing dysrrhythmia
    AS has no regression; males also affected (UBE3A); EEG more delta-dominant
  Mowat-Wilson (ZEB2): similar happy demeanor; corpus callosum agenesis; cardiac defects
  Pitt-Hopkins (TCF4): hyperventilation episodes; absent corpus callosum; no imprinting
  Phelan-McDermid (SHANK3 deletion 22q13.3): minimal speech; ASD features; no AS EEG pattern
  PWS (Prader-Willi): SAME LOCUS (15q11-q13); opposite — PATERNAL deletion (vs AS maternal)
    PWS: hypotonia → hyperphagia → obesity; minimal epilepsy; NO myoclonic/ataxia

KEY EXAM TRAP:
  CBZ/OXC are ABSOLUTE CI in AS — they can precipitate severe myoclonic status epilepticus.
  VPA + LTG combination is HIGH RISK (Stevens-Johnson syndrome) — avoid co-prescription.
  LTG ALONE is safe and useful for myoclonic in AS; the hazard is COMBINATION with VPA.
"""

import random

SEED = 287
random.seed(SEED)

# ── Genetic mechanism table ────────────────────────────────────────────────────
MECHANISMS = [
    {
        "mechanism": "Maternal deletion 15q11.2-q13 (~5 Mb)",
        "freq": 67,
        "detection": "CMA (chromosomal microarray) / FISH",
        "phenotype": "Classic severe AS; ataxia + hypopigmentation + microcephaly",
        "note": (
            "Most common mechanism (65-70%). Large ~5 Mb deletion removes OCA2 gene → "
            "hypopigmentation (blue eyes, fair skin) in deletion class. bp1-bp3 deletion (~70%) "
            "more severe than bp2-bp3 (~30%) due to additional genes deleted. "
            "All seizure types; severe speech absence; severe ataxia."
        ),
    },
    {
        "mechanism": "UBE3A point mutation / small indel",
        "freq": 12,
        "detection": "UBE3A gene sequencing (maternal origin confirmed)",
        "phenotype": "Classic AS; variable — milder than deletion class",
        "note": (
            "10-15% of AS. De novo or inherited from carrier mother. "
            "Must confirm MATERNAL origin of mutation (paternal UBE3A mutation does NOT cause AS). "
            "Variable severity; some may have single words; epilepsy equally prevalent. "
            "Nonsense, frameshift, missense (HECT domain), splice site variants described."
        ),
    },
    {
        "mechanism": "Paternal uniparental disomy (UPD15)",
        "freq": 5,
        "detection": "Methylation study + SNP array (no deletion; methylation abnormal)",
        "phenotype": "Mildest AS; occasional single words; milder ataxia",
        "note": (
            "3-7% of AS. Two paternal chromosome 15 copies; no maternal UBE3A. "
            "MILDEST phenotype: may have 1-2 words (atypical); epilepsy present but "
            "often less severe; ASD features more frequent. Not detected by FISH; "
            "methylation/SNP array required. Recurrence risk low (sporadic)."
        ),
    },
    {
        "mechanism": "Imprinting centre (IC) defect",
        "freq": 3,
        "detection": "Methylation study (abnormal) → IC sequencing",
        "phenotype": "Mildest-to-moderate AS; best candidate for ASO therapy",
        "note": (
            "2-3% of AS. Imprinting centre mutation on maternal chromosome → maternal allele "
            "adopts paternal methylation pattern → paternal silencing extends to maternal UBE3A. "
            "Most respond to ASO UBE3A-ATS knockdown in trials (de-represses paternal copy). "
            "Recurrence risk up to 50% if IC deletion inherited."
        ),
    },
    {
        "mechanism": "Clinical / undetected (normal methylation, sequencing normal)",
        "freq": 13,
        "detection": "Clinical diagnosis; long-read or RNA sequencing research",
        "phenotype": "Variable; may have mosaic or regulatory variant",
        "note": (
            "~10-15% of clinically diagnosed AS have normal standard workup. "
            "May represent deep intronic variants, regulatory mutations, mosaicism, "
            "or alternative genetic diagnoses. Long-read sequencing increasingly identifies "
            "structural variants missed by standard arrays."
        ),
    },
]

# ── Phenotype / severity by mechanism ─────────────────────────────────────────
MECHANISM_DIST = {
    "Deletion 15q11.2-q13 (~5 Mb)": 27,
    "UBE3A point mutation": 5,
    "Paternal UPD15": 2,
    "Imprinting centre defect": 1,
    "Clinical / undetected": 5,
}

# ── Seizure types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    "Myoclonic", "GTCS", "Atonic/drop attacks", "Typical absence",
    "Focal", "West syndrome/infantile spasms", "Atypical absence",
    "Status epilepticus", "NCSE",
]
SEIZURE_PROBS = [0.60, 0.45, 0.30, 0.25, 0.35, 0.20, 0.20, 0.15, 0.10]

# ── EEG patterns ──────────────────────────────────────────────────────────────
EEG_PATTERNS = [
    "High-amplitude rhythmic delta (2-3 Hz)",
    "Notched delta (delta + superimposed spikes)",
    "Large-amplitude slow spike-wave (4-6 Hz)",
    "Occipital dominance (eye-closure sensitive)",
    "Runs diffuse delta-theta",
]


def _make_patient(i):
    """Synthetic AS/UBE3A patient record (seed=287, deterministic)."""
    rng = random.Random(SEED + i * 61)

    # Genetic mechanism
    if i < 27:
        mechanism = "Deletion 15q11.2-q13"
        severity = "Severe"
        has_speech = False
        has_hypopig = rng.random() < 0.75   # OCA2 co-deletion → hypopig
        has_microcep = rng.random() < 0.60
        eeg_amp_uv = rng.randint(280, 500)
        ataxia_score = round(rng.uniform(6.0, 10.0), 1)   # 0-10 scale
        has_words = 0
        ubd_mech = rng.choice(["bp1-bp3", "bp2-bp3"])
    elif i < 32:
        mechanism = "UBE3A point mutation"
        severity = "Moderate-Severe"
        has_speech = rng.random() < 0.20   # ~20% have 1-2 words
        has_hypopig = rng.random() < 0.10   # No OCA2 co-deletion
        has_microcep = rng.random() < 0.45
        eeg_amp_uv = rng.randint(220, 420)
        ataxia_score = round(rng.uniform(5.0, 9.0), 1)
        has_words = rng.randint(0, 2) if has_speech else 0
        ubd_mech = rng.choice([
            "p.Arg67Trp (HECT N-lobe)", "p.Trp608Ter (null)",
            "p.Glu550Lys (HECT catalytic Cys)", "c.IVS7+1G>A (splice null)",
            "p.Leu502Pro (HECT misfolding)",
        ])
    elif i < 34:
        mechanism = "Paternal UPD15"
        severity = "Mild-Moderate"
        has_speech = rng.random() < 0.50   # May have 1-2 words
        has_hypopig = rng.random() < 0.05
        has_microcep = rng.random() < 0.25
        eeg_amp_uv = rng.randint(160, 320)
        ataxia_score = round(rng.uniform(3.0, 7.0), 1)
        has_words = rng.randint(1, 3) if has_speech else 0
        ubd_mech = "Paternal UPD15 (isodisomy/heterodisomy)"
    elif i == 34:
        mechanism = "Imprinting centre defect"
        severity = "Mild-Moderate"
        has_speech = rng.random() < 0.60
        has_hypopig = rng.random() < 0.05
        has_microcep = rng.random() < 0.20
        eeg_amp_uv = rng.randint(150, 300)
        ataxia_score = round(rng.uniform(2.5, 6.5), 1)
        has_words = rng.randint(1, 4) if has_speech else 0
        ubd_mech = "IC deletion/mutation (maternal)"
    else:
        mechanism = "Clinical / undetected"
        severity = rng.choice(["Moderate", "Severe"])
        has_speech = rng.random() < 0.25
        has_hypopig = rng.random() < 0.15
        has_microcep = rng.random() < 0.40
        eeg_amp_uv = rng.randint(200, 450)
        ataxia_score = round(rng.uniform(4.0, 9.5), 1)
        has_words = rng.randint(0, 2) if has_speech else 0
        ubd_mech = "Normal methylation + sequencing (deep investigation pending)"

    # Seizure types
    seizures = [s for s, p in zip(SEIZURE_TYPES, SEIZURE_PROBS) if rng.random() < p]
    no_seizure = rng.random() < 0.12   # ~12% AS patients seizure-free
    if no_seizure:
        seizures = []
    elif not seizures:
        seizures = ["Myoclonic"]

    # Sleep disturbance (90% AS)
    sleep_disturbed = rng.random() < 0.90

    # CBZ-triggered worsening (exam trap)
    cbz_exposed = rng.random() < 0.10   # ~10% received CBZ before correct diagnosis
    cbz_worsened = cbz_exposed

    # EEG pattern (multiple)
    n_eeg = rng.randint(2, 4)
    eeg_pats = rng.sample(EEG_PATTERNS, k=min(n_eeg, len(EEG_PATTERNS)))

    # Onset age (months)
    onset_months = round(rng.uniform(6, 36), 1)
    diagnosis_age_months = round(onset_months + rng.uniform(4, 30), 1)

    # Treatment response
    vpa_response = rng.choice(["Good", "Partial", "No seizure"]) if seizures else "No seizure"
    fenfluramine_used = rng.random() < 0.18
    lev_used = rng.random() < 0.55
    clonazepam_used = rng.random() < 0.40

    return {
        "id":                    f"AS-UBE3A-{SEED}-{i + 1:02d}",
        "mechanism":             mechanism,
        "severity":              severity,
        "has_any_speech":        has_speech,
        "word_count":            has_words,
        "hypopigmentation":      has_hypopig,
        "microcephaly":          has_microcep,
        "onset_age_months":      onset_months,
        "diagnosis_age_months":  diagnosis_age_months,
        "seizure_types":         seizures,
        "sleep_disturbed":       sleep_disturbed,
        "eeg_amp_uv":            eeg_amp_uv,
        "eeg_patterns":          eeg_pats,
        "ataxia_score_0_10":     ataxia_score,
        "cbz_exposed":           cbz_exposed,
        "cbz_worsened":          cbz_worsened,
        "vpa_response":          vpa_response,
        "fenfluramine_used":     fenfluramine_used,
        "lev_used":              lev_used,
        "clonazepam_used":       clonazepam_used,
        "genetic_detail":        ubd_mech,
    }


PATIENTS = [_make_patient(i) for i in range(40)]


def get_overview():
    n = len(PATIENTS)
    del_n   = sum(1 for p in PATIENTS if "Deletion" in p["mechanism"])
    mut_n   = sum(1 for p in PATIENTS if "point mutation" in p["mechanism"])
    upd_n   = sum(1 for p in PATIENTS if "UPD" in p["mechanism"])
    ic_n    = sum(1 for p in PATIENTS if "Imprinting" in p["mechanism"])
    unk_n   = sum(1 for p in PATIENTS if "Clinical" in p["mechanism"])
    epi_n   = sum(1 for p in PATIENTS if p["seizure_types"])
    myo_n   = sum(1 for p in PATIENTS if "Myoclonic" in p["seizure_types"])
    gtcs_n  = sum(1 for p in PATIENTS if "GTCS" in p["seizure_types"])
    atonic_n = sum(1 for p in PATIENTS if "Atonic" in p["seizure_types"])
    west_n  = sum(1 for p in PATIENTS if "West" in p["seizure_types"])
    se_n    = sum(1 for p in PATIENTS if "Status" in p["seizure_types"])
    sleep_n = sum(1 for p in PATIENTS if p["sleep_disturbed"])
    speech_n = sum(1 for p in PATIENTS if p["has_any_speech"])
    hypopig_n = sum(1 for p in PATIENTS if p["hypopigmentation"])
    microcep_n = sum(1 for p in PATIENTS if p["microcephaly"])
    cbz_w_n  = sum(1 for p in PATIENTS if p["cbz_worsened"])
    ffa_n    = sum(1 for p in PATIENTS if p["fenfluramine_used"])
    lev_n    = sum(1 for p in PATIENTS if p["lev_used"])
    clonaz_n = sum(1 for p in PATIENTS if p["clonazepam_used"])
    avg_amp  = round(sum(p["eeg_amp_uv"] for p in PATIENTS) / n, 0)
    avg_atax = round(sum(p["ataxia_score_0_10"] for p in PATIENTS) / n, 1)
    diag_delay = round(sum(p["diagnosis_age_months"] - p["onset_age_months"] for p in PATIENTS) / n, 1)

    return {
        "n_patients":     n,
        "seed":           SEED,
        "disease":        "Angelman Syndrome (UBE3A / E6-AP Ubiquitin Ligase Deficiency)",
        "gene":           "UBE3A",
        "locus":          "15q11.2-q13",
        "omim_gene":      "*601623",
        "omim_disease":   "#105830 (Angelman Syndrome)",
        "prevalence":     "~1:12,000–20,000 live births",
        "inheritance":    "Genomic imprinting — maternal LOF only; XL/AD does not apply",
        "mechanism_distribution": {
            "Deletion 15q11.2-q13 (~5 Mb)": del_n,
            "UBE3A point mutation": mut_n,
            "Paternal UPD15": upd_n,
            "Imprinting centre defect": ic_n,
            "Clinical / undetected": unk_n,
        },
        "epilepsy_features": {
            "any_seizures":             epi_n,
            "myoclonic":                myo_n,
            "gtcs":                     gtcs_n,
            "atonic_drop_attacks":      atonic_n,
            "west_infantile_spasms":    west_n,
            "status_epilepticus":       se_n,
            "cbz_worsened_n":           cbz_w_n,
        },
        "clinical_features": {
            "sleep_disturbed":          sleep_n,
            "any_speech":               speech_n,
            "hypopigmentation":         hypopig_n,
            "microcephaly":             microcep_n,
        },
        "eeg_summary": {
            "avg_amplitude_uv":         avg_amp,
            "typical_pattern":          "High-amplitude (200-500 µV) rhythmic delta 2-3 Hz",
            "pathognomonic_in_context": True,
        },
        "treatment_summary": {
            "fenfluramine_used":        ffa_n,
            "lev_used":                 lev_n,
            "clonazepam_used":          clonaz_n,
            "cbz_absolute_ci_violated": cbz_w_n,
        },
        "avg_ataxia_score_0_10":        avg_atax,
        "avg_diagnosis_delay_months":   diag_delay,
        "key_exam_facts": [
            "UBE3A IMPRINTING: only MATERNAL UBE3A expressed in neurons; paternal silenced by SNHG14/UBE3A-ATS antisense lncRNA",
            "MATERNAL LOF ONLY causes AS — paternal UBE3A mutation does NOT cause disease (paternal already silenced in neurons)",
            "FOUR MECHANISMS: deletion (67%) > UBE3A mutation (12%) > UPD15 (5%) > IC defect (3%) > clinical (13%)",
            "EEG CHARACTERISTIC: high-amplitude (200-500 µV) rhythmic delta 2-3 Hz; notched delta; occipital dominant — PATHOGNOMONIC in context",
            "EPILEPSY ~85%: myoclonic (60%) + GTCS (45%) + atonic drops (30%) + West syndrome (20%) — MULTIPLE types co-occur",
            "HAPPY DEMEANOR + absent speech + ataxic gait + hand flapping + fascination with water — ANGELMAN ANGELIC APPEARANCE",
            "CBZ/OXC ABSOLUTE CI — triggers/worsens myoclonic and atonic seizures; can precipitate myoclonic status epilepticus",
            "VPA+LTG HIGH RISK combination — Stevens-Johnson syndrome risk; LTG alone is safe and effective in AS",
            "FENFLURAMINE FDA-APPROVED 2023 for AS — serotonin + sigma-1 agonist; reduces all seizure types; cardiac monitoring mandatory",
            "SLEEP DISTURBANCE ~90% — severe; melatonin Level A first line; affects QoL more than seizures in some families",
            "PWS vs AS — SAME 15q11-q13 LOCUS: paternal deletion = Prader-Willi (obesity + minimal epilepsy); maternal deletion = Angelman (severe epilepsy + ataxia)",
            "ASO UBE3A-ATS knockdown (Phase I/II) — de-represses paternal UBE3A in neurons by silencing SNHG14 antisense transcript; IC defect class best candidates",
            "NO NBS currently — diagnosis by methylation study + CMA/FISH + UBE3A sequencing; mean diagnosis delay ~2.5 years from onset",
            "HYPOPIGMENTATION in deletion class only — OCA2 gene co-deleted with UBE3A in large 15q11-q13 deletion; absent in UPD/IC/mutation classes",
            "UPD15 mildest phenotype — may have 1-2 words; ASD features more prominent; epilepsy less severe; NO hypopigmentation",
        ],
    }


def get_breakdown():
    patients_out = []
    for p in PATIENTS:
        patients_out.append({
            "id":                    p["id"],
            "mechanism":             p["mechanism"],
            "severity":              p["severity"],
            "has_any_speech":        p["has_any_speech"],
            "word_count":            p["word_count"],
            "hypopigmentation":      p["hypopigmentation"],
            "microcephaly":          p["microcephaly"],
            "onset_age_months":      p["onset_age_months"],
            "diagnosis_age_months":  p["diagnosis_age_months"],
            "seizure_types":         p["seizure_types"],
            "sleep_disturbed":       p["sleep_disturbed"],
            "eeg_amp_uv":            p["eeg_amp_uv"],
            "eeg_patterns":          p["eeg_patterns"],
            "ataxia_score_0_10":     p["ataxia_score_0_10"],
            "cbz_exposed":           p["cbz_exposed"],
            "cbz_worsened":          p["cbz_worsened"],
            "vpa_response":          p["vpa_response"],
            "fenfluramine_used":     p["fenfluramine_used"],
            "lev_used":              p["lev_used"],
            "clonazepam_used":       p["clonazepam_used"],
            "genetic_detail":        p["genetic_detail"],
        })

    # Group by mechanism
    mech_groups = {}
    for p in PATIENTS:
        mech_groups.setdefault(p["mechanism"], []).append(p)

    by_mechanism = {}
    for mech, pts in mech_groups.items():
        by_mechanism[mech] = {
            "n":                     len(pts),
            "epilepsy_pct":          round(sum(1 for x in pts if x["seizure_types"]) / len(pts) * 100, 1),
            "myoclonic_pct":         round(sum(1 for x in pts if "Myoclonic" in x["seizure_types"]) / len(pts) * 100, 1),
            "west_pct":              round(sum(1 for x in pts if "West" in x["seizure_types"]) / len(pts) * 100, 1),
            "any_speech_pct":        round(sum(1 for x in pts if x["has_any_speech"]) / len(pts) * 100, 1),
            "hypopig_pct":           round(sum(1 for x in pts if x["hypopigmentation"]) / len(pts) * 100, 1),
            "sleep_disturbance_pct": round(sum(1 for x in pts if x["sleep_disturbed"]) / len(pts) * 100, 1),
            "avg_eeg_amp_uv":        round(sum(x["eeg_amp_uv"] for x in pts) / len(pts), 0),
            "avg_ataxia_score":      round(sum(x["ataxia_score_0_10"] for x in pts) / len(pts), 1),
            "avg_diag_delay_months": round(sum(x["diagnosis_age_months"] - x["onset_age_months"] for x in pts) / len(pts), 1),
        }

    # Seizure type counts
    seizure_counts = {}
    for p in PATIENTS:
        for s in p["seizure_types"]:
            seizure_counts[s] = seizure_counts.get(s, 0) + 1

    # EEG pattern frequency
    eeg_counts = {}
    for p in PATIENTS:
        for ep in p["eeg_patterns"]:
            eeg_counts[ep] = eeg_counts.get(ep, 0) + 1

    # Treatment summary
    n = len(PATIENTS)
    treatment_summary = {
        "vpa_first_line_n":               n,
        "clonazepam_myoclonic_n":         sum(1 for p in PATIENTS if p["clonazepam_used"]),
        "lev_broadspectrum_n":            sum(1 for p in PATIENTS if p["lev_used"]),
        "fenfluramine_used_n":            sum(1 for p in PATIENTS if p["fenfluramine_used"]),
        "melatonin_sleep_n":              sum(1 for p in PATIENTS if p["sleep_disturbed"]),
        "cbz_absolute_ci_violated_n":     sum(1 for p in PATIENTS if p["cbz_worsened"]),
        "vpa_good_response_n":            sum(1 for p in PATIENTS if p["vpa_response"] == "Good"),
        "vpa_partial_n":                  sum(1 for p in PATIENTS if p["vpa_response"] == "Partial"),
    }

    clinical_summary = {
        "pct_epilepsy":             round(sum(1 for p in PATIENTS if p["seizure_types"]) / n * 100, 1),
        "pct_sleep_disturbed":      round(sum(1 for p in PATIENTS if p["sleep_disturbed"]) / n * 100, 1),
        "pct_any_speech":           round(sum(1 for p in PATIENTS if p["has_any_speech"]) / n * 100, 1),
        "pct_hypopig":              round(sum(1 for p in PATIENTS if p["hypopigmentation"]) / n * 100, 1),
        "pct_microcephaly":         round(sum(1 for p in PATIENTS if p["microcephaly"]) / n * 100, 1),
        "pct_cbz_worsened":         round(sum(1 for p in PATIENTS if p["cbz_worsened"]) / n * 100, 1),
        "avg_eeg_amp_uv":           round(sum(p["eeg_amp_uv"] for p in PATIENTS) / n, 0),
    }

    return {
        "patients":          patients_out,
        "by_mechanism":      by_mechanism,
        "seizure_counts":    seizure_counts,
        "eeg_pattern_freq":  eeg_counts,
        "treatment_summary": treatment_summary,
        "clinical_summary":  clinical_summary,
    }


def get_definitions():
    return {
        "disease_name": "Angelman Syndrome (UBE3A / E6-AP Ubiquitin Ligase Deficiency)",
        "gene":         "UBE3A (15q11.2-q13) — MATERNALLY EXPRESSED ONLY in neurons",
        "locus":        "15q11.2-q13",
        "omim_gene":    "UBE3A *601623",
        "omim_disease": "#105830 (Angelman Syndrome)",
        "inheritance":  "Genomic imprinting — maternal LOF (paternal allele neuronal-silenced)",
        "terms": {
            "UBE3A_E6AP": (
                "UBE3A (Ubiquitin Protein Ligase E3A) = E6-AP (E6-Associated Protein). "
                "865 amino acids; HECT-domain E3 ubiquitin ligase. Locus 15q11.2-q13. "
                "Ubiquitylates target proteins for proteasomal degradation (e.g., p53, Arc/Arg3.1). "
                "CRITICAL: undergoes tissue-specific genomic imprinting — ONLY maternal allele "
                "expressed in post-mitotic neurons. Non-neuronal cells express both alleles. "
                "Maternal LOF → zero UBE3A in neurons → Angelman Syndrome."
            ),
            "Genomic_imprinting_UBE3A": (
                "Genomic imprinting = epigenetic mechanism where only one parental allele is "
                "expressed. UBE3A imprinting is NEURON-SPECIFIC. Paternal UBE3A silenced in "
                "neurons by SNHG14 (SNRPN host gene 14) lncRNA / UBE3A-ATS (antisense transcript) "
                "produced from the paternal imprinting centre (IC) at 15q11-q13. "
                "In non-neuronal cells: both alleles expressed (carrier mothers tolerate single "
                "maternal LOF allele without disease in peripheral tissues)."
            ),
            "Four_mechanisms_AS": (
                "Four genetic mechanisms causing AS, all resulting in absent maternal UBE3A in neurons: "
                "(1) Maternal deletion 15q11-q13 (~5 Mb) — 65-70%; detected by CMA/FISH. "
                "(2) UBE3A point mutation/indel — 10-15%; must confirm MATERNAL origin. "
                "(3) Paternal UPD15 — 3-7%; two paternal chromosome 15 copies; methylation/SNP array. "
                "(4) IC (imprinting centre) defect — 2-3%; maternal chromosome acquires paternal "
                "methylation pattern; IC sequencing required. "
                "Important: mechanism determines recurrence risk and therapy eligibility."
            ),
            "AS_EEG_pattern": (
                "Angelman EEG is CHARACTERISTIC (not diagnostic alone but highly suggestive): "
                "1. High-amplitude (200-500 µV) rhythmic delta at 2-3 Hz — frontally or occipitally "
                "dominant; runs of 2-3 seconds; often eye-closure sensitive. "
                "2. Notched delta — high-amplitude delta with superimposed multifocal spikes. "
                "3. Large-amplitude slow spike-wave at 4-6 Hz (theta range). "
                "4. Occipital dominance — high-amplitude delta/theta, attenuated by eye opening. "
                "In right clinical context (severe ID + absent speech + ataxia), this EEG is "
                "near-pathognomonic for AS."
            ),
            "CBZ_OXC_absolute_CI": (
                "Carbamazepine (CBZ) and oxcarbazepine (OXC) are ABSOLUTE CONTRAINDICATIONS in "
                "Angelman Syndrome. Mechanism: CBZ/OXC are sodium-channel blockers that worsen "
                "myoclonic and atonic seizures — the predominant seizure types in AS. "
                "Administration → paradoxical worsening of myoclonic status epilepticus and "
                "atonic drop attacks. Key exam point: AS patients prescribed CBZ before correct "
                "diagnosis (misdiagnosed as focal epilepsy) deteriorate rapidly. "
                "ALSO: vigabatrin and tiagabine can worsen myoclonic/absence — use with caution."
            ),
            "VPA_LTG_interaction": (
                "Valproate (VPA) + Lamotrigine (LTG) combination is HIGH RISK in AS. "
                "VPA inhibits LTG glucuronidation → plasma LTG levels double or triple → "
                "Stevens-Johnson syndrome (SJS/TEN) risk. This interaction is NOT specific to AS "
                "but is particularly dangerous in children given the polypharmacy needed for "
                "multi-seizure-type AS epilepsy. KEY RULE: if VPA is used, halve LTG dose; "
                "if LTG used alone without VPA it is safe and effective for myoclonic seizures in AS."
            ),
            "Fenfluramine_AS": (
                "Fenfluramine (FFA) — serotonin-releasing agent + sigma-1 receptor agonist. "
                "FDA-approved 2023 for adjunctive treatment of seizures associated with Angelman Syndrome. "
                "Mechanism: enhances serotonergic and sigma-1 pathways; reduces multi-seizure-type burden. "
                "Clinical trials (BUTTERFLY-1, BUTTERFLY-2): ~25-30% seizure frequency reduction. "
                "CARDIAC MONITORING mandatory (historical concern: serotonergic cardiac valvulopathy "
                "at obesity doses — rare at AS doses but echo required). "
                "Dose: 0.1-0.7 mg/kg/day; maximum 26 mg/day."
            ),
            "ASO_UBE3A_ATS": (
                "Antisense oligonucleotide (ASO) targeting UBE3A-ATS (also called SNHG14) — "
                "investigational gene therapy approach for AS. "
                "Mechanism: ASO silences the paternal SNHG14/UBE3A-ATS antisense lncRNA → "
                "de-represses the SILENCED PATERNAL UBE3A allele in neurons → restores UBE3A protein. "
                "Effect: paternal UBE3A becomes expressed in neurons (normally silenced). "
                "Best efficacy predicted for IC defect class (imprinting mechanism still present) "
                "and UPD class. Mouse models: dramatic seizure and behaviour improvement. "
                "Phase I/II trials (Roche/Genentech, GeneTx, Ionis) ongoing as of 2026."
            ),
            "Methylation_study_AS": (
                "DNA methylation study of 15q11-q13 imprinting centre (SNRPN locus) is the FIRST-LINE "
                "molecular test for AS. "
                "Normal methylation: both paternal (methylated) and maternal (unmethylated) bands present. "
                "Abnormal AS methylation: ONLY methylated (paternal) pattern — maternal band absent. "
                "Detects: deletion, UPD, IC defect (all show abnormal methylation). "
                "DOES NOT DETECT: UBE3A point mutations (normal methylation). "
                "After abnormal methylation: CMA/FISH (deletion?), SNP array (UPD?), IC sequencing (IC defect?). "
                "After normal methylation in suspected AS: UBE3A gene sequencing."
            ),
            "PWS_vs_AS": (
                "Prader-Willi Syndrome (PWS) vs Angelman Syndrome (AS) — SAME LOCUS, OPPOSITE PARENT: "
                "PWS: PATERNAL 15q11-q13 loss → maternal uniparental disomy of paternal-expressed genes → "
                "neonatal hypotonia, feeding difficulties → childhood hyperphagia → obesity, hypogonadism; "
                "MINIMAL epilepsy. "
                "AS: MATERNAL 15q11-q13 loss (UBE3A) → severe ID, absent speech, ataxia, happy demeanor, "
                "SEVERE epilepsy (~85%). "
                "EXAM TRAP: deletion at SAME location causes opposite syndrome depending on which parent "
                "transmitted the deletion."
            ),
            "UPD15_AS_phenotype": (
                "Paternal uniparental disomy of chromosome 15 (UPD15) in AS: "
                "Patient inherits TWO PATERNAL copies of chromosome 15 (no maternal copy). "
                "Result: no maternal UBE3A in neurons → AS. "
                "PHENOTYPE MILDEST: occasional 1-2 words (atypical); epilepsy often less severe; "
                "ASD features more prominent; NO hypopigmentation (no OCA2 deletion). "
                "Distinguished from deletion by: methylation study ABNORMAL, but CMA NORMAL (no deletion); "
                "SNP array shows loss of heterozygosity (LOH) at 15q11-q13 with no copy number change. "
                "Recurrence risk: very low (sporadic meiotic error)."
            ),
            "Hypopigmentation_deletion": (
                "Hypopigmentation (light complexion, blue eyes, fair hair) in AS occurs SPECIFICALLY "
                "in the DELETION class only. "
                "Mechanism: the ~5 Mb 15q11-q13 deletion removes OCA2 (oculocutaneous albinism II) gene, "
                "which encodes a melanosomal membrane transporter essential for melanin synthesis. "
                "Haploinsufficiency of OCA2 → reduced melanin → hypopigmentation. "
                "KEY DISCRIMINATOR: hypopigmentation PRESENT = deletion class; "
                "ABSENT in UPD/IC defect/point mutation classes. "
                "In the context of AS + hypopigmentation → almost certainly deletion class."
            ),
            "Sleep_AS": (
                "Sleep disturbance in AS: ~90% of patients have severely disrupted sleep. "
                "Characterised by reduced sleep duration, frequent nocturnal awakenings, early waking. "
                "Mechanism: UBE3A loss disrupts circadian rhythm regulation (UBE3A ubiquitylates clock proteins). "
                "Treatment Level A: melatonin 2-10 mg nocte (reduces sleep latency, improves duration). "
                "Clonazepam (benzodiazepine) also useful for myoclonic seizures + sedation. "
                "Families often report sleep disturbance as worse impact than epilepsy on QoL."
            ),
            "Arc_Arg31_UBE3A": (
                "Arc (Activity-Regulated Cytoskeleton-associated protein / Arg3.1) is a key UBE3A "
                "substrate in neurons. UBE3A ubiquitylates Arc → proteasomal degradation → "
                "regulates AMPA receptor trafficking and synaptic plasticity. "
                "UBE3A LOF → Arc accumulates → AMPA receptor internalisation impaired → "
                "dysregulated long-term potentiation (LTP) and memory formation. "
                "This Arc-UBE3A-AMPA pathway is central to the cognitive and epileptic phenotype of AS, "
                "and is a therapeutic target (Arc reduction strategies in early-stage research)."
            ),
            "Diagnosis_workup_AS": (
                "Step 1: Clinical suspicion — severe ID + absent speech + ataxia + happy demeanor + EEG. "
                "Step 2: DNA methylation study (SNRPN locus 15q11-q13) — FIRST test. "
                "Step 3a (abnormal methylation): CMA/FISH for deletion; SNP array for UPD; "
                "IC sequencing if no deletion/UPD. "
                "Step 3b (normal methylation): UBE3A sequencing (maternal inheritance + origin confirmed). "
                "Step 4 (normal methylation + normal UBE3A): consider other NDDs (clinical AS, 10-15%). "
                "Mean diagnosis delay: ~2-2.5 years from symptom onset in most studies. "
                "Neonatal EEG may show high-amplitude delta before clinical features are evident."
            ),
        },
    }
