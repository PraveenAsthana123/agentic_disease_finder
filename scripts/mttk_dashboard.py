#!/usr/bin/env python3
"""MT-TK — Mitochondrially Encoded tRNA-Lys — MERRF Syndrome
(Myoclonic Epilepsy with Ragged Red Fibres) — Multiple Symmetrical Lipomatosis —
CPEO / Exercise Intolerance — m.8344A>G the MOST COMMON MERRF mutation worldwide.

MT-TK (OMIM *590060) encodes the mitochondrial tRNA for lysine (anticodon UUU/C),
rCRS H-strand positions 8295–8364 (69 bp). Like MT-TL1, MT-TK is a tRNA gene
essential for translating ALL 13 mtDNA-encoded OXPHOS subunits — mutations impair
mt-translation globally, causing pan-OXPHOS deficiency (predominantly CI + CIV;
CII/SDH nuclear-encoded: NORMAL — the mt-translation biochemical fingerprint).

m.8344A>G (TΨC loop of tRNA-Lys) is the MOST COMMON MERRF mutation, accounting
for ~80-90% of MERRF cases worldwide. MERRF is the archetype Progressive Myoclonic
Epilepsy (PME) caused by an mtDNA mutation.

  MT-TK gene             OMIM *590060
  Primary disease        MERRF Syndrome (OMIM #545000)
                         Multiple Symmetrical Lipomatosis (Madelung / Launois-Bensaude)
                         CPEO / Exercise Intolerance / Myopathy
                         Overlap: Leigh-like (high heteroplasmy, rare)
  Protein product        tRNA-Lys (UUU/C anticodon) — 69 nucleotides; RNA gene
  Genome                 Mitochondrial DNA (mtDNA), H-strand, rCRS 8295–8364
  Inheritance            MATERNAL — heteroplasmic; heteroplasmy threshold critical
  Key mutation           m.8344A>G — TΨC loop disruption → impaired aminoacylation
                           → pan-OXPHOS (CI + CIV predominantly)

HETEROPLASMY THRESHOLD (m.8344A>G — blood underestimates by ~10-15%):
  <60% blood:             Asymptomatic carrier / elevated CK / exercise intolerance
  60-75% blood:           Partial MERRF — myoclonus / ataxia / moderate RRF
  75-90% blood:           Full MERRF — myoclonus + ataxia + RRF + dementia
  >90% blood:             Severe MERRF + multi-system failure + MSL + neuropathy
  Muscle biopsy preferred for heteroplasmy in equivocal blood cases (>20% higher)

PAN-OXPHOS (CI + CIV PREDOMINANTLY) — KEY DISTINGUISHER:
  MT-TK mutations impair mt-ribosome aminoacylation of Lys → defective translation
  of ALL 13 mtDNA-encoded OXPHOS subunits → CI + CIV reduced; CIII/CV variable;
  CII (SDH — nuclear-encoded) NORMAL → CII NORMAL is the mt-translation fingerprint.
  BN-PAGE: CI + CIV reduced pattern; CII band normal.
  MERRF CI+CIV pattern ≠ isolated CI (Leigh/LHON) ≠ isolated CIV (SURF1) ≠ CII (SDHA).

MERRF TRIAD — PROGRESSIVE MYOCLONIC EPILEPSY:
  1. Myoclonic epilepsy (action myoclonus + cortical myoclonus, generalized seizures)
  2. Cerebellar ataxia (progressive, truncal > limb)
  3. Ragged Red Fibres (RRF) on muscle biopsy (Gomori trichrome + SDH-positive/COX-negative)
  Plus: dementia, SNHL, short stature, peripheral neuropathy, optic atrophy,
        cardiomyopathy, Multiple Symmetrical Lipomatosis (MSL — Madelung's disease)

MULTIPLE SYMMETRICAL LIPOMATOSIS (MSL):
  10-20% of MERRF patients; symmetric non-encapsulated lipomas (neck, shoulders, upper
  trunk); PATHOGNOMONIC of MERRF when co-occurring with myoclonic epilepsy in a maternal
  pedigree; also Launois-Bensaude syndrome; brown adipose tissue (UCP1-enriched) with
  dysfunctional mitochondria; MSL in MERRF distinguishes from sporadic lipomas.

NO STROKE-LIKE EPISODES — KEY DISTINGUISHER from MELAS (MT-TL1):
  MERRF ≠ MELAS: SLE absent (or extremely rare, overlap phenotype only);
  myoclonus is the dominant epilepsy feature (action + cortical) not focal seizures;
  MRI: cerebellar atrophy + basal ganglia changes (NOT cortical crossing-territory lesions);
  m.8344A>G vs m.3243A>G distinguishes definitively.
"""

import random
from collections import Counter

SEED = 787
N_PATIENTS = 40

VARIANTS = [
    ("m.8344A>G", "TΨC loop", "~80% of MERRF worldwide; disrupts YARS2/KARS2 aminoacylation; pan-OXPHOS (CI+CIV); classic full MERRF triad at high heteroplasmy; MSL in 15%; most common mtDNA tRNA-Lys pathogenic mutation"),
    ("m.8356T>C", "Acceptor stem", "~10% of MERRF; structural acceptor stem disruption; MERRF phenotype (myoclonus + ataxia + RRF); generally milder than m.8344A>G; similar heteroplasmy threshold"),
    ("m.8363G>A", "T-loop", "~5%; T-loop structural perturbation; MERRF-like phenotype; variable penetrance; slower progression than m.8344A>G; fewer systemic features"),
    ("m.8296A>G", "Anticodon stem-loop", "~3%; diabetes + deafness (MIDD-like) rather than classic MERRF; low heteroplasmy → endocrine/auditory; very rare in classic MERRF cohorts"),
    ("Large deletion", "Multi-gene", "~2%; deletion spanning MT-TK region → KSS/CPEO/Pearson; multi-complex OXPHOS; annual Holter mandatory"),
]

PHENOTYPES = [
    "Full MERRF (myoclonus + ataxia + RRF + dementia)",
    "Partial MERRF (myoclonus + ataxia, mild RRF)",
    "MERRF + MSL (Multiple Symmetrical Lipomatosis)",
    "MERRF + CPEO (ophthalmoplegia overlap)",
    "Exercise intolerance / subclinical carrier",
]

OUTCOMES = [
    "Stable on LEV + clonazepam",
    "Progressive disability (ataxia dominant)",
    "Myoclonus controlled, ataxia progressing",
    "Lost to follow-up",
    "Deceased (respiratory failure / cardiomyopathy)",
]


def _make_patients():
    rng = random.Random(SEED)
    patients = []
    variant_weights = [0.80, 0.10, 0.05, 0.03, 0.02]
    phenotype_weights = [0.35, 0.28, 0.15, 0.12, 0.10]

    for i in range(N_PATIENTS):
        variant_idx = rng.choices(range(len(VARIANTS)), weights=variant_weights)[0]
        variant_name = VARIANTS[variant_idx][0]
        phenotype_idx = rng.choices(range(len(PHENOTYPES)), weights=phenotype_weights)[0]
        phenotype = PHENOTYPES[phenotype_idx]

        # Heteroplasmy: blood and muscle (blood ~10-15% lower)
        if variant_name == "Large deletion":
            het_blood = rng.randint(20, 65)
        elif variant_name == "m.8296A>G":
            het_blood = rng.randint(18, 55)
        else:
            het_blood = rng.randint(45, 95)
        het_muscle = min(100, het_blood + rng.randint(8, 18))

        # CI activity (% of mean normal — predominantly CI+CIV affected)
        if het_blood > 85:
            ci_pct = rng.randint(8, 28)
        elif het_blood > 70:
            ci_pct = rng.randint(20, 45)
        elif het_blood > 55:
            ci_pct = rng.randint(35, 60)
        else:
            ci_pct = rng.randint(50, 80)

        # CIV activity
        if het_blood > 80:
            civ_pct = rng.randint(10, 32)
        elif het_blood > 65:
            civ_pct = rng.randint(25, 50)
        else:
            civ_pct = rng.randint(45, 75)

        # Lactic acid
        lactic_acid = round(rng.uniform(1.8, 6.8) if het_blood > 65 else rng.uniform(1.0, 3.2), 1)

        # Clinical features (heteroplasmy-driven)
        myoclonus = het_blood > 60 or phenotype_idx <= 2
        ataxia = het_blood > 58 or phenotype_idx in [0, 1, 2, 3]
        ragged_red_fibres = het_blood > 55 or phenotype_idx in [0, 2]
        dementia = het_blood > 75 and phenotype_idx in [0, 2]
        snhl = rng.random() < (0.55 if het_blood > 65 else 0.25)
        short_stature = rng.random() < (0.35 if het_blood > 60 else 0.10)
        msl = phenotype_idx == 2 or (variant_name == "m.8344A>G" and rng.random() < 0.15)
        peripheral_neuropathy = het_blood > 70 and rng.random() < 0.45
        optic_atrophy = het_blood > 72 and rng.random() < 0.22
        cardiomyopathy = het_blood > 75 and rng.random() < 0.18
        cpeo = phenotype_idx == 3 or (variant_name == "Large deletion")

        # Seizures type
        seizure_type = "myoclonic" if myoclonus else ("absence" if rng.random() < 0.3 else "none")
        stroke_like_episode = False  # MERRF does NOT have SLE (distinguishes from MELAS)

        onset_weeks = rng.randint(6 * 52, 30 * 52) if het_blood > 60 else rng.randint(1 * 52, 5 * 52)

        outcome = rng.choices(OUTCOMES, weights=[0.40, 0.25, 0.20, 0.10, 0.05])[0]

        patients.append({
            "patient_id": f"MTTK-{i+1:03d}",
            "variant": variant_name,
            "phenotype": phenotype,
            "heteroplasmy_blood_pct": het_blood,
            "heteroplasmy_muscle_pct": het_muscle,
            "ci_activity_pct": ci_pct,
            "civ_activity_pct": civ_pct,
            "lactic_acid_mmolL": lactic_acid,
            "myoclonus": myoclonus,
            "cerebellar_ataxia": ataxia,
            "ragged_red_fibres": ragged_red_fibres,
            "dementia": dementia,
            "sensorineural_hearing_loss": snhl,
            "short_stature": short_stature,
            "multiple_symmetrical_lipomatosis": msl,
            "peripheral_neuropathy": peripheral_neuropathy,
            "optic_atrophy": optic_atrophy,
            "cardiomyopathy": cardiomyopathy,
            "cpeo": cpeo,
            "stroke_like_episode": stroke_like_episode,
            "seizure_type": seizure_type,
            "onset_weeks": onset_weeks,
            "outcome": outcome,
        })
    return patients


def get_overview():
    patients = _make_patients()
    n = len(patients)

    # Statistics
    avg_ci = round(sum(p["ci_activity_pct"] for p in patients) / n, 1)
    avg_civ = round(sum(p["civ_activity_pct"] for p in patients) / n, 1)
    avg_lactate = round(sum(p["lactic_acid_mmolL"] for p in patients) / n, 1)
    myoclonus_pct = round(sum(p["myoclonus"] for p in patients) / n * 100, 1)
    ataxia_pct = round(sum(p["cerebellar_ataxia"] for p in patients) / n * 100, 1)
    rrf_pct = round(sum(p["ragged_red_fibres"] for p in patients) / n * 100, 1)
    snhl_pct = round(sum(p["sensorineural_hearing_loss"] for p in patients) / n * 100, 1)
    dementia_pct = round(sum(p["dementia"] for p in patients) / n * 100, 1)
    msl_pct = round(sum(p["multiple_symmetrical_lipomatosis"] for p in patients) / n * 100, 1)
    neuropathy_pct = round(sum(p["peripheral_neuropathy"] for p in patients) / n * 100, 1)
    cardiomyopathy_pct = round(sum(p["cardiomyopathy"] for p in patients) / n * 100, 1)
    sle_pct = 0.0  # MERRF does NOT have SLE — distinguishes from MELAS

    pheno_dist = [
        {"phenotype": p, "count": sum(1 for pt in patients if pt["phenotype"] == p)}
        for p in PHENOTYPES
    ]

    summary_features = [
        {"feature": "Myoclonic epilepsy", "pct": myoclonus_pct},
        {"feature": "Cerebellar ataxia", "pct": ataxia_pct},
        {"feature": "Ragged red fibres (muscle biopsy)", "pct": rrf_pct},
        {"feature": "Sensorineural hearing loss", "pct": snhl_pct},
        {"feature": "Dementia / cognitive decline", "pct": dementia_pct},
        {"feature": "Multiple Symmetrical Lipomatosis (MSL)", "pct": msl_pct},
        {"feature": "Peripheral neuropathy", "pct": neuropathy_pct},
        {"feature": "Cardiomyopathy", "pct": cardiomyopathy_pct},
        {"feature": "Short stature", "pct": round(sum(p["short_stature"] for p in patients) / n * 100, 1)},
        {"feature": "Optic atrophy", "pct": round(sum(p["optic_atrophy"] for p in patients) / n * 100, 1)},
        {"feature": "CPEO", "pct": round(sum(p["cpeo"] for p in patients) / n * 100, 1)},
        {"feature": "Stroke-like Episodes (SLE)", "pct": sle_pct},
    ]

    heteroplasmy_map = [
        {"tier": "<60% blood (≈<70% muscle)", "phenotype": "Asymptomatic / elevated CK / exercise intolerance only", "merrf_severity": "None / subclinical", "vpa_ci": "YES — absolute CI"},
        {"tier": "60-75% blood (≈70-82% muscle)", "phenotype": "Partial MERRF — myoclonus + ataxia; moderate RRF", "merrf_severity": "Partial", "vpa_ci": "YES — absolute CI"},
        {"tier": "75-90% blood (≈83-95% muscle)", "phenotype": "Full MERRF — myoclonus + ataxia + RRF + dementia + SNHL", "merrf_severity": "Full", "vpa_ci": "YES — absolute CI"},
        {"tier": ">90% blood (≈>97% muscle)", "phenotype": "Severe MERRF + MSL + neuropathy + cardiomyopathy + multi-system", "merrf_severity": "Severe", "vpa_ci": "YES — absolute CI"},
    ]

    mol_features = [
        {"feature": "tRNA-Lys pan-OXPHOS (CI + CIV predominant)", "significance": "MT-TK tRNA-Lys mutation → impaired aminoacylation of all Lys codons in mt-ribosome → defective translation of all 13 mtDNA-encoded OXPHOS subunits; CI + CIV most severely reduced; CII (SDH, nuclear) NORMAL — mt-translation fingerprint"},
        {"feature": "m.8344A>G — TΨC loop (80% of MERRF)", "significance": "Most common MERRF mutation worldwide; TΨC loop disruption impairs YARS2/KARS2 aminoacylation recognition; heteroplasmy >85% blood → classic full MERRF triad; threshold genetics parallels MT-TL1 m.3243A>G"},
        {"feature": "Multiple Symmetrical Lipomatosis (MSL) — PATHOGNOMONIC of MERRF", "significance": "MSL (Madelung's disease / Launois-Bensaude) in 10-20% of MERRF patients; symmetric non-encapsulated cervical, shoulder, upper-truncal lipomas; brown adipose tissue with dysfunctional mitochondria (UCP1-enriched); PATHOGNOMONIC co-occurrence with myoclonic epilepsy in maternal pedigree"},
        {"feature": "NO Stroke-like Episodes — KEY DDx vs MELAS", "significance": "MERRF does NOT cause stroke-like episodes (SLE); SLE = MELAS/MT-TL1 hallmark; MERRF: cerebellar atrophy + BG changes on MRI — NOT cortical crossing-territory lesions; m.8344A>G vs m.3243A>G distinguishes definitively; WES misses both"},
        {"feature": "RRF COX-negative / SDH-positive (same mt-translation pattern)", "significance": "Ragged Red Fibres (Gomori trichrome) with COX-negative / SDH-positive pattern identical to MELAS; SSSVS (strongly SDH-stained small vessels) less prominent in MERRF than MELAS; RRF percentage correlates with heteroplasmy"},
        {"feature": "Muscle biopsy preferred over blood for heteroplasmy", "significance": "Muscle is post-mitotic; retains original heteroplasmy better than haematopoietic tissue; blood underestimates by ~10-15% in MERRF (less than MELAS's 20-30%); always quantify muscle m.8344A>G in equivocal blood cases"},
    ]

    alerts = [
        "⛔ VPA (Valproic Acid): ABSOLUTE CONTRAINDICATION — mt-ribosome inhibition + CoA sequestration → precipitates acute mt crisis; ALWAYS use LEV for myoclonic seizures in MERRF/MT-TK",
        "⛔ Metformin: ABSOLUTE CONTRAINDICATION — Complex I inhibitor, additive with CI deficiency → fatal lactic acidosis; use insulin for any DM; avoid all biguanides",
        "⛔ Linezolid: ABSOLUTE CI — inhibits mt-23S rRNA → collapses mt-translation of all 13 OXPHOS subunits; worsens pan-OXPHOS catastrophically",
        "⛔ Chloramphenicol: ABSOLUTE CI — mt-ribosome inhibitor; additive mt-translation shutdown",
        "⛔ Propofol: ABSOLUTE CI (PRIS) — propofol infusion syndrome; use sevoflurane/isoflurane for anaesthesia in MERRF",
        "⚠️ MSL (Madelung's / Multiple Symmetrical Lipomatosis): screen ALL maternal relatives with myoclonic epilepsy + symmetric cervical/shoulder lipomas — PATHOGNOMONIC combination",
        "⚠️ Progressive Myoclonic Epilepsy (PME) workup: MERRF is the most common mtDNA-PME; m.8344A>G testing MANDATORY in PME of unknown cause with maternal pedigree; other PME genes: EPM2A (Lafora), CSTB (ULD), DRPLA (ATN1)",
        "✅ LEV (Levetiracetam): preferred AED for myoclonus + generalized seizures; no mitochondrial toxicity; clonazepam adjunct for cortical myoclonus",
    ]

    return {
        "gene": "MT-TK",
        "n_patients": n,
        "seed": SEED,
        "cohort_statistics": {
            "avg_ci_activity_pct": avg_ci,
            "avg_civ_activity_pct": avg_civ,
            "avg_lactic_acid_mmolL": avg_lactate,
            "myoclonus_pct": myoclonus_pct,
            "cerebellar_ataxia_pct": ataxia_pct,
            "ragged_red_fibres_pct": rrf_pct,
            "sensorineural_hearing_loss_pct": snhl_pct,
            "dementia_pct": dementia_pct,
            "msl_pct": msl_pct,
            "peripheral_neuropathy_pct": neuropathy_pct,
            "cardiomyopathy_pct": cardiomyopathy_pct,
            "stroke_like_episode_pct": sle_pct,
        },
        "phenotype_distribution": pheno_dist,
        "cohort_summary_features": summary_features,
        "key_molecular_features": mol_features,
        "heteroplasmy_clinical_map": heteroplasmy_map,
        "clinical_alerts": alerts,
        "absolute_contraindications": [
            "VPA / Valproic Acid — ABSOLUTE CI: mt-ribosome inhibition + CoA sequestration → acute mt crisis; use LEV instead",
            "Metformin — ABSOLUTE CI: Complex I inhibitor → fatal lactic acidosis additive with CI deficiency; use insulin",
            "Linezolid — ABSOLUTE CI: mt-23S rRNA inhibitor → pan-OXPHOS collapse",
            "Chloramphenicol — ABSOLUTE CI: mt-ribosome inhibitor; same mechanism as linezolid",
            "Propofol — ABSOLUTE CI (PRIS): mitochondrial respiratory chain inhibition; use sevoflurane instead",
            "Fasting — NEVER fast in acute MT-TK crisis: GIR 6-8 mg/kg/min mandatory",
        ],
        "mandatory_acute_treatments": [
            "IV glucose (GIR 6-8 mg/kg/min) — NEVER fast in metabolic crisis",
            "IV Thiamine B1 (10-20 mg/kg) — mandatory empiric before glucose in Wernicke risk",
            "IV Levetiracetam — preferred AED for acute seizure / status myoclonus",
            "IV Clonazepam — acute cortical myoclonus adjunct (short-term only)",
            "Biotin 5-20 mg/day — pending BTBGD/SLC19A3 exclusion (mandatory empiric)",
        ],
        "maintenance_treatments": [
            "LEV (Levetiracetam) — preferred long-term AED: myoclonus + generalised seizures",
            "Clonazepam — adjunct for cortical myoclonus (tolerance risk; use cautiously long-term)",
            "Piracetam (off-label) — cortical myoclonus; not available all jurisdictions",
            "CoQ10 ubiquinol 10-20 mg/kg/day — Level C; electron transfer support",
            "Riboflavin B2 100-400 mg/day — Level C; FAD/FMN CI/CIII cofactor",
            "L-Carnitine 50-100 mg/kg/day — Level C; CoA buffering; acylcarnitine profile monitoring",
            "Thiamine B1 100-300 mg/day oral — maintenance",
        ],
    }


def get_breakdown():
    patients = _make_patients()
    n = len(patients)

    variant_rows = []
    for vname, domain, notes in VARIANTS:
        vp = [p for p in patients if p["variant"] == vname]
        if not vp:
            continue
        nv = len(vp)
        variant_rows.append({
            "variant": vname,
            "domain": domain,
            "severity": notes[:100],
            "n_patients": nv,
            "myoclonus_pct": round(sum(p["myoclonus"] for p in vp) / nv * 100, 1),
            "ataxia_pct": round(sum(p["cerebellar_ataxia"] for p in vp) / nv * 100, 1),
            "rrf_pct": round(sum(p["ragged_red_fibres"] for p in vp) / nv * 100, 1),
            "snhl_pct": round(sum(p["sensorineural_hearing_loss"] for p in vp) / nv * 100, 1),
            "dementia_pct": round(sum(p["dementia"] for p in vp) / nv * 100, 1),
            "msl_pct": round(sum(p["multiple_symmetrical_lipomatosis"] for p in vp) / nv * 100, 1),
            "lactic_acidosis_pct": round(sum(1 for p in vp if p["lactic_acid_mmolL"] > 2.5) / nv * 100, 1),
            "avg_ci_activity_pct": round(sum(p["ci_activity_pct"] for p in vp) / nv, 1),
            "avg_civ_activity_pct": round(sum(p["civ_activity_pct"] for p in vp) / nv, 1),
            "avg_heteroplasmy_blood_pct": round(sum(p["heteroplasmy_blood_pct"] for p in vp) / nv, 1),
            "avg_heteroplasmy_muscle_pct": round(sum(p["heteroplasmy_muscle_pct"] for p in vp) / nv, 1),
            "notes": notes,
        })

    # Heteroplasmy bands (blood)
    blood_bands = {"<60%": 0, "60-75%": 0, "75-90%": 0, ">90%": 0}
    muscle_bands = {"<60%": 0, "60-75%": 0, "75-90%": 0, ">90%": 0}
    for p in patients:
        for bands, key in [(blood_bands, "heteroplasmy_blood_pct"), (muscle_bands, "heteroplasmy_muscle_pct")]:
            h = p[key]
            if h < 60:
                bands["<60%"] += 1
            elif h < 75:
                bands["60-75%"] += 1
            elif h < 90:
                bands["75-90%"] += 1
            else:
                bands[">90%"] += 1

    ci_bands = {"<20%": 0, "20-40%": 0, "40-60%": 0, ">60%": 0}
    for p in patients:
        c = p["ci_activity_pct"]
        if c < 20:
            ci_bands["<20%"] += 1
        elif c < 40:
            ci_bands["20-40%"] += 1
        elif c < 60:
            ci_bands["40-60%"] += 1
        else:
            ci_bands[">60%"] += 1

    outcome_dist = Counter(p["outcome"] for p in patients)
    outcome_rows = [{"outcome": k, "count": v} for k, v in outcome_dist.most_common()]

    ddx_table = [
        {
            "entity": "MELAS Syndrome (MT-TL1 m.3243A>G — tRNA-Leu)",
            "distinguishing_feature": "MELAS: stroke-like episodes (SLE) = HALLMARK; SLE absent in MERRF; MELAS: cortical/subcortical lesions crossing vascular territories; MERRF: cerebellar atrophy + BG changes; MELAS: diabetes (MIDD) + SNHL prominent; MERRF: myoclonus = dominant seizure type (not focal); MSL absent in MELAS; m.3243A>G (tRNA-Leu) vs m.8344A>G (tRNA-Lys); Pan-OXPHOS both but MELAS more CI+CIII+CIV while MERRF more CI+CIV",
            "key_test": "m.8344A>G vs m.3243A>G targeted testing; MRI morphology (MERRF cerebellar atrophy vs MELAS cortical lesions); MSL presence; EEG (MERRF: generalised myoclonic vs MELAS: focal SLE-associated)",
        },
        {
            "entity": "Unverricht-Lundborg Disease (ULD / EPM1 — CSTB mutations)",
            "distinguishing_feature": "ULD: autosomal RECESSIVE (biallelic CSTB mutations); NO maternal pedigree; NO RRF on muscle biopsy; NO lactic acidosis; NO pan-OXPHOS deficiency; WES detects CSTB; MERRF: MATERNAL; RRF; pan-OXPHOS; lactic acidosis; MSL; m.8344A>G; Muscle biopsy: CSTB-ULD has NO mitochondrial pathology vs MERRF RRF+COX-negative fibres",
            "key_test": "Muscle biopsy (RRF/COX/SDH in MERRF; normal in ULD); m.8344A>G mtDNA testing; CSTB WES; respiratory chain enzymology; blood lactate; maternal pedigree",
        },
        {
            "entity": "Lafora Disease (EPM2A / NHLRC1 mutations)",
            "distinguishing_feature": "Lafora: autosomal recessive EPM2A/NHLRC1 mutations; Lafora bodies (periodic acid-Schiff positive inclusions) in axillary skin biopsy — PATHOGNOMONIC; rapidly progressive dementia; WES detects; NO maternal pedigree; NO RRF; NO lactic acidosis; MERRF: maternal; RRF; m.8344A>G; MSL; pan-OXPHOS",
            "key_test": "Axillary/eccrine sweat gland skin biopsy (Lafora bodies — PAS-positive); EPM2A/NHLRC1 WES; m.8344A>G mtDNA; muscle biopsy RRF (MERRF) vs normal (Lafora)",
        },
        {
            "entity": "DRPLA (Dentatorubral-Pallidoluysian Atrophy — ATN1 CAG expansion)",
            "distinguishing_feature": "DRPLA: autosomal dominant; ATN1 CAG trinucleotide repeat expansion; genetic anticipation; NO maternal-only pedigree; MRI: cerebellar + brainstem atrophy + white matter changes; NO RRF; NO lactic acidosis; NO pan-OXPHOS; diagnosis: ATN1 CAG repeat length; prevalent in Japan; MERRF: maternal; RRF; m.8344A>G; pan-OXPHOS",
            "key_test": "ATN1 CAG repeat PCR (DRPLA); m.8344A>G mtDNA (MERRF); muscle biopsy; maternal pedigree; lactic acidosis; respiratory chain enzymology",
        },
        {
            "entity": "POLG (Alpers / SANDO / progressive ataxia-neuropathy)",
            "distinguishing_feature": "POLG: autosomal recessive; hepatopathy (Alpers — VPA hepatotoxicity absolute CI in POLG too); mtDNA depletion/deletions (multiple); WES detects POLG; MERRF: maternal inheritance; pan-OXPHOS (point mutation not depletion); NO hepatopathy; MSL absent in POLG; m.8344A>G distinguishes definitively; mtDNA quantification normal in MERRF (heteroplasmy, not depletion)",
            "key_test": "Liver enzymes + bilirubin + PT (Alpers hepatopathy); mtDNA quantification (POLG depletion vs MERRF normal mtDNA copy number); POLG WES; m.8344A>G; respiratory chain enzymology",
        },
        {
            "entity": "Sporadic lipomatosis / Madelung's disease (alcohol-related)",
            "distinguishing_feature": "Sporadic MSL / Madelung's: symmetric cervical lipomas; STRONGLY associated with chronic alcohol use (mitochondrial dysfunction from ethanol); NO myoclonic epilepsy; NO RRF; NO maternal pedigree; no lactic acidosis; m.8344A>G negative; respiratory chain enzymology normal; MERRF-MSL: myoclonic epilepsy + maternal pedigree + RRF + m.8344A>G + pan-OXPHOS",
            "key_test": "Alcohol history; m.8344A>G testing; muscle biopsy (RRF absent in alcohol-MSL); respiratory chain enzymology; maternal family history of myoclonic epilepsy",
        },
        {
            "entity": "BTBGD (SLC19A3 — biotin-thiamine responsive basal ganglia disease)",
            "distinguishing_feature": "BTBGD: bilateral BG MRI identical to Leigh-like MELAS; autosomal recessive; WES detects SLC19A3; TREATABLE with biotin+thiamine; NO pan-OXPHOS; NO RRF; NO m.8344A>G; MERRF: pan-OXPHOS; RRF; maternal; m.8344A>G; BTBGD biotin-thiamine trial is MANDATORY before diagnosing any Leigh-like or mitochondrial-like syndome",
            "key_test": "SLC19A3 WES; empiric biotin+thiamine trial (BTBGD responds dramatically); m.8344A>G + respiratory chain enzymology (normal in BTBGD, pan-OXPHOS in MERRF)",
        },
    ]

    merrf_mgmt_table = [
        {"phase": "Acute status myoclonus / seizure", "treatment": "IV Levetiracetam (LEV)", "evidence": "Level C preferred AED", "notes": "No mt toxicity; preferred over VPA (absolute CI) and phenytoin; IV loading 30-60 mg/kg"},
        {"phase": "Acute cortical myoclonus", "treatment": "IV Clonazepam (short-term)", "evidence": "Level C adjunct", "notes": "Short-term only; tolerance risk; use with caution; benzodiazepine"},
        {"phase": "Acute metabolic crisis", "treatment": "IV glucose (GIR 6-8 mg/kg/min)", "evidence": "Mandatory", "notes": "NEVER fast; continuous glucose prevents energy failure; metabolic stress precipitates crisis"},
        {"phase": "Acute crisis", "treatment": "IV Thiamine B1 (10-20 mg/kg)", "evidence": "Mandatory empiric", "notes": "Before glucose in Wernicke risk; PDH cofactor; empiric in all mt disease crisis"},
        {"phase": "Acute crisis", "treatment": "IV/oral Biotin (5-20 mg)", "evidence": "Mandatory empiric", "notes": "Pending BTBGD exclusion; biotinidase deficiency must be ruled out first"},
        {"phase": "Maintenance epilepsy", "treatment": "LEV (Levetiracetam) oral", "evidence": "Level C — preferred long-term AED", "notes": "Myoclonus + generalised seizures; titrate 500-3000 mg/day; no mt toxicity"},
        {"phase": "Maintenance myoclonus", "treatment": "Piracetam (off-label)", "evidence": "Level C cortical myoclonus", "notes": "4.8-16.8 g/day; reduces cortical myoclonus; not available all jurisdictions"},
        {"phase": "Maintenance mitochondrial", "treatment": "CoQ10 ubiquinol 10-20 mg/kg/day", "evidence": "Level C", "notes": "Electron transfer; well-tolerated; ubiquinol preferred over ubiquinone"},
        {"phase": "Maintenance mitochondrial", "treatment": "Riboflavin B2 100-400 mg/day", "evidence": "Level C", "notes": "FAD/FMN cofactor for CI/CIII; no toxicity at recommended doses"},
        {"phase": "ABSOLUTE CI — myoclonic seizures", "treatment": "VPA / Valproic Acid", "evidence": "Absolute CI", "notes": "mt-ribosome inhibition + CoA sequestration → acute mt crisis; hepatotoxicity risk; NEVER in MT-TK/MERRF"},
        {"phase": "ABSOLUTE CI — all", "treatment": "Metformin", "evidence": "Absolute CI", "notes": "CI inhibitor → fatal lactic acidosis; avoid ALL biguanides; use insulin for DM"},
        {"phase": "ABSOLUTE CI — all", "treatment": "Propofol (PRIS)", "evidence": "Absolute CI", "notes": "Propofol infusion syndrome; use sevoflurane/isoflurane instead for anaesthesia"},
    ]

    return {
        "gene": "MT-TK",
        "n_patients": n,
        "seed": SEED,
        "variant_breakdown": variant_rows,
        "phenotype_distribution": {p["phenotype"]: p["count"] for p in [
            {"phenotype": ph, "count": sum(1 for pt in patients if pt["phenotype"] == ph)}
            for ph in PHENOTYPES
        ]},
        "heteroplasmy_bands_blood": blood_bands,
        "heteroplasmy_bands_muscle": muscle_bands,
        "ci_activity_bands": ci_bands,
        "outcome_distribution": outcome_rows,
        "differential_diagnosis": ddx_table,
        "merrf_management_table": merrf_mgmt_table,
        "patient_table": [
            {
                "id": p["patient_id"],
                "phenotype": p["phenotype"],
                "variant": p["variant"],
                "heteroplasmy_blood_pct": p["heteroplasmy_blood_pct"],
                "heteroplasmy_muscle_pct": p["heteroplasmy_muscle_pct"],
                "ci_pct": p["ci_activity_pct"],
                "civ_pct": p["civ_activity_pct"],
                "lactate": p["lactic_acid_mmolL"],
                "myoclonus": p["myoclonus"],
                "ataxia": p["cerebellar_ataxia"],
                "rrf": p["ragged_red_fibres"],
                "snhl": p["sensorineural_hearing_loss"],
                "msl": p["multiple_symmetrical_lipomatosis"],
                "sle": p["stroke_like_episode"],
                "outcome": p["outcome"],
            }
            for p in patients
        ],
    }


def get_definitions():
    return {
        "gene": "MT-TK",
        "omim_gene": "OMIM *590060",
        "full_name": "Mitochondrially Encoded tRNA-Lys",
        "protein_name": "tRNA-Lys (UUU/C anticodon) — 69-nucleotide RNA gene (no protein product); aminoacylates lysine onto mitochondrial ribosomes for translation of all 13 mtDNA-encoded OXPHOS subunits",
        "trna_length_nt": 69,
        "rcrs_positions": "8295-8364",
        "strand": "H-strand",
        "anticodon": "UUU/C (lysine, wobble anticodon)",
        "omim_diseases": {
            "MERRF": "MERRF Syndrome (OMIM #545000) — Myoclonic Epilepsy with Ragged Red Fibres; myoclonic epilepsy + cerebellar ataxia + RRF; dementia; SNHL; short stature; MSL; peripheral neuropathy; pan-OXPHOS (CI+CIV); m.8344A>G ~80% of cases; maternal inheritance",
            "MSL": "Multiple Symmetrical Lipomatosis (Launois-Bensaude / Madelung's) — symmetric non-encapsulated cervical/shoulder lipomas; 10-20% of MERRF; brown adipose tissue with dysfunctional mitochondria; PATHOGNOMONIC when co-occurring with myoclonic epilepsy in maternal pedigree",
            "CPEO": "Chronic Progressive External Ophthalmoplegia — ptosis + ophthalmoplegia; overlap with MERRF at high heteroplasmy or deletion variants; RRF on biopsy",
            "exercise_intolerance": "Exercise Intolerance / Myopathy — pan-OXPHOS deficiency on exertion; RRF (Gomori); elevated post-exercise lactate; may be only manifestation at low heteroplasmy",
        },
        "key_variants": {
            "m.8344A>G": "TΨC loop of tRNA-Lys — ~80-90% of MERRF worldwide; disrupts YARS2/KARS2 aminoacylation → pan-OXPHOS (CI+CIV predominantly); classic full MERRF triad at >85% blood heteroplasmy",
            "m.8356T>C": "Acceptor stem — ~10% of MERRF; structural disruption; MERRF phenotype; generally milder than m.8344A>G",
            "m.8363G>A": "T-loop — ~5%; T-loop structural perturbation; MERRF-like; variable penetrance; slower progression",
            "m.8296A>G": "Anticodon stem-loop — ~3%; diabetes + deafness (MIDD-like) rather than classic MERRF; low heteroplasmy phenotype",
            "large_deletion": "Deletion spanning MT-TK — KSS/CPEO/Pearson overlap; multi-complex OXPHOS; annual Holter mandatory",
        },
        "merrf_definition": "MERRF Syndrome: Progressive Myoclonic Epilepsy (PME) caused by mt-tRNA-Lys mutation; clinical features: myoclonic epilepsy (action + cortical myoclonus + generalised seizures), cerebellar ataxia, ragged red fibres (RRF) on muscle biopsy; plus: dementia, sensorineural hearing loss, short stature, peripheral neuropathy, Multiple Symmetrical Lipomatosis (MSL), optic atrophy, cardiomyopathy; onset typically 5-30 years; maternal family history; pan-OXPHOS (CI + CIV predominantly; CII NORMAL = mt-translation fingerprint); muscle biopsy: RRF (Gomori trichrome) + COX-negative/SDH-positive fibres",
        "msl_definition": "Multiple Symmetrical Lipomatosis (MSL) in MERRF: symmetric, non-encapsulated lipomas at cervical, parotid, shoulder, and upper-truncal locations; brown adipose tissue enriched with UCP1; dysfunctional OXPHOS in lipoma mitochondria; PATHOGNOMONIC of MERRF when co-occurring with myoclonic epilepsy in a maternal pedigree; also seen in alcohol-related Madelung's disease (exclude by alcohol history + m.8344A>G testing); clinical importance: identifies MERRF diagnosis in index case + mandates maternal cascade screening",
        "pan_oxphos_definition": "Pan-OXPHOS in MT-TK/MERRF: impaired aminoacylation of tRNA-Lys → defective translation of all 13 mtDNA-encoded OXPHOS subunits → CI + CIV predominantly reduced; CII (SDH — nuclear-encoded) NORMAL = mt-translation fingerprint identical to MT-TL1/MELAS; MERRF CI+CIV pattern differs from MT-TL1 MELAS (CI+CIII+CIV); BN-PAGE: CI + CIV reduced bands; CII normal; distinguishes from SURF1/SCO2 (isolated CIV) and MT-ND1-6 (isolated CI) and SDHA/SDHB (CII only)",
        "heteroplasmy_muscle_definition": "Muscle Heteroplasmy in MERRF: muscle (post-mitotic, high mitochondrial density) retains original heteroplasmy better than blood; blood underestimates by ~10-15% in m.8344A>G carriers (less than MELAS's 20-30%); muscle biopsy preferred in equivocal blood heteroplasmy; quantify m.8344A>G by allele-specific PCR or NGS on fresh-frozen muscle; always assess both blood AND muscle in diagnostic workup",
        "progressive_myoclonic_epilepsy_definition": "Progressive Myoclonic Epilepsy (PME) context: MERRF is the prototypical mtDNA-PME; other PME causes: Lafora (EPM2A/NHLRC1 — axillary skin biopsy Lafora bodies), ULD (CSTB — autosomal recessive), DRPLA (ATN1 CAG — Japan-prevalent), NCL (neuronal ceroid lipofuscinoses — CLN genes), Gaucher type 3 (GBA — visceral involvement); MERRF distinguished by maternal inheritance + RRF + pan-OXPHOS + m.8344A>G + MSL (if present)",
        "wes_coverage": "MT-TK is a mitochondrial tRNA gene (H-strand rCRS 8295-8364) — WES does NOT cover mitochondrial tRNA mutations; dedicated mtDNA sequencing required: (1) m.8344A>G targeted PCR for rapid diagnosis; (2) whole-mtDNA NGS panel for full coverage; (3) muscle biopsy for respiratory chain enzymology (CI+CIV in MERRF) + histochemistry (RRF/COX-negative/SDH-positive); (4) blood heteroplasmy quantitation with awareness of ~10-15% underestimation",
        "absolute_contraindications": {
            "VPA / Valproic Acid": "mt-ribosome inhibition + CoA sequestration → worsens pan-OXPHOS → acute crisis; hepatotoxicity risk; ABSOLUTE CI in ALL MT-TK/MERRF patients for seizure management; use LEV instead",
            "Metformin": "Complex I inhibitor — additive with CI deficiency in MERRF → fatal lactic acidosis; ABSOLUTE CI in all MT-TK carriers regardless of phenotype; use insulin for DM",
            "Linezolid": "Inhibits mt-23S rRNA → collapses mt-translation of all 13 OXPHOS subunits — catastrophic in MT-TK pan-OXPHOS context",
            "Chloramphenicol": "Mt-ribosome inhibitor — same mechanism as linezolid; ABSOLUTE CI",
            "Propofol": "PRIS (propofol infusion syndrome) + direct ETC inhibition; use sevoflurane/isoflurane for anaesthesia in MERRF/MT-TK",
            "Fasting": "GIR 6-8 mg/kg/min MANDATORY in acute crisis — NEVER fast; metabolic stress precipitates acute mt decompensation",
        },
        "recommended_treatments": {
            "lev_preferred_aed": "Levetiracetam — preferred AED (Level C); myoclonus + generalised seizures; 500-3000 mg/day; no mt toxicity",
            "clonazepam_adjunct": "Clonazepam — adjunct for cortical myoclonus (short-term; tolerance risk); 0.5-8 mg/day",
            "piracetam": "Piracetam (off-label) — cortical myoclonus Level C; 4.8-16.8 g/day",
            "coq10_ubiquinol": "Level C — 10-20 mg/kg/day ubiquinol",
            "riboflavin_b2": "Level C — 100-400 mg/day; FAD/FMN cofactor CI/CIII",
            "l_carnitine": "Level C — 50-100 mg/kg/day; CoA buffering; monitor acylcarnitine",
            "thiamine_b1": "Mandatory empiric — 10-20 mg/kg IV crisis; 100-300 mg/day maintenance",
            "biotin": "5-20 mg/day empiric — pending BTD/SLC19A3 exclusion",
            "gir": "GIR 6-8 mg/kg/min — NEVER fast in crisis",
        },
        "specialist_monitoring": {
            "Neurology": "MERRF: EEG (generalized myoclonic pattern; 4-6 Hz slow spike-wave; photosensitivity); brain MRI 12-monthly (cerebellar atrophy progression + BG changes); cognitive assessment; seizure diary",
            "Cardiology": "Annual ECG + Holter — cardiomyopathy 15-20%; conduction block (KSS deletion variant — pacemaker threshold PR >240ms); echocardiography annually if cardiomyopathy suspected",
            "Audiology": "Annual audiometry — SNHL in 40-60% MERRF; cochlear implant evaluation in profound loss",
            "Ophthalmology": "Annual fundus — optic atrophy 15-25%; pigmentary retinopathy (CPEO overlap); ptosis assessment (CPEO variant)",
            "Genetics": "Maternal cascade testing MANDATORY; m.8344A>G blood quantitation all maternal relatives; muscle biopsy if equivocal; pre-conceptional counselling (heteroplasmy segregation unpredictable — mtDNA bottleneck)",
            "Endocrinology": "m.8296A>G variant: DM + SNHL workup; insulin not metformin if DM develops in ANY MT-TK carrier",
            "Lipomatology / Surgery": "MSL: clinical surveillance; surgery cosmetic only (lipomas recur); document new lipoma sites; correlate with heteroplasmy; no medical regression therapy known",
        },
        "key_references": [
            "Shoffner JM et al. (1990) Myoclonic epilepsy and ragged-red fiber disease (MERRF) is associated with a mitochondrial DNA tRNA(Lys) mutation. Cell 61(6):931-937. [m.8344A>G FIRST DESCRIPTION — seminal paper defining MERRF/MT-TK link; Shoffner & Wallace group]",
            "Wallace DC et al. (1988) Mitochondrial DNA mutation associated with Leber's hereditary optic neuropathy. Science 242(4884):1427-1430. [Wallace group — foundational mtDNA disease framework preceding MERRF discovery]",
            "Lorenzoni PJ et al. (2014) MERRF: a journey in the neurological features of a MERRF/MELAS overlap syndrome and a review of clinical, biochemical, and genetic aspects. Arq Neuropsiquiatr 72(10):798-801. [MERRF clinical spectrum including overlap phenotypes]",
            "DiMauro S & Schon EA (2003) Mitochondrial respiratory-chain diseases. N Engl J Med 348(26):2656-2668. [Comprehensive review MERRF + other mtDNA diseases; treatment framework]",
            "Mancuso M et al. (2013) MERRF syndrome: contribution of m.8344A>G mutation to the phenotype variability. Neuromuscul Disord 23(6):497-502. [m.8344A>G heteroplasmy-phenotype correlation; MERRF variability]",
            "Klopstock T et al. (1994) A mutation in mitochondrial tRNA(Arg) associated with symmetric lipomatosis. Nat Genet 7(1):31-32. [MSL and mitochondrial tRNA mutations; Madelung-MERRF link]",
        ],
        "cohort_seed": SEED,
        "n_patients": N_PATIENTS,
        "generated": "2026-09-03",
    }


if __name__ == "__main__":
    import json
    overview = get_overview()
    print(f"MT-TK overview: {overview['n_patients']} patients, "
          f"avg CI {overview['cohort_statistics']['avg_ci_activity_pct']}%, "
          f"avg lactate {overview['cohort_statistics']['avg_lactic_acid_mmolL']} mmol/L")
    print(f"Myoclonus: {overview['cohort_statistics']['myoclonus_pct']}%")
    print(f"Ataxia: {overview['cohort_statistics']['cerebellar_ataxia_pct']}%")
    print(f"RRF: {overview['cohort_statistics']['ragged_red_fibres_pct']}%")
    print(f"MSL: {overview['cohort_statistics']['msl_pct']}%")
    print(f"SLE: {overview['cohort_statistics']['stroke_like_episode_pct']}% (must be 0 — MERRF not MELAS)")
    bd = get_breakdown()
    print(f"Variants: {len(bd['variant_breakdown'])}")
    defs = get_definitions()
    print(f"Key variants: {list(defs['key_variants'].keys())}")
    print("✅ MT-TK dashboard OK")
