#!/usr/bin/env python3
"""NARP — Neuropathy, Ataxia, and Retinitis Pigmentosa (MT-ATP6 / Complex V Disease).

Mitochondrial DNA Complex V Disease — maternal inheritance:
  NARP (Neuropathy, Ataxia, Retinitis Pigmentosa) OMIM #551500
  MILS (Maternally Inherited Leigh Syndrome) OMIM #500017
  Primary mutations: MT-ATP6 m.8993T>G (most common, ~70%) / m.8993T>C (milder, ~25%) / m.9176T>C (rare, ~5%)
  Gene: MT-ATP6 OMIM *516060

PATHOPHYSIOLOGY (Complex V defect):
MT-ATP6 encodes subunit a of the F0 component of mitochondrial ATP synthase (Complex V).
Subunit a (= subunit 6 / ATP6) lines the a-c10 interface in the F0 transmembrane rotor,
forming the proton half-channels that direct H+ flow through the c-ring. This rotary
proton translocation drives the F1 catalytic head to synthesise ATP from ADP + Pi.

The m.8993T>G mutation (p.Leu156Arg) introduces a positive charge in transmembrane
helix 4 of subunit a, disrupting the proton-conducting channel geometry and reducing
ATP synthase activity by 50-80%. The m.8993T>C mutation (p.Leu156Pro) causes a less
severe structural perturbation (~30-50% activity reduction), explaining the milder
NARP phenotype and lower penetrance in families.

HETEROPLASMY THRESHOLD (unique genotype-phenotype paradigm):
The clinical phenotype depends on the proportion of mutant to wild-type mtDNA:
  <70% mutant  → ASYMPTOMATIC or subclinical RP/neuropathy (carrier state)
  70-90% mutant → NARP: Neuropathy + Ataxia + Retinitis Pigmentosa (classic triad)
  >90% mutant  → MILS: Maternally Inherited Leigh Syndrome (neonatal/infantile, fatal)
This threshold relationship means maternal relatives of a MILS patient can have
NARP (intermediate heteroplasmy) or be asymptomatic carriers (<70%) — the SAME
family can span all three phenotypic categories. Blood heteroplasmy may underestimate
neuronal heteroplasmy due to the mitochondrial bottleneck during oogenesis.

NARP CLINICAL PRESENTATION:
  1. Retinitis Pigmentosa — rod-cone dystrophy; bone-spicule pigmentation; salt-and-pepper
     fundus early; macular degeneration; nyctalopia (night blindness) first symptom;
     reduced ERG amplitudes (rod > cone); CONTRAST with LHON (central vision loss, no RP)
  2. Cerebellar Ataxia — gait ataxia, truncal instability, dysarthria; variable progression
  3. Peripheral Neuropathy — sensory > motor; axonal pattern (EMG/NCS); reduced reflexes
  4. Epilepsy — 40% of NARP patients; focal > myoclonic; triggered by fever/fasting
  5. Cognitive decline / intellectual disability — 45%; variable severity
  6. Exercise intolerance — 60%; lactic acidosis on exertion
  7. SNHL — 35%; progressive sensorineural

MILS CLINICAL PRESENTATION (>90% heteroplasmy):
  1. Leigh syndrome MRI — 95%; bilateral symmetric T2-hyperintense lesions in putamen,
     caudate, brainstem (periaqueductal grey, inferior olivary nuclei), cerebellum
  2. Lactic acidosis — 100%; pH <7.2; high lactate:pyruvate ratio (>20)
  3. Onset — neonatal to early infancy; acute decompensation with febrile illness
  4. Hypotonia, developmental regression, feeding difficulties
  5. High mortality — median survival without treatment <3 years

KEY PRESCRIBING SAFETY:
  VPA ABSOLUTE CONTRAINDICATION (ALL MT-ATP6, both NARP and MILS)
  VGB ABSOLUTE CONTRAINDICATION (NARP only — RP present — retinal additive toxicity)
  KD CONTRAINDICATED (forces OXPHOS-dependent fatty acid beta-oxidation)
  Propofol AVOID PRIS (mitochondrial disease — all genotypes)
"""
from __future__ import annotations
import random
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────
SEED        = 585
N_PATIENTS  = 40
DISEASE     = "NARP (Neuropathy, Ataxia, Retinitis Pigmentosa) / MILS (Maternally Inherited Leigh Syndrome)"
GENE        = "MT-ATP6"
OMIM_GENE   = "*516060"
OMIM_NARP   = "#551500"
OMIM_MILS   = "#500017"
LOCUS       = "Mitochondrial DNA (mtDNA) — maternal inheritance"

# Mutation pool (m.8993T>G most common, m.8993T>C milder, m.9176T>C rare)
MUTATION_POOL    = ["m.8993T>G (p.Leu156Arg)", "m.8993T>C (p.Leu156Pro)", "m.9176T>C (p.Leu217Pro)"]
MUTATION_WEIGHTS = [0.70, 0.25, 0.05]

# Heteroplasmy range → phenotype:
# >90% → MILS; 70-90% → NARP; <70% → subclinical/carrier
PHENO_MILS = "MILS (Maternally Inherited Leigh Syndrome)"
PHENO_NARP = "NARP (Neuropathy + Ataxia + RP)"
PHENO_CARRIER = "Subclinical / Carrier"

def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient NARP/MILS cohort (seed-585)."""
    return random.Random(SEED)


def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient MT-ATP6 disease cohort (seed-585, ~80% NARP, ~15% MILS, ~5% subclinical)."""
    patients = []
    for i in range(N_PATIENTS):
        pid = f"NAP-{i+1:03d}"
        mutation = rng.choices(MUTATION_POOL, weights=MUTATION_WEIGHTS, k=1)[0]
        # Heteroplasmy — determines phenotype
        # NARP cohort enriched for symptomatic (60-100% range)
        raw_level = rng.uniform(60, 100)
        heteroplasmy_pct = round(raw_level, 1)
        if heteroplasmy_pct > 90:
            phenotype = PHENO_MILS
        elif heteroplasmy_pct >= 70:
            phenotype = PHENO_NARP
        else:
            phenotype = PHENO_CARRIER

        # Sex — equally distributed (NARP equal sex penetrance, unlike LHON)
        sex = "F" if rng.random() < 0.50 else "M"

        # Age of onset — NARP: young adult; MILS: infantile
        if phenotype == PHENO_MILS:
            onset_age = round(rng.uniform(0, 1.5), 1)
        elif phenotype == PHENO_NARP:
            # m.8993T>G tends earlier onset
            mu = 18 if "8993T>G" in mutation else 25
            onset_age = max(2.0, round(rng.gauss(mu, 7), 1))
        else:
            onset_age = round(rng.gauss(30, 10), 1)

        # Current age
        current_age = round(onset_age + rng.uniform(0, 20), 1)

        # Diagnosis delay
        dx_delay = round(rng.uniform(0.5, 6) if phenotype != PHENO_MILS else rng.uniform(0, 1), 1)

        # Clinical features (NARP triad + extras)
        rp             = rng.random() < (0.90 if phenotype == PHENO_NARP else 0.15 if phenotype == PHENO_MILS else 0.30)
        ataxia         = rng.random() < (0.85 if phenotype == PHENO_NARP else 0.70 if phenotype == PHENO_MILS else 0.10)
        neuropathy     = rng.random() < (0.85 if phenotype == PHENO_NARP else 0.50 if phenotype == PHENO_MILS else 0.10)
        epilepsy       = rng.random() < (0.40 if phenotype == PHENO_NARP else 0.70 if phenotype == PHENO_MILS else 0.05)
        cognitive      = rng.random() < (0.45 if phenotype == PHENO_NARP else 0.90 if phenotype == PHENO_MILS else 0.10)
        snhl           = rng.random() < (0.35 if phenotype == PHENO_NARP else 0.20 if phenotype == PHENO_MILS else 0.05)
        cardiomyopathy = rng.random() < (0.20 if phenotype == PHENO_NARP else 0.45 if phenotype == PHENO_MILS else 0.05)
        weakness       = rng.random() < (0.60 if phenotype == PHENO_NARP else 0.75 if phenotype == PHENO_MILS else 0.15)
        leigh_mri      = rng.random() < (0.10 if phenotype == PHENO_NARP else 0.95 if phenotype == PHENO_MILS else 0.00)

        # Lactic acidosis (resting or on exertion)
        lactic_acidosis = rng.random() < (0.55 if phenotype == PHENO_NARP else 0.95 if phenotype == PHENO_MILS else 0.10)

        # Treatment
        on_coq10     = rng.random() < 0.60
        on_idebenone = rng.random() < 0.35
        on_thiamine  = rng.random() < 0.45
        on_lev       = epilepsy and rng.random() < 0.70
        vpa_exposure = rng.random() < 0.07  # inadvertent VPA (always adverse event)

        # VA (NARP: RP-related visual loss; MILS: may have cortical visual impairment)
        if rp:
            va_categories = ["6/6-6/12", "6/12-6/24", "6/24-6/60", "<6/60", "Hand Motion/Perception"]
            va_weights    = [0.20, 0.30, 0.25, 0.15, 0.10]
        else:
            va_categories = ["6/6-6/12", "6/12-6/24", "6/24-6/60", "<6/60"]
            va_weights    = [0.60, 0.25, 0.10, 0.05]
        visual_acuity = rng.choices(va_categories, weights=va_weights, k=1)[0]

        patients.append({
            "pid": pid, "mutation": mutation, "heteroplasmy_pct": heteroplasmy_pct,
            "phenotype": phenotype, "sex": sex, "age_onset_years": onset_age,
            "current_age": current_age, "dx_delay_yr": dx_delay,
            "rp": rp, "ataxia": ataxia, "neuropathy": neuropathy, "epilepsy": epilepsy,
            "cognitive_decline": cognitive, "snhl": snhl, "cardiomyopathy": cardiomyopathy,
            "weakness": weakness, "leigh_mri": leigh_mri, "lactic_acidosis": lactic_acidosis,
            "on_coq10": on_coq10, "on_idebenone": on_idebenone, "on_thiamine": on_thiamine,
            "on_lev": on_lev, "vpa_exposure": vpa_exposure, "visual_acuity": visual_acuity,
        })
    return patients


# ── API functions ─────────────────────────────────────────────────────────────

def get_overview() -> dict[str, Any]:
    rng = _rng()
    patients = _build_cohort(rng)

    n_narp      = sum(1 for p in patients if p["phenotype"] == PHENO_NARP)
    n_mils      = sum(1 for p in patients if p["phenotype"] == PHENO_MILS)
    n_carrier   = sum(1 for p in patients if p["phenotype"] == PHENO_CARRIER)
    n_female    = sum(1 for p in patients if p["sex"] == "F")
    n_rp        = sum(1 for p in patients if p["rp"])
    n_ataxia    = sum(1 for p in patients if p["ataxia"])
    n_neuropathy= sum(1 for p in patients if p["neuropathy"])
    n_epilepsy  = sum(1 for p in patients if p["epilepsy"])
    n_leigh_mri = sum(1 for p in patients if p["leigh_mri"])
    n_vpa_exp   = sum(1 for p in patients if p["vpa_exposure"])
    n_8993tg    = sum(1 for p in patients if "8993T>G" in p["mutation"])
    n_8993tc    = sum(1 for p in patients if "8993T>C" in p["mutation"])
    n_9176tc    = sum(1 for p in patients if "9176T>C" in p["mutation"])

    avg_het    = round(sum(p["heteroplasmy_pct"] for p in patients) / N_PATIENTS, 1)
    avg_onset  = round(sum(p["age_onset_years"]  for p in patients) / N_PATIENTS, 1)
    avg_delay  = round(sum(p["dx_delay_yr"]      for p in patients) / N_PATIENTS, 1)

    return {
        "gene": "MT-ATP6",
        "protein": (
            "MT-ATP6: ATP synthase subunit a — 227 aa (subunit 6/a of F0-ATP synthase Complex V, "
            "mtDNA-encoded; forms the a-c10 interface proton half-channels in F0 transmembrane domain; "
            "together with the c-ring constitutes the rotary proton translocator driving F1 catalysis)"
        ),
        "disease": (
            "NARP (Neuropathy, Ataxia, Retinitis Pigmentosa) OMIM #551500 / "
            "MILS (Maternally Inherited Leigh Syndrome) OMIM #500017 — "
            "primary mtDNA mutations in MT-ATP6 impairing mitochondrial ATP synthase (Complex V) "
            "proton translocation; phenotype determined by heteroplasmy threshold (70-90% → NARP; "
            ">90% → MILS; <70% → subclinical/carrier)"
        ),
        "omim_gene": f"{OMIM_GENE} (MT-ATP6)",
        "omim_disease": f"{OMIM_NARP} (NARP) / {OMIM_MILS} (MILS)",
        "chromosome": LOCUS,
        "inheritance": (
            "MATERNAL (mitochondrial) — all children of affected/carrier MOTHER inherit the mtDNA "
            "mutation; fathers do NOT transmit; EQUAL sex penetrance in NARP/MILS (CONTRAST with "
            "LHON: 80-90% male); phenotype severity determined by HETEROPLASMY LEVEL, not sex; "
            "maternal relatives can span asymptomatic carrier → NARP → MILS within same family "
            "depending on heteroplasmy bottleneck during oogenesis"
        ),
        "onset": (
            f"NARP: typically teens to young adults (mean {avg_onset:.1f} yr in this cohort); "
            "onset range 5-50 yr depending on heteroplasmy; MILS: neonatal to early infancy (<18 months); "
            f"mean dx delay: {avg_delay:.1f} yr (NARP often misdiagnosed as hereditary ataxia or RP)"
        ),
        "mechanism": (
            "MT-ATP6 p.Leu156Arg (m.8993T>G) introduces a positive charge in TM helix 4 of subunit a "
            "→ disrupts the proton-conducting a-c interface → impairs H+ translocation through the c-ring "
            "→ reduces rotary torque driving F1 catalysis → ATP synthesis ↓50-80% → energy failure in "
            "high-demand post-mitotic cells (retinal photoreceptors, cerebellar Purkinje cells, dorsal "
            "root ganglion neurons, cochlear hair cells); p.Leu156Pro (m.8993T>C) → milder structural "
            "disruption → 30-50% ATP synthesis reduction → NARP at higher threshold"
        ),
        "mtdna_pattern": (
            "Primary mutations: m.8993T>G MT-ATP6 (~70% of NARP families) / "
            "m.8993T>C MT-ATP6 (~25%) / m.9176T>C MT-ATP6 (~5%); "
            "HETEROPLASMIC in blood (unlike LHON homoplasmy); "
            "mtDNA copy number NORMAL (no depletion); no mtDNA deletions; "
            "same cell contains both wild-type and mutant mtDNA — ratio determines phenotype"
        ),
        "cohort": f"40 patients · seed-{SEED}",
        "n_patients": N_PATIENTS,
        "kpis": [
            {"label": "NARP phenotype",         "value": f"{n_narp}/40",  "color": "#e65100"},
            {"label": "MILS phenotype",          "value": f"{n_mils}/40",  "color": "#b71c1c"},
            {"label": "Retinitis Pigmentosa",    "value": f"{round(100*n_rp/N_PATIENTS)}%",        "color": "#4a148c"},
            {"label": "Cerebellar Ataxia",       "value": f"{round(100*n_ataxia/N_PATIENTS)}%",    "color": "#1565c0"},
            {"label": "Peripheral Neuropathy",   "value": f"{round(100*n_neuropathy/N_PATIENTS)}%","color": "#2e7d32"},
            {"label": "Epilepsy",                "value": f"{round(100*n_epilepsy/N_PATIENTS)}%",  "color": "#880e4f"},
            {"label": "Leigh MRI (any)",         "value": f"{round(100*n_leigh_mri/N_PATIENTS)}%", "color": "#37474f"},
            {"label": "VPA exposure (adverse)",  "value": f"{n_vpa_exp}/40", "color": "#c62828"},
        ],
        "feature_bars": [
            {"label": "Retinitis Pigmentosa (rod-cone dystrophy — cardinal NARP)", "pct": round(100*n_rp/N_PATIENTS)},
            {"label": "Cerebellar ataxia (gait + truncal)", "pct": round(100*n_ataxia/N_PATIENTS)},
            {"label": "Peripheral neuropathy (sensory > motor, axonal)", "pct": round(100*n_neuropathy/N_PATIENTS)},
            {"label": "Weakness / exercise intolerance", "pct": round(100*sum(1 for p in patients if p["weakness"])/N_PATIENTS)},
            {"label": "Lactic acidosis (resting or exertional)", "pct": round(100*sum(1 for p in patients if p["lactic_acidosis"])/N_PATIENTS)},
            {"label": "Epilepsy (focal/myoclonic)", "pct": round(100*n_epilepsy/N_PATIENTS)},
            {"label": "Cognitive decline / intellectual disability", "pct": round(100*sum(1 for p in patients if p["cognitive_decline"])/N_PATIENTS)},
            {"label": "SNHL (sensorineural hearing loss)", "pct": round(100*sum(1 for p in patients if p["snhl"])/N_PATIENTS)},
            {"label": "Cardiomyopathy (HCM/DCM)", "pct": round(100*sum(1 for p in patients if p["cardiomyopathy"])/N_PATIENTS)},
            {"label": "Leigh syndrome MRI pattern (any)", "pct": round(100*n_leigh_mri/N_PATIENTS)},
            {"label": "m.8993T>G (most common / most severe)", "pct": round(100*n_8993tg/N_PATIENTS)},
            {"label": "m.8993T>C (milder / higher threshold)", "pct": round(100*n_8993tc/N_PATIENTS)},
        ],
        "stats": {
            "avg_heteroplasmy_pct": avg_het,
            "avg_onset_yr": avg_onset,
            "avg_dx_delay_yr": avg_delay,
            "n_narp": n_narp,
            "n_mils": n_mils,
            "n_carrier": n_carrier,
            "n_8993tg": n_8993tg,
            "n_8993tc": n_8993tc,
            "n_9176tc": n_9176tc,
            "n_female": n_female,
            "n_male": N_PATIENTS - n_female,
            "vpa_adverse_events": n_vpa_exp,
        },
        "ddx_comparison": [
            {
                "feature": "Inheritance",
                "narp": "Maternal (mtDNA) — all children of carrier mother",
                "lhon": "Maternal (mtDNA) — same mechanism",
                "opa1_adoa": "Autosomal Dominant (nuclear OPA1 3q29)",
                "friedreich": "Autosomal Recessive (nuclear FRDA 9q21)",
            },
            {
                "feature": "Visual loss type",
                "narp": "Retinitis Pigmentosa (rod-cone dystrophy) — peripheral first",
                "lhon": "Central vision loss (centrocaecal scotoma, RGC degeneration) — CONTRAST",
                "opa1_adoa": "Optic atrophy (temporal disc pallor, blue-yellow dyschromatopsia)",
                "friedreich": "Usually no severe RP (optic atrophy mild/late)",
            },
            {
                "feature": "Ataxia",
                "narp": "Cerebellar — prominent (85%)",
                "lhon": "Absent in isolated LHON",
                "opa1_adoa": "40% in OPA1-Plus (multisystem)",
                "friedreich": "Spinocerebellar — CARDINAL (100%)",
            },
            {
                "feature": "Neuropathy",
                "narp": "Axonal sensory > motor (85%)",
                "lhon": "Subclinical at most in isolated LHON",
                "opa1_adoa": "35% peripheral neuropathy in OPA1-Plus",
                "friedreich": "Sensory axonal — CARDINAL (100%)",
            },
            {
                "feature": "Cardiomyopathy",
                "narp": "20% NARP; 45% MILS (HCM)",
                "lhon": "Rare (<5%)",
                "opa1_adoa": "Rare",
                "friedreich": "HCM — CARDINAL (80-90%) — KEY DDx",
            },
            {
                "feature": "Heteroplasmy",
                "narp": "HETEROPLASMIC — threshold determines NARP vs MILS vs carrier",
                "lhon": "Usually HOMOPLASMIC (~85%); 15% heteroplasmic",
                "opa1_adoa": "N/A — nuclear AD gene, no heteroplasmy concept",
                "friedreich": "N/A — nuclear recessive (GAA repeat expansion)",
            },
            {
                "feature": "Leigh MRI",
                "narp": "MILS (>90% heteroplasmy): 95% Leigh pattern; NARP: 10%",
                "lhon": "Absent in LHON",
                "opa1_adoa": "Absent in pure ADOA",
                "friedreich": "Absent; cerebellar/dorsal column atrophy instead",
            },
            {
                "feature": "Sex bias",
                "narp": "EQUAL — M=F (CONTRAST with LHON)",
                "lhon": "80-90% MALE (X-linked modifier Xq21) — KEY DDx",
                "opa1_adoa": "Equal (AD, no sex modifier)",
                "friedreich": "Equal (AR)",
            },
        ],
    }


def get_breakdown() -> dict[str, Any]:
    rng = _rng()
    patients = _build_cohort(rng)

    # Per-mutation breakdown
    mut_summary = {}
    for label, key in [("m.8993T>G (p.Leu156Arg)", "8993T>G"),
                       ("m.8993T>C (p.Leu156Pro)", "8993T>C"),
                       ("m.9176T>C (p.Leu217Pro)", "9176T>C")]:
        pts = [p for p in patients if key in p["mutation"]]
        if pts:
            mut_summary[label] = {
                "n": len(pts),
                "pct": round(100 * len(pts) / N_PATIENTS),
                "n_mils": sum(1 for p in pts if p["phenotype"] == PHENO_MILS),
                "n_narp": sum(1 for p in pts if p["phenotype"] == PHENO_NARP),
                "avg_het": round(sum(p["heteroplasmy_pct"] for p in pts) / len(pts), 1),
                "avg_onset": round(sum(p["age_onset_years"] for p in pts) / len(pts), 1),
            }

    # Phenotype breakdown
    pheno_breakdown = {}
    for pheno in [PHENO_NARP, PHENO_MILS, PHENO_CARRIER]:
        pts = [p for p in patients if p["phenotype"] == pheno]
        if pts:
            pheno_breakdown[pheno] = {
                "n": len(pts),
                "pct": round(100 * len(pts) / N_PATIENTS),
                "avg_heteroplasmy": round(sum(p["heteroplasmy_pct"] for p in pts) / len(pts), 1),
                "pct_rp":     round(100 * sum(1 for p in pts if p["rp"]) / len(pts)),
                "pct_ataxia": round(100 * sum(1 for p in pts if p["ataxia"]) / len(pts)),
                "pct_neuro":  round(100 * sum(1 for p in pts if p["neuropathy"]) / len(pts)),
                "pct_epil":   round(100 * sum(1 for p in pts if p["epilepsy"]) / len(pts)),
                "pct_leigh":  round(100 * sum(1 for p in pts if p["leigh_mri"]) / len(pts)),
            }

    # Heteroplasmy distribution
    het_bins = {"<70%": 0, "70-80%": 0, "80-90%": 0, "90-95%": 0, ">95%": 0}
    for p in patients:
        h = p["heteroplasmy_pct"]
        if h < 70:
            het_bins["<70%"] += 1
        elif h < 80:
            het_bins["70-80%"] += 1
        elif h < 90:
            het_bins["80-90%"] += 1
        elif h < 95:
            het_bins["90-95%"] += 1
        else:
            het_bins[">95%"] += 1

    # Feature frequency bars
    feature_summary = {
        "Retinitis Pigmentosa": round(100 * sum(1 for p in patients if p["rp"]) / N_PATIENTS),
        "Cerebellar Ataxia": round(100 * sum(1 for p in patients if p["ataxia"]) / N_PATIENTS),
        "Peripheral Neuropathy": round(100 * sum(1 for p in patients if p["neuropathy"]) / N_PATIENTS),
        "Weakness/Exercise Intolerance": round(100 * sum(1 for p in patients if p["weakness"]) / N_PATIENTS),
        "Lactic Acidosis": round(100 * sum(1 for p in patients if p["lactic_acidosis"]) / N_PATIENTS),
        "Epilepsy": round(100 * sum(1 for p in patients if p["epilepsy"]) / N_PATIENTS),
        "Cognitive Decline/ID": round(100 * sum(1 for p in patients if p["cognitive_decline"]) / N_PATIENTS),
        "SNHL": round(100 * sum(1 for p in patients if p["snhl"]) / N_PATIENTS),
        "Cardiomyopathy": round(100 * sum(1 for p in patients if p["cardiomyopathy"]) / N_PATIENTS),
        "Leigh MRI (any)": round(100 * sum(1 for p in patients if p["leigh_mri"]) / N_PATIENTS),
    }

    # Treatment usage
    treatment_summary = {
        "CoQ10/ubiquinol (Level C)": round(100 * sum(1 for p in patients if p["on_coq10"]) / N_PATIENTS),
        "Idebenone (Level C)": round(100 * sum(1 for p in patients if p["on_idebenone"]) / N_PATIENTS),
        "Thiamine B1 (Level C)": round(100 * sum(1 for p in patients if p["on_thiamine"]) / N_PATIENTS),
        "LEV (preferred AED)": round(100 * sum(1 for p in patients if p["on_lev"]) / N_PATIENTS),
        "VPA exposure (ADVERSE EVENT)": round(100 * sum(1 for p in patients if p["vpa_exposure"]) / N_PATIENTS),
    }

    # Visual acuity distribution
    va_dist: dict[str, int] = {}
    for p in patients:
        va = p["visual_acuity"]
        va_dist[va] = va_dist.get(va, 0) + 1

    # Per-patient table (abbreviated)
    table = []
    for p in patients:
        pheno_short = "MILS" if p["phenotype"] == PHENO_MILS else ("NARP" if p["phenotype"] == PHENO_NARP else "Carrier")
        flags = []
        if p["rp"]:        flags.append("RP")
        if p["ataxia"]:    flags.append("Atx")
        if p["neuropathy"]:flags.append("Nrp")
        if p["epilepsy"]:  flags.append("Epi")
        if p["leigh_mri"]: flags.append("Leigh-MRI")
        if p["vpa_exposure"]: flags.append("⚠VPA-exposure")
        table.append({
            "pid": p["pid"],
            "mutation": p["mutation"],
            "het_pct": p["heteroplasmy_pct"],
            "phenotype": pheno_short,
            "sex": p["sex"],
            "onset_yr": p["age_onset_years"],
            "va": p["visual_acuity"],
            "flags": " · ".join(flags) if flags else "—",
        })

    return {
        "mutation_summary": mut_summary,
        "phenotype_breakdown": pheno_breakdown,
        "heteroplasmy_distribution": het_bins,
        "feature_frequencies": feature_summary,
        "treatment_summary": treatment_summary,
        "va_distribution": va_dist,
        "patients": table,
    }


def get_definitions() -> dict[str, Any]:
    return {
        "pharmacology": [
            {
                "term": "VPA — ABSOLUTE CONTRAINDICATION (ALL MT-ATP6 genotypes: NARP and MILS)",
                "definition": (
                    "Valproic acid (valproate/VPA/sodium valproate) is ABSOLUTELY CONTRAINDICATED in "
                    "ALL MT-ATP6 disease (NARP and MILS). VPA inhibits mitochondrial beta-oxidation "
                    "at multiple steps: (1) VPA metabolite propionate → inhibits succinyl-CoA synthetase "
                    "→ depletes free CoA → blocks β-oxidation at acyl-CoA dehydrogenase steps → reduces "
                    "FADH2/NADH supply to Complex II/I; (2) VPA → intramitochondrial acyl-CoA "
                    "sequestration → depletes free CoA pool → impairs Krebs cycle; (3) direct Complex V "
                    "inhibition reported in vitro. In NARP/MILS, Complex V is already dysfunctional "
                    "→ VPA compounds ATP synthesis failure → documented acute metabolic decompensation "
                    "and fulminant hepatotoxicity (Reye-like) in MILS. ALTERNATIVE AEDs: "
                    "LEV (first-line, renal excretion, zero mito toxicity), CLB (hepatic but no mito "
                    "toxicity), TPM (Level C), ZNS (Level C). Never prescribe VPA in suspected or "
                    "confirmed MT-ATP6 disease regardless of phenotype (NARP or MILS)."
                ),
            },
            {
                "term": "Vigabatrin — ABSOLUTE CONTRAINDICATION in NARP (RP present)",
                "definition": (
                    "Vigabatrin (VGB) is ABSOLUTELY CONTRAINDICATED in NARP because of its irreversible "
                    "retinal toxicity. VGB irreversibly inhibits GABA transaminase (GABA-T) → chronic "
                    "GABA excess in retina → retinal GABA receptor downregulation → amacrine cell + "
                    "retinal ganglion cell + photoreceptor loss → permanent bitemporal visual field "
                    "constriction (annular scotoma). In NARP, the retina (rod and cone photoreceptors) "
                    "is ALREADY degenerating from Complex V failure (retinitis pigmentosa). VGB adds "
                    "a second independent mechanism of photoreceptor damage → ADDITIVE irreversible "
                    "visual loss. ERG monitoring (mandatory for VGB in general population) does NOT "
                    "protect NARP patients adequately because baseline ERG is already subnormal from RP. "
                    "Alternative AED for epilepsy with NARP: LEV first-line; CLB second-line; "
                    "ACTH (if infantile spasms in MILS, as VGB would be standard but is CI here). "
                    "Note: VGB is NOT absolutely CI in MILS without RP (Leigh syndrome phenotype without "
                    "significant RP) but the overlap makes VGB hazardous in ANY MT-ATP6 patient."
                ),
            },
            {
                "term": "Ketogenic Diet — CONTRAINDICATED in NARP/MILS",
                "definition": (
                    "The ketogenic diet (KD) forces fatty acid beta-oxidation as the primary energy "
                    "substrate. Fatty acid beta-oxidation generates FADH2 (Complex II) and NADH "
                    "(Complex I), feeding into the electron transport chain; the resulting proton "
                    "gradient drives Complex V (ATP synthase) to synthesise ATP. In NARP/MILS, "
                    "Complex V is the primary defect → the cell cannot efficiently use the proton "
                    "gradient to make ATP, regardless of how much substrate is provided → KD increases "
                    "OXPHOS substrate load without improving ATP output → worsening energy failure "
                    "and potential ketoacidosis. KD is contraindicated in ALL primary Complex V "
                    "disorders (MT-ATP6 NARP/MILS, ATPAF2, ATP5A1 Leigh syndrome) and in "
                    "other mitochondrial disorders where oxidative phosphorylation is the rate-limiting "
                    "step. Modified Atkins diet (less ketogenic) is similarly contraindicated."
                ),
            },
            {
                "term": "Propofol — AVOID / PRIS Risk in NARP/MILS",
                "definition": (
                    "Propofol (2,6-diisopropylphenol) causes propofol infusion syndrome (PRIS) at "
                    "rates >4-5 mg/kg/hr for >48h, characterised by: metabolic acidosis, rhabdomyolysis, "
                    "cardiac failure, and death. Mechanism: propofol uncouples oxidative phosphorylation "
                    "and inhibits Complex I and Complex IV of the electron transport chain. In mitochondrial "
                    "disease including NARP/MILS, any OXPHOS compromise is amplified → standard propofol "
                    "doses that are tolerated in healthy patients can cause acute energy failure. "
                    "Anaesthetic advice for NARP/MILS: prefer TIVA-free technique; if propofol required "
                    "for induction, keep duration minimal; maintain adequate glucose; avoid prolonged "
                    "propofol infusion; alternative anaesthetic agents: sevoflurane (volatile, Complex I "
                    "effects less pronounced); ketamine (NMDA antagonist, no mito coupling effect); "
                    "fentanyl/morphine (no mito toxicity). ICU: use midazolam or dexmedetomidine instead."
                ),
            },
            {
                "term": "CoQ10/Ubiquinol — Level C, Primary Supportive Therapy",
                "definition": (
                    "Coenzyme Q10 (CoQ10, ubiquinone) and its reduced form ubiquinol are the most "
                    "widely used supplements in mitochondrial disease including NARP/MILS. Mechanism: "
                    "CoQ10 is the mobile electron carrier between Complex I/II and Complex III in the "
                    "inner mitochondrial membrane; supplementation aims to saturate the CoQ10 pool "
                    "to maximize electron transfer efficiency and reduce superoxide generation. Evidence: "
                    "Level C (case series, no RCT in NARP specifically); doses 300-600 mg/day in NARP "
                    "(ubiquinol form preferred for bioavailability); paediatric dosing 10-20 mg/kg/day. "
                    "Safety: well-tolerated; GI upset at high doses; no organ toxicity. Generally "
                    "continued lifelong. Monitor CoQ10 plasma levels (target >2.5 μmol/L)."
                ),
            },
            {
                "term": "Idebenone — Level C, Investigational in NARP",
                "definition": (
                    "Idebenone (short-chain CoQ10 analogue) bypasses Complex I by accepting electrons "
                    "from NADH via NQO1 cytoplasmic reductase, then shuttling to Complex III, partially "
                    "restoring electron flow. In NARP/MILS, Complex V is the primary defect (not "
                    "Complex I as in LHON); idebenone's bypass of Complex I does NOT directly address "
                    "the Complex V defect — however, idebenone may reduce superoxide from upstream "
                    "electron backup and provide neuroprotection. Evidence: Level C — extrapolated from "
                    "LHON (RHODOS trial, Level B), no specific NARP RCT; used off-label at 900 mg/day "
                    "(same as LHON dosing) in NARP centres. Not approved for NARP (only LHON in EU). "
                    "May be considered if idebenone was beneficial in a maternal relative with LHON."
                ),
            },
            {
                "term": "Levetiracetam (LEV) — PREFERRED AED in NARP/MILS",
                "definition": (
                    "Levetiracetam is first-line for epilepsy in NARP/MILS: (1) Renal excretion as "
                    "an inactive hydrolysate — no hepatic CYP450 metabolism → no enzyme induction → "
                    "no drug-drug interactions with CoQ10/idebenone/thiamine; (2) no CoA sequestration "
                    "→ no interference with mitochondrial metabolism; (3) SV2A synaptic vesicle protein "
                    "modulator mechanism has no mito toxicity; (4) available as IV formulation for "
                    "acute seizure management in metabolic crisis. In MILS with infantile spasms: "
                    "ACTH (first-line for IS; standard VGB is CI due to RP risk) or LEV/CLB; "
                    "pyridoxine 100 mg IV trial always indicated before confirming non-pyridoxine-dependent "
                    "aetiology. Avoid PHT/CBZ (enzyme induction reduces CoQ10 levels). Avoid VPA (ABSOLUTE CI)."
                ),
            },
            {
                "term": "Thiamine (Vitamin B1) — Level C Supplement",
                "definition": (
                    "Thiamine pyrophosphate (TPP) is the essential cofactor for pyruvate dehydrogenase "
                    "(PDH), alpha-ketoglutarate dehydrogenase (αKGDH), and branched-chain keto acid "
                    "dehydrogenase (BCKDH). In NARP/MILS, the primary defect is Complex V; however, "
                    "the resulting energy deficit stresses the entire TCA cycle and OXPHOS. Thiamine "
                    "supplementation (100-300 mg/day) is used empirically to ensure maximal TPP "
                    "cofactor availability for upstream dehydrogenases — this is particularly important "
                    "during metabolic crises where pyruvate accumulation worsens lactic acidosis. "
                    "Evidence: Level C. No specific NARP RCT; standard of care in many mitochondrial "
                    "centres. Also: thiamine deficiency is a cause of Leigh syndrome (DDx workup) "
                    "→ empiric thiamine is diagnostic/therapeutic in Leigh-pattern disease while "
                    "awaiting genetics. Safe, inexpensive, widely used."
                ),
            },
        ],
        "gene_concepts": [
            {
                "term": "MT-ATP6 (ATP Synthase Subunit a / Subunit 6)",
                "definition": (
                    "MT-ATP6 encodes the 227-amino-acid subunit a of the F0 component of mitochondrial "
                    "ATP synthase (Complex V). Subunit a contains 5 transmembrane helices and forms "
                    "the essential proton half-channels at the a-c ring interface. The c-ring (10 "
                    "subunits) rotates as H+ ions translocate through two aqueous half-channels in "
                    "subunit a: one half-channel faces the intermembrane space (proton entry), one "
                    "faces the matrix (proton exit). Each complete 360° rotation of the c10-ring "
                    "drives synthesis of 3 ATP molecules in the F1 head. MT-ATP6 is transcribed from "
                    "the heavy strand of mtDNA; it overlaps with MT-ATP8 (subunit 8/A6L) by 46 bp "
                    "on the same strand. Both subunits form the stator-adjacent a-subunit platform "
                    "that anchors F0 to F1 and stabilises the rotor axis (subunit gamma-delta-epsilon)."
                ),
            },
            {
                "term": "Heteroplasmy and the NARP/MILS Threshold",
                "definition": (
                    "Heteroplasmy refers to the coexistence of mutant and wild-type mtDNA copies within "
                    "the same cell (or tissue). Cells contain hundreds to thousands of mtDNA copies "
                    "(polyplasmia). In MT-ATP6 disease, the proportion of mutant mtDNA copies determines "
                    "the phenotypic outcome: when mutant load exceeds the biochemical threshold "
                    "(estimated at 70% for m.8993T>G), the residual wild-type copies can no longer "
                    "compensate for the Complex V defect → clinical disease appears. At >90%, the "
                    "compensation is negligible → severe infantile Leigh syndrome (MILS). "
                    "This threshold concept is UNIQUE to heteroplasmic mtDNA diseases — nuclear "
                    "dominant diseases (OPA1/ADOA) do NOT have this relationship. Clinically: "
                    "blood heteroplasmy may differ from brain/retinal heteroplasmy (tissue-specific "
                    "mitotic segregation); a child can have higher heteroplasmy than the mother "
                    "(germline bottleneck amplification) → unexpected MILS from NARP mother."
                ),
            },
            {
                "term": "Maternal Inheritance vs Autosomal Inheritance (DDx)",
                "definition": (
                    "MT-ATP6 disease follows strict maternal inheritance: sperm mitochondria are "
                    "ubiquitin-tagged and autophagocytosed within hours of fertilisation → all "
                    "paternal mtDNA is eliminated → offspring inherit ONLY maternal mtDNA. Therefore: "
                    "(1) ALL children of a carrier MOTHER inherit the mutation (100% transmission); "
                    "(2) AFFECTED FATHERS do NOT transmit to offspring (paternal non-transmission "
                    "is the key clinical DDx from autosomal dominant OPA1/ADOA); (3) pedigree analysis "
                    "should show maternal lineage transmission; if the father is the only affected "
                    "parent → strongly suggests nuclear gene (OPA1, FRDA, etc.), not MT-ATP6. "
                    "CRITICAL: NARP/MILS = EQUAL sex penetrance (unlike LHON 80-90% male); "
                    "this is because the NARP threshold mechanism (heteroplasmy level) is sex-independent, "
                    "while LHON penetrance is modified by an X-linked locus."
                ),
            },
            {
                "term": "Complex V (F1F0-ATP Synthase) — Structure and Function",
                "definition": (
                    "ATP synthase is the final enzyme of oxidative phosphorylation. It converts the "
                    "electrochemical H+ gradient (proton-motive force, Δp = ΔΨm + ΔpH) into ATP. "
                    "The enzyme has two structural domains: F0 (transmembrane, proton-conducting: "
                    "subunits a, b2, c10) and F1 (soluble, catalytic: α3β3γδε hetero-octamer). "
                    "The c10-ring rotates as protons flow through subunit a; rotation of γ (inside α3β3) "
                    "sequentially changes conformational states of the three β subunits (Open → Loose → "
                    "Tight) driving ATP binding-and-release. 10 protons translocated → 3 ATP synthesised. "
                    "In NARP: MT-ATP6 p.Leu156Arg disrupts the proton-conducting geometry of TM-helix 4 "
                    "of subunit a → proton translocation impaired → rotor slows → F1 catalysis decoupled "
                    "from F0 → ATP synthesis rate falls 50-80% → severe ATP deficit in high-demand cells."
                ),
            },
        ],
        "disease_concepts": [
            {
                "term": "NARP Triad — Neuropathy + Ataxia + Retinitis Pigmentosa",
                "definition": (
                    "The NARP triad defines the classic phenotype at 70-90% m.8993T>G heteroplasmy: "
                    "(1) Neuropathy: axonal sensory > motor; reduced or absent deep tendon reflexes; "
                    "distal sensory loss; abnormal NCS/EMG; caused by dorsal root ganglion neuron "
                    "energy failure (DRG neurons are large, metabolically active, unmyelinated or "
                    "thinly myelinated → vulnerable to ATP deficit); (2) Ataxia: cerebellar > sensory; "
                    "truncal > appendicular; Purkinje cells are the most metabolically demanding neurons "
                    "per volume → selectively vulnerable to Complex V failure → gait instability, "
                    "dysarthria, intention tremor; (3) Retinitis Pigmentosa: rod-cone dystrophy → "
                    "night blindness (nyctalopia) is first symptom → progressive peripheral visual "
                    "field loss → tunnel vision → eventual severe visual impairment; fundoscopy: "
                    "bone-spicule pigmentation, waxy disc pallor, arteriolar attenuation; ERG: "
                    "reduced/absent rod and cone amplitudes. Not all three components are always present "
                    "simultaneously — partial phenotypes occur in lower heteroplasmy ranges."
                ),
            },
            {
                "term": "MILS — Maternally Inherited Leigh Syndrome (>90% heteroplasmy)",
                "definition": (
                    "MILS is the severe end of MT-ATP6 disease, typically presenting in neonates or "
                    "infants. Leigh syndrome is defined by the neuropathological/MRI pattern of bilateral "
                    "symmetric subacute necrotising encephalomyelopathy: lesions in the basal ganglia "
                    "(putamen > caudate), thalamus, brainstem (periaqueductal grey, inferior olivary "
                    "nuclei, dentate nuclei), and spinal cord. These regions have the highest metabolic "
                    "activity in the CNS and are selectively devastated by ATP synthesis failure. "
                    "Clinical MILS: hypotonia, developmental regression, feeding difficulties, "
                    "episodic decompensation triggered by febrile illness/fasting, lactic acidosis "
                    "(pH <7.2, high L:P ratio), respiratory failure. MT-ATP6 MILS is among the most "
                    "common causes of Leigh syndrome alongside Complex I (NDUFS1/NDUFV1/ND3/ND5), "
                    "PDH, PC, and Biotinidase deficiency. MILS prognosis: poor without intervention; "
                    "median survival <3 years historically; supportive care (glucose infusion, "
                    "bicarbonate, organ support) may extend survival; no approved disease-modifying therapy."
                ),
            },
            {
                "term": "Retinitis Pigmentosa in NARP — Management",
                "definition": (
                    "RP in NARP is a rod-cone dystrophic process driven by photoreceptor energy failure "
                    "(rods and cones are metabolically among the most active cells in the body). "
                    "Key management points: (1) Annual ophthalmology review including VA, VF (Goldmann "
                    "or automated perimetry), and ERG; (2) No proven treatment to halt NARP-RP "
                    "progression (unlike RPE65-LCA, which has Luxturna gene therapy); (3) Vitamin A "
                    "supplementation: NOT recommended in mitochondrial RP — unlike in some non-mitochondrial "
                    "RP where high-dose Vitamin A palmitate slows rod function loss, the potential "
                    "oxidative stress of supra-physiological Vitamin A in NARP (already oxidatively "
                    "stressed mitochondria) may worsen rather than help; (4) Anti-VEGF: NOT indicated — "
                    "NARP-RP is dystrophic (NOT neovascular/wet AMD); (5) Low vision aids: essential as "
                    "RP progresses; (6) AVOID: vigabatrin (ABSOLUTE CI), chloroquine/hydroxychloroquine "
                    "(macular toxicity additive), amiodarone (DION risk)."
                ),
            },
            {
                "term": "Heteroplasmy Bottleneck — Why Children Can Be More Severely Affected",
                "definition": (
                    "During oogenesis, the primary oocyte undergoes a mitochondrial genetic bottleneck: "
                    "mtDNA copy number is dramatically reduced (~200 copies at the primary oocyte stage) "
                    "before amplification to the mature oocyte (~100,000 copies). This bottleneck means "
                    "that random genetic drift can dramatically amplify or reduce the proportion of "
                    "mutant mtDNA in each oocyte — entirely by chance. Therefore: a mother with 60% "
                    "m.8993T>G heteroplasmy (asymptomatic carrier) can have a child with >90% "
                    "heteroplasmy (MILS), or a child with <30% (asymptomatic). This stochastic "
                    "amplification makes reproductive counselling for NARP families complex: "
                    "maternal heteroplasmy level is an IMPERFECT predictor of offspring risk. "
                    "Preimplantation genetic testing with heteroplasmy quantification (PGT-M) can "
                    "identify embryos with low mutant load for preferential transfer, significantly "
                    "reducing the risk of MILS offspring — this is the most effective preventive option."
                ),
            },
            {
                "term": "Leigh Syndrome — MRI Pattern and DDx",
                "definition": (
                    "Leigh syndrome (subacute necrotising encephalomyelopathy) has a characteristic "
                    "MRI pattern: bilateral symmetric T2-hyperintense (FLAIR-hyperintense) lesions "
                    "in the basal ganglia (especially putamen), thalamus, brainstem, and dentate "
                    "nuclei. DWI may show restricted diffusion acutely (cytotoxic oedema). "
                    "Leigh syndrome is caused by many different mitochondrial enzyme defects — the "
                    "MRI pattern is the 'final common pathway' of OXPHOS failure in high-metabolic-rate "
                    "brain regions. Genetic DDx of Leigh/Leigh-like MRI pattern includes: "
                    "MT-ATP6 MILS (Complex V), Complex I (NDUFS1/NDUFV1/ND3/ND5), PDH deficiency "
                    "(SLC19A3 Biotin-Thiamine-responsive), PC (pyruvate carboxylase), SURF1 "
                    "(Complex IV assembly), LRPPRC (Complex IV translation), Biotinidase deficiency "
                    "(treatable — supplement biotin), BTBGD (SLC19A3 — supplement biotin + thiamine). "
                    "Treatable causes must always be excluded first (biotin and thiamine empirically)."
                ),
            },
        ],
        "prescribing_safety": [
            {
                "term": "Ethambutol — CAUTION (not absolute CI in NARP, unlike LHON)",
                "definition": (
                    "Ethambutol (an optic nerve toxin) is ABSOLUTE CI in LHON (Complex I, RGC "
                    "degeneration). In NARP, the primary visual complication is retinitis pigmentosa "
                    "(photoreceptor dystrophy), not optic nerve disease. Ethambutol targets Complex IV "
                    "via copper chelation, damaging optic nerve axons — a different cell type from RP. "
                    "Therefore ethambutol is not an absolute CI in NARP the way it is in LHON; however, "
                    "CAUTION is still warranted: NARP patients may have subclinical optic nerve "
                    "involvement; any ethambutol use should be accompanied by frequent VA/VF/colour "
                    "vision monitoring and stopped immediately at first visual complaint. "
                    "RECOMMENDATION: use ethambutol-sparing TB regimens (RIF+INH+PZA) in NARP "
                    "wherever possible; if ethambutol is unavoidable, monthly ophthalmology review."
                ),
            },
            {
                "term": "Amiodarone — AVOID in NARP (DION risk amplified by RP)",
                "definition": (
                    "Amiodarone causes drug-induced optic neuropathy (DION) in 1-2% of the general "
                    "population. In NARP, the optic nerve is potentially compromised by energy deficit "
                    "(mitochondrial bioenergetic stress affects all high-demand axons). Amiodarone DION "
                    "is an additional optic toxicity ON TOP of existing RP and any subclinical optic "
                    "nerve involvement → risk of compounding visual loss is elevated. "
                    "If amiodarone is cardiology-mandatory: ophthalmology baseline + 3-monthly "
                    "VA/VF/OCT-RNFL monitoring; stop amiodarone immediately at first optic nerve sign. "
                    "Alternative antiarrhythmics: sotalol (no optic toxicity), flecainide, dronedarone "
                    "(shorter half-life, less ocular accumulation). Discuss with cardiology."
                ),
            },
            {
                "term": "Linezolid — ABSOLUTE CONTRAINDICATION (ALL mitochondrial disease)",
                "definition": (
                    "Linezolid inhibits mitochondrial 23S rRNA (the same 50S ribosomal target as in "
                    "bacteria — oxazolidinone mechanism). This blocks synthesis of all 13 mtDNA-encoded "
                    "OXPHOS subunits (ND1-6, CytB, COX1-3, ATP6/8), causing drug-induced optic "
                    "neuropathy (DION) and peripheral neuropathy. In NARP/MILS with pre-existing "
                    "Complex V deficiency, any additional OXPHOS compromise → acute energy crisis. "
                    "Linezolid DION is partly reversible if stopped within 2-4 weeks (optic nerve) "
                    "but may be irreversible with prolonged exposure. AVOID in ALL mitochondrial "
                    "disease patients. Alternatives for MRSA/VRE: daptomycin IV, vancomycin, "
                    "tedizolid (weaker mitoribosome effect — still CAUTION in mito disease), "
                    "co-trimoxazole, ceftaroline."
                ),
            },
            {
                "term": "Acute Metabolic Crisis Management (NARP/MILS decompensation)",
                "definition": (
                    "Acute decompensation in NARP/MILS (triggered by febrile illness, fasting, surgery, "
                    "or VPA inadvertently prescribed): (1) IV dextrose 10% at GIR 6-8 mg/kg/min — "
                    "glucose bypasses fatty acid oxidation, providing direct glycolytic ATP without "
                    "requiring Complex V efficiency; MOST IMPORTANT acute intervention; (2) IV sodium "
                    "bicarbonate — correct lactic acidosis (pH <7.1 or bicarbonate <15); (3) IV "
                    "thiamine 100-200 mg bolus — empiric (PDH cofactor, reduces pyruvate → lactate); "
                    "(4) AVOID fasting — maintain glucose infusion perioperatively; (5) AVOID propofol "
                    "infusion (PRIS); (6) Riboflavin 100 mg IV/oral — cofactor Complex II; "
                    "(7) L-Carnitine 50-100 mg/kg IV — replenish if depleted by mitochondrial stress; "
                    "(8) ICU monitoring — lactate, pH, glucose q2h; ECG continuous (cardiomyopathy risk); "
                    "(9) AVOID VPA if seizures develop — use LEV IV 40 mg/kg loading dose instead."
                ),
            },
        ],
    }
