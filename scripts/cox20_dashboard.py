#!/usr/bin/env python3
"""COX20 — Complex IV (COX) Deficiency Nuclear Type 8 (COXPD8).

COX20 (also known as FAM36A) is a nuclear-encoded mitochondrial inner
membrane protein essential for the early co-translational assembly of
MT-CO2 (COX2), the second catalytic subunit of Complex IV.

  COX20 gene           OMIM *614698
  Disease (COXPD8)     OMIM #614607
  Complex IV (COX)     deficiency — isolated; Complexes I, II, III normal

PATHOPHYSIOLOGY (COX20 / early MT-CO2 co-translational assembly factor):
COX20 is a 116-amino acid protein with two transmembrane helices anchored
in the inner mitochondrial membrane (IMM), with its C-terminal domain
facing the intermembrane space (IMS). It functions analogously to COX14
but specifically for MT-CO2 (COX2) — the second mtDNA-encoded core subunit:

  • COX20 binds to newly synthesised MT-CO2 immediately after its
    release from the mitoribosome at the IMM surface.
  • This stabilises the nascent MT-CO2 polypeptide, preventing its
    premature degradation by IMM quality-control proteases.
  • Stabilised MT-CO2 is then transferred to the SCO1/SCO2/COA6
    copper-metalation machinery for CuA site assembly.
  • Without COX20, MT-CO2 is rapidly degraded → MT-CO2 module fails
    → Complex IV assembly is blocked → isolated COX deficiency (~10–30%).

KEY DISTINCTION FROM COX14 (COXPD6):
  • COX20 targets MT-CO2 (COX2); COX14 targets MT-CO1 (COX1)
  • COX20 deficiency = MILDER phenotype (childhood onset; survival to
    adulthood possible) vs COX14 (neonatal/infantile, often fatal)
  • Residual COX activity is HIGHER (~10–30%) than COX14 (<5%)
  • ATAXIA is a cardinal feature of COX20 (NOT present in COX14)
  • Cerebellar atrophy on MRI (vs Leigh-like for COX14 ~80%)

KEY CLINICAL DIFFERENTIATOR vs. OTHER COX ASSEMBLY FACTOR DISEASES:
  • NO HCM             — KEY DDx vs SCO2 (100%), COX15 (78%), COA6 (90%)
  • NO hepatopathy     — KEY DDx vs SCO1 (100% neonatal hepatic failure)
  • NO tubulopathy     — KEY DDx vs COX10 (65% Fanconi syndrome)
  • NO anaemia         — KEY DDx vs COX10 (80%)
  • Progressive ataxia — KEY DDx vs COX14 (regression not ataxia) and SURF1
  • Cerebellar atrophy — distinguishes from Leigh-dominant COX14 / SURF1
  • Milder COX defect (~10–30%) — less severe than COX14 (<5%) or SURF1 (<5%)
  • Childhood / juvenile onset — distinguishes from neonatal COXPD subtypes

MOLECULAR: Biallelic (AR) loss-of-function COX20 variants:
  — p.Asp2Gly (c.5A>G): N-terminal matrix-facing start region; first
    described variant (van Rahden 2015); disrupts protein import/processing;
    most documented allele (~30%); moderate-severe LOF with 15–25% COX.
  — p.Tyr65Cys (c.194A>G): TM2 helix; disrupts hydrophobic core packing;
    IMM anchoring impaired; destabilises COX20 ~ MT-CO2 binding; ~25%.
  — p.Arg37Trp (c.109C>T): TM1 helix; structural disruption; ~15%.
  — p.Gly79Glu (c.236G>A): TM2 helix hydrophobic core; severe; ~10%.
  — Splice / null (truncating): severe; complete LOF; ~20%.
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 621
DISEASE_ID   = "cox20"
DISEASE_NAME = "COX20 MITRAC-Assembly Complex IV Deficiency (COXPD8)"
GENE         = "COX20"
OMIM_GENE    = "*614698"
OMIM_DISEASE = "#614607"
CHROMOSOME   = "2q11.2"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Childhood to juvenile (typical 2–12 years; range 1–20 years)"
COHORT_SIZE  = 40
COLOR        = "#283593"   # dark indigo — MITRAC MT-CO2, milder spectrum
LIGHT        = "#e8eaf6"


# ── Genotype pool ─────────────────────────────────────────────────────────────
GENO_ASP2GLY_HOM   = "p.Asp2Gly homozygous (c.5A>G) — N-terminal import/processing"
GENO_ASP2GLY_CPX   = "p.Asp2Gly / p.Tyr65Cys — compound heterozygous"
GENO_ASP2GLY_NULL  = "p.Asp2Gly / splice null — compound heterozygous"
GENO_TYR65CYS_CPX  = "p.Tyr65Cys / p.Arg37Trp — compound heterozygous"
GENO_NULL_CPX      = "Biallelic splice/truncating null — compound heterozygous"

GENO_POOL    = [GENO_ASP2GLY_HOM, GENO_ASP2GLY_CPX, GENO_ASP2GLY_NULL,
                GENO_TYR65CYS_CPX, GENO_NULL_CPX]
GENO_WEIGHTS = [0.30, 0.25, 0.15, 0.10, 0.20]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient COX20 cohort (seed-621)."""
    return random.Random(SEED)


# ── Cohort generation ────────────────────────────────────────────────────────
def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient COX20 cohort (seed-621).

    COX20 deficiency is MILDER than COX14 — childhood onset, ataxia cardinal,
    cerebellar atrophy, higher COX residual (10–30%), longer survival.
    NO HCM, NO hepatopathy, NO renal tubulopathy.
    """
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno     = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex      = "F" if rng.random() < 0.50 else "M"
        is_null  = "null" in geno or "splice" in geno.lower() or "truncating" in geno

        # Childhood onset (milder than COX14 neonatal)
        onset_yr = round(rng.uniform(1.0, 5.0) if is_null else rng.uniform(2.0, 12.0), 1)
        lactate  = round(rng.uniform(2.5, 6.0) if is_null else rng.uniform(1.5, 4.5), 1)
        cox_pct  = round(rng.uniform(8.0, 18.0) if is_null else rng.uniform(10.0, 30.0), 1)

        # Clinical features — ataxia cardinal; cerebellar atrophy common
        # NO HCM / hepatopathy / tubulopathy — key DDx features
        has_ataxia        = True  # Cardinal feature of COX20 deficiency
        has_dysarthria    = rng.random() < 0.80
        has_intellectual  = rng.random() < 0.70
        has_spasticity    = rng.random() < 0.55
        has_cerebellar_mri = rng.random() < (0.90 if is_null else 0.75)
        has_leigh_mri     = rng.random() < (0.30 if is_null else 0.15)  # Rare; NOT cardinal
        has_hypotonia     = rng.random() < 0.60
        has_seizures      = rng.random() < 0.35
        has_regression    = rng.random() < 0.50  # Slower; not cardinal like COX14
        has_growth_fail   = rng.random() < 0.40
        has_myopathy      = rng.random() < 0.45
        has_neuropathy    = rng.random() < 0.30

        # Outcome — much better than COX14
        if is_null:
            survived = rng.random() < 0.70
        else:
            survived = rng.random() < 0.90

        patients.append({
            "id":               f"COX20-{i:03d}",
            "sex":              sex,
            "genotype":         geno,
            "onset_yr":         onset_yr,
            "lactate_mM":       lactate,
            "cox_pct":          cox_pct,
            "ataxia":           has_ataxia,
            "dysarthria":       has_dysarthria,
            "intellectual_dis": has_intellectual,
            "spasticity":       has_spasticity,
            "cerebellar_mri":   has_cerebellar_mri,
            "leigh_mri":        has_leigh_mri,
            "hypotonia":        has_hypotonia,
            "seizures":         has_seizures,
            "regression":       has_regression,
            "growth_failure":   has_growth_fail,
            "myopathy":         has_myopathy,
            "neuropathy":       has_neuropathy,
            "hcm":              False,   # NEVER — KEY DDx
            "hepatopathy":      False,   # NEVER — KEY DDx
            "tubulopathy":      False,   # NEVER — KEY DDx
            "survived_5yr":     survived,
        })
    return patients


# ── Feature extraction ────────────────────────────────────────────────────────
def _cohort_features(cohort: list[dict]) -> dict:
    n = len(cohort)
    pct = lambda key: round(sum(1 for p in cohort if p[key]) / n * 100)
    avg = lambda key: round(sum(p[key] for p in cohort) / n, 1)

    return {
        "total":              n,
        "pct_female":         round(sum(1 for p in cohort if p["sex"] == "F") / n * 100),
        "avg_onset_yr":       avg("onset_yr"),
        "avg_lactate":        avg("lactate_mM"),
        "avg_cox_pct":        avg("cox_pct"),
        "pct_ataxia":         100,   # Cardinal feature
        "pct_dysarthria":     pct("dysarthria"),
        "pct_intellectual":   pct("intellectual_dis"),
        "pct_spasticity":     pct("spasticity"),
        "pct_cerebellar_mri": pct("cerebellar_mri"),
        "pct_leigh_mri":      pct("leigh_mri"),
        "pct_hypotonia":      pct("hypotonia"),
        "pct_seizures":       pct("seizures"),
        "pct_regression":     pct("regression"),
        "pct_growth_fail":    pct("growth_failure"),
        "pct_myopathy":       pct("myopathy"),
        "pct_neuropathy":     pct("neuropathy"),
        "pct_hcm":            0,   # KEY DDx — always 0
        "pct_hepatopathy":    0,   # KEY DDx — always 0
        "pct_tubulopathy":    0,   # KEY DDx — always 0
        "pct_survived_5yr":   pct("survived_5yr"),
        "pct_deceased_5yr":   100 - pct("survived_5yr"),
    }


# ── Public API functions ──────────────────────────────────────────────────────
def get_overview() -> dict:
    rng    = _rng()
    cohort = _build_cohort(rng)
    feat   = _cohort_features(cohort)

    return {
        # Gene / disease identity
        "gene":         GENE,
        "alias":        "FAM36A",
        "protein":      "COX20 — 116 aa, ~13.6 kDa, two TM helices, IMS-facing C-terminus",
        "disease":      DISEASE_NAME,
        "omim_gene":    OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "chromosome":   CHROMOSOME,
        "inheritance":  INHERITANCE,
        "onset":        ONSET,
        "cohort":       f"{COHORT_SIZE} patients (seed-{SEED})",

        # KPI cards
        "kpis": [
            {"label": "COXPD type",         "value": "COXPD8"},
            {"label": "Protein size",        "value": "116 aa / 13.6 kDa"},
            {"label": "Assembly role",       "value": "MITRAC early MT-CO2"},
            {"label": "COX activity",        "value": f"~10–30% (avg {feat['avg_cox_pct']}%)"},
            {"label": "Onset (avg)",         "value": f"Avg {feat['avg_onset_yr']} yr"},
            {"label": "Lactate (avg)",       "value": f"{feat['avg_lactate']} mM"},
            {"label": "Ataxia",              "value": "100% ★ CARDINAL"},
            {"label": "Cerebellar MRI",      "value": f"{feat['pct_cerebellar_mri']}%"},
            {"label": "Leigh MRI",           "value": f"{feat['pct_leigh_mri']}% (rare)"},
            {"label": "HCM",                 "value": "0% ⚑ KEY DDx"},
            {"label": "Hepatopathy",         "value": "0% ⚑ KEY DDx"},
            {"label": "Tubulopathy",         "value": "0% ⚑ KEY DDx"},
        ],

        # Summary paragraph
        "summary": (
            f"COX20 (FAM36A) is a nuclear-encoded, 116-amino acid IMM-anchored protein "
            f"with two transmembrane helices that stabilises newly synthesised MT-CO2 "
            f"(COX2) immediately after mitoribosomal translation. Loss-of-function variants "
            f"(AR biallelic) cause Mitochondrial Complex IV Deficiency, Nuclear Type 8 "
            f"(COXPD8; OMIM #614607). Unlike the neonatal-lethal COX14 deficiency, COX20 "
            f"deficiency presents as a MILDER, childhood-onset progressive ataxia syndrome "
            f"with a residual COX activity of approximately 10–30% (vs <5% in COX14). "
            f"Cardinal features: progressive cerebellar ataxia (100%), dysarthria "
            f"({feat['pct_dysarthria']}%), intellectual disability ({feat['pct_intellectual']}%), "
            f"spasticity ({feat['pct_spasticity']}%), and cerebellar atrophy on MRI "
            f"({feat['pct_cerebellar_mri']}%). Key bedside differentiators: NO HCM "
            f"(unlike SCO2 100%, COX15 78%, COA6 90%), NO hepatopathy (unlike SCO1 100%), "
            f"NO renal tubulopathy (unlike COX10 65%). COX20 is mechanistically distinct "
            f"from SCO1/SCO2/COA6 (copper metalation) — it stabilises MT-CO2 upstream of "
            f"copper delivery. 5-year survival {feat['pct_survived_5yr']}% (much better "
            f"than COX14 ~35–45%). WES/WGS mandatory to distinguish from COX14, COA3, "
            f"COA6, SURF1 based on isolated COX deficiency alone."
        ),

        # Clinical feature bars
        "feature_bars": [
            {"label": "Ataxia (cardinal)",        "value": 100},
            {"label": "Dysarthria",                "value": feat["pct_dysarthria"]},
            {"label": "Intellectual disability",   "value": feat["pct_intellectual"]},
            {"label": "Spasticity",               "value": feat["pct_spasticity"]},
            {"label": "Cerebellar atrophy (MRI)",  "value": feat["pct_cerebellar_mri"]},
            {"label": "Hypotonia",                 "value": feat["pct_hypotonia"]},
            {"label": "Psychomotor regression",    "value": feat["pct_regression"]},
            {"label": "Myopathy",                  "value": feat["pct_myopathy"]},
            {"label": "Seizures",                  "value": feat["pct_seizures"]},
            {"label": "Peripheral neuropathy",     "value": feat["pct_neuropathy"]},
            {"label": "Leigh MRI (rare in COX20)", "value": feat["pct_leigh_mri"]},
            {"label": "HCM",                       "value": 0},
            {"label": "Hepatopathy",               "value": 0},
            {"label": "Renal tubulopathy",         "value": 0},
        ],

        # Absolute contraindications
        "absolute_ci": [
            {"drug": "VPA (valproate)",      "reason": "CoA sequestration + POLG inhibition — worsens COX depletion; hepatotoxicity risk elevated in COX disease"},
            {"drug": "Metformin",            "reason": "Complex I inhibition → uncontrolled lactic acidosis; fatal in OXPHOS disease regardless of residual COX level"},
            {"drug": "Propofol",             "reason": "PRIS — direct Complex IV inhibition; even 10–30% residual COX is insufficient buffer → cardiac/metabolic crisis"},
            {"drug": "Linezolid",            "reason": "mt-23S rRNA translation block → MT-CO2 synthesis abolished → residual COX eliminated; critically dangerous"},
            {"drug": "Chloramphenicol",      "reason": "mt-ribosome block (same mechanism as linezolid); abolishes MT-CO2 translation → COX collapse"},
            {"drug": "Ketogenic diet (KD)",  "reason": "Beta-oxidation obligatorily requires functional Complex IV; even partial COX deficiency risks fatal metabolic crisis under KD"},
        ],

        # Pathway context
        "pathway": {
            "name":   "MITRAC Early MT-CO2 Co-translational Assembly Pathway",
            "steps":  [
                "Mitoribosome synthesises MT-CO2 (COX2) polypeptide at IMM surface",
                "COX20 binds nascent MT-CO2 immediately post-translation — MITRAC checkpoint",
                "COX20 stabilises MT-CO2; prevents YME1L/AFG3L2 IMM protease degradation",
                "MT-CO2 maturation: CuA metalation by SCO1/SCO2 copper relay (COA6 assists)",
                "MT-CO2 module joins MT-CO1 module (COX14 pathway) at early CIV assembly",
                "Structural subunits (COX4, COX5B, COX6A, COX6B1, COX7A, COX7B) added",
                "MT-CO3 module and remaining nuclear-encoded subunits complete the core",
                "CIV dimerises; integrates into I+III+IV respiratory supercomplex",
            ],
            "cox20_step": 2,
            "footnote": (
                "COX20 acts at Step 2 — the earliest MT-CO2 quality-control checkpoint, "
                "analogous to COX14 for MT-CO1. Both are MITRAC-class factors on different targets."
            ),
        },

        # Rapid diagnosis checklist
        "diagnosis_checklist": [
            "Isolated COX deficiency 10–30% (muscle/fibroblasts) — CI/CII/CIII normal",
            "Progressive cerebellar ataxia (100%) + dysarthria in childhood/juvenile onset",
            "Cerebellar atrophy on MRI — NOT Leigh-like basal ganglia lesions (distinguishes COX20 from COX14/SURF1)",
            "Mild-to-moderate lactic acidosis (typically lactate <5 mM — milder than COX14)",
            "Intellectual disability + lower limb spasticity (common co-features)",
            "NO HCM on ECHO — rules out SCO2 (100%), COX15 (78%), COA6 (90%) as diagnosis",
            "NO hepatopathy / elevated LFTs — rules out SCO1 (100% hepatic failure) as diagnosis",
            "NO renal Fanconi / aminoaciduria — rules out COX10 (65% tubulopathy) as diagnosis",
            "WES/WGS + COX20 (FAM36A) gene: biallelic variants → confirm COXPD8",
            "CRITICAL DDx vs COX14 (COXPD6): childhood + ataxia + cerebellar = COX20 vs neonatal + Leigh + regression = COX14",
        ],
    }


def get_breakdown() -> dict:
    rng    = _rng()
    cohort = _build_cohort(rng)

    # Genotype distribution
    geno_counts: dict[str, int] = {}
    for p in cohort:
        geno_counts[p["genotype"]] = geno_counts.get(p["genotype"], 0) + 1

    geno_dist = sorted(
        [{"genotype": g, "count": c, "pct": round(c / len(cohort) * 100)}
         for g, c in geno_counts.items()],
        key=lambda x: -x["count"],
    )

    # COX % by genotype
    geno_cox: dict[str, list] = {}
    for p in cohort:
        geno_cox.setdefault(p["genotype"], []).append(p["cox_pct"])
    geno_avg_cox = {g: round(sum(v) / len(v), 1) for g, v in geno_cox.items()}

    # Per-patient table (first 20 for display)
    patient_table = [
        {
            "id":           p["id"],
            "sex":          p["sex"],
            "genotype":     p["genotype"][:55] + "…" if len(p["genotype"]) > 55 else p["genotype"],
            "onset_yr":     p["onset_yr"],
            "lactate":      p["lactate_mM"],
            "cox_pct":      p["cox_pct"],
            "ataxia":       "Yes ★" if p["ataxia"]          else "No",
            "cerebellar":   "Yes" if p["cerebellar_mri"]     else "No",
            "dysarthria":   "Yes" if p["dysarthria"]         else "No",
            "id_dis":       "Yes" if p["intellectual_dis"]   else "No",
            "hcm":          "No ⚑",
            "hepatopathy":  "No ⚑",
            "tubulopathy":  "No ⚑",
            "survived":     "Yes" if p["survived_5yr"]       else "No",
        }
        for p in cohort[:20]
    ]

    # DDx comparison table
    ddx_table = [
        {
            "gene":         "COX20 (this disease)",
            "locus":        "2q11.2",
            "disease":      "COXPD8",
            "hcm":          "0% (KEY DDx)",
            "hepatopathy":  "0% (KEY DDx)",
            "tubulopathy":  "0% (KEY DDx)",
            "leigh":        "~15% (rare)",
            "cox_defect":   "Isolated CIV (10–30%) — milder residual",
            "distinguisher":"Ataxia CARDINAL; childhood onset; cerebellar atrophy; MITRAC MT-CO2 assembly",
        },
        {
            "gene":         "COX14 (COXPD6)",
            "locus":        "12q24.31",
            "disease":      "COXPD6",
            "hcm":          "0%",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "~80% CARDINAL",
            "cox_defect":   "Isolated CIV (<5%) — lowest residual",
            "distinguisher":"Neonatal/infantile; Leigh MRI 80%; NO ataxia; MITRAC MT-CO1; COX <5%; high mortality Y1",
        },
        {
            "gene":         "SURF1 (COXPD1)",
            "locus":        "9q34.2",
            "disease":      "COXPD1",
            "hcm":          "~10%",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "95% CARDINAL",
            "cox_defect":   "Isolated CIV (<5%)",
            "distinguisher":"Most common Leigh COX cause; European c.845_846delCT founder; NO ataxia as cardinal feature",
        },
        {
            "gene":         "SCO2 (COXPD2)",
            "locus":        "22q13.33",
            "disease":      "COXPD2",
            "hcm":          "100% CARDINAL",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "55%",
            "cox_defect":   "Isolated CIV (<10%)",
            "distinguisher":"HCM 100% — KEY DDx vs COX20 0% HCM; copper chaperone for MT-CO2 CuA (downstream of COX20)",
        },
        {
            "gene":         "SCO1 (COXPD3)",
            "locus":        "17p13.2",
            "disease":      "COXPD3",
            "hcm":          "<5%",
            "hepatopathy":  "100% CARDINAL",
            "tubulopathy":  "0%",
            "leigh":        "45%",
            "cox_defect":   "Isolated CIV (<10%)",
            "distinguisher":"Neonatal hepatic failure 100% — KEY DDx vs COX20 0% hepatopathy; copper pathway",
        },
        {
            "gene":         "COX10 (COXPD4)",
            "locus":        "17p12",
            "disease":      "COXPD4",
            "hcm":          "<5%",
            "hepatopathy":  "0%",
            "tubulopathy":  "65% Fanconi + anaemia 80%",
            "leigh":        "88%",
            "cox_defect":   "Isolated CIV (<10%)",
            "distinguisher":"Renal tubulopathy 65% + anaemia 80% — KEY DDx vs COX20 0%; haem a (step 1)",
        },
        {
            "gene":         "COX15 (COXPD5)",
            "locus":        "10q24.2",
            "disease":      "COXPD5",
            "hcm":          "78%",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "82%",
            "cox_defect":   "Isolated CIV (<10%)",
            "distinguisher":"HCM 78% — KEY DDx vs COX20 0% HCM; haem a3 step (different pathway from MT-CO2)",
        },
        {
            "gene":         "COA6 (COXPD14)",
            "locus":        "1q42.2",
            "disease":      "COXPD14",
            "hcm":          "90% CARDINAL",
            "hepatopathy":  "35%",
            "tubulopathy":  "0%",
            "leigh":        "30%",
            "cox_defect":   "Isolated CIV (<10%)",
            "distinguisher":"HCM 90% + liver 35% — KEY DDx vs COX20; twin-CX9C copper chaperone for MT-CO2 CuA (downstream of COX20)",
        },
        {
            "gene":         "COA3/MITRAC12 (COXPD10)",
            "locus":        "17q24.2",
            "disease":      "COXPD10",
            "hcm":          "0%",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "60%",
            "cox_defect":   "Isolated CIV (<10%)",
            "distinguisher":"MITRAC partner for MT-CO1 (NOT MT-CO2); Leigh-dominant; WES distinguishes from COX20",
        },
    ]

    return {
        "cohort_size": COHORT_SIZE,
        "seed": SEED,
        "genotype_distribution": geno_dist,
        "genotype_avg_cox_pct": [
            {"genotype": g[:50] + "…" if len(g) > 50 else g, "avg_cox_pct": v}
            for g, v in sorted(geno_avg_cox.items(), key=lambda x: x[1])
        ],
        "patient_table": patient_table,
        "ddx_table": ddx_table,
        "absolute_ci_drugs": [
            {"drug": "VPA",             "mechanism": "CoA sequestration + POLG inhibition → worsens COX; hepatotoxicity"},
            {"drug": "Metformin",       "mechanism": "Complex I inhibitor → lactic acidosis, fatal in OXPHOS disease"},
            {"drug": "Propofol",        "mechanism": "PRIS: direct CIV inhibition; even 10–30% residual COX insufficient → crisis"},
            {"drug": "Linezolid",       "mechanism": "mt-23S rRNA: blocks MT-CO2 synthesis → abolishes residual COX20-stabilised CIV"},
            {"drug": "Chloramphenicol", "mechanism": "mt-ribosome block (same mechanism as linezolid)"},
            {"drug": "Ketogenic diet",  "mechanism": "FAO requires CIV; even partial deficiency risks fatal metabolic decompensation"},
        ],
        "treatment_ladder": [
            {"agent": "CoQ10 (ubiquinol)",          "dose": "10–30 mg/kg/day",     "level": "C", "note": "Electron shuttle support; standard mito cocktail"},
            {"agent": "Riboflavin (B2)",             "dose": "100–400 mg/day",      "level": "C", "note": "CI/CIII cofactor support"},
            {"agent": "Thiamine (B1)",               "dose": "5–10 mg/kg/day",      "level": "C", "note": "MANDATORY empiric — exclude SLC19A3/BTD Leigh mimics"},
            {"agent": "Biotin",                      "dose": "5–20 mg/day",         "level": "C", "note": "MANDATORY empiric — exclude biotinidase deficiency"},
            {"agent": "L-carnitine",                 "dose": "50–100 mg/kg/day",    "level": "C", "note": "Secondary carnitine deficiency prevention"},
            {"agent": "Physiotherapy / OT",          "dose": "Specialist-guided",   "level": "B", "note": "Ataxia + spasticity rehabilitation — key supportive therapy"},
            {"agent": "Speech therapy",              "dose": "Specialist-guided",   "level": "B", "note": "Dysarthria management — often the most functional intervention"},
            {"agent": "LEV (levetiracetam)",         "dose": "20–60 mg/kg/day",     "level": "C", "note": "Preferred AED for seizures — no mito toxicity, renal excretion"},
            {"agent": "Baclofen",                    "dose": "Per neurology team",   "level": "C", "note": "Spasticity management in lower-limb predominant cases"},
            {"agent": "GIR 6–8 periop.",             "dose": "Glucose infusion rate","level": "C", "note": "Never fast >4h; prevent catabolism + lactate surge"},
            {"agent": "Sevoflurane (NOT propofol)",  "dose": "Inhaled anaesthetic",  "level": "C", "note": "PRIS risk with propofol even at 10–30% residual COX"},
        ],
    }


def get_definitions() -> dict:
    return {
        "gene_id": DISEASE_ID,
        "glossary": [
            {"term": "COX20",
             "definition": "Cytochrome c Oxidase Assembly Factor 20 — nuclear-encoded, 116 aa, ~13.6 kDa; two transmembrane helices anchored in the IMM with C-terminus facing the IMS; originally annotated FAM36A; the essential early co-translational stabilising factor for MT-CO2 (COX2). Analogous to COX14 for MT-CO2 as COX14 is for MT-CO1."},
            {"term": "FAM36A",
             "definition": "Original family-member-with-unknown-function designation for COX20 on chromosome 2q11.2; the two names (COX20 / FAM36A) refer to the identical gene. Renamed COX20 after its function in Complex IV assembly was elucidated by van Rahden et al. (2015)."},
            {"term": "COXPD8",
             "definition": "Mitochondrial Complex IV Deficiency, Nuclear Type 8 — OMIM disease designation (#614607) for COX20-related COX deficiency. Distinguished by milder phenotype, childhood onset, and cerebellar ataxia as the cardinal feature."},
            {"term": "MT-CO2 (COX2)",
             "definition": "Mitochondrial-encoded COX subunit 2 (cytochrome c oxidase subunit II) — contains the binuclear CuA copper centre that accepts electrons from cytochrome c. COX20 is required for MT-CO2 early co-translational stabilisation; SCO1/SCO2/COA6 subsequently provide CuA metalation."},
            {"term": "MITRAC complex (COX20 branch)",
             "definition": "Mitochondrial Translation Regulation Assembly Intermediate of Cytochrome c oxidase — the MT-CO2 branch of MITRAC. COX20 binds nascent MT-CO2 at the mitoribosome, preventing IMM protease degradation (YME1L, m-AAA/AFG3L2). This is the earliest quality-control checkpoint for MT-CO2 biogenesis, upstream of copper metalation."},
            {"term": "CuA metalation",
             "definition": "The insertion of a binuclear copper (CuA) centre into MT-CO2 — carried out sequentially by SCO1 and SCO2, with COA6 as a copper chaperone cooperator. COX20 stabilises MT-CO2 upstream of this step; without COX20, MT-CO2 is degraded before CuA metalation can occur."},
            {"term": "Cerebellar atrophy",
             "definition": "Progressive loss of cerebellar volume visible on MRI — the hallmark neuroimaging finding in COX20 deficiency (~75–80% of cases). DISTINGUISHES COX20 from COX14/SURF1 (Leigh-like basal ganglia lesions) and from COX10/COX15 (also Leigh-dominant). Cerebellar ataxia is the cardinal clinical correlate."},
            {"term": "Isolated COX deficiency",
             "definition": "Complex IV (COX) enzyme activity below the reference range while Complexes I, II, and III remain normal. For COX20, residual COX is typically 10–30% (milder than the <5% in COX14 or SURF1). The biochemical fingerprint is identical across COXPD subtypes — WES/WGS mandatory to distinguish them."},
            {"term": "Progressive ataxia",
             "definition": "The cardinal clinical feature of COX20 deficiency — progressive unsteady gait, limb incoordination, and cerebellar signs (nystagmus, intention tremor) — not seen as a cardinal feature in COX14, SURF1, SCO1, SCO2, or COX15. Onset typically 2–12 years."},
            {"term": "YME1L / m-AAA / AFG3L2",
             "definition": "IMM quality-control proteases. Without COX20 protection, MT-CO2 is exposed to these proteases and rapidly degraded after translation. COX20 shields MT-CO2 during this vulnerable early assembly phase."},
            {"term": "PRIS",
             "definition": "Propofol Infusion Syndrome — potentially fatal metabolic acidosis + rhabdomyolysis + cardiac arrest from propofol's direct Complex IV inhibition. Even 10–30% residual COX in COX20 deficiency is insufficient buffer; sevoflurane must be used instead."},
            {"term": "WES/WGS",
             "definition": "Whole Exome Sequencing / Whole Genome Sequencing — mandatory diagnostic tool for COX20 and all COXPD subtypes. Biochemistry alone (isolated COX deficiency) cannot distinguish COX20 from COX14, SURF1, SCO1/SCO2, COX10, COX15, COA6, or COA3."},
            {"term": "p.Asp2Gly (c.5A>G)",
             "definition": "The founding pathogenic variant in COX20 — an N-terminal substitution that disrupts protein import processing and mitochondrial targeting. First described by van Rahden et al. (2015) in the initial COXPD8 family; the most frequently reported allele (~30%)."},
        ],
        "clinical_notes": [
            "COX20 deficiency should be suspected in any child or adolescent with progressive "
            "cerebellar ataxia + isolated COX deficiency when: (a) NO HCM on ECHO, (b) NO "
            "hepatopathy on LFTs, (c) NO renal Fanconi on urine amino acids/phosphate. The "
            "childhood onset and ataxia-dominant phenotype distinguish COX20 from the "
            "neonatal-lethal COX14 deficiency despite both being MITRAC-class disorders. "
            "WES/WGS is mandatory to distinguish COX20 from the broader isolated COX deficiency DDx.",

            "Biochemical workup: muscle + fibroblast respiratory chain enzyme assay — COX "
            "activity ~10–30% of normal with CI/CII/CIII normal is the cardinal finding. "
            "BN-PAGE shows partial reduction of assembled CIV (not absent as in COX14). "
            "Immunoblot: COX20 protein absent or reduced; MT-CO2 protein secondarily reduced "
            "from lack of stabilisation. Distinguish from SCO1/SCO2 (normal MT-CO2 protein "
            "but impaired CuA metalation vs COX20 = MT-CO2 degradation).",

            "Empiric treatment while awaiting molecular result: thiamine 5–10 mg/kg/day "
            "MANDATORY (exclude SLC19A3/BTD — curable mimics of mito ataxia) + biotin MANDATORY "
            "(BTD) + CoQ10 + riboflavin. Stop VPA immediately if inadvertently started. "
            "Avoid propofol categorically for any procedure. Physiotherapy is a critical early "
            "intervention for ataxia and spasticity — do not delay.",

            "Physiotherapy protocol: early-initiated gait training (Frenkel exercises), "
            "proprioceptive training, balance rehabilitation, ankle-foot orthoses for foot drop "
            "if spasticity is prominent, speech therapy for dysarthria. Neuropsychological "
            "assessment for intellectual disability support services. Baclofen or tizanidine "
            "for lower limb spasticity (avoid intrathecal baclofen pump if patient is active).",

            "Prognosis: SIGNIFICANTLY BETTER than COX14/SURF1 deficiency. Most patients "
            "survive to adulthood with progressive disability but not typically with early "
            "mortality. Functional independence varies — intellectual disability and ataxia "
            "severity are the main determinants. No disease-modifying therapy exists; "
            "gene therapy approaches for MITRAC-class assembly factors are in early investigation.",

            "Genetic counselling: 25% recurrence risk per pregnancy for carrier parents. "
            "Prenatal diagnosis by molecular genetics available once biallelic COX20 variants "
            "confirmed in proband. Siblings of affected children should have COX enzyme assay "
            "and molecular testing. Healthy carriers have no clinical manifestations.",
        ],
        "references": [
            {
                "citation": "van Rahden VA et al. (2015). Am J Hum Genet 97(5):761–768.",
                "note":     "First description of COX20 (FAM36A) pathogenic variants causing Complex IV deficiency with progressive ataxia (COXPD8). Demonstrated COX20 stabilises MT-CO2 at the early MITRAC checkpoint; p.Asp2Gly founding variant.",
            },
            {
                "citation": "Bourens M & Barrientos A (2017). Hum Mol Genet 26(21):4147–4157.",
                "note":     "Mechanistic characterisation of COX20 function — established COX20 as the specific early MT-CO2 assembly chaperone and defined its interaction with SCO1/SCO2 copper metalation machinery.",
            },
            {
                "citation": "Mick DU et al. (2012). Cell Metab 16(4):449–460.",
                "note":     "Original MITRAC complex characterisation. Established the parallel MITRAC MT-CO1 (COX14) and MT-CO2 (COX20) co-translational assembly modules as mechanistically distinct MITRAC branches.",
            },
            {
                "citation": "Gorman GS et al. (2016). Nat Rev Dis Primers 2:16080.",
                "note":     "Comprehensive mitochondrial disease review — clinical spectrum, epidemiology, and management framework applicable to all COXPD subtypes including COX20.",
            },
            {
                "citation": "Stroud DA et al. (2015). Cell Metab 21(1):108–119.",
                "note":     "Systematic survey of COX assembly factors and their interactions — places COX20 in the broader CIV assembly pathway context relative to SCO1/SCO2, COA6, and downstream factors.",
            },
        ],
        "inheritance_detail": (
            "COX20 (COXPD8) is autosomal recessive (AR). Both copies of the COX20 gene must "
            "carry pathogenic loss-of-function variants (biallelic) for disease to occur. "
            "Heterozygous carriers are clinically unaffected with normal COX enzyme activity. "
            "Each pregnancy of two carrier parents carries a 25% risk of an affected child. "
            "Prenatal molecular diagnosis is available once biallelic variants are confirmed "
            "in the proband. COX20 variants have been reported in multiple ethnic backgrounds "
            "without a single predominant founder allele (unlike COX14 where p.Arg15His is "
            "a Canadian founder and COX6B1 where p.Trp38Ser is a Turkish founder)."
        ),
        "management_summary": (
            "No disease-modifying therapy exists for COXPD8. Management is supportive: "
            "(1) Mitochondrial cocktail: CoQ10, riboflavin, thiamine (MANDATORY empiric), "
            "biotin (MANDATORY empiric), L-carnitine. "
            "(2) Rehabilitation: physiotherapy (ataxia + spasticity), speech therapy (dysarthria), "
            "OT — CRITICAL for functional independence in this milder disease. "
            "(3) Spasticity: baclofen or tizanidine as needed. "
            "(4) Seizures: levetiracetam preferred (no mito toxicity). "
            "(5) Energy substrate: GIR 6–8 mg/kg/min perioperative; fasting >4h prohibited. "
            "(6) Anaesthesia: sevoflurane inhalational ONLY — propofol absolutely contraindicated. "
            "(7) Absolute CI: VPA, propofol, metformin, linezolid, chloramphenicol, KD. "
            "(8) Multidisciplinary: neurology + metabolics + physiotherapy + neuro-rehab mandatory. "
            "(9) Genetic counselling: 25% recurrence; prenatal diagnosis available."
        ),
    }
