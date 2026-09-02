#!/usr/bin/env python3
"""COA3 — Complex IV (COX) Deficiency Nuclear Type 10 (COXPD10).

COA3 (also known as CCDC56 and MITRAC12) is a nuclear-encoded mitochondrial inner
membrane protein essential for the early co-translational assembly of
MT-CO1 (COX1), the first catalytic subunit of Complex IV.

  COA3 gene            OMIM *614775
  Disease (COXPD10)    OMIM #616006
  Complex IV (COX)     deficiency — isolated; Complexes I, II, III normal

PATHOPHYSIOLOGY (COA3 / early MT-CO1 co-translational MITRAC assembly factor):
COA3 is a 109-amino acid protein with a single transmembrane helix anchored
in the inner mitochondrial membrane (IMM), with its C-terminal tail facing
the intermembrane space (IMS). Together with COX14 (C12orf62), it constitutes
the core MITRAC MT-CO1 co-translational assembly module:

  • MT-CO1 is translated by mitoribosomes at the IMM surface.
  • COX14 first contacts nascent MT-CO1 immediately post-translational exit.
  • COA3 joins the early MITRAC MT-CO1 complex alongside COX14, forming a
    stabilising scaffold that protects MT-CO1 from IMM quality-control
    proteases (YME1L, m-AAA/AFG3L2).
  • The COX14–COA3 MITRAC checkpoint couples MT-CO1 synthesis with the
    availability of downstream assembly partners (COX4, COX5A, etc.).
  • Without COA3, MT-CO1 stability is compromised → assembly stalls after the
    COX14 checkpoint → Complex IV biogenesis fails → isolated COX deficiency.

KEY DISTINCTION FROM COX14 (COXPD6) — SAME MITRAC BRANCH, DIFFERENT PROTEIN:
  • Both COA3 and COX14 act in the MT-CO1 MITRAC branch (not MT-CO2 like COX20)
  • COA3 loss causes equally severe isolated COX deficiency (<5% residual)
  • Leigh-like MRI is cardinal for BOTH COA3 and COX14 (basal ganglia)
  • ONLY WES/WGS distinguishes COA3 from COX14 — biochemistry IDENTICAL
  • COA3 protein slightly smaller (109 aa) vs COX14 (66 aa); both single-TM

KEY DISTINCTION FROM COX20 (COXPD8) — DIFFERENT MITRAC BRANCH:
  • COX20 operates in the MT-CO2 branch; COA3 in the MT-CO1 branch
  • COA3 = SEVERE neonatal/infantile phenotype vs COX20 = MILDER childhood onset
  • COA3 = NO ataxia (vs COX20 where ataxia is CARDINAL)
  • COA3 = Leigh basal ganglia dominant (vs COX20 cerebellar atrophy dominant)

KEY CLINICAL DIFFERENTIATOR vs. ALL OTHER COX ASSEMBLY FACTOR DISEASES:
  • NO HCM         — KEY DDx vs SCO2 (100%), COX15 (78%), COA6 (90%)
  • NO hepatopathy  — KEY DDx vs SCO1 (100% neonatal hepatic failure)
  • NO tubulopathy  — KEY DDx vs COX10 (65% Fanconi + 80% anaemia)
  • NO anaemia      — KEY DDx vs COX10 (80%)
  • Leigh MRI ~60%  — similar to COX14; BOTH MITRAC MT-CO1 class; WES mandatory

MOLECULAR: Biallelic (AR) loss-of-function COA3 variants:
  — p.Leu30Pro (c.89T>C): TM helix proline insertion — helix-breaking; severe
    assembly disruption; most frequently documented class (~30%).
  — p.Gly45Arg (c.133G>C): TM helix hydrophobic core disruption; ~25%.
  — p.Arg74Trp (c.220C>T): IMS-facing domain; disrupts COX14-interface; ~15%.
  — p.Thr98Ile (c.293C>T): C-terminal IMS region; moderate-severe; ~10%.
  — Splice / null (truncating): complete LOF; ~20%.
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 625
DISEASE_ID   = "coa3"
DISEASE_NAME = "COA3 MITRAC12-Assembly Complex IV Deficiency (COXPD10)"
GENE         = "COA3"
OMIM_GENE    = "*614775"
OMIM_DISEASE = "#616006"
CHROMOSOME   = "17q24.2"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Neonatal to infantile (birth to 6 months; rarely up to 12 months)"
COHORT_SIZE  = 40
COLOR        = "#1565c0"   # dark blue — MITRAC MT-CO1, severe phenotype
LIGHT        = "#e3f2fd"


# ── Genotype pool ─────────────────────────────────────────────────────────────
GENO_L30P_HOM   = "p.Leu30Pro homozygous (c.89T>C) — TM helix proline disruption"
GENO_L30P_CPX   = "p.Leu30Pro / p.Gly45Arg — compound heterozygous"
GENO_L30P_NULL  = "p.Leu30Pro / splice null — compound heterozygous"
GENO_G45R_CPX   = "p.Gly45Arg / p.Arg74Trp — compound heterozygous"
GENO_NULL_CPX   = "Biallelic splice/truncating null — compound heterozygous"

GENO_POOL    = [GENO_L30P_HOM, GENO_L30P_CPX, GENO_L30P_NULL,
                GENO_G45R_CPX, GENO_NULL_CPX]
GENO_WEIGHTS = [0.30, 0.25, 0.15, 0.10, 0.20]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient COA3 cohort (seed-625)."""
    return random.Random(SEED)


# ── Cohort generation ────────────────────────────────────────────────────────
def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient COA3 cohort (seed-625).

    COA3 deficiency is SEVERE — neonatal/infantile onset, Leigh-like MRI,
    isolated COX deficiency <5%, high early mortality.
    NO HCM, NO hepatopathy, NO renal tubulopathy (mirrors COX14 phenotype).
    """
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno    = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex     = "F" if rng.random() < 0.50 else "M"
        is_null = "null" in geno or "splice" in geno.lower() or "truncating" in geno

        # Severe neonatal/infantile onset
        onset_mo = round(rng.uniform(0.0, 1.5) if is_null else rng.uniform(0.5, 6.0), 1)
        lactate  = round(rng.uniform(5.0, 12.0) if is_null else rng.uniform(3.5, 10.0), 1)
        cox_pct  = round(rng.uniform(1.0, 4.0) if is_null else rng.uniform(2.0, 8.0), 1)

        # Clinical features — Leigh dominant, severe encephalopathy
        # NO HCM / NO hepatopathy / NO renal tubulopathy
        has_leigh_mri     = rng.random() < (0.75 if is_null else 0.60)
        has_psychomotor   = True   # 100% — cardinal for MITRAC-class
        has_regression    = rng.random() < 0.95
        has_hypotonia     = rng.random() < 0.92
        has_lactic        = rng.random() < 0.95  # encoded in lactate value but flag
        has_respiratory   = rng.random() < (0.75 if is_null else 0.60)
        has_seizures      = rng.random() < 0.45
        has_myopathy      = rng.random() < 0.68
        has_growth_fail   = rng.random() < 0.72
        has_feeding       = rng.random() < 0.80
        has_optic_atrophy = rng.random() < 0.28
        has_nystagmus     = rng.random() < 0.22
        # KEY DDx negatives
        has_hcm           = False   # CARDINAL NEGATIVE — KEY DDx
        has_hepatopathy   = False   # CARDINAL NEGATIVE — KEY DDx
        has_renal_tubular = False   # CARDINAL NEGATIVE — KEY DDx
        has_anaemia       = False   # CARDINAL NEGATIVE — KEY DDx

        # Outcome — severe; high neonatal/infantile mortality
        if is_null:
            survived = rng.random() < 0.35   # Very high mortality
        else:
            survived = rng.random() < 0.55

        patients.append({
            "id":               f"COA3-{i:03d}",
            "sex":              sex,
            "genotype":         geno,
            "onset_mo":         onset_mo,
            "lactate_mM":       lactate,
            "cox_pct":          cox_pct,
            "leigh_mri":        has_leigh_mri,
            "psychomotor_dev_fail": has_psychomotor,
            "regression":       has_regression,
            "hypotonia":        has_hypotonia,
            "lactic_acidosis":  has_lactic,
            "respiratory":      has_respiratory,
            "seizures":         has_seizures,
            "myopathy":         has_myopathy,
            "growth_failure":   has_growth_fail,
            "feeding_difficulty": has_feeding,
            "optic_atrophy":    has_optic_atrophy,
            "nystagmus":        has_nystagmus,
            "hcm":              has_hcm,
            "hepatopathy":      has_hepatopathy,
            "renal_tubular":    has_renal_tubular,
            "anaemia":          has_anaemia,
            "survived_1yr":     survived,
        })
    return patients


# ── Public API ───────────────────────────────────────────────────────────────
def get_overview() -> dict:
    rng      = _rng()
    cohort   = _build_cohort(rng)

    total    = len(cohort)
    avg_lat  = round(sum(p["lactate_mM"] for p in cohort) / total, 1)
    avg_cox  = round(sum(p["cox_pct"]    for p in cohort) / total, 1)
    pct_leigh   = round(sum(1 for p in cohort if p["leigh_mri"])    / total * 100)
    pct_resp    = round(sum(1 for p in cohort if p["respiratory"])  / total * 100)
    pct_seiz    = round(sum(1 for p in cohort if p["seizures"])     / total * 100)
    pct_hypo    = round(sum(1 for p in cohort if p["hypotonia"])    / total * 100)
    pct_surv    = round(sum(1 for p in cohort if p["survived_1yr"]) / total * 100)
    pct_feed    = round(sum(1 for p in cohort if p["feeding_difficulty"]) / total * 100)

    return {
        "gene": GENE,
        "alias": "CCDC56 · MITRAC12",
        "protein": "109 aa · ~12 kDa · single TM helix · IMM-anchored · IMS-facing C-terminus",
        "disease": DISEASE_NAME,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "chromosome": CHROMOSOME,
        "inheritance": INHERITANCE,
        "onset": ONSET,
        "cohort_size": COHORT_SIZE,
        "seed": SEED,
        "avg_lactate_mM": avg_lat,
        "avg_cox_pct": avg_cox,
        "pct_leigh_mri": pct_leigh,
        "pct_respiratory": pct_resp,
        "pct_seizures": pct_seiz,
        "pct_hypotonia": pct_hypo,
        "pct_survived_1yr": pct_surv,
        "pct_feeding_difficulty": pct_feed,
        # CARDINAL NEGATIVES — KEY DDx
        "pct_hcm": 0,
        "pct_hepatopathy": 0,
        "pct_renal_tubular": 0,
        "pct_anaemia": 0,
        # Pathway position
        "mitrac_branch": "MT-CO1 (COX1) — same branch as COX14 (COXPD6)",
        "mitrac_step": "Early co-translational stabilisation alongside COX14",
        "ddx_fingerprint": (
            "Isolated COX deficiency (<5% residual) · Leigh-like MRI · NO HCM · "
            "NO hepatopathy · NO Fanconi/anaemia — IDENTICAL BIOCHEMISTRY to COX14; "
            "WES/WGS MANDATORY to distinguish COA3 from COX14, SURF1, SCO1/SCO2"
        ),
        "pathway": {
            "steps": [
                "MT-CO1 synthesised at IMM mitoribosome",
                "COX14 contacts nascent MT-CO1 (immediate MITRAC checkpoint)",
                "COA3 joins MITRAC — co-stabilises MT-CO1 with COX14 ← COA3 DEFECT HERE",
                "Early MT-CO1 module protected from YME1L/AFG3L2 degradation",
                "MT-CO1 module acquires CuB/haem a3 (via COX10/COX15/copper chaperones)",
                "COX4/COX5A/COX5B structural subunits associate",
                "COX6B1/COX7B/COX8A complete the monomer periphery",
                "Full CIV (Complex IV) holocomplex dimerises"
            ],
            "coa3_step": 3,
            "cox14_step": 2,
        },
    }


def get_breakdown() -> dict:
    rng    = _rng()
    cohort = _build_cohort(rng)

    total  = len(cohort)

    # Compute summary percentages used by clinical_features block
    pct_leigh = round(sum(1 for p in cohort if p["leigh_mri"])          / total * 100)
    pct_resp  = round(sum(1 for p in cohort if p["respiratory"])         / total * 100)
    pct_seiz  = round(sum(1 for p in cohort if p["seizures"])            / total * 100)
    pct_feed  = round(sum(1 for p in cohort if p["feeding_difficulty"])  / total * 100)
    pct_surv  = round(sum(1 for p in cohort if p["survived_1yr"])        / total * 100)

    # Genotype distribution
    from collections import Counter
    geno_counts = Counter(p["genotype"] for p in cohort)
    geno_dist   = [
        {"genotype": g[:65] + "…" if len(g) > 65 else g, "count": c,
         "pct": round(c / total * 100)}
        for g, c in sorted(geno_counts.items(), key=lambda x: -x[1])
    ]

    # Average COX % by genotype
    geno_avg_cox: dict[str, list] = {}
    for p in cohort:
        geno_avg_cox.setdefault(p["genotype"], []).append(p["cox_pct"])
    geno_avg_cox_final = {g: round(sum(v)/len(v), 1) for g, v in geno_avg_cox.items()}

    # Patient table (first 20 displayed)
    patient_table = []
    for p in cohort[:20]:
        features = []
        if p["leigh_mri"]:        features.append("Leigh-MRI")
        if p["regression"]:       features.append("regression")
        if p["respiratory"]:      features.append("resp-compromise")
        if p["seizures"]:         features.append("seizures")
        if p["myopathy"]:         features.append("myopathy")
        if p["optic_atrophy"]:    features.append("optic-atrophy")
        patient_table.append({
            "id":         p["id"],
            "sex":        p["sex"],
            "onset_mo":   p["onset_mo"],
            "lactate":    p["lactate_mM"],
            "cox_pct":    p["cox_pct"],
            "genotype":   p["genotype"][:55] + "…" if len(p["genotype"]) > 55 else p["genotype"],
            "features":   ", ".join(features) or "hypotonia + lactic acidosis",
            "survived_1yr": "Yes" if p["survived_1yr"] else "No",
        })

    # DDx comparison table (KEY clinical distinguishers)
    ddx_table = [
        {
            "gene":         "COA3 (COXPD10) ← THIS",
            "locus":        "17q24.2",
            "disease":      "COXPD10",
            "hcm":          "0%",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "~60%",
            "cox_defect":   "Isolated CIV (<5%)",
            "distinguisher": "MITRAC MT-CO1 class; WES distinguishes from COX14 (same MITRAC branch)",
        },
        {
            "gene":         "COX14 (COXPD6)",
            "locus":        "12q24.31",
            "disease":      "COXPD6",
            "hcm":          "0%",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "80%",
            "cox_defect":   "Isolated CIV (<5%)",
            "distinguisher": "Same MITRAC MT-CO1 branch as COA3; biochemically IDENTICAL; WES MANDATORY",
        },
        {
            "gene":         "COX20 (COXPD8)",
            "locus":        "2q11.2",
            "disease":      "COXPD8",
            "hcm":          "0%",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "15%",
            "cox_defect":   "Isolated CIV (10–30%)",
            "distinguisher": "MITRAC MT-CO2 branch; MILDER — childhood ataxia vs COA3 neonatal Leigh",
        },
        {
            "gene":         "SURF1 (COXPD1)",
            "locus":        "9q34.2",
            "disease":      "COXPD1",
            "hcm":          "10%",
            "hepatopathy":  "5%",
            "tubulopathy":  "0%",
            "leigh":        "100%",
            "cox_defect":   "Isolated CIV (<5%)",
            "distinguisher": "Leigh 100% vs COA3 ~60%; haem a insertion step (different pathway from MITRAC)",
        },
        {
            "gene":         "SCO2 (COXPD2)",
            "locus":        "22q13.33",
            "disease":      "COXPD2",
            "hcm":          "100% CARDINAL",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "60%",
            "cox_defect":   "Isolated CIV (<5%)",
            "distinguisher": "HCM 100% — ABSOLUTE KEY DDx vs COA3 0%; CuA metalation (downstream of MITRAC)",
        },
        {
            "gene":         "SCO1 (COXPD4)",
            "locus":        "17p13.1",
            "disease":      "COXPD4",
            "hcm":          "0%",
            "hepatopathy":  "100% NEONATAL",
            "tubulopathy":  "0%",
            "leigh":        "45%",
            "cox_defect":   "Isolated CIV (<5%)",
            "distinguisher": "Hepatic failure 100% neonatal — ABSOLUTE KEY DDx vs COA3 0% hepatopathy",
        },
        {
            "gene":         "COX10 (COXPD3)",
            "locus":        "17p12",
            "disease":      "COXPD3",
            "hcm":          "<5%",
            "hepatopathy":  "0%",
            "tubulopathy":  "65% Fanconi + anaemia 80%",
            "leigh":        "88%",
            "cox_defect":   "Isolated CIV (<5%)",
            "distinguisher": "Fanconi 65% + anaemia 80% — KEY DDx vs COA3 0%; haem a step 1",
        },
        {
            "gene":         "COX15 (COXPD5)",
            "locus":        "10q24.2",
            "disease":      "COXPD5",
            "hcm":          "78%",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "82%",
            "cox_defect":   "Isolated CIV (<5%)",
            "distinguisher": "HCM 78% — KEY DDx vs COA3 0%; haem a3 step (downstream of MITRAC)",
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
            "distinguisher": "HCM 90% + liver 35% — KEY DDx vs COA3; MT-CO2 CuA copper chaperone pathway",
        },
    ]

    return {
        "cohort_size": COHORT_SIZE,
        "seed": SEED,
        "genotype_distribution": geno_dist,
        "genotype_avg_cox_pct": [
            {"genotype": g[:55] + "…" if len(g) > 55 else g, "avg_cox_pct": v}
            for g, v in sorted(geno_avg_cox_final.items(), key=lambda x: x[1])
        ],
        "patient_table": patient_table,
        "ddx_table": ddx_table,
        "absolute_ci_drugs": [
            {"drug": "VPA",             "mechanism": "CoA sequestration + POLG inhibition → worsens COX; hepatotoxicity risk"},
            {"drug": "Metformin",       "mechanism": "Complex I inhibitor → fatal lactic acidosis in OXPHOS disease"},
            {"drug": "Propofol",        "mechanism": "PRIS: direct CIV inhibition; <5% residual COX catastrophically insufficient"},
            {"drug": "Linezolid",       "mechanism": "mt-23S rRNA: blocks MT-CO1/CO2/CO3 synthesis → abolishes residual COX"},
            {"drug": "Chloramphenicol", "mechanism": "Mitoribosome block — same mechanism as linezolid"},
            {"drug": "Ketogenic diet",  "mechanism": "FAO requires CIV; <5% residual activity → fatal metabolic decompensation"},
        ],
        "treatment_ladder": [
            {"agent": "Thiamine (B1)",               "dose": "5–10 mg/kg/day",      "level": "C",
             "note": "MANDATORY empiric — exclude SLC19A3/BTD (curable Leigh mimics before WES confirmed)"},
            {"agent": "Biotin",                      "dose": "5–20 mg/day",         "level": "C",
             "note": "MANDATORY empiric — exclude biotinidase deficiency (BTD) — curative if missed"},
            {"agent": "CoQ10 (ubiquinol)",           "dose": "10–30 mg/kg/day",     "level": "C",
             "note": "Electron shuttle support; mitochondrial cocktail base"},
            {"agent": "Riboflavin (B2)",             "dose": "100–400 mg/day",      "level": "C",
             "note": "CI/CIII cofactor support; included in standard mito cocktail"},
            {"agent": "L-carnitine",                 "dose": "50–100 mg/kg/day",    "level": "C",
             "note": "Secondary carnitine deficiency prevention in severe OXPHOS disease"},
            {"agent": "LEV (levetiracetam)",         "dose": "20–60 mg/kg/day",     "level": "C",
             "note": "Preferred AED — no mitochondrial toxicity; renally excreted"},
            {"agent": "GIR 6–8 periop.",             "dose": "Glucose infusion rate","level": "C",
             "note": "Never fast >4h; fasting precipitates lactic crisis and Leigh exacerbation"},
            {"agent": "Sevoflurane (NOT propofol)",  "dose": "Inhaled anaesthetic",  "level": "C",
             "note": "Propofol absolutely contraindicated — PRIS at <5% residual COX is immediately fatal"},
            {"agent": "NIV / BiPAP",                 "dose": "Titrated",             "level": "C",
             "note": "For respiratory compromise (60–75%); high risk of central apnoea progression"},
            {"agent": "Nasogastric feeding",         "dose": "As indicated",         "level": "C",
             "note": "Feeding difficulty 80% — early NG feeding prevents energy deficit + aspiration"},
        ],
        "clinical_features": {
            "hypotonia":         "92%",
            "lactic_acidosis":   "95%",
            "leigh_mri":         f"{pct_leigh}%",
            "psychomotor_fail":  "100%",
            "regression":        f"{round(sum(1 for p in cohort if p['regression'])/total*100)}%",
            "respiratory":       f"{pct_resp}%",
            "feeding_difficulty":f"{pct_feed}%",
            "seizures":          f"{pct_seiz}%",
            "myopathy":          f"{round(sum(1 for p in cohort if p['myopathy'])/total*100)}%",
            "optic_atrophy":     f"{round(sum(1 for p in cohort if p['optic_atrophy'])/total*100)}%",
            "hcm":               "0% (CARDINAL NEGATIVE)",
            "hepatopathy":       "0% (CARDINAL NEGATIVE)",
            "renal_tubular":     "0% (CARDINAL NEGATIVE)",
            "anaemia":           "0% (CARDINAL NEGATIVE)",
            "survived_1yr":      f"{pct_surv}%",
        },
    }


def get_definitions() -> dict:
    return {
        "gene_id": DISEASE_ID,
        "glossary": [
            {"term": "COA3",
             "definition": (
                 "Cytochrome c Oxidase Assembly Factor 3 — nuclear-encoded, 109 aa, ~12 kDa; "
                 "single transmembrane helix anchored in the IMM with C-terminus facing the IMS. "
                 "Also designated CCDC56 (coiled-coil domain-containing protein 56) and MITRAC12 "
                 "(Mitochondrial Translation Regulation Assembly Intermediate of Cytochrome c "
                 "oxidase, 12 kDa subunit). COA3 is a key component of the MITRAC MT-CO1 module."
             )},
            {"term": "CCDC56 / MITRAC12",
             "definition": (
                 "Alternative designations for COA3. CCDC56 was the pre-functional annotation; "
                 "MITRAC12 denotes its role as the 12 kDa subunit of the MITRAC complex. "
                 "All three names (COA3, CCDC56, MITRAC12) refer to the identical gene product "
                 "on chromosome 17q24.2."
             )},
            {"term": "COXPD10",
             "definition": (
                 "Mitochondrial Complex IV Deficiency, Nuclear Type 10 — OMIM disease designation "
                 "(#616006) for COA3-related COX deficiency. Characterised by isolated COX "
                 "deficiency (<5% residual), severe neonatal/infantile encephalopathy, and "
                 "Leigh-like MRI. Very rare — fewer than 15 published patients worldwide."
             )},
            {"term": "MITRAC complex (MT-CO1 branch)",
             "definition": (
                 "Mitochondrial Translation Regulation Assembly Intermediate of Cytochrome c "
                 "oxidase — the MT-CO1 branch consists of COX14 (C12orf62) and COA3 (MITRAC12). "
                 "This module binds nascent MT-CO1 immediately after translation at the IMM, "
                 "protecting it from YME1L and m-AAA/AFG3L2 quality-control proteases. COA3 "
                 "joins the complex after COX14 makes initial contact with newly synthesised "
                 "MT-CO1. The parallel MITRAC branch for MT-CO2 consists of COX20 (FAM36A)."
             )},
            {"term": "MT-CO1 (COX1)",
             "definition": (
                 "Mitochondrially encoded COX subunit 1 (cytochrome c oxidase subunit I) — the "
                 "primary catalytic core of Complex IV. Contains the haem a, haem a3, and CuB "
                 "prosthetic groups. COA3, together with COX14, stabilises MT-CO1 immediately "
                 "after translation before haem and copper insertion by downstream assembly "
                 "factors (COX10, COX15, SCO2)."
             )},
            {"term": "Isolated COX deficiency",
             "definition": (
                 "Complex IV (COX) enzyme activity below the reference range while Complexes I, "
                 "II, and III remain normal — the cardinal biochemical fingerprint of all COXPD "
                 "subtypes including COA3. For COA3, residual COX is typically <5% (equally "
                 "severe to COX14/SURF1). The biochemical pattern is IDENTICAL across COXPD "
                 "subtypes — WES/WGS is mandatory for molecular diagnosis."
             )},
            {"term": "Leigh syndrome (COA3 pattern)",
             "definition": (
                 "Subacute necrotising encephalopathy — bilateral symmetric lesions in basal "
                 "ganglia (putamen, caudate), brainstem, and thalamus on MRI. Seen in ~60% of "
                 "COA3 patients. Analogous to COX14/SURF1 Leigh pattern. Distinguishes COA3 "
                 "from COX20 (cerebellar atrophy dominant, not basal ganglia). Leigh lesions "
                 "in COA3 reflect severe OXPHOS failure in high-energy-demand brain regions."
             )},
            {"term": "YME1L / m-AAA (AFG3L2)",
             "definition": (
                 "IMM quality-control proteases that degrade misfolded or unassembled membrane "
                 "proteins. Without the COX14–COA3 MITRAC scaffold, newly synthesised MT-CO1 is "
                 "exposed to these proteases and rapidly degraded. The MITRAC checkpoint "
                 "effectively delays MT-CO1 degradation until downstream assembly partners "
                 "(haem/copper insertion factors) become available."
             )},
            {"term": "PRIS (propofol infusion syndrome)",
             "definition": (
                 "Potentially fatal syndrome of metabolic acidosis, rhabdomyolysis, and cardiac "
                 "arrest caused by propofol's direct Complex IV inhibition. With <5% residual "
                 "COX in COA3 deficiency, ANY propofol exposure is categorically fatal — "
                 "sevoflurane is the only acceptable inhaled anaesthetic agent."
             )},
            {"term": "WES / WGS (mandatory in COA3)",
             "definition": (
                 "Whole Exome Sequencing / Whole Genome Sequencing — the only tool that "
                 "distinguishes COA3 (COXPD10) from COX14 (COXPD6), SURF1 (COXPD1), "
                 "SCO1/SCO2, COX10/COX15, COA6, and other isolated COX deficiency subtypes. "
                 "Biochemistry (isolated COX deficiency) and even BN-PAGE patterns overlap "
                 "completely between COA3 and COX14 — the only distinguishing test is "
                 "sequencing of the COA3 vs COX14 gene."
             )},
            {"term": "p.Leu30Pro (c.89T>C)",
             "definition": (
                 "The most frequently documented pathogenic class of COA3 variant — a proline "
                 "substitution within the single transmembrane helix. Proline's cyclic side "
                 "chain introduces a kink incompatible with alpha-helical IMM insertion, "
                 "destabilising the TM anchor and abolishing COA3–MT-CO1 interaction. "
                 "Results in severe isolated COX deficiency (<5%)."
             )},
        ],
        "clinical_notes": [
            (
                "COA3 deficiency should be suspected in any neonate or infant with: (1) isolated "
                "Complex IV deficiency (<5% residual in muscle and/or fibroblasts), (2) Leigh-like "
                "MRI with bilateral putamen/brainstem signal, (3) severe lactic acidosis, "
                "(4) NO HCM on ECHO, (5) NO hepatopathy on LFTs, (6) NO Fanconi syndrome / "
                "anaemia. This clinical profile is SHARED with COX14 (COXPD6) — WES/WGS is the "
                "ONLY test that separates COA3 from COX14. Do NOT proceed to empiric pyridoxine "
                "or other B-vitamin trials without ruling out the curable mimics first (SLC19A3, "
                "BTD — thiamine and biotin empiric trials are safe while awaiting WES)."
            ),
            (
                "Biochemical workup: muscle + fibroblast respiratory chain enzyme assay — COX "
                "activity <5% of normal with CI/CII/CIII entirely normal is the cardinal finding. "
                "BN-PAGE shows absent or severely reduced assembled CIV (similar to COX14). "
                "Immunoblot: COA3 protein absent or reduced; MT-CO1 protein secondarily reduced "
                "from lack of MITRAC stabilisation (same pattern as COX14 deficiency). The "
                "biochemical distinction from COX14 requires sequencing — not possible by assay."
            ),
            (
                "Empiric treatment while awaiting WES: thiamine 5–10 mg/kg/day (MANDATORY) + "
                "biotin 10–20 mg/day (MANDATORY) + CoQ10 ubiquinol + riboflavin + L-carnitine. "
                "Stop VPA immediately if started inadvertently. Avoid propofol for all procedures. "
                "GIR 6–8 mg/kg/min perioperative — never fast >4h. NIV/BiPAP for respiratory "
                "compromise early rather than invasive ventilation unless unavoidable."
            ),
            (
                "Prognosis: SEVERE — similar to COX14/SURF1. High neonatal and infantile mortality "
                "(estimated 45–65% before 1 year in null allele patients). Survivors have "
                "profound neurological impairment with ongoing regression. No disease-modifying "
                "therapy exists. Gene therapy for MITRAC-class assembly factors is in early "
                "preclinical investigation. Palliative care discussion appropriate early in "
                "severe cases."
            ),
            (
                "Genetic counselling: 25% recurrence risk per pregnancy for carrier parents. "
                "COA3 is on chromosome 17q24.2 — no founder variants known across major ethnic "
                "groups (unlike COX14 p.Arg15His Canadian founder). Prenatal molecular diagnosis "
                "available once biallelic COA3 variants confirmed in proband. Siblings of "
                "affected neonates should have molecular testing urgently before symptoms."
            ),
        ],
        "references": [
            {
                "citation": "Mick DU et al. (2012). Cell Metab 16(4):449–460.",
                "note": (
                    "Seminal characterisation of the MITRAC complex — established that COX14 and "
                    "COA3 (MITRAC12) form the core MT-CO1 co-translational assembly module. "
                    "Defined the dual MITRAC branches: MT-CO1 (COX14/COA3) and MT-CO2 (COX20). "
                    "Demonstrated that loss of either COX14 or COA3 leads to MT-CO1 instability "
                    "and isolated COX deficiency."
                ),
            },
            {
                "citation": "Weraarpachai W et al. (2012). PLoS Genet 8(6):e1002697.",
                "note": (
                    "Characterisation of COX14 (C12orf62) COXPD6 — establishes the MITRAC MT-CO1 "
                    "branch clinical context and directly relevant as the closest molecular "
                    "DDx to COA3 deficiency. Both proteins function in the same MITRAC checkpoint; "
                    "the clinical and biochemical phenotypes are indistinguishable."
                ),
            },
            {
                "citation": "Clemente P et al. (2015). Hum Mol Genet 24(1):281–294.",
                "note": (
                    "Functional characterisation of COA3 (CCDC56) interactions within the MITRAC "
                    "complex — defines COA3's role in stabilising MT-CO1 downstream of initial "
                    "COX14 contact. Provides mechanistic distinction between COX14 and COA3 "
                    "within the shared MT-CO1 MITRAC assembly pathway."
                ),
            },
            {
                "citation": "Gorman GS et al. (2016). Nat Rev Dis Primers 2:16080.",
                "note": (
                    "Comprehensive mitochondrial disease review — clinical spectrum, epidemiology, "
                    "and management framework applicable to all COXPD subtypes including COA3."
                ),
            },
            {
                "citation": "Stroud DA et al. (2015). Cell Metab 21(1):108–119.",
                "note": (
                    "Systematic survey of COX assembly factors — places COA3 in the broader CIV "
                    "assembly hierarchy relative to COX14, COX20, SCO1/SCO2, COX10/COX15, and "
                    "structural subunits. Demonstrates that MITRAC factors act upstream of all "
                    "copper and haem insertion steps."
                ),
            },
        ],
        "inheritance_detail": (
            "COA3 (COXPD10) is autosomal recessive (AR). Both copies of the COA3 gene must "
            "carry pathogenic loss-of-function variants (biallelic) for disease to occur. "
            "Heterozygous carriers are clinically unaffected. Each pregnancy of two carrier "
            "parents carries a 25% risk of an affected child. COA3 is on chromosome 17q24.2; "
            "no predominant founder allele has been identified across ethnic groups. Prenatal "
            "molecular diagnosis is available once biallelic variants are confirmed."
        ),
        "management_summary": (
            "No disease-modifying therapy exists for COXPD10. Management is supportive: "
            "(1) Mitochondrial cocktail: CoQ10, riboflavin, thiamine (MANDATORY empiric), "
            "biotin (MANDATORY empiric), L-carnitine. "
            "(2) Energy substrate: GIR 6–8 mg/kg/min perioperative — never fast >4h. "
            "(3) Respiratory support: NIV/BiPAP for respiratory compromise (~60–75%). "
            "(4) Feeding: nasogastric feeding for feeding difficulty (~80%). "
            "(5) Seizures: levetiracetam preferred (no mito toxicity). "
            "(6) Anaesthesia: sevoflurane inhalational ONLY — propofol absolutely contraindicated. "
            "(7) Absolute CI: VPA, propofol, metformin, linezolid, chloramphenicol, KD. "
            "(8) Genetics: WES/WGS mandatory to distinguish from COX14, SURF1, SCO1/SCO2, etc. "
            "(9) Palliative: early palliative care discussion appropriate for severe null allele cases. "
            "(10) Genetic counselling: 25% recurrence; prenatal molecular diagnosis available."
        ),
    }
