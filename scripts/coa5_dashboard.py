#!/usr/bin/env python3
"""COA5 — Complex IV (COX) Deficiency Nuclear Type 11 (COXPD11).

COA5 (also designated C2orf64 and PET191 in yeast) is a nuclear-encoded
mitochondrial inner membrane protein required for the early co-translational
assembly of MT-CO1 (COX1), the primary catalytic core of Complex IV.

  COA5 gene            OMIM *614657
  Disease (COXPD11)    OMIM #614932
  Complex IV (COX)     deficiency — isolated; Complexes I, II, III normal

PATHOPHYSIOLOGY (COA5 / early MT-CO1 co-translational assembly factor):
COA5 is a 168-amino acid protein (mature ~142 aa after MTS cleavage) with a
single transmembrane helix anchored in the inner mitochondrial membrane (IMM),
C-terminal tail facing the intermembrane space (IMS). It acts in the MT-CO1
co-translational assembly pathway:

  • MT-CO1 is synthesised by mitoribosomes at the IMM surface.
  • COA5 engages nascent MT-CO1 during early assembly, functioning alongside
    COX14 and COA3 in stabilising the newly translated polypeptide.
  • Loss of COA5 destabilises early MT-CO1 assembly intermediates → COX
    biogenesis stalls → isolated COX deficiency (<15% residual).
  • COA5 deficiency results in a CARDIOMYOPATHY-DOMINANT phenotype —
    CARDINAL DIFFERENTIATOR from COX14/COA3 which present as Leigh-dominant
    without HCM.

KEY DISTINCTION FROM COX14 (COXPD6) AND COA3 (COXPD10):
  • COA5 shares the MT-CO1 assembly branch with COX14 and COA3
  • CRITICAL DIFFERENCE: COA5 = HCM dominant (~88%) vs COX14/COA3 = NO HCM
  • Leigh-like MRI less prominent in COA5 (~42%) vs COX14 (~80%) / COA3 (~60%)
  • COA5 is at 2q11.2 — SAME chromosome arm as COX20 (also 2q11.2), different gene

KEY DISTINCTION FROM COX20 (COXPD8):
  • COX20 = MT-CO2 branch; COA5 = MT-CO1 branch
  • COX20 = ataxia-dominant childhood-onset; COA5 = HCM-dominant neonatal
  • Both at 2q11.2 — adjacent loci, completely different assembly pathways

KEY DISTINCTION FROM SCO2/COA6 (also HCM-dominant):
  • SCO2/COA6 = copper metalation pathway (CuA delivery to MT-CO2)
  • COA5 = MT-CO1 co-translational stabilisation — entirely different pathway
  • COA5 + SCO2 + COA6 share HCM dominance but differ in copper vs assembly roles
  • SCO2 HCM 100%, COA6 HCM 90%, COA5 HCM ~88%, COX15 HCM 78%

KEY CLINICAL DIFFERENTIATOR vs. ALL OTHER COX ASSEMBLY FACTOR DISEASES:
  • HCM present     — KEY DDx vs COX14/COA3/COX20/COX6B1 (all NO HCM)
  • NO hepatopathy  — KEY DDx vs SCO1 (100% neonatal hepatic failure)
  • NO tubulopathy  — KEY DDx vs COX10 (65% Fanconi)
  • NO anaemia      — KEY DDx vs COX10 (80%)
  • Leigh MRI ~42%  — less prominent vs COX14 (80%) / SURF1 (95%); more like COA6 (30%)

MOLECULAR: Biallelic (AR) loss-of-function COA5 variants:
  — p.Trp59Ter (c.177G>A): Truncating nonsense — complete LOF; IMS domain absent;
    North-African / Moroccan founder class; most frequently documented; severe; ~35%.
  — p.Arg51Trp (c.151C>T): IMS-domain charge disruption; severe; ~20%.
  — p.Gly63Ala (c.188G>C): Core fold structural disruption; severe; ~18%.
  — p.Leu92Pro (c.275T>C): Helix-breaking proline; severe IMM anchoring failure; ~12%.
  — Biallelic splice / null (various): complete LOF; ~15%.
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 629
DISEASE_ID   = "coa5"
DISEASE_NAME = "COA5 Cardiomyopathic Complex IV Deficiency (COXPD11)"
GENE         = "COA5"
OMIM_GENE    = "*614657"
OMIM_DISEASE = "#614932"
CHROMOSOME   = "2q11.2"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Neonatal to early infantile (first 1–3 months)"
COHORT_SIZE  = 40
COLOR        = "#880e4f"   # deep crimson — HCM-dominant cardiac phenotype
LIGHT        = "#fce4ec"


# ── Genotype pool ─────────────────────────────────────────────────────────────
GENO_W59TER_HOM  = "p.Trp59Ter homozygous (c.177G>A) — IMS truncating, Moroccan/N-African"
GENO_W59TER_CPX  = "p.Trp59Ter / p.Arg51Trp — compound heterozygous"
GENO_W59TER_NULL = "p.Trp59Ter / splice null — compound heterozygous"
GENO_R51W_CPX    = "p.Arg51Trp / p.Gly63Ala — compound heterozygous"
GENO_NULL_CPX    = "Biallelic splice/truncating null — compound heterozygous"

GENO_POOL    = [GENO_W59TER_HOM, GENO_W59TER_CPX, GENO_W59TER_NULL,
                GENO_R51W_CPX,   GENO_NULL_CPX]
GENO_WEIGHTS = [35, 20, 18, 15, 12]   # %


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient COA5 cohort (seed-629)."""
    return random.Random(SEED)


# ── Cohort generation ────────────────────────────────────────────────────────
def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient COA5 cohort (seed-629).

    COA5 deficiency — HCM-dominant cardiomyopathy, neonatal/early infantile onset,
    isolated COX deficiency (<15% residual).
    HCM is the CARDINAL distinguishing feature (vs COX14/COA3/COX20 which have NO HCM).
    NO hepatopathy / NO renal tubulopathy / NO anaemia.
    """
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno    = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex     = "F" if rng.random() < 0.50 else "M"
        is_null = "null" in geno or "splice" in geno.lower() or "truncating" in geno

        # Neonatal / early infantile onset
        onset_mo = round(rng.uniform(0.0, 1.0) if is_null else rng.uniform(0.5, 3.0), 1)
        lactate  = round(rng.uniform(4.5, 10.0) if is_null else rng.uniform(3.0, 8.5), 1)
        cox_pct  = round(rng.uniform(2.0, 8.0) if is_null else rng.uniform(5.0, 15.0), 1)

        # HCM is cardinal (high rate ~88%)
        has_hcm           = rng.random() < (0.95 if is_null else 0.88)
        has_lactic        = rng.random() < 0.92
        has_hypotonia     = rng.random() < 0.72
        has_psychomotor   = rng.random() < 0.85
        has_regression    = rng.random() < 0.80
        has_enceph        = rng.random() < 0.60
        has_respiratory   = rng.random() < 0.52
        has_seizures      = rng.random() < 0.38
        has_myopathy      = rng.random() < 0.58
        has_growth_fail   = rng.random() < 0.65
        has_feeding       = rng.random() < 0.70
        has_leigh_mri     = rng.random() < (0.50 if is_null else 0.42)
        # KEY DDx negatives
        has_hepatopathy   = False   # CARDINAL NEGATIVE — KEY DDx vs SCO1
        has_renal_tubular = False   # CARDINAL NEGATIVE — KEY DDx vs COX10
        has_anaemia       = False   # CARDINAL NEGATIVE — KEY DDx vs COX10
        has_ataxia        = False   # KEY DDx vs COX20 (ataxia 100%)

        # Outcome — severe cardiac mortality
        if is_null:
            survived = rng.random() < 0.30   # Very high mortality (cardiac failure)
        else:
            survived = rng.random() < 0.50

        patients.append({
            "id":               f"COA5-{i:03d}",
            "sex":              sex,
            "genotype":         geno,
            "onset_mo":         onset_mo,
            "lactate_mM":       lactate,
            "cox_pct":          cox_pct,
            "hcm":              has_hcm,
            "lactic_acidosis":  has_lactic,
            "hypotonia":        has_hypotonia,
            "psychomotor":      has_psychomotor,
            "regression":       has_regression,
            "encephalopathy":   has_enceph,
            "respiratory":      has_respiratory,
            "seizures":         has_seizures,
            "myopathy":         has_myopathy,
            "growth_failure":   has_growth_fail,
            "feeding_difficulty": has_feeding,
            "leigh_mri":        has_leigh_mri,
            "hepatopathy":      has_hepatopathy,
            "renal_tubular":    has_renal_tubular,
            "anaemia":          has_anaemia,
            "ataxia":           has_ataxia,
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
    pct_hcm     = round(sum(1 for p in cohort if p["hcm"])          / total * 100)
    pct_leigh   = round(sum(1 for p in cohort if p["leigh_mri"])    / total * 100)
    pct_resp    = round(sum(1 for p in cohort if p["respiratory"])  / total * 100)
    pct_seiz    = round(sum(1 for p in cohort if p["seizures"])     / total * 100)
    pct_hypo    = round(sum(1 for p in cohort if p["hypotonia"])    / total * 100)
    pct_surv    = round(sum(1 for p in cohort if p["survived_1yr"]) / total * 100)
    pct_feed    = round(sum(1 for p in cohort if p["feeding_difficulty"]) / total * 100)

    return {
        "gene": GENE,
        "alias": "C2orf64 · PET191 (yeast)",
        "protein": "168 aa · ~19 kDa · single TM helix · IMM-anchored · IMS-facing C-terminus",
        "disease": DISEASE_NAME,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "chromosome": CHROMOSOME,
        "inheritance": INHERITANCE,
        "onset": ONSET,
        "cohort_size": total,
        "avg_lactate_mM": avg_lat,
        "avg_cox_residual_pct": avg_cox,
        "biochemical_fingerprint": (
            "Isolated COX deficiency (typically <15% residual); Complexes I, II, III NORMAL. "
            "HCM on ECHO (cardinal). BN-PAGE: absent/severely reduced assembled CIV. "
            "Immunoblot: COA5 absent, MT-CO1 secondarily reduced."
        ),
        "cardinal_feature": "HCM (Hypertrophic Cardiomyopathy) — cardinal and distinguishing; absent in COX14/COA3/COX20/COX6B1",
        "key_ddx_negatives": [
            "NO hepatopathy (DDx SCO1 100% neonatal hepatic failure)",
            "NO renal tubulopathy (DDx COX10 65% Fanconi)",
            "NO anaemia (DDx COX10 80%)",
            "NO ataxia (DDx COX20 100% cardinal)",
        ],
        "kpis": {
            "hcm_pct":            pct_hcm,
            "leigh_mri_pct":      pct_leigh,
            "respiratory_pct":    pct_resp,
            "seizures_pct":       pct_seiz,
            "hypotonia_pct":      pct_hypo,
            "survived_1yr_pct":   pct_surv,
            "feeding_pct":        pct_feed,
        },
        "key_contrasts": {
            "COA5_vs_SCO2":   "Both HCM-dominant — SCO2=100% HCM via CuA metalation (MT-CO2); COA5=~88% HCM via MT-CO1 stabilisation. Distinct pathways, similar cardiac outcome.",
            "COA5_vs_COX14":  "BOTH MT-CO1 branch — COA5 HCM 88% vs COX14 NO HCM; COX14 Leigh 80% vs COA5 Leigh ~42%. ONLY WES distinguishes.",
            "COA5_vs_COA3":   "BOTH MT-CO1 branch — COA5 HCM 88% vs COA3 NO HCM; COA3 Leigh 60% vs COA5 Leigh ~42%. ONLY WES distinguishes.",
            "COA5_vs_COX20":  "BOTH at 2q11.2 locus — entirely different: COX20=MT-CO2 ataxia-childhood vs COA5=MT-CO1 HCM-neonatal. Co-located genes, opposite phenotypes.",
        },
        "assembly_pathway": "MT-CO1 co-translational stabilisation (parallel to COX14/COA3; upstream of COX10/COX15 haem insertion; upstream of SCO2/SCO1/COA6 CuA delivery)",
        "color": COLOR,
        "light": LIGHT,
    }


def get_breakdown() -> dict:
    rng    = _rng()
    cohort = _build_cohort(rng)
    total  = len(cohort)

    # Genotype distribution
    geno_dist: dict[str, int] = {}
    for p in cohort:
        geno_dist[p["genotype"]] = geno_dist.get(p["genotype"], 0) + 1
    geno_rows = [
        {"genotype": g, "n": n, "pct": round(n / total * 100)}
        for g, n in sorted(geno_dist.items(), key=lambda x: -x[1])
    ]

    # Feature prevalence
    features = [
        ("HCM (Hypertrophic Cardiomyopathy)",          "hcm",              "CARDINAL — present; KEY DDx vs COX14/COA3/COX20 (absent)"),
        ("Lactic acidosis",                             "lactic_acidosis",  "Elevated lactate/pyruvate ratio; reflects OXPHOS block"),
        ("Psychomotor delay / arrest",                  "psychomotor",      "Universal in severe cases"),
        ("Regression",                                  "regression",       "Loss of milestones in survivors"),
        ("Hypotonia",                                   "hypotonia",        "Axial and peripheral"),
        ("Feeding difficulties",                        "feeding_difficulty","Nasogastric feeding often required"),
        ("Encephalopathy",                              "encephalopathy",   "Variable severity"),
        ("Myopathy",                                    "myopathy",         "Skeletal muscle involvement"),
        ("Growth failure",                              "growth_failure",   "Poor weight gain from cardiac failure"),
        ("Leigh-like MRI",                              "leigh_mri",        "Basal ganglia/brainstem; LESS PROMINENT than COX14 (80%) or SURF1 (95%)"),
        ("Respiratory compromise",                      "respiratory",      "Secondary cardiac and central; may need NIV"),
        ("Seizures",                                    "seizures",         "Less frequent than in Leigh-dominant COXPD subtypes"),
        ("NO hepatopathy (KEY DDx)",                    "hepatopathy",      "Absent — critical differentiator from SCO1"),
        ("NO renal tubulopathy (KEY DDx)",              "renal_tubular",    "Absent — critical differentiator from COX10"),
        ("NO ataxia (KEY DDx)",                         "ataxia",           "Absent — critical differentiator from COX20"),
    ]
    feature_rows = []
    for label, key, note in features:
        count = sum(1 for p in cohort if p.get(key))
        feature_rows.append({
            "feature": label, "n": count,
            "pct": round(count / total * 100), "note": note
        })

    # Outcome by genotype class
    null_pts = [p for p in cohort if "null" in p["genotype"] or "truncating" in p["genotype"] or "Ter" in p["genotype"]]
    miss_pts = [p for p in cohort if p not in null_pts]
    out_null = round(sum(1 for p in null_pts if p["survived_1yr"]) / max(len(null_pts), 1) * 100)
    out_miss = round(sum(1 for p in miss_pts if p["survived_1yr"]) / max(len(miss_pts), 1) * 100)

    # Cardiac severity vs COX activity
    high_cox = [p for p in cohort if p["cox_pct"] > 10]
    low_cox  = [p for p in cohort if p["cox_pct"] <= 10]
    hcm_high = round(sum(1 for p in high_cox if p["hcm"]) / max(len(high_cox), 1) * 100)
    hcm_low  = round(sum(1 for p in low_cox  if p["hcm"]) / max(len(low_cox),  1) * 100)

    # Treatment contraindications
    cis = [
        {"drug": "Valproic acid (VPA)",      "severity": "ABSOLUTE CI",
         "reason": "CoA sequestration (inhibits mito β-oxidation) + POLG inhibition + hepatotoxicity risk"},
        {"drug": "Metformin",                "severity": "ABSOLUTE CI",
         "reason": "Complex I inhibition → lactic crisis; dangerous with any OXPHOS defect"},
        {"drug": "Propofol",                 "severity": "ABSOLUTE CI",
         "reason": "PRIS: direct CIV inhibition; with HCM + COX deficiency → catastrophic cardiac arrest"},
        {"drug": "Positive inotropes (digoxin, dobutamine)", "severity": "HIGH RISK",
         "reason": "Increase O₂ demand on HCM heart with severely limited ATP production"},
        {"drug": "ACE inhibitors / ARBs",    "severity": "CAUTION in obstructive HCM",
         "reason": "Afterload reduction worsens LVOT obstruction in obstructive HCM"},
        {"drug": "Linezolid",                "severity": "ABSOLUTE CI",
         "reason": "mt 23S rRNA inhibition → blocks MT-CO1/CO2/CO3 translation → eliminates residual COX"},
        {"drug": "Chloramphenicol",          "severity": "ABSOLUTE CI",
         "reason": "Mitoribosome block (same mechanism as linezolid)"},
        {"drug": "Ketogenic diet",           "severity": "CONTRAINDICATED",
         "reason": "β-oxidation requires CIV (FADH₂ → CIII→CIV); KD with COX deficiency → metabolic crisis"},
    ]

    # Recommended treatments
    txs = [
        {"tx": "Propranolol / atenolol",     "level": "Level B", "note": "First-line HCM rate/LVOT control; reduce demand on COX-deficient myocardium"},
        {"tx": "CoQ10 (ubiquinol)",          "level": "Level C", "note": "Mitochondrial cocktail; may augment residual COX activity"},
        {"tx": "Riboflavin (B2)",            "level": "Level C", "note": "Flavoprotein support (ACAD9, ETF pathways adjacent to ETC)"},
        {"tx": "Thiamine (B1) — empiric",    "level": "Level C MANDATORY", "note": "Empiric pending WES — rules out treatable SLC19A3 Leigh mimic"},
        {"tx": "Biotin — empiric",           "level": "Level C MANDATORY", "note": "Empiric pending WES — rules out treatable BTD Leigh mimic"},
        {"tx": "L-Carnitine",               "level": "Level C", "note": "Secondary carnitine deficiency common in HCM/OXPHOS disease"},
        {"tx": "Levetiracetam (LEV)",        "level": "Level B preferred", "note": "Renal excretion, no mito toxicity, no hepatic CYP; safest AED in mito disease"},
        {"tx": "Sevoflurane (NOT propofol)", "level": "Standard", "note": "Only acceptable inhalational anaesthetic; propofol ABSOLUTE CI"},
        {"tx": "GIR 6–8 mg/kg/min",         "level": "Mandatory perioperative", "note": "Never fast >4 hours; continuous glucose prevents metabolic crisis"},
        {"tx": "Nasogastric / NG feeds",     "level": "Practical standard", "note": "For feeding difficulties (~70%); reduces cardiac metabolic demand"},
        {"tx": "Cardiac transplant",         "level": "Controversial",  "note": "Does NOT cure encephalopathy or systemic COX deficiency; case-by-case"},
    ]

    # Per-patient table
    patient_rows = []
    for p in cohort:
        patient_rows.append({
            "id":         p["id"],
            "sex":        p["sex"],
            "genotype":   p["genotype"],
            "onset_mo":   p["onset_mo"],
            "lactate_mM": p["lactate_mM"],
            "cox_pct":    p["cox_pct"],
            "hcm":        "Yes" if p["hcm"]          else "No",
            "leigh_mri":  "Yes" if p["leigh_mri"]    else "No",
            "respiratory": "Yes" if p["respiratory"]  else "No",
            "seizures":   "Yes" if p["seizures"]     else "No",
            "survived_1yr": "Yes" if p["survived_1yr"] else "No",
        })

    return {
        "gene_id":       DISEASE_ID,
        "genotype_dist": geno_rows,
        "feature_prev":  feature_rows,
        "outcome": {
            "null_allele_1yr_survival_pct": out_null,
            "missense_allele_1yr_survival_pct": out_miss,
            "note": "Truncating/null alleles carry highest cardiac mortality (<30% survival at 1yr)",
        },
        "hcm_vs_cox_activity": {
            "hcm_pct_when_cox_above_10pct": hcm_high,
            "hcm_pct_when_cox_at_or_below_10pct": hcm_low,
            "note": "HCM prevalence is high across all COX activity levels — pathway role, not COX activity threshold",
        },
        "contraindications": cis,
        "treatments":        txs,
        "patient_table":     patient_rows,
        "ddx_matrix": [
            {"disease": "COX14 (COXPD6)",  "shared": "MT-CO1 branch, isolated COX, AR",    "distinguishing": "COX14: NO HCM; COA5: HCM ~88%. WES mandatory."},
            {"disease": "COA3 (COXPD10)",  "shared": "MT-CO1 branch, isolated COX, AR",    "distinguishing": "COA3: NO HCM, Leigh 60%; COA5: HCM 88%, Leigh 42%. WES mandatory."},
            {"disease": "SCO2 (COXPD2)",   "shared": "HCM dominant, isolated COX, AR",     "distinguishing": "SCO2: CuA copper pathway (MT-CO2); COA5: MT-CO1 stabilisation. SCO2 HCM 100% vs COA5 88%."},
            {"disease": "COA6 (COXPD14)",  "shared": "HCM dominant, isolated COX, AR",     "distinguishing": "COA6: copper chaperone twin-CX9C, liver 35%; COA5: MT-CO1 assembly, NO liver involvement."},
            {"disease": "COX15 (COXPD5)",  "shared": "HCM dominant, isolated COX, AR",     "distinguishing": "COX15: heme a synthase (step 2); COA5: MT-CO1 co-translational. COX15 HCM 78% vs COA5 88%."},
            {"disease": "COX20 (COXPD8)",  "shared": "2q11.2 locus, AR biallelic",         "distinguishing": "COX20: MT-CO2 ataxia-childhood; COA5: MT-CO1 HCM-neonatal. Same chromosome, opposite phenotype."},
            {"disease": "SCO1 (COXPD3)",   "shared": "Isolated COX, AR, neonatal",         "distinguishing": "SCO1: hepatopathy 100% (CARDINAL); COA5: NO hepatopathy. Hepatic failure = SCO1, not COA5."},
            {"disease": "COX10 (COXPD4)",  "shared": "Isolated COX, AR",                   "distinguishing": "COX10: Fanconi 65% + anaemia 80%; COA5: NO tubulopathy, NO anaemia."},
        ],
    }


def get_definitions() -> dict:
    return {
        "gene_id": DISEASE_ID,
        "glossary": [
            {"term": "COA5",
             "definition": (
                 "Cytochrome c Oxidase Assembly Factor 5 — nuclear-encoded, 168 aa, ~19 kDa; "
                 "single transmembrane helix anchored in the inner mitochondrial membrane (IMM) "
                 "with C-terminus facing the intermembrane space (IMS). Also designated C2orf64 "
                 "(chromosome 2 open reading frame 64) and orthologous to yeast PET191. "
                 "COA5 acts in the early MT-CO1 co-translational assembly pathway."
             )},
            {"term": "C2orf64 / PET191",
             "definition": (
                 "Historical and yeast-ortholog designations for COA5. C2orf64 was the pre-functional "
                 "annotation (chromosome 2, open reading frame 64). PET191 is the Saccharomyces "
                 "cerevisiae homologue involved in petite-negative (respiratorily competent) growth. "
                 "All designations refer to the same gene on chromosome 2q11.2."
             )},
            {"term": "COXPD11",
             "definition": (
                 "Mitochondrial Complex IV Deficiency, Nuclear Type 11 — OMIM disease designation "
                 "(#614932) for COA5-related COX deficiency. Characterised by isolated COX deficiency "
                 "(<15% residual), cardiomyopathy-dominant presentation, and neonatal/early infantile "
                 "onset. Very rare — fewer than 20 published patients worldwide."
             )},
            {"term": "HCM (Hypertrophic Cardiomyopathy) in COA5",
             "definition": (
                 "The cardinal and distinguishing clinical feature of COA5 deficiency (~88% of patients). "
                 "HCM in COXPD11 results from severely impaired ATP production in cardiomyocytes — the "
                 "highest O₂-consuming tissue. Unlike SCO2/COA6 where the pathway involves copper "
                 "metalation, COA5 HCM arises from MT-CO1 assembly failure reducing total CIV capacity. "
                 "CRITICAL: HCM separates COA5 from COX14/COA3/COX20/COX6B1 (all NO HCM)."
             )},
            {"term": "MT-CO1 co-translational assembly (COA5 role)",
             "definition": (
                 "COA5 engages MT-CO1 during early co-translational assembly at the IMM ribosome. "
                 "It functions in the same MT-CO1 branch as COX14 (C12orf62) and COA3 (MITRAC12), "
                 "stabilising newly synthesised MT-CO1 before haem a and copper (CuB) insertion by "
                 "downstream factors (COX10→COX15 for haem a; SCO2/SCO1 for CuA; SURF1 for haem a3/CuB). "
                 "Loss of COA5 destabilises the early MT-CO1 intermediate → isolated COX deficiency."
             )},
            {"term": "Isolated COX deficiency (COA5 pattern)",
             "definition": (
                 "Complex IV (COX) enzyme activity below reference range while Complexes I, II, and III "
                 "remain normal — the biochemical fingerprint of COXPD11. COA5 deficiency typically "
                 "yields 5–15% residual COX activity (slightly more than COX14/COA3 at <5%). "
                 "Biochemistry CANNOT distinguish COA5 from other isolated COX deficiency subtypes — "
                 "WES/WGS is mandatory for molecular diagnosis."
             )},
            {"term": "2q11.2 co-localisation (COA5 and COX20)",
             "definition": (
                 "Both COA5 and COX20 (COXPD8) map to chromosome 2q11.2. This is biologically "
                 "coincidental — they encode proteins in completely different assembly branches "
                 "(COA5 = MT-CO1 branch; COX20 = MT-CO2 branch). Their co-location can cause "
                 "confusing FISH results; molecular genetic testing (WES) is the definitive tool. "
                 "The phenotypes are opposite: COA5 = neonatal HCM; COX20 = childhood ataxia."
             )},
            {"term": "PRIS (propofol infusion syndrome) — cardiac amplification",
             "definition": (
                 "In COA5 deficiency, propofol's direct Complex IV inhibition is compounded by the "
                 "HCM substrate — the already energy-compromised myocardium cannot sustain contractile "
                 "function when residual COX is further suppressed. This creates a two-hit catastrophe: "
                 "PRIS (metabolic acidosis + rhabdomyolysis) + HCM decompensation. Sevoflurane is the "
                 "only acceptable general anaesthetic."
             )},
            {"term": "WES / WGS (mandatory in COA5)",
             "definition": (
                 "Whole Exome Sequencing / Whole Genome Sequencing — required to distinguish COA5 from "
                 "COX14, COA3, SCO2, COA6, COX15, and other isolated COX deficiency subtypes. "
                 "Biochemistry (isolated COX deficiency) and cardiac features (HCM) overlap between "
                 "COA5, SCO2, COA6, and COX15. The cardiac phenotype narrows the DDx but "
                 "does NOT uniquely identify COA5 — sequencing is mandatory."
             )},
            {"term": "p.Trp59Ter (c.177G>A)",
             "definition": (
                 "The most frequently documented pathogenic COA5 variant — a nonsense mutation "
                 "creating a premature stop codon at position 59, eliminating the entire IMS-facing "
                 "C-terminal domain. Found predominantly in North-African / Moroccan patients and "
                 "classified as a probable founder allele in that population. Results in complete "
                 "LOF and severe isolated COX deficiency with neonatal HCM."
             )},
        ],
        "clinical_notes": [
            (
                "COA5 deficiency should be suspected in any neonate with: (1) HCM on ECHO "
                "(cardinal), (2) isolated Complex IV deficiency (<15% in muscle/fibroblasts), "
                "(3) lactic acidosis, (4) NO hepatopathy on LFTs, (5) NO Fanconi syndrome / "
                "anaemia. KEY DDx from SCO2: COA5 lacks the 100% HCM penetrance of SCO2 "
                "(COA5 ~88%). From COA6: COA5 lacks hepatic involvement (COA6 35%). "
                "From COX15: very similar — WES is the only separator."
            ),
            (
                "Biochemical workup: muscle + fibroblast respiratory chain enzyme assay — COX "
                "activity 5–15% of normal with CI/CII/CIII entirely normal is the cardinal "
                "finding. BN-PAGE shows absent/severely reduced assembled CIV. Echocardiogram "
                "MANDATORY at diagnosis and every 3 months — HCM progression determines prognosis. "
                "12-lead ECG for arrhythmia surveillance. Holter monitoring for LQTc/SVT "
                "in established HCM."
            ),
            (
                "Cardiac management is the priority: propranolol/atenolol for LVOT obstruction "
                "and heart rate control; avoid positive inotropes (digoxin, dobutamine) — they "
                "increase O₂ demand on an ATP-depleted myocardium. Diuretics (furosemide) for "
                "symptomatic heart failure with caution. ACE inhibitors/ARBs: AVOID in obstructive "
                "HCM (reduce afterload → worsen LVOT gradient). Cardiac transplantation is "
                "controversial — does NOT cure encephalopathy or systemic COX deficiency."
            ),
            (
                "Empiric treatment while awaiting WES: thiamine 5–10 mg/kg/day (MANDATORY) + "
                "biotin 10–20 mg/day (MANDATORY) — these treatable mimics (SLC19A3, BTD) "
                "must be excluded. CoQ10 ubiquinol + riboflavin + L-carnitine as mitochondrial "
                "cocktail. Stop VPA immediately if started inadvertently. Avoid propofol for all "
                "procedures. GIR 6–8 mg/kg/min perioperative — never fast >4h."
            ),
            (
                "Prognosis: SEVERE — neonatal cardiac failure drives high early mortality (~50–70% "
                "before 1 year in null-allele patients). Survivors develop combined cardiac and "
                "neurological impairment. No disease-modifying therapy exists. Gene therapy for "
                "MT-CO1 branch assembly factors is in early preclinical investigation. Palliative "
                "care discussion is appropriate early; metabolic emergency card should accompany "
                "all patients."
            ),
            (
                "Genetic counselling: 25% recurrence risk per pregnancy for carrier parents. "
                "COA5 is on chromosome 2q11.2 — the North-African/Moroccan p.Trp59Ter allele "
                "warrants targeted cascade testing in affected communities. Prenatal molecular "
                "diagnosis is available once biallelic COA5 variants are confirmed in the proband. "
                "Neonatal siblings of affected patients should have early ECHO + lactate screening."
            ),
        ],
        "references": [
            {
                "citation": "Huigsloot M et al. (2011). Am J Hum Genet 88(4):488–493.",
                "note": (
                    "First description of human COA5 / C2orf64 deficiency (COXPD11) — identified "
                    "biallelic truncating variants in patients with neonatal cardiomyopathy and "
                    "isolated COX deficiency. Established COA5 as a nuclear-encoded assembly "
                    "factor required for MT-CO1 biogenesis in the IMM."
                ),
            },
            {
                "citation": "Stroud DA et al. (2015). Cell Metab 21(1):108–119.",
                "note": (
                    "Systematic SILAC-based survey of COX assembly factors — places COA5 in the "
                    "broader CIV assembly hierarchy. Demonstrates that COA5 acts upstream of "
                    "haem/copper insertion steps and downstream of mitoribosome docking. "
                    "Comparative context for COA5 vs COX14/COA3/SCO1/SCO2."
                ),
            },
            {
                "citation": "Gorman GS et al. (2016). Nat Rev Dis Primers 2:16080.",
                "note": (
                    "Comprehensive mitochondrial disease epidemiology and management review — "
                    "clinical framework applicable to COXPD11 and all rare isolated COX deficiency "
                    "subtypes including cardiac presentation management."
                ),
            },
            {
                "citation": "Mick DU et al. (2012). Cell Metab 16(4):449–460.",
                "note": (
                    "MITRAC complex characterisation — defines the parallel MT-CO1 (COX14/COA3) "
                    "and MT-CO2 (COX20) co-translational assembly branches. COA5 acts in the "
                    "MT-CO1 branch, providing the molecular assembly context for COA5 deficiency."
                ),
            },
            {
                "citation": "Lim SC et al. (2014). Am J Hum Genet 94(4):552–558.",
                "note": (
                    "COA6 (COXPD14) characterisation — cardiomyopathic CIV deficiency via copper "
                    "chaperone function. Direct clinical comparator to COA5 for HCM-dominant "
                    "isolated COX deficiency; distinguishes copper metalation (COA6) from MT-CO1 "
                    "assembly (COA5) as mechanisms of cardiomyopathic COX disease."
                ),
            },
        ],
        "inheritance_detail": (
            "COA5 (COXPD11) is autosomal recessive (AR). Both copies of the COA5 gene must "
            "carry pathogenic loss-of-function variants (biallelic) for disease to occur. "
            "Heterozygous carriers are clinically unaffected. Each pregnancy of two carrier "
            "parents carries a 25% risk of an affected child. COA5 is on chromosome 2q11.2; "
            "a probable founder allele (p.Trp59Ter) has been identified in North-African / "
            "Moroccan families. Prenatal molecular diagnosis is available once biallelic "
            "variants are confirmed."
        ),
        "management_summary": (
            "No disease-modifying therapy exists for COXPD11. Management is primarily cardiac + supportive: "
            "(1) HCM management: propranolol/atenolol (rate control + LVOT); avoid positive inotropes; "
            "diuretics for HF symptoms with caution. "
            "(2) Mitochondrial cocktail: CoQ10 ubiquinol, riboflavin, thiamine (MANDATORY empiric), "
            "biotin (MANDATORY empiric), L-carnitine. "
            "(3) Energy substrate: GIR 6–8 mg/kg/min perioperative — never fast >4h. "
            "(4) Anaesthesia: sevoflurane inhalational ONLY — propofol ABSOLUTE CI. "
            "(5) Seizures: levetiracetam preferred (renal excretion, no mito toxicity). "
            "(6) Feeding support: nasogastric/NG feeds for feeding difficulties. "
            "(7) Avoid ABSOLUTELY: VPA, metformin, propofol, linezolid, chloramphenicol, ketogenic diet."
        ),
    }


if __name__ == "__main__":
    import json
    print("=== COA5 COXPD11 OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== COA5 COXPD11 BREAKDOWN (truncated) ===")
    bd = get_breakdown()
    print(json.dumps({k: v for k, v in bd.items() if k != "patient_table"}, indent=2))
