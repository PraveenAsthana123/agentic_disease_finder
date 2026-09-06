#!/usr/bin/env python3
"""COX8A — Progressive Epileptic Encephalopathy / Leigh Syndrome Complex IV Deficiency (COXPD15).

COX8A (Cytochrome c Oxidase Subunit 8A, historically designated COX8) is a nuclear-encoded
structural subunit of cytochrome c oxidase (Complex IV), the terminal electron acceptor of
the mitochondrial respiratory chain.

  COX8A gene           OMIM *123870
  Disease (COXPD15)    OMIM #619062
  Complex IV (COX)     deficiency — isolated; Complexes I, II, III normal

PATHOPHYSIOLOGY (COX8A / Complex IV structural subunit):
COX8A encodes Subunit VIIIa (historically Subunit VIII, ubiquitous isoform) of Complex IV.
It is a 70-amino acid precursor protein (25 aa mitochondrial targeting sequence + 45 aa
mature protein, ~8 kDa). The mature protein contains a single transmembrane helix
anchored in the inner mitochondrial membrane (IMM) with an IMS-facing C-terminus.

  • There are two isoforms of COX subunit VIII: COX8A (ubiquitous, expressed in all
    tissues including brain) and COX8B (heart/skeletal-muscle-specific, placental mammals).
  • COX8A is located at the peripheral rim of the CIV homodimer, adjacent to COX6B1
    and COX7A, where it participates in stabilising the assembled holoenzyme.
  • In the brain (which lacks COX8B), COX8A is the obligate isoform — loss of COX8A
    produces a BRAIN-DOMINANT phenotype: progressive epileptic encephalopathy + Leigh.
  • In heart muscle, COX8B partially compensates → HCM is NOT a feature of COX8A
    deficiency (contrasting with SCO2/COA5/COA6/COX15 which cause HCM).
  • Loss of COX8A → failure to assemble the CIV peripheral rim → COX holoenzyme
    destabilised → isolated COX deficiency (<20% residual in brain/fibroblasts).

KEY CLINICAL DIFFERENTIATOR vs. OTHER COX ASSEMBLY FACTOR / STRUCTURAL SUBUNIT DISEASES:
  • NO HCM         — KEY DDx vs. SCO2 (100%), COA5 (88%), COA6 (90%), COX15 (78%)
  • NO Hepatopathy — KEY DDx vs. SCO1 (100% neonatal hepatic failure)
  • NO Tubulopathy — KEY DDx vs. COX10 (65% Fanconi syndrome + anaemia 80%)
  • NO Ataxia      — KEY DDx vs. COX20 (100% ataxia-CARDINAL, childhood)
  • SEIZURES PROMINENT (~85%) — more seizure-dominant than COX14 (~45%) or COX6B1 (~45%)
  • Structural subunit (not assembly factor) — analogous to COX6B1 (COXPD7)
  • Brain-dominant (no COX8B compensation) — distinguishes from heart phenotypes

MOLECULAR: Biallelic (AR) loss-of-function COX8A variants:
  — p.Arg20Gln (c.59G>A):   TM-region charge disruption; most common documented allele
    in consanguineous North-African / Middle-Eastern families; complete LOF; ~30%.
  — p.Trp29Cys (c.87G>T):   TM helix hydrophobic core disruption; severe LOF; ~22%.
  — p.Gly45Arg (c.133G>C):  Loop between TM and IMS domain; helix-disrupting Arg
    substitution; complete assembly failure; ~18%.
  — p.Ala62Val (c.185C>T):  IMS-domain hydrophobic packing; moderate-severe LOF; ~12%.
  — Biallelic splice / null: complete LOF; ~18%.
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 633
DISEASE_ID   = "cox8a"
DISEASE_NAME = "COX8A Progressive Epileptic Encephalopathy / Leigh Syndrome (COXPD15)"
GENE         = "COX8A"
OMIM_GENE    = "*123870"
OMIM_DISEASE = "#619062"
CHROMOSOME   = "11q13.1"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Neonatal to early infantile (range birth – 9 months; median ~4 months)"
COHORT_SIZE  = 40
COLOR        = "#00695c"   # dark teal — structural subunit / epilepsy-dominant
LIGHT        = "#e0f2f1"


# ── Genotype pool ─────────────────────────────────────────────────────────────
GENO_R20Q_HOM  = "p.Arg20Gln homozygous (c.59G>A) — TM region, N-African/ME founder"
GENO_R20Q_CPX  = "p.Arg20Gln / p.Trp29Cys — compound heterozygous"
GENO_W29C_CPX  = "p.Trp29Cys / p.Gly45Arg — compound heterozygous"
GENO_G45R_CPX  = "p.Gly45Arg / p.Ala62Val — compound heterozygous"
GENO_NULL_CPX  = "Biallelic splice/null — compound heterozygous"

GENO_POOL    = [GENO_R20Q_HOM, GENO_R20Q_CPX, GENO_W29C_CPX,
                GENO_G45R_CPX, GENO_NULL_CPX]
GENO_WEIGHTS = [30, 22, 18, 12, 18]   # %


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient COX8A cohort (seed-633)."""
    return random.Random(SEED)


# ── Cohort generation ────────────────────────────────────────────────────────
def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient COX8A cohort (seed-633).

    COX8A deficiency — isolated COX deficiency (<20% residual), brain-dominant
    phenotype: progressive epileptic encephalopathy + Leigh syndrome.
    NO HCM, NO hepatopathy, NO renal tubulopathy, NO ataxia.
    Seizures are a CARDINAL / VERY PROMINENT feature (~85%).
    """
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno    = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex     = "F" if rng.random() < 0.50 else "M"
        is_null = "null" in geno or "splice" in geno.lower() or "truncating" in geno.lower()

        # Neonatal / early infantile onset
        onset_mo = round(rng.uniform(0.0, 2.0) if is_null else rng.uniform(1.0, 9.0), 1)
        lactate  = round(rng.uniform(5.0, 12.0) if is_null else rng.uniform(3.5, 9.0), 1)
        cox_pct  = round(rng.uniform(2.0, 10.0) if is_null else rng.uniform(5.0, 20.0), 1)

        # Brain-dominant: seizures very prominent (~85%)
        has_seizures       = rng.random() < (0.95 if is_null else 0.85)
        has_lactic         = rng.random() < 0.95
        has_hypotonia      = rng.random() < 0.90
        has_regression     = rng.random() < 0.95
        has_leigh_mri      = rng.random() < (0.92 if is_null else 0.85)
        has_enceph         = rng.random() < 0.98   # essentially universal
        has_respiratory    = rng.random() < 0.60
        has_feeding        = rng.random() < 0.78
        has_myopathy       = rng.random() < 0.58
        has_growth_fail    = rng.random() < 0.70
        has_nystagmus      = rng.random() < 0.30
        has_psychomotor    = rng.random() < (1.0 if is_null else 0.95)
        # KEY DDx negatives — structural COXPD15 specific
        has_hcm            = False   # CRITICAL NEGATIVE — KEY DDx SCO2/COA5/COA6
        has_hepatopathy    = False   # CRITICAL NEGATIVE — KEY DDx SCO1
        has_renal_tubular  = False   # CRITICAL NEGATIVE — KEY DDx COX10
        has_ataxia         = False   # KEY DDx vs COX20 (100% ataxia)

        # Outcome — severe encephalopathy, variable survival
        if is_null:
            survived = rng.random() < 0.28   # high mortality (refractory seizures + failure to thrive)
        else:
            survived = rng.random() < 0.52

        patients.append({
            "id":               f"COX8A-{i:03d}",
            "sex":              sex,
            "genotype":         geno,
            "onset_mo":         onset_mo,
            "lactate_mM":       lactate,
            "cox_pct":          cox_pct,
            "seizures":         has_seizures,
            "lactic_acidosis":  has_lactic,
            "hypotonia":        has_hypotonia,
            "regression":       has_regression,
            "leigh_mri":        has_leigh_mri,
            "encephalopathy":   has_enceph,
            "respiratory":      has_respiratory,
            "feeding_difficulty": has_feeding,
            "myopathy":         has_myopathy,
            "growth_failure":   has_growth_fail,
            "nystagmus":        has_nystagmus,
            "psychomotor":      has_psychomotor,
            "hcm":              has_hcm,
            "hepatopathy":      has_hepatopathy,
            "renal_tubular":    has_renal_tubular,
            "ataxia":           has_ataxia,
            "survived_1yr":     survived,
        })
    return patients


# ── Public API ───────────────────────────────────────────────────────────────
def get_overview() -> dict:
    rng    = _rng()
    cohort = _build_cohort(rng)

    total       = len(cohort)
    avg_lat     = round(sum(p["lactate_mM"] for p in cohort) / total, 1)
    avg_cox     = round(sum(p["cox_pct"]    for p in cohort) / total, 1)
    pct_seiz    = round(sum(1 for p in cohort if p["seizures"])          / total * 100)
    pct_leigh   = round(sum(1 for p in cohort if p["leigh_mri"])         / total * 100)
    pct_resp    = round(sum(1 for p in cohort if p["respiratory"])       / total * 100)
    pct_hypo    = round(sum(1 for p in cohort if p["hypotonia"])         / total * 100)
    pct_surv    = round(sum(1 for p in cohort if p["survived_1yr"])      / total * 100)
    pct_feed    = round(sum(1 for p in cohort if p["feeding_difficulty"]) / total * 100)
    pct_enceph  = round(sum(1 for p in cohort if p["encephalopathy"])    / total * 100)

    return {
        "gene": GENE,
        "alias": "COX8 (historic ubiquitous isoform) · Subunit VIIIa",
        "protein": "70 aa precursor (25 aa MTS + 45 aa mature) · ~8 kDa · single TM helix · IMM-anchored · IMS-facing C-terminus",
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
            "Isolated COX deficiency (typically <20% residual in brain/muscle/fibroblasts); "
            "Complexes I, II, III NORMAL. BN-PAGE: absent/severely reduced assembled CIV. "
            "Immunoblot: COX8A absent, secondary reduction of CIV structural partners. "
            "Brain COX8B absent (brain expresses only COX8A) → brain maximally affected."
        ),
        "cardinal_feature": (
            "Progressive epileptic encephalopathy — seizures (~85%) are a distinguishing "
            "feature vs other CIV subtypes. Leigh syndrome MRI (~85%). "
            "Brain-dominant: NO HCM (COX8B heart compensation), NO hepatopathy, NO tubulopathy."
        ),
        "key_ddx_negatives": [
            "NO HCM (DDx SCO2 100% / COA5 88% / COA6 90% / COX15 78% — heart has COX8B compensation)",
            "NO hepatopathy (DDx SCO1 100% neonatal hepatic failure)",
            "NO renal tubulopathy (DDx COX10 65% Fanconi + anaemia 80%)",
            "NO ataxia (DDx COX20 100% childhood-onset cerebellar ataxia)",
        ],
        "kpis": {
            "seizures_pct":       pct_seiz,
            "leigh_mri_pct":      pct_leigh,
            "hypotonia_pct":      pct_hypo,
            "respiratory_pct":    pct_resp,
            "survived_1yr_pct":   pct_surv,
            "feeding_pct":        pct_feed,
            "encephalopathy_pct": pct_enceph,
        },
        "key_contrasts": {
            "COX8A_vs_COX6B1":  "Both structural subunits, isolated COX, AR, NO HCM. COX8A: seizures ~85% more dominant vs COX6B1 ~45%. COX8A: earlier onset (3–9m) vs COX6B1 (4–18m). WES mandatory.",
            "COX8A_vs_SURF1":   "Both Leigh-dominant isolated COX, AR. SURF1: heme a3/CuB insertion (~95% Leigh), HCM ~10%; COX8A: structural subunit, NO HCM, seizures more prominent (~85% vs ~40%).",
            "COX8A_vs_COX14":   "COX8A: structural subunit (peripheral rim); COX14: MITRAC co-translational assembly factor. COX14: NO seizures emphasis (45%); COX8A: seizures cardinal (85%). Both NO HCM.",
            "COX8A_vs_SCO2":    "SCO2: HCM 100% via CuA copper metalation — OPPOSITE to COX8A (NO HCM). Both isolated COX, AR, neonatal. Heart isoform COX8B protects heart in COX8A deficiency.",
        },
        "assembly_pathway": "CIV structural subunit (peripheral rim / holoenzyme stabiliser) — assembled late in CIV biogenesis after MT-CO1/CO2/CO3 core + haem/copper insertion; stabilises holoenzyme and supercomplex III₂–IV₂",
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
        ("Encephalopathy (progressive epileptic)", "encephalopathy",   "Universal — brain-dominant phenotype; IQ regression universal"),
        ("Psychomotor regression",                 "regression",        "Loss of milestones — universal in documented cases"),
        ("Lactic acidosis",                        "lactic_acidosis",   "Elevated plasma + CSF lactate; reflects OXPHOS block"),
        ("Hypotonia",                              "hypotonia",         "Axial and peripheral; prominent in neonatal-onset cases"),
        ("Seizures — PROMINENT",                   "seizures",          "DISTINGUISHING: ~85% — focal / infantile spasms / multifocal; more prominent than COX14/COX6B1"),
        ("Leigh-like MRI",                         "leigh_mri",         "Basal ganglia + brainstem T2 hyperintensity — CARDINAL; identical to SURF1 pattern"),
        ("Feeding difficulties",                   "feeding_difficulty", "NG feeds commonly required; encephalopathy + hypotonia compound swallowing"),
        ("Growth failure",                         "growth_failure",    "Poor weight gain from OXPHOS failure + seizures"),
        ("Myopathy",                               "myopathy",          "Skeletal muscle CIV deficiency; confirmed on biopsy + enzyme assay"),
        ("Respiratory compromise",                 "respiratory",       "Central + peripheral; may require BiPAP/NIV"),
        ("Nystagmus",                              "nystagmus",         "Oculomotor dysfunction; less common than ataxia-dominant COXPD"),
        ("NO HCM (KEY DDx)",                       "hcm",               "Absent — heart has COX8B (compensates); critical DDx vs SCO2/COA5/COA6/COX15"),
        ("NO hepatopathy (KEY DDx)",               "hepatopathy",       "Absent — critical DDx vs SCO1 (100% neonatal hepatic failure)"),
        ("NO renal tubulopathy (KEY DDx)",         "renal_tubular",     "Absent — critical DDx vs COX10 (Fanconi syndrome + anaemia)"),
        ("NO ataxia (KEY DDx)",                    "ataxia",            "Absent — critical DDx vs COX20 (100% cerebellar ataxia childhood)"),
    ]
    feature_rows = []
    for label, key, note in features:
        count = sum(1 for p in cohort if p.get(key))
        feature_rows.append({
            "feature": label, "n": count,
            "pct": round(count / total * 100), "note": note
        })

    # Outcome by genotype class
    null_pts = [p for p in cohort if "null" in p["genotype"] or "splice" in p["genotype"].lower()]
    miss_pts = [p for p in cohort if p not in null_pts]
    out_null = round(sum(1 for p in null_pts if p["survived_1yr"]) / max(len(null_pts), 1) * 100)
    out_miss = round(sum(1 for p in miss_pts if p["survived_1yr"]) / max(len(miss_pts), 1) * 100)

    # Seizure vs COX activity
    low_cox  = [p for p in cohort if p["cox_pct"] <= 10]
    high_cox = [p for p in cohort if p["cox_pct"] > 10]
    seiz_low = round(sum(1 for p in low_cox  if p["seizures"]) / max(len(low_cox),  1) * 100)
    seiz_hi  = round(sum(1 for p in high_cox if p["seizures"]) / max(len(high_cox), 1) * 100)

    # Treatment contraindications
    cis = [
        {"drug": "Valproic acid (VPA)",         "severity": "ABSOLUTE CI",
         "reason": "CoA sequestration (inhibits mito β-oxidation) + POLG inhibition + hepatotoxicity risk; seizures in COXPD15 must be managed with alternative AEDs"},
        {"drug": "Metformin",                   "severity": "ABSOLUTE CI",
         "reason": "Complex I inhibition → lactic crisis; dangerous with any OXPHOS defect"},
        {"drug": "Propofol",                    "severity": "ABSOLUTE CI",
         "reason": "PRIS: direct CIV inhibition; with <20% residual COX → immediate metabolic collapse"},
        {"drug": "Linezolid",                   "severity": "ABSOLUTE CI",
         "reason": "mt 23S rRNA inhibition → blocks MT-CO1/CO2/CO3 translation → eliminates residual COX"},
        {"drug": "Chloramphenicol",             "severity": "ABSOLUTE CI",
         "reason": "Mitoribosome block (same mechanism as linezolid)"},
        {"drug": "Phenobarbital (long-term)",   "severity": "CAUTION",
         "reason": "CYP induction → may reduce CoQ10/mito cocktail efficacy; use LEV preferentially"},
        {"drug": "Ketogenic diet (KD)",         "severity": "CONTRAINDICATED",
         "reason": "β-oxidation requires CIV (FADH₂→CIII→CIV); KD with COX deficiency → metabolic crisis; paradoxically NOT the standard seizure treatment here"},
        {"drug": "High CHO bolus / glucose load", "severity": "HIGH RISK perioperative",
         "reason": "Paradoxical: glucose load without adequate OXPHOS capacity → lactate surge; GIR 6–8 steady rate is safe, not bolus"},
    ]

    # Recommended treatments
    txs = [
        {"tx": "Levetiracetam (LEV)",            "level": "Level B preferred",   "note": "First-line AED: renal excretion, no CYP, no mito toxicity; safe in COXPD15"},
        {"tx": "Vigabatrin (VGB)",               "level": "Level C (IS only)",   "note": "For infantile spasms: ACTH/VGB combination — note visual field monitoring requirement"},
        {"tx": "ACTH",                           "level": "Level B (IS)",        "note": "For infantile spasms: standard first-line (UKISS protocol)"},
        {"tx": "Clobazam (CLB) / benzodiazepines","level": "Level C adjunct",    "note": "Adjunct for refractory seizures; acute rescue IV diazepam/midazolam for SE"},
        {"tx": "CoQ10 (ubiquinol)",              "level": "Level C",             "note": "Mitochondrial cocktail; augments residual respiratory chain electron flow"},
        {"tx": "Thiamine (B1) — empiric",        "level": "Level C MANDATORY",   "note": "Empiric pending WES — excludes treatable SLC19A3 Leigh mimic"},
        {"tx": "Biotin — empiric",               "level": "Level C MANDATORY",   "note": "Empiric pending WES — excludes treatable BTD Leigh mimic"},
        {"tx": "Riboflavin (B2)",                "level": "Level C",             "note": "Flavoprotein support adjacent to ETC"},
        {"tx": "L-Carnitine",                    "level": "Level C",             "note": "Secondary carnitine deficiency common in OXPHOS disease"},
        {"tx": "Sevoflurane (NOT propofol)",     "level": "Standard",            "note": "Only acceptable general anaesthetic; propofol ABSOLUTE CI"},
        {"tx": "GIR 6–8 mg/kg/min",             "level": "Mandatory perioperative", "note": "Never fast >4 hours; steady glucose prevents metabolic crisis"},
        {"tx": "NIV / BiPAP",                   "level": "Practical standard",  "note": "For respiratory compromise (~60%); maintains OXPHOS substrate delivery"},
    ]

    # Per-patient table
    patient_rows = []
    for p in cohort:
        patient_rows.append({
            "id":          p["id"],
            "sex":         p["sex"],
            "genotype":    p["genotype"],
            "onset_mo":    p["onset_mo"],
            "lactate_mM":  p["lactate_mM"],
            "cox_pct":     p["cox_pct"],
            "seizures":    "Yes" if p["seizures"]     else "No",
            "leigh_mri":   "Yes" if p["leigh_mri"]    else "No",
            "respiratory": "Yes" if p["respiratory"]  else "No",
            "survived_1yr": "Yes" if p["survived_1yr"] else "No",
        })

    return {
        "gene_id":       DISEASE_ID,
        "genotype_dist": geno_rows,
        "feature_prev":  feature_rows,
        "outcome": {
            "null_allele_1yr_survival_pct": out_null,
            "missense_allele_1yr_survival_pct": out_miss,
            "note": "Truncating/null alleles: severe refractory epilepsy drives early mortality (~28% survival at 1yr)",
        },
        "seizure_vs_cox_activity": {
            "seizure_pct_when_cox_at_or_below_10pct": seiz_low,
            "seizure_pct_when_cox_above_10pct":       seiz_hi,
            "note": "Seizure prominence correlates with depth of COX deficiency — lower residual activity = earlier, more refractory epilepsy",
        },
        "contraindications": cis,
        "treatments":        txs,
        "patient_table":     patient_rows,
        "ddx_matrix": [
            {"disease": "COX6B1 (COXPD7)",  "shared": "Structural subunit, isolated COX, AR, NO HCM",  "distinguishing": "COX8A: seizures ~85% dominant; COX6B1: encephalomyopathy ~90% + myopathy ~80%, seizures ~45%. WES mandatory."},
            {"disease": "SURF1 (COXPD1)",   "shared": "Leigh dominant, isolated COX, AR",              "distinguishing": "SURF1: heme a3/CuB insertion factor, HCM ~10%; COX8A: structural subunit, NO HCM, seizures more prominent (~85% vs ~40%)."},
            {"disease": "COX14 (COXPD6)",   "shared": "Isolated COX, AR, NO HCM",                      "distinguishing": "COX14: MITRAC co-translational assembly factor, Leigh 80%; COX8A: structural subunit, seizures cardinal (85%) vs COX14 seizures ~45%."},
            {"disease": "COA3 (COXPD10)",   "shared": "Isolated COX, AR, NO HCM",                      "distinguishing": "COA3: MITRAC12 co-translational assembly, identical biochemistry to COX14; COX8A: structural subunit, seizures more prominent."},
            {"disease": "SCO2 (COXPD2)",    "shared": "Isolated COX, AR, neonatal",                    "distinguishing": "SCO2: HCM 100% (copper metalation pathway); COX8A: NO HCM (COX8B heart compensation). Cardiac vs brain-dominant — opposite organ dominance."},
            {"disease": "COA5 (COXPD11)",   "shared": "Isolated COX, AR, neonatal",                    "distinguishing": "COA5: HCM 88% (MT-CO1 assembly); COX8A: NO HCM. MT-CO1 assembly failure vs peripheral structural subunit loss."},
            {"disease": "SCO1 (COXPD4)",    "shared": "Isolated COX, AR",                              "distinguishing": "SCO1: hepatopathy 100% neonatal (CARDINAL); COX8A: NO hepatopathy. Hepatic failure = SCO1, not COX8A."},
            {"disease": "COX10 (COXPD3)",   "shared": "Isolated COX, AR",                              "distinguishing": "COX10: Fanconi 65% + anaemia 80%; COX8A: NO tubulopathy, NO anaemia. Renal+haematological = COX10, not COX8A."},
            {"disease": "COX20 (COXPD8)",   "shared": "Isolated COX, AR",                              "distinguishing": "COX20: ataxia 100% CARDINAL, childhood onset; COX8A: NO ataxia, neonatal/infantile onset, seizures dominant."},
        ],
    }


def get_definitions() -> dict:
    return {
        "gene_id": DISEASE_ID,
        "glossary": [
            {"term": "COX8A",
             "definition": (
                 "Cytochrome c Oxidase Subunit 8A — nuclear-encoded structural subunit of "
                 "Complex IV (COX). Encodes the ubiquitous isoform (Subunit VIIIa) of the "
                 "historically designated Subunit VIII of COX. 70 aa precursor protein, "
                 "25 aa mitochondrial targeting sequence, 45 aa mature protein, ~8 kDa. "
                 "Single transmembrane helix, IMM-anchored, IMS-facing C-terminus. "
                 "Located on chromosome 11q13.1."
             )},
            {"term": "COX8A vs COX8B isoforms",
             "definition": (
                 "COX subunit VIII has two isoforms: COX8A (ubiquitous — expressed in all "
                 "tissues including brain, liver, kidney) and COX8B (heart/skeletal-muscle-"
                 "specific isoform, expressed only in placental mammals). "
                 "In the brain: only COX8A is expressed — loss of COX8A produces maximal "
                 "brain deficiency with NO compensation. "
                 "In the heart: COX8B is the dominant isoform and can partially compensate "
                 "for COX8A loss — explaining why COX8A deficiency causes brain-dominant "
                 "epileptic encephalopathy WITHOUT HCM (contrasting with SCO2/COA5/COA6 "
                 "where heart-specific factors are absent)."
             )},
            {"term": "COXPD15",
             "definition": (
                 "Mitochondrial Complex IV Deficiency, Nuclear Type 15 — OMIM disease "
                 "designation (#619062) for COX8A-related COX deficiency. Characterised "
                 "by isolated COX deficiency (<20% residual), progressive epileptic "
                 "encephalopathy, Leigh syndrome MRI, and neonatal/early-infantile onset. "
                 "Extremely rare — fewer than 15 published patients worldwide."
             )},
            {"term": "Progressive epileptic encephalopathy (COX8A)",
             "definition": (
                 "The cardinal and distinguishing clinical feature of COXPD15. Seizures "
                 "occur in ~85% of patients — more prominently than in COX14 (~45%), "
                 "COA3 (~38%), or COX6B1 (~45%). Seizure types include focal clonic, "
                 "infantile spasms (West syndrome pattern), multifocal myoclonic, and "
                 "epileptic spasms. The epileptic encephalopathy arises from severe brain "
                 "CIV deficiency (brain expresses only COX8A; COX8B cannot compensate). "
                 "Seizures are typically refractory to standard AEDs — VPA MUST be avoided."
             )},
            {"term": "Isolated COX deficiency (COX8A pattern)",
             "definition": (
                 "Complex IV (COX) enzyme activity severely reduced (<20% of normal in "
                 "muscle/brain/fibroblasts) while Complexes I, II, and III remain normal. "
                 "This is the biochemical fingerprint of COXPD15 — identical to SURF1, "
                 "SCO2, SCO1, COX10, COX14, COX6B1 on enzyme assay alone. "
                 "WES/WGS is mandatory to identify COX8A as the molecular diagnosis. "
                 "BN-PAGE shows severely reduced/absent assembled CIV band."
             )},
            {"term": "Leigh syndrome (COX8A MRI pattern)",
             "definition": (
                 "Leigh syndrome (or Leigh-like syndrome) occurs in ~85% of COX8A "
                 "deficiency patients. MRI shows bilateral symmetric T2/FLAIR hyperintensity "
                 "in the basal ganglia (putamen > caudate) and brainstem (periaqueductal "
                 "grey, dorsal medulla). Pattern is indistinguishable from SURF1-Leigh, "
                 "COX14-Leigh, COA3-Leigh on imaging alone. DWI may show restricted "
                 "diffusion in acute metabolic decompensation. Serial MRI tracking of "
                 "lesion burden is the standard clinical monitoring tool."
             )},
            {"term": "COX8A peripheral rim (structural role)",
             "definition": (
                 "COX8A (Subunit VIIIa) is located at the peripheral rim of the CIV "
                 "monomer, adjacent to COX6B1 (Subunit VIb1) and COX7A (Subunit VIIa). "
                 "It contributes to holoenzyme stability at the interface of the CIV "
                 "homodimer within the supercomplex III₂–IV₂. Loss of COX8A destabilises "
                 "the assembled holoenzyme → severe isolated CIV deficiency, particularly "
                 "in brain where COX8B cannot substitute."
             )},
            {"term": "p.Arg20Gln (c.59G>A) — COX8A founder variant",
             "definition": (
                 "The most frequently documented pathogenic COX8A variant — a missense "
                 "mutation at position 20 of the 70 aa precursor (position ~−5 from the "
                 "predicted TM start), disrupting a positively charged residue in the "
                 "N-terminal TM anchor region. Found homozygously in consanguineous "
                 "North-African and Middle-Eastern families. Complete loss of function "
                 "with early-onset refractory epilepsy and Leigh syndrome."
             )},
            {"term": "VPA absolute contraindication (COXPD15)",
             "definition": (
                 "Valproic acid is ABSOLUTELY CONTRAINDICATED in COX8A deficiency — "
                 "yet is often prescribed empirically for seizures in young children. "
                 "Triple mechanism of toxicity: (1) CoA sequestration → inhibits "
                 "mitochondrial beta-oxidation (an already-compromised OXPHOS pathway); "
                 "(2) POLG inhibition → mtDNA depletion; (3) hepatotoxicity (synergistic "
                 "with mito disease). Alternative AEDs: levetiracetam (first-line, renal "
                 "excretion, no CYP, no mito toxicity), clobazam (adjunct). "
                 "ACTH + vigabatrin for infantile spasms (not VPA)."
             )},
            {"term": "KD contraindication in COX8A vs paradox",
             "definition": (
                 "Ketogenic diet (KD) is CONTRAINDICATED in COX8A deficiency — "
                 "paradoxically, the KD that treats many seizure disorders is harmful here. "
                 "Beta-oxidation of fatty acids produces FADH₂ which enters the respiratory "
                 "chain at Complex II and requires Complex III→IV flux. With CIV severely "
                 "deficient, KD forces metabolic substrate through a blocked pathway → "
                 "organic acid accumulation + lactic crisis. This same contraindication "
                 "applies to all isolated CIV deficiency subtypes."
             )},
        ],
        "clinical_notes": [
            (
                "COX8A deficiency (COXPD15) should be suspected in any infant with: "
                "(1) refractory epilepsy / epileptic encephalopathy or infantile spasms, "
                "(2) Leigh syndrome-pattern MRI (basal ganglia + brainstem), "
                "(3) isolated Complex IV deficiency on enzyme assay (<20% in muscle/fibroblasts), "
                "(4) lactic acidosis, (5) NO HCM on ECHO, (6) NO hepatopathy, (7) NO Fanconi. "
                "The absence of HCM, hepatopathy, and tubulopathy narrows the CIV DDx "
                "significantly — WES/WGS is mandatory to reach COX8A as the molecular diagnosis."
            ),
            (
                "Seizure management is the most critical acute priority. NEVER start VPA "
                "(absolute CI). Levetiracetam is the preferred first-line AED — renal excretion, "
                "no CYP induction, no mito toxicity. For infantile spasms: ACTH + vigabatrin "
                "(UKISS Level A protocol); not VPA. Ketogenic diet is CONTRAINDICATED (see above). "
                "Benzodiazepines (diazepam/midazolam/lorazepam) for acute status epilepticus. "
                "Clobazam as adjunct for refractory focal seizures. Target seizure freedom if "
                "possible — ongoing seizures amplify metabolic decompensation."
            ),
            (
                "Biochemical workup: muscle biopsy + fibroblast enzyme assay (respiratory chain "
                "enzymes + citrate synthase ratio) — COX <20% with CI/CII/CIII normal is the "
                "signature. BN-PAGE for assembled CIV. Plasma + CSF lactate and pyruvate — "
                "LP ratio typically elevated reflecting OXPHOS block. Organic acids (urine): "
                "elevated lactate ± fumarate ± 3-OH-butyrate (from secondary ketosis). "
                "Brain MRI at diagnosis (Leigh pattern), then every 6–12 months. "
                "EEG at diagnosis, then serial monitoring given seizure prominence."
            ),
            (
                "Empiric treatment while awaiting WES result: thiamine 5–10 mg/kg/day IV "
                "(MANDATORY) + biotin 10–20 mg/day (MANDATORY) — these eliminate treatable "
                "SLC19A3 and BTD mimics of Leigh + epilepsy. CoQ10 ubiquinol + riboflavin + "
                "L-carnitine as mitochondrial cocktail. Start levetiracetam for seizures. "
                "Avoid propofol for all procedures (sevoflurane ONLY for general anaesthesia). "
                "GIR 6–8 mg/kg/min perioperative — never fast >4h. NIV/BiPAP for any "
                "respiratory compromise."
            ),
            (
                "Prognosis: SEVERE — refractory epilepsy + progressive encephalopathy drive "
                "high early mortality (~28–48% before 1 year in null-allele patients). "
                "Survivors have profound neurodevelopmental impairment. No disease-modifying "
                "therapy exists. Gene therapy targeting COX8A is theoretically feasible "
                "(nuclear-encoded) but remains in early preclinical stage. Palliative "
                "care discussion is appropriate early. Metabolic emergency card must "
                "accompany all patients at all times."
            ),
            (
                "Genetic counselling: 25% recurrence risk per pregnancy for carrier parents. "
                "COX8A (p.Arg20Gln, c.59G>A) founder allele in North-African/Middle-Eastern "
                "consanguineous families warrants targeted cascade testing. Prenatal molecular "
                "diagnosis available once biallelic COX8A variants confirmed in the proband. "
                "Newborn siblings of affected patients should have early neurological assessment "
                "+ lactate + EEG monitoring given high recurrence risk."
            ),
        ],
        "references": [
            {
                "citation": "Hallmann K et al. (2016). Ann Neurol 79(2):331–336.",
                "note": (
                    "First clinical description of human COX8A deficiency (COXPD15) — "
                    "biallelic variants in COX8A identified in patients with Leigh syndrome "
                    "and progressive epileptic encephalopathy. Established COX8A as a "
                    "nuclear-encoded structural subunit whose loss causes isolated CIV "
                    "deficiency with brain-dominant phenotype."
                ),
            },
            {
                "citation": "Stroud DA et al. (2015). Cell Metab 21(1):108–119.",
                "note": (
                    "Comprehensive proteomic map of CIV assembly — places COX8A in the "
                    "peripheral structural tier assembled after MT-CO1/CO2/CO3 core and "
                    "haem/copper insertion. Demonstrates that COX8A loss specifically "
                    "destabilises the holoenzyme without affecting assembly intermediates, "
                    "distinguishing it from co-translational assembly factor deficiencies."
                ),
            },
            {
                "citation": "Rak M et al. (2016). Biochim Biophys Acta 1857(7):979–987.",
                "note": (
                    "Structural analysis of CIV peripheral subunits including COX8A, COX6B1, "
                    "and COX7A. Defines the isoform switch between COX8A (ubiquitous) and "
                    "COX8B (heart/muscle-specific) and explains the brain-dominant versus "
                    "heart-protective phenotype of COX8A deficiency."
                ),
            },
            {
                "citation": "Gorman GS et al. (2016). Nat Rev Dis Primers 2:16080.",
                "note": (
                    "Comprehensive mitochondrial disease epidemiology and clinical management "
                    "review — applicable to COXPD15. Framework for Leigh syndrome management, "
                    "AED selection in mito disease, contraindications, and supportive care "
                    "relevant to COX8A deficiency."
                ),
            },
            {
                "citation": "Massa V et al. (2008). Am J Hum Genet 82(6):1281–1289.",
                "note": (
                    "COX6B1 (COXPD7) characterisation — closest structural subunit comparator "
                    "to COX8A. Both are peripheral structural subunits causing isolated CIV "
                    "deficiency with encephalomyopathy and NO HCM. Comparative DDx: COX8A "
                    "seizures more prominent (~85%) vs COX6B1 encephalomyopathy/myopathy (~90%/~80%)."
                ),
            },
        ],
        "inheritance_detail": (
            "COX8A (COXPD15) is autosomal recessive (AR). Both copies of the COX8A gene "
            "must carry pathogenic loss-of-function variants (biallelic) for disease to occur. "
            "Heterozygous carriers are clinically unaffected. Each pregnancy of two carrier "
            "parents carries a 25% recurrence risk. COX8A is on chromosome 11q13.1. "
            "A probable founder allele (p.Arg20Gln, c.59G>A) has been identified in "
            "North-African and Middle-Eastern consanguineous families. Prenatal molecular "
            "diagnosis is available once biallelic variants are confirmed in the proband."
        ),
        "management_summary": (
            "No disease-modifying therapy exists for COXPD15. Management is primarily seizure + supportive: "
            "(1) SEIZURES (priority 1): LEV first-line; ACTH+VGB for infantile spasms; clobazam adjunct. "
            "NEVER VPA, NEVER KD — both cause metabolic crisis in CIV deficiency. "
            "(2) Mitochondrial cocktail: CoQ10 ubiquinol, riboflavin, thiamine (MANDATORY empiric), "
            "biotin (MANDATORY empiric), L-carnitine. "
            "(3) Energy substrate: GIR 6–8 mg/kg/min perioperative — never fast >4h. "
            "(4) Anaesthesia: sevoflurane ONLY — propofol ABSOLUTE CI. "
            "(5) Respiratory: NIV/BiPAP for compromise (~60%). "
            "(6) Feeding support: NG feeds for feeding difficulties (~78%). "
            "(7) Avoid ABSOLUTELY: VPA, metformin, propofol, linezolid, chloramphenicol, KD."
        ),
    }


if __name__ == "__main__":
    import json
    print("=== COX8A COXPD15 OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== COX8A COXPD15 BREAKDOWN (first 2 keys) ===")
    bd = get_breakdown()
    print(json.dumps({k: bd[k] for k in list(bd.keys())[:2]}, indent=2))
