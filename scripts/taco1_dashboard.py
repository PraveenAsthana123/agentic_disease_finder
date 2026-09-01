#!/usr/bin/env python3
"""TACO1 — Leigh Syndrome (childhood-onset) + Complex IV (COX) Deficiency due to
Translational Activator of Cytochrome c Oxidase Subunit I deficiency.

TACO1 (Translational Activator of Cytochrome c Oxidase I, also C17orf96) encodes
a nuclear-encoded mitochondrial RNA-binding protein (343 aa) that specifically
activates translation of the MT-CO1 (COX subunit 1) mRNA by the mitochondrial
ribosome. Loss of TACO1 reduces MT-CO1 translation → reduced COX1 protein →
impaired Complex IV (COX) assembly → isolated COX deficiency.

  TACO1 gene        OMIM *612958
  Disease           Leigh syndrome with Complex IV deficiency (OMIM #256000, from TACO1)
  Complex IV (COX)  deficiency — isolated; Complexes I, II, III normal
  Chromosome        17q23.3
  Inheritance       AR (autosomal recessive biallelic)

PATHOPHYSIOLOGY (TACO1 / MT-CO1 translational activation / Complex IV assembly):
TACO1 is a 343 amino acid mitochondrial matrix protein with RNA-binding activity:
  • Associates with the mitochondrial ribosome small subunit (mt-SSU)
  • Specifically binds the 5' UTR region of MT-CO1 mRNA
  • Promotes ribosome recruitment and translational initiation of MT-CO1
  • Does NOT activate MT-CO2 or MT-CO3 — mechanistically distinct from LRPPRC
    (which stabilises all mt-mRNAs via polyadenylation)

Mechanistic cascade:
  TACO1 deficiency
  → ↓ MT-CO1 (COX1 / Complex IV core subunit 1) translation rate
  → ↓ COX1 protein in mitochondrial inner membrane
  → Impaired Complex IV (COX) holoenzyme assembly
    (COX1 is the catalytic and assembly scaffold for all 13-subunit Complex IV)
  → Isolated Complex IV deficiency (<20-30% of control in severe cases)
  → Complexes I, II, III remain NORMAL — same biochemical fingerprint as
    COX10, COX15, SURF1, SCO1, SCO2

DISTINGUISHING FEATURE — LATER ONSET AND MILDER COURSE:
  • TACO1 onset: childhood (median 3-8 years) — MUCH LATER than SURF1/SCO2/COX15
    (infantile, 1-6 months)
  • Survival: many patients survive into teens/adulthood; milder than other COX diseases
  • Dysarthria (speech motor disorder): PROMINENT (~85%), cardinal differentiating feature
    from pure ataxic/encephalopathic Leigh — TACO1 patients often present with
    progressive speech deterioration as an early sign
  • Progressive ataxia, cognitive decline, and spasticity dominate the course
  • NO HCM (KEY DDx SCO2 100%/COX15 78%)
  • NO renal tubulopathy (KEY DDx COX10 65%)
  • NO hepatopathy (KEY DDx SCO1 100%)
  • NO iron overload (KEY DDx GRACILE)

MOLECULAR: Biallelic (AR) loss-of-function TACO1 variants:
  — c.472insC (p.Tyr158Leufs*): First reported human TACO1 mutations (Weraarpachai 2009);
    frameshift; homozygous in two Inuit siblings from northern Canada; complete absence of
    TACO1 protein; severe COX deficiency; Leigh MRI; progressive neurological decline
  — c.1A>C (p.Met1?): Start-codon disruption — no protein production; severe
  — c.500C>T (p.Ala167Val): Missense; reduced but not abolished TACO1; milder course
  — Compound heterozygous missense variants: various populations; partial residual function
  — Overall: <20 unrelated families reported worldwide (ultra-rare)

References:
  Weraarpachai W et al. Am J Hum Genet. 2009;85(6):751-757.
  Sasarman F et al. Hum Mol Genet. 2010;19(14):2958-2969.
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 603
DISEASE_ID   = "taco1"
DISEASE_NAME = "TACO1 Leigh Syndrome Childhood-Onset (Complex IV / COX Deficiency)"
GENE         = "TACO1"
PROTEIN      = "TACO1 — 343 aa, mitochondrial matrix, MT-CO1 mRNA translational activator"
OMIM_GENE    = "*612958"
OMIM_DISEASE = "#256000 (Leigh syndrome, TACO1-related Complex IV deficiency)"
CHROMOSOME   = "17q23.3"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Childhood (median 3-8 years; range: 1-15 years) — LATER than SURF1/SCO2/COX15"
COHORT_SIZE  = 40
COLOR        = "#00695c"   # teal — mRNA translation biology; distinct from red/magenta cardiac diseases
LIGHT        = "#e0f2f1"

# Genotype pool
GENO_C472INSC  = "c.472insC (p.Tyr158Leufs*) — frameshift homozygous; Inuit founder; first reported (Weraarpachai 2009)"
GENO_A1C       = "c.1A>C (p.Met1?) — start-codon disruption; no protein; severe"
GENO_A167V     = "c.500C>T (p.Ala167Val) — missense; partial residual function; milder"
GENO_COMP_MS   = "Compound heterozygous missense / missense — partial loss; moderate course"
GENO_COMP_NULL = "Missense / frameshift — compound heterozygous; null + hypomorph"

GENO_POOL    = [GENO_C472INSC, GENO_A1C, GENO_A167V, GENO_COMP_MS, GENO_COMP_NULL]
GENO_WEIGHTS = [0.35,           0.15,      0.20,        0.18,          0.12]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient TACO1 cohort (seed-603)."""
    return random.Random(SEED)


# ── Patient cohort ────────────────────────────────────────────────────────────
_TX_POOL = [
    "IV Dextrose GIR 6-8",
    "CoQ10/Ubiquinol",
    "Riboflavin B2",
    "Thiamine B1",
    "Biotin (empiric, pre-molecular)",
    "Succinate (anaplerotic)",
    "NaHCO3 (lactic acidosis)",
    "LEV (seizures)",
    "Physiotherapy (ataxia/spasticity)",
    "Speech therapy (dysarthria)",
    "Supportive ICU (crisis)",
    "Carnitine (secondary deficiency)",
    "NIV/BiPAP (respiratory failure)",
]

_OUTCOMES = [
    "Alive — moderate cognitive + motor impairment (childhood)",
    "Alive — severe disability, device-dependent (teen/adult)",
    "Alive — mild-moderate, ambulatory with support",
    "Died — Leigh crisis + respiratory failure",
    "Died — progressive neurological failure (young adult)",
]
_OUT_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]


def _generate_cohort(rng: random.Random) -> list[dict[str, Any]]:
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno           = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex            = rng.choice(["M", "F"])
        onset_yr       = rng.choices([1, 2, 3, 4, 5, 6, 8, 10, 12, 15],
                                      weights=[3, 5, 12, 14, 14, 12, 10, 10, 8, 5])[0] if rng.random() > 0.05 else rng.randint(1, 3)
        lactate        = round(rng.uniform(2.0, 10.0), 1)   # moderate — milder than SCO2/COX15
        cox_pct        = rng.randint(15, 35)                 # milder deficiency (~20-30%)

        has_dysarthria  = rng.random() < 0.85   # CARDINAL — prominent early feature
        has_ataxia      = rng.random() < 0.80
        has_cognitive   = rng.random() < 0.90   # progressive cognitive decline
        has_spasticity  = rng.random() < 0.70
        has_leigh_mri   = rng.random() < 0.75
        has_hypotonia   = rng.random() < 0.55   # LESS prominent than infantile diseases
        has_seizures    = rng.random() < 0.50
        has_nystagmus   = rng.random() < 0.40
        has_optic       = rng.random() < 0.30
        has_resp        = rng.random() < 0.40   # LESS respiratory than SURF1
        has_hcm         = rng.random() < 0.04   # RARE — KEY DDx from SCO2/COX15
        has_tubulopathy = rng.random() < 0.06   # RARE — KEY DDx from COX10
        has_regression  = rng.random() < 0.88

        feat_list = ["Lactic acidosis (moderate)"]
        if has_dysarthria:   feat_list.append("Dysarthria (CARDINAL — speech motor)")
        if has_cognitive:    feat_list.append("Cognitive decline / intellectual disability")
        if has_ataxia:       feat_list.append("Progressive ataxia")
        if has_spasticity:   feat_list.append("Spasticity")
        if has_regression:   feat_list.append("Psychomotor regression")
        if has_leigh_mri:    feat_list.append("Leigh/Leigh-like MRI")
        if has_hypotonia:    feat_list.append("Hypotonia")
        if has_seizures:     feat_list.append("Seizures")
        if has_nystagmus:    feat_list.append("Nystagmus")
        if has_optic:        feat_list.append("Optic atrophy")
        if has_resp:         feat_list.append("Respiratory compromise")
        if has_hcm:          feat_list.append("HCM (RARE)")
        if has_tubulopathy:  feat_list.append("Renal tubulopathy (RARE)")

        txs     = rng.sample(_TX_POOL, k=rng.randint(3, 6))
        outcome = rng.choices(_OUTCOMES, weights=_OUT_WEIGHTS)[0]

        patients.append({
            "id":              f"TACO1-{i:03d}",
            "geno":            geno,
            "sex":             sex,
            "onset_yr":        onset_yr,
            "lactate":         lactate,
            "cox_pct":         cox_pct,
            "has_dysarthria":  has_dysarthria,
            "has_ataxia":      has_ataxia,
            "has_cognitive":   has_cognitive,
            "has_spasticity":  has_spasticity,
            "has_leigh_mri":   has_leigh_mri,
            "has_hypotonia":   has_hypotonia,
            "has_seizures":    has_seizures,
            "has_nystagmus":   has_nystagmus,
            "has_optic":       has_optic,
            "has_resp":        has_resp,
            "has_hcm":         has_hcm,
            "has_tubulopathy": has_tubulopathy,
            "has_regression":  has_regression,
            "features":        ", ".join(feat_list[:7]),
            "treatments":      ", ".join(txs[:5]),
            "outcome":         outcome,
        })
    return patients


# ── Overview ──────────────────────────────────────────────────────────────────
def get_overview() -> dict[str, Any]:
    rng      = _rng()
    patients = _generate_cohort(rng)

    died  = sum(1 for p in patients if p["outcome"].startswith("Died"))
    alive = COHORT_SIZE - died

    def _pct(key: str) -> int:
        return round(sum(1 for p in patients if p[key]) / COHORT_SIZE * 100)

    feature_frequencies = {
        "Cognitive Decline / Intellectual Disability (CARDINAL)":   _pct("has_cognitive"),
        "Dysarthria — Speech Motor Disorder (CARDINAL DISTINGUISHING)": _pct("has_dysarthria"),
        "Psychomotor Regression":                                    _pct("has_regression"),
        "Progressive Ataxia":                                        _pct("has_ataxia"),
        "Spasticity":                                                _pct("has_spasticity"),
        "Leigh / Leigh-like MRI":                                    _pct("has_leigh_mri"),
        "Lactic Acidosis (moderate)":                                round(sum(1 for p in patients if p["lactate"] >= 2.0) / COHORT_SIZE * 100),
        "Seizures":                                                   _pct("has_seizures"),
        "Nystagmus":                                                  _pct("has_nystagmus"),
        "Optic Atrophy":                                              _pct("has_optic"),
        "Hypotonia":                                                  _pct("has_hypotonia"),
        "Respiratory Compromise (LESS than SURF1)":                  _pct("has_resp"),
        "HCM (RARE — KEY DDx SCO2 100% / COX15 78%)":               _pct("has_hcm"),
        "Renal Tubulopathy (RARE — KEY DDx COX10 65%)":             _pct("has_tubulopathy"),
        "NO Hepatopathy (KEY DDx SCO1/POLG/DGUOK)":                 100,
        "NO Iron Overload (KEY DDx GRACILE)":                        100,
        "Isolated COX Deficiency (CI/CII/CIII Normal)":              100,
        "Alive (longer survival vs SCO2/COX15)":                     round(alive / COHORT_SIZE * 100),
    }

    kpis = [
        {"label": "Cohort (n)",     "value": COHORT_SIZE,                                                       "color": COLOR},
        {"label": "Dysarthria",     "value": f"{feature_frequencies['Dysarthria — Speech Motor Disorder (CARDINAL DISTINGUISHING)']}%", "color": "#00897b"},
        {"label": "Leigh MRI",      "value": f"{feature_frequencies['Leigh / Leigh-like MRI']}%",               "color": "#6a1b9a"},
        {"label": "Lactic Acidosis","value": f"{feature_frequencies['Lactic Acidosis (moderate)']}%",            "color": "#b71c1c"},
        {"label": "Ataxia",         "value": f"{feature_frequencies['Progressive Ataxia']}%",                   "color": COLOR},
        {"label": "Cognitive",      "value": f"{feature_frequencies['Cognitive Decline / Intellectual Disability (CARDINAL)']}%", "color": "#00695c"},
        {"label": "Fatal",          "value": f"{round(died/COHORT_SIZE*100)}%",                                  "color": "#c62828"},
        {"label": "Seed",           "value": f"#{SEED}",                                                         "color": "#455a64"},
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "severity": "ABSOLUTE CI",
            "mechanism": (
                "CoA sequestration (depletes mitochondrial acetyl-CoA), POLG inhibition "
                "(reduces mtDNA copy number → further reduces MT-CO1 template available "
                "for the residual TACO1-mediated translation), and hepatotoxicity. "
                "In TACO1 disease, MT-CO1 translation is already specifically reduced; "
                "VPA's POLG inhibition compounds this by reducing the MT-CO1 mRNA template "
                "pool. VPA may precipitate fatal lactic acidosis or acute hepatic failure. "
                "Use LEV (renal excretion, no mitochondrial toxicity) as first-line AED."
            ),
        },
        {
            "drug": "Metformin",
            "severity": "ABSOLUTE CI",
            "mechanism": (
                "Inhibits Complex I (NADH:ubiquinone oxidoreductase) → forces dependence "
                "on glycolysis → lactic acidosis. TACO1 patients have impaired OXPHOS "
                "(isolated COX/Complex IV deficiency); adding a Complex I inhibitor creates "
                "a combined OXPHOS block → severe, potentially fatal lactic acidosis. "
                "If glucose intolerance develops in long-term survivors, insulin is the only "
                "safe glucose-lowering agent."
            ),
        },
        {
            "drug": "Linezolid",
            "severity": "ABSOLUTE CI — ESPECIALLY CRITICAL IN TACO1",
            "mechanism": (
                "Linezolid inhibits the mitochondrial 23S rRNA-equivalent (mt-LSU 16S rRNA), "
                "blocking translation of ALL 13 mtDNA-encoded OXPHOS subunits by the "
                "mitochondrial ribosome, including MT-CO1. "
                "In TACO1 disease, MT-CO1 translation is ALREADY specifically reduced "
                "because TACO1 activates MT-CO1 mRNA translation. Linezolid eliminates "
                "even this residual MT-CO1 translation entirely — the combined effect "
                "(TACO1 deficiency + ribosome block) produces near-complete absence of "
                "COX1 protein and catastrophic Complex IV failure. "
                "This drug is MORE dangerous in TACO1 than in diseases where COX1 "
                "translation is intact but downstream assembly is blocked (COX10, COX15, SURF1). "
                "Use non-oxazolidinone antibiotics; chloramphenicol has the same mechanism "
                "and is equally CONTRAINDICATED."
            ),
        },
        {
            "drug": "Propofol",
            "severity": "CONTRAINDICATED (PRIS risk)",
            "mechanism": (
                "Propofol Infusion Syndrome (PRIS): propofol directly inhibits Complex IV "
                "(cytochrome c oxidase) at the cytochrome aa3 site and impairs fatty acid "
                "beta-oxidation. TACO1 patients have intrinsic COX deficiency; propofol "
                "compounds this. Unlike SCO2/COX15 patients (with HCM), TACO1 patients "
                "do not have major cardiac PRIS amplification, but the neurological and "
                "metabolic risk from propofol-induced COX inhibition remains significant. "
                "Use volatile anaesthetics (sevoflurane) or dexmedetomidine for sedation. "
                "If propofol is unavoidable (e.g. RSI), restrict to single induction dose; "
                "NEVER use propofol infusion."
            ),
        },
        {
            "drug": "Ketogenic Diet (KD)",
            "severity": "CONTRAINDICATED",
            "mechanism": (
                "High-fat beta-oxidation generates FADH2 that must be re-oxidised via "
                "Complex II → ubiquinone → Complex III → cytochrome c → Complex IV (COX). "
                "With COX deficient in TACO1 disease, fat oxidation leads to reducing-equivalent "
                "accumulation, elevated plasma lactate, and acylcarnitine build-up. "
                "While TACO1 patients have a milder COX defect than COX15/SCO2, the KD "
                "metabolic load can still precipitate lactic crisis in the TACO1 context. "
                "IV dextrose (GIR 6-8 mg/kg/min) is the preferred energy substrate. "
                "Modified Atkins diet has NOT been validated and should be avoided."
            ),
        },
        {
            "drug": "Chloramphenicol",
            "severity": "ABSOLUTE CI (same mechanism as linezolid)",
            "mechanism": (
                "Chloramphenicol is a mitochondrial ribosome inhibitor (peptidyl transferase "
                "centre of the mt-LSU) — same functional site as linezolid. It blocks "
                "mitochondrial translation of all 13 OXPHOS subunits, including MT-CO1. "
                "In TACO1 patients, this eliminates residual MT-CO1 translation and causes "
                "the same compounding catastrophic COX deficiency as linezolid. Chloramphenicol "
                "has historically been used for certain infections but must never be used in "
                "any OXPHOS disease, especially TACO1 where MT-CO1 translation is the "
                "rate-limiting step for Complex IV assembly."
            ),
        },
    ]

    return {
        "gene":              GENE,
        "protein":           PROTEIN,
        "disease":           DISEASE_NAME,
        "omim_gene":         OMIM_GENE,
        "omim_disease":      OMIM_DISEASE,
        "chromosome":        CHROMOSOME,
        "inheritance":       INHERITANCE,
        "onset":             ONSET,
        "cohort":            f"{COHORT_SIZE} patients · seed-{SEED} · TACO1 biallelic (Leigh Syndrome + COX Deficiency + Dysarthria)",
        "mechanism": (
            "TACO1 encodes a 343 aa mitochondrial matrix RNA-binding protein that specifically "
            "activates translation of MT-CO1 (COX subunit 1) mRNA by the mitochondrial "
            "ribosome. TACO1 associates with the mitochondrial ribosome small subunit (mt-SSU) "
            "and binds the 5' regulatory region of MT-CO1 mRNA to promote ribosome engagement "
            "and translational initiation. Without TACO1, MT-CO1 mRNA is present but poorly "
            "translated → reduced COX1 protein → impaired Complex IV (COX) assembly → isolated "
            "COX deficiency (20-35% of control; milder than COX15/SCO2 which often show <10-18%). "
            "Complexes I, II, III are NORMAL — same biochemical fingerprint as COX10, COX15, "
            "SURF1, SCO1, SCO2, but at a less severe COX deficiency level. The clinical "
            "consequence is LATER ONSET (childhood, median 3-8 years) and a MILDER but "
            "progressive course with DYSARTHRIA as the cardinal distinguishing feature. "
            "TACO1 is mechanistically distinct from: COX10/COX15 (heme a biosynthesis failure), "
            "SCO1/SCO2 (copper delivery to COX2 CuA), SURF1 (COX1 heme a3/CuB insertion "
            "scaffold), and LRPPRC (polyadenylation of ALL mt-mRNAs vs TACO1 MT-CO1 specific)."
        ),
        "dysarthria_note": (
            "DYSARTHRIA — CARDINAL DISTINGUISHING FEATURE OF TACO1 vs OTHER COX-DEFICIENCY DISEASES:\n"
            "Dysarthria (speech motor disorder) is present in ~85% of TACO1 patients and is "
            "often the FIRST or most prominent clinical presentation:\n"
            "  • Speech production is impaired: slurred, slow, effortful — consistent with "
            "    cerebellar + corticospinal involvement of the motor speech circuits\n"
            "  • Typically precedes or equals ataxia onset; patients often present to "
            "    speech/language pathologists before the mitochondrial diagnosis is made\n"
            "  • In the Weraarpachai 2009 family: both Inuit siblings presented with "
            "    progressive dysarthria + cerebellar ataxia + cognitive decline in childhood\n\n"
            "Comparison with other COX-deficiency diseases (dysarthria frequency):\n"
            "  TACO1:  ~85% (CARDINAL) — dysarthria-prominent Leigh\n"
            "  SURF1:  ~30-40% (present, less dominant — respiratory + Leigh features dominate)\n"
            "  SCO2:   ~20% (HCM/cardiac death typically precedes neurological evolution)\n"
            "  COX15:  ~25% (cardiac death typically first)\n"
            "  COX10:  ~35% (tubulopathy + Leigh dominate)\n\n"
            "Management of dysarthria in TACO1:\n"
            "  • Regular speech-language pathology (SLP) assessment from diagnosis\n"
            "  • Augmentative/alternative communication (AAC) planning early\n"
            "  • Dysphagia assessment — swallowing impaired in advanced disease → PEG timing\n"
            "  • Physiotherapy for ataxia + spasticity (co-existing motor impairments)"
        ),
        "kpis":               kpis,
        "feature_frequencies": feature_frequencies,
        "contraindications":  contraindications,
    }


# ── Breakdown (patients + feature frequencies) ─────────────────────────────
def get_breakdown() -> dict[str, Any]:
    rng      = _rng()
    patients = _generate_cohort(rng)

    def _pct2(key: str) -> int:
        return round(sum(1 for p in patients if p[key]) / COHORT_SIZE * 100)

    feat_freq = {
        "Cognitive Decline / Intellectual Disability (CARDINAL)":    _pct2("has_cognitive"),
        "Dysarthria — Speech Motor Disorder (CARDINAL DISTINGUISHING)": _pct2("has_dysarthria"),
        "Psychomotor Regression":                                     _pct2("has_regression"),
        "Progressive Ataxia":                                         _pct2("has_ataxia"),
        "Spasticity":                                                  _pct2("has_spasticity"),
        "Leigh / Leigh-like MRI":                                      _pct2("has_leigh_mri"),
        "Lactic Acidosis (moderate, ≥2.0 mmol/L)":                   round(sum(1 for p in patients if p["lactate"] >= 2.0) / COHORT_SIZE * 100),
        "Seizures":                                                    _pct2("has_seizures"),
        "Nystagmus":                                                   _pct2("has_nystagmus"),
        "Optic Atrophy":                                               _pct2("has_optic"),
        "Hypotonia":                                                   _pct2("has_hypotonia"),
        "Respiratory Compromise (LESS than SURF1)":                   _pct2("has_resp"),
        "HCM (RARE — KEY DDx SCO2 100% / COX15 78%)":                _pct2("has_hcm"),
        "Renal Tubulopathy (RARE — KEY DDx COX10 65%)":              _pct2("has_tubulopathy"),
        "NO Hepatopathy (KEY DDx SCO1/POLG/DGUOK)":                  100,
        "NO Iron Overload (KEY DDx GRACILE)":                         100,
        "Isolated COX Deficiency (CI/CII/CIII Normal)":               100,
        "Died":                                                        round(sum(1 for p in patients if p["outcome"].startswith("Died")) / COHORT_SIZE * 100),
    }

    return {
        "patients":            patients,
        "feature_frequencies": feat_freq,
    }


# ── Definitions ────────────────────────────────────────────────────────────
def get_definitions() -> dict[str, Any]:
    pharmacology = [
        {
            "term": "TACO1 — Mitochondrial MT-CO1 mRNA Translational Activator (343 aa, 17q23.3)",
            "definition": (
                "TACO1 (Translational Activator of Cytochrome c Oxidase I, OMIM *612958) encodes "
                "a 343 amino acid nuclear-encoded protein that localises to the mitochondrial matrix:\n\n"
                "Molecular function:\n"
                "  • RNA-binding protein — binds the 5' regulatory region of MT-CO1 mRNA\n"
                "  • Associates with the mitochondrial ribosome small subunit (mt-SSU, 28S)\n"
                "  • Promotes translational initiation: facilitates ribosome recruitment and "
                "    engagement with the MT-CO1 mRNA leader sequence\n"
                "  • Specificity: activates MT-CO1 translation; does NOT activate MT-CO2 or "
                "    MT-CO3 — target-specific (contrast LRPPRC which stabilises all mt-mRNAs)\n\n"
                "Structural features:\n"
                "  • Pentatricopeptide repeat (PPR)-like motifs — consistent with mt-RNA binding\n"
                "  • No signal peptide; imported into mitochondria via N-terminal MTS\n"
                "  • Yeast homologue: PET309 (highly conserved function — activates COX1 mRNA "
                "    translation in S. cerevisiae via the AAAAUAA element in the 5' UTR)\n\n"
                "Consequence of TACO1 loss:\n"
                "  • MT-CO1 mRNA remains present but untranslated / poorly translated\n"
                "  • COX1 (Complex IV core subunit) protein severely reduced\n"
                "  • COX1 is the nucleation point for Complex IV assembly — 13 subunits cannot "
                "    form holoenzyme without COX1\n"
                "  • Isolated Complex IV deficiency (20-35% of control in fibroblasts/muscle)\n"
                "  • Complexes I, II, III, V: NORMAL — same fingerprint as COX10, COX15, SURF1"
            ),
        },
        {
            "term": "TACO1 vs LRPPRC vs SURF1 — Three Distinct MT-CO1 Regulatory Mechanisms",
            "definition": (
                "Three mechanistically distinct COX-deficiency diseases affecting MT-CO1:\n\n"
                "1. TACO1 (17q23.3) — translational activation:\n"
                "   MT-CO1 mRNA present but poorly translated → ↓ COX1 protein → COX deficiency\n"
                "   Clinical: childhood Leigh + dysarthria + ataxia; MILDER; no HCM/tubulopathy\n"
                "   Biochemistry: ↓ COX1 protein; MT-CO1 mRNA level relatively preserved\n\n"
                "2. LRPPRC (2p21) — mRNA stability/polyadenylation:\n"
                "   LRPPRC stabilises polyadenylation of ALL mt-mRNAs (MT-CO1, CO2, CO3, ND1-6, etc.)\n"
                "   Loss → rapid degradation of multiple mt-mRNAs → combined OXPHOS defect\n"
                "   Clinical: French-Canadian Leigh (p.Ala354Val founder); liver + brain\n"
                "   Key DDx: LRPPRC affects MULTIPLE mt-mRNAs; TACO1 is MT-CO1 SPECIFIC\n\n"
                "3. SURF1 (9q34.2) — COX1 heme a3/CuB insertion scaffold:\n"
                "   SURF1 is an IMM protein that facilitates insertion of heme a3 and CuB "
                "   into newly synthesised COX1 — structural assembly function\n"
                "   MT-CO1 mRNA and COX1 protein are synthesised normally; COX1 maturation fails\n"
                "   Clinical: infantile Leigh + respiratory failure; most common COX-Leigh cause\n\n"
                "Algorithm:\n"
                "  COX deficiency + Leigh + CHILDHOOD onset + DYSARTHRIA → TACO1\n"
                "  COX deficiency + Leigh + INFANTILE + respiratory/HCM dominant → SURF1/SCO2\n"
                "  COX deficiency + Leigh + French-Canadian + liver → LRPPRC\n"
                "  COX deficiency + Leigh + tubulopathy → COX10"
            ),
        },
        {
            "term": "Linezolid ABSOLUTE CI in TACO1 — Compounding MT-CO1 Translation Block",
            "definition": (
                "Linezolid (oxazolidinone antibiotic) inhibits the mitochondrial peptidyl "
                "transferase centre (large ribosomal subunit, mt-LSU 16S rRNA equivalent), "
                "blocking ALL mitochondrial translation — including MT-CO1.\n\n"
                "Why linezolid is UNIQUELY DANGEROUS in TACO1:\n\n"
                "In diseases like COX10 or SURF1:\n"
                "  • MT-CO1 is translated at normal rates; COX10 or SURF1 protein is absent\n"
                "  • Linezolid blocks MT-CO1 translation → worsens COX deficiency\n"
                "  • BUT: residual MT-CO1 translation was still occurring before linezolid\n\n"
                "In TACO1 disease:\n"
                "  • TACO1 deficiency already severely reduces MT-CO1 translation rate\n"
                "  • Linezolid inhibits the mitochondrial ribosome directly\n"
                "  • Combined effect: TACO1 loss (↓ initiation) + linezolid (↓ elongation) "
                "    → near-COMPLETE elimination of MT-CO1 translation\n"
                "  • This causes a CATASTROPHICALLY worse COX deficiency than either "
                "    mechanism alone — potentially triggering acute decompensation\n\n"
                "Clinical implication:\n"
                "  • ANY antibiotic for infection in TACO1 patients: ALWAYS verify it is NOT "
                "    an oxazolidinone (linezolid, tedizolid) or chloramphenicol\n"
                "  • Alternative agents for MRSA: daptomycin, vancomycin, trimethoprim\n"
                "  • For Gram-negative serious infections: beta-lactams, aminoglycosides (with "
                "    nephrology input — aminoglycosides have their own mitochondrial risk "
                "    via 12S rRNA, but less severe than mt-LSU inhibition)"
            ),
        },
    ]

    gene_concepts = [
        {
            "term": "TACO1 Genotype–Phenotype — c.472insC Inuit Founder and Allelic Series",
            "definition": (
                "First human TACO1 mutations (Weraarpachai et al., 2009, Am J Hum Genet 85:751):\n\n"
                "c.472insC (p.Tyr158Leufs*) — frameshift, homozygous:\n"
                "  • Two Inuit siblings from northern Canada (consanguineous pedigree)\n"
                "  • Complete loss of TACO1 protein (frameshift → NMD)\n"
                "  • Phenotype: childhood onset (ages 3-4 years) with dysarthria, ataxia, "
                "    cognitive decline; Leigh-like MRI; isolated COX deficiency in fibroblasts\n"
                "  • One sibling: survived to late childhood/adolescence; progressive disability\n"
                "  • One sibling: died in second decade\n\n"
                "c.1A>C (p.Met1?) — start-codon disruption:\n"
                "  • No TACO1 protein produced; severe COX deficiency\n"
                "  • Reported in isolated case; more severe phenotype\n\n"
                "Missense variants (e.g. p.Ala167Val, c.500C>T):\n"
                "  • Partial TACO1 protein function retained → milder COX deficiency (~25-35%)\n"
                "  • Later onset, slower progression, better survival\n\n"
                "General genotype–phenotype rule:\n"
                "  • Biallelic null (frameshift/stop) → severe: COX deficiency <20%; "
                "    earlier onset, faster progression\n"
                "  • At least one missense with partial residual function → milder: "
                "    COX deficiency 25-35%; later onset; longer survival\n"
                "  • ALL TACO1 patients: AVOID VPA, metformin, linezolid, chloramphenicol, propofol, KD"
            ),
        },
        {
            "term": "TACO1 in the COX-Deficiency Disease Spectrum — mRNA Translation vs Heme vs Copper vs Scaffold",
            "definition": (
                "COX-deficiency diseases grouped by mechanism and clinical profile:\n\n"
                "1. MT-CO1 mRNA TRANSLATION:\n"
                "   TACO1 (17q23.3) — translational activator of MT-CO1 specifically\n"
                "   Onset: childhood (LATER); Dysarthria CARDINAL; Leigh; NO HCM/tubulo; MILDER\n\n"
                "2. HEME A BIOSYNTHESIS (COX1 prosthetic groups):\n"
                "   COX10 (17p12) — protoheme IX farnesyltransferase; heme o Step 1\n"
                "   COX15 (10q24.2) — heme o oxidase / heme a synthase; heme a Step 2\n"
                "   Both: infantile Leigh + isolated COX deficiency\n"
                "   DDx: COX10 → tubulopathy 65%; COX15 → HCM 78%\n\n"
                "3. CUA COPPER DELIVERY (COX2 copper site):\n"
                "   SCO2 (22q13.33) — HCM 100% CARDINAL; infantile fatal\n"
                "   SCO1 (17p13.2) — hepatopathy 100% CARDINAL; neonatal fatal\n"
                "   COX17 (3q13.33) — upstream Cu donor [no confirmed human disease so far]\n\n"
                "4. COX1 SCAFFOLD / ASSEMBLY:\n"
                "   SURF1 (9q34.2) — COX1 heme a3/CuB insertion; Leigh 95%; respiratory 75%\n\n"
                "5. MT-mRNA STABILITY (all mt-mRNAs):\n"
                "   LRPPRC (2p21) — French-Canadian; combined COX + CI defect; liver + Leigh\n\n"
                "Clinical DDx summary (isolated COX + Leigh):\n"
                "  CHILDHOOD onset + Dysarthria → TACO1\n"
                "  Infantile + HCM 100%          → SCO2\n"
                "  Infantile + HCM 78%           → COX15\n"
                "  Infantile + Hepatopathy 100%  → SCO1\n"
                "  Infantile + Tubulopathy 65%   → COX10\n"
                "  Infantile + Leigh dominant     → SURF1"
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "Leigh Syndrome in TACO1 — Childhood-Onset Milder Bilateral Basal Ganglia Disease",
            "definition": (
                "Leigh syndrome (OMIM #256000) in TACO1 shares the radiological definition "
                "(bilateral symmetric T2/FLAIR signal in basal ganglia/brainstem) but differs "
                "from the infantile-lethal Leigh pattern in SURF1/SCO2:\n\n"
                "TACO1 Leigh characteristics:\n"
                "  • MRI: bilateral putamen/caudate/brainstem T2 hyperintensities in ~75%\n"
                "  • Onset typically AFTER first ambulation — motor/speech regression noted\n"
                "  • Metabolic crises less frequent and less severe than SURF1/SCO2\n"
                "  • CSF lactate elevated in majority but often <5 mmol/L (vs >10 in SURF1 crisis)\n"
                "  • Neurological progression is SLOW but relentless over years–decades\n\n"
                "Crisis triggers in TACO1:\n"
                "  • Febrile illness — same as all OXPHOS diseases; increases metabolic demand\n"
                "  • Fasting — restricts glucose → forces FA oxidation → complex IV demand\n"
                "  • Anaesthesia/surgery — propofol CONTRAINDICATED; use sevoflurane\n"
                "  • Sleep deprivation / physiological stress\n\n"
                "Emergency management (same principles as all Leigh):\n"
                "  1. IV dextrose GIR 6-8 mg/kg/min STAT\n"
                "  2. NaHCO3 if pH <7.2\n"
                "  3. LEV if seizures (ABSOLUTE CI VPA)\n"
                "  4. Manage temperature aggressively (fever is a metabolic stressor)\n"
                "  5. NEVER use linezolid or chloramphenicol for infections\n"
                "  6. NEVER fast — maintain continuous enteral/parenteral glucose"
            ),
        },
        {
            "term": "Dysarthria + Ataxia Management in TACO1 — Speech, Physio, AAC Planning",
            "definition": (
                "Management priorities for the cardinal neurological features of TACO1:\n\n"
                "DYSARTHRIA (speech motor disorder, ~85%):\n"
                "  Speech-Language Pathology (SLP) assessment:\n"
                "  • At diagnosis: baseline speech intelligibility assessment\n"
                "  • 6-12 monthly follow-up — track progressive deterioration\n"
                "  • Dysphagia screen: swallowing safety critical as disease progresses\n"
                "  Augmentative/Alternative Communication (AAC):\n"
                "  • Introduce AAC strategies EARLY — when speech is still >50% intelligible\n"
                "  • Low-tech (alphabet board) → high-tech (eye-gaze device) progression\n"
                "  • Family/school training on AAC strategies\n"
                "  PEG feeding (percutaneous gastrostomy):\n"
                "  • Consider when oral intake insufficient OR aspiration risk high\n"
                "  • Anaesthesia risk: use sevoflurane NOT propofol\n\n"
                "ATAXIA (~80%) + SPASTICITY (~70%):\n"
                "  Physiotherapy:\n"
                "  • Regular PT from diagnosis; maintain ambulation as long as possible\n"
                "  • Gait aids (walker, AFO orthoses) for cerebellar ataxia + spastic legs\n"
                "  • Fall prevention: environmental modifications, physiotherapy-guided\n"
                "  Spasticity management:\n"
                "  • Baclofen (GABA-B agonist): NOT mitochondrially toxic; first-line\n"
                "  • Tizanidine: consider if baclofen insufficient\n"
                "  • AVOID dantrolene in OXPHOS disease (interferes with muscle Ca2+ handling)\n\n"
                "COGNITIVE DECLINE (~90%):\n"
                "  • Neuropsychological assessment at diagnosis and 2-yearly\n"
                "  • Educational support (IEP/504 plan)\n"
                "  • Occupational therapy for adaptive skills\n"
                "  • Antidepressants if mood disorder develops — SSRIs preferred over TCAs "
                "    (TCAs have cardiac effects; SSRIs safe in mito disease)"
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "LEV (Levetiracetam) — Preferred AED in TACO1 Disease",
            "definition": (
                "Levetiracetam (LEV) is the AED of choice in TACO1 disease:\n\n"
                "  1. RENAL excretion: 66% excreted unchanged; no hepatic CYP metabolism\n"
                "     → No hepatotoxic metabolites; no CYP induction; no drug-drug interactions "
                "       with mitochondrial cofactors or co-medications\n"
                "  2. No mitochondrial toxicity: LEV does not inhibit any respiratory complex\n"
                "  3. Cardiac-safe: no effect on conduction or contractility\n"
                "  4. IV formulation: essential during Leigh crisis\n"
                "  5. Broad-spectrum: myoclonic + focal + generalised seizures\n\n"
                "Dosing: 20-60 mg/kg/day in 2 divided doses; IV loading 20-40 mg/kg over 15 min.\n\n"
                "AVOID:\n"
                "  • VPA (ABSOLUTE CI — CoA sequestration, POLG, hepatotoxicity)\n"
                "  • Phenobarbital (Complex I inhibitor — worsens OXPHOS imbalance)\n"
                "  • Carbamazepine (NOT a direct mito toxin but induces CYP3A4 → alters "
                "    levels of cofactors/drugs; cardiac conduction effects)\n"
                "  • Phenytoin (cardiac Class IB; hepatic; mito studies mixed — avoid)\n\n"
                "Second-line AED options (if LEV insufficient):\n"
                "  • Clobazam (benzodiazepine — safe in mito disease)\n"
                "  • Lamotrigine (limited mito data; generally considered acceptable)\n"
                "  • Topiramate (some carbonic anhydrase inhibition — may help acidosis "
                "    management; monitor renal stones; avoid in severe lactic acidosis)"
            ),
        },
        {
            "term": "CoQ10/Ubiquinol, Riboflavin, Thiamine, Biotin, Succinate — Cofactor Therapy in TACO1",
            "definition": (
                "Standard mitochondrial cofactor therapy in TACO1 (all Level C evidence):\n\n"
                "  CoQ10 / Ubiquinol:\n"
                "  • 300-600 mg/day adults; 10-30 mg/kg/day children\n"
                "  • Mobile electron carrier Complex I/II → III; may boost spare respiratory capacity\n"
                "  • Ubiquinol (reduced form) preferred — better oral bioavailability\n\n"
                "  Riboflavin (B2): 100-400 mg/day\n"
                "  • FMN/FAD cofactor for Complex I and Complex II; supports residual OXPHOS\n\n"
                "  Thiamine (B1): 100-300 mg/day — MANDATORY empirically in ALL Leigh\n"
                "  • PDH and alpha-KGDH cofactor; SLC19A3 (THTR2) deficiency is TREATABLE Leigh mimic\n"
                "  • Empiric thiamine until molecular diagnosis confirmed\n\n"
                "  Biotin: 5-20 mg/day — MANDATORY empirically\n"
                "  • Biotinidase deficiency (BTD) is TREATABLE Leigh mimic; give empirically\n\n"
                "  Succinate: 2-6 g/day\n"
                "  • Anaplerotic TCA cycle intermediate; bypasses Complex I → enters at Complex II\n"
                "  • Particularly useful in COX deficiency to maintain TCA cycle flux\n"
                "  • Oral succinate (e.g. sodium succinate); absorption variable\n\n"
                "  Carnitine: 50-100 mg/kg/day\n"
                "  • Secondary carnitine deficiency common in OXPHOS diseases\n\n"
                "  All cofactors discontinued if no benefit after 3-6 month trial; "
                "  never delay therapy for treatable mimics (SLC19A3, BTD)."
            ),
        },
        {
            "term": "IV Dextrose GIR 6-8 + Speech-Feeding Safety — Nutritional Management in TACO1",
            "definition": (
                "Nutritional management has unique complexity in TACO1 due to dysarthria/dysphagia:\n\n"
                "Energy substrate — crisis:\n"
                "  • IV Dextrose GIR 6-8 mg/kg/min during metabolic crisis\n"
                "  • NEVER FAST — maintain continuous enteral/parenteral glucose\n"
                "  • Target blood glucose 4-8 mmol/L; plasma lactate <5 mmol/L during stabilisation\n\n"
                "Dysarthria → dysphagia evolution:\n"
                "  • TACO1 patients often develop progressive dysphagia as disease advances\n"
                "  • Modified texture diet: pureed or minced based on IDDSI levels\n"
                "  • Thickened fluids: as per SLP assessment (IDDSI 1-4)\n"
                "  • Videofluoroscopic swallowing study (VFSS) for objective aspiration assessment\n\n"
                "PEG (percutaneous endoscopic gastrostomy) timing:\n"
                "  • Consider when: >10% weight loss, oral intake <50% needs, recurrent aspirations\n"
                "  • Anaesthesia for PEG: use sevoflurane; NEVER propofol infusion\n"
                "  • Post-PEG: continuous overnight feeds preferred (avoid fasting periods)\n\n"
                "Ongoing nutrition:\n"
                "  • High-carbohydrate, moderate-protein, low-fat macro distribution\n"
                "  • AVOID very high fat content (KD CONTRAINDICATED)\n"
                "  • Dietitian monitoring every 3-6 months; growth tracking in children\n"
                "  • Supplement: CoQ10, riboflavin, thiamine, biotin, carnitine, succinate"
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }
