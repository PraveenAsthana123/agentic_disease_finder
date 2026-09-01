#!/usr/bin/env python3
"""NDUFS2 — Leigh Syndrome Isolated Complex I Deficiency (Q-Module 49 kDa PSST Subunit / N2 Fe-S).

NDUFS2 (NADH:Ubiquinone Oxidoreductase Core Subunit S2), also known as the PSST subunit
(named for the plant homolog), is the 49 kDa core subunit of the Q-module of Complex I.
NDUFS2 harbours the N2 iron-sulfur cluster ([4Fe-4S]) — the TERMINAL and ONLY Fe-S cluster
in the Q-module, and the final electron carrier in the Fe-S relay chain before ubiquinone.
Biallelic loss-of-function variants abolish CI NADH oxidation → isolated CI deficiency →
Leigh / Leigh-like syndrome.

  NDUFS2 gene      OMIM *602985
  Disease          Leigh Syndrome (OMIM #256000) /
                   Mitochondrial Complex I Deficiency, Nuclear Type (NDUFS2)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       1q23.3

PATHOPHYSIOLOGY (Complex I / Q-module / NDUFS2 role):
  The Fe-S relay of Complex I transfers electrons from NADH to ubiquinone in a
  sequential chain through iron-sulfur clusters across both the N-module and Q-module:

    NADH → FMN/N3 (NDUFV1/51kDa) → N1a (NDUFV2/24kDa) → N1b/N4/N5 (NDUFS1/75kDa)
         → N6a/N6b (NDUFB9/NDUFS8) → N2 (NDUFS2/49kDa) → ubiquinone

  NDUFS2 and the N2 cluster occupy the TERMINAL position in this relay:
    - N2 is the ONLY Fe-S cluster in the Q-module
    - N2 is the last electron carrier before ubiquinone reduction
    - N2 sits at the mouth of the ubiquinone-binding channel in the Q-module
    - Without NDUFS2/N2: electrons cannot reach ubiquinone even if all upstream
      Fe-S clusters (N1b, N4, N5, N6a, N6b) are intact → CI deficiency

  Biochemical signature (IDENTICAL to NDUFS4/NDUFV1/NDUFS1 CI fingerprint):
    Complex I: 5–20% of control (severely reduced) — ISOLATED CI DEFICIENCY
    Complex II (SDHA): NORMAL (fully nuclear-encoded; NDUFS2 not required)
    Complex III: NORMAL
    Complex IV (COX): NORMAL — KEY DDx SURF1/SCO2/COX10/COX15 (CIV-only deficiency)

DISTINGUISHING FEATURES vs NDUFS4/NDUFV1/NDUFS1:
  NO Peripheral Neuropathy (vs NDUFS1 ~50% — IMPORTANT DDx point)
    • NDUFS2-CI-Leigh does NOT show the peripheral neuropathy that characterises NDUFS1
    • Absence of neuropathy helps narrow the CI-Leigh gene panel
  NO Olfactory Bulb Lesions (vs NDUFS4 ~52–65% — KEY DDx)
    • Olfactory bulb MRI hyperintensity on T2 is pathognomonic of NDUFS4 — not seen in NDUFS2
  NO Leukodystrophy (vs NDUFV1 ~40–50% — KEY DDx)
    • White matter T2 hyperintensity distinguishes NDUFV1 — not seen in NDUFS2
  HCM: ~8% (similar to NDUFS4/NDUFV1; much less than SCO2 100%)
  TERMINAL Fe-S Position (N2 = ubiquinone gate):
    • NDUFS2 loss eliminates the final Fe-S step → no electron entry to ubiquinone
    • Energetically, this is equivalent to NDUFS1/NDUFV1/NDUFS4 — complete CI deficiency

FOUNDER / RECURRENT MUTATIONS:
  p.Arg333Gln   c.998G>A  — recurrent; consanguineous Mediterranean/Middle-Eastern families;
                             first reported by Loeffen 2001; moderate-to-severe Leigh
  p.Glu59Lys    c.175G>A  — compound het with truncating allele; severe infantile Leigh
  p.Ala341Val   c.1022C>T — missense; partial N2 stability; intermediate severity
  Frameshift / nonsense (compound het) — complete CI abolition; severe Leigh; early death (<2 yr)

THERAPY — NDUFS2/CI-LEIGH SPECIFICS:
  No targeted N2 Fe-S cluster repair is clinically available.
  Management follows the CI-Leigh supportive protocol (cofactors + avoid CI toxins).
  Succinate bypass (Level C): CII-mediated electron entry to ubiquinol bypasses blocked NDUFS2/N2.
  Riboflavin B2 (Level C): FMN reinforces NDUFV1/N3 upstream; less direct than NDUFV1 deficiency.
  Physiotherapy: no peripheral neuropathy in NDUFS2 — focuses on central ataxia/dystonia.

References:
  Loeffen JL et al. Ann Neurol. 2001;49(2):195-201.
    (First NDUFS2 mutations in human CI deficiency — p.Arg333Gln identified)
  Bénit P et al. Hum Genet. 2003;113(1):32-37.
    (Additional NDUFS2 compound hets; genotype-phenotype)
  Fassone E, Rahman S. J Med Genet. 2012;49(9):578-590.
    (CI deficiency genetics review — NDUFS2 in context)
  Rötig A, Munnich A. Hum Mutat. 2003;21(6):607-614.
    (CI deficiency spectrum including NDUFS2)
  Schuelke M et al. J Inherit Metab Dis. 1999;22(2):175-183.
    (CI-Leigh genetic series)
  Fernandez-Vizarra E, Zeviani M. Hum Mol Genet. 2021;30(R2):R181-R197.
    (CI assembly and NDUFS2/N2 structural context)
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 613
DISEASE_ID   = "ndufs2"
DISEASE_NAME = (
    "NDUFS2 Leigh Syndrome — Isolated Complex I Deficiency "
    "(CI-Leigh / NDUFS2 Q-Module 49 kDa PSST Subunit / N2 Terminal Fe-S)"
)
GENE         = "NDUFS2"
PROTEIN      = (
    "NDUFS2 — 463 aa precursor / ~431 aa mature, Q-module PSST subunit (49 kDa), "
    "harbours N2 [4Fe-4S] — TERMINAL Fe-S cluster, final electron carrier to ubiquinone"
)
OMIM_GENE    = "*602985"
OMIM_DISEASE = "#256000 (Leigh Syndrome) / Mitochondrial Complex I Deficiency Nuclear Type (NDUFS2)"
CHROMOSOME   = "1q23.3"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = (
    "Infantile (3–18 months); rarely neonatal; delayed onset up to 2–4 yr in partial-function missense alleles"
)
COHORT_SIZE  = 40
COLOR        = "#01579b"   # dark blue — Q-module / N2 / ubiquinone-binding theme
LIGHT        = "#e3f2fd"

# Genotype pool
GENO_RGLQ  = "p.Arg333Gln (c.998G>A) / truncating (compound het) — recurrent severe; Mediterranean/Middle-Eastern"
GENO_ELYS  = "p.Glu59Lys (c.175G>A) / truncating (compound het) — severe infantile; multi-ethnic"
GENO_AVAL  = "p.Ala341Val (c.1022C>T) / missense (compound het) — intermediate; partial N2 stability"
GENO_NULL  = "Frameshift / nonsense (compound het) — complete CI abolition; severe Leigh; early death"
GENO_MIS2  = "Missense + splice / compound het — variable severity; N2 cluster instability"

GENO_POOL    = [GENO_RGLQ, GENO_ELYS, GENO_AVAL, GENO_NULL, GENO_MIS2]
GENO_WEIGHTS = [0.30,      0.20,      0.18,       0.17,      0.15]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG — reproducible 40-patient NDUFS2/CI-Leigh cohort (seed-613)."""
    return random.Random(SEED)


# ── Patient cohort ────────────────────────────────────────────────────────────
_TX_POOL = [
    "IV Dextrose GIR 6-8 (metabolic crisis — first-line)",
    "Riboflavin B2 (100–400 mg/day — CI-specific FMN precursor; upstream of NDUFS2)",
    "CoQ10/Ubiquinol (300–600 mg/day)",
    "Thiamine B1 (mandatory empiric, all Leigh until SLC19A3/BTD excluded)",
    "Biotin (mandatory empiric, all Leigh until BTD ruled out)",
    "Succinate (IV/oral — Complex II bypass; bypasses blocked NDUFS2/N2 entirely)",
    "NaHCO3 (lactic acidosis correction, pH <7.20)",
    "LEV (seizures — preferred AED, renal excretion, no mito toxicity)",
    "NIV/BiPAP (central respiratory compromise)",
    "Carnitine (secondary deficiency, 50–100 mg/kg/day)",
    "Physiotherapy / occupational therapy (central ataxia + dystonia — no peripheral neuropathy)",
    "Enteral feeds / NG (avoid fasting; high-carbohydrate; KD strictly CI)",
]

_OUTCOMES = [
    "Alive — stable moderate-to-severe Leigh; cofactor + respiratory support",
    "Alive — prolonged survival with missense alleles; partial CI residual; school-age",
    "Alive — severe Leigh; dependent care; ongoing multidisciplinary support",
    "Died — acute lactic crisis + respiratory failure in first 18 months (null alleles)",
    "Died — progressive brainstem Leigh + respiratory failure; 2–4 yr trajectory",
]
_OUT_WEIGHTS = [0.27, 0.15, 0.20, 0.22, 0.16]


def _generate_cohort(rng: random.Random) -> list[dict[str, Any]]:
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno        = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex         = rng.choice(["M", "F"])
        onset_mo    = rng.choices(
            [3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 24, 30, 36, 48],
            weights=[3, 5, 8, 10, 11, 10, 9, 9, 8, 7, 6, 5, 4, 5],
        )[0]
        onset_yr    = round(onset_mo / 12, 1)
        lactate_mm  = round(rng.uniform(3.0, 15.5), 1)
        ci_pct      = rng.randint(5, 21)                   # Complex I (% of control)
        cii_pct     = rng.randint(86, 118)                 # Complex II — NORMAL
        civ_pct     = rng.randint(82, 113)                 # Complex IV — NORMAL

        has_leigh_mri       = rng.random() < 0.80
        has_regression      = rng.random() < 0.95
        has_hypotonia       = rng.random() < 0.85
        has_lactic          = rng.random() < 0.88
        has_resp            = rng.random() < 0.58
        has_seizures        = rng.random() < 0.45
        has_ataxia          = rng.random() < 0.42
        has_dystonia        = rng.random() < 0.38
        has_optic           = rng.random() < 0.28
        has_myoclonus       = rng.random() < 0.20
        has_nystagmus       = rng.random() < 0.25
        has_spasticity      = rng.random() < 0.35
        has_hcm             = rng.random() < 0.08    # similar to NDUFS4/NDUFV1; much less than SCO2
        has_hepatopathy     = rng.random() < 0.04    # RARE — KEY DDx POLG/DGUOK
        has_iron            = False                   # NO iron overload — KEY DDx GRACILE
        has_tubulopathy     = rng.random() < 0.03    # RARE — KEY DDx COX10 (65%)
        has_olfactory       = False                   # NO olfactory bulb lesions — KEY DDx NDUFS4
        has_leukodystrophy  = rng.random() < 0.04    # RARE — KEY DDx NDUFV1 (40-50%)
        has_neuropathy      = False                   # NO peripheral neuropathy — KEY DDx NDUFS1 (50%)

        feat_list = [
            "Isolated Complex I deficiency (CII, CIII, CIV — NORMAL)",
            "Leigh / Leigh-like MRI (bilateral putamen + brainstem)",
        ]
        if has_regression:     feat_list.append("Psychomotor regression/arrest")
        if has_hypotonia:      feat_list.append("Hypotonia")
        if has_lactic:         feat_list.append("Lactic acidosis")
        if has_resp:           feat_list.append("Respiratory compromise (central)")
        if has_seizures:       feat_list.append("Seizures")
        if has_ataxia:         feat_list.append("Ataxia")
        if has_dystonia:       feat_list.append("Dystonia")
        if has_myoclonus:      feat_list.append("Myoclonus")
        if has_nystagmus:      feat_list.append("Nystagmus")
        if has_optic:          feat_list.append("Optic atrophy")
        if has_spasticity:     feat_list.append("Spasticity")
        if has_hcm:            feat_list.append("HCM (RARE — KEY DDx SCO2 100%)")
        if has_hepatopathy:    feat_list.append("Hepatopathy (RARE — KEY DDx POLG/DGUOK)")

        txs     = rng.sample(_TX_POOL, k=rng.randint(4, 7))
        outcome = rng.choices(_OUTCOMES, weights=_OUT_WEIGHTS)[0]

        patients.append({
            "id":                  f"NDUFS2-{i:03d}",
            "geno":                geno,
            "sex":                 sex,
            "onset_yr":            onset_yr,
            "lactate_mm":          lactate_mm,
            "ci_pct":              ci_pct,
            "cii_pct":             cii_pct,
            "civ_pct":             civ_pct,
            "has_leigh_mri":       has_leigh_mri,
            "has_leukodystrophy":  has_leukodystrophy,
            "has_regression":      has_regression,
            "has_hypotonia":       has_hypotonia,
            "has_lactic":          has_lactic,
            "has_neuropathy":      has_neuropathy,
            "has_resp":            has_resp,
            "has_seizures":        has_seizures,
            "has_myoclonus":       has_myoclonus,
            "has_ataxia":          has_ataxia,
            "has_optic":           has_optic,
            "has_nystagmus":       has_nystagmus,
            "has_dystonia":        has_dystonia,
            "has_spasticity":      has_spasticity,
            "has_hcm":             has_hcm,
            "has_hepatopathy":     has_hepatopathy,
            "has_iron":            has_iron,
            "has_tubulopathy":     has_tubulopathy,
            "has_olfactory":       has_olfactory,
            "features":            ", ".join(feat_list[:7]),
            "treatments":          ", ".join(txs[:5]),
            "outcome":             outcome,
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
        "Psychomotor Regression / Arrest (near-universal in Leigh)":             _pct("has_regression"),
        "Leigh / Leigh-like MRI (bilateral putamen + brainstem)":                _pct("has_leigh_mri"),
        "Isolated Complex I Deficiency (CII, CIII, CIV Normal — 100%)":         100,
        "Hypotonia":                                                             _pct("has_hypotonia"),
        "Lactic Acidosis (elevated baseline + crisis)":                          _pct("has_lactic"),
        "Respiratory Compromise (central)":                                      _pct("has_resp"),
        "Seizures":                                                              _pct("has_seizures"),
        "Ataxia":                                                                _pct("has_ataxia"),
        "Dystonia":                                                              _pct("has_dystonia"),
        "Myoclonus":                                                             _pct("has_myoclonus"),
        "Nystagmus":                                                             _pct("has_nystagmus"),
        "Optic Atrophy":                                                         _pct("has_optic"),
        "Spasticity":                                                            _pct("has_spasticity"),
        "HCM (~8% — similar to NDUFS4/NDUFV1; much less than SCO2 100%)":       _pct("has_hcm"),
        "Hepatopathy (RARE — KEY DDx POLG / DGUOK)":                            _pct("has_hepatopathy"),
        "NO Peripheral Neuropathy (KEY DDx NDUFS1 ~50% neuropathy)":            100,
        "NO Olfactory Bulb Lesions (KEY DDx NDUFS4 ~52–65%)":                   100,
        "NO Leukodystrophy (KEY DDx NDUFV1 ~40–50%)":                           100,
        "NO Iron Overload (KEY DDx GRACILE — 100% iron overload)":              100,
        "NO COX Deficiency (KEY DDx SURF1 / SCO2 / COX10 / COX15)":            100,
        "Alive (with support)":                                                  round(alive / COHORT_SIZE * 100),
    }

    kpis = [
        {"label": "Cohort (n)",      "value": COHORT_SIZE,                                                                        "color": COLOR},
        {"label": "Leigh MRI",       "value": f"{feature_frequencies['Leigh / Leigh-like MRI (bilateral putamen + brainstem)']}%", "color": "#4a148c"},
        {"label": "No Neuropathy",   "value": "100% — DDx NDUFS1",                                                                "color": "#2e7d32"},
        {"label": "Resp Compromise", "value": f"{feature_frequencies['Respiratory Compromise (central)']}%",                      "color": "#b71c1c"},
        {"label": "Hypotonia",       "value": f"{feature_frequencies['Hypotonia']}%",                                             "color": COLOR},
        {"label": "CI Activity",     "value": "5–21% control",                                                                    "color": "#c62828"},
        {"label": "Fatal",           "value": f"{round(died/COHORT_SIZE*100)}%",                                                  "color": "#b71c1c"},
        {"label": "Seed",            "value": f"#{SEED}",                                                                        "color": "#455a64"},
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "severity": "ABSOLUTE CI — DANGEROUS IN ALL LEIGH / CI DEFICIENCY",
            "mechanism": (
                "Triple mechanism in NDUFS2/CI-Leigh:\n"
                "1. CoA SEQUESTRATION: valproyl-CoA depletes the mitochondrial CoA pool "
                "   → TCA cycle collapses → OXPHOS substrate loss. In NDUFS2 deficiency "
                "   (CI already at 5–20%), CoA depletion tips the patient into irreversible "
                "   lactic crisis.\n"
                "2. POLG INHIBITION: VPA inhibits mitochondrial DNA polymerase gamma → "
                "   reduces mtDNA copy number → fewer MT-ND1–6 / MT-ND4L templates → "
                "   fewer mtDNA-encoded CI P-module subunits → CI assembly further impaired.\n"
                "3. HEPATOTOXICITY: direct hepatotoxic risk; hepatic OXPHOS failure → "
                "   impaired gluconeogenesis → hypoglycaemia → lactic crisis.\n"
                "Use LEV (levetiracetam) as first-line AED: renal excretion, no mito toxicity."
            ),
        },
        {
            "drug": "Metformin",
            "severity": "ABSOLUTE CI — DIRECT COMPLEX I INHIBITOR: CATASTROPHIC IN NDUFS2",
            "mechanism": (
                "Metformin inhibits Complex I (NADH dehydrogenase) at the ND1/quinone "
                "binding site adjacent to NDUFS2/N2 → blocks NADH → ubiquinone electron transfer.\n"
                "In NDUFS2 deficiency: CI is the primary disease locus (5–20% activity). "
                "Metformin's CI inhibition further suppresses this residual capacity → "
                "near-total CI shutdown → massive lactate accumulation (>15–20 mmol/L).\n\n"
                "The NDUFS2/N2 site and the ND1 metformin-binding site are in the same "
                "quinone-binding cavity — compounded pharmacodynamic disaster.\n"
                "Alternative for glucose management: insulin only. Never metformin."
            ),
        },
        {
            "drug": "Linezolid",
            "severity": "ABSOLUTE CI — mt-Ribosome 23S rRNA Block (ALL 7 ND mtDNA Subunits)",
            "mechanism": (
                "Linezolid inhibits the mitochondrial large ribosomal subunit (mt-LSU) 23S rRNA "
                "→ blocks synthesis of ALL 13 mitochondrially-encoded OXPHOS subunits.\n"
                "In NDUFS2/CI-Leigh:\n"
                "  • NDUFS2 itself is nuclear-encoded — BUT Complex I also requires 7 mtDNA-encoded "
                "    ND subunits (MT-ND1–6, MT-ND4L) for the P-module membrane arm\n"
                "  • Linezolid blocks synthesis of ALL 7 → CI P-module cannot assemble "
                "    → Q-module (NDUFS2/N2 terminal Fe-S) has no functional context → CI → zero\n"
                "  • A patient with 5–20% residual CI drops to near-zero: potentially fatal\n"
                "Alternatives: vancomycin IV, daptomycin, tigecycline (all ribosome-safe).\n"
                "Chloramphenicol has the SAME mt-ribosome mechanism → equally ABSOLUTE CI."
            ),
        },
        {
            "drug": "Ketogenic Diet (KD)",
            "severity": "CONTRAINDICATED in isolated CI deficiency",
            "mechanism": (
                "KD forces beta-oxidation of fatty acids as primary fuel. Beta-oxidation "
                "and the TCA cycle generate NADH that MUST be re-oxidised by Complex I.\n"
                "In NDUFS2 CI-Leigh: CI cannot re-oxidise NADH (5–20% activity; N2/ubiquinone "
                "electron transfer is blocked). Forcing KD → NADH accumulation → NAD+ "
                "depletion → TCA + beta-oxidation stall → worsened lactic acidosis.\n\n"
                "KD is beneficial in GLUT1-DS and PDHD — CONTRAINDICATED in CI deficiency."
            ),
        },
        {
            "drug": "Propofol",
            "severity": "AVOID / HIGH CAUTION — PRIS in CI-Leigh",
            "mechanism": (
                "Propofol Infusion Syndrome (PRIS): propofol inhibits Complex IV and "
                "uncouples fatty acid beta-oxidation.\n"
                "In NDUFS2/CI-Leigh: CI is the primary ETC bottleneck. Propofol's CIV "
                "inhibition creates a SECOND downstream ETC block → electrons trapped "
                "between CI (NDUFS2/N2 → ubiquinone exit point) and CIV → ROS burst.\n"
                "Use sevoflurane (volatile, no PRIS) or dexmedetomidine as alternatives."
            ),
        },
        {
            "drug": "Phenobarbital",
            "severity": "HIGH CAUTION — Secondary Complex I Inhibitor",
            "mechanism": (
                "Phenobarbital inhibits CI electron transport (quinone channel region). "
                "In NDUFS2/CI-Leigh: residual CI (5–20%) is critical. Phenobarbital's "
                "CI inhibition reduces this residual further → lactic decompensation risk.\n"
                "Use LEV (preferred) or clonazepam/CLB. "
                "Lowest effective dose + close lactate monitoring if unavoidable."
            ),
        },
        {
            "drug": "Chloramphenicol",
            "severity": "ABSOLUTE CI — Same Mechanism as Linezolid",
            "mechanism": (
                "Chloramphenicol binds the mt-LSU 23S rRNA and blocks mitochondrial "
                "ribosome peptidyl-transferase → identical mechanism to linezolid.\n"
                "Blocks synthesis of all 7 mtDNA-encoded ND subunits (P-module) → "
                "CI assembly collapses → catastrophic in NDUFS2/CI deficiency."
            ),
        },
    ]

    return {
        "gene":                 GENE,
        "protein":              PROTEIN,
        "disease":              DISEASE_NAME,
        "omim_gene":            OMIM_GENE,
        "omim_disease":         OMIM_DISEASE,
        "chromosome":           CHROMOSOME,
        "inheritance":          INHERITANCE,
        "onset":                ONSET,
        "cohort_size":          COHORT_SIZE,
        "color":                COLOR,
        "feature_frequencies":  feature_frequencies,
        "kpis":                 kpis,
        "contraindications":    contraindications,
    }


# ── Breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown() -> dict[str, Any]:
    rng      = _rng()
    patients = _generate_cohort(rng)

    def _pct(key: str) -> int:
        return round(sum(1 for p in patients if p[key]) / COHORT_SIZE * 100)

    feature_frequencies = {
        "Psychomotor Regression":                                         _pct("has_regression"),
        "Leigh / Leigh-like MRI":                                         _pct("has_leigh_mri"),
        "Hypotonia":                                                      _pct("has_hypotonia"),
        "Lactic Acidosis":                                                _pct("has_lactic"),
        "Respiratory Compromise":                                         _pct("has_resp"),
        "Seizures":                                                       _pct("has_seizures"),
        "Ataxia":                                                         _pct("has_ataxia"),
        "Dystonia":                                                       _pct("has_dystonia"),
        "Myoclonus":                                                      _pct("has_myoclonus"),
        "Nystagmus":                                                      _pct("has_nystagmus"),
        "Optic Atrophy":                                                  _pct("has_optic"),
        "Spasticity":                                                     _pct("has_spasticity"),
        "HCM (RARE — ~8%, similar to NDUFS4/NDUFV1)":                   _pct("has_hcm"),
        "Hepatopathy (RARE)":                                            _pct("has_hepatopathy"),
        "Peripheral Neuropathy (NEVER — KEY DDx NDUFS1 ~50%)":           0,
        "Olfactory Bulb MRI (NEVER — KEY DDx NDUFS4 52–65%)":           0,
        "Leukodystrophy / White Matter (NEVER — KEY DDx NDUFV1 40–50%)": 0,
    }

    genotype_distribution = {}
    for p in patients:
        g = p["geno"].split(" / ")[0]
        genotype_distribution[g] = genotype_distribution.get(g, 0) + 1

    ci_values  = [p["ci_pct"]  for p in patients]
    cii_values = [p["cii_pct"] for p in patients]
    civ_values = [p["civ_pct"] for p in patients]

    return {
        "patients":              patients,
        "feature_frequencies":   feature_frequencies,
        "genotype_distribution": genotype_distribution,
        "complex_activities": {
            "CI_mean":  round(sum(ci_values)  / len(ci_values), 1),
            "CI_range": f"{min(ci_values)}–{max(ci_values)}%",
            "CII_mean": round(sum(cii_values) / len(cii_values), 1),
            "CIV_mean": round(sum(civ_values) / len(civ_values), 1),
        },
    }


# ── Definitions ───────────────────────────────────────────────────────────────
def get_definitions() -> dict[str, Any]:
    pharmacology = [
        {
            "term": "Valproate / VPA — ABSOLUTE CI in NDUFS2/CI-Leigh",
            "definition": (
                "NEVER use valproate in any patient with NDUFS2 mutation or suspected CI-Leigh.\n\n"
                "Three independent lethal mechanisms:\n"
                "1. CoA SEQUESTRATION (valproyl-CoA): depletes mitochondrial CoA → TCA cycle "
                "   substrates drop → OXPHOS cannot produce ATP → lactic crisis in CI-deficient cell.\n"
                "2. POLG INHIBITION: valproate inhibits mtDNA polymerase gamma → mtDNA depletion "
                "   → fewer MT-ND1–ND6 / MT-ND4L templates → fewer P-module CI subunits "
                "   → assembly of Q-module (NDUFS2/N2 terminal Fe-S) blocked.\n"
                "3. HEPATOTOXICITY: VPA directly hepatotoxic → hepatic OXPHOS failure "
                "   → impaired gluconeogenesis → hypoglycaemia → crisis amplified.\n\n"
                "Preferred AED: LEVETIRACETAM (LEV) — renal excretion, zero mito toxicity. "
                "Second-line: clobazam (CLB), clonazepam — no mito toxicity."
            ),
        },
        {
            "term": "Metformin — ABSOLUTE CI in NDUFS2",
            "definition": (
                "Metformin is a direct Complex I inhibitor. The ND1/quinone binding site of CI "
                "is metformin's primary pharmacological target — adjacent to the NDUFS2/N2 "
                "terminal electron transfer site → blocks N2→ubiquinone electron flow.\n\n"
                "In NDUFS2 deficiency: this IS the disease locus. CI already at 5–20%.\n"
                "Metformin's CI inhibition removes this residual → near-zero CI activity → "
                "massive NADH accumulation → lactate surge (may exceed 20 mmol/L).\n\n"
                "Never use for glucose management in any CI-Leigh patient.\n"
                "Alternative: insulin (does not interact with OXPHOS)."
            ),
        },
        {
            "term": "Succinate — Level C (Complex I Bypass for NDUFS2/N2)",
            "definition": (
                "MECHANISTIC BASIS:\n"
                "  Succinate → Complex II (succinate dehydrogenase / SDHA) → ubiquinol "
                "  → Complex III → CIV → O2.\n"
                "  This bypasses the blocked N2→ubiquinone step entirely: the NDUFS2 N2 "
                "  cluster is not required for CII-mediated electron flow to ubiquinol.\n\n"
                "CLINICAL RATIONAL:\n"
                "  In CI deficiency, maintaining ubiquinol pool via CII sustains partial ATP "
                "  synthesis without relying on NDUFS2/N2.\n"
                "  Level C evidence; dose 2–8 g/day oral succinate or sodium succinate.\n"
                "  Most relevant in crisis: IV succinate (where available).\n\n"
                "CAUTION: Not a replacement for glucose (GIR 6-8 during crisis is first-line)."
            ),
        },
        {
            "term": "Riboflavin / Vitamin B2 — Level C (CI-specific, upstream of NDUFS2)",
            "definition": (
                "Riboflavin → riboflavin-5'-phosphate (FMN) + FAD.\n\n"
                "CI relevance (upstream of NDUFS2):\n"
                "  FMN binds NDUFV1 (51kDa/N3 subunit) — the FIRST electron acceptor from NADH.\n"
                "  Extra FMN may stabilise NDUFV1 and improve electron injection at N3,\n"
                "  potentially enhancing flux through the complete Fe-S relay to N2/ubiquinone.\n\n"
                "NOTE: Riboflavin is more directly targeted in NDUFV1 deficiency (FMN binds "
                "NDUFV1 active site directly). In NDUFS2 deficiency, the N2 cluster itself "
                "cannot be repaired pharmacologically. Riboflavin used empirically. Level C.\n"
                "Dose: 100–400 mg/day."
            ),
        },
        {
            "term": "Thiamine (B1) + Biotin — MANDATORY Empiric in ALL Leigh Presentations",
            "definition": (
                "Before NDUFS2 genotype is confirmed, every infant with Leigh/Leigh-like MRI "
                "+ lactic acidosis + CI biochemistry MUST receive empiric thiamine AND biotin.\n\n"
                "THIAMINE B1:\n"
                "  SLC19A3 deficiency (biotin-thiamine responsive basal ganglia disease / BTBGD) "
                "  and PDHC deficiency can mimic NDUFS2-Leigh.\n"
                "  Both are TREATABLE if thiamine started early.\n"
                "  Dose: 100–300 mg/day IV or oral.\n\n"
                "BIOTIN:\n"
                "  Biotinidase (BTD) deficiency causes Leigh-like neurological crisis responding "
                "  dramatically to biotin.\n"
                "  Dose: 5–10 mg/day.\n\n"
                "Never withhold empiric thiamine + biotin while awaiting genetics in Leigh syndrome."
            ),
        },
        {
            "term": "Acute Crisis Protocol — NDUFS2/CI-Leigh Metabolic Emergency",
            "definition": (
                "STEP 1 — IV DEXTROSE STAT:\n"
                "  GIR 6-8 mg/kg/min; NEVER fast. Trigger: deterioration, lactate >5, fever.\n\n"
                "STEP 2 — HOLD MITOCHONDRIAL TOXINS:\n"
                "  Check for inadvertent metformin, phenobarbital, linezolid, propofol, VPA. "
                "  Stop immediately.\n\n"
                "STEP 3 — NaHCO3 IV (pH <7.20):\n"
                "  0.5–1 mEq/kg over 1–2h; lactate monitoring q2h; target pH >7.25.\n\n"
                "STEP 4 — IV RIBOFLAVIN + THIAMINE (100 mg each IV if available).\n\n"
                "STEP 5 — IV SUCCINATE (metabolic centre, if available):\n"
                "  0.5–1 g/kg/day → CII-bypass of blocked NDUFS2/N2 terminal Fe-S.\n\n"
                "STEP 6 — SEIZURES → LEV IV: 20–40 mg/kg loading. ABSOLUTE CI VPA.\n\n"
                "STEP 7 — RESPIRATORY → NIV/BiPAP: SpO2 <92% or RR >40.\n"
                "  Anaesthesia: sevoflurane, NOT propofol.\n\n"
                "EMERGENCY CARD: IV dextrose immediately; NEVER VPA, metformin, linezolid, "
                "chloramphenicol, propofol, or ketogenic diet. Contact metabolic neurology."
            ),
        },
    ]

    gene_concepts = [
        {
            "term": "NDUFS2 Gene Structure and Expression",
            "definition": (
                "Gene: NDUFS2 (NADH:Ubiquinone Oxidoreductase Core Subunit S2)\n"
                "Also known as: PSST (named after the plant homolog)\n"
                "Chromosome: 1q23.3\n"
                "Protein: 463 aa precursor; MTS ~32 aa; mature form ~431 aa; ~49 kDa\n\n"
                "NDUFS2 is a core subunit of the Q-module (peripheral arm, membrane interface).\n"
                "Ubiquitous expression across tissues; highest in brain, heart, skeletal muscle.\n\n"
                "OMIM *602985 (gene); #256000 (Leigh Syndrome) / CI Deficiency (NDUFS2)\n\n"
                "Inheritance: AR — biallelic (compound heterozygous or homozygous) "
                "loss-of-function variants required for disease."
            ),
        },
        {
            "term": "N2 Fe-S Cluster — Terminal Electron Transfer to Ubiquinone",
            "definition": (
                "NDUFS2 harbours the N2 iron-sulfur cluster ([4Fe-4S]) — the TERMINAL and "
                "ONLY Fe-S cluster in the Q-module.\n\n"
                "N2's unique structural position:\n"
                "  • Located at the junction of the Q-module and the ubiquinone-binding cavity\n"
                "  • Directly reduces ubiquinone to ubiquinol (the FINAL Fe-S electron transfer)\n"
                "  • All upstream clusters (N1b, N4, N5 in NDUFS1; N6a/N6b) feed into N2\n\n"
                "Complete Fe-S relay (N2 as terminus):\n"
                "  NADH → FMN/N3 (NDUFV1) → N1a (NDUFV2) → N1b/N4/N5 (NDUFS1) "
                "  → N6a/N6b (NDUFB9/NDUFS8) → N2 (NDUFS2) → ubiquinone\n\n"
                "Loss of NDUFS2/N2: the entire upstream Fe-S relay is intact but electron "
                "transfer to ubiquinone is blocked at the final step → complete CI deficiency. "
                "This is mechanistically equivalent to losing any upstream Fe-S cluster."
            ),
        },
        {
            "term": "NDUFS2 vs NDUFS1 vs NDUFV1 vs NDUFS4 — Distinguishing CI-Leigh Series",
            "definition": (
                "All four cause isolated CI deficiency + Leigh syndrome.\n"
                "Biochemical fingerprint is identical: CI 5–20%, CII/CIII/CIV normal.\n"
                "Clinical differentiation:\n\n"
                "NDUFS4 (175 aa, accessory subunit, 5q11.2):\n"
                "  • Olfactory bulb MRI: ~52–65% (PATHOGNOMONIC — not seen in NDUFS2)\n"
                "  • Severe central apnoea dominant\n\n"
                "NDUFV1 (464 aa, N-module FMN, 11q13.2):\n"
                "  • Leukodystrophy / white matter T2: ~40–50% (DISTINGUISHING)\n"
                "  • Myoclonus more prominent (~38–40%)\n\n"
                "NDUFS1 (727 aa, IP1/75kDa, 2q33.3):\n"
                "  • Peripheral neuropathy: ~50% (DISTINGUISHING — not seen in NDUFS2)\n"
                "  • HCM slightly higher (~12%)\n\n"
                "NDUFS2 (463 aa, PSST/49kDa, 1q23.3) — THIS DISEASE:\n"
                "  • NO peripheral neuropathy (KEY DDx vs NDUFS1)\n"
                "  • NO olfactory bulb lesions (KEY DDx vs NDUFS4)\n"
                "  • NO leukodystrophy (KEY DDx vs NDUFV1)\n"
                "  • HCM ~8% (similar to NDUFS4/NDUFV1)\n"
                "  • Standard Leigh MRI: bilateral putamen + brainstem T2 (~80%)\n"
                "  • N2 = TERMINAL Fe-S → final Q-module block (vs N-module blocks in others)"
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "NDUFS2/N2 — Q-Module Terminal Position and CI Deficiency",
            "definition": (
                "NDUFS2 biallelic variants → loss of N2 Fe-S cluster → CI deficiency.\n\n"
                "The Q-module position is unique:\n"
                "  • N2 is the ONLY Fe-S cluster in the Q-module\n"
                "  • N2 sits at the mouth of the ubiquinone-binding channel\n"
                "  • The ND1 membrane subunit (metformin target) also contributes to this cavity\n"
                "  • NDUFS2/N2 loss → cannot reduce ubiquinone → NADH cannot be re-oxidised\n"
                "  → NADH/NAD+ ratio ↑ → lactate/pyruvate ↑ → lactic acidosis\n\n"
                "Biochemical criteria:\n"
                "  Complex I activity: 5–20% of age-matched controls\n"
                "  Complexes II, III, IV: NORMAL\n"
                "  Lactate/pyruvate ratio: typically >20\n"
                "  Plasma lactate: 3–15 mmol/L baseline; crisis >15 mmol/L\n\n"
                "Inheritance: AR — biallelic. Siblings at 25% risk."
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Complete Prescribing Safety Card — NDUFS2 CI-Leigh",
            "definition": (
                "ABSOLUTE CONTRAINDICATIONS (never use):\n"
                "  ▪ Valproate / VPA (triple mito-toxicity: CoA/POLG/hepatotoxicity)\n"
                "  ▪ Metformin (direct CI inhibitor at ND1/N2 quinone-binding site)\n"
                "  ▪ Linezolid (23S rRNA → blocks all 7 mtDNA ND subunits → CI loss)\n"
                "  ▪ Chloramphenicol (same mt-ribosome mechanism as linezolid)\n"
                "  ▪ Ketogenic Diet (forces NADH overload → N2/ubiquinone blocked)\n\n"
                "HIGH CAUTION / AVOID:\n"
                "  ▪ Propofol (PRIS: CIV inhibition creates 2nd ETC block downstream of NDUFS2)\n"
                "  ▪ Phenobarbital (secondary CI inhibitor; use only if no alternative)\n"
                "  ▪ Fasting (no fasting >4 h; GIR 6-8 during illness/crisis)\n\n"
                "PREFERRED / SAFE:\n"
                "  ▪ LEV (levetiracetam) — AED first-line; renal; no mito toxicity\n"
                "  ▪ Clonazepam / CLB (clobazam) — benzodiazepines; no mito toxicity\n"
                "  ▪ Sevoflurane — anaesthetic choice (not propofol)\n"
                "  ▪ Dexmedetomidine — sedation alternative to propofol\n"
                "  ▪ Insulin — glucose management (not metformin)\n"
                "  ▪ Baclofen — spasticity\n"
                "  ▪ Physiotherapy / OT — central ataxia + dystonia (no peripheral neuropathy)"
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = get_overview()
    print(json.dumps(ov, indent=2)[:2000])
    print("\n=== BREAKDOWN (patients[:3]) ===")
    bk = get_breakdown()
    print(json.dumps({"patients": bk["patients"][:3], "feature_frequencies": bk["feature_frequencies"]}, indent=2))
    print("\n=== DEFINITIONS (first term) ===")
    df = get_definitions()
    print(df["pharmacology"][0]["term"])
