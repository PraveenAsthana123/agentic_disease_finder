#!/usr/bin/env python3
"""NDUFS5 — Leigh Syndrome Isolated Complex I Deficiency (N-Module Peripheral Structural Subunit / NDUFS1 Contact / No Fe-S Cluster).

NDUFS5 (NADH:Ubiquinone Oxidoreductase Core Subunit S5) is a small peripheral
structural subunit of the N-module (hydrophilic arm) of Complex I.  At 106 aa
precursor (~90 aa mature, ~10.8 kDa) it is one of the smallest CI core subunits.
NDUFS5 does NOT carry an Fe-S cluster — its function is structural and stabilising:
it contacts NDUFS1 (IP1/75 kDa subunit, which carries N5 [4Fe-4S]) and contributes
to N-module assembly and stability.

  NDUFS5 gene      OMIM *603847
  Disease          Leigh Syndrome (OMIM #256000) /
                   Mitochondrial Complex I Deficiency, Nuclear Type 5 (OMIM #618235)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       1p34.3

PATHOPHYSIOLOGY (Complex I / N-module peripheral structural role / NDUFS5):
  The Fe-S electron relay chain of Complex I transfers electrons from NADH → ubiquinone
  through a series of iron-sulfur clusters.  NDUFS5 sits in the periphery of the
  N-module but does NOT carry a cluster itself:

    NDUFV1  (N3, [4Fe-4S]) ← FMN primary NADH acceptor (N-module)
    NDUFV2  (N1b, [2Fe-2S]) ← second relay step (N-module)
    NDUFS7  (N4, [4Fe-4S]) ← third relay (Q-module/N-module junction)
    NDUFS8  (N6a, [4Fe-4S]) ← fourth relay (Q-module approach / TYKY)
    NDUFS8  (N6b, [4Fe-4S]) ← fifth relay (same TYKY subunit)
    NDUFS1  (N5, [4Fe-4S]) ← peripheral arm relay ← NDUFS5 contacts HERE
    NDUFS2  (N2, [4Fe-4S]) ← TERMINAL carrier → ubiquinone reduction

  NDUFS5's peripheral structural role:
    1. NDUFS5 contacts NDUFS1 (IP1/75 kDa) — the subunit that carries the N5
       [4Fe-4S] cluster in the peripheral relay between N6b (NDUFS8) and N2 (NDUFS2).
    2. NDUFS5 does NOT carry a cluster itself — its loss causes CI assembly failure,
       not a direct Fe-S electron transfer block.
    3. CI sub-assembly intermediates appear on BN-PAGE (similar to NDUFS3/NDUFS4
       assembly failure patterns) — unlike NDUFS7/NDUFS8 which show cleaner absent CI.
    4. Without NDUFS5, the fully assembled CI holocomplex cannot form.  Result:
       isolated CI deficiency 5–20%, CII/CIII/CIV NORMAL.

  Key contrast with Fe-S relay subunits (NDUFS7, NDUFS8, NDUFS2, NDUFS1):
    • NDUFS7 / NDUFS8 / NDUFS1 / NDUFS2 carry actual Fe-S clusters — their loss
      creates a DIRECT electron transfer block.  BN-PAGE shows absent/severely
      reduced CI with a relatively clean pattern (full complex cannot assemble
      once relay clusters are absent but fewer sub-assembly bands).
    • NDUFS5 has NO cluster — its loss causes assembly failure (structural role).
      BN-PAGE shows sub-assembly intermediates similar to NDUFS3 (scaffold) and
      NDUFS4 (N-module accessory) — NOT a clean absent CI.

  Biochemical signature (IDENTICAL to all CI-Leigh nuclear subunit mutations):
    Complex I: 5–20% of control (severely reduced) — ISOLATED CI DEFICIENCY
    Complex II (SDHA): NORMAL (fully nuclear-encoded; NDUFS5 not required for CII)
    Complex III: NORMAL
    Complex IV (COX): NORMAL — KEY DDx SURF1/SCO2/COX10/COX15 (CIV-only deficiency)

DISTINGUISHING FEATURES vs NDUFS1/NDUFS4/NDUFV1/NDUFS7/NDUFS8:
  vs NDUFS4 (accessory N-module, 5q11.2):
    • NO olfactory bulb lesions in NDUFS5 (NDUFS4: ~52–65% — PATHOGNOMONIC)
    • Both: assembly failure on BN-PAGE (but NDUFS4 has distinct olfactory signature)
  vs NDUFV1 (N-module FMN/N3, 11q13.2):
    • NO leukodystrophy / white matter T2 signal in NDUFS5 (NDUFV1: ~40–50%)
  vs NDUFS1 (IP1/75kDa/N5 cluster carrier, 2q33.3):
    • NO peripheral neuropathy in NDUFS5 (NDUFS1: ~50% — CRITICAL DDx)
    • NDUFS5 contacts NDUFS1 structurally but does NOT carry N5 itself
  vs NDUFS7 (N4/Q-N junction, 19p13.3):
    • NDUFS7 = direct N4 Fe-S relay block; cleaner BN-PAGE absent CI
    • NDUFS5 = structural assembly failure; BN-PAGE sub-assembly intermediates
  vs NDUFS8 (TYKY/N6a-N6b, 11q13.2):
    • NDUFS8 = dual N6a+N6b Fe-S direct relay block; cleaner absent CI on BN-PAGE
    • NDUFS5 = peripheral structural/stabilising role; assembly failure on BN-PAGE

FOUNDER / RECURRENT MUTATIONS:
  p.Arg81Cys   c.241C>T — severe infantile; N-module contact region; compound het
  p.Glu68Lys   c.202G>A — moderate; partial CI residual; compound het
  p.Trp85*     c.255G>A — null allele; severe; homozygous in consanguineous families
  p.Ala72Val   c.215C>T — milder course; partial assembly; compound het
  c.IVS2+1G>T  (splice donor) — partial residual CI; milder Leigh course

THERAPY — NDUFS5/CI-LEIGH SPECIFICS:
  No targeted NDUFS5 structural restoration is clinically available.
  Management follows the CI-Leigh supportive protocol (cofactors + avoid CI toxins).
  Succinate bypass (Level C): CII-mediated electron entry to ubiquinol bypasses the
    NDUFS5 assembly-failure CI block ENTIRELY — electrons enter at ubiquinol directly.
  Riboflavin B2 (Level C): FMN reinforces NDUFV1/N3 upstream; limited direct benefit
    given the assembly failure (no Fe-S cluster target in NDUFS5 itself).
  No peripheral neuropathy: physiotherapy focuses on central ataxia/dystonia only
    (no neuropathic foot drop, no EMG needed, no orthotics).

References:
  Mayr JA et al. J Med Genet. 2012.
    (CI subunit mutations including small peripheral subunits such as NDUFS5)
  Fassone E, Rahman S. J Med Genet. 2012;49(9):578-590.
    (CI deficiency genetics review — NDUFS5 in context of CI-Leigh series)
  Sazanov LA. Nat Rev Mol Cell Biol. 2015;16(6):375-388.
    (CI structural review — NDUFS5 peripheral arm / N-module context)
  Bénit P et al. Hum Mutat. 2004.
    (Nuclear-encoded CI subunit screening series including NDUFS5)
  Tucker EJ et al. Nat Genet. 2011;43(10):983-986.
    (CI nuclear subunit mutations — next-generation sequencing discovery)
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 623
DISEASE_ID   = "ndufs5"
DISEASE_NAME = (
    "NDUFS5 Leigh Syndrome — Isolated Complex I Deficiency "
    "(CI-Leigh / NDUFS5 N-Module Peripheral Structural Subunit / NDUFS1 Contact / No Fe-S Cluster)"
)
GENE         = "NDUFS5"
PROTEIN      = (
    "NDUFS5 — 106 aa precursor / ~90 aa mature, ~10.8 kDa; peripheral structural subunit of the "
    "N-module hydrophilic arm; contacts NDUFS1 (IP1/75 kDa / N5 [4Fe-4S]); NO Fe-S cluster in "
    "NDUFS5 itself — structural/stabilising role; loss causes CI assembly failure "
    "(sub-assembly intermediates on BN-PAGE), not a direct Fe-S relay block"
)
OMIM_GENE    = "*603847"
OMIM_DISEASE = "#256000 (Leigh Syndrome) / Mitochondrial Complex I Deficiency Nuclear Type 5 (#618235)"
CHROMOSOME   = "1p34.3"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Infantile / early childhood (6–24 months)"
COHORT_SIZE  = 40
COLOR        = "#4a148c"   # deep purple — N-module peripheral stabiliser, no Fe-S cluster
LIGHT        = "#f3e5f5"

# Genotype pool
GENO_ARG81CYS   = "p.Arg81Cys / truncating (c.241C>T compound het) — severe infantile; N-module contact region"
GENO_GLU68LYS   = "p.Glu68Lys / missense (c.202G>A compound het) — moderate; partial CI residual"
GENO_TRP85STOP  = "p.Trp85* homozygous (c.255G>A) — null allele; severe; consanguineous"
GENO_ALA72VAL   = "p.Ala72Val / missense (c.215C>T compound het) — milder course; partial assembly"
GENO_SPLICE     = "p.Glu68Lys / c.IVS2+1G>T (splice donor) — partial residual CI; milder Leigh"

GENO_POOL    = [GENO_ARG81CYS, GENO_GLU68LYS, GENO_TRP85STOP, GENO_ALA72VAL, GENO_SPLICE]
GENO_WEIGHTS = [0.28,          0.22,           0.20,            0.17,           0.13]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG — reproducible 40-patient NDUFS5/CI-Leigh cohort (seed-623)."""
    return random.Random(SEED)


# ── Patient cohort ────────────────────────────────────────────────────────────
_TX_POOL = [
    "IV Dextrose GIR 6-8 (metabolic crisis — first-line)",
    "Riboflavin B2 (100–400 mg/day — CI-specific FMN precursor; N-module upstream of NDUFS5 assembly block)",
    "CoQ10/Ubiquinol (300–600 mg/day)",
    "Thiamine B1 (mandatory empiric, all Leigh until SLC19A3/BTD excluded)",
    "Biotin (mandatory empiric, all Leigh until BTD ruled out)",
    "Succinate (IV/oral — Complex II bypass; bypasses blocked CI assembly entirely)",
    "NaHCO3 (lactic acidosis correction, pH <7.20)",
    "LEV (seizures — preferred AED, renal excretion, no mito toxicity)",
    "NIV/BiPAP (central respiratory compromise)",
    "Carnitine (secondary deficiency, 50–100 mg/kg/day)",
    "Physiotherapy / occupational therapy (central ataxia + dystonia — no peripheral neuropathy)",
    "Enteral feeds / NG (avoid fasting; high-carbohydrate; KD strictly CI)",
]

_OUTCOMES = [
    "Alive — stable moderate-to-severe Leigh; cofactor + respiratory support",
    "Alive — prolonged survival with splice/missense alleles; partial CI assembly residual; school-age",
    "Alive — severe Leigh; dependent care; ongoing multidisciplinary support",
    "Died — acute lactic crisis + respiratory failure in first 18 months (null/frameshift alleles)",
    "Died — progressive brainstem Leigh + respiratory failure; 2–4 yr trajectory",
]
_OUT_WEIGHTS = [0.28, 0.12, 0.22, 0.22, 0.16]


def _generate_cohort(rng: random.Random) -> list[dict[str, Any]]:
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno        = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex         = rng.choice(["M", "F"])
        onset_mo    = rng.choices(
            [3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 24, 30, 36, 48],
            weights=[3, 5, 8, 12, 13, 11, 9, 8, 7, 6, 5, 4, 4, 5],
        )[0]
        onset_yr    = round(onset_mo / 12, 1)
        lactate_mm  = round(rng.uniform(4.2, 20.0), 1)
        ci_pct      = round(rng.uniform(5.0, 21.0))          # Complex I (% of control)
        cii_pct     = round(rng.uniform(88, 108))             # Complex II — NORMAL
        civ_pct     = round(rng.uniform(85, 105))             # Complex IV — NORMAL

        has_regression      = rng.random() < 0.95
        has_leigh_mri       = rng.random() < 0.88
        has_hypotonia       = rng.random() < 0.85
        has_lactic          = rng.random() < 0.90
        has_resp            = rng.random() < 0.58
        has_seizures        = rng.random() < 0.48
        has_ataxia          = rng.random() < 0.42
        has_dystonia        = rng.random() < 0.35
        has_myoclonus       = rng.random() < 0.18
        has_nystagmus       = rng.random() < 0.15
        has_optic           = rng.random() < 0.25
        has_spasticity      = rng.random() < 0.32
        has_hcm             = rng.random() < 0.06   # ~6% — low; contrast SCO2 100%/NDUFV2 80%
        has_hepatopathy     = rng.random() < 0.03   # RARE — KEY DDx POLG/DGUOK
        has_iron            = False                  # NO iron overload — KEY DDx GRACILE
        has_tubulopathy     = rng.random() < 0.03   # RARE — KEY DDx COX10 (65%)
        has_olfactory       = False                  # NO olfactory bulb lesions — KEY DDx NDUFS4
        has_leukodystrophy  = rng.random() < 0.04   # RARE — KEY DDx NDUFV1 (40–50%)
        has_neuropathy      = False                  # NO peripheral neuropathy — KEY DDx NDUFS1 (50%)

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
        if has_hcm:            feat_list.append("HCM (RARE ~6% — KEY DDx SCO2 100%/NDUFV2 80%)")
        if has_hepatopathy:    feat_list.append("Hepatopathy (RARE — KEY DDx POLG/DGUOK)")

        txs     = rng.sample(_TX_POOL, k=rng.randint(4, 7))
        outcome = rng.choices(_OUTCOMES, weights=_OUT_WEIGHTS)[0]

        patients.append({
            "id":                  f"NDUFS5-{i:03d}",
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
        "HCM (~6% — low; contrast SCO2 100%, NDUFV2 80%)":                      _pct("has_hcm"),
        "Hepatopathy (RARE — KEY DDx POLG / DGUOK)":                            _pct("has_hepatopathy"),
        "NO Peripheral Neuropathy (KEY DDx NDUFS1 ~50% neuropathy)":            100,
        "NO Olfactory Bulb Lesions (KEY DDx NDUFS4 ~52–65%)":                   100,
        "NO Leukodystrophy (KEY DDx NDUFV1 ~40–50%)":                           100,
        "NO Iron Overload (KEY DDx GRACILE — 100% iron overload)":              100,
        "NO COX Deficiency (KEY DDx SURF1 / SCO2 / COX10 / COX15)":            100,
        "Alive (with support)":                                                  round(alive / COHORT_SIZE * 100),
    }

    kpis = [
        {"label": "Cohort (n)",       "value": COHORT_SIZE,                                                                        "color": COLOR},
        {"label": "Leigh MRI",        "value": f"{feature_frequencies['Leigh / Leigh-like MRI (bilateral putamen + brainstem)']}%", "color": "#6a1b9a"},
        {"label": "No Neuropathy",    "value": "100% — DDx NDUFS1",                                                                "color": "#2e7d32"},
        {"label": "Resp Compromise",  "value": f"{feature_frequencies['Respiratory Compromise (central)']}%",                      "color": "#b71c1c"},
        {"label": "Hypotonia",        "value": f"{feature_frequencies['Hypotonia']}%",                                             "color": COLOR},
        {"label": "CI Activity",      "value": "5–20% control",                                                                    "color": "#c62828"},
        {"label": "Fatal",            "value": f"{round(died/COHORT_SIZE*100)}%",                                                  "color": "#b71c1c"},
        {"label": "Seed",             "value": f"#{SEED}",                                                                        "color": "#455a64"},
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "severity": "ABSOLUTE CI — DANGEROUS IN ALL LEIGH / CI DEFICIENCY",
            "mechanism": (
                "Triple mechanism in NDUFS5/CI-Leigh:\n"
                "1. CoA SEQUESTRATION: valproyl-CoA depletes the mitochondrial CoA pool "
                "   → TCA cycle collapses → OXPHOS substrate loss. In NDUFS5 deficiency "
                "   (CI already at 5–20% due to assembly failure — holocomplex cannot form), "
                "   CoA depletion tips the patient into irreversible lactic crisis.\n"
                "2. POLG INHIBITION: VPA inhibits mitochondrial DNA polymerase gamma → "
                "   reduces mtDNA copy number → fewer MT-ND1–6 / MT-ND4L templates → "
                "   fewer mtDNA-encoded CI P-module subunits → CI assembly further impaired "
                "   (NDUFS5 structural failure + reduced P-module supply = compound failure).\n"
                "3. HEPATOTOXICITY: direct hepatotoxic risk; hepatic OXPHOS failure → "
                "   impaired gluconeogenesis → hypoglycaemia → lactic crisis.\n"
                "Use LEV (levetiracetam) as first-line AED: renal excretion, no mito toxicity."
            ),
        },
        {
            "drug": "Metformin",
            "severity": "ABSOLUTE CI — DIRECT COMPLEX I INHIBITOR: CATASTROPHIC IN NDUFS5",
            "mechanism": (
                "Metformin inhibits Complex I (NADH dehydrogenase) at the ND1/quinone "
                "binding site.\n"
                "In NDUFS5 deficiency: CI is already at 5–20% activity due to N-module "
                "assembly failure (NDUFS5 loss prevents holocomplex formation — the assembled "
                "CI holoenyme cannot form without NDUFS5 stabilising NDUFS1/N5 contact). "
                "Metformin's CI inhibition further blocks the residual ND1 quinone-binding "
                "capacity of the sub-assembled CI fragments → near-total CI shutdown → "
                "massive lactate accumulation (>15–20 mmol/L).\n\n"
                "Alternative for glucose management: insulin only. Never metformin."
            ),
        },
        {
            "drug": "Linezolid",
            "severity": "ABSOLUTE CI — mt-Ribosome 23S rRNA Block (ALL 7 ND mtDNA Subunits)",
            "mechanism": (
                "Linezolid inhibits the mitochondrial large ribosomal subunit (mt-LSU) 23S rRNA "
                "→ blocks synthesis of ALL 13 mitochondrially-encoded OXPHOS subunits.\n"
                "In NDUFS5/CI-Leigh:\n"
                "  • NDUFS5 itself is nuclear-encoded — BUT Complex I also requires 7 mtDNA-encoded "
                "    ND subunits (MT-ND1–6, MT-ND4L) for the P-module membrane arm\n"
                "  • Linezolid blocks synthesis of ALL 7 → CI P-module cannot assemble\n"
                "  • In NDUFS5 deficiency the N-module assembly is already impaired; "
                "    P-module loss removes the remaining CI assembly scaffold entirely → CI → zero\n"
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
                "In NDUFS5 CI-Leigh: CI assembly failure (N-module structural collapse — "
                "holocomplex cannot form without NDUFS5; 5–20% residual activity from "
                "sub-assembled CI fragments). Forcing KD → NADH accumulation → NAD+ depletion "
                "→ TCA + beta-oxidation stall → worsened lactic acidosis.\n\n"
                "KD is beneficial in GLUT1-DS and PDHD — CONTRAINDICATED in CI deficiency."
            ),
        },
        {
            "drug": "Propofol",
            "severity": "AVOID / HIGH CAUTION — PRIS in CI-Leigh",
            "mechanism": (
                "Propofol Infusion Syndrome (PRIS): propofol inhibits Complex IV and "
                "uncouples fatty acid beta-oxidation.\n"
                "In NDUFS5/CI-Leigh: CI is the primary ETC bottleneck (assembly failure — "
                "N-module structural role of NDUFS5 lost; holocomplex cannot form). "
                "Propofol's CIV inhibition creates a SECOND downstream ETC block → electrons "
                "trapped between assembly-impaired CI and CIV-inhibited CIV → ROS burst.\n"
                "Use sevoflurane (volatile, no PRIS) or dexmedetomidine as alternatives."
            ),
        },
        {
            "drug": "Phenobarbital",
            "severity": "HIGH CAUTION — Secondary Complex I Inhibitor",
            "mechanism": (
                "Phenobarbital inhibits CI electron transport (quinone channel region). "
                "In NDUFS5/CI-Leigh: residual CI (5–20%, from sub-assembled CI fragments) "
                "is critical. Phenobarbital's CI inhibition reduces this residual further "
                "→ lactic decompensation risk.\n"
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
                "CI assembly collapses further in NDUFS5 assembly-deficient state → "
                "catastrophic near-zero CI activity."
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
        "Psychomotor Regression":                                          _pct("has_regression"),
        "Leigh / Leigh-like MRI":                                          _pct("has_leigh_mri"),
        "Hypotonia":                                                       _pct("has_hypotonia"),
        "Lactic Acidosis":                                                 _pct("has_lactic"),
        "Respiratory Compromise":                                          _pct("has_resp"),
        "Seizures":                                                        _pct("has_seizures"),
        "Ataxia":                                                          _pct("has_ataxia"),
        "Dystonia":                                                        _pct("has_dystonia"),
        "Myoclonus":                                                       _pct("has_myoclonus"),
        "Nystagmus":                                                       _pct("has_nystagmus"),
        "Optic Atrophy":                                                   _pct("has_optic"),
        "Spasticity":                                                      _pct("has_spasticity"),
        "HCM (RARE — ~6%, low vs SCO2 100%/NDUFV2 80%)":                  _pct("has_hcm"),
        "Hepatopathy (RARE)":                                              _pct("has_hepatopathy"),
        "Peripheral Neuropathy (NEVER — KEY DDx NDUFS1 ~50%)":            0,
        "Olfactory Bulb MRI (NEVER — KEY DDx NDUFS4 52–65%)":            0,
        "Leukodystrophy / White Matter (NEVER — KEY DDx NDUFV1 40–50%)":  0,
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
            "term": "Valproate / VPA — ABSOLUTE CI in NDUFS5/CI-Leigh",
            "definition": (
                "NEVER use valproate in any patient with NDUFS5 mutation or suspected CI-Leigh.\n\n"
                "Three independent lethal mechanisms:\n"
                "1. CoA SEQUESTRATION (valproyl-CoA): depletes mitochondrial CoA → TCA cycle "
                "   substrates drop → OXPHOS cannot produce ATP → lactic crisis in CI-assembly-deficient cell.\n"
                "2. POLG INHIBITION: valproate inhibits mtDNA polymerase gamma → mtDNA depletion "
                "   → fewer MT-ND1–ND6 / MT-ND4L templates → fewer P-module CI subunits → CI assembly "
                "   further compromised (NDUFS5 structural failure + reduced P-module supply).\n"
                "3. HEPATOTOXICITY: VPA directly hepatotoxic → hepatic OXPHOS failure "
                "   → impaired gluconeogenesis → hypoglycaemia → crisis amplified.\n\n"
                "Preferred AED: LEVETIRACETAM (LEV) — renal excretion, zero mito toxicity. "
                "Second-line: clobazam (CLB), clonazepam — no mito toxicity."
            ),
        },
        {
            "term": "Succinate — Level C (Complex II Bypass — Bypasses NDUFS5 CI Assembly Failure Entirely)",
            "definition": (
                "MECHANISTIC BASIS:\n"
                "  Succinate → Complex II (succinate dehydrogenase / SDHA) → ubiquinol "
                "  → Complex III → CIV → O2.\n"
                "  This bypasses the NDUFS5 CI assembly failure ENTIRELY: neither the "
                "  N-module structural collapse (NDUFS5 loss), nor the downstream Fe-S relay "
                "  (NDUFS1/N5, NDUFS2/N2), nor the upstream N-module (NDUFV1/N3, NDUFV2/N1b) "
                "  are required for CII→ubiquinol electron entry.\n\n"
                "CLINICAL RATIONALE:\n"
                "  In CI deficiency with N-module assembly failure, maintaining ubiquinol pool "
                "  via CII sustains partial ATP synthesis without relying on assembled CI.\n"
                "  Level C evidence; dose 2–8 g/day oral succinate or sodium succinate.\n"
                "  Most relevant in crisis: IV succinate (where available).\n\n"
                "CAUTION: Not a replacement for glucose (GIR 6-8 during crisis is first-line)."
            ),
        },
        {
            "term": "Riboflavin / Vitamin B2 — Level C (CI-specific, upstream of NDUFS5 assembly block)",
            "definition": (
                "Riboflavin → riboflavin-5'-phosphate (FMN) + FAD.\n\n"
                "CI relevance (upstream of NDUFS5 assembly block):\n"
                "  FMN binds NDUFV1 (51kDa/N3 subunit) — the FIRST electron acceptor from NADH.\n"
                "  Extra FMN may stabilise NDUFV1 and improve electron injection at N3.\n"
                "  However, in NDUFS5 deficiency the N-module assembly is impaired — without NDUFS5 "
                "  stabilising NDUFS1/N5 contact, the N-module cannot form the complete holocomplex.\n"
                "  Unlike NDUFV1 deficiency (where FMN directly targets the defective active site), "
                "  riboflavin cannot repair the NDUFS5 structural deficit. Empiric Level C use.\n\n"
                "NOTE: Riboflavin is most directly targeted in NDUFV1 deficiency (FMN active site). "
                "In NDUFS5 deficiency the assembly failure cannot be repaired pharmacologically. "
                "Riboflavin used empirically alongside succinate. Level C.\n"
                "Dose: 100–400 mg/day."
            ),
        },
        {
            "term": "Thiamine (B1) + Biotin — MANDATORY Empiric in ALL Leigh Presentations",
            "definition": (
                "Before NDUFS5 genotype is confirmed, every infant with Leigh/Leigh-like MRI "
                "+ lactic acidosis + CI biochemistry MUST receive empiric thiamine AND biotin.\n\n"
                "THIAMINE B1:\n"
                "  SLC19A3 deficiency (biotin-thiamine responsive basal ganglia disease / BTBGD) "
                "  and PDHC deficiency can mimic NDUFS5-Leigh.\n"
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
            "term": "Acute Crisis Protocol — NDUFS5/CI-Leigh Metabolic Emergency",
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
                "  0.5–1 g/kg/day → CII-bypass of NDUFS5 CI assembly failure entirely.\n\n"
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
            "term": "NDUFS5 Gene Structure and Expression (N-Module Peripheral Structural Subunit)",
            "definition": (
                "Gene: NDUFS5 (NADH:Ubiquinone Oxidoreductase Core Subunit S5)\n"
                "Chromosome: 1p34.3\n"
                "Protein: 106 aa precursor; mature form ~90 aa; ~10.8 kDa\n\n"
                "NDUFS5 is a small peripheral structural subunit of the N-module (hydrophilic arm) "
                "of Complex I. It contacts NDUFS1 (IP1/75 kDa), which carries the N5 [4Fe-4S] cluster "
                "in the Fe-S relay chain. NDUFS5 itself carries NO Fe-S cluster — its role is "
                "structural: anchoring and stabilising the N-module periphery, enabling NDUFS1/N5 "
                "to be correctly positioned within the holocomplex relay chain.\n\n"
                "OMIM *603847 (gene); #256000 (Leigh Syndrome) / #618235 (CI Deficiency Nuclear Type 5)\n\n"
                "Inheritance: AR — biallelic (compound heterozygous or homozygous) "
                "loss-of-function variants required for disease.\n\n"
                "Ubiquitous expression; highest in brain, heart, skeletal muscle."
            ),
        },
        {
            "term": "NDUFS5 Peripheral Structural Role — Contact with NDUFS1/N5 and Assembly Failure Mechanism",
            "definition": (
                "NDUFS5 is unique among the CI core subunits discussed in this series:\n"
                "it is the ONLY one that carries NO Fe-S cluster yet is required for CI assembly.\n\n"
                "Structural contacts:\n"
                "  NDUFS5 contacts NDUFS1 (IP1/75 kDa subunit), which carries the N5 [4Fe-4S] cluster.\n"
                "  NDUFS1 is the relay subunit bridging NDUFS8/N6b and NDUFS2/N2 (terminal).\n"
                "  Without NDUFS5, NDUFS1 cannot be correctly positioned within the N-module, "
                "  and the fully assembled CI holocomplex cannot form.\n\n"
                "Assembly failure consequences:\n"
                "  • BN-PAGE: CI sub-assembly intermediates (visible bands below the full CI band)\n"
                "  • Unlike NDUFS7/NDUFS8 (direct Fe-S loss → cleaner absent/reduced CI band)\n"
                "  • Similar pattern to NDUFS3 (Q-module scaffold) and NDUFS4 (N-module accessory)\n"
                "  • Biochemical result: identical isolated CI deficiency (5–20%)\n\n"
                "This assembly-failure pattern is the key BN-PAGE clue pointing toward a structural "
                "subunit (NDUFS3/NDUFS4/NDUFS5) rather than a direct Fe-S relay subunit."
            ),
        },
        {
            "term": "CI Fe-S Relay N3→N1b→N4→N6a→N6b→N5→N2→UQ — NDUFS5 Structural Context",
            "definition": (
                "The full electron relay chain through Complex I Fe-S clusters:\n\n"
                "  NADH → FMN (NDUFV1) → N3 → N1b (NDUFV2) → N4 (NDUFS7) "
                "→ N6a (NDUFS8) → N6b (NDUFS8) → N5 (NDUFS1) → N2 (NDUFS2) → Ubiquinone\n\n"
                "NDUFS5 position: peripheral structural stabiliser contacting NDUFS1/N5.\n"
                "NDUFS5 does NOT occupy a relay cluster step — it enables NDUFS1 to occupy step 6.\n\n"
                "Consequence of NDUFS5 loss:\n"
                "  • NDUFS5 absent → NDUFS1 incorrectly positioned in N-module\n"
                "  • CI holocomplex assembly fails → sub-assembly intermediates on BN-PAGE\n"
                "  • All relay steps (N3 through N2) are non-functional because the assembled\n"
                "    holocomplex cannot form — even though the individual cluster-carrying subunits\n"
                "    (NDUFS7, NDUFS8, NDUFS1, NDUFS2) may individually be intact\n"
                "  • NADH cannot be re-oxidised → lactic acidosis\n\n"
                "Therapeutic implication: succinate (CII bypass) donates electrons directly "
                "to ubiquinol, bypassing the entire unassembled CI."
            ),
        },
        {
            "term": "NDUFS5 vs NDUFS7 vs NDUFS8 vs NDUFS1 vs NDUFS2 vs NDUFS3 vs NDUFS4 vs NDUFV1 — CI-Leigh Series",
            "definition": (
                "All cause isolated CI deficiency + Leigh syndrome.\n"
                "Biochemical fingerprint is identical: CI 5–20%, CII/CIII/CIV normal.\n"
                "Clinical differentiation:\n\n"
                "NDUFS4 (175 aa, accessory, 5q11.2):\n"
                "  • Olfactory bulb MRI: ~52–65% (PATHOGNOMONIC — not seen in NDUFS5)\n"
                "  • Also assembly failure on BN-PAGE (similar pattern to NDUFS5)\n\n"
                "NDUFV1 (464 aa, N-module FMN/N3, 11q13.2):\n"
                "  • Leukodystrophy / white matter T2: ~40–50% (DISTINGUISHING)\n\n"
                "NDUFS1 (727 aa, IP1/75kDa, N5 carrier, 2q33.3):\n"
                "  • Peripheral neuropathy: ~50% (DISTINGUISHING — not seen in NDUFS5)\n"
                "  • NDUFS5 contacts NDUFS1 structurally; NDUFS1 carries N5 Fe-S\n\n"
                "NDUFS7 (213 aa, N4/20kDa, 19p13.3):\n"
                "  • Direct N4 Fe-S relay block; BN-PAGE: cleaner absent CI\n"
                "  • NDUFS5 = assembly failure; BN-PAGE: sub-assembly intermediates\n\n"
                "NDUFS8 (201 aa, TYKY/N6a-N6b, 11q13.2):\n"
                "  • Dual N6a+N6b Fe-S direct relay block; cleaner BN-PAGE absent CI\n"
                "  • NDUFS5 = structural/peripheral; assembly failure pattern on BN-PAGE\n\n"
                "NDUFS2 (463 aa, PSST/N2/49kDa, 1q23.3):\n"
                "  • Terminal N2 Fe-S loss; direct relay block; requires genetics vs NDUFS5\n\n"
                "NDUFS3 (264 aa, QP-C scaffold, 11p11.11):\n"
                "  • Q-module scaffold failure → CI sub-assembly intermediates on BN-PAGE\n"
                "  • Very similar BN-PAGE pattern to NDUFS5 (both assembly failures)\n"
                "  • NDUFS5 = N-module peripheral stabiliser; NDUFS3 = Q-module scaffold\n\n"
                "NDUFS5 (106 aa, peripheral N-module stabiliser, 1p34.3) — THIS DISEASE:\n"
                "  • NO Fe-S cluster (structural/stabilising role — not a relay carrier)\n"
                "  • Assembly failure → sub-assembly intermediates (similar to NDUFS3/NDUFS4)\n"
                "  • NO peripheral neuropathy (KEY DDx vs NDUFS1)\n"
                "  • NO olfactory bulb lesions (KEY DDx vs NDUFS4)\n"
                "  • NO leukodystrophy (KEY DDx vs NDUFV1)\n"
                "  • HCM ~6% (low — contrast SCO2 100%, NDUFV2 80%)"
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "NDUFS5 N-Module Assembly Failure and CI Deficiency",
            "definition": (
                "NDUFS5 biallelic variants → loss of peripheral N-module structural support → "
                "CI assembly failure (holocomplex cannot form).\n\n"
                "The NDUFS5 assembly failure cascade:\n"
                "  1. NDUFS5 absent/non-functional → NDUFS1 (IP1/75 kDa / N5 [4Fe-4S]) "
                "     cannot be correctly anchored in the N-module periphery\n"
                "  2. N-module assembly is impaired → CI holocomplex cannot form\n"
                "  3. CI sub-assembly intermediates accumulate (BN-PAGE pattern)\n"
                "  4. CI overall: 5–20% residual (partial activity from sub-assembled fragments)\n"
                "  5. Residual NADH cannot be re-oxidised at full rate → NADH/NAD+ ↑\n"
                "  → lactate/pyruvate ratio ↑ → lactic acidosis\n\n"
                "Biochemical criteria:\n"
                "  Complex I activity: 5–20% of age-matched controls\n"
                "  Complexes II, III, IV: NORMAL\n"
                "  Lactate/pyruvate ratio: typically >20\n"
                "  Plasma lactate: 4.2–20.0 mmol/L\n"
                "  BN-PAGE: CI sub-assembly intermediates (similar to NDUFS3/NDUFS4 assembly failure)\n\n"
                "Inheritance: AR — biallelic. Siblings at 25% risk."
            ),
        },
        {
            "term": "DDx — NDUFS5 vs NDUFS4 vs NDUFS3 vs NDUFS7 vs NDUFS8 vs NDUFV1 vs NDUFS1",
            "definition": (
                "All produce isolated CI deficiency + Leigh MRI. Key distinguishing features:\n\n"
                "NDUFS4: Olfactory bulb lesions (52–65%) — ABSENT in NDUFS5\n"
                "NDUFV1: Leukodystrophy (40–50%) — ABSENT in NDUFS5\n"
                "NDUFS1: Peripheral neuropathy (50%) — ABSENT in NDUFS5\n"
                "NDUFS7: N4 single-cluster direct relay block → cleaner absent CI on BN-PAGE\n"
                "         vs NDUFS5 assembly failure → sub-assembly intermediates\n"
                "NDUFS8: Dual N6a+N6b direct relay block → cleaner absent CI on BN-PAGE\n"
                "         vs NDUFS5 assembly failure → sub-assembly intermediates\n"
                "NDUFS3: Q-module scaffold failure → CI sub-assembly intermediates (SIMILAR BN-PAGE)\n"
                "         NDUFS5 = N-module peripheral stabiliser; NDUFS3 = Q-module scaffold\n"
                "         Both: assembly failure; require genetics for definitive distinction\n\n"
                "NDUFS5 DDx fingerprint:\n"
                "  ✓ Isolated CI deficiency (5–20%)\n"
                "  ✓ Leigh MRI (bilateral putamen + brainstem)\n"
                "  ✓ BN-PAGE: sub-assembly intermediates (assembly failure, not direct relay block)\n"
                "  ✗ NO peripheral neuropathy\n"
                "  ✗ NO olfactory bulb lesions\n"
                "  ✗ NO leukodystrophy\n"
                "  ✗ NO COX deficiency\n"
                "  ✗ NO iron overload\n"
                "  → Requires genetic confirmation (WES/targeted panel)"
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Complete Prescribing Safety Card — NDUFS5 CI-Leigh",
            "definition": (
                "ABSOLUTE CONTRAINDICATIONS (never use):\n"
                "  ▪ Valproate / VPA (triple mito-toxicity: CoA/POLG/hepatotoxicity)\n"
                "  ▪ Metformin (direct CI inhibitor at ND1/quinone-binding site; "
                "    NDUFS5 assembly failure already limits CI to 5–20% — metformin removes residual)\n"
                "  ▪ Linezolid (23S rRNA → blocks all 7 mtDNA ND subunits → CI loss)\n"
                "  ▪ Chloramphenicol (same mt-ribosome mechanism as linezolid)\n"
                "  ▪ Ketogenic Diet (forces NADH overload → assembly-failed CI cannot re-oxidise NADH)\n\n"
                "HIGH CAUTION / AVOID:\n"
                "  ▪ Propofol (PRIS: CIV inhibition creates 2nd ETC block downstream of assembly-impaired CI)\n"
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
