#!/usr/bin/env python3
"""NDUFS1 — Leigh Syndrome Isolated Complex I Deficiency (N-Module 75 kDa Iron-Sulfur Protein IP1).

NDUFS1 (NADH:Ubiquinone Oxidoreductase Core Subunit S1), also known as the 75 kDa
iron-sulfur protein (IP1), is the LARGEST nuclear-encoded subunit of Complex I.
NDUFS1 occupies the N-module (peripheral arm) and binds iron-sulfur clusters N1b,
N4, and N5 — the central electron relay stations between the FMN/N3 site at NDUFV1
and the quinone-binding site (N2) of the Q-module.  Biallelic loss-of-function
variants abolish CI NADH oxidation → isolated CI deficiency → Leigh / Leigh-like syndrome
± peripheral neuropathy.

  NDUFS1 gene      OMIM *157655
  Disease          Mitochondrial Complex I Deficiency, Nuclear Type 5 (OMIM #618226)
                   / Leigh Syndrome (OMIM #256000)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       2q33.3

PATHOPHYSIOLOGY (Complex I / N-module / NDUFS1 role):
  The N-module peripheral arm carries out NADH oxidation via a sequential iron-sulfur
  electron relay:
    NADH → FMN (at NDUFV1/51 kDa/N3) → N1a (NDUFV2) → N1b (NDUFS1) →
    N4 (NDUFS1) → N5 (NDUFS1) → N6a/N6b (NDUFB9/NDUFS8) → N2 (Q-module) →
    ubiquinone → Complex III → Complex IV → O2

  NDUFS1 is the backbone of the iron-sulfur relay: it binds THREE Fe-S clusters
  (N1b, N4, N5) that are obligatory intermediates in electron transfer.
  Without NDUFS1:
    – Fe-S clusters N1b, N4, N5 cannot be incorporated
    – Electron relay from FMN/N3 (NDUFV1) to N2/ubiquinone is broken
    – N-module cannot assemble; CI sub-complex accumulates
    – NADH oxidation is abolished → NADH/NAD+ ratio ↑ → lactic acidosis
    – Residual activity in partial-loss alleles → milder phenotype

  Biochemical signature (IDENTICAL to NDUFS4/NDUFV1 CI fingerprint):
    Complex I: 5–20% of control (severely reduced) — ISOLATED CI DEFICIENCY
    Complex II (SDHA): NORMAL (fully nuclear-encoded; NDUFS1 not required)
    Complex III: NORMAL
    Complex IV (COX): NORMAL — KEY DDx LRPPRC (CI+CIV), SURF1/SCO2/COX10/COX15 (CIV)

DISTINGUISHING FEATURES vs NDUFS4/NDUFV1:
  Peripheral Neuropathy: 45–55% — DISTINGUISHING (more than NDUFS4 or NDUFV1)
    • Axonal or demyelinating sensorimotor neuropathy
    • EMG/NCS abnormalities may precede or accompany Leigh MRI lesions
    • Important clinical clue: no other common CI-Leigh gene (NDUFS4, NDUFV1) shows
      this frequency of peripheral neuropathy
  NO olfactory bulb lesions: NDUFS1 does NOT show the olfactory bulb MRI pattern
    characteristic of NDUFS4 (~52–65%)
  NO leukodystrophy: NDUFS1 does NOT show the white matter T2 hyperintensity pattern
    characteristic of NDUFV1 (~40–50%)
  HCM: ~12% — slightly higher than NDUFS4/NDUFV1 (both ~5%), much less than SCO2 (100%)

FOUNDER / RECURRENT MUTATIONS:
  p.Arg196His   c.587G>A  — recurrent severe; consanguineous Arab/Iranian families
  p.Arg196Cys   c.586C>T  — severe compound het; multi-ethnic
  p.Gln522Lys   c.1564C>A — compound het; intermediate severity; partial residual CI
  p.Asp252Gly   c.755A>G  — compound het; variable severity
  Frameshift / nonsense — complete CI abolition; severe Leigh; early death (< 2 yr)

THERAPY — NDUFS1/CI-LEIGH SPECIFICS:
  Iron-sulfur cluster repair:
    NDUFS1 itself binds 3 Fe-S clusters; no targeted Fe-S cluster replacement is
    clinically available. Management is supportive CI-Leigh protocol (cofactors + avoid CI toxins).
  Riboflavin B2 (Level C):
    Riboflavin → FMN; reinforces NDUFV1 (FMN-binding, N3) which is upstream of NDUFS1.
    Less directly targeted than in NDUFV1 deficiency but used empirically in all CI-Leigh.
  Succinate bypass (Level C):
    Succinate → Complex II (SDHA) → ubiquinol → Complex III → CIV.
    Bypasses blocked Complex I entirely. Level C; 2–8 g/day.
  Peripheral neuropathy management:
    No disease-modifying therapy for the neuropathy component.
    Physiotherapy, orthotics (foot drop, hand function), pain (gabapentin/pregabalin).
    Monitor for respiratory muscle weakness (neuropathy + CI myopathy combination).

References:
  Procaccio V, Wallace DC. Mol Genet Metab. 2004;83(1-2):160-168.
    (Early NDUFS1 mutation characterisation in CI deficiency)
  Bénit P et al. Hum Mutat. 2001;18(3):232-238.
    (NDUFS1 mutations in mitochondrial CI deficiency with Leigh syndrome)
  Loeffen JL et al. Ann Neurol. 1998;43(1):109-120.
    (Clinical series CI-Leigh with molecular diagnosis)
  Fassone E, Rahman S. J Med Genet. 2012;49(9):578-590.
    (CI deficiency genetics review — NDUFS1 in context)
  Rötig A, Munnich A. Hum Mutat. 2003;21(6):607-614.
    (CI deficiency spectrum review; NDUFS1 genotype-phenotype)
  Schuelke M et al. J Inherit Metab Dis. 1999;22:175-183.
    (CI-Leigh genetic series including NDUFS1)
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 611
DISEASE_ID   = "ndufs1"
DISEASE_NAME = (
    "NDUFS1 Leigh Syndrome — Isolated Complex I Deficiency "
    "(CI-Leigh / NDUFS1 N-Module 75 kDa Iron-Sulfur Protein IP1)"
)
GENE         = "NDUFS1"
PROTEIN      = (
    "NDUFS1 — 727 aa precursor / ~704 aa mature, N-module iron-sulfur protein (IP1), "
    "75 kDa — binds Fe-S clusters N1b, N4, N5 — CENTRAL electron relay NDUFV1→N2→ubiquinone"
)
OMIM_GENE    = "*157655"
OMIM_DISEASE = "#618226 (Mitochondrial Complex I Deficiency, Nuclear Type 5 / Leigh Syndrome)"
CHROMOSOME   = "2q33.3"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = (
    "Infantile (3–18 months); rarely neonatal; delayed onset up to 3–4 yr in missense alleles"
)
COHORT_SIZE  = 40
COLOR        = "#006064"   # dark teal — Fe-S cluster / IP module / electron relay theme
LIGHT        = "#e0f7fa"

# Genotype pool
GENO_RHIS  = "p.Arg196His (c.587G>A) / truncating (compound het) — recurrent severe; Arab/Iranian consanguineous"
GENO_RCYS  = "p.Arg196Cys (c.586C>T) / truncating (compound het) — severe; multi-ethnic"
GENO_QLYS  = "p.Gln522Lys (c.1564C>A) / missense (compound het) — intermediate; partial CI residual"
GENO_DGLY  = "p.Asp252Gly (c.755A>G) / missense (compound het) — variable severity"
GENO_NULL  = "Frameshift / nonsense (compound het) — complete CI abolition; severe Leigh; early death"

GENO_POOL    = [GENO_RHIS, GENO_RCYS, GENO_QLYS, GENO_DGLY, GENO_NULL]
GENO_WEIGHTS = [0.28,      0.22,      0.20,       0.15,      0.15]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG — reproducible 40-patient NDUFS1/CI-Leigh cohort (seed-611)."""
    return random.Random(SEED)


# ── Patient cohort ────────────────────────────────────────────────────────────
_TX_POOL = [
    "IV Dextrose GIR 6-8 (metabolic crisis — first-line)",
    "Riboflavin B2 (100–400 mg/day — CI-specific FMN precursor; upstream of NDUFS1)",
    "CoQ10/Ubiquinol (300–600 mg/day)",
    "Thiamine B1 (mandatory empiric, all Leigh until SLC19A3/BTD excluded)",
    "Biotin (mandatory empiric, all Leigh until BTD ruled out)",
    "Succinate (IV/oral — Complex I bypass via CII — CI-SPECIFIC)",
    "NaHCO3 (lactic acidosis correction, pH <7.20)",
    "LEV (seizures — preferred AED, renal excretion, no mito toxicity)",
    "NIV/BiPAP (central ± peripheral respiratory compromise)",
    "Carnitine (secondary deficiency, 50–100 mg/kg/day)",
    "Physiotherapy / orthotics (peripheral neuropathy — foot drop, hand function)",
    "Gabapentin / pregabalin (neuropathic pain — peripheral neuropathy component)",
    "Enteral feeds / NG (avoid fasting; high-carbohydrate; KD strictly CI)",
]

_OUTCOMES = [
    "Alive — stable moderate-to-severe Leigh; cofactor + respiratory + neuropathy support",
    "Alive — prolonged survival with missense alleles; partial CI residual; school-age",
    "Alive — severe Leigh + peripheral neuropathy; dependent care; ongoing support",
    "Died — acute lactic crisis + respiratory failure in first 18 months (null alleles)",
    "Died — progressive brainstem Leigh + respiratory muscle failure; 2–5 yr trajectory",
]
_OUT_WEIGHTS = [0.25, 0.17, 0.20, 0.20, 0.18]


def _generate_cohort(rng: random.Random) -> list[dict[str, Any]]:
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno        = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex         = rng.choice(["M", "F"])
        onset_mo    = rng.choices(
            [3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 24, 30, 36, 48],
            weights=[3, 4, 7, 10, 11, 10, 9, 10, 8, 8, 7, 5, 4, 4],
        )[0]
        onset_yr    = round(onset_mo / 12, 1)
        lactate_mm  = round(rng.uniform(3.0, 16.0), 1)   # mmol/L
        ci_pct      = rng.randint(5, 22)                  # Complex I (% of control)
        cii_pct     = rng.randint(85, 118)                # Complex II — NORMAL
        civ_pct     = rng.randint(80, 112)                # Complex IV — NORMAL

        has_leigh_mri       = rng.random() < 0.82   # bilateral putamen/brainstem T2
        has_regression      = rng.random() < 0.95   # near-universal in Leigh
        has_hypotonia       = rng.random() < 0.85
        has_lactic          = rng.random() < 0.90
        has_neuropathy      = rng.random() < 0.50   # DISTINGUISHING — peripheral neuropathy
        has_resp            = rng.random() < 0.60
        has_seizures        = rng.random() < 0.52
        has_ataxia          = rng.random() < 0.52
        has_dystonia        = rng.random() < 0.42
        has_optic           = rng.random() < 0.32
        has_myoclonus       = rng.random() < 0.22
        has_nystagmus       = rng.random() < 0.28
        has_spasticity      = rng.random() < 0.40
        has_hcm             = rng.random() < 0.12   # slightly higher than NDUFS4/NDUFV1
        has_hepatopathy     = rng.random() < 0.05   # RARE — KEY DDx POLG/DGUOK
        has_iron            = False                  # NO iron overload — KEY DDx GRACILE
        has_tubulopathy     = rng.random() < 0.04   # RARE — KEY DDx COX10 (65%)
        has_olfactory       = False                  # NO olfactory bulb lesions — KEY DDx NDUFS4
        has_leukodystrophy  = rng.random() < 0.06   # RARE — KEY DDx NDUFV1 (40-50%)

        feat_list = [
            "Isolated Complex I deficiency (CII, CIII, CIV — NORMAL)",
            "Leigh / Leigh-like MRI (bilateral putamen + brainstem)",
        ]
        if has_regression:     feat_list.append("Psychomotor regression/arrest")
        if has_neuropathy:     feat_list.append("Peripheral neuropathy (DISTINGUISHING — axonal/demyelinating)")
        if has_hypotonia:      feat_list.append("Hypotonia")
        if has_lactic:         feat_list.append("Lactic acidosis")
        if has_resp:           feat_list.append("Respiratory compromise (central ± peripheral)")
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
            "id":                  f"NDUFS1-{i:03d}",
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
        "Peripheral Neuropathy (DISTINGUISHING — axonal/demyelinating)":         _pct("has_neuropathy"),
        "Hypotonia":                                                             _pct("has_hypotonia"),
        "Lactic Acidosis (elevated baseline + crisis)":                          _pct("has_lactic"),
        "Respiratory Compromise (central ± peripheral)":                         _pct("has_resp"),
        "Seizures":                                                              _pct("has_seizures"),
        "Ataxia":                                                                _pct("has_ataxia"),
        "Dystonia":                                                              _pct("has_dystonia"),
        "Myoclonus":                                                             _pct("has_myoclonus"),
        "Nystagmus":                                                             _pct("has_nystagmus"),
        "Optic Atrophy":                                                         _pct("has_optic"),
        "Spasticity":                                                            _pct("has_spasticity"),
        "HCM (~12% — slightly higher than NDUFS4/NDUFV1, much less than SCO2)": _pct("has_hcm"),
        "Hepatopathy (RARE — KEY DDx POLG / DGUOK)":                            _pct("has_hepatopathy"),
        "NO Olfactory Bulb Lesions (KEY DDx NDUFS4 ~52–65%)":                   100,
        "NO Leukodystrophy (KEY DDx NDUFV1 ~40–50%)":                           100,
        "NO Iron Overload (KEY DDx GRACILE — 100% iron overload)":              100,
        "NO COX Deficiency (KEY DDx SURF1 / SCO2 / COX10 / COX15)":            100,
        "Alive (with support)":                                                  round(alive / COHORT_SIZE * 100),
    }

    kpis = [
        {"label": "Cohort (n)",         "value": COHORT_SIZE,                                                                       "color": COLOR},
        {"label": "Leigh MRI",          "value": f"{feature_frequencies['Leigh / Leigh-like MRI (bilateral putamen + brainstem)']}%", "color": "#4a148c"},
        {"label": "Neuropathy",         "value": f"{feature_frequencies['Peripheral Neuropathy (DISTINGUISHING — axonal/demyelinating)']}%", "color": "#e65100"},
        {"label": "Resp Compromise",    "value": f"{feature_frequencies['Respiratory Compromise (central ± peripheral)']}%",         "color": "#b71c1c"},
        {"label": "Hypotonia",          "value": f"{feature_frequencies['Hypotonia']}%",                                            "color": COLOR},
        {"label": "CI Activity",        "value": "5–20% control",                                                                   "color": "#c62828"},
        {"label": "Fatal",              "value": f"{round(died/COHORT_SIZE*100)}%",                                                 "color": "#b71c1c"},
        {"label": "Seed",               "value": f"#{SEED}",                                                                       "color": "#455a64"},
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "severity": "ABSOLUTE CI — DANGEROUS IN ALL LEIGH / CI DEFICIENCY",
            "mechanism": (
                "Triple mechanism in NDUFS1/CI-Leigh:\n"
                "1. CoA SEQUESTRATION: valproyl-CoA depletes the mitochondrial CoA pool "
                "   → TCA cycle collapses → OXPHOS substrate loss. In NDUFS1 deficiency "
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
            "severity": "ABSOLUTE CI — DIRECT COMPLEX I INHIBITOR: CATASTROPHIC IN NDUFS1",
            "mechanism": (
                "Metformin inhibits Complex I (NADH dehydrogenase) at the ND1/NDUFS2 quinone "
                "binding site → blocks NADH → ubiquinone electron transfer.\n"
                "In NDUFS1 deficiency: CI is the primary disease locus (5–20% activity). "
                "Metformin's CI inhibition further suppresses this residual capacity → "
                "near-total CI shutdown → massive lactate accumulation (>15–20 mmol/L).\n\n"
                "Direct pharmacodynamic catastrophe: the drug's target IS the disease locus.\n"
                "Alternative for glucose management: insulin only. Never metformin."
            ),
        },
        {
            "drug": "Linezolid",
            "severity": "ABSOLUTE CI — mt-Ribosome 23S rRNA Block (ALL 7 ND mtDNA Subunits)",
            "mechanism": (
                "Linezolid inhibits the mitochondrial large ribosomal subunit (mt-LSU) 23S rRNA "
                "→ blocks synthesis of ALL 13 mitochondrially-encoded OXPHOS subunits.\n"
                "In NDUFS1/CI-Leigh:\n"
                "  • NDUFS1 itself is nuclear-encoded — BUT Complex I also requires 7 mtDNA-encoded "
                "    ND subunits (MT-ND1–6, MT-ND4L) for the P-module membrane arm\n"
                "  • Linezolid blocks synthesis of ALL 7 → CI P-module cannot assemble "
                "    → N-module (NDUFS1 Fe-S relay) has no membrane anchor → CI activity → zero\n"
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
                "In NDUFS1 CI-Leigh: CI cannot re-oxidise NADH (5–20% activity). "
                "Forcing KD → NADH accumulation → NAD+ depletion → TCA + beta-oxidation "
                "stall → worsened lactic acidosis and acute metabolic decompensation.\n\n"
                "KD is beneficial in GLUT1-DS and PDHD — CONTRAINDICATED in CI deficiency."
            ),
        },
        {
            "drug": "Propofol",
            "severity": "AVOID / HIGH CAUTION — PRIS in CI-Leigh",
            "mechanism": (
                "Propofol Infusion Syndrome (PRIS): propofol inhibits Complex IV and "
                "uncouples fatty acid beta-oxidation.\n"
                "In NDUFS1/CI-Leigh: CI is the primary ETC bottleneck. Propofol's CIV "
                "inhibition creates a SECOND downstream ETC block → electrons trapped "
                "between CI (N-module/NDUFS1 Fe-S relay) and CIV → ROS burst → oxidative crisis.\n"
                "Note: NDUFS1 patients with peripheral neuropathy may have respiratory muscle "
                "weakness → require sedation/ventilation more often → PRIS risk elevated.\n"
                "Use sevoflurane (volatile, no PRIS) or dexmedetomidine as alternatives."
            ),
        },
        {
            "drug": "Phenobarbital",
            "severity": "HIGH CAUTION — Secondary Complex I Inhibitor",
            "mechanism": (
                "Phenobarbital inhibits CI electron transport (quinone channel region). "
                "In NDUFS1/CI-Leigh: residual CI (5–20%) is critical. Phenobarbital's "
                "CI inhibition reduces this residual further → lactic decompensation risk.\n"
                "Use LEV (preferred) or clonazepam/CLB (benzodiazepines — no mito toxicity). "
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
                "CI assembly collapses → catastrophic in NDUFS1/CI deficiency."
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
        "Peripheral Neuropathy (DISTINGUISHING)":                         _pct("has_neuropathy"),
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
        "HCM (RARE — slightly > NDUFS4/NDUFV1)":                        _pct("has_hcm"),
        "Hepatopathy (RARE)":                                            _pct("has_hepatopathy"),
        "Olfactory Bulb MRI (NEVER — KEY DDx NDUFS4)":                   0,
        "Leukodystrophy / White Matter (RARELY — KEY DDx NDUFV1)":       _pct("has_leukodystrophy"),
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
            "term": "Valproate / VPA — ABSOLUTE CI in NDUFS1/CI-Leigh",
            "definition": (
                "NEVER use valproate in any patient with NDUFS1 mutation or suspected CI-Leigh.\n\n"
                "Three independent lethal mechanisms:\n"
                "1. CoA SEQUESTRATION (valproyl-CoA): depletes mitochondrial CoA → TCA cycle "
                "   substrates drop → OXPHOS cannot produce ATP → lactic crisis in CI-deficient cell.\n"
                "2. POLG INHIBITION: valproate inhibits mtDNA polymerase gamma → mtDNA depletion "
                "   → fewer MT-ND1–ND6 / MT-ND4L templates → fewer P-module CI subunits "
                "   → assembly of N-module (NDUFS1 Fe-S relay) blocked.\n"
                "3. HEPATOTOXICITY: VPA directly hepatotoxic → hepatic OXPHOS failure "
                "   → impaired gluconeogenesis → hypoglycaemia → crisis amplified.\n\n"
                "Preferred AED: LEVETIRACETAM (LEV) — renal excretion, zero mito toxicity. "
                "Second-line: clobazam (CLB), clonazepam — benzodiazepines, no mito toxicity. "
                "Dystonia: trihexyphenidyl (THP); baclofen. "
                "NEVER: phenytoin, carbamazepine (Na-channel blockers may worsen CI)."
            ),
        },
        {
            "term": "Metformin — ABSOLUTE CI in NDUFS1",
            "definition": (
                "Metformin is a direct Complex I inhibitor. The quinone binding site (ND1 subunit) "
                "of CI is metformin's primary pharmacological target → blocks NADH → UQ electron transfer.\n\n"
                "In NDUFS1 deficiency: this IS the disease locus. CI already at 5–20%.\n"
                "Metformin's CI inhibition removes this residual → near-zero CI activity → "
                "massive NADH accumulation → lactate surge (may exceed 20 mmol/L).\n\n"
                "Pharmacological catastrophe: drug target = disease locus.\n"
                "Never use for glucose management in any CI-Leigh patient.\n"
                "Alternative: insulin (does not interact with OXPHOS)."
            ),
        },
        {
            "term": "Riboflavin / Vitamin B2 — Level C (CI-specific, upstream of NDUFS1)",
            "definition": (
                "Riboflavin → riboflavin-5'-phosphate (FMN) + FAD.\n\n"
                "CI relevance (upstream of NDUFS1):\n"
                "  FMN binds NDUFV1 (the FP1/51kDa/N3 subunit upstream of NDUFS1).\n"
                "  Extra FMN may stabilise NDUFV1 and enhance electron injection at N3,\n"
                "  partially improving flux through the Fe-S relay (N1b→N4→N5 at NDUFS1).\n\n"
                "NOTE: Riboflavin is MORE directly targeted in NDUFV1 deficiency (FMN binds "
                "NDUFV1 active site directly). In NDUFS1 deficiency the mechanism is less direct "
                "but still used empirically. Level C evidence. Dose: 100–400 mg/day.\n\n"
                "Also: riboflavin is essential for FAD-dependent fatty acid oxidation enzymes — "
                "deficiency worsens overall OXPHOS substrate supply."
            ),
        },
        {
            "term": "Succinate — Level C (Complex I Bypass for NDUFS1)",
            "definition": (
                "MECHANISTIC BASIS:\n"
                "  Succinate → Complex II (succinate dehydrogenase / SDHA) → ubiquinol "
                "  → Complex III → CIV → O2.\n"
                "  This bypasses the blocked CI entirely: NDUFS1 Fe-S relay is irrelevant "
                "  for Complex II-mediated electron flow.\n\n"
                "CLINICAL RATIONAL:\n"
                "  In CI deficiency, maintaining ubiquinol pool via CII sustains partial ATP "
                "  synthesis without relying on CI NADH→ubiquinone.\n"
                "  Level C evidence; dose 2–8 g/day oral succinate or sodium succinate.\n"
                "  Most relevant in crisis: IV succinate (where available).\n\n"
                "CAUTION: Not a replacement for glucose (GIR 6-8 during crisis is first-line)."
            ),
        },
        {
            "term": "Peripheral Neuropathy Management in NDUFS1",
            "definition": (
                "NDUFS1-CI-Leigh patients have ~50% peripheral neuropathy — DISTINGUISHING "
                "from NDUFS4 (no neuropathy) and NDUFV1 (no neuropathy).\n\n"
                "No disease-modifying treatment for the neuropathy component is available.\n"
                "Management is supportive:\n"
                "  • Physiotherapy: maintain gait, strength, prevent contractures\n"
                "  • Orthotics: ankle-foot orthoses (AFO) for foot drop; wrist splints\n"
                "  • Pain: gabapentin 5–15 mg/kg/day or pregabalin (watch for sedation)\n"
                "  • NCS/EMG: monitor progression; plan physiotherapy intensity\n"
                "  • Respiratory: axonal neuropathy + CI myopathy → combined respiratory\n"
                "    muscle weakness → spirometry q6 months; low threshold for NIV/BiPAP\n\n"
                "IMPORTANT: Co-existing peripheral neuropathy may mask or worsen the "
                "apparent severity of central Leigh features — full neurological assessment needed."
            ),
        },
        {
            "term": "Thiamine (B1) + Biotin — MANDATORY Empiric in ALL Leigh Presentations",
            "definition": (
                "Before NDUFS1 genotype is confirmed, every infant with Leigh/Leigh-like MRI "
                "+ lactic acidosis + CI biochemistry MUST receive empiric thiamine AND biotin.\n\n"
                "THIAMINE B1:\n"
                "  SLC19A3 deficiency (biotin-thiamine responsive basal ganglia disease / BTBGD) "
                "  and PDHC (pyruvate dehydrogenase complex) deficiency can mimic NDUFS1-Leigh.\n"
                "  Both are TREATABLE if thiamine started early — missed diagnoses have "
                "  caused preventable death/disability.\n"
                "  Dose: 100–300 mg/day IV or oral; safe, negligible toxicity.\n\n"
                "BIOTIN:\n"
                "  Biotinidase (BTD) deficiency and holocarboxylase synthetase deficiency "
                "  cause Leigh-like neurological crisis that responds dramatically to biotin.\n"
                "  Dose: 5–10 mg/day; safe.\n\n"
                "Never withhold empiric thiamine + biotin while awaiting genetics in Leigh syndrome."
            ),
        },
        {
            "term": "Acute Crisis Protocol — NDUFS1/CI-Leigh Metabolic Emergency",
            "definition": (
                "STEP 1 — IV DEXTROSE STAT:\n"
                "  GIR 6-8 mg/kg/min; maximise glucose for residual CI; NEVER fast.\n"
                "  Trigger: clinical deterioration, lactate >5 mmol/L, any fever/illness.\n\n"
                "STEP 2 — HOLD MITOCHONDRIAL TOXINS:\n"
                "  Check for inadvertent metformin, phenobarbital, linezolid, propofol, VPA. "
                "  Stop immediately if any are running.\n\n"
                "STEP 3 — NaHCO3 IV (pH <7.20):\n"
                "  0.5–1 mEq/kg over 1–2h; continuous lactate monitoring q2h; target pH >7.25.\n\n"
                "STEP 4 — IV RIBOFLAVIN + THIAMINE (100 mg each IV if available):\n"
                "  Riboflavin: upstream CI support (FMN at NDUFV1/N3, preceding NDUFS1).\n"
                "  Thiamine: MANDATORY in any acute encephalopathy (SLC19A3/BTD mimic).\n\n"
                "STEP 5 — IV SUCCINATE (metabolic centre, if available):\n"
                "  0.5–1 g/kg/day → CII-mediated bypass of blocked CI.\n\n"
                "STEP 6 — SEIZURES → LEV IV:\n"
                "  20–40 mg/kg loading; simultaneous IV dextrose; ABSOLUTE CI VPA.\n\n"
                "STEP 7 — RESPIRATORY → NIV/BiPAP:\n"
                "  SpO2 <92% or RR >40 → BiPAP; intubation: sevoflurane, NOT propofol.\n"
                "  CAUTION: NDUFS1 peripheral neuropathy may cause rapid respiratory\n"
                "  muscle decompensation — earlier ventilatory support may be needed.\n\n"
                "EMERGENCY CARD:\n"
                "  Carried at all times. ER: IV dextrose immediately; NEVER VPA, metformin, "
                "linezolid, chloramphenicol, propofol, or ketogenic diet. "
                "Contact metabolic neurology (phone on card)."
            ),
        },
    ]

    gene_concepts = [
        {
            "term": "NDUFS1 Gene Structure and Expression",
            "definition": (
                "Gene: NDUFS1 (NADH:Ubiquinone Oxidoreductase Core Subunit S1)\n"
                "Chromosome: 2q33.3\n"
                "Protein: 727 aa precursor; MTS (mitochondrial targeting sequence) ~23 aa; "
                "mature form ~704 aa; molecular weight ~75 kDa\n\n"
                "NDUFS1 is the largest nuclear-encoded subunit of Complex I.\n"
                "It is exclusively expressed in mitochondria and is ubiquitous across tissues "
                "(highest expression: brain, heart, skeletal muscle — tissues with highest "
                "OXPHOS demand, explaining predominant neurological + cardiac phenotype).\n\n"
                "OMIM *157655 (gene); #618226 (disease: Mitochondrial Complex I Deficiency, "
                "Nuclear Type 5 / Leigh syndrome)\n\n"
                "Inheritance: AR — biallelic (compound heterozygous or homozygous) "
                "loss-of-function variants required for disease."
            ),
        },
        {
            "term": "Fe-S Cluster Binding (N1b, N4, N5) — NDUFS1 Role in Electron Relay",
            "definition": (
                "NDUFS1 binds three iron-sulfur clusters that form the core of the "
                "N-module electron relay chain:\n\n"
                "  Fe-S Cluster N1b: [2Fe-2S]; accepts electrons from N1a (NDUFV2/24 kDa)\n"
                "  Fe-S Cluster N4:  [4Fe-4S]; passes electrons from N1b to N5\n"
                "  Fe-S Cluster N5:  [4Fe-4S]; passes electrons to N6a/N6b (Q-module entry)\n\n"
                "The complete relay: NADH → FMN/N3 (NDUFV1) → N1a (NDUFV2) → "
                "N1b/N4/N5 (NDUFS1) → N6a/N6b (NDUFB9/NDUFS8) → N2 (Q-module) → ubiquinone\n\n"
                "NDUFS1 is literally the backbone of this relay — without it, electron "
                "transfer from NDUFV1/FMN to ubiquinone is impossible regardless of whether "
                "NDUFV1 is intact or not. This is why biallelic NDUFS1 variants cause "
                "complete isolated CI deficiency identical to NDUFS4 or NDUFV1."
            ),
        },
        {
            "term": "NDUFS1 vs NDUFS4 vs NDUFV1 — Distinguishing the CI-Leigh Series",
            "definition": (
                "All three cause isolated CI deficiency + Leigh syndrome.\n"
                "Biochemical fingerprint is identical: CI 5–20%, CII/CIII/CIV normal.\n"
                "Clinical differentiation:\n\n"
                "NDUFS4 (175 aa, accessory subunit, 5q11.2):\n"
                "  • Olfactory bulb MRI lesions: ~52–65% (PATHOGNOMONIC of NDUFS4 — NOT seen\n"
                "    in NDUFS1 or NDUFV1)\n"
                "  • Severe respiratory failure (central apnoea) dominant\n"
                "  • NO leukodystrophy, NO neuropathy\n\n"
                "NDUFV1 (464 aa, N-module CORE/FMN, 11q13.2):\n"
                "  • Leukodystrophy / white matter T2: ~40–50% (DISTINGUISHING vs NDUFS4)\n"
                "  • Myoclonus more prominent (~38–40% vs ~22% NDUFS1)\n"
                "  • NO olfactory bulb lesions, NO significant neuropathy\n\n"
                "NDUFS1 (727 aa, IP1/75kDa, 2q33.3):\n"
                "  • Peripheral neuropathy: ~50% (DISTINGUISHING vs NDUFS4/NDUFV1)\n"
                "  • NO olfactory bulb lesions (DDx NDUFS4)\n"
                "  • NO significant leukodystrophy (DDx NDUFV1)\n"
                "  • HCM slightly higher (~12%) than NDUFS4/NDUFV1 (~5%)\n"
                "  • Combined central + peripheral neuroaxonal disease pattern"
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "Mitochondrial Complex I Deficiency Nuclear Type 5 (OMIM #618226)",
            "definition": (
                "NDUFS1 biallelic variants → Mitochondrial Complex I Deficiency, Nuclear Type 5.\n\n"
                "This designation reflects that nuclear-encoded CI subunit variants are the cause "
                "(not mtDNA-encoded ND subunits). 'Type 5' refers to the NDUFS1 gene specifically.\n\n"
                "Biochemical criteria:\n"
                "  Complex I activity: 5–20% of age-matched controls (isolated CI deficiency)\n"
                "  Complexes II, III, IV: NORMAL\n"
                "  Lactate/pyruvate ratio: typically >20 (NADH/NAD+ imbalance)\n"
                "  Plasma lactate: 3–15 mmol/L baseline; crisis >15 mmol/L\n\n"
                "The 'nuclear type' designation is clinically important because:\n"
                "  • AR inheritance (not maternal) — siblings at 25% risk\n"
                "  • De novo variants very rare (unlike mtDNA variants)\n"
                "  • Reproductive options: preimplantation genetic testing (PGT) possible\n"
                "  • mtDNA-based therapies (gene therapy in trial) do NOT apply"
            ),
        },
        {
            "term": "Peripheral Neuropathy in CI-Leigh — NDUFS1 Distinctive Feature",
            "definition": (
                "Peripheral neuropathy occurs in ~50% of NDUFS1-CI-Leigh patients and is the "
                "MOST IMPORTANT clinical distinguisher within the CI-Leigh series.\n\n"
                "Mechanism:\n"
                "  Peripheral axons are long, post-mitotic, and entirely OXPHOS-dependent "
                "  for axonal transport and membrane maintenance. CI deficiency → "
                "  ATP failure → axonal degeneration (length-dependent axonopathy) or "
                "  Schwann cell myelin instability (demyelinating neuropathy).\n\n"
                "NCS/EMG findings:\n"
                "  Axonal: reduced SNAP/CMAP amplitudes; relatively preserved velocities\n"
                "  Demyelinating: slowed velocities; prolonged distal latencies\n"
                "  Mixed: combination picture common in mitochondrial neuropathies\n\n"
                "Clinical impact:\n"
                "  Foot drop, hand weakness, areflexia, sensory loss\n"
                "  Combined with central ataxia/dystonia: severe functional impairment\n"
                "  Respiratory muscle: both CI myopathy AND neuropathy contribute — "
                "  lower threshold for respiratory support"
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Complete Prescribing Safety Card — NDUFS1 CI-Leigh",
            "definition": (
                "ABSOLUTE CONTRAINDICATIONS (never use):\n"
                "  ▪ Valproate / VPA (triple mito-toxicity: CoA/POLG/hepatotoxicity)\n"
                "  ▪ Metformin (direct CI inhibitor — drug target IS disease locus)\n"
                "  ▪ Linezolid (23S rRNA → blocks all 7 mtDNA ND subunits → CI loss)\n"
                "  ▪ Chloramphenicol (same mt-ribosome mechanism as linezolid)\n"
                "  ▪ Ketogenic Diet (forces NADH overload → CI cannot reoxidise)\n\n"
                "HIGH CAUTION / AVOID:\n"
                "  ▪ Propofol (PRIS: CIV inhibition creates 2nd ETC block; risk ↑ in CI)\n"
                "  ▪ Phenobarbital (secondary CI inhibitor; use only if no alternative)\n"
                "  ▪ Aminoglycosides (mt-ribosome risk in susceptible patients)\n"
                "  ▪ Fasting (no fasting >4 h; GIR 6-8 during illness/crisis)\n\n"
                "PREFERRED / SAFE:\n"
                "  ▪ LEV (levetiracetam) — AED first-line; renal; no mito toxicity\n"
                "  ▪ Clonazepam / CLB (clobazam) — benzodiazepines; no mito toxicity\n"
                "  ▪ Sevoflurane — anaesthetic choice (not propofol)\n"
                "  ▪ Dexmedetomidine — sedation alternative to propofol\n"
                "  ▪ Gabapentin / pregabalin — neuropathic pain (peripheral neuropathy)\n"
                "  ▪ Baclofen — spasticity\n"
                "  ▪ Insulin — glucose management (not metformin)"
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
