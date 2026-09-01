#!/usr/bin/env python3
"""NDUFS3 — Leigh Syndrome Isolated Complex I Deficiency (Q-Module 30 kDa QP-C Scaffold Subunit).

NDUFS3 (NADH:Ubiquinone Oxidoreductase Core Subunit S3), also known as the QP-C subunit
(Quinone-binding Protein C, 30 kDa), is a core structural SCAFFOLD subunit of the Q-module
of Complex I. Unlike NDUFS2 (which directly harbours the N2 Fe-S cluster), NDUFS3 serves
as the structural platform that POSITIONS NDUFS2 (N2 terminal Fe-S) and NDUFA9 within the
Q-module membrane-peripheral arm junction. Loss of NDUFS3 → Q-module assembly failure →
isolated CI deficiency → Leigh / Leigh-like syndrome.

  NDUFS3 gene      OMIM *603846
  Disease          Leigh Syndrome (OMIM #256000) /
                   Mitochondrial Complex I Deficiency, Nuclear Type (NDUFS3)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       11p11.11

PATHOPHYSIOLOGY (Complex I / Q-module / NDUFS3 scaffold role):
  The Q-module occupies the membrane-peripheral arm junction of Complex I and
  contains the ubiquinone-binding cavity. Key Q-module subunits:

    NDUFS3 (30 kDa / QP-C) — SCAFFOLD: structural platform for NDUFS2 + NDUFA9
    NDUFS2 (49 kDa / PSST) — N2 [4Fe-4S] terminal Fe-S cluster + ubiquinone reduction
    NDUFA9  (39 kDa)        — NADPH-binding; contacts NDUFS3/NDUFS2 in Q-module

  NDUFS3's critical role in Q-module assembly:
    1. NDUFS3 forms a stable subcomplex with NDUFS2 and NDUFA9 early in CI assembly
    2. NDUFS3 positions NDUFS2/N2 at the correct geometry for ubiquinone reduction
    3. Loss of NDUFS3 → NDUFS2 becomes mispositioned / unstable → N2 cannot deliver
       electrons to ubiquinone despite being structurally intact
    4. ASSEMBLY DEFECT: BN-PAGE shows CI sub-assembly intermediates that stall before
       complete Q-module formation (distinguishes from some other CI defects)

  Biochemical signature (IDENTICAL to NDUFS2/NDUFS4/NDUFV1/NDUFS1 CI fingerprint):
    Complex I: 5–20% of control (severely reduced) — ISOLATED CI DEFICIENCY
    Complex II (SDHA): NORMAL (fully nuclear-encoded; NDUFS3 not required for CII)
    Complex III: NORMAL
    Complex IV (COX): NORMAL — KEY DDx SURF1/SCO2/COX10/COX15 (CIV-only deficiency)

DISTINGUISHING FEATURES vs NDUFS2/NDUFS4/NDUFV1/NDUFS1:
  MECHANISM DIFFERENCE vs NDUFS2 (both Q-module):
    • NDUFS2: DIRECT N2 Fe-S cluster loss (carrier subunit) → electron transfer failure
    • NDUFS3: ASSEMBLY SCAFFOLD loss → N2 mispositioning → Q-module assembly stall
    • BN-PAGE may show CI assembly intermediates in NDUFS3 (less in NDUFS2 direct N2 loss)
    • Clinically indistinguishable without genetics; subtle BN-PAGE differences
  NO Peripheral Neuropathy (vs NDUFS1 ~50% — CRITICAL DDx point)
  NO Olfactory Bulb Lesions (vs NDUFS4 ~52–65% — KEY DDx)
  NO Leukodystrophy (vs NDUFV1 ~40–50% — KEY DDx)
  HCM: ~10% (slightly higher than NDUFS2 ~8%; much less than SCO2 100%)
  Q-Module Assembly Failure Fingerprint:
    • Fibroblast BN-PAGE: CI assembly intermediates present (sub-880 kDa bands)
    • Distinguishes from N2 direct loss (NDUFS2) on specialized native gel analysis

FOUNDER / RECURRENT MUTATIONS:
  p.Arg199Cys   c.595C>T  — recurrent; European/Dutch consanguineous families;
                             first reported in human CI deficiency; moderate-to-severe Leigh
  p.Arg199His   c.596G>A  — same codon (Arg199), different substitution; severe
  p.Glu74Lys    c.220G>A  — compound het with truncating allele; severe infantile Leigh
  Frameshift / nonsense (compound het) — complete CI assembly failure; severe Leigh

THERAPY — NDUFS3/CI-LEIGH SPECIFICS:
  No targeted NDUFS3 scaffold stabilisation is clinically available.
  Management follows the CI-Leigh supportive protocol (cofactors + avoid CI toxins).
  Succinate bypass (Level C): CII-mediated electron entry to ubiquinol bypasses blocked
    Q-module entirely — does NOT require NDUFS3/NDUFS2/N2 pathway.
  Riboflavin B2 (Level C): FMN reinforces NDUFV1/N3 upstream; less direct.
  Physiotherapy: no peripheral neuropathy — focuses on central ataxia/dystonia.

References:
  Bénit P et al. Hum Genet. 2003;113(1):32-37.
    (NDUFS3 compound hets in human CI deficiency; genotype-phenotype)
  Loeffen JL et al. Ann Neurol. 2001;49(2):195-201.
    (Early CI subunit mutations including Q-module subunits)
  Fassone E, Rahman S. J Med Genet. 2012;49(9):578-590.
    (CI deficiency genetics review — NDUFS3 in context of Q-module series)
  Sazanov LA. Nat Rev Mol Cell Biol. 2015;16(6):375-388.
    (CI structural review — NDUFS3 Q-module scaffold position)
  Rötig A, Munnich A. Hum Mutat. 2003;21(6):607-614.
    (CI deficiency spectrum including NDUFS3)
  Fernandez-Vizarra E, Zeviani M. Hum Mol Genet. 2021;30(R2):R181-R197.
    (CI assembly — NDUFS3/NDUFS2 Q-module subcomplex)
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 615
DISEASE_ID   = "ndufs3"
DISEASE_NAME = (
    "NDUFS3 Leigh Syndrome — Isolated Complex I Deficiency "
    "(CI-Leigh / NDUFS3 Q-Module 30 kDa QP-C Scaffold Subunit)"
)
GENE         = "NDUFS3"
PROTEIN      = (
    "NDUFS3 — 264 aa precursor / ~235 aa mature, Q-module QP-C scaffold subunit (30 kDa), "
    "positions NDUFS2/N2 and NDUFA9; Q-module assembly platform"
)
OMIM_GENE    = "*603846"
OMIM_DISEASE = "#256000 (Leigh Syndrome) / Mitochondrial Complex I Deficiency Nuclear Type (NDUFS3)"
CHROMOSOME   = "11p11.11"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = (
    "Infantile (3–18 months); rarely neonatal; delayed onset up to 3 yr in partial-assembly missense alleles"
)
COHORT_SIZE  = 40
COLOR        = "#1565c0"   # deep blue — Q-module scaffold / NDUFS3 assembly theme
LIGHT        = "#e3f2fd"

# Genotype pool
GENO_RCYS  = "p.Arg199Cys (c.595C>T) / truncating (compound het) — recurrent; European/Dutch; moderate-severe"
GENO_RHIS  = "p.Arg199His (c.596G>A) / truncating (compound het) — same codon; severe; multi-ethnic"
GENO_ELYS  = "p.Glu74Lys (c.220G>A) / truncating (compound het) — severe infantile; scaffold base"
GENO_NULL  = "Frameshift / nonsense (compound het) — complete CI assembly failure; severe Leigh"
GENO_MIS2  = "Missense + splice / compound het — variable severity; partial Q-module assembly"

GENO_POOL    = [GENO_RCYS, GENO_RHIS, GENO_ELYS, GENO_NULL, GENO_MIS2]
GENO_WEIGHTS = [0.30,      0.22,      0.18,       0.16,      0.14]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG — reproducible 40-patient NDUFS3/CI-Leigh cohort (seed-615)."""
    return random.Random(SEED)


# ── Patient cohort ────────────────────────────────────────────────────────────
_TX_POOL = [
    "IV Dextrose GIR 6-8 (metabolic crisis — first-line)",
    "Riboflavin B2 (100–400 mg/day — CI-specific FMN precursor; upstream of Q-module)",
    "CoQ10/Ubiquinol (300–600 mg/day)",
    "Thiamine B1 (mandatory empiric, all Leigh until SLC19A3/BTD excluded)",
    "Biotin (mandatory empiric, all Leigh until BTD ruled out)",
    "Succinate (IV/oral — Complex II bypass; bypasses blocked NDUFS3 Q-module entirely)",
    "NaHCO3 (lactic acidosis correction, pH <7.20)",
    "LEV (seizures — preferred AED, renal excretion, no mito toxicity)",
    "NIV/BiPAP (central respiratory compromise)",
    "Carnitine (secondary deficiency, 50–100 mg/kg/day)",
    "Physiotherapy / occupational therapy (central ataxia + dystonia — no peripheral neuropathy)",
    "Enteral feeds / NG (avoid fasting; high-carbohydrate; KD strictly CI)",
]

_OUTCOMES = [
    "Alive — stable moderate-to-severe Leigh; cofactor + respiratory support",
    "Alive — prolonged survival with missense alleles; partial CI assembly; school-age",
    "Alive — severe Leigh; dependent care; ongoing multidisciplinary support",
    "Died — acute lactic crisis + respiratory failure in first 18 months (null alleles)",
    "Died — progressive brainstem Leigh + respiratory failure; 2–4 yr trajectory",
]
_OUT_WEIGHTS = [0.27, 0.14, 0.21, 0.22, 0.16]


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

        has_leigh_mri       = rng.random() < 0.82
        has_regression      = rng.random() < 0.95
        has_hypotonia       = rng.random() < 0.85
        has_lactic          = rng.random() < 0.88
        has_resp            = rng.random() < 0.58
        has_seizures        = rng.random() < 0.45
        has_ataxia          = rng.random() < 0.40
        has_dystonia        = rng.random() < 0.35
        has_optic           = rng.random() < 0.25
        has_myoclonus       = rng.random() < 0.18
        has_nystagmus       = rng.random() < 0.22
        has_spasticity      = rng.random() < 0.32
        has_hcm             = rng.random() < 0.10    # ~10%, slightly higher than NDUFS2 8%
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
            "id":                  f"NDUFS3-{i:03d}",
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
        "HCM (~10% — slightly higher than NDUFS2 8%; much less than SCO2 100%)": _pct("has_hcm"),
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
        {"label": "Leigh MRI",        "value": f"{feature_frequencies['Leigh / Leigh-like MRI (bilateral putamen + brainstem)']}%", "color": "#4a148c"},
        {"label": "No Neuropathy",    "value": "100% — DDx NDUFS1",                                                                "color": "#2e7d32"},
        {"label": "Resp Compromise",  "value": f"{feature_frequencies['Respiratory Compromise (central)']}%",                      "color": "#b71c1c"},
        {"label": "Hypotonia",        "value": f"{feature_frequencies['Hypotonia']}%",                                             "color": COLOR},
        {"label": "CI Activity",      "value": "5–21% control",                                                                    "color": "#c62828"},
        {"label": "Fatal",            "value": f"{round(died/COHORT_SIZE*100)}%",                                                  "color": "#b71c1c"},
        {"label": "Seed",             "value": f"#{SEED}",                                                                        "color": "#455a64"},
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "severity": "ABSOLUTE CI — DANGEROUS IN ALL LEIGH / CI DEFICIENCY",
            "mechanism": (
                "Triple mechanism in NDUFS3/CI-Leigh:\n"
                "1. CoA SEQUESTRATION: valproyl-CoA depletes the mitochondrial CoA pool "
                "   → TCA cycle collapses → OXPHOS substrate loss. In NDUFS3 deficiency "
                "   (CI already at 5–20% due to Q-module assembly failure), CoA depletion "
                "   tips the patient into irreversible lactic crisis.\n"
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
            "severity": "ABSOLUTE CI — DIRECT COMPLEX I INHIBITOR: CATASTROPHIC IN NDUFS3",
            "mechanism": (
                "Metformin inhibits Complex I (NADH dehydrogenase) at the ND1/quinone "
                "binding site.\n"
                "In NDUFS3 deficiency: CI Q-module is already assembly-defective (5–20% "
                "activity). Metformin's CI inhibition further suppresses this residual "
                "capacity → near-total CI shutdown → massive lactate accumulation (>15–20 mmol/L).\n\n"
                "NDUFS3 loss destabilises NDUFS2/N2 positioning. Metformin then blocks "
                "the residual N2→ubiquinone transfer that still occurs → compounded disaster.\n"
                "Alternative for glucose management: insulin only. Never metformin."
            ),
        },
        {
            "drug": "Linezolid",
            "severity": "ABSOLUTE CI — mt-Ribosome 23S rRNA Block (ALL 7 ND mtDNA Subunits)",
            "mechanism": (
                "Linezolid inhibits the mitochondrial large ribosomal subunit (mt-LSU) 23S rRNA "
                "→ blocks synthesis of ALL 13 mitochondrially-encoded OXPHOS subunits.\n"
                "In NDUFS3/CI-Leigh:\n"
                "  • NDUFS3 itself is nuclear-encoded — BUT Complex I also requires 7 mtDNA-encoded "
                "    ND subunits (MT-ND1–6, MT-ND4L) for the P-module membrane arm\n"
                "  • Linezolid blocks synthesis of ALL 7 → CI P-module cannot assemble "
                "    → Q-module scaffold (NDUFS3) has no functional context → CI → zero\n"
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
                "In NDUFS3 CI-Leigh: CI Q-module is assembly-defective (5–20% activity). "
                "Forcing KD → NADH accumulation → NAD+ depletion → TCA + beta-oxidation "
                "stall → worsened lactic acidosis.\n\n"
                "KD is beneficial in GLUT1-DS and PDHD — CONTRAINDICATED in CI deficiency."
            ),
        },
        {
            "drug": "Propofol",
            "severity": "AVOID / HIGH CAUTION — PRIS in CI-Leigh",
            "mechanism": (
                "Propofol Infusion Syndrome (PRIS): propofol inhibits Complex IV and "
                "uncouples fatty acid beta-oxidation.\n"
                "In NDUFS3/CI-Leigh: CI is the primary ETC bottleneck. Propofol's CIV "
                "inhibition creates a SECOND downstream ETC block → electrons trapped "
                "between assembly-defective CI (NDUFS3 Q-module) and CIV → ROS burst.\n"
                "Use sevoflurane (volatile, no PRIS) or dexmedetomidine as alternatives."
            ),
        },
        {
            "drug": "Phenobarbital",
            "severity": "HIGH CAUTION — Secondary Complex I Inhibitor",
            "mechanism": (
                "Phenobarbital inhibits CI electron transport (quinone channel region). "
                "In NDUFS3/CI-Leigh: residual CI (5–20%) is critical. Phenobarbital's "
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
                "CI assembly collapses → catastrophic in NDUFS3/CI deficiency."
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
        "HCM (RARE — ~10%, slightly higher than NDUFS2 ~8%)":             _pct("has_hcm"),
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
            "term": "Valproate / VPA — ABSOLUTE CI in NDUFS3/CI-Leigh",
            "definition": (
                "NEVER use valproate in any patient with NDUFS3 mutation or suspected CI-Leigh.\n\n"
                "Three independent lethal mechanisms:\n"
                "1. CoA SEQUESTRATION (valproyl-CoA): depletes mitochondrial CoA → TCA cycle "
                "   substrates drop → OXPHOS cannot produce ATP → lactic crisis in CI-deficient cell.\n"
                "2. POLG INHIBITION: valproate inhibits mtDNA polymerase gamma → mtDNA depletion "
                "   → fewer MT-ND1–ND6 / MT-ND4L templates → fewer P-module CI subunits "
                "   → assembly of Q-module (NDUFS3 scaffold + NDUFS2/N2) further blocked.\n"
                "3. HEPATOTOXICITY: VPA directly hepatotoxic → hepatic OXPHOS failure "
                "   → impaired gluconeogenesis → hypoglycaemia → crisis amplified.\n\n"
                "Preferred AED: LEVETIRACETAM (LEV) — renal excretion, zero mito toxicity. "
                "Second-line: clobazam (CLB), clonazepam — no mito toxicity."
            ),
        },
        {
            "term": "Metformin — ABSOLUTE CI in NDUFS3",
            "definition": (
                "Metformin is a direct Complex I inhibitor. The ND1/quinone binding site of CI "
                "is metformin's primary pharmacological target — in the same Q-module cavity "
                "that NDUFS3 scaffolds and NDUFS2/N2 occupies.\n\n"
                "In NDUFS3 deficiency: Q-module is already assembly-defective. CI at 5–20%.\n"
                "Metformin's CI inhibition removes this residual → near-zero CI activity → "
                "massive NADH accumulation → lactate surge (may exceed 20 mmol/L).\n\n"
                "Never use for glucose management in any CI-Leigh patient.\n"
                "Alternative: insulin (does not interact with OXPHOS)."
            ),
        },
        {
            "term": "Succinate — Level C (Complex I Bypass for NDUFS3 Q-Module Failure)",
            "definition": (
                "MECHANISTIC BASIS:\n"
                "  Succinate → Complex II (succinate dehydrogenase / SDHA) → ubiquinol "
                "  → Complex III → CIV → O2.\n"
                "  This bypasses the NDUFS3 Q-module assembly failure ENTIRELY: neither "
                "  NDUFS3 (scaffold) nor NDUFS2/N2 (terminal Fe-S) are required for CII→ubiquinol.\n\n"
                "CLINICAL RATIONALE:\n"
                "  In CI deficiency, maintaining ubiquinol pool via CII sustains partial ATP "
                "  synthesis without relying on the NDUFS3/NDUFS2/N2 pathway.\n"
                "  Level C evidence; dose 2–8 g/day oral succinate or sodium succinate.\n"
                "  Most relevant in crisis: IV succinate (where available).\n\n"
                "CAUTION: Not a replacement for glucose (GIR 6-8 during crisis is first-line)."
            ),
        },
        {
            "term": "Riboflavin / Vitamin B2 — Level C (CI-specific, upstream of Q-module)",
            "definition": (
                "Riboflavin → riboflavin-5'-phosphate (FMN) + FAD.\n\n"
                "CI relevance (upstream of NDUFS3 Q-module):\n"
                "  FMN binds NDUFV1 (51kDa/N3 subunit) — the FIRST electron acceptor from NADH.\n"
                "  Extra FMN may stabilise NDUFV1 and improve electron injection at N3,\n"
                "  potentially enhancing flux through the Fe-S relay toward the Q-module.\n\n"
                "NOTE: Riboflavin is more directly targeted in NDUFV1 deficiency. In NDUFS3 "
                "deficiency, the Q-module scaffold cannot be repaired pharmacologically. "
                "Riboflavin used empirically. Level C.\n"
                "Dose: 100–400 mg/day."
            ),
        },
        {
            "term": "Thiamine (B1) + Biotin — MANDATORY Empiric in ALL Leigh Presentations",
            "definition": (
                "Before NDUFS3 genotype is confirmed, every infant with Leigh/Leigh-like MRI "
                "+ lactic acidosis + CI biochemistry MUST receive empiric thiamine AND biotin.\n\n"
                "THIAMINE B1:\n"
                "  SLC19A3 deficiency (biotin-thiamine responsive basal ganglia disease / BTBGD) "
                "  and PDHC deficiency can mimic NDUFS3-Leigh.\n"
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
            "term": "Acute Crisis Protocol — NDUFS3/CI-Leigh Metabolic Emergency",
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
                "  0.5–1 g/kg/day → CII-bypass of NDUFS3 Q-module assembly failure.\n\n"
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
            "term": "NDUFS3 Gene Structure and Expression",
            "definition": (
                "Gene: NDUFS3 (NADH:Ubiquinone Oxidoreductase Core Subunit S3)\n"
                "Also known as: QP-C (Quinone-binding Protein C, 30 kDa subunit)\n"
                "Chromosome: 11p11.11\n"
                "Protein: 264 aa precursor; MTS ~29 aa; mature form ~235 aa; ~30 kDa\n\n"
                "NDUFS3 is a core subunit of the Q-module (peripheral arm, membrane interface).\n"
                "Ubiquitous expression across tissues; highest in brain, heart, skeletal muscle.\n\n"
                "OMIM *603846 (gene); #256000 (Leigh Syndrome) / CI Deficiency (NDUFS3)\n\n"
                "Inheritance: AR — biallelic (compound heterozygous or homozygous) "
                "loss-of-function variants required for disease."
            ),
        },
        {
            "term": "NDUFS3 Q-Module Scaffold Role — Assembly Platform for NDUFS2/N2 and NDUFA9",
            "definition": (
                "NDUFS3 serves as the structural SCAFFOLD of the Q-module — distinct from "
                "NDUFS2 which directly carries the N2 Fe-S cluster.\n\n"
                "NDUFS3 structural functions:\n"
                "  1. Forms a stable trimeric subcomplex with NDUFS2 (N2/49kDa) + NDUFA9 (39kDa)\n"
                "  2. Positions NDUFS2/N2 at the correct geometry for ubiquinone reduction\n"
                "  3. Anchors the Q-module to the peripheral arm membrane junction\n"
                "  4. NDUFA9 binding via NDUFS3 connects the NADPH regulatory site to Q-module\n\n"
                "NDUFS3 vs NDUFS2 — same Q-module, different consequences:\n"
                "  NDUFS2 loss → N2 Fe-S cluster directly absent → terminal electron transfer block\n"
                "  NDUFS3 loss → N2 still present but mispositioned → Q-module assembly stall\n"
                "  BN-PAGE distinguishes: NDUFS3 loss → CI sub-assembly intermediates visible\n"
                "  Both → identical biochemical result: isolated CI deficiency (5–20%)"
            ),
        },
        {
            "term": "NDUFS3 vs NDUFS2 vs NDUFS1 vs NDUFV1 vs NDUFS4 — CI-Leigh Series",
            "definition": (
                "All five cause isolated CI deficiency + Leigh syndrome.\n"
                "Biochemical fingerprint is identical: CI 5–20%, CII/CIII/CIV normal.\n"
                "Clinical differentiation:\n\n"
                "NDUFS4 (175 aa, accessory, 5q11.2):\n"
                "  • Olfactory bulb MRI: ~52–65% (PATHOGNOMONIC — not seen in NDUFS3)\n"
                "  • Severe central apnoea dominant\n\n"
                "NDUFV1 (464 aa, N-module FMN, 11q13.2):\n"
                "  • Leukodystrophy / white matter T2: ~40–50% (DISTINGUISHING)\n"
                "  • Myoclonus more prominent (~38–40%)\n\n"
                "NDUFS1 (727 aa, IP1/75kDa, 2q33.3):\n"
                "  • Peripheral neuropathy: ~50% (DISTINGUISHING — not seen in NDUFS3)\n"
                "  • HCM slightly higher (~12%)\n\n"
                "NDUFS2 (463 aa, PSST/N2/49kDa, 1q23.3):\n"
                "  • Direct N2 Fe-S carrier loss; HCM ~8%\n"
                "  • Clinically identical to NDUFS3; BN-PAGE may differ\n\n"
                "NDUFS3 (264 aa, QP-C/30kDa, 11p11.11) — THIS DISEASE:\n"
                "  • Q-module ASSEMBLY SCAFFOLD (positions NDUFS2/N2 + NDUFA9)\n"
                "  • NO peripheral neuropathy (KEY DDx vs NDUFS1)\n"
                "  • NO olfactory bulb lesions (KEY DDx vs NDUFS4)\n"
                "  • NO leukodystrophy (KEY DDx vs NDUFV1)\n"
                "  • HCM ~10% (slightly higher than NDUFS2 8%)\n"
                "  • BN-PAGE: CI sub-assembly intermediates (scaffold failure pattern)"
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "NDUFS3 Q-Module Assembly Failure and CI Deficiency",
            "definition": (
                "NDUFS3 biallelic variants → loss of Q-module scaffold → CI assembly failure.\n\n"
                "The assembly failure cascade:\n"
                "  1. NDUFS3 absent → NDUFS2/N2 cannot be correctly positioned\n"
                "  2. NDUFA9 binding disrupted → Q-module subcomplex unstable\n"
                "  3. Q-module fails to join with P-module (membrane arm) + N-module\n"
                "  4. CI holoenzyme assembly stalls → CI 5–20% residual (partial assembly)\n"
                "  5. Residual NADH cannot be re-oxidised at full rate → NADH/NAD+ ↑\n"
                "  → lactate/pyruvate ratio ↑ → lactic acidosis\n\n"
                "Biochemical criteria:\n"
                "  Complex I activity: 5–20% of age-matched controls\n"
                "  Complexes II, III, IV: NORMAL\n"
                "  Lactate/pyruvate ratio: typically >20\n"
                "  Plasma lactate: 3–15 mmol/L baseline; crisis >15 mmol/L\n"
                "  BN-PAGE: CI sub-assembly intermediates (<880 kDa bands) — NDUFS3 specific\n\n"
                "Inheritance: AR — biallelic. Siblings at 25% risk."
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Complete Prescribing Safety Card — NDUFS3 CI-Leigh",
            "definition": (
                "ABSOLUTE CONTRAINDICATIONS (never use):\n"
                "  ▪ Valproate / VPA (triple mito-toxicity: CoA/POLG/hepatotoxicity)\n"
                "  ▪ Metformin (direct CI inhibitor at ND1/quinone-binding Q-module site)\n"
                "  ▪ Linezolid (23S rRNA → blocks all 7 mtDNA ND subunits → CI loss)\n"
                "  ▪ Chloramphenicol (same mt-ribosome mechanism as linezolid)\n"
                "  ▪ Ketogenic Diet (forces NADH overload → assembly-failed Q-module worsened)\n\n"
                "HIGH CAUTION / AVOID:\n"
                "  ▪ Propofol (PRIS: CIV inhibition creates 2nd ETC block downstream of NDUFS3)\n"
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
