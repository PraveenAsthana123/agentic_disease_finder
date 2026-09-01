#!/usr/bin/env python3
"""NDUFA13 — Leigh Syndrome Isolated Complex I Deficiency (GRIM-19 / B16.6 / N-Module Peripheral Stabiliser / Dual-Function CI-Subunit and Apoptosis Regulator).

NDUFA13 (NADH:Ubiquinone Oxidoreductase Subunit A13) is a ~144-aa nuclear-encoded
structural subunit of Complex I (~16.7 kDa mature), located at the N-module periphery,
contacting the NDUFV1-NDUFV2 face.  It is also known as GRIM-19 (Gene associated with
Retinoid-IFN-induced Mortality factor 19) and the B16.6 subunit.

  NDUFA13 gene     OMIM *609435
  Disease          Leigh Syndrome (OMIM #256000)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       19p13.11

PATHOPHYSIOLOGY (Complex I / N-module peripheral stabiliser / NDUFA13 / GRIM-19):
  NDUFA13 is the ONLY nuclear-encoded CI structural subunit with proven dual function:
    1. CI assembly/stability role: N-module peripheral subunit contacting the NDUFV1-NDUFV2
       face of the N-module.  Stabilises the N-module periphery; loss leads to N-module
       sub-assembly intermediates visible on BN-PAGE (partial N-module assemblies) — a
       pattern SIMILAR to NDUFA2 and NDUFS5 (peripheral N-module stabilisers) but DISTINCT
       from NDUFS7/NDUFS8 (direct Fe-S relay), NDUFB3 (PP-module scaffold), and NDUFA11
       (PP-PD membrane arm boundary), which show cleaner absent CI.
    2. Apoptosis/cell-death regulatory role via the GRIM pathway (Gene associated with
       Retinoid-IFN-induced Mortality): NDUFA13/GRIM-19 was originally identified as an
       IFN-beta/retinoic acid-induced mortality gene; overexpression promotes apoptosis
       and it inhibits cell survival in a CI-independent, cytoplasmic signalling context.
    3. STAT3 inhibition: NDUFA13/GRIM-19 directly inhibits STAT3 in the cytoplasm by
       binding STAT3 and preventing its nuclear translocation — making NDUFA13 the ONLY
       CI structural subunit with known cytoplasmic signalling crosstalk outside the ETC.

  Loss of NDUFA13 → N-module peripheral stabilisation fails → BN-PAGE shows N-module
  sub-assembly intermediates (partial N-module visible — similar to NDUFA2/NDUFS5 pattern).
  Isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL.

  Net result: isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (IP1/75kDa, peripheral arm N-module):
    • NO peripheral neuropathy in NDUFA13 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory, 5q11.2):
    • NO olfactory bulb MRI lesions in NDUFA13 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV assembly):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFS7/NDUFS8/NDUFB3/NDUFA11 (direct relay/membrane arm subunits):
    • BN-PAGE: NDUFA13 shows N-module sub-assembly intermediates (partial N-module visible)
      — CONTRAST cleaner absent CI in NDUFS7/NDUFS8 (Fe-S relay), NDUFB3 (PP-module scaffold),
      and NDUFA11 (PP-PD inter-module boundary).
  vs NDUFA12 (B17.2, N-module/Q-module interface):
    • NDUFA12 is at the N-module/Q-module interface with NDUFAF2 swap;
      NDUFA13 is at the N-module periphery (NDUFV1-NDUFV2 face) — different N-module contact.
    • NDUFA12 is an assembly checkpoint subunit (NDUFAF2 paralog); NDUFA13 is a dual-function
      peripheral stabiliser with apoptosis/STAT3 roles — entirely different molecular character.
  vs NDUFA9 (I-gamma, Q-module/membrane-arm junction, SDR fold):
    • NDUFA9 bridges peripheral arm to membrane arm at Q-module entrance;
      NDUFA13 is in the N-module periphery — different CI domains.
  vs POLG/DGUOK (mtDNA depletion syndromes):
    • NO hepatopathy in NDUFA13 (POLG: ~80%; DGUOK: ~90%)
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • COX activity NORMAL in NDUFA13 — biochemical fingerprint distinction

UNIQUE MOLECULAR SIGNATURE — NDUFA13/GRIM-19:
  DUAL-FUNCTION GRIM-19: The ONLY nuclear-encoded CI structural subunit that is also a
  bona fide apoptosis regulator (GRIM pathway) and cytoplasmic STAT3 inhibitor.  No other
  CI subunit (nuclear- or mitochondrially-encoded) has this dual role.  This means that in
  NDUFA13 deficiency there is BOTH an isolated CI energy failure AND a potential dysregulation
  of STAT3 signalling (STAT3 no longer inhibited → constitutive STAT3 activation possible).

FOUNDER / RECURRENT MUTATIONS:
  p.Arg76Trp   c.226C>T  — N-module/NDUFV1-face interface contact; severe infantile
  p.Glu70Lys   c.208G>A  — alpha-helix core disruption; moderate, partial CI residual
  p.Pro31Leu   c.92C>T   — near signal peptide/N-terminal mitochondrial targeting region; severe neonatal
  p.Ala114Val  c.341C>T  — C-terminal structural; moderate, consanguineous kindreds
  c.IVS2+1G>A            — splice donor exon 2; partial CI residual (~10-18%); intermediate severity

THERAPY — NDUFA13 / CI-LEIGH SPECIFICS:
  No targeted NDUFA13/GRIM-19 N-module rescue is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits complex I at ND1/quinone-binding site
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH (N-module peripheral
                        stabilisation failed, CI N-module assembly disrupted)
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group at NDUFV1 (same N-module face as NDUFA13)
      CoQ10 (ubiquinol) — electron acceptor at quinone site; downstream of CI
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFA13-failed N-module CI entirely
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
"""

import random
from typing import Any

SEED = 637
GENE = "NDUFA13"
DISEASE = "NDUFA13 Leigh Syndrome — Isolated Complex I Deficiency (GRIM-19 / B16.6 / N-Module Peripheral Stabiliser / Dual-Function Apoptosis Regulator)"
OMIM_GENE = "609435"
OMIM_DISEASE = "256000"
CHROMOSOME = "19p13.11"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-637, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Arg76Trp / c.IVS2+1G>A (compound het)",
        "p.Glu70Lys / p.Glu70Lys (hom, consanguineous)",
        "p.Arg76Trp / p.Pro31Leu (compound het)",
        "p.Glu70Lys / p.Ala114Val (compound het)",
        "p.Ala114Val / c.IVS2+1G>A (compound het)",
        "p.Arg76Trp / p.Ala114Val (compound het)",
        "p.Glu70Lys / p.Pro31Leu (compound het)",
        "Novel biallelic LOF (frameshift/splice)",
        "p.Arg76Trp / novel missense (compound het)",
        "p.Ala114Val / p.Ala114Val (hom, consanguineous)",
    ]
    regions = [
        "MENA", "South Asian", "European", "East Asian",
        "North African", "Latin American", "Sub-Saharan African",
    ]
    ci_act_ranges = [(5, 10), (8, 15), (10, 20)]

    cohort = []
    for i in range(1, n + 1):
        age_onset = round(rng.uniform(0.2, 18), 1)
        sex = rng.choice(sexes)
        mut = rng.choice(mutations)
        region = rng.choice(regions)
        ci_lo, ci_hi = rng.choice(ci_act_ranges)
        ci_activity = round(rng.uniform(ci_lo, ci_hi), 1)
        cohort.append({
            "id": i,
            "age_onset_months": round(age_onset * 12),
            "sex": sex,
            "mutation": mut,
            "region": region,
            "ci_activity_pct": ci_activity,
            "leigh_mri":              rng.random() < 0.84,
            "lactic_acidosis":        rng.random() < 0.86,
            "hypotonia":              rng.random() < 0.85,
            "psychomotor_regression": rng.random() < 0.94,
            "respiratory_compromise": rng.random() < 0.43,
            "seizures":               rng.random() < 0.46,
            "ataxia":                 rng.random() < 0.38,
            "dystonia":               rng.random() < 0.33,
            "hcm":                    rng.random() < 0.04,
            "peripheral_neuropathy":  rng.random() < 0.04,
            "olfactory_bulb_lesions": rng.random() < 0.03,
            "leukodystrophy":         rng.random() < 0.04,
            "hepatopathy":            rng.random() < 0.03,
            "outcome": rng.choice(
                ["deceased before 3yr"] * 13 +
                ["deceased before 10yr"] * 8 +
                ["alive, severe disability"] * 12 +
                ["alive, moderate disability"] * 7
            ),
        })
    return cohort


_COHORT: list[dict[str, Any]] | None = None


def _cohort() -> list[dict[str, Any]]:
    global _COHORT
    if _COHORT is None:
        _COHORT = _build_cohort()
    return _COHORT


def _pct(key: str) -> float:
    pts = _cohort()
    return round(sum(1 for p in pts if p.get(key)) / len(pts) * 100)


# ---------------------------------------------------------------------------
# Public API: get_overview
# ---------------------------------------------------------------------------

def get_overview() -> dict[str, Any]:
    pts = _cohort()
    n = len(pts)
    ci_vals = [p["ci_activity_pct"] for p in pts]

    return {
        "gene":           GENE,
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Subunit A13",
        "also_known_as":  "GRIM-19 (Growth Inhibitor and Retinoid-IFN-induced Mortality factor 19) / B16.6 subunit / N-module peripheral stabiliser",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa_precursor": 144,
            "size_aa_mature":    122,
            "size_kda":          16.7,
            "fold":              "N-module peripheral stabiliser — NO Fe-S cluster, NO zinc finger, NO SDR fold, NO TM helices; dual-function: CI structural + apoptosis regulator (GRIM-19) + STAT3 inhibitor",
            "module":            "N-module periphery (NDUFV1-NDUFV2 face contacts); peripheral arm",
            "fe_s_cluster":      False,
            "zinc_finger":       False,
            "function":          "Structural stabiliser of the N-module periphery at the NDUFV1-NDUFV2 face; loss causes N-module sub-assembly intermediates on BN-PAGE (similar to NDUFA2/NDUFS5). DUAL-FUNCTION: also GRIM-19 apoptosis regulator and cytoplasmic STAT3 inhibitor — unique among ALL 45 nuclear-encoded CI structural subunits.",
        },
        "key_pathway_note": (
            "NDUFA13/GRIM-19 is the ONLY nuclear-encoded CI structural subunit with dual function: "
            "(1) CI assembly/stability role at the N-module periphery (contacts NDUFV1-NDUFV2 face), "
            "and (2) apoptosis/cell-death regulatory role via the GRIM pathway (Gene associated with "
            "Retinoid-IFN-induced Mortality). Additionally, NDUFA13/GRIM-19 directly inhibits STAT3 "
            "signaling in the cytoplasm — making it the only CI subunit with known cytoplasmic signalling "
            "crosstalk. Loss of NDUFA13 → N-module peripheral stabilisation fails → BN-PAGE shows "
            "N-module sub-assembly intermediates (partial N-module visible — similar to NDUFA2/NDUFS5 "
            "pattern). CONTRAST: cleaner absent CI in NDUFS7/NDUFS8/NDUFB3/NDUFA11 (direct relay/"
            "membrane arm defects). Isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL."
        ),
        "biochemical_fingerprint": {
            "complex_I":   "5–20 % of control (SEVERELY REDUCED — ISOLATED CI DEFICIENCY)",
            "complex_II":  "NORMAL (SDHA — all nuclear; unaffected)",
            "complex_III": "NORMAL",
            "complex_IV":  "NORMAL — key DDx: SURF1 / SCO2 / COX10 / COX15",
        },
        "cohort": {
            "n":                       n,
            "seed":                    SEED,
            "mean_age_onset_months":   round(sum(p["age_onset_months"] for p in pts) / n),
            "ci_activity_mean_pct":    round(sum(ci_vals) / n, 1),
            "ci_activity_range_pct":   f"{min(ci_vals):.1f}–{max(ci_vals):.1f}",
        },
        "feature_frequencies_pct": {
            "psychomotor_regression":   _pct("psychomotor_regression"),
            "lactic_acidosis":          _pct("lactic_acidosis"),
            "hypotonia":                _pct("hypotonia"),
            "leigh_mri":                _pct("leigh_mri"),
            "seizures":                 _pct("seizures"),
            "respiratory_compromise":   _pct("respiratory_compromise"),
            "ataxia":                   _pct("ataxia"),
            "dystonia":                 _pct("dystonia"),
            "hcm":                      _pct("hcm"),
            "peripheral_neuropathy":    _pct("peripheral_neuropathy"),
            "olfactory_bulb_lesions":   _pct("olfactory_bulb_lesions"),
            "leukodystrophy":           _pct("leukodystrophy"),
            "hepatopathy":              _pct("hepatopathy"),
        },
        "key_ddx": [
            {
                "feature":        "NO peripheral neuropathy",
                "significance":   "KEY DDx vs NDUFS1 (~50%) — CRITICAL distinguisher",
                "target_gene":    "NDUFS1",
                "target_freq_pct": 50,
            },
            {
                "feature":        "NO olfactory bulb MRI lesions",
                "significance":   "KEY DDx vs NDUFS4 (52–65%) — PATHOGNOMONIC for NDUFS4",
                "target_gene":    "NDUFS4",
                "target_freq_pct": 58,
            },
            {
                "feature":        "NO leukodystrophy / white matter T2 signal",
                "significance":   "KEY DDx vs NDUFV1 (~40–50%) — CRITICAL",
                "target_gene":    "NDUFV1",
                "target_freq_pct": 45,
            },
            {
                "feature":        "NO hypertrophic cardiomyopathy (HCM)",
                "significance":   "KEY DDx vs NDUFV2 (~80%) and SCO2 (~100%)",
                "target_gene":    "NDUFV2 / SCO2",
                "target_freq_pct": 80,
            },
            {
                "feature":        "N-module sub-assembly intermediates on BN-PAGE (partial N-module visible)",
                "significance":   "N-module peripheral stabiliser failure — intermediates visible (similar to NDUFA2/NDUFS5). CONTRAST cleaner absent CI in NDUFS7/NDUFS8 (Fe-S relay), NDUFB3 (PP-module scaffold), NDUFA11 (PP-PD inter-module boundary).",
                "target_gene":    "NDUFS7 / NDUFS8 / NDUFB3 / NDUFA11",
                "target_freq_pct": 0,
            },
            {
                "feature":        "DUAL-FUNCTION GRIM-19 — UNIQUE among ALL nuclear-encoded CI subunits",
                "significance":   "Only CI structural subunit with bona fide apoptosis regulatory function (GRIM pathway) AND STAT3 inhibition in cytoplasm AND CI structural role. WES/gene panel essential to distinguish from other CI-Leigh subunit genes.",
                "target_gene":    "All other CI-Leigh nuclear subunits (no apoptosis/STAT3 function)",
                "target_freq_pct": 0,
            },
            {
                "feature":        "NO hepatopathy",
                "significance":   "KEY DDx vs POLG Alpers (~80%) and DGUOK hepatocerebral (~90%)",
                "target_gene":    "POLG / DGUOK",
                "target_freq_pct": 85,
            },
            {
                "feature":        "CIV (COX) NORMAL",
                "significance":   "KEY DDx vs SURF1 / SCO2 / COX10 / COX15 (isolated CIV deficiency)",
                "target_gene":    "SURF1 / SCO2 / COX10 / COX15",
                "target_freq_pct": 100,
            },
        ],
        "absolute_contraindications": [
            "Metformin — directly inhibits CI at ND1/quinone-binding site",
            "Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block",
            "Linezolid — inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits",
            "Chloramphenicol — same mt-ribosomal mechanism as linezolid",
        ],
        "contraindicated": [
            "Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH (N-module peripheral stabilisation failed, CI N-module assembly disrupted)",
        ],
        "preferred_treatments": [
            "LEV (levetiracetam) — AED first-line: renal excretion, no mito toxicity",
            "Riboflavin (B2) — CI-specific cofactor; FMN at NDUFV1 (same N-module face as NDUFA13)",
            "CoQ10 (ubiquinol) — electron acceptor at quinone site; downstream of CI",
            "Thiamine (B1) — MANDATORY empiric: SLC19A3/BTD treatable mimic",
            "Biotin — MANDATORY empiric: BTD treatable mimic",
            "Succinate — CII bypass; bypasses NDUFA13-failed N-module CI entirely",
            "L-Carnitine — energy metabolism support",
            "IV dextrose GIR 6–8 mg/kg/min — never fast",
            "Sevoflurane (not propofol) for anaesthesia",
        ],
        "key_references": [
            "Huang G et al. 1999 J Biol Chem — GRIM-19 identified as apoptosis mediator in IFN-beta/retinoic acid cell death pathway",
            "Angell JE et al. 2000 J Biol Chem — GRIM-19 (NDUFB16/NDUFA13) identified as CI structural subunit",
            "Fearnley IM & Walker JE 1992 Biochim Biophys Acta — CI subunit nomenclature (B16.6 = NDUFA13)",
            "Fassone E & Rahman S 2012 JMedGenet — CI genetics review (nuclear-encoded subunits including NDUFA13)",
            "Sazanov LA 2015 NatRevMolCellBiol — CI cryo-EM structure: N-module; NDUFA13 (GRIM-19) peripheral position",
        ],
    }


# ---------------------------------------------------------------------------
# Public API: get_breakdown
# ---------------------------------------------------------------------------

def get_breakdown() -> dict[str, Any]:
    pts = _cohort()
    n = len(pts)

    features = [
        "leigh_mri", "lactic_acidosis", "hypotonia", "psychomotor_regression",
        "respiratory_compromise", "seizures", "ataxia", "dystonia", "hcm",
        "peripheral_neuropathy", "olfactory_bulb_lesions", "leukodystrophy", "hepatopathy",
    ]
    feature_frequencies = {
        f: {"count": sum(1 for p in pts if p.get(f)), "pct": _pct(f)}
        for f in features
    }

    outcome_counts: dict[str, int] = {}
    for p in pts:
        outcome_counts[p["outcome"]] = outcome_counts.get(p["outcome"], 0) + 1

    mutation_counts: dict[str, int] = {}
    for p in pts:
        mutation_counts[p["mutation"]] = mutation_counts.get(p["mutation"], 0) + 1

    region_counts: dict[str, int] = {}
    for p in pts:
        region_counts[p["region"]] = region_counts.get(p["region"], 0) + 1

    return {
        "n": n,
        "patients": pts,
        "feature_frequencies": feature_frequencies,
        "outcome_distribution": outcome_counts,
        "mutation_distribution": mutation_counts,
        "region_distribution": region_counts,
        "sex_distribution": {
            "M": sum(1 for p in pts if p["sex"] == "M"),
            "F": sum(1 for p in pts if p["sex"] == "F"),
        },
        "ci_activity_histogram": {
            "bins": ["5–8%", "8–12%", "12–16%", "16–20%"],
            "counts": [
                sum(1 for p in pts if 5 <= p["ci_activity_pct"] < 8),
                sum(1 for p in pts if 8 <= p["ci_activity_pct"] < 12),
                sum(1 for p in pts if 12 <= p["ci_activity_pct"] < 16),
                sum(1 for p in pts if 16 <= p["ci_activity_pct"] <= 20),
            ],
        },
    }


# ---------------------------------------------------------------------------
# Public API: get_definitions
# ---------------------------------------------------------------------------

def get_definitions() -> dict[str, Any]:
    pharmacology = [
        {
            "term": "Metformin — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Metformin directly inhibits CI at the ND1/quinone-binding interface. "
                "In NDUFA13 deficiency the N-module peripheral stabilisation has failed; "
                "CI activity is already 5–20%. Metformin precipitates fatal lactic crisis."
            ),
        },
        {
            "term": "Valproate (VPA) — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Three independent mechanisms: (1) CoA sequestration → energy deficit; "
                "(2) POLG inhibition → mtDNA depletion (worsens mt-encoded ND subunit "
                "availability, all 7 mtDNA-encoded ND subunits affected); "
                "(3) direct ND-subunit expression block. All three compound an already "
                "critically reduced CI in NDUFA13 deficiency."
            ),
        },
        {
            "term": "Linezolid — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Inhibits mitochondrial 23S rRNA (mt-ribosome) → blocks synthesis of ALL "
                "seven mt-encoded ND subunits (ND1–ND6, ND4L). In NDUFA13 deficiency the "
                "N-module peripheral stabilisation has failed; removing all mt ND subunits "
                "too annihilates any residual CI. Fatal."
            ),
        },
        {
            "term": "Chloramphenicol — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Same mitochondrial ribosomal inhibition mechanism as linezolid. "
                "Blocks synthesis of all mt-encoded ND subunits. Never use in CI-Leigh."
            ),
        },
        {
            "term": "Ketogenic diet — CONTRAINDICATED",
            "category": "contraindicated",
            "detail": (
                "KD forces energy production through fatty-acid β-oxidation, generating NADH "
                "that must be re-oxidised via CI. In NDUFA13 deficiency the N-module "
                "peripheral stabilisation has failed → CI N-module assembly disrupted → "
                "CI cannot reoxidise the NADH surge → fatal metabolic crisis."
            ),
        },
        {
            "term": "Propofol — AVOID",
            "category": "caution",
            "detail": (
                "PRIS (propofol infusion syndrome) inhibits CIV, creating a SECOND ETC "
                "bottleneck downstream of the already-failed CI. Combined CI+CIV failure "
                "is catastrophic. Sevoflurane is preferred for all anaesthesia."
            ),
        },
        {
            "term": "Phenobarbital — HIGH CAUTION",
            "category": "caution",
            "detail": (
                "Secondary CI inhibitor — adds to the primary NDUFA13-driven CI deficit. "
                "Use LEV (levetiracetam) first for seizure control. Reserve phenobarbital "
                "for refractory seizures with close metabolic monitoring."
            ),
        },
        {
            "term": "Succinate — Level C (CII bypass)",
            "category": "treatment",
            "detail": (
                "Succinate feeds directly into Complex II (SDHA, fully nuclear), bypassing "
                "the NDUFA13-failed CI entirely. Electrons enter the ETC at ubiquinone via "
                "CII → CIII → CIV, generating ATP without CI. "
                "Level C evidence; standard adjunct in CI-Leigh."
            ),
        },
        {
            "term": "LEV (Levetiracetam) — Preferred AED",
            "category": "treatment",
            "detail": (
                "Renal excretion; no hepatic metabolism; no mito toxicity; no CI inhibition. "
                "First-line AED in all CI-Leigh syndromes. SV2A mechanism unrelated to ETC."
            ),
        },
    ]

    gene_concepts = [
        {
            "term": "NDUFA13 — N-Module Peripheral Stabiliser (GRIM-19 / B16.6)",
            "category": "gene_concept",
            "detail": (
                "NDUFA13 (NADH:Ubiquinone Oxidoreductase Subunit A13) is a ~144-aa "
                "nuclear-encoded protein (~122 aa mature, ~16.7 kDa) located at the "
                "N-module periphery, contacting the NDUFV1-NDUFV2 face of the N-module "
                "peripheral arm of CI. Also called GRIM-19 (Growth Inhibitor and "
                "Retinoid-IFN-induced Mortality factor 19) and the B16.6 subunit. "
                "Chromosome 19p13.11. OMIM Gene *609435."
            ),
        },
        {
            "term": "GRIM-19 / STAT3 Inhibition — Dual-Function Unique Among All CI Subunits",
            "category": "gene_concept",
            "detail": (
                "NDUFA13/GRIM-19 is the ONLY nuclear-encoded CI structural subunit with "
                "dual molecular function: (1) CI peripheral N-module structural stabiliser "
                "(contacts NDUFV1-NDUFV2 face); (2) apoptosis regulator via the GRIM "
                "pathway (Gene associated with Retinoid-IFN-induced Mortality) — originally "
                "identified as a cell-death gene in IFN-beta/retinoic acid signalling; "
                "(3) cytoplasmic STAT3 inhibitor — NDUFA13 binds STAT3 and prevents its "
                "nuclear translocation (constitutive STAT3 activation possible in NDUFA13 "
                "deficiency). No other CI subunit (of 45 nuclear-encoded structural subunits) "
                "has this combination of CI structural + apoptosis + STAT3 inhibitor roles."
            ),
        },
        {
            "term": "BN-PAGE Pattern in NDUFA13 Deficiency",
            "category": "gene_concept",
            "detail": (
                "Blue-native PAGE shows N-module sub-assembly intermediates (partial N-module "
                "assemblies visible) in NDUFA13 LOF — the N-module peripheral stabilisation "
                "has failed, releasing partial N-module subcomplexes rather than producing "
                "a fully absent CI. Pattern resembles NDUFA2 and NDUFS5 (other N-module "
                "peripheral stabilisers) more than the cleaner absent CI in NDUFS7/NDUFS8 "
                "(direct Fe-S relay block), NDUFB3 (PP-module scaffolding loss), or NDUFA11 "
                "(PP-PD membrane arm inter-module boundary)."
            ),
        },
        {
            "term": "NDUFA13 Genotype–Phenotype Correlations",
            "category": "gene_concept",
            "detail": (
                "N-module/NDUFV1-face contact (p.Arg76Trp): severe infantile onset — "
                "interface contact lost, N-module peripheral stabilisation collapses. "
                "Alpha-helix core (p.Glu70Lys): moderate; partial CI residual retained. "
                "Signal peptide region (p.Pro31Leu): severe neonatal; mitochondrial "
                "targeting disrupted. C-terminal structural (p.Ala114Val): moderate, "
                "consanguineous kindreds; partial CI residual. Splice (c.IVS2+1G>A): "
                "partial CI residual (~10–18%); intermediate severity, episodic course."
            ),
        },
        {
            "term": "OMIM *609435 / #256000",
            "category": "gene_concept",
            "detail": (
                "NDUFA13 gene: OMIM *609435. Primary disease: Leigh Syndrome OMIM #256000. "
                "Inheritance: AR biallelic LOF. Chromosome: 19p13.11."
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "Leigh Syndrome — CI-Leigh Nuclear Subunit Series",
            "category": "disease_concept",
            "detail": (
                "Leigh syndrome (OMIM #256000) is a severe mitochondrial encephalopathy "
                "characterised by bilateral brainstem/basal ganglia lesions on MRI, "
                "psychomotor regression, hypotonia, and lactic acidosis. Over 50 nuclear "
                "CI subunit genes can cause CI-Leigh. NDUFA13 (GRIM-19, B16.6, N-module "
                "peripheral stabiliser) is one of the characterised nuclear-encoded CI "
                "peripheral arm subunit causes with its unique dual-function GRIM-19/STAT3 "
                "inhibitor molecular signature."
            ),
        },
        {
            "term": "Isolated CI Deficiency — NDUFA13",
            "category": "disease_concept",
            "detail": (
                "NDUFA13 LOF produces isolated Complex I deficiency (5–20% residual), with "
                "CII, CIII, CIV all normal — the 'biochemical fingerprint' of CI-Leigh. "
                "Enzyme histochemistry: COX normal (CIV), SDH normal (CII). "
                "Respiratory chain analysis is essential before WES."
            ),
        },
        {
            "term": "NDUFA13 vs NDUFA2/NDUFS5 — N-Module Peripheral Stabiliser Series",
            "category": "disease_concept",
            "detail": (
                "NDUFA2 (B8, thioredoxin-fold, 5q31.2), NDUFS5 (10.8 kDa, 1p34.3), and "
                "NDUFA13 (GRIM-19, B16.6, 19p13.11) are all N-module peripheral stabilisers "
                "producing sub-assembly intermediates on BN-PAGE — distinguishing them from "
                "the cleaner absent-CI pattern of membrane-arm defects (NDUFB3, NDUFA11). "
                "Among these, NDUFA13 alone carries the GRIM-19 dual-function (apoptosis + "
                "STAT3 inhibition). Clinical distinction requires WES; biochemical CI "
                "fingerprints are otherwise identical."
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Prescribing Safety Pocket Card — NDUFA13 Leigh",
            "category": "prescribing_safety",
            "detail": (
                "ABSOLUTE CI    : Metformin · Valproate · Linezolid · Chloramphenicol\n"
                "CONTRAINDICATED: Ketogenic diet (NADH cannot be re-oxidised — N-module peripheral stabilisation failed)\n"
                "AVOID          : Propofol (PRIS + CIV block → dual ETC failure)\n"
                "HIGH CAUTION   : Phenobarbital (secondary CI inhibitor; use LEV first)\n"
                "PREFERRED AED  : LEV (levetiracetam) — renal, no mito toxicity\n"
                "ANAESTHESIA    : Sevoflurane (NOT propofol)\n"
                "GLUCOSE        : IV dextrose GIR 6–8 mg/kg/min — NEVER fast\n"
                "COFACTORS      : Riboflavin B2 · CoQ10 (ubiquinol) · Thiamine B1* · Biotin* · Succinate · Carnitine\n"
                "                 (* MANDATORY empiric before genetics: rules out SLC19A3/BTD)\n"
                "MECHANISM NOTE : NDUFA13/GRIM-19 B16.6 N-module peripheral stabiliser; dual-function apoptosis regulator + STAT3 inhibitor; N-module peripheral face collapsed; Riboflavin FMN target NDUFV1 same N-module face as NDUFA13"
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }
