#!/usr/bin/env python3
"""NDUFA12 — Leigh Syndrome Isolated Complex I Deficiency (N-Module/Q-Module Interface / NDUFAF2-Paralog Assembly-Swap Subunit / B17.2).

NDUFA12 (NADH:Ubiquinone Oxidoreductase Subunit A12) is a 145-aa nuclear-encoded
structural subunit of Complex I (~13.5 kDa mature), located at the interface between
the N-module (NADH-oxidation module) and the Q-module (quinone-reduction module) in
the peripheral arm.  It carries NO iron-sulfur cluster and NO zinc finger; its role
is purely structural.

  NDUFA12 gene     OMIM *614846
  Disease          Leigh Syndrome (OMIM #256000) / CI Deficiency Nuclear Type 11 (OMIM #618234)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       12q22

PATHOPHYSIOLOGY (Complex I / N-module–Q-module interface / NDUFA12):
  NDUFA12 resides at the N-module/Q-module interface in the peripheral arm of
  Complex I.  It is unique among CI subunits in being the structural counterpart
  of NDUFAF2 (also called NDUFA12L), an assembly factor that transiently occupies
  the same binding site during CI assembly.  At the final stage of peripheral-arm
  assembly NDUFAF2 is replaced ("swapped out") by the structural subunit NDUFA12;
  this assembly-swap checkpoint is essential for CI holocomplex maturation.

  The Fe-S electron relay (upstream, in the N-module and Q-module):
    NDUFV1  (N3, [4Fe-4S]) ← FMN primary NADH acceptor (N-module)
    NDUFV2  (N1b, [2Fe-2S]) ← second relay step (N-module)
    NDUFS7  (N4, [4Fe-4S]) ← third relay (Q-module)
    NDUFS8  (N6a/N6b, dual [4Fe-4S]) ← fourth/fifth relay (TYKY, Q-module)
    NDUFS1  (N5, [4Fe-4S]) ← peripheral arm relay
    NDUFS2  (N2, [4Fe-4S]) ← TERMINAL relay → ubiquinone reduction

  NDUFA12's unique role:
    1. NDUFA12 does NOT carry an Fe-S cluster, zinc finger, or SDR fold.  It is
       a purely structural subunit that bridges the N-module and Q-module.
    2. NDUFAF2 (NDUFA12L) — the assembly factor paralog — transiently occupies
       NDUFA12's binding site during peripheral arm assembly.  NDUFAF2 must be
       displaced and replaced by NDUFA12 for mature CI to form.
    3. Loss of NDUFA12 → the NDUFAF2→NDUFA12 assembly-swap cannot be completed
       → CI peripheral arm peripheral arm cannot fully mature → CI holocomplex
       assembly fails → isolated CI deficiency.
    4. BN-PAGE in NDUFA12 LOF: N-module/Q-module sub-assembly intermediates —
       the partial peripheral arm (N-module or Q-module fragments) fails to
       integrate.  Pattern overlaps NDUFA2 (N/ND2-module junction) and NDUFS5
       (N-module peripheral stabiliser), but is distinct from NDUFA9 (I-gamma
       Q/membrane-arm junction) and from Fe-S relay blocks (NDUFS7/NDUFS8
       cleaner absent CI).
    5. Net result: isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL.

  NDUFAF2–NDUFA12 assembly-swap — mechanistic note:
    • NDUFAF2 (NDUFA12L) is a 165-aa paralog of NDUFA12.  During early CI
      peripheral-arm assembly NDUFAF2 stabilises the nascent N-module/Q-module
      contact surface; once assembly is complete NDUFAF2 dissociates and is
      replaced by NDUFA12 (the permanent structural subunit).
    • This is the only known subunit/assembly-factor "swap" in the entire CI
      assembly pathway: NDUFA12 is the only mature CI subunit that directly
      replaces a specific assembly factor from its own binding site.
    • NDUFAF2 mutations cause a separate CI assembly disease (OMIM #618229),
      distinct from NDUFA12 deficiency, because the assembly factor itself
      (not its replacement) is disrupted.

  Biochemical signature (identical to all CI-Leigh nuclear subunit mutations):
    Complex I: 5–20% of control (severely reduced) — ISOLATED CI DEFICIENCY
    Complex II (SDHA): NORMAL
    Complex III: NORMAL
    Complex IV (COX): NORMAL — KEY DDx SURF1 / SCO2 / COX10 / COX15

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (IP1/75kDa, peripheral arm N-module):
    • NO peripheral neuropathy in NDUFA12 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory, 5q11.2):
    • NO olfactory bulb MRI lesions in NDUFA12 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV assembly):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFS7 / NDUFS8 (direct Fe-S relay subunits):
    • BN-PAGE: NDUFA12 shows N/Q-module sub-assembly intermediates (interface failure)
      vs NDUFS7/NDUFS8 cleaner absent CI (direct relay block)
  vs NDUFA9 (Q/membrane-arm junction, SDR fold):
    • NDUFA9 is at the Q-module↔proximal-membrane-arm junction (I-gamma checkpoint)
      NDUFA12 is at the N-module↔Q-module interface — entirely different location
      in the peripheral arm; different fold (NDUFA12 has no SDR fold)
  vs NDUFA2 (N-module/ND2-module junction, thioredoxin fold):
    • NDUFA2 bridges the N-module to the membrane arm via ND2-module; NDUFA12
      bridges N-module to Q-module within the peripheral arm; different assembly
      stage, different structural fold
  vs NDUFAF2 (NDUFA12's paralog, CI assembly factor):
    • NDUFAF2 mutations disrupt the early assembly-factor phase; NDUFA12 mutations
      prevent the FINAL swap step — phenotypically similar CI-Leigh but genetically
      distinct. NDUFAF2 disease = CI Deficiency Type 10 (OMIM #618229)
  vs POLG/DGUOK (mtDNA depletion syndromes):
    • NO hepatopathy in NDUFA12 (POLG: ~80%; DGUOK: ~90%)
  vs GRACILE (BCS1L/Complex III):
    • NO iron overload; NO aminoaciduria; NO neonatal cholestasis
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • COX activity NORMAL in NDUFA12 — biochemical fingerprint distinction

FOUNDER / RECURRENT MUTATIONS:
  p.Arg98Trp   c.292C>T   — N/Q-module interface contact residue; severe infantile onset
  p.Ser128Leu  c.383C>T   — C-terminal structural region; moderate phenotype
  p.Ile105Thr  c.314T>C   — core hydrophobic packing; intermediate severity
  p.Arg94Trp   c.280C>T   — interface contact; severe early infantile
  c.IVS2+1G>A             — splice donor exon 2; partial CI residual (~15–20%); moderate episodic

THERAPY — NDUFA12 / CI-LEIGH SPECIFICS:
  No targeted NDUFA12 assembly-swap rescue is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits complex I at ND1/quinone-binding site
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH (N/Q junction failed)
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group at NDUFV1 upstream of NDUFA12
      CoQ10 (ubiquinol) — electron acceptor at quinone site; downstream of NDUFA12
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFA12-failed N/Q-module junction entirely
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
"""

import random
from typing import Any

SEED = 633
GENE = "NDUFA12"
DISEASE = "NDUFA12 Leigh Syndrome — Isolated Complex I Deficiency (N-Module/Q-Module Interface / NDUFAF2-Paralog Assembly-Swap Subunit / B17.2)"
OMIM_GENE = "614846"
OMIM_DISEASE = "256000"
OMIM_DISEASE_CI = "618234"
CHROMOSOME = "12q22"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-633, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Arg98Trp / c.IVS2+1G>A (compound het)",
        "p.Arg98Trp / p.Arg98Trp (hom missense)",
        "p.Arg94Trp / p.Ser128Leu (compound het)",
        "p.Ile105Thr / c.IVS2+1G>A (compound het)",
        "p.Arg94Trp / p.Arg94Trp (hom null-like, consanguineous)",
        "p.Ser128Leu / c.IVS2+1G>A (compound het)",
        "p.Arg98Trp / p.Ile105Thr (compound het)",
        "p.Arg94Trp / p.Ile105Thr (compound het)",
        "Novel biallelic LOF (frameshift/splice)",
        "p.Arg98Trp / novel missense (compound het)",
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
            "lactic_acidosis":        rng.random() < 0.87,
            "hypotonia":              rng.random() < 0.86,
            "psychomotor_regression": rng.random() < 0.95,
            "respiratory_compromise": rng.random() < 0.42,
            "seizures":               rng.random() < 0.48,
            "ataxia":                 rng.random() < 0.40,
            "dystonia":               rng.random() < 0.35,
            "hcm":                    rng.random() < 0.05,
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
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Subunit A12",
        "also_known_as":  "B17.2 subunit (N-module/Q-module interface, NDUFAF2-paralog assembly-swap)",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "omim_disease_ci": OMIM_DISEASE_CI,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa_precursor": 145,
            "size_aa_mature":    119,
            "size_kda":          13.5,
            "fold":              "Structural interface subunit — NO Fe-S cluster, NO zinc finger, NO SDR fold",
            "module":            "N-module/Q-module interface (peripheral arm, NDUFAF2-paralog assembly-swap checkpoint)",
            "fe_s_cluster":      False,
            "zinc_finger":       False,
            "function":          "Structural bridging at N-module/Q-module interface in peripheral arm; replaces NDUFAF2 (NDUFA12L assembly factor) at final peripheral-arm assembly-swap checkpoint; unique NDUFAF2→NDUFA12 swap mechanism; loss stalls peripheral arm maturation",
        },
        "key_pathway_note": (
            "NDUFA12 occupies the N-module/Q-module interface in the CI peripheral arm. "
            "It is the ONLY mature CI subunit that directly replaces a specific assembly factor "
            "(NDUFAF2/NDUFA12L) from its own binding site — a unique 'assembly-swap' checkpoint. "
            "During CI assembly, NDUFAF2 stabilises the N/Q-module contact surface; at the final "
            "peripheral-arm maturation step NDUFAF2 dissociates and NDUFA12 takes its place. "
            "Loss of NDUFA12 → the swap cannot occur → peripheral arm cannot fully mature → "
            "CI holocomplex assembly fails → isolated CI deficiency.  NDUFA12 carries NO "
            "iron-sulfur cluster, NO zinc finger, NO SDR fold — purely structural interface role. "
            "BN-PAGE: N/Q-module sub-assembly intermediates (peripheral arm dissociated), "
            "overlapping NDUFA2/NDUFS5 junction defects but at a different peripheral-arm "
            "checkpoint from NDUFA9 (I-gamma Q/membrane-arm) or NDUFS7/NDUFS8 (Fe-S relay)."
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
                "feature":        "N/Q-module sub-assembly intermediates on BN-PAGE (N-module/Q-module interface failure)",
                "significance":   "Peripheral arm interface defect — overlaps NDUFA2/NDUFS5; DISTINCT from I-gamma junction (NDUFA9), Fe-S relay block (NDUFS7/NDUFS8 cleaner absent CI), and NDUFS6 zinc-finger collapse",
                "target_gene":    "NDUFS7 / NDUFS8",
                "target_freq_pct": 0,
            },
            {
                "feature":        "NDUFAF2→NDUFA12 assembly-swap — unique mechanism; NDUFA12 is the only mature CI subunit that directly replaces an assembly factor from its own binding site",
                "significance":   "Distinguish NDUFA12 deficiency (swap cannot complete) from NDUFAF2 deficiency (swap cannot begin): both CI-Leigh but different genes — WES/gene panel essential",
                "target_gene":    "NDUFAF2 (CI Deficiency Type 10, OMIM #618229)",
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
            "Metformin — directly inhibits CI at ND1/quinone-binding site (Q-module; NDUFA12 N/Q interface collapses the route to this site)",
            "Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block",
            "Linezolid — inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits",
            "Chloramphenicol — same mt-ribosomal mechanism as linezolid",
        ],
        "contraindicated": [
            "Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH (N/Q-module interface collapsed, CI assembly failed)",
        ],
        "preferred_treatments": [
            "LEV (levetiracetam) — AED first-line: renal excretion, no mito toxicity",
            "Riboflavin (B2) — CI-specific cofactor; FMN at NDUFV1 upstream of NDUFA12 N/Q interface",
            "CoQ10 (ubiquinol) — electron acceptor downstream of the N/Q interface",
            "Thiamine (B1) — MANDATORY empiric: SLC19A3/BTD treatable mimic",
            "Biotin — MANDATORY empiric: BTD treatable mimic",
            "Succinate — CII bypass; bypasses NDUFA12-failed N/Q-module interface entirely",
            "L-Carnitine — energy metabolism support",
            "IV dextrose GIR 6–8 mg/kg/min — never fast",
            "Sevoflurane (not propofol) for anaesthesia",
        ],
        "key_references": [
            "Haack TB et al. 2012 Nat Genet — nuclear CI subunit screening series including NDUFA12",
            "Fassone E & Rahman S 2012 JMedGenet — CI genetics review (nuclear-encoded subunits)",
            "Sazanov LA 2015 NatRevMolCellBiol — CI cryo-EM structure: NDUFA12 (B17.2) at N/Q interface",
            "Stroud DA et al. 2016 Nature — CI assembly pathway: NDUFAF2→NDUFA12 swap checkpoint",
            "Guerrero-Castillo S et al. 2017 Cell Metab — CI assembly intermediates: NDUFAF2 and NDUFA12 swap",
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
                "In NDUFA12 deficiency the N/Q-module interface has collapsed; the electron "
                "pathway from NADH cannot reach the quinone-binding site. CI activity is "
                "already 5–20%. Metformin precipitates fatal lactic crisis."
            ),
        },
        {
            "term": "Valproate (VPA) — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Three independent mechanisms: (1) CoA sequestration → energy deficit; "
                "(2) POLG inhibition → mtDNA depletion (worsens mt-ND subunit availability); "
                "(3) direct ND-subunit expression block. In NDUFA12 deficiency all three "
                "compound an already critically reduced CI."
            ),
        },
        {
            "term": "Linezolid — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Inhibits mitochondrial 23S rRNA (mt-ribosome) → blocks synthesis of ALL "
                "seven mt-encoded ND subunits (ND1–ND6, ND4L). In NDUFA12 deficiency the "
                "nuclear interface subunit is absent; removing all mt ND subunits too "
                "annihilates any residual CI. Fatal."
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
                "that must be re-oxidised via CI. In NDUFA12 deficiency the N/Q-module "
                "interface has failed → CI cannot reoxidise the NADH surge → fatal "
                "metabolic crisis."
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
                "Secondary CI inhibitor — adds to the primary NDUFA12-driven CI deficit. "
                "Use LEV (levetiracetam) first for seizure control. Reserve phenobarbital "
                "for refractory seizures with close metabolic monitoring."
            ),
        },
        {
            "term": "Succinate — Level C (CII bypass)",
            "category": "treatment",
            "detail": (
                "Succinate feeds directly into Complex II (SDHA, fully nuclear), bypassing "
                "the NDUFA12-failed N/Q-module interface of CI entirely. Electrons enter the "
                "ETC at ubiquinone via CII → CIII → CIV, generating ATP without CI. "
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
            "term": "NDUFA12 — N-Module/Q-Module Interface Structural Subunit (B17.2)",
            "category": "gene_concept",
            "detail": (
                "NDUFA12 (NADH:Ubiquinone Oxidoreductase Subunit A12) is a 145-aa "
                "nuclear-encoded protein (~119 aa mature, ~13.5 kDa) located at the interface "
                "of the N-module and Q-module in the CI peripheral arm.  It carries NO "
                "iron-sulfur cluster, NO zinc finger, NO SDR fold — purely structural.  "
                "Also called B17.2 subunit.  Chromosome 12q22.  OMIM Gene *614846."
            ),
        },
        {
            "term": "NDUFAF2→NDUFA12 Assembly-Swap — Unique CI Checkpoint",
            "category": "gene_concept",
            "detail": (
                "NDUFA12 is the ONLY mature CI subunit that directly replaces a specific "
                "assembly factor (NDUFAF2/NDUFA12L) from its own binding site. "
                "During early CI peripheral-arm assembly NDUFAF2 stabilises the N/Q-module "
                "contact surface; at the final peripheral-arm maturation step NDUFAF2 "
                "dissociates and NDUFA12 takes its exact place.  Loss of NDUFA12 → the "
                "swap cannot occur → peripheral arm cannot fully mature → CI assembly fails. "
                "This swap is the only such subunit/assembly-factor replacement in the CI "
                "assembly pathway."
            ),
        },
        {
            "term": "BN-PAGE Pattern in NDUFA12 Deficiency",
            "category": "gene_concept",
            "detail": (
                "Blue-native PAGE shows N/Q-module sub-assembly intermediates in NDUFA12 LOF — "
                "the peripheral arm is partially assembled but the N-module/Q-module interface "
                "cannot be completed (NDUFAF2 cannot be displaced by NDUFA12). "
                "Pattern overlaps other peripheral-arm/junction defects (NDUFA2 at N/ND2-module, "
                "NDUFS5 peripheral stabiliser, NDUFA9 I-gamma Q/membrane-arm). "
                "Contrast: NDUFS7/NDUFS8 (direct Fe-S relay block) show cleaner absent-CI; "
                "NDUFS6 shows Q-module zinc-finger intermediates."
            ),
        },
        {
            "term": "NDUFA12 vs NDUFAF2 — Different Genes, Similar Phenotype",
            "category": "gene_concept",
            "detail": (
                "NDUFAF2 (NDUFA12L) — OMIM *609653 — is NDUFA12's paralog and assembly-factor "
                "predecessor at the same binding site.  NDUFAF2 mutations cause CI Deficiency "
                "Type 10 (OMIM #618229), phenotypically similar CI-Leigh but genetically "
                "distinct.  NDUFAF2 deficiency disrupts the early assembly-factor phase; "
                "NDUFA12 deficiency disrupts the final swap step.  WES/gene panel required "
                "to distinguish — biochemical CI fingerprints are identical."
            ),
        },
        {
            "term": "OMIM *614846 / #256000 / #618234",
            "category": "gene_concept",
            "detail": (
                "NDUFA12 gene: OMIM *614846. Primary disease: Leigh Syndrome OMIM #256000. "
                "CI Deficiency Nuclear Type 11: OMIM #618234. "
                "Inheritance: AR biallelic LOF. Chromosome: 12q22."
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
                "CI subunit genes can cause CI-Leigh. NDUFA12 (CI Nuclear Type 11) is one "
                "of the characterised nuclear-encoded CI interface subunit causes."
            ),
        },
        {
            "term": "Isolated CI Deficiency — Nuclear Type 11 (NDUFA12)",
            "category": "disease_concept",
            "detail": (
                "NDUFA12 LOF produces isolated Complex I deficiency (5–20% residual), with "
                "CII, CIII, CIV all normal — the 'biochemical fingerprint' of CI-Leigh. "
                "CI Nuclear Type 11 (OMIM #618234) specifically names NDUFA12 as the causal "
                "gene. Enzyme histochemistry: COX normal (CIV), SDH normal (CII). "
                "Respiratory chain analysis is essential before WES."
            ),
        },
        {
            "term": "NDUFA12 Genotype–Phenotype Correlations",
            "category": "disease_concept",
            "detail": (
                "Interface contact missense (p.Arg98Trp, p.Arg94Trp): severe infantile onset, "
                "CI <10%, rapid deterioration — N/Q-module interface contact lost. "
                "Core structural missense (p.Ile105Thr): intermediate severity; partial "
                "interface; some residual CI assembly. Hypomorphic splice (c.IVS2+1G>A): "
                "partial CI residual (15–20%), moderate course with episodic crises. "
                "C-terminal missense (p.Ser128Leu): milder/moderate, partial assembly."
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Prescribing Safety Pocket Card — NDUFA12 Leigh",
            "category": "prescribing_safety",
            "detail": (
                "ABSOLUTE CI    : Metformin · Valproate · Linezolid · Chloramphenicol\n"
                "CONTRAINDICATED: Ketogenic diet (NADH cannot be re-oxidised — N/Q interface failed)\n"
                "AVOID          : Propofol (PRIS + CIV block → dual ETC failure)\n"
                "HIGH CAUTION   : Phenobarbital (secondary CI inhibitor; use LEV first)\n"
                "PREFERRED AED  : LEV (levetiracetam) — renal, no mito toxicity\n"
                "ANAESTHESIA    : Sevoflurane (NOT propofol)\n"
                "GLUCOSE        : IV dextrose GIR 6–8 mg/kg/min — NEVER fast\n"
                "COFACTORS      : Riboflavin B2 · CoQ10 (ubiquinol) · Thiamine B1* · Biotin* · Succinate · Carnitine\n"
                "                 (* MANDATORY empiric before genetics: rules out SLC19A3/BTD)\n"
                "MECHANISM NOTE : NDUFA12 N/Q-module interface (B17.2); NDUFAF2→NDUFA12 assembly-swap failed; metformin ND1 target in Q-module"
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }
