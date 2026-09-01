#!/usr/bin/env python3
"""NDUFS8 — Leigh Syndrome Isolated Complex I Deficiency (N6a/N6b [4Fe-4S] Fe-S Clusters / TYKY Subunit).

NDUFS8 (NADH:Ubiquinone Oxidoreductase Core Subunit S8), also known as the TYKY subunit
(named after two CXXC tetracysteine motifs that co-ordinate two [4Fe-4S] Fe-S clusters),
occupies the Q-module/N-module approach region of Complex I. NDUFS8 carries BOTH the N6a
AND N6b [4Fe-4S] clusters — the 5th and 6th iron-sulfur clusters in the electron relay chain
(counting from NDUFV1/FMN → NDUFS2/N2 → ubiquinone). NDUFS8 sits downstream of NDUFS7/N4
in the relay and upstream of the terminal NDUFS2/N2 cluster.

  NDUFS8 gene      OMIM *602141
  Disease          Leigh Syndrome (OMIM #256000) /
                   Mitochondrial Complex I Deficiency, Nuclear Type 2 (OMIM #618222)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       11q13.2

PATHOPHYSIOLOGY (Complex I / N6a + N6b Fe-S clusters / NDUFS8 TYKY Q-module approach role):
  The Fe-S electron relay chain of Complex I transfers electrons from NADH → ubiquinone
  through a series of iron-sulfur clusters in the following order:

    NDUFV1  (N3, [4Fe-4S]) ← FMN primary NADH acceptor (N-module)
    NDUFV2  (N1b, [2Fe-2S]) ← second relay step (N-module)
    NDUFS7  (N4, [4Fe-4S]) ← third relay (Q-module/N-module junction)
    NDUFS8  (N6a, [4Fe-4S]) ← FOURTH relay — THIS SUBUNIT (Q-module approach)
    NDUFS8  (N6b, [4Fe-4S]) ← FIFTH relay — SAME SUBUNIT (N6b branch)
    NDUFS1  (N5, [4Fe-4S]) ← peripheral arm relay
    NDUFS2  (N2, [4Fe-4S]) ← TERMINAL carrier → ubiquinone reduction (Q-module)

  NDUFS8's critical dual-cluster role:
    1. NDUFS8 is unique in carrying TWO Fe-S clusters (N6a + N6b) via two CXXC motifs
    2. N6a and N6b sit between the N4 (NDUFS7) and N2 (NDUFS2) relay steps
    3. Loss of NDUFS8 → BOTH N6a and N6b absent → electrons from NDUFS7/N4 cannot
       propagate toward the terminal NDUFS2/N2 cluster
    4. DIRECT DUAL ELECTRON TRANSFER BLOCK: unlike NDUFS3 (scaffold/assembly failure),
       NDUFS8/N6a-N6b loss is a primary electron relay failure at two consecutive steps
    5. BN-PAGE: absent/severely reduced CI; cleaner CI loss (fewer assembly intermediates)
       than NDUFS3/Q-module scaffold failure; similar to NDUFS7

  Biochemical signature (IDENTICAL to all CI-Leigh Fe-S relay mutations):
    Complex I: 5–20% of control (severely reduced) — ISOLATED CI DEFICIENCY
    Complex II (SDHA): NORMAL (fully nuclear-encoded; NDUFS8 not required for CII)
    Complex III: NORMAL
    Complex IV (COX): NORMAL — KEY DDx SURF1/SCO2/COX10/COX15 (CIV-only deficiency)

DISTINGUISHING FEATURES vs NDUFS4/NDUFV1/NDUFS1/NDUFS2/NDUFS3/NDUFS7:
  vs NDUFS4 (accessory N-module, 5q11.2):
    • NO olfactory bulb lesions in NDUFS8 (NDUFS4: ~52–65% — PATHOGNOMONIC)
  vs NDUFV1 (N-module FMN/N3, 11q13.2):
    • NO leukodystrophy / white matter T2 in NDUFS8 (NDUFV1: ~40–50% — DISTINGUISHING)
  vs NDUFS1 (IP1/75kDa/N-module Fe-S N1b-N5, 2q33.3):
    • NO peripheral neuropathy in NDUFS8 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS7 (N4/20kDa/Q-N junction, 19p13.3):
    • Both produce isolated CI + Leigh; NDUFS7 = N4 single-cluster block;
      NDUFS8 = dual N6a+N6b block (two consecutive Fe-S clusters)
    • BN-PAGE pattern similar (cleaner CI loss vs NDUFS3 scaffold)
    • Clinically indistinguishable without genetics — requires WES/panel
  vs NDUFS2 (PSST/49kDa/N2 terminal, 1q23.3):
    • NDUFS2 = terminal Fe-S N2 carrier (ubiquinone reduction step)
    • NDUFS8 = penultimate N6a/N6b relay (one step upstream of NDUFS2/N2)
    • Both → identical isolated CI; indistinguishable without genetics
  vs NDUFS3 (QP-C scaffold, 11p11.11):
    • NDUFS3 = Q-module scaffold failure → CI assembly intermediates on BN-PAGE
    • NDUFS8 = direct N6a/N6b Fe-S electron relay block → cleaner BN-PAGE CI loss
    • Both → identical isolated CI deficiency (5–20%), no peripheral neuropathy

FOUNDER / RECURRENT MUTATIONS:
  p.Arg94Cys   c.280C>T — first NDUFS8 mutation in human CI deficiency (Loeffen 1998
                          NatGenet); severe compound heterozygous; CXXC-1 motif region
  p.Thr112Ala  c.334A>G — moderate; partial N6a retention; compound het; intermediate
  p.Arg102His  c.305G>A — severe infantile; CI <10%; North African families
  p.Ala85Val   c.254C>T — early-onset severe; compound het; European
  c.IVS3+1G>T  (splice) — partial residual CI; milder Leigh course; some N6 retained

THERAPY — NDUFS8/CI-LEIGH SPECIFICS:
  No targeted NDUFS8 N6a/N6b Fe-S cluster reconstitution is clinically available.
  Management follows the CI-Leigh supportive protocol (cofactors + avoid CI toxins).
  Succinate bypass (Level C): CII-mediated electron entry to ubiquinol bypasses the
    NDUFS8 N6a/N6b block ENTIRELY — electrons enter at ubiquinol directly via SDHA.
  Riboflavin B2 (Level C): FMN reinforces NDUFV1/N3 upstream; electrons still blocked
    at N6a/N6b. Riboflavin most targeted in NDUFV1 deficiency; empiric here. Level C.
  No peripheral neuropathy: physiotherapy focuses on central ataxia/dystonia only.

References:
  Loeffen JL et al. Nat Genet. 1998;20(4):338-342.
    (First NDUFS8 mutation in human CI deficiency; p.Arg94Cys compound het)
  Bénit P et al. Hum Mutat. 2001;17(5):382-390.
    (NDUFS8 compound hets in CI deficiency series; genotype-phenotype)
  Fassone E, Rahman S. J Med Genet. 2012;49(9):578-590.
    (CI deficiency genetics review — NDUFS8 in context of Fe-S relay series)
  Sazanov LA. Nat Rev Mol Cell Biol. 2015;16(6):375-388.
    (CI structural review — NDUFS8 N6a/N6b cluster positions in Fe-S relay)
  Loeffen J et al. Ann Neurol. 2000;47(2):169-178.
    (NDUFS8 and other CI subunit mutations — clinical series)
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 619
DISEASE_ID   = "ndufs8"
DISEASE_NAME = (
    "NDUFS8 Leigh Syndrome — Isolated Complex I Deficiency "
    "(CI-Leigh / NDUFS8 TYKY Subunit N6a + N6b [4Fe-4S] Fe-S Clusters)"
)
GENE         = "NDUFS8"
PROTEIN      = (
    "NDUFS8 — 201 aa precursor / ~185 aa mature, TYKY subunit (Q-module approach), "
    "carries DUAL N6a + N6b [4Fe-4S] Fe-S clusters via two CXXC tetracysteine motifs — "
    "4th and 5th electron relay bridge from NDUFS7/N4 toward NDUFS2/N2"
)
OMIM_GENE    = "*602141"
OMIM_DISEASE = "#256000 (Leigh Syndrome) / Mitochondrial Complex I Deficiency Nuclear Type 2 (#618222)"
CHROMOSOME   = "11q13.2"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Infantile / neonatal (typically 0–18 months)"
COHORT_SIZE  = 40
COLOR        = "#004d40"   # dark teal — N6a/N6b dual Fe-S cluster / TYKY NDUFS8 theme
LIGHT        = "#e0f2f1"

# Genotype pool
GENO_ARG94CYS_COMP  = "p.Arg94Cys / p.Thr112Ala (c.280C>T / c.334A>G) — first reported; Loeffen 1998 NatGenet; CXXC-1 motif; severe"
GENO_ARG102HIS      = "p.Arg102His / p.Arg94Cys (c.305G>A / c.280C>T) — severe infantile; CI <10%; compound het"
GENO_ALA85VAL_COMP  = "p.Ala85Val / p.Thr112Ala (c.254C>T / c.334A>G) — severe early-onset; compound het; European"
GENO_THR112ALA_HOM  = "p.Thr112Ala / p.Thr112Ala (c.334A>G hom) — moderate; partial N6a retention; consanguineous"
GENO_SPLICE         = "p.Arg94Cys / c.IVS3+1G>T (splice) — partial residual CI; milder Leigh course; some N6 retained"

GENO_POOL    = [GENO_ARG94CYS_COMP, GENO_ARG102HIS, GENO_ALA85VAL_COMP, GENO_THR112ALA_HOM, GENO_SPLICE]
GENO_WEIGHTS = [0.30,               0.20,            0.18,                0.18,               0.14]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG — reproducible 40-patient NDUFS8/CI-Leigh cohort (seed-619)."""
    return random.Random(SEED)


# ── Patient cohort ────────────────────────────────────────────────────────────
_TX_POOL = [
    "IV Dextrose GIR 6-8 (metabolic crisis — first-line)",
    "Riboflavin B2 (100–400 mg/day — CI-specific FMN precursor; upstream of NDUFS8 N6a/N6b block)",
    "CoQ10/Ubiquinol (300–600 mg/day)",
    "Thiamine B1 (mandatory empiric, all Leigh until SLC19A3/BTD excluded)",
    "Biotin (mandatory empiric, all Leigh until BTD ruled out)",
    "Succinate (IV/oral — Complex II bypass; bypasses blocked NDUFS8 N6a/N6b Fe-S clusters entirely)",
    "NaHCO3 (lactic acidosis correction, pH <7.20)",
    "LEV (seizures — preferred AED, renal excretion, no mito toxicity)",
    "NIV/BiPAP (central respiratory compromise)",
    "Carnitine (secondary deficiency, 50–100 mg/kg/day)",
    "Physiotherapy / occupational therapy (central ataxia + dystonia — no peripheral neuropathy)",
    "Enteral feeds / NG (avoid fasting; high-carbohydrate; KD strictly CI)",
]

_OUTCOMES = [
    "Alive — stable moderate-to-severe Leigh; cofactor + respiratory support",
    "Alive — prolonged survival with splice/missense alleles; partial N6 Fe-S residual; school-age",
    "Alive — severe Leigh; dependent care; ongoing multidisciplinary support",
    "Died — acute lactic crisis + respiratory failure in first 18 months (null/severe alleles)",
    "Died — progressive brainstem Leigh + respiratory failure; 2–4 yr trajectory",
]
_OUT_WEIGHTS = [0.27, 0.13, 0.22, 0.22, 0.16]


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
        lactate_mm  = round(rng.uniform(4.5, 22.0), 1)
        ci_pct      = round(rng.uniform(5.0, 21.0))          # Complex I (% of control)
        cii_pct     = round(rng.uniform(88, 108))             # Complex II — NORMAL
        civ_pct     = round(rng.uniform(85, 105))             # Complex IV — NORMAL

        has_leigh_mri       = rng.random() < 0.85
        has_regression      = rng.random() < 0.95
        has_hypotonia       = rng.random() < 0.88
        has_lactic          = rng.random() < 0.88
        has_resp            = rng.random() < 0.55
        has_seizures        = rng.random() < 0.48
        has_ataxia          = rng.random() < 0.40
        has_dystonia        = rng.random() < 0.32
        has_myoclonus       = rng.random() < 0.20
        has_nystagmus       = rng.random() < 0.20
        has_optic           = rng.random() < 0.28
        has_spasticity      = rng.random() < 0.35
        has_hcm             = rng.random() < 0.05   # ~5% — similar to NDUFS7 ~6%
        has_hepatopathy     = rng.random() < 0.03   # RARE — KEY DDx POLG/DGUOK
        has_iron            = False                  # NO iron overload — KEY DDx GRACILE
        has_tubulopathy     = rng.random() < 0.03   # RARE — KEY DDx COX10 (65%)
        has_olfactory       = False                  # NO olfactory bulb lesions — KEY DDx NDUFS4
        has_leukodystrophy  = rng.random() < 0.04   # RARE — KEY DDx NDUFV1 (40-50%)
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
        if has_hcm:            feat_list.append("HCM (RARE — KEY DDx SCO2 100%)")
        if has_hepatopathy:    feat_list.append("Hepatopathy (RARE — KEY DDx POLG/DGUOK)")

        txs     = rng.sample(_TX_POOL, k=rng.randint(4, 7))
        outcome = rng.choices(_OUTCOMES, weights=_OUT_WEIGHTS)[0]

        patients.append({
            "id":                  f"NDUFS8-{i:03d}",
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
        "HCM (~5% — lower than NDUFS7 ~6%, NDUFS1 ~12%; much less than SCO2 100%)": _pct("has_hcm"),
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
        {"label": "CI Activity",      "value": "5–20% control",                                                                    "color": "#c62828"},
        {"label": "Fatal",            "value": f"{round(died/COHORT_SIZE*100)}%",                                                  "color": "#b71c1c"},
        {"label": "Seed",             "value": f"#{SEED}",                                                                        "color": "#455a64"},
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "severity": "ABSOLUTE CI — DANGEROUS IN ALL LEIGH / CI DEFICIENCY",
            "mechanism": (
                "Triple mechanism in NDUFS8/CI-Leigh:\n"
                "1. CoA SEQUESTRATION: valproyl-CoA depletes the mitochondrial CoA pool "
                "   → TCA cycle collapses → OXPHOS substrate loss. In NDUFS8 deficiency "
                "   (CI already at 5–20% due to N6a/N6b Fe-S dual electron transfer block), "
                "   CoA depletion tips the patient into irreversible lactic crisis.\n"
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
            "severity": "ABSOLUTE CI — DIRECT COMPLEX I INHIBITOR: CATASTROPHIC IN NDUFS8",
            "mechanism": (
                "Metformin inhibits Complex I (NADH dehydrogenase) at the ND1/quinone "
                "binding site.\n"
                "In NDUFS8 deficiency: CI N6a/N6b Fe-S clusters are absent/non-functional "
                "(5–20% activity). The N6a/N6b block means N2 (NDUFS2) cannot receive electrons "
                "from upstream relay steps, so N2→ubiquinone transfer already fails upstream. "
                "Metformin's CI inhibition blocks residual ND1 quinone-binding capacity "
                "→ near-total CI shutdown → massive lactate accumulation (>15–20 mmol/L).\n\n"
                "Alternative for glucose management: insulin only. Never metformin."
            ),
        },
        {
            "drug": "Linezolid",
            "severity": "ABSOLUTE CI — mt-Ribosome 23S rRNA Block (ALL 7 ND mtDNA Subunits)",
            "mechanism": (
                "Linezolid inhibits the mitochondrial large ribosomal subunit (mt-LSU) 23S rRNA "
                "→ blocks synthesis of ALL 13 mitochondrially-encoded OXPHOS subunits.\n"
                "In NDUFS8/CI-Leigh:\n"
                "  • NDUFS8 itself is nuclear-encoded — BUT Complex I also requires 7 mtDNA-encoded "
                "    ND subunits (MT-ND1–6, MT-ND4L) for the P-module membrane arm\n"
                "  • Linezolid blocks synthesis of ALL 7 → CI P-module cannot assemble "
                "    → NDUFS8 N6a/N6b relay subunit has no functional context → CI → zero\n"
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
                "In NDUFS8 CI-Leigh: CI N6a/N6b Fe-S clusters are absent — the dual electron "
                "relay from NDUFS7/N4 toward NDUFS2/N2 is broken (5–20% activity). "
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
                "In NDUFS8/CI-Leigh: CI is the primary ETC bottleneck (N6a/N6b electron relay broken). "
                "Propofol's CIV inhibition creates a SECOND downstream ETC block → electrons "
                "trapped between N6-deficient CI and CIV → ROS burst.\n"
                "Use sevoflurane (volatile, no PRIS) or dexmedetomidine as alternatives."
            ),
        },
        {
            "drug": "Phenobarbital",
            "severity": "HIGH CAUTION — Secondary Complex I Inhibitor",
            "mechanism": (
                "Phenobarbital inhibits CI electron transport (quinone channel region). "
                "In NDUFS8/CI-Leigh: residual CI (5–20%) is critical. Phenobarbital's "
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
                "CI assembly collapses → catastrophic in NDUFS8/CI deficiency."
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
        "HCM (RARE — ~5%, similar to NDUFS7 ~6%)":                        _pct("has_hcm"),
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
            "term": "Valproate / VPA — ABSOLUTE CI in NDUFS8/CI-Leigh",
            "definition": (
                "NEVER use valproate in any patient with NDUFS8 mutation or suspected CI-Leigh.\n\n"
                "Three independent lethal mechanisms:\n"
                "1. CoA SEQUESTRATION (valproyl-CoA): depletes mitochondrial CoA → TCA cycle "
                "   substrates drop → OXPHOS cannot produce ATP → lactic crisis in CI-deficient cell.\n"
                "2. POLG INHIBITION: valproate inhibits mtDNA polymerase gamma → mtDNA depletion "
                "   → fewer MT-ND1–ND6 / MT-ND4L templates → fewer P-module CI subunits "
                "   → CI further compromised in NDUFS8 N6a/N6b Fe-S deficient state.\n"
                "3. HEPATOTOXICITY: VPA directly hepatotoxic → hepatic OXPHOS failure "
                "   → impaired gluconeogenesis → hypoglycaemia → crisis amplified.\n\n"
                "Preferred AED: LEVETIRACETAM (LEV) — renal excretion, zero mito toxicity. "
                "Second-line: clobazam (CLB), clonazepam — no mito toxicity."
            ),
        },
        {
            "term": "Succinate — Level C (Complex II Bypass — Bypasses NDUFS8 N6a/N6b Clusters Entirely)",
            "definition": (
                "MECHANISTIC BASIS:\n"
                "  Succinate → Complex II (succinate dehydrogenase / SDHA) → ubiquinol "
                "  → Complex III → CIV → O2.\n"
                "  This bypasses the NDUFS8 N6a/N6b Fe-S dual electron transfer block ENTIRELY: "
                "  neither NDUFS8 (N6a/N6b relay), NDUFS7 (N4 junction), nor NDUFS2/N2 (terminal) "
                "  nor NDUFV1/N3 (FMN) are required for CII→ubiquinol electron entry.\n\n"
                "CLINICAL RATIONALE:\n"
                "  In CI deficiency with N6a/N6b block, maintaining ubiquinol pool via CII sustains "
                "  partial ATP synthesis without relying on the broken NDUFS8 Fe-S relay.\n"
                "  Level C evidence; dose 2–8 g/day oral succinate or sodium succinate.\n"
                "  Most relevant in crisis: IV succinate (where available).\n\n"
                "CAUTION: Not a replacement for glucose (GIR 6-8 during crisis is first-line)."
            ),
        },
        {
            "term": "Riboflavin / Vitamin B2 — Level C (CI-specific, upstream of NDUFS8 N6a/N6b block)",
            "definition": (
                "Riboflavin → riboflavin-5'-phosphate (FMN) + FAD.\n\n"
                "CI relevance (upstream of NDUFS8 N6a/N6b Fe-S block):\n"
                "  FMN binds NDUFV1 (51kDa/N3 subunit) — the FIRST electron acceptor from NADH.\n"
                "  Extra FMN may stabilise NDUFV1 and improve electron injection at N3.\n"
                "  However, in NDUFS8 deficiency the N6a/N6b clusters are absent — electrons still "
                "  cannot propagate past N6a/N6b toward NDUFS2/N2 → limited direct benefit.\n\n"
                "NOTE: Riboflavin is more directly targeted in NDUFV1 deficiency (FMN active site). "
                "In NDUFS8 deficiency the N6a/N6b block cannot be repaired pharmacologically. "
                "Riboflavin used empirically alongside succinate. Level C.\n"
                "Dose: 100–400 mg/day."
            ),
        },
        {
            "term": "Thiamine (B1) + Biotin — MANDATORY Empiric in ALL Leigh Presentations",
            "definition": (
                "Before NDUFS8 genotype is confirmed, every infant with Leigh/Leigh-like MRI "
                "+ lactic acidosis + CI biochemistry MUST receive empiric thiamine AND biotin.\n\n"
                "THIAMINE B1:\n"
                "  SLC19A3 deficiency (biotin-thiamine responsive basal ganglia disease / BTBGD) "
                "  and PDHC deficiency can mimic NDUFS8-Leigh.\n"
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
            "term": "Acute Crisis Protocol — NDUFS8/CI-Leigh Metabolic Emergency",
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
                "  0.5–1 g/kg/day → CII-bypass of NDUFS8 N6a/N6b Fe-S block entirely.\n\n"
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
            "term": "NDUFS8 Gene Structure and Expression (TYKY Subunit)",
            "definition": (
                "Gene: NDUFS8 (NADH:Ubiquinone Oxidoreductase Core Subunit S8)\n"
                "Also known as: TYKY subunit (bovine analog nomenclature — two CXXC motifs\n"
                "  for Fe-S cluster coordination: N6a and N6b [4Fe-4S])\n"
                "Chromosome: 11q13.2\n"
                "Protein: 201 aa precursor; mature form ~185 aa; ~23 kDa (TYKY)\n\n"
                "NDUFS8 is the only Complex I subunit to carry TWO [4Fe-4S] clusters (N6a and N6b) "
                "via two tetracysteine CXXC motifs. It bridges the N4 (NDUFS7) relay step "
                "and the terminal N2 (NDUFS2) cluster.\n"
                "Ubiquitous expression across tissues; highest in brain, heart, skeletal muscle.\n\n"
                "OMIM *602141 (gene); #256000 (Leigh Syndrome) / #618222 (CI Deficiency Nuclear Type 2)\n\n"
                "Inheritance: AR — biallelic (compound heterozygous or homozygous) "
                "loss-of-function variants required for disease."
            ),
        },
        {
            "term": "NDUFS8 Dual N6a + N6b Fe-S Clusters — TYKY Electron Relay Role",
            "definition": (
                "NDUFS8 carries BOTH N6a AND N6b [4Fe-4S] clusters — making it unique among "
                "CI Fe-S subunits as the only one coordinating two consecutive Fe-S centres.\n\n"
                "Position in the Fe-S electron relay chain:\n"
                "  NDUFV1  → N3 [4Fe-4S] (FMN primary NADH acceptor)\n"
                "  NDUFV2  → N1b [2Fe-2S] (second relay)\n"
                "  NDUFS7  → N4 [4Fe-4S] (third relay — Q/N-module junction)\n"
                "  NDUFS8  → N6a [4Fe-4S] (FOURTH relay — Q-module approach — THIS SUBUNIT)\n"
                "  NDUFS8  → N6b [4Fe-4S] (FIFTH relay — Q-module — SAME SUBUNIT)\n"
                "  NDUFS1  → N5 [4Fe-4S] (peripheral arm relay)\n"
                "  NDUFS2  → N2 [4Fe-4S] (TERMINAL — ubiquinone reduction)\n\n"
                "Loss of NDUFS8:\n"
                "  • BOTH N6a AND N6b absent → two consecutive relay steps fail\n"
                "  • Electrons from NDUFS7/N4 cannot propagate toward NDUFS2/N2\n"
                "  • Dual block is functionally more disruptive than single-cluster loss\n"
                "  • BN-PAGE: absent/severely reduced CI, similar to NDUFS7 (cleaner than NDUFS3)\n"
                "  • Biochemical result: identical isolated CI deficiency (5–20%)"
            ),
        },
        {
            "term": "CI Fe-S Relay N3→N1b→N4→N6a→N6b→N5→N2→UQ — NDUFS8 in Context",
            "definition": (
                "The full electron relay chain through Complex I Fe-S clusters:\n\n"
                "  NADH → FMN (NDUFV1) → N3 → N1b (NDUFV2) → N4 (NDUFS7) "
                "→ N6a (NDUFS8) → N6b (NDUFS8) → N5 (NDUFS1) → N2 (NDUFS2) → Ubiquinone\n\n"
                "NDUFS8 position: relay steps 4 and 5 of 8 (N6a + N6b clusters).\n\n"
                "Clinical consequence of N6a/N6b loss:\n"
                "  • The entire downstream relay (N5, N2) is starved of electrons\n"
                "  • Ubiquinone cannot be reduced despite N2 (NDUFS2) being structurally intact\n"
                "  • Electrons pile up at N4 (NDUFS7) → increased ROS\n"
                "  • NADH cannot be reoxidised → lactic acidosis\n\n"
                "Therapeutic implication: succinate (CII bypass) donates electrons directly "
                "to ubiquinol, bypassing the entire NDUFS8 N6a/N6b dual block."
            ),
        },
        {
            "term": "NDUFS8 vs NDUFS7 vs NDUFS2 vs NDUFS1 vs NDUFV1 vs NDUFS4 — CI-Leigh Series",
            "definition": (
                "All cause isolated CI deficiency + Leigh syndrome.\n"
                "Biochemical fingerprint is identical: CI 5–20%, CII/CIII/CIV normal.\n"
                "Clinical differentiation:\n\n"
                "NDUFS4 (175 aa, accessory, 5q11.2):\n"
                "  • Olfactory bulb MRI: ~52–65% (PATHOGNOMONIC — not seen in NDUFS8)\n\n"
                "NDUFV1 (464 aa, N-module FMN/N3, 11q13.2):\n"
                "  • Leukodystrophy / white matter T2: ~40–50% (DISTINGUISHING)\n\n"
                "NDUFS1 (727 aa, IP1/75kDa, 2q33.3):\n"
                "  • Peripheral neuropathy: ~50% (DISTINGUISHING — not seen in NDUFS8)\n\n"
                "NDUFS7 (213 aa, N4/20kDa, 19p13.3):\n"
                "  • Single N4 Fe-S cluster block vs NDUFS8 dual N6a+N6b block\n"
                "  • Clinically identical to NDUFS8; BN-PAGE similar; requires genetics\n\n"
                "NDUFS2 (463 aa, PSST/N2/49kDa, 1q23.3):\n"
                "  • Terminal N2 Fe-S carrier loss; NDUFS8 is one step upstream (N6a/N6b)\n"
                "  • Clinically indistinguishable; requires WES/panel\n\n"
                "NDUFS3 (264 aa, QP-C/30kDa, 11p11.11):\n"
                "  • Q-module ASSEMBLY SCAFFOLD → CI sub-assembly intermediates on BN-PAGE\n"
                "  • NDUFS8 = direct dual N6a/N6b Fe-S relay block → cleaner BN-PAGE\n\n"
                "NDUFS8 (201 aa, TYKY/N6a-N6b, 11q13.2) — THIS DISEASE:\n"
                "  • Dual N6a + N6b Fe-S electron relay block (4th and 5th relay steps)\n"
                "  • NO peripheral neuropathy (KEY DDx vs NDUFS1)\n"
                "  • NO olfactory bulb lesions (KEY DDx vs NDUFS4)\n"
                "  • NO leukodystrophy (KEY DDx vs NDUFV1)\n"
                "  • HCM ~5% (lower than NDUFS7 ~6%, NDUFS3 ~10%, NDUFS1 ~12%)"
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "NDUFS8 N6a/N6b Fe-S Dual Electron Transfer Block and CI Deficiency",
            "definition": (
                "NDUFS8 biallelic variants → loss of N6a AND N6b [4Fe-4S] clusters → CI dual electron relay block.\n\n"
                "The N6a/N6b block cascade:\n"
                "  1. NDUFS8 absent/non-functional → N6a and N6b Fe-S clusters absent\n"
                "  2. Electrons from NDUFS7/N4 (Q/N-junction) cannot reach NDUFS1/N5\n"
                "  3. NDUFS2/N2 (terminal carrier) is starved — ubiquinone cannot be reduced\n"
                "  4. CI overall: 5–20% residual (partial activity from alternative pathways)\n"
                "  5. Residual NADH cannot be re-oxidised at full rate → NADH/NAD+ ↑\n"
                "  → lactate/pyruvate ratio ↑ → lactic acidosis\n\n"
                "Biochemical criteria:\n"
                "  Complex I activity: 5–20% of age-matched controls\n"
                "  Complexes II, III, IV: NORMAL\n"
                "  Lactate/pyruvate ratio: typically >20\n"
                "  Plasma lactate: 4.5–22 mmol/L\n"
                "  BN-PAGE: absent/severely reduced CI; cleaner than NDUFS3 scaffold loss\n\n"
                "Inheritance: AR — biallelic. Siblings at 25% risk."
            ),
        },
        {
            "term": "DDx — NDUFS8 vs NDUFS7 vs NDUFS4 vs NDUFV1 vs NDUFS1 vs NDUFS2 vs NDUFS3",
            "definition": (
                "All produce isolated CI deficiency + Leigh MRI. Key distinguishing features:\n\n"
                "NDUFS4: Olfactory bulb lesions (52–65%) — ABSENT in NDUFS8\n"
                "NDUFV1: Leukodystrophy (40–50%) — ABSENT in NDUFS8\n"
                "NDUFS1: Peripheral neuropathy (50%) — ABSENT in NDUFS8\n"
                "NDUFS7: N4 single-cluster block (junction) vs NDUFS8 dual N6a+N6b block (approach)\n"
                "         Clinically indistinguishable; BN-PAGE similar; requires genetics\n"
                "NDUFS2: Terminal N2 loss (ubiquinone reduction step) vs NDUFS8 N6a/N6b\n"
                "         (one/two steps upstream); clinically identical without genetics\n"
                "NDUFS3: Q-module scaffold failure (assembly intermediates on BN-PAGE)\n"
                "         vs NDUFS8 direct dual Fe-S electron relay block (cleaner BN-PAGE)\n\n"
                "NDUFS8 DDx fingerprint:\n"
                "  ✓ Isolated CI deficiency (5–20%)\n"
                "  ✓ Leigh MRI (bilateral putamen + brainstem)\n"
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
            "term": "Complete Prescribing Safety Card — NDUFS8 CI-Leigh",
            "definition": (
                "ABSOLUTE CONTRAINDICATIONS (never use):\n"
                "  ▪ Valproate / VPA (triple mito-toxicity: CoA/POLG/hepatotoxicity)\n"
                "  ▪ Metformin (direct CI inhibitor at ND1/quinone-binding site; "
                "    NDUFS8/N6a-N6b already blocks N2→UQ — metformin removes residual)\n"
                "  ▪ Linezolid (23S rRNA → blocks all 7 mtDNA ND subunits → CI loss)\n"
                "  ▪ Chloramphenicol (same mt-ribosome mechanism as linezolid)\n"
                "  ▪ Ketogenic Diet (forces NADH overload → N6a/N6b-blocked relay worsened)\n\n"
                "HIGH CAUTION / AVOID:\n"
                "  ▪ Propofol (PRIS: CIV inhibition creates 2nd ETC block downstream of NDUFS8 N6)\n"
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
