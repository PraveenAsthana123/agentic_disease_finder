#!/usr/bin/env python3
"""NDUFV2 — Leigh Syndrome Isolated Complex I Deficiency (N1b [2Fe-2S] Fe-S Cluster / 24 kDa N-module).

NDUFV2 (NADH:Ubiquinone Oxidoreductase Core Subunit V2), also known as the 24 kDa subunit,
is an N-module subunit of Complex I that carries the sole [2Fe-2S] N1b iron-sulfur cluster —
the SECOND electron relay step in the CI Fe-S chain (after NDUFV1/FMN/N3 and before NDUFS7/N4).
The N1b cluster sits structurally close to the NADH oxidation site; loss of NDUFV2 blocks the
second electron relay step, producing isolated CI deficiency identical biochemically to other
CI-Leigh Fe-S relay subunit diseases.

  NDUFV2 gene      OMIM *600532
  Disease          Leigh Syndrome (OMIM #256000) /
                   Mitochondrial Complex I Deficiency, Nuclear Type 4 (OMIM #618259)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       18p11.22

PATHOPHYSIOLOGY (Complex I / N1b [2Fe-2S] Fe-S cluster / NDUFV2 N-module role):
  The Fe-S electron relay chain of Complex I transfers electrons from NADH → ubiquinone
  through a series of iron-sulfur clusters in the following order:

    NDUFV1  (N3, [4Fe-4S]) ← FMN primary NADH acceptor (N-module)
    NDUFV2  (N1b, [2Fe-2S]) ← SECOND relay step (N-module) — THIS SUBUNIT
    NDUFS7  (N4, [4Fe-4S]) ← third relay (Q-module/N-module junction)
    NDUFS8  (N6a, [4Fe-4S]) ← fourth relay — Q-module approach (TYKY)
    NDUFS8  (N6b, [4Fe-4S]) ← fifth relay — same subunit
    NDUFS1  (N5, [4Fe-4S]) ← peripheral arm relay
    NDUFS2  (N2, [4Fe-4S]) ← TERMINAL carrier → ubiquinone reduction (Q-module)

  NDUFV2's critical N1b [2Fe-2S] relay role:
    1. N1b is the only [2Fe-2S] cluster in the entire CI Fe-S relay chain
    2. N1b sits structurally adjacent to the NDUFV1 FMN NADH-oxidation site in the N-module
    3. Loss of NDUFV2 → N1b absent → electrons from NDUFV1/N3 (FMN NADH acceptor) cannot
       propagate toward the NDUFS7/N4 cluster
    4. DIRECT SINGLE ELECTRON TRANSFER BLOCK: N1b loss = second relay step failure,
       but downstream relays (N4, N6a, N6b, N5, N2) remain structurally present
    5. BN-PAGE: absent/severely reduced CI; cleaner CI loss pattern (Fe-S relay failure
       rather than scaffold/assembly failure like NDUFS3)

  Biochemical signature (IDENTICAL to all CI-Leigh Fe-S relay mutations):
    Complex I: 5–20% of control (severely reduced) — ISOLATED CI DEFICIENCY
    Complex II (SDHA): NORMAL (fully nuclear-encoded; NDUFV2 not required for CII)
    Complex III: NORMAL
    Complex IV (COX): NORMAL — KEY DDx SURF1/SCO2/COX10/COX15 (CIV-only deficiency)

DISTINGUISHING FEATURES vs NDUFV1/NDUFS4/NDUFS1/NDUFS7/NDUFS8/SCO2:
  vs NDUFV1 (N-module FMN/N3, 11q13.2):
    • NDUFV1: ~40–50% leukodystrophy/WM T2 — DISTINGUISHING; NDUFV2: rare WM (<5%)
    • Both N-module; NDUFV2 N1b sits downstream of NDUFV1 N3 in the relay
    • HCM ~80% in NDUFV2 vs ~8–10% in NDUFV1 — KEY DISTINGUISHER
  vs NDUFS4 (N-module accessory, 5q11.2):
    • NO olfactory bulb lesions in NDUFV2 (NDUFS4: ~52–65% — PATHOGNOMONIC)
    • HCM ~80% NDUFV2 vs very low in NDUFS4
  vs NDUFS1 (IP1/75kDa/N-module Fe-S, 2q33.3):
    • NO peripheral neuropathy in NDUFV2 (NDUFS1: ~50% — CRITICAL DDx)
    • HCM ~80% NDUFV2 vs rare in NDUFS1
  vs NDUFS7/NDUFS8 (Q-module junction / Q-module approach):
    • All produce isolated CI + Leigh; NDUFV2 = N1b (2nd step); NDUFS7 = N4 (3rd);
      NDUFS8 = N6a/N6b (4th/5th)
    • HCM ~80% in NDUFV2 vs ~6–7% in NDUFS7/NDUFS8 — KEY DISTINGUISHER
  vs SCO2 (COX assembly, 22q13.33):
    • SCO2: HCM 100% but CIV deficiency — NOT CI; COX/CIV reduced
    • NDUFV2: HCM 80% but ISOLATED CI deficiency (CII/CIII/CIV NORMAL) — biochemical
      fingerprint differentiates; SCO2 HCM > NDUFV2 HCM but CI vs CIV is key

DISTINCTIVE CARDIAC PHENOTYPE — HCM ~80%:
  NDUFV2 is unique among CI Fe-S relay subunits in its HIGH rate of hypertrophic
  cardiomyopathy (~80%). This is the highest HCM rate in the CI Fe-S relay series:
    NDUFS7:  ~6%    NDUFS8:  ~5%    NDUFS2:  ~8%    NDUFS3:  ~10%
    NDUFV1:  ~8–10% NDUFS4:  low    NDUFS1:  rare
  NDUFV2:  ~80% — DISTINCTIVE, comparable only to SCO2 (100% CIV) and COX15 (cardiac)
  Mechanism: N1b [2Fe-2S] loss → severe NADH/NAD+ imbalance in cardiomyocytes (which
  have the highest CI-dependent OXPHOS demand) → compensatory hypertrophy → HCM + LVOT
  Digoxin is ABSOLUTE CI (positive inotrope + LVOT obstruction in HCM).
  Propranolol is first-line for HCM (beta-blocker, reduces LVOT gradient).

FOUNDER / RECURRENT MUTATIONS:
  p.Ala59Val   c.176C>T — N1b [2Fe-2S] binding region; severe infantile; Haut 2003 series
  p.Pro19Leu   c.56C>T  — signal peptide; severe neonatal; HCM 100%
  p.Arg193Cys  c.577C>T — consanguineous hom; HCM 90%; Leigh MRI 75%; moderate-severe
  p.Glu178Lys  c.532G>A — compound het with Ala59Val; intermediate; partial N1b retention
  p.Thr155Met  c.464C>T — signal peptide cleavage impaired; milder N1b retention

THERAPY — NDUFV2/CI-LEIGH SPECIFICS:
  No targeted NDUFV2 N1b [2Fe-2S] cluster reconstitution is clinically available.
  Management follows the CI-Leigh supportive protocol (cofactors + avoid CI toxins).
  Succinate bypass (Level C): CII-mediated electron entry to ubiquinol bypasses the
    NDUFV2 N1b block ENTIRELY — electrons enter at ubiquinol directly via SDHA.
  Riboflavin B2 (Level C): FMN reinforces NDUFV1/N3 upstream of NDUFV2/N1b; some
    theoretical benefit at adjacent NDUFV1 site. Most targeted in NDUFV1 deficiency.
  Propranolol: first-line beta-blocker for HCM/LVOT obstruction (distinctive ~80% HCM).
  Digoxin: ABSOLUTE CI — positive inotrope in HCM with LVOT obstruction = catastrophic.
  No peripheral neuropathy: physiotherapy focuses on central ataxia/dystonia only.

References:
  Haut S et al. Pediatrics. 2003;111(4):197-201.
    (First NDUFV2 CI deficiency reported)
  van den Heuvel LP et al. Ann Neurol. 1998;44(6):975-977.
    (CI subunit mutation series)
  Fassone E, Rahman S. J Med Genet. 2012;49(9):578-590.
    (CI deficiency genetics review — NDUFV2 in context of Fe-S relay series)
  Sazanov LA. Nat Rev Mol Cell Biol. 2015;16(6):375-388.
    (CI structural review — N1b position in N-module, adjacent to FMN site)
  Bénit P et al. Am J Hum Genet. 2003;72(6):1344-1351.
    (CI subunit mutations systematic series)
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 621
DISEASE_ID   = "ndufv2"
DISEASE_NAME = (
    "NDUFV2 Leigh Syndrome — Isolated Complex I Deficiency "
    "(CI-Leigh / NDUFV2 24 kDa N-module N1b [2Fe-2S] Fe-S Cluster / 2nd Electron Relay Step)"
)
GENE         = "NDUFV2"
PROTEIN      = (
    "NDUFV2 — 249 aa precursor / ~24 kDa mature, N-module subunit, "
    "carries single N1b [2Fe-2S] Fe-S cluster — 2nd electron relay step "
    "from NDUFV1/FMN/N3 toward NDUFS7/N4 → NDUFS8/N6a-N6b → NDUFS2/N2 → ubiquinone"
)
OMIM_GENE    = "*600532"
OMIM_DISEASE = "#256000 (Leigh Syndrome) / Mitochondrial Complex I Deficiency Nuclear Type 4 (#618259)"
CHROMOSOME   = "18p11.22"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Infantile / neonatal (typically 0–12 months)"
COHORT_SIZE  = 40
COLOR        = "#1a237e"   # dark indigo — N1b [2Fe-2S] 2nd relay / NDUFV2 N-module theme
LIGHT        = "#e8eaf6"

# Genotype pool (from task specification)
GENO_POOL = [
    "p.Ala59Val / p.Arg193Cys (c.176C>T / c.577C>T) — compound het; N1b [2Fe-2S] binding region; severe infantile; Haut 2003 series",
    "p.Pro19Leu / p.Ala59Val (c.56C>T / c.176C>T) — signal peptide + N1b domain; severe neonatal; HCM 100%",
    "p.Arg193Cys homozygous (c.577C>T hom) — consanguineous; HCM 90%; Leigh MRI 75%; moderate-severe",
    "p.Glu178Lys / p.Ala59Val (c.532G>A / c.176C>T) — compound het; intermediate; partial N1b Fe-S retention",
    "p.Thr155Met / p.Pro19Leu (c.464C>T / c.56C>T) — signal peptide cleavage impaired; milder N1b retention",
]
GENO_WEIGHTS = [0.30, 0.22, 0.20, 0.16, 0.12]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG — reproducible 40-patient NDUFV2/CI-Leigh cohort (seed-621)."""
    return random.Random(SEED)


# ── Patient cohort ────────────────────────────────────────────────────────────
_TX_POOL = [
    "IV Dextrose GIR 6-8 (metabolic crisis — first-line)",
    "Riboflavin B2 (100–400 mg/day — CI-specific FMN precursor; adjacent to NDUFV2 N1b upstream)",
    "CoQ10/Ubiquinol (300–600 mg/day)",
    "Thiamine B1 (mandatory empiric, all Leigh until SLC19A3/BTD excluded)",
    "Biotin (mandatory empiric, all Leigh until BTD ruled out)",
    "Succinate (IV/oral — Complex II bypass; bypasses blocked NDUFV2 N1b [2Fe-2S] cluster entirely)",
    "NaHCO3 (lactic acidosis correction, pH <7.20)",
    "LEV (seizures — preferred AED, renal excretion, no mito toxicity)",
    "NIV/BiPAP (central respiratory compromise)",
    "Carnitine (secondary deficiency, 50–100 mg/kg/day)",
    "Propranolol (beta-blocker for HCM/LVOT obstruction — first-line cardiac)",
    "Physiotherapy / occupational therapy (central ataxia + dystonia — no peripheral neuropathy)",
    "Enteral feeds / NG (avoid fasting; high-carbohydrate; KD strictly CI)",
    "Cardiac monitoring / echocardiography (HCM surveillance every 6–12 months)",
]

_OUTCOMES = [
    "Alive — stable moderate-to-severe Leigh + HCM; cofactor + cardiac + respiratory support",
    "Alive — prolonged survival with partial N1b Fe-S residual; ongoing HCM management; school-age",
    "Alive — severe Leigh + HCM requiring propranolol; dependent care; ongoing multidisciplinary support",
    "Died — acute lactic crisis + respiratory failure in first 12 months (null/severe alleles)",
    "Died — progressive Leigh + HCM decompensation + respiratory failure; 1–3 yr trajectory",
]
_OUT_WEIGHTS = [0.25, 0.12, 0.20, 0.25, 0.18]


def _generate_cohort(rng: random.Random) -> list[dict[str, Any]]:
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno        = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex         = rng.choice(["M", "F"])
        onset_mo    = rng.choices(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 24, 30],
            weights=[4, 5, 8, 10, 11, 10, 9, 8, 7, 8, 6, 5, 4, 5],
        )[0]
        onset_yr    = round(onset_mo / 12, 1)
        lactate_mm  = round(rng.uniform(5.0, 24.0), 1)
        ci_pct      = round(rng.uniform(5.0, 21.0))          # Complex I (% of control)
        cii_pct     = round(rng.uniform(88, 108))             # Complex II — NORMAL
        civ_pct     = round(rng.uniform(85, 105))             # Complex IV — NORMAL

        has_leigh_mri       = rng.random() < 0.78
        has_regression      = rng.random() < 0.97
        has_hypotonia       = rng.random() < 0.88
        has_lactic          = rng.random() < 0.90
        has_resp            = rng.random() < 0.50
        has_seizures        = rng.random() < 0.45
        has_ataxia          = rng.random() < 0.38
        has_dystonia        = rng.random() < 0.30
        has_myoclonus       = rng.random() < 0.18
        has_nystagmus       = rng.random() < 0.18
        has_optic           = rng.random() < 0.22
        has_spasticity      = rng.random() < 0.30
        has_hcm             = rng.random() < 0.80   # ~80% — DISTINCTIVE, highest in CI Fe-S relay series
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
        if has_hcm:            feat_list.append("HCM ~80% — DISTINCTIVE (highest CI Fe-S relay series)")
        if has_resp:           feat_list.append("Respiratory compromise (central)")
        if has_seizures:       feat_list.append("Seizures")
        if has_ataxia:         feat_list.append("Ataxia")
        if has_dystonia:       feat_list.append("Dystonia")
        if has_myoclonus:      feat_list.append("Myoclonus")
        if has_nystagmus:      feat_list.append("Nystagmus")
        if has_optic:          feat_list.append("Optic atrophy")
        if has_spasticity:     feat_list.append("Spasticity")
        if has_hepatopathy:    feat_list.append("Hepatopathy (RARE — KEY DDx POLG/DGUOK)")

        txs     = rng.sample(_TX_POOL, k=rng.randint(4, 7))
        outcome = rng.choices(_OUTCOMES, weights=_OUT_WEIGHTS)[0]

        patients.append({
            "id":                  f"NDUFV2-{i:03d}",
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
        "HCM (~80% — DISTINCTIVE: highest among CI Fe-S relay subunits)":       _pct("has_hcm"),
        "Hypotonia":                                                             _pct("has_hypotonia"),
        "Lactic Acidosis (elevated baseline + crisis)":                          _pct("has_lactic"),
        "Respiratory Compromise (central)":                                      _pct("has_resp"),
        "Seizures":                                                              _pct("has_seizures"),
        "Ataxia":                                                                _pct("has_ataxia"),
        "Dystonia":                                                              _pct("has_dystonia"),
        "Optic Atrophy":                                                         _pct("has_optic"),
        "Spasticity":                                                            _pct("has_spasticity"),
        "Myoclonus":                                                             _pct("has_myoclonus"),
        "Nystagmus":                                                             _pct("has_nystagmus"),
        "Hepatopathy (RARE — KEY DDx POLG / DGUOK)":                            _pct("has_hepatopathy"),
        "NO Peripheral Neuropathy (KEY DDx NDUFS1 ~50% neuropathy)":            100,
        "NO Olfactory Bulb Lesions (KEY DDx NDUFS4 ~52–65%)":                   100,
        "NO Leukodystrophy (KEY DDx NDUFV1 ~40–50%)":                           100,
        "NO Iron Overload (KEY DDx GRACILE — 100% iron overload)":              100,
        "NO COX Deficiency (KEY DDx SURF1 / SCO2 / COX10 / COX15)":            100,
        "Alive (with support)":                                                  round(alive / COHORT_SIZE * 100),
    }

    kpis = [
        {"label": "Cohort (n)",       "value": COHORT_SIZE,                                                                         "color": COLOR},
        {"label": "HCM",              "value": f"{feature_frequencies['HCM (~80% — DISTINCTIVE: highest among CI Fe-S relay subunits)']}% — DISTINCTIVE", "color": "#ad1457"},
        {"label": "Leigh MRI",        "value": f"{feature_frequencies['Leigh / Leigh-like MRI (bilateral putamen + brainstem)']}%",  "color": "#4a148c"},
        {"label": "No Neuropathy",    "value": "100% — DDx NDUFS1",                                                                 "color": "#2e7d32"},
        {"label": "Resp Compromise",  "value": f"{feature_frequencies['Respiratory Compromise (central)']}%",                       "color": "#b71c1c"},
        {"label": "CI Activity",      "value": "5–20% control",                                                                     "color": "#c62828"},
        {"label": "Fatal",            "value": f"{round(died/COHORT_SIZE*100)}%",                                                   "color": "#b71c1c"},
        {"label": "Seed",             "value": f"#{SEED}",                                                                         "color": "#455a64"},
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "severity": "ABSOLUTE CI — DANGEROUS IN ALL LEIGH / CI DEFICIENCY",
            "mechanism": (
                "Triple mechanism in NDUFV2/CI-Leigh:\n"
                "1. CoA SEQUESTRATION: valproyl-CoA depletes the mitochondrial CoA pool "
                "   → TCA cycle collapses → OXPHOS substrate loss. In NDUFV2 deficiency "
                "   (CI already at 5–20% due to N1b [2Fe-2S] electron transfer block), "
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
            "severity": "ABSOLUTE CI — DIRECT COMPLEX I INHIBITOR: CATASTROPHIC IN NDUFV2",
            "mechanism": (
                "Metformin inhibits Complex I (NADH dehydrogenase) at the ND1/quinone "
                "binding site.\n"
                "In NDUFV2 deficiency: CI N1b [2Fe-2S] cluster is absent/non-functional "
                "(5–20% activity). The N1b block means electrons from NDUFV1/N3 cannot "
                "propagate toward NDUFS7/N4 and subsequently to N2 (NDUFS2)/ubiquinone. "
                "Metformin's CI inhibition blocks residual ND1 quinone-binding capacity "
                "→ near-total CI shutdown → massive lactate accumulation (>15–20 mmol/L).\n\n"
                "Alternative for glucose management: insulin only. Never metformin."
            ),
        },
        {
            "drug": "Digoxin",
            "severity": "ABSOLUTE CI — HCM + LVOT Obstruction (NDUFV2 HCM ~80%)",
            "mechanism": (
                "NDUFV2 produces HCM in ~80% of patients — the highest rate among CI Fe-S "
                "relay subunit diseases. HCM in NDUFV2 typically involves LVOT (left ventricular "
                "outflow tract) obstruction.\n"
                "Digoxin is a positive inotrope: increases contractility → WORSENS LVOT "
                "obstruction in hypertrophic obstructive cardiomyopathy → acute haemodynamic "
                "collapse and sudden death risk.\n"
                "First-line: propranolol (beta-blocker reduces LVOT gradient + HR control).\n"
                "Second-line: verapamil or disopyramide (if propranolol insufficient).\n"
                "NEVER digoxin, NEVER inotropes in HCM with LVOT obstruction."
            ),
        },
        {
            "drug": "Linezolid",
            "severity": "ABSOLUTE CI — mt-Ribosome 23S rRNA Block (ALL 7 ND mtDNA Subunits)",
            "mechanism": (
                "Linezolid inhibits the mitochondrial large ribosomal subunit (mt-LSU) 23S rRNA "
                "→ blocks synthesis of ALL 13 mitochondrially-encoded OXPHOS subunits.\n"
                "In NDUFV2/CI-Leigh:\n"
                "  • NDUFV2 itself is nuclear-encoded — BUT Complex I also requires 7 mtDNA-encoded "
                "    ND subunits (MT-ND1–6, MT-ND4L) for the P-module membrane arm\n"
                "  • Linezolid blocks synthesis of ALL 7 → CI P-module cannot assemble "
                "    → NDUFV2 N1b relay subunit has no functional context → CI → zero\n"
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
                "In NDUFV2 CI-Leigh: CI N1b [2Fe-2S] cluster is absent — the electron "
                "relay from NDUFV1/N3 toward NDUFS7/N4 is broken (5–20% activity). "
                "Forcing KD → NADH accumulation → NAD+ depletion → TCA + beta-oxidation "
                "stall → worsened lactic acidosis.\n\n"
                "KD is beneficial in GLUT1-DS and PDHD — CONTRAINDICATED in CI deficiency."
            ),
        },
        {
            "drug": "Propofol",
            "severity": "AVOID / HIGH CAUTION — PRIS in CI-Leigh + HCM amplifies risk",
            "mechanism": (
                "Propofol Infusion Syndrome (PRIS): propofol inhibits Complex IV and "
                "uncouples fatty acid beta-oxidation.\n"
                "In NDUFV2/CI-Leigh: CI is the primary ETC bottleneck (N1b electron relay broken). "
                "Propofol's CIV inhibition creates a SECOND downstream ETC block → electrons "
                "trapped between N1b-deficient CI and CIV → ROS burst.\n"
                "Additionally: NDUFV2 patients with HCM (~80%) have pre-existing cardiac "
                "vulnerability — propofol's negative inotropy risks haemodynamic collapse.\n"
                "Use sevoflurane (volatile, no PRIS) or dexmedetomidine as alternatives."
            ),
        },
        {
            "drug": "Phenobarbital",
            "severity": "HIGH CAUTION — Secondary Complex I Inhibitor",
            "mechanism": (
                "Phenobarbital inhibits CI electron transport (quinone channel region). "
                "In NDUFV2/CI-Leigh: residual CI (5–20%) is critical. Phenobarbital's "
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
                "CI assembly collapses → catastrophic in NDUFV2/CI deficiency."
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
        "Psychomotor Regression":                                             _pct("has_regression"),
        "Leigh / Leigh-like MRI":                                             _pct("has_leigh_mri"),
        "HCM (~80% — DISTINCTIVE, highest in CI Fe-S relay series)":         _pct("has_hcm"),
        "Hypotonia":                                                          _pct("has_hypotonia"),
        "Lactic Acidosis":                                                    _pct("has_lactic"),
        "Respiratory Compromise":                                             _pct("has_resp"),
        "Seizures":                                                           _pct("has_seizures"),
        "Ataxia":                                                             _pct("has_ataxia"),
        "Dystonia":                                                           _pct("has_dystonia"),
        "Optic Atrophy":                                                      _pct("has_optic"),
        "Spasticity":                                                         _pct("has_spasticity"),
        "Myoclonus":                                                          _pct("has_myoclonus"),
        "Nystagmus":                                                          _pct("has_nystagmus"),
        "Hepatopathy (RARE)":                                                 _pct("has_hepatopathy"),
        "Peripheral Neuropathy (NEVER — KEY DDx NDUFS1 ~50%)":               0,
        "Olfactory Bulb MRI (NEVER — KEY DDx NDUFS4 52–65%)":               0,
        "Leukodystrophy / White Matter (RARE <5% — KEY DDx NDUFV1 40–50%)":  _pct("has_leukodystrophy"),
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
            "term": "Valproate / VPA — ABSOLUTE CI in NDUFV2/CI-Leigh",
            "definition": (
                "NEVER use valproate in any patient with NDUFV2 mutation or suspected CI-Leigh.\n\n"
                "Three independent lethal mechanisms:\n"
                "1. CoA SEQUESTRATION (valproyl-CoA): depletes mitochondrial CoA → TCA cycle "
                "   substrates drop → OXPHOS cannot produce ATP → lactic crisis in CI-deficient cell.\n"
                "2. POLG INHIBITION: valproate inhibits mtDNA polymerase gamma → mtDNA depletion "
                "   → fewer MT-ND1–ND6 / MT-ND4L templates → fewer P-module CI subunits "
                "   → CI further compromised in NDUFV2 N1b [2Fe-2S] deficient state.\n"
                "3. HEPATOTOXICITY: VPA directly hepatotoxic → hepatic OXPHOS failure "
                "   → impaired gluconeogenesis → hypoglycaemia → crisis amplified.\n\n"
                "Preferred AED: LEVETIRACETAM (LEV) — renal excretion, zero mito toxicity. "
                "Second-line: clobazam (CLB), clonazepam — no mito toxicity."
            ),
        },
        {
            "term": "Digoxin — ABSOLUTE CI in NDUFV2 (HCM ~80% / LVOT obstruction)",
            "definition": (
                "NDUFV2 causes HCM in ~80% of patients — the highest rate in the CI Fe-S relay series.\n"
                "HCM in NDUFV2 is typically hypertrophic OBSTRUCTIVE cardiomyopathy (HOCM) with "
                "LVOT (left ventricular outflow tract) obstruction.\n\n"
                "DIGOXIN MECHANISM AND DANGER:\n"
                "  Digoxin is a positive inotrope (increases systolic contractility via Na+/K+-ATPase "
                "  inhibition → intracellular Ca2+ ↑).\n"
                "  In HOCM: increased contractility → septal hypertrophy contracts MORE forcefully "
                "  → LVOT obstruction WORSENS → acute haemodynamic collapse → sudden death.\n\n"
                "CORRECT TREATMENT:\n"
                "  • Propranolol: beta-blocker, first-line — reduces HR, decreases contractility,\n"
                "    prolongs diastolic filling, reduces LVOT gradient.\n"
                "  • Verapamil (non-DHP CCB): alternative if beta-blocker not tolerated.\n"
                "  • Disopyramide: Class Ia antiarrhythmic; reduces LVOT gradient (second-line).\n"
                "  • AVOID all positive inotropes: digoxin, dopamine, dobutamine.\n"
                "  • AVOID vasodilators: nitrates, ACE inhibitors in obstructive HCM.\n"
                "  • AVOID dehydration/hypovolaemia: worsens LVOT gradient."
            ),
        },
        {
            "term": "Succinate — Level C (Complex II Bypass — Bypasses NDUFV2 N1b Cluster Entirely)",
            "definition": (
                "MECHANISTIC BASIS:\n"
                "  Succinate → Complex II (succinate dehydrogenase / SDHA) → ubiquinol "
                "  → Complex III → CIV → O2.\n"
                "  This bypasses the NDUFV2 N1b [2Fe-2S] electron transfer block ENTIRELY: "
                "  neither NDUFV2 (N1b relay), NDUFV1 (N3/FMN), nor NDUFS7/N4, NDUFS8/N6a-N6b, "
                "  NDUFS2/N2 are required for CII→ubiquinol electron entry.\n\n"
                "CLINICAL RATIONALE:\n"
                "  In CI deficiency with N1b block, maintaining ubiquinol pool via CII sustains "
                "  partial ATP synthesis without relying on the broken NDUFV2 N1b Fe-S relay.\n"
                "  Level C evidence; dose 2–8 g/day oral succinate or sodium succinate.\n"
                "  Most relevant in crisis: IV succinate (where available).\n\n"
                "CAUTION: Not a replacement for glucose (GIR 6-8 during crisis is first-line)."
            ),
        },
        {
            "term": "Riboflavin / Vitamin B2 — Level C (CI-specific, adjacent to NDUFV2 N1b at NDUFV1 N3)",
            "definition": (
                "Riboflavin → riboflavin-5'-phosphate (FMN) + FAD.\n\n"
                "CI relevance (N1b proximity to NDUFV1 FMN site):\n"
                "  FMN binds NDUFV1 (51kDa/N3 subunit) — the FIRST electron acceptor from NADH.\n"
                "  NDUFV2/N1b is the IMMEDIATE downstream relay step from NDUFV1/N3.\n"
                "  Extra FMN may stabilise NDUFV1 and improve electron injection at N3; however, "
                "  in NDUFV2 deficiency the N1b cluster is absent — electrons from N3 still "
                "  cannot propagate toward NDUFS7/N4 → limited direct benefit.\n\n"
                "NOTE: Riboflavin is more directly targeted in NDUFV1 deficiency (FMN active site). "
                "In NDUFV2 deficiency the N1b block cannot be repaired pharmacologically. "
                "Riboflavin used empirically alongside succinate. Level C.\n"
                "Dose: 100–400 mg/day."
            ),
        },
        {
            "term": "Propranolol — First-line HCM Management in NDUFV2",
            "definition": (
                "NDUFV2 produces HCM in ~80% of patients — the HIGHEST rate in the CI Fe-S "
                "relay series (NDUFS7: ~6%, NDUFS8: ~5%, NDUFS2: ~8%).\n\n"
                "MECHANISM OF HCM IN NDUFV2:\n"
                "  N1b [2Fe-2S] loss → CI activity 5–20% → NADH cannot be reoxidised at full "
                "  rate in cardiomyocytes (highest CI-dependent OXPHOS demand in the body) "
                "  → energy deficit + NADH/NAD+ imbalance → compensatory hypertrophy → HCM.\n\n"
                "PROPRANOLOL MECHANISM:\n"
                "  Non-selective beta-blocker: reduces HR → longer diastolic filling time "
                "  → reduced LVOT gradient; reduces contractility → reduces outflow obstruction.\n"
                "  Start: 1–2 mg/kg/day in 3–4 divided doses; titrate to effect.\n\n"
                "CARDIAC MONITORING:\n"
                "  Echocardiography every 6–12 months in all NDUFV2 patients.\n"
                "  Serial ECG for arrhythmia detection.\n"
                "  Cardiology co-management mandatory.\n\n"
                "CONTRAINDICATIONS IN NDUFV2 HCM:\n"
                "  Digoxin (ABSOLUTE CI), dopamine, dobutamine, nitrates, ACE inhibitors in HOCM."
            ),
        },
        {
            "term": "Thiamine (B1) + Biotin — MANDATORY Empiric in ALL Leigh Presentations",
            "definition": (
                "Before NDUFV2 genotype is confirmed, every infant with Leigh/Leigh-like MRI "
                "+ lactic acidosis + CI biochemistry MUST receive empiric thiamine AND biotin.\n\n"
                "THIAMINE B1:\n"
                "  SLC19A3 deficiency (biotin-thiamine responsive basal ganglia disease / BTBGD) "
                "  and PDHC deficiency can mimic NDUFV2-Leigh.\n"
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
            "term": "Acute Crisis Protocol — NDUFV2/CI-Leigh Metabolic Emergency",
            "definition": (
                "STEP 1 — IV DEXTROSE STAT:\n"
                "  GIR 6-8 mg/kg/min; NEVER fast. Trigger: deterioration, lactate >5, fever.\n\n"
                "STEP 2 — HOLD MITOCHONDRIAL TOXINS:\n"
                "  Check for inadvertent metformin, phenobarbital, linezolid, propofol, VPA, digoxin. "
                "  Stop immediately.\n\n"
                "STEP 3 — NaHCO3 IV (pH <7.20):\n"
                "  0.5–1 mEq/kg over 1–2h; lactate monitoring q2h; target pH >7.25.\n\n"
                "STEP 4 — IV RIBOFLAVIN + THIAMINE (100 mg each IV if available).\n\n"
                "STEP 5 — IV SUCCINATE (metabolic centre, if available):\n"
                "  0.5–1 g/kg/day → CII-bypass of NDUFV2 N1b [2Fe-2S] block entirely.\n\n"
                "STEP 6 — SEIZURES → LEV IV: 20–40 mg/kg loading. ABSOLUTE CI VPA.\n\n"
                "STEP 7 — RESPIRATORY → NIV/BiPAP: SpO2 <92% or RR >40.\n"
                "  Anaesthesia: sevoflurane, NOT propofol.\n\n"
                "STEP 8 — CARDIAC: if HCM present → propranolol as soon as stable.\n"
                "  NEVER digoxin. Echocardiography if haemodynamic instability.\n\n"
                "EMERGENCY CARD: IV dextrose immediately; NEVER VPA, metformin, linezolid, "
                "chloramphenicol, propofol, digoxin, or ketogenic diet. Contact metabolic neurology + cardiology."
            ),
        },
    ]

    gene_concepts = [
        {
            "term": "NDUFV2 Gene Structure and Expression (24 kDa N-module Subunit)",
            "definition": (
                "Gene: NDUFV2 (NADH:Ubiquinone Oxidoreductase Core Subunit V2)\n"
                "Also known as: 24 kDa subunit (bovine analog nomenclature)\n"
                "Chromosome: 18p11.22\n"
                "Protein: 249 aa precursor; ~24 kDa mature form (after signal peptide cleavage)\n\n"
                "NDUFV2 is one of three N-module subunits in the flavoprotein (FP) fraction "
                "of Complex I alongside NDUFV1 (51 kDa) and NDUFS1 (75 kDa). It carries "
                "the sole [2Fe-2S] N1b cluster in the CI Fe-S relay chain — the only "
                "[2Fe-2S] cluster; all others are [4Fe-4S].\n"
                "Ubiquitous expression across tissues; highest in heart and brain.\n\n"
                "OMIM *600532 (gene); #256000 (Leigh Syndrome) / #618259 (CI Deficiency Nuclear Type 4)\n\n"
                "Inheritance: AR — biallelic (compound heterozygous or homozygous) "
                "loss-of-function variants required for disease."
            ),
        },
        {
            "term": "NDUFV2 N1b [2Fe-2S] Cluster — Unique [2Fe-2S] Electron Relay Role",
            "definition": (
                "NDUFV2 carries the N1b [2Fe-2S] cluster — the ONLY [2Fe-2S] cluster in the "
                "Complex I Fe-S relay chain. All other Fe-S clusters are [4Fe-4S] type.\n\n"
                "Position in the Fe-S electron relay chain:\n"
                "  NDUFV1  → N3 [4Fe-4S] (FMN primary NADH acceptor)\n"
                "  NDUFV2  → N1b [2Fe-2S] (SECOND relay — THIS SUBUNIT)\n"
                "  NDUFS7  → N4 [4Fe-4S] (third relay — Q/N-module junction)\n"
                "  NDUFS8  → N6a [4Fe-4S] (fourth relay — Q-module approach)\n"
                "  NDUFS8  → N6b [4Fe-4S] (fifth relay — same subunit)\n"
                "  NDUFS1  → N5 [4Fe-4S] (peripheral arm relay)\n"
                "  NDUFS2  → N2 [4Fe-4S] (TERMINAL — ubiquinone reduction)\n\n"
                "Loss of NDUFV2:\n"
                "  • N1b absent → electrons from NDUFV1/N3 (FMN site) cannot propagate forward\n"
                "  • All downstream relay steps (N4, N6a, N6b, N5, N2) remain structurally intact\n"
                "  • N2→ubiquinone transfer fails due to upstream electron starvation\n"
                "  • BN-PAGE: absent/severely reduced CI; cleaner pattern (Fe-S relay block)\n"
                "  • Biochemical result: isolated CI deficiency (5–20%)"
            ),
        },
        {
            "term": "CI Fe-S Relay FMN→N3→N1b→N4→N6a→N6b→N5→N2→UQ — NDUFV2 in Context",
            "definition": (
                "The full electron relay chain through Complex I Fe-S clusters:\n\n"
                "  NADH → FMN (NDUFV1) → N3 → N1b (NDUFV2) → N4 (NDUFS7) "
                "→ N6a (NDUFS8) → N6b (NDUFS8) → N5 (NDUFS1) → N2 (NDUFS2) → Ubiquinone\n\n"
                "NDUFV2 position: relay step 2 of 8 (N1b [2Fe-2S] cluster).\n\n"
                "Clinical consequence of N1b loss:\n"
                "  • The entire downstream relay (N4, N6a, N6b, N5, N2) is starved of electrons\n"
                "  • Ubiquinone cannot be reduced despite N2 (NDUFS2) being structurally intact\n"
                "  • Electrons pile up at N3 (NDUFV1) → increased ROS at FMN site\n"
                "  • NADH cannot be reoxidised → severe lactic acidosis\n"
                "  • Cardiomyocytes (highest OXPHOS demand) → NADH/NAD+ imbalance → HCM\n\n"
                "Therapeutic implication: succinate (CII bypass) donates electrons directly "
                "to ubiquinol, bypassing the entire NDUFV2 N1b block."
            ),
        },
        {
            "term": "NDUFV2 vs NDUFV1 vs NDUFS7 vs NDUFS8 vs NDUFS1 vs NDUFS4 vs SCO2 — CI-Leigh & HCM DDx Series",
            "definition": (
                "All CI-Leigh causes isolated CI deficiency + Leigh syndrome (CI 5–20%, CII/CIII/CIV normal).\n"
                "SCO2 causes HCM + CIV deficiency (not CI).\n"
                "Clinical differentiation:\n\n"
                "NDUFV1 (464 aa, N-module FMN/N3, 11q13.2):\n"
                "  • Leukodystrophy / white matter T2: ~40–50% (DISTINGUISHING vs NDUFV2 <5%)\n"
                "  • HCM: ~8–10% (vs NDUFV2 ~80% — major distinguisher)\n\n"
                "NDUFS4 (175 aa, accessory, 5q11.2):\n"
                "  • Olfactory bulb MRI: ~52–65% (PATHOGNOMONIC — not seen in NDUFV2)\n\n"
                "NDUFS1 (727 aa, IP1/75kDa, 2q33.3):\n"
                "  • Peripheral neuropathy: ~50% (DISTINGUISHING — not seen in NDUFV2)\n\n"
                "NDUFS7 (213 aa, N4/20kDa, 19p13.3):\n"
                "  • Single N4 Fe-S cluster block (Q/N-junction) vs NDUFV2 N1b (2nd step)\n"
                "  • HCM: ~6% (vs NDUFV2 ~80%)\n\n"
                "NDUFS8 (201 aa, TYKY/N6a-N6b, 11q13.2):\n"
                "  • Dual N6a+N6b Fe-S block (4th/5th steps) vs NDUFV2 N1b (2nd step)\n"
                "  • HCM: ~5% (vs NDUFV2 ~80%)\n\n"
                "SCO2 (COX15 assembly, 22q13.33):\n"
                "  • HCM: 100% — but CIV deficiency (not CI); COX/CIV reduced on respiratory chain\n"
                "  • NDUFV2: HCM ~80% but ISOLATED CI (CII/CIII/CIV NORMAL) — biochemical fingerprint\n\n"
                "NDUFV2 (249 aa, N1b/[2Fe-2S], 18p11.22) — THIS DISEASE:\n"
                "  • N1b [2Fe-2S] electron relay block (2nd relay step)\n"
                "  • HCM ~80% — DISTINCTIVE, highest in CI Fe-S relay series\n"
                "  • NO peripheral neuropathy (KEY DDx vs NDUFS1)\n"
                "  • NO olfactory bulb lesions (KEY DDx vs NDUFS4)\n"
                "  • NO leukodystrophy / WM T2 (KEY DDx vs NDUFV1 40–50%)\n"
                "  • NO COX deficiency (KEY DDx vs SCO2 CIV-selective)"
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "NDUFV2 N1b [2Fe-2S] Electron Transfer Block and CI Deficiency",
            "definition": (
                "NDUFV2 biallelic variants → loss of N1b [2Fe-2S] cluster → CI 2nd-step electron relay block.\n\n"
                "The N1b block cascade:\n"
                "  1. NDUFV2 absent/non-functional → N1b [2Fe-2S] cluster absent\n"
                "  2. Electrons from NDUFV1/N3 (FMN NADH acceptor) cannot reach NDUFS7/N4\n"
                "  3. All downstream Fe-S clusters (N4, N6a, N6b, N5, N2) starved of electrons\n"
                "  4. NDUFS2/N2 (terminal carrier) cannot reduce ubiquinone\n"
                "  5. CI overall: 5–20% residual (partial activity from alternative pathways)\n"
                "  6. Residual NADH cannot be re-oxidised at full rate → NADH/NAD+ ↑\n"
                "  → lactate/pyruvate ratio ↑ → lactic acidosis\n"
                "  7. Cardiomyocytes (highest CI-dependent OXPHOS demand) → NADH/NAD+ imbalance "
                "     → severe compensatory hypertrophy → HCM ~80%\n\n"
                "Biochemical criteria:\n"
                "  Complex I activity: 5–20% of age-matched controls\n"
                "  Complexes II, III, IV: NORMAL\n"
                "  Lactate/pyruvate ratio: typically >20\n"
                "  Plasma lactate: 5–24 mmol/L\n"
                "  BN-PAGE: absent/severely reduced CI\n\n"
                "Inheritance: AR — biallelic. Siblings at 25% risk."
            ),
        },
        {
            "term": "DDx — NDUFV2 vs NDUFV1 vs NDUFS7 vs NDUFS8 vs NDUFS4 vs NDUFS1 vs SCO2",
            "definition": (
                "CI-Leigh genes produce identical biochemistry: CI 5–20%, CII/CIII/CIV NORMAL.\n"
                "SCO2 produces HCM + CIV deficiency (biochemically distinct).\n"
                "Key distinguishing features:\n\n"
                "NDUFS4: Olfactory bulb lesions (52–65%) — ABSENT in NDUFV2\n"
                "NDUFV1: Leukodystrophy (40–50%) — ABSENT in NDUFV2 (<5%)\n"
                "NDUFS1: Peripheral neuropathy (50%) — ABSENT in NDUFV2\n"
                "NDUFS7: N4 single-cluster block; HCM ~6% vs NDUFV2 HCM ~80%\n"
                "NDUFS8: Dual N6a+N6b block; HCM ~5% vs NDUFV2 HCM ~80%\n"
                "SCO2:   HCM 100% but CIV deficiency (not CI) — biochemical fingerprint differentiates\n\n"
                "NDUFV2 DDx fingerprint:\n"
                "  HCM ~80% — DISTINCTIVE (highest in CI Fe-S relay series)\n"
                "  Isolated CI deficiency (5–20%)\n"
                "  Leigh MRI (bilateral putamen + brainstem ~78%)\n"
                "  NO peripheral neuropathy\n"
                "  NO olfactory bulb lesions\n"
                "  NO leukodystrophy / WM T2 (<5%)\n"
                "  NO COX deficiency (vs SCO2)\n"
                "  → Requires genetic confirmation (WES/targeted panel)"
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Complete Prescribing Safety Card — NDUFV2 CI-Leigh",
            "definition": (
                "ABSOLUTE CONTRAINDICATIONS (never use):\n"
                "  ▪ Valproate / VPA (triple mito-toxicity: CoA/POLG/hepatotoxicity)\n"
                "  ▪ Metformin (direct CI inhibitor at ND1/quinone-binding site; "
                "    NDUFV2/N1b already blocks electrons from reaching N4/N2 → metformin removes residual)\n"
                "  ▪ Linezolid (23S rRNA → blocks all 7 mtDNA ND subunits → CI P-module loss)\n"
                "  ▪ Chloramphenicol (same mt-ribosome mechanism as linezolid)\n"
                "  ▪ Ketogenic Diet (forces NADH overload → N1b-blocked relay worsened)\n"
                "  ▪ Digoxin (positive inotrope → WORSENS LVOT obstruction in HCM ~80%)\n"
                "  ▪ Positive inotropes (dopamine, dobutamine) — same HCM/LVOT mechanism\n\n"
                "HIGH CAUTION / AVOID:\n"
                "  ▪ Propofol (PRIS: CIV inhibition + HCM cardiac vulnerability → haemodynamic collapse)\n"
                "  ▪ Phenobarbital (secondary CI inhibitor; use only if no alternative)\n"
                "  ▪ Fasting (no fasting >4 h; GIR 6-8 during illness/crisis)\n"
                "  ▪ ACE inhibitors / vasodilators in obstructive HCM (reduce preload → worsen LVOT)\n\n"
                "PREFERRED / SAFE:\n"
                "  ▪ Propranolol — HCM first-line (LVOT gradient reduction)\n"
                "  ▪ LEV (levetiracetam) — AED first-line; renal; no mito toxicity\n"
                "  ▪ Clonazepam / CLB (clobazam) — benzodiazepines; no mito toxicity\n"
                "  ▪ Sevoflurane — anaesthetic choice (not propofol)\n"
                "  ▪ Dexmedetomidine — sedation alternative to propofol\n"
                "  ▪ Insulin — glucose management (not metformin)\n"
                "  ▪ Verapamil — HCM second-line (if propranolol insufficient)\n"
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
