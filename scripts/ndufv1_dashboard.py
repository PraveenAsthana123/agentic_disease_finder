#!/usr/bin/env python3
"""NDUFV1 — Leigh Syndrome Isolated Complex I Deficiency (N-Module FMN Core Subunit).

NDUFV1 (NADH:Ubiquinone Oxidoreductase Core Subunit V1), also known as the 51 kDa
flavoprotein subunit (FP1), is the CORE catalytic subunit of the N-module of
Complex I (NADH dehydrogenase).  NDUFV1 is the primary NADH-binding subunit: it binds
FMN (flavin mononucleotide) and iron-sulfur cluster N3 and is the site of NADH electron
donation into the ETC.  Biallelic loss-of-function variants abolish CI NADH oxidation
→ isolated CI deficiency → Leigh / Leigh-like syndrome.

  NDUFV1 gene      OMIM *161015
  Disease          Leigh/Leigh-like Syndrome, nuclear-encoded Complex I deficiency
                   (OMIM #252010 — Mitochondrial Complex I Deficiency, Nuclear Type 4)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       11q13.2

PATHOPHYSIOLOGY (Complex I / N-module / NDUFV1 role):
  NDUFV1 is the CORE subunit of the N-module (N = NADH-binding peripheral arm).
  The N-module is the catalytic heart of Complex I:
    • NDUFV1 (51 kDa, FP1) — binds FMN and iron-sulfur cluster N3
    • NDUFV2 (24 kDa, FP2) — binds iron-sulfur clusters N1a/N1b
    • NDUFS1 (75 kDa, IP1) — binds iron-sulfur clusters N4, N5
    • Together: FMN at NDUFV1 accepts electrons from NADH → passes electrons
      along the iron-sulfur relay (N3 → N1 → N4 → N5 → N6 → N2) to ubiquinone

  NDUFV1 vs NDUFS4 role in N-module:
    • NDUFV1 IS the FMN-binding catalytic subunit — PRIMARY electron acceptor
    • NDUFS4 is an ACCESSORY subunit — stabilises N-module assembly but does not
      bind FMN directly
    → Riboflavin (→ FMN) therapy is EVEN MORE directly relevant in NDUFV1 deficiency
      than in NDUFS4: extra FMN substrate directly reinforces the NDUFV1 active site

  Without NDUFV1:
    – FMN cannot be incorporated into the N-module
    – N-module cannot assemble properly; CI sub-complex accumulates
    – NADH oxidation is abolished → NADH/NAD+ ratio ↑ → lactic acidosis
    – Residual activity in missense alleles (partial FMN binding) → milder phenotype

  Biochemical signature:
    Complex I: 5–20% of control (severely reduced) — ISOLATED CI DEFICIENCY
    Complex II (SDHA): NORMAL (fully nuclear-encoded; NDUFV1 not required)
    Complex III: NORMAL
    Complex IV (COX): NORMAL — KEY DDx from LRPPRC (CI+CIV), SURF1/SCO2 (CIV only)

DISTINGUISHING FEATURES vs NDUFS4:
  Leukodystrophy (white matter T2 hyperintensity): 40–50% — DISTINGUISHING (NDUFV1)
    • NDUFS4 white matter is predominantly spared (gray matter / deep nuclei + brainstem)
    • Dedicated cerebral white matter sequences needed (FLAIR, DWI)
  No olfactory bulb lesions: NDUFV1 does NOT show the olfactory bulb MRI pattern
    characteristic of NDUFS4 (~52–65% in NDUFS4)
  Myoclonus: 30–40% — more prominent than in NDUFS4

FOUNDER / RECURRENT MUTATIONS:
  c.1268C>T (p.Thr423Met)  — recurrent pathogenic missense; partial CI residual; milder
  c.983T>C  (p.Ile328Thr)  — recurrent missense; intermediate severity
  c.1156C>T (p.Arg386Cys)  — consanguineous (Arab/Middle Eastern) families; severe
  Frameshift / nonsense    — complete CI abolition; severe Leigh; early death

THERAPY — RIBOFLAVIN RATIONALE (NDUFV1-SPECIFIC — MOST DIRECT CI):
  NDUFV1 IS the FMN-binding subunit of Complex I.
  Riboflavin → riboflavin-5'-phosphate (FMN) → binds NDUFV1 active site directly.
  In partial-function NDUFV1 missense alleles, exogenous FMN may stabilise NDUFV1
  conformation and restore partial NADH oxidation capacity.
  Level C evidence; typically 100–400 mg/day; biochemical response testing recommended.

SUCCINATE BYPASS RATIONALE:
  Succinate → Complex II (SDHA) → ubiquinol → Complex III → CIV.
  Bypasses the blocked Complex I entirely. Level C; 2–8 g/day.

References:
  Schuelke M et al. Nat Genet. 1999;21(3):260–261. (First NDUFV1 mutations in Leigh syndrome)
  Loeffen J et al. Hum Genet. 1998;103(4):429–434. (Early NDUFV1 characterisation)
  Rötig A, Munnich A. Hum Mutat. 2003;21(6):607–614. (CI deficiency spectrum review)
  Lebon S et al. Brain. 2007;130(Pt 5):1275–1283. (NDUFV1 phenotype series)
  Fassone E, Rahman S. J Med Genet. 2012;49(9):578–590. (CI deficiency genetics review)
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 609
DISEASE_ID   = "ndufv1"
DISEASE_NAME = (
    "NDUFV1 Leigh Syndrome — Isolated Complex I Deficiency "
    "(CI-Leigh / NDUFV1 N-Module FMN Core Subunit)"
)
GENE         = "NDUFV1"
PROTEIN      = (
    "NDUFV1 — 464 aa precursor / ~433 aa mature, N-module CORE subunit (51 kDa FP1), "
    "FMN-binding + iron-sulfur cluster N3 — PRIMARY NADH electron acceptor at CI"
)
OMIM_GENE    = "*161015"
OMIM_DISEASE = "#252010 (Mitochondrial Complex I Deficiency, Nuclear Type 4 / Leigh syndrome)"
CHROMOSOME   = "11q13.2"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = (
    "Infantile (3–18 months); rarely neonatal; delayed onset up to 2–3yr in missense alleles"
)
COHORT_SIZE  = 40
COLOR        = "#1a237e"   # deep indigo — CI N-module (FMN catalytic core, primary NADH site)
LIGHT        = "#e8eaf6"

# Genotype pool
GENO_TMET  = "c.1268C>T (p.Thr423Met) / truncating (compound het) — recurrent missense; partial residual CI"
GENO_ITHR  = "c.983T>C (p.Ile328Thr) / truncating (compound het) — recurrent missense; intermediate severity"
GENO_RCYS  = "c.1156C>T (p.Arg386Cys) homozygous — consanguineous (Arab/Middle Eastern); severe"
GENO_NULL  = "Frameshift / nonsense (compound het) — complete CI abolition; severe Leigh; early death"
GENO_MMIS  = "Missense / missense (compound het) — partial FMN binding residual; milder; longer survival"

GENO_POOL    = [GENO_TMET, GENO_ITHR, GENO_RCYS, GENO_NULL, GENO_MMIS]
GENO_WEIGHTS = [0.30,      0.25,       0.18,       0.15,      0.12]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG — reproducible 40-patient NDUFV1/CI-Leigh cohort (seed-609)."""
    return random.Random(SEED)


# ── Patient cohort ────────────────────────────────────────────────────────────
_TX_POOL = [
    "IV Dextrose GIR 6-8 (metabolic crisis — first-line)",
    "Riboflavin B2 (100–400 mg/day — MOST DIRECT FMN precursor, NDUFV1 active site)",
    "CoQ10/Ubiquinol (300–600 mg/day)",
    "Thiamine B1 (mandatory empiric, all Leigh until SLC19A3/BTD excluded)",
    "Biotin (mandatory empiric, all Leigh until BTD ruled out)",
    "Succinate (IV/oral — Complex I bypass via CII — CI-SPECIFIC)",
    "NaHCO3 (lactic acidosis correction, pH <7.20)",
    "LEV (seizures — preferred AED, renal excretion, no mito toxicity)",
    "Clobazam / CLB (benzodiazepine — no mito toxicity; myoclonus adjunct)",
    "NIV/BiPAP (central respiratory compromise from brainstem Leigh lesions)",
    "Carnitine (secondary deficiency, 50–100 mg/kg/day)",
    "Enteral feeds / NG (avoid fasting; high-carbohydrate; KD strictly CI)",
]

_OUTCOMES = [
    "Alive — stable moderate-to-severe Leigh; ongoing cofactor + respiratory support",
    "Alive — prolonged survival with missense alleles; partial FMN residual; school-age",
    "Alive — severe white matter disease + developmental impairment; ongoing support",
    "Died — acute lactic crisis + respiratory failure in first 18 months (null alleles)",
    "Died — progressive brainstem + white matter failure; 2–5yr trajectory",
]
_OUT_WEIGHTS = [0.25, 0.17, 0.20, 0.20, 0.18]


def _generate_cohort(rng: random.Random) -> list[dict[str, Any]]:
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno        = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex         = rng.choice(["M", "F"])
        onset_mo    = rng.choices(
            [3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 24, 30],
            weights=[3, 5, 8, 12, 12, 10, 10, 10, 8, 8, 8, 6],
        )[0]
        onset_yr    = round(onset_mo / 12, 1)
        lactate_mm  = round(rng.uniform(3.2, 15.0), 1)   # mmol/L
        ci_pct      = rng.randint(5, 22)                  # Complex I (% of control)
        cii_pct     = rng.randint(85, 118)                # Complex II — NORMAL
        civ_pct     = rng.randint(80, 112)                # Complex IV — NORMAL

        has_leigh_mri       = rng.random() < 0.88   # bilateral putamen/brainstem T2 hyperintensities
        has_leukodystrophy  = rng.random() < 0.45   # white matter — DISTINGUISHING vs NDUFS4
        has_regression      = rng.random() < 0.97   # universal in Leigh
        has_hypotonia       = rng.random() < 0.88
        has_lactic          = rng.random() < 0.92
        has_resp            = rng.random() < 0.63
        has_seizures        = rng.random() < 0.65
        has_myoclonus       = rng.random() < 0.38   # more common than in NDUFS4
        has_ataxia          = rng.random() < 0.48
        has_optic           = rng.random() < 0.40
        has_nystagmus       = rng.random() < 0.38
        has_dystonia        = rng.random() < 0.50
        has_spasticity      = rng.random() < 0.42
        has_hcm             = rng.random() < 0.06   # RARE — KEY DDx SCO2 (100%)
        has_hepatopathy     = rng.random() < 0.06   # RARE — KEY DDx POLG/DGUOK
        has_iron            = False                  # NO iron overload — KEY DDx GRACILE
        has_tubulopathy     = rng.random() < 0.06   # RARE — KEY DDx COX10 (65%)
        has_olfactory       = False                  # NO olfactory bulb lesions — KEY DDx NDUFS4

        feat_list = [
            "Isolated Complex I deficiency (CII, CIII, CIV — NORMAL)",
            "Leigh / Leigh-like MRI (bilateral putamen + brainstem)",
        ]
        if has_regression:     feat_list.append("Psychomotor regression/arrest")
        if has_leukodystrophy: feat_list.append("Leukodystrophy / white matter (DISTINGUISHING — not NDUFS4)")
        if has_hypotonia:      feat_list.append("Hypotonia")
        if has_lactic:         feat_list.append("Lactic acidosis")
        if has_resp:           feat_list.append("Respiratory compromise / central apnoea")
        if has_seizures:       feat_list.append("Seizures")
        if has_myoclonus:      feat_list.append("Myoclonus (more prominent than in NDUFS4)")
        if has_ataxia:         feat_list.append("Ataxia")
        if has_dystonia:       feat_list.append("Dystonia")
        if has_nystagmus:      feat_list.append("Nystagmus")
        if has_optic:          feat_list.append("Optic atrophy")
        if has_spasticity:     feat_list.append("Spasticity")
        if has_hcm:            feat_list.append("HCM (RARE — KEY DDx SCO2 100%)")
        if has_hepatopathy:    feat_list.append("Hepatopathy (RARE — KEY DDx POLG/DGUOK)")

        txs     = rng.sample(_TX_POOL, k=rng.randint(4, 7))
        outcome = rng.choices(_OUTCOMES, weights=_OUT_WEIGHTS)[0]

        patients.append({
            "id":                  f"NDUFV1-{i:03d}",
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
        "Psychomotor Regression / Arrest (universal in Leigh)":          _pct("has_regression"),
        "Leigh / Leigh-like MRI (bilateral putamen + brainstem)":        _pct("has_leigh_mri"),
        "Leukodystrophy / White Matter T2 Signal (DISTINGUISHING)":      _pct("has_leukodystrophy"),
        "Isolated Complex I Deficiency (CII, CIII, CIV Normal — 100%)": 100,
        "Hypotonia":                                                     _pct("has_hypotonia"),
        "Lactic Acidosis (elevated baseline + crisis)":                  _pct("has_lactic"),
        "Respiratory Compromise / Central Apnoea":                       _pct("has_resp"),
        "Seizures (multifocal, myoclonic common)":                       _pct("has_seizures"),
        "Myoclonus (more prominent than NDUFS4)":                        _pct("has_myoclonus"),
        "Dystonia":                                                      _pct("has_dystonia"),
        "Ataxia":                                                        _pct("has_ataxia"),
        "Nystagmus":                                                     _pct("has_nystagmus"),
        "Optic Atrophy":                                                 _pct("has_optic"),
        "Spasticity":                                                    _pct("has_spasticity"),
        "HCM (RARE — KEY DDx SCO2 100%)":                               _pct("has_hcm"),
        "Hepatopathy (RARE — KEY DDx POLG / DGUOK)":                    _pct("has_hepatopathy"),
        "NO Olfactory Bulb Lesions (KEY DDx NDUFS4 ~52–65%)":           100,
        "NO HCM Dominant (KEY DDx SCO2 100% / COX15 78%)":              100,
        "NO Iron Overload (KEY DDx GRACILE — 100% iron overload)":      100,
        "NO COX Deficiency (KEY DDx SURF1 / SCO2 / COX10 / COX15)":    100,
        "Alive (with support)":                                          round(alive / COHORT_SIZE * 100),
    }

    kpis = [
        {"label": "Cohort (n)",         "value": COHORT_SIZE,                                                           "color": COLOR},
        {"label": "Leigh MRI",          "value": f"{feature_frequencies['Leigh / Leigh-like MRI (bilateral putamen + brainstem)']}%", "color": "#6a1b9a"},
        {"label": "Leukodystrophy",     "value": f"{feature_frequencies['Leukodystrophy / White Matter T2 Signal (DISTINGUISHING)']}%", "color": COLOR},
        {"label": "Resp Compromise",    "value": f"{feature_frequencies['Respiratory Compromise / Central Apnoea']}%",  "color": "#e65100"},
        {"label": "Hypotonia",          "value": f"{feature_frequencies['Hypotonia']}%",                                "color": COLOR},
        {"label": "CI Activity",        "value": "5–20% control",                                                       "color": "#c62828"},
        {"label": "Fatal",              "value": f"{round(died/COHORT_SIZE*100)}%",                                     "color": "#b71c1c"},
        {"label": "Seed",               "value": f"#{SEED}",                                                            "color": "#455a64"},
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "severity": "ABSOLUTE CI — DANGEROUS IN ALL LEIGH / CI DEFICIENCY",
            "mechanism": (
                "Triple mechanism in NDUFV1/CI-Leigh:\n"
                "1. CoA SEQUESTRATION: valproyl-CoA depletes the mitochondrial CoA pool "
                "   → TCA cycle collapses → OXPHOS substrate loss. In NDUFV1 deficiency "
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
            "severity": "ABSOLUTE CI — DIRECT COMPLEX I INHIBITOR: CATASTROPHIC IN NDUFV1",
            "mechanism": (
                "Metformin inhibits Complex I (NADH dehydrogenase) at the ND1/NDUFS2 quinone "
                "binding site → blocks NADH → ubiquinone electron transfer.\n"
                "In NDUFV1 deficiency: CI is the primary disease locus (5–20% activity). "
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
                "In NDUFV1/CI-Leigh:\n"
                "  • NDUFV1 itself is nuclear-encoded — BUT Complex I also requires 7 mtDNA-encoded "
                "    ND subunits (MT-ND1–6, MT-ND4L) for the P-module membrane arm\n"
                "  • Linezolid blocks synthesis of ALL 7 → CI P-module cannot assemble "
                "    → N-module (NDUFV1) has no membrane anchor → CI activity → zero\n"
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
                "In NDUFV1 CI-Leigh: CI cannot re-oxidise NADH (5–20% activity). "
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
                "In NDUFV1/CI-Leigh: CI is the primary ETC bottleneck. Propofol's CIV "
                "inhibition creates a SECOND downstream ETC block → electrons trapped "
                "between CI (N-module/NDUFV1) and CIV → ROS burst → oxidative crisis.\n"
                "Use sevoflurane (volatile, no PRIS) or dexmedetomidine as alternatives."
            ),
        },
        {
            "drug": "Phenobarbital",
            "severity": "HIGH CAUTION — Secondary Complex I Inhibitor",
            "mechanism": (
                "Phenobarbital inhibits CI electron transport (quinone channel region). "
                "In NDUFV1/CI-Leigh: residual CI (5–20%) is critical. Phenobarbital's "
                "CI inhibition reduces this residual further → lactic decompensation risk.\n"
                "Use LEV (preferred) or CLB (benzodiazepine — no mito toxicity) instead. "
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
                "CI assembly collapses → catastrophic in NDUFV1/CI deficiency."
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
        "Psychomotor Regression":                                   _pct("has_regression"),
        "Leigh / Leigh-like MRI":                                   _pct("has_leigh_mri"),
        "Leukodystrophy / White Matter (DISTINGUISHING)":           _pct("has_leukodystrophy"),
        "Hypotonia":                                                _pct("has_hypotonia"),
        "Lactic Acidosis":                                          _pct("has_lactic"),
        "Respiratory Compromise":                                   _pct("has_resp"),
        "Seizures":                                                 _pct("has_seizures"),
        "Myoclonus (more common than NDUFS4)":                      _pct("has_myoclonus"),
        "Dystonia":                                                 _pct("has_dystonia"),
        "Ataxia":                                                   _pct("has_ataxia"),
        "Nystagmus":                                                _pct("has_nystagmus"),
        "Optic Atrophy":                                            _pct("has_optic"),
        "Spasticity":                                               _pct("has_spasticity"),
        "HCM (RARE)":                                              _pct("has_hcm"),
        "Hepatopathy (RARE)":                                      _pct("has_hepatopathy"),
        "Olfactory Bulb MRI (NEVER — KEY DDx NDUFS4)":             0,
    }

    genotype_distribution = {}
    for p in patients:
        g = p["geno"].split(" — ")[0]
        genotype_distribution[g] = genotype_distribution.get(g, 0) + 1

    ci_values  = [p["ci_pct"]  for p in patients]
    cii_values = [p["cii_pct"] for p in patients]
    civ_values = [p["civ_pct"] for p in patients]

    return {
        "patients":              patients,
        "feature_frequencies":   feature_frequencies,
        "genotype_distribution": genotype_distribution,
        "biochemistry_summary": {
            "complex_I_mean_pct":   round(sum(ci_values)  / len(ci_values),  1),
            "complex_II_mean_pct":  round(sum(cii_values) / len(cii_values), 1),
            "complex_IV_mean_pct":  round(sum(civ_values) / len(civ_values), 1),
            "note": (
                "CII and CIV NORMAL — ISOLATED CI deficiency is the biochemical fingerprint. "
                "NDUFV1 (FMN core) vs NDUFS4 (accessory): both give identical biochemistry "
                "but NDUFV1 is the primary NADH-binding catalytic subunit (riboflavin MORE direct)."
            ),
        },
        "onset_summary": {
            "median_onset_mo": 9,
            "range_mo": "3–30 months (missense alleles up to 2–3 yr)",
            "note": (
                "Slightly later onset range than NDUFS4 due to missense allele proportion; "
                "null alleles present ≤6 months."
            ),
        },
    }


# ── Definitions ───────────────────────────────────────────────────────────────
def get_definitions() -> dict[str, Any]:
    pharmacology = [
        {
            "term": "NDUFV1 — FMN-Binding Core Subunit of Complex I N-Module",
            "definition": (
                "NDUFV1 (NADH:Ubiquinone Oxidoreductase Core Subunit V1, 51 kDa, FP1) is the "
                "PRIMARY NADH-oxidising subunit of Complex I (NADH dehydrogenase).\n\n"
                "Structure:\n"
                "  • 464 aa precursor; ~433 aa mature after MTS cleavage; 11q13.2\n"
                "  • Localised in the N-module (NADH-binding peripheral arm) of CI\n"
                "  • Contains FMN (Flavin MonoNucleotide) — the primary electron acceptor site\n"
                "  • Contains iron-sulfur cluster N3 — part of the N-module electron relay\n\n"
                "Function:\n"
                "  1. NADH binds the NDUFV1 active site → transfers 2 electrons to FMN\n"
                "  2. FMN (NDUFV1) → Fe-S cluster N3 → N1a/N1b (NDUFV2) → N4/N5 (NDUFS1)\n"
                "  3. N2 (NDUFS7) → ubiquinone → Complex III → Complex IV → O2\n\n"
                "Disease consequence:\n"
                "  Loss of NDUFV1 function → FMN cannot be incorporated into N-module → "
                "N-module cannot assemble → CI stalls as sub-complex → NADH oxidation "
                "abolished → NADH/NAD+ ↑ → lactic acidosis → Leigh syndrome.\n\n"
                "NDUFV1 vs NDUFS4 distinction:\n"
                "  NDUFV1 IS the FMN-binding catalytic subunit (core). "
                "NDUFS4 is an ACCESSORY subunit that stabilises N-module assembly but "
                "does NOT bind FMN directly. Both cause isolated CI Leigh, but riboflavin "
                "supplementation is MOST DIRECTLY relevant in NDUFV1 (FMN is at NDUFV1 itself)."
            ),
        },
        {
            "term": "Riboflavin (Vitamin B2) → FMN — NDUFV1-Specific Mechanism",
            "definition": (
                "The therapeutic rationale for riboflavin in NDUFV1 deficiency:\n\n"
                "Riboflavin (vitamin B2)\n"
                "  → Riboflavin kinase → Riboflavin-5'-phosphate (FMN)\n"
                "  → FMN adenylyltransferase → FAD (flavin adenine dinucleotide)\n\n"
                "FMN is the DIRECT prosthetic group at the NDUFV1 active site:\n"
                "  • Extra dietary/supplemental riboflavin → extra cellular FMN\n"
                "  • In partial-function NDUFV1 missense variants (e.g., p.Thr423Met, "
                "    p.Ile328Thr): residual NDUFV1 protein with partial FMN-binding capacity "
                "    may be stabilised by elevated FMN concentration\n"
                "  • Mechanism: mass-action stabilisation of the FMN–NDUFV1 complex\n\n"
                "Clinical use: 100–400 mg/day oral (2-week biochemical response test: "
                "measure CI activity in fibroblasts/muscle before vs. after riboflavin loading).\n"
                "Level C evidence; more directly relevant in NDUFV1 than in any other "
                "CI subunit deficiency, since FMN is at NDUFV1 itself (not distal).\n\n"
                "Distinction: NDUFS4 (accessory) does not bind FMN — riboflavin still "
                "beneficial (FMN needed for N-module function) but less mechanistically direct."
            ),
        },
        {
            "term": "Isolated Complex I Deficiency — Biochemical Fingerprint",
            "definition": (
                "NDUFV1 mutations produce ISOLATED Complex I deficiency:\n"
                "  CI: 5–20% of control activity (severely reduced)\n"
                "  CII (SDHA): NORMAL (independent nuclear-encoded enzyme)\n"
                "  CIII: NORMAL (UQCR subunits — not related to CI N-module)\n"
                "  CIV (COX): NORMAL — KEY DDx point:\n"
                "    • SURF1/SCO2/SCO1/COX10/COX15/TACO1 → ISOLATED CIV deficiency\n"
                "    • LRPPRC → COMBINED CI+CIV deficiency\n"
                "    • NDUFV1/NDUFS4 → ISOLATED CI — CIV NORMAL\n\n"
                "Citrate synthase: often elevated (mitochondrial proliferation response)\n\n"
                "Tissue: fibroblast CI activity is diagnostic in most NDUFV1 cases; "
                "muscle biopsy for histology (ragged-red fibres occasionally present)."
            ),
        },
        {
            "term": "Succinate Bypass in CI Deficiency",
            "definition": (
                "Succinate → Complex II (SDHA, succinate dehydrogenase) → ubiquinol "
                "→ Complex III → Complex IV → O2.\n\n"
                "In CI deficiency (NDUFV1): succinate provides an alternative electron "
                "entry point that BYPASSES the blocked CI N-module entirely.\n"
                "CI is not used; the electron relay proceeds via CII → CIII → CIV.\n\n"
                "This CI bypass:\n"
                "  • Maintains some NAD+ regeneration via CII-linked TCA cycle steps\n"
                "  • Partially restores ETC flux and ATP generation\n"
                "  • Reduces NADH/NAD+ ratio → less lactate accumulation\n\n"
                "Dose: 2–8 g/day orally in divided doses; IV at metabolic centres.\n"
                "Level C; NOT useful in isolated COX diseases (SURF1/SCO2) since "
                "CIV remains the bottleneck there."
            ),
        },
    ]

    gene_concepts = [
        {
            "term": "NDUFV1 Gene — Chromosome 11q13.2",
            "definition": (
                "NDUFV1 (OMIM *161015) maps to chromosome 11q13.2.\n"
                "Inheritance: AR (autosomal recessive); biallelic loss-of-function.\n"
                "Allele classes:\n"
                "  • Missense (e.g., p.Thr423Met, p.Ile328Thr, p.Arg386Cys): partial residual "
                "    FMN-binding function → milder phenotype, later onset, longer survival\n"
                "  • Frameshift / nonsense: complete loss of NDUFV1 → severe Leigh, "
                "    early infantile onset, rapidly fatal without intensive support\n"
                "  • Compound heterozygotes: most patients (one missense + one truncating "
                "    allele is most common; determines phenotype severity)\n\n"
                "Prenatal diagnosis: gene sequencing of chorionic villous / amniocentesis sample.\n"
                "Newborn screening: not currently standard; consider in at-risk families.\n"
                "WES/WGS: recommended first-line in any Leigh/CI deficiency workup."
            ),
        },
        {
            "term": "NDUFV1 vs NDUFS4 — Genotype-Phenotype Comparison",
            "definition": (
                "Both cause isolated Complex I Leigh syndrome (AR); key differences:\n\n"
                "NDUFV1 (FMN core subunit, 51 kDa):\n"
                "  • Gene: NDUFV1, 11q13.2, OMIM *161015\n"
                "  • Role: FMN-binding catalytic core; DIRECT NADH oxidation site\n"
                "  • Leukodystrophy: 40–50% (WHITE MATTER involvement)\n"
                "  • Olfactory bulb MRI: ABSENT (0%) — critical DDx from NDUFS4\n"
                "  • Myoclonus: 30–40% (more common)\n"
                "  • Riboflavin: FMN directly at NDUFV1 → MOST DIRECT mechanism\n\n"
                "NDUFS4 (N-module accessory subunit, 175 aa):\n"
                "  • Gene: NDUFS4, 5q11.2-q13.3, OMIM *602694\n"
                "  • Role: N-module assembly stabilisation (accessory, not FMN-binding)\n"
                "  • Leukodystrophy: RARE (predominantly gray matter / deep nuclei)\n"
                "  • Olfactory bulb MRI: 52–65% — DISTINGUISHING FEATURE\n"
                "  • Myoclonus: less prominent\n"
                "  • Riboflavin: relevant (FMN needed) but less mechanistically direct\n\n"
                "Biochemistry: IDENTICAL (both isolated CI deficiency, CII/CIII/CIV normal). "
                "MRI pattern (leukodystrophy vs olfactory bulb) is the key distinguisher "
                "before molecular results are available."
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "NDUFV1 Leigh Syndrome — DDx Table",
            "definition": (
                "NDUFV1 differential diagnosis in the CI-Leigh context:\n\n"
                "vs NDUFS4 (CI-Leigh, 5q):\n"
                "  • NDUFV1: leukodystrophy 40–50%; NO olfactory bulb lesions\n"
                "  • NDUFS4: olfactory bulb lesions 52–65%; no leukodystrophy\n"
                "  • Biochemistry identical → MRI distinguishes before genetics\n\n"
                "vs SURF1/SCO2/COX10/COX15/TACO1 (CIV-Leigh):\n"
                "  • All have ISOLATED CIV deficiency; CI NORMAL\n"
                "  • NDUFV1 has ISOLATED CI; CIV NORMAL → biochemistry distinguishes\n\n"
                "vs LRPPRC (CI+CIV Leigh):\n"
                "  • LRPPRC: COMBINED CI+CIV deficiency 100%; episodic crises; French-Canadian\n"
                "  • NDUFV1: isolated CI only; CIV NORMAL\n\n"
                "vs POLG-Leigh:\n"
                "  • POLG: hepatopathy common (80%); EPC (epilepsia partialis continua); "
                "    mtDNA depletion on muscle/liver biopsy\n"
                "  • NDUFV1: hepatopathy RARE (<7%); no EPC; mtDNA copy number NORMAL\n\n"
                "vs GRACILE:\n"
                "  • GRACILE: iron overload (ferritin >2000), IUGR, cholestasis\n"
                "  • NDUFV1: NO iron overload\n\n"
                "vs SLC19A3 (biotin-thiamine-responsive basal ganglia disease):\n"
                "  • TREATABLE MIMIC: fever-triggered crises, bilateral striatal lesions\n"
                "  • MANDATORY: biotin + thiamine empirically in ALL Leigh presentations "
                "    before molecular diagnosis — SLC19A3 / BTD must be excluded first"
            ),
        },
        {
            "term": "Leukodystrophy in NDUFV1 — Mechanism and Significance",
            "definition": (
                "White matter disease (leukodystrophy) occurs in ~40–50% of NDUFV1 cases "
                "and is the key MRI feature distinguishing NDUFV1 from NDUFS4.\n\n"
                "Mechanism:\n"
                "  • Cerebral white matter (myelin) has high energy demand for maintenance "
                "    of the myelin sheath, axonal ion pumps, and oligodendrocyte survival\n"
                "  • Oligodendrocytes are particularly vulnerable to OXPHOS failure because "
                "    they have high mitochondrial density and limited glycolytic reserve\n"
                "  • In NDUFV1/CI deficiency: oligodendrocyte OXPHOS failure → myelin "
                "    maintenance failure → axonal demyelination → T2 white matter signal\n"
                "  • Why NDUFV1 > NDUFS4 for white matter: possibly related to the degree "
                "    of NADH/NAD+ redox imbalance (NDUFV1 = catalytic core vs NDUFS4 = "
                "    accessory stabiliser — residual activity distribution may differ)\n\n"
                "MRI features:\n"
                "  • Bilateral, symmetric periventricular and/or deep white matter T2/FLAIR "
                "    hyperintensity on MRI\n"
                "  • Often accompanies standard Leigh bilateral putamen/brainstem lesions\n"
                "  • DWI: may show restricted diffusion in acute demyelination phases\n\n"
                "Clinical significance:\n"
                "  • Leukodystrophy can cause spasticity, visual impairment, cognitive decline "
                "    independent of the Leigh gray-matter component\n"
                "  • Dedicated white matter MRI sequences required (FLAIR + DWI)"
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "NDUFV1 Prescribing Safety Matrix",
            "definition": (
                "ABSOLUTE CONTRAINDICATIONS (never use):\n"
                "  • Valproate (VPA): triple mechanism — CoA sequestration, POLG inhibition, hepatotoxicity\n"
                "  • Metformin: direct CI inhibitor at disease locus — catastrophic lactic acidosis\n"
                "  • Linezolid: 23S rRNA block → eliminates all 7 mtDNA-encoded ND subunits\n"
                "  • Chloramphenicol: same mt-ribosome mechanism as linezolid\n"
                "  • Ketogenic diet: forces NADH generation → worsens CI block → lactic crisis\n\n"
                "AVOID / HIGH CAUTION:\n"
                "  • Propofol: PRIS (CIV inhibition) + beta-oxidation uncoupling → second ETC block\n"
                "  • Phenobarbital: secondary CI inhibitor (quinone channel); lowest dose if unavoidable\n"
                "  • Fasting: NEVER fast a CI-Leigh patient (fasting → fat mobilisation → NADH ↑ at CI)\n\n"
                "PREFERRED ALTERNATIVES:\n"
                "  • AED: LEV (levetiracetam) — renal excretion, no mito toxicity, IV formulation\n"
                "  • AED: CLB (clobazam) — benzodiazepine, no mito toxicity; good for myoclonus\n"
                "  • Anaesthesia: sevoflurane (volatile; no PRIS) + dexmedetomidine\n"
                "  • Glucose: IV dextrose GIR 6-8 (crisis) or continuous enteral high-carb\n\n"
                "MANDATORY EMPIRICS (all Leigh before molecular dx):\n"
                "  • Thiamine B1 (100-300 mg/day) — SLC19A3 / BRBGD and PDH treatable mimics\n"
                "  • Biotin (5-20 mg/day) — BTD (biotinidase deficiency) treatable mimic\n"
                "  • Riboflavin B2 (100-400 mg/day) — FMN precursor; MOST DIRECTLY relevant in NDUFV1"
            ),
        },
        {
            "term": "Acute Crisis Protocol — NDUFV1/CI-Leigh Metabolic Emergency",
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
                "  Riboflavin: direct FMN precursor at NDUFV1 active site (most direct CI therapy).\n"
                "  Thiamine: MANDATORY in any acute encephalopathy (SLC19A3 mimic).\n\n"
                "STEP 5 — IV SUCCINATE (metabolic centre, if available):\n"
                "  0.5–1 g/kg/day → CII-mediated bypass of blocked CI.\n\n"
                "STEP 6 — SEIZURES → LEV IV:\n"
                "  20–40 mg/kg loading; simultaneous IV dextrose; ABSOLUTE CI VPA.\n"
                "  Myoclonus: add CLB (clobazam) or clonazepam (benzodiazepine — no mito toxicity).\n\n"
                "STEP 7 — RESPIRATORY → NIV/BiPAP:\n"
                "  SpO2 <92% or RR >40 → BiPAP; intubation: sevoflurane, NOT propofol.\n\n"
                "EMERGENCY CARD:\n"
                "  Carried at all times. ER: IV dextrose immediately; NEVER VPA, metformin, "
                "linezolid, chloramphenicol, propofol, or ketogenic diet. "
                "Contact metabolic neurology (phone on card)."
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
