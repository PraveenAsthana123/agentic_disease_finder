#!/usr/bin/env python3
"""NDUFS4 — Leigh Syndrome Complex I Deficiency (Nuclear-Encoded CI Subunit S4).

NDUFS4 (NADH:Ubiquinone Oxidoreductase Subunit S4, also known as the AQDQ subunit,
175 aa after MTS cleavage) is a nuclear-encoded accessory subunit of NADH dehydrogenase
(Complex I, NDU N-module).  NDUFS4 is required for the final assembly of the N-module
peripheral arm of Complex I and for stabilisation of the fully assembled 45-subunit CI
holocomplex.  Biallelic loss-of-function mutations abolish CI activity → Leigh syndrome
(most severe end of the mtDNA-encoded / nuclear-encoded CI deficiency spectrum).

  NDUFS4 gene      OMIM *602694
  Disease          Leigh Syndrome Complex I Deficiency (OMIM #256000; CI-Leigh)
  Inheritance      AR (autosomal recessive biallelic), predominantly null alleles
  Chromosome       5q11.2-q13.3

FOUNDER MUTATIONS:
  c.462delA (p.Lys154Asnfs*16) — Dutch/North European founder; most common CI-Leigh allele
  in European populations.  Carrier frequency in Netherlands ~1:500.
  c.316C>T (p.Arg106*)         — recurrent nonsense; multiple ethnicities
  c.44insA  (p.Thr15Asnfs*17) — infantile-onset frameshift; severe phenotype
  Exon 1–2 deletions           — large deletion alleles; severe; seen in consanguineous families

PATHOPHYSIOLOGY (Complex I / N-module / NDUFS4 role):
  NDUFS4 is part of the N-module (N = NADH-binding peripheral arm):
    • N-module subunits: NDUFS4, NDUFS6, NDUFA12, NDUFB9, NDUFA2, NDUFA6, NDUFS2, NDUFS3
    • N-module contains: FMN (flavin mononucleotide, primary NADH acceptor) and iron-sulfur
      clusters N1a–N2 that form the electron relay from NADH to ubiquinone
    • NDUFS4 stabilises the assembled N-module; without NDUFS4:
        – N-module fails to associate with Q-module and P-module
        – CI stalls as a late-stage assembly intermediate (~830 kDa sub-complex)
        – No functional NADH → ubiquinone electron transfer
        – NADH accumulates → pyruvate → lactate → lactic acidosis

  Biochemical signature:
    Complex I: 5–20% of control (severely reduced) — ISOLATED CI DEFICIENCY
    Complex II (SDHA): NORMAL (fully nuclear-encoded, NDUFS4 not required)
    Complex III: NORMAL (mt-encoded + nuclear assembly, but NDUFS4 not involved)
    Complex IV (COX): NORMAL — KEY DDx from LRPPRC (combined CI+CIV) and SURF1 (CIV only)
    Citrate synthase: NORMAL (often elevated — mitochondrial proliferation marker)

DISTINGUISHING MRI FEATURE:
  Olfactory bulb + olfactory cortex lesions: reported in ~60–70% of NDUFS4 cases.
  First described systematically in NDUFS4 knockout mice (Kruse SE 2008 Cell Metab);
  confirmed in human cohorts (Budde 2000; multiple subsequent series).  This lesion pattern
  is NOT seen in SURF1, SCO2, COX10, COX15, or LRPPRC disease, making it a radiological
  distinguisher for CI-Leigh from COX-Leigh.  Standard Leigh MRI (bilateral putamen/brainstem)
  also present in 90-95%.

THERAPY — RIBOFLAVIN RATIONALE (CI-SPECIFIC):
  Riboflavin → FMN (Flavin MonoNucleotide) → primary electron acceptor at Complex I.
  In residual-function missense NDUFS4 alleles, extra FMN substrate may stabilise partial
  CI assembly or enhance residual N-module function.  Level C evidence; typically 100–400 mg/day.
  IMPORTANT: riboflavin is more directly relevant in CI deficiency than in isolated COX disease
  (where FMN is less critical for CIV activity).

SUCCINATE BYPASS RATIONALE:
  Succinate → Complex II (SDHA, succinate dehydrogenase) → ubiquinol → Complex III → CIV.
  Succinate bypasses the blocked Complex I entirely, providing alternative electrons to the
  ubiquinone pool.  Level C; IV or oral (typically 2–8 g/day in divided doses).

References:
  Budde SM et al. Am J Hum Genet. 2000;67(4):1063–1069. (First NDUFS4 mutations, c.462delA)
  Kruse SE et al. Cell Metab. 2008;7(4):312–320.     (NDUFS4 knockout mouse; olfactory lesions)
  Anderson SL et al. J Inherit Metab Dis. 2008.       (NDUFS4 natural history review)
  Finsterer J. J Inherit Metab Dis. 2008.            (CI-Leigh treatment overview)
  Blokpoel RG et al. Eur J Paediatric Neurol. 2018.  (Clinical spectrum, c.462delA cohort)
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 607
DISEASE_ID   = "ndufs4"
DISEASE_NAME = "NDUFS4 Leigh Syndrome — Isolated Complex I Deficiency (CI-Leigh / NDUFS4 N-Module)"
GENE         = "NDUFS4"
PROTEIN      = "NDUFS4 — 175 aa, N-module accessory subunit, CI (NDU), stabilises N-module peripheral arm"
OMIM_GENE    = "*602694"
OMIM_DISEASE = "#256000 (Leigh Syndrome Complex I Deficiency — NDUFS4)"
CHROMOSOME   = "5q11.2-q13.3"
INHERITANCE  = "AR (autosomal recessive biallelic); predominantly null/frameshift alleles"
ONSET        = "Infantile (3–15 months; median ~6–9 months); neonatal in severe null alleles"
COHORT_SIZE  = 40
COLOR        = "#1b5e20"   # deep forest green — Complex I (entry point of ETC, electron energy)
LIGHT        = "#e8f5e9"

# Genotype pool
GENO_c462delA_HOM  = "c.462delA homozygous (p.Lys154Asnfs*16) — Dutch/N.European founder; most common CI-Leigh allele"
GENO_c462delA_NULL = "c.462delA / truncating (compound het) — European; severe; no residual NDUFS4"
GENO_c316CT        = "c.316C>T (p.Arg106*) / frameshift (compound het) — recurrent nonsense; severe"
GENO_MISS_MISS     = "Missense / missense (compound het) — partial residual CI; milder; longer survival"
GENO_DEL_MISS      = "Large exon deletion / missense (compound het) — consanguineous; severe-moderate"

GENO_POOL    = [GENO_c462delA_HOM, GENO_c462delA_NULL, GENO_c316CT, GENO_MISS_MISS, GENO_DEL_MISS]
GENO_WEIGHTS = [0.35,               0.25,                0.20,         0.12,            0.08]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient NDUFS4/CI-Leigh cohort (seed-607)."""
    return random.Random(SEED)


# ── Patient cohort ────────────────────────────────────────────────────────────
_TX_POOL = [
    "IV Dextrose GIR 6-8 (metabolic crisis)",
    "CoQ10/Ubiquinol (300-600 mg/day)",
    "Riboflavin B2 (100-400 mg/day — CI-specific FMN precursor)",
    "Thiamine B1 (empiric, mandatory until molecular dx)",
    "Biotin (empiric, mandatory until BTD ruled out)",
    "Succinate (IV/oral — Complex I bypass via CII)",
    "NaHCO3 (lactic acidosis correction)",
    "LEV (seizures — preferred AED)",
    "NIV/BiPAP (central respiratory compromise)",
    "Enteral feeds / NG (avoid fasting; high-carbohydrate diet)",
    "Carnitine (secondary deficiency)",
    "Oxygen supplementation (acute lactic crisis)",
]

_OUTCOMES = [
    "Alive — stable, moderate Leigh phenotype; ongoing cofactor therapy (infantile-onset null)",
    "Alive — prolonged survival, partial residual CI (missense alleles), teenager",
    "Alive — severe developmental impairment, recurrent crises, continued intensive support",
    "Died — acute lactic crisis + respiratory failure in first year (severe null alleles)",
    "Died — progressive brainstem failure + apnoea (classic Leigh trajectory, 2–4yr)",
]
_OUT_WEIGHTS = [0.28, 0.15, 0.22, 0.18, 0.17]


def _generate_cohort(rng: random.Random) -> list[dict[str, Any]]:
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno        = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex         = rng.choice(["M", "F"])
        onset_mo    = rng.choices([3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18],
                                   weights=[4, 6, 8, 12, 12, 10, 10, 8, 8, 6, 6])[0]
        onset_yr    = round(onset_mo / 12, 1)
        lactate_mm  = round(rng.uniform(3.5, 14.0), 1)   # mmol/L (elevated; crisis >10)
        ci_pct      = rng.randint(5, 22)                  # Complex I (% of control)
        cii_pct     = rng.randint(85, 115)                # Complex II (NORMAL)
        civ_pct     = rng.randint(80, 110)                # Complex IV (NORMAL — KEY DDx SURF1/LRPPRC)

        has_leigh_mri    = rng.random() < 0.93   # bilateral putamen + brainstem T2 hyperintensities
        has_olfactory    = rng.random() < 0.65   # olfactory bulb lesions — DISTINGUISHING feature
        has_regression   = rng.random() < 0.98   # universal in Leigh
        has_hypotonia    = rng.random() < 0.92
        has_lactic       = rng.random() < 0.92   # always high during crisis; baseline often elevated
        has_resp         = rng.random() < 0.68   # central apnoea from brainstem lesions
        has_seizures     = rng.random() < 0.58
        has_ataxia       = rng.random() < 0.50
        has_optic        = rng.random() < 0.38   # optic atrophy
        has_nystagmus    = rng.random() < 0.42
        has_dystonia     = rng.random() < 0.45
        has_spasticity   = rng.random() < 0.35
        has_hcm          = rng.random() < 0.06   # RARE in NDUFS4 — KEY DDx SCO2 (100%)
        has_hepatopathy  = rng.random() < 0.07   # RARE — KEY DDx POLG/DGUOK (common)
        has_iron         = False                  # NO iron overload — KEY DDx GRACILE (100%)
        has_tubulopathy  = rng.random() < 0.06   # RARE — KEY DDx COX10 (65%)

        feat_list = ["Isolated Complex I deficiency (CII, CIII, CIV — NORMAL)", "Leigh / Leigh-like MRI (CARDINAL)"]
        if has_regression:  feat_list.append("Psychomotor regression/arrest")
        if has_hypotonia:   feat_list.append("Hypotonia")
        if has_lactic:      feat_list.append("Lactic acidosis")
        if has_olfactory:   feat_list.append("Olfactory bulb lesions on MRI (DISTINGUISHING — CI-Leigh)")
        if has_resp:        feat_list.append("Respiratory compromise / central apnoea")
        if has_seizures:    feat_list.append("Seizures")
        if has_ataxia:      feat_list.append("Ataxia")
        if has_dystonia:    feat_list.append("Dystonia")
        if has_nystagmus:   feat_list.append("Nystagmus")
        if has_optic:       feat_list.append("Optic atrophy")
        if has_spasticity:  feat_list.append("Spasticity")
        if has_hcm:         feat_list.append("HCM (RARE — KEY DDx SCO2 100%)")
        if has_hepatopathy: feat_list.append("Hepatopathy (RARE — KEY DDx POLG/DGUOK)")
        if has_tubulopathy: feat_list.append("Renal tubulopathy (RARE — KEY DDx COX10 65%)")

        txs     = rng.sample(_TX_POOL, k=rng.randint(4, 7))
        outcome = rng.choices(_OUTCOMES, weights=_OUT_WEIGHTS)[0]

        patients.append({
            "id":             f"NDUFS4-{i:03d}",
            "geno":           geno,
            "sex":            sex,
            "onset_yr":       onset_yr,
            "lactate_mm":     lactate_mm,
            "ci_pct":         ci_pct,
            "cii_pct":        cii_pct,
            "civ_pct":        civ_pct,
            "has_leigh_mri":  has_leigh_mri,
            "has_olfactory":  has_olfactory,
            "has_regression": has_regression,
            "has_hypotonia":  has_hypotonia,
            "has_lactic":     has_lactic,
            "has_resp":       has_resp,
            "has_seizures":   has_seizures,
            "has_ataxia":     has_ataxia,
            "has_optic":      has_optic,
            "has_nystagmus":  has_nystagmus,
            "has_dystonia":   has_dystonia,
            "has_spasticity": has_spasticity,
            "has_hcm":        has_hcm,
            "has_hepatopathy":has_hepatopathy,
            "has_iron":       has_iron,
            "has_tubulopathy":has_tubulopathy,
            "features":       ", ".join(feat_list[:7]),
            "treatments":     ", ".join(txs[:5]),
            "outcome":        outcome,
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
        "Psychomotor Regression / Arrest (universal in Leigh)":        _pct("has_regression"),
        "Leigh / Leigh-like MRI (bilateral putamen + brainstem)":      _pct("has_leigh_mri"),
        "Isolated Complex I Deficiency (CII, CIII, CIV Normal — 100%)": 100,
        "Hypotonia":                                                   _pct("has_hypotonia"),
        "Lactic Acidosis (elevated baseline + crisis)":                _pct("has_lactic"),
        "Olfactory Bulb Lesions on MRI (DISTINGUISHING CI-Leigh)":    _pct("has_olfactory"),
        "Respiratory Compromise / Central Apnoea":                     _pct("has_resp"),
        "Seizures":                                                    _pct("has_seizures"),
        "Dystonia":                                                    _pct("has_dystonia"),
        "Ataxia":                                                      _pct("has_ataxia"),
        "Nystagmus":                                                   _pct("has_nystagmus"),
        "Optic Atrophy":                                               _pct("has_optic"),
        "Spasticity":                                                  _pct("has_spasticity"),
        "HCM (RARE — KEY DDx SCO2 100%)":                             _pct("has_hcm"),
        "Hepatopathy (RARE — KEY DDx POLG / DGUOK)":                  _pct("has_hepatopathy"),
        "NO Hepatopathy (KEY DDx POLG / DGUOK — common in those)":    100,
        "NO HCM Dominant (KEY DDx SCO2 100% / COX15 78%)":            100,
        "NO Iron Overload (KEY DDx GRACILE — 100% iron overload)":    100,
        "NO COX Deficiency (KEY DDx SURF1 / SCO2 / COX10 / COX15)":  100,
        "Dutch / N.European Founder c.462delA (homozygous or compound het)": round(
            sum(1 for p in patients if "Dutch" in p["geno"] or "c.462delA" in p["geno"])
            / COHORT_SIZE * 100),
        "Alive (with support)":                                        round(alive / COHORT_SIZE * 100),
    }

    kpis = [
        {"label": "Cohort (n)",        "value": COHORT_SIZE,                                                        "color": COLOR},
        {"label": "Leigh MRI",         "value": f"{feature_frequencies['Leigh / Leigh-like MRI (bilateral putamen + brainstem)']}%", "color": "#6a1b9a"},
        {"label": "Olfactory Bulb MRI","value": f"{feature_frequencies['Olfactory Bulb Lesions on MRI (DISTINGUISHING CI-Leigh)']}%", "color": COLOR},
        {"label": "Resp Compromise",   "value": f"{feature_frequencies['Respiratory Compromise / Central Apnoea']}%", "color": "#e65100"},
        {"label": "Hypotonia",         "value": f"{feature_frequencies['Hypotonia']}%",                             "color": COLOR},
        {"label": "CI Activity",       "value": "5–20% control",                                                    "color": "#c62828"},
        {"label": "Fatal",             "value": f"{round(died/COHORT_SIZE*100)}%",                                  "color": "#b71c1c"},
        {"label": "Seed",              "value": f"#{SEED}",                                                         "color": "#455a64"},
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "severity": "ABSOLUTE CI — DANGEROUS IN ALL LEIGH / CI DEFICIENCY",
            "mechanism": (
                "Triple mechanism:\n"
                "1. CoA SEQUESTRATION: valproyl-CoA depletes the mitochondrial CoA pool "
                "   → acetyl-CoA and succinyl-CoA collapse → TCA cycle fails → OXPHOS substrate lost. "
                "   In NDUFS4/CI-Leigh, OXPHOS is already critically compromised (CI 5-20%); "
                "   CoA depletion tips the patient into irreversible lactic crisis.\n"
                "2. POLG INHIBITION: VPA inhibits mitochondrial DNA polymerase gamma → "
                "   reduces mtDNA copy number → fewer mtDNA-encoded ETC subunit templates. "
                "   Even though NDUFS4 itself is nuclear-encoded, CI also contains 7 mtDNA-encoded "
                "   ND subunits (MT-ND1–6, MT-ND4L) that are REQUIRED for CI structure; "
                "   reducing their synthesis worsens the already-assembled CI deficit.\n"
                "3. HEPATOTOXICITY: direct hepatotoxic risk. Although NDUFS4 is not primarily "
                "   a hepatic disease, any added hepatic stress in an OXPHOS-deficient child "
                "   risks metabolic decompensation (hepatic OXPHOS failure → impaired gluconeogenesis "
                "   → hypoglycaemia → lactic crisis).\n"
                "Use LEV (levetiracetam) as first-line AED: renal excretion, no mito toxicity."
            ),
        },
        {
            "drug": "Metformin",
            "severity": "ABSOLUTE CI — DIRECT COMPLEX I INHIBITOR: CATASTROPHIC IN NDUFS4",
            "mechanism": (
                "Metformin's primary mechanism of action is INHIBITION OF COMPLEX I (NADH dehydrogenase):\n"
                "  • Metformin → inhibits CI at the ND1/NDUFS2 quinone binding site → blocks "
                "    NADH → ubiquinone electron transfer.\n"
                "  • In NDUFS4 deficiency, CI is already at 5–20% of normal activity. "
                "    Metformin directly inhibits this residual CI, causing near-total CI shutdown.\n"
                "  • Result: NADH accumulates massively → pyruvate diverted entirely to lactate "
                "    → severe, potentially fatal lactic acidosis (lactate >15–20 mmol/L).\n\n"
                "This is a DIRECT pharmacodynamic catastrophe: the drug's target (CI) is the "
                "patient's primary disease locus. No exception for diabetes, PCOS, or insulin "
                "resistance in NDUFS4 carriers or affected patients.\n\n"
                "Alternative for glucose management: insulin only. "
                "Never: metformin, canagliflozin, empagliflozin (also mitotoxic at high dose)."
            ),
        },
        {
            "drug": "Phenobarbital",
            "severity": "HIGH CAUTION — Secondary Complex I Inhibitor",
            "mechanism": (
                "Phenobarbital and other barbiturates inhibit Complex I electron transport "
                "(binding near the quinone channel region) in addition to their GABA-A allosteric "
                "mechanism. In NDUFS4/CI-Leigh:\n"
                "  • Residual CI (5-20%) is the only electron-transfer capacity this patient has\n"
                "  • Phenobarbital's CI inhibition further reduces this residual capacity\n"
                "  • Risk of lactic decompensation at therapeutic doses\n\n"
                "Use LEV (preferred) or CLB (clobazam, benzodiazepine — no mito toxicity) instead. "
                "If phenobarbital is unavoidable, use the lowest effective dose with close lactate "
                "monitoring. Avoid long-term phenobarbital maintenance in any CI-Leigh patient."
            ),
        },
        {
            "drug": "Ketogenic Diet (KD)",
            "severity": "CONTRAINDICATED in isolated CI deficiency",
            "mechanism": (
                "The ketogenic diet forces beta-oxidation of fatty acids as the primary fuel. "
                "Beta-oxidation generates NADH (from acyl-CoA dehydrogenases and the TCA cycle "
                "step catalysed by isocitrate dehydrogenase and alpha-KGDH), which MUST be "
                "re-oxidised by Complex I (NADH dehydrogenase).\n\n"
                "In NDUFS4/CI-Leigh: CI cannot re-oxidise NADH efficiently (5-20% activity). "
                "Forcing high fat → beta-oxidation → NADH accumulation → NAD+ depletion → "
                "TCA cycle and beta-oxidation stall → massive NADH/NAD+ ratio elevation → "
                "worsened lactic acidosis and potential acute metabolic decompensation.\n\n"
                "KD is beneficial in Glucose Transporter Deficiency (GLUT1-DS) and Pyruvate "
                "Dehydrogenase Deficiency (PDHD) — but CONTRAINDICATED in Complex I (and III/IV) "
                "deficiencies where re-oxidising NADH at CI is the blocked step."
            ),
        },
        {
            "drug": "Linezolid",
            "severity": "ABSOLUTE CI — mt-Ribosome 23S rRNA Block",
            "mechanism": (
                "Linezolid inhibits the mitochondrial large ribosomal subunit (mt-LSU) 23S rRNA "
                "→ blocks elongation of ALL 13 mitochondrially-encoded OXPHOS subunit proteins.\n\n"
                "In NDUFS4/CI-Leigh:\n"
                "  • NDUFS4 itself is nuclear-encoded — but Complex I also requires 7 mtDNA-encoded "
                "    subunits (MT-ND1, MT-ND2, MT-ND3, MT-ND4, MT-ND4L, MT-ND5, MT-ND6)\n"
                "  • These are translated by the mitochondrial ribosome (the linezolid target)\n"
                "  • Linezolid → stops synthesis of all 7 ND subunits → residual CI assembly "
                "    intermediates cannot complete → CI activity collapses to near zero\n"
                "  • In a patient with already 5-20% CI → further complete CI block is fatal\n\n"
                "Alternatives for MRSA/VRE: vancomycin IV, daptomycin, tigecycline. "
                "Chloramphenicol has the SAME mt-ribosome mechanism and is equally CI."
            ),
        },
        {
            "drug": "Propofol",
            "severity": "AVOID / HIGH CAUTION — PRIS risk in CI-Leigh",
            "mechanism": (
                "Propofol Infusion Syndrome (PRIS): propofol inhibits electron transport at "
                "Complex IV (cytochrome c oxidase) and uncouples fatty acid beta-oxidation.\n\n"
                "In NDUFS4/CI-Leigh:\n"
                "  • CI is already the rate-limiting step in the ETC\n"
                "  • Propofol's CIV inhibition creates a second bottleneck downstream of CI\n"
                "  • Electrons accumulate between CI and CIV → increased ROS → oxidative stress\n"
                "  • Combined CI (NDUFS4) + CIV (propofol) inhibition → complete ETC collapse\n\n"
                "Use sevoflurane (volatile anaesthetic — no PRIS mechanism) or dexmedetomidine "
                "(alpha-2 agonist — no OXPHOS inhibition) as alternatives. "
                "If propofol is unavoidable for induction: use lowest dose, shortest duration; "
                "switch to sevoflurane maintenance immediately."
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
        "Psychomotor Regression":                                _pct("has_regression"),
        "Leigh/Leigh-like MRI":                                  _pct("has_leigh_mri"),
        "Olfactory Bulb MRI Lesions (DISTINGUISHING)":          _pct("has_olfactory"),
        "Hypotonia":                                             _pct("has_hypotonia"),
        "Lactic Acidosis":                                       _pct("has_lactic"),
        "Respiratory Compromise":                                _pct("has_resp"),
        "Seizures":                                              _pct("has_seizures"),
        "Dystonia":                                              _pct("has_dystonia"),
        "Ataxia":                                                _pct("has_ataxia"),
        "Nystagmus":                                             _pct("has_nystagmus"),
        "Optic Atrophy":                                         _pct("has_optic"),
        "Spasticity":                                            _pct("has_spasticity"),
        "HCM (RARE)":                                           _pct("has_hcm"),
        "Hepatopathy (RARE)":                                   _pct("has_hepatopathy"),
        "Renal Tubulopathy (RARE)":                             _pct("has_tubulopathy"),
    }

    genotype_distribution = {}
    for p in patients:
        g = p["geno"].split(" — ")[0]   # short label
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
            "note": "CII and CIV are NORMAL in NDUFS4 — isolated CI deficiency is the biochemical fingerprint",
        },
        "onset_summary": {
            "median_onset_mo": 8,
            "range_mo": "3–18 months",
            "note": "Earlier onset in biallelic null alleles (c.462delA homozygous: median 6 mo)",
        },
    }


# ── Definitions ───────────────────────────────────────────────────────────────
def get_definitions() -> dict[str, Any]:
    pharmacology = [
        {
            "term": "Metformin — ABSOLUTE CI in NDUFS4: Direct CI Inhibitor at Disease Target",
            "definition": (
                "Metformin is ABSOLUTELY CONTRAINDICATED in NDUFS4 and all other Complex I "
                "deficiency diseases:\n\n"
                "Mechanism: metformin binds the quinone access channel on the NDUFS2/ND1 "
                "subunit interface → directly inhibits NADH → ubiquinone electron transfer.\n\n"
                "In NDUFS4/CI-Leigh:\n"
                "  • Residual CI activity: 5–20% of control\n"
                "  • Metformin inhibits this residual activity → functional CI approaches 0%\n"
                "  • NADH cannot be oxidised → pyruvate MUST become lactate (LDH reaction)\n"
                "  • Lactate rises at 0.5–3 mmol/L per hour until fatal lactic acidosis\n\n"
                "This is a PHARMACODYNAMIC ABSOLUTE: the drug's target IS the patient's "
                "molecular defect.  No dose, no formulation, no scenario makes metformin "
                "safe in any patient with confirmed or suspected CI deficiency.\n\n"
                "If glucose intolerance or diabetes develops in a long-term NDUFS4 survivor:\n"
                "  • Insulin (basal–bolus or pump): safe — no OXPHOS inhibition\n"
                "  • GLP-1 agonists (semaglutide, liraglutide): use with caution; evidence limited\n"
                "  • NEVER: metformin, canagliflozin (SGLT2i at high dose also CI-inhibitory)"
            ),
        },
        {
            "term": "VPA — ABSOLUTE CI: Triple Mechanism Catastrophe in CI-Leigh",
            "definition": (
                "Valproate is ABSOLUTELY CONTRAINDICATED in NDUFS4 / CI-Leigh disease:\n\n"
                "1. CoA SEQUESTRATION (most immediate):\n"
                "   VPA → valproyl-CoA via ACSM (medium-chain acyl-CoA synthetase) → depletes "
                "   mitochondrial CoA pool → acetyl-CoA ↓↓ → TCA cycle substrate collapses "
                "   → OXPHOS substrate deficit on top of already-deficient CI.\n\n"
                "2. POLG INHIBITION (additive):\n"
                "   VPA inhibits mitochondrial DNA polymerase gamma (POLG) → mtDNA depletion "
                "   → ↓ MT-ND1–6 and MT-ND4L mRNA → fewer ND subunits available for "
                "   already-impaired CI N-module assembly.  Compounds NDUFS4 deficiency at "
                "   the upstream mtDNA template level.\n\n"
                "3. DIRECT OXPHOS INHIBITION (additional):\n"
                "   VPA uncouples mitochondrial membrane potential (ΔΨm) and inhibits "
                "   beta-oxidation → further NADH load on the already-blocked CI.\n\n"
                "AED FIRST LINE = LEVETIRACETAM (LEV):\n"
                "  • Renal excretion (66% unchanged) — no hepatic CYP involvement\n"
                "  • No mitochondrial toxicity (does not inhibit CI, CII, CIII, or CIV)\n"
                "  • IV formulation: critical for crisis seizure management\n"
                "  • IV loading: 20–40 mg/kg over 15 min in acute setting"
            ),
        },
        {
            "term": "Riboflavin (B2) — Level C: FMN Precursor, Complex I Specific",
            "definition": (
                "Riboflavin is the precursor to:\n"
                "  • FMN (Flavin MonoNucleotide) — the PRIMARY electron acceptor at Complex I "
                "    N-module; NADH donates 2e⁻ to FMN at the NDUFV1 subunit → first step "
                "    of CI electron relay to ubiquinone\n"
                "  • FAD (Flavin Adenine Dinucleotide) — electron carrier for Complex II, "
                "    fatty acid beta-oxidation, and the TCA cycle (SDHA)\n\n"
                "Rationale in NDUFS4:\n"
                "  Riboflavin supplementation increases FMN availability. In patients with "
                "  PARTIAL (residual-function) NDUFS4 alleles — particularly missense variants "
                "  where NDUFS4 protein is present but conformationally destabilised — "
                "  extra FMN substrate may stabilise the partially assembled N-module and "
                "  enhance residual CI activity.\n\n"
                "  Evidence level: C (case series + physiological rationale). "
                "  Clinical response is allele-specific: null alleles (c.462delA hom) show "
                "  less biochemical response than missense compounds.\n\n"
                "Dosing: 100–400 mg/day orally in divided doses (children: 5-10 mg/kg/day). "
                "Urinary flavinuria (yellow urine) indicates saturation — normal at these doses. "
                "Riboflavin is MORE specifically relevant for CI deficiency than for isolated "
                "COX diseases (SURF1, SCO2, COX10) where FMN is not rate-limiting for CIV."
            ),
        },
        {
            "term": "Succinate — Level C: Complex I Bypass via Complex II (SDHA)",
            "definition": (
                "Succinate provides electrons directly to the ubiquinone pool via "
                "Complex II (succinate dehydrogenase / SDHA), BYPASSING Complex I entirely:\n\n"
                "  Succinate → [CII/SDHA] → Ubiquinol (QH₂) → [CIII] → Cytochrome c → [CIV] → O₂\n\n"
                "In NDUFS4/CI-Leigh:\n"
                "  • CI (NADH → ubiquinol route): BLOCKED (5–20% activity)\n"
                "  • CII (succinate → ubiquinol route): NORMAL (SDHA is fully nuclear-encoded "
                "    and not affected by NDUFS4 deficiency)\n\n"
                "Supplemental succinate aims to maintain downstream ETC flux (CIII → CIV → ATP "
                "synthesis) using the intact CII route, partially compensating for the blocked CI.\n\n"
                "This BYPASS strategy is not available in isolated COX diseases (SURF1, SCO2, etc.) "
                "where the blockade is downstream of CII — making succinate uniquely valuable "
                "in CI deficiency.\n\n"
                "Dosing: 2–8 g/day orally (1–2 g/dose q6-8h); IV succinic acid in acute crisis "
                "(experimental, metabolic centres only). Note: succinate does not cross the "
                "blood–brain barrier well — CNS benefit is indirect via systemic acidosis reduction."
            ),
        },
        {
            "term": "Olfactory Bulb Lesions — DISTINGUISHING MRI Feature of NDUFS4/CI-Leigh",
            "definition": (
                "Olfactory bulb and olfactory cortex T2/FLAIR hyperintensity lesions are a "
                "distinguishing MRI feature of NDUFS4 deficiency (CI-Leigh), first described "
                "systematically in NDUFS4 knockout mice (Kruse et al., Cell Metab 2008) and "
                "subsequently confirmed in human cohorts.\n\n"
                "Mechanism hypothesis:\n"
                "  • Olfactory bulb neurons are among the highest oxidative-demand cells in "
                "    the CNS (continuous Ca²⁺ signalling + rapid action potential generation)\n"
                "  • Their dependence on CI-mediated NADH oxidation makes them particularly "
                "    vulnerable to NDUFS4-type CI deficiency\n"
                "  • Olfactory system lacks full retrograde trophic protection against OXPHOS failure\n\n"
                "Clinical significance:\n"
                "  • NOT seen in SURF1 (COX-Leigh), SCO2, COX10, COX15, or LRPPRC disease\n"
                "  • Present in ~60–70% of NDUFS4 cases on MRI (T2/FLAIR thin-section protocol)\n"
                "  • May be subtle: require dedicated coronal olfactory bulb sequences\n"
                "  • Helps distinguish CI-Leigh (NDUFS4) from COX-Leigh (SURF1) without waiting "
                "    for enzyme or genetic results in the acute setting\n\n"
                "Other CI-Leigh genes (NDUFV1, NDUFA9) generally show STANDARD Leigh MRI "
                "(putamen + brainstem) without prominent olfactory bulb involvement, "
                "suggesting NDUFS4 specifically may drive olfactory vulnerability."
            ),
        },
    ]

    gene_concepts = [
        {
            "term": "NDUFS4 — N-Module Subunit S4 (AQDQ): Role in Complex I Assembly",
            "definition": (
                "NDUFS4 (also called AQDQ subunit for its amino acid composition) is one of "
                "the 45 subunits of mammalian Complex I (NADH:Ubiquinone Oxidoreductase).\n\n"
                "Complex I architecture (key modules):\n"
                "  • N-module (Peripheral arm, matrix-facing): NADH binding + initial electron "
                "    acceptance. Contains FMN and iron-sulfur clusters N1a, N3, N4, N5, N6a, N6b.\n"
                "  • Q-module (peripheral arm, proximal): ubiquinone binding pocket (ND2, ND3, ND4L)\n"
                "  • P-module (membrane arm): proton pumping (ND2, ND4, ND5, ND1)\n\n"
                "NDUFS4 function:\n"
                "  • Localises to the N-module tip (distal from membrane)\n"
                "  • Required for correct assembly of the N-module onto the nascent CI complex\n"
                "  • Without NDUFS4: CI stalls as an ~830 kDa assembly intermediate "
                "    (Q+P modules partially assembled, N-module cannot dock)\n"
                "  • The 830 kDa intermediate lacks FMN and cannot transfer electrons\n"
                "  • No NDUFS4 → no functional CI → isolated CI deficiency\n\n"
                "Nuclear vs. mitochondrial encoded subunits in CI:\n"
                "  • 7 subunits: mtDNA-encoded (MT-ND1–6, MT-ND4L) — targeted by linezolid\n"
                "  • 38 subunits: nuclear-encoded (NDUFS1-8, NDUFV1-3, NDUFA1-13, etc.) "
                "    — including NDUFS4\n"
                "  All 45 must assemble correctly for functional CI."
            ),
        },
        {
            "term": "Isolated vs. Combined OXPHOS Deficiency — DDx Framework",
            "definition": (
                "The OXPHOS deficiency pattern in respiratory chain spectrophotometry distinguishes "
                "NDUFS4/CI-Leigh from other Leigh syndrome causes:\n\n"
                "NDUFS4 / CI-Leigh pattern:\n"
                "  Complex I:    5–20% of control   (MARKEDLY REDUCED)\n"
                "  Complex II:   85–115% (NORMAL)\n"
                "  Complex III:  80–110% (NORMAL)\n"
                "  Complex IV:   80–110% (NORMAL)\n"
                "  → ISOLATED CI deficiency (CII, CIII, CIV all normal)\n\n"
                "Compare to:\n"
                "  SURF1/SCO2/COX10/COX15: Isolated CIV (COX) deficiency (CI normal)\n"
                "  LRPPRC (LSFC):          Combined CI + CIV deficiency (both reduced)\n"
                "  TACO1:                  Isolated CIV (specific MT-CO1 translation block)\n"
                "  GRACILE (BCS1L):        Isolated CIII deficiency (RISP insertion block)\n"
                "  POLG / DGUOK:           Multiple OXPHOS deficiencies (mtDNA depletion)\n"
                "  MELAS:                  CI + CIV combined (mt-tRNA-Leu → multiple subunits)\n\n"
                "Citrate synthase (CS):\n"
                "  Often 130–200% of control in Leigh syndrome (mitochondrial proliferation "
                "  compensatory response). CI % should be expressed per CS activity "
                "(CI/CS ratio) to correct for mitochondrial mass differences."
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "Leigh Syndrome — Neuropathological and Radiological Criteria",
            "definition": (
                "Leigh syndrome (LS, subacute necrotising encephalomyelopathy) is defined by:\n\n"
                "Neuropathological criteria (original Rahman 1996 / Sofou 2018):\n"
                "  1. Symmetrical focal necrotic lesions in basal ganglia, thalamus, "
                "     brainstem, cerebellum, or spinal cord\n"
                "  2. Capillary proliferation in affected areas\n"
                "  3. Reactive gliosis\n"
                "  4. Relative preservation of neurons in necrotic foci\n\n"
                "MRI criteria (clinical diagnosis):\n"
                "  • T2/FLAIR bilateral symmetric hyperintensity in putamen (most common)\n"
                "  • Caudate, globus pallidus, thalamus, subthalamic nuclei\n"
                "  • Brainstem tegmentum (periaqueductal grey, medullary olive)\n"
                "  • DWI restriction in acute lesions\n"
                "  • Leigh MRI + genetic cause + biochemical defect = definite Leigh\n\n"
                "Genetic causes (>75 genes, OMIM 256000):\n"
                "  • CI deficiency (~35% of all Leigh): NDUFS4, NDUFV1, NDUFA9, NDUFAF4/NUBPL, "
                "    NDUFS1, NDUFAF5, etc.\n"
                "  • CIV deficiency (~20%): SURF1, SCO2, SCO1, COX10, COX15, TACO1, LRPPRC\n"
                "  • mtDNA mutations (~15%): MT-ATP6 (MILS), MT-TL1 (MELAS), mtDNA deletions\n"
                "  • PDHD, PC, POLG, TWNK, SLC25A4, mtDNA depletion syndromes (~30%)"
            ),
        },
        {
            "term": "NDUFS4 — Key DDx Distinguishers from Other Leigh Syndrome Genes",
            "definition": (
                "Critical distinguishing features of NDUFS4/CI-Leigh vs. other Leigh causes:\n\n"
                "vs. SURF1 (COX-Leigh — isolated CIV):\n"
                "  NDUFS4: CI 5-20%; CIV NORMAL | SURF1: CIV 5-20%; CI NORMAL\n"
                "  NDUFS4: olfactory bulb MRI 65% | SURF1: olfactory bulb NOT involved\n"
                "  Both: NO HCM, NO hepatopathy, NO iron overload\n\n"
                "vs. SCO2 (fatal infantile HCM, CIV):\n"
                "  SCO2: HCM 100% CARDINAL | NDUFS4: HCM <6% (RARE)\n"
                "  → If HCM present, strongly favours SCO2 (or COX15 if milder HCM 78%)\n\n"
                "vs. LRPPRC (combined CI+CIV — LSFC):\n"
                "  NDUFS4: CI ↓↓, CIV NORMAL | LRPPRC: BOTH CI + CIV reduced\n"
                "  LRPPRC: episodic metabolic crises CARDINAL | NDUFS4: less episodic (more progressive)\n"
                "  LRPPRC: French-Canadian founder (SLSJ) | NDUFS4: Dutch/N.European founder\n\n"
                "vs. GRACILE (BCS1L — CIII):\n"
                "  GRACILE: Iron overload ferritin >2000 + cholestasis + IUGR | NDUFS4: NO iron\n"
                "  GRACILE: CIII isolated | NDUFS4: CI isolated\n\n"
                "vs. POLG / DGUOK (mtDNA depletion):\n"
                "  POLG: hepatopathy + EPC common | NDUFS4: hepatopathy RARE\n"
                "  DGUOK: rotary nystagmus 90% pathognomonic | NDUFS4: nystagmus less specific\n"
                "  mtDNA depletion: MULTIPLE OXPHOS complexes | NDUFS4: isolated CI only\n\n"
                "vs. MTFMT (mt-formyl transferase — CI translation):\n"
                "  MTFMT: similar CI-Leigh phenotype | NDUFS4: c.462delA Dutch founder key\n"
                "  Both: isolated CI deficiency + Leigh MRI + similar management"
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "LEV — Preferred AED in NDUFS4/CI-Leigh (First-Line, IV-Capable)",
            "definition": (
                "Levetiracetam (LEV) is the AED of choice in all Leigh syndrome / Complex I "
                "deficiency:\n\n"
                "1. RENAL excretion: 66% unchanged via kidney; no hepatic CYP metabolism "
                "   → No CYP interactions with cofactor supplements or mitochondrial medications\n"
                "2. No Complex I inhibition: does not bind NDUFS4, NDUFS2, or any CI subunit\n"
                "3. Broad-spectrum: focal + generalised + myoclonic seizures (all common in Leigh)\n"
                "4. IV formulation: critical in acute crisis with seizures\n"
                "   Loading: 20–40 mg/kg IV over 15 min\n"
                "   Maintenance: 20–60 mg/kg/day divided q8-12h\n"
                "5. Behavioural side effects (irritability, aggression) monitored; "
                "   pyridoxine 50–100 mg/day may mitigate LEV-related irritability\n\n"
                "AVOID / ABSOLUTE CI in NDUFS4:\n"
                "  • VPA: ABSOLUTE CI (triple mito toxicity)\n"
                "  • Phenobarbital: HIGH CAUTION (direct CI inhibitor)\n"
                "  • Phenytoin: hepatic metabolism (CYP2C9/2C19 — variable CI activity "
                "    in OXPHOS-stressed liver)\n"
                "  • Carbamazepine (CBZ): CYP3A4 inducer; cardiovascular effects; "
                "    acceptable only if LEV and CLB fail\n"
                "  • Tiagabine (TGB): ABSOLUTE CI (GABA reuptake block → NCSE risk)"
            ),
        },
        {
            "term": "Crisis Management Protocol — NDUFS4/CI-Leigh Acute Decompensation",
            "definition": (
                "Acute metabolic decompensation (lactic crisis) in NDUFS4/CI-Leigh:\n\n"
                "TRIGGER recognition:\n"
                "  • Any febrile illness, surgery, fasting, anaesthesia, infection\n"
                "  • Signs: acute encephalopathy, worsening hypotonia, new seizures, "
                "    respiratory compromise, lactate >10 mmol/L\n\n"
                "IMMEDIATE interventions:\n"
                "  1. IV DEXTROSE: GIR 6-8 mg/kg/min immediately "
                "(glucose-centric: feed the residual CI pathway with substrates; "
                "AVOID fasting at all times in CI-Leigh)\n"
                "  2. STOP any potential mitochondrial toxins: hold phenobarbital, "
                "     discontinue any inadvertent metformin (if prescribed by another service)\n"
                "  3. NaHCO3 IV: if pH <7.20 or HCO3 <12 mEq/L\n"
                "     (0.5–1 mEq/kg IV over 1-2h; recheck pH q2h)\n"
                "  4. IV RIBOFLAVIN + THIAMINE: 100 mg each IV if available\n"
                "     (both are IV-safe; thiamine mandatory in ANY acute encephalopathy)\n"
                "  5. IV SUCCINATE (if available, metabolic centre): 0.5–1 g/kg/day "
                "     → provides CII-mediated electron input, bypasses blocked CI\n"
                "  6. SEIZURES: LEV IV 20-40 mg/kg loading dose\n"
                "  7. RESPIRATORY: NIV/BiPAP if SpO2 <92% or respiratory rate >40; "
                "     intubation only if unavoidable (sevoflurane, NOT propofol for induction)\n\n"
                "EMERGENCY CARD:\n"
                "  Carried by family at all times. ER instructions: IV dextrose immediately; "
                "NEVER VPA, metformin, linezolid, propofol, or ketogenic diet. "
                "Contact metabolic neurology (phone number on card)."
            ),
        },
        {
            "term": "Cofactor Therapy — CoQ10, Riboflavin, Thiamine, Biotin, Succinate in CI-Leigh",
            "definition": (
                "Standard mitochondrial cofactor therapy in NDUFS4/CI-Leigh (all Level C evidence "
                "unless stated):\n\n"
                "CoQ10 / Ubiquinol: 300-600 mg/day adults; 10-30 mg/kg/day children\n"
                "  • Mobile electron carrier: CI → ubiquinone → CIII\n"
                "  • Supplemental CoQ10 may provide extra acceptor capacity for residual CI\n"
                "  • Ubiquinol (reduced form) preferred: better bioavailability\n\n"
                "Riboflavin (B2): 100-400 mg/day — CI SPECIFIC\n"
                "  • FMN precursor: the primary electron acceptor at the NDUFV1 N-module site\n"
                "  • More relevant in CI deficiency than in COX diseases\n"
                "  • Biochemical response testing (CI activity before/after riboflavin loading "
                "    2 weeks) can guide continuation\n\n"
                "Thiamine (B1): 100-300 mg/day — MANDATORY EMPIRIC IN ALL LEIGH\n"
                "  • PDH and alpha-KGDH cofactor; SLC19A3 (biotin-thiamine-responsive "
                "    basal ganglia disease) is a CURABLE Leigh mimic\n"
                "  • Give empirically BEFORE molecular diagnosis in ALL Leigh presentations\n\n"
                "Biotin: 5-20 mg/day — MANDATORY EMPIRIC\n"
                "  • Biotinidase deficiency (BTD) is a CURABLE mimic of Leigh syndrome\n"
                "  • Give empirically until BTD enzyme activity confirmed\n\n"
                "Succinate: 2-8 g/day — CI BYPASS (most CI-specific supplement)\n"
                "  • Feeds CII directly → bypasses blocked CI entirely\n"
                "  • Not useful in isolated COX diseases (SURF1/SCO2) as it doesn't bypass CIV\n\n"
                "Carnitine: 50-100 mg/kg/day\n"
                "  • Secondary carnitine deficiency common; repletes carnitine pool\n"
                "  • Avoid if CI-driven crisis is active (transient acylcarnitine accumulation "
                "    can occur and should not be confused with primary carnitine disorder)"
            ),
        },
    ]

    return {
        "pharmacology":      pharmacology,
        "gene_concepts":     gene_concepts,
        "disease_concepts":  disease_concepts,
        "prescribing_safety":prescribing_safety,
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
