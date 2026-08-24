#!/usr/bin/env python3
"""PYCR1 (Pyrroline-5-Carboxylate Reductase 1 Deficiency) Epilepsy Dashboard.

PYCR1 encodes Pyrroline-5-Carboxylate Reductase 1, the mitochondrial enzyme that
catalyses the FINAL step of de novo proline synthesis:
  P5C (Delta-1-Pyrroline-5-Carboxylate) + NADPH → L-Proline + NADP+

PROLINE SYNTHESIS PATHWAY (PYCR1 is step 2, immediately downstream of ALDH18A1):
  L-Glutamate
    → [ALDH18A1/P5CS, ATP+2NADPH] → P5C / Glutamate-5-Semialdehyde
    → [PYCR1, NADPH] → L-Proline (final product — exited to cytoplasm for protein synthesis)

  Proline catabolism (opposite direction):
    L-Proline → [PRODH] → P5C → [ALDH4A1] → L-Glutamate

PYCR1 ENZYMATIC FUNCTION:
  319 aa; mitochondrial matrix; NADPH-dependent; homopentameric ring complex;
  reduces the cyclic imino acid P5C (= open-chain GSA in equilibrium) to L-Proline.
  Three human PYCR isozymes: PYCR1 (mito, major), PYCR2 (mito, brain-enriched), PYCRL (cytoplasmic).
  PYCR1 provides ~60-70% of de novo proline in peripheral tissues + CNS.

PYCR1 LOF — PROLINE LOW; ORNITHINE NORMAL-LOW; P5C MILDLY ELEVATED:
  PYCR1 cannot convert P5C → Proline →
  → Proline CANNOT be synthesised de novo → Proline CRITICALLY LOW (hypoProlinemia)
  → P5C mildly ACCUMULATES (PYCR1 substrate backs up — NOT pathognomonic like ALDH4A1)
  → P5C does NOT rise enough for significant PLP inactivation (contrast with ALDH4A1)
  → PLP NORMAL (no secondary B6 deficiency)
  → Ornithine LOW-NORMAL (P5C ↔ Ornithine via OAT still partially functional)
  → Collagen: proline-deficient → hydroxyproline LOW → defective collagen triple helix

  THE CRITICAL DISTINCTIONS within the proline pathway:
  PRODH LOF   → Proline ELEVATED (catabolism step 1 blocked)
  ALDH4A1 LOF → Proline MARKEDLY ELEVATED + P5C MARKEDLY ELEVATED + PLP LOW (secondary B6 defic.)
  ALDH18A1 LOF → Proline CRITICALLY LOW + P5C NOT MADE + Ornithine LOW (synthesis entry step)
  PYCR1 LOF  → Proline CRITICALLY LOW + P5C MILDLY ELEVATED + Ornithine LOW-NORMAL (synthesis exit)

THE EPILEPTOGENIC MECHANISMS IN PYCR1 DEFICIENCY:
  Mechanism 1 — Proline depletion → reduced glutamate/GABA substrate:
    Proline is a precursor for glutamate in neurons (reverse catabolism: Proline→P5C→Glu).
    Low proline → reduced available glutamate → secondary GABA substrate depletion.
    CNS-level proline depletion disrupts the glutamate/GABA balance toward excitation.

  Mechanism 2 — Disrupted PYCR1–PYCR2 mitochondrial redox axis (NADP+/NADPH cycling):
    PYCR1/2 recycle NADP+ to NADPH in mitochondria; this is tightly coupled to the
    malate-aspartate shuttle and mitochondrial redox homeostasis.
    PYCR1 LOF → excess NADP+ (oxidized) → mitochondrial oxidative stress.
    Reactive oxygen species (ROS) increase → neuronal mitochondrial dysfunction.
    Mitochondrial dysfunction → epileptogenesis (similar to other mitochondrial epilepsies).

  Mechanism 3 — Structural: proline-deficient collagen → CNS vascular fragility:
    Proline/hydroxyproline are essential for collagen Gly-Pro-Hyp triple helix stability.
    PYCR1 LOF → collagen proline content LOW → defective connective tissue throughout.
    CNS: leptomeningeal and vascular collagen deficiency → cortical structural fragility.
    In severe cases: simplified gyral pattern, periventricular leukomalacia reported.

  Mechanism 4 — Polyamine pathway disruption (secondary to ornithine depletion):
    Ornithine feeds polyamine synthesis: Ornithine → Putrescine → Spermidine → Spermine.
    PYCR1 LOF with OAT reversal impaired → ornithine mildly LOW-NORMAL → polyamine depletion.
    Polyamines modulate NMDA receptor gating; their loss → altered neuronal excitability.

WHY DISTINCT FROM ALDH18A1 (upstream P5CS):
  ALDH18A1 LOF → cannot MAKE P5C at all (Glu→P5C blocked) → Ornithine VERY LOW (<30)
  PYCR1 LOF    → CAN make P5C (ALDH18A1 intact) → P5C mildly elevated; Ornithine mildly LOW-NORMAL
  Both share:  Proline CRITICALLY LOW; collagen failure; cutis laxa (PYCR1 less severe than ALDH18A1)
  ALDH18A1 has ALDH5A1/GSS domain deficiency component + citrulline/arginine cascade
  PYCR1:  ornithine cascade milder; P5C is made but cannot be reduced; distinct variants

PYCR1 DISEASE:
  Autosomal Recessive Cutis Laxa Type IIB (ARCL2B) / De Barsy syndrome-like
  OMIM Gene: *179035   OMIM Disease: #612940
  Chromosome: 17q25.3
  Protein: 319 aa; mitochondrial; NADPH-dependent; homopentameric
  Inheritance: AR — biallelic LOF
  Prevalence: ~80–150 cases worldwide 2026 (ultrarare; many undiagnosed pre-WES era)

EPILEPSY IN PYCR1 DEFICIENCY:
  Seizures in 40–55% overall (ranges by severity):
    Severe neonatal-onset AR: 55–65% seizures
    Classic infantile AR: 40–50% seizures
    Mild AR: 15–25% seizures
  Seizure types: Infantile spasms/West syndrome (~30% of seizure patients),
    multifocal clonic (~25%), GTCS (~20%), focal cortical (~15%), absence-like (~10%)
  Drug-resistant epilepsy: 15–25% (less than ALDH18A1)
  Pyridoxine: NO response (PLP is NORMAL; no P5C-PLP inactivation — unlike ALDH4A1)
  B6 is NOT indicated in PYCR1 (P5C does NOT accumulate sufficiently to inactivate PLP)

CLINICAL FEATURES:
  Cutis laxa (lax/wrinkled/loose skin): 90% — hallmark; present at birth
  Facial dysmorphism: 85% — frontal bossing, downslanting palpebral fissures, beaked nose
  Intellectual disability: 75% severe, 20% moderate (less severe than ALDH18A1 ~90%)
  Microcephaly: 45%
  Brain MRI: simplified gyral pattern (30%), periventricular leukomalacia (25%),
    thin corpus callosum (20%), white matter abnormalities (35%)
  Joint laxity: 65% (less than ALDH18A1 90%)
  Failure to thrive: 70%
  Cataracts: 20% (less than ALDH18A1 60–75%) — proline less critical for PYCR1 ocular lens
  Feeding difficulties: 60%
  Ophthalmology: strabismus 30%, nystagmus 15%

TREATMENT:
  L-Proline supplementation: Level A (primary treatment — same as ALDH18A1)
    Dose: 100–400 mg/kg/day (may need less than ALDH18A1 as some OAT-reverse compensation)
    Goal: plasma proline 150–300 µmol/L (normal 100–260; avoid excess which can cause PRODH-type)
  L-Ornithine: Level B (if ornithine measured LOW)
  Vitamin C: Level B (cofactor for prolyl hydroxylase — collagen synthesis support)
  Protein intake: ADEQUATE (NOT restricted — restriction worsens proline deficiency; same ALDH18A1)
  LEV: Level B (first-line AED, no metabolic interactions)
  ACTH/Vigabatrin: Level A for infantile spasms
  VPA: MODERATE RISK (hyperammonemia risk if ornithine low; same mechanism as ALDH18A1)
  Pyridoxine/B6: NOT INDICATED (PLP normal — no P5C-PLP inactivation, unlike ALDH4A1)
  Protein restriction: ABSOLUTE CONTRAINDICATION (same as ALDH18A1 — worsens proline deficiency)
  Physical therapy: supportive for connective tissue features
  Ophthalmology surveillance: yearly (strabismus, cataracts)
  Cardiac monitoring: vascular collagen deficiency risk

KEY VARIANTS (selected, approximate frequencies):
  p.Arg119Pro (substrate-binding pocket) — 22% — severe AR, most common worldwide
  p.Leu175Pro (dimer/pentamer interface) — 18% — severe AR, pentamer destabilization
  p.Ala225Val (NADPH-binding domain) — 15% — moderate AR, partial activity
  p.Gly93Ser (N-terminal active site loop) — 12% — severe AR, null activity
  p.Arg251Cys (C-terminal mitochondrial targeting) — 10% — moderate AR
  c.IVS5+1G>A (splice null) — 9% — severe AR, exon skip → framshift
  p.Tyr156Cys (cofactor coordination) — 8% — moderate-severe AR
  p.Val277Ile (mild partial function) — 6% — mild AR attenuated phenotype

PHENOTYPE DISTRIBUTION (40-patient cohort, seed 157):
  Severe neonatal-onset AR: 40% (16/40)
  Classic infantile AR: 45% (18/40)
  Mild AR (attenuated): 15% (6/40)

BIOCHEMICAL KEY NEGATIVES (what is NORMAL in PYCR1 — rules out other disorders):
  P5C NOT MARKEDLY ELEVATED — KEY vs ALDH4A1 (where P5C is PATHOGNOMONIC HIGH)
  PLP NORMAL — KEY vs ALDH4A1 (no B6 deficiency)
  alpha-AASA NORMAL — KEY vs ALDH7A1/PDE (antiquitin deficiency)
  Pipecolic acid NORMAL — KEY vs ALDH7A1/PDE
  MMA NORMAL — KEY vs MMUT/cobalamin disorders
  tHcy NORMAL — KEY vs CBS/MTHFR/MTR/MTRR/AHCY
  Methionine NORMAL — KEY vs GNMT/MAT1A/AHCY
  Lactate NORMAL–mildly elevated — KEY vs POLG/mitochondrial epilepsies
  Ammonia NORMAL–borderline — (contrast ALDH18A1 where ornithine MORE depleted)
  Sarcosine NORMAL — KEY vs SARDH
"""

import random
import math

_SEED = 157
_N = 40


def _rng():
    return random.Random(_SEED)


def get_overview():
    rng = _rng()

    # Phenotype distribution
    phenotypes = {
        "Severe-Neonatal-AR": {"n": 16, "pct": 40},
        "Classic-Infantile-AR": {"n": 18, "pct": 45},
        "Mild-AR-Attenuated": {"n": 6, "pct": 15},
    }

    # KPI stats — generated deterministically
    # Proline critically LOW (synthesis failure) — <50 umol/L (normal 100–260)
    prolins = [rng.gauss(38, 9) for _ in range(_N)]
    prolins = [max(5, p) for p in prolins]

    # P5C mildly elevated (substrate backs up, NOT pathognomonic)
    p5cs = [rng.gauss(9.2, 2.8) for _ in range(_N)]
    p5cs = [max(4, p) for p in p5cs]

    # PLP NORMAL (no B6 deficiency — unlike ALDH4A1)
    plps = [rng.gauss(62, 12) for _ in range(_N)]
    plps = [max(35, min(110, p)) for p in plps]

    # Ornithine mildly LOW-NORMAL
    ornitines = [rng.gauss(41, 10) for _ in range(_N)]
    ornitines = [max(18, p) for p in ornitines]

    # tHcy NORMAL
    thcys = [rng.gauss(9.8, 2.1) for _ in range(_N)]

    # Seizures: 40–55% overall
    n_seizures = round(0.475 * _N)  # 19/40
    n_dre = round(0.175 * _N)  # 7/40
    n_b6_trial = round(0.12 * _N)  # 5/40 (tried but none responded)
    n_b6_resp = 0  # 0 — PLP normal, B6 has no mechanism
    n_idd = round(0.80 * _N)  # 32/40 (75% severe + moderate mild in sample)
    n_nbs = round(0.28 * _N)  # 11/40 (some with expanded NBS proline detection)
    n_cutis_laxa = round(0.90 * _N)  # 36/40
    n_proline_supp = round(0.82 * _N)  # 33/40

    return {
        "dashboard_id": "pycr1",
        "title": "PYCR1 Epilepsy Dashboard",
        "subtitle": "Autosomal Recessive Cutis Laxa 2B — Pyrroline-5-Carboxylate Reductase 1 Deficiency / Proline SYNTHESIS Exit Step Failure / Proline CRITICALLY LOW",
        "gene": "PYCR1",
        "disease_name": "Autosomal Recessive Cutis Laxa Type 2B (PYCR1 Deficiency / ARCL2B)",
        "chromosome": "17q25.3",
        "inheritance": "Autosomal Recessive — biallelic LOF; AR only",
        "omim_gene": "OMIM *179035",
        "omim_disease": "OMIM #612940",
        "protein_size": "319 aa; NADPH-dependent mitochondrial enzyme; homopentameric ring complex; Pyrroline-5-Carboxylate Reductase family",
        "prevalence": "~80–150 cases worldwide (2026); ultrarare; significantly underdiagnosed pre-WES/WGS era",
        "cohort_n": _N,
        "phenotype_distribution": phenotypes,
        "function": (
            "PYCR1 catalyses the FINAL step of de novo proline synthesis: "
            "P5C (Delta-1-Pyrroline-5-Carboxylate) + NADPH + H⁺ → L-Proline + NADP⁺. "
            "This is step 2 of proline synthesis, immediately downstream of ALDH18A1/P5CS "
            "(which makes P5C from glutamate). Three human PYCR isozymes exist: "
            "PYCR1 (mitochondrial, major — ~60–70%), PYCR2 (mitochondrial, brain-enriched), "
            "PYCRL (cytoplasmic). PYCR1 additionally maintains mitochondrial NADP⁺/NADPH "
            "redox homeostasis via its catalytic cycling."
        ),
        "mechanism": (
            "PYCR1 LOF → P5C cannot be reduced to proline → proline CRITICALLY LOW "
            "(hypoProlinemia; plasma proline <50 µmol/L; normal 100–260). "
            "P5C mildly accumulates (substrate backup) — but NOT to levels seen in ALDH4A1 "
            "(where P5C is PATHOGNOMONIC HIGH >80 µmol/L). PLP is NOT inactivated by P5C "
            "in PYCR1 (P5C levels too low for significant Schiff-base formation with PLP). "
            "Collagen: proline is required for Gly-Pro-Hyp repeats in collagen triple helix; "
            "PYCR1 LOF → hydroxyproline LOW → defective collagen → cutis laxa (hallmark). "
            "CNS: mitochondrial NADP⁺/NADPH redox disruption + proline depletion in neurons "
            "→ reduced glutamate substrate → GABA synthesis compromise → seizures. "
            "Oxidative stress from disrupted PYCR1–PYCR2 NADPH recycling axis amplifies "
            "neuronal damage and epileptogenicity."
        ),
        "key_positive_features": (
            "Proline CRITICALLY LOW (<50 µmol/L; normal 100–260) — PRIMARY BIOMARKER · "
            "Cutis laxa (lax/wrinkled skin) at birth — CLINICAL HALLMARK (90%) · "
            "Facial dysmorphism (frontal bossing, beaked nose) — 85% · "
            "P5C mildly elevated (9–12 µmol/L normal <6) — substrate backup from ALDH18A1; "
            "NOT as high as ALDH4A1 (PATHOGNOMONIC >80 µmol/L) · "
            "Ornithine mildly LOW-NORMAL (35–50 µmol/L) — OAT reversal partially compensates"
        ),
        "key_negative_features": (
            "P5C NOT MARKEDLY ELEVATED — KEY vs ALDH4A1 (pathognomonic >80 µmol/L in ALDH4A1) · "
            "PLP NORMAL — KEY vs ALDH4A1 (no secondary B6 deficiency; B6 has NO indication in PYCR1) · "
            "alpha-AASA NORMAL — KEY vs ALDH7A1/PDE (antiquitin; pathognomonic elevated) · "
            "Pipecolic acid NORMAL — KEY vs ALDH7A1/PDE · "
            "MMA NORMAL — KEY vs MMUT/cobalamin disorders · "
            "tHcy NORMAL (<15 µmol/L) — KEY vs CBS/MTHFR/MTR/MTRR/AHCY · "
            "Methionine NORMAL — KEY vs GNMT/MAT1A/AHCY · "
            "Ammonia NORMAL–borderline — KEY vs ALDH18A1 (where ornithine MORE depleted) · "
            "Proline NOT elevated — KEY vs PRODH/ALDH4A1 (catabolism disorders; proline HIGH)"
        ),
        "nbs_primary": (
            "Plasma amino acids — proline CRITICALLY LOW (<50 µmol/L) on expanded NBS panel. "
            "Standard NBS C3/C4 acylcarnitine panels do NOT detect PYCR1. "
            "Low proline on amino acid panel triggers reflex PYCR1 sequencing. "
            "Clinical diagnosis often before NBS: cutis laxa at birth + low proline."
        ),
        "nbs_secondary": (
            "P5C plasma assay (mildly elevated, not pathognomonic); "
            "Ornithine (mildly LOW-NORMAL); PLP (NORMAL); "
            "urine hydroxyproline (LOW — collagen synthesis marker); "
            "WES/WGS for PYCR1 (17q25.3) biallelic variants — mandatory for confirmation"
        ),
        "kpi": {
            "avg_proline_umol_l": round(sum(prolins) / _N, 1),
            "avg_p5c_umol_l": round(sum(p5cs) / _N, 1),
            "avg_plp_nmol_l": round(sum(plps) / _N, 1),
            "avg_ornithine_umol_l": round(sum(ornitines) / _N, 1),
            "avg_thcy_umol_l": round(sum(thcys) / _N, 1),
            "pct_seizures": round(100 * n_seizures / _N),
            "pct_dre": round(100 * n_dre / _N),
            "pct_b6_trial": round(100 * n_b6_trial / _N),
            "pct_b6_responded": 0,
            "pct_idd": round(100 * n_idd / _N),
            "pct_nbs_detected": round(100 * n_nbs / _N),
            "pct_cutis_laxa": round(100 * n_cutis_laxa / _N),
            "pct_proline_supplemented": round(100 * n_proline_supp / _N),
            "pct_p5c_normal": 0,   # P5C mildly elevated (not normal, not pathognomonic)
            "pct_plp_normal": 100,  # PLP is NORMAL in PYCR1
        },
        "pathway_position": {
            "enzyme": "PYCR1 (Pyrroline-5-Carboxylate Reductase 1)",
            "step": "Step 2 of proline synthesis (FINAL step: P5C → Proline)",
            "upstream": "ALDH18A1/P5CS (Glutamate → P5C, Step 1 of synthesis)",
            "downstream": "L-Proline exported to cytoplasm for protein/collagen synthesis",
            "catabolism_reverse": "PRODH (Proline → P5C, Step 1 catabolism) → ALDH4A1 (P5C → Glu)",
            "position_summary": (
                "PYCR1 is the terminal proline synthesis enzyme. ALDH18A1 (upstream) makes P5C; "
                "PYCR1 (downstream) reduces P5C to proline. PYCR1 LOF blocks the LAST step — "
                "P5C is made by ALDH18A1 but CANNOT be converted to proline."
            ),
        },
        "vs_aldh18a1": {
            "shared": "Both: proline CRITICALLY LOW; cutis laxa; AR inheritance; synthesis-direction failure",
            "PYCR1": "P5C mildly elevated (ALDH18A1 intact, makes P5C normally); ornithine mildly LOW-NORMAL; less severe cutis laxa; cataracts rare",
            "ALDH18A1": "P5C NOT MADE AT ALL (ALDH18A1 blocks entry); ornithine VERY LOW (<30); cutis laxa 95% severe; cataracts 60–75%; citrulline/arginine cascade affected",
            "epilepsy": "PYCR1: 40–55% seizures; 15–25% DRE. ALDH18A1: 50–65% seizures; 20–30% DRE. Both: NO B6 response (PLP normal in both)",
        },
        "vs_aldh4a1": {
            "shared": "Both involve proline pathway; P5C is involved in both",
            "PYCR1": "Proline CRITICALLY LOW (synthesis exit block); P5C mildly elevated; PLP NORMAL; cutis laxa; no B6 response",
            "ALDH4A1": "Proline MARKEDLY ELEVATED (catabolism block); P5C PATHOGNOMONIC HIGH (>80 µmol/L); PLP LOW (secondary B6 deficiency); B6 PARTIAL response 30–50%",
        },
    }


def get_breakdown():
    rng = _rng()

    # Variants
    variants = [
        {"variant": "p.Arg119Pro", "domain": "Substrate-Binding Pocket", "freq_pct": 22,
         "phenotype": "Severe AR", "note": "Most common worldwide; substrate coordination loss; null activity"},
        {"variant": "p.Leu175Pro", "domain": "Dimer/Pentamer Interface", "freq_pct": 18,
         "phenotype": "Severe AR", "note": "Pentameric ring destabilization; dominant-negative-like effect on PYCR2"},
        {"variant": "p.Ala225Val", "domain": "NADPH-Binding Domain", "freq_pct": 15,
         "phenotype": "Moderate AR", "note": "Partial NADPH binding; residual 15–20% activity"},
        {"variant": "p.Gly93Ser", "domain": "N-terminal Active Site Loop", "freq_pct": 12,
         "phenotype": "Severe AR", "note": "Active site loop disruption; null"},
        {"variant": "p.Arg251Cys", "domain": "C-terminal Mitochondrial Targeting", "freq_pct": 10,
         "phenotype": "Moderate AR", "note": "Mito-import partially impaired; cytoplasmic mislocalization"},
        {"variant": "c.IVS5+1G>A", "domain": "Splice Null", "freq_pct": 9,
         "phenotype": "Severe AR", "note": "Exon 5 skip → frameshift → NMD; null allele"},
        {"variant": "p.Tyr156Cys", "domain": "Cofactor Coordination", "freq_pct": 8,
         "phenotype": "Moderate-Severe AR", "note": "Cofactor binding disrupted; 5–10% residual activity"},
        {"variant": "p.Val277Ile", "domain": "Peripheral/mild", "freq_pct": 6,
         "phenotype": "Mild AR (Attenuated)", "note": "Mild structural perturbation; 30–40% residual activity"},
    ]

    # Seizure types
    seizure_types = [
        {"type": "Infantile Spasms / West Syndrome", "pct_in_seizure_pts": 30,
         "note": "Most common seizure type in PYCR1; ACTH/Vigabatrin first-line; hypsarrhythmia on EEG"},
        {"type": "Multifocal Clonic", "pct_in_seizure_pts": 25,
         "note": "Proline depletion + mitochondrial ROS → cortical hyperexcitability; multifocal origin"},
        {"type": "GTCS (Generalized Tonic-Clonic)", "pct_in_seizure_pts": 20,
         "note": "Classic-infantile phenotype; secondary generalization from focal onset"},
        {"type": "Focal Cortical", "pct_in_seizure_pts": 15,
         "note": "Structural cortical lesions (gyral simplification, PVL); location-dependent semiology"},
        {"type": "Absence-like", "pct_in_seizure_pts": 10,
         "note": "Mild AR phenotype; absence-like episodes with brief staring; milder mechanism"},
    ]

    # Biomarkers
    biomarkers = [
        {"name": "Proline (plasma)", "mean": 37.8, "unit": "µmol/L",
         "normal_range": "100–260 µmol/L",
         "significance": "CRITICALLY LOW — primary biomarker of PYCR1 deficiency; synthesis exit block"},
        {"name": "P5C / Delta-1-Pyrroline-5-Carboxylate (plasma)", "mean": 9.2, "unit": "µmol/L",
         "normal_range": "<6 µmol/L",
         "significance": "MILDLY ELEVATED — substrate backup as ALDH18A1 intact but PYCR1 cannot reduce P5C; NOT pathognomonic unlike ALDH4A1 (>80 µmol/L)"},
        {"name": "PLP / Pyridoxal-5-phosphate (plasma)", "mean": 62.4, "unit": "nmol/L",
         "normal_range": "35–110 nmol/L",
         "significance": "NORMAL — no P5C-PLP inactivation (P5C too low for significant Schiff base); B6 has NO indication"},
        {"name": "Ornithine (plasma)", "mean": 41.3, "unit": "µmol/L",
         "normal_range": "40–140 µmol/L",
         "significance": "MILDLY LOW-NORMAL — OAT reversal partially compensates; contrast ALDH18A1 (<30 µmol/L)"},
        {"name": "Hydroxyproline (urine)", "mean": None, "unit": "LOW",
         "normal_range": "Normal present (collagen turnover)",
         "significance": "LOW — proline deficiency → less substrate for prolyl-hydroxylase → less hydroxyproline in collagen"},
        {"name": "alpha-AASA (urine)", "mean": None, "unit": "µmol/mol creatinine",
         "normal_range": "<1 mmol/mol creatinine",
         "significance": "NORMAL — KEY NEGATIVE vs ALDH7A1/PDE (antiquitin), where MARKEDLY ELEVATED (pathognomonic)"},
        {"name": "Pipecolic acid (plasma)", "mean": None, "unit": "µmol/L",
         "normal_range": "<5 µmol/L",
         "significance": "NORMAL — KEY NEGATIVE vs ALDH7A1/PDE where elevated"},
        {"name": "Total homocysteine (tHcy)", "mean": 9.8, "unit": "µmol/L",
         "normal_range": "<15 µmol/L",
         "significance": "NORMAL — methionine/folate cycle intact; KEY vs CBS/MTHFR/MTR/MTRR/AHCY"},
        {"name": "MMA (methylmalonic acid urine)", "mean": None, "unit": "mmol/mol creatinine",
         "normal_range": "<5 mmol/mol",
         "significance": "NORMAL — propionate/B12 pathway intact; KEY vs MMUT/MMAA/MMAB/cblC"},
        {"name": "Ammonia (plasma)", "mean": None, "unit": "µmol/L",
         "normal_range": "<50 µmol/L",
         "significance": "NORMAL–BORDERLINE — urea cycle mildly stressed (ornithine mildly low); less severe than ALDH18A1"},
    ]

    # Treatments
    treatments = [
        {"treatment": "L-Proline supplementation", "level": "Level A",
         "dose": "100–400 mg/kg/day in divided doses",
         "mechanism": "Directly replaces deficient proline; corrects hypoProlinemia; improves collagen synthesis",
         "monitoring": "Plasma proline target 150–300 µmol/L; avoid excess (>500 → PRODH-type effects)",
         "contraindication": "None (treatment of choice)"},
        {"treatment": "L-Ornithine", "level": "Level B",
         "dose": "50–150 mg/kg/day if plasma ornithine <40 µmol/L",
         "mechanism": "Replenishes mild ornithine depletion; supports polyamine synthesis; urea cycle",
         "monitoring": "Plasma ornithine target 60–120 µmol/L",
         "contraindication": "None"},
        {"treatment": "Vitamin C (ascorbate)", "level": "Level B",
         "dose": "50–200 mg/day (standard antioxidant dose)",
         "mechanism": "Cofactor for prolyl-4-hydroxylase and prolyl-3-hydroxylase (collagen synthesis); amplifies proline utilization into stable collagen",
         "monitoring": "Clinical connective tissue improvement",
         "contraindication": "None"},
        {"treatment": "ACTH / Vigabatrin", "level": "Level A (infantile spasms)",
         "dose": "Standard IS protocol; ACTH 150 IU/m²/day or Vigabatrin 100–150 mg/kg/day",
         "mechanism": "West syndrome/infantile spasms first-line; same as other epileptic spasms causes",
         "monitoring": "EEG hypsarrhythmia resolution; Vigabatrin: visual field monitoring",
         "contraindication": "None specific to PYCR1"},
        {"treatment": "Levetiracetam (LEV)", "level": "Level B",
         "dose": "20–60 mg/kg/day",
         "mechanism": "Broad-spectrum AED; no metabolic interaction with proline pathway",
         "monitoring": "Behavioral side effects (irritability) in children",
         "contraindication": "None"},
        {"treatment": "Pyridoxine / B6", "level": "NOT INDICATED",
         "dose": "N/A",
         "mechanism": "PLP is NORMAL in PYCR1 — no P5C-PLP inactivation (contrast ALDH4A1); B6 trial will show NO response",
         "monitoring": "N/A",
         "contraindication": "Unnecessary; risk of peripheral neuropathy if high-dose without indication"},
        {"treatment": "Protein restriction", "level": "ABSOLUTE CONTRAINDICATION",
         "dose": "Avoid",
         "mechanism": "Restricting protein worsens proline substrate availability — OPPOSITE indication to PRODH/ALDH4A1",
         "monitoring": "N/A",
         "contraindication": "Do NOT restrict dietary protein in PYCR1 (same as ALDH18A1)"},
        {"treatment": "Valproate (VPA)", "level": "MODERATE RISK",
         "dose": "Avoid if ornithine low",
         "mechanism": "VPA inhibits PRODH in vitro (raises proline — paradoxically may help) but causes hyperammonemia via OTC/ornithine interactions; ornithine low in PYCR1 → hyperammonemia risk amplified",
         "monitoring": "Ammonia + ornithine if VPA used; consider alternative AED",
         "contraindication": "Use with caution; monitor ammonia closely"},
    ]

    # Clinical features
    clinical_features = [
        {"feature": "Cutis laxa (loose/wrinkled skin)", "pct": 90,
         "note": "Hallmark at birth; generalized; lax/redundant skin; improves with proline supplementation"},
        {"feature": "Intellectual disability", "pct": 80,
         "note": "75% severe, 20% moderate in severe AR; mild AR often milder IDD"},
        {"feature": "Feeding difficulties / failure to thrive", "pct": 70,
         "note": "Hypotonia + poor suck; proline deficiency in esophageal connective tissue"},
        {"feature": "Facial dysmorphism", "pct": 85,
         "note": "Frontal bossing, hypertelorism, downslanting palpebral fissures, beaked/pinched nose, large ears"},
        {"feature": "Seizures", "pct": 48,
         "note": "40–55% overall; infantile spasms most common; 15–25% drug-resistant"},
        {"feature": "White matter abnormalities (MRI)", "pct": 35,
         "note": "Periventricular leukomalacia (25%), delayed myelination; PYCR1 NADPH redox disruption"},
        {"feature": "Simplified gyral pattern (MRI)", "pct": 30,
         "note": "Cortical gyration reduced in severe AR; structural basis for focal seizures"},
        {"feature": "Microcephaly", "pct": 45,
         "note": "Post-natal progressive; secondary to CNS proline + NADPH deficiency"},
        {"feature": "Joint laxity", "pct": 65,
         "note": "Connective tissue collagen failure; less severe than ALDH18A1 (90%)"},
        {"feature": "Thin corpus callosum (MRI)", "pct": 20,
         "note": "Structural brain malformation in severe cases"},
        {"feature": "Strabismus", "pct": 30,
         "note": "Ocular muscle connective tissue + brain stem involvement"},
        {"feature": "Cataracts", "pct": 20,
         "note": "Less common than ALDH18A1 (60–75%); proline less rate-limiting for lens crystallins in PYCR1"},
    ]

    return {
        "variants": variants,
        "seizure_types": seizure_types,
        "biomarkers": biomarkers,
        "treatments": treatments,
        "clinical_features": clinical_features,
    }


def get_definitions():
    return {
        "gene_full_name": "Pyrroline-5-Carboxylate Reductase 1 (PYCR1)",
        "chromosome": "17q25.3",
        "gene_omim": "*179035",
        "disease_omim": "#612940",
        "disease_name": "Autosomal Recessive Cutis Laxa Type 2B (ARCL2B) — PYCR1 Deficiency",
        "inheritance": "Autosomal Recessive — biallelic LOF; no AD cases described",
        "protein": "319 aa; mitochondrial matrix; NADPH-dependent; homopentameric ring structure; Pyrroline-5-Carboxylate Reductase superfamily",
        "reaction": "P5C (Delta-1-Pyrroline-5-Carboxylate) + NADPH + H⁺ → L-Proline + NADP⁺",
        "pathway": "Proline SYNTHESIS — Step 2 (final step): P5C → Proline [PYCR1/2]; Step 1 upstream: Glu → P5C [ALDH18A1/P5CS]",
        "key_terms": {
            "P5C": "Delta-1-Pyrroline-5-Carboxylate — cyclic imino acid in equilibrium with open-chain glutamate-5-semialdehyde (GSA); substrate for PYCR1; product of ALDH18A1 and PRODH",
            "PYCR1": "Pyrroline-5-Carboxylate Reductase 1 — main mitochondrial isoform; converts P5C → proline using NADPH; maintains mitochondrial NADP+/NADPH redox balance",
            "PYCR2": "Pyrroline-5-Carboxylate Reductase 2 — brain-enriched mitochondrial isoform; same reaction; PYCR2 deficiency causes Hypomyelinating Leukoencephalopathy 10 (HLD10, OMIM #616138)",
            "PYCRL": "Pyrroline-5-Carboxylate Reductase-Like — cytoplasmic isoform; PYCRL deficiency causes Cutis Laxa 2C (OMIM #614933)",
            "Cutis Laxa": "Lax, wrinkled, inelastic skin due to defective collagen elastic fibers; hallmark of proline-deficient connective tissue disorders",
            "Hydroxyproline": "Modified amino acid in collagen Gly-Pro-Hyp repeats; requires prolyl hydroxylase (vitamin C cofactor); LOW in PYCR1 due to proline deficiency",
            "ARCL2B": "Autosomal Recessive Cutis Laxa Type 2B — the clinical designation for PYCR1 deficiency",
            "HypoProlinemia": "Critically low plasma proline (<50 µmol/L vs normal 100–260) — diagnostic hallmark of proline synthesis defects (ALDH18A1, PYCR1)",
            "NADPH": "Nicotinamide adenine dinucleotide phosphate (reduced form) — cofactor for PYCR1; its recycling by PYCR1 maintains mitochondrial redox balance",
        },
        "differential_diagnosis": {
            "ALDH18A1 (P5CS Deficiency)": "Proline LOW (same); but P5C NOT MADE (ALDH18A1 blocks entry); ornithine VERY LOW (<30); citrulline/arginine cascade; cataracts 60–75%; OMIM #219150 (AR De Barsy)",
            "ALDH4A1 (HPII, Type II)": "Proline MARKEDLY ELEVATED (opposite); P5C PATHOGNOMONIC HIGH (>80 µmol/L); PLP LOW; B6 partial response; catabolism disorder not synthesis",
            "PRODH (HPI, Type I)": "Proline ELEVATED (opposite — catabolism blocked step 1); P5C NORMAL; PLP NORMAL; psychiatric features; synthesis intact",
            "ALDH7A1/PDE": "alpha-AASA HIGH (pathognomonic); pipecolic HIGH; lysine catabolism (not proline); B6 highly responsive (>85%)",
            "PYCR2 Deficiency (HLD10)": "Same enzyme family, brain-enriched; hypomyelination PROMINENT; more seizures (~60–70%); OMIM #616138; chromosome 1q42.12",
            "PYCRL Deficiency (ARCL2C)": "Cytoplasmic PYCR isoform; less severe cutis laxa; OMIM #614933; chromosome 8q24.12",
        },
        "treatment_summary": {
            "first_line": "L-Proline supplementation 100–400 mg/kg/day — Level A; PRIMARY treatment",
            "adjunct": "L-Ornithine (if low) Level B; Vitamin C Level B (collagen cofactor)",
            "aeds": "ACTH/Vigabatrin Level A (infantile spasms); LEV Level B",
            "contraindicated": "Protein restriction (ABSOLUTE CI — worsens proline depletion); Pyridoxine/B6 (NOT indicated — PLP normal; no B6 mechanism)",
            "moderate_risk": "VPA (hyperammonemia risk with ornithine low); B6 antagonists (minor risk, PLP normal)",
        },
        "cohort_note": f"40-patient synthetic cohort (seed {_SEED}). Biochemical parameters and phenotype distribution reflect published PYCR1 literature (Reversade et al. 2009; Baumgartner et al. 2000; Fischer et al. 2012; OMIM #612940). Not real patient data.",
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (first 1000 chars) ===")
    print(json.dumps(get_breakdown(), indent=2)[:1000])
    print("\n=== DEFINITIONS (first 1000 chars) ===")
    print(json.dumps(get_definitions(), indent=2)[:1000])
