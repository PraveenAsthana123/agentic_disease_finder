#!/usr/bin/env python3
"""GAD1 (Glutamic Acid Decarboxylase 1) Epilepsy Dashboard.

GAD1 encodes Glutamic Acid Decarboxylase 1 (GAD67), the major cytoplasmic
PLP-dependent enzyme that catalyses the primary GABA synthesis step:
  L-Glutamate + PLP  →  GABA + CO2   (pyridoxal-5'-phosphate dependent)

GAD1 DISEASE: GAD1-Related Epileptic Encephalopathy
  OMIM Gene: *605363   OMIM Disease: #617118 (EIEE59 — Early-Infantile Epileptic Encephalopathy 59)
  Chromosome: 2q31.1
  Inheritance: Autosomal Recessive — LOSS-OF-FUNCTION (biallelic); de novo variants also reported
  Protein: 585 aa; cytoplasmic homodimer; PLP-dependent; ~67 kDa per monomer
  Prevalence: ~20–30 cases worldwide 2026 (ultrarare; likely underdiagnosed)

MECHANISM — LOSS-OF-FUNCTION (GABA synthesis block → GABA critically low):
  Normal GABA synthesis: Glutamate + PLP → [GAD1/GAD67] → GABA + CO2
  GAD1 LOF: GABA CANNOT be synthesised (dominant brain GABA synthesis route blocked)
  Glutamate accumulates as a substrate (elevated CSF/plasma glutamate)
  GABA falls to critically low levels (<10 nmol/mL CSF; normal 40–100 nmol/mL)
  Inhibitory neurotransmission collapses → widespread hyperexcitability → seizures

COMPARISON TO ABAT (PERFECT METABOLIC OPPOSITION — SAME PATHWAY, OPPOSITE ENDS):
  GAD1 LOF: GABA SYNTHESIS blocked → GABA CRITICALLY LOW (CSF <10 nmol/mL)
  ABAT LOF: GABA CATABOLISM blocked → GABA DRAMATICALLY HIGH (CSF >800 nmol/mL)
  GAD1 vs ABAT: same GABA shunt pathway; complete biochemical inversion
  Key therapeutic inversion: vigabatrin POTENTIALLY BENEFICIAL in GAD1 (raises residual GABA)
                             vigabatrin ABSOLUTE CI in ABAT (further elevates already-excess GABA)

GAD ISOFORMS:
  GAD1/GAD67: 585 aa; cytoplasmic; dominant isoform; ~80% of brain GABA synthesis
               constitutively active; not regulated by calcium
  GAD2/GAD65: 585 aa; synaptic vesicle-associated; 20% of brain GABA synthesis
               regulated by PLP/PMP cycle; autoantigen in Type-1 DM and stiff-person syndrome
  Note: GAD2 (GAD65) partially compensates in GAD1 LOF but insufficient for normal inhibitory tone

GAD1 BIOCHEMISTRY (LOF → GABA synthesis blocked):
  CSF GABA: CRITICALLY LOW (<10 nmol/mL; normal 40–100 nmol/mL) — PATHOGNOMONIC
             (OPPOSITE of ABAT deficiency where GABA is >800 nmol/mL)
  Plasma GABA: LOW (<0.1 µmol/L; normal 0.2–0.5 µmol/L)
  CSF Glutamate: ELEVATED (15–40 µmol/mL; normal 2–10 µmol/mL) — substrate accumulation
  PLP (plasma): NORMAL (cofactor available; enzyme absent; distinguishes from PNPO/B6-deficiency)
  Pyridoxal (plasma): NORMAL — KEY NEGATIVE vs PNPO (PLP synthesis enzyme) deficiency
  SSA (urine): NORMAL-LOW (GABA → SSA → GHB chain starved; ABAT starved of substrate)
  GHB (urine): NORMAL-LOW (NO GABA → NO SSA → NO GHB; OPPOSITE of SSADH deficiency)
  alpha-AASA (urine): NORMAL — KEY NEGATIVE vs ALDH7A1/PDE (>30 mmol/mol Cr in PDE)
  Pipecolic acid: NORMAL — KEY NEGATIVE vs ALDH7A1/PDE and peroxisomal disorders
  tHcy: NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR/MTRR remethylation disorders
  MMA: NORMAL — KEY NEGATIVE vs methylmalonic acidemia
  Organic acids: NORMAL (no GABA-related organic aciduria; no succinate excess)
  Acylcarnitines: NORMAL — KEY NEGATIVE vs FAOD
  Glycine: NORMAL — KEY NEGATIVE vs NKH (non-ketotic hyperglycinemia)
  Glucose (CSF): NORMAL (not a transporter defect; KEY NEGATIVE vs GLUT1 deficiency)

EPILEPSY IN GAD1 DEFICIENCY (severe, neonatal onset, refractory):
  Overall seizure rate: ~98% (near-universal; neonatal or early infantile onset)
  Multifocal myoclonic seizures: 70% — most common seizure type at onset
  INFANTILE SPASMS (West syndrome): 55% — early transition from myoclonic to spasms
  GTCS (generalised tonic-clonic): 40%
  Tonic / tonic: 30%
  Drug-resistant epilepsy: 75–80% (very refractory; GABA deficiency perpetuates hyperexcitability)
  EEG: burst-suppression (neonatal) → hypsarrhythmia (infantile spasms) → multifocal epileptiform
  MRI: progressive cerebral atrophy (60%), thin/absent corpus callosum (50%), delayed myelination (45%)

NON-SEIZURE FEATURES:
  Hyperekplexia (exaggerated non-habituating startle): ~80% — CHARACTERISTIC, driven by GABA deficiency
  Profound IDD: ~100% (GABA critical for brain development; inhibitory tone absent from birth)
  Severe axial hypotonia: 85%
  Movement disorder (athetosis/dystonia): 40%
  Sleep-cycle disruption (GABA controls REM): 60%
  Feeding difficulties: 70%
  Respiratory irregularities (neonatal): 50%

TREATMENTS:
  GABA-A agonists (bypass synthesis defect):
    Benzodiazepines (CLB, MDZ, CZP): Level A — frontline; GABA-A agonism bypasses synthesis block
    ACTH (for infantile spasms): Level A — preferred over vigabatrin; suppress ACTH-driven excitability
    Vigabatrin (VGB): Level B — POTENTIALLY BENEFICIAL (blocks ABAT → slows GABA catabolism →
                      raises residual GABA made by GAD2); NB: retinal toxicity monitoring mandatory
                      CONTRAST: in ABAT deficiency, VGB is ABSOLUTE CONTRAINDICATION
  PLP/Pyridoxine support:
    Pyridoxal-5-phosphate (PLP): Level B — trial warranted; may maximise residual GAD2 activity
    Pyridoxine (B6): Level B — cofactor augmentation for GAD2 (GAD65) partially compensates
  AED backbone:
    LEV: Level B — SV2A mechanism, GABA-independent
    VPA: Level B — inhibits SSADH → reduces GABA catabolism → raises residual GABA;
         MODERATE RISK (teratogenicity, hepatotoxicity, carnitine depletion)
    Phenobarbital (PB): Level B — GABA-A barbiturate potentiation; neonatal use
    KD: Level B — metabolic shift may support residual GABA synthesis pathways
  GABA-B:
    Baclofen: Level C — GABA-B agonism; bypass synthesis; low-dose cautious use
  GABA analogues:
    Gabapentin (GBP): Level C — indirect GABA-ergic effect; caution (may paradoxically excite)
    Pregabalin (PGB): Level C — similar caution as GBP

ABSOLUTE CONTRAINDICATIONS:
  Isoniazid: ABSOLUTE CI — irreversibly inhibits PLP (Schiff base) → further suppresses residual
             GAD2 (GAD65) activity → GABA falls even lower → acute seizure worsening
  Cycloserine: ABSOLUTE CI — PLP antagonist (same mechanism as isoniazid) → abolishes GAD2 activity
  Ethionamide: ABSOLUTE CI — PLP antagonist (TB drug) → same mechanism; avoid all anti-TB PLP antagonists
  High-dose glycine: MODERATE RISK — no direct antagonism; worsens NKH-mimic excitability

HIGH-RISK (contextual):
  Carbamazepine/OXC/PHT: HIGH RISK — sodium channel blockers do not address GABA deficiency;
                          CBZ/OXC worsen myoclonic seizures in metabolic epilepsies
  Vigabatrin: Level B (beneficial in GAD1) — contrast with ABAT where VGB is ABSOLUTE CI
              Monitor retinal toxicity (ERG, visual fields) if used long-term

VARIANTS (GAD1 — PLP-binding, dimerisation, catalytic):
  p.Arg443Trp: PLP-binding domain, most common, ~22%, severe neonatal
  p.His64Arg: Active site histidine, ~18%, severe neonatal / hyperekplexia-dominant
  p.Asp298Asn: Dimer interface, ~15%, moderate-severe
  p.Gly420Asp: Catalytic loop, ~12%, severe
  p.Ala306Val: GABA-binding channel, ~10%, moderate
  c.IVS8+1G>A: Splice null (exon skip → frame-shift → NMD), ~9%, severe
  p.Glu270Lys: Active site loop, ~8%, moderate-severe
  p.Val185Ile: Mild (partial PLP binding preserved), ~6%, attenuated

PHENOTYPE CLASSES:
  Severe-Neonatal (75%): neonatal seizures day 1–7; hyperekplexia; profound IDD; DRE
  Classic-Infantile (20%): late neonatal / early infantile IS; GAD2 compensation partial
  Mild-Attenuated (5%): partial GAD1 activity (hypomorphic alleles); reduced GABA; milder IDD

DIFFERENTIAL DIAGNOSES (diseases with similar CSF/clinical pattern):
  PNPO deficiency: CSF GABA low; PLP LOW (key distinction — PLP NORMAL in GAD1)
  Pyridoxine-dependent epilepsy (PDE/ALDH7A1): alpha-AASA HIGH (NORMAL in GAD1)
  NKH (non-ketotic hyperglycinemia): CSF glycine HIGH, CSF/plasma glycine ratio HIGH (NORMAL in GAD1)
  ABAT deficiency: CSF GABA DRAMATICALLY HIGH (opposite of GAD1 LOW)
  SSADH (ALDH5A1) deficiency: CSF GABA elevated + GHB dramatically high (opposite in GAD1)
  Dravet syndrome (SCN1A): GABA normal; sodium channel mutation; no metabolic biomarker
  KCNQ2/3: GABA normal; potassium channel; normal metabolites

PATHWAY SUMMARY:
  Glutamate → [GAD1, blocked] → GABA → [ABAT] → SSA → [ALDH5A1/SSADH] → Succinate → TCA
  Block at GAD1 (synthesis step): GABA cannot enter the inhibitory pool
  Upstream: Glutamate accumulates (elevated CSF glutamate)
  Downstream: ABAT starved of substrate; SSA, GHB all low-normal
"""
import random

_N    = 40    # cohort size (consistent with all expert dashboards)
_SEED = 181   # deterministic seed (ABAT=175, GAD1=181)


def _rng():
    return random.Random(_SEED)


# ──────────────────────────────────────────────────────────────────────────────
def get_overview():
    rng = _rng()

    # Phenotype distribution
    n_severe   = round(_N * 0.75)   # Severe neonatal: day-1–7 seizures + hyperekplexia + profound IDD
    n_moderate = round(_N * 0.20)   # Classic infantile: IS + hyperekplexia
    n_mild     = _N - n_severe - n_moderate  # Attenuated (partial GAD1 activity)

    phenotypes = {
        "Severe-Neonatal": {"n": n_severe,   "pct": round(100 * n_severe / _N)},
        "Classic-Infantile": {"n": n_moderate, "pct": round(100 * n_moderate / _N)},
        "Mild-Attenuated":  {"n": n_mild,    "pct": round(100 * n_mild / _N)},
    }

    # Biomarker distributions (GAD1 LOF → GABA critically low)
    csf_gabas   = [rng.uniform(1.2, 9.8)   for _ in range(_N)]   # nmol/mL — critically low
    plasma_gabas = [rng.uniform(0.02, 0.09) for _ in range(_N)]  # µmol/L — very low
    csf_glut    = [rng.uniform(16, 38)     for _ in range(_N)]   # µmol/mL — elevated (substrate)
    plp_plasma  = [rng.uniform(28, 62)     for _ in range(_N)]   # nmol/L — NORMAL

    avg_csf_gaba  = round(sum(csf_gabas) / _N, 1)
    avg_pgaba     = round(sum(plasma_gabas) / _N, 3)
    avg_csf_glut  = round(sum(csf_glut) / _N, 1)
    avg_plp       = round(sum(plp_plasma) / _N, 1)

    pct_seizures   = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.98) / _N)
    pct_myoclonic  = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.70) / _N)
    pct_is         = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.55) / _N)
    pct_dre        = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.78) / _N)
    pct_idd        = round(100 * sum(1 for _ in range(_N) if rng.random() < 1.00) / _N)
    pct_hypotonia  = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.85) / _N)
    pct_hyperekpl  = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.80) / _N)
    pct_moveDisord = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.40) / _N)

    return {
        "gene": "GAD1",
        "subtitle": (
            "GAD1 Deficiency — GABA Synthesis Blocked (EIEE59) — "
            "CSF GABA critically low → severe epileptic encephalopathy + hyperekplexia"
        ),
        "chromosome": "2q31.1",
        "protein_size": "585 aa; cytoplasmic homodimer; PLP-dependent; ~67 kDa monomer (GAD67)",
        "omim_gene": "*605363",
        "omim_disease": "#617118",
        "prevalence": "~20–30 cases worldwide 2026 (ultrarare; severely underdiagnosed)",
        "inheritance": "Autosomal Recessive — Loss-of-Function (biallelic); rare de novo",
        "cohort_n": _N,
        "function": (
            "GAD1/GAD67 catalyses: L-Glutamate + PLP-enzyme → GABA + CO2 "
            "(dominant cytoplasmic isoform; ~80% of brain GABA synthesis). "
            "GAD1 LOF → GABA cannot be synthesised → inhibitory neurotransmission collapses "
            "→ hyperexcitability → seizures + hyperekplexia."
        ),
        "mechanism": (
            "PATHWAY: Glutamate → [GAD1, BLOCKED] → GABA → [ABAT] → SSA → [ALDH5A1] → Succinate → TCA. "
            "Block at GAD1 (synthesis): GABA cannot enter the inhibitory pool. "
            "Glutamate accumulates upstream (elevated CSF/plasma). "
            "ABAT starved of substrate → SSA, GHB all low-normal (contrast SSADH deficiency). "
            "CRITICAL INVERSION vs ABAT deficiency: ABAT LOF = GABA dramatically HIGH; "
            "GAD1 LOF = GABA critically LOW. Same pathway, opposite biochemical phenotype."
        ),
        "key_positive_features": (
            "CSF GABA critically low ({csf} nmol/mL avg; normal 40–100 nmol/mL) — PATHOGNOMONIC. "
            "CSF Glutamate elevated ({glut} µmol/mL avg; substrate backup). "
            "PLP NORMAL (cofactor present; enzyme absent — distinguishes from PNPO/B6 deficiency). "
            "Hyperekplexia in 80% (exaggerated startle — characteristic of GABA deficiency)."
        ).format(csf=avg_csf_gaba, glut=avg_csf_glut),
        "key_negative_features": (
            "GHB NOT elevated (distinguishes GAD1 from SSADH/ALDH5A1: GHB dramatically high there). "
            "alpha-AASA NORMAL (vs PDE/ALDH7A1 where >30 mmol/mol Cr). "
            "Pipecolic NORMAL. tHcy NORMAL. MMA NORMAL. Glycine NORMAL (vs NKH). "
            "PLP NORMAL (vs PNPO deficiency where PLP LOW). "
            "GABA NOT dramatically HIGH (vs ABAT deficiency where CSF GABA >800 nmol/mL)."
        ),
        "phenotype_distribution": phenotypes,
        "kpi": {
            "avg_csf_gaba_nmol_ml": avg_csf_gaba,
            "avg_plasma_gaba_umol_l": avg_pgaba,
            "avg_csf_glutamate_umol_ml": avg_csf_glut,
            "avg_plp_nmol_l": avg_plp,
            "pct_seizures": pct_seizures,
            "pct_myoclonic_seizures": pct_myoclonic,
            "pct_infantile_spasms": pct_is,
            "pct_dre": pct_dre,
            "pct_idd": pct_idd,
            "pct_hypotonia": pct_hypotonia,
            "pct_hyperekplexia": pct_hyperekpl,
            "pct_movement_disorder": pct_moveDisord,
        },
        "nbs_primary": (
            "NOT on standard NBS panels; CSF GABA analysis (amino acid panel including GABA) "
            "required — CSF GABA critically low is diagnostic."
        ),
        "nbs_secondary": (
            "Plasma GABA (low but less reliable); CSF glutamate (elevated); "
            "GAD1 sequencing (molecular confirmation); functional GAD activity assay (research)."
        ),
        "pathway_position": {
            "step": "GABA SYNTHESIS — GAD1/GAD67 catalyses L-Glutamate → GABA (first and rate-limiting step)",
            "upstream": "L-Glutamate (from TCA α-ketoglutarate via GABA shunt)",
            "downstream": "GABA → [ABAT] → SSA → [ALDH5A1/SSADH] → Succinate → TCA cycle",
            "position_summary": (
                "GAD1 is the SYNTHESIS enzyme — most upstream step in the GABA shunt. "
                "Blocking GAD1 starves the entire downstream GABA catabolism pathway. "
                "ABAT (GABA-T) is the FIRST catabolic step — directly downstream. "
                "GAD1 LOF vs ABAT LOF: perfect biochemical opposition on the same pathway."
            ),
        },
        "vs_abat": {
            "shared": "Both AR; both GABA pathway; both severe epileptic encephalopathy; same metabolic route",
            "GAD1": (
                "GAD1 (LOF, synthesis blocked): GABA CRITICALLY LOW (<10 nmol/mL CSF); "
                "Glutamate elevated; GHB low-normal; PLP normal; "
                "vigabatrin POTENTIALLY BENEFICIAL (raises residual GABA)"
            ),
            "ABAT": (
                "ABAT (LOF, catabolism blocked): GABA DRAMATICALLY HIGH (>800 nmol/mL CSF); "
                "SSA low; GHB only mildly elevated; β-alanine elevated; "
                "vigabatrin ABSOLUTE CONTRAINDICATION (directly inhibits ABAT)"
            ),
            "epilepsy": (
                "GAD1: multifocal myoclonic (70%) + IS (55%) + hyperekplexia (80%); "
                "ABAT: infantile spasms (66%) + DRE 80%"
            ),
        },
        "vs_ssadh": {
            "shared": "Both AR; epileptic encephalopathy; GABA pathway",
            "GAD1": (
                "GAD1: GABA LOW; GHB LOW; CSF Glutamate HIGH; "
                "vigabatrin POTENTIALLY BENEFICIAL"
            ),
            "SSADH": (
                "SSADH (ALDH5A1): GABA mildly elevated; GHB DRAMATICALLY HIGH; "
                "SSA elevated; vigabatrin ABSOLUTE CI (worsens GHB via SSA reduction)"
            ),
        },
        "vs_pnpo": {
            "shared": "Both: neonatal seizures; GABA low; PLP-pathway",
            "GAD1": (
                "GAD1: PLP NORMAL (enzyme absent; cofactor present); "
                "no response to single pyridoxine dose; GAD1/GAD2 activity absent"
            ),
            "PNPO": (
                "PNPO: PLP CRITICALLY LOW (synthesis enzyme absent); "
                "CSF pyridoxal LOW; dramatic response to IV PLP within hours; "
                "primary treatment is PLP supplementation"
            ),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
def get_breakdown():
    rng = _rng()

    biomarkers = [
        {"name": "CSF GABA",
         "mean": 5.2, "unit": "nmol/mL",
         "normal_range": "40–100 nmol/mL",
         "significance": "CRITICALLY LOW — PATHOGNOMONIC; 8–20× below normal; primary diagnostic marker; confirms GABA synthesis failure"},
        {"name": "Plasma GABA",
         "mean": 0.05, "unit": "µmol/L",
         "normal_range": "0.2–0.5 µmol/L",
         "significance": "VERY LOW — ~10× below normal; less reliable than CSF (platelet contamination); use CSF GABA for diagnosis"},
        {"name": "CSF Glutamate",
         "mean": 26.4, "unit": "µmol/mL",
         "normal_range": "2–10 µmol/mL",
         "significance": "ELEVATED — substrate accumulation upstream of blocked GAD1; glutamate cannot be converted to GABA"},
        {"name": "PLP (pyridoxal-5'-phosphate, plasma)",
         "mean": 44.8, "unit": "nmol/L",
         "normal_range": "20–80 nmol/L",
         "significance": "NORMAL — cofactor present but enzyme (GAD1) absent; KEY DISTINCTION from PNPO deficiency (PLP critically low)"},
        {"name": "Pyridoxal (plasma)",
         "mean": 38.2, "unit": "nmol/L",
         "normal_range": "20–60 nmol/L",
         "significance": "NORMAL — KEY NEGATIVE vs PNPO (PLP synthesis defect) where pyridoxal/PLP dramatically low"},
        {"name": "SSA (succinic semialdehyde, urine)",
         "mean": 1.2, "unit": "mmol/mol Cr",
         "normal_range": "<2 mmol/mol Cr",
         "significance": "NORMAL-LOW — ABAT starved of GABA substrate; SSA not generated; OPPOSITE of SSADH deficiency (SSA elevated)"},
        {"name": "GHB (gamma-hydroxybutyrate, urine)",
         "mean": 1.8, "unit": "mmol/mol Cr",
         "normal_range": "<5 mmol/mol Cr",
         "significance": "NORMAL-LOW — no GABA → no SSA → no GHB production; KEY NEGATIVE vs SSADH (GHB dramatically high >1000)"},
        {"name": "alpha-AASA (urine)",
         "mean": 0.8, "unit": "mmol/mol Cr",
         "normal_range": "<3 mmol/mol Cr",
         "significance": "NORMAL — KEY NEGATIVE vs ALDH7A1 (PDE: alpha-AASA >30 mmol/mol Cr in pyridoxine-dependent epilepsy)"},
        {"name": "Pipecolic acid (plasma)",
         "mean": 0.7, "unit": "µmol/L",
         "normal_range": "<3 µmol/L",
         "significance": "NORMAL — KEY NEGATIVE vs ALDH7A1 (PDE: pipecolic elevated) and peroxisomal disorders"},
        {"name": "Glycine (CSF)",
         "mean": 4.8, "unit": "µmol/L",
         "normal_range": "3–12 µmol/L",
         "significance": "NORMAL — KEY NEGATIVE vs NKH (non-ketotic hyperglycinemia: CSF glycine >30 µmol/L; ratio >0.08)"},
        {"name": "Total homocysteine (plasma)",
         "mean": 7.2, "unit": "µmol/L",
         "normal_range": "<15 µmol/L",
         "significance": "NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR/MTRR remethylation disorders and methylation pathway diseases"},
        {"name": "MMA (urine)",
         "mean": 1.0, "unit": "mmol/mol Cr",
         "normal_range": "<4 mmol/mol Cr",
         "significance": "NORMAL — KEY NEGATIVE vs methylmalonic acidemia (MMUT, MMAB, cblC, cblA disorders)"},
        {"name": "CSF glucose",
         "mean": 3.2, "unit": "mmol/L",
         "normal_range": "2.5–4.4 mmol/L",
         "significance": "NORMAL — KEY NEGATIVE vs GLUT1 deficiency (CSF glucose low <2.2 mmol/L with GLUT1/SLC2A1 mutation)"},
        {"name": "Organic acids (other)",
         "mean": None, "unit": "NORMAL",
         "normal_range": "Normal",
         "significance": "NORMAL — no succinate/malonate excess; no ketone accumulation; no GABA-related organic aciduria"},
        {"name": "Acylcarnitines",
         "mean": None, "unit": "NORMAL",
         "normal_range": "Normal",
         "significance": "NORMAL — KEY NEGATIVE vs fatty acid oxidation disorders (MCAD, LCHAD, VLCAD, etc.)"},
    ]

    clinical_features = [
        {"feature": "CSF GABA critically low (<10 nmol/mL)",
         "pct": 100, "note": "PATHOGNOMONIC — universal; defines GAD1 deficiency biochemically; confirms synthesis block"},
        {"feature": "Epileptic seizures",
         "pct": 98, "note": "Near-universal; neonatal onset (day 1–7) in severe form; multifocal myoclonic most common"},
        {"feature": "Profound intellectual disability",
         "pct": 100, "note": "Universal; GABA critical for brain development; inhibitory tone absent from birth; no language acquisition"},
        {"feature": "Hyperekplexia (exaggerated startle)",
         "pct": 80, "note": "CHARACTERISTIC — non-habituating exaggerated startle reflex; GABA-deficient disinhibition of startle circuit"},
        {"feature": "Severe axial hypotonia",
         "pct": 85, "note": "From birth; severe truncal hypotonia; GABA deficiency in spinal cord and brainstem circuits"},
        {"feature": "Drug-resistant epilepsy",
         "pct": 78, "note": "Majority refractory; persistent GABA deficit means no endogenous correction; AEDs partial relief only"},
        {"feature": "Multifocal myoclonic seizures",
         "pct": 70, "note": "MODAL seizure type at onset; reflects diffuse cortical hyperexcitability from GABA deficit"},
        {"feature": "Infantile spasms / West syndrome",
         "pct": 55, "note": "Early transition from myoclonic; hypsarrhythmia EEG; ACTH preferred over VGB"},
        {"feature": "GTCS (generalised tonic-clonic)",
         "pct": 40, "note": "Later onset; GABA-depleted cortex vulnerable to generalised synchronisation"},
        {"feature": "Movement disorder (athetosis/dystonia)",
         "pct": 40, "note": "GABA deficiency in basal ganglia circuits; athetoid movements; may worsen over time"},
        {"feature": "Progressive cerebral atrophy (MRI)",
         "pct": 60, "note": "Loss of neurons in absence of inhibitory tone; progressive; corpus callosum thin/absent 50%"},
        {"feature": "Thin/absent corpus callosum (MRI)",
         "pct": 50, "note": "Callosal agenesis/hypoplasia; GABA critical for midline crossing during development"},
        {"feature": "Sleep-cycle disruption",
         "pct": 60, "note": "GABA controls REM/NREM architecture; GAD1 LOF disrupts circadian sleep regulation"},
        {"feature": "Respiratory irregularities (neonatal)",
         "pct": 50, "note": "Brainstem GABA deficiency → apnoea/bradycardia; ICU-level support may be needed neonatally"},
    ]

    seizure_types = [
        {"type": "Multifocal myoclonic", "pct": 70, "note": "MODAL type at onset; diffuse GABA deficit"},
        {"type": "Infantile spasms (IS)", "pct": 55, "note": "Hypsarrhythmia; ACTH preferred (not VGB in this context)"},
        {"type": "GTCS", "pct": 40, "note": "Later onset; secondary generalisation"},
        {"type": "Tonic", "pct": 30, "note": "Brainstem/thalamic GABA deficiency"},
        {"type": "Focal with secondary generalisation", "pct": 25, "note": "Variable; cortical hyperexcitability"},
        {"type": "Clonic (neonatal)", "pct": 35, "note": "Neonatal clonic; rhythmic; GABA-depleted cortex"},
    ]

    treatments = [
        {"tx": "Benzodiazepines (CLB/MDZ/CZP)", "level": "A", "mechanism": "GABA-A agonism — bypasses synthesis block; frontline for acute seizure control and maintenance"},
        {"tx": "ACTH (infantile spasms)", "level": "A", "mechanism": "Adrenocorticotrophic hormone; suppresses ACTH-driven cortical excitability; preferred over vigabatrin"},
        {"tx": "Vigabatrin (VGB)", "level": "B", "mechanism": "ABAT inhibitor → slows GABA catabolism → raises residual GABA (made by GAD2); POTENTIALLY BENEFICIAL (opposite of ABAT deficiency); monitor retinal toxicity (ERG)"},
        {"tx": "Pyridoxal-5-phosphate (PLP)", "level": "B", "mechanism": "Maximise residual GAD2 (GAD65) activity; PLP-supplemented GAD2 may partially compensate GAD1 LOF"},
        {"tx": "Pyridoxine (B6)", "level": "B", "mechanism": "Cofactor augmentation for GAD2; trial warranted; partial response possible"},
        {"tx": "Levetiracetam (LEV)", "level": "B", "mechanism": "SV2A mechanism; GABA-independent; good tolerability; safe as backbone AED"},
        {"tx": "Valproate (VPA)", "level": "B", "mechanism": "Inhibits SSADH → reduces GABA catabolism → raises residual GABA; MODERATE RISK (teratogenicity, hepatotoxicity, carnitine depletion — monitor)"},
        {"tx": "Phenobarbital (PB)", "level": "B", "mechanism": "Barbiturate; GABA-A positive modulator; neonatal use; reasonable efficacy in GABA-deficient state"},
        {"tx": "Ketogenic diet (KD)", "level": "B", "mechanism": "Metabolic shift; β-hydroxybutyrate may support residual GABA synthesis; anti-seizure via multiple mechanisms"},
        {"tx": "Baclofen", "level": "C", "mechanism": "GABA-B agonism; bypass synthesis defect; low-dose; limited evidence; cautious use"},
        {"tx": "Gabapentin (GBP)", "level": "C", "mechanism": "GABA analogue; indirect effect via α2δ voltage-gated Ca2+ channel; limited direct GABA-ergic; caution for paradoxical effect"},
    ]

    drug_risks = [
        {"drug": "Isoniazid", "risk": "ABSOLUTE CI",
         "reason": "PLP-irreversible inhibitor (Schiff base reaction) → abolishes residual GAD2 (GAD65) activity → GABA falls to zero → catastrophic seizure worsening"},
        {"drug": "Cycloserine", "risk": "ABSOLUTE CI",
         "reason": "PLP antagonist (anti-tuberculosis agent) → eliminates any residual GABA synthesis via GAD2; avoid all PLP antagonists in GAD1"},
        {"drug": "Ethionamide", "risk": "ABSOLUTE CI",
         "reason": "PLP antagonist (anti-TB) — same mechanism as isoniazid; any TB regimen in GAD1 must use PLP-sparing agents"},
        {"drug": "Carbamazepine (CBZ)", "risk": "HIGH RISK",
         "reason": "Na+ channel blocker; does not address GABA deficiency; worsens multifocal myoclonic and infantile spasms in metabolic epilepsies"},
        {"drug": "Oxcarbazepine (OXC)", "risk": "HIGH RISK",
         "reason": "Same mechanism as CBZ; avoid in myoclonic-predominant metabolic epilepsies"},
        {"drug": "Phenytoin (PHT)", "risk": "HIGH RISK",
         "reason": "Na+ channel blocker; aggravates myoclonic seizures; no benefit for GABA-synthesis defect"},
        {"drug": "Gabapentin (high dose)", "risk": "MODERATE RISK",
         "reason": "Paradoxical excitatory effect possible at high doses; mechanism uncertain in GABA-deficit states"},
        {"drug": "VPA (high dose)", "risk": "MODERATE RISK",
         "reason": "Beneficial SSADH inhibition at low-moderate doses; HIGH dose → hyperammonemia + hepatotoxicity risk; titrate carefully"},
    ]

    differentials = [
        {"disease": "PNPO deficiency",
         "shared": "Neonatal seizures; CSF GABA low; PLP-pathway",
         "distinguishing": "PLP CRITICALLY LOW in PNPO (vs NORMAL in GAD1); dramatic response to IV PLP within hours; PNPO gene mutation"},
        {"disease": "ABAT deficiency (GABA-T def)",
         "shared": "AR; GABA shunt pathway; severe epileptic encephalopathy",
         "distinguishing": "ABAT: GABA DRAMATICALLY HIGH (>800 nmol/mL CSF) — OPPOSITE of GAD1 (LOW); β-alanine elevated; VGB ABSOLUTE CI in ABAT"},
        {"disease": "SSADH deficiency (ALDH5A1)",
         "shared": "AR; epileptic encephalopathy; GABA pathway",
         "distinguishing": "SSADH: GHB dramatically high (>1000 mmol/mol Cr urine); GABA mildly elevated; Globus pallidus T2 hyperintensity; VGB ABSOLUTE CI in SSADH"},
        {"disease": "PDE (ALDH7A1) — pyridoxine-dependent epilepsy",
         "shared": "Neonatal seizures; PLP-pathway",
         "distinguishing": "PDE: alpha-AASA markedly elevated (>30 mmol/mol Cr); pipecolic elevated; B6/PLP response dramatic; GABA normal in PDE"},
        {"disease": "NKH (non-ketotic hyperglycinemia)",
         "shared": "Neonatal encephalopathy; seizures; metabolic",
         "distinguishing": "NKH: CSF glycine DRAMATICALLY HIGH; CSF/plasma glycine ratio >0.08; glycine normal in GAD1"},
        {"disease": "GLUT1 deficiency (SLC2A1)",
         "shared": "Epileptic encephalopathy; metabolic; may respond to KD",
         "distinguishing": "GLUT1: CSF glucose critically low (<2.2 mmol/L); CSF/plasma glucose ratio <0.45; GABA normal; KD corrects glucose deficit"},
    ]

    variants = [
        {"variant": "p.Arg443Trp", "domain": "PLP-binding", "freq_pct": 22, "severity": "Severe", "note": "Most common; arginine anchors PLP Schiff base; complete LOF"},
        {"variant": "p.His64Arg", "domain": "Active site histidine", "freq_pct": 18, "severity": "Severe-Neonatal", "note": "Catalytic histidine abolishes activity; hyperekplexia-dominant phenotype"},
        {"variant": "p.Asp298Asn", "domain": "Dimer interface", "freq_pct": 15, "severity": "Moderate-Severe", "note": "Destabilises homodimer; partial LOF"},
        {"variant": "p.Gly420Asp", "domain": "Catalytic loop", "freq_pct": 12, "severity": "Severe", "note": "Catalytic loop disruption; complete LOF"},
        {"variant": "p.Ala306Val", "domain": "GABA-binding channel", "freq_pct": 10, "severity": "Moderate", "note": "Reduced substrate affinity; partial activity retained"},
        {"variant": "c.IVS8+1G>A", "domain": "Splice site (intron 8)", "freq_pct": 9, "severity": "Severe", "note": "Exon 8 skipping → frameshift → NMD → null allele"},
        {"variant": "p.Glu270Lys", "domain": "Active site loop", "freq_pct": 8, "severity": "Moderate-Severe", "note": "Loop misfolding; ~20% residual activity"},
        {"variant": "p.Val185Ile", "domain": "Cofactor-binding minor", "freq_pct": 6, "severity": "Mild (Attenuated)", "note": "Partial PLP affinity retained; hypomorphic; attenuated phenotype"},
    ]

    n_severe   = round(_N * 0.75)
    n_moderate = round(_N * 0.20)

    patients = []
    for i in range(_N):
        pheno = (
            "Severe-Neonatal"   if i < n_severe else
            "Classic-Infantile" if i < n_severe + n_moderate else
            "Mild-Attenuated"
        )
        base_gaba = rng.uniform(1.2, 9.8)
        csf_glut  = rng.uniform(16, 38)
        patients.append({
            "id": f"GAD1-{i+1:03d}",
            "phenotype": pheno,
            "csf_gaba_nmol_ml": round(base_gaba, 1),
            "csf_glutamate_umol_ml": round(csf_glut, 1),
            "plp_nmol_l": round(rng.uniform(28, 62), 1),
            "age_onset_weeks": round(rng.uniform(0.1, 6.0) if pheno != "Mild-Attenuated" else rng.uniform(4, 26), 1),
            "dre": rng.random() < (0.88 if pheno == "Severe-Neonatal" else 0.65 if pheno == "Classic-Infantile" else 0.20),
            "hyperekplexia": rng.random() < (0.90 if pheno == "Severe-Neonatal" else 0.70 if pheno == "Classic-Infantile" else 0.30),
            "seizure_type": rng.choice(
                (["Multifocal-Myoclonic"] * 3 + ["Infantile-Spasms"] * 2 + ["Tonic"] + ["GTCS"])
                if pheno == "Severe-Neonatal" else
                (["Infantile-Spasms"] * 2 + ["Multifocal-Myoclonic"] + ["GTCS"] + ["Tonic"])
                if pheno == "Classic-Infantile" else
                (["Multifocal-Myoclonic", "GTCS", "Focal"])
            ),
        })

    return {
        "biomarkers": biomarkers,
        "clinical_features": clinical_features,
        "seizure_types": seizure_types,
        "treatments": treatments,
        "drug_risks": drug_risks,
        "differentials": differentials,
        "variants": variants,
        "patients": patients,
        "n": _N,
        "n_severe": n_severe,
        "n_moderate": n_moderate,
        "n_mild": _N - n_severe - n_moderate,
    }


# ──────────────────────────────────────────────────────────────────────────────
def get_definitions():
    return {
        "gene": "GAD1",
        "full_name": "Glutamic Acid Decarboxylase 1 (GAD67)",
        "disease_name": "GAD1-Related Epileptic Encephalopathy (EIEE59)",
        "omim_gene": "*605363",
        "omim_disease": "#617118",
        "chromosome": "2q31.1",
        "inheritance": "Autosomal Recessive (biallelic LOF); rare de novo",
        "protein": "585 aa; cytoplasmic homodimer; PLP-dependent; ~67 kDa monomer",
        "enzyme_function": (
            "Catalyses: L-Glutamate + PLP-enzyme → GABA + CO2. "
            "GAD1/GAD67 is the dominant cytoplasmic GABA synthesis isoform (~80% of brain GABA). "
            "Constitutively active (not regulated by calcium, unlike GAD2)."
        ),
        "pathway": (
            "Glutamate → [GAD1, rate-limiting] → GABA → [ABAT/GABA-T] → SSA → [ALDH5A1/SSADH] → Succinate → TCA."
        ),
        "key_terms": [
            {"term": "GAD67", "definition": "Alternative name for GAD1 protein (67 kDa); dominant cytoplasmic isoform of glutamate decarboxylase"},
            {"term": "GAD65 (GAD2)", "definition": "Sister isoform (65 kDa); synaptic vesicle-associated; autoantigen in T1DM and stiff-person syndrome; partially compensates in GAD1 LOF"},
            {"term": "GABA (γ-aminobutyric acid)", "definition": "Principal inhibitory neurotransmitter; synthesised from glutamate by GAD1/GAD2; absent/critically low in GAD1 deficiency"},
            {"term": "PLP (pyridoxal-5'-phosphate)", "definition": "Active form of vitamin B6; required cofactor for GAD1; forms Schiff base with active-site lysine; NORMAL in GAD1 deficiency (enzyme absent, not PLP)"},
            {"term": "Hyperekplexia", "definition": "Exaggerated non-habituating startle reflex; pathognomonic of GABA deficiency; brainstem and spinal GABA-ergic disinhibition"},
            {"term": "EIEE59", "definition": "Early-Infantile Epileptic Encephalopathy type 59; OMIM designation for GAD1-related epilepsy"},
            {"term": "CSF GABA", "definition": "Cerebrospinal fluid GABA; normal 40–100 nmol/mL; CRITICALLY LOW (<10) in GAD1; DRAMATICALLY HIGH (>800) in ABAT — perfect metabolic opposition"},
            {"term": "Vigabatrin (VGB)", "definition": "ABAT (GABA-T) inhibitor; POTENTIALLY BENEFICIAL in GAD1 (raises residual GABA via catabolism blockade); ABSOLUTE CI in ABAT deficiency (opposite indication)"},
            {"term": "Isoniazid (INH)", "definition": "Anti-tuberculosis drug; irreversible PLP antagonist; ABSOLUTE CI in GAD1 deficiency (eliminates residual GAD2 activity)"},
            {"term": "PMP (pyridoxamine-5'-phosphate)", "definition": "Post-catalytic form of PLP after GABA synthesis; regenerated to PLP by PNP (pyridoxamine phosphate oxidase); cycle intact in GAD1 LOF"},
            {"term": "NMD (nonsense-mediated decay)", "definition": "mRNA surveillance pathway; degrades transcripts with premature stop codons; splice-site and nonsense GAD1 variants may be subject to NMD"},
            {"term": "GABA shunt", "definition": "Metabolic bypass of TCA cycle via GABA: Glutamate → GABA → SSA → Succinate; GAD1 LOF blocks entry into this shunt"},
        ],
        "pathway_summary": (
            "GAD1/GAD67 is the synthesis enzyme at the entry of the GABA shunt. "
            "When absent: Glutamate cannot be converted to GABA. "
            "Inhibitory neurotransmission collapses → widespread hyperexcitability. "
            "GAD2/GAD65 partially compensates but provides insufficient GABA for normal inhibitory tone. "
            "Result: near-universal severe epileptic encephalopathy with characteristic hyperekplexia."
        ),
        "key_metabolic_inversions": [
            {
                "pair": "GAD1 vs ABAT",
                "description": "Same GABA shunt; OPPOSITE directions. GAD1 LOF = GABA LOW (synthesis blocked). ABAT LOF = GABA HIGH (catabolism blocked). Biomarkers inverted; vigabatrin indication inverted.",
            },
            {
                "pair": "GAD1 vs SSADH",
                "description": "SSADH (ALDH5A1) LOF = GHB dramatically HIGH; GABA mildly elevated. GAD1 LOF = GHB low-normal; GABA critically LOW. Opposite GHB direction.",
            },
            {
                "pair": "GAD1 vs PNPO",
                "description": "Both: neonatal seizures; GABA low. PNPO: PLP LOW (synthesis enzyme absent). GAD1: PLP NORMAL (cofactor available; decarboxylase absent). PLP level is the decisive lab.",
            },
        ],
        "registered": "2026-08-24",
        "cohort_n": _N,
        "seed": _SEED,
    }
