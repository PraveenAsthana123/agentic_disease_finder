#!/usr/bin/env python3
"""DLD (Dihydrolipoamide Dehydrogenase, L-protein) Epilepsy Dashboard.

DLD encodes dihydrolipoamide dehydrogenase — the L-protein (E3 subunit) shared among FOUR
mitochondrial alpha-keto acid dehydrogenase complexes:
  1. GCS  — Glycine Cleavage System (L-protein)
  2. PDH  — Pyruvate Dehydrogenase Complex (E3 subunit)
  3. αKGDH — Alpha-Ketoglutarate Dehydrogenase Complex (E3 subunit)
  4. BCKDH — Branched-Chain Keto Acid Dehydrogenase Complex (E3 subunit)

DLD: 509 amino acids; FAD-dependent homodimer; 7q31.1; mitochondrial matrix.
DLD regenerates the oxidised lipoyl groups on E2 subunits of each complex by:
  dihydrolipoamide + NAD⁺ → lipoamide + NADH   (FAD as intermediate, recycled by NAD⁺)

DLD LOF — COMBINED FOUR-COMPLEX BLOCK (UNIQUE vs GLDC/AMT/GCSH pure NKH):
  All four complexes simultaneously impaired → combined metabolic catastrophe:
  • PDH block  → Lactate↑↑, Pyruvate↑↑, lactate:pyruvate ratio normal/increased
  • αKGDH block → 2-Hydroxyglutarate↑, Alpha-ketoglutarate↑
  • BCKDH block → BCAA↑ (Leu/Ile/Val), Branched-chain keto acids↑
  • GCS block  → Glycine↑ (mild-moderate; less than GLDC/AMT/GCSH pure NKH)
  Net: combined lactic acidosis + organic aciduria + partial NKH-like glycine elevation.

CRITICAL DISTINCTION FROM PURE NKH (GLDC/AMT/GCSH):
  DLD-deficiency is NOT pure NKH. Glycine elevation is:
    (a) Less severe: CSF glycine typically 20–120 µM (vs 100–2,000 µM in GLDC/AMT/GCSH NKH)
    (b) CSF:plasma glycine ratio often <0.08 (below NKH diagnostic threshold)
    (c) ACCOMPANIED by elevated lactate, BCAA, alpha-KG — biochemically distinct
  Gene panel mandatory: DLD must be sequenced alongside GLDC, AMT, GCSH.

PATHOPHYSIOLOGY — COMBINED MITOCHONDRIAL DYSFUNCTION + PARTIAL GLYCINE ACCUMULATION:
  1. GCS partial block: glycine mildly elevated → some NMDAr GluN1 co-agonist site occupancy
  2. PDH block: lactate/pyruvate → ATP deficit, cytotoxic oedema, Leigh-like lesions
  3. αKGDH block: TCA cycle impaired at 2-oxoglutarate step → succinyl-CoA deficit
  4. BCKDH block: BCAA/keto acid accumulation → neurotoxicity (similar to classic MSUD)
  EEG: epileptiform activity driven by combined mechanisms (ATP failure + partial glycine ↑)

EPIDEMIOLOGY:
  ~200 DLD-deficiency cases worldwide 2026. OMIM gene *238331; disease #246900.
  AR biallelic LOF; 7q31.1. Rare founder allele in Ashkenazi Jewish (p.Gly194Cys).
  Phenotype: Neonatal encephalopathy (~40%) · Infantile-onset (~35%) · Late-onset/episodic (~20%) · Atypical (~5%).

KEY VARIANTS:
  p.Gly194Cys: Ashkenazi Jewish founder allele → atypical/mild, late-onset, episodic crises
  p.Arg182His: severe neonatal encephalopathy, lactic acidosis, high early mortality
  p.Ile12Thr: MSUD-3 phenotype (BCAA-dominant)
  p.Glu375Lys: liver involvement, moderate
  Null alleles (frameshift/nonsense): severe neonatal, near-universal mortality without Tx

TREATMENT:
  1. Low-BCAA / low-protein diet (Level A): reduce BCKDH substrate load; leucine-free formula
  2. Thiamine / B1 (Level A): PDH + αKGDH cofactor (100–300 mg/day); some responders
  3. Riboflavin / B2 (Level A): FAD cofactor for DLD itself; 50–200 mg/day
  4. L-Carnitine (Level A): secondary depletion from organic acid accumulation
  5. Emergency glucose + bicarbonate (Level A): catabolism/crisis → IV dextrose + NaHCO₃
  6. LEV (Level B): first-line AED; no BCAA or organic acid interaction
  7. KD (Level C): CAUTION — fasting component hazardous; modified Atkins only with close monitoring

HIGH RISK DRUGS:
  VPA: ABSOLUTE CI — triple mechanism: (1) hepatotoxicity (mitochondrial disease context, POLG1-
    like risk), (2) carnitine depletion (worsens secondary carnitine deficit already present),
    (3) inhibits already-compromised BCKDH and mitochondrial β-oxidation. Equivalent to
    POLG1-related mitochondrial epilepsy CI.
  Fasting/Catabolism: EXTREME HAZARD — surges all four substrate streams simultaneously;
    BCAA/keto acids + lactate + 2-HG spike. IV dextrose (GIR ≥6–8 mg/kg/min) MANDATORY in crisis.
  Phenobarbital: CAUTION — GABA-A sedation may mask metabolic crisis; monitor lactate.
  Ketogenic Diet (standard): HIGH RISK — fat-based energy promotes ketosis; ketone bodies compete
    with already-impaired mitochondrial substrate handling; modified Atkins only under metabolic control.

GENETICS SUMMARY:
  Gene: DLD · 7q31.1 · 509 aa · FAD-dependent homodimer · mitochondrial matrix
  AR biallelic LOF · OMIM gene *238331 · disease #246900
  ~80+ pathogenic variants; p.Gly194Cys is the ONLY known significant founder allele
"""

import random
from datetime import datetime

SEED = 20260822
rng = random.Random(SEED)

def _rng_choice(items): return rng.choice(items)
def _rng_int(lo, hi): return rng.randint(lo, hi)
def _rng_float(lo, hi, dec=2): return round(rng.uniform(lo, hi), dec)

# Colour: dark teal — mitochondrial complex / FAD / combined BCAA+GCS block
COLOUR = "#00695c"  # dark teal — L-protein / FAD / four-complex hub


# ── cohort ──────────────────────────────────────────────────────────────────
PHENOTYPES = [
    {"label": "Neonatal Encephalopathy", "pct": 40, "color": "#b71c1c"},
    {"label": "Infantile-Onset (Progressive)", "pct": 35, "color": "#e65100"},
    {"label": "Late-Onset / Episodic", "pct": 20, "color": "#f9a825"},
    {"label": "Atypical", "pct": 5, "color": "#558b2f"},
]

SEIZURE_TYPES = [
    {"type": "Myoclonic", "pct": 62},
    {"type": "Tonic", "pct": 55},
    {"type": "Focal with secondary generalisation", "pct": 48},
    {"type": "Epileptic Spasms (IS)", "pct": 38},
    {"type": "Absence-like", "pct": 22},
    {"type": "Status epilepticus", "pct": 30},
]

TRIGGER_TYPES = [
    {"trigger": "Febrile illness / infection", "pct": 88},
    {"trigger": "Fasting / prolonged catabolism", "pct": 82},
    {"trigger": "High protein intake (esp. BCAA)", "pct": 74},
    {"trigger": "Surgery / anaesthesia", "pct": 58},
    {"trigger": "Physiological stress (dehydration)", "pct": 52},
    {"trigger": "Intercurrent GI illness", "pct": 44},
]

TREATMENTS = [
    {"drug": "Low-BCAA diet (leucine-free formula)", "level": "A", "response_pct": 72, "color": "#00695c"},
    {"drug": "Thiamine (B1)", "level": "A", "response_pct": 55, "color": "#00838f"},
    {"drug": "Riboflavin (B2, FAD cofactor for DLD)", "level": "A", "response_pct": 48, "color": "#0277bd"},
    {"drug": "L-Carnitine (secondary depletion)", "level": "A", "response_pct": 65, "color": "#2e7d32"},
    {"drug": "IV Glucose + NaHCO₃ (crisis)", "level": "A", "response_pct": 80, "color": "#1565c0"},
    {"drug": "Levetiracetam (LEV)", "level": "B", "response_pct": 58, "color": "#4a148c"},
    {"drug": "Modified Atkins (caution)", "level": "C", "response_pct": 32, "color": "#827717"},
]

HIGH_RISK_DRUGS = [
    {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI", "mechanism": "Hepatotoxicity + carnitine depletion + BCKDH inhibition (triple CI, POLG1-like risk)"},
    {"drug": "Fasting / catabolism", "risk": "EXTREME HAZARD", "mechanism": "Surges all four substrate streams; IV glucose MANDATORY (GIR ≥6 mg/kg/min)"},
    {"drug": "Standard Ketogenic Diet", "risk": "HIGH RISK", "mechanism": "Fat-based energy → ketosis → competes with impaired mitochondrial substrate handling"},
    {"drug": "Phenobarbital", "risk": "CAUTION", "mechanism": "Sedation may mask metabolic crisis; monitor lactate closely"},
]

KEY_VARIANTS = [
    {"variant": "p.Gly194Cys", "effect": "Ashkenazi founder allele → atypical/mild; late-onset episodic crises", "phenotype": "Atypical/Late"},
    {"variant": "p.Arg182His", "effect": "FAD-binding domain; severe neonatal; lactic acidosis; high early mortality", "phenotype": "Neonatal"},
    {"variant": "p.Ile12Thr", "effect": "MSUD-3 phenotype; BCAA-dominant biochemistry", "phenotype": "Infantile"},
    {"variant": "p.Glu375Lys", "effect": "Moderate; liver involvement; partial DLD activity retained", "phenotype": "Infantile"},
    {"variant": "Null (frameshift/nonsense)", "effect": "Complete DLD absence; severe neonatal; near-universal mortality without Tx", "phenotype": "Neonatal"},
]

BIOMARKERS = [
    {"name": "Plasma Lactate", "normal": "<2.2 mmol/L", "dld_range": "3–20 mmol/L", "significance": "PDH block; acute crisis indicator"},
    {"name": "Plasma Pyruvate", "normal": "<0.17 mmol/L", "dld_range": "0.3–1.5 mmol/L", "significance": "PDH block; L:P ratio may be normal"},
    {"name": "Plasma BCAA (Leu/Ile/Val)", "normal": "Leu <130 µmol/L", "dld_range": "Leu 150–900 µmol/L", "significance": "BCKDH block; Leu most toxic"},
    {"name": "Plasma Alpha-ketoglutarate", "normal": "<20 µmol/L", "dld_range": "40–400 µmol/L", "significance": "αKGDH block; TCA impairment"},
    {"name": "Plasma Glycine", "normal": "150–420 µmol/L", "dld_range": "350–900 µmol/L", "significance": "GCS partial block; less severe than GLDC/AMT/GCSH"},
    {"name": "CSF:plasma glycine ratio", "normal": "<0.02", "dld_range": "0.02–0.07 (below NKH threshold)", "significance": "Below 0.08 NKH threshold — key DLD vs pure NKH distinction"},
    {"name": "Urine organic acids", "normal": "—", "dld_range": "2-HG↑, branched-chain keto acids↑, lactate↑", "significance": "Combined pattern diagnostic"},
]


def _make_patient(i):
    phenotype = rng.choices(
        [p["label"] for p in PHENOTYPES],
        weights=[p["pct"] for p in PHENOTYPES]
    )[0]

    age_map = {
        "Neonatal Encephalopathy": _rng_int(0, 7),
        "Infantile-Onset (Progressive)": _rng_int(1, 6),
        "Late-Onset / Episodic": _rng_int(5, 18),
        "Atypical": _rng_int(2, 20),
    }
    age_mo = age_map[phenotype]

    plasma_lactate = _rng_float(3.5, 18.0)
    bcaa_leu = _rng_int(180, 850)
    glycine_plasma = _rng_int(350, 880)
    csf_glycine = _rng_int(20, 115)
    csf_plasma_ratio = round(csf_glycine / glycine_plasma, 3) if glycine_plasma else 0

    vpa_avoided = rng.random() > 0.08
    on_thiamine = rng.random() > 0.35
    on_riboflavin = rng.random() > 0.40
    on_carnitine = rng.random() > 0.20
    on_lev = rng.random() > 0.40

    dre = phenotype in ("Neonatal Encephalopathy", "Infantile-Onset (Progressive)") and rng.random() > 0.55

    return {
        "id": f"DLD-{i:03d}",
        "phenotype": phenotype,
        "onset_age_months": age_mo,
        "plasma_lactate_mmol": plasma_lactate,
        "plasma_leu_umol": bcaa_leu,
        "plasma_glycine_umol": glycine_plasma,
        "csf_glycine_umol": csf_glycine,
        "csf_plasma_ratio": csf_plasma_ratio,
        "vpa_avoided": vpa_avoided,
        "on_thiamine": on_thiamine,
        "on_riboflavin": on_riboflavin,
        "on_carnitine": on_carnitine,
        "on_lev": on_lev,
        "dre": dre,
        "variant": rng.choice([v["variant"] for v in KEY_VARIANTS]),
    }


PATIENTS = [_make_patient(i) for i in range(1, 41)]


def get_overview():
    n = len(PATIENTS)
    avg_lactate = round(sum(p["plasma_lactate_mmol"] for p in PATIENTS) / n, 2)
    avg_glycine = round(sum(p["plasma_glycine_umol"] for p in PATIENTS) / n)
    avg_leu = round(sum(p["plasma_leu_umol"] for p in PATIENTS) / n)
    avg_csf_ratio = round(sum(p["csf_plasma_ratio"] for p in PATIENTS) / n, 3)
    dre_n = sum(1 for p in PATIENTS if p["dre"])
    vpa_avoided_n = sum(1 for p in PATIENTS if p["vpa_avoided"])

    return {
        "dashboard": "DLD Epilepsy (Dihydrolipoamide Dehydrogenase Deficiency / L-protein / Four-Complex Mitochondrial Block)",
        "gene": "DLD",
        "protein": "L-protein (E3 subunit — shared among GCS / PDH / αKGDH / BCKDH)",
        "omim_gene": "*238331",
        "omim_disease": "#246900",
        "locus": "7q31.1",
        "aa_length": 509,
        "cofactor": "FAD (flavin adenine dinucleotide)",
        "mechanism": "Homodimer; oxidises dihydrolipoamide via FAD; electrons transferred to NAD⁺ → NADH; shared E3 of all four alpha-keto acid DH complexes",
        "inheritance": "AR biallelic LOF",
        "prevalence": "~200 cases worldwide (2026); rarer than GLDC/AMT/GCSH individually",
        "founder_allele": "p.Gly194Cys (Ashkenazi Jewish → atypical/late-onset phenotype)",
        "dld_vs_nkh_key_distinction": "DLD deficiency is NOT pure NKH — combined four-complex block; CSF:plasma glycine ratio typically <0.08 (below NKH threshold); accompanied by lactate↑, BCAA↑, αKG↑",
        "kpis": {
            "cohort_n": n,
            "avg_plasma_lactate_mmol": avg_lactate,
            "avg_plasma_glycine_umol": avg_glycine,
            "avg_plasma_leu_umol": avg_leu,
            "avg_csf_plasma_glycine_ratio": avg_csf_ratio,
            "csf_ratio_below_nkh_threshold_008_pct": round(sum(1 for p in PATIENTS if p["csf_plasma_ratio"] < 0.08) / n * 100, 1),
            "dre_pct": round(dre_n / n * 100, 1),
            "vpa_avoided_pct": round(vpa_avoided_n / n * 100, 1),
        },
        "phenotype_distribution": PHENOTYPES,
        "four_complex_block": [
            {"complex": "GCS (Glycine Cleavage System)", "result": "Glycine↑ (mild-moderate)", "diagnostic": "Glycine, CSF:plasma ratio"},
            {"complex": "PDH (Pyruvate Dehydrogenase)", "result": "Lactate↑↑, Pyruvate↑↑", "diagnostic": "Plasma lactate, L:P ratio"},
            {"complex": "αKGDH (Alpha-KG Dehydrogenase)", "result": "2-HG↑, α-KG↑", "diagnostic": "Urine organic acids"},
            {"complex": "BCKDH (Branched-Chain KA-DH)", "result": "BCAA↑, keto acids↑", "diagnostic": "Plasma amino acids (Leu/Ile/Val)"},
        ],
        "high_risk_drugs": HIGH_RISK_DRUGS,
        "generated": datetime.utcnow().isoformat() + "Z",
    }


def get_breakdown():
    return {
        "dashboard": "DLD Breakdown",
        "cohort_n": len(PATIENTS),
        "phenotype_distribution": PHENOTYPES,
        "seizure_types": SEIZURE_TYPES,
        "trigger_types": TRIGGER_TYPES,
        "treatments": TREATMENTS,
        "key_variants": KEY_VARIANTS,
        "biomarkers": BIOMARKERS,
        "patients_sample": PATIENTS[:10],
        "generated": datetime.utcnow().isoformat() + "Z",
    }


def get_definitions():
    return {
        "dashboard": "DLD Definitions",
        "gene_card": {
            "gene": "DLD",
            "full_name": "Dihydrolipoamide Dehydrogenase",
            "alias": "L-protein (GCS) / E3 subunit (PDH, αKGDH, BCKDH)",
            "locus": "7q31.1",
            "aa_length": 509,
            "structure": "FAD-dependent homodimer; lipoamide-binding + FAD/NAD-binding domains",
            "cofactor": "FAD (oxidised by NAD⁺ to regenerate lipoamide)",
            "complexes_shared": ["GCS (L-protein)", "PDH (E3)", "αKGDH (E3)", "BCKDH (E3)"],
            "reaction": "Dihydrolipoamide + NAD⁺ → Lipoamide + NADH + H⁺",
            "inheritance": "AR biallelic LOF",
            "omim_gene": "*238331",
            "omim_disease": "#246900",
            "other_name": "Maple Syrup Urine Disease Type III (MSUD3) / LA-Hyperglycemia / DLD deficiency",
        },
        "key_concepts": [
            {"term": "DLD / L-protein", "definition": "E3 subunit shared by GCS, PDH, αKGDH, and BCKDH; oxidises dihydrolipoyl arms via FAD; regenerates lipoamide for next catalytic cycle"},
            {"term": "Combined four-complex block", "definition": "DLD LOF simultaneously impairs all four complexes → combined lactic acidosis (PDH) + organic aciduria (αKGDH) + BCAA elevation (BCKDH) + partial glycine elevation (GCS)"},
            {"term": "Partial GCS block (DLD vs GLDC/AMT/GCSH)", "definition": "GCS L-protein (DLD) is needed to regenerate oxidised H-protein, but H-protein cycling is partially maintained; glycine elevation is less severe than in pure NKH (GLDC/AMT/GCSH)"},
            {"term": "CSF:plasma glycine ratio in DLD", "definition": "Typically 0.02–0.07 (below the 0.08 NKH diagnostic threshold) — KEY distinguishing feature from pure NKH"},
            {"term": "L:P ratio (Lactate:Pyruvate)", "definition": "Elevated in PDH block; DLD: L:P may be normal or slightly elevated (lactate and pyruvate both rise together); helps distinguish PDH block from complex I defect"},
            {"term": "p.Gly194Cys Ashkenazi founder", "definition": "Most common DLD variant globally; Ashkenazi Jewish carrier frequency ~1:100; produces atypical/mild DLD deficiency with late-onset episodic crises; some residual DLD activity retained"},
            {"term": "Episodic metabolic crisis", "definition": "Acute decompensation triggered by illness/fasting/surgery; surges in lactate, BCAA, and alpha-KG simultaneously; medical emergency requiring IV glucose (GIR ≥6–8 mg/kg/min)"},
            {"term": "VPA absolute CI in DLD", "definition": "Triple mechanism: (1) mitochondrial hepatotoxicity (same risk as POLG1 context), (2) carnitine depletion (worsens already-depleted secondary carnitine), (3) inhibits BCKDH — never use in DLD deficiency"},
            {"term": "Thiamine responsiveness", "definition": "~40–55% of DLD patients show partial biochemical response to high-dose thiamine (B1) — always trial before assuming no response"},
            {"term": "OMIM #246900", "definition": "DLD deficiency (also listed as MSUD Type III / lipoamide dehydrogenase deficiency); AR; 7q31.1; DLD gene *238331"},
        ],
        "thresholds": [
            {"parameter": "Plasma lactate (normal)", "threshold": "<2.2 mmol/L", "dld_range": "3–20 mmol/L (crisis >10)"},
            {"parameter": "Plasma pyruvate (normal)", "threshold": "<0.17 mmol/L", "dld_range": "0.3–1.5 mmol/L"},
            {"parameter": "Plasma leucine (normal)", "threshold": "<130 µmol/L", "dld_range": "150–900 µmol/L"},
            {"parameter": "Plasma glycine (normal)", "threshold": "150–420 µmol/L", "dld_range": "350–900 µmol/L (less than GLDC/AMT/GCSH)"},
            {"parameter": "CSF:plasma glycine ratio", "threshold": "Normal <0.02; NKH ≥0.08", "dld_range": "0.02–0.07 — below NKH threshold (key distinction)"},
            {"parameter": "IV glucose infusion rate (crisis)", "threshold": "GIR ≥6–8 mg/kg/min", "dld_range": "Target euglycemia + suppress catabolism"},
        ],
        "differential_diagnosis": [
            {"condition": "GLDC-NKH", "distinguishing": "Pure NKH; CSF:plasma glycine ≥0.08; NO lactate/BCAA elevation; GLDC gene"},
            {"condition": "AMT-NKH", "distinguishing": "Pure NKH; CSF:plasma glycine ≥0.08; 5,10-methyleneTHF deficit; AMT gene"},
            {"condition": "GCSH-NKH", "distinguishing": "Pure NKH; CSF:plasma glycine ≥0.08; GCSH gene; rarest NKH"},
            {"condition": "Classic MSUD (BCKDHA/B/DBT)", "distinguishing": "BCAA-dominant; no GCS block; no significant glycine elevation; maple syrup odour"},
            {"condition": "PDH deficiency", "distinguishing": "Lactate-dominant; no BCAA/glycine elevation; PDHA1/PDHB/PDHX/DLAT mutations"},
            {"condition": "Propionic acidemia", "distinguishing": "Propionyl-CoA accumulation; methylcitrate↑; no DLD gene"},
        ],
        "generated": datetime.utcnow().isoformat() + "Z",
    }
