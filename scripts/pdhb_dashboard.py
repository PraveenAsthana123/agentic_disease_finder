#!/usr/bin/env python3
"""PDHB (Pyruvate Dehydrogenase E1-beta Subunit) Epilepsy Dashboard.

PDHB encodes the E1β subunit of the Pyruvate Dehydrogenase Complex (PDH complex / PDC):
  PDC structure: E1 (E1α₂E1β₂ tetramer, PDHA1+PDHB) + E2 (DLAT, 24-mer) + E3BP (PDHX) + E3 (DLD)
  PDH complex catalyses: Pyruvate + CoA + NAD⁺ → Acetyl-CoA + CO₂ + NADH  (irreversible)
  E1β role (PDHB): structural β-subunit of E1 heterotetramer; E1β does NOT directly bind TPP or pyruvate;
                   instead, E1β is REQUIRED for correct folding, assembly, and stability of E1α.
                   Without functional E1β, the E1α₂β₂ heterotetramer cannot assemble → E1α is degraded
                   → entire E1 component becomes non-functional → PDH complex inoperable.

PDHB LOF — IDENTICAL BIOCHEMICAL PHENOTYPE TO PDHA1:
  No E1β → E1α degraded → E1 heterotetramer cannot form → PDH complex inactive
  Pyruvate accumulates → LDH converts pyruvate → lactate (lactic acidosis)
  IDENTICAL to PDHA1: normal L:P ratio (10–20), elevated alanine, CSF lactate > plasma lactate
  CRITICAL DISTINCTION: PDHB is AUTOSOMAL RECESSIVE (AR, 3p14.3) vs PDHA1 X-linked dominant (Xp22.12)
  No sex bias in PDHB: males and females equally affected
  No X-inactivation variable severity (as seen in PDHA1 females)

CRITICAL BIOCHEMICAL FINGERPRINT (identical to PDHA1 — PDH block fingerprint):
  • Lactate:Pyruvate (L:P) ratio NORMAL (~10–20) — BOTH rise together (same as PDHA1)
  • CSF lactate typically > plasma lactate (brain preferentially affected)
  • CSF pyruvate elevated (>0.17 mmol/L)
  • Plasma alanine elevated (pyruvate transamination surrogate marker)
  • NO BCAA elevation (unlike DLD which blocks BCKDH too)
  • NO elevated 2-hydroxyglutarate (unlike DLD/αKGDH block)
  The ONLY biochemical way to distinguish PDHB from PDHA1: gene panel (PDHA1 vs PDHB sequencing)

AUTOSOMAL RECESSIVE — KEY DIFFERENCE FROM PDHA1:
  PDHB = 3p14.3; Autosomal Recessive (AR)
  Both males and females equally affected (no sex bias, unlike PDHA1 X-linked dominant)
  No X-inactivation mosaicism in females (→ no "mild episodic females" phenotype as in PDHA1)
  Carrier parents: typically unaffected (no haploinsufficiency for AR)
  Consanguinity increases risk; de novo mutations rare in AR disease

STRUCTURAL BRAIN ANOMALIES (less common than PDHA1):
  1. Leigh syndrome — symmetric necrotising lesions basal ganglia + brainstem (~45%) — prominent
  2. Corpus callosum agenesis/dysgenesis (~40%: LESS than PDHA1 ~70%, as no X-linked sex bias)
  3. Ventriculomegaly + cortical atrophy (energy failure during neuronal migration)
  4. Periventricular white matter changes (energy failure during myelination)
  Note: CC agenesis less common in PDHB vs PDHA1 because PDHB females are as severely affected as males
  (no X-inactivation-mediated protection), but the overall population has milder spectrum due to AR genetics

PHENOTYPE CLASSES (PDHB, 2026):
  Leigh Syndrome (infantile BG/brainstem lesions): most common (~40%)
  Severe Neonatal (neonatal lactic acidosis + early death): ~20%
  Childhood Episodic (exercise/illness-triggered lactic acidosis): ~30%
  Mild Subacute/Juvenile (chronic partial enzyme deficiency): ~10%
  ~50–80 PDHB cases worldwide 2026 (ultra-rare); OMIM gene *179060; disease #614111

KEY VARIANTS:
  p.Ile199Thr: most commonly reported (European); affects E1β structural beta-propeller domain;
               disrupts E1α–E1β assembly; infantile Leigh syndrome; moderate–severe
  p.Arg161His: affects E1α–E1β heterodimer interface; moderate; childhood episodic lactic acidosis;
               some residual E1 complex activity (E1α₂β₂ partially assembles)
  p.Leu290Arg: C-terminal structural disruption; misfolding of E1β → rapid degradation; severe Leigh
  p.Glu179Lys: near E1α-binding interface; moderate; episodic childhood phenotype
  Null variants (nonsense / frameshift): complete E1β absence; severe neonatal phenotype; early death
  Large genomic deletions: complete loss; severe neonatal phenotype

TREATMENT (identical indications to PDHA1 — same PDH complex block):
  1. Ketogenic diet (Level A — FIRST LINE): same mechanism as PDHA1; ketones bypass PDH block;
     ketones → Acetyl-CoA via thiolase; most effective intervention for PDHB
  2. Thiamine / B1 (Level A): TPP cofactor for E1α; high-dose trial all patients (100–600 mg/day);
     may stabilise residual E1α activity even when E1β is partially reduced;
     thiamine responsiveness is LESS common in PDHB (E1β not TPP-binding subunit) but still trialled
  3. L-Carnitine (Level B): secondary depletion; supports β-oxidation for ketone production on KD
  4. DCA (Dichloroacetate, Level B): inhibits PDK1/3 → prevents E1α Ser293 phosphorylation;
     if any residual E1 complex activity remains, DCA may increase activity;
     NEUROPATHY risk limits long-term use
  5. LEV (Level B): first-line AED; no metabolic interaction with PDH pathway

HIGH-RISK DRUGS (identical to PDHA1):
  VPA: ABSOLUTE CI — mitochondrial hepatotoxicity + carnitine depletion (same as PDHA1)
  High-carbohydrate diet: EXTREME HAZARD (same as PDHA1 — floods pyruvate)
  Glucose-only IV drip: HIGH RISK (pyruvate load without ketone buffer)

GENETICS SUMMARY:
  Gene: PDHB · 3p14.3 · 359 aa (including mitochondrial targeting sequence) · E1β subunit
  Autosomal Recessive (AR) · OMIM gene *179060 · disease #614111
  ~50–80 pathogenic variants reported; no founder allele; pan-ethnic ultra-rare
  E1β structural role: required for E1α₂β₂ heterotetrameric E1 assembly and E1α stability
"""

import random
from datetime import datetime

SEED = 20260824
rng = random.Random(SEED)

def _rng_choice(items): return rng.choice(items)
def _rng_int(lo, hi): return rng.randint(lo, hi)
def _rng_float(lo, hi, dec=2): return round(rng.uniform(lo, hi), dec)

# Colour: deep red-orange — pyruvate block / lactic acidosis / AR PDH deficiency
COLOUR = "#d84315"  # deep red-orange — E1β structural role / AR PDH complex / Leigh syndrome


# ── cohort ──────────────────────────────────────────────────────────────────
PHENOTYPES = [
    {"label": "Leigh Syndrome (Infantile, BG/Brainstem Lesions)", "pct": 40, "color": "#b71c1c"},
    {"label": "Severe Neonatal (Lactic Acidosis + Early Death)", "pct": 20, "color": "#d84315"},
    {"label": "Childhood Episodic (Exercise/Illness-Triggered LA)", "pct": 30, "color": "#f57f17"},
    {"label": "Mild Subacute/Juvenile (Chronic Partial Deficiency)", "pct": 10, "color": "#558b2f"},
]

SEIZURE_TYPES = [
    {"type": "Focal with secondary generalisation", "pct": 60},
    {"type": "Tonic / tonic-clonic", "pct": 55},
    {"type": "Epileptic Spasms (IS)", "pct": 38},
    {"type": "Myoclonic", "pct": 32},
    {"type": "Absence-like episodic", "pct": 25},
    {"type": "Status epilepticus (febrile / metabolic)", "pct": 20},
]

TRIGGER_TYPES = [
    {"trigger": "Febrile illness / infection", "pct": 80},
    {"trigger": "Carbohydrate-heavy meal / glucose load", "pct": 72},
    {"trigger": "Exercise (exertion-induced lactic acidosis)", "pct": 65},
    {"trigger": "Surgery / anaesthesia (glucose drip)", "pct": 58},
    {"trigger": "Prolonged fasting (unmanaged)", "pct": 50},
    {"trigger": "Intercurrent GI illness + vomiting (missed KD)", "pct": 42},
]

TREATMENTS = [
    {"drug": "Ketogenic Diet (KD)", "level": "A", "response_pct": 76, "color": "#00695c"},
    {"drug": "Thiamine (B1, TPP cofactor for E1α)", "level": "A", "response_pct": 35, "color": "#00838f"},
    {"drug": "L-Carnitine (secondary depletion + KD support)", "level": "B", "response_pct": 58, "color": "#2e7d32"},
    {"drug": "Dichloroacetate (DCA — PDK inhibitor)", "level": "B", "response_pct": 40, "color": "#0277bd"},
    {"drug": "Riboflavin (B2, minimal direct PDH effect)", "level": "C", "response_pct": 15, "color": "#827717"},
    {"drug": "Levetiracetam (LEV)", "level": "B", "response_pct": 52, "color": "#4a148c"},
]

HIGH_RISK_DRUGS = [
    {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI",
     "mechanism": "Mitochondrial hepatotoxicity (POLG1/MELAS-equivalent risk in PDH complex deficiency) + carnitine depletion (impairs β-oxidation needed for KD ketone production; destroys primary treatment mechanism)"},
    {"drug": "High-carbohydrate diet / high-glucose IV", "risk": "EXTREME HAZARD",
     "mechanism": "Floods pyruvate → worsens lactic acidosis; PDHB LOF → E1 complex inactive → pyruvate cannot enter TCA; use balanced glucose or KD-adjusted IV in crisis"},
    {"drug": "Glucose-only IV drip (acute crisis)", "risk": "HIGH RISK",
     "mechanism": "Pyruvate load without ketone buffer → acute decompensation in PDH-deficient patient; prefer saline + carnitine; only use glucose if frank hypoglycaemia + KD maintained"},
    {"drug": "Phenobarbital", "risk": "CAUTION",
     "mechanism": "Respiratory depression + sedation may mask evolving metabolic crisis; monitor lactate and respiratory status closely in all PDH deficiency patients"},
    {"drug": "Standard high-carbohydrate enteral feeding", "risk": "CONTRAINDICATED in KD-dependent patients",
     "mechanism": "Breaks ketosis; re-exposes E1-deficient neurons to pyruvate-only energy substrate; substitute KD-compatible enteral formula"},
]

KEY_VARIANTS = [
    {"variant": "p.Ile199Thr", "effect": "Most common reported; affects E1β structural beta-propeller domain; disrupts E1α–E1β assembly; European; infantile Leigh syndrome; moderate–severe", "phenotype": "Leigh/Infantile"},
    {"variant": "p.Arg161His", "effect": "Affects E1α–E1β heterodimer interface; some residual E1 complex activity; childhood episodic lactic acidosis; moderate", "phenotype": "Episodic/Childhood"},
    {"variant": "p.Leu290Arg", "effect": "C-terminal structural disruption; E1β misfolding → rapid proteasomal degradation; complete E1 complex loss; severe Leigh syndrome", "phenotype": "Leigh/Severe"},
    {"variant": "p.Glu179Lys", "effect": "Near E1α-binding interface; moderate; episodic childhood phenotype; partial E1 assembly possible", "phenotype": "Episodic/Childhood"},
    {"variant": "Null (nonsense/frameshift)", "effect": "Complete E1β absence; E1α₂β₂ cannot assemble; E1α rapidly degraded; severe neonatal lactic acidosis; early death without KD", "phenotype": "Severe Neonatal"},
    {"variant": "Large genomic deletion", "effect": "Complete gene deletion; no E1β protein; severe neonatal phenotype; E1 complex fully absent; pan-ethnic", "phenotype": "Severe Neonatal"},
]

BIOMARKERS = [
    {"name": "Plasma Lactate", "normal": "<2.2 mmol/L", "pdhb_range": "3–22 mmol/L", "significance": "PDH block → pyruvate→lactate via LDH; primary biochemical indicator; same pattern as PDHA1"},
    {"name": "Plasma Pyruvate", "normal": "<0.17 mmol/L", "pdhb_range": "0.3–1.7 mmol/L", "significance": "Accumulates (E1 cannot decarboxylate pyruvate); KEY: L:P ratio normal (both rise equally)"},
    {"name": "L:P ratio (Lactate:Pyruvate)", "normal": "10–20", "pdhb_range": "10–20 (NORMAL — KEY)", "significance": "Normal L:P distinguishes PDH deficiency (PDHB or PDHA1) from Complex I deficiency (L:P >25). GENE PANEL needed to distinguish PDHB vs PDHA1 — identical biochemistry"},
    {"name": "Plasma Alanine", "normal": "150–450 µmol/L", "pdhb_range": "480–1400 µmol/L", "significance": "Pyruvate transamination surrogate; sensitive chronic marker; same in PDHB and PDHA1"},
    {"name": "CSF Lactate", "normal": "<2.1 mmol/L", "pdhb_range": "3–10 mmol/L (often > plasma)", "significance": "Brain preferentially affected by PDH block; CSF > plasma lactate typical of PDH deficiency"},
    {"name": "CSF Pyruvate", "normal": "<0.17 mmol/L", "pdhb_range": "0.3–1.1 mmol/L", "significance": "Elevated CSF pyruvate confirms PDH block at brain level; drawn with simultaneous plasma sample"},
    {"name": "Plasma BCAA (Leu/Ile/Val)", "normal": "Normal", "pdhb_range": "NORMAL — key negative", "significance": "PDHB does NOT block BCKDH (unlike DLD/E3). Normal BCAA distinguishes PDHB from DLD deficiency"},
    {"name": "Plasma 2-hydroxyglutarate", "normal": "Normal", "pdhb_range": "NORMAL — key negative", "significance": "PDHB does NOT block αKGDH (unlike DLD/E3). Normal 2-HG distinguishes PDHB from DLD deficiency"},
    {"name": "MRI Brain", "normal": "Normal", "pdhb_range": "Leigh syndrome (BG/brainstem); CC agenesis/dysgenesis (less than PDHA1 ~40% vs 70%)", "significance": "Leigh syndrome most prominent in PDHB; CC agenesis less frequent than PDHA1 (no X-linked sex bias)"},
    {"name": "PDH enzyme activity (fibroblasts)", "normal": ">1.0 nmol/min/mg", "pdhb_range": "<0.3 nmol/min/mg (severe); 0.3–0.7 (moderate)", "significance": "Confirmatory; fibroblasts preferred; molecular testing (PDHA1 + PDHB panel) mandatory to gene-assign"},
]

STRUCTURAL_ANOMALIES = [
    {"anomaly": "Leigh syndrome (BG + brainstem symmetric lesions)", "frequency_pct": 65, "significance": "Most prominent PDHB structural finding; T2 hyperintense periaqueductal grey, putamen, caudate; energy-failure pattern; more common in PDHB vs PDHA1"},
    {"anomaly": "Corpus callosum agenesis/dysgenesis", "frequency_pct": 40, "significance": "Less frequent than PDHA1 (~70%); occurs in PDHB but not as hallmark; splenium affected first; indicates early prenatal energy failure"},
    {"anomaly": "Ventriculomegaly", "frequency_pct": 42, "significance": "Secondary to CC hypoplasia and cortical atrophy from energy failure during development"},
    {"anomaly": "Periventricular white matter T2 changes", "frequency_pct": 38, "significance": "Myelination requires high energy; PDH block → dysmyelination; T2 hyperintensity periventricular; distinguishes from PDHA1 CC-dominant pattern"},
    {"anomaly": "Cortical atrophy / simplified gyration", "frequency_pct": 22, "significance": "Neuronal energy failure during migration; not true lissencephaly"},
    {"anomaly": "Brainstem hypoplasia", "frequency_pct": 18, "significance": "Occurs in Leigh syndrome variants; periaqueductal grey and dorsal brainstem nuclei vulnerable"},
]


def _make_patient(i):
    sex = rng.choice(["M", "F"])  # 1:1 sex ratio (AR, no sex bias)
    phenotype = rng.choices(
        [p["label"] for p in PHENOTYPES],
        weights=[p["pct"] for p in PHENOTYPES]
    )[0]

    age_map = {
        "Leigh Syndrome (Infantile, BG/Brainstem Lesions)": _rng_int(2, 24),
        "Severe Neonatal (Lactic Acidosis + Early Death)": _rng_int(0, 2),
        "Childhood Episodic (Exercise/Illness-Triggered LA)": _rng_int(24, 120),
        "Mild Subacute/Juvenile (Chronic Partial Deficiency)": _rng_int(60, 180),
    }
    age_mo = age_map.get(phenotype, _rng_int(1, 24))

    plasma_lactate = _rng_float(3.2, 20.0)
    plasma_pyruvate = _rng_float(0.30, 1.60)
    lp_ratio = round(plasma_lactate / plasma_pyruvate, 1)
    csf_lactate = _rng_float(plasma_lactate * 0.85, plasma_lactate * 1.45)
    plasma_alanine = _rng_int(490, 1380)

    on_kd = rng.random() > 0.28
    on_thiamine = rng.random() > 0.32
    on_carnitine = rng.random() > 0.38
    on_dca = rng.random() > 0.68
    on_lev = rng.random() > 0.48
    vpa_avoided = rng.random() > 0.07

    cc_agenesis = rng.random() > 0.60  # ~40% (less than PDHA1's 70%)
    leigh_lesions = phenotype == "Leigh Syndrome (Infantile, BG/Brainstem Lesions)" or rng.random() > 0.55
    dre = phenotype in ("Leigh Syndrome (Infantile, BG/Brainstem Lesions)", "Severe Neonatal (Lactic Acidosis + Early Death)") and rng.random() > 0.52

    return {
        "id": f"PDHB-{i:03d}",
        "sex": sex,
        "phenotype": phenotype,
        "onset_age_months": age_mo,
        "plasma_lactate_mmol": plasma_lactate,
        "plasma_pyruvate_mmol": plasma_pyruvate,
        "lp_ratio": lp_ratio,
        "csf_lactate_mmol": round(csf_lactate, 2),
        "plasma_alanine_umol": plasma_alanine,
        "cc_agenesis": cc_agenesis,
        "leigh_lesions": leigh_lesions,
        "on_kd": on_kd,
        "on_thiamine": on_thiamine,
        "on_carnitine": on_carnitine,
        "on_dca": on_dca,
        "on_lev": on_lev,
        "vpa_avoided": vpa_avoided,
        "dre": dre,
        "variant": rng.choice([v["variant"] for v in KEY_VARIANTS]),
    }


PATIENTS = [_make_patient(i) for i in range(1, 41)]


def get_overview():
    n = len(PATIENTS)
    avg_lactate = round(sum(p["plasma_lactate_mmol"] for p in PATIENTS) / n, 2)
    avg_pyruvate = round(sum(p["plasma_pyruvate_mmol"] for p in PATIENTS) / n, 3)
    avg_lp = round(sum(p["lp_ratio"] for p in PATIENTS) / n, 1)
    avg_alanine = round(sum(p["plasma_alanine_umol"] for p in PATIENTS) / n)
    cc_n = sum(1 for p in PATIENTS if p["cc_agenesis"])
    leigh_n = sum(1 for p in PATIENTS if p["leigh_lesions"])
    dre_n = sum(1 for p in PATIENTS if p["dre"])
    vpa_avoided_n = sum(1 for p in PATIENTS if p["vpa_avoided"])
    on_kd_n = sum(1 for p in PATIENTS if p["on_kd"])
    male_n = sum(1 for p in PATIENTS if p["sex"] == "M")
    normal_lp_n = sum(1 for p in PATIENTS if 10 <= p["lp_ratio"] <= 20)

    return {
        "dashboard": "PDHB Epilepsy (Pyruvate Dehydrogenase Complex Deficiency — E1β Subunit / Autosomal Recessive / PDH Block / Leigh Syndrome)",
        "gene": "PDHB",
        "protein": "Pyruvate Dehydrogenase E1-beta subunit (E1β; PDC-E1β)",
        "omim_gene": "*179060",
        "omim_disease": "#614111",
        "locus": "3p14.3",
        "aa_length": 359,
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF; males and females equally affected; no sex bias (unlike PDHA1 X-linked dominant); carriers unaffected",
        "cofactor": "None directly for E1β; E1β supports E1α which binds Thiamine Pyrophosphate (TPP/B1)",
        "mechanism": "E1β (PDHB) is the structural β-subunit of E1 heterotetramer; PDHB LOF → E1α₂β₂ cannot assemble → E1α degraded → PDH complex inactive → pyruvate cannot be converted to Acetyl-CoA → lactic acidosis + TCA energy failure",
        "pdh_complex_components": [
            {"component": "E1 (PDHA1 + PDHB)", "function": "Pyruvate decarboxylation; E1α binds TPP; E1β required for E1α₂β₂ heterotetrameric assembly and E1α stability"},
            {"component": "E2 (DLAT)", "function": "Dihydrolipoyl acetyltransferase; transfers acetyl group from E1 to CoA → Acetyl-CoA"},
            {"component": "E3BP (PDHX)", "function": "Bridges E3 (DLD) to E2 core; no catalytic activity"},
            {"component": "E3 (DLD)", "function": "Dihydrolipoamide dehydrogenase; regenerates oxidised lipoyl arm (shared with αKGDH, BCKDH, GCS)"},
            {"component": "PDK1/2/3/4", "function": "Pyruvate dehydrogenase kinase; phosphorylates E1α Ser293/300/232 → INACTIVATES PDH complex"},
            {"component": "PDP1/2", "function": "Pyruvate dehydrogenase phosphatase; dephosphorylates E1α → ACTIVATES PDH complex"},
        ],
        "key_distinguishing_from_pdha1": "PDHB is AUTOSOMAL RECESSIVE (AR, 3p14.3) vs PDHA1 X-linked Dominant (Xp22.12). Biochemically IDENTICAL — gene panel (PDHA1 + PDHB) is MANDATORY to distinguish. PDHB: no sex bias; CC agenesis less common (~40% vs 70% PDHA1); Leigh syndrome more prominent. Ultra-rare: ~50–80 cases worldwide vs ~500+ for PDHA1.",
        "key_distinguishing_feature": "L:P ratio NORMAL (10–20) in PDHB deficiency — both lactate and pyruvate rise equally (PDH block); distinguishes from Complex I deficiency (L:P >25). Biochemically indistinguishable from PDHA1 — requires PDHA1 + PDHB gene panel.",
        "structural_brain_hallmark": "Leigh syndrome (basal ganglia + brainstem symmetric T2 lesions, 65%) + corpus callosum agenesis/dysgenesis (~40%, less common than PDHA1); periventricular white matter changes; no pachygyria; no NKH burst-suppression EEG",
        "kpis": {
            "cohort_n": n,
            "male_pct": round(male_n / n * 100, 1),
            "avg_plasma_lactate_mmol": avg_lactate,
            "avg_plasma_pyruvate_mmol": avg_pyruvate,
            "avg_lp_ratio": avg_lp,
            "normal_lp_ratio_10_20_pct": round(normal_lp_n / n * 100, 1),
            "avg_plasma_alanine_umol": avg_alanine,
            "cc_agenesis_pct": round(cc_n / n * 100, 1),
            "leigh_lesions_pct": round(leigh_n / n * 100, 1),
            "dre_pct": round(dre_n / n * 100, 1),
            "on_kd_pct": round(on_kd_n / n * 100, 1),
            "vpa_avoided_pct": round(vpa_avoided_n / n * 100, 1),
        },
        "phenotype_distribution": PHENOTYPES,
        "high_risk_drugs": HIGH_RISK_DRUGS,
        "structural_anomalies": STRUCTURAL_ANOMALIES,
        "generated": datetime.utcnow().isoformat() + "Z",
    }


def get_breakdown():
    return {
        "dashboard": "PDHB Breakdown",
        "cohort_n": len(PATIENTS),
        "phenotype_distribution": PHENOTYPES,
        "seizure_types": SEIZURE_TYPES,
        "trigger_types": TRIGGER_TYPES,
        "treatments": TREATMENTS,
        "key_variants": KEY_VARIANTS,
        "biomarkers": BIOMARKERS,
        "structural_anomalies": STRUCTURAL_ANOMALIES,
        "patients_sample": PATIENTS[:10],
        "generated": datetime.utcnow().isoformat() + "Z",
    }


def get_definitions():
    return {
        "dashboard": "PDHB Definitions",
        "gene_card": {
            "gene": "PDHB",
            "full_name": "Pyruvate Dehydrogenase E1-beta Subunit",
            "alias": "PDC-E1β / PDHB / PDHBD",
            "locus": "3p14.3",
            "aa_length": 359,
            "structure": "E1β subunit (structural); pairs with E1α (PDHA1) to form E1α₂β₂ heterotetramer; E1β does NOT bind TPP or pyruvate but is required for E1α stability and heterotetrameric assembly",
            "cofactor": "None directly; E1β supports E1α's TPP-binding domain (Thiamine Pyrophosphate / B1 is cofactor for E1α)",
            "reaction": "E1β required for E1α₂β₂ assembly → PDHB LOF → E1α degraded → no E1 component → PDH complex inactive → pyruvate accumulates (identical biochemical outcome to PDHA1)",
            "inheritance": "Autosomal Recessive (AR) — biallelic LOF; carriers unaffected; no sex bias; consanguinity increases risk",
            "omim_gene": "*179060",
            "omim_disease": "#614111",
            "pdha1_vs_pdhb": "BIOCHEMICALLY IDENTICAL: same L:P ratio, same lactate/pyruvate/alanine pattern, same treatment. Gene panel (PDHA1 + PDHB) mandatory for molecular assignment. Inheritance differs: PDHB=AR (3p14.3) vs PDHA1=XLD (Xp22.12).",
        },
        "key_concepts": [
            {"term": "PDHB / E1β subunit — structural role", "definition": "E1β does not catalyse any chemical reaction itself. Its sole role is structural: E1β pairs with E1α to form the E1α₂β₂ heterotetramer. Without functional E1β, E1α cannot fold correctly or assemble → E1α is degraded by the mitochondrial quality-control protease system → PDH complex lacks its E1 component entirely."},
            {"term": "Normal L:P ratio (KEY PDH fingerprint — identical to PDHA1)", "definition": "In PDHB deficiency, both lactate AND pyruvate accumulate equally (pyruvate→lactate via LDH). L:P ratio = 10–20 (NORMAL). Complex I deficiency: NADH accumulates → L:P >25 (lactate rises preferentially). This distinguishes PDH deficiency (PDHB or PDHA1) from respiratory chain disorders. Gene panel needed to distinguish PDHB from PDHA1."},
            {"term": "Autosomal Recessive — critical difference from PDHA1", "definition": "PDHB is at 3p14.3 (autosomal). Both males and females are equally affected. No X-inactivation variable severity seen in PDHA1 females. Carrier parents are unaffected. Consanguinity is a risk factor. De novo mutations are rare (AR disease — usually requires two inherited LOF alleles). This is the ONLY clinically actionable difference between PDHB and PDHA1: inheritance pattern and genetic counselling."},
            {"term": "PDHB vs PDHA1 — gene panel mandatory", "definition": "PDHB and PDHA1 deficiencies produce biochemically IDENTICAL phenotypes (lactic acidosis, normal L:P ratio, elevated alanine, CSF lactate > plasma lactate, no BCAA/2-HG elevation). The ONLY way to distinguish them is molecular diagnosis (gene panel including PDHA1 + PDHB + DLAT + PDHX). Never assume PDHA1 without testing PDHB; never assume PDHB without testing PDHA1."},
            {"term": "Leigh syndrome — most prominent structural finding in PDHB", "definition": "Symmetric bilateral necrotising lesions in basal ganglia (putamen, caudate), brainstem (periaqueductal grey, dorsal medulla), and thalami; T2 hyperintense on MRI. Occurs in ~65% of PDHB patients — more prominent than in PDHA1 where CC agenesis is the hallmark. Energy-failure driven necrosis at highest-metabolic-demand grey-matter structures."},
            {"term": "Corpus callosum agenesis — less common in PDHB vs PDHA1", "definition": "CC agenesis occurs in ~40% of PDHB patients vs ~70% in PDHA1. In PDHA1, the X-linked dominant pattern creates severe hemizygous males who often have CC agenesis as the dominant structural anomaly. In PDHB (AR), the distribution of phenotypes is less skewed toward neonatal CC agenesis, and Leigh syndrome predominates instead."},
            {"term": "Ketogenic diet — FIRST LINE in PDHB (same as PDHA1)", "definition": "KD provides ketones (acetoacetate + β-hydroxybutyrate) → Acetyl-CoA via thiolase, BYPASSING the E1-blocked PDH complex. Ketones enter TCA cycle downstream of the PDH step. Same mechanism and same efficacy level as in PDHA1. Essential to start KD as early as possible in all PDHB phenotypes except mild episodic."},
            {"term": "Thiamine responsiveness in PDHB", "definition": "E1β does not directly bind TPP; thiamine (B1) cofactor is for E1α (PDHA1). However, high-dose thiamine trial is still recommended in PDHB because TPP binding by E1α may help stabilise residual E1α and partially reconstitute E1 complex activity even with reduced E1β. Thiamine responsiveness is LESS common in PDHB (~20–35%) than in PDHA1 (~40–55%) but always trialled before declaring non-responsive."},
            {"term": "DCA in PDHB — limited efficacy vs PDHA1", "definition": "DCA inhibits PDK → prevents E1α phosphorylation → keeps PDH complex active. In PDHB, if some residual E1β function remains (partial-loss variants like p.Ile199Thr, p.Arg161His), DCA may increase residual E1 complex activity and reduce lactate. If E1β is completely absent (null/deletion variants), DCA cannot rescue PDH complex activity. Trial DCA only in partial-loss PDHB variants."},
            {"term": "VPA absolute CI in PDHB (same as PDHA1)", "definition": "Valproic acid in PDH complex deficiency: (1) mitochondrial hepatotoxicity — same POLG1/MELAS-equivalent risk; (2) carnitine depletion — impairs β-oxidation → reduces ketone production → undermines KD therapy. ABSOLUTE contraindication. Never prescribe VPA in any PDH complex subunit deficiency (PDHA1, PDHB, DLAT, PDHX, DLD)."},
            {"term": "OMIM #614111", "definition": "Pyruvate Dehydrogenase E1-beta Deficiency; autosomal recessive; PDHB gene *179060; ultra-rare (~50–80 cases worldwide 2026); biochemically identical to PDHA1 (#312170) but AR inheritance and distinct molecular diagnosis."},
        ],
        "thresholds": [
            {"parameter": "Plasma lactate (normal)", "threshold": "<2.2 mmol/L", "pdhb_range": "3–22 mmol/L (crisis >12)"},
            {"parameter": "Plasma pyruvate (normal)", "threshold": "<0.17 mmol/L", "pdhb_range": "0.3–1.7 mmol/L"},
            {"parameter": "L:P ratio (KEY distinguisher)", "threshold": "Normal 10–20; CI defect >25", "pdhb_range": "10–20 (NORMAL — both lactate AND pyruvate rise equally)"},
            {"parameter": "Plasma alanine (normal)", "threshold": "150–450 µmol/L", "pdhb_range": "480–1400 µmol/L (chronic marker)"},
            {"parameter": "CSF lactate (normal)", "threshold": "<2.1 mmol/L", "pdhb_range": "3–10 mmol/L; CSF > plasma typical"},
            {"parameter": "Plasma BCAA (leucine, isoleucine, valine)", "threshold": "Normal", "pdhb_range": "NORMAL — key negative (DLD/E3 deficiency causes BCAA elevation)"},
            {"parameter": "PDH enzyme activity (fibroblasts)", "threshold": ">1.0 nmol/min/mg protein", "pdhb_range": "<0.3 nmol/min/mg (severe); 0.3–0.7 (moderate)"},
            {"parameter": "Ketone bodies on KD (target)", "threshold": "β-OH-butyrate ≥2–4 mmol/L", "pdhb_range": "Confirms ketosis; therapeutic bypass of PDH E1 block; monitor daily"},
        ],
        "differential_diagnosis": [
            {"condition": "PDHA1 deficiency (E1α subunit, XLD)", "distinguishing": "Biochemically IDENTICAL — requires gene panel. PDHA1: X-linked dominant; males severely affected (hemizygous); females variable (X-inactivation); CC agenesis ~70% (vs ~40% PDHB); de novo ~65–80%; OMIM *300502/#312170"},
            {"condition": "Complex I (NADH:ubiquinone oxidoreductase) deficiency", "distinguishing": "L:P ratio >25 (NADH accumulates → lactate preferentially); normal pyruvate; ND gene or NDUF genes; no CC agenesis"},
            {"condition": "DLD deficiency (E3 subunit, FOUR-complex block)", "distinguishing": "COMBINED PDH + αKGDH + BCKDH + GCS block; BCAA elevated (Leu/Ile/Val, BCKDH blocked); 2-hydroxyglutarate elevated (αKGDH blocked); glycine mildly elevated (GCS partial); AR 7q31.1; NOT pure PDH block"},
            {"condition": "DLAT deficiency (E2 subunit)", "distinguishing": "AR; DLAT gene (11q23.1); identical PDH block biochemistry; no BCAA/2-HG elevation (unlike DLD); gene panel distinguishes from PDHB and PDHA1"},
            {"condition": "PDHX deficiency (E3BP)", "distinguishing": "AR; PDHX gene (11p13); identical PDH block biochemistry; E3BP links DLD to E2 core; management same (KD + thiamine); gene panel distinguishes"},
            {"condition": "Pyruvate Carboxylase deficiency (PC)", "distinguishing": "PC converts pyruvate → OAA; BIOTIN responsive; hyperammonaemia; citrin-like pattern; AR PC gene; L:P may be elevated differently"},
            {"condition": "MELAS (Mitochondrial Encephalomyopathy, Lactic Acidosis, Stroke)", "distinguishing": "L:P >20; stroke-like episodes (cortical, not BG-predominant); maternal inheritance; mtDNA m.3243A>G; no CC agenesis; elevated lactate in CSF without proportional pyruvate"},
            {"condition": "Fumarase deficiency", "distinguishing": "Progressive encephalopathy; gyral abnormalities; fumaric aciduria on urine organic acids; AR FH gene; no pyruvate accumulation"},
        ],
        "generated": datetime.utcnow().isoformat() + "Z",
    }
