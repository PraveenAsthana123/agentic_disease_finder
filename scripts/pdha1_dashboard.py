#!/usr/bin/env python3
"""PDHA1 (Pyruvate Dehydrogenase E1-alpha Subunit) Epilepsy Dashboard.

PDHA1 encodes the E1α subunit of the Pyruvate Dehydrogenase Complex (PDH complex / PDC):
  PDC structure: E1 (E1α₂E1β₂ tetramer, PDHA1+PDHB) + E2 (DLAT, 24-mer) + E3BP (PDHX) + E3 (DLD)
  PDH complex catalyses: Pyruvate + CoA + NAD⁺ → Acetyl-CoA + CO₂ + NADH  (irreversible)
  E1α step (PDHA1): Pyruvate → Hydroxyethyl-TPP (using thiamine pyrophosphate, TPP/B1)
                    Then E1β completes transfer to E2-lipoyl arm → Acetyl-CoA

PDHA1 LOF — PYRUVATE CANNOT ENTER TCA CYCLE VIA ACETYL-COA:
  Pyruvate accumulates → LDH converts pyruvate → lactate (lactic acidosis)
  Brain-specific catastrophe: neurons rely almost exclusively on pyruvate oxidation for energy;
  ketones partially bypass PDH block (acetyl-CoA from β-oxidation).
  PDH complex is regulated by PDK1-4 (phosphorylates/inactivates E1α Ser293/Ser300/Ser232)
  and PDP1/2 (dephosphorylates/activates E1α) — DCA activates by inhibiting PDK.

CRITICAL BIOCHEMICAL FINGERPRINT (distinguishes PDHA1 from respiratory chain disorders):
  • Lactate:Pyruvate (L:P) ratio NORMAL (~10–20) — BOTH rise together (unlike complex I where L:P >25)
  • CSF lactate typically > plasma lactate (brain generates excess pyruvate/lactate)
  • CSF pyruvate elevated (>0.17 mmol/L)
  • CSF glucose normal but reduced relative to plasma glucose (impaired neuronal energy utilisation)
  • Plasma amino acids: alanine elevated (pyruvate transamination surrogate marker)

X-LINKED INHERITANCE (UNIQUE among common metabolic epilepsies):
  PDHA1 = Xp22.12; X-linked dominant (XLD)
  Males (hemizygous): almost universally severe — neonatal lactic acidosis / Leigh syndrome / early death
  Females (heterozygous): variable due to X-inactivation skewing:
    Favourable X-inactivation → mild/episodic; Unfavourable → as severe as males
  De novo mutations common (~65–80%); no significant founder allele (pan-ethnic).

STRUCTURAL BRAIN ANOMALIES (hallmark — UNIQUE to PDHA1 among metabolic DEEs):
  1. Corpus callosum agenesis/dysgenesis (most common, ~70% severe males)
  2. Leigh syndrome — symmetric necrotising lesions basal ganglia + brainstem + periaqueductal grey
  3. Ventriculomegaly + cortical atrophy (energy failure during neuronal migration)
  4. Periventricular nodular heterotopia (rare)
  5. Colpocephaly (dilatation of occipital horns from CC hypoplasia)
  No pachygyria (unlike peroxisomal ZSD); no NKH-EEG burst-suppression pattern

PHENOTYPE CLASSES (PDHA1, 2026):
  Severe Neonatal (males): neonatal lactic acidosis + CC agenesis + seizures + early death (~30%)
  Leigh Syndrome (males, infantile): basal ganglia + brainstem Leigh lesions + psychomotor regression (~35%)
  Episodic / Ataxic (females and mosaic males): episodic lactic acidosis + ataxia + partial seizures (~25%)
  Dysmorphic (males, neonatal): facial features (prominent forehead, broad nasal bridge) + CC agenesis (~10%)
  ~500+ PDHA1 cases worldwide 2026; OMIM gene *300502; disease #312170

KEY VARIANTS:
  p.Arg302His: most common pathogenic; CpG hotspot; moderate–severe; some thiamine-responsive
  p.Arg302Cys: second most common; CpG hotspot; moderate; similar to Arg302His
  p.Ala321Val: moderate; juvenile episodic ataxia phenotype; partial E1α activity retained
  p.Gly194Arg: European; moderate; CC dysgenesis typical
  Large deletions / nonsense (hemizygous males): complete E1α absence; severe neonatal; near-universal early death
  Mosaic (somatic): variable severity; may have intermittent lactic acidosis only

TREATMENT:
  1. Ketogenic diet (Level A — FIRST LINE): bypasses PDH block → ketones provide acetyl-CoA directly;
     most effective in PDHA1 (unlike most mitochondrial conditions where KD is risky)
  2. Thiamine / B1 (Level A): TPP cofactor for E1α; 100–600 mg/day; trial all patients;
     thiamine-responsive variants (p.Arg302His) show significant biochemical improvement
  3. L-Carnitine (Level B): secondary depletion; supports β-oxidation for ketone production
  4. DCA (Dichloroacetate, Level B): inhibits PDK1/3 → keeps E1α dephosphorylated (active form);
     reduces plasma/CSF lactate; NEUROPATHY risk (limit cumulative dose); alternative when KD not tolerated
  5. Riboflavin / B2 (Level C): minimal direct PDH effect; given if E3/DLD component involved
  6. LEV (Level B): first-line AED; no metabolic interaction with PDH pathway

HIGH-RISK DRUGS:
  VPA: ABSOLUTE CI — same mitochondrial hepatotoxicity risk as POLG1 + MELAS; carnitine depletion
       worsens already-impaired fatty acid oxidation required for ketone production
  High-carbohydrate diet: EXTREME HAZARD — floods pyruvate → worsens lactic acidosis;
       avoid high-glucose IV unless KD ketones are maintained; use balanced infusion
  Glucose-only IV drip (acute crisis): DANGEROUS — pyruvate load without ketone buffer;
       prefer saline + carnitine + low-glucose during acute crises unless frank hypoglycaemia
  Phenobarbital: CAUTION — respiratory depression + sedation may mask metabolic crisis evolution
  Fasting (unmanaged): CAUTION — catabolism raises fatty acids → can paradoxically help (ketones) BUT
       uncontrolled fasting → gluconeogenesis surge → pyruvate flux → crisis

GENETICS SUMMARY:
  Gene: PDHA1 · Xp22.12 · 390 aa (including 29-aa mitochondrial targeting sequence) · E1α subunit
  X-linked dominant (XLD) · OMIM gene *300502 · disease #312170
  ~500+ pathogenic variants; no founder allele; de novo ~65–80%
  PDH complex regulation: PDK1/2/3/4 (inactivate by phospho-Ser293) · PDP1/2 (activate)
  DCA mechanism: DCA inhibits PDK → prevents E1α phosphorylation → more active PDH complex
"""

import random
from datetime import datetime

SEED = 20260823
rng = random.Random(SEED)

def _rng_choice(items): return rng.choice(items)
def _rng_int(lo, hi): return rng.randint(lo, hi)
def _rng_float(lo, hi, dec=2): return round(rng.uniform(lo, hi), dec)

# Colour: deep amber-orange — PDH block / lactate / pyruvate / energy failure
COLOUR = "#e65100"  # deep orange — pyruvate block / lactic acidosis / Leigh syndrome


# ── cohort ──────────────────────────────────────────────────────────────────
PHENOTYPES = [
    {"label": "Severe Neonatal (CC Agenesis + Lactic Acidosis)", "pct": 30, "color": "#b71c1c"},
    {"label": "Leigh Syndrome (Infantile, BG/Brainstem Lesions)", "pct": 35, "color": "#e65100"},
    {"label": "Episodic Ataxia / Lactic Acidosis (Female / Mosaic)", "pct": 25, "color": "#f9a825"},
    {"label": "Dysmorphic Neonatal (CC Agenesis + Facial Features)", "pct": 10, "color": "#558b2f"},
]

SEIZURE_TYPES = [
    {"type": "Focal with secondary generalisation", "pct": 65},
    {"type": "Tonic / tonic-clonic", "pct": 58},
    {"type": "Myoclonic", "pct": 42},
    {"type": "Epileptic Spasms (IS)", "pct": 35},
    {"type": "Absence-like episodic", "pct": 28},
    {"type": "Status epilepticus (febrile / metabolic)", "pct": 22},
]

TRIGGER_TYPES = [
    {"trigger": "Febrile illness / infection", "pct": 82},
    {"trigger": "Carbohydrate-heavy meal / glucose load", "pct": 75},
    {"trigger": "Prolonged fasting (unmanaged)", "pct": 55},
    {"trigger": "Surgery / anaesthesia (glucose drip)", "pct": 62},
    {"trigger": "Physiological stress / illness", "pct": 48},
    {"trigger": "Intercurrent GI illness + vomiting (missed KD)", "pct": 44},
]

TREATMENTS = [
    {"drug": "Ketogenic Diet (KD)", "level": "A", "response_pct": 78, "color": "#00695c"},
    {"drug": "Thiamine (B1, TPP cofactor)", "level": "A", "response_pct": 52, "color": "#00838f"},
    {"drug": "L-Carnitine (secondary depletion + KD support)", "level": "B", "response_pct": 60, "color": "#2e7d32"},
    {"drug": "Dichloroacetate (DCA — PDK inhibitor)", "level": "B", "response_pct": 45, "color": "#0277bd"},
    {"drug": "Riboflavin (B2, minimal PDH effect)", "level": "C", "response_pct": 18, "color": "#827717"},
    {"drug": "Levetiracetam (LEV)", "level": "B", "response_pct": 55, "color": "#4a148c"},
]

HIGH_RISK_DRUGS = [
    {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI",
     "mechanism": "Mitochondrial hepatotoxicity (POLG1-equivalent risk in PDH deficiency) + carnitine depletion (impairs β-oxidation needed for KD ketone production)"},
    {"drug": "High-carbohydrate diet / high-glucose IV", "risk": "EXTREME HAZARD",
     "mechanism": "Floods pyruvate → worsens lactic acidosis; pyruvate cannot enter TCA via E1α; use balanced glucose or KD-adjusted IV in crisis"},
    {"drug": "Glucose-only IV drip (acute crisis)", "risk": "HIGH RISK",
     "mechanism": "Pyruvate load without ketone buffer → acute decompensation; prefer saline + carnitine; only use glucose if frank hypoglycaemia + KD maintained"},
    {"drug": "Phenobarbital", "risk": "CAUTION",
     "mechanism": "Respiratory depression + sedation may mask evolving metabolic crisis; monitor lactate and respiratory status closely"},
    {"drug": "Standard high-carbohydrate enteral feeding", "risk": "CONTRAINDICATED in KD-dependent patients",
     "mechanism": "Breaks ketosis; re-exposes E1α-deficient neurons to pyruvate-only energy substrate; substitute KD-compatible formula"},
]

KEY_VARIANTS = [
    {"variant": "p.Arg302His", "effect": "Most common; CpG hotspot; moderate–severe; thiamine-responsive subset; Leigh syndrome", "phenotype": "Leigh/Episodic"},
    {"variant": "p.Arg302Cys", "effect": "Second most common; CpG hotspot; moderate; similar to Arg302His; CC dysgenesis", "phenotype": "Leigh/Neonatal"},
    {"variant": "p.Ala321Val", "effect": "Partial E1α activity retained; juvenile episodic ataxia phenotype; lactic acidosis on illness", "phenotype": "Episodic Ataxia"},
    {"variant": "p.Gly194Arg", "effect": "European; moderate–severe; CC dysgenesis; infantile Leigh", "phenotype": "Leigh"},
    {"variant": "Large deletion / nonsense (hemizygous male)", "effect": "Complete E1α absence; severe neonatal lactic acidosis; CC agenesis; near-universal early death", "phenotype": "Severe Neonatal"},
    {"variant": "Mosaic (somatic)", "effect": "Variable severity based on degree of mosaicism; may present with intermittent lactic acidosis only", "phenotype": "Episodic/Mild"},
]

BIOMARKERS = [
    {"name": "Plasma Lactate", "normal": "<2.2 mmol/L", "pdha1_range": "3–25 mmol/L", "significance": "PDH block → pyruvate→lactate via LDH; primary indicator"},
    {"name": "Plasma Pyruvate", "normal": "<0.17 mmol/L", "pdha1_range": "0.3–1.8 mmol/L", "significance": "Accumulates (cannot enter TCA); KEY: L:P ratio normal (both rise)"},
    {"name": "L:P ratio (Lactate:Pyruvate)", "normal": "10–20", "pdha1_range": "10–20 (NORMAL — KEY)", "significance": "Normal L:P distinguishes PDH deficiency from complex I (>25 in CI defect)"},
    {"name": "Plasma Alanine", "normal": "150–450 µmol/L", "pdha1_range": "500–1500 µmol/L", "significance": "Pyruvate transamination surrogate; most sensitive chronic marker"},
    {"name": "CSF Lactate", "normal": "<2.1 mmol/L", "pdha1_range": "3–12 mmol/L (often > plasma)", "significance": "Brain preferentially affected; CSF lactate > plasma lactate typical"},
    {"name": "CSF Pyruvate", "normal": "<0.17 mmol/L", "pdha1_range": "0.3–1.2 mmol/L", "significance": "Elevated CSF pyruvate diagnostic for PDH block at brain level"},
    {"name": "CSF Glucose", "normal": "2.5–4.5 mmol/L (≥60% plasma)", "pdha1_range": "Normal absolute but reduced ratio", "significance": "Neurons cannot utilise glucose without PDH; relatively low utilisation"},
    {"name": "MRI Brain", "normal": "Normal", "pdha1_range": "CC agenesis/dysgenesis; Leigh lesions BG/BS; ventriculomegaly", "significance": "Structural brain anomaly near-universal in severe phenotypes; guides prognosis"},
    {"name": "PDH enzyme activity (fibroblasts/lymphocytes)", "normal": ">1.0 nmol/min/mg", "pdha1_range": "<0.3 nmol/min/mg (severe); 0.3–0.7 (moderate)", "significance": "Confirmatory; fibroblasts preferred; may be falsely normal if X-inactivation favourable"},
]

STRUCTURAL_ANOMALIES = [
    {"anomaly": "Corpus callosum agenesis/dysgenesis", "frequency_pct": 72, "significance": "Hallmark of severe PDHA1; splenium affected first; absent CC = severe neonatal phenotype"},
    {"anomaly": "Leigh syndrome (BG + brainstem symmetric lesions)", "frequency_pct": 58, "significance": "T2 hyperintense; periaqueductal grey, putamen, caudate, dorsal medulla; energy-failure pattern"},
    {"anomaly": "Ventriculomegaly", "frequency_pct": 48, "significance": "Secondary to CC hypoplasia and cortical atrophy from energy failure"},
    {"anomaly": "Colpocephaly (occipital horn dilatation)", "frequency_pct": 35, "significance": "Associated with CC agenesis; occipital horns disproportionately enlarged"},
    {"anomaly": "Cortical atrophy / simplified gyration", "frequency_pct": 25, "significance": "Neuronal energy failure during migration; not true lissencephaly (unlike ZSD)"},
    {"anomaly": "Periventricular nodular heterotopia", "frequency_pct": 8, "significance": "Rare; neuronal migration disruption due to energy failure during migration"},
]


def _make_patient(i):
    sex = rng.choice(["M", "M", "M", "F"])  # 3:1 male:female (X-linked)
    phenotype = rng.choices(
        [p["label"] for p in PHENOTYPES],
        weights=[p["pct"] for p in PHENOTYPES]
    )[0]

    # Females tend toward episodic; males toward severe
    if sex == "F" and "Severe Neonatal" in phenotype:
        phenotype = "Episodic Ataxia / Lactic Acidosis (Female / Mosaic)"
    if sex == "F" and "Dysmorphic" in phenotype:
        phenotype = "Leigh Syndrome (Infantile, BG/Brainstem Lesions)"

    age_map = {
        "Severe Neonatal (CC Agenesis + Lactic Acidosis)": _rng_int(0, 1),
        "Leigh Syndrome (Infantile, BG/Brainstem Lesions)": _rng_int(2, 18),
        "Episodic Ataxia / Lactic Acidosis (Female / Mosaic)": _rng_int(12, 60),
        "Dysmorphic Neonatal (CC Agenesis + Facial Features)": _rng_int(0, 2),
    }
    age_mo = age_map.get(phenotype, _rng_int(1, 24))

    plasma_lactate = _rng_float(3.5, 22.0)
    plasma_pyruvate = _rng_float(0.32, 1.65)
    lp_ratio = round(plasma_lactate / plasma_pyruvate, 1)
    csf_lactate = _rng_float(plasma_lactate * 0.9, plasma_lactate * 1.5)
    plasma_alanine = _rng_int(520, 1450)

    on_kd = rng.random() > 0.25
    on_thiamine = rng.random() > 0.30
    on_carnitine = rng.random() > 0.35
    on_dca = rng.random() > 0.65
    on_lev = rng.random() > 0.45
    vpa_avoided = rng.random() > 0.06

    cc_agenesis = phenotype in ("Severe Neonatal (CC Agenesis + Lactic Acidosis)", "Dysmorphic Neonatal (CC Agenesis + Facial Features)") or rng.random() > 0.55
    leigh_lesions = phenotype == "Leigh Syndrome (Infantile, BG/Brainstem Lesions)" or rng.random() > 0.70
    dre = phenotype in ("Severe Neonatal (CC Agenesis + Lactic Acidosis)", "Leigh Syndrome (Infantile, BG/Brainstem Lesions)") and rng.random() > 0.50

    return {
        "id": f"PDHA1-{i:03d}",
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
        "dashboard": "PDHA1 Epilepsy (Pyruvate Dehydrogenase Complex Deficiency — E1α Subunit / X-linked / PDH Block / Leigh Syndrome / Corpus Callosum Agenesis)",
        "gene": "PDHA1",
        "protein": "Pyruvate Dehydrogenase E1-alpha subunit (E1α; PDC-E1α)",
        "omim_gene": "*300502",
        "omim_disease": "#312170",
        "locus": "Xp22.12",
        "aa_length": 390,
        "inheritance": "X-linked Dominant (XLD) — males hemizygous (severe); females heterozygous (variable X-inactivation)",
        "cofactor": "Thiamine Pyrophosphate (TPP/B1) — required for E1α decarboxylation of pyruvate",
        "mechanism": "E1α (PDHA1) catalyses pyruvate → hydroxyethyl-TPP (decarboxylation); PDHA1 LOF → pyruvate cannot be converted to Acetyl-CoA → lactic acidosis + TCA cycle energy failure",
        "pdh_complex_components": [
            {"component": "E1 (PDHA1 + PDHB)", "function": "Pyruvate decarboxylation; E1α binds TPP; TPP-Mg²⁺ attacks pyruvate carbonyl"},
            {"component": "E2 (DLAT)", "function": "Dihydrolipoyl acetyltransferase; transfers acetyl group from E1 to CoA → Acetyl-CoA"},
            {"component": "E3BP (PDHX)", "function": "Bridges E3 (DLD) to E2 core; no catalytic activity"},
            {"component": "E3 (DLD)", "function": "Dihydrolipoamide dehydrogenase; regenerates oxidised lipoyl arm (shared with αKGDH, BCKDH, GCS)"},
            {"component": "PDK1/2/3/4", "function": "Pyruvate dehydrogenase kinase; phosphorylates E1α Ser293/300/232 → INACTIVATES PDH complex"},
            {"component": "PDP1/2", "function": "Pyruvate dehydrogenase phosphatase; dephosphorylates E1α → ACTIVATES PDH complex"},
        ],
        "key_distinguishing_feature": "L:P ratio NORMAL (10–20) in PDHA1 deficiency — both lactate and pyruvate rise together; distinguishes from Complex I deficiency where L:P >25 (NADH accumulation drives LDH toward lactate preferentially)",
        "structural_brain_hallmark": "Corpus callosum agenesis/dysgenesis (~70% severe males) + Leigh syndrome (basal ganglia + brainstem symmetric T2 lesions); NO pachygyria (unlike ZSD); NO NKH burst-suppression EEG",
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
        "dashboard": "PDHA1 Breakdown",
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
        "dashboard": "PDHA1 Definitions",
        "gene_card": {
            "gene": "PDHA1",
            "full_name": "Pyruvate Dehydrogenase E1-alpha Subunit",
            "alias": "PDC-E1α / PDHA / PDHAD",
            "locus": "Xp22.12",
            "aa_length": 390,
            "structure": "E1α₂E1β₂ heterotetrameric E1 component; E1α contains TPP-binding domain + phosphorylation regulatory sites (Ser293/300/232)",
            "cofactor": "Thiamine Pyrophosphate (TPP) — TPP-Mg²⁺ attacks pyruvate carbonyl to initiate decarboxylation",
            "reaction": "Pyruvate + TPP-E1α → Hydroxyethyl-TPP-E1α + CO₂ → Acetyl-dihydrolipoamide-E2 → Acetyl-CoA (via E2 DLAT)",
            "inheritance": "X-linked Dominant (XLD) — PDHA1 is on X chromosome; males hemizygous (always severe); females heterozygous (variable by X-inactivation)",
            "omim_gene": "*300502",
            "omim_disease": "#312170",
            "regulation": "PDK1/2/3/4 phosphorylate Ser293/300/232 → inactivate; PDP1/2 dephosphorylate → activate; DCA inhibits PDK → therapeutic activation of residual PDH",
        },
        "key_concepts": [
            {"term": "PDHA1 / E1α subunit", "definition": "Catalytic α-subunit of E1 component of PDH complex; contains TPP-binding domain; PDHA1 LOF → pyruvate cannot be oxidised to Acetyl-CoA → lactic acidosis + brain energy failure"},
            {"term": "Normal L:P ratio (KEY PDH fingerprint)", "definition": "In PDHA1 deficiency, both lactate AND pyruvate accumulate (pyruvate→lactate via LDH equally); L:P ratio = 10–20 (normal). Complex I deficiency: NADH accumulates → L:P >25 (lactate rises preferentially). This is the single most important biochemical distinction."},
            {"term": "X-linked dominant (XLD) inheritance", "definition": "PDHA1 at Xp22.12; males hemizygous → always severely affected; females heterozygous → severity depends on X-inactivation ratio (favourable X-inactivation → mild episodic; unfavourable → severe)"},
            {"term": "Corpus callosum agenesis (CC agenesis)", "definition": "Most common structural brain anomaly in PDHA1; callosal neurons are highly energy-dependent during development; PDH block during embryogenesis → agenesis/dysgenesis; hallmark of severe neonatal/dysmorphic phenotype"},
            {"term": "Leigh syndrome pattern", "definition": "Symmetric bilateral necrotising lesions in basal ganglia (putamen, caudate), brainstem (periaqueductal grey, dorsal medulla), and occasionally thalami; T2 hyperintense on MRI; result of high energy demands in these grey-matter structures + PDH block"},
            {"term": "Ketogenic diet — FIRST LINE in PDHA1 (unlike most mitochondrial disorders)", "definition": "KD provides ketones (acetoacetate + β-hydroxybutyrate) → these enter TCA cycle as Acetyl-CoA via thiolase BYPASSING the PDH step; reduces reliance on pyruvate oxidation; most effective metabolic epilepsy intervention for PDHA1"},
            {"term": "Dichloroacetate (DCA)", "definition": "Halogenated acid that inhibits PDK1/3 → prevents E1α Ser293 phosphorylation → keeps PDH complex in active (dephosphorylated) state; increases residual PDH activity; reduces lactate 30–50%; NEUROPATHY risk limits long-term use (monitor nerve conduction)"},
            {"term": "Thiamine responsiveness in PDHA1", "definition": "~40–55% of patients with specific variants (e.g. p.Arg302His, p.Arg302Cys) show partial biochemical improvement with high-dose thiamine (100–600 mg/day); TPP cofactor stabilises mutant E1α conformation; always trial before declaring non-responsive"},
            {"term": "Alanine as pyruvate surrogate marker", "definition": "Alanine aminotransferase (ALT) converts pyruvate + glutamate → alanine + α-KG; plasma alanine 500–1500 µmol/L in PDHA1 deficiency; most sensitive CHRONIC biomarker (lactate may normalise between crises, alanine remains elevated)"},
            {"term": "VPA absolute CI in PDHA1", "definition": "Same mitochondrial hepatotoxicity risk as POLG1/MELAS context; carnitine depletion by VPA impairs β-oxidation — β-oxidation is the KEY metabolic pathway supplying ketones for KD therapy; VPA destroys the therapeutic mechanism"},
            {"term": "OMIM #312170", "definition": "Pyruvate Dehydrogenase Complex Deficiency (PDCD); X-linked; PDHA1 gene *300502; most common cause of PDH complex deficiency (~75% of all PDCD cases)"},
        ],
        "thresholds": [
            {"parameter": "Plasma lactate (normal)", "threshold": "<2.2 mmol/L", "pdha1_range": "3–25 mmol/L (crisis >15)"},
            {"parameter": "Plasma pyruvate (normal)", "threshold": "<0.17 mmol/L", "pdha1_range": "0.3–1.8 mmol/L"},
            {"parameter": "L:P ratio (key distinguisher)", "threshold": "Normal 10–20; CI defect >25", "pdha1_range": "10–20 (NORMAL — both lactate AND pyruvate rise)"},
            {"parameter": "Plasma alanine (normal)", "threshold": "150–450 µmol/L", "pdha1_range": "500–1500 µmol/L (chronic marker)"},
            {"parameter": "CSF lactate (normal)", "threshold": "<2.1 mmol/L", "pdha1_range": "3–12 mmol/L; CSF > plasma typical"},
            {"parameter": "PDH enzyme activity (normal)", "threshold": ">1.0 nmol/min/mg protein", "pdha1_range": "<0.3 nmol/min/mg (severe); 0.3–0.7 (moderate)"},
            {"parameter": "Ketone bodies on KD (target)", "threshold": "β-OH-butyrate ≥2–4 mmol/L", "pdha1_range": "Confirms ketosis; therapeutic bypass of PDH block; monitor daily"},
        ],
        "differential_diagnosis": [
            {"condition": "Complex I (NADH:ubiquinone oxidoreductase) deficiency", "distinguishing": "L:P ratio >25 (NADH accumulates → lactate preferentially); normal pyruvate; ND genes or NDUF genes; no CC agenesis"},
            {"condition": "Complex IV (Cytochrome c oxidase) deficiency", "distinguishing": "L:P ratio >25; SCO1/SCO2/SURF1/COX gene panel; hepatopathy + Leigh; no corpus callosum anomaly"},
            {"condition": "DLD deficiency (E3 subunit)", "distinguishing": "COMBINED four-complex block; BCAA elevated (BCKDH block); αKG elevated; glycine mildly elevated (GCS block); different gene (DLD, AR, 7q31.1); L:P may be elevated vs normal in PDHA1"},
            {"condition": "PDHB deficiency (E1β subunit)", "distinguishing": "Identical biochemistry to PDHA1; AR (autosomal recessive, not X-linked); PDHB gene; rare"},
            {"condition": "DLAT deficiency (E2 subunit)", "distinguishing": "AR; DLAT gene; identical PDH block biochemistry; no lipoic acid synthesis involvement (unlike E3 / DLD)"},
            {"condition": "PDHX deficiency (E3BP)", "distinguishing": "AR; PDHX gene; E3BP links DLD to E2 core; identical PDH block; important to distinguish as management same (KD + thiamine)"},
            {"condition": "Pyruvate Carboxylase deficiency (PC)", "distinguishing": "PC converts pyruvate → OAA (gluconeogenesis/anaplerosis); lactate elevated but BIOTIN responsive; hyperammonaemia; different biochemical pattern"},
            {"condition": "MELAS (Mitochondrial Encephalomyopathy, Lactic Acidosis, Stroke)", "distinguishing": "L:P >20; stroke-like episodes (cortical, not basal ganglia predominant); maternal inheritance; mtDNA m.3243A>G; no CC agenesis"},
        ],
        "generated": datetime.utcnow().isoformat() + "Z",
    }
