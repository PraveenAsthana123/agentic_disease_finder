#!/usr/bin/env python3
"""GLUD1 (Glutamate Dehydrogenase 1) Epilepsy Dashboard.

GLUD1 encodes Glutamate Dehydrogenase 1, the mitochondrial enzyme that catalyses:
  L-Glutamate + NAD(P)+ → α-Ketoglutarate + NH4+ + NAD(P)H  (oxidative deamination)
  Reverse: α-KG + NH4+ + NADPH → L-Glutamate + NADP+          (reductive amination)

GLUD1 DISEASE: Hyperinsulinism-Hyperammonemia (HHS) syndrome
  (also called Hyperinsulinemic Hypoglycemia-5 / HHF5 / Familial Hyperinsulinism-5)
  OMIM Gene: *138130   OMIM Disease: #606762
  Chromosome: 10q23.3
  Inheritance: Autosomal Dominant — GAIN-OF-FUNCTION (GoF) de novo or familial
  Protein: 505 aa; mitochondrial matrix; NAD(P)H-dependent; homohexameric; GTP-regulated
  Prevalence: ~200–300 cases worldwide 2026 (rare; likely under-diagnosed)

MECHANISM — GAIN-OF-FUNCTION (distinct from all other metabolic epilepsies which are LOF):
  Normal: GLUD1 strictly regulated by GTP (allosteric INHIBITOR at regulatory domain)
          → prevents uncontrolled glutamate catabolism
  GoF mutations: reduced GTP-sensitivity OR increased ADP-activation
          → GLUD1 runs hyperactively even when GTP says "stop"
  Two-organ consequence:
    1. Pancreatic β-cells: excess Glu → excess α-KG → excess ATP → K-ATP channels CLOSE
       → membrane depolarisation → voltage-Ca++ channel open → insulin EXCESS → HYPOGLYCEMIA
    2. Liver: excess Glu oxidation → excess NH4+ released → HYPERAMMONEMIA
  Leucine amplification: leucine allosterically activates GLUD1 at same domain as GoF mutations
    → protein-sensitive hypoglycemia is a hallmark (leucine load triggers attacks)

GLUD1 GoF BIOCHEMISTRY:
  Blood glucose: CRITICALLY LOW during episodes (<2.5 mmol/L / <45 mg/dL) — PATHOGNOMONIC
  Plasma insulin: HIGH (>3 μU/mL when glucose <2.5 mmol/L) — inappropriate hypersecretion
  Plasma ammonia: PERSISTENTLY ELEVATED (100–500 μmol/L; normal <50)  — PATHOGNOMONIC
  Plasma glutamate: LOW (hyperactive GLUD1 consumes glutamate)
  Plasma glutamine: LOW-NORMAL (depleted glutamate substrate)
  Plasma alanine: LOW-NORMAL (reduced transamination)
  alpha-KG/2-OG: MILDLY ELEVATED (product backup)
  Leucine/BCAA sensitivity: POSITIVE — unique metabolic stress test for GLUD1
  alpha-AASA: NORMAL — KEY NEGATIVE vs ALDH7A1 (PDE)
  Pipecolic acid: NORMAL — KEY NEGATIVE vs ALDH7A1
  MMA: NORMAL — KEY NEGATIVE vs MMUT/MMAB/cblC
  tHcy: NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR/MTRR/AHCY
  Methionine: NORMAL — KEY NEGATIVE vs CBS/GNMT/MAT1A
  Organic acids: NORMAL or minor non-specific findings
  Acylcarnitines: NORMAL — KEY NEGATIVE vs FAODs
  MRS brain: glutamate/glutamine (Glx) peak LOW-NORMAL (unusual for metabolic disorders)

EPILEPSY IN GLUD1 DEFICIENCY (predominantly absence — highly characteristic):
  Overall seizure rate: 75–85% (driven by hyperammonemia + glutamate imbalance)
  ABSENCE seizures: 65% of seizure patients — MOST CHARACTERISTIC seizure type
    (NH4+ → GABA-A receptor modulation at extrasynaptic receptors → absence phenotype)
  Febrile seizures: 30% (often first presentation, misdiagnosed as febrile convulsions)
  GTCS: 25%
  Myoclonic: 20%
  Drug-resistant epilepsy: 15–20% (less than most metabolic epilepsies)
  Hypoglycemia-triggered seizures: 50% — metabolic seizure (not epileptic per se)
  EEG: 3 Hz spike-and-wave (absence) or 2-3.5 Hz generalised in 55%
  MRI: usually NORMAL; hippocampal atrophy in 15% after prolonged hypoglycemia

TREATMENT (GLUD1 GoF — opposite direction to LOF diseases):
  Diazoxide (K-ATP opener): Level A PRIMARY — 5–15 mg/kg/day — prevents hypoglycemia ~95%
  Leucine/protein restriction: Level B — reduces stimulus for hyperactivation
  Octreotide (somatostatin analog): Level B — insulin suppression if diazoxide fails
  Levetiracetam: Level B first-line AED — if epilepsy persists after metabolic control
  Phenobarbital: Level B — older option for persistent seizures
  Valproate: HIGH RISK — inhibits urea cycle → ammonia MARKEDLY WORSE
  High-protein diet: ABSOLUTE CONTRAINDICATION — triggers hypoglycemia via leucine
  Glutamate supplements: ABSOLUTE CONTRAINDICATION — feeds overactive GLUD1
  Ethanol: HIGH RISK — shifts NAD+/NADH → activates GLUD1 → worsens hypoglycemia
  Ketogenic diet: MODERATE RISK — protein component may trigger hypoglycemia episodes
  Vigabatrin: MODERATE RISK — GABA-T inhibition may worsen ammonia handling
"""

import random

_SEED = 169
_N = 40


def _rng():
    return random.Random(_SEED)


def get_overview():
    rng = _rng()

    # Phenotype distribution
    n_severe   = round(_N * 0.50)    # Classic severe: early hypoglycemia + epilepsy + IDD
    n_moderate = round(_N * 0.35)    # Moderate
    n_mild     = _N - n_severe - n_moderate  # Mild/attenuated

    phenotypes = {
        "Classic-Severe": {"n": n_severe,   "pct": round(100 * n_severe / _N)},
        "Moderate":       {"n": n_moderate, "pct": round(100 * n_moderate / _N)},
        "Mild-Attenuated": {"n": n_mild,    "pct": round(100 * n_mild / _N)},
    }

    # Biomarker distributions (simulate realistic GoF values)
    glucoses  = [rng.uniform(1.5, 2.5) for _ in range(_N)]   # mmol/L — very low during episodes
    insulins  = [rng.uniform(5, 28) for _ in range(_N)]      # μU/mL — high when hypoglycemic
    ammonias  = [rng.uniform(98, 490) for _ in range(_N)]    # μmol/L — persistently elevated
    glutamates = [rng.uniform(18, 42) for _ in range(_N)]    # μmol/L — LOW (consumed)
    glutamines = [rng.uniform(380, 560) for _ in range(_N)]  # μmol/L — low-normal

    # Clinical flags
    n_hypoglycemia   = round(_N * 0.90)
    n_seizures       = round(_N * 0.80)
    n_absence        = round(_N * 0.65)   # most characteristic type
    n_dre            = round(_N * 0.18)
    n_idd            = round(_N * 0.60)
    n_diazoxide_resp = round(_N * 0.92)   # excellent diazoxide response
    n_leucine_sens   = round(_N * 0.88)   # protein/leucine-sensitive
    n_nbs            = round(_N * 0.35)   # NBS — hypoglycemia flagged if screened

    return {
        "gene": "GLUD1",
        "subtitle": (
            "Hyperinsulinism-Hyperammonemia (HHS) Syndrome / Hyperinsulinemic Hypoglycemia-5 (HHF5) "
            "(OMIM #606762) — GLUD1 Gain-of-Function; mitochondrial glutamate dehydrogenase; "
            "GTP-regulatory domain mutations → hyperactive enzyme → dual pathology: HYPOGLYCEMIA + HYPERAMMONEMIA"
        ),
        "chromosome": "10q23.3",
        "protein_size": "505 aa; mitochondrial matrix; NAD(P)H-dependent; homohexameric; GTP allosteric inhibitor domain",
        "omim_gene": "*138130",
        "omim_disease": "#606762",
        "prevalence": "~200–300 cases worldwide 2026 (rare; under-diagnosed in mild forms)",
        "inheritance": "Autosomal Dominant — GAIN-OF-FUNCTION (de novo ~60% | familial ~40%)",
        "cohort_n": _N,
        "cohort_seed": _SEED,

        "function": (
            "GLUD1 catalyses L-Glutamate + NAD(P)⁺ → α-Ketoglutarate + NH₄⁺ + NAD(P)H "
            "(oxidative deamination; reversible). Normally tightly regulated by GTP (allosteric INHIBITOR). "
            "GoF mutations reduce GTP-sensitivity → GLUD1 runs constitutively → excess α-KG in β-cells "
            "(→ insulin) + excess NH₄⁺ in liver (→ hyperammonemia). "
            "Leucine (BCAA) is an allosteric ACTIVATOR of GLUD1 at the same regulatory domain as GoF mutations "
            "→ explains the pathognomonic protein/leucine-sensitive hypoglycemia."
        ),
        "mechanism": (
            "HYPOGLYCEMIA: Excess α-KG → excess NADH → excess ATP in β-cells → K-ATP channels CLOSE "
            "→ membrane depolarisation → voltage-gated Ca²⁺ channels open → insulin EXOCYTOSIS → "
            "DISPROPORTIONATE INSULIN SECRETION relative to glucose → severe hypoglycemia. "
            "HYPERAMMONEMIA: Excess Glu oxidation → excess NH₄⁺ released from mitochondrial matrix → "
            "overwhelms urea cycle → persistent plasma ammonia 100–500 μmol/L. "
            "EPILEPSY: Dual mechanism — (1) recurrent hypoglycemia → brain energy failure → seizures; "
            "(2) hyperammonemia → glutamine-GABA imbalance → extrasynaptic GABA-A modulation → "
            "ABSENCE seizures (3 Hz spike-wave); (3) low synaptic glutamate → NMDA hypoactivation."
        ),
        "key_positive_features": (
            "Glucose CRITICALLY LOW (episodes <2.5 mmol/L), Insulin HIGH (>3 μU/mL inappropriate), "
            "Ammonia PERSISTENTLY ELEVATED (100–500 μmol/L), Glutamate LOW. "
            "Leucine/protein challenge POSITIVE (triggers hypoglycemia within 20 min). "
            "Diazoxide RESPONSIVE in 92%."
        ),
        "key_negative_features": (
            "alpha-AASA NORMAL (KEY vs ALDH7A1-PDE), Pipecolic NORMAL, MMA NORMAL (KEY vs MMUT/cblC), "
            "tHcy NORMAL (KEY vs CBS/MTHFR/MTR), Methionine NORMAL (KEY vs CBS/GNMT/MAT1A), "
            "Acylcarnitines NORMAL (KEY vs FAODs), Organic acids NORMAL, "
            "GAA NORMAL (KEY vs GAMT-AGAT), Sarcosine NORMAL, SAM NORMAL."
        ),

        "kpi": {
            "avg_glucose_mmol_l": round(sum(glucoses) / _N, 1),
            "avg_insulin_uU_ml":  round(sum(insulins) / _N, 1),
            "avg_ammonia_umol_l": round(sum(ammonias) / _N, 0),
            "avg_glutamate_umol_l": round(sum(glutamates) / _N, 1),
            "avg_glutamine_umol_l": round(sum(glutamines) / _N, 0),
            "pct_hypoglycemia":    round(100 * n_hypoglycemia / _N),
            "pct_seizures":        round(100 * n_seizures / _N),
            "pct_absence":         round(100 * n_absence / _N),
            "pct_dre":             round(100 * n_dre / _N),
            "pct_idd":             round(100 * n_idd / _N),
            "pct_diazoxide_resp":  round(100 * n_diazoxide_resp / _N),
            "pct_leucine_sens":    round(100 * n_leucine_sens / _N),
            "pct_nbs":             round(100 * n_nbs / _N),
        },

        "phenotype_distribution": phenotypes,

        "nbs_primary":   "Hypoglycemia flagged (if glucose <2.0 mmol/L at heel-prick) — ~35% detected",
        "nbs_secondary": "Ammonia (if measured) + leucine-stimulated insulin assay — NOT standard NBS panel",

        "pathway_position": {
            "step": "Glutamate → α-Ketoglutarate (TCA cycle entry / nitrogen disposal step)",
            "upstream": "L-Glutamate (major excitatory neurotransmitter; proline/ornithine precursor)",
            "downstream": "α-Ketoglutarate (TCA cycle; connects glutamate → energy; NH₄⁺ → urea cycle)",
            "position_summary": (
                "GLUD1 sits at the GLUTAMATE CATABOLISM NODE — central hub linking amino acid catabolism "
                "to TCA cycle. GoF → excess flux → β-cell energy surplus (insulin) + hepatic NH₄⁺ overload."
            ),
        },

        "vs_gamt": {
            "shared": "Both cause epilepsy + IDD via metabolic imbalance; both involve mitochondrial metabolism",
            "GLUD1": "GoF; ammonia HIGH (100–500 µmol/L); glucose LOW (hypoglycemia); glutamate LOW; creatine NORMAL",
            "GAMT":  "LOF; guanidinoacetate HIGH (50–300 µmol/L); creatine ABSENT; ammonia MILD; glucose NORMAL",
            "epilepsy": "GLUD1: absence dominant (65%), ammonia-driven; GAMT: DRE severe (60–80%), GAA-neurotoxicity",
        },
        "vs_aldh18a1": {
            "shared": "Both involve glutamate-related metabolic pathways; both AR-spectrum diseases with epilepsy",
            "GLUD1": "GoF; ammonia HIGH; glucose LOW; glutamate LOW; dominant inheritance; proline NORMAL",
            "ALDH18A1": "LOF; proline CRITICALLY LOW; ornithine LOW; cutis laxa; P5C low/normal; recessive",
            "epilepsy": "GLUD1: absence/generalized (75–85%); ALDH18A1: infantile spasms dominant (50–65%)",
        },
    }


def get_breakdown():
    rng = _rng()

    biomarkers = [
        {"name": "Blood glucose (episode nadir)",      "mean": 1.9,   "unit": "mmol/L",  "normal_range": "3.5–6.0 mmol/L",    "significance": "CRITICALLY LOW — hypoglycemia; KEY diagnostic trigger"},
        {"name": "Plasma insulin (during hypoglycemia)","mean": 14.2, "unit": "μU/mL",   "normal_range": "<3 μU/mL at glucose <2.5", "significance": "PATHOLOGICALLY HIGH — inappropriate secretion; confirms hyperinsulinism"},
        {"name": "Plasma ammonia",                     "mean": 287,   "unit": "μmol/L",  "normal_range": "<50 μmol/L",         "significance": "PERSISTENTLY ELEVATED — 100–500 µmol/L; PATHOGNOMONIC for HHS"},
        {"name": "Plasma glutamate",                   "mean": 31.4,  "unit": "μmol/L",  "normal_range": "40–120 μmol/L",      "significance": "LOW — hyperactive GLUD1 consumes glutamate; key diagnostic clue"},
        {"name": "Plasma glutamine",                   "mean": 475,   "unit": "μmol/L",  "normal_range": "400–700 μmol/L",     "significance": "LOW-NORMAL — partial compensatory synthesis insufficient"},
        {"name": "Plasma alanine",                     "mean": 218,   "unit": "μmol/L",  "normal_range": "170–450 μmol/L",     "significance": "LOW-NORMAL — transamination substrate depleted"},
        {"name": "alpha-KG / 2-OG (plasma)",           "mean": 28,    "unit": "μmol/L",  "normal_range": "10–20 μmol/L",       "significance": "MILDLY ELEVATED — GLUD1 product; confirms excess flux"},
        {"name": "Leucine challenge response",         "mean": None,  "unit": "POSITIVE 88% (glucose drop >30%)", "normal_range": "No significant drop", "significance": "Leucine allosteric activator of GLUD1 → diagnostic stress test"},
        {"name": "alpha-AASA (urine)",                 "mean": 0.8,   "unit": "mmol/mol Cr", "normal_range": "<3 mmol/mol Cr", "significance": "NORMAL — KEY NEGATIVE vs ALDH7A1 (PDE > 30)"},
        {"name": "Pipecolic acid (plasma)",            "mean": 0.9,   "unit": "μmol/L",  "normal_range": "<3 μmol/L",          "significance": "NORMAL — KEY NEGATIVE vs ALDH7A1 (PDE elevated)"},
        {"name": "MMA (urine)",                        "mean": 1.2,   "unit": "mmol/mol Cr", "normal_range": "<4 mmol/mol Cr", "significance": "NORMAL — KEY NEGATIVE vs MMUT/cblC"},
        {"name": "Total homocysteine (plasma)",        "mean": 7.8,   "unit": "μmol/L",  "normal_range": "<15 μmol/L",         "significance": "NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR/MTRR"},
        {"name": "Plasma methionine",                  "mean": 28,    "unit": "μmol/L",  "normal_range": "15–45 μmol/L",       "significance": "NORMAL — KEY NEGATIVE vs CBS (high) / GNMT (high)"},
        {"name": "GAA (plasma)",                       "mean": 1.8,   "unit": "μmol/L",  "normal_range": "<3 μmol/L",          "significance": "NORMAL — KEY NEGATIVE vs GAMT (50–300 µmol/L)"},
        {"name": "Creatine (plasma)",                  "mean": 42,    "unit": "μmol/L",  "normal_range": "20–80 μmol/L",       "significance": "NORMAL — KEY NEGATIVE vs GAMT/AGAT (absent creatine)"},
    ]

    clinical_features = [
        {"feature": "Hyperammonemia (NH₃ > 100 µmol/L)",   "pct": 100, "note": "UNIVERSAL — persistently elevated; does NOT require protein load"},
        {"feature": "Hypoglycemia episodes",                "pct": 90,  "note": "Glucose <2.5 mmol/L; fasting AND protein-sensitive"},
        {"feature": "Diazoxide response",                   "pct": 92,  "note": "EXCELLENT — 92% respond; hallmark of HHS; confirms K-ATP mechanism"},
        {"feature": "Leucine/protein sensitivity",          "pct": 88,  "note": "Pathognomonic — hypoglycemia within 20 min of protein load"},
        {"feature": "Epilepsy (overall)",                   "pct": 80,  "note": "Predominantly absence (65%) + febrile (30%) + GTCS (25%) + myoclonic (20%)"},
        {"feature": "Absence seizures",                     "pct": 65,  "note": "MOST CHARACTERISTIC — 3 Hz spike-wave; ammonia-driven GABA-A modulation"},
        {"feature": "Intellectual disability",              "pct": 60,  "note": "Mild-moderate (IQ 50–80 range); from recurrent hypoglycemia + hyperammonemia"},
        {"feature": "Learning difficulties / ADHD-like",    "pct": 60,  "note": "Attention and learning problems even without formal IDD criteria"},
        {"feature": "Febrile seizures",                     "pct": 30,  "note": "Often FIRST presentation — misdiagnosed as simple febrile convulsions"},
        {"feature": "GTCS",                                 "pct": 25,  "note": "Less frequent than absence; secondary generalisation from severe hypoglycemia"},
        {"feature": "Myoclonic seizures",                   "pct": 20,  "note": "May be hypoglycemia-triggered or true epileptic; 3-Hz Glx imbalance"},
        {"feature": "Drug-resistant epilepsy",              "pct": 18,  "note": "Less than other metabolic epilepsies; metabolic control (diazoxide) reduces seizures"},
        {"feature": "Macrocephaly / normal head circumference","pct": 78, "note": "Usually NORMAL head size — distinct from proline/creatine disorders"},
        {"feature": "MRI normal",                          "pct": 72,  "note": "Usually normal; hippocampal atrophy in 15% after prolonged hypoglycemia"},
        {"feature": "Cushing-like features (diazoxide SE)", "pct": 35,  "note": "Fluid retention, hirsutism — side effects of diazoxide; not disease feature"},
    ]

    variants = [
        {"variant": "p.Ser445Leu", "domain": "Antenna / GTP-regulatory exon 11", "freq_pct": 25, "phenotype": "Classic-Severe", "note": "Most common worldwide; reduces GTP inhibitory binding 10-fold"},
        {"variant": "p.Glu446Lys", "domain": "Antenna exon 11 adjacent",          "freq_pct": 20, "phenotype": "Classic-Severe", "note": "Adjacent to p.Ser445; severe GTP insensitivity; high ammonia"},
        {"variant": "p.Arg221Cys", "domain": "GTP-binding regulatory exon 7",     "freq_pct": 15, "phenotype": "Moderate",       "note": "GTP-binding domain; moderate phenotype; familial cases"},
        {"variant": "p.Gly446Val", "domain": "Antenna exon 11",                   "freq_pct": 12, "phenotype": "Moderate",       "note": "Val substitution less severe than Lys at same residue"},
        {"variant": "p.His454Tyr", "domain": "Pivot helix exon 11",               "freq_pct": 10, "phenotype": "Moderate",       "note": "Pivot helix mutation; intermediate GTP response; common in Asia"},
        {"variant": "p.Thr374Ile", "domain": "Pivot helix exon 10",               "freq_pct": 8,  "phenotype": "Moderate",       "note": "Pivot helix — allosteric transmission; moderate metabolic phenotype"},
        {"variant": "c.IVS9+1G>A","domain": "Splice — exon 9/10 junction",        "freq_pct": 6,  "phenotype": "Moderate-Severe", "note": "Rare splice variant; truncation of regulatory domain"},
        {"variant": "p.Ala445Val", "domain": "Antenna exon 11",                   "freq_pct": 4,  "phenotype": "Mild-Attenuated", "note": "Val much milder than Leu at 445; mild hypoglycemia; normal IDD"},
    ]

    seizure_types = [
        {
            "type": "Absence (3 Hz spike-wave)",
            "pct_in_seizure_pts": 65,
            "note": "MOST CHARACTERISTIC — ammonia-driven GABA-A modulation at extrasynaptic receptors; may appear as 'staring spells'; can precede diagnosis",
        },
        {
            "type": "Febrile seizures",
            "pct_in_seizure_pts": 30,
            "note": "Often FIRST event — commonly misdiagnosed; fever → catabolism → hypoglycemia → seizure; check ammonia + glucose in every febrile seizure",
        },
        {
            "type": "GTCS (generalised tonic-clonic)",
            "pct_in_seizure_pts": 25,
            "note": "Secondary from severe hypoglycemia or high ammonia; responds to metabolic control",
        },
        {
            "type": "Myoclonic",
            "pct_in_seizure_pts": 20,
            "note": "May be hypoglycemia-triggered (metabolic) or true epileptic; 3-Hz glutamine imbalance",
        },
        {
            "type": "Hypoglycemia-triggered seizures",
            "pct_in_seizure_pts": 50,
            "note": "Metabolic (not epileptic) — prevented by diazoxide/feeding; critical to distinguish from epileptic seizures",
        },
    ]

    treatments = [
        {"treatment": "Diazoxide (Proglycem)", "level": "Level A — PRIMARY",  "dose": "5–15 mg/kg/day in 2–3 divided doses",
         "mechanism": "K-ATP channel OPENER → counteracts GLUD1 GoF β-cell depolarisation → reduces insulin → prevents hypoglycemia",
         "contraindication": "None absolute; monitor fluid retention, pulmonary hypertension (reduce dose)"},
        {"treatment": "Leucine/protein restriction",       "level": "Level B",          "dose": "Leucine <50 mg/kg/day; avoid high-protein meals",
         "mechanism": "Reduces leucine allosteric activation of GLUD1 → lessens protein-triggered hypoglycemia",
         "contraindication": "Extreme restriction causes malnutrition — balanced restriction, not elimination"},
        {"treatment": "Octreotide (somatostatin analog)",  "level": "Level B",          "dose": "2–10 µg/kg/day SC or IV",
         "mechanism": "Somatostatin receptor activation → inhibits insulin secretion → backup when diazoxide inadequate",
         "contraindication": "GI side effects; tachyphylaxis with chronic use"},
        {"treatment": "Levetiracetam (LEV)",               "level": "Level B — AED",   "dose": "20–60 mg/kg/day divided",
         "mechanism": "SV2A modulation; first-line AED when seizures persist after metabolic control",
         "contraindication": "Behavioural side effects; monitor"},
        {"treatment": "Phenobarbital",                     "level": "Level B — AED",   "dose": "3–5 mg/kg/day",
         "mechanism": "GABA-A positive modulator; older option; may help absence-like component",
         "contraindication": "Sedation; cognitive impact with chronic use"},
        {"treatment": "High-protein diet / protein loading","level": "ABSOLUTE CONTRAINDICATION","dose": "N/A",
         "mechanism": "Provides leucine → allosteric GLUD1 activation → precipitates severe hypoglycemia",
         "contraindication": "OPPOSITE of most metabolic diseases — protein restriction needed here"},
        {"treatment": "L-Glutamate supplementation",       "level": "ABSOLUTE CONTRAINDICATION","dose": "N/A",
         "mechanism": "Feeds overactive GLUD1 → more α-KG → more insulin + more NH₄⁺",
         "contraindication": "Directly worsens both hypoglycemia and hyperammonemia"},
        {"treatment": "Valproate (VPA)",                   "level": "HIGH RISK",        "dose": "N/A",
         "mechanism": "Inhibits urea cycle + carnitine depletion → ammonia MARKEDLY WORSE (already 100–500)",
         "contraindication": "Pre-existing hyperammonemia makes VPA potentially catastrophic; avoid"},
        {"treatment": "Ethanol",                           "level": "HIGH RISK",        "dose": "N/A",
         "mechanism": "Raises NADH/NAD+ ratio → activates GLUD1 reductive direction → hypoglycemia + lactic acidosis",
         "contraindication": "Patients (and families) must be counselled re: alcohol risks"},
        {"treatment": "Ketogenic diet",                    "level": "MODERATE RISK",    "dose": "If used: very low protein-KD variant",
         "mechanism": "High fat reduces glucose need; BUT protein component triggers leucine-mediated hypoglycemia",
         "contraindication": "Standard KD (4:1 ratio with protein) contraindicated; specialist supervision essential"},
    ]

    return {
        "biomarkers": biomarkers,
        "clinical_features": clinical_features,
        "variants": variants,
        "seizure_types": seizure_types,
        "treatments": treatments,
    }


def get_definitions():
    return {
        "gene_full_name": "GLUD1 — Glutamate Dehydrogenase 1 (mitochondrial, ubiquitous)",
        "chromosome": "10q23.3",
        "gene_omim": "*138130",
        "disease_omim": "#606762",
        "disease_name": (
            "Hyperinsulinism-Hyperammonemia (HHS) Syndrome / "
            "Hyperinsulinemic Hypoglycemia type 5 (HHF5) / "
            "Familial Hyperinsulinism with Hyperammonemia"
        ),
        "inheritance": "Autosomal Dominant — GAIN-OF-FUNCTION (de novo ~60%; familial ~40%)",
        "protein": "505 aa; mitochondrial matrix; homohexameric; NAD(P)H-dependent; GTP allosteric inhibitor domain (exons 10-12 = 'antenna')",
        "reaction": "L-Glutamate + NAD(P)⁺ ⇌ α-Ketoglutarate + NH₄⁺ + NAD(P)H  [GoF → constitutively forward]",
        "pathway": (
            "Glutamate catabolism node: Glutamate → TCA cycle (α-KG) + urea cycle (NH₄⁺). "
            "Central to amino acid metabolism, nitrogen disposal, and β-cell energy sensing."
        ),
        "cohort_note": (
            f"Synthetic cohort n={_N}, seed={_SEED}. Biomarker values modelled on published HHS case series "
            "(Stanley et al. 1998, Hsu et al. 2001, Bahi-Buisson et al. 2008, Martínez et al. 2019). "
            "All patient data are simulated for dashboard demonstration — not real patients."
        ),
        "key_terms": {
            "Hyperinsulinism-Hyperammonemia (HHS) syndrome":
                "Dual pathology from GLUD1 GoF: excess insulin (hypoglycemia) + excess ammonia (hyperammonemia). "
                "First described by Stanley et al. 1998. Both features are ALWAYS present simultaneously.",
            "K-ATP channel (KATP)":
                "ATP-sensitive potassium channel (SUR1/Kir6.2) in pancreatic β-cells. "
                "Normally closes when ATP rises → depolarisation → insulin release. "
                "GLUD1 GoF → excess ATP from excess α-KG → permanent K-ATP closure → permanent insulin → hypoglycemia.",
            "GTP allosteric inhibitor domain (antenna)":
                "Exons 10–12 of GLUD1 encode the 'antenna' regulatory domain. GTP normally binds here to INHIBIT GLUD1. "
                "GoF mutations in this domain reduce GTP binding affinity → enzyme cannot be shut off.",
            "Leucine allosteric activation":
                "Leucine (BCAA) binds to the same regulatory domain as GoF mutations → allosteric ACTIVATOR. "
                "Pathognomonic: protein meal → leucine rise → GLUD1 activation → β-cell ATP spike → insulin → hypoglycemia.",
            "Diazoxide":
                "K-ATP channel OPENER (Kir6.2 activator). Counteracts GLUD1 GoF β-cell hyperdepolarisation. "
                "Primary treatment for HHS — 92% respond. Reduces insulin secretion without impairing counterregulation.",
            "Absence seizures (3 Hz spike-wave) in HHS":
                "Ammonia disrupts extrasynaptic GABA-A receptor function and alters glutamine/glutamate ratio → "
                "thalamo-cortical rhythm dysregulation → 3 Hz spike-wave discharge = absence phenotype. "
                "Most characteristic seizure type in GLUD1 HHS.",
            "Protein-sensitive hypoglycemia":
                "Hallmark of GLUD1 HHS — unlike congenital hyperinsulinism from other genes (KCNJ11, ABCC8) "
                "where protein does NOT trigger hypoglycemia. Leucine-stimulated insulin test is diagnostic.",
        },
        "differential_diagnosis": {
            "KATP-HI (ABCC8/KCNJ11 — congenital hyperinsulinism)":
                "Also causes hypoglycemia + high insulin. KEY DISTINCTION: ABCC8/KCNJ11 → ammonia NORMAL; "
                "NO leucine-sensitive hypoglycemia; AR or AD diffuse/focal; diazoxide response variable",
            "GAMT deficiency (GAMT gene)":
                "Also AR + epilepsy + IDD. KEY DISTINCTION: GAMT → GAA MARKEDLY HIGH (50–300); creatine ABSENT; "
                "glucose NORMAL; ammonia only mildly elevated; no hyperinsulinism",
            "Urea cycle disorders (OTC/CPS1/ASS1/ASL/ARG1)":
                "Also hyperammonemia. KEY DISTINCTION: UCD → ammonia from PROTEIN load only (not fasting); "
                "orotic acid HIGH (OTC); amino acid profile abnormal; glucose NORMAL; NO hyperinsulinism",
            "ALDH7A1 (Pyridoxine-Dependent Epilepsy)":
                "Also epilepsy + IDD. KEY DISTINCTION: ALDH7A1 → alpha-AASA MARKEDLY HIGH; pipecolic HIGH; "
                "glucose NORMAL; ammonia NORMAL; responds to B6 (GLUD1 does NOT)",
            "Medium-chain acyl-CoA dehydrogenase deficiency (MCAD)":
                "Also hypoglycemia (fasting). KEY DISTINCTION: MCAD → acylcarnitine C8/C10 HIGH; "
                "no hyperinsulinism; no hyperammonemia; organic acids (suberate, sebacate)",
        },
        "treatment_summary": {
            "level_a_first_line": "Diazoxide 5–15 mg/kg/day (K-ATP opener; 92% response)",
            "level_b_adjunct": "Leucine/protein restriction + Octreotide (backup) + LEV (AED)",
            "absolute_contraindications": "High-protein diet (triggers leucine-HHS); L-Glutamate supplements (feeds GLUD1)",
            "high_risk": "Valproate (worsens hyperammonemia markedly); Ethanol (activates GLUD1)",
            "moderate_risk": "Ketogenic diet (protein component); Vigabatrin (GABA-T inhibition + ammonia)",
            "monitoring": "Glucose (CGMS ideal), ammonia (monthly), liver function, EEG (if absence suspected)",
            "inheritance_note": "AD GoF — 50% offspring risk if familial; genetic testing siblings + parents",
        },
    }
