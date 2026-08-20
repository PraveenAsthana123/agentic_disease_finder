#!/usr/bin/env python3
"""PHYH (Phytanoyl-CoA Hydroxylase) Deficiency — Adult Refsum Disease (ARD) Epilepsy Dashboard.

PHYH encodes phytanoyl-CoA 2-hydroxylase (phytanoyl-CoA dioxygenase), a 338 aa peroxisomal
matrix enzyme that catalyses the FIRST STEP of alpha-oxidation of phytanic acid.

PEROXISOMAL ALPHA-OXIDATION PATHWAY — PHYH position:
  Substrate: Phytanic acid (3,7,11,15-tetramethylhexadecanoic acid) from diet (dairy, ruminant fat)
  ─ Step 1: PHYH (PTS2; 10p13): Phytanoyl-CoA + 2-oxoglutarate + O₂ →
              2-hydroxyphytanoyl-CoA + succinate + CO₂  [2-OG dioxygenase]
  ─ Step 2: HACL1: 2-Hydroxyphytanoyl-CoA lyase → pristanal + formyl-CoA
  ─ Step 3: ALDH3A2/Prox-ALDH: Pristanal → pristanic acid (activated to pristanoyl-CoA)
  ─ Then: AMACR (racemization) → ACOX2 (Step1 PeroxBO) → HSD17B4 (Steps2+3) → SCP2 (Step4)
  ─────────────────────────────────────────────────────────────────────────────────────────────
  PHYH = FIRST COMMITTED STEP of alpha-oxidation. Without PHYH, phytanic acid ACCUMULATES.
  Pristanic acid cannot form (alpha-oxidation blocked upstream) — PRISTANIC NORMAL or mildly elevated.
  Beta-oxidation of straight-chain VLCFA (ACOX1) is UNAFFECTED → VLCFA NORMAL.
  Plasmalogen synthesis (PEX7/GNPAT/AGPS/FAR1) is UNAFFECTED → PLASMALOGENS NORMAL.

BIOCHEMICAL PROFILE (PHYH deficiency / Adult Refsum Disease):
  Phytanic acid:  SEVERELY ELEVATED — alpha-oxidation blocked at step 1 (major biomarker)
  Pristanic acid: NORMAL to mildly elevated — not formed (alpha-ox blocked before pristanic)
  VLCFA (C26:0):  NORMAL — straight-chain ACOX1 pathway unaffected
  Plasmalogens:   NORMAL — PTS2/plasmalogen biosynthesis unaffected (PEX7 intact in PHYH disease)
  Pipecolic acid: NORMAL — PTS1 import intact (excludes ZSD)
  THCA / DHCA:    NORMAL — bile acid beta-oxidation branch unaffected (AMACR/SCP2/HSD17B4 intact)

CRITICAL BIOCHEMICAL DISTINCTIONS:
  PHYH vs AMACR:
    Phytanic acid SEVERELY ELEVATED in PHYH — NORMAL in AMACR (key marker).
    Pristanic acid ELEVATED in AMACR (racemase block) — NORMAL/mild in PHYH (alpha-ox blocked before pristanic forms).
    THCA/DHCA ELEVATED in AMACR — NORMAL in PHYH.
    Clinical overlap: both adult-onset; both neuropathy + retinopathy.
    PHYH: anosmia ~70% (rare in AMACR); seizures ~20% (less than AMACR ~60%); cardiac arrhythmia ~40%.
    Plasma phytanic acid measurement is the DEFINING distinction.
  PHYH vs PEX7 (RCDP1):
    PEX7 (RCDP1): phytanic acid ELEVATED (PHYH depends on PEX7 for import) + plasmalogens SEVERELY LOW.
    PHYH: plasmalogens NORMAL (PEX7 intact; PHYH itself is the deficient enzyme).
    PEX7/RCDP1: rhizomelia + stippled epiphyses + cataracts (neonatal severe).
    PHYH/ARD: NO rhizomelia; adult-onset; milder systemic features.
    RBC plasmalogens MANDATORY — if low → PEX7/RCDP1 not PHYH.
  PHYH vs ZSD (PEX1/PEX6):
    ZSD: ALL peroxisomal pathways impaired (VLCFA + phytanic + plasmalogens + pristanic ALL abnormal).
    PHYH: isolated phytanic elevation — VLCFA, plasmalogens, pristanic, pipecolic all NORMAL.
    ZSD: neonatal/infantile onset; PHYH/ARD: adolescent/adult onset.
  PHYH vs ABCD1 (X-ALD):
    ABCD1: VLCFA C26:0 SEVERELY ELEVATED — NORMAL in PHYH.
    ABCD1: adrenal insufficiency (ABSOLUTE CI PHT/CBZ) — PHYH: NO adrenal disease.
    ABCD1: phytanic NORMAL; PHYH: phytanic SEVERELY ELEVATED.
  PHYH vs SCP2/HSD17B4:
    SCP2/HSD17B4: pristanic + THCA ELEVATED — NORMAL in PHYH.
    Both VLCFA NORMAL; adult or mixed onset.
    Phytanic acid profile is the key separation.
  VGB in PHYH vs other peroxisomal disorders:
    PHYH: VGB ABSOLUTE CI — RP in ~95% (near-universal). VGB irreversibly worsens VF.
    AMACR: VGB RELATIVE CI — RP ~45%.
    SCP2: VGB RELATIVE CI — RP ~20%.
    PHYH carries the HIGHEST VGB visual risk of ALL peroxisomal epilepsy disorders.

PHYH PROTEIN BIOLOGY:
  Gene PHYH (also PAHX) at 10p13 (OMIM Gene: *602026).
  338 aa; peroxisomal matrix enzyme. PTS2 targeting signal (N-terminal RLx₅HL motif).
  Enzyme class: 2-oxoglutarate-dependent dioxygenase (iron(II) and 2-OG cofactors).
  Reaction: phytanoyl-CoA + 2-oxoglutarate + O₂ → 2-hydroxyphytanoyl-CoA + succinate + CO₂.
  Phytanic acid is a 3-methyl-branched chain fatty acid obtained exclusively from diet
  (chlorophyll degradation → phytol; dairy/ruminant fat/fish; cannot be synthesized by humans).
  Expressed: liver > kidney > brain; detectable most tissues.
  NOTE: Second gene for ARD: PEX7 — but PEX7 mutations also cause RCDP1 (more severe).

OMIM: Gene *602026 (PHYH) | Disease #266500 (Refsum Disease / ARD)
LOCUS: 10p13
EPIDEMIOLOGY: ~200 cases reported worldwide 2026 (more common than AMACR/SCP2).
  First gene: Jansen et al. 1997 (Nature Genetics) — cloned PHYH.
  Classic adult-onset (teens to 40s; range childhood to 60s).
  Second locus: PEX7 mutations (Zellweger spectrum / RCDP1 — usually more severe).
  Carrier frequency estimated ~1/200,000 births (rare but more common than AMACR).

CLINICAL PHENOTYPE (Adult Refsum Disease / PHYH deficiency):
  Adolescent/adult-onset (typically teens to early 40s).
  Core tetrad: RP + cerebellar ataxia + peripheral neuropathy + elevated CSF protein.
  Full spectrum:
  • Retinitis pigmentosa (RP) — ~95%: rod-cone dystrophy, progressive; night blindness first
  • Cerebellar ataxia — ~90%: gait ataxia, dysmetria, intention tremor
  • Peripheral polyneuropathy — ~90%: hypertrophic demyelinating (CSF protein elevated)
  • Anosmia / hyposmia — ~70%: early symptom; pathognomonic clinical clue
  • Sensorineural hearing loss (SNHL) — ~50%
  • Cardiac conduction defects — ~40%: AV block, arrhythmia; LIFE-THREATENING
  • Epilepsy — ~20%: less prominent than AMACR/SCP2; multifocal, myoclonic
  • Ichthyosis — ~25%: mildly scaly skin; lamellar
  • Epiphyseal dysplasia — ~30%: asymmetric shortening of metacarpals/metatarsals
  • Cognitive decline — ~35%: later-stage; mild to moderate
  • Drug resistance: ~30% of epilepsy patients
  FASTING HAZARD: phytanic acid stored in adipose; illness/fasting → lipolysis → phytanic surge
  → acute polyneuropathy/cardiac arrhythmia/visual deterioration (phytanic crisis).

ERT/HSCT/GENE THERAPY:
  No ERT (2026) — PHYH is a peroxisomal matrix enzyme (PTS2); not secreted; no delivery route.
  No HSCT — ARD is NOT inflammatory demyelination; HSCT targets only inflammatory demyelinators.
  No gene therapy approved 2026.

DIETARY THERAPY (GOLD STANDARD):
  Phytol-restricted diet: Level A — REDUCES PHYTANIC ACID INTAKE DIRECTLY.
    Restrict: dairy fat (butter, cream, whole milk, cheese), ruminant fat (beef, lamb tallow),
    oily fish (herring, tuna, cod liver oil), chlorophyll-rich foods (spinach in large amounts).
    Phytol is cleaved from chlorophyll by gut bacteria → phytanic acid in the portal circulation.
    Compliant patients achieve 50–80% reduction in plasma phytanic acid.
    Practical target: plasma phytanic acid < 200 µmol/L (ideally < 50 µmol/L).
  Plasmapheresis / LDL apheresis: Level B — for acute phytanic crisis or rapid pre-treatment reduction.
  DHA supplementation: NOT routinely indicated (alpha-oxidation does not impair DHA synthesis;
    peroxisomal beta-oxidation of DHA intermediates is via HSD17B4/ACOX2, not PHYH).
  Fasting prevention: IV 10% dextrose MANDATORY during any illness, surgery, or prolonged fasting.
    Adipose lipolysis during fasting releases accumulated phytanic acid → systemic crisis.

TREATMENT PRINCIPLES:
  LEV first-line for seizures (no peroxisomal interactions; safe in adult-onset disease).
  PHT/CBZ: CAN USE for focal seizures — NO adrenal insufficiency in PHYH (contrast ABCD1 ABSOLUTE CI).
  VGB: ABSOLUTE CI — RP in ~95% of PHYH patients. VGB causes irreversible peripheral VF constriction.
    COMBINED: RP + VGB = catastrophic, irreversible additive visual loss. ABSOLUTE CI in ARD.
    This is THE STRONGEST VGB contraindication in any peroxisomal epilepsy disorder.
  VPA: RELATIVE CI — hepatotoxicity risk (liver processes phytanic acid load); POLG1 MANDATORY CPIC A.
  Clonazepam: adjunct for ataxia/myoclonus (Level C).
  Lorenzo's Oil: NOT APPLICABLE — VLCFA normal; oil targets VLCFA elongation (ABCD1 mechanism).
  Cardiac monitoring: Holter ECG, echocardiography — AV block/arrhythmia life-threatening.
  No ERT. No HSCT. Diet is the primary disease-modifying therapy.
"""

import random
random.seed(62)


# ── Overview (KPIs + summary) ─────────────────────────────────────────────────

def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 20,
        "retinitis_pigmentosa_pct": 95,
        "cerebellar_ataxia_pct": 90,
        "polyneuropathy_pct": 90,
        "anosmia_pct": 70,
        "snhl_pct": 50,
        "cardiac_pct": 40,
        "ichthyosis_pct": 25,
        "epiphyseal_dysplasia_pct": 30,
        "cognitive_decline_pct": 35,
        "drug_resistance_pct": 30,
        "phytanic_elevated_pct": 100,
        "pristanic_normal_pct": 88,
        "vlcfa_normal_pct": 98,
        "plasmalogen_normal_pct": 97,
        "pipecolic_normal_pct": 99,
        "adrenal_insufficiency_pct": 0,
        "omim_gene": "602026",
        "omim_disease": "266500",
        "locus": "10p13",
        "vlcfa_elevated": False,
        "phytanic_elevated": True,
        "inheritance": (
            "Autosomal recessive (AR), biallelic LOF — both sexes equally; "
            "second gene: PEX7 (RCDP1 + phytanic elevation — distinguished by rhizomelia + plasmalogens LOW)"
        ),
        "common_variant": (
            "No single dominant founder mutation in PHYH; >50 disease-causing variants described. "
            "~200 cases worldwide 2026; more common than AMACR/SCP2 (~25-30) but still ultra-rare."
        ),
        "onset_age": (
            "Adolescent/adult-onset (typically teens to early 40s, range childhood to 60s) — "
            "EARLIER than AMACR/SCP2 (~20s-60s); MUCH LATER than ACOX1/HSD17B4 (neonatal)"
        ),
        "disease_mechanism": (
            "PHYH (phytanoyl-CoA 2-hydroxylase, 338 aa, PTS2-targeted) catalyses STEP 1 of "
            "alpha-oxidation: phytanoyl-CoA + 2-oxoglutarate + O₂ → 2-hydroxyphytanoyl-CoA "
            "+ succinate + CO₂. Without PHYH, phytanic acid CANNOT undergo alpha-oxidation and "
            "accumulates to severely toxic levels (>500 µmol/L plasma in untreated patients). "
            "Phytanic acid integrates into myelin and biological membranes, disrupting "
            "structure — mechanism of neuropathy, retinopathy, and cardiac toxicity. "
            "VLCFA, plasmalogens, pristanic acid, pipecolic acid ALL NORMAL — uniquely isolated "
            "alpha-oxidation defect. Dietary phytol → phytanic acid via gut bacteria; "
            "dietary restriction is the primary treatment."
        ),
        "critical_distinctions": [
            "PHYTANIC ACID SEVERELY ELEVATED (>200 µmol/L plasma) — the defining biomarker; NORMAL in AMACR/SCP2/HSD17B4/ZSD",
            "VLCFA NORMAL — excludes ABCD1 (X-ALD), ZSD, HSD17B4, ACOX1; phytanic is the ONLY elevated peroxisomal metabolite",
            "PLASMALOGENS NORMAL — excludes ZSD (PEX1/PEX6) and ALL RCDP types (PEX7/GNPAT/AGPS/FAR1)",
            "PRISTANIC ACID NORMAL — excludes AMACR, SCP2, HSD17B4 (branched-chain beta-oxidation diseases)",
            "VGB ABSOLUTE CI — RP ~95% in ARD; VGB irreversible VF loss = CATASTROPHIC additive damage",
            "ANOSMIA/HYPOSMIA ~70% — pathognomonic clinical clue; rare in other peroxisomal disorders",
            "FASTING HAZARD: adipose phytanic mobilization → phytanic crisis (cardiac/neurological); IV dextrose MANDATORY",
            "PHYTOL-RESTRICTED DIET = GOLD STANDARD (Level A) — only treatment that actually reduces phytanic acid",
            "EPILEPSY LESS PROMINENT (~20%) than AMACR (~60%) or SCP2 (~40%) — RP + ataxia + neuropathy dominate",
            "PEX7 second gene for ARD: PEX7 mutations also elevate phytanic BUT cause RCDP (rhizomelia + low plasmalogens)",
            "CARDIAC MONITORING ESSENTIAL — AV block/arrhythmia in ~40%; potentially life-threatening",
            "No ERT (PTS2 matrix enzyme, not secreted). No HSCT (non-inflammatory). No Lorenzo's oil (VLCFA normal)",
        ],
        "nbs_positive_rate": (
            "Plasma phytanic acid >10 µmol/L (normal <10; ARD typically >200 µmol/L untreated). "
            "Not on standard NBS panels. Diagnosis usually delayed 5–15 years from first symptoms. "
            "RP onset often precedes diagnosis by years (misdiagnosed as isolated retinitis pigmentosa)."
        ),
    }


# ── Breakdown (patients + seizure types + triggers + monitoring + lifecycle + treatments) ──

def get_breakdown():
    # 3 phenotypic classes
    phenotypic_classes = [
        {
            "class": "Classic Tetrad (RP + Ataxia + Neuropathy + CSF protein)",
            "pct": 55,
            "count": 22,
            "description": (
                "All 4 classic features present: retinitis pigmentosa, cerebellar ataxia, "
                "peripheral polyneuropathy, elevated CSF protein (>100 mg/dL). "
                "Anosmia in ~75%. Cardiac involvement in ~45%. Seizures in ~18%. "
                "Most common presentation — classic Refsum disease phenotype."
            ),
            "seizure_control": "~75% controlled on LEV monotherapy; ~25% drug resistant (multifocal/myoclonic)",
        },
        {
            "class": "Attenuated (RP-Dominant + Mild Neuropathy)",
            "pct": 30,
            "count": 12,
            "description": (
                "RP as presenting and predominant feature; neuropathy mild or absent initially. "
                "Anosmia common (~65%). Cardiac monitoring critical even in attenuated cases. "
                "Seizures in ~15%. Often misdiagnosed as isolated hereditary RP for years. "
                "Phytanic acid elevated but may be lower than classic tetrad."
            ),
            "seizure_control": "~80% seizure-free or controlled; mild drug resistance",
        },
        {
            "class": "Cardiac-Prominent (Arrhythmia + RP + Ataxia)",
            "pct": 15,
            "count": 6,
            "description": (
                "Cardiac arrhythmia/AV block as life-threatening dominant feature. "
                "RP + cerebellar ataxia present. Seizures in ~25% (often myoclonic). "
                "Fasting crisis episodes most severe in this class — IV dextrose critical. "
                "Pacemaker implantation in some patients. Drug resistance ~35% in seizure subset."
            ),
            "seizure_control": "~65% controlled; higher drug resistance than other classes",
        },
    ]

    # 40 synthetic patients
    sz_types_pool = [
        "Multifocal myoclonic", "Focal occipital", "Focal-BTCS", "Myoclonic-atonic", "Absence-like"
    ]
    pheno_dist = (
        ["Classic Tetrad"] * 22 +
        ["Attenuated (RP-Dominant)"] * 12 +
        ["Cardiac-Prominent"] * 6
    )
    patients = []
    for i in range(40):
        sex = random.choice(["Male", "Female"])
        pheno = pheno_dist[i]
        onset = random.randint(10, 42)
        has_seizures = random.random() < 0.20
        sz_type = random.choice(sz_types_pool) if has_seizures else "None"
        drug_resistant = has_seizures and random.random() < 0.30
        has_rp = random.random() < 0.95
        has_ataxia = random.random() < 0.90
        has_neuropathy = random.random() < 0.90
        has_anosmia = random.random() < 0.70
        has_cardiac = random.random() < 0.40
        phytanic = random.uniform(180, 850)
        pristanic = random.uniform(1.0, 12.0)  # normal to mildly elevated
        vlcfa_c26 = random.uniform(0.45, 0.85)  # normal range
        patients.append({
            "id": i + 1,
            "phenotypic_class": pheno,
            "sex": sex,
            "onset_age": onset,
            "primary_seizure": sz_type,
            "drug_resistant": drug_resistant,
            "has_rp": has_rp,
            "has_ataxia": has_ataxia,
            "has_neuropathy": has_neuropathy,
            "has_anosmia": has_anosmia,
            "has_cardiac": has_cardiac,
            "phytanic_umol_L": round(phytanic, 1),
            "pristanic_umol_L": round(pristanic, 1),
            "vlcfa_c26_umol_L": round(vlcfa_c26, 2),
        })

    seizure_types = [
        {
            "type": "Multifocal myoclonic",
            "pct": 40,
            "notes": "Most common type in ARD; correlates with cerebellar ataxia; responds to LEV/CLZ",
        },
        {
            "type": "Focal occipital (RP-related)",
            "pct": 30,
            "notes": "Visual aura; cortical spread from occipital cortex damaged by phytanic lipid accumulation",
        },
        {
            "type": "Focal-BTCS",
            "pct": 20,
            "notes": "Secondary generalisation from focal onset; multifocal EEG correlate",
        },
        {
            "type": "Myoclonic-atonic",
            "pct": 15,
            "notes": "Atonic component associated with cerebellar dysfunction; CLZ adjunct helpful",
        },
        {
            "type": "Absence-like",
            "pct": 10,
            "notes": "Brief staring spells; EEG shows diffuse slowing rather than 3 Hz SW; LEV effective",
        },
    ]

    triggers = [
        {
            "trigger": "Fasting / acute illness (phytanic surge)",
            "pct": 75,
            "mechanism": "Adipose lipolysis releases stored phytanic acid → plasma spike → acute neurological decompensation; IV dextrose MANDATORY",
        },
        {
            "trigger": "Sleep deprivation",
            "pct": 50,
            "mechanism": "Standard lowering of seizure threshold; combined with CNS phytanic toxicity",
        },
        {
            "trigger": "Dietary phytol load (non-compliant diet)",
            "pct": 60,
            "mechanism": "Large phytol intake → gut bacteria → phytanic acid surge; diet compliance critical",
        },
        {
            "trigger": "Fever / intercurrent infection",
            "pct": 45,
            "mechanism": "Increased metabolic demand + catabolism → phytanic mobilisation from fat stores",
        },
        {
            "trigger": "Cardiac arrhythmia episode",
            "pct": 25,
            "mechanism": "AV block/arrhythmia → transient cerebral hypoperfusion → lowers seizure threshold",
        },
        {
            "trigger": "Surgery / anaesthesia",
            "pct": 20,
            "mechanism": "Fasting pre-operatively → phytanic surge; IV dextrose and phytanic monitoring mandatory peri-operatively",
        },
        {
            "trigger": "Missed AED dose",
            "pct": 35,
            "mechanism": "Standard AED non-compliance effect; higher risk in drug-resistant subset",
        },
    ]

    monitoring = [
        {"parameter": "Plasma phytanic acid (µmol/L)", "frequency": "Every 3 months (target <200; ideally <50)", "reason": "Primary disease biomarker; guides diet compliance"},
        {"parameter": "Plasma pristanic acid", "frequency": "Every 6 months", "reason": "Should be NORMAL; elevation suggests misdiagnosis or co-pathology"},
        {"parameter": "VLCFA panel (C26:0)", "frequency": "Baseline; repeat if diagnosis questioned", "reason": "MUST remain NORMAL; elevation suggests ABCD1/ZSD co-pathology"},
        {"parameter": "RBC plasmalogens", "frequency": "Baseline", "reason": "MUST remain NORMAL; low → PEX7/RCDP1 not PHYH"},
        {"parameter": "ERG (electroretinogram) + visual fields", "frequency": "Every 6 months", "reason": "RP monitoring; VGB ABSOLUTE CI — baseline MANDATORY before any AED change"},
        {"parameter": "Holter ECG / 24h monitoring", "frequency": "Every 6–12 months", "reason": "Cardiac arrhythmia/AV block life-threatening; pacemaker if needed"},
        {"parameter": "Echocardiogram", "frequency": "Annually", "reason": "Cardiomyopathy surveillance; phytanic acid disrupts cardiac myocyte membranes"},
        {"parameter": "Nerve conduction studies / EMG", "frequency": "Annually", "reason": "Polyneuropathy progression monitoring; hypertrophic demyelinating pattern"},
        {"parameter": "LFTs (liver function)", "frequency": "Every 3 months if on VPA (RELATIVE CI)", "reason": "VPA hepatotoxicity + hepatic phytanic processing burden"},
        {"parameter": "Audiogram (pure tone / SNHL)", "frequency": "Annually", "reason": "SNHL ~50%; progressive; early hearing aid/cochlear implant planning"},
    ]

    lifecycle = [
        {
            "stage": "Pre-diagnosis / Diagnostic Delay",
            "description": "Average 5–15 years from first symptoms to diagnosis. RP often misdiagnosed as isolated hereditary RP. Anosmia ignored. Diagnosis triggered by elevated phytanic acid on peroxisomal screen.",
        },
        {
            "stage": "Acute Crisis (Phytanic Surge)",
            "description": "Triggered by illness/fasting. Acute worsening of neuropathy, cardiac arrhythmia, visual deterioration. Plasmapheresis Level B; IV dextrose MANDATORY. Seizure risk elevated during crisis.",
        },
        {
            "stage": "Diet-Controlled Stable Phase",
            "description": "Strict phytol-restricted diet; plasma phytanic <200 µmol/L. Neurological stabilisation or slow improvement. RP continues to progress despite diet. Cardiac monitoring ongoing.",
        },
        {
            "stage": "Progressive Neurological Phase",
            "description": "RP progression despite diet compliance (structural damage pre-existing). Cerebellar ataxia worsens slowly. Neuropathy may stabilise or plateau. Epilepsy management if seizures present.",
        },
        {
            "stage": "Advanced / Late Phase",
            "description": "Severe visual impairment (many patients legally blind). Significant ataxia. Cardiac pacemaker in some. Cognitive decline ~35%. Multidisciplinary support: ophthalmology, cardiology, neurology.",
        },
    ]

    treatments = [
        {
            "name": "Phytol-restricted diet",
            "type": "Disease-modifying (dietary)",
            "status": "GOLD STANDARD (Level A)",
            "mechanism": "Restricts dietary phytol input → reduces gut-derived phytanic acid production; target plasma phytanic <200 µmol/L (ideally <50 µmol/L)",
            "contra": "None — primary therapy; must be maintained lifelong",
        },
        {
            "name": "Plasmapheresis / LDL apheresis",
            "type": "Disease-modifying (procedural)",
            "status": "Level B (acute crisis / rapid reduction)",
            "mechanism": "Removes phytanic acid from plasma; rapid reduction in acute phytanic crisis; also used pre-operatively",
            "contra": "Procedural risks; not sustainable long-term — bridge to dietary control",
        },
        {
            "name": "IV 10% Dextrose (fasting protocol)",
            "type": "Crisis prevention",
            "status": "MANDATORY during illness/fasting/surgery",
            "mechanism": "Prevents adipose lipolysis → blocks phytanic acid release from fat stores during catabolic states",
            "contra": "Hyperglycaemia risk — standard glucose monitoring; but fasting is more dangerous",
        },
        {
            "name": "LEV (Levetiracetam)",
            "type": "AED",
            "status": "FIRST-LINE for seizures",
            "mechanism": "SV2A modulator; no peroxisomal or hepatic interactions; safe in ARD",
            "contra": "Behavioural side effects; monitor in cerebellar/cognitive impairment patients",
        },
        {
            "name": "Clonazepam",
            "type": "AED / Adjunct",
            "status": "Level C (myoclonus/ataxia adjunct)",
            "mechanism": "GABA-A modulator; myoclonic seizures + cerebellar ataxia; sedation risk in ataxic patients",
            "contra": "Sedation — worsens ataxia if over-dosed; titrate slowly",
        },
        {
            "name": "PHT (Phenytoin)",
            "type": "AED",
            "status": "CAN USE (no adrenal CI)",
            "mechanism": "Sodium channel blocker; NO adrenal mechanism — distinct from ABCD1 where ABSOLUTE CI",
            "contra": "Avoid in severe ataxia (cerebellar toxicity); CYP450 interactions",
        },
        {
            "name": "CBZ (Carbamazepine)",
            "type": "AED",
            "status": "CAN USE (no adrenal CI)",
            "mechanism": "Sodium channel blocker; focal seizures; no adrenal insufficiency in PHYH",
            "contra": "Monitor LFTs; hyponatraemia risk",
        },
        {
            "name": "VPA (Valproate)",
            "type": "AED",
            "status": "RELATIVE CI",
            "mechanism": "VPA hepatotoxicity + hepatic phytanic/bile acid metabolic burden; POLG1 MANDATORY CPIC A before prescribing",
            "contra": "Hepatotoxicity, POLG1 carriers ABSOLUTE CI; avoid as first-line in ARD",
        },
        {
            "name": "VGB (Vigabatrin)",
            "type": "AED",
            "status": "ABSOLUTE CI — RP ~95%",
            "mechanism": "GABA transaminase inhibitor; irreversible peripheral visual field constriction in 30-40% of all users",
            "contra": "ABSOLUTE CI in ARD: RP in ~95% patients + VGB = catastrophic irreversible additive visual loss. STRONGEST VGB CI of all peroxisomal epilepsy disorders. NEVER USE in ARD.",
        },
        {
            "name": "Lorenzo's Oil",
            "type": "Dietary supplement",
            "status": "NOT APPLICABLE",
            "mechanism": "Erucic + oleic acid → competitive inhibition of VLCFA elongase. VLCFA NORMAL in PHYH — no mechanism benefit",
            "contra": "NOT applicable; VLCFA normal",
        },
        {
            "name": "ERT (Enzyme replacement therapy)",
            "type": "Biological therapy",
            "status": "NO ERT available (2026)",
            "mechanism": "PHYH is a peroxisomal matrix enzyme (PTS2-targeted); not secreted; no systemic delivery feasible",
            "contra": "Not applicable — no approved or in-development ERT for PHYH 2026",
        },
        {
            "name": "HSCT (Bone marrow transplant)",
            "type": "Cellular therapy",
            "status": "NO HSCT",
            "mechanism": "HSCT targets only inflammatory demyelinating disorders (ABCD1 cerebral form). ARD is metabolic/non-inflammatory",
            "contra": "Not indicated. Non-inflammatory disease mechanism",
        },
        {
            "name": "Cardiac management (pacemaker / antiarrhythmic)",
            "type": "Cardiac intervention",
            "status": "As clinically indicated",
            "mechanism": "AV block/arrhythmia in ~40%; pacemaker implantation in severe cases; antiarrhythmic drugs as bridging",
            "contra": "Antiarrhythmic drug interactions with AEDs — monitor",
        },
    ]

    return {
        "phenotypic_classes": phenotypic_classes,
        "patients": patients,
        "seizure_types": seizure_types,
        "triggers": triggers,
        "monitoring": monitoring,
        "lifecycle": lifecycle,
        "treatments": treatments,
    }


# ── Definitions ───────────────────────────────────────────────────────────────

def get_definitions():
    return {
        "key_concepts": [
            {
                "term": "PHYH — Phytanoyl-CoA 2-Hydroxylase",
                "definition": (
                    "PHYH (gene: PHYH/PAHX, 10p13, OMIM *602026) encodes a 338 aa 2-oxoglutarate-dependent "
                    "dioxygenase that catalyses the FIRST STEP of peroxisomal alpha-oxidation: "
                    "phytanoyl-CoA + 2-oxoglutarate + O₂ → 2-hydroxyphytanoyl-CoA + succinate + CO₂. "
                    "PTS2 targeted (N-terminal RLx₅HL motif). Deficiency causes Adult Refsum Disease (ARD). "
                    "~200 cases worldwide 2026 (more common than AMACR/SCP2 but still ultra-rare)."
                ),
            },
            {
                "term": "Biochemical Signature",
                "definition": (
                    "Phytanic acid SEVERELY ELEVATED (>200 µmol/L untreated, normal <10 µmol/L). "
                    "Pristanic acid NORMAL to mildly elevated (alpha-oxidation blocked BEFORE pristanic forms). "
                    "VLCFA (C26:0) NORMAL — straight-chain ACOX1 pathway unaffected. "
                    "Plasmalogens NORMAL — PTS2/plasmalogen biosynthesis via PEX7/GNPAT/AGPS intact. "
                    "THCA/DHCA NORMAL — bile acid beta-oxidation branch (AMACR/HSD17B4/SCP2) intact. "
                    "Isolated alpha-oxidation defect: ALL other peroxisomal markers NORMAL."
                ),
            },
            {
                "term": "Alpha-oxidation — the PHYH Step",
                "definition": (
                    "Phytanic acid (3-methyl-branched chain) CANNOT undergo direct beta-oxidation "
                    "(methyl group at C3 position blocks beta-oxidation enzymes). "
                    "Alpha-oxidation removes ONE CARBON first: phytanic acid → via PHYH alpha-hydroxylation "
                    "→ 2-hydroxyphytanoyl-CoA → HACL1 lyase → pristanal → ALDH3A2 → pristanic acid. "
                    "Then pristanic acid undergoes branched-chain beta-oxidation (AMACR→ACOX2→HSD17B4→SCP2). "
                    "PHYH deficiency blocks the ENTIRE alpha-oxidation pathway — phytanic accumulates. "
                    "Pristanic acid CANNOT be formed from phytanic (pathway blocked upstream)."
                ),
            },
            {
                "term": "Phytanic Acid — Source and Toxicity",
                "definition": (
                    "Phytanic acid is a 3,7,11,15-tetramethylhexadecanoic acid obtained EXCLUSIVELY FROM DIET "
                    "(humans cannot synthesize it). Sources: dairy fat (butter, cream, cheese), "
                    "ruminant fat (beef, lamb), oily fish (herring, tuna, cod liver oil). "
                    "Chlorophyll contains phytol → gut bacteria cleave phytol → phytanic acid. "
                    "Toxicity mechanism: phytanic acid integrates into biological membranes "
                    "(replaces fatty acids in phospholipids) → disrupts myelin, retinal, cardiac membranes. "
                    "Normal plasma phytanic: <10 µmol/L; ARD untreated: typically 200–800+ µmol/L."
                ),
            },
            {
                "term": "Fasting Hazard — Phytanic Crisis",
                "definition": (
                    "Phytanic acid is stored in adipose tissue. During FASTING, illness, or surgery, "
                    "adipose lipolysis releases stored phytanic into circulation → RAPID PLASMA SURGE. "
                    "Phytanic crisis: acute worsening of polyneuropathy, cardiac arrhythmia, visual "
                    "deterioration, seizures. MANDATORY: IV 10% dextrose during any catabolic state "
                    "(illness, surgery, prolonged fasting) to prevent lipolysis. "
                    "PRE-OPERATIVE PROTOCOL: phytanic acid measurement 48h pre-op; plasmapheresis "
                    "if >200 µmol/L; IV dextrose intra- and post-operatively."
                ),
            },
            {
                "term": "Retinitis Pigmentosa (RP) — ~95% in ARD",
                "definition": (
                    "Rod-cone dystrophy: night blindness first (rods), then peripheral visual field loss, "
                    "then central vision. Fundus: 'bone-spicule' pigment deposits; attenuated vessels; "
                    "disc pallor. ERG: markedly reduced/absent rod responses. "
                    "RP is PROGRESSIVE even with optimal dietary control — structural damage pre-existing. "
                    "KEY: VGB ABSOLUTE CI — vigabatrin irreversibly worsens peripheral VF in RP patients. "
                    "ERG + formal VF testing MANDATORY before initiating any AED."
                ),
            },
            {
                "term": "VGB — ABSOLUTE CI (strongest in all peroxisomal epilepsy)",
                "definition": (
                    "Vigabatrin (VGB) causes irreversible peripheral visual field constriction in 30–40% "
                    "of all long-term users (mechanism: GABA accumulation → retinal toxicity). "
                    "In ARD: RP present in ~95% of patients — VGB causes CATASTROPHIC ADDITIVE "
                    "irreversible visual loss on top of already-severe retinal degeneration. "
                    "This is the STRONGEST VGB contraindication of ANY peroxisomal epilepsy disorder: "
                    "ABSOLUTE CI (do NOT use under any circumstances in ARD). "
                    "Contrast: AMACR (RP ~45%): RELATIVE CI. SCP2 (RP ~20%): RELATIVE CI. "
                    "ARD/PHYH (RP ~95%): ABSOLUTE CI — categorically different risk level."
                ),
            },
            {
                "term": "VPA — RELATIVE CI + POLG1 Mandatory",
                "definition": (
                    "VPA hepatotoxicity risk (standard; highest in <2 years, polypharmacy). "
                    "In ARD: liver processes phytanic acid load — hepatic metabolic burden elevated. "
                    "VPA adds to this burden. RELATIVE CI. "
                    "POLG1 genotyping MANDATORY (CPIC Grade A) before prescribing VPA in any "
                    "peroxisomal or mitochondrial disorder. POLG1 carriers: VPA ABSOLUTE CI "
                    "(mitochondrial hepatotoxicity). LEV is first-line; avoid VPA unless essential."
                ),
            },
            {
                "term": "PHT/CBZ — CAN USE (No Adrenal CI)",
                "definition": (
                    "PHYH deficiency does NOT cause adrenal insufficiency. "
                    "PHT/CBZ are CAN USE for focal seizures — NO adrenal mechanism CI. "
                    "CRITICAL CONTRAST: ABCD1 (X-ALD): PHT/CBZ = ABSOLUTE CI "
                    "(CYP450 induction → cortisol degradation → adrenal crisis in adrenal-insufficient patients). "
                    "In ARD: adrenal glands unaffected; PHT/CBZ safe. "
                    "CBZ: monitor for hyponatraemia; PHT: avoid in severe cerebellar ataxia (cerebellar toxicity risk)."
                ),
            },
            {
                "term": "Phytol-Restricted Diet — Level A (Gold Standard)",
                "definition": (
                    "The ONLY treatment that directly reduces phytanic acid load. Level A evidence. "
                    "Restrict: butter, cream, full-fat dairy (cheese, whole milk), beef tallow, lamb, "
                    "oily fish (herring, sardines, tuna, mackerel), spinach large portions. "
                    "Phytol from dietary chlorophyll is metabolized by gut bacteria → phytanic acid. "
                    "Target: plasma phytanic <200 µmol/L (ideally <50 µmol/L). "
                    "Strict compliance: 50–80% plasma phytanic reduction within 3–6 months. "
                    "Adequate calorie intake CRITICAL — caloric restriction → fat mobilization → phytanic surge. "
                    "Dietitian supervision essential; carbohydrate-rich diet to prevent lipolysis."
                ),
            },
            {
                "term": "Plasmapheresis / LDL Apheresis — Level B",
                "definition": (
                    "Rapid removal of phytanic acid from plasma during acute crisis or pre-treatment. "
                    "Level B evidence. Used when plasma phytanic is very high (>400 µmol/L) "
                    "or acute phytanic crisis. Also used pre-operatively if phytanic >200 µmol/L. "
                    "Temporary bridge — plasma phytanic rebounds without dietary control. "
                    "Multiple sessions often required. Centre expertise needed."
                ),
            },
            {
                "term": "PHYH vs PEX7 (RCDP1) — Critical Distinction",
                "definition": (
                    "BOTH PHYH deficiency AND PEX7 (RCDP1) cause phytanic acid elevation "
                    "(PEX7 imports PHYH into peroxisome via PTS2 receptor — without PEX7, PHYH is mislocalised). "
                    "DISTINCTION: PEX7/RCDP1 also has plasmalogens SEVERELY LOW (GNPAT/AGPS/FAR1 not imported). "
                    "PEX7/RCDP1: rhizomelia + stippled epiphyses + cataracts (neonatal severe phenotype). "
                    "PHYH/ARD: NO rhizomelia; NO plasmalogen deficiency; adult-onset. "
                    "RBC plasmalogen measurement MANDATORY to distinguish — if low → PEX7, not PHYH."
                ),
            },
            {
                "term": "Anosmia / Hyposmia (~70%) — Pathognomonic Clue",
                "definition": (
                    "Loss of or reduced sense of smell (anosmia/hyposmia) is present in ~70% of ARD. "
                    "Often earliest and most overlooked symptom — patients may not report it spontaneously. "
                    "Ask specifically about smell loss in any adult with RP + peripheral neuropathy. "
                    "Pathognomonic clinical clue: combination of RP + cerebellar ataxia + polyneuropathy "
                    "+ anosmia should IMMEDIATELY trigger plasma phytanic acid measurement."
                ),
            },
            {
                "term": "Cardiac Monitoring — Life-Threatening Arrhythmia",
                "definition": (
                    "Cardiac arrhythmia/AV block in ~40% of ARD patients. Mechanism: "
                    "phytanic acid integrates into cardiac myocyte membranes → disrupts ion channel function. "
                    "Life-threatening: sudden cardiac death reported in ARD. "
                    "MONITORING: 24h Holter ECG annually; echocardiogram annually. "
                    "Pacemaker implantation for complete AV block. Drug-drug interactions between "
                    "antiarrhythmics and AEDs must be considered."
                ),
            },
            {
                "term": "No ERT / No HSCT / No Lorenzo's Oil",
                "definition": (
                    "No ERT: PHYH is peroxisomal matrix enzyme (PTS2-targeted); not secreted; no receptor for systemic delivery. "
                    "No HSCT: ARD is metabolic/non-inflammatory; HSCT targets only inflammatory demyelination (ABCD1 cerebral). "
                    "No Lorenzo's Oil: oil works by competitive VLCFA elongase inhibition — VLCFA NORMAL in PHYH. Not applicable. "
                    "Diet is the primary and most effective disease-modifying therapy (Level A)."
                ),
            },
            {
                "term": "Diagnostic Workup",
                "definition": (
                    "Step 1: Adult with RP + cerebellar ataxia + polyneuropathy ± anosmia → "
                    "plasma phytanic acid IMMEDIATELY. "
                    "Step 2: Phytanic acid >10 µmol/L → confirm ARD. "
                    "Step 3: RBC plasmalogens → MUST be NORMAL (low → PEX7/RCDP1). "
                    "Step 4: VLCFA panel → NORMAL (elevated → ABCD1/ZSD/HSD17B4/ACOX1). "
                    "Step 5: Plasma pristanic → NORMAL or mildly elevated "
                    "(severely elevated → AMACR/SCP2 not PHYH). "
                    "Step 6: PHYH gene sequencing → biallelic pathogenic variants. "
                    "Step 7: If PHYH normal → PEX7 sequencing. "
                    "Step 8: Baseline ERG + VF before any AED — VGB ABSOLUTE CI. "
                    "Step 9: 24h Holter, echocardiogram, audiogram at diagnosis. "
                    "Step 10: POLG1 genotyping before VPA. "
                    "Step 11: Initiate phytol-restricted diet + dietitian referral immediately."
                ),
            },
        ],
        "diagnostic_algorithm": [
            "Adult/adolescent with RP + cerebellar ataxia + peripheral neuropathy ± anosmia → measure plasma phytanic acid FIRST",
            "Plasma phytanic acid >10 µmol/L (untreated ARD typically >200 µmol/L) — confirms ARD",
            "RBC plasmalogens → MUST be NORMAL; if low → PEX7/RCDP1 not PHYH",
            "VLCFA panel (C26:0, C24/22 ratio) → NORMAL; if elevated → ABCD1/ZSD/HSD17B4 first",
            "Plasma pristanic acid → NORMAL (if severely elevated → AMACR/SCP2 — different disease)",
            "Plasma THCA/DHCA → NORMAL (if elevated → HSD17B4/SCP2 — branched-chain beta-ox disorder)",
            "Pipecolic acid → NORMAL (if elevated → ZSD — PEX1/PEX6)",
            "Confirm PHYH enzyme activity in leukocytes/fibroblasts (optional — gene sequencing often sufficient)",
            "PHYH gene sequencing — identify biallelic pathogenic variants; if negative → PEX7 sequencing",
            "Baseline ERG + formal visual field testing MANDATORY before any AED initiation",
            "24h Holter ECG + echocardiogram + audiogram at diagnosis (cardiac/SNHL surveillance)",
            "POLG1 genotyping (CPIC A) before any VPA prescription",
            "Start phytol-restricted diet immediately + dietitian referral; target phytanic <200 µmol/L",
        ],
        "pharmacological_distinctions": [
            {"drug": "LEV", "status": "FIRST-LINE", "reason": "No peroxisomal interactions; safe in adult-onset ARD; myoclonus-effective"},
            {"drug": "Clonazepam", "status": "Level C (myoclonus/ataxia)", "reason": "Cerebellar ataxia + myoclonic seizures; sedation risk in ataxic patients"},
            {"drug": "PHT", "status": "CAN USE (no adrenal CI)", "reason": "No adrenal insufficiency in PHYH; avoid in severe ataxia (cerebellar toxicity)"},
            {"drug": "CBZ", "status": "CAN USE (no adrenal CI)", "reason": "No adrenal insufficiency; focal seizures; monitor LFTs + sodium"},
            {"drug": "LTG", "status": "Second-line", "reason": "Focal/myoclonic seizures; slow titration; no peroxisomal interaction"},
            {"drug": "VPA", "status": "RELATIVE CI", "reason": "Hepatotoxicity + hepatic phytanic burden + POLG1 MANDATORY CPIC A"},
            {"drug": "VGB", "status": "ABSOLUTE CI — RP ~95%", "reason": "Strongest VGB contraindication in peroxisomal epilepsy: RP 95% + VGB = catastrophic irreversible additive visual loss"},
            {"drug": "Phytol-restricted diet", "status": "GOLD STANDARD Level A", "reason": "Primary disease-modifying therapy; reduces phytanic acid 50-80%; lifelong"},
            {"drug": "Plasmapheresis / LDL apheresis", "status": "Level B (acute crisis)", "reason": "Rapid plasma phytanic reduction; acute crisis or pre-operative bridge"},
            {"drug": "IV dextrose (illness protocol)", "status": "MANDATORY during fasting/illness", "reason": "Prevents adipose lipolysis → phytanic surge → crisis"},
            {"drug": "Cardiac management (pacemaker)", "status": "As indicated", "reason": "AV block/arrhythmia ~40%; life-threatening; annual Holter mandatory"},
            {"drug": "Lorenzo's Oil", "status": "NOT APPLICABLE", "reason": "VLCFA normal in PHYH; oil targets VLCFA elongation — no mechanism benefit"},
            {"drug": "ERT", "status": "NO ERT available", "reason": "PTS2 matrix enzyme; not secreted; no systemic delivery feasible"},
            {"drug": "HSCT", "status": "NO HSCT", "reason": "Non-inflammatory metabolic disorder; HSCT only for inflammatory demyelination"},
        ],
        "differential_diagnoses": [
            {
                "disease": "AMACR / Adult Refsum Disease (Racemase)",
                "key_distinction": "Phytanic NORMAL in AMACR — SEVERELY ELEVATED in PHYH. Pristanic ELEVATED in AMACR — NORMAL in PHYH. Epilepsy ~60% AMACR vs ~20% PHYH. THCA/DHCA elevated in AMACR — NORMAL in PHYH.",
                "shared_features": "Adult-onset, VLCFA NORMAL, peripheral neuropathy, retinopathy, AR",
            },
            {
                "disease": "PEX7 (RCDP1)",
                "key_distinction": "BOTH elevate phytanic acid (PEX7 imports PHYH). BUT PEX7: plasmalogens SEVERELY LOW (PHYH: NORMAL). PEX7: rhizomelia + stippled epiphyses (neonatal). PHYH: no rhizomelia, adult onset. RBC plasmalogens MANDATORY.",
                "shared_features": "Phytanic acid elevated, AR, peroxisomal disease, cataracts",
            },
            {
                "disease": "ZSD (PEX1/PEX6)",
                "key_distinction": "ZSD: ALL peroxisomal pathways impaired (VLCFA + phytanic + plasmalogens + pristanic + pipecolic ALL abnormal). PHYH: ONLY phytanic elevated — ALL other markers NORMAL. ZSD: neonatal. PHYH: adolescent/adult.",
                "shared_features": "Phytanic elevated, peroxisomal disease, AR",
            },
            {
                "disease": "ABCD1 (X-ALD)",
                "key_distinction": "ABCD1: VLCFA SEVERELY ELEVATED — NORMAL in PHYH. ABCD1: adrenal insufficiency (PHT/CBZ ABSOLUTE CI) — PHYH: NO adrenal disease. ABCD1: phytanic NORMAL. X-linked (males most severe).",
                "shared_features": "Peroxisomal disease, neurological involvement",
            },
            {
                "disease": "HSD17B4 / SCP2 / AMACR (branched-chain beta-ox)",
                "key_distinction": "These diseases: pristanic + THCA ELEVATED — NORMAL in PHYH. Phytanic NORMAL in SCP2/HSD17B4/AMACR — SEVERELY ELEVATED in PHYH. Different pathway entirely.",
                "shared_features": "Peroxisomal disease, AR, adult onset (SCP2/AMACR), neuropathy",
            },
            {
                "disease": "Isolated Hereditary Retinitis Pigmentosa",
                "key_distinction": "Many genes cause isolated RP without systemic features. ARD has RP PLUS anosmia + ataxia + neuropathy. Plasma phytanic acid normal in isolated RP. KEY: always ask about anosmia + ataxia in RP patients.",
                "shared_features": "Retinitis pigmentosa, visual loss",
            },
            {
                "disease": "Charcot-Marie-Tooth Disease (CMT)",
                "key_distinction": "CMT: isolated demyelinating/axonal neuropathy. PHYH: neuropathy PLUS RP + cerebellar ataxia + elevated phytanic. Plasma phytanic normal in CMT. SNHL can occur in both.",
                "shared_features": "Peripheral polyneuropathy, hypertrophic nerves, SNHL",
            },
        ],
    }
