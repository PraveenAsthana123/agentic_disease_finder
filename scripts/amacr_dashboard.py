#!/usr/bin/env python3
"""AMACR (Alpha-methylacyl-CoA Racemase) Deficiency Epilepsy Dashboard — seed data module.

AMACR encodes alpha-methylacyl-CoA racemase (also called 2-methylacyl-CoA racemase), a 382 aa
peroxisomal and mitochondrial enzyme that catalyses the racemization of (R)-pristanoyl-CoA and
(R)-di-/tri-hydroxycholestanoyl-CoA (R-THCA-CoA, R-DHCA-CoA) to their corresponding (S)-forms.

PEROXISOMAL BETA-OXIDATION PATHWAY (Branched-Chain) — AMACR position:
  Substrate: Pristanic acid (phytanic acid → via PHYH alpha-oxidation → pristanic acid-CoA)
             THCA-CoA / DHCA-CoA (bile acid intermediates)
  ─ AMACR STEP (prerequisite): (R)-pristanoyl-CoA → (S)-pristanoyl-CoA  [RACEMIZATION]
  Step 1: ACOX2 (acyl-CoA oxidase 2) — FAD-linked oxidation (needs S-form)
  Step 2: HSD17B4/DBP (domain 1, enoyl-CoA hydratase) — hydration
  Step 3: HSD17B4/DBP (domain 2, 3-hydroxyacyl-CoA dehydrogenase) — dehydrogenation
  Step 4: SCPx/SCP2 (3-oxoacyl-CoA thiolase) — thiolytic cleavage → propionyl-CoA + acyl-CoA
  ────────────────────────────────────────────────────────────────────────────────────────────
  AMACR = PREREQUISITE racemization step — ACOX2/HSD17B4/SCP2 CANNOT process R-form substrates.
  Without AMACR, R-pristanoyl-CoA and R-THCA-CoA ACCUMULATE; entire branched-chain pathway stalls.
  [Straight-chain VLCFA uses ACOX1 which DOES NOT require racemization → VLCFA NORMAL in AMACR]

BIOCHEMICAL PROFILE (AMACR deficiency):
  Pristanic acid:   SEVERELY ELEVATED — R-form cannot enter beta-oxidation (racemase block)
  THCA / DHCA:      ELEVATED — bile acid chain shortening blocked upstream of ACOX2
  VLCFA (C26:0):    NORMAL — straight-chain ACOX1 pathway does NOT need racemization
  Phytanic acid:    NORMAL or mildly elevated — PHYH alpha-oxidation intact; phytanic → pristanic step OK
  Plasmalogens:     NORMAL — PTS2/plasmalogen biosynthesis unaffected (FAR1/AGPS/GNPAT/PEX7 intact)

CRITICAL BIOCHEMICAL DISTINCTIONS:
  AMACR vs SCP2 (SCPx / 3-oxoacyl-CoA thiolase Step 4):
    Nearly IDENTICAL plasma biochemistry — both pristanic ELEVATED, THCA/DHCA ELEVATED, VLCFA NORMAL.
    AMACR blocks racemization BEFORE ACOX2 (no entry to beta-oxidation).
    SCP2 blocks thiolytic cleavage AFTER three complete steps.
    KEY: Gene sequencing (AMACR vs SCP2/SCPx) is the ONLY reliable way to distinguish these two.
    Epilepsy more prominent in AMACR (~60%) than SCP2 (~40%); focal temporal seizures characteristic.
    Azoospermia: AMACR ~50-60% males; SCP2 ~95% males.
  AMACR vs HSD17B4 (D-Bifunctional Protein / DBP):
    VLCFA NORMAL in AMACR — SIGNIFICANTLY ELEVATED in HSD17B4. CRITICAL DISTINCTION.
    Both pristanic + THCA elevated.
    HSD17B4: neonatal severe onset. AMACR: adult onset.
    VLCFA measurement MANDATORY — if VLCFA elevated → HSD17B4 first.
  AMACR vs PHYH (Adult Refsum Disease / ARD):
    Phytanic acid SEVERELY ELEVATED in Refsum (NORMAL in AMACR).
    VLCFA NORMAL in both.
    Refsum: RP + polyneuropathy + anosmia + deafness; AMACR: epilepsy + neuropathy + retinopathy.
    Pristanic elevated in AMACR; may be mildly elevated in Refsum (different mechanism).
  AMACR vs ACOX1 (Pseudo-NALD):
    VLCFA ELEVATED in ACOX1 (NORMAL in AMACR).
    ACOX1: neonatal infantile onset. AMACR: adult onset.
    Pristanic NORMAL in ACOX1 (ELEVATED in AMACR).
  AMACR vs ZSD (PEX1/PEX6):
    ZSD: ALL peroxisomal pathways affected (VLCFA + plasmalogens + pristanic + THCA all abnormal).
    AMACR: only branched-chain pathway (VLCFA NORMAL, plasmalogens NORMAL).
  AMACR Prostate Cancer Misdiagnosis Warning:
    p.Ser113Leu (rs10794086) is a common prostate cancer risk allele (NOT pathogenic for neurological AMACR).
    Leukocyte/fibroblast AMACR enzyme activity assay MANDATORY — prostate cancer marker does not predict
    neurological disease. Biochemical confirmation (plasma pristanic + THCA) required before genetic dx.

AMACR PROTEIN BIOLOGY:
  Gene AMACR at 5p13.2 (OMIM Gene: *604489).
  382 aa; dual subcellular localization: peroxisomal matrix (major) + mitochondrial matrix (minor).
  Both peroxisomal and mitochondrial isoforms require racemization for their respective substrates.
  Active site: conserved Ser and His residues; Mg2+ independent (unlike some other racemases).
  Substrates: (2R)-pristanoyl-CoA, (25R)-THCA-CoA, (25R)-DHCA-CoA — all 2-methyl-branched.
  Peroxisomal targeting: PTS1-like C-terminal signal (AKL in human).
  Expressed: liver (highest), small intestine, adrenal cortex, prostate (hence prostate cancer marker).
  NOTE: AMACR is the ONLY known enzyme catalysing this specific racemization in humans.

OMIM: Gene *604489 (AMACR) | Disease #614307 (adult-onset neurological disease + neuropathy)
LOCUS: 5p13.2
EPIDEMIOLOGY: Extremely rare — ~25-30 cases published worldwide 2026.
  First neurological case: Ferdinandusse et al. 2000 (Nature Genetics) — adult with liver disease.
  Neurological phenotype fully characterized by Setchell et al. and Wanders group.
  Adult-onset phenotype (20s–60s): focal epilepsy + peripheral neuropathy + pigmentary retinopathy.
  Both sexes affected equally (AR); males may have azoospermia (~50-60%).

CLINICAL PHENOTYPE:
  Adult-onset (typically 20s–60s). Core features:
  • Epilepsy (~60%) — most characteristic: focal temporal lobe seizures, secondary generalization
  • Peripheral polyneuropathy — axonal, sensorimotor; length-dependent lower limbs
  • Pigmentary retinopathy (~45%) — less severe than ABCD1 or RCDP; rod–cone dystrophy
  • Cognitive decline — progressive, late-stage
  • Tremor / cerebellar ataxia (~35%)
  • Azoospermia (~50-60% of affected males)
  • Spastic paraparesis (~25%)
  • Mild hepatomegaly (elevated liver enzymes in subset — bile acid synthesis disruption)
  • SNHL: ~30%
  • MRI: white matter changes variable; less severe leukodystrophy than HSD17B4 or ABCD1

ERT/HSCT/GENE THERAPY:
  No ERT (2026) — AMACR is a peroxisomal matrix enzyme; no secreted form for systemic delivery.
  No HSCT — AMACR deficiency is NOT inflammatory demyelination; HSCT targets only inflammatory demyelinators.
  No gene therapy approved 2026.

DIET THERAPY:
  Phytol-restricted diet: Level C — reduce dietary phytol → less phytanic acid (gut phytol → phytanic acid
  via gut microbiome/hepatic metabolism) → less pristanic acid substrate load for peroxisomal beta-oxidation.
  Practical: restrict dairy fat, ruminant fat, certain fish (similar to SCP2 dietary advice).
  Limited evidence base (very rare disease); reduces pristanic acid burden in reported cases.
  DHA supplementation: Level C — theoretical benefit from DHA synthesis pathway downstream of peroxisomes.

TREATMENT PRINCIPLES:
  LEV first-line for seizures (no peroxisomal interactions; well-tolerated in adult-onset disease).
  Focal temporal seizures: OXC/CBZ may be used — NO adrenal insufficiency (unlike ABCD1 where CI).
  VPA: RELATIVE CI — standard hepatotoxicity risk + bile acid metabolism disruption (THCA/DHCA
    accumulation adds hepatic metabolic burden); POLG1 MANDATORY CPIC A before prescribing.
  VGB: RELATIVE CI — VGB causes irreversible peripheral visual field constriction; in AMACR patients
    with already-present pigmentary retinopathy (~45%), VGB MARKEDLY INCREASES visual loss risk.
    Higher-risk than SCP2 (retinopathy is more common in AMACR than SCP2).
  Clonazepam: adjunct for myoclonus (Level C).
  PHT/CBZ: CAN USE for focal seizures — no adrenal insufficiency mechanism (distinct from ABCD1).
  Lorenzo oil: NOT APPLICABLE — AMACR blocks racemization, not VLCFA import (ABCD1) or ACOX1 step 1.
  No ERT. No HSCT. Phytol-restricted diet Level C.
"""

import random
random.seed(61)


# ── Overview (KPIs + summary) ─────────────────────────────────────────────────

def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 60,
        "polyneuropathy_pct": 80,
        "retinopathy_pct": 45,
        "tremor_ataxia_pct": 35,
        "azoospermia_pct_males": 55,
        "cognitive_decline_pct": 70,
        "drug_resistance_pct": 25,
        "spastic_paraparesis_pct": 25,
        "snhl_pct": 30,
        "hepatomegaly_pct": 30,
        "vlcfa_normal_pct": 90,
        "pristanic_elevated_pct": 100,
        "thca_elevated_pct": 92,
        "phytanic_normal_pct": 88,
        "plasmalogen_normal_pct": 98,
        "adrenal_insufficiency_pct": 0,
        "omim_gene": "604489",
        "omim_disease": "614307",
        "locus": "5p13.2",
        "vlcfa_elevated": False,
        "inheritance": "Autosomal recessive (AR), biallelic LOF — both sexes equally; males: azoospermia ~55%",
        "common_variant": "No common founder mutation — all private variants; ~25-30 cases worldwide 2026; extremely rare",
        "onset_age": "Adult-onset (typically 20s–60s) — similar to SCP2; older onset than ACOX1/HSD17B4 (neonatal)",
        "disease_mechanism": (
            "AMACR (alpha-methylacyl-CoA racemase, 382 aa, PTS1: AKL) catalyses the PREREQUISITE "
            "racemization step: (R)-pristanoyl-CoA → (S)-pristanoyl-CoA and (R)-THCA/DHCA-CoA → "
            "(S)-forms. Without racemization, ACOX2/HSD17B4/SCP2 beta-oxidation steps CANNOT "
            "process branched-chain substrates (they require S-stereoisomers). "
            "AMACR deficiency → pristanic acid SEVERELY ELEVATED + THCA/DHCA ELEVATED; "
            "VLCFA NORMAL (straight-chain ACOX1 pathway does NOT require racemization). "
            "KEY DISTINCTION from SCP2: biochemically near-identical → gene sequencing MANDATORY. "
            "Epilepsy MORE PROMINENT in AMACR (~60%) than SCP2 (~40%). "
            "Focal temporal lobe seizures characteristic. VGB RELATIVE CI (retinopathy ~45% — "
            "higher than SCP2; VGB markedly increases visual loss risk). "
            "VPA RELATIVE CI (bile acid burden + POLG1 mandatory). No ERT. No HSCT. "
            "LEV first-line. Phytol-restricted diet Level C."
        ),
        "nbs_positive_rate": (
            "Not in standard NBS (adult-onset disease). Pristanic acid + THCA/DHCA plasma elevation "
            "in adult with focal epilepsy + neuropathy + retinopathy → suspect AMACR or SCP2. "
            "VLCFA NORMAL → excludes ZSD/HSD17B4/ACOX1. Leukocyte AMACR enzyme activity assay + "
            "gene sequencing (AMACR vs SCP2) required for definitive diagnosis. "
            "Prostate cancer allele p.Ser113Leu NOT pathogenic for neurological disease."
        ),
        "critical_distinctions": [
            "AMACR vs SCP2: biochemically near-identical → GENE SEQUENCING ONLY distinguishes",
            "AMACR epilepsy more prominent (~60%) than SCP2 (~40%) — focal temporal lobe",
            "VLCFA NORMAL — excludes ZSD (PEX1/PEX6), HSD17B4, ACOX1",
            "Phytanic NORMAL — excludes PHYH/Adult Refsum Disease",
            "VGB RELATIVE CI (pigmentary retinopathy ~45% — highest visual risk in this group)",
            "No adrenal insufficiency — PHT/CBZ/OXC CAN USE (contrast ABCD1 absolute CI)",
            "Prostate cancer allele p.Ser113Leu ≠ neurological pathogenic variant",
        ],
    }


# ── Breakdown (patients, phenotypes, triggers, treatments) ─────────────────────

def get_breakdown():
    random.seed(61)
    onset_ages = [random.randint(22, 62) for _ in range(40)]

    # Phenotypic classes
    phenotypic_classes = [
        {
            "class": "Epilepsy-Predominant",
            "pct": 40,
            "description": "Focal temporal lobe epilepsy as presenting feature; neuropathy develops later",
            "count": 16,
            "seizure_control": "Partial (drug-resistant in ~30%)",
        },
        {
            "class": "Neuropathy-Predominant",
            "pct": 35,
            "description": "Peripheral polyneuropathy (axonal, sensorimotor) presenting feature; seizures secondary",
            "count": 14,
            "seizure_control": "Good with LEV (60%)",
        },
        {
            "class": "Mixed / Multisystem",
            "pct": 25,
            "description": "Simultaneous epilepsy + neuropathy + retinopathy + cognitive decline",
            "count": 10,
            "seizure_control": "Difficult (polypharmacy needed)",
        },
    ]

    # Seizure types
    seizure_types = [
        {"type": "Focal temporal (aware)", "pct": 55, "notes": "Most characteristic — often aura with epigastric rising"},
        {"type": "Focal to bilateral tonic-clonic", "pct": 40, "notes": "Secondarily generalised from temporal focus"},
        {"type": "Myoclonic", "pct": 25, "notes": "Particularly with progressive cognitive decline"},
        {"type": "Focal parietal-occipital", "pct": 15, "notes": "Rare; correlates with retinopathy severity"},
        {"type": "Status epilepticus (rare)", "pct": 5, "notes": "Precipitated by metabolic decompensation"},
    ]

    # Triggers
    triggers = [
        {"trigger": "Dietary phytol overload (dairy/ruminant fat)", "pct": 45, "mechanism": "Increases pristanic acid substrate load"},
        {"trigger": "Febrile illness / infection", "pct": 38, "mechanism": "Metabolic stress; increased peroxisomal demand"},
        {"trigger": "Sleep deprivation", "pct": 30, "mechanism": "Universal seizure threshold lowering"},
        {"trigger": "Missed AED dose", "pct": 25, "mechanism": "Subtherapeutic drug level"},
        {"trigger": "VPA initiation without POLG1 screen", "pct": 18, "mechanism": "Hepatotoxicity + bile acid burden"},
        {"trigger": "VGB initiation (visual loss)", "pct": 15, "mechanism": "Additive retinopathy + VF constriction risk"},
        {"trigger": "Fasting / metabolic decompensation", "pct": 12, "mechanism": "Ketosis alters peroxisomal fatty acid flux"},
    ]

    # Monitoring parameters
    monitoring = [
        {"parameter": "Plasma pristanic acid", "frequency": "Every 6 months", "target": "Reduction from baseline with phytol diet"},
        {"parameter": "THCA / DHCA (plasma)", "frequency": "Every 6 months", "target": "Monitor bile acid intermediate burden"},
        {"parameter": "Plasma phytanic acid", "frequency": "Annually", "target": "Normal (confirm PHYH intact)"},
        {"parameter": "VLCFA panel", "frequency": "At diagnosis (baseline only)", "target": "Must be NORMAL — if elevated, reconsider dx"},
        {"parameter": "LFTs (ALT/AST/GGT)", "frequency": "Every 3 months if on VPA (if used)", "target": "< 3x ULN"},
        {"parameter": "Visual field + ERG", "frequency": "Annually", "target": "Monitor retinopathy progression"},
        {"parameter": "Nerve conduction studies", "frequency": "Annually", "target": "Track neuropathy progression"},
        {"parameter": "POLG1 genotype", "frequency": "Once (before VPA)", "target": "Mandatory CPIC A"},
        {"parameter": "Semen analysis (males)", "frequency": "At diagnosis", "target": "Azoospermia screening"},
    ]

    # Lifecycle stages
    lifecycle = [
        {"stage": "Pre-diagnosis (adult prodrome)", "age_range": "20s–40s", "features": "Subtle temporal lobe aura; peripheral tingling; cholestasis"},
        {"stage": "Diagnosis", "age_range": "30s–50s", "features": "Pristanic + THCA elevated; VLCFA normal; AMACR gene sequencing"},
        {"stage": "Early management", "age_range": "Diagnosis + 2 years", "features": "LEV initiated; phytol-restricted diet; POLG1 screen before VPA"},
        {"stage": "Established disease", "age_range": "Diagnosis + 5–10 years", "features": "Polypharmacy if drug-resistant; neuropathy + retinopathy progression"},
        {"stage": "Advanced disease", "age_range": "Diagnosis + 10+ years", "features": "Cognitive decline; mobility limitation; visual impairment"},
        {"stage": "Palliative / supportive", "age_range": "Variable", "features": "Symptomatic management; multidisciplinary care"},
    ]

    # Treatment catalog
    treatments = [
        {"drug": "Levetiracetam (LEV)", "level": "FIRST-LINE", "indication": "Focal + generalised seizures", "ci": "None peroxisomal-specific"},
        {"drug": "Lamotrigine (LTG)", "level": "Second-line adjunct", "indication": "Focal seizures", "ci": "Slow titration needed"},
        {"drug": "Oxcarbazepine (OXC)", "level": "Second-line (focal)", "indication": "Focal temporal lobe", "ci": "No adrenal CI — unlike ABCD1"},
        {"drug": "Carbamazepine (CBZ)", "level": "CAN USE (focal)", "indication": "Focal seizures", "ci": "No adrenal CI; monitor LFTs"},
        {"drug": "Phenytoin (PHT)", "level": "CAN USE", "indication": "Focal; SE protocol", "ci": "No adrenal CI — unlike ABCD1"},
        {"drug": "Clonazepam (CLZ)", "level": "Adjunct (myoclonus)", "indication": "Myoclonic jerks", "ci": "Sedation; tolerance"},
        {"drug": "Valproate (VPA)", "level": "RELATIVE CI", "indication": "Avoid if possible", "ci": "Hepatotoxicity + THCA/DHCA bile acid burden; POLG1 MANDATORY"},
        {"drug": "Vigabatrin (VGB)", "level": "RELATIVE CI", "indication": "Avoid", "ci": "Additive retinopathy (pigmentary retinopathy ~45%) + VF constriction; HIGH visual risk"},
        {"drug": "Lorenzo's Oil", "level": "NOT APPLICABLE", "indication": "N/A", "ci": "Targets VLCFA elongation; VLCFA NORMAL in AMACR — no mechanism"},
        {"drug": "Phytol-Restricted Diet", "level": "Level C", "indication": "All AMACR patients", "ci": "None; reduces pristanic substrate load"},
        {"drug": "DHA Supplementation", "level": "Level C", "indication": "Theoretical neuroprotection", "ci": "Minimal risk; limited evidence"},
    ]

    patients = []
    sexes = ["Male"] * 22 + ["Female"] * 18
    random.shuffle(sexes)
    phenotype_pool = ["Epilepsy-Predominant"] * 16 + ["Neuropathy-Predominant"] * 14 + ["Mixed/Multisystem"] * 10
    random.shuffle(phenotype_pool)
    seizure_pool = ["Focal temporal"] * 22 + ["Focal-BTCS"] * 16 + ["Myoclonic"] * 10 + ["None"] * 16
    random.shuffle(seizure_pool)

    for i in range(40):
        sex = sexes[i]
        age = onset_ages[i]
        pheno = phenotype_pool[i]
        sz = seizure_pool[i % len(seizure_pool)]
        has_retinopathy = random.random() < 0.45
        has_neuropathy = random.random() < 0.80
        has_ataxia = random.random() < 0.35
        azoospermia = (sex == "Male" and random.random() < 0.55)
        pristanic = round(random.uniform(15.0, 80.0), 1)
        thca = round(random.uniform(2.0, 25.0), 1)
        vlcfa = round(random.uniform(0.5, 0.95), 2)  # Normal range
        patients.append({
            "id": i + 1,
            "sex": sex,
            "onset_age": age,
            "phenotypic_class": pheno,
            "primary_seizure": sz,
            "has_retinopathy": has_retinopathy,
            "has_polyneuropathy": has_neuropathy,
            "has_ataxia_tremor": has_ataxia,
            "azoospermia": azoospermia if sex == "Male" else None,
            "pristanic_umol_L": pristanic,
            "thca_umol_L": thca,
            "vlcfa_c26_umol_L": vlcfa,
            "drug_resistant": random.random() < 0.25,
            "current_aed": random.choice(["LEV", "LEV+LTG", "LEV+OXC", "LEV+CBZ", "LEV+CLZ"]),
            "diet_compliance": random.choice(["Full", "Partial", "None"]),
        })

    return {
        "phenotypic_classes": phenotypic_classes,
        "seizure_types": seizure_types,
        "triggers": triggers,
        "monitoring": monitoring,
        "lifecycle": lifecycle,
        "treatments": treatments,
        "patients": patients,
        "summary": {
            "total_patients": 40,
            "epilepsy_pct": 60,
            "drug_resistant_pct": 25,
            "retinopathy_pct": 45,
            "polyneuropathy_pct": 80,
            "azoospermia_males_pct": 55,
        },
    }


# ── Definitions (glossary, diagnostic algorithm, pharmacological distinctions) ──

def get_definitions():
    return {
        "key_concepts": [
            {
                "term": "Alpha-methylacyl-CoA Racemase (AMACR)",
                "definition": (
                    "382 aa peroxisomal (+ mitochondrial) enzyme encoded by AMACR at 5p13.2. "
                    "Catalyses the PREREQUISITE racemization of (R)-2-methyl-branched fatty acyl-CoA "
                    "esters — pristanoyl-CoA, THCA-CoA, DHCA-CoA — to their (S)-stereoisomers. "
                    "ACOX2/HSD17B4/SCP2 beta-oxidation steps require S-form substrates. "
                    "Without AMACR, the entire branched-chain peroxisomal beta-oxidation stalls."
                ),
            },
            {
                "term": "Biochemical Signature",
                "definition": (
                    "Pristanic acid SEVERELY ELEVATED (R-form cannot enter beta-oxidation). "
                    "THCA/DHCA ELEVATED (bile acid chain shortening blocked). "
                    "VLCFA (C26:0) NORMAL (straight-chain ACOX1 does NOT need racemisation). "
                    "Phytanic acid NORMAL (PHYH alpha-oxidation intact). "
                    "Plasmalogens NORMAL (PTS2 pathway unaffected). "
                    "Identical to SCP2 profile — gene sequencing MANDATORY to distinguish."
                ),
            },
            {
                "term": "AMACR vs SCP2 — Key Clinical Distinction",
                "definition": (
                    "Biochemically near-identical: both pristanic + THCA elevated; VLCFA normal. "
                    "Gene sequencing (AMACR gene vs SCP2 gene) is the ONLY way to distinguish. "
                    "AMACR: epilepsy ~60% (more prominent); SCP2: epilepsy ~40%, movement disorder ~90% (more prominent). "
                    "AMACR: focal temporal lobe characteristic; SCP2: extrapyramidal movement disorder. "
                    "AMACR: retinopathy ~45% (VGB higher risk); SCP2: retinopathy ~20% (lower)."
                ),
            },
            {
                "term": "Prostate Cancer Allele Warning",
                "definition": (
                    "p.Ser113Leu (rs10794086) in AMACR is a COMMON prostate cancer risk allele (~5–15% of general pop). "
                    "It is NOT pathogenic for neurological AMACR deficiency. "
                    "Misidentification of this variant as causative is a known diagnostic pitfall. "
                    "MANDATORY: leukocyte AMACR enzyme activity assay + plasma biochemistry (pristanic + THCA) "
                    "before attributing neurological disease to any AMACR variant."
                ),
            },
            {
                "term": "VGB — RELATIVE CI (HIGH visual risk)",
                "definition": (
                    "Vigabatrin causes irreversible peripheral visual field constriction in ~30-40% of long-term users. "
                    "In AMACR patients with PIGMENTARY RETINOPATHY (~45%), VGB poses an ADDITIVE, compounded "
                    "visual loss risk. This is higher than SCP2 (where retinopathy ~20%). "
                    "VGB RELATIVE CI — avoid unless no alternatives; if used, ERG + VF monitoring every 3 months."
                ),
            },
            {
                "term": "VPA — RELATIVE CI + POLG1 Mandatory",
                "definition": (
                    "VPA carries standard hepatotoxicity risk (especially <2 years, polypharmacy). "
                    "THCA/DHCA accumulation in AMACR adds hepatic metabolic burden (bile acid intermediates "
                    "are hepatotoxic). Combined risk: VPA + THCA/DHCA accumulation → compounded hepatotoxicity. "
                    "POLG1 genotyping MANDATORY (CPIC A) before prescribing VPA in any peroxisomal disorder. "
                    "POLG1 carriers: VPA absolutely contraindicated (mitochondrial hepatotoxicity)."
                ),
            },
            {
                "term": "PHT/CBZ/OXC — CAN USE (No Adrenal CI)",
                "definition": (
                    "AMACR deficiency does NOT cause adrenal insufficiency (adrenal is unaffected). "
                    "PHT/CBZ/OXC are safe to use for focal seizures — NO adrenal mechanism CI. "
                    "This CONTRASTS with ABCD1 (X-ALD) where PHT/CBZ are ABSOLUTE CI due to adrenal crisis risk. "
                    "PHT/CBZ/OXC are appropriate focal seizure drugs in AMACR."
                ),
            },
            {
                "term": "Phytol-Restricted Diet",
                "definition": (
                    "Level C evidence. Dietary phytol (from chlorophyll in green vegetables, and ruminant fat) "
                    "is metabolized via gut bacteria → phytanic acid → via PHYH → pristanic acid. "
                    "Reducing dietary phytol lowers pristanic acid substrate load for peroxisomal beta-oxidation. "
                    "Practical: limit dairy fat, ruminant fat, certain oily fish. "
                    "Can reduce plasma pristanic acid ~20-40% in compliant patients. Limited evidence base."
                ),
            },
            {
                "term": "DHA Supplementation",
                "definition": (
                    "Level C. DHA (docosahexaenoic acid) is synthesized via peroxisomal pathway. "
                    "Theoretical neuroprotective benefit by replacing deficient peroxisome-derived DHA. "
                    "Limited evidence in AMACR specifically; extrapolated from Zellweger/peroxisomal disorders."
                ),
            },
            {
                "term": "No ERT Applicable",
                "definition": (
                    "Enzyme replacement therapy (ERT) requires: (a) secreted enzyme form, (b) receptor-mediated "
                    "uptake. AMACR is a peroxisomal/mitochondrial MATRIX enzyme — not secreted; no receptor "
                    "for systemic delivery. No ERT is feasible in 2026."
                ),
            },
            {
                "term": "No HSCT Applicable",
                "definition": (
                    "HSCT benefits only inflammatory demyelinating diseases (ABCD1/X-ALD cerebral form). "
                    "AMACR deficiency is a metabolic disorder WITHOUT inflammatory demyelination. "
                    "HSCT cannot correct the intracellular racemization defect. Not indicated."
                ),
            },
            {
                "term": "Lorenzo's Oil — NOT APPLICABLE",
                "definition": (
                    "Lorenzo's Oil (erucic acid + oleic acid) works by competitive inhibition of VLCFA elongase, "
                    "reducing VLCFA synthesis. VLCFA is NORMAL in AMACR deficiency — the oil has no mechanism "
                    "to reduce pristanic acid or THCA. NOT applicable."
                ),
            },
            {
                "term": "Diagnostic Workup",
                "definition": (
                    "Step 1: Plasma VLCFA panel → NORMAL (excludes ZSD, ABCD1, HSD17B4, ACOX1). "
                    "Step 2: Plasma pristanic + THCA/DHCA → ELEVATED (confirms branched-chain block). "
                    "Step 3: Plasma phytanic acid → NORMAL (excludes Refsum disease/PHYH). "
                    "Step 4: Leukocyte AMACR enzyme activity → REDUCED/ABSENT (confirms AMACR deficiency). "
                    "Step 5: AMACR gene sequencing — biallelic pathogenic variants. "
                    "Step 6: Rule out SCP2 with SCP2/SCPx gene sequencing if AMACR enzyme normal but biochemistry positive."
                ),
            },
            {
                "term": "Inheritance",
                "definition": (
                    "Autosomal recessive (AR), biallelic loss-of-function mutations. "
                    "Both sexes equally affected. No sex-specific lethality. "
                    "Males: azoospermia in ~50-60% (AMACR required for testicular germ cell lipid metabolism). "
                    "Carrier parents (1/2 normal enzyme activity): asymptomatic."
                ),
            },
            {
                "term": "OMIM & Locus",
                "definition": (
                    "Gene AMACR: OMIM *604489, chromosome 5p13.2. "
                    "Disease OMIM: #614307 (adult-onset neurological AMACR deficiency). "
                    "~25-30 cases worldwide published by 2026. Extremely rare; under-diagnosed. "
                    "First neurological case: Ferdinandusse et al. 2000, Nat Genet."
                ),
            },
        ],
        "diagnostic_algorithm": [
            "Adult with focal temporal epilepsy + peripheral neuropathy ± retinopathy — consider peroxisomal screen",
            "Step 1: VLCFA panel (C26:0, C24/C22 ratio) → if NORMAL, ZSD/ABCD1/HSD17B4/ACOX1 excluded",
            "Step 2: Plasma pristanic acid → if SEVERELY ELEVATED → branched-chain peroxisomal defect",
            "Step 3: Plasma THCA/DHCA → ELEVATED → confirms branched-chain beta-oxidation block",
            "Step 4: Plasma phytanic acid → NORMAL → excludes PHYH/Adult Refsum Disease",
            "Step 5: Plasma/RBC plasmalogens → NORMAL → excludes ZSD/RCDP",
            "Step 6: Leukocyte AMACR enzyme activity assay → if LOW → AMACR deficiency confirmed",
            "Step 7: AMACR gene sequencing → identify biallelic pathogenic variants",
            "Step 8: If AMACR enzyme normal but biochemistry positive → SCP2/SCPx gene sequencing",
            "Step 9: Rule out p.Ser113Leu prostate cancer allele — NOT pathogenic for neurological disease",
            "Step 10: Screen for POLG1 variants (CPIC A) BEFORE initiating VPA",
            "Step 11: ERG + visual field BEFORE VGB (retinopathy baseline; VGB RELATIVE CI)",
            "Step 12: Initiate phytol-restricted diet + LEV + multidisciplinary follow-up",
        ],
        "pharmacological_distinctions": [
            {"drug": "LEV", "status": "FIRST-LINE", "reason": "No peroxisomal interactions; safe in adult-onset disease"},
            {"drug": "OXC", "status": "CAN USE (focal)", "reason": "No adrenal CI; appropriate for temporal lobe epilepsy"},
            {"drug": "CBZ", "status": "CAN USE", "reason": "No adrenal CI; monitor LFTs if baseline elevated"},
            {"drug": "PHT", "status": "CAN USE", "reason": "No adrenal CI — distinct from ABCD1 where ABSOLUTE CI"},
            {"drug": "LTG", "status": "Second-line", "reason": "Focal seizures; slow titration; no peroxisomal interaction"},
            {"drug": "CLZ", "status": "Adjunct (myoclonus)", "reason": "Myoclonic jerks; sedation risk"},
            {"drug": "VPA", "status": "RELATIVE CI", "reason": "Hepatotoxicity + THCA/DHCA bile acid burden + POLG1 mandatory CPIC A"},
            {"drug": "VGB", "status": "RELATIVE CI (HIGH visual risk)", "reason": "Pigmentary retinopathy ~45% in AMACR — additive VF constriction; highest visual risk in peroxisomal group"},
            {"drug": "Lorenzo's Oil", "status": "NOT APPLICABLE", "reason": "VLCFA normal in AMACR; oil targets VLCFA elongation only"},
            {"drug": "ERT", "status": "NO ERT available", "reason": "AMACR is peroxisomal matrix enzyme; not secreted; no systemic delivery feasible"},
            {"drug": "HSCT", "status": "NO HSCT", "reason": "Non-inflammatory metabolic disorder; HSCT only for inflammatory demyelination (ABCD1 cerebral)"},
            {"drug": "Phytol diet", "status": "Level C", "reason": "Reduces pristanic acid substrate load; practical dietary restriction"},
            {"drug": "DHA", "status": "Level C", "reason": "Theoretical neuroprotection; extrapolated from ZSD data"},
        ],
        "differential_diagnoses": [
            {
                "disease": "SCP2/SCPx Deficiency",
                "key_distinction": "Near-identical biochemistry — gene sequencing ONLY distinguishes. SCP2: movement disorder ~90% > epilepsy ~40%; AMACR: epilepsy ~60% > movement disorder",
                "shared_features": "Pristanic elevated, THCA elevated, VLCFA normal, adult onset, AR",
            },
            {
                "disease": "HSD17B4 (D-Bifunctional Protein)",
                "key_distinction": "VLCFA SIGNIFICANTLY ELEVATED in HSD17B4 — NORMAL in AMACR. HSD17B4: neonatal onset. AMACR: adult onset",
                "shared_features": "Pristanic elevated, THCA elevated",
            },
            {
                "disease": "PHYH / Adult Refsum Disease",
                "key_distinction": "Phytanic acid SEVERELY ELEVATED in Refsum — NORMAL in AMACR. Refsum: RP + anosmia + deafness + ichthyosis",
                "shared_features": "VLCFA normal, adult onset, neuropathy, retinopathy",
            },
            {
                "disease": "ACOX1 (Pseudo-NALD)",
                "key_distinction": "VLCFA ELEVATED in ACOX1 — NORMAL in AMACR. ACOX1: neonatal/infantile onset — not adult",
                "shared_features": "Peroxisomal disease, AR",
            },
            {
                "disease": "ZSD (PEX1/PEX6)",
                "key_distinction": "ZSD: ALL pathways affected (VLCFA + plasmalogens + pristanic all abnormal). AMACR: VLCFA + plasmalogens NORMAL",
                "shared_features": "Pristanic elevated, peroxisomal disease",
            },
            {
                "disease": "ABCD1 (X-ALD)",
                "key_distinction": "ABCD1: VLCFA SEVERELY ELEVATED — NORMAL in AMACR. ABCD1: adrenal insufficiency + inflammatory demyelination; AMACR: no adrenal disease. PHT/CBZ ABSOLUTE CI in ABCD1, CAN USE in AMACR",
                "shared_features": "Peroxisomal disease, neurological involvement",
            },
            {
                "disease": "RCDP (PEX7/GNPAT/AGPS/FAR1)",
                "key_distinction": "RCDP: plasmalogens SEVERELY LOW — NORMAL in AMACR. RCDP: rhizomelia, neonatal onset, PTS2 defect",
                "shared_features": "AR peroxisomal disease",
            },
        ],
    }
