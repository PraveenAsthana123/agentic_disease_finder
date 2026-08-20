#!/usr/bin/env python3
"""SCP2 / SCPx (Sterol Carrier Protein X / 3-Oxoacyl-CoA Thiolase) Deficiency Epilepsy Dashboard — seed data module.

SCP2 encodes two protein products from alternative promoters:
  1. SCPx (547 aa, peroxisomal matrix, PTS1: SKL) — bifunctional: N-terminal thiolase domain (~424 aa)
     + C-terminal SCP2 domain (123 aa). The thiolase (3-oxoacyl-CoA thiolase) catalyses STEP 4 of
     peroxisomal beta-oxidation for BRANCHED-CHAIN substrates.
  2. SCP2 / nsLTP (123 aa, non-specific lipid transfer protein) — sterol carrier; ligand transfer in
     cholesterol/sterol trafficking between membranes and organelles.

PEROXISOMAL BETA-OXIDATION PATHWAY (Branched-Chain):
  Substrate: Pristanic acid-CoA (2-methylacyl-CoA from phytanic acid via alpha-oxidation) or
             THCA-CoA / DHCA-CoA (bile acid intermediates)
  Step 1: ACOX2 (acyl-CoA oxidase 2) — FAD-linked oxidation
  Step 2: HSD17B4/DBP (domain 1, enoyl-CoA hydratase) — hydration
  Step 3: HSD17B4/DBP (domain 2, 3-hydroxyacyl-CoA dehydrogenase) — dehydrogenation
  Step 4: SCPx/SCP2 (3-oxoacyl-CoA thiolase) — thiolytic cleavage → propionyl-CoA + acyl-CoA (shortened)
  ────────────────────────────────────────────────────────────────────────────────────────────────────────
  SCP2 (SCPx) = STEP 4 (thiolytic cleavage) for ALL branched-chain substrates
  [Straight-chain VLCFA uses a different peroxisomal thiolase (ACAA1); SCPx is branched-chain-specific]

BIOCHEMICAL PROFILE (SCP2 deficiency):
  Pristanic acid: ELEVATED (cannot complete step 4 thiolysis → pristanoyl-CoA accumulates after step 3)
  THCA / DHCA:    ELEVATED (bile acid intermediate chain shortening blocked at thiolytic step)
  VLCFA (C26:0):  NORMAL or only mildly elevated (straight-chain beta-oxidation = ACAA1, not SCPx)
  Phytanic acid:  NORMAL (PHYH alpha-oxidation is intact; phytanic → pristanic step is normal)
  Plasmalogens:   NORMAL (PTS2/plasmalogen biosynthesis intact — FAR1/AGPS/GNPAT/PEX7 all normal)

CRITICAL BIOCHEMICAL DISTINCTIONS:
  SCP2 vs HSD17B4 (D-Bifunctional Protein):
    VLCFA: NORMAL in SCP2 (SIGNIFICANTLY ELEVATED in HSD17B4) — CRITICAL DISTINCTION.
    Both have pristanic ELEVATED and THCA/DHCA ELEVATED (different steps, same biochemical outcome).
    Plasmalogens NORMAL in BOTH.
    KEY: VLCFA measurement MANDATORY — if VLCFA elevated → think HSD17B4 first; if VLCFA NORMAL
    but pristanic + THCA elevated → think SCP2 (thiolase) or AMACR (racemase).
  SCP2 vs AMACR (Alpha-methylacyl-CoA Racemase):
    Both pristanic and THCA elevated; VLCFA normal/mildly elevated in both.
    AMACR blocks racemization of (R)-pristanic-CoA → (S)-form BEFORE step 1 (ACOX2); SCP2 blocks step 4.
    Biochemically very similar — gene sequencing distinguishes.
    AMACR: adult-onset epilepsy prominent (focal, temporal); SCP2: movement disorder more prominent.
  SCP2 vs PHYH (Adult Refsum Disease):
    Phytanic acid SEVERELY ELEVATED in Refsum (NORMAL in SCP2).
    VLCFA NORMAL in both. Pristanic mildly elevated in Refsum; SEVERELY elevated in SCP2.
    Refsum: RP + polyneuropathy + anosmia + deafness; SCP2: movement disorder + azoospermia.
  SCP2 vs ACOX1 (Pseudo-NALD):
    VLCFA ELEVATED in ACOX1 (NORMAL in SCP2).
    Pristanic NORMAL in ACOX1 (ELEVATED in SCP2).
    ACOX1: neonatal onset; SCP2: adult onset.

SCP2 PROTEIN BIOLOGY:
  Gene SCP2 at 1p32.3 (OMIM Gene: *184755).
  SCPx = 547 aa; peroxisomal matrix; PTS1 (C-terminal SKL).
  Two functional regions: thiolase domain (N-terminal, ~424 aa) + SCP2 lipid-transfer domain (C-terminal, 123 aa).
  SCPx thiolase cleaves 2-methyl-branched 3-oxoacyl-CoA → propionyl-CoA + (n-2)-acyl-CoA.
  Thiolase activity: requires CoA; Mg2+ co-factor; active site Cys95 (nucleophilic) + Cys458 (acid-base).
  SCPx is the only peroxisomal thiolase active on branched-chain substrates; ACAA1 is the straight-chain
  thiolase → hence VLCFA beta-oxidation is unaffected in SCP2 deficiency.

OMIM: Gene *184755 (SCP2) | Disease #613706 (adult-onset leukoencephalopathy + dystonia + hypogonadism)
LOCUS: 1p32.3
EPIDEMIOLOGY: Extremely rare — ~20 cases published worldwide 2026.
  First described by Ferdinandusse et al. 2006 (Brain) — adult male with polyneuropathy, movement disorder,
  azoospermia. Additional cases from Wanders lab and Dutch group.
  Adult-onset phenotype: completely distinct from neonatal peroxisomal disorders.

CLINICAL PHENOTYPE:
  Adult-onset (typically 20s–50s) — COMPLETELY different from HSD17B4 (neonatal) and ACOX1 (infantile).
  Core features (~20 published cases):
  • Movement disorder (extrapyramidal): dystonia, chorea, myoclonus, athetosis — most prominent feature
  • Peripheral polyneuropathy (axonal): length-dependent, lower limbs first
  • Seizures: present in ~40% — myoclonic, focal/temporal; less severe than neonatal peroxisomal disorders
  • Azoospermia / male infertility (SCPx critical for testicular germ cell maturation — both thiolase activity
    and SCP2 lipid transfer function required for spermatogenesis)
  • Cognitive decline (mild–moderate, later stage)
  • Elevated liver enzymes in subset
  • Leukoencephalopathy on MRI (white matter T2 changes, variable severity)
  • SNHL: ~30% (lower than neonatal peroxisomal disorders)

MALE REPRODUCTIVE PHENOTYPE:
  Azoospermia in virtually all affected males (near 100%) — one of the most clinically distinctive features.
  SCPx is abundantly expressed in sertoli cells and primary spermatocytes.
  SCP2 domain mediates cholesterol and oxysterol transfer critical for testicular steroidogenesis and
  spermatid maturation. Loss → severe hypospermatogenesis → azoospermia.
  This male-predominant phenotype resembles AMACR (also causes azoospermia in some).

ERT/HSCT/GENE THERAPY:
  No ERT (2026) — SCPx is a peroxisomal matrix enzyme; no secreted form for systemic delivery.
  No HSCT — SCP2 deficiency is NOT inflammatory demyelination; HSCT targets only inflammatory demyelinators.
  No gene therapy approved 2026.

DIET THERAPY:
  Phytol-restricted diet: Level C — reduces dietary phytol → less phytanic acid (via gut microbiome and
  chlorophyll phytol) → less pristanic acid substrate for peroxisomal beta-oxidation.
  Practical: avoid large amounts of dairy, ruminant fats, certain fish. Limited evidence base (very rare disease).

TREATMENT PRINCIPLES:
  LEV first-line for seizures (no peroxisomal interactions; no adrenal mechanism).
  Clonazepam for myoclonus (Level C).
  PHT/CBZ: CAN USE (no adrenal insufficiency — distinct from ABCD1 where ABSOLUTE CI).
  VPA: RELATIVE CI — hepatotoxicity (standard 3 mechanisms) + bile acid metabolism disruption (THCA/DHCA
  accumulation adds hepatic metabolic burden, similar to HSD17B4 reasoning); POLG1 MANDATORY CPIC A.
  VGB: RELATIVE CI — VGB causes irreversible peripheral visual field constriction; in adult-onset patients
  with already-present polyneuropathy and potential visual deficits from movement disorder, VGB poses
  additional quality-of-life risk; no absolute CI mechanism but risk:benefit unfavourable.
  Lorenzo oil: NOT APPLICABLE — SCP2 blocks step 4 (thiolase), not VLCFA import (ABCD1) or step 1
  (ACOX1). Lorenzo oil reduces VLCFA elongation but cannot affect pristanic/THCA thiolysis.
"""

import random
random.seed(53)


# ── Overview (KPIs + summary) ─────────────────────────────────────────────────

def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 40,
        "movement_disorder_pct": 90,
        "azoospermia_pct_males": 95,
        "polyneuropathy_pct": 75,
        "drug_resistance_pct": 20,
        "leuko_pct": 55,
        "snhl_pct": 30,
        "adrenal_insufficiency_pct": 0,
        "vlcfa_normal_pct": 85,
        "pristanic_elevated_pct": 100,
        "thca_elevated_pct": 92,
        "plasmalogen_normal_pct": 98,
        "omim_gene": "184755",
        "omim_disease": "613706",
        "locus": "1p32.3",
        "vlcfa_elevated": False,
        "inheritance": "Autosomal recessive (AR), biallelic LOF — both sexes; males more severely affected due to azoospermia",
        "common_variant": "No common founder mutation — all private variants; ~20 cases worldwide 2026; extremely rare",
        "onset_age": "Adult-onset (typically 20s–50s) — COMPLETELY different from HSD17B4 (neonatal) and ACOX1 (infantile)",
        "disease_mechanism": (
            "SCP2/SCPx (547 aa, PTS1: SKL) is the Step 4 (thiolytic cleavage) enzyme for BRANCHED-CHAIN "
            "peroxisomal beta-oxidation. SCPx thiolase cleaves 2-methyl-branched 3-oxoacyl-CoA substrates "
            "(pristanoyl-CoA, THCA-CoA, DHCA-CoA) into propionyl-CoA + shortened acyl-CoA. "
            "SCP2 deficiency → pristanic acid ELEVATED + THCA/DHCA ELEVATED; VLCFA NORMAL (straight-chain "
            "thiolase ACAA1 is intact). KEY DISTINCTION from HSD17B4: VLCFA NORMAL in SCP2 "
            "(SIGNIFICANTLY ELEVATED in HSD17B4). Adult-onset movement disorder + polyneuropathy + seizures "
            "(~40%) + azoospermia (males ~95%). No adrenal insufficiency. PHT/CBZ CAN USE. "
            "VPA RELATIVE CI (bile acid burden + POLG1). VGB RELATIVE CI (quality-of-life risk in adult). "
            "No ERT. No HSCT. LEV first-line. Phytol-restricted diet Level C."
        ),
        "nbs_positive_rate": (
            "Not in standard NBS (adult-onset disease). Pristanic acid + THCA/DHCA plasma in adult "
            "with movement disorder + polyneuropathy + azoospermia. VLCFA normal → consider SCP2 or AMACR. "
            "Gene sequencing (SCP2 + AMACR panel) to distinguish. ~20 cases worldwide; most undiagnosed."
        ),
        "key_concepts": [
            "SCP2/SCPx (547 aa, PTS1-SKL, 1p32.3): Step 4 (thiolytic cleavage) branched-chain peroxisomal beta-oxidation — cleaves 2-methyl-branched 3-oxoacyl-CoA → propionyl-CoA + shortened acyl-CoA",
            "SCP2 deficiency → PRISTANIC SEVERELY ELEVATED + THCA/DHCA ELEVATED; VLCFA NORMAL (straight-chain thiolase ACAA1 intact) — CRITICAL DISTINCTION FROM HSD17B4 (VLCFA ELEVATED in HSD17B4)",
            "ADULT-ONSET phenotype (20s–50s) — COMPLETELY distinct from HSD17B4 (neonatal) and ACOX1 (infantile); ~20 cases worldwide 2026; AR biallelic LOF; no founder mutation",
            "Core triad: movement disorder (dystonia/chorea/myoclonus, 90%) + polyneuropathy (75%) + azoospermia in males (~95%) — azoospermia is the most specific clinical marker",
            "Seizures in ~40% — myoclonic and focal; LESS SEVERE than neonatal peroxisomal disorders; drug resistance ~20%",
            "No adrenal insufficiency — PHT/CBZ/OXC CAN USE (no adrenal mechanism; contrast ABCD1 ABSOLUTE CI)",
            "VPA: RELATIVE CI — hepatotoxicity (3 mechanisms) + bile acid accumulation (THCA/DHCA disrupts hepatic metabolism); POLG1 MANDATORY (CPIC Grade A)",
            "VGB: RELATIVE CI — irreversible peripheral VF constriction in adult with existing polyneuropathy/movement disorder; risk:benefit unfavourable; no absolute CI mechanism",
            "Phytol-restricted diet Level C — reduces dietary precursor load (phytol → phytanic → pristanic); practical dietary guidance but limited evidence base given disease rarity",
            "No ERT (2026) — SCPx is peroxisomal matrix enzyme (PTS1-SKL); no secreted isoform; ERT cannot reach peroxisomal matrix",
            "No HSCT — SCP2 deficiency is substrate accumulation toxicity (NOT inflammatory demyelination); HSCT targets only inflammatory demyelinators (Krabbe, ABCD1-CCALD)",
            "SCPx also produces shorter SCP2 protein (123 aa, same gene, alternative promoter) — non-specific lipid transfer protein (nsLTP); cholesterol/sterol trafficking; both products needed for spermatogenesis",
            "KEY DISTINCTION from AMACR: both have pristanic + THCA elevated + VLCFA normal; AMACR blocks racemase step BEFORE step 1 (ACOX2); SCP2 blocks thiolase step AFTER step 3 (HSD17B4); gene sequencing distinguishes",
            "KEY DISTINCTION from PHYH (Refsum): phytanic acid SEVERELY ELEVATED in Refsum (NORMAL in SCP2); pristanic mildly elevated in Refsum (SEVERELY elevated in SCP2); different dietary intervention needed",
            "LEV first-line. Clonazepam for myoclonus. PHT/CBZ focal (CAN USE). POLG1 MANDATORY before VPA. Phytol-restricted diet Level C. DHA Level C. No ERT. No HSCT.",
        ],
        "standards": [
            "Ferdinandusse S, Jimenez-Sanchez G, Koster J, et al. A novel bile acid biosynthesis defect due to a deficiency of peroxisomal ABCD3. Hum Mol Genet. 2015;24(2):361–370.",
            "Ferdinandusse S, Kostopoulos P, Denis S, et al. Mutations in the gene encoding peroxisomal sterol carrier protein X (SCPx) cause leukencephalopathy with dystonia and motor neuropathy. Am J Hum Genet. 2006;78(6):1046–1052.",
            "Waterham HR, Ferdinandusse S, Wanders RJA. Human disorders of peroxisome metabolism and biogenesis. Biochim Biophys Acta. 2016;1863(5):922–933.",
            "Wanders RJA, Waterham HR. Biochemistry of mammalian peroxisomes revisited. Annu Rev Biochem. 2006;75:295–332.",
            "CPIC Guideline — Valproic Acid and POLG1 (Grade A). cpicpgx.org. 2023.",
            "Engelen M, Ofman R, Dijkgraaf MGW, et al. Treatment of adult patients with X-linked adrenoleukodystrophy. Expert Opin Pharmacother. 2012;13(2):265–274.",
        ],
    }


# ── Breakdown (patients + seizures + treatments) ─────────────────────────────

def get_breakdown():
    etiologies = [
        {
            "name": "Classic Adult-Onset (Movement Disorder + Polyneuropathy + Azoospermia)",
            "pct": 60,
            "n": 24,
            "sex": "M/F (males more severely affected — azoospermia in ~95% males)",
            "onset_age": "20s–40s (adult-onset; mean ~32 years at symptom onset)",
            "seizure_risk": "35–45%",
            "eeg": "Multifocal myoclonic discharges, temporal focal epileptiform activity, photoparoxysmal response occasionally",
            "mri": "White matter T2 hyperintensities (leukoencephalopathy), predominantly subcortical/periventricular; variable severity",
            "diet_therapy": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic null or missense variants abolishing SCPx thiolase activity. "
                "Both thiolase (N-terminal) and SCP2 lipid-transfer (C-terminal) functions impaired. "
                "Pristanic acid SEVERELY ELEVATED. THCA/DHCA elevated. VLCFA NORMAL. "
                "Adult-onset movement disorder (dystonia/chorea/myoclonus 90%) dominant. "
                "Peripheral polyneuropathy (axonal, length-dependent, lower limbs first) 75%. "
                "Azoospermia in ALL affected males (~95%). Seizures 40%. "
                "Cognitive decline mild-moderate later in disease. Leukoencephalopathy on MRI ~55%. "
                "Elevated liver enzymes in ~35%. Drug resistance ~20%. SNHL ~30%."
            ),
        },
        {
            "name": "Predominant Neuropathy Variant (Sensorimotor > Movement Disorder)",
            "pct": 25,
            "n": 10,
            "sex": "M/F equal",
            "onset_age": "30s–50s",
            "seizure_risk": "30–40%",
            "eeg": "Focal temporal discharges, occasional generalized myoclonic correlates, normal interictal in subset",
            "mri": "Mild–moderate periventricular T2 signal; some patients near-normal early",
            "diet_therapy": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic variants with partial residual SCPx thiolase activity (hypomorphic alleles). "
                "Peripheral neuropathy prominent (steppage gait, areflexia, EMG: axonal). "
                "Movement disorder present but milder (myoclonus >> dystonia). "
                "Pristanic elevated; THCA elevated; VLCFA normal. Plasmalogens normal. "
                "Seizures ~35% (mostly focal and myoclonic). Azoospermia in affected males. "
                "Cognition relatively preserved early; slow progression."
            ),
        },
        {
            "name": "Severe Early-Adult Onset (Rapid Progression + Leukoencephalopathy)",
            "pct": 15,
            "n": 6,
            "sex": "M/F equal",
            "onset_age": "Late teens–late 20s",
            "seizure_risk": "50–65%",
            "eeg": "Generalised spike-wave, myoclonic runs, background slowing with theta",
            "mri": "Confluent white matter T2 signal (diffuse leukoencephalopathy), bilateral symmetric, early severe",
            "diet_therapy": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic null variants (frameshift + nonsense) — complete absence of SCPx activity. "
                "Most severe phenotype. Earlier onset (late teens-20s). "
                "Rapid leukoencephalopathy progression. Seizures more prominent (50–65%). "
                "Severe drug-resistant myoclonic epilepsy requiring combination AED therapy. "
                "Pristanic SEVERELY elevated (highest levels). THCA/DHCA markedly elevated. "
                "Azoospermia in all males. Cognitive impairment prominent early. "
                "Life expectancy significantly reduced; dependent ADL by 30–40s."
            ),
        },
    ]

    phenotypes = (
        [("Classic", "Moderate")] * 24
        + [("Neuropathy-predominant", "Mild")] * 10
        + [("Severe early-adult", "Severe")] * 6
    )
    sexes = ["M"] * 22 + ["F"] * 18
    random.shuffle(sexes)

    genotype_map = {
        "Classic": [
            "Biallelic missense (thiolase domain)",
            "Frameshift + missense (compound heterozygous)",
            "Splice-site + nonsense",
        ],
        "Neuropathy-predominant": [
            "Biallelic hypomorphic missense",
            "Missense + splice (partial activity residual)",
            "Compound het — mild + moderate alleles",
        ],
        "Severe early-adult": [
            "Biallelic null (frameshift + nonsense)",
            "Large deletion 1p32.3",
            "Homozygous frameshift",
        ],
    }

    seizure_map = {
        "Severe": ["Generalised myoclonic", "Focal with bilateral spread", "SE (myoclonic status)"],
        "Moderate": ["Focal temporal", "Myoclonic", None],
        "Mild": ["Focal temporal", None, None],
    }

    trigger_pool = [
        "Sleep deprivation",
        "Febrile illness",
        "Missed AED",
        "Stress / exertion",
        "Metabolic decompensation",
        "Fasting / dietary lapse",
        "Photosensitivity",
    ]

    treatment_pool = [
        "LEV (first-line)",
        "Clonazepam (myoclonus — Level C)",
        "Phytol-restricted diet (Level C)",
        "OXC/CBZ (focal — CAN USE, no adrenal)",
        "DHA supplementation (Level C)",
        "Lacosamide (focal adjunct — Level C)",
    ]

    patients = []
    for i, ((pheno, severity), sex) in enumerate(zip(phenotypes, sexes)):
        pid = f"SCP2-{i+1:02d}"

        if severity == "Severe":
            onset_yr = random.randint(17, 25)
            has_sz = random.random() < 0.58
            pristanic = round(random.uniform(4.0, 18.0), 2)
            vlcfa_c26 = round(random.uniform(0.20, 0.50), 3)  # normal range <0.3 µmol/L (mildly elevated in some)
            leuko_grade = random.choice(["Severe", "Severe", "Moderate"])
            movement_disorder = True
        elif severity == "Moderate":
            onset_yr = random.randint(22, 45)
            has_sz = random.random() < 0.40
            pristanic = round(random.uniform(2.5, 10.0), 2)
            vlcfa_c26 = round(random.uniform(0.15, 0.40), 3)
            leuko_grade = random.choice(["Moderate", "Mild", "None"])
            movement_disorder = True
        else:
            onset_yr = random.randint(30, 52)
            has_sz = random.random() < 0.33
            pristanic = round(random.uniform(1.5, 6.0), 2)
            vlcfa_c26 = round(random.uniform(0.12, 0.30), 3)
            leuko_grade = random.choice(["None", "Mild"])
            movement_disorder = random.random() < 0.65

        seizure_type = random.choice(seizure_map[severity]) if has_sz else None
        drug_resistant = has_sz and random.random() < 0.20
        triggers = random.sample(trigger_pool, k=random.randint(1, 3)) if has_sz else []
        current_treatment = random.sample(treatment_pool, k=random.randint(2, 3))
        genotype = random.choice(genotype_map[pheno])

        patients.append({
            "id": pid,
            "phenotype": pheno,
            "severity": severity,
            "sex": sex,
            "onset_age_years": onset_yr,
            "has_seizures": has_sz,
            "seizure_type": seizure_type,
            "drug_resistant": drug_resistant,
            "movement_disorder": movement_disorder,
            "polyneuropathy": random.random() < (0.90 if severity == "Severe" else 0.75 if severity == "Moderate" else 0.55),
            "azoospermia": sex == "M" and random.random() < 0.95,
            "snhl": random.random() < (0.40 if severity == "Severe" else 0.30 if severity == "Moderate" else 0.20),
            "leukodystrophy_grade": leuko_grade,
            "pristanic_umol_l": pristanic,
            "vlcfa_c26_umol_l": vlcfa_c26,
            "thca_elevated": random.random() < (0.98 if severity == "Severe" else 0.90),
            "plasmalogen_normal": True,
            "genotype": genotype,
            "current_treatment": current_treatment,
            "triggers": triggers,
            "polg1_tested": True,
            "phytol_restricted_diet": random.random() < 0.55,
        })

    seizure_types = [
        {
            "type": "Myoclonic",
            "pct": 42,
            "preferred_tx": "LEV (first-line) + clonazepam (add-on Level C)",
            "notes": "Most common seizure type in SCP2 — multifocal myoclonus, often worse in morning. Clonazepam effective for myoclonus control. VPA RELATIVE CI. Valproate POLG1 mandatory before use.",
        },
        {
            "type": "Focal (temporal onset)",
            "pct": 35,
            "preferred_tx": "LEV, OXC (CAN USE — no adrenal mechanism)",
            "notes": "Temporal lobe origin common (pristanic acid toxicity / leukoencephalopathy pattern). PHT/CBZ/OXC can be used (no adrenal insufficiency — unlike ABCD1 where enzyme inducers = ABSOLUTE CI).",
        },
        {
            "type": "Generalised tonic-clonic (secondary generalisation)",
            "pct": 15,
            "preferred_tx": "LEV + OXC combination",
            "notes": "Secondary generalisation from focal onset; not primary GTC. LEV + OXC combination reasonable. VPA RELATIVE CI (POLG1 mandatory, bile acid burden). IV LEV for acute management.",
        },
        {
            "type": "Myoclonic status epilepticus",
            "pct": 8,
            "preferred_tx": "IV LEV + clonazepam; avoid VPA acute",
            "notes": "Rare, seen in severe early-adult phenotype. IV lorazepam/diazepam → IV LEV. VPA caution (RELATIVE CI) — if used acutely, hepatic monitoring mandatory. IV glucose if prolonged fasting.",
        },
    ]

    triggers = [
        {"trigger": "Sleep deprivation", "pct": 62},
        {"trigger": "Febrile / intercurrent illness", "pct": 48},
        {"trigger": "Missed AED", "pct": 38},
        {"trigger": "Stress / physical exertion", "pct": 35},
        {"trigger": "Metabolic decompensation", "pct": 22},
        {"trigger": "Dietary lapse (high phytol intake)", "pct": 18},
        {"trigger": "Photosensitivity (subset)", "pct": 12},
    ]

    monitoring = [
        {
            "parameter": "Plasma pristanic acid",
            "threshold": "Target: normal (<3.0 µmol/L); SEVERELY ELEVATED in SCP2; primary biochemical marker; goal: reduce with phytol-restricted diet",
            "frequency": "Every 6 months; every 3 months on dietary intervention",
        },
        {
            "parameter": "Plasma THCA / DHCA (bile acid intermediates)",
            "threshold": "ELEVATED in SCP2 (step 4 thiolytic block); distinguishes from PHYH/Refsum (normal in Refsum); guides diet and monitoring",
            "frequency": "Every 6 months",
        },
        {
            "parameter": "Plasma VLCFA (C26:0, C24:0, C26:0/C22:0 ratio)",
            "threshold": "NORMAL or only mildly elevated — KEY DISTINCTION from HSD17B4/ACOX1/ZSD (all significantly elevated); confirms single-thiolase defect",
            "frequency": "At diagnosis; every 12 months for monitoring",
        },
        {
            "parameter": "Plasma phytanic acid",
            "threshold": "NORMAL — confirms SCP2 (not PHYH/Refsum where phytanic SEVERELY ELEVATED); obtain at diagnosis to exclude Refsum; alpha-oxidation intact in SCP2",
            "frequency": "At diagnosis",
        },
        {
            "parameter": "RBC plasmalogens (C16:0-DMA, C18:0-DMA)",
            "threshold": "NORMAL — confirms plasmalogen biosynthesis intact (PTS2/plasmalogen pathway unaffected); rules out RCDP (severely low) and ZSD (low)",
            "frequency": "At diagnosis",
        },
        {
            "parameter": "Liver function tests (LFTs) + bile acid profile",
            "threshold": "Elevated LFTs in ~35%; THCA/DHCA accumulation causes hepatic metabolic stress; critical if VPA used (RELATIVE CI — bile acid burden adds to 3 standard mechanisms)",
            "frequency": "Every 3 months on VPA; every 6 months otherwise",
        },
        {
            "parameter": "EMG / nerve conduction study (polyneuropathy)",
            "threshold": "Axonal sensorimotor neuropathy — velocity mildly reduced, amplitude decreased; monitor progression; lower limbs first",
            "frequency": "Every 12 months; or new symptoms",
        },
        {
            "parameter": "Brain MRI (leukoencephalopathy + movement disorder progression)",
            "threshold": "White matter T2 hyperintensities; track progression annually; severe cases: confluent periventricular signal",
            "frequency": "Every 12 months",
        },
        {
            "parameter": "Semen analysis (males — azoospermia)",
            "threshold": "Azoospermia in ~95% of affected males; fertility counselling mandatory; sperm banking prior to confirmed diagnosis impossible (diagnostic delay common)",
            "frequency": "At diagnosis in males; fertility counselling offered",
        },
    ]

    lifecycle = [
        {
            "stage": "Pre-symptomatic adult (teens–20s)",
            "features": "Biochemical diagnosis only if family member identified. Pristanic + THCA elevated; VLCFA normal. Normal neurological exam early. Males: azoospermia may be initial presenting symptom. Phytol-restricted diet started.",
        },
        {
            "stage": "Early symptomatic (20s–30s)",
            "features": "Movement disorder onset (myoclonus most common first sign). Subtle polyneuropathy (paresthesias, reduced reflexes lower limbs). Seizures begin (40% by this stage). EEG shows myoclonic discharges. MRI may show early white matter changes. Azoospermia confirmed in males.",
        },
        {
            "stage": "Moderate disease (30s–40s)",
            "features": "Movement disorder progresses (dystonia + chorea + myoclonus combination). Polyneuropathy established (EMG: axonal). Seizure control with LEV +/- clonazepam. Leukoencephalopathy progresses on MRI. Cognitive slowing begins. Dietary compliance reviewed.",
        },
        {
            "stage": "Advanced disease (40s–50s)",
            "features": "Severe dyskinesia + dystonia; wheelchair dependency in severe cases. Drug-resistant seizures in ~20%. Significant cognitive impairment. Swallowing difficulties in severe. SNHL may worsen. Hepatic monitoring ongoing (LFTs + bile acid profile).",
        },
        {
            "stage": "Late stage (50s+)",
            "features": "Rare survival information — very few published cases followed past 50s. In classic phenotype: severe neurological disability; dependent ADL. In neuropathy-predominant variant: slower progression. Palliative and supportive care dominant.",
        },
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "SV2A modulator",
            "level": "First-line (Level A — all forms, all seizure types)",
            "dose": "500–3000 mg/day in 2 divided doses; titrate based on response and tolerability",
            "notes": (
                "No peroxisomal interactions; no adrenal effect; no impact on pristanic/THCA metabolism. "
                "Effective for both myoclonic and focal seizure types in SCP2. "
                "IV formulation for acute management / status epilepticus. "
                "Safe across all SCP2 phenotypes. First choice without exception."
            ),
            "ci": "None specific to SCP2",
        },
        {
            "drug": "Clonazepam (CZP)",
            "class": "Benzodiazepine — GABA-A potentiator",
            "level": "Level C (myoclonus — highly effective)",
            "dose": "0.5–4 mg/day in 2–3 divided doses; start low (0.25 mg/day), titrate slowly",
            "notes": (
                "Particularly effective for cortical myoclonus and action myoclonus in SCP2. "
                "No peroxisomal interactions. "
                "Sedation is the main limitation — start low. Tolerance may develop with prolonged use. "
                "Preferred for myoclonus over valproate (which is RELATIVE CI in SCP2)."
            ),
            "ci": "Respiratory depression; sedation risk; avoid combination with other CNS depressants without monitoring",
        },
        {
            "drug": "Phytol-restricted diet",
            "class": "Dietary intervention",
            "level": "Level C",
            "dose": "Reduce phytol-containing foods: dairy fat, ruminant fats, certain oily fish; supervised by metabolic dietitian",
            "notes": (
                "Phytol (from chlorophyll in green vegetables, dairy, ruminant fat) → phytanic acid → "
                "(via PHYH alpha-oxidation) → pristanic acid → enters peroxisomal beta-oxidation. "
                "Restricting phytol reduces pristanic acid substrate load for SCPx thiolase. "
                "Limited evidence base (very rare disease, no RCTs); biochemical response variable. "
                "Metabolic dietitian supervision mandatory — ensure adequate nutrition. "
                "NOT a cure; adjunct only. Similar rationale to Refsum disease diet."
            ),
            "ci": "Risk of nutritional deficiency if too restrictive — dietitian supervision mandatory",
        },
        {
            "drug": "OXC / CBZ (Oxcarbazepine / Carbamazepine)",
            "class": "Sodium channel blocker",
            "level": "Level B (focal seizures — CAN USE)",
            "dose": "OXC 300–1800 mg/day; CBZ 200–1200 mg/day",
            "notes": (
                "CAN BE USED in SCP2 — no adrenal insufficiency (contrast ABCD1/X-ALD: PHT/CBZ = "
                "ABSOLUTE CI because CYP3A4 cortisol degradation → adrenal crisis). "
                "SCP2 has no adrenal involvement → enzyme inducers safe from adrenal perspective. "
                "Standard hepatic and cardiac monitoring applies. "
                "Effective for focal temporal seizures in SCP2."
            ),
            "ci": "None specific to SCP2; standard SJS/TEN monitoring; hyponatraemia with OXC",
        },
        {
            "drug": "DHA (Docosahexaenoic acid)",
            "class": "Omega-3 supplementation",
            "level": "Level C",
            "dose": "500–1000 mg/day in adults",
            "notes": (
                "Secondary DHA deficit from impaired peroxisomal beta-oxidation of omega-3 precursors. "
                "Neuronal membrane integrity and synaptogenesis. Safe, no significant drug interactions. "
                "Less definitive evidence than in neonatal peroxisomal disorders but used empirically."
            ),
            "ci": "None",
        },
        {
            "drug": "Lacosamide",
            "class": "Selective sodium channel slow-inactivation modulator",
            "level": "Level C adjunct (focal seizures)",
            "dose": "100–400 mg/day in 2 divided doses",
            "notes": (
                "Adjunct for focal seizures inadequately controlled with LEV + OXC. "
                "No peroxisomal interactions. No adrenal mechanism. "
                "Useful in combination with LEV or OXC for refractory focal seizures."
            ),
            "ci": "PR prolongation (ECG monitoring at initiation); adjust for hepatic impairment",
        },
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "level": "RELATIVE CI — hepatotoxicity + POLG1 MANDATORY (CPIC Grade A) + bile acid burden",
            "reason": (
                "THREE hepatotoxicity mechanisms: (a) carnitine depletion, (b) peroxisomal beta-oxidation "
                "interference (VPA is metabolised via peroxisomal beta-oxidation — in SCP2 this adds to "
                "existing thiolase deficiency block at step 4, worsening substrate accumulation), "
                "(c) mitochondrial toxicity (POLG1 mandatory CPIC A). "
                "ADDITIONAL CONCERN in SCP2: THCA/DHCA accumulation already disrupts bile acid "
                "hepatic metabolism — VPA hepatotoxicity risk increased. POLG1 test MANDATORY before VPA. "
                "LFT monitoring q3 months if used. Clonazepam + LEV preferred for myoclonus."
            ),
            "alternative": "LEV (first-line) · Clonazepam (myoclonus) · OXC (focal)",
        },
        {
            "drug": "Vigabatrin (VGB)",
            "level": "RELATIVE CI — irreversible peripheral VF constriction in adult with polyneuropathy/movement disorder",
            "reason": (
                "VGB causes irreversible bilateral peripheral visual field constriction (NAION-like). "
                "In adult SCP2 patients who already have polyneuropathy and movement disorder affecting "
                "mobility and independence, additional irreversible visual impairment significantly "
                "reduces quality of life. NO ABSOLUTE CI mechanism (no retinal dystrophy like HSD17B4/ACOX1) "
                "but risk:benefit ratio unfavourable in most SCP2 cases. "
                "Monthly VEP/VF mandatory if VGB used. Baseline ophthalmological assessment required."
            ),
            "alternative": "LEV (first-line) · Clonazepam (myoclonus)",
        },
        {
            "drug": "PHT/CBZ (phenytoin/carbamazepine) — clarification",
            "level": "CAN USE — no adrenal mechanism (CRITICAL CONTRAST to ABCD1)",
            "reason": (
                "PHT/CBZ carry NO adrenal-insufficiency risk in SCP2 (no adrenal crisis mechanism). "
                "CRITICAL CONTRAST: In ABCD1/X-ALD, PHT/CBZ = ABSOLUTE CI (CYP3A4 enzyme inducers "
                "accelerate cortisol metabolism → adrenal crisis). SCP2 does NOT cause adrenal insufficiency → "
                "PHT/CBZ safe. Standard hepatic and cardiac monitoring applies. "
                "Prefer OXC over CBZ (fewer drug interactions, better tolerability in adults with polyneuropathy)."
            ),
            "alternative": "LEV preferred first-line; OXC preferred over CBZ for focal seizures",
        },
        {
            "drug": "Lorenzo oil",
            "level": "NOT APPLICABLE — mechanism mismatch (not a VLCFA transport or elongation problem)",
            "reason": (
                "Lorenzo oil inhibits VLCFA elongase → may reduce plasma C26:0 but VLCFA is NORMAL in SCP2. "
                "The metabolic block in SCP2 is at step 4 THIOLASE (branched-chain) — Lorenzo oil "
                "does NOT affect SCPx thiolase activity or pristanic/THCA accumulation. "
                "No rationale and no benefit in SCP2 deficiency."
            ),
            "alternative": "Phytol-restricted diet (Level C) · LEV · Clonazepam for myoclonus",
        },
        {
            "drug": "Fasting / prolonged caloric restriction",
            "level": "CAUTION — metabolic stress increases peroxisomal beta-oxidation substrate load",
            "reason": (
                "Fasting mobilises stored fatty acids including branched-chain substrates. "
                "In SCP2 deficiency, increased substrate flux through blocked step 4 → pristanic + "
                "THCA accumulation surge. Less acute hazard than PHYH/Refsum (where phytanic surge causes "
                "acute paralysis) but still advisable to avoid prolonged fasting. "
                "Regular meals; avoid prolonged caloric restriction. IV glucose if acute illness causes anorexia."
            ),
            "alternative": "Regular meals; avoid prolonged fasting; IV glucose if acutely ill with anorexia",
        },
    ]

    return {
        "etiologies": etiologies,
        "patients": patients,
        "seizure_types": seizure_types,
        "triggers": triggers,
        "monitoring": monitoring,
        "lifecycle": lifecycle,
        "treatments": treatments,
        "contraindications": contraindications,
    }


# ── Definitions ───────────────────────────────────────────────────────────────

def get_definitions():
    return {
        "key_concepts": [
            "SCP2/SCPx (547 aa, PTS1-SKL, 1p32.3, OMIM *184755): Step 4 (thiolytic cleavage) branched-chain peroxisomal beta-oxidation — cleaves 2-methyl-branched 3-oxoacyl-CoA → propionyl-CoA + shortened acyl-CoA",
            "SCP2 DEFICIENCY → PRISTANIC SEVERELY ELEVATED + THCA/DHCA ELEVATED; VLCFA NORMAL — straight-chain beta-oxidation (using ACAA1 thiolase) is INTACT in SCP2 deficiency",
            "VLCFA NORMAL IN SCP2 — CRITICAL DISTINCTION from HSD17B4 (VLCFA significantly elevated in HSD17B4): both have pristanic + THCA elevated, but VLCFA normal SCP2 vs elevated HSD17B4",
            "ADULT-ONSET (~20s–50s): COMPLETELY distinct from neonatal/infantile peroxisomal disorders (HSD17B4, ACOX1, PEX1/6); ~20 cases worldwide 2026; AR biallelic LOF SCP2; no founder mutation",
            "Core clinical triad: (1) movement disorder (dystonia/chorea/myoclonus, 90%); (2) polyneuropathy (75%); (3) azoospermia in males (~95%) — most specific clinical marker for SCP2 vs AMACR",
            "Seizures in ~40%: myoclonic (42%) + focal temporal (35%) — less severe than neonatal peroxisomal; drug resistance ~20%; LEV + clonazepam combination effective",
            "SCP2 gene produces TWO proteins: SCPx (547 aa, thiolase + SCP2 domain) and SCP2/nsLTP (123 aa, lipid transfer only). Both required for spermatogenesis → azoospermia when SCP2 LOF",
            "No adrenal insufficiency — PHT/CBZ/OXC CAN USE; critical contrast to ABCD1 (PHT/CBZ ABSOLUTE CI — enzyme inducers cause cortisol drop → adrenal crisis). SCP2 has NO adrenal mechanism",
            "VPA: RELATIVE CI — 3 hepatotoxicity mechanisms + bile acid burden (THCA/DHCA accumulation worsens hepatic stress); POLG1 MANDATORY (CPIC Grade A); use clonazepam + LEV for myoclonus instead",
            "VGB: RELATIVE CI — irreversible peripheral VF constriction; in adult with polyneuropathy + movement disorder, additional visual impairment severely impacts quality-of-life; monthly VEP/VF if used",
            "Phytol-restricted diet Level C — reduces dietary phytol → phytanic → pristanic substrate load; metabolic dietitian supervision; limited evidence (very rare disease); not curative",
            "No ERT (2026) — SCPx is peroxisomal matrix enzyme (PTS1-SKL); systemic ERT cannot reach peroxisomal matrix; no secreted isoform",
            "No HSCT — SCP2 deficiency is substrate toxicity (pristanic/THCA accumulation), NOT inflammatory demyelination; HSCT targets inflammatory demyelinators (Krabbe, ABCD1-CCALD) only",
            "KEY DISTINCTION from AMACR: both have pristanic + THCA elevated + VLCFA normal; AMACR blocks racemase step BEFORE ACOX2 (step 1); SCP2 blocks thiolase AFTER HSD17B4 (step 3); gene sequencing distinguishes; azoospermia commoner in SCP2",
            "KEY DISTINCTION from PHYH/Refsum: phytanic SEVERELY ELEVATED in Refsum (NORMAL in SCP2); dietary phytol restriction needed in BOTH but Refsum also restricts phytol more strictly (phytanic the substrate, not pristanic)",
        ],
        "diagnostic_algorithm": [
            "Adult (20s–50s) with movement disorder (myoclonus/dystonia/chorea) + polyneuropathy +/- seizures → order plasma pristanic acid + VLCFA panel + THCA/DHCA + phytanic acid SIMULTANEOUSLY.",
            "If VLCFA NORMAL + pristanic ELEVATED + THCA elevated + phytanic NORMAL → strongly suggests SCP2 or AMACR (branched-chain beta-oxidation defect at thiolase or racemase step).",
            "If VLCFA ELEVATED + pristanic elevated → consider HSD17B4 (steps 2+3 block; both VLCFA + pristanic elevated); order RBC plasmalogens to distinguish from ZSD.",
            "If phytanic SEVERELY ELEVATED + VLCFA NORMAL + pristanic mildly elevated → consider PHYH/Refsum (alpha-oxidation defect); PHYH gene sequencing.",
            "Plasma VLCFA NORMAL in SCP2 — MANDATORY measurement to exclude ZSD/HSD17B4/ACOX1 where VLCFA elevated; if VLCFA unexpectedly elevated, reconsider and order additional peroxisomal panel.",
            "RBC plasmalogens: NORMAL in SCP2 (confirms plasmalogen biosynthesis pathway intact — PEX7/GNPAT/AGPS/FAR1 all normal); severely low → RCDP or ZSD.",
            "SCP2 vs AMACR distinction: AMACR blocks racemase (before step 1); SCP2 blocks thiolase (step 4); biochemically near-identical; GENE SEQUENCING of SCP2 + AMACR panel required to distinguish.",
            "In affected males: semen analysis (azoospermia ~95%); fertility counselling; sperm banking if pre-diagnosis semen available (rarely possible due to diagnostic delay).",
            "Brain MRI: white matter T2 changes (leukoencephalopathy) in ~55%; predominantly subcortical/periventricular; variable severity; not specific for SCP2.",
            "EMG/NCS: axonal sensorimotor neuropathy (reduced amplitude, mildly reduced velocity); distinguishes from demyelinating neuropathies.",
            "SCPx fibroblast enzyme assay (3-oxoacyl-CoA thiolase activity) — gold standard biochemical confirmation; available in specialised peroxisomal labs.",
            "SCP2 gene sequencing: confirms biallelic LOF variants; identifies domain affected (thiolase domain vs SCP2 domain); POLG1 MANDATORY before any VPA consideration.",
        ],
        "pharmacological_distinctions": [
            "VLCFA NORMAL IN SCP2 — THE KEY BIOCHEMICAL MARKER distinguishing SCP2 from HSD17B4 (VLCFA elevated in HSD17B4), ACOX1 (VLCFA elevated), and ZSD (VLCFA elevated). If VLCFA is normal but pristanic elevated → SCP2 or AMACR.",
            "PRISTANIC ACID: SEVERELY ELEVATED in SCP2 (same as HSD17B4). But THCA also elevated in BOTH. The distinguishing feature is VLCFA (normal SCP2 vs elevated HSD17B4). Pristanic + THCA elevation = peroxisomal branched-chain block.",
            "PHT/CBZ/OXC = CAN USE in SCP2 (no adrenal insufficiency). CRITICAL CONTRAST: ABCD1/X-ALD = ABSOLUTE CI (CYP3A4 cortisol degradation → adrenal crisis in ABCD1 males). SCP2 has NO adrenal involvement. Same safe profile as HSD17B4, ACOX1, RCDP series.",
            "VPA: RELATIVE CI in SCP2 — THREE hepatotoxicity mechanisms (carnitine depletion, peroxisomal beta-ox interference at step 4 block, mitochondrial toxicity POLG1 CPIC A) + bile acid burden (THCA/DHCA accumulation adds hepatic stress). Clonazepam + LEV preferred for myoclonus.",
            "VGB: RELATIVE CI in SCP2 (DIFFERENT MECHANISM from neonatal disorders). In neonatal disorders: retinal dystrophy + photoreceptor damage. In SCP2: irreversible peripheral VF constriction in adult with pre-existing polyneuropathy and movement disorder = significant QoL impact. Monthly VEP/VF if used.",
            "Clonazepam: PREFERRED for myoclonus in SCP2 (Level C) — avoids VPA (RELATIVE CI). No peroxisomal interactions. Low dose, slow titration. Effective for action myoclonus and cortical myoclonus in adult-onset peroxisomal disease.",
            "Lorenzo oil: NOT APPLICABLE in SCP2 — VLCFA is normal and Lorenzo oil targets VLCFA elongase; pristanic/THCA accumulation (step 4 block) is UNAFFECTED by Lorenzo oil. No rationale.",
            "Phytol-restricted diet: Level C in SCP2 (different from Refsum where strictly required). Reduces dietary precursor load → less pristanic acid substrate. Practical dietary guidance. Metabolic dietitian supervision mandatory.",
            "DHA supplementation: Level C — secondary DHA deficit from impaired peroxisomal beta-oxidation; less acute than in neonatal forms but used empirically for neuronal membrane support.",
            "Fasting: CAUTION (not EXTREME HAZARD like PHYH/Refsum where acute neuropathy deterioration occurs). In SCP2, fasting increases substrate flux through blocked step 4 → pristanic surge. Regular meals advisable; IV glucose if acutely ill with anorexia.",
            "No ERT — SCPx is peroxisomal matrix enzyme (PTS1-SKL); systemic ERT cannot reach peroxisomal matrix.",
            "No HSCT — substrate accumulation toxicity, not inflammatory demyelination; HSCT does not address peroxisomal thiolase deficiency.",
        ],
        "differential_diagnosis": [
            {
                "condition": "HSD17B4 (D-Bifunctional Protein / DBP / MFP-2 deficiency)",
                "distinction": "VLCFA SIGNIFICANTLY ELEVATED in HSD17B4 (NORMAL in SCP2) — CRITICAL DISTINCTION. Both have pristanic + THCA elevated. HSD17B4 = neonatal onset; SCP2 = adult onset. HSD17B4 blocks steps 2+3 for ALL substrates (VLCFA + pristanic + THCA); SCP2 blocks step 4 for branched-chain only (straight-chain VLCFA via ACAA1 is intact in SCP2).",
            },
            {
                "condition": "AMACR (Alpha-methylacyl-CoA Racemase deficiency)",
                "distinction": "Biochemically near-identical: both pristanic + THCA elevated + VLCFA normal. AMACR blocks racemase step BEFORE ACOX2 (step 1); SCP2 blocks thiolase AFTER HSD17B4 (step 3). Clinical: AMACR may present with adult-onset epilepsy as prominent feature; SCP2: movement disorder + azoospermia more prominent. GENE SEQUENCING (SCP2 + AMACR panel) required to distinguish — biochemistry alone CANNOT.",
            },
            {
                "condition": "PHYH (Phytanoyl-CoA 2-Hydroxylase deficiency / Adult Refsum Disease)",
                "distinction": "Phytanic SEVERELY ELEVATED in Refsum (NORMAL in SCP2). Pristanic mildly elevated in Refsum (SEVERELY elevated in SCP2). Refsum: RP + polyneuropathy + anosmia + SNHL; SCP2: movement disorder + azoospermia + leukoencephalopathy. THCA normal in Refsum. Alpha-oxidation blocked in Refsum (upstream of alpha-ox); in SCP2 alpha-oxidation INTACT.",
            },
            {
                "condition": "ACOX1 (Pseudo-NALD / Acyl-CoA Oxidase 1 deficiency)",
                "distinction": "VLCFA ELEVATED in ACOX1 (NORMAL in SCP2). Pristanic NORMAL in ACOX1 (ELEVATED in SCP2). ACOX1 = neonatal onset (0–3 months); SCP2 = adult onset. ACOX1 blocks step 1 of straight-chain VLCFA only; SCP2 blocks step 4 of branched-chain.",
            },
            {
                "condition": "ZSD / PEX1-PEX6 (Zellweger Spectrum Disorders)",
                "distinction": "VLCFA ELEVATED in ZSD (NORMAL in SCP2). Plasmalogens SEVERELY LOW in ZSD (NORMAL in SCP2). ZSD: ALL peroxisomal functions impaired (biogenesis disorder); SCP2: single enzyme defect. ZSD: neonatal/infantile onset; SCP2: adult. ZSD: phytanic + pristanic + pipecolic ALL elevated (SCP2: only pristanic + THCA elevated).",
            },
            {
                "condition": "ABCD1 (X-linked Adrenoleukodystrophy)",
                "distinction": "VLCFA ELEVATED in ABCD1 (NORMAL in SCP2). Pristanic NORMAL in ABCD1 (ELEVATED in SCP2). ABCD1: adrenal insufficiency mandatory (SCP2: absent). ABCD1: X-linked males; SCP2: AR both sexes. PHT/CBZ ABSOLUTE CI in ABCD1 (CAN USE in SCP2 — no adrenal mechanism).",
            },
            {
                "condition": "RCDP (PEX7/GNPAT/AGPS/FAR1 — Rhizomelic Chondrodysplasia Punctata)",
                "distinction": "Plasmalogens SEVERELY LOW in RCDP (NORMAL in SCP2). VLCFA NORMAL in RCDP (NORMAL in SCP2 too — but pristanic pattern different). RCDP: rhizomelia + stippled epiphyses + cataracts (structural skeletal malformations not seen in SCP2). RCDP: neonatal/infantile; SCP2: adult-onset.",
            },
        ],
        "standards": [
            "Ferdinandusse S, Kostopoulos P, Denis S, et al. Mutations in the gene encoding peroxisomal sterol carrier protein X (SCPx) cause leukencephalopathy with dystonia and motor neuropathy. Am J Hum Genet. 2006;78(6):1046–1052.",
            "Waterham HR, Ferdinandusse S, Wanders RJA. Human disorders of peroxisome metabolism and biogenesis. Biochim Biophys Acta. 2016;1863(5):922–933.",
            "Wanders RJA, Waterham HR. Biochemistry of mammalian peroxisomes revisited. Annu Rev Biochem. 2006;75:295–332.",
            "Ferdinandusse S, Denis S, Mooijer PA, et al. Identification of the peroxisomal beta-oxidation enzymes involved in the degradation of long-chain dicarboxylic acids. J Lipid Res. 2004;45(6):1104–1111.",
            "CPIC Guideline — Valproic Acid and POLG1 (Grade A). cpicpgx.org. 2023.",
            "Wanders RJA, Komen J, Ferdinandusse S. Phytanic acid metabolism in health and disease. Biochim Biophys Acta. 2011;1811(9):498–507.",
        ],
    }
