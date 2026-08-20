#!/usr/bin/env python3
"""ACOX1 / Pseudo-Neonatal Adrenoleukodystrophy (Pseudo-NALD / ACOX1 deficiency) Epilepsy Dashboard — seed data module.

ACOX1 encodes Acyl-CoA Oxidase 1 (Straight-Chain Acyl-CoA Oxidase / SCOX), a 660 aa
peroxisomal matrix enzyme (PTS1: SRL) that functions as a homotrimer. It catalyses the
first (rate-limiting) step of peroxisomal straight-chain VLCFA beta-oxidation:
  Fatty acyl-CoA → 2-trans-enoyl-CoA + H₂O₂
Loss-of-function → VLCFA (C24:0, C26:0, C26:1) accumulate in plasma, brain, adrenal
cortex, and testes.

CRITICAL BIOCHEMICAL DISTINCTIONS:
  ACOX1 vs ZSD (PEX1/PEX6):
    VLCFA ELEVATED in BOTH → plasma VLCFA panel alone CANNOT distinguish.
    Plasmalogens (RBC): NORMAL in ACOX1 (peroxisome biogenesis intact) vs SEVERELY LOW in ZSD.
    Peroxisome number NORMAL or INCREASED in ACOX1 vs ABSENT/REDUCED in ZSD.
    → Plasmalogens MANDATORY simultaneously with VLCFA when VLCFA is elevated.
  ACOX1 vs ABCD1 (X-ALD):
    ACOX1 AR (both sexes equal) vs ABCD1 X-linked (males primarily affected).
    No adrenal insufficiency in ACOX1 (contrast: adrenal insufficiency UNIVERSAL in ABCD1).
    PHT/CBZ/OXC CAN be used in ACOX1 vs ABSOLUTE CI in ABCD1 (cortisol degradation → adrenal crisis).
    No inflammatory demyelination (CCALD pattern) in ACOX1.
  ACOX1 vs HSD17B4 (DBP deficiency):
    HSD17B4 blocks BOTH straight-chain AND branched-chain oxidation → pristanic SEVERELY ELEVATED.
    In ACOX1 pristanic NORMAL (pristanic beta-oxidation via ACOX2, not ACOX1).
  ACOX1 vs PHYH (Adult Refsum):
    Phytanic acid NORMAL in ACOX1 (PHYH / alpha-oxidation unaffected).
    Phytanic ELEVATED in Refsum; VLCFA NORMAL in Refsum.

ACOX1 PROTEIN BIOLOGY:
  660 aa, peroxisomal matrix enzyme, PTS1 tripeptide SRL, functional homotrimer.
  Catalyses Step 1 (rate-limiting) of straight-chain VLCFA peroxisomal beta-oxidation.
  ACOX2 handles branched-chain (pristanic) beta-oxidation → ACOX1 deficiency does NOT affect pristanic.
  Deficiency → H₂O₂ is NOT generated from VLCFA → oxidative signalling disruption in peroxisome.
  VLCFA accumulate in CNS white matter (myelin), retinal pigment epithelium, adrenal cortex.

LOCUS: 17q25.1  |  OMIM GENE: *609751  |  OMIM DISEASE: #264490
PROTEIN: 660 aa, peroxisomal matrix (PTS1-SRL), homotrimer
EPIDEMIOLOGY: ~50 cases published worldwide (2026); may be significantly underdiagnosed.
  All variants private (no common founder mutation). AR biallelic LOF.
  Initially described by Poll-The et al. 1988 (Am J Hum Genet).
  Distinguished from ZSD only by plasmalogens — VLCFA elevated in both.

LORENZO OIL IN ACOX1:
  Lorenzo oil inhibits VLCFA elongase → reduces C24:0/C26:0 synthesis → lowers plasma VLCFA.
  Mechanism mismatch: ACOX1 deficiency = OXIDATION failure (cannot degrade VLCFA).
    Lorenzo oil reduces VLCFA synthesis but cannot restore ACOX1 beta-oxidation activity.
  In ABCD1: Lorenzo oil reduces VLCFA import load; partial biochemical benefit observed.
  In ACOX1: no proven clinical benefit; NOT recommended.

FASTING HAZARD:
  Peroxisomal beta-oxidation integral to fasting fatty acid metabolism.
  Fasting → VLCFA accumulation surge, risk of metabolic decompensation.
  IV dextrose MANDATORY perioperatively; avoid prolonged fasting.

HSCT NOTE:
  HSCT arrests INFLAMMATORY demyelination (Krabbe/GALC, CCALD/ABCD1).
  ACOX1 pathology = VLCFA-mediated neurodegeneration; NOT primarily inflammatory.
  Exception: Attenuated ACOX1 may have neuroinflammatory component — HSCT being studied.
  No HSCT for classic severe form (non-inflammatory VLCFA accumulation).

PHENOTYPIC SPECTRUM:
  Classic Severe (biallelic null, 55%): Neonatal hypotonia, seizures 90%, retinal degeneration
    (ERG abnormal 85%), SNHL 65%, leukodystrophy (posterior predominant WM), no adrenal insufficiency,
    progressive cognitive decline. Median survival <3 years. VLCFA severely elevated.
  Intermediate (null/hypomorphic, 30%): Onset 3-12 months. Seizures 60-70%. Moderate leukodystrophy.
    Some survival to teen years.
  Attenuated (hypomorphic/hypomorphic, 15%): Onset 1-10 years. Seizures 25-40%. May have autoimmune
    neuroinflammatory features. Leukodystrophy mild-moderate. Survival into adulthood possible.
"""

import random
random.seed(46)


# ── Overview (KPIs + summary) ─────────────────────────────────────────────────

def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 72,
        "classic_severe_pct": 55,
        "intermediate_pct": 30,
        "attenuated_pct": 15,
        "drug_resistance_pct": 35,
        "retinal_degeneration_pct": 85,
        "snhl_pct": 65,
        "leukodystrophy_pct": 100,
        "adrenal_insufficiency_pct": 0,
        "on_dha_pct": 45,
        "omim_gene": "609751",
        "omim_disease": "264490",
        "locus": "17q25.1",
        "vlcfa_elevated": True,
        "vlcfa_elevated_pct": 100,
        "phytanic_normal_pct": 100,
        "plasmalogen_normal_pct": 100,
        "inheritance": "Autosomal recessive (AR), biallelic LOF",
        "common_variant": "No founder mutation — all private variants; ~50 cases worldwide 2026",
        "disease_mechanism": (
            "ACOX1 (Acyl-CoA Oxidase 1) is a 660 aa peroxisomal matrix enzyme (PTS1: SRL, homotrimer) "
            "that catalyses the first rate-limiting step of straight-chain VLCFA peroxisomal "
            "beta-oxidation: fatty acyl-CoA → 2-trans-enoyl-CoA + H₂O₂. ACOX1 LOF → VLCFA "
            "(C26:0, C24:0, C26:1) accumulate in plasma, brain white matter, retinal pigment "
            "epithelium, and adrenal cortex. Plasma VLCFA is identical to ZSD on screening — "
            "RBC plasmalogens MANDATORY to distinguish (NORMAL in ACOX1 vs SEVERELY LOW in ZSD). "
            "Phytanic NORMAL (PHYH/alpha-oxidation unaffected). Pristanic NORMAL (ACOX2-mediated). "
            "No adrenal insufficiency (contrast ABCD1). PHT/CBZ/OXC can be used."
        ),
        "nbs_positive_rate": (
            "Not in standard NBS; plasma VLCFA on DBS advocated; plasmalogens also needed on DBS. "
            "VLCFA elevation on NBS cannot distinguish ACOX1 from ZSD without plasmalogens. "
            "~50 cases worldwide; likely underdiagnosed — often missed or misclassified as ZSD."
        ),
        "key_concepts": [
            "ACOX1 = Acyl-CoA Oxidase 1 (660 aa, PTS1-SRL, homotrimer, peroxisomal matrix): catalyses first rate-limiting step of straight-chain VLCFA beta-oxidation",
            "ACOX1 deficiency ~50 cases worldwide 2026; AR biallelic LOF; all variants private (no founder allele); likely underdiagnosed",
            "ACOX1 LOF → VLCFA (C26:0, C24:0) severely elevated → same plasma VLCFA as ZSD; RBC plasmalogens MANDATORY to distinguish",
            "Plasmalogens (RBC) NORMAL — peroxisome biogenesis INTACT; KEY DISTINCTION from ZSD (PEX1/PEX6) where plasmalogens SEVERELY LOW",
            "Phytanic acid NORMAL — PHYH (alpha-oxidation) intact and imported via PTS2; ACOX1 = straight-chain oxidase only",
            "Pristanic acid NORMAL — pristanic beta-oxidation via ACOX2 (not ACOX1); KEY DISTINCTION from HSD17B4/DBP where pristanic SEVERELY ELEVATED",
            "No adrenal insufficiency — ACOX1 does not cause clinically significant adrenal crisis (contrast ABCD1 where adrenal insufficiency is universal)",
            "Retinal degeneration (ERG abnormal 85%) — VLCFA accumulate in retinal pigment epithelium; VGB adds retinal toxicity risk (additive)",
            "Leukodystrophy — posterior-predominant WM signal abnormality; unlike ZSD (cortical malformation pattern); Loes score tracks progression",
            "VGB: RELATIVE CI (retinal degeneration + VGB peripheral VF constriction = additive); NOT absolute CI — prefer ACTH for IS",
            "VPA: RELATIVE CI — hepatotoxicity; POLG1 MANDATORY (CPIC A); three mechanisms: carnitine depletion, peroxisomal beta-oxidation interference, mitochondrial toxicity",
            "PHT/CBZ/OXC: CAN BE USED — no adrenal mechanism; CRITICAL contrast to ABCD1 where PHT/CBZ = ABSOLUTE CI (cortisol degradation → adrenal crisis)",
            "Lorenzo oil NOT recommended in ACOX1 — inhibits VLCFA elongase (reduces synthesis) but cannot restore beta-oxidation; mechanism mismatch",
            "No ERT (2026) — ACOX1 is peroxisomal matrix enzyme; systemic ERT cannot cross peroxisomal membrane as functional enzyme",
            "No HSCT for classic severe form — VLCFA neurodegeneration not inflammatory; HSCT arrests inflammatory demyelination only (Krabbe, CCALD)",
        ],
        "standards": [
            "Poll-The BT, Roels F, Ogier H, et al. A new peroxisomal disorder with enlarged peroxisomes and a specific deficiency of acyl-CoA oxidase. Am J Hum Genet. 1988;42(3):422–434.",
            "Ferdinandusse S, Denis S, Hogenhout EM, et al. Clinical, biochemical, and mutational spectrum of peroxisomal acyl-coenzyme A oxidase deficiency. Hum Mutat. 2007;28(9):904–912.",
            "Watkins PA, Chen WW, Harris CJ, et al. Peroxisomal bifunctional enzyme deficiency. J Clin Invest. 1989;83(3):771–777.",
            "Waterham HR, Ferdinandusse S, Wanders RJA. Human disorders of peroxisome metabolism. Biochim Biophys Acta. 2016;1863(5):922–933.",
            "Wanders RJA, Waterham HR. Peroxisomal disorders I: biochemistry and genetics of peroxisome biogenesis disorders. Clin Genet. 2005;67(2):107–133.",
            "CPIC Guideline — Valproic Acid and POLG1 (Grade A). cpicpgx.org. 2023.",
        ],
    }


# ── Breakdown (patients + seizures + treatments) ─────────────────────────────

def get_breakdown():
    etiologies = [
        {
            "name": "Classic Severe ACOX1 (Biallelic Null / Severe LOF)",
            "pct": 55,
            "n": 22,
            "sex": "M/F equal (AR)",
            "onset_age": "Neonatal (0–1 month)",
            "seizure_risk": "90%",
            "eeg": "Multifocal clonic discharges, hypsarrhythmia (IS), burst-suppression in neonates",
            "mri": "Severe posterior-predominant leukodystrophy, diffuse WM T2 hyperintensity, cortex relatively spared initially; Loes score high",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic null (frameshift/nonsense) or null + splice LOF → complete ACOX1 enzyme "
                "absence → no VLCFA beta-oxidation → C26:0 severely elevated (>1.5 µmol/L; normal "
                "<0.3). Plasmalogens NORMAL (peroxisome biogenesis intact). Phytanic NORMAL. "
                "Retinal degeneration ERG abnormal 85%. SNHL 65%. No adrenal insufficiency. "
                "All variants private. Median survival <3 years. VLCFA identical to ZSD on plasma "
                "panel — plasmalogens MANDATORY to distinguish."
            ),
        },
        {
            "name": "Intermediate ACOX1 (Null + Hypomorphic or Splice + Missense)",
            "pct": 30,
            "n": 12,
            "sex": "M/F equal (AR)",
            "onset_age": "Infantile (3–12 months)",
            "seizure_risk": "60–70%",
            "eeg": "Modified hypsarrhythmia, focal spikes, multifocal discharges, diffuse slowing",
            "mri": "Moderate posterior-predominant leukodystrophy; WM T2 changes; partial myelination",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "One null + one hypomorphic missense → partial ACOX1 enzyme activity. Residual "
                "VLCFA oxidation reduces severity. C26:0 elevated but less than classic severe. "
                "Moderate leukodystrophy. SNHL 50-60%. Retinal degeneration moderate (ERG abnormal). "
                "Seizures 60-70%; drug resistance ~25%. Some survival to teen years. "
                "Plasmalogens NORMAL; phytanic NORMAL."
            ),
        },
        {
            "name": "Attenuated ACOX1 (Biallelic Hypomorphic Missense)",
            "pct": 15,
            "n": 6,
            "sex": "M/F equal (AR)",
            "onset_age": "Childhood (1–10 years)",
            "seizure_risk": "25–40%",
            "eeg": "Focal discharges, sparse multifocal spikes, may show theta/delta slowing",
            "mri": "Mild-moderate leukodystrophy; predominantly posterior; some inflammatory pattern possible",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic hypomorphic missense → residual ACOX1 enzyme activity (~15-40% of normal). "
                "C26:0 mildly-moderately elevated. Mild-moderate leukodystrophy. May have "
                "autoimmune/neuroinflammatory features (HSCT being studied experimentally). "
                "Seizures 25-40%. Survival into adulthood possible. Plasmalogens NORMAL; phytanic NORMAL. "
                "Variant: e.g. p.Arg372Gln/missense compound heterozygous."
            ),
        },
    ]

    phenotypes = (
        [("Classic Severe ACOX1", "Severe")] * 22
        + [("Intermediate ACOX1", "Intermediate")] * 12
        + [("Attenuated ACOX1", "Attenuated")] * 6
    )
    sexes = ["M"] * 20 + ["F"] * 20
    random.shuffle(sexes)

    genotype_map = {
        "Classic Severe ACOX1": [
            "Biallelic null (frameshift/nonsense)",
            "Null + splice LOF",
            "Homozygous large deletion",
        ],
        "Intermediate ACOX1": [
            "Null + hypomorphic missense",
            "Splice + missense",
            "Compound heterozygous missense",
        ],
        "Attenuated ACOX1": [
            "Biallelic hypomorphic missense",
            "Compound hypomorphic",
            "Missense p.Arg372Gln/missense",
        ],
    }

    seizure_map = {
        "Severe": ["Neonatal multifocal clonic", "Infantile spasms", "Status epilepticus"],
        "Intermediate": ["Infantile spasms", "Focal with bilateral spread", "Myoclonic"],
        "Attenuated": ["Focal with bilateral spread", "Myoclonic", None],
    }

    trigger_pool = [
        "Febrile illness",
        "Intercurrent infection",
        "Sleep deprivation",
        "Missed AED",
        "Metabolic decompensation",
        "Stress",
        "Anaesthesia",
    ]

    treatment_pool = [
        "LEV (first-line)",
        "Phenobarbital (neonatal — Level B)",
        "ACTH (IS — Level B, preferred over VGB)",
        "OXC/CBZ (focal — Level B; CAN USE — no adrenal mechanism)",
        "DHA supplementation (Level C)",
        "CLB (adjunct — Level C)",
    ]

    patients = []
    for i, ((pheno, severity), sex) in enumerate(zip(phenotypes, sexes)):
        pid = f"ACOX1-{i+1:02d}"

        if severity == "Severe":
            age_mo = random.randint(0, 1)
            has_sz = random.random() < 0.90
            vlcfa_c26 = round(random.uniform(0.8, 3.5), 2)
            leuko_grade = random.choice(["Severe", "Severe", "Moderate"])
        elif severity == "Intermediate":
            age_mo = random.randint(3, 12)
            has_sz = random.random() < 0.65
            vlcfa_c26 = round(random.uniform(0.5, 1.8), 2)
            leuko_grade = random.choice(["Moderate", "Moderate", "Severe"])
        else:
            age_mo = random.randint(12, 120)
            has_sz = random.random() < 0.32
            vlcfa_c26 = round(random.uniform(0.3, 0.9), 2)
            leuko_grade = random.choice(["Mild", "Moderate"])

        seizure_type = random.choice(seizure_map[severity]) if has_sz else None
        drug_resistant = has_sz and random.random() < 0.35
        triggers = random.sample(trigger_pool, k=random.randint(2, 4)) if has_sz else []
        current_treatment = random.sample(treatment_pool, k=random.randint(2, 3))
        genotype = random.choice(genotype_map[pheno])

        patients.append({
            "id": pid,
            "phenotype": pheno,
            "severity": severity,
            "sex": sex,
            "onset_age_months": age_mo,
            "has_seizures": has_sz,
            "seizure_type": seizure_type,
            "drug_resistant": drug_resistant,
            "retinal_degeneration": random.random() < (0.85 if severity == "Severe" else 0.60 if severity == "Intermediate" else 0.30),
            "snhl": random.random() < (0.65 if severity == "Severe" else 0.50 if severity == "Intermediate" else 0.20),
            "leukodystrophy_grade": leuko_grade,
            "vlcfa_c26_umol_l": vlcfa_c26,
            "phytanic_normal": True,
            "plasmalogen_normal": True,
            "genotype": genotype,
            "current_treatment": current_treatment,
            "triggers": triggers,
            "polg1_tested": True,
        })

    seizure_types = [
        {
            "type": "Neonatal multifocal clonic",
            "pct": 38,
            "preferred_tx": "LEV + phenobarbital (neonatal)",
            "notes": "Onset first days-weeks of life; multifocal or migratory clonic; EEG burst-suppression or multifocal spikes; IV LEV + phenobarbital standard.",
        },
        {
            "type": "Infantile spasms",
            "pct": 25,
            "preferred_tx": "ACTH Level B (VGB RELATIVE CI — leukodystrophy/retinal risk)",
            "notes": "Hypsarrhythmia on EEG; ACTH preferred over VGB — retinal degeneration (ERG abnormal 85%) + VGB retinal toxicity = additive; monthly VEP/VF if VGB used.",
        },
        {
            "type": "Focal with bilateral spread",
            "pct": 20,
            "preferred_tx": "LEV, OXC (CAN USE — no adrenal mechanism)",
            "notes": "Temporal or occipital onset; PHT/CBZ/OXC can be used (no adrenal insufficiency unlike ABCD1 where enzyme inducers = ABSOLUTE CI).",
        },
        {
            "type": "Myoclonic",
            "pct": 12,
            "preferred_tx": "LEV, CLB",
            "notes": "Myoclonic jerks with or without EEG correlate; LEV first-line; CLB add-on (Level C); avoid VPA (RELATIVE CI — hepatotoxicity, POLG1 mandatory).",
        },
        {
            "type": "Status epilepticus",
            "pct": 5,
            "preferred_tx": "IV LEV, benzodiazepines",
            "notes": "Emergency management: IV lorazepam/diazepam → IV LEV; fosphenytoin acceptable (no adrenal mechanism); IV dextrose mandatory (fasting hazard in ACOX1).",
        },
    ]

    triggers = [
        {"trigger": "Febrile illness", "pct": 62},
        {"trigger": "Intercurrent infection", "pct": 50},
        {"trigger": "Sleep deprivation", "pct": 38},
        {"trigger": "Missed AED", "pct": 30},
        {"trigger": "Metabolic decompensation", "pct": 28},
        {"trigger": "Stress", "pct": 22},
        {"trigger": "Anaesthesia (IV dextrose mandatory — fasting hazard)", "pct": 15},
    ]

    monitoring = [
        {
            "parameter": "Plasma VLCFA (C26:0 and C26:0/C22:0 ratio)",
            "threshold": "C26:0 >1.5 µmol/L (normal <0.3); C26:0/C22:0 ratio >0.05",
            "frequency": "Every 6 months",
        },
        {
            "parameter": "DHA level",
            "threshold": "Target >75 µg/mL; low → supplement 200 mg/day (infants)",
            "frequency": "Every 3 months",
        },
        {
            "parameter": "Liver function tests (LFTs)",
            "threshold": "Normal; q3 months if VPA used (RELATIVE CI — hepatotoxicity 3 mechanisms)",
            "frequency": "q3 months on VPA; annually otherwise",
        },
        {
            "parameter": "Visual acuity + ERG (electroretinogram)",
            "threshold": "ERG abnormal 85% at diagnosis; monitor progression; VEP + VF mandatory if VGB used",
            "frequency": "Every 6 months",
        },
        {
            "parameter": "Audiogram (SNHL)",
            "threshold": "SNHL in 65%; early identification for hearing aids/cochlear implant planning",
            "frequency": "Annually",
        },
        {
            "parameter": "Brain MRI (Loes score — leukodystrophy progression)",
            "threshold": "Posterior-predominant WM T2 signal; track Loes score each scan",
            "frequency": "Every 12 months",
        },
        {
            "parameter": "Adrenal function (ACTH-stimulation test)",
            "threshold": "Clinically significant adrenal insufficiency rare in ACOX1 (contrast ABCD1 mandatory); baseline warranted",
            "frequency": "Annually",
        },
    ]

    lifecycle = [
        {
            "stage": "Neonatal (0–1 mo)",
            "features": "Hypotonia, feeding difficulties, seizures begin (multifocal clonic); VLCFA severely elevated; plasmalogens NORMAL; ERG ordered; IV dextrose if fasting risk.",
        },
        {
            "stage": "Infantile (1–12 mo)",
            "features": "Leukodystrophy progression on MRI; infantile spasms onset; ACTH Level B (preferred over VGB — retinal degeneration additive); retinal dysfunction begins (ERG abnormal); DHA supplementation started.",
        },
        {
            "stage": "Early childhood (1–5 y)",
            "features": "Progressive posterior-predominant leukodystrophy; SNHL worsens; visual decline; drug-resistant seizures in 35%; hearing aids; cochlear implant evaluation; physiotherapy.",
        },
        {
            "stage": "Late childhood (5–12 y)",
            "features": "Severe cognitive decline in classic form; spastic quadriplegia; seizure burden often lower (controlled or burned out); Loes score high in severe form; attenuated form may remain ambulatory.",
        },
        {
            "stage": "Adolescent (12–18 y)",
            "features": "Complete neurological disability in classic severe form; attenuated form may be ambulatory with support; leukodystrophy plateau or slow progression; seizure management ongoing.",
        },
        {
            "stage": "Adult (>18 y)",
            "features": "Rare in classic severe form (median survival <3 y); intermediate survival to teen years in some; attenuated form may survive into adulthood with moderate disability; DHA + seizure management continued.",
        },
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "SV2A modulator",
            "level": "First-line (Level A — all forms)",
            "dose": "20–60 mg/kg/day in 2 divided doses; IV for SE",
            "notes": (
                "No peroxisomal interactions; no adrenal effect; no VLCFA impact. "
                "Safe across all ACOX1 phenotypes. Preferred across neonatal, infantile, and childhood onset. "
                "IV formulation suitable for SE and neonatal emergencies."
            ),
            "ci": "None specific to ACOX1",
        },
        {
            "drug": "Phenobarbital",
            "class": "Barbiturate — GABA-A potentiator",
            "level": "Level B (neonatal seizures)",
            "dose": "Loading 20 mg/kg IV; maintenance 3–5 mg/kg/day",
            "notes": (
                "Standard neonatal seizure management alongside LEV. "
                "Respiratory monitoring required in neonates. "
                "No peroxisomal interactions. No adrenal mechanism in ACOX1."
            ),
            "ci": "Respiratory depression; use with monitoring in neonatal ICU",
        },
        {
            "drug": "ACTH / Corticotropin",
            "class": "Infantile spasms — hormone therapy",
            "level": "Level B (IS — preferred over VGB)",
            "dose": "150 IU/m²/day IM × 2 weeks then taper",
            "notes": (
                "Preferred over VGB for IS in ACOX1: retinal degeneration (ERG 85%) + VGB retinal "
                "toxicity = additive risk → ACTH avoids this. Adrenal axis intact in ACOX1 (no "
                "adrenal insufficiency); ACTH does not carry ABCD1-class adrenal risk. "
                "POLG1 status must be confirmed before any VPA co-prescription."
            ),
            "ci": "Monitor glycaemia, BP, infection; immunosuppression risk",
        },
        {
            "drug": "OXC / CBZ (Oxcarbazepine / Carbamazepine)",
            "class": "Sodium channel blocker",
            "level": "Level B (focal seizures — CAN USE)",
            "dose": "OXC 10–40 mg/kg/day; CBZ 5–20 mg/kg/day (standard weight-based titration)",
            "notes": (
                "CAN BE USED in ACOX1 — no adrenal mechanism (contrast ABCD1/X-ALD: PHT/CBZ = "
                "ABSOLUTE CI because CYP3A4-mediated cortisol degradation → adrenal crisis). "
                "ACOX1 has no adrenal insufficiency → enzyme inducers safe. Standard hepatic monitoring."
            ),
            "ci": "None specific to ACOX1; standard SJS/TEN monitoring; hyponatraemia with OXC",
        },
        {
            "drug": "DHA (Docosahexaenoic acid)",
            "class": "Omega-3 supplementation",
            "level": "Level C",
            "dose": "200 mg/day infants; 500 mg/day children",
            "notes": (
                "Reduces secondary DHA-plasmalogen deficiency caused by VLCFA accumulation. "
                "Safe, no drug interactions. Used across all ACOX1 phenotypes for neuronal membrane support."
            ),
            "ci": "None",
        },
        {
            "drug": "Clobazam (CLB)",
            "class": "Benzodiazepine — 1,5-BZD",
            "level": "Level C adjunct",
            "dose": "0.1–0.5 mg/kg/day in 2 divided doses",
            "notes": (
                "Add-on for focal and myoclonic seizures. No peroxisomal interactions. "
                "Sedation risk in severe forms with leukodystrophy. Tolerance may develop."
            ),
            "ci": "None specific to ACOX1",
        },
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "level": "RELATIVE CI — hepatotoxicity + POLG1 MANDATORY (CPIC Grade A)",
            "reason": (
                "THREE hepatotoxicity mechanisms: (a) carnitine depletion (inhibits carnitine transport), "
                "(b) peroxisomal beta-oxidation interference (direct inhibition of ACOX1 substrate processing), "
                "(c) mitochondrial toxicity (POLG1 mandatory CPIC A). Mitochondrial co-pathology possible "
                "in ACOX1. LFT monitoring q3 months if used. POLG1 test MANDATORY before VPA — never skip."
            ),
            "alternative": "LEV (first-line) · OXC/CBZ (focal) · CLB (adjunct)",
        },
        {
            "drug": "Vigabatrin (VGB)",
            "level": "RELATIVE CI — retinal toxicity additive (HIGH RISK)",
            "reason": (
                "Retinal degeneration already present in ACOX1 (ERG abnormal 85%); VLCFA accumulate in "
                "retinal pigment epithelium. VGB causes irreversible peripheral visual field constriction. "
                "Additive retinal risk — NOT absolute CI (no universal severe retinopathy as in ZSD), "
                "but HIGH RISK. Monthly VEP/VF mandatory if VGB used. Prefer ACTH for IS."
            ),
            "alternative": "ACTH (Level B for IS) · LEV (focal/myoclonic)",
        },
        {
            "drug": "PHT/CBZ (phenytoin/carbamazepine) — routine use clarification",
            "level": "CAN USE — no adrenal mechanism (CRITICAL CONTRAST to ABCD1)",
            "reason": (
                "PHT/CBZ carry NO adrenal-insufficiency risk in ACOX1 (no adrenal crisis mechanism). "
                "CRITICAL CONTRAST: In ABCD1/X-ALD, PHT/CBZ = ABSOLUTE CI because CYP3A4 enzyme "
                "inducers accelerate cortisol metabolism → adrenal crisis. ACOX1 has no adrenal "
                "insufficiency → PHT/CBZ safe. Standard hepatic and cardiac monitoring applies."
            ),
            "alternative": "LEV preferred first-line; PHT/CBZ acceptable when clinically indicated",
        },
        {
            "drug": "Lorenzo oil",
            "level": "NOT RECOMMENDED — mechanism mismatch",
            "reason": (
                "Lorenzo oil inhibits VLCFA elongase (reduces C24:0/C26:0 synthesis) → may partially "
                "lower plasma VLCFA but does NOT restore ACOX1 beta-oxidation (the actual defect). "
                "No proven clinical benefit in ACOX1 deficiency. Mechanism mismatch: ACOX1 = oxidation "
                "failure (cannot degrade VLCFA); Lorenzo oil reduces synthesis but degradation remains "
                "blocked. Contrast ABCD1 where Lorenzo oil reduces VLCFA import and has partial biochemical "
                "response (though limited clinical benefit)."
            ),
            "alternative": "DHA supplementation (Level C) · seizure management with LEV",
        },
        {
            "drug": "Fasting / prolonged caloric restriction",
            "level": "HAZARD — metabolic decompensation risk",
            "reason": (
                "Peroxisomal beta-oxidation integral to fasting fatty acid metabolism. Fasting → VLCFA "
                "accumulation surge and risk of acute metabolic decompensation. IV dextrose MANDATORY "
                "perioperatively. Avoid prolonged fasting. Preoperative glucose infusion protocol required. "
                "Contrast with RCDP (fasting less critical) — VLCFA accumulation active hazard in ACOX1."
            ),
            "alternative": "IV 10% dextrose perioperatively; nasogastric feeds if oral feeding not possible",
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
            "ACOX1 = Acyl-CoA Oxidase 1 (660 aa, PTS1-SRL, homotrimer): catalyzes first rate-limiting step of peroxisomal straight-chain VLCFA beta-oxidation (fatty acyl-CoA → 2-trans-enoyl-CoA)",
            "ACOX1 deficiency → VLCFA accumulate (C26:0 severely elevated) → identical plasma VLCFA as ZSD on screening; distinguished by plasmalogens NORMAL",
            "Plasmalogens (RBC) NORMAL — peroxisome biogenesis INTACT in ACOX1; KEY DISTINCTION from ZSD/PEX1/PEX6 where plasmalogens severely low",
            "Phytanic acid NORMAL — PHYH (alpha-oxidation) intact and imported; ACOX1 = straight-chain oxidase only",
            "Pristanic acid NORMAL — pristanic beta-oxidation via ACOX2 (not ACOX1); KEY DISTINCTION from HSD17B4/DBP where pristanic ELEVATED",
            "No adrenal insufficiency — adrenal cortex affected by VLCFA accumulation in ACOX1, but NOT clinically significant adrenal crisis (contrast: ABCD1 = adrenal insufficiency universal; PHT/CBZ = ABSOLUTE CI in ABCD1)",
            "Retinal degeneration (ERG abnormal 85%) — VLCFA accumulate in retinal pigment epithelium; VGB adds risk (retinal toxicity additive); prefer ACTH over VGB for IS",
            "Leukodystrophy — diffuse WM signal abnormality; unlike ZSD (megalencephalic/cortical malformation pattern), ACOX1 leukodystrophy is posterior predominant",
            "VGB: RELATIVE CI (retinal degeneration + VGB peripheral VF loss = additive); not ABSOLUTE CI unlike ZSD. ACTH preferred for IS.",
            "VPA: RELATIVE CI — hepatotoxicity; POLG1 MANDATORY (CPIC A). Three mechanisms: carnitine depletion, beta-oxidation interference (peroxisomal), mitochondrial toxicity.",
            "PHT/CBZ/OXC: CAN BE USED — no adrenal mechanism; critical distinction from ABCD1 (PHT/CBZ = ABSOLUTE CI in ABCD1 → cortisol degradation → adrenal crisis)",
            "Lorenzo oil NOT recommended in ACOX1 — Lorenzo oil reduces VLCFA synthesis (inhibits elongase) but does not restore beta-oxidation; mechanism mismatch with ACOX1 (oxidation failure vs import failure in ABCD1)",
            "DHA supplementation Level C — reduces secondary DHA-plasmalogen deficiency; safe, no drug interactions; 200 mg/day infants",
            "No ERT (2026) — ACOX1 is peroxisomal matrix enzyme; systemic enzyme replacement cannot cross peroxisomal membrane as functional enzyme",
            "No HSCT — ACOX1 pathology is VLCFA-mediated neurodegeneration; HSCT arrests INFLAMMATORY demyelination (Krabbe, CCALD) — not relevant to non-inflammatory VLCFA accumulation (except attenuated form which may have inflammatory component — HSCT being explored experimentally)",
        ],
        "diagnostic_algorithm": [
            "Newborn with hypotonia, seizures, dysmorphic features → order plasma VLCFA panel (C26:0, C24:0, C26:0/C22:0 ratio, C24:0/C22:0 ratio) — elevated VLCFA points to peroxisomal disease.",
            "If VLCFA elevated → order RBC plasmalogens (C16:0-DMA, C18:0-DMA) SIMULTANEOUSLY — do not wait.",
            "Plasmalogens NORMAL → ACOX1 deficiency OR single enzyme peroxisomal beta-oxidation defect (not ZSD/PBD); proceed to ACOX1 workup.",
            "Plasmalogens SEVERELY LOW → ZSD (PEX1/PEX6 most common); proceed to PEX gene panel; do not assume ACOX1.",
            "Phytanic acid (to distinguish PHYH/Refsum where phytanic ELEVATED); Pristanic acid (to distinguish HSD17B4/DBP — elevated in DBP).",
            "Pristanic acid NORMAL → favors ACOX1 (not HSD17B4 where pristanic severely elevated); phytanic NORMAL → not Refsum.",
            "Brain MRI: leukodystrophy pattern (posterior predominant WM signal abnormality, cortex relatively spared initially); Loes score baseline.",
            "ERG (electroretinogram): abnormal in 85% of ACOX1 — retinal degeneration early feature; order at diagnosis; critical before considering VGB.",
            "Audiogram: SNHL in 65% — order at diagnosis; early fitting of hearing aids if abnormal.",
            "Adrenal function: ACTH stimulation test (unlike ABCD1, clinically significant adrenal insufficiency rare in ACOX1, but baseline testing warranted).",
            "Fibroblast ACOX1 enzyme assay (gold standard): severely reduced activity confirms diagnosis; available at specialist peroxisomal biochemistry labs.",
            "ACOX1 gene sequencing: confirms biallelic LOF; guides family counseling; variant database sparse (all private); POLG1 MANDATORY before any VPA consideration.",
        ],
        "pharmacological_distinctions": [
            "VLCFA panel identical to ZSD on plasma screening → must always check plasmalogens simultaneously to distinguish ACOX1 (normal plasmalogens) from ZSD (severely low); VLCFA alone insufficient.",
            "VGB: RELATIVE CI in ACOX1 (retinal degeneration + VGB VF loss = additive); contrast with ZSD where VGB is HIGH RISK and some experts use ABSOLUTE CI (universal severe retinopathy).",
            "VGB in ACOX1 vs RCDP: RCDP has cataracts (different structure); ACOX1 has retinal degeneration (ERG abnormal). VGB risk mechanism differs but both are high risk due to additive visual impairment.",
            "PHT/CBZ/OXC = CAN USE in ACOX1 (no adrenal insufficiency). CRITICAL CONTRAST: in ABCD1/X-ALD, PHT/CBZ = ABSOLUTE CI because enzyme inducers accelerate cortisol metabolism → adrenal crisis.",
            "VPA: RELATIVE CI in ACOX1; THREE mechanisms: (a) carnitine depletion (inhibits carnitine transport), (b) peroxisomal beta-oxidation interference (direct inhibition of ACOX1 substrate processing), (c) mitochondrial toxicity (POLG1 mandatory CPIC A). LFT q3 months.",
            "Lorenzo oil: NOT RECOMMENDED in ACOX1 — reduces C26:0 synthesis via elongase inhibition but cannot restore C26:0 oxidation (the actual defect). Contrast: ABCD1 where Lorenzo oil normalizes plasma VLCFA (reduces import load; clinical benefit limited but biochemical response seen).",
            "DHA: Level C (200–500 mg/day); reduces secondary DHA deficit; safe; no drug interactions.",
            "ACTH for IS: Level B; preferred over VGB (avoids additive retinal risk). Same preference as PEX7/RCDP1.",
            "Fasting: HAZARD — peroxisomal beta-oxidation essential for fasting fatty acid metabolism; fasting triggers VLCFA surge and metabolic decompensation. IV glucose perioperatively mandatory.",
            "No ERT (2026) — ACOX1 is peroxisomal matrix enzyme; no secreted form; systemic delivery cannot cross peroxisomal membrane.",
            "No HSCT for classic severe form — VLCFA accumulation is not inflammatory; HSCT is effective only for inflammatory demyelination (Krabbe GALC, CCALD due to ABCD1). Exception: attenuated ACOX1 may have neuroinflammatory component (being studied).",
            "POLG1 test MANDATORY before VPA (CPIC Grade A) — mitochondrial co-disease risk; never skip regardless of ACOX1 diagnosis.",
        ],
        "differential_diagnosis": [
            {
                "condition": "ZSD-Severe (PEX1/PEX6)",
                "distinction": "VLCFA elevated in BOTH; plasmalogens SEVERELY LOW in ZSD (NORMAL in ACOX1); ZSD shows dysmorphic features, cortical malformations (pachygyria); PEX gene panel confirms. Most important differential.",
            },
            {
                "condition": "ABCD1 (X-ALD)",
                "distinction": "VLCFA elevated in BOTH; X-linked (only males affected clinically); adrenal insufficiency mandatory; PHT/CBZ ABSOLUTE CI in ABCD1; Lorenzo oil partially effective in ABCD1; ACOX1 AR + no adrenal + PHT/CBZ safe.",
            },
            {
                "condition": "HSD17B4 (DBP deficiency — OMIM #261515)",
                "distinction": "VLCFA elevated in BOTH; pristanic acid SEVERELY ELEVATED in HSD17B4 (NORMAL in ACOX1); both are AR; DBP deficiency affects both enoyl-CoA hydratase AND 3-hydroxyacyl-CoA dehydrogenase; gene sequencing confirms.",
            },
            {
                "condition": "PHYH deficiency (Adult Refsum Disease)",
                "distinction": "Phytanic acid ELEVATED in Refsum (NORMAL in ACOX1); VLCFA normal in Refsum; onset adolescent-adult (vs neonatal in ACOX1); RP + polyneuropathy + anosmia hallmarks; no leukodystrophy.",
            },
            {
                "condition": "AMACR deficiency",
                "distinction": "Pristanic elevated, VLCFA mildly elevated, bile acid intermediates elevated; phytanic mildly elevated; adult-onset; AR; gene sequencing distinguishes from ACOX1 (pristanic NORMAL in ACOX1).",
            },
            {
                "condition": "NPC1/NPC2 (Niemann-Pick C)",
                "distinction": "Cholesterol storage disorder; VLCFA NORMAL; vertical supranuclear gaze palsy pathognomonic; filipin staining of fibroblasts positive; NPC1/NPC2 gene panel; no plasmalogen or VLCFA abnormality.",
            },
            {
                "condition": "MLD (ARSA — Metachromatic Leukodystrophy)",
                "distinction": "Leukodystrophy superficially similar (WM signal); VLCFA NORMAL; arylsulfatase A enzyme deficiency; urine sulfatides elevated; AR ARSA; responds to HSCT (unlike ACOX1 which does not respond to HSCT for classic form).",
            },
        ],
        "standards": [
            "Poll-The BT, Roels F, Ogier H, et al. A new peroxisomal disorder with enlarged peroxisomes and a specific deficiency of acyl-CoA oxidase. Am J Hum Genet. 1988;42(3):422–434.",
            "Ferdinandusse S, Denis S, Hogenhout EM, et al. Clinical, biochemical, and mutational spectrum of peroxisomal acyl-coenzyme A oxidase deficiency. Hum Mutat. 2007;28(9):904–912.",
            "Watkins PA, Chen WW, Harris CJ, et al. Peroxisomal bifunctional enzyme deficiency. J Clin Invest. 1989;83(3):771–777.",
            "Waterham HR, Ferdinandusse S, Wanders RJA. Human disorders of peroxisome metabolism. Biochim Biophys Acta. 2016;1863(5):922–933.",
            "Wanders RJA, Waterham HR. Peroxisomal disorders I: biochemistry and genetics of peroxisome biogenesis disorders. Clin Genet. 2005;67(2):107–133.",
            "CPIC Guideline — Valproic Acid and POLG1 (Grade A). cpicpgx.org. 2023.",
        ],
    }
