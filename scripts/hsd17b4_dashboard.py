#!/usr/bin/env python3
"""HSD17B4 / D-Bifunctional Protein Deficiency (DBP Deficiency / MFP-2 Deficiency) Epilepsy Dashboard — seed data module.

HSD17B4 encodes the D-bifunctional protein (DBP), also called Multifunctional Protein 2 (MFP-2),
a 736 aa peroxisomal matrix enzyme (PTS1: SRL) that catalyses steps 2 AND 3 of peroxisomal
beta-oxidation for ALL substrates:
  Step 2: 2-enoyl-CoA hydratase (domain 1, N-terminal)
  Step 3: 3-hydroxyacyl-CoA dehydrogenase (domain 2, middle)
  Domain 3 (C-terminal): Sterol carrier protein 2-like (SCP2) — structural

Substrates handled by HSD17B4 (steps 2+3):
  • Straight-chain VLCFA (C24:0, C26:0) — after ACOX1 step 1
  • Branched-chain fatty acids (pristanic acid, 2-methylacyl-CoA) — after ACOX2 step 1
  • Bile acid intermediates (THCA/DHCA) — after ACOX2 step 1
  HSD17B4 deficiency → ALL three substrate classes accumulate.

CRITICAL BIOCHEMICAL DISTINCTIONS:
  HSD17B4 vs ACOX1 (Pseudo-NALD):
    PRISTANIC ACID: SEVERELY ELEVATED in HSD17B4 (NORMAL in ACOX1) — KEY DISTINCTION.
    ACOX1 blocks only step 1 of VLCFA oxidation → pristanic NORMAL (ACOX2/step1 of pristanic intact,
    but steps 2-3 also affected in HSD17B4 → pristanic accumulates in HSD17B4 not ACOX1).
    Both have VLCFA elevated; both have plasmalogens NORMAL.
    → Pristanic acid measurement MANDATORY to distinguish HSD17B4 from ACOX1.
  HSD17B4 vs ZSD (PEX1/PEX6):
    Both have VLCFA elevated; ZSD has SEVERELY LOW plasmalogens (NORMAL/near-normal in HSD17B4).
    ZSD has absent/reduced peroxisomes (peroxisome biogenesis disorder); HSD17B4 = single enzyme deficiency.
    ZSD phytanic elevated; HSD17B4 pristanic is the primary elevated marker.
    → Plasmalogens distinguish (severely low in ZSD, normal in HSD17B4).
  HSD17B4 vs ABCD1 (X-ALD):
    X-linked in ABCD1 (males primarily); AR in HSD17B4 (both sexes).
    Adrenal insufficiency universal in ABCD1; ABSENT in HSD17B4.
    PHT/CBZ ABSOLUTE CI in ABCD1; CAN USE in HSD17B4 (no adrenal mechanism).
    ABCD1: VLCFA elevated, pristanic NORMAL; HSD17B4: BOTH VLCFA and pristanic elevated.
  HSD17B4 vs PHYH (Adult Refsum):
    Refsum: phytanic SEVERELY ELEVATED, VLCFA NORMAL, pristanic normal/mildly elevated.
    HSD17B4: VLCFA + pristanic + THCA ELEVATED; phytanic NORMAL or mildly elevated.
    Refsum: adult-onset; HSD17B4: neonatal/infantile onset.

HSD17B4 PROTEIN BIOLOGY:
  736 aa, peroxisomal matrix (PTS1: SRL, C-terminal), single polypeptide with 3 functional domains.
  Domain 1 (2-enoyl-CoA hydratase, aa 1-~291): converts 2-trans-enoyl-CoA → L-3-hydroxyacyl-CoA (step 2).
  Domain 2 (3-hydroxyacyl-CoA dehydrogenase, aa ~293-~574): converts L-3-hydroxyacyl-CoA → 3-ketoacyl-CoA (step 3).
  Domain 3 (SCP2-like, aa ~575-736): sterol carrier / lipid transfer function.
  Note: MFP-1 (EHHADH) handles the same reactions but for shorter and different chain-length substrates;
    MFP-2 (HSD17B4) is the dominant enzyme for VLCFA and branched-chain substrates.

DBP DEFICIENCY TYPES (based on residual domain activity):
  Type I (~50%): Both hydratase (D1) AND 3-hydroxyacyl-CoA dehydrogenase (D2) deficient.
    → ALL peroxisomal beta-oxidation substrates affected.
    → Most severe: neonatal, ZSD-like, VLCFA + pristanic + THCA all elevated.
  Type II (~30%): Hydratase (D1) deficient only; 3-hydroxyacyl-CoA dehydrogenase (D2) intact.
    → VLCFA + pristanic + THCA all still accumulate (hydratase required for all).
    → Clinically similar to Type I (severe). Both domain activities needed for any substrate.
  Type III (~20%): 3-hydroxyacyl-CoA dehydrogenase (D2) deficient; hydratase (D1) intact.
    → Primarily affects sex steroid biosynthesis in gonadal tissue.
    → Peroxisomal disease may be milder; VLCFA + pristanic still affected but degree variable.
    → Some type III patients survive longer; adult-onset in rare cases.

LOCUS: 5q23.1  |  OMIM GENE: *601860  |  OMIM DISEASE: #261515
PROTEIN: 736 aa, peroxisomal matrix (PTS1-SRL), 3-domain bifunctional
EPIDEMIOLOGY: ~100 cases published worldwide (2026); likely underdiagnosed.
  Initially described by Watkins PA et al. 1989 (J Clin Invest).
  Spectrum: neonatal severe (types I, II) → milder childhood (type III rare).
  Distinguishing from ACOX1 requires pristanic acid measurement (both have VLCFA elevated).

BILE ACID SYNTHESIS NOTE:
  THCA (trihydroxycholestanoic acid) and DHCA (dihydroxycholestanoic acid) require HSD17B4
  for peroxisomal shortening to CDCA (chenodeoxycholic acid) and CA (cholic acid).
  HSD17B4 deficiency → THCA/DHCA accumulate in plasma and urine.
  THCA/DHCA elevation = additional distinguishing biochemical marker vs ACOX1.

RETINAL/VISUAL INVOLVEMENT:
  Retinal dystrophy in 75% (photoreceptor loss + pigmentary changes).
  Optic atrophy in 60%.
  Combined effect: VGB is HIGH RISK (additive to existing visual impairment).
  Contrast: ACOX1 where VGB = RELATIVE CI (ERG 85%); HSD17B4 where additional optic atrophy
  raises the visual risk profile.

ERT/HSCT/GENE THERAPY:
  No ERT (2026) — HSD17B4 is peroxisomal matrix enzyme; systemic ERT cannot reach peroxisomal matrix.
  No HSCT — DBP deficiency is NOT primarily inflammatory demyelination (contrast Krabbe/GALC, ABCD1-CCALD).
  Gene therapy: experimental; no approved therapy 2026.

MALE GONADAL PHENOTYPE (Type III):
  HSD17B4 has 17β-hydroxysteroid dehydrogenase activity in gonads (different from peroxisomal role).
  Type III: 17β-HSD deficiency in testes → testosterone ↓, androstenedione ↑ in males.
  Some type III males may have sex reversal or ambiguous genitalia in addition to or instead of
  neurological disease. This gonadal phenotype distinguishes type III from I and II.
"""

import random
random.seed(47)


# ── Overview (KPIs + summary) ─────────────────────────────────────────────────

def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 82,
        "type_i_pct": 50,
        "type_ii_pct": 30,
        "type_iii_pct": 20,
        "drug_resistance_pct": 38,
        "retinal_dystrophy_pct": 75,
        "optic_atrophy_pct": 60,
        "snhl_pct": 85,
        "leukodystrophy_pct": 90,
        "adrenal_insufficiency_pct": 0,
        "on_dha_pct": 50,
        "omim_gene": "601860",
        "omim_disease": "261515",
        "locus": "5q23.1",
        "vlcfa_elevated": True,
        "vlcfa_elevated_pct": 100,
        "pristanic_elevated_pct": 100,
        "thca_elevated_pct": 85,
        "plasmalogen_normal_pct": 92,
        "inheritance": "Autosomal recessive (AR), biallelic LOF — both sexes equal",
        "common_variant": "No common founder mutation — all private variants; ~100 cases worldwide 2026; underdiagnosed",
        "disease_mechanism": (
            "HSD17B4 (D-bifunctional protein / MFP-2) is a 736 aa peroxisomal matrix enzyme (PTS1: SRL) "
            "with 3 domains: enoyl-CoA hydratase (D1), 3-hydroxyacyl-CoA dehydrogenase (D2), SCP2-like (D3). "
            "HSD17B4 catalyses steps 2+3 of peroxisomal beta-oxidation for ALL substrates: "
            "VLCFAs (C24:0, C26:0), branched-chain (pristanic acid, 2-methylacyl-CoA), and "
            "bile acid intermediates (THCA, DHCA). LOF → VLCFA + pristanic + THCA/DHCA all accumulate. "
            "KEY DISTINCTION from ACOX1: pristanic SEVERELY ELEVATED in HSD17B4 (NORMAL in ACOX1). "
            "Plasmalogens NORMAL (contrast ZSD where severely low). No adrenal insufficiency (contrast ABCD1). "
            "PHT/CBZ CAN BE USED. VGB HIGH RISK (retinal dystrophy + optic atrophy additive). VPA RELATIVE CI."
        ),
        "nbs_positive_rate": (
            "Not in standard NBS; plasma VLCFA + pristanic + THCA/DHCA on DBS advocated. "
            "VLCFA alone CANNOT distinguish HSD17B4 from ACOX1 — pristanic MANDATORY. "
            "Plasmalogens to distinguish from ZSD. ~100 cases worldwide; underdiagnosed."
        ),
        "key_concepts": [
            "HSD17B4 = D-bifunctional protein / MFP-2 (736 aa, PTS1-SRL): steps 2+3 of peroxisomal beta-oxidation for ALL substrates (VLCFA + pristanic + THCA/DHCA)",
            "HSD17B4 deficiency ~100 cases worldwide 2026; AR biallelic LOF; all private variants; likely underdiagnosed",
            "HSD17B4 LOF → VLCFA (C26:0) ELEVATED + pristanic SEVERELY ELEVATED + THCA/DHCA elevated → multi-substrate peroxisomal beta-oxidation failure",
            "PRISTANIC ACID SEVERELY ELEVATED — KEY DISTINCTION from ACOX1 (where pristanic NORMAL because ACOX2 step 1 of pristanic intact but steps 2-3 by HSD17B4 are deficient)",
            "Plasmalogens (RBC) NORMAL / near-normal — PTS2 pathway (PEX7/GNPAT/AGPS/FAR1) INTACT; KEY DISTINCTION from ZSD where plasmalogens SEVERELY LOW",
            "No adrenal insufficiency — HSD17B4 does NOT cause adrenal crisis (contrast ABCD1 where adrenal insufficiency universal and PHT/CBZ = ABSOLUTE CI)",
            "PHT/CBZ/OXC: CAN USE in HSD17B4 (no adrenal mechanism) — CRITICAL CONTRAST to ABCD1 (ABSOLUTE CI) and same as ACOX1",
            "VGB: HIGH RISK — retinal dystrophy (75%) + optic atrophy (60%) + VGB irreversible peripheral VF constriction = additive. Risk higher than ACOX1 (which is RELATIVE CI only — no optic atrophy)",
            "VPA: RELATIVE CI — hepatotoxicity + POLG1 MANDATORY (CPIC Grade A); bile acid metabolism disruption in HSD17B4 adds hepatic concern",
            "DBP types: Type I (hydratase D1 + 3-HSD D2 deficient, ~50%) · Type II (hydratase D1 only, ~30%) · Type III (3-HSD D2 only, ~20%) — Type III often milder",
            "THCA/DHCA ELEVATED — bile acid intermediates; HSD17B4 required for peroxisomal chain shortening of bile acid precursors to CDCA/CA; additional biochemical marker vs ACOX1",
            "No ERT (2026) — HSD17B4 is peroxisomal matrix enzyme; systemic ERT cannot reach peroxisomal matrix as functional enzyme",
            "No HSCT — DBP deficiency is NOT inflammatory demyelination; HSCT targets Krabbe/GALC and ABCD1-CCALD (both inflammatory); DBP = substrate accumulation neurotoxicity",
            "DHA supplementation Level C — secondary DHA deficit from impaired peroxisomal synthesis; 200–500 mg/day; safe",
            "LEV first-line all forms. Phenobarbital neonatal. ACTH Level B IS (preferred over VGB — optic atrophy additive). OXC/CBZ focal (safe — no adrenal).",
        ],
        "standards": [
            "Watkins PA, Chen WW, Harris CJ, et al. Peroxisomal bifunctional enzyme deficiency. J Clin Invest. 1989;83(3):771–777.",
            "Ferdinandusse S, Denis S, Mooijer PA, et al. Identification of the peroxisomal beta-oxidation enzymes involved in the degradation of long-chain dicarboxylic acids. J Lipid Res. 2004;45(6):1104–1111.",
            "Möller G, van Grunsven EG, Wanders RJA, Adamski J. Molecular basis of D-bifunctional protein deficiency. Mol Cell Endocrinol. 2001;171(1–2):61–70.",
            "van Grunsven EG, van Berkel E, Ijlst L, et al. Peroxisomal D-hydroxyacyl-CoA dehydrogenase deficiency: resolution of the enzyme defect and its molecular basis in bifunctional protein deficiency. Proc Natl Acad Sci USA. 1998;95(5):2128–2133.",
            "Waterham HR, Ferdinandusse S, Wanders RJA. Human disorders of peroxisome metabolism. Biochim Biophys Acta. 2016;1863(5):922–933.",
            "CPIC Guideline — Valproic Acid and POLG1 (Grade A). cpicpgx.org. 2023.",
        ],
    }


# ── Breakdown (patients + seizures + treatments) ─────────────────────────────

def get_breakdown():
    etiologies = [
        {
            "name": "DBP Type I (Hydratase D1 + 3-HSD D2 Both Deficient)",
            "pct": 50,
            "n": 20,
            "sex": "M/F equal (AR)",
            "onset_age": "Neonatal (0–1 month)",
            "seizure_risk": "90%",
            "eeg": "Multifocal clonic discharges, burst-suppression, hypsarrhythmia (IS), generalized slowing",
            "mri": "Severe leukodystrophy (posterior/parieto-occipital predominant), pachygyria/simplified gyri in subset, diffuse WM T2 hyperintensity",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic mutations abolishing both hydratase (domain 1) AND 3-hydroxyacyl-CoA "
                "dehydrogenase (domain 2) activity → complete block of steps 2 AND 3 of peroxisomal "
                "beta-oxidation for ALL substrates. VLCFA (C26:0) severely elevated. Pristanic "
                "SEVERELY ELEVATED (hallmark). THCA/DHCA elevated. Plasmalogens near-normal. "
                "Neonatal hypotonia 95%. Seizures 90%. SNHL 85%. Retinal dystrophy 80%. "
                "Optic atrophy 65%. Leukodystrophy severe. Median survival 1–3 years."
            ),
        },
        {
            "name": "DBP Type II (Hydratase D1 Deficient Only)",
            "pct": 30,
            "n": 12,
            "sex": "M/F equal (AR)",
            "onset_age": "Neonatal to early infantile (0–3 months)",
            "seizure_risk": "75–85%",
            "eeg": "Multifocal spikes, modified hypsarrhythmia, focal temporal/occipital discharges",
            "mri": "Moderate–severe leukodystrophy; posterior predominant; some simplified gyration",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Hydratase (domain 1) deficient; 3-hydroxyacyl-CoA dehydrogenase (domain 2) intact. "
                "Despite D2 being intact, hydratase activity is required for ALL substrates → VLCFA, "
                "pristanic, THCA/DHCA all still accumulate (cannot proceed without step 2). "
                "Clinical severity similar to Type I. Pristanic SEVERELY ELEVATED. THCA elevated. "
                "Seizures 75-85%. SNHL 85%. Retinal dystrophy 70%. Survival <5 years typical. "
                "Plasmalogens normal (PTS2 intact). Gene: HSD17B4 biallelic LOF in hydratase domain."
            ),
        },
        {
            "name": "DBP Type III (3-HSD D2 Deficient Only — Milder Peroxisomal Form)",
            "pct": 20,
            "n": 8,
            "sex": "M/F equal; males may have gonadal phenotype",
            "onset_age": "Early childhood (3–24 months) or later",
            "seizure_risk": "50–65%",
            "eeg": "Focal discharges, myoclonic, relatively less severe hypsarrhythmia",
            "mri": "Mild–moderate leukodystrophy; some patients near-normal early",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "3-hydroxyacyl-CoA dehydrogenase (domain 2) deficient; hydratase (domain 1) intact. "
                "Peroxisomal beta-oxidation partially impaired at step 3 → VLCFA + pristanic accumulate "
                "but degree variable (residual activity possible via alternative routes in some substrates). "
                "MILDER neurological course. Males: testosterone deficiency, androstenedione elevated "
                "(gonadal 17β-HSD role). Seizures 50-65%. Survival into childhood/adolescence possible. "
                "Retinal dystrophy 55%. SNHL 60%. Some patients survive to adulthood (rare). "
                "Gene: HSD17B4 biallelic LOF in dehydrogenase domain (not hydratase domain)."
            ),
        },
    ]

    phenotypes = (
        [("DBP Type I", "Severe")] * 20
        + [("DBP Type II", "Moderate")] * 12
        + [("DBP Type III", "Mild")] * 8
    )
    sexes = ["M"] * 20 + ["F"] * 20
    random.shuffle(sexes)

    genotype_map = {
        "DBP Type I": [
            "Biallelic null (frameshift + nonsense)",
            "Null + splice LOF (both domains)",
            "Homozygous large deletion 5q23.1",
        ],
        "DBP Type II": [
            "Biallelic missense in hydratase domain (D1)",
            "Null + hydratase-domain missense",
            "Splice variant + hydratase LOF",
        ],
        "DBP Type III": [
            "Biallelic missense in 3-HSD domain (D2)",
            "Compound heterozygous D2 domain",
            "Null + D2-domain hypomorphic",
        ],
    }

    seizure_map = {
        "Severe": ["Neonatal multifocal clonic", "Infantile spasms", "Status epilepticus"],
        "Moderate": ["Infantile spasms", "Focal with bilateral spread", "Myoclonic"],
        "Mild": ["Focal with bilateral spread", "Myoclonic", None],
    }

    trigger_pool = [
        "Febrile illness",
        "Intercurrent infection",
        "Sleep deprivation",
        "Missed AED",
        "Metabolic decompensation",
        "Stress / illness",
        "Fasting / caloric restriction",
    ]

    treatment_pool = [
        "LEV (first-line)",
        "Phenobarbital (neonatal — Level B)",
        "ACTH (IS — Level B, preferred over VGB)",
        "OXC/CBZ (focal — CAN USE, no adrenal)",
        "DHA supplementation (Level C)",
        "CLB (adjunct — Level C)",
    ]

    patients = []
    for i, ((pheno, severity), sex) in enumerate(zip(phenotypes, sexes)):
        pid = f"HSD17B4-{i+1:02d}"

        if severity == "Severe":
            age_mo = random.randint(0, 1)
            has_sz = random.random() < 0.90
            vlcfa_c26 = round(random.uniform(0.8, 4.0), 2)
            pristanic = round(random.uniform(3.5, 15.0), 2)
            thca_elevated = True
            leuko_grade = random.choice(["Severe", "Severe", "Moderate"])
        elif severity == "Moderate":
            age_mo = random.randint(0, 3)
            has_sz = random.random() < 0.80
            vlcfa_c26 = round(random.uniform(0.6, 2.5), 2)
            pristanic = round(random.uniform(2.0, 10.0), 2)
            thca_elevated = True
            leuko_grade = random.choice(["Moderate", "Moderate", "Severe"])
        else:
            age_mo = random.randint(3, 24)
            has_sz = random.random() < 0.58
            vlcfa_c26 = round(random.uniform(0.4, 1.5), 2)
            pristanic = round(random.uniform(0.8, 4.0), 2)
            thca_elevated = random.random() < 0.65
            leuko_grade = random.choice(["Mild", "Moderate"])

        seizure_type = random.choice(seizure_map[severity]) if has_sz else None
        drug_resistant = has_sz and random.random() < 0.38
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
            "retinal_dystrophy": random.random() < (0.80 if severity == "Severe" else 0.70 if severity == "Moderate" else 0.55),
            "optic_atrophy": random.random() < (0.65 if severity == "Severe" else 0.55 if severity == "Moderate" else 0.35),
            "snhl": random.random() < (0.85 if severity == "Severe" else 0.80 if severity == "Moderate" else 0.60),
            "leukodystrophy_grade": leuko_grade,
            "vlcfa_c26_umol_l": vlcfa_c26,
            "pristanic_umol_l": pristanic,
            "thca_elevated": thca_elevated,
            "plasmalogen_normal": True,
            "genotype": genotype,
            "current_treatment": current_treatment,
            "triggers": triggers,
            "polg1_tested": True,
        })

    seizure_types = [
        {
            "type": "Neonatal multifocal clonic",
            "pct": 35,
            "preferred_tx": "LEV + phenobarbital (neonatal)",
            "notes": "Onset first days-weeks of life; multifocal or migratory clonic; EEG burst-suppression or multifocal spikes; IV LEV + phenobarbital standard first-line.",
        },
        {
            "type": "Infantile spasms",
            "pct": 27,
            "preferred_tx": "ACTH Level B (VGB HIGH RISK — retinal dystrophy + optic atrophy additive)",
            "notes": "Hypsarrhythmia on EEG; ACTH strongly preferred over VGB — retinal dystrophy (75%) + optic atrophy (60%) + VGB peripheral VF constriction = HIGH additive visual risk. Monthly VEP/VF if VGB used.",
        },
        {
            "type": "Focal with bilateral spread",
            "pct": 20,
            "preferred_tx": "LEV, OXC (CAN USE — no adrenal mechanism)",
            "notes": "Temporal or parieto-occipital onset (posterior leukodystrophy); PHT/CBZ/OXC can be used (no adrenal insufficiency unlike ABCD1 where enzyme inducers = ABSOLUTE CI).",
        },
        {
            "type": "Myoclonic",
            "pct": 13,
            "preferred_tx": "LEV, CLB",
            "notes": "Myoclonic jerks with or without EEG correlate; LEV first-line; CLB add-on (Level C); avoid VPA (RELATIVE CI — hepatotoxicity + POLG1 mandatory + bile acid metabolism disruption).",
        },
        {
            "type": "Status epilepticus",
            "pct": 5,
            "preferred_tx": "IV LEV, benzodiazepines",
            "notes": "Emergency management: IV lorazepam/diazepam → IV LEV; fosphenytoin acceptable (no adrenal mechanism); IV dextrose if prolonged fasting (metabolic decompensation risk).",
        },
    ]

    triggers = [
        {"trigger": "Febrile illness", "pct": 65},
        {"trigger": "Intercurrent infection", "pct": 55},
        {"trigger": "Sleep deprivation", "pct": 40},
        {"trigger": "Missed AED", "pct": 32},
        {"trigger": "Metabolic decompensation", "pct": 30},
        {"trigger": "Stress / illness", "pct": 25},
        {"trigger": "Fasting / caloric restriction", "pct": 18},
    ]

    monitoring = [
        {
            "parameter": "Plasma VLCFA (C26:0, C24:0, C26:0/C22:0 ratio)",
            "threshold": "C26:0 >1.5 µmol/L (normal <0.3); elevation in ALL types; required for diagnosis",
            "frequency": "Every 6 months",
        },
        {
            "parameter": "Plasma pristanic acid",
            "threshold": "Target: normal (<3.0 µmol/L); SEVERELY ELEVATED in HSD17B4 — KEY DISTINCTION from ACOX1 (normal <1.0)",
            "frequency": "Every 6 months",
        },
        {
            "parameter": "Plasma THCA / DHCA (bile acid intermediates)",
            "threshold": "Elevated in Types I and II; distinguishes HSD17B4 from ACOX1 (normal in ACOX1); guides dietary intervention",
            "frequency": "Every 6 months",
        },
        {
            "parameter": "RBC plasmalogens (C16:0-DMA, C18:0-DMA)",
            "threshold": "NORMAL or near-normal (distinguishes from ZSD where severely low); obtain at diagnosis",
            "frequency": "At diagnosis; annually",
        },
        {
            "parameter": "Visual acuity + ERG + VEP + visual fields",
            "threshold": "Retinal dystrophy 75%, optic atrophy 60%; monthly if VGB used; HIGH priority vs ACOX1 (higher risk profile with combined retinal + optic atrophy)",
            "frequency": "Every 6 months; monthly if VGB",
        },
        {
            "parameter": "Liver function tests (LFTs) + bile acid profile",
            "threshold": "HSD17B4 causes hepatic bile acid accumulation (THCA/DHCA); monitor LFTs — especially critical if VPA used (RELATIVE CI — 3 mechanisms)",
            "frequency": "Every 3 months on VPA; every 6 months otherwise",
        },
        {
            "parameter": "Audiogram (SNHL)",
            "threshold": "SNHL in 85%; highest among peroxisomal single-enzyme deficiencies; early hearing aids/cochlear implant planning",
            "frequency": "Every 6 months",
        },
        {
            "parameter": "Brain MRI (leukodystrophy progression)",
            "threshold": "Posterior-predominant WM T2 signal; track Loes-equivalent score; simplified gyration in subset",
            "frequency": "Every 12 months",
        },
    ]

    lifecycle = [
        {
            "stage": "Neonatal (0–1 mo)",
            "features": "Hypotonia (95%), feeding difficulties, seizures begin (multifocal clonic); VLCFA + pristanic + THCA elevated; plasmalogens NORMAL; ERG ordered at diagnosis; IV dextrose if fasting risk.",
        },
        {
            "stage": "Infantile (1–12 mo)",
            "features": "Leukodystrophy progression; infantile spasms onset; ACTH Level B preferred (VGB HIGH RISK — retinal dystrophy + optic atrophy additive); audiological assessment (SNHL 85%); DHA supplementation started.",
        },
        {
            "stage": "Early childhood (1–5 y)",
            "features": "Progressive posterior-predominant leukodystrophy; SNHL worsens (85%); visual decline (retinal dystrophy + optic atrophy combined); drug-resistant seizures in 38%; hearing aids; cochlear implant evaluation. Type I/II: often neurologically severe.",
        },
        {
            "stage": "Late childhood (5–12 y)",
            "features": "Severe neurological disability in types I and II (median survival 1–3 years); type III may remain ambulatory; spastic quadriplegia; seizure burden may decrease; bile acid monitoring ongoing.",
        },
        {
            "stage": "Adolescent (12–18 y)",
            "features": "Very rare in types I/II (very few survive); type III possible — ambulatory with support, moderate cognitive impairment; males with type III may have gonadal phenotype (testosterone deficiency); seizure management continues.",
        },
        {
            "stage": "Adult (>18 y)",
            "features": "Extremely rare in types I/II; type III rare survivors into adulthood reported; DHA + seizure management; bile acid monitoring; in males, testosterone supplementation may be required (type III gonadal involvement).",
        },
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "SV2A modulator",
            "level": "First-line (Level A — all forms)",
            "dose": "20–60 mg/kg/day in 2 divided doses; IV for SE",
            "notes": (
                "No peroxisomal interactions; no adrenal effect; no VLCFA/pristanic impact. "
                "Safe across all HSD17B4 types. Preferred across neonatal, infantile, and childhood onset. "
                "IV formulation suitable for SE and neonatal emergencies."
            ),
            "ci": "None specific to HSD17B4",
        },
        {
            "drug": "Phenobarbital",
            "class": "Barbiturate — GABA-A potentiator",
            "level": "Level B (neonatal seizures)",
            "dose": "Loading 20 mg/kg IV; maintenance 3–5 mg/kg/day",
            "notes": (
                "Standard neonatal seizure management alongside LEV. "
                "Respiratory monitoring required in neonates. "
                "No peroxisomal interactions. No adrenal mechanism in HSD17B4."
            ),
            "ci": "Respiratory depression; use with monitoring in neonatal ICU",
        },
        {
            "drug": "ACTH / Corticotropin",
            "class": "Infantile spasms — hormone therapy",
            "level": "Level B (IS — strongly preferred over VGB)",
            "dose": "150 IU/m²/day IM × 2 weeks then taper",
            "notes": (
                "STRONGLY preferred over VGB for IS in HSD17B4: retinal dystrophy (75%) + optic atrophy (60%) "
                "+ VGB VF constriction = HIGH additive visual risk (higher risk profile than ACOX1). "
                "Adrenal axis intact in HSD17B4 (no adrenal insufficiency). "
                "POLG1 status must be confirmed before any VPA co-prescription."
            ),
            "ci": "Monitor glycaemia, BP, infection; immunosuppression risk",
        },
        {
            "drug": "OXC / CBZ (Oxcarbazepine / Carbamazepine)",
            "class": "Sodium channel blocker",
            "level": "Level B (focal seizures — CAN USE)",
            "dose": "OXC 10–40 mg/kg/day; CBZ 5–20 mg/kg/day",
            "notes": (
                "CAN BE USED in HSD17B4 — no adrenal mechanism (contrast ABCD1/X-ALD: PHT/CBZ = "
                "ABSOLUTE CI because CYP3A4-mediated cortisol degradation → adrenal crisis). "
                "HSD17B4 has no adrenal insufficiency → enzyme inducers safe. Standard hepatic monitoring. "
                "Same safety profile as ACOX1 and RCDP series for this drug class."
            ),
            "ci": "None specific to HSD17B4; standard SJS/TEN monitoring; hyponatraemia with OXC",
        },
        {
            "drug": "DHA (Docosahexaenoic acid)",
            "class": "Omega-3 supplementation",
            "level": "Level C",
            "dose": "200 mg/day infants; 500 mg/day children",
            "notes": (
                "Reduces secondary DHA-plasmalogen deficit from impaired peroxisomal beta-oxidation "
                "of omega-3 precursors. Retinal DHA supplementation may partially support photoreceptors. "
                "Safe, no drug interactions. Used across all HSD17B4 types."
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
                "Sedation risk in severe forms. Tolerance may develop with prolonged use."
            ),
            "ci": "None specific to HSD17B4",
        },
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "level": "RELATIVE CI — hepatotoxicity + POLG1 MANDATORY (CPIC Grade A) + bile acid metabolism disruption",
            "reason": (
                "THREE hepatotoxicity mechanisms: (a) carnitine depletion, (b) peroxisomal beta-oxidation "
                "interference (direct inhibition of HSD17B4 substrate processing — NOTE: worsens already "
                "deficient enzyme pathway), (c) mitochondrial toxicity (POLG1 mandatory CPIC A). "
                "ADDITIONAL CONCERN in HSD17B4: disrupted bile acid metabolism (THCA/DHCA accumulation) "
                "increases hepatic metabolic burden — VPA hepatotoxicity risk is HIGHER than in ACOX1. "
                "LFT monitoring q3 months if used. POLG1 test MANDATORY before VPA."
            ),
            "alternative": "LEV (first-line) · OXC/CBZ (focal) · CLB (adjunct)",
        },
        {
            "drug": "Vigabatrin (VGB)",
            "level": "HIGH RISK — retinal toxicity additive (higher risk than ACOX1 due to combined retinal + optic atrophy)",
            "reason": (
                "Retinal dystrophy in 75% + optic atrophy in 60% + VGB irreversible peripheral visual "
                "field constriction = HIGH additive visual risk. RISK IS HIGHER THAN ACOX1 (which has "
                "only retinal degeneration without optic atrophy). Use only if no alternative; "
                "monthly VEP/VF mandatory if VGB used. Prefer ACTH for IS. "
                "NOT absolute CI if no alternative exists — but HIGH RISK. Monthly monitoring essential."
            ),
            "alternative": "ACTH (Level B for IS, strongly preferred) · LEV (focal/myoclonic)",
        },
        {
            "drug": "PHT/CBZ (phenytoin/carbamazepine) — clarification",
            "level": "CAN USE — no adrenal mechanism (CRITICAL CONTRAST to ABCD1)",
            "reason": (
                "PHT/CBZ carry NO adrenal-insufficiency risk in HSD17B4 (no adrenal crisis mechanism). "
                "CRITICAL CONTRAST: In ABCD1/X-ALD, PHT/CBZ = ABSOLUTE CI because CYP3A4 enzyme "
                "inducers accelerate cortisol metabolism → adrenal crisis. HSD17B4 has no adrenal "
                "insufficiency → PHT/CBZ safe. Standard hepatic and cardiac monitoring applies."
            ),
            "alternative": "LEV preferred first-line; PHT/CBZ acceptable when clinically indicated",
        },
        {
            "drug": "Lorenzo oil",
            "level": "NOT RECOMMENDED — mechanism mismatch (same as ACOX1)",
            "reason": (
                "Lorenzo oil inhibits VLCFA elongase → may partially lower plasma VLCFA but does NOT "
                "restore HSD17B4 steps 2+3 of beta-oxidation. Pristanic acid does NOT decrease "
                "(pristanic elongase is not the therapeutic target). THCA/DHCA also unaffected. "
                "No proven clinical benefit in HSD17B4 deficiency."
            ),
            "alternative": "DHA supplementation (Level C) · seizure management with LEV",
        },
        {
            "drug": "Fasting / prolonged caloric restriction",
            "level": "HAZARD — metabolic decompensation risk (pristanic + VLCFA surge)",
            "reason": (
                "Peroxisomal beta-oxidation integral to fasting fatty acid metabolism. Fasting → "
                "VLCFA + pristanic accumulation surge and risk of acute metabolic decompensation. "
                "Bile acid synthesis also impaired → additional hepatic metabolic stress. "
                "IV dextrose MANDATORY perioperatively. Avoid prolonged fasting."
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
            "HSD17B4 = D-bifunctional protein (DBP) / MFP-2 (736 aa, PTS1-SRL): steps 2+3 of peroxisomal beta-oxidation for ALL substrates — VLCFAs (C24:0, C26:0), branched-chain (pristanic), bile acid intermediates (THCA/DHCA)",
            "HSD17B4 deficiency → VLCFA (C26:0) ELEVATED + PRISTANIC SEVERELY ELEVATED + THCA/DHCA ELEVATED → multi-substrate peroxisomal beta-oxidation failure (contrast ACOX1: only step 1 straight-chain VLCFA blocked → pristanic NORMAL)",
            "PRISTANIC ACID SEVERELY ELEVATED — THE KEY DISTINCTION from ACOX1: ACOX1 blocks only step 1 of straight-chain VLCFA; ACOX2 (step 1 of pristanic) is intact in ACOX1 but steps 2-3 (HSD17B4) are deficient in HSD17B4 → pristanic accumulates in HSD17B4 not ACOX1",
            "THCA/DHCA ELEVATED — bile acid intermediates cannot undergo peroxisomal chain shortening without HSD17B4 → additional biochemical marker distinguishing HSD17B4 from ACOX1 (THCA/DHCA normal in ACOX1)",
            "Plasmalogens NORMAL / near-normal — PTS2 pathway intact (HSD17B4 is PTS1 matrix enzyme); KEY DISTINCTION from ZSD (PEX1/PEX6) where plasmalogens severely low",
            "No adrenal insufficiency — HSD17B4 does NOT cause adrenal crisis; PHT/CBZ/OXC CAN BE USED; critical contrast to ABCD1 (ABSOLUTE CI — adrenal insufficiency universal)",
            "VGB: HIGH RISK in HSD17B4 (retinal dystrophy 75% + optic atrophy 60% + VGB VF constriction = HIGH additive risk); risk HIGHER than ACOX1 (which has retinal degeneration only, no optic atrophy); prefer ACTH for IS",
            "VPA: RELATIVE CI — hepatotoxicity (3 mechanisms: carnitine depletion, peroxisomal beta-ox interference worsening already deficient HSD17B4 pathway, mitochondrial toxicity); ADDITIONALLY bile acid accumulation (THCA/DHCA) increases hepatic burden; POLG1 MANDATORY CPIC A",
            "DBP Type I (50%): Both hydratase (D1) + 3-HSD (D2) deficient → ALL substrates affected → most severe (ZSD-like, neonatal onset, median survival 1–3 years)",
            "DBP Type II (30%): Only hydratase (D1) deficient → same clinical severity as Type I (hydratase required for ALL substrates even if D2 intact) → neonatal onset severe",
            "DBP Type III (20%): Only 3-HSD (D2) deficient → milder peroxisomal disease; males may have gonadal phenotype (testosterone biosynthesis affected by same D2 activity in gonads)",
            "No ERT (2026) — HSD17B4 is peroxisomal matrix enzyme; systemic ERT cannot reach peroxisomal matrix as functional enzyme",
            "No HSCT — DBP deficiency is substrate accumulation neurotoxicity (NOT inflammatory demyelination like Krabbe/GALC or ABCD1-CCALD); HSCT is only effective for inflammatory demyelination",
            "DHA supplementation Level C (200–500 mg/day); secondary DHA deficit from impaired peroxisomal synthesis of precursors; safe, no drug interactions",
            "LEV first-line all forms. Phenobarbital neonatal. ACTH Level B IS (strongly preferred over VGB — combined retinal + optic atrophy risk). OXC/CBZ focal (CAN USE — no adrenal). POLG1 MANDATORY before VPA. Lorenzo oil NOT recommended.",
        ],
        "diagnostic_algorithm": [
            "Newborn with hypotonia, seizures, dysmorphic features OR neonatal seizures in female (equal sex ratio, AR) → order plasma VLCFA panel (C26:0, C24:0, C26:0/C22:0 ratio) — elevated VLCFA points to peroxisomal disease.",
            "If VLCFA elevated → order RBC plasmalogens SIMULTANEOUSLY — NORMAL/near-normal plasmalogens indicate single-enzyme defect (HSD17B4 or ACOX1), NOT ZSD/PBD (ZSD = severely low plasmalogens).",
            "Order plasma pristanic acid SIMULTANEOUSLY with VLCFA — PRISTANIC SEVERELY ELEVATED → HSD17B4 (NOT ACOX1 where pristanic NORMAL); this single test distinguishes the two.",
            "Order plasma THCA/DHCA (bile acid intermediates) — elevated in HSD17B4 (normal in ACOX1); confirms HSD17B4 diagnosis category.",
            "Phytanic acid (to distinguish PHYH/Refsum — in Refsum phytanic SEVERELY ELEVATED, VLCFA NORMAL; in HSD17B4 phytanic may be NORMAL or mildly elevated).",
            "Brain MRI: leukodystrophy pattern (posterior/parieto-occipital predominant WM T2 signal); simplified gyration or pachygyria in subset; more severe than ACOX1 in typical cases.",
            "ERG (electroretinogram) + VEP + visual fields: retinal dystrophy 75%, optic atrophy 60%; order at diagnosis; CRITICAL before considering VGB (HIGH RISK in HSD17B4).",
            "Audiogram: SNHL in 85% — highest among peroxisomal single-enzyme defects; order at diagnosis; early hearing aids if abnormal.",
            "In males with type III presentation: testosterone/androstenedione ratio — testosterone LOW, androstenedione HIGH (gonadal 17β-HSD role of same enzyme); may co-present with neurological disease.",
            "Fibroblast HSD17B4 enzyme assay (gold standard): hydratase and/or dehydrogenase activity measured separately → defines type I/II/III.",
            "HSD17B4 gene sequencing: confirms biallelic LOF; identifies domain affected (D1 hydratase vs D2 dehydrogenase); guides type classification. POLG1 MANDATORY before any VPA consideration.",
            "Liver biopsy NOT required but if performed: hepatomegaly with bile acid accumulation pattern; peroxisomes normal in number (single-enzyme defect); rules out ZSD where peroxisomes absent/reduced.",
        ],
        "pharmacological_distinctions": [
            "PRISTANIC ACID is THE KEY MARKER: SEVERELY ELEVATED in HSD17B4 (NORMAL in ACOX1). Both have VLCFA elevated → pristanic measurement MANDATORY when VLCFA elevated; VLCFA alone CANNOT distinguish.",
            "VGB: HIGH RISK in HSD17B4 (retinal dystrophy + optic atrophy combined = higher visual impairment than ACOX1 where only retinal degeneration; no optic atrophy). Risk profile HIGHER than ACOX1. Prefer ACTH for IS in BOTH, but urgency greater in HSD17B4.",
            "PHT/CBZ/OXC = CAN USE in HSD17B4 (no adrenal insufficiency). CRITICAL CONTRAST: In ABCD1/X-ALD, PHT/CBZ = ABSOLUTE CI (CYP3A4 cortisol degradation → adrenal crisis). Same safe profile as ACOX1, RCDP series — no adrenal mechanism.",
            "VPA: RELATIVE CI in HSD17B4 — THREE mechanisms (carnitine depletion, peroxisomal beta-ox interference — exacerbates already-deficient HSD17B4 pathway, mitochondrial toxicity). ADDITIONAL CONCERN: bile acid metabolism disrupted (THCA accumulation) → hepatic burden HIGHER than ACOX1. POLG1 MANDATORY CPIC A. LFT q3 months.",
            "VPA hepatic risk is HIGHER in HSD17B4 than ACOX1 due to combined: (1) standard 3 mechanisms + (2) disrupted bile acid processing (THCA/DHCA accumulation → additional hepatic metabolic stress). This distinction is clinically significant.",
            "Lorenzo oil: NOT RECOMMENDED (same as ACOX1) — inhibits elongase but does NOT restore HSD17B4 steps 2+3 of beta-oxidation; pristanic acid and THCA/DHCA do NOT decrease with Lorenzo oil (different metabolic pathway).",
            "DHA: Level C (200–500 mg/day); secondary DHA deficit from impaired peroxisomal beta-oxidation; retinal photoreceptors may benefit from DHA supplementation (important given retinal dystrophy 75%).",
            "ACTH for IS: Level B; STRONGLY preferred over VGB in HSD17B4 (combined retinal + optic atrophy = higher VGB risk than ACOX1). Same preference but MORE URGENT than in RCDP series or ACOX1.",
            "No ERT (2026) — HSD17B4 peroxisomal matrix enzyme (PTS1); systemic delivery cannot cross peroxisomal membrane; no secreted isoform exists.",
            "No HSCT — not inflammatory demyelination; HSCT targets Krabbe (psychosine) and CCALD (ABCD1/inflammatory); DBP deficiency = substrate toxicity (VLCFA + pristanic + THCA accumulation); HSCT not effective.",
            "POLG1 test MANDATORY before VPA (CPIC Grade A) — never skip; bile acid monitoring (THCA/DHCA) every 6 months regardless of treatment choice.",
            "Fasting HAZARD: IV glucose perioperatively mandatory (same as ACOX1); additional concern: impaired bile acid synthesis under metabolic stress; avoid prolonged fasting in all HSD17B4 types.",
        ],
        "differential_diagnosis": [
            {
                "condition": "ACOX1 (Pseudo-NALD — ACOX1 deficiency)",
                "distinction": "VLCFA elevated in BOTH; PRISTANIC NORMAL in ACOX1 (SEVERELY ELEVATED in HSD17B4) — KEY DISTINCTION; THCA/DHCA normal in ACOX1; HSD17B4 blocks steps 2+3 affecting all substrates while ACOX1 blocks only step 1 of straight-chain; plasmalogens normal in BOTH.",
            },
            {
                "condition": "ZSD-Severe (PEX1/PEX6)",
                "distinction": "VLCFA elevated in BOTH; plasmalogens SEVERELY LOW in ZSD (NORMAL in HSD17B4); ZSD affects ALL peroxisomal enzymes (biogenesis disorder); ZSD has dysmorphic features, cortical malformations more prominent; phytanic/pristanic both elevated in ZSD; PEX gene panel confirms.",
            },
            {
                "condition": "ABCD1 (X-ALD)",
                "distinction": "VLCFA elevated in BOTH; X-linked (males primarily in ABCD1); adrenal insufficiency mandatory in ABCD1 (absent in HSD17B4); PHT/CBZ ABSOLUTE CI in ABCD1 (CAN USE in HSD17B4); ABCD1: VLCFA elevated, pristanic NORMAL; HSD17B4: BOTH VLCFA and pristanic elevated.",
            },
            {
                "condition": "PHYH deficiency (Adult Refsum Disease)",
                "distinction": "Phytanic SEVERELY ELEVATED in Refsum (normal/mildly elevated in HSD17B4); VLCFA NORMAL in Refsum (elevated in HSD17B4); pristanic mildly elevated in Refsum (severely elevated in HSD17B4); adult-onset RP + polyneuropathy + anosmia in Refsum; neonatal onset in HSD17B4.",
            },
            {
                "condition": "AMACR deficiency (alpha-methylacyl-CoA racemase)",
                "distinction": "Pristanic elevated in BOTH (AMACR blocks conversion of R-pristanyl-CoA to S-pristanyl-CoA before ACOX2/HSD17B4); VLCFA mildly elevated in AMACR; THCA/DHCA elevated in AMACR; adult-onset typically in AMACR (contrast neonatal in HSD17B4); AR; gene sequencing distinguishes.",
            },
            {
                "condition": "RCDP (PEX7/GNPAT/AGPS/FAR1)",
                "distinction": "Plasmalogens SEVERELY LOW in RCDP (NORMAL in HSD17B4); VLCFA NORMAL in RCDP (elevated in HSD17B4); RCDP: rhizomelia, stippled epiphyses, cataracts (structural malformations); no pristanic elevation pattern in RCDP; completely different biochemical and clinical profile.",
            },
            {
                "condition": "SCP2/SCPx deficiency (sterol carrier protein)",
                "distinction": "Pristanic elevated in BOTH (SCPx is thiolase for step 4 of branched-chain beta-oxidation — different step from HSD17B4 steps 2+3); VLCFA only mildly elevated in SCP2; bile acid abnormalities in both; gene sequencing on HSD17B4 vs SCP2 distinguishes; very rare.",
            },
        ],
        "standards": [
            "Watkins PA, Chen WW, Harris CJ, et al. Peroxisomal bifunctional enzyme deficiency. J Clin Invest. 1989;83(3):771–777.",
            "Ferdinandusse S, Denis S, Mooijer PA, et al. Identification of the peroxisomal beta-oxidation enzymes involved in the degradation of long-chain dicarboxylic acids. J Lipid Res. 2004;45(6):1104–1111.",
            "Möller G, van Grunsven EG, Wanders RJA, Adamski J. Molecular basis of D-bifunctional protein deficiency. Mol Cell Endocrinol. 2001;171(1–2):61–70.",
            "van Grunsven EG, van Berkel E, Ijlst L, et al. Peroxisomal D-hydroxyacyl-CoA dehydrogenase deficiency: resolution of the enzyme defect and its molecular basis in bifunctional protein deficiency. Proc Natl Acad Sci USA. 1998;95(5):2128–2133.",
            "Waterham HR, Ferdinandusse S, Wanders RJA. Human disorders of peroxisome metabolism. Biochim Biophys Acta. 2016;1863(5):922–933.",
            "CPIC Guideline — Valproic Acid and POLG1 (Grade A). cpicpgx.org. 2023.",
        ],
    }
