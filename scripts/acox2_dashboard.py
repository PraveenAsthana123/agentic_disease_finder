#!/usr/bin/env python3
"""ACOX2 (Acyl-CoA Oxidase 2) Deficiency — Branched-Chain Peroxisomal Beta-Oxidation Step 1 Epilepsy Dashboard.

ACOX2 encodes acyl-CoA oxidase 2 (branched-chain fatty acid oxidase / pristanoyl-CoA oxidase),
a ~672 aa peroxisomal matrix enzyme. PTS1 targeted (C-terminal SRL signal).

PEROXISOMAL BRANCHED-CHAIN BETA-OXIDATION — ACOX2 position:
  ─ Prerequisite: AMACR (racemization): (2R)-pristanoyl-CoA → (2S)-pristanoyl-CoA
  ─ Step 1: ACOX2 (FAD-dependent oxidase, homotrimer, PTS1-SRL):
              (2S)-pristanoyl-CoA + FAD → 2-trans-pristanoyl-CoA + FADH₂
    ALSO: ACOX2 oxidises C27 bile acid CoA esters:
              THCA-CoA → 2-trans-THCA-CoA  (THCA = trihydroxycholestanoic acid)
              DHCA-CoA → 2-trans-DHCA-CoA  (DHCA = dihydroxycholestanoic acid)
  ─ Step 2 + 3: HSD17B4 (MFP2/D-bifunctional protein, 3-domain, PTS1-SRL)
  ─ Step 4: SCP2/SCPx (thiolytic cleavage) → propionyl-CoA + shorter acyl-CoA released
  ─────────────────────────────────────────────────────────────────────────────────────────────
  ACOX2 = FIRST COMMITTED OXIDATION STEP of branched-chain peroxisomal beta-oxidation.
  Without ACOX2, pristanoyl-CoA and bile acid CoA esters CANNOT progress.
  PRISTANIC ACID ACCUMULATES. THCA and DHCA ACCUMULATE (bile acid intermediates).
  VLCFA (straight-chain, oxidised by ACOX1) is UNAFFECTED → VLCFA NORMAL.
  Plasmalogen synthesis (PTS2 pathway: PEX7/GNPAT/AGPS/FAR1) UNAFFECTED → PLASMALOGENS NORMAL.
  Alpha-oxidation (PHYH) UNAFFECTED → PHYTANIC NORMAL.

ACOX2 vs ACOX1 — THE CRITICAL DISTINCTION:
  ACOX1 (PSEUDO-NALD, 17q25.1, ~660 aa, PTS1-SRL, HOMOTRIMER):
    Substrate: palmitoyl-CoA, C22:0, C24:0, C26:0 (straight-chain VLCFA) — step 1 of VLCFA beta-oxidation
    VLCFA (C26:0) ELEVATED in ACOX1 deficiency. Pristanic NORMAL.
  ACOX2 (3p25.1, ~672 aa, PTS1-SRL, HOMOTRIMER):
    Substrate: (2S)-pristanoyl-CoA, THCA-CoA, DHCA-CoA (branched-chain + bile acid CoA esters)
    Pristanic ELEVATED, THCA/DHCA ELEVATED in ACOX2 deficiency. VLCFA NORMAL.
  Plasma VLCFA panel is THE KEY DISTINCTION: VLCFA ↑ → ACOX1; VLCFA NORMAL + pristanic ↑ → ACOX2.

BIOCHEMICAL PROFILE (ACOX2 deficiency):
  Pristanic acid:  SEVERELY ELEVATED — branched-chain step 1 blocked; cannot progress to HSD17B4/SCP2
  THCA (3α,7α,12α-trihydroxycholestanoic acid): ELEVATED — bile acid CoA ester oxidation blocked
  DHCA (3α,7α-dihydroxycholestanoic acid):       ELEVATED — bile acid CoA ester oxidation blocked
  VLCFA (C26:0):  NORMAL — straight-chain ACOX1 pathway unaffected (KEY: excludes ACOX1/ZSD/ABCD1)
  Phytanic acid:  NORMAL — alpha-oxidation via PHYH unaffected
  Plasmalogens:   NORMAL — PTS2/plasmalogen biosynthesis intact (PEX7/GNPAT/AGPS/FAR1 unaffected)
  Pipecolic acid: NORMAL — PTS1 general import intact (excludes ZSD)

CRITICAL BIOCHEMICAL DISTINCTIONS:
  ACOX2 vs AMACR:
    AMACR is UPSTREAM of ACOX2 (racemization step — prerequisite for ACOX2 to act).
    Both cause: pristanic ↑, THCA/DHCA ↑, VLCFA NORMAL, phytanic NORMAL, plasmalogens NORMAL.
    Distinction: GENE SEQUENCING ONLY — biochemical profile identical.
    Clinical: ACOX2 — more hepatic involvement (cholestatic liver disease prominent).
    AMACR: azoospermia males ~55%, focal temporal epilepsy ~60% (higher than ACOX2 ~35%).
  ACOX2 vs HSD17B4 (MFP2/D-bifunctional protein):
    HSD17B4: STEPS 2+3 of ALL substrates (VLCFA + branched-chain + bile acids).
    HSD17B4 deficiency: pristanic ↑ + THCA ↑ + VLCFA ↑ (D-bifunctional handles all).
    ACOX2: VLCFA NORMAL — KEY distinction (HSD17B4 deficiency has VLCFA elevated from backup use).
    HSD17B4: neonatal/severe onset; ACOX2: adult onset — very different age of presentation.
  ACOX2 vs SCP2 (SCPx/sterol carrier protein X):
    SCP2: STEP 4 (thiolytic cleavage) — same biochemical profile as ACOX2 (pristanic ↑, THCA ↑, VLCFA NORMAL).
    Distinction: GENE SEQUENCING ONLY — biochemical profile near-identical.
    Clinical: SCP2 — azoospermia males ~95% (MOST specific SCP2 marker). ACOX2 — hepatic dysfunction more prominent.
  ACOX2 vs ACOX1 (Pseudo-NALD):
    ACOX1: VLCFA SEVERELY ELEVATED — NORMAL in ACOX2. This is THE distinguishing plasma test.
    ACOX1: pristanic NORMAL — ELEVATED in ACOX2. Plasma VLCFA panel separates them immediately.
    Clinical: ACOX1 — neonatal onset, severe. ACOX2 — adult onset (20s-50s).

ACOX2 PROTEIN BIOLOGY:
  Gene ACOX2 (also BRCOX, PRCOX) at 3p25.1 (OMIM Gene: *601641).
  ~672 aa; peroxisomal matrix enzyme. PTS1 targeting signal (C-terminal -SRL).
  Enzyme class: FAD-dependent acyl-CoA oxidase; forms homodimers/homotrimers in peroxisomal matrix.
  Reaction: (2S)-pristanoyl-CoA + FAD → 2-trans-pristanoyl-CoA + FADH₂ (rate-limiting step 1).
  Also catalyses: THCA-CoA + FAD → 2-trans-THCA-CoA; DHCA-CoA + FAD → 2-trans-DHCA-CoA.
  FAD produced is reoxidised by the electron transfer flavoprotein (ETF) system.
  Expression: liver (highest), peroxisomal matrix; also kidney, intestine.
  Bile acid consequence: THCA/DHCA accumulate → impaired primary bile acid synthesis
    (chenodeoxycholate and cholate cannot be formed from C27 precursors → cholestatic liver disease).

EPIDEMIOLOGY:
  Ultra-rare: fewer than 30 cases worldwide described 2026 (in clinical literature).
  OMIM Gene: *601641 (ACOX2).
  AR biallelic LOF; both sexes equally affected.
  First human cases described ~2015-2020 (Ferdinandusse/van Bokhoven group and others).
  Classic adult-onset (20s-50s; range late teens to 60s).
  Locus: 3p25.1.

CLINICAL PHENOTYPE (ACOX2 deficiency):
  Adult-onset (typically 20s-50s) — contrasts with ACOX1/HSD17B4 (neonatal/infantile).
  Hepatic features are prominent (distinguishes ACOX2 from SCP2/AMACR):
  • Cholestatic liver disease / hepatic steatosis — ~70%: elevated liver enzymes (ALT/AST/GGT/ALP),
    hepatomegaly; due to THCA/DHCA accumulation impairing primary bile acid synthesis
  • Peripheral polyneuropathy — ~75%: axonal or mixed; sensorimotor; progressive
  • Cerebellar ataxia — ~65%: gait instability, dysmetria; correlates with posterior white matter changes
  • Epilepsy — ~35%: myoclonic, focal; less prominent than AMACR (~60%) or SCP2 (~40%)
  • Pigmentary retinopathy — ~30%: less common than AMACR (~45%) or SCP2 (~20%)
  • Azoospermia (males) — ~45%: bile acid accumulation disrupts spermatogenesis
  • Sensorineural hearing loss — ~30%
  • Cognitive decline / white matter changes — ~40%: MRI periventricular/posterior WM abnormalities
  • Drug resistance (of those with epilepsy) — ~20%
  Diagnostic delay: typically 5-15 years (ultra-rare; pristanic/THCA not on standard metabolic panels).

ERT/HSCT/GENE THERAPY:
  No ERT (2026) — ACOX2 is a peroxisomal matrix enzyme (PTS1-SRL); not secreted; no delivery route.
  No HSCT — ACOX2 deficiency is NOT inflammatory demyelination; HSCT targets only inflammatory demyelinators.
  No gene therapy approved 2026.

TREATMENT PRINCIPLES:
  LEV first-line for seizures (no peroxisomal or hepatic interactions; safe in adult-onset disease).
  PHT/CBZ: CAN USE — NO adrenal insufficiency in ACOX2 (contrast ABCD1: ABSOLUTE CI).
  VGB: RELATIVE CI — retinopathy ~30%; VGB peripheral VF constriction additive risk (less severe than AMACR).
  VPA: RELATIVE CI — hepatotoxicity risk ELEVATED (cholestatic liver disease ~70% + bile acid burden);
    POLG1 MANDATORY CPIC Grade A before prescribing VPA in any peroxisomal disease.
  Clonazepam: adjunct for myoclonic seizures / cerebellar ataxia (Level C).
  Phytol-restricted diet: Level C — reduces dietary pristanic precursor (phytol → phytanic → pristanic).
  DHA supplementation: Level C (limited evidence, used empirically in other peroxisomal diseases).
  Liver monitoring: ALT/AST/GGT/ALP every 3-6 months; avoid hepatotoxic drugs.
  No Lorenzo's Oil: VLCFA NORMAL — oil targets VLCFA elongation, no mechanism benefit.
  No ERT. No HSCT. Diet and symptomatic management are primary therapies.
"""

import random
random.seed(74)


# ── Overview (KPIs + summary) ─────────────────────────────────────────────────

def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 35,
        "hepatic_pct": 70,
        "polyneuropathy_pct": 75,
        "cerebellar_ataxia_pct": 65,
        "retinopathy_pct": 30,
        "azoospermia_males_pct": 45,
        "snhl_pct": 30,
        "cognitive_decline_pct": 40,
        "drug_resistance_pct": 20,
        "pristanic_elevated_pct": 100,
        "thca_dhca_elevated_pct": 100,
        "vlcfa_normal_pct": 98,
        "plasmalogen_normal_pct": 97,
        "phytanic_normal_pct": 96,
        "pipecolic_normal_pct": 99,
        "adrenal_insufficiency_pct": 0,
        "omim_gene": "601641",
        "locus": "3p25.1",
        "vlcfa_elevated": False,
        "pristanic_elevated": True,
        "inheritance": (
            "Autosomal recessive (AR), biallelic LOF — both sexes equally affected. "
            "Ultra-rare: <30 cases worldwide 2026. Adult onset (20s-50s)."
        ),
        "common_variant": (
            "No single founder mutation; multiple private missense/LOF variants across ACOX2 "
            "described. Fewer than 30 cases confirmed worldwide 2026 (OMIM Gene: *601641). "
            "Pristanic acid measurement + THCA/DHCA + VLCFA normal is the key diagnostic triad."
        ),
        "onset_age": (
            "Adult-onset (typically 20s-50s, range late teens to 60s) — "
            "COMPLETELY DIFFERENT from ACOX1/HSD17B4 (neonatal/infantile); "
            "similar age range to SCP2 and AMACR (other adult branched-chain peroxisomal diseases)"
        ),
        "disease_mechanism": (
            "ACOX2 (~672 aa, PTS1-SRL, peroxisomal matrix, homotrimer) catalyses "
            "STEP 1 of branched-chain peroxisomal beta-oxidation: "
            "(2S)-pristanoyl-CoA + FAD → 2-trans-pristanoyl-CoA + FADH₂. "
            "Also oxidises C27 bile acid CoA esters: THCA-CoA and DHCA-CoA → cannot form "
            "primary bile acids (chenodeoxycholate/cholate) → THCA/DHCA accumulate → "
            "cholestatic liver disease (distinguishing feature from SCP2/AMACR). "
            "Pristanic acid accumulates (step 1 blocked; HSD17B4/SCP2 cannot act without ACOX2 product). "
            "VLCFA (ACOX1 pathway), phytanic (PHYH alpha-oxidation), plasmalogens (PTS2 pathway) "
            "ALL NORMAL — isolated branched-chain + bile acid peroxisomal oxidation defect."
        ),
        "critical_distinctions": [
            "PRISTANIC ACID SEVERELY ELEVATED — step 1 of branched-chain beta-oxidation blocked; KEY biomarker alongside THCA/DHCA",
            "THCA and DHCA ELEVATED — bile acid CoA ester oxidation blocked; hepatic cholestasis results from impaired primary bile acid synthesis",
            "VLCFA C26:0 NORMAL — KEY DISTINCTION from ACOX1 (VLCFA ↑↑), ZSD (all elevated), HSD17B4 (VLCFA + pristanic elevated); plasma VLCFA panel immediate differentiator",
            "PHYTANIC ACID NORMAL — excludes PHYH (Refsum disease); alpha-oxidation via PHYH intact in ACOX2 deficiency",
            "PLASMALOGENS NORMAL — excludes ALL ZSD (PEX1/PEX6) and RCDP types (PEX7/GNPAT/AGPS/FAR1); PTS2 pathway intact",
            "HEPATIC INVOLVEMENT ~70% — MOST PROMINENT DISTINGUISHING FEATURE from SCP2/AMACR (cholestatic liver disease due to THCA/DHCA accumulation; bile acid synthesis impaired)",
            "ACOX2 vs SCP2/AMACR: BIOCHEMICAL PROFILE NEAR-IDENTICAL — gene sequencing MANDATORY to distinguish (same pristanic ↑ + THCA ↑ + VLCFA normal)",
            "VPA RELATIVE CI — ELEVATED RISK: cholestatic liver disease ~70% + bile acid burden = highest hepatotoxicity risk of adult branched-chain peroxisomal group; POLG1 MANDATORY CPIC A",
            "VGB RELATIVE CI — retinopathy ~30% (lower than AMACR ~45%); VGB peripheral VF risk additive; ophthalmology baseline mandatory",
            "LEV first-line. PHT/CBZ CAN USE (no adrenal). Liver enzymes every 3-6 months. No ERT. No HSCT. No Lorenzo's Oil (VLCFA normal).",
            "Azoospermia males ~45% (bile acid disrupts spermatogenesis — less specific than SCP2 ~95%, but check in male patients)",
            "Diagnostic delay typically 5-15 years — pristanic/THCA not on standard NBS or routine metabolic panels; peroxisomal screen required",
        ],
        "nbs_positive_rate": (
            "Not on standard newborn screening (NBS) panels. Plasma pristanic acid + THCA/DHCA "
            "required (not part of routine acylcarnitine or amino acid profiles). "
            "Diagnosis typically after 5-15 years of symptoms. Peroxisomal speciality laboratory mandatory. "
            "Target: plasma pristanic <5 µmol/L (normal); ACOX2 untreated: typically 20-120 µmol/L."
        ),
    }


# ── Breakdown (patients + seizure types + triggers + monitoring + lifecycle + treatments) ──

def get_breakdown():
    # 3 phenotypic classes
    phenotypic_classes = [
        {
            "class": "Hepato-Neurological (Liver + Neuropathy Dominant)",
            "pct": 50,
            "count": 20,
            "description": (
                "Cholestatic liver disease (ALT/AST/GGT elevated, hepatomegaly) as prominent feature "
                "alongside peripheral polyneuropathy. Cerebellar ataxia in ~60%. Seizures in ~30%. "
                "THCA/DHCA accumulation impairs bile acid synthesis — hepatic steatosis/inflammation. "
                "Most common ACOX2 phenotype; distinguishes ACOX2 from SCP2 and AMACR (which have less hepatic involvement). "
                "Azoospermia in males ~50%. Retinopathy less common (~25%)."
            ),
            "seizure_control": "~80% controlled on LEV; ~20% drug resistant; VPA avoided (liver disease)",
        },
        {
            "class": "Neurological-Dominant (Ataxia + Neuropathy, Mild Hepatic)",
            "pct": 35,
            "count": 14,
            "description": (
                "Cerebellar ataxia and peripheral neuropathy dominate; hepatic enzymes mildly elevated only. "
                "White matter changes (posterior periventricular) on MRI. Seizures in ~40% — "
                "myoclonic and focal. Retinopathy in ~35%. SNHL in ~35%. Cognitive decline mild-moderate. "
                "Biochemistry: pristanic + THCA elevated; VLCFA normal — similar to SCP2/AMACR."
            ),
            "seizure_control": "~75% controlled; drug resistance ~25% in seizure subset",
        },
        {
            "class": "Mixed Multisystem (Liver + Neurology + Retina)",
            "pct": 15,
            "count": 6,
            "description": (
                "All three systems involved: significant liver disease, polyneuropathy, cerebellar ataxia, "
                "AND retinopathy. Highest epilepsy rate (~45%). Most drug-resistant (~35% of seizure subset). "
                "Azoospermia in males ~60%. SNHL prominent. Cognitive decline moderate-severe. "
                "Worst overall prognosis; most complex management. Liver transplant considered in some cases."
            ),
            "seizure_control": "~65% controlled; highest drug resistance; LEV + CLZ combination often required",
        },
    ]

    # 40 synthetic patients
    sz_types_pool = [
        "Myoclonic", "Focal temporal", "Focal-BTCS", "Myoclonic-atonic", "Focal parietal"
    ]
    pheno_dist = (
        ["Hepato-Neurological"] * 20 +
        ["Neurological-Dominant"] * 14 +
        ["Mixed Multisystem"] * 6
    )
    patients = []
    for i in range(40):
        sex = random.choice(["Male", "Female"])
        pheno = pheno_dist[i]
        onset = random.randint(19, 54)
        has_seizures = random.random() < 0.35
        sz_type = random.choice(sz_types_pool) if has_seizures else "None"
        drug_resistant = has_seizures and random.random() < 0.20
        has_hepatic = random.random() < 0.70
        has_neuropathy = random.random() < 0.75
        has_ataxia = random.random() < 0.65
        has_retinopathy = random.random() < 0.30
        has_azoospermia = (sex == "Male") and (random.random() < 0.45)
        has_snhl = random.random() < 0.30
        pristanic = random.uniform(20.0, 110.0)  # elevated (normal <5 µmol/L)
        thca = random.uniform(5.0, 45.0)          # elevated (normal <2 µmol/L)
        vlcfa_c26 = random.uniform(0.46, 0.88)    # normal range
        patients.append({
            "id": i + 1,
            "phenotypic_class": pheno,
            "sex": sex,
            "onset_age": onset,
            "primary_seizure": sz_type,
            "drug_resistant": drug_resistant,
            "has_hepatic": has_hepatic,
            "has_neuropathy": has_neuropathy,
            "has_ataxia": has_ataxia,
            "has_retinopathy": has_retinopathy,
            "has_azoospermia": has_azoospermia if sex == "Male" else None,
            "has_snhl": has_snhl,
            "pristanic_umol_L": round(pristanic, 1),
            "thca_umol_L": round(thca, 1),
            "vlcfa_c26_umol_L": round(vlcfa_c26, 2),
        })

    seizure_types = [
        {
            "type": "Myoclonic",
            "pct": 45,
            "notes": "Most common type; correlates with cerebellar ataxia; responds to LEV/CLZ; VPA RELATIVE CI (liver disease)",
        },
        {
            "type": "Focal temporal / frontotemporal",
            "pct": 35,
            "notes": "Similar to AMACR (temporal lobe involvement); white matter lesions contribute; LEV effective",
        },
        {
            "type": "Focal-BTCS (bilateral tonic-clonic spread)",
            "pct": 25,
            "notes": "Secondary generalisation from focal onset; drug resistance in ~20% of this subtype",
        },
        {
            "type": "Myoclonic-atonic",
            "pct": 15,
            "notes": "Atonic component associated with cerebellar dysfunction; CLZ adjunct; sedation risk in ataxic patients",
        },
        {
            "type": "Focal parietal / occipital",
            "pct": 12,
            "notes": "Retinopathy-related cortical changes (~30% have retinopathy); less common than AMACR occipital involvement",
        },
    ]

    triggers = [
        {
            "trigger": "Intercurrent illness / fever",
            "pct": 55,
            "mechanism": "Metabolic stress increases peroxisomal substrate load; hepatic function temporarily worsened; lowers seizure threshold",
        },
        {
            "trigger": "Sleep deprivation",
            "pct": 50,
            "mechanism": "Standard seizure threshold lowering; combined with underlying CNS white matter pathology",
        },
        {
            "trigger": "High-phytol diet (non-compliant)",
            "pct": 40,
            "mechanism": "Dietary phytol → phytanic acid → pristanic acid (alpha-oxidation intact); excessive pristanic substrate worsens ACOX2 block",
        },
        {
            "trigger": "Missed AED dose",
            "pct": 38,
            "mechanism": "Standard AED non-compliance; higher risk in drug-resistant subset (~20%)",
        },
        {
            "trigger": "Hepatic decompensation episode",
            "pct": 30,
            "mechanism": "THCA/DHCA accumulation → acute cholestatic episode → systemic toxicity → lowered seizure threshold; unique to ACOX2",
        },
        {
            "trigger": "Stress / metabolic decompensation",
            "pct": 28,
            "mechanism": "Physical/emotional stress; increased branched-chain substrate load; hepatic capacity reduced",
        },
        {
            "trigger": "Drugs affecting CYP450 / hepatic metabolism",
            "pct": 20,
            "mechanism": "Hepatic enzyme inducers may alter bile acid metabolism; hepatically processed drugs increase liver burden in cholestatic patients",
        },
    ]

    monitoring = [
        {"parameter": "Plasma pristanic acid (µmol/L)", "frequency": "Every 3-6 months", "reason": "Primary branched-chain biomarker; guides dietary compliance; target <20 µmol/L on treatment"},
        {"parameter": "Plasma THCA and DHCA (µmol/L)", "frequency": "Every 3-6 months", "reason": "Bile acid intermediates; hepatic cholestasis correlates with THCA/DHCA burden; distinguish from primary sclerosing cholangitis"},
        {"parameter": "VLCFA panel (C26:0, C24/22 ratio)", "frequency": "Baseline; repeat if diagnosis questioned", "reason": "MUST remain NORMAL; elevation → ACOX1 or ZSD or HSD17B4 co-pathology — immediately changes diagnosis"},
        {"parameter": "Liver function tests (ALT/AST/GGT/ALP/bilirubin)", "frequency": "Every 3 months", "reason": "Cholestatic liver disease ~70%; THCA/DHCA accumulation; guide drug choices (VPA RELATIVE CI)"},
        {"parameter": "RBC plasmalogens", "frequency": "Baseline", "reason": "MUST remain NORMAL; low → ZSD or RCDP co-pathology, not ACOX2"},
        {"parameter": "Plasma phytanic acid", "frequency": "Baseline; every 12 months", "reason": "MUST remain NORMAL; elevated → PHYH/Refsum or alpha-oxidation disorder — different disease"},
        {"parameter": "ERG + visual fields (ophthalmology)", "frequency": "Every 6-12 months", "reason": "Retinopathy ~30%; VGB RELATIVE CI — baseline ERG mandatory before AED choice; track progression"},
        {"parameter": "Nerve conduction studies / EMG", "frequency": "Annually", "reason": "Polyneuropathy monitoring; axonal or mixed pattern; tracks disease progression"},
        {"parameter": "Brain MRI (white matter protocol)", "frequency": "At diagnosis; every 2 years", "reason": "Posterior periventricular white matter changes; cerebellar atrophy monitoring"},
        {"parameter": "Semen analysis / LH/FSH (males)", "frequency": "At diagnosis; annually if planning fertility", "reason": "Azoospermia ~45% in males; bile acid disruption of spermatogenesis; fertility counselling"},
        {"parameter": "Audiogram (pure tone audiometry)", "frequency": "Annually", "reason": "SNHL ~30%; early intervention with hearing aids; cochlear implant planning if progressive"},
    ]

    lifecycle = [
        {
            "stage": "Pre-diagnosis / Diagnostic Delay (typically 5-15 years)",
            "description": "Pristanic/THCA not on standard panels. Hepatic dysfunction often attributed to common causes (fatty liver, alcohol). Neurological features may suggest CMT or SCA. Peroxisomal screen (pristanic + THCA + VLCFA) required for diagnosis.",
        },
        {
            "stage": "Hepatic Manifestation Phase",
            "description": "Elevated liver enzymes (ALT/AST/GGT/ALP); mild to moderate hepatomegaly; cholestatic pattern. Bile acid synthesis impaired (THCA/DHCA cannot be converted to chenodeoxycholate/cholate). VPA and other hepatotoxic drugs should be avoided or used with extreme caution.",
        },
        {
            "stage": "Neurological Onset / Progressive Phase",
            "description": "Peripheral neuropathy (axonal or mixed) progressing over years. Cerebellar ataxia emerges. White matter lesions on MRI. Epilepsy in ~35% — often myoclonic correlating with cerebellar ataxia. AED selection critical: LEV first-line, VPA avoided given liver disease.",
        },
        {
            "stage": "Stable / Treatment-Optimised Phase",
            "description": "Phytol-restricted diet reduces dietary pristanic precursor load. Liver function monitored every 3 months. Seizure control on LEV ± CLZ in ~80%. Retinopathy monitoring every 6-12 months. Fertility counselling for males.",
        },
        {
            "stage": "Advanced / Late Phase",
            "description": "Significant cerebellar ataxia; neuropathy limits mobility. Liver function may deteriorate (liver transplant considered in severe cholestasis). Drug resistance in ~20% of epilepsy subset. Multidisciplinary management: hepatology, neurology, ophthalmology, genetics.",
        },
    ]

    treatments = [
        {
            "name": "LEV (Levetiracetam)",
            "type": "AED",
            "status": "FIRST-LINE for seizures",
            "mechanism": "SV2A modulator; no peroxisomal, hepatic, or bile acid interactions; safe in ACOX2 deficiency",
            "contra": "Behavioural side effects; monitor in cerebellar/cognitive impairment patients",
        },
        {
            "name": "Clonazepam",
            "type": "AED / Adjunct",
            "status": "Level C (myoclonus/ataxia adjunct)",
            "mechanism": "GABA-A modulator; myoclonic seizures + cerebellar ataxia; sedation risk in ataxic patients",
            "contra": "Sedation worsens ataxia if over-dosed; titrate slowly; tolerated by most adult patients",
        },
        {
            "name": "PHT (Phenytoin)",
            "type": "AED",
            "status": "CAN USE (no adrenal CI)",
            "mechanism": "Sodium channel blocker; NO adrenal mechanism — distinct from ABCD1 where ABSOLUTE CI",
            "contra": "Avoid in severe ataxia (cerebellar toxicity risk); CYP450 enzyme induction — may alter bile acid metabolism; monitor LFTs",
        },
        {
            "name": "CBZ (Carbamazepine)",
            "type": "AED",
            "status": "CAN USE (no adrenal CI)",
            "mechanism": "Sodium channel blocker; focal seizures; no adrenal insufficiency in ACOX2",
            "contra": "CYP3A4 inducer — may increase hepatic load; monitor LFTs given cholestatic liver disease; hyponatraemia risk",
        },
        {
            "name": "LTG (Lamotrigine)",
            "type": "AED",
            "status": "Second-line",
            "mechanism": "Sodium channel blocker; focal and myoclonic seizures; slow titration; no peroxisomal interaction",
            "contra": "Slow titration required; rash risk; no specific ACOX2 contraindication",
        },
        {
            "name": "VPA (Valproate)",
            "type": "AED",
            "status": "RELATIVE CI — ELEVATED RISK in ACOX2",
            "mechanism": "VPA hepatotoxicity risk is ELEVATED in ACOX2 deficiency: cholestatic liver disease ~70% + bile acid burden higher than SCP2/AMACR. POLG1 MANDATORY CPIC Grade A before prescribing.",
            "contra": "HEPATOTOXICITY RISK HIGHEST of all adult branched-chain peroxisomal diseases (cholestatic liver disease + bile acid burden). POLG1 genotyping MANDATORY before any VPA use. Avoid as first-line. LEV strongly preferred.",
        },
        {
            "name": "VGB (Vigabatrin)",
            "type": "AED",
            "status": "RELATIVE CI — retinopathy ~30%",
            "mechanism": "GABA transaminase inhibitor; irreversible peripheral VF constriction in 30-40% of all long-term users",
            "contra": "RELATIVE CI: retinopathy ~30% in ACOX2 (less than AMACR ~45% or PHYH ~95%). VGB + retinopathy = additive irreversible VF loss. ERG + formal VF testing MANDATORY at baseline. Avoid if retinopathy present.",
        },
        {
            "name": "Phytol-restricted diet",
            "type": "Disease-modifying (dietary)",
            "status": "Level C (limited evidence — empirical)",
            "mechanism": "Reduces dietary phytol (phytanic acid precursor); phytanic → pristanic via alpha-oxidation (PHYH intact); reducing pristanic substrate load may slow ACOX2 block burden",
            "contra": "Less effective than in PHYH/Refsum (Level A) — phytol restriction only reduces pristanic precursor, does not treat the block itself. Dietitian guidance required; adequate caloric intake essential.",
        },
        {
            "name": "DHA supplementation",
            "type": "Supportive (dietary)",
            "status": "Level C (empirical — used in other peroxisomal diseases)",
            "mechanism": "DHA (docosahexaenoic acid) supplementation may partially compensate for peroxisomal beta-oxidation impairment; limited evidence specific to ACOX2",
            "contra": "No specific ACOX2 contraindication; monitor LFTs (liver burden); empirical use based on other peroxisomal disease data",
        },
        {
            "name": "Ursodeoxycholic acid (UDCA)",
            "type": "Hepatoprotective / Bile acid modulator",
            "status": "Level C (cholestatic liver disease management)",
            "mechanism": "UDCA is a hydrophilic bile acid; partially replaces toxic accumulated THCA/DHCA; hepatoprotective in cholestatic conditions; reduces hepatic inflammation",
            "contra": "Not disease-modifying; symptomatic hepatic benefit only; monitor LFTs; used in other cholestatic diseases (PBC, PSC) — empirical in ACOX2",
        },
        {
            "name": "Lorenzo's Oil",
            "type": "Dietary supplement",
            "status": "NOT APPLICABLE",
            "mechanism": "Erucic + oleic acid → competitive inhibition of VLCFA elongase. VLCFA NORMAL in ACOX2 — no mechanism benefit",
            "contra": "NOT applicable; VLCFA normal in ACOX2 deficiency",
        },
        {
            "name": "ERT (Enzyme replacement therapy)",
            "type": "Biological therapy",
            "status": "NO ERT available (2026)",
            "mechanism": "ACOX2 is a peroxisomal matrix enzyme (PTS1-SRL targeted); not secreted; no systemic delivery feasible",
            "contra": "Not applicable — no approved or in-development ERT for ACOX2 2026",
        },
        {
            "name": "HSCT (Bone marrow transplant)",
            "type": "Cellular therapy",
            "status": "NO HSCT",
            "mechanism": "HSCT targets only inflammatory demyelinating disorders (ABCD1 cerebral form). ACOX2 deficiency is metabolic/non-inflammatory",
            "contra": "Not indicated. Non-inflammatory disease mechanism. HSCT risks outweigh any speculative benefit.",
        },
        {
            "name": "Liver transplant",
            "type": "Surgical intervention",
            "status": "Considered in advanced hepatic failure (case-by-case)",
            "mechanism": "Replaces cholestatic liver — may reduce THCA/DHCA accumulation if liver was primary site of toxicity. Does not correct neurological ACOX2 deficiency.",
            "contra": "Neurological features may continue post-transplant (systemic ACOX2 deficiency). High surgical risk. Only in severe refractory hepatic failure. Experimental in ACOX2.",
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
                "term": "ACOX2 — Acyl-CoA Oxidase 2 (Branched-Chain / Pristanoyl-CoA Oxidase)",
                "definition": (
                    "ACOX2 (gene: ACOX2/BRCOX/PRCOX, 3p25.1, OMIM Gene *601641) encodes a ~672 aa "
                    "FAD-dependent acyl-CoA oxidase that catalyses STEP 1 of branched-chain peroxisomal "
                    "beta-oxidation: (2S)-pristanoyl-CoA + FAD → 2-trans-pristanoyl-CoA + FADH₂. "
                    "Also oxidises C27 bile acid CoA esters (THCA-CoA, DHCA-CoA). "
                    "PTS1-targeted (C-terminal SRL signal); forms homotrimers in peroxisomal matrix. "
                    "Deficiency causes pristanic acid and THCA/DHCA accumulation; cholestatic liver disease "
                    "results from impaired primary bile acid synthesis. Ultra-rare: <30 cases worldwide 2026. AR."
                ),
            },
            {
                "term": "Biochemical Signature",
                "definition": (
                    "Pristanic acid SEVERELY ELEVATED — branched-chain step 1 completely blocked. "
                    "THCA and DHCA ELEVATED — C27 bile acid CoA ester oxidation blocked → impaired primary "
                    "bile acid synthesis → cholestatic liver disease. "
                    "VLCFA (C26:0) NORMAL — ACOX1 straight-chain pathway unaffected (KEY: excludes ACOX1/ZSD/ABCD1). "
                    "Phytanic acid NORMAL — PHYH alpha-oxidation intact. "
                    "Plasmalogens NORMAL — PTS2 pathway intact. Pipecolic acid NORMAL — ZSD excluded. "
                    "Biochemical profile identical to SCP2/AMACR — gene sequencing MANDATORY to distinguish."
                ),
            },
            {
                "term": "ACOX2 Pathway Position — Branched-Chain Peroxisomal Beta-Oxidation",
                "definition": (
                    "Branched-chain fatty acids (pristanic, phytanic) cannot undergo direct beta-oxidation "
                    "without alpha-oxidation (phytanic→pristanic via PHYH) and racemization (AMACR). "
                    "Pathway: PHYH (alpha-ox) → AMACR (racemization prerequisite) → ACOX2 (STEP 1: oxidation) "
                    "→ HSD17B4/MFP2 (steps 2+3: hydration + dehydrogenation) → SCP2/SCPx (step 4: thiolysis). "
                    "ACOX2 deficiency: step 1 blocked → pristanoyl-CoA and THCA/DHCA-CoA cannot progress. "
                    "CONTRAST: ACOX1 handles VLCFA (straight-chain step 1) — entirely different substrate pathway."
                ),
            },
            {
                "term": "ACOX2 vs ACOX1 — THE Plasma Panel Distinction",
                "definition": (
                    "ACOX1 (17q25.1): catalyses step 1 of VLCFA (straight-chain) beta-oxidation. "
                    "ACOX1 deficiency: VLCFA C26:0 SEVERELY ELEVATED; pristanic NORMAL. "
                    "ACOX2 (3p25.1): catalyses step 1 of PRISTANOYL-CoA/THCA-CoA (branched-chain + bile acids). "
                    "ACOX2 deficiency: VLCFA NORMAL; pristanic SEVERELY ELEVATED; THCA/DHCA ELEVATED. "
                    "THE DISTINCTION: plasma VLCFA panel immediately separates ACOX1 from ACOX2. "
                    "VLCFA ↑ → ACOX1 deficiency. VLCFA NORMAL + pristanic ↑ + THCA ↑ → ACOX2 (or SCP2/AMACR). "
                    "Age of onset also differs: ACOX1 neonatal; ACOX2 adult (20s-50s)."
                ),
            },
            {
                "term": "ACOX2 vs SCP2/AMACR — Gene Sequencing Mandatory",
                "definition": (
                    "ACOX2, SCP2, and AMACR deficiencies share near-identical biochemical profiles: "
                    "pristanic acid ↑, THCA ↑, DHCA ↑, VLCFA NORMAL, phytanic NORMAL, plasmalogens NORMAL. "
                    "Cannot be distinguished by plasma metabolites alone — GENE SEQUENCING MANDATORY. "
                    "Clinical clues: ACOX2 — cholestatic liver disease prominent (~70%); SCP2 — azoospermia "
                    "~95% (most specific SCP2 marker); AMACR — focal temporal epilepsy ~60% more prominent. "
                    "Sequencing panel: ACOX2 + SCP2 + AMACR should be ordered together when pristanic + THCA elevated."
                ),
            },
            {
                "term": "THCA/DHCA and Cholestatic Liver Disease",
                "definition": (
                    "THCA (3α,7α,12α-trihydroxycholestanoic acid) and DHCA (3α,7α-dihydroxycholestanoic acid) "
                    "are C27 bile acid precursors that require ACOX2 to enter peroxisomal beta-oxidation. "
                    "Without ACOX2, THCA/DHCA accumulate → primary bile acids (chenodeoxycholate, cholate) "
                    "cannot be synthesized → impaired bile flow → cholestatic liver disease. "
                    "This hepatic involvement (~70%) is the MOST DISTINGUISHING CLINICAL FEATURE of ACOX2 "
                    "compared to SCP2 (~10% hepatic) and AMACR (~20% hepatic). "
                    "Monitor LFTs every 3 months. UDCA may be used empirically. VPA RELATIVE CI (elevated hepatotoxicity risk)."
                ),
            },
            {
                "term": "VPA — RELATIVE CI with ELEVATED RISK in ACOX2",
                "definition": (
                    "VPA hepatotoxicity risk is HIGHEST in ACOX2 deficiency among adult branched-chain "
                    "peroxisomal diseases: cholestatic liver disease ~70% (baseline) + bile acid burden + VPA hepatotoxicity = "
                    "significant additive hepatic risk. RELATIVE CI (not absolute — may be used if essential). "
                    "POLG1 genotyping MANDATORY (CPIC Grade A) before prescribing VPA in any peroxisomal/mitochondrial disorder. "
                    "POLG1 carriers: VPA ABSOLUTE CI (mitochondrial hepatotoxicity). "
                    "LEV is strongly preferred as first-line. Avoid VPA unless LEV/LTG/CLZ failed and no hepatic contraindication."
                ),
            },
            {
                "term": "VGB — RELATIVE CI in ACOX2",
                "definition": (
                    "VGB causes irreversible peripheral visual field constriction (30-40% of all long-term users). "
                    "In ACOX2 deficiency: retinopathy ~30% — VGB + retinopathy = additive irreversible VF loss. "
                    "RELATIVE CI (not absolute — lower retinopathy rate than AMACR ~45% or PHYH ~95%). "
                    "ERG + formal VF MANDATORY at baseline before AED selection. "
                    "If retinopathy present on baseline ERG: VGB contraindicated. If ERG normal: use with extreme caution + 6-monthly monitoring. "
                    "CONTRAST: PHYH/ARD (RP ~95%): ABSOLUTE CI. ACOX2 (retinopathy ~30%): RELATIVE CI."
                ),
            },
            {
                "term": "PHT/CBZ — CAN USE (No Adrenal CI)",
                "definition": (
                    "ACOX2 deficiency does NOT cause adrenal insufficiency. "
                    "PHT/CBZ are CAN USE for focal seizures — NO adrenal mechanism CI. "
                    "CRITICAL CONTRAST: ABCD1 (X-ALD): PHT/CBZ = ABSOLUTE CI "
                    "(CYP450 induction → cortisol degradation → adrenal crisis in adrenal-insufficient patients). "
                    "In ACOX2: adrenal glands unaffected; PHT/CBZ safe. "
                    "CAUTION: CYP450 enzyme induction by PHT/CBZ may alter hepatic bile acid metabolism — "
                    "monitor LFTs in cholestatic patients when initiating enzyme-inducing AEDs."
                ),
            },
            {
                "term": "Azoospermia (Males ~45%)",
                "definition": (
                    "Pristanic acid and bile acid accumulation disrupts spermatogenesis — "
                    "azoospermia or severe oligospermia in ~45% of affected males. "
                    "Less specific than SCP2 (~95% azoospermia — most specific SCP2 marker). "
                    "Semen analysis at diagnosis for males; LH/FSH; fertility counselling. "
                    "Practical: azoospermia in adult male with pristanic ↑ + THCA ↑ → ACOX2 or SCP2 "
                    "(gene sequencing mandatory); SCP2 more likely if azoospermia is the dominant feature."
                ),
            },
            {
                "term": "Diagnostic Workup",
                "definition": (
                    "Step 1: Adult with cholestatic liver disease + peripheral neuropathy ± ataxia ± seizures → "
                    "peroxisomal screen (plasma pristanic + THCA/DHCA + VLCFA + phytanic + plasmalogens). "
                    "Step 2: Pristanic ↑ + THCA ↑ + VLCFA NORMAL → branched-chain peroxisomal disease "
                    "(ACOX2 vs SCP2 vs AMACR). "
                    "Step 3: RBC plasmalogens → MUST be NORMAL (low → ZSD/RCDP, not ACOX2). "
                    "Step 4: VLCFA panel NORMAL (elevated → ACOX1/ZSD/HSD17B4 — different disease). "
                    "Step 5: Plasma phytanic → NORMAL (elevated → PHYH/Refsum — different disease). "
                    "Step 6: GENE SEQUENCING PANEL (ACOX2 + SCP2 + AMACR) — cannot distinguish biochemically. "
                    "Step 7: Baseline ERG + VF before any AED (VGB decision). "
                    "Step 8: LFTs + abdominal ultrasound (hepatic involvement). "
                    "Step 9: Semen analysis in males. "
                    "Step 10: POLG1 genotyping before VPA."
                ),
            },
            {
                "term": "UDCA in Cholestatic Liver Disease",
                "definition": (
                    "Ursodeoxycholic acid (UDCA) is a hydrophilic bile acid that partially displaces "
                    "toxic THCA/DHCA from the bile acid pool. Level C evidence in ACOX2 (empirical). "
                    "Used in other cholestatic conditions (primary biliary cholangitis, PSC). "
                    "May reduce hepatic inflammation and improve liver enzyme profile. "
                    "Monitor LFTs every 3 months. Not disease-modifying (does not restore ACOX2 activity). "
                    "Combination with phytol-restricted diet may reduce overall substrate load."
                ),
            },
            {
                "term": "No ERT / No HSCT / No Lorenzo's Oil",
                "definition": (
                    "No ERT: ACOX2 is a peroxisomal matrix enzyme (PTS1-SRL targeted); not secreted; "
                    "no receptor for systemic delivery — cannot be delivered as intravenous enzyme. "
                    "No HSCT: ACOX2 deficiency is metabolic/non-inflammatory; HSCT only targets "
                    "inflammatory demyelination (ABCD1 cerebral form). "
                    "No Lorenzo's Oil: erucic/oleic acid targets VLCFA elongase — VLCFA NORMAL in ACOX2; "
                    "no mechanism for benefit. NOT applicable."
                ),
            },
            {
                "term": "Phytol-Restricted Diet (Level C) vs PHYH Gold Standard (Level A)",
                "definition": (
                    "In PHYH/Refsum: phytol-restricted diet is GOLD STANDARD (Level A) — directly reduces "
                    "phytanic acid (the substrate for PHYH). "
                    "In ACOX2: diet is Level C (empirical) — phytol → phytanic → pristanic via alpha-oxidation "
                    "(PHYH/alpha-oxidation intact in ACOX2). Restricting phytol reduces pristanic precursor load "
                    "but does not block the enzyme step. Less impactful than in Refsum disease. "
                    "Still recommended to reduce overall branched-chain substrate burden. Dietitian guidance required."
                ),
            },
            {
                "term": "White Matter Changes on MRI",
                "definition": (
                    "Posterior periventricular white matter signal abnormalities on T2/FLAIR MRI in ~40% of ACOX2 patients. "
                    "Cerebellar atrophy correlates with ataxia severity. "
                    "Mechanism: pristanic acid and THCA accumulation in CNS membranes disrupts myelin integrity. "
                    "Pattern differs from ABCD1 (parieto-occipital + genu/splenium enhancement, garland pattern) "
                    "and from ACOX1 (posterior predominant with leukodystrophy). "
                    "MRI at diagnosis; repeat every 2 years or if neurological deterioration."
                ),
            },
        ],
        "diagnostic_algorithm": [
            "Adult with unexplained cholestatic liver disease + peripheral neuropathy ± cerebellar ataxia ± myoclonic seizures → peroxisomal screen",
            "Plasma pristanic acid (target <5 µmol/L; ACOX2 untreated: 20-120 µmol/L) — ELEVATED",
            "Plasma THCA and DHCA (bile acid intermediates) — ELEVATED in ACOX2",
            "Plasma VLCFA panel (C26:0, C24:0/C22:0 ratio) → MUST be NORMAL in ACOX2 (elevated → ACOX1 / ZSD / HSD17B4)",
            "Plasma phytanic acid → MUST be NORMAL (elevated → PHYH/Refsum — different disease)",
            "RBC plasmalogens → MUST be NORMAL (low → ZSD or RCDP — different disease class)",
            "Plasma pipecolic acid → NORMAL (elevated → ZSD — PEX1/PEX6 deficiency)",
            "Gene sequencing panel: ACOX2 + SCP2 + AMACR — biochemical profile identical; sequencing mandatory",
            "Baseline ERG + formal visual field testing MANDATORY before AED selection (VGB decision)",
            "LFTs (ALT/AST/GGT/ALP/bilirubin) + abdominal ultrasound — hepatic involvement assessment",
            "Semen analysis + LH/FSH in males (azoospermia ~45%); fertility counselling",
            "POLG1 genotyping (CPIC Grade A) before any VPA prescription",
            "Brain MRI (white matter + cerebellar protocol) at diagnosis",
        ],
        "pharmacological_distinctions": [
            {"drug": "LEV", "status": "FIRST-LINE", "reason": "No peroxisomal or hepatic interactions; safe in adult-onset disease; myoclonus-effective; preferred over VPA given liver disease"},
            {"drug": "Clonazepam", "status": "Level C (myoclonus/ataxia)", "reason": "Cerebellar ataxia + myoclonic seizures; sedation risk in ataxic patients; titrate slowly"},
            {"drug": "PHT", "status": "CAN USE (no adrenal CI)", "reason": "No adrenal insufficiency in ACOX2; CYP450 induction — monitor LFTs in cholestatic disease; avoid in severe ataxia"},
            {"drug": "CBZ", "status": "CAN USE (no adrenal CI)", "reason": "No adrenal insufficiency; focal seizures; CYP3A4 induction — monitor LFTs; hyponatraemia risk"},
            {"drug": "LTG", "status": "Second-line", "reason": "Focal/myoclonic seizures; no peroxisomal interaction; slow titration; rash risk"},
            {"drug": "VPA", "status": "RELATIVE CI — ELEVATED RISK", "reason": "HIGHEST hepatotoxicity risk of adult branched-chain peroxisomal diseases (cholestatic liver ~70% + bile acid burden). POLG1 MANDATORY CPIC A. Avoid first-line; LEV strongly preferred"},
            {"drug": "VGB", "status": "RELATIVE CI — retinopathy ~30%", "reason": "Retinopathy ~30%; VGB + retinopathy = additive irreversible VF loss. ERG MANDATORY baseline. If retinopathy present: contraindicated. Less severe than PHYH (RP ~95% = ABSOLUTE CI)"},
            {"drug": "Phytol-restricted diet", "status": "Level C (empirical)", "reason": "Reduces dietary pristanic precursor; less effective than Refsum (Level A) — does not treat the enzyme defect; dietitian guidance essential"},
            {"drug": "DHA", "status": "Level C (empirical)", "reason": "Used empirically in other peroxisomal diseases; limited ACOX2-specific evidence; no specific contraindication"},
            {"drug": "UDCA", "status": "Level C (cholestatic support)", "reason": "Hydrophilic bile acid; reduces hepatic toxicity of THCA/DHCA; empirical in ACOX2; monitor LFTs"},
            {"drug": "Lorenzo's Oil", "status": "NOT APPLICABLE", "reason": "VLCFA normal in ACOX2; oil targets VLCFA elongase — no mechanism benefit"},
            {"drug": "ERT", "status": "NO ERT available", "reason": "PTS1-SRL matrix enzyme; not secreted; no systemic delivery feasible 2026"},
            {"drug": "HSCT", "status": "NO HSCT", "reason": "Non-inflammatory metabolic disease; HSCT only for inflammatory demyelination (ABCD1 cerebral form)"},
            {"drug": "Liver transplant", "status": "Case-by-case (advanced hepatic failure)", "reason": "May reduce THCA/DHCA liver burden; neurological ACOX2 deficiency continues post-transplant; experimental"},
        ],
        "differential_diagnoses": [
            {
                "disease": "SCP2 Deficiency (SCPx / Sterol Carrier Protein X)",
                "key_distinction": "BIOCHEMICAL PROFILE NEAR-IDENTICAL — gene sequencing ONLY distinction. SCP2: azoospermia males ~95% (most specific SCP2 marker); movement disorder (dystonia/chorea) ~90%; less hepatic than ACOX2. ACOX2: cholestatic liver disease ~70% (dominant distinguishing feature). Both: pristanic ↑, THCA ↑, VLCFA NORMAL.",
                "shared_features": "Pristanic elevated, THCA/DHCA elevated, VLCFA NORMAL, adult onset, AR, neuropathy, ataxia",
            },
            {
                "disease": "AMACR Deficiency (Alpha-methylacyl-CoA Racemase)",
                "key_distinction": "BIOCHEMICAL PROFILE NEAR-IDENTICAL — gene sequencing MANDATORY. AMACR is UPSTREAM of ACOX2 (racemization prerequisite). AMACR: focal temporal epilepsy ~60% (most prominent), azoospermia ~55%. ACOX2: cholestatic liver disease ~70% (dominant). Both: pristanic ↑, THCA ↑, VLCFA NORMAL. No prostate cancer allele confusion (p.Ser113Leu) in ACOX2.",
                "shared_features": "Pristanic elevated, THCA elevated, VLCFA NORMAL, adult onset 20s-60s, AR, neuropathy",
            },
            {
                "disease": "ACOX1 Deficiency (Pseudo-Neonatal-Adrenoleukodystrophy)",
                "key_distinction": "VLCFA SEVERELY ELEVATED in ACOX1 — NORMAL in ACOX2. This single test immediately separates them. ACOX1: neonatal/infantile onset, leukodystrophy, retinal degeneration ~85%, VGB RELATIVE CI. ACOX2: adult onset, cholestatic liver, pristanic ↑ (ACOX1 pristanic NORMAL).",
                "shared_features": "Peroxisomal disease, AR, epilepsy, neuropathy, ataxia",
            },
            {
                "disease": "HSD17B4 Deficiency (D-Bifunctional Protein / MFP2)",
                "key_distinction": "HSD17B4: handles ALL substrates (steps 2+3 for VLCFA + branched-chain). VLCFA ELEVATED in HSD17B4 — NORMAL in ACOX2 (KEY distinction). HSD17B4: neonatal/severe onset (Type I); very different age presentation. ACOX2: adult onset. Pristanic ↑ and THCA ↑ in both, but VLCFA separates them.",
                "shared_features": "Pristanic elevated, THCA elevated, AR, peroxisomal disease",
            },
            {
                "disease": "ZSD (ZSD PEX1/PEX6 Deficiency)",
                "key_distinction": "ZSD: ALL peroxisomal functions impaired — VLCFA ↑ + plasmalogens ↓ + phytanic ↑ + pristanic ↑ + pipecolic ↑ (ALL abnormal). ACOX2: ONLY pristanic + THCA elevated; ALL other markers NORMAL. ZSD: neonatal (Zellweger syndrome) or infantile (NALD/IRD). ACOX2: adult onset.",
                "shared_features": "Pristanic elevated, peroxisomal disease, AR, epilepsy, neuropathy",
            },
            {
                "disease": "PHYH Deficiency (Adult Refsum Disease)",
                "key_distinction": "PHYH: phytanic acid SEVERELY ELEVATED — NORMAL in ACOX2. Phytanic is the defining biomarker of Refsum. ACOX2: pristanic ↑ — NORMAL in PHYH. THCA: NORMAL in PHYH — ELEVATED in ACOX2. One plasma test (phytanic vs pristanic) distinguishes these diseases.",
                "shared_features": "Adult onset, VLCFA NORMAL, peroxisomal disease, AR, peripheral neuropathy",
            },
            {
                "disease": "Primary Sclerosing Cholangitis (PSC) / Primary Biliary Cholangitis (PBC)",
                "key_distinction": "Cholestatic liver disease common to ACOX2 (~70%). PSC/PBC: elevated alkaline phosphatase/bilirubin; anti-mitochondrial Ab (PBC); imaging shows biliary strictures (PSC). ACOX2: normal biliary imaging; pristanic ↑ + THCA ↑. Peroxisomal screen mandatory in young adults with unexplained cholestasis + neurological features.",
                "shared_features": "Cholestatic liver disease, elevated ALP/GGT, hepatomegaly",
            },
        ],
    }
