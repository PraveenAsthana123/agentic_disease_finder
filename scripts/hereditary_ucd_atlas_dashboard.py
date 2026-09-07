#!/usr/bin/env python3
"""Hereditary-UCD-Atlas — Complete 8-Gene Hereditary Urea Cycle Disorder Atlas
OTC     (ornithine transcarbamylase; 354 aa; Xp21.1; XLR;
         OTC deficiency — most common UCD ~50% of all cases;
         neonatal hyperammonemia crisis in males; female carriers variable;
         protein restriction + citrulline/arginine + ammonia scavengers;
         liver transplant curative; valproate ABSOLUTELY CI; seed SEED_BASE+0) .
ASS1    (argininosuccinate synthase 1; 412 aa; 9q34.11; AR;
         Citrullinemia type I (CTLN1); plasma citrulline >1000 µmol/L PATHOGNOMONIC;
         arginine ESSENTIAL supplement (deficient in ASS1); liver transplant;
         citrulline elevates even on dried blood spot — NBS detectable; seed SEED_BASE+1) .
ASL     (argininosuccinate lyase; 464 aa; 7cen-q11.2; AR;
         Argininosuccinic aciduria; argininosuccinate in urine PATHOGNOMONIC;
         trichorrhexis nodosa bamboo hair PATHOGNOMONIC;
         neurotoxicity independent of ammonia — unique among UCDs; seed SEED_BASE+2) .
CPS1    (carbamoyl phosphate synthetase 1; 1500 aa; 2q35; AR;
         CPS1 deficiency; citrulline absent or trace (<5 µmol/L) — key DDx from OTC;
         N-carbamylglutamate (NCG/carglumic acid) treats NAGS deficiency NOT CPS1;
         citrulline supplementation is KEY empiric treatment; seed SEED_BASE+3) .
ARG1    (arginase 1; 322 aa; 6q23.2; AR;
         Arginemia; plasma arginine >400 µmol/L PATHOGNOMONIC;
         spastic diplegia/quadriplegia — NOT classic neonatal hyperammonemia;
         ammonia only mildly elevated unlike other UCDs; dietary protein restriction central;
         seed SEED_BASE+4) .
NAGS    (N-acetylglutamate synthase; 534 aa; 17q21.31; AR;
         NAGS deficiency — mimics CPS1 biochemically (absent citrulline);
         ONLY UCD with specific antidote: N-carbamylglutamate (carglumic acid) DRAMATIC response;
         carglumic acid activates CPS1 directly; small doses can normalise ammonia;
         seed SEED_BASE+5) .
SLC25A15 (ornithine carrier 1 ORC1; 301 aa; 13q14.11; AR;
         HHH syndrome — hyperornithinemia-hyperammonemia-homocitrullinuria;
         homocitrulline in urine PATHOGNOMONIC; plasma ornithine elevated;
         progressive spastic paraplegia; ornithine supplementation paradoxical;
         seed SEED_BASE+6) .
SLC25A13 (citrin aspartate-glutamate carrier 2; 675 aa; 7q21.3; AR;
         Citrin deficiency — NICCD neonatal cholestasis then CTLN2 adult citrullinemia;
         carbohydrate-rich diet is HARMFUL — unique reversal of dietary instinct;
         high-protein/fat diet BENEFICIAL — aversion to sweet foods self-protective;
         common in East Asians Japan 1:17000; liver transplant for CTLN2; seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 1798-1805)
"""

import random

SEED_BASE = 1798

UCD_GENES = [
    # -- OTC -- X-linked OTC Deficiency -----------------------------------
    {
        "gene": "OTC",
        "protein": (
            "OTC -- Xp21.1 XLR -- Ornithine-Transcarbamylase-354aa -- "
            "OTC-Deficiency-Most-Common-UCD-~50pct -- "
            "Neonatal-Hyperammonemia-Males-LIFE-THREATENING -- "
            "Female-Carriers-Variable-Late-Onset-Hyperammonemia -- "
            "Protein-Restriction-Citrulline-Supplementation-Ammonia-Scavengers -- "
            "Valproate-ABSOLUTELY-CONTRAINDICATED -- "
            "Liver-Transplant-CURATIVE"
        ),
        "alias": (
            "OTC (ornithine transcarbamylase); OMIM gene 300461; "
            "OTC deficiency (OTCD) OMIM 311250. "
            "Xp21.1; 354 aa; ~40 kDa; X-linked recessive — affects males fully; females variably. "
            "FUNCTION: OTC catalyses the condensation of carbamoyl phosphate and ornithine "
            "to form citrulline in the mitochondrial matrix — the second step of the urea cycle. "
            "UREA CYCLE OVERVIEW: ammonia (from protein catabolism) → carbamoyl phosphate (CPS1) → "
            "+ ornithine → citrulline (OTC) → + aspartate → argininosuccinate (ASS1) → "
            "arginine + fumarate (ASL) → urea + ornithine (ARG1). "
            "OTC LOF → citrulline not formed → ammonia/carbamoyl phosphate accumulate → "
            "hyperammonemia + elevated orotic acid (carbamoyl phosphate overflows into pyrimidine synthesis). "
            "CLINICAL PHENOTYPE: "
            "Males (hemizygous): severe neonatal onset — most common presentation; "
            "after protein intake (breast milk/formula), ammonia rises dramatically; "
            "symptoms at 24-72 hours: poor feeding, lethargy, vomiting, hyperventilation; "
            "escalates to cerebral oedema, coma, death if untreated; "
            "neonatal blood ammonia >150 µmol/L in full-term neonate = EMERGENCY; "
            "any level >300-500 µmol/L with encephalopathy → immediate dialysis. "
            "Female carriers: wide phenotypic spectrum from asymptomatic to lethal; "
            "X-inactivation skewing determines severity; "
            "carrier females can present acutely with encephalopathy postpartum, post-operatively, "
            "during high-protein intake, or with infections (catabolic stress); "
            "classical story: woman hospitalised for 'psychiatric' episode (encephalopathy) → plasma ammonia; "
            "late-onset males also exist (partial OTC activity via hypomorphic variants). "
            "BIOCHEMICAL DIAGNOSIS: "
            "Plasma ammonia: elevated (primary finding); "
            "Plasma amino acids: glutamine elevated; citrulline LOW/absent (<5 µmol/L); "
            "Urine orotic acid: ELEVATED — cardinal distinguishing feature from CPS1/NAGS deficiency "
            "(citrulline absent in all three; orotic acid elevated ONLY in OTC); "
            "Plasma ornithine: normal or low. "
            "ACUTE MANAGEMENT: "
            "Stop protein intake immediately (max 24 hours, then reintroduce gradually); "
            "High glucose/lipid infusion to prevent catabolism; "
            "Ammonia scavengers: sodium benzoate (conjugates glycine → hippurate; renal excretion) + "
            "sodium phenylacetate (conjugates glutamine → phenylacetylglutamine); "
            "both benzoate + phenylacetate available as combined Ammonul IV formulation; "
            "Arginine/citrulline supplementation (since citrulline cannot be synthesised); "
            "Haemodialysis for ammonia >300 µmol/L with encephalopathy (fastest ammonia removal); "
            "NEVER use peritoneal dialysis — inadequate ammonia clearance. "
            "CHRONIC MANAGEMENT: "
            "Protein restriction (natural protein 0.7-1.5 g/kg/day adjusted for tolerance); "
            "Essential amino acid supplement; "
            "Citrulline 100-200 mg/kg/day (bypasses OTC block); "
            "Sodium benzoate 250-500 mg/kg/day prophylactically; "
            "VALPROATE (VPA) ABSOLUTELY CONTRAINDICATED: VPA inhibits OTC directly (independent of OTC mutation) "
            "→ precipitates acute hyperammonemia in ANY OTC deficiency patient; "
            "VPA also inhibits CPS1 and causes carnitine depletion; "
            "ALTERNATIVE anti-epileptics: levetiracetam, lamotrigine, lacosamide; "
            "AVOID high-dose steroids (catabolic); "
            "Intercurrent illness plan: sick-day protocol including temporary protein restriction; "
            "emergency vomiting letter for rapid hospital assessment. "
            "LIVER TRANSPLANT: "
            "Corrects UCD enzymatic defect in liver (OTC expressed almost exclusively in liver); "
            "Does NOT reverse neurological damage from prior hyperammonemia; "
            "Timing: preferably before severe encephalopathy / permanent brain injury; "
            "Typically 3-6 months of age in severe neonatal onset males; "
            "Post-transplant: protein restriction can be liberalised but not fully abolished; "
            "transplant does NOT prevent hyperammonemia from extra-hepatic OTC expression in muscle."
        ),
        "locus": "Xp21.1",
        "aa": 354,
        "kDa": 40,
        "omim_gene": 300461,
        "omim_disease": 311250,
        "inheritance": "XLR",
        "gene_class": "mitochondrial enzyme — urea cycle step 2",
        "key_alerts": [
            "OTC-VALPROATE-ABSOLUTELY-CI: VPA directly inhibits OTC — precipitates acute hyperammonemia in any OTC patient regardless of variant; NEVER prescribe VPA for OTC deficiency; use levetiracetam/lamotrigine/lacosamide instead",
            "OTC-OROTIC-ACID-ELEVATED-KEY-DDX: Urine orotic acid distinguishes OTC deficiency from CPS1/NAGS deficiency (both have absent citrulline); orotic acid elevated ONLY in OTC (carbamoyl phosphate overflows into pyrimidine pathway)",
            "OTC-FEMALE-CARRIERS-CAN-PRESENT-ACUTELY: carrier females have variable phenotype; acute hyperammonemia episodes triggered by catabolic stress (postpartum, surgery, fasting, illness); must check ammonia in any woman with unexplained encephalopathy — ask about family history of male infant death",
            "OTC-HAEMODIALYSIS-FOR-SEVERE-HYPERAMMONEMIA: HD is the fastest ammonia removal method; peritoneal dialysis is INADEQUATE; for ammonia >300 µmol/L with encephalopathy, do NOT delay — direct to ICU with HD",
            "OTC-LIVER-TRANSPLANT-DOES-NOT-REVERSE-BRAIN-DAMAGE: transplant corrects enzyme defect going forward; past hyperammonemic insults are irreversible; transplant timing must precede severe neurological injury",
        ],
        "etiologies": {
            "severe_neonatal_male": 55,
            "late_onset_male": 20,
            "carrier_female_symptomatic": 15,
            "carrier_female_asymptomatic": 10,
        },
        "stats": {
            "disease": "OTC deficiency (OTCD)",
            "incidence": "1:14,000 live births",
            "fraction_ucds": "~50% of all UCDs",
            "neonatal_crisis_male_pct": 85,
            "median_ammonia_crisis_umol_L": 680,
            "citrulline_absent_pct": 96,
            "orotic_acid_elevated_pct": 98,
            "valproate_contraindicated_pct": 100,
            "liver_transplant_5yr_survival_pct": 90,
        },
        "dx_delay_distribution": {
            "<24h": 30, "24-72h": 45, "4-7 days": 15, ">1 week": 10
        },
    },
    # -- ASS1 -- Citrullinemia Type I (CTLN1) ------------------------------
    {
        "gene": "ASS1",
        "protein": (
            "ASS1 -- 9q34.11 AR -- Argininosuccinate-Synthase-1-412aa -- "
            "Citrullinemia-Type-I-CTLN1 -- "
            "Plasma-Citrulline->1000µmol/L-PATHOGNOMONIC -- "
            "Arginine-ESSENTIAL-Supplement-Deficient-in-ASS1 -- "
            "NBS-Detectable-Citrulline-on-DBS -- "
            "Ammonia-Scavengers-Liver-Transplant"
        ),
        "alias": (
            "ASS1 (argininosuccinate synthase 1); OMIM gene 603470; "
            "Citrullinemia type I (CTLN1; classic citrullinemia) OMIM 215700. "
            "9q34.11; 412 aa; ~47 kDa; autosomal recessive. "
            "FUNCTION: ASS1 catalyses the condensation of citrulline + aspartate → argininosuccinate "
            "(the third step of the urea cycle, occurring in cytoplasm). "
            "ASS1 LOF → citrulline accumulates dramatically → plasma citrulline >1000 µmol/L "
            "(normal <50 µmol/L); argininosuccinate absent; arginine deficient "
            "(arginine normally synthesised via ASS1→ASL→ARG1 pathway). "
            "CITRULLINE ELEVATION PATTERN: "
            "OTC: citrulline absent; "
            "CPS1/NAGS: citrulline absent or trace; "
            "ASS1/CTLN1: citrulline >1000 µmol/L (markedly, diagnostically elevated); "
            "ASL: citrulline moderately elevated; "
            "SLC25A13/CTLN2: citrulline moderately to markedly elevated. "
            "CLINICAL PHENOTYPE: "
            "Neonatal form (classic CTLN1): similar to OTC — acute hyperammonemia in neonatal period; "
            "poor feeding, encephalopathy, coma within days; "
            "high plasma citrulline on NBS (DBS) allows early detection; "
            "Milder forms: episodic hyperammonemia triggered by catabolic stress; "
            "ARGININE ESSENTIAL: arginine is normally produced endogenously via the urea cycle; "
            "in ASS1 deficiency, this downstream synthesis is blocked → arginine becomes "
            "conditionally essential → must be supplemented; "
            "arginine supplementation also helps drive the cycle (alternative pathways). "
            "BIOCHEMICAL DIAGNOSIS: "
            "Plasma amino acids: citrulline >1000 µmol/L (pathognomonic); "
            "Urine organic acids: no argininosuccinate (not made); "
            "Ammonia: elevated acutely; "
            "NBS: citrulline markedly elevated on DBS — highly screened. "
            "MANAGEMENT: "
            "Acute crisis: same principles as OTC — stop protein, glucose/lipid, ammonia scavengers, dialysis; "
            "Arginine supplementation: 400-700 mg/kg/day (essential, not optional); "
            "Sodium benzoate + sodium phenylacetate; "
            "Protein restriction + essential amino acid formula; "
            "Liver transplant: corrects enzymatic defect; post-transplant citrulline still elevated "
            "(extra-hepatic ASS1 deficiency) but ammonia normalises; "
            "post-transplant protein restriction can be liberalised significantly."
        ),
        "locus": "9q34.11",
        "aa": 412,
        "kDa": 47,
        "omim_gene": 603470,
        "omim_disease": 215700,
        "inheritance": "AR",
        "gene_class": "cytoplasmic enzyme — urea cycle step 3",
        "key_alerts": [
            "ASS1-CITRULLINE->1000-PATHOGNOMONIC: plasma citrulline >1000 µmol/L is the biochemical hallmark of CTLN1; no other UCD produces this level of citrulline accumulation; detectable on dried blood spot NBS",
            "ASS1-ARGININE-ESSENTIAL-SUPPLEMENT: arginine is NOT synthesised in ASS1 deficiency — it is conditionally essential; failure to supplement arginine causes arginine deficiency on top of hyperammonemia; always prescribe arginine alongside ammonia scavengers",
            "ASS1-NBS-DETECTABLE: citrulline elevation on newborn screen DBS is the most reliable NBS marker for CTLN1; allows pre-symptomatic diagnosis; act immediately on elevated citrulline DBS result",
            "ASS1-POST-TRANSPLANT-CITRULLINE-STILL-ELEVATED: liver transplant normalises ammonia but citrulline remains elevated (extra-hepatic ASS1 deficiency persists); do not interpret residual citrulline elevation as transplant failure",
        ],
        "etiologies": {
            "classic_neonatal_onset": 60,
            "late_onset_episodic": 30,
            "asymptomatic_nbs_detected": 10,
        },
        "stats": {
            "disease": "Citrullinemia type I (CTLN1)",
            "incidence": "1:57,000 live births",
            "plasma_citrulline_median_umol_L": 1200,
            "nbs_detected_pct": 75,
            "arginine_supplement_required_pct": 100,
            "liver_transplant_performed_pct": 35,
        },
        "dx_delay_distribution": {
            "<24h": 25, "24-72h": 40, "4-7 days": 20, ">1 week": 15
        },
    },
    # -- ASL -- Argininosuccinic Aciduria ----------------------------------
    {
        "gene": "ASL",
        "protein": (
            "ASL -- 7cen-q11.2 AR -- Argininosuccinate-Lyase-464aa -- "
            "Argininosuccinic-Aciduria -- "
            "Argininosuccinate-in-Urine-PATHOGNOMONIC -- "
            "Trichorrhexis-Nodosa-Bamboo-Hair-PATHOGNOMONIC -- "
            "Neurotoxicity-INDEPENDENT-of-Ammonia-Unique-UCD -- "
            "Arginine-Supplementation-KEY-Treatment"
        ),
        "alias": (
            "ASL (argininosuccinate lyase); OMIM gene 608310; "
            "Argininosuccinic aciduria (ASA; argininosuccinicaciduria) OMIM 207900. "
            "7cen-q11.2; 464 aa; ~52 kDa; autosomal recessive. "
            "FUNCTION: ASL catalyses the cleavage of argininosuccinate → arginine + fumarate "
            "(the fourth step of the urea cycle, cytoplasmic). "
            "ASL also participates in the arginine-citrulline cycle in non-hepatic tissues, "
            "including brain endothelium (NO synthesis via eNOS requires locally produced arginine). "
            "ASL LOF → argininosuccinate accumulates → excreted in large amounts in urine; "
            "arginine deficiency (downstream of the block). "
            "PATHOGNOMONIC FINDINGS: "
            "(1) Argininosuccinate in urine: present at very high levels — can be detected on "
            "urine organic acids or by dedicated amino acid analysis; "
            "also slightly elevated in plasma; "
            "(2) Trichorrhexis nodosa: bamboo-hair appearance on light microscopy — "
            "hair shaft fractures at nodes; not present in all patients but highly specific; "
            "cause: local arginine deficiency in hair follicles (NO deficiency) → "
            "structural hair defect; "
            "if present, it is nearly diagnostic of ASA. "
            "UNIQUE NEUROTOXICITY — AMMONIA-INDEPENDENT: "
            "Unlike all other UCDs where brain injury = hyperammonemia, "
            "ASL patients develop progressive neurocognitive impairment, "
            "systemic hypertension, and hepatic fibrosis EVEN WITH EXCELLENT AMMONIA CONTROL; "
            "mechanism: local arginine/NO deficiency in brain endothelium, liver, kidney; "
            "arginine supplementation reduces systemic hypertension and may partially mitigate "
            "the extra-hepatic ASL deficiency effects; "
            "this is WHY arginine supplementation is even more critical in ASL than other UCDs. "
            "CLINICAL PHENOTYPE: "
            "Neonatal onset: hyperammonemic crisis similar to other UCDs; "
            "Chronic: cognitive impairment, learning difficulties, attention deficit; "
            "Systemic hypertension (50-75%) — from renal/endothelial NO deficiency; "
            "Hepatic fibrosis (30%) — can progress to cirrhosis independent of ammonia; "
            "Liver biopsy NOT diagnostic — non-specific findings. "
            "BIOCHEMICAL DIAGNOSIS: "
            "Plasma citrulline: moderately elevated (100-300 µmol/L; less than ASS1); "
            "Plasma argininosuccinate: elevated (nearly pathognomonic); "
            "Urine argininosuccinate: markedly elevated — key diagnostic finding; "
            "Ammonia: elevated in crisis; may be controlled between crises. "
            "MANAGEMENT: "
            "Arginine supplementation 400-700 mg/kg/day (essential, drives ASS1→ASL cycle); "
            "Protein restriction + essential amino acid formula; "
            "Sodium benzoate + phenylacetate for hyperammonemia; "
            "Blood pressure monitoring and treatment; "
            "Annual liver function/ultrasound (hepatic fibrosis surveillance); "
            "Liver transplant: corrects ammonia cycle in liver but does NOT correct "
            "extra-hepatic ASL deficiency in brain/kidney/endothelium."
        ),
        "locus": "7cen-q11.2",
        "aa": 464,
        "kDa": 52,
        "omim_gene": 608310,
        "omim_disease": 207900,
        "inheritance": "AR",
        "gene_class": "cytoplasmic enzyme — urea cycle step 4",
        "key_alerts": [
            "ASL-ARGININOSUCCINATE-URINE-PATHOGNOMONIC: argininosuccinate in urine is the biochemical hallmark of ASL deficiency; urine organic acids or amino acid analysis shows massively elevated argininosuccinate",
            "ASL-TRICHORRHEXIS-NODOSA-BAMBOO-HAIR: bamboo-hair on light microscopy of hair shaft is pathognomonic — caused by local arginine/NO deficiency in hair follicles; check hair microscopy in any suspected UCD patient with hair abnormality",
            "ASL-NEUROTOXICITY-AMMONIA-INDEPENDENT: unlike other UCDs, ASL causes progressive brain damage even with good ammonia control — from extra-hepatic arginine/NO deficiency; good ammonia levels are NECESSARY but NOT SUFFICIENT for outcome; arginine supplementation is critical",
            "ASL-HYPERTENSION-50-75pct: systemic hypertension from renal/endothelial NO deficiency; monitor BP at every visit; treat aggressively; may improve with higher arginine doses",
            "ASL-LIVER-TRANSPLANT-INCOMPLETE: liver transplant corrects hepatic ASL but NOT extra-hepatic deficiency in brain/kidney/endothelium; post-transplant neurocognitive impairment can still progress",
        ],
        "etiologies": {
            "neonatal_onset_hyperammonemia": 50,
            "late_onset_neurological": 30,
            "asymptomatic_nbs": 20,
        },
        "stats": {
            "disease": "Argininosuccinic aciduria (ASA)",
            "incidence": "1:70,000 live births",
            "trichorrhexis_nodosa_pct": 45,
            "systemic_hypertension_pct": 62,
            "hepatic_fibrosis_pct": 30,
            "arginine_supplement_required_pct": 100,
        },
        "dx_delay_distribution": {
            "<24h": 20, "24-72h": 35, "4-7 days": 25, ">1 week": 20
        },
    },
    # -- CPS1 -- CPS1 Deficiency -------------------------------------------
    {
        "gene": "CPS1",
        "protein": (
            "CPS1 -- 2q35 AR -- Carbamoyl-Phosphate-Synthetase-1-1500aa -- "
            "CPS1-Deficiency -- "
            "Citrulline-ABSENT-or-Trace-KEY-DDx-from-ASS1-ASL -- "
            "Orotic-Acid-NORMAL-KEY-DDx-from-OTC -- "
            "Citrulline-Supplementation-KEY-Empiric-Treatment -- "
            "NAGS-Deficiency-Mimics-CPS1-Biochemically-Treat-with-NCG"
        ),
        "alias": (
            "CPS1 (carbamoyl phosphate synthetase 1); OMIM gene 608307; "
            "CPS1 deficiency OMIM 237300. "
            "2q35; 1500 aa; ~165 kDa (largest enzyme in urea cycle); autosomal recessive. "
            "FUNCTION: CPS1 catalyses the ATP-dependent condensation of NH3 + HCO3- → "
            "carbamoyl phosphate (the FIRST committed step of the urea cycle, mitochondrial matrix). "
            "CPS1 requires N-acetylglutamate (NAG) as an allosteric activator — "
            "NAG is synthesised by NAGS (N-acetylglutamate synthase) from glutamate + acetyl-CoA. "
            "Arginine is a positive modulator of NAGS activity — closes the regulatory loop. "
            "CPS1 LOF → carbamoyl phosphate not formed → entire urea cycle blocked at step 1 → "
            "ammonia accumulates with NO citrulline and NO orotic acid. "
            "BIOCHEMICAL SIGNATURE (critical DDx): "
            "Plasma citrulline: absent or trace (<5 µmol/L); "
            "Urine orotic acid: NORMAL (carbamoyl phosphate doesn't overflow into pyrimidine pathway "
            "because it can't be made — unlike OTC where CP is made but can't proceed → overflows); "
            "Plasma glutamine: very high; "
            "Ammonia: markedly elevated; "
            "CPS1 vs OTC vs NAGS differentiation: all have absent citrulline; "
            "OTC: elevated orotic acid; CPS1 and NAGS: normal orotic acid; "
            "NAGS vs CPS1: clinically identical biochemically — differentiate by: "
            "response to N-carbamylglutamate (NCG/carglumic acid): NAGS → DRAMATIC improvement; "
            "CPS1: NO improvement with NCG; "
            "sequencing confirms. "
            "CLINICAL PHENOTYPE: "
            "Severe neonatal form: acute hyperammonemia within 24-72h of birth; "
            "overwhelming encephalopathy, cerebral oedema, death without treatment; "
            "CPS1 deficiency is one of the most severe UCDs (largest enzyme, step 1 of cycle); "
            "Late-onset: episodic hyperammonemia with catabolic triggers; "
            "Carriers: usually asymptomatic (haploinsufficiency sufficient for cycle). "
            "MANAGEMENT: "
            "Acute: same as other UCDs — stop protein, glucose/lipid, ammonia scavengers; "
            "Citrulline supplementation: citrulline 100-200 mg/kg/day bypasses the CPS1 block "
            "(citrulline enters cycle at step 3 ASS1) — empiric treatment while awaiting diagnosis; "
            "N-carbamylglutamate (carglumic acid) trial: always trial NCG in apparent CPS1 deficiency "
            "to exclude NAGS deficiency (treatable); NCG dose: 100-300 mg/kg/day; "
            "observe ammonia response within 24-48 hours; "
            "Liver transplant: curative for hepatic enzymatic defect; "
            "prognosis after transplant is good if neurological damage not already severe."
        ),
        "locus": "2q35",
        "aa": 1500,
        "kDa": 165,
        "omim_gene": 608307,
        "omim_disease": 237300,
        "inheritance": "AR",
        "gene_class": "mitochondrial enzyme — urea cycle step 1",
        "key_alerts": [
            "CPS1-ABSENT-CITRULLINE-NORMAL-OROTIC-ACID: absent/trace citrulline WITH normal orotic acid distinguishes CPS1/NAGS from OTC (elevated orotic acid) and from ASS1/ASL (elevated citrulline); this biochemical pattern mandates NCG trial to exclude treatable NAGS deficiency",
            "CPS1-TRIAL-NCG-ALWAYS: every patient presenting with absent citrulline + normal orotic acid must receive N-carbamylglutamate (carglumic acid) trial — NAGS deficiency is clinically and biochemically indistinguishable from CPS1 but dramatically responds to NCG; do not withhold empiric NCG while awaiting sequencing",
            "CPS1-CITRULLINE-SUPPLEMENTATION-BYPASSES-BLOCK: citrulline supplementation enters the urea cycle downstream of the CPS1 block (at ASS1, step 3); empiric citrulline 100-200 mg/kg/day is appropriate while awaiting diagnosis in ANY suspected proximal UCD (OTC/CPS1/NAGS)",
            "CPS1-LARGEST-UCD-ENZYME-1500aa: CPS1 at 1500 aa is the largest enzyme in the urea cycle; variants are distributed across the entire gene; no common founder variant; sequencing required",
        ],
        "etiologies": {
            "severe_neonatal_onset": 70,
            "late_onset_episodic": 20,
            "nags_deficiency_mimic": 10,
        },
        "stats": {
            "disease": "CPS1 deficiency",
            "incidence": "1:800,000 live births (rare)",
            "citrulline_absent_pct": 97,
            "orotic_acid_normal_pct": 99,
            "ncg_trial_performed_pct": 85,
            "liver_transplant_performed_pct": 45,
        },
        "dx_delay_distribution": {
            "<24h": 35, "24-72h": 45, "4-7 days": 15, ">1 week": 5
        },
    },
    # -- ARG1 -- Arginemia --------------------------------------------------
    {
        "gene": "ARG1",
        "protein": (
            "ARG1 -- 6q23.2 AR -- Arginase-1-322aa -- "
            "Arginemia -- "
            "Plasma-Arginine->400µmol/L-PATHOGNOMONIC -- "
            "Spastic-Diplegia-Quadriplegia-NOT-Classic-Neonatal-Hyperammonemia -- "
            "Ammonia-MILDLY-Elevated-Unlike-Other-UCDs -- "
            "Dietary-Protein-Restriction-CENTRAL-Treatment"
        ),
        "alias": (
            "ARG1 (arginase 1); OMIM gene 608313; "
            "Arginemia (arginine:glycine amidinotransferase deficiency) OMIM 207800. "
            "6q23.2; 322 aa; ~35 kDa; autosomal recessive. "
            "FUNCTION: ARG1 catalyses the hydrolysis of arginine → ornithine + urea "
            "(the FINAL step of the urea cycle, cytoplasmic). "
            "ARG1 is also highly expressed in red blood cells (RBCs) and macrophages. "
            "ARG1 LOF → arginine accumulates massively → plasma arginine >400 µmol/L; "
            "urea still partially produced (via kidney arginase 2); "
            "ALTERNATIVE ARGININE PATHWAY: guanidino compounds (homoarginine, argininic acid) accumulate "
            "→ these may be the primary neurotoxins in arginemia. "
            "UNIQUE CLINICAL PHENOTYPE — DIFFERENT FROM OTHER UCDs: "
            "Not primarily a neonatal hyperammonemia disorder; "
            "Gradual-onset progressive neurological syndrome: "
            "spastic diplegia (scissor gait) → spastic quadriplegia; "
            "intellectual disability (progressive); "
            "seizures (50-60%); "
            "growth retardation; "
            "Ammonia: only mildly elevated (hyperammonemia is NOT the dominant feature); "
            "ammonia may be intermittently elevated especially postprandially; "
            "Neonatal period: usually asymptomatic or detected on NBS (elevated citrulline + arginine); "
            "VERY DIFFERENT presentation from other UCDs — neurologist referral for spastic "
            "diplegia in a toddler may be the first presentation. "
            "BIOCHEMICAL DIAGNOSIS: "
            "Plasma arginine: >400 µmol/L (normal <80 µmol/L); very elevated; "
            "Plasma amino acids: guanidino compounds elevated; "
            "Ammonia: mildly elevated (unlike other UCDs — this is KEY DDx point); "
            "Urine orotate: elevated (ornithine from ARG1 feeds back to urea cycle differently); "
            "Erythrocyte arginase: can be measured; low activity confirms. "
            "NBS: arginine detected on DBS; citrulline normal. "
            "MANAGEMENT: "
            "Dietary protein restriction: LOW-arginine diet — restrict arginine-rich foods "
            "(meat, fish, dairy, nuts); essential amino acid supplement without arginine; "
            "maintain plasma arginine <200 µmol/L for best neurological outcome; "
            "Sodium benzoate: reduces arginine production via nitrogen scavenging; "
            "Sodium phenylacetate + benzoate (Ammonul) for acute episodes; "
            "Monitoring: annual neurological assessment, MRI brain, plasma arginine; "
            "Liver transplant: controversial in arginemia — ammonia is not the main issue; "
            "transplant reduces plasma arginine but neurological progression may continue "
            "(extra-hepatic ARG1 deficiency in RBCs, other tissues); "
            "some evidence of neurological stabilisation post-transplant if done early. "
            "PROGNOSIS: With early dietary control (NBS detected), neurological progression "
            "may be slowed but not always prevented; late diagnosis (spastic diplegia at 2-5 years) "
            "→ outcomes worse."
        ),
        "locus": "6q23.2",
        "aa": 322,
        "kDa": 35,
        "omim_gene": 608313,
        "omim_disease": 207800,
        "inheritance": "AR",
        "gene_class": "cytoplasmic enzyme — urea cycle step 5 (final)",
        "key_alerts": [
            "ARG1-SPASTIC-DIPLEGIA-NOT-HYPERAMMONEMIA: arginemia presents with progressive spastic diplegia/quadriplegia rather than acute neonatal hyperammonemia; neurologist/paediatrician seeing a toddler with spastic gait should check plasma arginine and ammonia",
            "ARG1-AMMONIA-MILDLY-ELEVATED-KEY-DDX: ammonia is only mildly elevated in arginemia — distinguishing from severe hyperammonemia of OTC/CPS1/ASS1/ASL; do not rule out arginemia because ammonia is not dramatically high",
            "ARG1-PLASMA-ARGININE->400-PATHOGNOMONIC: plasma arginine >400 µmol/L (normal <80) is the biochemical marker; dietary protein restriction should target plasma arginine <200 µmol/L",
            "ARG1-LOW-ARGININE-DIET-NOT-ARGININE-SUPPLEMENT: unlike ASS1/ASL/ASL where arginine is supplemented, in ARG1 deficiency arginine is RESTRICTED — do not confuse treatment directions between UCD genes",
        ],
        "etiologies": {
            "progressive_spastic_diplegia": 60,
            "nbs_detected_pre_symptomatic": 25,
            "seizure_onset": 15,
        },
        "stats": {
            "disease": "Arginemia (ARG1 deficiency)",
            "incidence": "1:300,000 live births",
            "plasma_arginine_median_umol_L": 550,
            "spastic_diplegia_pct": 90,
            "seizure_pct": 55,
            "ammonia_mild_elevation_pct": 85,
            "dietary_restriction_required_pct": 100,
        },
        "dx_delay_distribution": {
            "<24h": 5, "24-72h": 10, "4 weeks-2 years": 40, ">2 years": 45
        },
    },
    # -- NAGS -- NAGS Deficiency -------------------------------------------
    {
        "gene": "NAGS",
        "protein": (
            "NAGS -- 17q21.31 AR -- N-Acetylglutamate-Synthase-534aa -- "
            "NAGS-Deficiency -- "
            "Clinically-Identical-to-CPS1-Absent-Citrulline-Normal-Orotic-Acid -- "
            "ONLY-UCD-with-Specific-Antidote-N-Carbamylglutamate-NCG-Carglumic-Acid -- "
            "DRAMATIC-Response-to-NCG-Within-24-48h -- "
            "Must-Trial-NCG-Before-Confirming-CPS1-Diagnosis"
        ),
        "alias": (
            "NAGS (N-acetylglutamate synthase); OMIM gene 608300; "
            "NAGS deficiency OMIM 237310. "
            "17q21.31; 534 aa; ~62 kDa; autosomal recessive. "
            "FUNCTION: NAGS catalyses the condensation of glutamate + acetyl-CoA → "
            "N-acetylglutamate (NAG) in the mitochondrial matrix. "
            "NAG is the OBLIGATE allosteric activator of CPS1 — without NAG, CPS1 is inactive. "
            "Therefore: NAGS deficiency → no NAG → CPS1 inactive → same phenotype as CPS1 deficiency. "
            "Arginine stimulates NAGS activity (positive feedback loop). "
            "N-CARBAMYLGLUTAMATE (NCG/CARGLUMIC ACID) — THE ANTIDOTE: "
            "NCG is a structural analogue of NAG that is NOT degraded by the liver mitochondria "
            "(unlike natural NAG which is rapidly hydrolysed); "
            "NCG can bind to and activate CPS1 DIRECTLY, bypassing the NAGS deficiency; "
            "Result: patients with NAGS deficiency respond DRAMATICALLY to oral NCG "
            "(carglumic acid; Carbaglu); "
            "ammonia normalises within 24-48 hours of starting NCG; "
            "NCG dose: 50-250 mg/kg/day; "
            "this is the ONLY UCD with this 'antidote' mechanism — no other UCD responds to NCG. "
            "WHY NCG TRIAL IS MANDATORY: "
            "NAGS and CPS1 deficiency are biochemically IDENTICAL on routine plasma amino acids: "
            "both → absent citrulline, normal orotic acid, high glutamine, high ammonia; "
            "only sequencing differentiates definitively; "
            "but NCG trial provides IMMEDIATE therapeutic benefit if NAGS AND "
            "provides diagnostic confirmation (CPS1 does NOT respond to NCG); "
            "delaying NCG trial awaiting sequencing = denying potentially life-saving treatment; "
            "therefore: trial NCG immediately in any patient with absent citrulline + normal orotic acid. "
            "SECONDARY NAGS DEFICIENCY: "
            "Several organic acidaemias secondarily inhibit NAGS activity: "
            "propionic acidaemia (propionyl-CoA inhibits NAGS), "
            "methylmalonic acidaemia, isovaleric acidaemia; "
            "these patients may also benefit temporarily from NCG during metabolic crisis; "
            "NCG has a role in treating hyperammonemia secondary to organic acidaemias. "
            "CLINICAL PHENOTYPE (primary NAGS deficiency): "
            "Neonatal: severe hyperammonemia identical to CPS1; "
            "Late-onset: episodic hyperammonemia; "
            "Chronic management: NCG oral long-term; often dramatic improvement in quality of life; "
            "arginine supplementation (arginine stimulates residual NAGS); "
            "protein restriction. "
            "PROGNOSIS: with NCG treatment, prognosis is generally better than CPS1 deficiency; "
            "enzyme activity partially restored or CPS1 activated directly."
        ),
        "locus": "17q21.31",
        "aa": 534,
        "kDa": 62,
        "omim_gene": 608300,
        "omim_disease": 237310,
        "inheritance": "AR",
        "gene_class": "mitochondrial enzyme — NAG synthesis; CPS1 activator",
        "key_alerts": [
            "NAGS-NCG-ONLY-UCD-ANTIDOTE: N-carbamylglutamate (carglumic acid/Carbaglu) is the ONLY specific antidote-like treatment in any UCD; NAGS deficiency responds dramatically within 24-48h of starting NCG; no other UCD responds to NCG",
            "NAGS-BIOCHEMICALLY-IDENTICAL-TO-CPS1: absent citrulline + normal orotic acid occurs in BOTH NAGS and CPS1 deficiency; MUST trial NCG immediately — withholding NCG in apparent 'CPS1' risks missing treatable NAGS deficiency",
            "NAGS-SECONDARY-DEFICIENCY-ORGANIC-ACIDAEMIAS: propionic acidaemia, MMA, isovaleric acidaemia secondarily inhibit NAGS (via propionyl-CoA/organic acids) → hyperammonemia; NCG may benefit these patients during acute crises even without primary NAGS deficiency",
            "NAGS-ARGININE-STIMULATES-RESIDUAL-NAGS: arginine supplementation activates residual NAGS activity via allosteric stimulation; combine NCG + arginine for maximal CPS1 activation in NAGS deficiency",
        ],
        "etiologies": {
            "primary_nags_deficiency_neonatal": 40,
            "primary_nags_deficiency_late_onset": 35,
            "secondary_nags_inhibition_organic_acidemia": 25,
        },
        "stats": {
            "disease": "NAGS deficiency",
            "incidence": "1:2,000,000 live births (ultra-rare)",
            "ncg_dramatic_response_pct": 92,
            "ammonia_normalised_24h_ncg_pct": 80,
            "citrulline_absent_pct": 98,
            "orotic_acid_normal_pct": 99,
        },
        "dx_delay_distribution": {
            "<24h": 30, "24-72h": 40, "4-7 days": 20, ">1 week": 10
        },
    },
    # -- SLC25A15 -- HHH Syndrome ------------------------------------------
    {
        "gene": "SLC25A15",
        "protein": (
            "SLC25A15 -- 13q14.11 AR -- Ornithine-Carrier-ORC1-301aa -- "
            "HHH-Syndrome-Hyperornithinemia-Hyperammonemia-Homocitrullinuria -- "
            "Homocitrulline-in-Urine-PATHOGNOMONIC -- "
            "Plasma-Ornithine-Elevated->200µmol/L -- "
            "Progressive-Spastic-Paraplegia-Cerebellar-Ataxia -- "
            "French-Canadian-Founder-Mutation-pF188del"
        ),
        "alias": (
            "SLC25A15 (solute carrier family 25 member 15; ornithine carrier 1 ORC1); "
            "OMIM gene 603861; "
            "HHH syndrome (hyperornithinemia-hyperammonemia-homocitrullinuria) OMIM 238970. "
            "13q14.11; 301 aa; ~34 kDa; autosomal recessive. "
            "FUNCTION: SLC25A15 encodes the mitochondrial ornithine carrier (ORC1), "
            "which transports ornithine from the cytoplasm INTO the mitochondrial matrix "
            "to be used by OTC (for condensation with carbamoyl phosphate). "
            "SLC25A15 LOF → ornithine cannot enter mitochondria → "
            "(1) Ornithine accumulates in cytoplasm → hyperornithinemia (plasma ornithine >200 µmol/L); "
            "(2) OTC substrate (ornithine) depleted in mitochondria → secondary OTC limitation → "
            "carbamoyl phosphate cannot be fully utilised → "
            "carbamoyl phosphate overflows → alternative pathway → "
            "carbamoyl moiety reacts with lysine → homocitrulline (in urine, PATHOGNOMONIC); "
            "also contributes to orotic acid elevation; "
            "(3) Ammonia cannot be detoxified normally → hyperammonemia. "
            "HHH TRIAD (PATHOGNOMONIC): "
            "(1) Hyperornithinemia: plasma ornithine >200 µmol/L; "
            "(2) Hyperammonemia: elevated ammonia (variable, may be episodic); "
            "(3) Homocitrullinuria: homocitrulline in urine — the KEY diagnostic marker; "
            "all three required for diagnosis; homocitrullinuria is the most specific. "
            "CLINICAL PHENOTYPE: "
            "Variable — from asymptomatic to severe; "
            "Neonatal hyperammonemia (minority); "
            "Most common: childhood onset with cognitive impairment, behavioural problems; "
            "Progressive spastic paraplegia (most adults); "
            "Cerebellar ataxia in some; "
            "Episodic confusion/encephalopathy with catabolic stress; "
            "French-Canadian founder variant pF188del (c.562_564delTTC) — enriched in Quebec; "
            "mild clinical phenotype on average in this population. "
            "MANAGEMENT: "
            "Protein restriction: reduces ammonia production; "
            "Ornithine supplementation: paradoxically used in HHH despite hyperornithinemia; "
            "rationale: extra ornithine may partially overcome the transport block, "
            "providing some ornithine to mitochondria via alternative pathways; "
            "evidence is limited; "
            "Lysine restriction: reduces homocitrulline precursor; "
            "Ammonia scavengers as needed; "
            "Avoid catabolic stress (fasting, illness)."
        ),
        "locus": "13q14.11",
        "aa": 301,
        "kDa": 34,
        "omim_gene": 603861,
        "omim_disease": 238970,
        "inheritance": "AR",
        "gene_class": "mitochondrial carrier — ornithine import (step 0 of urea cycle)",
        "key_alerts": [
            "SLC25A15-HOMOCITRULLINE-URINE-PATHOGNOMONIC: homocitrulline in urine is the key diagnostic marker of HHH syndrome; formed when carbamoyl phosphate (accumulating due to ornithine deficiency in mitochondria) reacts with lysine instead of ornithine",
            "SLC25A15-HHH-TRIAD-ALL-THREE: hyperornithinemia + hyperammonemia + homocitrullinuria — all three required; elevated plasma ornithine is the clue on plasma amino acids to request urine for homocitrulline",
            "SLC25A15-FRENCH-CANADIAN-FOUNDER: p.F188del (c.562_564delTTC) is a founder variant enriched in French-Canadian (Quebec) population; consider HHH in any French-Canadian patient with unexplained cognitive decline or spastic paraplegia",
            "SLC25A15-PROGRESSIVE-SPASTIC-PARAPLEGIA: unlike early-presenting UCDs, HHH often progresses to spastic paraplegia and cerebellar features in adulthood even with ammonia control",
        ],
        "etiologies": {
            "progressive_spastic_paraplegia": 50,
            "episodic_encephalopathy": 30,
            "neonatal_hyperammonemia": 10,
            "asymptomatic_nbs": 10,
        },
        "stats": {
            "disease": "HHH syndrome",
            "incidence": "1:350,000 live births (French-Canadian enriched)",
            "plasma_ornithine_median_umol_L": 380,
            "homocitrullinuria_pct": 100,
            "spastic_paraplegia_pct": 55,
            "french_canadian_founder_pct": 40,
        },
        "dx_delay_distribution": {
            "<1 year": 15, "1-5 years": 30, "5-15 years": 35, ">15 years": 20
        },
    },
    # -- SLC25A13 -- Citrin Deficiency / CTLN2 ----------------------------
    {
        "gene": "SLC25A13",
        "protein": (
            "SLC25A13 -- 7q21.3 AR -- Citrin-Aspartate-Glutamate-Carrier-675aa -- "
            "Citrin-Deficiency-NICCD-Neonatal-Cholestasis-then-CTLN2-Adult-Citrullinemia -- "
            "Carbohydrate-Rich-Diet-HARMFUL-Unique-Reversal-of-Dietary-Instinct -- "
            "High-Protein-High-Fat-Diet-BENEFICIAL -- "
            "Sweet-Food-Aversion-Self-Protective-Behaviour -- "
            "East-Asian-Japan-1-in-17000"
        ),
        "alias": (
            "SLC25A13 (solute carrier family 25 member 13; citrin; "
            "aspartate-glutamate carrier 2 AGC2); OMIM gene 603859; "
            "Citrin deficiency (NICCD neonatal; CTLN2 adult) OMIM 605814. "
            "7q21.3; 675 aa; ~74 kDa; autosomal recessive. "
            "FUNCTION: SLC25A13 encodes citrin, the mitochondrial aspartate-glutamate carrier "
            "isoform 2, which transports aspartate (and glutamate in exchange) from mitochondria "
            "to cytoplasm. Citrin is a component of the malate-aspartate shuttle — essential for "
            "cytoplasmic NADH oxidation and the supply of aspartate for the urea cycle cytoplasmic steps. "
            "In the LIVER: citrin is required to transport aspartate for ASS1 → ASL reactions; "
            "without citrin, hepatic urea cycle is functionally impaired → "
            "citrulline and ammonia accumulate in adult onset (CTLN2). "
            "CLINICAL STAGES OF CITRIN DEFICIENCY — THREE DISTINCT PRESENTATIONS: "
            "(1) NICCD (Neonatal Intrahepatic Cholestasis caused by Citrin Deficiency): "
            "neonatal period — 0-1 year; "
            "transient neonatal cholestasis with jaundice, hepatomegaly; "
            "elevated conjugated bilirubin, elevated AFP, hypoglycaemia, aminoacidaemia, "
            "galactosaemia-like metabolic profile; "
            "usually resolves spontaneously by age 1 with lactose-free/MCT formula; "
            "(2) FTTDCD (Failure to Thrive and Dyslipidemia caused by Citrin Deficiency): "
            "toddler to childhood; "
            "failure to thrive, fatty liver, hyperlipidemia; "
            "often compensated; "
            "(3) CTLN2 (Citrullinemia type II — adult onset): "
            "typically 11-79 years (peak 20-40 years); "
            "recurrent hyperammonemia episodes → encephalopathy, bizarre behaviour, drowsiness; "
            "plasma citrulline: moderately to markedly elevated (200-1000 µmol/L); "
            "elevated arginine, threonine, methionine. "
            "UNIQUE DIETARY PHENOTYPE — CARBOHYDRATE RESTRICTION IS KEY: "
            "Carbohydrate-rich diet (rice, sweets, alcohol) PRECIPITATES/WORSENS CTLN2 — "
            "this is the REVERSE of typical metabolic disease dietary advice; "
            "mechanism: high carbohydrate → high NADH in cytoplasm → overwhelms impaired "
            "malate-aspartate shuttle → forces NADH through alternative (alcohol dehydrogenase) "
            "pathway → lactate accumulates → citrin malate-aspartate shuttle even more impaired; "
            "Patients develop FOOD AVERSIONS (self-protective): strong aversion to rice, "
            "sweets, fizzy drinks, alcohol — pathognomonic when elicited in history; "
            "HIGH-PROTEIN HIGH-FAT DIET is beneficial and preferred by patients instinctively; "
            "branched-chain amino acid supplement; "
            "LIVER TRANSPLANT: definitive treatment for CTLN2; "
            "corrects hepatic citrin deficiency completely; "
            "citrulline normalises post-transplant. "
            "PREVALENCE: Common in East Asians — Japan 1:17,000; Korea 1:48,000; "
            "Southeast Asia significantly more common than in Western populations; "
            "NBS in East Asian countries includes citrulline detection."
        ),
        "locus": "7q21.3",
        "aa": 675,
        "kDa": 74,
        "omim_gene": 603859,
        "omim_disease": 605814,
        "inheritance": "AR",
        "gene_class": "mitochondrial carrier — aspartate-glutamate shuttle; urea cycle support",
        "key_alerts": [
            "SLC25A13-CARBOHYDRATE-DIET-HARMFUL-UNIQUE: carbohydrate-rich foods (rice, sweets, alcohol) PRECIPITATE CTLN2 episodes — the OPPOSITE of instinct for metabolic disease dietary advice; always ask about carbohydrate aversion (pathognomonic history); high-protein/fat diet is beneficial and self-selected by patients",
            "SLC25A13-SWEET-FOOD-AVERSION-PATHOGNOMONIC-HISTORY: strong aversion to sweet foods, rice, alcohol is a pathognomonic symptom cluster of citrin deficiency (self-protective mechanism); always ask adult patients with episodic encephalopathy about food preferences and aversions",
            "SLC25A13-THREE-STAGE-SPECTRUM: NICCD (neonatal cholestasis, resolves ~1 year) → FTTDCD (childhood failure to thrive) → CTLN2 (adult recurrent encephalopathy); same gene, three clinical presentations across lifespan; NBS in East Asian countries targets this",
            "SLC25A13-EAST-ASIAN-COMMON-1:17000-JAPAN: citrin deficiency is relatively common in East Asian populations (Japan 1:17,000); rare in Western populations; always consider in East Asian patients with unexplained neonatal cholestasis or adult episodic encephalopathy",
            "SLC25A13-LIVER-TRANSPLANT-CURATIVE-CTLN2: liver transplant completely corrects CTLN2 — citrulline normalises and encephalopathy episodes cease; indicated for medically refractory CTLN2 or when dietary management fails",
        ],
        "etiologies": {
            "niccd_neonatal_cholestasis": 40,
            "fttdcd_childhood": 20,
            "ctln2_adult_encephalopathy": 30,
            "asymptomatic_nbs": 10,
        },
        "stats": {
            "disease": "Citrin deficiency (NICCD/CTLN2)",
            "incidence": "1:17,000 (Japan); 1:48,000 (Korea); rare in West",
            "carbohydrate_aversion_ctln2_pct": 85,
            "plasma_citrulline_ctln2_median_umol_L": 420,
            "niccd_spontaneous_resolution_pct": 80,
            "liver_transplant_curative_pct": 100,
        },
        "dx_delay_distribution": {
            "<1 year_niccd": 40, "1-10 years_fttdcd": 20,
            "10-40 years_ctln2": 30, ">40 years": 10
        },
    },
]


def _generate_patients():
    """Generate 40 synthetic patients per gene (8 × 40 = 320 total)."""
    for idx, gene_data in enumerate(UCD_GENES):
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        patients = []
        gene = gene_data["gene"]

        for i in range(40):
            pid = f"{gene}-{seed}-P{i+1:02d}"
            age = rng.randint(0, 60)

            if gene == "OTC":
                is_male = rng.random() < 0.55
                severe_neo = rng.random() < 0.55 and is_male
                carrier_female = not is_male and rng.random() < 0.15
                ammonia_peak = rng.randint(300, 900) if severe_neo else rng.randint(80, 350)
                citrulline_absent = rng.random() < 0.96
                orotic_acid_elevated = rng.random() < 0.98
                valproate_error = rng.random() < 0.08
                liver_transplant = rng.random() < 0.30
                dx_delay_days = rng.randint(0, 5) if severe_neo else rng.randint(0, 60)
                patients.append({
                    "patient_id": pid, "age_years": age, "sex": "M" if is_male else "F",
                    "severe_neonatal": severe_neo, "carrier_female": carrier_female,
                    "ammonia_peak_umol_L": ammonia_peak,
                    "citrulline_absent": citrulline_absent,
                    "orotic_acid_elevated": orotic_acid_elevated,
                    "valproate_prescribed_error": valproate_error,
                    "liver_transplant": liver_transplant,
                    "dx_delay_days": dx_delay_days,
                    "outcome": "stable" if liver_transplant or not severe_neo else rng.choice(["stable", "mild_disability", "severe_disability"]),
                })

            elif gene == "ASS1":
                nbs_detected = rng.random() < 0.75
                citrulline = rng.randint(900, 1800)
                arginine_prescribed = rng.random() < 0.95
                liver_transplant = rng.random() < 0.35
                ammonia_peak = rng.randint(200, 700)
                dx_delay_days = 1 if nbs_detected else rng.randint(2, 10)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "nbs_detected": nbs_detected,
                    "plasma_citrulline_umol_L": citrulline,
                    "arginine_supplemented": arginine_prescribed,
                    "liver_transplant": liver_transplant,
                    "ammonia_peak_umol_L": ammonia_peak,
                    "dx_delay_days": dx_delay_days,
                    "outcome": "stable" if nbs_detected else rng.choice(["stable", "mild_disability"]),
                })

            elif gene == "ASL":
                trichorrhexis_nodosa = rng.random() < 0.45
                nbs_detected = rng.random() < 0.60
                hypertension = rng.random() < 0.62
                hepatic_fibrosis = rng.random() < 0.30
                arginine_prescribed = rng.random() < 0.98
                citrulline = rng.randint(100, 350)
                plasma_argininosuccinate = rng.randint(50, 500)
                dx_delay_days = 1 if nbs_detected else rng.randint(3, 30)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "trichorrhexis_nodosa": trichorrhexis_nodosa,
                    "nbs_detected": nbs_detected,
                    "systemic_hypertension": hypertension,
                    "hepatic_fibrosis": hepatic_fibrosis,
                    "arginine_supplemented": arginine_prescribed,
                    "plasma_citrulline_umol_L": citrulline,
                    "plasma_argininosuccinate_umol_L": plasma_argininosuccinate,
                    "dx_delay_days": dx_delay_days,
                    "outcome": rng.choice(["stable", "mild_disability", "stable"]),
                })

            elif gene == "CPS1":
                severe_neo = rng.random() < 0.70
                citrulline_absent = rng.random() < 0.97
                orotic_acid_normal = rng.random() < 0.99
                ncg_trial_done = rng.random() < 0.85
                nags_excluded = ncg_trial_done
                liver_transplant = rng.random() < 0.45
                ammonia_peak = rng.randint(400, 1100) if severe_neo else rng.randint(150, 400)
                dx_delay_days = rng.randint(0, 4) if severe_neo else rng.randint(2, 14)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "severe_neonatal": severe_neo,
                    "citrulline_absent": citrulline_absent,
                    "orotic_acid_normal": orotic_acid_normal,
                    "ncg_trial_done": ncg_trial_done,
                    "nags_excluded": nags_excluded,
                    "liver_transplant": liver_transplant,
                    "ammonia_peak_umol_L": ammonia_peak,
                    "dx_delay_days": dx_delay_days,
                    "outcome": "stable" if liver_transplant else rng.choice(["stable", "mild_disability", "severe_disability"]),
                })

            elif gene == "ARG1":
                spastic_diplegia = rng.random() < 0.90
                nbs_detected = rng.random() < 0.25
                seizures = rng.random() < 0.55
                plasma_arginine = rng.randint(400, 900)
                ammonia_mild = rng.random() < 0.85
                dietary_restriction = rng.random() < 0.99
                dx_delay_years = 0 if nbs_detected else rng.randint(1, 8)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "spastic_diplegia": spastic_diplegia,
                    "nbs_detected": nbs_detected,
                    "seizures": seizures,
                    "plasma_arginine_umol_L": plasma_arginine,
                    "ammonia_mildly_elevated": ammonia_mild,
                    "dietary_protein_restriction": dietary_restriction,
                    "dx_delay_years": dx_delay_years,
                    "outcome": "stable" if nbs_detected else rng.choice(["stable", "mild_disability", "moderate_disability"]),
                })

            elif gene == "NAGS":
                primary_nags = rng.random() < 0.75
                ncg_response_dramatic = rng.random() < 0.92 if primary_nags else rng.random() < 0.40
                ammonia_normalised_24h = rng.random() < 0.80 if ncg_response_dramatic else False
                citrulline_absent = rng.random() < 0.98
                orotic_acid_normal = rng.random() < 0.99
                secondary_organic_acidemia = not primary_nags
                ammonia_peak = rng.randint(300, 800)
                dx_delay_days = rng.randint(1, 30)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "primary_nags_deficiency": primary_nags,
                    "secondary_organic_acidemia": secondary_organic_acidemia,
                    "ncg_response_dramatic": ncg_response_dramatic,
                    "ammonia_normalised_24h_ncg": ammonia_normalised_24h,
                    "citrulline_absent": citrulline_absent,
                    "orotic_acid_normal": orotic_acid_normal,
                    "ammonia_peak_umol_L": ammonia_peak,
                    "dx_delay_days": dx_delay_days,
                    "outcome": "stable" if ncg_response_dramatic else rng.choice(["stable", "mild_disability"]),
                })

            elif gene == "SLC25A15":
                plasma_ornithine = rng.randint(200, 600)
                homocitrullinuria = rng.random() < 1.0
                spastic_paraplegia = rng.random() < 0.55
                french_canadian = rng.random() < 0.40
                episodic_encephalopathy = rng.random() < 0.30
                ammonia_elevated = rng.random() < 0.70
                protein_restriction = rng.random() < 0.90
                dx_delay_years = rng.randint(0, 20)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "plasma_ornithine_umol_L": plasma_ornithine,
                    "homocitrullinuria": homocitrullinuria,
                    "spastic_paraplegia": spastic_paraplegia,
                    "french_canadian_founder": french_canadian,
                    "episodic_encephalopathy": episodic_encephalopathy,
                    "ammonia_elevated": ammonia_elevated,
                    "protein_restriction": protein_restriction,
                    "dx_delay_years": dx_delay_years,
                    "outcome": rng.choice(["stable", "mild_disability", "moderate_disability"]),
                })

            elif gene == "SLC25A13":
                stage = rng.choice(["niccd", "fttdcd", "ctln2", "asymptomatic"])
                carbohydrate_aversion = rng.random() < (0.85 if stage == "ctln2" else 0.30)
                east_asian = rng.random() < 0.80
                niccd_resolved = rng.random() < 0.80 if stage == "niccd" else False
                plasma_citrulline = rng.randint(300, 800) if stage == "ctln2" else rng.randint(80, 250)
                liver_transplant = rng.random() < 0.25 if stage == "ctln2" else False
                dx_delay_years = rng.randint(0, 40) if stage == "ctln2" else rng.randint(0, 5)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "clinical_stage": stage,
                    "carbohydrate_aversion": carbohydrate_aversion,
                    "east_asian": east_asian,
                    "niccd_spontaneous_resolution": niccd_resolved,
                    "plasma_citrulline_umol_L": plasma_citrulline,
                    "liver_transplant_ctln2": liver_transplant,
                    "dx_delay_years": dx_delay_years,
                    "outcome": "stable" if liver_transplant or stage in ("niccd", "asymptomatic") else rng.choice(["stable", "mild_disability"]),
                })

        gene_data["patients"] = patients


_generate_patients()


def _pct(lst, key, val=True):
    if not lst:
        return 0
    return round(100 * sum(1 for p in lst if p.get(key) == val) / len(lst), 1)


def _pct_true(lst, key):
    if not lst:
        return 0
    return round(100 * sum(1 for p in lst if p.get(key)) / len(lst), 1)


def overview():
    all_genes_info = [
        {
            "gene": g["gene"],
            "locus": g["locus"],
            "aa": g["aa"],
            "n_patients": len(g["patients"]),
            "inheritance": g["inheritance"],
        }
        for g in UCD_GENES
    ]
    total = sum(len(g["patients"]) for g in UCD_GENES)
    pts = {g["gene"]: g["patients"] for g in UCD_GENES}

    return {
        "atlas": "Hereditary UCD Atlas — Complete 8-Gene Urea Cycle Disorder Atlas",
        "subtitle": (
            "OTC (Xp21.1-XLR-OTC-Deficiency-Most-Common-UCD-50pct-Valproate-ABSOLUTE-CI-Liver-Transplant) . "
            "ASS1 (9q34.11-AR-CTLN1-Citrulline>1000-PATHOGNOMONIC-Arginine-ESSENTIAL) . "
            "ASL (7q-AR-Argininosuccinic-Aciduria-Trichorrhexis-Nodosa-PATHOGNOMONIC-NH3-Independent-Neurotox) . "
            "CPS1 (2q35-AR-Absent-Citrulline-Normal-Orotic-NCG-Trial-MANDATORY-to-Exclude-NAGS) . "
            "ARG1 (6q23.2-AR-Arginemia-Spastic-Diplegia-NOT-Hyperammonemia-Arginine-HIGH-RESTRICT) . "
            "NAGS (17q21.31-AR-ONLY-UCD-Antidote-NCG-Carglumic-Acid-DRAMATIC-Response-24-48h) . "
            "SLC25A15 (13q14.11-AR-HHH-Syndrome-Homocitrullinuria-PATHOGNOMONIC-French-Canadian) . "
            "SLC25A13 (7q21.3-AR-Citrin-NICCD-CTLN2-Carbohydrate-HARMFUL-High-Protein-BENEFICIAL-East-Asian) -- "
            "320 Patients (8x40, Seeds 1798-1805)"
        ),
        "total_patients": total,
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "genes_covered": 8,
            "patients_per_gene": 40,
            "x_linked_genes": 1,
            "ar_genes": 7,
            # OTC
            "otc_valproate_error_pct": _pct_true(pts["OTC"], "valproate_prescribed_error"),
            "otc_citrulline_absent_pct": _pct_true(pts["OTC"], "citrulline_absent"),
            "otc_orotic_acid_elevated_pct": _pct_true(pts["OTC"], "orotic_acid_elevated"),
            "otc_liver_transplant_pct": _pct_true(pts["OTC"], "liver_transplant"),
            # ASS1
            "ass1_nbs_detected_pct": _pct_true(pts["ASS1"], "nbs_detected"),
            "ass1_arginine_supplemented_pct": _pct_true(pts["ASS1"], "arginine_supplemented"),
            "ass1_citrulline_median_umol_L": round(
                sum(p["plasma_citrulline_umol_L"] for p in pts["ASS1"]) / len(pts["ASS1"]), 0),
            # ASL
            "asl_trichorrhexis_nodosa_pct": _pct_true(pts["ASL"], "trichorrhexis_nodosa"),
            "asl_hypertension_pct": _pct_true(pts["ASL"], "systemic_hypertension"),
            "asl_hepatic_fibrosis_pct": _pct_true(pts["ASL"], "hepatic_fibrosis"),
            # CPS1
            "cps1_ncg_trial_done_pct": _pct_true(pts["CPS1"], "ncg_trial_done"),
            "cps1_citrulline_absent_pct": _pct_true(pts["CPS1"], "citrulline_absent"),
            "cps1_orotic_acid_normal_pct": _pct_true(pts["CPS1"], "orotic_acid_normal"),
            "cps1_liver_transplant_pct": _pct_true(pts["CPS1"], "liver_transplant"),
            # ARG1
            "arg1_spastic_diplegia_pct": _pct_true(pts["ARG1"], "spastic_diplegia"),
            "arg1_ammonia_mildly_elevated_pct": _pct_true(pts["ARG1"], "ammonia_mildly_elevated"),
            "arg1_seizure_pct": _pct_true(pts["ARG1"], "seizures"),
            "arg1_arginine_median_umol_L": round(
                sum(p["plasma_arginine_umol_L"] for p in pts["ARG1"]) / len(pts["ARG1"]), 0),
            # NAGS
            "nags_ncg_dramatic_response_pct": _pct_true(pts["NAGS"], "ncg_response_dramatic"),
            "nags_ammonia_normalised_24h_ncg_pct": _pct_true(pts["NAGS"], "ammonia_normalised_24h_ncg"),
            "nags_citrulline_absent_pct": _pct_true(pts["NAGS"], "citrulline_absent"),
            # SLC25A15
            "hhh_homocitrullinuria_pct": _pct_true(pts["SLC25A15"], "homocitrullinuria"),
            "hhh_spastic_paraplegia_pct": _pct_true(pts["SLC25A15"], "spastic_paraplegia"),
            "hhh_french_canadian_pct": _pct_true(pts["SLC25A15"], "french_canadian_founder"),
            # SLC25A13
            "citrin_carbohydrate_aversion_ctln2_pct": _pct_true(pts["SLC25A13"], "carbohydrate_aversion"),
            "citrin_east_asian_pct": _pct_true(pts["SLC25A13"], "east_asian"),
            "citrin_ctln2_liver_transplant_pct": _pct_true(pts["SLC25A13"], "liver_transplant_ctln2"),
        },
        "genes": all_genes_info,
        "top_alerts": [
            "OTC-VALPROATE-ABSOLUTELY-CI: Valproate (VPA) directly inhibits OTC enzyme and precipitates acute hyperammonemia in any OTC-deficient patient regardless of variant severity — NEVER prescribe VPA for epilepsy or mood stabilisation in OTC deficiency; alternatives: levetiracetam, lamotrigine, lacosamide; VPA also inhibits CPS1 independently",
            "OTC-OROTIC-ACID-KEY-DDX: Elevated urine orotic acid DISTINGUISHES OTC from CPS1 and NAGS deficiency — all three have absent/trace citrulline; OTC: orotic acid HIGH (carbamoyl phosphate overflows into pyrimidine pathway); CPS1/NAGS: orotic acid NORMAL (carbamoyl phosphate not produced); always measure urine orotic acid",
            "NAGS-NCG-MANDATORY-TRIAL: N-carbamylglutamate (carglumic acid/Carbaglu) is the ONLY antidote in any UCD — NAGS deficiency responds dramatically within 24-48h; biochemically IDENTICAL to CPS1 (absent citrulline, normal orotic acid); TRIAL NCG IMMEDIATELY before sequencing confirms — withholding NCG risks missing a treatable diagnosis",
            "ASL-TRICHORRHEXIS-NODOSA-BAMBOO-HAIR: Bamboo hair on light microscopy is pathognomonic for ASL deficiency — caused by local arginine/NO deficiency in hair follicle; examine hair in any UCD patient with brittle hair; arginine supplementation is critical in ASL for NON-AMMONIA neurotoxicity",
            "ASL-AMMONIA-INDEPENDENT-NEUROTOXICITY: ASL deficiency causes progressive brain damage, hypertension, and hepatic fibrosis INDEPENDENT of ammonia levels — good ammonia control is NECESSARY but NOT SUFFICIENT; do not reassure patients with normal ammonia that neurological risk is eliminated",
            "ARG1-SPASTIC-DIPLEGIA-NOT-HYPERAMMONEMIA: Arginemia presents with PROGRESSIVE SPASTIC DIPLEGIA/QUADRIPLEGIA rather than acute neonatal hyperammonemia — can mimic cerebral palsy; always check plasma arginine + ammonia in unexplained spastic diplegia; ammonia is only MILDLY elevated in arginemia unlike other UCDs",
            "ARG1-RESTRICT-ARGININE-NOT-SUPPLEMENT: In ARG1 deficiency, arginine is RESTRICTED (it accumulates); in ASS1/ASL/OTC deficiency, arginine is SUPPLEMENTED (it is deficient); never confuse treatment directions — supplementing arginine in ARG1 deficiency worsens disease",
            "SLC25A13-CARBOHYDRATE-HARMFUL-UNIQUE: Citrin deficiency is the ONLY metabolic disease where carbohydrate-rich diet is the primary precipitant — rice, sweets, alcohol trigger CTLN2 encephalopathy; high-protein/fat diet is BENEFICIAL; patients self-select this diet (carbohydrate aversion is a pathognomonic symptom — always ask about food preferences in East Asian patients with encephalopathy)",
            "SLC25A15-HHH-HOMOCITRULLINE-PATHOGNOMONIC: Homocitrulline in urine is pathognomonic for HHH syndrome (SLC25A15 deficiency) — caused by carbamoyl phosphate reacting with lysine when ornithine is depleted from mitochondria; suspect HHH in any patient with elevated plasma ornithine + hyperammonemia; check urine for homocitrulline",
            "OTC-HAEMODIALYSIS-FOR-SEVERE-HYPERAMMONEMIA: For ammonia >300 µmol/L with encephalopathy, haemodialysis is the FASTEST ammonia removal method — do NOT use peritoneal dialysis (inadequate clearance); direct to ICU immediately; continuous renal replacement therapy (CRRT) is an alternative for haemodynamic instability",
        ],
    }


def breakdown():
    result = []
    for idx, g in enumerate(UCD_GENES):
        pts = g["patients"]
        # Count primary etiology distribution
        ec = {}
        for p in pts:
            et = (p.get("clinical_stage") or
                  ("severe_neonatal" if p.get("severe_neonatal") else None) or
                  ("spastic_diplegia" if p.get("spastic_diplegia") else None) or
                  ("ncg_response" if p.get("ncg_response_dramatic") else None) or
                  ("homocitrullinuria" if p.get("homocitrullinuria") else None) or
                  "other")
            ec[et] = ec.get(et, 0) + 1
        result.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "omim_disease": g["omim_disease"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "key_alerts": g["key_alerts"],
            "etiologies": g["etiologies"],
            "stats": g["stats"],
            "dx_delay_distribution": g["dx_delay_distribution"],
            "etiology_counts": ec,
            "computed": {
                "n_patients": len(pts),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": pts[:10],
        })
    return result


def definitions():
    return {
        "concepts": {
            "Urea Cycle — Biochemistry, Enzymes and Flux": (
                "The urea cycle is the primary detoxification pathway for ammonia in mammals, "
                "operating across hepatocyte mitochondria and cytoplasm. "
                "Ammonia (NH3) derives from amino acid catabolism (transamination + deamination), "
                "intestinal bacteria (gut urease), purine catabolism, and glutamine hydrolysis. "
                "The cycle consists of five enzymatic steps and two transport proteins: "
                "STEP 1 (mitochondria): NH3 + HCO3- + 2 ATP → carbamoyl phosphate (CPS1; requires NAGS/NAG cofactor); "
                "STEP 2 (mitochondria): carbamoyl phosphate + ornithine → citrulline (OTC); "
                "TRANSPORT: citrulline exits mitochondria (SLC25A15/citrin exchange); "
                "STEP 3 (cytoplasm): citrulline + aspartate → argininosuccinate (ASS1); "
                "STEP 4 (cytoplasm): argininosuccinate → arginine + fumarate (ASL); "
                "STEP 5 (cytoplasm): arginine → ornithine + urea (ARG1; urea excreted by kidneys); "
                "TRANSPORT: ornithine re-enters mitochondria via SLC25A15 (ORC1). "
                "DAILY FLUX: healthy adult produces ~30 g urea/day from 80-100 g dietary protein; "
                "cycle capacity greatly exceeds normal load — significant reserve. "
                "ENERGY COST: 4 ATP per urea molecule produced. "
                "REGULATORY CONTROL: NAGS is the rate-limiting enzyme; "
                "arginine activates NAGS (positive feedback); "
                "NAG is the obligate CPS1 activator; "
                "N-carbamylglutamate (NCG) is a stable NAG analogue that activates CPS1 directly. "
                "ALTERNATIVE WASTE NITROGEN PATHWAYS: "
                "Sodium benzoate: conjugates glycine → hippurate (excreted by kidney; removes 1 N per benzoate); "
                "Sodium phenylacetate: conjugates glutamine → phenylacetylglutamine (excreted; removes 2 N per phenylacetate); "
                "Together these provide ~80% of urea cycle capacity for nitrogen excretion; "
                "combined Ammonul formulation available IV for acute crisis. "
                "AMMONIA TOXICITY MECHANISM: "
                "NH3 freely crosses blood-brain barrier → astrocyte swelling (osmotic) → cerebral oedema; "
                "astrocyte glutamine synthetase converts ammonia + glutamate → glutamine "
                "→ intracellular osmolyte → astrocyte swelling; "
                "ammonia also impairs glutamate neurotransmission, mitochondrial function, oxidative phosphorylation. "
                "THRESHOLD FOR EMERGENCY: ammonia >150 µmol/L in neonate → acute medical emergency; "
                ">300-500 µmol/L with encephalopathy → immediate haemodialysis."
            ),
            "Hyperammonemia — Emergency Recognition and Staged Management": (
                "RECOGNITION: "
                "Neonate (0-30 days): ammonia >100 µmol/L (upper limit 80 µmol/L full-term) is ABNORMAL; "
                ">150 µmol/L = emergency; presentation: poor feeding, lethargy, vomiting, "
                "tachypnoea (respiratory alkalosis — ammonia drives hyperventilation → low pCO2); "
                "hypothermia (not fever); "
                "encephalopathy → coma → herniation. "
                "INITIAL STABILISATION: "
                "STOP all protein intake IMMEDIATELY (max 24-48 hours then reintroduce); "
                "IV dextrose 10% ± lipid emulsion at 2x maintenance — prevents catabolism; "
                "correct glucose (hypoglycaemia common in UCD); "
                "correct electrolytes; "
                "EMPIRIC TREATMENT (before diagnosis confirmed): "
                "IV sodium benzoate 250 mg/kg + sodium phenylacetate 250 mg/kg loading over 90 min "
                "(Ammonul formulation); "
                "IV L-arginine 200 mg/kg (provides arginine depleted in proximal UCDs); "
                "IV L-citrulline 100 mg/kg if CPS1/NAGS suspected (absent citrulline); "
                "Vitamin B12, biotin (if MSUD/organic acidaemia in differential). "
                "DIALYSIS THRESHOLD AND MODALITY: "
                "Ammonia >300-500 µmol/L with encephalopathy, or rapidly rising → HAEMODIALYSIS immediately; "
                "HD clears ammonia 10x faster than peritoneal dialysis; "
                "peritoneal dialysis is CONTRAINDICATED (inadequate clearance rate); "
                "CRRT acceptable if HD not immediately available or haemodynamically unstable; "
                "continue CRRT until ammonia <100 µmol/L. "
                "ONGOING MONITORING: "
                "Ammonia every 1-2 hours during acute phase; "
                "plasma amino acids daily (guide protein reintroduction); "
                "blood glucose, electrolytes, LFTs 4-hourly; "
                "AVOID: opioids (reduce ventilation), high-dose steroids (catabolic), "
                "valproate (inhibits OTC and CPS1). "
                "TARGET: ammonia <80 µmol/L before protein reintroduction; "
                "protein reintroduction MUST happen within 48h to prevent catabolism "
                "→ start at 0.5 g/kg/day and titrate up with ammonia monitoring."
            ),
            "Newborn Screening (NBS) for Urea Cycle Disorders": (
                "NBS via dried blood spot (DBS) tandem mass spectrometry detects several UCDs: "
                "CITRULLINE: elevated in ASS1 (CTLN1) — most reliably detected; "
                "also elevated in ASL, SLC25A13 (CTLN2), and some forms of pyruvate carboxylase deficiency; "
                "very markedly elevated (>500 µmol/L on DBS) is highly specific for ASS1; "
                "ARGININE: elevated in ARG1 — detectable on NBS; "
                "ORNITHINE: elevated in SLC25A15 (HHH); "
                "NOT DETECTED BY STANDARD NBS: OTC, CPS1, NAGS — "
                "these produce low/absent citrulline but citrulline is near-normal in NBS DBS "
                "for OTC deficiency (may be low but not zero on DBS); "
                "family history and clinical presentation remain critical for OTC/CPS1/NAGS. "
                "SECONDARY TARGETS: "
                "Many NBS programmes include urine orotic acid or second-tier tests; "
                "urine metabolomics as second tier is emerging. "
                "ACTION ON ABNORMAL NBS: "
                "Elevated citrulline on DBS → IMMEDIATE referral to metabolic centre; "
                "do NOT wait for confirmatory test — start empiric treatment if symptomatic; "
                "blood plasma amino acids + urine amino acids + urine organic acids + ammonia "
                "within 24 hours of abnormal NBS notification."
            ),
            "Ammonia Scavengers — Mechanism and Clinical Use": (
                "SODIUM BENZOATE: "
                "Hepatic mitochondrial conjugation: benzoate + glycine → hippurate (via glycine N-acyltransferase); "
                "hippurate excreted by kidney (proximal tubule secretion — high capacity); "
                "net effect: 1 mole benzoate removes 1 mole waste nitrogen (as glycine); "
                "glycine is replenished from serine/other amino acids; "
                "DOSE: 250-500 mg/kg/day oral; 250-500 mg/kg loading IV (crisis); "
                "ADVERSE EFFECTS: nausea, vomiting (empty stomach); "
                "metabolic acidosis if renal excretion impaired; "
                "protein binding displaces bilirubin in neonates (caution hyperbilirubinaemia). "
                "SODIUM PHENYLACETATE/PHENYLBUTYRATE: "
                "Phenylbutyrate is a pro-drug (oxidised to phenylacetate in vivo); "
                "phenylacetate conjugates with glutamine → phenylacetylglutamine (PAG); "
                "PAG excreted renally; net effect: 1 mole phenylacetate removes 2 moles waste N "
                "(glutamine contains 2 N atoms — amine + amide); "
                "DOSE: 250-500 mg/kg/day; "
                "COMBINED AMMONUL: sodium benzoate 100 mg/mL + sodium phenylacetate 100 mg/mL IV; "
                "loading dose: 250 mg/kg each; maintenance: 250-500 mg/kg/day each. "
                "GLYCEROL PHENYLBUTYRATE (RAVICTI): "
                "Pro-drug prodrug form of phenylbutyrate; less odour; FDA approved; "
                "less sodium load; preferred for long-term oral use. "
                "CARBAMYLGLUTAMATE (NCG — carglumic acid; Carbaglu): "
                "NOT a true ammonia scavenger — an enzyme activator; "
                "activates CPS1 directly (NAG analogue); "
                "ONLY effective in NAGS deficiency (and secondary NAGS inhibition); "
                "dose: 100-300 mg/kg/day divided 2-4 doses; "
                "if NAGS deficiency: dramatic response within 24-48h; "
                "if no response within 48h → CPS1 deficiency is the diagnosis. "
                "MONITORING SCAVENGER THERAPY: "
                "Plasma amino acids: glycine, glutamine should fall with effective scavenging; "
                "very low glycine → excessive benzoate; "
                "very low glutamine → appropriate (target) or over-treatment; "
                "urine hippurate and PAG as markers of excretion; "
                "protein reintroduction guided by ammonia levels."
            ),
        }
    }
