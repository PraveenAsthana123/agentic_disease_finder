#!/usr/bin/env python3
"""Hereditary-Primary-Hyperaldosteronism-Atlas — Complete 8-Gene Atlas
(KCNJ5 · CLCN2 · CACNA1H · CACNA1D · ATP1A1 · ATP2B3 · ARMC5 · PRKACA).

KCNJ5    (Potassium Inwardly Rectifying Channel J5 / Kir3.4 / GIRK4; 419 aa; ~47 kDa; 11q24.3; AD;
          FH3 (Familial Hyperaldosteronism type 3) — germline KCNJ5 GOF;
          most common somatic mutation in aldosterone-producing adenomas (APAs): 35-40%;
          selectivity filter mutations (G151R/L168R) → depolarisation → Ca²⁺ influx → aldosterone;
          seed SEED_BASE+0).
CLCN2    (Chloride Voltage-Gated Channel 2 / ClC-2; 898 aa; ~99 kDa; 3q27.3; AD;
          FH2 (Familial Hyperaldosteronism type 2) — germline CLCN2 GOF;
          ClC-2 normally hyperpolarising in ZG cells; GOF → depolarisation → aldosterone;
          seed SEED_BASE+1).
CACNA1H  (Calcium Voltage-Gated Channel Subunit Alpha1 H / Cav3.2; 2353 aa; ~262 kDa; 16p13.3; AD;
          FH4 (Familial Hyperaldosteronism type 4) and PASNA syndrome;
          T-type Ca²⁺ channel GOF → increased Ca²⁺ influx → aldosterone synthase activation;
          seed SEED_BASE+2).
CACNA1D  (Calcium Voltage-Gated Channel Subunit Alpha1 D / Cav1.3; 2181 aa; ~250 kDa; 3p14.3; AD;
          PASNA syndrome (Primary Aldosteronism, Seizures, Neurological Abnormalities);
          somatic CACNA1D mutations in ~10% of aldosterone-producing adenomas;
          L-type Ca²⁺ channel gain-of-function;
          seed SEED_BASE+3).
ATP1A1   (ATPase Na+/K+ Transporting Subunit Alpha 1; 1023 aa; ~113 kDa; 1p13.1; AD;
          severe early-onset primary aldosteronism + hypertension;
          germline ATP1A1 mutations (p.Leu104Arg, p.Phe100del) cause ZG-restricted adrenal cortex dysfunction;
          somatic ATP1A1 mutations in ~6% of APAs (second most common somatic driver);
          seed SEED_BASE+4).
ATP2B3   (ATPase Plasma Membrane Ca²⁺ Transporting 3 / PMCA3; 1220 aa; ~137 kDa; Xq28; X-linked;
          somatic ATP2B3 mutations in ~2% of APAs predominantly in men;
          rare germline X-linked cases — male probands with severe early-onset PA;
          altered Ca²⁺ extrusion → sustained intracellular Ca²⁺ → CYP11B2 induction;
          seed SEED_BASE+5).
ARMC5    (Armadillo Repeat Containing 5; 1064 aa; ~120 kDa; 16p11.2; AD;
          BMAH (Bilateral Macronodular Adrenal Hyperplasia) — most common hereditary bilateral PA;
          ~50% of familial BMAH; second hit required (somatic) — biallelic inactivation in nodules;
          aberrant receptor expression (GIP, LH, ADH, beta-adrenergic) drives cortisol AND aldosterone;
          ARMC5 germline + somatic second hit = tumour suppressor model;
          seed SEED_BASE+6).
PRKACA   (Protein Kinase cAMP-Activated Catalytic Subunit Alpha; 351 aa; ~41 kDa; 19p13.12; AD somatic;
          bilateral adrenal cortical hyperplasia with mixed cortisol + aldosterone excess;
          germline PRKACA gain-of-function → Cushing syndrome with bilateral adrenal hyperplasia;
          somatic Leu206Arg mutation in cortisol/aldosterone co-secreting adenomas;
          PKA pathway activation → both CYP17A1 (cortisol) and CYP11B2 (aldosterone) induction;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2502-2509).
"""

import random

SEED_BASE = 2502

PA_GENES = [
    # -- KCNJ5 -- Kir3.4 / GIRK4 -- FH3 / Somatic APA Most Common --------------------------------
    {
        "gene": "KCNJ5",
        "alt_name": (
            "KCNJ5 (KCNJ5-419aa-11q24.3 / AD-GOF-FH3-Germline + Somatic-APA-Most-Common-35-40pct -- "
            "Kir3.4-GIRK4-Potassium-Inwardly-Rectifying-Channel -- "
            "G151R-L168R-Selectivity-Filter-Mutations-PATHOGNOMONIC -- "
            "FH3-Most-Common-Hereditary-PA-After-FH1 -- "
            "Somatic-KCNJ5-APAs-Female-Predominant-Large-Adenomas -- "
            "OMIM-Gene-600734-FH3-613677)"
        ),
        "protein": (
            "KCNJ5 -- 11q24.3 AD-GOF -- KCNJ5-419aa -- "
            "Kir3.4/GIRK4-47kDa-Potassium-Channel-2TM-1P-Domain -- "
            "Zona-Glomerulosa-ZG-Cell-Expressed-Sets-Membrane-Resting-Potential -- "
            "OMIM-Gene-600734"
        ),
        "locus": "11q24.3",
        "protein_size": "419 aa / ~47 kDa",
        "inheritance": (
            "Two clinical contexts with distinct genetics: "
            "(1) GERMLINE AD GOF → FH3 (Familial Hyperaldosteronism type 3): "
            "severe hypertension from early childhood (infancy to adolescence); "
            "bilateral adrenal hyperplasia or bilateral APAs; "
            "KCNJ5 germline mutations (G151R — most common FH3 mutation in Japan/sporadic; "
            "T158A — original Choi 2011 family with gigantism of adrenal glands; "
            "I157S, Y152C — reported in sporadic and familial FH3); "
            "responds poorly to adrenalectomy alone if bilateral hyperplasia; "
            "requires bilateral adrenalectomy + steroid replacement in severe cases. "
            "(2) SOMATIC KCNJ5 mutations → aldosterone-producing adenoma (APA): "
            "most common somatic driver: ~35-40% of APAs; "
            "predominantly affects women (F:M ratio ~3-4:1 for KCNJ5-mutated APAs); "
            "larger adenomas than other somatic mutations; "
            "G151R and L168R most common somatic selectivity filter mutations; "
            "KCNJ5 Kir3.4 normally allows K+ efflux maintaining hyperpolarisation; "
            "GOF mutations alter selectivity filter → Na+ influx instead of K+ → "
            "membrane depolarisation → voltage-gated Ca²⁺ channel opening → "
            "intracellular Ca²⁺ rise → CYP11B2/aldosterone synthase induction."
        ),
        "disease_category": (
            "FH3 (OMIM #613677): AD germline KCNJ5 GOF; childhood-onset severe PA; bilateral adrenal hyperplasia; "
            "KCNJ5-mutated somatic APA: most common subtype; large adenoma; female predominant; "
            "primary hyperaldosteronism with hypertension + hypokalaemia + suppressed renin"
        ),
        "disease_pathway": (
            "KCNJ5 ALDOSTERONE PATHWAY: "
            "Normal ZG cell: Kir3.4 (KCNJ5) channels maintain K+ conductance → "
            "membrane hyperpolarised at approximately -80 mV → "
            "Voltage-gated Ca²⁺ channels (Cav1.3/CACNA1D) remain closed; "
            "angiotensin II (AT1R) or hyperkalaemia → membrane depolarisation → "
            "Ca²⁺ entry → calmodulin/CaMKII → ATF1/CREB → CYP11B2 promoter → aldosterone. "
            "KCNJ5 GOF (selectivity filter mutation G151R/L168R): "
            "K+ selectivity lost → Na+ influx replaces K+ efflux → "
            "membrane CONSTITUTIVELY DEPOLARISED → "
            "Cav channels always open → chronic Ca²⁺ influx → "
            "constitutive aldosterone production independent of normal regulators; "
            "renin suppressed (feedback) → aldosterone-to-renin ratio (ARR) markedly elevated."
        ),
        "pathognomonic": (
            "KCNJ5 CLINICAL PEARLS: "
            "FH3 PRESENTS IN CHILDHOOD/INFANCY: unlike sporadic PA (adults); "
            "bilateral adrenal enlargement on CT — may look like bilateral cortical hyperplasia; "
            "KCNJ5-somatic APAs are LARGEST among somatic APA subtypes — more visible on CT/MRI; "
            "FEMALE PREDOMINANCE for somatic KCNJ5 APAs (3-4:1) — "
            "if young woman with PA and visible adenoma on CT → high probability KCNJ5-mutated; "
            "SOMATIC MUTATION DIAGNOSIS requires adrenal venous sampling (AVS) + "
            "adenoma tissue genetic testing post-adrenalectomy; "
            "AVS MANDATORY before unilateral adrenalectomy — imaging alone unreliable in PA; "
            "POST-ADRENALECTOMY: hypokalaemia resolves rapidly; hypertension improves in ~50% cured, "
            "50% improved — earlier surgery = better BP outcomes; "
            "FH3 GERMLINE: all children of affected parent should be screened; "
            "genetic testing confirms FH3 → bilateral adrenalectomy + steroid replacement + fludrocortisone."
        ),
        "treatment": (
            "PRIMARY HYPERALDOSTERONISM TREATMENT FRAMEWORK: "
            "UNILATERAL APA (including KCNJ5-mutated): unilateral laparoscopic adrenalectomy — "
            "CURE of biochemical PA in >95%; hypertension cure ~50%, improvement ~45%; "
            "BILATERAL PA or MRA-treated PA (without surgery): mineralocorticoid receptor antagonist (MRA); "
            "spironolactone (first-line, 25-400 mg/day) or eplerenone (more selective, fewer sex effects); "
            "target: serum K+ >3.5 mmol/L + blood pressure controlled; "
            "potassium supplementation during MRA titration; "
            "BILATERAL ADRENALECTOMY (FH3 germline): if bilateral hyperplasia unresponsive to MRA; "
            "requires lifelong glucocorticoid + mineralocorticoid replacement; "
            "KCNJ5-specific: no targeted KCNJ5 channel inhibitor clinically available; "
            "MRA is definitive medical treatment."
        ),
        "key_features": [
            "Hypertension (often severe, treatment-resistant, young onset in FH3)",
            "Hypokalaemia (K+ <3.5 mmol/L, often <3.0) — muscle weakness, cramps, polyuria",
            "Suppressed plasma renin activity (PRA <1.0 ng/mL/h) or renin concentration <5 mU/L",
            "Elevated aldosterone (>550 pmol/L or >20 ng/dL when hypokalaemic)",
            "Elevated aldosterone-to-renin ratio (ARR >30-40 ng/dL per ng/mL/h = screening positive)",
            "Bilateral adrenal enlargement (FH3 germline) or unilateral adenoma (somatic APA)",
            "Large adenoma on CT/MRI — KCNJ5-APAs characteristically larger than other somatic types",
        ],
        "key_ddx": {
            "FH1_GRA": (
                "Familial Hyperaldosteronism type 1 / Glucocorticoid-Remediable Aldosteronism (GRA): "
                "hybrid CYP11B1/CYP11B2 gene on unequal crossing-over; "
                "aldosterone production driven by ACTH (not angiotensin II) → suppressible by dexamethasone; "
                "DDx: dexamethasone suppression test → aldosterone normalises in FH1 (not in FH3); "
                "18-hydroxycortisol and 18-oxocortisol markedly elevated in FH1 (urine steroids); "
                "genetic test: Southern blot or long-range PCR for chimeric gene; "
                "KCNJ5 does NOT suppress with dexamethasone."
            ),
            "FH2_CLCN2": (
                "FH2 (CLCN2 GOF): milder PA; potassium usually ≥3.0; "
                "bilateral adrenal disease on AVS — no dominant side; "
                "CT may show bilateral micronodules or be normal; "
                "responds well to spironolactone; "
                "CLCN2 mutations not concentrated in selectivity filter; "
                "genetic panel includes both KCNJ5 and CLCN2 for familial PA workup."
            ),
            "Essential_HTN_With_Hypokalaemia": (
                "ALWAYS screen for PA before diagnosing diuretic-induced hypokalaemia: "
                "stop diuretics 4 weeks before ARR; replace potassium before ARR; "
                "ARR >30-40 on two separate occasions = positive screen → confirmatory test; "
                "confirmatory: oral sodium loading (3-day) or saline infusion test (2L IV 4h) → "
                "aldosterone remains >5 ng/dL after saline = confirmed PA; "
                "KCNJ5-FH3 history: early childhood hypertension in parent + child = FH3 until proven otherwise."
            ),
        },
        "systemic_involvement": (
            "CARDIOVASCULAR: left ventricular hypertrophy disproportionate to blood pressure (aldosterone-mediated); "
            "atrial fibrillation risk 5-12x increased vs essential HTN (independent of BP); "
            "premature cardiovascular events; arterial stiffness; "
            "RENAL: hypertension-mediated nephropathy; proteinuria; KCNJ5 expressed in kidney; "
            "METABOLIC: hypokalaemia causes insulin resistance + glucose intolerance; "
            "metabolic alkalosis (KHCO3 loss in exchange for Na+); "
            "NEUROLOGICAL (FH3 severe cases): intellectual disability, developmental delay reported in severe childhood-onset FH3."
        ),
        "cascade_testing": (
            "FH3 GERMLINE CASCADE: 50% of first-degree relatives affected; "
            "test children of FH3 parent from infancy (blood pressure + ARR from age 3-5); "
            "early diagnosis and MRA or adrenalectomy prevents end-organ damage; "
            "paediatric endocrinologist referral mandatory for FH3; "
            "SOMATIC APA: post-adrenalectomy somatic testing does not affect family screening; "
            "if patient age <40 + severe PA + family history: always offer germline KCNJ5 testing."
        ),
        "emergency_protocol": (
            "HYPOKALAEMIC PARALYSIS (K+ <2.5 mmol/L): "
            "IV potassium replacement + cardiac monitoring; "
            "start spironolactone 50-200 mg once K+ >3.0; "
            "avoid ACE inhibitors (aldosterone-mediated volume expansion masks response); "
            "HYPERTENSIVE CRISIS with PA: IV nitroprusside or labetalol + IV K+ simultaneously; "
            "DO NOT give loop diuretics (worsen hypokalaemia); "
            "FH3 NEWBORN: measure cord blood K+ + aldosterone if parent has known FH3; "
            "neonatal presentation: failure to thrive, polyuria, severe hypertension — emergency adrenalectomy may be needed."
        ),
    },

    # -- CLCN2 -- ClC-2 Chloride Channel -- FH2 ---------------------------------------------------
    {
        "gene": "CLCN2",
        "alt_name": (
            "CLCN2 (CLCN2-898aa-3q27.3 / AD-GOF-FH2 -- "
            "CLC-2-Chloride-Voltage-Gated-Channel-2 -- "
            "FH2-Bilateral-PA-Milder-Phenotype -- "
            "OMIM-Gene-600570-FH2-605635)"
        ),
        "protein": (
            "CLCN2 -- 3q27.3 AD-GOF -- CLCN2-898aa -- "
            "CLC-2-99kDa-Voltage-Gated-Inwardly-Rectifying-Cl−-Channel -- "
            "ZG-Lateral-Membrane-Expressed-Normally-Stabilises-Hyperpolarised-State -- "
            "OMIM-Gene-600570"
        ),
        "locus": "3q27.3",
        "protein_size": "898 aa / ~99 kDa",
        "inheritance": (
            "Autosomal dominant germline GOF mutations → FH2 (Familial Hyperaldosteronism type 2): "
            "historically FH2 was a heterogeneous category before CLCN2 discovery (Scholl 2018); "
            "CLCN2 GOF mutations include: p.Arg172Gln, p.Thr381Met, p.Leu266Pro (partial list); "
            "mechanism: normally ClC-2 mediates Cl− efflux, maintaining ZG resting potential; "
            "GOF → increased Cl− conductance → depolarisation → Ca²⁺ entry → CYP11B2 induction; "
            "bilateral adrenal involvement (unlike unilateral APA); "
            "milder than FH3 — often normo/mildly hypokalaemic; "
            "ARR elevated but aldosterone levels less dramatically high than FH3; "
            "penetrance variable; some carriers have biochemical PA without overt hypertension; "
            "age of onset: adult (typically 30-50s unlike FH3 childhood onset); "
            "responds well to mineralocorticoid receptor antagonists (spironolactone/eplerenone); "
            "bilateral adrenalectomy rarely required."
        ),
        "disease_category": (
            "FH2 (OMIM #605635): AD CLCN2 GOF; bilateral PA; adult onset; milder than FH3; "
            "hypokalaemia often absent or mild (K+ 3.0-3.5); elevated ARR; responds to MRA"
        ),
        "disease_pathway": (
            "CLCN2 ALDOSTERONE PATHWAY: "
            "ZG cell at rest: ClC-2 (CLCN2) opens at hyperpolarised potentials → "
            "Cl− efflux maintains low intracellular Cl− → augments resting hyperpolarisation; "
            "CLCN2 GOF: channel opens more readily + higher Cl− conductance → "
            "Cl− efflux creates inward positive current equivalent → ZG depolarisation; "
            "depolarisation → Cav3.2 (CACNA1H) and Cav1.3 (CACNA1D) T/L-type Ca²⁺ channels open; "
            "Ca²⁺ influx → calmodulin kinase II (CaMKII) activation → "
            "SF1/ATF1/CREB transcription factors → CYP11B2 promoter → aldosterone synthase; "
            "BILATERAL because ZG throughout both adrenals is affected by germline GOF; "
            "NO dominant adenoma — diffuse bilateral hyperplasia is common."
        ),
        "pathognomonic": (
            "CLCN2 CLINICAL PEARLS: "
            "FH2 was previously a diagnostic exclusion category (familial PA not FH1/FH3/FH4); "
            "CLCN2 accounts for significant proportion of previously-labelled FH2 families; "
            "MILDER phenotype than FH3 — hypokalaemia often absent (K+ may be normal); "
            "ARR elevated but absolute aldosterone often only mildly elevated (10-30 ng/dL range); "
            "BILATERAL DISEASE on AVS (no dominant lateralisation) → guide toward MRA rather than surgery; "
            "PENETRANCE VARIABLE: not all CLCN2 GOF carriers will have clinical PA; "
            "FAMILY SCREENING: ARR in all first-degree relatives of CLCN2 mutation carriers; "
            "SPIRONOLACTONE RESPONSE: excellent — blood pressure normalises; K+ corrects; "
            "AVOID unnecessarily labelling as 'essential hypertension with mild hypokalaemia' — "
            "check ARR in any patient with resistant hypertension + family history of hypertension."
        ),
        "treatment": (
            "FH2 TREATMENT: "
            "MRA is first-line and highly effective: spironolactone 25-200 mg/day; "
            "eplerenone preferred in men (avoids gynecomastia) or women (avoids menstrual irregularity); "
            "target: BP <130/80 + K+ >3.5 without supplements; "
            "bilateral adrenalectomy RARELY needed in FH2 — reserve for MRA intolerance; "
            "annual monitoring: ARR + K+ + BP + renal function; "
            "amiloride (K+-sparing diuretic, ENaC blocker) is alternative if MRA not tolerated; "
            "finerenone (newer non-steroidal MRA) — emerging option; "
            "no selective CLCN2 channel inhibitor currently available."
        ),
        "key_features": [
            "Bilateral PA — diffuse bilateral adrenal hyperplasia or bilateral micronodules on CT",
            "Adult-onset hypertension (30-50s) — milder than FH3",
            "Hypokalaemia often absent or mild (K+ 3.0-3.5 mmol/L)",
            "Elevated ARR (typically >30-40 ng/dL per ng/mL/h) on confirmatory testing",
            "Variable penetrance — some carriers biochemically positive without hypertension",
            "No dominant unilateral lesion on AVS (bilateral excess equally from both sides)",
            "Family history of hypertension across multiple generations (AD inheritance)",
        ],
        "key_ddx": {
            "FH3_KCNJ5": (
                "FH3 (KCNJ5): more severe, childhood onset, more profound hypokalaemia, "
                "bilateral hyperplasia often massive (CT enlarged glands); "
                "FH2/CLCN2: adult onset, milder, K+ often normal, glands may appear normal on CT; "
                "genetic panel distinguishes."
            ),
            "Sporadic_Bilateral_PA": (
                "Primary bilateral PA without family history — AVS confirms bilateral excess; "
                "treated with MRA same as FH2; "
                "genetic testing recommended in bilateral PA age <40 or positive family history; "
                "CLCN2 identified in some apparently sporadic bilateral PA cases."
            ),
            "Secondary_Hyperaldosteronism": (
                "Renin ELEVATED in secondary hyperaldosteronism (renal artery stenosis, heart failure, cirrhosis); "
                "PRIMARY PA: renin suppressed; ARR markedly elevated; "
                "measure sitting renin + aldosterone after 15 min rest; "
                "stop aldosterone antagonists 6 weeks and beta-blockers 2 weeks before testing."
            ),
        },
        "systemic_involvement": (
            "CARDIOVASCULAR: similar to sporadic PA — LVH, AF risk, arterial stiffness; "
            "milder than FH3 given less severe biochemistry; "
            "RENAL: proteinuria and early nephropathy if untreated; "
            "ENDOCRINE: no extra-adrenal features in FH2 (unlike FH4/PASNA); "
            "NEUROLOGICAL: none beyond hypertension-mediated risk."
        ),
        "cascade_testing": (
            "First-degree relatives of CLCN2 mutation carriers: ARR + genetic testing; "
            "penetrance incomplete — genetic result plus biochemistry needed to determine clinical PA; "
            "children: measure BP from age 5; ARR from age 10-15; "
            "if ARR positive: confirmatory saline infusion test → if confirmed PA: start MRA."
        ),
        "emergency_protocol": (
            "Rare acute emergency in FH2 given milder phenotype; "
            "if K+ <2.8 mmol/L: urgent IV K+ replacement + cardiac monitoring; "
            "start spironolactone 100-200 mg after K+ >3.0; "
            "AVOID stopping MRA abruptly in established FH2 — aldosterone rebound."
        ),
    },

    # -- CACNA1H -- Cav3.2 T-type Calcium Channel -- FH4 / PASNA --------------------------------
    {
        "gene": "CACNA1H",
        "alt_name": (
            "CACNA1H (CACNA1H-2353aa-16p13.3 / AD-GOF-FH4-PASNA -- "
            "Cav3.2-T-type-Ca2+-Channel-Voltage-Gated -- "
            "FH4-Childhood-PA-Seizures-PASNA -- "
            "OMIM-Gene-607904-FH4-617027)"
        ),
        "protein": (
            "CACNA1H -- 16p13.3 AD-GOF -- CACNA1H-2353aa -- "
            "Cav3.2-T-type-262kDa-Voltage-Gated-Ca2+-Channel-Low-Threshold-Activation -- "
            "ZG-Expressed-Depolarisation-Amplifier-Ca2+-Entry -- "
            "OMIM-Gene-607904"
        ),
        "locus": "16p13.3",
        "protein_size": "2353 aa / ~262 kDa",
        "inheritance": (
            "Autosomal dominant germline GOF mutations → FH4 (primary presentation) and PASNA: "
            "FH4 (OMIM #617027): Familial Hyperaldosteronism type 4; "
            "early-onset severe PA (childhood); bilateral adrenal hyperplasia; "
            "severe hypertension; often profound hypokalaemia; "
            "PASNA (Primary Aldosteronism, Seizures, and Neurological Abnormalities): "
            "PA + epilepsy (various seizure types) + neurodevelopmental abnormalities; "
            "may overlap with/distinct from FH4 depending on mutation; "
            "key mutations: p.Met1549Val, p.Phe270Leu, p.Ser196Leu, p.Thr455Met; "
            "Cav3.2 T-type Ca²⁺ channels activate at lower membrane potentials (low-threshold); "
            "GOF → channels activate more easily and/or inactivate slower → "
            "greater Ca²⁺ flux at subthreshold potentials → constitutive aldosterone synthesis. "
            "Both adrenal and neuronal Cav3.2 expression explains dual adrenal + CNS phenotype in PASNA."
        ),
        "disease_category": (
            "FH4 (OMIM #617027): AD CACNA1H GOF; childhood PA; bilateral adrenal hyperplasia; severe BP; "
            "PASNA: PA + epilepsy + neurodevelopmental delay; Cav3.2 neuronal expression causes neurological features"
        ),
        "disease_pathway": (
            "CACNA1H/Cav3.2 PATHWAY: "
            "T-type Ca²⁺ channels (Cav3.1/Cav3.2/Cav3.3) have LOW THRESHOLD for activation (~-55 mV); "
            "ZG cells near their threshold potential — small depolarisation → T-type channels open; "
            "CACNA1H GOF: window current increased (activation shifted negative + inactivation shifted positive); "
            "ZG cells experience sustained low-level Ca²⁺ influx even at resting potential; "
            "Ca²⁺ → CaMKII → ATF1/SF1/CREB → CYP11B2 transcription → aldosterone; "
            "CNS effect (PASNA): hippocampal and cortical neurons also express Cav3.2; "
            "GOF in CNS → hyperexcitability → seizure threshold lowered; "
            "developmental neurotoxicity from chronic Cav3.2 GOF during brain maturation; "
            "MIBEFRADIL (T-type blocker): reduces aldosterone in vitro; not clinically approved for PA."
        ),
        "pathognomonic": (
            "CACNA1H CLINICAL PEARLS: "
            "FH4 MIMICS FH3 clinically: distinguish by genetic panel; "
            "PASNA TRIAD: PA + SEIZURES + NEURODEVELOPMENTAL DELAY = CACNA1H until proven otherwise; "
            "SEIZURES in PA patient under age 30 + family history of hypertension → CACNA1H panel; "
            "BILATERAL ADRENAL HYPERPLASIA — AVS shows bilateral excess without dominant side; "
            "CHILDREN with severe resistant hypertension + hypokalaemia + seizures: "
            "check ARR early and include CACNA1H in genetic panel; "
            "T-TYPE CHANNEL ANTAGONISTS: ethosuximide and zonisamide (antiepileptics that block Cav3.2) "
            "may reduce aldosterone AND seizures — dual benefit but not formally approved for PA; "
            "MRA (spironolactone) controls aldosterone excess; separate antiepileptic for seizures; "
            "PROGNOSIS: seizure control often achieved with antiepileptics; PA managed with MRA ± surgery."
        ),
        "treatment": (
            "FH4/PASNA TREATMENT: "
            "MRA (spironolactone 50-400 mg) for PA component — effective in bilateral disease; "
            "eplerenone if sex-hormone side effects problematic; "
            "ANTIEPILEPTICS for seizures: ethosuximide (Cav3.2 blocker) may provide dual benefit; "
            "valproate, levetiracetam used per seizure type; "
            "neurodevelopmental support: early intervention for developmental delay; "
            "bilateral adrenalectomy: reserved for refractory PA with severe bilateral hyperplasia; "
            "no selective CACNA1H inhibitor approved for clinical use."
        ),
        "key_features": [
            "Childhood-onset severe hypertension (may present in first decade)",
            "Bilateral adrenal hyperplasia (not unilateral adenoma) on CT",
            "Severe hypokalaemia (K+ often <2.5 mmol/L)",
            "Seizures in PASNA — various types (absence, focal, generalised)",
            "Neurodevelopmental delay / intellectual disability in PASNA",
            "Suppressed renin + elevated aldosterone + elevated ARR",
            "Poor response to usual antihypertensives — MRA required",
        ],
        "key_ddx": {
            "FH3_KCNJ5_vs_FH4_CACNA1H": (
                "Both: childhood onset, bilateral, severe; "
                "FH3: no seizures/neurodevelopmental features; KCNJ5 selectivity filter mutations; "
                "FH4/PASNA: seizures + developmental delay = CACNA1H; "
                "genetic panel distinguishes — always test BOTH in childhood-onset severe PA."
            ),
            "Epilepsy_with_Hypertension": (
                "If child has seizures + hypertension: check ARR before treating epilepsy alone; "
                "undiagnosed PA can cause severe hypertension-mediated encephalopathy mimicking epilepsy; "
                "in PASNA: PA itself contributes to seizure threshold via electrolyte imbalance + vascular."
            ),
            "NF1_or_pheochromocytoma": (
                "Hypertension + neurological features in childhood → exclude phaeochromocytoma (plasma metanephrines); "
                "PA (elevated ARR with suppressed renin) distinguishes from phaeochromocytoma (catecholamine excess); "
                "both can cause severe childhood hypertension."
            ),
        },
        "systemic_involvement": (
            "NEUROLOGICAL (PASNA): seizures, intellectual disability, ADHD, autism spectrum features; "
            "CARDIOVASCULAR: severe childhood hypertension → LVH, premature CVD if untreated; "
            "ADRENAL: bilateral cortical hyperplasia — adrenal incidentaloma workup should include ARR; "
            "RENAL: hypertensive nephropathy from uncontrolled BP; "
            "no consistent skeletal, dermatological, or ophthalmic involvement."
        ),
        "cascade_testing": (
            "First-degree relatives of CACNA1H mutation carriers: ARR + genetic testing; "
            "any child with seizures from affected family: ARR before antiepileptics alone; "
            "prenatal: de novo mutations possible — not all cases are familial; "
            "EEG + ARR together in paediatric referrals with PA + neurological features."
        ),
        "emergency_protocol": (
            "STATUS EPILEPTICUS in PASNA: standard benzodiazepine protocol + "
            "IMMEDIATELY check serum K+ and correct if <2.5 mmol/L (hypokalaemia lowers seizure threshold); "
            "IV K+ while arranging ARR + aldosterone; "
            "start IV antihypertensive (labetalol/hydralazine) if BP >160/110 in child; "
            "spironolactone once K+ stable."
        ),
    },

    # -- CACNA1D -- Cav1.3 L-type Calcium Channel -- PASNA / Somatic APA -------------------------
    {
        "gene": "CACNA1D",
        "alt_name": (
            "CACNA1D (CACNA1D-2181aa-3p14.3 / AD-GOF-PASNA + Somatic-APA-~10pct -- "
            "Cav1.3-L-type-Ca2+-Channel-Voltage-Gated -- "
            "PASNA-Primary-Aldosteronism-Seizures-Neurological-Abnormalities -- "
            "OMIM-Gene-114206-PASNA-615474)"
        ),
        "protein": (
            "CACNA1D -- 3p14.3 AD-GOF -- CACNA1D-2181aa -- "
            "Cav1.3-L-type-250kDa-Voltage-Gated-Ca2+-Channel-Intermediate-Threshold-Activation -- "
            "ZG-and-Neuronal-Expression-Drives-Aldosterone-Synthesis-and-CNS-Excitability -- "
            "OMIM-Gene-114206"
        ),
        "locus": "3p14.3",
        "protein_size": "2181 aa / ~250 kDa",
        "inheritance": (
            "Two contexts: "
            "(1) GERMLINE AD GOF → PASNA syndrome (OMIM #615474): "
            "Primary Aldosteronism + Seizures + Neurological Abnormalities; "
            "de novo or familial AD GOF mutations; "
            "PASNA originally described for CACNA1D (Scholl 2013); "
            "CACNA1H also causes PASNA overlap → genetic panel includes BOTH; "
            "bilateral adrenal hyperplasia; severe childhood hypertension; "
            "epilepsy (focal or generalised); intellectual disability; autism-spectrum features; "
            "CACNA1D Cav1.3 has LOWER activation threshold than classical L-type Cav1.2; "
            "GOF mutations: p.Gly403Arg (most studied PASNA mutation), p.Phe747Leu, p.Val401Leu. "
            "(2) SOMATIC mutations in APAs: "
            "~10% of aldosterone-producing adenomas carry somatic CACNA1D mutations; "
            "p.Ser652Leu is the most common somatic CACNA1D mutation in APAs; "
            "UNILATERAL adenoma — surgically curable; "
            "somatic CACNA1D APAs: women slightly predominant, smaller adenomas than KCNJ5-APAs. "
            "Cav1.3 (L-type, intermediate threshold) normally contributes to aldosterone secretion "
            "under angiotensin II stimulation; GOF → constitutive Ca²⁺ entry → autonomous aldosterone."
        ),
        "disease_category": (
            "PASNA (OMIM #615474): germline CACNA1D GOF; PA + epilepsy + neurodevelopmental abnormalities; "
            "Somatic CACNA1D APA: ~10% of APAs; unilateral surgically curable; female-slight predominant"
        ),
        "disease_pathway": (
            "CACNA1D/Cav1.3 PATHWAY: "
            "Cav1.3 is an L-type Ca²⁺ channel with activation threshold ~-55 mV (lower than Cav1.2 at -30 mV); "
            "expressed in ZG, sinoatrial node, cochlear hair cells, and neurons; "
            "ZG cell: angiotensin II → AT1R → PLC → IP3 + DAG → ER Ca²⁺ release + small depolarisation; "
            "Cav1.3 OPENS at this intermediate potential → large Ca²⁺ influx → CaMKII → CYP11B2; "
            "CACNA1D GOF: activation shifted negative (channels open at more negative potentials); "
            "slower inactivation (window current increased); "
            "CONSTITUTIVE Ca²⁺ entry independent of angiotensin II → autonomous aldosterone; "
            "CNS: Cav1.3 in hippocampus/thalamus → GOF → seizure generation; "
            "cochlea: Cav1.3 in inner hair cells → sensorineural hearing loss possible in some PASNA cases; "
            "sinoatrial node: Cav1.3 drives pacemaker current → sinus bradycardia or AV block rarely reported."
        ),
        "pathognomonic": (
            "CACNA1D CLINICAL PEARLS: "
            "PASNA: PA + SEIZURES + NEURODEVELOPMENTAL DELAY — ALWAYS include CACNA1D and CACNA1H on panel; "
            "DE NOVO MUTATIONS COMMON IN PASNA — not all cases are familial (sporadic germline); "
            "COCHLEAR INVOLVEMENT: sensorineural hearing loss may occur — audiometry in PASNA; "
            "CARDIAC: sinoatrial node dysfunction (sinus bradycardia) reported — ECG in PASNA; "
            "SOMATIC CACNA1D APAs: smaller than KCNJ5-APAs; may not always be visible on CT; "
            "AVS mandatory — CT unreliable even in smaller adenomas; "
            "L-TYPE BLOCKER: dihydropyridine calcium channel blockers (CCBs — amlodipine, nifedipine) "
            "reduce aldosterone modestly (Cav1.3 is L-type) — useful adjunct in CACNA1D-driven PA; "
            "AVOID verapamil/diltiazem — non-dihydropyridine CCBs worsen bradycardia if Cav1.3 cardiac involvement."
        ),
        "treatment": (
            "PASNA TREATMENT: "
            "MRA for PA; dihydropyridine CCBs may provide modest additional aldosterone reduction; "
            "antiepileptics for seizures (valproate, levetiracetam, lamotrigine); "
            "audiological support if SNHL present; "
            "cardiac monitoring — Holter if sinus bradycardia/AV block; "
            "SOMATIC CACNA1D APA: laparoscopic unilateral adrenalectomy — curative of biochemical PA; "
            "pre-operative potassium optimisation; "
            "post-operative monitoring for transient hypoaldosteronism (contralateral adrenal suppressed)."
        ),
        "key_features": [
            "PA + epilepsy + neurodevelopmental delay (PASNA triad) — de novo or AD",
            "Bilateral adrenal hyperplasia (germline PASNA) OR unilateral APA (somatic)",
            "Possible sensorineural hearing loss (cochlear Cav1.3) — audiometry recommended",
            "Possible sinus bradycardia (sinoatrial node Cav1.3) — ECG in PASNA",
            "Severe childhood hypertension and hypokalaemia in germline PASNA",
            "Somatic APAs: ~10% of APAs; moderate biochemical severity; surgically curable",
        ],
        "key_ddx": {
            "CACNA1H_vs_CACNA1D_PASNA": (
                "Clinically near-identical — both cause PASNA triad; "
                "CACNA1D: may have hearing loss + sinoatrial involvement (check audiogram + ECG); "
                "CACNA1H: less cardiac/audiological involvement; "
                "genetic panel (both genes) required — cannot distinguish clinically."
            ),
            "Timothy_Syndrome_CACNA1C": (
                "CACNA1C (Cav1.2 GOF) → Timothy syndrome: QT prolongation, arrhythmia, autism — "
                "NOT associated with PA; "
                "CACNA1D is the aldosterone-relevant L-type channel, not CACNA1C; "
                "always confirm which CACNA1 gene mutated in reported cases."
            ),
        },
        "systemic_involvement": (
            "NEUROLOGICAL: seizures, intellectual disability, autism-spectrum, ADHD; "
            "AUDIOLOGICAL: SNHL (cochlear Cav1.3 expressed in inner hair cells) — "
            "audiometry recommended at diagnosis; "
            "CARDIAC: sinoatrial node Cav1.3 → possible sinus bradycardia; Holter if symptomatic; "
            "ADRENAL: bilateral cortical hyperplasia (germline) or unilateral APA (somatic); "
            "RENAL: hypertensive nephropathy."
        ),
        "cascade_testing": (
            "De novo mutations common — not always inherited; "
            "if familial: 50% of first-degree relatives at risk; "
            "audiometry in all confirmed CACNA1D carriers (even without symptoms); "
            "ECG in all carriers; ARR in all carriers."
        ),
        "emergency_protocol": (
            "Same as CACNA1H PASNA: K+ correction + antihypertensive + ARR; "
            "ADDITIONALLY: if bradycardia with haemodynamic compromise → transcutaneous pacing risk if Cav1.3 cardiac involvement; "
            "avoid verapamil/diltiazem in CACNA1D PASNA (may exacerbate bradycardia)."
        ),
    },

    # -- ATP1A1 -- Na+/K+ ATPase Alpha-1 -- Severe Germline PA + Somatic APA -------------------
    {
        "gene": "ATP1A1",
        "alt_name": (
            "ATP1A1 (ATP1A1-1023aa-1p13.1 / AD-Germline-Severe-Early-Onset + Somatic-APA-~6pct -- "
            "Na+K+ATPase-Alpha1-Subunit-Electrogenic-Pump -- "
            "Severe-Hypertension-Young-Onset-Germline -- "
            "OMIM-Gene-182310)"
        ),
        "protein": (
            "ATP1A1 -- 1p13.1 AD-Germline/Somatic -- ATP1A1-1023aa -- "
            "Na+K+ATPase-Alpha1-113kDa-Catalytic-Pump-Subunit-P-type-ATPase -- "
            "ZG-Expressed-Normally-Maintains-Na+/K+-Gradient-Membrane-Potential -- "
            "OMIM-Gene-182310"
        ),
        "locus": "1p13.1",
        "protein_size": "1023 aa / ~113 kDa",
        "inheritance": (
            "Two contexts: "
            "(1) GERMLINE AD mutations → severe early-onset primary aldosteronism: "
            "reported germline mutations: p.Leu104Arg, p.Phe100del (among others); "
            "characteristically YOUNG-onset PA (paediatric or young adult); "
            "severe hypertension; profound hypokalaemia; bilateral adrenal cortical disease; "
            "rare but more severe than sporadic PA; "
            "Na+/K+ ATPase pump dysfunction → altered Na+ and K+ gradients → "
            "ZG membrane depolarisation → Ca²⁺ entry → CYP11B2 induction; "
            "similar mechanism to KCNJ5 but through Na+ gradient disruption rather than K+ channel. "
            "(2) SOMATIC mutations in APAs: "
            "~6% of aldosterone-producing adenomas carry somatic ATP1A1 mutations; "
            "second most common somatic driver after KCNJ5 (~35%); "
            "somatic ATP1A1 mutations: p.Leu104Arg most common in somatic APAs also; "
            "associated with small cortisol-cosecretory APAs in some studies; "
            "bilateral AVS often shows unilateral dominance → adrenalectomy curative."
        ),
        "disease_category": (
            "Germline ATP1A1: severe early-onset hereditary PA; bilateral adrenal disease; rare; "
            "Somatic ATP1A1 APA: ~6% of APAs; unilateral; second most common somatic APA driver; "
            "both: hypertension + hypokalaemia + suppressed renin + elevated ARR"
        ),
        "disease_pathway": (
            "ATP1A1/Na+K+ATPase PATHWAY: "
            "Na+/K+ ATPase (3 Na+ out: 2 K+ in per cycle, electrogenic): "
            "maintains low intracellular Na+; maintains high intracellular K+; "
            "contributes to ZG resting membrane potential through electrogenic pump current; "
            "ATP1A1 LOF/dysfunction (pump-of-function mutations altering substrate selectivity): "
            "intracellular Na+ rises (pump less effective at Na+ extrusion); "
            "Na+ accumulation → reversed membrane potential contribution → ZG depolarisation; "
            "OR altered ion binding pocket → Na+ permeability through pump channel mode → "
            "additional depolarising current; "
            "depolarisation → Cav T/L-type channels → Ca²⁺ → CYP11B2 → aldosterone; "
            "pathogenic ATP1A1 mutations cluster around Na+/K+ binding pockets (TM4-TM5-TM6 segments)."
        ),
        "pathognomonic": (
            "ATP1A1 CLINICAL PEARLS: "
            "SOMATIC ATP1A1 APAs may be SMALL — not always visible on CT; "
            "AVS mandatory: ~30% of CT-detected 'normal' adrenals have biochemically dominant unilateral PA; "
            "CORTISOL COSECRETION: ATP1A1 somatic APAs have higher rates of autonomous cortisol cosecretion "
            "(vs KCNJ5-APAs) — measure dexamethasone suppression test pre-operatively; "
            "ATP1A1 APA + cortisol cosecretion: bilateral metabolic risk (glucose, bone, cardiovascular); "
            "post-adrenalectomy: monitor for cortisol insufficiency if cosecretion present; "
            "GERMLINE: very rare — consider in paediatric PA with no KCNJ5/CLCN2/CACNA1H/CACNA1D; "
            "cardiac glycosides (digoxin) target Na+/K+ ATPase → could worsen ATP1A1 germline PA theoretically — "
            "use with caution if digoxin prescribed concomitantly."
        ),
        "treatment": (
            "SOMATIC ATP1A1 APA: laparoscopic adrenalectomy — curative; "
            "pre-operative: correct K+ (spironolactone 50-200 mg) + anti-hypertensives; "
            "if cortisol cosecretion: perioperative hydrocortisone cover; "
            "post-operative: monitor for transient hypoaldosteronism (spironolactone bridge during recovery); "
            "GERMLINE: MRA first-line; bilateral adrenalectomy if refractory; "
            "no ATP1A1-targeted therapy available."
        ),
        "key_features": [
            "Severe PA with young or paediatric onset (germline)",
            "Bilateral adrenal cortical hyperplasia (germline) or small unilateral APA (somatic)",
            "Possible cortisol cosecretion (somatic APAs — check dexamethasone suppression)",
            "Profound hypokalaemia and treatment-resistant hypertension",
            "Small somatic APAs — may be CT-occult; AVS mandatory for lateralisation",
            "Second most common somatic APA driver after KCNJ5",
        ],
        "key_ddx": {
            "KCNJ5_Somatic_vs_ATP1A1_Somatic": (
                "KCNJ5 APAs: larger, more visible on CT, predominantly women, no cortisol cosecretion; "
                "ATP1A1 APAs: smaller, possible cortisol cosecretion, less sex-skewed; "
                "molecular subtyping requires adenoma genetic testing post-adrenalectomy; "
                "clinically indistinguishable pre-operatively — AVS + adrenalectomy + tissue genetics."
            ),
            "Cortisol_Producing_ACA": (
                "Autonomous cortisol secretion in adrenal incidentaloma → 1mg DST non-suppression; "
                "if also PA (elevated ARR): check for cortisol cosecretory APA (ATP1A1 or CACNA1D more likely); "
                "bilateral adrenal adenomas with both cortisol + aldosterone excess → "
                "adrenalectomy of dominant side (AVS) + hydrocortisone cover post-op."
            ),
        },
        "systemic_involvement": (
            "CARDIOVASCULAR: LVH, atrial fibrillation — same as all PA forms; "
            "METABOLIC: if cortisol cosecretion — glucose intolerance, osteoporosis, central adiposity; "
            "RENAL: hypertensive nephropathy; hypokalaemic nephropathy if chronic; "
            "ENDOCRINE: cortisol excess markers if cosecretion (buffalo hump, thin skin, easy bruising)."
        ),
        "cascade_testing": (
            "Somatic ATP1A1 mutations: no cascade testing for family (somatic); "
            "germline ATP1A1: first-degree relatives ARR + genetic testing; "
            "presymptomatic family testing in germline cases — paediatric hypertension check."
        ),
        "emergency_protocol": (
            "Same as other hereditary PA: K+ correction + antihypertensive; "
            "if cortisol cosecretion identified pre-operatively: "
            "hydrocortisone 100 mg IV at induction → 50 mg every 8h for 24h → taper over 5-7 days post-adrenalectomy; "
            "monitor for adrenal insufficiency symptoms (dizziness, nausea, hypotension) after adrenalectomy."
        ),
    },

    # -- ATP2B3 -- PMCA3 Plasma Membrane Calcium Pump -- Somatic APA / X-linked ----------------
    {
        "gene": "ATP2B3",
        "alt_name": (
            "ATP2B3 (ATP2B3-1220aa-Xq28 / X-linked-Somatic-APA-~2pct + Rare-Germline-Male -- "
            "PMCA3-Plasma-Membrane-Ca2+-ATPase-3 -- "
            "Impaired-Ca2+-Extrusion-Sustained-Aldosterone -- "
            "OMIM-Gene-300014)"
        ),
        "protein": (
            "ATP2B3 -- Xq28 X-linked-Somatic/Germline -- ATP2B3-1220aa -- "
            "PMCA3-Plasma-Membrane-Ca2+-ATPase-3-137kDa-ATP-Driven-Ca2+-Extrusion -- "
            "ZG-Expressed-Normally-Terminates-Ca2+-Signal-After-Aldosterone-Stimulus -- "
            "OMIM-Gene-300014"
        ),
        "locus": "Xq28",
        "protein_size": "1220 aa / ~137 kDa",
        "inheritance": (
            "Two contexts: "
            "(1) SOMATIC mutations (predominantly in men): "
            "~2% of aldosterone-producing adenomas carry somatic ATP2B3 mutations; "
            "predominantly affects men (X-linked — hemizygous somatic mutation in males); "
            "PMCA3 normally pumps Ca²⁺ out of ZG cells after aldosterone stimulus terminates; "
            "ATP2B3 LOF → impaired Ca²⁺ extrusion → sustained intracellular Ca²⁺ → "
            "prolonged CYP11B2 activation → excess autonomous aldosterone. "
            "(2) GERMLINE X-linked: "
            "rare germline ATP2B3 mutations in males with severe early-onset PA; "
            "hemizygous males fully affected; carrier females mildly affected or unaffected; "
            "adrenal venous sampling confirms unilateral or bilateral excess; "
            "X-linked inheritance: no male-to-male transmission; "
            "all daughters of affected males are obligate carriers; "
            "~50% of sons of carrier females affected."
        ),
        "disease_category": (
            "Somatic ATP2B3 APA: X-linked ~2% of APAs; predominantly male; unilateral APA; "
            "Germline ATP2B3: rare X-linked hereditary PA; hemizygous males severely affected; "
            "mechanism: impaired Ca²⁺ clearance → prolonged aldosterone production"
        ),
        "disease_pathway": (
            "ATP2B3/PMCA3 PATHWAY: "
            "Normal ZG stimulation: AT1R activation → IP3 → ER Ca²⁺ release → Ca²⁺ influx → aldosterone; "
            "Termination: PMCA3 (ATP2B3) + SERCA pumps restore intracellular Ca²⁺ to basal levels; "
            "ATP2B3 LOF: PMCA3 cannot extrude Ca²⁺ effectively → intracellular Ca²⁺ remains elevated; "
            "sustained Ca²⁺ → persistent CaMKII activity → prolonged ATF1/SF1 transcriptional activation; "
            "CYP11B2 expression maintained constitutively → excess aldosterone production; "
            "ATP2B3 mutations identified in APAs: p.Leu425Pro, p.Leu425_Val426del, p.Ala426Pro — "
            "all affect transmembrane helix region critical for Ca²⁺ transport mechanism."
        ),
        "pathognomonic": (
            "ATP2B3 CLINICAL PEARLS: "
            "MALE PREDOMINANCE: X-linked gene → hemizygous males fully express phenotype; "
            "if young male with PA and no KCNJ5/ATP1A1 mutation found on adenoma: test ATP2B3; "
            "SOMATIC ATP2B3 APAs: similar size to ATP1A1 APAs (smaller than KCNJ5); "
            "AVS mandatory — CT may not see small APA; "
            "GERMLINE CARRIER FEMALES: usually mild or subclinical PA — ARR may be borderline; "
            "X-LINKED PEDIGREE: all sons of carrier female: 50% affected; "
            "all daughters of affected male: obligate carriers; no male-to-male transmission; "
            "DISTINGUISH FROM FH3/KCNJ5: FH3 is AD (affects both sexes); "
            "ATP2B3 germline: X-linked pattern (male predominance, maternal inheritance); "
            "ADRENALECTOMY: curative for somatic ATP2B3 APA (unilateral confirmed on AVS); "
            "post-operative monitoring for hypoaldosteronism."
        ),
        "treatment": (
            "SOMATIC ATP2B3 APA: laparoscopic adrenalectomy — curative biochemical; "
            "pre-operative MRA to control K+ and BP; "
            "GERMLINE ATP2B3 MALE: MRA or adrenalectomy; "
            "CARRIER FEMALES: annual ARR monitoring; treat if biochemically confirmed PA; "
            "no ATP2B3-targeted therapy; general PA management applies."
        ),
        "key_features": [
            "Male predominance (X-linked gene) — somatic in hemizygous males",
            "Unilateral APA on AVS (somatic) — may be small/CT-occult",
            "~2% of APAs — relatively rare somatic subtype",
            "Germline X-linked rare hereditary PA in males",
            "Standard PA biochemistry: suppressed renin, elevated aldosterone, elevated ARR",
            "No extra-adrenal features (unlike CACNA1H/CACNA1D PASNA)",
        ],
        "key_ddx": {
            "KCNJ5_vs_ATP2B3_Somatic": (
                "KCNJ5: female predominant, larger APAs; "
                "ATP2B3: male predominant, smaller APAs, X-linked; "
                "molecular subtyping by adenoma genetic testing distinguishes; "
                "clinically: male patient with PA + small APA → consider ATP2B3."
            ),
            "Hypokalaemia_in_Male_Hypertensive": (
                "Young male with hypertension + hypokalaemia: check ARR; "
                "if ARR positive: consider KCNJ5 (most common), then ATP2B3 and ATP1A1; "
                "X-linked history (maternal hypertension/hypokalaemia in male relatives, "
                "no male-to-male transmission) suggests ATP2B3."
            ),
        },
        "systemic_involvement": (
            "No extra-adrenal systemic features beyond PA complications; "
            "CARDIOVASCULAR: PA-mediated LVH, atrial fibrillation, premature CVD; "
            "RENAL: hypertensive nephropathy; "
            "METABOLIC: hypokalaemia, metabolic alkalosis."
        ),
        "cascade_testing": (
            "Germline ATP2B3 X-linked: maternal carrier testing; "
            "sons of carrier females: 50% affected — measure ARR from adolescence; "
            "daughters of affected males: all obligate carriers — annual ARR monitoring; "
            "no cascade testing needed for somatic mutation (not inherited)."
        ),
        "emergency_protocol": (
            "Hypokalaemic emergency same as other PA: "
            "K+ replacement + cardiac monitoring + MRA; "
            "no unique emergency considerations beyond standard PA management."
        ),
    },

    # -- ARMC5 -- Armadillo Repeat Protein -- BMAH (Bilateral Macronodular Adrenal Hyperplasia) -
    {
        "gene": "ARMC5",
        "alt_name": (
            "ARMC5 (ARMC5-1064aa-16p11.2 / AD-Tumour-Suppressor-BMAH-Most-Common-~50pct -- "
            "Armadillo-Repeat-Containing-Protein-5 -- "
            "Bilateral-Macronodular-Adrenal-Hyperplasia-ACTH-Independent -- "
            "OMIM-Gene-615549-BMAH-615830)"
        ),
        "protein": (
            "ARMC5 -- 16p11.2 AD-LOF -- ARMC5-1064aa -- "
            "Armadillo-Repeat-Protein-5-120kDa-Tumour-Suppressor-Role -- "
            "ZG-and-ZF-Expressed-Regulates-Adrenocortical-Cell-Proliferation-Steroidogenesis -- "
            "OMIM-Gene-615549"
        ),
        "locus": "16p11.2",
        "protein_size": "1064 aa / ~120 kDa",
        "inheritance": (
            "Autosomal dominant LOF (tumour suppressor model — two-hit): "
            "germline ARMC5 mutation (first hit) + somatic second hit within each nodule; "
            "~50% of familial BMAH (bilateral macronodular adrenal hyperplasia) carry germline ARMC5 mutations; "
            "BMAH also known as AIMAH (ACTH-independent macronodular adrenal hyperplasia) or PBMAH; "
            "characterised by bilateral MASSIVE adrenal enlargement with multiple large cortical nodules (>1 cm each); "
            "ABERRANT RECEPTOR EXPRESSION (AbeR): nodules express ectopic receptors for GIP, LH, ADH (V2R), "
            "beta-adrenergic, serotonin, glucagon → food-dependent or posture-dependent cortisol (and sometimes aldosterone); "
            "FOOD-DEPENDENT CORTISOL: classic — cortisol rises AFTER EATING (GIP from gut → GIP receptor on nodule → "
            "cortisol, GIP expression aberrantly gained in ARMC5-BMAH); "
            "variable cortisol excess (subclinical to overt Cushing's), aldosterone excess (PA), "
            "or mixed cortisol+aldosterone phenotype; "
            "penetrance high but variable age of presentation (30-60s typical); "
            "MEN1 and Carney complex excluded in familial BMAH before ARMC5 testing."
        ),
        "disease_category": (
            "BMAH (OMIM #615830): AD ARMC5 LOF; bilateral massive adrenal hyperplasia; "
            "cortisol ± aldosterone excess (ACTH-independent); aberrant receptor expression; "
            "food-dependent cortisol PATHOGNOMONIC; ~50% of familial BMAH"
        ),
        "disease_pathway": (
            "ARMC5 TUMOUR SUPPRESSOR PATHWAY: "
            "ARMC5 protein: armadillo repeat motifs suggest role in Wnt/beta-catenin pathway "
            "and adrenocortical cell fate; "
            "ARMC5 interacts with STT3B (oligosaccharyltransferase) → regulates protein glycosylation; "
            "LOF MECHANISM: biallelic inactivation within each nodule (first hit germline; "
            "second hit somatic — different in each nodule = POLYCLONAL ORIGIN); "
            "Loss of ARMC5 → uncontrolled adrenocortical cell proliferation → "
            "bilateral massive enlargement (adrenals may reach 500g total, normal 4g each); "
            "ABERRANT RECEPTOR: ARMC5 LOF linked to aberrant expression of Gs-coupled receptors; "
            "GIP receptor on nodule → postprandial GIP → cAMP → steroidogenesis; "
            "PKA pathway activation (as in PRKACA) → both CYP17A1 (cortisol) and CYP11B2 (aldosterone)."
        ),
        "pathognomonic": (
            "ARMC5 CLINICAL PEARLS: "
            "BILATERAL MASSIVE ADRENAL ENLARGEMENT on CT — adrenals can be enormous (nodules >3-4 cm each); "
            "FOOD-DEPENDENT CORTISOL: cortisol rises 2-3h after eating = PATHOGNOMONIC for BMAH-GIP receptor; "
            "test: measure cortisol fasting vs 2h after standard meal; "
            "ABERRANT RECEPTOR TESTING: measure cortisol response to: mixed meal, LHRH (LH receptor), "
            "terlipressin (V2R), isoproterenol (beta-AR), glucagon; "
            "identifies which receptor mediates excess → guide pharmacological targeting; "
            "CORTISOL vs ALDOSTERONE EXCESS: ARMC5-BMAH more often cortisol excess; "
            "measure both: 24h urine free cortisol, 1mg DST, serum DHEAS; "
            "AND ARR + aldosterone (PA may coexist with Cushing's); "
            "MEN1 EXCLUSION: MEN1 can also cause bilateral adrenal hyperplasia → test MEN1 before ARMC5 "
            "if concurrent pancreatic/pituitary/parathyroid features; "
            "SUBCLINICAL HYPERCORTISOLISM: most ARMC5-BMAH first detected as bilateral adrenal incidentalomas; "
            "annual follow-up for progression from subclinical to overt Cushing's."
        ),
        "treatment": (
            "ARMC5-BMAH TREATMENT: "
            "SUBCLINICAL/MILD: annual monitoring (UFC, 1mg DST, ARR); treat hypertension/DM/osteoporosis; "
            "OVERT CUSHING: bilateral adrenalectomy (sequential or simultaneous) + lifelong steroid replacement; "
            "UNILATERAL ADRENALECTOMY: debulking; reduces but rarely cures Cushing's (bilateral disease); "
            "MEDICAL: pasireotide (SST5 agonist) if GIP-mediated (less effective); "
            "osilodrostat, metyrapone, ketoconazole (adrenal steroidogenesis inhibitors) for Cushing's; "
            "PA COMPONENT: spironolactone/eplerenone; "
            "ABERRANT RECEPTOR TARGETING: if LH-dependent → LHRH analogue (leuprolide) to suppress LH; "
            "if GIP-dependent → octreotide reduces GIP (indirect); "
            "ARMC5-specific: no targeted therapy available."
        ),
        "key_features": [
            "Bilateral massive adrenal enlargement on CT (multiple nodules >1 cm; total adrenal volume markedly increased)",
            "ACTH-independent cortisol ± aldosterone excess",
            "Food-dependent cortisol — rises after meals (GIP receptor) PATHOGNOMONIC",
            "Aberrant adrenal receptor expression (GIP, LH, ADH, beta-AR)",
            "Subclinical hypercortisolism (bilateral incidentaloma) → evolves to overt Cushing's",
            "AD inheritance with tumour suppressor two-hit mechanism",
            "MEN1, Carney complex, McCune-Albright excluded before ARMC5 testing",
        ],
        "key_ddx": {
            "Cushings_Disease_ACTH_Dependent": (
                "ACTH ELEVATED in Cushing's disease (pituitary adenoma) and ectopic ACTH; "
                "ARMC5-BMAH: ACTH suppressed (ACTH-independent); "
                "measure plasma ACTH (morning): ACTH <2 pmol/L = ACTH-independent → adrenal CT; "
                "bilateral adrenal enlargement + suppressed ACTH → ARMC5 testing + aberrant receptor screen."
            ),
            "PPNAD_Carney_Complex": (
                "PPNAD (primary pigmented nodular adrenocortical disease) — small bilateral nodules (<1 cm); "
                "PRKAR1A mutation (Carney complex); "
                "ARMC5-BMAH: LARGE bilateral nodules; no Carney complex features (lentiginosis, cardiac myxoma); "
                "PRKAR1A genetic testing distinguishes."
            ),
            "MEN1_Associated_Bilateral_Adrenal": (
                "MEN1 (11q13.1 LOF): can cause bilateral adrenal cortical hyperplasia; "
                "check MEN1 features: primary HPT (most common first), pancreatic NETs, pituitary adenoma; "
                "ARMC5: no MEN1 features; pure bilateral adrenal enlargement; "
                "genetic panel: MEN1 then ARMC5 if MEN1 negative."
            ),
        },
        "systemic_involvement": (
            "ENDOCRINE: cortisol excess → DM, osteoporosis, hypertension, weight gain, easy bruising, thin skin; "
            "PA component → hypokalaemia, hypertension; "
            "CARDIOVASCULAR: Cushing's-mediated accelerated atherosclerosis + PA-mediated LVH; "
            "BONE: fragility fractures from cortisol-mediated osteoporosis; "
            "METABOLIC: glucose intolerance to frank diabetes; "
            "PSYCHIATRIC: depression, cognitive impairment (Cushing's); "
            "NO extra-adrenal tumour risk in ARMC5 (unlike MEN1)."
        ),
        "cascade_testing": (
            "First-degree relatives of ARMC5 carriers: adrenal imaging (CT) + ARMC5 genetic testing; "
            "if CT shows bilateral adrenal enlargement: full hormonal evaluation (UFC, DST, ARR); "
            "consider imaging from age 20 in confirmed carriers; "
            "post-adrenalectomy carriers still have risk of contralateral progression — ongoing surveillance."
        ),
        "emergency_protocol": (
            "ADRENAL CRISIS after bilateral adrenalectomy: "
            "hydrocortisone 100 mg IV stat + 200 mg/24h continuous infusion; "
            "normal saline 1L IV over 1h; "
            "monitor glucose + electrolytes; "
            "SICK DAY RULES: triple oral hydrocortisone for fever/illness/surgery; "
            "medical alert bracelet/card mandatory for all post-bilateral-adrenalectomy patients."
        ),
    },

    # -- PRKACA -- PKA Catalytic Subunit Alpha -- Mixed Cortisol/Aldosterone Bilateral HA --------
    {
        "gene": "PRKACA",
        "alt_name": (
            "PRKACA (PRKACA-351aa-19p13.12 / AD-GOF-Bilateral-Adrenal-Hyperplasia-Cortisol+Aldosterone -- "
            "PKA-Catalytic-Subunit-Alpha-cAMP-Signalling -- "
            "Germline-GOF-Bilateral-BAH + Somatic-Leu206Arg-Cortisol-APA -- "
            "OMIM-Gene-601639)"
        ),
        "protein": (
            "PRKACA -- 19p13.12 AD-GOF -- PRKACA-351aa -- "
            "PKA-Catalytic-Alpha-41kDa-cAMP-Activated-Serine-Threonine-Kinase -- "
            "Adrenal-ZF-and-ZG-Expressed-Drives-Steroidogenesis-Downstream-ACTH-Receptor -- "
            "OMIM-Gene-601639"
        ),
        "locus": "19p13.12",
        "protein_size": "351 aa / ~41 kDa",
        "inheritance": (
            "Two contexts: "
            "(1) GERMLINE AD GOF → bilateral adrenal cortical hyperplasia (BAH) with Cushing syndrome: "
            "PRKACA germline mutations (p.Leu205Arg, p.Trp197Gly and others) → "
            "constitutive PKA activation independent of cAMP; "
            "bilateral adrenocortical hyperplasia + Cushing syndrome; "
            "may have concurrent autonomous aldosterone excess (PA); "
            "overlaps with Carney complex but PRKAR1A negative; "
            "Cushing features dominant (cortisol) with variable PA component; "
            "very rare germline cause of bilateral adrenal disease. "
            "(2) SOMATIC GOF → cortisol-producing adenoma (CPA): "
            "PRKACA p.Leu206Arg somatic mutation = most common somatic mutation in cortisol-producing APAs; "
            "~36-40% of overt adrenal Cushing's unilateral adenomas carry PRKACA p.Leu206Arg; "
            "some PRKACA-mutated cortisol adenomas also produce aldosterone (co-secretory); "
            "unilateral adenoma — curative with adrenalectomy; "
            "mechanism: PRKACA GOF → constitutive PKA → phosphorylation of CREB → "
            "CYP17A1 (cortisol) and SF1 → may also activate CYP11B2 (aldosterone) in some adenomas."
        ),
        "disease_category": (
            "Germline PRKACA: rare bilateral BAH with Cushing ± PA; "
            "Somatic PRKACA p.Leu206Arg: most common mutation in overt adrenal Cushing's adenoma; "
            "some co-secrete aldosterone; unilateral CPA — adrenalectomy curative"
        ),
        "disease_pathway": (
            "PRKACA/PKA PATHWAY: "
            "Normal ACTH axis: ACTH → MC2R (ACTH receptor) → Gsα → adenylyl cyclase → cAMP → "
            "cAMP binds regulatory subunits (PRKAR1A/PRKAR2) of PKA holoenzyme → "
            "catalytic subunits (PRKACA/PRKACB) released → activated; "
            "phosphorylate CREB, STAR (StAR cholesterol transporter) → steroidogenesis; "
            "PRKACA GOF (Leu206Arg): leucine in activation loop mutated → "
            "catalytic subunit constitutively active (does NOT need cAMP for release); "
            "ACTH-INDEPENDENT steroidogenesis → cortisol (dominant); "
            "also drives CYP11B2 in ZG-adjacent cells or in co-secretory adenomas → aldosterone; "
            "GERMLINE: whole adrenal constitutively activated → bilateral hyperplasia; "
            "SOMATIC: single clone constitutively activated → unilateral adenoma."
        ),
        "pathognomonic": (
            "PRKACA CLINICAL PEARLS: "
            "CORTISOL IS DOMINANT: PA is secondary/co-secretory finding — always measure both; "
            "SOMATIC PRKACA Leu206Arg: present in majority of overt adrenal Cushing's adenomas → "
            "if adrenal Cushing's confirmed → tissue PRKACA genotyping in adenoma; "
            "PA IN PRKACA-ADENOMA: ~10-15% of PRKACA-mutated adenomas co-secrete aldosterone → "
            "measure ARR in ALL adrenal Cushing's patients pre-operatively; "
            "PERIOPERATIVE CORTISOL COVER MANDATORY: contralateral adrenal chronically suppressed; "
            "hydrocortisone taper 6-12 months post-adrenalectomy; "
            "DEXAMETHASONE SUPPRESSION: FAILS (1mg overnight; 2-day low-dose) in both PRKACA somatic CPA and bilateral BAH; "
            "GERMLINE: very rare — consider in bilateral Cushing's + no CRH/ACTH pituitary source + "
            "PRKAR1A/ARMC5 negative; "
            "PRKACA germline ≠ Carney complex (PRKAR1A = Carney complex)."
        ),
        "treatment": (
            "SOMATIC PRKACA CPA: laparoscopic unilateral adrenalectomy — curative; "
            "mandatory perioperative hydrocortisone cover; "
            "HPA axis recovery: 6-12 months of hydrocortisone taper post-adrenalectomy; "
            "if PA cosecretion: spironolactone pre-operatively; monitor ARR post-operatively; "
            "GERMLINE BILATERAL BAH: bilateral adrenalectomy if overt Cushing's; "
            "steroidogenesis inhibitors (osilodrostat, metyrapone) for Cushing's if surgery deferred; "
            "for PA component: MRA while planning adrenalectomy."
        ),
        "key_features": [
            "Overt adrenal Cushing syndrome (cortisol excess dominant in somatic adenoma)",
            "ACTH-independent cortisol: ACTH suppressed + 1mg DST non-suppression + elevated UFC",
            "Unilateral adrenal adenoma on CT (somatic) or bilateral hyperplasia (germline BAH)",
            "Possible PA co-secretion (~10-15% of somatic PRKACA adenomas)",
            "Perioperative glucocorticoid cover mandatory — contralateral adrenal suppressed",
            "PRKACA p.Leu206Arg: most common mutation in overt adrenal Cushing's adenoma (~36-40%)",
        ],
        "key_ddx": {
            "Cushing_Disease_vs_Adrenal_CPA": (
                "Cushing's disease (pituitary ACTH): ACTH elevated; bilateral adrenal enlargement (stimulated); "
                "fails high-dose DST in ~90%; CRH test → ACTH + cortisol rise in pituitary origin; "
                "PRKACA CPA: ACTH suppressed; unilateral adenoma; no CRH response; "
                "Inferior petrosal sinus sampling if doubt."
            ),
            "Carney_Complex_PRKAR1A": (
                "Carney complex (PRKAR1A LOF): PPNAD (small bilateral nodules); "
                "extra-adrenal features: cardiac myxoma, lentiginosis, schwannoma, thyroid/testicular tumours; "
                "PRKACA: no Carney features; usually unilateral (somatic); "
                "PRKAR1A negative + bilateral cortisol excess → consider PRKACA germline."
            ),
            "ARMC5_vs_PRKACA_Bilateral": (
                "ARMC5-BMAH: massive nodules; food-dependent cortisol; aberrant receptor; "
                "PRKACA germline BAH: smaller nodules; no food-dependent cortisol; pure PKA activation; "
                "molecular genetic testing distinguishes."
            ),
        },
        "systemic_involvement": (
            "CUSHING SYNDROME FEATURES: central obesity, hypertension, DM, osteoporosis, "
            "purple striae, easy bruising, proximal myopathy, thin skin, poor wound healing; "
            "PSYCHIATRIC: depression, anxiety, cognitive impairment; "
            "CARDIOVASCULAR: cortisol-mediated atherosclerosis + PA-mediated LVH if co-secretory; "
            "REPRODUCTIVE: menstrual irregularity, reduced fertility; "
            "IMMUNE: immunosuppression — opportunistic infection risk; "
            "BONE: vertebral fractures from osteoporosis."
        ),
        "cascade_testing": (
            "Somatic PRKACA: no cascade testing (somatic); "
            "germline PRKACA: very rare; first-degree relatives: adrenal CT + hormonal screen (UFC, DST, ARR); "
            "exclude PRKAR1A/Carney first; "
            "if germline confirmed: cardiac echo (cardiac myxoma risk low but exclude Carney)."
        ),
        "emergency_protocol": (
            "ADRENAL CRISIS POST-ADRENALECTOMY: "
            "hydrocortisone 100 mg IV stat + 50 mg IV q8h for 24h → oral taper; "
            "IV saline 1L over 1h; "
            "glucocorticoid emergency card mandatory; "
            "SICK DAY RULES: triple dose hydrocortisone for illness; "
            "BILATERAL ADRENALECTOMY patients: same as post-ARMC5 adrenalectomy — "
            "carry hydrocortisone IM emergency kit."
        ),
    },
]


def _make_cohort(gene_data: dict, seed: int, n: int = 40) -> list:
    r = random.Random(seed)
    gene = gene_data["gene"]

    pres_map = {
        "KCNJ5":   ["severe_hypertension_childhood", "hypertension_with_hypokalaemia",
                    "resistant_hypertension", "incidental_ARR_elevation", "family_screening"],
        "CLCN2":   ["adult_hypertension_mild", "incidental_elevated_ARR", "family_screening",
                    "hypokalaemia_incidental", "resistant_hypertension"],
        "CACNA1H": ["childhood_hypertension_severe", "seizures_with_PA",
                    "neurodevelopmental_delay_plus_hypertension", "bilateral_adrenal_hyperplasia",
                    "family_screening_FH4"],
        "CACNA1D": ["PASNA_triad_PA_seizures_neurological", "childhood_bilateral_PA",
                    "SNHL_plus_PA", "unilateral_APA_small", "de_novo_germline_PASNA"],
        "ATP1A1":  ["severe_young_onset_PA", "unilateral_APA_small_CT_occult",
                    "cortisol_cosecretory_APA", "PA_with_1mg_DST_nonsuppression",
                    "resistant_hypertension_bilateral"],
        "ATP2B3":  ["young_male_PA", "unilateral_APA_male_predominant",
                    "X_linked_PA_family_history", "small_APA_CT_occult_AVS_positive",
                    "incidental_bilateral_AVS_unilateral_dominant"],
        "ARMC5":   ["bilateral_adrenal_incidentaloma", "food_dependent_cortisol",
                    "subclinical_Cushings_bilateral", "bilateral_macronodular_hyperplasia",
                    "bilateral_cortisol_plus_aldosterone_excess"],
        "PRKACA":  ["overt_adrenal_Cushings_syndrome", "bilateral_bilateral_hyperplasia_ACTH_independent",
                    "mixed_cortisol_aldosterone_adenoma", "adrenal_incidentaloma_Cushings",
                    "resistant_Cushings_post_TSS_failure"],
    }

    mgmt_map = {
        "KCNJ5":   ["unilateral_adrenalectomy_APA", "bilateral_adrenalectomy_FH3",
                    "spironolactone_MRA", "eplerenone_MRA", "potassium_replacement_plus_MRA"],
        "CLCN2":   ["spironolactone_MRA_long_term", "eplerenone_MRA",
                    "amiloride_alternative", "annual_ARR_monitoring", "family_cascade_testing"],
        "CACNA1H": ["spironolactone_plus_antiepileptic", "bilateral_adrenalectomy_refractory",
                    "ethosuximide_dual_benefit", "valproate_seizure_control",
                    "eplerenone_plus_levetiracetam"],
        "CACNA1D": ["unilateral_adrenalectomy_somatic_APA", "spironolactone_plus_antiepileptic_PASNA",
                    "CCB_dihydropyridine_adjunct", "audiological_support",
                    "cardiac_monitoring_Holter"],
        "ATP1A1":  ["unilateral_adrenalectomy_APA", "perioperative_hydrocortisone_if_cosecretion",
                    "spironolactone_preop", "DST_cortisol_workup",
                    "post_adrenalectomy_monitoring"],
        "ATP2B3":  ["unilateral_adrenalectomy_AVS_confirmed", "spironolactone_preop_MRA",
                    "carrier_female_annual_ARR", "post_adrenalectomy_hypoaldosteronism_monitoring",
                    "eplerenone_preop"],
        "ARMC5":   ["bilateral_adrenalectomy_overt_Cushings", "steroidogenesis_inhibitor_osilodrostat",
                    "unilateral_debulking_partial", "aberrant_receptor_targeting_octreotide",
                    "annual_surveillance_subclinical"],
        "PRKACA":  ["unilateral_adrenalectomy_CPA", "perioperative_hydrocortisone_cover_mandatory",
                    "HPA_axis_taper_6_12_months", "spironolactone_if_PA_cosecretion",
                    "osilodrostat_if_surgery_deferred"],
    }

    age_ranges = {
        "KCNJ5":   (0, 45),   # FH3 childhood; somatic APA 30-60
        "CLCN2":   (25, 60),  # adult onset
        "CACNA1H": (0, 20),   # childhood/adolescence
        "CACNA1D": (0, 40),   # childhood germline; adult somatic
        "ATP1A1":  (10, 50),  # severe early-onset germline; somatic adult
        "ATP2B3":  (20, 55),  # male APAs adult
        "ARMC5":   (30, 65),  # bilateral BMAH adult
        "PRKACA":  (25, 65),  # Cushing's adenoma
    }

    pres_list = pres_map.get(gene, ["hypertension_with_elevated_ARR"])
    mgmt_list = mgmt_map.get(gene, ["MRA_spironolactone"])
    age_min, age_max = age_ranges.get(gene, (20, 55))

    patients = []
    for i in range(n):
        age_dx = r.randint(age_min, age_max)
        age_curr = age_dx + r.randint(1, 20)
        patients.append({
            "id": f"{gene}-{seed}-{i+1:02d}",
            "gene": gene,
            "age_at_diagnosis": age_dx,
            "age_current": min(age_curr, 75),
            "presentation": r.choice(pres_list),
            "management": r.choice(mgmt_list),
            "outcome": r.choice([
                "biochemical_cure_post_adrenalectomy",
                "controlled_on_MRA",
                "annual_surveillance",
                "bilateral_adrenalectomy_steroid_replacement",
                "MDT_review_ongoing",
            ]),
        })
    return patients


def get_overview() -> dict:
    cohorts = []
    for i, g in enumerate(PA_GENES):
        seed = SEED_BASE + i
        cohort = _make_cohort(g, seed)
        avg_age_dx = round(sum(p["age_at_diagnosis"] for p in cohort) / len(cohort), 1)
        cohorts.append({
            "gene": g["gene"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "disease_summary": g["disease_category"][:120],
            "patients": len(cohort),
            "avg_age_at_dx": avg_age_dx,
        })

    total_patients = sum(c["patients"] for c in cohorts)
    return {
        "atlas": "Hereditary-Primary-Hyperaldosteronism-Atlas",
        "subtitle": (
            "Complete 8-Gene Reference — KCNJ5 · CLCN2 · CACNA1H · CACNA1D · ATP1A1 · ATP2B3 · ARMC5 · PRKACA"
        ),
        "total_patients": total_patients,
        "genes_covered": len(PA_GENES),
        "seed_range": f"{SEED_BASE}–{SEED_BASE + len(PA_GENES) - 1}",
        "cohort_size_per_gene": 40,
        "domain": (
            "Hereditary Primary Hyperaldosteronism & Bilateral Adrenal Disease — "
            "FH1-FH4, PASNA, BMAH, Mixed Cortisol/Aldosterone Secretion, Somatic APAs"
        ),
        "key_diagnostic_tests": [
            "Aldosterone-to-Renin Ratio (ARR): >30-40 ng/dL per ng/mL/h (or equivalent) = positive screen",
            "Plasma aldosterone ≥15 ng/dL + suppressed renin = highly suggestive",
            "Confirmatory: saline infusion test (2L IV NaCl over 4h) — aldosterone >5 ng/dL after = confirmed PA",
            "OR oral sodium loading (3g NaCl/day × 3 days) — urine aldosterone >12 mcg/day = confirmed",
            "Adrenal CT (3mm cuts): identify unilateral adenoma vs bilateral hyperplasia — "
            "CT unreliable in PA (~30% of APAs CT-normal or mislead); AVS MANDATORY before adrenalectomy",
            "Adrenal Venous Sampling (AVS): lateralisation index >4:1 (with ACTH) confirms unilateral side",
            "Stop aldosterone antagonists 6 weeks, beta-blockers 2 weeks, ACE-I/ARB 2 weeks before ARR",
            "Dexamethasone Suppression Test (1mg overnight): if ARMC5/PRKACA suspected — exclude cortisol cosecretion",
            "Genetic panel: KCNJ5, CLCN2, CACNA1H, CACNA1D, ATP1A1, ATP2B3, ARMC5 — in familial PA or age <40",
            "Aberrant receptor testing (ARMC5-BMAH): cortisol response to mixed meal, GnRH, terlipressin, isoproterenol",
            "Food-dependent cortisol test: fasting vs 2h post-meal cortisol — rise ≥50% suggests GIP-receptor BMAH",
            "Urine catecholamines/metanephrines: exclude phaeochromocytoma before PA workup",
        ],
        "key_emergency_rules": [
            "HYPOKALAEMIC PARALYSIS (K+ <2.5): IV K+ replacement + cardiac monitoring + MRA — do NOT give loop diuretics",
            "AVS MANDATORY before unilateral adrenalectomy — CT alone misleads in ~30% of cases",
            "BILATERAL ADRENALECTOMY (ARMC5/PRKACA): lifelong steroid replacement mandatory — crisis risk",
            "PRKACA/ATP1A1 cortisol cosecretion: perioperative hydrocortisone cover — contralateral adrenal suppressed",
            "FH3 CHILDHOOD SEVERE (KCNJ5/CACNA1H): K+ <2.0 + severe BP in child = emergency spironolactone IV K+",
            "PASNA SEIZURES (CACNA1H/CACNA1D): correct K+ urgently — hypokalaemia lowers seizure threshold",
            "DO NOT STOP MRA abruptly in established PA — aldosterone rebound causes severe hypertension",
        ],
        "cohorts": cohorts,
    }


def get_breakdown() -> dict:
    breakdown_by_gene = {}
    for i, g in enumerate(PA_GENES):
        seed = SEED_BASE + i
        cohort = _make_cohort(g, seed)
        presentations = {}
        managements = {}
        outcomes = {}
        for p in cohort:
            pres = p["presentation"]
            presentations[pres] = presentations.get(pres, 0) + 1
            mgmt = p["management"]
            managements[mgmt] = managements.get(mgmt, 0) + 1
            out = p["outcome"]
            outcomes[out] = outcomes.get(out, 0) + 1

        breakdown_by_gene[g["gene"]] = {
            "gene": g["gene"],
            "alt_name": g["alt_name"],
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "disease_category": g["disease_category"],
            "disease_pathway": g["disease_pathway"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "key_features": g["key_features"],
            "key_ddx": g["key_ddx"],
            "systemic_involvement": g["systemic_involvement"],
            "cascade_testing": g["cascade_testing"],
            "emergency_protocol": g["emergency_protocol"],
            "cohort_size": len(cohort),
            "presentation_distribution": presentations,
            "management_distribution": managements,
            "outcome_distribution": outcomes,
            "sample_patients": cohort[:5],
        }
    return {"breakdown_by_gene": breakdown_by_gene, "total_genes": len(PA_GENES)}


def get_definitions() -> dict:
    return {
        "atlas_domain": "Hereditary-Primary-Hyperaldosteronism-Atlas — Complete 8-Gene Aldosterone & Bilateral Adrenal Disease Reference",
        "key_definitions": {
            "Aldosterone_to_Renin_Ratio_ARR": (
                "ARR = plasma aldosterone (ng/dL) / plasma renin activity (ng/mL/h); "
                "positive screen: ARR >30-40 (units: ng/dL per ng/mL/h); "
                "or ARR >91 pmol/L per mU/L (SI) depending on local assay; "
                "measure: seated position after 15-30 min rest; morning (8-10 AM preferred); "
                "optimal conditions: K+ repleted (>3.5 mmol/L); off aldosterone antagonists 6 weeks; "
                "off beta-blockers 2 weeks (raise renin falsely → lower ARR); "
                "off ACE-I/ARB 2 weeks (raise renin); "
                "diuretics stopped 4 weeks; "
                "FALSE POSITIVE ARR: beta-blockers suppress renin → low renin → falsely high ARR; "
                "correct by switching to alpha-blocker (doxazosin) or CCB during washout."
            ),
            "Familial_Hyperaldosteronism_FH_Types": (
                "FH1 (GRA): chimeric CYP11B1::CYP11B2; ACTH-driven aldosterone; dexamethasone suppressible; "
                "diagnosed by Southern blot or long-range PCR (NOT routine sequencing); "
                "FH2 (CLCN2): bilateral bilateral PA; adult onset; milder; MRA responsive; "
                "FH3 (KCNJ5): germline GOF; childhood onset; severe; bilateral hyperplasia; "
                "most common hereditary bilateral severe PA; "
                "FH4 (CACNA1H): childhood; bilateral; T-type Ca²⁺ channel; may co-occur with PASNA; "
                "FH5 — not formally established yet; "
                "PASNA syndrome: CACNA1D or CACNA1H + seizures + neurodevelopmental abnormalities."
            ),
            "Adrenal_Venous_Sampling_AVS": (
                "Gold standard for PA lateralisation before adrenalectomy; "
                "bilateral simultaneous or sequential sampling of adrenal veins (right is difficult — "
                "adrenal vein drains directly into IVC); "
                "ACTH stimulation (cosyntropin 250 mcg IV): increases selectivity and cortisol gradient; "
                "selectivity index: adrenal vein cortisol / IVC cortisol >3 (with ACTH) = adequate sampling; "
                "lateralisation index (LI): dominant:non-dominant aldosterone-to-cortisol ratio; "
                "LI >4 (with ACTH cosyntropin) = unilateral dominant side → adrenalectomy; "
                "LI <3 = bilateral → medical management (MRA); "
                "3-4: equivocal → multidisciplinary review; "
                "CT MISLEADS in ~30%: adenoma visible on CT may be non-functioning; true APA CT-occult; "
                "AVS mandatory in all PA patients being considered for adrenalectomy."
            ),
            "Bilateral_Macronodular_Adrenal_Hyperplasia_BMAH": (
                "ACTH-independent bilateral adrenal cortical hyperplasia with large nodules (>1 cm); "
                "adrenals massively enlarged (total weight may reach 500g vs normal 4g each); "
                "aberrant receptor expression (GIP, LH, ADH, beta-AR, 5-HT) mediates cortisol ± aldosterone; "
                "ARMC5 germline mutation in ~50% of familial BMAH; "
                "two-hit tumour suppressor model: germline LOF + somatic second hit per nodule; "
                "clinical: subclinical to overt Cushing's; PA may coexist; "
                "FOOD-DEPENDENT CORTISOL: GIP receptor → cortisol rises after eating PATHOGNOMONIC; "
                "aberrant receptor screen guides targeted medical therapy; "
                "bilateral adrenalectomy for overt Cushing's (sequential or simultaneous)."
            ),
            "PASNA_Syndrome": (
                "Primary Aldosteronism + Seizures + Neurological Abnormalities; "
                "caused by GOF mutations in CACNA1D (original 2013 Scholl description) or CACNA1H; "
                "adrenal: bilateral cortical hyperplasia → PA; "
                "CNS: Cav3.2 (CACNA1H) or Cav1.3 (CACNA1D) neuronal GOF → seizure threshold lowered; "
                "neurodevelopmental abnormalities: intellectual disability, autism spectrum; "
                "audiological: SNHL in CACNA1D cases (cochlear Cav1.3); "
                "cardiac: sinoatrial node dysfunction in CACNA1D; "
                "treatment: MRA for PA + antiepileptics (ethosuximide — Cav3.2 blocker — dual benefit in CACNA1H); "
                "de novo mutations common (not always familial)."
            ),
            "Mineralocorticoid_Receptor_Antagonists_MRA": (
                "First-line medical treatment for bilateral PA and non-surgical PA; "
                "SPIRONOLACTONE: non-selective MRA; 25-400 mg/day; "
                "sex-hormone side effects: gynaecomastia (men), menstrual irregularity (women); "
                "EPLERENONE: selective MRA; less sex-hormone effects; "
                "100-300 mg/day; preferred in men, women with menstrual symptoms; "
                "FINERENONE: non-steroidal MRA; emerging; less renal side effects in diabetic nephropathy; "
                "TARGET: K+ >3.5 without supplements; BP <130/80; "
                "annual monitoring: ARR (may normalise), K+, creatinine, eGFR; "
                "AMILORIDE: ENaC blocker (not MRA); alternative in pregnancy (spironolactone teratogenic); "
                "PREGNANCY: amiloride preferred; eplerenone only if necessary; spironolactone AVOIDED."
            ),
            "Glucocorticoid_Remediable_Aldosteronism_GRA_FH1": (
                "FH1 / GRA: ACTH drives aldosterone production via chimeric CYP11B1::CYP11B2 gene; "
                "aldosterone production suppressible by exogenous glucocorticoid (dexamethasone); "
                "DIAGNOSTIC TEST: dexamethasone 0.5 mg QID × 2 days → aldosterone normalises (ARR falls); "
                "MARKERS: urinary 18-hydroxycortisol and 18-oxocortisol markedly elevated; "
                "GENETIC DIAGNOSIS: Southern blot or long-range PCR — "
                "chimeric gene NOT detectable by standard Sanger or WES/WGS; specific assay required; "
                "TREATMENT: dexamethasone 0.125-0.25 mg nightly (lowest dose that normalises K+ + BP); "
                "OR MRA alternative; "
                "RISK: hypertension + early stroke risk — treat aggressively; "
                "DISTINGUISH FROM FH3/KCNJ5: FH3 does NOT suppress with dexamethasone."
            ),
            "Primary_Aldosteronism_Screening_Indications": (
                "SCREEN ALL: hypertension Stage 2+ (BP >160/100); resistant hypertension (≥3 drugs); "
                "hypertension + spontaneous or diuretic-induced hypokalaemia; "
                "hypertension + adrenal incidentaloma; "
                "hypertension + sleep apnoea; "
                "hypertension + family history of PA; "
                "PA PREVALENCE: ~10% of all hypertension clinic patients; often underdiagnosed; "
                "ARR is simple outpatient test — stop offending medications first; "
                "POSITIVE ARR: 2 separate positive screens OR one unequivocal + confirmatory test → confirmed PA; "
                "POST-DIAGNOSIS: always refer to endocrinology + AVS if surgical candidate."
            ),
        },
        "clinical_pearls": [
            "ARR >30-40 on 2 occasions = positive screen — do not treat hypertension before confirming PA",
            "AVS MANDATORY before adrenalectomy — CT misleads in 30%; non-functioning CT adenoma common",
            "FH1/GRA: ACTH-driven; aldosterone suppressible by dexamethasone; diagnose by long-range PCR not WES",
            "FH3 (KCNJ5 germline): childhood onset + bilateral hyperplasia + severe hypokalaemia = FH3",
            "PASNA (CACNA1D/CACNA1H): PA + seizures + developmental delay — audiogram + ECG + ARR",
            "ARMC5-BMAH: food-dependent cortisol (rises after eating) PATHOGNOMONIC — check aberrant receptor panel",
            "PRKACA somatic Leu206Arg: most common mutation in overt adrenal Cushing's — always check ARR too",
            "ATP1A1 and ATP2B3 APAs: small, CT-occult — AVS positive but CT-negative; do not cancel AVS",
            "Post-adrenalectomy cortisol cover: mandatory if cortisol cosecretion or bilateral adrenalectomy",
            "MRA during bilateral PA or pre-surgical PA: correct K+ to >3.5 before adrenalectomy",
        ],
    }
