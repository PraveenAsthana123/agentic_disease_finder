#!/usr/bin/env python3
"""Hereditary-Glycogen-Storage-Myopathy-Atlas — Complete 8-Gene Glycogen Storage Disease (GSD)
Muscle Spectrum Atlas
(GAA · PYGM · PFKM · LAMP2 · AGL · GBE1 · GYS1 · PGAM2).

GAA     (Acid alpha-glucosidase; 762 aa; 17q25.3; AR;
         Pompe Disease / GSD-II / Acid Maltase Deficiency;
         CLASSIC INFANTILE: cardiomegaly + hypotonia + respiratory failure PATHOGNOMONIC;
         LATE-ONSET (LOPD): proximal myopathy + respiratory failure, NO cardiomegaly;
         ERT: alglucosidase alfa / avalglucosidase alfa / cipaglucosidase+miglustat;
         CRIM-NEGATIVE: immune tolerance induction MANDATORY with ERT;
         GAA enzyme assay DBS diagnostic gold standard;
         seed SEED_BASE+0).
PYGM    (Myophosphorylase / Glycogen phosphorylase muscle isoform; 842 aa; 11q13.1; AR;
         McArdle Disease / GSD-V / Myophosphorylase Deficiency;
         SECOND WIND PHENOMENON — most pathognomonic feature in all myopathology;
         NO LACTATE RISE on forearm exercise test (normal ammonia rise);
         CK markedly elevated at baseline (10-100×);
         p.Trp798Ter (W797X) most common European founder (65% alleles);
         seed SEED_BASE+1).
PFKM    (Phosphofructokinase muscle isoform; 780 aa; 12q13.11; AR;
         Tarui Disease / GSD-VII / PFK-M deficiency;
         OUT-OF-WIND (carbohydrate load worsens — opposite of second wind);
         HEMOLYTIC ANEMIA + GOUT (hyperuricemia) PATHOGNOMONIC triad with myopathy;
         Ashkenazi Jewish founder: p.Arg370Ter + p.Ala539Thr;
         seed SEED_BASE+2).
LAMP2   (Lysosomal-associated membrane protein 2; 410 aa; Xq24; X-linked dominant;
         Danon Disease;
         TRIAD in males: MASSIVE HCM + MYOPATHY + INTELLECTUAL DISABILITY PATHOGNOMONIC;
         FEMALES: HCM dominant, myopathy mild (X-inactivation);
         WPW PRE-EXCITATION on ECG + HCM pattern PATHOGNOMONIC;
         MANDATORY ICD / cardiac transplant often <30y;
         RETINITIS PIGMENTOSA 70% males;
         seed SEED_BASE+3).
AGL     (Amylo-1,6-glucosidase / Debranching enzyme; 1532 aa; 1p21.2; AR;
         Cori-Forbes Disease / GSD-III / Debranching Enzyme Deficiency;
         IIIa (liver + muscle, 85%) vs IIIb (liver only, 15%);
         LIVER disease childhood: hepatomegaly, fasting hypoglycemia, transaminases;
         MUSCLE disease adults: proximal + distal weakness, CK elevated;
         NO CIRRHOSIS key DDx from GSD-I (Fanconi-Bickel);
         seed SEED_BASE+4).
GBE1    (Glycogen branching enzyme; 702 aa; 3p24.2; AR;
         Andersen Disease / GSD-IV / Branching Enzyme Deficiency;
         MOST SEVERE GSD: classic infantile hepatic cirrhosis → death <5y without LT;
         PAS-POSITIVE POLYGLUCOSAN BODIES biopsy PATHOGNOMONIC;
         ADULT POLYGLUCOSAN BODY DISEASE (APBD): upper + lower motor neuron + dementia;
         seed SEED_BASE+5).
GYS1    (Glycogen synthase 1 muscle isoform; 700 aa; 19q13.33; AR;
         GSD-0a / Muscle Glycogen Synthase Deficiency;
         EXERCISE-INDUCED VENTRICULAR ARRHYTHMIA + SUDDEN DEATH PATHOGNOMONIC;
         LOW muscle glycogen (paradoxical storage disease — cannot synthesize glycogen);
         HCM + ventricular arrhythmia → MANDATORY ICD;
         seed SEED_BASE+6).
PGAM2   (Phosphoglycerate mutase 2 muscle isoform; 254 aa; 7p13; AR;
         GSD-X / Muscle PGAM Deficiency;
         Exercise intolerance + MYOGLOBINURIA after intense exercise;
         TUBULAR AGGREGATES on biopsy;
         African American founder: p.Trp78Stop (W78X) — most common allele;
         NO LACTATE RISE on forearm exercise test;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2246-2253).
"""

import random

SEED_BASE = 2246

GSD_GENES = [
    # -- GAA — Pompe Disease, Acid Maltase Deficiency -----------------------------------
    {
        "gene": "GAA",
        "alt_name": (
            "GAA (GAA-762aa-17q25.3 / AR — Pompe-Disease-GSD-II-Acid-Maltase-Deficiency — "
            "CLASSIC-INFANTILE-Cardiomegaly-Hypotonia-Respiratory-Failure-PATHOGNOMONIC — "
            "LATE-ONSET-LOPD-Proximal-Myopathy-Respiratory-NO-Cardiomegaly — "
            "ERT-Alglucosidase-Avalglucosidase-Cipaglucosidase-Miglustat-FDA-Approved — "
            "CRIM-NEGATIVE-Immune-Tolerance-Induction-MANDATORY)"
        ),
        "protein": (
            "GAA -- 17q25.3 AR -- GAA-762aa -- "
            "Acid-Alpha-1,4-Glucosidase-110kDa-Lysosomal-Glycogen-Hydrolysis -- "
            "GSD-II-OMIM-232300 -- "
            "CLASSIC-INFANTILE-Massive-Cardiomegaly-Profound-Hypotonia-Respiratory-Failure-PATHOGNOMONIC -- "
            "LATE-ONSET-LOPD-Proximal-Limb-Girdle-Diaphragm-Myopathy-No-Cardiomegaly -- "
            "ERT-FDA-APPROVED-Alglucosidase-Alfa-Myozyme-Lumizyme-2006 -- "
            "Avalglucosidase-Alfa-Nexviazyme-2021-Higher-Glycan-M6P-Uptake -- "
            "Cipaglucosidase-Alfa-Plus-Miglustat-Pombiliti-Opfolda-2023 -- "
            "CRIM-NEGATIVE-Cross-Reactive-Immune-Material-ITI-Mandatory-Before-ERT -- "
            "GAA-Enzyme-Assay-DBS-Dried-Blood-Spot-Diagnostic-Gold-Standard -- "
            "NBS-Newborn-Screening-Expanding-Internationally -- "
            "OMIM-Gene-GAA-606800-Disease-GSD-II-232300"
        ),
        "locus": "17q25.3",
        "protein_size": "762 aa / 110 kDa",
        "inheritance": (
            "AR (biallelic LOF); "
            "Classic infantile: severe biallelic null variants; profound GAA deficiency (<1% residual activity); "
            "Late-onset (LOPD): one or both hypomorphic alleles; GAA activity 1-10% residual; "
            "CK: elevated (3-30×), especially LOPD; "
            "Classic infantile onset: birth – 2 months; LOPD: childhood to adulthood (1st-7th decade); "
            "Respiratory failure: LOPD — diaphragm early, FVC supine << sitting (key indicator); "
            "Cardiac: classic infantile = massive cardiomegaly (PR short, high-voltage ECG); LOPD cardiac minimal; "
            "GAA enzyme activity: DBS or leukocytes; <1% = classic; 1-10% = LOPD; "
            "Pseudodeficiency alleles: p.Gly576Ser + p.Glu689Lys in trans → normal GAA activity"
        ),
        "key_features": [
            "CLASSIC INFANTILE: Massive cardiomegaly + profound hypotonia (floppy baby) + respiratory failure — PATHOGNOMONIC triad",
            "SHORT PR interval + HIGH voltage QRS on ECG — classic infantile cardiac signature; PATHOGNOMONIC",
            "LATE-ONSET (LOPD): proximal myopathy (pelvic + shoulder girdle) + respiratory failure; NO cardiomegaly",
            "DIAPHRAGM INVOLVEMENT — FVC supine vs sitting discordance; FVC supine 75-80% of sitting = diaphragm weakness",
            "GAA enzyme assay on DBS — diagnostic; <1% activity = classic infantile; 1-10% = LOPD",
            "CRIM-NEGATIVE patients: cross-reactive immune material absent → antibody response destroys ERT → ITI mandatory",
            "ERT: alglucosidase alfa (2006), avalglucosidase alfa (2021 — 3× higher M6P-receptor uptake), cipaglucosidase+miglustat (2023)",
            "Vacuolar myopathy on biopsy: PAS-positive glycogen accumulation + acid phosphatase positive (lysosomal)",
        ],
        "treatment": (
            "ERT — enzyme replacement therapy: "
            "Alglucosidase alfa (Myozyme/Lumizyme) 20 mg/kg IV biweekly — FDA 2006; "
            "Avalglucosidase alfa (Nexviazyme) 20 mg/kg IV biweekly — FDA 2021 (preferred: 3× M6P density); "
            "Cipaglucosidase alfa + miglustat (Pombiliti + Opfolda) — FDA 2023 (chaperone-ERT co-administration). "
            "CRIM-negative: immune tolerance induction (ITI) with rituximab + methotrexate + IVIG BEFORE first ERT dose. "
            "Respiratory: FVC sitting + supine every 6 months; NIV when FVC <50% or nocturnal desaturation; "
            "diaphragm weakness screen — supine FVC <75% sitting FVC = diaphragmatic involvement. "
            "Cardiac: ECG + echocardiogram at diagnosis; monitor LV mass (classic infantile). "
            "Physiotherapy: strengthening; aquatherapy; PT/OT. "
            "Nutrition: high-protein diet supportive. "
            "NBS: dried blood spot newborn screening — expanding in North America, EU, Asia."
        ),
        "monitoring": [
            "GAA enzyme activity: DBS or leukocytes at diagnosis; post-ERT antibody titres",
            "CRIM status: Western blot before ERT initiation",
            "Respiratory: FVC sitting + supine every 6 months; nocturnal oximetry; sleep study",
            "Cardiac: ECG + echo baseline; every 6-12 months (classic infantile); annually (LOPD)",
            "Motor: 6-minute walk test; timed 10m walk; 4-stair climb; MRC scale",
            "Swallowing: FEES/MBSS annually; SLP assessment",
            "Liver enzymes: ALT/AST/GGT; hepatomegaly monitoring",
            "ERT antibodies: IgG anti-GAA titre at 3, 6, 12 months; watch for declining response",
        ],
    },
    # -- PYGM — McArdle Disease, second wind phenomenon ----------------------------------
    {
        "gene": "PYGM",
        "alt_name": (
            "PYGM (PYGM-842aa-11q13.1 / AR — McArdle-Disease-GSD-V-Myophosphorylase-Deficiency — "
            "SECOND-WIND-PHENOMENON-MOST-PATHOGNOMONIC-Feature-In-Myopathology — "
            "NO-LACTATE-RISE-Forearm-Exercise-Test-Normal-Ammonia — "
            "CK-10-100x-Elevated-Baseline — "
            "p.Trp798Ter-W797X-Most-Common-European-65pct)"
        ),
        "protein": (
            "PYGM -- 11q13.1 AR -- PYGM-842aa -- "
            "Muscle-Glycogen-Phosphorylase-Myophosphorylase-97kDa-Glycogen-Breakdown -- "
            "GSD-V-OMIM-232600 -- "
            "SECOND-WIND-PHENOMENON-PATHOGNOMONIC-Rest-2-3min-Then-Resume-Exercise-Without-Cramps -- "
            "NO-LACTATE-RISE-Forearm-Exercise-Test-DIAGNOSTIC -- "
            "NORMAL-AMMONIA-RISE-Forearm-Exercise-Test-Confirms-Effort -- "
            "CK-10-100x-Elevated-Baseline-PATHOGNOMONIC-for-GSD-V -- "
            "MYOGLOBINURIA-After-Intense-Exercise-Rhabdomyolysis-Risk -- "
            "p.Trp798Ter-W797X-65pct-European-Alleles-Most-Common-Founder -- "
            "AEROBIC-EXERCISE-TRAINING-Paradoxically-IMPROVES-Exercise-Tolerance -- "
            "No-FDA-Approved-Disease-Modifying-Therapy-2026 -- "
            "OMIM-Gene-PYGM-608455-Disease-GSD-V-232600"
        ),
        "locus": "11q13.1",
        "protein_size": "842 aa / 97 kDa",
        "inheritance": (
            "AR (biallelic LOF); "
            "p.Trp798Ter (W797X): most common European allele (~65% alleles); "
            "p.Gly205Ser: 2nd most common; "
            "Spanish founders: c.2392T>C (p.Phe798Leu); "
            "CK: markedly elevated at baseline (10-100× ULN); elevated even at rest; "
            "Onset: childhood-adolescence (exercise intolerance); some only in adulthood; "
            "Fixed weakness: rare — most patients remain ambulant; "
            "Myoglobinuria: after intense anaerobic exercise; AKI risk; "
            "No cardiac involvement"
        ),
        "key_features": [
            "SECOND WIND PHENOMENON — most pathognomonic feature in all myopathology: cramps at 5-10 min exercise → rest 2-3 min → resume without cramps",
            "Second wind mechanism: FFAs + hepatic glucose rescue muscle metabolism (myophosphorylase absent → cannot break down muscle glycogen)",
            "NO LACTATE RISE on forearm ischemic exercise test — diagnostic; normal ammonia rise confirms patient effort",
            "CK markedly elevated at baseline (10-100× ULN) — highest of any muscle disease at rest; exercise causes further spike",
            "MYOGLOBINURIA after intense/anaerobic exercise — red-brown urine; AKI risk; avoid intense exercise",
            "p.Trp798Ter (W797X) — most common European allele (65%); PCR-based screening available",
            "Exercise training paradox: regular aerobic exercise IMPROVES symptoms (increases oxidative capacity)",
            "No FDA-approved therapy; sucrose administration pre-exercise: blunts cramp by providing exogenous glucose",
        ],
        "treatment": (
            "Exercise management: moderate aerobic activity — 30-60 min 3-5×/week improves oxidative capacity. "
            "AVOID intense anaerobic exercise → myoglobinuria + rhabdomyolysis + AKI. "
            "Pre-exercise glucose: sucrose (75g in 500ml) 30 min before planned intense activity — blunts cramps (Level B). "
            "Protein diet: high-protein (1.4-1.5 g/kg/day) — may improve strength; avoid carbohydrate-only loading. "
            "Acute rhabdomyolysis: IV fluids; monitor creatinine; urine myoglobin. "
            "Physiotherapy: aerobic conditioning; graded exercise programme; swimming pool therapy. "
            "Vitamin B6 (pyridoxine): open-label; may marginally help; no controlled trial. "
            "Creatine monohydrate: low-dose trial (60 mg/kg/day); conflicting evidence. "
            "Gene therapy: preclinical stage 2026."
        ),
        "monitoring": [
            "CK: at baseline, post-exercise, annually — markedly elevated even at rest is normal for GSD-V",
            "Renal: creatinine + urinalysis after any myoglobinuria episode; urine myoglobin",
            "Cardiac: ECG baseline; no significant cardiac involvement expected",
            "Exercise test: VO2max testing; 6-minute walk; standardized exercise protocol",
            "Myoglobinuria history: document episodes; identify triggers; educate on AKI risk",
            "Muscle MRI: fat replacement pattern (posterior thigh, soleus characteristic); monitor",
            "Swallowing: rarely affected; SLP if dysphagia reported",
        ],
    },
    # -- PFKM — Tarui Disease, out-of-wind, hemolysis + gout ----------------------------
    {
        "gene": "PFKM",
        "alt_name": (
            "PFKM (PFKM-780aa-12q13.11 / AR — Tarui-Disease-GSD-VII-PFK-M-Deficiency — "
            "OUT-OF-WIND-Carbohydrate-Load-WORSENS-Symptoms-Opposite-McArdle — "
            "HEMOLYTIC-ANEMIA-Plus-GOUT-Hyperuricemia-PATHOGNOMONIC-Triad — "
            "Ashkenazi-Jewish-Founder-p.Arg370Ter-p.Ala539Thr)"
        ),
        "protein": (
            "PFKM -- 12q13.11 AR -- PFKM-780aa -- "
            "Phosphofructokinase-Muscle-Isoform-85kDa-Glycolysis-Rate-Limiting-Enzyme -- "
            "GSD-VII-OMIM-232800 -- "
            "OUT-OF-WIND-High-Carbohydrate-WORSENS-Cramps-OPPOSITE-McArdle-PATHOGNOMONIC -- "
            "HEMOLYTIC-ANEMIA-Partial-PFK-Deficiency-Red-Cells-PATHOGNOMONIC-Jaundice-Bilirubin -- "
            "GOUT-Hyperuricemia-From-Hemolysis-Purine-Release-PATHOGNOMONIC -- "
            "NO-LACTATE-RISE-Forearm-Exercise-Test-Same-McArdle-Diagnostic -- "
            "Ashkenazi-Jewish-Founder-p.Arg370Ter-p.Ala539Thr -- "
            "Japanese-Founder-In-Frame-Exon5-Deletion -- "
            "Fixed-Weakness-Adults-Rare-But-More-Than-McArdle -- "
            "OMIM-Gene-PFKM-610681-Disease-GSD-VII-232800"
        ),
        "locus": "12q13.11",
        "protein_size": "780 aa / 85 kDa",
        "inheritance": (
            "AR (biallelic LOF); "
            "Ashkenazi Jewish founders: p.Arg370Ter (most common) + p.Ala539Thr; "
            "Japanese founder: in-frame exon 5 deletion; "
            "Hemolytic anemia: PFK-M deficiency in RBCs (partial) → hemolysis; "
            "CK: elevated at baseline; further spike with exercise; "
            "Onset: childhood; "
            "No cardiac involvement; "
            "Myoglobinuria: can occur with intense exercise; "
            "Fixed weakness: more common than McArdle in adults"
        ),
        "key_features": [
            "OUT-OF-WIND: carbohydrate load WORSENS exercise tolerance — OPPOSITE of McArdle (second wind); pathognomonic for Tarui",
            "Mechanism: high carbs → glucose enters glycolysis → blocks fatty acid usage → PFKM deficiency worsens FFA competition",
            "HEMOLYTIC ANEMIA — partial PFK deficiency in red cells (RBCs have M + L subunits); jaundice + elevated bilirubin",
            "GOUT (hyperuricemia) — from hemolysis → excess purine release; gout attacks in adults; pathognomonic triad",
            "NO LACTATE RISE on forearm exercise test — same as McArdle but different pre-exercise feeding effect distinguishes them",
            "Ashkenazi Jewish founder mutations: p.Arg370Ter + p.Ala539Thr — targeted sequencing available",
            "Fixed weakness: more common than McArdle in adulthood; distal involvement reported",
            "Myoglobinuria: can occur; AKI risk with intense exercise",
        ],
        "treatment": (
            "Exercise management: avoid intense anaerobic exercise; moderate aerobic conditioning. "
            "AVOID high-carbohydrate meals before exercise (worsens cramps — opposite of McArdle). "
            "Pre-exercise: ketogenic snack or fasting — reduces glycolytic substrate competition. "
            "Gout management: allopurinol (xanthine oxidase inhibitor) — standard therapy for hyperuricemia. "
            "Hemolytic anemia: folic acid supplementation; avoid oxidative drugs; no transfusion unless severe. "
            "Hydration: maintain during exercise to reduce rhabdomyolysis risk. "
            "Myoglobinuria: IV fluids; renal protection (same protocol as McArdle). "
            "No FDA-approved disease-modifying therapy 2026."
        ),
        "monitoring": [
            "CBC: hemoglobin + reticulocyte count; bilirubin — hemolytic anemia surveillance",
            "Uric acid: annual; treat if >6 mg/dL to prevent gout crystallopathy",
            "Renal: creatinine; urinalysis post-exercise; myoglobinuria episodes",
            "CK: baseline; exercise-induced; annually",
            "Liver: bilirubin; LFTs (hemolysis + jaundice monitoring)",
            "Exercise test: standardized protocol; document exercise tolerance; VO2max",
            "Muscle MRI: fat replacement pattern monitoring",
        ],
    },
    # -- LAMP2 — Danon Disease, X-linked dominant ----------------------------------------
    {
        "gene": "LAMP2",
        "alt_name": (
            "LAMP2 (LAMP2-410aa-Xq24 / X-Linked-Dominant — Danon-Disease — "
            "TRIAD-Males-MASSIVE-HCM-MYOPATHY-INTELLECTUAL-DISABILITY-PATHOGNOMONIC — "
            "FEMALES-HCM-Dominant-Myopathy-Mild-X-Inactivation — "
            "WPW-PRE-EXCITATION-ECG-Plus-HCM-PATHOGNOMONIC — "
            "MANDATORY-ICD-Cardiac-Transplant-Often-Under-30y — "
            "RETINITIS-PIGMENTOSA-70pct-Males)"
        ),
        "protein": (
            "LAMP2 -- Xq24 X-Linked-Dominant -- LAMP2-410aa -- "
            "Lysosomal-Associated-Membrane-Protein-2-45kDa-Autophagic-Flux-Glycoprotein -- "
            "Danon-Disease-OMIM-300257 -- "
            "TRIAD-Males-MASSIVE-HCM-CARDIAC-HYPERTROPHY-MYOPATHY-INTELLECTUAL-DISABILITY-PATHOGNOMONIC -- "
            "FEMALES-HCM-Only-Or-Mild-Myopathy-Variable-X-Inactivation -- "
            "WPW-Wolff-Parkinson-White-Pre-Excitation-Pattern-ECG-PATHOGNOMONIC-Plus-HCM -- "
            "MANDATORY-ICD-Arrhythmia-Sudden-Death-Prevention -- "
            "CARDIAC-TRANSPLANT-Males-Under-30-Years-Common -- "
            "RETINITIS-PIGMENTOSA-70pct-Males-Annual-Ophthalmology -- "
            "Autophagic-Vacuolar-Myopathy-LAMP2-Biopsy-Autophagic-Vacuoles -- "
            "OMIM-Gene-LAMP2-309060-Disease-300257"
        ),
        "locus": "Xq24",
        "protein_size": "410 aa / 45 kDa",
        "inheritance": (
            "X-linked dominant (XLD); "
            "Males (hemizygous): severe — classic triad; "
            "Females (heterozygous): variable — HCM dominant; intellectual disability and myopathy variable (X-inactivation); "
            "CK: elevated (myopathy component); males more than females; "
            "Onset: childhood–adolescence males; childhood–adulthood females; "
            "Cardiac: MASSIVE HCM (LV wall thickness >20 mm common in males); "
            "WPW: pre-excitation on ECG pathognomonic combination with HCM; "
            "Retinitis pigmentosa: 70% males, 30% females; "
            "Intellectual disability: males ~90%; females ~40%"
        ),
        "key_features": [
            "MALE TRIAD: MASSIVE HCM + MYOPATHY + INTELLECTUAL DISABILITY — PATHOGNOMONIC for Danon Disease in males",
            "WPW PRE-EXCITATION pattern on ECG PLUS HCM — this ECG + echo combination pathognomonic for Danon",
            "HCM in males: MASSIVE (LV wall >20 mm); rapidly progressive; ICD mandatory for SCD prevention",
            "CARDIAC TRANSPLANT: males often require transplant before age 30 — Danon is most severe X-linked cardiomyopathy",
            "FEMALES: milder disease (X-inactivation); HCM present but less severe; some present with Wolff-Parkinson-White only",
            "RETINITIS PIGMENTOSA: 70% males; progressive visual loss; annual ophthalmology referral mandatory",
            "AUTOPHAGIC VACUOLAR MYOPATHY: LAMP2 is lysosomal membrane protein; loss → autophagic material accumulates in vacuoles",
            "Intellectual disability: males ~90%; cognitive testing at diagnosis; educational support planning",
        ],
        "treatment": (
            "Cardiac — PRIMARY URGENCY: "
            "ICD: implant early (symptomatic arrhythmia OR non-sustained VT OR massive HCM); mandatory in all hemizygous males. "
            "Beta-blockers + ACE-i/ARB for HCM: symptom management; do NOT use in decompensated HF. "
            "Cardiac transplant: evaluate early; consider if EF declining or refractory HF — transplanted hearts do NOT recur Danon. "
            "Mavacamten: NOT indicated in Danon (mechanism different from sarcomeric HCM). "
            "Arrhythmia: ablation for accessory pathway (WPW) — reduce arrhythmia burden; ICD still mandatory. "
            "Retinitis pigmentosa: low vision aids; vitamin A palmitate evidence-limited; annual ophthalmology. "
            "Myopathy: physiotherapy; mobility aids; avoid vigorous exercise. "
            "Intellectual disability: special education; occupational therapy; cognitive support. "
            "Gene therapy (LAMP2B): clinical trials ongoing 2026 — MYVAL trial phase 1/2."
        ),
        "monitoring": [
            "Cardiac: ECG + Holter + echo EVERY 6-12 MONTHS — rapid HCM progression in males",
            "ICD interrogation: device check every 3-6 months; arrhythmia log",
            "Ophthalmology: annual fundoscopy + ERG; visual field testing — retinitis pigmentosa surveillance",
            "CK: baseline + annually; reflects myopathy severity",
            "Neuropsychological: cognitive assessment at diagnosis; educational planning",
            "Genetic: family cascade — all female relatives at risk (XLD); prenatal counseling",
            "Transplant eligibility: annual transplant centre review for males from diagnosis",
            "Exercise ECG or Holter: pre-exercise clearance; SCD risk stratification",
        ],
    },
    # -- AGL — Cori-Forbes Disease, GSD-III, debranching enzyme -------------------------
    {
        "gene": "AGL",
        "alt_name": (
            "AGL (AGL-1532aa-1p21.2 / AR — Cori-Forbes-Disease-GSD-III-Debranching-Enzyme — "
            "GSDIIIa-Liver-Plus-Muscle-85pct-vs-GSDIIIb-Liver-Only-15pct — "
            "LIVER-Disease-Childhood-Hepatomegaly-Fasting-Hypoglycemia-Transaminases — "
            "MUSCLE-Disease-Adults-Proximal-Distal-Weakness-CK-Elevated — "
            "NO-CIRRHOSIS-Key-DDx-GSD-I)"
        ),
        "protein": (
            "AGL -- 1p21.2 AR -- AGL-1532aa -- "
            "Amylo-1,6-Glucosidase-4-Alpha-Glucanotransferase-175kDa-Debranching-Enzyme -- "
            "GSD-III-OMIM-232400 -- "
            "GSDIIIa-85pct-Liver-Plus-Muscle-Involvement -- "
            "GSDIIIb-15pct-Liver-Only-Specific-Exon-3-Mutations -- "
            "LIVER-CHILDHOOD-Hepatomegaly-Fasting-Hypoglycemia-Short-Stature-Transaminases -- "
            "LIVER-IMPROVES-Puberty-Key-Feature -- "
            "MUSCLE-ADULTS-Proximal-Distal-Myopathy-CK-Elevated -- "
            "NO-CIRRHOSIS-KEY-DDx-GSD-I-Fanconi-Bickel -- "
            "Raw-Cornstarch-Fasting-Prevention-Standard-Dietary-Therapy -- "
            "OMIM-Gene-AGL-610860-Disease-GSD-III-232400"
        ),
        "locus": "1p21.2",
        "protein_size": "1532 aa / 175 kDa",
        "inheritance": (
            "AR (biallelic LOF); "
            "IIIa vs IIIb: p.Arg-1228* and specific exon 3 mutations → IIIb (liver only); most = IIIa (liver + muscle); "
            "CK: variable — elevated in IIIa muscle involvement; normal in IIIb; "
            "Liver disease: hepatomegaly; transaminases elevated (ALT/AST); fasting hypoglycemia; short stature in childhood; "
            "Liver IMPROVES after puberty — glycogen storage load decreases; "
            "Muscle disease: adults (3rd-5th decade) — proximal + distal weakness, CK markedly elevated; "
            "No cirrhosis (unlike GSD-I, GSD-IV) — crucial diagnostic distinction; "
            "Cardiac: HCM rarely; assess echocardiogram"
        ),
        "key_features": [
            "IIIa vs IIIb: IIIa = liver + muscle (85%); IIIb = liver only (15% — specific AGL mutations exon 3)",
            "LIVER DISEASE IN CHILDHOOD: hepatomegaly, fasting hypoglycemia, elevated transaminases, short stature",
            "LIVER IMPROVES AT PUBERTY — key natural history; liver disease often resolves in adulthood",
            "MUSCLE DISEASE IN ADULTS (IIIa): proximal + distal weakness, markedly elevated CK, progressive",
            "NO CIRRHOSIS — critical DDx from GSD-I; GSD-III liver does NOT progress to cirrhosis (rare exceptions late)",
            "CK markedly elevated in IIIa — liver disease may mask with normal CK; muscle biopsy for IIIa vs IIIb",
            "Fasting hypoglycemia: short fasting windows → ketotic hypoglycemia; cornstarch prevents",
            "Cardiac: HCM rarely associated; echocardiogram at diagnosis and if symptoms",
        ],
        "treatment": (
            "Dietary: high-protein diet (2-3 g/kg/day) — reduces hypoglycemia, preserves muscle. "
            "Raw (uncooked) cornstarch: 1.5-2 g/kg q4-6h — sustained glucose release; prevents fasting hypoglycemia. "
            "AVOID prolonged fasting: emergency protocol (IV glucose) for illness/anesthesia. "
            "Muscle disease: physiotherapy; low-impact exercise; avoid intense exercise. "
            "No FDA-approved ERT (unlike Pompe) — dietary management is primary therapy 2026. "
            "Liver: most improves after puberty — hepatologist follow-up; avoid hepatotoxic drugs. "
            "Cardiac: ACE-i + beta-blocker if HCM confirmed; echocardiogram monitoring. "
            "Research: gene therapy trials planned."
        ),
        "monitoring": [
            "Liver: LFTs (ALT/AST/GGT) every 6 months childhood; annually adult; abdominal ultrasound",
            "Blood glucose: fasting glucose; ketone monitoring; hypoglycemia diary",
            "CK: every 6-12 months — track IIIa muscle progression",
            "Cardiac: echo + ECG at diagnosis; annually if any HCM features",
            "Muscle: physiotherapy assessment; 6-minute walk test; grip strength (IIIa)",
            "Growth: height velocity in childhood — short stature from chronic metabolic disease",
            "Fasting tolerance test: quantify hypoglycemia threshold; adjust cornstarch frequency",
        ],
    },
    # -- GBE1 — Andersen Disease, GSD-IV, most severe GSD --------------------------------
    {
        "gene": "GBE1",
        "alt_name": (
            "GBE1 (GBE1-702aa-3p24.2 / AR — Andersen-Disease-GSD-IV-Branching-Enzyme — "
            "MOST-SEVERE-GSD-Classic-Infantile-Hepatic-Cirrhosis-Death-Under-5y-Without-LT — "
            "PAS-POSITIVE-POLYGLUCOSAN-BODIES-Biopsy-PATHOGNOMONIC — "
            "ADULT-POLYGLUCOSAN-BODY-Disease-APBD-UMN-LMN-Dementia-Late-Onset)"
        ),
        "protein": (
            "GBE1 -- 3p24.2 AR -- GBE1-702aa -- "
            "1,4-Alpha-Glucan-Branching-Enzyme-80kDa-Glycogen-Branching-Normal-Soluble-Glycogen -- "
            "GSD-IV-OMIM-232500 -- "
            "MOST-SEVERE-GSD-Classic-Hepatic-Form -- "
            "PAS-POSITIVE-POLYGLUCOSAN-BODIES-Tissue-Deposits-PATHOGNOMONIC-All-Forms -- "
            "CLASSIC-INFANTILE-Hepatic-Cirrhosis-Portal-Hypertension-Death-Under-5y-Without-LT -- "
            "FATAL-PERINATAL-Neuromuscular-Fetal-Hydrops-Arthrogryposis-APGAR-Zero -- "
            "NON-PROGRESSIVE-Neuromuscular-Fixed-CNS-Myopathy-Better-Prognosis -- "
            "ADULT-POLYGLUCOSAN-BODY-Disease-APBD-Late-Onset-UMN-LMN-Dementia-Neurogenic-Bladder -- "
            "Liver-Transplant-Curative-Hepatic-Form -- "
            "OMIM-Gene-GBE1-607839-Disease-GSD-IV-232500"
        ),
        "locus": "3p24.2",
        "protein_size": "702 aa / 80 kDa",
        "inheritance": (
            "AR (biallelic LOF); "
            "Clinical forms (genotype partially correlates): "
            "1. Classic hepatic (most common): infantile cirrhosis, portal hypertension; LT indicated; "
            "2. Fatal perinatal neuromuscular: fetal hydrops, reduced fetal movement, arthrogryposis, APGAR 0, death hours-days; "
            "3. Non-progressive neuromuscular: congenital myopathy + mild CNS; cardiomyopathy; better prognosis; "
            "4. Adult polyglucosan body disease (APBD): late-onset UMN + LMN + dementia + neurogenic bladder; "
            "CK: variable; elevated in neuromuscular forms; "
            "Cardiac: HCM + DCM in non-progressive neuromuscular form"
        ),
        "key_features": [
            "PAS-POSITIVE POLYGLUCOSAN BODIES — tissue accumulation of abnormal, unbranched polysaccharide — PATHOGNOMONIC all forms",
            "CLASSIC HEPATIC FORM: most common; infantile hepatic cirrhosis → portal hypertension → variceal bleeding → death <5y WITHOUT liver transplant",
            "LIVER TRANSPLANT CURATIVE for hepatic form — prevents death; polyglucosan in other tissues may still progress post-LT",
            "FATAL PERINATAL NEUROMUSCULAR: fetal hydrops + arthrogryposis + reduced fetal movement; APGAR 0; hours to days survival",
            "NON-PROGRESSIVE NEUROMUSCULAR: congenital hypotonia + cardiomyopathy; less severe; no liver failure",
            "ADULT POLYGLUCOSAN BODY DISEASE (APBD): late-onset (5th-7th decade); UMN + LMN + cognitive decline + neurogenic bladder",
            "APBD: often misdiagnosed as ALS or MS — polyglucosan on sural nerve biopsy or brain MRI (diffuse white matter) diagnostic",
            "Most SEVERE GSD overall — hepatic form = universally fatal without liver transplant",
        ],
        "treatment": (
            "Hepatic form — LIVER TRANSPLANT: "
            "Early listing for liver transplant; 5-year survival post-LT >80%; "
            "LT indicated when cirrhosis + portal hypertension OR declining synthetic function. "
            "Bridging: nutritional support; ursodeoxycholic acid; propranolol for varices. "
            "Perinatal form: palliative care; goals of care discussion antenatally. "
            "Non-progressive neuromuscular: physiotherapy; cardiac management (ACE-i + BB for HCM/DCM); "
            "respiratory support if ventilatory compromise. "
            "APBD: symptomatic — "
            "Neurogenic bladder: intermittent catheterization; anticholinergics; "
            "Spasticity: baclofen; physiotherapy; "
            "Dementia: cognitive support; safety planning; "
            "Triheptanoin (anaplerotic substrate): investigational for APBD; small studies. "
            "No FDA-approved specific therapy 2026."
        ),
        "monitoring": [
            "Hepatic form: LFTs + coagulation + albumin every 3-6 months; portal pressure (varices screen); liver ultrasound",
            "LT evaluation: hepatology team; PELD/MELD score; donor matching",
            "Cardiac: echo + ECG at diagnosis; every 6-12 months (non-progressive neuromuscular form)",
            "Neurological: cognitive screening (APBD); motor function (UMN/LMN); bladder ultrasound",
            "Respiratory: FVC + nocturnal oximetry (neuromuscular forms); sleep study",
            "Biopsy: liver or muscle — PAS-positive polyglucosan bodies confirm diagnosis",
            "Genetic: family cascade; prenatal testing in affected families",
        ],
    },
    # -- GYS1 — GSD-0a, ventricular arrhythmia + sudden death ---------------------------
    {
        "gene": "GYS1",
        "alt_name": (
            "GYS1 (GYS1-700aa-19q13.33 / AR — GSD-0a-Muscle-Glycogen-Synthase-Deficiency — "
            "EXERCISE-INDUCED-VENTRICULAR-ARRHYTHMIA-SUDDEN-DEATH-PATHOGNOMONIC — "
            "LOW-Muscle-Glycogen-Paradoxical-Cannot-Synthesize-Glycogen — "
            "HCM-Plus-Ventricular-Arrhythmia-MANDATORY-ICD)"
        ),
        "protein": (
            "GYS1 -- 19q13.33 AR -- GYS1-700aa -- "
            "Glycogen-Synthase-1-Muscle-Isoform-81kDa-Glycogen-Synthesis-Catalysis -- "
            "GSD-0a-OMIM-611556 -- "
            "EXERCISE-INDUCED-VENTRICULAR-ARRHYTHMIA-SUDDEN-CARDIAC-DEATH-PATHOGNOMONIC -- "
            "LOW-Muscle-Glycogen-Paradoxical-Storage-Disease-Empty-Glycogen-Stores -- "
            "HCM-Hypertrophic-Cardiomyopathy-Plus-Ventricular-Arrhythmia-High-SCD-Risk -- "
            "MANDATORY-ICD-All-Symptomatic-Patients-And-High-Risk-Cases -- "
            "CK-Normal-To-Mildly-Elevated -- "
            "Rare-GSD-Fewer-Than-100-Cases-Reported -- "
            "Exercise-Intolerance-Fatigue-Exertional-Syncope -- "
            "OMIM-Gene-GYS1-138570-Disease-GSD-0a-611556"
        ),
        "locus": "19q13.33",
        "protein_size": "700 aa / 81 kDa",
        "inheritance": (
            "AR (biallelic LOF); "
            "Rare: fewer than 100 cases reported worldwide 2026; "
            "GYS1 encodes muscle isoform (GYS2 encodes liver isoform — separate gene, GSD-0b); "
            "Paradoxical: GSD-0 is a STORAGE DISEASE with LOW glycogen — cannot synthesize glycogen; "
            "CK: normal to mildly elevated; "
            "Onset: childhood–adolescence; exercise-induced symptoms; "
            "Cardiac: HCM (hypertrophic cardiomyopathy) + ventricular arrhythmia; "
            "Sudden death: exercise-triggered VT/VF; "
            "Exercise intolerance: fatigue, exertional syncope, dizziness"
        ),
        "key_features": [
            "EXERCISE-INDUCED VENTRICULAR ARRHYTHMIA — VT/VF during physical activity → SUDDEN CARDIAC DEATH; PATHOGNOMONIC for GYS1",
            "PARADOXICAL GSD: LOW muscle glycogen despite being a 'glycogen storage disease' — GYS1 cannot SYNTHESIZE glycogen",
            "HCM: hypertrophic cardiomyopathy present in majority; massive LV hypertrophy possible",
            "HCM + EXERCISE-INDUCED ARRHYTHMIA combination → very high SCD risk; ICD mandatory",
            "Exercise restriction: NO competitive sports; no strenuous exercise; ICD essential",
            "CK: normal to mildly elevated — does NOT reflect disease severity (unlike McArdle/Tarui)",
            "Exertional syncope or palpitations in young patient with HCM → consider GYS1",
            "Beta-blockers: suppress exercise-induced arrhythmia; required alongside ICD",
        ],
        "treatment": (
            "CARDIAC URGENCY — PRIMARY: "
            "ICD (implantable cardioverter-defibrillator): mandatory in symptomatic patients + strong recommendation in asymptomatic HCM+arrhythmia risk. "
            "Beta-blockers: nadolol or propranolol — suppresses exercise-induced VT; reduce HR and arrhythmia trigger. "
            "Lifestyle: ABSOLUTE exercise restriction — NO competitive sport, no strenuous physical activity. "
            "HCM management: "
            "If LVOT obstruction: disopyramide; avoid strong vasodilators (nitrates); "
            "Mavacamten: may help LVOT obstruction (off-label GYS1 — limited data). "
            "Dietary: moderate-fat diet — FFAs as primary fuel when glycogen unavailable; avoid high-glycemic loads. "
            "No FDA-approved disease-modifying therapy 2026. "
            "Genetic counseling: siblings at 25% risk; cascade screening with echo + exercise ECG."
        ),
        "monitoring": [
            "Cardiac: ECG + Holter + echo EVERY 6 MONTHS — arrhythmia and HCM progression tracking",
            "ICD: interrogation every 3 months; shock log review; lead integrity",
            "Exercise stress test: supervised low-intensity; monitor for arrhythmia induction",
            "Beta-blocker: dose titration; resting HR target 55-65 bpm",
            "Genetic: family cascade; echo + exercise ECG in all siblings",
            "Muscle: MRC scale; 6-minute walk; motor function annually",
            "CK: baseline; annual monitoring; not main disease marker",
        ],
    },
    # -- PGAM2 — GSD-X, muscle PGAM deficiency, tubular aggregates ----------------------
    {
        "gene": "PGAM2",
        "alt_name": (
            "PGAM2 (PGAM2-254aa-7p13 / AR — GSD-X-Muscle-Phosphoglycerate-Mutase-Deficiency — "
            "Exercise-Intolerance-MYOGLOBINURIA-After-Intense-Exercise — "
            "TUBULAR-AGGREGATES-Biopsy-Pathognomonic-PGAM2 — "
            "African-American-Founder-p.Trp78Stop-W78X-Most-Common-Allele — "
            "NO-LACTATE-RISE-Forearm-Exercise-Test)"
        ),
        "protein": (
            "PGAM2 -- 7p13 AR -- PGAM2-254aa -- "
            "Phosphoglycerate-Mutase-Muscle-Isoform-2-29kDa-Glycolysis-Late-Step -- "
            "GSD-X-OMIM-261670 -- "
            "Exercise-Intolerance-MYOGLOBINURIA-Intense-Anaerobic-Exercise -- "
            "TUBULAR-AGGREGATES-Sarcoplasmic-Reticulum-Derived-Biopsy-PATHOGNOMONIC-PGAM2 -- "
            "African-American-Founder-p.Trp78Stop-W78X-Most-Common-Allele -- "
            "NO-LACTATE-RISE-Forearm-Ischemic-Exercise-Test-Same-McArdle-Tarui -- "
            "CK-Markedly-Elevated-After-Exercise -- "
            "Rare-GSD-Fewer-Than-150-Cases-Worldwide -- "
            "OMIM-Gene-PGAM2-612931-Disease-GSD-X-261670"
        ),
        "locus": "7p13",
        "protein_size": "254 aa / 29 kDa",
        "inheritance": (
            "AR (biallelic LOF); "
            "African American founder: p.Trp78Stop (W78X) — most common allele worldwide; "
            "Rare: <150 cases reported 2026; "
            "CK: markedly elevated after exercise; may be mildly elevated at rest; "
            "Onset: childhood–adolescence; "
            "Exercise intolerance: primarily intense anaerobic exercise triggers symptoms; "
            "Myoglobinuria: after intense exercise; AKI risk; "
            "No cardiac involvement; "
            "Tubular aggregates on biopsy: SR-derived structures; not specific but characteristic"
        ),
        "key_features": [
            "EXERCISE INTOLERANCE with MYOGLOBINURIA after intense anaerobic exercise — red-brown urine; AKI risk",
            "TUBULAR AGGREGATES on muscle biopsy — sarcoplasmic reticulum-derived; characteristic for PGAM2 myopathy",
            "NO LACTATE RISE on forearm exercise test — same as McArdle/Tarui; differentiating requires enzyme testing",
            "African American founder mutation: p.Trp78Stop (W78X) — most common allele; PCR-based screening available",
            "CK markedly elevated after exercise; baseline CK mildly elevated",
            "PGAM2 catalyzes late glycolytic step (3-phosphoglycerate → 2-phosphoglycerate) — glycolytic block late in pathway",
            "Rare disease: <150 cases worldwide; high index of suspicion needed in African American with exercise myoglobinuria",
            "No second wind (unlike McArdle) — sustained exercise impaired throughout without improvement",
        ],
        "treatment": (
            "Exercise management: avoid intense anaerobic exercise triggers; moderate aerobic activity tolerated. "
            "Acute myoglobinuria/rhabdomyolysis: "
            "IV fluids (normal saline) — high volume; urine output target >200 mL/h; "
            "Monitor creatinine; electrolytes (hyperkalemia, hypocalcemia); "
            "Urine myoglobin; renal protection priority. "
            "Chronic prevention: aerobic conditioning at sub-anaerobic threshold; monitor exercise response. "
            "Pre-exercise sucrose: limited data (McArdle approach); less evidence in PGAM2. "
            "Physiotherapy: aerobic training programme; swimming; cycling (preferred — less myoglobinuria risk). "
            "No FDA-approved disease-modifying therapy 2026."
        ),
        "monitoring": [
            "CK: baseline; after exercise episodes; annually — track rhabdomyolysis risk",
            "Renal: creatinine + urinalysis; urine myoglobin after exercise episodes",
            "Myoglobinuria diary: document frequency, trigger activities, urine color",
            "Exercise test: sub-anaerobic threshold testing; VO2max; identify safe exercise zones",
            "Genetic: family cascade; African American families — targeted W78X screening",
            "Muscle biopsy: tubular aggregates + PGAM enzyme histochemistry if diagnosis uncertain",
            "Cardiac: ECG baseline; no significant cardiac involvement expected",
        ],
    },
]


# ---------------------------------------------------------------------------
# Patient data generation (8 × 40 patients, seeds 2246-2253)
# ---------------------------------------------------------------------------

def _generate_patients(gene_entry: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene = gene_entry["gene"]
    patients = []
    for i in range(n):
        pid = f"{gene}-{seed}-{i+1:02d}"
        onset = 0
        ck_base = 0
        cardiac = False
        myoglobinuria = False
        arrhythmia = False

        if gene == "GAA":
            subtype = rng.choice(["classic_infantile"] * 12 + ["lopd"] * 28)
            onset = rng.uniform(0, 0.3) if subtype == "classic_infantile" else rng.uniform(5, 60)
            ck_base = rng.uniform(200, 600) if subtype == "classic_infantile" else rng.uniform(300, 1500)
            cardiac = subtype == "classic_infantile"
            on_ert = rng.random() < 0.85
            crim_neg = rng.random() < 0.25 if subtype == "classic_infantile" else rng.random() < 0.05
            patients.append({
                "id": pid, "gene": gene, "seed": seed,
                "subtype": subtype.replace("_", " ").title(),
                "onset_years": round(onset, 1),
                "ck_iul": int(ck_base),
                "cardiac_hcm": cardiac,
                "on_ert": on_ert,
                "crim_negative": crim_neg,
                "niv_required": rng.random() < (0.90 if subtype == "classic_infantile" else 0.45),
                "outcome": rng.choice(["ambulant", "ambulant", "ambulant", "wheelchair", "ventilator"]),
            })
        elif gene == "PYGM":
            onset = rng.uniform(8, 35)
            ck_base = rng.uniform(1000, 8000)
            second_wind = rng.random() < 0.90
            myoglobinuria = rng.random() < 0.55
            patients.append({
                "id": pid, "gene": gene, "seed": seed,
                "onset_years": round(onset, 1),
                "ck_iul": int(ck_base),
                "second_wind": second_wind,
                "myoglobinuria": myoglobinuria,
                "cardiac_hcm": False,
                "outcome": rng.choice(["ambulant"] * 8 + ["ambulant_restricted"]),
            })
        elif gene == "PFKM":
            onset = rng.uniform(5, 30)
            ck_base = rng.uniform(500, 5000)
            myoglobinuria = rng.random() < 0.40
            hemolysis = rng.random() < 0.95
            gout = rng.random() < 0.60
            patients.append({
                "id": pid, "gene": gene, "seed": seed,
                "onset_years": round(onset, 1),
                "ck_iul": int(ck_base),
                "hemolytic_anemia": hemolysis,
                "gout": gout,
                "myoglobinuria": myoglobinuria,
                "cardiac_hcm": False,
                "outcome": rng.choice(["ambulant"] * 7 + ["ambulant_restricted", "ambulant_restricted"]),
            })
        elif gene == "LAMP2":
            sex = rng.choice(["male"] * 20 + ["female"] * 20)
            onset = rng.uniform(8, 20) if sex == "male" else rng.uniform(15, 40)
            ck_base = rng.uniform(500, 3000) if sex == "male" else rng.uniform(100, 800)
            cardiac = True
            retinitis = rng.random() < (0.70 if sex == "male" else 0.30)
            wpw = rng.random() < (0.80 if sex == "male" else 0.40)
            id_present = rng.random() < (0.90 if sex == "male" else 0.40)
            transplant = rng.random() < (0.55 if sex == "male" else 0.10)
            patients.append({
                "id": pid, "gene": gene, "seed": seed,
                "sex": sex,
                "onset_years": round(onset, 1),
                "ck_iul": int(ck_base),
                "cardiac_hcm": cardiac,
                "wpw_preexcitation": wpw,
                "retinitis_pigmentosa": retinitis,
                "intellectual_disability": id_present,
                "transplanted": transplant,
                "icd_implanted": rng.random() < (0.85 if sex == "male" else 0.40),
                "outcome": rng.choice(["transplanted", "icd_managed", "icd_managed", "icd_managed", "deteriorating"]),
            })
        elif gene == "AGL":
            subtype = rng.choice(["IIIa"] * 34 + ["IIIb"] * 6)
            onset = rng.uniform(0.5, 5)
            ck_base = rng.uniform(200, 3000) if subtype == "IIIa" else rng.uniform(50, 200)
            cardiac = rng.random() < 0.10
            patients.append({
                "id": pid, "gene": gene, "seed": seed,
                "subtype": subtype,
                "onset_years": round(onset, 1),
                "ck_iul": int(ck_base),
                "cardiac_hcm": cardiac,
                "liver_fibrosis": rng.random() < 0.15,
                "muscle_disease": subtype == "IIIa" and rng.random() < 0.80,
                "on_cornstarch": rng.random() < 0.90,
                "outcome": rng.choice(["ambulant"] * 7 + ["ambulant_restricted", "ambulant_restricted", "ambulant"]),
            })
        elif gene == "GBE1":
            form = rng.choice(
                ["classic_hepatic"] * 18 + ["fatal_perinatal"] * 8 + ["nonprogressive_nm"] * 10 + ["apbd"] * 4
            )
            onset = (
                0.25 if form == "classic_hepatic" else
                -0.25 if form == "fatal_perinatal" else
                0.0 if form == "nonprogressive_nm" else
                rng.uniform(40, 65)
            )
            ck_base = rng.uniform(100, 400)
            cardiac = rng.random() < 0.40 if form == "nonprogressive_nm" else False
            patients.append({
                "id": pid, "gene": gene, "seed": seed,
                "form": form.replace("_", " ").title(),
                "onset_years": round(onset, 1),
                "ck_iul": int(ck_base),
                "cardiac_hcm": cardiac,
                "liver_transplant": rng.random() < 0.60 if form == "classic_hepatic" else False,
                "survived": form not in ("fatal_perinatal",) and rng.random() < 0.75,
                "outcome": (
                    "deceased" if form == "fatal_perinatal" else
                    "post_lt" if form == "classic_hepatic" and rng.random() < 0.55 else
                    "managed"
                ),
            })
        elif gene == "GYS1":
            onset = rng.uniform(6, 20)
            ck_base = rng.uniform(100, 400)
            arrhythmia = rng.random() < 0.85
            cardiac = True
            icd = rng.random() < 0.80
            patients.append({
                "id": pid, "gene": gene, "seed": seed,
                "onset_years": round(onset, 1),
                "ck_iul": int(ck_base),
                "cardiac_hcm": cardiac,
                "ventricular_arrhythmia": arrhythmia,
                "icd_implanted": icd,
                "exertional_syncope": rng.random() < 0.65,
                "outcome": rng.choice(["icd_managed"] * 6 + ["stable"] * 3 + ["scd_prevented"]),
            })
        elif gene == "PGAM2":
            onset = rng.uniform(8, 30)
            ck_base = rng.uniform(200, 800)
            myoglobinuria = rng.random() < 0.75
            tubular_agg = rng.random() < 0.90
            patients.append({
                "id": pid, "gene": gene, "seed": seed,
                "onset_years": round(onset, 1),
                "ck_iul": int(ck_base),
                "myoglobinuria": myoglobinuria,
                "tubular_aggregates": tubular_agg,
                "cardiac_hcm": False,
                "aki_episodes": rng.randint(0, 3),
                "outcome": rng.choice(["ambulant"] * 7 + ["ambulant_restricted", "ambulant_restricted"]),
            })

    return patients


def _aggregate_cohort() -> list:
    all_patients = []
    for i, gene_entry in enumerate(GSD_GENES):
        seed = SEED_BASE + i
        all_patients.extend(_generate_patients(gene_entry, seed))
    return all_patients


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def overview() -> dict:
    cohort = _aggregate_cohort()
    total = len(cohort)

    gene_counts = {}
    cardiac = sum(1 for p in cohort if p.get("cardiac_hcm"))
    myoglobinuria_ct = sum(1 for p in cohort if p.get("myoglobinuria"))
    arrhythmia_ct = sum(1 for p in cohort if p.get("ventricular_arrhythmia"))
    icd_ct = sum(1 for p in cohort if p.get("icd_implanted"))
    transplant_ct = sum(1 for p in cohort if p.get("liver_transplant") or p.get("transplanted"))
    on_ert_ct = sum(1 for p in cohort if p.get("on_ert"))
    second_wind_ct = sum(1 for p in cohort if p.get("second_wind"))
    hemolysis_ct = sum(1 for p in cohort if p.get("hemolytic_anemia"))

    avg_onset = round(
        sum(p.get("onset_years", 0) for p in cohort if p.get("onset_years", 0) >= 0) /
        max(sum(1 for p in cohort if p.get("onset_years", 0) >= 0), 1), 1
    )
    avg_ck = round(sum(p.get("ck_iul", 0) for p in cohort) / total)

    for p in cohort:
        g = p["gene"]
        gene_counts[g] = gene_counts.get(g, 0) + 1

    return {
        "atlas": "Hereditary-Glycogen-Storage-Myopathy-Atlas",
        "subtitle": (
            "Complete 8-Gene Glycogen Storage Disease (GSD) Muscle Spectrum Atlas — "
            "GAA (Pompe / GSD-II) · PYGM (McArdle / GSD-V) · PFKM (Tarui / GSD-VII) · "
            "LAMP2 (Danon) · AGL (Cori-Forbes / GSD-III) · GBE1 (Andersen / GSD-IV) · "
            "GYS1 (GSD-0a) · PGAM2 (GSD-X) — 320-patient aggregate (8×40, seeds 2246-2253)"
        ),
        "kpis": {
            "total_patients": total,
            "genes_covered": len(GSD_GENES),
            "cardiac_hcm_pct": round(100 * cardiac / total),
            "myoglobinuria_pct": round(100 * myoglobinuria_ct / total),
            "arrhythmia_pct": round(100 * arrhythmia_ct / total),
            "icd_pct": round(100 * icd_ct / total),
            "transplant_pct": round(100 * transplant_ct / total),
            "on_ert_pct": round(100 * on_ert_ct / total),
            "second_wind_pct": round(100 * second_wind_ct / total),
            "hemolytic_anemia_pct": round(100 * hemolysis_ct / total),
            "avg_onset_years": avg_onset,
            "avg_ck_iul": avg_ck,
        },
        "gene_summary": gene_counts,
        "pathognomonic_features": {
            "GAA":   "CLASSIC INFANTILE: massive cardiomegaly + hypotonia + short PR + high-voltage QRS on ECG",
            "PYGM":  "SECOND WIND PHENOMENON — exercise cramps resolve after 2-3 min rest; most pathognomonic feature in myopathology",
            "PFKM":  "OUT-OF-WIND (carbs WORSEN symptoms) + HEMOLYTIC ANEMIA + GOUT triad",
            "LAMP2": "WPW PRE-EXCITATION + HCM on ECG/echo; TRIAD (males): massive HCM + myopathy + intellectual disability",
            "AGL":   "IIIa vs IIIb distinction; liver disease childhood then muscle disease adulthood; NO CIRRHOSIS",
            "GBE1":  "PAS-POSITIVE POLYGLUCOSAN BODIES on biopsy; most severe GSD; classic = infantile hepatic cirrhosis",
            "GYS1":  "EXERCISE-INDUCED VENTRICULAR ARRHYTHMIA + HCM; LOW muscle glycogen (paradoxical GSD)",
            "PGAM2": "TUBULAR AGGREGATES on biopsy; myoglobinuria + exercise intolerance; African American W78X founder",
        },
        "inheritance_map": {
            "GAA": "AR", "PYGM": "AR", "PFKM": "AR", "LAMP2": "X-linked dominant",
            "AGL": "AR", "GBE1": "AR", "GYS1": "AR", "PGAM2": "AR",
        },
        "protein_sizes": {g["gene"]: g["protein_size"] for g in GSD_GENES},
        "loci": {g["gene"]: g["locus"] for g in GSD_GENES},
        "key_distinctions": [
            "SECOND WIND (PYGM McArdle) vs OUT-OF-WIND (PFKM Tarui): carbs improve McArdle, worsen Tarui",
            "LAMP2 WPW + HCM on ECG/echo — most diagnostically specific ECG-echo combination in cardiomyopathy genetics",
            "GBE1 MOST SEVERE GSD — hepatic form universally fatal without liver transplant by age 5",
            "GYS1 PARADOXICAL: GSD with LOW glycogen — cannot synthesize, not store; SCD dominant feature",
            "GAA CRIM-NEGATIVE: immune tolerance induction mandatory BEFORE ERT or ERT destroys itself",
            "AGL liver improves at puberty — key natural history; muscle disease emerges in adults (IIIa)",
            "PGAM2 tubular aggregates — African American W78X founder; myoglobinuria main risk",
            "PFKM hemolysis + gout: full Tarui triad = myopathy + hemolytic anemia + hyperuricemia/gout",
        ],
        "critical_treatments": {
            "GAA":   "ERT (avalglucosidase alfa preferred 2026); CRIM-negative: ITI mandatory first",
            "PYGM":  "Aerobic exercise conditioning; sucrose pre-exercise (Level B); avoid anaerobic bursts",
            "PFKM":  "Avoid high-carb pre-exercise; allopurinol for gout; hydration for myoglobinuria",
            "LAMP2": "ICD mandatory; cardiac transplant evaluation; beta-blocker + ACE-i for HCM",
            "AGL":   "Raw cornstarch + high-protein diet; hepatologist; physiotherapy for muscle IIIa",
            "GBE1":  "Liver transplant for hepatic form; palliative care perinatal; APBD symptomatic",
            "GYS1":  "ICD mandatory; strict exercise restriction; beta-blocker; no competitive sport",
            "PGAM2": "Avoid intense anaerobic exercise; IV fluids for rhabdomyolysis; aerobic conditioning",
        },
    }


def breakdown() -> dict:
    cohort = _aggregate_cohort()
    patients_out = []
    for p in cohort:
        entry = next(g for g in GSD_GENES if g["gene"] == p["gene"])
        patients_out.append({
            **p,
            "protein": entry["protein"],
            "alt_name": entry["alt_name"],
            "inheritance": entry["inheritance"],
            "key_features": entry["key_features"],
            "treatment_summary": entry["treatment"][:300],
        })
    return {
        "atlas": "Hereditary-Glycogen-Storage-Myopathy-Atlas",
        "total": len(patients_out),
        "patients": patients_out,
        "gene_profiles": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "key_features": g["key_features"],
                "treatment": g["treatment"],
                "monitoring": g["monitoring"],
            }
            for g in GSD_GENES
        ],
    }


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Glycogen-Storage-Myopathy-Atlas",
        "glossary": {
            "Glycogen Storage Disease (GSD)": (
                "Inherited metabolic disorders of glycogen synthesis or degradation affecting muscle and/or liver. "
                "Caused by enzyme deficiencies in glycogen metabolism pathways. "
                "Numbered I through XV; muscle GSD types (II, V, VII, 0a, X, etc.) cause myopathy ± rhabdomyolysis. "
                "Liver GSD types (I, III, IV, VI, IX, 0b) cause hepatomegaly ± hypoglycemia."
            ),
            "Pompe Disease (GSD-II, GAA)": (
                "Lysosomal storage disease — acid alpha-glucosidase (GAA) deficiency. "
                "Glycogen accumulates in lysosomes of muscle and heart. "
                "Classic infantile: massive cardiomegaly + profound hypotonia + respiratory failure. "
                "Late-onset (LOPD): proximal myopathy + respiratory failure; NO cardiomegaly. "
                "ERT (enzyme replacement therapy) is standard of care."
            ),
            "Enzyme Replacement Therapy (ERT) — Pompe": (
                "IV infusion of recombinant GAA enzyme. "
                "Alglucosidase alfa (Myozyme/Lumizyme): FDA 2006; 20 mg/kg biweekly. "
                "Avalglucosidase alfa (Nexviazyme): FDA 2021; 3× M6P receptor density → better uptake. "
                "Cipaglucosidase alfa + miglustat: FDA 2023; chaperone prevents enzyme degradation en route. "
                "Switch to newer ERTs recommended for suboptimal responders."
            ),
            "CRIM-Negative (Cross-Reactive Immune Material)": (
                "GAA protein absent in patient tissue. "
                "On ERT initiation, immune system recognizes infused GAA as foreign → high-titer antibodies. "
                "Antibodies destroy ERT → inefficacy + anaphylaxis. "
                "Immune tolerance induction (ITI): rituximab + methotrexate + IVIG BEFORE first ERT dose. "
                "CRIM status: assessed by Western blot at diagnosis — mandatory before ERT."
            ),
            "Second Wind Phenomenon (PYGM McArdle)": (
                "Most pathognomonic feature in myopathology. "
                "Patient exercises → cramps at 5-10 min (anaerobic glycolysis required; myophosphorylase absent). "
                "REST 2-3 minutes → cramps resolve → can resume exercise without cramps. "
                "Mechanism: hepatic glucose output + fatty acid mobilization rescues muscle metabolism. "
                "Absent in all other GSDs — specific to GSD-V (McArdle)."
            ),
            "Out-of-Wind Phenomenon (PFKM Tarui)": (
                "OPPOSITE of second wind in McArdle. "
                "Carbohydrate ingestion WORSENS exercise tolerance in Tarui disease. "
                "Mechanism: high glucose → forces glycolysis → PFK-M block → lactate accumulation; "
                "blocks FFA mobilization by insulin → double impairment. "
                "Pathognomonic for GSD-VII — differentiates from McArdle."
            ),
            "Forearm Ischemic Exercise Test": (
                "Diagnostic test for glycolytic/glycogenolytic enzyme deficiencies. "
                "Patient squeezes dynamometer maximally for 1 minute with inflated sphygmomanometer (ischemia). "
                "Normal: lactate rises ≥3× baseline + ammonia rises ≥3× baseline. "
                "GSD-V (McArdle), GSD-VII (Tarui), GSD-X (PGAM2): NO lactate rise; normal ammonia rise. "
                "Flat lactate + flat ammonia: patient did not make full effort."
            ),
            "Danon Disease (LAMP2)": (
                "X-linked dominant lysosomal disease — LAMP2 protein deficiency. "
                "LAMP2 maintains lysosomal membrane integrity for autophagic flux. "
                "Loss → autophagic material accumulates → autophagic vacuolar myopathy. "
                "Males: classic triad (HCM + myopathy + intellectual disability); "
                "Females: HCM dominant, other features variable (X-inactivation)."
            ),
            "Wolff-Parkinson-White (WPW) + HCM": (
                "ECG-echo combination pathognomonic for Danon Disease. "
                "WPW: delta wave + short PR interval on ECG → accessory pathway. "
                "Plus massive HCM on echocardiography → Danon disease until proven otherwise. "
                "Also seen in: Fabry disease, Pompe (classic infantile), PRKAG2 cardiomyopathy. "
                "Danon: WPW + HCM + X-linked dominant = most likely diagnosis."
            ),
            "Polyglucosan Bodies (GBE1)": (
                "Abnormal, poorly branched polysaccharide chains — linear glucan polymer. "
                "Accumulate because branching enzyme (GBE1) absent → amylopectin-like structure. "
                "PAS-positive on histochemistry (strongly diastase-resistant). "
                "Deposits in liver, muscle, CNS, peripheral nerves. "
                "Adult polyglucosan body disease (APBD): polyglucosan in CNS → upper/lower motor neuron + dementia."
            ),
            "GSD-0a (GYS1)": (
                "Paradoxical glycogen storage disease — muscle glycogen synthase (GYS1) ABSENT → LOW glycogen. "
                "Cannot synthesize glycogen in muscle → exercise-induced metabolic crisis. "
                "Presents as HCM + exercise-induced arrhythmia → sudden cardiac death. "
                "GSD-0b: liver isoform (GYS2) — hepatic hypoglycemia, different gene. "
                "GSD-0a muscle form: cardiac manifestation dominates; extremely rare."
            ),
            "Tubular Aggregates (PGAM2)": (
                "Sarcoplasmic reticulum-derived membrane structures seen on electron microscopy. "
                "H&E: eosinophilic inclusions; modified Gomori trichrome: red-purple. "
                "Not specific to PGAM2 — also in: STIM1/ORAI1 (tubular aggregate myopathy), "
                "phosphoglycerate kinase deficiency, ageing. "
                "In PGAM2 context + exercise-triggered myoglobinuria → GSD-X diagnosis."
            ),
            "Vacuolar Myopathy (GAA)": (
                "Muscle biopsy finding in Pompe disease. "
                "Vacuoles contain glycogen (PAS-positive) + lysosomal debris. "
                "Acid phosphatase positive (lysosomal) — distinguishes from neutral lipid storage or glycogen. "
                "Classic infantile: vacuoles in cardiomyocytes + skeletal muscle. "
                "LOPD: vacuoles in type I (slow-twitch) fibers predominantly."
            ),
            "Raw Cornstarch Therapy (AGL, other liver GSD)": (
                "Uncooked cornstarch: slow starch digestion → sustained glucose release over 4-6 hours. "
                "Prevents fasting hypoglycemia in hepatic GSD (GSD-III, GSD-VI, GSD-IX). "
                "Dose: 1.5-2 g/kg every 4-6 hours; adjust by fasting glucose monitoring. "
                "Cooked starch: rapidly digested — does NOT provide sustained glucose; must use RAW (uncooked)."
            ),
            "DBS Enzyme Assay (GAA)": (
                "Dried blood spot enzyme activity measurement — gold standard for Pompe diagnosis. "
                "<1% residual GAA activity: classic infantile (severe). "
                "1-10% residual: late-onset (LOPD). "
                "Pseudodeficiency alleles (p.Gly576Ser + p.Glu689Lys in trans) → false-low DBS; "
                "confirm with leukocyte assay + genotyping."
            ),
        },
        "diagnostic_algorithm": [
            "1. Exercise intolerance + myoglobinuria → forearm exercise test: NO LACTATE RISE → GSD-V / VII / X differential",
            "2. Second wind present → PYGM (McArdle GSD-V); Out-of-wind with carbs → PFKM (Tarui GSD-VII)",
            "3. HCM + WPW on ECG → LAMP2 (Danon); massive HCM + triad in male → Danon confirmed",
            "4. Floppy infant + cardiomegaly + short PR ECG → GAA enzyme assay DBS; <1% activity → classic Pompe",
            "5. Proximal myopathy + diaphragm weakness + NO cardiomegaly → LOPD (GAA); DBS enzyme assay",
            "6. Infant hepatomegaly + fasting hypoglycemia → AGL (GSD-III); CK elevated = IIIa; biopsy PAS",
            "7. Infantile hepatic cirrhosis → GBE1 (GSD-IV); PAS+ polyglucosan bodies biopsy; liver transplant",
            "8. Young patient + HCM + exercise-induced syncope/VT → GYS1; low muscle glycogen; ICD",
            "9. Exercise myoglobinuria + tubular aggregates biopsy → PGAM2; African American: W78X screening",
            "10. Targeted gene panel: GAA, PYGM, PFKM, LAMP2, AGL, GBE1, GYS1, PGAM2 — confirm genotype",
            "11. WES if panel negative; LAMP2 (X-linked): check hemizygous males + heterozygous females",
            "12. ERT eligibility: GAA confirmed → CRIM status before ERT; LAMP2 cardiac: transplant evaluation",
        ],
        "references": [
            "van der Ploeg AT, Reuser AJ. Pompe's disease. Lancet. 2008;372(9646):1342-1353.",
            "Quinlivan R et al. McArdle disease: a clinical review. J Neurol Neurosurg Psychiatry. 2010;81(11):1182-1188.",
            "Musumeci O et al. Tarui's disease and distal glycogenosis: clinical and genetic update. Acta Myol. 2012.",
            "Sugie K et al. Danon disease: a phenotypic expression of LAMP-2 deficiency. Acta Neuropathol. 2002.",
            "Sentner CP et al. Glycogen storage disease type III: diagnosis, genotype, management, clinical course and outcome. J Inherit Metab Dis. 2016.",
            "Akman HO et al. Neutral lipid storage disease with subclinical myopathy due to a retrotransposal insertion in the PNPLA2 gene. Neuromuscul Disord. 2010.",
            "Kollberg G et al. Cardiomyopathy and exercise intolerance in muscle glycogen storage disease 0. N Engl J Med. 2007;357(15):1507-1514.",
            "DiMauro S, Tsujino S. Nonlysosomal glycogenoses. In: Engel AG, Franzini-Armstrong C (eds). Myology. 2004.",
        ],
        "standards": [
            "European Pompe Consortium (EPC) — diagnosis and management guidelines",
            "TREAT-NMD GSD Advisory Committee — exercise testing protocols",
            "GSD Network International — clinical guidance GAA/PYGM/PFKM",
            "American College of Cardiology/AHA — HCM management guidelines (LAMP2/GYS1)",
            "ACMG/AMP 2015 — variant classification criteria (all GSD genes)",
            "ICMR — India Rare Disease Policy 2021 (ERT access GAA)",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2)[:3000])
    print("\n=== DEFINITIONS ===")
    print(json.dumps(definitions(), indent=2)[:2000])
