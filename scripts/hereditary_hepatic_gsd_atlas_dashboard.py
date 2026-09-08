#!/usr/bin/env python3
"""Hereditary-Hepatic-Glycogen-Storage-Disease-Atlas — Complete 8-Gene Hepatic GSD Spectrum Atlas
(G6PC · SLC37A4 · GAA · AGL · GBE1 · PYGL · PHKA2 · GYS2).

G6PC     (Glucose-6-Phosphatase Catalytic Subunit; 357 aa; 17q21.31; AR;
          GSD Ia — Von Gierke Disease;
          METABOLIC QUARTET PATHOGNOMONIC: lactic acidosis + hyperuricemia +
          hyperlipidemia + fasting hypoglycemia;
          hepatomegaly + renal enlargement; glucagon test FAILS (no glucose rise);
          cornstarch therapy; hepatic adenomas 70%+ adults → HCC surveillance;
          platelet dysfunction / bleeding diathesis; OMIM 232200;
          seed SEED_BASE+0).
SLC37A4  (Solute Carrier Family 37 Member 4 / G6PT; 429 aa; 11q23.3; AR;
          GSD Ib — IDENTICAL to GSD Ia metabolically PLUS CYCLIC NEUTROPENIA
          PATHOGNOMONIC, Crohn's-like IBD, recurrent infections;
          G-CSF mandatory for neutropenia;
          Empagliflozin (SGLT2i) approved 2023 for neutropenia in GSD Ib;
          p.G149E most common European allele; OMIM 232220;
          seed SEED_BASE+1).
GAA      (Acid Alpha-Glucosidase; 952 aa; 17q25.3; AR;
          GSD II — Pompe Disease — lysosomal glycogen storage;
          IOPD: cardiomyopathy/HCM massive + hypotonia → death <1yr untreated;
          LOPD: proximal myopathy + respiratory failure, NO cardiomyopathy;
          CRIM status critical (high-titre antibody → poorer response);
          alglucosidase alfa FDA 2006 / avalglucosidase alfa (Nexviazyme) FDA 2021;
          c.-32-13T>G (IVS1) late-onset Caucasian allele; NBS detected;
          OMIM 232300; seed SEED_BASE+2).
AGL      (Amylo-1,6-Glucosidase; 1532 aa; 1p21.2; AR;
          GSD IIIa/b — Cori/Forbes Disease — debranching enzyme deficiency;
          LIMIT DEXTRINOSIS; IIIa (liver+muscle 85%) vs IIIb (liver-only 15%);
          CK elevated only in IIIa; hepatomegaly → cirrhosis risk in adulthood;
          HIGH-PROTEIN DIET beneficial (provides muscle substrate);
          OMIM 232400; seed SEED_BASE+3).
GBE1     (Glycogen Branching Enzyme; 702 aa; 3p12.3; AR;
          GSD IV — Andersen Disease — branching enzyme deficiency;
          amylopectin-like (polyglucosan) deposits;
          CLASSIC FORM: neonatal hepatic failure → cirrhosis → liver transplant;
          NON-PROGRESSIVE HEPATIC: p.Y329S Ashkenazi founder → survives without transplant;
          ADULT-ONSET APBD: polyglucosan body disease ≥40yr neurological;
          liver transplant curative for hepatic form; OMIM 232500;
          seed SEED_BASE+4).
PYGL     (Liver Glycogen Phosphorylase; 846 aa; 14q22.1; AR;
          GSD VI — Hers Disease — liver phosphorylase deficiency;
          BENIGN COURSE (hepatomegaly resolves with age);
          fasting hypoglycemia mild; hyperketonemia prominent;
          NO myopathy/cardiac; generally asymptomatic adults;
          OMIM 232700; seed SEED_BASE+5).
PHKA2    (Phosphorylase Kinase Alpha2 Subunit; 1235 aa; Xp22.13; XLR;
          GSD IXa — liver phosphorylase kinase alpha2 deficiency;
          X-LINKED RECESSIVE (most common X-linked GSD);
          MOST COMMON CHILDHOOD GSD after GSD III;
          hepatomegaly + growth retardation + fasting ketosis;
          NO MUSCLE INVOLVEMENT; TRANSIENT (symptoms often resolve by puberty);
          OMIM 306000; seed SEED_BASE+6).
GYS2     (Liver Glycogen Synthase; 703 aa; 12p12.1; AR;
          GSD 0 — liver glycogen synthase deficiency;
          FASTING HYPOGLYCEMIA + HYPERKETONAEMIA WITHOUT HEPATOMEGALY — PATHOGNOMONIC;
          NO HEPATOMEGALY (cannot synthesize glycogen → no storage → no enlargement);
          postprandial HYPERGLYCEMIA; NO lactic acidosis;
          HIGH-PROTEIN DIET + frequent feeds; OMIM 240600;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2238-2245).
"""

import random

SEED_BASE = 2238

GSD_GENES = [
    # -- G6PC — GSD Ia, Von Gierke Disease -----------------------------------------------
    {
        "gene": "G6PC",
        "alt_name": (
            "G6PC (G6PC-357aa-17q21.31 / AR — GSD-Ia-Von-Gierke-Disease — "
            "METABOLIC-QUARTET-PATHOGNOMONIC-Lactic-Acidosis+Hyperuricemia+Hyperlipidemia+Fasting-Hypoglycemia — "
            "Hepatomegaly-Renal-Enlargement — "
            "Glucagon-Test-FAILS-No-Glucose-Rise — "
            "Cornstarch-Therapy-Hepatic-Adenomas-70pct-Adults-HCC-Surveillance)"
        ),
        "protein": (
            "G6PC -- 17q21.31 AR -- G6PC-357aa -- "
            "Glucose-6-Phosphatase-Catalytic-Subunit-36kDa-ER-Membrane-Final-Step-Hepatic-Glucose-Output -- "
            "GSD-Ia-Von-Gierke-OMIM-232200 -- "
            "METABOLIC-QUARTET-PATHOGNOMONIC-Lactic-Acidosis-Hyperuricemia-Hyperlipidemia-Fasting-Hypoglycemia -- "
            "Hepatomegaly-Massive-Renal-Enlargement-Both-Pathognomonic-Pair -- "
            "Glucagon-Stimulation-Test-FAILS-No-Rise-In-Glucose-Confirms-G6Pase-Block -- "
            "Cornstarch-Therapy-Uncooked-2g-kg-q4h-Nocturnal-Continuous-Feed-Infants -- "
            "Hepatic-Adenomas-70pct-Adults-Annual-USS-HCC-Surveillance-Mandatory -- "
            "Platelet-Dysfunction-Bleeding-Diathesis-Epistaxis-Easy-Bruising -- "
            "OMIM-Gene-G6PC-613742-Disease-GSD1a-232200"
        ),
        "locus": "17q21.31",
        "protein_size": "357 aa / 36 kDa",
        "inheritance": (
            "AR (biallelic loss-of-function); "
            "G6PC = glucose-6-phosphatase catalytic subunit — anchored in ER membrane; "
            "catalyses final step of hepatic glucose output (G6P → glucose + Pi); "
            "deficiency → G6P accumulates → diverted to glycolysis (lactate), pentose phosphate (uric acid), "
            "lipogenesis (triglycerides) → metabolic quartet; "
            "METABOLIC QUARTET PATHOGNOMONIC: lactic acidosis + hyperuricemia + hyperlipidemia + "
            "fasting hypoglycemia — all four present simultaneously distinguishes GSD Ia from other GSDs; "
            "Glucagon stimulation test: no glucose rise (G6Pase absent) — confirms diagnosis; "
            "Hepatomegaly: massive, present from infancy; renal enlargement on USS; "
            "Hepatic adenomas: develop in 70%+ adults — annual ultrasound + AFP mandatory; "
            "HCC risk: ~10% adenomas → malignant transformation; "
            "Platelet dysfunction: impaired ADP-induced aggregation — bleeding diathesis; "
            "Cornstarch: complex glucose polymer → slow absorption → prevents hypoglycemia"
        ),
        "key_features": [
            "METABOLIC QUARTET — PATHOGNOMONIC: simultaneous lactic acidosis + hyperuricemia + "
            "hyperlipidemia + fasting hypoglycemia; all four biochemical abnormalities together "
            "distinguish GSD Ia from all other hepatic GSDs",
            "GLUCAGON STIMULATION TEST FAILS — no rise in blood glucose (G6Pase cannot release G6P); "
            "galactose + fructose infusion also fail (both metabolised via G6P); "
            "confirms G6Pase block; distinguishes from GSD III where glucagon response blunted",
            "HEPATOMEGALY + RENAL ENLARGEMENT — both present from infancy; "
            "massive hepatomegaly (abdomen protrudes); bilateral renal enlargement on USS; "
            "doll's face appearance (fat cheeks, short stature); hypotonia",
            "HEPATIC ADENOMAS — develop in 70%+ adults (>20 yr); annual liver USS + AFP mandatory; "
            "~10% malignant transformation → HCC; no regression without glycaemic control; "
            "sorafenib/resection for malignant adenomas",
            "CORNSTARCH THERAPY — uncooked corn starch 2g/kg q4h (adults q6h); slow-release glucose; "
            "prevents fasting hypoglycemia; nocturnal continuous enteral feed in infants; "
            "modified cornstarch (Glycosade) q8-10h extends intervals",
            "PLATELET DYSFUNCTION / BLEEDING DIATHESIS — impaired ADP-induced aggregation; "
            "epistaxis, easy bruising; preoperative dextrose loading mandatory to correct; "
            "DDAVP used perioperatively; avoid aspirin/NSAIDs",
            "LACTIC ACIDOSIS — chronic mild (2-5 mmol/L); increases with fasting; "
            "tissue metabolism of elevated lactate is adaptive; pH rarely <7.3 unless infection/fasting",
            "GOUT + HYPERURICEMIA — uric acid elevated (G6P → pentose phosphate shunt); "
            "gout attacks in adolescence/adulthood; allopurinol/febuxostat treatment",
        ],
        "treatment": (
            "Diet: frequent feeds + uncooked cornstarch 2g/kg q4h (adults q6-8h); "
            "nocturnal continuous nasogastric glucose infusion in infants (target glucose >3.5 mmol/L). "
            "Modified cornstarch (Glycosade): extended-release, q8-10h intervals. "
            "Avoid fasting >3-4 hours; avoid galactose-rich + fructose-rich foods. "
            "Lactic acidosis: metabolic control; sodium bicarbonate if acute severe acidosis. "
            "Hyperuricemia: allopurinol 100-300mg/day if symptomatic gout; febuxostat alternative. "
            "Hyperlipidemia: dietary fat restriction; fibrates (avoid statins — myopathy risk). "
            "Hepatic adenomas: annual USS + AFP; resection/ablation if malignant transformation. "
            "Bleeding: preoperative cornstarch loading + DDAVP; avoid aspirin. "
            "Gene therapy: AAV-mediated G6PC replacement — Phase III clinical trials 2024-2026."
        ),
        "monitoring": [
            "Glucose: continuous glucose monitor (CGM) or frequent capillary glucose; target >3.5 mmol/L",
            "Lactate: quarterly; target <2.5 mmol/L on treatment",
            "Uric acid: 6-monthly; gout surveillance; allopurinol dose adjustment",
            "Lipids: triglycerides + cholesterol quarterly (hyperlipidemia monitoring)",
            "Liver: annual USS + AFP (adenoma/HCC surveillance); MRI if adenoma detected",
            "Renal: annual GFR + urine microalbumin (renal GSD Ia nephropathy — focal segmental GS)",
            "Bone density: DEXA every 2-3 years (chronic lactic acidosis → osteoporosis risk)",
            "Haematology: FBC + platelet function testing annually; coagulation screen pre-surgery",
        ],
    },
    # -- SLC37A4 — GSD Ib, G6P Translocase Deficiency ------------------------------------
    {
        "gene": "SLC37A4",
        "alt_name": (
            "SLC37A4 (SLC37A4-429aa-11q23.3 / AR — GSD-Ib-G6PT-Deficiency — "
            "IDENTICAL-to-GSD-Ia-metabolically-PLUS-CYCLIC-NEUTROPENIA-PATHOGNOMONIC — "
            "Crohn's-Like-IBD-Recurrent-Infections — "
            "G-CSF-Mandatory-for-Neutropenia — "
            "Empagliflozin-SGLT2i-Approved-2023-GSD-Ib-Neutropenia — "
            "p.G149E-Most-Common-European)"
        ),
        "protein": (
            "SLC37A4 -- 11q23.3 AR -- SLC37A4-429aa -- "
            "Glucose-6-Phosphate-Translocase-G6PT-46kDa-ER-Membrane-Transports-G6P-into-ER-Lumen -- "
            "GSD-Ib-OMIM-232220 -- "
            "METABOLIC-QUARTET-IDENTICAL-GSD-Ia-PLUS-CYCLIC-NEUTROPENIA-PATHOGNOMONIC -- "
            "Crohns-Like-IBD-Oral-Ulcers-Perianal-Disease-Neutrophil-Gut-Dysfunction -- "
            "G-CSF-Granulocyte-Colony-Stimulating-Factor-Mandatory-Neutropenia-ANC<500 -- "
            "Empagliflozin-SGLT2i-2023-Approved-GSD1b-Neutropenia-Mechanism-1,5-AG-Reduction -- "
            "p.G149E-c.446G>A-Most-Common-European-Allele-GSD-Ib -- "
            "OMIM-Gene-SLC37A4-602671-Disease-GSD1b-232220"
        ),
        "locus": "11q23.3",
        "protein_size": "429 aa / 46 kDa",
        "inheritance": (
            "AR (biallelic loss-of-function); "
            "SLC37A4 = G6P translocase — transports G6P across ER membrane into ER lumen "
            "where G6Pase (G6PC) cleaves it; "
            "deficiency → G6P cannot enter ER → G6Pase cannot act → identical metabolic quartet to GSD Ia; "
            "p.G149E (c.446G>A): most common European allele; "
            "EXTRA feature not in GSD Ia: CYCLIC NEUTROPENIA PATHOGNOMONIC — "
            "G6PT expressed in neutrophil ER → LOF → neutrophil energy failure → cyclic ANC nadirs; "
            "ANC <500/μL → recurrent bacterial infections, oral ulcers, perianal disease; "
            "Crohn's-like IBD: neutrophil dysfunction + gut inflammation; "
            "G-CSF (filgrastim): mandatory when ANC <500; improves neutrophil count + gut disease; "
            "Empagliflozin (SGLT2i): FDA/EMA 2023 approval for GSD Ib neutropenia — "
            "reduces 1,5-anhydroglucitol (1,5-AG) accumulated in neutrophils via SGLT2 inhibition"
        ),
        "key_features": [
            "METABOLIC QUARTET IDENTICAL TO GSD Ia — lactic acidosis + hyperuricemia + hyperlipidemia + "
            "fasting hypoglycemia; hepatomegaly + renal enlargement; cornstarch therapy identical; "
            "clinically indistinguishable from GSD Ia on metabolic parameters alone",
            "CYCLIC NEUTROPENIA — PATHOGNOMONIC distinguisher from GSD Ia: "
            "ANC <500/μL in cyclic nadirs; G6PT expressed in neutrophil ER → neutrophil LOF; "
            "recurrent bacterial infections; oral ulcers; perianal abscess; IBD-like gut disease",
            "CROHN'S-LIKE IBD — intestinal inflammation resembling Crohn's disease; "
            "oral ulcers + perianal disease; G-CSF treatment improves gut disease as well as neutropenia; "
            "neutrophil gut dysfunction (not classical autoimmune IBD mechanism)",
            "G-CSF MANDATORY — filgrastim when ANC <500/μL (or symptomatic neutropenia); "
            "dose: 5-10 μg/kg/day SC; ANC target 1000-2000/μL; "
            "IBD symptoms improve with G-CSF; splenomegaly risk with long-term G-CSF",
            "EMPAGLIFLOZIN (SGLT2i) — FDA/EMA approved 2023 for GSD Ib neutropenia; "
            "mechanism: SGLT2 inhibition reduces 1,5-AG uptake in neutrophils → reduces 1,5-AG-6-phosphate "
            "accumulation → improves neutrophil glucose metabolism; dose 10-25mg/day; "
            "reduces G-CSF requirements; monitor for urinary infections + DKA risk",
            "p.G149E (c.446G>A) — most common European allele; ~35-40% GSD Ib alleles; "
            "PCR genotyping available; missense with minimal residual G6PT activity",
            "Hepatic adenomas: same risk as GSD Ia — 70%+ adults; annual USS + AFP mandatory; "
            "HCC surveillance identical to GSD Ia",
            "DDAVP perioperative: platelet dysfunction identical to GSD Ia; preoperative preparation mandatory",
        ],
        "treatment": (
            "Metabolic: identical to GSD Ia — cornstarch + nocturnal feeds; avoid fasting. "
            "Neutropenia: G-CSF (filgrastim) 5-10 μg/kg/day SC when ANC <500/μL; "
            "target ANC 1000-2000/μL. "
            "Empagliflozin: 10-25mg/day PO — reduces 1,5-AG neutrophil toxicity; "
            "approved 2023 for GSD Ib neutropenia; monitor UTI + urinary ketones. "
            "IBD: G-CSF improves gut disease; mesalazine for mild IBD; "
            "antibiotics (metronidazole/ciprofloxacin) for perianal disease. "
            "Hepatic adenomas: annual USS + AFP; resection if malignant. "
            "Transplant: liver ± intestinal transplant if refractory adenomas or HCC. "
            "Prophylaxis: antimicrobial prophylaxis during neutropenic episodes; "
            "oral hygiene programme; dental monitoring for ulcers."
        ),
        "monitoring": [
            "ANC: full blood count weekly until stable on G-CSF; monthly once stable",
            "Glucose: CGM or frequent capillary glucose; target >3.5 mmol/L",
            "Lactate/uric acid/lipids: quarterly (identical metabolic quartet monitoring to GSD Ia)",
            "Liver: annual USS + AFP (adenoma/HCC surveillance)",
            "Renal: annual GFR + microalbumin (nephropathy as in GSD Ia)",
            "IBD: annual colonoscopy if IBD symptoms; faecal calprotectin surveillance",
            "Empagliflozin monitoring: urine culture (UTI), urinary ketones, eGFR",
            "Splenomegaly: USS annually (G-CSF long-term effect)",
        ],
    },
    # -- GAA — GSD II, Pompe Disease ------------------------------------------------------
    {
        "gene": "GAA",
        "alt_name": (
            "GAA (GAA-952aa-17q25.3 / AR — GSD-II-Pompe-Disease-Lysosomal-Glycogen-Storage — "
            "IOPD-HCM-Massive-Cardiomyopathy-Hypotonia-Death-Under-1yr-Untreated — "
            "LOPD-Proximal-Myopathy-Respiratory-Failure-NO-Cardiomyopathy — "
            "CRIM-Status-Critical-High-Titre-Antibody-Poorer-Response — "
            "Alglucosidase-Alfa-FDA-2006-Avalglucosidase-Alfa-Nexviazyme-FDA-2021 — "
            "c.-32-13T>G-IVS1-Late-Onset-Caucasian-Allele-NBS-Detected)"
        ),
        "protein": (
            "GAA -- 17q25.3 AR -- GAA-952aa -- "
            "Acid-Alpha-Glucosidase-105kDa-Lysosomal-Enzyme-Glycogen-Degradation-Lysosome -- "
            "GSD-II-Pompe-Disease-OMIM-232300 -- "
            "IOPD-Infantile-Onset-Pompe-Massive-HCM-Cardiomyopathy-Hypotonia-Respiratory-Failure -- "
            "LOPD-Late-Onset-Proximal-Limb-Girdle-Myopathy-Respiratory-Decline-No-Cardiomyopathy -- "
            "CRIM-Status-Cross-Reactive-Immunological-Material-High-Titre-INHIBITORY-Antibody-Poorer-ERT-Response -- "
            "Alglucosidase-Alfa-Myozyme-FDA-2006-ERT-First-Line -- "
            "Avalglucosidase-Alfa-Nexviazyme-FDA-2021-Superior-Mannose-6P-Uptake -- "
            "c.-32-13T>G-IVS1-Splice-Site-Residual-Activity-Late-Onset-Caucasian-Allele -- "
            "NBS-DETECTED-Alpha-Glucosidase-Activity-DBS -- "
            "OMIM-Gene-GAA-606800-Disease-GSD2-232300"
        ),
        "locus": "17q25.3",
        "protein_size": "952 aa / 105 kDa",
        "inheritance": (
            "AR (biallelic); "
            "GAA = acid alpha-glucosidase — lysosomal enzyme; only lysosomal glycogen storage disease; "
            "deficiency → glycogen accumulates in lysosomes of all tissues; "
            "IOPD (Infantile-Onset Pompe Disease): null/null or severe mutations; "
            "massive HCM (cardiomegaly on CXR), generalised hypotonia, respiratory failure; "
            "death <1yr without ERT; NBS → early ERT dramatically improves survival; "
            "LOPD (Late-Onset Pompe Disease): at least one allele with residual activity "
            "(c.-32-13T>G/IVS1 most common); proximal myopathy (Gowers sign) + respiratory failure; "
            "NO cardiomyopathy in LOPD; "
            "CRIM status: CRIM+ patients make some GAA protein → tolerate ERT; "
            "CRIM− patients (null/null) → high-titre inhibitory antibodies → prophylactic ITI mandatory; "
            "Avalglucosidase alfa (Nexviazyme): engineered high M6P → superior lysosomal uptake; "
            "c.-32-13T>G: residual splicing → ~2% activity; late-onset Caucasian allele"
        ),
        "key_features": [
            "IOPD — MASSIVE HCM: cardiomegaly on CXR (cardiothoracic ratio >0.5); "
            "generalised hypotonia (floppy infant); feeding difficulties; respiratory failure; "
            "death <1yr untreated; ERT + NBS have transformed outcomes — "
            "NBS-detected IOPD treated within 3 months has near-normal cardiac function at 5yr",
            "LOPD — NO CARDIOMYOPATHY: proximal limb-girdle weakness; Gowers sign; "
            "respiratory failure (diaphragm + intercostal involvement); FVC declines 1-2%/year; "
            "NIV mandatory when FVC <50% or nocturnal desaturation; "
            "glycogen accumulation on muscle biopsy (PAS + acid phosphatase staining)",
            "CRIM STATUS CRITICAL — CRIM-negative patients (make NO GAA protein) develop "
            "high-titre inhibitory antibodies to ERT → dramatically worsens response; "
            "ITI (immune tolerance induction) with rituximab + methotrexate + IVIG mandatory "
            "in CRIM-negative before/at ERT start",
            "AVALGLUCOSIDASE ALFA (Nexviazyme, FDA 2021) — engineered with high mannose-6-phosphate "
            "content → superior CI-M6PR-mediated lysosomal uptake vs alglucosidase alfa; "
            "PROPEL trial: superior motor outcomes in LOPD; preferred over alglucosidase alfa 2024",
            "c.-32-13T>G (IVS1-13T>G) — most common LOPD allele in Caucasians; "
            "allows residual splicing (~2-5% normal GAA activity); "
            "compound heterozygote with severe null → LOPD phenotype; "
            "homozygous IVS1/IVS1 → very mild LOPD",
            "NBS DETECTED — dried blood spot alpha-glucosidase activity; "
            "pseudodeficiency alleles (GAA p.Glu689Lys) cause false positives on NBS — "
            "CLIA-confirmed sequencing mandatory for all NBS positives before treatment",
            "Respiratory monitoring: FVC every 6 months; NIV/CPAP when FVC <50%; "
            "tracheostomy in advanced IOPD if NIV insufficient",
            "Muscle biopsy: PAS stain + acid phosphatase (lysosomal glycogen) + EM — "
            "membrane-bound glycogen vacuoles pathognomonic on EM",
        ],
        "treatment": (
            "ERT: avalglucosidase alfa (Nexviazyme) 20mg/kg IV q2w — PREFERRED FIRST LINE 2024; "
            "alglucosidase alfa (Myozyme/Lumizyme) 20mg/kg IV q2w — alternative. "
            "CRIM-negative IOPD: ITI protocol — rituximab 375mg/m² × 4 doses + "
            "methotrexate 0.5mg/kg/week + IVIG 2g/kg/month — start before/at ERT. "
            "LOPD respiratory: NIV when FVC <50% or nocturnal desaturation; "
            "physiotherapy: airway clearance; respiratory muscle training. "
            "Cardiac: digoxin + diuretics in IOPD acute HF; ACE inhibitor if DCM; "
            "pacemaker if conduction defect. "
            "Physiotherapy: motor skills rehabilitation; orthotics; hydrotherapy. "
            "Gene therapy: AAV9-GAA clinical trials 2024-2026 (MiniApe-AAV). "
            "NBS: ERT within first month of life for IOPD — target before symptoms."
        ),
        "monitoring": [
            "Cardiac: echocardiogram every 3 months in IOPD first year; 6-monthly thereafter",
            "Respiratory: FVC + FVC supine every 6 months; polysomnography annually",
            "Motor: 6-minute walk test, Pompe-specific motor function scale every 6 months",
            "GAA antibody titres: every 3 months first year on ERT (CRIM monitoring)",
            "Urine Hex4: urinary hexose tetrasaccharide (Hex4/GLC4) — ERT response biomarker",
            "Creatine kinase: quarterly (muscle disease activity)",
            "Liver USS: hepatomegaly monitoring (less severe than GSD I)",
            "NBS follow-up: DBS GAA enzyme + sequencing within 2 weeks of positive screen",
        ],
    },
    # -- AGL — GSD IIIa/b, Cori/Forbes Disease --------------------------------------------
    {
        "gene": "AGL",
        "alt_name": (
            "AGL (AGL-1532aa-1p21.2 / AR — GSD-IIIa-b-Cori-Forbes-Debranching-Enzyme-Deficiency — "
            "LIMIT-DEXTRINOSIS — "
            "IIIa-Liver-Plus-Muscle-85pct-vs-IIIb-Liver-Only-15pct — "
            "CK-Elevated-Only-IIIa — "
            "Hepatomegaly-Cirrhosis-Risk-Adulthood — "
            "HIGH-PROTEIN-DIET-Beneficial-Muscle-Substrate)"
        ),
        "protein": (
            "AGL -- 1p21.2 AR -- AGL-1532aa -- "
            "Amylo-1-6-Glucosidase-4-Alpha-Glucanotransferase-170kDa-Debranching-Enzyme-Glycogen-Branch-Removal -- "
            "GSD-IIIa-b-Cori-Forbes-OMIM-232400 -- "
            "LIMIT-DEXTRINOSIS-Abnormal-Glycogen-Short-Outer-Chains-Accumulate -- "
            "IIIa-Liver-AND-Muscle-85pct-All-GSD-III-CK-ELEVATED -- "
            "IIIb-Liver-ONLY-15pct-CK-NORMAL-No-Myopathy -- "
            "Hepatomegaly-Childhood-Cirrhosis-Fibrosis-Risk-Adulthood -- "
            "HIGH-PROTEIN-DIET-2-3g-kg-Provides-Alanine-Gluconeogenic-Substrate -- "
            "Glucagon-Test-Partially-Blunted-Unlike-GSD-Ia-Complete-Fail -- "
            "OMIM-Gene-AGL-610860-Disease-GSD3-232400"
        ),
        "locus": "1p21.2",
        "protein_size": "1532 aa / 170 kDa",
        "inheritance": (
            "AR (biallelic); "
            "AGL = glycogen debranching enzyme — bifunctional: "
            "(1) glucanotransferase: moves 3 glucose units from branch to main chain; "
            "(2) glucosidase: cleaves final branch-point glucose; "
            "deficiency → glycogen catabolism stops at branch points → limit dextrin accumulates; "
            "GSD IIIa: liver + skeletal muscle affected (85%); CK elevated (myopathy); "
            "GSD IIIb: liver only (15%); CK normal; no myopathy; "
            "Phenotype distinction: critical — IIIb has no muscle disease, better long-term prognosis; "
            "Hepatomegaly: present from infancy; hepatic fibrosis → cirrhosis in adulthood (20%); "
            "Glucagon test: partial response (unlike GSD Ia where completely flat); "
            "HIGH-PROTEIN DIET: provides alanine (gluconeogenic substrate) → compensates for "
            "partial glucose release; beneficial for muscle substrate in IIIa"
        ),
        "key_features": [
            "LIMIT DEXTRINOSIS — glycogen with abnormally short outer chains (limit dextrin); "
            "PAS stain: excess glycogen in liver + muscle (IIIa); "
            "electron microscopy: abnormal glycogen particle morphology; "
            "distinctive biochemical fingerprint on glycogen structure analysis",
            "GSD IIIa vs IIIb DISTINCTION CRITICAL — IIIa (85%): liver + muscle; CK elevated; "
            "progressive myopathy in adulthood (proximal + distal); "
            "IIIb (15%): liver only; CK normal; no myopathy; better long-term prognosis; "
            "gene panel cannot always predict phenotype — isoform analysis and clinical follow-up required",
            "HEPATOMEGALY → CIRRHOSIS RISK — hepatomegaly prominent in childhood; "
            "improves with puberty/metabolic control; "
            "hepatic fibrosis develops in 20% adults → cirrhosis → portal hypertension; "
            "AFP + liver USS annually for HCC surveillance (lower risk than GSD Ia adenomas)",
            "HIGH-PROTEIN DIET (2-3g/kg/day) — provides alanine and other gluconeogenic amino acids; "
            "compensates for reduced G6P availability; improves muscle strength in IIIa; "
            "protein as energy source when carbohydrate-induced hypoglycemia occurs; "
            "combined with uncooked cornstarch",
            "CK ELEVATED IN IIIa ONLY — baseline CK 200-5000 IU/L in IIIa; "
            "normal CK distinguishes IIIb; "
            "progressive myopathy in IIIa adulthood: proximal then distal weakness; "
            "EMG: myopathic + neuropathic pattern in advanced IIIa",
            "GLUCAGON TEST: PARTIAL RESPONSE — unlike GSD Ia (no response at all); "
            "blunted lactate rise (some glycolysis preserved via peripheral breakdown); "
            "blood glucose rises partially (galactose can still produce glucose via non-G6P pathways)",
            "Fasting hypoglycemia: present but generally milder than GSD Ia; "
            "lactic acidosis absent (G6Pase intact — lactate can be cleared); "
            "hyperuricemia mild; hyperlipidemia present but less severe",
            "Neonatal presentation: hepatomegaly + fasting hypoglycemia + mildly elevated transaminases; "
            "initial presentation often confused with GSD Ia until forearm test + glucagon test",
        ],
        "treatment": (
            "Diet: uncooked cornstarch 2g/kg q4-6h; high-protein diet 2-3g/kg/day; "
            "avoid prolonged fasting; frequent small meals. "
            "Nocturnal feeds: continuous enteral glucose overnight in infants. "
            "Hepatic fibrosis: avoid hepatotoxic drugs; annual liver USS + fibroscan. "
            "Myopathy (IIIa): physiotherapy; resistance training; aerobic exercise programme; "
            "high-protein diet to provide muscle substrate. "
            "Liver transplant: if cirrhosis/portal hypertension; corrects liver disease but NOT muscle. "
            "Gene therapy: dual-function AGL AAV delivery — Phase I trials 2025. "
            "Anaesthesia: preoperative glucose loading; avoid prolonged fasting perioperatively."
        ),
        "monitoring": [
            "Glucose: fasting glucose quarterly; CGM if recurrent hypoglycemia",
            "CK: quarterly in IIIa (myopathy progression marker)",
            "Liver: LFTs + albumin quarterly; annual fibroscan (cirrhosis surveillance)",
            "USS + AFP: annual liver USS + AFP for HCC surveillance",
            "Cardiac: echocardiogram annually (cardiomyopathy in GSD IIIa reported)",
            "Motor: annual physiotherapy functional assessment in IIIa",
            "EMG/NCS: every 2-3 years in IIIa if weakness progresses",
            "Genetic: IIIa vs IIIb phenotype classification from biopsy isoform analysis",
        ],
    },
    # -- GBE1 — GSD IV, Andersen Disease -------------------------------------------------
    {
        "gene": "GBE1",
        "alt_name": (
            "GBE1 (GBE1-702aa-3p12.3 / AR — GSD-IV-Andersen-Disease-Branching-Enzyme-Deficiency — "
            "Amylopectin-Like-Polyglucosan-Deposits — "
            "CLASSIC-FORM-Neonatal-Hepatic-Failure-Cirrhosis-Liver-Transplant-Only-Hope — "
            "NON-PROGRESSIVE-HEPATIC-p.Y329S-Ashkenazi-Survives-Without-Transplant — "
            "ADULT-ONSET-APBD-Polyglucosan-Body-Disease-Neurological-40yr-Plus)"
        ),
        "protein": (
            "GBE1 -- 3p12.3 AR -- GBE1-702aa -- "
            "Glycogen-Branching-Enzyme-80kDa-Adds-Alpha-1-6-Branches-to-Glycogen-Chain -- "
            "GSD-IV-Andersen-Disease-OMIM-232500 -- "
            "AMYLOPECTIN-LIKE-POLYGLUCOSAN-Deposits-Long-Unbranched-Chains-Like-Plant-Starch -- "
            "CLASSIC-HEPATIC-FORM-Neonatal-Liver-Failure-Cirrhosis-6-18mths-Death-Without-Transplant -- "
            "NON-PROGRESSIVE-HEPATIC-p.Y329S-Ashkenazi-Jewish-Founder-Survives-Into-Adulthood -- "
            "ADULT-ONSET-APBD-Polyglucosan-Body-Disease-Neurological-Corticospinal-Peripheral-Nerve -- "
            "Liver-Transplant-CURATIVE-Hepatic-Form -- "
            "OMIM-Gene-GBE1-607839-Disease-GSD4-232500"
        ),
        "locus": "3p12.3",
        "protein_size": "702 aa / 80 kDa",
        "inheritance": (
            "AR (biallelic); "
            "GBE1 = glycogen branching enzyme — adds alpha-1,6 branch points to glycogen; "
            "deficiency → linear, poorly-branched amylopectin-like glycogen (polyglucosan) accumulates; "
            "3 distinct phenotypes: "
            "(1) CLASSIC HEPATIC: progressive hepatic failure → cirrhosis → death by 2-4yr without transplant; "
            "null/severe alleles; liver transplant CURATIVE for hepatic manifestation; "
            "(2) NON-PROGRESSIVE HEPATIC: p.Y329S (Ashkenazi Jewish founder) — residual enzyme activity; "
            "hepatomegaly + transaminase elevation but no progression to cirrhosis; survives without transplant; "
            "(3) ADULT-ONSET APBD (Adult Polyglucosan Body Disease): ≥40yr; corticospinal + peripheral neuropathy; "
            "GBE1 mutations in this form have significant residual activity; "
            "Biopsy: PAS-positive, diastase-resistant deposits = polyglucosan bodies pathognomonic"
        ),
        "key_features": [
            "CLASSIC FORM — progressive hepatic failure: hepatomegaly → cirrhosis → portal hypertension "
            "→ liver failure by 12-24 months; hypotonia + failure to thrive; "
            "liver transplant is the ONLY hope — curative for hepatic disease; "
            "wait-list mortality high; living donor transplant preferred",
            "NON-PROGRESSIVE HEPATIC FORM — p.Y329S (Ashkenazi Jewish founder allele): "
            "residual enzyme activity (~14%); hepatomegaly + elevated transaminases present; "
            "but NO progression to cirrhosis in most patients; survives into adulthood; "
            "targeted testing of p.Y329S in Ashkenazi Jewish children with unexplained hepatomegaly",
            "ADULT-ONSET APBD (Adult Polyglucosan Body Disease): GBE1 biallelic mutations "
            "with residual activity; age ≥40yr; corticospinal tract involvement (spastic paraparesis) + "
            "peripheral neuropathy + bladder dysfunction + dementia; "
            "PAS-positive diastase-resistant inclusions in neurons + peripheral nerve on biopsy",
            "POLYGLUCOSAN BODIES — PAS-positive + diastase-RESISTANT deposits (unlike normal glycogen "
            "which is diastase-sensitive); long unbranched glucose chains; accumulate in liver, muscle, "
            "heart, neurons; hallmark on biopsy under light microscopy + EM",
            "LIVER TRANSPLANT CURATIVE FOR HEPATIC FORM — correct indication: progressive cirrhosis "
            "(classic form); contraindication: non-progressive form (p.Y329S) does not need transplant; "
            "transplant does NOT cure extra-hepatic disease (cardiomyopathy, neuromuscular); "
            "post-transplant GBE1 activity from donor liver corrects hepatic polyglucosan",
            "CARDIAC INVOLVEMENT — cardiomyopathy (dilated or hypertrophic) may accompany classic form; "
            "separate from hepatic disease; may not resolve post-transplant; "
            "echocardiogram mandatory pre-transplant",
            "Fetal hydrops form: most severe; biallelic null; widespread polyglucosan in all tissues; "
            "stillbirth or neonatal death; diagnosis on placental biopsy/autopsy",
            "Neuromuscular form (childhood): hypotonia + myopathy + hepatomegaly + cardiomyopathy; "
            "overlaps classic but with neurological features; intermediate prognosis",
        ],
        "treatment": (
            "Classic hepatic form: liver transplant — ONLY curative therapy; "
            "timing: before decompensation (Pedi-MELD ≥12 or escalating); "
            "living related donor preferred (shorter wait). "
            "Non-progressive form (p.Y329S): supportive — dietary management; avoid hepatotoxins; "
            "annual USS + LFTs; no transplant required. "
            "Pre-transplant: manage portal hypertension (propranolol + endoscopy); "
            "nutritional support (nasogastric/PEG); coagulation support (vitamin K + FFP). "
            "APBD (adult): supportive — physiotherapy for spasticity; bladder management; "
            "acetylcysteine + 4-aminobutyric acid — anecdotal. "
            "Post-transplant: immunosuppression per centre protocol; "
            "monitor for extra-hepatic disease progression. "
            "Gene therapy: AAV-GBE1 — preclinical 2024-2026."
        ),
        "monitoring": [
            "LFTs: monthly in progressive hepatic form (cirrhosis trajectory)",
            "Coagulation: INR + albumin monthly (hepatic synthetic function)",
            "Portal hypertension: endoscopy 6-monthly for varices; USS + Doppler quarterly",
            "Cardiac: echocardiogram 6-monthly (cardiomyopathy surveillance)",
            "Neurology: annual developmental assessment in childhood; APBD — neurological exam + MRI annually",
            "Post-transplant: annual LFTs + biopsy; monitor extra-hepatic manifestations",
            "Genetic: p.Y329S PCR screen in Ashkenazi Jewish children with hepatomegaly",
            "APBD: EMG/NCS every 2 years; urodynamics if bladder dysfunction",
        ],
    },
    # -- PYGL — GSD VI, Hers Disease -----------------------------------------------------
    {
        "gene": "PYGL",
        "alt_name": (
            "PYGL (PYGL-846aa-14q22.1 / AR — GSD-VI-Hers-Disease-Liver-Phosphorylase-Deficiency — "
            "BENIGN-COURSE-Hepatomegaly-Resolves-with-Age — "
            "Fasting-Hypoglycemia-Mild-Hyperketonemia-Prominent — "
            "NO-Myopathy-NO-Cardiac — "
            "Generally-Asymptomatic-Adults)"
        ),
        "protein": (
            "PYGL -- 14q22.1 AR -- PYGL-846aa -- "
            "Liver-Glycogen-Phosphorylase-97kDa-Hepatic-Glycogenolysis-Allosterically-Regulated -- "
            "GSD-VI-Hers-Disease-OMIM-232700 -- "
            "BENIGN-COURSE-Hepatomegaly-Prominent-Childhood-Resolves-Puberty -- "
            "Fasting-Hypoglycemia-MILD-Hyperketonemia-PROMINENT-Compensatory-FAO -- "
            "NO-Myopathy-PYGL-Liver-Isoform-Only -- "
            "NO-Cardiac-Involvement -- "
            "Adults-Generally-Asymptomatic-Hepatomegaly-Resolves -- "
            "Cornstarch-Avoidance-Fasting-Treatment-Sufficient -- "
            "OMIM-Gene-PYGL-613741-Disease-GSD6-232700"
        ),
        "locus": "14q22.1",
        "protein_size": "846 aa / 97 kDa",
        "inheritance": (
            "AR (biallelic); "
            "PYGL = liver glycogen phosphorylase — liver-specific isoform (PYGM = muscle isoform); "
            "catalyses phosphorolysis of glycogen alpha-1,4 linkages in liver; "
            "deficiency → impaired hepatic glycogenolysis → glycogen accumulates in liver; "
            "BENIGN COURSE: mildest of the hepatic GSDs; "
            "hepatomegaly: prominent in childhood (3-10yr); typically resolves by puberty; "
            "fasting hypoglycemia: mild and intermittent; rarely symptomatic; "
            "hyperketonemia: prominent (compensatory FFA oxidation → ketogenesis); "
            "no lactic acidosis (G6Pase intact + gluconeogenesis normal → lactate cleared); "
            "no hyperuricemia (pentose phosphate shunt not overloaded); "
            "no myopathy (liver-specific isoform only); "
            "adults: generally asymptomatic; hepatomegaly resolved; normal biochemistry"
        ),
        "key_features": [
            "BENIGN COURSE — mildest hepatic GSD; hepatomegaly resolves by puberty; "
            "adults generally asymptomatic with normal biochemistry; "
            "no long-term liver disease, no adenomas, no cirrhosis (distinguishes from GSD Ia/III)",
            "HEPATOMEGALY PROMINENT IN CHILDHOOD — massive hepatomegaly age 3-10yr; "
            "can cause abdominal distension + short stature; "
            "typically resolves spontaneously by puberty without specific treatment; "
            "biopsy: PAS-positive glycogen excess in hepatocytes",
            "FASTING HYPOGLYCEMIA MILD — intermittent; rarely severe; "
            "simple dietary management (frequent feeds + cornstarch) usually sufficient; "
            "ketotic hypoglycemia (prominent ketones with hypoglycemia = characteristic pattern); "
            "NO lactic acidosis + NO hyperuricemia (distinguishes from GSD Ia)",
            "HYPERKETONEMIA PROMINENT — compensatory FAO → ketogenesis elevated; "
            "urine ketones strongly positive with fasting; "
            "characteristic: PROMINENT KETOSIS with only MILD hypoglycemia (unusual combination); "
            "distinguishes GSD VI from GSD Ia (no ketones in GSD Ia due to high insulin/glucose cycling)",
            "NO MYOPATHY — PYGL is liver-specific; skeletal muscle uses PYGM; "
            "CK normal; no exercise intolerance; "
            "critical DDx from GSD V (McArdle/PYGM) which affects only muscle",
            "NO CARDIAC INVOLVEMENT — cardiomyopathy absent; echocardiogram normal; "
            "distinguishes from Pompe (GAA) where cardiac involvement major",
            "GROWTH RETARDATION — short stature + delayed bone age in childhood; "
            "improves with treatment + puberty; adult height usually normal",
            "Generally no treatment beyond childhood dietary management; "
            "no need for nasogastric feeds or continuous glucose infusion in most patients",
        ],
        "treatment": (
            "Diet: frequent meals + uncooked cornstarch 1-2g/kg at bedtime; "
            "avoid prolonged fasting (>4-6 hours); "
            "high-carbohydrate diet during intercurrent illness. "
            "Most patients: minimal treatment required beyond childhood; "
            "no nocturnal feeds needed in most. "
            "Ketotic hypoglycemia episodes: oral glucose (sugary drink); "
            "IV dextrose if severe/unresponsive. "
            "Hepatomegaly monitoring: confirm resolution by adolescence; "
            "annual USS until resolved. "
            "Growth: monitor growth velocity; nutritional support if growth failure. "
            "Adults: annual GP review + fasting glucose; no specific intervention usually needed."
        ),
        "monitoring": [
            "Glucose: fasting glucose every 6 months in childhood; annually in adults",
            "Ketones: urinary/blood ketones with fasting illness episodes",
            "LFTs: 6-monthly in childhood; annually once hepatomegaly resolved",
            "USS: annual liver USS until hepatomegaly confirmed resolved",
            "Growth: height/weight every 6 months in childhood (growth retardation surveillance)",
            "Adults: annual fasting glucose + LFTs; USS every 2-3 years",
            "No cardiac/muscle monitoring required (unlike GSD III/IV/II)",
        ],
    },
    # -- PHKA2 — GSD IXa, X-Linked Phosphorylase Kinase Deficiency -----------------------
    {
        "gene": "PHKA2",
        "alt_name": (
            "PHKA2 (PHKA2-1235aa-Xp22.13 / XLR — GSD-IXa-Liver-Phosphorylase-Kinase-Alpha2-Deficiency — "
            "X-LINKED-RECESSIVE-Most-Common-X-Linked-GSD — "
            "MOST-COMMON-CHILDHOOD-GSD-After-GSD-III — "
            "Hepatomegaly-Growth-Retardation-Fasting-Ketosis — "
            "NO-MUSCLE-INVOLVEMENT — "
            "TRANSIENT-Symptoms-Often-Resolve-Puberty)"
        ),
        "protein": (
            "PHKA2 -- Xp22.13 XLR -- PHKA2-1235aa -- "
            "Phosphorylase-Kinase-Regulatory-Alpha2-Liver-Subunit-135kDa-Activates-PYGL -- "
            "GSD-IXa-OMIM-306000 -- "
            "X-LINKED-RECESSIVE-MOST-COMMON-X-LINKED-GSD-All-Males-Affected -- "
            "MOST-COMMON-CHILDHOOD-GSD-After-GSD-III-Frequency -- "
            "Hepatomegaly-Prominent-Childhood-Growth-Retardation-Short-Stature -- "
            "Fasting-KETOSIS-HYPOGLYCEMIA-Mild -- "
            "NO-MUSCLE-INVOLVEMENT-PHKA2-Liver-Specific-Alpha2-Subunit -- "
            "TRANSIENT-Hepatomegaly-Growth-Normal-By-Puberty -- "
            "Carrier-Females-Mild-Phenotype -- "
            "OMIM-Gene-PHKA2-300798-Disease-GSD9a-306000"
        ),
        "locus": "Xp22.13",
        "protein_size": "1235 aa / 135 kDa",
        "inheritance": (
            "XLR (X-linked recessive); "
            "PHKA2 = alpha-2 regulatory subunit of liver phosphorylase kinase (PhK); "
            "PhK complex: alpha2-beta-gamma-delta; activates PYGL (liver glycogen phosphorylase) "
            "by phosphorylation; "
            "XLR → all hemizygous males affected; carrier females: usually mild or subclinical; "
            "most common X-linked GSD; "
            "one of the most common childhood hepatic GSDs (after GSD III); "
            "TRANSIENT course: hepatomegaly + growth retardation in childhood; "
            "hepatomegaly typically resolves by puberty; growth normalises; "
            "NO MUSCLE INVOLVEMENT: PHKA2 = liver-specific alpha-2 subunit; "
            "muscle PhK uses PHKA1 (alpha-1 subunit, different gene); "
            "fasting ketosis + mild hypoglycemia; no lactic acidosis; no HCC/adenomas; "
            "generally excellent prognosis"
        ),
        "key_features": [
            "X-LINKED RECESSIVE — all hemizygous males affected; "
            "carrier females usually mild or asymptomatic; "
            "maternal family history of unexplained childhood hepatomegaly in males — X-linked pattern",
            "MOST COMMON CHILDHOOD GSD AFTER GSD III — high prevalence; "
            "often underdiagnosed; erythrocyte PhK activity testing historically used for diagnosis; "
            "GSD IX now grouped by gene (PHKA2/PHKB/PHKG2/PHKA1)",
            "HEPATOMEGALY + GROWTH RETARDATION — prominent hepatomegaly in childhood; "
            "short stature + delayed bone age; transaminases mildly elevated; "
            "abdominal protrusion from hepatomegaly; mimics GSD VI clinically",
            "TRANSIENT COURSE — hepatomegaly typically resolves by puberty; "
            "growth velocity normalises; adult males usually asymptomatic; "
            "excellent long-term prognosis (unlike GSD Ia/III/IV); "
            "no cirrhosis/adenoma risk; no HCC",
            "NO MUSCLE INVOLVEMENT — PHKA2 = liver-specific subunit; "
            "CK normal; exercise tolerance normal; no rhabdomyolysis; "
            "critical DDx from GSD IXd (PHKA1 — X-linked muscle PhK deficiency with myopathy)",
            "FASTING KETOSIS — prominent ketones with fasting (similar to GSD VI); "
            "mild hypoglycemia; no lactic acidosis; no hyperuricemia; "
            "distinguishes from GSD Ia where ketosis is suppressed by high insulin",
            "CARRIER FEMALES — often mild hepatomegaly + elevated transaminases; "
            "may be misdiagnosed as idiopathic hepatomegaly; "
            "X-inactivation skewing → phenotypic variability in carrier females",
            "Diagnosis: erythrocyte PhK activity (low in GSD IXa) + PHKA2 sequencing; "
            "liver biopsy: PAS-positive glycogen accumulation + PhK activity low on liver enzyme assay",
        ],
        "treatment": (
            "Diet: frequent meals; uncooked cornstarch 1-2g/kg at bedtime; "
            "high-carbohydrate diet; avoid prolonged fasting. "
            "Most patients: minimal treatment beyond childhood dietary measures. "
            "Ketotic hypoglycemia: oral glucose during fasting illness; "
            "IV dextrose if severe. "
            "Growth: nutritional support if significant growth retardation; "
            "growth hormone not indicated (not deficient). "
            "Reassurance: hepatomegaly will resolve by puberty in most; "
            "excellent prognosis communicated to families. "
            "Adults: no specific treatment; annual check if residual symptoms."
        ),
        "monitoring": [
            "Glucose: fasting glucose every 6 months in childhood; annually once asymptomatic",
            "LFTs: 6-monthly in childhood; annually once improving",
            "USS: annual liver USS until hepatomegaly resolved",
            "Growth: height/weight every 6 months (growth retardation surveillance)",
            "CK: baseline only (confirm normal — no muscle involvement)",
            "Adults: annual fasting glucose + LFTs; USS every 3 years",
            "Carrier females: LFTs annually; USS if hepatomegaly symptoms",
        ],
    },
    # -- GYS2 — GSD 0, Liver Glycogen Synthase Deficiency --------------------------------
    {
        "gene": "GYS2",
        "alt_name": (
            "GYS2 (GYS2-703aa-12p12.1 / AR — GSD-0-Liver-Glycogen-Synthase-Deficiency — "
            "FASTING-HYPOGLYCEMIA-PLUS-HYPERKETONAEMIA-WITHOUT-HEPATOMEGALY-PATHOGNOMONIC — "
            "NO-HEPATOMEGALY-Cannot-Synthesize-Glycogen-No-Storage-No-Enlargement — "
            "Postprandial-HYPERGLYCEMIA — "
            "NO-Lactic-Acidosis — "
            "HIGH-PROTEIN-DIET-Plus-Frequent-Feeds)"
        ),
        "protein": (
            "GYS2 -- 12p12.1 AR -- GYS2-703aa -- "
            "Liver-Glycogen-Synthase-80kDa-Rate-Limiting-Enzyme-Glycogen-Synthesis-UDP-Glucose-to-Glycogen -- "
            "GSD-0-OMIM-240600 -- "
            "FASTING-HYPOGLYCEMIA-PLUS-HYPERKETONAEMIA-WITHOUT-HEPATOMEGALY-PATHOGNOMONIC-Triad -- "
            "NO-HEPATOMEGALY-Cannot-Synthesize-Glycogen-No-Storage-Accumulation-No-Enlargement -- "
            "Postprandial-HYPERGLYCEMIA-Glucose-Cannot-Be-Stored-As-Glycogen -- "
            "NO-LACTIC-ACIDOSIS-Distinguishes-From-GSD-Ia -- "
            "HIGH-PROTEIN-DIET-Protein-As-Alternative-Fuel-Gluconeogenesis -- "
            "Frequent-Feeds-Uncooked-Cornstarch-To-Prevent-Fasting-Hypoglycemia -- "
            "OMIM-Gene-GYS2-138571-Disease-GSD0-240600"
        ),
        "locus": "12p12.1",
        "protein_size": "703 aa / 80 kDa",
        "inheritance": (
            "AR (biallelic); "
            "GYS2 = liver glycogen synthase — rate-limiting enzyme for glycogen synthesis; "
            "catalyses transfer of glucose from UDP-glucose to growing glycogen chain; "
            "deficiency → CANNOT SYNTHESIZE LIVER GLYCOGEN → no glycogen stored → no hepatomegaly; "
            "fasting: no glycogen reserve → rapid fasting hypoglycemia + compensatory FAO → "
            "prominent hyperketonemia; "
            "postprandial: glucose cannot be stored → postprandial hyperglycemia; "
            "NO lactic acidosis (G6Pase + glycolysis intact); "
            "no hyperuricemia; no hyperlipidemia; "
            "PATHOGNOMONIC TRIAD: fasting hypoglycemia + hyperketonemia WITHOUT hepatomegaly; "
            "distinguishes GSD 0 from all storage GSDs (which all have hepatomegaly); "
            "diagnosis: liver GYS2 enzyme assay + GYS2 sequencing; "
            "often underdiagnosed — no hepatomegaly so often missed"
        ),
        "key_features": [
            "FASTING HYPOGLYCEMIA + HYPERKETONAEMIA WITHOUT HEPATOMEGALY — PATHOGNOMONIC TRIAD: "
            "the combination of ketotic hypoglycemia with complete absence of hepatomegaly "
            "is unique to GSD 0; all other hepatic GSDs have hepatomegaly; "
            "GSD 0 is unique because it is a SYNTHETIC defect (cannot make glycogen), "
            "not a DEGRADATION defect (cannot break it down)",
            "NO HEPATOMEGALY — PATHOGNOMONIC AND PARADOXICAL: GYS2 deficiency means NO glycogen "
            "can be synthesised → no storage → NO hepatomegaly; "
            "abdominal examination NORMAL; liver USS NORMAL size; "
            "this absence of hepatomegaly in a child with ketotic hypoglycemia is the diagnostic clue",
            "POSTPRANDIAL HYPERGLYCEMIA — glucose absorbed from meals cannot be stored as glycogen; "
            "postprandial glucose peaks 12-15 mmol/L; "
            "pattern: high postprandial glucose → rapid fasting hypoglycemia (no glycogen buffer); "
            "continuous glucose monitoring shows characteristic saw-tooth pattern",
            "NO LACTIC ACIDOSIS — G6Pase fully intact; gluconeogenesis normal; lactate cleared normally; "
            "distinguishes from GSD Ia (high lactate) and GSD III (mild lactate); "
            "key negative finding in metabolic screen",
            "HIGH-PROTEIN DIET — protein provides alanine + other gluconeogenic amino acids; "
            "compensates for absent glycogen reserve; muscle protein catabolism → gluconeogenesis; "
            "combined with frequent feeds + uncooked cornstarch to prevent hypoglycemia",
            "KETOTIC HYPOGLYCEMIA PATTERN — prominently ketotic at hypoglycemia (urine ketones 3-4+); "
            "FAO compensates for absent glycogen → ketogenesis; "
            "distinguishes from hyperinsulinism (suppressed ketones) and GSD Ia (no ketones)",
            "Often underdiagnosed/delayed diagnosis — no hepatomegaly means GSD not initially suspected; "
            "diagnosis often after investigation of recurrent ketotic hypoglycemia in childhood; "
            "MRI liver: normal liver size on imaging clinches distinction from storage GSDs",
            "Adults: generally asymptomatic; postprandial hyperglycemia may persist; "
            "diabetes risk with age; annual fasting glucose + HbA1c monitoring",
        ],
        "treatment": (
            "Frequent feeds: every 3-4 hours; no prolonged fasting. "
            "Uncooked cornstarch: 1-2g/kg at bedtime (slow-release glucose overnight). "
            "High-protein diet: 2-3g/kg/day — gluconeogenic substrate. "
            "Avoid simple sugars: reduces postprandial hyperglycemia peaks. "
            "Sick-day rules: increase frequency of feeds; monitor glucose; "
            "IV dextrose if severe hypoglycemia (cannot respond to glucagon — no glycogen). "
            "Adults: balanced diet; avoid prolonged fasting; annual HbA1c + glucose. "
            "No hepatomegaly/adenoma/HCC surveillance needed (no glycogen storage). "
            "Continuous glucose monitor: useful for managing saw-tooth glucose pattern."
        ),
        "monitoring": [
            "Glucose: fasting glucose every 6 months; CGM for postprandial pattern assessment",
            "Ketones: blood/urine ketones during fasting illness (hyperketonaemia monitoring)",
            "HbA1c: annually in adults (postprandial hyperglycemia → diabetes risk)",
            "LFTs: annually (confirm persistently normal — no liver disease)",
            "USS: at diagnosis (confirm no hepatomegaly); every 3-5 years thereafter",
            "Height/weight: 6-monthly in childhood (growth monitoring)",
            "Adults: annual fasting glucose + HbA1c + LFTs; no HCC/adenoma surveillance",
        ],
    },
]

# ──────────────────────────────────────────────────────────────────
# Patient simulation
# ──────────────────────────────────────────────────────────────────
def _make_cohort(gene_entry: dict, seed: int) -> list[dict]:
    random.seed(seed)
    gene = gene_entry["gene"]
    n = 40
    patients = []

    # Gene-specific parameter tables
    onset_ranges = {
        "G6PC":    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 3],          # neonatal/infantile
        "SLC37A4": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4],          # neonatal/infantile
        "GAA":     [0, 0, 0, 0, 1, 10, 15, 20, 25, 30, 35, 40, 45, 50], # bimodal IOPD/LOPD
        "AGL":     [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 8],          # infantile/early childhood
        "GBE1":    [0, 0, 0, 0, 1, 1, 2, 40, 45, 50, 55, 60, 62, 65],   # bimodal classic/APBD
        "PYGL":    [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8],          # early childhood
        "PHKA2":   [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 8],          # early childhood
        "GYS2":    [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6],          # early childhood hypoglycemia
    }

    ck_ranges = {
        "G6PC":    (50, 200),       # modest (no myopathy)
        "SLC37A4": (50, 200),       # identical to G6PC
        "GAA":     (500, 5000),     # elevated (IOPD/LOPD)
        "AGL":     (200, 2000),     # IIIa only elevated
        "GBE1":    (50, 500),       # variable
        "PYGL":    (30, 150),       # mild (benign course)
        "PHKA2":   (30, 100),       # low (no myopathy)
        "GYS2":    (20, 80),        # very low (no muscle involvement)
    }

    ambulant_prob = {
        "G6PC":    0.95,
        "SLC37A4": 0.90,
        "GAA":     0.70,   # IOPD severe, LOPD variable
        "AGL":     0.85,
        "GBE1":    0.70,   # classic form severe
        "PYGL":    0.98,
        "PHKA2":   0.98,
        "GYS2":    0.99,
    }

    ventilator_prob = {
        "G6PC":    0.03,
        "SLC37A4": 0.05,
        "GAA":     0.35,   # IOPD high; LOPD respiratory failure
        "AGL":     0.08,
        "GBE1":    0.15,
        "PYGL":    0.01,
        "PHKA2":   0.01,
        "GYS2":    0.01,
    }

    cardiac_prob = {
        "G6PC":    0.05,   # platelet dysfunction not cardiac per se; rarely cardiac
        "SLC37A4": 0.05,
        "GAA":     0.55,   # IOPD massive HCM; LOPD no cardiac
        "AGL":     0.15,   # cardiomyopathy reported in IIIa
        "GBE1":    0.30,   # cardiomyopathy in classic/neuromuscular form
        "PYGL":    0.02,
        "PHKA2":   0.02,
        "GYS2":    0.02,
    }

    liver_prob = {
        "G6PC":    1.00,   # always liver involved
        "SLC37A4": 1.00,
        "GAA":     0.80,   # hepatomegaly in IOPD; less in LOPD
        "AGL":     1.00,
        "GBE1":    0.95,
        "PYGL":    1.00,
        "PHKA2":   1.00,
        "GYS2":    0.00,   # PATHOGNOMONIC: NO hepatomegaly
    }

    # Key variants per gene
    variants = {
        "G6PC":    ["p.R83C", "p.Q347X", "p.G188S", "p.R83H"],
        "SLC37A4": ["p.G149E", "p.W118R", "p.G339C"],
        "GAA":     ["c.-32-13T>G", "p.W746S", "p.Del exon18"],
        "AGL":     ["p.R1228X", "p.W1327X", "c.4260_4262delGAG"],
        "GBE1":    ["p.Y329S (Ashkenazi NP)", "p.R515C classic", "p.G149R"],
        "PYGL":    ["p.G233D", "p.R146W", "c.2349+1G>A"],
        "PHKA2":   ["p.E261K", "p.R531W", "c.3614G>A"],
        "GYS2":    ["p.M48T", "p.L237P", "c.1412G>A"],
    }

    key_findings = {
        "G6PC": [
            "Metabolic quartet: lactic acidosis + hyperuricemia + hyperlipidemia + fasting hypoglycemia",
            "Massive hepatomegaly + renal enlargement on USS",
            "Glucagon stimulation test: no glucose rise",
            "Hepatic adenoma detected on surveillance USS",
            "Platelet dysfunction + bleeding diathesis",
            "Doll's face + short stature + protuberant abdomen",
        ],
        "SLC37A4": [
            "Metabolic quartet identical to GSD Ia + cyclic neutropenia",
            "ANC <500/μL — recurrent infections + oral ulcers",
            "Crohn's-like IBD + perianal disease",
            "G-CSF therapy initiated for neutropenia",
            "Empagliflozin started — G-CSF dose reduced",
            "Hepatomegaly + renal enlargement + lactic acidosis",
        ],
        "GAA": [
            "IOPD: massive HCM + hypotonia + respiratory failure — ERT initiated within 3 months",
            "LOPD: proximal weakness + FVC 45% — NIV commenced",
            "CRIM-negative — ITI protocol before ERT",
            "NBS positive: DBS GAA activity low — confirmed by sequencing",
            "Avalglucosidase alfa (Nexviazyme) — superior response to alglucosidase alfa",
            "c.-32-13T>G homozygous — LOPD late presentation age 35yr",
        ],
        "AGL": [
            "GSD IIIa: hepatomegaly + elevated CK — liver + muscle phenotype",
            "GSD IIIb: hepatomegaly only — normal CK; liver-restricted phenotype",
            "Limit dextrin on glycogen structure analysis",
            "Hepatic fibrosis on fibroscan — cirrhosis surveillance initiated",
            "High-protein diet commenced — muscle strength improved",
            "Glucagon test: partial blunted response (not flat as in GSD Ia)",
        ],
        "GBE1": [
            "Classic hepatic form: progressive cirrhosis — liver transplant listed",
            "p.Y329S Ashkenazi: non-progressive hepatomegaly — no transplant required",
            "Polyglucosan bodies: PAS-positive diastase-resistant — biopsy pathognomonic",
            "APBD adult: spastic paraparesis + peripheral neuropathy age 52yr",
            "Post-liver-transplant: hepatic disease resolved; monitoring extra-hepatic",
            "Cardiomyopathy detected — echocardiogram mandated pre-transplant",
        ],
        "PYGL": [
            "Hepatomegaly prominent childhood — resolved by age 14yr",
            "Mild fasting hypoglycemia + prominent ketosis",
            "No lactic acidosis; no hyperuricemia; CK normal",
            "No myopathy; no cardiac involvement",
            "Cornstarch bedtime dose — no nocturnal hypoglycemia",
            "Adult: asymptomatic; liver USS normal",
        ],
        "PHKA2": [
            "Hepatomegaly + short stature + fasting ketosis in 4-year-old male",
            "X-linked — maternal uncle had similar hepatomegaly childhood",
            "Erythrocyte PhK activity low; PHKA2 sequencing confirmed",
            "Hepatomegaly resolving by age 13yr; height catching up",
            "No muscle involvement; CK normal",
            "Carrier mother: mild hepatomegaly + elevated ALT",
        ],
        "GYS2": [
            "Ketotic hypoglycemia without hepatomegaly — pathognomonic pattern",
            "Postprandial glucose 14 mmol/L — cannot store glucose as glycogen",
            "Liver USS: normal size (no hepatomegaly)",
            "No lactic acidosis; no hepatomegaly — atypical GSD presentation",
            "High-protein diet + cornstarch bedtime — hypoglycemia resolved",
            "CGM: saw-tooth pattern — high postprandial → rapid fasting dip",
        ],
    }

    treatments = {
        "G6PC": [
            "Uncooked cornstarch + nocturnal nasogastric glucose infusion",
            "Allopurinol 200mg/day for hyperuricemia",
            "Cornstarch q4h + fibrate for hypertriglyceridemia",
            "Annual USS surveillance — adenoma resected",
            "Glycosade (modified cornstarch) q8h — improved glycaemic control",
            "IV dextrose perioperatively + DDAVP for platelet dysfunction",
        ],
        "SLC37A4": [
            "Cornstarch + G-CSF 7 μg/kg/day SC — ANC maintained >1000",
            "Empagliflozin 10mg/day — G-CSF dose halved",
            "Metronidazole + mesalazine for IBD flare",
            "Cornstarch regimen identical to GSD Ia + G-CSF protocol",
            "Antimicrobial prophylaxis during neutropenic nadir",
            "Empagliflozin 25mg/day — sustained neutrophil improvement",
        ],
        "GAA": [
            "Avalglucosidase alfa 20mg/kg IV q2w — IOPD — initiated at 6 weeks NBS",
            "Alglucosidase alfa + ITI (rituximab + MTX) — CRIM-negative",
            "NIV BiPAP nocturnal + physiotherapy — LOPD respiratory management",
            "Avalglucosidase alfa — improved 6-minute walk vs alglucosidase alfa",
            "Gene therapy trial enrolled — AAV9-GAA Phase I/II",
            "NBS follow-up: ERT initiated day 18 — cardiac function normalised 6 months",
        ],
        "AGL": [
            "Uncooked cornstarch + high-protein diet 2.5g/kg/day",
            "Physiotherapy resistance training — IIIa myopathy management",
            "High-protein diet — CK improved; cornstarch nocturnal",
            "Liver transplant listed — cirrhosis with portal hypertension",
            "Annual fibroscan + AFP — no hepatic decompensation yet",
            "Cornstarch + protein supplementation — growth velocity normalised",
        ],
        "GBE1": [
            "Liver transplant age 18 months — classic hepatic cirrhosis",
            "Supportive management p.Y329S — no transplant; annual USS",
            "Physiotherapy for spasticity + APBD neuropathy",
            "Propranolol + endoscopy — varices surveillance pre-transplant",
            "Post-transplant immunosuppression + extra-hepatic monitoring",
            "Nutritional support + vitamin K pre-transplant coagulopathy",
        ],
        "PYGL": [
            "Cornstarch 1.5g/kg bedtime — fasting hypoglycemia prevented",
            "Frequent meals; no nocturnal feeds required",
            "Dietary modification only — no pharmacological treatment",
            "Annual USS until hepatomegaly resolved age 15yr",
            "Adult management: dietary advice only; annual GP review",
            "Cornstarch at illness — increased frequency during intercurrent illness",
        ],
        "PHKA2": [
            "Frequent meals + cornstarch 1g/kg bedtime",
            "Dietary management only — hepatomegaly expected to resolve",
            "Reassurance — explained transient natural history to family",
            "Cornstarch nocturnal — no more fasting hypoglycemia episodes",
            "Growth tracking — catching up by age 10yr on dietary treatment",
            "Annual USS + LFTs — hepatomegaly progressively resolving",
        ],
        "GYS2": [
            "Frequent feeds q3h + cornstarch 1.5g/kg bedtime",
            "High-protein diet 2.5g/kg + avoid prolonged fasting",
            "CGM fitted — postprandial peaks identified; cornstarch adjusted",
            "Emergency IV dextrose protocol — no glucagon response expected",
            "Frequent feeds + protein snacks — school management plan issued",
            "Cornstarch + complex carbohydrate — overnight hypoglycemia resolved",
        ],
    }

    for i in range(n):
        pid = f"{gene}-{seed}-{i+1:03d}"
        onset_age = float(random.choice(onset_ranges.get(gene, [2, 3, 4, 5])))
        current_age = onset_age + random.randint(3, 35)
        ck_lo, ck_hi = ck_ranges.get(gene, (50, 500))
        ck_peak = random.randint(ck_lo, ck_hi)

        ambulant = random.random() < ambulant_prob.get(gene, 0.90)
        ventilator = random.random() < ventilator_prob.get(gene, 0.05)
        cardiac_involvement = random.random() < cardiac_prob.get(gene, 0.05)
        liver_involvement = random.random() < liver_prob.get(gene, 0.90)

        variant_1 = random.choice(variants.get(gene, ["p.Unknown"]))
        variant_2_pool = [v for v in variants.get(gene, []) if v != variant_1]
        variant_2 = random.choice(variant_2_pool) if variant_2_pool and random.random() < 0.55 else None

        key_finding = random.choice(key_findings.get(gene, ["Hepatic glycogen storage"]))
        treatment = random.choice(treatments.get(gene, ["Dietary management"]))

        # Gene-specific extra flags
        metabolic_quartet = (gene in ("G6PC", "SLC37A4")) and random.random() < 0.95
        cyclic_neutropenia = (gene == "SLC37A4") and random.random() < 0.90
        empagliflozin_treated = (gene == "SLC37A4") and random.random() < 0.40
        iopd_form = (gene == "GAA") and (onset_age <= 1.0) and random.random() < 0.85
        crim_negative = (gene == "GAA") and random.random() < 0.20
        nbs_detected = (gene == "GAA") and random.random() < 0.80
        limit_dextrinosis = (gene == "AGL") and random.random() < 0.95
        polyglucosan_deposits = (gene == "GBE1") and random.random() < 0.90
        hepatic_adenoma = (gene in ("G6PC", "SLC37A4")) and (current_age >= 20) and random.random() < 0.70
        benign_course = (gene in ("PYGL", "PHKA2", "GYS2"))
        no_hepatomegaly = (gene == "GYS2")
        postprandial_hyperglycemia = (gene == "GYS2") and random.random() < 0.90

        patients.append({
            "id": pid,
            "gene": gene,
            "onset_age": onset_age,
            "current_age": round(current_age, 1),
            "ck_peak": ck_peak,
            "ambulant": ambulant,
            "ventilator": ventilator,
            "cardiac_involvement": cardiac_involvement,
            "liver_involvement": liver_involvement,
            "variant_1": variant_1,
            "variant_2": variant_2,
            "key_finding": key_finding,
            "treatment": treatment,
            "metabolic_quartet": metabolic_quartet,
            "cyclic_neutropenia": cyclic_neutropenia,
            "empagliflozin_treated": empagliflozin_treated,
            "iopd_form": iopd_form,
            "crim_negative": crim_negative,
            "nbs_detected": nbs_detected,
            "limit_dextrinosis": limit_dextrinosis,
            "polyglucosan_deposits": polyglucosan_deposits,
            "hepatic_adenoma": hepatic_adenoma,
            "benign_course": benign_course,
            "no_hepatomegaly": no_hepatomegaly,
            "postprandial_hyperglycemia": postprandial_hyperglycemia,
        })
    return patients


def _aggregate_cohort():
    all_patients = []
    for idx, entry in enumerate(GSD_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(entry, seed))
    return all_patients


# ──────────────────────────────────────────────────────────────────
# API response builders
# ──────────────────────────────────────────────────────────────────
def overview() -> dict:
    cohort = _aggregate_cohort()
    total = len(cohort)
    ambulant = sum(1 for p in cohort if p["ambulant"])
    ventilator = sum(1 for p in cohort if p["ventilator"])
    cardiac = sum(1 for p in cohort if p["cardiac_involvement"])
    liver = sum(1 for p in cohort if p["liver_involvement"])
    metabolic_quartet = sum(1 for p in cohort if p["metabolic_quartet"])
    cyclic_neutropenia = sum(1 for p in cohort if p["cyclic_neutropenia"])
    iopd = sum(1 for p in cohort if p["iopd_form"])
    nbs = sum(1 for p in cohort if p["nbs_detected"])
    adenoma = sum(1 for p in cohort if p["hepatic_adenoma"])
    no_hepatomegaly = sum(1 for p in cohort if p["no_hepatomegaly"])
    avg_onset = round(sum(p["onset_age"] for p in cohort) / total, 1)
    avg_ck = round(sum(p["ck_peak"] for p in cohort) / total, 0)

    gene_counts = {}
    for g in GSD_GENES:
        gn = g["gene"]
        subset = [p for p in cohort if p["gene"] == gn]
        gene_counts[gn] = {
            "n": len(subset),
            "ambulant_pct": round(100 * sum(1 for p in subset if p["ambulant"]) / len(subset)),
            "ventilator_pct": round(100 * sum(1 for p in subset if p["ventilator"]) / len(subset)),
            "cardiac_pct": round(100 * sum(1 for p in subset if p["cardiac_involvement"]) / len(subset)),
            "liver_pct": round(100 * sum(1 for p in subset if p["liver_involvement"]) / len(subset)),
            "avg_onset": round(sum(p["onset_age"] for p in subset) / len(subset), 1),
            "avg_ck_peak": round(sum(p["ck_peak"] for p in subset) / len(subset), 0),
        }

    return {
        "atlas": "Hereditary-Hepatic-Glycogen-Storage-Disease-Atlas",
        "subtitle": "Complete 8-Gene Hepatic GSD Spectrum Atlas",
        "genes": [g["gene"] for g in GSD_GENES],
        "gene_count": len(GSD_GENES),
        "total_patients": total,
        "seeds": list(range(SEED_BASE, SEED_BASE + len(GSD_GENES))),
        "kpis": {
            "total_patients": total,
            "ambulant_pct": round(100 * ambulant / total),
            "ventilator_pct": round(100 * ventilator / total),
            "cardiac_involvement_pct": round(100 * cardiac / total),
            "liver_involvement_pct": round(100 * liver / total),
            "metabolic_quartet_pct": round(100 * metabolic_quartet / total),
            "cyclic_neutropenia_pct": round(100 * cyclic_neutropenia / total),
            "iopd_form_pct": round(100 * iopd / total),
            "nbs_detected_pct": round(100 * nbs / total),
            "hepatic_adenoma_pct": round(100 * adenoma / total),
            "no_hepatomegaly_pct": round(100 * no_hepatomegaly / total),
            "avg_onset_years": avg_onset,
            "avg_ck_peak_iul": int(avg_ck),
        },
        "gene_summary": gene_counts,
        "pathognomonic_features": {
            "G6PC":    "METABOLIC QUARTET: lactic acidosis + hyperuricemia + hyperlipidemia + fasting hypoglycemia; hepatomegaly + renal enlargement; glucagon test FAILS",
            "SLC37A4": "IDENTICAL to GSD Ia PLUS CYCLIC NEUTROPENIA; Crohn's-like IBD; G-CSF mandatory; empagliflozin 2023",
            "GAA":     "IOPD: massive HCM + hypotonia; LOPD: proximal myopathy + respiratory failure NO cardiomyopathy; CRIM status critical; NBS detected",
            "AGL":     "LIMIT DEXTRINOSIS; IIIa (liver+muscle) vs IIIb (liver-only); CK elevated only IIIa; hepatomegaly → cirrhosis risk",
            "GBE1":    "POLYGLUCOSAN DEPOSITS; CLASSIC: cirrhosis → liver transplant; p.Y329S Ashkenazi non-progressive; APBD adult neurological",
            "PYGL":    "BENIGN COURSE: hepatomegaly resolves with age; hyperketonemia prominent; NO myopathy/cardiac; adults asymptomatic",
            "PHKA2":   "XLR; TRANSIENT course — resolves by puberty; NO muscle involvement; most common childhood GSD after GSD III",
            "GYS2":    "FASTING HYPOGLYCEMIA + HYPERKETONAEMIA WITHOUT HEPATOMEGALY — PATHOGNOMONIC; postprandial hyperglycemia; NO lactic acidosis",
        },
        "inheritance_map": {
            "G6PC": "AR", "SLC37A4": "AR", "GAA": "AR", "AGL": "AR",
            "GBE1": "AR", "PYGL": "AR", "PHKA2": "XLR", "GYS2": "AR",
        },
        "protein_sizes": {g["gene"]: g["protein_size"] for g in GSD_GENES},
        "loci": {g["gene"]: g["locus"] for g in GSD_GENES},
        "key_pharmacological_distinctions": [
            "G6PC: cornstarch therapy + nocturnal feeds; allopurinol for gout; DDAVP perioperative; AAV gene therapy Phase III 2024-2026",
            "SLC37A4: identical dietary to GSD Ia + G-CSF mandatory for neutropenia; empagliflozin 2023 approved; antimicrobial prophylaxis",
            "GAA: avalglucosidase alfa (Nexviazyme) preferred over alglucosidase alfa; ITI mandatory for CRIM-negative; NBS → ERT within 1 month",
            "AGL: uncooked cornstarch + HIGH-PROTEIN DIET 2-3g/kg; physiotherapy for IIIa; liver transplant if cirrhosis (does NOT cure muscle)",
            "GBE1: liver transplant curative for hepatic form; NO treatment changes p.Y329S non-progressive; APBD supportive only",
            "PYGL: minimal treatment; cornstarch bedtime; most patients resolve without pharmacological therapy by adulthood",
            "PHKA2: dietary management + cornstarch; G-CSF NOT needed; symptoms resolve by puberty — reassurance is key treatment",
            "GYS2: frequent feeds + high-protein diet; cornstarch bedtime; CGM for saw-tooth glucose pattern; NO HCC/adenoma surveillance needed",
        ],
        "critical_treatment_alerts": {
            "G6PC":    "GLUCAGON WILL NOT RAISE GLUCOSE — IV dextrose mandatory in crisis; DDAVP perioperative; adenoma HCC surveillance annual",
            "SLC37A4": "G-CSF mandatory when ANC <500; empagliflozin reduces G-CSF need; metabolic management identical to GSD Ia",
            "GAA":     "CRIM-negative ITI before ERT — otherwise inhibitory antibodies destroy ERT efficacy; avalglucosidase alfa preferred 2024",
            "AGL":     "HIGH-PROTEIN DIET 2-3g/kg (not just cornstarch); liver transplant corrects LIVER not MUSCLE in IIIa",
            "GBE1":    "p.Y329S does NOT need transplant; classic form needs transplant URGENTLY before decompensation",
            "PYGL":    "Benign — resist over-treating; most patients need only dietary advice; hepatomegaly resolves spontaneously",
            "PHKA2":   "Reassurance is the primary treatment — X-linked; transient; resolves by puberty; NO muscle involvement (unlike GSD IXd)",
            "GYS2":    "NO HEPATOMEGALY is the diagnostic clue — GSD 0 missed because no liver enlargement; glucagon WILL NOT WORK (no glycogen to release)",
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
        "atlas": "Hereditary-Hepatic-Glycogen-Storage-Disease-Atlas",
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
        "atlas": "Hereditary-Hepatic-Glycogen-Storage-Disease-Atlas",
        "glossary": {
            "Glycogen Storage Disease (GSD)": (
                "Hereditary disorders of glycogen metabolism caused by enzyme defects in glycogen "
                "synthesis, degradation, or regulation. Hepatic GSDs primarily affect liver glycogen "
                "metabolism; clinical hallmarks: hepatomegaly, fasting hypoglycemia, and metabolic "
                "disturbances varying by enzyme block. CK elevation signifies muscle involvement "
                "(GSD IIIa, GSD II) whereas liver-only GSDs have normal CK."
            ),
            "Metabolic Quartet (GSD Ia/GSD Ib)": (
                "PATHOGNOMONIC simultaneous biochemical tetrad of GSD Ia (G6PC) and GSD Ib (SLC37A4): "
                "(1) Lactic acidosis — G6P → glycolysis → excess lactate; "
                "(2) Hyperuricemia — G6P → pentose phosphate pathway → purine overproduction + "
                "lactate competes with urate for renal excretion; "
                "(3) Hyperlipidemia — G6P → lipogenesis + triglyceride overproduction; "
                "(4) Fasting hypoglycemia — G6Pase absent → no glucose release from liver. "
                "All four present simultaneously = GSD Ia/Ib until proven otherwise."
            ),
            "Cyclic Neutropenia (GSD Ib/SLC37A4)": (
                "PATHOGNOMONIC feature distinguishing GSD Ib from GSD Ia. "
                "G6P translocase (SLC37A4) expressed in neutrophil ER → LOF → neutrophil energy "
                "deficit → cyclic ANC nadirs <500/μL. "
                "Consequences: recurrent bacterial infections, oral/perianal ulcers, Crohn's-like IBD. "
                "G-CSF mandatory when ANC <500/μL. "
                "Empagliflozin (SGLT2i): approved 2023 — reduces 1,5-anhydroglucitol neutrophil toxicity."
            ),
            "Pompe Disease / GSD II (GAA)": (
                "Lysosomal glycogen storage disease — the only GSD caused by a lysosomal enzyme defect. "
                "IOPD (Infantile-Onset): null alleles → massive HCM + hypotonia → death <1yr without ERT; "
                "NBS → ERT within first month → near-normal cardiac function at 5yr. "
                "LOPD (Late-Onset): c.-32-13T>G (IVS1) residual splicing → proximal myopathy + "
                "respiratory failure; NO cardiomyopathy. "
                "Enzyme replacement therapy (ERT): avalglucosidase alfa (Nexviazyme) FDA 2021 preferred."
            ),
            "CRIM Status (GAA/Pompe)": (
                "Cross-Reactive Immunological Material status: "
                "CRIM-positive: patient makes some (possibly truncated) GAA protein → tolerates ERT. "
                "CRIM-negative: null/null — makes NO GAA protein → immune system sees ERT as foreign "
                "→ high-titre inhibitory antibodies → dramatically worsened ERT response. "
                "ITI (Immune Tolerance Induction): rituximab + methotrexate + IVIG — mandatory in "
                "CRIM-negative before/at ERT start; prevents antibody formation."
            ),
            "Limit Dextrinosis (GSD III/AGL)": (
                "Pathological glycogen with abnormally short outer branch chains (limit dextrin). "
                "AGL = bifunctional glycogen debranching enzyme (glucanotransferase + glucosidase); "
                "deficiency → catabolism stops at branch points → limit dextrin accumulates. "
                "PAS stain: excess glycogen in liver (± muscle in IIIa). "
                "GSD IIIa: liver + muscle (85%); CK elevated. "
                "GSD IIIb: liver only (15%); CK normal."
            ),
            "Polyglucosan Bodies (GSD IV/GBE1)": (
                "Poorly branched, amylopectin-like polysaccharide accumulating in GBE1 deficiency. "
                "PAS-positive + diastase-RESISTANT (unlike normal glycogen, which is diastase-sensitive). "
                "Accumulate in: hepatocytes (classic form), neurons + peripheral nerves (APBD). "
                "Classic form: progressive cirrhosis → liver transplant. "
                "Adult-onset APBD: neurological — spastic paraparesis + peripheral neuropathy + "
                "bladder dysfunction ≥40yr."
            ),
            "Cornstarch Therapy": (
                "Uncooked corn starch (complex glucose polymer): absorbed slowly → sustained glucose "
                "release → prevents fasting hypoglycemia. "
                "Standard dose: 2g/kg q4h (adults q6-8h). "
                "Glycosade (modified cornstarch): q8-10h extended release. "
                "Used in: GSD Ia (G6PC), GSD Ib (SLC37A4), GSD III (AGL), GSD VI (PYGL), GSD IXa (PHKA2), GSD 0 (GYS2). "
                "NOT effective in GSD II (Pompe — lysosomal, not cytoplasmic glycogen)."
            ),
            "GSD 0 Paradox (GYS2)": (
                "GSD 0 is unique among GSDs: it is a SYNTHETIC defect (cannot make glycogen), "
                "not a degradation defect. "
                "Consequence: NO hepatomegaly (nothing to store → no enlargement). "
                "PATHOGNOMONIC TRIAD: fasting hypoglycemia + hyperketonemia WITHOUT hepatomegaly. "
                "Postprandial hyperglycemia (cannot store glucose). "
                "Often underdiagnosed because 'GSD without hepatomegaly' is counterintuitive."
            ),
            "Empagliflozin (SGLT2i) for GSD Ib": (
                "FDA/EMA approved 2023 for GSD Ib-associated neutropenia. "
                "Mechanism: SGLT2 inhibition blocks intestinal reabsorption of 1,5-anhydroglucitol (1,5-AG); "
                "1,5-AG-6-phosphate accumulates in neutrophils in GSD Ib → neutrophil dysfunction; "
                "empagliflozin reduces 1,5-AG levels → improves neutrophil function. "
                "Clinical: reduces G-CSF dose requirements; improves IBD symptoms. "
                "Monitoring: UTI, urinary ketones, eGFR."
            ),
            "Hepatic Adenoma and HCC Risk (GSD Ia/Ib)": (
                "Hepatic adenomas: develop in 70%+ adults with GSD Ia/Ib >20yr. "
                "Mechanism: poor metabolic control → hepatocyte proliferation → adenoma. "
                "Malignant transformation: ~10% → hepatocellular carcinoma (HCC). "
                "Surveillance: annual liver USS + AFP mandatory from age 15yr. "
                "MRI if adenoma ≥3cm or rapid growth. "
                "Resection/ablation if malignant transformation confirmed. "
                "Improved metabolic control (cornstarch + nocturnal feeds) reduces adenoma burden."
            ),
            "Glucagon Stimulation Test in GSD": (
                "Diagnostic discriminator: 1mg glucagon IV/IM → measure blood glucose at 0, 15, 30, 60 min. "
                "GSD Ia: FLAT response — no glucose rise (G6Pase absent → G6P cannot be hydrolysed). "
                "GSD III: BLUNTED partial response (some gluconeogenesis via alternate pathways). "
                "GSD VI/IXa: near-normal or mild reduction (glycogenolysis partially reduced). "
                "GSD 0: paradoxical — some gluconeogenesis response but no glycogenolysis. "
                "Important: galactose + fructose infusion also fail in GSD Ia (both metabolised via G6P)."
            ),
            "Liver Transplantation in GSD": (
                "GSD IV (GBE1 classic): CURATIVE for hepatic disease; transplant provides donor "
                "GBE1 enzyme in new liver; polyglucosan cleared from hepatocytes. "
                "Caution: does NOT cure extra-hepatic manifestations (cardiomyopathy, neuromuscular). "
                "GSD III (AGL): corrects liver disease but NOT muscle disease in IIIa; "
                "myopathy continues to progress post-transplant. "
                "GSD Ia/Ib: transplant for HCC or refractory adenomas; "
                "corrects hepatic metabolic defect but careful perioperative glucose management mandatory."
            ),
        },
        "diagnostic_algorithm": [
            "1. Hepatomegaly + fasting hypoglycemia in child → hepatic GSD screen",
            "2. Assess for METABOLIC QUARTET: lactic acidosis + hyperuricemia + hyperlipidemia → "
            "   GSD Ia (G6PC) or GSD Ib (SLC37A4); glucagon test: FLAT response confirms G6Pase block",
            "3. CYCLIC NEUTROPENIA present? Yes → GSD Ib (SLC37A4); No → GSD Ia (G6PC); "
            "   check ANC, oral ulcers, IBD-like symptoms",
            "4. No metabolic quartet → check CK: elevated CK + hepatomegaly → GSD IIIa (AGL); "
            "   normal CK + hepatomegaly → GSD IIIb / GSD VI / GSD IXa",
            "5. Glucagon test partial (blunted) response → GSD III (AGL); "
            "   check glycogen structure: limit dextrin pattern confirms AGL",
            "6. Massive HCM + hypotonia in infant → GAA (Pompe IOPD); "
            "   DBS alpha-glucosidase activity → NBS confirmation; check CRIM status",
            "7. Proximal myopathy + respiratory failure (adult) NO cardiac → "
            "   GAA (Pompe LOPD); c.-32-13T>G sequencing; FVC + polysomnography",
            "8. Progressive liver failure + cirrhosis (infant/child) → GBE1 (Andersen); "
            "   PAS-diastase-resistant deposits on biopsy; p.Y329S (Ashkenazi) → non-progressive form",
            "9. Hepatomegaly + KETOTIC hypoglycemia + NO lactic acidosis + NO hyperuricemia "
            "   + MILD course → GSD VI (PYGL) or GSD IXa (PHKA2); "
            "   X-linked family history → PHKA2; erythrocyte PhK activity",
            "10. KETOTIC HYPOGLYCEMIA WITHOUT HEPATOMEGALY (liver USS normal) → GSD 0 (GYS2); "
            "    postprandial hyperglycemia on CGM; liver GYS2 enzyme + sequencing",
            "11. Gene panel: G6PC, SLC37A4, GAA, AGL, GBE1, PYGL, PHKA2 (X-linked), GYS2",
            "12. APBD (adult ≥40yr, neurological + peripheral neuropathy) → GBE1 sequencing; "
            "    sural nerve biopsy: PAS-diastase-resistant polyglucosan bodies",
        ],
        "references": [
            "Kishnani PS et al. Glycogen storage disease type I: diagnosis and management guidelines. "
            "Genet Med. 2014;16(11):e1.",
            "Melis D et al. GSD Ib and SGLT2 inhibitors (empagliflozin). NEJM. 2023.",
            "Kishnani PS et al. International Pompe disease consortium: ERT guidelines. "
            "J Inherit Metab Dis. 2009.",
            "Case LE et al. First international consensus conference on Pompe disease. "
            "Mol Genet Metab. 2012.",
            "Rake JP et al. GSD Ia European study. Eur J Pediatr. 2002.",
            "Demo E et al. GSD III: clinical and genetic features. Pediatrics. 2007.",
            "Bali DS et al. Pompe disease. In: Adam MP et al. GeneReviews. NCBI. 2021.",
            "Bhatt KP et al. Andersen disease (GSD IV). GeneReviews. 2024.",
        ],
        "standards": [
            "ACMG/AMP Variant Interpretation Framework (Richards et al. 2015)",
            "SSIEM (Society for the Study of Inborn Errors of Metabolism) GSD Guidelines",
            "FDA Approval: Nexviazyme (avalglucosidase alfa) for Pompe disease (2021)",
            "EMA/FDA Approval: Empagliflozin for GSD Ib neutropenia (2023)",
            "BIMDG (British Inherited Metabolic Disease Group) GSD Emergency Guidelines",
            "RCPCH Newborn Bloodspot Screening Programme — GAA (Pompe) panel",
            "ESPE/ESPKU Hepatic GSD Clinical Practice Guidelines 2023",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2)[:3000])
    print("\n=== DEFINITIONS ===")
    defs = definitions()
    print(json.dumps({"atlas": defs["atlas"], "glossary_keys": list(defs["glossary"].keys())}, indent=2))
    print("\n=== BREAKDOWN (first patient) ===")
    bk = breakdown()
    print(json.dumps({"total": bk["total"], "first_patient": bk["patients"][0]}, indent=2))
