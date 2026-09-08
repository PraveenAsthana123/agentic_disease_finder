#!/usr/bin/env python3
"""Hereditary-Hepatic-Disease-Atlas — Complete 8-Gene Hereditary Liver Disease Atlas.

ATP7B    (copper-transporting ATPase beta; 1465 aa; 13q14.3; AR;
          Wilson disease — Kayser-Fleischer rings PATHOGNOMONIC;
          seed SEED_BASE+0).
HFE      (homeostatic iron regulator; 348 aa; 6p21.3; AR;
          Hereditary Haemochromatosis Type 1 — C282Y homozygous 85pct;
          seed SEED_BASE+1).
SERPINA1 (alpha-1-antitrypsin; 418 aa; 14q32.13; AR/codominant;
          Alpha-1-Antitrypsin Deficiency — PiZZ hepatic cirrhosis + pulmonary emphysema;
          seed SEED_BASE+2).
JAG1     (Jagged-1; 1218 aa; 20p12.2; AD;
          Alagille syndrome — butterfly vertebrae + paucity of bile ducts;
          seed SEED_BASE+3).
ABCB11   (bile-salt export pump / BSEP; 1321 aa; 2q31.1; AR;
          PFIC2 — most severe progressive familial intrahepatic cholestasis;
          seed SEED_BASE+4).
ATP8B1   (ATPase phospholipid transporting 8B1 / FIC1; 1251 aa; 18q21.31; AR;
          PFIC1 / Byler disease — extrahepatic manifestations + steatosis post-LT;
          seed SEED_BASE+5).
SLC25A13 (citrin / mitochondrial aspartate-glutamate carrier 2; 675 aa; 7q21.3; AR;
          Citrinemia / NICCD — neonatal intrahepatic cholestasis;
          seed SEED_BASE+6).
NPC1     (Niemann-Pick C1; 1278 aa; 18q11.2; AR;
          NPC disease — vertical supranuclear gaze palsy PATHOGNOMONIC + hepatosplenomegaly;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2054-2061).
"""

import random

SEED_BASE = 2054

HD_GENES = [
    # -- ATP7B — Wilson Disease (AR) -------------------------------------------
    {
        "gene": "ATP7B",
        "alt_name": "ATP7B (ATP7B-1465aa-13q14.3 / AR — Wilson-Disease-Copper-ATPase — Kayser-Fleischer-Rings-PATHOGNOMONIC — Penicillamine-Trientine-Zinc)",
        "protein": (
            "ATP7B -- 13q14.3 AR -- ATP7B-1465aa -- "
            "Wilson-Disease-Copper-Transporting-ATPase-Beta -- "
            "Kayser-Fleischer-Rings-KFR-Slit-Lamp-PATHOGNOMONIC-Neurological-Psychiatric -- "
            "Serum-Ceruloplasmin-LOW-24h-Urine-Copper-HIGH-Liver-Biopsy-Quantitative-Copper -- "
            "Penicillamine-Trientine-First-Line-Zinc-Maintenance-Tetrathiomolybdate -- "
            "MOST-COMMON-GENE-Wilson-Disease"
        ),
        "locus": "13q14.3",
        "protein_size": "1465 aa",
        "inheritance": (
            "AR (autosomal recessive) — ATP7B biallelic loss-of-function; "
            "Prevalence 1 in 30,000 worldwide; carrier frequency 1 in 90; "
            ">700 pathogenic variants; H1069Q most common European (30-40%); "
            "R778L most common East Asian (30-50%); "
            "Compound heterozygotes common; genotype-phenotype correlation poor; "
            "NO parent affected (both carrier) — test sibs mandatory; "
            "Sibling risk 25%; first-degree screening: serum ceruloplasmin"
        ),
        "age_of_onset": (
            "Hepatic presentation: 5-35 yr (most 10-25 yr); "
            "Neurological presentation: teens to early 30s; "
            "Psychiatric presentation: adolescence — mimics schizophrenia/depression; "
            "Neonatal fulminant hepatic failure: rare — must exclude Wilson in ALL acute liver failure in young; "
            "Coombs-negative haemolytic anaemia: hallmark of fulminant WD — icteric plasma + haematuria; "
            "Hepatic: acute hepatitis / cirrhosis / fulminant failure; "
            "Neurological: dysarthria (most common), dystonia, Parkinsonism, ataxia; "
            "Psychiatric: personality change, irritability, psychosis — often first presentation"
        ),
        "key_biomarker": (
            "Kayser-Fleischer rings: copper deposits in Descemet membrane — slit-lamp MANDATORY; "
            "Serum ceruloplasmin: low (<0.2 g/L) in 90% of neurological WD; "
            "24h urinary copper: >1.6 μmol/24h (>100 μg/24h) diagnostic; "
            ">25 μmol/24h (>1600 μg/24h) in symptomatic; "
            "Liver biopsy: quantitative copper >250 μg/g dry weight; "
            "Liver MRI: hepatic cirrhosis; 'face of giant panda' midbrain sign (neurological WD); "
            "Brain MRI: bilateral symmetrical basal ganglia + thalamus + brainstem T2 hyperintensities; "
            "ATP7B sequencing: confirm biallelic pathogenic variants; "
            "Penicillamine challenge test: urine copper >25 μmol/24h post-challenge"
        ),
        "pathognomonic": (
            "KAYSER-FLEISCHER RINGS on slit-lamp examination = Wilson disease until proven otherwise; "
            "ABSENT in majority of hepatic-only presentation (10-50%); "
            "Coombs-NEGATIVE haemolytic anaemia + acute liver failure in young adult = WD fulminant — EMERGENCY; "
            "Alkaline phosphatase LOW + haemolysis + acute liver failure = pathognomonic pattern Wilson fulminant; "
            "FACE OF GIANT PANDA SIGN on brain MRI = Wilson midbrain involvement; "
            "DISTINGUISH from other causes of cirrhosis + neuropsychiatric: always exclude WD age 5-40 yr"
        ),
        "treatment": (
            "Chelation — symptomatic patients: D-penicillamine (GI side effects, worsening neuro 20%); "
            "Trientine: better tolerated, preferred for neurological WD — do not worsen neuro; "
            "Zinc: maintenance therapy (blocks gut copper absorption); preferred in asymptomatic/pregnant; "
            "Tetrathiomolybdate: investigational — rapid copper detoxification; "
            "Liver transplantation: corrects hepatic defect — fulminant failure / decompensated cirrhosis; "
            "MONITORING: 24h urine copper + LFTs 6-monthly; "
            "AVOID: foods high in copper (shellfish, liver, nuts, chocolate) especially first 1-2 years; "
            "PREGNANCY: continue zinc (safest) or trientine — do NOT stop treatment"
        ),
        "critical_flags": [
            "KAYSER-FLEISCHER-RINGS-SLIT-LAMP-PATHOGNOMONIC",
            "Coombs-NEGATIVE-HAEMOLYSIS-ACUTE-LIVER-FAILURE-EMERGENCY",
            "AlkPhos-LOW-IN-FULMINANT-WILSON-PATHOGNOMONIC",
            "CHELATION-MAY-WORSEN-NEURO-FIRST-WEEKS-WARN-PATIENT",
            "PENICILLAMINE-AVOID-NEURO-WD-USE-TRIENTINE",
            "LIVER-TRANSPLANT-CORRECTS-HEPATIC-DEFECT",
            "EXCLUDE-WILSON-ALL-ACUTE-LIVER-FAILURE-AGE-5-40yr",
            "SIBLING-TESTING-MANDATORY",
        ],
    },
    # -- HFE — Hereditary Haemochromatosis Type 1 (AR) -------------------------
    {
        "gene": "HFE",
        "alt_name": "HFE (HFE-348aa-6p21.3 / AR — Hereditary-Haemochromatosis-Type-1 — C282Y-Homozygous-85pct — Phlebotomy-CURATIVE — Bronze-Diabetes-PATHOGNOMONIC)",
        "protein": (
            "HFE -- 6p21.3 AR -- HFE-348aa -- "
            "Homeostatic-Iron-Regulator-HFE-MHC-Class-I-Like-Hepcidin-BMP6 -- "
            "Hereditary-Haemochromatosis-Type-1-HH1 -- "
            "C282Y-p.Cys282Tyr-85pct-European-Founder -- "
            "H63D-Compound-Heterozygote-Modifier -- "
            "Phlebotomy-CURATIVE-If-Pre-Cirrhosis -- "
            "Bronze-Diabetes-Cardiomyopathy-Hypogonadism-Arthropathy"
        ),
        "locus": "6p21.3",
        "protein_size": "348 aa",
        "inheritance": (
            "AR (autosomal recessive) — HFE biallelic; "
            "C282Y homozygous: 85% of clinical HH in Northern Europeans; "
            "Penetrance C282Y/C282Y: ~30% males, ~1-5% females (menstrual iron loss protective); "
            "H63D/H63D: low penetrance — rarely causes clinical iron overload; "
            "C282Y/H63D compound heterozygote: mild increase risk, rarely clinically significant; "
            "Prevalence C282Y/C282Y: 1 in 200-400 Northern Europeans; "
            "Autosomal dominant modifiers: HAMP, HJV, TFR2, SLC40A1 (types 2-4) — HFE-negative HH"
        ),
        "age_of_onset": (
            "Biochemical: transferrin saturation >45% detectable from 20s in males; "
            "Clinical: symptoms usually 40-60 yr males; 50-70 yr females (post-menopausal); "
            "Fatigue + arthralgia (metacarpophalangeal joints 2nd-3rd): often first symptoms; "
            "Hepatomegaly + elevated LFTs: iron deposition in hepatocytes; "
            "Bronze skin pigmentation + diabetes: advanced iron overload; "
            "Hypogonadism: iron deposits in pituitary + gonads; "
            "Dilated cardiomyopathy + arrhythmia: cardiac iron (potentially reversible with phlebotomy); "
            "Arthropathy: MCP joints 2nd-3rd — chondrocalcinosis (calcium pyrophosphate deposition)"
        ),
        "key_biomarker": (
            "Transferrin saturation (TS): >45% fasting = screen positive; "
            "Serum ferritin: markedly elevated (>1000 μg/L diagnostic; >300 males, >200 females suggest HH); "
            "HFE genotyping: C282Y/C282Y confirms HH Type 1; "
            "Liver biopsy: hepatic iron concentration >36 μmol/g dry weight (Grade 3-4); "
            "Liver MRI (T2*/R2*): non-invasive iron quantification — MRI signal loss in liver; "
            "Hepatic iron index (HII): >2 mol/g/year (before liver biopsy era); "
            "Glucose/HbA1c: diabetes (bronze diabetes); "
            "LH/FSH/testosterone: hypogonadotrophic hypogonadism; "
            "ECHO: cardiomyopathy"
        ),
        "pathognomonic": (
            "BRONZE SKIN PIGMENTATION + DIABETES + CIRRHOSIS = bronze diabetes triad — HH advanced; "
            "MCP 2nd-3rd JOINT ARTHROPATHY + elevated ferritin = always test HFE; "
            "CHONDROCALCINOSIS on X-ray in young patient + high ferritin = HH; "
            "TRANSFERRIN SATURATION >45% fasting = mandatory HFE genotyping; "
            "IRON DEPOSITS PITUITARY = hypogonadotrophic hypogonadism + HH; "
            "DISTINGUISH from secondary iron overload (haemolysis, transfusions, NAFLD): TS is key"
        ),
        "treatment": (
            "Phlebotomy: 500 mL (250 mg iron) weekly until ferritin <50 μg/L, TS <30%; "
            "Maintenance: phlebotomy every 2-4 months lifelong to keep ferritin 50-100 μg/L; "
            "CURATIVE if pre-cirrhosis — liver disease can resolve; "
            "Post-cirrhosis: phlebotomy continues but cirrhosis irreversible — HCC surveillance; "
            "Iron chelation: only if phlebotomy intolerant (cardiac failure, anaemia); "
            "DIET: avoid excess vitamin C (enhances iron absorption); limit alcohol; avoid red meat excess; "
            "FAMILY SCREENING MANDATORY: first-degree relatives — TS + ferritin; "
            "HCC surveillance: 6-monthly AFP + ultrasound if cirrhosis present"
        ),
        "critical_flags": [
            "TRANSFERRIN-SATURATION->45pct-FASTING-MANDATORY-HFE-TEST",
            "PHLEBOTOMY-CURATIVE-PRE-CIRRHOSIS",
            "FIRST-DEGREE-FAMILY-SCREENING-MANDATORY",
            "HCC-SURVEILLANCE-CIRRHOSIS-6-MONTHLY",
            "MCP-2nd-3rd-ARTHROPATHY-CHONDROCALCINOSIS-HH-FLAG",
            "AVOID-VITAMIN-C-SUPPLEMENTS-IRON-ABSORPTION",
            "CARDIAC-IRON-CARDIOMYOPATHY-REVERSIBLE-PHLEBOTOMY",
        ],
    },
    # -- SERPINA1 — Alpha-1-Antitrypsin Deficiency (AR/codominant) -------------
    {
        "gene": "SERPINA1",
        "alt_name": "SERPINA1 (SERPINA1-418aa-14q32.13 / AR-Codominant — Alpha-1-Antitrypsin-Deficiency — PiZZ-Liver-Cirrhosis-Panacinar-Emphysema-Lower-Lobe — Augmentation-Therapy — Smoking-ABSOLUTELY-CI)",
        "protein": (
            "SERPINA1 -- 14q32.13 AR-Codominant -- SERPINA1-418aa -- "
            "Alpha-1-Antitrypsin-Serine-Protease-Inhibitor-Neutrophil-Elastase -- "
            "PiZZ-Homozygous-95pct-Clinical-AATD -- "
            "Hepatic-ER-Accumulation-PiZ-Polymer-Cirrhosis-HCC -- "
            "Pulmonary-Panacinar-Emphysema-Lower-Lobe-NE-Destruction -- "
            "Augmentation-Intravenous-AAT-Therapy-Pulmonary -- "
            "SMOKING-ABSOLUTELY-CONTRAINDICATED"
        ),
        "locus": "14q32.13",
        "protein_size": "418 aa",
        "inheritance": (
            "Codominant AR — SERPINA1 biallelic; "
            "PiZZ (Glu342Lys): most common severe genotype — serum AAT ~15% normal; "
            "PiSZ: intermediate — pulmonary risk; hepatic risk lower; "
            "PiSS: minimal clinical significance; "
            "Null variants: lowest AAT — NO hepatic disease (no misfolded protein in ER); "
            "Prevalence PiZZ: 1 in 2000-3000 Europeans; "
            "Carrier PiMZ: ~3% Northern Europeans — modestly increased lung disease risk; "
            "Hepatic disease mechanism: PiZ polymer misfolded protein accumulates in hepatocyte ER — NOT deficiency"
        ),
        "age_of_onset": (
            "Neonatal hepatitis: prolonged jaundice 1-4 months (10-15% PiZZ neonates); "
            "Childhood cirrhosis: 2-10% PiZZ develop significant liver disease by adulthood; "
            "Adult hepatic: cirrhosis / HCC (increased risk 20-30x); "
            "Adult pulmonary: emphysema 20-40 yr (SMOKERS) or 50-70 yr (non-smokers); "
            "Pancreatitis: rare association; "
            "WEGENER-LIKE VASCULITIS: ANCA-associated — rare; "
            "Panniculitis: neutrophilic; "
            "Pulmonary disease severity DRAMATICALLY WORSENED by smoking — 10-15 yr earlier onset"
        ),
        "key_biomarker": (
            "Serum AAT level: <11 μmol/L (<0.57 g/L) in PiZZ — quantitative; "
            "SERPINA1 phenotyping (isoelectric focusing): defines Pi type (ZZ, SZ, MZ, null); "
            "SERPINA1 genotyping: confirms alleles; "
            "PAS-positive diastase-resistant globules in hepatocytes: PiZ polymer on liver biopsy; "
            "Spirometry: obstructive pattern (FEV1/FVC <0.7); "
            "CT thorax: panacinar emphysema basal/lower lobe predominant; "
            "LFTs: elevated transaminases, cholestatic pattern (neonates); "
            "Liver MRI/biopsy: fibrosis grading; "
            "GGT: elevated in PiZZ hepatic disease; "
            "DISTINGUISH from upper lobe centracinar emphysema of smoking — AATD is LOWER lobe basal"
        ),
        "pathognomonic": (
            "LOWER LOBE PANACINAR EMPHYSEMA in patient under 45 = AATD until proven otherwise; "
            "PAS-POSITIVE DIASTASE-RESISTANT GLOBULES in hepatocytes = PiZ polymer = AATD liver; "
            "NEONATAL CHOLESTASIS + elevated GGT + PiZZ phenotype = AATD liver disease; "
            "EARLY EMPHYSEMA NON-SMOKER: test serum AAT — AATD forgotten diagnosis; "
            "SMOKING ABSOLUTELY CONTRAINDICATED — accelerates emphysema by 10-15 yr; "
            "DISTINGUISH: upper lobe emphysema = smoking COPD; lower lobe = AATD"
        ),
        "treatment": (
            "Pulmonary augmentation: IV alpha-1-antitrypsin (Prolastin/Aralast/Zemaira) 60 mg/kg weekly; "
            "Slows emphysema progression in PiZZ with FEV1 35-65%; "
            "SMOKING CESSATION: most impactful single intervention — ABSOLUTELY MANDATORY; "
            "Standard COPD: LABA/LAMA, ICS if frequent exacerbations; pulmonary rehab; "
            "Liver disease: standard hepatic management; transplantation cures liver + corrects AAT; "
            "Liver transplant: PiM phenotype post-transplant — NOT PiZZ — corrects deficiency; "
            "Gene therapy (investigational): liver-directed AAV — AATD liver + lung; "
            "HCC surveillance: if cirrhosis 6-monthly AFP + USS; "
            "FAMILY SCREENING: all first-degree relatives — serum AAT + phenotyping"
        ),
        "critical_flags": [
            "SMOKING-ABSOLUTELY-CONTRAINDICATED-10-15yr-EARLIER-EMPHYSEMA",
            "LOWER-LOBE-PANACINAR-EMPHYSEMA-YOUNG-ADULT-TEST-AATD",
            "PAS-POSITIVE-DIASTASE-RESISTANT-GLOBULES-PATHOGNOMONIC-LIVER",
            "AUGMENTATION-THERAPY-PULMONARY-ONLY-NOT-HEPATIC",
            "LIVER-TRANSPLANT-CORRECTS-BOTH-DEFICIENCY-AND-MISFOLDING",
            "NEONATAL-CHOLESTASIS-ELEVATED-GGT-TEST-SERPINA1",
            "FIRST-DEGREE-FAMILY-SCREENING-MANDATORY",
        ],
    },
    # -- JAG1 — Alagille Syndrome (AD) ------------------------------------------
    {
        "gene": "JAG1",
        "alt_name": "JAG1 (JAG1-1218aa-20p12.2 / AD — Alagille-Syndrome — Butterfly-Vertebrae-PATHOGNOMONIC — Paucity-Intrahepatic-Bile-Ducts — Posterior-Embryotoxon — Pulmonary-Artery-Stenosis)",
        "protein": (
            "JAG1 -- 20p12.2 AD -- JAG1-1218aa -- "
            "Jagged-1-Notch-Ligand-Extracellular-DSL-Domain-Notch-Signalling -- "
            "Alagille-Syndrome-ALGS1 -- "
            "Butterfly-Vertebrae-Paucity-Intrahepatic-Bile-Ducts-Posterior-Embryotoxon -- "
            "Pulmonary-Artery-Stenosis-Peripheral-PA-Stenosis-Cardiovascular -- "
            "Cholestatic-Liver-Disease-Pruritus-High-GGT -- "
            "Notch2-Gene-Alagille-Syndrome-2"
        ),
        "locus": "20p12.2",
        "protein_size": "1218 aa",
        "inheritance": (
            "AD (autosomal dominant) — JAG1 haploinsufficiency; "
            "~94% of Alagille cases; NOTCH2 causes ~2-3% (ALGS2); "
            "50% de novo; 50% inherited from parent (often mildly affected); "
            "Variable expressivity — wide phenotypic range within families; "
            "Deletion 20p12: detected by CMA — some larger deletions; "
            "Parental assessment MANDATORY (may be mildly affected — posterior embryotoxon alone); "
            "Penetrance near complete but expressivity highly variable"
        ),
        "age_of_onset": (
            "Neonatal: cholestasis — conjugated jaundice first weeks of life; "
            "Pruritus: severe + debilitating — often from 6-12 months; quality of life issue; "
            "Failure to thrive: fat malabsorption (fat-soluble vitamins ADEK); "
            "Cardiovascular: peripheral pulmonary artery stenosis — murmur at birth; TOF in 15%; "
            "Ocular: posterior embryotoxon (seen on slit-lamp 80-90%); "
            "Renal: renal tubular acidosis / renovascular hypertension (40%); "
            "Facial: triangular face, prominent forehead, deep-set eyes, pointed chin; "
            "Skeletal: butterfly vertebrae (80%) — incidental X-ray finding"
        ),
        "key_biomarker": (
            "Liver biopsy: paucity of interlobular bile ducts (bile duct-to-portal tract ratio <0.4); "
            "GGT: markedly elevated (high-GGT cholestasis — distinguishes PFIC1); "
            "Total bile acids: elevated; "
            "Fat-soluble vitamins: ADEK levels; "
            "Echocardiogram: pulmonary artery stenosis / TOF; "
            "Spine X-ray: butterfly vertebrae (sagittal cleft) — pathognomonic; "
            "Slit-lamp: posterior embryotoxon (peripheral corneal opacity Schwalbe line prominence); "
            "Renal ultrasound + DMSA: renal involvement; "
            "JAG1 sequencing/deletion analysis: confirms diagnosis; "
            "Cholangiography: paucity of intrahepatic bile ducts"
        ),
        "pathognomonic": (
            "BUTTERFLY VERTEBRAE (sagittal cleft vertebral bodies) on X-ray + cholestasis = Alagille until proven otherwise; "
            "PAUCITY OF INTRAHEPATIC BILE DUCTS (<0.4 duct:portal ratio) on liver biopsy = ALGS; "
            "POSTERIOR EMBRYOTOXON on slit-lamp + cholestasis = screen JAG1; "
            "PERIPHERAL PULMONARY ARTERY STENOSIS + cholestatic infant = ALGS; "
            "HIGH GGT cholestasis in neonate: ALGS or PFIC2 (BSEP) — distinguish from PFIC1 (NORMAL/LOW GGT); "
            "TRIANGULAR FACE + POSTERIOR EMBRYOTOXON: facial gestalt of Alagille"
        ),
        "treatment": (
            "Pruritus management: cholestyramine / rifampicin / naltrexone / sertraline; "
            "Maralixibat (IBAT inhibitor): FDA 2021 for Alagille — ileal bile acid transporter blockade; "
            "Odevixibat: alternative IBAT inhibitor; "
            "Fat-soluble vitamins: ADEK supplementation — MANDATORY; monitor levels; "
            "MCT formula: medium-chain triglycerides (bypass bile acid absorption); "
            "Cardiovascular: pulmonary artery stenosis intervention if severe; TOF repair; "
            "Liver transplantation: 20-30% require by adulthood (progressive liver disease); "
            "MONITORING: annual LFTs, fat-soluble vitamins, renal function, ECHO; "
            "OPHTHALMOLOGY: annual slit-lamp (posterior embryotoxon progression); "
            "RENAL: blood pressure monitoring (renovascular hypertension)"
        ),
        "critical_flags": [
            "BUTTERFLY-VERTEBRAE-PATHOGNOMONIC",
            "HIGH-GGT-CHOLESTASIS-DISTINGUISH-PFIC1-LOW-GGT",
            "FAT-SOLUBLE-VITAMINS-ADEK-MANDATORY",
            "MARALIXIBAT-FDA-2021-IBAT-INHIBITOR",
            "POSTERIOR-EMBRYOTOXON-SLIT-LAMP-PATHOGNOMONIC",
            "PULMONARY-ARTERY-STENOSIS-TOF-15pct",
            "PARENTAL-ASSESSMENT-MANDATORY-VARIABLE-EXPRESSIVITY",
        ],
    },
    # -- ABCB11 — PFIC2 / BSEP (AR) --------------------------------------------
    {
        "gene": "ABCB11",
        "alt_name": "ABCB11 (ABCB11-1321aa-2q31.1 / AR — PFIC2-BSEP-Deficiency — HIGH-GGT — Most-Severe-PFIC — Cholangiocarcinoma-HCC-Risk-Childhood — Vanishing-Bile-Duct-Syndrome)",
        "protein": (
            "ABCB11 -- 2q31.1 AR -- ABCB11-1321aa -- "
            "BSEP-Bile-Salt-Export-Pump-ABC-Transporter-Canalicular -- "
            "PFIC2-Progressive-Familial-Intrahepatic-Cholestasis-Type-2 -- "
            "High-GGT-DISTINGUISH-PFIC1-Low-GGT -- "
            "Conjugated-Hyperbilirubinemia-HIGH-Bile-Acids-Serum -- "
            "Cholangiocarcinoma-HCC-Childhood-RISK -- "
            "Liver-Transplant-CURATIVE -- "
            "BRIC2-Benign-Recurrent-Intrahepatic-Cholestasis-Type-2-Partial"
        ),
        "locus": "2q31.1",
        "protein_size": "1321 aa",
        "inheritance": (
            "AR (autosomal recessive) — ABCB11 biallelic loss-of-function; "
            "PFIC2: severe null mutations — progressive; "
            "BRIC2: partial function — benign recurrent cholestasis; "
            "Carrier parents: drug-induced cholestasis risk / ICP (intrahepatic cholestasis of pregnancy); "
            "Prevalence PFIC2: 1 in 100,000; "
            "PFIC2 most common form of PFIC (most severe); "
            "E297G: Europe founder; D482G: Japan founder; "
            "Genotype-phenotype: null/null = PFIC (severe); missense = BRIC2 or intermediate"
        ),
        "age_of_onset": (
            "Neonatal: jaundice from birth / early infancy; "
            "Severe cholestasis: pruritus very severe (scratching, excoriation, sleep disruption); "
            "Failure to thrive: fat malabsorption; "
            "Cirrhosis: rapidly progressive — LT often needed childhood; "
            "HCC and cholangiocarcinoma: childhood risk (much younger than other PFIC); "
            "BRIC2: intermittent attacks cholestasis weeks-months, normalises between; "
            "Triggers BRIC2: oral contraceptives, pregnancy (ICP), infections; "
            "PFIC2 bile acids serum: very high (>200 μmol/L); GGT HIGH (distinguish PFIC1)"
        ),
        "key_biomarker": (
            "GGT: HIGH (>3x ULN) — CRITICAL DDx from PFIC1 (LOW GGT) and PFIC3 (HIGH GGT); "
            "Serum bile acids: markedly elevated (100-300 μmol/L); "
            "Conjugated bilirubin: elevated; "
            "LFTs: elevated; "
            "Liver biopsy: cholestasis, giant cell transformation, paucity of bile ducts; "
            "Immunohistochemistry: BSEP staining absent/reduced on canalicular membrane; "
            "ABCB11 sequencing: identifies pathogenic variants; "
            "Abdominal ultrasound: gallstones (bile salt changes); hepatomegaly; "
            "MRCP: biliary anatomy; "
            "AFP: HCC surveillance (elevated early)"
        ),
        "pathognomonic": (
            "HIGH GGT cholestasis + very high serum bile acids + ABSENT BSEP on IHC = PFIC2; "
            "CHILDHOOD HCC OR CHOLANGIOCARCINOMA with progressive cholestasis = consider PFIC2; "
            "HIGH GGT separates PFIC2 from PFIC1 (LOW GGT) — critical diagnostic DDx; "
            "BSEP-NEGATIVE immunostaining on liver biopsy = PFIC2 hallmark; "
            "DRUG-INDUCED CHOLESTASIS in parent of PFIC child = suspect ABCB11 carrier; "
            "INTRAHEPATIC CHOLESTASIS OF PREGNANCY in mother of PFIC2 child = ABCB11 carrier"
        ),
        "treatment": (
            "UDCA (ursodeoxycholic acid): first-line — partial responders; "
            "Cholestyramine: pruritus management; "
            "Odevixibat / Maralixibat (IBAT inhibitors): emerging for PFIC2 pruritus; "
            "Fat-soluble vitamins ADEK: supplementation mandatory; "
            "Biliary diversion (partial external): reduces bile acid pool — buys time; "
            "Liver transplantation: definitive CURATIVE — cures PFIC2 completely (no extrahepatic disease); "
            "HCC surveillance: 6-monthly AFP + USS even in childhood; "
            "AVOID hepatotoxic drugs; "
            "GENETIC COUNSELLING: carrier parents — ICP drug-induced cholestasis risk"
        ),
        "critical_flags": [
            "HIGH-GGT-CHOLESTASIS-DISTINGUISH-PFIC1-LOW-GGT",
            "CHILDHOOD-HCC-CHOLANGIOCARCINOMA-RISK-SURVEILLANCE-MANDATORY",
            "BSEP-ABSENT-IHC-PATHOGNOMONIC",
            "LIVER-TRANSPLANT-CURATIVE-NO-EXTRAHEPATIC-DISEASE",
            "CARRIER-PARENT-ICP-DRUG-INDUCED-CHOLESTASIS-RISK",
            "FAT-SOLUBLE-VITAMINS-ADEK-MANDATORY",
            "MARALIXIBAT-ODEVIXIBAT-IBAT-PRURITUS",
        ],
    },
    # -- ATP8B1 — PFIC1 / Byler Disease (AR) -----------------------------------
    {
        "gene": "ATP8B1",
        "alt_name": "ATP8B1 (ATP8B1-1251aa-18q21.31 / AR — PFIC1-Byler-Disease-FIC1-Deficiency — LOW-GGT — Extrahepatic-Manifestations-Diarrhoea-Steatosis-Post-LT)",
        "protein": (
            "ATP8B1 -- 18q21.31 AR -- ATP8B1-1251aa -- "
            "FIC1-Aminophospholipid-Flippase-P4-ATPase-Phosphatidylserine-Asymmetry -- "
            "PFIC1-Progressive-Familial-Intrahepatic-Cholestasis-Type-1-Byler-Disease -- "
            "LOW-Normal-GGT-CRITICAL-DDx-PFIC2-High-GGT -- "
            "EXTRAHEPATIC-Diarrhoea-Pancreatic-Insufficiency-Hearing-Loss-Steatosis-Post-LT -- "
            "BRIC1-Benign-Recurrent-Intrahepatic-Cholestasis-Type-1-Partial -- "
            "Liver-Transplant-DOES-NOT-CURE-Extrahepatic-Manifestations"
        ),
        "locus": "18q21.31",
        "protein_size": "1251 aa",
        "inheritance": (
            "AR (autosomal recessive) — ATP8B1 biallelic; "
            "Byler disease: named after Amish kindred — founder I661T; "
            "Severe (PFIC1): null mutations — progressive liver disease; "
            "BRIC1: partial function — episodic benign cholestasis; "
            "Extrahepatic FIC1 expression: gut, pancreas, kidney, ear — explains extrahepatic features; "
            "Prevalence PFIC1: 1 in 100,000-500,000; "
            "Amish I661T: founder in Old Order Amish community; "
            "D554N: European founder"
        ),
        "age_of_onset": (
            "Neonatal: cholestasis from birth; "
            "LOW GGT cholestasis: distinguishes from PFIC2 (HIGH GGT) — critical DDx; "
            "Pruritus: severe — earlier and more severe than PFIC2; "
            "Diarrhoea: watery, fatty — extrahepatic manifestation (gut FIC1); "
            "Pancreatitis: exocrine pancreatic insufficiency; "
            "Hearing loss: sensorineural (FIC1 in cochlea); "
            "Growth retardation: severe; "
            "Post-LT steatohepatitis: fatty liver + worsening diarrhoea after LT (FIC1 still absent); "
            "Progressive cirrhosis: slower than PFIC2 but still requires LT"
        ),
        "key_biomarker": (
            "GGT: LOW or NORMAL (<1.5x ULN) — PATHOGNOMONIC of PFIC1 vs HIGH GGT in PFIC2; "
            "Serum bile acids: very elevated; "
            "ALT/AST: elevated; bilirubin elevated; "
            "Liver biopsy: bland cholestasis — 'Byler bile' (granular bile canaliculi on EM); "
            "Electron microscopy: coarse granular bile (Byler bile) — pathognomonic EM finding; "
            "FIC1 immunostaining: absent/reduced; "
            "ATP8B1 sequencing: pathogenic variants; "
            "Stool: fatty, malabsorption; "
            "Sweat chloride: elevated (FIC1 sweat gland) — CF DDx; "
            "Pancreatic function: steatorrhoea + fat-soluble vitamin deficiency"
        ),
        "pathognomonic": (
            "LOW GGT cholestasis + severe pruritus + diarrhoea in infant = PFIC1/FIC1; "
            "BYLER BILE on electron microscopy (coarse granular canalicular content) = PFIC1; "
            "POST-TRANSPLANT STEATOHEPATITIS + WORSENING DIARRHOEA = PFIC1 (extrahepatic FIC1 not corrected); "
            "LOW GGT separates PFIC1 from PFIC2 (HIGH GGT) — most critical single test; "
            "SWEAT CHLORIDE ELEVATED + cholestasis + diarrhoea = PFIC1 (FIC1 in sweat gland) — CF DDx; "
            "BILIARY DIVERSION: more effective in PFIC1 than PFIC2 (delays LT)"
        ),
        "treatment": (
            "UDCA: first-line; "
            "Cholestyramine: pruritus; "
            "Fat-soluble vitamins ADEK: MANDATORY; "
            "Biliary diversion (partial external or internal): MOST EFFECTIVE in PFIC1 — delays LT; "
            "IBAT inhibitors (odevixibat/maralixibat): evidence accumulating; "
            "Liver transplantation: corrects liver — does NOT cure extrahepatic disease; "
            "POST-LT: steatohepatitis, worsening diarrhoea — anticipate and manage; "
            "Pancreatic enzyme replacement: if exocrine insufficiency; "
            "Hearing: audiological follow-up; "
            "NUTRITION: intensive support — MCT formula, ADEK"
        ),
        "critical_flags": [
            "LOW-GGT-CHOLESTASIS-PATHOGNOMONIC-PFIC1-DISTINGUISH-PFIC2",
            "BYLER-BILE-EM-GRANULAR-PATHOGNOMONIC",
            "POST-LT-STEATOHEPATITIS-WORSENING-DIARRHOEA-WARN-FAMILY",
            "LT-DOES-NOT-CURE-EXTRAHEPATIC-MANIFESTATIONS",
            "BILIARY-DIVERSION-MOST-EFFECTIVE-PFIC1",
            "FAT-SOLUBLE-VITAMINS-ADEK-MANDATORY",
            "DIARRHOEA-EXTRAHEPATIC-FIC1-GUT-EXPRESSION",
        ],
    },
    # -- SLC25A13 — Citrinemia / NICCD (AR) ------------------------------------
    {
        "gene": "SLC25A13",
        "alt_name": "SLC25A13 (SLC25A13-675aa-7q21.3 / AR — Citrinemia-NICCD-Citrullinemia-Type-II — Neonatal-Intrahepatic-Cholestasis-HIGH-AFP — Protein-and-Fat-Diet-Preference-PATHOGNOMONIC — East-Asian-Founder)",
        "protein": (
            "SLC25A13 -- 7q21.3 AR -- SLC25A13-675aa -- "
            "Citrin-Mitochondrial-Aspartate-Glutamate-Carrier-2-AGC2 -- "
            "Citrinemia-Citrullinemia-Type-II -- "
            "NICCD-Neonatal-Intrahepatic-Cholestasis-Citrin-Deficiency -- "
            "FTTDCD-Failure-to-Thrive-Dyslipidaemia-Citrin-Deficiency-Childhood -- "
            "CTLN2-Adult-Citrullinemia-Type-II-Hyperammonemia-Encephalopathy -- "
            "East-Asian-Founder-IVS16ins3kb-85pct-Japanese -- "
            "Protein-Fat-Diet-Preference-PATHOGNOMONIC-Avoidance-Carbohydrates"
        ),
        "locus": "7q21.3",
        "protein_size": "675 aa",
        "inheritance": (
            "AR (autosomal recessive) — SLC25A13 biallelic; "
            "IVS16ins3kb: 85% Japanese NICCD alleles; "
            "851del4: 12% Japanese; "
            "East Asian predominance: Japan, China, Korea, Taiwan; "
            "Rare in Europeans; "
            "Three age-dependent phenotypes: NICCD (neonatal), FTTDCD (childhood), CTLN2 (adult); "
            "Most NICCD resolves spontaneously by 12 months; "
            "CTLN2 may develop later (even decades after NICCD)"
        ),
        "age_of_onset": (
            "NICCD (Neonatal): 1-6 months — cholestatic jaundice, elevated AFP, galactosaemia-like; "
            "Spontaneous resolution NICCD: 12 months in most; "
            "FTTDCD (Childhood): 2-11 yr — failure to thrive, dyslipidaemia, fatty liver; "
            "CTLN2 (Adult): 20-50 yr — sudden onset hyperammonemia encephalopathy; "
            "Protein and fat preference: PATHOGNOMONIC — avoidance of carbohydrates/sweets all ages; "
            "CTLN2 triggers: surgery, alcohol, carbohydrate loading; "
            "CTLN2 encephalopathy: nocturnal delirium, aggression, psychiatric — EMERGENCY; "
            "Pancreatitis: CTLN2 risk"
        ),
        "key_biomarker": (
            "Plasma amino acids: citrulline elevated (>100 μmol/L in CTLN2); threonine elevated; "
            "Plasma ammonia: elevated in CTLN2 (>100 μmol/L) — hyperammonemia; "
            "Urine organic acids: not typical; "
            "Serum AFP: markedly elevated in NICCD (>100 ng/mL); "
            "Galactose: NICCD galactosaemia-like metabolites; "
            "LFTs: elevated in NICCD; "
            "NBS (newborn screening): citrulline elevated; "
            "Lipid profile: dyslipidaemia FTTDCD + CTLN2 (hypertriglyceridaemia); "
            "SLC25A13 sequencing: confirms diagnosis; "
            "Liver biopsy: steatosis, cholestasis (NICCD); fatty liver (FTTDCD)"
        ),
        "pathognomonic": (
            "PROTEIN AND FAT DIETARY PREFERENCE + CARBOHYDRATE AVOIDANCE at any age = Citrinemia; "
            "NOCTURNAL DELIRIUM + HYPERAMMONEMIA + CITRULLINE ELEVATED = CTLN2 — EMERGENCY; "
            "NEONATAL CHOLESTASIS + VERY HIGH AFP + galactosaemia-like + East Asian = NICCD; "
            "CTLN2 TRIGGERED BY SURGERY / ALCOHOL / CARBOHYDRATE LOADING; "
            "HIGH CITRULLINE + HIGH AMMONIA + protein preference: CTLN2 DDx Urea Cycle; "
            "DISTINGUISH from OTC deficiency: citrulline HIGH in CTLN2; LOW in OTC"
        ),
        "treatment": (
            "NICCD: lactose-free formula + MCT; galactose restriction if galactosaemia-like; "
            "Natural resolution NICCD: 12 months; "
            "CTLN2 ACUTE: arginine infusion + low-carbohydrate diet; avoid carbohydrate loading; "
            "CTLN2 AVOID: carbohydrates, alcohol, sugar solutions IV; "
            "Sodium pyruvate: investigational CTLN2; "
            "MCT supplementation: helps in FTTDCD + CTLN2; "
            "Liver transplantation: definitive for CTLN2 — curative; "
            "DIET: high protein + fat, low carbohydrate ALWAYS; "
            "MONITORING: plasma amino acids 6-monthly; ammonia; "
            "AVOID SURGERY WITHOUT PERIOPERATIVE CARBOHYDRATE RESTRICTION"
        ),
        "critical_flags": [
            "PROTEIN-FAT-PREFERENCE-CARBOHYDRATE-AVOIDANCE-PATHOGNOMONIC",
            "CTLN2-NOCTURNAL-DELIRIUM-HYPERAMMONEMIA-EMERGENCY",
            "AVOID-CARBOHYDRATE-LOADING-GLUCOSE-IV-TRIGGERS-CTLN2",
            "VERY-HIGH-AFP-NICCD-NEONATAL",
            "LIVER-TRANSPLANT-CURATIVE-CTLN2",
            "EAST-ASIAN-FOUNDER-IVS16ins3kb",
            "CTLN2-TRIGGERED-SURGERY-ALCOHOL",
        ],
    },
    # -- NPC1 — Niemann-Pick Disease Type C1 (AR) -------------------------------
    {
        "gene": "NPC1",
        "alt_name": "NPC1 (NPC1-1278aa-18q11.2 / AR — Niemann-Pick-Disease-Type-C — Vertical-Supranuclear-Gaze-Palsy-PATHOGNOMONIC — Miglustat — Hepatosplenomegaly — Intracellular-Cholesterol-Trafficking)",
        "protein": (
            "NPC1 -- 18q11.2 AR -- NPC1-1278aa -- "
            "NPC1-Lysosomal-Membrane-Protein-13-TM-Helices-Sterol-Sensing-Domain -- "
            "Niemann-Pick-Disease-Type-C1-NPC1-95pct-All-NPC -- "
            "Intracellular-Cholesterol-Unesterified-Lysosomal-Accumulation -- "
            "Vertical-Supranuclear-Gaze-Palsy-VSGP-PATHOGNOMONIC-All-Ages -- "
            "Hepatosplenomegaly-Neonatal-Cholestasis-Infants -- "
            "Miglustat-Substrate-Reduction-Therapy -- "
            "I1061T-Most-Common-Western-15-20pct"
        ),
        "locus": "18q11.2",
        "protein_size": "1278 aa",
        "inheritance": (
            "AR (autosomal recessive) — NPC1 biallelic; "
            "NPC1: 95% of NPC; NPC2: 5% (soluble lysosomal protein); "
            "I1061T: most common variant Western patients (15-20%); "
            "P1007A, G992W: other common variants; "
            "Prevalence: 1 in 100,000-150,000; "
            "Wide allelic heterogeneity >300 variants; "
            "Genotype-phenotype: null = severe neonatal; partial function = later onset; "
            "Infantile neurological form: more severe than juvenile/adult onset"
        ),
        "age_of_onset": (
            "Fetal hydrops / neonatal: non-immune hydrops + hepatosplenomegaly; "
            "Neonatal cholestasis: conjugated jaundice resolving by 6 months; "
            "Early infantile: hypotonia, delayed milestones (before 2 yr) — most severe; "
            "Late infantile (3-5 yr): ataxia, dysarthria, gelastic cataplexy; "
            "Juvenile (6-15 yr): VSGP + ataxia + dysarthria — most common presentation; "
            "Adult (>15 yr): psychiatric (psychosis, bipolar-like), cognitive decline + VSGP; "
            "Gelastic cataplexy: sudden laughter-triggered loss of muscle tone — PATHOGNOMONIC; "
            "Progressive neurological course: dysphagia → aspiration → death"
        ),
        "key_biomarker": (
            "Vertical supranuclear gaze palsy (VSGP): saccades impaired vertically (especially downward); "
            "Oxysterols (plasma): 25-hydroxycholesterol + 7-ketocholesterol elevated — NPC1/2 screening biomarker; "
            "Filipin staining: unesterified cholesterol accumulation in fibroblast lysosomes — GOLD STANDARD; "
            "NPC1/NPC2 sequencing: pathogenic variants; "
            "Liver biopsy: foam cells (lipid-laden macrophages) in neonatal; "
            "Bone marrow: foam cells (sea-blue histiocytes); "
            "Brain MRI: cerebellar atrophy (progressive); thalamic atrophy; "
            "EEG: non-specific; "
            "CSF: NPC-related biomarkers (calbindin, chitotriosidase) — research; "
            "MRS: NAA reduction cerebellum"
        ),
        "pathognomonic": (
            "VERTICAL SUPRANUCLEAR GAZE PALSY (VSGP) + ataxia + hepatosplenomegaly = NPC until proven otherwise; "
            "GELASTIC CATAPLEXY (laughter-triggered muscle tone loss) = NPC pathognomonic feature; "
            "FILIPIN TEST (unesterified cholesterol accumulation in fibroblasts) = GOLD STANDARD; "
            "NEONATAL CHOLESTASIS RESOLVING + LATER NEUROLOGICAL DECLINE = think NPC; "
            "ADULT PSYCHIATRIC PRESENTATION + VERTICAL GAZE PALSY: NPC must be excluded; "
            "ELEVATED PLASMA OXYSTEROLS: fast screening test — 7-ketocholesterol + 25-OH cholesterol"
        ),
        "treatment": (
            "Miglustat (Zavesca): substrate reduction therapy — slows neurological progression; "
            "NOT curative — stabilises neurological; started as early as possible; "
            "Arimoclomol (investigational): chaperone to rescue misfolded NPC1 protein; "
            "Cyclodextrin (HPBCD): investigational — direct cholesterol mobiliser; "
            "Symptomatic: anti-epileptic (levetiracetam preferred); "
            "Dysphagia: gastrostomy — aspiration prevention; "
            "Cataplexy: sodium oxybate / clomipramine; "
            "MONITORING: VSGP severity, swallow assessment, BMI, hepatic; "
            "Liver transplantation: corrects hepatic disease ONLY — does NOT treat neurological; "
            "FAMILY SCREENING: siblings at risk — oxysterols + NPC1 sequencing"
        ),
        "critical_flags": [
            "VERTICAL-SUPRANUCLEAR-GAZE-PALSY-PATHOGNOMONIC",
            "GELASTIC-CATAPLEXY-LAUGHTER-TRIGGERED-PATHOGNOMONIC",
            "FILIPIN-TEST-GOLD-STANDARD-UNESTERIFIED-CHOLESTEROL",
            "MIGLUSTAT-SLOWS-NOT-CURATIVE",
            "LIVER-TRANSPLANT-DOES-NOT-TREAT-NEUROLOGICAL",
            "ADULT-PSYCHIATRIC-ALWAYS-EXCLUDE-NPC",
            "OXYSTEROLS-PLASMA-FAST-SCREENING",
            "NEONATAL-CHOLESTASIS-RESOLVING-LATER-NEURO-THINK-NPC",
        ],
    },
]


def _generate_cohort():
    """Generate 8 × 40 = 320 patient records, one seed per gene."""
    all_patients = []
    for idx, gene_data in enumerate(HD_GENES):
        rng = random.Random(SEED_BASE + idx)
        gene = gene_data["gene"]
        for i in range(40):
            age = rng.randint(0, 65)
            if gene == "ATP7B":
                hepatic = rng.random() < 0.55
                neuro = rng.random() < 0.45
                psychiatric = rng.random() < 0.35
                kf_rings = neuro or psychiatric
                fulminant = (not neuro) and (not psychiatric) and rng.random() < 0.08
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "hepatic_presentation": hepatic,
                    "neurological_presentation": neuro,
                    "psychiatric_presentation": psychiatric,
                    "kf_rings": kf_rings,
                    "fulminant": fulminant,
                    "liver_disease": True,
                })
            elif gene == "HFE":
                c282y_hom = rng.random() < 0.85
                cirrhosis = rng.random() < 0.25
                diabetes = rng.random() < 0.20
                cardiomyopathy = rng.random() < 0.10
                arthropathy = rng.random() < 0.40
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": max(30, age),
                    "c282y_homozygous": c282y_hom,
                    "cirrhosis": cirrhosis,
                    "diabetes": diabetes,
                    "cardiomyopathy": cardiomyopathy,
                    "arthropathy": arthropathy,
                    "liver_disease": cirrhosis,
                })
            elif gene == "SERPINA1":
                pizz = rng.random() < 0.90
                pulmonary = rng.random() < 0.70
                hepatic = rng.random() < 0.40
                smoker = rng.random() < 0.30
                cirrhosis = hepatic and rng.random() < 0.35
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "pizz_genotype": pizz,
                    "pulmonary_disease": pulmonary,
                    "hepatic_disease": hepatic,
                    "smoker": smoker,
                    "cirrhosis": cirrhosis,
                    "liver_disease": hepatic,
                })
            elif gene == "JAG1":
                cholestasis = rng.random() < 0.95
                pa_stenosis = rng.random() < 0.85
                butterfly_vertebrae = rng.random() < 0.80
                posterior_embryotoxon = rng.random() < 0.85
                liver_transplant = rng.random() < 0.25
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": max(0, min(age, 10)),
                    "cholestasis": cholestasis,
                    "pulmonary_artery_stenosis": pa_stenosis,
                    "butterfly_vertebrae": butterfly_vertebrae,
                    "posterior_embryotoxon": posterior_embryotoxon,
                    "liver_transplant": liver_transplant,
                    "liver_disease": cholestasis,
                })
            elif gene == "ABCB11":
                pfic2 = rng.random() < 0.70
                high_ggt = True  # ALWAYS high in PFIC2
                hcc = rng.random() < 0.08
                liver_transplant = pfic2 and rng.random() < 0.55
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": max(0, min(age, 5)),
                    "pfic2": pfic2,
                    "high_ggt": high_ggt,
                    "hcc": hcc,
                    "liver_transplant": liver_transplant,
                    "liver_disease": True,
                })
            elif gene == "ATP8B1":
                pfic1 = rng.random() < 0.65
                low_ggt = True  # ALWAYS low in PFIC1
                diarrhoea = rng.random() < 0.80
                post_lt_steatosis = pfic1 and rng.random() < 0.60
                liver_transplant = pfic1 and rng.random() < 0.50
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": max(0, min(age, 3)),
                    "pfic1": pfic1,
                    "low_ggt": low_ggt,
                    "diarrhoea": diarrhoea,
                    "post_lt_steatosis": post_lt_steatosis,
                    "liver_transplant": liver_transplant,
                    "liver_disease": True,
                })
            elif gene == "SLC25A13":
                niccd = rng.random() < 0.60
                ctln2 = rng.random() < 0.25
                high_afp = niccd and rng.random() < 0.90
                carbo_avoidance = True  # pathognomonic
                hyperammonemia = ctln2
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "niccd": niccd,
                    "ctln2": ctln2,
                    "high_afp": high_afp,
                    "carbohydrate_avoidance": carbo_avoidance,
                    "hyperammonemia": hyperammonemia,
                    "liver_disease": niccd or ctln2,
                })
            elif gene == "NPC1":
                vsgp = rng.random() < 0.90
                hepatosplenomegaly = rng.random() < 0.85
                gelastic_cataplexy = rng.random() < 0.55
                neonatal_cholestasis = rng.random() < 0.35
                psychiatric = rng.random() < 0.30
                all_patients.append({
                    "gene": gene, "patient_id": f"{gene}-{i+1:03d}",
                    "age_at_presentation": age,
                    "vsgp": vsgp,
                    "hepatosplenomegaly": hepatosplenomegaly,
                    "gelastic_cataplexy": gelastic_cataplexy,
                    "neonatal_cholestasis": neonatal_cholestasis,
                    "psychiatric_presentation": psychiatric,
                    "liver_disease": hepatosplenomegaly,
                })
    return all_patients


def overview():
    """Return atlas-level aggregate overview."""
    patients = _generate_cohort()
    liver_patients = sum(1 for p in patients if p.get("liver_disease"))
    cirrhosis_patients = sum(1 for p in patients if p.get("cirrhosis"))
    transplant_patients = sum(1 for p in patients if p.get("liver_transplant"))
    neonatal_patients = sum(
        1 for p in patients
        if p.get("age_at_presentation", 99) < 1 or p.get("neonatal_cholestasis") or p.get("niccd")
    )
    neurological_patients = sum(
        1 for p in patients
        if p.get("neurological_presentation") or p.get("vsgp") or p.get("psychiatric_presentation")
    )
    pulmonary_patients = sum(1 for p in patients if p.get("pulmonary_disease"))
    hcc_patients = sum(1 for p in patients if p.get("hcc"))
    return {
        "atlas": "Hereditary-Hepatic-Disease-Atlas",
        "genes": [g["gene"] for g in HD_GENES],
        "total_patients": len(patients),
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "liver_disease_patients": liver_patients,
        "cirrhosis_patients": cirrhosis_patients,
        "transplant_patients": transplant_patients,
        "neonatal_cholestasis_patients": neonatal_patients,
        "neurological_patients": neurological_patients,
        "pulmonary_patients": pulmonary_patients,
        "hcc_patients": hcc_patients,
    }


def breakdown():
    """Return per-gene breakdown with clinical details."""
    patients = _generate_cohort()
    result = {}
    for gene_data in HD_GENES:
        gene = gene_data["gene"]
        gene_patients = [p for p in patients if p["gene"] == gene]
        result[gene] = {
            "gene": gene,
            "alt_name": gene_data["alt_name"],
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"],
            "patient_count": len(gene_patients),
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "critical_flags": gene_data["critical_flags"],
            "age_of_onset": gene_data["age_of_onset"],
            "key_biomarker": gene_data["key_biomarker"],
        }
    return result


def definitions():
    """Return gene definitions, glossary and surveillance protocols."""
    return {
        "genes": {g["gene"]: g["protein"] for g in HD_GENES},
        "glossary": {
            "Wilson Disease": "Autosomal recessive copper metabolism disorder — ATP7B; hepatic + neurological + psychiatric; Kayser-Fleischer rings pathognomonic; treated with chelation (penicillamine/trientine) or zinc",
            "Hereditary Haemochromatosis": "Iron overload disorder — HFE C282Y; hepatic cirrhosis, bronze diabetes, cardiomyopathy, arthropathy; phlebotomy curative if pre-cirrhotic",
            "Alpha-1-Antitrypsin Deficiency": "SERPINA1 PiZZ — ER protein misfolding → hepatic cirrhosis; NE deficiency → lower lobe panacinar emphysema; smoking absolutely contraindicated",
            "Alagille Syndrome": "JAG1/NOTCH2 haploinsufficiency; paucity intrahepatic bile ducts; butterfly vertebrae pathognomonic; pulmonary artery stenosis; posterior embryotoxon",
            "PFIC2 (BSEP Deficiency)": "ABCB11 biallelic — bile salt export pump absent; HIGH GGT cholestasis; most severe PFIC; childhood HCC/cholangiocarcinoma risk; LT curative",
            "PFIC1 (FIC1 Deficiency)": "ATP8B1 biallelic — FIC1 phospholipid flippase absent; LOW GGT cholestasis; extrahepatic: diarrhoea, pancreatitis, hearing loss; post-LT steatohepatitis",
            "Citrinemia / NICCD": "SLC25A13 biallelic — citrin mitochondrial carrier deficiency; NICCD (neonatal) → CTLN2 (adult hyperammonemia); protein/fat dietary preference pathognomonic",
            "Niemann-Pick Disease Type C": "NPC1/NPC2 biallelic — intracellular cholesterol trafficking defect; vertical supranuclear gaze palsy pathognomonic; miglustat substrate reduction therapy",
            "Kayser-Fleischer Rings": "Copper deposits in Descemet membrane peripheral cornea; requires slit-lamp examination; present in most neurological Wilson disease; may be absent in hepatic-only WD",
            "Transferrin Saturation": "Fasting TS >45% = screen positive for haemochromatosis; >70% = very high iron overload; key biochemical screening test for HFE",
            "PFIC": "Progressive Familial Intrahepatic Cholestasis — group of AR disorders impairing bile formation; PFIC1 (FIC1/ATP8B1), PFIC2 (BSEP/ABCB11), PFIC3 (MDR3/ABCB4), PFIC4-6; distinguished by GGT level",
            "IBAT Inhibitor": "Ileal bile acid transporter inhibitor — reduces enterohepatic bile acid recirculation; maralixibat (FDA 2021 Alagille), odevixibat; reduces pruritus and serum bile acids in PFIC",
            "Vertical Supranuclear Gaze Palsy": "Impaired voluntary vertical eye movements (especially downward saccades) with preserved reflex (doll's eye); pathognomonic of NPC; due to riMLF/INC lesion from cholesterol accumulation",
            "Byler Bile": "Coarse granular canalicular content seen on electron microscopy in PFIC1; pathognomonic ultrastructural finding; distinguishes PFIC1 from other causes of LOW GGT cholestasis",
            "Filipin Test": "Fluorescent staining of unesterified cholesterol in cultured fibroblasts; gold standard for NPC diagnosis; cholesterol accumulates in lysosomes; used alongside oxysterols and sequencing",
        },
        "surveillance_protocols": {
            "ATP7B": "24h urine copper + serum ceruloplasmin + LFTs 6-monthly; annual neurological assessment (neurological WD); copper-restricted diet; avoid hepatotoxins; chelation compliance monitoring; brain MRI if neurological change; sibling screening mandatory",
            "HFE": "Phlebotomy to ferritin <50 μg/L; maintenance 2-4x/year; LFTs + ferritin 6-monthly; HCC surveillance (6-monthly AFP + USS if cirrhosis); ECHO if cardiomyopathy; bone density (hypogonadism); family screening — first-degree relatives TS + ferritin",
            "SERPINA1": "Spirometry 1-2 yearly; CT thorax at baseline; serum AAT 6-monthly (augmentation monitoring); LFTs + USS liver annually; HCC surveillance if cirrhosis; smoking cessation absolute; family screening; fat-soluble vitamins",
            "JAG1": "LFTs + bile acids + fat-soluble vitamins ADEK 3-monthly (cholestatic phase); annual ECHO (PA stenosis); ophthalmology + slit-lamp annually; spine X-ray; renal function + BP; pruritus scoring; maralixibat compliance",
            "ABCB11": "LFTs + GGT + serum bile acids 3-monthly; AFP + USS 6-monthly (HCC from childhood); MRCP biliary; fat-soluble vitamins ADEK; LT listing assessment; IBAT inhibitor therapy monitoring; post-LT LFTs",
            "ATP8B1": "LFTs + GGT + serum bile acids 3-monthly; stool fat (steatorrhoea); fat-soluble vitamins ADEK; growth monitoring; audiological assessment; biliary diversion assessment; post-LT: LFTs + weight + diarrhoea severity",
            "SLC25A13": "Plasma amino acids (citrulline) 6-monthly; plasma ammonia; lipid profile; liver USS; diet diary (carbohydrate restriction); CTLN2 trigger avoidance — no surgery without protocol; LT assessment for CTLN2",
            "NPC1": "VSGP assessment 6-monthly; swallow assessment annually; brain MRI yearly; oxysterols monitoring; miglustat compliance; cataplexy diary; dysphagia management; ECHO; hepatosplenomegaly USS; family oxysterols screening",
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Liver disease patients: {ov['liver_disease_patients']}")
    print(f"Neonatal cholestasis patients: {ov['neonatal_cholestasis_patients']}")
    print(f"Neurological patients: {ov['neurological_patients']}")
    print(f"HCC patients: {ov['hcc_patients']}")
