#!/usr/bin/env python3
"""Hepatic-Atlas — Complete 8-Gene Hereditary Liver Disease Atlas
ATP7B   (Wilson's disease; ~1465 aa; 13q14.3; copper-transporting P1B-ATPase; AR; 1:30,000;
         Kayser-Fleischer rings PATHOGNOMONIC; liver + neuropsychiatric + haemolytic anaemia;
         penicillamine/trientine/zinc; tetrathiomolybdate; liver transplant curative) ·
HFE     (Hereditary haemochromatosis type 1; ~348 aa; 6p21.3; HFE protein; AR; C282Y/H63D founder;
         iron overload — liver/heart/endocrine; phlebotomy first-line; 1:200–1:400 Northern European) ·
SERPINA1 (Alpha-1-antitrypsin deficiency; ~418 aa; 14q32.1; serine protease inhibitor; codominant;
          PiZZ genotype; liver cirrhosis + emphysema; Z-polymer hepatocyte entrapment;
          augmentation therapy/resmetirom; lung transplant; liver transplant curative) ·
ABCB11  (PFIC2 / BSEP deficiency; ~1321 aa; 2q31.1; bile salt export pump; AR;
         progressive familial intrahepatic cholestasis type 2; odevixibat FDA 2021; maralixibat;
         high HCC risk even in children — surveillance mandatory) ·
ATP8B1  (PFIC1 / FIC1 deficiency / Byler disease; ~1251 aa; 18q21.31; aminophospholipid flippase;
         AR; low-GGT cholestasis PATHOGNOMONIC; extrahepatic features — diarrhoea, deafness;
         liver transplant does NOT cure extrahepatic features) ·
ABCB4   (PFIC3 / MDR3 deficiency; ~1279 aa; 7q21.12; phosphatidylcholine flippase (MDR3);
         AR; HIGH-GGT cholestasis PATHOGNOMONIC distinguishes from PFIC1/2; cholelithiasis;
         ursodeoxycholic acid (UDCA) effective in partial deficiency) ·
JAG1    (Alagille syndrome type 1 / ALGS1; ~1218 aa; 20p12.2; Jagged-1 Notch ligand; AD;
         bile duct paucity + butterfly vertebrae + peripheral pulmonary stenosis + posterior embryotoxon;
         50% de novo; xanthomata; ritonavir-based Notch therapy emerging) ·
SLC25A13 (Citrin deficiency; ~675 aa; 7q21.3; mitochondrial aspartate-glutamate carrier 2 (citrin);
           AR; East Asian founder (851del4 Japan/Korea/China); neonatal intrahepatic cholestasis (NICCD)
           → adult-onset citrullinaemia type II (CTLN2); HIGH-CARBOHYDRATE DIET ABSOLUTELY CI — triggers
           hyperammonaemia; protein/fat diet effective; arginine supplementation)
320-patient aggregate cohort (8 × 40, seeds 1150–1157)
"""

import random

SEED_BASE = 1150

HEPATIC_GENES = [
    # ── ATP7B — Wilson's Disease ──────────────────────────────────────────
    {
        "gene": "ATP7B",
        "protein": "ATP7B (Wilson's disease protein / Cu-P1B-ATPase)",
        "alias": (
            "ATP7B; OMIM gene 606882; 13q14.3; ~1465 aa; Wilson's disease (OMIM #277900); "
            "AR; prevalence 1:30,000–1:10,000; copper-transporting P1B-ATPase expressed in hepatocytes; "
            "Kayser-Fleischer (KF) rings PATHOGNOMONIC (95% neuropsychiatric, 50% hepatic alone); "
            "serum ceruloplasmin low <0.2 g/L; 24h urine copper >100 µg/day diagnostic; "
            "penicillamine/trientine/zinc/tetrathiomolybdate; liver transplant curative"
        ),
        "aa": "~1465 aa",
        "kDa": "~165 kDa",
        "gene_class": "P1B-type ATPase (heavy metal transporter); copper (Cu²⁺→Cu⁺) pump; hepatocyte trans-Golgi network; biliary copper export; ceruloplasmin cuproprotein loading",
        "locus": "13q14.3",
        "omim_gene": 606882,
        "omim_disease": 277900,
        "phenotype": "Wilson's disease — copper accumulation in liver, brain, cornea, kidneys; liver disease (hepatitis/cirrhosis), neuropsychiatric (tremor/dysarthria/dystonia), KF rings; haemolytic anaemia in acute liver failure",
        "disease": (
            "ATP7B encodes a hepatocyte trans-Golgi network P1B-ATPase that pumps copper into "
            "bile for excretion and loads copper onto apoceruloplasmin to form holo-ceruloplasmin. "
            "NORMAL FUNCTION: In the liver, ATP7B maintains intracellular copper homeostasis — "
            "at low copper, ATP7B is retained in trans-Golgi loading apoceruloplasmin; at high copper, "
            "ATP7B traffics to canalicular membrane pumping copper into bile. "
            "PATHOMECHANISM: LOF variants → failure of biliary copper excretion → copper "
            "accumulates in hepatocytes → hepatocyte toxicity → spills into circulation → "
            "deposits in brain (basal ganglia, brainstem), cornea (Descemet membrane → "
            "KF rings), kidneys (Fanconi tubulopathy), and other organs. "
            "Defective apoceruloplasmin loading → low serum ceruloplasmin (an acute-phase reactant — "
            "can be normal in inflammation; low baseline is supportive not diagnostic). "
            "CLINICAL PRESENTATIONS: (1) HEPATIC (50% of presentations): ranging from elevated LFTs, "
            "acute hepatitis, fulminant acute liver failure (ALF — with Coombs-negative haemolytic "
            "anaemia PATHOGNOMONIC: massive hepatocyte necrosis releases copper → red cell lysis), "
            "to cirrhosis. (2) NEUROPSYCHIATRIC (40%): KF rings almost universal (95%) in neuro WD; "
            "tremor (wing-beat/Holmes), dysarthria, dystonia, Parkinsonism, psychiatric changes "
            "(personality change, psychosis, depression) — any young person with unexplained "
            "neuropsychiatric symptoms must have KF rings checked by slit-lamp. "
            "(3) MIXED HEPATIC + NEURO (10%). "
            "DIAGNOSIS: 24h urine copper >100 µg/day (>250 µg/day on D-penicillamine challenge); "
            "serum ceruloplasmin <0.2 g/L; KF rings on slit-lamp; liver copper >250 µg/g dry weight; "
            "ATP7B sequencing (>700 variants known; H1069Q most common Central/East European ~35%; "
            "R778L most common East Asian ~30%). Leipzig score ≥4 = diagnosis. "
            "TREATMENT: Penicillamine (chelating agent — mobilises copper into urine; "
            "initial worsening of neuro symptoms 20-50% → switch to trientine first-line neuro); "
            "Trientine (triethylenetetramine — chelator; better tolerated neuro; FDA 2022 trientine tetrahydrochloride = Cuvrior); "
            "Zinc (blocks intestinal copper absorption by inducing metallothionein in enterocytes; "
            "first-line in presymptomatic/pregnant; SLOWER onset — not for acute presentations); "
            "Tetrathiomolybdate (ammonium/bis-choline salt — binds copper in intestine and blood; "
            "promising for neuro WD; ALXN2103/WTX101 trials); "
            "LIVER TRANSPLANT: curative (replaces the defective hepatic ATP7B) — indicated for "
            "ALF, decompensated cirrhosis, or failed medical therapy. "
            "MONITORING: 24h urine copper, LFTs, neuropsychiatric assessment every 6-12 months."
        ),
        "inheritance": "AR (autosomal recessive); compound heterozygous most common in European populations; consanguineous families → homozygous; 30-50% present between age 5-35 years; rarely diagnosed >50y",
        "hallmark": "Kayser-Fleischer rings (KF — golden-brown corneal deposits at Descemet membrane, seen on slit-lamp) PATHOGNOMONIC in neuropsychiatric WD; low ceruloplasmin; 24h urine copper >100 µg/day; Coombs-negative haemolytic anaemia in ALF; liver copper >250 µg/g dry weight",
        "key_ddx": "Autoimmune hepatitis (similar biochemistry; ANA/ASMA positive; KF absent — biopsy distinguishes), NAFLD/NASH (metabolic risk factors; KF absent), Neonatal haemochromatosis (different gene, neonatal, KF absent), Indian childhood cirrhosis (dietary copper excess, different geography, KF present occasionally)",
        "treatment_alert": "Penicillamine: neurological worsening 20-50% — use trientine first-line for neuropsychiatric WD; zinc safe maintenance/presymptomatic; tetrathiomolybdate emerging neuro; liver transplant curative. ACUTE WD ALF: haemolysis (Coombs-negative) + ALF = liver transplant urgently; chelators ineffective in ALF — not enough time",
        "seed": 1150,
        "cohort_n": 40,
        # Clinical rates
        "liver_disease_rate": 0.85,
        "neuro_rate": 0.55,
        "kf_ring_rate": 0.72,
        "haemolysis_rate": 0.20,
        "psychiatric_rate": 0.35,
        "renal_involvement_rate": 0.25,
        "drug_error_rate": 0.08,
        "diagnosis_delay_rate": 0.60,     # >2y delay to diagnosis
        "transplant_rate": 0.15,
        "surveillance_adherent_rate": 0.72,
        "severity_weights": {"Mild": 0.25, "Moderate": 0.40, "Severe": 0.35},
        "primary_complication": "Hepatic cirrhosis / neuropsychiatric decompensation",
        "liver_biochem": "Elevated AST/ALT (often AST>ALT), low ceruloplasmin, elevated 24h urine copper",
        "ggt_pattern": "Variable (low in acute haemolysis; elevated in cholestasis)",
    },

    # ── HFE — Hereditary Haemochromatosis ──────────────────────────────────
    {
        "gene": "HFE",
        "protein": "HFE protein (homeostatic iron regulator)",
        "alias": (
            "HFE; OMIM gene 613609; 6p21.3; ~348 aa; Hereditary haemochromatosis type 1 (OMIM #235200); "
            "AR (C282Y homozygous most severe); C282Y (p.Cys282Tyr) founder mutation ~85% HH1 N European; "
            "H63D compound heterozygous C282Y/H63D mild; iron overload — liver (cirrhosis/HCC), heart "
            "(cardiomyopathy/arrhythmia), endocrine (bronze diabetes, hypogonadism), joints (CPPD); "
            "phlebotomy first-line; 1:200–1:400 Northern European; low penetrance (1-5% C282Y/C282Y develop severe disease)"
        ),
        "aa": "~348 aa",
        "kDa": "~38 kDa",
        "gene_class": "HLA class I-like protein; hepcidin regulator; interacts with transferrin receptor 1 (TfR1) and β2-microglobulin; modulates BMP-SMAD signalling pathway for hepcidin induction",
        "locus": "6p21.3",
        "omim_gene": 613609,
        "omim_disease": 235200,
        "phenotype": "Hereditary haemochromatosis type 1 — progressive iron overload; liver fibrosis/cirrhosis, hepatocellular carcinoma (HCC 20-fold risk), cardiomyopathy, diabetes mellitus (bronze diabetes), hypogonadism, arthropathy (MCP joints 2-3 PATHOGNOMONIC), bronze skin",
        "disease": (
            "HFE encodes a non-classical MHC class I-like protein that regulates body iron "
            "homeostasis by modulating hepcidin expression (the master iron hormone). "
            "NORMAL FUNCTION: HFE forms a complex with β2-microglobulin and transferrin receptor 1 "
            "(TfR1) on hepatocytes, sensing plasma transferrin-saturation; at high iron, HFE-TfR1 "
            "complex dissociates → HFE joins BMP receptor complex (BMPR2/ALK2 + HJV co-receptor) "
            "→ activates BMP-SMAD1/5/8 signalling → hepcidin transcription → hepcidin secreted → "
            "binds and degrades ferroportin-1 (FPN1) on duodenal enterocytes and macrophages → "
            "reduces iron absorption and recycling → iron homeostasis. "
            "PATHOMECHANISM: C282Y (p.Cys282Tyr) disrupts a critical disulphide bond (Cys260-Cys282) "
            "in the α3 domain → HFE protein misfolded → retained in ER → fails to reach cell surface → "
            "cannot associate with TfR1 → BMP-SMAD signalling blunted → inappropriately low hepcidin "
            "→ unchecked duodenal iron absorption → systemic iron overload. "
            "IRON DEPOSITION: liver (hepatocyte predominant — periportal zone 1 first), heart, "
            "pancreas (β-cells → diabetes), pituitary/gonads (hypogonadism), joints (CPPD arthropathy "
            "in MCP2-3 joints PATHOGNOMONIC — pseudogout pattern), skin (melanin + iron → bronze). "
            "CLINICAL TRIAD (classic): CIRRHOSIS + DIABETES + BRONZE SKIN = BRONZE DIABETES. "
            "LABORATORY: Serum ferritin >300 µg/L (male) or >200 µg/L (female) + "
            "transferrin saturation >45% — highly suggestive. Genotyping (C282Y, H63D). "
            "MRI liver T2* quantification of iron burden. Liver biopsy (hepatic iron index ≥1.9 "
            "for haemochromatosis in heterozygotes). "
            "PENETRANCE: Low — only 1-5% of C282Y homozygotes develop severe iron overload; "
            "modifier genes (HFE2/HJV, TfR2, HAMP/hepcidin) and environmental factors (alcohol, "
            "menstruation) modify penetrance. "
            "TREATMENT: PHLEBOTOMY (venesection) 1 unit (~500 mL, ~250 mg iron) weekly until "
            "ferritin <50 µg/L; then maintenance every 2-3 months. Highly effective if pre-cirrhotic "
            "(reverses fibrosis, prevents HCC — does NOT reverse established cirrhosis or diabetes). "
            "HCC risk 20-fold elevated even with adequate phlebotomy if cirrhosis established → "
            "6-monthly ultrasound + AFP surveillance mandatory. "
            "CHELATION (deferoxamine/deferasirox): only if phlebotomy not possible (anaemia). "
            "FAMILY SCREENING: first-degree relatives mandatory (HFE genotyping + iron studies)."
        ),
        "inheritance": "AR for severe phenotype (C282Y/C282Y); C282Y carrier (C282Y/WT) heterozygous — rarely symptomatic; C282Y/H63D compound heterozygous — mild disease; H63D/H63D — rarely symptomatic",
        "hallmark": "Transferrin saturation >45% + serum ferritin elevated; C282Y/C282Y genotype; MCP2-3 arthropathy PATHOGNOMONIC (CPPD/pseudogout); bronze skin; hepatic iron overload on MRI; HCC risk 20-fold in cirrhosis",
        "key_ddx": "Secondary haemosiderosis (transfusion-dependent anaemia — no HFE mutation), Haemolytic anaemia (elevated ferritin acute-phase; transferrin saturation elevated but no C282Y), African iron overload (ferroportin disease — SLC40A1 mutation), Acaeruloplasminaemia (no copper transporter — ceruloplasmin absent, iron deposits brain + liver)",
        "treatment_alert": "Phlebotomy weekly until ferritin <50; maintenance every 2-3 months lifelong. HCC surveillance (ultrasound + AFP 6-monthly) MANDATORY if cirrhosis established — 20-fold risk remains even after iron depletion. Alcohol avoidance critical — potentiates hepatic iron toxicity. Vitamin C supplementation increases iron absorption — AVOID during iron loading phase",
        "seed": 1151,
        "cohort_n": 40,
        "liver_disease_rate": 0.80,
        "neuro_rate": 0.05,
        "kf_ring_rate": 0.00,             # KF rings NOT in HFE haemochromatosis
        "haemolysis_rate": 0.05,
        "psychiatric_rate": 0.10,
        "renal_involvement_rate": 0.10,
        "drug_error_rate": 0.05,
        "diagnosis_delay_rate": 0.70,     # often detected incidentally on blood tests
        "transplant_rate": 0.05,
        "surveillance_adherent_rate": 0.68,
        "severity_weights": {"Mild": 0.40, "Moderate": 0.35, "Severe": 0.25},
        "primary_complication": "Hepatic cirrhosis / HCC / bronze diabetes",
        "liver_biochem": "Elevated ferritin, transferrin saturation >45%; AST/ALT mildly elevated (Zn-ferritin normal in acute-phase — transferrin saturation more specific)",
        "ggt_pattern": "Mild elevation or normal in early stage",
    },

    # ── SERPINA1 — Alpha-1-Antitrypsin Deficiency ─────────────────────────
    {
        "gene": "SERPINA1",
        "protein": "Alpha-1-antitrypsin (A1AT / AAT / alpha1-proteinase inhibitor)",
        "alias": (
            "SERPINA1; OMIM gene 107400; 14q32.1; ~418 aa (mature); serine protease inhibitor (serpin); "
            "codominant inheritance; PiZZ genotype most severe; liver disease (Z-polymer hepatocyte entrapment) + "
            "lung disease (emphysema — neutrophil elastase imbalance); prevalence PiZZ 1:3,500 European; "
            "augmentation therapy (Prolastin/Zemaira) for lung; liver transplant curative; "
            "fazirsiran (RNA interference, hepatocyte-targeted) FDA 2024 for liver disease; "
            "lomitapide investigational"
        ),
        "aa": "~418 aa (418 aa mature; 394 aa after secretory peptide)",
        "kDa": "~52 kDa (glycosylated ~55 kDa)",
        "gene_class": "Serine protease inhibitor (serpin superfamily); suicide substrate inhibitor; reactive centre loop (RCL) baits target protease; inhibits neutrophil elastase (NE) / cathepsin G / proteinase 3 in lung; hepatocyte secreted into plasma",
        "locus": "14q32.1",
        "omim_gene": 107400,
        "omim_disease": 613490,
        "phenotype": "Alpha-1-antitrypsin deficiency — dual organ disease: liver (Z-polymer hepatocyte intracellular entrapment → neonatal hepatitis → cirrhosis → HCC) + lung (neutrophil elastase imbalance → panacinar emphysema; lower lobes; early onset <45y smokers; COPD)",
        "disease": (
            "SERPINA1 encodes alpha-1-antitrypsin (A1AT), the most abundant plasma protease inhibitor "
            "produced by hepatocytes and secreted into plasma at ~1-3 g/L. "
            "NORMAL FUNCTION: A1AT inhibits neutrophil elastase (NE) in the lung — after NE is released "
            "by activated neutrophils during infection/inflammation, A1AT binds and inactivates NE "
            "(suicide substrate mechanism — RCL inserts into NE active site irreversibly). "
            "Without A1AT, unopposed NE destroys alveolar wall elastin → emphysema. "
            "ALLELE NOMENCLATURE (Pi system): Wild-type M (PiMM — normal); "
            "Z allele (p.Glu342Lys) — most common disease allele (95% severe deficiency): "
            "Glu342Lys disrupts the salt bridge stabilising the E-A sheet of the serpin body → "
            "Z-A1AT polymerises within the ER of hepatocytes → POLYMER RETENTION → "
            "hepatocyte endoplasmic reticulum (ER) accumulation → ER stress → hepatocyte death "
            "by apoptosis → inflammation → neonatal hepatitis → progressive cirrhosis → HCC. "
            "S allele (p.Glu264Val): mild reduction in A1AT secretion; SZ compound heterozygous — "
            "intermediate risk. Null alleles (stop, frameshift): no protein produced — lung disease "
            "but NO liver disease (no protein to polymerise). "
            "LIVER DISEASE (PiZZ): neonatal hepatitis (10-15% of PiZZ neonates; conjugated hyperbilirubinaemia; "
            "resolves in most by age 6 months — 20% progress to cirrhosis); childhood cirrhosis; "
            "adult-onset cirrhosis; HCC (hepatocellular carcinoma). PAS-positive diastase-resistant "
            "(PAS-D) globules in periportal hepatocytes — PATHOGNOMONIC on liver biopsy. "
            "LUNG DISEASE: panacinar emphysema predominantly lower lobes (opposite of smoking — upper lobes); "
            "onset <45 years in smokers or <50 years non-smokers; early-onset COPD without obvious cause "
            "must prompt A1AT testing. Bronchiectasis in some. "
            "DIAGNOSIS: serum A1AT level <0.8 g/L (or <11 µmol/L) — note: A1AT is an acute-phase "
            "reactant → may be falsely normal in inflammation; confirm with Pi phenotyping (isoelectric "
            "focusing — IEF) or SERPINA1 genotyping. "
            "TREATMENT — LUNG: Augmentation therapy (A1AT concentrate IV weekly — Prolastin-C/Zemaira/ "
            "Glassia): slows CT densitometry-measured emphysema progression; indicated when A1AT <0.8 g/L "
            "+ FEV1 25-80% predicted + non-smoker/ex-smoker. Bronchodilators, pulmonary rehab. "
            "Lung transplant for severe COPD. "
            "TREATMENT — LIVER: No established medical therapy for liver disease until fazirsiran "
            "(ARO-AAT, RNAi hepatocyte-targeted — reduces Z-polymer production → reduces ER stress) "
            "FDA 2024 for liver disease. Liver transplant curative (donor liver has normal SERPINA1 → "
            "normal A1AT production; also corrects lung disease long-term). "
            "SMOKING: ABSOLUTELY CONTRAINDICATED — dramatically accelerates emphysema (10-fold faster "
            "progression); single most important modifiable risk factor."
        ),
        "inheritance": "Codominant (Pi system — both alleles contribute to phenotype); PiZZ = both alleles Z; SZ compound heterozygous = intermediate; heterozygous MZ = carrier, usually clinically normal lung but slight liver risk",
        "hallmark": "Low serum A1AT <0.8 g/L; Z allele on Pi phenotyping/genotyping; PAS-D globules periportal hepatocytes on liver biopsy PATHOGNOMONIC; panacinar emphysema lower lobes on CT; early-onset COPD <45y",
        "key_ddx": "COPD from smoking (upper lobe emphysema; normal A1AT), Cystic fibrosis (CFTR — different gene; sweat test; obstructive + pancreatic; A1AT normal), Neonatal hepatitis (multiple causes; A1AT level + Pi typing mandatory in any neonatal cholestasis), Autoimmune hepatitis (ANA/ASMA; responds to steroids; PAS-D globules absent)",
        "treatment_alert": "SMOKING ABSOLUTELY CI — 10-fold acceleration of emphysema; single most critical intervention. Augmentation therapy (A1AT IV weekly) for lung disease (FEV1 25-80%, A1AT <0.8 g/L). Fazirsiran FDA 2024 for Z-polymer liver disease. Liver transplant curative (also corrects lung). Acute-phase response falsely normalises A1AT — Pi typing/genotyping mandatory for accurate diagnosis",
        "seed": 1152,
        "cohort_n": 40,
        "liver_disease_rate": 0.55,
        "neuro_rate": 0.05,
        "kf_ring_rate": 0.00,
        "haemolysis_rate": 0.05,
        "psychiatric_rate": 0.10,
        "renal_involvement_rate": 0.05,
        "drug_error_rate": 0.05,
        "diagnosis_delay_rate": 0.75,
        "transplant_rate": 0.12,
        "surveillance_adherent_rate": 0.65,
        "severity_weights": {"Mild": 0.30, "Moderate": 0.40, "Severe": 0.30},
        "primary_complication": "COPD emphysema / hepatic cirrhosis / HCC",
        "liver_biochem": "Elevated ALT/AST; low serum A1AT; PiZZ on isoelectric focusing; PAS-D globules on biopsy",
        "ggt_pattern": "Variable; cholestatic pattern in neonates; elevated in cirrhosis",
    },

    # ── ABCB11 — PFIC2 / BSEP Deficiency ─────────────────────────────────
    {
        "gene": "ABCB11",
        "protein": "BSEP (bile salt export pump / ABCB11)",
        "alias": (
            "ABCB11; OMIM gene 603201; 2q31.1; ~1321 aa; PFIC2 = Progressive Familial Intrahepatic Cholestasis Type 2 "
            "(OMIM #601847); AR; bile salt export pump on canalicular membrane (rate-limiting step for primary bile acid export); "
            "LOW GGT cholestasis PATHOGNOMONIC (distinguishes from PFIC3/MDR3); HIGH serum bile acids (>200 µmol/L); "
            "HCC risk even in young children; odevixibat (IBAT inhibitor) FDA 2021; liver transplant; "
            "BSEP disease = PFIC2 + drug-induced cholestasis + intrahepatic cholestasis of pregnancy ICP"
        ),
        "aa": "~1321 aa",
        "kDa": "~140 kDa",
        "gene_class": "ABC transporter (ATP-binding cassette superfamily B); canalicular bile salt export pump; transports primary bile acids (taurocholate > glycocholate) from hepatocyte cytoplasm into bile canaliculus",
        "locus": "2q31.1",
        "omim_gene": 603201,
        "omim_disease": 601847,
        "phenotype": "PFIC2 — severe progressive cholestasis in infancy/early childhood; LOW GGT (PATHOGNOMONIC — distinguishes from PFIC3); very high serum bile acids (>200 µmol/L); pruritus (intractable); jaundice; hepatosplenomegaly; HCC risk in first decade of life (even without cirrhosis)",
        "disease": (
            "ABCB11 encodes BSEP (bile salt export pump), the rate-limiting transporter for primary "
            "bile acid export from hepatocytes across the canalicular membrane into bile. "
            "NORMAL FUNCTION: BSEP transports conjugated primary bile acids (taurocholate, "
            "glycocholate, taurochenodeoxycholate) in an ATP-dependent manner from hepatocyte "
            "cytosol into the bile canaliculus → drives bile flow (bile acid-dependent bile flow ~50%). "
            "PATHOMECHANISM: BSEP LOF → bile acids accumulate in hepatocytes → bile acid toxicity "
            "→ hepatocyte necrosis/apoptosis → progressive fibrosis → cirrhosis. "
            "KEY DIAGNOSTIC FEATURE — LOW GGT: GGT (gamma-glutamyltransferase) is expressed in "
            "biliary epithelium and depends on bile flow for release into serum. In PFIC1 (ATP8B1) "
            "and PFIC2 (ABCB11), bile flow is reduced from the outset — GGT is NOT elevated because "
            "biliary epithelial GGT cannot be released (insufficient bile flow). This contrasts with "
            "PFIC3 (ABCB4/MDR3), in which bile flow is present but lacks phospholipids → bile "
            "becomes lithogenic → biliary epithelial damage → GGT released → HIGH GGT. "
            "LOW GGT + cholestasis + pruritus in an infant = PFIC1 or PFIC2 → GGT pattern determines "
            "whether further testing needed for PFIC1 vs PFIC2. "
            "HCC RISK: BSEP-deficient patients have a very high risk of hepatocellular carcinoma (HCC) "
            "even in the first decade of life, even without established cirrhosis — the retained bile "
            "acids are directly genotoxic. HCC surveillance from age 3-6 months mandatory. "
            "MILD FORMS: Heterozygous variants + hot conditions → transient neonatal cholestasis (TNC); "
            "intrahepatic cholestasis of pregnancy (ICP) — heterozygous ABCB11 variants in ~15% ICP; "
            "drug-induced cholestasis (cyclosporin A, rifampicin, oestrogens — all inhibit BSEP). "
            "TREATMENT: (1) ODEVIXIBAT (Bylvay — IBAT inhibitor, ileal bile acid transporter): "
            "FDA approved April 2021 for PFIC2; blocks intestinal bile acid reabsorption → reduces "
            "total bile acid pool → reduces hepatocyte bile acid overload; reduces pruritus and "
            "cholestasis in ~40-50% responders. (2) MARALIXIBAT: similar IBAT inhibitor. "
            "(3) Partial internal biliary diversion (PIBD) / external biliary diversion — surgical. "
            "(4) Liver transplant: definitive; BUT BSEP absent from donor liver not an issue "
            "(donor has normal BSEP). NOTE: post-transplant de novo BSEP antibodies in some patients — "
            "recurrent disease rarely."
        ),
        "inheritance": "AR (autosomal recessive); compound heterozygous common; D482G most common European mild allele; E297G moderate; BSEP2-null alleles (D482G/any null) moderate-severe; null/null severe",
        "hallmark": "LOW GGT cholestasis in infancy (PATHOGNOMONIC — distinguishes from PFIC3 HIGH GGT); very high serum bile acids (>200 µmol/L); intractable pruritus; HCC risk even in childhood; normal or low GGT with elevated ALT/bilirubin",
        "key_ddx": "PFIC1/ATP8B1 (also LOW GGT but extrahepatic features: diarrhoea, deafness, pancreatitis; liver histology: mild to moderate cholestasis; canalicular BSEP expression normal in PFIC1), PFIC3/ABCB4 (HIGH GGT — critical distinguisher; cholelithiasis; phospholipid absent in bile), Alagille syndrome/JAG1 (bile duct paucity; butterfly vertebrae; cardiac; GGT elevated or variable), Biliary atresia (surgical emergency; neonatal acholic stools; DISIDA scan; bile duct absent; GGT elevated)",
        "treatment_alert": "HCC SURVEILLANCE from age 3-6 months mandatory even pre-cirrhotic (6-monthly ultrasound + AFP). LOW GGT cholestasis = PFIC1 or PFIC2 — never reassure as benign. Odevixibat FDA 2021 for PFIC2 pruritus and cholestasis. BSEP inhibitors (cyclosporin, rifampicin, oestrogens, certain antibiotics) trigger acute cholestasis in BSEP-deficient or heterozygous patients",
        "seed": 1153,
        "cohort_n": 40,
        "liver_disease_rate": 0.98,
        "neuro_rate": 0.00,
        "kf_ring_rate": 0.00,
        "haemolysis_rate": 0.05,
        "psychiatric_rate": 0.05,
        "renal_involvement_rate": 0.05,
        "drug_error_rate": 0.08,
        "diagnosis_delay_rate": 0.45,
        "transplant_rate": 0.55,
        "surveillance_adherent_rate": 0.75,
        "severity_weights": {"Mild": 0.15, "Moderate": 0.35, "Severe": 0.50},
        "primary_complication": "Cholestatic cirrhosis / HCC (even in childhood)",
        "liver_biochem": "LOW GGT (PATHOGNOMONIC); elevated bilirubin (conjugated); elevated bile acids >200 µmol/L; elevated ALT; ALP mildly elevated",
        "ggt_pattern": "LOW GGT (distinguishes from PFIC3) — bile flow insufficient to release GGT from biliary epithelium",
    },

    # ── ATP8B1 — PFIC1 / FIC1 Deficiency / Byler Disease ─────────────────
    {
        "gene": "ATP8B1",
        "protein": "FIC1 (familial intrahepatic cholestasis 1 / ATP8B1)",
        "alias": (
            "ATP8B1; OMIM gene 602397; 18q21.31; ~1251 aa; PFIC1 = Byler disease (OMIM #211600); "
            "AR; aminophospholipid flippase (type 4 P-type ATPase); transports phosphatidylserine (PS) "
            "and phosphatidylethanolamine (PE) from outer to inner leaflet of canalicular membrane; "
            "LOW GGT cholestasis (same as PFIC2); EXTRAHEPATIC FEATURES distinguish PFIC1 — "
            "diarrhoea/malabsorption, sensorineural hearing loss, pancreatitis; "
            "liver transplant does NOT cure extrahepatic features (FIC1 expressed in intestine/ear not just liver)"
        ),
        "aa": "~1251 aa",
        "kDa": "~140 kDa",
        "gene_class": "P4-type ATPase (aminophospholipid flippase); maintains phospholipid asymmetry of canalicular membrane bilayer; ATP8B1 + CDC50A complex required for activity; hepatocyte canalicular membrane + intestinal enterocytes + inner ear",
        "locus": "18q21.31",
        "omim_gene": 602397,
        "omim_disease": 211600,
        "phenotype": "PFIC1 (Byler disease) — LOW GGT progressive cholestasis in infancy; extrahepatic features DISTINGUISH from PFIC2: watery diarrhoea/malabsorption, SNHL (sensorineural hearing loss), pancreatitis; liver transplant corrects liver but NOT extrahepatic features (FIC1 expressed in intestine and ear)",
        "disease": (
            "ATP8B1 encodes FIC1 (familial intrahepatic cholestasis 1), a P4-type ATPase "
            "aminophospholipid flippase that maintains the asymmetric distribution of phospholipids "
            "in the canalicular membrane bilayer — specifically translocating phosphatidylserine (PS) "
            "and phosphatidylethanolamine (PE) from the outer (canalicular) to inner (cytoplasmic) leaflet. "
            "NORMAL FUNCTION: Phospholipid asymmetry of the canalicular membrane is essential for "
            "its detergent resistance — bile acids are powerful detergents; the inner leaflet's "
            "sphingomyelin-rich composition provides a protective barrier. FIC1 also regulates "
            "farnesoid X receptor (FXR) activity — FXR is the bile acid nuclear receptor that "
            "induces BSEP expression and downregulates bile acid synthesis (FIC1 → FXR → BSEP). "
            "PATHOMECHANISM: ATP8B1 LOF → disrupted PS/PE asymmetry → canalicular membrane fragility "
            "→ increased bile acid toxicity + FXR dysfunction → reduced BSEP expression (secondary) "
            "→ bile acid accumulation → hepatocellular damage. "
            "FIC1 is expressed in HEPATOCYTES + INTESTINAL ENTEROCYTES + INNER EAR. "
            "HEPATIC: identical cholestasis to PFIC2 with LOW GGT (same mechanism of low bile flow "
            "preventing GGT release from biliary epithelium). "
            "EXTRAHEPATIC FEATURES (NOT present in PFIC2 — critical distinguisher): "
            "(1) DIARRHOEA/MALABSORPTION: FIC1 in intestinal enterocytes → PS flipping in enterocyte "
            "brush border → critical for lipid absorption; ATP8B1 LOF → watery diarrhoea, fat malabsorption, "
            "fat-soluble vitamin deficiency (A/D/E/K); worsens after liver transplant (restored bile "
            "flow → more bile acid to malabsorbing intestine). "
            "(2) SENSORINEURAL HEARING LOSS (SNHL): FIC1 in outer hair cells of cochlea; LOF → "
            "cochlear dysfunction → SNHL (mild-moderate). "
            "(3) PANCREATITIS: FIC1 in pancreatic ductal cells; LOF → recurrent pancreatitis. "
            "BYLER DISEASE vs BYLER SYNDROME: Byler disease = ATP8B1 (PFIC1); "
            "Byler syndrome (a historical term) = ABCB11 (PFIC2) in some classifications — "
            "now differentiated by LOW GGT + extrahepatic features vs LOW GGT without extrahepatic. "
            "LIVER TRANSPLANT: corrects hepatic cholestasis BUT diarrhoea may WORSEN post-transplant "
            "(restored bile flow + malabsorbing intestine with defective FIC1 → more bile acid in "
            "intestine → worse diarrhoea); SNHL and pancreatitis persist (FIC1 still absent in ear/pancreas). "
            "TREATMENT: odevixibat/maralixibat (IBAT inhibitors — same as PFIC2); partial biliary "
            "diversion; 4-phenylbutyrate (FXR activator — experimental); liver transplant (liver disease "
            "corrected; extrahepatic not corrected)."
        ),
        "inheritance": "AR; compound heterozygous or homozygous; I661T most common European mild/moderate; D554N and R600W severe; G308V Amish founder mutation (Byler kindred); mild alleles = benign recurrent intrahepatic cholestasis type 1 (BRIC1)",
        "hallmark": "LOW GGT cholestasis (same as PFIC2) + EXTRAHEPATIC FEATURES (diarrhoea, SNHL, pancreatitis) PATHOGNOMONIC to PFIC1 and distinguishing from PFIC2; liver transplant does NOT cure extrahepatic manifestations; FXR dysfunction reduces BSEP expression secondarily",
        "key_ddx": "PFIC2/ABCB11 (ALSO low GGT but NO extrahepatic features; BSEP absent on immunostaining; HCC risk higher in PFIC2), PFIC3/ABCB4 (HIGH GGT — definitive distinguisher from both PFIC1 and PFIC2), Biliary atresia (GGT elevated; acholic stools; urgent Kasai)",
        "treatment_alert": "Liver transplant corrects liver disease BUT WORSENS diarrhoea (restored bile flow to FIC1-deficient intestine); counsel families pre-transplant. SNHL: audiological follow-up mandatory. PFIC1 vs PFIC2 differentiation critical — BSEP immunostaining on liver biopsy: BSEP absent = PFIC2; BSEP present (secondary FXR suppression) = PFIC1. Odevixibat can improve pruritus; FXR agonists (obeticholic acid) investigational",
        "seed": 1154,
        "cohort_n": 40,
        "liver_disease_rate": 0.98,
        "neuro_rate": 0.20,               # SNHL
        "kf_ring_rate": 0.00,
        "haemolysis_rate": 0.00,
        "psychiatric_rate": 0.05,
        "renal_involvement_rate": 0.05,
        "drug_error_rate": 0.06,
        "diagnosis_delay_rate": 0.50,
        "transplant_rate": 0.60,
        "surveillance_adherent_rate": 0.72,
        "severity_weights": {"Mild": 0.20, "Moderate": 0.40, "Severe": 0.40},
        "primary_complication": "Cholestatic cirrhosis / fat malabsorption / SNHL / pancreatitis",
        "liver_biochem": "LOW GGT (PATHOGNOMONIC); elevated bilirubin; elevated bile acids; elevated ALT; diarrhoea/malabsorption",
        "ggt_pattern": "LOW GGT — same mechanism as PFIC2; BUT BSEP immunostaining present (FXR suppression secondary)",
    },

    # ── ABCB4 — PFIC3 / MDR3 Deficiency ──────────────────────────────────
    {
        "gene": "ABCB4",
        "protein": "MDR3 / ABCB4 (phosphatidylcholine flippase / multidrug resistance protein 3)",
        "alias": (
            "ABCB4; OMIM gene 171060; 7q21.12; ~1279 aa; PFIC3 = Progressive Familial Intrahepatic Cholestasis Type 3 "
            "(OMIM #602347); AR; phosphatidylcholine (PC) flippase on canalicular membrane; "
            "HIGH GGT cholestasis PATHOGNOMONIC — distinguishes from PFIC1 (low GGT) and PFIC2 (low GGT); "
            "cholelithiasis (gallstones — lithogenic bile lacking PC to solubilise cholesterol/bile acid); "
            "ursodeoxycholic acid (UDCA) effective in partial deficiency (residual MDR3 function); "
            "cholangiopathy / biliary cirrhosis"
        ),
        "aa": "~1279 aa",
        "kDa": "~140 kDa",
        "gene_class": "ABC transporter (MDR subfamily, same as ABCB11/BSEP); canalicular phosphatidylcholine (PC) flippase; translocates PC from inner to outer leaflet of canalicular membrane → PC secreted into bile as phospholipid micelles protecting bile duct epithelium from bile acid detergency",
        "locus": "7q21.12",
        "omim_gene": 171060,
        "omim_disease": 602347,
        "phenotype": "PFIC3 — HIGH GGT progressive cholestasis (biliary epithelial damage by unprotected bile acids); cholelithiasis (lithogenic bile lacking PC); cholangiopathy/sclerosing cholangitis pattern; biliary cirrhosis; UDCA effective in partial deficiency; intrahepatic cholestasis of pregnancy (ICP) in heterozygous females",
        "disease": (
            "ABCB4 encodes MDR3 (multidrug resistance protein 3), a phosphatidylcholine (PC) "
            "flippase on the canalicular membrane that translocates PC from the inner (cytoplasmic) "
            "leaflet to the outer (canalicular lumen) leaflet, resulting in PC secretion into bile. "
            "NORMAL FUNCTION: PC secreted by MDR3 forms mixed micelles with bile acids in bile "
            "canaliculi and ductules, solubilising cholesterol and — critically — protecting "
            "the biliary epithelium (cholangiocytes) from the detergent toxicity of bile acids. "
            "Without PC, bile acids act as detergents on cholangiocytes → cholangiocyte damage. "
            "PATHOMECHANISM: ABCB4 LOF → absence/reduction of PC in bile → 'toxic bile' (high "
            "bile acid-to-phospholipid ratio) → cholangiocyte membrane dissolution → "
            "inflammatory bile duct injury → cholangiopathy → biliary fibrosis → biliary cirrhosis. "
            "Also, PC is required to hold cholesterol in micellar solution — without PC, cholesterol "
            "precipitates → cholelithiasis (cholesterol/pigment gallstones from infancy/childhood). "
            "CRITICAL GGT DISTINCTION: Because cholangiocytes are damaged by toxic bile, GGT "
            "(expressed in cholangiocytes) is released into serum → HIGH GGT cholestasis. "
            "This is THE KEY diagnostic feature distinguishing PFIC3 (HIGH GGT) from PFIC1 and PFIC2 "
            "(both LOW GGT). HIGH GGT + cholestasis in a child/young adult = PFIC3 until proven otherwise. "
            "UDCA RESPONSE: In patients with partial MDR3 function (residual PC secretion), UDCA "
            "(ursodeoxycholic acid — 15-20 mg/kg/day) can significantly improve cholestasis by "
            "replacing toxic primary bile acids with less detergent UDCA → relieves cholangiocyte toxicity. "
            "UDCA is effective in ~70% of PFIC3 patients with partial function (null/null respond poorly). "
            "ICP (INTRAHEPATIC CHOLESTASIS OF PREGNANCY): heterozygous ABCB4 females → reduced PC "
            "secretion during oestrogen-induced increase in bile acid load in pregnancy → ICP (pruritus, "
            "elevated bile acids, elevated AST/ALT in 3rd trimester → fetal distress). "
            "TREATMENT: UDCA first-line (effective if residual function); cholestyramine for pruritus; "
            "biliary diversion if UDCA fails; liver transplant for cirrhosis (complete disease correction "
            "unlike PFIC1 — no extrahepatic involvement). Maralixibat/odevixibat less data for PFIC3. "
            "MONITORING: Abdominal ultrasound for gallstones (from infancy); LFTs; liver biopsy for staging."
        ),
        "inheritance": "AR (PFIC3 severe form); heterozygous ABCB4 variants → ICP in pregnancy, low-phospholipid-associated cholelithiasis (LPAC syndrome — recurrent gallstones <40y on US; responds to UDCA)",
        "hallmark": "HIGH GGT cholestasis (PATHOGNOMONIC — opposite of PFIC1/PFIC2); cholelithiasis (gallstones from childhood/infancy — low-phospholipid bile); cholangiopathy pattern; UDCA responsive in partial deficiency; biliary cirrhosis",
        "key_ddx": "PFIC1 (LOW GGT — opposite; extrahepatic features), PFIC2 (LOW GGT — opposite; HIGH HCC risk), Primary sclerosing cholangitis (PSC — associated with IBD; MRCP beaded ducts; p-ANCA; ABCB4 sequence normal), Biliary atresia (neonatal acholic stools; GGT elevated; DISIDA scan)",
        "treatment_alert": "UDCA (15-20 mg/kg/day) first-line — effective if residual MDR3 function; null/null may not respond. HIGH GGT cholestasis in child = PFIC3 differential — ABCB4 sequencing mandatory. ICP in pregnancy: UDCA reduces fetal distress; delivery at 37w if bile acids >40 µmol/L. LPAC syndrome (recurrent stones <40y): UDCA + cholecystectomy prevents recurrence; bile phospholipids absent in ERCP bile confirms diagnosis",
        "seed": 1155,
        "cohort_n": 40,
        "liver_disease_rate": 0.95,
        "neuro_rate": 0.00,
        "kf_ring_rate": 0.00,
        "haemolysis_rate": 0.00,
        "psychiatric_rate": 0.05,
        "renal_involvement_rate": 0.00,
        "drug_error_rate": 0.06,
        "diagnosis_delay_rate": 0.55,
        "transplant_rate": 0.35,
        "surveillance_adherent_rate": 0.70,
        "severity_weights": {"Mild": 0.25, "Moderate": 0.40, "Severe": 0.35},
        "primary_complication": "Biliary cirrhosis / cholelithiasis / cholangiopathy",
        "liver_biochem": "HIGH GGT (PATHOGNOMONIC); elevated bilirubin; elevated ALP (biliary pattern); elevated ALT; bile acids elevated",
        "ggt_pattern": "HIGH GGT — cholangiocyte damage by toxic bile (low-PC); definitive distinguisher from PFIC1/PFIC2",
    },

    # ── JAG1 — Alagille Syndrome ──────────────────────────────────────────
    {
        "gene": "JAG1",
        "protein": "Jagged-1 (JAG1 / Jagged ligand 1 / Serrate)",
        "alias": (
            "JAG1; OMIM gene 601920; 20p12.2; ~1218 aa; Alagille syndrome type 1 / ALGS1 (OMIM #118450); "
            "AD; Notch1 ligand; bile duct paucity (intrahepatic) + 4 major criteria: "
            "butterfly vertebrae, posterior embryotoxon, peripheral pulmonary stenosis, facial gestalt "
            "(broad forehead, deep-set eyes, pointed chin); 50% de novo; 1:30,000–1:70,000; "
            "variable expressivity (widely); xanthomata (hypercholesterolaemia); "
            "maralixibat FDA 2021 for pruritus; liver transplant ~30% by adulthood; "
            "vascular anomalies (intracranial/renal arteries) — major cause of morbidity/mortality"
        ),
        "aa": "~1218 aa",
        "kDa": "~134 kDa",
        "gene_class": "Notch signalling ligand (DSL family); cell-surface receptor ligand activating Notch1/2 intracellular signalling; required for biliary epithelium differentiation (bile duct morphogenesis), cardiac outflow tract, vertebral segmentation, eye development, facial morphogenesis",
        "locus": "20p12.2",
        "omim_gene": 601920,
        "omim_disease": 118450,
        "phenotype": "Alagille syndrome (ALGS1) — chronic cholestasis from bile duct paucity; butterfly vertebrae (70%); posterior embryotoxon (78-90%); peripheral pulmonary stenosis/cardiac (85-90%); characteristic facial gestalt; xanthomata; fat-soluble vitamin deficiency; renal anomalies; vascular anomalies (ICH/renal artery stenosis — major mortality cause)",
        "disease": (
            "JAG1 encodes Jagged-1, a cell-surface ligand for Notch receptors (Notch1, Notch2) "
            "that activates Notch intracellular signalling (NICD release → CBF1/RBPJ-κ transcription). "
            "NORMAL FUNCTION: JAG1-Notch signalling during embryogenesis is essential for: "
            "(1) BILE DUCT MORPHOGENESIS: Notch2 activation in periportal hepatocytes drives their "
            "specification into biliary epithelial cells (cholangiocytes) and bile duct formation. "
            "(2) CARDIAC OUTFLOW TRACT: septation of pulmonary and aortic outflows. "
            "(3) VERTEBRAL SEGMENTATION: somite boundary formation → butterfly vertebra patterning. "
            "(4) EYE: anterior chamber development → posterior embryotoxon (Schwalbe's line anteriorly displaced). "
            "(5) FACIAL: midface morphogenesis. "
            "PATHOMECHANISM: JAG1 haploinsufficiency (LOF — frameshift/nonsense/large deletions ~60%; "
            "missense ~30%; whole gene deletion ~7%) → reduced Notch signalling → "
            "bile duct PAUCITY (reduced number of interlobular bile ducts per portal tract — "
            "ductopenia: <0.5 bile ducts per portal tract is diagnostic criterion). "
            "CLINICAL CRITERIA (3/5 = diagnosis if JAG1+ or 3 relatives): "
            "(1) CHOLESTASIS (bile duct paucity on biopsy) — neonatal jaundice (most common presentation); "
            "(2) CARDIAC DEFECTS (85-90%): peripheral pulmonary stenosis (PPS) most common (70%); "
            "Tetralogy of Fallot (7-10%); complex CHD; intracardiac obstruction; "
            "(3) BUTTERFLY VERTEBRAE (70%): incomplete posterior vertebral arch fusion → "
            "butterfly appearance on AP spine X-ray PATHOGNOMONIC; "
            "(4) POSTERIOR EMBRYOTOXON (78-90%): Schwalbe's line visually displaced anteriorly "
            "on slit-lamp (present in 8-15% general population but >78% ALGS); "
            "(5) FACIAL GESTALT: prominent forehead, deep-set eyes, hypertelorism, broad nasal bridge, "
            "pointed chin — becomes more pronounced with age. "
            "VASCULAR ANOMALIES (major cause of mortality 34% in some series): "
            "intracranial vascular anomalies (ICVA — aneurysms, Moyamoya, arteriovenous malformations), "
            "renal artery stenosis, aortic coarctation → intracranial haemorrhage major cause of death. "
            "PRURITUS: severe, often intractable; the dominant quality-of-life problem; "
            "driven by serum bile acids. "
            "TREATMENT: (1) MARALIXIBAT (Livmarli — IBAT inhibitor): FDA Nov 2021 for pruritus in ALGS; "
            "reduces intestinal bile acid reabsorption → lowers systemic bile acids → reduces pruritus. "
            "(2) ODEVIXIBAT: similar IBAT inhibitor. "
            "(3) Ursodeoxycholic acid (UDCA): limited efficacy in ALGS (bile ducts structurally absent "
            "— different from PFIC3 where PC absent; UDCA mechanism less applicable). "
            "(4) Liver transplant (~30% by adulthood): improves liver disease but does NOT prevent "
            "vascular complications (ICVA present pre-transplant); does NOT correct cardiac/vertebral/eye "
            "anomalies. "
            "VARIABLE EXPRESSIVITY: familial cases can have different manifestations — same mutation → "
            "minimal features in parent, severe cholestasis in child (or vice versa). "
            "NOTCH3 variants cause cerebral autosomal dominant arteriopathy (CADASIL) — completely different."
        ),
        "inheritance": "AD (autosomal dominant) with incomplete penetrance; highly variable expressivity; 50% de novo mutations; NOTCH2 mutations → ALGS2 (rare, similar phenotype, more renal involvement)",
        "hallmark": "Bile duct paucity on liver biopsy (<0.5 ducts/portal tract); butterfly vertebrae (70%) PATHOGNOMONIC on X-ray; posterior embryotoxon on slit-lamp; peripheral pulmonary stenosis; characteristic facial gestalt; xanthomata/hypercholesterolaemia; vascular anomalies (ICH risk)",
        "key_ddx": "Biliary atresia (extrahepatic — complete obstruction; GGT elevated; urgent surgical; bile ducts present pre-Kasai), PFIC2 (LOW GGT; no cardiac/vertebral/eye features; ABCB11 sequencing), Neonatal cholestasis workup (A1AT first; CMV/EBV; metabolic screen — all must be excluded), Sclerosing cholangitis (adolescent/adult onset; IBD link; MRCP beaded ducts; p-ANCA)",
        "treatment_alert": "Vascular anomalies (ICVA) are the leading cause of mortality in ALGS — MRI brain + MRA + renal arteries MANDATORY at diagnosis and if neurological symptoms. Maralixibat FDA 2021 for pruritus. Pruritus is the dominant QoL problem — antihistamines often ineffective; rifampicin/naltrexone second-line; IBAT inhibitors most effective. Liver transplant does NOT correct cardiac/skeletal/eye/vascular features; address each independently",
        "seed": 1156,
        "cohort_n": 40,
        "liver_disease_rate": 0.90,
        "neuro_rate": 0.20,               # ICVA/vascular
        "kf_ring_rate": 0.00,
        "haemolysis_rate": 0.00,
        "psychiatric_rate": 0.10,
        "renal_involvement_rate": 0.40,
        "drug_error_rate": 0.07,
        "diagnosis_delay_rate": 0.40,
        "transplant_rate": 0.30,
        "surveillance_adherent_rate": 0.72,
        "severity_weights": {"Mild": 0.25, "Moderate": 0.45, "Severe": 0.30},
        "primary_complication": "Cholestasis / vascular anomaly (ICH) / cardiac / fat-soluble vitamin deficiency",
        "liver_biochem": "Elevated bilirubin (conjugated); elevated GGT (variable — biliary involvement); elevated ALP; elevated cholesterol/xanthomata; fat-soluble vitamin deficiency (A/D/E/K)",
        "ggt_pattern": "Variable (often elevated — biliary origin); not the key distinguisher (use cardiac/skeletal/eye criteria + biopsy for duct paucity)",
    },

    # ── SLC25A13 — Citrin Deficiency / NICCD / CTLN2 ─────────────────────
    {
        "gene": "SLC25A13",
        "protein": "Citrin (SLC25A13 / mitochondrial aspartate-glutamate carrier 2 / AGC2)",
        "alias": (
            "SLC25A13; OMIM gene 603859; 7q21.3; ~675 aa; Citrin deficiency (OMIM #603471); "
            "AR; mitochondrial aspartate-glutamate carrier (AGC2); "
            "NICCD (neonatal intrahepatic cholestasis caused by citrin deficiency) in infants → "
            "FTTDCD (failure to thrive + dyslipidaemia in citrin-deficient children) adolescents → "
            "CTLN2 (adult-onset type II citrullinaemia / hyperammonaemia) in adults; "
            "East Asian founder mutations (851del4 Japan/Korea/China 70-80% affected alleles); "
            "HIGH-CARBOHYDRATE DIET ABSOLUTELY CI — triggers hyperammonaemia and neuropsychiatric crisis in CTLN2; "
            "protein + fat diet preferred; arginine supplementation reduces ammonia; liver transplant curative CTLN2"
        ),
        "aa": "~675 aa",
        "kDa": "~74 kDa",
        "gene_class": "Mitochondrial inner membrane carrier (SLC25 family); imports aspartate from mitochondrial matrix into cytoplasm in exchange for glutamate → essential for malate-aspartate shuttle (MAS) (cytoplasmic NADH → mitochondria); urea cycle (aspartate is substrate for argininosuccinate synthetase, step 3); gluconeogenesis; de novo purine synthesis",
        "locus": "7q21.3",
        "omim_gene": 603859,
        "omim_disease": 603471,
        "phenotype": "Citrin deficiency — three age-related presentations: (1) NICCD (neonates): cholestasis + hypoproteinaemia + galactosaemia-like + fat-soluble vitamin deficiency; spontaneous resolution by 12 months in most; (2) FTTDCD (children): failure to thrive + dyslipidaemia + hypoglycaemia; (3) CTLN2 (adults): episodic hyperammonaemia + neuropsychiatric crisis triggered by HIGH CARBOHYDRATE INTAKE — PATHOGNOMONIC",
        "disease": (
            "SLC25A13 encodes citrin (AGC2 — aspartate-glutamate carrier 2), a mitochondrial inner "
            "membrane antiporter that exchanges mitochondrial aspartate for cytoplasmic glutamate. "
            "NORMAL FUNCTION: Citrin is the liver-specific isoform (AGC1/SLC25A12 = neuronal isoform). "
            "Citrin drives the malate-aspartate shuttle (MAS) — the main mechanism for shuttling "
            "reducing equivalents (cytoplasmic NADH) into mitochondria for oxidative phosphorylation. "
            "In the urea cycle: aspartate (exported by citrin from mitochondria) is the direct nitrogen "
            "donor for argininosuccinate synthetase (ASS1) at urea cycle step 3. "
            "In gluconeogenesis: OAA (oxaloacetate) → aspartate export → cytoplasmic OAA pool "
            "for PEPCK → PEP → glucose (gluconeogenesis from amino acids). "
            "PATHOMECHANISM: Citrin LOF → "
            "(1) Blocked MAS → cytoplasmic NADH accumulates → inhibits glycolysis + gluconeogenesis → "
            "impairs hepatic glucose production + causes cytoplasmic reducing stress. "
            "(2) Reduced aspartate export → urea cycle step 3 (ASS1) is aspartate-limited → "
            "citrulline cannot be converted to argininosuccinate → citrulline accumulates "
            "(citrullinaemia) → nitrogen cannot be detoxified as urea → ammonia accumulates. "
            "(3) CARBOHYDRATE PARADOX: glucose metabolism generates large amounts of cytoplasmic NADH "
            "→ worsens MAS block (more NADH substrate, no exit route) → exacerbates urea cycle failure "
            "→ hyperammonaemia triggered by eating high-carbohydrate foods (rice/sweets/bread/alcohol). "
            "CITRIN-DEFICIENT PATIENTS SELF-SELECT PROTEIN + FAT-RICH DIETS (often apparent from childhood) "
            "— this is a compensatory mechanism that reduces MAS burden. HIGH-CARBOHYDRATE DIET ABSOLUTELY "
            "CONTRAINDICATED IN CTLN2 (can precipitate fatal hyperammonaemic encephalopathy). "
            "AGE-RELATED PRESENTATIONS: "
            "(1) NICCD (neonates/infants): cholestasis (elevated bilirubin + bile acids), "
            "elevated liver enzymes, hypoproteinaemia, hypoglycaemia, galactosaemia-like picture "
            "(elevated galactose — galactose metabolism also NADH-dependent → blocked MAS), "
            "fat-soluble vitamin deficiency; usually resolves by 12 months; "
            "rare if high-carbohydrate formula feeding → worsens (switch to MCT/high-fat formula). "
            "(2) FTTDCD (1-11y): failure to thrive, dyslipidaemia (elevated TG, low HDL), "
            "hypoglycaemia, fatigue, pancreatitis — incomplete expression of CTLN2 features. "
            "(3) CTLN2 (>11y, usually adults): sudden-onset episodic hyperammonaemia (NH3 >200 µmol/L) "
            "triggered by carbohydrate intake (rice, sweets, sports drinks, alcohol) → "
            "encephalopathy (confusion, disorientation, aggression, hallucinations, coma, death); "
            "elevated plasma citrulline (>100 µmol/L — PATHOGNOMONIC); "
            "often preceded by carbohydrate craving/avoidance history. "
            "DIAGNOSIS: plasma amino acids (citrulline elevated); plasma ammonia during acute episode; "
            "NICCD: urine organic acids (galactitol); newborn screening: citrulline on MS/MS; "
            "SLC25A13 sequencing (851del4 founder mutation in East Asians). "
            "TREATMENT — CTLN2: Absolute prohibition of high-carbohydrate diet and alcohol; "
            "protein + fat-rich diet; arginine supplementation (bypasses the ASS1 aspartate-limitation "
            "by providing arginine directly → replenishes urea cycle distal to the block); "
            "sodium benzoate + phenylacetate (alternative nitrogen excretion); "
            "LIVER TRANSPLANT: curative for CTLN2 (hepatic citrin restored) — recommended early before "
            "irreversible neurological damage; recurrence ZERO post-transplant. "
            "NICCD: MCT-enriched formula (reduces NADH generation from fatty acid oxidation); "
            "most NICCD resolves spontaneously by 12 months."
        ),
        "inheritance": "AR; East Asian founder: 851del4 (c.851_854del, p.Met285ProfsTer4) accounts for ~70% Japanese/Korean/Chinese alleles; IVS16ins3kb second most common Japan; prevalent in East Asia (carrier rate ~1:70 Japanese), rare in non-East-Asian populations",
        "hallmark": "CTLN2: episodic hyperammonaemia triggered by HIGH-CARBOHYDRATE DIET (PATHOGNOMONIC — rice/sweets/alcohol trigger); elevated plasma citrulline (>100 µmol/L); patient self-selects protein/fat-rich diet from childhood; NICCD: neonatal cholestasis + galactosaemia-like picture; FTTDCD: failure to thrive + dyslipidaemia",
        "key_ddx": "Urea cycle defect (especially ASS1/citrullinaemia type I — also elevated citrulline but elevated urinary orotic acid; different gene; no carbohydrate trigger; present neonatally), Ornithine transcarbamoylase (OTC) deficiency (X-linked; elevated glutamine/orotic; no citrulline elevation), Organic acidaemia (elevated organic acids; citrulline normal or low), NAFLD/NASH (Western diet; no citrulline; no carbohydrate-hyperammonaemia link)",
        "treatment_alert": "HIGH-CARBOHYDRATE DIET ABSOLUTELY CI in CTLN2 — even one high-carb meal can trigger fatal hyperammonaemia. Carbohydrate-containing IV fluids (glucose drips) also potentially dangerous in CTLN2. Protein + fat diet + arginine supplementation is treatment AND prevention. Liver transplant curative for CTLN2 — refer early. NICCD: switch to MCT/high-fat formula — NOT standard high-carb infant formula. East Asian child with recurrent neuropsychiatric episodes + hyperammonaemia + carbohydrate link = CTLN2 until proven otherwise",
        "seed": 1157,
        "cohort_n": 40,
        "liver_disease_rate": 0.65,        # NICCD → resolves + CTLN2 recurs
        "neuro_rate": 0.70,                # CTLN2 encephalopathy
        "kf_ring_rate": 0.00,
        "haemolysis_rate": 0.05,
        "psychiatric_rate": 0.45,          # CTLN2 psychiatric symptoms
        "renal_involvement_rate": 0.05,
        "drug_error_rate": 0.12,           # carbohydrate diet error common
        "diagnosis_delay_rate": 0.65,
        "transplant_rate": 0.30,
        "surveillance_adherent_rate": 0.70,
        "severity_weights": {"Mild": 0.20, "Moderate": 0.45, "Severe": 0.35},
        "primary_complication": "Hyperammonaemic encephalopathy (CTLN2) / neonatal cholestasis (NICCD)",
        "liver_biochem": "Elevated citrulline; elevated ammonia (episodic); elevated LFTs in NICCD; galactosaemia-like markers in NICCD; plasma amino acid chromatography essential",
        "ggt_pattern": "Variable in NICCD; often normal between CTLN2 episodes (liver structurally normal in CTLN2)",
    },
]


def _gen_patients(gene_data: dict) -> list:
    """Generate deterministic 40-patient cohort for one gene."""
    rng = random.Random(gene_data["seed"])
    patients = []
    sev_choices = list(gene_data["severity_weights"].keys())
    sev_probs = list(gene_data["severity_weights"].values())

    for i in range(gene_data["cohort_n"]):
        pid = i + 1
        # weighted severity
        sev = rng.choices(sev_choices, weights=sev_probs, k=1)[0]
        age_dx = rng.randint(0, 45) if gene_data["gene"] not in ("ATP7B", "HFE") else rng.randint(5, 55)
        sex = rng.choice(["M", "F"])

        liver_dis = rng.random() < gene_data["liver_disease_rate"]
        neuro = rng.random() < gene_data["neuro_rate"]
        kf = rng.random() < gene_data["kf_ring_rate"]
        haemo = rng.random() < gene_data["haemolysis_rate"]
        psych = rng.random() < gene_data["psychiatric_rate"]
        renal = rng.random() < gene_data["renal_involvement_rate"]
        drug_err = rng.random() < gene_data["drug_error_rate"]
        delay = rng.random() < gene_data["diagnosis_delay_rate"]
        transplant = rng.random() < gene_data["transplant_rate"]
        surveillance = rng.random() < gene_data["surveillance_adherent_rate"]

        patients.append({
            "pid": f"{gene_data['gene']}-{pid:03d}",
            "gene": gene_data["gene"],
            "severity": sev,
            "age_at_dx_years": age_dx,
            "sex": sex,
            "liver_disease": liver_dis,
            "neuro_involvement": neuro,
            "kf_rings": kf,
            "haemolytic_anaemia": haemo,
            "psychiatric_involvement": psych,
            "renal_involvement": renal,
            "drug_error": drug_err,
            "diagnosis_delayed": delay,
            "received_transplant": transplant,
            "surveillance_adherent": surveillance,
        })
    return patients


def _gen_cohort() -> list:
    all_patients = []
    for gd in HEPATIC_GENES:
        all_patients.extend(_gen_patients(gd))
    return all_patients


def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in patients:
        sev[p["severity"]] = sev.get(p["severity"], 0) + 1

    liver_n = sum(1 for p in patients if p["liver_disease"])
    neuro_n = sum(1 for p in patients if p["neuro_involvement"])
    kf_n = sum(1 for p in patients if p["kf_rings"])
    haemo_n = sum(1 for p in patients if p["haemolytic_anaemia"])
    psych_n = sum(1 for p in patients if p["psychiatric_involvement"])
    renal_n = sum(1 for p in patients if p["renal_involvement"])
    drug_err_n = sum(1 for p in patients if p["drug_error"])
    delay_n = sum(1 for p in patients if p["diagnosis_delayed"])
    transplant_n = sum(1 for p in patients if p["received_transplant"])
    surveillance_n = sum(1 for p in patients if p["surveillance_adherent"])

    gene_stats = {}
    for gd in HEPATIC_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        ng = len(gpts)
        gene_stats[gd["gene"]] = {
            "liver_pct": round(100 * sum(1 for p in gpts if p["liver_disease"]) / ng, 1),
            "neuro_pct": round(100 * sum(1 for p in gpts if p["neuro_involvement"]) / ng, 1),
            "transplant_pct": round(100 * sum(1 for p in gpts if p["received_transplant"]) / ng, 1),
            "drug_error_pct": round(100 * sum(1 for p in gpts if p["drug_error"]) / ng, 1),
            "ggt_pattern": gd["ggt_pattern"],
        }

    disease_cat = {
        "Wilson's Disease (ATP7B)": round(100 * 40 / n, 1),
        "Hereditary Haemochromatosis (HFE)": round(100 * 40 / n, 1),
        "Alpha-1-Antitrypsin Deficiency (SERPINA1)": round(100 * 40 / n, 1),
        "PFIC2 / BSEP Deficiency (ABCB11)": round(100 * 40 / n, 1),
        "PFIC1 / FIC1 Deficiency (ATP8B1)": round(100 * 40 / n, 1),
        "PFIC3 / MDR3 Deficiency (ABCB4)": round(100 * 40 / n, 1),
        "Alagille Syndrome (JAG1)": round(100 * 40 / n, 1),
        "Citrin Deficiency / CTLN2 (SLC25A13)": round(100 * 40 / n, 1),
    }

    kpis = [
        {"label": "Total Patients", "value": str(n)},
        {"label": "Genes Covered", "value": "8"},
        {"label": "Liver Disease", "value": f"{round(100*liver_n/n,1)}%"},
        {"label": "Neuro Involvement", "value": f"{round(100*neuro_n/n,1)}%"},
        {"label": "KF Rings (WD)", "value": f"{round(100*kf_n/n,1)}%"},
        {"label": "Transplant Rate", "value": f"{round(100*transplant_n/n,1)}%"},
        {"label": "Drug Error", "value": f"{round(100*drug_err_n/n,1)}%"},
        {"label": "Delayed Dx", "value": f"{round(100*delay_n/n,1)}%"},
    ]

    return {
        "atlas_name": "Hepatic-Atlas",
        "atlas_subtitle": (
            "ATP7B·HFE·SERPINA1·ABCB11·ATP8B1·ABCB4·JAG1·SLC25A13 — "
            "320 patients (8×40, seeds 1150–1157)"
        ),
        "n_genes": 8,
        "n_patients": n,
        "seeds": "1150–1157",
        "description": (
            "Comprehensive atlas of 8 major hereditary liver diseases: "
            "ATP7B/Wilson's disease (AR; copper accumulation; KF rings; ceruloplasmin low; "
            "24h urine copper >100 µg; penicillamine/trientine/zinc; liver transplant curative; "
            "H1069Q Central/East European, R778L East Asian); "
            "HFE/Hereditary haemochromatosis type 1 (AR; C282Y/H63D; iron overload; liver/heart/pancreas/joints; "
            "MCP2-3 arthropathy PATHOGNOMONIC; phlebotomy first-line; HCC 20-fold risk in cirrhosis; "
            "low penetrance 1-5% C282Y/C282Y; 1:200-400 N European); "
            "SERPINA1/A1AT deficiency (codominant; PiZZ; Z-polymer ER entrapment → liver; "
            "unopposed neutrophil elastase → emphysema lower lobes; smoking ABSOLUTELY CI; "
            "augmentation therapy lung; fazirsiran FDA 2024 liver; transplant curative); "
            "ABCB11/PFIC2 (AR; BSEP; LOW GGT PATHOGNOMONIC; HCC risk childhood even pre-cirrhotic; "
            "odevixibat FDA 2021; bile acid toxicity); "
            "ATP8B1/PFIC1/Byler (AR; FIC1; LOW GGT + EXTRAHEPATIC features: diarrhoea/SNHL/pancreatitis; "
            "liver transplant does NOT cure extrahepatic); "
            "ABCB4/PFIC3 (AR; MDR3; HIGH GGT PATHOGNOMONIC — opposite of PFIC1/2; cholelithiasis; "
            "UDCA effective partial deficiency); "
            "JAG1/Alagille (AD; 50% de novo; bile duct paucity; butterfly vertebrae; PPS; "
            "posterior embryotoxon; vascular anomalies ICH — major mortality cause; "
            "maralixibat FDA 2021 pruritus); "
            "SLC25A13/Citrin deficiency (AR; East Asian; NICCD → CTLN2; "
            "HIGH CARBOHYDRATE ABSOLUTELY CI triggers hyperammonaemia; protein/fat diet; "
            "arginine supplementation; liver transplant curative CTLN2)."
        ),
        "aggregate_clinical": {
            "liver_disease_pct": round(100 * liver_n / n, 1),
            "neuro_involvement_pct": round(100 * neuro_n / n, 1),
            "kf_rings_pct": round(100 * kf_n / n, 1),
            "haemolytic_anaemia_pct": round(100 * haemo_n / n, 1),
            "psychiatric_involvement_pct": round(100 * psych_n / n, 1),
            "renal_involvement_pct": round(100 * renal_n / n, 1),
            "drug_error_pct": round(100 * drug_err_n / n, 1),
            "diagnosis_delayed_pct": round(100 * delay_n / n, 1),
            "transplant_rate_pct": round(100 * transplant_n / n, 1),
            "surveillance_adherent_pct": round(100 * surveillance_n / n, 1),
        },
        "drug_alerts": [
            {
                "type": "danger",
                "title": "SLC25A13 (CITRIN / CTLN2): HIGH-CARBOHYDRATE DIET ABSOLUTELY CONTRAINDICATED",
                "body": (
                    "In adult-onset type II citrullinaemia (CTLN2 — citrin deficiency), "
                    "carbohydrate metabolism generates large cytoplasmic NADH loads that cannot be "
                    "shuttled into mitochondria (malate-aspartate shuttle requires citrin). "
                    "Excess cytoplasmic NADH inhibits the urea cycle (aspartate supply to ASS1 fails) "
                    "→ citrulline accumulates → ammonia cannot be detoxified → acute hyperammonaemia "
                    "(NH3 >200 µmol/L within hours of a carbohydrate-rich meal). "
                    "Triggers: rice, bread, sweets, fruit juice, sports drinks, glucose IV infusions, "
                    "alcohol (converts NAD+ → NADH). "
                    "PRESENTATION: sudden-onset confusion, agitation, hallucinations, aggression, coma "
                    "— frequently misdiagnosed as psychiatric emergency. "
                    "TREATMENT: protein + fat diet; arginine supplementation; avoid all carbohydrate loads; "
                    "LIVER TRANSPLANT curative (eliminates the hepatic citrin deficiency). "
                    "East Asian child or adult with episodic neuropsychiatric symptoms after carbohydrate "
                    "intake = CTLN2 until proven otherwise — measure plasma ammonia + amino acids. "
                    "IV GLUCOSE DRIPS CONTRAINDICATED during acute episodes."
                ),
            },
            {
                "type": "danger",
                "title": "SERPINA1 (A1AT DEFICIENCY): SMOKING ABSOLUTELY CONTRAINDICATED — 10× FASTER EMPHYSEMA",
                "body": (
                    "Alpha-1-antitrypsin deficiency (PiZZ) results in unprotected neutrophil elastase "
                    "(NE) activity in the lung — even in non-smokers, NE gradually destroys alveolar "
                    "elastin (panacinar emphysema, lower lobes, onset by age 40-50). "
                    "Smoking amplifies this by: (1) recruiting massive numbers of neutrophils → more NE; "
                    "(2) tobacco smoke oxidises the reactive-centre loop methionine of any residual A1AT "
                    "→ inactivates functional A1AT. Combined effect: 10-fold acceleration of emphysema. "
                    "A PiZZ smoker develops severe COPD by age 30-35, compared to ~50 years in non-smokers. "
                    "Passive smoke also significantly increases NE burden. "
                    "SINGLE MOST IMPORTANT INTERVENTION: smoking cessation absolutely mandatory — "
                    "more effective than augmentation therapy in modifying disease course. "
                    "Augmentation therapy (IV A1AT weekly) is only indicated if: FEV1 25-80% predicted, "
                    "ex-smoker/non-smoker, established emphysema by CT densitometry. "
                    "Cannabis smoking equally dangerous — cannabinoid-induced NE release + smoke toxicity."
                ),
            },
            {
                "type": "danger",
                "title": "PFIC1/2 (ATP8B1/ABCB11): LOW GGT + CHOLESTASIS — DO NOT REASSURE AS BENIGN",
                "body": (
                    "In PFIC type 1 (ATP8B1/FIC1 deficiency) and type 2 (ABCB11/BSEP deficiency), "
                    "serum GGT is paradoxically LOW or NORMAL despite severe cholestasis — because "
                    "bile flow is insufficient to release GGT from biliary epithelium into serum. "
                    "LOW GGT cholestasis is SEVERE and progressive — not a sign of mild or benign disease. "
                    "PFIC2 specifically carries a very high risk of hepatocellular carcinoma (HCC) "
                    "even in the first decade of life, even before cirrhosis develops — retained bile "
                    "acids are directly genotoxic to hepatocytes. "
                    "HCC surveillance (6-monthly ultrasound + AFP) must begin from 3-6 months of age. "
                    "GGT IS the key diagnostic tool for PFIC typing: "
                    "LOW GGT = PFIC1 (ATP8B1) or PFIC2 (ABCB11) → differentiate by extrahepatic "
                    "features (PFIC1: diarrhoea/SNHL/pancreatitis) + BSEP immunostaining. "
                    "HIGH GGT = PFIC3 (ABCB4/MDR3) → cholelithiasis + cholangiopathy. "
                    "Never tell a parent 'the GGT is normal so the cholestasis is milder.'"
                ),
            },
            {
                "type": "warning",
                "title": "WILSON'S DISEASE (ATP7B): PENICILLAMINE CAN WORSEN NEUROLOGICAL SYMPTOMS — USE TRIENTINE FIRST FOR NEURO",
                "body": (
                    "Penicillamine (D-penicillamine) is a copper chelating agent that mobilises "
                    "copper from tissue deposits into the bloodstream and urine. In patients with "
                    "neuropsychiatric Wilson's disease, initiation of penicillamine can cause "
                    "paradoxical worsening of neurological symptoms in 20-50% of patients. "
                    "MECHANISM: rapid copper mobilisation from hepatic stores → transient rise in "
                    "circulating copper → increased copper deposition in brain → neurological deterioration "
                    "(tremor, dysarthria, dystonia worsen acutely; some patients never recover "
                    "to pre-treatment neurological baseline). "
                    "RECOMMENDATION: For NEUROPSYCHIATRIC Wilson's disease presentation, "
                    "use TRIENTINE (triethylenetetramine dihydrochloride / Syprine / Cuvrior — FDA 2022 "
                    "for trientine tetrahydrochloride) or ZINC as first-line — lower risk of neurological "
                    "worsening. Reserve penicillamine for predominantly hepatic Wilson's disease "
                    "(no neurological features). "
                    "ZINC: safest option for presymptomatic/maintenance — slower copper depletion "
                    "(induces intestinal metallothionein trapping copper in enterocytes); NOT for acute "
                    "presentations (too slow). TETRATHIOMOLYBDATE: most promising for neuro WD (clinical trials)."
                ),
            },
            {
                "type": "warning",
                "title": "ATP8B1 (PFIC1): LIVER TRANSPLANT WORSENS DIARRHOEA — COUNSEL FAMILIES",
                "body": (
                    "In PFIC1 (FIC1/ATP8B1 deficiency), liver transplant corrects the hepatic "
                    "cholestasis and prevents cirrhosis — but FIC1 is also expressed in intestinal "
                    "enterocytes and cochlear cells. "
                    "POST-TRANSPLANT: bile flow is restored (normal donor liver) but the intestine "
                    "still lacks FIC1 → bile acids delivered to the FIC1-deficient intestine "
                    "WORSEN malabsorption and diarrhoea (more bile acids = more detergent effect "
                    "on malabsorbing enterocytes). "
                    "CONSEQUENCE: diarrhoea often worsens significantly post-transplant in PFIC1. "
                    "Sensorineural hearing loss (SNHL) and pancreatitis also PERSIST (cochlear "
                    "and pancreatic FIC1 not restored by liver transplant). "
                    "PRE-TRANSPLANT COUNSELLING: families must be informed that diarrhoea, SNHL, "
                    "and pancreatitis will not be cured by liver transplant — only the hepatic "
                    "cholestasis is corrected. Post-transplant nutritional support for malabsorption "
                    "requires close dietetic follow-up. "
                    "PFIC2/ABCB11: liver transplant corrects cholestasis AND HCC risk; "
                    "no extrahepatic issues — better post-transplant outcome than PFIC1."
                ),
            },
            {
                "type": "warning",
                "title": "HFE (HAEMOCHROMATOSIS): HCC SURVEILLANCE MANDATORY EVEN AFTER IRON DEPLETION",
                "body": (
                    "In hereditary haemochromatosis (HFE/C282Y), successful iron depletion by phlebotomy "
                    "reduces ferritin to target (<50 µg/L) and prevents most complications if started "
                    "before cirrhosis develops. HOWEVER: if hepatic cirrhosis has already developed, "
                    "the HCC risk (approximately 20-fold elevated versus general population) PERSISTS "
                    "even after complete iron depletion — the cirrhotic liver remains at high HCC risk "
                    "regardless of iron normalisation. "
                    "Alcohol use dramatically potentiates iron toxicity and further increases HCC risk "
                    "(alcohol interferes with hepcidin production and increases intestinal iron absorption). "
                    "MANDATORY SURVEILLANCE in HFE cirrhosis: 6-monthly liver ultrasound + serum AFP. "
                    "Phlebotomy does NOT prevent HCC if cirrhosis is established; however, "
                    "phlebotomy before cirrhosis prevents cirrhosis and effectively eliminates HCC risk "
                    "— early screening (family members, incidental ferritin elevation) is critical. "
                    "Also monitor for cardiac arrhythmia, diabetes, and hypogonadism annually."
                ),
            },
        ],
        "critical_rules": [
            "SLC25A13/CTLN2: HIGH-CARBOHYDRATE DIET (rice, sweets, IV glucose) ABSOLUTELY CI — triggers fatal hyperammonaemia; protein + fat diet + arginine supplementation mandatory",
            "SERPINA1/PiZZ: SMOKING ABSOLUTELY CI — 10-fold acceleration of emphysema; most impactful intervention; augmentation therapy (IV weekly) for lung if ex-smoker FEV1 25-80%",
            "ABCB11/PFIC2: HCC surveillance from 3-6 months mandatory even pre-cirrhotic — bile acids directly genotoxic; LOW GGT does NOT mean mild disease",
            "ATP8B1/PFIC1 vs ABCB11/PFIC2: both LOW GGT — differentiate by extrahepatic features (PFIC1: diarrhoea/SNHL/pancreatitis) + BSEP immunostaining on liver biopsy",
            "ABCB4/PFIC3: HIGH GGT — definitive distinguisher from PFIC1 (low) and PFIC2 (low); UDCA first-line for partial MDR3 deficiency; cholelithiasis from infancy",
            "ATP7B (Wilson's disease): penicillamine WORSENS neuro symptoms in 20-50% — use trientine/zinc first-line for neuropsychiatric presentation; liver transplant curative if ALF/decompensated cirrhosis",
            "JAG1/Alagille: vascular anomalies (ICVA) leading cause of mortality — MRI brain + MRA MANDATORY; butterfly vertebrae on X-ray PATHOGNOMONIC; maralixibat FDA 2021 for pruritus",
            "HFE haemochromatosis: HCC surveillance (6-monthly ultrasound + AFP) mandatory in cirrhosis even after iron depletion; phlebotomy pre-cirrhosis prevents HCC; C282Y/C282Y low penetrance (1-5%)",
        ],
        "pathway_targets": {
            "ATP7B_copper": "Copper-P1B-ATPase biliary excretion — penicillamine/trientine chelation; zinc (enterocyte metallothionein); tetrathiomolybdate (extrahepatic binding); liver transplant (ATP7B restored)",
            "HFE_iron": "BMP-SMAD-hepcidin axis — phlebotomy (iron removal); deferoxamine/deferasirox (chelation if phlebotomy impossible); hepcidin-mimetic drugs (investigational); no gene therapy yet",
            "SERPINA1_lung": "A1AT antiprotease protection — augmentation therapy (IV A1AT weekly); smoking cessation (most effective); fazirsiran (RNAi reduces Z-polymer hepatocyte load); liver transplant (restores A1AT production)",
            "ABCB11_ABCB4_bile": "Canalicular bile acid/PC export — odevixibat/maralixibat (IBAT inhibitors reduce intestinal BA reabsorption); biliary diversion; liver transplant; UDCA for PFIC3 partial deficiency",
            "ATP8B1_phospholipid": "Canalicular phospholipid asymmetry — odevixibat; 4-phenylbutyrate (FXR activator experimental); liver transplant (corrects liver not intestine/ear)",
            "JAG1_notch": "Notch signalling (biliary morphogenesis) — maralixibat/odevixibat for pruritus; UDCA limited efficacy; liver transplant (corrects cholestasis); vascular surveillance (ICH prevention)",
            "SLC25A13_MAS": "Malate-aspartate shuttle + urea cycle aspartate supply — high-protein/fat diet (reduce cytoplasmic NADH from carbohydrates); arginine supplementation (bypass ASS1 aspartate limitation); sodium benzoate (alternative N excretion); liver transplant (curative CTLN2)",
        },
        "severity": sev,
        "disease_category_breakdown": disease_cat,
        "gene_stats": gene_stats,
        "kpis": kpis,
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    genes_out = []
    for gd in HEPATIC_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        n = len(gpts)

        liver_pct = round(100 * sum(1 for p in gpts if p["liver_disease"]) / n, 1)
        neuro_pct = round(100 * sum(1 for p in gpts if p["neuro_involvement"]) / n, 1)
        kf_pct = round(100 * sum(1 for p in gpts if p["kf_rings"]) / n, 1)
        haemo_pct = round(100 * sum(1 for p in gpts if p["haemolytic_anaemia"]) / n, 1)
        psych_pct = round(100 * sum(1 for p in gpts if p["psychiatric_involvement"]) / n, 1)
        renal_pct = round(100 * sum(1 for p in gpts if p["renal_involvement"]) / n, 1)
        drug_err_pct = round(100 * sum(1 for p in gpts if p["drug_error"]) / n, 1)
        delay_pct = round(100 * sum(1 for p in gpts if p["diagnosis_delayed"]) / n, 1)
        transplant_pct = round(100 * sum(1 for p in gpts if p["received_transplant"]) / n, 1)
        surveillance_pct = round(100 * sum(1 for p in gpts if p["surveillance_adherent"]) / n, 1)
        mean_age_dx = round(sum(p["age_at_dx_years"] for p in gpts) / n, 1)

        genes_out.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "gene_class": gd["gene_class"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "phenotype": gd["phenotype"],
            "disease": gd["disease"],
            "inheritance": gd["inheritance"],
            "hallmark": gd["hallmark"],
            "key_ddx": gd["key_ddx"],
            "treatment_alert": gd["treatment_alert"],
            "primary_complication": gd["primary_complication"],
            "liver_biochem": gd["liver_biochem"],
            "ggt_pattern": gd["ggt_pattern"],
            "seed": gd["seed"],
            "cohort_n": n,
            "mean_age_at_dx_years": mean_age_dx,
            "liver_disease_pct": liver_pct,
            "neuro_involvement_pct": neuro_pct,
            "kf_rings_pct": kf_pct,
            "haemolytic_anaemia_pct": haemo_pct,
            "psychiatric_involvement_pct": psych_pct,
            "renal_involvement_pct": renal_pct,
            "drug_error_pct": drug_err_pct,
            "diagnosis_delayed_pct": delay_pct,
            "transplant_pct": transplant_pct,
            "surveillance_adherent_pct": surveillance_pct,
            "severity_weights": gd["severity_weights"],
        })

    return {"genes": genes_out}


def get_definitions() -> list:
    return [
        {
            "term": "GGT Paradox in Cholestasis — HIGH vs LOW GGT as PFIC Classifier",
            "definition": (
                "Gamma-glutamyltransferase (GGT) is a glycoprotein enzyme expressed predominantly "
                "in biliary epithelial cells (cholangiocytes) and released into serum when bile "
                "flow carries GGT from the canalicular membrane into the biliary tree and then into serum. "
                "HIGH GGT in cholestasis: indicates BILE FLOW IS PRESENT (even if obstructed downstream) "
                "→ GGT is being released from cholangiocytes into bile → measured in serum. "
                "Examples: biliary atresia, primary sclerosing cholangitis (PSC), PFIC3 (ABCB4/MDR3). "
                "LOW GGT in cholestasis: indicates BILE FLOW IS INSUFFICIENT at the canalicular level "
                "→ GGT cannot leave cholangiocytes → artificially low. "
                "PFIC1 (ATP8B1) and PFIC2 (ABCB11): primary failure of bile acid export or "
                "phospholipid asymmetry → no effective bile flow → LOW GGT despite severe cholestasis. "
                "CLINICAL RULE: Low GGT + cholestasis in an infant = PFIC1 or PFIC2. "
                "High GGT + cholestasis in an infant/child = PFIC3, Alagille, or biliary atresia. "
                "This is the FIRST BRANCH POINT in the diagnostic algorithm for neonatal/infantile cholestasis "
                "AFTER biliary atresia is excluded by DISIDA scan + liver biopsy."
            ),
        },
        {
            "term": "Wilson's Disease Diagnostic Workup — Leipzig Score",
            "definition": (
                "Wilson's disease diagnosis uses the Leipzig scoring system (score ≥4 = diagnosis): "
                "Kayser-Fleischer rings: +2 (present) / 0 (absent). "
                "Serum ceruloplasmin: <0.1 g/L = +2; 0.1-0.2 g/L = +1; ≥0.2 g/L = 0. "
                "Neurological symptoms / psychiatric symptoms: +2 (if severe) / +1 (if mild). "
                "24h urine copper: >100 µg/day = +2; 40-100 µg/day (on normal diet) = +1. "
                "Liver copper (biopsy): >250 µg/g DW = +2; 50-250 = +1; Rhodanine-positive granules = +1. "
                "Coombs-negative haemolytic anaemia: +1. "
                "IMPORTANT CAVEATS: Ceruloplasmin is an acute-phase reactant — can be falsely normal "
                "in inflammation/infection; low ceruloplasmin alone is NOT diagnostic (Wilson's disease "
                "requires ≥4 on Leipzig score). KF rings absent in ~50% of purely hepatic Wilson's disease. "
                "24h urine copper >250 µg/day on penicillamine challenge confirms diagnosis if borderline. "
                "For ACUTE LIVER FAILURE with Wilson's disease: ratio of ALP (IU/L) to bilirubin "
                "(µmol/L) <4 is highly suggestive (normal ALP relative to severity of jaundice — "
                "because released copper inhibits ALP). Coombs-negative haemolytic anaemia + "
                "acute liver failure in a young patient = Wilson's disease until proven otherwise. "
                "ATP7B sequencing identifies >700 known mutations (most labs test hot-spot panel first)."
            ),
        },
        {
            "term": "Malate-Aspartate Shuttle (MAS) and Citrin Deficiency",
            "definition": (
                "The malate-aspartate shuttle (MAS) is the primary mechanism for transferring "
                "reducing equivalents (cytoplasmic NADH) into the mitochondrial matrix in hepatocytes: "
                "STEP 1: Cytoplasmic MDH (malate dehydrogenase) reduces OAA + NADH → malate + NAD+. "
                "STEP 2: Malate enters mitochondria via the malate-α-ketoglutarate carrier (SLC25A11). "
                "STEP 3: Mitochondrial MDH oxidises malate + NAD+ → OAA + NADH (inside mitochondria). "
                "STEP 4: OAA + glutamate → aspartate + α-ketoglutarate (transamination). "
                "STEP 5: ASPARTATE exits mitochondria via CITRIN (SLC25A13) in exchange for glutamate → "
                "this is the step blocked in citrin deficiency. "
                "STEP 6: Cytoplasmic aspartate → transamination back to OAA (closing the cycle). "
                "CITRIN DEFICIENCY CONSEQUENCE: Step 5 blocked → aspartate accumulates in mitochondria → "
                "cytoplasmic NADH cannot be cleared → cytoplasmic reducing stress → gluconeogenesis inhibited "
                "(PEPCK requires cytoplasmic NAD+) → carbohydrate metabolism worsens the blockade "
                "(glucose → pyruvate generates even more NADH → more NADH backup → more urea cycle failure). "
                "HIGH PROTEIN/FAT diet helps: amino acids/fatty acids can be oxidised via pathways "
                "less dependent on MAS; ketone bodies provide energy without generating cytoplasmic NADH. "
                "ARGININE supplementation: bypasses the blocked ASS1 step (aspartate-limited) by "
                "providing arginine directly → ornithine cycle continues → ammonia detoxification."
            ),
        },
        {
            "term": "Hepcidin-Ferroportin Iron Axis and HFE Haemochromatosis",
            "definition": (
                "Iron homeostasis in the body is controlled by the HEPCIDIN-FERROPORTIN axis: "
                "FERROPORTIN-1 (FPN1/SLC40A1): the only known cellular iron exporter; expressed on "
                "duodenal enterocytes (exports dietary iron into blood), macrophages (exports recycled iron "
                "from RBC catabolism), and hepatocytes (exports stored iron). "
                "HEPCIDIN (HAMP): a 25-amino-acid peptide produced by hepatocytes in response to iron "
                "loading, inflammation (BMP-SMAD1/5/8 + IL-6-JAK-STAT3 pathways) → binds ferroportin "
                "→ ubiquitin-mediated ferroportin internalisation and degradation → reduces iron export "
                "→ reduces serum iron → reduces iron absorption. "
                "HFE MECHANISM: HFE protein senses transferrin saturation (HFE/TfR1 complex) → "
                "at high TfSat, HFE dissociates from TfR1 → joins BMP co-receptor complex "
                "(HJV + BMPR1/2 + ALK2) → activates BMP-SMAD → hepcidin transcription. "
                "HFE LOF (C282Y/C282Y): HFE cannot reach cell surface (disulphide bond disrupted) → "
                "HFE-TfR1 complex cannot form → BMP-SMAD signalling blunted → inappropriately low "
                "hepcidin → unchecked ferroportin activity → maximum iron absorption/export → "
                "progressive systemic iron overload. "
                "IRON DEPOSITION ORDER: liver (periportal zone 1) → heart → pancreas → pituitary → joints. "
                "PHLEBOTOMY MECHANISM: removes iron-loaded red blood cells → haematopoiesis regenerates "
                "using stored iron → hepatic iron stores depleted over months; maintains iron depletion "
                "by creating a continuous iron demand (new erythropoiesis)."
            ),
        },
        {
            "term": "Alpha-1-Antitrypsin (A1AT) — Z-Polymer Mechanism",
            "definition": (
                "Alpha-1-antitrypsin (A1AT, encoded by SERPINA1) is a 52-kDa serine protease inhibitor "
                "synthesised by hepatocytes and secreted into plasma. It inhibits neutrophil elastase (NE) "
                "by a SUICIDE SUBSTRATE mechanism: the RCL (reactive centre loop) of A1AT inserts into "
                "the NE active site as a substrate → NE cleaves the Met358-Ser359 bond → A1AT undergoes "
                "irreversible conformational change trapping NE → stable irreversible inhibitory complex. "
                "Z ALLELE MECHANISM: p.Glu342Lys disrupts a critical salt bridge (Glu342–Lys290) that "
                "stabilises the hydrophobic core of the serpin → Z-A1AT is structurally unstable → "
                "in the hepatocyte ER, the polymerisation-prone β-sheet A is exposed → "
                "Z-A1AT molecules self-polymerise (RCL of one Z-A1AT inserts into the β-sheet A "
                "of another — domain-swapping polymer) → polymers retained in hepatocyte ER → "
                "ER stress → unfolded protein response (UPR) → hepatocyte apoptosis → liver disease. "
                "Only ~10-15% of Z-A1AT reaches plasma (the rest retained as polymers). "
                "LUNG: the small amount of Z-A1AT that reaches the lung is functionally impaired "
                "(partially polymerised, reduced NE inhibitory capacity) → unopposed NE → "
                "progressive elastin destruction → panacinar emphysema (lower zones — lowest lung is "
                "highest NE concentration from gravity-dependent macrophage distribution). "
                "NULL ALLELES: no protein produced → lung disease but NO liver disease "
                "(no polymer to accumulate in hepatocytes). Key clinical distinction."
            ),
        },
        {
            "term": "PFIC Triad — PFIC1 vs PFIC2 vs PFIC3 Differential Diagnosis Table",
            "definition": (
                "PROGRESSIVE FAMILIAL INTRAHEPATIC CHOLESTASIS (PFIC) — three main genetic types: "
                "PFIC1 (ATP8B1 / FIC1 / Byler disease): LOW GGT; inheritance AR; protein = aminophospholipid flippase; "
                "extrahepatic features = diarrhoea, sensorineural hearing loss (SNHL), pancreatitis; "
                "BSEP expression on IHC = PRESENT (but reduced function secondary to FXR suppression); "
                "post-transplant = diarrhoea WORSENS; HCC risk = LOW (less than PFIC2); "
                "treatment = odevixibat, biliary diversion, liver transplant. "
                "PFIC2 (ABCB11 / BSEP deficiency): LOW GGT; inheritance AR; protein = bile salt export pump; "
                "extrahepatic features = NONE; BSEP expression on IHC = ABSENT (definitive test); "
                "post-transplant = excellent (full correction); HCC risk = HIGH (even in childhood); "
                "treatment = odevixibat FDA 2021, maralixibat, liver transplant. "
                "PFIC3 (ABCB4 / MDR3 / phosphatidylcholine flippase): HIGH GGT (cholangiocyte damage); "
                "inheritance AR; protein = PC flippase; cholelithiasis from infancy; "
                "cholangiopathy/sclerosing pattern; extrahepatic = NONE; BSEP expression = PRESENT; "
                "HCC risk = MODERATE; treatment = UDCA (effective if partial function), liver transplant. "
                "KEY RULE: GGT measurement is the FIRST BRANCH POINT. "
                "LOW GGT cholestasis in infant → PFIC1 vs PFIC2 → differentiate by BSEP IHC + extrahepatic features. "
                "HIGH GGT cholestasis in infant → PFIC3 vs Alagille vs biliary atresia → "
                "liver biopsy (duct paucity = Alagille; bile duct present = PFIC3 or other)."
            ),
        },
        {
            "term": "Alagille Syndrome — Notch Signalling and Five Cardinal Features",
            "definition": (
                "Alagille syndrome (ALGS) is caused by haploinsufficiency of JAG1 (Jagged-1) — "
                "a ligand for Notch receptors 1 and 2. JAG1 binding to Notch on adjacent cells "
                "triggers γ-secretase-mediated cleavage of Notch → NICD (Notch intracellular domain) "
                "release → nuclear NICD/RBPJ-κ complex → target gene transcription. "
                "FIVE CARDINAL FEATURES of ALGS (≥3 of 5 + cholestasis = clinical diagnosis without "
                "genetic testing; ≥1 + genotype positive = molecular diagnosis): "
                "(1) CHOLESTASIS: paucity of interlobular bile ducts (<0.5 bile ducts per portal tract "
                "on liver biopsy); driven by JAG1-Notch2 requirement for bile duct morphogenesis. "
                "(2) CARDIAC: peripheral pulmonary stenosis (PPS) 70% + other CHD; JAG1-Notch1 "
                "required for cardiac outflow tract development. "
                "(3) SKELETAL: butterfly vertebrae (70%); incomplete posterior vertebral arch formation; "
                "JAG1-Notch in somite segmentation. "
                "(4) OCULAR: posterior embryotoxon (78-90%); anteriorly displaced Schwalbe's line; "
                "slit-lamp examination; present in 8-15% of general population (low specificity alone). "
                "(5) FACIAL: broad forehead, deep-set eyes, pointed chin, bulbous nose tip; "
                "more pronounced with age. "
                "VASCULAR ANOMALIES (not a cardinal feature but major mortality cause 34%): "
                "intracranial vascular anomalies (ICVA — aneurysms, Moyamoya); renal artery stenosis; "
                "abdominal aortic coarctation. MRI brain + MRA mandatory at diagnosis. "
                "VARIABLE EXPRESSIVITY: same mutation → minimal features in parent, severe phenotype in child; "
                "50% de novo. NOTCH2 mutations → ALGS2 (rarer; more renal anomalies; otherwise similar)."
            ),
        },
        {
            "term": "Kayser-Fleischer (KF) Rings — Pathophysiology and Diagnostic Significance",
            "definition": (
                "Kayser-Fleischer (KF) rings are golden-brown or grey-green deposits of copper "
                "in the peripheral cornea, located in the posterior layers of the corneal stroma "
                "at the level of Descemet's membrane (the posterior limiting membrane of the cornea). "
                "PATHOPHYSIOLOGY: in Wilson's disease, excess copper overflows from the liver into "
                "plasma → copper deposits in Descemet's membrane preferentially at the superior pole "
                "first (KF rings begin superiorly → inferior pole → circumferential completion as "
                "copper accumulates over months-years). "
                "APPEARANCE: brownish-green ring at the corneal periphery (limbus), 1-3 mm wide; "
                "visible on slit-lamp examination (required for detection — usually not visible to the "
                "naked eye except in advanced cases). "
                "PREVALENCE IN WILSON'S DISEASE: 95%+ in neuropsychiatric WD (virtually universal); "
                "~50% in hepatic-only WD (copper may not yet have overflowed to cornea). "
                "DIAGNOSTIC SIGNIFICANCE: KF rings are PATHOGNOMONIC for Wilson's disease in the "
                "context of liver disease or neuropsychiatric symptoms — virtually no other condition "
                "produces this finding (except advanced primary biliary cholangitis/PBC — but clinical "
                "context is entirely different). "
                "ABSENCE: KF rings absent does NOT exclude Wilson's disease, especially hepatic presentations "
                "in children (KF rings take time to develop). "
                "FOLLOW-UP: KF rings regress with effective copper chelation therapy — ring disappearance "
                "confirms therapeutic response (monitored by slit-lamp every 6-12 months). "
                "OTHER CAUSES OF CORNEAL DEPOSITS: Arcus senilis (lipid, elderly, not copper; "
                "circumferential white, not at Descemet); Fleischer rings in keratoconus (iron in "
                "epithelium, not copper, green-brown at base of cone)."
            ),
        },
    ]
