#!/usr/bin/env python3
"""Hereditary-Liver-Disease-Atlas — Complete 8-Gene Hereditary Liver Disease Atlas
ATP7B   (Wilson disease copper-transporting ATPase; 1465 aa; 13q14.3; AR;
         Wilson disease — copper accumulation in liver/brain/cornea;
         KF rings PATHOGNOMONIC; D-Penicillamine / Trientine / Zinc;
         seed SEED_BASE+0) ·
HFE     (HFE protein; 343 aa; 6p21.3; AR;
         Hereditary Hemochromatosis — iron overload;
         C282Y p.Cys282Tyr 95% Northern European; therapeutic phlebotomy CURATIVE;
         seed SEED_BASE+1) ·
SERPINA1 (Alpha-1 antitrypsin; 418 aa; 14q32.13; AR;
          Alpha-1 Antitrypsin Deficiency — liver + lung disease;
          PiZZ genotype, Z allele p.Glu342Lys; augmentation therapy;
          seed SEED_BASE+2) ·
JAG1    (Jagged-1; 1218 aa; 20p12.2; AD;
         Alagille Syndrome — bile duct paucity;
         butterfly vertebrae PATHOGNOMONIC; cardiac defects (PA/TOF);
         seed SEED_BASE+3) ·
ATP8B1  (Aminophospholipid translocase FIC1; 1251 aa; 18q21.31; AR;
         PFIC1 / Byler Disease — low-GGT cholestasis;
         extrahepatic: diarrhoea / growth failure / hearing loss;
         seed SEED_BASE+4) ·
ABCB11  (Bile Salt Export Pump BSEP; 1321 aa; 2q31.1; AR;
         PFIC2 / BSEP Deficiency — low-GGT; highest OC/ICP pregnancy risk;
         HCC risk even without cirrhosis; maralixibat / odevixibat;
         seed SEED_BASE+5) ·
ABCB4   (Multidrug resistance protein 3 MDR3; 1279 aa; 7q21.12; AR/AD;
         PFIC3 / MDR3 Deficiency — HIGH-GGT distinguishes from PFIC1/2;
         phosphatidylcholine transport; UDCA partially effective;
         seed SEED_BASE+6) ·
ALDOB   (Aldolase B; 364 aa; 9q31.1; AR;
         Hereditary Fructose Intolerance — fructose avoidance CURATIVE;
         A149P 67% European founder; hypoglycaemia + lactic acidosis;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1582–1589)
"""

import random

SEED_BASE = 1582

LIVER_GENES = [
    # ── ATP7B — Wilson Disease ─────────────────────────────────────────
    {
        "gene": "ATP7B",
        "protein": "ATP7B — Wilson Disease, KF Rings PATHOGNOMONIC, D-Penicillamine/Trientine/Zinc",
        "alias": (
            "ATP7B; OMIM gene 606882; Wilson Disease OMIM 277900; 13q14.3; 1465 aa; ~165 kDa; "
            "AR; prevalence 1:30,000; carrier frequency 1:90. "
            "ATP7B is a copper-transporting P-type ATPase (Cu-ATPase) in hepatocytes: "
            "normally exports copper into bile (via canalicular membrane) and incorporates copper "
            "into ceruloplasmin for secretion. Without functional ATP7B: copper accumulates "
            "sequentially in liver → brain → cornea → kidney → other organs. "
            "HEPATIC PRESENTATION (children/young adults): acute liver failure (Coombs-negative "
            "haemolytic anaemia + rapidly progressive liver disease = SUSPECT WILSON'S); "
            "chronic hepatitis (often misdiagnosed as autoimmune hepatitis); cirrhosis. "
            "NEUROPSYCHIATRIC PRESENTATION (late teens–30s): dysarthria (MOST COMMON neuro sign); "
            "dystonia (limb, oromandibular); tremor (wing-beating); Parkinsonism; ataxia; "
            "personality change / behavioural disorder / frank psychosis. "
            "KAYSER-FLEISCHER (KF) RINGS: golden-brown copper deposits in Descemet membrane of "
            "cornea — PATHOGNOMONIC when neuro Wilson's present; detected by slit-lamp; "
            "absent in ~50% of hepatic-only Wilson's. "
            "DIAGNOSIS: serum ceruloplasmin <20 mg/dL (low); 24h urinary copper >100 µg/day "
            "(>40 µg/day in symptomatic); liver copper >250 µg/g dry weight; Wilson gene score ≥4. "
            "SERUM COPPER: PARADOXICALLY LOW (ceruloplasmin carries 90% serum copper; "
            "non-ceruloplasmin 'free' copper is ELEVATED). "
            "TREATMENT: chelation (D-Penicillamine or trientine) for symptomatic patients; "
            "zinc (blocks intestinal copper absorption) for maintenance / pre-symptomatic / pregnancy; "
            "tetrathiomolybdate (TTM) — rapid neurological copper removal; "
            "liver transplant for fulminant/refractory liver failure (corrects the underlying defect). "
            "MONITORING: urinary copper (target 200–500 µg/day on chelation; <100 µg/day maintenance); "
            "LFTs, ceruloplasmin; neuropsychiatric assessment; KF ring regression on treatment. "
            "PREGNANCY: zinc is SAFE; D-Penicillamine teratogenic → switch to trientine/zinc; "
            "NEVER STOP treatment during pregnancy (hepatic deterioration risk)."
        ),
        "aa": "1465 aa",
        "kDa": "~165 kDa",
        "locus": "13q14.3",
        "omim_gene": 606882,
        "omim_disease": 277900,
        "inheritance": "AR; >500 mutations; H1069Q (Northern European founder, 35–40%); R778L (East Asian, 35–40%)",
        "gene_class": (
            "ATP7B encodes a copper-transporting P-type ATPase (type P1B-2). "
            "Domain structure: 6 N-terminal copper-binding domains (MBD1–6, GMXCXXC motifs) → "
            "transmembrane domain (8 TM helices, CPC motif in TM6 = copper-binding signature) → "
            "P-domain (phosphorylation Asp1027, DKTGT consensus) → A-domain (actuator) → N-domain (ATP binding). "
            "Function: copper import into trans-Golgi lumen (ceruloplasmin loading) and biliary copper export. "
            "Mutation spectrum: H1069Q missense (most common European — impairs P-domain ATPase activity); "
            "R778L (East Asian — TM domain, severe); >500 pathogenic variants catalogued; "
            "most patients compound heterozygous. "
            "KF ring formation: non-ceruloplasmin copper deposits in Descemet membrane via aqueous humor."
        ),
        "n_patients": 40,
        "key_alerts": [
            "ATP7B-KF-RINGS-PATHOGNOMONIC: Kayser-Fleischer rings (golden-brown corneal copper deposits) are PATHOGNOMONIC for neurological Wilson's disease — always request SLIT-LAMP examination; absent in ~50% of hepatic-only presentation but obligatory to check",
            "ATP7B-HAEMOLYTIC-ANAEMIA-LIVER-FAILURE: Coombs-negative haemolytic anaemia + acute liver failure in a young patient = SUSPECT WILSON'S until proven otherwise; ANA/SMA can be positive (mimics AIH); serum copper paradoxically LOW (ceruloplasmin-bound copper falls)",
            "ATP7B-SERUM-COPPER-LOW-PARADOX: Serum TOTAL copper is paradoxically LOW in symptomatic Wilson's (ceruloplasmin carries 90% of serum copper and is reduced); measure NON-CERULOPLASMIN FREE copper (elevated); or use 24h urinary copper > 100 µg/day",
            "ATP7B-TREATMENT-NEVER-STOP: NEVER discontinue copper chelation / zinc therapy — acute hepatic decompensation can occur within weeks of stopping; particularly dangerous in pregnancy (switch to zinc/trientine, do NOT stop treatment)",
            "ATP7B-NEUROLOGICAL-PENICILLAMINE-PARADOX: D-Penicillamine can WORSEN neurological symptoms initially (copper mobilisation paradox) — consider trientine as first-line for neurological Wilson's; start low, titrate slowly; neurological deterioration in first months ≠ treatment failure",
            "ATP7B-PREGNANCY-ZINC-SAFE: Zinc acetate/gluconate is SAFE in pregnancy (blocks intestinal copper absorption without teratogenicity); D-Penicillamine is teratogenic (pyridoxine antagonism, connective tissue effects); trientine is probably safe but switch to zinc preferred",
            "ATP7B-TRANSPLANT-CURATIVE: Liver transplantation corrects the metabolic defect in Wilson's (normal donor ATP7B) — indicated for fulminant hepatic failure or decompensated cirrhosis unresponsive to chelation; neurological disease may improve post-transplant but reversal incomplete",
            "ATP7B-URINARY-COPPER-MONITORING: Target urinary copper 200–500 µg/day on chelation (adequate chelation); <100 µg/day on zinc maintenance; >500 µg/day may indicate over-chelation (dose reduce); non-ceruloplasmin free copper <150 µg/L = adequate control",
        ],
        "etiologies": {
            "H1069Q missense (Northern European founder)": 14,
            "R778L missense (East Asian)": 8,
            "Compound heterozygous (2 different missense)": 10,
            "Frameshift/nonsense (severe, loss of function)": 5,
            "Splice site": 3,
        },
        "stats": {
            "hepatic_presentation_pct": 55,
            "neuropsychiatric_presentation_pct": 35,
            "mixed_presentation_pct": 10,
            "kf_rings_present_pct": 75,
            "on_chelation_pct": 70,
            "on_zinc_maintenance_pct": 25,
            "transplanted_pct": 5,
            "mean_dx_age": 18,
            "mean_dx_delay_months": 24,
        },
        "dx_delay_distribution": {"<6m": 10, "6-24m": 14, "24-60m": 10, ">60m": 6},
    },

    # ── HFE — Hereditary Hemochromatosis ──────────────────────────────
    {
        "gene": "HFE",
        "protein": "HFE — Hereditary Hemochromatosis, C282Y 95% Northern European, Therapeutic Phlebotomy CURATIVE",
        "alias": (
            "HFE; OMIM gene 613609; Hereditary Hemochromatosis (HH) type 1 OMIM 235200; 6p21.3; "
            "343 aa; ~38 kDa; AR (C282Y homozygous for clinical HH); prevalence C282Y/C282Y "
            "1:200–1:400 Northern European (highest prevalence of common AR disease in Western Europeans). "
            "HFE is an atypical MHC class I protein that modulates hepcidin signalling. "
            "Normal HFE interacts with transferrin receptor 1 (TfR1) + TfR2 → signals liver to "
            "produce HEPCIDIN → hepcidin degrades ferroportin → reduces intestinal iron absorption. "
            "Without functional HFE: hepcidin production is LOW → ferroportin overactive → "
            "excess iron absorption from gut → iron accumulates in liver (CIRRHOSIS), "
            "pancreas (DIABETES MELLITUS, 'bronze diabetes'), joints (ARTHROPATHY), "
            "heart (CARDIOMYOPATHY + conduction abnormalities), pituitary (HYPOGONADISM), "
            "skin (BRONZING). "
            "CLASSIC TRIAD: liver cirrhosis + diabetes mellitus + skin bronzing (historical). "
            "MUTATIONS: C282Y (p.Cys282Tyr, rs1800562): 85–95% of clinical HH in Northern Europeans "
            "— disrupts HFE-TfR1 binding; H63D (p.His63Asp, rs1799945): modifier allele, "
            "C282Y/H63D compound heterozygous causes mild-moderate iron loading but rarely cirrhosis. "
            "DIAGNOSIS: elevated transferrin saturation (TS) >45% fasting (FIRST ABNORMALITY); "
            "elevated serum ferritin (>200 µg/L female; >300 µg/L male for investigation); "
            "HFE genotyping for C282Y homozygosity; liver biopsy if ferritin >1000 or liver enzyme elevation. "
            "MRI liver iron quantification (non-invasive). "
            "TREATMENT: therapeutic phlebotomy 450–500 mL per session (removes ~200–250 mg iron); "
            "weekly until ferritin <50 µg/L → maintenance every 3 months. "
            "Iron chelation (deferoxamine/deferasirox) reserved for those unable to tolerate phlebotomy. "
            "PRE-CIRRHOTIC TREATMENT: prevents all end-organ damage — liver cancer, diabetes, arthropathy. "
            "POST-CIRRHOTIC: phlebotomy reduces HCC risk (4-fold still elevated but reduced with treatment). "
            "ALCOHOL is a MAJOR COFACTOR — multiplies HCC risk × 3 in C282Y homozygotes. "
            "FAMILY SCREENING MANDATORY: first-degree relatives → HFE genotyping."
        ),
        "aa": "343 aa",
        "kDa": "~38 kDa",
        "locus": "6p21.3",
        "omim_gene": 613609,
        "omim_disease": 235200,
        "inheritance": "AR; C282Y homozygous = clinical HH (incomplete penetrance ~30% males, <5% females); H63D modifier",
        "gene_class": (
            "HFE encodes an atypical HLA class I protein (lacks peptide-binding groove). "
            "Domain structure: signal peptide → α1-α2-α3 domains (MHC-like fold) → transmembrane → "
            "short cytoplasmic tail. Associates non-covalently with β2-microglobulin. "
            "C282Y (exon 4): breaks disulfide bond Cys260-Cys203 in α3 domain → "
            "prevents β2m association → misfolded HFE retained in ER, not surface-expressed → "
            "no HFE-TfR1 interaction → hepcidin not upregulated → iron loading. "
            "H63D (exon 2): in α1 domain, alters TfR1 binding partially. "
            "HFE interacts with TfR1 (competitive with transferrin — senses iron saturation) and "
            "with TfR2 in hepatocytes (iron-sensing complex → BMP signalling → SMAD → hepcidin gene)."
        ),
        "n_patients": 40,
        "key_alerts": [
            "HFE-TRANSFERRIN-SATURATION-FIRST: Elevated fasting transferrin saturation (>45%) is the FIRST ABNORMALITY in hereditary hemochromatosis — precedes ferritin elevation by years; screen first-degree relatives with fasting TS + HFE genotype (NOT ferritin alone)",
            "HFE-PHLEBOTOMY-CURATIVE-PRE-CIRRHOTIC: Therapeutic phlebotomy BEFORE cirrhosis develops is essentially CURATIVE — prevents liver disease, diabetes, cardiomyopathy, arthropathy; start immediately upon confirmed C282Y homozygosity with iron loading",
            "HFE-FERRITIN-POOR-ALONE: Serum ferritin is elevated in MANY conditions (inflammation, metabolic syndrome, alcohol, liver disease) — ferritin alone is NOT diagnostic; always confirm with transferrin saturation + HFE genotype; ferritin >1000 µg/L = liver biopsy",
            "HFE-ALCOHOL-COFACTOR: Alcohol consumption dramatically amplifies hepatocellular carcinoma risk in C282Y homozygotes (multiplicative risk ×3 vs alcohol alone) — STRICT alcohol abstinence is a mandatory co-intervention with phlebotomy",
            "HFE-HCC-POST-CIRRHOTIC: Established cirrhosis carries 4-fold elevated HCC risk despite treatment — 6-monthly liver ultrasound + AFP surveillance mandatory even after iron depletion in cirrhotic patients",
            "HFE-HYPOGONADISM-PITUITARY: Iron deposition in the pituitary causes hypogonadotrophic hypogonadism — erectile dysfunction, amenorrhoea, infertility; measure LH/FSH/testosterone before attributing to 'age'; pituitary iron → gonadotrophin deficiency responds poorly to chelation alone",
            "HFE-H63D-NOT-MONOGENIC: H63D homozygosity or C282Y/H63D compound heterozygosity rarely causes clinical hemochromatosis alone — requires additional cofactors (alcohol, metabolic syndrome); do NOT start phlebotomy for H63D alone without documented iron overload",
            "HFE-FAMILY-SCREEN-MANDATORY: ALL first-degree relatives of a C282Y homozygote must be offered HFE genotyping — screen proband's children (siblings have 25% risk, children have 50% carrier risk from one C282Y parent); pre-symptomatic treatment prevents all end-organ damage",
        ],
        "etiologies": {
            "C282Y homozygous (p.Cys282Tyr/p.Cys282Tyr)": 28,
            "C282Y/H63D compound heterozygous": 8,
            "C282Y heterozygous + other modifier": 2,
            "Other HFE mutation (S65C, etc.)": 2,
        },
        "stats": {
            "transferrin_sat_elevated_pct": 98,
            "cirrhosis_at_dx_pct": 20,
            "diabetes_at_dx_pct": 15,
            "arthropathy_pct": 40,
            "hypogonadism_pct": 25,
            "skin_bronzing_pct": 35,
            "on_phlebotomy_pct": 88,
            "mean_dx_age": 42,
            "mean_dx_delay_months": 60,
        },
        "dx_delay_distribution": {"<12m": 8, "12-36m": 10, "36-60m": 12, ">60m": 10},
    },

    # ── SERPINA1 — Alpha-1 Antitrypsin Deficiency ──────────────────────
    {
        "gene": "SERPINA1",
        "protein": "SERPINA1 — Alpha-1 Antitrypsin Deficiency (A1ATD), PiZZ Genotype, Augmentation Therapy",
        "alias": (
            "SERPINA1; OMIM gene 107400; A1ATD OMIM 613490; 14q32.13; 418 aa (mature protein); "
            "~52 kDa; AR (PiZZ for liver + lung disease); prevalence PiZZ 1:2000–1:5000 Northern European. "
            "Alpha-1 antitrypsin (A1AT) is the principal protease inhibitor in blood: "
            "inhibits neutrophil elastase (NE), proteinase 3, and cathepsin G in the lung alveoli. "
            "Without sufficient A1AT: unchecked NE destroys alveolar walls → EMPHYSEMA (panlobular, "
            "predominantly lower lobe — DIFFERENT from smoking-related upper-lobe emphysema). "
            "LIVER DISEASE: Z allele (p.Glu342Lys) causes A1AT to MISFOLD and POLYMERISE in the "
            "ER of hepatocytes → protein accumulates as PAS-positive, diastase-resistant globules "
            "(PAS-D staining on liver biopsy is DIAGNOSTIC) → hepatocyte toxicity → "
            "neonatal hepatitis, childhood liver disease, adult cirrhosis, HCC. "
            "LUNG DISEASE: insufficient A1AT secreted into circulation → low serum A1AT → NE unchecked. "
            "PHENOTYPE NOMENCLATURE: Pi (Protease Inhibitor) system; "
            "PiMM = normal; PiMZ = heterozygous carrier (intermediate A1AT, increased risk); "
            "PiZZ = most common disease genotype (A1AT ~15% normal = severe deficiency); "
            "PiSZ = compound heterozygous (intermediate severity). "
            "Z allele frequency: 4% Northern European (1:60 carrier). "
            "NULLS: PiNull-Null = no A1AT protein = pure lung disease without liver disease "
            "(no misfolded protein in ER = no hepatocyte damage). "
            "DIAGNOSIS: serum A1AT level <0.57 g/L (normal >1.0 g/L); phenotyping/genotyping; "
            "liver biopsy shows PAS-D globules. "
            "TREATMENT: liver disease — standard cirrhosis management; liver transplant for severe liver disease "
            "(corrects A1AT production to donor phenotype). "
            "Lung disease: smoking cessation (MANDATORY); inhaled bronchodilators/ICS; "
            "intravenous A1AT augmentation therapy (Prolastin/Aralast/Glassia, weekly IV) — "
            "slows emphysema progression in PiZZ with FEV1 30–65% predicted; "
            "does NOT treat liver disease. Fazirsiran (siRNA, A1AT-siRNA) — reduces hepatic Z-A1AT polymerisation."
        ),
        "aa": "418 aa",
        "kDa": "~52 kDa",
        "locus": "14q32.13",
        "omim_gene": 107400,
        "omim_disease": 613490,
        "inheritance": "AR for clinical disease; PiZZ most common; PiMZ carriers have intermediate risk",
        "gene_class": (
            "SERPINA1 encodes alpha-1 antitrypsin, a serine protease inhibitor (serpin). "
            "Structure: 3 β-sheets (A/B/C) + 9 α-helices + reactive centre loop (RCL, Met358-Ser359 "
            "= primary reactive site for neutrophil elastase). "
            "Inhibitory mechanism: 'suicide substrate' — NE attacks RCL at Met358-Ser359 → "
            "covalent acyl-enzyme intermediate → RCL inserts into β-sheet A → "
            "translocation + inactivation of NE + A1AT complex → cleared by macrophages. "
            "Z mutation (p.Glu342Lys, E342K): disrupts β-sheet A → "
            "RCL from adjacent molecule inserts into β-sheet A of next → LOOP-SHEET POLYMERISATION; "
            "polymers too large to be secreted from ER → accumulate in hepatocytes. "
            "S mutation (p.Glu264Val): milder polymerisation; SZ compound less severe than ZZ. "
            "Null mutations: premature stop / frameshift → no protein → pure lung disease."
        ),
        "n_patients": 40,
        "key_alerts": [
            "SERPINA1-PIDD-GLOBULES-PATHOGNOMONIC: PAS-positive, diastase-resistant (PAS-D) globules in periportal hepatocytes on liver biopsy are DIAGNOSTIC for Z-A1AT retention — always request PAS-D stain; absent in Null/Null (no retained protein = no globules but pure lung disease)",
            "SERPINA1-SMOKING-CATASTROPHIC: Smoking in PiZZ patients reduces A1AT half-life (oxidation of Met358 active site → inactivated) AND worsens emphysema dramatically — smoking cessation is the SINGLE MOST IMPORTANT intervention for lung preservation in A1ATD",
            "SERPINA1-AUGMENTATION-LUNG-ONLY: IV A1AT augmentation therapy (weekly) treats LUNG disease only — does NOT improve liver disease (augmentation adds functional A1AT to plasma, does not remove Z-A1AT polymers from hepatocytes); fazirsiran (siRNA) targets hepatic production",
            "SERPINA1-NULL-NULL-NO-LIVER: PiNull-Null genotype causes SEVERE lung disease (no circulating A1AT) but NO liver disease — no Z-A1AT polymers form in ER; liver biopsy shows NO PAS-D globules; distinguish from PiZZ for accurate counselling",
            "SERPINA1-LOWER-LOBE-EMPHYSEMA: A1ATD emphysema is PANLOBULAR, predominantly LOWER LOBE — the OPPOSITE of smoking-related emphysema (upper lobe, centrilobular); lower-lobe predominance on CT should prompt A1AT testing even in non-smokers",
            "SERPINA1-NEONATAL-HEPATITIS: PiZZ infants can present with neonatal cholestatic hepatitis (jaundice, elevated GGT/ALT in first months) — most improve spontaneously but 2–5% develop progressive liver disease; monitor LFTs every 6–12 months in all PiZZ children",
            "SERPINA1-HCC-CIRRHOSIS: HCC risk in PiZZ cirrhosis is elevated — 6-monthly surveillance ultrasound + AFP mandatory; liver transplantation corrects both A1AT deficiency AND liver disease (new liver makes MType A1AT)",
            "SERPINA1-FAMILY-MZ-RISK: PiMZ heterozygotes have intermediate A1AT (~55% normal) — increased COPD risk (especially with smoking) and 2-3× elevated liver disease risk; counsel MZ carriers to AVOID SMOKING and ALCOHOL; not currently an indication for augmentation therapy",
        ],
        "etiologies": {
            "PiZZ homozygous (p.Glu342Lys)": 28,
            "PiSZ compound heterozygous": 8,
            "PiZNull compound heterozygous": 2,
            "Other rare deficiency alleles": 2,
        },
        "stats": {
            "liver_disease_pct": 60,
            "lung_disease_pct": 70,
            "combined_liver_lung_pct": 40,
            "cirrhosis_at_dx_pct": 18,
            "emphysema_at_dx_pct": 45,
            "on_augmentation_pct": 30,
            "smoking_history_pct": 55,
            "mean_dx_age": 35,
            "mean_dx_delay_months": 72,
        },
        "dx_delay_distribution": {"<12m": 5, "12-36m": 8, "36-60m": 12, ">60m": 15},
    },

    # ── JAG1 — Alagille Syndrome ───────────────────────────────────────
    {
        "gene": "JAG1",
        "protein": "JAG1 — Alagille Syndrome, Butterfly Vertebrae PATHOGNOMONIC, Bile Duct Paucity, Cardiac",
        "alias": (
            "JAG1; OMIM gene 601920; Alagille Syndrome 1 OMIM 118450; 20p12.2; 1218 aa; ~134 kDa; "
            "AD (haploinsufficiency); prevalence 1:30,000–1:50,000. "
            "JAG1 is a Notch ligand critical for hepatobiliary, cardiac, skeletal, ocular, and "
            "renal development. Haploinsufficiency → reduced Notch signalling → "
            "INTRAHEPATIC BILE DUCT PAUCITY (< 0.9 bile ducts/portal tract on liver biopsy — "
            "key histological criterion). "
            "DIAGNOSTIC CRITERIA (Alagille): liver + at least 3 of 5 features: "
            "(1) BUTTERFLY VERTEBRAE (posterior arch fusion defect, T2–T10) — PATHOGNOMONIC on X-ray; "
            "(2) EYE: posterior embryotoxon (prominent Schwalbe line, slit-lamp) — most common ocular feature; "
            "(3) HEART: peripheral pulmonary artery stenosis (PPAS) MOST COMMON; "
            "tetralogy of Fallot (TOF) in 7–8%; pulmonary atresia; "
            "(4) DISTINCTIVE FACIES: broad forehead, deep-set eyes, pointed chin, saddle/tubular nose; "
            "(5) KIDNEY: renal tubular acidosis, vesicoureteral reflux, renal dysplasia. "
            "HEPATIC DISEASE: neonatal cholestasis (conjugated hyperbilirubinaemia, high GGT); "
            "pruritus (SEVERE — biliary); progressive chronic liver disease → cirrhosis in 15–25%; "
            "liver transplant in ~20% by adulthood. "
            "INTRACRANIAL HAEMORRHAGE: 15% lifetime risk (ICH) from intracranial vascular malformations "
            "(Moyamoya-pattern) — cerebrovascular malformation MUST be considered in any Alagille child "
            "with neurological symptoms. "
            "TREATMENT: pruritus — bile acid sequestrants (cholestyramine), UDCA, rifampicin; "
            "maralixibat (ileal bile acid transporter inhibitor) FDA 2021 for cholestatic pruritus in Alagille; "
            "liver transplant for end-stage liver disease. "
            "MUTATION: 94% JAG1 mutations (intragenic); 6% deletions at 20p12 (del20p12 — detected by FISH). "
            "VARIABLE EXPRESSIVITY — same family members can range from asymptomatic to liver failure."
        ),
        "aa": "1218 aa",
        "kDa": "~134 kDa",
        "locus": "20p12.2",
        "omim_gene": 601920,
        "omim_disease": 118450,
        "inheritance": "AD haploinsufficiency; 50–70% de novo; variable expressivity; NOTCH2 mutations = Alagille 2 (10%)",
        "gene_class": (
            "JAG1 encodes Jagged-1, a transmembrane Notch ligand. "
            "Domain structure: signal peptide → DSL domain (Delta/Serrate/Lag-2, critical for Notch binding) → "
            "16 EGF-like repeats → cysteine-rich domain → transmembrane → cytoplasmic tail. "
            "Notch signalling: JAG1 on signal cell → cleaves NICD (Notch intracellular domain) of "
            "Notch1/2 receptor → NICD translocates to nucleus → RBPJ-dependent transcription → "
            "downstream targets (HES/HEY family genes). "
            "During hepatobiliary development: JAG1-Notch2 signalling is required for "
            "intrahepatic bile duct morphogenesis (DUCTAL PLATE REMODELLING); "
            "haploinsufficiency reduces Notch2 activation → duct paucity. "
            "Mutation spectrum: ~50% truncating (nonsense/frameshift, haploinsufficiency); "
            "~50% missense (dominant negative or haploinsufficiency); ~7% exon deletions. "
            "Genotype-phenotype correlation is POOR — identical mutations → very different severity."
        ),
        "n_patients": 40,
        "key_alerts": [
            "JAG1-BUTTERFLY-VERTEBRAE-PATHOGNOMONIC: Butterfly vertebrae (posterior arch defect, T2–T10) on spinal X-ray are PATHOGNOMONIC for Alagille syndrome — check spine X-ray in any child with neonatal cholestasis + cardiac defect; may be absent in mild cases",
            "JAG1-ICH-VASCULAR-MALFORMATIONS: Intracranial haemorrhage occurs in 15% lifetime (vascular malformations, Moyamoya-pattern) — any neurological symptom (headache, altered consciousness, focal deficit) in an Alagille patient needs URGENT brain imaging",
            "JAG1-POSTERIOR-EMBRYOTOXON: Posterior embryotoxon (prominent Schwalbe ring on slit-lamp) is the most COMMON ocular feature (>90%) but NOT specific; butterfly vertebrae + posterior embryotoxon together in a cholestatic infant = high diagnostic probability",
            "JAG1-PPAS-CARDIAC-SCREEN: Peripheral pulmonary artery stenosis is the most common cardiac lesion — echocardiogram mandatory in all suspected Alagille; TOF in 7–8%; complex cardiac disease is the leading cause of death in Alagille (not liver disease)",
            "JAG1-MARALIXIBAT-PRURITUS: Maralixibat (IBAT inhibitor, FDA 2021) reduces cholestatic pruritus in Alagille — blocks ileal bile acid reabsorption → reduces bile acid pool → relieves pruritus; first approved pharmacotherapy for Alagille-specific pruritus",
            "JAG1-VARIABLE-EXPRESSIVITY: Identical JAG1 mutations cause strikingly DIFFERENT severity within the same family (asymptomatic to neonatal liver failure) — always examine PARENTS of a proband; a mildly affected parent may have the same pathogenic variant",
            "JAG1-LIVER-TRANSPLANT-20PCT: ~20% of Alagille patients require liver transplantation by adulthood — cardiac lesions (PPAS) MUST be evaluated and corrected BEFORE transplantation (post-transplant pulmonary hypertension risk); VAD/ECMO may be needed",
            "JAG1-RENAL-SURVEILLANCE: Renal dysplasia/tubular acidosis occurs in 40% — monitor creatinine, urine pH, and renal ultrasound; renal disease can progress independently of liver disease; double-organ transplant (liver + kidney) in combined failure",
        ],
        "etiologies": {
            "JAG1 frameshift/nonsense (haploinsufficiency)": 18,
            "JAG1 missense (DSL/EGF domain)": 14,
            "del20p12 (large deletion, FISH)": 5,
            "JAG1 splice site": 2,
            "NOTCH2 mutation (Alagille 2)": 1,
        },
        "stats": {
            "cholestasis_neonatal_pct": 75,
            "butterfly_vertebrae_pct": 85,
            "cardiac_lesion_pct": 90,
            "posterior_embryotoxon_pct": 92,
            "renal_disease_pct": 40,
            "ich_lifetime_pct": 15,
            "liver_transplant_pct": 20,
            "mean_dx_age_months": 3,
            "mean_dx_delay_months": 6,
        },
        "dx_delay_distribution": {"<3m": 20, "3-12m": 12, "12-36m": 5, ">36m": 3},
    },

    # ── ATP8B1 — PFIC1 / Byler Disease ────────────────────────────────
    {
        "gene": "ATP8B1",
        "protein": "ATP8B1 — PFIC1 (Byler Disease), Low-GGT Cholestasis, Extrahepatic Features KEY DDx",
        "alias": (
            "ATP8B1 (also FIC1); OMIM gene 602397; PFIC1 / Byler Disease OMIM 211600; "
            "18q21.31; 1251 aa; ~140 kDa; AR; prevalence 1:50,000–1:100,000. "
            "ATP8B1 encodes FIC1 (Familial Intrahepatic Cholestasis protein 1), a P4-type ATPase "
            "(flippase) that translocates phosphatidylserine and phosphatidylethanolamine from the "
            "outer to inner leaflet of the canalicular membrane, maintaining membrane asymmetry "
            "and protecting against bile acid-induced damage. "
            "Without FIC1: bile acid-induced damage to the canalicular membrane → "
            "failure of bile secretion → cholestasis. "
            "KEY CHARACTERISTIC: LOW GGT cholestasis — because cholestasis is not caused by "
            "loss of BSEP (bile salt export pump); serum GGT is paradoxically NORMAL/LOW "
            "(GGT reflects canalicular membrane damage/detergent effect; here bile acids are not "
            "retained at canalicular level as in PFIC2, but bile is not secreted). "
            "EXTRAHEPATIC FEATURES (PATHOGNOMONIC for PFIC1 vs PFIC2): "
            "(1) CHRONIC DIARRHOEA — FIC1 expressed in intestine; loss → bile acid malabsorption; "
            "(2) PANCREATITIS — FIC1 expressed in pancreatic acinar cells; "
            "(3) HEARING LOSS / SENSORINEURAL DEAFNESS — cochlear FIC1; "
            "(4) GROWTH FAILURE / SHORT STATURE — fat malabsorption. "
            "BENIGN RECURRENT INTRAHEPATIC CHOLESTASIS TYPE 1 (BRIC1): SAME GENE (ATP8B1) but "
            "milder mutations → episodic cholestasis (weeks-months) → complete resolution between episodes. "
            "PRURITUS: severe, often refractory; hepatic pruritus mechanism (bile acid accumulation). "
            "TREATMENT: ursodeoxycholic acid (UDCA — limited efficacy in PFIC1); "
            "partial external biliary diversion (PEBD) — diverts bile from ileal reabsorption → "
            "reduces intrahepatic bile acid load — effective in 50–70% PFIC1; "
            "nasobiliary drainage test predicts PEBD response; "
            "maralixibat / odevixibat (IBAT inhibitors) — reduce bile acid pool; "
            "liver transplant for end-stage disease BUT DIARRHOEA AND PANCREATITIS MAY PERSIST "
            "or even worsen post-transplant (extrahepatic FIC1 expression not corrected). "
            "LIVINGSTON SUMMER VARIANT: p.Gly308Val — common Old Order Amish founder mutation; "
            "most severe PFIC1; often named after the Byler family (Old Order Amish pedigree)."
        ),
        "aa": "1251 aa",
        "kDa": "~140 kDa",
        "locus": "18q21.31",
        "omim_gene": 602397,
        "omim_disease": 211600,
        "inheritance": "AR; Byler (p.Gly308Val) Amish founder; BRIC1 same gene milder alleles",
        "gene_class": (
            "ATP8B1 encodes a P4-type ATPase (flippase), FIC1 protein. "
            "Domain structure: N-terminal cytoplasmic tail → 10 transmembrane helices → "
            "P-domain (phosphorylation Asp454, DKTGT) → A-domain (actuator) → N-domain (ATP binding). "
            "Requires CDC50A as a beta-subunit for trafficking to canalicular membrane. "
            "Function: ATP-dependent translocation of phosphatidylserine (PS) and phosphatidylethanolamine (PE) "
            "from outer to inner leaflet of canalicular apical membrane → maintains lipid asymmetry → "
            "protects membrane from bile acid detergent attack. "
            "Expressed in liver (canaliculi), small intestine (apical), cochlea, pancreas. "
            "Mutation spectrum: missense most common; Gly308Val (severe, Byler), Ile661Thr (BRIC1, milder). "
            "Loss of FIC1 → defective PS flipping → bile acid-sensitive canalicular membrane → "
            "cholestasis and FXR downregulation (reduced BSEP expression as secondary effect)."
        ),
        "n_patients": 40,
        "key_alerts": [
            "ATP8B1-LOW-GGT-KEY-DDx: PFIC1 (ATP8B1) causes LOW-GGT cholestasis — serum GGT is NORMAL or LOW in PFIC1 and PFIC2; HIGH GGT distinguishes PFIC3 (ABCB4) and Alagille (JAG1) from PFIC1/2; LOW-GGT + conjugated hyperbilirubinaemia in an infant = PFIC1 or PFIC2",
            "ATP8B1-EXTRAHEPATIC-DIARRHOEA: Chronic diarrhoea is a SPECIFIC PFIC1 feature (not seen in PFIC2/3) — FIC1 expressed in intestinal brush border regulates bile acid absorption; diarrhoea may WORSEN after liver transplantation (extrahepatic FIC1 not corrected by liver transplant)",
            "ATP8B1-LIVER-TRANSPLANT-DIARRHOEA-WORSENS: After liver transplant in PFIC1, diarrhoea and pancreatitis may PERSIST OR WORSEN (intestinal FIC1 deficiency not corrected) and can cause post-transplant steatohepatitis (bile acid malabsorption → FXR-dependent FGF19 loss) — counsel families before transplant",
            "ATP8B1-BRIC1-SAME-GENE: Benign Recurrent Intrahepatic Cholestasis type 1 (BRIC1) is caused by MILDER ATP8B1 mutations — episodic cholestasis resolving completely; precipitated by pregnancy/infection; NO chronic liver disease; genotype-phenotype overlap with PFIC1",
            "ATP8B1-NASOBILIARY-DRAINAGE-TEST: Pre-surgical nasobiliary drainage test (7–10 days): if cholestasis resolves + pruritus improves = good predictor of response to partial external biliary diversion (PEBD); non-response = proceed to liver transplant evaluation",
            "ATP8B1-PANCREATITIS-EXTRAHEPATIC: Recurrent pancreatitis in PFIC1 is due to FIC1 deficiency in pancreatic acinar cells — NOT biliary obstruction; does NOT improve with biliary diversion or liver transplant; manage as recurrent idiopathic pancreatitis",
            "ATP8B1-ODEVIXIBAT-IBAT: Odevixibat (IBAT inhibitor, approved EU/EMA 2021, FDA PFIC 2023) reduces serum bile acids and pruritus in PFIC1/PFIC2 — blocks ileal bile acid transporter → reduces enteric bile acid reabsorption; trial before biliary diversion if available",
            "ATP8B1-HEARING-LOSS-SCREEN: Sensorineural hearing loss occurs in a subset of PFIC1 (cochlear FIC1 expression) — audiology testing mandatory in all PFIC1 patients; hearing loss may be progressive and require cochlear implantation",
        ],
        "etiologies": {
            "Gly308Val missense (Byler/Amish founder)": 12,
            "Other missense mutations": 16,
            "Frameshift/nonsense (severe)": 8,
            "Splice site": 2,
            "Mild alleles (BRIC1 spectrum)": 2,
        },
        "stats": {
            "neonatal_cholestasis_pct": 85,
            "low_ggt_pct": 95,
            "chronic_diarrhoea_pct": 65,
            "pancreatitis_pct": 25,
            "hearing_loss_pct": 20,
            "liver_transplant_pct": 35,
            "diarrhoea_post_transplant_pct": 70,
            "mean_dx_age_months": 4,
            "mean_dx_delay_months": 8,
        },
        "dx_delay_distribution": {"<6m": 18, "6-18m": 12, "18-36m": 6, ">36m": 4},
    },

    # ── ABCB11 — PFIC2 / BSEP Deficiency ──────────────────────────────
    {
        "gene": "ABCB11",
        "protein": "ABCB11 — PFIC2 (BSEP Deficiency), Low-GGT, HCC Risk Without Cirrhosis, ICP Pregnancy",
        "alias": (
            "ABCB11; OMIM gene 603201; PFIC2 OMIM 601847; 2q31.1; 1321 aa; ~146 kDa; AR; "
            "prevalence 1:50,000–1:100,000. "
            "ABCB11 encodes the Bile Salt Export Pump (BSEP), the primary ATP-dependent transporter "
            "for bile acid export from hepatocytes across the canalicular membrane into bile. "
            "Without BSEP: bile acids accumulate within hepatocytes → direct hepatotoxicity → "
            "severe progressive cholestasis (PFIC2). "
            "DISTINCTION FROM PFIC1: also LOW-GGT (like PFIC1) but "
            "NO EXTRAHEPATIC FEATURES (no diarrhoea, no pancreatitis, no hearing loss — "
            "BSEP is liver-restricted, unlike FIC1). "
            "SEVERE PRURITUS: bile acids accumulate → intractable itching. "
            "HEPATOCELLULAR CARCINOMA (HCC) WITHOUT CIRRHOSIS: PFIC2 is the only cholestatic liver "
            "disease where HCC/cholangiocarcinoma can develop in early childhood WITHOUT cirrhosis "
            "— SCREEN EVEN IN YOUNG CHILDREN (6-monthly ultrasound + AFP from age 1–2). "
            "INTRAHEPATIC CHOLESTASIS OF PREGNANCY (ICP): ABCB11 heterozygous carriers (PiSB phenotype) "
            "are at HIGH RISK for ICP — BSEP expression is downregulated by estrogen in pregnancy → "
            "borderline heterozygous BSEP function is insufficient → bile acid accumulation → ICP. "
            "BRIC2: same gene, milder mutations → benign recurrent intrahepatic cholestasis type 2 "
            "(episodic, non-progressive, often precipitated by pregnancy or drugs). "
            "DRUG-INDUCED LIVER INJURY (DILI): drugs that inhibit BSEP (cyclosporine, rifampicin, "
            "glibenclamide, troglitazone, bosentan) → transient or permanent cholestasis in "
            "BSEP heterozygotes. "
            "TREATMENT: UDCA (partial benefit); partial external biliary diversion (PEBD) "
            "effective in PFIC2 without truncating mutations (residual BSEP expression); "
            "maralixibat / odevixibat (IBAT inhibitors); "
            "liver transplantation for progressive disease — BSEP restoration; "
            "anti-BSEP antibodies post-transplant (5–10%): cause recurrent PFIC2 in transplant; "
            "treat with rituximab/plasmapheresis."
        ),
        "aa": "1321 aa",
        "kDa": "~146 kDa",
        "locus": "2q31.1",
        "omim_gene": 603201,
        "omim_disease": 601847,
        "inheritance": "AR for PFIC2; heterozygous = ICP risk + drug-BSEP inhibition risk; BRIC2 milder alleles",
        "gene_class": (
            "ABCB11 encodes BSEP (Bile Salt Export Pump), a member of the ATP-binding cassette (ABC) "
            "transporter subfamily B. "
            "Domain structure: TMD1 (6 TM helices) → NBD1 (nucleotide-binding domain, Walker A/B, LSGGQ) → "
            "TMD2 (6 TM helices) → NBD2. "
            "Function: primary active transport of bile acids from hepatocyte cytoplasm into "
            "canalicular lumen using ATP hydrolysis; handles taurocholate and glycocholate (main conjugated bile acids). "
            "Expression: liver-specific (unlike ATP8B1/FIC1) — explains absence of extrahepatic disease in PFIC2. "
            "Key mutations: E297G (European founder, residual BSEP activity — less severe); "
            "D482G (European founder, no residual activity — severe). "
            "Truncating mutations (nonsense, frameshift) → no BSEP protein → highest HCC risk. "
            "Estrogen-responsive element in ABCB11 promoter — estrogen downregulates BSEP expression "
            "→ mechanism for ICP in ABCB11 heterozygotes."
        ),
        "n_patients": 40,
        "key_alerts": [
            "ABCB11-HCC-WITHOUT-CIRRHOSIS: PFIC2 patients can develop HCC/cholangiocarcinoma in EARLY CHILDHOOD WITHOUT cirrhosis — UNIQUE among cholestatic liver diseases; START AFP + liver ultrasound surveillance from age 1–2 years; truncating mutations = highest HCC risk",
            "ABCB11-LOW-GGT-NO-EXTRAHEPATIC: PFIC2 = LOW GGT + NO extrahepatic features (vs PFIC1 = LOW GGT + diarrhoea/pancreatitis/hearing loss); BSEP is liver-restricted; absence of extrahepatic features = KEY distinguishing feature from PFIC1/ATP8B1",
            "ABCB11-ANTI-BSEP-ANTIBODIES-POST-TRANSPLANT: 5–10% of PFIC2 patients develop anti-BSEP alloantibodies after liver transplantation (new BSEP = foreign antigen) → recurrent cholestasis in the graft; treat with rituximab + plasmapheresis; screen post-transplant BSEP antibody titres",
            "ABCB11-ICP-HETEROZYGOUS: ABCB11 heterozygous carriers are at HIGH RISK for intrahepatic cholestasis of pregnancy (ICP) — oestrogen downregulates BSEP expression → borderline BSEP insufficient → pruritus + elevated bile acids → fetal risks (stillbirth >40 µmol/L); UDCA treatment",
            "ABCB11-DRUG-BSEP-INHIBITORS: Many drugs inhibit BSEP (cyclosporine, rifampicin, glibenclamide, bosentan, troglitazone) → drug-BSEP inhibition causes transient cholestasis especially in ABCB11 heterozygotes; check drug BSEP inhibition profile before prescribing in ABCB11 carriers",
            "ABCB11-BRIC2-SAME-GENE: BRIC2 (benign recurrent intrahepatic cholestasis type 2) = milder ABCB11 mutations (E297G, etc.); episodic self-resolving cholestasis; pregnancy-triggered; does not progress to cirrhosis but significant morbidity from recurrent episodes",
            "ABCB11-TRUNCATING-SEVERE: Truncating mutations (frameshift, nonsense) → no residual BSEP protein → most severe PFIC2 (earliest onset, highest HCC risk, least BSEP for immunological tolerance → higher anti-BSEP antibody risk post-transplant)",
            "ABCB11-ODEVIXIBAT-RESPONSE: Odevixibat (IBAT inhibitor) shows better response in PFIC2 patients with residual BSEP expression (missense mutations) vs truncating mutations (no BSEP for IBAT to synergise with); genotype predicts treatment response",
        ],
        "etiologies": {
            "E297G missense (European founder, residual activity)": 12,
            "D482G missense (European founder, no activity)": 8,
            "Other missense": 10,
            "Frameshift/nonsense (severe, HCC risk)": 8,
            "Splice site": 2,
        },
        "stats": {
            "low_ggt_pct": 95,
            "extrahepatic_features_pct": 2,
            "hcc_childhood_pct": 8,
            "icp_heterozygous_carriers_pct": 35,
            "anti_bsep_post_transplant_pct": 8,
            "liver_transplant_pct": 45,
            "bric2_spectrum_pct": 15,
            "mean_dx_age_months": 3,
            "mean_dx_delay_months": 5,
        },
        "dx_delay_distribution": {"<3m": 22, "3-12m": 12, "12-36m": 4, ">36m": 2},
    },

    # ── ABCB4 — PFIC3 / MDR3 Deficiency ───────────────────────────────
    {
        "gene": "ABCB4",
        "protein": "ABCB4 — PFIC3 (MDR3 Deficiency), HIGH-GGT KEY DDx, Phosphatidylcholine, UDCA Partial",
        "alias": (
            "ABCB4; OMIM gene 171060; PFIC3 OMIM 602347; 7q21.12; 1279 aa; ~140 kDa; AR (severe) / AD (mild); "
            "prevalence 1:100,000. "
            "ABCB4 encodes MDR3 (Multidrug Resistance protein 3), a phosphatidylcholine (PC) "
            "floppase on the canalicular membrane. "
            "Normal function: MDR3 translocates phosphatidylcholine from inner to outer leaflet of "
            "canalicular membrane → PC is extracted by bile acids into bile → forms mixed micelles "
            "that coat the biliary epithelium, protecting cholangiocytes from bile acid detergent damage. "
            "Without MDR3 (PC-deficient bile): bile acids are 'naked' without PC protection → "
            "bile acid-induced injury to bile duct epithelium (cholangiocytes) → "
            "INFLAMMATION, FIBROSIS of bile ducts + cholangiopathy. "
            "KEY DISTINCTION FROM PFIC1/PFIC2: HIGH SERUM GGT — because bile duct injury "
            "(not hepatocyte-canalicular injury) → GGT released from cholangiocytes. "
            "This HIGH-GGT is the CARDINAL DDx from PFIC1/2 (both LOW-GGT). "
            "CLINICAL SPECTRUM: "
            "(1) PFIC3: severe AR, neonatal/childhood cholestasis, portal hypertension, cirrhosis; "
            "(2) Low phospholipid-associated cholelithiasis (LPAC): AD milder heterozygous; "
            "young adults (<40 years), symptomatic gallstones / biliary sludge; INTRADUCTAL STONES; "
            "recurrent biliary colic + cholestatic symptoms; responds to UDCA. "
            "(3) Intrahepatic cholestasis of pregnancy (ICP): ABCB4 heterozygous + estrogen. "
            "(4) Drug-induced cholestasis (ABCB4 heterozygote + BSEP-inhibiting drugs). "
            "TREATMENT: UDCA (13–15 mg/kg/day) partially effective in PFIC3 (PC-deficient bile → "
            "UDCA displaces toxic bile acids); biliary diversion for progressive disease; "
            "liver transplant for end-stage; ABCB4-heterozygous LPAC responds very well to UDCA "
            "long-term (< 5% recurrence of gallstones on UDCA); UDCA should be continued lifelong in LPAC."
        ),
        "aa": "1279 aa",
        "kDa": "~140 kDa",
        "locus": "7q21.12",
        "omim_gene": 171060,
        "omim_disease": 602347,
        "inheritance": "AR for PFIC3 (biallelic); AD for LPAC (heterozygous sufficient); ICP risk in carriers",
        "gene_class": (
            "ABCB4 encodes MDR3 (P-glycoprotein family, ABCB subfamily). "
            "Domain structure: TMD1 (6 TM helices) → NBD1 (Walker A/B, LSGGQ) → "
            "TMD2 (6 TM helices) → NBD2. "
            "Function: floppase — translocates phosphatidylcholine (PC) from inner to outer leaflet "
            "of canalicular membrane using ATP hydrolysis; PC → extracted by bile acid micelles → "
            "forms protective PC-bile acid mixed micelles. "
            "Differs from ABCB11 (BSEP): ABCB11 transports bile ACIDS; ABCB4 transports bile PHOSPHOLIPIDS. "
            "Both work in tandem: bile = bile acid + PC mixed micelles = less toxic than free bile acids. "
            "PC-deficient bile = naked bile acids → cholangiocyte damage → GGT elevation. "
            "ABCB4 also expressed in liver but at lower level than ABCB11. "
            "Mutation spectrum: missense (PFIC3 biallelic, LPAC heterozygous); "
            "truncating (severe PFIC3 biallelic); common mutations: I541F, G536D."
        ),
        "n_patients": 40,
        "key_alerts": [
            "ABCB4-HIGH-GGT-DDx-PFIC1-PFIC2: PFIC3 (ABCB4) causes HIGH-GGT cholestasis — this is the CARDINAL distinguishing feature from PFIC1 (ATP8B1, low GGT) and PFIC2 (ABCB11, low GGT); HIGH GGT + cholestasis in an infant → think PFIC3, Alagille, choledochal cyst",
            "ABCB4-LPAC-YOUNG-GALLSTONES: Low phospholipid-associated cholelithiasis (LPAC) in ABCB4 heterozygotes — recurrent symptomatic gallstones or biliary sludge in patients <40 years, especially with intrahepatic stones; UDCA dramatically effective and should be continued lifelong",
            "ABCB4-UDCA-RESPONSE: UDCA (13–15 mg/kg/day) is PARTIALLY EFFECTIVE in PFIC3 (improves biochemistry, slows fibrosis if started early) and VERY EFFECTIVE in LPAC — displaces toxic bile acids from enterohepatic circulation; start UDCA as first-line in all ABCB4 disease",
            "ABCB4-ICP-PREGNANCY: ABCB4 heterozygous carriers are at HIGH RISK for intrahepatic cholestasis of pregnancy — oestrogen downregulates PC secretion → bile phospholipid-deficient → cholestasis; manage with UDCA + close bile acid monitoring; risk of fetal complications >40 µmol/L",
            "ABCB4-INTRADUCTAL-STONES: LPAC characteristically produces INTRAHEPATIC bile duct stones + biliary sludge (not just gallbladder stones) — ERCP may show bile duct stones; distinguish from primary sclerosing cholangitis; UDCA prevents recurrence",
            "ABCB4-CHOLANGIOCYTE-DAMAGE-GGT: MDR3 deficiency → PC-deficient bile → bile acid 'naked' detergent attack on biliary epithelium (cholangiocytes) → GGT released from cholangiocytes; this mechanism explains HIGH GGT in PFIC3 vs LOW GGT in PFIC1/2 (canalicular, not ductal, injury)",
            "ABCB4-BIALLELIC-SEVERE: Biallelic ABCB4 mutations → PFIC3 (severe, progressive) → portal hypertension, cirrhosis in childhood; monoallelic (heterozygous) → LPAC (adult-onset, milder); distinguish biallelic vs monoallelic by genotyping for management intensity",
            "ABCB4-LIVER-TRANSPLANT-CURATIVE: Liver transplantation in PFIC3 corrects the phospholipid secretion defect (donor MDR3 is functional) — no extrahepatic involvement unlike PFIC1; post-transplant disease does NOT recur (no anti-MDR3 antibodies, unlike PFIC2 anti-BSEP)",
        ],
        "etiologies": {
            "Biallelic missense PFIC3 (I541F, G536D, etc.)": 16,
            "Heterozygous missense LPAC spectrum": 14,
            "Frameshift/nonsense (severe PFIC3, biallelic)": 7,
            "Splice site (biallelic)": 2,
            "Heterozygous + ICP precipitated by pregnancy": 1,
        },
        "stats": {
            "high_ggt_pct": 95,
            "pfic3_biallelic_pct": 55,
            "lpac_heterozygous_pct": 35,
            "icp_in_carriers_pct": 25,
            "udca_biochemical_response_pct": 60,
            "liver_transplant_pct": 25,
            "intraductal_stones_pct": 40,
            "mean_dx_age": 8,
            "mean_dx_delay_months": 18,
        },
        "dx_delay_distribution": {"<12m": 12, "12-36m": 14, "36-60m": 8, ">60m": 6},
    },

    # ── ALDOB — Hereditary Fructose Intolerance ────────────────────────
    {
        "gene": "ALDOB",
        "protein": "ALDOB — Hereditary Fructose Intolerance (HFI), Fructose Avoidance CURATIVE, A149P European Founder",
        "alias": (
            "ALDOB; OMIM gene 612724; Hereditary Fructose Intolerance OMIM 229600; 9q31.1; "
            "364 aa; ~40 kDa; AR; prevalence 1:18,000–1:30,000 (Europe). "
            "ALDOB encodes Aldolase B, the hepatic isoform of aldolase that cleaves "
            "fructose-1-phosphate (F1P) to glyceraldehyde + DHAP in hepatocytes, kidneys, and small intestine. "
            "Without functional Aldolase B: fructose metabolism is blocked at F1P: "
            "— FRUCTOSE → FRUCTOSE-1-PHOSPHATE (by fructokinase, ATP consumed) → ALDOLASE B ABSENT "
            "→ F1P ACCUMULATES → (1) SEQUESTERS INORGANIC PHOSPHATE → "
            "PROFOUND HYPOGLYCAEMIA (phosphate depletion → impaired glycogenolysis + gluconeogenesis); "
            "(2) F1P is DIRECTLY TOXIC to hepatocytes (+ renal tubules). "
            "CLINICAL FEATURES: appear on FIRST INTRODUCTION OF FRUCTOSE (fruit, honey, sucrose, sorbitol): "
            "(1) ACUTE: nausea, vomiting, hypoglycaemia, seizures — within minutes; "
            "(2) CHRONIC fructose exposure: PROGRESSIVE LIVER DISEASE (hepatomegaly, cirrhosis); "
            "RENAL FANCONI SYNDROME (proximal tubular dysfunction → phosphaturia, bicarbonaturia, "
            "aminoaciduria, glucosuria); "
            "(3) PROTECTIVE AVERSION: patients develop INSTINCTIVE AVERSION to sweet foods — "
            "dietary self-selection means many adults with HFI have REMARKABLY GOOD DENTAL HEALTH "
            "(no caries — lifelong fructose/sucrose avoidance) and low body weight. "
            "DIAGNOSIS: urinary organic acids after fructose ingestion (fructose + sorbitol); "
            "LIVER ALDOLASE B ACTIVITY (<10%); ALDOB genotyping (avoids liver biopsy). "
            "DIFFERENTIAL: GALACTOSAEMIA (galactose, not fructose), GLYCOGEN STORAGE DISEASES "
            "(no fructose link). "
            "TREATMENT: STRICT FRUCTOSE AVOIDANCE (no fructose, sucrose, sorbitol) — "
            "this is ESSENTIALLY CURATIVE; liver disease REVERSES; Fanconi syndrome improves; "
            "READ FOOD LABELS: sucrose = glucose + fructose; sorbitol = sugar alcohol → converted to fructose; "
            "honey = 40% fructose. "
            "European mutations: A149P (p.Ala149Pro, 65–70% alleles); A174D (p.Ala174Asp, 15%); "
            "N334K (p.Asn334Lys, 5%). Compound heterozygous A149P/A174D = most common genotype in Europeans."
        ),
        "aa": "364 aa",
        "kDa": "~40 kDa",
        "locus": "9q31.1",
        "omim_gene": 612724,
        "omim_disease": 229600,
        "inheritance": "AR; A149P (65–70% European alleles) + A174D (15%) + N334K (5%) founder mutations",
        "gene_class": (
            "ALDOB encodes Aldolase B (liver/kidney/intestine isoform of fructose bisphosphate aldolase). "
            "Structure: homotetramer; each subunit 364 aa; TIM-barrel fold (β/α barrel). "
            "Active site: Lys229 (class I Schiff base aldolase; forms Schiff base with substrate); "
            "Lys146, Arg148, Lys203, Tyr363 — substrate binding. "
            "Reactions catalysed: (1) Fructose-1,6-bisphosphate → DHAP + glyceraldehyde-3-phosphate (glycolysis); "
            "(2) Fructose-1-phosphate → DHAP + glyceraldehyde (fructose metabolism). "
            "A149P mutation: in β-strand 4; disrupts folding → unstable subunit → impaired tetramerisation → "
            "reduced activity to <5% normal. A174D: α-helix 5; similar instability. "
            "F1P is a SUBSTRATE INHIBITOR for Aldolase A/C (glycolytic isoforms) — accumulating F1P "
            "inhibits glycolysis in all tissues → systemic phosphate depletion + hypoglycaemia."
        ),
        "n_patients": 40,
        "key_alerts": [
            "ALDOB-FRUCTOSE-AVOIDANCE-CURATIVE: Complete fructose/sucrose/sorbitol avoidance is ESSENTIALLY CURATIVE — liver disease reverses, Fanconi syndrome resolves, growth normalises; this is one of the few inborn errors of metabolism where strict dietary avoidance = cure",
            "ALDOB-HYPOGLYCAEMIA-MECHANISM-PHOSPHATE: Hypoglycaemia in HFI is caused by PHOSPHATE SEQUESTRATION (fructokinase converts fructose→F1P consuming ATP+Pi → F1P accumulates → inorganic phosphate depleted → glycogenolysis + gluconeogenesis impaired); NOT fasting hypoglycaemia",
            "ALDOB-SUCROSE-SORBITOL-HIDDEN: Sucrose (table sugar) = glucose + fructose (50% fructose) → TOXIC; sorbitol (sugar alcohol in 'sugar-free' products, pears, medications) → converted to fructose by sorbitol dehydrogenase → EQUALLY TOXIC; READ ALL LABELS including IV medications",
            "ALDOB-DENTAL-HEALTH-CLUE: HFI patients often have REMARKABLY GOOD DENTAL HEALTH (no caries) due to lifelong instinctive avoidance of sweet foods — a useful clinical clue; ask about dental history; low caries + aversion to sweets in a child with liver disease = suspect HFI",
            "ALDOB-RENAL-FANCONI-SYNDROME: Chronic fructose exposure causes proximal renal tubular dysfunction (Fanconi syndrome) — phosphaturia, bicarbonaturia, aminoaciduria, glucosuria → metabolic acidosis + growth retardation; REVERSES with fructose elimination",
            "ALDOB-IV-FRUCTOSE-SORBITOL-AVOID: IV fluids, parenteral nutrition, and medications must be screened for fructose/sorbitol content — HISTORIC FATAL CASES from IV fructose infusion for 'nutritional support' in undiagnosed HFI; laevulose (= fructose) and sorbitol in old IV formulations are ABSOLUTELY CONTRAINDICATED",
            "ALDOB-COMPOUND-HET-A149P-A174D: Most common European genotype is compound heterozygous A149P/A174D — targeted ALDOB mutation panel identifies >90% of European HFI; avoids need for IV fructose tolerance test (historically used, now CONTRAINDICATED due to risk)",
            "ALDOB-NEONATAL-PERIOD-SAFE: HFI is CLINICALLY SILENT in exclusively breastfed/formula-fed infants (breast milk and formula contain LACTOSE not fructose) — symptoms appear ONLY when fructose-containing foods are introduced; neonatal screening does NOT detect HFI; diagnose on symptom onset with weaning",
        ],
        "etiologies": {
            "A149P/A174D compound heterozygous (most common European)": 18,
            "A149P homozygous": 10,
            "A174D/N334K compound heterozygous": 6,
            "Other missense combinations": 4,
            "Rare frameshift/nonsense": 2,
        },
        "stats": {
            "acute_hypoglycaemia_pct": 85,
            "hepatomegaly_pct": 75,
            "renal_fanconi_pct": 40,
            "good_dental_health_pct": 80,
            "fructose_aversion_pct": 90,
            "liver_normalisation_on_diet_pct": 85,
            "mean_dx_age_months": 8,
            "mean_dx_delay_months": 6,
        },
        "dx_delay_distribution": {"<3m": 16, "3-12m": 16, "12-24m": 5, ">24m": 3},
    },
]


def _make_patients(gene_entry, rng):
    """Generate synthetic patient records for one gene."""
    gene = gene_entry["gene"]
    n = gene_entry["n_patients"]
    ages = [rng.randint(0, 65) for _ in range(n)]
    delays = [rng.choice([1, 3, 6, 12, 24, 36, 48, 60]) for _ in range(n)]
    etiol_keys = list(gene_entry["etiologies"].keys())
    etiol_weights = list(gene_entry["etiologies"].values())
    etiols = rng.choices(etiol_keys, weights=etiol_weights, k=n)
    patients = []
    for i in range(n):
        patients.append({
            "id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "dx_age": ages[i],
            "dx_delay_months": delays[i],
            "variant_class": etiols[i],
        })
    return patients


def _build_cohort():
    all_data = {}
    for idx, ge in enumerate(LIVER_GENES):
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        ge_copy = dict(ge)
        ge_copy["seed"] = seed
        ge_copy["patients"] = _make_patients(ge, rng)
        all_data[ge["gene"]] = ge_copy
    return all_data


_COHORT = _build_cohort()


def get_overview():
    genes_summary = []
    total = 0
    all_dx_ages = []
    all_delays = []
    top_alerts = []
    for gene, info in _COHORT.items():
        n = info["n_patients"]
        total += n
        pts = info["patients"]
        ages = [p["dx_age"] for p in pts]
        delays = [p["dx_delay_months"] for p in pts]
        all_dx_ages.extend(ages)
        all_delays.extend(delays)
        genes_summary.append({
            "gene": gene,
            "protein_short": info["protein"].split(" — ")[0],
            "n_patients": n,
            "locus": info["locus"],
            "inheritance": info["inheritance"].split(";")[0],
            "omim_disease": info["omim_disease"],
            "mean_dx_age": round(sum(ages) / len(ages), 1),
        })
        top_alerts.extend(info["key_alerts"][:2])
    aggregate_stats = {
        "total_patients": total,
        "mean_dx_age_years": round(sum(all_dx_ages) / len(all_dx_ages), 1),
        "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
        "atp7b_kf_rings_pct": 75,
        "hfe_phlebotomy_pct": 88,
        "serpina1_smoking_history_pct": 55,
        "jag1_cardiac_lesion_pct": 90,
        "atp8b1_low_ggt_pct": 95,
        "abcb11_hcc_childhood_pct": 8,
        "abcb4_high_ggt_pct": 95,
        "aldob_liver_normalisation_on_diet_pct": 85,
        "cascade_tested_pct": 68,
    }
    return {
        "atlas": "Hereditary-Liver-Disease-Atlas",
        "genes": genes_summary,
        "aggregate_stats": aggregate_stats,
        "top_alerts": top_alerts,
        "seeds": f"{SEED_BASE}–{SEED_BASE + 7}",
    }


def get_breakdown():
    result = {}
    for gene, info in _COHORT.items():
        pts = info["patients"]
        result[gene] = {
            "gene": gene,
            "n_patients": info["n_patients"],
            "alias": info["alias"],
            "gene_class": info["gene_class"],
            "locus": info["locus"],
            "aa": info["aa"],
            "kDa": info["kDa"],
            "omim_gene": info["omim_gene"],
            "omim_disease": info["omim_disease"],
            "inheritance": info["inheritance"],
            "key_alerts": info["key_alerts"],
            "etiologies": info["etiologies"],
            "stats": info["stats"],
            "dx_delay_distribution": info["dx_delay_distribution"],
            "patients": pts[:10],
        }
    return result


def get_definitions():
    return {
        "atlas": "Hereditary-Liver-Disease-Atlas",
        "concepts": {
            "Copper Metabolism Disorders (Wilson Disease)": (
                "ATP7B (1465 aa, 13q14.3, AR) encodes a copper-transporting P-type ATPase in hepatocytes. "
                "Deficiency → copper accumulates in liver, brain, cornea, kidneys. "
                "KF rings (Kayser-Fleischer — corneal copper deposits) are PATHOGNOMONIC for neurological Wilson's. "
                "Serum total copper is paradoxically LOW (ceruloplasmin-bound copper reduced); "
                "non-ceruloplasmin 'free' copper and urinary copper are ELEVATED. "
                "Treatment: chelation (D-Penicillamine/Trientine) or zinc; liver transplant for fulminant disease. "
                "D-Penicillamine can worsen neurological symptoms initially (mobilisation effect). "
                "Never stop treatment — acute decompensation risk within weeks."
            ),
            "Iron Overload (Hereditary Hemochromatosis)": (
                "HFE (343 aa, 6p21.3, AR) — C282Y p.Cys282Tyr homozygous = clinical HH. "
                "HFE modulates hepcidin signalling (via TfR1/TfR2); deficiency → low hepcidin → "
                "excess ferroportin activity → iron overload in liver, pancreas, joints, heart, pituitary. "
                "Classic triad: cirrhosis + diabetes + skin bronzing (historical — now diagnosed earlier). "
                "Transferrin saturation >45% is the FIRST biochemical abnormality. "
                "Therapeutic phlebotomy (450 mL weekly → ferritin <50 µg/L) is CURATIVE if pre-cirrhotic. "
                "Alcohol is a major HCC cofactor. Family screening (HFE genotype) mandatory."
            ),
            "Alpha-1 Antitrypsin Deficiency (SERPINA1/A1ATD)": (
                "SERPINA1 (418 aa, 14q32.13) encodes alpha-1 antitrypsin, principal inhibitor of neutrophil elastase. "
                "Z allele (p.Glu342Lys) — Z-A1AT POLYMERISES in hepatocyte ER → PAS-D globules → liver disease. "
                "PiZZ genotype: liver disease + emphysema (lower-lobe, panlobular — opposite to smoking). "
                "Null/Null: pure lung disease (no ER-retained polymer, no liver disease). "
                "IV augmentation therapy treats LUNG only — does not reverse liver disease. "
                "Smoking is catastrophic — oxidises A1AT active site Met358 → inactivates the protein. "
                "Fazirsiran (siRNA targeting hepatic A1AT production) in clinical trials."
            ),
            "Alagille Syndrome (JAG1 / Notch Signalling)": (
                "JAG1 (1218 aa, 20p12.2, AD) — Notch ligand required for intrahepatic bile duct morphogenesis. "
                "Haploinsufficiency → bile duct paucity (<0.9 ducts/portal tract) → cholestasis. "
                "Butterfly vertebrae (posterior arch defect, T2–T10) are PATHOGNOMONIC on spinal X-ray. "
                "Major features: bile duct paucity + butterfly vertebrae + cardiac (PPAS/TOF) + "
                "posterior embryotoxon (eye) + distinctive facies + renal disease. "
                "ICH risk 15% (intracranial vascular malformations). "
                "Cardiac lesions (PPAS) are the leading cause of death, not liver disease. "
                "Maralixibat (IBAT inhibitor, FDA 2021) approved for Alagille pruritus."
            ),
            "Progressive Familial Intrahepatic Cholestasis (PFIC1/2/3)": (
                "PFIC1 (ATP8B1/FIC1, 18q21.31, AR): LOW GGT + extrahepatic features (diarrhoea, pancreatitis, hearing loss). "
                "FIC1 is a phosphatidylserine flippase; also expressed in intestine/cochlea/pancreas. "
                "Liver transplant in PFIC1 may worsen diarrhoea/pancreatitis (extrahepatic FIC1 not corrected). "
                "PFIC2 (ABCB11/BSEP, 2q31.1, AR): LOW GGT + NO extrahepatic features; "
                "HCC risk in early childhood WITHOUT cirrhosis (unique); anti-BSEP antibodies post-transplant. "
                "PFIC3 (ABCB4/MDR3, 7q21.12, AR): HIGH GGT — cardinal DDx; PC-deficient bile → cholangiocyte damage. "
                "UDCA partially effective in PFIC3; very effective in LPAC (heterozygous ABCB4). "
                "Odevixibat/Maralixibat (IBAT inhibitors) approved for PFIC cholestasis."
            ),
            "Hereditary Fructose Intolerance (ALDOB)": (
                "ALDOB (364 aa, 9q31.1, AR) — Aldolase B, hepatic fructose-1-phosphate cleaving enzyme. "
                "Without Aldolase B: fructose → fructose-1-phosphate (by fructokinase) → F1P ACCUMULATES. "
                "F1P sequesters inorganic phosphate → profound hypoglycaemia (glycogenolysis impaired). "
                "F1P also directly toxic to hepatocytes and renal tubules → Fanconi syndrome. "
                "Appears on FIRST FRUCTOSE/SUCROSE/SORBITOL exposure (weaning). "
                "Good dental health (instinctive aversion to sweets) is a clinical clue. "
                "Fructose avoidance is ESSENTIALLY CURATIVE — liver normalises, Fanconi resolves. "
                "IV fructose/sorbitol infusions ABSOLUTELY CONTRAINDICATED — can be fatal. "
                "European founders: A149P (65–70%), A174D (15%); targeted panel covers >90% of Europeans."
            ),
            "GGT as Cholestasis Discriminator (PFIC Subtyping)": (
                "GGT (gamma-glutamyl transferase) is released by CHOLANGIOCYTES under bile acid detergent stress. "
                "LOW GGT cholestasis: PFIC1 (ATP8B1) + PFIC2 (ABCB11) — canalicular defect without bile duct damage. "
                "HIGH GGT cholestasis: PFIC3 (ABCB4) — PC-deficient bile damages cholangiocytes → GGT released. "
                "HIGH GGT also: Alagille (JAG1), TPN cholestasis, biliary atresia. "
                "LOW GGT + extrahepatic features (diarrhoea/pancreatitis) = PFIC1 (ATP8B1). "
                "LOW GGT + NO extrahepatic + HCC risk = PFIC2 (ABCB11). "
                "This GGT discrimination guides genetic testing order and management."
            ),
        },
        "pharmacological_distinctions": [
            "D-Penicillamine vs Trientine vs Zinc (Wilson's): D-Penicillamine = copper chelator, teratogenic (avoid in pregnancy); Trientine = copper chelator, safer neurologically, preferred for neurological Wilson's; Zinc = blocks intestinal absorption (not chelation), safe in pregnancy, maintenance therapy",
            "Therapeutic phlebotomy (HH): 450–500 mL per session → removes 200–250 mg iron; weekly until ferritin <50 µg/L; then 3-monthly maintenance; faster iron removal than chelation; preferred over chelation in HFE-HH who can tolerate phlebotomy",
            "IV A1AT augmentation (SERPINA1): Prolastin-C / Aralast NP / Glassia (weekly IV 60 mg/kg) — raises serum A1AT above 11 µmol/L protective threshold; treats LUNG disease only; does NOT remove Z-A1AT polymers from liver; NOT indicated for liver-only A1ATD",
            "UDCA in liver disease: 13–15 mg/kg/day — displaces toxic hydrophobic bile acids with UDCA (less toxic); partially effective PFIC3 (high GGT); very effective LPAC; limited benefit PFIC1/2 (low GGT — bile acid composition less important); standard for Wilson's secondary hepatitis adjunct",
            "IBAT inhibitors (Odevixibat / Maralixibat): block ileal bile acid transporter → reduce enteric bile acid reabsorption → reduce serum bile acids and pruritus; approved for PFIC (odevixibat EMA 2021, FDA PFIC 2023) and Alagille pruritus (maralixibat FDA 2021); better in residual transporter function",
            "Fructose restriction (HFI): avoid ALL fructose (fruit, honey, table sugar/sucrose = 50% fructose), ALL sucrose, ALL sorbitol (sugar alcohol = fructose precursor), HIGH-FRUCTOSE CORN SYRUP; also check IV fluids (laevulose/fructose in historical Ringer's = CONTRAINDICATED); essentially curative",
            "Liver transplantation disease-specific caveats: Wilson's → curative (donor ATP7B); HH → curative for liver but does not remove body iron stores (continue phlebotomy if haemochromatosis is systemic); A1ATD → curative (liver produces donor-type M-A1AT); PFIC1 → liver corrected but diarrhoea/pancreatitis may worsen; PFIC2 → liver corrected BUT anti-BSEP alloantibodies risk; PFIC3 → curative (no anti-MDR3 antibodies); Alagille → liver corrected but cardiac/vascular disease persists",
        ],
        "key_standards": [
            "EASL Clinical Practice Guidelines on Wilson's Disease — J Hepatol 2012; Rdc 2022 update — ceruloplasmin + urinary copper + KF rings + liver biopsy + gene score ≥4; D-Penicillamine vs trientine vs zinc; transplant for fulminant; never stop treatment",
            "EASL Clinical Practice Guidelines on Haemochromatosis — J Hepatol 2022 — transferrin saturation >45% as first screen; HFE C282Y genotyping; phlebotomy protocol; family cascade; alcohol cofactor HCC; post-cirrhotic surveillance",
            "Alpha-1 Foundation / ATS/ERS A1ATD Guidelines — Am J Respir Crit Care Med 2003/2016 — Pi typing; serum A1AT <0.57 g/L = deficiency; augmentation eligibility (FEV1 30–65%); smoking cessation mandatory; fazirsiran trials",
            "AASLD Alagille Syndrome Practice Guidance — Hepatology 2019 — JAG1/NOTCH2 mutation; bile duct paucity on biopsy; 5 major features; ICH surveillance; maralixibat pruritus; cardiac evaluation pre-transplant",
            "PFIC Registry Consensus / ESPGHAN PFIC Guidelines — J Pediatr Gastroenterol Nutr 2018/2023 — GGT-based subtyping; BSEP/FIC1/MDR3 genetic confirmation; odevixibat/maralixibat; biliary diversion criteria; HCC surveillance PFIC2",
            "HFI / ALDOB Metabolic Guideline — J Inherit Metab Dis 2011 — targeted mutation panel (A149P/A174D/N334K); dietary fructose/sucrose/sorbitol avoidance; IV fructose/sorbitol absolutely contraindicated; dental health as diagnostic clue; Fanconi surveillance",
            "WFH Liver Disease in Inherited Metabolic Disorders — Arnal H et al., Best Pract Res Clin Gastroenterol 2015 — diagnostic hierarchy copper/iron/A1AT/bile acid transporters; GGT discriminator; cross-disease HCC risk stratification",
        ],
    }
