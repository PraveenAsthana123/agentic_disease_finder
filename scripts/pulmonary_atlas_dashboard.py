#!/usr/bin/env python3
"""Pulmonary Hereditary Disease Atlas — Complete 8-Gene Hereditary Pulmonary Disease Atlas
CFTR    (Cystic Fibrosis — most common lethal AR lung disease in Caucasians;
         ΔF508 70%; elexacaftor/tezacaftor/ivacaftor FDA 2019; 7q31.2; 1480 aa) ·
SERPINA1 (Alpha-1-Antitrypsin Deficiency — PiZZ emphysema + liver disease;
          E342K misfolding; augmentation therapy; smoking ABSOLUTELY CI; 14q32.13; 418 aa) ·
BMPR2   (Hereditary PAH — 70-80% familial PAH; AD; incomplete penetrance;
          ERA+PDE5i+prostacyclin; pregnancy teratogenic; 2q33.1; 1038 aa) ·
SFTPB   (Surfactant Protein B deficiency — lethal neonatal; AR; 121ins2;
          lung transplant only curative; exogenous surfactant ineffective; 2p11.2; 381 aa) ·
SFTPC   (Surfactant Protein C dysfunction — childhood ILD; AD; I73T;
          hydroxychloroquine first-line; lung transplant refractory; 8p21.3; 197 aa) ·
ABCA3   (ABCA3 surfactant dysfunction — most common genetic surfactant cause; AR;
          small dense lamellar bodies EM PATHOGNOMONIC; lung Tx severe; 16p13.3; 1704 aa) ·
NKX2-1  (Brain-Lung-Thyroid syndrome — Brain+Lung+Thyroid triad; AD haploinsufficiency;
          antipsychotics CI chorea; L-thyroxine urgent; 14q13.3; 401 aa) ·
FLCN    (Birt-Hogg-Dubé — fibrofolliculomas+cysts+renal tumours; AD;
          pneumothorax 25-35%; renal RCC 7%; pleurodesis after 1st recurrence; 17p11.2; 579 aa)
320-patient aggregate cohort (8 × 40, seeds 1190–1197)
"""

import random

SEED_BASE = 1190

PULMONARY_GENES = [
    # ── CFTR — Cystic Fibrosis ───────────────────────────────────────────────
    {
        "gene": "CFTR",
        "protein": "Cystic fibrosis transmembrane conductance regulator",
        "alias": (
            "CFTR; OMIM gene 602421; 7q31.2; 1480 aa; CF OMIM #219700; "
            "AR; prevalence ~1 in 2500 (Northern European); carrier ~1 in 25; "
            "most common life-limiting AR disease in Caucasians"
        ),
        "aa": "1480 aa",
        "kDa": "~168 kDa",
        "gene_class": (
            "ABC transporter — ATP-binding cassette subfamily C; "
            "chloride/bicarbonate channel expressed at apical surface of epithelial cells; "
            "two TMDs, two NBDs, one regulatory R-domain; "
            "CFTR opens (conducts Cl⁻/HCO₃⁻) only when R-domain is phosphorylated (PKA) "
            "AND ATP is bound to NBDs; "
            "ΔF508 (most common, 70% CF alleles) → protein misfolding → ER retention → "
            "proteasomal degradation → absent apical CFTR → thick dehydrated mucus; "
            "Class I (stop/frameshift): no protein; Class II (ΔF508): misfolded ER retention; "
            "Class III (G551D): gating defect; Class IV: conductance defect; "
            "Class V: splicing reduction; Class VI: premature degradation"
        ),
        "locus": "7q31.2",
        "omim_gene": 602421,
        "omim_disease": 219700,
        "phenotype": (
            "Progressive obstructive lung disease (bronchiectasis, recurrent P. aeruginosa/MRSA); "
            "pancreatic exocrine insufficiency (85% — steatorrhoea, malabsorption, CFRD); "
            "meconium ileus (15-20% neonates); male infertility (CBAVD — congenital bilateral "
            "absence vas deferens in ~98% male CF); "
            "CF-related diabetes (CFRD, 40-50% of adults); "
            "liver disease (biliary cirrhosis 5-10%, portal hypertension); "
            "sinusitis, nasal polyps, CF-related arthropathy; "
            "sweat chloride >60 mmol/L DIAGNOSTIC (PATHOGNOMONIC)"
        ),
        "hallmark": (
            "SWEAT CHLORIDE >60 mmol/L — diagnostic gold standard; "
            "NEWBORN SCREENING: immunoreactive trypsinogen (IRT) + CFTR genotyping; "
            "SALTY SWEAT historically noted by parents ('salty baby kiss'); "
            "FINGER CLUBBING in progressive lung disease; "
            "STEATORRHOEA + FAILURE TO THRIVE in infants with pancreatic insufficiency; "
            "MALE INFERTILITY — CBAVD even in mild CF (5T/TGXXXXX alleles)"
        ),
        "treatment_alert": (
            "CFTR MODULATORS (Trikafta/ETI) — elexacaftor/tezacaftor/ivacaftor FDA 2019; "
            "approved age ≥2 years with ≥1 ΔF508 allele or responsive mutation; "
            "dramatically improves FEV1 (+10-14 pp), sweat chloride, exacerbation rate; "
            "DO NOT miss eligibility — check CFTR2.org mutation database; "
            "PANCREATIC ENZYME REPLACEMENT THERAPY (PERT) — with ALL meals and snacks "
            "(MANDATORY in pancreatic insufficient); "
            "AMINOGLYCOSIDES — ototoxic/nephrotoxic in CF; limit cumulative dose; "
            "monitor levels; avoid if possible with ETI era reduced exacerbations; "
            "DNASE (dornase alfa) — daily; reduces mucus viscosity; "
            "AIRWAY CLEARANCE — physiotherapy twice daily"
        ),
        "key_ddx": (
            "Primary ciliary dyskinesia (PCD): situs inversus 50%; "
            "normal sweat chloride; ciliary biopsy/nasal NO/DNAI1/DNAH5; "
            "Bronchiectasis (other): post-infectious, immunodeficiency (CVID/IgA); "
            "Shwachman-Diamond: pancreatic insufficiency + neutropenia + skeletal; "
            "Allergic bronchopulmonary aspergillosis (ABPA): complicates CF (10-15%); "
            "check specific IgE, total IgE, Aspergillus precipitins; "
            "NTM infection: Mycobacterium abscessus in CF — difficult to treat; "
            "Hyperoxaluria: urinary stones in CF due to fat malabsorption"
        ),
        "gfr_pattern": (
            "Renal function usually normal unless aminoglycoside nephrotoxicity; "
            "hyperoxaluria → calcium oxalate nephrolithiasis (poor fat absorption → "
            "oxalate not bound → absorbed); CKD rare unless drug toxicity or transplant"
        ),
        "proteinuria_pattern": (
            "Not primary; aminoglycoside nephrotoxicity may cause tubular proteinuria; "
            "CFRD-related glomerular disease rare but possible in long-standing hyperglycaemia"
        ),
        "primary_complication": "Progressive bronchiectasis + P. aeruginosa colonisation → respiratory failure",
        "disease_detail": (
            "CFTR protein forms a chloride/bicarbonate channel on the apical membrane of airway, "
            "intestinal, pancreatic ductal, hepatobiliary, and sweat gland epithelial cells.\n\n"
            "In the airway: CFTR secretes Cl⁻ and HCO₃⁻ into the airway surface liquid (ASL). "
            "Loss of CFTR function → dehydrated, acidic ASL → mucus dehydration → "
            "mucociliary clearance failure → chronic infection (Staphylococcus aureus early, "
            "Pseudomonas aeruginosa later — once established, impossible to eradicate) → "
            "inflammatory neutrophil-driven cycle → bronchiectasis.\n\n"
            "ΔF508 mechanism: phenylalanine 508 deletion in NBD1 → protein misfolding → "
            "recognition by ER quality control (Hsp70/CHIP) → ubiquitin-proteasomal degradation "
            "→ no CFTR at apical membrane. ETI corrects this: tezacaftor + elexacaftor stabilise "
            "ΔF508 CFTR in ER → ivacaftor potentiates gating of corrected protein at surface.\n\n"
            "Pancreatic: CFTR in ductal cells secretes HCO₃⁻ to alkalinise pancreatic juice. "
            "Loss → inspissated secretions → ductal obstruction → autodigestion → "
            "exocrine insufficiency (85% CF). Endocrine (islets of Langerhans) also damaged → CFRD.\n\n"
            "Hepatic: CFTR in biliary cholangiocytes → bile dehydration → "
            "focal biliary cirrhosis → multilobular cirrhosis in 5-10% → portal hypertension."
        ),
        "inheritance": "AR (autosomal recessive); de novo rare",
        "variants": [
            {"variant": "p.Phe508del (ΔF508)", "effect": "Class II — misfolding/ER retention", "frequency": "~70% CF alleles worldwide"},
            {"variant": "p.Gly551Asp (G551D)", "effect": "Class III — gating defect; ivacaftor target", "frequency": "~5% CF alleles"},
            {"variant": "p.Asn1303Lys (N1303K)", "effect": "Class II — misfolding", "frequency": "~2% CF alleles"},
            {"variant": "p.Trp1282Ter (W1282X)", "effect": "Class I — premature stop", "frequency": "~2%, Ashkenazi enriched"},
            {"variant": "IVS8-5T", "effect": "Class V — reduced splicing → CBAVD only", "frequency": "~10% general population"},
        ],
        "drug_ci": [
            "Aminoglycosides (tobramycin) — ototoxic and nephrotoxic; limit cumulative courses; TDM mandatory",
            "Ciprofloxacin + antacids/calcium/iron — chelation reduces bioavailability; separate by 2 hours",
            "NSAIDs (high-dose ibuprofen) — controversial; some evidence slows decline but GI/renal risk",
            "Ivacaftor monotherapy in ΔF508 homozygotes — do NOT use without corrector (elexacaftor/tezacaftor)",
        ],
    },

    # ── SERPINA1 — Alpha-1-Antitrypsin Deficiency ────────────────────────────
    {
        "gene": "SERPINA1",
        "protein": "Alpha-1-antitrypsin (α1AT)",
        "alias": (
            "SERPINA1; A1AT; AAT; PI; OMIM gene 107400; 14q32.13; 418 aa; AATD OMIM #613490; "
            "AR (severe form) / codominant; prevalence PiZZ ~1 in 3500 (Northern European); "
            "most common genetic cause of liver disease in paediatrics and emphysema in adults <45"
        ),
        "aa": "418 aa",
        "kDa": "~52 kDa (glycosylated ~55 kDa)",
        "gene_class": (
            "Serine protease inhibitor (serpin) superfamily — SERPINA1; "
            "synthesised by hepatocytes (90%); also macrophages, monocytes; "
            "major physiological role: inhibit neutrophil elastase (NE) in lung; "
            "Z allele (p.Glu342Lys): glutamate-342 → lysine → misfolding of reactive-centre loop "
            "domain → E342K opens hydrophobic pocket → Z polymers form in ER of hepatocytes → "
            "liver disease (polymerisation) + lung disease (insufficient secreted α1AT → "
            "unrestrained NE destroys alveolar walls → lower lobe panacinar emphysema); "
            "S allele (p.Glu264Val): milder misfolding, fewer polymers; "
            "PiMZ: intermediate; PiZZ most severe; PiSZ intermediate risk; "
            "M allele: normal; null alleles: no protein but no liver disease"
        ),
        "locus": "14q32.13",
        "omim_gene": 107400,
        "omim_disease": 613490,
        "phenotype": (
            "Pulmonary: Early-onset emphysema (3rd-4th decade); pan-acinar LOWER LOBE distribution "
            "(opposite of smoking-related upper lobe centrilobular); "
            "rapid decline in non-smokers ~30-60 mL FEV1/year (smokers ~200 mL/year); "
            "wheeze, dyspnoea, exercise intolerance; "
            "Hepatic: neonatal cholestasis (jaundice in 10% of PiZZ neonates); "
            "childhood hepatitis (usually resolves); adult cirrhosis (2-10% PiZZ); "
            "HCC risk (3-5% with cirrhosis); "
            "Panniculitis: necrotising skin inflammation (rare — α1AT deficiency panniculitis); "
            "Vasculitis: ANCA-associated in rare cases"
        ),
        "hallmark": (
            "LOWER LOBE EMPHYSEMA — panacinar lower zone distribution on CT; "
            "EARLY ONSET (<45y) emphysema — trigger AATD screen; "
            "SERUM α1AT <0.8 g/L (or <11 μM) — deficiency threshold; "
            "PiZZ on PI*ZZ genotyping — definitive diagnosis; "
            "PAS-D POSITIVE GLOBULES in hepatocytes — misfolded Z-polymer accumulation on liver biopsy"
        ),
        "treatment_alert": (
            "SMOKING — ABSOLUTELY CONTRAINDICATED; accelerates lung destruction 3-10x vs non-smokers; "
            "smoking cessation is THE most important intervention; "
            "AUGMENTATION THERAPY (IV α1AT weekly): Prolastin, Zemaira, Aralast — "
            "slows CT emphysema progression (RAPID study); NOT curative; "
            "threshold for Tx: serum α1AT <0.8 g/L + FEV1 25-65% (moderate disease); "
            "LIVER TRANSPLANT: curative for liver disease (replaces defective gene); "
            "recurrence zero (graft has donor SERPINA1 genotype); "
            "DANAZOL — some protocols for panniculitis but evidence limited; "
            "do NOT use in liver disease (hepatotoxic)"
        ),
        "key_ddx": (
            "COPD/emphysema (smoking-related): upper lobe centrilobular; AATD: lower lobe panacinar; "
            "asthma: spirometry reversible; AATD: fixed obstruction; "
            "Connective tissue disease lung: ILD pattern vs emphysema; "
            "Biliary atresia: neonatal jaundice — GGT HIGH (AATD normal early); "
            "Neonatal hepatitis syndrome: AATD must be excluded with PiZZ genotyping + liver Bx"
        ),
        "gfr_pattern": (
            "Generally normal renal function; "
            "ANCA-associated vasculitis (rare) → pauci-immune GN → AKI/CKD; "
            "augmentation therapy IV — no nephrotoxicity"
        ),
        "proteinuria_pattern": (
            "Not primary; ANCA vasculitis → haematuria + proteinuria in rare cases; "
            "nephrotic syndrome not associated with AATD directly"
        ),
        "primary_complication": "Panacinar lower-lobe emphysema + respiratory failure (accelerated by smoking)",
        "disease_detail": (
            "α1AT is the primary inhibitor of neutrophil elastase (NE) in the lung alveolus. "
            "NE degrades elastin, collagen, and other matrix proteins — vital for killing bacteria "
            "but destructive if unrestrained. α1AT forms a 1:1 irreversible complex with NE.\n\n"
            "Z allele mechanism: Glu342Lys introduces a basic residue that forms an intramolecular "
            "salt bridge with Lys290, opening a hydrophobic cavity → loop-sheet insertion "
            "polymer formation in hepatocyte ER → protein retained → "
            "(1) low circulating α1AT → NE unchecked in lung → elastolysis → panacinar emphysema; "
            "(2) Z-polymer accumulation → ER stress → UPR → hepatocyte apoptosis → liver fibrosis.\n\n"
            "Clinical sequence: PiZZ infants may have neonatal cholestasis (10%); "
            "most PiZZ adults have normal liver enzymes until 4th-6th decade; "
            "cirrhosis develops in ~10% of PiZZ adults. Lung: FEV1 decline begins in 2nd decade "
            "— emphysema symptomatic in non-smokers by 40s, smokers by 30s.\n\n"
            "Augmentation therapy weekly IV infusions of purified human α1AT maintain serum "
            "trough level >0.8 g/L, protecting alveoli from NE-mediated destruction. "
            "Does not reverse existing emphysema. Does not prevent liver disease "
            "(polymers are the problem in liver, not deficiency)."
        ),
        "inheritance": "AR (PiZZ severe); codominant (PiMZ, PiSZ intermediate); null alleles AR",
        "variants": [
            {"variant": "p.Glu342Lys (Z allele)", "effect": "Polymerisation → liver retention + lung deficiency", "frequency": "~95% of severe AATD"},
            {"variant": "p.Glu264Val (S allele)", "effect": "Mild polymerisation, less severe", "frequency": "~15% Southern European"},
            {"variant": "Null alleles (PI*Q0)", "effect": "No protein → lung disease, NO liver disease", "frequency": "Rare; <1%"},
        ],
        "drug_ci": [
            "Smoking — ABSOLUTELY CONTRAINDICATED; 3-10x acceleration of FEV1 decline",
            "Danazol — hepatotoxic; avoid in AATD liver disease",
            "Theophylline — narrow TI; avoid in severe disease; drug-drug interactions",
        ],
    },

    # ── BMPR2 — Hereditary Pulmonary Arterial Hypertension ──────────────────
    {
        "gene": "BMPR2",
        "protein": "Bone morphogenetic protein receptor type II",
        "alias": (
            "BMPR2; BMPR-II; OMIM gene 600799; 2q33.1; 1038 aa; HPAH OMIM #178600; "
            "AD; incomplete penetrance (~20% F, ~10% M); "
            "70-80% of familial PAH, ~25% of idiopathic PAH carry BMPR2 mutation; "
            "most common hereditary PAH gene worldwide"
        ),
        "aa": "1038 aa",
        "kDa": "~115 kDa",
        "gene_class": (
            "TGF-β superfamily type II serine/threonine kinase receptor; "
            "BMPR2 is an extracellular ligand-binding + transmembrane + kinase domain receptor; "
            "BMP2/4/7 binds BMPR2 → recruits BMPR1A/1B type I receptor → "
            "phosphorylates SMAD1/5/8 → SMAD4 complex → nuclear transcription of "
            "anti-proliferative, pro-apoptotic genes in pulmonary arterial smooth muscle cells (PASMC); "
            "LOF mutations (frameshift, nonsense, splice, missense) → haploinsufficiency → "
            "impaired BMP anti-proliferative signalling → PASMC proliferation + apoptosis resistance "
            "→ vascular remodelling → progressive pulmonary arterial obstruction → "
            "right heart failure; "
            "penetrance modifier: oestrogen (estradiol) → worse in females; "
            "inflammatory hits (inflammation, drugs) → trigger in mutation carriers"
        ),
        "locus": "2q33.1",
        "omim_gene": 600799,
        "omim_disease": 178600,
        "phenotype": (
            "Dyspnoea on exertion (insidious onset — median 2.5 years to diagnosis); "
            "fatigue, syncope on exertion (presyncope suggests low CO); "
            "RHC: mPAP >20 mmHg + PVR >2 WU (Wood Units) + PCWP <15 mmHg; "
            "6MWD reduced; NTproBNP elevated; echocardiography: TR + dilated RV; "
            "right heart failure: elevated JVP, peripheral oedema, hepatomegaly; "
            "haemoptysis in severe disease; HPAH: earlier onset + more severe vs iPAH"
        ),
        "hallmark": (
            "PROGRESSIVE EXERTIONAL DYSPNOEA — no obvious cause; "
            "RIGHT HEART CATHETER (RHC) mandatory for diagnosis — echo insufficient; "
            "YOUNG WOMAN with exertional syncope = PAH until proven otherwise; "
            "ELEVATED NTproBNP + DILATED RV on echo = initiate urgent workup; "
            "BMPR2 GENOTYPING — offered to all PAH patients (familial or idiopathic); "
            "GENETIC COUNSELLING — 1st-degree relatives of BMPR2+ patients"
        ),
        "treatment_alert": (
            "ENDOTHELIN RECEPTOR ANTAGONISTS (ERA) — bosentan, ambrisentan, macitentan; "
            "TERATOGENIC — mandatory contraception; "
            "BOSENTAN + CYCLOSPORIN — absolutely CI (liver toxicity + drug interaction); "
            "BOSENTAN + GLYBURIDE (glibenclamide) — absolutely CI (severe hypoglycaemia); "
            "PREGNANCY — ABSOLUTELY CI in severe PAH (mortality 30-50% peripartum); "
            "Termination or prevention of pregnancy mandatory in WHO FC III/IV; "
            "PDE5 INHIBITORS: sildenafil, tadalafil — do NOT combine with nitrates (severe hypotension); "
            "PROSTACYCLINS: IV epoprostenol (gold standard), SC/IV treprostinil, inhaled iloprost; "
            "WARFARIN: INR 1.5-2.5 in some centres (controversial — no RCT benefit); "
            "COMBINATION therapy now standard: ERA + PDE5i ± prostacyclin"
        ),
        "key_ddx": (
            "Post-capillary PH (heart failure): PCWP >15 mmHg on RHC; LVEF may be preserved; "
            "Group 3 PH (lung disease/hypoxia): COPD, ILD — lower PVR, clear lung disease; "
            "CTEPH (chronic thromboembolic PH): V/Q scan + CTPA — potentially surgically curable; "
            "Connective tissue disease PAH (SSc, SLE, MCTD): screen all CTD annually; "
            "Porto-pulmonary hypertension: portal hypertension causes — liver disease; "
            "Drug-induced PAH: fenfluramine (historical), dasatinib, methamphetamine"
        ),
        "gfr_pattern": (
            "Renal impairment in severe PAH from low cardiac output (cardiorenal syndrome); "
            "right heart failure → venous congestion → hepatic/renal congestion; "
            "BMPR2 mutations: no direct renal phenotype; "
            "AKI common in decompensated RHF"
        ),
        "proteinuria_pattern": (
            "Mild proteinuria in cardiorenal syndrome; "
            "not a primary renal disease"
        ),
        "primary_complication": "Right heart failure → death (median survival without therapy ~2.8 years from diagnosis)",
        "disease_detail": (
            "The BMP-SMAD signalling axis normally maintains pulmonary vascular homeostasis. "
            "In healthy PASMCs, BMP2/4/7 binding to BMPR2 recruits BMPR1A/B → "
            "phospho-SMAD1/5/8 → SMAD4 complex → transcription of ID1/ID2/ID3 "
            "and other anti-proliferative/pro-apoptotic genes → PASMC quiescence.\n\n"
            "BMPR2 haploinsufficiency disrupts this balance: the remaining BMPR2 allele "
            "cannot sustain adequate BMP signalling → TGF-β (pro-proliferative) signalling "
            "predominates → PASMC proliferative shift → medial hypertrophy → intimal "
            "remodelling → progressive obliterative vasculopathy → elevated PVR → "
            "right ventricular pressure overload → RVH → RV dilation → tricuspid "
            "regurgitation → right heart failure.\n\n"
            "Penetrance is incomplete (20% F, 10% M lifetime) — additional hits required: "
            "inflammation, oestrogen, SEROTONIN transporter polymorphisms, "
            "KCNK3 or CAV1 co-mutations can increase penetrance.\n\n"
            "Modern treatment: upfront triple combination (ERA + PDE5i + parenteral prostacyclin) "
            "for WHO FC III/IV HPAH has transformed prognosis — "
            "4-year survival now ~60-70% vs historical ~20%."
        ),
        "inheritance": "AD with incomplete penetrance (~20% females, ~10% males)",
        "variants": [
            {"variant": "Large deletions/duplications (~25%)", "effect": "Haploinsufficiency — most common class", "frequency": "~25% BMPR2 mutations"},
            {"variant": "Missense (kinase domain)", "effect": "Dominant-negative + haploinsufficiency", "frequency": "~40% BMPR2 mutations"},
            {"variant": "Frameshift/nonsense", "effect": "Haploinsufficiency via NMD", "frequency": "~35% BMPR2 mutations"},
        ],
        "drug_ci": [
            "ERA (bosentan/ambrisentan/macitentan) + pregnancy — ABSOLUTELY TERATOGENIC; mandatory contraception",
            "Bosentan + glyburide (glibenclamide) — severe hypoglycaemia; ABSOLUTELY CI",
            "Bosentan + cyclosporin — hepatotoxicity and drug level interactions; CI",
            "Sildenafil/tadalafil + nitrates — severe hypotension; CI",
            "Pregnancy in WHO FC III/IV PAH — mortality 30-50%; contraception or termination mandatory",
            "Calcium channel blockers — only in vasoreactive patients (10%); fatal in non-responders",
        ],
    },

    # ── SFTPB — Surfactant Protein B Deficiency ──────────────────────────────
    {
        "gene": "SFTPB",
        "protein": "Pulmonary surfactant-associated protein B",
        "alias": (
            "SFTPB; SP-B; OMIM gene 178640; 2p11.2; 381 aa (proprotein); "
            "SP-B OMIM #265120; AR; rare (~1 in 1.5 million births); "
            "most severe of all genetic surfactant dysfunction disorders"
        ),
        "aa": "381 aa (proprotein); mature 79 aa",
        "kDa": "~42 kDa (pro); ~8.7 kDa (mature dimer)",
        "gene_class": (
            "Saposin-like lipid transfer protein (SAPB/SAPLIP domain); "
            "produced by type II alveolar epithelial cells; "
            "essential for formation of tubular myelin (precursor to monolayer surface film); "
            "SP-B promotes phospholipid monolayer adsorption at air-liquid interface → "
            "reduces alveolar surface tension → prevents alveolar collapse at end-expiration; "
            "complete absence → no tubular myelin → no functional surfactant monolayer → "
            "alveolar collapse → respiratory failure at birth; "
            "121ins2 (OMIM frameshift) → most common CF-like recessive lethal mutation in Europeans; "
            "also required for processing of SP-C proprotein (SFTPC) — "
            "SP-B deficiency → abnormal proSP-C → secondary SP-C dysfunction"
        ),
        "locus": "2p11.2",
        "omim_gene": 178640,
        "omim_disease": 265120,
        "phenotype": (
            "NEONATAL LETHAL RESPIRATORY DISTRESS: full-term infant (not premature!) with "
            "progressive respiratory failure from birth; no response to exogenous surfactant; "
            "diffuse alveolar damage pattern on chest X-ray; "
            "CXR: bilateral ground-glass opacification (mimics severe prematurity RDS but TERM infant); "
            "BAL/lung biopsy: absent SP-B protein; abnormal lamellar bodies; "
            "elevated KL-6 (surfactant apoprotein marker); "
            "death within weeks-months without lung transplantation; "
            "HETEROZYGOTES (SP-B ±): usually asymptomatic carriers"
        ),
        "hallmark": (
            "FULL-TERM INFANT with progressive respiratory failure + NO response to exogenous surfactant = "
            "SFTPB deficiency until proven otherwise; "
            "ABSENT SP-B on BAL immunostaining or Western blot PATHOGNOMONIC; "
            "121ins2 (c.397_398insGGAGTGGGATTTG) frameshift → MOST COMMON SP-B null allele; "
            "LUNG BIOPSY: DIP/NSIP/PAP pattern + absent SP-B staining + abnormal LBs on EM"
        ),
        "treatment_alert": (
            "EXOGENOUS SURFACTANT (beractant, calfactant) — INEFFECTIVE in complete SFTPB deficiency; "
            "DO NOT repeat multiple courses when no response — consider genetic diagnosis; "
            "LUNG TRANSPLANTATION — ONLY curative treatment; "
            "bilateral lung transplant in infancy (few centres worldwide); "
            "Tx survival: 50% at 5 years; rejection and infection major risks; "
            "VENTILATORY SUPPORT — palliative bridge to transplant only; "
            "HIGH-FREQUENCY OSCILLATORY VENTILATION (HFOV) as bridge; "
            "GENETIC DIAGNOSIS — URGENTLY needed for any full-term infant with unexplained "
            "progressive respiratory failure unresponsive to surfactant; "
            "AUTOPSY — SP-B staining + EM mandatory in unexplained infant respiratory death"
        ),
        "key_ddx": (
            "Prematurity RDS: preterm (<34 weeks); responds to exogenous surfactant; "
            "Congenital pneumonia: maternal GBS, PROM; cultures positive; responds to antibiotics; "
            "SFTPC mutation: AD; onset often weeks-months; SP-B may be present; "
            "ABCA3 mutation: EM shows small dense LBs (SP-B absent→large LBs absent); "
            "NKX2-1 deletion: triad with thyroid/brain; SP-B detectable; "
            "Alveolar capillary dysplasia (ACD): cardiac anomalies, no lobar pattern; "
            "Congenital lymphatic disorders: chylothorax pattern"
        ),
        "gfr_pattern": (
            "Renal function normal in SFTPB deficiency; "
            "AKI in severe hypoxia or post-transplant nephrotoxicity (calcineurin inhibitors)"
        ),
        "proteinuria_pattern": (
            "Not a renal disease; "
            "post-transplant: calcineurin inhibitor nephrotoxicity → mild proteinuria possible"
        ),
        "primary_complication": "Neonatal lethal respiratory failure — no response to standard surfactant therapy",
        "disease_detail": (
            "Pulmonary surfactant is a lipoprotein complex (~90% lipid, ~10% protein) that reduces "
            "alveolar surface tension to near-zero at end-expiration, preventing alveolar collapse.\n\n"
            "Surfactant proteins: SP-A (immunological), SP-B (structural/functional, ESSENTIAL), "
            "SP-C (structural), SP-D (immunological).\n\n"
            "SP-B is the only surfactant protein that is absolutely essential for life. "
            "Without SP-B: (1) tubular myelin cannot form — the precursor lattice that stores "
            "and organises surfactant phospholipids; (2) surface-active phospholipid "
            "monolayer cannot adsorb at the air-liquid interface → surface tension remains high "
            "→ alveolar collapse at each expiration → progressive respiratory failure.\n\n"
            "Additionally, SP-B is required for proteolytic processing of proSP-C to mature SP-C "
            "by NAPSA (napsin A) in type II cells → SP-B deficiency → secondary SP-C dysfunction → "
            "worsening surfactant inadequacy.\n\n"
            "121ins2 mutation: insertion of 3 nucleotides after c.397 → frameshift → premature stop → "
            "no SP-B mRNA (NMD) → complete absence of protein → "
            "diagnosis by Western blot (absent band) or IHC on BAL cells or lung biopsy.\n\n"
            "Treatment: supportive ventilation + urgent referral for lung transplant evaluation."
        ),
        "inheritance": "AR; heterozygotes normal; complete penetrance in homozygotes/compound hets",
        "variants": [
            {"variant": "c.397_398insGGAGTGGGATTTG (121ins2)", "effect": "Frameshift → NMD → absent SP-B", "frequency": "~70% European SFTPB alleles"},
            {"variant": "Various missense/splice (compound het)", "effect": "Partial/complete deficiency — variable severity", "frequency": "Remainder"},
        ],
        "drug_ci": [
            "Exogenous surfactant (beractant/poractant) — INEFFECTIVE; do not repeat if no response",
            "Corticosteroids — ineffective for neonatal presentation; may delay diagnosis",
            "High cumulative oxygen exposure — worsens oxidative injury; minimise FiO2",
        ],
    },

    # ── SFTPC — Surfactant Protein C Dysfunction ─────────────────────────────
    {
        "gene": "SFTPC",
        "protein": "Pulmonary surfactant-associated protein C",
        "alias": (
            "SFTPC; SP-C; OMIM gene 178620; 8p21.3; 197 aa (proprotein); "
            "chILD OMIM #610913; AD; de novo in ~50%; "
            "most common genetic cause of childhood ILD (chILD)"
        ),
        "aa": "197 aa (proprotein); mature 35 aa",
        "kDa": "~21 kDa (pro); ~3.7 kDa (mature)",
        "gene_class": (
            "Hydrophobic surfactant protein; produced exclusively by type II alveolar pneumocytes; "
            "proSP-C (197 aa) contains: signal peptide + mature SP-C (N-terminal) + BRICHOS domain (C-terminal); "
            "BRICHOS domain acts as intramolecular chaperone preventing aggregation during ER processing; "
            "I73T (most common pathogenic variant) in the linker between mature and BRICHOS → "
            "misfolding → ER retention → ER stress → UPR → type II pneumocyte injury/apoptosis; "
            "absence of mature SP-C from alveolar surface → impaired surfactant function; "
            "AD gain-of-toxic-function through ER stress (not haploinsufficiency); "
            "variable penetrance — same I73T can cause neonatal, childhood, or adult ILD"
        ),
        "locus": "8p21.3",
        "omim_gene": 178620,
        "omim_disease": 610913,
        "phenotype": (
            "Variable presentation: neonatal ILD (severe, early); childhood ILD (most common); "
            "adult ILD (familial IPF presentation); "
            "Childhood ILD (chILD): tachypnoea, retractions, hypoxia; "
            "feeding difficulties and failure to thrive; "
            "HRCT: ground-glass opacification, interstitial thickening, cysts, consolidation; "
            "Lung biopsy: DIP (desquamative interstitial pneumonia), NSIP, PAP, or mixed pattern; "
            "FEV1/FVC normal or restrictive; DLCO reduced; "
            "Natural history: some improve with age (especially DIP pattern); some progressive"
        ),
        "hallmark": (
            "CHILDHOOD ILD with FAMILY HISTORY (parent affected) — think SFTPC; "
            "HRCT: DIP-like GGO + basal predominant + ± cysts; "
            "LUNG BIOPSY: macrophage accumulation (DIP pattern) + proSP-C ER retention staining; "
            "GENETIC TESTING: SFTPC sequencing + SP-B panel (exclude SFTPB compound het); "
            "I73T — MOST COMMON pathogenic SFTPC variant; de novo in ~50%"
        ),
        "treatment_alert": (
            "HYDROXYCHLOROQUINE (HCQ) — first-line in SFTPC chILD; "
            "mechanism: lysosomal pH stabilisation → reduced ER stress; "
            "dose: 5-10 mg/kg/day (max 400 mg/day); "
            "monitor ophthalmology annually (retinal toxicity with long-term use); "
            "SYSTEMIC CORTICOSTEROIDS (prednisolone) — second-line; "
            "pulse IV methylprednisolone for acute exacerbation; "
            "AZITHROMYCIN — anti-inflammatory; third-line adjunct; "
            "CHLOROQUINE — DO NOT use in children (more toxic than HCQ); "
            "LUNG TRANSPLANTATION — for progressive/refractory disease; "
            "wait for response to HCQ before listing (may stabilise); "
            "SUPPLEMENTAL O₂ — maintain SpO₂ >93%; "
            "PIRFENIDONE/NINTEDANIB — under investigation in adults with SFTPC ILD (not standard)"
        ),
        "key_ddx": (
            "SFTPB deficiency: AR, lethal neonatal, absent SP-B; more severe; "
            "ABCA3 mutation: AR, both alleles, EM small dense LBs; "
            "NKX2-1 deletion: AD, brain+lung+thyroid triad; "
            "Hypersensitivity pneumonitis: exposure history, serum precipitins; "
            "Pulmonary alveolar proteinosis (PAP): GM-CSF autoantibodies, BAL milky fluid; "
            "Idiopathic pulmonary fibrosis (IPF): adults, UIP pattern, no chILD pattern"
        ),
        "gfr_pattern": (
            "Renal function normal; "
            "hydroxychloroquine: no nephrotoxicity at therapeutic doses; "
            "AKI in severe hypoxia"
        ),
        "proteinuria_pattern": (
            "Not a renal disease; hydroxychloroquine: no proteinuria"
        ),
        "primary_complication": "Progressive childhood ILD → respiratory failure (variable — may stabilize with HCQ)",
        "disease_detail": (
            "proSP-C is unique among surfactant proteins in being AD toxic-gain-of-function rather "
            "than loss-of-function. Pathogenic SFTPC variants (especially I73T) impair "
            "the BRICHOS chaperone domain, causing proSP-C misfolding in the type II cell ER.\n\n"
            "Misfolded proSP-C: (1) not processed to mature SP-C → absent/reduced SP-C "
            "at alveolar surface → surfactant dysfunction; (2) accumulates in ER → "
            "ER stress → unfolded protein response (UPR) → CHOP-mediated apoptosis "
            "of type II pneumocytes → ILD pattern on biopsy.\n\n"
            "The BRICHOS domain (aa 88-197 of proSP-C) normally folds away from the mature "
            "domain, chaperoning it through processing. I73T disrupts the linker between "
            "mature and BRICHOS → aberrant proSP-C conformation → ER retention.\n\n"
            "Histopathological patterns seen: DIP (most common in chILD — macrophage "
            "accumulation in alveoli), NSIP (ground glass with basal subpleural sparing), "
            "PAP (lipoproteinaceous alveolar material), or mixed patterns.\n\n"
            "Hydroxychloroquine mechanism: alkalinises endo-lysosomes, reducing "
            "autophagic stress from accumulated proSP-C aggregates, "
            "thereby reducing type II cell injury."
        ),
        "inheritance": "AD; de novo ~50%; some familial (parent-to-child); variable penetrance",
        "variants": [
            {"variant": "p.Ile73Thr (I73T)", "effect": "ER retention → UPR → type II cell injury", "frequency": "Most common SFTPC pathogenic variant"},
            {"variant": "p.Leu188Gln", "effect": "BRICHOS domain disruption → aggregation", "frequency": "Second most common"},
            {"variant": "Exon 4 deletion", "effect": "Frame disruption → proSP-C aggregation", "frequency": "Rare"},
        ],
        "drug_ci": [
            "Chloroquine (not hydroxychloroquine) — more toxic, avoid in children",
            "NSAIDs — not CI per se, but anti-inflammatory approach should be through HCQ/steroids",
            "High-dose long-term corticosteroids without steroid-sparing — bone/metabolic toxicity",
        ],
    },

    # ── ABCA3 — ABCA3 Surfactant Dysfunction ─────────────────────────────────
    {
        "gene": "ABCA3",
        "protein": "ATP-binding cassette transporter A3",
        "alias": (
            "ABCA3; ABC-C; OMIM gene 601615; 16p13.3; 1704 aa; "
            "ABCA3 lung disease OMIM #610921; AR; "
            "most common genetic cause of surfactant dysfunction and chILD "
            "after ruling out SFTPB and SFTPC"
        ),
        "aa": "1704 aa",
        "kDa": "~190 kDa",
        "gene_class": (
            "ABC transporter subfamily A; localised to lamellar body (LB) limiting membrane "
            "in type II alveolar cells; "
            "transports phosphatidylcholine (PC) and phosphatidylglycerol (PG) into lamellar bodies; "
            "requires ATP hydrolysis at two NBDs; "
            "LBs are lysosome-related organelles that store and secrete surfactant; "
            "ABCA3 deficiency → impaired phospholipid loading → small, dense, "
            "eccentrically-positioned lamellar bodies on EM (PATHOGNOMONIC); "
            "insufficient surfactant phospholipids → reduced surface activity → ILD/RDS; "
            "severity depends on allele type: null+null (frameshift/nonsense) → severe neonatal; "
            "missense+null → intermediate; missense+missense → mild adult ILD"
        ),
        "locus": "16p13.3",
        "omim_gene": 601615,
        "omim_disease": 610921,
        "phenotype": (
            "Spectrum: (1) severe neonatal: term infant with RDS, similar to SFTPB but "
            "may be less immediately lethal; (2) childhood ILD: tachypnoea, hypoxia, "
            "failure to thrive; (3) adult ILD (rare — milder missense alleles); "
            "HRCT: bilateral GGO ± consolidation ± crazy paving; "
            "Lung biopsy: PAP pattern with cholesterol clefts, macrophage accumulation, DIP/NSIP; "
            "ABCA3 IHC: normal localisation but reduced function; "
            "EM: PATHOGNOMONIC small dense lamellar bodies (eccentric dense inclusions)"
        ),
        "hallmark": (
            "SMALL DENSE ECCENTRICALLY-POSITIONED LAMELLAR BODIES on EM — PATHOGNOMONIC for ABCA3 deficiency; "
            "contrasts with: normal large LBs (healthy), absent LBs (SFTPB), "
            "large irregular LBs (SFTPC); "
            "ABCA3 SEQUENCING + MLPA essential when EM shows this pattern; "
            "2-allele disease (AR) — both alleles must be identified for prognosis"
        ),
        "treatment_alert": (
            "NO CURATIVE THERAPY except lung transplantation (severe cases); "
            "HYDROXYCHLOROQUINE — often tried as in SFTPC chILD; variable response; "
            "SYSTEMIC CORTICOSTEROIDS — some response in inflammatory pattern; "
            "LUNG TRANSPLANT — for null+null genotype severe disease; "
            "SUPPLEMENTAL O₂ — maintain SpO₂ >93%; "
            "HIGH FiO₂ PROLONGED — worsens oxidative injury; use minimum FiO₂; "
            "PROGNOSIS BY GENOTYPE: null/null → poor (often transplant); "
            "missense/null → intermediate; missense/missense → often manageable"
        ),
        "key_ddx": (
            "SFTPB deficiency: absent SP-B; 121ins2 common; neonatal lethal; LBs absent not small; "
            "SFTPC mutation: AD; I73T; normal-sized LBs with ER proSP-C retention; "
            "Pulmonary alveolar proteinosis (PAP): GM-CSF autoantibody; whole-lung lavage curative; "
            "NKX2-1: triad brain+lung+thyroid; ABCA3 normal LB staining; "
            "Niemann-Pick type B: sphingomyelin accumulation; "
            "Hermansky-Pudlak: albinism + bleeding diathesis + ILD"
        ),
        "gfr_pattern": (
            "Renal function normal; not a renal disease; "
            "post-transplant calcineurin inhibitor nephrotoxicity possible"
        ),
        "proteinuria_pattern": (
            "Not a primary renal disease"
        ),
        "primary_complication": "Progressive ILD / neonatal respiratory failure → lung transplantation for severe genotypes",
        "disease_detail": (
            "ABCA3 is a lipid flippase on the lamellar body limiting membrane of type II alveolar cells. "
            "Its primary substrate is dipalmitoylphosphatidylcholine (DPPC, the major surface-active "
            "component of surfactant) and PG. ATP hydrolysis drives inward transport → "
            "phospholipid accumulation inside LBs → normal large round LBs (~0.5-1 μm, "
            "tightly packed lamellae) visible on EM.\n\n"
            "ABCA3 LOF: impaired lipid loading → LBs are SMALL (0.1-0.3 μm), DENSE, "
            "ECCENTRICALLY POSITIONED with an 'eccentric inclusion' — a dense core "
            "surrounded by thin membranes. This EM finding is PATHOGNOMONIC.\n\n"
            "Functional surfactant requires adequate DPPC content. Without ABCA3, "
            "surfactant phospholipid content is reduced → biophysical surface tension "
            "reduction is impaired → ILD/respiratory distress.\n\n"
            "Genotype-phenotype: null + null (two loss-of-function alleles: frameshift, "
            "nonsense, splice) → severe neonatal (similar to SFTPB deficiency, often requiring "
            "transplant). Missense + null → milder, childhood ILD. "
            "Missense + missense (if both partially functional) → mild or adult-onset ILD.\n\n"
            "ABCA3 mutations account for ~30% of unexplained chILD not explained by "
            "SFTPB or SFTPC mutations — it is the most common recessive surfactant gene."
        ),
        "inheritance": "AR; both alleles must be mutated for disease",
        "variants": [
            {"variant": "p.Glu292Val (E292V)", "effect": "Common missense — moderate severity", "frequency": "Most common European ABCA3 missense"},
            {"variant": "c.1421delC (frameshift)", "effect": "Null allele → severe neonatal", "frequency": "Common null allele"},
            {"variant": "Various missense NBD domains", "effect": "Reduced ATPase activity → intermediate", "frequency": "Multiple described"},
        ],
        "drug_ci": [
            "Exogenous surfactant — ineffective in ABCA3 deficiency (same mechanism as SFTPB)",
            "Prolonged high FiO2 — oxidative worsening; minimise supplemental O2",
            "Tacrolimus (post-transplant) — nephrotoxic; TDM mandatory",
        ],
    },

    # ── NKX2-1 — Brain-Lung-Thyroid Syndrome ─────────────────────────────────
    {
        "gene": "NKX2-1",
        "protein": "Homeobox protein Nkx-2.1 (thyroid transcription factor 1, TTF-1)",
        "alias": (
            "NKX2-1; TTF1; TITF1; OMIM gene 600635; 14q13.3; 401 aa; "
            "Brain-Lung-Thyroid syndrome OMIM #610978; "
            "AD (haploinsufficiency); ~50% de novo; "
            "also known as benign hereditary chorea (BHC) when lung/thyroid spared"
        ),
        "aa": "401 aa",
        "kDa": "~46 kDa",
        "gene_class": (
            "Homeodomain transcription factor (NK2 family); "
            "expressed in: (1) thyroid — controls thyroglobulin, thyroperoxidase, TSHR transcription; "
            "(2) lung — type II alveolar cells: controls SFTPA, SFTPB, SFTPC, ABCA3 transcription; "
            "(3) basal ganglia (striatum, caudate, putamen) — controls cholinergic interneuron "
            "differentiation; "
            "haploinsufficiency → reduced transcription of all targets simultaneously → "
            "three-organ syndrome; "
            "TTF-1/NKX2-1 also a lung adenocarcinoma marker (IHC positivity in >70% of lung AC); "
            "brain phenotype: chorea, hypotonia, ataxia — often improving in adolescence; "
            "NOT a malignant chorea (distinguish from Huntington); "
            "lung: neonatal RDS (type II cell maturation failure), childhood ILD, or adult ILD; "
            "thyroid: congenital hypothyroidism (CH) — may be mild; "
            "penetrance of all three organs: ~50%; brain-only ('BHC'): ~30%; full triad: ~30%"
        ),
        "locus": "14q13.3",
        "omim_gene": 600635,
        "omim_disease": 610978,
        "phenotype": (
            "Brain-Lung-Thyroid triad (complete syndrome): "
            "BRAIN: chorea (involuntary writhing/jerking movements, onset <5y), hypotonia, "
            "ataxia, cognitive impairment; typically IMPROVING by adolescence (benign); "
            "LUNG: neonatal RDS (term infant), recurrent infections, childhood ILD; "
            "THYROID: congenital hypothyroidism (CH) — TSH elevated + fT4 low at NBS; "
            "Partial forms: brain-only (BHC) — chorea without lung/thyroid; "
            "brain+thyroid without lung; lung+thyroid without chorea; "
            "NBS: TSH elevated → triggers paediatric endocrine workup → genetic testing reveals "
            "NKX2-1 → unexpected neurological signs found"
        ),
        "hallmark": (
            "TRIAD: Chorea + Lung disease + Congenital Hypothyroidism = NKX2-1 UNTIL PROVEN OTHERWISE; "
            "BENIGN HEREDITARY CHOREA — improving with age — NOT Huntington (progressive); "
            "CONGENITAL HYPOTHYROIDISM on NBS in a child with movement disorder; "
            "NEONATAL RDS in a TERM INFANT without another explanation; "
            "TTF-1 (NKX2-1) IHC — positive in lung biopsy of type II cells (reduces in deficiency)"
        ),
        "treatment_alert": (
            "CONGENITAL HYPOTHYROIDISM — start L-THYROXINE IMMEDIATELY on NBS positive; "
            "do NOT wait for further genetic confirmation; early treatment prevents intellectual disability; "
            "CHOREA: "
            "ANTIPSYCHOTICS (haloperidol, risperidone) — worsen chorea in NKX2-1; "
            "CLASSIC NEUROLEPTICS (haloperidol) — ABSOLUTELY CI — extrapyramidal worsening; "
            "Tetrabenazine (VMAT2 inhibitor) or clonazepam — may help chorea; "
            "PHYSIOTHERAPY for hypotonia + ataxia; "
            "LUNG: treat as per chILD protocol (HCQ ± steroids); "
            "LUNG INFECTIONS: prophylactic measures; RSV vaccine (palivizumab) in infancy; "
            "PROGNOSIS: chorea often improves spontaneously by adolescence; "
            "lung disease may stabilise; hypothyroidism lifelong L-thyroxine"
        ),
        "key_ddx": (
            "Huntington disease: progressive (not improving); CAG expansion; adult onset usually; "
            "Chorea-acanthocytosis: elevated CK, acanthocytes on blood film, VPS13A; "
            "GLUT1 deficiency: seizures + chorea + low CSF glucose; "
            "Wilson disease: liver disease + Kayser-Fleischer rings + low ceruloplasmin; "
            "Sydenham chorea: post-streptococcal; self-limiting; elevated ASOT; "
            "SFTPB deficiency: neonatal lethal; SP-B absent; NKX2-1 SP-B detectable (reduced)"
        ),
        "gfr_pattern": (
            "Renal function normal; "
            "hypothyroidism (if untreated) → decreased GFR → reverses with L-thyroxine"
        ),
        "proteinuria_pattern": (
            "Not a renal disease; "
            "hypothyroidism → reduced tubular protein handling → minor proteinuria possible; "
            "resolves with treatment"
        ),
        "primary_complication": "Congenital hypothyroidism (intellectual disability if untreated) + chorea (improving) + childhood ILD",
        "disease_detail": (
            "NKX2-1/TTF-1 is a master transcription factor for thyroid, lung alveolar, "
            "and forebrain striatal development. Its homeodomain (HD) binds the consensus "
            "sequence CAAG on target gene promoters.\n\n"
            "Lung: NKX2-1 binds promoters of SFTPA, SFTPB, SFTPC, ABCA3 — "
            "all major surfactant genes. NKX2-1 haploinsufficiency → reduced surfactant "
            "protein transcription → type II cell dysfunction → RDS at birth and/or ILD.\n\n"
            "Thyroid: NKX2-1 controls PAX8 co-activation of thyroglobulin (Tg), "
            "thyroperoxidase (TPO), and TSHR transcription. "
            "Haploinsufficiency → smaller thyroid gland → reduced T4 production → CH.\n\n"
            "Brain: NKX2-1 is expressed in GABAergic interneurons of the striatum during "
            "embryonic development. Haploinsufficiency → impaired interneuron specification "
            "→ dysregulation of striatal circuitry → choreiform movements. "
            "The chorea is 'benign' because the striatum matures postnatally and "
            "compensatory mechanisms improve motor control by adolescence.\n\n"
            "Molecular: NKX2-1 mutations include deletions (detected by MLPA/CMA), "
            "frameshift/nonsense (haploinsufficiency), and missense (HD domain — reduced DNA binding). "
            "Phenotype does NOT correlate perfectly with mutation type.\n\n"
            "Treatment priority: congenital hypothyroidism is the most time-sensitive — "
            "delayed treatment → irreversible intellectual disability."
        ),
        "inheritance": "AD; ~50% de novo; variable expressivity",
        "variants": [
            {"variant": "Deletion 14q13.3 (various sizes)", "effect": "NKX2-1 haploinsufficiency ± MBIP deletion", "frequency": "~30% NKX2-1 BLT cases"},
            {"variant": "Frameshift/nonsense", "effect": "Haploinsufficiency — full BLT", "frequency": "~40%"},
            {"variant": "Missense (homeodomain)", "effect": "Reduced DNA binding — partial phenotype", "frequency": "~30%"},
        ],
        "drug_ci": [
            "Antipsychotics (haloperidol, chlorpromazine, risperidone high dose) — worsen chorea; ABSOLUTELY CI classic neuroleptics",
            "Untreated hypothyroidism — delay in L-thyroxine → irreversible intellectual disability",
            "Metoclopramide (dopamine antagonist) — worsens movement disorder",
        ],
    },

    # ── FLCN — Birt-Hogg-Dubé Syndrome ──────────────────────────────────────
    {
        "gene": "FLCN",
        "protein": "Folliculin",
        "alias": (
            "FLCN; FLCN; OMIM gene 607273; 17p11.2; 579 aa; "
            "Birt-Hogg-Dubé syndrome OMIM #135150; "
            "AD; LOF tumour suppressor; prevalence ~1 in 200,000; "
            "triad: skin fibrofolliculomas + pulmonary cysts + renal tumours"
        ),
        "aa": "579 aa",
        "kDa": "~64 kDa",
        "gene_class": (
            "FLCN (folliculin) tumour suppressor — interacts with FNIP1/FNIP2 (folliculin-interacting "
            "proteins) and AMPK (AMP kinase); "
            "forms complex with FNIP1/2 → inhibits mTORC1 and mTORC2 signalling; "
            "LOF → mTORC1 hyperactivation → increased protein synthesis, cell growth, "
            "metabolic shift → tumour suppressor loss in kidney (oncocytoma/hybrid chromophobe); "
            "also regulates lysosome biogenesis via TFEB/TFE3; "
            "pulmonary cysts: FLCN LOF in lung → impaired cell-cell junction integrity "
            "(E-cadherin/occludin dysregulation) → alveolar-alveolar/alveolar-airway "
            "connections → cysts (not bullae); cysts rupture → pneumothorax; "
            "dominant-negative haploinsufficiency mechanism in cyst formation; "
            "skin: fibrofolliculomas = hamartomas of hair follicle-derived cells (PATHOGNOMONIC)"
        ),
        "locus": "17p11.2",
        "omim_gene": 607273,
        "omim_disease": 135150,
        "phenotype": (
            "SKIN: fibrofolliculomas — small (1-4 mm), skin-coloured papules on face/neck/upper trunk "
            "(pathognomonic); trichodiscomas, acrochordons (skin tags); onset 2nd-3rd decade; "
            "LUNG: bilateral, basilar-predominant, thin-walled cysts; usually NO symptoms; "
            "normal spirometry; normal DLCO; spontaneous pneumothorax (SP) in 25-35% lifetime "
            "(most below age 40); recurrent SP common; "
            "RENAL: bilateral, multifocal renal tumours; "
            "oncocytoma (35%), chromophobe RCC (35%), hybrid oncocytic/chromophobe tumour (30%); "
            "clear cell RCC rare; Wilms tumour and oncocytoma benign; "
            "chromophobe RCC metastatic potential ~7% lifetime; "
            "COLON: polyps — case reports; uncertain significance"
        ),
        "hallmark": (
            "SPONTANEOUS PNEUMOTHORAX in young adult with SKIN PAPULES — BHD until proven; "
            "FIBROFOLLICULOMAS on face/neck — PATHOGNOMONIC skin finding (biopsy if uncertain); "
            "BILATERAL CYSTS on CT chest (basilar, thin-walled) + NO emphysema history; "
            "FLCN SEQUENCING — germline detection; 84+/-9% pathogenic in BHD clinical criteria; "
            "RENAL SURVEILLANCE: annual USS/MRI from age 18-20; do NOT delay"
        ),
        "treatment_alert": (
            "PNEUMOTHORAX MANAGEMENT: "
            "First SP: aspiration or chest drain — standard management; "
            "Second SP: PLEURODESIS strongly recommended (chemical or surgical); "
            "AVOID sclerotherapy alone without definitive surgery if transplant possible (pleural "
            "adhesions compromise subsequent transplant); "
            "PLEURODESIS: discuss BHD diagnosis and implications before surgery; "
            "RENAL SURGERY: 'radical nephrectomy' must be AVOIDED for BHD renal tumours; "
            "NEPHRON-SPARING SURGERY MANDATORY — multiple tumours at different sites/times; "
            "ACTIVE SURVEILLANCE for tumours <3 cm (not curative surgery); "
            ">3 cm or growth >0.5 cm/year → nephron-sparing resection; "
            "mTOR INHIBITORS (everolimus, sirolimus) — experimental; not standard; "
            "SKIN: cosmetic laser/dermabrasion for fibrofolliculomas; no curative treatment needed; "
            "HIGH-ALTITUDE ACTIVITIES and FLYING: theoretical risk of SP — no absolute CI "
            "but counsel patients; commercial flying acceptable in stable patients"
        ),
        "key_ddx": (
            "LAM (lymphangioleiomyomatosis): females; TSC2/TCS1; chylous effusion; VEGF-D; "
            "LIP (lymphoid interstitial pneumonia): ground glass; HRCT different; "
            "alpha-1-AT deficiency emphysema: lower lobe; obstructive spirometry; panacinar; "
            "Marfan syndrome: tall stature, aortic dilation, lens dislocation, apical bullae; "
            "EDS: skin hyperextensibility, joint hypermobility, no cysts; "
            "Primary spontaneous pneumothorax (no BHD): young tall thin male; no cysts; "
            "VHL: RCC (clear cell) + retinal/CNS haemangioblastoma; no skin fibrofolliculomas"
        ),
        "gfr_pattern": (
            "GFR normal unless bilateral tumours removed (if radical nephrectomy performed — "
            "which should be AVOIDED); "
            "CKD from multiple nephron-sparing surgeries possible (cumulative nephron loss); "
            "bilateral tumour load → staged resection to preserve function"
        ),
        "proteinuria_pattern": (
            "Not a primary renal disease; "
            "proteinuria suggests malignant transformation or post-surgical change; "
            "routine urinalysis annually"
        ),
        "primary_complication": "Recurrent spontaneous pneumothorax + renal tumours (chromophobe RCC ~7% metastatic risk)",
        "disease_detail": (
            "FLCN (folliculin) is a tumour suppressor that negatively regulates mTORC1. "
            "FLCN-FNIP1/2 complex GAP (GTPase-activating protein) activity on RRAG GTPases "
            "→ RRAG·GDP → mTORC1 dissociation from lysosomal surface → mTORC1 OFF.\n\n"
            "FLCN LOF (haploinsufficiency + somatic second-hit in tumours) → "
            "constitutive mTORC1 activation → increased S6K, 4EBP1 phosphorylation → "
            "protein synthesis, lipid synthesis, cell proliferation → "
            "renal oncocytoma/hybrid tumour/chromophobe RCC.\n\n"
            "Pulmonary cysts: mechanism distinct from mTOR. FLCN haploinsufficiency → "
            "impaired AMPK-mediated phosphorylation of proteins maintaining alveolar cell "
            "adhesion junctions (E-cadherin, occludin) → mechanical fragility → "
            "alveolar overdistension → cyst formation. Cysts are bilateral, "
            "lower/medial lobe, thin-walled — NOT bullae. Normal lung function typically "
            "preserved (no airflow obstruction). Cysts rupture → pneumothorax.\n\n"
            "Skin fibrofolliculomas: FLCN somatic second-hit in hair follicle "
            "fibroblasts → mTORC1-driven hamartoma growth.\n\n"
            "Surveillance protocol: annual renal MRI (without gadolinium if possible); "
            "tumours <3 cm: observe; >3 cm or rapid growth: nephron-sparing resection. "
            "Skin surveillance and pneumothorax prevention education at every visit."
        ),
        "inheritance": "AD (tumour suppressor — germline LOF + somatic second-hit in tumours)",
        "variants": [
            {"variant": "c.1285dupC (Ins C in exon 11)", "effect": "Frameshift — most common Western mutation", "frequency": "~35% BHD families"},
            {"variant": "c.1285_1288delCins11", "effect": "Complex indel in exon 11 hotspot", "frequency": "~10% BHD"},
            {"variant": "Deletions (CMA/MLPA)", "effect": "Haploinsufficiency", "frequency": "~5-10% BHD"},
        ],
        "drug_ci": [
            "Radical nephrectomy — ABSOLUTELY CI in BHD; nephron-sparing mandatory (bilateral multifocal tumours)",
            "mTOR inhibitors (everolimus/sirolimus) — experimental only; not standard therapy; drug interactions",
            "Sclerotherapy pneumothorax without discussion of transplant implications — adhesions compromise future surgery",
        ],
    },
]


# ── Patient cohort generation ────────────────────────────────────────────────

def _make_cohort(gene: dict, seed: int, n: int = 40) -> list:
    """Generate a deterministic synthetic patient cohort for a hereditary pulmonary gene."""
    rng = random.Random(seed)
    gene_id = gene["gene"]
    patients = []

    for i in range(n):
        age = rng.randint(1, 60)
        sex = rng.choice(["M", "F"])

        if gene_id == "CFTR":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[20, 50, 30])[0]
            fev1_pct = rng.randint(30, 90) if severity != "Mild" else rng.randint(70, 100)
            pa_colonised = rng.random() < (0.80 if age > 18 else 0.35)
            cfrd = rng.random() < (0.45 if age > 25 else 0.08)
            pi = rng.random() < 0.85
            drug_error = rng.random() < 0.12  # missed modulator eligibility
            dx_delayed = rng.random() < 0.15
            esrd = rng.random() < 0.02
            htn = rng.random() < 0.12
            transplant = rng.random() < (0.08 if severity == "Severe" else 0.01)

        elif gene_id == "SERPINA1":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[15, 45, 40])[0]
            smoker = rng.random() < 0.30  # drug error if still smoking
            fev1_pct = rng.randint(20, 60) if severity == "Severe" else rng.randint(45, 85)
            liver_disease = rng.random() < 0.08
            augmentation_tx = severity in ("Moderate", "Severe") and rng.random() < 0.55
            drug_error = smoker  # smoking in AATD = drug error
            dx_delayed = rng.random() < 0.45  # average 5-6y diagnostic delay
            esrd = rng.random() < 0.02
            htn = rng.random() < 0.20
            transplant = rng.random() < (0.05 if severity == "Severe" else 0.01)

        elif gene_id == "BMPR2":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[10, 40, 50])[0]
            female = sex == "F"
            fc_class = rng.choices([2, 3, 4], weights=[30, 50, 20])[0]
            drug_error = rng.random() < 0.20  # ERA pregnancy, bosentan+glyburide
            dx_delayed = rng.random() < 0.55  # PAH diagnostic delay ~2.5 years
            esrd = rng.random() < 0.05
            htn = rng.random() < 0.15
            transplant = rng.random() < (0.12 if severity == "Severe" else 0.02)
            pneumothorax = False

        elif gene_id == "SFTPB":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[5, 15, 80])[0]
            age = rng.randint(0, 2)  # neonatal/infant
            drug_error = rng.random() < 0.60  # exogenous surfactant given repeatedly
            dx_delayed = rng.random() < 0.50
            esrd = rng.random() < 0.05
            htn = rng.random() < 0.05
            transplant = rng.random() < 0.40  # lung Tx rate

        elif gene_id == "SFTPC":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[25, 50, 25])[0]
            age = rng.randint(0, 45)
            drug_error = rng.random() < 0.15  # chloroquine instead of HCQ
            dx_delayed = rng.random() < 0.40
            esrd = rng.random() < 0.02
            htn = rng.random() < 0.08
            transplant = rng.random() < (0.10 if severity == "Severe" else 0.02)

        elif gene_id == "ABCA3":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[15, 40, 45])[0]
            age = rng.randint(0, 25)
            drug_error = rng.random() < 0.50  # repeated ineffective exogenous surfactant
            dx_delayed = rng.random() < 0.45
            esrd = rng.random() < 0.02
            htn = rng.random() < 0.05
            transplant = rng.random() < (0.25 if severity == "Severe" else 0.03)

        elif gene_id == "NKX2-1":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[35, 45, 20])[0]
            chorea = rng.random() < 0.85
            hypothyroid = rng.random() < 0.70
            lung_disease = rng.random() < 0.60
            drug_error = rng.random() < 0.30  # antipsychotic given for chorea
            dx_delayed = rng.random() < 0.50
            esrd = rng.random() < 0.01
            htn = rng.random() < 0.10
            transplant = rng.random() < 0.02

        elif gene_id == "FLCN":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[45, 40, 15])[0]
            pneumothorax = rng.random() < 0.30
            renal_tumour = rng.random() < 0.30
            fibrofolliculoma = rng.random() < 0.85
            radical_nephrectomy = renal_tumour and rng.random() < 0.25  # surgical error
            drug_error = radical_nephrectomy
            dx_delayed = rng.random() < 0.40
            esrd = rng.random() < 0.03  # post-radical-nephrectomy CKD
            htn = rng.random() < 0.15
            transplant = False

        surveillance_adherent = rng.random() < 0.60

        p = {
            "id": f"{gene_id}-{i+1:03d}",
            "gene": gene_id,
            "age": age,
            "sex": sex,
            "severity": severity,
            "esrd": esrd,
            "hypertension": htn,
            "transplant": transplant,
            "drug_error": drug_error,
            "dx_delayed": dx_delayed,
            "surveillance_adherent": surveillance_adherent,
        }
        patients.append(p)

    return patients


def _cohort_stats(patients: list) -> dict:
    n = len(patients)
    if n == 0:
        return {}
    def pct(key):
        return round(sum(1 for p in patients if p.get(key)) / n * 100, 1)
    sev = {s: round(sum(1 for p in patients if p["severity"] == s) / n * 100, 1)
           for s in ["Mild", "Moderate", "Severe"]}
    return {
        "n": n,
        "esrd_pct": pct("esrd"),
        "htn_pct": pct("hypertension"),
        "transplant_pct": pct("transplant"),
        "drug_error_pct": pct("drug_error"),
        "dx_delayed_pct": pct("dx_delayed"),
        "surveillance_adherent_pct": pct("surveillance_adherent"),
        "severity": sev,
    }


def _build_all_patients():
    all_patients = []
    for idx, gene in enumerate(PULMONARY_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(gene, seed=seed, n=40)
        all_patients.extend(cohort)
    return all_patients


ALL_PATIENTS = _build_all_patients()


# ── API response functions ───────────────────────────────────────────────────

def get_overview() -> dict:
    n = len(ALL_PATIENTS)
    agg = _cohort_stats(ALL_PATIENTS)
    return {
        "atlas_name": "Pulmonary Atlas",
        "atlas_subtitle": (
            "Complete 8-Gene Hereditary Pulmonary Disease Atlas — "
            "CFTR · SERPINA1 · BMPR2 · SFTPB · SFTPC · ABCA3 · NKX2-1 · FLCN"
        ),
        "n_genes": 8,
        "n_patients": n,
        "seeds": "1190–1197",
        "description": (
            "Comprehensive hereditary pulmonary disease reference covering the 8 most clinically "
            "significant monogenic lung disorders: "
            "Cystic Fibrosis (CFTR AR — ΔF508 70%; ETI/Trikafta FDA 2019; sweat Cl >60 PATHOGNOMONIC); "
            "Alpha-1-AT Deficiency (SERPINA1 — PiZZ panacinar lower emphysema; smoking ABSOLUTELY CI; "
            "augmentation therapy); "
            "Hereditary PAH (BMPR2 AD — 70-80% familial PAH; ERA+PDE5i+prostacyclin; "
            "ERA TERATOGENIC; bosentan+glyburide ABSOLUTELY CI); "
            "SP-B Deficiency (SFTPB AR — neonatal lethal; exogenous surfactant INEFFECTIVE; "
            "lung transplant only curative); "
            "SP-C Dysfunction (SFTPC AD — childhood ILD; I73T ER stress; HCQ first-line); "
            "ABCA3 Dysfunction (ABCA3 AR — small dense LBs PATHOGNOMONIC; most common genetic "
            "surfactant cause); "
            "Brain-Lung-Thyroid (NKX2-1 AD — chorea+ILD+hypothyroidism triad; "
            "antipsychotics ABSOLUTELY CI; L-thyroxine urgent); "
            "Birt-Hogg-Dubé (FLCN AD — fibrofolliculomas+cysts+RCC; "
            "radical nephrectomy ABSOLUTELY CI; pneumothorax 25-35%)"
        ),
        "aggregate_clinical": {
            "esrd_pct": agg.get("esrd_pct", 0),
            "hypertension_pct": agg.get("htn_pct", 0),
            "transplant_rate_pct": agg.get("transplant_pct", 0),
            "drug_error_pct": agg.get("drug_error_pct", 0),
            "diagnosis_delayed_pct": agg.get("dx_delayed_pct", 0),
            "surveillance_adherent_pct": agg.get("surveillance_adherent_pct", 0),
            "severity_mild_pct": agg.get("severity", {}).get("Mild", 0),
            "severity_moderate_pct": agg.get("severity", {}).get("Moderate", 0),
            "severity_severe_pct": agg.get("severity", {}).get("Severe", 0),
        },
        "drug_alerts": [
            {
                "type": "danger",
                "title": "SERPINA1-AATD: Smoking ABSOLUTELY CONTRAINDICATED — 3-10x Accelerated Emphysema",
                "body": (
                    "In PiZZ Alpha-1-Antitrypsin Deficiency, smoking destroys alpha-1-AT function "
                    "by direct oxidation of the methionine 358 active site, while simultaneously "
                    "recruiting additional neutrophils to the lung. PiZZ non-smokers lose "
                    "~30-60 mL FEV1/year; PiZZ smokers lose up to 200 mL FEV1/year. "
                    "Smoking cessation is the single most important intervention, more effective "
                    "than augmentation therapy in slowing FEV1 decline."
                ),
            },
            {
                "type": "danger",
                "title": "BMPR2-HPAH: ERA Teratogenic + Pregnancy ABSOLUTELY CI in WHO FC III/IV",
                "body": (
                    "Endothelin receptor antagonists (bosentan, ambrisentan, macitentan) are "
                    "Category X teratogens. Mandatory double contraception during treatment. "
                    "Pregnancy in WHO FC III/IV PAH carries 30-50% peripartum maternal mortality. "
                    "Bosentan + glyburide is an ABSOLUTE contraindication (severe hypoglycaemia). "
                    "Bosentan + cyclosporin is ABSOLUTELY CI (hepatotoxicity + drug levels)."
                ),
            },
            {
                "type": "danger",
                "title": "SFTPB: Exogenous Surfactant INEFFECTIVE — Lung Transplant is ONLY Curative Treatment",
                "body": (
                    "In complete SP-B deficiency, exogenous surfactant (beractant, calfactant, poractant) "
                    "cannot function without SP-B — the protein required to form tubular myelin "
                    "and organise the phospholipid monolayer is absent. Repeated surfactant doses "
                    "delay diagnosis. Any term infant with progressive RDS unresponsive to surfactant "
                    "requires urgent SFTPB genetic testing and lung transplant referral."
                ),
            },
            {
                "type": "danger",
                "title": "NKX2-1: Antipsychotics ABSOLUTELY CONTRAINDICATED — Worsen Chorea",
                "body": (
                    "Classic neuroleptics (haloperidol, chlorpromazine) and high-dose atypical "
                    "antipsychotics worsen choreiform movements in Brain-Lung-Thyroid syndrome "
                    "by dopamine receptor blockade in an already dysregulated striatal circuit. "
                    "Tetrabenazine (VMAT2 inhibitor) or clonazepam are preferred for chorea control. "
                    "Congenital hypothyroidism from NKX2-1 must be treated urgently with L-thyroxine."
                ),
            },
            {
                "type": "danger",
                "title": "FLCN-BHD: Radical Nephrectomy ABSOLUTELY CONTRAINDICATED — Nephron-Sparing Mandatory",
                "body": (
                    "BHD patients develop bilateral multifocal renal tumours at different sites and "
                    "different times throughout their life. Radical nephrectomy removes the entire "
                    "affected kidney leaving only one functioning kidney with future tumour risk. "
                    "All BHD renal surgery must be nephron-sparing (partial nephrectomy). "
                    "Tumours <3 cm: active surveillance. >3 cm or rapid growth: nephron-sparing resection."
                ),
            },
            {
                "type": "warning",
                "title": "CFTR: Check CFTR2 Mutation Database — Do NOT Miss Modulator Eligibility",
                "body": (
                    "Elexacaftor/tezacaftor/ivacaftor (ETI, Trikafta) is approved for ≥1 ΔF508 allele "
                    "OR a responsive mutation listed in the FDA label. "
                    "Before concluding a patient is ineligible, verify BOTH alleles against CFTR2.org "
                    "and the FDA-approved mutation list. Missing modulator eligibility is a "
                    "significant clinical error in 2024+ CF care."
                ),
            },
        ],
        "critical_rules": [
            "SERPINA1-AATD: Smoking ABSOLUTELY CI — stop before or instead of augmentation therapy",
            "BMPR2-HPAH: ERA (bosentan/ambrisentan/macitentan) TERATOGENIC — mandatory contraception",
            "BMPR2-HPAH: Bosentan + glyburide ABSOLUTELY CI — severe hypoglycaemia",
            "BMPR2-HPAH: Pregnancy in WHO FC III/IV = 30-50% maternal mortality; contraception mandatory",
            "SFTPB: Exogenous surfactant INEFFECTIVE in complete SP-B deficiency — do NOT repeat; transplant",
            "SFTPC: Hydroxychloroquine first-line; NOT chloroquine (more toxic in children)",
            "ABCA3: Small dense LBs on EM = PATHOGNOMONIC; null+null genotype → lung transplant",
            "NKX2-1: Antipsychotics ABSOLUTELY CI — worsen chorea; use tetrabenazine/clonazepam",
            "NKX2-1: L-thyroxine START IMMEDIATELY on NBS positive — delay = intellectual disability",
            "FLCN: Radical nephrectomy ABSOLUTELY CI — nephron-sparing surgery MANDATORY for all BHD renal tumours",
        ],
        "pathway_targets": {
            "CFTR":     "CFTR channel (ETI correctors/potentiators — class II/III modulators)",
            "SERPINA1": "Neutrophil elastase inhibition (α1AT augmentation + smoking cessation)",
            "BMPR2":    "BMP-SMAD axis (ERA + PDE5i + prostacyclin — triple combination)",
            "SFTPB":    "Surfactant lipid organisation (lung transplant — no other curative target)",
            "SFTPC":    "ER stress UPR (HCQ lysosomal stabilisation + steroids)",
            "ABCA3":    "Lamellar body lipid loading (HCQ/steroids + lung transplant severe)",
            "NKX2-1":   "Thyroid/lung transcription (L-thyroxine; HCQ lung; tetrabenazine chorea)",
            "FLCN":     "mTORC1 tumour suppression (surveillance + nephron-sparing + pleurodesis)",
        },
        "kpis": [
            {"label": "Total Patients", "value": str(n)},
            {"label": "Genes Covered", "value": "8"},
            {"label": "Cohort Seeds", "value": "1190–1197"},
            {"label": "Drug Errors (Agg)", "value": f"{agg.get('drug_error_pct', 0)}%"},
            {"label": "Dx Delayed (Agg)", "value": f"{agg.get('dx_delayed_pct', 0)}%"},
            {"label": "Transplant Rate", "value": f"{agg.get('transplant_pct', 0)}%"},
        ],
        "disease_category_breakdown": {
            "Airway Disease (CF)": 12.5,
            "Protease Inhibitor (AATD)": 12.5,
            "Vascular Remodelling (HPAH)": 12.5,
            "Surfactant Deficiency (SP-B)": 12.5,
            "Surfactant Dysfunction (SP-C)": 12.5,
            "Lipid Transport (ABCA3)": 12.5,
            "Transcription Factor (NKX2-1)": 12.5,
            "Tumour Suppressor (FLCN/BHD)": 12.5,
        },
    }


def get_breakdown() -> dict:
    genes_out = []
    for idx, gene_def in enumerate(PULMONARY_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(gene_def, seed=seed, n=40)
        stats = _cohort_stats(cohort)
        genes_out.append({
            "gene": gene_def["gene"],
            "protein": gene_def["protein"],
            "alias": gene_def["alias"],
            "aa": gene_def["aa"],
            "kDa": gene_def["kDa"],
            "gene_class": gene_def["gene_class"],
            "locus": gene_def["locus"],
            "omim_gene": gene_def["omim_gene"],
            "omim_disease": gene_def["omim_disease"],
            "phenotype": gene_def["phenotype"],
            "hallmark": gene_def["hallmark"],
            "treatment_alert": gene_def["treatment_alert"],
            "key_ddx": gene_def["key_ddx"],
            "gfr_pattern": gene_def["gfr_pattern"],
            "proteinuria_pattern": gene_def["proteinuria_pattern"],
            "primary_complication": gene_def["primary_complication"],
            "disease_detail": gene_def["disease_detail"],
            "inheritance": gene_def["inheritance"],
            "variants": gene_def.get("variants", []),
            "drug_ci": gene_def.get("drug_ci", []),
            "cohort_n": len(cohort),
            "cohort_stats": {
                "esrd_pct": stats.get("esrd_pct", 0),
                "htn_pct": stats.get("htn_pct", 0),
                "transplant_pct": stats.get("transplant_pct", 0),
                "drug_error_pct": stats.get("drug_error_pct", 0),
                "dx_delayed_pct": stats.get("dx_delayed_pct", 0),
                "surveillance_adherent_pct": stats.get("surveillance_adherent_pct", 0),
                "severity": stats.get("severity", {}),
            },
        })
    return {"genes": genes_out}


def get_definitions() -> list:
    return [
        {
            "term": "Sweat Chloride Test (Gibson-Cooke pilocarpine iontophoresis)",
            "full": ">60 mmol/L diagnostic for CF; 30-59 borderline; <30 normal",
            "explanation": (
                "The sweat chloride test is the gold-standard diagnostic test for cystic fibrosis. "
                "Pilocarpine is delivered by iontophoresis to stimulate sweat production; "
                "sweat is collected and chloride concentration measured. "
                "Normal: <30 mmol/L. Borderline: 30-59 mmol/L. Diagnostic for CF: ≥60 mmol/L. "
                "In CF, CFTR is absent from sweat gland duct epithelium → "
                "chloride secreted into sweat cannot be reabsorbed → high sweat Cl⁻. "
                "Minimum 75 mg sweat required for valid result. "
                "False positives: Addison's disease, hypothyroidism, malnutrition. "
                "False negatives: oedema, skin conditions, insufficient sweat collection. "
                "CF NEWBORN SCREENING: IRT (immunoreactive trypsinogen) is a blood test — "
                "if elevated → CFTR mutation panel → if 2 mutations OR 1 mutation + equivocal → "
                "sweat chloride test for confirmation."
            ),
        },
        {
            "term": "PiZZ Phenotype (Alpha-1-Antitrypsin Deficiency)",
            "full": "PiZZ = two Z alleles (p.Glu342Lys) → <15% normal α1AT → severe AATD",
            "explanation": (
                "The Pi (Protease Inhibitor) phenotyping system characterises α1AT alleles: "
                "M allele = normal (PiMM = 100% α1AT). "
                "Z allele (p.Glu342Lys, E342K) → misfolded protein → ER retention → "
                "only ~15% of normal α1AT secreted → severe deficiency. "
                "PiZZ: homozygous Z → <15% α1AT → emphysema + liver disease risk. "
                "PiSZ: one S (p.Glu264Val) + one Z → ~35% → intermediate lung risk. "
                "PiMZ: one M + one Z → ~50% → mild lung risk (borderline threshold). "
                "Null/null: no protein → lung disease but NO liver disease (no ER accumulation). "
                "Phenotyping methods: isoelectric focusing (gold standard), genotyping PCR for Z/S, "
                "quantitative serum α1AT level (<0.8 g/L = deficiency threshold). "
                "Treatment threshold: serum α1AT <0.8 g/L + FEV1 25-65% predicted "
                "→ augmentation therapy (IV purified human α1AT, weekly)."
            ),
        },
        {
            "term": "Right Heart Catheterisation (RHC) — PAH Diagnosis",
            "full": "mPAP >20 mmHg + PVR >2 WU + PCWP <15 mmHg = pre-capillary PAH",
            "explanation": (
                "Pulmonary arterial hypertension (PAH) cannot be diagnosed by echocardiography alone. "
                "RHC is the GOLD STANDARD — mandatory before initiating PAH therapy. "
                "Measured parameters: "
                "(1) mPAP (mean pulmonary arterial pressure) — normal <20 mmHg; "
                "PAH ≥20 mmHg by 2022 ESC/ERS updated threshold (was 25 mmHg previously); "
                "(2) PCWP (pulmonary capillary wedge pressure) — normal <15 mmHg; "
                "PCWP ≥15 → post-capillary PH (heart failure) — treat the left heart; "
                "(3) PVR (pulmonary vascular resistance) >2 Wood Units (WU) = significant vascular disease; "
                "Acute vasoreactivity testing (inhaled NO, IV epoprostenol): "
                "if mPAP drops ≥10 mmHg to <40 mmHg with improved CO = VASOREACTIVE — "
                "long-acting CCBs (amlodipine/diltiazem) may be used (ONLY ~10% are vasoreactive). "
                "Non-vasoreactive (90%): CCBs fatal — use ERA+PDE5i+prostacyclin."
            ),
        },
        {
            "term": "Tubular Myelin (Pulmonary Surfactant Architecture)",
            "full": "Lattice structure in alveolar hypophase — requires SP-B ABSOLUTELY; absent in SFTPB deficiency",
            "explanation": (
                "Pulmonary surfactant is stored in lamellar bodies and secreted into the alveolar "
                "hypophase (thin liquid layer lining alveolar surface). Upon secretion, "
                "lamellar body membranes unravel and assemble into tubular myelin — "
                "a square lattice structure visible on EM at the alveolar surface. "
                "Tubular myelin serves as a reservoir and organiser for the surface-active "
                "phospholipid monolayer that lowers surface tension. "
                "SP-B is ABSOLUTELY REQUIRED for tubular myelin formation — "
                "SP-B crosslinks membrane leaflets via its saposin domain. "
                "Without SP-B: no tubular myelin → no functional surface film → "
                "alveolar collapse at end-expiration → respiratory failure. "
                "SP-A also contributes to tubular myelin stability but is not essential. "
                "Exogenous surfactants (beractant = bovine extract containing SP-B) "
                "contain SP-B — but they CANNOT compensate for complete SFTPB deficiency "
                "because the endogenous processing machinery (type II cells) cannot "
                "form tubular myelin without endogenous SP-B."
            ),
        },
        {
            "term": "Small Dense Lamellar Bodies on EM (ABCA3 Deficiency)",
            "full": "Eccentric dense inclusions in small LBs — PATHOGNOMONIC for ABCA3 mutation",
            "explanation": (
                "Lamellar bodies (LBs) are lysosome-related organelles in type II alveolar cells "
                "that store and secrete pulmonary surfactant. On transmission electron microscopy (TEM): "
                "Normal LBs: round, ~0.5-1.0 μm diameter, tightly stacked concentric lamellae. "
                "ABCA3-deficient LBs: SMALL (~0.1-0.3 μm), DENSE, eccentrically positioned "
                "with a dense amorphous core (described as 'fried egg' or 'dense inclusion') — "
                "this results from impaired phospholipid loading by the ABCA3 transporter; "
                "without adequate DPPC/PG loading, LBs cannot form normal lamellae. "
                "Contrast with: SP-B deficiency — LBs present but abnormal/absent; "
                "SP-C mutation — LBs normal morphology (dysfunction is at ER processing stage). "
                "Clinical use: EM of lung biopsy or BAL pellet → small dense LBs → "
                "prompt ABCA3 genetic testing (both alleles needed for diagnosis). "
                "Cannot diagnose by light microscopy alone."
            ),
        },
        {
            "term": "Hydroxychloroquine (HCQ) in Childhood ILD",
            "full": "First-line treatment in SFTPC and ABCA3 chILD; lysosomal pH mechanism",
            "explanation": (
                "Hydroxychloroquine is a diprotic weak base that accumulates in acidic organelles "
                "(lysosomes, endosomes, autophagosomes) and raises their pH by proton buffering. "
                "In SFTPC I73T chILD: HCQ reduces autophagic stress from ER-retained misfolded "
                "proSP-C aggregates; reduced ER stress → less UPR-mediated type II cell apoptosis "
                "→ clinical improvement (GGO reduction on HRCT, improved oxygen requirement). "
                "Dose: 5-10 mg/kg/day in children (max 400 mg/day). "
                "Response: 3-6 months to assess; some patients show dramatic improvement. "
                "Monitoring: annual ophthalmology (retinal hydroxychloroquine toxicity with "
                "cumulative dose >5 mg/kg/day for >5 years → bull's eye maculopathy → "
                "permanent vision loss — Humphrey visual field + OCT annually). "
                "Chloroquine (NOT hydroxychloroquine): more toxic, especially hepatic — "
                "AVOID in children; use HCQ only."
            ),
        },
        {
            "term": "Brain-Lung-Thyroid Triad (NKX2-1/TTF-1 Syndrome)",
            "full": "Chorea + ILD + Congenital Hypothyroidism — AD NKX2-1 haploinsufficiency",
            "explanation": (
                "NKX2-1 (NK2 homeobox 1, also called TTF-1, Thyroid Transcription Factor 1) "
                "is a master transcription factor expressed in three organs: thyroid, lung, and brain. "
                "Haploinsufficiency → three-organ syndrome with variable penetrance per organ. "
                "BRAIN (most penetrant, ~85%): chorea (involuntary, choreiform movements) "
                "from impaired GABAergic striatal interneuron development. "
                "Benign hereditary chorea: chorea ALONE (brain only) — improves with age. "
                "THYROID (~70%): congenital hypothyroidism (CH) — elevated TSH at newborn screening; "
                "may be mild. L-thyroxine must start immediately. "
                "LUNG (~60%): neonatal RDS, childhood ILD, recurrent infections. "
                "Full 'BLT triad' in ~30% of cases. "
                "CRITICAL: hypothyroidism is the most time-critical — "
                "delayed L-thyroxine causes permanent intellectual disability. "
                "Chorea is NOT malignant (unlike Huntington) — improves spontaneously by adolescence. "
                "Antipsychotics WORSEN chorea — do not prescribe."
            ),
        },
        {
            "term": "Fibrofolliculoma (Birt-Hogg-Dubé)",
            "full": "Small hair follicle hamartoma on face/neck — PATHOGNOMONIC for BHD syndrome",
            "explanation": (
                "Fibrofolliculomas are benign hamartomas of hair follicle origin, "
                "the cutaneous hallmark of Birt-Hogg-Dubé (BHD) syndrome (FLCN mutations). "
                "Appearance: multiple small (1-4 mm), skin-coloured or pale, dome-shaped papules; "
                "distribution: face (nose, cheeks, nasolabial folds), neck, upper trunk; "
                "onset: typically 2nd-3rd decade; increase in number with age. "
                "Histology: expanded fibrous stroma surrounding a distorted hair follicle "
                "with 'anastomosing strands of epithelium' — PATHOGNOMONIC on biopsy. "
                "Distinguished from: sebaceous hyperplasia (lobular, no fibrous stroma), "
                "molluscum contagiosum (Henderson-Patterson bodies on biopsy), "
                "trichilemmoma (Cowden syndrome — PTEN mutation), "
                "angiofibromas (tuberous sclerosis — TSC1/2 mutation, central face distribution). "
                "Clinical significance: presence of fibrofolliculomas in a patient with "
                "spontaneous pneumothorax or renal tumour = DIAGNOSTIC of BHD; "
                "FLCN germline sequencing confirms. "
                "Treatment: cosmetic (laser, dermabrasion); medically benign but diagnostically crucial."
            ),
        },
        {
            "term": "BHD Renal Tumour Surveillance and Nephron-Sparing Mandate",
            "full": "Annual renal MRI from age 20; tumour >3 cm → nephron-sparing surgery; radical nephrectomy ABSOLUTELY CI",
            "explanation": (
                "BHD patients develop bilateral, multifocal renal tumours throughout their lifetime. "
                "Tumour types: hybrid oncocytic/chromophobe tumour (~50%, benign), "
                "chromophobe RCC (~35%, low metastatic risk ~7%), "
                "clear cell RCC (rare in BHD — only 2-3% vs >70% in sporadic RCC). "
                "Surveillance: annual renal MRI (preferred over CT — avoids radiation in young patients) "
                "from age 20 years; ultrasound if MRI not available/tolerated. "
                "Management thresholds: <3 cm → active surveillance (not surgery); "
                ">3 cm OR growth >0.5 cm/year → nephron-sparing resection (partial nephrectomy). "
                "RADICAL NEPHRECTOMY IS ABSOLUTELY CONTRAINDICATED in BHD: "
                "patients will develop tumours in the contralateral kidney, "
                "and in the same kidney (multi-focal disease). Removing an entire kidney "
                "leaves only one kidney at future tumour risk → eventual anephric state → dialysis. "
                "All BHD renal surgery must be nephron-sparing — even large tumours. "
                "Refer to urologist experienced with hereditary kidney cancer syndromes."
            ),
        },
        {
            "term": "BHD Pneumothorax — Pleurodesis after First Recurrence",
            "full": "25-35% lifetime SP risk in BHD; pleurodesis after ≥1 recurrence; radical surgery not needed",
            "explanation": (
                "Spontaneous pneumothorax (SP) is the most acute complication of BHD lung cysts. "
                "Lifetime risk: 25-35% (vs ~0.01% general population). "
                "Most occur in 3rd-4th decade. Cysts do NOT cause airflow obstruction — "
                "spirometry and DLCO are typically NORMAL in BHD. "
                "First SP: standard aspiration or intercostal chest drain; "
                "consider VATS (video-assisted thoracoscopic surgery) with bleb stapling + pleurodesis "
                "if large/persistent. "
                "Recurrence: high (>75% after first SP). "
                "After first or second SP: PLEURODESIS strongly recommended "
                "(mechanical abrasion + talc poudrage or pleurectomy). "
                "IMPORTANT: avoid chemical pleurodesis if lung transplantation is being "
                "considered (pleural adhesions complicate transplant surgery significantly). "
                "High-altitude activities and SCUBA diving: counsel (theoretical pressure changes "
                "may precipitate SP) but no absolute evidence-based CI for commercial flying. "
                "BHD cysts DO NOT respond to pleurodesis — pleurodesis prevents recurrence, "
                "not cyst formation."
            ),
        },
    ]
