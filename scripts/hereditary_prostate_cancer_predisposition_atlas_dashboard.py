#!/usr/bin/env python3
"""Hereditary-Prostate-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
BRCA2   (BRCA2 DNA repair associated; 3418aa; 13q12.3; AD LOF;
         HBOC — prostate 15-20x RR; lethal/metastatic enriched;
         PROfound olaparib FDA 2020 HR 0.22; PSMA-PET MANDATORY;
         seed SEED_BASE+0) .
BRCA1   (BRCA1 DNA repair associated; 1863aa; 17q21.31; AD LOF;
         HBOC — prostate 2-3x RR; weaker than BRCA2;
         PARPi less effective HR 0.82; cascade HBOC family;
         seed SEED_BASE+1) .
ATM     (ATM serine/threonine kinase; 3056aa; 11q22.3; AD AR LOF;
         prostate 2-4x RR; PROfound olaparib modest HR 0.72;
         biallelic AT radiosensitivity ABSOLUTE; ceralasertib ATRi trials;
         seed SEED_BASE+2) .
CHEK2   (Checkpoint kinase 2; 543aa; 22q12.1; AD LOF;
         prostate 2-3x RR; c.1100delC Northern European; I157T Central/Eastern European;
         NO PARPi standard — NOT a HRD gene; ADT standard;
         seed SEED_BASE+3) .
HOXB13  (Homeobox B13; 284aa; 17q21.32; AD LOF;
         G84E founder Scandinavian; 4-8x RR prostate; prostate-ONLY gene;
         NO targeted therapy; PSA surveillance from 40yr;
         seed SEED_BASE+4) .
MSH2    (MutS homolog 2; 934aa; 2p21; AD LOF;
         Lynch type 2 — prostate 5-10x RR; dMMR Pembrolizumab mPRIMO;
         urothelial 14% HIGHEST Lynch gene; EPCAM 3'-deletion MLPA MANDATORY;
         seed SEED_BASE+5) .
PALB2   (Partner and localiser of BRCA2; 1186aa; 16p12.2; AD LOF;
         HBOC2 — prostate 2-4x emerging; PARPi sensitive; BRCA1-BRCA2 bridge;
         TBCRC048 82% ORR; FA-N biallelic;
         seed SEED_BASE+6) .
NBN     (Nibrin; 754aa; 8q21.3; AR AD LOF;
         Nijmegen Breakage Syndrome; 657del5 Slavic founder;
         prostate 3-4x RR heterozygous; MRN complex; radiation sensitivity intermediate;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3390-3397)
"""
import random

SEED_BASE = 3390

ATLAS_GENES = [
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "FANCD1-HR-Mediator-384kDa-BRC-Repeats-RAD51-Loader-"
            "Prostate-15-20x-RR-LETHAL-Metastatic-Enriched-"
            "PROfound-Olaparib-FDA2020-HR-0.22-PATHOGNOMONIC-BRCA2-HRD-"
            "PSMA-PET-MANDATORY-DDR-Defect-Cisplatin-Carboplatin-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 prostate cancer molecular context: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; FANCD1 — Fanconi anaemia type D1; "
            "  BRC repeats (aa 1002-2085): 8 repeats — each binds RAD51; "
            "  OB folds (aa 2400-3186): ssDNA binding; "
            "  C-terminal BRCA2-DBD: RAD51 filament nucleation on ssDNA; "
            "CANCER RISKS (PROSTATE FOCUS): "
            "  PROSTATE: 15-20x RR — lifetime 60% in BRCA2 carriers (vs 11% general); "
            "  Metastatic at diagnosis: 3x more likely; lethal prostate enriched BRCA2; "
            "  PSA velocity and PSMA-PET: MANDATORY for BRCA2 prostate surveillance; "
            "  BREAST male: 6% lifetime HIGHEST male BRCA gene; "
            "  PANCREATIC: 5-7% lifetime; olaparib POLO maintenance; "
            "  OVARIAN: 11-17% in female relatives; "
            "KEY MANAGEMENT (PROSTATE): "
            "  Annual PSA from 40yr (35yr if strong family history); "
            "  PSMA-PET MANDATORY for staging in BRCA2 — detects micro-metastases standard CT misses; "
            "  OLAPARIB: PROfound trial HR 0.22 (BRCA1/2 cohort) — FDA approved 2020 mCRPC; "
            "  RUCAPARIB: also approved mCRPC BRCA2 (TRITON2); "
            "  PLATINUM CHEMOTHERAPY: carboplatin + docetaxel — HRD synergy; "
            "  GENETIC COUNSELLING: all first-degree male relatives — prostate surveillance; "
            "SOMATIC BRCA2: 10-15% prostate tumours somatic — secondary hit confirms biallelic LOF"
        ),
        "syndrome": "Hereditary Breast-Ovarian Cancer (HBOC type 1 — male phenotype dominant prostate/male breast)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "prostate_risk": "15-20x RR; lifetime ~60%; lethal/metastatic phenotype enriched",
        "pathognomonic": "Lethal/high-grade prostate cancer family + male breast + pancreatic; BRCA2 DFT signature",
        "key_avoid": "Do NOT treat BRCA2 mCRPC without olaparib consideration — HR 0.22; AVOID enzalutamide alone without olaparib in BRCA2 biomarker-unselected",
        "key_rule": "BRCA2 > BRCA1 > ATM for prostate lethal risk. PSMA-PET MANDATORY in BRCA2 staging. Annual PSA from 40yr.",
        "surveillance": "Annual PSA + DRE from 40yr (35yr if early-onset family); PSMA-PET at staging; cascade male relatives",
        "targeted_rx": "Olaparib PROfound FDA2020; rucaparib TRITON2; platinum+docetaxel; cabazitaxel BRCA2-enriched response",
    },
    {
        "gene": "BRCA1",
        "protein": (
            "BRCA1 -- 17q21.31 Autosomal-Dominant-LOF -- 1863aa -- "
            "RING-E3-Ligase-HR-Scaffold-208kDa-BARD1-Heterodimer-"
            "Prostate-2-3x-RR-WEAKER-Than-BRCA2-"
            "PARPi-LESS-EFFECTIVE-HR-0.82-PROfound-"
            "Female-HBOC-Cascade-MANDATORY-Breast-70-80pct-Ovarian-39-44pct-OMIM-113705"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "1863 aa / 208 kDa / 17q21.31 BRCA1 prostate cancer molecular context: "
            "STRUCTURE: "
            "  1863 aa / 208 kDa; RING domain (aa 1-109): E3 ubiquitin ligase with BARD1; "
            "  BRCT domains (aa 1646-1863): binds pSer motifs — DNA damage signalling; "
            "  BRCA1 forms foci at DSBs: scaffold for HR repair complex; "
            "CANCER RISKS (PROSTATE FOCUS): "
            "  PROSTATE: 2-3x RR (weaker than BRCA2 15-20x); predominantly ERG-negative; "
            "  PROSTATE NOTE: PARPi HR 0.82 in BRCA1 vs HR 0.22 in BRCA2 — significant distinction; "
            "  BREAST female: 70-80% lifetime — TNBC enriched; "
            "  OVARIAN: 39-44% lifetime HGSOC; "
            "KEY MANAGEMENT (PROSTATE): "
            "  Annual PSA from 40yr; "
            "  CASCADE: female relatives MANDATORY — breast MRI + RRSO; "
            "  OLAPARIB: modest benefit HR 0.82 in BRCA1 mCRPC — still approved; "
            "  Genomic testing: distinguish BRCA1 from BRCA2 — critical for PARPi benefit prediction; "
            "  BRCA1 prostate: ERG-negative enriched; distinct biology from BRCA2 "
        ),
        "syndrome": "Hereditary Breast-Ovarian Cancer (HBOC type 1 — female phenotype dominant; prostate moderate elevation)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "prostate_risk": "2-3x RR; prostate NOT dominant BRCA1 phenotype",
        "pathognomonic": "Female relatives with breast/ovarian; BRCA1 DFT signature; TNBC in family",
        "key_avoid": "Do NOT conflate BRCA1 prostate risk with BRCA2 — PARPi HR 0.82 vs 0.22; counsel female relatives PRIMARILY",
        "key_rule": "BRCA1 prostate: CASCADE FEMALE RELATIVES first. PARPi still approved but modest. Annual PSA from 40yr.",
        "surveillance": "Annual PSA + DRE from 40yr; female relatives: annual breast MRI + RRSO counselling 35-40yr",
        "targeted_rx": "Olaparib PROfound FDA2020 (BRCA1/2 combined but BRCA1 modest); standard CRPC regimens",
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Autosomal-Dominant-AR-LOF -- 3056aa -- "
            "PI3K-Like-Kinase-350kDa-DNA-Damage-Sensor-DSB-Master-"
            "Prostate-2-4x-RR-Monoallelic-"
            "PROfound-Olaparib-Modest-HR-0.72-"
            "Biallelic-A-T-RADIOSENSITIVITY-ABSOLUTE-Telangiectasias-PATHOGNOMONIC-"
            "Ceralasertib-ATRi-Trials-AFP-Elevated-OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM prostate cancer molecular context: "
            "STRUCTURE: "
            "  3056 aa / 350 kDa; PI3K-like domain (aa 2712-3056): kinase; "
            "  HEAT repeats (aa 1-1500): scaffold; FAT/FATC domains; "
            "  Activates p53, CHEK2, BRCA1, H2AX at DSBs; "
            "CANCER RISKS (PROSTATE FOCUS): "
            "  PROSTATE: 2-4x RR heterozygous — intermediate DDR gene; "
            "  PROfound trial: ATM cohort HR 0.72 (modest, unlike BRCA2 HR 0.22); "
            "  BREAST female: 30-35% monoallelic — intermediate risk; "
            "  BIALLELIC: Ataxia-Telangiectasia (A-T) — cerebellar ataxia + telangiectasias + RADIOSENSITIVITY ABSOLUTE; "
            "KEY MANAGEMENT (PROSTATE): "
            "  Annual PSA from 40yr; "
            "  RADIATION SENSITIVITY: monoallelic ATM — intermediate radiosensitivity (not absolute CI like biallelic); "
            "  Reduce RT dose or fractionation in ATM carriers when treating prostate; "
            "  OLAPARIB: approved mCRPC ATM cohort (PROfound) — modest benefit HR 0.72; "
            "  CERALASERTIB (ATR inhibitor): ATM-deficient tumours — synthetic lethal synergy; "
            "  DDR-selected trial enrollment priority for ATM mCRPC"
        ),
        "syndrome": "Ataxia-Telangiectasia (biallelic); hereditary prostate predisposition (monoallelic)",
        "inheritance": "AR biallelic (A-T) / AD monoallelic (prostate/breast predisposition)",
        "prostate_risk": "2-4x RR monoallelic; PROfound ATM cohort HR 0.72",
        "pathognomonic": "Biallelic A-T: cerebellar ataxia + oculomotor apraxia + telangiectasias + AFP elevated + radiosensitivity",
        "key_avoid": "BIALLELIC ATM: RADIATION ABSOLUTELY CONTRAINDICATED; monoallelic: reduce RT dose; Do NOT conflate ATM PARPi benefit (modest) with BRCA2 (dramatic)",
        "key_rule": "ATM prostate: PARPi approved but modest. Ceralasertib ATRi trials preferred. Annual PSA from 40yr. Female relatives: breast 30-35%.",
        "surveillance": "Annual PSA + DRE from 40yr; biallelic carriers: annual brain MRI + immunology; monoallelic female relatives: annual breast MRI from 30yr",
        "targeted_rx": "Olaparib PROfound FDA2020 (modest HR 0.72); ceralasertib ATRi trials; platinum-based chemo ATM-deficient",
    },
    {
        "gene": "CHEK2",
        "protein": (
            "CHEK2 -- 22q12.1 Autosomal-Dominant-LOF -- 543aa -- "
            "Checkpoint-Kinase2-61kDa-FHA-Domain-Kinase-Domain-"
            "Prostate-2-3x-RR-Moderate-"
            "c.1100delC-Northern-European-I157T-Central-Eastern-European-"
            "NO-PARPi-Standard-NOT-HRD-Gene-"
            "ADT-Standard-Moderate-Risk-OMIM-604373"
        ),
        "locus": "22q12.1",
        "protein_size": (
            "543 aa / 61 kDa / 22q12.1 CHEK2 prostate cancer molecular context: "
            "STRUCTURE: "
            "  543 aa / 61 kDa; FHA domain (aa 92-175): phosphopeptide binding; "
            "  Kinase domain (aa 220-501): phosphorylates p53, BRCA1, CDC25A; "
            "  Activated by ATM at DSBs; checkpoint signalling; "
            "CANCER RISKS (PROSTATE FOCUS): "
            "  PROSTATE: 2-3x RR moderate — c.1100delC Northern European founder; I157T Central/Eastern European; "
            "  CHEK2 prostate: NOT a HRD gene — PARPi does NOT work (no DNA repair defect like BRCA2); "
            "  BREAST female: 20-25% moderate risk; bilateral elevated; "
            "  COLON: moderate 2x RR; "
            "KEY MANAGEMENT (PROSTATE): "
            "  Annual PSA from 40yr (or 5yr earlier than youngest affected relative); "
            "  CHEK2 PROSTATE = MODERATE RISK — intensive surveillance, not prophylactic surgery; "
            "  ADT standard treatment — no special pathway modification; "
            "  NO PARPI: CHEK2 is NOT a DDR/HRD gene — olaparib NOT indicated; "
            "  Population screening priority: c.1100delC 1-2% Northern European population; "
            "KEY DISTINCTION: CHEK2 vs BRCA2 — same-sized moderate/moderate elevation but COMPLETELY DIFFERENT treatment: "
            "CHEK2 NO PARPi; BRCA2 strong PARPi. Do NOT conflate."
        ),
        "syndrome": "Li-Fraumeni syndrome type 2 (LFS2, controversial; predominantly moderate-risk cancer predisposition)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "prostate_risk": "2-3x RR; c.1100delC Northern European founder; I157T Central/Eastern European",
        "pathognomonic": "Moderate multi-cancer risk family; c.1100delC Northern European ancestry; NOT HRD",
        "key_avoid": "NO PARPi for CHEK2 prostate — CHEK2 is a checkpoint gene, NOT a HRD gene; Do NOT conflate with BRCA2 treatment strategy",
        "key_rule": "CHEK2 prostate: moderate risk — surveillance intensification, not prophylactic intervention. ADT standard. NO TARGETED THERAPY beyond standard.",
        "surveillance": "Annual PSA + DRE from 40yr; female relatives: annual breast imaging from 25-30yr; colonoscopy from 40yr",
        "targeted_rx": "Standard CRPC regimens (enzalutamide/abiraterone/docetaxel/cabazitaxel); NO PARPi; NO platinum enrichment",
    },
    {
        "gene": "HOXB13",
        "protein": (
            "HOXB13 -- 17q21.32 Autosomal-Dominant-LOF -- 284aa -- "
            "Homeobox-B13-32kDa-Homeodomain-TF-Prostate-Differentiation-"
            "G84E-Founder-Scandinavian-Finnish-Northern-European-"
            "4-8x-RR-Prostate-ONLY-Gene-"
            "NO-Targeted-Therapy-Annual-PSA-40yr-"
            "NO-Other-Cancer-Risk-PROSTATE-SPECIFIC-OMIM-604607"
        ),
        "locus": "17q21.32",
        "protein_size": (
            "284 aa / 32 kDa / 17q21.32 HOXB13 prostate cancer molecular context: "
            "STRUCTURE: "
            "  284 aa / 32 kDa; Homeodomain (aa 199-258): DNA binding TAAT motif; "
            "  N-terminal domain (aa 1-198): protein-protein interaction; "
            "  Transcription factor — prostate epithelium differentiation; "
            "CANCER RISKS (PROSTATE FOCUS): "
            "  PROSTATE: 4-8x RR — highest relative risk prostate-specific gene; "
            "  G84E variant: 1-4% Northern European/Scandinavian/Finnish men; "
            "  PROSTATE-ONLY: no other cancer risk elevated — HOXB13 is prostate-specific; "
            "  Early-onset prostate: <50yr enriched in G84E carriers; "
            "  May 2012 NEJM — Ewing 2012: first major HOXB13 G84E publication; "
            "KEY MANAGEMENT: "
            "  Annual PSA from 40yr (or 35yr if early-onset family); "
            "  HOXB13 G84E: targeted sequencing FIRST (not full gene) — G84E is 95% of pathogenic variants; "
            "  NO TARGETED THERAPY: HOXB13 is NOT a DNA repair gene — no PARPi, no platinum; "
            "  Standard CRPC treatment when disease progresses; "
            "  POPULATION PREVALENCE: G84E 1-4% Scandinavian/Finnish — consider population screening in high-prevalence regions; "
            "KEY DISTINCTION: HOXB13 prostate-ONLY (no breast, ovarian, pancreatic, etc.). Do NOT extend surveillance beyond prostate."
        ),
        "syndrome": "Hereditary Prostate Cancer type 1 (HPCA1 — HOXB13 G84E founder)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "prostate_risk": "4-8x RR; G84E founder Northern European; prostate-ONLY gene",
        "pathognomonic": "Familial prostate cancer Scandinavian/Finnish ancestry; early-onset prostate; G84E 95% HOXB13 variants",
        "key_avoid": "Do NOT extend HOXB13 surveillance beyond prostate — NO other cancer risk; Do NOT use PARPi (not a DDR gene)",
        "key_rule": "HOXB13 = prostate-ONLY gene. G84E targeted test first. Annual PSA from 40yr. NO OTHER CANCER RISK. Standard CRPC treatment.",
        "surveillance": "Annual PSA + DRE from 40yr; targeted HOXB13 G84E test first-degree males; no surveillance for other cancers",
        "targeted_rx": "Standard CRPC regimens only (enzalutamide/abiraterone/docetaxel); NO PARPi; NO platinum enrichment",
    },
    {
        "gene": "MSH2",
        "protein": (
            "MSH2 -- 2p21 Autosomal-Dominant-LOF -- 934aa -- "
            "MutSalpha-MutSbeta-MSH6-MSH3-Scaffold-105kDa-MMR-"
            "Prostate-5-10x-RR-Lynch-Type2-"
            "dMMR-Pembrolizumab-mPRIMO-PATHOGNOMONIC-"
            "Urothelial-14pct-HIGHEST-Lynch-Gene-"
            "EPCAM-3prime-Deletion-MLPA-MANDATORY-OMIM-609309"
        ),
        "locus": "2p21",
        "protein_size": (
            "934 aa / 105 kDa / 2p21 MSH2 prostate cancer molecular context: "
            "STRUCTURE: "
            "  934 aa / 105 kDa; MutSα: MSH2+MSH6 (base substitutions + small indels); "
            "  MutSβ: MSH2+MSH3 (large indels); ATPase domain drives mismatch recognition; "
            "  MSH2 scaffold: loss → MSH6 and MSH3 also lost on IHC (cascade loss); "
            "CANCER RISKS (PROSTATE FOCUS): "
            "  PROSTATE: 5-10x RR Lynch type 2 — dMMR prostate; "
            "  dMMR PROSTATE: pembrolizumab mPRIMO trial; immunotherapy response HIGH; "
            "  COLORECTAL: 52-82% Lynch type 2; "
            "  UROTHELIAL: 14% — HIGHEST Lynch gene for bladder/ureter; "
            "  ENDOMETRIAL: 40-60% in female relatives; "
            "  Muir-Torre: sebaceous gland tumours PATHOGNOMONIC; "
            "KEY MANAGEMENT (PROSTATE): "
            "  Annual PSA + DRE from 40yr; "
            "  IMMUNOTHERAPY FIRST: pembrolizumab mPRIMO — dMMR prostate responds exceptionally; "
            "  TMB-high and MSI-H testing MANDATORY before CRPC treatment decisions; "
            "  EPCAM 3'-deletion: upstream MSH2 silencing — MLPA MANDATORY for MSH2 families; "
            "  UROTHELIAL SURVEILLANCE: annual urine cytology + cystoscopy from 25yr (14% risk); "
            "  Colonoscopy from 25yr annually"
        ),
        "syndrome": "Lynch syndrome type 2 (Hereditary Non-Polyposis Colorectal Cancer type 2) / Muir-Torre variant",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "prostate_risk": "5-10x RR; dMMR prostate; pembrolizumab mPRIMO excellent response",
        "pathognomonic": "dMMR/MSI-H prostate + urothelial tumours + Muir-Torre sebaceous; MSH2+MSH6 IHC loss",
        "key_avoid": "EPCAM 3'-deletion MLPA MANDATORY — MSH2 standard sequencing MISSES EPCAM deletions; Do NOT use PARPi for MSH2 (immunotherapy FIRST)",
        "key_rule": "MSH2 dMMR prostate: pembrolizumab FIRST LINE. EPCAM MLPA MANDATORY. Annual urine cytology from 25yr for urothelial 14%.",
        "surveillance": "Annual PSA + DRE from 40yr; annual urine cytology + cystoscopy from 25yr; annual colonoscopy from 25yr; annual skin exam Muir-Torre",
        "targeted_rx": "Pembrolizumab mPRIMO (dMMR prostate); dostarlimab; standard CRPC regimens; EPCAM MLPA first",
    },
    {
        "gene": "PALB2",
        "protein": (
            "PALB2 -- 16p12.2 Autosomal-Dominant-LOF -- 1186aa -- "
            "WD40-BRCA1-BRCA2-Bridge-131kDa-FANCN-"
            "Prostate-2-4x-RR-Emerging-"
            "PARPi-Sensitive-HRD-Positive-"
            "Breast-53pct-DOMINATES-Phenotype-TBCRC048-82pct-ORR-"
            "FA-N-Biallelic-OMIM-610355"
        ),
        "locus": "16p12.2",
        "protein_size": (
            "1186 aa / 131 kDa / 16p12.2 PALB2 prostate cancer molecular context: "
            "STRUCTURE: "
            "  1186 aa / 131 kDa; BRCA1-binding domain (aa 1-40): N-terminal coiled-coil; "
            "  WD40 repeat domain (aa 853-1186): BRCA2 interaction; "
            "  PALB2 bridges BRCA1-BRCA2: BRCA1 recruits PALB2 recruits BRCA2 at DSBs; "
            "CANCER RISKS (PROSTATE FOCUS): "
            "  PROSTATE: 2-4x RR — emerging, increasing evidence; "
            "  HRD-positive: PALB2 loss → HR defect → PARPi sensitivity (similar to BRCA2 biology); "
            "  BREAST: 53% lifetime DOMINATES phenotype — HIGHEST non-BRCA2 risk; "
            "  BREAST TBCRC048: 82% ORR olaparib in PALB2 metastatic breast; "
            "  OVARIAN: 3-5% moderate; "
            "  PANCREATIC: 3-4x RR; "
            "KEY MANAGEMENT (PROSTATE): "
            "  Annual PSA from 40yr; "
            "  PARPi SENSITIVITY: PALB2 HR-deficient — olaparib expected similar to BRCA2 (not yet separate FDA approval for prostate); "
            "  HRD scar testing (SBS3 signature): confirms PALB2-driven HR defect in tumour; "
            "  ENROL in DDR-selected trials — PALB2 emerging biomarker; "
            "  CASCADE: female relatives PRIMARY — 53% breast; male relatives — annual PSA; "
            "FA-N biallelic: severe Fanconi anaemia — birth defects, AML risk; avoid alkylating agents"
        ),
        "syndrome": "PALB2-associated HBOC (HBOC2) / Fanconi anaemia type N (biallelic)",
        "inheritance": "AD LOF (monoallelic prostate/breast); AR biallelic (Fanconi anaemia type N)",
        "prostate_risk": "2-4x RR; PARPi sensitive (HRD); emerging biomarker for mCRPC trials",
        "pathognomonic": "Female relatives with breast 53% DOMINANT; PALB2 HRD scar SBS3 signature; FA-N biallelic severe",
        "key_avoid": "Do NOT dismiss PALB2 prostate as low-risk — PARPi sensitivity SIMILAR to BRCA2; CASCADE FEMALE RELATIVES (53% breast) PRIMARILY",
        "key_rule": "PALB2: HRD gene — PARPi sensitivity established. Enrol in DDR trials. Annual PSA from 40yr. Female relatives breast surveillance MANDATORY.",
        "surveillance": "Annual PSA + DRE from 40yr; female relatives: annual breast MRI from 25-30yr; pancreatic EUS/MRI from 50yr",
        "targeted_rx": "Olaparib (off-label/trial for prostate PALB2); TBCRC048 data; platinum+docetaxel HRD synergy; enrol DDR-selected trials",
    },
    {
        "gene": "NBN",
        "protein": (
            "NBN -- 8q21.3 Autosomal-Recessive-AD-LOF -- 754aa -- "
            "Nibrin-MRN-Complex-85kDa-FHA-BRCT-Domains-"
            "657del5-Slavic-Polish-Czech-Russian-Founder-"
            "Prostate-3-4x-RR-Heterozygous-"
            "Nijmegen-Breakage-Syndrome-Biallelic-"
            "Radiation-Sensitivity-Intermediate-MRN-Complex-OMIM-602667"
        ),
        "locus": "8q21.3",
        "protein_size": (
            "754 aa / 85 kDa / 8q21.3 NBN prostate cancer molecular context: "
            "STRUCTURE: "
            "  754 aa / 85 kDa; FHA domain (aa 24-109): phosphopeptide binding; "
            "  BRCT domains (aa 110-325): DNA damage signalling; "
            "  C-terminal (aa 665-754): MRE11 binding → MRN complex formation; "
            "  MRN complex: MRE11-RAD50-NBN — DSB sensing, resection, ATM activation; "
            "CANCER RISKS (PROSTATE FOCUS): "
            "  PROSTATE: 3-4x RR heterozygous — 657del5 Slavic founder 1% Polish/Czech population; "
            "  Hereditary prostate cancer: NBN 5-15% familial prostate in Slavic populations; "
            "  NHL and other lymphoid: NBN cancer spectrum biallelic + monoallelic; "
            "  BREAST female: moderate 2-3x RR; "
            "KEY MANAGEMENT (PROSTATE): "
            "  Annual PSA from 40yr; "
            "  RADIATION SENSITIVITY: NBN monoallelic — intermediate (not absolute CI like biallelic); "
            "  Reduce RT dose if treating NBN prostate carriers; "
            "  BIALLELIC NBS: Nijmegen Breakage Syndrome — microcephaly, ID, immunodeficiency, Bird-like face, ABSOLUTELY CI TBI; "
            "  657del5 TARGETED TESTING: Slavic ancestry — 95% of NBN pathogenic variants; "
            "  MRN complex dysregulation: DDR pathway — platinum sensitivity possible; "
            "POPULATION NOTE: 657del5 1% in Polish/Czech men — population-level prevalence in Slavic communities"
        ),
        "syndrome": "Nijmegen Breakage Syndrome (biallelic); hereditary prostate predisposition (monoallelic 657del5)",
        "inheritance": "AR biallelic (NBS); AD monoallelic cancer predisposition",
        "prostate_risk": "3-4x RR heterozygous; 657del5 Slavic founder; prostate + lymphoid spectrum",
        "pathognomonic": "Biallelic NBS: microcephaly + bird-like face + ID + immunodeficiency + café-au-lait; radiosensitivity ABSOLUTE biallelic",
        "key_avoid": "BIALLELIC NBN: TBI ABSOLUTELY CONTRAINDICATED (radiation + alkylating = catastrophic); monoallelic: reduce RT dose; standard Slavic founder 657del5 targeted test",
        "key_rule": "NBN prostate: targeted 657del5 test Slavic ancestry. Intermediate radiation sensitivity monoallelic — reduce RT dose. Annual PSA from 40yr.",
        "surveillance": "Annual PSA + DRE from 40yr; targeted 657del5 test Slavic males; biallelic NBS: annual brain MRI + immunology + CBC; monoallelic female: breast annual MRI",
        "targeted_rx": "Standard CRPC regimens; platinum + docetaxel (DDR synergy emerging); no specific approved targeted therapy; radiation dose-reduce monoallelic",
    },
]

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]

_TUMOUR_TYPES = {
    "BRCA2": ["High-grade prostate adenocarcinoma", "Metastatic CRPC", "De-novo metastatic prostate", "Lethal prostate BRCA2", "Male breast adenocarcinoma"],
    "BRCA1": ["Prostate adenocarcinoma ERG-negative", "Metastatic CRPC BRCA1", "High-grade prostate", "Female breast TNBC (relative)", "Uterine serous (female relative)"],
    "ATM":   ["Prostate adenocarcinoma ATM-deficient", "Metastatic CRPC ATM", "Intermediate-grade prostate", "Breast adenocarcinoma (female relative)", "CRPC ATM-deficient"],
    "CHEK2": ["Prostate adenocarcinoma CHEK2", "Bilateral prostate (metachronous)", "High-grade prostate CHEK2", "Breast IDC (female relative)", "CRPC CHEK2"],
    "HOXB13":["Hereditary prostate G84E", "Early-onset prostate <50yr", "High-volume prostate HOXB13", "Clinically significant prostate", "Familial prostate HOXB13"],
    "MSH2":  ["dMMR prostate MSH2", "High-grade prostate Lynch", "Metastatic dMMR prostate", "Urothelial cancer (bladder)", "Urothelial cancer (ureter)"],
    "PALB2": ["Prostate adenocarcinoma PALB2", "HRD-positive prostate PALB2", "Metastatic CRPC PALB2", "Breast IDC (female relative)", "Pancreatic ductal adenocarcinoma (relative)"],
    "NBN":   ["Hereditary prostate NBN", "Prostate adenocarcinoma 657del5", "Intermediate/high-grade prostate", "NHL Slavic family", "Prostate + lymphoma family"],
}

_VARIANTS_BY_GENE = {
    "BRCA2": ["c.5946delT (p.Ser1982Argfs)", "c.9976A>T (p.Lys3326Ter)", "c.7007G>A (p.Gly2336Asp)", "c.8351G>C (p.Arg2784Thr)", "Large exonic deletion BRCA2"],
    "BRCA1": ["c.5266dupC (p.Gln1756fs)", "c.181T>G (p.Cys61Gly)", "c.4035delA (p.Asn1345Lysfs)", "Large deletion BRCA1 exons 11-13", "c.3700_3704del5 BRCA1"],
    "ATM":   ["c.7271T>G (p.Val2424Gly)", "c.6095G>A (p.Gly2032Glu)", "c.5557G>T (p.Glu1853Ter)", "c.3161C>G (p.Pro1054Arg)", "c.748C>T (p.Arg250Cys)"],
    "CHEK2": ["c.1100delC (p.Thr367Metfs)", "c.470T>C (p.Ile157Thr)", "c.1283C>T (p.Ser428Phe)", "c.319+2T>A splice CHEK2", "Large exon deletion CHEK2 exon 10"],
    "HOXB13":["c.251G>A (p.Gly84Glu — G84E founder)", "c.91C>T (p.Arg31Cys)", "c.140T>C (p.Leu47Pro)", "c.203C>T (p.Ala68Val)", "c.275G>T (p.Arg92Ile)"],
    "MSH2":  ["c.1906G>C (p.Ala636Pro)", "c.388_389delGA (p.Glu130Asnfs)", "Large deletion MSH2 exons 1-6", "EPCAM 3'-end deletion upstream", "c.942+3A>T splice MSH2"],
    "PALB2": ["c.2257C>T (p.Gln753Ter)", "c.3113G>A (p.Trp1038Ter)", "c.1592delT (p.Leu531Cysfs)", "c.509_510delGA (p.Arg170Ilefs)", "Large deletion PALB2"],
    "NBN":   ["c.657_661del5 (p.Lys219Asnfs — 657del5 founder)", "c.698_701delAACA (p.Lys233Asnfs)", "c.511A>G (p.Ile171Val)", "c.643C>T (p.Arg215Trp)", "Large deletion NBN exon 6"],
}

TREATMENT_PROTOCOLS_BY_GENE = {
    "BRCA2": ["Olaparib PROfound FDA2020 mCRPC HR 0.22", "Rucaparib TRITON2 mCRPC", "Carboplatin+docetaxel HRD synergy", "Cabazitaxel BRCA2 enriched response", "PSMA-PET staging MANDATORY"],
    "BRCA1": ["Olaparib PROfound FDA2020 mCRPC HR 0.82 (modest)", "Standard CRPC enzalutamide/abiraterone", "Carboplatin+docetaxel standard", "Cabazitaxel standard 2nd line", "Annual PSA cascade female relatives"],
    "ATM":   ["Olaparib PROfound mCRPC HR 0.72 (modest ATM)", "Ceralasertib ATRi trials ATM-deficient", "Platinum+docetaxel DDR synergy", "Standard enzalutamide/abiraterone", "RT dose-reduce monoallelic carriers"],
    "CHEK2": ["Standard enzalutamide/abiraterone", "Standard docetaxel/cabazitaxel", "No PARPi — NOT a HRD gene", "ADT standard", "Annual PSA surveillance from 40yr"],
    "HOXB13":["Standard enzalutamide/abiraterone", "Standard docetaxel/cabazitaxel", "No targeted therapy available", "ADT standard CRPC", "Annual PSA from 40yr — G84E targeted test"],
    "MSH2":  ["Pembrolizumab mPRIMO dMMR prostate", "Dostarlimab MSI-H", "Ipilimumab+nivolumab dMMR prostate emerging", "Annual urine cytology urothelial 14%", "EPCAM MLPA MANDATORY"],
    "PALB2": ["Olaparib off-label/trial PALB2 prostate HRD", "Platinum+docetaxel HRD synergy", "DDR-selected trial enrollment priority", "Standard enzalutamide/abiraterone", "Female relatives TBCRC048 data breast"],
    "NBN":   ["Standard CRPC regimens", "Platinum+docetaxel DDR emerging", "RT dose-reduce monoallelic", "No specific approved targeted therapy", "657del5 targeted test Slavic ancestry"],
}

SURVEILLANCE_BY_GENE = {
    "BRCA2": ["Annual PSA + DRE from 40yr (35yr family history)", "PSMA-PET at diagnosis/staging", "Cascade male first-degree relatives", "Female relatives: RRSO + breast MRI", "Annual breast exam male breast 6%"],
    "BRCA1": ["Annual PSA + DRE from 40yr", "Female relatives: annual breast MRI + RRSO 35-40yr MANDATORY", "Cascade all first-degree relatives", "No PSMA-PET routine (BRCA1 weaker prostate)", "Annual male breast exam (lower risk than BRCA2)"],
    "ATM":   ["Annual PSA + DRE from 40yr", "Female relatives: annual breast MRI from 30yr", "Biallelic A-T: annual brain MRI + immunology + AFP", "RT consultation: dose-reduce for monoallelic", "DDR trial eligibility review"],
    "CHEK2": ["Annual PSA + DRE from 40yr (5yr earlier than youngest affected)", "Female relatives: annual breast imaging from 25-30yr", "Colonoscopy from 40yr", "No urothelial surveillance", "Population carrier testing c.1100delC Northern European"],
    "HOXB13":["Annual PSA + DRE from 40yr (35yr if strong family history)", "First-degree male relatives: targeted G84E test", "No other cancer surveillance (prostate-ONLY gene)", "No female relative cancer risk from HOXB13", "Population screening Scandinavian/Finnish populations"],
    "MSH2":  ["Annual PSA + DRE from 40yr", "Annual urine cytology + cystoscopy from 25yr (urothelial 14%)", "Annual colonoscopy from 25yr", "Annual skin exam Muir-Torre sebaceous", "EPCAM deletion MLPA MANDATORY cascade"],
    "PALB2": ["Annual PSA + DRE from 40yr", "Female relatives: annual breast MRI from 25-30yr MANDATORY", "Pancreatic EUS/MRI from 50yr", "DDR trial eligibility review", "FA-N biallelic: haematology + CBC monitoring"],
    "NBN":   ["Annual PSA + DRE from 40yr", "657del5 targeted test Slavic males first-degree", "Biallelic NBS: annual brain MRI + immunology", "Monoallelic female relatives: annual breast MRI", "RT consultation: intermediate sensitivity"],
}


def _make_patients(gene: str, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    g = next(g for g in ATLAS_GENES if g["gene"] == gene)
    tumours = _TUMOUR_TYPES.get(gene, ["Prostate cancer NOS"])
    variants = _VARIANTS_BY_GENE.get(gene, ["Pathogenic variant"])
    pts = []
    for i in range(n):
        age = rng.randint(45, 80)
        pts.append({
            "patient_id": f"{gene[:3]}-HPCA-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_dx": age,
            "tumour_type": rng.choice(tumours),
            "variant": rng.choice(variants),
            "stage": rng.choice(["Localised", "Locally Advanced", "Metastatic", "mCRPC"]),
            "parpi_eligible": rng.random() < (0.85 if gene == "BRCA2" else 0.50 if gene in ("BRCA1","ATM","PALB2") else 0.10),
            "immunotherapy_eligible": rng.random() < (0.75 if gene == "MSH2" else 0.10),
            "radiation_given": rng.random() < (0.20 if gene in ("ATM","NBN") else 0.55),
            "relapse": rng.random() < 0.45,
            "psma_pet_done": rng.random() < (0.80 if gene == "BRCA2" else 0.35),
        })
    return pts


def generate_overview() -> dict:
    cohorts = {}
    for i, g in enumerate(ATLAS_GENES):
        gene = g["gene"]
        pts = _make_patients(gene, SEED_BASE + i)
        cohorts[gene] = pts

    total = sum(len(v) for v in cohorts.values())
    gene_counts = {g: len(pts) for g, pts in cohorts.items()}
    overall_parpi_rate = round(
        100 * sum(p["parpi_eligible"] for pts in cohorts.values() for p in pts) / total, 1
    )
    overall_immuno_rate = round(
        100 * sum(p["immunotherapy_eligible"] for pts in cohorts.values() for p in pts) / total, 1
    )
    psma_rate = round(
        100 * sum(p["psma_pet_done"] for pts in cohorts.values() for p in pts) / total, 1
    )
    mean_age = round(
        sum(p["age_at_dx"] for pts in cohorts.values() for p in pts) / total, 1
    )
    metastatic_pct = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["stage"] in ("Metastatic", "mCRPC")) / total, 1
    )
    atm_nbn_rt_rate = round(
        100 * sum(p["radiation_given"] for g in ("ATM","NBN") for p in cohorts.get(g,[])) /
        max(sum(len(cohorts.get(g,[])) for g in ("ATM","NBN")), 1), 1
    )

    return {
        "atlas": "Hereditary-Prostate-Cancer-Predisposition-Atlas",
        "genes": _GENE_LIST,
        "total_patients": total,
        "gene_counts": gene_counts,
        "seed_range": f"{SEED_BASE}-{SEED_BASE+7}",
        "parpi_eligible_rate_pct": overall_parpi_rate,
        "immunotherapy_eligible_rate_pct": overall_immuno_rate,
        "psma_pet_done_rate_pct": psma_rate,
        "mean_age_at_dx": mean_age,
        "metastatic_pct": metastatic_pct,
        "atm_nbn_rt_avoidance_note": f"ATM+NBN cohort: radiation_given {atm_nbn_rt_rate}% — target <30% in DDR carriers (reduce dose)",
        "key_facts": [
            "BRCA2: 15-20x RR prostate; lethal/metastatic enriched; olaparib PROfound HR 0.22; PSMA-PET MANDATORY",
            "BRCA1 vs BRCA2: same gene family but PARPi HR 0.82 vs 0.22 — CRITICAL DISTINCTION for treatment",
            "ATM: 2-4x RR; olaparib modest HR 0.72; ceralasertib ATRi preferred; radiation sensitivity INTERMEDIATE (not absolute CI)",
            "CHEK2: moderate 2-3x RR; NO PARPi — NOT a HRD gene; ADT standard; c.1100delC Northern European 1-2%",
            "HOXB13 G84E: 4-8x RR prostate-ONLY gene; NO other cancer risk; targeted G84E test first; Scandinavian/Finnish founder",
            "MSH2: dMMR prostate 5-10x RR; pembrolizumab mPRIMO FIRST LINE; urothelial 14% HIGHEST Lynch urothelial",
            "PALB2: HRD gene like BRCA2; 2-4x RR prostate emerging; PARPi sensitivity; female relatives breast 53% DOMINANT",
            "NBN 657del5: Slavic founder 1%; 3-4x RR; intermediate radiation sensitivity; NBS biallelic — TBI ABSOLUTELY CI",
        ],
    }


def generate_breakdown() -> dict:
    breakdown = {}
    from collections import Counter
    for i, g in enumerate(ATLAS_GENES):
        gene = g["gene"]
        pts = _make_patients(gene, SEED_BASE + i)
        tumour_counts = Counter(p["tumour_type"] for p in pts)
        variant_counts = Counter(p["variant"] for p in pts)
        top_tumours = tumour_counts.most_common(3)
        top_variants = variant_counts.most_common(3)
        breakdown[gene] = {
            "gene_info": {
                "gene": gene,
                "protein": g["protein"],
                "locus": g["locus"],
                "syndrome": g["syndrome"],
                "inheritance": g["inheritance"],
                "prostate_risk": g["prostate_risk"],
                "pathognomonic": g["pathognomonic"],
                "key_avoid": g["key_avoid"],
                "key_rule": g["key_rule"],
                "surveillance": g["surveillance"],
                "targeted_rx": g["targeted_rx"],
            },
            "n": len(pts),
            "parpi_eligible_pct": round(100 * sum(1 for p in pts if p["parpi_eligible"]) / len(pts), 1),
            "immunotherapy_eligible_pct": round(100 * sum(1 for p in pts if p["immunotherapy_eligible"]) / len(pts), 1),
            "psma_pet_pct": round(100 * sum(1 for p in pts if p["psma_pet_done"]) / len(pts), 1),
            "relapse_pct": round(100 * sum(1 for p in pts if p["relapse"]) / len(pts), 1),
            "mean_age": round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "top_tumour_types": [{"type": t, "count": c} for t, c in top_tumours],
            "top_variants": [{"variant": v, "count": c} for v, c in top_variants],
            "treatment_protocols": TREATMENT_PROTOCOLS_BY_GENE[gene],
            "surveillance_protocols": SURVEILLANCE_BY_GENE[gene],
        }
    return {"breakdown": breakdown, "genes": _GENE_LIST}


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Prostate-Cancer-Predisposition-Atlas",
        "definitions": {
            "brca2_prostate_lethal": (
                "BRCA2 HBOC prostate: FANCD1-HR-Mediator 3418aa — prostate 15-20x RR; "
                "LETHAL PHENOTYPE ENRICHED: BRCA2 prostate = higher grade, more metastatic at diagnosis, shorter survival; "
                "OLAPARIB PROfound: HR 0.22 BRCA2 cohort mCRPC — FDA 2020; rucaparib TRITON2 also approved; "
                "PSMA-PET MANDATORY: BRCA2 prostate micro-metastases missed by CT/bone scan; "
                "MALE BREAST 6%: higher than general population (0.1%) — annual breast exam"
            ),
            "brca1_vs_brca2_prostate": (
                "BRCA1 vs BRCA2 PROSTATE CRITICAL DISTINCTION: "
                "BRCA2: 15-20x RR prostate; olaparib HR 0.22 DRAMATIC; LETHAL enriched; PSMA-PET MANDATORY; "
                "BRCA1: 2-3x RR prostate; olaparib HR 0.82 MODEST; standard-risk biology; "
                "DO NOT conflate: same PARPi but very different efficacy; "
                "Female relatives: BRCA1 DOMINANT (breast 70-80%, ovarian 39-44%) — RRSO + MRI cascade"
            ),
            "atm_chek2_distinction": (
                "ATM vs CHEK2 CRITICAL DISTINCTION: "
                "ATM: DNA repair kinase — HR pathway adjacent; olaparib HR 0.72 modest PROfound; ceralasertib ATRi trials; "
                "ATM: intermediate radiation sensitivity monoallelic — REDUCE RT DOSE; biallelic = A-T RADIATION ABSOLUTELY CI; "
                "CHEK2: checkpoint signalling only — NOT a HRD gene; NO PARPi; standard prostate treatment; "
                "CHEK2 c.1100delC: 1-2% Northern European population — moderate risk"
            ),
            "hoxb13_prostate_only": (
                "HOXB13 G84E: Homeobox transcription factor — prostate-ONLY cancer gene; "
                "G84E FOUNDER: 1-4% Scandinavian/Finnish/Northern European men; 4-8x RR prostate; "
                "PROSTATE-ONLY: NO other cancer elevation — DO NOT extend surveillance beyond prostate; "
                "NO TARGETED THERAPY: HOXB13 is NOT a DNA repair gene; standard treatment; "
                "TARGETED TEST FIRST: G84E accounts for 95% HOXB13 pathogenic variants — targeted test before full sequencing"
            ),
            "msh2_dmm_prostate": (
                "MSH2 Lynch type 2: MutSα/β scaffold — prostate 5-10x RR dMMR/MSI-H; "
                "PEMBROLIZUMAB mPRIMO TRIAL: dMMR prostate responds exceptionally to checkpoint inhibitor; "
                "IMMUNOTHERAPY FIRST LINE: dMMR prostate — pembrolizumab before PARPi consideration; "
                "UROTHELIAL 14%: HIGHEST Lynch gene for bladder/ureter — annual urine cytology MANDATORY; "
                "EPCAM 3'-deletion MLPA MANDATORY: standard MSH2 sequencing misses EPCAM upstream deletions"
            ),
            "palb2_hrd_prostate": (
                "PALB2 HBOC2: BRCA1-BRCA2 bridge 1186aa — prostate 2-4x RR HRD-positive; "
                "HRD GENE: PALB2 loss = HR defect — PARPi sensitivity similar to BRCA2 biology; "
                "TBCRC048 82% ORR: olaparib breast — same HRD mechanism applies prostate; "
                "BREAST 53% DOMINANT: female relatives breast risk DOMINATES PALB2 phenotype — cascade female FIRST; "
                "FA-N BIALLELIC: severe Fanconi anaemia — avoid alkylating agents"
            ),
            "nbn_657del5_slavic": (
                "NBN 657del5: Nibrin MRN complex 754aa — 3-4x RR prostate monoallelic; "
                "SLAVIC FOUNDER: 657del5 1% Polish/Czech/Russian population — prevalent Slavic ancestry; "
                "INTERMEDIATE RADIATION SENSITIVITY: monoallelic NBN — reduce RT dose (not absolute CI); "
                "NBS BIALLELIC: microcephaly + bird-like face + ID + immunodeficiency + TBI ABSOLUTELY CONTRAINDICATED; "
                "MRN COMPLEX: NBN-MRE11-RAD50 — DSB sensing; ATM activation impaired"
            ),
            "cascade_testing": (
                "CASCADE TESTING Hereditary Prostate Cancer Predisposition: "
                "BRCA2/BRCA1: all first-degree — males PSA from 40yr; females RRSO + breast MRI; "
                "ATM: males PSA from 40yr; females breast MRI from 30yr; biallelic: annual brain MRI; "
                "CHEK2: males PSA from 40yr; females breast annual imaging from 25-30yr; "
                "HOXB13: male first-degree G84E targeted test; NO female surveillance; "
                "MSH2: colonoscopy 25yr; urine cytology 25yr; skin exam (Muir-Torre); "
                "PALB2: females breast MRI MANDATORY from 25-30yr; males PSA 40yr; "
                "NBN: Slavic ancestry 657del5 targeted test; males PSA 40yr; RT consultation"
            ),
        },
        "key_clinical_distinctions": [
            "BRCA2 15-20x RR: lethal prostate enriched; olaparib HR 0.22; PSMA-PET MANDATORY — highest-impact germline gene",
            "BRCA1 vs BRCA2: PARPi HR 0.82 vs 0.22 — CRITICAL; counsel female relatives primarily for BRCA1",
            "ATM vs CHEK2: ATM = DDR gene (PARPi modest); CHEK2 = checkpoint only (NO PARPi) — treatment diverges completely",
            "HOXB13 G84E: prostate-ONLY — 4-8x RR but NO other cancer; DO NOT extend surveillance; targeted G84E test first",
            "MSH2 dMMR prostate: pembrolizumab FIRST LINE (mPRIMO); urothelial 14% HIGHEST Lynch — annual cystoscopy",
            "PALB2: HRD gene — PARPi sensitivity like BRCA2; female relatives breast 53% DOMINANT — cascade females first",
            "NBN 657del5: Slavic founder 1%; intermediate RT sensitivity monoallelic; NBS biallelic TBI ABSOLUTELY CI",
            "Universal germline testing: all mCRPC patients AND high-risk localised prostate — identifies DDR/Lynch for targeted therapy",
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2))
