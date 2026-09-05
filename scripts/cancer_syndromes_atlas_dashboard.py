#!/usr/bin/env python3
"""Hereditary Cancer Syndromes Atlas — Complete 8-Gene Hereditary Cancer Predisposition Atlas
BRCA1   (Hereditary Breast–Ovarian Cancer type 1 — 1863 aa; 17q21.31; PALB2 partner;
         olaparib/PARP inhibitors FDA 2018; bilateral RRM + RRSO; risk-reducing surgery) ·
BRCA2   (Hereditary Breast–Ovarian Cancer type 2 — 3418 aa; 13q12.3; RAD51 loader;
         male breast 6%; pancreatic 5-7%; olaparib/rucaparib/niraparib FDA 2016+) ·
MLH1    (Lynch syndrome — 756 aa; 3p22.2; MutL alpha heterodimer; universal MMR IHC;
         MSI-H colorectal + endometrial + ovarian; aspirin chemopreventive Level A) ·
MSH2    (Lynch syndrome / EPCAM — 934 aa; 2p21; MutS alpha; urothelial 14%;
         EPCAM 3′ deletion → MSH2 epigenetic silencing; annual urine cytology) ·
TP53    (Li-Fraumeni syndrome — 393 aa; 17p13.1; guardian of genome; radiation ABSOLUTELY CI;
         whole-body MRI annually; choroid plexus carcinoma PATHOGNOMONIC child LFS) ·
RB1     (Hereditary retinoblastoma — 928 aa; 13q14.2; retinoblastoma protein; leukocoria
         PATHOGNOMONIC; trilateral (pineoblastoma) 5%; radiation CI → osteosarcoma;
         red-reflex exam at birth mandatory) ·
CDKN2A  (Familial atypical multiple-mole melanoma + pancreatic — 156 aa; 9p21.3; p16/INK4a;
         CDK4/6 brake; multiple atypical nevi PATHOGNOMONIC; pancreatic cancer 17-39x risk;
         whole-body dermoscopy + EUS/MRCP annually) ·
APC     (Familial adenomatous polyposis — 2843 aa; 5q22.2; β-catenin destruction complex;
         ≥100 colonic adenomas PATHOGNOMONIC; prophylactic colectomy 20-25 yo;
         sulindac/celecoxib NOT curative; CHRPE ocular marker; desmoid tumours >codon 1310)
320-patient aggregate cohort (8 × 40, seeds 1198–1205)
"""

import random

SEED_BASE = 1198

CANCER_GENES = [
    # ── BRCA1 — Hereditary Breast–Ovarian Cancer 1 ──────────────────────────
    {
        "gene": "BRCA1",
        "protein": "BRCA1 DNA repair associated protein",
        "alias": (
            "BRCA1; OMIM gene 113705; 17q21.31; 1863 aa; HBOC1 OMIM #604370; "
            "AD; lifetime breast cancer risk 70-80%; ovarian cancer 44-46%; "
            "RING/BRCT domain — ubiquitin E3 ligase + DNA damage sensor"
        ),
        "aa": "1863 aa",
        "kDa": "~220 kDa",
        "gene_class": (
            "Tumour suppressor — RING domain (E3 ubiquitin ligase, BARD1 heterodimer), "
            "coiled-coil nuclear export, BRCT domain (phosphoprotein binding — pS1524/ATM); "
            "recruits RAD51 via PALB2/BRCA2 for homologous recombination (HR) double-strand break repair; "
            "LOF → HR deficiency → replication fork instability → chromosome breaks; "
            "BRCA1 mutations: 185delAG (Ashkenazi Jewish founder, 1% carrier frequency), "
            "5382insC (Ashkenazi); frameshift/nonsense most common (60-70%); "
            "large genomic rearrangements (MLPA mandatory — Sanger misses 5-10%)"
        ),
        "locus": "17q21.31",
        "omim_gene": 113705,
        "omim_disease": 604370,
        "phenotype": (
            "Breast cancer (lifetime 70-80%): triple-negative subtype most common (60-80%); "
            "younger age at onset (median 40-45 vs 60 general population); "
            "bilateral risk (~30% contralateral at 10 years after first BC); "
            "Ovarian cancer (lifetime 44-46%): high-grade serous carcinoma (HGSC) predominant; "
            "fallopian tube and primary peritoneal cancer included; "
            "Male breast cancer: 1-2% lifetime (vs 0.1% general); "
            "Pancreatic cancer: 2-3x relative risk; "
            "Prostate cancer (2x RR, aggressive); "
            "Contralateral breast cancer risk ~30% without preventive surgery"
        ),
        "hallmark": (
            "TRIPLE-NEGATIVE breast cancer + YOUNG onset (<50) + family history = BRCA1 until proven otherwise; "
            "BRCA1-associated HGSC ovarian: platinum-sensitive at first line; "
            "ASHKENAZI JEWISH ancestry: 185delAG + 5382insC account for 95% BRCA1 mutations — "
            "direct testing without full panel saves time; "
            "BILATERAL BREAST CANCER risk ~30%: contralateral risk-reducing mastectomy discussion mandatory; "
            "MLPA for large rearrangements: 5-10% BRCA1 mutations are large deletions missed by sequencing"
        ),
        "treatment_alert": (
            "OLAPARIB (PARP inhibitor) FDA 2018 — metastatic BRCA1/2 HER2- breast cancer; "
            "olaparib + pembrolizumab early TNBC neoadjuvant combinations; "
            "RISK-REDUCING BILATERAL SALPINGO-OOPHORECTOMY (RRSO) 35-40 yo "
            "(after childbearing): reduces ovarian cancer risk 80-96%; also reduces breast cancer risk 50% if pre-menopausal; "
            "RISK-REDUCING MASTECTOMY (RRM): reduces breast cancer risk by 90-95%; "
            "ANNUAL MRI + mammography from age 25-30 (MRI higher sensitivity in dense breasts); "
            "PLATINUM chemotherapy: BRCA1/2 tumours hypersensitive to cisplatin/carboplatin "
            "(HR deficiency = BRCAness); "
            "CHEMOPREVENTION: tamoxifen/raloxifene reduce ER+ risk but BRCA1 tumours are mostly ER-; "
            "GERMLINE vs SOMATIC BRCA1: germline requires family cascade testing; somatic does not"
        ),
        "key_ddx": (
            "BRCA2 (TNBC less common, pancreatic more common); "
            "PALB2 (moderate risk, similar HR deficiency); "
            "TP53 Li-Fraumeni (childhood cancers, radiation CI); "
            "RAD51C/RAD51D (ovarian dominant); "
            "PTEN Cowden (breast + thyroid + GI hamartomas)"
        ),
        "gfr_pattern": "N/A — not a nephropathy gene",
        "proteinuria_pattern": "N/A",
        "primary_complication": "Breast/ovarian cancer; bilateral cancer risk; treatment decisions",
        "disease_detail": (
            "Hereditary Breast–Ovarian Cancer type 1 (HBOC1): BRCA1 germline pathogenic "
            "variants follow Knudson two-hit model (AD predisposition, AR second hit somatic). "
            "BRCA1 185delAG: 1% Ashkenazi carrier frequency. "
            "Mean age breast cancer ~42 years; ovarian cancer ~50 years. "
            "PARP inhibitors exploit HR deficiency (synthetic lethality): "
            "olaparib, rucaparib, niraparib, talazoparib all approved in various settings. "
            "HbOC surveillance: annual MRI+mammography from 25; RRSO 35-40 yo; "
            "cascade testing all first-degree relatives."
        ),
        "inheritance": "Autosomal dominant (AD)",
        "variants": [
            {"name": "185delAG (c.68_69del)", "frequency": "~1% Ashkenazi; 25-35% BRCA1 Ashkenazi"},
            {"name": "5382insC (c.5266dup)", "frequency": "~1.5% Ashkenazi; 30-40% BRCA1 Ashkenazi"},
            {"name": "4153delA (Scottish/N. European)", "frequency": "5-8% of BRCA1 mutations"},
            {"name": "Large genomic rearrangements (exon del/dup)", "frequency": "5-10% BRCA1 mutations (MLPA required)"},
        ],
        "drug_ci": [
            "OLAPARIB: nausea/anaemia expected — not CI but monitor CBC monthly",
            "TAMOXIFEN: limited benefit in BRCA1 (mostly ER-negative tumours); discuss carefully",
            "HORMONE REPLACEMENT THERAPY post-RRSO: short-term HRT acceptable for menopausal symptoms — does NOT negate ovarian cancer reduction benefit",
        ],
    },

    # ── BRCA2 — Hereditary Breast–Ovarian Cancer 2 ──────────────────────────
    {
        "gene": "BRCA2",
        "protein": "BRCA2 DNA repair associated protein",
        "alias": (
            "BRCA2; OMIM gene 600185; 13q12.3; 3418 aa; HBOC2 OMIM #612555; "
            "AD; lifetime breast 69-75%; ovarian 17-24%; pancreatic 5-7%; male breast 6%; "
            "RAD51 paralogue loader — homologous recombination"
        ),
        "aa": "3418 aa",
        "kDa": "~384 kDa",
        "gene_class": (
            "Tumour suppressor — BRC repeats (8×, bind RAD51 monomers); "
            "C-terminal DNA-binding domain; OB-fold (ssDNA); Tower domain; "
            "BRCA2 loads RAD51 onto RPA-coated ssDNA at resected DSB ends → homologous recombination; "
            "LOF → RAD51 cannot be loaded → HR deficiency → error-prone NHEJ → chromosome instability; "
            "Biallelic BRCA2 LOF → Fanconi anaemia complementation group D1 (FA-D1): "
            "bone marrow failure + AML + medulloblastoma + Wilms tumour in children; "
            "6174delT: most common Ashkenazi BRCA2 mutation (~1.5% carrier frequency)"
        ),
        "locus": "13q12.3",
        "omim_gene": 600185,
        "omim_disease": 612555,
        "phenotype": (
            "Breast cancer (lifetime 69-75%): ER+ predominant (vs BRCA1 triple-negative); "
            "male breast cancer 6% lifetime (100-fold increased vs 0.06% general); "
            "Ovarian cancer (lifetime 17-24%): HGSC like BRCA1 but lower absolute risk; "
            "Pancreatic adenocarcinoma (lifetime 5-7%; 3.5-10x RR): BRCA2 most common "
            "hereditary gene for pancreatic cancer; "
            "Prostate cancer (8-9% lifetime; 8x RR aggressive disease — GGG 3-5); "
            "Melanoma (2x RR); "
            "Biallelic BRCA2: Fanconi anaemia — bone marrow failure, VACTERL, medulloblastoma"
        ),
        "hallmark": (
            "MALE BREAST CANCER + family history = BRCA2 until proven otherwise (6% vs BRCA1 1-2%); "
            "PANCREATIC CANCER at young age (<60) with BRCA2: olaparib maintenance post-platinum; "
            "PROSTATE CANCER: BRCA2 → aggressive disease GGG≥3; olaparib FDA 2020 mCRPC; "
            "BIALLELIC BRCA2: Fanconi anaemia — ALWAYS consider in aplastic anaemia + cancer in child; "
            "6174delT: Ashkenazi founder (1.5% carrier); screen Ashkenazi three-site panel first"
        ),
        "treatment_alert": (
            "OLAPARIB FDA 2018 (breast), FDA 2019 (ovarian maintenance), FDA 2020 (pancreatic+prostate); "
            "RUCAPARIB, NIRAPARIB, TALAZOPARIB all approved in BRCA1/2 settings; "
            "PLATINUM SENSITIVITY: BRCA2 pancreatic and ovarian tumours respond to platinum; "
            "maintenance olaparib after platinum response in pancreatic BRCA2 (POLO trial); "
            "PROSTATE: olaparib + abiraterone combination approved (PROpel trial); "
            "MALE BREAST CANCER surveillance: annual mammography from age 35-40; "
            "RRSO: discuss 40-45yo for ovarian cancer risk reduction; "
            "PANCREATIC SURVEILLANCE: annual EUS + MRCP from age 50 or 10 years before earliest affected relative"
        ),
        "key_ddx": (
            "BRCA1 (TNBC dominant, higher ovarian risk); "
            "PALB2 (moderate breast + pancreatic risk); "
            "ATM (moderate breast, pancreatic, prostate); "
            "STK11 (Peutz-Jeghers — pancreatic + GI polyps); "
            "CDKN2A (FAMMM — pancreatic + melanoma)"
        ),
        "gfr_pattern": "N/A",
        "proteinuria_pattern": "N/A",
        "primary_complication": "Breast/ovarian/pancreatic/prostate cancer; PARP inhibitor therapy",
        "disease_detail": (
            "Hereditary Breast–Ovarian Cancer type 2 (HBOC2): larger protein (3418 aa) "
            "than BRCA1; similar surveillance principles but distinct cancer spectrum. "
            "Male breast cancer 6% lifetime: BRCA2 accounts for 10-20% of male breast cancer. "
            "Pancreatic cancer: olaparib maintenance post-platinum (POLO trial, FDA 2019). "
            "Biallelic BRCA2 = Fanconi anaemia complementation group D1 (FA-D1)."
        ),
        "inheritance": "Autosomal dominant (AD) [biallelic = Fanconi FA-D1, AR]",
        "variants": [
            {"name": "6174delT (c.5946delT)", "frequency": "~1.5% Ashkenazi; most common BRCA2 Ashkenazi"},
            {"name": "999del5 (Icelandic founder)", "frequency": "40% of Icelandic BRCA2"},
            {"name": "Large genomic rearrangements", "frequency": "2-5% BRCA2 mutations (MLPA required)"},
        ],
        "drug_ci": [
            "OLAPARIB + strong CYP3A4 inhibitors (itraconazole, clarithromycin): reduce olaparib dose to 100 mg BD",
            "AVOID: MDS/AML risk with prolonged PARP inhibitor — monitor CBC quarterly",
        ],
    },

    # ── MLH1 — Lynch Syndrome ────────────────────────────────────────────────
    {
        "gene": "MLH1",
        "protein": "MutL protein homolog 1",
        "alias": (
            "MLH1; OMIM gene 120436; 3p22.2; 756 aa; Lynch syndrome OMIM #120435; "
            "AD; most common Lynch gene; CRC lifetime 25-70%; endometrial 25-60%; "
            "MutL alpha heterodimer (MLH1-PMS2); mismatch repair"
        ),
        "aa": "756 aa",
        "kDa": "~85 kDa",
        "gene_class": (
            "Mismatch repair (MMR) — MutL alpha heterodimer (MLH1+PMS2); "
            "MLH1: ATPase N-domain + endonuclease C-domain; "
            "also forms MutL beta (MLH1+PMS1) and MutL gamma (MLH1+MLH3) — meiotic crossing over; "
            "MLH1 LOF → MMR failure → microsatellite instability (MSI-H) → accumulation "
            "of thousands of insertion/deletion mutations in microsatellite repeats; "
            "MMR-deficient tumours: HIGH tumour mutational burden (TMB-H) → immunogenic → "
            "respond to immune checkpoint inhibitors (pembrolizumab FDA 2017 — MSI-H/dMMR); "
            "MLH1 promoter hypermethylation: sporadic MSI-H CRC (not hereditary Lynch); "
            "IHC loss MLH1/PMS2 together: promoter methylation (sporadic) vs germline MLH1 (Lynch)"
        ),
        "locus": "3p22.2",
        "omim_gene": 120436,
        "omim_disease": 120435,
        "phenotype": (
            "Colorectal cancer (CRC): lifetime 25-70%; right-sided predominance (70%); "
            "MSI-H histology (Crohn-like reaction, mucinous); synchronous/metachronous lesions; "
            "Endometrial cancer: lifetime 25-60% — most common extra-colonic Lynch tumour; "
            "Ovarian cancer: 5-12% lifetime; "
            "Gastric cancer (1-13%); "
            "Hepatobiliary (2-7%); "
            "Urological (2-4%); "
            "Sebaceous skin neoplasms (Muir-Torre variant — sebaceous adenoma PATHOGNOMONIC mucocutaneous); "
            "Brain tumours (Turcot variant — glioblastoma/medulloblastoma)"
        ),
        "hallmark": (
            "AMSTERDAM II CRITERIA (clinical diagnosis before genetic era): "
            "3+ relatives with Lynch tumour, 2+ generations, 1+ diagnosed <50; "
            "UNIVERSAL TUMOR TESTING — IHC for MLH1/MSH2/MSH6/PMS2 on ALL colorectal + endometrial cancers "
            "(Bethesda guidelines extended, now most guidelines universal); "
            "MSI-H TUMOUR: microsatellite instability by PCR/NGS or dMMR by IHC; "
            "MLH1/PMS2 IHC absent: BRAF V600E testing (if mutated = sporadic; if WT = Lynch until germline excluded); "
            "SEBACEOUS ADENOMA on skin biopsy: Muir-Torre = Lynch variant — gene panel mandatory"
        ),
        "treatment_alert": (
            "PEMBROLIZUMAB FDA 2017 — MSI-H/dMMR solid tumours (tissue-agnostic approval); "
            "DOSTARLIMAB FDA 2023 — dMMR endometrial cancer; "
            "ASPIRIN CAPP2 trial: 600 mg/day for ≥2 years → 50% CRC reduction in Lynch "
            "(Level A evidence — recommend aspirin 75-150 mg daily for Lynch); "
            "COLONOSCOPY every 1-2 years from age 25 (or 5 years before earliest CRC); "
            "PROPHYLACTIC HYSTERECTOMY + BILATERAL SALPINGO-OOPHORECTOMY: "
            "offered after childbearing in Lynch women (reduces endometrial/ovarian risk ~95%); "
            "5-FLUOROURACIL adjuvant chemotherapy: LESS effective in MSI-H stage II CRC "
            "(possible HARMFUL — discuss carefully); "
            "SULINDAC: NOT recommended as primary chemoprevention in Lynch (insufficient evidence)"
        ),
        "key_ddx": (
            "MSH2 (urothelial cancer higher 14%); "
            "MSH6 (later-onset, milder CRC phenotype, endometrial dominant); "
            "PMS2 (lowest penetrance Lynch gene, CRC ~15-20%); "
            "Sporadic MSI-H CRC: BRAF V600E + MLH1 promoter methylation distinguishes; "
            "Constitutional MMR deficiency (biallelic MLH1): childhood CRC + brain tumour + NF1 phenotype"
        ),
        "gfr_pattern": "N/A",
        "proteinuria_pattern": "N/A",
        "primary_complication": "Colorectal/endometrial/ovarian cancer; MSI-H tumour immunotherapy eligibility",
        "disease_detail": (
            "Lynch syndrome (LS): most common hereditary colorectal cancer syndrome. "
            "MLH1 most prevalent Lynch gene (30-40% of LS families). "
            "Universal tumour MMR/MSI testing: cost-effective and detects Lynch. "
            "Aspirin CAPP2: 600mg/day ≥2 years → 50% CRC reduction (evidence strong). "
            "Pembrolizumab MSI-H: landmark approval across tumour types. "
            "Biallelic MLH1 = Constitutional MMR deficiency (CMMRD): de novo NF1 + brain + CRC."
        ),
        "inheritance": "Autosomal dominant (AD) [biallelic = CMMRD, AR]",
        "variants": [
            {"name": "Promoter hypermethylation (c.1-34G>T)", "frequency": "Sporadic — NOT hereditary Lynch"},
            {"name": "c.793C>T (p.Arg265Cys)", "frequency": "Common European pathogenic missense"},
            {"name": "Large rearrangements (exon del/dup)", "frequency": "10-20% MLH1 (MLPA mandatory)"},
        ],
        "drug_ci": [
            "5-FLUOROURACIL adjuvant: possibly HARMFUL in MSI-H stage II CRC — review evidence before prescribing",
            "SULINDAC: NOT established as Lynch chemoprevention (unlike FAP where evidence stronger)",
        ],
    },

    # ── MSH2 — Lynch Syndrome / EPCAM ────────────────────────────────────────
    {
        "gene": "MSH2",
        "protein": "MutS protein homolog 2",
        "alias": (
            "MSH2; OMIM gene 609309; 2p21-p16.3; 934 aa; Lynch syndrome OMIM #609309; "
            "AD; EPCAM 3′ deletion causes MSH2 silencing (non-coding Lynch); "
            "urothelial cancer 14% lifetime (highest of Lynch genes); "
            "MutS alpha (MSH2-MSH6) + MutS beta (MSH2-MSH3) — mismatch recognition"
        ),
        "aa": "934 aa",
        "kDa": "~105 kDa",
        "gene_class": (
            "Mismatch repair (MMR) — MutS alpha (MSH2+MSH6) recognises base-base mismatches "
            "and 1-2 nucleotide insertions/deletions; "
            "MutS beta (MSH2+MSH3) recognises larger loop mismatches (2-10 nt); "
            "MSH2 ATPase domain drives conformational change on mismatch binding → recruits MutL alpha; "
            "EPCAM (epithelial cell adhesion molecule) immediately upstream (3p22.2); "
            "Large EPCAM 3′ deletion → transcriptional read-through into MSH2 → MSH2 promoter "
            "hypermethylation → MSH2 silenced → Lynch syndrome WITHOUT MSH2 sequence variant; "
            "IHC: MSH2/MSH6 absent together (MSH2 mutation or EPCAM deletion); "
            "EPCAM deletion: Lynch confirmed by MLPA, not sequence panel alone"
        ),
        "locus": "2p21-p16.3",
        "omim_gene": 609309,
        "omim_disease": 609309,
        "phenotype": (
            "Colorectal cancer (CRC): lifetime 25-60%; right-sided predominant; "
            "Urothelial cancer (renal pelvis + ureter + bladder): 14% lifetime — "
            "HIGHEST of all Lynch genes; annual urine cytology + ultrasound essential; "
            "Endometrial cancer: 25-50% lifetime; "
            "Ovarian cancer: 5-12% lifetime; "
            "Sebaceous neoplasms: Muir-Torre variant (MSH2 most common gene); "
            "Small bowel cancer: 1-4%; "
            "Gastric cancer: 2-8%; "
            "EPCAM-Lynch: same cancer spectrum as MSH2 Lynch"
        ),
        "hallmark": (
            "UROTHELIAL CANCER in young patient + family history: MSH2/EPCAM Lynch until excluded; "
            "EPCAM DELETION — MLPA mandatory: sequence analysis misses EPCAM deletions (common cause MSH2-Lynch); "
            "MSH2/MSH6 absent on IHC: reflex EPCAM MLPA + germline MSH2 sequencing; "
            "MUIR-TORRE SYNDROME: sebaceous gland tumour on skin biopsy + visceral Lynch cancer "
            "= MSH2 most common gene; keratoacanthoma also Muir-Torre; "
            "ANNUAL URINE CYTOLOGY from age 30-35: only Lynch gene where urothelial surveillance is Level A"
        ),
        "treatment_alert": (
            "PEMBROLIZUMAB FDA 2017 — MSI-H/dMMR applies equally to MSH2 Lynch cancers; "
            "COLONOSCOPY every 1-2 years from age 25; "
            "UROTHELIAL SURVEILLANCE: annual urine cytology + renal ultrasound from age 30-35; "
            "LOW-DOSE CT UROGRAPHY every 2-3 years after age 35; "
            "PROPHYLACTIC HYSTERECTOMY + BSO: offered Lynch women after childbearing; "
            "ASPIRIN 75-150 mg/day: Level A Lynch chemoprevention (CAPP2 data); "
            "EPCAM DELETION FAMILIES: same risk management as MSH2 germline Lynch; "
            "cascade test all first-degree relatives including EPCAM"
        ),
        "key_ddx": (
            "MLH1 (MLH1/PMS2 IHC absent vs MSH2/MSH6 absent — different pairs distinguishable); "
            "MSH6 (milder CRC, later onset, endometrial dominant); "
            "EPCAM deletion (MSH2/MSH6 IHC absent but no MSH2 sequence variant — MLPA required); "
            "Sporadic MSI-H: MLH1 methylation (NOT MSH2/MSH6 pathway usually)"
        ),
        "gfr_pattern": "N/A",
        "proteinuria_pattern": "N/A",
        "primary_complication": "Colorectal/urothelial/endometrial cancer; EPCAM deletion missed by sequencing alone",
        "disease_detail": (
            "MSH2 Lynch syndrome: urothelial cancer risk 14% — highest of Lynch genes. "
            "EPCAM deletion mechanism unique: no MSH2 coding variant but MSH2 silenced. "
            "Always test EPCAM by MLPA when MSH2/MSH6 IHC absent and no MSH2 sequence variant. "
            "Muir-Torre: MSH2 most common gene — sebaceous adenoma = Lynch cancer marker."
        ),
        "inheritance": "Autosomal dominant (AD); EPCAM deletion — non-coding Lynch mechanism",
        "variants": [
            {"name": "EPCAM 3′ large deletion", "frequency": "3-10% of MSH2/MSH6-deficient Lynch families"},
            {"name": "c.942+3A>T (splice)", "frequency": "Common pathogenic splice variant"},
            {"name": "Large rearrangements (exon del/dup)", "frequency": "20-30% of MSH2 mutations"},
        ],
        "drug_ci": [
            "UROLOGICAL PROCEDURES: be aware of urothelial cancer risk — hematuria in Lynch MSH2 = urgent urothelial workup",
        ],
    },

    # ── TP53 — Li-Fraumeni Syndrome ──────────────────────────────────────────
    {
        "gene": "TP53",
        "protein": "Cellular tumor antigen p53",
        "alias": (
            "TP53; OMIM gene 191170; 17p13.1; 393 aa; Li-Fraumeni syndrome OMIM #151623; "
            "AD; de novo 20%; guardian of the genome; "
            "radiation ABSOLUTELY CI — dramatically increases second malignancy risk; "
            "annual whole-body MRI + breast MRI + neuroimaging + adrenal imaging"
        ),
        "aa": "393 aa",
        "kDa": "~43.7 kDa",
        "gene_class": (
            "Tumour suppressor — DNA-binding domain (DBD, residues 102-292) contacts major groove; "
            "tetramerisation domain; N-terminal transactivation domain (TAD1 + TAD2); "
            "C-terminal regulatory domain; "
            "TP53 acts as homotetramer; transcription factor inducing: "
            "CDKN1A/p21 (cell cycle arrest), MDM2 (autoregulatory feedback), "
            "BAX/PUMA (apoptosis), TIGAR (ROS), DDB2/XPC (nucleotide excision repair); "
            "LOF → cells fail to arrest or apoptose after DNA damage → malignant transformation; "
            "Gain-of-function (GOF) TP53 mutations (R175H, R248W, R248Q, R273H, R273C, R282W): "
            "dominant negative AND neomorphic oncogenic activity; "
            "De novo TP53 mutations: ~20% LFS families (no family history); "
            "R337H founder mutation in Southern Brazil (population frequency ~0.3% — regional)"
        ),
        "locus": "17p13.1",
        "omim_gene": 191170,
        "omim_disease": 151623,
        "phenotype": (
            "Lifetime cancer risk >90% by age 70 (effectively universal); "
            "Sarcomas (osteosarcoma + soft-tissue sarcoma): most common LFS tumour; "
            "Breast cancer: 25-31% by age 60; premenopausal; young onset; "
            "Brain tumours: glioblastoma, astrocytoma, choroid plexus carcinoma; "
            "Choroid plexus carcinoma (CPC): PATHOGNOMONIC for LFS in children <3 years; "
            "Adrenocortical carcinoma (ACC): childhood ACC (50-80% have germline TP53); "
            "Leukemia/lymphoma; "
            "Colorectal cancer; "
            "Lung cancer; "
            "Multiple synchronous or metachronous primary cancers: hallmark of LFS"
        ),
        "hallmark": (
            "CHOROID PLEXUS CARCINOMA in child <3 years: TP53 LFS in 50-80% — reflex germline testing; "
            "CHILDHOOD ADRENOCORTICAL CARCINOMA: 50-80% have germline TP53; "
            "MULTIPLE PRIMARY CANCERS at young age + sarcoma = LFS classic presentation; "
            "RADIATION ABSOLUTELY CONTRAINDICATED in LFS (radiation → second malignancy risk dramatically elevated; "
            "use proton therapy or avoid if possible); "
            "DE NOVO TP53: 20% LFS — family history may be absent but still very high personal risk"
        ),
        "treatment_alert": (
            "RADIATION ABSOLUTELY CONTRAINDICATED in germline TP53 — use proton beam, surgery, or systemic "
            "therapy instead wherever possible; "
            "ANNUAL WHOLE-BODY MRI (WBMRI): detects sarcomas, adrenal tumours, renal, lymphoma pre-symptomatic; "
            "ANNUAL BRAIN MRI (T1+T2+FLAIR+DWI); "
            "ANNUAL BREAST MRI + mammography from age 20-25 (breast MRI from 20 given radiation concern); "
            "ANNUAL ABDOMINAL ULTRASOUND: adrenocortical carcinoma + renal; "
            "COLONOSCOPY: every 2-5 years from age 25; "
            "MAMMOGRAPHY: debate due to radiation concern — breast MRI preferred for surveillance; "
            "AVOID MAMMOGRAPHY or minimise radiation dose where possible; "
            "MDM2 INHIBITORS (idasanutlin, navtemadlin): clinical trials — p53 reactivation strategies"
        ),
        "key_ddx": (
            "PTEN Cowden (breast + thyroid + GI hamartomas; Lhermitte-Duclos); "
            "CHEK2 (moderate breast/CRC risk; NOT associated with sarcoma); "
            "PALB2 (breast + pancreatic only); "
            "RB1 (retinoblastoma not sarcoma spectrum predominantly); "
            "De novo TP53 somatic (tumour-only — not heritable)"
        ),
        "gfr_pattern": "N/A",
        "proteinuria_pattern": "N/A",
        "primary_complication": "Multiple cancers (sarcoma/breast/brain/ACC); radiation ABSOLUTELY CI; surveillance adherence",
        "disease_detail": (
            "Li-Fraumeni syndrome (LFS): caused by germline TP53 pathogenic variants. "
            "Core LFS cancers: sarcoma, breast cancer, brain tumour, ACC, leukemia. "
            "Whole-body MRI surveillance detects pre-symptomatic tumours. "
            "Radiation: must be avoided/minimised — dramatically increases second cancer risk. "
            "De novo rate 20%: counsel even without family history. "
            "R337H: Brazilian founder (0.3% S. Brazil) — population-level impact."
        ),
        "inheritance": "Autosomal dominant (AD); ~20% de novo",
        "variants": [
            {"name": "R175H (c.524G>A)", "frequency": "Most common GOF hotspot (~6% TP53 pathogenic variants)"},
            {"name": "R248W/Q (c.742C>T/742C>A)", "frequency": "Common GOF hotspot (~5% each)"},
            {"name": "R273H/C (c.818G>A/817C>T)", "frequency": "Common GOF hotspot"},
            {"name": "R337H (c.1010G>A)", "frequency": "Brazilian founder; ~0.3% Southern Brazil population"},
        ],
        "drug_ci": [
            "RADIATION: ABSOLUTELY CONTRAINDICATED — germline TP53 increases radiation-induced second malignancy risk dramatically; use proton beam or avoid RT",
            "MAMMOGRAPHY: minimise radiation dose; prefer MRI for surveillance (radiation concern)",
        ],
    },

    # ── RB1 — Hereditary Retinoblastoma ─────────────────────────────────────
    {
        "gene": "RB1",
        "protein": "Retinoblastoma-associated protein",
        "alias": (
            "RB1; OMIM gene 614041; 13q14.2; 928 aa; retinoblastoma OMIM #180200; "
            "AD; hereditary 40% bilateral, de novo 10-15%; "
            "leukocoria PATHOGNOMONIC (white pupillary reflex); "
            "first tumour suppressor gene identified; Knudson two-hit model"
        ),
        "aa": "928 aa",
        "kDa": "~110 kDa",
        "gene_class": (
            "Tumour suppressor — pocket protein family (pRb, p107, p130); "
            "LXCXE binding cleft binds viral oncoproteins (HPV E7, Adeno E1A, SV40 T-Ag); "
            "Cyclin D/CDK4-6 phosphorylates pRb → releases E2F transcription factors → S phase; "
            "hypophosphorylated pRb: binds E2F1-3 → G1 arrest (RB tumour suppression); "
            "LOF both alleles (Knudson two-hit) → uncontrolled E2F → S-phase entry → retinoblastoma; "
            "Hereditary: germline LOF (one hit) + somatic second hit (Knudson, 1971 — Nobel Prize model); "
            "Bilateral/multifocal retinoblastoma = almost always hereditary (germline RB1); "
            "Unilateral unifocal: 85% sporadic; 15% germline RB1 (screen even unilateral)"
        ),
        "locus": "13q14.2",
        "omim_gene": 614041,
        "omim_disease": 180200,
        "phenotype": (
            "Retinoblastoma (benign nomenclature: intraocular; malignant: extraocular): "
            "presents age 0-5 years (median 2 years bilateral, 3 years unilateral); "
            "Leukocoria (white pupillary reflex) PATHOGNOMONIC — detected on red-reflex exam; "
            "Strabismus — second most common presentation (loss of central vision); "
            "Bilateral/multifocal: virtually always hereditary RB1; "
            "Trilateral retinoblastoma: bilateral Rb + pineoblastoma/suprasellar PNET (~5%); "
            "Second malignant neoplasms (SMN): osteosarcoma (most common SMN), "
            "soft tissue sarcoma, melanoma, brain tumour — risk 10-20x general population; "
            "Radiation CI: dramatically increases SMN risk (osteosarcoma 1500x risk after orbital RT)"
        ),
        "hallmark": (
            "LEUKOCORIA (white pupil on fundal exam / photo) = PATHOGNOMONIC retinoblastoma "
            "until proven otherwise — URGENT ophthalmology referral; "
            "RED-REFLEX EXAM AT BIRTH and all newborn/infant well visits MANDATORY "
            "(misses = diagnostic delay → extraocular spread); "
            "BILATERAL/MULTIFOCAL retinoblastoma: hereditary RB1 (>95% of bilateral cases); "
            "TRILATERAL RETINOBLASTOMA: annual brain MRI for 5 years post-diagnosis; "
            "RADIATION ABSOLUTELY CI in hereditary RB1: osteosarcoma risk 1500x with orbital RT; "
            "ALL retinoblastoma survivors: annual surveillance for SMN lifelong"
        ),
        "treatment_alert": (
            "ENUCLEATION: advanced intraocular disease — globe-sparing preferred where feasible; "
            "INTRA-ARTERIAL CHEMOTHERAPY (IAC) — ophthalmic artery infusion (melphalan): "
            "globe-sparing primary treatment for unilateral advanced; "
            "SYSTEMIC CHEMOTHERAPY (VEC — vincristine/etoposide/carboplatin): bilateral RB; "
            "RADIATION ABSOLUTELY CONTRAINDICATED — hereditary RB1 (osteosarcoma SMN risk 1500x); "
            "use focal consolidation (laser/cryotherapy/brachytherapy) NOT external beam RT; "
            "ANNUAL SURVEILLANCE: ophthalmological exams under anaesthesia until age 5; "
            "then regular ophthalmic follow-up lifelong; "
            "SECOND MALIGNANT NEOPLASM surveillance: bone scintigraphy/MRI for osteosarcoma; "
            "PINEOBLASTOMA: annual brain MRI for 5 years in trilateral risk period; "
            "GENETIC COUNSELLING: 40-45% chance of passing RB1 to offspring"
        ),
        "key_ddx": (
            "Persistent fetal vasculature (PFV): leukocoria, usually unilateral, microphthalmos; "
            "Coats disease: leukocoria, male predominant, non-hereditary; "
            "Toxocara (ocular larva migrans): peripheral retinal granuloma; "
            "Retinopathy of prematurity (ROP): bilateral, preterm, retinal neovascularisation; "
            "PHPV (persistent hyperplastic primary vitreous)"
        ),
        "gfr_pattern": "N/A",
        "proteinuria_pattern": "N/A",
        "primary_complication": "Retinoblastoma; radiation CI (osteosarcoma SMN); second malignancy surveillance",
        "disease_detail": (
            "Hereditary retinoblastoma (RB1 germline): archetype of tumour suppressor two-hit model. "
            "Bilateral/multifocal = hereditary (>95%). Red-reflex exam: earliest detection at birth. "
            "Radiation CI: orbital RT causes osteosarcoma 1500x risk — global shift to non-radiation protocols. "
            "Trilateral: pineoblastoma 5% of bilateral RB — annual brain MRI mandatory."
        ),
        "inheritance": "Autosomal dominant (AD); ~10-15% de novo in bilateral",
        "variants": [
            {"name": "Nonsense/frameshift (50-60%)", "frequency": "Most common — complete LOF"},
            {"name": "Splice site (10-20%)", "frequency": "Variable penetrance by site"},
            {"name": "Large deletions (5-10%)", "frequency": "MLPA required; may include contiguous genes → 13q deletion syndrome (ID)"},
            {"name": "Low-level mosaicism", "frequency": "5-10% — deep sequencing required; false negatives in standard testing"},
        ],
        "drug_ci": [
            "EXTERNAL BEAM RADIATION THERAPY (EBRT): ABSOLUTELY CONTRAINDICATED in hereditary RB1 — risk osteosarcoma 1500-fold increased; use IAC/laser/cryo/brachytherapy instead",
            "ETOPOSIDE (VEC): secondary AML risk (topoisomerase II inhibitor) — balance benefit vs risk in bilateral RB",
        ],
    },

    # ── CDKN2A — FAMMM / Familial Pancreatic Cancer ─────────────────────────
    {
        "gene": "CDKN2A",
        "protein": "Cyclin-dependent kinase inhibitor 2A (p16-INK4a / p14-ARF)",
        "alias": (
            "CDKN2A; OMIM gene 600160; 9p21.3; 156 aa (p16) / 132 aa (p14-ARF — alternate reading frame); "
            "FAMMM OMIM #155601; AD; melanoma lifetime 28-67% (geography-dependent); "
            "pancreatic cancer 17-39x relative risk (lifetime 15-24%); "
            "multiple atypical nevi PATHOGNOMONIC phenotype"
        ),
        "aa": "156 aa (p16/INK4a); 132 aa (p14/ARF, alternate reading frame)",
        "kDa": "~16 kDa (p16)",
        "gene_class": (
            "Tumour suppressor — CDKN2A locus encodes TWO distinct proteins via alternate reading frames: "
            "(1) p16/INK4a: ankyrin repeat protein; inhibits CDK4/CDK6 → prevents phosphorylation of "
            "pRb → maintains G1 checkpoint (cannot proceed without functional pRb); "
            "(2) p14/ARF: entirely different protein (alternate exon 1β + different reading frame 2/3); "
            "stabilises p53 by binding MDM2 → prevents MDM2-mediated p53 degradation; "
            "CDKN2A LOF → CDK4/6 unchecked → pRb hyperphosphorylated → E2F released → "
            "G1/S unchecked AND p53 destabilised (ARF loss); "
            "Somatic CDKN2A homozygous deletion: most common somatic tumour suppressor loss in solid tumours; "
            "Germline CDKN2A: FAMMM (familial atypical multiple-mole melanoma) + hereditary pancreatic cancer"
        ),
        "locus": "9p21.3",
        "omim_gene": 600160,
        "omim_disease": 155601,
        "phenotype": (
            "Familial atypical multiple-mole melanoma (FAMMM): "
            "≥50-100 melanocytic nevi including dysplastic nevi PATHOGNOMONIC; "
            "Melanoma: lifetime 28-67% (Australia/Europe-dependent); young onset (<50); "
            "multiple primary melanomas (10-15%); "
            "Pancreatic adenocarcinoma: 17-39x RR; lifetime 15-24% (highest of any hereditary cancer gene); "
            "Pancreatic cancer mean onset ~58 years; "
            "Ocular melanoma (uveal): 3-5x RR; "
            "Oral/oropharyngeal squamous: risk data emerging"
        ),
        "hallmark": (
            "MULTIPLE ATYPICAL (DYSPLASTIC) NEVI + family history melanoma = CDKN2A FAMMM; "
            "≥50 NEVI with ≥1 atypical/dysplastic: refer to genetics + dermatology; "
            "PANCREATIC CANCER RISK — most important counselling point: CDKN2A carriers "
            "have among the highest hereditary pancreatic cancer risks of any gene (15-24%); "
            "ANNUAL EUS + MRCP from age 40 (or 10 years before earliest pancreatic cancer in family); "
            "WHOLE-BODY DERMOSCOPY annually by melanoma-trained dermatologist; "
            "SUN PROTECTION MANDATORY — UV is a critical co-carcinogen in CDK2A carriers"
        ),
        "treatment_alert": (
            "PALBOCICLIB/RIBOCICLIB/ABEMACICLIB (CDK4/6 inhibitors FDA 2015+): "
            "effective in HR+ HER2- breast cancer — note CDKN2A LOF renders CDK4/6 HYPERACTIVE; "
            "CDK4/6 inhibitors relevant in CDKN2A-deficient tumours; "
            "PANCREATIC CANCER SURVEILLANCE: annual EUS + MRCP from age 40 "
            "(or 10y before earliest family PC); consider aspirin 81mg; "
            "MELANOMA SURVEILLANCE: whole-body dermoscopy + dermoscopy mapping annually; "
            "BASELINE TOTAL-BODY PHOTOGRAPHY for nevi monitoring; "
            "PEMBROLIZUMAB / NIVOLUMAB (PD-1 inhibitors): standard for advanced melanoma "
            "(CDKN2A LOF associated with immunotherapy response); "
            "SUN PROTECTION: SPF 50+, UV clothing, avoid tanning beds absolutely; "
            "OPHTHALMOLOGICAL SURVEILLANCE: uveal melanoma screening (controversial — annual dilated fundal)"
        ),
        "key_ddx": (
            "BAP1 (uveal melanoma + mesothelioma + RCC); "
            "BRCA2 (pancreatic cancer + breast/ovarian); "
            "ATM (pancreatic + breast, moderate risk); "
            "STK11 Peutz-Jeghers (pancreatic + GI polyps + mucocutaneous pigmentation); "
            "MITF (melanoma, no pancreatic risk); "
            "Sporadic multiple nevi (no pancreatic risk component)"
        ),
        "gfr_pattern": "N/A",
        "proteinuria_pattern": "N/A",
        "primary_complication": "Melanoma + pancreatic cancer dual risk; annual EUS surveillance mandatory",
        "disease_detail": (
            "CDKN2A FAMMM: dual cancer predisposition — melanoma + pancreatic cancer. "
            "Encodes two distinct proteins (p16/INK4a + p14/ARF) from alternate reading frames. "
            "Pancreatic cancer risk 15-24%: highest among hereditary cancer genes. "
            "EUS+MRCP annual surveillance detects early resectable pancreatic cancer. "
            "CDK4/6 inhibitors exploit CDK4/6 hyperactivity in CDKN2A-null tumours."
        ),
        "inheritance": "Autosomal dominant (AD)",
        "variants": [
            {"name": "p.Ala148Thr (Europe)", "frequency": "Common pathogenic missense"},
            {"name": "p16-Leiden (19bp del)", "frequency": "Dutch/Scandinavian founder"},
            {"name": "CDKN2A deletion exon 1α/2 (entire gene)", "frequency": "5-10% (MLPA required)"},
        ],
        "drug_ci": [
            "TANNING BEDS: ABSOLUTELY CI — UV radiation is the primary modifiable risk factor in CDKN2A carriers",
            "IMMUNOSUPPRESSANTS post-transplant: dramatically increase melanoma risk — dermatology surveillance intensify",
        ],
    },

    # ── APC — Familial Adenomatous Polyposis ─────────────────────────────────
    {
        "gene": "APC",
        "protein": "Adenomatous polyposis coli protein",
        "alias": (
            "APC; OMIM gene 611731; 5q22.2; 2843 aa; FAP OMIM #175100; "
            "AD; ≥100 colonic adenomas PATHOGNOMONIC; "
            "prophylactic colectomy by age 20-25 mandatory; "
            "CHRPE (congenital hypertrophy of RPE) ocular marker; "
            "desmoid tumours codon >1310 association"
        ),
        "aa": "2843 aa",
        "kDa": "~311 kDa",
        "gene_class": (
            "Tumour suppressor — β-catenin destruction complex component "
            "(APC + AXIN1 + GSK3β + CK1 → phosphorylate β-catenin → proteasomal degradation); "
            "APC loss → free β-catenin accumulates → nuclear translocation → TCF/LEF target gene activation "
            "(MYC, CCND1, AXIN2) → proliferation; "
            "APC also functions in chromosome segregation, cell migration, Wnt gradient; "
            "Genotype-phenotype: codon 1309 → >1000 polyps, earliest onset; "
            "codon 169-1309 → classic FAP; codon <169 or >1309 → attenuated/extracolonic features; "
            "codon >1310 (esp. 1310-2843): desmoid tumour risk strongly associated; "
            "Somatic APC: first hit in >80% sporadic colorectal cancer (biallelic somatic); "
            "MUTYH-associated polyposis (MAP): similar phenotype, AR, biallelic MUTYH LOF"
        ),
        "locus": "5q22.2",
        "omim_gene": 611731,
        "omim_disease": 175100,
        "phenotype": (
            "Classic FAP: ≥100 (up to thousands) colorectal adenomas by age 20-30s "
            "(PATHOGNOMONIC); untreated CRC by age 40 nearly 100%; "
            "Duodenal/periampullary adenomas: Spigelman staging; Stage IV → prophylactic pancreaticoduodenectomy risk; "
            "Desmoid tumours (15-20%): locally invasive fibromatosis — mesenteric desmoids post-surgery; "
            "Gastric fundic gland polyps (benign, >90%); "
            "Thyroid cancer (papillary, 2-3%); "
            "Hepatoblastoma (childhood, <1%); "
            "Medulloblastoma (Turcot variant, <1%); "
            "CHRPE (congenital hypertrophy of retinal pigment epithelium): "
            "bilateral, ≥4 lesions PATHOGNOMONIC for FAP (not cancer); "
            "Attenuated FAP (AFAP): <100 polyps, later onset, right-sided predominant"
        ),
        "hallmark": (
            "≥100 COLONIC ADENOMAS on colonoscopy = FAP diagnosis until APC excluded; "
            "CHRPE (bilateral ≥4 lesions): PATHOGNOMONIC — present in 60-80% classic FAP; "
            "early routine ophthalmological finding prior to symptoms; "
            "DESMOID TUMOURS: pathognomonic when familial + codon >1310 APC; "
            "risk increased 20-fold post-abdominal surgery (especially surgery triggers); "
            "CODON 1309 (c.3927_3931del): most severe FAP — >1000 polyps, CRC by age 25; "
            "ATTENUATED FAP: <100 polyps, late onset — COLONOSCOPY NOT SIGMOIDOSCOPY (right-sided)"
        ),
        "treatment_alert": (
            "PROPHYLACTIC COLECTOMY: mandatory by age 20-25 in classic FAP "
            "(before CRC inevitable); ileal pouch anal anastomosis (IPAA) standard; "
            "SULINDAC (NSAID COX-1/2): reduces polyp burden — adjunct NOT curative; "
            "useful pre-operatively and for duodenal polyps; "
            "CELECOXIB (COX-2 selective): FDA approved FAP polyp reduction — NOT curative; "
            "cardiovascular risk monitoring required; "
            "DUODENAL SURVEILLANCE: upper GI endoscopy every 1-5 years (Spigelman stage dependent); "
            "Stage IV Spigelman: consider prophylactic Whipple or extended duodenectomy; "
            "DESMOID MANAGEMENT: sulindac + tamoxifen (anti-oestrogen) first-line; "
            "imatinib (tyrosine kinase inhibitor) second-line; avoid surgery if possible (triggers growth); "
            "THYROID SURVEILLANCE: annual thyroid ultrasound from age 15-20 (papillary thyroid); "
            "HEPATOBLASTOMA: AFP + ultrasound in children <5 years of APC families"
        ),
        "key_ddx": (
            "MUTYH-associated polyposis (MAP): AR biallelic MUTYH — 10-100 polyps, CRC risk; "
            "Attenuated FAP (AFAP): <100 polyps, APC codon <169 or >1595; "
            "Serrated polyposis syndrome: serrated not adenomatous polyps; "
            "Lynch syndrome: CRC without polyposis; "
            "Gardner syndrome = FAP + extracolonic (CHRPE + desmoid + osteoma + epidermoid cyst)"
        ),
        "gfr_pattern": "N/A",
        "proteinuria_pattern": "N/A",
        "primary_complication": "CRC (nearly 100% by 40 if untreated); desmoid tumours post-surgery; duodenal cancer",
        "disease_detail": (
            "Familial adenomatous polyposis (FAP): classic form has ≥100 adenomas by 20s. "
            "Untreated CRC essentially 100% by age 40. Prophylactic colectomy 20-25 yo mandatory. "
            "Desmoid tumours: mesenteric, locally aggressive, codon >1310 APC; avoid repeat surgery. "
            "Sulindac/celecoxib: reduce polyp burden but do not cure or replace colectomy. "
            "CHRPE: bilateral ≥4 lesions — ocular pathognomonic marker. "
            "MUTYH-polyposis: clinical mimic, AR — counsel differently (siblings 25% risk, not 50%)."
        ),
        "inheritance": "Autosomal dominant (AD); ~25% de novo",
        "variants": [
            {"name": "c.3927_3931delAAAGA (codon 1309, 5bp del)", "frequency": "5-7% FAP; most severe classic (>1000 polyps, CRC by 25)"},
            {"name": "c.2437_2438delTTinsAATT", "frequency": "Northern European common pathogenic indel"},
            {"name": "c.3183_3187delACAAA (codon 1061)", "frequency": "Common 5bp deletion"},
            {"name": "Large rearrangements", "frequency": "10-15% FAP (MLPA required)"},
        ],
        "drug_ci": [
            "SULINDAC/CELECOXIB: NOT curative for FAP — colectomy MUST NOT be delayed waiting for chemical response; adjunct only",
            "SURGERY (abdominal) in desmoid-prone genotype (codon >1310): AVOID unless absolutely necessary — surgery triggers desmoid growth",
            "TAMOXIFEN in desmoids: anti-oestrogen effect on desmoid fibroblasts — use with sulindac as first-line desmoid therapy",
        ],
    },
]


def _make_cohort(gene: dict, seed: int, n: int = 40) -> list:
    """Generate a deterministic synthetic patient cohort for a hereditary cancer gene."""
    rng = random.Random(seed)
    gene_id = gene["gene"]
    patients = []

    for i in range(n):
        age = rng.randint(18, 75)
        sex = rng.choice(["M", "F"])

        if gene_id == "BRCA1":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[20, 45, 35])[0]
            breast_cancer = rng.random() < 0.72
            ovarian_cancer = rng.random() < 0.44
            tnbc = breast_cancer and rng.random() < 0.70
            drug_error = rng.random() < 0.14  # missed PARP inhibitor eligibility or tamoxifen in ER-
            dx_delayed = rng.random() < 0.25
            rrso_done = rng.random() < 0.55
            transplant = False
            esrd = rng.random() < 0.01
            htn = rng.random() < 0.18

        elif gene_id == "BRCA2":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[22, 48, 30])[0]
            breast_cancer = rng.random() < 0.71
            ovarian_cancer = rng.random() < 0.20
            male_bc = sex == "M" and rng.random() < 0.06
            pancreatic = rng.random() < 0.06
            drug_error = rng.random() < 0.12  # missed olaparib maintenance after platinum
            dx_delayed = rng.random() < 0.22
            rrso_done = rng.random() < 0.50
            transplant = False
            esrd = rng.random() < 0.01
            htn = rng.random() < 0.18

        elif gene_id == "MLH1":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[25, 45, 30])[0]
            crc = rng.random() < 0.50
            endometrial = sex == "F" and rng.random() < 0.40
            msi_h = rng.random() < 0.95
            drug_error = rng.random() < 0.18  # 5-FU given for MSI-H stage II CRC
            dx_delayed = rng.random() < 0.35  # universal testing not done
            transplant = False
            esrd = rng.random() < 0.02
            htn = rng.random() < 0.25

        elif gene_id == "MSH2":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[25, 45, 30])[0]
            crc = rng.random() < 0.45
            urothelial = rng.random() < 0.14
            endometrial = sex == "F" and rng.random() < 0.40
            epcam_missed = rng.random() < 0.15  # EPCAM deletion not checked (drug error = diagnostic error)
            drug_error = epcam_missed
            dx_delayed = rng.random() < 0.38
            transplant = False
            esrd = rng.random() < 0.02
            htn = rng.random() < 0.25

        elif gene_id == "TP53":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[10, 35, 55])[0]
            sarcoma = rng.random() < 0.30
            breast_cancer = rng.random() < 0.28
            brain_tumour = rng.random() < 0.15
            acc = rng.random() < 0.08
            radiation_given = rng.random() < 0.22  # drug error: radiation given despite germline TP53
            drug_error = radiation_given
            dx_delayed = rng.random() < 0.40  # de novo TP53 high rate
            transplant = False
            esrd = rng.random() < 0.03
            htn = rng.random() < 0.15
            age = rng.randint(1, 65)  # LFS affects childhood onwards

        elif gene_id == "RB1":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[15, 40, 45])[0]
            bilateral = rng.random() < 0.40  # bilateral = hereditary
            trilateral = bilateral and rng.random() < 0.05
            radiation_given = rng.random() < 0.28  # drug error: orbital EBRT in hereditary RB
            drug_error = radiation_given
            smn_osteosarcoma = radiation_given and rng.random() < 0.30
            dx_delayed = rng.random() < 0.20  # red-reflex missed
            transplant = False
            esrd = rng.random() < 0.01
            htn = rng.random() < 0.10
            age = rng.randint(0, 10)  # childhood onset

        elif gene_id == "CDKN2A":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[20, 45, 35])[0]
            melanoma = rng.random() < 0.45
            pancreatic = rng.random() < 0.18
            multiple_melanoma = melanoma and rng.random() < 0.12
            sun_protection_missed = rng.random() < 0.30  # no UV protection (drug error category)
            drug_error = sun_protection_missed
            dx_delayed = rng.random() < 0.42  # multiple nevi not recognised as FAMMM
            transplant = False
            esrd = rng.random() < 0.02
            htn = rng.random() < 0.20

        elif gene_id == "APC":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[15, 40, 45])[0]
            colectomy_delayed = rng.random() < 0.20  # colectomy not done by 25
            desmoid = rng.random() < 0.18
            duodenal_advanced = rng.random() < 0.12
            crc_developed = colectomy_delayed and rng.random() < 0.55
            drug_error = colectomy_delayed  # relying on sulindac/celecoxib instead of colectomy
            dx_delayed = rng.random() < 0.15
            transplant = False
            esrd = rng.random() < 0.01
            htn = rng.random() < 0.22

        else:
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[25, 45, 30])[0]
            drug_error = rng.random() < 0.15
            dx_delayed = rng.random() < 0.30
            transplant = False
            esrd = rng.random() < 0.02
            htn = rng.random() < 0.20

        surveillance_adherent = rng.random() < 0.62

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
    for idx, gene in enumerate(CANCER_GENES):
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
        "atlas_name": "Cancer Syndromes Atlas",
        "atlas_subtitle": (
            "Complete 8-Gene Hereditary Cancer Predisposition Syndrome Atlas — "
            "BRCA1 · BRCA2 · MLH1 · MSH2 · TP53 · RB1 · CDKN2A · APC"
        ),
        "n_genes": 8,
        "n_patients": n,
        "seeds": "1198–1205",
        "description": (
            "Comprehensive hereditary cancer predisposition reference covering the 8 most clinically "
            "significant monogenic cancer syndromes: "
            "HBOC type 1 (BRCA1 AD — 70-80% breast lifetime; 44-46% ovarian; olaparib PARP inhibitor; "
            "RRSO 35-40 yo; TNBC predominant); "
            "HBOC type 2 (BRCA2 AD — 69-75% breast; 17-24% ovarian; 5-7% pancreatic; "
            "male breast 6%; PARP inhibitors + platinum); "
            "Lynch syndrome MLH1 (AD — 25-70% CRC; 25-60% endometrial; MSI-H; "
            "pembrolizumab; aspirin Level A; 5-FU HARMFUL MSI-H stage II); "
            "Lynch syndrome MSH2/EPCAM (AD — urothelial cancer 14% highest Lynch; "
            "EPCAM deletion MLPA mandatory — Sanger misses); "
            "Li-Fraumeni syndrome (TP53 AD — radiation ABSOLUTELY CI; "
            "annual whole-body MRI; choroid plexus carcinoma PATHOGNOMONIC child); "
            "Hereditary retinoblastoma (RB1 AD — leukocoria PATHOGNOMONIC; "
            "radiation CI 1500x osteosarcoma; red-reflex exam mandatory at birth); "
            "FAMMM + hereditary pancreatic (CDKN2A AD — multiple atypical nevi PATHOGNOMONIC; "
            "pancreatic cancer 17-39x risk; annual EUS+MRCP from 40); "
            "Familial adenomatous polyposis (APC AD — ≥100 adenomas PATHOGNOMONIC; "
            "prophylactic colectomy 20-25 yo mandatory; sulindac/celecoxib NOT curative; "
            "CHRPE ocular marker; desmoid codon >1310)"
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
                "title": "TP53-LFS: Radiation ABSOLUTELY CONTRAINDICATED — Second Malignancy Risk Dramatically Elevated",
                "body": (
                    "Germline TP53 pathogenic variants (Li-Fraumeni syndrome) confer extreme sensitivity "
                    "to radiation-induced DNA damage. External beam radiation therapy causes dramatically "
                    "elevated second malignant neoplasm (SMN) risk — osteosarcomas, brain tumours, sarcomas "
                    "in the radiation field. Proton beam therapy or surgery must replace RT wherever possible. "
                    "Mammography: minimise radiation exposure; use MRI as primary surveillance."
                ),
            },
            {
                "type": "danger",
                "title": "RB1: Radiation ABSOLUTELY CONTRAINDICATED in Hereditary Retinoblastoma — 1500x Osteosarcoma Risk",
                "body": (
                    "Orbital external beam radiotherapy (EBRT) in hereditary RB1 carriers causes osteosarcoma "
                    "risk approximately 1500-fold above baseline. The radiation field hits the growing facial "
                    "skeleton in children with germline RB1 LOF in all cells. Modern protocols use "
                    "intra-arterial chemotherapy (IAC), intravitreal melphalan, laser photocoagulation, "
                    "and cryotherapy to achieve globe salvage without EBRT. EBRT is now reserved only "
                    "for extraocular disease where benefit clearly outweighs risk."
                ),
            },
            {
                "type": "danger",
                "title": "MSH2/EPCAM: EPCAM 3′ Deletion MISSED by Sequencing — MLPA Mandatory",
                "body": (
                    "EPCAM large 3′ deletions cause MSH2 promoter hypermethylation and Lynch syndrome "
                    "WITHOUT any MSH2 coding sequence variant. Standard gene panel sequencing (NGS) "
                    "WILL NOT detect this mechanism. When MSH2/MSH6 are absent on IHC and no MSH2 "
                    "sequence variant is found, EPCAM MLPA is mandatory. Missing EPCAM deletion = "
                    "missed Lynch diagnosis and failure to test family members."
                ),
            },
            {
                "type": "danger",
                "title": "APC-FAP: Sulindac/Celecoxib NOT Curative — Prophylactic Colectomy MUST NOT Be Delayed",
                "body": (
                    "NSAIDs (sulindac, celecoxib) reduce adenoma burden in FAP but are NOT curative. "
                    "Colorectal cancer risk approaches 100% by age 40 in untreated classic FAP. "
                    "Prophylactic colectomy (IPAA or IRA) must be performed by age 20-25 years. "
                    "NSAIDs are adjuncts for duodenal polyp control and pre-operative burden reduction, "
                    "never a substitute for surgery. Relying on NSAIDs without colectomy = lethal error."
                ),
            },
            {
                "type": "danger",
                "title": "MLH1-Lynch: 5-Fluorouracil May Be Harmful in MSI-H Stage II Colorectal Cancer",
                "body": (
                    "Multiple retrospective analyses suggest 5-fluorouracil (5-FU) based adjuvant "
                    "chemotherapy may be less effective or potentially harmful in MSI-H (dMMR) "
                    "stage II colorectal cancer compared to observation alone. Possible mechanism: "
                    "MMR proteins process 5-FU-induced DNA damage; MMR-deficient cells may tolerate "
                    "5-FU. This does not apply to stage III MSI-H CRC where chemotherapy benefit "
                    "is retained. Tumour MSI status must be checked before prescribing adjuvant 5-FU."
                ),
            },
            {
                "type": "warning",
                "title": "BRCA1-HBOC: Tamoxifen Limited Benefit — BRCA1 Cancers Mostly Triple-Negative (ER-)",
                "body": (
                    "Tamoxifen and raloxifene reduce ER-positive breast cancer risk by 40-50% in the "
                    "general population. However, BRCA1-associated cancers are predominantly "
                    "triple-negative (ER-/PR-/HER2- in 60-80%). Chemoprevention with anti-oestrogens "
                    "has limited applicability to BRCA1 carriers. Risk-reducing mastectomy (RRM) "
                    "and RRSO remain the most effective risk-reduction strategies. "
                    "Discuss benefit-risk of tamoxifen in BRCA1 carriers carefully."
                ),
            },
        ],
        "critical_rules": [
            "TP53-LFS: Radiation ABSOLUTELY CI — second malignancy risk dramatically elevated; use proton beam or surgery",
            "RB1: Orbital EBRT ABSOLUTELY CI in hereditary RB1 — 1500x osteosarcoma risk; use IAC/laser/cryo",
            "MSH2/EPCAM: MLPA MANDATORY when MSH2/MSH6 IHC absent without MSH2 sequence variant — EPCAM deletion",
            "APC-FAP: Sulindac/celecoxib NOT curative — prophylactic colectomy by age 20-25 MANDATORY",
            "MLH1-Lynch: 5-FU POSSIBLY HARMFUL in MSI-H stage II CRC — check MMR status before adjuvant",
            "BRCA1/2: PARP inhibitors (olaparib/rucaparib) — verify eligibility before excluding (check FDA label)",
            "CDKN2A-FAMMM: Annual EUS+MRCP from age 40 for pancreatic cancer (17-39x risk)",
            "RB1: RED-REFLEX EXAM at birth — leukocoria PATHOGNOMONIC; immediate ophthalmology referral",
            "BRCA1: Annual breast MRI (NOT mammography alone) from age 25 — dense breasts, young patients",
            "MLH1/MSH2: Aspirin 75-150 mg/day — Level A chemoprevention in Lynch syndrome (CAPP2)",
        ],
        "pathway_targets": {
            "BRCA1":  "HR repair (PARP inhibitors exploit HR deficiency — synthetic lethality)",
            "BRCA2":  "RAD51/HR repair (PARP inhibitors, platinum sensitivity — pancreatic/ovarian/breast)",
            "MLH1":   "MMR/MSI-H (pembrolizumab PD-1; aspirin; MMR restoration strategies)",
            "MSH2":   "MutS alpha MMR (pembrolizumab; urothelial surveillance; EPCAM check)",
            "TP53":   "p53 axis (MDM2 inhibitors trials; avoid radiation; whole-body MRI)",
            "RB1":    "pRb/CDK4-6/E2F axis (CDK4/6 inhibitors; globe-sparing chemotherapy)",
            "CDKN2A": "p16/CDK4-6/pRb + p14/MDM2/p53 (CDK4/6 inhibitors; PD-1 melanoma; EUS)",
            "APC":    "β-catenin/Wnt (colectomy; sulindac; tankyrase inhibitors experimental)",
        },
        "kpis": [
            {"label": "Total Patients", "value": str(n)},
            {"label": "Genes Covered", "value": "8"},
            {"label": "Cohort Seeds", "value": "1198–1205"},
            {"label": "Drug Errors (Agg)", "value": f"{agg.get('drug_error_pct', 0)}%"},
            {"label": "Dx Delayed (Agg)", "value": f"{agg.get('dx_delayed_pct', 0)}%"},
            {"label": "Surveillance Adherent", "value": f"{agg.get('surveillance_adherent_pct', 0)}%"},
        ],
        "disease_category_breakdown": {
            "Homologous Recombination (BRCA1)": 12.5,
            "HR Repair + Pancreatic (BRCA2)": 12.5,
            "Mismatch Repair CRC (MLH1)": 12.5,
            "MMR Urothelial (MSH2/EPCAM)": 12.5,
            "Guardian of Genome (TP53/LFS)": 12.5,
            "Cell Cycle pRb (RB1)": 12.5,
            "CDK4-6 Brake/FAMMM (CDKN2A)": 12.5,
            "Wnt/β-catenin Polyposis (APC)": 12.5,
        },
    }


def get_breakdown() -> dict:
    genes_out = []
    for idx, gene_def in enumerate(CANCER_GENES):
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
            "term": "Homologous Recombination (HR) Deficiency",
            "full": "Inability to repair double-strand DNA breaks via error-free homologous recombination",
            "explanation": (
                "BRCA1 and BRCA2 are essential for HR-mediated DSB repair. LOF → error-prone NHEJ "
                "→ chromosome instability. HR deficiency = 'BRCAness' — tumours sensitive to "
                "platinum agents and PARP inhibitors."
            ),
        },
        {
            "term": "PARP Inhibitors (PARPi)",
            "full": "Poly(ADP-ribose) polymerase inhibitors — olaparib, rucaparib, niraparib, talazoparib",
            "explanation": (
                "PARP1/2 normally repair single-strand DNA breaks. PARPi block PARP + trap PARP on DNA "
                "(PARP trapping). In HR-deficient (BRCA1/2-LOF) cells, SSBs → DSBs not repaired → "
                "cell death. Synthetic lethality: PARPi effective only in HR-deficient background."
            ),
        },
        {
            "term": "MSI-H / dMMR",
            "full": "Microsatellite instability-high / deficient mismatch repair",
            "explanation": (
                "MMR proteins (MLH1, MSH2, MSH6, PMS2) correct base-base mismatches and "
                "insertion/deletion loops at microsatellite repeats. LOF → thousands of mutations "
                "accumulate at microsatellites (MSI-H). MSI-H/dMMR tumours: high TMB → neoantigen-rich "
                "→ respond to checkpoint inhibitors (pembrolizumab FDA 2017 tissue-agnostic)."
            ),
        },
        {
            "term": "Knudson Two-Hit Hypothesis",
            "full": "Both alleles of a tumour suppressor gene must be inactivated for cancer to develop",
            "explanation": (
                "Proposed by Alfred Knudson 1971 for retinoblastoma. Hereditary cases: germline LOF "
                "(first hit present in all cells) + somatic LOF (second hit in target tissue) → cancer. "
                "Sporadic: two somatic hits required. Explains why hereditary cancers are bilateral/multifocal "
                "and earlier onset. Template for all tumour suppressor genes (RB1, TP53, APC, BRCA1/2)."
            ),
        },
        {
            "term": "Leukocoria",
            "full": "White pupillary reflex (absent red reflex) on ophthalmoscopy",
            "explanation": (
                "Pathognomonic for retinoblastoma — white retinal mass reflects light back through pupil. "
                "Also caused by: persistent fetal vasculature (PFV), Coats disease, toxocara, cataract. "
                "Absent red reflex on ANY routine examination → urgent ophthalmology referral (within 24-48h). "
                "Photography (Brückner test) detects subtle leukocoria."
            ),
        },
        {
            "term": "CHRPE (Congenital Hypertrophy of the Retinal Pigment Epithelium)",
            "full": "Bilateral, multifocal pigmented flat retinal lesions — FAP ocular marker",
            "explanation": (
                "Bilateral ≥4 CHRPE lesions (pigmented 'bear tracks') are pathognomonic for FAP "
                "in 60-80% of classic FAP families. Not pre-malignant. Detected on dilated fundoscopy. "
                "Unilateral or solitary CHRPE is common in general population and NOT associated with FAP. "
                "Bilateral multifocal pattern is the specific FAP marker."
            ),
        },
        {
            "term": "EPCAM Deletion (Lynch MSH2)",
            "full": "Large 3′ deletion of EPCAM gene causing MSH2 promoter hypermethylation",
            "explanation": (
                "EPCAM (epithelial cell adhesion molecule) is immediately upstream of MSH2 on 2p22. "
                "Large deletions removing the 3′ end of EPCAM cause transcriptional read-through into "
                "MSH2 promoter → MSH2 promoter methylation → MSH2 silencing. "
                "Result: Lynch syndrome with MSH2/MSH6-absent IHC but NO MSH2 coding sequence variant. "
                "NOT detected by standard sequencing — MLPA is mandatory."
            ),
        },
        {
            "term": "Universal MMR/MSI Tumour Testing",
            "full": "IHC for MLH1/MSH2/MSH6/PMS2 or MSI PCR on ALL colorectal and endometrial cancers",
            "explanation": (
                "Cost-effective Lynch detection strategy: test every CRC and EC for dMMR/MSI. "
                "MLH1/PMS2 absent: BRAF V600E + MLH1 methylation (sporadic) vs germline MLH1 (Lynch). "
                "MSH2/MSH6 absent: MSH2 germline + EPCAM MLPA. "
                "MSH6/PMS2 absent alone: germline MSH6 or PMS2 (lower penetrance Lynch). "
                "Identifies ~3-5% of CRC as Lynch-associated."
            ),
        },
        {
            "term": "RRSO (Risk-Reducing Bilateral Salpingo-Oophorectomy)",
            "full": "Prophylactic removal of fallopian tubes and ovaries to reduce BRCA1/2 cancer risk",
            "explanation": (
                "BRCA1: RRSO at 35-40 years (after childbearing) reduces ovarian/fallopian/peritoneal "
                "cancer risk by 80-96% AND reduces breast cancer risk by ~50% in premenopausal women. "
                "BRCA2: RRSO at 40-45 years. Short-term HRT post-RRSO for vasomotor symptoms is acceptable "
                "and does NOT negate the cancer risk reduction. Fallopian tube is the most common site of "
                "BRCA1/2-associated HGSC origin."
            ),
        },
        {
            "term": "Desmoid Tumour (APC/FAP)",
            "full": "Locally invasive fibromatosis — mesenteric, extra-abdominal; aggressive post-surgery",
            "explanation": (
                "Desmoid tumours in FAP: benign but locally aggressive fibromatosis; APC codon >1310 "
                "strongly associated. Mesenteric desmoids compress bowel/blood vessels/ureters. "
                "Surgery is a trigger — aggressive growth post-laparotomy. "
                "First-line: sulindac + tamoxifen. Second-line: imatinib. "
                "Avoid surgery if at all possible; if unavoidable, inform patient of desmoid trigger risk."
            ),
        },
    ]
