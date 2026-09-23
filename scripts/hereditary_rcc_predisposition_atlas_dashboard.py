#!/usr/bin/env python3
"""Hereditary-Renal-Cell-Carcinoma-Predisposition-Atlas -- Complete 8-Gene Reference
VHL     (Von Hippel-Lindau tumour suppressor; 284aa; 3p25.3; AD LOF;
         Von Hippel-Lindau Syndrome;
         clear cell RCC 80-85% lifetime -- HIGHEST single gene;
         haemangioblastoma CNS/retina PATHOGNOMONIC;
         phaeochromocytoma + pancreatic cysts + polycythaemia (EPO);
         belzutifan (HIF-2α inhibitor) FDA 2021 VHL-disease RCC;
         seed SEED_BASE+0) .
FH      (Fumarate hydratase; 510aa; 1q43; AD LOF;
         Hereditary Leiomyomatosis and Renal Cell Carcinoma (HLRCC);
         type 2 papillary RCC -- VERY AGGRESSIVE; NO watchful waiting;
         2SC IHC PATHOGNOMONIC; uterine fibroids PATHOGNOMONIC (women);
         annual MRI from age 8yr MANDATORY;
         seed SEED_BASE+1) .
FLCN    (Folliculin; 579aa; 17p11.2; AD LOF;
         Birt-Hogg-Dubé Syndrome;
         chromophobe/hybrid oncocytic RCC 25-34% lifetime;
         fibrofolliculomas skin PATHOGNOMONIC; pulmonary cysts + pneumothorax 30-50%;
         MTOR pathway; cabozantinib/everolimus;
         seed SEED_BASE+2) .
SDHB    (Succinate dehydrogenase iron-sulfur subunit B; 280aa; 1p36.13; AD LOF;
         SDH-deficient PPGL/RCC Syndrome;
         malignant phaeochromocytoma 40%; SDH-deficient RCC 3-5%;
         succinate IHC accumulation PATHOGNOMONIC; SDHB IHC loss PATHOGNOMONIC;
         sunitinib/pazopanib for RCC; 177Lu-DOTATATE PPGL;
         seed SEED_BASE+3) .
BAP1    (BRCA1-associated protein 1; 729aa; 3p21.1; AD LOF;
         Tumor Predisposition Syndrome (TPDS/BAP1-TPDS);
         clear cell RCC ~50% lifetime; uveal melanoma HIGHEST risk;
         mesothelioma 30-60%; MBAITs (melanocytic BAP1-associated intradermal tumour) PATHOGNOMONIC;
         BAP1 IHC nuclear loss PATHOGNOMONIC; nivolumab + ipilimumab;
         seed SEED_BASE+4) .
MET     (MET proto-oncogene (hepatocyte growth factor receptor); 1390aa; 7q31.2; AD GOF;
         Hereditary Papillary Renal Cell Carcinoma type 1 (HPRC);
         type 1 papillary RCC; bilateral/multifocal; MET exon 14-18 GOF hotspots;
         cabozantinib FDA 2016; tepotinib FDA 2021; 3cm active surveillance rule;
         seed SEED_BASE+5) .
TSC2    (Tuberous sclerosis complex 2 / tuberin; 1807aa; 16p13.3; AD LOF;
         Tuberous Sclerosis Complex type 2;
         angiomyolipoma PATHOGNOMONIC; chromophobe/clear cell RCC 2-3%;
         cortical tubers PATHOGNOMONIC; subependymal giant cell astrocytoma (SEGA) PATHOGNOMONIC;
         everolimus mTOR FDA 2012; embolisation AML >4cm;
         seed SEED_BASE+6) .
PTEN    (Phosphatase and tensin homologue; 403aa; 10q23.31; AD LOF;
         Cowden Syndrome / PTEN Hamartoma Tumour Syndrome (PHTS);
         RCC 34% lifetime; macrocephaly PATHOGNOMONIC; Lhermitte-Duclos PATHOGNOMONIC;
         breast 85% lifetime; everolimus + lenvatinib FDA 2019 (RCC);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3358-3365)
"""
import random

SEED_BASE = 3358

ATLAS_GENES = [
    {
        "gene": "VHL",
        "protein": (
            "VHL -- 3p25.3 Autosomal-Dominant-LOF -- 284aa -- "
            "Von-Hippel-Lindau-30kDa-HIF-Alpha-E3-Ligase-Substrate-Adaptor-"
            "ccRCC-80-85pct-HIGHEST-Single-Gene-"
            "Haemangioblastoma-CNS-Retina-PATHOGNOMONIC-"
            "Belzutifan-HIF2alpha-Inhibitor-FDA2021-OMIM-193300"
        ),
        "locus": "3p25.3",
        "protein_size": (
            "284 aa / 30 kDa / 3p25.3 VHL encodes the Von Hippel-Lindau tumour suppressor: "
            "STRUCTURE: "
            "  284 aa / 30 kDa; two functional domains: alpha domain (aa 63-154) and beta domain (aa 155-192); "
            "  VHL protein is the substrate-recognition subunit of the E3 ubiquitin ligase complex "
            "  (VHL-elongin B-elongin C-Cul2-Rbx1); "
            "  Beta domain (hydrophobic core): direct binding to HIF-1α and HIF-2α proline hydroxylation sites; "
            "  HIF-α pVHL recognition: VHL captures hydroxylated HIF-1α/HIF-2α → ubiquitination → proteasomal degradation; "
            "  VHL LOF → HIF-1α/HIF-2α accumulate → transcriptional activation of VEGF, PDGF, EPO, CAIX, GLUT1; "
            "  Pseudohypoxic state drives angiogenesis, RCC proliferation, polycythaemia (EPO); "
            "  Type 1 VHL (LOF only): ccRCC, haemangioblastoma, EPO; phaeochromocytoma RARE type 1; "
            "  Type 2 VHL (GOF + LOF): phaeochromocytoma; A=low ccRCC; B=high ccRCC; C=pheo only (no RCC); "
            "VHL SYNDROME MANIFESTATIONS: "
            "  CLEAR CELL RCC: 80-85% lifetime risk — HIGHEST single-gene hereditary RCC; "
            "  Bilateral, multifocal; surgery when any lesion ≥3cm (3cm rule); "
            "  Belzutifan FDA 2021 (first HIF-2α inhibitor for VHL-disease RCC/haemangioblastoma/pNET); "
            "  VEGFR TKI (sunitinib, pazopanib, cabozantinib): systemic disease; "
            "  HAEMANGIOBLASTOMA: CNS (cerebellum, brainstem, spinal cord) + retina PATHOGNOMONIC; "
            "  Retinal haemangioblastoma (von Hippel tumour): earliest manifestation (mean age 25yr); "
            "  Annual ophthalmology from age 1yr; annual MRI brain + spine from age 11yr; "
            "  PHAEOCHROMOCYTOMA: type 2 only (5-10% VHL overall); annual biochemistry from 5yr; "
            "  PANCREATIC MANIFESTATIONS: cysts (common, benign), serous cystadenoma, pNET (10-17%); "
            "  POLYCYTHAEMIA: EPO-driven from ccRCC or haemangioblastoma; "
            "  ENDOLYMPHATIC SAC TUMOUR (ELST): inner ear; hearing loss + tinnitus ELST presentation; "
            "SURVEILLANCE VHL: "
            "  Annual ophthalmology (retinal haemangioblastoma) from age 1yr; "
            "  Annual brain + spine MRI from age 11yr (haemangioblastoma); "
            "  Annual abdominal MRI from age 15yr (RCC + pNET + cysts); "
            "  Annual biochemistry (plasma metanephrines/urine catecholamines) from age 5yr (pheo); "
            "  Annual audiological assessment for ELST; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Von Hippel-Lindau (VHL) Syndrome",
        "rcc_risk": "Clear cell RCC 80-85% lifetime — HIGHEST single gene; bilateral/multifocal; belzutifan HIF-2α FDA 2021; 3cm surgery rule",
        "pathognomonic": "Haemangioblastoma CNS/retinal PATHOGNOMONIC; VHL + ccRCC + pheo + pancreatic cysts PATHOGNOMONIC triad",
        "key_avoid": "DELAY SURGERY BEYOND 3cm — VHL RCC 3cm rule: operate at 3cm to prevent metastasis; do NOT watch >3cm lesions",
        "surveillance": "Annual ophthalmology from 1yr; annual brain+spine MRI from 11yr; annual abdominal MRI from 15yr; biochemistry from 5yr",
        "targeted_rx": "Belzutifan (HIF-2α inhibitor) FDA 2021 VHL-disease; sunitinib/pazopanib/cabozantinib metastatic; nivolumab+ipilimumab IO",
        "key_rule": "BELZUTIFAN FDA 2021 — first HIF-2α inhibitor for VHL-RCC/haemangioblastoma/pNET; VEGFR TKI metastatic; 3cm surgery rule",
    },
    {
        "gene": "FH",
        "protein": (
            "FH -- 1q43 Autosomal-Dominant-LOF -- 510aa -- "
            "Fumarate-Hydratase-55kDa-TCA-Cycle-Enzyme-Fumarate-Malate-"
            "HLRCC-Type2-Papillary-RCC-VERY-AGGRESSIVE-NO-Watchful-Waiting-"
            "2SC-IHC-PATHOGNOMONIC-Uterine-Fibroids-PATHOGNOMONIC-Annual-MRI-8yr-OMIM-150800"
        ),
        "locus": "1q43",
        "protein_size": (
            "510 aa / 55 kDa / 1q43 FH encodes fumarate hydratase (fumarase): "
            "STRUCTURE: "
            "  510 aa / 55 kDa; homotetrameric TCA cycle enzyme; dual localisation: mitochondria + cytoplasm; "
            "  N-terminal mitochondrial targeting sequence (aa 1-30): cleaved in mitochondria; "
            "  Active site: four subunit-bridging residues; catalyses fumarate → malate (TCA); "
            "  FH LOF → fumarate accumulates → competitive inhibition of PHD enzymes (prolyl hydroxylases); "
            "  PHD inhibition → HIF-1α/HIF-2α NOT hydroxylated → NOT captured by VHL → accumulate; "
            "  Fumarate also inhibits α-KG-dependent dioxygenases (TET2/EGLN): hypermethylation; "
            "  2-succinocysteine (2SC): fumarate + cysteine → 2SC; 2SC IHC detects fumarate accumulation; "
            "  2SC IHC POSITIVE PATHOGNOMONIC FH-deficiency in any tumour; "
            "HLRCC (Hereditary Leiomyomatosis and Renal Cell Carcinoma): "
            "  TYPE 2 PAPILLARY RCC: VERY AGGRESSIVE; early metastasis even from small primary; "
            "  NO watchful waiting — aggressive surgery at detection regardless of size; "
            "  5yr OS <50% metastatic; curative surgery is only curative option; "
            "  Single kidneys at presentation common (bilateral RCC rare vs VHL); "
            "  Histology: type 2 papillary (large nucleoli with orange/eosinophilic nucleolus PATHOGNOMONIC); "
            "  BEVACIZUMAB + ERLOTINIB: standard systemic HLRCC RCC (VEGF + EGFR pathway); "
            "  2SC IHC: 100% sensitivity for FH-deficiency in RCC; "
            "UTERINE FIBROIDS (women): "
            "  Uterine fibroids (leiomyomas): PATHOGNOMONIC when multiple, young-onset (20s), large; "
            "  Cutaneous leiomyomas: multiple, painful lesions on trunk/extremities PATHOGNOMONIC; "
            "  Symptomatic management: progestins/GnRH antagonists; myomectomy if fertility desired; "
            "SURVEILLANCE FH/HLRCC: "
            "  Annual abdominal MRI from age 8yr MANDATORY (RCC surveillance); "
            "  Annual gynaecological USS + examination (women) from menarche; "
            "  Skin examination annually (cutaneous leiomyomas); "
            "  Surgical consultation immediately on any detected RCC (NO watchful waiting); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Hereditary Leiomyomatosis and Renal Cell Carcinoma (HLRCC)",
        "rcc_risk": "Type 2 papillary RCC — VERY AGGRESSIVE; NO watchful waiting; 2SC IHC PATHOGNOMONIC; annual MRI from age 8yr",
        "pathognomonic": "2SC IHC PATHOGNOMONIC FH-deficiency; uterine fibroids young-onset large PATHOGNOMONIC; cutaneous leiomyomas PATHOGNOMONIC",
        "key_avoid": "WATCHFUL WAITING — FH/HLRCC type 2 papillary RCC is extremely aggressive; operate immediately at any detection regardless of size",
        "surveillance": "Annual abdominal MRI from age 8yr MANDATORY; annual gynaecological USS (women); skin exam; NO watchful waiting",
        "targeted_rx": "Aggressive surgery immediately at detection; bevacizumab + erlotinib systemic HLRCC; MET inhibitors (HLRCC can have MET activation)",
        "key_rule": "NO WATCHFUL WAITING — HLRCC type 2 papillary RCC metastasises early even from small lesions; bevacizumab+erlotinib systemic",
    },
    {
        "gene": "FLCN",
        "protein": (
            "FLCN -- 17p11.2 Autosomal-Dominant-LOF -- 579aa -- "
            "Folliculin-64kDa-MTOR-AMPK-Lysosomal-Scaffold-"
            "Birt-Hogg-Dube-Chromophobe-Hybrid-Oncocytic-RCC-25-34pct-"
            "Fibrofolliculomas-Skin-PATHOGNOMONIC-Pulmonary-Cysts-Pneumothorax-30-50pct-"
            "Cabozantinib-Everolimus-MTOR-OMIM-135150"
        ),
        "locus": "17p11.2",
        "protein_size": (
            "579 aa / 64 kDa / 17p11.2 FLCN encodes folliculin: "
            "STRUCTURE: "
            "  579 aa / 64 kDa; lysosomal surface scaffold protein; intrinsically disordered; "
            "  FLCN interacts with FNIP1/FNIP2 (folliculin-interacting proteins) and AMPK; "
            "  FLCN-FNIP complex: positive regulator of MTOR complex 1 (mTORC1) at the lysosome; "
            "  FLCN LOF → constitutive AMPK activation → mTORC1 suppression inconsistently; "
            "  Net effect: dysregulated MTOR signalling + activated HIF pathway via AMPK; "
            "  FLCN also regulates lysosomal biogenesis via TFEB; "
            "BIRT-HOGG-DUBÉ (BHD) SYNDROME: "
            "  RCC RISK: 25-34% lifetime; mostly chromophobe (34%) and hybrid oncocytic (50%); "
            "  Clear cell RCC also elevated (6% of FLCN RCC); "
            "  Bilateral, multifocal; 3cm surgery rule (same as VHL); "
            "  Chromophobe RCC in BHD: better prognosis than FH-type; mTOR pathway driver; "
            "  FIBROFOLLICULOMAS: hair follicle hamartomas (skin); face/neck/trunk PATHOGNOMONIC BHD; "
            "  Trichodiscomas: related follicular lesion; pathognomonic in BHD context; "
            "  Acrochordons (skin tags): associated feature; "
            "  PULMONARY CYSTS + PNEUMOTHORAX: "
            "  Multiple bilateral pulmonary cysts (lower lobe predominant) in 80-90% FLCN; "
            "  Spontaneous pneumothorax 30-50% lifetime (vs 1% general); "
            "  Avoid scuba diving (barotrauma) and smoking; "
            "  CT chest at diagnosis then every 3-5yr; first pneumothorax → thoracoscopic surgery with pleurodesis; "
            "TREATMENT FLCN BHD: "
            "  Surgery (partial nephrectomy if feasible) for localised RCC; "
            "  Everolimus (mTOR inhibitor): rational for FLCN chromophobe RCC; sunitinib second-line; "
            "  Cabozantinib: tyrosine kinase inhibitor; MET + VEGFR2 activity; "
            "  Nivolumab + ipilimumab: IO checkpoint for metastatic chromophobe; "
            "SURVEILLANCE FLCN BHD: "
            "  Annual abdominal MRI from age 20yr (chromophobe RCC surveillance); "
            "  Annual skin exam (fibrofolliculomas — cosmetic + diagnostic); "
            "  CT chest at diagnosis; avoid scuba diving; advise pneumothorax management; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Birt-Hogg-Dubé (BHD) Syndrome",
        "rcc_risk": "Chromophobe/hybrid oncocytic RCC 25-34% lifetime; bilateral/multifocal; 3cm surgery rule; everolimus mTOR pathway",
        "pathognomonic": "Fibrofolliculomas (hair follicle hamartomas) skin PATHOGNOMONIC BHD; bilateral pulmonary cysts + pneumothorax 30-50%",
        "key_avoid": "SCUBA DIVING — pulmonary cysts + pneumothorax risk; avoid barotrauma; smoking accelerates pulmonary disease",
        "surveillance": "Annual abdominal MRI from age 20yr; annual skin exam; CT chest at diagnosis; 3cm surgery rule for RCC",
        "targeted_rx": "Partial nephrectomy localised; everolimus mTOR inhibitor chromophobe; cabozantinib metastatic; nivolumab+ipilimumab IO",
        "key_rule": "BILATERAL PULMONARY CYSTS + PNEUMOTHORAX 30-50% — avoid scuba diving; fibrofolliculomas PATHOGNOMONIC; annual abdominal MRI from 20yr",
    },
    {
        "gene": "SDHB",
        "protein": (
            "SDHB -- 1p36.13 Autosomal-Dominant-LOF -- 280aa -- "
            "Succinate-Dehydrogenase-Iron-Sulfur-Subunit-32kDa-Complex-II-"
            "SDH-Deficient-PPGL-RCC-Malignant-Pheo-40pct-"
            "Succinate-IHC-Accumulation-PATHOGNOMONIC-SDHB-IHC-Loss-PATHOGNOMONIC-"
            "177Lu-DOTATATE-PPGL-Sunitinib-RCC-OMIM-185470"
        ),
        "locus": "1p36.13",
        "protein_size": (
            "280 aa / 32 kDa / 1p36.13 SDHB encodes succinate dehydrogenase subunit B (iron-sulfur protein): "
            "STRUCTURE: "
            "  280 aa / 32 kDa; nuclear-encoded mitochondrial protein; inner mitochondrial membrane; "
            "  SDH Complex II: four subunits (SDHA, SDHB, SDHC, SDHD) + assembly factors (SDHAF1, SDHAF2); "
            "  SDHB: iron-sulfur clusters (3Fe-4S, 4Fe-4S, 2Fe-2S) that mediate electron transfer; "
            "  Function: oxidises succinate → fumarate (TCA cycle) + reduces ubiquinone → ubiquinol (ETC); "
            "  SDHB LOF → succinate accumulates → competitive PHD enzyme inhibition → HIF-1α/HIF-2α accumulate; "
            "  Same mechanism as FH LOF; succinate IHC detects accumulation in tumour cytoplasm; "
            "  SDHB IHC: SDHB antibody stains mitochondria normally; absent in any SDH-LOF tumour; "
            "HEREDITARY SDH-DEFICIENT PPGL/RCC: "
            "  SDHB: highest malignancy risk of all SDHx; malignant PPGL 40% lifetime; "
            "  Phaeochromocytoma 15-20%; Paraganglioma 30-40% (extra-adrenal); "
            "  Head and neck paraganglioma: carotid body/glomus jugulare/glomus tympanicum; "
            "  SDH-DEFICIENT RCC: 3-5% SDHB carriers; chromophobe-like histology; "
            "  SUCCINATE IHC: cytoplasmic granular staining PATHOGNOMONIC SDH-deficiency; "
            "  SDHB IHC: loss PATHOGNOMONIC in any SDH-mutant tumour (all SDHx LOF types); "
            "  Malignant criteria: PPGL: regional lymph node, hepatic, pulmonary, bone metastases; "
            "TREATMENT SDH/PPGL: "
            "  177Lu-DOTATATE (PRRT): peptide receptor radionuclide therapy FDA 2018 for SSTR2+ PPGL/PGL; "
            "  Sunitinib: systemic treatment for SDH-deficient/malignant PPGL; "
            "  Sunitinib/pazopanib: SDH-deficient RCC; "
            "  Surgical resection: localised PPGL (laparoscopic preferred); "
            "  Pre-surgical alpha-blockade (phenoxybenzamine): MANDATORY before pheo surgery; "
            "SURVEILLANCE SDHB: "
            "  Annual plasma/urine catecholamines/metanephrines from age 6yr; "
            "  Annual MRI whole-body (MRI preferred: no radiation) from age 6yr; "
            "  Annual abdominal MRI (SDH-RCC surveillance); "
            "  68Ga-DOTATATE PET-CT: most sensitive for SSTR2+ PGL; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "SDH-deficient PPGL/RCC Syndrome (SDHB-associated Hereditary PPGL)",
        "rcc_risk": "SDH-deficient RCC 3-5% SDHB; malignant PPGL 40% SDHB — HIGHEST SDHx malignancy; succinate IHC PATHOGNOMONIC",
        "pathognomonic": "Succinate IHC cytoplasmic accumulation PATHOGNOMONIC; SDHB IHC nuclear loss PATHOGNOMONIC; SDHB highest malignancy of SDHx",
        "key_avoid": "SURGERY WITHOUT ALPHA-BLOCKADE — pre-surgical phenoxybenzamine MANDATORY before phaeochromocytoma/paraganglioma resection; hypertensive crisis risk",
        "surveillance": "Annual plasma/urine catecholamines from 6yr; annual whole-body MRI from 6yr; 68Ga-DOTATATE PET most sensitive PGL",
        "targeted_rx": "177Lu-DOTATATE FDA 2018 SSTR2+ PPGL; sunitinib SDH-deficient RCC; sunitinib/pazopanib malignant PPGL; alpha-block pre-op",
        "key_rule": "ALPHA-BLOCKADE MANDATORY before pheo surgery; 177Lu-DOTATATE SSTR2+ PPGL; SDHB HIGHEST SDHx malignancy risk 40%; annual MRI from 6yr",
    },
    {
        "gene": "BAP1",
        "protein": (
            "BAP1 -- 3p21.1 Autosomal-Dominant-LOF -- 729aa -- "
            "BRCA1-Associated-Protein1-80kDa-Ubiquitin-C-Terminal-Hydrolase-"
            "TPDS-ccRCC-50pct-Uveal-Melanoma-HIGHEST-Mesothelioma-30-60pct-"
            "MBAITs-Cutaneous-PATHOGNOMONIC-BAP1-IHC-Nuclear-Loss-PATHOGNOMONIC-"
            "Nivolumab-Ipilimumab-IO-OMIM-603089"
        ),
        "locus": "3p21.1",
        "protein_size": (
            "729 aa / 80 kDa / 3p21.1 BAP1 encodes BRCA1-associated protein-1: "
            "STRUCTURE: "
            "  729 aa / 80 kDa; nuclear deubiquitylase (DUB); ubiquitin C-terminal hydrolase domain (aa 1-240); "
            "  UCH domain: cleaves K63 ubiquitin from H2A (histone 2A); epigenetic tumour suppressor; "
            "  BAP1 anchors the Polycomb Repressive Deubiquitylase (PR-DUB) complex (with ASXL1/ASXL2, FOXK1); "
            "  BRCA1 binding domain (aa 350-550): BRCA1 C-terminus interaction; "
            "  BAP1 LOF → H2Aub accumulation → Polycomb-mediated gene silencing dysregulation; "
            "  BAP1 also regulates DNA damage response and cell cycle checkpoint; "
            "  IHC: BAP1 normally stains nuclei; loss of nuclear staining = functional LOF; "
            "BAP1 TUMOR PREDISPOSITION SYNDROME (TPDS): "
            "  UVEAL MELANOMA: 25-50% lifetime; HIGHEST germline risk for uveal melanoma; "
            "  Metastatic uveal melanoma: liver dominant (90%); tebentafusp (T-cell receptor bispecific) FDA 2022; "
            "  CLEAR CELL RCC: ~50% lifetime risk; 2-3x elevated; "
            "  BAP1-null ccRCC: poor differentiation; BAP1 IHC nuclear loss PATHOGNOMONIC; "
            "  MESOTHELIOMA: 30-60% lifetime; pleural (most common), peritoneal, pericardial; "
            "  AVOID ASBESTOS/SILICA ABSOLUTELY — BAP1 + exposure = synergistic mesothelioma risk; "
            "  MBAITs (Melanocytic BAP1-Associated Intradermal Tumours): "
            "  Cutaneous BAP1-null naevi (spitzoid morphology); multiple, raised, pink-brown; PATHOGNOMONIC BAP1-TPDS; "
            "  Histologically distinct from melanoma but contain BAP1-null cells; skin biopsy diagnostic; "
            "  CUTANEOUS MELANOMA: 5-10% BAP1-TPDS; "
            "TREATMENT BAP1: "
            "  Nivolumab + ipilimumab: IO checkpoint for metastatic ccRCC (BAP1-null enriched); "
            "  Tebentafusp FDA 2022: HLA-A*02:01+ metastatic uveal melanoma; "
            "  Cytoreductive surgery: peritoneal mesothelioma BAP1-TPDS; "
            "  Belzutifan: investigational for BAP1-null ccRCC (HIF pathway active); "
            "SURVEILLANCE BAP1: "
            "  Annual ophthalmological MRI/USS + dilated fundus exam from age 11yr (uveal melanoma); "
            "  Annual abdominal MRI from age 30yr (ccRCC + peritoneal); "
            "  Annual CT chest (pleural mesothelioma) from age 30yr; "
            "  Annual skin surveillance (MBAITs + cutaneous melanoma); "
            "  AVOID ASBESTOS ABSOLUTELY; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "BAP1 Tumor Predisposition Syndrome (BAP1-TPDS)",
        "rcc_risk": "Clear cell RCC ~50% lifetime; BAP1-null IHC PATHOGNOMONIC; uveal melanoma HIGHEST risk; mesothelioma 30-60%; IO nivolumab+ipilimumab",
        "pathognomonic": "MBAITs (melanocytic BAP1-associated intradermal tumours) PATHOGNOMONIC; BAP1 IHC nuclear loss PATHOGNOMONIC; uveal melanoma + RCC + mesothelioma triad",
        "key_avoid": "ASBESTOS/SILICA ABSOLUTELY — BAP1 + asbestos exposure → synergistic mesothelioma risk; HLA-A*02:01 test for tebentafusp eligibility",
        "surveillance": "Annual ophthalmological exam from 11yr; annual abdominal MRI from 30yr; annual CT chest from 30yr; annual skin surveillance",
        "targeted_rx": "Nivolumab+ipilimumab IO ccRCC; tebentafusp FDA 2022 uveal melanoma HLA-A*02:01+; belzutifan investigational BAP1-null ccRCC",
        "key_rule": "AVOID ASBESTOS ABSOLUTELY — BAP1 + asbestos = synergistic mesothelioma; MBAITs PATHOGNOMONIC; tebentafusp FDA 2022 uveal melanoma",
    },
    {
        "gene": "MET",
        "protein": (
            "MET -- 7q31.2 Autosomal-Dominant-GOF -- 1390aa -- "
            "MET-RTK-156kDa-HGF-Receptor-Tyrosine-Kinase-HPRC-"
            "Type1-Papillary-RCC-Bilateral-Multifocal-GOF-Hotspots-"
            "Cabozantinib-FDA2016-Tepotinib-FDA2021-3cm-Active-Surveillance-OMIM-164860"
        ),
        "locus": "7q31.2",
        "protein_size": (
            "1390 aa / 156 kDa / 7q31.2 MET encodes the hepatocyte growth factor receptor (HGF-R / c-MET): "
            "STRUCTURE: "
            "  1390 aa / 156 kDa; receptor tyrosine kinase; single-pass type I transmembrane; "
            "  Extracellular alpha-chain + beta-chain (disulfide-linked): Sema domain (HGF binding); "
            "  Transmembrane domain (aa 955-975); "
            "  Intracellular kinase domain (aa 1073-1346): activation loop Tyr1234/Tyr1235 (phosphorylation sites); "
            "  Juxtamembrane domain (Tyr1003): CBL-binding negative regulatory site; "
            "  Multisubstrate docking site (Tyr1349/Tyr1356): PI3K, STAT3, SHP2, GAB1 binding; "
            "  MET GOF mutations → constitutive kinase activation → MAPK, PI3K-AKT, STAT3 pathways; "
            "HPRC (Hereditary Papillary Renal Cell Carcinoma type 1): "
            "  TYPE 1 PAPILLARY RCC: basophilic papillary RCC; bilateral multifocal (3-3000 lesions per kidney); "
            "  Germline GOF mutations: exon 14-18 activating (Tyr1235Asp most common); "
            "  LATE ONSET: mean age 60s; high penetrance but delayed; "
            "  ACTIVE SURVEILLANCE: 3cm rule — monitor until lesion ≥3cm then operate; "
            "  Surgery delayed as long as possible (bilateral disease → staged partial nephrectomies); "
            "  MET INHIBITORS: "
            "  Cabozantinib FDA 2016 (VEGFR2 + MET + AXL): standard of care advanced RCC; "
            "  Tepotinib FDA 2021 (selective MET inhibitor): MET exon 14 skipping NSCLC; investigational HPRC; "
            "  Savolitinib: selective MET; MRC SAVOIR trial papillary RCC; "
            "  Capmatinib FDA 2020: selective MET; exon 14 skipping NSCLC; investigational HPRC; "
            "SURVEILLANCE MET/HPRC: "
            "  Annual abdominal MRI from age 30yr; "
            "  3cm active surveillance rule; "
            "  Staged partial nephrectomies when lesions reach 3cm (bilateral disease); "
            "  Nephron-sparing approach MANDATORY (bilateral disease → renal preservation essential); "
        ),
        "inheritance": "AD GOF",
        "syndrome": "Hereditary Papillary Renal Cell Carcinoma type 1 (HPRC)",
        "rcc_risk": "Type 1 papillary RCC; bilateral multifocal; GOF kinase activation; 3cm active surveillance rule; cabozantinib FDA 2016",
        "pathognomonic": "Bilateral multifocal type 1 papillary RCC PATHOGNOMONIC MET GOF; late-onset bilateral papillary (3-3000 lesions/kidney)",
        "key_avoid": "NEPHRECTOMY FIRST — bilateral disease requires staged partial nephrectomies; nephron preservation MANDATORY; avoid total nephrectomy",
        "surveillance": "Annual abdominal MRI from age 30yr; 3cm active surveillance rule; staged partial nephrectomies for bilateral disease",
        "targeted_rx": "Cabozantinib FDA 2016 (MET+VEGFR2+AXL); tepotinib FDA 2021 selective MET; savolitinib investigational papillary RCC; partial nephrectomy",
        "key_rule": "3cm ACTIVE SURVEILLANCE RULE — bilateral multifocal disease; partial nephrectomies staged; cabozantinib GOF MET; nephron sparing MANDATORY",
    },
    {
        "gene": "TSC2",
        "protein": (
            "TSC2 -- 16p13.3 Autosomal-Dominant-LOF -- 1807aa -- "
            "Tuberin-200kDa-mTORC1-RAS-GAP-MTOR-Regulator-"
            "TSC-Angiomyolipoma-PATHOGNOMONIC-Cortical-Tubers-PATHOGNOMONIC-SEGA-PATHOGNOMONIC-"
            "Everolimus-mTOR-FDA2012-Embolisation-AML-4cm-OMIM-613254"
        ),
        "locus": "16p13.3",
        "protein_size": (
            "1807 aa / 200 kDa / 16p13.3 TSC2 encodes tuberin: "
            "STRUCTURE: "
            "  1807 aa / 200 kDa; RAP GAP (GTPase-activating protein) domain (aa 1517-1674): Rheb-GAP; "
            "  Forms heterodimer with TSC1 (hamartin); TSC1-TSC2 complex is critical negative regulator of mTORC1; "
            "  Rheb-GTP activates mTORC1; TSC2 GAP converts Rheb-GTP → Rheb-GDP → mTORC1 inhibition; "
            "  TSC2 LOF → Rheb remains GTP-loaded → constitutive mTORC1 activation; "
            "  mTORC1 activation → S6K1/4EBP1 phosphorylation → increased protein synthesis, cell growth; "
            "  PTEN-PI3K-AKT-TSC2 axis: AKT phosphorylates TSC2 → inhibits GAP activity; "
            "  TSC2 mutation: 65% of TSC (vs TSC1 35%); TSC2 associated with SEVERER phenotype; "
            "TUBEROUS SCLEROSIS COMPLEX (TSC) TYPE 2: "
            "  ANGIOMYOLIPOMA (AML): fat + blood vessel + smooth muscle; PATHOGNOMONIC in TSC; "
            "  Bilateral renal AML in 55-75% TSC; treatment if >4cm (embolisation preferred; partial nephrectomy); "
            "  EMBOLISATION (TAE): recommended for AML >4cm to prevent haemorrhage (Wunderlich syndrome); "
            "  Avoid everolimus-related immunosuppression if active infection before embolisation; "
            "  RCC: chromophobe RCC 2-3% TSC2; also clear cell RCC; typically favourable prognosis; "
            "  CORTICAL TUBERS: benign hamartomas; PATHOGNOMONIC TSC; cause epilepsy (85% TSC); "
            "  SUBEPENDYMAL GIANT CELL ASTROCYTOMA (SEGA): PATHOGNOMONIC TSC; obstructs CSF; everolimus FDA 2012; "
            "  PULMONARY LYMPHANGIOLEIOMYOMATOSIS (LAM): women of reproductive age; progressive cystic lung; "
            "  Sirolimus/everolimus: FDA approved for TSC-LAM; "
            "  SKIN LESIONS: ash leaf macule PATHOGNOMONIC (white hypopigmented macule); "
            "  Facial angiofibromas PATHOGNOMONIC (red papules, nose-cheeks); "
            "  Shagreen patch PATHOGNOMONIC (collagenoma back/lumbar); "
            "TREATMENT TSC2: "
            "  Everolimus (mTOR inhibitor): SEGA FDA 2012; renal AML FDA 2012; TSC-LAM FDA 2016; "
            "  Embolisation: preferred over partial nephrectomy for AML >4cm (nephron sparing); "
            "  Vigabatrin: first-line infantile spasms in TSC; "
            "  Rapalogs (sirolimus): LAM + other TSC manifestations; "
            "SURVEILLANCE TSC2: "
            "  Annual brain MRI from diagnosis (SEGA); "
            "  Annual renal USS/MRI from age 1yr (AML surveillance); "
            "  Annual pulmonary function + CT chest from age 18 (LAM women); "
            "  Annual ophthalmology (retinal hamartoma, not as high risk as VHL); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Tuberous Sclerosis Complex type 2 (TSC2)",
        "rcc_risk": "Angiomyolipoma PATHOGNOMONIC; chromophobe RCC 2-3%; everolimus FDA 2012 AML/SEGA/LAM; embolisation AML >4cm",
        "pathognomonic": "Cortical tubers + SEGA + angiomyolipoma triad PATHOGNOMONIC TSC; ash leaf macule PATHOGNOMONIC; facial angiofibromas PATHOGNOMONIC",
        "key_avoid": "AML >4cm WITHOUT EMBOLISATION — haemorrhage (Wunderlich syndrome); avoid surgery first for AML; embolisation preserves nephrons",
        "surveillance": "Annual brain MRI (SEGA); annual renal USS/MRI from 1yr (AML); annual pulmonary function from 18yr (LAM women); annual ophthalmology",
        "targeted_rx": "Everolimus FDA 2012 (SEGA+AML+LAM); embolisation AML >4cm; vigabatrin infantile spasms; sirolimus LAM",
        "key_rule": "EVEROLIMUS FDA 2012 — SEGA + AML + LAM all mTOR-driven; embolisation preferred over nephrectomy for AML >4cm; angiomyolipoma PATHOGNOMONIC",
    },
    {
        "gene": "PTEN",
        "protein": (
            "PTEN -- 10q23.31 Autosomal-Dominant-LOF -- 403aa -- "
            "PTEN-47kDa-PI3K-AKT-mTOR-Phosphatase-"
            "Cowden-PHTS-RCC-34pct-Breast-85pct-"
            "Macrocephaly-PATHOGNOMONIC-Lhermitte-Duclos-PATHOGNOMONIC-"
            "Everolimus-Lenvatinib-FDA2019-RCC-OMIM-601728"
        ),
        "locus": "10q23.31",
        "protein_size": (
            "403 aa / 47 kDa / 10q23.31 PTEN encodes phosphatase and tensin homologue: "
            "STRUCTURE: "
            "  403 aa / 47 kDa; dual-specificity phosphatase (lipid + protein); "
            "  N-terminal PIP binding domain (aa 1-14): membrane anchoring; "
            "  Phosphatase domain (aa 14-185): active site Cys124; catalytic mechanism; "
            "  C2 domain (aa 186-351): Ca2+-independent membrane binding; "
            "  C-terminal tail (aa 352-403): PDZ-binding motif; regulation; "
            "  PTEN dephosphorylates PIP3 → PIP2; opposes PI3K; "
            "  PTEN LOF → PIP3 accumulates → constitutive AKT activation → mTORC1 + mTORC2 activation; "
            "  mTOR hyperactivation → cell growth, proliferation, survival, metabolism; "
            "  PTEN also functions in nucleus (chromosomal stability, DNA repair); "
            "COWDEN SYNDROME / PHTS (PTEN HAMARTOMA TUMOUR SYNDROME): "
            "  RCC RISK: 34% lifetime (clear cell predominant; also papillary); elevated 5-7x general; "
            "  Annual abdominal USS from age 40yr; MRI preferred if available; "
            "  BREAST CANCER: 85% lifetime (women) — HIGHEST risk; annual breast MRI + mammography from 30; "
            "  MACROCEPHALY: head circumference >97th centile PATHOGNOMONIC (OFC >58cm adult women/60cm men); "
            "  LHERMITTE-DUCLOS (LDD): dysplastic cerebellar gangliocytoma PATHOGNOMONIC Cowden; "
            "  LDD = adult cerebellar lesion with tigroid striping on MRI → molecular PTEN testing MANDATORY; "
            "  THYROID CANCER: follicular 10% (multi-nodular goitre common); annual thyroid USS from 18yr; "
            "  ENDOMETRIAL CANCER: 28-44% lifetime; annual endometrial biopsy from 35yr; RRSO 40-45yr if surgery; "
            "  COLORECTAL: 9-16%; colonoscopy every 5yr from 35yr; "
            "  SKIN: trichilemmomas (follicular infundibulum); oral papillomatosis; PATHOGNOMONIC Cowden; "
            "TREATMENT PTEN/COWDEN RCC: "
            "  Everolimus + lenvatinib FDA 2019 (previously treated advanced RCC); "
            "  Nivolumab + ipilimumab: IO first-line metastatic; "
            "  Partial nephrectomy: localised RCC (bilateral risk); "
            "  AKT inhibitors (capivasertib, ipatasertib): clinical trials PTEN-null RCC; "
            "SURVEILLANCE PTEN COWDEN: "
            "  Annual breast MRI + mammography from age 30 (women); "
            "  Annual thyroid USS from age 18yr; "
            "  Annual endometrial biopsy from age 35yr (women); "
            "  Annual abdominal USS from 40yr (RCC); colonoscopy every 5yr from 35yr; "
            "  Annual brain MRI at diagnosis (Lhermitte-Duclos — adults with cerebellar mass = test PTEN); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Cowden Syndrome / PTEN Hamartoma Tumour Syndrome (PHTS)",
        "rcc_risk": "RCC 34% lifetime (clear cell, 5-7x elevated); macrocephaly PATHOGNOMONIC; Lhermitte-Duclos PATHOGNOMONIC; everolimus+lenvatinib FDA 2019",
        "pathognomonic": "Macrocephaly >97th centile PATHOGNOMONIC; Lhermitte-Duclos cerebellar PATHOGNOMONIC; trichilemmomas skin PATHOGNOMONIC Cowden",
        "key_avoid": "ADULT CEREBELLAR MASS WITHOUT PTEN TESTING — Lhermitte-Duclos PATHOGNOMONIC Cowden; test PTEN immediately; oral papillomatosis may precede diagnosis",
        "surveillance": "Annual breast MRI from 30 (women); annual thyroid USS from 18yr; annual endometrial biopsy from 35yr; abdominal USS from 40yr",
        "targeted_rx": "Everolimus + lenvatinib FDA 2019 (RCC); nivolumab+ipilimumab IO; AKT inhibitors investigational PTEN-null; partial nephrectomy localised",
        "key_rule": "LHERMITTE-DUCLOS PATHOGNOMONIC — adult cerebellar dysplastic gangliocytoma = test PTEN; macrocephaly PATHOGNOMONIC; RCC 34% lifetime",
    },
]

# Tumour types per gene
TUMOUR_TYPES_BY_GENE: dict = {
    "VHL":  ["Clear Cell RCC VHL (bilateral/multifocal)", "Haemangioblastoma CNS (cerebellum/spinal)", "Retinal Haemangioblastoma VHL (von Hippel tumour)", "VHL + Phaeochromocytoma (type 2)", "VHL ccRCC + Pancreatic NET Concurrent"],
    "FH":   ["HLRCC Type 2 Papillary RCC (aggressive)", "HLRCC Metastatic at Presentation", "Uterine Leiomyosarcoma FH", "Cutaneous Leiomyoma + RCC FH", "HLRCC Stage IV at Detection (small primary)"],
    "FLCN": ["Chromophobe RCC BHD (bilateral)", "Hybrid Oncocytic RCC BHD", "BHD Clear Cell RCC", "BHD + Spontaneous Pneumothorax", "Bilateral Chromophobe RCC BHD Multifocal"],
    "SDHB": ["Malignant Paraganglioma (extra-adrenal) SDHB", "Malignant Phaeochromocytoma SDHB", "SDH-Deficient RCC (chromophobe-like)", "Head and Neck PGL SDHB (carotid body)", "PPGL + Bone Metastases SDHB (malignant)"],
    "BAP1": ["Clear Cell RCC BAP1-null", "Uveal Melanoma BAP1 + Hepatic Metastases", "Pleural Mesothelioma BAP1", "MBAITs (Cutaneous BAP1-null Nevi)", "BAP1 ccRCC + Mesothelioma Concurrent"],
    "MET":  ["Type 1 Papillary RCC HPRC Bilateral", "Type 1 Papillary RCC HPRC Multifocal (>10 lesions)", "HPRC Late-Onset Bilateral (age 60s)", "Type 1 Papillary RCC + Cabozantinib Response", "HPRC Staged Partial Nephrectomies Bilateral"],
    "TSC2": ["Renal Angiomyolipoma TSC (bilateral)", "Chromophobe RCC TSC2", "SEGA (Subependymal Giant Cell Astrocytoma) TSC", "TSC AML >4cm Haemorrhage (Wunderlich)", "TSC LAM (Pulmonary Lymphangioleiomyomatosis)"],
    "PTEN": ["Clear Cell RCC Cowden (PHTS)", "Breast Cancer Cowden (premenopausal)", "Lhermitte-Duclos (Cerebellar Gangliocytoma) PTEN", "Endometrial Cancer PHTS", "RCC + Breast + Thyroid Concurrent PTEN"],
}

# Pathogenic variants per gene
VARIANTS_BY_GENE: dict = {
    "VHL":  ["p.Tyr98His (type 2B — high RCC risk)", "VHL exon 1-3 deletion (MLPA)", "p.Arg167Trp (type 2A — moderate)", "p.Pro86Ser (type 2C — pheo only, no RCC)", "c.500G>T p.Ser167Ile (type 1 — ccRCC dominant)", "p.Gly114Asp (type 1 — ccRCC + haemangioblastoma)"],
    "FH":   ["p.Arg233His (catalytic site FH LOF)", "p.Arg190His (FH missense — 2SC accumulation confirmed)", "c.1431_1434del (frameshift)", "1q43 deletion MLPA (large deletion)", "p.Glu319Ter (truncating)", "p.Leu346Val (splice-affecting missense)"],
    "FLCN": ["c.1285del (frameshift — exon 11 hotspot)", "c.1733insC (insertion BHD)", "p.Phe157Leu (missense)", "17p11.2 deletion MLPA", "c.1285delC (founder Japanese BHD)", "p.Arg239Ter (truncating)"],
    "SDHB": ["p.Pro197Arg (missense pathogenic)", "p.Arg46Ter (truncating)", "p.Arg27Ter (truncating)", "1p36.13 deletion MLPA", "p.Cys101Tyr (iron-sulfur cluster disruption)", "c.137G>A p.Arg46Gln (pathogenic missense)"],
    "BAP1": ["p.Glu729Ter (C-terminus truncating)", "p.Trp38Ter (UCH domain LOF)", "3p21.1 deletion MLPA", "p.Phe170Leu (UCH domain missense)", "c.1967delG (frameshift)", "p.Gln684Ter (truncating exon 16)"],
    "MET":  ["p.Tyr1235Asp (activation loop GOF — most common HPRC)", "p.Tyr1248Asp (activation loop GOF)", "p.Met1268Thr (juxtamembrane GOF)", "p.Val1110Ile (GOF — attenuated)", "p.Glu1161Lys (GOF kinase)", "p.His1112Arg (GOF transmembrane/juxtamembrane)"],
    "TSC2": ["p.Arg611Gln (GTPase domain — severe)", "16p13.3 deletion contiguous TSC2+PKD1 (MLPA)", "p.Arg905Ter (truncating)", "c.1525C>T p.Arg509Ter", "p.Tyr1640Asn (GAP domain missense)", "c.2713del (frameshift exon 23)"],
    "PTEN": ["p.Arg130Gln (phosphatase active site LOF)", "p.Arg233Ter (truncating exon 7)", "10q23.31 deletion MLPA", "p.Gly129Glu (missense catalytic)", "c.697C>T p.Arg233Cys (missense)", "p.Tyr180Cys (C2 domain — milder)"],
}

# Treatment protocols per gene
TREATMENT_PROTOCOLS_BY_GENE: dict = {
    "VHL":  ["Belzutifan (HIF-2α inhibitor) FDA 2021 — VHL-disease RCC/haemangioblastoma/pNET", "VEGFR TKI (sunitinib, pazopanib, cabozantinib) for metastatic ccRCC", "Nivolumab + ipilimumab first-line metastatic (IO + IO)", "Partial nephrectomy when lesion reaches 3cm (bilateral 3cm rule)", "Laser photocoagulation/cryotherapy retinal haemangioblastoma", "VEGFR TKI + PD-1 combo (axitinib + pembrolizumab FDA 2019)"],
    "FH":   ["Aggressive surgery IMMEDIATELY at detection (no watchful waiting)", "Bevacizumab + erlotinib (VEGF + EGFR dual blockade) systemic HLRCC", "MET inhibitors (cabozantinib): MET activation common in HLRCC RCC", "Radical nephrectomy if partial not feasible (aggressive type 2 pRCC)", "Annual abdominal MRI from age 8yr (pre-emptive early detection)", "Platinum-based chemotherapy for metastatic (HIF pathway active)"],
    "FLCN": ["Partial nephrectomy (bilateral disease — nephron preservation)", "Everolimus (mTOR inhibitor) for metastatic chromophobe RCC", "Cabozantinib (MET + VEGFR2 + AXL) for advanced BHD RCC", "Thoracoscopic pleurodesis after first pneumothorax", "Nivolumab + ipilimumab IO for metastatic chromophobe", "Annual abdominal MRI from age 20yr"],
    "SDHB": ["177Lu-DOTATATE (PRRT) FDA 2018 SSTR2+ malignant PPGL", "Pre-surgical alpha-blockade (phenoxybenzamine) MANDATORY before pheo resection", "Sunitinib systemic SDH-deficient/malignant PPGL", "Sunitinib/pazopanib SDH-deficient RCC", "68Ga-DOTATATE PET-CT for staging PPGL", "Cabozantinib (MET + VEGFR2) advanced PPGL investigational"],
    "BAP1": ["Nivolumab + ipilimumab first-line metastatic BAP1-null ccRCC", "Tebentafusp FDA 2022 (HLA-A*02:01+) metastatic uveal melanoma", "Cytoreductive surgery + HIPEC (peritoneal mesothelioma BAP1)", "Belzutifan investigational BAP1-null ccRCC (HIF pathway active)", "Chemotherapy (pemetrexed + cisplatin) pleural mesothelioma", "Annual ophthalmology + brain MRI (uveal melanoma surveillance from 11yr)"],
    "MET":  ["Cabozantinib FDA 2016 (MET + VEGFR2 + AXL) — advanced papillary RCC", "Tepotinib FDA 2021 selective MET inhibitor — MET GOF papillary RCC investigational", "Savolitinib selective MET — MRC SAVOIR trial papillary RCC", "Staged partial nephrectomies (bilateral disease — nephron preservation MANDATORY)", "3cm active surveillance rule before surgical intervention", "Nivolumab + cabozantinib IO+TKI combination metastatic"],
    "TSC2": ["Everolimus FDA 2012 (SEGA + AML + LAM — mTOR triple-indication)", "Transcatheter arterial embolisation (TAE) for AML >4cm (preferred over surgery)", "Sirolimus (rapamycin): LAM stabilisation + tuberous sclerosis manifestations", "Vigabatrin: first-line infantile spasms TSC", "Surgery: partial nephrectomy for RCC; resect SEGA if everolimus fails", "Annual renal USS/MRI from age 1yr"],
    "PTEN": ["Everolimus + lenvatinib FDA 2019 advanced RCC (previously treated)", "Nivolumab + ipilimumab IO first-line metastatic ccRCC", "Annual breast MRI + mammography from age 30yr (women — 85% lifetime risk)", "AKT inhibitors (capivasertib, ipatasertib) investigational PTEN-null RCC", "Endometrial biopsy/surveillance from age 35yr; RRSO 40-45yr option", "Annual thyroid USS from 18yr (follicular TC 10% PHTS)"],
}

# Surveillance protocols per gene
SURVEILLANCE_BY_GENE: dict = {
    "VHL":  ["Annual ophthalmology from age 1yr (retinal haemangioblastoma)", "Annual brain + spine MRI from age 11yr (CNS haemangioblastoma)", "Annual abdominal MRI from age 15yr (ccRCC + pNET + cysts)", "Annual plasma metanephrines/urine catecholamines from age 5yr (pheo)", "Annual audiological assessment (ELST)", "3cm surgery rule: operate when any RCC reaches 3cm"],
    "FH":   ["Annual abdominal MRI from age 8yr MANDATORY (aggressive type 2 papillary RCC)", "Annual gynaecological USS + examination (women) from menarche", "Annual skin examination (cutaneous leiomyomas)", "Immediate surgical consultation on any detected RCC (NO watchful waiting)", "Genetic cascade testing first-degree relatives", "Annual clinical review for uterine fibroid symptoms (women)"],
    "FLCN": ["Annual abdominal MRI from age 20yr (chromophobe RCC surveillance)", "Annual skin examination (fibrofolliculomas — diagnostic + cosmetic)", "CT chest at diagnosis then every 3-5yr (pulmonary cysts)", "Advise pneumothorax emergency protocol + avoid scuba diving", "Genetic cascade testing first-degree relatives", "Annual blood pressure (renal surveillance)"],
    "SDHB": ["Annual plasma/urine metanephrines/catecholamines from age 6yr", "Annual whole-body MRI from age 6yr (PPGL + RCC surveillance)", "68Ga-DOTATATE PET-CT for staging SSTR2+ PPGL", "Annual abdominal MRI (SDH-deficient RCC)", "Pre-surgical alpha-blockade before ANY PPGL resection", "Annual clinical examination (hypertension, headache, palpitations)"],
    "BAP1": ["Annual ophthalmological MRI/USS + fundus exam from age 11yr (uveal melanoma)", "Annual abdominal MRI from age 30yr (ccRCC + peritoneal)", "Annual CT chest from age 30yr (pleural mesothelioma)", "Annual skin surveillance (MBAITs + cutaneous melanoma)", "Avoid asbestos/silica — ABSOLUTE contraindication", "HLA-A*02:01 typing at diagnosis (tebentafusp eligibility for uveal melanoma)"],
    "MET":  ["Annual abdominal MRI from age 30yr (bilateral papillary RCC surveillance)", "3cm active surveillance rule — operate at 3cm", "Staged partial nephrectomies bilateral disease", "Annual blood pressure + renal function (post-nephrectomy)", "Genetic cascade testing first-degree relatives", "Annual clinical review at specialist centre"],
    "TSC2": ["Annual brain MRI from diagnosis (SEGA)", "Annual renal USS + MRI from age 1yr (AML surveillance)", "Annual pulmonary function tests + CT chest from age 18yr (LAM women)", "Annual ophthalmology (retinal hamartoma)", "Annual neurodevelopmental assessment in childhood", "Annual skin examination (ash leaf macules, angiofibromas, Shagreen patch)"],
    "PTEN": ["Annual breast MRI + mammography from age 30yr (women — breast 85%)", "Annual thyroid USS from age 18yr (follicular TC)", "Annual endometrial biopsy from age 35yr (women — endometrial 28-44%)", "Annual abdominal USS from age 40yr (RCC surveillance)", "Colonoscopy every 5yr from age 35yr (colorectal 9-16%)", "Annual brain MRI at diagnosis (Lhermitte-Duclos)"],
}

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]


def _make_patient(gene_idx: int, patient_idx: int) -> dict:
    seed = SEED_BASE + gene_idx + patient_idx * len(ATLAS_GENES)
    rng = random.Random(seed)
    gene = _GENE_LIST[gene_idx]
    gene_info = ATLAS_GENES[gene_idx]
    tumour_types = TUMOUR_TYPES_BY_GENE[gene]
    variants = VARIANTS_BY_GENE[gene]
    treatments = TREATMENT_PROTOCOLS_BY_GENE[gene]

    age_at_dx = rng.randint(22, 74)
    tumour_type = rng.choice(tumour_types)
    variant = rng.choice(variants)
    treatment = rng.choice(treatments)
    # Radiation rate: VHL/FH/SDHB/TSC2 rarely radiated (surgery/targeted preferred); MET/FLCN/BAP1/PTEN moderate
    cr = rng.random() < 0.53
    radiation = rng.random() < (0.05 if gene in ("VHL", "FH", "SDHB", "TSC2") else 0.16)
    relapse = rng.random() < 0.38 if cr else rng.random() < 0.61

    return {
        "patient_id": f"HRCC-{gene}-{patient_idx:03d}",
        "gene": gene,
        "syndrome": gene_info["syndrome"],
        "age_at_dx": age_at_dx,
        "tumour_type": tumour_type,
        "variant": variant,
        "treatment": treatment,
        "cr": cr,
        "radiation": radiation,
        "relapse": relapse,
    }


def _generate_cohort() -> list:
    patients = []
    for gi in range(len(ATLAS_GENES)):
        for pi in range(40):
            patients.append(_make_patient(gi, pi))
    return patients


def generate_overview() -> dict:
    cohort = _generate_cohort()
    n = len(cohort)
    cr_n = sum(1 for p in cohort if p["cr"])
    rad_n = sum(1 for p in cohort if p["radiation"])
    relapse_n = sum(1 for p in cohort if p["relapse"])
    mean_age = round(sum(p["age_at_dx"] for p in cohort) / n, 1)

    gene_summary = {}
    for g in _GENE_LIST:
        pts = [p for p in cohort if p["gene"] == g]
        gene_summary[g] = {
            "n": len(pts),
            "cr_pct": round(100 * sum(1 for p in pts if p["cr"]) / len(pts), 1),
            "radiation_pct": round(100 * sum(1 for p in pts if p["radiation"]) / len(pts), 1),
            "relapse_pct": round(100 * sum(1 for p in pts if p["relapse"]) / len(pts), 1),
            "mean_age": round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
        }

    return {
        "atlas": "Hereditary-RCC-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Renal Cell Carcinoma Predisposition Reference — VHL-FH-FLCN-SDHB-BAP1-MET-TSC2-PTEN",
        "seeds": f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
        "total_patients": n,
        "cr_pct": round(100 * cr_n / n, 1),
        "radiation_pct": round(100 * rad_n / n, 1),
        "relapse_pct": round(100 * relapse_n / n, 1),
        "mean_age_at_dx": mean_age,
        "genes": _GENE_LIST,
        "gene_summary": gene_summary,
        "key_rules": [
            "BELZUTIFAN FDA 2021 (VHL) — first HIF-2α inhibitor for VHL-disease RCC/haemangioblastoma/pNET; 3cm surgery rule",
            "NO WATCHFUL WAITING (FH/HLRCC) — type 2 papillary RCC very aggressive; operate immediately at detection regardless of size",
            "ANNUAL MRI FROM AGE 8yr MANDATORY (FH) — HLRCC RCC can present young; no watchful waiting ever",
            "ALPHA-BLOCKADE MANDATORY before PPGL surgery (SDHB) — phenoxybenzamine pre-op; hypertensive crisis risk",
            "177Lu-DOTATATE FDA 2018 (SDHB) — peptide receptor radionuclide therapy SSTR2+ malignant PPGL",
            "AVOID ASBESTOS ABSOLUTELY (BAP1) — BAP1 + asbestos = synergistic mesothelioma; tebentafusp FDA 2022 uveal melanoma",
            "NEPHRON SPARING MANDATORY (MET/HPRC) — bilateral multifocal disease; staged partial nephrectomies; cabozantinib GOF MET",
            "EVEROLIMUS FDA 2012 (TSC2) — mTOR inhibitor triple-indication: SEGA+AML+LAM; embolisation AML >4cm preferred",
            "LHERMITTE-DUCLOS PATHOGNOMONIC (PTEN) — adult cerebellar mass = test PTEN immediately; macrocephaly >97th percentile PATHOGNOMONIC",
        ],
    }


def generate_breakdown() -> dict:
    cohort = _generate_cohort()
    breakdown = {}
    for gi, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        pts = [p for p in cohort if p["gene"] == gene]
        tumour_counts: dict = {}
        for p in pts:
            tumour_counts[p["tumour_type"]] = tumour_counts.get(p["tumour_type"], 0) + 1
        top_tumours = sorted(tumour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        variant_counts: dict = {}
        for p in pts:
            variant_counts[p["variant"]] = variant_counts.get(p["variant"], 0) + 1
        top_variants = sorted(variant_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        breakdown[gene] = {
            "gene": gene,
            "protein": gene_info["protein"],
            "locus": gene_info["locus"],
            "syndrome": gene_info["syndrome"],
            "inheritance": gene_info["inheritance"],
            "rcc_risk": gene_info["rcc_risk"],
            "pathognomonic": gene_info["pathognomonic"],
            "key_avoid": gene_info["key_avoid"],
            "key_rule": gene_info["key_rule"],
            "surveillance": gene_info["surveillance"],
            "targeted_rx": gene_info["targeted_rx"],
            "n_patients": len(pts),
            "cr_pct": round(100 * sum(1 for p in pts if p["cr"]) / len(pts), 1),
            "radiation_pct": round(100 * sum(1 for p in pts if p["radiation"]) / len(pts), 1),
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
        "atlas": "Hereditary-RCC-Predisposition-Atlas",
        "definitions": {
            "vhl_syndrome": (
                "Von Hippel-Lindau Syndrome: VHL germline LOF — HIF-α E3-ligase subunit; "
                "Clear cell RCC 80-85% lifetime — HIGHEST single-gene hereditary RCC; bilateral/multifocal; "
                "BELZUTIFAN FDA 2021: first HIF-2α inhibitor for VHL-disease RCC/haemangioblastoma/pNET; "
                "3cm surgery rule: operate at 3cm to prevent metastasis; "
                "Haemangioblastoma CNS/retinal PATHOGNOMONIC; phaeochromocytoma type 2 only; "
                "Annual ophthalmology from 1yr; annual brain+spine MRI from 11yr; annual abdominal MRI from 15yr."
            ),
            "hlrcc_fh": (
                "HLRCC (Hereditary Leiomyomatosis and RCC): FH germline LOF — fumarate hydratase TCA enzyme; "
                "Type 2 papillary RCC VERY AGGRESSIVE — NO watchful waiting at any size; "
                "2SC IHC PATHOGNOMONIC FH-deficiency (fumarate + cysteine → 2SC accumulates); "
                "Uterine fibroids young-onset large PATHOGNOMONIC; cutaneous leiomyomas PATHOGNOMONIC; "
                "Bevacizumab + erlotinib systemic HLRCC; annual MRI from age 8yr MANDATORY."
            ),
            "bhd_flcn": (
                "Birt-Hogg-Dubé (BHD) Syndrome: FLCN germline LOF — folliculin mTOR-AMPK lysosomal scaffold; "
                "Chromophobe/hybrid oncocytic RCC 25-34% lifetime; bilateral multifocal; "
                "Fibrofolliculomas (hair follicle hamartomas) skin PATHOGNOMONIC; "
                "Bilateral pulmonary cysts + spontaneous pneumothorax 30-50% PATHOGNOMONIC; "
                "Avoid scuba diving (barotrauma); everolimus mTOR or cabozantinib systemic; "
                "Annual abdominal MRI from age 20yr."
            ),
            "sdhb_ppgl_rcc": (
                "SDH-deficient PPGL/RCC (SDHB): SDHB germline LOF — succinate dehydrogenase iron-sulfur subunit; "
                "Malignant phaeochromocytoma/paraganglioma 40% SDHB — HIGHEST SDHx malignancy risk; "
                "SDH-deficient RCC 3-5%; succinate IHC PATHOGNOMONIC; SDHB IHC nuclear loss PATHOGNOMONIC; "
                "ALPHA-BLOCKADE MANDATORY before any PPGL resection; "
                "177Lu-DOTATATE FDA 2018 SSTR2+ malignant PPGL; annual whole-body MRI from 6yr."
            ),
            "bap1_tpds": (
                "BAP1 Tumor Predisposition Syndrome (TPDS): BAP1 germline LOF — BRCA1-associated protein-1 DUB; "
                "Clear cell RCC ~50% lifetime; uveal melanoma HIGHEST germline risk; mesothelioma 30-60%; "
                "MBAITs (melanocytic BAP1-associated intradermal tumours) PATHOGNOMONIC; "
                "BAP1 IHC nuclear loss PATHOGNOMONIC; AVOID ASBESTOS ABSOLUTELY; "
                "Tebentafusp FDA 2022 (HLA-A*02:01+) metastatic uveal melanoma; nivolumab+ipilimumab ccRCC."
            ),
            "hprc_met": (
                "HPRC (Hereditary Papillary RCC type 1): MET germline GOF — HGF receptor kinase; "
                "Type 1 papillary RCC — bilateral multifocal (3-3000 lesions per kidney); late onset (mean 60s); "
                "3cm ACTIVE SURVEILLANCE RULE — monitor until 3cm then operate; "
                "Nephron sparing MANDATORY (bilateral disease); staged partial nephrectomies; "
                "Cabozantinib FDA 2016 (MET+VEGFR2+AXL); tepotinib FDA 2021 selective MET investigational."
            ),
            "tsc_tsc2": (
                "Tuberous Sclerosis Complex type 2 (TSC2): TSC2 germline LOF — tuberin mTOR regulator; "
                "Angiomyolipoma bilateral PATHOGNOMONIC; chromophobe RCC 2-3%; "
                "Cortical tubers PATHOGNOMONIC; SEGA PATHOGNOMONIC; LAM (women); "
                "EVEROLIMUS FDA 2012 — triple indication: SEGA + AML + LAM; "
                "Embolisation preferred over surgery for AML >4cm (nephron preservation); vigabatrin infantile spasms."
            ),
            "cowden_pten": (
                "Cowden Syndrome / PHTS: PTEN germline LOF — PI3K-AKT phosphatase; "
                "RCC 34% lifetime (clear cell; 5-7x elevated); "
                "Macrocephaly >97th centile PATHOGNOMONIC; Lhermitte-Duclos PATHOGNOMONIC; "
                "Breast 85% lifetime (women — highest non-BRCA1 risk); endometrial 28-44%; thyroid follicular 10%; "
                "Everolimus + lenvatinib FDA 2019 (RCC); annual breast MRI from 30yr (women)."
            ),
            "cascade_testing": (
                "CASCADE TESTING — Hereditary RCC Predisposition: "
                "1. VHL: belzutifan FDA 2021; 3cm surgery rule; annual ophthalmology 1yr; brain+spine MRI 11yr; abdominal MRI 15yr; "
                "2. FH: NO watchful waiting; annual MRI from 8yr; bevacizumab+erlotinib systemic; 2SC IHC PATHOGNOMONIC; "
                "3. FLCN: annual MRI from 20yr; fibrofolliculomas PATHOGNOMONIC; pneumothorax 30-50%; avoid scuba; "
                "4. SDHB: alpha-block pre-op MANDATORY; 177Lu-DOTATATE SSTR2+; annual MRI+biochemistry from 6yr; "
                "5. BAP1: avoid asbestos ABSOLUTELY; tebentafusp uveal melanoma; MBAITs PATHOGNOMONIC; annual MRI 30yr; "
                "6. MET: 3cm surveillance rule; staged partial nephrectomies; nephron sparing; cabozantinib; "
                "7. TSC2: everolimus triple-indication; embolisation AML >4cm; annual renal MRI from 1yr; "
                "8. PTEN: Lhermitte-Duclos PATHOGNOMONIC; annual breast MRI from 30yr; macrocephaly PATHOGNOMONIC."
            ),
        },
        "key_clinical_rules": [
            {
                "rule": "BELZUTIFAN FDA 2021 — HIF-2α inhibitor VHL-disease (RCC/haemangioblastoma/pNET)",
                "gene": "VHL",
                "rationale": "VHL LOF → HIF-2α constitutive activation → VEGF/EPO/CAIX transcription; belzutifan directly inhibits HIF-2α-EPAS1 (first-in-class); FDA 2021 approval for VHL-disease-associated RCC, haemangioblastomas, and pNETs; avoids nephrectomy + preserves renal function",
                "consequence": "Not using belzutifan in VHL-disease RCC → unnecessary nephrectomy or radical surgery; missed triple-indication drug covering all three major VHL manifestations simultaneously",
            },
            {
                "rule": "NO WATCHFUL WAITING — FH/HLRCC type 2 papillary RCC",
                "gene": "FH",
                "rationale": "HLRCC type 2 papillary RCC is one of the most aggressive hereditary RCCs; metastases documented from primaries <1cm; standard VHL/MET 3cm watchful waiting rule does NOT apply; aggressive surgery is only curative modality; delay beyond detection is life-threatening",
                "consequence": "Watchful waiting in HLRCC → high likelihood of metastatic disease at second scan; preventable mortality; 5yr OS <50% once metastatic",
            },
            {
                "rule": "ALPHA-BLOCKADE MANDATORY before PPGL resection (SDHB)",
                "gene": "SDHB",
                "rationale": "Phaeochromocytoma/paraganglioma secrete catecholamines; surgical manipulation → catecholamine storm → hypertensive crisis → stroke/MI/death; phenoxybenzamine (irreversible alpha-blocker) pre-operative coverage for 1-2 weeks MANDATORY; beta-blockade added after adequate alpha-blockade (never first)",
                "consequence": "Surgery without pre-op alpha-blockade → intraoperative catecholamine crisis → cardiovascular emergency; preventable mortality documented in multiple case series",
            },
            {
                "rule": "AVOID ASBESTOS ABSOLUTELY — BAP1 + asbestos synergistic mesothelioma",
                "gene": "BAP1",
                "rationale": "BAP1 germline LOF carriers exposed to asbestos have dramatically elevated mesothelioma risk vs asbestos alone or BAP1 alone; synergistic interaction between genetic predisposition and occupational/environmental exposure; lifetime BAP1 mesothelioma risk 30-60% is further amplified by asbestos",
                "consequence": "Asbestos exposure in BAP1 carrier → mesothelioma development near-certain with sufficient exposure; preventable by absolute exposure avoidance and occupational protection",
            },
            {
                "rule": "NEPHRON SPARING MANDATORY — MET/HPRC bilateral multifocal disease",
                "gene": "MET",
                "rationale": "HPRC features 3-3000 bilateral papillary RCC lesions per kidney; radical nephrectomy of first kidney → immediate dependence on contralateral kidney which also has disease; renal function must be preserved; staged partial nephrectomies with 3cm surveillance rule maximises functional kidney tissue while preventing metastasis",
                "consequence": "Radical nephrectomy in bilateral HPRC → end-stage renal disease requiring dialysis while second kidney continues to develop RCC; preventable by staged partial nephrectomies",
            },
            {
                "rule": "LHERMITTE-DUCLOS PATHOGNOMONIC — adult cerebellar mass = test PTEN immediately",
                "gene": "PTEN",
                "rationale": "Lhermitte-Duclos disease (dysplastic cerebellar gangliocytoma) is the strongest pathognomonic sign for Cowden Syndrome/PHTS; any adult with cerebellar gangliocytoma must be tested for PTEN germline LOF; unrecognised PTEN carrier misses breast/endometrial/thyroid/RCC surveillance; macrocephaly >97th centile is the second pathognomonic sign",
                "consequence": "Unrecognised PTEN → missed 85% breast cancer + 34% RCC + 28-44% endometrial surveillance; preventable cancer mortality in high-penetrance syndrome",
            },
            {
                "rule": "177Lu-DOTATATE FDA 2018 — SSTR2+ malignant PPGL (SDHB)",
                "gene": "SDHB",
                "rationale": "Most SDHx PPGL/PGL are SSTR2-positive; 177Lu-DOTATATE (Lutathera) delivers targeted beta-irradiation to SSTR2-expressing tumours; FDA 2018 approval for gastroenteropancreatic NETs extended to SSTR2+ PPGL in practice; most effective systemic option for malignant paraganglioma vs sunitinib alone",
                "consequence": "Not testing SSTR2 (68Ga-DOTATATE PET) in SDHB malignant PPGL → missed 177Lu-DOTATATE eligibility; sunitinib alone inferior to PRRT in SSTR2+ disease",
            },
            {
                "rule": "EVEROLIMUS FDA 2012 — TSC2 triple-indication mTOR inhibitor",
                "gene": "TSC2",
                "rationale": "TSC2 LOF → mTORC1 constitutive activation; everolimus (mTOR inhibitor) has three FDA-approved TSC indications in one drug: (1) SEGA (2012), (2) renal AML (2012), (3) TSC-LAM (2016); embolisation preferred over surgery for AML >4cm to preserve nephrons while everolimus is added",
                "consequence": "Not using everolimus in TSC SEGA or AML → progressive SEGA causing obstructive hydrocephalus; AML haemorrhage (Wunderlich syndrome) if not embolised at >4cm; preventable renal morbidity",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(f"Total patients: {ov['total_patients']}")
    print(f"Genes: {ov['genes']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"CR%: {ov['cr_pct']}")
    print(f"Mean age dx: {ov['mean_age_at_dx']}")
    print("\n=== BREAKDOWN KEYS ===")
    bd = generate_breakdown()
    for g in bd["genes"]:
        print(f"  {g}: n={bd['breakdown'][g]['n_patients']}, CR={bd['breakdown'][g]['cr_pct']}%")
    print("\n=== DEFINITIONS ===")
    df = generate_definitions()
    for k in list(df["definitions"].keys())[:3]:
        print(f"  {k}: {df['definitions'][k][:60]}...")
