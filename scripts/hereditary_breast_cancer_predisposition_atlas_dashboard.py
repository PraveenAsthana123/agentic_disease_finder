#!/usr/bin/env python3
"""Hereditary-Breast-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
BRCA1   (BRCA1 DNA repair associated; 1863aa; 17q21.31; AD LOF;
         Hereditary Breast and Ovarian Cancer Syndrome type 1 (HBOC1);
         breast 46-87% lifetime; TNBC-enriched 70-80%; RRSO 35-40yr;
         olaparib OlympiAD+OLYMPIA FDA; bilateral mastectomy 90-95% risk reduction;
         seed SEED_BASE+0) .
BRCA2   (BRCA2 DNA repair associated; 3418aa; 13q12.3; AD LOF;
         HBOC2 / Fanconi Anaemia type D1 (biallelic);
         breast 38-65% lifetime; heterogeneous histology; RRSO 40-45yr;
         male breast 6% HIGHEST hereditary male breast; olaparib; platinum;
         seed SEED_BASE+1) .
PALB2   (Partner and localiser of BRCA2; 1186aa; 16p12.2; AD LOF;
         HBOC3 / Fanconi Anaemia type N (biallelic);
         breast 53% lifetime -- HIGHEST non-BRCA2 PARPi-sensitive gene;
         TBCRC048 olaparib 82% ORR; BRCA1-BRCA2 bridge scaffold;
         seed SEED_BASE+2) .
CHEK2   (Checkpoint kinase 2; 543aa; 22q12.1; AD LOF;
         Hereditary Breast Cancer -- moderate risk;
         1100delC Northern/Western European 20-25% lifetime breast;
         I157T Central/Eastern European 18-20% lifetime; NO PARPi standard;
         seed SEED_BASE+3) .
ATM     (Ataxia-telangiectasia mutated; 3056aa; 11q22.3; AD/AR LOF;
         Ataxia-Telangiectasia (biallelic) / Moderate HBOC (monoallelic);
         monoallelic 30-35% moderate breast; telangiectasias PATHOGNOMONIC biallelic;
         RADIOSENSITIVITY-ABSOLUTE biallelic; Ceralasertib ATRi synthetic lethality;
         seed SEED_BASE+4) .
CDH1    (E-cadherin; 882aa; 16q22.1; AD LOF;
         Hereditary Diffuse Gastric Cancer (HDGC);
         lobular breast cancer 42% lifetime PATHOGNOMONIC;
         TOTAL GASTRECTOMY MANDATORY 20-30yr; mammogram unreliable for lobular ILC;
         seed SEED_BASE+5) .
STK11   (Serine/threonine kinase 11 / LKB1; 433aa; 19p13.3; AD LOF;
         Peutz-Jeghers Syndrome (PJS);
         breast 50% lifetime -- HIGHEST penetrance non-BRCA single-gene breast;
         mucocutaneous macules PATHOGNOMONIC; SCTAT ovarian PATHOGNOMONIC;
         seed SEED_BASE+6) .
NF1     (Neurofibromin 1; 2839aa; 17q11.2; AD LOF;
         Neurofibromatosis type 1;
         breast 50% elevated risk early onset <40yr;
         cafe-au-lait macules ≥6 PATHOGNOMONIC; selumetinib FDA2020;
         annual MRI breast from 30yr mandatory;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3366-3373)
"""
import random

SEED_BASE = 3366

ATLAS_GENES = [
    {
        "gene": "BRCA1",
        "protein": (
            "BRCA1 -- 17q21.31 Autosomal-Dominant-LOF -- 1863aa -- "
            "RING-BRCT-HR-Repair-Scaffold-207kDa-"
            "Breast-46-87pct-HIGHEST-HBOC1-TNBC-Enriched-70-80pct-"
            "RRSO-35-40yr-Olaparib-OlympiAD-OLYMPIA-FDA-"
            "Bilateral-Mastectomy-90-95pct-Risk-Reduction-OMIM-113705"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "1863 aa / 207 kDa / 17q21.31 BRCA1 encodes a nuclear phosphoprotein: "
            "STRUCTURE: "
            "  1863 aa / 207 kDa; N-terminal RING domain (aa 1-109): E3 ubiquitin ligase with BARD1 heterodimer; "
            "  Central region (aa 200-1649): BRCA1 coiled-coil; interaction with RAD51, CtIP, PALB2; "
            "  C-terminal BRCT repeat domain (aa 1646-1863): phosphopeptide binding; ATM/CHEK2 S1387/S1524 phosphorylation; "
            "  Nuclear localisation signals: NLS1 (503-508) and NLS2 (606-615); "
            "FUNCTION: "
            "  Homologous recombination (HR) repair master scaffold: senses DSBs, recruits RAD51 via PALB2-BRCA2; "
            "  BRCA1-BARD1: H2A-K119 ubiquitination at DSB flanks; "
            "  BRCA1-BACH1/BRIP1: 5' end resection via CtIP; "
            "  BRCA1-CtIP-MRN: end resection initiation; "
            "  BRCA1-ABRAXAS-RAP80: checkpoint maintenance; "
            "  LOF → defective HR → error-prone NHEJ → genomic instability → BRCA1-mutated tumour signature (SBS3); "
            "CANCER RISKS: "
            "  BREAST: 46-87% lifetime (range reflects missense vs truncating, founder mutations, family history); "
            "  TNBC ENRICHED: 70-80% of BRCA1 breast tumours are TNBC (ER-/PR-/HER2-); "
            "  OVARIAN: 39-44% lifetime (HGSOC predominant); "
            "  FALLOPIAN TUBE: included in ovarian risk estimate; "
            "  PANCREATIC: 2-3x elevated; PROSTATE: 2-3x elevated (weaker than BRCA2); "
            "KEY MANAGEMENT: "
            "  RRSO 35-40yr post-childbearing: reduces ovarian risk 80-96% AND breast risk ~50% premenopausal; "
            "  BILATERAL MASTECTOMY: reduces breast risk 90-95%; combined with RRSO reduces overall BRCA1 cancer mortality; "
            "  MRI BREAST from 25yr annually MANDATORY; alternating MRI/mammography 6-monthly; "
            "  OLAPARIB: FDA-approved metastatic/adjuvant BRCA1/2 breast (OlympiAD 2017, OlympiA/OLYMPIA 2021); "
            "  PLATINUM SENSITIVITY: cisplatin/carboplatin — BRCA1 HR-deficient tumours respond; "
            "  TNBC NEOADJUVANT: platinum containing regimen preferred; pCR strongly predicts survival; "
            "  CONTRALATERAL RISK: 40-65% 20-year risk if unaffected breast retained — contralateral prophylactic mastectomy discussion mandatory"
        ),
        "syndrome": "HBOC1 (Hereditary Breast and Ovarian Cancer Syndrome type 1)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "breast_risk": "46-87% lifetime",
        "pathognomonic": "TNBC histology in BRCA1-carrier — not strictly pathognomonic but strongly predictive; HRD scar (SBS3) on tumour sequencing",
        "key_avoid": "Do NOT omit PARPi if HRD/BRCA1 confirmed — do NOT substitute platinum alone as equivalent to olaparib in metastatic setting",
        "key_rule": "RRSO 35-40yr MANDATORY post-childbearing — reduces ovarian risk 80-96%; bilateral mastectomy reduces breast risk 90-95%",
        "surveillance": "Annual MRI breast from 25yr; alternating MRI + mammography; monthly self-exam; RRSO by 40yr",
        "targeted_rx": "Olaparib (OlympiAD/OlympiA); Platinum-containing chemotherapy; Pembrolizumab (TNBC neoadjuvant — KEYNOTE-522)",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "RAD51-Loader-BRC-Repeats-384kDa-"
            "Breast-38-65pct-Male-Breast-6pct-HIGHEST-"
            "RRSO-40-45yr-Olaparib-PROfound-FDA2020-"
            "Fanconi-FANCD1-Biallelic-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 is the largest HR repair protein: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; 8 BRC repeats (aa 1002-2085): each repeat binds RAD51 directly; "
            "  N-terminal PALB2 binding domain (aa 10-40): WD40 interaction with PALB2; "
            "  OB folds in C-terminus (aa 2402-3190): ssDNA and RPA binding; "
            "  Nuclear export/import signals; FANCD1 in Fanconi Anaemia ICL pathway; "
            "FUNCTION: "
            "  RAD51 mediator: recruits RAD51 to ssDNA at DSB via BRC repeats; "
            "  PALB2-BRCA1-BRCA2 ternary complex: BRCA1 brings scaffold → PALB2 bridges → BRCA2 loads RAD51; "
            "  FANCD1 (biallelic): integral to Fanconi ICL repair; "
            "  LOF → HR deficiency → cisplatin/olaparib sensitivity (SBS3); "
            "CANCER RISKS: "
            "  BREAST (female): 38-65% lifetime; heterogeneous histology — ER+/HER2- most common; "
            "  BREAST (male): 6% lifetime — HIGHEST hereditary male breast; annual mammography from 40yr; "
            "  OVARIAN: 17% lifetime (lower than BRCA1 but significant); RRSO 40-45yr; "
            "  PANCREATIC: 5-7% (3-4x elevated) — olaparib maintenance POLO trial; "
            "  PROSTATE: 15-20% lifetime — most lethal; olaparib PROfound FDA2020; PSMA-PET mandatory for staging; "
            "  MELANOMA: 2-3x elevated; "
            "KEY MANAGEMENT: "
            "  RRSO 40-45yr (5yr later than BRCA1 — lower/later ovarian risk); "
            "  MALE BRCA2: annual mammography from 40yr; annual prostate PSA from 40yr; "
            "  OLAPARIB: metastatic breast (OlympiAD) + adjuvant (OlympiA) + prostate (PROfound) + pancreatic (POLO); "
            "  CONTRALATERAL RISK: 30-55% 20-year; prophylactic contralateral mastectomy discussion mandatory"
        ),
        "syndrome": "HBOC2 (Hereditary Breast and Ovarian Cancer Syndrome type 2) / Fanconi Anaemia-D1 (biallelic)",
        "inheritance": "AD LOF (autosomal dominant LOF); AR LOF biallelic = Fanconi D1",
        "breast_risk": "38-65% lifetime female; 6% lifetime male",
        "pathognomonic": "Male breast cancer with BRCA2 carrier status — not strictly pathognomonic but strongly predictive",
        "key_avoid": "Do NOT delay RRSO to 50yr in BRCA2 — ovarian risk starts materialising late 30s; do NOT omit prostate PSA in male carriers",
        "key_rule": "MALE BREAST + BRCA2: annual mammography from 40yr; MALE PROSTATE: PSA annual from 40yr; RRSO 40-45yr",
        "surveillance": "Annual MRI breast from 25yr; RRSO 40-45yr; male: mammography + PSA from 40yr; skin surveillance for melanoma",
        "targeted_rx": "Olaparib (OlympiAD/OlympiA breast; PROfound prostate; POLO pancreatic); Platinum; Bevacizumab",
    },
    {
        "gene": "PALB2",
        "protein": (
            "PALB2 -- 16p12.2 Autosomal-Dominant-LOF -- 1186aa -- "
            "WD40-BRCA1-BRCA2-Bridge-131kDa-"
            "Breast-53pct-HIGHEST-Non-BRCA2-PARPi-"
            "TBCRC048-Olaparib-82pct-ORR-"
            "Fanconi-FANCN-Biallelic-OMIM-610355"
        ),
        "locus": "16p12.2",
        "protein_size": (
            "1186 aa / 131 kDa / 16p12.2 PALB2 bridges BRCA1 and BRCA2: "
            "STRUCTURE: "
            "  1186 aa / 131 kDa; N-terminal coiled-coil (aa 1-200): BRCA1 binding; "
            "  Central region (aa 200-900): self-association, RAD51AP1, chromatin interaction; "
            "  C-terminal WD40 domain (aa 853-1186): BRCA2 binding (N-terminus aa 10-40); "
            "  FANCN: integral to ICL repair pathway as FANCD2-FANCI substrate; "
            "FUNCTION: "
            "  BRCA1-PALB2-BRCA2 ternary complex: PALB2 is the essential adapter; "
            "  LOF disrupts BRCA2 nuclear localisation at DSB foci; "
            "  RAD51 filament formation requires intact PALB2-BRCA2 interaction; "
            "  HR-deficient tumours: high HRD score, SBS3, cisplatin/olaparib sensitivity; "
            "CANCER RISKS: "
            "  BREAST (female): 53% lifetime — HIGHEST non-BRCA2 gene; "
            "  PANCREATIC: 3-4x elevated; "
            "  OVARIAN: ~3-5% (lower than BRCA1/2); RRSO timing controversial (NICE 2023); "
            "  MALE BREAST: emerging evidence 2-3x elevated; "
            "KEY MANAGEMENT: "
            "  ANNUAL MRI from 30yr — risk equivalent to BRCA2 carriers; "
            "  OLAPARIB TBCRC048: 82% ORR in PALB2-mutated metastatic breast — HIGHEST response of any non-BRCA gene; "
            "  BILATERAL MASTECTOMY: discussed given 53% risk; "
            "  RRSO TIMING CONTROVERSIAL: some guidelines recommend 45-50yr; others defer; "
            "  BIALLELIC FA-N: Fanconi Anaemia — bone marrow failure, congenital abnormalities, early-onset cancer; "
            "  AVOID ALKYLATING AGENTS in biallelic — cyclophosphamide absolutely CI in FA-N"
        ),
        "syndrome": "Moderate HBOC / Fanconi Anaemia-N (FANCN, biallelic)",
        "inheritance": "AD LOF (autosomal dominant); AR biallelic = FA-N",
        "breast_risk": "53% lifetime — HIGHEST non-BRCA PARPi-sensitive",
        "pathognomonic": "Biallelic PALB2 = Fanconi phenotype (FANCN); monoallelic: no pathognomonic feature",
        "key_avoid": "Do NOT equate PALB2 to ATM/CHEK2 (lower risk genes) — PALB2 breast risk is BRCA2-equivalent; do NOT defer MRI beyond 30yr",
        "key_rule": "Annual MRI breast from 30yr MANDATORY; TBCRC048 olaparib 82% ORR HIGHEST non-BRCA response",
        "surveillance": "Annual MRI breast from 30yr; alternating MRI+mammography; pancreatic MRI/EUS consider from 50yr",
        "targeted_rx": "Olaparib (TBCRC048 82% ORR); Platinum-containing chemotherapy; Niraparib",
    },
    {
        "gene": "CHEK2",
        "protein": (
            "CHEK2 -- 22q12.1 Autosomal-Dominant-LOF -- 543aa -- "
            "CHK2-FHA-Kinase-62kDa-"
            "1100delC-Northern-European-20-25pct-Breast-"
            "I157T-Central-Eastern-European-18-20pct-"
            "Moderate-Risk-NO-PARPi-Standard-OMIM-604373"
        ),
        "locus": "22q12.1",
        "protein_size": (
            "543 aa / 62 kDa / 22q12.1 CHEK2 is the ATM-activated checkpoint kinase: "
            "STRUCTURE: "
            "  543 aa / 62 kDa; SQ/TQ cluster (aa 1-100): ATM substrate sites T68; "
            "  FHA domain (aa 115-175): phosphopeptide binding (pT68 dimerisation); "
            "  Kinase domain (aa 210-486): serine/threonine kinase; "
            "  Two pathogenic variants predominate: 1100delC (frameshift, Northern/Western Europe) and I157T (missense, Central/Eastern Europe); "
            "FUNCTION: "
            "  ATM activates CHEK2 (pT68 → dimerisation → auto-phosphorylation); "
            "  CHEK2 phosphorylates BRCA1 S988, CDC25A, CDC25C, TP53 S20 — cell cycle arrest; "
            "  LOF: impaired G1/S and G2/M checkpoint arrest after DSB; "
            "  Missense I157T: partial LOF, lower penetrance than 1100delC; "
            "CANCER RISKS: "
            "  1100delC: 20-25% lifetime breast female (allelic dosage effect); "
            "  I157T: 18-20% lifetime breast female (weaker kinase impairment); "
            "  BOTH: moderate risk (neither achieves BRCA1/2 level risk); "
            "  CONTRALATERAL: 2-3x elevated after ipsilateral breast cancer; "
            "  COLON: 2-3x elevated; PROSTATE: 2-3x elevated; THYROID: modest elevation; "
            "KEY MANAGEMENT: "
            "  Risk-stratified: ANNUAL MRI from 30-40yr if additional risk factors (dense breast, family history); "
            "  NOT PARPi standard: CHEK2 breast tumours are NOT necessarily HR-deficient; platinum/PARPi not indicated without somatic HRD confirmation; "
            "  NO prophylactic surgery as default — moderate risk managed with surveillance not prophylaxis; "
            "  CONTRALATERAL SURVEILLANCE: high vigilance after ipsilateral breast cancer; "
            "  1100delC vs I157T: DO NOT CONFLATE penetrance — must specify variant to counsel correctly"
        ),
        "syndrome": "Moderate Hereditary Breast Cancer Susceptibility (non-HBOC)",
        "inheritance": "AD LOF (autosomal dominant; incomplete penetrance)",
        "breast_risk": "1100delC 20-25%; I157T 18-20% lifetime",
        "pathognomonic": "No pathognomonic clinical feature — molecular diagnosis only",
        "key_avoid": "Do NOT treat as BRCA equivalent — moderate risk; do NOT offer PARPi without somatic HRD evidence; do NOT conflate I157T and 1100delC",
        "key_rule": "NO PARPi standard — CHEK2 breast tumours not reliably HR-deficient; risk-stratified surveillance not universal prophylaxis",
        "surveillance": "Annual MRI from 30-40yr (risk-stratified per national guidelines); mammography; contralateral vigilance",
        "targeted_rx": "No approved targeted therapy for CHEK2 germline; somatic HRD confirmation before PARPi",
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Autosomal-Dominant-LOF (monoallelic) / AR-LOF (biallelic) -- 3056aa -- "
            "PI3K-PIKK-Master-DSB-Kinase-350kDa-"
            "Monoallelic-30-35pct-Moderate-Breast-"
            "Biallelic-Ataxia-Telangiectasia-RADIOSENSITIVITY-ABSOLUTE-"
            "Telangiectasias-PATHOGNOMONIC-Ceralasertib-OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM is the master DSB sensor kinase: "
            "STRUCTURE: "
            "  3056 aa / 350 kDa; PI3K-like kinase family (PIKK); "
            "  FAT domain (aa 1960-2566): regulatory; KINASE domain (aa 2712-2966): SQ/TQ substrate selectivity; "
            "  FATC domain (aa 2967-3056): essential for kinase activity; "
            "  MRN complex (MRE11-RAD50-NBN) activates ATM at DSBs (H2AX phosphorylation γH2AX); "
            "FUNCTION: "
            "  Phosphorylates >900 substrates on SQ/TQ: H2AX S139, BRCA1 S1387/S1524, CHEK2 T68, TP53 S15; "
            "  Master DDR orchestrator: G1/S, intra-S, G2/M checkpoints; "
            "  Homologous recombination initiation; apoptosis via TP53; "
            "  Biallelic LOF: A-T — cerebellar ataxia, oculocutaneous telangiectasias PATHOGNOMONIC, immunodeficiency, "
            "  elevated AFP (diagnostic), radiosensitivity ABSOLUTE — radiation causes severe tissue reactions; "
            "CANCER RISKS: "
            "  MONOALLELIC female breast: 30-35% lifetime moderate risk (intermediate between CHEK2 and BRCA2); "
            "  MONOALLELIC male breast: 2-4x elevated; "
            "  MONOALLELIC pancreatic: 5-8x elevated; lymphoma/leukaemia (biallelic); "
            "  BIALLELIC A-T: lymphoma T-cell 70-100x; leukaemia 30-50x; breast 50% by 50yr; "
            "KEY MANAGEMENT: "
            "  ANNUAL MRI from 30-40yr (similar to CHEK2 moderate risk); "
            "  RADIOSENSITIVITY WARNING: BIALLELIC patients ABSOLUTELY CI for RT — severe complications; "
            "  MONOALLELIC carriers: discuss RT risk (mild intermediate radiosensitivity — not absolute CI but disclose); "
            "  CERALASERTIB (AZD6738, ATRi): synthetic lethality with ATM LOF — clinical trials; "
            "  OLAPARIB: modest benefit in ATM-mutated breast (PROfound prostate HR 0.72; breast emerging data); "
            "  ELEVATED AFP: serum AFP elevated in biallelic A-T — diagnostic marker (not tumour); "
            "  LIVE VACCINES: biallelic A-T may have impaired immune response — discuss with immunologist"
        ),
        "syndrome": "Moderate HBOC (monoallelic) / Ataxia-Telangiectasia (biallelic, AR)",
        "inheritance": "AD LOF monoallelic (moderate risk); AR biallelic = Ataxia-Telangiectasia (A-T)",
        "breast_risk": "30-35% lifetime monoallelic; ~50% by 50yr biallelic A-T",
        "pathognomonic": "TELANGIECTASIAS (oculocutaneous, bulbar conjunctiva) PATHOGNOMONIC for biallelic A-T; elevated AFP diagnostic",
        "key_avoid": "RADIATION ABSOLUTELY CI in biallelic A-T — severe complications; monoallelic: intermediate — disclose risk; do NOT equate to BRCA2",
        "key_rule": "RADIOSENSITIVITY-ABSOLUTE biallelic: RT contraindicated; AFP elevated = diagnostic marker not tumour; Ceralasertib ATRi synthetic lethality",
        "surveillance": "Annual MRI breast from 30-40yr; biallelic A-T: annual neurological review, immunology, AFP, CBC",
        "targeted_rx": "Ceralasertib (ATRi, clinical trial); Olaparib (emerging data); Platinum",
    },
    {
        "gene": "CDH1",
        "protein": (
            "CDH1 -- 16q22.1 Autosomal-Dominant-LOF -- 882aa -- "
            "E-Cadherin-97kDa-Epithelial-Adhesion-"
            "HDGC-Diffuse-Gastric-67-83pct-Lobular-Breast-42pct-"
            "Total-Gastrectomy-MANDATORY-20-30yr-"
            "Mammogram-Unreliable-Lobular-ILC-OMIM-192090"
        ),
        "locus": "16q22.1",
        "protein_size": (
            "882 aa / 97 kDa / 16q22.1 CDH1 encodes E-cadherin, the epithelial calcium-dependent cell adhesion molecule: "
            "STRUCTURE: "
            "  882 aa / 97 kDa; signal peptide + prodomain (aa 1-154); "
            "  5 extracellular cadherin repeat domains (EC1-EC5, aa 155-701): calcium-binding + homophilic binding; "
            "  Transmembrane domain (aa 702-723); "
            "  Intracellular tail (aa 724-882): β-catenin binding domain (aa 838-882); direct β-catenin stabilisation at AJs; "
            "  LOF → loss of epithelial polarisation → diffuse (signet ring cell) invasion pattern; "
            "FUNCTION: "
            "  Calcium-dependent homophilic adhesion at adherens junctions (AJ) between epithelial cells; "
            "  β-catenin sequestration: cytoplasmic CDH1 retains β-catenin at membrane; "
            "  CDH1 LOF → β-catenin released → WNT-pathway activation; "
            "  E-cadherin loss is the hallmark of epithelial-to-mesenchymal transition (EMT); "
            "  HDGC mechanism: germline CDH1 LOF + somatic second hit → diffuse gastric signet ring cell carcinoma; "
            "CANCER RISKS: "
            "  DIFFUSE GASTRIC CANCER (DGC): 67-83% lifetime (HIGHEST gastric cancer risk of any gene); "
            "  LOBULAR BREAST CANCER (ILC): 42% lifetime female — PATHOGNOMONIC association (CDH1 LOF ↔ ILC); "
            "  Lobular carcinoma in situ (LCIS): frequent precursor in CDH1 carriers; "
            "  MALE breast: 2-3x modest elevation; "
            "KEY MANAGEMENT: "
            "  TOTAL GASTRECTOMY MANDATORY 20-30yr: only curative intervention; surveillance gastroscopy UNRELIABLE for DGC; "
            "  ANNUAL MRI BREAST from 30yr: mammography unreliable for ILC (lobular pattern ill-defined on mammogram); "
            "  ILC diagnosis: MRI superior to mammography for lobular detection; "
            "  GASTROSCOPY (Cambridge Protocol): annual + multiple biopsies if gastrectomy deferred — NOT a substitute; "
            "  PROPHYLACTIC MASTECTOMY: discussed given 42% ILC risk; bilateral risk-reducing mastectomy an option; "
            "  UTERINE FIBROIDS (FH overlap): CDH1 is NOT associated with uterine fibroids — CDH1 = lobular breast only"
        ),
        "syndrome": "Hereditary Diffuse Gastric Cancer (HDGC)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "breast_risk": "42% lifetime lobular ILC — PATHOGNOMONIC CDH1 association",
        "pathognomonic": "Lobular breast cancer (ILC) in CDH1 carrier context is PATHOGNOMONIC — ILC is the CDH1-associated breast phenotype",
        "key_avoid": "MAMMOGRAM UNRELIABLE for lobular ILC — always add MRI; do NOT defer total gastrectomy beyond 30yr; gastroscopy is NOT a substitute for gastrectomy",
        "key_rule": "TOTAL GASTRECTOMY MANDATORY 20-30yr (primary HDGC indication); annual breast MRI from 30yr; mammogram unreliable for ILC",
        "surveillance": "Annual MRI breast from 30yr; annual gastroscopy (Cambridge Protocol) if gastrectomy deferred; total gastrectomy by 30yr",
        "targeted_rx": "No germline-specific targeted therapy; gastric: capecitabine ± trastuzumab (HER2+); breast ILC: endocrine therapy (ER+)",
    },
    {
        "gene": "STK11",
        "protein": (
            "STK11 -- 19p13.3 Autosomal-Dominant-LOF -- 433aa -- "
            "LKB1-AMPK-Kinase-48kDa-"
            "PJS-Breast-50pct-HIGHEST-Non-BRCA-Single-Gene-"
            "Mucocutaneous-Macules-PATHOGNOMONIC-"
            "SCTAT-Ovarian-PATHOGNOMONIC-GI-Endoscopy-8yr-OMIM-602216"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "433 aa / 48 kDa / 19p13.3 STK11 (LKB1) is the master AMPK kinase: "
            "STRUCTURE: "
            "  433 aa / 48 kDa; N-terminal regulatory domain (aa 1-44); "
            "  Central kinase domain (aa 49-309): serine/threonine kinase; "
            "  C-terminal tail (aa 310-433): membrane association, STRAD/MO25 complex formation; "
            "  LKB1-STRAD-MO25 heterotrimeric complex: pseudokinase STRAD activates LKB1; "
            "FUNCTION: "
            "  AMPK activation: LKB1 phosphorylates AMPKα T172 — master energy sensor; "
            "  mTOR suppression via AMPK-TSC1/2 axis; "
            "  LOF → mTORC1 constitutively active → cell growth and proliferation; "
            "  Polarity: LKB1-AMPK establishes epithelial cell polarity — loss → invasive phenotype; "
            "  PJS mechanism: STK11 germline LOF + somatic second hit → hamartomatous polyposis; "
            "CANCER RISKS: "
            "  BREAST: 50% lifetime — HIGHEST breast penetrance of any non-BRCA1/2 single gene; "
            "  PANCREATIC: 11-36% lifetime — HIGHEST and EARLIEST (from age 30yr); "
            "  GASTROINTESTINAL: colorectal 39%, gastric 29%, small bowel 13%; "
            "  OVARIAN: 21% lifetime (SCTAT = sex cord tumour with annular tubules PATHOGNOMONIC); "
            "  CERVICAL: 10% (adenoma malignum cervix PATHOGNOMONIC); "
            "  LUNG: 7-17% adenocarcinoma; "
            "  TESTICULAR: LCCSCT (large cell calcifying Sertoli cell tumour) PATHOGNOMONIC in males; "
            "KEY MANAGEMENT: "
            "  ANNUAL MRI breast from 25yr MANDATORY — highest non-BRCA breast risk gene; "
            "  GI ENDOSCOPY from 8yr: upper + lower GI every 2-3yr; small bowel capsule endoscopy; "
            "  ANNUAL MRI/EUS PANCREATIC from 30yr — earliest onset pancreatic cancer surveillance; "
            "  MUCOCUTANEOUS MACULES: perioral, buccal mucosa, digits — present in childhood PATHOGNOMONIC; "
            "  SCTAT diagnosis: STK11 carrier + ovarian mass + sex cord histology = PATHOGNOMONIC; "
            "  GYNAECOLOGICAL: annual review from 18yr; cervical smear + endocervical sampling"
        ),
        "syndrome": "Peutz-Jeghers Syndrome (PJS)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "breast_risk": "50% lifetime — HIGHEST non-BRCA1/2 single-gene breast cancer risk",
        "pathognomonic": "MUCOCUTANEOUS MACULES (perioral, buccal, digital, nail bed) PATHOGNOMONIC for PJS; SCTAT (ovarian) PATHOGNOMONIC; LCCSCT (testicular) PATHOGNOMONIC males",
        "key_avoid": "Do NOT miss PJS diagnosis if perioral macules present — universal cancer risk management; do NOT defer GI surveillance to adulthood — from age 8yr",
        "key_rule": "GI ENDOSCOPY from 8yr MANDATORY; annual MRI breast from 25yr; pancreatic MRI/EUS from 30yr EARLIEST onset; SCTAT ovarian = PATHOGNOMONIC",
        "surveillance": "Annual MRI breast from 25yr; GI endoscopy from 8yr every 2-3yr; pancreatic MRI/EUS from 30yr; annual gynaecological review",
        "targeted_rx": "No approved STK11-specific targeted therapy; mTOR inhibitors (everolimus) investigational; immunotherapy — STK11 LOF predicts immunotherapy RESISTANCE in NSCLC",
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RAS-GAP-"
            "Breast-50pct-Elevated-Early-Onset-LT-40yr-"
            "Cafe-au-Lait-6-Macules-PATHOGNOMONIC-"
            "Selumetinib-FDA2020-Plexiform-NF-OMIM-613113"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 encodes neurofibromin, the RAS GTPase activating protein: "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; RAS-GAP domain (GRD, aa 1198-1530): accelerates RAS GTP hydrolysis by 1000-fold; "
            "  Sec14 domain (aa 1560-1700): lipid-binding; "
            "  PH-like domain; nuclear localisation signals; "
            "  Alternative splicing: exon 9a (CNS isoform), exon 23a (GRD isoform — reduced GAP activity); "
            "FUNCTION: "
            "  RAS-GAP: neurofibromin converts RAS-GTP → RAS-GDP — principal NEGATIVE regulator of RAS; "
            "  LOF → RAS-GTP accumulates → RAF-MEK-ERK and PI3K-AKT-mTOR constitutively active; "
            "  Schwann cell predisposition: NF1 LOF in Schwann cells → neurofibroma, MPNST; "
            "  Breast predisposition mechanism: RAS pathway overactivation in breast epithelium; "
            "  KNUDSON TWO-HIT: somatic second hit in target tissue drives tumour; "
            "CANCER RISKS: "
            "  BREAST: 50% elevated risk, early onset <40yr — annual MRI from 30yr mandatory; "
            "  MPNST (malignant peripheral nerve sheath tumour): 8-13% lifetime — HIGHEST life-threatening NF1 cancer; "
            "  OPTIC PATHWAY GLIOMA: 15-20% (children); selumetinib first-line; "
            "  GLIOBLASTOMA: 2-3x; LEUKAEMIA: JMML in children (somatic NF1 LOF; very rare germline-associated); "
            "  GASTROINTESTINAL STROMAL TUMOUR: 3-7% (NF1-type: KIT/PDGFRA WT, imatinib resistance — different biology); "
            "  ADRENOCORTICAL: 1-3x modest elevation; "
            "KEY MANAGEMENT: "
            "  ANNUAL MRI BREAST from 30yr MANDATORY: NF1 breast risk requires MRI not just mammogram; "
            "  EARLY ONSET BREAST: highest risk in 30s-40s; bilateral mastectomy discussed; "
            "  SELUMETINIB FDA 2020: MEK inhibitor for NF1 paediatric plexiform neurofibroma (first approved therapy); "
            "  AVOID RADIATION if possible: RT can stimulate MPNST in NF1 carriers — not absolute CI but discuss; "
            "  CAFE-AU-LAIT MACULES: ≥6 macules >0.5cm prepubertal (>1.5cm postpubertal) PATHOGNOMONIC criteria; "
            "  MPNST SURVEILLANCE: annual whole-body MRI if deep plexiform NF on prior imaging; "
            "  OPHTHALMOLOGY: Lisch nodules (iris hamartomas) PATHOGNOMONIC in adults; annual slit-lamp"
        ),
        "syndrome": "Neurofibromatosis Type 1 (NF1)",
        "inheritance": "AD LOF (autosomal dominant; 50% de novo)",
        "breast_risk": "50% elevated risk early onset <40yr; annual MRI from 30yr mandatory",
        "pathognomonic": "CAFE-AU-LAIT MACULES ≥6 (prepubertal >0.5cm, postpubertal >1.5cm) PATHOGNOMONIC; LISCH NODULES (iris hamartomas) PATHOGNOMONIC adults; plexiform NF PATHOGNOMONIC",
        "key_avoid": "AVOID RT if possible — stimulates MPNST development; do NOT miss NF1 breast risk — annual MRI mandatory from 30yr; GISTs are KIT-WT in NF1 = imatinib RESISTANT",
        "key_rule": "Annual MRI breast from 30yr MANDATORY; CAFE-AU-LAIT ≥6 = NF1 diagnosis; MPNST whole-body MRI surveillance; selumetinib FDA2020 plexiform NF",
        "surveillance": "Annual MRI breast from 30yr; whole-body MRI for MPNST if plexiform NF; annual ophthalmology; annual blood pressure (renovascular HT); annual dermatology",
        "targeted_rx": "Selumetinib (MEK inhibitor, FDA2020 — plexiform NF children); Cabozantinib (MPNST investigational); Binimetinib+Ribociclib (MPNST trials)",
    },
]

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]

TUMOUR_TYPES_BY_GENE = {
    "BRCA1": ["Invasive-Breast-TNBC", "Invasive-Breast-ER+", "DCIS", "Ovarian-HGSOC", "Fallopian-Tube"],
    "BRCA2": ["Invasive-Breast-ER+", "Invasive-Breast-HER2+", "TNBC", "Ovarian", "Male-Breast"],
    "PALB2": ["Invasive-Breast-TNBC", "Invasive-Breast-ER+", "DCIS", "Pancreatic", "Ovarian"],
    "CHEK2": ["Invasive-Breast-ER+", "Invasive-Breast-HER2+", "DCIS", "Contralateral-Breast", "Colon"],
    "ATM":   ["Invasive-Breast-ER+", "Invasive-Breast-TNBC", "Pancreatic", "Lymphoma", "Contralateral-Breast"],
    "CDH1":  ["Invasive-Lobular-ILC", "LCIS", "Diffuse-Gastric-DGC", "Breast-Bilateral", "Gastric-Early"],
    "STK11": ["Invasive-Breast-ER+", "Invasive-Breast-TNBC", "Pancreatic", "Colorectal-PJS", "Ovarian-SCTAT"],
    "NF1":   ["Invasive-Breast-ER+", "Invasive-Breast-TNBC", "MPNST", "Optic-Glioma", "GIST-NF1-Type"],
}

VARIANTS_BY_GENE = {
    "BRCA1": ["c.68_69delAG (185delAG) AJ-Founder", "c.5266dupC (5382insC) AJ/Slavic-Founder",
              "c.3756_3759delGTCT", "c.1016dupA", "c.5123C>A p.Ala1708Glu"],
    "BRCA2": ["c.5946delT (6174delT) AJ-Founder", "c.9976A>T p.Lys3326Ter",
              "c.1832G>A p.Arg611Ile", "c.4479_4482delTGAA", "c.6275_6276delTT"],
    "PALB2": ["c.1592delT p.Leu531fs", "c.3113G>A p.Trp1038Ter",
              "c.2257C>T p.Arg753Ter", "c.3201C>A p.Tyr1067Ter", "c.2816T>G p.Leu939Trp"],
    "CHEK2": ["c.1100delC p.Thr367fs (Northern-European)", "c.470T>C p.Ile157Thr (Central-Eastern-European)",
              "c.1283C>T p.Ser428Phe", "c.444+1G>A (splice)", "c.319+2T>A (splice)"],
    "ATM":   ["c.7271T>G p.Val2424Gly (Breast-Founder)", "c.1066-6T>G (splice)",
              "c.3161C>T p.Ser1054Phe", "c.8147T>C p.Val2716Ala", "c.2250G>A p.Trp750Ter"],
    "CDH1":  ["c.1137G>A p.Trp379Ter (Portuguese-Founder)", "c.2440G>T p.Glu814Ter",
              "c.1212dupC", "c.387+1G>A (splice)", "c.1792C>T p.Arg598Ter"],
    "STK11": ["c.962C>T p.Pro321Leu", "c.1A>G p.Met1Val",
              "c.793C>T p.Arg265Trp", "Exon1-2-deletion (common)", "c.460G>A p.Asp154Asn"],
    "NF1":   ["c.2041C>T p.Arg681Ter", "c.6789-1G>T (splice)",
              "Exon1-27-deletion (large)", "c.7126C>T p.Arg2376Ter", "c.2033dup p.Glu679Glyfs"],
}

TREATMENT_PROTOCOLS_BY_GENE = {
    "BRCA1": [
        "Olaparib 300mg BID (OlympiAD: metastatic; OlympiA: adjuvant post-neoadjuvant chemotherapy)",
        "Neoadjuvant: cisplatin + cyclophosphamide OR carboplatin + docetaxel preferred in TNBC",
        "Pembrolizumab + chemotherapy (KEYNOTE-522: pCR 64.8% vs 51.2%; EFS benefit)",
        "Bilateral prophylactic mastectomy: reduces breast risk 90-95%",
        "RRSO 35-40yr post-childbearing: reduces ovarian risk 80-96%; reduces breast risk ~50%",
        "Platinum maintenance post-response in TNBC metastatic setting",
    ],
    "BRCA2": [
        "Olaparib 300mg BID (OlympiAD metastatic; OlympiA adjuvant; PROfound prostate; POLO pancreatic)",
        "Platinum-containing regimen: carboplatin preferred in HER2-negative metastatic",
        "RRSO 40-45yr; MALE: annual PSA from 40yr; annual mammography from 40yr",
        "Male breast: standard breast cancer treatment; aromatase inhibitor caution if premenopausal woman",
        "NIRAPARIB (BRACA1/2 eligible); TALAZOPARIB (FDA-approved HER2-negative BRCA1/2 metastatic)",
        "Enhertu (trastuzumab deruxtecan) if HER2-low metastatic (DESTINY-Breast04)",
    ],
    "PALB2": [
        "Olaparib (TBCRC048: 82% ORR in PALB2-mutated metastatic — HIGHEST non-BRCA response)",
        "Platinum-based chemotherapy: cisplatin/carboplatin (HR-deficient; SBS3 signature)",
        "Annual MRI breast from 30yr; alternating MRI + mammography every 6 months",
        "Bilateral mastectomy discussed given 53% lifetime risk",
        "Niraparib (PALB2 eligible in some PARPi trials); Rucaparib emerging data",
        "BIALLELIC (FA-N): allogeneic HSCT — bone marrow failure; avoid alkylating agents absolutely",
    ],
    "CHEK2": [
        "NO PARPi standard — CHEK2 tumours not reliably HR-deficient",
        "Annual MRI breast from 30-40yr (risk-stratified); alternating mammography",
        "Contralateral breast cancer: elevated 2-3x — contralateral prophylactic mastectomy discussed after ipsilateral BC",
        "Standard chemotherapy per histology/receptor subtype (ER+: endocrine therapy; TNBC: standard regimen)",
        "No approved targeted therapy for CHEK2 germline; somatic HRD testing before PARPi consideration",
        "Chemoprevention: tamoxifen (premenopausal) or aromatase inhibitor (postmenopausal) for risk reduction",
    ],
    "ATM": [
        "Ceralasertib (AZD6738, ATRi): synthetic lethality with ATM LOF — clinical trials (OLAPCO + others)",
        "Olaparib: modest benefit monoallelic ATM (PROfound prostate HR 0.72; breast: emerging data)",
        "RT RISK DISCLOSURE: monoallelic — intermediate radiosensitivity; discuss; biallelic A-T: RT ABSOLUTELY CI",
        "Annual MRI breast from 30-40yr; risk stratified management",
        "Biallelic A-T: IVIG for immunodeficiency; physiotherapy for ataxia; AFP monitoring; avoid RT absolutely",
        "Pancreatic: annual MRI/EUS from 40yr (ATM 5-8x elevated pancreatic risk)",
    ],
    "CDH1": [
        "TOTAL PROPHYLACTIC GASTRECTOMY 20-30yr: only curative intervention for HDGC",
        "Annual breast MRI from 30yr: mammogram unreliable for ILC (lobular ill-defined mammographically)",
        "Prophylactic (contralateral) mastectomy: discussed given 42% ILC risk",
        "ILC endocrine therapy (ER+): aromatase inhibitor preferred over tamoxifen post-menopause",
        "Gastroscopy Cambridge Protocol: annual + multiple (30+) biopsies if gastrectomy deferred",
        "HDGC gastric: chemotherapy FLOT or ECF/ECX; trastuzumab if HER2+; no PARPi",
    ],
    "STK11": [
        "Annual MRI breast from 25yr MANDATORY (50% lifetime HIGHEST non-BRCA)",
        "GI endoscopy from 8yr: upper + lower every 2-3yr; small bowel capsule endoscopy",
        "Annual MRI/EUS pancreatic from 30yr: STK11 EARLIEST onset pancreatic cancer (from 30yr)",
        "Bilateral mastectomy: discussed given 50% lifetime risk; individual decision",
        "Gynaecological: annual review; SCTAT surveillance; adenoma malignum cervix awareness",
        "mTOR inhibitors (everolimus): investigational for PJS polyposis; not standard",
    ],
    "NF1": [
        "Annual MRI breast from 30yr MANDATORY: early onset <40yr elevated risk",
        "SELUMETINIB FDA 2020: MEK inhibitor for paediatric NF1 plexiform neurofibroma (OPG first-line)",
        "MPNST: multimodal — surgery + RT (not preferred in NF1) ± chemotherapy (doxorubicin + ifosfamide)",
        "Avoid RT if possible for breast cancer in NF1 — risk of secondary MPNST",
        "GIST-NF1: KIT-WT — imatinib RESISTANT; sunitinib or everolimus preferred",
        "Binimetinib + ribociclib (MEKi + CDK4/6i): MPNST clinical trials",
    ],
}

SURVEILLANCE_BY_GENE = {
    "BRCA1": ["Annual MRI breast from 25yr", "Alternating MRI + mammography 6-monthly", "RRSO 35-40yr", "Annual pelvic review until RRSO", "Monthly self-exam"],
    "BRCA2": ["Annual MRI breast from 25yr", "RRSO 40-45yr", "Male: annual mammography + PSA from 40yr", "Skin surveillance melanoma", "Pancreatic EUS consider 50yr+"],
    "PALB2": ["Annual MRI breast from 30yr", "Alternating MRI + mammography", "Pancreatic MRI/EUS consider 50yr+", "RRSO timing controversial — discuss 45-50yr"],
    "CHEK2": ["Annual MRI 30-40yr risk-stratified", "Alternating mammography", "Contralateral vigilance after ipsilateral BC", "Colon: colonoscopy from 40yr (2-3x elevated)"],
    "ATM":   ["Annual MRI breast from 30-40yr", "Biallelic A-T: AFP, CBC, immunoglobulins annually", "Pancreatic EUS/MRI from 40yr", "Physiotherapy ataxia assessment biallelic"],
    "CDH1":  ["Annual MRI breast from 30yr", "Total gastrectomy by 30yr", "Annual gastroscopy Cambridge Protocol if deferred", "Annual gynaecological review"],
    "STK11": ["Annual MRI breast from 25yr", "GI endoscopy from 8yr every 2-3yr", "Pancreatic MRI/EUS from 30yr", "Annual gynaecological review from 18yr", "LCCSCT males: annual scrotal USS"],
    "NF1":   ["Annual MRI breast from 30yr", "Whole-body MRI if plexiform NF (MPNST risk)", "Annual ophthalmology (Lisch nodules; OPG)", "Annual blood pressure (renovascular HT)", "Annual dermatology"],
}


def _generate_patient(gene_idx: int, patient_idx: int) -> dict:
    gene_info = ATLAS_GENES[gene_idx]
    gene = gene_info["gene"]
    seed = SEED_BASE + gene_idx + patient_idx * len(ATLAS_GENES)
    rng = random.Random(seed)

    age_ranges = {
        "BRCA1": (28, 52), "BRCA2": (30, 58), "PALB2": (30, 56),
        "CHEK2": (38, 65), "ATM": (35, 62), "CDH1": (28, 55),
        "STK11": (25, 55), "NF1": (28, 48),
    }
    lo, hi = age_ranges[gene]
    age = rng.randint(lo, hi)

    tumours = TUMOUR_TYPES_BY_GENE[gene]
    tumour = rng.choice(tumours)
    variant = rng.choice(VARIANTS_BY_GENE[gene])

    cr_rates = {"BRCA1": 0.72, "BRCA2": 0.68, "PALB2": 0.71, "CHEK2": 0.64,
                "ATM": 0.59, "CDH1": 0.58, "STK11": 0.62, "NF1": 0.48}
    radiation_rates = {"BRCA1": 0.38, "BRCA2": 0.42, "PALB2": 0.36, "CHEK2": 0.52,
                       "ATM": 0.30, "CDH1": 0.32, "STK11": 0.44, "NF1": 0.22}
    relapse_rates = {"BRCA1": 0.28, "BRCA2": 0.30, "PALB2": 0.27, "CHEK2": 0.24,
                     "ATM": 0.35, "CDH1": 0.40, "STK11": 0.33, "NF1": 0.38}

    return {
        "gene": gene,
        "patient_id": f"HBCA-{gene}-{seed}",
        "age_at_dx": age,
        "tumour_type": tumour,
        "variant": variant,
        "cr": rng.random() < cr_rates[gene],
        "radiation": rng.random() < radiation_rates[gene],
        "relapse": rng.random() < relapse_rates[gene],
    }


def _generate_cohort() -> list:
    cohort = []
    for gi in range(len(ATLAS_GENES)):
        for pi in range(40):
            cohort.append(_generate_patient(gi, pi))
    return cohort


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
        "atlas": "Hereditary-Breast-Cancer-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Breast Cancer Predisposition Reference — BRCA1-BRCA2-PALB2-CHEK2-ATM-CDH1-STK11-NF1",
        "seeds": f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
        "total_patients": n,
        "cr_pct": round(100 * cr_n / n, 1),
        "radiation_pct": round(100 * rad_n / n, 1),
        "relapse_pct": round(100 * relapse_n / n, 1),
        "mean_age_at_dx": mean_age,
        "genes": _GENE_LIST,
        "gene_summary": gene_summary,
        "key_rules": [
            "OLAPARIB FDA (BRCA1/BRCA2) — OlympiAD metastatic + OlympiA adjuvant; PALB2 TBCRC048 82% ORR HIGHEST non-BRCA",
            "RRSO 35-40yr MANDATORY (BRCA1) — reduces ovarian risk 80-96%; 40-45yr for BRCA2",
            "NO PARPi standard (CHEK2) — CHEK2 tumours not reliably HR-deficient; do NOT conflate with BRCA",
            "RADIOSENSITIVITY-ABSOLUTE (ATM biallelic A-T) — RT absolutely CI biallelic; disclose risk monoallelic",
            "TOTAL GASTRECTOMY MANDATORY 20-30yr (CDH1) — HDGC; mammogram unreliable for lobular ILC; annual MRI 30yr",
            "BREAST 50% LIFETIME (STK11) — HIGHEST non-BRCA single-gene; GI endoscopy from 8yr MANDATORY; SCTAT PATHOGNOMONIC",
            "ANNUAL MRI BREAST from 30yr (NF1) — early onset <40yr; cafe-au-lait ≥6 PATHOGNOMONIC; selumetinib FDA2020",
            "MAMMOGRAM UNRELIABLE for lobular ILC (CDH1) — MRI mandatory; CDH1 = lobular breast phenotype",
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
            "breast_risk": gene_info["breast_risk"],
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
        "atlas": "Hereditary-Breast-Cancer-Predisposition-Atlas",
        "definitions": {
            "hboc1_brca1": (
                "HBOC1 (BRCA1): RING-BRCT HR repair scaffold — breast 46-87% lifetime; TNBC-enriched 70-80%; "
                "RRSO 35-40yr MANDATORY (reduces ovarian 80-96%; breast ~50%); "
                "BILATERAL MASTECTOMY 90-95% risk reduction; "
                "OLAPARIB OlympiAD (metastatic) + OlympiA/OLYMPIA (adjuvant) — FDA approved; "
                "PLATINUM: cisplatin/carboplatin TNBC neoadjuvant preferred; "
                "PEMBROLIZUMAB + chemotherapy: KEYNOTE-522 pCR 64.8% TNBC neoadjuvant"
            ),
            "hboc2_brca2": (
                "HBOC2 (BRCA2): RAD51 mediator — BRC repeats; breast 38-65% lifetime (ER+ more common); "
                "MALE BREAST 6% HIGHEST hereditary — annual mammography from 40yr; "
                "RRSO 40-45yr (5yr later than BRCA1); "
                "OLAPARIB: metastatic breast + prostate PROfound + pancreatic POLO; "
                "FANCD1 biallelic: FA-D1 — childhood desmoplastic medulloblastoma + Wilms + AVOID alkylating agents"
            ),
            "palb2_hboc3": (
                "PALB2 (BRCA1-BRCA2 Bridge): WD40 scaffold — breast 53% lifetime HIGHEST non-BRCA PARPi-sensitive; "
                "TBCRC048 OLAPARIB: 82% ORR HIGHEST response of any non-BRCA gene; "
                "RRSO TIMING CONTROVERSIAL — some guidelines 45-50yr; "
                "BIALLELIC FA-N: Fanconi Anaemia — bone marrow failure; AVOID alkylating agents; "
                "Annual MRI breast from 30yr — treat as BRCA2-equivalent for surveillance"
            ),
            "chek2_moderate": (
                "CHEK2 (CHK2 Checkpoint Kinase): FHA+kinase — ATM-activated DSB checkpoint; "
                "1100delC (Northern/Western European): 20-25% lifetime breast; "
                "I157T (Central/Eastern European): 18-20% lifetime breast; "
                "MODERATE RISK — NO PARPi standard; surveillance not universal prophylaxis; "
                "CONTRALATERAL 2-3x elevated after ipsilateral BC; "
                "DO NOT CONFLATE WITH BRCA1/2 — moderate not high risk"
            ),
            "atm_moderate_at": (
                "ATM (Ataxia-Telangiectasia Mutated): PIKK kinase — master DSB sensor; "
                "MONOALLELIC: 30-35% moderate breast; pancreatic 5-8x; "
                "BIALLELIC A-T: cerebellar ataxia PATHOGNOMONIC; telangiectasias PATHOGNOMONIC; AFP elevated DIAGNOSTIC; "
                "RADIOSENSITIVITY-ABSOLUTE BIALLELIC: RT absolutely CI; disclose risk monoallelic; "
                "CERALASERTIB (ATRi): synthetic lethality clinical trials; "
                "BIALLELIC: immunodeficiency — live vaccines discuss with immunologist"
            ),
            "cdh1_hdgc_ilc": (
                "CDH1 (E-Cadherin): calcium-dependent adhesion — AJ adherens junction scaffold; "
                "HDGC: diffuse gastric 67-83% HIGHEST; TOTAL GASTRECTOMY MANDATORY 20-30yr; "
                "LOBULAR BREAST ILC: 42% lifetime PATHOGNOMONIC — mammogram unreliable for lobular pattern; "
                "ANNUAL MRI BREAST from 30yr — MRI detects ILC reliably; mammogram adds false negative risk; "
                "GASTROSCOPY Cambridge Protocol: annual + ≥30 biopsies if gastrectomy deferred — NOT substitute; "
                "PROPHYLACTIC MASTECTOMY: discussed for 42% ILC risk; bilateral risk-reducing mastectomy an option"
            ),
            "stk11_pjs_breast": (
                "STK11/LKB1 (AMPK Kinase): LKB1-STRAD-MO25 complex — mTOR suppressor; "
                "BREAST 50% LIFETIME HIGHEST non-BRCA single-gene penetrance; "
                "MUCOCUTANEOUS MACULES PATHOGNOMONIC: perioral + buccal + digital in childhood; "
                "SCTAT (sex cord tumour with annular tubules) OVARIAN PATHOGNOMONIC; "
                "LCCSCT TESTICULAR PATHOGNOMONIC in males; "
                "GI ENDOSCOPY from 8yr MANDATORY: upper + lower every 2-3yr; small bowel capsule; "
                "PANCREATIC MRI/EUS from 30yr: EARLIEST onset of all hereditary pancreatic cancer syndromes"
            ),
            "nf1_breast_mpnst": (
                "NF1 (Neurofibromin): RAS-GAP — neurofibromin converts RAS-GTP→GDP; "
                "BREAST: 50% elevated early onset <40yr; ANNUAL MRI from 30yr MANDATORY; "
                "CAFE-AU-LAIT MACULES ≥6 (>0.5cm prepubertal) PATHOGNOMONIC; "
                "LISCH NODULES (iris hamartomas) PATHOGNOMONIC adults; "
                "MPNST: 8-13% lifetime HIGHEST life-threatening NF1 cancer; whole-body MRI surveillance; "
                "SELUMETINIB FDA 2020: first approved MEK inhibitor plexiform NF paediatric; "
                "AVOID RT if possible — risk secondary MPNST in RT field; GIST-NF1 = KIT-WT imatinib RESISTANT"
            ),
            "cascade_testing": (
                "CASCADE TESTING Hereditary Breast Cancer Predisposition: "
                "index case identified → first-degree relatives tested (50% chance each); "
                "BRCA1/2/PALB2: offer genetic testing to all first-degree regardless of age; "
                "CHEK2/ATM/NF1: first-degree relatives at 50% risk — test proactively; "
                "CDH1: ALL first-degree relatives — total gastrectomy decision depends on carrier status; "
                "STK11: PJS clinical diagnosis may precede molecular — mucocutaneous macules = test immediately; "
                "Predictive testing: carrier status determines lifelong surveillance intensity and prophylactic intervention timing"
            ),
        },
        "key_clinical_distinctions": [
            "BRCA1 vs BRCA2: TNBC 70-80% BRCA1 vs ER+ more common BRCA2; RRSO 35-40yr vs 40-45yr; male breast BRCA2 6% HIGHEST",
            "PALB2 vs CHEK2: PALB2 53% risk = BRCA2-equivalent — treat as high risk; CHEK2 18-25% = moderate risk — surveillance not prophylaxis",
            "ATM biallelic vs monoallelic: BIALLELIC A-T = telangiectasias + cerebellar ataxia + RT ABSOLUTELY CI; MONOALLELIC = moderate breast risk + intermediate RT sensitivity",
            "CDH1 ILC vs other breast: mammogram UNRELIABLE for lobular — MRI mandatory; ILC = CDH1 carrier breast phenotype PATHOGNOMONIC",
            "STK11 breast 50%: HIGHEST non-BRCA single-gene — annual MRI from 25yr; also HIGHEST pancreatic risk onset earliest",
            "NF1 breast early onset <40yr: cafe-au-lait ≥6 PATHOGNOMONIC; selumetinib FDA2020; AVOID RT → MPNST risk; GIST imatinib RESISTANT",
            "CHEK2 1100delC (Northern European) vs I157T (Central/Eastern European): DO NOT CONFLATE penetrance — different variant, different counselling",
            "PALB2 biallelic FA-N: AVOID alkylating agents; BRCA2 biallelic FA-D1: childhood desmoplastic medulloblastoma + Wilms",
        ],
    }
