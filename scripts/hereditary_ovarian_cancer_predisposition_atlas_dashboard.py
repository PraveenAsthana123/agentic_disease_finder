#!/usr/bin/env python3
"""Hereditary-Ovarian-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
BRCA1   (BRCA1 DNA repair associated; 1863aa; 17q21.31; AD LOF;
         Hereditary Breast and Ovarian Cancer Syndrome type 1 (HBOC1);
         ovarian 39-44% lifetime HIGHEST HGSOC; RRSO 35-40yr MANDATORY;
         olaparib FDA maintenance 1st-line + recurrence; niraparib; rucaparib;
         seed SEED_BASE+0) .
BRCA2   (BRCA2 DNA repair associated; 3418aa; 13q12.3; AD LOF;
         HBOC2 / Fanconi Anaemia type D1 (biallelic);
         ovarian 11-17% lifetime; later onset ~55yr vs BRCA1 ~50yr;
         olaparib PROfound; RRSO 40-45yr;
         seed SEED_BASE+1) .
BRIP1   (BRCA1-interacting protein 1 / FANCJ; 1249aa; 17q23.2; AD LOF;
         HOCA — ovarian 5-8x relative risk; NO significant breast risk;
         moderate-risk gene; RRSO 45-50yr; Fanconi Anaemia FANCJ biallelic;
         seed SEED_BASE+2) .
RAD51C  (RAD51 paralog C / FANCO; 376aa; 17q22; AD LOF;
         HOCA — ovarian 6% lifetime 5-7x relative risk; NO breast risk;
         Fanconi Anaemia FANCO biallelic; RRSO 45-50yr;
         seed SEED_BASE+3) .
RAD51D  (RAD51 paralog D; 328aa; 17q12; AD LOF;
         HOCA — ovarian 10% lifetime 7-10x relative risk; NO breast risk;
         PARP inhibitor sensitivity; RRSO 45-50yr;
         seed SEED_BASE+4) .
PALB2   (Partner and localiser of BRCA2 / FANCN; 1186aa; 16p12.2; AD LOF;
         HOCA-moderate — ovarian 3-5% lifetime; breast 53% HIGHEST non-BRCA;
         TBCRC048 olaparib 82% ORR; Fanconi Anaemia FANCN biallelic;
         seed SEED_BASE+5) .
MLH1    (MutL homolog 1; 756aa; 3p22.2; AD LOF;
         Lynch syndrome type 1 — ovarian 8-13% endometrioid/clear cell NOT HGSOC;
         MSI-H PATHOGNOMONIC; pembrolizumab FDA 2017 tumour-agnostic;
         H. pylori eradication MANDATORY; aspirin CAPP2 50% risk reduction;
         seed SEED_BASE+6) .
STK11   (Serine/threonine kinase 11 / LKB1; 433aa; 19p13.3; AD LOF;
         Peutz-Jeghers Syndrome — SCTAT (sex cord tumour with annular tubules) PATHOGNOMONIC;
         mucocutaneous macules PATHOGNOMONIC; ovarian 21% lifetime;
         GI endoscopy from 8yr MANDATORY; pancreatic MRI/EUS from 30yr;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3374-3381)
"""
import random

SEED_BASE = 3374

ATLAS_GENES = [
    {
        "gene": "BRCA1",
        "protein": (
            "BRCA1 -- 17q21.31 Autosomal-Dominant-LOF -- 1863aa -- "
            "RING-BRCT-HR-Repair-Scaffold-208kDa-"
            "Ovarian-39-44pct-HIGHEST-HGSOC-Fallopian-Tube-Origin-"
            "RRSO-35-40yr-MANDATORY-Reduces-Ovarian-80-96pct-"
            "Olaparib-1st-Line-Maintenance-PAOLA-1-PRIMA-Niraparib-"
            "Bevacizumab-Combination-GOG218-ICON7-OMIM-113705"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "1863 aa / 208 kDa / 17q21.31 BRCA1 ovarian cancer molecular context: "
            "STRUCTURE: "
            "  1863 aa / 208 kDa; RING domain (aa 1-109): E3 ubiquitin ligase with BARD1; "
            "  BRCT repeat domain (aa 1646-1863): phosphopeptide binding; ATM/CHEK2 phosphorylation; "
            "  Central HR scaffold: recruits RAD51 via PALB2-BRCA2 axis; "
            "CANCER RISKS (OVARIAN FOCUS): "
            "  OVARIAN: 39-44% lifetime (HGSOC predominant; fallopian tube origin established; peritoneal 3-5%); "
            "  ONSET: median 50yr vs sporadic 63yr; BRCA1 earlier than BRCA2 (~5yr gap); "
            "  BREAST: 72% lifetime (co-expressed risk — managed separately); "
            "  FALLOPIAN TUBE: included in RRSO risk reduction target; "
            "KEY MANAGEMENT (OVARIAN): "
            "  RRSO 35-40yr post-childbearing: reduces ovarian/FT/peritoneal risk 80-96%; "
            "  OLAPARIB 1st-line maintenance: PAOLA-1 (olaparib+bevacizumab) HR+ subgroup; PRIMA (niraparib); "
            "  OLAPARIB recurrence maintenance: SOLO2 — 19.1 vs 5.5mo PFS; "
            "  BEVACIZUMAB: GOG218 / ICON7 — survival benefit in advanced disease; "
            "  PLATINUM SENSITIVITY: carboplatin/paclitaxel standard; HRD score informs PARPi eligibility; "
            "  SURVEILLANCE post-RRSO: annual CA-125 + pelvic exam (low yield — surgical prevention preferred)"
        ),
        "syndrome": "HBOC1 (Hereditary Breast and Ovarian Cancer Syndrome type 1)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "ovarian_risk": "39-44% lifetime",
        "pathognomonic": "HGSOC on pathology in BRCA1 carrier; HRD scar (SBS3) on tumour sequencing",
        "key_avoid": "Do NOT omit PARPi maintenance if BRCA1-mutated HGSOC — olaparib/niraparib/rucaparib standard of care",
        "key_rule": "RRSO 35-40yr MANDATORY post-childbearing — reduces ovarian risk 80-96%; fallopian tube salpingectomy alone insufficient for BRCA1",
        "surveillance": "Transvaginal USS + CA-125 every 6mo pre-RRSO (low sensitivity — supplementary); RRSO preferred; annual breast MRI from 25yr",
        "targeted_rx": "Olaparib (PAOLA-1/SOLO2); Niraparib (PRIMA/NOVA); Rucaparib (ARIEL); Bevacizumab (GOG218/ICON7); Pembrolizumab (KEYNOTE-100 HRD+)",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "RAD51-Mediator-BRC-Repeats-HR-Scaffold-384kDa-FANCD1-"
            "Ovarian-11-17pct-Later-Onset-55yr-vs-BRCA1-50yr-"
            "RRSO-40-45yr-5yr-Later-Than-BRCA1-"
            "Olaparib-SOLO1-SOLO2-PROfound-POLO-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 ovarian cancer molecular context: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; 8 BRC repeats (aa 1002-2085): RAD51 binding — each 35aa; "
            "  C-terminal DBD: ssDNA binding; nuclear export (NLS aa 3263-3269); "
            "  PALB2 interaction: WD40 domain; BRCA1-BRCA2 bridge; "
            "CANCER RISKS (OVARIAN FOCUS): "
            "  OVARIAN: 11-17% lifetime; later onset than BRCA1 (median 55yr vs 50yr); "
            "  MORE OFTEN ER+ histology; mixed histology; "
            "  FALLOPIAN TUBE primary: BRCA2 also a major cause; "
            "KEY MANAGEMENT (OVARIAN): "
            "  RRSO 40-45yr (5yr later than BRCA1 — lifetime risk lower); "
            "  OLAPARIB: SOLO1 (1st-line maintenance after platinum CR/PR), SOLO2 (recurrence); "
            "  BIALLELIC (FA-D1): childhood desmoplastic medulloblastoma + Wilms + AVOID alkylating agents; "
            "  PLATINUM SENSITIVITY: carboplatin/paclitaxel; HRD score predicts PARPi benefit; "
            "  MALE CARRIERS: prostate cancer 24-40% lifetime — annual PSA from 40yr"
        ),
        "syndrome": "HBOC2 (Hereditary Breast and Ovarian Cancer Syndrome type 2) / Fanconi Anaemia D1 (biallelic)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function; biallelic FA-D1)",
        "ovarian_risk": "11-17% lifetime",
        "pathognomonic": "HGSOC (later onset ~55yr) in BRCA2 carrier; desmoplastic medulloblastoma (biallelic FA-D1)",
        "key_avoid": "Biallelic FA-D1: AVOID alkylating agents (cyclophosphamide, melphalan) — cross-linking DNA damage lethal; AVOID high-dose RT",
        "key_rule": "RRSO 40-45yr (5yr later than BRCA1 — reflects lower absolute ovarian risk); prostate surveillance from 40yr for male carriers",
        "surveillance": "Transvaginal USS + CA-125 every 6mo pre-RRSO; RRSO 40-45yr; annual breast MRI from 25yr; prostate PSA from 40yr (males)",
        "targeted_rx": "Olaparib (SOLO1/SOLO2); Niraparib (PRIMA); Bevacizumab; Platinum-containing chemotherapy",
    },
    {
        "gene": "BRIP1",
        "protein": (
            "BRIP1 -- 17q23.2 Autosomal-Dominant-LOF -- 1249aa -- "
            "BRCA1-Interacting-Helicase-FANCJ-140kDa-5prime-to-3prime-DNA-Helicase-"
            "Ovarian-5-8x-Relative-Risk-NO-Significant-Breast-Risk-"
            "RRSO-45-50yr-Moderate-Risk-Gene-"
            "Fanconi-Anaemia-FANCJ-Biallelic-OMIM-605882"
        ),
        "locus": "17q23.2",
        "protein_size": (
            "1249 aa / 140 kDa / 17q23.2 BRIP1/FANCJ ovarian cancer context: "
            "STRUCTURE: "
            "  1249 aa / 140 kDa; DEAH helicase domain (aa 1-833); BRCA1-BRCT binding domain; "
            "  5' to 3' DNA helicase activity; resolves G-quadruplex structures; "
            "  FA pathway: FANCJ — ICL repair downstream of FANCD2 monoubiquitination; "
            "CANCER RISKS (OVARIAN FOCUS): "
            "  OVARIAN: 5-8x relative risk (absolute risk ~5-8% lifetime); "
            "  IMPORTANT: NO significant breast cancer risk (key distinction from BRCA1/BRCA2); "
            "  HGSOC histology predominant — similar to BRCA1/2; "
            "KEY MANAGEMENT (OVARIAN): "
            "  RRSO 45-50yr (later than BRCA1/2 — lower absolute risk; post-childbearing); "
            "  MODERATE RISK — PARPi eligibility varies by HRD score; emerging clinical data; "
            "  BIALLELIC FA-J: Fanconi Anaemia — bone marrow failure, AML, solid tumours in childhood; "
            "  SURVEILLANCE: some guidelines recommend transvaginal USS + CA-125 pre-RRSO; "
            "  NO breast surveillance intensification required (unlike BRCA1/2)"
        ),
        "syndrome": "Hereditary Ovarian Cancer type 3 (HOCA3) / Fanconi Anaemia J (FANCJ biallelic)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function; biallelic FA-J)",
        "ovarian_risk": "5-8x relative risk (~5-8% lifetime)",
        "pathognomonic": "HGSOC in BRIP1 carrier — no breast pathognomonic feature",
        "key_avoid": "Do NOT assume breast risk (unlike BRCA1/2) — BRIP1 is ovarian-selective moderate risk",
        "key_rule": "RRSO 45-50yr post-childbearing — NO breast prophylaxis required; distinguish from BRCA1/2 management",
        "surveillance": "Transvaginal USS + CA-125 6-monthly pre-RRSO (some guidelines); RRSO 45-50yr",
        "targeted_rx": "Platinum-containing chemotherapy; PARPi emerging (HRD dependent); olaparib off-label if HRD+",
    },
    {
        "gene": "RAD51C",
        "protein": (
            "RAD51C -- 17q22 Autosomal-Dominant-LOF -- 376aa -- "
            "RAD51-Paralog-C-FANCO-40kDa-HR-Repair-Mediator-"
            "Ovarian-6pct-Lifetime-5-7x-Relative-Risk-NO-Breast-Risk-"
            "RRSO-45-50yr-"
            "Fanconi-Anaemia-FANCO-Biallelic-OMIM-602774"
        ),
        "locus": "17q22",
        "protein_size": (
            "376 aa / 40 kDa / 17q22 RAD51C/FANCO ovarian cancer context: "
            "STRUCTURE: "
            "  376 aa / 40 kDa; RecA/RAD51 ATPase domain; "
            "  Forms two distinct complexes: BCDX2 (RAD51B-RAD51C-RAD51D-XRCC2) and CX3 (RAD51C-XRCC3); "
            "  FA pathway: FANCO — ICL repair; FANCD2 monoubiquitination dependent; "
            "CANCER RISKS (OVARIAN FOCUS): "
            "  OVARIAN: ~6% lifetime (5-7x relative risk); "
            "  NO significant breast cancer risk (key clinical distinction); "
            "  HGSOC histology — HRD scar similar to BRCA1/2; "
            "KEY MANAGEMENT (OVARIAN): "
            "  RRSO 45-50yr; "
            "  BIALLELIC FA-O: Fanconi Anaemia FANCO — rare, childhood onset; "
            "  PARPi sensitivity: emerging data — HRD score predicts; "
            "  SURVEILLANCE: transvaginal USS + CA-125 pre-RRSO"
        ),
        "syndrome": "Hereditary Ovarian Cancer type 4 (HOCA4) / Fanconi Anaemia O (FANCO biallelic)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function; biallelic FA-O)",
        "ovarian_risk": "~6% lifetime (5-7x relative risk)",
        "pathognomonic": "HGSOC in RAD51C carrier; HRD scar — similar molecular phenotype to BRCA1/2",
        "key_avoid": "Do NOT intensify breast surveillance — RAD51C is ovarian-selective; avoid conflating with BRCA1/2 risk profile",
        "key_rule": "RRSO 45-50yr; NO breast prophylaxis; PARPi eligibility based on HRD score (emerging evidence)",
        "surveillance": "Transvaginal USS + CA-125 6-monthly pre-RRSO; RRSO 45-50yr",
        "targeted_rx": "Platinum-containing chemotherapy; PARPi emerging (HRD dependent)",
    },
    {
        "gene": "RAD51D",
        "protein": (
            "RAD51D -- 17q12 Autosomal-Dominant-LOF -- 328aa -- "
            "RAD51-Paralog-D-37kDa-BCDX2-Complex-HR-Mediator-"
            "Ovarian-10pct-Lifetime-7-10x-Relative-Risk-HIGHEST-Non-BRCA-Paralog-"
            "RRSO-45-50yr-PARP-Inhibitor-Sensitivity-"
            "OMIM-602954"
        ),
        "locus": "17q12",
        "protein_size": (
            "328 aa / 37 kDa / 17q12 RAD51D ovarian cancer context: "
            "STRUCTURE: "
            "  328 aa / 37 kDa; RecA/RAD51 ATPase domain; "
            "  BCDX2 complex: RAD51B-RAD51C-RAD51D-XRCC2 — early HR mediator; "
            "  Involved in RAD51 nucleoprotein filament stabilisation on ssDNA; "
            "CANCER RISKS (OVARIAN FOCUS): "
            "  OVARIAN: ~10% lifetime (7-10x relative risk) — HIGHEST among RAD51 paralogs; "
            "  NO significant breast cancer risk; "
            "  HGSOC histology — HRD positive; "
            "KEY MANAGEMENT (OVARIAN): "
            "  RRSO 45-50yr; "
            "  PARP INHIBITOR SENSITIVITY: established in vitro + emerging clinical data; "
            "  Olaparib off-label if HRD+; "
            "  SURVEILLANCE: transvaginal USS + CA-125 pre-RRSO"
        ),
        "syndrome": "Hereditary Ovarian Cancer (HOCA) — RAD51D type",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "ovarian_risk": "~10% lifetime (7-10x relative risk)",
        "pathognomonic": "HGSOC in RAD51D carrier; PARPi sensitivity in vitro",
        "key_avoid": "Do NOT assume breast risk — RAD51D is ovarian-selective; distinct from BRCA1/2",
        "key_rule": "RRSO 45-50yr; PARPi sensitivity — use HRD score to guide olaparib eligibility",
        "surveillance": "Transvaginal USS + CA-125 6-monthly pre-RRSO; RRSO 45-50yr",
        "targeted_rx": "Platinum-containing chemotherapy; Olaparib off-label (HRD+); PARPi sensitivity established",
    },
    {
        "gene": "PALB2",
        "protein": (
            "PALB2 -- 16p12.2 Autosomal-Dominant-LOF -- 1186aa -- "
            "WD40-Domain-BRCA1-BRCA2-Bridge-131kDa-FANCN-"
            "Ovarian-3-5pct-Moderate-Risk-Breast-53pct-HIGHEST-Non-BRCA2-"
            "TBCRC048-Olaparib-82pct-ORR-BRCA2-Anchor-"
            "Fanconi-Anaemia-FANCN-Biallelic-OMIM-610355"
        ),
        "locus": "16p12.2",
        "protein_size": (
            "1186 aa / 131 kDa / 16p12.2 PALB2 ovarian and breast cancer context: "
            "STRUCTURE: "
            "  1186 aa / 131 kDa; WD40 domain (aa 853-1186): BRCA2 N-terminal binding; "
            "  Coiled-coil (aa 44-170): BRCA1 binding domain; "
            "  PALB2 physically links BRCA1 to BRCA2 — essential HR scaffold bridge; "
            "CANCER RISKS: "
            "  OVARIAN: 3-5% lifetime (moderate risk — lower than BRCA1/2); "
            "  BREAST: 53% lifetime HIGHEST non-BRCA2 PARPi-sensitive gene (dominating risk); "
            "  PANCREATIC: 3-4x elevated; "
            "KEY MANAGEMENT (OVARIAN vs BREAST): "
            "  RRSO TIMING: ovarian risk 3-5% moderate — some guidelines 45-50yr; breast priority often drives earlier RRSO; "
            "  TBCRC048 OLAPARIB: 82% ORR — HIGHEST response rate of any non-BRCA gene; "
            "  BIALLELIC FA-N: Fanconi Anaemia — AVOID alkylating agents; "
            "  BREAST MANAGEMENT DOMINATES surveillance intensity"
        ),
        "syndrome": "HBOC3 / Hereditary Ovarian Cancer moderate / Fanconi Anaemia N (biallelic)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function; biallelic FA-N)",
        "ovarian_risk": "3-5% lifetime (moderate)",
        "pathognomonic": "Breast 53% lifetime PATHOGNOMONIC priority; SCTAT ovarian not specific to PALB2",
        "key_avoid": "Biallelic FA-N: AVOID alkylating agents (cyclophosphamide, melphalan — cross-linking lethal)",
        "key_rule": "BREAST 53% risk DOMINATES management — annual MRI from 25yr; RRSO timing guided by breast risk + ovarian moderate risk",
        "surveillance": "Annual breast MRI from 25yr (priority); transvaginal USS + CA-125 (moderate ovarian risk); RRSO 45-50yr or breast-driven earlier",
        "targeted_rx": "Olaparib TBCRC048 (breast HER2-negative, 82% ORR); platinum-containing chemotherapy; olaparib off-label (ovarian HRD+)",
    },
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 756aa -- "
            "MMR-MutL-Alpha-Scaffold-85kDa-Lynch-Syndrome-Type1-"
            "Ovarian-8-13pct-Endometrioid-Clear-Cell-NOT-HGSOC-"
            "MSI-H-PATHOGNOMONIC-Pembrolizumab-FDA2017-Tumour-Agnostic-"
            "Aspirin-CAPP2-50pct-Risk-Reduction-H-Pylori-Eradication-MANDATORY-"
            "OMIM-120436"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "756 aa / 85 kDa / 3p22.2 MLH1 Lynch syndrome ovarian cancer context: "
            "STRUCTURE: "
            "  756 aa / 85 kDa; MLH1-PMS2 heterodimer = MutLα (dominant MMR complex); "
            "  ATPase domain (N-terminal); endonuclease domain (PMS2); "
            "  Responsible for mismatch repair — strand discrimination by PCNA; "
            "CANCER RISKS (OVARIAN): "
            "  OVARIAN: 8-13% lifetime (endometrioid + clear cell — NOT HGSOC); "
            "  ENDOMETRIAL: 40-60% lifetime HIGHEST Lynch cancer for females; "
            "  COLORECTAL: 80% cumulative lifetime risk (HIGHEST Lynch gene); "
            "  IMPORTANT DISTINCTION: MLH1 ovarian cancer is ENDOMETRIOID/CLEAR CELL not HGSOC; "
            "  PARPi NOT standard (dMMR — different from HRD/BRCA pathway); "
            "KEY MANAGEMENT (OVARIAN): "
            "  MSI-H IHC: loss of MLH1 + PMS2 staining PATHOGNOMONIC (somatic methylation vs germline — distinguish!); "
            "  PEMBROLIZUMAB FDA 2017: tumour-agnostic dMMR/MSI-H approval; dMMR Lynch ovarian qualifies; "
            "  ANNUAL TRANSVAGINAL USS + CA-125: low yield but guideline recommended; "
            "  RISK-REDUCING HYSTERECTOMY + BSO: recommended 35-40yr post-childbearing; "
            "  ASPIRIN CAPP2: 600mg/day — 50% colorectal+endometrial+ovarian risk reduction; start early; "
            "  H. PYLORI ERADICATION: MANDATORY — reduces gastric cancer risk"
        ),
        "syndrome": "Lynch Syndrome Type 1 (MLH1) — ovarian endometrioid/clear cell; endometrial HIGHEST",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "ovarian_risk": "8-13% lifetime (endometrioid/clear cell histology)",
        "pathognomonic": "MSI-H on tumour testing PATHOGNOMONIC; MLH1+PMS2 IHC loss; endometrioid/clear cell histology (NOT HGSOC)",
        "key_avoid": "Do NOT use PARPi standard of care (Lynch ovarian is dMMR not HRD — different mechanism); check somatic promoter methylation before germline testing",
        "key_rule": "PEMBROLIZUMAB FDA 2017 tumour-agnostic — dMMR/MSI-H Lynch ovarian cancer qualifies; ASPIRIN CAPP2 from diagnosis; H. pylori eradication MANDATORY",
        "surveillance": "Annual transvaginal USS + CA-125; risk-reducing hysterectomy + BSO 35-40yr; annual colonoscopy; H. pylori test + eradicate; aspirin CAPP2",
        "targeted_rx": "Pembrolizumab (dMMR/MSI-H tumour-agnostic FDA 2017); Dostarlimab (GARNET); Aspirin CAPP2",
    },
    {
        "gene": "STK11",
        "protein": (
            "STK11 -- 19p13.3 Autosomal-Dominant-LOF -- 433aa -- "
            "LKB1-48kDa-AMPK-Activating-Master-Kinase-"
            "Peutz-Jeghers-Syndrome-PJS-SCTAT-Ovarian-PATHOGNOMONIC-"
            "Mucocutaneous-Macules-PATHOGNOMONIC-Ovarian-21pct-Lifetime-"
            "GI-Endoscopy-8yr-MANDATORY-Pancreatic-MRI-EUS-30yr-"
            "OMIM-602216"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "433 aa / 48 kDa / 19p13.3 STK11/LKB1 Peutz-Jeghers ovarian cancer context: "
            "STRUCTURE: "
            "  433 aa / 48 kDa; Serine/threonine kinase domain (aa 49-309); "
            "  LKB1-STRAD-MO25 heterotrimer — activates AMPK family kinases; "
            "  AMPK activation → mTORC1 suppression → metabolic checkpoint; "
            "CANCER RISKS (OVARIAN FOCUS): "
            "  OVARIAN: 21% lifetime (SCTAT sex cord tumour with annular tubules PATHOGNOMONIC); "
            "  SCTAT: bilateral, multifocal, calcified — PJS-SCTAT is almost always benign; sporadic SCTAT = SMAD4 somatic (usually malignant); "
            "  BREAST: 50% lifetime HIGHEST non-BRCA single-gene (co-dominates management); "
            "  PANCREATIC: 11-36% HIGHEST hereditary EARLIEST ONSET from age 30yr; "
            "  GI: small bowel > colon hamartomatous polyps — intussusception emergency; "
            "KEY MANAGEMENT (OVARIAN): "
            "  SCTAT MONITORING: annual pelvic USS; SCTAT-PJS almost always benign — observe vs excise; "
            "  ADENOMA MALIGNUM CERVIX (mucinous adenocarcinoma): PATHOGNOMONIC — annual cervical screening; "
            "  LCCSCT (large cell calcifying Sertoli cell tumour): testicular PATHOGNOMONIC males; "
            "  RISK-REDUCING BSO: ovarian cancer 21% — discuss timing post-childbearing; "
            "PATHOGNOMONIC FEATURES: "
            "  MUCOCUTANEOUS MACULES: perioral + buccal mucosa + digits in childhood — test immediately; "
            "  SCTAT: bilateral calcified sex cord tumours — PJS-associated almost always benign"
        ),
        "syndrome": "Peutz-Jeghers Syndrome (PJS) — SCTAT ovarian PATHOGNOMONIC",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "ovarian_risk": "21% lifetime",
        "pathognomonic": "SCTAT (sex cord tumour with annular tubules) PATHOGNOMONIC for PJS; mucocutaneous macules PATHOGNOMONIC PJS diagnosis",
        "key_avoid": "SCTAT-PJS almost always benign — avoid aggressive surgery unless symptomatic/growing; distinguish from sporadic SCTAT (SMAD4 somatic — malignant behaviour)",
        "key_rule": "SCTAT bilateral calcified = PJS PATHOGNOMONIC — check mucocutaneous macules; GI endoscopy from 8yr MANDATORY; EARLIEST pancreatic cancer onset from 30yr",
        "surveillance": "Annual pelvic USS (SCTAT); annual cervical smear (adenoma malignum); GI endoscopy from 8yr; pancreatic MRI/EUS from 30yr; annual breast MRI from 25yr",
        "targeted_rx": "Emerging mTOR inhibitors (sirolimus/everolimus — LKB1-AMPK-mTOR axis); standard chemotherapy for ovarian malignancy",
    },
]

# Treatment protocols per gene
TREATMENT_PROTOCOLS_BY_GENE = {
    "BRCA1": ["Olaparib PAOLA-1 (1st-line + bevacizumab)", "Niraparib PRIMA (1st-line maintenance)", "Olaparib SOLO2 (recurrence maintenance)", "Bevacizumab GOG218/ICON7", "Carboplatin/Paclitaxel standard", "Pembrolizumab KEYNOTE-100 (HRD+ subset)"],
    "BRCA2": ["Olaparib SOLO1/SOLO2", "Niraparib PRIMA/NOVA", "Bevacizumab combination", "Carboplatin/Paclitaxel", "Rucaparib ARIEL"],
    "BRIP1": ["Carboplatin/Paclitaxel standard", "Olaparib off-label (HRD+)", "Bevacizumab", "Clinical trials (PARPi sensitivity emerging)"],
    "RAD51C": ["Carboplatin/Paclitaxel standard", "PARPi emerging (HRD score guided)", "Bevacizumab", "Olaparib off-label HRD+"],
    "RAD51D": ["Carboplatin/Paclitaxel standard", "Olaparib off-label (HRD+)", "PARPi sensitivity established preclinically", "Clinical trial enrolment recommended"],
    "PALB2": ["Olaparib TBCRC048 (breast primary — 82% ORR)", "Carboplatin/Paclitaxel (ovarian)", "Olaparib off-label (ovarian HRD+)", "Platinum sensitivity"],
    "MLH1": ["Pembrolizumab FDA 2017 (dMMR/MSI-H tumour-agnostic)", "Dostarlimab GARNET", "Carboplatin/Paclitaxel (standard cytotoxic)", "Aspirin CAPP2 (prevention)"],
    "STK11": ["Carboplatin/Paclitaxel (standard)", "Everolimus/Sirolimus (mTOR inhibitors — LKB1 loss)", "Pembrolizumab (KRAS co-mutation — STK11 predicts resistance in NSCLC but not ovarian)"],
}

# Surveillance protocols per gene
SURVEILLANCE_BY_GENE = {
    "BRCA1": ["Annual MRI breast from 25yr", "Transvaginal USS + CA-125 6-monthly (pre-RRSO)", "RRSO 35-40yr post-childbearing MANDATORY", "Annual pelvic exam post-RRSO"],
    "BRCA2": ["Annual MRI breast from 25yr", "Transvaginal USS + CA-125 6-monthly (pre-RRSO)", "RRSO 40-45yr post-childbearing", "Annual PSA from 40yr (males)"],
    "BRIP1": ["Transvaginal USS + CA-125 6-monthly (pre-RRSO)", "RRSO 45-50yr", "No breast intensification required"],
    "RAD51C": ["Transvaginal USS + CA-125 6-monthly (pre-RRSO)", "RRSO 45-50yr", "No breast intensification required"],
    "RAD51D": ["Transvaginal USS + CA-125 6-monthly (pre-RRSO)", "RRSO 45-50yr", "No breast intensification required"],
    "PALB2": ["Annual MRI breast from 25yr (priority)", "Transvaginal USS + CA-125 (moderate ovarian risk)", "RRSO 45-50yr or breast-driven earlier"],
    "MLH1": ["Annual transvaginal USS + CA-125", "Annual colonoscopy", "Risk-reducing hysterectomy + BSO 35-40yr", "Annual cervical smear", "H. pylori test + eradication", "Aspirin CAPP2"],
    "STK11": ["Annual pelvic USS (SCTAT)", "Annual cervical smear (adenoma malignum)", "GI endoscopy (upper + lower) from 8yr every 2-3yr", "Pancreatic MRI/EUS from 30yr", "Annual breast MRI from 25yr", "Small bowel capsule endoscopy"],
}

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]

_TUMOUR_TYPES = {
    "BRCA1": ["HGSOC", "Fallopian tube primary", "Peritoneal primary", "Concurrent breast+ovarian"],
    "BRCA2": ["HGSOC", "Mixed histology ovarian", "Fallopian tube primary", "Concurrent breast+ovarian"],
    "BRIP1": ["HGSOC", "Fallopian tube primary", "High-grade ovarian NOS"],
    "RAD51C": ["HGSOC", "HRD-positive ovarian", "Fallopian tube primary"],
    "RAD51D": ["HGSOC", "HRD-positive ovarian", "High-grade ovarian NOS"],
    "PALB2": ["HGSOC (ovarian)", "TNBC/HER2-neg breast (dominant)", "Pancreatic"],
    "MLH1": ["Endometrioid ovarian", "Clear cell ovarian", "Concurrent endometrial+ovarian"],
    "STK11": ["SCTAT (sex cord ovarian)", "Adenoma malignum cervix", "Mucinous ovarian"],
}

_VARIANTS_BY_GENE = {
    "BRCA1": ["c.68_69delAG (185delAG)", "c.5266dupC (5382insC)", "c.3756_3759delGTCT", "c.1687C>T (p.Gln563Ter)", "c.5123C>A (p.Ala1708Glu)"],
    "BRCA2": ["c.6174delT (Ashkenazi)", "c.9976A>T (p.Lys3326Ter)", "c.7685delC", "c.3396delA", "c.5682C>G (p.Asn1894Lys)"],
    "BRIP1": ["c.2392C>T (p.Arg798Ter)", "c.1702_1703delAA", "c.2295delT", "c.3113C>T (p.Arg1038Ter)", "c.561+1G>A (splice)"],
    "RAD51C": ["c.905-2A>G (splice)", "c.376C>T (p.Arg126Ter)", "c.837+1G>A (splice)", "c.706C>T (p.Arg236Ter)", "c.416G>A (p.Arg139Gln)"],
    "RAD51D": ["c.694C>T (p.Arg232Ter)", "c.270_271delAA", "c.742C>T (p.Arg248Ter)", "c.577-2A>C (splice)", "c.880C>T (p.Gln294Ter)"],
    "PALB2": ["c.1592delT (p.Leu531Ter)", "c.3256C>T (p.Gln1086Ter)", "c.2257C>T (p.Arg753Ter)", "c.3549C>A (p.Tyr1183Ter)", "c.1053+1G>A (splice)"],
    "MLH1": ["c.1852_1854delAAG", "c.199G>A (p.Val67Met)", "c.677G>T (p.Arg226Ter)", "c.454+1G>A (splice)", "c.117-1G>T (splice)"],
    "STK11": ["c.863_866delATGT", "c.465del (p.Lys155Asnfs)", "c.290C>T (p.Ala97Val)", "c.920-2A>G (splice)", "c.1062_1063insT"],
}


def _make_patients(gene: str, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    g = next(g for g in ATLAS_GENES if g["gene"] == gene)
    tumours = _TUMOUR_TYPES.get(gene, ["Ovarian cancer NOS"])
    variants = _VARIANTS_BY_GENE.get(gene, ["Pathogenic variant"])
    pts = []
    for i in range(n):
        age = rng.randint(35, 72)
        pts.append({
            "patient_id": f"{gene[:3]}-OCA-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_dx": age,
            "tumour_type": rng.choice(tumours),
            "variant": rng.choice(variants),
            "stage": rng.choice(["I", "II", "III", "IV"]),
            "parp_inhibitor": rng.random() < (0.85 if gene in ("BRCA1","BRCA2") else 0.30 if gene in ("RAD51C","RAD51D","BRIP1") else 0.20),
            "platinum_response": rng.random() < (0.80 if gene in ("BRCA1","BRCA2","RAD51C","RAD51D","BRIP1","PALB2") else 0.60),
            "rrso_performed": rng.random() < (0.70 if gene in ("BRCA1","BRCA2") else 0.40),
            "relapse": rng.random() < 0.45,
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
    overall_parp_rate = round(
        100 * sum(p["parp_inhibitor"] for pts in cohorts.values() for p in pts) / total, 1
    )
    overall_platinum_response = round(
        100 * sum(p["platinum_response"] for pts in cohorts.values() for p in pts) / total, 1
    )
    rrso_rate = round(
        100 * sum(p["rrso_performed"] for pts in cohorts.values() for p in pts) / total, 1
    )
    mean_age = round(
        sum(p["age_at_dx"] for pts in cohorts.values() for p in pts) / total, 1
    )
    stage_iii_iv = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["stage"] in ("III","IV")) / total, 1
    )

    return {
        "atlas": "Hereditary-Ovarian-Cancer-Predisposition-Atlas",
        "genes": _GENE_LIST,
        "total_patients": total,
        "gene_counts": gene_counts,
        "seed_range": f"{SEED_BASE}-{SEED_BASE+7}",
        "parp_inhibitor_rate_pct": overall_parp_rate,
        "platinum_response_rate_pct": overall_platinum_response,
        "rrso_rate_pct": rrso_rate,
        "mean_age_at_dx": mean_age,
        "stage_iii_iv_pct": stage_iii_iv,
        "key_facts": [
            "BRCA1: 39-44% lifetime ovarian risk HIGHEST; RRSO 35-40yr MANDATORY; HGSOC fallopian tube origin",
            "BRCA2: 11-17% lifetime; later onset ~55yr vs BRCA1 ~50yr; RRSO 40-45yr",
            "BRIP1/RAD51C/RAD51D: moderate risk 5-10%; NO breast risk — key distinction; RRSO 45-50yr",
            "RAD51D: HIGHEST among paralogs ~10% lifetime; PARP inhibitor sensitivity established",
            "PALB2: ovarian 3-5% moderate; breast 53% DOMINATES — TBCRC048 olaparib 82% ORR",
            "MLH1 Lynch: ovarian 8-13% ENDOMETRIOID/CLEAR CELL — NOT HGSOC; pembrolizumab FDA2017",
            "STK11 PJS: SCTAT PATHOGNOMONIC (almost always benign); ovarian 21%; adenoma malignum cervix",
            "CASCADE TESTING: all first-degree relatives; gene-specific RRSO timing critical",
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
                "ovarian_risk": g["ovarian_risk"],
                "pathognomonic": g["pathognomonic"],
                "key_avoid": g["key_avoid"],
                "key_rule": g["key_rule"],
                "surveillance": g["surveillance"],
                "targeted_rx": g["targeted_rx"],
            },
            "n": len(pts),
            "parp_inhibitor_pct": round(100 * sum(1 for p in pts if p["parp_inhibitor"]) / len(pts), 1),
            "platinum_response_pct": round(100 * sum(1 for p in pts if p["platinum_response"]) / len(pts), 1),
            "rrso_pct": round(100 * sum(1 for p in pts if p["rrso_performed"]) / len(pts), 1),
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
        "atlas": "Hereditary-Ovarian-Cancer-Predisposition-Atlas",
        "definitions": {
            "brca1_hgsoc": (
                "BRCA1 (HBOC1): RING-BRCT HR repair scaffold — ovarian 39-44% lifetime HIGHEST; "
                "HGSOC fallopian tube origin; RRSO 35-40yr MANDATORY (reduces ovarian 80-96%); "
                "OLAPARIB: PAOLA-1 (1st-line + bevacizumab) + SOLO2 (recurrence) FDA approved; "
                "NIRAPARIB PRIMA; RUCAPARIB ARIEL; BEVACIZUMAB GOG218/ICON7"
            ),
            "brca2_later_onset": (
                "BRCA2 (HBOC2 / FANCD1): RAD51 mediator BRC repeats — ovarian 11-17% later onset ~55yr; "
                "RRSO 40-45yr (5yr later than BRCA1 — lower absolute risk); "
                "OLAPARIB SOLO1/SOLO2; biallelic FA-D1: AVOID alkylating agents; "
                "MALE BREAST 6% prostate 24-40% — annual PSA from 40yr"
            ),
            "brip1_fancj_no_breast": (
                "BRIP1/FANCJ: 5'-3' helicase — ovarian 5-8x relative risk; "
                "KEY DISTINCTION: NO significant breast cancer risk (unlike BRCA1/2); "
                "RRSO 45-50yr; FANCJ biallelic: Fanconi Anaemia; "
                "PARPi emerging — olaparib off-label if HRD+ score"
            ),
            "rad51c_fanco_no_breast": (
                "RAD51C/FANCO: BCDX2+CX3 complexes — ovarian ~6% lifetime 5-7x; "
                "KEY DISTINCTION: NO significant breast cancer risk; "
                "FANCO biallelic: Fanconi Anaemia O (rare); RRSO 45-50yr; "
                "HRD scar positive — PARPi sensitivity emerging"
            ),
            "rad51d_highest_paralog": (
                "RAD51D: BCDX2 complex — ovarian ~10% lifetime 7-10x HIGHEST paralog; "
                "KEY DISTINCTION: NO significant breast cancer risk; "
                "PARP INHIBITOR SENSITIVITY: established in vitro + emerging clinical; "
                "RRSO 45-50yr; olaparib off-label HRD+"
            ),
            "palb2_breast_dominates": (
                "PALB2/FANCN: WD40 domain BRCA1-BRCA2 bridge — ovarian 3-5% moderate; "
                "BREAST 53% DOMINATES: TBCRC048 olaparib 82% ORR HIGHEST non-BRCA; "
                "BIALLELIC FA-N: AVOID alkylating agents; "
                "RRSO timing: breast-driven often earlier; annual MRI breast from 25yr"
            ),
            "mlh1_lynch_endometrioid": (
                "MLH1 Lynch type 1: MutLα scaffold — ovarian 8-13% ENDOMETRIOID/CLEAR CELL NOT HGSOC; "
                "MSI-H IHC PATHOGNOMONIC: MLH1+PMS2 loss; somatic methylation vs germline — distinguish; "
                "PEMBROLIZUMAB FDA 2017 tumour-agnostic dMMR/MSI-H; "
                "ASPIRIN CAPP2 600mg/day — 50% risk reduction; H. PYLORI ERADICATION MANDATORY; "
                "PARPi NOT standard (dMMR ≠ HRD pathway)"
            ),
            "stk11_sctat_pathognomonic": (
                "STK11/LKB1: LKB1-STRAD-MO25 AMPK kinase — PJS — ovarian 21% SCTAT PATHOGNOMONIC; "
                "SCTAT BILATERAL CALCIFIED PJS = almost always benign (sporadic SCTAT = SMAD4 = malignant); "
                "ADENOMA MALIGNUM CERVIX PATHOGNOMONIC — annual cervical smear; "
                "MUCOCUTANEOUS MACULES PATHOGNOMONIC PJS diagnosis; "
                "GI ENDOSCOPY FROM 8yr MANDATORY; PANCREATIC MRI/EUS FROM 30yr EARLIEST onset"
            ),
            "cascade_testing": (
                "CASCADE TESTING Hereditary Ovarian Cancer Predisposition: "
                "index case → first-degree relatives; "
                "BRCA1/2: all first-degree regardless of age — RRSO timing critical (35-40yr vs 40-45yr); "
                "BRIP1/RAD51C/RAD51D: first-degree females — ovarian selective moderate risk; "
                "PALB2: first-degree — breast 53% DOMINATES — treat as BRCA2-equivalent surveillance; "
                "MLH1: cascade includes males (colorectal 80%) — colonoscopy + aspirin for all carriers; "
                "STK11: PJS clinical diagnosis (macules) may identify carriers before molecular testing"
            ),
        },
        "key_clinical_distinctions": [
            "BRCA1 vs BRCA2: BRCA1 39-44% earlier onset ~50yr RRSO 35-40yr; BRCA2 11-17% later ~55yr RRSO 40-45yr",
            "BRIP1/RAD51C/RAD51D vs BRCA1/2: NO breast risk — key distinction; RRSO 45-50yr vs 35-45yr",
            "RAD51D HIGHEST paralog ~10%: PARPi sensitivity established; olaparib off-label if HRD+",
            "PALB2 breast DOMINATES management 53%: ovarian 3-5% moderate; TBCRC048 olaparib 82% ORR",
            "MLH1 ovarian: ENDOMETRIOID/CLEAR CELL NOT HGSOC — NO PARPi standard; pembrolizumab dMMR/MSI-H",
            "STK11 SCTAT: BILATERAL CALCIFIED in PJS almost always BENIGN; sporadic SCTAT (SMAD4) = malignant",
            "RRSO TIMING: 35-40yr BRCA1 > 40-45yr BRCA2 > 45-50yr BRIP1/RAD51C/RAD51D/PALB2",
            "Lynch vs HRD: MLH1 Lynch ovarian is dMMR — pembrolizumab; BRCA1/2/RAD51x is HRD — PARPi",
        ],
    }
