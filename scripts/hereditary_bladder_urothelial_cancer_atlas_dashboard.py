#!/usr/bin/env python3
"""Hereditary-Bladder-&-Urothelial-Cancer-Predisposition-Atlas — Complete 8-Gene Reference
MSH2   (MutS Homolog 2; 934aa; 2p21; AD LOF;
         Lynch Syndrome Type 2 / Muir-Torre Syndrome;
         UROTHELIAL CANCER 25% LIFETIME — HIGHEST SINGLE-GENE RISK PATHOGNOMONIC;
         Upper tract urothelial carcinoma (renal pelvis/ureter) + bladder;
         Sebaceous adenomas/carcinomas of face = Muir-Torre PATHOGNOMONIC;
         EPCAM 3-prime deletion silences MSH2 — check EPCAM in all MSH2-negative Lynch;
         seed SEED_BASE+0) ·
MLH1   (MutL Homolog 1; 793aa; 3p22.2; AD LOF;
         Lynch Syndrome Type 1;
         Urothelial cancer 2–4% lifetime; upper tract > bladder in Lynch;
         MLH1 epigenetic silencing in Lynch-LIKE (constitutional MLH1 methylation);
         Aspirin CAPP2 — 50% cancer risk reduction;
         seed SEED_BASE+1) ·
MSH6   (MutS Homolog 6; 1360aa; 2p16.3; AD LOF;
         Lynch Syndrome Type 3;
         Urothelial cancer 7–11% lifetime; MSI-L/MSS in 30% — FALSE NEGATIVE IHC PITFALL;
         HIGHEST Lynch endometrial risk (40-71%); IHC PRIMARY SCREEN;
         seed SEED_BASE+2) ·
BRCA2  (Breast Cancer Gene 2; 3418aa; 13q12.3; AD LOF;
         Hereditary Breast & Ovarian Cancer (HBOC);
         Bladder/urothelial cancer 2–4x elevated risk; cisplatin/platinum-sensitive;
         PARP inhibitor olaparib FDA-approved; Fanconi Anemia type D1 biallelic;
         seed SEED_BASE+3) ·
RB1    (Retinoblastoma Gene 1; 928aa; 13q14.2; AD LOF;
         Hereditary Retinoblastoma;
         Secondary muscle-invasive bladder SCC 15–20x elevated risk after RT;
         BCG CONTRAINDICATED in RB1-null bladder (lacks immune checkpoint); CDK4/6i INACTIVE;
         Bilateral germline until proven otherwise;
         seed SEED_BASE+4) ·
TP53   (Tumour Protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome;
         Bladder/urothelial SCC documented in LFS; p53-abnormal serous subtype;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually;
         seed SEED_BASE+5) ·
HRAS   (Harvey RAS; 189aa; 11p15.5; AD GOF;
         Costello Syndrome;
         Bladder TCC and rhabdomyosarcoma PATHOGNOMONIC in adolescents/young adults;
         HRAS G12S 80% of Costello; papillomata PATHOGNOMONIC; MEK inhibitor trametinib;
         seed SEED_BASE+6) ·
CHEK2  (Checkpoint Kinase 2; 543aa; 22q12.1; AD LOF;
         CHEK2 c.1100delC founder — 1% Northern European;
         Bladder cancer 2–3x moderate risk elevation; intermediate-penetrance gene;
         Breast cancer 2–4x (women); prostate 2x; thyroid elevated;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3166-3173)
"""
import random

SEED_BASE = 3166

ATLAS_GENES = [
    {
        "gene": "MSH2",
        "protein": (
            "MSH2 -- 2p21 Autosomal-Dominant-LOF -- 934aa -- "
            "MutS-Homolog2-105kDa-MutSalpha-MutSbeta-"
            "Lynch-Type2-Urothelial-25pct-HIGHEST-PATHOGNOMONIC-"
            "Muir-Torre-Sebaceous-Face-PATHOGNOMONIC-EPCAM-3prime-Lynch-MMR-OMIM-609309"
        ),
        "locus": "2p21",
        "protein_size": (
            "934 aa / 2p21 MSH2 encodes MutS Homolog 2 (MSH2): "
            "STRUCTURE: "
            "  MutSα heterodimer with MSH6 (105 kDa + 160 kDa): recognises single base mismatches + 1-2 nt IDLs; "
            "  MutSβ heterodimer with MSH3: recognises 2-8 nt insertion/deletion loops; "
            "  ATPase domain (AAA+ family): mismatch-triggered ATP hydrolysis drives strand discrimination; "
            "  Clamp domain: embraces mismatched DNA; "
            "  MSH2 is the obligate heterodimer partner — without MSH2 both MutSα and MutSβ are non-functional; "
            "LYNCH SYNDROME TYPE 2: "
            "  MSH2 LOF → loss of MutSα/MutSβ → microsatellite instability HIGH (MSI-H) → Lynch Syndrome; "
            "  UROTHELIAL CANCER: 25% lifetime risk — HIGHEST of all Lynch syndrome genes for urothelial tract; "
            "  Upper tract urothelial carcinoma (renal pelvis, ureter) predominates over bladder in Lynch; "
            "  Annual urine cytology + upper tract imaging (CT urography) from age 30–35 RECOMMENDED; "
            "EPCAM 3-PRIME DELETION: "
            "  EPCAM (2p21, adjacent to MSH2) 3-prime deletion → read-through transcription → epigenetic silencing of MSH2; "
            "  IHC shows MSH2 LOSS but MSH2 sequencing/MLPA is NORMAL — EPCAM DELETION IS THE PATHOGENIC VARIANT; "
            "  EPCAM deletion restricted to Lynch urothelial risk (lower colorectal risk vs MSH2 coding variant); "
            "  Check EPCAM deletion in ALL cases where MSH2 IHC lost but sequencing negative; "
            "MUIR-TORRE SYNDROME: "
            "  MSH2 (and MLH1) variants → Muir-Torre phenotype; "
            "  Sebaceous adenomas, sebaceous carcinomas, keratoacanthomas of face/scalp/eyelid = PATHOGNOMONIC; "
            "  Any sebaceous tumour in patient <60yr → MSH2/MLH1 germline testing; "
            "TREATMENT: "
            "  MSI-H urothelial cancer: pembrolizumab (PD-1) FDA-approved (MSI-H indication); "
            "  Cisplatin-based gemcitabine-cisplatin for muscle-invasive Lynch bladder; "
            "  Aspirin 600mg daily (CAPP2): ~50% colorectal + likely urothelial risk reduction; "
            "  Annual upper tract imaging + cystoscopy from age 30–35 in MSH2 carriers"
        ),
        "inheritance": "Autosomal Dominant LOF (heterozygous sufficient for Lynch risk; biallelic causes CMMRD)",
        "cancer_risk": "Urothelial 25% lifetime (HIGHEST Lynch gene); CRC 30-40%; Endometrial 40-60%; Ovarian 8-12%; Gastric 5-10%; Sebaceous skin (Muir-Torre)",
        "pathognomonic": "Sebaceous adenomas/carcinomas of face in patient <60yr + ANY cancer = Muir-Torre = MSH2/MLH1 germline PATHOGNOMONIC",
        "surveillance_key": "Annual CT urography + urine cytology from age 30-35; cystoscopy if symptoms; Annual colonoscopy; Gynae surveillance annually from age 30-35",
        "key_distinctions": [
            "MSH2-UROTHELIAL-25PCT-HIGHEST-ALL-LYNCH-GENES",
            "EPCAM-3PRIME-DELETION-SILENCES-MSH2-SEQUENCE-NORMAL",
            "MUIR-TORRE-SEBACEOUS-FACE-PATHOGNOMONIC",
            "MSI-H-PEMBROLIZUMAB-FDA-APPROVED",
            "UPPER-TRACT-PREDOMINANT-OVER-BLADDER",
            "ANNUAL-CT-UROGRAPHY-FROM-AGE-30-35",
        ],
    },
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 793aa -- "
            "MutL-Homolog1-90kDa-MutLalpha-"
            "Lynch-Type1-Urothelial-2-4pct-Constitutional-Methylation-Epigenetic-"
            "Aspirin-CAPP2-50pct-OMIM-120436"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "793 aa / 3p22.2 MLH1 encodes MutL Homolog 1 (MLH1): "
            "STRUCTURE: "
            "  MutLα heterodimer with PMS2 (90 kDa + 96 kDa): nick-directed excision/re-synthesis after MSH recognition; "
            "  N-terminal ATPase domain; C-terminal dimerisation domain; "
            "  Endonuclease in MutLα conferred by PMS2 DQHAX2 motif (MLH1 alone lacks catalytic activity); "
            "LYNCH SYNDROME TYPE 1: "
            "  MLH1 LOF → MutLα non-functional → MSI-H → Lynch Syndrome; "
            "  Urothelial cancer 2–4% lifetime (lower than MSH2 but still meaningful); "
            "  CRC 30-40% lifetime; Endometrial 40-50%; "
            "CONSTITUTIONAL MLH1 METHYLATION (epigenetic Lynch): "
            "  Germline constitutional methylation of MLH1 promoter → equivalent to LOF variant; "
            "  Standard sequencing MISSES this — requires methylation-specific PCR or EPIC methylation array; "
            "  IHC shows MLH1/PMS2 co-loss (same as MLH1 pathogenic variant); "
            "  De novo (not inherited) in ~80% — siblings at lower risk than classic autosomal dominant Lynch; "
            "  Transmissible to offspring if in germline; "
            "SOMATIC BIALLELIC MLH1 METHYLATION (Lynch-LIKE): "
            "  Sporadic colorectal/endometrial cancers with MLH1/PMS2 IHC loss + BRAF V600E: sporadic; "
            "  BRAF V600E in tumour = SPORADIC MLH1 methylation — NOT Lynch; do NOT do germline testing; "
            "SURVEILLANCE: "
            "  Annual colonoscopy from age 25; "
            "  CT urography + urine cytology from age 30-35; "
            "  Aspirin 600mg daily (CAPP2 RCT): 37-50% CRC risk reduction; likely extends to urothelial; "
            "TREATMENT: "
            "  MSI-H urothelial: pembrolizumab; "
            "  Neoadjuvant cisplatin-gemcitabine for MIBC"
        ),
        "inheritance": "Autosomal Dominant LOF (heterozygous); constitutional methylation = epigenetic equivalent",
        "cancer_risk": "Urothelial 2-4% lifetime; CRC 30-40%; Endometrial 40-50%; Ovarian 6-8%; Gastric 5-10%",
        "pathognomonic": "MLH1/PMS2 co-loss on IHC without BRAF V600E = germline MLH1 variant (or constitutional methylation) PATHOGNOMONIC Lynch",
        "surveillance_key": "Annual colonoscopy from 25; CT urography + cytology from 30-35; Aspirin 600mg CAPP2; gynaecological annual from 30-35",
        "key_distinctions": [
            "BRAF-V600E-IN-TUMOUR-EXCLUDES-LYNCH-SPORADIC",
            "CONSTITUTIONAL-METHYLATION-SEQUENCING-MISSES",
            "MLH1-PMS2-CO-LOSS-IHC-PATHOGNOMONIC",
            "ASPIRIN-CAPP2-50PCT-RISK-REDUCTION",
            "UROTHELIAL-LOWER-RISK-THAN-MSH2",
            "CT-UROGRAPHY-FROM-AGE-30-35",
        ],
    },
    {
        "gene": "MSH6",
        "protein": (
            "MSH6 -- 2p16.3 Autosomal-Dominant-LOF -- 1360aa -- "
            "MutS-Homolog6-160kDa-MutSalpha-"
            "Lynch-Type3-Urothelial-7-11pct-MSI-L-30pct-FALSE-NEGATIVE-IHC-PRIMARY-"
            "HIGHEST-Lynch-Endometrial-40-71pct-OMIM-600678"
        ),
        "locus": "2p16.3",
        "protein_size": (
            "934 aa MSH2 / 1360 aa MSH6 form MutSα heterodimer: "
            "STRUCTURE: "
            "  MSH6 (160 kDa) PWWP domain reads H3K36me3 (active chromatin): directs MMR to newly synthesised strand; "
            "  PCNA interaction domain: links MMR to replication fork; "
            "  MSH6 is unique to MutSα (not MutSβ) — specifically recognises single bp mismatches; "
            "LYNCH SYNDROME TYPE 3 (MSH6): "
            "  MSH6 LOF → MutSα non-functional → Lynch Syndrome Type 3; "
            "  UROTHELIAL CANCER 7–11% lifetime — intermediate between MSH2 (25%) and MLH1 (2-4%); "
            "  MSI-L OR MSS in 30% of MSH6-associated tumours — CRITICAL FALSE NEGATIVE PITFALL; "
            "  IHC (MSH6 loss) IS THE PRIMARY SCREEN — PCR-based MSI may miss MSH6 Lynch tumours; "
            "  Always do IHC first for Lynch evaluation; "
            "MSH6 ENDOMETRIAL RISK (HIGHEST LYNCH GENE): "
            "  Endometrial cancer 40–71% lifetime in MSH6 carriers — highest of all Lynch genes for endometrium; "
            "  Premenopausal onset (mean age 46yr vs 62yr sporadic); "
            "  Annual endometrial biopsy from age 30-35; BSO at family completion recommended; "
            "UROTHELIAL SURVEILLANCE: "
            "  Annual CT urography + urine cytology from age 30-35; "
            "  Cystoscopy if haematuria; "
            "MSI TESTING NUANCE: "
            "  MSH6 tumours: 5-mononucleotide repeat MSI panel may show MSS or MSI-L; "
            "  Extended panel or IHC always needed; "
            "  Pembrolizumab active in MSH6 Lynch tumours — confirm with IHC if MSI panel negative"
        ),
        "inheritance": "Autosomal Dominant LOF (heterozygous; incomplete penetrance; late onset vs MSH2/MLH1)",
        "cancer_risk": "Urothelial 7-11% lifetime; Endometrial 40-71% (HIGHEST Lynch); CRC 10-22%; Ovarian 7-11%",
        "pathognomonic": "MSH6 IHC loss in any tumour = Lynch Type 3; MSI-L/MSS does NOT exclude Lynch when MSH6 lost on IHC",
        "surveillance_key": "IHC PRIMARY SCREEN (MSI may miss); Annual CT urography from 30-35; Annual endometrial biopsy from 30-35; BSO at family completion",
        "key_distinctions": [
            "MSI-L-30PCT-FALSE-NEGATIVE-PCR-PITFALL",
            "IHC-PRIMARY-SCREEN-ALWAYS",
            "ENDOMETRIAL-40-71PCT-HIGHEST-LYNCH",
            "UROTHELIAL-7-11PCT-INTERMEDIATE",
            "LATE-ONSET-INCOMPLETE-PENETRANCE",
            "PEMBROLIZUMAB-ACTIVE-CONFIRM-IHC",
        ],
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-"
            "HBOC-Bladder-2-4x-Cisplatin-Platinum-Sensitive-"
            "PARP-Olaparib-FDA-FA-D1-Biallelic-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 13q12.3 BRCA2 encodes BRCA2 (Breast Cancer Gene 2): "
            "STRUCTURE: "
            "  BRC repeats (8 repeats, aa 1002-2082): bind RAD51 monomers; "
            "  C-terminal DNA binding domain: dsDNA/ssDNA binding, PALB2 interaction; "
            "  OB-fold domains; Tower domain (architectural); "
            "  BRCA2 is the HR mediator — loads RAD51 onto ssDNA at resected DSB ends; "
            "HBOC AND BLADDER CANCER: "
            "  BRCA2 LOF → impaired HR → accumulation of DSBs → genomic instability; "
            "  Bladder/urothelial cancer risk 2–4x elevated in BRCA2 carriers; "
            "  Transitional cell carcinoma (TCC) predominates; "
            "  BRCA2 somatic mutations present in ~5% of sporadic bladder cancers; "
            "PLATINUM/CISPLATIN SENSITIVITY: "
            "  BRCA2-deficient tumours hypersensitive to platinum agents (cisplatin, carboplatin, oxaliplatin); "
            "  Cisplatin + gemcitabine = standard MIBC regimen — particularly active in BRCA2; "
            "  Carboplatin if cisplatin ineligible (renal impairment) — still active in HRD; "
            "PARP INHIBITOR — OLAPARIB: "
            "  Olaparib FDA-approved in BRCA1/2 germline carriers with multiple cancer types; "
            "  Synthetic lethality: BRCA2 LOF + PARP inhibition → catastrophic replication fork collapse; "
            "  Bladder: investigational (phase II trials ongoing); "
            "  HRD score (Myriad MyChoice) identifies non-BRCA2 HR-deficient bladder tumours responsive to PARP; "
            "FANCONI ANEMIA TYPE D1 (BIALLELIC BRCA2): "
            "  Biallelic BRCA2 → FA-D1 — most severe FA; "
            "  Medulloblastoma and AML in childhood; "
            "  AVOID RADIATION ABSOLUTELY (biallelic); "
            "SURVEILLANCE: "
            "  No specific bladder surveillance guideline; "
            "  If haematuria → urgent urology referral; "
            "  Annual breast MRI + mammo women from age 25; annual PSA men from 40"
        ),
        "inheritance": "Autosomal Dominant LOF (heterozygous HBOC); biallelic = FA-D1 (severe, childhood onset)",
        "cancer_risk": "Bladder/urothelial 2-4x; Breast (women) 40-65%; Ovarian 10-20%; Pancreatic 5-7%; Male breast 6-8%; Prostate 8-15%",
        "pathognomonic": "Biallelic BRCA2 = FA-D1: medulloblastoma + AML in childhood PATHOGNOMONIC; AVOID RADIATION ABSOLUTELY",
        "surveillance_key": "Haematuria → urgent urology; Annual breast MRI from 25 (women); Annual PSA from 40 (men); Olaparib/platinum if MIBC",
        "key_distinctions": [
            "PLATINUM-CISPLATIN-SENSITIVE-MIBC",
            "PARP-OLAPARIB-FDA-HRD",
            "BLADDER-2-4X-RISK",
            "FA-D1-BIALLELIC-MEDULLOBLASTOMA-AML",
            "HRD-SCORE-BEYOND-BRCA2",
            "AVOID-RADIATION-BIALLELIC-ONLY",
        ],
    },
    {
        "gene": "RB1",
        "protein": (
            "RB1 -- 13q14.2 Autosomal-Dominant-LOF -- 928aa -- "
            "pRB-110kDa-E2F-Repressor-"
            "Hereditary-Retinoblastoma-Secondary-Bladder-SCC-15-20x-RT-"
            "BCG-CONTRAINDICATED-RB1-null-CDK4-6i-INACTIVE-Bilateral-Germline-OMIM-614041"
        ),
        "locus": "13q14.2",
        "protein_size": (
            "928 aa / 13q14.2 RB1 encodes pRB (Retinoblastoma protein): "
            "STRUCTURE: "
            "  Pocket domain (A-box + B-box): binds E2F transcription factors; "
            "  CDK4/CDK6-cyclin D phosphorylation → releases E2F → S-phase entry; "
            "  Hypophosphorylated pRB = cell cycle arrest G1; "
            "  Hyperphosphorylated (CDK4/6) = pRB inactivated = S-phase entry; "
            "HEREDITARY RETINOBLASTOMA: "
            "  Germline RB1 LOF → predisposition to bilateral retinoblastoma (first hit); "
            "  Bilateral retinoblastoma = GERMLINE UNTIL PROVEN OTHERWISE; "
            "  Trilateral (pinealoblastoma): intracranial 5-15% in bilateral germline cases; "
            "SECONDARY BLADDER CANCER: "
            "  After external beam RT for retinoblastoma: muscle-invasive bladder SCC 15–20x elevated; "
            "  RT at 13q field → second RB1 hit in bladder epithelium; "
            "  Onset typically 15–50yr after retinoblastoma treatment; "
            "  Annual urine cytology from age 20 in RB1 germline carriers treated with RT; "
            "BCG CONTRAINDICATED IN RB1-NULL BLADDER: "
            "  BCG immunotherapy requires intact immune recognition of tumour antigens; "
            "  RB1-null bladder tumours: impaired antigen presentation pathway → BCG CONTRAINDICATED; "
            "  Immune checkpoint inhibitors (pembrolizumab, atezolizumab) preferred; "
            "CDK4/6 INHIBITOR INACTIVE: "
            "  CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib) require functional pRB; "
            "  RB1-null = pRB absent → CDK4/6i have NO antiproliferative target → INACTIVE; "
            "  RB1 IHC loss mandatory before CDK4/6i prescription; "
            "SURVEILLANCE: "
            "  Annual urine cytology from age 20 (post-RT cases); "
            "  Full-body CT every 3-5yr from age 15; "
            "  Avoid radiation to lower abdomen/pelvis if at all possible"
        ),
        "inheritance": "Autosomal Dominant LOF (two-hit model: germline + somatic second hit required for retinoblastoma)",
        "cancer_risk": "Retinoblastoma (bilateral PATHOGNOMONIC germline); Secondary bladder SCC 15-20x (post-RT); Osteosarcoma 2000x (post-RT); Soft tissue sarcoma elevated",
        "pathognomonic": "Bilateral retinoblastoma = GERMLINE RB1 until proven; secondary bladder SCC after RT to 13q field in RB1 germline carrier PATHOGNOMONIC",
        "surveillance_key": "Annual urine cytology from age 20 (post-RT); BCG CONTRAINDICATED (use pembrolizumab); CDK4/6i INACTIVE in RB1-null; avoid pelvic RT",
        "key_distinctions": [
            "BILATERAL-RETINOBLASTOMA-GERMLINE-PATHOGNOMONIC",
            "SECONDARY-BLADDER-SCC-15-20X-POST-RT",
            "BCG-CONTRAINDICATED-RB1-NULL",
            "CDK4-6I-INACTIVE-RB1-NULL",
            "ANNUAL-URINE-CYTOLOGY-FROM-AGE-20",
            "AVOID-PELVIC-RT-AT-ALL-COSTS",
        ],
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-"
            "LFS-Bladder-SCC-Urothelial-AVOID-RADIATION-ABSOLUTELY-"
            "WBMRI-Annually-Toronto-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 17p13.1 TP53 encodes p53 tumour suppressor: "
            "STRUCTURE: "
            "  N-terminal transactivation domains (TAD1 aa1-40, TAD2 aa40-67): "
            "    bind MDM2 (ubiquitin E3 ligase, dominant negative inhibitor); "
            "    bind p300/CBP coactivators; "
            "  Proline-rich domain (aa64-92): apoptosis determination; "
            "  DNA-binding domain (DBD, aa94-292): hot-spot mutation cluster (R175H, G245S, R248W, R248Q, R273H, R273C, R282W); "
            "  Tetramerisation domain (TET, aa323-356): p53 active as tetramer; "
            "  C-terminal regulatory domain (aa356-393): Lys-acetylation/methylation fine-tuning; "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  Germline TP53 LOF → LFS (OMIM #151623); "
            "  Core LFS cancers: soft tissue sarcoma, osteosarcoma, premenopausal breast (50%), CPC age <5, adrenocortical; "
            "  Bladder and urothelial SCC documented in LFS — particularly squamous subtype; "
            "  p53-abnormal (diffuse IHC) = serous-type bladder cancer; "
            "AVOID RADIATION ABSOLUTELY: "
            "  TP53 LOF → impaired G1/S checkpoint → radiation-induced second cancers (radiation-induced sarcoma); "
            "  Standard RT dose → accelerated second primaries in RT field; "
            "  LFS bladder: cisplatin-based chemotherapy preferred; radical cystectomy over RT; "
            "  AVOID RADIOTHERAPY TO BLADDER/PELVIS IN LFS — use surgery-first; "
            "WBMRI TORONTO PROTOCOL: "
            "  Annual whole-body MRI (WBMRI, Toronto protocol) is the cornerstone of LFS surveillance; "
            "  Detects sarcomas, adrenocortical, CNS tumours years before symptomatic; "
            "  Add annual brain MRI (CPC risk); "
            "  Annual colonoscopy from 25; annual breast MRI women from 20; "
            "p53 IHC IN BLADDER: "
            "  Diffuse strong p53 IHC (>70% cells) or complete loss = abnormal p53 = usually TP53 mutation; "
            "  Normal p53 IHC = wild-type function (patchy moderate); "
            "  LFS bladder often p53 diffuse IHC"
        ),
        "inheritance": "Autosomal Dominant LOF (de novo 7-20%; AD familial; dominant-negative gain-of-function hotspot alleles add oncogenic function)",
        "cancer_risk": "Bladder/urothelial SCC (LFS); Breast 30-50% by 60yr; Sarcoma (soft tissue + bone); CPC age<5 PATHOGNOMONIC; Adrenocortical carcinoma (ACC); Brain tumours",
        "pathognomonic": "Adrenocortical carcinoma (ACC) in child <4yr = LFS TP53 PATHOGNOMONIC (60-80% germline TP53); CPC age <5 = LFS PATHOGNOMONIC",
        "surveillance_key": "WBMRI annually (Toronto); brain MRI annually; AVOID RADIATION ABSOLUTELY; surgery-first for bladder; radical cystectomy not RT",
        "key_distinctions": [
            "AVOID-RADIATION-ABSOLUTELY",
            "WBMRI-TORONTO-ANNUALLY",
            "RADICAL-CYSTECTOMY-NOT-RT",
            "p53-DIFFUSE-IHC-LFS-BLADDER",
            "ACC-AGE-4-PATHOGNOMONIC-LFS",
            "CPC-AGE-5-PATHOGNOMONIC-LFS",
        ],
    },
    {
        "gene": "HRAS",
        "protein": (
            "HRAS -- 11p15.5 Autosomal-Dominant-GOF -- 189aa -- "
            "p21ras-21kDa-GTPase-RAS-MAPK-"
            "Costello-Syndrome-Bladder-TCC-RMS-PATHOGNOMONIC-Adolescent-"
            "G12S-80pct-Papillomata-PATHOGNOMONIC-MEK-Trametinib-OMIM-218040"
        ),
        "locus": "11p15.5",
        "protein_size": (
            "189 aa / 11p15.5 HRAS encodes Harvey RAS p21: "
            "STRUCTURE: "
            "  p21 RAS GTPase (21 kDa, smallest RAS family member); "
            "  G1-G5 nucleotide-binding loops; Switch I (aa30-40) + Switch II (aa60-76): GTP-dependent conformational change; "
            "  CAAX motif (C186): farnesylation → membrane anchoring; "
            "  Intrinsic GTPase activity (very low) — accelerated by GAPs (RasGAP, NF1); "
            "  GEFs (SOS1, SOS2) exchange GDP → GTP → ACTIVE; GAPs hydrolyse GTP → GDP → INACTIVE; "
            "COSTELLO SYNDROME (HRAS GOF): "
            "  Germline de-novo GOF HRAS variants → Costello syndrome (OMIM #218040); "
            "  G12S accounts for 80% of Costello syndrome; G12A, G12V, G13C also found; "
            "  RASopathy: HRAS GOF → hyperactivated RAS-MAPK-ERK pathway; "
            "  FACIAL FEATURES: coarse face, full cheeks, macrocephaly, relative macroglossia — Costello PATHOGNOMONIC; "
            "  Loose skin (hyperextensible), papillomata of face/perianal PATHOGNOMONIC; "
            "  Hypertrophic cardiomyopathy (60%), pulmonary valve stenosis; "
            "  Intellectual disability, feeding difficulties, hypotonia; "
            "BLADDER TUMOURS — PATHOGNOMONIC: "
            "  Bladder TCC (transitional cell carcinoma): 4% lifetime — occurs in adolescents/young adults (median 10yr); "
            "  Rhabdomyosarcoma (RMS) of bladder — embryonal RMS PATHOGNOMONIC Costello; "
            "  HRAS GOF = somatic driver of 10% sporadic bladder cancers too; "
            "  Annual renal/bladder USS from age 5; urine cytology from age 8; "
            "PAPILLOMATA PATHOGNOMONIC: "
            "  Benign papillomata of nares, perianal region, perioral = PATHOGNOMONIC Costello; "
            "  Distinct from neurofibromas (NF1) — flat-topped verrucous papillomata; "
            "MEK INHIBITOR — TRAMETINIB: "
            "  HRAS GOF → MEK activation (KRAS/NRAS/HRAS all signal through RAF-MEK-ERK); "
            "  MEK inhibitor trametinib active in HRAS-driven malignancies; "
            "  Also active in salivary HRAS-mutant undifferentiated carcinomas; "
            "  Lonafarnib (farnesyl transferase inhibitor) investigational in Costello; "
            "DIAGNOSIS: "
            "  In any child with papillomata + coarse features + cardiomyopathy → HRAS germline panel; "
            "  Distinguish Noonan (PTPN11, SOS1, RAF1, KRAS, RIT1) vs Costello (HRAS)"
        ),
        "inheritance": "Autosomal Dominant GOF (virtually all de novo; recurrence risk ~1% if parental somatic mosaicism excluded)",
        "cancer_risk": "Bladder TCC/RMS 4% (adolescent PATHOGNOMONIC); Rhabdomyosarcoma (embryonal) elevated; Neuroblastoma; Skin papillomata (benign but PATHOGNOMONIC)",
        "pathognomonic": "Bladder TCC/RMS in adolescent + papillomata + coarse facies + cardiomyopathy = Costello HRAS PATHOGNOMONIC; perianal papillomata PATHOGNOMONIC",
        "surveillance_key": "Annual renal/bladder USS from age 5; urine cytology from age 8; MEK inhibitor trametinib for HRAS-driven malignancy; annual echo (HCM)",
        "key_distinctions": [
            "BLADDER-TCC-RMS-ADOLESCENT-PATHOGNOMONIC",
            "PAPILLOMATA-PERIANAL-FACE-PATHOGNOMONIC",
            "G12S-80PCT-COSTELLO",
            "MEK-TRAMETINIB-ACTIVE",
            "HRAS-10PCT-SPORADIC-BLADDER-SOMATIC",
            "DDX-NOONAN-PTPN11-SOS1-VS-COSTELLO-HRAS",
        ],
    },
    {
        "gene": "CHEK2",
        "protein": (
            "CHEK2 -- 22q12.1 Autosomal-Dominant-LOF -- 543aa -- "
            "CHK2-60kDa-Checkpoint-Kinase-"
            "c.1100delC-1pct-Northern-European-Bladder-2-3x-Moderate-"
            "Breast-2-4x-Prostate-2x-Intermediate-Penetrance-OMIM-604373"
        ),
        "locus": "22q12.1",
        "protein_size": (
            "543 aa / 22q12.1 CHEK2 encodes CHK2 (Checkpoint kinase 2): "
            "STRUCTURE: "
            "  SQ/TQ cluster domain (SCD): ATM phosphorylation target after DSB (T68); "
            "  FHA domain (fork head-associated): phospho-threonine reader; dimerisation; "
            "  Kinase domain (KD): serine/threonine kinase activity; "
            "  ATM → T68-CHEK2 → auto-phosphorylation → dimer release → active monomer; "
            "  Active CHK2 phosphorylates CDC25A/C (S-phase + mitotic checkpoint), BRCA1, MDM2, p53, PML; "
            "CHEK2 c.1100delC (FOUNDER VARIANT): "
            "  Northern European founder variant (~1% carrier frequency in Scandinavia, Netherlands, Poland); "
            "  5382insC (BRCA1-like founder in Ashkenazi); "
            "  Moderate penetrance intermediate-risk gene — NOT high-penetrance like BRCA1/2; "
            "BLADDER CANCER RISK: "
            "  CHEK2 c.1100delC: bladder cancer 2–3x moderate elevation; "
            "  Mechanism: impaired ATM-CHK2 checkpoint → endogenous DSB accumulation → urothelial instability; "
            "  No dedicated bladder surveillance guideline for CHEK2 alone (risk insufficient); "
            "  If CHEK2 + family history bladder → discuss annual urine cytology; "
            "BREAST, PROSTATE, THYROID RISKS: "
            "  Women: breast cancer 2–4x (lifetime ~20-25%); annual breast MRI from 35-40; "
            "  Men: prostate cancer ~2x; PSA from age 40-45; "
            "  Thyroid cancer elevated (~3x); annual thyroid USS; "
            "CLINICAL CLASSIFICATION: "
            "  CHEK2 = INTERMEDIATE penetrance (vs HIGH penetrance BRCA1/BRCA2); "
            "  NOT Lynch, NOT HBOC high-penetrance; "
            "  Risk modification by polygenic risk score (PRS) important for management; "
            "  CHEK2 + BRCA2 in same family = independent risk (not redundant); "
            "MANAGEMENT: "
            "  Breast MRI from age 35-40 (women); "
            "  No olaparib indication without additional HRD evidence; "
            "  Discuss aspirin for colorectal modest risk reduction; "
            "  Family cascade testing — particularly for breast/prostate risk counselling"
        ),
        "inheritance": "Autosomal Dominant LOF (intermediate penetrance; c.1100delC founder Northern European; I157T common variant moderate risk)",
        "cancer_risk": "Bladder 2-3x moderate; Breast (women) 2-4x (~20-25% lifetime); Prostate 2x; Thyroid ~3x; CRC modestly elevated",
        "pathognomonic": "No single CHEK2 PATHOGNOMONIC feature; context: CHEK2 c.1100delC + family history bladder + ≥1 other CHEK2-associated cancer = CHEK2 syndrome",
        "surveillance_key": "Annual breast MRI from 35-40 (women); PSA from 40-45 (men); Annual thyroid USS; Discuss bladder cytology if family history; no olaparib alone",
        "key_distinctions": [
            "C1100DELC-1PCT-NORTHERN-EUROPEAN-FOUNDER",
            "BLADDER-2-3X-MODERATE-INTERMEDIATE-PENETRANCE",
            "NOT-HIGH-PENETRANCE-NOT-BRCA-EQUIVALENT",
            "PRS-MODIFIES-RISK-MANAGEMENT",
            "BREAST-MRI-FROM-35-40-WOMEN",
            "NO-OLAPARIB-WITHOUT-HRD-EVIDENCE",
        ],
    },
]


# ── Clinical thresholds and surveillance ages ──────────────────────────────────
SURVEILLANCE_START = {
    "MSH2":  {"urothelial": 30, "colonoscopy": 25, "gynaecology": 30},
    "MLH1":  {"urothelial": 30, "colonoscopy": 25, "gynaecology": 30},
    "MSH6":  {"urothelial": 30, "colonoscopy": 25, "gynaecology": 30},
    "BRCA2": {"breast_mri": 25, "psa": 40, "urothelial": None},
    "RB1":   {"urothelial_cytology": 20, "full_body_ct": 15},
    "TP53":  {"wbmri": 18, "brain_mri": 18, "colonoscopy": 25, "breast_mri": 20},
    "HRAS":  {"bladder_uss": 5, "urine_cytology": 8, "echo": 1},
    "CHEK2": {"breast_mri": 35, "psa": 40, "thyroid_uss": 25},
}

CANCER_RISKS = {
    "MSH2":  {"urothelial_pct": 25, "crc_pct": 35, "endometrial_pct": 50, "sebaceous_skin": True},
    "MLH1":  {"urothelial_pct": 3,  "crc_pct": 35, "endometrial_pct": 45, "braf_v600e_excludes": True},
    "MSH6":  {"urothelial_pct": 9,  "crc_pct": 16, "endometrial_pct": 56, "msi_l_pitfall": True},
    "BRCA2": {"urothelial_rr": 3.0, "breast_pct": 52, "ovarian_pct": 15, "parp_eligible": True},
    "RB1":   {"secondary_bladder_rr": 17.5, "bcg_contraindicated": True, "cdk46i_inactive": True},
    "TP53":  {"avoid_radiation": True, "wbmri_toronto": True, "acc_pathognomonic": True},
    "HRAS":  {"bladder_tcc_pct": 4, "rms_bladder": True, "papillomata_pathognomonic": True},
    "CHEK2": {"bladder_rr": 2.5, "breast_rr": 3.0, "intermediate_penetrance": True},
}


def _rng(seed):
    rng = random.Random(seed)
    return rng


def _patient_row(rng, gene_info, gene_idx):
    """Generate one simulated patient for a given gene cohort."""
    gene = gene_info["gene"]

    age_onset_ranges = {
        "MSH2":  (28, 68), "MLH1": (35, 72), "MSH6": (40, 75),
        "BRCA2": (32, 70), "RB1":  (22, 62), "TP53": (25, 65),
        "HRAS":  (4, 24),  "CHEK2":(38, 78),
    }
    cancer_type_map = {
        "MSH2":  ["Upper-tract-urothelial", "Bladder-TCC", "CRC", "Endometrial", "Sebaceous-carcinoma"],
        "MLH1":  ["Upper-tract-urothelial", "Bladder-TCC", "CRC", "Endometrial", "Gastric"],
        "MSH6":  ["Bladder-TCC", "Upper-tract-urothelial", "Endometrial", "CRC", "Ovarian"],
        "BRCA2": ["Bladder-TCC", "Breast", "Ovarian", "Pancreatic", "Prostate"],
        "RB1":   ["Bilateral-retinoblastoma", "Secondary-bladder-SCC", "Osteosarcoma", "Soft-tissue-sarcoma"],
        "TP53":  ["Bladder-SCC", "Soft-tissue-sarcoma", "Osteosarcoma", "Adrenocortical-Ca", "Breast"],
        "HRAS":  ["Bladder-TCC", "Rhabdomyosarcoma-bladder", "Papillomata-skin", "Neuroblastoma"],
        "CHEK2": ["Bladder-TCC", "Breast", "Prostate", "Thyroid-carcinoma", "CRC"],
    }
    variant_types = {
        "MSH2":  ["c.1906G>C p.Ala636Pro", "del exon1-6 EPCAM3'", "c.388_389del p.Asn130fs", "c.942+3A>T splice"],
        "MLH1":  ["c.1852_1853del p.Lys618fs", "c.676C>T p.Arg226*", "c.350C>T p.Thr117Met", "c.1039-8T>A splice", "c.793-2A>G splice"],
        "MSH6":  ["c.3959_3962del p.Thr1320fs", "c.3261del p.Cys1088fs", "c.1A>G p.Met1Val", "c.3557G>A p.Gly1186Glu"],
        "BRCA2": ["c.5946del p.Ser1982fs", "c.7617+1G>A splice", "c.3847_3848del p.Val1283fs", "c.8487+1G>A splice"],
        "RB1":   ["c.958C>T p.Arg320*", "del 13q14.2", "c.1654C>T p.Arg552*", "c.2501G>A p.Arg834Gln"],
        "TP53":  ["c.817C>T p.Arg273Cys", "c.524G>A p.Arg175His", "c.742C>T p.Arg248Trp", "c.818G>A p.Arg273His"],
        "HRAS":  ["c.34G>A p.Gly12Ser", "c.35G>C p.Gly12Ala", "c.37G>T p.Gly13Cys"],
        "CHEK2": ["c.1100del p.Thr367fs", "c.470T>C p.Ile157Thr", "c.592+1G>A splice", "c.1214G>C p.Ser405Thr"],
    }
    treatment_map = {
        "MSH2":  ["Pembrolizumab-MSI-H", "Cisplatin-Gemcitabine-MIBC", "Aspirin-600mg-CAPP2", "Annual-CT-urography"],
        "MLH1":  ["Pembrolizumab-MSI-H", "Cisplatin-Gemcitabine-MIBC", "Aspirin-600mg-CAPP2", "Annual-CT-urography"],
        "MSH6":  ["Pembrolizumab-IHC-confirmed", "Cisplatin-Gemcitabine-MIBC", "Aspirin-600mg", "Annual-CT-urography"],
        "BRCA2": ["Cisplatin-Gemcitabine-MIBC", "Olaparib-maintenance-PARP", "Radical-cystectomy", "Annual-breast-MRI"],
        "RB1":   ["Radical-cystectomy", "Pembrolizumab-not-BCG", "Atezolizumab-checkpoint", "Avoid-pelvic-RT"],
        "TP53":  ["Radical-cystectomy-surgery-first", "Cisplatin-based-chemo", "WBMRI-Toronto-annual", "Avoid-RT-absolutely"],
        "HRAS":  ["Trametinib-MEK-inhibitor", "Radical-cystectomy", "Annual-USS-bladder", "Lonafarnib-investigational"],
        "CHEK2": ["Cisplatin-Gemcitabine-MIBC", "Breast-MRI-annual", "PSA-annual-40plus", "Thyroid-USS-annual"],
    }

    lo, hi = age_onset_ranges[gene]
    age = rng.randint(lo, hi)
    cancer = rng.choice(cancer_type_map[gene])
    variant = rng.choice(variant_types[gene])
    treatment = rng.choice(treatment_map[gene])
    fam_hist = rng.choice(["yes", "no", "unknown"])
    staging = rng.choice(["Ta", "T1", "T2", "T3", "T4", "M1", "pT2N0M0", "pT3N1M0"])
    return {
        "age_at_diagnosis": age,
        "cancer_type": cancer,
        "variant": variant,
        "treatment": treatment,
        "family_history": fam_hist,
        "staging": staging,
        "gene": gene,
    }


def generate_overview() -> dict:
    return {
        "atlas": "Hereditary-Bladder-Urothelial-Cancer-Predisposition-Atlas",
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": 320,
        "genes": [g["gene"] for g in ATLAS_GENES],
        "gene_count": 8,
        "cohort_per_gene": 40,
        "inheritance_modes": {g["gene"]: g["inheritance"] for g in ATLAS_GENES},
        "cancer_risks": CANCER_RISKS,
        "pathognomonic_features": {
            "MSH2":  "Sebaceous adenomas/carcinomas of face/scalp + ANY cancer patient <60yr = Muir-Torre = MSH2/MLH1 PATHOGNOMONIC; urothelial 25% HIGHEST",
            "MLH1":  "MLH1/PMS2 co-loss on IHC without BRAF V600E = Lynch Type 1 PATHOGNOMONIC; BRAF V600E excludes Lynch",
            "MSH6":  "MSH6 IHC loss = Lynch Type 3; MSI-L/MSS in 30% — IHC PRIMARY not PCR; endometrial 40-71% HIGHEST Lynch",
            "BRCA2": "Biallelic BRCA2 = FA-D1 medulloblastoma + AML childhood PATHOGNOMONIC; AVOID RADIATION biallelic",
            "RB1":   "Bilateral retinoblastoma = GERMLINE PATHOGNOMONIC; secondary bladder SCC post-RT 15-20x; BCG CONTRAINDICATED",
            "TP53":  "ACC age<4 = LFS PATHOGNOMONIC; CPC age<5 PATHOGNOMONIC; AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually",
            "HRAS":  "Perianal/facial papillomata + coarse facies + HCM + bladder TCC in adolescent = Costello HRAS PATHOGNOMONIC",
            "CHEK2": "c.1100delC 1% Northern European founder; bladder 2-3x moderate; breast MRI from 35-40; intermediate penetrance",
        },
        "key_distinctions": {
            "MSH2":  "UROTHELIAL-25PCT-HIGHEST / EPCAM-3PRIME-DELETION / MUIR-TORRE-PATHOGNOMONIC / PEMBROLIZUMAB-MSI-H",
            "MLH1":  "BRAF-V600E-EXCLUDES-LYNCH / CONSTITUTIONAL-METHYLATION-MISSES-SEQUENCING / ASPIRIN-CAPP2-50PCT",
            "MSH6":  "MSI-L-FALSE-NEGATIVE-30PCT / IHC-PRIMARY-ALWAYS / ENDOMETRIAL-HIGHEST-LYNCH / LATE-ONSET",
            "BRCA2": "PLATINUM-CISPLATIN-SENSITIVE / PARP-OLAPARIB-FDA / FA-D1-BIALLELIC / HRD-SCORE",
            "RB1":   "BCG-CONTRAINDICATED / CDK4-6I-INACTIVE / SECONDARY-BLADDER-SCC-POST-RT / AVOID-PELVIC-RT",
            "TP53":  "AVOID-RADIATION-ABSOLUTELY / WBMRI-TORONTO / SURGERY-FIRST / RADICAL-CYSTECTOMY-NOT-RT",
            "HRAS":  "G12S-80PCT-COSTELLO / PAPILLOMATA-PATHOGNOMONIC / MEK-TRAMETINIB / ANNUAL-USS-FROM-AGE-5",
            "CHEK2": "C1100DELC-FOUNDER / INTERMEDIATE-PENETRANCE / PRS-MODIFIES-RISK / BREAST-MRI-FROM-35",
        },
        "surveillance_start_ages": SURVEILLANCE_START,
        "clinical_pearls": [
            "MSH2 carries the HIGHEST urothelial cancer risk of all Lynch genes (25% lifetime); annual CT urography + urine cytology from age 30-35",
            "MSH6 Lynch tumours show MSI-L or MSS in 30% — always use IHC (MSH6 protein loss) as PRIMARY screen; PCR-based MSI may falsely reassure",
            "RB1-null bladder tumours: BCG CONTRAINDICATED (lacks antigen presentation target); use pembrolizumab or atezolizumab instead",
            "TP53/LFS bladder: AVOID RADIATION ABSOLUTELY — radical cystectomy + cisplatin chemotherapy preferred; RT accelerates second primaries",
            "HRAS Costello: perianal papillomata + coarse facies + HCM → bladder USS from age 5; trametinib (MEK) active in HRAS-driven TCC/RMS",
            "EPCAM 3-prime deletion silences MSH2 but MSH2 sequencing/MLPA is NORMAL — always test EPCAM in MSH2-IHC-loss but sequence-negative cases",
            "BRCA2 bladder: cisplatin-gemcitabine particularly active (HR deficiency); consider PARP inhibitor olaparib maintenance (investigational for bladder)",
            "CHEK2 c.1100delC 1% Northern European; bladder risk 2-3x moderate — intermediate penetrance; PRS modifies management threshold",
        ],
        "genes_detail": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "inheritance": g["inheritance"],
                "cancer_risk_summary": g["cancer_risk"],
                "pathognomonic": g["pathognomonic"],
                "surveillance_key": g["surveillance_key"],
                "protein_summary": g["protein"][:120],
                "key_distinctions": g["key_distinctions"],
            }
            for g in ATLAS_GENES
        ],
    }


def generate_breakdown() -> dict:
    rows = []
    for i, gi in enumerate(ATLAS_GENES):
        rng = _rng(SEED_BASE + i)
        patients = [_patient_row(rng, gi, i) for _ in range(40)]
        ages = [p["age_at_diagnosis"] for p in patients]
        bladder_n = sum(
            1 for p in patients
            if "bladder" in p["cancer_type"].lower()
            or "urothelial" in p["cancer_type"].lower()
            or "tcc" in p["cancer_type"].lower()
            or "rms" in p["cancer_type"].lower()
        )
        rows.append({
            "gene":               gi["gene"],
            "locus":              gi["locus"],
            "n":                  40,
            "mean_age_onset":     round(sum(ages) / len(ages), 1),
            "min_age":            min(ages),
            "max_age":            max(ages),
            "bladder_urothelial_n": bladder_n,
            "bladder_urothelial_pct": round(bladder_n / 40 * 100, 1),
            "seed":               SEED_BASE + i,
            "cancer_risk":        gi["cancer_risk"],
            "key_distinctions":   gi["key_distinctions"],
            "patients":           patients,
        })
    return {
        "atlas":        "Hereditary-Bladder-Urothelial-Cancer-Predisposition-Atlas",
        "seed_range":   f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": 320,
        "breakdown":    rows,
    }


def generate_definitions() -> dict:
    defs = [
        {
            "term": "MSH2 / MUIR-TORRE / LYNCH TYPE 2",
            "definition": (
                "MSH2 (MutS Homolog 2) — 934aa / 105 kDa / 2p21 / AD LOF\n"
                "Lynch Syndrome Type 2 — HIGHEST urothelial cancer risk of all Lynch genes.\n\n"
                "UROTHELIAL RISK 25% LIFETIME:\n"
                "  Upper tract urothelial carcinoma (UTUC) renal pelvis/ureter > bladder TCC in Lynch.\n"
                "  Annual CT urography + urine cytology from age 30-35.\n"
                "  Cystoscopy annually or if haematuria.\n\n"
                "EPCAM 3-PRIME DELETION:\n"
                "  EPCAM gene (2p21, adjacent) 3-prime deletion → read-through into MSH2 → epigenetic silencing.\n"
                "  IHC: MSH2 lost. Sequencing/MLPA of MSH2: NORMAL. Diagnosis = EPCAM deletion.\n"
                "  Check EPCAM in ALL IHC-positive (MSH2 loss) but sequence-negative cases.\n"
                "  EPCAM deletion: primarily colorectal + urothelial risk (lower endometrial vs MSH2 coding variants).\n\n"
                "MUIR-TORRE SYNDROME (MSH2/MLH1):\n"
                "  Sebaceous adenomas, sebaceous carcinomas, keratoacanthomas of face/scalp/eyelid.\n"
                "  ANY sebaceous neoplasm in patient <60yr = Muir-Torre = MSH2/MLH1 germline testing MANDATORY.\n"
                "  Pathognomonic association: sebaceous tumour + visceral Lynch cancer.\n\n"
                "TREATMENT / PHARMACOLOGICAL PRECISION:\n"
                "  MSI-H Lynch urothelial: pembrolizumab (PD-1) FDA-approved (MSI-H tumour-agnostic indication).\n"
                "  Cisplatin-gemcitabine: standard MIBC; active regardless of MSH2 status.\n"
                "  Aspirin 600mg/day (CAPP2 RCT): ~50% CRC risk reduction; extrapolated to urothelial.\n"
                "  Radical cystectomy for MIBC; ureteroscopy for low-grade UTUC if feasible.\n"
            ),
        },
        {
            "term": "MLH1 / LYNCH TYPE 1 / CONSTITUTIONAL METHYLATION",
            "definition": (
                "MLH1 (MutL Homolog 1) — 793aa / 90 kDa / 3p22.2 / AD LOF\n"
                "Lynch Syndrome Type 1 — urothelial cancer 2-4% lifetime.\n\n"
                "BRAF V600E EXCLUDES LYNCH:\n"
                "  BRAF V600E in MLH1/PMS2-lost tumour = SPORADIC MLH1 methylation.\n"
                "  Do NOT perform germline testing if tumour BRAF V600E detected.\n"
                "  BRAF V600E pathway: sporadic hypermethylation of MLH1 → MSI-H → sporadic Lynch-like CRC.\n\n"
                "CONSTITUTIONAL MLH1 METHYLATION (epigenetic Lynch):\n"
                "  Germline constitutional methylation of MLH1 promoter CpGs.\n"
                "  Indistinguishable from LOF variant by IHC (MLH1/PMS2 co-loss).\n"
                "  Standard sequencing/MLPA MISSES this — requires methylation-specific PCR or EPIC array.\n"
                "  ~80% de novo (new in that individual); transmissible to offspring.\n\n"
                "PHARMACOLOGICAL PRECISION:\n"
                "  MSI-H Lynch urothelial: pembrolizumab.\n"
                "  Aspirin 600mg/day (CAPP2): 37-50% CRC/Lynch cancer risk reduction.\n"
                "  Annual colonoscopy from 25; annual CT urography + cytology from 30-35.\n"
            ),
        },
        {
            "term": "MSH6 / LYNCH TYPE 3 / MSI-L FALSE NEGATIVE PITFALL",
            "definition": (
                "MSH6 (MutS Homolog 6) — 1360aa / 160 kDa / 2p16.3 / AD LOF\n"
                "Lynch Syndrome Type 3 — urothelial 7-11%; endometrial 40-71% HIGHEST Lynch.\n\n"
                "MSI-L FALSE NEGATIVE PITFALL:\n"
                "  MSH6 Lynch tumours show MSI-L or MSS (microsatellite stable) in 30%.\n"
                "  Standard PCR 5-mononucleotide repeat panel may report MSS/MSI-L even in Lynch.\n"
                "  IHC (MSH6 protein loss) IS THE PRIMARY SCREEN — never rely on PCR alone.\n"
                "  Extended MSI panel (more loci) or IHC always needed.\n\n"
                "ENDOMETRIAL RISK (HIGHEST LYNCH GENE 40-71%):\n"
                "  Premenopausal onset (mean 46yr vs 62yr sporadic).\n"
                "  Annual endometrial biopsy from age 30-35; BSO at family completion.\n\n"
                "PHARMACOLOGICAL PRECISION:\n"
                "  Pembrolizumab active but CONFIRM with IHC (MSH6 loss) if PCR-based MSI negative.\n"
                "  Late onset + incomplete penetrance vs MSH2/MLH1.\n"
            ),
        },
        {
            "term": "BRCA2 / HBOC / BLADDER / PARP OLAPARIB",
            "definition": (
                "BRCA2 (Breast Cancer Gene 2) — 3418aa / 384 kDa / 13q12.3 / AD LOF\n"
                "HBOC — bladder/urothelial 2-4x elevated risk.\n\n"
                "PLATINUM SENSITIVITY:\n"
                "  BRCA2 LOF → HR deficiency → platinum agent hypersensitivity.\n"
                "  Cisplatin + gemcitabine (standard MIBC) particularly active in BRCA2 carriers.\n"
                "  HRD score (Myriad MyChoice) identifies non-BRCA2 HR-deficient bladder tumours.\n\n"
                "PARP INHIBITOR (OLAPARIB):\n"
                "  Olaparib FDA-approved (germline BRCA1/2, multiple tumour types).\n"
                "  Bladder: phase II trials ongoing — not yet standard of care for bladder.\n"
                "  Synthetic lethality: BRCA2 LOF + PARP inhibition → catastrophic DSB accumulation.\n\n"
                "FANCONI ANEMIA TYPE D1 (BIALLELIC):\n"
                "  Biallelic BRCA2 = FA-D1: medulloblastoma + AML in childhood PATHOGNOMONIC.\n"
                "  AVOID RADIATION ABSOLUTELY (biallelic — lethal radiosensitivity).\n"
                "  Heterozygous HBOC carriers: radiation risk modestly elevated but NOT absolutely contraindicated.\n"
            ),
        },
        {
            "term": "RB1 / RETINOBLASTOMA / SECONDARY BLADDER / BCG-CONTRAINDICATED / CDK4-6I-INACTIVE",
            "definition": (
                "RB1 (Retinoblastoma Protein) — 928aa / 110 kDa / 13q14.2 / AD LOF\n"
                "Hereditary Retinoblastoma + secondary bladder SCC 15-20x post-RT.\n\n"
                "BILATERAL RETINOBLASTOMA = GERMLINE PATHOGNOMONIC:\n"
                "  Bilateral retinoblastoma: germline RB1 until proven otherwise.\n"
                "  Unilateral: 15% germline; bilateral: >95% germline.\n"
                "  Trilateral (pinealoblastoma): 5-15% bilateral germline cases (intracranial).\n\n"
                "SECONDARY BLADDER SCC (POST-RT):\n"
                "  External beam RT to 13q field → second RB1 hit in bladder epithelium → muscle-invasive bladder SCC.\n"
                "  Risk 15-20x; onset 15-50yr after retinoblastoma treatment.\n"
                "  Annual urine cytology from age 20 in post-RT RB1 germline carriers.\n\n"
                "BCG CONTRAINDICATED IN RB1-NULL BLADDER:\n"
                "  BCG requires intact immune antigen presentation (E2F-RB1 pathway active); RB1 null = pathway absent.\n"
                "  BCG fails in RB1-null; use immune checkpoint inhibitors (pembrolizumab, atezolizumab).\n\n"
                "CDK4/6 INHIBITORS INACTIVE IN RB1-NULL:\n"
                "  CDK4/6i (palbociclib, ribociclib, abemaciclib) block CDK4/6 → keep RB1 hypophosphorylated → arrest.\n"
                "  RB1 null = pRB protein absent → CDK4/6i have no target → INACTIVE.\n"
                "  RB1 IHC MANDATORY before CDK4/6i prescription.\n"
            ),
        },
        {
            "term": "TP53 / LFS / AVOID RADIATION ABSOLUTELY / WBMRI TORONTO",
            "definition": (
                "TP53 (Tumour Protein p53) — 393aa / 43 kDa / 17p13.1 / AD LOF\n"
                "Li-Fraumeni Syndrome — bladder/urothelial SCC; AVOID RADIATION ABSOLUTELY.\n\n"
                "AVOID RADIATION ABSOLUTELY:\n"
                "  TP53 LOF → impaired G1/S checkpoint → ionising radiation → accelerated second primaries.\n"
                "  RT field → radiation-induced sarcoma/second cancers within 5-10yr.\n"
                "  LFS bladder: RADICAL CYSTECTOMY preferred over radiotherapy.\n"
                "  If RT unavoidable: lower dose, tightly conformal; document decision carefully.\n\n"
                "WBMRI TORONTO PROTOCOL:\n"
                "  Annual whole-body MRI (Toronto protocol) — cornerstone LFS surveillance.\n"
                "  Detects sarcomas, ACC, CNS tumours years before symptomatic.\n"
                "  Annual brain MRI (CPC age <5 = PATHOGNOMONIC LFS).\n"
                "  Annual breast MRI from age 20 (women); annual colonoscopy from 25.\n\n"
                "p53 IHC IN BLADDER:\n"
                "  Diffuse strong p53 IHC (>70% cells) = TP53 gain-of-function hotspot variant.\n"
                "  Complete loss = null/truncating variant.\n"
                "  Patchy moderate = wild-type p53 pattern.\n"
            ),
        },
        {
            "term": "HRAS / COSTELLO SYNDROME / BLADDER TCC-RMS / PAPILLOMATA PATHOGNOMONIC",
            "definition": (
                "HRAS (Harvey RAS) — 189aa / 21 kDa / 11p15.5 / AD GOF\n"
                "Costello Syndrome RASopathy — bladder TCC/RMS PATHOGNOMONIC in adolescents.\n\n"
                "COSTELLO SYNDROME (HRAS GOF):\n"
                "  G12S: 80% of Costello syndrome — the predominant pathogenic variant.\n"
                "  RASopathy → hyperactivated RAS-MAPK-ERK.\n"
                "  Coarse facies, papillomata (perianal/perioral/nasal) PATHOGNOMONIC.\n"
                "  Hypertrophic cardiomyopathy (60%), pulmonary valve stenosis.\n\n"
                "BLADDER TCC/RMS PATHOGNOMONIC:\n"
                "  Bladder TCC: ~4% lifetime; onset adolescence/young adult (median 10yr).\n"
                "  Embryonal rhabdomyosarcoma (eRMS) bladder: characteristic Costello tumour.\n"
                "  Annual bladder USS from age 5; urine cytology from age 8.\n\n"
                "MEK INHIBITOR — TRAMETINIB:\n"
                "  HRAS GOF → RAF-MEK-ERK → MEK inhibitor trametinib active.\n"
                "  Also active in: HRAS-mutant salivary carcinoma, HRAS-mutant thyroid.\n"
                "  Lonafarnib (farnesyl transferase inhibitor) investigational for Costello.\n\n"
                "DDx RASopathies:\n"
                "  PTPN11/SOS1/KRAS/RIT1/RAF1: Noonan; BRAF/MAP2K1-2: CFC; HRAS: Costello.\n"
                "  Costello = most severe + highest malignancy risk of the RASopathies.\n"
            ),
        },
        {
            "term": "CHEK2 / c.1100delC FOUNDER / BLADDER MODERATE RISK / INTERMEDIATE PENETRANCE",
            "definition": (
                "CHEK2 (Checkpoint Kinase 2) — 543aa / 60 kDa / 22q12.1 / AD LOF\n"
                "Intermediate-penetrance gene — bladder 2-3x; c.1100delC 1% Northern European founder.\n\n"
                "c.1100delC FOUNDER VARIANT:\n"
                "  ~1% carrier frequency in Northern Europe (Scandinavia, Netherlands, Poland).\n"
                "  Frameshift p.Thr367fs — LOF.\n"
                "  I157T common missense: moderate risk (lower than c.1100delC).\n\n"
                "BLADDER RISK — MODERATE (2-3x):\n"
                "  ATM → T68-CHK2 → impaired checkpoint → endogenous DSB → bladder instability.\n"
                "  Not high enough for dedicated bladder surveillance unless FH bladder.\n"
                "  If CHEK2 + family history bladder → discuss annual urine cytology.\n\n"
                "INTERMEDIATE PENETRANCE — NOT BRCA-EQUIVALENT:\n"
                "  Lifetime breast cancer ~20-25% (women) — intermediate (vs BRCA1 60-70%).\n"
                "  Annual breast MRI from age 35-40 (if FH/PRS confirms moderate-high composite risk).\n"
                "  PRS (polygenic risk score) critical: low PRS + CHEK2 → annual mammogram sufficient.\n"
                "  No olaparib indication without HRD evidence or somatic second hit.\n\n"
                "CASCADE TESTING:\n"
                "  Most important for breast/prostate cancer risk; bladder is secondary consideration.\n"
                "  Discuss aspirin for modest CRC risk modulation.\n"
            ),
        },
        {
            "term": "CASCADE TESTING — Hereditary Bladder/Urothelial Cancer",
            "definition": (
                "CASCADE TESTING PRIORITIES for Hereditary Bladder/Urothelial Cancer Predisposition:\n\n"
                "TIER 1 — HIGH YIELD (>10% lifetime urothelial risk):\n"
                "  MSH2 (25%) → ALL first-degree relatives; start surveillance at age 30-35.\n"
                "  MSH6 (7-11%) → All FDRs; note MSI-L pitfall; IHC primary.\n\n"
                "TIER 2 — MODERATE YIELD (2-5% lifetime urothelial risk):\n"
                "  MLH1 (2-4%) → All FDRs; exclude BRAF V600E (sporadic) and constitutional methylation.\n"
                "  BRCA2 (2-4x RR) → All FDRs; cisplatin/olaparib eligibility.\n\n"
                "TIER 3 — SYNDROME-SPECIFIC (paediatric/young adult bladder):\n"
                "  RB1 → Siblings of bilateral retinoblastoma; annual cytology from age 20 post-RT.\n"
                "  HRAS Costello → De novo (siblings low risk); USS from age 5; MEK trametinib.\n\n"
                "TIER 4 — MODERATE RISK (discuss, no dedicated bladder surveillance):\n"
                "  TP53 LFS → All FDRs; avoid RT; WBMRI Toronto annually.\n"
                "  CHEK2 → FDRs; breast/prostate primary concern; bladder if FH.\n\n"
                "IHC PITFALL RULES:\n"
                "  1. MSH6 IHC loss → Lynch regardless of MSI-L/MSS PCR result.\n"
                "  2. MLH1/PMS2 co-loss + BRAF V600E in tumour → sporadic (NOT Lynch).\n"
                "  3. MSH2 IHC loss + normal sequencing → CHECK EPCAM 3-prime deletion.\n"
                "  4. Normal IHC does NOT exclude CHEK2/BRCA2/TP53/HRAS — genes not tested by IHC.\n\n"
                "UNIVERSAL TUMOUR TESTING:\n"
                "  MSI/MMR IHC on ALL new urothelial carcinomas (analogous to universal CRC/endometrial testing).\n"
                "  Identifies Lynch carriers + selects MSI-H pembrolizumab candidates simultaneously.\n"
            ),
        },
    ]

    return {
        "atlas":       "Hereditary-Bladder-Urothelial-Cancer-Predisposition-Atlas",
        "seed_range":  f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": defs,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(json.dumps({k: v for k, v in ov.items() if k not in ("genes_detail",)}, indent=2))
    print("\n=== BREAKDOWN summary ===")
    br = generate_breakdown()
    for row in br["breakdown"]:
        print(f"  {row['gene']:6s} n={row['n']} mean_age={row['mean_age_onset']} "
              f"bladder/urothelial_n={row['bladder_urothelial_n']} ({row['bladder_urothelial_pct']}%)")
    print("\n=== DEFINITIONS (terms only) ===")
    df = generate_definitions()
    for d in df["definitions"]:
        print(f"  {d['term']}")
