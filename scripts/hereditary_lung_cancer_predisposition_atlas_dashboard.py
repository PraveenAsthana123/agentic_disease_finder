#!/usr/bin/env python3
"""Hereditary-Lung-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
EGFR   (Epidermal growth factor receptor; 1210aa; 7p11.2; AD GOF germline;
         Familial NSCLC — germline sensitizing EGFR (E709K, R108K, T790M germline) <1%;
         EGFR germline = lifelong osimertinib eligibility; lung adenocarcinoma primary;
         seed SEED_BASE+0) .
BRCA2  (DNA repair homologous recombination; 3418aa; 13q12.3; AD LOF;
         HBOC2 — lung adenocarcinoma 2-3x RR; cisplatin/PARPi sensitivity;
         PROfound trial: HRD-positive; pancreatic cancer 10x; prostate 15-20x;
         seed SEED_BASE+1) .
TP53   (Tumour suppressor p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome (LFS) — lung cancer 2-5x elevated;
         AVOID RADIATION ABSOLUTELY — radiation-induced secondary cancers;
         WB-MRI Toronto Protocol MANDATORY annual; seed SEED_BASE+2) .
STK11  (Serine-threonine kinase 11 / LKB1; 433aa; 19p13.3; AD LOF;
         Peutz-Jeghers Syndrome — NSCLC 16x RR HIGHEST hereditary lung risk;
         pulmonary hamartomas; mucocutaneous macules PATHOGNOMONIC;
         STK11 co-mutation KRAS drives NSCLC PD-L1-negative cold tumour (immunotherapy-resistant);
         seed SEED_BASE+3) .
DICER1 (RNA endoribonuclease; 1922aa; 14q32.13; AD LOF;
         DICER1 syndrome — pleuropulmonary blastoma TYPE I/II/III PATHOGNOMONIC childhood;
         pulmonary blastoma (adult) PATHOGNOMONIC; cystic nephroma; thyroid multinodular;
         DICER1 = hotspot somatic second hit (metal-binding RNase IIIb domain D1709/E1705);
         seed SEED_BASE+4) .
BAP1   (BRCA1-associated protein 1; 729aa; 3p21.1; AD LOF;
         BAP1 Tumour Predisposition Syndrome — mesothelioma 8-10% HIGHEST hereditary;
         lung adenocarcinoma elevated; uveal melanoma 50% lifetime PATHOGNOMONIC;
         BAP1-IHC nuclear loss confirms SDH/germline; BAPomas (MBAITs) PATHOGNOMONIC;
         seed SEED_BASE+5) .
FLCN   (Folliculin; 579aa; 17p11.2; AD LOF;
         Birt-Hogg-Dubé syndrome (BHD) — pulmonary cysts bilateral PATHOGNOMONIC;
         SPONTANEOUS PNEUMOTHORAX HIGHEST hereditary risk (40%); fibrofolliculomas PATHOGNOMONIC;
         ccRCC + chromophobe RCC; mTORC1 dysregulation via AMPK pathway;
         seed SEED_BASE+6) .
NF1    (Neurofibromin RAS-GAP; 2839aa; 17q11.2; AD LOF;
         Neurofibromatosis type 1 — NSCLC 2-3x; parenchymal NF1 involvement;
         MPNST 8-13% PATHOGNOMONIC sarcomatous; selumetinib FDA2020 MEK inhibitor;
         plexiform neurofibromas airways; café-au-lait 6+ PATHOGNOMONIC;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3414-3421)
"""
import random

SEED_BASE = 3414

ATLAS_GENES = [
    {
        "gene": "EGFR",
        "protein": (
            "EGFR -- 7p11.2 Autosomal-Dominant-GOF-Germline -- 1210aa -- "
            "EGFR-Receptor-Tyrosine-Kinase-134kDa-HER1-ErbB1-"
            "Germline-Sensitizing-Familial-NSCLC-"
            "E709K-R108K-T790M-Germline-Osimertinib-Eligible-Lifelong-"
            "Lung-Adenocarcinoma-Primary-EGFR-Targeted-FDA2013-2015-OMIM-131550"
        ),
        "locus": "7p11.2",
        "protein_size": (
            "1210 aa / 134 kDa / 7p11.2 EGFR lung cancer molecular context: "
            "STRUCTURE: "
            "  1210 aa / 134 kDa; extracellular domain (L1-CR1-L2-CR2 aa 1-621); "
            "  Single-pass transmembrane (aa 622-644); "
            "  Intracellular kinase domain (aa 712-979) with activation loop; "
            "  C-terminal tail: phosphorylation sites for STAT3, PI3K, RAS activation; "
            "  Germline GOF mutations: E709K (exon 18), R108K (exon 3), T790M (exon 20 germline = rare); "
            "CANCER RISKS (LUNG FOCUS): "
            "  Familial NSCLC: <1% all familial NSCLC carry germline sensitizing EGFR; "
            "  Lung adenocarcinoma overwhelmingly predominant histology; "
            "  Germline EGFR T790M: 3x lung cancer lifetime; de-novo resistance mechanism acquired; "
            "  Germline E709K: compound heterozygous with somatic L858R — dual activation; "
            "KEY MANAGEMENT: "
            "  Osimertinib 3rd-gen EGFR-TKI: FDA2015/2020 — germline sensitizing EGFR eligible lifelong; "
            "  FLAURA trial: osimertinib OS 38.6mo vs 31.8mo first-line metastatic; "
            "  Germline T790M: osimertinib first-line — do NOT use 1st/2nd-gen TKI; "
            "  Cascade germline testing for first-degree relatives if germline EGFR confirmed; "
            "  Annual low-dose CT lung: germline EGFR carriers from age 30"
        ),
        "syndrome": "Familial NSCLC / Germline EGFR sensitizing mutations — lung adenocarcinoma predisposition",
        "inheritance": "AD GOF germline (autosomal dominant gain-of-function); kinase activation",
        "lung_risk": "Familial NSCLC <1% hereditary; lung adenocarcinoma primary; 2-3x lifetime RR germline EGFR",
        "pathognomonic": "Germline T790M = lifelong osimertinib eligibility PATHOGNOMONIC; compound EGFR germline+somatic = dual activation",
        "key_avoid": "Do NOT use 1st/2nd-generation EGFR-TKI (gefitinib/erlotinib/afatinib) if germline T790M known — primary resistance; use osimertinib",
        "key_rule": "Germline EGFR: osimertinib is drug of choice (3rd-gen covers T790M resistance). Annual low-dose CT from age 30. Cascade first-degree germline testing. Lung adenocarcinoma nearly exclusive histology.",
        "surveillance": "Annual low-dose CT lung from age 30; annual molecular testing plasma ctDNA; cascade germline EGFR testing first-degree relatives; bronchoscopy only if CT-positive",
        "targeted_rx": "Osimertinib FDA2015/2020 3rd-gen EGFR-TKI first-line (covers T790M); erlotinib/gefitinib 1st-gen (non-T790M only); amivantamab+lazertinib exon 20 ins; platinum-pemetrexed + EGFR-TKI advanced NSCLC",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "BRCA2-FANCD1-Homologous-Recombination-Mediator-384kDa-"
            "HBOC2-Lung-Adenocarcinoma-2-3x-RR-"
            "Cisplatin-PARPi-Sensitivity-HRD-Positive-"
            "PROfound-Trial-Olaparib-FDA2020-Prostate-15-20x-Pancreatic-10x-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 lung cancer molecular context: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; PALB2-binding domain (N-terminus); "
            "  8 BRC repeats (aa 1002-2085): RAD51-binding — HR mediator; "
            "  DNA-binding domain (DBD aa 2402-3190); OB folds; tower domain; "
            "  C-terminal BRCA1-binding via BRCT domains; "
            "  BRCA2 LOF → RAD51 cannot load onto ssDNA → HR abolished → NHEJ error-prone; "
            "CANCER RISKS (LUNG FOCUS): "
            "  Lung adenocarcinoma: 2-3x RR germline BRCA2 (absolute risk ~5-6% lifetime); "
            "  Predominantly lung adenocarcinoma (non-squamous); "
            "  HRD-positive tumours: cisplatin/carboplatin hypersensitivity (platinum-induced ICL); "
            "  PARPi sensitivity established: olaparib SOLO trials + PROfound; "
            "  BRCA2 lung tumours: genomic instability signature 3 + elevated mutational burden; "
            "KEY MANAGEMENT: "
            "  Annual low-dose CT from age 40 (or 10yr before youngest affected); "
            "  Platinum-based first-line preferred (HRD cisplatin sensitivity); "
            "  PARPi maintenance after platinum response: olaparib/rucaparib/niraparib; "
            "  Germline BRCA2 = HRD testing MANDATORY on tumour (Myriad myChoice or Foundation); "
            "  Immunotherapy + PARPi combinations in BRCA2-mutant NSCLC — emerging data"
        ),
        "syndrome": "HBOC2 (Hereditary Breast-Ovarian Cancer type 2) — multi-cancer predisposition including lung",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); LOH 13q12.3 somatic second hit",
        "lung_risk": "Lung adenocarcinoma 2-3x RR; HRD-positive PARPi sensitive; cisplatin/carboplatin hypersensitive",
        "pathognomonic": "BRCA2 lung adenocarcinoma: HRD signature 3 + cisplatin sensitivity; PARPi eligible; germline confirmed on tumour",
        "key_avoid": "Do NOT skip HRD/platinum sensitivity testing in BRCA2 germline lung cancer — PARPi maintenance post-platinum is an actionable treatment benefit",
        "key_rule": "BRCA2 lung: platinum-first (HRD cisplatin sensitivity) + PARPi maintenance. HRD testing MANDATORY. Annual low-dose CT from age 40. Cascade BRCA2 first-degree testing.",
        "surveillance": "Annual low-dose CT lung from age 40; annual breast MRI (women); annual prostate PSA men from 40yr; annual ca125 + pelvic USS women; cascade BRCA2 germline first-degree relatives",
        "targeted_rx": "Platinum (cisplatin/carboplatin) HRD-sensitive first-line NSCLC; olaparib FDA2020 PARPi maintenance HRD+; rucaparib/niraparib PARPi alternatives; bevacizumab+chemotherapy advanced NSCLC; pembrolizumab + PARPi emerging",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-Tumour-Suppressor-43kDa-Transcription-Factor-"
            "Li-Fraumeni-Syndrome-LFS-Lung-2-5x-Elevated-"
            "AVOID-RADIATION-ABSOLUTELY-Secondary-Cancer-Risk-"
            "WB-MRI-Toronto-Protocol-MANDATORY-Annual-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 lung cancer molecular context: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; N-terminal transactivation domain 1+2 (aa 1-67); "
            "  Proline-rich region (aa 68-98); sequence-specific DNA-binding domain (aa 102-292); "
            "  Tetramerisation domain (aa 325-356); C-terminal regulatory domain (aa 363-393); "
            "  Hotspot mutations: R175H, G245S, R248W/Q, R249S, R273H/C, R282W; "
            "  p53 LOF → no MDM2-mediated G1 arrest → genomic instability accumulation; "
            "CANCER RISKS (LUNG FOCUS): "
            "  Lung cancer: 2-5x elevated LFS lifetime; adeno + squamous + SCLC all elevated; "
            "  RADIATION-INDUCED secondary lung cancer: CATASTROPHIC risk in LFS — AVOID absolutely; "
            "  Median age LFS lung cancer: 30-35yr (much younger than sporadic 60+yr); "
            "  LFS cumulative cancer risk: >90% by age 70 (all sites); breast 25-35%; sarcoma 25-30%; "
            "KEY MANAGEMENT: "
            "  WB-MRI Toronto Protocol: annual whole-body MRI + brain MRI — MANDATORY LFS surveillance; "
            "  Low-dose CT chest: annual from diagnosis (no radiation — CT not avoided, radiation therapy avoided); "
            "  NEVER use radiation therapy for LFS lung cancer — radiation-field sarcoma + secondary malignancy; "
            "  Immune checkpoint inhibitors preferred over radiation consolidation in stage III; "
            "  Chemotherapy: standard platinum-based; PARPi utility in TP53-co-mutant HRD tumours (limited)"
        ),
        "syndrome": "Li-Fraumeni Syndrome (LFS) — multi-cancer predisposition; sarcoma/breast/lung/brain/adrenal",
        "inheritance": "AD LOF (autosomal dominant); dominant negative effect hotspot mutants sequester WT p53",
        "lung_risk": "Lung cancer 2-5x elevated; all histologies; median onset 30-35yr (much earlier than sporadic)",
        "pathognomonic": "LFS = early-onset multi-cancer; radiation-induced sarcoma PATHOGNOMONIC in-field; TP53 R175H/R248W hotspot dominant-negative; WB-MRI mandatory",
        "key_avoid": "AVOID RADIATION ABSOLUTELY in LFS — radiation-induced field sarcoma documented, catastrophic second cancer risk. Never use radiation consolidation or SBRT; prefer immunotherapy/surgery",
        "key_rule": "LFS: WB-MRI Toronto Protocol annual MANDATORY (no radiation). Annual low-dose CT chest. AVOID all radiation therapy. TP53 hotspot dominant-negative = functional oncogene. Cascade germline testing first-degree.",
        "surveillance": "Annual WB-MRI + brain MRI (Toronto Protocol MANDATORY); annual low-dose CT chest; annual breast MRI (women) from age 20; annual dermatology; cascade TP53 germline testing first-degree relatives",
        "targeted_rx": "Platinum-based chemotherapy first-line (avoid bleomycin-pulmonary toxicity); immune checkpoint inhibitors (avoid radiation consolidation); APR-246 (p53 reactivator — clinical trials); sotorasib/adagrasib co-mutant KRAS; MDM2 inhibitors (clinical trial)",
    },
    {
        "gene": "STK11",
        "protein": (
            "STK11 -- 19p13.3 Autosomal-Dominant-LOF -- 433aa -- "
            "LKB1-Serine-Threonine-Kinase-48kDa-AMPK-Master-Regulator-"
            "Peutz-Jeghers-Syndrome-NSCLC-16x-RR-HIGHEST-Hereditary-Lung-"
            "Pulmonary-Hamartomas-STK11-KRAS-Co-Mutation-Immunotherapy-Resistant-"
            "Mucocutaneous-Macules-PATHOGNOMONIC-GI-Polyposis-OMIM-602216"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "433 aa / 48 kDa / 19p13.3 STK11 lung cancer molecular context: "
            "STRUCTURE: "
            "  433 aa / 48 kDa; N-terminal regulatory domain; "
            "  Serine-threonine kinase domain (aa 49-309); activation loop Thr185; "
            "  C-terminal farnesylation CAAX motif — membrane targeting; "
            "  STK11 phosphorylates AMPK Thr172 → AMPK activation; "
            "  STK11 LOF → AMPK inactivated → mTORC1 hyperactivation + HIF-1alpha → tumourigenesis; "
            "CANCER RISKS (LUNG FOCUS): "
            "  NSCLC: 16x RR — HIGHEST hereditary lung cancer relative risk; "
            "  Adenocarcinoma overwhelmingly predominant (vs PJS GI polyps — hamartomatous); "
            "  Pulmonary hamartomas: benign endobronchial/parenchymal — PATHOGNOMONIC PJS lung manifestation; "
            "  STK11-co-mutant KRAS (co-occurring somatic): COLD TUMOUR — PD-L1 negative, immunotherapy-RESISTANT; "
            "  STK11 loss → neutrophil/MDM2 immunosuppressive infiltrate → no T-cell activation; "
            "KEY MANAGEMENT: "
            "  Annual low-dose CT chest from age 25 (10yr before youngest affected family member); "
            "  STK11+KRAS co-mutation: immunotherapy (PD-1/PD-L1) INEFFECTIVE — use chemotherapy; "
            "  MEK inhibitors (trametinib) + KRAS inhibitors (sotorasib) in STK11/KRAS co-mutant; "
            "  GI endoscopy every 2-3yr from age 8 (Peutz-Jeghers GI polyp surveillance); "
            "  Breast MRI annually women age 25+ (PJS breast 50% lifetime)"
        ),
        "syndrome": "Peutz-Jeghers Syndrome (PJS) — NSCLC 16x RR HIGHEST hereditary; GI hamartomatous polyps; mucocutaneous macules",
        "inheritance": "AD LOF (autosomal dominant); haploinsufficiency sufficient for lung cancer predisposition",
        "lung_risk": "NSCLC 16x RR HIGHEST hereditary lung risk; pulmonary hamartomas PATHOGNOMONIC PJS lung",
        "pathognomonic": "Mucocutaneous perioral/genital macules PATHOGNOMONIC PJS; STK11+KRAS co-mutation = immunotherapy-resistant COLD TUMOUR; pulmonary hamartomas",
        "key_avoid": "Do NOT use immunotherapy (PD-1/PD-L1) alone in STK11+KRAS co-mutant NSCLC — primary resistance. Also do NOT confuse pulmonary hamartomas with malignant lesions on CT (benign PJS lung finding)",
        "key_rule": "STK11/PJS: NSCLC 16x RR = HIGHEST hereditary. STK11+KRAS = immunotherapy-cold: use chemotherapy + KRAS inhibitor. Annual low-dose CT from age 25. GI scope from age 8. Breast MRI women 25+.",
        "surveillance": "Annual low-dose CT chest from age 25; annual breast MRI women from 25yr; GI endoscopy every 2-3yr from age 8; annual pelvic USS + Ca-125 women (ovarian sex cord tumour 21%); annual testicular USS males (SCTAT)",
        "targeted_rx": "Platinum+pemetrexed chemotherapy (immunotherapy-resistant STK11/KRAS); sotorasib/adagrasib KRAS G12C inhibitor in co-mutant; trametinib MEK inhibitor STK11/KRAS; everolimus mTORC1 (STK11 directly suppresses mTORC1); no established PARPi indication",
    },
    {
        "gene": "DICER1",
        "protein": (
            "DICER1 -- 14q32.13 Autosomal-Dominant-LOF -- 1922aa -- "
            "DICER1-RNase-III-Endoribonuclease-218kDa-miRNA-Processor-"
            "DICER1-Syndrome-Pleuropulmonary-Blastoma-PPB-TYPE-I-II-III-PATHOGNOMONIC-"
            "Pulmonary-Blastoma-Adult-PATHOGNOMONIC-"
            "Somatic-Hotspot-D1709-E1705-Metal-Binding-RNase-IIIb-OMIM-606241"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "1922 aa / 218 kDa / 14q32.13 DICER1 lung cancer molecular context: "
            "STRUCTURE: "
            "  1922 aa / 218 kDa; N-terminal helicase domain (DEXDc + HELICc); "
            "  PAZ domain (aa 859-974): binds 3' end of pre-miRNA; "
            "  RNase IIIa domain (aa 1278-1327): cleaves miRNA guide strand; "
            "  RNase IIIb domain (aa 1358-1530): cleaves passenger strand; "
            "  METAL-BINDING ACTIVE SITE: D1709, E1705 — hotspot somatic second-hit mutations; "
            "  Mg2+ coordinated at metal-binding site — catalytic activity; "
            "  DICER1 LOF → impaired miRNA biogenesis → oncogenic derepression; "
            "CANCER RISKS (LUNG FOCUS): "
            "  Pleuropulmonary blastoma (PPB): TYPE I (purely cystic) / TYPE II (mixed cystic-solid) / TYPE III (solid) — PATHOGNOMONIC DICER1 syndrome childhood lung malignancy; "
            "  PPB median age: Type I <2yr; Type II 2-3yr; Type III 3-4yr — all CHILDHOOD; "
            "  Pulmonary blastoma (adult): PATHOGNOMONIC when DICER1 germline confirmed; "
            "  Cystic nephroma 10%; thyroid multinodular/differentiated thyroid cancer 8-10%; "
            "  Ovarian sex cord-stromal SLCT 10-15%; pineal region tumours; "
            "KEY MANAGEMENT: "
            "  PPB surveillance: annual low-dose CT from birth to age 8 in DICER1 families; "
            "  PPB treatment: multimodal surgery + IVADo (ifosfamide/vincristine/actinomycin/doxorubicin); "
            "  Adult pulmonary blastoma: surgery + cisplatin-based; "
            "  Somatic hotspot (D1709/E1705): second hit required for malignant transformation (two-hit); "
            "  Thyroid surveillance: annual USS from age 8 (thyroid cancer risk 8-10%)"
        ),
        "syndrome": "DICER1 Syndrome — PPB type I/II/III PATHOGNOMONIC; pleuropulmonary blastoma; thyroid; ovarian SLCT",
        "inheritance": "AD LOF (autosomal dominant); two-hit model: germline LOF + somatic hotspot RNase IIIb D1709/E1705",
        "lung_risk": "PPB type I/II/III PATHOGNOMONIC childhood; pulmonary blastoma adult PATHOGNOMONIC; cystic lung lesions in infancy",
        "pathognomonic": "Pleuropulmonary blastoma (PPB) type I/II/III = PATHOGNOMONIC DICER1 syndrome; pulmonary blastoma adult PATHOGNOMONIC; somatic RNase IIIb D1709/E1705 hotspot",
        "key_avoid": "Do NOT delay PPB diagnosis — Type I → Type III progression is documented; annual CT from birth MANDATORY in DICER1 families (not optional). Do NOT mistake Type I PPB cysts for benign congenital pulmonary airway malformation (CPAM)",
        "key_rule": "DICER1: PPB Type I/II/III PATHOGNOMONIC childhood lung. Annual CT from birth to age 8. Two-hit model (germline LOF + somatic RNase IIIb hotspot). Adult pulmonary blastoma = germline DICER1 until proven otherwise. Thyroid USS from age 8.",
        "surveillance": "Annual low-dose CT chest from birth to age 8 (PPB surveillance MANDATORY); annual thyroid USS from age 8; annual pelvic USS girls (ovarian SLCT); ophthalmology + brain MRI if symptoms (ciliary body, pineal); cascade DICER1 germline testing",
        "targeted_rx": "PPB IVADo multimodal: ifosfamide/vincristine/actinomycin/doxorubicin; adult pulmonary blastoma: cisplatin+paclitaxel; bevacizumab PPB refractory; regorafenib DICER1-mutant advanced (emerging); thyroid DTC: radioiodine if metastatic",
    },
    {
        "gene": "BAP1",
        "protein": (
            "BAP1 -- 3p21.1 Autosomal-Dominant-LOF -- 729aa -- "
            "BAP1-Deubiquitinase-80kDa-BRCA1-Associated-"
            "BAP1-Tumour-Predisposition-Syndrome-TPDS-"
            "Mesothelioma-8-10pct-HIGHEST-Hereditary-"
            "Uveal-Melanoma-50pct-Lifetime-PATHOGNOMONIC-"
            "BAPomas-MBAITs-PATHOGNOMONIC-BAP1-IHC-Nuclear-Loss-OMIM-603089"
        ),
        "locus": "3p21.1",
        "protein_size": (
            "729 aa / 80 kDa / 3p21.1 BAP1 lung/mesothelioma molecular context: "
            "STRUCTURE: "
            "  729 aa / 80 kDa; UCH (ubiquitin C-terminal hydrolase) catalytic domain (aa 1-240); "
            "  BARD1-interacting domain; nuclear localisation signal (NLS); "
            "  BRCA1-binding BARD1-interacting region; HCF-1/YY1 chromatin remodelling complex; "
            "  BAP1 deubiquitinates H2AK119ub1 (Polycomb repression) → chromatin activation; "
            "  BAP1 LOF → H2AK119ub1 accumulation → silencing of tumour suppressors; "
            "CANCER RISKS (LUNG/PLEURAL FOCUS): "
            "  Malignant pleural mesothelioma (MPM): 8-10% BAP1 germline — HIGHEST hereditary mesothelioma risk; "
            "  Asbestos co-exposure: synergistic — germline BAP1 + asbestos = 100x MPM risk vs general pop; "
            "  Lung adenocarcinoma: elevated 2-3x (parenchymal, not pleural); "
            "  Uveal melanoma: 50% lifetime PATHOGNOMONIC BAP1-TPDS; "
            "  BAPomas (MBAITs = melanocytic BAP1-altered intradermal tumours): PATHOGNOMONIC skin; "
            "  BAP1-IHC: nuclear loss confirms SDH/germline BAP1 — universal IHC marker; "
            "KEY MANAGEMENT: "
            "  ASBESTOS AVOIDANCE MANDATORY for all BAP1 germline carriers — removes synergistic MPM trigger; "
            "  Annual ophthalmology (uveal melanoma MANDATORY); "
            "  Annual CT chest + abdomen (MPM surveillance); "
            "  BAP1-IHC on all mesothelioma specimens — identifies germline carriers for cascade testing; "
            "  Immunotherapy: pembrolizumab/nivolumab mesothelioma (BAP1 deficiency = immune active tumour)"
        ),
        "syndrome": "BAP1 Tumour Predisposition Syndrome (BAP1-TPDS) — mesothelioma + uveal melanoma + lung + cutaneous BAPomas",
        "inheritance": "AD LOF (autosomal dominant); nuclear loss on IHC = functional marker",
        "lung_risk": "Mesothelioma 8-10% HIGHEST hereditary; lung adenocarcinoma 2-3x elevated; asbestos synergy 100x MPM",
        "pathognomonic": "BAPomas/MBAITs skin PATHOGNOMONIC BAP1-TPDS; uveal melanoma 50% PATHOGNOMONIC; BAP1-IHC nuclear loss = global SDHx/BAP1 marker; mesothelioma 8-10% HIGHEST",
        "key_avoid": "ASBESTOS AVOIDANCE MANDATORY for BAP1 germline — synergistic 100x MPM risk. Do NOT miss BAPomas on skin exam (PATHOGNOMONIC trigger for cascade BAP1 testing). BAP1 IHC nuclear loss = check germline",
        "key_rule": "BAP1: asbestos ABSOLUTELY avoided. Annual ophthalmology (uveal melanoma 50%). Annual CT chest (mesothelioma). BAPomas = PATHOGNOMONIC — germline test. BAP1-IHC nuclear loss on mesothelioma = cascade germline testing family.",
        "surveillance": "Annual CT chest + upper abdomen (MPM + lung adenocarcinoma surveillance); annual ophthalmology (uveal melanoma MANDATORY); annual dermatology (BAPomas/MBAITs + melanoma); annual renal USS (RCC); cascade BAP1 germline testing first-degree relatives",
        "targeted_rx": "Pembrolizumab/nivolumab immunotherapy MPM (BAP1-deficient immune active); platinum+pemetrexed MPM (cisplatin-sensitive); bevacizumab+pemetrexed+cisplatin MPM; HDAC inhibitors BAP1-deficient preclinical; targeted: no approved BAP1-specific agent",
    },
    {
        "gene": "FLCN",
        "protein": (
            "FLCN -- 17p11.2 Autosomal-Dominant-LOF -- 579aa -- "
            "Folliculin-64kDa-AMPK-RAGULATOR-mTORC1-Regulator-"
            "Birt-Hogg-Dube-Syndrome-BHD-"
            "Pulmonary-Cysts-Bilateral-PATHOGNOMONIC-"
            "Spontaneous-Pneumothorax-40pct-HIGHEST-Hereditary-Risk-"
            "Fibrofolliculomas-Trichodiscomas-PATHOGNOMONIC-ccRCC-Chromophobe-OMIM-607273"
        ),
        "locus": "17p11.2",
        "protein_size": (
            "579 aa / 64 kDa / 17p11.2 FLCN lung molecular context: "
            "STRUCTURE: "
            "  579 aa / 64 kDa; N-terminal coiled-coil region; "
            "  DENN-like domain (differentially expressed in normal/neoplastic cells); "
            "  FLCN-FNIP1/2 complex: interacts with AMPK at lysosomal surface; "
            "  FLCN-FNIP complex activates mTORC1 via Rag-GTPase: RagA/B GTP-loading; "
            "  FLCN LOF → AMPK constitutively active → mTORC1 suppression → TFEB nuclear → lysosomal biogenesis; "
            "  FLCN also activates RhoA/ROCK pathway; downstream of AMPK; "
            "CANCER RISKS (LUNG FOCUS): "
            "  Pulmonary cysts: bilateral basal/subpleural PATHOGNOMONIC BHD (80-90% carriers); "
            "  Spontaneous pneumothorax: 40% HIGHEST hereditary risk in BHD (cyst rupture); "
            "  Pneumothorax recurrence risk HIGH: pleural scarification at first episode MANDATORY counselling; "
            "  NSCLC: 2-3x elevated; predominantly adenocarcinoma in FLCN carriers; "
            "  ccRCC: 20-30% bilateral multifocal; chromophobe RCC + hybrid oncocytoma; "
            "KEY MANAGEMENT: "
            "  Pneumothorax: chest tube immediate; pleurodesis surgical after first episode in BHD strongly advised; "
            "  Annual low-dose CT chest (cyst surveillance + lung cancer from age 30); "
            "  Annual renal MRI/USS (ccRCC/chromophobe RCC — resect >3cm); "
            "  Fibrofolliculomas: trichodiscoma/acrochordon skin — PATHOGNOMONIC (dermatology annual); "
            "  mTORC1 inhibitors (everolimus): experimental FLCN-related RCC"
        ),
        "syndrome": "Birt-Hogg-Dubé Syndrome (BHD) — pulmonary cysts + pneumothorax + fibrofolliculomas + RCC",
        "inheritance": "AD LOF (autosomal dominant); haploinsufficiency drives cyst formation + RCC",
        "lung_risk": "Pulmonary cysts bilateral PATHOGNOMONIC; pneumothorax 40% HIGHEST hereditary; NSCLC 2-3x elevated",
        "pathognomonic": "Bilateral basal pulmonary cysts PATHOGNOMONIC BHD; fibrofolliculomas/trichodiscomas PATHOGNOMONIC skin BHD; spontaneous pneumothorax at 40% HIGHEST hereditary risk",
        "key_avoid": "Do NOT manage first BHD pneumothorax conservatively without pleurodesis counselling — 40% recurrence and HIGHEST hereditary pneumothorax risk. Do NOT confuse FLCN cysts with LAM (LAM = women of childbearing age, chylous effusion, sirolimus-responsive)",
        "key_rule": "FLCN/BHD: pneumothorax 40% — pleurodesis after first episode strongly advised. Bilateral pulmonary cysts PATHOGNOMONIC. Annual CT + renal MRI. Fibrofolliculomas = diagnostic trigger. Distinguish from LAM clinically.",
        "surveillance": "Annual low-dose CT chest from age 30 (cysts + NSCLC); annual renal MRI (ccRCC bilateral multifocal); annual dermatology (fibrofolliculomas/trichodiscomas PATHOGNOMONIC); spirometry annual (obstructive pattern BHD); cascade FLCN germline testing",
        "targeted_rx": "Pleurodesis surgical first-episode pneumothorax BHD (MANDATORY counselling); everolimus mTORC1 inhibitor FLCN-RCC (emerging); sunitinib/cabozantinib chromophobe RCC advanced; platinum+pemetrexed NSCLC; no approved FLCN-specific lung agent",
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RAS-GTPase-Activating-Protein-"
            "Neurofibromatosis-Type-1-NSCLC-2-3x-RR-"
            "MPNST-8-13pct-PATHOGNOMONIC-Sarcomatous-Transformation-"
            "Selumetinib-FDA2020-MEK-Inhibitor-"
            "Cafe-au-Lait-Macules-6plus-PATHOGNOMONIC-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 lung cancer molecular context: "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; RAS-GAP catalytic domain (aa 1177-1530) central; "
            "  Sec14-homology domain; PH-like domain; "
            "  CRAL-TRIO lipid-binding domain; three ARM-repeat units; "
            "  GRD (GAP-related domain aa 1198-1530): RAS Gln61 hydrolysis → RAS-GDP (inactive); "
            "  NF1 LOF → constitutive RAS-GTP → RAF/MEK/ERK + PI3K/AKT activation; "
            "CANCER RISKS (LUNG FOCUS): "
            "  NSCLC: 2-3x elevated lifetime RR; predominantly adenocarcinoma; "
            "  NF1-somatic mutations: 11% sporadic NSCLC (acquired somatic — NOT hereditary); "
            "  Plexiform neurofibromas airways: endobronchial compression → airway obstruction + infection; "
            "  MPNST (malignant peripheral nerve sheath tumour): 8-13% NF1 lifetime — thoracic MPNST can mimic lung mass; "
            "  Internal neurofibromas mediastinum/chest wall; GIST 7% (distinct from KIT-GIST); "
            "KEY MANAGEMENT: "
            "  Annual chest imaging (CXR/CT) for plexiform neurofibroma + mediastinal NF monitoring; "
            "  MPNST: surgery + doxorubicin/ifosfamide; selumetinib (MEK inhibitor) FDA2020 plexiform neurofibroma; "
            "  NF1-somatic NSCLC: MEK/ERK pathway — trametinib + KRAS inhibitors if co-mutant; "
            "  Lung adenocarcinoma: standard platinum-based + NF1 pathway inhibitors emerging; "
            "  Annual ophthalmology (Lisch nodules PATHOGNOMONIC + optic pathway glioma)"
        ),
        "syndrome": "Neurofibromatosis Type 1 (NF1) — multi-system RASopathy; NSCLC 2-3x; MPNST 8-13%; plexiform neurofibromas",
        "inheritance": "AD LOF (autosomal dominant); haploinsufficiency + somatic LOH drives full NF1 tumorigenesis",
        "lung_risk": "NSCLC 2-3x elevated; plexiform neurofibromas airways endobronchial; thoracic MPNST mimics lung mass",
        "pathognomonic": "Café-au-lait macules 6+ PATHOGNOMONIC NF1; Lisch nodules iris PATHOGNOMONIC; MPNST in plexiform NF 8-13% PATHOGNOMONIC malignant transformation; axillary/inguinal freckling PATHOGNOMONIC",
        "key_avoid": "Do NOT confuse thoracic MPNST with primary lung cancer on CT — MPNST = sarcoma (NOT NSCLC), treated with doxorubicin/ifosfamide, NOT platinum-based chemotherapy. NF1-somatic NSCLC ≠ NF1-germline: germline = NF1 syndrome (widespread disease)",
        "key_rule": "NF1: MPNST 8-13% = sarcoma NOT lung cancer — doxorubicin/ifosfamide, NOT platinum. Selumetinib FDA2020 plexiform neurofibroma. Annual chest CT + ophthalmology + clinical NF exam. Café-au-lait 6+ PATHOGNOMONIC.",
        "surveillance": "Annual full-body clinical exam (skin + neuro + ophthalmology); annual ophthalmology (Lisch nodules + optic glioma); annual CT chest/abdomen (plexiform + MPNST); annual MRI brain/spine if neurological symptoms; cascade NF1 germline testing",
        "targeted_rx": "Selumetinib FDA2020 MEK inhibitor NF1 plexiform neurofibroma; doxorubicin+ifosfamide MPNST standard; trametinib MEK inhibitor MPNST emerging; binimetinib MEK NF1 adult trials; standard platinum NSCLC NF1-germline; RAS inhibitors NF1-somatic NSCLC",
    },
]

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]

_TUMOUR_TYPES = {
    "EGFR":   ["Lung adenocarcinoma germline EGFR", "EGFR exon 19 del NSCLC", "EGFR T790M germline NSCLC", "EGFR L858R + germline NSCLC", "Stage IV metastatic NSCLC EGFR"],
    "BRCA2":  ["Lung adenocarcinoma BRCA2 HRD", "NSCLC HRD-positive platinum-sensitive", "Stage III NSCLC BRCA2", "Metastatic lung BRCA2 PARPi eligible", "NSCLC co-mutant BRCA2+TP53"],
    "TP53":   ["LFS lung adenocarcinoma", "LFS early-onset NSCLC age 30-35", "LFS squamous cell lung carcinoma", "LFS SCLC small cell lung", "Radiation-induced secondary lung LFS"],
    "STK11":  ["NSCLC PJS STK11 adenocarcinoma", "STK11+KRAS co-mutant NSCLC immunotherapy-cold", "Pulmonary hamartoma PJS benign", "Stage IV NSCLC STK11 LOF", "PJS multifocal lung adenocarcinoma"],
    "DICER1": ["PPB type I cystic pleuropulmonary", "PPB type II mixed cystic-solid", "PPB type III solid childhood", "Pulmonary blastoma adult DICER1", "Cystic lung lesion DICER1 syndrome"],
    "BAP1":   ["Malignant pleural mesothelioma BAP1", "Lung adenocarcinoma BAP1-TPDS", "Biphasic mesothelioma BAP1 germline", "Stage III MPM BAP1 immunotherapy", "Peritoneal mesothelioma BAP1"],
    "FLCN":   ["Bilateral pulmonary cysts BHD", "Spontaneous pneumothorax FLCN BHD", "NSCLC adenocarcinoma FLCN carrier", "Recurrent pneumothorax BHD FLCN", "Parenchymal cysts + NSCLC BHD"],
    "NF1":    ["Thoracic MPNST NF1 malignant", "NSCLC adenocarcinoma NF1 germline", "Plexiform neurofibroma endobronchial NF1", "Mediastinal NF1 internal neurofibroma", "NF1-somatic NSCLC co-mutation"],
}

_VARIANTS_BY_GENE = {
    "EGFR":   ["c.2125G>A (p.Glu709Lys — E709K germline)", "c.323G>A (p.Arg108Lys — R108K germline)", "c.2369C>T (p.Thr790Met — T790M germline)", "c.2573T>G (p.Leu858Arg + germline)", "Exon 19 del germline EGFR"],
    "BRCA2":  ["c.5946delT (p.Ser1982Argfs — Ashkenazi)", "c.6174delT (p.Ser2092Tyrfs — Ashkenazi)", "c.3036_3039delACAA (BRCA2 exon 11)", "Large exon 14-26 deletion BRCA2", "c.9976A>T (p.Lys3326Ter — benign)"],
    "TP53":   ["c.524G>A (p.Arg175His — R175H hotspot)", "c.742C>T (p.Arg248Trp — R248W hotspot)", "c.743G>A (p.Arg248Gln — R248Q hotspot)", "c.817C>T (p.Arg273Cys — R273C)", "c.844C>T (p.Arg282Trp — R282W)"],
    "STK11":  ["c.471+1G>A splice STK11", "c.858del (p.Lys286Asnfs STK11)", "Large deletion STK11 exons 1-4", "c.1062+1G>T splice STK11", "c.865A>T (p.Lys289Ter STK11)"],
    "DICER1": ["c.5125G>A (p.Asp1709Asn — D1709N hotspot)", "c.5113G>A (p.Glu1705Lys — E1705K hotspot)", "c.1A>G (p.Met1Val start-codon DICER1)", "Large exon deletion DICER1", "c.5438T>A (p.Leu1813His DICER1 RNase IIIb)"],
    "BAP1":   ["c.588_592delCATTA (p.Tyr196Ter BAP1)", "c.2116C>T (p.Arg706Ter BAP1 NLS)", "Large deletion BAP1 exons 4-7", "c.1214delC (p.Pro405Leufs BAP1)", "c.656C>T (p.Pro219Leu BAP1 UCH)"],
    "FLCN":   ["c.1285delC (p.His429Thrfs FLCN)", "c.1285dupC (p.His429Profs FLCN)", "c.1378C>T (p.Arg460Ter FLCN)", "Large exon 9-14 deletion FLCN", "c.1062+1G>A splice FLCN"],
    "NF1":    ["c.2033_2046del14 (p.Met678Argfs NF1)", "c.6579C>A (p.Tyr2193Ter NF1)", "Large deletion NF1 17q11.2 microdeletion", "c.2407C>T (p.Arg803Ter NF1)", "c.4006C>T (p.Arg1336Trp NF1 GAP)"],
}

TREATMENT_PROTOCOLS_BY_GENE = {
    "EGFR":   ["Osimertinib 3rd-gen EGFR-TKI FDA2015/2020 (germline T790M or sensitizing mutation)", "Erlotinib/gefitinib 1st-gen EGFR-TKI (non-T790M only)", "Amivantamab+lazertinib exon 20 insertion EGFR", "Platinum+pemetrexed+EGFR-TKI advanced NSCLC", "Low-dose CT surveillance + ctDNA monitoring germline EGFR"],
    "BRCA2":  ["Platinum-based chemotherapy HRD-sensitive first-line NSCLC", "Olaparib PARPi maintenance FDA2020 HRD+ post-platinum", "Rucaparib/niraparib PARPi alternatives BRCA2-mutant", "Bevacizumab+platinum+pemetrexed advanced NSCLC", "Pembrolizumab+PARPi combination BRCA2-mutant NSCLC emerging"],
    "TP53":   ["Platinum-based chemotherapy standard NSCLC LFS", "Immune checkpoint inhibitors (avoid radiation consolidation) stage III LFS", "APR-246 (p53 reactivator — clinical trials TP53 LOF)", "Sotorasib/adagrasib KRAS co-mutant LFS NSCLC", "MDM2 inhibitors (HDM201/RG7388 — clinical trial TP53 wildtype co-mutation)"],
    "STK11":  ["Platinum+pemetrexed chemotherapy (immunotherapy-resistant STK11/KRAS)", "Sotorasib/adagrasib KRAS G12C in STK11+KRAS co-mutant NSCLC", "Trametinib MEK inhibitor STK11/KRAS co-mutant", "Everolimus mTORC1 inhibitor (STK11 directly suppresses mTORC1)", "Carboplatin+paclitaxel+bevacizumab STK11 advanced NSCLC"],
    "DICER1": ["IVADo multimodal PPB (ifosfamide/vincristine/actinomycin/doxorubicin)", "Cisplatin+paclitaxel adult pulmonary blastoma DICER1", "Bevacizumab PPB refractory/metastatic", "Regorafenib DICER1-mutant advanced (emerging data)", "Thyroid DTC radioiodine DICER1 syndrome if metastatic"],
    "BAP1":   ["Pembrolizumab/nivolumab immunotherapy MPM (BAP1-deficient immune active tumour)", "Platinum+pemetrexed MPM cisplatin-sensitive first-line", "Bevacizumab+pemetrexed+cisplatin MPM (MAPS trial)", "HDAC inhibitors BAP1-deficient (preclinical emerging)", "Maintenance nivolumab MPM post-platinum (CheckMate 743)"],
    "FLCN":   ["Pleurodesis surgical first-episode pneumothorax BHD (MANDATORY counselling)", "Everolimus mTORC1 inhibitor FLCN-RCC emerging", "Sunitinib/cabozantinib chromophobe RCC advanced", "Platinum+pemetrexed NSCLC FLCN carrier standard", "Chest tube + VATS pleurodesis recurrent pneumothorax BHD"],
    "NF1":    ["Selumetinib FDA2020 MEK inhibitor NF1 plexiform neurofibroma", "Doxorubicin+ifosfamide MPNST standard sarcoma regimen", "Trametinib MEK inhibitor MPNST/NF1 emerging", "Binimetinib MEK NF1 adult clinical trials", "Standard platinum NSCLC NF1-germline (no specific pathway agent approved)"],
}

SURVEILLANCE_BY_GENE = {
    "EGFR":   ["Annual low-dose CT lung from age 30 (germline EGFR carriers)", "Annual ctDNA plasma monitoring EGFR TKI resistance", "Cascade germline EGFR testing first-degree relatives", "Bronchoscopy if CT-suspicious mass (not prophylactic)", "Annual molecular reflex testing on progression"],
    "BRCA2":  ["Annual low-dose CT lung from age 40 (or 10yr before youngest affected)", "Annual breast MRI women BRCA2 (primary HBOC surveillance)", "Annual prostate PSA men from age 40yr BRCA2", "Annual Ca-125 + pelvic USS women (ovarian risk)", "Cascade BRCA2 germline testing first-degree relatives"],
    "TP53":   ["Annual WB-MRI + brain MRI Toronto Protocol MANDATORY (no radiation)", "Annual low-dose CT chest (CT not avoided; radiation therapy avoided)", "Annual breast MRI women from age 20yr LFS", "Annual dermatology LFS", "Cascade TP53 germline testing first-degree relatives"],
    "STK11":  ["Annual low-dose CT chest from age 25 (16x RR NSCLC)", "Annual breast MRI women from age 25 (PJS breast 50% lifetime)", "GI endoscopy every 2-3yr from age 8 (PJS GI polyp surveillance)", "Annual pelvic USS + Ca-125 women (ovarian sex cord tumour 21%)", "Annual testicular USS males (SCTAT PJS)"],
    "DICER1": ["Annual low-dose CT chest from birth to age 8 (PPB MANDATORY)", "Annual thyroid USS from age 8 (thyroid cancer 8-10%)", "Annual pelvic USS girls (ovarian SLCT DICER1)", "Ophthalmology + brain MRI if ciliary/pineal symptoms", "Cascade DICER1 germline testing first-degree relatives"],
    "BAP1":   ["Annual CT chest + upper abdomen (MPM + lung adenocarcinoma)", "Annual ophthalmology MANDATORY (uveal melanoma 50% lifetime)", "Annual dermatology (BAPomas/MBAITs PATHOGNOMONIC + cutaneous melanoma)", "Annual renal USS/MRI (RCC BAP1 elevated)", "Cascade BAP1 germline testing first-degree relatives"],
    "FLCN":   ["Annual low-dose CT chest from age 30 (cysts + NSCLC surveillance)", "Annual renal MRI (ccRCC + chromophobe bilateral multifocal)", "Annual dermatology (fibrofolliculomas/trichodiscomas PATHOGNOMONIC)", "Annual spirometry (obstructive pattern BHD)", "Cascade FLCN germline testing first-degree relatives"],
    "NF1":    ["Annual full-body clinical exam (skin + neuro + ophthalmology)", "Annual ophthalmology (Lisch nodules PATHOGNOMONIC + optic pathway glioma)", "Annual CT chest/abdomen (plexiform + MPNST + mediastinal NF)", "Annual MRI brain/spine if neurological symptoms (MPNST)", "Cascade NF1 germline testing first-degree relatives"],
}


def _make_patients(gene: str, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    g = next(g for g in ATLAS_GENES if g["gene"] == gene)
    tumours = _TUMOUR_TYPES.get(gene, ["NSCLC NOS"])
    variants = _VARIANTS_BY_GENE.get(gene, ["Pathogenic variant"])
    pts = []
    for i in range(n):
        age = (
            rng.randint(25, 45) if gene == "DICER1" else  # childhood/young adult PPB
            rng.randint(30, 55) if gene == "TP53" else    # LFS younger onset
            rng.randint(35, 75)
        )
        pts.append({
            "patient_id": f"{gene[:4]}-HLUNG-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_dx": age,
            "tumour_type": rng.choice(tumours),
            "variant": rng.choice(variants),
            "stage": rng.choice(["Localised", "Locally Advanced", "Metastatic"]),
            "egfr_tki_eligible": rng.random() < (
                0.90 if gene == "EGFR" else
                0.05
            ),
            "parpi_eligible": rng.random() < (
                0.70 if gene == "BRCA2" else
                0.15 if gene in ("TP53", "BAP1") else
                0.05
            ),
            "immunotherapy_resistant": rng.random() < (
                0.80 if gene == "STK11" else
                0.30
            ),
            "pneumothorax": rng.random() < (
                0.40 if gene == "FLCN" else
                0.05
            ),
            "radiation_avoided": rng.random() < (
                0.95 if gene == "TP53" else
                0.20
            ),
            "relapse": rng.random() < 0.38,
            "mpnst_risk": rng.random() < (
                0.11 if gene == "NF1" else
                0.02
            ),
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

    egfr_tki_rate = round(
        100 * sum(p["egfr_tki_eligible"] for pts in cohorts.values() for p in pts) / total, 1
    )
    parpi_rate = round(
        100 * sum(p["parpi_eligible"] for pts in cohorts.values() for p in pts) / total, 1
    )
    imm_resistant_rate = round(
        100 * sum(p["immunotherapy_resistant"] for pts in cohorts.values() for p in pts) / total, 1
    )
    pneumothorax_rate = round(
        100 * sum(p["pneumothorax"] for pts in cohorts.values() for p in pts) / total, 1
    )
    mean_age = round(
        sum(p["age_at_dx"] for pts in cohorts.values() for p in pts) / total, 1
    )

    return {
        "atlas": "Hereditary-Lung-Cancer-Predisposition-Atlas",
        "genes": _GENE_LIST,
        "total_patients": total,
        "gene_counts": gene_counts,
        "seed_range": f"{SEED_BASE}-{SEED_BASE+7}",
        "egfr_tki_eligible_rate_pct": egfr_tki_rate,
        "parpi_eligible_rate_pct": parpi_rate,
        "immunotherapy_resistant_rate_pct": imm_resistant_rate,
        "pneumothorax_rate_pct": pneumothorax_rate,
        "mean_age_at_dx": mean_age,
        "key_facts": [
            "EGFR germline: sensitizing mutations (E709K, R108K, T790M germline) — osimertinib 3rd-gen EGFR-TKI eligible; do NOT use 1st/2nd-gen TKI with germline T790M",
            "BRCA2 lung: 2-3x RR adenocarcinoma; HRD cisplatin-sensitive + PARPi eligible (olaparib FDA2020 post-platinum maintenance)",
            "TP53/LFS: lung 2-5x elevated; AVOID RADIATION ABSOLUTELY — radiation-field sarcoma catastrophic; WB-MRI Toronto Protocol MANDATORY annual",
            "STK11/PJS: NSCLC 16x RR HIGHEST hereditary lung risk; STK11+KRAS co-mutation = immunotherapy-COLD tumour; use chemotherapy + KRAS inhibitor",
            "DICER1: PPB type I/II/III PATHOGNOMONIC childhood; pulmonary blastoma adult PATHOGNOMONIC; annual CT from birth to age 8 MANDATORY",
            "BAP1/TPDS: mesothelioma 8-10% HIGHEST hereditary; ASBESTOS AVOIDANCE MANDATORY; annual ophthalmology (uveal melanoma 50% PATHOGNOMONIC)",
            "FLCN/BHD: bilateral pulmonary cysts PATHOGNOMONIC; pneumothorax 40% HIGHEST hereditary risk; pleurodesis after first episode strongly advised",
            "NF1: thoracic MPNST 8-13% PATHOGNOMONIC — NOT lung cancer (doxorubicin/ifosfamide NOT platinum); selumetinib FDA2020 plexiform neurofibroma",
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
                "lung_risk": g["lung_risk"],
                "pathognomonic": g["pathognomonic"],
                "key_avoid": g["key_avoid"],
                "key_rule": g["key_rule"],
                "surveillance": g["surveillance"],
                "targeted_rx": g["targeted_rx"],
            },
            "n": len(pts),
            "egfr_tki_eligible_pct": round(100 * sum(1 for p in pts if p["egfr_tki_eligible"]) / len(pts), 1),
            "parpi_eligible_pct": round(100 * sum(1 for p in pts if p["parpi_eligible"]) / len(pts), 1),
            "immunotherapy_resistant_pct": round(100 * sum(1 for p in pts if p["immunotherapy_resistant"]) / len(pts), 1),
            "pneumothorax_pct": round(100 * sum(1 for p in pts if p["pneumothorax"]) / len(pts), 1),
            "radiation_avoided_pct": round(100 * sum(1 for p in pts if p["radiation_avoided"]) / len(pts), 1),
            "relapse_pct": round(100 * sum(1 for p in pts if p["relapse"]) / len(pts), 1),
            "mpnst_risk_pct": round(100 * sum(1 for p in pts if p["mpnst_risk"]) / len(pts), 1),
            "mean_age": round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "top_tumour_types": [{"type": t, "count": c} for t, c in top_tumours],
            "top_variants": [{"variant": v, "count": c} for v, c in top_variants],
            "treatment_protocols": TREATMENT_PROTOCOLS_BY_GENE[gene],
            "surveillance_protocols": SURVEILLANCE_BY_GENE[gene],
        }
    return {"breakdown": breakdown, "genes": _GENE_LIST}


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Lung-Cancer-Predisposition-Atlas",
        "definitions": {
            "egfr_germline_tki_eligibility": (
                "EGFR germline: 1210aa RTK 7p11.2 GOF; sensitizing mutations (E709K, R108K, T790M germline) — familial NSCLC <1%; "
                "OSIMERTINIB 3RD-GEN EGFR-TKI: drug of choice for germline T790M (covers acquired resistance mechanism at baseline); "
                "Do NOT use 1st/2nd-gen TKI (gefitinib/erlotinib/afatinib) if T790M germline confirmed — primary resistance; "
                "Annual low-dose CT from age 30; cascade first-degree germline testing; lung adenocarcinoma exclusive histology."
            ),
            "brca2_hrd_parpi": (
                "BRCA2 lung: 3418aa FANCD1 HR mediator 13q12.3 LOF; lung adenocarcinoma 2-3x RR; "
                "HRD = homologous recombination deficiency — cisplatin/carboplatin hypersensitivity (platinum-induced ICL cannot be repaired); "
                "PARPI MAINTENANCE: olaparib FDA2020 post-platinum response — MANDATORY HRD testing (Myriad myChoice or Foundation); "
                "Genomic scar signature 3 (HRD) — BRCA2 lung tumours identifiable by mutational signature."
            ),
            "tp53_lfs_radiation_avoidance": (
                "TP53/LFS lung: 393aa p53 tumour suppressor 17p13.1 LOF; lung cancer 2-5x elevated; "
                "AVOID RADIATION ABSOLUTELY: radiation-field sarcoma catastrophic in LFS — radiation-induced secondary cancers documented; "
                "WB-MRI TORONTO PROTOCOL: annual whole-body MRI + brain MRI MANDATORY LFS surveillance (no ionising radiation); "
                "Hotspot dominant-negative mutations (R175H, R248W, R273H) sequester WT p53 — GOF properties; "
                "Median LFS lung cancer onset 30-35yr — much younger than sporadic (60+yr)."
            ),
            "stk11_pjs_nsclc_immunotherapy_resistance": (
                "STK11/PJS lung: 433aa LKB1 kinase 19p13.3 LOF; NSCLC 16x RR HIGHEST hereditary lung risk; "
                "STK11+KRAS co-mutation: COLD TUMOUR — PD-L1 negative, immunotherapy-RESISTANT due to neutrophil/MDM2 immunosuppressive microenvironment; "
                "MANAGEMENT IMPLICATION: DO NOT use PD-1/PD-L1 immunotherapy alone in STK11+KRAS NSCLC; use platinum chemotherapy + KRAS inhibitor (sotorasib/adagrasib); "
                "Mucocutaneous perioral macules PATHOGNOMONIC PJS — annual low-dose CT from age 25."
            ),
            "dicer1_ppb_pathognomonic": (
                "DICER1 syndrome: 1922aa RNase-III endoribonuclease 14q32.13 LOF; PPB type I (cystic)/II (mixed)/III (solid) PATHOGNOMONIC childhood lung malignancy; "
                "TWO-HIT MODEL: germline LOF (exon) + somatic hotspot RNase IIIb D1709/E1705 (metal-binding site); "
                "ANNUAL CT FROM BIRTH TO AGE 8: MANDATORY in DICER1 families — PPB type I → III progression documented; "
                "Do NOT confuse Type I PPB cysts with benign CPAM — molecular testing MANDATORY if DICER1 family history; "
                "Adult pulmonary blastoma = PATHOGNOMONIC when DICER1 germline confirmed."
            ),
            "bap1_mesothelioma_asbestos": (
                "BAP1-TPDS: 729aa deubiquitinase 3p21.1 LOF; mesothelioma 8-10% HIGHEST hereditary risk; "
                "ASBESTOS SYNERGY: germline BAP1 + asbestos co-exposure = 100x MPM risk vs general population — ASBESTOS AVOIDANCE MANDATORY for all carriers; "
                "UVEAL MELANOMA 50% PATHOGNOMONIC: annual ophthalmology MANDATORY; "
                "BAPomas/MBAITs skin PATHOGNOMONIC — triggers cascade BAP1 germline testing; "
                "BAP1-IHC nuclear loss = universal SDHx/BAP1 marker on mesothelioma specimens."
            ),
            "flcn_bhd_pneumothorax": (
                "FLCN/BHD: 579aa folliculin 17p11.2 LOF; bilateral basal pulmonary cysts PATHOGNOMONIC (80-90% carriers); "
                "PNEUMOTHORAX 40% HIGHEST HEREDITARY RISK: cyst rupture — pleurodesis surgical after FIRST episode strongly advised; "
                "Do NOT confuse BHD cysts with LAM (LAM = women of childbearing age, chylous effusion, sirolimus-responsive vs BHD = both sexes, no chylous effusion); "
                "Fibrofolliculomas/trichodiscomas PATHOGNOMONIC skin — triggers FLCN cascade testing; "
                "mTORC1 dysregulation via AMPK: everolimus experimental for FLCN-related RCC."
            ),
            "nf1_mpnst_vs_nsclc": (
                "NF1 lung: 2839aa neurofibromin RAS-GAP 17q11.2 LOF; NSCLC 2-3x elevated; "
                "MPNST CRITICAL DISTINCTION: thoracic MPNST (8-13% NF1 lifetime) MIMICS lung mass on CT — "
                "MPNST = SARCOMA (doxorubicin/ifosfamide, NOT platinum); do NOT treat as primary lung cancer; "
                "SELUMETINIB FDA2020: MEK inhibitor for plexiform neurofibroma; "
                "NF1-somatic mutations in 11% SPORADIC NSCLC (acquired — NOT NF1 germline syndrome); "
                "Café-au-lait macules 6+ PATHOGNOMONIC NF1; Lisch nodules iris PATHOGNOMONIC."
            ),
        },
        "key_clinical_distinctions": [
            "EGFR germline T790M: osimertinib ONLY (1st/2nd-gen TKI = primary resistance). E709K/R108K = sensitizing = all-gen TKI eligible",
            "BRCA2 lung: HRD cisplatin-sensitive → PARPi maintenance post-platinum MANDATORY HRD test first",
            "TP53/LFS: AVOID RADIATION ABSOLUTELY — WB-MRI Toronto annual. Hotspot R175H/R248W = dominant-negative GOF oncogene activity",
            "STK11+KRAS co-mutant NSCLC: PD-L1 cold, immunotherapy-RESISTANT — chemotherapy + KRAS inhibitor, NOT immunotherapy alone",
            "DICER1 PPB: annual CT birth to age 8 MANDATORY. Type I → III progression. Two-hit: germline LOF + somatic D1709/E1705 hotspot",
            "BAP1: asbestos MANDATORY avoided (100x synergistic MPM). Annual ophthalmology (uveal 50%). BAPomas = PATHOGNOMONIC cascade trigger",
            "FLCN BHD: bilateral pulmonary cysts ≠ LAM (distinguish clinically). Pneumothorax 40% — pleurodesis first episode",
            "NF1 MPNST vs NSCLC: thoracic MPNST = sarcoma (doxorubicin/ifosfamide) NOT lung cancer (platinum). Selumetinib FDA2020 plexiform NF",
        ],
    }
