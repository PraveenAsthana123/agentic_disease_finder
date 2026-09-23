#!/usr/bin/env python3
"""Hereditary-Pancreatic-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
BRCA2   (BRCA2 homologous recombination scaffold; 3418aa; 13q12.3; AD LOF;
         HBOC / Fanconi Anaemia D1 (biallelic);
         pancreatic cancer 5-7% lifetime; relative risk 3.5-10x;
         olaparib FDA 2019 POLO trial BRCA1/2 germline pancreatic;
         platinum-based chemo (HRD sensitivity);
         annual pancreatic MRI/EUS from 50;
         seed SEED_BASE+0) .
CDKN2A  (Cyclin-dependent kinase inhibitor 2A p16/INK4A; 156aa; 9p21.3; AD LOF;
         FAMMM (Familial Atypical Multiple Mole Melanoma);
         pancreatic cancer 17-39% lifetime -- HIGHEST CDKN2A germline risk;
         p16 IHC loss PATHOGNOMONIC in PDAC;
         annual MRI/EUS from 40 MANDATORY;
         palbociclib/CDK4-6i investigational;
         seed SEED_BASE+1) .
ATM     (Ataxia-Telangiectasia Mutated; 3056aa; 11q22.3; AR + AD LOF;
         Ataxia-Telangiectasia (biallelic) / Hereditary ATM (monoallelic AD LOF);
         pancreatic cancer 5-8x elevated monoallelic;
         RADIOSENSITIVITY ABSOLUTE CI in biallelic;
         olaparib / ceralasertib ATRi;
         annual pancreatic MRI/EUS from 50 monoallelic;
         seed SEED_BASE+2) .
PALB2   (Partner and localiser of BRCA2; 1186aa; 16p12.2; AD LOF;
         HBOC type 2 (monoallelic) / Fanconi Anaemia N (biallelic);
         pancreatic cancer 2-3% lifetime; 3-4x elevated;
         BRCA1-BRCA2 chromatin bridge; FA-N biallelic BRCA2 phenocopy;
         olaparib investigational; breast 53% lifetime (TBCRC048);
         seed SEED_BASE+3) .
STK11   (Serine/threonine kinase 11 LKB1; 433aa; 19p13.3; AD LOF;
         Peutz-Jeghers Syndrome (PJS);
         pancreatic cancer 11-36% lifetime -- HIGHEST single hereditary syndrome;
         mucocutaneous melanin macules PATHOGNOMONIC; SCTAT ovary PATHOGNOMONIC;
         annual MRI/EUS from AGE 30 -- earliest hereditary onset;
         seed SEED_BASE+4) .
MLH1    (MutL homologue 1; 756aa; 3p22.2; AD LOF;
         Lynch Syndrome type 1 / CMMRD (biallelic);
         pancreatic cancer 3.7-6% lifetime; 3-4x elevated;
         MSI-H PATHOGNOMONIC; BRAF V600E absent DDx sporadic methylation;
         pembrolizumab FDA 2017 any MSI-H/dMMR tumour; aspirin 600mg CAPP2;
         seed SEED_BASE+5) .
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome;
         R337H Brazilian founder associated with pancreatic and adrenal;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually;
         APR-246/eprenetapopt p53 reactivator investigational;
         seed SEED_BASE+6) .
PRSS1   (Cationic trypsinogen; 247aa; 7q34; AD GOF;
         Hereditary Pancreatitis (HP);
         pancreatic cancer 40-55% lifetime -- HIGHEST single gene cumulative;
         NO SMOKING ABSOLUTELY -- multiplies PC risk 40x in HP context;
         recurrent acute pancreatitis childhood PATHOGNOMONIC;
         annual pancreatic imaging from 40yr; TPIAT for intractable pain;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3342-3349)
"""
import random

SEED_BASE = 3342

ATLAS_GENES = [
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-HBOC-FA-D1-"
            "Pancreatic-Cancer-5-7pct-3.5-10x-RR-"
            "Olaparib-FDA2019-POLO-Trial-Platinum-Sensitive-"
            "Annual-Pancreatic-MRI-EUS-from-50-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 encodes breast cancer susceptibility protein 2: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; nuclear scaffold protein for homologous recombination (HR); "
            "  N-terminal transactivation domain; "
            "  8 BRC repeats (aa 1002-2082): RAD51 binding — BRCA2 loads RAD51 onto ssDNA; "
            "  OB folds (aa 2402-3186): ssDNA binding domain; "
            "  C-terminal RAD51 binding motif + nuclear localisation signal; "
            "  BRCA2 LOF → RAD51 cannot load → HR repair fails → NHEJ errors → chromosomal instability; "
            "  PALB2 binding domain (aa 10-40): anchors BRCA2 to chromatin at DSB sites via PALB2; "
            "HBOC (BRCA2 monoallelic — germline LOF): "
            "  Pancreatic ductal adenocarcinoma (PDAC): 5-7% lifetime vs 1.5% general (3.5-10x RR); "
            "  Breast cancer (female): 69-85% lifetime; breast cancer (male): 8-10% lifetime; "
            "  Ovarian cancer: 18-27% lifetime (high-grade serous); "
            "  Prostate cancer: 30-40% lifetime (aggressive, early-onset); "
            "  Annual pancreatic MRI/EUS from age 50 (or 10yr before youngest affected relative); "
            "  Annual breast MRI + mammography from age 30 (women); "
            "FANCONI ANAEMIA D1 (biallelic BRCA2): "
            "  Biallelic — PALB2 FA-D1 (most severe FA complementation group); "
            "  Childhood tumours: medulloblastoma, Wilms tumour, AML (very high risk); "
            "  AVOID ALKYLATING AGENTS in biallelic FA-D1 (mitomycin C, cyclophosphamide); "
            "  Bone marrow failure: median onset age 5yr; "
            "BRCA2 PANCREATIC CANCER SPECIFICS: "
            "  PDAC histology: standard ductal adenocarcinoma; "
            "  Platinum sensitivity due to HRD (homologous recombination deficiency); "
            "  Gemcitabine + cisplatin preferred first-line over FOLFIRINOX in BRCA2; "
            "  OLAPARIB (PARP inhibitor): FDA 2019 POLO trial — germline BRCA1/2 pancreatic; "
            "  Maintenance olaparib after platinum response: median PFS 7.4 vs 3.8 mo (POLO); "
            "  RUCAPARIB: investigational pancreatic BRCA2; "
            "SURVEILLANCE BRCA2 PANCREATIC: "
            "  Annual pancreatic MRI/EUS from age 50 (or 10yr before youngest relative); "
            "  Annual breast MRI + mammography from 30 (women); "
            "  Annual prostate PSA from age 40 (men); "
            "  Annual CA-19-9 as adjunct (not screening alone — low sensitivity); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "HBOC (Hereditary Breast–Ovarian Cancer) / Fanconi Anaemia D1 (biallelic)",
        "pc_risk": "Pancreatic cancer 5-7% lifetime; 3.5-10x RR; olaparib FDA 2019 maintenance",
        "pathognomonic": "Biallelic FA-D1: childhood medulloblastoma + Wilms + AML; PDAC in HBOC family",
        "key_avoid": "ALKYLATING AGENTS in biallelic FA-D1; cisplatin preferred over cyclophosphamide",
        "surveillance": "Annual pancreatic MRI/EUS from 50; annual breast MRI from 30; annual prostate PSA (men) from 40",
        "targeted_rx": "Olaparib FDA 2019 (POLO trial) germline BRCA2 pancreatic; platinum-based (HRD sensitivity)",
        "key_rule": "OLAPARIB MAINTENANCE FDA APPROVED — after platinum response in germline BRCA1/2 PDAC; annual MRI/EUS from 50",
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 Autosomal-Dominant-LOF -- 156aa -- "
            "p16-INK4A-17kDa-CDK4-6-Inhibitor-FAMMM-"
            "Pancreatic-17-39pct-HIGHEST-CDKN2A-Germline-Risk-"
            "Annual-MRI-EUS-from-40-MANDATORY-"
            "Melanoma-25-36x-CDK4-6i-Palbociclib-Investigational-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / 17 kDa / 9p21.3 CDKN2A encodes cyclin-dependent kinase inhibitor 2A (p16/INK4A): "
            "STRUCTURE: "
            "  156 aa / 17 kDa; ankyrin repeat domain protein; "
            "  4 ankyrin repeats (aa 8-155): CDK4/CDK6 binding pocket; "
            "  CDKN2A LOF → CDK4/6 cannot be inhibited → phospho-pRb → E2F transcription → G1/S bypass; "
            "  Alternative reading frame: p14ARF (exon 1β alternate) — regulates MDM2/TP53; "
            "  CDKN2A LOF also disrupts p14ARF → MDM2 uninhibited → TP53 degradation; "
            "  Biallelic deletion 9p21.3 (somatic): most common alteration in PDAC (~90-95%); "
            "FAMMM (Familial Atypical Multiple Mole Melanoma): "
            "  PANCREATIC CANCER RISK: 17-39% lifetime — HIGHEST CDKN2A germline risk for any single cancer; "
            "  Melanoma risk: 25-36x elevated lifetime; "
            "  Non-melanoma: oropharyngeal, upper GI, bladder cancer elevated; "
            "  p16 IHC loss PATHOGNOMONIC in PDAC (somatic in 95% sporadic; germline in FAMMM context); "
            "CDKN2A PANCREATIC CANCER SPECIFICS: "
            "  PDAC is the cancer where germline CDKN2A confers highest absolute risk; "
            "  Onset typically 5-10yr earlier than sporadic PDAC; "
            "  Annual pancreatic MRI/EUS from age 40 (or 10yr before youngest relative); "
            "  No CDKN2A-specific approved targeted therapy for PDAC yet; "
            "  CDK4/6 INHIBITORS (palbociclib, ribociclib, abemaciclib): FDA approved breast cancer; "
            "  Investigational in CDKN2A-null PDAC: CDKN2A LOF predicts RESISTANCE to CDK4/6i (CDK4/6 already active); "
            "  Note: CDK4/6i may benefit CDKN2A-intact tumours with CDK4 amplification; "
            "MELANOMA CONCURRENT SURVEILLANCE: "
            "  Annual total body skin exam + dermoscopy (melanoma 25-36x elevated); "
            "  Avoid UV / tanning beds ABSOLUTELY; "
            "  Ophthalmology screening (uveal melanoma); "
            "SURVEILLANCE CDKN2A PANCREATIC: "
            "  Annual pancreatic MRI/EUS from age 40; "
            "  Annual total body skin exam + dermoscopy; "
            "  Ophthalmology screen uveal melanoma; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Familial Atypical Multiple Mole Melanoma (FAMMM)",
        "pc_risk": "Pancreatic cancer 17-39% lifetime — HIGHEST CDKN2A germline risk for any single cancer",
        "pathognomonic": "p16 IHC loss PATHOGNOMONIC in PDAC; FAMMM atypical nevi + melanoma + pancreatic cancer",
        "key_avoid": "DELAY PANCREATIC SURVEILLANCE — annual MRI/EUS from 40 is mandatory; UV ABSOLUTELY AVOID (melanoma)",
        "surveillance": "Annual pancreatic MRI/EUS from 40 MANDATORY; annual total body skin exam + dermoscopy; avoid UV",
        "targeted_rx": "Palbociclib/abemaciclib CDK4/6i investigational CDKN2A-null; FOLFIRINOX or gemcitabine+nab-paclitaxel",
        "key_rule": "ANNUAL PANCREATIC MRI/EUS FROM AGE 40 MANDATORY — 17-39% lifetime PDAC risk; concurrent melanoma 25-36x elevated",
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Autosomal-Recessive/Dominant-LOF -- 3056aa -- "
            "ATM-350kDa-PI3K-Like-Kinase-DNA-DSB-Sensor-"
            "Ataxia-Telangiectasia-Biallelic-AR-Hereditary-ATM-Monoallelic-AD-"
            "Pancreatic-5-8x-Monoallelic-RADIOSENSITIVITY-ABSOLUTE-Biallelic-"
            "Olaparib-Ceralasertib-ATRi-OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM encodes ataxia-telangiectasia mutated kinase: "
            "STRUCTURE: "
            "  3056 aa / 350 kDa; PI3K-like serine/threonine kinase; "
            "  HEAT repeats (aa 1-1960): protein-protein interactions, FAT domain; "
            "  FAT domain (aa 1960-2566): regulatory; "
            "  Kinase domain (aa 2712-2962): phosphorylates H2AX (γ-H2AX DSB marker), CHK2, BRCA1; "
            "  FATC domain (aa 2962-3056): essential for kinase activity; "
            "  ATM senses DNA double-strand breaks → autophosphorylation → MRN complex (MRE11-RAD50-NBS1); "
            "  ATM LOF → impaired DSB signalling → checkpoint failure → HR repair defect; "
            "ATAXIA-TELANGIECTASIA (biallelic AR): "
            "  Progressive cerebellar ataxia from age 2-3yr PATHOGNOMONIC; "
            "  Oculomotor apraxia PATHOGNOMONIC (horizontal saccade loss); "
            "  Bulbar telangiectasiae (conjunctival, cutaneous) PATHOGNOMONIC; "
            "  Combined immunodeficiency (B + T cell); recurrent sinopulmonary infections; "
            "  RADIOSENSITIVITY ABSOLUTE CI — biallelic ATM: standard RT doses cause catastrophic toxicity; "
            "  DNA repair impaired → ionising radiation → chromosomal catastrophe; "
            "  Lymphoma/leukaemia: 70-80% lifetime in A-T biallelic; "
            "HEREDITARY ATM (monoallelic AD LOF): "
            "  Pancreatic cancer: 5-8x elevated lifetime risk (monoallelic); "
            "  Breast cancer: 3-4x elevated (female monoallelic); "
            "  Gastric cancer, colorectal cancer: modest elevation; "
            "  Annual pancreatic MRI/EUS from age 50 (monoallelic); "
            "  Annual breast MRI from age 40 (monoallelic female); "
            "ATM PANCREATIC CANCER TREATMENT: "
            "  Platinum-based (HRD-like sensitivity in ATM-null); "
            "  OLAPARIB: BRCA-like sensitivity in ATM LOF (HRD via synthetic lethality); "
            "  CERALASERTIB (AZD6738): ATR inhibitor — ATM-null cells are ATR-dependent; "
            "  Olaparib + ceralasertib: synthetic lethal combination in ATM LOF; "
            "  AVOID RADIATION in biallelic carriers; monoallelic: standard RT doses appear tolerated; "
            "SURVEILLANCE ATM: "
            "  Annual pancreatic MRI/EUS from age 50 (monoallelic); "
            "  Annual breast MRI from age 40 (monoallelic female); "
            "  Annual AFP + LDH (lymphoma screen in A-T biallelic); "
        ),
        "inheritance": "AR (biallelic) + AD LOF (monoallelic)",
        "syndrome": "Ataxia-Telangiectasia (biallelic AR) / Hereditary ATM (monoallelic AD LOF)",
        "pc_risk": "Pancreatic cancer 5-8x elevated monoallelic; A-T biallelic: lymphoma 70-80% dominant risk",
        "pathognomonic": "Biallelic A-T: cerebellar ataxia + oculomotor apraxia + telangiectasiae PATHOGNOMONIC TRIAD",
        "key_avoid": "RADIOSENSITIVITY ABSOLUTE in biallelic ATM — standard RT doses cause chromosomal catastrophe",
        "surveillance": "Annual pancreatic MRI/EUS from 50 (monoallelic); annual breast MRI from 40 (monoallelic female)",
        "targeted_rx": "Olaparib (HRD-like ATM null); ceralasertib ATRi (synthetic lethality ATM-null); platinum-based",
        "key_rule": "RADIOSENSITIVITY ABSOLUTE in biallelic ATM — avoid standard RT; ceralasertib synthetic lethal for ATM LOF",
    },
    {
        "gene": "PALB2",
        "protein": (
            "PALB2 -- 16p12.2 Autosomal-Dominant-LOF -- 1186aa -- "
            "PALB2-131kDa-WD40-BRCA2-Bridge-HBOC-2-FA-N-"
            "Pancreatic-2-3pct-3-4x-Elevated-"
            "Breast-53pct-Lifetime-TBCRC048-Olaparib-82pct-ORR-"
            "Biallelic-FA-N-BRCA2-Phenocopy-OMIM-610355"
        ),
        "locus": "16p12.2",
        "protein_size": (
            "1186 aa / 131 kDa / 16p12.2 PALB2 encodes partner and localiser of BRCA2: "
            "STRUCTURE: "
            "  1186 aa / 131 kDa; adaptor/scaffold protein bridging BRCA1-BRCA2 at DNA damage sites; "
            "  N-terminal coiled-coil domain (aa 1-45): BRCA1 binding; "
            "  WD40 repeat domain (aa 853-1186): BRCA2 and RAD51C binding; "
            "  Central region (aa 200-850): chromatin association; "
            "  PALB2 ANCHORS BRCA2 to chromatin at DSB sites — loss → BRCA2 cannot reach DSBs; "
            "  PALB2 LOF → HR repair failure phenotypically identical to BRCA2 LOF; "
            "HBOC TYPE 2 (PALB2 monoallelic): "
            "  Breast cancer: 53% lifetime (female monoallelic — NEJM 2014 Antoniou); "
            "  Pancreatic cancer: 2-3% lifetime; 3-4x elevated; "
            "  Ovarian cancer: modest elevation (2-3x vs general); "
            "  TBCRC048 trial: olaparib 82% ORR in PALB2-mutant breast cancer — highest PARP response; "
            "  Annual breast MRI + mammography from age 30; "
            "  Annual pancreatic MRI/EUS from age 50; "
            "FANCONI ANAEMIA N (biallelic PALB2): "
            "  Biallelic LOF → FA-N complementation group; "
            "  Phenocopy of FA-D1 (BRCA2 biallelic): medulloblastoma, Wilms, leukaemia risk; "
            "  Bone marrow failure; AVOID ALKYLATING AGENTS in FA-N; "
            "PALB2 PANCREATIC CANCER: "
            "  PDAC histology: standard ductal; HRD phenotype (BRCA2-like); "
            "  Platinum sensitivity (gemcitabine + cisplatin preferred); "
            "  OLAPARIB: FDA for BRCA1/2; investigational for PALB2 pancreatic; "
            "  ATR inhibitors + olaparib: PALB2-null synthetic lethality investigated; "
            "SURVEILLANCE PALB2: "
            "  Annual pancreatic MRI/EUS from age 50; "
            "  Annual breast MRI + mammography from age 30 (women); "
            "  Annual CBC (FA-N biallelic marrow failure); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "HBOC type 2 (monoallelic) / Fanconi Anaemia N (biallelic)",
        "pc_risk": "Pancreatic cancer 2-3% lifetime; 3-4x elevated; breast 53% lifetime (TBCRC048 olaparib 82% ORR)",
        "pathognomonic": "BRCA1-BRCA2 chromatin bridge; FA-N biallelic = BRCA2 phenocopy (childhood brain + haematological)",
        "key_avoid": "ALKYLATING AGENTS in biallelic FA-N; annual pancreatic AND breast surveillance mandatory",
        "surveillance": "Annual pancreatic MRI/EUS from 50; annual breast MRI + mammography from 30",
        "targeted_rx": "Olaparib (FDA breast; investigational pancreatic); TBCRC048 olaparib 82% ORR PALB2 breast; platinum",
        "key_rule": "PALB2 BRIDGES BRCA1-BRCA2 — both pancreatic MRI/EUS (from 50) AND breast MRI (from 30) mandatory",
    },
    {
        "gene": "STK11",
        "protein": (
            "STK11 -- 19p13.3 Autosomal-Dominant-LOF -- 433aa -- "
            "LKB1-48kDa-AMPK-Master-Kinase-PJS-"
            "Pancreatic-11-36pct-HIGHEST-Hereditary-Syndrome-"
            "Annual-MRI-EUS-from-AGE-30-Earliest-All-Hereditary-"
            "Mucocutaneous-Macules-PATHOGNOMONIC-SCTAT-Ovary-PATHOGNOMONIC-OMIM-602216"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "433 aa / 48 kDa / 19p13.3 STK11 encodes serine/threonine kinase 11 (LKB1): "
            "STRUCTURE: "
            "  433 aa / 48 kDa; master serine/threonine kinase; AMPK kinase family; "
            "  N-terminal regulatory domain; "
            "  Kinase domain (aa 49-309): ATP binding site; activation loop T185/T189; "
            "  C-terminal farnesylation signal (aa 429-433): membrane localisation; "
            "  STK11 activates AMPK → mTORC1 suppression → metabolic checkpoint; "
            "  STK11 LOF → AMPK not activated → mTOR constitutive → proliferation; "
            "  STK11 LOF also → dysregulated KRAS-driven signalling synergy; "
            "  KRAS co-mutation + STK11 LOF: profound immunotherapy resistance in NSCLC; "
            "PEUTZ-JEGHERS SYNDROME (PJS): "
            "  PANCREATIC CANCER: 11-36% lifetime — HIGHEST single hereditary syndrome for pancreatic; "
            "  ANNUAL PANCREATIC MRI/EUS FROM AGE 30 — EARLIEST hereditary pancreatic onset of all syndromes; "
            "  Mucocutaneous melanin macules (lips, buccal mucosa, genital mucosa): PATHOGNOMONIC; "
            "  Hamartomatous polyps GI (stomach, small bowel, colon): benign but with cancer risk; "
            "  GI endoscopy from age 8yr MANDATORY; "
            "  SCTAT (sex cord tumour with annular tubules): ovary — PATHOGNOMONIC PJS; "
            "  Cervical adenoma malignum: rare; "
            "  Breast cancer: 45-50% lifetime (women PJS); "
            "STK11 NSCLC NOTE: "
            "  STK11 somatic LOF in NSCLC: profound resistance to anti-PD-1 immunotherapy; "
            "  KRAS-STK11 co-mutation: worst prognosis NSCLC; immunotherapy non-responder; "
            "  This is SOMATIC NSCLC context; germline PJS does not cause NSCLC preferentially; "
            "STK11 PANCREATIC CANCER: "
            "  PDAC in PJS: 11-36% lifetime; onset from age 30yr (earliest of all hereditary syndromes); "
            "  Pathology: standard PDAC; IPMN (intraductal papillary mucinous neoplasm) also elevated; "
            "  No specific targeted therapy; FOLFIRINOX or gemcitabine + nab-paclitaxel; "
            "  Annual MRI/EUS pancreas: preferred over CT (radiation concern, young onset age 30); "
            "SURVEILLANCE STK11: "
            "  Annual pancreatic MRI/EUS from age 30 MANDATORY (earliest of all); "
            "  GI endoscopy (upper + lower) from age 8yr; "
            "  Annual gynaecologic USS (SCTAT, cervical adenoma malignum); "
            "  Annual breast MRI + mammography from age 25 (women); "
            "  Annual testicular exam (Sertoli cell tumour risk males); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Peutz-Jeghers Syndrome (PJS)",
        "pc_risk": "Pancreatic cancer 11-36% lifetime — HIGHEST single hereditary syndrome; onset from age 30 (earliest)",
        "pathognomonic": "Mucocutaneous melanin macules lips/buccal/genital PATHOGNOMONIC; SCTAT ovary PATHOGNOMONIC PJS",
        "key_avoid": "DELAY PANCREATIC MRI/EUS — must start at age 30 (earliest onset of all hereditary pancreatic syndromes)",
        "surveillance": "Annual pancreatic MRI/EUS from 30 MANDATORY; GI endoscopy from 8yr; annual gynaecologic + breast",
        "targeted_rx": "FOLFIRINOX or gemcitabine + nab-paclitaxel (no specific targeted); immunotherapy resistance in KRAS/STK11",
        "key_rule": "ANNUAL PANCREATIC MRI/EUS FROM AGE 30 — STK11 has EARLIEST hereditary pancreatic onset; 11-36% lifetime risk",
    },
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 756aa -- "
            "MutL-Homologue1-85kDa-MMR-Scaffold-Lynch-Syndrome-Type1-CMMRD-Biallelic-"
            "Pancreatic-3.7-6pct-3-4x-MSI-H-PATHOGNOMONIC-"
            "Pembrolizumab-FDA2017-ANY-MSI-H-Aspirin-600mg-CAPP2-"
            "Muir-Torre-Sebaceous-PATHOGNOMONIC-OMIM-120436"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "756 aa / 85 kDa / 3p22.2 MLH1 encodes MutL homologue 1: "
            "STRUCTURE: "
            "  756 aa / 85 kDa; ATP-dependent endonuclease scaffold protein; "
            "  N-terminal ATPase domain (aa 1-335): binds ATP; regulates endonuclease activity; "
            "  Dimerisation linker (aa 336-501); "
            "  C-terminal dimerisation domain (aa 502-756): PMS2 binding — MutLα heterodimer; "
            "  MLH1 + PMS2 = MutLα (major repair complex); MLH1 + MLH3 = MutLγ; "
            "  MLH1 LOF → mismatch accumulation → microsatellite instability (MSI); "
            "  MMR loss → thousands of frameshifts in repeat tracts → neoantigen generation; "
            "LYNCH SYNDROME TYPE 1: "
            "  Pancreatic cancer: 3.7-6% lifetime Lynch (3-4x elevated vs 1.5% general); "
            "  Colorectal cancer: 40-80% lifetime (DOMINANT cancer Lynch); "
            "  Endometrial cancer: 25-60% lifetime (women); "
            "  Gastric cancer: 6-13% lifetime; "
            "  Ovarian cancer: 4-20% lifetime; "
            "  MSI-H status PATHOGNOMONIC Lynch (absent BRAF V600E DDx sporadic MLH1 methylation); "
            "  Muir-Torre variant: sebaceous adenomas/carcinomas PATHOGNOMONIC MLH1 Lynch; "
            "CMMRD (Constitutional Mismatch Repair Deficiency — biallelic MLH1): "
            "  Biallelic → cafe-au-lait-like patches (NF1 phenocopy); brain tumours childhood; "
            "  Colorectal and lymphoma risk 100% by age 25; "
            "MSI-H/dMMR TARGETED THERAPY — KEY CLINICAL RULE: "
            "  PEMBROLIZUMAB: FDA 2017 — FIRST EVER tumour-agnostic approval: any MSI-H/dMMR solid tumour; "
            "  Dostarlimab: FDA 2021 dMMR solid tumours; "
            "  ASPIRIN 600mg daily: CAPP2 trial — reduces Lynch syndrome cancer risk ~50% at 10yr; "
            "  Aspirin must be taken daily; CAPP2 confirmed at 10yr follow-up (Burn et al.); "
            "SURVEILLANCE MLH1: "
            "  Colonoscopy every 1-2yr from age 25yr; "
            "  Annual endometrial biopsy from age 35yr (women); "
            "  Annual pancreatic MRI/EUS from age 50yr; "
            "  Annual gastroscopy from age 30-35yr (gastric cancer risk); "
            "  Aspirin 600mg daily (CAPP2 protocol) after colonoscopy exclusion of existing polyps; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Lynch Syndrome type 1 / CMMRD (biallelic)",
        "pc_risk": "Pancreatic cancer 3.7-6% lifetime Lynch; 3-4x elevated; dominant risk is colorectal 40-80%",
        "pathognomonic": "MSI-H PATHOGNOMONIC Lynch; BRAF V600E absent DDx sporadic methylation; Muir-Torre sebaceous PATHOGNOMONIC",
        "key_avoid": "STANDARD CHEMO without MSI testing — pembrolizumab FDA approved any MSI-H/dMMR; test every PDAC",
        "surveillance": "Colonoscopy every 1-2yr from 25; annual endometrial from 35; annual pancreatic MRI/EUS from 50",
        "targeted_rx": "Pembrolizumab FDA2017 MSI-H any histology; dostarlimab; aspirin 600mg CAPP2 daily",
        "key_rule": "PEMBROLIZUMAB FDA 2017 — first tumour-agnostic approval any MSI-H/dMMR; test MSI on EVERY pancreatic tumour",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-"
            "R337H-Brazilian-Founder-Pancreatic-ACC-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-Annual-"
            "APR-246-Eprenetapopt-p53-Reactivator-MDM2i-Investigational-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes tumour protein p53: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; homotetrameric transcription factor; the genome guardian; "
            "  N-terminal transactivation domain (aa 1-42): MDM2 binding (E3 ubiquitin ligase feedback); "
            "  Proline-rich region (aa 40-90): apoptosis regulation; "
            "  DNA-binding domain (aa 94-292): majority of hotspot mutations cluster here; "
            "  Hotspots: R175H (conformational GOF), G245S, R248W, R248Q, R273H, R273C, R282W; "
            "  R337H: Brazilian founder variant — specific structural alteration; "
            "  Tetramerisation domain (aa 323-356): required for functional tetramer; "
            "  C-terminal regulatory (aa 356-393): post-translational modifications; "
            "  TP53 activates CDKN1A (p21) → G1/S arrest; PUMA/NOXA/BBC3 → apoptosis; "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  Near 100% penetrance by age 70; ~20% de novo germline mutations; "
            "  Core cancers: soft tissue sarcoma 50-60%; premenopausal breast <45yr 30%; "
            "  Brain (glioma, CPC) 15%; adrenocortical carcinoma child PATHOGNOMONIC; "
            "  Osteosarcoma; leukemia; "
            "TP53 R337H BRAZILIAN FOUNDER VARIANT: "
            "  R337H is a temperature-sensitive variant (structural alteration); "
            "  Brazilian founder allele — present in ~0.3% Southern Brazilian population; "
            "  Strongly associated with adrenocortical carcinoma (pediatric) AND pancreatic cancer; "
            "  R337H confers moderate TP53 LOF: lower penetrance than classical hotspot mutations; "
            "  Surveillance in R337H: annual WBMRI; annual pancreatic MRI from 40-45yr; "
            "AVOID RADIATION ABSOLUTELY: "
            "  TP53 germline LOF → impaired G1/S checkpoint → radiation → secondary malignancy in field; "
            "  Secondary sarcomas in prior RT fields documented in LFS families; "
            "  PREFER SURGERY over RT consolidation; MRI-based surveillance (avoid CT radiation); "
            "EMERGING THERAPIES: "
            "  APR-246 / eprenetapopt: p53 reactivator — converts mutant p53 to WT conformation; "
            "  MDM2 inhibitors (milademetan, HDM201): investigational; block MDM2-p53 interaction; "
            "  RG7388 / idasanutlin: MDM2i clinical trials; "
            "SURVEILLANCE LFS: "
            "  WBMRI Toronto protocol annually; brain MRI annually; "
            "  Annual breast MRI from age 20-25yr (women); "
            "  Annual abdominopelvic USS every 3-4 months <18yr (adrenocortical); "
            "  Annual pancreatic MRI from age 40-45yr (R337H / LFS with pancreatic family history); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Li-Fraumeni Syndrome (LFS)",
        "pc_risk": "Elevated in LFS; R337H Brazilian founder strongly associated with pancreatic and adrenal cancers",
        "pathognomonic": "ACC child PATHOGNOMONIC LFS; CPC child PATHOGNOMONIC; R337H Brazilian founder → pancreatic + ACC",
        "key_avoid": "RADIATION — AVOID RADIATION ABSOLUTELY in germline TP53; use MRI not CT for surveillance",
        "surveillance": "WBMRI Toronto annually; annual brain MRI; annual breast MRI from 20-25; annual pancreatic MRI from 40 (R337H)",
        "targeted_rx": "Surgery preferred over RT; APR-246/eprenetapopt p53 reactivator investigational; MDM2i investigational",
        "key_rule": "AVOID RADIATION ABSOLUTELY — TP53 germline + radiation → secondary sarcoma/carcinoma; R337H → pancreatic surveillance",
    },
    {
        "gene": "PRSS1",
        "protein": (
            "PRSS1 -- 7q34 Autosomal-Dominant-GOF -- 247aa -- "
            "Cationic-Trypsinogen-26kDa-Serine-Protease-Hereditary-Pancreatitis-"
            "Pancreatic-40-55pct-Lifetime-HIGHEST-Single-Gene-Cumulative-"
            "NO-SMOKING-ABSOLUTELY-40x-Multiplier-"
            "R122H-p.Arg122His-Most-Common-GOF-TPIAT-OMIM-167800"
        ),
        "locus": "7q34",
        "protein_size": (
            "247 aa / 26 kDa / 7q34 PRSS1 encodes cationic trypsinogen (trypsinogen-1): "
            "STRUCTURE: "
            "  247 aa / 26 kDa; serine endoprotease precursor; "
            "  Signal peptide (aa 1-15): ER targeting; "
            "  Activation peptide (aa 16-24): trypsinogen → trypsin by enterokinase/trypsin; "
            "  Serine protease domain (aa 25-247): catalytic triad H57-D102-S195; "
            "  Ca2+ binding site (aa 77-79): stabilises trypsin structure; "
            "  Autolysis site R122: cleavage here inactivates trypsin (protective); "
            "  PRSS1 p.R122H GOF: autolysis site destroyed → trypsin cannot self-inactivate → "
            "  premature trypsinogen activation in pancreatic duct → recurrent pancreatitis; "
            "  p.N29I GOF: promotes trypsinogen activation; second most common HP variant; "
            "  p.A16V: signal peptide variant; promotes premature trypsinogen targeting; "
            "HEREDITARY PANCREATITIS (HP): "
            "  PANCREATIC CANCER: 40-55% LIFETIME CUMULATIVE — HIGHEST of all hereditary pancreatic genes; "
            "  Onset pancreatitis: childhood (mean age 10yr); recurrent acute episodes; "
            "  Chronic pancreatitis progression: fibrosis, calcifications, exocrine/endocrine failure; "
            "  Recurrent acute pancreatitis childhood onset PATHOGNOMONIC for HP; "
            "  Calcifications on imaging (CT/MRCP) — characteristic; "
            "NO SMOKING ABSOLUTELY: "
            "  Smoking multiplies lifetime PDAC risk 40-FOLD in hereditary pancreatitis context; "
            "  Smoking cessation is the SINGLE most impactful modifiable risk factor in HP; "
            "  Alcohol avoidance also mandatory; "
            "ANNUAL PANCREATIC IMAGING FROM AGE 40: "
            "  Annual MRI/MRCP + EUS from age 40yr (pancreatic cancer surveillance); "
            "  Total pancreatectomy with islet autotransplantation (TPIAT): for intractable pain; "
            "  TPIAT preserves endocrine function (autologous islets re-infused); "
            "  Pancreatic enzyme replacement therapy (PERT) for exocrine insufficiency; "
            "CONCURRENT RISKS: "
            "  Diabetes mellitus: high risk from recurrent pancreatitis (endocrine damage); "
            "  Annual HbA1c from age 30yr; "
            "  CFTR modifier testing: CFTR variants co-operate with PRSS1 in HP (compound HP); "
            "SURVEILLANCE PRSS1: "
            "  Annual pancreatic MRI/MRCP + EUS from age 40yr; "
            "  Annual HbA1c (diabetes from exocrine/endocrine failure); "
            "  CFTR modifier testing at diagnosis; "
            "  Smoking cessation ABSOLUTE priority; alcohol avoidance; "
            "  Pancreatic enzyme replacement (PERT) as needed for exocrine insufficiency; "
        ),
        "inheritance": "AD GOF",
        "syndrome": "Hereditary Pancreatitis (HP)",
        "pc_risk": "Pancreatic cancer 40-55% lifetime cumulative — HIGHEST single gene; smoking multiplies risk 40x",
        "pathognomonic": "Recurrent acute pancreatitis childhood onset PATHOGNOMONIC; calcifications imaging; NO smoking ABSOLUTELY",
        "key_avoid": "SMOKING ABSOLUTELY — multiplies PDAC risk 40-fold in HP; alcohol avoidance also mandatory",
        "surveillance": "Annual pancreatic MRI/EUS from 40; annual HbA1c; CFTR testing; smoking cessation ABSOLUTE priority",
        "targeted_rx": "FOLFIRINOX or gemcitabine+nab-paclitaxel; PERT for exocrine insufficiency; TPIAT intractable pain",
        "key_rule": "NO SMOKING ABSOLUTELY in hereditary pancreatitis — 40x PDAC risk multiplier; annual MRI/EUS from age 40",
    },
]

# Tumour types per gene
TUMOUR_TYPES_BY_GENE: dict = {
    "BRCA2":  ["Pancreatic Ductal Adenocarcinoma (PDAC)", "PDAC + Olaparib Maintenance", "PDAC Platinum-Sensitive HRD", "Breast Cancer (HBOC)", "Ovarian Cancer HGSOC", "PDAC + Cisplatin Response"],
    "CDKN2A": ["Pancreatic Ductal Adenocarcinoma (PDAC)", "PDAC FAMMM Context", "Cutaneous Melanoma (FAMMM)", "PDAC + Melanoma Concurrent", "Uveal Melanoma (FAMMM)"],
    "ATM":    ["Pancreatic Ductal Adenocarcinoma (PDAC)", "PDAC ATM-null HRD", "Breast Cancer Monoallelic ATM", "PDAC + Olaparib Ceralasertib", "Lymphoma A-T Biallelic"],
    "PALB2":  ["Pancreatic Ductal Adenocarcinoma (PDAC)", "Breast Cancer PALB2 (53% lifetime)", "PDAC Platinum-Sensitive", "PDAC + Olaparib Investigational", "FA-N Biallelic Childhood Tumour"],
    "STK11":  ["Pancreatic Ductal Adenocarcinoma (PJS)", "PDAC IPMN-Associated PJS", "Breast Cancer PJS (45-50%)", "SCTAT Ovary (PATHOGNOMONIC)", "GI Hamartomatous Polyp PJS"],
    "MLH1":   ["Pancreatic Ductal Adenocarcinoma MSI-H", "Colorectal Cancer Lynch (dominant)", "Endometrial Cancer Lynch", "PDAC + Pembrolizumab Response", "Gastric Cancer Lynch"],
    "TP53":   ["Pancreatic Ductal Adenocarcinoma (R337H)", "Adrenocortical Carcinoma Child (LFS)", "Osteosarcoma (LFS)", "Breast Cancer Premenopausal LFS", "Brain Tumour (CPC/Glioma) LFS"],
    "PRSS1":  ["Pancreatic Ductal Adenocarcinoma (HP)", "Chronic Pancreatitis → PDAC", "PDAC Smoking-Accelerated HP", "PDAC + Exocrine Insufficiency", "PDAC Early Onset HP (age 40-55)"],
}

# Pathogenic variants per gene
VARIANTS_BY_GENE: dict = {
    "BRCA2":  ["p.Trp3189Ter (protein truncating)", "p.Ser1982Arg (BRC repeat 7)", "p.Lys3326Ter (founder-like, attenuated)", "p.Glu1308Ter (exon 10)", "13q12.3 large deletion (MLPA)", "c.9382del (frameshift exon 23)", "p.Gln2829Ter"],
    "CDKN2A": ["p.Gly101Trp (p16 hotspot Europe)", "p.Ala148Thr (FAMMM)", "p.Val118Asp (ankyrin repeat)", "9p21.3 deletion (MLPA)", "c.67+1G>A (splice donor)", "p.Pro48Thr (FAMMM)"],
    "ATM":    ["p.Arg3008Ter (truncating monoallelic)", "p.Ser707Leu (missense checkpoint)", "p.Val2718Ala (kinase domain)", "11q22.3 deletion (MLPA)", "c.3161del (frameshift)", "p.Arg2872Ter"],
    "PALB2":  ["p.Tyr1183Ter (WD40 truncating)", "p.Leu939Trp (WD40 missense)", "c.2816delA (frameshift)", "16p12.2 deletion (MLPA)", "p.Lys940Ter (truncating)", "c.3113+1G>A (splice)"],
    "STK11":  ["p.Gln369Ter (kinase domain stop)", "p.Asp194Tyr (activation loop)", "STK11 exon 1-10 deletion (MLPA)", "c.1062+1G>A (splice)", "p.Phe354Leu (kinase missense)", "p.Glu170Lys"],
    "MLH1":   ["p.Val384Asp (missense pathogenic)", "p.Arg265Cys (MMR defect)", "3p22.2 deletion (MLPA)", "c.1852_1854del (in-frame del)", "c.676C>T p.Arg226Ter", "c.1852G>A splice promoter"],
    "TP53":   ["p.Arg175His (DBD hotspot GOF)", "p.Arg337His (Brazilian founder)", "p.Arg248Trp (hotspot)", "p.Gly245Ser (hotspot)", "p.Arg273His (hotspot)", "c.559+1G>T (splice IVS5)"],
    "PRSS1":  ["p.Arg122His (R122H most common HP GOF)", "p.Asn29Ile (N29I second most common)", "p.Ala16Val (signal peptide)", "p.Arg122Cys (autolysis site variant)", "PRSS1 p.Lys23Arg", "p.Glu79Lys (Ca2+ binding)"],
}

# Treatment protocols per gene
TREATMENT_PROTOCOLS_BY_GENE: dict = {
    "BRCA2":  ["Olaparib maintenance FDA 2019 (POLO trial) after platinum response", "Gemcitabine + cisplatin (platinum-based HRD sensitivity)", "FOLFIRINOX (alternative first-line PDAC)", "Rucaparib investigational pancreatic BRCA2", "Annual pancreatic MRI/EUS from age 50", "Annual breast MRI + mammography from 30"],
    "CDKN2A": ["FOLFIRINOX or gemcitabine + nab-paclitaxel (standard PDAC)", "Palbociclib CDK4/6i investigational CDKN2A-null", "Annual pancreatic MRI/EUS from age 40 MANDATORY", "Annual total body skin exam + dermoscopy (melanoma)", "UV avoidance ABSOLUTELY (melanoma 25-36x)", "Uveal melanoma screening ophthalmology"],
    "ATM":    ["Olaparib (HRD-like synthetic lethality ATM null)", "Ceralasertib (ATRi) + olaparib synthetic lethality ATM-null", "Gemcitabine + cisplatin (platinum-sensitive)", "Annual pancreatic MRI/EUS from age 50 (monoallelic)", "Annual breast MRI from 40 (monoallelic female)", "AVOID RADIATION in biallelic ATM"],
    "PALB2":  ["Olaparib FDA breast; investigational pancreatic PALB2", "Gemcitabine + cisplatin (HRD platinum-sensitive)", "Annual pancreatic MRI/EUS from age 50", "Annual breast MRI + mammography from 30 (women)", "ATR inhibitor + olaparib PALB2-null investigational", "AVOID ALKYLATING in FA-N biallelic"],
    "STK11":  ["FOLFIRINOX (first-line PJS PDAC)", "Gemcitabine + nab-paclitaxel (alternative PDAC)", "Annual pancreatic MRI/EUS from age 30 MANDATORY", "GI endoscopy (upper + lower) from age 8yr", "Annual gynaecologic USS (SCTAT, cervical)", "Annual breast MRI + mammography from 25 (women)"],
    "MLH1":   ["Pembrolizumab FDA 2017 any MSI-H/dMMR (first tumour-agnostic approval)", "Dostarlimab FDA 2021 dMMR solid tumours", "Aspirin 600mg daily CAPP2 trial (50% Lynch cancer reduction)", "Colonoscopy every 1-2yr from age 25yr", "Annual endometrial biopsy from age 35yr (women)", "Annual pancreatic MRI/EUS from age 50yr"],
    "TP53":   ["Surgery preferred over RT (AVOID RADIATION ABSOLUTELY)", "APR-246/eprenetapopt p53 reactivator investigational", "MDM2 inhibitors (milademetan, idasanutlin) investigational", "WBMRI Toronto protocol annually", "Annual pancreatic MRI from 40-45 (R337H / LFS family history)", "Annual breast MRI from 20-25 (women)"],
    "PRSS1":  ["FOLFIRINOX or gemcitabine + nab-paclitaxel (standard PDAC)", "Pancreatic enzyme replacement therapy (PERT) exocrine insufficiency", "TPIAT (total pancreatectomy with islet autotransplantation) intractable pain", "Annual pancreatic MRI/MRCP + EUS from age 40", "Annual HbA1c monitoring (diabetes risk)", "Smoking cessation ABSOLUTE priority"],
}

# Surveillance protocols per gene
SURVEILLANCE_BY_GENE: dict = {
    "BRCA2":  ["Annual pancreatic MRI/EUS from age 50 (or 10yr before youngest relative)", "Annual breast MRI + mammography from age 30 (women)", "Annual prostate PSA from age 40 (men)", "Annual CA-19-9 as adjunct (low sensitivity alone)", "Annual ovarian USS + CA-125 (women with oophorectomy deferred)", "Genetic cascade testing first-degree relatives"],
    "CDKN2A": ["Annual pancreatic MRI/EUS from age 40 MANDATORY", "Annual total body skin exam + dermoscopy (melanoma 25-36x)", "Annual ophthalmology screen (uveal melanoma)", "Avoid UV + tanning beds ABSOLUTELY", "Annual GI endoscopy from age 50 (oropharyngeal/upper GI risk)", "Genetic cascade testing first-degree relatives"],
    "ATM":    ["Annual pancreatic MRI/EUS from age 50 (monoallelic)", "Annual breast MRI from age 40 (monoallelic female)", "Annual AFP + CBC (A-T biallelic lymphoma surveillance)", "Physical therapy (cerebellar ataxia A-T biallelic)", "Annual immunoglobulin levels (A-T biallelic immunodeficiency)", "Annual gastric/colorectal endoscopy from age 50"],
    "PALB2":  ["Annual pancreatic MRI/EUS from age 50", "Annual breast MRI + mammography from age 30 (women)", "Annual ovarian USS (modest ovarian cancer elevation)", "CBC monitoring (FA-N biallelic marrow failure)", "Genetic cascade testing first-degree relatives", "Annual prostate PSA from age 45 (men PALB2)"],
    "STK11":  ["Annual pancreatic MRI/EUS from age 30 MANDATORY (earliest onset)", "GI endoscopy upper + lower from age 8yr", "Annual gynaecologic USS + PAP (SCTAT ovary; cervical adenoma)", "Annual breast MRI + mammography from age 25 (women)", "Annual testicular exam (Sertoli cell tumour males)", "Annual CA-19-9 as adjunct pancreatic surveillance"],
    "MLH1":   ["Colonoscopy every 1-2yr from age 25yr", "Annual endometrial biopsy from age 35yr (women)", "Annual pancreatic MRI/EUS from age 50yr", "Annual gastroscopy from age 30-35yr (gastric cancer)", "Aspirin 600mg daily CAPP2 (after exclusion existing polyps)", "Annual urinary cytology from age 30-35yr (urothelial Lynch)"],
    "TP53":   ["WBMRI Toronto protocol annually", "Annual brain MRI (glioma/CPC surveillance)", "Annual thyroid USS", "Annual breast MRI from age 20-25yr (women)", "Annual abdominopelvic USS every 3-4 months <18yr (adrenocortical)", "Annual pancreatic MRI from age 40-45yr (R337H / LFS + PC family history)"],
    "PRSS1":  ["Annual pancreatic MRI/MRCP + EUS from age 40yr", "Annual HbA1c (diabetes risk from recurrent pancreatitis)", "CFTR modifier testing at diagnosis", "Smoking cessation support — ABSOLUTE priority", "Alcohol avoidance counselling", "Annual CA-19-9 as adjunct imaging surveillance"],
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

    age_at_dx = rng.randint(28, 76)
    tumour_type = rng.choice(tumour_types)
    variant = rng.choice(variants)
    treatment = rng.choice(treatments)
    cr = rng.random() < 0.54
    radiation = rng.random() < (0.04 if gene in ("TP53", "ATM") else 0.18)
    relapse = rng.random() < 0.38 if cr else rng.random() < 0.62

    return {
        "patient_id": f"HPC-{gene}-{patient_idx:03d}",
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
        "atlas": "Hereditary-Pancreatic-Cancer-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Pancreatic Cancer Predisposition Reference — BRCA2-CDKN2A-ATM-PALB2-STK11-MLH1-TP53-PRSS1",
        "seeds": f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
        "total_patients": n,
        "cr_pct": round(100 * cr_n / n, 1),
        "radiation_pct": round(100 * rad_n / n, 1),
        "relapse_pct": round(100 * relapse_n / n, 1),
        "mean_age_at_dx": mean_age,
        "genes": _GENE_LIST,
        "gene_summary": gene_summary,
        "key_rules": [
            "OLAPARIB MAINTENANCE FDA APPROVED (BRCA2) — POLO trial: germline BRCA2 PDAC; annual MRI/EUS from 50",
            "ANNUAL PANCREATIC MRI/EUS FROM AGE 40 MANDATORY (CDKN2A) — 17-39% lifetime PDAC + melanoma 25-36x",
            "RADIOSENSITIVITY ABSOLUTE (ATM biallelic) — standard RT catastrophic; ceralasertib synthetic lethal",
            "STK11 ANNUAL MRI/EUS FROM AGE 30 — EARLIEST hereditary onset; 11-36% lifetime PDAC in PJS",
            "PEMBROLIZUMAB FDA 2017 ANY MSI-H (MLH1) — first tumour-agnostic approval; test MSI on every PDAC",
            "AVOID RADIATION ABSOLUTELY (TP53) — secondary malignancy; R337H Brazilian founder → pancreatic risk",
            "NO SMOKING ABSOLUTELY (PRSS1) — smoking multiplies PDAC risk 40-fold in hereditary pancreatitis",
            "PALB2 BRIDGES BRCA1-BRCA2 — pancreatic MRI/EUS from 50 AND breast MRI from 30 both mandatory",
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
            "pc_risk": gene_info["pc_risk"],
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
        "atlas": "Hereditary-Pancreatic-Cancer-Predisposition-Atlas",
        "definitions": {
            "hboc_brca2": (
                "HBOC BRCA2 (Hereditary Breast–Ovarian Cancer): BRCA2 germline LOF; "
                "Pancreatic cancer 5-7% lifetime (3.5-10x RR); breast 69-85% (female); ovarian 18-27%; "
                "Olaparib FDA 2019 POLO trial maintenance after platinum response; "
                "Annual pancreatic MRI/EUS from age 50; annual breast MRI from 30. "
                "Biallelic FA-D1: avoid alkylating agents; childhood medulloblastoma + Wilms + AML."
            ),
            "fammm_cdkn2a": (
                "FAMMM (Familial Atypical Multiple Mole Melanoma): CDKN2A germline LOF; "
                "Pancreatic cancer 17-39% lifetime — HIGHEST CDKN2A germline single-cancer risk; "
                "Melanoma 25-36x elevated concurrently; "
                "Annual pancreatic MRI/EUS from age 40 MANDATORY; annual total body skin exam; "
                "p16 IHC loss PATHOGNOMONIC in PDAC (somatic 95% sporadic + germline FAMMM context)."
            ),
            "hereditary_atm": (
                "Hereditary ATM (monoallelic AD LOF): Pancreatic cancer 5-8x elevated; "
                "Biallelic Ataxia-Telangiectasia: cerebellar ataxia + oculomotor apraxia + telangiectasiae PATHOGNOMONIC TRIAD; "
                "RADIOSENSITIVITY ABSOLUTE CONTRAINDICATION in biallelic ATM — standard RT doses catastrophic; "
                "Olaparib (HRD-like) + ceralasertib (ATRi) synthetic lethality ATM-null; "
                "Annual pancreatic MRI/EUS from 50; annual breast MRI from 40 (monoallelic female)."
            ),
            "hboc2_palb2": (
                "HBOC type 2 (PALB2 monoallelic): PALB2 anchors BRCA2 to chromatin at DSB sites; "
                "Pancreatic cancer 2-3% lifetime (3-4x); breast cancer 53% lifetime (NEJM 2014); "
                "TBCRC048 olaparib 82% ORR PALB2-mutant breast — highest PARP response rate; "
                "Biallelic FA-N: BRCA2 phenocopy; childhood medulloblastoma + Wilms; avoid alkylating; "
                "Annual pancreatic MRI/EUS from 50; annual breast MRI from 30."
            ),
            "pjs_stk11": (
                "Peutz-Jeghers Syndrome (PJS): STK11 germline LOF; "
                "Pancreatic cancer 11-36% lifetime — HIGHEST single hereditary syndrome for PDAC; "
                "ONSET FROM AGE 30yr — earliest of ALL hereditary pancreatic syndromes; "
                "Mucocutaneous melanin macules lips/buccal/genital PATHOGNOMONIC; "
                "SCTAT (sex cord tumour with annular tubules) ovary PATHOGNOMONIC; "
                "GI endoscopy from age 8yr; annual pancreatic MRI/EUS from 30 MANDATORY."
            ),
            "lynch_mlh1": (
                "Lynch Syndrome type 1 (MLH1): MLH1 germline LOF → MSI-H/dMMR tumours; "
                "Pancreatic cancer 3.7-6% lifetime (3-4x elevated); CRC 40-80% dominant; "
                "MSI-H PATHOGNOMONIC Lynch; BRAF V600E absent DDx sporadic MLH1 methylation; "
                "PEMBROLIZUMAB FDA 2017 — FIRST tumour-agnostic approval: any MSI-H/dMMR solid tumour; "
                "ASPIRIN 600mg daily CAPP2 reduces Lynch cancer risk ~50% at 10yr; "
                "Muir-Torre sebaceous adenomas/carcinomas PATHOGNOMONIC MLH1."
            ),
            "lfs_tp53": (
                "Li-Fraumeni Syndrome (LFS): TP53 germline LOF; "
                "Pancreatic cancer elevated in LFS; R337H Brazilian founder allele → pancreatic + adrenal; "
                "AVOID RADIATION ABSOLUTELY; use MRI-based surveillance (avoid CT radiation); "
                "WBMRI Toronto protocol annually; ACC child PATHOGNOMONIC; "
                "APR-246/eprenetapopt p53 reactivator investigational; MDM2i investigational."
            ),
            "hereditary_pancreatitis_prss1": (
                "Hereditary Pancreatitis (HP): PRSS1 germline GOF (p.R122H most common); "
                "Pancreatic cancer 40-55% LIFETIME CUMULATIVE — HIGHEST of all hereditary pancreatic genes; "
                "Recurrent acute pancreatitis childhood onset PATHOGNOMONIC; calcifications imaging; "
                "NO SMOKING ABSOLUTELY — smoking multiplies PDAC risk 40-fold in HP context; "
                "Annual pancreatic MRI/EUS from age 40; TPIAT for intractable pain; "
                "Annual HbA1c; CFTR modifier testing at diagnosis."
            ),
            "cascade_testing": (
                "CASCADE TESTING — Hereditary Pancreatic Cancer: "
                "1. BRCA2: annual MRI/EUS from 50; olaparib maintenance FDA; platinum-based first-line; "
                "2. CDKN2A: annual MRI/EUS from 40 MANDATORY; total body skin exam; UV avoidance; "
                "3. ATM: annual MRI/EUS from 50 (monoallelic); ceralasertib synthetic lethal; avoid RT biallelic; "
                "4. PALB2: annual MRI/EUS from 50; annual breast MRI from 30; olaparib investigational; "
                "5. STK11: annual MRI/EUS from 30 MANDATORY; GI endoscopy from 8yr; gynaecologic USS; "
                "6. MLH1: pembrolizumab any MSI-H; aspirin 600mg CAPP2; colonoscopy 1-2yr from 25; "
                "7. TP53: WBMRI annually; avoid radiation; pancreatic MRI from 40-45 R337H/LFS; "
                "8. PRSS1: NO SMOKING ABSOLUTELY; annual MRI/EUS from 40; PERT; TPIAT if intractable."
            ),
        },
        "key_clinical_rules": [
            {
                "rule": "OLAPARIB MAINTENANCE FDA APPROVED (BRCA2)",
                "gene": "BRCA2",
                "rationale": "POLO trial 2019: germline BRCA1/2 PDAC — olaparib maintenance after platinum response significantly extends PFS (7.4 vs 3.8 months)",
                "consequence": "Missed BRCA2 testing → no olaparib maintenance; platinum not selected → HRD sensitivity unexploited; preventable progression",
            },
            {
                "rule": "ANNUAL PANCREATIC MRI/EUS FROM AGE 40 MANDATORY (CDKN2A)",
                "gene": "CDKN2A",
                "rationale": "CDKN2A germline FAMMM confers 17-39% lifetime PDAC risk — highest single-cancer germline CDKN2A risk; concurrent melanoma 25-36x also demands surveillance",
                "consequence": "No pancreatic surveillance → PDAC diagnosed at advanced/metastatic stage; melanoma also missed without dermatology protocol",
            },
            {
                "rule": "RADIOSENSITIVITY ABSOLUTE IN BIALLELIC ATM",
                "gene": "ATM",
                "rationale": "Biallelic ATM impairs DSB repair; standard RT doses cause chromosomal catastrophe; RADIOSENSITIVITY is an absolute contraindication in A-T",
                "consequence": "Standard RT in biallelic ATM → severe radiation toxicity, accelerated carcinogenesis, potentially fatal complications",
            },
            {
                "rule": "ANNUAL PANCREATIC MRI/EUS FROM AGE 30 (STK11)",
                "gene": "STK11",
                "rationale": "PJS STK11 confers 11-36% lifetime PDAC risk with earliest onset (from age 30yr) of all hereditary pancreatic syndromes",
                "consequence": "Surveillance starting age 50 (standard) misses the critical 30-50yr window where STK11 PDAC develops in PJS",
            },
            {
                "rule": "PEMBROLIZUMAB FDA 2017 ANY MSI-H/dMMR (MLH1)",
                "gene": "MLH1",
                "rationale": "First tumour-agnostic FDA approval: pembrolizumab works in any MSI-H/dMMR solid tumour regardless of histology; Lynch MLH1 PDAC is MSI-H and pembrolizumab-sensitive",
                "consequence": "Not testing MSI in PDAC → missed pembrolizumab eligibility; major therapeutic opportunity foregone in Lynch pancreatic cancer",
            },
            {
                "rule": "AVOID RADIATION ABSOLUTELY (TP53)",
                "gene": "TP53",
                "rationale": "TP53 germline LOF → impaired G1/S checkpoint → ionising radiation causes secondary sarcoma/carcinoma in RT field",
                "consequence": "Secondary malignancy in RT field within 5-10yr; accelerated carcinogenesis in LFS; prefer surgery-first approach",
            },
            {
                "rule": "NO SMOKING ABSOLUTELY IN HEREDITARY PANCREATITIS (PRSS1)",
                "gene": "PRSS1",
                "rationale": "Smoking multiplies PDAC lifetime risk 40-fold in hereditary pancreatitis; already 40-55% baseline lifetime risk without smoking; the combination is catastrophic",
                "consequence": "PRSS1 + smoking → near-certain PDAC; smoking cessation is the SINGLE highest-impact modifiable intervention in HP",
            },
            {
                "rule": "PALB2 BRIDGES BRCA1-BRCA2 — DUAL SURVEILLANCE MANDATORY",
                "gene": "PALB2",
                "rationale": "PALB2 monoallelic confers both pancreatic (2-3% lifetime) and breast (53% lifetime) cancer risk; both require independent surveillance protocols starting at different ages",
                "consequence": "Surveillance for pancreatic only (or breast only) in PALB2 → the other organ cancer missed; TBCRC048 olaparib opportunity also missed",
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
