#!/usr/bin/env python3
"""Hereditary-Biliary-Tract-Cancer-Predisposition-Atlas — Complete 8-Gene Reference
BRCA1  (BRCA1; 1863aa; 17q21.31; AD LOF;
         HBOC;
         Intrahepatic cholangiocarcinoma 2-4x RR; gallbladder cancer 2-3x;
         Olaparib / cisplatin PARP-sensitive;
         seed SEED_BASE+0) ·
BRCA2  (BRCA2; 3418aa; 13q12.3; AD LOF;
         HBOC;
         iCCA + gallbladder 5-7x RR — HIGHEST BRCA biliary;
         Platinum / PARP olaparib; Fanconi Anemia-D1 biallelic;
         seed SEED_BASE+1) ·
BAP1   (BRCA1-Associated Protein 1; 729aa; 3p21.1; AD LOF;
         BAP1 Tumour Predisposition Syndrome;
         Intrahepatic CCA 40-50% lifetime — HIGHEST BILIARY RISK;
         BAP1-null IHC PATHOGNOMONIC; Mesothelioma 30-60x; uveal melanoma;
         AVOID ASBESTOS; Tazemetostat EZH2i;
         seed SEED_BASE+2) ·
MSH2   (MutS Homolog 2; 934aa; 2p21; AD LOF;
         Lynch Syndrome Type 2;
         Biliary tract 2-4% lifetime; Muir-Torre sebaceous PATHOGNOMONIC;
         EPCAM 3-prime deletion silences MSH2;
         seed SEED_BASE+3) ·
STK11  (Serine-Threonine Kinase 11 / LKB1; 433aa; 19p13.3; AD LOF;
         Peutz-Jeghers Syndrome;
         Gallbladder cancer 5-13%; intrahepatic bile duct 5-10%;
         Mucocutaneous macules lips/oral PATHOGNOMONIC;
         Pancreatic 30% — dominant biliary-adjacent risk;
         seed SEED_BASE+4) ·
ATM    (ATM Serine/Threonine Kinase; 3056aa; 11q22.3; AD/AR LOF;
         Ataxia-Telangiectasia (biallelic) / moderate biliary risk (monoallelic);
         Biliary tract 2-4x RR monoallelic; Radiosensitivity ABSOLUTE biallelic;
         Ceralasertib ATRi; Olaparib in ATM-mutant biliary;
         seed SEED_BASE+5) ·
CDKN2A (Cyclin-Dependent Kinase Inhibitor 2A; 156aa; 9p21.3; AD LOF;
         Familial Atypical Multiple-Mole Melanoma;
         Biliary tract / ampullary 2-3x RR; Pancreatic 20x — DOMINANT RISK;
         Melanoma 25-36x; CDK4/6 inhibitor pathway;
         seed SEED_BASE+6) ·
PALB2  (Partner and Localiser of BRCA2; 1186aa; 16p12.2; AD LOF;
         HBOC-2 / Fanconi Anemia-N biallelic;
         Biliary tract 2-3x RR; Breast 53% lifetime; PARP olaparib TBCRC048;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3182-3189)
"""
import random

SEED_BASE = 3182

ATLAS_GENES = [
    {
        "gene": "BRCA1",
        "protein": (
            "BRCA1 -- 17q21.31 Autosomal-Dominant-LOF -- 1863aa -- "
            "BRCA1-213kDa-HR-Scaffold-RING-E3-Ubiquitin-"
            "HBOC-iCCA-2-4x-Gallbladder-2-3x-"
            "Platinum-Olaparib-PARP-Sensitive-OMIM-113705"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "1863 aa / 17q21.31 BRCA1 encodes BRCA1 (Breast Cancer Type 1 Susceptibility Protein): "
            "STRUCTURE: "
            "  213 kDa; RING domain (N-terminal, E3 ubiquitin ligase with BARD1 heterodimer); "
            "  BRCT tandem repeats (C-terminal, phosphopeptide binding — ATM/CHEK2 signalling); "
            "  Coiled-coil motif (PALB2 interaction); nuclear localisation signals; "
            "  LOF → impaired HR (homologous recombination) → BER/NHEJ fallback → chromosomal instability; "
            "BILIARY TRACT RISK: "
            "  Intrahepatic cholangiocarcinoma (iCCA): 2-4x RR; lifetime absolute risk ~3-6%; "
            "  Gallbladder cancer: 2-3x RR; "
            "  Extrahepatic bile duct: 2-3x RR; "
            "  BRCA1 biliary risk lower than BRCA2 — combined panel captures both; "
            "PARP INHIBITOR SENSITIVITY (PLATINUM / OLAPARIB): "
            "  BRCA1-LOF → HRD (HR deficiency) → reliant on PARP1 for SSB repair → PARP trapping lethal; "
            "  Olaparib FDA-approved for BRCA1/2-mutant cancers (expanded solid tumour indication); "
            "  Cisplatin/gemcitabine (ABC-02 backbone) + olaparib being trialled in biliary; "
            "  Pembrolizumab (IO) synergy explored for HRD biliary tumours; "
            "HBOC SURVEILLANCE: "
            "  Breast: annual MRI + mammogram from age 25-30; prophylactic mastectomy option >35yr; "
            "  Ovarian: RRSO (risk-reducing salpingo-oophorectomy) 35-40yr (BRCA1) — 96% risk reduction; "
            "  Biliary: no proven surveillance interval — ultrasound + CA 19-9 annually if first-degree relative with CCA; "
            "  Pancreatic: annual MRI/MRCP + EUS from age 50 (CAPS5 criteria)"
        ),
        "inheritance": "Autosomal Dominant LOF (de novo ~10%; heterozygous; biallelic = embryonic lethal; Fanconi Anemia-S = biallelic BRCA1)",
        "cancer_risk": "Breast 72% lifetime; Ovarian 44% lifetime; iCCA 2-4x RR; Gallbladder 2-3x RR; Pancreatic 2-3x RR",
        "pathognomonic": "Young-onset triple-negative breast cancer + BRCA1 germline; RRSO before 40yr in BRCA1 obligate; Fanconi-S biallelic",
        "surveillance_key": "Annual breast MRI + mammo from 25; RRSO 35-40yr; Annual pancreatic MRI/EUS from 50; Biliary USS + CA 19-9 if FH CCA",
        "key_distinctions": [
            "TNBC-YOUNG-ONSET-PATHOGNOMONIC",
            "RRSO-35-40yr-BRCA1",
            "PARP-OLAPARIB-HR-DEFICIENCY",
            "PLATINUM-CISPLATIN-SENSITIVE",
            "BILIARY-2-4X-LOWER-THAN-BRCA2",
            "FANCONI-S-BIALLELIC-RARE",
        ],
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-RAD51-Mediator-"
            "HBOC-iCCA-Gallbladder-5-7x-HIGHEST-BRCA-BILIARY-"
            "Platinum-Olaparib-FA-D1-Biallelic-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 13q12.3 BRCA2 encodes BRCA2 (Breast Cancer Type 2 Susceptibility Protein): "
            "STRUCTURE: "
            "  384 kDa; 8 BRC repeats (RAD51 binding — critical for HR); "
            "  OB folds at C-terminus (ssDNA binding); tower domain (BRCA2-DNA complex); "
            "  N-terminal PALB2-binding domain; LOF → RAD51 cannot load onto resected DSBs → HR failure; "
            "BILIARY TRACT RISK — HIGHEST BRCA: "
            "  Intrahepatic CCA: 5-7x RR; lifetime absolute risk ~6-10%; "
            "  Gallbladder cancer: 5-6x RR; "
            "  Biliary tract combined: 5-7x RR — HIGHEST among BRCA1/BRCA2 for biliary; "
            "  Ampullary carcinoma: 3-5x RR (periampullary biliary junction); "
            "PARP INHIBITOR / PLATINUM: "
            "  Olaparib: FDA-approved multiple solid tumour indications incl. pancreatic; "
            "  POLO trial: olaparib maintenance after platinum-iCCA — biliary extrapolation ongoing; "
            "  Cisplatin-based: HR-deficient tumours → platinum hypersensitive; "
            "  Rucaparib, niraparib: alternative PARPi options; "
            "FANCONI ANEMIA D1 (BIALLELIC): "
            "  Biallelic BRCA2 = FA-D1 (most severe FA complementation group); "
            "  Medulloblastoma + Wilms + AML in childhood = FA-D1 presentation; "
            "  ALL FA genotyping panels must include BRCA2 biallelic search; "
            "HBOC SURVEILLANCE: "
            "  Breast: annual MRI + mammo from age 25; prophylactic mastectomy option; "
            "  Ovarian: RRSO 40-45yr (BRCA2 — slightly later than BRCA1 as ovarian onset later); "
            "  Pancreatic: annual MRI/MRCP + EUS from age 50 (or 10yr before youngest FDR with PDAC); "
            "  Male breast: annual breast exam + MRI from 35; PSA for prostate from 40 (10-20x RR)"
        ),
        "inheritance": "Autosomal Dominant LOF (de novo ~10%; heterozygous; biallelic = FA-D1; male carriers: breast 8%, prostate 20x RR)",
        "cancer_risk": "Breast 69% lifetime (women); Male breast 8%; Ovarian 17%; Prostate 20x; iCCA/Gallbladder 5-7x; Pancreatic 5-10x",
        "pathognomonic": "BRCA2 biliary 5-7x = HIGHEST RR among BRCA genes for biliary tract; FA-D1 biallelic = medulloblastoma-Wilms-AML childhood PATHOGNOMONIC",
        "surveillance_key": "Annual breast MRI + mammo from 25; RRSO 40-45yr; Annual pancreatic MRI/EUS from 50; Annual PSA from 40 (men); Biliary USS + CA 19-9",
        "key_distinctions": [
            "BILIARY-5-7X-HIGHEST-BRCA-BILIARY-RISK",
            "PARP-OLAPARIB-FDA-SOLID-TUMOUR",
            "FA-D1-BIALLELIC-CHILDHOOD-MEDULLOBLASTOMA",
            "PROSTATE-20X-RR-MALE-CARRIERS",
            "RRSO-40-45yr-LATER-THAN-BRCA1",
            "PLATINUM-CISPLATIN-SENSITIVE-ALL-HRD",
        ],
    },
    {
        "gene": "BAP1",
        "protein": (
            "BAP1 -- 3p21.1 Autosomal-Dominant-LOF -- 729aa -- "
            "BAP1-80kDa-H2A-Deubiquitinase-ASXL-PR-DUB-"
            "TPDS-iCCA-40-50pct-LIFETIME-HIGHEST-BAP1-null-IHC-"
            "Mesothelioma-30-60x-Uveal-Melanoma-PATHOGNOMONIC-"
            "AVOID-ASBESTOS-Tazemetostat-EZH2i-OMIM-603089"
        ),
        "locus": "3p21.1",
        "protein_size": (
            "3p21.1 BAP1 encodes BAP1 (BRCA1-Associated Protein 1): "
            "STRUCTURE: "
            "  729 aa / 80 kDa nuclear deubiquitylase (DUB); "
            "  UCH domain (ubiquitin C-terminal hydrolase) — deubiquitylates H2AK119ub1 (polycomb mark); "
            "  HCF-1 binding domain; ASXL1/2 interaction (PR-DUB complex formation); "
            "  NLS (nuclear localisation signal); LOF → H2A hyperubiquitylation → PRC1 dysregulation → aberrant epigenome; "
            "INTRAHEPATIC CCA — HIGHEST LIFETIME RISK: "
            "  iCCA: 40-50% lifetime risk in TPDS carriers — highest among all biliary cancer genes; "
            "  Median onset iCCA: 55-65yr (earlier than sporadic iCCA median 70yr); "
            "  BAP1-null IHC (loss of nuclear BAP1 staining) = surrogate biomarker in tumour tissue; "
            "  Biliary tract: PATHOGNOMONIC BAP1-null IHC in iCCA + germline BAP1 LOF; "
            "BAP1 TUMOUR PREDISPOSITION SYNDROME (TPDS): "
            "  Four PATHOGNOMONIC tumours: uveal melanoma, iCCA, mesothelioma, clear cell RCC; "
            "  BAP1-positive atypical intradermal melanocytic proliferations (BAPomas / MBAITs) — skin; "
            "  MBAITs: compound melanocytic lesions on trunk/extremities = PATHOGNOMONIC TPDS; "
            "  Annual whole-body skin exam; annual ophthalmology (uveal melanoma); "
            "MESOTHELIOMA: "
            "  30-60x RR (highest single-gene mesothelioma risk known); "
            "  AVOID ASBESTOS ABSOLUTELY — synergy between BAP1 LOF + asbestos exposure → mesothelioma; "
            "  Annual CT thorax from age 50 if occupational asbestos exposure history; "
            "  Tazemetostat (EZH2 inhibitor): approved mesothelioma — BAP1-LOF sensitises to EZH2i; "
            "TREATMENT: "
            "  iCCA: CisGem (cisplatin-gemcitabine) + durvalumab (TOPAZ-1); biliary DNA-repair pathway; "
            "  Tazemetostat for mesothelioma (FDA 2020); nivolumab-ipilimumab emerging iCCA; "
            "  PARP inhibitors: exploratory in BAP1-deficient tumours (HRD-adjacent mechanism)"
        ),
        "inheritance": "Autosomal Dominant LOF (de novo ~10%; penetrance high ~50% lifetime major cancer; somatic second hit Knudson model)",
        "cancer_risk": "iCCA 40-50% lifetime (HIGHEST); Mesothelioma 30-60x RR; Uveal melanoma 35x; ccRCC 2-5x; Cutaneous melanoma 5x",
        "pathognomonic": "BAP1-null IHC in iCCA + germline BAP1 = PATHOGNOMONIC TPDS; MBAITs/BAPomas on skin = TPDS PATHOGNOMONIC; AVOID ASBESTOS ABSOLUTELY",
        "surveillance_key": "Annual ophthalmology from 30 (uveal melanoma); Annual CT thorax from 50; Annual liver MRI/USS; Whole-body skin exam annually; AVOID ASBESTOS",
        "key_distinctions": [
            "ICCA-40-50PCT-HIGHEST-BILIARY-GENE",
            "BAP1-NULL-IHC-PATHOGNOMONIC",
            "MBAIT-SKIN-PATHOGNOMONIC-TPDS",
            "MESOTHELIOMA-30-60X-AVOID-ASBESTOS",
            "TAZEMETOSTAT-EZH2i-MESOTHELIOMA",
            "UVEAL-MELANOMA-35X-ANNUAL-OPHTH",
        ],
    },
    {
        "gene": "MSH2",
        "protein": (
            "MSH2 -- 2p21 Autosomal-Dominant-LOF -- 934aa -- "
            "MSH2-105kDa-MutSalpha-MutSbeta-MMR-ATPase-"
            "Lynch-Type2-Biliary-2-4pct-Muir-Torre-Sebaceous-PATHOGNOMONIC-"
            "EPCAM-3prime-Deletion-Silences-MSH2-OMIM-609309"
        ),
        "locus": "2p21",
        "protein_size": (
            "934 aa / 2p21 MSH2 encodes MutS Homolog 2 (MSH2): "
            "STRUCTURE: "
            "  105 kDa; ATPase domain; mismatch-binding clamp; forms MutSα heterodimer with MSH6 (mismatches); "
            "  Forms MutSβ heterodimer with MSH3 (small insertion/deletion loops); "
            "  LOF → impaired mismatch recognition → microsatellite instability (MSI-H) → hypermutation; "
            "BILIARY TRACT IN LYNCH SYNDROME: "
            "  Biliary tract: 2-4% lifetime risk in Lynch syndrome (MSH2 highest biliary risk among MMR genes); "
            "  Ampullary cancer: 3-5x RR; "
            "  Gallbladder: 2-3x RR; "
            "  Hepatobiliary presentations may precede colorectal diagnosis in MSH2 carriers; "
            "MUIR-TORRE SYNDROME — PATHOGNOMONIC: "
            "  Muir-Torre = Lynch + sebaceous neoplasms (sebaceous adenoma, sebaceoma, sebaceous carcinoma) ± keratoacanthoma; "
            "  Sebaceous tumour outside eyelid in ANY patient → MSH2/MSH6 germline testing MANDATORY; "
            "  MSH2 accounts for >60% of Muir-Torre; MSH6 second; "
            "  Sebaceous carcinoma (eyelid or extraocular) = Lynch sentinel lesion; "
            "EPCAM 3-PRIME DELETION: "
            "  EPCAM gene (2p21, immediately 5-prime of MSH2) 3-prime deletions → read-through transcription → MSH2 promoter methylation; "
            "  MSH2 silenced by epigenetic mechanism — SEQUENCING MISSES THIS; "
            "  MLPA or deletion analysis required when MSH2 IHC null but no coding variant found; "
            "  Reported in ~3% of Lynch families; predominantly colorectal and urothelial presentation; "
            "LYNCH MANAGEMENT: "
            "  Colonoscopy 1-2yr from age 25; aspirin 600mg/day CAPP2 protocol (50% CRC risk reduction); "
            "  Annual endometrial biopsy/USS from 35yr (women); annual urine cytology (urothelial 25% in MSH2); "
            "  Upper GI: gastroscopy 2-3yr from age 30-35 in Lynch"
        ),
        "inheritance": "Autosomal Dominant LOF (de novo ~5%; heterozygous; biallelic = CMMRD — constitutional mismatch repair deficiency)",
        "cancer_risk": "CRC 52% lifetime; Endometrial 60% (women); Urothelial 25%; Biliary 2-4% lifetime; Ovarian 15%; Gastric 10%",
        "pathognomonic": "Sebaceous neoplasm any site = Muir-Torre = MSH2 PATHOGNOMONIC; EPCAM deletion silent on sequencing — MLPA mandatory",
        "surveillance_key": "Annual colonoscopy from 25; CAPP2 aspirin 600mg; Annual endometrial from 35; Annual urine cytology; Sebaceous tumour → urgent MSH2 testing",
        "key_distinctions": [
            "BILIARY-2-4PCT-LYNCH-MSH2-HIGHEST-MMR",
            "MUIR-TORRE-SEBACEOUS-PATHOGNOMONIC",
            "EPCAM-3PRIME-DELETION-MISSES-SEQUENCING",
            "UROTHELIAL-25PCT-MSH2-HIGHEST-LYNCH",
            "CAPP2-ASPIRIN-50PCT-CRC-REDUCTION",
            "CMMRD-BIALLELIC-PD1-IO-HYPERMUTATOR",
        ],
    },
    {
        "gene": "STK11",
        "protein": (
            "STK11 -- 19p13.3 Autosomal-Dominant-LOF -- 433aa -- "
            "STK11-LKB1-48kDa-AMPK-Master-Kinase-"
            "Peutz-Jeghers-Gallbladder-5-13pct-Biliary-5-10pct-"
            "Pancreatic-30pct-Mucocutaneous-Macules-PATHOGNOMONIC-"
            "KRAS-CoMut-IO-Resistance-OMIM-602216"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "433 aa / 19p13.3 STK11 encodes Serine/Threonine Kinase 11 (LKB1): "
            "STRUCTURE: "
            "  48 kDa; activating kinase for AMPK (AMP-activated protein kinase) family; "
            "  LOF → loss of mTOR suppression via AMPK → cellular energy misregulation; "
            "  N-terminal regulatory domain; kinase catalytic domain; C-terminal STRAD-binding; "
            "PEUTZ-JEGHERS SYNDROME (PJS): "
            "  Mucocutaneous pigmented macules: lips, oral mucosa, perioral, fingers, toes — PATHOGNOMONIC; "
            "  Hamartomatous GI polyps (PJ-type) — can intussuscept (emergency); "
            "  Cumulative cancer risk by age 70: ~93% (any cancer); "
            "BILIARY TRACT RISK: "
            "  Gallbladder cancer: 5-13% lifetime risk (HIGHEST GI-adjacent cancer after pancreatic in PJS); "
            "  Intrahepatic bile duct / intrahepatic CCA: 5-10%; "
            "  Annual biliary USS from age 30; MRCP if USS inconclusive; "
            "  Cholecystectomy discussion in PJS with gallbladder polyps (>1cm) or thickening; "
            "PANCREATIC CANCER — DOMINANT RISK: "
            "  Pancreatic ductal adenocarcinoma (PDAC): 30% lifetime — DOMINANT STK11 malignancy; "
            "  Annual pancreatic MRI/MRCP + EUS from age 30-35 (CAPS5 STK11 criteria); "
            "  KRAS co-mutation in STK11-deficient NSCLC → immunotherapy RESISTANCE (relevant in STK11-mutant lung); "
            "OTHER PJS SURVEILLANCE: "
            "  Colonoscopy + gastroscopy 2yr from age 8; polypectomy to prevent intussusception; "
            "  Annual breast MRI + mammo from age 25 (women: 32-54% breast cancer lifetime); "
            "  Annual cervical smear; testicular exam (SCTAT males); Annual breast cancer surveillance"
        ),
        "inheritance": "Autosomal Dominant LOF (de novo ~45%; heterozygous; NO biallelic human syndrome known)",
        "cancer_risk": "Any cancer 93%; Pancreatic 30%; Breast 32-54% (women); GI small bowel 13%; Colorectal 39%; Gallbladder 5-13%; Biliary 5-10%",
        "pathognomonic": "Mucocutaneous macules (lips/perioral/fingers) = PJS PATHOGNOMONIC; SCTAT testis males PATHOGNOMONIC PJS",
        "surveillance_key": "Annual pancreatic MRI/EUS from 30; Annual biliary USS from 30; Annual breast MRI from 25; GI endoscopy 2yr from 8; Cholecystectomy if GB polyp >1cm",
        "key_distinctions": [
            "GALLBLADDER-5-13PCT-HIGHEST-BILIARY",
            "PANCREATIC-30PCT-DOMINANT-RISK",
            "MUCOCUTANEOUS-MACULES-PATHOGNOMONIC",
            "KRAS-COMUT-IO-RESISTANCE-NSCLC",
            "INTUSSUSCEPTION-EMERGENCY-PJ-POLYP",
            "SCTAT-TESTIS-MALES-PATHOGNOMONIC",
        ],
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Autosomal-Dominant-LOF / AR-LOF -- 3056aa -- "
            "ATM-350kDa-PI3K-Like-Kinase-DSB-Sensor-"
            "Ataxia-Telangiectasia-Biallelic-Radiosensitivity-ABSOLUTE-"
            "Monoallelic-Biliary-2-4x-Ceralasertib-ATRi-Olaparib-OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 11q22.3 ATM encodes Ataxia Telangiectasia Mutated kinase: "
            "STRUCTURE: "
            "  350 kDa; PI3K-like kinase (PIKK family); FAT + kinase + FATC domains; "
            "  Activated by DSBs via MRN complex → phosphorylates H2AX, CHK2, p53, BRCA1; "
            "  Master regulator of DSB signalling; LOF → impaired G1, S, G2/M checkpoints; "
            "MONOALLELIC — BILIARY TRACT RISK: "
            "  Biliary tract: 2-4x RR monoallelic ATM; CCA absolute risk ~3-5% lifetime; "
            "  Gallbladder: 2-3x RR; "
            "  Pancreatic: 3-5x RR monoallelic (second dominant biliary-adjacent risk); "
            "  ATM c.7271T>G (p.Val2424Gly) — recurrent pathogenic variant; "
            "BIALLELIC — ATAXIA-TELANGIECTASIA (A-T): "
            "  Cerebellar ataxia + telangiectasias (conjunctival, auricular) PATHOGNOMONIC; "
            "  Radiosensitivity ABSOLUTE — RT causes catastrophic secondary tumours; "
            "  Immunodeficiency (IgA, IgG2 deficiency) → recurrent sinopulmonary infections; "
            "  Elevated AFP in children (A-T hallmark); elevated CEA; "
            "  Risk: lymphoma/leukaemia 20-30% biallelic; ALL; T-cell lymphoma; "
            "TREATMENT — ATM-DEFICIENT BILIARY: "
            "  Olaparib: FDA-approved for ATM-mutant prostate (PROFOUND); biliary extrapolation; "
            "  Ceralasertib (ATRi AZD6738) + olaparib: synergistic in ATM-LOF tumours; "
            "  Cisplatin hypersensitivity (DDR-deficient tumours); "
            "  Pembrolizumab (immunotherapy): response in ATM-mutant biliary; "
            "SURVEILLANCE MONOALLELIC: "
            "  Annual pancreatic MRI/EUS from 45; biliary USS + CA 19-9 annually from 45; "
            "  Breast MRI + mammo from 40 (ATM monoallelic 2-3x breast risk); "
            "  NO RT for any indication if monoallelic ATM (high secondary cancer risk)"
        ),
        "inheritance": "AR (biallelic = A-T); AD monoallelic (50% allele carriers; biliary/breast/pancreatic moderate risk; NOT an A-T phenotype)",
        "cancer_risk": "A-T biallelic: lymphoma 20-30%, leukaemia; Monoallelic: biliary 2-4x, pancreatic 3-5x, breast 2-3x, prostate 2x",
        "pathognomonic": "Cerebellar ataxia + conjunctival telangiectasias + elevated AFP = A-T PATHOGNOMONIC; RADIOSENSITIVITY ABSOLUTE biallelic",
        "surveillance_key": "Annual pancreatic MRI/EUS from 45; Annual biliary USS from 45; Breast MRI from 40 (monoallelic); AVOID RT (monoallelic); Olaparib/ceralasertib in tumours",
        "key_distinctions": [
            "BILIARY-2-4X-MONOALLELIC",
            "A-T-RADIOSENSITIVITY-ABSOLUTE-BIALLELIC",
            "CERALASERTIB-ATRi-ATM-DEFICIENT",
            "OLAPARIB-FDA-ATM-PROSTATE-EXTRAPOLATION",
            "ATAXIA-TELANGIECTASIA-AFP-ELEVATED",
            "PANCREATIC-3-5X-MONOALLELIC",
        ],
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 Autosomal-Dominant-LOF -- 156aa -- "
            "p16-INK4A-CDK4-6-Inhibitor-16kDa-"
            "FAMM-Biliary-Ampullary-2-3x-Pancreatic-20x-DOMINANT-"
            "Melanoma-25-36x-CDK4-6i-Palbociclib-Abemaciclib-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / 9p21.3 CDKN2A encodes p16-INK4A (Cyclin-Dependent Kinase Inhibitor 2A): "
            "STRUCTURE: "
            "  16 kDa tumour suppressor; ankyrin repeat domain; inhibits CDK4/CDK6-cyclin D1; "
            "  Same locus encodes p14-ARF (alternative reading frame) → MDM2 inhibition → p53 stabilisation; "
            "  LOF → unrestrained CDK4/6 → hyperphosphorylated Rb → G1/S bypass; "
            "BILIARY TRACT RISK: "
            "  Biliary tract / ampullary carcinoma: 2-3x RR; "
            "  Pancreatic (PDAC): 20x RR — DOMINANT CDKN2A malignancy; "
            "  Somatic CDKN2A deletion: most common alteration in pancreatic cancer; "
            "  Germline CDKN2A: responsible for ~3-5% of familial pancreatic cancer; "
            "MELANOMA — FAMM: "
            "  Familial Atypical Multiple-Mole Melanoma (FAMM / FAMMM): "
            "  Multiple dysplastic naevi (>50) + family history of melanoma; "
            "  Cutaneous melanoma: 25-36% lifetime risk; "
            "  Annual dermatology + dermoscopy from age 18; "
            "  MLPA mandatory (9p21 deletion can span CDKN2A not detected by sequencing); "
            "CDK4/6 INHIBITOR SENSITIVITY: "
            "  CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib) target the CDK4/6-cyclin D pathway; "
            "  CDKN2A-LOF tumours: CDK4/6 directly activated — sensitivity exploration ongoing; "
            "  Biliary CCA CDK4/6i trials recruiting; "
            "SURVEILLANCE: "
            "  Annual pancreatic MRI/MRCP + EUS from age 40 (or 10yr before youngest FDR with PDAC); "
            "  Annual biliary USS + CA 19-9 + CEA from 45; "
            "  Annual dermatoscopy + whole-body photography from 18; "
            "  Smoking cessation counselling MANDATORY (synergistic pancreatic risk)"
        ),
        "inheritance": "Autosomal Dominant LOF (penetrance 30-50%; germline deletion misses on sequencing → MLPA mandatory); p14-ARF same locus distinct isoform",
        "cancer_risk": "Melanoma 25-36%; Pancreatic 20x RR; Biliary 2-3x; Oral SCC 10-30x (FAMM); Lung 3-5x",
        "pathognomonic": "Multiple dysplastic naevi + pancreatic cancer in family = FAMM/CDKN2A PATHOGNOMONIC; MLPA mandatory (deletion misses sequencing)",
        "surveillance_key": "Annual pancreatic MRI/EUS from 40; Annual biliary USS from 45; Annual whole-body dermoscopy from 18; MLPA if IHC/sequencing negative",
        "key_distinctions": [
            "PANCREATIC-20X-DOMINANT-RISK",
            "BILIARY-AMPULLARY-2-3X",
            "MELANOMA-25-36PCT-FAMM",
            "CDK4-6I-SENSITIVITY-PATHWAY",
            "MLPA-MANDATORY-DELETION-MISSES-SEQUENCING",
            "P14-ARF-SAME-LOCUS-MDM2",
        ],
    },
    {
        "gene": "PALB2",
        "protein": (
            "PALB2 -- 16p12.2 Autosomal-Dominant-LOF -- 1186aa -- "
            "PALB2-131kDa-WD40-BRCA1-BRCA2-Bridge-"
            "HBOC-2-Biliary-2-3x-Breast-53pct-"
            "Olaparib-TBCRC048-82pct-ORR-FA-N-Biallelic-OMIM-610355"
        ),
        "locus": "16p12.2",
        "protein_size": (
            "1186 aa / 16p12.2 PALB2 encodes PALB2 (Partner and Localiser of BRCA2): "
            "STRUCTURE: "
            "  131 kDa coiled-coil protein; "
            "  N-terminal coiled-coil domain (BRCA1 interaction — nuclear foci recruitment); "
            "  WD40 repeat domain (C-terminal; BRCA2 interaction + chromatin retention); "
            "  PALB2 bridges BRCA1 and BRCA2 in HR complex — without PALB2, BRCA2 cannot access DSB; "
            "  LOF → HR failure equivalent to BRCA1/BRCA2 LOF (but intermediate penetrance); "
            "BILIARY TRACT RISK: "
            "  Biliary tract: 2-3x RR monoallelic PALB2; lifetime absolute biliary risk ~3-5%; "
            "  Gallbladder: 2-3x RR; ampullary: 2x RR; "
            "  PALB2 biliary risk: equivalent to ATM monoallelic; lower than BAP1; "
            "BREAST CANCER — 53% LIFETIME: "
            "  Breast cancer 53% lifetime (PALB2 consortium 2014 NEJM — Lord et al.); "
            "  Risk approaches BRCA2 level — PALB2 now classified HBOC high-risk gene; "
            "  Annual breast MRI + mammo from age 30; prophylactic mastectomy option from 35; "
            "OLAPARIB TBCRC048 — 82% ORR: "
            "  TBCRC048 trial: olaparib in PALB2-mutant advanced breast → 82% ORR; "
            "  Highest PARP inhibitor ORR reported in any BRCA-like gene; "
            "  Biliary PALB2-mutant: platinum sensitivity + olaparib exploration; "
            "FANCONI ANEMIA N (BIALLELIC): "
            "  Biallelic PALB2 = FA-N; childhood presentation: AML, Wilms tumour, medulloblastoma; "
            "  BRCA2 (FA-D1) and PALB2 (FA-N) share similar phenotypic spectrum; "
            "PANCREATIC: "
            "  Pancreatic cancer: 3-5% lifetime; PALB2 accounts for ~4% of familial PDAC; "
            "  Annual pancreatic MRI/EUS from 50 (CAPS5)"
        ),
        "inheritance": "Autosomal Dominant LOF (monoallelic = HBOC-2; biallelic = FA-N; population frequency ~1 in 400; Polish W1140E founder 1 in 250)",
        "cancer_risk": "Breast 53%; Pancreatic 3-5%; Biliary 2-3x; Ovarian 5%; Male breast 1%; FA-N biallelic: AML-Wilms-medulloblastoma childhood",
        "pathognomonic": "PALB2 82% ORR olaparib TBCRC048 = HIGHEST PARPi response of BRCA-like genes; FA-N biallelic = childhood AML/Wilms PATHOGNOMONIC",
        "surveillance_key": "Annual breast MRI + mammo from 30; Annual pancreatic MRI/EUS from 50; Annual biliary USS from 50; Olaparib in PALB2-mutant tumours; FA-N biallelic → HSCT",
        "key_distinctions": [
            "BREAST-53PCT-HBOC-2-HIGH-RISK",
            "OLAPARIB-TBCRC048-82PCT-ORR-HIGHEST",
            "BILIARY-2-3X-EQUIVALENT-ATM",
            "FA-N-BIALLELIC-AML-WILMS-MEDULLOBLASTOMA",
            "POLISH-W1140E-FOUNDER-1-IN-250",
            "PANCREATIC-3-5PCT-CAPS5",
        ],
    },
]


def _make_patients(gene_info, n=40):
    rng = random.Random(gene_info["seed"])
    gene = gene_info["gene"]

    base_age_map = {
        "BRCA1":  (40, 65),
        "BRCA2":  (45, 68),
        "BAP1":   (50, 70),
        "MSH2":   (38, 62),
        "STK11":  (35, 62),
        "ATM":    (48, 70),
        "CDKN2A": (42, 68),
        "PALB2":  (45, 68),
    }
    lo, hi = base_age_map.get(gene, (40, 65))

    biliary_pct_map = {
        "BRCA1":  (8,  18),   # 2-4x RR, absolute ~3-6%
        "BRCA2":  (15, 30),   # 5-7x RR, absolute ~6-10%
        "BAP1":   (40, 55),   # 40-50% lifetime — HIGHEST
        "MSH2":   (8,  18),   # 2-4% Lynch biliary
        "STK11":  (20, 38),   # Gallbladder 5-13% + bile duct 5-10% combined
        "ATM":    (8,  18),   # 2-4x RR monoallelic
        "CDKN2A": (8,  18),   # 2-3x RR biliary/ampullary
        "PALB2":  (8,  16),   # 2-3x RR
    }
    b_lo, b_hi = biliary_pct_map.get(gene, (8, 18))

    pts = []
    for i in range(n):
        age = rng.randint(lo, hi)
        has_biliary = rng.random() < rng.uniform(b_lo / 100, b_hi / 100)
        is_icca = has_biliary and gene in ("BAP1", "BRCA1", "BRCA2", "ATM")
        is_gallbladder = has_biliary and gene in ("STK11", "MSH2")
        pts.append({
            "id": f"{gene}-{gene_info['seed']}-{i+1:03d}",
            "gene": gene,
            "age_onset": age,
            "has_biliary": has_biliary,
            "is_icca": is_icca,
            "is_gallbladder": is_gallbladder,
            "hrd_tumour": gene in ("BRCA1", "BRCA2", "PALB2") and has_biliary,
            "parp_candidate": gene in ("BRCA1", "BRCA2", "PALB2", "ATM") and has_biliary,
        })
    return pts


def generate_overview():
    all_patients = []
    for idx, g in enumerate(ATLAS_GENES):
        g["seed"] = SEED_BASE + idx
        all_patients.extend(_make_patients(g))

    total = len(all_patients)
    biliary_n = sum(1 for p in all_patients if p["has_biliary"])
    icca_n = sum(1 for p in all_patients if p["is_icca"])
    parp_n = sum(1 for p in all_patients if p["parp_candidate"])
    mean_age = round(sum(p["age_onset"] for p in all_patients) / total, 1)

    gene_summary = []
    for g in ATLAS_GENES:
        pts = [p for p in all_patients if p["gene"] == g["gene"]]
        bil_count = sum(1 for p in pts if p["has_biliary"])
        gene_summary.append({
            "gene":               g["gene"],
            "locus":              g["locus"],
            "n":                  len(pts),
            "biliary_n":          bil_count,
            "biliary_pct":        round(bil_count / len(pts) * 100, 1),
            "mean_age_onset":     round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "parp_pct":           round(sum(1 for p in pts if p["parp_candidate"]) / len(pts) * 100, 1),
            "inheritance":        g["inheritance"].split("(")[0].strip(),
            "key_distinctions":   g["key_distinctions"],
        })

    return {
        "atlas":                    "Hereditary-Biliary-Tract-Cancer-Predisposition-Atlas",
        "seed_range":               f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":           total,
        "genes_n":                  len(ATLAS_GENES),
        "biliary_total_n":          biliary_n,
        "biliary_total_pct":        round(biliary_n / total * 100, 1),
        "icca_n":                   icca_n,
        "icca_pct":                 round(icca_n / total * 100, 1),
        "parp_candidate_n":         parp_n,
        "parp_candidate_pct":       round(parp_n / total * 100, 1),
        "mean_age_onset":           mean_age,
        "gene_summary":             gene_summary,
        "genes_detail": [
            {
                "gene":             g["gene"],
                "protein":          g["protein"],
                "locus":            g["locus"],
                "protein_size":     g["protein_size"],
                "inheritance":      g["inheritance"],
                "cancer_risk":      g["cancer_risk"],
                "pathognomonic":    g["pathognomonic"],
                "surveillance_key": g["surveillance_key"],
                "key_distinctions": g["key_distinctions"],
            }
            for g in ATLAS_GENES
        ],
    }


def generate_breakdown():
    rows = []
    for idx, g in enumerate(ATLAS_GENES):
        g["seed"] = SEED_BASE + idx
        pts = _make_patients(g)
        bil_n = sum(1 for p in pts if p["has_biliary"])
        icca_n = sum(1 for p in pts if p["is_icca"])
        parp_n = sum(1 for p in pts if p["parp_candidate"])
        rows.append({
            "gene":               g["gene"],
            "locus":              g["locus"],
            "n":                  len(pts),
            "biliary_n":          bil_n,
            "biliary_pct":        round(bil_n / len(pts) * 100, 1),
            "icca_n":             icca_n,
            "icca_pct":           round(icca_n / len(pts) * 100, 1),
            "parp_n":             parp_n,
            "parp_pct":           round(parp_n / len(pts) * 100, 1),
            "mean_age_onset":     round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "seed":               g["seed"],
            "pathognomonic":      g["pathognomonic"],
            "surveillance_key":   g["surveillance_key"],
            "inheritance":        g["inheritance"].split("(")[0].strip(),
            "key_distinctions":   g["key_distinctions"],
        })
    return {
        "atlas":     "Hereditary-Biliary-Tract-Cancer-Predisposition-Atlas",
        "breakdown": rows,
    }


def generate_definitions():
    defs = [
        {
            "term": "BRCA1 / HBOC / iCCA 2-4x RR / OLAPARIB PARP SENSITIVITY",
            "definition": (
                "BRCA1 — 1863aa / 213 kDa / 17q21.31 / AD LOF\n"
                "HBOC — intrahepatic CCA 2-4x RR; olaparib/cisplatin PARP-sensitive.\n\n"
                "BILIARY TRACT RISK:\n"
                "  iCCA: 2-4x RR; lifetime absolute ~3-6%; gallbladder 2-3x RR.\n"
                "  Ampullary: 2-3x RR; biliary risk lower than BRCA2.\n"
                "  Annual biliary USS + CA 19-9 if first-degree relative with CCA.\n\n"
                "PARP INHIBITOR:\n"
                "  Olaparib (PARP1/2 inhibitor): FDA-approved in BRCA1/2-deficient solid tumours.\n"
                "  HR deficiency → PARP trapping lethal → synthetic lethality.\n"
                "  Cisplatin hypersensitivity: HR-deficient tumours respond to platinum backbone.\n\n"
                "HBOC SURVEILLANCE:\n"
                "  Annual breast MRI + mammogram from age 25-30.\n"
                "  RRSO 35-40yr (BRCA1 — ovarian onset earlier than BRCA2).\n"
                "  Annual pancreatic MRI/EUS from age 50 (CAPS5).\n"
                "  BRCA1 biliary: USS + CA 19-9 annually if CCA family history."
            ),
        },
        {
            "term": "BRCA2 / HBOC / iCCA-GALLBLADDER 5-7x HIGHEST / FA-D1 BIALLELIC",
            "definition": (
                "BRCA2 — 3418aa / 384 kDa / 13q12.3/ AD LOF\n"
                "HBOC — biliary 5-7x RR HIGHEST; platinum/PARP; FA-D1 biallelic.\n\n"
                "BILIARY HIGHEST RR:\n"
                "  iCCA + gallbladder 5-7x RR = HIGHEST among BRCA1/BRCA2 for biliary tract.\n"
                "  Ampullary: 3-5x RR; lifetime absolute biliary ~6-10%.\n\n"
                "OLAPARIB / POLO TRIAL:\n"
                "  POLO trial: olaparib maintenance after platinum in BRCA1/2-mutant PDAC.\n"
                "  Biliary extrapolation: cisplatin-gem + durvalumab (TOPAZ-1) + olaparib exploratory.\n\n"
                "FANCONI ANEMIA D1 (BIALLELIC):\n"
                "  FA-D1 = biallelic BRCA2; most severe FA group.\n"
                "  Childhood: medulloblastoma + Wilms + AML = PATHOGNOMONIC FA-D1.\n"
                "  ALL FA genotyping panels must include biallelic BRCA2 search.\n\n"
                "HBOC SURVEILLANCE:\n"
                "  RRSO 40-45yr (BRCA2 — slightly later than BRCA1).\n"
                "  Annual PSA from 40 in males (prostate 20x RR).\n"
                "  Annual pancreatic MRI/EUS from 50."
            ),
        },
        {
            "term": "BAP1 / TPDS / iCCA 40-50% HIGHEST / BAP1-NULL IHC PATHOGNOMONIC / AVOID ASBESTOS",
            "definition": (
                "BAP1 — 729aa / 80 kDa / 3p21.1 / AD LOF\n"
                "TPDS — iCCA 40-50% lifetime HIGHEST; BAP1-null IHC PATHOGNOMONIC.\n\n"
                "iCCA — HIGHEST LIFETIME BILIARY RISK:\n"
                "  iCCA: 40-50% lifetime — HIGHEST of ALL biliary predisposition genes.\n"
                "  BAP1-null IHC in tumour = surrogate biomarker for BAP1 germline testing.\n"
                "  Annual liver MRI/USS + CA 19-9 from age 40 in TPDS carriers.\n\n"
                "TPDS PATHOGNOMONIC TUMOURS:\n"
                "  BAP1-positive atypical melanocytic proliferations (BAPomas / MBAITs) on skin.\n"
                "  Uveal melanoma: 35x RR; annual ophthalmology from 30.\n"
                "  Mesothelioma: 30-60x RR — AVOID ASBESTOS ABSOLUTELY.\n"
                "  Clear cell RCC: 2-5x RR.\n\n"
                "TREATMENT:\n"
                "  Tazemetostat (EZH2 inhibitor): FDA 2020 mesothelioma.\n"
                "  CisGem + durvalumab (TOPAZ-1) for iCCA.\n"
                "  AVOID ASBESTOS — synergy with BAP1 LOF → mesothelioma risk multiplicative."
            ),
        },
        {
            "term": "MSH2 / LYNCH T2 / BILIARY 2-4% / MUIR-TORRE SEBACEOUS PATHOGNOMONIC / EPCAM DELETION",
            "definition": (
                "MSH2 — 934aa / 105 kDa / 2p21 / AD LOF\n"
                "Lynch Type 2 — biliary 2-4%; Muir-Torre PATHOGNOMONIC; EPCAM deletion.\n\n"
                "BILIARY TRACT IN LYNCH:\n"
                "  Biliary tract: 2-4% lifetime (highest biliary risk among Lynch MMR genes).\n"
                "  Ampullary: 3-5x RR; gallbladder 2-3x RR.\n\n"
                "MUIR-TORRE — PATHOGNOMONIC:\n"
                "  Sebaceous neoplasm (adenoma, sebaceoma, SCC) + Lynch = Muir-Torre.\n"
                "  Sebaceous tumour ANYWHERE outside eyelid → MSH2/MSH6 testing MANDATORY.\n"
                "  >60% of Muir-Torre caused by MSH2.\n\n"
                "EPCAM 3-PRIME DELETION:\n"
                "  EPCAM deletion → read-through → MSH2 promoter methylation → MSH2 silenced.\n"
                "  Sequencing MISSES this — MLPA required when MSH2 IHC null + no coding variant.\n\n"
                "LYNCH MANAGEMENT:\n"
                "  Colonoscopy 1-2yr from age 25; aspirin CAPP2 protocol 600mg/day.\n"
                "  Annual urine cytology (urothelial 25% MSH2 — highest Lynch MMR gene)."
            ),
        },
        {
            "term": "STK11 / PEUTZ-JEGHERS / GALLBLADDER 5-13% / PANCREATIC 30% / MUCOCUTANEOUS MACULES PATHOGNOMONIC",
            "definition": (
                "STK11 (LKB1) — 433aa / 48 kDa / 19p13.3 / AD LOF\n"
                "Peutz-Jeghers — gallbladder 5-13%; pancreatic 30% dominant; macules PATHOGNOMONIC.\n\n"
                "MUCOCUTANEOUS MACULES — PATHOGNOMONIC:\n"
                "  Perioral, oral mucosal, finger, toe macules = PJS PATHOGNOMONIC.\n"
                "  Present from infancy; fade in adulthood (oral persist).\n\n"
                "GALLBLADDER + BILIARY RISK:\n"
                "  Gallbladder cancer: 5-13% lifetime = HIGHEST gallbladder risk in panel.\n"
                "  Bile duct (intrahepatic + extrahepatic): 5-10%.\n"
                "  Annual biliary USS from age 30; cholecystectomy if GB polyp >1 cm.\n\n"
                "PANCREATIC — DOMINANT RISK:\n"
                "  PDAC: 30% lifetime = dominant malignancy in PJS.\n"
                "  Annual pancreatic MRI/MRCP + EUS from age 30-35 (earlier than other genes).\n\n"
                "SURVEILLANCE:\n"
                "  GI endoscopy (gastroscopy + colonoscopy) 2yr from age 8 (polyp surveillance).\n"
                "  Breast: annual MRI + mammo from 25 (women: 32-54%).\n"
                "  SCTAT testis in males PATHOGNOMONIC."
            ),
        },
        {
            "term": "ATM / BILIARY 2-4x MONOALLELIC / RADIOSENSITIVITY ABSOLUTE BIALLELIC / CERALASERTIB",
            "definition": (
                "ATM — 3056aa / 350 kDa / 11q22.3 / AR-AD LOF\n"
                "A-T biallelic; monoallelic biliary 2-4x; radiosensitivity; ceralasertib.\n\n"
                "MONOALLELIC BILIARY RISK:\n"
                "  Biliary tract: 2-4x RR; lifetime absolute ~3-5%.\n"
                "  Pancreatic: 3-5x RR; annual MRI/EUS from 45.\n"
                "  Olaparib: ATM-mutant solid tumours (PROFOUND prostate; biliary extrapolation).\n\n"
                "BIALLELIC — ATAXIA-TELANGIECTASIA:\n"
                "  Cerebellar ataxia + conjunctival telangiectasias + elevated AFP = PATHOGNOMONIC.\n"
                "  RADIOSENSITIVITY ABSOLUTE — RT causes catastrophic secondary tumours.\n"
                "  Immunodeficiency → recurrent sinopulmonary; IgA + IgG2 deficiency.\n\n"
                "CERALASERTIB (ATRi AZD6738):\n"
                "  ATM-LOF → ATR becomes sole response kinase → ATRi synthetic lethality.\n"
                "  Ceralasertib + olaparib: synergistic in ATM-deficient tumours.\n\n"
                "SURVEILLANCE MONOALLELIC:\n"
                "  Annual pancreatic MRI/EUS from 45; biliary USS + CA 19-9 from 45.\n"
                "  Breast MRI from 40; AVOID RT for any indication."
            ),
        },
        {
            "term": "CDKN2A / FAMM / PANCREATIC 20x DOMINANT / BILIARY 2-3x / CDK4-6 INHIBITOR PATHWAY",
            "definition": (
                "CDKN2A (p16-INK4A) — 156aa / 16 kDa / 9p21.3 / AD LOF\n"
                "FAMM — pancreatic 20x dominant; biliary 2-3x; melanoma 25-36%; CDK4/6i.\n\n"
                "BILIARY / AMPULLARY RISK:\n"
                "  Biliary tract + ampullary carcinoma: 2-3x RR; lifetime absolute ~3-4%.\n"
                "  Somatic CDKN2A deletion: most common alteration in pancreatic cancer.\n\n"
                "PANCREATIC — DOMINANT (20x RR):\n"
                "  Annual pancreatic MRI/MRCP + EUS from age 40 (or 10yr before youngest FDR).\n"
                "  3-5% of familial pancreatic cancer = germline CDKN2A.\n\n"
                "FAMM MELANOMA:\n"
                "  Multiple dysplastic naevi + family melanoma = FAMM.\n"
                "  Melanoma: 25-36% lifetime; annual dermatoscopy + full-body photography.\n"
                "  MLPA mandatory (9p21 deletion misses on sequencing alone).\n\n"
                "CDK4/6 INHIBITOR PATHWAY:\n"
                "  p16-INK4A normally inhibits CDK4/6; LOF → CDK4/6 hyperactivated.\n"
                "  Palbociclib, ribociclib, abemaciclib — CDK4/6i trials in CDKN2A-LOF biliary."
            ),
        },
        {
            "term": "PALB2 / HBOC-2 / BILIARY 2-3x / OLAPARIB 82% ORR TBCRC048 / FA-N BIALLELIC",
            "definition": (
                "PALB2 — 1186aa / 131 kDa / 16p12.2 / AD LOF\n"
                "HBOC-2 — biliary 2-3x; breast 53%; olaparib 82% ORR TBCRC048; FA-N biallelic.\n\n"
                "BILIARY RISK:\n"
                "  Biliary tract: 2-3x RR; lifetime absolute ~3-5%.\n"
                "  Gallbladder: 2-3x RR; ampullary: 2x RR.\n\n"
                "OLAPARIB TBCRC048 — HIGHEST PARPi RESPONSE:\n"
                "  TBCRC048 trial: olaparib in PALB2-mutant advanced breast → 82% ORR.\n"
                "  HIGHEST PARPi ORR of any BRCA-like gene; PALB2 now classified HBOC high-risk.\n"
                "  Biliary PALB2: platinum + olaparib extrapolation ongoing.\n\n"
                "FANCONI ANEMIA N (BIALLELIC):\n"
                "  FA-N = biallelic PALB2; childhood AML + Wilms + medulloblastoma = PATHOGNOMONIC.\n"
                "  PALB2 bridges BRCA1-BRCA2 — without PALB2, BRCA2 cannot access DSB.\n\n"
                "SURVEILLANCE:\n"
                "  Annual breast MRI + mammo from 30 (breast 53% lifetime).\n"
                "  Annual pancreatic MRI/EUS from 50 (CAPS5).\n"
                "  Annual biliary USS from 50; olaparib for PALB2-mutant advanced tumours."
            ),
        },
        {
            "term": "CASCADE TESTING — Hereditary Biliary Tract Cancer",
            "definition": (
                "CASCADE TESTING PRIORITIES for Hereditary Biliary Tract Cancer Predisposition:\n\n"
                "TIER 1 — HIGHEST BILIARY RISK (definitive biliary predisposition):\n"
                "  BAP1 TPDS: ALL first-degree relatives; iCCA 40-50% lifetime — HIGHEST;\n"
                "    → Annual liver MRI/USS; ophthalmology from 30; AVOID ASBESTOS ABSOLUTELY.\n"
                "  STK11 PJS: family cascade for PJS macules + biliary 5-13%;\n"
                "    → Annual pancreatic MRI/EUS from 30; annual biliary USS; GB polyp → cholecystectomy.\n\n"
                "TIER 2 — MODERATE BILIARY RISK (HBOC/Lynch biliary component):\n"
                "  BRCA2: biliary 5-7x RR (HIGHEST BRCA); olaparib/platinum.\n"
                "    → CAPS5 pancreatic surveillance; RRSO 40-45yr; annual biliary USS + CA19-9.\n"
                "  BRCA1: biliary 2-4x; platinum/olaparib; RRSO 35-40yr.\n"
                "  MSH2: Lynch biliary 2-4%; Muir-Torre sebaceous → urgent testing.\n"
                "    → CAPP2 aspirin; annual urine cytology; EPCAM MLPA mandatory.\n\n"
                "TIER 3 — LOWER BILIARY / SHARED DDR RISK:\n"
                "  ATM monoallelic: biliary 2-4x; pancreatic 3-5x; ceralasertib+olaparib.\n"
                "    → Annual pancreatic MRI/EUS from 45; biliary USS + CA19-9 from 45.\n"
                "  CDKN2A: biliary 2-3x; pancreatic 20x DOMINANT; melanoma 25-36%.\n"
                "    → Annual pancreatic MRI/EUS from 40; dermoscopy annually; MLPA mandatory.\n"
                "  PALB2: biliary 2-3x; breast 53%; olaparib 82% ORR TBCRC048.\n"
                "    → Annual breast MRI from 30; annual pancreatic MRI/EUS from 50.\n\n"
                "PATHOGNOMONIC POINTERS:\n"
                "  iCCA with BAP1-null IHC → germline BAP1 testing MANDATORY.\n"
                "  Sebaceous neoplasm (non-eyelid) → MSH2/MSH6 Muir-Torre testing.\n"
                "  PJS macules + gallbladder/biliary → STK11 cascade.\n"
                "  Uveal melanoma + mesothelioma in family → BAP1 TPDS.\n"
                "  Young CCA + HRD tumour signature → BRCA1/BRCA2/PALB2/ATM panel.\n"
            ),
        },
    ]

    return {
        "atlas":       "Hereditary-Biliary-Tract-Cancer-Predisposition-Atlas",
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
        print(f"  {row['gene']:8s} n={row['n']} mean_age={row['mean_age_onset']} "
              f"biliary_n={row['biliary_n']} ({row['biliary_pct']}%)")
    print("\n=== DEFINITIONS (terms only) ===")
    df = generate_definitions()
    for d in df["definitions"]:
        print(f"  {d['term']}")
