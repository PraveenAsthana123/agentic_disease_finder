#!/usr/bin/env python3
"""Hereditary-Biliary-Tract-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
BAP1    (BRCA1-associated protein 1; 729aa; 3p21.1; AD LOF;
         TPDS (Tumour Predisposition through DEUBIQUITINASE) — intrahepatic CCA elevated;
         mesothelioma 30-60% lifetime HIGHEST hereditary; uveal melanoma PATHOGNOMONIC;
         MBAITs (melanocytic BAP1-mutated atypical intradermal tumours) PATHOGNOMONIC;
         seed SEED_BASE+0).
BRCA1   (Breast cancer gene 1; 1863aa; 17q21.31; AD LOF;
         HBOC — CCA 2-3x lifetime elevated (HRD-driven); gallbladder 1.5-2x;
         cisplatin preferred for HRD CCA; olaparib HRD-positive biliary;
         seed SEED_BASE+1).
BRCA2   (Breast cancer gene 2; 3418aa; 13q12.3; AD LOF;
         HBOC — CCA 3-4x lifetime elevated STRONGEST BRCA CCA risk;
         olaparib POLO trial biliary CCA (HRD-positive); cisplatin + gemcitabine HRD;
         pancreatic 5-7% dominant extracolonic;
         seed SEED_BASE+2).
ATM     (Ataxia telangiectasia mutated; 3056aa; 11q22.3; AD LOF;
         A-T heterozygote — CCA 2-4x elevated; radiation sensitivity MANDATORY;
         RT reduce 20-30% even monoallelic heterozygotes; biallelic A-T: ALL/lymphoma 100x;
         seed SEED_BASE+3).
MLH1    (MutL homolog 1; 756aa; 3p22.2; AD LOF;
         Lynch syndrome type 1 — bile duct CCA 2-4x elevated; MSI-H PATHOGNOMONIC;
         pembrolizumab FDA 2017 dMMR/MSI-H tumour-agnostic; aspirin CAPP2 50% risk reduction;
         seed SEED_BASE+4).
MSH2    (MutS homolog 2; 934aa; 2p21; AD LOF;
         Lynch syndrome type 2 — biliary tract elevated; Muir-Torre sebaceous PATHOGNOMONIC;
         EPCAM 3-prime deletion silences MSH2 — MLPA MANDATORY; urothelial 10-14% HIGHEST;
         seed SEED_BASE+5).
CDKN2A  (Cyclin-dependent kinase inhibitor 2A; 156aa; 9p21.3; AD LOF;
         FAMMM — CCA 3-5x elevated; pancreatic 17-39% HIGHEST hereditary;
         p16/INK4A + p14/ARF dual tumour suppressors; annual MRI/EUS from 40yr MANDATORY;
         seed SEED_BASE+6).
STK11   (Serine/threonine kinase 11 / LKB1; 433aa; 19p13.3; AD LOF;
         Peutz-Jeghers — biliary polyps + CCA risk elevated; gastric 29%;
         mucocutaneous macules PATHOGNOMONIC; SCTAT ovarian PATHOGNOMONIC;
         small bowel 13% HIGHEST; pancreatic 11-36% HIGHEST;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3518-3525)
"""
import random

SEED_BASE = 3518

ATLAS_GENES = [
    {
        "gene": "BAP1",
        "protein": (
            "BAP1 -- 3p21.1 Autosomal-Dominant-LOF -- 729aa -- "
            "BRCA1-Associated-Protein-1-DUB-80kDa-"
            "TPDS-Intrahepatic-CCA-Elevated-"
            "Mesothelioma-30-60pct-Lifetime-HIGHEST-Hereditary-"
            "Uveal-Melanoma-PATHOGNOMONIC-MBAITs-PATHOGNOMONIC-"
            "RCC-14-17pct-Belzutifan-HIF2alpha-OMIM-614327"
        ),
        "locus": "3p21.1",
        "protein_size": (
            "729 aa / 80 kDa / 3p21.1 BAP1 biliary cancer molecular context: "
            "STRUCTURE: "
            "  729 aa / 80 kDa; Catalytic UCH domain (aa 1-240): deubiquitinase activity; "
            "  Nuclear localisation signals (aa 520-540, aa 714-729); "
            "  BRCA1 binding (aa 570-595); ASXL1/2 binding (HCF-1, FOXK complexes); "
            "  BAP1 = epigenetic tumour suppressor — removes H2AK119 ubiquitin (PRC1 target); "
            "CANCER RISKS (BILIARY FOCUS): "
            "  INTRAHEPATIC CCA: elevated (2-3x vs population); "
            "  MESOTHELIOMA: 30-60% lifetime HIGHEST hereditary risk — pleural dominant; "
            "  UVEAL MELANOMA: lifetime strongly elevated PATHOGNOMONIC in BAP1-TPDS families; "
            "  RCC (clear cell): 14-17% lifetime; "
            "  CUTANEOUS MELANOMA: elevated; "
            "MBAITs: "
            "  Melanocytic BAP1-mutated atypical intradermal tumours — PATHOGNOMONIC for germline BAP1; "
            "  Eruptive, dome-shaped, pink-brown, paucimelanocytic on dermoscopy; "
            "  May simulate Spitz naevus or melanoma — IHC BAP1 loss confirms; "
            "KEY MANAGEMENT (BILIARY): "
            "  ANNUAL MRI LIVER from 30yr (intrahepatic CCA surveillance); "
            "  ANNUAL MRCP if biliary dilatation; "
            "  AVOID ASBESTOS ABSOLUTELY (mesothelioma 30-60% + asbestos exposure = synergistic); "
            "  ANNUAL CHEST CT from 30yr (mesothelioma + pleural); "
            "  ANNUAL UVEAL MELANOMA EXAM (ophthalmology); "
            "  ANNUAL KIDNEY US/MRI; "
            "  SKIN EXAM: MBAITs → dermatologist, IHC BAP1 on lesion"
        ),
        "syndrome": "BAP1 Tumour Predisposition Syndrome (TPDS)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "biliary_risk": "Intrahepatic CCA 2-3x elevated; mesothelioma 30-60% HIGHEST hereditary risk",
        "pathognomonic": "MBAITs (melanocytic BAP1-mutated atypical intradermal tumours) PATHOGNOMONIC; uveal melanoma PATHOGNOMONIC in TPDS families",
        "key_avoid": "AVOID ASBESTOS ABSOLUTELY (mesothelioma risk 30-60% — asbestos synergistic = near-certain mesothelioma); do NOT miss MBAITs (pink dome skin lesions) — IHC BAP1 loss confirms germline; do NOT ignore uveal melanoma history in family (PATHOGNOMONIC trigger for BAP1 testing)",
        "key_rule": "AVOID ASBESTOS ABSOLUTELY. Rule: MBAITs + uveal melanoma in family = BAP1 TPDS until proven otherwise. Annual chest CT (mesothelioma) + liver MRI (CCA) + ophthal exam (uveal melanoma) from 30yr. Belzutifan (HIF2α inhibitor) FDA2021 for VHL-driven cancers — BAP1-RCC emerging data",
        "surveillance": "Annual chest CT from 30yr (mesothelioma); annual liver MRI + MRCP from 30yr (CCA); annual ophthalmology exam (uveal melanoma); annual kidney MRI; annual skin exam (MBAITs); chest CT-guided biopsy for pleural thickening",
        "targeted_rx": "Pembrolizumab FDA2017 dMMR (if MSI-H biliary); gemcitabine+cisplatin+durvalumab TOPAZ-1 (any advanced BTC); IDH1i ivosidenib if IDH1-somatic (not germline BAP1-specific); tebentafusp FDA2022 uveal melanoma; nivolumab mesothelioma",
    },
    {
        "gene": "BRCA1",
        "protein": (
            "BRCA1 -- 17q21.31 Autosomal-Dominant-LOF -- 1863aa -- "
            "HR-Scaffold-208kDa-RING-E3-Ligase-BRCT-FANCS-"
            "HBOC-CCA-2-3x-Elevated-Gallbladder-1.5-2x-"
            "HRD-Cisplatin-Preferred-Olaparib-HRD-Positive-"
            "Breast-70-72pct-Ovarian-44-46pct-DOMINANT-OMIM-113705"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "1863 aa / 208 kDa / 17q21.31 BRCA1 biliary cancer molecular context: "
            "STRUCTURE: "
            "  1863 aa / 208 kDa; RING domain (aa 1-109): E3 ubiquitin ligase (with BARD1); "
            "  Coiled-coil (aa 1391-1424): PALB2 binding; "
            "  BRCT domains (aa 1646-1863): pSer phosphopeptide recognition (phospho-BARD1); "
            "  FA complementation group S (FANCS) — biallelic = embryonic lethal; "
            "CANCER RISKS (BILIARY FOCUS): "
            "  INTRAHEPATIC CCA: 2-3x elevated (HRD-driven biliary vulnerability); "
            "  GALLBLADDER: 1.5-2x elevated; "
            "  PANCREATIC: 2-3x elevated; "
            "  BREAST: 70-72% lifetime DOMINANT; OVARIAN: 44-46% lifetime; "
            "HRD IN BILIARY: "
            "  BRCA1 LOF → HRD scar (SBS3 signature); "
            "  Cisplatin PREFERRED over oxaliplatin in HRD CCA (platinum backbone gemcitabine+cisplatin); "
            "  Olaparib HRD-positive biliary (clinical trials, off-label after platinum); "
            "KEY MANAGEMENT (BILIARY): "
            "  ANNUAL MRCP + liver MRI from 40yr (CCA surveillance — starts later than BRCA2); "
            "  HRD TESTING on biliary tumour — guides cisplatin + PARPi eligibility; "
            "  CISPLATIN PREFERRED: BRCA1 CCA → gemcitabine+cisplatin (not gemcitabine+oxaliplatin); "
            "  PRIMARY FOCUS: breast MRI from 25yr, RRSO at 35-40yr"
        ),
        "syndrome": "Hereditary Breast and Ovarian Cancer syndrome type 1 (HBOC1 / Fanconi S)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "biliary_risk": "Intrahepatic CCA 2-3x elevated; gallbladder 1.5-2x; pancreatic 2-3x",
        "pathognomonic": "HRD scar (SBS3) on biliary tumour; BRCA1+BARD1 IHC loss; somatic second-hit required for tumour",
        "key_avoid": "Do NOT miss cisplatin preference for HRD biliary (gemcitabine+CISPLATIN, not oxaliplatin); do NOT offer olaparib for BRCA1 CCA without HRD testing on tumour (HRD test confirms PARPi eligibility beyond germline alone)",
        "key_rule": "BRCA1 biliary: HRD scar drives cisplatin sensitivity. Rule: any BRCA1 carrier with biliary cancer → HRD test on tumour + cisplatin-based regimen (gemcitabine+cisplatin TOPAZ-1 backbone). Olaparib HRD-positive biliary (POLO-like emerging data). RRSO 35-40yr (ovarian 44-46% dominant)",
        "surveillance": "Annual liver MRI + MRCP from 40yr; annual breast MRI+mammography from 25yr; RRSO at 35-40yr; HRD testing on biliary/pancreatic tumours; annual gastroscopy from 50yr (gastric elevated)",
        "targeted_rx": "Gemcitabine+cisplatin+durvalumab TOPAZ-1; olaparib HRD-positive (off-label biliary); niraparib HRD-positive; pembrolizumab dMMR/MSI-H (rare biliary); RRSO risk-reducing; bilateral mastectomy risk-reducing",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "HR-Mediator-384kDa-BRC-Repeats-FANCD1-RAD51-Loader-"
            "HBOC-CCA-3-4x-STRONGEST-BRCA-CCA-Risk-Gallbladder-2-3x-"
            "Olaparib-POLO-Biliary-HRD-Cisplatin-Gemcitabine-"
            "Breast-69-72pct-Ovarian-17-22pct-Pancreatic-5-7pct-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 biliary cancer molecular context: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; NLS (aa 3263-3269): nuclear localisation; "
            "  8 BRC repeats (aa 1002-2085): RAD51 filament binding; "
            "  C-terminal DNA-binding domain (aa 2396-3190): ssDNA binding; "
            "  OB folds: protects RAD51 filaments at stalled forks; "
            "  FA complementation group D1 (FANCD1) — biallelic = Fanconi anemia D1 (embryonal tumours); "
            "CANCER RISKS (BILIARY FOCUS): "
            "  INTRAHEPATIC/EXTRAHEPATIC CCA: 3-4x elevated STRONGEST BRCA biliary risk; "
            "  GALLBLADDER: 2-3x elevated; "
            "  PANCREATIC: 5-7% lifetime dominant extracolonic; "
            "  BREAST: 69-72% (male breast cancer elevated 6-8%); OVARIAN: 17-22%; "
            "POLO TRIAL BILIARY: "
            "  POLO trial (olaparib BRCA1/2 pancreatic maintenance) opened biliary pathway; "
            "  BRCA2 biliary CCA — off-label olaparib maintenance after platinum response; "
            "  BGB-290 (pamiparib), niraparib biliary data emerging; "
            "KEY MANAGEMENT (BILIARY): "
            "  ANNUAL MRCP + liver MRI + EUS from 40yr (CCA + pancreatic surveillance BRCA2); "
            "  HRD TESTING: SBS3 + HRDetect on biliary tumour → olaparib eligibility; "
            "  CISPLATIN PREFERRED: gemcitabine+cisplatin+durvalumab (TOPAZ-1 backbone)"
        ),
        "syndrome": "Hereditary Breast and Ovarian Cancer syndrome type 2 (HBOC2 / Fanconi D1)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "biliary_risk": "CCA 3-4x elevated STRONGEST BRCA biliary risk; gallbladder 2-3x; pancreatic 5-7% dominant",
        "pathognomonic": "HRD scar (SBS3) + BRCA2 biallelic inactivation on biliary tumour; male breast cancer in BRCA2 family",
        "key_avoid": "Do NOT miss BRCA2 in male breast cancer (6-8% risk — PATHOGNOMONIC trigger for germline testing); BRCA2 has STRONGER biliary CCA risk than BRCA1 (3-4x vs 2-3x); do NOT skip pancreatic surveillance (EUS+MRI from 40yr — 5-7% dominant)",
        "key_rule": "BRCA2 biliary: strongest BRCA CCA risk (3-4x). Rule: BRCA2 carrier with advanced CCA → HRD test on tumour + gemcitabine+cisplatin backbone + consider olaparib maintenance (POLO paradigm). Annual EUS+MRI pancreas from 40yr (pancreatic 5-7% dominant extracolonic risk)",
        "surveillance": "Annual MRCP+liver MRI+EUS from 40yr; annual breast MRI+mammography from 25yr; RRSO at 40-45yr (ovarian 17-22%); HRD testing on biliary/pancreatic tumours; annual gastroscopy from 50yr; prostate PSA from 40yr (BRCA2 prostate 15%)",
        "targeted_rx": "Gemcitabine+cisplatin+durvalumab TOPAZ-1; olaparib maintenance HRD-positive (POLO paradigm); niraparib; pembrolizumab dMMR/MSI-H; ivosidenib IDH1-somatic (somatic co-mutation); bilateral mastectomy risk-reducing",
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Autosomal-Dominant-LOF -- 3056aa -- "
            "PI3K-Like-Serine-Threonine-Kinase-350kDa-DSB-Sensor-"
            "AT-Heterozygote-CCA-2-4x-Elevated-RT-Sensitivity-MANDATORY-"
            "Reduce-RT-20-30pct-Even-Monoallelic-Biliary-Cholangiocarcinoma-"
            "Biallelic-AT-ALL-Lymphoma-100x-AVOID-Alkylating-RT-OMIM-208900"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM biliary cancer molecular context: "
            "STRUCTURE: "
            "  3056 aa / 350 kDa; HEAT repeats (aa 1-3000): BRCA1/H2AX interaction; "
            "  FAT domain (aa 2570-2950): regulatory; "
            "  PI3K kinase domain (aa 2712-3011): Ser/Thr phosphorylation cascade; "
            "  FATC domain (aa 3011-3056): essential for kinase activity; "
            "CANCER RISKS (BILIARY FOCUS): "
            "  CCA: 2-4x elevated (ATM heterozygote — DDR impairment → biliary HRD); "
            "  PANCREATIC: 5-8x elevated (second after BRCA2 for pancreatic risk heterozygote); "
            "  BREAST: 2-3x moderate (heterozygote); "
            "  BIALLELIC A-T: ALL/lymphoma 100x, radiation ABSOLUTE CI; "
            "RADIATION SENSITIVITY (CRITICAL for BILIARY): "
            "  Monoallelic ATM heterozygote: radiosensitive (RT reduce 20-30%); "
            "  Communicate radiosensitivity to radiation oncologist BEFORE RT for biliary CCA; "
            "  Stereotactic body RT (SBRT) for biliary: DOSE REDUCE and careful planning; "
            "  Do NOT use standard biliary RT fractionation without ATM-awareness; "
            "KEY MANAGEMENT (BILIARY): "
            "  ANNUAL EUS + MRI pancreas from 40yr (pancreatic 5-8x elevated); "
            "  RT radiosensitivity communication mandatory; "
            "  Ceralasertib (ATRi) synthetic lethality with ATM LOF (clinical trials)"
        ),
        "syndrome": "Ataxia-telangiectasia (biallelic) / ATM heterozygote — cancer predisposition syndrome",
        "inheritance": "AD LOF heterozygote (monoallelic cancer risk); AR biallelic = full A-T syndrome",
        "biliary_risk": "CCA 2-4x elevated (ATM heterozygote); pancreatic 5-8x elevated (heterozygote)",
        "pathognomonic": "A-T syndrome (biallelic): cerebellar ataxia + telangiectasias + immunodeficiency; IgA deficiency biallelic; radiosensitivity absolute biallelic",
        "key_avoid": "Do NOT use standard RT fractionation for biliary CCA in ATM carrier without dose reduction (20-30% reduction mandatory even monoallelic); biallelic A-T: AVOID alkylating agents AND RT ABSOLUTELY — combined immunodeficiency + DNA repair failure; ceralasertib ATRi: AVOID in biallelic A-T (already maximally sensitised)",
        "key_rule": "ATM monoallelic = radiosensitive. Rule: COMMUNICATE ATM status to radiation oncologist before any RT for biliary CCA. Reduce RT dose 20-30% even for heterozygotes. Annual EUS+MRI pancreas from 40yr (5-8x elevated — comparable to BRCA2). Ceralasertib (ATRi) synthetic lethality with ATM LOF in biliary tumours",
        "surveillance": "Annual EUS+MRI pancreas from 40yr; annual MRCP+liver MRI from 40yr; annual breast MRI from 40yr (2-3x); prostate PSA from 40yr; communicate RT radiosensitivity at cancer diagnosis",
        "targeted_rx": "Gemcitabine+cisplatin+durvalumab TOPAZ-1; ceralasertib (ATRi, AZD6738) ATM-LOF synthetic lethality (clinical trials); olaparib HRD-positive biliary (ATM LOF → HRD); pembrolizumab dMMR/MSI-H; RT dose-reduced (20-30%) mandatory",
    },
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 756aa -- "
            "MutLalpha-Scaffold-85kDa-PMS2-Heterodimer-"
            "Lynch1-Bile-Duct-CCA-2-4x-MSI-H-PATHOGNOMONIC-"
            "Pembrolizumab-FDA2017-dMMR-Tumour-Agnostic-"
            "Aspirin-CAPP2-600mg-50pct-Risk-Reduction-OMIM-120436"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "756 aa / 85 kDa / 3p22.2 MLH1 biliary cancer molecular context: "
            "STRUCTURE: "
            "  756 aa / 85 kDa; N-terminal ATPase (aa 1-340): MutLα complex; "
            "  C-terminal (aa 499-756): PMS2 dimerisation; "
            "CANCER RISKS (BILIARY FOCUS): "
            "  BILE DUCT CCA: 2-4x elevated (Lynch type 1); "
            "  GALLBLADDER: modestly elevated; "
            "  CRC: 52-82% dominant; ENDOMETRIAL: 40-60%; "
            "MSI-H BILIARY: "
            "  Biliary CCA MSI-H (~5-10% of CCA overall) → pembrolizumab FDA2017 dMMR/MSI-H; "
            "  Lynch-associated biliary CCA typically MSI-H (high pembrolizumab response); "
            "  IHC MMR all biliary cancers recommended — Lynch detection; "
            "SOMATIC MLH1 METHYLATION: "
            "  Somatic promoter methylation mimics Lynch in biliary tumours — confirm germline; "
            "  BRAF V600E somatic = sporadic methylation (not Lynch) in CRC (but BRAF rare in biliary); "
            "KEY MANAGEMENT (BILIARY): "
            "  IHC MMR on all biliary cancer specimens (Lynch detection); "
            "  PEMBROLIZUMAB for MSI-H Lynch biliary CCA; "
            "  ASPIRIN CAPP2 600mg/day Lynch chemoprevention (CRC dominant)"
        ),
        "syndrome": "Lynch syndrome type 1 (HNPCC type 1)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "biliary_risk": "Bile duct CCA 2-4x elevated (Lynch1); gallbladder modestly elevated; CRC 52-82% dominant",
        "pathognomonic": "MSI-H on biliary tumour + IHC MLH1+PMS2 loss; Lynch-biliary CCA responds to pembrolizumab",
        "key_avoid": "Do NOT miss Lynch in biliary CCA — IHC MMR mandatory on all biliary specimens (Lynch-biliary MSI-H = pembrolizumab eligible); do NOT call Lynch without excluding somatic MLH1 promoter methylation (confirm germline testing); aspirin CAPP2 is Lynch-specific (not general biliary chemoprevention)",
        "key_rule": "Universal MMR IHC on ALL biliary cancers (Lynch detection — MSI-H biliary = pembrolizumab eligible). Rule: aspirin CAPP2 600mg/day is Lynch chemoprevention (50% CRC risk reduction RCT). Lynch biliary CCA historically missed — IHC mandatory",
        "surveillance": "Annual colonoscopy from 25yr; IHC MMR on biliary specimen at diagnosis; annual endometrial sampling from 30yr; annual gastroscopy; aspirin CAPP2 600mg/day",
        "targeted_rx": "Pembrolizumab FDA2017 dMMR/MSI-H tumour-agnostic; dostarlimab; gemcitabine+cisplatin+durvalumab TOPAZ-1 backbone; FOLFOX/CAPOX CRC; aspirin CAPP2 600mg chemoprevention",
    },
    {
        "gene": "MSH2",
        "protein": (
            "MSH2 -- 2p21 Autosomal-Dominant-LOF -- 934aa -- "
            "MutSalpha-MutSbeta-Scaffold-105kDa-MSH6-MSH3-"
            "Lynch2-Biliary-Tract-Elevated-Muir-Torre-Sebaceous-PATHOGNOMONIC-"
            "EPCAM-3prime-Deletion-MLPA-MANDATORY-"
            "Urothelial-10-14pct-HIGHEST-Lynch-OMIM-609309"
        ),
        "locus": "2p21",
        "protein_size": (
            "934 aa / 105 kDa / 2p21 MSH2 biliary cancer molecular context: "
            "STRUCTURE: "
            "  934 aa / 105 kDa; MutSα (MSH2-MSH6): base substitutions; "
            "  MutSβ (MSH2-MSH3): larger indels; "
            "  MSH2 loss disables BOTH complexes; "
            "CANCER RISKS (BILIARY FOCUS): "
            "  BILIARY TRACT (CCA + gallbladder): elevated (Lynch type 2); "
            "  UROTHELIAL: 10-14% HIGHEST single Lynch gene; "
            "  CRC: 45-75%; ENDOMETRIAL: 25-50%; "
            "  MUIR-TORRE: sebaceous adenoma/carcinoma PATHOGNOMONIC Lynch; "
            "EPCAM DELETION: "
            "  3-prime EPCAM deletion → epigenetic MSH2 silencing; "
            "  Urothelial-dominant Lynch when EPCAM-MSH2; "
            "  MLPA MANDATORY — not detected by sequencing alone; "
            "KEY MANAGEMENT (BILIARY): "
            "  IHC MMR on biliary cancer specimen; "
            "  PEMBROLIZUMAB if MSI-H Lynch biliary; "
            "  ANNUAL CYSTOSCOPY + urine cytology (urothelial 10-14% dominant)"
        ),
        "syndrome": "Lynch syndrome type 2 / Muir-Torre syndrome",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "biliary_risk": "Biliary tract elevated (Lynch2); urothelial 10-14% HIGHEST; CRC 45-75% dominant",
        "pathognomonic": "Sebaceous carcinoma/adenoma PATHOGNOMONIC Muir-Torre; IHC MSH2+MSH6 both lost; EPCAM deletion silences MSH2",
        "key_avoid": "Do NOT miss EPCAM deletion (MLPA mandatory — not detected by sequencing); Muir-Torre sebaceous skin tumour = PATHOGNOMONIC Lynch MSH2 → trigger germline testing; annual cystoscopy (urothelial 10-14% highest Lynch gene — dominant in EPCAM-MSH2)",
        "key_rule": "EPCAM 3-prime deletion → MSH2 silencing → urothelial-dominant Lynch. Rule: MLPA mandatory all MSH2-negative cases by sequencing. Muir-Torre sebaceous = PATHOGNOMONIC Lynch type 2 → mandatory germline referral. Annual cystoscopy + urine cytology for MSH2 Lynch",
        "surveillance": "Annual colonoscopy from 25yr; annual cystoscopy+urine cytology (urothelial dominant); IHC MMR biliary specimen; annual skin exam Muir-Torre; annual endometrial sampling; aspirin CAPP2",
        "targeted_rx": "Pembrolizumab FDA2017 dMMR/MSI-H; dostarlimab; gemcitabine+cisplatin+durvalumab TOPAZ-1; FOLFOX/CAPOX CRC; atezolizumab MSI-H; aspirin CAPP2",
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 Autosomal-Dominant-LOF -- 156aa -- "
            "p16-INK4A-p14-ARF-Dual-Tumour-Suppressor-17kDa-"
            "FAMMM-CCA-3-5x-Elevated-Pancreatic-17-39pct-HIGHEST-Hereditary-"
            "Annual-MRI-EUS-40yr-MANDATORY-Melanoma-28-67pct-"
            "Palbociclib-CDK4-6i-RESISTANCE-p16-Null-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / 17 kDa / 9p21.3 CDKN2A biliary cancer molecular context: "
            "STRUCTURE: "
            "  156 aa / 17 kDa; ankyrin repeats (aa 1-130): CDK4/CDK6 inhibition (p16/INK4A); "
            "  Alternate reading frame: p14/ARF (completely different protein) → MDM2 inhibitor → p53 stabilisation; "
            "CANCER RISKS (BILIARY FOCUS): "
            "  CCA (extrahepatic dominant): 3-5x elevated; "
            "  PANCREATIC: 17-39% lifetime HIGHEST hereditary PDAC risk; "
            "  MELANOMA: 28-67% lifetime (dominant risk in FAMMM); "
            "  GASTRIC: elevated; "
            "PALBOCICLIB RESISTANCE IN CDKN2A-NULL BILIARY: "
            "  Somatic CDKN2A deletion COMMON in biliary CCA (30-40% tumours); "
            "  Germline CDKN2A → somatic second hit → CDK4/6 pathway ACTIVATED; "
            "  PALBOCICLIB (CDK4/6i) INEFFECTIVE if p16/INK4A already null; "
            "  AVAPRITINIB/ABEMACICLIB not affected by CDKN2A null; "
            "KEY MANAGEMENT (BILIARY): "
            "  ANNUAL MRI PANCREAS + EUS from 40yr MANDATORY (pancreatic 17-39%); "
            "  ANNUAL MRCP from 45yr (biliary CCA surveillance); "
            "  ANNUAL FULL BODY SKIN EXAM from 18yr (melanoma 28-67%); "
            "  ANNUAL MRI skin-mapping high-risk sites"
        ),
        "syndrome": "Familial Atypical Multiple Mole Melanoma (FAMMM) / Melanoma-Pancreatic Cancer syndrome",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "biliary_risk": "CCA 3-5x elevated; pancreatic 17-39% HIGHEST hereditary; melanoma 28-67% dominant",
        "pathognomonic": "Multiple atypical/dysplastic naevi (FAMMM); pancreatic cancer + melanoma family history PATHOGNOMONIC trigger; somatic CDKN2A loss = palbociclib RESISTANCE in CCA",
        "key_avoid": "Do NOT use palbociclib (CDK4/6i) in CDKN2A-null biliary CCA (resistance — p16 null = CDK4/6 pathway bypass); annual pancreatic MRI+EUS MANDATORY from 40yr (17-39% — highest single-gene pancreatic risk hereditary); full body skin exam MANDATORY from 18yr (melanoma 28-67%)",
        "key_rule": "CDKN2A = HIGHEST single-gene hereditary pancreatic risk (17-39%). Rule: annual MRI+EUS pancreas from 40yr MANDATORY for all CDKN2A carriers. FAMMM atypical naevi → full body map annually. Palbociclib RESISTANCE if p16/INK4A null biliary tumour — check before CDK4/6i therapy",
        "surveillance": "Annual MRI+EUS pancreas from 40yr; annual MRCP from 45yr; annual full body skin exam+mapping from 18yr; annual TVUS endometrial (females); colonoscopy from 50yr",
        "targeted_rx": "Gemcitabine+cisplatin+durvalumab TOPAZ-1; pembrolizumab MSI-H; FOLFIRINOX/gemcitabine-based pancreatic CCA; CDK4/6i AVOID (p16 null resistance); MEK inhibitors for FAMMM melanoma context; immunotherapy biliary PDL1+",
    },
    {
        "gene": "STK11",
        "protein": (
            "STK11 -- 19p13.3 Autosomal-Dominant-LOF -- 433aa -- "
            "LKB1-AMPK-Master-Kinase-48kDa-"
            "PJS-Biliary-Polyps-CCA-Elevated-Gastric-29pct-"
            "Mucocutaneous-Macules-PATHOGNOMONIC-"
            "SCTAT-Ovarian-PATHOGNOMONIC-Adenoma-Malignum-Cervix-PATHOGNOMONIC-"
            "Small-Bowel-13pct-HIGHEST-Pancreatic-11-36pct-HIGHEST-OMIM-175200"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "433 aa / 48 kDa / 19p13.3 STK11 biliary cancer molecular context: "
            "STRUCTURE: "
            "  433 aa / 48 kDa; Kinase domain (aa 51-313): AMPK activation; "
            "  N-terminal (aa 1-50): farnesylation; "
            "  C-terminal (aa 314-433): STRADα/MO25 binding; "
            "CANCER RISKS (BILIARY FOCUS): "
            "  BILIARY POLYPS: hamartomatous polyps in gallbladder and biliary tree; "
            "  CCA: elevated (biliary hamartomas → malignant transformation); "
            "  GALLBLADDER: 2-3x elevated; "
            "  PANCREATIC: 11-36% lifetime HIGHEST hereditary PDAC; "
            "  GASTRIC: 29%; SMALL BOWEL: 13% HIGHEST; CRC: 39%; "
            "PJS PATHOGNOMONIC: "
            "  Perioral mucocutaneous macules (lips, buccal mucosa): PATHOGNOMONIC — fade with age; "
            "  SCTAT (sex-cord tumour with annular tubules): ovarian PATHOGNOMONIC; "
            "  Adenoma malignum cervix: PATHOGNOMONIC; "
            "  Hamartomatous polyps: small bowel dominant, biliary/gallbladder; "
            "KEY MANAGEMENT (BILIARY): "
            "  ANNUAL EUS + MRI pancreas from 30yr (EARLIEST hereditary PDAC surveillance); "
            "  ANNUAL MRCP from 30yr (biliary polyps + CCA surveillance); "
            "  CAPSULE ENDOSCOPY + MR enterography from 8yr (small bowel HIGHEST 13%); "
            "  ANNUAL EGD + COLONOSCOPY from 18yr"
        ),
        "syndrome": "Peutz-Jeghers syndrome (PJS)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "biliary_risk": "Biliary polyps + CCA elevated; gallbladder 2-3x; pancreatic 11-36% HIGHEST hereditary",
        "pathognomonic": "Perioral mucocutaneous macules PATHOGNOMONIC (fade with age); SCTAT ovarian PATHOGNOMONIC; adenoma malignum cervix PATHOGNOMONIC; biliary hamartomatous polyps",
        "key_avoid": "Do NOT miss perioral macules (fade with age — diagnose in childhood); do NOT miss biliary hamartomatous polyps on EGD/MRCP (malignant transformation risk); do NOT delay pancreatic surveillance to 40yr — START FROM 30yr (earliest hereditary PDAC syndrome); intussusception in small bowel = surgical emergency",
        "key_rule": "STK11 = EARLIEST pancreatic surveillance (from 30yr — earlier than all other hereditary syndromes). Rule: annual MRCP from 30yr for biliary polyps + EUS+MRI pancreas from 30yr (11-36% PDAC). Perioral macules PATHOGNOMONIC — diagnose in childhood before age-related fading",
        "surveillance": "Annual MRCP+biliary MRI from 30yr; annual EUS+MRI pancreas from 30yr; capsule endoscopy+MR enterography from 8yr; annual EGD+colonoscopy from 18yr; annual breast MRI from 25yr (50-54%); annual gynaecological exam (SCTAT, adenoma malignum); testicular exam annual (males)",
        "targeted_rx": "Gemcitabine+cisplatin+durvalumab TOPAZ-1; FOLFIRINOX/gemcitabine-based PDAC (STK11 pancreatic); sirolimus/rapamycin mTOR (LKB1-AMPK pathway); pembrolizumab MSI-H; surgical debulking biliary polyp burden; MRCP-guided biliary stenting if obstruction",
    },
]


def _generate_patient(gene_data: dict, seed: int) -> dict:
    rng = random.Random(seed)
    gene = gene_data["gene"]

    age_ranges = {
        "BAP1":   (35, 65), "BRCA1":  (38, 70), "BRCA2":  (40, 72),
        "ATM":    (42, 68), "MLH1":   (35, 65), "MSH2":   (32, 62),
        "CDKN2A": (38, 70), "STK11":  (25, 62),
    }
    lo, hi = age_ranges.get(gene, (35, 65))
    age = rng.randint(lo, hi)

    biliary_types = {
        "BAP1":   ["Intrahepatic CCA", "Mesothelioma (primary)", "Uveal melanoma (primary)", "RCC (primary)"],
        "BRCA1":  ["Intrahepatic CCA", "Extrahepatic CCA", "Gallbladder carcinoma", "Pancreatic (primary)"],
        "BRCA2":  ["Intrahepatic CCA", "Extrahepatic CCA", "Gallbladder carcinoma", "Pancreatic (primary)"],
        "ATM":    ["Intrahepatic CCA", "Extrahepatic CCA", "Pancreatic (primary)", "Gallbladder carcinoma"],
        "MLH1":   ["Bile duct CCA (Lynch)", "Intrahepatic CCA", "CRC (primary Lynch)", "Endometrial (primary)"],
        "MSH2":   ["Bile duct CCA (Lynch)", "Gallbladder carcinoma", "Urothelial (primary)", "CRC (primary)"],
        "CDKN2A": ["Extrahepatic CCA", "Intrahepatic CCA", "Pancreatic (primary)", "Melanoma (primary)"],
        "STK11":  ["Biliary/gallbladder (PJS)", "Pancreatic (primary PJS)", "Gastric (primary PJS)", "Small bowel (primary)"],
    }
    bwts = {
        "BAP1":   [0.40, 0.35, 0.15, 0.10],
        "BRCA1":  [0.45, 0.25, 0.15, 0.15],
        "BRCA2":  [0.40, 0.30, 0.15, 0.15],
        "ATM":    [0.35, 0.25, 0.30, 0.10],
        "MLH1":   [0.40, 0.20, 0.25, 0.15],
        "MSH2":   [0.35, 0.20, 0.30, 0.15],
        "CDKN2A": [0.30, 0.25, 0.35, 0.10],
        "STK11":  [0.30, 0.40, 0.20, 0.10],
    }
    btypes = biliary_types.get(gene, ["Intrahepatic CCA"])
    bwt = bwts.get(gene, [1.0/len(btypes)]*len(btypes))
    tumour_type = rng.choices(btypes, weights=bwt)[0]

    treatments = {
        "BAP1":   ["Gemcitabine+Cisplatin+Durvalumab TOPAZ-1", "Nivolumab mesothelioma", "Tebentafusp uveal melanoma FDA2022", "Pembrolizumab MSI-H", "Surveillance only"],
        "BRCA1":  ["Gemcitabine+Cisplatin+Durvalumab TOPAZ-1", "Olaparib HRD-positive maintenance", "Pembrolizumab dMMR", "RRSO risk-reducing", "Bilateral mastectomy"],
        "BRCA2":  ["Gemcitabine+Cisplatin+Durvalumab TOPAZ-1", "Olaparib HRD-positive POLO-paradigm", "Niraparib HRD", "Gemcitabine+nab-paclitaxel PDAC", "Pembrolizumab MSI-H"],
        "ATM":    ["Gemcitabine+Cisplatin+Durvalumab TOPAZ-1", "Ceralasertib ATRi synthetic lethality", "Olaparib HRD-positive", "RT dose-reduced 20-30% MANDATORY", "EUS surveillance PDAC"],
        "MLH1":   ["Pembrolizumab dMMR FDA2017", "Gemcitabine+Cisplatin+Durvalumab TOPAZ-1", "Dostarlimab", "Aspirin CAPP2 Lynch prevention", "FOLFOX CRC dominant"],
        "MSH2":   ["Pembrolizumab dMMR FDA2017", "Gemcitabine+Cisplatin+Durvalumab TOPAZ-1", "Atezolizumab MSI-H", "Aspirin CAPP2", "Annual cystoscopy urothelial dominant"],
        "CDKN2A": ["Gemcitabine+Cisplatin+Durvalumab TOPAZ-1", "FOLFIRINOX PDAC", "Pembrolizumab MSI-H", "Annual MRI+EUS PDAC surveillance", "Immunotherapy PDL1+"],
        "STK11":  ["Gemcitabine+Cisplatin+Durvalumab TOPAZ-1", "FOLFIRINOX PDAC STK11", "Sirolimus mTOR PJS", "Biliary stenting MRCP", "Surgical polypectomy debulking"],
    }
    treatment = rng.choice(treatments.get(gene, ["Gemcitabine+Cisplatin+Durvalumab"]))

    hrd_positive   = gene in ("BRCA1", "BRCA2", "ATM") and rng.random() < 0.72
    msi_h          = gene in ("MLH1", "MSH2") and rng.random() < 0.80
    immunotherapy  = (hrd_positive or msi_h) and rng.random() < 0.65
    mbait          = gene == "BAP1" and rng.random() < 0.45
    uveal_melanoma = gene == "BAP1" and rng.random() < 0.20
    adherent       = rng.random() < 0.76

    return {
        "gene":            gene,
        "age_at_dx":       age,
        "tumour_type":     tumour_type,
        "treatment":       treatment,
        "hrd_positive":    hrd_positive,
        "msi_h":           msi_h,
        "immunotherapy_eligible": immunotherapy,
        "mbait_present":   mbait,
        "uveal_melanoma":  uveal_melanoma,
        "surveillance_adherent": adherent,
    }


def _build_cohorts() -> dict:
    cohorts = {}
    for i, gd in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        cohorts[gd["gene"]] = [_generate_patient(gd, seed * 1000 + j) for j in range(40)]
    return cohorts


def generate_overview() -> dict:
    cohorts = _build_cohorts()
    all_pts = [p for c in cohorts.values() for p in c]

    gene_counts   = {g: len(cohorts[g]) for g in cohorts}
    ages          = [p["age_at_dx"] for p in all_pts]
    immunotherapy = round(100 * sum(1 for p in all_pts if p["immunotherapy_eligible"]) / len(all_pts), 1)
    hrd_rate      = round(100 * sum(1 for p in all_pts if p["hrd_positive"]) / len(all_pts), 1)
    msi_h_rate    = round(100 * sum(1 for p in all_pts if p["msi_h"]) / len(all_pts), 1)
    adherent_rate = round(100 * sum(1 for p in all_pts if p["surveillance_adherent"]) / len(all_pts), 1)
    mbait_n       = sum(1 for p in all_pts if p["mbait_present"])
    uveal_n       = sum(1 for p in all_pts if p["uveal_melanoma"])

    key_facts = [
        "BAP1 TPDS: mesothelioma 30-60% HIGHEST hereditary; MBAITs PATHOGNOMONIC; AVOID ASBESTOS ABSOLUTELY",
        "BRCA2 > BRCA1 for biliary CCA risk (3-4x vs 2-3x); olaparib POLO paradigm for HRD-positive biliary",
        "ATM monoallelic = radiosensitive biliary CCA — REDUCE RT 20-30% even heterozygotes; communicate to RO",
        "MLH1 Lynch biliary: MSI-H → pembrolizumab FDA2017; aspirin CAPP2 600mg 50% Lynch risk reduction (RCT)",
        "MSH2 EPCAM 3-prime deletion silences MSH2 → MLPA MANDATORY; urothelial 10-14% HIGHEST Lynch gene",
        "CDKN2A: palbociclib (CDK4/6i) RESISTANCE in p16-null biliary; annual MRI+EUS pancreas from 40yr MANDATORY",
        "STK11 PJS: biliary polyps + pancreatic 11-36% → annual MRCP+EUS from 30yr EARLIEST hereditary PDAC",
        "TOPAZ-1 regimen (gemcitabine+cisplatin+durvalumab): first-line standard for advanced biliary tract cancer",
        "Universal MMR IHC on ALL biliary cancers: Lynch detection — MSI-H = pembrolizumab eligible",
        "HRD testing (SBS3+HRDetect) on biliary tumour: guides cisplatin preference + olaparib eligibility",
    ]

    # Compute biliary-specific stats for the page
    _biliary_kw = ("CCA", "Gallbladder", "Biliary", "biliary", "Bile")
    def _is_biliary(tt): return any(k in tt for k in _biliary_kw)
    def _is_icca(tt):    return "Intrahepatic" in tt
    def _is_parp(p):     return p["hrd_positive"] or p["msi_h"]

    biliary_total_n    = sum(1 for p in all_pts if _is_biliary(p["tumour_type"]))
    icca_n             = sum(1 for p in all_pts if _is_icca(p["tumour_type"]))
    parp_candidate_n   = sum(1 for p in all_pts if _is_parp(p))
    total_n            = len(all_pts)

    gene_summary = []
    for gd in ATLAS_GENES:
        gene    = gd["gene"]
        pts     = cohorts[gene]
        bn      = sum(1 for p in pts if _is_biliary(p["tumour_type"]))
        gene_summary.append({
            "gene":           gene,
            "locus":          gd["locus"],
            "n":              len(pts),
            "biliary_n":      bn,
            "biliary_pct":    round(100 * bn / len(pts), 1),
            "key_distinctions": [gd["biliary_risk"][:80]],
        })

    genes_detail = [{
        "gene":           gd["gene"],
        "inheritance":    gd["inheritance"],
        "cancer_risk":    gd["biliary_risk"],
        "pathognomonic":  gd["pathognomonic"],
        "surveillance_key": gd["surveillance"][:120],
    } for gd in ATLAS_GENES]

    return {
        "atlas":          "Hereditary-Biliary-Cancer-Predisposition-Atlas",
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "genes_n":        len(ATLAS_GENES),
        "total_patients": total_n,
        "seed_range":     f"{SEED_BASE}–{SEED_BASE + 7}",
        "gene_counts":    gene_counts,
        "biliary_total_n":   biliary_total_n,
        "biliary_total_pct": round(100 * biliary_total_n / total_n, 1),
        "icca_n":            icca_n,
        "icca_pct":          round(100 * icca_n / total_n, 1),
        "parp_candidate_n":  parp_candidate_n,
        "parp_candidate_pct": round(100 * parp_candidate_n / total_n, 1),
        "gene_summary":   gene_summary,
        "genes_detail":   genes_detail,
        "mean_age_at_dx": round(sum(ages) / len(ages), 1),
        "min_age_at_dx":  min(ages),
        "max_age_at_dx":  max(ages),
        "immunotherapy_eligible_pct": immunotherapy,
        "hrd_positive_pct": hrd_rate,
        "msi_h_rate_pct": msi_h_rate,
        "surveillance_adherent_pct": adherent_rate,
        "mbait_cases":    mbait_n,
        "uveal_melanoma_cases": uveal_n,
        "key_facts":      key_facts,
    }


def generate_breakdown() -> dict:
    cohorts = _build_cohorts()
    _biliary_kw = ("CCA", "Gallbladder", "Biliary", "biliary", "Bile")
    def _is_biliary(tt): return any(k in tt for k in _biliary_kw)
    def _is_icca(tt):    return "Intrahepatic" in tt

    _gene_locus = {gd["gene"]: gd["locus"] for gd in ATLAS_GENES}
    _gene_pathog = {gd["gene"]: gd["pathognomonic"] for gd in ATLAS_GENES}
    _gene_surv  = {gd["gene"]: gd["surveillance"][:120] for gd in ATLAS_GENES}
    _gene_dist  = {gd["gene"]: [gd["biliary_risk"][:80], gd["key_rule"][:80]] for gd in ATLAS_GENES}

    breakdown_array = []
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        pts  = cohorts[gene]
        bn   = sum(1 for p in pts if _is_biliary(p["tumour_type"]))
        icca_n = sum(1 for p in pts if _is_icca(p["tumour_type"]))
        breakdown_array.append({
            "gene":           gene,
            "locus":          gd["locus"],
            "n":              len(pts),
            "mean_age_onset": round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "biliary_n":      bn,
            "biliary_pct":    round(100 * bn / len(pts), 1),
            "icca_n":         icca_n,
            "seed":           SEED_BASE + i,
            "key_distinctions": _gene_dist[gene],
            "pathognomonic":  _gene_pathog[gene],
            "surveillance_key": _gene_surv[gene],
            "immunotherapy_eligible_n": sum(1 for p in pts if p["immunotherapy_eligible"]),
            "hrd_positive_n":           sum(1 for p in pts if p["hrd_positive"]),
            "msi_h_n":                  sum(1 for p in pts if p["msi_h"]),
        })
    return {"breakdown": breakdown_array, "total_patients": sum(len(v) for v in cohorts.values())}


def generate_definitions() -> dict:
    defs = {}
    for gd in ATLAS_GENES:
        defs[gd["gene"]] = {
            "gene":          gd["gene"],
            "locus":         gd["locus"],
            "protein_size":  gd["protein_size"],
            "syndrome":      gd["syndrome"],
            "inheritance":   gd["inheritance"],
            "biliary_risk":  gd["biliary_risk"],
            "pathognomonic": gd["pathognomonic"],
            "key_avoid":     gd["key_avoid"],
            "key_rule":      gd["key_rule"],
            "surveillance":  gd["surveillance"],
            "targeted_rx":   gd["targeted_rx"],
        }

    key_distinctions = [
        "BAP1 vs BRCA2 biliary: BAP1=mesothelioma dominant + MBAITs PATHOGNOMONIC; BRCA2=CCA strongest BRCA risk (3-4x) + POLO paradigm",
        "BRCA1 vs BRCA2 biliary: BRCA2 higher biliary risk (3-4x vs 2-3x); BRCA2 pancreatic 5-7% dominant extracolonic; both cisplatin-preferred HRD",
        "ATM radiosensitivity: monoallelic heterozygote = reduce RT 20-30% MANDATORY even in biliary CCA — not just biallelic A-T",
        "MLH1 vs MSH2 biliary Lynch: MLH1=bile-duct dominant CCA; MSH2=urothelial 10-14% HIGHEST + Muir-Torre sebaceous PATHOGNOMONIC",
        "CDKN2A palbociclib resistance: p16/INK4A null → CDK4/6 pathway bypassed → CDK4/6 inhibitors INEFFECTIVE in biliary CCA",
        "STK11 pancreatic surveillance: FROM 30YR (earliest hereditary PDAC syndrome — earlier than BRCA2 at 40yr, CDKN2A at 40yr)",
        "BAP1 AVOID ASBESTOS ABSOLUTELY: mesothelioma risk 30-60% baseline + asbestos = near-certain mesothelioma + synergistic dose",
        "Universal MMR IHC on biliary cancers: Lynch CCA historically missed — MSI-H biliary = pembrolizumab FDA2017 eligible",
        "TOPAZ-1 first line: gemcitabine+cisplatin+durvalumab = standard of care advanced biliary (all subtypes)",
        "HRD test guides cisplatin: BRCA1/2/ATM biliary → SBS3+HRDetect on tumour → cisplatin preferred (TOPAZ-1 backbone) + olaparib eligibility",
    ]

    # Convert to array format for page rendering
    defs_array = [{
        "term": f"{gd['gene']} — {gd['syndrome']}",
        "definition": (
            f"LOCUS: {gd['locus']} | INHERITANCE: {gd['inheritance']}\n"
            f"BILIARY RISK: {gd['biliary_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "definitions": defs_array,
        "gene_defs":   defs,
        "key_clinical_distinctions": key_distinctions,
        "seed_range": f"{SEED_BASE}–{SEED_BASE + 7}",
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(json.dumps({k: v for k, v in ov.items() if k != 'key_facts'}, indent=2))
    print("Key facts:", len(ov["key_facts"]))
