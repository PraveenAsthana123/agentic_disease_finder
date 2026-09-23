#!/usr/bin/env python3
"""Hereditary-Melanoma-Predisposition-Atlas -- Complete 8-Gene Reference
CDKN2A (Cyclin-dependent kinase inhibitor 2A; 156aa; 9p21.3; AD LOF;
         FAMMM / Familial Atypical Multiple Mole Melanoma;
         p16-INK4a + p14-ARF dual transcripts — HIGHEST risk melanoma gene;
         melanoma 40-50x RR; pancreatic cancer 20x;
         seed SEED_BASE+0) .
CDK4   (Cyclin-dependent kinase 4; 303aa; 12q14.1; AD GOF;
         Familial Melanoma CDK4 — R24C/R24H/G101W p16-binding pocket mutation;
         CDK4 gain-of-function bypasses G1/S arrest;
         melanoma 40-50x RR; CDK4/6i paradox (palbociclib);
         seed SEED_BASE+1) .
BAP1   (BRCA1-associated protein 1; 729aa; 3p21.1; AD LOF;
         BAP1 Tumour Predisposition Syndrome — uveal melanoma 50% PATHOGNOMONIC;
         cutaneous melanoma elevated; mesothelioma 8-10%;
         BAPomas/MBAITs PATHOGNOMONIC skin lesions;
         seed SEED_BASE+2) .
MITF   (Microphthalmia-associated transcription factor; 520aa; 3p14.1; AD LOF;
         MITF E318K European founder — melanoma 5x RR;
         uveal melanoma elevated; renal cell carcinoma 3-5x;
         melanocyte master regulator TF; SUMO acceptor K316R;
         seed SEED_BASE+3) .
POT1   (Protection of telomeres 1; 634aa; 7q31.33; AD LOF;
         POT1 Familial Melanoma — telomere capping defect;
         melanoma 4-6x RR; thyroid cancer elevated; glioma elevated;
         telomere ELONGATION paradox in carriers (distinguish from short telomere syndromes);
         seed SEED_BASE+4) .
TERT   (Telomerase reverse transcriptase; 1132aa; 5p15.33; AD LOF germline;
         TERT germline + promoter variants — melanoma predisposition;
         somatic C228T/C250T most common melanoma somatic mutations (NOT germline);
         germline LOF = short telomere disorders (DC/IPF) vs promoter GOF = melanoma;
         seed SEED_BASE+5) .
MC1R   (Melanocortin 1 receptor; 317aa; 16q24.3; AR/modifier;
         Red hair/skin phenotype modifier — R151C/R160W/D294H high-risk variants;
         ~2-3x RR per variant; critical CDKN2A risk amplifier;
         NOT standalone high-penetrance gene but mandates intensive surveillance;
         seed SEED_BASE+6) .
NF1    (Neurofibromin RAS-GAP; 2839aa; 17q11.2; AD LOF;
         NF1-associated melanoma — melanoma 2-3x RR;
         MPNST 8-13% PATHOGNOMONIC (sarcoma — doxorubicin, NOT immunotherapy);
         café-au-lait 6+ macules PATHOGNOMONIC; Lisch nodules PATHOGNOMONIC;
         selumetinib FDA2020 MEK inhibitor plexiform neurofibroma;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3422-3429)
"""
import random

SEED_BASE = 3422

ATLAS_GENES = [
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 Autosomal-Dominant-LOF -- 156aa -- "
            "p16-INK4a-15kDa-CDK4/6-Inhibitor-DUAL-TRANSCRIPT-p14-ARF-14kDa-MDM2-Sequestration-"
            "FAMMM-Familial-Atypical-Multiple-Mole-Melanoma-"
            "Melanoma-40-50x-RR-HIGHEST-Hereditary-"
            "Pancreatic-Cancer-20x-HIGHEST-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / 15 kDa / 9p21.3 CDKN2A melanoma molecular context: "
            "STRUCTURE: "
            "  156 aa / 15 kDa (p16-INK4a isoform); 4 ankyrin repeats — CDK4/CDK6 binding; "
            "  p14-ARF: 132aa / 14kDa — alternative reading frame, same exon 2, different promoter; "
            "  p16: binds CDK4/6 → prevents cyclin-D1/CDK4-6 → prevents RB1 phosphorylation → G1 arrest; "
            "  p14-ARF: sequesters MDM2 → stabilises p53 → p53 transcriptional activation; "
            "  DUAL LOSS: p16 LOF → CDK4/6 hyperactive → RB1 bypassed; p14 LOF → MDM2 degrades p53; "
            "CANCER RISKS: "
            "  Cutaneous melanoma: 40-50x RR HIGHEST hereditary melanoma gene; ~67% penetrance by age 80yr; "
            "  Pancreatic cancer: 20x RR HIGHEST; cumulative risk ~17% by age 75yr; "
            "  Multiple primary melanomas PATHOGNOMONIC: 2nd primary risk 3-5x after first diagnosis; "
            "  Atypical/dysplastic naevi 50+: PATHOGNOMONIC FAMMM phenotype; "
            "  Ocular melanoma moderately elevated; "
            "KEY MANAGEMENT: "
            "  Annual full-body dermatology skin exam from age 18yr MANDATORY; "
            "  Annual endoscopic surveillance for pancreatic cancer from age 40-50yr or 10yr before youngest case; "
            "  Cascade CDKN2A germline testing first-degree relatives; "
            "  Sun-protective behaviour MANDATORY (UV-radiation is the major environmental trigger); "
            "  Dermoscopy MANDATORY for atypical naevi surveillance"
        ),
        "syndrome": "FAMMM — Familial Atypical Multiple Mole Melanoma; pancreatic cancer predisposition",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); LOH 9p21.3 somatic second hit",
        "melanoma_risk": "Melanoma 40-50x RR HIGHEST hereditary risk; ~67% penetrance by age 80yr; multiple primaries",
        "pathognomonic": "50+ atypical naevi FAMMM PATHOGNOMONIC; multiple primary melanomas PATHOGNOMONIC; pancreatic cancer cluster",
        "key_avoid": "Do NOT delay annual skin surveillance — UV exposure + germline CDKN2A = 40-50x amplified risk; screen pancreas from age 40-50yr",
        "key_rule": "CDKN2A: annual dermatology from age 18 MANDATORY. Annual EUS/MRI-MRCP pancreas from age 40-50. Cascade testing. Sun protection lifelong. Multiple primaries high risk.",
        "surveillance": "Annual full-body skin exam + dermoscopy from age 18yr; annual whole-body photography baseline 18yr; annual EUS + MRI-MRCP pancreas from 40-50yr; annual ophthalmology; cascade CDKN2A first-degree germline testing",
        "targeted_rx": "Pembrolizumab FDA2015 PD-1 advanced/unresectable melanoma; nivolumab FDA2015; BRAF+MEK inhibitors (dabrafenib+trametinib) if BRAF V600E/K somatic (not determined by CDKN2A germline); ipilimumab+nivolumab combo advanced melanoma; CDK4/6 inhibitor palbociclib emerging",
    },
    {
        "gene": "CDK4",
        "protein": (
            "CDK4 -- 12q14.1 Autosomal-Dominant-GOF -- 303aa -- "
            "CDK4-34kDa-Cyclin-D1-Partner-Serine-Threonine-Kinase-"
            "R24C-R24H-p16-Binding-Pocket-ABROGATED-G1-S-Checkpoint-Bypass-"
            "Familial-Melanoma-CDK4-40-50x-RR-"
            "CDK4-6-Inhibitor-Palbociclib-Paradox-OMIM-123829"
        ),
        "locus": "12q14.1",
        "protein_size": (
            "303 aa / 34 kDa / 12q14.1 CDK4 melanoma molecular context: "
            "STRUCTURE: "
            "  303 aa / 34 kDa; N-terminal PSTAIRE helix (cyclin D1-binding); "
            "  P-loop ATP-binding domain; activation T-loop (T172 phosphorylation by CAK); "
            "  p16/INK4a-binding pocket: R24 (aa 24) in DFG-motif adjacent — hotspot mutations; "
            "  CDK4 GOF mutations: R24C (c.70CT), R24H (c.71GA), G101W (c.301GA); "
            "  GOF mechanism: R24C/R24H disrupts p16-INK4a binding → CDK4 cannot be inhibited by p16; "
            "  Cyclin-D1/CDK4 phosphorylates RB1 → releases E2F transcription factors → S-phase entry; "
            "CANCER RISKS: "
            "  Cutaneous melanoma: 40-50x RR (comparable to CDKN2A); "
            "  Lower penetrance than CDKN2A in some cohorts (~50% by age 80yr); "
            "  CDK4 germline mutations: <5% all familial melanoma (rarer than CDKN2A); "
            "  Multiple primary melanomas risk elevated (same as CDKN2A phenotype); "
            "KEY MANAGEMENT: "
            "  Annual full-body skin exam from age 18yr MANDATORY; "
            "  CDK4/6 inhibitor PARADOX: palbociclib/ribociclib/abemaciclib primarily target CDK4 GOF → "
            "  theoretically less effective in CDK4-R24C melanoma (p16 already non-functional); "
            "  BRAF somatic testing of tumour mandatory (60-70% melanoma BRAF V600E); "
            "  Pancreatic surveillance LESS established than CDKN2A (CDK4 primarily melanoma-only risk)"
        ),
        "syndrome": "Familial Melanoma CDK4 — rare high-penetrance melanoma predisposition; distinct from CDKN2A FAMMM",
        "inheritance": "AD GOF (autosomal dominant gain-of-function) — kinase active but p16-resistant",
        "melanoma_risk": "Melanoma 40-50x RR; ~50% penetrance by age 80yr; less pancreatic risk than CDKN2A",
        "pathognomonic": "CDK4 R24C/R24H = G1/S checkpoint abolished; p16-INK4a cannot inhibit CDK4; atypical naevi phenotype",
        "key_avoid": "Do NOT rely solely on CDK4/6 inhibitors in CDK4-R24C tumours — GOF mutation abrogates p16-inhibitory mechanism; BRAF somatic testing mandatory",
        "key_rule": "CDK4 familial melanoma: annual dermatology from 18yr MANDATORY (same as CDKN2A). BRAF somatic testing of tumour mandatory. CDK4/6i palbociclib mechanistic paradox in germline CDK4 carriers.",
        "surveillance": "Annual full-body skin exam + dermoscopy from age 18yr; baseline whole-body photography 18yr; cascade CDK4 germline testing first-degree relatives; annual ophthalmology; BRAF/NRAS/NF1 somatic panel mandatory on tumour",
        "targeted_rx": "Pembrolizumab FDA2015 PD-1 checkpoint inhibitor advanced melanoma; dabrafenib+trametinib BRAF+MEK if somatic BRAF V600E (CDK4 germline ≠ BRAF somatic); ipilimumab CTLA4; CDK4/6 inhibitor palbociclib (mechanistic note: GOF R24C carriers may respond differently); nivolumab+ipilimumab combo",
    },
    {
        "gene": "BAP1",
        "protein": (
            "BAP1 -- 3p21.1 Autosomal-Dominant-LOF -- 729aa -- "
            "BAP1-80kDa-Deubiquitinase-BRCA1-Associated-H2AK119ub1-"
            "BAP1-TPDS-Uveal-Melanoma-50pct-PATHOGNOMONIC-"
            "Cutaneous-Melanoma-Elevated-Mesothelioma-8-10pct-"
            "BAPomas-MBAITs-PATHOGNOMONIC-OMIM-603089"
        ),
        "locus": "3p21.1",
        "protein_size": (
            "729 aa / 80 kDa / 3p21.1 BAP1 melanoma molecular context: "
            "STRUCTURE: "
            "  729 aa / 80 kDa; N-terminal UCH domain (ubiquitin C-terminal hydrolase, aa 1-240); "
            "  BRCA1/2-binding domain; C-terminal BRCT-like domain; nuclear localisation signals; "
            "  BAP1 deubiquitinates H2AK119ub1 (histone H2A monoubiquitylated at K119); "
            "  PRC1 complex antagonist → chromatin remodelling → tumour suppressor function; "
            "  BAP1 LOF → H2AK119ub1 accumulation → transcriptional repression of tumour suppressors; "
            "CANCER RISKS (MELANOMA FOCUS): "
            "  Uveal melanoma: 50% lifetime PATHOGNOMONIC — highest penetrance in BAP1-TPDS; "
            "  Cutaneous melanoma: 3-5x RR elevated; atypical BAP1-mutated melanocytic tumours; "
            "  Mesothelioma: 8-10% lifetime HIGHEST hereditary MPM risk; "
            "  Renal cell carcinoma (clear cell): 10-14% lifetime; "
            "  BAP1 somatic inactivation: most common mutation in metastatic uveal melanoma; "
            "KEY MANAGEMENT: "
            "  Annual ophthalmology with UV/dilated slit-lamp + fundal photography MANDATORY; "
            "  ASBESTOS AVOIDANCE MANDATORY: BAP1 + asbestos co-exposure = ~100x MPM risk; "
            "  Annual full-body skin exam from age 20yr; "
            "  BAPomas/MBAITs: benign intradermal melanocytic tumours — PATHOGNOMONIC cascade trigger; "
            "  BAP1-IHC nuclear loss on any biopsy = universal screening trigger for germline testing"
        ),
        "syndrome": "BAP1 Tumour Predisposition Syndrome (BAP1-TPDS) — uveal melanoma, cutaneous melanoma, mesothelioma, RCC",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); LOH 3p21.1 somatic second hit",
        "melanoma_risk": "Uveal melanoma 50% PATHOGNOMONIC; cutaneous melanoma 3-5x elevated; multiple tumour types",
        "pathognomonic": "BAPomas/MBAITs = PATHOGNOMONIC cascade trigger; uveal melanoma 50% PATHOGNOMONIC; BAP1-IHC nuclear loss universal marker",
        "key_avoid": "ASBESTOS AVOIDANCE MANDATORY — BAP1 + asbestos synergy ~100x MPM risk; do NOT miss annual ophthalmology (uveal melanoma 50% lifetime)",
        "key_rule": "BAP1: annual ophthalmology MANDATORY (uveal 50%). Annual skin exam. ASBESTOS MANDATORY avoided. BAPomas = cascade germline test trigger. BAP1-IHC nuclear loss on ANY specimen = screen germline.",
        "surveillance": "Annual ophthalmology UV slit-lamp + fundal photography from age 20yr; annual full-body skin exam + dermoscopy; annual chest CT from 40yr (mesothelioma/RCC); annual renal ultrasound/MRI; cascade BAP1 germline testing first-degree relatives",
        "targeted_rx": "Tebentafusp FDA2022 (uveal melanoma — first FDA-approved TCR bispecific; BAP1-mutant metastatic uveal melanoma responds); pembrolizumab advanced cutaneous melanoma; selumetinib MEK emerging BAP1 RCC; ipilimumab+nivolumab metastatic uveal melanoma (GNAQ/GNA11 primary); cisplatin/pemetrexed mesothelioma",
    },
    {
        "gene": "MITF",
        "protein": (
            "MITF -- 3p14.1 Autosomal-Dominant-LOF -- 520aa -- "
            "MITF-59kDa-bHLH-LZ-Master-Melanocyte-Transcription-Factor-"
            "E318K-European-Founder-SUMO-Acceptor-K316-Disruption-"
            "Melanoma-5x-RR-Uveal-Melanoma-Renal-Cell-Carcinoma-3-5x-"
            "Waardenburg-Type2A-LOF-Tietz-OMIM-156845"
        ),
        "locus": "3p14.1",
        "protein_size": (
            "520 aa / 59 kDa / 3p14.1 MITF melanoma molecular context: "
            "STRUCTURE: "
            "  520 aa / 59 kDa; N-terminal transactivation domain; bHLH domain (aa 213-268); "
            "  Leucine-zipper dimerisation domain; C-terminal domain; "
            "  E318K mutation (c.952GA): SUMO acceptor K316 disruption → impaired SUMOylation; "
            "  SUMOylation normally represses MITF transcriptional activity; E318K → enhanced MITF → "
            "  increased proliferation/survival; promotes MET pathway; "
            "  MITF = master melanocyte regulator: controls DCT, TYR, TYRP1, MLANA, RAB27A; "
            "CANCER RISKS: "
            "  Melanoma: 5x RR European E318K founder; ~15% penetrance by age 80yr (moderate penetrance); "
            "  Uveal melanoma: 3-4x RR in MITF E318K carriers; "
            "  Renal cell carcinoma: 3-5x RR (MITF on TFE3/TFEB pathway); "
            "  E318K population frequency: ~2% European general population; 5-8% melanoma cohorts; "
            "KEY MANAGEMENT: "
            "  Annual full-body skin exam from age 25yr; "
            "  Annual ophthalmology from age 25yr (uveal melanoma elevated); "
            "  MITF E318K: moderate-penetrance — risk stratify using personal/family history; "
            "  Melanoma surveillance intensified if additional MC1R variants co-inherited; "
            "  Renal imaging consideration from age 40yr"
        ),
        "syndrome": "MITF E318K familial melanoma; uveal melanoma predisposition; renal cell carcinoma elevation",
        "inheritance": "AD LOF E318K (impaired SUMOylation → MITF gain of transcriptional activity paradox)",
        "melanoma_risk": "Melanoma 5x RR; uveal melanoma 3-4x; RCC 3-5x; moderate penetrance ~15% by age 80yr",
        "pathognomonic": "MITF E318K + MC1R = compounded melanoma risk; RCC + melanoma combination in one patient raises MITF suspicion",
        "key_avoid": "Do NOT classify MITF E318K as benign variant — moderate-penetrance melanoma predisposition; annual skin + ophthalmology surveillance required",
        "key_rule": "MITF E318K: annual skin exam from 25yr. Annual ophthalmology. Renal imaging from 40yr. Risk compounded with MC1R variants. Moderate penetrance — counsel appropriately.",
        "surveillance": "Annual full-body skin exam + dermoscopy from age 25yr; annual ophthalmology slit-lamp from 25yr; renal ultrasound/MRI from age 40yr; cascade MITF germline testing if family history confirms segregation; BRAF/NRAS somatic panel on tumour",
        "targeted_rx": "Pembrolizumab PD-1 advanced melanoma; BRAF+MEK inhibitors if somatic BRAF V600 (independent of MITF germline); ipilimumab CTLA4; tebentafusp uveal melanoma if BAP1/uveal component; sunitinib/pazopanib RCC component",
    },
    {
        "gene": "POT1",
        "protein": (
            "POT1 -- 7q31.33 Autosomal-Dominant-LOF -- 634aa -- "
            "POT1-71kDa-OB-Fold-Telomere-Single-Strand-TTAGGG-Capping-"
            "CST-Complex-AAA-Domain-CTC1-STN1-"
            "Familial-Melanoma-Thyroid-Glioma-Elevated-"
            "Telomere-ELONGATION-Paradox-OMIM-606478"
        ),
        "locus": "7q31.33",
        "protein_size": (
            "634 aa / 71 kDa / 7q31.33 POT1 melanoma molecular context: "
            "STRUCTURE: "
            "  634 aa / 71 kDa; N-terminal OB1 fold (aa 1-89) + OB2 fold (aa 100-190) — ssDNA binding; "
            "  C-terminal domain: TPP1-binding; helical domain; POT1 binds 3' single-strand G-overhang; "
            "  CST (CTC1-STN1-TEN1) complex participates in fill-in synthesis; "
            "  POT1 LOF: telomere ends uncapped → ATR kinase activation → G-overhang extension; "
            "  PARADOX: POT1 LOF → telomere ELONGATION (not shortening) — ATM/ATR disabled at telomere; "
            "  Distinguish from germline TERT/TERC LOF (short telomere diseases: DC/IPF/pulmonary fibrosis); "
            "CANCER RISKS: "
            "  Cutaneous melanoma: 4-6x RR; European and South Asian founder variants described; "
            "  Thyroid cancer (follicular/papillary): 3-5x RR; "
            "  Glioma: 2-3x RR; chronic lymphocytic leukaemia: elevated; "
            "  Angiosarcoma: case reports in POT1 LOF carriers; "
            "KEY MANAGEMENT: "
            "  Annual full-body skin exam from age 25yr; "
            "  Annual thyroid ultrasound from age 25yr; "
            "  Brain MRI consideration from 40yr (glioma surveillance); "
            "  IMPORTANT: POT1 LOF → LONG telomeres — not a short telomere syndrome (DKC1/TERC LOF); "
            "  Do NOT confuse with dyskeratosis congenita (DC) telomere biology"
        ),
        "syndrome": "POT1 Familial Melanoma — multi-cancer predisposition (melanoma + thyroid + glioma)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); haploinsufficiency; telomere elongation paradox",
        "melanoma_risk": "Melanoma 4-6x RR; thyroid 3-5x; glioma 2-3x; CLL elevated",
        "pathognomonic": "Telomere ELONGATION (NOT shortening) in carriers PATHOGNOMONIC — distinguishes from DC/IPF; melanoma+thyroid cluster raises POT1 suspicion",
        "key_avoid": "Do NOT confuse POT1 LOF (long telomeres → melanoma/thyroid/glioma) with TERT/TERC/DKC1 LOF (short telomeres → DC/IPF/BMF) — opposite telomere phenotype",
        "key_rule": "POT1: annual skin exam from 25yr. Annual thyroid US from 25yr. Telomere ELONGATION distinguishes from short-telomere syndromes. Brain MRI from 40yr. Cascade testing.",
        "surveillance": "Annual full-body skin exam + dermoscopy from age 25yr; annual thyroid ultrasound from 25yr; brain MRI from age 40yr (glioma); annual lymphocyte count (CLL risk); telomere length testing (long telomeres PATHOGNOMONIC POT1 vs short = DC); cascade POT1 germline first-degree relatives",
        "targeted_rx": "Pembrolizumab PD-1 advanced melanoma; BRAF+MEK inhibitors if somatic BRAF V600; radioiodine I-131 thyroid cancer; temozolomide glioma; ibrutinib CLL component; ipilimumab+nivolumab advanced melanoma",
    },
    {
        "gene": "TERT",
        "protein": (
            "TERT -- 5p15.33 Autosomal-Dominant -- 1132aa -- "
            "TERT-127kDa-Telomerase-Reverse-Transcriptase-Catalytic-Subunit-"
            "Germline-LOF-Short-Telomere-DC-IPF-vs-Promoter-GOF-Melanoma-"
            "Somatic-C228T-C250T-Most-Common-Melanoma-Somatic-Mutation-GLOBALLY-"
            "Familial-Melanoma-Pulmonary-Fibrosis-Dyskeratosis-OMIM-187270"
        ),
        "locus": "5p15.33",
        "protein_size": (
            "1132 aa / 127 kDa / 5p15.33 TERT melanoma molecular context: "
            "STRUCTURE: "
            "  1132 aa / 127 kDa; N-terminal TEN domain (TERC anchoring); "
            "  TRBD domain (TERC RNA-binding); RT domain (reverse transcriptase); "
            "  CTE domain (C-terminal extension); "
            "  TERT + TERC (RNA template) = telomerase core complex; "
            "GERMLINE vs SOMATIC CRITICAL DISTINCTION: "
            "  Germline TERT LOF (coding): short telomere syndromes — DC, HH, IPF, liver cirrhosis; "
            "  Somatic TERT promoter C228T (chr5:1,295,228) + C250T (chr5:1,295,250): "
            "    → creates de-novo ETS transcription factor binding site → TERT promoter activation; "
            "    → Most common somatic MELANOMA mutation globally (60-70% melanoma); "
            "  Germline TERT promoter variants: elevated familial melanoma risk (moderate); "
            "CANCER RISKS: "
            "  Germline TERT promoter variants: melanoma 3-5x RR (moderate hereditary risk); "
            "  Germline TERT LOF coding: NOT primarily melanoma — DC/IPF/HH phenotype; "
            "KEY MANAGEMENT: "
            "  Annual full-body skin exam from age 25yr (germline promoter variant carriers); "
            "  TERT promoter somatic testing: routine on melanoma biopsy (high TERT somatic rate); "
            "  Distinguish germline promoter vs somatic: blood DNA vs tumour DNA testing mandatory"
        ),
        "syndrome": "TERT germline promoter — familial melanoma; TERT coding LOF — short telomere syndrome (DC/IPF/HH)",
        "inheritance": "AD — germline TERT promoter GOF variants → melanoma; germline TERT coding LOF → DC/IPF",
        "melanoma_risk": "Germline TERT promoter: melanoma 3-5x RR moderate; somatic C228T/C250T most common melanoma mutation 60-70%",
        "pathognomonic": "TERT C228T/C250T somatic PATHOGNOMONIC of melanoma (most frequent somatic melanoma mutation); germline promoter vs somatic distinction MANDATORY",
        "key_avoid": "Do NOT conflate germline TERT promoter (melanoma risk) with germline TERT coding LOF (dyskeratosis congenita/IPF) — completely different disease spectra; blood vs tumour DNA testing mandatory",
        "key_rule": "TERT germline promoter: annual skin exam from 25yr. Distinguish germline vs somatic C228T/C250T (tumour testing). Germline coding LOF = DC/IPF NOT melanoma. Moderate penetrance melanoma predisposition.",
        "surveillance": "Annual full-body skin exam + dermoscopy from age 25yr (germline promoter variants); blood DNA TERT germline testing to distinguish germline vs somatic; annual thyroid US (TERT amplified thyroid cancers); telomere length testing (normal or long in germline promoter variants vs short in coding LOF)",
        "targeted_rx": "Pembrolizumab PD-1 melanoma TERT promoter somatic positive; BRAF+MEK inhibitors if somatic BRAF V600; ipilimumab CTLA4; nivolumab+ipilimumab combo; TERT promoter status used as prognostic biomarker (C228T = worse prognosis)",
    },
    {
        "gene": "MC1R",
        "protein": (
            "MC1R -- 16q24.3 AR-Modifier -- 317aa -- "
            "MC1R-35kDa-GPCR-7-TM-Melanocortin-1-Receptor-"
            "Red-Hair-Color-Fair-Skin-Freckles-Phenotype-"
            "R151C-R160W-D294H-High-Risk-Variants-2-3x-RR-per-Variant-"
            "CDKN2A-Risk-Amplifier-Modifier-NOT-Standalone-OMIM-155555"
        ),
        "locus": "16q24.3",
        "protein_size": (
            "317 aa / 35 kDa / 16q24.3 MC1R melanoma molecular context: "
            "STRUCTURE: "
            "  317 aa / 35 kDa; 7 transmembrane G-protein coupled receptor; "
            "  Extracellular N-terminus; 3 extracellular loops; intracellular C-terminus; "
            "  MC1R binds MSH (α-melanocyte stimulating hormone): cAMP → eumelanin (dark, UV-protective); "
            "  MC1R LOF: ASIP dominant → phaeomelanin (red/yellow, UV-sensitising); "
            "  High-risk variants R151C/R160W/D294H: loss-of-function (RHC = Red Hair Colour variants); "
            "  Low-risk variants V60L/V92M: partial function reduction; "
            "CANCER RISKS: "
            "  Cutaneous melanoma: 2-3x RR per single high-risk variant; "
            "  CRITICAL: MC1R AMPLIFIES CDKN2A RISK — CDKN2A carrier + MC1R = up to 100x RR; "
            "  Squamous cell carcinoma: 2-3x (UV-damage accumulation); "
            "  BCC basal cell carcinoma: elevated; "
            "  NOT a standalone high-penetrance cancer predisposition gene; "
            "KEY MANAGEMENT: "
            "  Annual full-body skin exam + dermoscopy if 2+ high-risk variants or + CDKN2A; "
            "  MC1R compound with CDKN2A = MANDATORY intensive surveillance from age 18yr; "
            "  Strict sun protection MANDATORY (MC1R impaired UV-protection); "
            "  MC1R testing in CDKN2A families: critical for risk stratification; "
            "  NOT recommended as standalone population screening — use in familial context"
        ),
        "syndrome": "MC1R red hair/fair skin melanoma modifier — risk amplifier for CDKN2A and standalone moderate risk",
        "inheritance": "AR modifier (biallelic high-risk variants) or AD modifier (single high-risk variant moderate risk)",
        "melanoma_risk": "2-3x RR per single high-risk variant; CDKN2A+MC1R compound = up to 100x RR; SCC/BCC also elevated",
        "pathognomonic": "Red hair, fair skin, freckles phenotype; MC1R R151C/R160W/D294H = high-risk PATHOGNOMONIC variants; CDKN2A+MC1R amplification",
        "key_avoid": "Do NOT report MC1R in isolation as high-penetrance cancer gene — it is a modifier; compound with CDKN2A or CDK4 requires CDKN2A-level surveillance; avoid sun tanning/UV beds",
        "key_rule": "MC1R modifier: annual skin exam if 2+ high-risk variants OR + CDKN2A. CDKN2A+MC1R = maximum melanoma risk compound. Strict sun protection. Not recommended as standalone population screen.",
        "surveillance": "Annual full-body skin exam from age 25yr if 2+ MC1R variants; intensive annual from 18yr if MC1R + CDKN2A; baseline whole-body photography; strict broad-spectrum sunscreen (SPF50+) MANDATORY lifelong; avoid tanning beds absolutely",
        "targeted_rx": "Same as CDKN2A (pembrolizumab/nivolumab advanced melanoma; BRAF+MEK if somatic BRAF V600); MC1R does not directly predict treatment response; tumour BRAF/NRAS/NF1 somatic panel mandatory",
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RAS-GAP-Ras-GTPase-Accelerating-Protein-"
            "NF1-Associated-Melanoma-2-3x-RR-"
            "MPNST-8-13pct-SARCOMA-NOT-Melanoma-Doxorubicin-"
            "Cafe-au-Lait-Macules-PATHOGNOMONIC-Selumetinib-FDA2020-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 melanoma molecular context: "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; GRD domain (RAS-GAP, aa 1198-1530) — accelerates RAS GTP hydrolysis; "
            "  PH domain; SEC14 domain; ARM repeats; C-terminal PDZ-binding; "
            "  NF1 LOF → RAS-GTP accumulation → MAPK/ERK + PI3K/AKT hyperactivation; "
            "  Somatic NF1 mutations: 11-14% sporadic cutaneous melanoma (acquired); "
            "  Germline NF1 = NF1 syndrome (not pure melanoma predisposition); "
            "CANCER RISKS (MELANOMA FOCUS): "
            "  Cutaneous melanoma: 2-3x RR elevated in NF1 germline carriers; "
            "  NF1 TRIPLE WT MELANOMA: NF1 somatic LOF + wild-type BRAF + wild-type NRAS — RAS-MAP hyperactive; "
            "  MPNST: 8-13% lifetime PATHOGNOMONIC — DO NOT MISIDENTIFY as melanoma; SARCOMA; "
            "  Desmoid tumours; GIST; optic pathway glioma (children); "
            "CRITICAL CLINICAL DISTINCTION: "
            "  NF1-associated melanoma: NF1 somatic mutation driver → RAS pathway → MEK inhibitor targeted; "
            "  MPNST MIMICS melanoma on imaging — MPNST = sarcoma (doxorubicin/ifosfamide, NOT immunotherapy); "
            "KEY MANAGEMENT: "
            "  Annual full-body skin exam from age 18yr; "
            "  Selumetinib FDA2020 MEK inhibitor for plexiform neurofibromas; "
            "  Annual whole-body MRI for MPNST surveillance in high-risk NF1 (plexiform neurofibromas); "
            "  Café-au-lait 6+ macules PATHOGNOMONIC NF1 (>5mm prepubertal, >15mm postpubertal)"
        ),
        "syndrome": "Neurofibromatosis type 1 (NF1) — multi-tumour predisposition; MPNST, melanoma, GBM, GIST",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); de novo 50% cases; haploinsufficiency",
        "melanoma_risk": "Melanoma 2-3x RR; NF1 triple-WT melanoma subtype (NF1-somatic); MPNST 8-13% NOT melanoma",
        "pathognomonic": "Café-au-lait macules 6+ PATHOGNOMONIC NF1; Lisch nodules PATHOGNOMONIC; MPNST PATHOGNOMONIC (sarcoma)",
        "key_avoid": "CRITICAL: Do NOT treat NF1-associated MPNST as melanoma (immunotherapy). MPNST = sarcoma → doxorubicin/ifosfamide; immunotherapy NOT standard for MPNST. Selumetinib for plexiform NF, NOT MPNST",
        "key_rule": "NF1: annual skin exam from 18yr. MPNST = sarcoma NOT melanoma — doxorubicin/ifosfamide. Selumetinib FDA2020 plexiform neurofibroma. Café-au-lait 6+ PATHOGNOMONIC. Annual MRI plexiform surveillance.",
        "surveillance": "Annual full-body skin exam from age 18yr; annual ophthalmology (optic glioma children); annual whole-body MRI if plexiform neurofibromas (MPNST surveillance); annual blood pressure check (hypertension NF1); cascade NF1 germline first-degree relatives; selumetinib FDA2020 eligibility if plexiform neurofibroma",
        "targeted_rx": "Selumetinib FDA2020 MEK inhibitor plexiform neurofibroma (not MPNST); pembrolizumab NF1-mutant melanoma (NF1 triple-WT high TMB); doxorubicin+ifosfamide MPNST first-line sarcoma; trabectedin/pazopanib MPNST second-line; BRAF+MEK if rare BRAF V600 co-mutation; cobimetinib+atezolizumab NF1 melanoma emerging",
    },
]

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]

TREATMENT_PROTOCOLS_BY_GENE = {
    "CDKN2A": [
        "Pembrolizumab FDA2015 PD-1 200mg q3w advanced/unresectable melanoma",
        "Nivolumab FDA2015 PD-1 240mg q2w advanced melanoma",
        "Dabrafenib+trametinib BRAF+MEK if somatic BRAF V600E/K (check tumour, not germline)",
        "Ipilimumab+nivolumab FDA2015 combo first-line advanced melanoma",
        "CDK4/6 inhibitor palbociclib: emerging single-arm studies CDKN2A-deficient melanoma",
        "Wide local excision primary melanoma; SLNB if Breslow >0.8mm",
        "Adjuvant pembrolizumab/nivolumab stage III after resection",
        "Pancreatic cancer: gemcitabine+nab-paclitaxel or FOLFIRINOX first-line",
    ],
    "CDK4": [
        "Pembrolizumab PD-1 advanced melanoma (same as CDKN2A)",
        "BRAF+MEK inhibitor if somatic BRAF V600 (mandatory tumour testing)",
        "CDK4/6 inhibitors (palbociclib/ribociclib): mechanistic caution in CDK4-R24C — p16 non-functional",
        "Ipilimumab+nivolumab combo advanced melanoma",
        "Wide local excision + SLNB primary cutaneous melanoma",
        "Adjuvant pembrolizumab/nivolumab stage III resected melanoma",
        "Nivolumab 240mg q2w or pembrolizumab 200mg q3w first-line unresectable",
    ],
    "BAP1": [
        "Tebentafusp FDA2022 TCR bispecific — uveal melanoma (HLA-A*02:01 required); OS benefit",
        "Pembrolizumab advanced cutaneous BAP1-mutant melanoma",
        "Ipilimumab+nivolumab metastatic uveal melanoma (GNAQ/GNA11 somatic primary)",
        "Selumetinib MEK inhibitor emerging BAP1-mutant RCC/uveal",
        "Cisplatin+pemetrexed mesothelioma first-line if pleural component",
        "Sunitinib/pazopanib metastatic clear-cell RCC component",
        "Annual ophthalmology slit-lamp MANDATORY; focal laser/brachytherapy small uveal",
        "Uveal melanoma: enucleation vs proton beam vs plaque brachytherapy based on size",
    ],
    "MITF": [
        "Pembrolizumab PD-1 advanced melanoma",
        "BRAF+MEK inhibitors if somatic BRAF V600 on tumour (MITF germline ≠ BRAF somatic)",
        "Tebentafusp uveal melanoma component if HLA-A*02:01",
        "Sunitinib/pazopanib metastatic RCC component",
        "Wide local excision primary melanoma",
        "Adjuvant nivolumab/pembrolizumab stage III melanoma",
        "MTOR inhibitors (everolimus) MITF-TFE3/TFEB pathway RCC (experimental)",
    ],
    "POT1": [
        "Pembrolizumab PD-1 advanced melanoma",
        "BRAF+MEK inhibitors if somatic BRAF V600",
        "Radioiodine I-131 thyroid differentiated component",
        "Temozolomide + radiotherapy glioma component",
        "Ibrutinib/venetoclax CLL component if eligible",
        "Wide local excision + SLNB primary melanoma",
        "Adjuvant pembrolizumab/nivolumab stage III melanoma",
    ],
    "TERT": [
        "Pembrolizumab PD-1 advanced melanoma (TERT promoter somatic positive = high TMB)",
        "BRAF+MEK if somatic BRAF V600 (TERT C228T/C250T coexists with BRAF V600E in ~50%)",
        "Nivolumab+ipilimumab combo advanced melanoma",
        "TERT promoter C228T = poor prognostic marker — consider aggressive initial systemic therapy",
        "Wide local excision primary melanoma",
        "Adjuvant therapy stage III per BRAF status",
    ],
    "MC1R": [
        "Same systemic agents as sporadic melanoma (MC1R does not predict response)",
        "Pembrolizumab/nivolumab PD-1 advanced melanoma",
        "BRAF+MEK if somatic BRAF V600E/K (60-70% melanoma)",
        "Ipilimumab+nivolumab combo advanced melanoma",
        "Wide local excision + SLNB primary cutaneous melanoma",
        "Strict photobioprotection SPF50+ lifelong",
        "Dermoscopy surveillance 6-monthly if 2+ high-risk MC1R variants",
    ],
    "NF1": [
        "Selumetinib FDA2020 MEK inhibitor — plexiform neurofibromas (NOT MPNST)",
        "Doxorubicin+ifosfamide first-line MPNST (sarcoma regimen, NOT melanoma immunotherapy)",
        "Trabectedin/pazopanib second-line MPNST",
        "Pembrolizumab NF1-mutant melanoma (NF1 triple-WT high TMB — immunotherapy responsive)",
        "Cobimetinib+atezolizumab NF1 melanoma emerging MEK+checkpoint",
        "Temozolomide+bevacizumab glioma component",
        "Wide local excision + SLNB cutaneous melanoma",
    ],
}

SURVEILLANCE_BY_GENE = {
    "CDKN2A": [
        "Annual full-body skin exam + dermoscopy from age 18yr MANDATORY",
        "Baseline whole-body photography age 18yr",
        "Annual EUS + MRI-MRCP pancreas from age 40-50yr (10yr before youngest case)",
        "Annual ophthalmology from 25yr",
        "Cascade CDKN2A germline first-degree relatives",
        "Sun protection SPF50+ lifelong MANDATORY",
    ],
    "CDK4": [
        "Annual full-body skin exam + dermoscopy from age 18yr MANDATORY",
        "Baseline whole-body photography 18yr",
        "Annual BRAF/NRAS/NF1 somatic panel on any melanoma biopsy",
        "Cascade CDK4 germline first-degree relatives",
        "Ophthalmology annual",
    ],
    "BAP1": [
        "Annual ophthalmology UV slit-lamp + fundal photography from age 20yr MANDATORY",
        "Annual full-body skin exam from 20yr",
        "Annual chest CT from 40yr (mesothelioma)",
        "Annual renal ultrasound/MRI from 30yr (RCC)",
        "ASBESTOS avoidance MANDATORY lifelong",
        "Cascade BAP1 first-degree germline testing",
        "BAP1-IHC on any biopsy showing nuclear loss → germline test trigger",
    ],
    "MITF": [
        "Annual full-body skin exam + dermoscopy from age 25yr",
        "Annual ophthalmology slit-lamp from 25yr",
        "Renal ultrasound/MRI from age 40yr",
        "Cascade MITF first-degree germline testing",
    ],
    "POT1": [
        "Annual full-body skin exam + dermoscopy from age 25yr",
        "Annual thyroid ultrasound from age 25yr",
        "Brain MRI from age 40yr",
        "Annual lymphocyte/CBC count (CLL)",
        "Telomere length testing (long telomeres PATHOGNOMONIC POT1)",
        "Cascade POT1 germline testing first-degree relatives",
    ],
    "TERT": [
        "Annual full-body skin exam from age 25yr (germline promoter variants)",
        "Blood DNA TERT germline vs somatic distinction",
        "Annual thyroid ultrasound (TERT amplified thyroid risk)",
        "Cascade testing if familial melanoma with TERT promoter germline confirmed",
    ],
    "MC1R": [
        "Annual full-body skin exam from 25yr (2+ high-risk variants) or 18yr (+ CDKN2A)",
        "Baseline whole-body photography",
        "Strict SPF50+ sun protection lifelong MANDATORY",
        "Tanning beds — ABSOLUTE CI",
        "MC1R testing in CDKN2A families for risk stratification",
        "Dermoscopy 6-monthly if 2+ high-risk variants",
    ],
    "NF1": [
        "Annual full-body skin exam from age 18yr",
        "Annual ophthalmology (optic pathway glioma in children)",
        "Annual whole-body MRI if plexiform neurofibromas (MPNST surveillance)",
        "Annual blood pressure check (renal artery stenosis NF1)",
        "Selumetinib FDA2020 eligibility assessment if plexiform neurofibroma symptomatic",
        "Cascade NF1 germline first-degree relatives",
    ],
}


def _make_patients(gene: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    n = 40

    melanoma_subtypes = {
        "CDKN2A": [
            ("Superficial spreading melanoma", 0.45),
            ("Nodular melanoma", 0.20),
            ("Lentigo maligna melanoma", 0.15),
            ("Acral lentiginous melanoma", 0.05),
            ("Multiple primary melanoma", 0.15),
        ],
        "CDK4": [
            ("Superficial spreading melanoma", 0.50),
            ("Nodular melanoma", 0.20),
            ("Multiple primary melanoma", 0.15),
            ("Lentigo maligna melanoma", 0.10),
            ("Desmoplastic melanoma", 0.05),
        ],
        "BAP1": [
            ("Uveal melanoma", 0.50),
            ("Cutaneous melanoma", 0.25),
            ("BAPoma/MBAIT benign", 0.15),
            ("Mesothelioma", 0.10),
        ],
        "MITF": [
            ("Superficial spreading melanoma", 0.40),
            ("Uveal melanoma", 0.30),
            ("Renal cell carcinoma", 0.20),
            ("Nodular melanoma", 0.10),
        ],
        "POT1": [
            ("Cutaneous melanoma", 0.50),
            ("Thyroid cancer (follicular)", 0.25),
            ("Thyroid cancer (papillary)", 0.15),
            ("Glioma (low-grade)", 0.10),
        ],
        "TERT": [
            ("Cutaneous melanoma TERT-C228T", 0.55),
            ("Cutaneous melanoma TERT-C250T", 0.30),
            ("Thyroid cancer (TERT amplified)", 0.10),
            ("Urothelial carcinoma (TERT somatic)", 0.05),
        ],
        "MC1R": [
            ("Superficial spreading melanoma", 0.55),
            ("Nodular melanoma", 0.20),
            ("Lentigo maligna melanoma", 0.15),
            ("Squamous cell carcinoma skin", 0.10),
        ],
        "NF1": [
            ("Cutaneous melanoma (NF1-triple-WT)", 0.45),
            ("Plexiform neurofibroma", 0.25),
            ("MPNST (malignant peripheral nerve sheath tumour)", 0.15),
            ("Optic pathway glioma", 0.10),
            ("Dermal neurofibroma", 0.05),
        ],
    }

    variants_by_gene = {
        "CDKN2A": [
            "p.Gly23Asp (p16 ankyrin-repeat)",
            "p.Arg24Pro (p16 CDK4/6-binding surface)",
            "p.Ala57Val (p16 hydrophobic core)",
            "p.Val59Gly (p16 ankyrin-repeat 2)",
            "c.IVS1-105A>G (intronic splice CDKN2A)",
            "p.Pro114Leu (p16 ankyrin-repeat 3)",
            "p.Glu27Ter (p16 truncation)",
            "p.Arg80Ter (p16 truncation)",
        ],
        "CDK4": [
            "p.Arg24Cys (R24C — p16-binding pocket ablated)",
            "p.Arg24His (R24H — p16-binding pocket)",
            "p.Gly101Trp (G101W — activation segment)",
            "p.Arg24Ser (rare R24 variant)",
        ],
        "BAP1": [
            "p.Gln684Ter (truncation UCH domain near)",
            "p.Trp10Ter (truncation early)",
            "p.Arg60Pro (UCH catalytic domain)",
            "p.Leu394Pro (BRCA1-binding region)",
            "p.Asn591Asp (BRCT-like domain)",
            "c.IVS10+1G>A (splice donor exon 10)",
            "p.Arg215Ter (truncation)",
        ],
        "MITF": [
            "p.Glu318Lys (E318K SUMO acceptor K316-adjacent — European founder)",
            "p.Ile212Met (bHLH dimerisation)",
            "p.Arg217His (bHLH DNA contact)",
            "p.Arg259Gly (leucine zipper hinge)",
        ],
        "POT1": [
            "p.Tyr36Cys (OB1 fold ssDNA binding)",
            "p.Phe62Ile (OB1 hydrophobic core)",
            "p.Gly274Ser (OB2 fold interface)",
            "p.Arg117Cys (OB1-OB2 linker)",
            "c.IVS7+1G>T (splice exon 7)",
        ],
        "TERT": [
            "c.-124C>T (TERT promoter C228T — de-novo ETS site)",
            "c.-146C>T (TERT promoter C250T — de-novo ETS site)",
            "c.-57A>C (TERT promoter rare familial variant)",
            "p.Ala279Thr (germline RT domain moderate)",
            "p.Arg631Gly (germline TRBD domain)",
        ],
        "MC1R": [
            "p.Arg151Cys (R151C high-risk RHC variant)",
            "p.Arg160Trp (R160W high-risk RHC variant)",
            "p.Asp294His (D294His high-risk RHC variant)",
            "p.Val60Leu (V60L low-risk RHC)",
            "p.Val92Met (V92M low-risk RHC)",
            "p.Arg163Gln (R163Q moderate RHC)",
        ],
        "NF1": [
            "p.Arg1534Ter (GRD truncation — NF1)",
            "p.Glu1200Lys (GRD RAS-GAP contact)",
            "c.IVS14+1G>A (splice intron 14 NF1)",
            "del exons 4-7 (large deletion NF1)",
            "p.Tyr489Ter (early truncation)",
            "p.Leu847Pro (ARM repeat hydrophobic)",
        ],
    }

    age_ranges = {
        "CDKN2A": (22, 68),
        "CDK4":   (25, 70),
        "BAP1":   (28, 72),
        "MITF":   (30, 72),
        "POT1":   (28, 70),
        "TERT":   (30, 72),
        "MC1R":   (35, 78),
        "NF1":    (20, 68),
    }

    subtypes = melanoma_subtypes[gene]
    variants = variants_by_gene[gene]
    age_lo, age_hi = age_ranges[gene]

    patients = []
    for i in range(n):
        age = rng.randint(age_lo, age_hi)
        r = rng.random()
        cumulative = 0.0
        tumour = subtypes[-1][0]
        for t, p in subtypes:
            cumulative += p
            if r < cumulative:
                tumour = t
                break
        variant = rng.choice(variants)
        sex = rng.choice(["M", "F"])
        stage = rng.choices(["I", "II", "III", "IV"], weights=[0.30, 0.30, 0.25, 0.15])[0]
        braf_somatic = gene not in ("BAP1", "NF1") and rng.random() < 0.62
        immunotherapy = rng.random() < (0.70 if gene != "NF1" else 0.55)
        mpnst_risk = (gene == "NF1") and (rng.random() < 0.11)
        uveal_component = (gene in ("BAP1", "MITF")) and (rng.random() < (0.50 if gene == "BAP1" else 0.30))
        multiple_primaries = (gene in ("CDKN2A", "CDK4")) and (rng.random() < 0.18)
        pancreatic_risk = (gene == "CDKN2A") and (rng.random() < 0.17)
        relapse = (stage in ("III", "IV")) and (rng.random() < 0.45)

        patients.append({
            "patient_id": f"{gene[:3].upper()}-{seed:04d}-{i+1:02d}",
            "gene": gene,
            "age_at_dx": age,
            "sex": sex,
            "tumour_type": tumour,
            "stage": stage,
            "variant": variant,
            "braf_somatic": braf_somatic,
            "immunotherapy": immunotherapy,
            "mpnst_risk": mpnst_risk,
            "uveal_component": uveal_component,
            "multiple_primaries": multiple_primaries,
            "pancreatic_risk": pancreatic_risk,
            "relapse": relapse,
        })
    return patients


def generate_overview() -> dict:
    from collections import Counter
    cohorts = {}
    for i, g in enumerate(ATLAS_GENES):
        cohorts[g["gene"]] = _make_patients(g["gene"], SEED_BASE + i)
    total = sum(len(pts) for pts in cohorts.values())
    gene_counts = {gene: len(pts) for gene, pts in cohorts.items()}
    braf_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["braf_somatic"]) / total, 1
    )
    immunotherapy_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["immunotherapy"]) / total, 1
    )
    mpnst_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["mpnst_risk"]) / total, 1
    )
    uveal_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["uveal_component"]) / total, 1
    )
    multiple_primaries_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["multiple_primaries"]) / total, 1
    )
    pancreatic_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["pancreatic_risk"]) / total, 1
    )
    mean_age = round(
        sum(p["age_at_dx"] for pts in cohorts.values() for p in pts) / total, 1
    )

    return {
        "atlas": "Hereditary-Melanoma-Predisposition-Atlas",
        "genes": _GENE_LIST,
        "total_patients": total,
        "gene_counts": gene_counts,
        "seed_range": f"{SEED_BASE}-{SEED_BASE+7}",
        "braf_somatic_rate_pct": braf_rate,
        "immunotherapy_eligible_rate_pct": immunotherapy_rate,
        "mpnst_risk_rate_pct": mpnst_rate,
        "uveal_component_rate_pct": uveal_rate,
        "multiple_primaries_rate_pct": multiple_primaries_rate,
        "pancreatic_risk_rate_pct": pancreatic_rate,
        "mean_age_at_dx": mean_age,
        "key_facts": [
            "CDKN2A/FAMMM: melanoma 40-50x RR HIGHEST hereditary risk; pancreatic cancer 20x; annual skin exam from 18yr + EUS/MRI-MRCP pancreas from 40-50yr MANDATORY",
            "CDK4 R24C/R24H: p16-INK4a binding pocket abrogated; melanoma 40-50x RR; CDK4/6 inhibitor PARADOX — p16 already non-functional; BRAF somatic test mandatory",
            "BAP1-TPDS: uveal melanoma 50% PATHOGNOMONIC; asbestos MANDATORY avoided (100x MPM synergy); tebentafusp FDA2022 uveal melanoma (TCR bispecific HLA-A*02:01)",
            "MITF E318K European founder: melanoma 5x RR; uveal melanoma elevated; RCC 3-5x; SUMO acceptor K316 disruption → enhanced transcriptional activity",
            "POT1 familial melanoma: TELOMERE ELONGATION (NOT shortening) PATHOGNOMONIC — distinguish from DC/IPF; melanoma 4-6x; thyroid 3-5x; glioma elevated",
            "TERT C228T/C250T: most common MELANOMA somatic mutation globally (60-70%); germline TERT promoter = moderate melanoma risk; germline TERT coding LOF = DC/IPF (DIFFERENT phenotype)",
            "MC1R R151C/R160W/D294H: 2-3x RR per variant; CRITICAL CDKN2A amplifier (compound = up to 100x); strict sun protection MANDATORY; tanning beds ABSOLUTE CI",
            "NF1 melanoma: MPNST 8-13% PATHOGNOMONIC = SARCOMA (doxorubicin/ifosfamide NOT immunotherapy); selumetinib FDA2020 plexiform NF; NF1 triple-WT melanoma = immunotherapy-responsive",
        ],
    }


def generate_breakdown() -> dict:
    from collections import Counter
    breakdown = {}
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
                "melanoma_risk": g["melanoma_risk"],
                "pathognomonic": g["pathognomonic"],
                "key_avoid": g["key_avoid"],
                "key_rule": g["key_rule"],
                "surveillance": g["surveillance"],
                "targeted_rx": g["targeted_rx"],
            },
            "n": len(pts),
            "braf_somatic_pct": round(100 * sum(1 for p in pts if p["braf_somatic"]) / len(pts), 1),
            "immunotherapy_pct": round(100 * sum(1 for p in pts if p["immunotherapy"]) / len(pts), 1),
            "mpnst_risk_pct": round(100 * sum(1 for p in pts if p["mpnst_risk"]) / len(pts), 1),
            "uveal_component_pct": round(100 * sum(1 for p in pts if p["uveal_component"]) / len(pts), 1),
            "multiple_primaries_pct": round(100 * sum(1 for p in pts if p["multiple_primaries"]) / len(pts), 1),
            "pancreatic_risk_pct": round(100 * sum(1 for p in pts if p["pancreatic_risk"]) / len(pts), 1),
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
        "atlas": "Hereditary-Melanoma-Predisposition-Atlas",
        "definitions": {
            "cdkn2a_fammm_highest_risk": (
                "CDKN2A/FAMMM: 156aa / 15kDa; 9p21.3; DUAL TRANSCRIPT: p16-INK4a (CDK4/6 inhibitor) + p14-ARF (MDM2 sequestration); "
                "MELANOMA 40-50x RR HIGHEST: ~67% penetrance by age 80yr; multiple primary melanomas PATHOGNOMONIC; "
                "PANCREATIC CANCER 20x RR: annual EUS + MRI-MRCP from age 40-50yr MANDATORY; "
                "50+ ATYPICAL NAEVI: FAMMM phenotype PATHOGNOMONIC; annual dermatology from 18yr MANDATORY."
            ),
            "cdk4_gof_p16_binding_paradox": (
                "CDK4 familial melanoma: 303aa / 34kDa; 12q14.1; GOF R24C/R24H/G101W — p16-INK4a binding pocket abrogated; "
                "CDK4 cannot be inhibited by p16 → G1/S checkpoint bypassed → cyclin-D1/CDK4 hyperactive → RB1 phosphorylated; "
                "CDK4/6 INHIBITOR PARADOX: palbociclib/ribociclib target CDK4 via p16 binding mechanism — "
                "in CDK4-R24C tumours where p16 cannot bind CDK4, these agents may have altered efficacy; "
                "BRAF somatic testing MANDATORY (60-70% melanoma BRAF V600 — independent of CDK4 germline)."
            ),
            "bap1_uveal_asbestos_tebentafusp": (
                "BAP1-TPDS: 729aa / 80kDa; 3p21.1; deubiquitinase H2AK119ub1; "
                "UVEAL MELANOMA 50% PATHOGNOMONIC — tebentafusp FDA2022 TCR bispecific gp100/HLA-A*02:01 — first FDA-approved uveal melanoma treatment; "
                "ASBESTOS SYNERGY: BAP1 germline + asbestos = ~100x MPM risk — ASBESTOS AVOIDANCE MANDATORY ALL CARRIERS; "
                "BAPOMAS/MBAITs: benign intradermal melanocytic tumours on skin = PATHOGNOMONIC cascade trigger for germline testing; "
                "BAP1-IHC nuclear loss on ANY specimen (melanoma/MPM/RCC) = trigger germline BAP1 testing."
            ),
            "mitf_e318k_sumo_founder": (
                "MITF E318K: 520aa / 59kDa; 3p14.1; bHLH-LZ master melanocyte TF; "
                "E318K = SUMO acceptor K316-adjacent disruption → impaired SUMOylation → enhanced MITF transcriptional activity; "
                "EUROPEAN FOUNDER: ~2% European general population; melanoma 5x RR; uveal melanoma elevated; RCC 3-5x; "
                "MODERATE PENETRANCE: ~15% lifetime melanoma risk — risk stratify with family history and additional MC1R variants; "
                "Annual skin exam from 25yr; annual ophthalmology from 25yr; renal imaging from 40yr."
            ),
            "pot1_telomere_elongation_paradox": (
                "POT1 familial melanoma: 634aa / 71kDa; 7q31.33; OB-fold telomere capping protein; "
                "TELOMERE ELONGATION PARADOX: POT1 LOF → telomere ends uncapped → ATR activation → G-overhang extension → LONG telomeres; "
                "CRITICAL DISTINCTION from short telomere syndromes (DC/IPF/HH): DKC1/TERC/TERT coding LOF = SHORT telomeres; "
                "POT1 LOF = LONG telomeres — completely opposite phenotype from DC; "
                "Multi-cancer risk: melanoma 4-6x + thyroid 3-5x + glioma 2-3x + CLL elevated; "
                "Annual thyroid ultrasound from 25yr MANDATORY."
            ),
            "tert_promoter_somatic_vs_germline": (
                "TERT: 1132aa / 127kDa; 5p15.33; telomerase catalytic reverse transcriptase; "
                "SOMATIC C228T/C250T (most common MELANOMA somatic mutation globally — 60-70% melanoma): "
                "   creates de-novo ETS binding site → TERT promoter activation → telomerase immortalisation; "
                "   POOR PROGNOSIS BIOMARKER in melanoma; coexists with BRAF V600E in ~50% melanoma; "
                "GERMLINE PROMOTER VARIANTS: moderate melanoma 3-5x RR — different from somatic; "
                "GERMLINE CODING LOF: SHORT TELOMERE SYNDROME — DC/IPF/liver cirrhosis (NOT melanoma pathway); "
                "CRITICAL: blood vs tumour DNA testing MANDATORY to distinguish germline from somatic."
            ),
            "mc1r_modifier_cdkn2a_amplifier": (
                "MC1R: 317aa / 35kDa; 16q24.3; 7-TM GPCR melanocortin receptor; "
                "HIGH-RISK VARIANTS R151C/R160W/D294His: LOF → phaeomelanin (UV-sensitising) instead of eumelanin (UV-protective); "
                "STANDALONE RISK: 2-3x RR per single high-risk variant (NOT high-penetrance alone); "
                "CDKN2A AMPLIFIER: CDKN2A carrier + MC1R high-risk variant = up to 100x compounded RR CRITICAL; "
                "ALL MC1R CARRIERS: strict SPF50+ MANDATORY lifelong; tanning beds ABSOLUTE CI; "
                "Clinical utility: MC1R testing in CDKN2A families for risk stratification — NOT for population-wide standalone screening."
            ),
            "nf1_mpnst_vs_melanoma_critical": (
                "NF1: 2839aa / 319kDa; 17q11.2; RAS-GAP; NF1 LOF → RAS-GTP accumulation → MAPK hyperactivation; "
                "NF1-ASSOCIATED MELANOMA: 2-3x RR; NF1 triple-WT subtype (NF1-somatic + BRAF-WT + NRAS-WT) = high TMB immunotherapy-responsive; "
                "MPNST CRITICAL DISTINCTION: 8-13% lifetime PATHOGNOMONIC; "
                "MPNST = MALIGNANT PERIPHERAL NERVE SHEATH TUMOUR = SARCOMA: "
                "   Doxorubicin+ifosfamide first-line (sarcoma regimen); "
                "   Immunotherapy NOT standard for MPNST; "
                "   Do NOT treat NF1 MPNST as melanoma; "
                "SELUMETINIB FDA2020: MEK inhibitor for PLEXIFORM NEUROFIBROMAS (not MPNST); "
                "CAFÉ-AU-LAIT 6+macules PATHOGNOMONIC (>5mm prepubertal / >15mm postpubertal)."
            ),
        },
        "key_clinical_distinctions": [
            "CDKN2A: melanoma 40-50x + pancreatic 20x — annual dermatology 18yr + EUS/MRI-MRCP pancreas 40-50yr MANDATORY",
            "CDK4 R24C/R24H: p16 binding abrogated → CDK4/6i palbociclib paradox — BRAF somatic test mandatory",
            "BAP1: asbestos MANDATORY avoided (100x MPM). Annual ophthalmology (uveal 50%). Tebentafusp FDA2022 uveal melanoma",
            "MITF E318K: SUMO acceptor disruption → enhanced transcriptional activity; moderate penetrance; annual skin+ophthalmology from 25yr",
            "POT1: TELOMERE ELONGATION (not shortening) PATHOGNOMONIC — opposite of DC/IPF; melanoma+thyroid+glioma multi-cancer",
            "TERT C228T/C250T: most common SOMATIC melanoma mutation globally; germline promoter = moderate risk; germline coding LOF = DC/IPF NOT melanoma",
            "MC1R: standalone 2-3x; + CDKN2A compound = up to 100x; tanning beds ABSOLUTE CI; CDKN2A+MC1R always surveillance-intensive",
            "NF1 MPNST: SARCOMA = doxorubicin/ifosfamide NOT immunotherapy. Selumetinib plexiform NF (not MPNST). NF1 triple-WT melanoma = immunotherapy-responsive",
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2))
