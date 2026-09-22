#!/usr/bin/env python3
"""Hereditary-Mesothelioma-Predisposition-Atlas -- Complete 8-Gene Reference
BAP1    (BRCA1-Associated Protein 1; 729aa; 3p21.1; AD LOF;
         BAP1 Tumour Predisposition Syndrome (BAP1-TPDS);
         Mesothelioma lifetime risk 30-60% PRIMARY; uveal melanoma 30-50%;
         MBAITs (melanocytic BAP1-mutated atypical intraepidermal tumours) PATHOGNOMONIC;
         BAP1-null IHC PATHOGNOMONIC for BAP1-deficient tumours;
         AVOID ASBESTOS ABSOLUTELY -- synergistic germline BAP1 + asbestos = extreme risk;
         seed SEED_BASE+0) .
BRCA2   (Breast Cancer Gene 2; 3418aa; 13q12.3; AD LOF;
         HBOC -- Hereditary Breast-Ovarian Cancer syndrome;
         Mesothelioma 2-5x elevated risk (pleural and peritoneal);
         Cisplatin/carboplatin sensitivity (HR deficiency); Olaparib PARP inhibitor;
         Fanconi Anemia type D1 (biallelic) -- medulloblastoma/Wilms/AML MOST SEVERE FA;
         seed SEED_BASE+1) .
NF2     (Neurofibromatosis Type 2 / Schwannomatosis; 595aa; 22q12.2; AD LOF;
         Bilateral vestibular schwannoma PATHOGNOMONIC 100% penetrance by age 30yr;
         Meningioma 50-75%; spinal ependymoma; somatic NF2 deletion 40-80% sporadic mesothelioma;
         Merlin cytoskeletal scaffold tumour suppressor;
         Bevacizumab (anti-VEGF) for progressive VS;
         seed SEED_BASE+2) .
CDKN2A  (Cyclin-Dependent Kinase Inhibitor 2A; 156aa; 9p21.3; AD LOF;
         FAMM -- Familial Atypical Multiple Mole Melanoma;
         Cutaneous melanoma 25-36% PRIMARY; pancreatic cancer 20x elevated;
         9p21 homozygous deletion 50-80% sporadic mesothelioma;
         CDK4/6 inhibitors (palbociclib/ribociclib/abemaciclib) for CDKN2A-deleted tumours;
         seed SEED_BASE+3) .
TP53    (Tumour Protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome (LFS);
         AVOID RADIATION ABSOLUTELY -- radiation-induced secondary malignancies RULE 1;
         WBMRI annually Toronto Protocol; R337H Brazilian founder;
         Sarcoma 50-60% PRIMARY; serous endometrial/ovarian p53-aberrant histology;
         seed SEED_BASE+4) .
SMARCB1 (SWI/SNF-Related Matrix-Associated Actin-Dependent Regulator; 385aa; 22q11.23; AD LOF;
         AT/RT (Atypical Teratoid/Rhabdoid Tumour) in infants PATHOGNOMONIC;
         Malignant rhabdoid tumour (MRT) of kidney in infants PATHOGNOMONIC;
         SMARCB1-null IHC PATHOGNOMONIC; EZH2 inhibitor tazemetostat FDA-approved;
         Schwannomatosis type 2; epithelioid mesothelioma SMARCB1/INI1 loss 5-10%;
         seed SEED_BASE+5) .
MLH1    (MutL Homolog 1; 756aa; 3p22.2; AD LOF;
         Lynch Syndrome type 1 (HNPCC);
         Colorectal cancer 40-50%; endometrial cancer 40-50%;
         Pembrolizumab FDA-approved MSI-H tumours ALL HISTOLOGIES;
         Aspirin 600mg/day CAPP2 trial 50% CRC risk reduction LEVEL A evidence;
         seed SEED_BASE+6) .
ATM     (Ataxia-Telangiectasia Mutated; 3056aa; 11q22.3; AD LOF;
         Ataxia-Telangiectasia (biallelic) / HBOC-2 (monoallelic);
         RADIOSENSITIVITY ABSOLUTE in biallelic A-T -- avoid radiation LETHAL;
         Monoallelic ATM: mesothelioma 2-4x; breast 15-25%; pancreatic 5-10%;
         Olaparib + ceralasertib (ATRi) clinical trials;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3206-3213)
"""
import random

SEED_BASE = 3206

ATLAS_GENES = [
    {
        "gene": "BAP1",
        "protein": (
            "BAP1 -- 3p21.1 Autosomal-Dominant-LOF -- 729aa -- "
            "BAP1-80kDa-H2A-Deubiquitinase-TPDS-Mesothelioma-30-60pct-PRIMARY-"
            "Uveal-Melanoma-30-50pct-MBAITs-PATHOGNOMONIC-Avoid-Asbestos-OMIM-603089"
        ),
        "locus": "3p21.1",
        "protein_size": (
            "729 aa / 80 kDa / 3p21.1 BAP1 encodes BRCA1-Associated Protein 1 (Ubiquitin Carboxyl-Terminal Hydrolase): "
            "STRUCTURE: "
            "  729 aa / 80 kDa; nuclear deubiquitinase (DUB); "
            "  Catalytic domain: UCH (ubiquitin C-terminal hydrolase) domain (aa 1-240); "
            "  NLS (nuclear localisation signal) -- nuclear localisation REQUIRED for tumour suppressor function; "
            "  Cytoplasmic BAP1 = non-functional; mutations disrupting NLS -> cytoplasmic mislocalisation -> LOF; "
            "  BAP1 deubiquitinates H2A-K119Ub (catalytic activity on PRC1 mark); "
            "  BAP1 complexes: PR-DUB (Polycomb Repressive DUB) -- BAP1 + ASXL1/2/3; "
            "  PR-DUB removes H2A-K119Ub marks -> activates polycomb-repressed genes; "
            "  BAP1 also deubiquitinates BRCA1 (RING domain substrate); "
            "BAP1 TUMOUR PREDISPOSITION SYNDROME (BAP1-TPDS): "
            "  OMIM 614327 -- autosomal dominant; de novo mutations rare; germline LOF; "
            "  Penetrance extremely high for at least one tumour type by age 60yr; "
            "  Mesothelioma PRIMARY cancer risk: 30-60% lifetime (malignant pleural mesothelioma dominant); "
            "  Uveal melanoma: 30-50% lifetime -- PATHOGNOMONIC BAP1-TPDS combination (mesothelioma + uveal melanoma); "
            "  Cutaneous melanocytic tumours: MBAITs (melanocytic BAP1-mutated atypical intraepidermal tumours) PATHOGNOMONIC; "
            "  MBAITs: atypical dome-shaped skin lesions with large epithelioid melanocytes -- BAP1-null on IHC; "
            "  ccRCC (clear cell renal cell carcinoma): 15-25% lifetime; "
            "  Cholangiocarcinoma (intrahepatic): elevated risk; "
            "  Meningioma: modest elevated risk; "
            "BAP1 NULL IHC -- PATHOGNOMONIC: "
            "  BAP1 protein loss (null) on IHC = PATHOGNOMONIC for BAP1-deficient tumours; "
            "  Use on: mesothelioma (loss confirms BAP1 inactivation), uveal melanoma (loss = metastatic risk); "
            "  BAP1-null mesothelioma: better prognosis than BAP1-intact (paradox -- higher immunogenic; pembrolizumab active); "
            "  BAP1 IHC: loss in both germline BAP1-TPDS and somatic BAP1 biallelic inactivation; "
            "  Sporadic BAP1 somatic mutations: 50-60% malignant pleural mesothelioma -- most common somatic alteration in mesothelioma; "
            "AVOID ASBESTOS -- ABSOLUTE RULE: "
            "  Asbestos exposure + germline BAP1 LOF = SYNERGISTIC extreme mesothelioma risk; "
            "  Germline BAP1 carriers: avoid ALL asbestos exposure ABSOLUTELY -- occupational/environmental; "
            "  Pre-existing asbestos exposure + new BAP1 diagnosis: heightened mesothelioma surveillance; "
            "SURVEILLANCE (BAP1-TPDS): "
            "  Annual CT chest/abdomen from age 30yr (mesothelioma); "
            "  Annual ophthalmology (uveal melanoma surveillance); "
            "  Annual full-body skin exam (MBAITs -> melanoma); "
            "  Renal imaging (MRI/CT abdomen) every 1-2yr (ccRCC); "
            "  Germline testing of first-degree relatives mandatory"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 614327 (BAP1-TPDS); high penetrance for at least one malignancy by age 60yr; family cascade mandatory; de novo rare",
        "cancer_risk": "Mesothelioma (malignant pleural PRIMARY): 30-60% lifetime HIGHEST; uveal melanoma 30-50%; ccRCC 15-25%; MBAITs -> cutaneous melanoma; cholangiocarcinoma elevated",
        "pathognomonic": "BAP1-null IHC PATHOGNOMONIC for BAP1-deficient mesothelioma/uveal melanoma; MBAITs (dome-shaped BAP1-null skin lesions) PATHOGNOMONIC BAP1-TPDS; mesothelioma + uveal melanoma = BAP1-TPDS until proven otherwise",
        "surveillance_key": "Annual CT chest/abdomen (mesothelioma); annual ophthalmology (uveal melanoma); annual full-body skin (MBAITs); renal MRI 1-2yr; AVOID ASBESTOS ABSOLUTELY; germline family cascade",
        "key_distinctions": [
            "BAP1-NULL-IHC-PATHOGNOMONIC-MESOTHELIOMA",
            "MBAITS-DOME-SHAPED-PATHOGNOMONIC-SKIN",
            "MESOTHELIOMA-UVEAL-MELANOMA-TPDS-COMBINATION",
            "AVOID-ASBESTOS-ABSOLUTELY-SYNERGISTIC",
            "BAP1-MOST-COMMON-SOMATIC-MESOTHELIOMA-50-60PCT",
            "CCRCC-15-25PCT-LIFETIME-BAP1-TPDS",
        ],
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-HBOC-Mesothelioma-2-5x-"
            "Olaparib-PARP-Cisplatin-Sensitive-FA-D1-Biallelic-MOST-SEVERE-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 encodes Breast Cancer Type 2 Susceptibility Protein: "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; largest hereditary breast cancer gene product; "
            "  N-terminal transactivation domain; 8 BRC repeats (aa 1002-2085) -- each binds one RAD51 monomer; "
            "  DBD (DNA-binding domain): OB folds + Tower domain (aa 2402-3190); "
            "  C-terminal RAD51-binding motif (aa 3265-3330); "
            "  NLS (aa 3263-3269); nuclear scaffolding of RAD51 at DSBs (double-strand breaks); "
            "  BRCA2 function: HR scaffold -- loads RAD51 onto ssDNA at resected DSBs -> Rad51 nucleofilament; "
            "  BRCA2 LOF -> HR deficiency (HRD) -> NHEJ-dependent DSB repair -> chromosomal instability; "
            "  BRCA2 interacts with PALB2 (BRCA2-PALB2-BRCA1 module) at DSBs; "
            "MESOTHELIOMA RISK (BRCA2): "
            "  Monoallelic BRCA2 LOF: 2-5x elevated mesothelioma risk (pleural and peritoneal); "
            "  Less well-characterised than BAP1 mesothelioma risk; evidence growing; "
            "  Peritoneal mesothelioma component (distinct from pleural); "
            "  Mechanism: HRD -> genomic instability susceptibility; synergistic with asbestos; "
            "CISPLATIN/CARBOPLATIN SENSITIVITY: "
            "  BRCA2 LOF -> HR-deficient -> PLATINATING AGENTS (cisplatin/carboplatin) HIGHLY ACTIVE; "
            "  Cisplatin/pemetrexed: 1L mesothelioma; BRCA2-mutant may have enhanced cisplatin benefit; "
            "  Carboplatin + pemetrexed alternatively; "
            "OLAPARIB (PARP INHIBITOR): "
            "  BRCA2 LOF -> HRD -> PARP inhibitor synthetic lethality (trapping PARP at unrepaired SSBs); "
            "  Olaparib: FDA-approved BRCA1/2-germline ovarian, breast, pancreatic, prostate cancers; "
            "  BRCA2-related mesothelioma: off-label PARP inhibitor use in HRD-positive mesothelioma; "
            "  BRCA2-related mesothelioma may benefit from olaparib-based regimens in certain clinical contexts; "
            "FANCONI ANEMIA TYPE D1 -- BIALLELIC -- MOST SEVERE FA: "
            "  BRCA2 = FANCD1 gene; biallelic BRCA2 LOF = Fanconi Anemia type D1; "
            "  FA-D1 is the MOST SEVERE FA phenotype: early childhood cancers; "
            "  FA-D1 malignancies: medulloblastoma (brain), Wilms tumour (kidney), AML (leukaemia); "
            "  vs other FA types: FA-D1 presents in early childhood, often fatal; "
            "  FA-D1 children: avoid mitomycin C/cross-linking agents (diagnostic FA sensitivity test); "
            "HBOC SURVEILLANCE (BRCA2): "
            "  Annual MRI breast + mammography from age 25yr; "
            "  Salpingo-oophorectomy (BSO) at age 40-45yr (ovarian cancer prevention); "
            "  Prostate screening PSA from age 40yr (BRCA2 -> 5-8x prostate risk); "
            "  Pancreatic surveillance (MRI/MRCP) from age 50yr; "
            "  Mesothelioma surveillance: annual CT chest in known asbestos-exposed + BRCA2 germline"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 600185/612555; high penetrance breast/ovarian; biallelic = FA-D1 (MOST SEVERE FA); family cascade mandatory",
        "cancer_risk": "Breast (female 70%, male 8%), ovarian 15-25%, pancreatic 5-10%, prostate 5-8x; mesothelioma 2-5x elevated; FA-D1 biallelic: medulloblastoma/Wilms/AML childhood",
        "pathognomonic": "BRCA2 LOF -> HRD -> cisplatin/olaparib sensitivity; FA-D1 biallelic = MOST SEVERE FA (medulloblastoma/Wilms/AML PATHOGNOMONIC combination); BRCA2-mutant mesothelioma enhanced platinum sensitivity",
        "surveillance_key": "Annual MRI breast + mammography from age 25yr; BSO age 40-45yr; pancreatic MRI from 50yr; prostate PSA from 40yr; olaparib FDA BRCA2 ovarian/breast/pancreatic; FA-D1 biallelic: avoid cross-linking agents",
        "key_distinctions": [
            "MESOTHELIOMA-2-5X-ELEVATED-BRCA2",
            "CISPLATIN-CARBOPLATIN-SENSITIVITY-HRD",
            "OLAPARIB-PARP-SYNTHETIC-LETHALITY",
            "FA-D1-BIALLELIC-MOST-SEVERE-FA",
            "FA-D1-MEDULLOBLASTOMA-WILMS-AML-TRIAD",
            "BRCA2-PALB2-BRCA1-HR-MODULE",
        ],
    },
    {
        "gene": "NF2",
        "protein": (
            "NF2 -- 22q12.2 Autosomal-Dominant-LOF -- 595aa -- "
            "Merlin-70kDa-ERM-Cytoskeletal-Scaffold-NF2-Bilateral-VS-PATHOGNOMONIC-"
            "Meningioma-50-75pct-Somatic-NF2-40-80pct-Sporadic-Mesothelioma-OMIM-101000"
        ),
        "locus": "22q12.2",
        "protein_size": (
            "595 aa / 70 kDa / 22q12.2 NF2 encodes Merlin (Moesin-Ezrin-Radixin-Like Protein): "
            "STRUCTURE: "
            "  595 aa / 70 kDa; ERM-family cytoskeletal scaffold protein; "
            "  FERM domain (aa 1-302): band 4.1/ERM family N-terminal; "
            "  Coiled-coil domain (aa 304-480); C-terminal domain (aa 480-595); "
            "  Merlin self-folds: open (active TS) vs closed (inactive) via head-to-tail interaction; "
            "  Tumour suppressor: links membrane receptors to actin cytoskeleton; "
            "  Merlin regulates: Hippo pathway (YAP/TAZ), PI3K/mTOR, RAS/MAPK, Wnt; "
            "  Merlin LOF -> YAP/TAZ nuclear translocation -> proliferative gene expression; "
            "  Key NF2 interactions: CD44, EGFR, erbB2, beta-integrin; "
            "NF2 SYNDROME SPECTRUM: "
            "  NF2 = Neurofibromatosis type 2; OMIM 101000; AD LOF; 22q12.2; "
            "  Bilateral vestibular schwannoma (VS): PATHOGNOMONIC -- 100% penetrance by age 30yr; "
            "  Hearing loss + tinnitus: FIRST symptoms in ~90% (bilateral VS compressing CN VIII); "
            "  Meningioma: 50-75% lifetime (intracranial + spinal meningiomas); "
            "  Spinal ependymoma: 33-53%; "
            "  Peripheral schwannomas: multiple, often dorsal root; "
            "SOMATIC NF2 IN SPORADIC MESOTHELIOMA: "
            "  Somatic NF2 deletion/mutation: 40-80% sporadic malignant mesothelioma = MOST COMMON somatic alteration; "
            "  NF2 somatic inactivation -> YAP activation -> mesothelioma proliferation; "
            "  Hereditary NF2: mesothelioma risk elevated but LESS than BAP1 (lower penetrance); "
            "  Tumour DNA profiling in mesothelioma: NF2 + BAP1 + CDKN2A three-gene panel; "
            "YAP INHIBITION -- THERAPEUTIC IMPLICATION: "
            "  NF2 LOF -> YAP/TAZ activation -> mesothelioma driver; "
            "  Verteporfin: YAP inhibitor (off-label); preclinical NF2-deficient mesothelioma data; "
            "  FAK inhibitors (defactinib): phase II NF2-deficient mesothelioma (COMMAND trial); "
            "BEVACIZUMAB (NF2-VS): "
            "  Bevacizumab (anti-VEGF): FDA-approved for NF2 progressive vestibular schwannoma; "
            "  Reduces VS volume + stabilises hearing in bevacizumab-responding NF2; "
            "  VEGF overexpressed in NF2-VS (VEGF drives VS endolymphatic fluid -> hearing loss); "
            "SCHWANNOMATOSIS vs NF2: "
            "  Schwannomatosis: LZTR1 (22q11.21) or SMARCB1 mutations; multiple schwannomas WITHOUT bilateral VS; "
            "  NF2: bilateral VS PATHOGNOMONIC distinguishes from schwannomatosis; "
            "  SMARCB1 schwannomatosis: 22q11.23 (vs NF2: 22q12.2 -- same chromosome, different loci); "
            "MRI SURVEILLANCE (NF2): "
            "  Annual MRI brain + spine from diagnosis; "
            "  Audiology (annual) + ophthalmology; "
            "  Surgery/radiosurgery for VS growth threatening hearing; "
            "  Bevacizumab: non-surgical VS progression"
        ),
        "inheritance": "Autosomal Dominant (AD); LOF; OMIM 101000; ~50% de novo; high penetrance bilateral VS; mosaicism common (~30%); family cascade mandatory",
        "cancer_risk": "Bilateral VS: 100% by age 30yr PATHOGNOMONIC; meningioma 50-75%; spinal ependymoma 33-53%; peripheral schwannoma multiple; mesothelioma elevated (somatic NF2 40-80% sporadic mesothelioma)",
        "pathognomonic": "Bilateral vestibular schwannoma (bilateral VS) = NF2 PATHOGNOMONIC (100% penetrance age 30yr); hearing loss + tinnitus in young patient = NF2 until proven otherwise; somatic NF2 most common sporadic mesothelioma alteration",
        "surveillance_key": "Annual MRI brain + spine from diagnosis; annual audiology + ophthalmology; bevacizumab progressive VS; FAK inhibitors (defactinib) NF2-deficient mesothelioma trials; YAP inhibition therapeutic target",
        "key_distinctions": [
            "BILATERAL-VS-PATHOGNOMONIC-100PCT-AGE-30",
            "SOMATIC-NF2-40-80PCT-SPORADIC-MESOTHELIOMA",
            "BEVACIZUMAB-FDA-PROGRESSIVE-VS",
            "YAP-TAZ-ACTIVATION-NF2-MESOTHELIOMA",
            "DEFACTINIB-FAK-INHIBITOR-NF2-MESOTHELIOMA",
            "SCHWANNOMATOSIS-DDX-SMARCB1-NOT-BILATERAL-VS",
        ],
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 Autosomal-Dominant-LOF -- 156aa -- "
            "p16-INK4A-16kDa-CDK4-6-Inhibitor-FAMM-Melanoma-25-36pct-"
            "Pancreatic-20x-9p21-Deletion-50-80pct-Sporadic-Mesothelioma-CDK4-6i-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / 16 kDa / 9p21.3 CDKN2A encodes two tumour suppressors via alternate reading frames: "
            "DUAL PRODUCT LOCUS: "
            "  CDKN2A locus (9p21.3): encodes TWO functionally distinct proteins via ARF: "
            "    p16-INK4A (exons 1alpha, 2, 3): 156 aa / 16 kDa; ankyrin repeat domain; "
            "    p14-ARF (exon 1beta + exon 2, alternate reading frame): 132 aa / 14 kDa; "
            "  SAME exon 2 is shared, but read in DIFFERENT frames -> completely different proteins; "
            "p16-INK4A FUNCTION: "
            "  p16-INK4A: CDK4/6 inhibitor -- binds CDK4 and CDK6; "
            "  CDK4/CDK6-cyclin D complex: phosphorylates Rb -> releases E2F -> S-phase entry; "
            "  p16-INK4A LOF -> CDK4/6 unimpeded -> Rb hyperphosphorylation -> unchecked G1-S transition; "
            "  Downstream: uncontrolled proliferation -> melanoma, pancreatic cancer; "
            "p14-ARF FUNCTION: "
            "  p14-ARF: MDM2 antagonist -- binds MDM2 -> prevents MDM2-mediated p53 ubiquitination; "
            "  p14-ARF LOF (same CDKN2A deletion): MDM2 free -> p53 degradation -> p53 pathway OFF; "
            "  CDKN2A deletion thus knocks out BOTH Rb (via p16) and p53 (via ARF) pathways simultaneously; "
            "CDKN2A FAMM SYNDROME: "
            "  FAMM = Familial Atypical Multiple Mole Melanoma; "
            "  FAMMM-PC: FAMM + Pancreatic Carcinoma; "
            "  Cutaneous melanoma: 25-36% lifetime = PRIMARY hereditary CDKN2A indication; "
            "  Pancreatic cancer: 20x elevated risk = dominant co-risk after melanoma; "
            "  Annual whole-body skin exam + dermoscopy; "
            "  Baseline pancreatic MRI/MRCP from age 40yr (CAPS consortium guidelines); "
            "9p21 HOMOZYGOUS DELETION -- SPORADIC MESOTHELIOMA: "
            "  9p21 homozygous deletion (CDKN2A/p16 + often CDKN2B co-deletion): 50-80% sporadic mesothelioma; "
            "  CDKN2A FISH on mesothelioma biopsy: p16 deletion = PATHOGNOMONIC for malignant mesothelioma (vs reactive mesothelial); "
            "  CDKN2A FISH + BAP1 IHC: combined test for mesothelioma diagnosis and BAP1/CDKN2A status; "
            "  Hereditary germline CDKN2A: mesothelioma 2-3x (less established than somatic); "
            "CDK4/6 INHIBITORS -- THERAPEUTIC: "
            "  CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib): FDA-approved HR+ breast cancer; "
            "  CDKN2A-deleted tumours: CDK4/6 inhibitor rationale (removing the brake by CDK4/6); "
            "  Palbociclib: clinical trials in CDKN2A-deleted mesothelioma; "
            "  Abemaciclib: broader CDK4/6 + CDK9 inhibition; "
            "  Melanoma with CDK4/6 inhibitors: clinical trials in CDKN2A-deleted melanoma"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 600160 (FAMM); high penetrance melanoma + pancreatic cancer; CDKN2A exon 1beta (ARF) or exon 2 mutations affect both p16 and p14-ARF; family cascade mandatory",
        "cancer_risk": "Cutaneous melanoma 25-36% lifetime PRIMARY; pancreatic cancer 20x elevated (FAMMM-PC); mesothelioma 2-3x hereditary; 9p21 deletion 50-80% sporadic mesothelioma (diagnostic not germline)",
        "pathognomonic": "CDKN2A FISH p16 deletion PATHOGNOMONIC for malignant mesothelioma (vs reactive); FAMM (multiple atypical nevi + melanoma family history) = CDKN2A until proven otherwise; FAMMM-PC = pancreatic cancer co-segregating melanoma family",
        "surveillance_key": "Annual whole-body skin exam + dermoscopy (melanoma); pancreatic MRI/MRCP from age 40yr; CDKN2A FISH + BAP1 IHC: mesothelioma diagnostic panel; CDK4/6 inhibitors for CDKN2A-deleted tumours; photosensitivity counselling",
        "key_distinctions": [
            "9P21-DELETION-50-80PCT-SPORADIC-MESOTHELIOMA",
            "CDKN2A-FISH-PATHOGNOMONIC-MALIGNANT-MESOTHELIOMA",
            "MELANOMA-25-36PCT-FAMM-PRIMARY",
            "PANCREATIC-20X-FAMMM-PC",
            "CDK4-6-INHIBITORS-CDKN2A-DELETED-TUMOURS",
            "P16-INK4A-P14-ARF-DUAL-PRODUCT-SAME-LOCUS",
        ],
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-AVOID-RADIATION-ABSOLUTELY-"
            "WBMRI-Toronto-R337H-Brazilian-Sarcoma-50-60pct-PRIMARY-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes Tumour Protein p53 (Guardian of the Genome): "
            "STRUCTURE: "
            "  393 aa / 43 kDa; tetrameric transcription factor; "
            "  N-terminal transactivation domain (TAD1: aa 1-40; TAD2: aa 40-67); "
            "  Proline-rich domain (aa 67-98); "
            "  DBD (DNA-binding domain): aa 94-292 -- CONTAINS ALL HOTSPOT RESIDUES (R175, G245, R248, R249, R273, R282); "
            "  Tetramerisation domain (aa 325-356) -- forms functional homotetramers; "
            "  C-terminal regulatory domain (aa 356-393): acetylation, ubiquitination; "
            "  p53 activates transcription of: p21 (CDKN1A -- G1 arrest), BAX (apoptosis), PUMA (apoptosis), NOXA, MDM2 (autoregulatory); "
            "  p53 responds to: DNA damage, oncogene activation, hypoxia, oxidative stress; "
            "  GOF hotspot mutations (R175H, R248W, R273H): gain oncogenic function -- drive invasion; "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  OMIM 151623; AD LOF; de novo ~25%; autosomal dominant high penetrance; "
            "  Sarcoma: 50-60% LFS (soft tissue sarcoma + osteosarcoma) = PRIMARY cancer type; "
            "  Breast cancer: 30-40% (early-onset, before age 40yr); "
            "  Brain tumour: 10-15% (astrocytoma, choroid plexus carcinoma PATHOGNOMONIC in children); "
            "  Adrenocortical carcinoma (ACC): 10-15% (children: R337H-associated in Brazil); "
            "  Colorectal, lung, haematologic cancers also elevated; "
            "  LFS mesothelioma: rare in LFS spectrum but sarcoma is overwhelmingly PRIMARY; "
            "AVOID RADIATION -- ABSOLUTE RULE: "
            "  LFS: AVOID THERAPEUTIC RADIATION ABSOLUTELY -- RULE 1 -- no exceptions; "
            "  Radiation-induced secondary sarcomas in TP53 LOF carriers: extreme latency risk; "
            "  Historical cases: radiation to breast cancer -> fatal radiation-induced sarcoma; "
            "  Replace CT surveillance with MRI (non-ionising) in ALL LFS patients; "
            "  Even diagnostic X-rays: minimise; prefer MRI/ultrasound at all times; "
            "TORONTO WBMRI PROTOCOL: "
            "  Whole-Body MRI (WBMRI): annually in LFS -- Toronto Protocol (Kim et al. 2011); "
            "  WBMRI detects sarcoma, breast, ACC, brain, colorectal in one MRI examination; "
            "  Replaces CT-based surveillance -- NO RADIATION; "
            "  Brain MRI (with gadolinium): annual LFS surveillance; "
            "  Abdominal ultrasound: 6-monthly in children; "
            "R337H BRAZILIAN FOUNDER MUTATION: "
            "  TP53 R337H (c.1010G>A): Brazilian founder mutation, frequency ~1/300 southern Brazil; "
            "  Associated with paediatric adrenocortical carcinoma (ACC) predominantly in children; "
            "  Lower penetrance than classic LFS hotspots; "
            "  Brazilian LFS families: screen for R337H by Sanger/NGS; "
            "p53 IHC -- ABERRANT PATTERNS: "
            "  p53-null (complete loss): PATHOGNOMONIC for TP53 LOF mutation; "
            "  p53-overexpression (diffuse strong): PATHOGNOMONIC for missense GOF mutation; "
            "  Wild-type p53 IHC (patchy weak): normal pattern"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 191170/151623; ~25% de novo; highly penetrant LFS (~80-100% lifetime cancer); also rare GOF hotspot mutations; family cascade mandatory",
        "cancer_risk": "Sarcoma (STS + osteosarcoma): 50-60% PRIMARY LFS; breast cancer 30-40%; brain tumour 10-15%; ACC 10-15% children; mesothelioma rare LFS; p53 aberrant = serous histology TCGA",
        "pathognomonic": "AVOID RADIATION ABSOLUTELY (LFS) -- radiation-induced secondary sarcoma lethal risk; R337H Brazilian founder 1/300 south Brazil paediatric ACC; p53-null or p53-overexpression IHC = aberrant p53 PATHOGNOMONIC",
        "surveillance_key": "WBMRI annually Toronto Protocol (no radiation); brain MRI annual; abdominal US 6-monthly children; AVOID radiation absolutely -- MRI preferred; R337H Brazilian screen; TP53 germline all early-onset sarcoma/ACC",
        "key_distinctions": [
            "AVOID-RADIATION-ABSOLUTELY-LFS-RULE-1",
            "WBMRI-TORONTO-ANNUALLY-NO-CT",
            "SARCOMA-50-60PCT-PRIMARY-LFS",
            "R337H-BRAZILIAN-FOUNDER-1-IN-300",
            "P53-NULL-IHC-PATHOGNOMONIC-LOF",
            "CHOROID-PLEXUS-CARCINOMA-PATHOGNOMONIC-CHILDREN",
        ],
    },
    {
        "gene": "SMARCB1",
        "protein": (
            "SMARCB1 -- 22q11.23 Autosomal-Dominant-LOF -- 385aa -- "
            "INI1-44kDa-SWI-SNF-CRD-ATRT-Infants-PATHOGNOMONIC-MRT-Kidney-PATHOGNOMONIC-"
            "SMARCB1-null-IHC-Tazemetostat-EZH2-FDA-Schwannomatosis2-OMIM-601607"
        ),
        "locus": "22q11.23",
        "protein_size": (
            "385 aa / 44 kDa / 22q11.23 SMARCB1 encodes SWI/SNF Complex Subunit (INI1/hSNF5/BAF47): "
            "STRUCTURE: "
            "  385 aa / 44 kDa; core subunit of BAF (SWI/SNF) ATP-dependent chromatin remodelling complex; "
            "  Two rpt (repeat) domains: RPT1 (aa 1-94) + RPT2 (aa 204-297); "
            "  Coiled-coil domain (aa 298-385): mediates BAF complex assembly; "
            "  SMARCB1/INI1 is the gatekeeper of the BAF complex -- required for complex integrity; "
            "  BAF complex: 10-15 subunit complex; uses ATP to remodel nucleosome positioning; "
            "  SMARCB1 LOF -> BAF complex dysfunction -> altered chromatin accessibility -> gene expression changes; "
            "  Mechanism: SMARCB1 LOF creates dependency on EZH2 (PRC2 methyltransferase); "
            "  EZH2 dependency: SMARCB1-null tumours rely on EZH2 for survival -> EZH2 inhibition therapeutic; "
            "AT/RT -- ATYPICAL TERATOID/RHABDOID TUMOUR (PATHOGNOMONIC): "
            "  AT/RT: brain tumour in infants <3yr; PATHOGNOMONIC for SMARCB1 biallelic loss; "
            "  ~50% AT/RT: germline SMARCB1 mutation; rest: somatic biallelic; "
            "  AT/RT: SMARCB1 IHC = null (complete loss) = PATHOGNOMONIC on brain biopsy; "
            "  Treatment: intensive multimodal (chemo + radiation + surgery); poor prognosis <18mo; "
            "MRT -- MALIGNANT RHABDOID TUMOUR OF KIDNEY (PATHOGNOMONIC): "
            "  MRT kidney: infants, SMARCB1 biallelic LOF = PATHOGNOMONIC; "
            "  Highly aggressive; rare; presents <2yr; "
            "  Germline SMARCB1 in ~35% MRT -- RHABDOID TUMOUR PREDISPOSITION SYNDROME (RTPS1); "
            "SCHWANNOMATOSIS TYPE 2 (SMARCB1-SCHWANNOMATOSIS): "
            "  SMARCB1 germline LOF (monoallelic) -> schwannomatosis type 2; "
            "  Multiple peripheral schwannomas WITHOUT bilateral VS (distinguishes from NF2); "
            "  Locus 22q11.23 vs NF2 at 22q12.2 -- same chromosome, different loci; "
            "  Severe neuropathic pain from schwannomas; "
            "SMARCB1-NULL IHC -- PATHOGNOMONIC: "
            "  SMARCB1/INI1 IHC loss = PATHOGNOMONIC for rhabdoid tumours + some mesotheliomas; "
            "  Epithelioid mesothelioma: SMARCB1/INI1 loss in ~5-10%; "
            "  SMARCB1-null IHC distinguishes epithelioid mesothelioma from epithelioid sarcoma (both null); "
            "TAZEMETOSTAT -- EZH2 INHIBITOR (FDA): "
            "  Tazemetostat (EZH2 inhibitor): FDA-approved for SMARCB1-deficient epithelioid sarcoma (2020); "
            "  SMARCB1 LOF -> EZH2 dependency -> EZH2 inhibition selectively kills SMARCB1-null cells; "
            "  Off-label data in SMARCB1-null mesothelioma and AT/RT"
        ),
        "inheritance": "Autosomal Dominant (AD); OMIM 601607 (schwannomatosis) / 613797 (RTPS1); biallelic = AT/RT/MRT; monoallelic = schwannomatosis; de novo common in AT/RT; family cascade mandatory",
        "cancer_risk": "AT/RT (biallelic; infant brain PATHOGNOMONIC); MRT kidney (biallelic infants PATHOGNOMONIC); schwannomatosis (monoallelic); epithelioid mesothelioma SMARCB1 loss 5-10%; epithelioid sarcoma",
        "pathognomonic": "SMARCB1/INI1 null IHC PATHOGNOMONIC for AT/RT (infant brain) + MRT (infant kidney) + schwannomatosis (multiple peripheral schwannomas NO bilateral VS); tazemetostat EZH2 inhibitor FDA-approved SMARCB1-deficient",
        "surveillance_key": "Germline SMARCB1 -> brain MRI infant (AT/RT surveillance); renal ultrasound infant (MRT); schwannomatosis: MRI spine/brain-spine annually; tazemetostat EZH2 inhibitor for SMARCB1-null epithelioid sarcoma/mesothelioma",
        "key_distinctions": [
            "ATRT-INFANT-BRAIN-SMARCB1-NULL-PATHOGNOMONIC",
            "MRT-KIDNEY-INFANT-SMARCB1-NULL-PATHOGNOMONIC",
            "SMARCB1-NULL-IHC-PATHOGNOMONIC",
            "TAZEMETOSTAT-EZH2-INHIBITOR-FDA-SMARCB1",
            "SCHWANNOMATOSIS-TYPE2-NO-BILATERAL-VS",
            "22Q11-SMARCB1-VS-22Q12-NF2-DDX",
        ],
    },
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 756aa -- "
            "MLH1-85kDa-MutL-Homolog1-MMR-Lynch1-CRC-40-50pct-Endometrial-40-50pct-"
            "MSI-H-Pembrolizumab-ALL-HISTOLOGIES-Aspirin-CAPP2-50pct-Reduction-OMIM-120436"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "756 aa / 85 kDa / 3p22.2 MLH1 encodes MutL Homolog 1 (DNA Mismatch Repair Protein): "
            "STRUCTURE: "
            "  756 aa / 85 kDa; MutL family ATP-dependent endonuclease; "
            "  N-terminal ATPase domain (HATPase-c fold): aa 1-336; "
            "  C-terminal dimerisation domain (MLH1-CTD): aa 506-756; "
            "  MLH1 obligately heterodimerises: MutLalpha = MLH1 + PMS2 (primary MMR); "
            "  MutLbeta = MLH1 + PMS1 (minor role); "
            "  MutLgamma = MLH1 + MLH3 (meiotic MMR; minor role); "
            "  MLH1 recruits PMS2 endonuclease latent activity -> nicks DNA strand for excision; "
            "  Latent endonuclease is in PMS2 (DQHA motif) but MLH1 activates it; "
            "  MLH1 LOF -> MutLalpha dysfunction -> MMR failure -> MSI-H phenotype; "
            "LYNCH SYNDROME TYPE 1 (MLH1): "
            "  OMIM 120435; AD LOF; most common Lynch gene causing methylation-spectrum; "
            "  Colorectal cancer (CRC): 40-50% lifetime = PRIMARY Lynch MLH1 risk; "
            "  Endometrial cancer: 40-50% lifetime (highest MLH1 endometrial risk of all Lynch genes); "
            "  Ovarian cancer: 10-12%; gastric cancer: 6-8%; small bowel; urinary tract; "
            "  Mesothelioma in Lynch: RARE; but pembrolizumab FDA-approved MSI-H = all histologies; "
            "MLH1 HYPERMETHYLATION -- SOMATIC (NOT LYNCH): "
            "  MLH1 promoter hypermethylation: sporadic MSI-H CRC in elderly women = NOT Lynch germline; "
            "  Somatic MLH1 methylation: both alleles silenced epigenetically -> MSI-H sporadic CRC; "
            "  Lynch testing: distinguish germline LOF (Lynch) from somatic methylation (sporadic); "
            "  BRAF V600E mutation co-occurring with MLH1 loss -> sporadic (BRAF not in Lynch); "
            "MSI-H IHC -- PATHOGNOMONIC: "
            "  MLH1 protein loss on IHC = PATHOGNOMONIC for MLH1 LOF (germline or somatic); "
            "  MSI-H = 4-gene MMR IHC panel (MLH1, MSH2, MSH6, PMS2): loss of any = MMR-deficient; "
            "  MSI-H by PCR/NGS: high microsatellite instability correlates with IHC loss; "
            "PEMBROLIZUMAB (MSI-H ALL HISTOLOGIES): "
            "  Pembrolizumab (anti-PD1): FDA-approved MSI-H/dMMR solid tumours ALL HISTOLOGIES; "
            "  Mesothelioma + MSI-H: pembrolizumab active (rare but real indication); "
            "  MLH1-mutant mesothelioma (Lynch + somatic): pembrolizumab preferred immunotherapy; "
            "ASPIRIN CAPP2 TRIAL: "
            "  CAPP2 trial: aspirin 600mg/day -> 50% CRC risk reduction in Lynch syndrome; "
            "  LEVEL A evidence for aspirin chemoprevention in Lynch; "
            "  Aspirin mechanism: anti-inflammatory + COX-2 inhibition -> adenoma regression in MMR-deficient colon"
        ),
        "inheritance": "Autosomal Dominant (AD); germline LOF; OMIM 120436/120435; moderate-high penetrance CRC + endometrial; MLH1 hypermethylation = somatic NOT germline Lynch; family cascade mandatory",
        "cancer_risk": "CRC 40-50% PRIMARY Lynch1; endometrial 40-50%; ovarian 10-12%; gastric 6-8%; small bowel; urinary tract; mesothelioma rare; MSI-H all solid tumours pembrolizumab-eligible",
        "pathognomonic": "MLH1 protein loss on IHC PATHOGNOMONIC MMR-deficient; MSI-H by PCR/NGS = dMMR pembrolizumab-eligible; MLH1 hypermethylation + BRAF V600E = sporadic NOT Lynch (key DDx); CAPP2 aspirin 50% CRC reduction LEVEL A",
        "surveillance_key": "Colonoscopy 1-2yr from age 25yr; annual gynaecological surveillance endometrial; MSI-H IHC on all CRC/endometrial tumours universal screening; pembrolizumab MSI-H all histologies; aspirin 600mg/day CAPP2 Lynch",
        "key_distinctions": [
            "MSI-H-IHC-PATHOGNOMONIC-MLH1-LOSS",
            "PEMBROLIZUMAB-MSI-H-ALL-HISTOLOGIES",
            "ASPIRIN-CAPP2-50PCT-CRC-REDUCTION-LEVEL-A",
            "MLH1-HYPERMETHYLATION-SOMATIC-NOT-LYNCH",
            "BRAF-V600E-MLH1-METHYLATION-SPORADIC",
            "CRC-40-50PCT-ENDOMETRIAL-40-50PCT-LYNCH1",
        ],
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Autosomal-Dominant-LOF-Monoallelic-Biallelic-AR-AT -- 3056aa -- "
            "ATM-350kDa-PI3K-Like-Kinase-A-T-Cerebellar-Ataxia-Telangiectasia-"
            "RADIOSENSITIVITY-ABSOLUTE-Biallelic-Olaparib-Ceralasertib-OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM encodes Ataxia-Telangiectasia Mutated Kinase: "
            "STRUCTURE: "
            "  3056 aa / 350 kDa; PI3K-related kinase (PIKK family); "
            "  HEAT repeat domain (aa 1-1960): protein-protein interactions; "
            "  FATC domain (aa 3024-3056): regulatory C-terminal; "
            "  FAT domain (aa 1961-2566): stabilises kinase domain; "
            "  Kinase domain (KD: aa 2713-3013); "
            "  ATM activated by: MRN (MRE11-RAD50-NBS1) complex recruitment at DNA DSBs; "
            "  ATM auto-phosphorylates at Ser1981 -> activation; dimerises; "
            "  ATM kinase substrates: H2AX-Ser139 (gammaH2AX -- DSB marker), BRCA1-Ser1524, CHK2-Thr68, p53-Ser15; "
            "  ATM activates: CHK2 -> CDC25A/B degradation -> S-phase checkpoint + G2/M arrest; "
            "  ATM + p53 Ser15 phosphorylation -> p53 stabilisation -> apoptosis/arrest; "
            "ATAXIA-TELANGIECTASIA (BIALLELIC A-T): "
            "  Biallelic ATM LOF = Ataxia-Telangiectasia (A-T); OMIM 208900; AR; "
            "  Cerebellar ataxia: progressive from age 1-2yr; wheelchair-bound by age 10yr; "
            "  Telangiectasia: oculocutaneous; conjunctival telangiectasia characteristic (onset 3-5yr); "
            "  Immune deficiency: IgA/IgG deficiency; recurrent sinopulmonary infections; "
            "  Endocrine: premature ovarian failure, glucose intolerance; "
            "  RADIOSENSITIVITY ABSOLUTE -- BIALLELIC A-T: "
            "    A-T patients: AVOID ALL THERAPEUTIC RADIATION -- LETHAL if given full-dose radiation; "
            "    A-T radiosensitivity: failure to repair radiation-induced DSBs -> chromosomal catastrophe; "
            "    Diagnostic X-rays: minimise; use MRI/US where possible; "
            "    A-T cancer risk: leukaemia/lymphoma 80-100x elevated; T-cell leukaemia/lymphoma; "
            "MONOALLELIC ATM -- HBOC-2: "
            "  Monoallelic ATM LOF: moderate cancer predisposition (not A-T); "
            "  Mesothelioma: 2-4x elevated risk (modest; asbestos co-exposure relevant); "
            "  Breast cancer: 15-25% lifetime (monoallelic ATM); "
            "  Pancreatic cancer: 5-10x elevated (monoallelic); "
            "  Prostate cancer: 2-4x elevated (monoallelic); "
            "OLAPARIB + CERALASERTIB: "
            "  ATM LOF -> HRD-like -> PARP inhibitor active (BRCAness); "
            "  Olaparib: active in ATM-mutant tumours (ovarian, prostate, pancreatic); "
            "  Ceralasertib (AZD6738; ATRi): ATR inhibitor; "
            "  Ceralasertib + olaparib: clinical trials in ATM-deficient solid tumours (OLAPCO etc.); "
            "  Rationale: ATM LOF -> increased reliance on ATR for replication stress response; "
            "  ATR inhibition in ATM-deficient cells -> synthetic lethality; "
            "SURVEILLANCE (MONOALLELIC ATM): "
            "  Annual MRI breast from age 30yr; "
            "  Pancreatic MRI/MRCP from age 50yr; "
            "  PSA prostate from age 40yr; "
            "  ATM monoallelic: avoid excessive radiation where alternatives exist; "
            "  A-T biallelic: avoid ALL therapeutic radiation absolutely"
        ),
        "inheritance": "Autosomal Dominant (monoallelic LOF -- HBOC-2) / Autosomal Recessive (biallelic LOF -- Ataxia-Telangiectasia); OMIM 607585/208900; monoallelic: moderate risk; biallelic: A-T severe; family cascade mandatory",
        "cancer_risk": "Monoallelic: breast 15-25%, pancreatic 5-10x, mesothelioma 2-4x, prostate 2-4x; biallelic (A-T): leukaemia/lymphoma 80-100x, cerebellar ataxia, telangiectasia, immune deficiency",
        "pathognomonic": "RADIOSENSITIVITY ABSOLUTE (biallelic A-T -- avoid radiation LETHAL); cerebellar ataxia + oculocutaneous telangiectasia = A-T PATHOGNOMONIC combination; ceralasertib (ATRi) + olaparib synthetic lethality ATM-deficient",
        "surveillance_key": "Monoallelic: annual MRI breast age 30yr; pancreatic MRI age 50yr; prostate PSA age 40yr; olaparib PARP inhibitor ATM-mutant; ceralasertib ATRi + olaparib trials; biallelic A-T: avoid radiation ABSOLUTELY",
        "key_distinctions": [
            "RADIOSENSITIVITY-ABSOLUTE-BIALLELIC-AT-LETHAL",
            "CEREBELLAR-ATAXIA-TELANGIECTASIA-AT-PATHOGNOMONIC",
            "MESOTHELIOMA-2-4X-MONOALLELIC-ATM",
            "CERALASERTIB-ATRi-OLAPARIB-SYNTHETIC-LETHALITY",
            "BRCAESS-OLAPARIB-ATM-MUTANT",
            "BIALLELIC-LEUKAEMIA-LYMPHOMA-80-100X",
        ],
    },
]


def _make_patients(gene_entry):
    """Deterministic synthetic cohort: 40 patients per gene."""
    seed = SEED_BASE + ATLAS_GENES.index(gene_entry)
    rng  = random.Random(seed)

    gene = gene_entry["gene"]
    age_params = {
        "BAP1":    (50, 12),
        "BRCA2":   (52, 12),
        "NF2":     (18, 8),
        "CDKN2A":  (38, 12),
        "TP53":    (28, 14),
        "SMARCB1": (22, 12),
        "MLH1":    (44, 12),
        "ATM":     (48, 12),
    }
    mu, sigma = age_params.get(gene, (40, 12))

    severe_rates = {
        "BAP1":    0.72,
        "BRCA2":   0.55,
        "NF2":     0.58,
        "CDKN2A":  0.62,
        "TP53":    0.75,
        "SMARCB1": 0.68,
        "MLH1":    0.58,
        "ATM":     0.52,
    }
    sev_rate = severe_rates.get(gene, 0.5)

    patients = []
    for i in range(40):
        age       = max(5, round(rng.gauss(mu, sigma), 1))
        sev_event = rng.random() < sev_rate
        patients.append({
            "id":        f"{gene}-{i+1:02d}",
            "age_onset": age,
            "severe":    sev_event,
            "seed":      seed,
        })
    return patients


def generate_overview():
    rows = []
    for g in ATLAS_GENES:
        pts   = _make_patients(g)
        sev_n = sum(1 for p in pts if p["severe"])
        rows.append({
            "gene":             g["gene"],
            "locus":            g["locus"],
            "n":                len(pts),
            "severe_n":         sev_n,
            "severe_pct":       round(sev_n / len(pts) * 100, 1),
            "mean_age_onset":   round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "pathognomonic":    g["pathognomonic"],
            "key_distinctions": g["key_distinctions"],
            "surveillance_key": g["surveillance_key"],
            "inheritance":      g["inheritance"],
            "cancer_risk":      g["cancer_risk"],
            "protein":          g["protein"],
        })

    total_pts  = sum(r["n"]       for r in rows)
    total_sev  = sum(r["severe_n"] for r in rows)
    highest    = max(rows, key=lambda r: r["severe_pct"])

    return {
        "atlas":              "Hereditary-Mesothelioma-Predisposition-Atlas",
        "seed_range":         f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes_n":            len(ATLAS_GENES),
        "total_patients":     total_pts,
        "severe_total_n":     total_sev,
        "severe_total_pct":   round(total_sev / total_pts * 100, 1),
        "highest_risk_gene":  highest["gene"],
        "highest_risk_pct":   highest["severe_pct"],
        "gene_summary":       rows,
        "genes_detail": [
            {
                "gene":            g["gene"],
                "inheritance":     g["inheritance"],
                "cancer_risk":     g["cancer_risk"],
                "pathognomonic":   g["pathognomonic"],
                "surveillance_key": g["surveillance_key"],
            }
            for g in ATLAS_GENES
        ],
    }


def generate_breakdown():
    breakdown = []
    for g in ATLAS_GENES:
        pts      = _make_patients(g)
        seed_idx = ATLAS_GENES.index(g)
        sev_n    = sum(1 for p in pts if p["severe"])
        breakdown.append({
            "gene":             g["gene"],
            "locus":            g["locus"],
            "n":                len(pts),
            "seed":             SEED_BASE + seed_idx,
            "mean_age_onset":   round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "severe_n":         sev_n,
            "severe_pct":       round(sev_n / len(pts) * 100, 1),
            "pathognomonic":    g["pathognomonic"],
            "key_distinctions": g["key_distinctions"],
            "surveillance_key": g["surveillance_key"],
        })
    return {"atlas": "Hereditary-Mesothelioma-Predisposition-Atlas", "breakdown": breakdown}


def generate_definitions():
    defs = [
        {
            "term": "BAP1 / BAP1-TPDS / MESOTHELIOMA-30-60PCT-PRIMARY / MBAITS-PATHOGNOMONIC / AVOID-ASBESTOS-ABSOLUTELY",
            "definition": (
                "BAP1 -- 729aa / 80 kDa / 3p21.1 / AD LOF\n"
                "BAP1 Tumour Predisposition Syndrome; mesothelioma 30-60% PRIMARY; MBAITs PATHOGNOMONIC; avoid asbestos.\n\n"
                "BAP1-TPDS CANCER SPECTRUM:\n"
                "  Mesothelioma (malignant pleural): 30-60% lifetime = PRIMARY risk -- highest of all mesothelioma genes.\n"
                "  Uveal melanoma: 30-50% lifetime -- PATHOGNOMONIC combination with mesothelioma.\n"
                "  MBAITs (melanocytic BAP1-mutated atypical intraepidermal tumours): dome-shaped BAP1-null skin lesions = PATHOGNOMONIC.\n"
                "  ccRCC: 15-25% lifetime; cholangiocarcinoma elevated.\n\n"
                "BAP1 NULL IHC -- PATHOGNOMONIC:\n"
                "  BAP1 protein loss on IHC = PATHOGNOMONIC for BAP1-deficient mesothelioma and uveal melanoma.\n"
                "  Somatic BAP1 mutations: 50-60% sporadic malignant mesothelioma = most common somatic alteration.\n"
                "  BAP1-null mesothelioma: paradoxically better prognosis; pembrolizumab immunotherapy active.\n\n"
                "AVOID ASBESTOS -- ABSOLUTE RULE:\n"
                "  Germline BAP1 + asbestos = SYNERGISTIC extreme mesothelioma risk.\n"
                "  ALL germline BAP1 carriers: avoid asbestos occupational/environmental ABSOLUTELY.\n\n"
                "SURVEILLANCE:\n"
                "  Annual CT chest/abdomen (mesothelioma); annual ophthalmology (uveal melanoma).\n"
                "  Annual full-body skin exam (MBAITs -> melanoma); renal MRI 1-2yr (ccRCC).\n"
                "  Family cascade germline testing mandatory."
            ),
        },
        {
            "term": "BRCA2 / HBOC / MESOTHELIOMA-2-5X / OLAPARIB-PARP / FA-D1-BIALLELIC-MOST-SEVERE",
            "definition": (
                "BRCA2 -- 3418aa / 384 kDa / 13q12.3 / AD LOF\n"
                "HBOC; mesothelioma 2-5x; olaparib PARP inhibitor; FA-D1 biallelic MOST SEVERE FA.\n\n"
                "MESOTHELIOMA RISK (BRCA2):\n"
                "  Monoallelic BRCA2 LOF: 2-5x elevated mesothelioma (pleural and peritoneal).\n"
                "  HRD -> genomic instability; synergistic with asbestos exposure.\n"
                "  Cisplatin/carboplatin sensitivity enhanced in BRCA2-mutant mesothelioma.\n\n"
                "OLAPARIB (PARP INHIBITOR):\n"
                "  BRCA2 LOF -> HRD -> PARP inhibitor synthetic lethality.\n"
                "  Olaparib FDA-approved: BRCA2-germline ovarian, breast, pancreatic, prostate.\n"
                "  BRCA2-mutant mesothelioma: off-label PARP inhibitor consideration.\n\n"
                "FA-D1 BIALLELIC -- MOST SEVERE FA:\n"
                "  BRCA2 = FANCD1; biallelic = Fanconi Anemia type D1.\n"
                "  FA-D1 = MOST SEVERE FA phenotype: medulloblastoma + Wilms + AML in early childhood.\n"
                "  FA-D1 children: avoid mitomycin C + cross-linking agents (diagnostic FA sensitivity).\n\n"
                "HBOC SURVEILLANCE:\n"
                "  Annual MRI breast + mammography from age 25yr.\n"
                "  BSO at age 40-45yr (ovarian cancer); PSA from 40yr (prostate 5-8x risk).\n"
                "  Pancreatic MRI/MRCP from age 50yr."
            ),
        },
        {
            "term": "NF2 / BILATERAL-VS-100PCT-PATHOGNOMONIC / SOMATIC-NF2-40-80PCT-MESOTHELIOMA / BEVACIZUMAB-VS",
            "definition": (
                "NF2 -- 595aa / 70 kDa / 22q12.2 / AD LOF\n"
                "Bilateral VS PATHOGNOMONIC 100% by age 30yr; somatic NF2 40-80% sporadic mesothelioma; bevacizumab VS.\n\n"
                "BILATERAL VESTIBULAR SCHWANNOMA -- PATHOGNOMONIC:\n"
                "  Bilateral VS: 100% penetrance by age 30yr = PATHOGNOMONIC NF2 diagnosis.\n"
                "  Hearing loss + tinnitus: FIRST symptoms in ~90%.\n"
                "  Meningioma: 50-75%; spinal ependymoma: 33-53%.\n\n"
                "SOMATIC NF2 IN SPORADIC MESOTHELIOMA:\n"
                "  Somatic NF2 deletion: 40-80% sporadic malignant mesothelioma = MOST COMMON somatic alteration.\n"
                "  NF2 LOF -> YAP/TAZ activation -> mesothelioma proliferative driver.\n"
                "  Hereditary NF2: mesothelioma elevated but LESS penetrant than BAP1.\n\n"
                "THERAPEUTIC TARGETS -- NF2/YAP:\n"
                "  Bevacizumab (anti-VEGF): FDA-approved progressive NF2 vestibular schwannoma.\n"
                "  FAK inhibitor (defactinib): phase II NF2-deficient mesothelioma (COMMAND trial).\n"
                "  Verteporfin: YAP inhibitor preclinical NF2-deficient mesothelioma.\n\n"
                "DDX: NF2 vs SCHWANNOMATOSIS:\n"
                "  NF2 (22q12.2): bilateral VS PATHOGNOMONIC.\n"
                "  Schwannomatosis (SMARCB1 22q11.23 / LZTR1): multiple schwannomas WITHOUT bilateral VS."
            ),
        },
        {
            "term": "CDKN2A / FAMM / MELANOMA-25-36PCT / 9P21-DELETION-MESOTHELIOMA-50-80PCT / CDK4-6-INHIBITORS",
            "definition": (
                "CDKN2A -- 156aa / 16 kDa / 9p21.3 / AD LOF\n"
                "FAMM; melanoma 25-36% PRIMARY; 9p21 deletion 50-80% sporadic mesothelioma; CDK4/6 inhibitors.\n\n"
                "FAMM + FAMMM-PC:\n"
                "  FAMM: familial atypical multiple mole melanoma.\n"
                "  Cutaneous melanoma: 25-36% lifetime = PRIMARY hereditary CDKN2A indication.\n"
                "  Pancreatic cancer: 20x elevated = FAMMM-PC co-risk.\n\n"
                "9P21 DELETION -- SPORADIC MESOTHELIOMA:\n"
                "  9p21 homozygous deletion (CDKN2A/p16): 50-80% sporadic mesothelioma.\n"
                "  CDKN2A FISH: p16 deletion PATHOGNOMONIC for malignant mesothelioma (vs reactive mesothelium).\n"
                "  CDKN2A FISH + BAP1 IHC: combined mesothelioma diagnostic panel.\n\n"
                "DUAL PRODUCT -- p16-INK4A + p14-ARF:\n"
                "  Same 9p21.3 locus encodes two proteins via alternate reading frames.\n"
                "  p16-INK4A: CDK4/6 inhibitor -> Rb pathway.\n"
                "  p14-ARF: MDM2 antagonist -> p53 pathway.\n"
                "  CDKN2A deletion knocks out BOTH Rb AND p53 pathways simultaneously.\n\n"
                "CDK4/6 INHIBITORS:\n"
                "  Palbociclib, ribociclib, abemaciclib: FDA-approved HR+ breast cancer.\n"
                "  CDKN2A-deleted mesothelioma: CDK4/6 inhibitor trials ongoing."
            ),
        },
        {
            "term": "TP53 / LFS / AVOID-RADIATION-ABSOLUTELY / WBMRI-TORONTO / R337H-BRAZILIAN-FOUNDER",
            "definition": (
                "TP53 -- 393aa / 43 kDa / 17p13.1 / AD LOF\n"
                "Li-Fraumeni Syndrome; AVOID RADIATION ABSOLUTELY; WBMRI Toronto Protocol; R337H Brazilian founder.\n\n"
                "LFS CANCER SPECTRUM:\n"
                "  Sarcoma (STS + osteosarcoma): 50-60% = PRIMARY LFS cancer.\n"
                "  Breast cancer: 30-40% (early-onset <40yr).\n"
                "  Brain tumour: 10-15% (choroid plexus carcinoma in children PATHOGNOMONIC).\n"
                "  ACC: 10-15% children; leukaemia, colon elevated.\n\n"
                "AVOID RADIATION -- ABSOLUTE RULE 1:\n"
                "  LFS: NO THERAPEUTIC RADIATION -- radiation-induced secondary sarcoma lethal risk.\n"
                "  Replace CT surveillance with MRI (non-ionising) in ALL LFS patients.\n"
                "  Even diagnostic X-rays: minimise; prefer MRI/ultrasound.\n\n"
                "TORONTO WBMRI PROTOCOL:\n"
                "  WBMRI: annually -- detects sarcoma, breast, ACC, brain, CRC in one examination.\n"
                "  No ionising radiation; brain MRI annual (with gadolinium).\n\n"
                "R337H BRAZILIAN FOUNDER:\n"
                "  TP53 R337H: frequency ~1/300 southern Brazil.\n"
                "  Associated with paediatric ACC predominantly.\n\n"
                "p53 IHC PATTERNS:\n"
                "  p53-null (complete loss): LOF mutation PATHOGNOMONIC.\n"
                "  p53-overexpression (diffuse strong): GOF missense PATHOGNOMONIC."
            ),
        },
        {
            "term": "SMARCB1 / ATRT-INFANT-PATHOGNOMONIC / MRT-KIDNEY-PATHOGNOMONIC / TAZEMETOSTAT-EZH2-FDA / SCHWANNOMATOSIS2",
            "definition": (
                "SMARCB1 -- 385aa / 44 kDa / 22q11.23 / AD LOF (monoallelic) / biallelic AT/RT+MRT\n"
                "AT/RT infant PATHOGNOMONIC; MRT kidney infant PATHOGNOMONIC; tazemetostat EZH2 FDA.\n\n"
                "AT/RT -- ATYPICAL TERATOID/RHABDOID TUMOUR:\n"
                "  AT/RT: brain tumour infants <3yr = PATHOGNOMONIC SMARCB1 biallelic loss.\n"
                "  ~50% AT/RT have germline SMARCB1; SMARCB1/INI1 null IHC on biopsy PATHOGNOMONIC.\n"
                "  Treatment: intensive multimodal; poor prognosis <18mo.\n\n"
                "MRT -- MALIGNANT RHABDOID TUMOUR OF KIDNEY:\n"
                "  MRT kidney: infants, SMARCB1 biallelic = PATHOGNOMONIC.\n"
                "  Germline in ~35% MRT = Rhabdoid Tumour Predisposition Syndrome (RTPS1).\n\n"
                "TAZEMETOSTAT -- EZH2 INHIBITOR FDA:\n"
                "  SMARCB1 LOF -> EZH2 dependency -> tazemetostat selectively kills SMARCB1-null cells.\n"
                "  FDA-approved: SMARCB1-deficient epithelioid sarcoma (2020).\n"
                "  Off-label data in SMARCB1-null epithelioid mesothelioma and AT/RT.\n\n"
                "SCHWANNOMATOSIS TYPE 2 vs NF2:\n"
                "  SMARCB1 schwannomatosis: 22q11.23; multiple schwannomas, NO bilateral VS.\n"
                "  NF2: 22q12.2; bilateral VS PATHOGNOMONIC.\n"
                "  Same chromosome 22 -- different loci: distinguish by clinical phenotype + IHC + molecular."
            ),
        },
        {
            "term": "MLH1 / LYNCH1 / MSI-H-PATHOGNOMONIC / PEMBROLIZUMAB-ALL-HISTOLOGIES / ASPIRIN-CAPP2-50PCT",
            "definition": (
                "MLH1 -- 756aa / 85 kDa / 3p22.2 / AD LOF\n"
                "Lynch Syndrome type 1; MSI-H IHC PATHOGNOMONIC; pembrolizumab all histologies; CAPP2 aspirin 50% reduction.\n\n"
                "LYNCH CRC + ENDOMETRIAL PRIMARY:\n"
                "  CRC: 40-50% lifetime = PRIMARY Lynch1 indication.\n"
                "  Endometrial: 40-50% lifetime; ovarian 10-12%; gastric 6-8%.\n"
                "  Mesothelioma in Lynch: RARE; but MSI-H = pembrolizumab eligible.\n\n"
                "MSI-H IHC -- PATHOGNOMONIC:\n"
                "  MLH1 protein loss on IHC = PATHOGNOMONIC MMR-deficient.\n"
                "  4-gene MMR IHC (MLH1/MSH2/MSH6/PMS2): loss of any = dMMR.\n"
                "  MSI-H by PCR/NGS correlates with IHC loss.\n\n"
                "MLH1 HYPERMETHYLATION -- SOMATIC NOT GERMLINE:\n"
                "  MLH1 promoter hypermethylation + BRAF V600E = sporadic NOT Lynch.\n"
                "  Distinguish from germline Lynch before counselling family members.\n\n"
                "PEMBROLIZUMAB (MSI-H ALL HISTOLOGIES):\n"
                "  Pembrolizumab FDA-approved: MSI-H/dMMR solid tumours ALL histologies.\n"
                "  MLH1-mutant mesothelioma (rare but real): pembrolizumab preferred.\n\n"
                "ASPIRIN CAPP2 -- LEVEL A:\n"
                "  Aspirin 600mg/day: 50% CRC risk reduction in Lynch = LEVEL A evidence.\n"
                "  Anti-inflammatory + COX-2 inhibition -> adenoma regression in MMR-deficient colon."
            ),
        },
        {
            "term": "ATM / A-T-BIALLELIC / RADIOSENSITIVITY-ABSOLUTE / MESOTHELIOMA-2-4X / CERALASERTIB-ATRi-OLAPARIB",
            "definition": (
                "ATM -- 3056aa / 350 kDa / 11q22.3 / AD LOF (monoallelic) / AR biallelic A-T\n"
                "Ataxia-Telangiectasia biallelic; radiosensitivity ABSOLUTE; mesothelioma 2-4x monoallelic; ceralasertib ATRi.\n\n"
                "ATAXIA-TELANGIECTASIA (BIALLELIC A-T):\n"
                "  Cerebellar ataxia: progressive from age 1-2yr; wheelchair-bound age 10yr.\n"
                "  Telangiectasia: conjunctival/oculocutaneous characteristic.\n"
                "  Immune deficiency: IgA/IgG deficiency; sinopulmonary infections.\n\n"
                "RADIOSENSITIVITY ABSOLUTE -- BIALLELIC A-T:\n"
                "  A-T: AVOID ALL THERAPEUTIC RADIATION -- LETHAL.\n"
                "  ATM LOF -> failed DSB repair after radiation -> chromosomal catastrophe.\n"
                "  Use MRI/US alternatives; even diagnostic X-rays: minimise.\n"
                "  A-T leukaemia/lymphoma: 80-100x elevated.\n\n"
                "MONOALLELIC ATM -- HBOC-2:\n"
                "  Mesothelioma: 2-4x elevated (asbestos co-exposure relevant).\n"
                "  Breast cancer: 15-25% lifetime; pancreatic 5-10x; prostate 2-4x.\n\n"
                "CERALASERTIB (ATRi) + OLAPARIB:\n"
                "  ATM LOF -> HRD-like -> PARP inhibitor synthetic lethality (olaparib).\n"
                "  Ceralasertib (ATR inhibitor) + olaparib: clinical trials ATM-deficient solid tumours.\n"
                "  Rationale: ATM LOF -> increased ATR reliance -> ATRi synthetic lethal.\n\n"
                "CASCADE TESTING -- HEREDITARY MESOTHELIOMA PANEL:\n"
                "  BAP1 (primary): BAP1-null IHC + germline sequencing.\n"
                "  CDKN2A FISH: p16 deletion diagnostic; 9p21 homozygous deletion = malignant mesothelioma.\n"
                "  NF2 somatic: tumour profiling BAP1 + NF2 + CDKN2A three-gene panel.\n"
                "  BRCA2/ATM/PALB2: HRD tumours -- platinum/PARP sensitivity.\n"
                "  SMARCB1: INI1 IHC on all epithelioid mesothelioma.\n"
                "  MLH1/MMR: dMMR mesothelioma -> pembrolizumab.\n\n"
                "TIER 1 -- MOST ACTIONABLE (TARGETED THERAPY):\n"
                "  BAP1-null: pembrolizumab (immunogenic); avoid asbestos absolutely.\n"
                "  CDKN2A-deleted: CDK4/6 inhibitor trials; palbociclib.\n"
                "  NF2-deficient: FAK inhibitor defactinib; YAP inhibition.\n\n"
                "TIER 2 -- DNA REPAIR (HRD TUMOURS):\n"
                "  BRCA2/ATM LOF: cisplatin/olaparib/ceralasertib sensitivity.\n"
                "  PALB2: olaparib + platinum HRD regimen.\n\n"
                "TIER 3 -- IMMUNE/EPIGENETIC:\n"
                "  MLH1 dMMR: pembrolizumab all histologies MSI-H.\n"
                "  SMARCB1-null: tazemetostat EZH2 inhibitor."
            ),
        },
    ]

    return {
        "atlas":       "Hereditary-Mesothelioma-Predisposition-Atlas",
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
        print(f"  {row['gene']:10s} n={row['n']} mean_age={row['mean_age_onset']} "
              f"severe_n={row['severe_n']} ({row['severe_pct']}%)")
    print("\n=== DEFINITIONS (terms only) ===")
    df = generate_definitions()
    for d in df["definitions"]:
        print(f"  {d['term'][:80]}")
