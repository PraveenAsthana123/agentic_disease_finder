#!/usr/bin/env python3
"""Hereditary-CNS-Brain-Tumor-Predisposition-Atlas -- Complete 8-Gene Reference
NF1     (Neurofibromin 1; 2839aa; 17q11.2; AD LOF;
         Neurofibromatosis type 1;
         Optic glioma grade-II PATHOGNOMONIC; MPNST 8-13% HIGHEST;
         selumetinib MEKi FDA2020 NF1-optic glioma;
         café-au-lait macules ≥6 PATHOGNOMONIC; Lisch nodules PATHOGNOMONIC;
         seed SEED_BASE+0) .
NF2     (Neurofibromin 2 / merlin / schwannomin; 595aa; 22q12.2; AD LOF;
         Neurofibromatosis type 2;
         bilateral vestibular schwannomas PATHOGNOMONIC (90-95%);
         bevacizumab VEGF inhibitor hearing preservation;
         juvenile posterior subcapsular cataract 80% PATHOGNOMONIC;
         seed SEED_BASE+1) .
VHL     (Von Hippel-Lindau; 213aa; 3p25.3; AD LOF;
         Von Hippel-Lindau disease;
         hemangioblastoma cerebellar/spinal PATHOGNOMONIC (60-80%);
         retinal hemangioblastoma PATHOGNOMONIC; ccRCC 50x;
         belzutifan HIF-2α inhibitor FDA2021;
         seed SEED_BASE+2) .
TSC1    (Tuberous Sclerosis Complex 1 / hamartin; 1164aa; 9q34.13; AD LOF;
         Tuberous sclerosis complex type 1;
         SEGA PATHOGNOMONIC 5-20%; cortical tubers PATHOGNOMONIC;
         everolimus mTOR FDA-approved SEGA+AML;
         cardiac rhabdomyoma neonatal PATHOGNOMONIC; shagreen patch PATHOGNOMONIC;
         seed SEED_BASE+3) .
PTEN    (Phosphatase and tensin homolog; 403aa; 10q23.31; AD LOF;
         Cowden syndrome / PHTS;
         Lhermitte-Duclos disease PATHOGNOMONIC adult;
         macrocephaly PATHOGNOMONIC 90% OFC>97th percentile;
         breast 85% HIGHEST; everolimus mTOR;
         seed SEED_BASE+4) .
SMARCB1 (SWI/SNF related matrix associated actin dependent regulator B1 / INI1; 385aa; 22q11.23; AD LOF;
         Rhabdoid tumor predisposition syndrome 2 (RTPS2);
         ATRT under 3yr PATHOGNOMONIC; INI1 IHC loss PATHOGNOMONIC;
         tazemetostat EZH2 inhibitor FDA2020;
         schwannomatosis NF3-like;
         seed SEED_BASE+5) .
SUFU    (Suppressor of fused; 484aa; 10q24.32; AD LOF;
         Gorlin-like / SHH pathway;
         medulloblastoma SHH HIGHEST germline risk 50-60% lifetime;
         desmoplastic/nodular MB PATHOGNOMONIC;
         AVOID radiation young children; AVOID vismodegib developing skeleton;
         seed SEED_BASE+6) .
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni syndrome;
         DIPG H3K27M PATHOGNOMONIC LFS children (20-30% pediatric DIPG);
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annual;
         ONC201/dordaviprone FDA2022 H3K27M+;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3246-3253)
"""
import random

SEED_BASE = 3246

ATLAS_GENES = [
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RAS-GAP-NF1-Optic-Glioma-PATHOGNOMONIC-"
            "MPNST-8-13pct-HIGHEST-Selumetinib-MEKi-FDA2020-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 encodes Neurofibromin (RAS-GAP tumour suppressor): "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; cytoplasmic RAS GTPase-activating protein (GAP); "
            "  GRD (GTPase-related domain, aa 1175-1530): catalytic RAS-GAP core; "
            "  Accelerates RAS-GTP -> RAS-GDP hydrolysis (400x vs intrinsic rate); "
            "  PH domain (aa 1560-1672): lipid binding; membrane recruitment; "
            "  SEC14-like domain (aa 1572-1837): lipid transfer; "
            "  Syn domain (aa 1726-1837): synapsin-like; neuronal function; "
            "  NF1 LOF -> RAS-GTP accumulates -> RAF-MEK-ERK hyperactivation -> tumour; "
            "  NF1 is the most common dominant tumour suppressor syndrome (1/3,000 births); "
            "NEUROFIBROMATOSIS TYPE 1 (NF1): "
            "  OMIM 162200; AD LOF; 1/3,000 births; 50% de novo; full penetrance; "
            "  Diagnostic criteria (NIH): ≥6 café-au-lait macules ≥5mm (prepubertal) or ≥15mm (postpubertal) PATHOGNOMONIC; "
            "  Lisch nodules (iris hamartomas): >90% adults PATHOGNOMONIC; "
            "  Axillary/inguinal freckling PATHOGNOMONIC (Crowe sign); "
            "  Cutaneous neurofibromas: appear puberty; plexiform neurofibroma 25-30%; "
            "  Optic pathway glioma (OPG): 15-20% NF1 children; grade II pilocytic astrocytoma; "
            "  OPG: typically asymptomatic; proptosis/vision loss = treatment indication; "
            "  Orbital plexiform NF: proptosis PATHOGNOMONIC (periorbital); "
            "BRAIN TUMOUR PREDISPOSITION (NF1): "
            "  Optic glioma grade II PATHOGNOMONIC: 15-20% NF1 children (most common NF1 CNS tumour); "
            "  Low-grade astrocytoma (brainstem, cerebellum, thalamus): 5-10%; "
            "  GBM: 1-2% NF1 (elevated vs population); "
            "  MPNST (malignant peripheral nerve sheath tumour): 8-13% lifetime HIGHEST (sarcoma); "
            "  MPNST arises from plexiform NF; rapid growth = malignant transformation signal; "
            "  NF1-MPNST: 5-year survival 20-50% (worse prognosis vs sporadic MPNST); "
            "  Breast cancer: 3-5x elevated (similar to BRCA1 mutation carriers); "
            "SELUMETINIB (MEK INHIBITOR -- FDA2020): "
            "  Selumetinib (AZD6244/Koselugo): MEK1/2 inhibitor; first FDA-approved NF1-specific drug; "
            "  FDA2020: NF1 symptomatic inoperable plexiform neurofibromas (children ≥2yr); "
            "  SPRINT trial: 71% plexiform NF volume reduction ≥20% (vs 0% placebo); "
            "  Optic pathway glioma: selumetinib active (off-label expanding); "
            "  AVOID radiation NF1: secondary MPNST risk (radiation-induced NF1-field malignancy); "
            "  Everolimus (mTOR): alternative for refractory plexiform NF; "
            "SURVEILLANCE (NF1): "
            "  Annual ophthalmology from birth to age 8yr (OPG screening); "
            "  Annual MRI brain/spine if OPG or neurological symptoms; "
            "  Annual blood pressure (renal artery stenosis, pheo risk NF1 type 2 locus); "
            "  Annual plexiform NF examination (growth = MPNST risk); "
            "  Annual breast MRI from age 30yr (breast cancer risk); "
            "  Cascade: first-degree relatives NF1 clinical + molecular testing"
        ),
        "inheritance": "AD LOF; OMIM 162200; 1/3,000 births = most common dominant cancer predisposition; 50% de novo; full penetrance; expressivity highly variable; NF1 second allele somatic LOF required for tumour (Knudson two-hit)",
        "cancer_risk": "Optic glioma grade II 15-20% PATHOGNOMONIC; low-grade astrocytoma 5-10%; GBM 1-2%; MPNST 8-13% HIGHEST sarcoma risk in NF1; breast 3-5x elevated; plexiform NF 25-30%",
        "pathognomonic": "Café-au-lait macules ≥6 PATHOGNOMONIC; Lisch nodules (iris hamartomas) >90% adults PATHOGNOMONIC; axillary freckling PATHOGNOMONIC; optic glioma grade II PATHOGNOMONIC NF1 children; plexiform neurofibroma PATHOGNOMONIC",
        "surveillance_key": "Annual ophthalmology from birth to 8yr (OPG); selumetinib MEKi FDA2020 plexiform NF; AVOID radiation (secondary MPNST risk); annual breast MRI from 30yr; MRI brain/spine if neurological symptoms",
        "key_distinctions": [
            "OPTIC-GLIOMA-GRADE-II-PATHOGNOMONIC-15-20PCT-NF1-CHILDREN",
            "MPNST-8-13PCT-HIGHEST-SARCOMA-NF1",
            "SELUMETINIB-FDA2020-FIRST-NF1-SPECIFIC-DRUG",
            "AVOID-RADIATION-SECONDARY-MPNST-NF1",
            "CAFE-AU-LAIT-6-PATHOGNOMONIC",
            "LISCH-NODULES-PATHOGNOMONIC-90PCT-ADULTS",
        ],
    },
    {
        "gene": "NF2",
        "protein": (
            "NF2 -- 22q12.2 Autosomal-Dominant-LOF -- 595aa -- "
            "Merlin-Schwannomin-66kDa-ERM-Family-Bilateral-VS-PATHOGNOMONIC-"
            "Bevacizumab-VEGF-Aspirin-Cataract-LATS1-2-Meningioma-OMIM-101000"
        ),
        "locus": "22q12.2",
        "protein_size": (
            "595 aa / 66 kDa / 22q12.2 NF2 encodes Merlin (schwannomin; ERM family tumour suppressor): "
            "STRUCTURE: "
            "  595 aa / 66 kDa; ERM (ezrin-radixin-moesin) family scaffold; "
            "  FERM domain (aa 1-310): membrane-cytoskeleton linker; tumour suppressor activity; "
            "  Alpha-helical domain (aa 311-478): self-association; autoinhibition; "
            "  C-terminal tail (aa 479-595): intramolecular interaction; isoform regulation; "
            "  Merlin closed (inactive) conformation: FERM + C-tail interaction; "
            "  Merlin open (active/tumour-suppressive) conformation: loss of self-association; "
            "  Merlin suppresses LATS1/2 (Hippo pathway) -> YAP/TAZ nuclear exclusion; "
            "  Merlin LOF -> LATS1/2 inactive -> YAP/TAZ nuclear (oncogenic) -> meningioma/schwannoma; "
            "  NF2 LOF also activates: mTOR, PI3K, focal adhesion kinase (FAK) pathways; "
            "NEUROFIBROMATOSIS TYPE 2 (NF2): "
            "  OMIM 101000; AD LOF; 1/25,000-1/60,000 births; 50% de novo; "
            "  Diagnostic criteria (Manchester): bilateral VS (most specific) or unilateral VS age <30yr + first-degree relative NF2; "
            "  Bilateral vestibular schwannomas: 90-95% NF2 by age 30yr PATHOGNOMONIC; "
            "  Unilateral VS age <30yr: strong NF2 suspicion; "
            "  Meningioma: 50-80% NF2 (multiple meningiomas PATHOGNOMONIC); "
            "  Ependymoma: 30-53% (spinal ependymoma predominant); "
            "  Cataract: juvenile posterior subcapsular or cortical 80% NF2 PATHOGNOMONIC; "
            "  NF2 vs NF1: NF2 has minimal/no café-au-lait macules (KEY DISTINCTION from NF1); "
            "VESTIBULAR SCHWANNOMA MANAGEMENT (NF2): "
            "  Bevacizumab (VEGF inhibitor): NF2-associated VS -- hearing preservation; "
            "  Bevacizumab: tumour volume reduction 40-50%; hearing improvement 36%; "
            "  Aspirin 325mg/day: emerging anti-merlin-LOF signal; NF2 tumour growth inhibition; "
            "  Surgery: VS resection when vestibular > hearing preservation priority; "
            "  Radiosurgery (GK): option for small-medium VS (radiation OK in NF2 unlike NF1/LFS); "
            "  MRI brain + full spine at diagnosis (ependymoma baseline); "
            "LATS1/2 SOMATIC CO-INACTIVATION (MENINGIOMA): "
            "  NF2-null meningioma: LATS1/2 somatic co-inactivation in 30-40% meningioma; "
            "  LATS1/2 LOF -> YAP nuclear -> E3 ubiquitin ligase PRAJA2 degradation; "
            "  FAK inhibitors (defactinib): NF2-null meningioma + VS trials ongoing; "
            "  AR (androgen receptor) blockade: meningioma emerging (sex-hormone driven); "
            "SURVEILLANCE (NF2): "
            "  Annual MRI brain + full spine from diagnosis; "
            "  Annual audiogram (mandatory -- hearing baseline and monitoring); "
            "  Annual ophthalmology (cataract, retinal hamartoma); "
            "  Avoid noise trauma (acoustic neuroma worsening); "
            "  Cascade: first-degree relatives NF2 clinical evaluation + MRI"
        ),
        "inheritance": "AD LOF; OMIM 101000; 1/25,000-1/60,000 births; 50% de novo; NF2 second allele somatic LOF required for tumour; Knudson two-hit; expressivity variable (Wishart vs Gardner subtypes); somatic mosaicism common",
        "cancer_risk": "Bilateral VS 90-95% PATHOGNOMONIC by 30yr; meningioma 50-80% (multiple PATHOGNOMONIC); spinal ependymoma 30-53%; juvenile posterior subcapsular cataract 80% PATHOGNOMONIC; schwannomatosis overlap",
        "pathognomonic": "Bilateral vestibular schwannomas PATHOGNOMONIC (90-95% NF2); juvenile posterior subcapsular/cortical cataract 80% PATHOGNOMONIC; multiple meningiomas PATHOGNOMONIC; NF2 LACKS café-au-lait macules (distinguishes from NF1)",
        "surveillance_key": "Annual MRI brain + full spine; annual audiogram MANDATORY; bevacizumab VEGF for VS hearing preservation; aspirin 325mg emerging; avoid noise trauma; radiosurgery OK (unlike NF1/LFS); FAK inhibitors defactinib trials",
        "key_distinctions": [
            "BILATERAL-VS-PATHOGNOMONIC-90-95PCT-NF2",
            "JUVENILE-POSTERIOR-SUBCAPSULAR-CATARACT-80PCT-PATHOGNOMONIC",
            "ANNUAL-AUDIOGRAM-MANDATORY",
            "BEVACIZUMAB-VEGF-HEARING-PRESERVATION",
            "LATS1-2-SOMATIC-COINACTIVATION-MENINGIOMA",
            "NF2-NO-CAFE-AU-LAIT-DISTINGUISHES-FROM-NF1",
        ],
    },
    {
        "gene": "VHL",
        "protein": (
            "VHL -- 3p25.3 Autosomal-Dominant-LOF -- 213aa -- "
            "pVHL-24kDa-HIF-Ubiquitin-E3-Ligase-Hemangioblastoma-PATHOGNOMONIC-"
            "ccRCC-50x-Belzutifan-HIF2a-FDA2021-ELST-PATHOGNOMONIC-OMIM-193300"
        ),
        "locus": "3p25.3",
        "protein_size": (
            "213 aa / 24 kDa / 3p25.3 VHL encodes pVHL (von Hippel-Lindau protein; HIF-2α E3 ubiquitin ligase adaptor): "
            "STRUCTURE: "
            "  213 aa / 24 kDa; substrate recognition subunit of CRL2-VHL E3 ubiquitin ligase; "
            "  Alpha domain (aa 63-154): elongin-B/C binding; CRL2 complex formation; "
            "  Beta domain (aa 1-62, 155-213): HIF-alpha recognition; "
            "  pVHL binds prolyl-hydroxylated HIF-1α/HIF-2α -> ubiquitination -> proteasomal degradation; "
            "  Under normoxia: PHD (prolyl hydroxylase) hydroxylates HIF-alpha -> pVHL recognition -> degradation; "
            "  VHL LOF -> unhydroxylated HIF-alpha accumulates (pseudo-hypoxia) -> VEGF, EPO, PDGF, Glut1 upregulation; "
            "  HIF-2α (EPAS1): principal oncogenic driver in VHL-null ccRCC and hemangioblastoma; "
            "  pVHL also: regulates microtubule stability, primary cilia formation, NFkB; "
            "VHL DISEASE: "
            "  OMIM 193300; AD LOF; 1/36,000 births; 20% de novo; "
            "  Penetrance: near-complete by age 65yr; biallelic somatic VHL 70% sporadic ccRCC; "
            "  VHL type 1: LOF/truncating -> hemangioblastoma + ccRCC (no pheo); "
            "  VHL type 2A: missense -> hemangioblastoma + pheo (low ccRCC risk); "
            "  VHL type 2B: missense -> hemangioblastoma + pheo + ccRCC (highest risk); "
            "  VHL type 2C: missense -> pheo only (no CNS/ccRCC); "
            "CNS TUMOUR PREDISPOSITION (VHL): "
            "  Hemangioblastoma cerebellar: 60-80% lifetime PATHOGNOMONIC (VHL most common hereditary HGB cause); "
            "  Hemangioblastoma spinal cord: 13-50% (thoracic predominant); "
            "  Hemangioblastoma brainstem: 5-10%; "
            "  ELST (endolymphatic sac tumour): 10-15% VHL PATHOGNOMONIC (bilateral ELST = VHL until proven otherwise); "
            "  Retinal hemangioblastoma: 40-60% (earliest lesion, often precedes CNS); "
            "  Pancreatic cysts 35-70%; pancreatic NET 5-17%; "
            "BELZUTIFAN (HIF-2α INHIBITOR -- FDA2021): "
            "  Belzutifan (PT2977/MK-6482): oral HIF-2α (EPAS1) inhibitor; "
            "  FDA2021: VHL disease-associated ccRCC + CNS hemangioblastoma + pNET; "
            "  LITESPARK-004 trial: 49% ORR ccRCC; 30% ORR CNS HGB; "
            "  Mechanism: occupies PAS-B pocket of HIF-2α -> disrupts HIF-2α/ARNT dimerisation; "
            "  AVOID radiation: VHL hemangioblastoma surgery preferred; SRS for inaccessible small HGB; "
            "SURVEILLANCE (VHL): "
            "  Annual MRI brain + full spine from age 11yr (CNS HGB); "
            "  Annual retinal examination from age 1yr (retinal HGB earliest); "
            "  Annual abdominal MRI/US from age 15yr (ccRCC, pheo, pancreatic); "
            "  Annual urine catecholamines / plasma metanephrines (pheo); "
            "  Annual audiogram (ELST); "
            "  Cascade: all first-degree relatives VHL molecular testing"
        ),
        "inheritance": "AD LOF; OMIM 193300; 1/36,000 births; 20% de novo; VHL type 1 vs 2A/2B/2C genotype-phenotype; near-complete penetrance age 65yr; biallelic somatic VHL = 70% sporadic ccRCC",
        "cancer_risk": "Hemangioblastoma cerebellar/spinal 60-80% PATHOGNOMONIC; retinal hemangioblastoma 40-60% PATHOGNOMONIC; ccRCC 24-45% (50x elevated); pheo 10-20% (type 2); ELST 10-15% PATHOGNOMONIC; pNET 5-17%",
        "pathognomonic": "Hemangioblastoma cerebellar/spinal PATHOGNOMONIC (60-80%); retinal hemangioblastoma PATHOGNOMONIC (40-60%); bilateral ELST PATHOGNOMONIC VHL; ccRCC 50x elevated; belzutifan FDA2021 HIF-2α inhibitor first-in-class",
        "surveillance_key": "Annual MRI brain + spine from age 11yr; annual retinal exam from age 1yr; belzutifan FDA2021 VHL-ccRCC + CNS-HGB; annual metanephrines (pheo); annual audiogram (ELST); VHL type 1 vs 2 genotype-phenotype",
        "key_distinctions": [
            "HEMANGIOBLASTOMA-PATHOGNOMONIC-60-80PCT-VHL",
            "RETINAL-HEMANGIOBLASTOMA-PATHOGNOMONIC-40-60PCT",
            "ELST-BILATERAL-PATHOGNOMONIC-VHL",
            "BELZUTIFAN-HIF2A-FDA2021-FIRST-IN-CLASS",
            "VHL-TYPE1-NO-PHEO-TYPE2-PHEO",
            "CCRCC-50X-ELEVATED-VHL",
        ],
    },
    {
        "gene": "TSC1",
        "protein": (
            "TSC1 -- 9q34.13 Autosomal-Dominant-LOF -- 1164aa -- "
            "Hamartin-130kDa-TSC1-TSC2-mTORC1-SEGA-PATHOGNOMONIC-"
            "Cortical-Tubers-PATHOGNOMONIC-Everolimus-FDA-AML-LAM-OMIM-191100"
        ),
        "locus": "9q34.13",
        "protein_size": (
            "1164 aa / 130 kDa / 9q34.13 TSC1 encodes Hamartin (TSC1-TSC2 complex subunit): "
            "STRUCTURE: "
            "  1164 aa / 130 kDa; scaffold protein; no enzymatic activity; "
            "  Coiled-coil domain (aa 730-1164): TSC2 (tuberin) interaction; stabilises TSC2; "
            "  EZRIN-RADIXIN-MOESIN-binding domain: cytoskeletal anchoring; "
            "  TSC1-TSC2 heterodimerises to form TSC (tuberous sclerosis complex): "
            "    TSC2 GTPase-activating protein (GAP) towards RHEB; "
            "    TSC1 stabilises TSC2 (prevents ubiquitination); "
            "  TSC1 LOF -> TSC complex destabilisation -> RHEB-GTP accumulates -> mTORC1 constitutive; "
            "  mTORC1: phosphorylates S6K1 + 4E-BP1 -> protein synthesis, autophagy suppression; "
            "  TSC1 null -> mTORC1 hyperactivation -> hamartoma (benign, highly proliferative); "
            "TUBEROUS SCLEROSIS COMPLEX (TSC1): "
            "  OMIM 191100 (TSC1); OMIM 613254 (TSC2); AD LOF; 1/6,000-1/10,000 births; "
            "  TSC1 generally milder than TSC2 (same clinical spectrum, different severity); "
            "  Cortical tubers: white matter lesions (focal cortical dysplasia) PATHOGNOMONIC; present at birth; "
            "  Cortical tubers: epilepsy 85% (infantile spasms -> focal seizures); "
            "  Subependymal nodules (SEN): calcified nodules at lateral ventricle walls; "
            "  SEGA (subependymal giant cell astrocytoma): 5-20% TSC PATHOGNOMONIC (grows from SEN); "
            "CNS TUMOUR PREDISPOSITION (TSC1): "
            "  SEGA: WHO grade I; located at foramen of Monro; causes obstructive hydrocephalus; "
            "  SEGA growth criterion: >1cm or growing on serial MRI = treatment indication; "
            "  SEGA before everolimus: surgical resection (risk: hydrocephalus); "
            "  SEGA after everolimus: 35% SEGA volume reduction; hydrocephalus avoidance; "
            "  Cortical tubers epilepsy: surgery (tubectomy) if pharmacoresistant; "
            "  Thalamic tubers: risk of acute neurological deterioration; "
            "EVEROLIMUS (mTOR INHIBITOR -- FDA-APPROVED TSC): "
            "  Everolimus (RAD001/Afinitor): oral mTOR (mTORC1) inhibitor; rapamycin analogue; "
            "  FDA-approved (TSC): SEGA (size >1cm or growing); renal AML >3cm; LAM; "
            "  EXIST-1 trial (SEGA): 35% volume reduction response; tumour size stabilisation 85%; "
            "  EXIST-2 trial (renal AML): 42% response rate; "
            "  Cardiac rhabdomyoma: regress spontaneously by age 2yr (mTOR therapy reserve for obstructive); "
            "PATHOGNOMONIC FEATURES (TSC): "
            "  Cardiac rhabdomyoma (neonatal): PATHOGNOMONIC TSC; "
            "  Shagreen patch (leathery connective tissue naevus): PATHOGNOMONIC (50-80%); "
            "  Ash-leaf spots (hypomelanotic macules): present 90% early life; "
            "  Facial angiofibromas: adult onset; adenoma sebaceum (historical misnomer); "
            "  Ungual/periungual fibromas: Koenen tumours; PATHOGNOMONIC; "
            "  Pulmonary LAM: TSC2-predominant; rare in TSC1; "
            "SURVEILLANCE (TSC1): "
            "  MRI brain at diagnosis + every 1-3yr until age 25yr (SEGA watch); "
            "  Annual abdominal MRI (renal AML); "
            "  Annual echocardiogram if cardiac rhabdomyoma (infants); "
            "  Annual ophthalmology (retinal hamartomas); "
            "  EEG if seizures; "
            "  Neuropsychological testing 3-yearly (cognitive, ASD, ADHD)"
        ),
        "inheritance": "AD LOF; OMIM 191100 (TSC1); 1/6,000-1/10,000 births; 2/3 de novo; TSC1 generally milder than TSC2; two-hit Knudson model (second somatic hit for hamartoma formation); TSC2 more severe cognitive/epilepsy burden",
        "cancer_risk": "SEGA 5-20% PATHOGNOMONIC (grade I); cortical tubers epilepsy 85%; renal AML near-universal (80%); renal cell carcinoma rare but elevated; pulmonary LAM TSC2 > TSC1; cardiac rhabdomyoma neonatal PATHOGNOMONIC",
        "pathognomonic": "SEGA PATHOGNOMONIC (grows from SEN at foramen of Monro); cortical tubers PATHOGNOMONIC (epilepsy 85%); cardiac rhabdomyoma neonatal PATHOGNOMONIC; shagreen patch PATHOGNOMONIC; ungual fibromas (Koenen) PATHOGNOMONIC",
        "surveillance_key": "MRI brain every 1-3yr until age 25yr (SEGA watch); everolimus FDA-approved SEGA >1cm; annual abdominal MRI (renal AML); shagreen patch + cardiac rhabdomyoma neonatal = TSC diagnostic; TSC1 milder than TSC2",
        "key_distinctions": [
            "SEGA-PATHOGNOMONIC-5-20PCT-TSC-FORAMEN-MONRO",
            "CORTICAL-TUBERS-PATHOGNOMONIC-EPILEPSY-85PCT",
            "EVEROLIMUS-FDA-SEGA-AML-LAM-mTOR",
            "CARDIAC-RHABDOMYOMA-NEONATAL-PATHOGNOMONIC",
            "SHAGREEN-PATCH-PATHOGNOMONIC",
            "TSC1-MILDER-THAN-TSC2",
        ],
    },
    {
        "gene": "PTEN",
        "protein": (
            "PTEN -- 10q23.31 Autosomal-Dominant-LOF -- 403aa -- "
            "PTEN-47kDa-PI3K-Phosphatase-Cowden-PHTS-LDD-PATHOGNOMONIC-"
            "Macrocephaly-PATHOGNOMONIC-Breast-85pct-Everolimus-mTOR-OMIM-158350"
        ),
        "locus": "10q23.31",
        "protein_size": (
            "403 aa / 47 kDa / 10q23.31 PTEN encodes PTEN (phosphatase and tensin homolog; PI3K antagonist): "
            "STRUCTURE: "
            "  403 aa / 47 kDa; dual-specificity phosphatase (protein + lipid phosphatase); "
            "  N-terminal PBD domain (aa 6-15): PIP2 binding; membrane targeting; "
            "  Phosphatase domain (aa 14-185): catalytic; C124 active-site cysteine; "
            "  PTEN catalysis: PIP3 -> PIP2 (dephosphorylation; opposes PI3K reaction); "
            "  C2 domain (aa 186-351): membrane association; phospholipid binding; "
            "  C-terminal tail (aa 352-403): PDZ-binding (MAGI1/2/3); stability regulation; "
            "  PTEN nuclear functions: chromosomal stability, DNA repair (HR), centromere regulation; "
            "  PTEN LOF -> PIP3 accumulates -> AKT constitutive -> mTORC1 -> growth/survival; "
            "COWDEN SYNDROME / PHTS (PTEN HAMARTOMA TUMOUR SYNDROME): "
            "  OMIM 158350 (Cowden); AD LOF; 1/200,000-1/250,000 births; 10-44% de novo; "
            "  PHTS = umbrella term: Cowden + Bannayan-Riley-Ruvalcaba + Proteus-like; "
            "  Macrocephaly: 90%+ OFC >97th percentile PATHOGNOMONIC; "
            "  Mucocutaneous lesions: trichilemmomas PATHOGNOMONIC; papillomatous papules; oral papillomas; "
            "  ASD/intellectual disability: 20-30% PTEN germline; macrocephaly + ASD -> PTEN screen MANDATORY; "
            "CNS TUMOUR PREDISPOSITION (PTEN): "
            "  Lhermitte-Duclos disease (LDD): dysplastic cerebellar gangliocytoma PATHOGNOMONIC; "
            "  LDD in adult = PTEN mutation until proven otherwise (90% adult LDD = PTEN germline); "
            "  LDD clinical: progressive cerebellar ataxia, raised ICP; tiger-stripe MRI PATHOGNOMONIC; "
            "  LDD: MRI tiger-stripe cerebellar cortex pattern = PATHOGNOMONIC; "
            "  Glioma: 2-4x elevated (non-LDD); "
            "  Meningioma: 2-3x elevated; "
            "DOMINANT CANCER SPECTRUM (PTEN): "
            "  Breast: 85% lifetime HIGHEST (PTEN breast = BRCA1-level risk); "
            "  Endometrial: 28% lifetime; "
            "  Thyroid follicular: 35% (not papillary); "
            "  Colorectal: 2-3x elevated; "
            "  Renal cell carcinoma (chromophobe): emerging; "
            "EVEROLIMUS / mTOR (PTEN): "
            "  Everolimus (mTOR inhibitor): suppresses PTEN-LOF-driven mTORC1; "
            "  AML (renal): FDA-approved; use in PTEN-related hamartomas investigational; "
            "  LDD: surgery (posterior fossa decompression) = definitive; no pharmacological RCT; "
            "SURVEILLANCE (PTEN PHTS): "
            "  Annual breast MRI from age 30yr (or 5-10yr before earliest family case); "
            "  Annual endometrial ultrasound from age 30-35yr; "
            "  Annual thyroid ultrasound; "
            "  Annual dermatology (mucocutaneous); "
            "  MRI brain at diagnosis (LDD baseline); annual if LDD found; "
            "  Macrocephaly + ASD in child: PTEN germline testing MANDATORY"
        ),
        "inheritance": "AD LOF; OMIM 158350 (Cowden/PHTS); 1/200,000-1/250,000 births; 10-44% de novo; variable expressivity; PTEN nuclear localisation signal variants = milder phenotype; germline vs somatic mosaicism affects penetrance",
        "cancer_risk": "Breast 85% HIGHEST; endometrial 28%; thyroid follicular 35%; LDD (dysplastic cerebellar gangliocytoma) PATHOGNOMONIC; glioma 2-4x; meningioma 2-3x; macrocephaly 90% PATHOGNOMONIC; ASD/ID 20-30%",
        "pathognomonic": "Lhermitte-Duclos disease PATHOGNOMONIC adult (tiger-stripe MRI); macrocephaly OFC >97th %ile PATHOGNOMONIC (90%); trichilemmomas PATHOGNOMONIC mucocutaneous; macrocephaly + ASD = PTEN screen MANDATORY",
        "surveillance_key": "Annual breast MRI from 30yr; annual endometrial US from 30-35yr; LDD surgery (posterior fossa decompression); macrocephaly + ASD = PTEN germline MANDATORY; adult LDD = PTEN mutation until proven otherwise",
        "key_distinctions": [
            "LDD-PATHOGNOMONIC-ADULT-90PCT-PTEN",
            "TIGER-STRIPE-CEREBELLAR-MRI-PATHOGNOMONIC-LDD",
            "MACROCEPHALY-90PCT-PATHOGNOMONIC",
            "MACROCEPHALY-PLUS-ASD-PTEN-SCREEN-MANDATORY",
            "BREAST-85PCT-HIGHEST-PTEN",
            "TRICHILEMMOMAS-PATHOGNOMONIC-MUCOCUTANEOUS",
        ],
    },
    {
        "gene": "SMARCB1",
        "protein": (
            "SMARCB1 -- 22q11.23 Autosomal-Dominant-LOF -- 385aa -- "
            "INI1-BAF47-44kDa-SWI-SNF-ATRT-Under3yr-PATHOGNOMONIC-"
            "INI1-IHC-Loss-PATHOGNOMONIC-Tazemetostat-EZH2i-FDA2020-OMIM-609322"
        ),
        "locus": "22q11.23",
        "protein_size": (
            "385 aa / 44 kDa / 22q11.23 SMARCB1 encodes INI1/BAF47 (SWI/SNF chromatin remodeling complex core subunit): "
            "STRUCTURE: "
            "  385 aa / 44 kDa; core structural subunit of SWI/SNF (BAF) complex; "
            "  N-terminal repeat (Rpt1, aa 1-50) and (Rpt2, aa 51-100): evolutionarily conserved; "
            "  Homology region (aa 100-165): contacts BAF155/BAF170 (SMARCC1/2); "
            "  C-terminal coiled-coil (aa 300-385): complex assembly; transcriptional activation; "
            "  SMARCB1 is ALWAYS present in canonical BAF complex; "
            "  SWI/SNF uses ATPase (SMARCA4/SMARCA2) to remodel nucleosomes; "
            "  SMARCB1 null -> SWI/SNF complex destabilised -> chromatin closes -> tumour suppressor silencing; "
            "  SMARCB1 null -> EZH2 (PRC2 methyltransferase) unopposed -> H3K27me3 accumulation -> gene silencing; "
            "  EZH2 inhibition (tazemetostat): restores tumour suppressor gene expression in SMARCB1-null; "
            "RHABDOID TUMOUR PREDISPOSITION SYNDROME 2 (RTPS2): "
            "  OMIM 609322; AD LOF germline SMARCB1; ~100 families worldwide 2026; "
            "  RTPS2 vs RTPS1 (SMARCA4 germline -- different gene, same syndrome, brain-specific); "
            "  ATRT (atypical teratoid/rhabdoid tumour): most aggressive childhood brain tumour; "
            "  ATRT: WHO grade IV; median survival < 2yr; "
            "  SMARCB1 germline in 25-35% of all ATRT patients; "
            "  ATRT predominantly < age 3yr PATHOGNOMONIC; bilateral/multifocal ATRT = SMARCB1 germline; "
            "CNS TUMOUR (SMARCB1): "
            "  ATRT subtypes: ATRT-TYR, ATRT-SHH, ATRT-MYC (molecular classification); "
            "  INI1 IHC loss: nuclear loss in ATRT/MRT cells PATHOGNOMONIC (100% sensitivity for SMARCB1-null); "
            "  MRT (malignant rhabdoid tumour) extracranial: kidney (RTK), soft tissue; "
            "  Epithelioid sarcoma: SMARCB1-null (10-12% germline); "
            "  Schwannomatosis (NF3-like): germline SMARCB1 LOF (peripheral nerve sheath); "
            "TAZEMETOSTAT (EZH2 INHIBITOR -- FDA2020): "
            "  Tazemetostat (EPZ-6438): oral EZH2 methyltransferase inhibitor; "
            "  FDA2020: epithelioid sarcoma (SMARCB1-null); FDA2020: EZH2-mutant follicular lymphoma; "
            "  ATRT: tazemetostat + multi-agent regimen trials ongoing; "
            "  Mechanism: inhibits EZH2 -> reduces H3K27me3 -> re-expresses tumour suppressors in SMARCB1-null; "
            "ATRT TREATMENT: "
            "  Multi-agent chemotherapy: ICE (ifosfamide, carboplatin, etoposide) based; "
            "  Intrathecal chemotherapy (marizomib, methotrexate): leptomeningeal disease; "
            "  HSCT consolidation: autologous HSCT after intensive induction; "
            "  Radiation: SMARCB1 germline = debate (LFS-like secondary risk younger children); "
            "  Radiation in ATRT: considered >3yr; craniospinal avoided < 3yr germline; "
            "SURVEILLANCE (SMARCB1): "
            "  MRI brain + full spine every 3 months (first 2yr post-ATRT); "
            "  Annual MRI brain + spine (germline carrier screening relatives); "
            "  Annual abdominal US (MRT kidney); "
            "  Cascade: parents + siblings (50% risk AD germline)"
        ),
        "inheritance": "AD LOF germline; OMIM 609322 (RTPS2); ~100 families worldwide 2026; 50% de novo in ATRT setting; SMARCB1 biallelic somatic = sporadic ATRT/MRT (no germline); germline in 25-35% of all ATRT; mosaic germline described",
        "cancer_risk": "ATRT (WHO grade IV) under 3yr PATHOGNOMONIC (25-35% ATRT = germline SMARCB1); MRT extracranial (kidney/soft tissue); epithelioid sarcoma; schwannomatosis; bilateral/multifocal ATRT = SMARCB1 germline",
        "pathognomonic": "ATRT under 3yr PATHOGNOMONIC; bilateral/multifocal ATRT PATHOGNOMONIC SMARCB1 germline; INI1 IHC nuclear loss PATHOGNOMONIC (100% sensitivity); tazemetostat EZH2i FDA2020 SMARCB1-null; multi-agent ICE + HSCT consolidation",
        "surveillance_key": "MRI brain + spine q3mo first 2yr post-ATRT; tazemetostat EZH2i FDA2020 epithelioid sarcoma; ICE-based + intrathecal + HSCT ATRT; INI1 IHC loss diagnostic; radiation debate in germline SMARCB1 young children",
        "key_distinctions": [
            "ATRT-UNDER-3YR-PATHOGNOMONIC",
            "INI1-IHC-NUCLEAR-LOSS-PATHOGNOMONIC-100PCT",
            "BILATERAL-MULTIFOCAL-ATRT-SMARCB1-GERMLINE",
            "TAZEMETOSTAT-EZH2i-FDA2020-SMARCB1-NULL",
            "SMARCB1-GERMLINE-25-35PCT-ALL-ATRT",
            "ICE-INTRATHECAL-HSCT-CONSOLIDATION-ATRT",
        ],
    },
    {
        "gene": "SUFU",
        "protein": (
            "SUFU -- 10q24.32 Autosomal-Dominant-LOF -- 484aa -- "
            "SUFU-54kDa-Suppressor-Fused-SHH-Pathway-Medulloblastoma-SHH-"
            "50-60pct-HIGHEST-Desmoplastic-Nodular-MB-PATHOGNOMONIC-OMIM-607035"
        ),
        "locus": "10q24.32",
        "protein_size": (
            "484 aa / 54 kDa / 10q24.32 SUFU encodes Suppressor of Fused (SHH pathway negative regulator): "
            "STRUCTURE: "
            "  484 aa / 54 kDa; cytoplasmic scaffold; no enzymatic activity; "
            "  N-terminal SUFU domain (aa 1-250): GLI binding; sequesters GLI in cytoplasm; "
            "  Middle domain (aa 251-380): SUFU self-dimerisation; "
            "  C-terminal domain (aa 381-484): regulates GLI1/2/3 nuclear import; "
            "  SUFU binds GLI1/2/3 transcription factors -> cytoplasmic retention -> no SHH target gene activation; "
            "  SHH binding to PTCH1 -> SMO activation -> SUFU release of GLI -> nuclear GLI -> SHH targets; "
            "  SUFU LOF -> GLI1/2 constitutively nuclear -> SHH target genes (PTCH1, HHIP, GLI1, CCND1) active; "
            "  SHH pathway: PTCH1 (Gorlin BCC-dominant) -- SUFU (MB-dominant) -- SMO (drug target); "
            "SUFU vs PTCH1 DISTINCTION: "
            "  PTCH1 germline (Gorlin syndrome): BCC 90% dominant; MB 5-10%; "
            "  SUFU germline: MB 50-60% dominant HIGHEST; BCC 5-10% (much less than PTCH1); "
            "  SUFU = medulloblastoma-dominant; PTCH1 = BCC-dominant; "
            "  SUFU MB germline: most common germline MB predisposition gene (SHH subgroup); "
            "  Adult SHH-MB: SUFU or PTCH1 germline testing MANDATORY in all adult SHH-MB; "
            "MEDULLOBLASTOMA SHH SUBTYPE (SUFU): "
            "  OMIM 607035; AD LOF; rare; "
            "  Medulloblastoma SHH: 25-30% of all MB; germline most common in SHH-MB; "
            "  SUFU germline: MB lifetime risk 50-60% HIGHEST of all SHH pathway germline genes; "
            "  Desmoplastic/nodular MB (DNMB): PATHOGNOMONIC SUFU germline; lateral cerebellar; "
            "  Infant onset: median age 2yr in SUFU MB (younger than PTCH1 or SMO-MB); "
            "  SUFU MB: good prognosis in infants (standard risk DNM-MB; CSI can often be avoided); "
            "HEDGEHOG INHIBITORS (SUFU MB): "
            "  Vismodegib (GDC-0449): SMO inhibitor; FDA2012 BCC; active in SHH-MB; "
            "  Sonidegib (LDE225): SMO inhibitor; FDA2015 BCC; SHH-MB active; "
            "  CRITICAL: AVOID vismodegib/sonidegib in children with developing skeleton (growth plate closure PERMANENT); "
            "  Children: SMO inhibitors cause premature growth plate fusion -> short stature permanent; "
            "  SMO inhibitor use in children: investigational only; bridge to HSCT or short-course; "
            "  Adult SHH-MB: vismodegib/sonidegib clinically active; "
            "  AVOID radiation young children (SUFU): secondary malignancy risk + neurocognitive; "
            "GNAS/APC CO-MUTATIONS: "
            "  GNAS/APC somatic co-mutations in sporadic SHH-MB (not germline); "
            "  SUFU germline loss -> second somatic hit required for MB (Knudson); "
            "SURVEILLANCE (SUFU): "
            "  MRI brain + spine at diagnosis; every 3 months post-MB for 2yr; "
            "  Annual MRI surveillance (germline carrier without MB: from age 1yr to 5yr); "
            "  Annual skin examination from age 20yr (BCC, though rare in SUFU vs PTCH1); "
            "  Adult SHH-MB: SUFU + PTCH1 germline testing MANDATORY (all adult SHH-MB); "
            "  Cascade: first-degree relatives SUFU testing"
        ),
        "inheritance": "AD LOF; OMIM 607035; rare; 50-60% MB lifetime HIGHEST SHH-pathway germline predisposition; infant-onset median 2yr; second somatic hit required (Knudson two-hit); SMO activation most common somatic sporadic SHH-MB",
        "cancer_risk": "MB SHH subtype 50-60% lifetime HIGHEST germline risk; desmoplastic/nodular MB PATHOGNOMONIC; infant median 2yr onset; BCC 5-10% (much less than PTCH1-Gorlin); adult SHH-MB: SUFU/PTCH1 testing MANDATORY",
        "pathognomonic": "Desmoplastic/nodular MB (DNMB) PATHOGNOMONIC SUFU germline; MB 50-60% HIGHEST SHH germline risk; SUFU = MB-dominant (vs PTCH1 = BCC-dominant); AVOID vismodegib/sonidegib developing skeleton (permanent growth plate)",
        "surveillance_key": "Annual MRI from age 1yr to 5yr (germline carrier); adult SHH-MB SUFU+PTCH1 germline testing MANDATORY; AVOID vismodegib/sonidegib children (growth plate fusion permanent); AVOID radiation young children; desmoplastic-nodular MB good prognosis infants",
        "key_distinctions": [
            "MB-SHH-50-60PCT-HIGHEST-SUFU-GERMLINE",
            "DESMOPLASTIC-NODULAR-MB-PATHOGNOMONIC-SUFU",
            "SUFU-MB-DOMINANT-VS-PTCH1-BCC-DOMINANT",
            "AVOID-VISMODEGIB-CHILDREN-GROWTH-PLATE-PERMANENT",
            "ADULT-SHH-MB-SUFU-PTCH1-TESTING-MANDATORY",
            "INFANT-ONSET-MEDIAN-2YR-SUFU",
        ],
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-DIPG-H3K27M-PATHOGNOMONIC-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-ONC201-FDA2022-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes Tumour Protein p53 (Guardian of the Genome): "
            "STRUCTURE: "
            "  393 aa / 43 kDa; tetrameric transcription factor; "
            "  N-terminal transactivation domain (TAD1: aa 1-40; TAD2: aa 40-67); "
            "  Proline-rich domain (PRD, aa 67-98); "
            "  DBD (DNA-binding domain): aa 94-292 -- ALL hotspot residues (R175, G245, R248, R249, R273, R282); "
            "  Tetramerisation domain (TET, aa 325-356): 4 subunits form active tetramer; "
            "  C-terminal regulatory domain (CTD, aa 356-393): acetylation K372, K373, K382; "
            "  p53 activates: p21 (G1 arrest), PUMA, NOXA, BAX (apoptosis), MDM2 (autoregulation); "
            "  MDM2 ubiquitinates p53 -> proteasomal degradation (negative feedback); "
            "  GOF hotspot mutations: dominant-negative tetramer poisoning by mutant subunit; "
            "LI-FRAUMENI SYNDROME (LFS) -- CNS SPECTRUM: "
            "  OMIM 151623; AD LOF; germline TP53 pathogenic; "
            "  LFS CNS tumours: DIPG 20-30% of paediatric DIPG = TP53 germline; GBM 2-5%; PNET/MB; "
            "  Classic LFS core: sarcoma 25-30% DOMINANT; breast 25-30%; adrenocortical carcinoma 50-70% paediatric; "
            "  Penetrance: >90% by age 60yr (female); 73% by age 60yr (male); "
            "DIPG / DIFFUSE MIDLINE GLIOMA (TP53 LFS): "
            "  DIPG (diffuse intrinsic pontine glioma): pontine; H3K27M IHC PATHOGNOMONIC; "
            "  H3K27M mutation (H3F3A K27M or HIST1H3B K27M): diffuse midline glioma grade IV; "
            "  TP53 germline in 20-30% of paediatric DIPG patients; "
            "  H3K27M + TP53 germline: most common germline context for DIPG; "
            "  ONC201 (dordaviprone): DRD2/3 antagonist + CLpP agonist; FDA2022 H3K27M+ diffuse midline glioma; "
            "  ONC201 STELLAR trial: 36% ORR H3K27M+ DIPG; 9.5 months median OS; "
            "  Radiation: standard-of-care for DIPG (palliative); HOWEVER: LFS = AVOID RADIATION ABSOLUTELY; "
            "  RADIATION DILEMMA IN LFS DIPG: radiation standard-of-care vs. ABSOLUTE CI in LFS; clinical judgment; "
            "AVOID RADIATION ABSOLUTELY (LFS CNS): "
            "  TP53 LOF -> defective p53-mediated apoptosis of radiation-damaged cells; "
            "  Radiation -> secondary malignancy (osteosarcoma, angiosarcoma, secondary GBM) in LFS field; "
            "  WBMRI (whole-body MRI) annually: Toronto Protocol; replaces CT/PET; "
            "  Brain MRI 6-monthly (age <35yr); annual thereafter; "
            "  NO ionising imaging (CT, PET) for surveillance in LFS; "
            "TORONTO PROTOCOL (WBMRI LFS): "
            "  Annual WBMRI: skull base to proximal femur; "
            "  6-monthly brain MRI; "
            "  Annual breast MRI (no mammography); annual abdominal US; "
            "  Annual dermatology (soft tissue sarcoma surface); "
            "  R337H Brazilian founder: ~0.3% carrier frequency in southern Brazil; "
            "  NO chemotherapy de-escalation in TP53 germline CNS tumours; "
            "SURVEILLANCE (TP53 LFS CNS): "
            "  Annual WBMRI + 6-monthly brain MRI; "
            "  Annual FBC + LDH (sarcoma/leukaemia); "
            "  Annual adrenal US (ACC surveillance from birth); "
            "  Breast MRI from age 20yr (no mammography); "
            "  Cascade: all first-degree relatives TP53 germline testing"
        ),
        "inheritance": "AD LOF; OMIM 151623 (LFS); penetrance >90% (female) by age 60yr; R337H Brazilian founder 0.3% southern Brazil; de novo ~20%; hotspot GOF mutations (dominant-negative) worst prognosis; no chemotherapy de-escalation TP53 CNS",
        "cancer_risk": "DIPG 20-30% of paediatric DIPG = TP53 germline PATHOGNOMONIC; GBM 2-5%; sarcoma 25-30% DOMINANT; breast 25-30%; adrenocortical carcinoma 50-70% paediatric; PNET/MB elevated; penetrance >90% female by 60yr",
        "pathognomonic": "H3K27M IHC PATHOGNOMONIC DIPG/diffuse midline glioma; AVOID RADIATION ABSOLUTELY (secondary sarcoma/GBM); WBMRI Toronto Protocol MANDATORY; ONC201/dordaviprone FDA2022 H3K27M+; R337H Brazilian founder 0.3%",
        "surveillance_key": "Annual WBMRI Toronto Protocol; 6-monthly brain MRI; NO CT/PET (ionising); ONC201 FDA2022 H3K27M+ DIPG; radiation dilemma in LFS DIPG (clinical judgment); no chemotherapy de-escalation; R337H Brazilian founder; breast MRI 20yr",
        "key_distinctions": [
            "DIPG-H3K27M-PATHOGNOMONIC-LFS-20-30PCT-PEDIATRIC-DIPG",
            "AVOID-RADIATION-ABSOLUTELY-LFS-CNS",
            "ONC201-DORDAVIPRONE-FDA2022-H3K27M-PLUS",
            "WBMRI-TORONTO-PROTOCOL-MANDATORY-NO-CT-PET",
            "RADIATION-DILEMMA-LFS-DIPG-CLINICAL-JUDGMENT",
            "R337H-BRAZILIAN-FOUNDER-0-3PCT",
        ],
    },
]

# --------------------------------------------------------------------------- #
# Simulated patient cohort
# --------------------------------------------------------------------------- #

GENE_AGE_PARAMS = {
    "NF1":     {"mean_age": 14, "sd": 8},
    "NF2":     {"mean_age": 27, "sd": 9},
    "VHL":     {"mean_age": 35, "sd": 10},
    "TSC1":    {"mean_age": 10, "sd": 6},
    "PTEN":    {"mean_age": 38, "sd": 12},
    "SMARCB1": {"mean_age": 2,  "sd": 1},
    "SUFU":    {"mean_age": 3,  "sd": 2},
    "TP53":    {"mean_age": 9,  "sd": 5},
}

PATHOGENIC_VARIANTS = {
    "NF1":     ["c.2542G>A p.Gly848Arg", "c.1849C>T p.Arg617Ter", "c.4537C>T p.Arg1513Ter",
                "c.6791_6792delAA p.Lys2264ArgfsTer10", "c.3721C>T p.Arg1241Ter"],
    "NF2":     ["c.784C>T p.Arg262Ter", "c.109C>T p.Arg37Ter", "c.519_523delATCTG p.Ile174SerFsTer9",
                "c.1022delA p.Tyr341Ter", "c.880-1G>A p.splicing"],
    "VHL":     ["c.500T>A p.Val167Asp", "c.499G>T p.Val167Leu", "c.292C>T p.Arg98Ter",
                "c.482G>T p.Arg161Ter", "c.220C>T p.Arg74Cys"],
    "TSC1":    ["c.1105C>T p.Arg369Ter", "c.689_692delAGAG p.Glu230GlyFsTer14",
                "c.2143C>T p.Arg715Ter", "c.1907_1908delAA p.Lys636ArgFsTer11",
                "c.430C>T p.Arg144Ter"],
    "PTEN":    ["c.388C>T p.Arg130Ter", "c.697C>T p.Arg233Ter", "c.800delA p.Asn267ThrFsTer18",
                "c.323T>G p.Leu108Arg", "c.634C>T p.Arg212Ter"],
    "SMARCB1": ["c.601C>T p.Arg201Ter", "c.472C>T p.Arg158Ter", "c.1_2delATG p.Met1fsTer",
                "c.157C>T p.Arg53Ter", "c.1007G>A p.Arg336His"],
    "SUFU":    ["c.1022C>T p.Ser341Phe", "c.530G>A p.Trp177Ter", "c.1408C>T p.Arg470Cys",
                "c.251+1G>A p.splicing", "c.1081C>T p.Gln361Ter"],
    "TP53":    ["c.817C>T p.Arg273Cys (R273C)", "c.742C>T p.Arg248Trp (R248W)",
                "c.524G>A p.Arg175His (R175H)", "c.1009C>T p.Arg337His (R337H Brazilian)",
                "c.733G>A p.Gly245Ser (G245S)"],
}

TUMOR_TYPES = {
    "NF1":     ["Optic Glioma Grade II (OPG)", "Low-Grade Astrocytoma (LGA)",
                "MPNST (Malignant Peripheral Nerve Sheath Tumour)", "GBM (Glioblastoma)", "Plexiform Neurofibroma"],
    "NF2":     ["Vestibular Schwannoma (bilateral)", "Meningioma (intracranial)", "Spinal Ependymoma",
                "Unilateral Vestibular Schwannoma", "Multiple Meningiomas"],
    "VHL":     ["Cerebellar Hemangioblastoma", "Spinal Hemangioblastoma", "Retinal Hemangioblastoma",
                "ccRCC (Clear Cell Renal Cell Carcinoma)", "ELST (Endolymphatic Sac Tumour)"],
    "TSC1":    ["SEGA (Subependymal Giant Cell Astrocytoma)", "Cortical Tuber (Epilepsy-related)",
                "Renal AML (Angiomyolipoma)", "Cardiac Rhabdomyoma", "SEN (Subependymal Nodule)"],
    "PTEN":    ["Lhermitte-Duclos Disease (LDD)", "Glioma (PTEN-associated)", "Breast Cancer (PHTS)",
                "Endometrial Carcinoma", "Meningioma (PHTS)"],
    "SMARCB1": ["ATRT (Atypical Teratoid/Rhabdoid Tumour)", "MRT (Malignant Rhabdoid Tumour, renal)",
                "Epithelioid Sarcoma (SMARCB1-null)", "Schwannomatosis", "MRT (soft tissue)"],
    "SUFU":    ["Medulloblastoma SHH (Desmoplastic/Nodular)", "Medulloblastoma SHH (Classic)",
                "Medulloblastoma SHH (infant, extensive nodularity)", "BCC (Basal Cell Carcinoma)",
                "Desmoplastic Medulloblastoma (cerebellar hemisphere)"],
    "TP53":    ["DIPG (Diffuse Intrinsic Pontine Glioma, H3K27M+)", "GBM (LFS-associated)",
                "Diffuse Midline Glioma H3K27M+", "PNET/Medulloblastoma (LFS)",
                "Adrenocortical Carcinoma (paediatric)"],
}

TREATMENT_PROTOCOLS = {
    "NF1":     ["Selumetinib MEKi (optic glioma/plexiform NF)", "Carboplatin + vincristine (OPG first-line child)",
                "Everolimus mTOR (plexiform NF refractory)", "Surgery (MPNST resection wide margin)",
                "Surveillance only (asymptomatic OPG)"],
    "NF2":     ["Bevacizumab IV (VS hearing preservation)", "Surgery: VS resection (hearing-priority failure)",
                "Radiosurgery GK (small-medium VS)", "Aspirin 325mg/day (emerging anti-tumour)",
                "FAK inhibitor defactinib (meningioma trials)"],
    "VHL":     ["Belzutifan HIF-2α inhibitor (ccRCC + CNS HGB)", "Surgery: HGB resection (symptomatic)",
                "Sunitinib/pazopanib (ccRCC pre-belzutifan)", "SRS radiosurgery (small inaccessible HGB)",
                "Laser photocoagulation (retinal HGB small)"],
    "TSC1":    ["Everolimus mTOR (SEGA + renal AML)", "Surgery: SEGA resection (obstructive hydrocephalus)",
                "Vigabatrin / antiepileptics (cortical tuber epilepsy)", "Rapamycin (pulmonary LAM)",
                "Surgery: tubectomy (pharmacoresistant epilepsy)"],
    "PTEN":    ["Surgery: posterior fossa decompression (LDD)", "Everolimus mTOR (renal hamartomas)",
                "Prophylactic risk-reducing mastectomy (breast 85%)", "Annual breast MRI surveillance",
                "Hysterectomy (endometrial risk reduction)"],
    "SMARCB1": ["ICE (ifosfamide + carboplatin + etoposide) -- ATRT induction", "Intrathecal methotrexate (ATRT leptomeningeal)",
                "Tazemetostat EZH2i (epithelioid sarcoma FDA2020)", "Autologous HSCT (ATRT consolidation)",
                "Multi-agent + intrathecal + HSCT (full ATRT regimen)"],
    "SUFU":    ["Head-only RT + adjuvant chemo (MB SHH >3yr standard risk)", "Craniospinal irradiation CSI (MB SHH high-risk adult)",
                "Carboplatin + vincristine baby-brain (MB <3yr avoid CSI)", "Vismodegib SMOi (adult SHH-MB only -- NOT children)",
                "Multi-agent chemo + ASCT (MB infant SUFU high-risk)"],
    "TP53":    ["ONC201 dordaviprone (H3K27M+ DIPG FDA2022)", "Focal RT (DIPG palliative -- LFS debate)",
                "Temozolomide (GBM LFS; NO RT preferred)", "Sarcoma-directed chemo (LFS sarcoma dominant)",
                "WBMRI surveillance annual (Toronto Protocol -- NO CT/PET)"],
}

SURVEILLANCE_PROTOCOLS = {
    "NF1":     ["Annual ophthalmology from birth to age 8yr (OPG)", "Annual plexiform NF exam (MPNST growth signal)",
                "Annual blood pressure (renal artery stenosis)", "Annual breast MRI from age 30yr"],
    "NF2":     ["Annual MRI brain + full spine", "Annual audiogram (MANDATORY hearing baseline)",
                "Annual ophthalmology (cataract monitoring)", "Avoid noise trauma"],
    "VHL":     ["Annual MRI brain + spine from age 11yr", "Annual retinal exam from age 1yr",
                "Annual abdominal MRI (ccRCC, pheo, pancreatic) from age 15yr", "Annual urine catecholamines / plasma metanephrines"],
    "TSC1":    ["MRI brain every 1-3yr until age 25yr (SEGA watch)", "Annual abdominal MRI (renal AML)",
                "EEG if seizures present", "Neuropsychological testing 3-yearly (ASD, ADHD)"],
    "PTEN":    ["Annual breast MRI from age 30yr", "Annual endometrial US from age 30-35yr",
                "Annual thyroid US", "MRI brain at diagnosis (LDD baseline)"],
    "SMARCB1": ["MRI brain + spine q3months post-ATRT (2yr)", "Annual abdominal US (MRT renal)",
                "Annual MRI spine (schwannomatosis surveillance)", "Annual FBC (chemotherapy monitoring)"],
    "SUFU":    ["MRI brain + spine q3months post-MB (2yr)", "Annual MRI brain from age 1yr to 5yr (carrier)",
                "Annual skin exam from age 20yr (BCC surveillance)", "Adult SHH-MB: SUFU + PTCH1 germline testing"],
    "TP53":    ["Annual WBMRI Toronto Protocol (NO CT/PET)", "6-monthly brain MRI (age <35yr)",
                "Annual adrenal US from birth (ACC)", "Breast MRI from age 20yr (NO mammography)"],
}


def _make_patients(gene_def: dict, seed: int, n: int = 40) -> list[dict]:
    rng = random.Random(seed)
    params = GENE_AGE_PARAMS[gene_def["gene"]]
    variants = PATHOGENIC_VARIANTS[gene_def["gene"]]
    tumor_list = TUMOR_TYPES[gene_def["gene"]]
    treatments = TREATMENT_PROTOCOLS[gene_def["gene"]]
    surveillance = SURVEILLANCE_PROTOCOLS[gene_def["gene"]]
    patients = []
    for i in range(n):
        age = max(0, int(rng.gauss(params["mean_age"], params["sd"])))
        gender = rng.choice(["M", "F", "F"])
        variant = rng.choice(variants)
        tumor_type = rng.choice(tumor_list)
        response = rng.choice(["CR", "PR", "SD", "PD", "CR (on targeted therapy)"])
        tx = rng.choice(treatments)
        surv = rng.choice(surveillance)
        relapse = rng.random() < 0.30
        resection = rng.choice(["GTR", "STR", "Biopsy", "Surgery-Not-Indicated"])
        # Radiation: False for TP53 (AVOID) and SMARCB1 young infants; varies others
        if gene_def["gene"] == "TP53":
            radiation_received = False
        elif gene_def["gene"] == "SMARCB1":
            radiation_received = age >= 3 and rng.random() < 0.40
        elif gene_def["gene"] == "SUFU":
            radiation_received = age >= 3 and rng.random() < 0.55
        else:
            radiation_received = rng.random() < 0.35
        targeted_therapy = gene_def["gene"] in ("NF1", "NF2", "VHL", "TSC1", "PTEN", "SMARCB1", "SUFU", "TP53") and rng.random() < 0.50
        patients.append({
            "patient_id": f"{gene_def['gene']}-CNS-{seed}-{i+1:03d}",
            "gene": gene_def["gene"],
            "locus": gene_def["locus"],
            "age_dx": age,
            "sex": gender,
            "variant": variant,
            "tumor_type": tumor_type,
            "resection": resection,
            "radiation_received": radiation_received,
            "targeted_therapy": targeted_therapy,
            "treatment": tx,
            "response": response,
            "relapse": relapse,
            "surveillance_note": surv,
            "protein_detail": gene_def["protein"],
            "inheritance": gene_def["inheritance"],
            "cancer_risk": gene_def["cancer_risk"],
            "pathognomonic": gene_def["pathognomonic"],
            "key_distinctions": gene_def["key_distinctions"],
        })
    return patients


def _build_all_patients() -> list[dict]:
    all_pts = []
    for i, gdef in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        all_pts.extend(_make_patients(gdef, seed))
    return all_pts


def _json_safe(obj):
    """Recursively convert non-JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    return str(obj)


# --------------------------------------------------------------------------- #
# Public API functions
# --------------------------------------------------------------------------- #

def generate_overview() -> dict:
    patients = _build_all_patients()
    gene_counts = {}
    for p in patients:
        g = p["gene"]
        gene_counts.setdefault(g, {"total": 0, "cr": 0, "relapse": 0,
                                   "gtr": 0, "radiation": 0, "targeted": 0})
        gene_counts[g]["total"] += 1
        if "CR" in p["response"]:
            gene_counts[g]["cr"] += 1
        if p["relapse"]:
            gene_counts[g]["relapse"] += 1
        if p.get("resection") == "GTR":
            gene_counts[g]["gtr"] += 1
        if p.get("radiation_received"):
            gene_counts[g]["radiation"] += 1
        if p.get("targeted_therapy"):
            gene_counts[g]["targeted"] += 1

    mean_age = sum(p["age_dx"] for p in patients) / len(patients)
    tumor_types_all = [p["tumor_type"] for p in patients]
    top_tumor_types: dict = {}
    for t in tumor_types_all:
        top_tumor_types[t] = top_tumor_types.get(t, 0) + 1
    top_tumor_types = dict(sorted(top_tumor_types.items(), key=lambda x: -x[1])[:10])

    cr_total = sum(1 for p in patients if "CR" in p["response"])
    relapse_total = sum(1 for p in patients if p["relapse"])
    gtr_total = sum(1 for p in patients if p.get("resection") == "GTR")
    radiation_total = sum(1 for p in patients if p.get("radiation_received"))
    targeted_total = sum(1 for p in patients if p.get("targeted_therapy"))

    gene_summaries = []
    for gdef in ATLAS_GENES:
        g = gdef["gene"]
        gc = gene_counts.get(g, {})
        total = gc.get("total", 0)
        gene_summaries.append({
            "gene": g,
            "locus": gdef["locus"],
            "inheritance": gdef["inheritance"][:80] + "...",
            "cancer_risk_summary": gdef["cancer_risk"][:120] + "...",
            "pathognomonic": gdef["pathognomonic"][:120] + "...",
            "n_patients": total,
            "cr_n": gc.get("cr", 0),
            "cr_pct": round(gc.get("cr", 0) / total * 100, 1) if total else 0,
            "relapse_n": gc.get("relapse", 0),
            "relapse_pct": round(gc.get("relapse", 0) / total * 100, 1) if total else 0,
            "gtr_n": gc.get("gtr", 0),
            "radiation_n": gc.get("radiation", 0),
            "targeted_n": gc.get("targeted", 0),
        })

    return _json_safe({
        "atlas": "Hereditary-CNS-Brain-Tumor-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Hereditary CNS & Brain Tumor Predisposition Reference",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": len(patients),
        "mean_age_dx": round(mean_age, 1),
        "cr_total": cr_total,
        "cr_pct": round(cr_total / len(patients) * 100, 1),
        "relapse_total": relapse_total,
        "relapse_pct": round(relapse_total / len(patients) * 100, 1),
        "gtr_total": gtr_total,
        "gtr_pct": round(gtr_total / len(patients) * 100, 1),
        "targeted_therapy_total": targeted_total,
        "targeted_pct": round(targeted_total / len(patients) * 100, 1),
        "radiation_total": radiation_total,
        "radiation_pct": round(radiation_total / len(patients) * 100, 1),
        "top_tumor_types": top_tumor_types,
        "gene_summaries": gene_summaries,
        "clinical_pearls": [
            "NF1: optic glioma grade II PATHOGNOMONIC (15-20%); selumetinib MEKi FDA2020; AVOID radiation (secondary MPNST)",
            "NF2: bilateral VS PATHOGNOMONIC (90-95%); annual audiogram MANDATORY; bevacizumab VEGF for hearing preservation",
            "VHL: hemangioblastoma cerebellar/spinal PATHOGNOMONIC (60-80%); belzutifan HIF-2α FDA2021; VHL type 1 vs 2 genotype-phenotype",
            "TSC1: SEGA PATHOGNOMONIC at foramen of Monro (5-20%); everolimus FDA-approved SEGA; cardiac rhabdomyoma neonatal PATHOGNOMONIC",
            "PTEN: Lhermitte-Duclos (LDD) PATHOGNOMONIC adult = PTEN until proven otherwise; macrocephaly + ASD = PTEN screen MANDATORY",
            "SMARCB1: ATRT under 3yr PATHOGNOMONIC; INI1 IHC nuclear loss PATHOGNOMONIC (100%); tazemetostat EZH2i FDA2020",
            "SUFU: MB SHH 50-60% lifetime HIGHEST; desmoplastic/nodular MB PATHOGNOMONIC; AVOID vismodegib children (growth plate permanent)",
            "TP53 LFS: DIPG H3K27M PATHOGNOMONIC in LFS children; AVOID RADIATION ABSOLUTELY; ONC201 FDA2022 H3K27M+; WBMRI Toronto annual",
        ],
        "key_management_rules": [
            "TP53 LFS: AVOID RADIATION ABSOLUTELY (secondary sarcoma/GBM); WBMRI Toronto annually; no CT/PET surveillance",
            "NF1: AVOID radiation (secondary MPNST risk); selumetinib FDA2020 plexiform NF and OPG",
            "SUFU: AVOID vismodegib/sonidegib in children with developing skeleton (permanent growth plate closure)",
            "VHL: belzutifan FDA2021 for VHL-ccRCC + CNS HGB; annual retinal exam from age 1yr; VHL type genotype-phenotype",
            "TSC1: everolimus FDA SEGA >1cm or growing; surgery if obstructive hydrocephalus; cortical tuber epilepsy 85%",
            "SMARCB1 ATRT: ICE-based + intrathecal + HSCT consolidation; tazemetostat for epithelioid sarcoma; radiation debate <3yr",
        ],
    })


def generate_breakdown() -> dict:
    patients = _build_all_patients()
    gene_data = {}
    for p in patients:
        g = p["gene"]
        gene_data.setdefault(g, [])
        gene_data[g].append(p)

    breakdown = []
    for gdef in ATLAS_GENES:
        g = gdef["gene"]
        pts = gene_data.get(g, [])
        if not pts:
            continue

        ages = [p["age_dx"] for p in pts]
        mean_age = sum(ages) / len(ages)
        cr_n = sum(1 for p in pts if "CR" in p["response"])
        relapse_n = sum(1 for p in pts if p["relapse"])
        gtr_n = sum(1 for p in pts if p.get("resection") == "GTR")
        radiation_n = sum(1 for p in pts if p.get("radiation_received"))
        targeted_n = sum(1 for p in pts if p.get("targeted_therapy"))

        tumor_counts: dict = {}
        for p in pts:
            t = p["tumor_type"]
            tumor_counts[t] = tumor_counts.get(t, 0) + 1

        variant_counts: dict = {}
        for p in pts:
            v = p["variant"]
            variant_counts[v] = variant_counts.get(v, 0) + 1

        tx_counts: dict = {}
        for p in pts:
            t = p["treatment"]
            tx_counts[t] = tx_counts.get(t, 0) + 1

        resection_counts: dict = {}
        for p in pts:
            r = p["resection"]
            resection_counts[r] = resection_counts.get(r, 0) + 1

        breakdown.append({
            "gene": g,
            "locus": gdef["locus"],
            "protein": gdef["protein"],
            "inheritance": gdef["inheritance"],
            "cancer_risk": gdef["cancer_risk"],
            "pathognomonic": gdef["pathognomonic"],
            "surveillance_key": gdef["surveillance_key"],
            "key_distinctions": gdef["key_distinctions"],
            "n_patients": len(pts),
            "mean_age_dx": round(mean_age, 1),
            "min_age_dx": min(ages),
            "max_age_dx": max(ages),
            "cr_n": cr_n,
            "cr_pct": round(cr_n / len(pts) * 100, 1),
            "relapse_n": relapse_n,
            "relapse_pct": round(relapse_n / len(pts) * 100, 1),
            "gtr_n": gtr_n,
            "gtr_pct": round(gtr_n / len(pts) * 100, 1) if len(pts) else 0,
            "radiation_n": radiation_n,
            "radiation_pct": round(radiation_n / len(pts) * 100, 1) if len(pts) else 0,
            "targeted_n": targeted_n,
            "targeted_pct": round(targeted_n / len(pts) * 100, 1) if len(pts) else 0,
            "top_tumor_types": dict(sorted(tumor_counts.items(), key=lambda x: -x[1])[:5]),
            "top_variants": dict(sorted(variant_counts.items(), key=lambda x: -x[1])[:4]),
            "top_treatments": dict(sorted(tx_counts.items(), key=lambda x: -x[1])[:4]),
            "resection_distribution": resection_counts,
            "patients": pts,
        })

    return _json_safe({
        "atlas": "Hereditary-CNS-Brain-Tumor-Predisposition-Atlas",
        "total_genes": len(breakdown),
        "breakdown": breakdown,
    })


def generate_definitions() -> dict:
    definitions = {}
    for gdef in ATLAS_GENES:
        g = gdef["gene"]
        definitions[g] = {
            "gene": g,
            "locus": gdef["locus"],
            "full_protein_detail": gdef["protein_size"],
            "inheritance": gdef["inheritance"],
            "cancer_risk": gdef["cancer_risk"],
            "pathognomonic": gdef["pathognomonic"],
            "surveillance_key": gdef["surveillance_key"],
            "key_distinctions": gdef["key_distinctions"],
            "pathogenic_variants": PATHOGENIC_VARIANTS[g],
            "tumor_types": TUMOR_TYPES[g],
            "treatment_protocols": TREATMENT_PROTOCOLS[g],
            "surveillance_protocols": SURVEILLANCE_PROTOCOLS[g],
        }

    return _json_safe({
        "atlas": "Hereditary-CNS-Brain-Tumor-Predisposition-Atlas",
        "definitions": definitions,
        "key_rules": {
            "TP53_AVOID_RADIATION_CNS": (
                "AVOID radiation ABSOLUTELY in LFS (TP53 germline) -- secondary sarcoma/GBM in radiation field; "
                "WBMRI Toronto Protocol annually (NOT CT/PET); radiation dilemma in DIPG: LFS is ABSOLUTE CI "
                "but focal RT is standard palliation -- clinical judgment required; ONC201 FDA2022 H3K27M+"
            ),
            "NF1_AVOID_RADIATION_MPNST": (
                "AVOID radiation in NF1 tumours -- secondary MPNST risk in radiation field; "
                "selumetinib MEKi FDA2020 for NF1 symptomatic plexiform NF and OPG; "
                "MPNST 8-13% HIGHEST sarcoma risk in NF1 -- wide surgical margin required"
            ),
            "SUFU_VISMODEGIB_CHILDREN": (
                "NEVER use vismodegib/sonidegib Hedgehog inhibitors in children with developing skeleton; "
                "PERMANENT growth plate closure -> short stature irreversible; "
                "adult SHH-MB: SUFU + PTCH1 germline testing MANDATORY; "
                "SUFU = MB-dominant (50-60%) NOT BCC-dominant (distinguishes from PTCH1-Gorlin)"
            ),
            "VHL_BELZUTIFAN_HIF2A": (
                "Belzutifan (HIF-2α inhibitor) FDA2021: VHL-disease ccRCC + CNS hemangioblastoma + pNET; "
                "VHL type 1 (LOF/truncating) = hemangioblastoma + ccRCC (NO pheo); "
                "VHL type 2 (missense) = hemangioblastoma + pheo ± ccRCC; "
                "bilateral ELST = VHL until proven otherwise PATHOGNOMONIC"
            ),
            "TSC1_EVEROLIMUS_SEGA": (
                "Everolimus mTOR FDA-approved: SEGA >1cm or growing (avoid surgery + hydrocephalus risk); "
                "TSC1 generally milder than TSC2; cortical tubers cause epilepsy 85%; "
                "cardiac rhabdomyoma neonatal = PATHOGNOMONIC TSC; shagreen patch PATHOGNOMONIC"
            ),
            "PTEN_LDD_ADULT": (
                "Adult Lhermitte-Duclos disease (dysplastic cerebellar gangliocytoma) = PTEN mutation until proven otherwise (90%); "
                "tiger-stripe MRI PATHOGNOMONIC LDD; posterior fossa decompression surgery = definitive treatment; "
                "macrocephaly OFC >97th percentile + ASD = PTEN germline screening MANDATORY"
            ),
            "SMARCB1_INI1_ATRT": (
                "ATRT under age 3yr + INI1 IHC nuclear loss = SMARCB1 germline testing MANDATORY; "
                "bilateral/multifocal ATRT = SMARCB1 germline until proven otherwise; "
                "tazemetostat EZH2i FDA2020 for SMARCB1-null epithelioid sarcoma; "
                "HSCT consolidation for ATRT (autologous after ICE-based induction)"
            ),
            "NF2_AUDIOGRAM_MANDATORY": (
                "Annual audiogram MANDATORY in NF2 from diagnosis (baseline hearing + serial monitoring); "
                "bevacizumab VEGF inhibitor for VS -- 40-50% volume reduction + hearing preservation 36%; "
                "NF2 LACKS café-au-lait macules (distinguishes from NF1); "
                "juvenile posterior subcapsular cataract 80% NF2 PATHOGNOMONIC"
            ),
        },
        "cascade_testing_rule": (
            "NF1/NF2/VHL/TSC1/PTEN/SUFU/TP53: cascade to all first-degree relatives (AD -- 50% risk per child). "
            "SMARCB1 (AD): first-degree relatives 50% risk; de novo ~50% of ATRT germline. "
            "VHL type: genotype-phenotype cascade -- test for pheo risk in type 2 relatives. "
            "SUFU: adult SHH-MB relatives require SUFU + PTCH1 co-testing (SHH pathway). "
            "TP53 R337H: cascade in southern Brazil (1/300 carrier frequency population screening). "
            "NF2 mosaicism: offspring risk <50% if mosaic proband (requires molecular analysis)."
        ),
    })


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2)[:3000])
    print("\n--- breakdown (first gene) ---")
    bd = generate_breakdown()
    print(json.dumps(bd["breakdown"][0], indent=2)[:2000])
