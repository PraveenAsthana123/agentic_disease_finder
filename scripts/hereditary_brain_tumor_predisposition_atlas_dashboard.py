#!/usr/bin/env python3
"""Hereditary-Brain-Tumor-Predisposition-Atlas -- Complete 8-Gene Reference
TP53   (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome — brain tumors glioblastoma/astrocytoma/DIPG;
         AVOID RADIATION ABSOLUTELY; WB-MRI Toronto protocol;
         seed SEED_BASE+0) .
NF1    (Neurofibromin RAS-GAP; 2839aa; 17q11.2; AD LOF;
         Optic pathway glioma 15-20% PATHOGNOMONIC; LGG/pilocytic astrocytoma;
         café-au-lait macules 6+ PATHOGNOMONIC; Lisch nodules PATHOGNOMONIC;
         selumetinib FDA2020;
         seed SEED_BASE+1) .
NF2    (Merlin/schwannomin; 595aa; 22q12.2; AD LOF;
         Bilateral vestibular schwannoma PATHOGNOMONIC; meningioma; ependymoma;
         Bevacizumab reduces vestibular schwannoma;
         seed SEED_BASE+2) .
VHL    (Von Hippel-Lindau tumour suppressor; 213aa; 3p25.3; AD LOF;
         CNS hemangioblastoma PATHOGNOMONIC; retinal angioma PATHOGNOMONIC;
         cerebellar/spinal hemangioblastoma; Belzutifan FDA2021 HIF-2alpha inhibitor;
         seed SEED_BASE+3) .
PTCH1  (Patched 1; 1447aa; 9q22.32; AD LOF;
         Gorlin/NBCCS — desmoplastic medulloblastoma 5% lifetime;
         BCC 1000s lifetime; odontogenic keratocysts PATHOGNOMONIC;
         AVOID RADIATION ABSOLUTELY (massive BCC);
         seed SEED_BASE+4) .
TSC2   (Tuberin; 1807aa; 16p13.3; AD LOF;
         SEGA (subependymal giant cell astrocytoma) PATHOGNOMONIC;
         cortical tubers PATHOGNOMONIC; subependymal nodules;
         Everolimus FDA2012 SEGA;
         seed SEED_BASE+5) .
PTEN   (Phosphatase and tensin homologue; 403aa; 10q23.31; AD LOF;
         Cowden/PHTS — Lhermitte-Duclos dysplastic cerebellar gangliocytoma PATHOGNOMONIC;
         macrocephaly PATHOGNOMONIC; adult onset;
         seed SEED_BASE+6) .
SUFU   (Suppressor of fused; 484aa; 10q24.32; AD LOF;
         SHH medulloblastoma most penetrant germline SHH gene;
         medulloblastoma 33% lifetime; meningioma elevated;
         SUFU Gorlin-like;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3430-3437)
"""
import random

SEED_BASE = 3430

ATLAS_GENES = [
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-Guardian-of-Genome-"
            "LFS-Li-Fraumeni-Syndrome-Brain-Tumors-GBM-Astrocytoma-DIPG-"
            "AVOID-RADIATION-ABSOLUTELY-WB-MRI-Toronto-Protocol-"
            "OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 brain tumor molecular context: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; N-terminal transactivation domains (TAD1 aa 1-40, TAD2 aa 40-67); "
            "  Proline-rich domain (aa 67-98); DNA-binding domain (DBD aa 94-292) — hotspot mutations; "
            "  Tetramerisation domain (aa 325-356); C-terminal regulatory domain; "
            "  p53 responds to genotoxic stress: activates CDKN1A/p21, MDM2, PUMA, NOXA; "
            "  TP53 LOF → cell cycle arrest and apoptosis abolished → genome instability; "
            "CANCER RISKS (BRAIN TUMOR FOCUS): "
            "  Brain tumors: GBM, astrocytoma, medulloblastoma (WNT/SHH), DIPG (H3K27M TP53 co-mutation); "
            "  LFS spectrum lifetime: breast 54%, sarcoma 30%, brain 20-26%, adrenocortical 5-10%; "
            "  DIPG + TP53 germline: H3K27M mutation + TP53 germline = ultra-high-risk; "
            "  Pediatric LFS brain tumors: DIPG, plexiform astrocytoma, choroid plexus carcinoma; "
            "KEY MANAGEMENT: "
            "  AVOID RADIATION ABSOLUTELY — radiation in LFS carriers → secondary malignancies (sarcoma); "
            "  WB-MRI Toronto protocol: annual whole-body MRI + annual brain MRI MANDATORY; "
            "  Brain MRI frequency: 6-monthly in LFS if prior brain tumor diagnosis; "
            "  Proton therapy or photon-sparing if radiation unavoidable — multidisciplinary decision; "
            "  ONC201/Dordaviprone FDA2022 for H3K27M DIPG (independent of TP53 germline status)"
        ),
        "syndrome": "Li-Fraumeni Syndrome (LFS) — multi-cancer predisposition; brain, breast, sarcoma, ACC, leukemia",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); de novo ~25% LFS; dominant negative R248W/R273H",
        "brain_tumor_risk": "Brain tumors 20-26% lifetime in LFS; GBM/astrocytoma/DIPG/medulloblastoma; DIPG+TP53 ultra-high-risk",
        "pathognomonic": "LFS multi-cancer cluster PATHOGNOMONIC; radiation sensitivity = secondary sarcoma; H3K27M+TP53 DIPG PATHOGNOMONIC",
        "key_avoid": "AVOID RADIATION ABSOLUTELY — radiation in LFS = secondary sarcoma risk massively elevated; WB-MRI Toronto protocol MANDATORY annual",
        "key_rule": "TP53/LFS: AVOID RADIATION ABSOLUTELY. Annual WB-MRI Toronto protocol. Annual brain MRI. H3K27M DIPG + TP53 = ultra-high-risk. ONC201 H3K27M. Cascade testing first-degree relatives.",
        "surveillance": "Annual whole-body MRI Toronto protocol from diagnosis MANDATORY; annual brain MRI; 6-monthly brain MRI if prior brain tumor; annual breast MRI women from age 20yr; annual abdominal US + blood count; cascade TP53 germline first-degree relatives",
        "targeted_rx": "ONC201/Dordaviprone FDA2022 H3K27M-mutant DIPG (independent of germline TP53); temozolomide+bevacizumab GBM; pembrolizumab hypermutant TP53-associated GBM; avoid CCNU/alkylating agents if alternative; proton therapy if radiation unavoidable; ACT-like regimen DIPG+H3K27M+TP53",
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RAS-GAP-Ras-GTPase-Accelerating-Protein-"
            "Optic-Pathway-Glioma-15-20pct-PATHOGNOMONIC-"
            "LGG-Pilocytic-Astrocytoma-Cafe-au-Lait-Macules-6plus-Lisch-Nodules-"
            "Selumetinib-FDA2020-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 brain tumor molecular context: "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; GRD domain (RAS-GAP, aa 1198-1530) — accelerates RAS GTP hydrolysis; "
            "  PH domain; SEC14 domain; ARM repeats; C-terminal PDZ-binding; "
            "  NF1 LOF → RAS-GTP accumulation → MAPK/ERK + PI3K/AKT hyperactivation; "
            "  BRAF fusion (KIAA1549::BRAF) most common driver in NF1-associated LGG; "
            "CANCER RISKS (BRAIN TUMOR FOCUS): "
            "  Optic pathway glioma (OPG): 15-20% NF1 carriers — PATHOGNOMONIC; typically age <10yr; "
            "  Low-grade glioma (LGG)/pilocytic astrocytoma: most common NF1 CNS tumor; "
            "  Brainstem glioma: NF1-associated brainstem glioma = generally indolent vs. sporadic; "
            "  High-grade glioma: rare; NF1-associated HGG occurs but less common than LGG; "
            "  MPNST: 8-13% lifetime PATHOGNOMONIC (sarcoma — doxorubicin, NOT immunotherapy); "
            "KEY MANAGEMENT: "
            "  Annual ophthalmology from age 1yr (optic glioma surveillance) MANDATORY; "
            "  Visual evoked potentials (VEP) annual if OPG detected; "
            "  Selumetinib FDA2020 MEK inhibitor — OPG + plexiform neurofibromas; "
            "  Café-au-lait 6+ macules PATHOGNOMONIC NF1 (>5mm prepubertal, >15mm postpubertal); "
            "  Lisch nodules (iris hamartomas) PATHOGNOMONIC — slit-lamp mandatory"
        ),
        "syndrome": "Neurofibromatosis type 1 (NF1) — optic glioma, LGG, MPNST, plexiform neurofibroma, café-au-lait",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); de novo 50% cases; haploinsufficiency",
        "brain_tumor_risk": "OPG 15-20% PATHOGNOMONIC; LGG/pilocytic astrocytoma most common CNS tumor; brainstem glioma indolent",
        "pathognomonic": "Café-au-lait macules 6+ PATHOGNOMONIC; Lisch nodules PATHOGNOMONIC; OPG 15-20% PATHOGNOMONIC; MPNST 8-13% PATHOGNOMONIC",
        "key_avoid": "Do NOT miss annual ophthalmology from age 1yr (OPG surveillance). Do NOT treat MPNST as glioma — MPNST = sarcoma (doxorubicin). Selumetinib for OPG/plexiform NF, not MPNST",
        "key_rule": "NF1: annual ophthalmology from 1yr MANDATORY (OPG 15-20%). Selumetinib FDA2020 OPG+plexiform NF. Café-au-lait 6+ PATHOGNOMONIC. MPNST = sarcoma doxorubicin NOT glioma treatment. Annual MRI plexiform.",
        "surveillance": "Annual ophthalmology + VEP from age 1yr; MRI brain+orbits if OPG suspected/detected; annual whole-body MRI if plexiform neurofibromas (MPNST); annual blood pressure check; selumetinib eligibility if OPG visual decline or plexiform symptomatic; cascade NF1 first-degree germline testing",
        "targeted_rx": "Selumetinib FDA2020 MEK inhibitor OPG+plexiform neurofibromas (not MPNST); carboplatin+vincristine LGG first-line chemotherapy; BRAF/MEK inhibitors BRAF-V600E NF1-HGG; doxorubicin+ifosfamide MPNST sarcoma first-line; trabectedin/pazopanib MPNST second-line; bevacizumab NF1-associated LGG refractory",
    },
    {
        "gene": "NF2",
        "protein": (
            "NF2 -- 22q12.2 Autosomal-Dominant-LOF -- 595aa -- "
            "Merlin-Schwannomin-66kDa-ERM-FERM-Band-4.1-Scaffold-"
            "Bilateral-Vestibular-Schwannoma-PATHOGNOMONIC-"
            "Meningioma-Multiple-Ependymoma-Spinal-"
            "Bevacizumab-Reduces-VS-OMIM-607379"
        ),
        "locus": "22q12.2",
        "protein_size": (
            "595 aa / 66 kDa / 22q12.2 NF2 brain tumor molecular context: "
            "STRUCTURE: "
            "  595 aa / 66 kDa; FERM domain (band 4.1-ezrin-radixin-moesin, aa 1-339) — links cytoskeleton to membrane; "
            "  Alpha-helical domain; C-terminal unique domain; "
            "  Merlin acts as tumour suppressor: inhibits CRL4-DCAF1 E3 ubiquitin ligase; "
            "  Merlin LOF → YAP/TAZ pathway activation → proliferation; "
            "  Somatic NF2 mutations: 50-60% sporadic meningiomas, 70-80% sporadic schwannomas; "
            "CANCER RISKS (BRAIN TUMOR FOCUS): "
            "  Bilateral vestibular schwannoma (VS): 90-95% lifetime PATHOGNOMONIC — defines NF2 diagnosis; "
            "  Meningioma: 50-80% lifetime — multiple, spinal, intracranial (WHO grade 1-2); "
            "  Ependymoma: 30-53% — spinal ependymoma most common; intramedullary; "
            "  Cataract (posterior subcapsular): 80% — PATHOGNOMONIC NF2 childhood sign; "
            "  Cranial nerve schwannoma (other): CN3-12 non-VS schwannoma; "
            "KEY MANAGEMENT: "
            "  Annual audiology/audiogram from diagnosis MANDATORY (bilateral hearing loss); "
            "  Annual MRI brain+spine from age 10-12yr MANDATORY; "
            "  Bevacizumab: reduces VS volume and improves/stabilises hearing in NF2; "
            "  Surgery vs bevacizumab vs stereotactic radiosurgery: multidisciplinary per VS size/growth; "
            "  Cochlear implant consideration if bilateral deafness; "
            "  Cataract: 80% — annual ophthalmology PATHOGNOMONIC NF2 diagnostic clue in children"
        ),
        "syndrome": "Neurofibromatosis type 2 (NF2) — bilateral VS, meningioma, ependymoma, schwannoma, cataract",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); de novo 50% cases; haploinsufficiency; somatic mosaicism 25-30%",
        "brain_tumor_risk": "Bilateral VS 90-95% PATHOGNOMONIC; meningioma 50-80%; ependymoma 30-53% spinal; other cranial nerve schwannoma",
        "pathognomonic": "Bilateral vestibular schwannoma PATHOGNOMONIC (defines NF2 diagnosis); posterior subcapsular cataract 80% PATHOGNOMONIC; spinal ependymoma cluster",
        "key_avoid": "Do NOT use stereotactic radiosurgery (SRS/Gamma Knife) in young NF2 patients without counselling — radiation-induced malignant transformation risk in NF2; Bevacizumab preferred for growing VS in hearing-intact patients",
        "key_rule": "NF2: annual audiology MANDATORY (bilateral VS 90-95%). Bevacizumab reduces VS hearing loss. Annual MRI brain+spine from age 10-12yr. Cataract 80% = PATHOGNOMONIC NF2 childhood clue. SRS radiation caution young patients.",
        "surveillance": "Annual audiology+audiogram from diagnosis; annual MRI brain+whole spine from age 10-12yr; annual ophthalmology (cataract); annual dermatology (peripheral schwannomas); bevacizumab eligibility if growing VS with hearing; cochlear implant planning bilateral deafness; cascade NF2 germline first-degree relatives",
        "targeted_rx": "Bevacizumab (anti-VEGF) — NF2 VS volume reduction + hearing improvement (level B evidence); lapatinib EGFR/HER2 VS (emerging); everolimus mTOR meningioma (NF2-associated); surgery VS (wait-watch vs resect by growth rate and hearing); stereotactic radiosurgery VS (caution young patients); spinal ependymoma: complete resection first-line",
    },
    {
        "gene": "VHL",
        "protein": (
            "VHL -- 3p25.3 Autosomal-Dominant-LOF -- 213aa -- "
            "pVHL-24kDa-HIF-Alpha-E3-Ubiquitin-Ligase-Adaptor-"
            "CNS-Hemangioblastoma-PATHOGNOMONIC-"
            "Retinal-Angioma-PATHOGNOMONIC-Cerebellar-Spinal-Hemangioblastoma-"
            "Belzutifan-FDA2021-HIF-2alpha-Inhibitor-OMIM-608537"
        ),
        "locus": "3p25.3",
        "protein_size": (
            "213 aa / 24 kDa / 3p25.3 VHL brain tumor molecular context: "
            "STRUCTURE: "
            "  213 aa / 24 kDa; alpha domain (aa 154-213) + beta domain (aa 63-154); "
            "  VHL forms E3 ubiquitin ligase complex with elongin B, elongin C, cullin 2, RBX1; "
            "  pVHL polyubiquitinates HIF-1alpha and HIF-2alpha (prolyl hydroxylase-dependent); "
            "  VHL LOF → HIF-alpha stabilisation → VEGF/EPO/PDGF/TGFalpha overexpression; "
            "  Type 2B VHL (R167W/Y98H): highest hemangioblastoma + pheochromocytoma + ccRCC risk; "
            "CANCER RISKS (BRAIN TUMOR FOCUS): "
            "  CNS hemangioblastoma: 60-80% lifetime PATHOGNOMONIC — cerebellar, spinal cord, brainstem; "
            "  Retinal hemangioblastoma/angioma: 50-60% PATHOGNOMONIC — leading cause of VHL vision loss; "
            "  Endolymphatic sac tumour (ELST): 10-15% — hearing loss, tinnitus; "
            "  pNETs: 15-17% (non-functional, clear cell); "
            "  ccRCC: 70% lifetime; "
            "KEY MANAGEMENT: "
            "  Annual MRI brain+spine from age 15yr MANDATORY; "
            "  Annual ophthalmology laser/photodynamic therapy for retinal angiomas; "
            "  Belzutifan FDA2021: HIF-2alpha inhibitor — VHL-associated hemangioblastoma + ccRCC + pNETs; "
            "  Hemangioblastoma surveillance: watch-and-wait for asymptomatic; surgery if symptomatic/growing; "
            "  Annual audiometry (ELST); MRI petrous bone if ELST suspected"
        ),
        "syndrome": "Von Hippel-Lindau disease (VHL) — hemangioblastoma, retinal angioma, ccRCC, pNET, pheochromocytoma, ELST",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); somatic second hit LOH 3p25.3; VHL type 2B highest brain risk",
        "brain_tumor_risk": "CNS hemangioblastoma 60-80% PATHOGNOMONIC; retinal angioma 50-60% PATHOGNOMONIC; ELST 10-15% hearing loss",
        "pathognomonic": "CNS hemangioblastoma PATHOGNOMONIC; retinal angioma PATHOGNOMONIC; ELST hearing loss PATHOGNOMONIC; HIF pathway signature",
        "key_avoid": "Do NOT delay annual retinal surveillance — retinal angioma untreated = permanent visual loss; Belzutifan FDA2021 for growing/symptomatic hemangioblastoma; watch-and-wait acceptable for small asymptomatic CNS hemangioblastoma",
        "key_rule": "VHL: annual MRI brain+spine from age 15yr MANDATORY. Annual ophthalmology laser retinal angioma. Belzutifan FDA2021 CNS hemangioblastoma+ccRCC+pNET. ELST audiometry annual. Cascade VHL germline first-degree relatives.",
        "surveillance": "Annual MRI brain+spine from age 15yr; annual ophthalmology + retinal fluorescein angiography from age 5yr; annual abdominal MRI (ccRCC+pNET+pheochromocytoma); annual audiometry + MRI petrous bone (ELST); annual urine catecholamines (pheochromocytoma); belzutifan eligibility for multiple/growing tumours; cascade VHL germline first-degree relatives",
        "targeted_rx": "Belzutifan FDA2021 (MK-6482) HIF-2alpha inhibitor — VHL-associated hemangioblastoma + ccRCC + pNETs (PR 36% hemangioblastoma); surgery symptomatic hemangioblastoma; laser photocoagulation or photodynamic therapy retinal angioma; sunitinib/pazopanib ccRCC; streptozocin/everolimus pNETs; cochlear implant ELST deafness",
    },
    {
        "gene": "PTCH1",
        "protein": (
            "PTCH1 -- 9q22.32 Autosomal-Dominant-LOF -- 1447aa -- "
            "Patched1-161kDa-SHH-Receptor-12-TM-Gorlin-NBCCS-"
            "Desmoplastic-Medulloblastoma-5pct-Lifetime-"
            "BCC-1000s-Lifetime-OKC-PATHOGNOMONIC-"
            "AVOID-RADIATION-ABSOLUTELY-Vismodegib-FDA2012-OMIM-601309"
        ),
        "locus": "9q22.32",
        "protein_size": (
            "1447 aa / 161 kDa / 9q22.32 PTCH1 brain tumor molecular context: "
            "STRUCTURE: "
            "  1447 aa / 161 kDa; 12 transmembrane domains; two extracellular loops; "
            "  Sterol-sensing domain (SSD, aa 616-785); "
            "  PTCH1 inhibits Smoothened (SMO) in absence of SHH ligand; "
            "  SHH binds PTCH1 → relieves SMO inhibition → GLI transcription factors activated; "
            "  PTCH1 LOF → constitutive SMO activation → SHH pathway hyperactive → proliferation; "
            "CANCER RISKS (BRAIN TUMOR FOCUS): "
            "  Desmoplastic/nodular medulloblastoma: 5% lifetime in Gorlin — SHH subtype; "
            "  Medulloblastoma in Gorlin = HIGHEST risk in FIRST 5 YEARS of life; "
            "  CRITICAL: radiation for medulloblastoma → HUNDREDS of BCCs along radiation field; "
            "  BCC: 1000+ lifetime lifetime PATHOGNOMONIC — jaw BCC; radiation-triggered BCC explosion; "
            "  Odontogenic keratocysts (OKC): PATHOGNOMONIC Gorlin diagnostic criterion; "
            "  Calcification of falx cerebri: PATHOGNOMONIC Gorlin diagnostic criterion; "
            "KEY MANAGEMENT: "
            "  AVOID RADIATION ABSOLUTELY — radiation triggers massive BCC explosion in PTCH1 carriers; "
            "  Medulloblastoma: avoid cranial radiation; proton therapy or chemotherapy-only protocols; "
            "  Annual dermatology from age 10yr MANDATORY; "
            "  Annual dental panoramic X-ray from age 8yr (OKC surveillance); "
            "  Vismodegib FDA2012 SMO inhibitor — BCC Gorlin (not medulloblastoma — teratogenic/toxicity children)"
        ),
        "syndrome": "Gorlin syndrome / Naevoid Basal Cell Carcinoma Syndrome (NBCCS) — BCC, medulloblastoma, OKC, calcified falx",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); de novo 20-30%; haploinsufficiency + somatic LOH",
        "brain_tumor_risk": "Desmoplastic medulloblastoma 5% lifetime in first 5yr; radiation → BCC explosion along field; SHH subtype",
        "pathognomonic": "OKC PATHOGNOMONIC; calcified falx cerebri PATHOGNOMONIC; BCC <20yr PATHOGNOMONIC; Gorlin multi-criterion diagnosis",
        "key_avoid": "AVOID RADIATION ABSOLUTELY — PTCH1 Gorlin + radiation = hundreds of BCCs along radiation field; medulloblastoma Gorlin = proton therapy or chemo-only; NEVER standard craniospinal irradiation Gorlin medulloblastoma",
        "key_rule": "PTCH1/Gorlin: AVOID RADIATION ABSOLUTELY (massive BCC). Medulloblastoma = proton/chemo-only protocol. Annual dermatology from 10yr. Annual dental OPG from 8yr. Vismodegib BCC adults. Cascade PTCH1 germline first-degree relatives.",
        "surveillance": "Annual full-body dermatology from age 10yr; annual dental panoramic X-ray from age 8yr (OKC); MRI brain from birth to age 5yr (medulloblastoma surveillance children); echocardiography (cardiac fibroma childhood); ovarian fibroma women; renal ultrasound; cascade PTCH1 germline first-degree relatives",
        "targeted_rx": "Vismodegib FDA2012 (SMO inhibitor) Gorlin BCC adults — 43% BCC complete response; sonidegib FDA2015 BCC adults; proton therapy medulloblastoma (AVOID conventional photon craniospinal irradiation); carboplatin+vincristine+cyclophosphamide medulloblastoma chemotherapy-only Gorlin; AVOID radiation field planning near skin for any Gorlin treatment; surgical excision early OKCs",
    },
    {
        "gene": "TSC2",
        "protein": (
            "TSC2 -- 16p13.3 Autosomal-Dominant-LOF -- 1807aa -- "
            "Tuberin-198kDa-mTOR-GAP-Hamartin-TSC1-Heterodimerisation-"
            "SEGA-PATHOGNOMONIC-Cortical-Tubers-PATHOGNOMONIC-"
            "Subependymal-Nodules-Cardiac-Rhabdomyoma-Fetal-"
            "Everolimus-FDA2012-SEGA-OMIM-191092"
        ),
        "locus": "16p13.3",
        "protein_size": (
            "1807 aa / 198 kDa / 16p13.3 TSC2 brain tumor molecular context: "
            "STRUCTURE: "
            "  1807 aa / 198 kDa; N-terminal coiled-coil (TSC1/hamartin binding); "
            "  GAP domain (aa 1517-1674) — GTPase activating for RHEB; "
            "  TSC1/TSC2 complex: inhibits RHEB → inhibits mTORC1; "
            "  TSC2 LOF → RHEB-GTP accumulates → mTORC1 constitutively active → protein synthesis/cell growth; "
            "  TSC2 mutations: more severe phenotype than TSC1 (higher seizure burden, more SEGA); "
            "CANCER RISKS (BRAIN TUMOR FOCUS): "
            "  SEGA (subependymal giant cell astrocytoma): 5-20% TSC lifetime PATHOGNOMONIC; "
            "    SEGA = WHO grade 1 near foramen of Monro → obstructive hydrocephalus; "
            "  Cortical tubers: PATHOGNOMONIC; epileptogenicity major; number correlates with cognitive impairment; "
            "  Subependymal nodules (SENs): PATHOGNOMONIC; precursors to SEGA (some); "
            "  Astrocytoma (tuber-associated): rare transformation; "
            "KEY MANAGEMENT: "
            "  Everolimus FDA2012: mTOR inhibitor — SEGA volume reduction PATHOGNOMONIC indication; "
            "  MRI brain every 1-3yr monitoring SEN→SEGA growth; 6-monthly if growing SEN; "
            "  Annual EEG (epilepsy common — cortical tubers); "
            "  Vigabatrin first-line infantile spasms TSC (EEG monitoring mandatory — visual field); "
            "  Everolimus also FDA-approved: TSC renal AML (2013) + TSC LAM (2016)"
        ),
        "syndrome": "Tuberous sclerosis complex (TSC) — SEGA, cortical tubers, SEN, cardiac rhabdomyoma, renal AML, LAM",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); de novo 65% TSC2; TSC2 > TSC1 severity; haploinsufficiency",
        "brain_tumor_risk": "SEGA 5-20% PATHOGNOMONIC (foramen Monro obstruction); cortical tubers PATHOGNOMONIC; SEN → SEGA progression 10-15yr",
        "pathognomonic": "SEGA near foramen of Monro PATHOGNOMONIC; cortical tubers PATHOGNOMONIC; SEN PATHOGNOMONIC; cardiac rhabdomyoma fetal/neonatal PATHOGNOMONIC",
        "key_avoid": "Do NOT miss growing SEN on serial MRI — SEN → SEGA → obstructive hydrocephalus; Everolimus FDA2012 initiated EARLY for growing SEGA; vigabatrin infantile spasms TSC (visual field monitoring mandatory)",
        "key_rule": "TSC2: annual MRI brain MANDATORY (SEN→SEGA surveillance). Everolimus FDA2012 growing SEGA. Cortical tubers PATHOGNOMONIC. Vigabatrin infantile spasms. Cardiac rhabdomyoma fetal PATHOGNOMONIC TSC. Cascade testing.",
        "surveillance": "MRI brain every 1-3yr (SEN/SEGA surveillance); 6-monthly MRI if SEN growing; annual EEG (epilepsy); annual renal MRI (AML surveillance); annual echocardiography (cardiac rhabdomyoma); annual pulmonary function + CT chest women (LAM); ophthalmic examination (retinal hamartomas); cascade TSC1/TSC2 germline first-degree relatives",
        "targeted_rx": "Everolimus FDA2012 SEGA volume reduction (50% reduction in 26 weeks); everolimus FDA2013 renal AML (>3cm); everolimus FDA2016 LAM; vigabatrin infantile spasms TSC (visual field VEP monitoring); CBD (cannabidiol) Epidyolex FDA2018 TSC seizures; mTOR inhibitor rapamycin skin angiofibromas topical; surgery SEGA if hydrocephalus acute; cannabis-based CBD focal seizures TSC",
    },
    {
        "gene": "PTEN",
        "protein": (
            "PTEN -- 10q23.31 Autosomal-Dominant-LOF -- 403aa -- "
            "PTEN-47kDa-Dual-Phosphatase-PI3K-PIP3-Phosphatase-"
            "Cowden-PHTS-Lhermitte-Duclos-PATHOGNOMONIC-"
            "Macrocephaly-PATHOGNOMONIC-Adult-Onset-"
            "Breast-85pct-Thyroid-35pct-OMIM-601728"
        ),
        "locus": "10q23.31",
        "protein_size": (
            "403 aa / 47 kDa / 10q23.31 PTEN brain tumor molecular context: "
            "STRUCTURE: "
            "  403 aa / 47 kDa; N-terminal PBD domain (phosphoinositide-binding); "
            "  Phosphatase domain (aa 7-185) — dual-specificity phosphatase; "
            "  C2 domain (aa 186-351) — membrane-binding; C-terminal PDZ-binding domain; "
            "  PTEN dephosphorylates PIP3 → PIP2 (PI3K antagonist); "
            "  PTEN LOF → PI3K/AKT/mTOR constitutively active → cell growth/survival; "
            "  Nuclear PTEN: chromosome stability, DNA repair (independent of PI3K); "
            "CANCER RISKS (BRAIN TUMOR FOCUS): "
            "  Lhermitte-Duclos dysplastic cerebellar gangliocytoma: PATHOGNOMONIC PTEN — striped/striated MRI pattern; "
            "  Macrocephaly: 90% PTEN/PHTS carriers PATHOGNOMONIC — HC >2 SD above mean; "
            "  PTEN brain tumors: LDD, astrocytoma, glioblastoma (somatic PTEN common); "
            "  Adult onset Cowden: breast 85% HIGHEST; thyroid 35%; endometrial 28%; colorectal elevated; "
            "  PTEN autism spectrum: macrocephaly + developmental delay without obvious cancer — PHTS; "
            "KEY MANAGEMENT: "
            "  Annual MRI brain (Lhermitte-Duclos baseline, serial surveillance); "
            "  Macrocephaly in any child: measure HC — if >2SD → PTEN germline testing; "
            "  Annual breast MRI from age 25-30yr (breast 85% HIGHEST); "
            "  Annual thyroid US from age 18yr; "
            "  LDD: monitor — complete cerebellar resection if symptomatic; recurrence possible"
        ),
        "syndrome": "Cowden syndrome / PTEN Hamartoma Tumour Syndrome (PHTS) — LDD, macrocephaly, breast 85%, thyroid, endometrial",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); de novo ~15%; haploinsufficiency; somatic LOH 10q23",
        "brain_tumor_risk": "Lhermitte-Duclos PATHOGNOMONIC; macrocephaly 90% PATHOGNOMONIC; astrocytoma/GBM elevated; LDD stripe MRI sign",
        "pathognomonic": "Lhermitte-Duclos dysplastic cerebellar gangliocytoma PATHOGNOMONIC; macrocephaly PATHOGNOMONIC; MRI striped cerebellum PATHOGNOMONIC LDD; adult-onset Cowden",
        "key_avoid": "Do NOT dismiss macrocephaly — HC >2 SD + any family history → PTEN germline testing; LDD is adult-onset Cowden pathognomonic brain lesion (cerebellar stripes on MRI); annual breast MRI MANDATORY from 25-30yr (breast 85% HIGHEST)",
        "key_rule": "PTEN/Cowden: macrocephaly >2 SD PATHOGNOMONIC → PTEN germline test. Lhermitte-Duclos PATHOGNOMONIC cerebellar gangliocytoma. Annual breast MRI from 25-30yr MANDATORY (85% risk). Annual thyroid US. LDD serial MRI brain.",
        "surveillance": "Annual MRI brain from LDD diagnosis; annual breast MRI+mammography from age 25-30yr MANDATORY; annual thyroid ultrasound from 18yr; colonoscopy from age 35yr; annual endometrial biopsy from 30-35yr; annual dermatology (trichilemmomas PHTS skin sign); cascade PTEN germline first-degree relatives",
        "targeted_rx": "LDD: surgical cerebellar resection if symptomatic (obstructive hydrocephalus); everolimus/temsirolimus mTOR pathway (experimental PTEN LDD); breast cancer: standard systemic therapy ± olaparib HR-deficient; thyroid cancer: standard per histology; PI3K inhibitors (alpelisib) PTEN-null solid tumors emerging; endometrial cancer: carboplatin+paclitaxel ± pembrolizumab",
    },
    {
        "gene": "SUFU",
        "protein": (
            "SUFU -- 10q24.32 Autosomal-Dominant-LOF -- 484aa -- "
            "SUFU-54kDa-Suppressor-of-Fused-SHH-Pathway-Regulator-"
            "SHH-Medulloblastoma-33pct-HIGHEST-Germline-SHH-Gene-"
            "Meningioma-Elevated-SUFU-Gorlin-Like-"
            "Desmoplastic-Nodular-Medulloblastoma-PATHOGNOMONIC-OMIM-607035"
        ),
        "locus": "10q24.32",
        "protein_size": (
            "484 aa / 54 kDa / 10q24.32 SUFU brain tumor molecular context: "
            "STRUCTURE: "
            "  484 aa / 54 kDa; N-terminal domain (aa 1-268); C-terminal domain (aa 269-484); "
            "  SUFU binds GLI transcription factors (GLI1/2/3) in cytoplasm; "
            "  SUFU-GLI complex sequesters GLI → prevents nuclear translocation → SHH pathway inhibited; "
            "  SUFU LOF → GLI1/2 constitutive nuclear activity → SHH target genes active; "
            "  SUFU LOF = HIGHEST penetrance germline SHH pathway gene for medulloblastoma; "
            "  Distinct from PTCH1 Gorlin: SUFU has higher medulloblastoma penetrance, less BCC; "
            "CANCER RISKS (BRAIN TUMOR FOCUS): "
            "  SHH medulloblastoma: 33% lifetime HIGHEST germline SHH gene penetrance; "
            "  Desmoplastic/nodular medulloblastoma PATHOGNOMONIC: classic SUFU histology; "
            "  Medulloblastoma in SUFU: typically FIRST 5 YEARS of life (infants/toddlers); "
            "  Meningioma: elevated lifetime risk (different from PTCH1 BCC profile); "
            "  SUFU Gorlin-like: BCCs less prominent than PTCH1 Gorlin — meningioma and MB dominant; "
            "KEY MANAGEMENT: "
            "  MRI brain surveillance from birth to age 5yr MANDATORY — medulloblastoma FIRST 5 YEARS; "
            "  Annual MRI brain ages 5-25yr (meningioma risk); "
            "  AVOID RADIATION for medulloblastoma treatment (SHH pathway gene — similar PTCH1 concern); "
            "  Desmoplastic nodular MB: chemotherapy-only protocol in infants if possible; "
            "  Gorlin-like: dermatology surveillance BCC from age 15yr (less prominent than PTCH1)"
        ),
        "syndrome": "SUFU-associated SHH medulloblastoma predisposition — Gorlin-like; medulloblastoma 33% HIGHEST; meningioma elevated",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); de novo ~30%; haploinsufficiency; LOH somatic second hit",
        "brain_tumor_risk": "SHH medulloblastoma 33% HIGHEST germline SHH gene; desmoplastic nodular MB PATHOGNOMONIC; meningioma elevated",
        "pathognomonic": "Desmoplastic/nodular medulloblastoma PATHOGNOMONIC SUFU; SHH subtype medulloblastoma germline; first 5yr of life medulloblastoma peak; SUFU Gorlin-like (less BCC than PTCH1)",
        "key_avoid": "AVOID RADIATION for SUFU medulloblastoma in infants/young children — SHH pathway gene + radiation = secondary BCC/malignancy risk; chemotherapy-only protocols preferred infants; MRI brain surveillance from BIRTH mandatory",
        "key_rule": "SUFU: MRI brain from birth to 5yr MANDATORY (MB 33% HIGHEST germline SHH). Desmoplastic nodular MB PATHOGNOMONIC. AVOID RADIATION infants (SHH pathway). Meningioma annual MRI from age 15yr. Gorlin-like BCC surveillance. Cascade SUFU germline.",
        "surveillance": "MRI brain every 6-12 months from birth to age 5yr; annual MRI brain ages 5-25yr (meningioma); annual dermatology from age 15yr (Gorlin-like BCC surveillance); PTCH1 Gorlin full criteria if BCC prominent; cascade SUFU germline first-degree relatives; OKC dental panoramic if Gorlin-like features",
        "targeted_rx": "Desmoplastic nodular MB infants: chemotherapy-only baby-brain protocol (carboplatin, vincristine, methotrexate) — avoid craniospinal irradiation; vismodegib SMO inhibitor SHH medulloblastoma adults (AVOID infants/children — teratogenicity + bone growth); sonidegib FDA2015 SHH MB adults; meningioma: surgery if symptomatic/growing; SHH MB: SONIC/PNET2 protocol; BCC adults: vismodegib FDA2012 or sonidegib",
    },
]

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]

TREATMENT_PROTOCOLS_BY_GENE = {
    "TP53": [
        "ONC201/Dordaviprone FDA2022 H3K27M-mutant DIPG (independent of TP53 germline)",
        "Temozolomide+bevacizumab GBM standard (TP53 does not predict response)",
        "AVOID radiation in LFS — proton therapy if radiation unavoidable (multidisciplinary decision)",
        "Pembrolizumab hypermutant TP53-associated GBM (high TMB >10 mut/Mb)",
        "ACT-like regimen DIPG+H3K27M+TP53 (ONC201 + radiation-sparing protocol)",
        "Carboplatin+vincristine LGG pediatric LFS",
        "Annual WB-MRI Toronto protocol surveillance",
        "Cascade germline TP53 testing all first-degree relatives",
    ],
    "NF1": [
        "Selumetinib FDA2020 MEK inhibitor — OPG+plexiform neurofibromas (not MPNST)",
        "Carboplatin+vincristine LGG first-line chemotherapy (NF1-LGG standard)",
        "BRAF+MEK inhibitors dabrafenib+trametinib BRAF-V600E NF1-HGG",
        "Bevacizumab anti-VEGF NF1-LGG refractory",
        "Doxorubicin+ifosfamide MPNST first-line sarcoma regimen (NOT immunotherapy)",
        "Trabectedin/pazopanib MPNST second-line",
        "Surveillance: annual ophthalmology VEP from age 1yr (OPG 15-20%)",
        "Annual whole-body MRI plexiform neurofibromas (MPNST risk)",
    ],
    "NF2": [
        "Bevacizumab anti-VEGF — VS volume reduction + hearing preservation (level B evidence)",
        "Lapatinib EGFR/HER2 VS (emerging NF2)",
        "Everolimus mTOR meningioma NF2-associated (emerging)",
        "Surgery VS (wait-watch vs resect by growth rate and hearing status)",
        "Stereotactic radiosurgery VS (caution young NF2 patients — malignant transformation risk)",
        "Complete surgical resection spinal ependymoma first-line",
        "Cochlear implant bilateral deafness NF2 (bilateral VS)",
        "Annual audiogram + annual MRI brain+spine MANDATORY",
    ],
    "VHL": [
        "Belzutifan FDA2021 (MK-6482) HIF-2alpha inhibitor — hemangioblastoma + ccRCC + pNETs",
        "Surgery symptomatic CNS hemangioblastoma (cerebellum/spine decompression)",
        "Laser photocoagulation or photodynamic therapy retinal angioma (annual ophthalmology)",
        "Sunitinib/pazopanib ccRCC (VHL-associated, belzutifan preferred)",
        "Streptozocin/everolimus pNETs VHL-associated",
        "Annual MRI brain+spine + annual abdominal MRI MANDATORY",
        "Cochlear implant/hearing aid ELST-associated deafness",
        "Cascade VHL germline first-degree relatives",
    ],
    "PTCH1": [
        "Vismodegib FDA2012 SMO inhibitor Gorlin BCC adults (43% complete BCC response)",
        "Sonidegib FDA2015 BCC adults (alternative SMO inhibitor)",
        "AVOID radiation: proton therapy medulloblastoma (NEVER conventional craniospinal irradiation Gorlin)",
        "Carboplatin+vincristine+cyclophosphamide medulloblastoma chemotherapy-only protocol Gorlin",
        "Surgical OKC excision + marsupialisation (repeat — high recurrence)",
        "Annual dermatology from age 10yr (BCC surveillance)",
        "Annual dental OPG from age 8yr (OKC surveillance)",
        "Cascade PTCH1 germline first-degree relatives",
    ],
    "TSC2": [
        "Everolimus FDA2012 SEGA (50% volume reduction in 26 weeks) — growing SEGA foramen Monro",
        "Everolimus FDA2013 renal AML >3cm (TSC)",
        "Everolimus FDA2016 LAM (pulmonary lymphangioleiomyomatosis women TSC)",
        "Vigabatrin infantile spasms TSC first-line (visual field VEP monitoring mandatory)",
        "CBD (cannabidiol) Epidyolex FDA2018 TSC seizures focal+generalised",
        "Rapamycin topical skin angiofibromas (mTOR inhibitor)",
        "Surgery SEGA if acute hydrocephalus (emergency VP shunt or direct resection)",
        "Annual MRI brain every 1-3yr (SEN→SEGA surveillance)",
    ],
    "PTEN": [
        "LDD cerebellar resection if symptomatic (obstructive hydrocephalus) — surgery first-line",
        "Annual breast MRI from age 25-30yr MANDATORY (breast 85% HIGHEST lifetime risk)",
        "PI3K inhibitors alpelisib PTEN-null solid tumors (emerging/experimental)",
        "Everolimus/temsirolimus mTOR pathway PTEN-deficient tumors (experimental LDD)",
        "Breast cancer: standard systemic therapy ± olaparib HR-deficient",
        "Thyroid cancer: standard per histology (papillary/follicular)",
        "Endometrial cancer: carboplatin+paclitaxel ± pembrolizumab dMMR",
        "Annual MRI brain surveillance LDD (serial monitoring cerebellar striped lesion)",
    ],
    "SUFU": [
        "Desmoplastic nodular MB infants: chemotherapy-only baby-brain protocol (AVOID craniospinal irradiation)",
        "Carboplatin+vincristine+methotrexate infants MB SUFU Gorlin-like",
        "Vismodegib SMO inhibitor SHH MB adults (AVOID infants/children — toxicity/teratogenicity)",
        "Sonidegib FDA2015 SHH medulloblastoma adults",
        "Meningioma: surgery if symptomatic/growing; everolimus experimental",
        "BCC adults: vismodegib FDA2012 or sonidegib (less prominent than PTCH1 Gorlin)",
        "Annual MRI brain surveillance meningioma (ages 5-25yr)",
        "Cascade SUFU germline first-degree relatives",
    ],
}

SURVEILLANCE_BY_GENE = {
    "TP53": [
        "Annual whole-body MRI Toronto protocol MANDATORY from age of diagnosis",
        "Annual brain MRI (brain tumor surveillance LFS)",
        "6-monthly brain MRI if prior brain tumor diagnosis",
        "Annual breast MRI women from age 20yr",
        "Annual abdominal US + blood count",
        "AVOID radiation in all LFS carriers — proton sparing if essential",
        "Cascade TP53 germline first-degree relatives",
    ],
    "NF1": [
        "Annual ophthalmology + VEP from age 1yr MANDATORY (OPG 15-20%)",
        "MRI brain+orbits if OPG suspected or visual decline",
        "Annual whole-body MRI if plexiform neurofibromas (MPNST surveillance)",
        "Annual blood pressure check",
        "Selumetinib eligibility if OPG with visual decline or symptomatic plexiform NF",
        "Annual dermatology (café-au-lait, neurofibromas)",
        "Cascade NF1 germline first-degree relatives",
    ],
    "NF2": [
        "Annual audiology + audiogram from diagnosis MANDATORY",
        "Annual MRI brain + whole spine from age 10-12yr MANDATORY",
        "Annual ophthalmology (posterior subcapsular cataract, epiretinal membrane)",
        "Annual dermatology (peripheral schwannomas)",
        "Bevacizumab eligibility if VS growing with hearing",
        "Cochlear implant planning if bilateral deafness developing",
        "Cascade NF2 germline first-degree relatives",
    ],
    "VHL": [
        "Annual MRI brain+spine from age 15yr MANDATORY",
        "Annual ophthalmology + retinal fluorescein angiography from age 5yr",
        "Annual abdominal MRI (ccRCC + pNET + pheochromocytoma)",
        "Annual urine catecholamines/metanephrines (pheochromocytoma)",
        "Annual audiometry + MRI petrous bone (ELST)",
        "Belzutifan eligibility if multiple/growing tumours",
        "Cascade VHL germline first-degree relatives",
    ],
    "PTCH1": [
        "Annual full-body dermatology from age 10yr (BCC surveillance)",
        "Annual dental panoramic X-ray from age 8yr (OKC surveillance)",
        "MRI brain from birth to age 5yr (medulloblastoma surveillance children)",
        "Echocardiography childhood (cardiac fibroma)",
        "Pelvic US women (ovarian fibroma)",
        "AVOID radiation ALL imaging where possible",
        "Cascade PTCH1 germline first-degree relatives",
    ],
    "TSC2": [
        "MRI brain every 1-3yr (SEN→SEGA surveillance)",
        "6-monthly MRI if growing SEN near foramen of Monro",
        "Annual EEG (epilepsy surveillance, cortical tubers)",
        "Annual renal MRI (AML surveillance)",
        "Annual echocardiography (cardiac rhabdomyoma)",
        "Annual pulmonary function + CT chest women (LAM)",
        "Cascade TSC1/TSC2 germline first-degree relatives",
    ],
    "PTEN": [
        "Annual MRI brain from LDD diagnosis (serial surveillance)",
        "Annual breast MRI + mammography from age 25-30yr MANDATORY",
        "Annual thyroid ultrasound from age 18yr",
        "Colonoscopy from age 35yr",
        "Annual endometrial biopsy/sampling from age 30-35yr",
        "Annual dermatology (trichilemmomas PHTS skin sign)",
        "Cascade PTEN germline first-degree relatives",
    ],
    "SUFU": [
        "MRI brain every 6-12 months from birth to age 5yr MANDATORY",
        "Annual MRI brain ages 5-25yr (meningioma surveillance)",
        "Annual dermatology from age 15yr (Gorlin-like BCC)",
        "Dental panoramic X-ray if OKC suspected (Gorlin-like)",
        "AVOID radiation in SHH pathway gene carriers when possible",
        "Cascade SUFU germline first-degree relatives",
    ],
}


def _make_patients(gene: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    n = 40

    brain_tumor_subtypes = {
        "TP53": [
            ("Glioblastoma (GBM)", 0.30),
            ("DIPG (H3K27M-mutant)", 0.25),
            ("Astrocytoma grade 2-3", 0.25),
            ("Medulloblastoma (WNT/SHH)", 0.10),
            ("Choroid plexus carcinoma", 0.10),
        ],
        "NF1": [
            ("Optic pathway glioma", 0.35),
            ("Low-grade glioma (pilocytic astrocytoma)", 0.30),
            ("Brainstem glioma (NF1-associated)", 0.15),
            ("MPNST (malignant peripheral nerve sheath)", 0.12),
            ("High-grade glioma (NF1-HGG)", 0.08),
        ],
        "NF2": [
            ("Vestibular schwannoma (bilateral)", 0.45),
            ("Meningioma (multiple)", 0.30),
            ("Spinal ependymoma", 0.15),
            ("Cranial nerve schwannoma (non-VS)", 0.10),
        ],
        "VHL": [
            ("CNS hemangioblastoma (cerebellar)", 0.40),
            ("Spinal hemangioblastoma", 0.25),
            ("Retinal angioma", 0.20),
            ("Endolymphatic sac tumour (ELST)", 0.15),
        ],
        "PTCH1": [
            ("Basal cell carcinoma (Gorlin)", 0.50),
            ("Desmoplastic medulloblastoma (SHH)", 0.25),
            ("Odontogenic keratocyst (OKC)", 0.15),
            ("Cardiac fibroma", 0.10),
        ],
        "TSC2": [
            ("SEGA (subependymal giant cell astrocytoma)", 0.30),
            ("Cortical tuber-associated epilepsy", 0.30),
            ("Renal angiomyolipoma", 0.25),
            ("Pulmonary LAM", 0.15),
        ],
        "PTEN": [
            ("Lhermitte-Duclos (dysplastic cerebellar gangliocytoma)", 0.35),
            ("Breast carcinoma (Cowden)", 0.30),
            ("Thyroid carcinoma (follicular/papillary)", 0.20),
            ("Endometrial carcinoma", 0.15),
        ],
        "SUFU": [
            ("Desmoplastic/nodular medulloblastoma (SHH)", 0.50),
            ("Meningioma (SUFU Gorlin-like)", 0.25),
            ("Basal cell carcinoma (Gorlin-like)", 0.15),
            ("Astrocytoma (low-grade)", 0.10),
        ],
    }

    variants_by_gene = {
        "TP53": [
            "p.Arg248Trp (R248W — dominant negative DBD hotspot)",
            "p.Arg273His (R273H — DNA contact hotspot)",
            "p.Gly245Ser (G245S — structural hotspot)",
            "p.Arg175His (R175H — structural hotspot conformational change)",
            "p.Arg337His (R337H — Brazilian founder tetramerisation domain)",
            "p.Arg248Gln (R248Q — DBD hotspot)",
            "c.IVS6+1G>A (splice donor intron 6)",
        ],
        "NF1": [
            "p.Arg1534Ter (GRD truncation)",
            "p.Glu1200Lys (GRD RAS-GAP contact)",
            "c.IVS14+1G>A (splice intron 14)",
            "del exons 4-7 (large deletion NF1)",
            "p.Tyr489Ter (early truncation)",
            "p.Leu847Pro (ARM repeat hydrophobic)",
        ],
        "NF2": [
            "p.Gln530Ter (truncation FERM C-terminal)",
            "p.Glu456Ter (truncation helix domain)",
            "p.Arg341Ter (FERM domain truncation)",
            "c.IVS8+1G>A (splice donor exon 8)",
            "p.Leu64Pro (FERM N-terminal hydrophobic core)",
            "p.Tyr66Ter (early truncation NF2)",
        ],
        "VHL": [
            "p.Arg167Trp (R167W — Type 2B highest hemangioblastoma)",
            "p.Tyr98His (Y98H — Type 2B hemangioblastoma+pheochromocytoma)",
            "p.Leu118Pro (alpha-domain packing)",
            "c.IVS2-1G>C (splice acceptor intron 2)",
            "p.Val130Glu (HIF-binding interface)",
            "p.Ser65Trp (beta-domain VHL)",
        ],
        "PTCH1": [
            "p.Arg544Ter (extracellular loop 1 truncation)",
            "p.Leu1079Pro (TM domain 8 structural)",
            "c.IVS14+2T>C (splice donor exon 14 PTCH1)",
            "p.Glu1093Lys (second extracellular loop)",
            "del exons 1-2 (large deletion PTCH1)",
            "p.Gln434Ter (SSD-adjacent truncation)",
        ],
        "TSC2": [
            "p.Arg611Ter (GAP domain truncation)",
            "p.Arg1620Ter (near C-terminus GAP domain)",
            "c.IVS23+1G>A (splice donor exon 23 TSC2)",
            "p.Leu1524Arg (RHEB-GAP catalytic residue)",
            "del exons 17-21 (large deletion TSC2)",
            "p.Arg905Gln (coiled-coil TSC1-binding region)",
        ],
        "PTEN": [
            "p.Arg130Ter (R130X phosphatase domain truncation)",
            "p.Cys124Ser (C124S phosphatase catalytic residue)",
            "p.Glu157Lys (phosphatase domain structural)",
            "c.IVS5+1G>T (splice donor intron 5)",
            "p.Thr167Pro (C2 domain stability)",
            "del exons 3-6 (large deletion PTEN)",
        ],
        "SUFU": [
            "p.Gln394Ter (C-terminal domain truncation)",
            "p.Trp535Ter (near C-terminus SUFU)",
            "p.Ala361Val (N-C domain linker)",
            "c.IVS10+1G>A (splice donor exon 10 SUFU)",
            "p.Arg123Ter (N-terminal domain early truncation)",
            "p.Leu412Arg (C-terminal domain structural)",
        ],
    }

    age_ranges = {
        "TP53": (5, 55),
        "NF1":  (3, 45),
        "NF2":  (15, 55),
        "VHL":  (18, 60),
        "PTCH1": (1, 30),
        "TSC2": (0, 25),
        "PTEN": (25, 65),
        "SUFU": (0, 20),
    }

    subtypes = brain_tumor_subtypes[gene]
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
        stage = rng.choices(["I", "II", "III", "IV"], weights=[0.35, 0.30, 0.20, 0.15])[0]
        radiation_contraindicated = gene in ("TP53", "PTCH1", "SUFU") and rng.random() < 0.80
        targeted_therapy = rng.random() < 0.65
        sega_risk = (gene == "TSC2") and (rng.random() < 0.18)
        hemangioblastoma = (gene == "VHL") and (rng.random() < 0.70)
        lhermitte_duclos = (gene == "PTEN") and (rng.random() < 0.35)
        medulloblastoma_shh = (gene in ("PTCH1", "SUFU")) and (rng.random() < (0.05 if gene == "PTCH1" else 0.33))
        bilateral_vs = (gene == "NF2") and (rng.random() < 0.90)
        optic_glioma = (gene == "NF1") and (rng.random() < 0.18)
        relapse = (stage in ("III", "IV")) and (rng.random() < 0.45)

        patients.append({
            "patient_id": f"{gene[:4].upper()}-{seed:04d}-{i+1:02d}",
            "gene": gene,
            "age_at_dx": age,
            "sex": sex,
            "tumour_type": tumour,
            "stage": stage,
            "variant": variant,
            "radiation_contraindicated": radiation_contraindicated,
            "targeted_therapy": targeted_therapy,
            "sega_risk": sega_risk,
            "hemangioblastoma": hemangioblastoma,
            "lhermitte_duclos": lhermitte_duclos,
            "medulloblastoma_shh": medulloblastoma_shh,
            "bilateral_vs": bilateral_vs,
            "optic_glioma": optic_glioma,
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
    radiation_ci_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["radiation_contraindicated"]) / total, 1
    )
    targeted_therapy_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["targeted_therapy"]) / total, 1
    )
    sega_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["sega_risk"]) / total, 1
    )
    hemangioblastoma_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["hemangioblastoma"]) / total, 1
    )
    medulloblastoma_shh_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["medulloblastoma_shh"]) / total, 1
    )
    bilateral_vs_rate = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["bilateral_vs"]) / total, 1
    )
    mean_age = round(
        sum(p["age_at_dx"] for pts in cohorts.values() for p in pts) / total, 1
    )

    return {
        "atlas": "Hereditary-Brain-Tumor-Predisposition-Atlas",
        "genes": _GENE_LIST,
        "total_patients": total,
        "gene_counts": gene_counts,
        "seed_range": f"{SEED_BASE}-{SEED_BASE+7}",
        "radiation_contraindicated_rate_pct": radiation_ci_rate,
        "targeted_therapy_rate_pct": targeted_therapy_rate,
        "sega_risk_rate_pct": sega_rate,
        "hemangioblastoma_rate_pct": hemangioblastoma_rate,
        "medulloblastoma_shh_rate_pct": medulloblastoma_shh_rate,
        "bilateral_vs_rate_pct": bilateral_vs_rate,
        "mean_age_at_dx": mean_age,
        "key_facts": [
            "TP53/LFS: brain tumors 20-26% lifetime (GBM/DIPG/astrocytoma); AVOID RADIATION ABSOLUTELY; WB-MRI Toronto protocol MANDATORY; ONC201 H3K27M DIPG FDA2022",
            "NF1: optic pathway glioma 15-20% PATHOGNOMONIC; café-au-lait 6+ PATHOGNOMONIC; Lisch nodules PATHOGNOMONIC; selumetinib FDA2020 OPG+plexiform NF; MPNST 8-13% = SARCOMA (doxorubicin NOT glioma treatment)",
            "NF2: bilateral vestibular schwannoma 90-95% PATHOGNOMONIC; meningioma 50-80%; ependymoma 30-53%; cataract 80% PATHOGNOMONIC; bevacizumab reduces VS volume + hearing",
            "VHL: CNS hemangioblastoma 60-80% PATHOGNOMONIC; retinal angioma 50-60% PATHOGNOMONIC; ELST hearing loss; belzutifan FDA2021 HIF-2alpha inhibitor hemangioblastoma+ccRCC+pNET",
            "PTCH1/Gorlin: desmoplastic medulloblastoma 5% first 5yr; AVOID RADIATION ABSOLUTELY (radiation = 1000s of BCCs in field); OKC PATHOGNOMONIC; calcified falx PATHOGNOMONIC; vismodegib BCC adults FDA2012",
            "TSC2: SEGA near foramen of Monro PATHOGNOMONIC — obstructive hydrocephalus risk; cortical tubers PATHOGNOMONIC; everolimus FDA2012 SEGA; vigabatrin infantile spasms TSC; cardiac rhabdomyoma fetal PATHOGNOMONIC",
            "PTEN/Cowden: Lhermitte-Duclos PATHOGNOMONIC cerebellar gangliocytoma (striped MRI); macrocephaly 90% PATHOGNOMONIC; breast 85% HIGHEST; annual breast MRI 25-30yr MANDATORY",
            "SUFU: SHH medulloblastoma 33% HIGHEST germline SHH gene; desmoplastic nodular MB PATHOGNOMONIC; AVOID RADIATION infants; MRI brain from BIRTH to 5yr MANDATORY; meningioma elevated",
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
                "brain_tumor_risk": g["brain_tumor_risk"],
                "pathognomonic": g["pathognomonic"],
                "key_avoid": g["key_avoid"],
                "key_rule": g["key_rule"],
                "surveillance": g["surveillance"],
                "targeted_rx": g["targeted_rx"],
            },
            "n": len(pts),
            "radiation_contraindicated_pct": round(100 * sum(1 for p in pts if p["radiation_contraindicated"]) / len(pts), 1),
            "targeted_therapy_pct": round(100 * sum(1 for p in pts if p["targeted_therapy"]) / len(pts), 1),
            "sega_risk_pct": round(100 * sum(1 for p in pts if p["sega_risk"]) / len(pts), 1),
            "hemangioblastoma_pct": round(100 * sum(1 for p in pts if p["hemangioblastoma"]) / len(pts), 1),
            "medulloblastoma_shh_pct": round(100 * sum(1 for p in pts if p["medulloblastoma_shh"]) / len(pts), 1),
            "bilateral_vs_pct": round(100 * sum(1 for p in pts if p["bilateral_vs"]) / len(pts), 1),
            "optic_glioma_pct": round(100 * sum(1 for p in pts if p["optic_glioma"]) / len(pts), 1),
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
        "atlas": "Hereditary-Brain-Tumor-Predisposition-Atlas",
        "definitions": {
            "tp53_lfs_avoid_radiation_wb_mri": (
                "TP53/LFS: 393aa / 43kDa; 17p13.1; guardian of the genome tumour suppressor; "
                "BRAIN TUMORS 20-26% LIFETIME: GBM, astrocytoma grade 2-4, DIPG (H3K27M co-mutation), medulloblastoma, choroid plexus carcinoma; "
                "DIPG + TP53 GERMLINE + H3K27M: ultra-high-risk combination — ONC201/Dordaviprone FDA2022 H3K27M-mutant DIPG; "
                "AVOID RADIATION ABSOLUTELY: radiation in LFS → secondary sarcomas, secondary BCC, secondary ACC along radiation field; "
                "WB-MRI TORONTO PROTOCOL: annual whole-body MRI + annual brain MRI MANDATORY from diagnosis — comprehensive LFS surveillance; "
                "De novo TP53 ~25% LFS: always test both parents before cascade planning."
            ),
            "nf1_optic_glioma_pathognomonic_selumetinib": (
                "NF1: 2839aa / 319kDa; 17q11.2; RAS-GAP neurofibromin; NF1 LOF → RAS-GTP → MAPK/ERK hyperactivation; "
                "OPTIC PATHWAY GLIOMA 15-20% PATHOGNOMONIC: typically age <10yr — annual ophthalmology + VEP from age 1yr MANDATORY; "
                "CAFÉ-AU-LAIT 6+ MACULES PATHOGNOMONIC: >5mm prepubertal / >15mm postpubertal — diagnostic criterion NF1; "
                "LISCH NODULES PATHOGNOMONIC: iris hamartomas — slit-lamp exam; "
                "SELUMETINIB FDA2020: MEK inhibitor — OPG with visual decline + plexiform neurofibromas — NOT MPNST; "
                "MPNST CRITICAL: 8-13% lifetime SARCOMA — doxorubicin/ifosfamide, NOT glioma treatment; NOT immunotherapy."
            ),
            "nf2_bilateral_vs_pathognomonic_bevacizumab": (
                "NF2: 595aa / 66kDa; 22q12.2; merlin/schwannomin FERM scaffold; merlin LOF → YAP/TAZ activation; "
                "BILATERAL VESTIBULAR SCHWANNOMA 90-95% PATHOGNOMONIC: defines NF2 diagnosis; annual audiogram MANDATORY; "
                "MENINGIOMA 50-80%: multiple intracranial + spinal; WHO grade 1-2; surgery if symptomatic/growing; "
                "EPENDYMOMA 30-53%: spinal intramedullary most common; complete resection first-line; "
                "POSTERIOR SUBCAPSULAR CATARACT 80%: PATHOGNOMONIC NF2 — childhood diagnostic clue; annual ophthalmology; "
                "BEVACIZUMAB: reduces VS volume + improves/stabilises hearing (level B evidence); preferred over SRS in hearing-intact young NF2; "
                "SOMATIC MOSAICISM 25-30% NF2: milder mosaic NF2 may miss germline test — skin biopsy from tumour if germline negative."
            ),
            "vhl_hemangioblastoma_pathognomonic_belzutifan": (
                "VHL: 213aa / 24kDa; 3p25.3; pVHL E3 ubiquitin ligase adaptor — HIF-alpha ubiquitination; "
                "CNS HEMANGIOBLASTOMA 60-80% PATHOGNOMONIC: cerebellar + spinal cord + brainstem; annual MRI brain+spine from 15yr; "
                "RETINAL ANGIOMA 50-60% PATHOGNOMONIC: leading cause VHL vision loss — annual fluorescein angiography from age 5yr; "
                "ELST (ENDOLYMPHATIC SAC TUMOUR) 10-15%: hearing loss/tinnitus — PATHOGNOMONIC petrous bone tumour; annual audiometry; "
                "BELZUTIFAN FDA2021: HIF-2alpha inhibitor — PR 36% CNS hemangioblastoma + ccRCC + pNETs (FDA-approved indication); "
                "TYPE 2B VHL (R167W/Y98H): highest hemangioblastoma + pheochromocytoma + ccRCC risk — most dangerous VHL subtype."
            ),
            "ptch1_gorlin_avoid_radiation_bcc_medulloblastoma": (
                "PTCH1/Gorlin: 1447aa / 161kDa; 9q22.32; 12-TM SHH receptor; PTCH1 LOF → SMO constitutive → GLI active; "
                "DESMOPLASTIC MEDULLOBLASTOMA 5%: in FIRST 5 YEARS of life — PATHOGNOMONIC SHH subtype Gorlin; "
                "AVOID RADIATION ABSOLUTELY: craniospinal irradiation in Gorlin → HUNDREDS of BCCs along radiation field — NEVER CSI Gorlin MB; "
                "MEDULLOBLASTOMA GORLIN: proton therapy only if radiation essential OR chemotherapy-only protocol; "
                "BCC 1000s LIFETIME: PATHOGNOMONIC Gorlin — annual dermatology from 10yr; vismodegib FDA2012 SMO inhibitor adults; "
                "OKC PATHOGNOMONIC: odontogenic keratocysts — annual dental panoramic from 8yr — high recurrence; "
                "CALCIFIED FALX PATHOGNOMONIC: diagnostic criterion Gorlin — plain skull X-ray / CT head."
            ),
            "tsc2_sega_pathognomonic_everolimus": (
                "TSC2: 1807aa / 198kDa; 16p13.3; tuberin mTOR-GAP; TSC1-TSC2 complex inhibits RHEB → mTORC1; "
                "SEGA PATHOGNOMONIC: subependymal giant cell astrocytoma near foramen of Monro → obstructive hydrocephalus; 5-20% TSC; "
                "EVEROLIMUS FDA2012: mTOR inhibitor — 50% SEGA volume reduction in 26 weeks — FIRST-LINE for growing SEGA; "
                "CORTICAL TUBERS PATHOGNOMONIC: epileptogenicity; number correlates with cognitive burden; "
                "SUBEPENDYMAL NODULES (SEN): PATHOGNOMONIC precursors — serial MRI every 1-3yr for SEN→SEGA transition; "
                "CARDIAC RHABDOMYOMA PATHOGNOMONIC: fetal/neonatal — regression in early childhood; echocardiography mandatory; "
                "VIGABATRIN: first-line infantile spasms TSC — visual field VEP monitoring MANDATORY (retinal toxicity); "
                "TSC2 > TSC1 SEVERITY: TSC2 mutations = more severe seizure + cognitive + SEGA burden."
            ),
            "pten_lhermitte_duclos_macrocephaly_pathognomonic": (
                "PTEN/Cowden PHTS: 403aa / 47kDa; 10q23.31; dual-specificity phosphatase PIP3→PIP2; PTEN LOF → PI3K/AKT/mTOR constitutive; "
                "LHERMITTE-DUCLOS (LDD) PATHOGNOMONIC: dysplastic cerebellar gangliocytoma — striped/striated MRI cerebellar pattern PATHOGNOMONIC; "
                "MACROCEPHALY 90% PATHOGNOMONIC: HC >2 SD above mean — PTEN germline test if macrocephaly + any family history; "
                "ADULT ONSET COWDEN: breast 85% HIGHEST LIFETIME; thyroid 35%; endometrial 28%; colorectal elevated; "
                "ANNUAL BREAST MRI MANDATORY: from age 25-30yr MANDATORY (breast 85% = highest single-gene breast risk after BRCA1/2); "
                "PTEN AUTISM SPECTRUM: macrocephaly + developmental delay without obvious cancer = PHTS — test PTEN germline; "
                "NUCLEAR PTEN FUNCTION: chromosome stability independent of PI3K — explains radiosensitivity component."
            ),
            "sufu_shh_medulloblastoma_highest_germline": (
                "SUFU: 484aa / 54kDa; 10q24.32; GLI1/2/3 cytoplasmic sequestration — SHH pathway brake; "
                "SHH MEDULLOBLASTOMA 33% LIFETIME: HIGHEST penetrance germline SHH gene (higher than PTCH1 for MB); "
                "DESMOPLASTIC NODULAR MEDULLOBLASTOMA PATHOGNOMONIC: classic SUFU histology; "
                "FIRST 5 YEARS: medulloblastoma peak infancy/toddler — MRI brain from BIRTH every 6-12 months MANDATORY; "
                "AVOID RADIATION IN INFANTS: SHH pathway gene + radiation → secondary BCCs/malignancies; chemotherapy-only baby-brain protocol preferred; "
                "MENINGIOMA ELEVATED: SUFU Gorlin-like — less BCC than PTCH1 Gorlin; meningioma + medulloblastoma = SUFU profile; "
                "VISMODEGIB ADULTS ONLY: SMO inhibitor SHH MB — AVOID infants/children (teratogenicity, bone growth effects, not age-appropriate); "
                "GORLIN-LIKE OVERLAP: SUFU carriers may partially meet Gorlin criteria (less OKC/calcified falx than PTCH1)."
            ),
        },
        "key_clinical_distinctions": [
            "TP53/LFS: AVOID RADIATION ABSOLUTELY (secondary sarcoma). WB-MRI Toronto protocol MANDATORY annual. DIPG+H3K27M+TP53 = ultra-high-risk. ONC201 H3K27M DIPG FDA2022",
            "NF1: OPG 15-20% PATHOGNOMONIC — annual ophthalmology from 1yr. MPNST = SARCOMA (doxorubicin NOT glioma). Selumetinib FDA2020 OPG+plexiform. Café-au-lait 6+ PATHOGNOMONIC",
            "NF2: bilateral VS 90-95% PATHOGNOMONIC. Bevacizumab reduces VS hearing loss. Cataract 80% PATHOGNOMONIC NF2 childhood. SRS radiation caution young NF2 (malignant transformation)",
            "VHL: CNS hemangioblastoma PATHOGNOMONIC. Retinal angioma PATHOGNOMONIC. Belzutifan FDA2021 HIF-2alpha. Type 2B R167W = highest hemangioblastoma+pheochromocytoma+ccRCC risk",
            "PTCH1/Gorlin: AVOID RADIATION ABSOLUTELY (1000s BCCs in field). Desmoplastic MB first 5yr. OKC + calcified falx PATHOGNOMONIC. Vismodegib BCC adults FDA2012. Proton/chemo-only MB",
            "TSC2: SEGA PATHOGNOMONIC foramen Monro (obstructive hydrocephalus). Everolimus FDA2012 SEGA. Cortical tubers epilepsy. Vigabatrin infantile spasms (VEP monitoring). Cardiac rhabdomyoma fetal PATHOGNOMONIC",
            "PTEN/Cowden: LDD PATHOGNOMONIC (striped cerebellum MRI). Macrocephaly 90% PATHOGNOMONIC → PTEN germline test. Breast 85% HIGHEST. Annual breast MRI from 25-30yr MANDATORY",
            "SUFU: MB 33% HIGHEST germline SHH. Desmoplastic nodular MB PATHOGNOMONIC. MRI brain from BIRTH mandatory. AVOID radiation infants (SHH pathway). Vismodegib adults only (NOT infants)",
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2))
