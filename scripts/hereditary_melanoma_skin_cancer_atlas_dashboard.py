#!/usr/bin/env python3
"""Hereditary-Melanoma-Skin-Cancer-Atlas — Complete 8-Gene Melanoma & Skin Cancer Predisposition Atlas
CDKN2A (p16-INK4a/p14-ARF; 156aa/132aa; 9p21.3; AD LOF;
         Familial Atypical Multiple Mole Melanoma (FAMMM); MOST COMMON hereditary melanoma 30-40%;
         Melanoma lifetime risk 58-76% (Australia high-incidence); PANCREATIC CANCER 17-FOLD RR;
         Dual-locus: p16^INK4a inhibits CDK4/6; p14^ARF stabilises p53 via MDM2 inhibition;
         seed SEED_BASE+0) ·
CDK4   (cyclin-dependent kinase 4; 303aa; 12q14.1; AD GOF p.R24C/H;
         FAMMM type 2; RARE <1% familial melanoma; R24 is p16-binding site;
         R24 mutation → p16 cannot bind → unrestrained CDK4 → RB1 hyperphosphorylation → cell cycle;
         IDENTICAL SURVEILLANCE PROTOCOL to CDKN2A; seed SEED_BASE+1) ·
BAP1   (BRCA1-associated protein 1; 729aa; 3p21.1; AD LOF;
         BAP1 Tumour Predisposition Syndrome (BAP1-TPDS);
         UVEAL MELANOMA lifetime 50% PATHOGNOMONIC; BAPomas (MBAITs) PATHOGNOMONIC on skin;
         mesothelioma 8-10%; cutaneous melanoma; RCC; cholangiocarcinoma;
         BAP1 IHC nuclear loss confirms somatic/germline; seed SEED_BASE+2) ·
PTCH1  (Patched 1; 1447aa; 9q22.32; AD LOF;
         Gorlin Syndrome / Basal Cell Nevus Syndrome (BCNS);
         MULTIPLE BCCs from puberty PATHOGNOMONIC; ODONTOGENIC KERATOCYSTS jaw PATHOGNOMONIC;
         CALCIFIED FALX CEREBRI 65%; desmoplastic SHH medulloblastoma 3-5%;
         AVOID RADIATION ABSOLUTELY; vismodegib/sonidegib FDA approved; seed SEED_BASE+3) ·
SUFU   (Suppressor of Fused; 484aa; 10q24.32; AD LOF;
         Gorlin-like SHH pathway syndrome; SHH MEDULLOBLASTOMA HIGHER RISK THAN PTCH1 (>10%);
         BRAIN MRI MANDATORY from childhood; adult: BCC + meningioma;
         less odontogenic keratocysts than PTCH1; seed SEED_BASE+4) ·
MITF   (Microphthalmia-associated transcription factor; 526aa; 3p14.1; AD GOF p.E318K;
         Melanoma-Astrocytoma Syndrome; E318K removes SUMO site → increased transcriptional activity;
         MODERATE PENETRANCE 14-20% melanoma lifetime; RCC 10-15%; meningioma;
         MITF = master melanocyte regulator (tyrosinase, DCT, PMEL); seed SEED_BASE+5) ·
POT1   (Protection of Telomeres 1; 634aa; 7q31.33; AD LOF;
         Familial melanoma 3-4%; GLIOMA risk elevated; CLL associations;
         POT1 = shelterin complex single-strand 3' telomere overhang protector;
         POT1 LOF → telomere elongation + chromosomal instability → oncogenesis;
         seed SEED_BASE+6) ·
RB1    (Retinoblastoma tumour suppressor; 928aa; 13q14.2; AD LOF;
         Hereditary Retinoblastoma; BILATERAL RETINOBLASTOMA PATHOGNOMONIC for germline;
         TRILATERAL RB (bilateral + pineoblastoma) HIGHLY SPECIFIC germline RB1;
         SECONDARY CANCER RISK 30-40% by age 50 (osteosarcoma most common);
         AVOID RADIATION (radiation-field sarcoma documented); seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3102-3109)
"""
import random

SEED_BASE = 3102

ATLAS_GENES = [
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 Autosomal-Dominant-LOF -- 156aa (p16^INK4a) / 132aa (p14^ARF) -- "
            "p16-INK4a-CDK4-6-Inhibitor-Cell-Cycle-G1-S-Checkpoint-AND-"
            "p14-ARF-MDM2-Inhibitor-p53-Stabiliser-Dual-Tumour-Suppressor-Locus-"
            "FAMMM-Familial-Atypical-Multiple-Mole-Melanoma-MOST-COMMON-30-40pct-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa (p16^INK4a) / 132 aa (p14^ARF) / 9p21.3 CDKN2A locus encodes two distinct tumour suppressors "
            "via alternative reading frames: "
            "p16^INK4a: "
            "  STRUCTURE: 4 ankyrin repeats; binds CDK4/6 catalytic pocket; "
            "  FUNCTION: p16 binds CDK4/6 → prevents CDK4/6-cyclin D complex formation → "
            "    RB1 remains hypophosphorylated → E2F transcription factors sequestered → "
            "    G1 cell cycle arrest (cell cannot enter S phase); "
            "  CDK4 R24C/H mutation: disrupts p16 binding site → p16-resistant CDK4 → bypass G1 checkpoint; "
            "p14^ARF (ARF = Alternate Reading Frame): "
            "  STRUCTURE: entirely different amino acid sequence from p16 (different exon 1β); "
            "  FUNCTION: p14^ARF binds MDM2 → sequesters MDM2 in nucleolus → "
            "    MDM2 cannot ubiquitinate p53 → p53 stabilisation → apoptosis/senescence; "
            "  ARF-MDM2-p53 pathway is a second independent tumour suppressor axis at 9p21.3; "
            "FAMMM SYNDROME (Familial Atypical Multiple Mole Melanoma): "
            "  Clinical features: multiple (>50) atypical/dysplastic nevi; family history melanoma; "
            "    large nevi with irregular borders, varied pigmentation; "
            "  CDKN2A pathogenic variant: 30-40% of familial melanoma kindreds; "
            "  Melanoma penetrance: highly variable by geography: "
            "    High-incidence (Australia): 58-76% lifetime by age 80; "
            "    Low-incidence (Netherlands): 28% lifetime; "
            "    Intermediate (US/Europe): 40-58% lifetime; "
            "PANCREATIC CANCER: "
            "  17-fold relative risk (CDKN2A carriers vs general population); "
            "  Lifetime risk ~17% in CDKN2A-melanoma kindreds; "
            "  EUS (endoscopic ultrasound) + MRI/MRCP annually from age 40-45 recommended; "
            "  p16 loss is the MOST COMMON somatic event in pancreatic adenocarcinoma (>90%); "
            "GERMLINE TESTING: "
            "  Sequencing + MLPA (deletions account for 15-30% CDKN2A variants); "
            "  Founder mutations: p.G101W (Mediterranean/European); c.225-243del19 (Dutch/Scandinavian); "
            "    IVS2-105A>G (intronic, European); p.R24X; p.M53I; c.IVS1+1G>A; "
            "SURVEILLANCE: "
            "  Annual full-body skin examination (dermatoscopy) from age 10-12; "
            "  Total body photography (TBP) + sequential dermoscopy; "
            "  Pancreatic: EUS + MRI annually from age 40; "
            "  Ophthalmology: uveal melanoma surveillance (annual ocular examination); "
            "RISK MODIFIERS: "
            "  MC1R variant co-inheritance: CDKN2A carriers with MC1R red-hair variants have 2-3x further increased risk; "
            "  UV exposure: major environmental modifier -- sun protection mandatory; "
            "  Nevi count: high atypical nevi burden correlates with highest risk."
        ),
        "inheritance": (
            "AD LOF 9p21.3 -- FAMMM / Familial Melanoma. Penetrance 28-76% lifetime (geography-dependent). "
            "CDKN2A deletions detected only by MLPA (15-30% of variants). "
            "p16^INK4a and p14^ARF both disrupted by exon 2 mutations; exon 1α-only mutations affect p16 only; "
            "exon 1β-only mutations affect p14ARF only. "
            "Pancreatic cancer 17x RR -- requires dedicated EUS/MRI surveillance."
        ),
        "disease_category": "Familial Melanoma / FAMMM / Pancreatic Cancer Predisposition",
        "patient_generator_params": {
            "melanoma_primary_risk": 0.62,
            "uveal_melanoma_risk": 0.03,
            "pancreatic_cancer_risk": 0.16,
            "multiple_primaries_risk": 0.18,
            "atypical_nevi_burden": 0.90,
            "bcc_risk": 0.06,
            "glioma_risk": 0.01,
            "age_range": (25, 72),
            "mean_age_dx": 45,
            "severity_dist": {"severe": 0.35, "moderate": 0.45, "mild": 0.20},
            "dermoscopy_surveillance": 0.75,
            "sun_protection_compliant": 0.60,
            "pancreatic_surveillance_done": 0.45,
            "immunotherapy_eligible": 0.40,
            "avoid_radiation": 0.0,
            "bapoma_skin_lesion": 0.0,
            "retinoblastoma_risk": 0.0,
            "mesothelioma_risk": 0.0,
            "rcc_risk": 0.02,
            "odontogenic_keratocyst": 0.0,
        },
    },
    {
        "gene": "CDK4",
        "protein": (
            "CDK4 -- 12q14.1 Autosomal-Dominant-GOF-p.R24C-p.R24H -- 303aa -- "
            "Cyclin-D-Dependent-Kinase-4-p16-Binding-Pocket-R24-Mutation-"
            "FAMMM-Type2-RARE-lt1pct-Familial-Melanoma-Identical-Surveillance-CDKN2A-OMIM-123829"
        ),
        "locus": "12q14.1",
        "protein_size": (
            "303 aa / 33 kDa (CDK4; cyclin-dependent kinase 4; 12q14.1; "
            "STRUCTURE: kinase domain; T-loop activation; PSTAIRE-like helix; p16-binding hydrophobic groove; "
            "FUNCTION: "
            "  CDK4-cyclin D complex: phosphorylates RB1 at multiple sites → "
            "    pRB releases E2F transcription factors → S-phase gene transcription → cell cycle progression; "
            "  T-loop: phosphorylated by CDK7/CAK for full activation; "
            "  p16 inhibition: p16^INK4a binds CDK4 hydrophobic groove → displaces cyclin D → kinase inactive; "
            "R24 MUTATION (FAMMM type 2 GOF): "
            "  R24 residue: critical p16-binding contact in CDK4 hydrophobic groove; "
            "  p.R24C (Arg→Cys): loss of basic charge → p16 binding abolished; CDK4 constitutively active; "
            "  p.R24H (Arg→His): partial loss of p16 binding; similar melanoma predisposition; "
            "  Effect: p16 CANNOT inhibit R24-mutant CDK4 → unrestrained RB1 phosphorylation → "
            "    persistent E2F activation → melanocyte proliferation → melanoma; "
            "  ALL OTHER CDK4 functions preserved; the mutation SPECIFICALLY abolishes p16-mediated inhibition; "
            "FAMILIAL MELANOMA TYPE 2 (CDK4): "
            "  Prevalence: <1% of familial melanoma kindreds (CDKN2A is 30-40x more common); "
            "  Phenotype: IDENTICAL to CDKN2A-melanoma: multiple atypical nevi, family history, early onset; "
            "  Penetrance: estimated similar to CDKN2A (~50-70% lifetime in high-incidence regions); "
            "  PANCREATIC RISK: limited data (some pedigrees), but CDK4 R24 mutations increase pancreatic risk; "
            "SURVEILLANCE (IDENTICAL TO CDKN2A): "
            "  Annual full-body skin examination + dermatoscopy; "
            "  Total body photography; "
            "  Pancreatic surveillance EUS/MRI from age 40 (limited evidence, extrapolated from CDKN2A); "
            "MOLECULAR DISTINCTION: "
            "  CDKN2A LOF: tumour suppressor LOSS → p16 protein absent → CDK4 uninhibited; "
            "  CDK4 R24 GOF: CDK4 resistant to p16 inhibition → same downstream effect; "
            "  Functional equivalence: both converge on unrestrained CDK4-RB1-E2F pathway; "
            "  CDKN2A somatic loss is the MOST COMMON event in sporadic melanoma (>50%); "
            "  CDK4 amplification (somatic): common in acral/mucosal melanoma; germline R24 is distinct; "
            "CLINICAL TESTING: "
            "  CDK4 sequencing specifically for exon 2 (R24C/H); "
            "  Family history melanoma + negative CDKN2A: test CDK4 R24; "
            "  Only 2 pathogenic variants known (R24C and R24H) -- hotspot mutations only."
        ),
        "inheritance": (
            "AD GOF 12q14.1 -- FAMMM Type 2. CDK4 R24C or R24H mutations only (hotspot). "
            "RARE: <1% of familial melanoma. Identical clinical phenotype and surveillance protocol to CDKN2A. "
            "Cyclin D-CDK4 complex remains functional but becomes p16-resistant. "
            "CDK4 amplification (somatic) in 10-15% sporadic melanoma is distinct from germline GOF."
        ),
        "disease_category": "Familial Melanoma Type 2 / CDK4 p16-Resistant",
        "patient_generator_params": {
            "melanoma_primary_risk": 0.58,
            "uveal_melanoma_risk": 0.02,
            "pancreatic_cancer_risk": 0.08,
            "multiple_primaries_risk": 0.15,
            "atypical_nevi_burden": 0.88,
            "bcc_risk": 0.04,
            "glioma_risk": 0.01,
            "age_range": (28, 70),
            "mean_age_dx": 46,
            "severity_dist": {"severe": 0.30, "moderate": 0.50, "mild": 0.20},
            "dermoscopy_surveillance": 0.72,
            "sun_protection_compliant": 0.62,
            "pancreatic_surveillance_done": 0.35,
            "immunotherapy_eligible": 0.38,
            "avoid_radiation": 0.0,
            "bapoma_skin_lesion": 0.0,
            "retinoblastoma_risk": 0.0,
            "mesothelioma_risk": 0.0,
            "rcc_risk": 0.02,
            "odontogenic_keratocyst": 0.0,
        },
    },
    {
        "gene": "BAP1",
        "protein": (
            "BAP1 -- 3p21.1 Autosomal-Dominant-LOF -- 729aa -- "
            "BRCA1-Associated-Protein-1-Deubiquitinase-Polycomb-Repressor-"
            "BAP1-Tumour-Predisposition-Syndrome-BAP1-TPDS-"
            "UVEAL-MELANOMA-50pct-Lifetime-BAPomas-MBAITs-PATHOGNOMONIC-"
            "Mesothelioma-Cutaneous-Melanoma-RCC-Cholangiocarcinoma-OMIM-603089"
        ),
        "locus": "3p21.1",
        "protein_size": (
            "729 aa / 80 kDa (BAP1; BRCA1-associated protein 1; 3p21.1; "
            "STRUCTURE: "
            "  N-terminal UCH (ubiquitin C-terminal hydrolase) domain: deubiquitinase catalytic activity; "
            "  NLS (nuclear localisation signal): nuclear function; "
            "  C-terminal BRCA1-BARD1 interaction region (HBM = HHBM motif); "
            "  ASXL-binding region: interaction with ASXL1/2 → Polycomb repressive deubiquitinase complex (PR-DUB); "
            "FUNCTION: "
            "  Deubiquitinase: removes H2Aub1 (monoubiquitination of histone H2A at K119); "
            "  PR-DUB complex (BAP1-ASXL1): removes H2A K119ub → activates Polycomb target genes; "
            "  DNA damage response: required for homologous recombination; BRCA1-interacting; "
            "  Cell cycle regulation: RB1-associated transcriptional regulation; "
            "  Stem cell pluripotency maintenance via PR-DUB; "
            "BAP1-TPDS (BAP1 Tumour Predisposition Syndrome): "
            "UVEAL MELANOMA (UM): "
            "  MOST COMMON hereditary UM syndrome; lifetime risk ~50%; "
            "  Choroidal/ciliary body melanoma; "
            "  BAP1-mutated UM: WORSE PROGNOSIS (class 2 tumour, metastatic risk >50%); "
            "  Annual ophthalmic examination: pupil dilation + indirect ophthalmoscopy; "
            "  Tumour biopsy: cytogenetic class 1 vs class 2 (monosomy 3 / BAP1 somatic loss in class 2); "
            "BAPomas / MBAITs (Melanocytic BAP1-mutated Atypical Intradermal Tumors): "
            "  PATHOGNOMONIC for germline BAP1 mutation; "
            "  Pink/skin-coloured intradermal papules (2-5mm); often multiple; face/trunk/limbs; "
            "  Histology: large epithelioid spitzoid melanocytes with abundant pale cytoplasm; "
            "  IHC: BAP1 nuclear staining LOST in BAP1-mutated lesions (retained in normal); "
            "  NOT malignant per se but sentinel for germline BAP1 status; "
            "  BIOPSY any suspicious lesion; genotype-histotype correlation; "
            "MESOTHELIOMA: "
            "  8-10% lifetime in BAP1-TPDS; peritoneal + pleural subtypes; "
            "  Age at diagnosis younger than sporadic mesothelioma (~50-60 vs 70+); "
            "  Asbestos exposure compounds risk multiplicatively; "
            "  Annual CT chest/abdomen screening discussed from age 30-35; "
            "CUTANEOUS MELANOMA: elevated lifetime risk; earlier onset; multiple primaries; "
            "RENAL CELL CARCINOMA (RCC): clear cell type; ~5-7% lifetime; "
            "CHOLANGIOCARCINOMA: intrahepatic biliary; rare but recognised; "
            "IHC SCREENING: "
            "  All uveal melanoma biopsies: BAP1 nuclear IHC; loss = possible germline testing; "
            "  Mesothelioma: BAP1 IHC loss in ~60% sporadic + germline; "
            "SURVEILLANCE: "
            "  Annual ophthalmic (UM); annual dermatology (cutaneous melanoma + BAPoma surveillance); "
            "  CT chest/abdomen annually from 30-35 (mesothelioma); renal ultrasound/MRI."
        ),
        "inheritance": (
            "AD LOF 3p21.1 -- BAP1-TPDS. UVEAL MELANOMA ~50% lifetime (class 2 = poor prognosis). "
            "BAPomas (MBAITs) PATHOGNOMONIC for germline BAP1 -- biopsy shows nuclear BAP1 IHC loss. "
            "Somatic second hit in tumours (two-hit model). Asbestos co-exposure compounds mesothelioma risk. "
            "Multiple tumour types: UM + cutaneous melanoma + mesothelioma + RCC + cholangiocarcinoma."
        ),
        "disease_category": "BAP1 Tumour Predisposition Syndrome / Uveal Melanoma",
        "patient_generator_params": {
            "melanoma_primary_risk": 0.20,
            "uveal_melanoma_risk": 0.50,
            "pancreatic_cancer_risk": 0.02,
            "multiple_primaries_risk": 0.25,
            "atypical_nevi_burden": 0.30,
            "bcc_risk": 0.03,
            "glioma_risk": 0.0,
            "age_range": (30, 70),
            "mean_age_dx": 50,
            "severity_dist": {"severe": 0.40, "moderate": 0.45, "mild": 0.15},
            "dermoscopy_surveillance": 0.65,
            "sun_protection_compliant": 0.55,
            "pancreatic_surveillance_done": 0.10,
            "immunotherapy_eligible": 0.30,
            "avoid_radiation": 0.0,
            "bapoma_skin_lesion": 0.72,
            "retinoblastoma_risk": 0.0,
            "mesothelioma_risk": 0.09,
            "rcc_risk": 0.06,
            "odontogenic_keratocyst": 0.0,
        },
    },
    {
        "gene": "PTCH1",
        "protein": (
            "PTCH1 -- 9q22.32 Autosomal-Dominant-LOF -- 1447aa -- "
            "Patched-1-Sonic-Hedgehog-SHH-Pathway-Transmembrane-Receptor-"
            "Gorlin-Syndrome-Basal-Cell-Nevus-Syndrome-BCNS-"
            "MULTIPLE-BCCs-PATHOGNOMONIC-ODONTOGENIC-KERATOCYSTS-PATHOGNOMONIC-"
            "AVOID-RADIATION-ABSOLUTELY-Vismodegib-FDA2012-OMIM-601309"
        ),
        "locus": "9q22.32",
        "protein_size": (
            "1447 aa / 160 kDa (PTCH1; Patched 1; 12 TM domains; sterol-sensing domain SSD; "
            "STRUCTURE: "
            "  12 transmembrane domains; N-terminal and C-terminal cytoplasmic domains; "
            "  SSD (sterol-sensing domain): TM2-TM6 region, similar to NPC1 and HMGCR; "
            "  Two large extracellular loops: SHH ligand-binding sites; "
            "FUNCTION (SHH PATHWAY): "
            "  Without SHH ligand: PTCH1 inhibits Smoothened (SMO) → SMO sequestered → "
            "    Gli transcription factors processed to repressor forms (Gli2R, Gli3R) → pathway OFF; "
            "  SHH binding to PTCH1: SHH displaces PTCH1 → SMO de-repressed → activates Gli1/2 → "
            "    target gene transcription (CCND1, PTCH1, GLI1, SNAI1) → pathway ON; "
            "  PTCH1 LOF: constitutive SMO activity (SHH-independent) → Gli activators dominant → "
            "    uncontrolled proliferation (especially in skin basal cells, cerebellum granule precursors); "
            "GORLIN SYNDROME (BCNS): "
            "MULTIPLE BCCs: "
            "  BCCs develop from puberty onwards; may number in hundreds; "
            "  Any site but face/trunk predominate; "
            "  PATHOGNOMONIC: BCCs in young (<30yr) individual, multiple BCCs, jaw keratocysts; "
            "  Histology: nodular, superficial, or morphoeic (infiltrating); "
            "  Management: surgical excision; curettage; imiquimod/5-FU topical; "
            "    Vismodegib (hedgehog inhibitor) FDA2012: first-line for locally advanced/metastatic BCC "
            "      and Gorlin syndrome BCC burden reduction; "
            "    Sonidegib (FDA2015): alternative hedgehog inhibitor; "
            "    Teratogenicity: mandatory contraception during HHI therapy; "
            "ODONTOGENIC KERATOCYSTS (OKC): "
            "  Multiple jaw keratocysts = PATHOGNOMONIC for Gorlin; "
            "  Appear 1st-2nd decade; maxilla + mandible; may be asymptomatic (OPG essential); "
            "  Recurrence after surgery: high; marsupialization → enucleation strategy; "
            "  OPG (orthopantomogram) every 1-2yr from childhood; "
            "CALCIFIED FALX CEREBRI: "
            "  65% of Gorlin patients; bilateral calcification; visible on skull X-ray/CT; "
            "  Earlier onset than physiological calcification (>50yr normal); "
            "  Pathognomonic when calcification before age 20; "
            "DESMOPLASTIC/SHH MEDULLOBLASTOMA: "
            "  3-5% of Gorlin patients; childhood (3-5yr peak); posterior fossa; "
            "  SHH-subtype medulloblastoma (desmoplastic/nodular or MBEN); "
            "  Annual brain MRI age 1-15yr mandatory; "
            "SKELETAL: bifid ribs (40%); vertebral fusions; scoliosis; frontal bossing; ocular hypertelorism; "
            "OVARIAN FIBROMA: 20% females; calcified; may cause Meigs syndrome (ascites + hydrothorax); "
            "AVOID RADIATION ABSOLUTELY: "
            "  Radiation-induced BCCs documented in radiation field after XRT for medulloblastoma; "
            "  Classic example: Gorlin child treated with posterior fossa XRT → "
            "    hundreds of BCCs developing within radiation field 2-5yr later; "
            "  Proton therapy or surgery PREFERRED for medulloblastoma in Gorlin; "
            "  No radiotherapy for BCCs (radiation-induced field change → more BCCs); "
            "HEDGEHOG PATHWAY INHIBITORS: "
            "  Mechanism: SMO antagonist → blocks downstream SHH signalling; "
            "  Vismodegib (GDC-0449): 150mg daily oral; BCC response rate ~48% (advanced); "
            "  Sonidegib (LDE225): 200mg daily; approved locally advanced BCC; "
            "  Toxicity: muscle cramps, alopecia, dysgeusia, fatigue; teratogenic; cycle on/off."
        ),
        "inheritance": (
            "AD LOF 9q22.32 -- Gorlin Syndrome (BCNS). BCCs from puberty PATHOGNOMONIC. "
            "Jaw odontogenic keratocysts PATHOGNOMONIC (OPG annually from childhood). "
            "AVOID RADIATION ABSOLUTELY -- radiation-induced BCCs documented. "
            "Medulloblastoma 3-5%: proton therapy preferred. Vismodegib/sonidegib hedgehog inhibitors approved. "
            "Founder mutation: Australian c.2383C>T; Italian c.2158insC."
        ),
        "disease_category": "Gorlin Syndrome / Basal Cell Nevus Syndrome / SHH Pathway",
        "patient_generator_params": {
            "melanoma_primary_risk": 0.05,
            "uveal_melanoma_risk": 0.01,
            "pancreatic_cancer_risk": 0.01,
            "multiple_primaries_risk": 0.85,
            "atypical_nevi_burden": 0.10,
            "bcc_risk": 0.92,
            "glioma_risk": 0.0,
            "age_range": (15, 65),
            "mean_age_dx": 28,
            "severity_dist": {"severe": 0.30, "moderate": 0.50, "mild": 0.20},
            "dermoscopy_surveillance": 0.80,
            "sun_protection_compliant": 0.70,
            "pancreatic_surveillance_done": 0.05,
            "immunotherapy_eligible": 0.15,
            "avoid_radiation": 0.95,
            "bapoma_skin_lesion": 0.0,
            "retinoblastoma_risk": 0.0,
            "mesothelioma_risk": 0.0,
            "rcc_risk": 0.01,
            "odontogenic_keratocyst": 0.80,
            "medulloblastoma_risk": 0.04,
            "vismodegib_treatment": 0.45,
        },
    },
    {
        "gene": "SUFU",
        "protein": (
            "SUFU -- 10q24.32 Autosomal-Dominant-LOF -- 484aa -- "
            "Suppressor-of-Fused-SHH-Pathway-Negative-Regulator-Gli-Transcription-Factor-"
            "Gorlin-Like-Syndrome-SHH-MEDULLOBLASTOMA-HIGHER-RISK-PTCH1-"
            "BRAIN-MRI-MANDATORY-Childhood-Adult-BCC-Meningioma-OMIM-607035"
        ),
        "locus": "10q24.32",
        "protein_size": (
            "484 aa / 54 kDa (SUFU; Suppressor of Fused; 10q24.32; "
            "STRUCTURE: "
            "  N-terminal SUFU domain: Gli-binding interface; "
            "  Conserved central region: membrane contact; "
            "  C-terminal domain: cytoplasmic Gli retention; "
            "FUNCTION (SHH PATHWAY NEGATIVE REGULATOR): "
            "  SUFU is a DIRECT inhibitor of Gli transcription factors (Gli1, Gli2, Gli3); "
            "  SUFU binds Gli in the cytoplasm → prevents Gli nuclear translocation → "
            "    Gli target genes NOT transcribed (pathway OFF even without PTCH1); "
            "  Pathway activation: upon SMO activation → Kif7 + SUFU complex dissociates → "
            "    Gli released → Gli2/Gli1 nuclear entry → target gene transcription; "
            "  SUFU LOF: Gli proteins constitutively active (nuclear) → unrestrained SHH target transcription; "
            "GORLIN-LIKE SYNDROME (SUFU): "
            "SHH MEDULLOBLASTOMA: "
            "  SUFU germline LOF: >10% risk of desmoplastic/SHH medulloblastoma (HIGHER than PTCH1 3-5%); "
            "  Age: childhood (18 months - 7yr); posterior fossa; desmoplastic subtype; "
            "  BRAIN MRI MANDATORY: "
            "    Annual MRI brain/spine from age 1 to 10 (some guidelines to age 15-20); "
            "    High sensitivity for early medulloblastoma detection; "
            "  Prognosis: standard-risk SHH medulloblastoma -- 5yr OS ~75-85%; "
            "  SMO inhibitors (vismodegib): some response in SHH medulloblastoma; adult recurrence; "
            "BCC: "
            "  Multiple BCCs similar to Gorlin but less severe burden; "
            "  Age of onset later than PTCH1 (20s-30s vs puberty); "
            "  Fewer in number than PTCH1-Gorlin; "
            "MENINGIOMA: "
            "  ~2% of sporadic meningioma have SUFU germline variant; "
            "  SUFU germline carriers have elevated meningioma risk; "
            "  Annual or biennial brain MRI from age 20-25 for meningioma surveillance; "
            "ODONTOGENIC KERATOCYSTS: "
            "  Less frequent than PTCH1-Gorlin; approximately 20-30% (vs 70-80% in Gorlin); "
            "  OPG recommended in childhood; "
            "CALCIFICATIONS: calcified falx less common than PTCH1; "
            "PTCH1 vs SUFU DISTINCTION: "
            "  Medulloblastoma risk: SUFU HIGHER (>10%) > PTCH1 (3-5%); "
            "  BCC burden: PTCH1 HIGHER (hundreds BCCs from puberty); SUFU fewer, later onset; "
            "  Keratocysts: PTCH1 HIGHER; "
            "  Meningioma: SUFU association; not PTCH1; "
            "  Both: AVOID RADIATION (radiation-induced BCCs + brain tumour risk)."
        ),
        "inheritance": (
            "AD LOF 10q24.32 -- Gorlin-Like SHH Pathway Syndrome. "
            "SHH MEDULLOBLASTOMA >10% (HIGHER THAN PTCH1). BRAIN MRI MANDATORY annually age 1-15. "
            "BCC + meningioma in adults. AVOID RADIATION. Less OKC than PTCH1."
        ),
        "disease_category": "Gorlin-Like SHH Syndrome / SUFU / SHH Medulloblastoma",
        "patient_generator_params": {
            "melanoma_primary_risk": 0.04,
            "uveal_melanoma_risk": 0.01,
            "pancreatic_cancer_risk": 0.01,
            "multiple_primaries_risk": 0.50,
            "atypical_nevi_burden": 0.05,
            "bcc_risk": 0.55,
            "glioma_risk": 0.0,
            "age_range": (5, 65),
            "mean_age_dx": 22,
            "severity_dist": {"severe": 0.35, "moderate": 0.45, "mild": 0.20},
            "dermoscopy_surveillance": 0.75,
            "sun_protection_compliant": 0.68,
            "pancreatic_surveillance_done": 0.05,
            "immunotherapy_eligible": 0.12,
            "avoid_radiation": 0.90,
            "bapoma_skin_lesion": 0.0,
            "retinoblastoma_risk": 0.0,
            "mesothelioma_risk": 0.0,
            "rcc_risk": 0.01,
            "odontogenic_keratocyst": 0.25,
            "medulloblastoma_risk": 0.12,
            "vismodegib_treatment": 0.30,
        },
    },
    {
        "gene": "MITF",
        "protein": (
            "MITF -- 3p14.1 Autosomal-Dominant-GOF-p.E318K -- 526aa -- "
            "Microphthalmia-Associated-Transcription-Factor-Master-Melanocyte-Regulator-"
            "Melanoma-Astrocytoma-Syndrome-E318K-SUMO-Site-Loss-"
            "MODERATE-PENETRANCE-14-20pct-Melanoma-Lifetime-RCC-Meningioma-OMIM-156845"
        ),
        "locus": "3p14.1",
        "protein_size": (
            "526 aa / 59 kDa (MITF; microphthalmia-associated transcription factor; 3p14.1; "
            "STRUCTURE: "
            "  Basic helix-loop-helix leucine zipper (bHLH-LZ) family; "
            "  Homodimerises and heterodimerises (with TFE3, TFEB, TFEC); "
            "  Transactivation domain (TAD): binds CBP/p300; "
            "  E318 residue: SUMOylation site (lysine 316 vicinity -- note: human MITF E318K); "
            "FUNCTION (MASTER MELANOCYTE TRANSCRIPTION FACTOR): "
            "  MELANOGENESIS: activates tyrosinase (TYR), TYRP1, DCT (melanin synthesis enzymes); "
            "  MELANOCYTE SURVIVAL: activates BCL2, CDK2; "
            "  DIFFERENTIATION: activates PMEL (melanosomes), MLANA, RAB27A; "
            "  In melanoma: MITF acts as rheostat: "
            "    High MITF: differentiated, proliferative phenotype; "
            "    Low MITF: invasive, therapy-resistant phenotype (EMT-like); "
            "E318K MUTATION (MELANOMA-PREDISPOSING GOF): "
            "  E318K disrupts SUMO consensus motif (ψKxE: the E becomes K) → "
            "    MITF NO LONGER SUMOylated at K316 → "
            "    Prolonged transcriptional activity (SUMOylation normally limits MITF activity); "
            "  E318K MITF: HIGHER transcriptional activity → "
            "    enhanced melanocyte proliferation AND differentiation target gene expression; "
            "  E318K is a GAIN-OF-FUNCTION: protein still made (not LOF), activity increased; "
            "  E318K allele frequency: ~1-2% in melanoma patients (vs 0.5% general population); "
            "MELANOMA-ASTROCYTOMA SYNDROME: "
            "  Named for co-occurrence of melanoma AND astrocytoma in some MITF E318K families; "
            "  Melanoma: moderate penetrance 14-20% lifetime (lower than CDKN2A); "
            "  Astrocytoma/glioma: elevated but not precisely quantified; "
            "  Not all E318K carriers develop melanoma (incomplete penetrance + genetic modifiers); "
            "RENAL CLEAR CELL CARCINOMA: "
            "  ~10-15% elevated risk in MITF E318K carriers; "
            "  MITF family (MITF/TFE3/TFEB/TFEC) = TFE/MiT family; "
            "  TFE3/TFEB translocations in renal cell carcinoma (paediatric); "
            "  MITF E318K → renal RCC via shared TFE-family biology; "
            "SURVEILLANCE: "
            "  Annual full-body skin examination + dermoscopy; "
            "  Renal ultrasound annually from 30-35; "
            "  Ophthalmology (uveal melanoma rare but elevated); "
            "CLINICAL NOTE: "
            "  E318K is the only well-validated pathogenic MITF melanoma-predisposing variant; "
            "  Other MITF variants (W34X, IVS2+1G>A): cause Waardenburg syndrome type 2A (hearing loss + pigmentation)."
        ),
        "inheritance": (
            "AD GOF 3p14.1 -- Melanoma-Astrocytoma Syndrome. MITF E318K removes SUMOylation → increased transcriptional output. "
            "MODERATE penetrance 14-20% melanoma lifetime. RCC 10-15%. Annual dermoscopy + renal surveillance. "
            "E318K frequency 1-2% in melanoma patients; rare in general population."
        ),
        "disease_category": "Melanoma-Astrocytoma Syndrome / MITF E318K / Moderate Penetrance",
        "patient_generator_params": {
            "melanoma_primary_risk": 0.17,
            "uveal_melanoma_risk": 0.04,
            "pancreatic_cancer_risk": 0.01,
            "multiple_primaries_risk": 0.10,
            "atypical_nevi_burden": 0.35,
            "bcc_risk": 0.04,
            "glioma_risk": 0.08,
            "age_range": (35, 74),
            "mean_age_dx": 52,
            "severity_dist": {"severe": 0.20, "moderate": 0.50, "mild": 0.30},
            "dermoscopy_surveillance": 0.65,
            "sun_protection_compliant": 0.58,
            "pancreatic_surveillance_done": 0.05,
            "immunotherapy_eligible": 0.35,
            "avoid_radiation": 0.0,
            "bapoma_skin_lesion": 0.0,
            "retinoblastoma_risk": 0.0,
            "mesothelioma_risk": 0.0,
            "rcc_risk": 0.12,
            "odontogenic_keratocyst": 0.0,
        },
    },
    {
        "gene": "POT1",
        "protein": (
            "POT1 -- 7q31.33 Autosomal-Dominant-LOF -- 634aa -- "
            "Protection-of-Telomeres-1-Shelterin-Complex-ssDNA-3prime-Overhang-Cap-"
            "Familial-Melanoma-3-4pct-GLIOMA-Elevated-CLL-Telomere-Elongation-"
            "Chromosomal-Instability-Oncogenesis-OMIM-606478"
        ),
        "locus": "7q31.33",
        "protein_size": (
            "634 aa / 70 kDa (POT1; Protection of Telomeres 1; 7q31.33; "
            "STRUCTURE: "
            "  Two OB (oligonucleotide/oligosaccharide binding) folds: OB1 + OB2 (N-terminal DNA binding); "
            "  C-terminal domain: TPP1 interaction (shelterin complex); "
            "  OB3: additional interaction domain; "
            "FUNCTION (SHELTERIN TELOMERE PROTECTION COMPLEX): "
            "  Shelterin complex: TRF1-TRF2-RAP1-TIN2-TPP1-POT1 (6 proteins); "
            "  POT1 specifically binds single-stranded TTAGGG 3' overhang (T-loop structure); "
            "  Functions: "
            "    1. Overhang protection: prevents ATR kinase from recognising 3' overhang as ssDNA damage; "
            "    2. Telomerase regulation: POT1-TPP1 interaction regulates telomerase access and processivity; "
            "    3. Replication: coordinates telomere replication with telomerase; "
            "    4. T-loop stability: maintains T-loop (3' overhang tucked into duplex DNA); "
            "POT1 LOF: "
            "  Loss of 3' overhang protection → ATR activation at telomeres → DNA damage signalling; "
            "  Paradoxically: TELOMERE ELONGATION (not shortening) in POT1-LOF carriers; "
            "    Mechanism: POT1 loss → TPP1 unrestrained → INCREASED telomerase recruitment and processivity; "
            "    Longer telomeres → more cell divisions before senescence → increased oncogenic opportunity; "
            "  Chromosomal instability: fragile telomeres, sister chromatid fusions; "
            "FAMILIAL MELANOMA: "
            "  POT1 variants account for ~3-4% of familial melanoma kindreds (some studies 5-10%); "
            "  TTAGGG-binding domain mutations most pathogenic (OB1/OB2 missense); "
            "  Melanoma lifetime risk: similar penetrance to CDK4; moderate-high; "
            "  Multiple atypical nevi in some kindreds; "
            "GLIOMA: "
            "  Elevated glioma risk in POT1-LOF carriers (glioblastoma + lower-grade glioma); "
            "  Brain MRI surveillance considered in POT1 carriers with family history of brain tumours; "
            "CLL / LYMPHOMA: "
            "  POT1 somatic mutations found in 5% CLL; also Hodgkin lymphoma associations; "
            "  Germline: some families show CLL co-occurrence with melanoma; "
            "ANGIOSARCOMA: rare associations in some kindreds; "
            "THYROID CANCER: papillary thyroid carcinoma in some POT1 families; "
            "SURVEILLANCE: "
            "  Annual full-body skin examination + dermoscopy; "
            "  Brain MRI if family history of glioma (no universal consensus); "
            "  Consider whole-blood count periodically (CLL surveillance)."
        ),
        "inheritance": (
            "AD LOF 7q31.33 -- Familial Melanoma (3-4% of familial melanoma). "
            "POT1 LOF → telomere ELONGATION (paradoxical) + chromosomal instability. "
            "Glioma risk elevated; CLL associations. Annual dermoscopy mandatory."
        ),
        "disease_category": "Familial Melanoma / Telomere Biology / POT1-Shelterin",
        "patient_generator_params": {
            "melanoma_primary_risk": 0.48,
            "uveal_melanoma_risk": 0.03,
            "pancreatic_cancer_risk": 0.02,
            "multiple_primaries_risk": 0.12,
            "atypical_nevi_burden": 0.55,
            "bcc_risk": 0.03,
            "glioma_risk": 0.08,
            "age_range": (30, 72),
            "mean_age_dx": 50,
            "severity_dist": {"severe": 0.30, "moderate": 0.48, "mild": 0.22},
            "dermoscopy_surveillance": 0.68,
            "sun_protection_compliant": 0.58,
            "pancreatic_surveillance_done": 0.08,
            "immunotherapy_eligible": 0.40,
            "avoid_radiation": 0.0,
            "bapoma_skin_lesion": 0.0,
            "retinoblastoma_risk": 0.0,
            "mesothelioma_risk": 0.0,
            "rcc_risk": 0.03,
            "odontogenic_keratocyst": 0.0,
        },
    },
    {
        "gene": "RB1",
        "protein": (
            "RB1 -- 13q14.2 Autosomal-Dominant-LOF -- 928aa -- "
            "Retinoblastoma-Tumour-Suppressor-pRB-E2F-Repressor-Cell-Cycle-G1-S-Checkpoint-"
            "Hereditary-Retinoblastoma-BILATERAL-PATHOGNOMONIC-"
            "TRILATERAL-RB-Pineoblastoma-PATHOGNOMONIC-"
            "SECONDARY-CANCER-30-40pct-Osteosarcoma-Most-Common-AVOID-RADIATION-OMIM-614041"
        ),
        "locus": "13q14.2",
        "protein_size": (
            "928 aa / 110 kDa (RB1; Retinoblastoma protein pRB; 13q14.2; "
            "STRUCTURE: "
            "  RB pocket domain (A/B domains): E2F transcription factor binding; "
            "  Spacer region between A and B: cyclin D binding; "
            "  C-terminal domain: nuclear functions; "
            "  Multiple CDK phosphorylation sites: S249, T252, T356, T373, S608, S612, T787, T821, T826; "
            "FUNCTION: "
            "  HYPOPHOSPHORYLATED pRB (active): binds E2F transcription factors → "
            "    E2F cannot activate S-phase genes → G1 arrest maintained; "
            "  CDK4/6-cyclin D: phosphorylates pRB → releases E2F → S-phase entry; "
            "  CDK2-cyclin E: hyperphosphorylates pRB → commitment to S-phase (restriction point passed); "
            "  pRB also: chromatin remodelling (recruits HDAC), chromosome stability (centromere structure); "
            "HEREDITARY RETINOBLASTOMA: "
            "BILATERAL RETINOBLASTOMA: "
            "  Bilateral tumours = hereditary RB1 until proven otherwise; "
            "  Bilateral incidence: 40% of all retinoblastoma (vs unilateral 60%); "
            "  Age of diagnosis: bilateral YOUNGER (mean 15 months) than unilateral sporadic (mean 24 months); "
            "  EUA (examination under anaesthesia) every 3-6 weeks during active surveillance; "
            "  Treatment: focal therapy (laser/cryotherapy) + intravitreal/intra-arterial chemotherapy; "
            "    Systemic chemotherapy (carboplatin+vincristine+etoposide) for advanced; "
            "    Enucleation: last resort for large tumours with no visual potential; "
            "TRILATERAL RETINOBLASTOMA: "
            "  BILATERAL RB + PINEOBLASTOMA (primitive neuroectodermal midline tumour); "
            "  PATHOGNOMONIC for germline RB1 mutation; "
            "  Incidence: 3-5% of hereditary RB (bilateral cases); "
            "  Onset: 20-36 months; often fatal (5yr OS <10% without aggressive treatment); "
            "  Surveillance: annual brain MRI age 0-5yr for all bilateral/hereditary RB; "
            "GENETICS: "
            "  De novo germline mutation: 15% of cases; "
            "  Inherited: 85% of hereditary cases from an affected parent OR new germline mutation; "
            "  Mosaicism: 10-15% of bilateral RB are mosaic → may have milder phenotype; "
            "  Knudson two-hit model: germline LOF = 1st hit (heterozygous); somatic LOF = 2nd hit → tumour; "
            "SECONDARY CANCERS (HEREDITARY RB SURVIVORS): "
            "  Lifetime risk ~30-40% by age 50 (WITHOUT radiation) → ~50-60% WITH radiation; "
            "  OSTEOSARCOMA most common: proximal femur/distal radius; often radiation-field but NOT exclusively; "
            "  Sarcoma types: osteosarcoma > fibrosarcoma > chondrosarcoma > rhabdomyosarcoma; "
            "  Other: melanoma, lung carcinoma, bladder carcinoma, brain tumours; "
            "AVOID RADIATION: "
            "  Radiation therapy for primary RB in hereditary carriers: "
            "    DRAMATICALLY increases sarcoma risk in radiation field; "
            "    External beam RT historically used → radiation-field osteosarcoma at 10-30yr; "
            "  Modern management: intra-arterial chemotherapy + focal therapy AVOIDS EBRT; "
            "  If XRT unavoidable: minimise field size and dose; "
            "SECONDARY CANCER SURVEILLANCE (HEREDITARY RB SURVIVORS): "
            "  Full-body MRI annually from puberty; "
            "  Annual dermatological examination; "
            "  Orthopaedic surveillance; "
            "  Lifelong oncology follow-up."
        ),
        "inheritance": (
            "AD LOF 13q14.2 -- Hereditary Retinoblastoma. BILATERAL RETINOBLASTOMA PATHOGNOMONIC for germline. "
            "TRILATERAL RB (bilateral + pineoblastoma) HIGHLY SPECIFIC germline RB1. "
            "SECONDARY CANCER 30-40% lifetime (osteosarcoma most common). "
            "AVOID RADIATION ABSOLUTELY in hereditary RB survivors (radiation-field sarcoma). "
            "Knudson two-hit model: germline LOF + somatic LOF = tumour."
        ),
        "disease_category": "Hereditary Retinoblastoma / Secondary Cancer Predisposition",
        "patient_generator_params": {
            "melanoma_primary_risk": 0.08,
            "uveal_melanoma_risk": 0.02,
            "pancreatic_cancer_risk": 0.02,
            "multiple_primaries_risk": 0.35,
            "atypical_nevi_burden": 0.15,
            "bcc_risk": 0.04,
            "glioma_risk": 0.05,
            "age_range": (0, 55),
            "mean_age_dx": 18,
            "severity_dist": {"severe": 0.40, "moderate": 0.45, "mild": 0.15},
            "dermoscopy_surveillance": 0.55,
            "sun_protection_compliant": 0.60,
            "pancreatic_surveillance_done": 0.05,
            "immunotherapy_eligible": 0.20,
            "avoid_radiation": 0.90,
            "bapoma_skin_lesion": 0.0,
            "retinoblastoma_risk": 0.95,
            "mesothelioma_risk": 0.0,
            "rcc_risk": 0.03,
            "odontogenic_keratocyst": 0.0,
            "secondary_sarcoma_risk": 0.30,
        },
    },
]


def _generate_patients_for_gene(gene: str, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene_info = next(g for g in ATLAS_GENES if g["gene"] == gene)
    params = gene_info["patient_generator_params"]

    mutation_sets = {
        "CDKN2A": ["p.G101W", "c.225-243del19", "IVS2-105A>G", "p.R24X", "p.M53I", "c.1_471del", "p.A148T", "p.L16P"],
        "CDK4":   ["p.R24C", "p.R24H", "p.R24C(de_novo)", "p.R24H(familial)", "p.R24C(founder)", "p.R24H(missense)", "p.R24C(het)", "p.R24H(het)"],
        "BAP1":   ["p.L178X", "p.K561X", "c.1754delA", "p.Q682X", "c.2097+1G>A", "p.H94R", "p.E716K", "c.592delC"],
        "PTCH1":  ["c.2383C>T", "p.Q716X", "c.2158insC", "p.R168X", "c.1216C>T", "p.G509R", "IVS13+1G>A", "p.E1121K"],
        "SUFU":   ["p.R389C", "c.1022dupA", "p.Q438X", "p.R419X", "c.467delG", "p.Y466C", "IVS6+1G>T", "p.L325R"],
        "MITF":   ["p.E318K", "p.E318K(index)", "p.E318K(familial)", "p.E318K(de_novo)", "p.E318K(founder)", "p.E318K(het)", "p.E318K(c.952G>A)", "p.E318K(validated)"],
        "POT1":   ["p.R117C", "p.Y36C", "p.I78T", "p.Q94E", "p.Q94del", "p.T120I", "p.L28fs", "p.P446S"],
        "RB1":    ["p.R579X", "c.958+2T>A", "p.R251Q", "p.E137X", "c.607+1G>T", "p.R320W", "exon_7-8_del", "p.L670R"],
    }

    severity_opts = (
        ["severe"] * int(params["severity_dist"]["severe"] * 100) +
        ["moderate"] * int(params["severity_dist"]["moderate"] * 100) +
        ["mild"] * int(params["severity_dist"]["mild"] * 100)
    )
    mutations = mutation_sets.get(gene, ["unknown_variant"])
    age_range = params.get("age_range", (20, 70))

    patients = []
    for i in range(n):
        age_dx = rng.randint(max(0, age_range[0]), age_range[1])
        sev = rng.choice(severity_opts)

        melanoma = rng.random() < params.get("melanoma_primary_risk", 0.0)
        uveal_mel = rng.random() < params.get("uveal_melanoma_risk", 0.0)
        pancreatic = rng.random() < params.get("pancreatic_cancer_risk", 0.0)
        multiple_primaries = rng.random() < params.get("multiple_primaries_risk", 0.0)
        atypical_nevi = rng.random() < params.get("atypical_nevi_burden", 0.0)
        bcc = rng.random() < params.get("bcc_risk", 0.0)
        glioma = rng.random() < params.get("glioma_risk", 0.0)
        rcc = rng.random() < params.get("rcc_risk", 0.0)
        mesothelioma = rng.random() < params.get("mesothelioma_risk", 0.0)
        bapoma = rng.random() < params.get("bapoma_skin_lesion", 0.0)
        retinoblastoma = rng.random() < params.get("retinoblastoma_risk", 0.0)
        odontogenic_keratocyst = rng.random() < params.get("odontogenic_keratocyst", 0.0)
        dermoscopy_done = rng.random() < params.get("dermoscopy_surveillance", 0.0)
        sun_protection = rng.random() < params.get("sun_protection_compliant", 0.0)
        pancreatic_surveillance = rng.random() < params.get("pancreatic_surveillance_done", 0.0)
        immunotherapy = rng.random() < params.get("immunotherapy_eligible", 0.0)
        avoid_radiation = rng.random() < params.get("avoid_radiation", 0.0)
        medulloblastoma = rng.random() < params.get("medulloblastoma_risk", 0.0)
        vismodegib = rng.random() < params.get("vismodegib_treatment", 0.0)
        secondary_sarcoma = rng.random() < params.get("secondary_sarcoma_risk", 0.0)

        patients.append({
            "id": f"{gene}-{i+1:02d}",
            "gene": gene,
            "age_at_diagnosis_yrs": age_dx,
            "mutation": rng.choice(mutations),
            "severity": sev,
            "melanoma_primary": melanoma,
            "uveal_melanoma": uveal_mel,
            "pancreatic_cancer": pancreatic,
            "multiple_primaries": multiple_primaries,
            "atypical_nevi_burden": atypical_nevi,
            "bcc": bcc,
            "glioma": glioma,
            "rcc": rcc,
            "mesothelioma": mesothelioma,
            "bapoma_skin_lesion": bapoma,
            "retinoblastoma": retinoblastoma,
            "odontogenic_keratocyst": odontogenic_keratocyst,
            "dermoscopy_surveillance_done": dermoscopy_done,
            "sun_protection_compliant": sun_protection,
            "pancreatic_surveillance_done": pancreatic_surveillance,
            "immunotherapy_eligible": immunotherapy,
            "avoid_radiation_flag": avoid_radiation,
            "medulloblastoma": medulloblastoma,
            "vismodegib_treatment": vismodegib,
            "secondary_sarcoma": secondary_sarcoma,
        })
    return patients


# ─── API generators ────────────────────────────────────────────────────────────

def generate_overview() -> dict:
    """Overview data for Hereditary-Melanoma-Skin-Cancer-Atlas."""
    return {
        "atlas":          "Hereditary-Melanoma-Skin-Cancer-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary Melanoma and Skin Cancer Predisposition Atlas "
            "(CDKN2A-CDK4-BAP1-PTCH1-SUFU-MITF-POT1-RB1)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "CDKN2A": (
                "AD LOF 9p21.3 (p16^INK4a/p14^ARF; 156aa/132aa; FAMMM; familial melanoma 30-40%; "
                "melanoma lifetime 28-76% geography-dependent; PANCREATIC 17-FOLD RR; "
                "dual tumour suppressor locus -- p16 inhibits CDK4/6; p14ARF stabilises p53 via MDM2; "
                "MLPA essential -- deletions 15-30%)"
            ),
            "CDK4": (
                "AD GOF 12q14.1 (CDK4 p.R24C/H; 303aa; FAMMM type 2; RARE <1%; "
                "R24 = p16-binding pocket -- R24 mutation → p16 cannot bind CDK4 → unrestrained G1→S; "
                "IDENTICAL SURVEILLANCE to CDKN2A; only R24C and R24H are known pathogenic hotspots)"
            ),
            "BAP1": (
                "AD LOF 3p21.1 (BAP1; 729aa; BAP1-TPDS; UVEAL MELANOMA ~50% lifetime; "
                "BAPomas/MBAITs PATHOGNOMONIC (nuclear BAP1 IHC loss confirms); "
                "mesothelioma 8-10%; cutaneous melanoma; RCC; cholangiocarcinoma; "
                "annual ophthalmic + dermatology + CT surveillance)"
            ),
            "PTCH1": (
                "AD LOF 9q22.32 (Patched-1; 1447aa; Gorlin/BCNS; BCCs from puberty PATHOGNOMONIC; "
                "ODONTOGENIC KERATOCYSTS jaw PATHOGNOMONIC; CALCIFIED FALX CEREBRI 65%; "
                "medulloblastoma SHH-type 3-5%; AVOID RADIATION ABSOLUTELY; "
                "vismodegib FDA2012 / sonidegib FDA2015 hedgehog inhibitors)"
            ),
            "SUFU": (
                "AD LOF 10q24.32 (SUFU; 484aa; Gorlin-like SHH syndrome; "
                "SHH MEDULLOBLASTOMA >10% (HIGHER THAN PTCH1); BRAIN MRI MANDATORY age 1-15; "
                "adult BCC + meningioma; less OKC than PTCH1; AVOID RADIATION)"
            ),
            "MITF": (
                "AD GOF 3p14.1 (MITF p.E318K; 526aa; melanoma-astrocytoma syndrome; "
                "E318K removes SUMOylation → increased transcriptional activity; "
                "MODERATE PENETRANCE 14-20% melanoma lifetime; RCC 10-15%; meningioma; "
                "MITF = master melanocyte/melanoma transcription factor)"
            ),
            "POT1": (
                "AD LOF 7q31.33 (POT1; 634aa; shelterin complex ssDNA 3' overhang; "
                "familial melanoma 3-4%; GLIOMA elevated; CLL associations; "
                "POT1 LOF → telomere ELONGATION (paradoxical) + chromosomal instability)"
            ),
            "RB1": (
                "AD LOF 13q14.2 (pRB; 928aa; hereditary retinoblastoma; "
                "BILATERAL RETINOBLASTOMA PATHOGNOMONIC for germline; "
                "TRILATERAL RB (bilateral + pineoblastoma) HIGHLY SPECIFIC; "
                "SECONDARY CANCER 30-40% by age 50 (osteosarcoma most common); "
                "AVOID RADIATION ABSOLUTELY -- radiation-field sarcoma documented)"
            ),
        },
        "key_clinical_rules": [
            "CDKN2A: MLPA ESSENTIAL alongside sequencing -- 15-30% of pathogenic variants are deletions not detected by sequencing alone",
            "CDKN2A/CDK4: Annual full-body dermatoscopy + total body photography from age 10-12; pancreatic EUS+MRI from age 40 (CDKN2A: 17-fold RR)",
            "CDK4: ONLY p.R24C and p.R24H are validated pathogenic -- all other CDK4 variants currently VUS; do not act on VUS",
            "BAP1: BAPomas (MBAITs) PATHOGNOMONIC -- biopsy any unusual spitzoid/epithelioid skin lesion in BAP1 families and confirm nuclear BAP1 IHC loss",
            "BAP1: Annual ophthalmic examination (pupil dilation + indirect ophthalmoscopy) from diagnosis for uveal melanoma surveillance",
            "PTCH1: AVOID RADIATION ABSOLUTELY -- radiation-induced BCCs in radiation field documented; proton therapy for medulloblastoma preferred",
            "PTCH1: Orthopantomogram (OPG) annually from childhood for odontogenic keratocysts (may be asymptomatic jaw lesions)",
            "SUFU: BRAIN MRI MANDATORY annually from age 1-15 for SHH medulloblastoma (>10% risk -- higher than PTCH1)",
            "SUFU: AVOID RADIATION -- same rationale as PTCH1; SHH pathway defect + radiation = synergistic tumour risk",
            "MITF E318K: MODERATE penetrance -- do not assume all E318K carriers will develop melanoma; annual dermoscopy + renal USS",
            "POT1: Annual dermoscopy + glioma surveillance (brain MRI if family history glioma); CLL monitoring with FBC",
            "RB1: BILATERAL RETINOBLASTOMA = germline RB1 until proven otherwise -- molecular testing mandatory in all bilateral cases",
            "RB1: TRILATERAL RB = bilateral RB + pineoblastoma -- brain MRI annual age 0-5yr for all hereditary RB cases",
            "RB1: SECONDARY CANCER surveillance lifelong -- full-body MRI from puberty; osteosarcoma most common secondary malignancy",
            "RB1: AVOID RADIATION for primary RB treatment in hereditary cases -- radiation-field osteosarcoma at 10-30yr; use intra-arterial chemotherapy",
        ],
        "gene_panel_note": (
            "Hereditary melanoma and skin cancer panel (clinical 2024): "
            "HIGH-PENETRANCE MELANOMA: CDKN2A, CDK4, BAP1; "
            "SHH/BCC PATHWAY: PTCH1, SUFU; "
            "MODERATE PENETRANCE MELANOMA: MITF (E318K), POT1; "
            "RETINOBLASTOMA/SECONDARY CANCER: RB1; "
            "CLINICAL DECISION TREE: "
            "  Familial melanoma (>=2 melanoma FDR/SDR): CDKN2A first → CDK4 if negative → POT1; "
            "  Young melanoma (<40yr) + pancreatic: CDKN2A (17x pancreatic RR); "
            "  Uveal melanoma + spitzoid skin lesions: BAP1-TPDS; "
            "  Multiple BCCs (puberty) + jaw keratocysts: PTCH1; "
            "  Childhood SHH medulloblastoma: SUFU first (>10% risk) then PTCH1 (3-5% risk); "
            "  Bilateral retinoblastoma: RB1 always; "
            "  Melanoma + RCC (no family history): MITF E318K; "
            "SURVEILLANCE OVERLAP: "
            "  All genes: annual full-body skin examination + dermoscopy; sun protection mandatory; "
            "  CDKN2A/CDK4/POT1: pancreatic + glioma surveillance; "
            "  BAP1: annual ophthalmology + CT surveillance; "
            "  PTCH1/SUFU: brain MRI (medulloblastoma); OPG (keratocysts); "
            "  RB1: ophthalmology from birth + secondary cancer surveillance from puberty"
        ),
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-Melanoma-Skin-Cancer-Atlas."""
    all_patients = []
    for i, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        patients = _generate_patients_for_gene(gene, SEED_BASE + i)
        all_patients.extend(patients)

    per_gene = {}
    for gene_info in ATLAS_GENES:
        gene = gene_info["gene"]
        pts = [p for p in all_patients if p["gene"] == gene]
        n = len(pts)

        def pct(key):
            return round(sum(1 for p in pts if p.get(key)) / n * 100, 1)

        per_gene[gene] = {
            "gene": gene,
            "locus": gene_info["locus"],
            "disease_category": gene_info["disease_category"],
            "patient_count": n,
            "melanoma_primary_pct": pct("melanoma_primary"),
            "uveal_melanoma_pct": pct("uveal_melanoma"),
            "pancreatic_cancer_pct": pct("pancreatic_cancer"),
            "multiple_primaries_pct": pct("multiple_primaries"),
            "atypical_nevi_pct": pct("atypical_nevi_burden"),
            "bcc_pct": pct("bcc"),
            "glioma_pct": pct("glioma"),
            "rcc_pct": pct("rcc"),
            "mesothelioma_pct": pct("mesothelioma"),
            "bapoma_pct": pct("bapoma_skin_lesion"),
            "retinoblastoma_pct": pct("retinoblastoma"),
            "odontogenic_keratocyst_pct": pct("odontogenic_keratocyst"),
            "dermoscopy_surveillance_pct": pct("dermoscopy_surveillance_done"),
            "sun_protection_pct": pct("sun_protection_compliant"),
            "pancreatic_surveillance_pct": pct("pancreatic_surveillance_done"),
            "immunotherapy_eligible_pct": pct("immunotherapy_eligible"),
            "avoid_radiation_pct": pct("avoid_radiation_flag"),
            "medulloblastoma_pct": pct("medulloblastoma"),
            "vismodegib_treatment_pct": pct("vismodegib_treatment"),
            "secondary_sarcoma_pct": pct("secondary_sarcoma"),
            "mean_age_at_dx": round(sum(p["age_at_diagnosis_yrs"] for p in pts) / n, 1),
            "severity": {
                "severe_pct":   round(sum(1 for p in pts if p["severity"] == "severe") / n * 100, 1),
                "moderate_pct": round(sum(1 for p in pts if p["severity"] == "moderate") / n * 100, 1),
                "mild_pct":     round(sum(1 for p in pts if p["severity"] == "mild") / n * 100, 1),
            },
            "mutation_distribution": {
                m: sum(1 for p in pts if p["mutation"] == m)
                for m in set(p["mutation"] for p in pts)
            },
            "clinical_rule": next(
                (r for r in [
                    "CDKN2A: MLPA essential; pancreatic 17x RR; annual dermoscopy from age 10",
                    "CDK4: only p.R24C/H pathogenic; identical surveillance to CDKN2A",
                    "BAP1: BAPomas PATHOGNOMONIC; annual ophthalmic for uveal melanoma",
                    "PTCH1: AVOID RADIATION; OPG annually; vismodegib for BCC burden",
                    "SUFU: BRAIN MRI MANDATORY age 1-15; medulloblastoma >10% risk",
                    "MITF: E318K only; moderate penetrance; RCC 10-15%; annual dermoscopy",
                    "POT1: familial melanoma 3-4%; glioma elevated; telomere elongation",
                    "RB1: BILATERAL = germline; TRILATERAL PATHOGNOMONIC; AVOID RADIATION; secondary sarcoma 30-40%",
                ] if gene.upper() in r), "Annual dermoscopy mandatory"
            ),
        }

    # Aggregate stats
    total_melanoma = sum(1 for p in all_patients if p["melanoma_primary"])
    total_uveal = sum(1 for p in all_patients if p["uveal_melanoma"])
    total_bcc = sum(1 for p in all_patients if p["bcc"])
    total_rb = sum(1 for p in all_patients if p["retinoblastoma"])
    total_meso = sum(1 for p in all_patients if p["mesothelioma"])
    total_glioma = sum(1 for p in all_patients if p["glioma"])

    return {
        "atlas": "Hereditary-Melanoma-Skin-Cancer-Atlas",
        "total_patients": len(all_patients),
        "per_gene": per_gene,
        "aggregate": {
            "melanoma_primary_total": total_melanoma,
            "melanoma_primary_pct": round(total_melanoma / 320 * 100, 1),
            "uveal_melanoma_total": total_uveal,
            "uveal_melanoma_pct": round(total_uveal / 320 * 100, 1),
            "bcc_total": total_bcc,
            "bcc_pct": round(total_bcc / 320 * 100, 1),
            "retinoblastoma_total": total_rb,
            "retinoblastoma_pct": round(total_rb / 320 * 100, 1),
            "mesothelioma_total": total_meso,
            "mesothelioma_pct": round(total_meso / 320 * 100, 1),
            "glioma_total": total_glioma,
            "glioma_pct": round(total_glioma / 320 * 100, 1),
            "mean_age_at_dx_all_genes": round(
                sum(p["age_at_diagnosis_yrs"] for p in all_patients) / 320, 1
            ),
        },
    }


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-Melanoma-Skin-Cancer-Atlas."""
    definitions = [
        {
            "term": "CDKN2A-FAMMM-Surveillance-Protocol",
            "definition": (
                "CDKN2A Familial Atypical Multiple Mole Melanoma (FAMMM) -- clinical management: "
                "SURVEILLANCE STANDARD: "
                "  Annual full-body skin examination: dermatologist + dermoscopy; "
                "  Total body photography (TBP): baseline at first visit; annual or biennial comparison; "
                "  Sequential digital dermoscopy (SDD): follow atypical nevi over time; "
                "  Begin age 10-12 (or at diagnosis of pathogenic variant, whichever earlier); "
                "  Self-examination education: monthly; "
                "SUN PROTECTION: "
                "  SPF50+ broad-spectrum daily; reapply every 2 hours; "
                "  Avoid midday sun (10am-3pm); "
                "  UV-protective clothing + sunglasses; "
                "  Avoid artificial UV (tanning beds: PROHIBITED); "
                "  Vitamin D supplementation: recommended to compensate for sun avoidance; "
                "ATYPICAL NEVI MANAGEMENT: "
                "  >50 nevi or >1 clinically atypical (ABCDE): classify as high-risk; "
                "  Excise: rapidly changing lesions, dermoscopy-atypical, concerning features; "
                "  Biopsy criteria: ABCDE + Ugly Duckling sign; "
                "PANCREATIC SURVEILLANCE (CDKN2A specifically): "
                "  17-fold relative risk for pancreatic ductal adenocarcinoma; "
                "  EUS annually from age 40-45 (preferred modality: detects small lesions pre-malignant); "
                "  MRI/MRCP: alternating or combined with EUS; "
                "  CA19-9: limited sensitivity as standalone but add to surveillance; "
                "  Consider joining CAPS/EUROPAC registry for expert surveillance centres; "
                "  Smoking cessation MANDATORY (smoking + CDKN2A = additive pancreatic risk); "
                "MELANOMA TREATMENT: "
                "  Surgery: wide local excision (WLE) 1-2cm margins stage I-II; "
                "  Sentinel lymph node biopsy for T1b (0.8-1.0mm) and T2+; "
                "  Adjuvant: anti-PD-1 (nivolumab/pembrolizumab) stage III-IV; "
                "  BRAF V600E: test all advanced melanoma → BRAF/MEK inhibitor (dabrafenib+trametinib) if BRAF positive; "
                "  CDKN2A carriers: targeted therapy eligibility same as sporadic melanoma; "
                "MC1R CO-INHERITANCE: "
                "  CDKN2A + MC1R red-hair variants: 2-3x FURTHER increased melanoma risk; "
                "  Most aggressive risk group: CDKN2A carrier + multiple MC1R variants + sun-exposed."
            ),
        },
        {
            "term": "BAP1-TPDS-Uveal-Melanoma-Management",
            "definition": (
                "BAP1 Tumour Predisposition Syndrome -- uveal melanoma and multi-tumour surveillance: "
                "UVEAL MELANOMA SURVEILLANCE: "
                "  Annual ophthalmic examination: "
                "    Dilated fundus examination (indirect ophthalmoscopy); "
                "    Slit lamp examination of iris + anterior segment; "
                "    Ocular coherence tomography (OCT): macular + choroidal lesions; "
                "    B-scan ultrasound: posterior segment masses; "
                "    Fundus photography: baseline + annual comparison; "
                "  Begin from time of germline BAP1 diagnosis (any age); "
                "  ANY suspicious lesion: urgent uveal oncology referral; "
                "BAP1 IHC IN UVEAL MELANOMA: "
                "  All uveal melanoma biopsies: BAP1 nuclear IHC; "
                "  BAP1 nuclear loss: 50% of UM → Class 2 tumour (monosomy 3 associated); "
                "  Class 2 UM: 50% 5yr metastatic rate; "
                "  BAP1 IHC loss in UM triggers germline testing; "
                "BAPomas (MBAITs): "
                "  PATHOGNOMONIC for germline BAP1 mutation; "
                "  All skin-coloured/pink intradermal spitzoid papules in BAP1 family: biopsy; "
                "  Histology: large epithelioid melanocytes, nuclear BAP1 IHC absent; "
                "  NOT pre-malignant per se (distinct from conventional melanoma-risk naevi); "
                "  Register lesion count; annual dermatology examination; "
                "MESOTHELIOMA SURVEILLANCE: "
                "  Annual/biennial CT chest+abdomen from age 30-35; "
                "  Asbestos exposure history: imperative (multiplicative risk with BAP1); "
                "  Peritoneal mesothelioma: CT abdomen (peritoneal thickening, ascites); "
                "  Serum mesothelin: emerging biomarker; "
                "RCC SURVEILLANCE: "
                "  Annual renal ultrasound from age 30; "
                "  MRI abdomen if ultrasound abnormal; "
                "  BAP1-associated RCC: often clear cell type; good prognosis detected early; "
                "GENETIC COUNSELLING: "
                "  Family cascade testing: all first-degree relatives; "
                "  BAP1 IHC on tumour: cost-effective first-screen in uveal melanoma; "
                "  Germline test: uveal oncology → genetics referral pathway."
            ),
        },
        {
            "term": "Gorlin-BCNS-PTCH1-Radiation-Avoidance",
            "definition": (
                "Gorlin Syndrome (BCNS) -- PTCH1 -- clinical management with critical radiation avoidance: "
                "DIAGNOSIS CRITERIA (Kimonis 2004, modified): "
                "MAJOR CRITERIA (2 major = diagnosis): "
                "  1. >= 2 BCCs before age 20, or >= 5 BCCs at any age; "
                "  2. Odontogenic keratocysts (OKC) jaw (histologically confirmed); "
                "  3. Calcified falx cerebri (early age, <20yr, or lamellar); "
                "  4. Palmar/plantar pits (epidermal ridges absent in pits); "
                "  5. First-degree relative with BCNS; "
                "MINOR CRITERIA (1 major + 2 minor = diagnosis): "
                "  Bifid/fused/extra ribs; vertebral anomalies; frontal bossing; medulloblastoma (SHH type); "
                "  Cardiac fibroma; ovarian fibroma (calcified); "
                "  Macrocephaly (OFC > 97th centile); Sprengel deformity; "
                "  Lamellar calcification of falx (bilateral); "
                "BCC MANAGEMENT: "
                "  Surgical excision: mainstay for individual BCCs; "
                "  Topical: imiquimod (5%), 5-fluorouracil cream (superficial BCCs); "
                "  Photodynamic therapy: small superficial BCCs; "
                "  Vismodegib (150mg daily): for MULTIPLE BCCs (burden reduction in BCNS); "
                "    BCC count reduction; contraindicated pregnancy; "
                "    Toxicities: muscle cramps, dysgeusia, alopecia, fatigue; "
                "    Cyclic use (12 months on, 8 weeks off): reduces toxicity; "
                "  Sonidegib (200mg daily): alternative SMO inhibitor; "
                "OKC (Odontogenic Keratocyst) MANAGEMENT: "
                "  OPG annually from age 8-10; "
                "  Treatment: marsupialization (first) → enucleation; Carnoy's solution; "
                "  High recurrence: 25-60% after simple enucleation; "
                "  OKC ≠ cyst → histology mandatory (aggressive course); "
                "RADIATION AVOIDANCE (CRITICAL): "
                "  EBRT to chest/head/neck in BCNS → radiation-field BCCs within 2-5 years; "
                "  Classic case: BCNS child with medulloblastoma treated with craniospinal irradiation "
                "    → hundreds of BCCs developing throughout radiation field; "
                "  Proton therapy for medulloblastoma: preferred (reduced exit dose); "
                "  No EBRT for BCCs (will not be curative and accelerates new BCC formation); "
                "  Avoid diagnostic X-rays where CT/MRI alternative exists; "
                "MEDULLOBLASTOMA MANAGEMENT: "
                "  Surgical resection: primary treatment; "
                "  Chemotherapy: carboplatin/etoposide/cyclophosphamide (avoid XRT especially <5yr); "
                "  Surveillance: annual brain MRI age 1-15yr; "
                "  Vismodegib/sonidegib: active in SHH medulloblastoma recurrence (SAML trials)."
            ),
        },
        {
            "term": "SUFU-SHH-Medulloblastoma-Surveillance-Protocol",
            "definition": (
                "SUFU Gorlin-Like Syndrome -- SHH Medulloblastoma Surveillance: "
                "SHH MEDULLOBLASTOMA RISK: "
                "  SUFU germline LOF: >10% lifetime risk of SHH-subtype medulloblastoma; "
                "  Risk HIGHER than PTCH1 (3-5%); "
                "  Peak age: 18 months - 7 years (desmoplastic/nodular subtype); "
                "  Also: MBEN (medulloblastoma with extensive nodularity) -- youngest patients; "
                "BRAIN MRI PROTOCOL: "
                "  Annual MRI brain (T1 post-contrast + T2/FLAIR + DWI) from age 1-15yr; "
                "  Age 0-1yr: MRI every 6 months; "
                "  Age 1-10yr: annual MRI; "
                "  Age 10-15yr: annual or biennial (reduce after peak risk period); "
                "  Spine MRI: if brain lesion found (leptomeningeal seeding evaluation); "
                "TREATMENT OF SUFU-ASSOCIATED MEDULLOBLASTOMA: "
                "  Surgery: maximal safe resection; "
                "  Chemotherapy-only protocols (AVOID RADIATION in SUFU germline): "
                "    Baby Brain Tumour Consortium or similar (age <3-5yr): high-dose chemotherapy; "
                "    Vismodegib/sonidegib (SMO inhibitors): active in SHH-MB recurrence; "
                "    AVOID craniospinal XRT in SUFU germline carriers (same radiation-field tumour risk as PTCH1); "
                "MENINGIOMA SURVEILLANCE: "
                "  SUFU germline: elevated meningioma risk (biennial brain MRI from age 20-25 in adults); "
                "ADULT BCC MANAGEMENT: "
                "  Similar to PTCH1 but typically fewer and later onset BCCs; "
                "  Annual dermatological examination; sun protection; vismodegib if burden high; "
                "DISTINCTION FROM PTCH1 GORLIN: "
                "  Medulloblastoma: SUFU HIGHER (>10%) > PTCH1 (3-5%); "
                "  BCC burden: PTCH1 HIGHER (hundreds from puberty) > SUFU (tens, adult); "
                "  OKC: PTCH1 HIGHER (70-80%) > SUFU (20-30%); "
                "  Meningioma: SUFU association; PTCH1 less clear; "
                "TESTING: "
                "  Childhood SHH medulloblastoma: SUFU germline testing FIRST (PTCH1 if SUFU negative); "
                "  Gorlin-like phenotype without OKC or minimal BCC: consider SUFU."
            ),
        },
        {
            "term": "RB1-Hereditary-Retinoblastoma-Secondary-Cancer-Protocol",
            "definition": (
                "Hereditary Retinoblastoma RB1 -- secondary cancer surveillance and management: "
                "PRIMARY RB DIAGNOSIS: "
                "  Bilateral or multifocal RB = GERMLINE RB1 until proven otherwise; "
                "  Unilateral RB: 15% germline (younger age, family history); "
                "  Genetic testing: blood RB1 sequencing + MLPA; if negative → tumour tissue testing; "
                "  Mosaicism: 10-15% hereditary RB → may be missed on blood DNA → "
                "    deep sequencing or tumour-derived DNA testing; "
                "OPHTHALMIC MANAGEMENT: "
                "  EUA (examination under anaesthesia) every 3-6 weeks during active treatment; "
                "  Intra-arterial chemotherapy (IAC): ophthalmic artery catheterisation → "
                "    melphalan ± topotecan ± carboplatin; globe salvage rate ~80%; "
                "  Intravitreal chemotherapy: melphalan for vitreous seeds; "
                "  Focal consolidation: laser photocoagulation, cryotherapy; "
                "  Systemic chemotherapy: carboplatin+vincristine+etoposide (advanced, bilateral); "
                "  Enucleation: last resort for large tumour, no visual potential, no response; "
                "    NEVER first-line for bilateral (vision in at least one eye paramount); "
                "RADIATION AVOIDANCE: "
                "  EBRT in hereditary RB: documents radiation-field osteosarcoma at 10-30yr delay; "
                "  Historical: craniospinal EBRT → sarcoma in every radiation-exposed area; "
                "  Current: EBRT avoided where IAC/focal therapy feasible; "
                "  If EBRT unavoidable: minimal field, proton preferred, documented MDT decision; "
                "TRILATERAL RB: "
                "  Bilateral RB + pineoblastoma (pineal or suprasellar PNET); "
                "  Brain MRI annually age 0-5yr: all bilateral/hereditary RB; "
                "  Prognosis: poor (5yr OS <15% without aggressive treatment); "
                "  Treatment: surgery + high-dose chemotherapy + SCT; extremely challenging; "
                "SECONDARY CANCER SURVEILLANCE (ADULT HEREDITARY RB SURVIVORS): "
                "  Annual full-body MRI from puberty (or 18yr if puberty not established): "
                "    Whole-body MRI: best for soft tissue sarcomas (osteosarcoma, fibrosarcoma); "
                "    Supplemented by local imaging if symptomatic; "
                "  Annual dermatological examination (melanoma elevated); "
                "  Annual FBC + LDH (haematological); "
                "  Bone: orthopaedic examination; DEXA; "
                "  Ophthalmology: annual contralateral eye; annual for treated eye; "
                "SECONDARY SARCOMA TYPES: "
                "  Osteosarcoma most common (proximal femur, distal radius, facial bones in radiation field); "
                "  Fibrosarcoma, leiomyosarcoma, rhabdomyosarcoma; "
                "  Melanoma, lung, bladder, breast cancers also elevated; "
                "  WITHOUT radiation: 30-40% 50yr cumulative risk; "
                "  WITH radiation: 50-60% 50yr cumulative risk (radiation-field risk highest)."
            ),
        },
        {
            "term": "Hereditary-Melanoma-Panel-Testing-Algorithm",
            "definition": (
                "Hereditary melanoma and skin cancer -- multi-gene panel testing algorithm (2024): "
                "INDICATIONS FOR PANEL TESTING: "
                "  >= 2 melanoma in FDR/SDR; "
                "  >= 3 melanoma in extended family; "
                "  Personal history melanoma < 40yr; "
                "  Multiple primary melanomas in same individual (2+ at any age); "
                "  Melanoma + pancreatic cancer (same individual or first-degree relative); "
                "  Bilateral retinoblastoma (RB1 mandatory); "
                "  Multiple BCCs before age 30 + jaw cysts; "
                "  Uveal melanoma + spitzoid skin lesions (BAPomas); "
                "  Melanoma + RCC (same individual or family); "
                "MINIMUM PANEL (suspected hereditary melanoma): CDKN2A, CDK4, BAP1, POT1, MITF-E318K; "
                "EXTENDED PANEL: add PTCH1, SUFU, RB1, BRCA2, TERT, ACD, TERF2IP, MC1R; "
                "INTERPRETATION: "
                "  CDKN2A pathogenic: FAMMM protocol + pancreatic surveillance; "
                "  CDK4 R24C/H: identical to CDKN2A surveillance; "
                "  BAP1 pathogenic: BAP1-TPDS protocol (uveal + BAPoma + mesothelioma + RCC); "
                "  PTCH1/SUFU pathogenic: Gorlin protocol (BCC + OKC + radiation avoidance); "
                "  MITF E318K: moderate penetrance; annual dermoscopy + renal USS; "
                "  POT1 pathogenic: familial melanoma + glioma surveillance; "
                "  RB1 pathogenic: retinoblastoma ophthalmology + secondary cancer surveillance; "
                "VUS MANAGEMENT: "
                "  Do NOT act on VUS for prophylactic surgery or major management change; "
                "  Annual dermoscopy for all VUS carriers in high-risk families; "
                "  Reclassification tracking: ClinVar, InSiGHT, LOVD; "
                "GENETIC COUNSELLING: "
                "  Pre-test: implications (surveillance, family cascade, insurance, psychological); "
                "  Post-test positive: management pathway + cascade testing of first-degree relatives; "
                "  Children: defer CDKN2A/CDK4/POT1/MITF testing to teen years unless surveillance actionable before; "
                "  Paediatric: RB1 + PTCH1 + SUFU testing in childhood where surveillance is actionable."
            ),
        },
        {
            "term": "Melanoma-Immunotherapy-and-Targeted-Therapy-in-Hereditary-Context",
            "definition": (
                "Melanoma treatment in hereditary predisposition context -- immunotherapy and targeted therapy: "
                "IMMUNOTHERAPY (anti-PD-1): "
                "  Pembrolizumab (Keytruda, KEYNOTE-716): adjuvant stage IIB-IV; "
                "  Nivolumab (Opdivo, CheckMate 238): adjuvant stage III-IV; "
                "  ipilimumab+nivolumab (CheckMate 067): advanced/unresectable stage IV; "
                "  TMB-high tumours: pembrolizumab FDA2020 (KEYNOTE-158); "
                "HEREDITARY CONTEXT AND IMMUNOTHERAPY: "
                "  CDKN2A/CDK4 melanoma: standard immunotherapy eligibility (same as sporadic); "
                "  BAP1-associated cutaneous melanoma: PD-1 response similar to sporadic; "
                "  MITF/POT1 melanoma: no specific immunotherapy contraindications; "
                "  RB1-associated secondary melanoma: immunotherapy eligibility depends on comorbidities; "
                "TARGETED THERAPY: "
                "  BRAF V600E/K status: test ALL advanced melanoma; "
                "  BRAF V600E (40-50% melanoma): dabrafenib+trametinib (FDA2013) OR vemurafenib+cobimetinib; "
                "  BRAF V600K: encorafenib+binimetinib (FDA2018); "
                "  NRAS Q61 (15-20%): MEK inhibitor (binimetinib) -- limited response; "
                "  cKIT amplification/mutation (mucosal/acral melanoma 15%): imatinib/sunitinib; "
                "CDKN2A LOSS AND CDK INHIBITORS: "
                "  Palbociclib/ribociclib/abemaciclib (CDK4/6 inhibitors): "
                "    Active in CDK4/6-amplified tumours; "
                "    Conceptually relevant to CDKN2A-LOF tumours (unrestrained CDK4) but clinical trial data limited in melanoma; "
                "    Ongoing trials: CDK4/6 inhibitors in CDKN2A-mutated/amplified melanoma; "
                "HEDGEHOG INHIBITORS IN SKIN CANCER: "
                "  Vismodegib (Erivedge, FDA2012): locally advanced BCC + BCNS (metastatic BCC); "
                "  Sonidegib (Odomzo, FDA2015): locally advanced BCC; "
                "  Both: FDA-approved for BCC; not for basal cell components of other malignancies; "
                "  Teratogenicity: mandatory contraception during and for 24 months (vismodegib) / 20 months (sonidegib) after; "
                "UVEAL MELANOMA TREATMENT (BAP1-TPDS): "
                "  Proton beam therapy / plaque brachytherapy: primary treatment; "
                "  Liver-directed therapy: TACE, Y90 radioembolism for hepatic metastases; "
                "  Tebentafusp (ImmTAC): HLA-A*02:01 restricted; FDA2022 for unresectable UM; "
                "    Bispecific: gp100 TCR x anti-CD3; T-cell redirector; "
                "    BAP1-loss UM: may respond similarly to BAP1-intact (subset analysis limited)."
            ),
        },
    ]
    return {
        "atlas": "Hereditary-Melanoma-Skin-Cancer-Atlas",
        "count": len(definitions),
        "definitions": definitions,
    }
