#!/usr/bin/env python3
"""Hereditary-Brain-CNS-Tumour-Predisposition-Atlas — Complete 8-Gene Hereditary Brain & CNS Tumour Atlas
TP53   (Tumour Protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome;
         Choroid Plexus Carcinoma age <5yr PATHOGNOMONIC; Glioblastoma; Medulloblastoma;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually;
         seed SEED_BASE+0) ·
PTCH1  (Patched-1; 1447aa; 9q22.32; AD LOF;
         Gorlin Syndrome / NBCCS;
         Desmoplastic medulloblastoma SHH-activated PATHOGNOMONIC infratentorial child;
         Calcified falx cerebri PATHOGNOMONIC; OKCs jaw; BCCs hundreds;
         AVOID RADIATION ABSOLUTELY — field cancerisation; Vismodegib/Sonidegib;
         seed SEED_BASE+1) ·
SUFU   (Suppressor of Fused; 484aa; 10q24.32; AD LOF;
         BCNS type 2 / SUFU-related Gorlin;
         Childhood medulloblastoma — SHH-activated desmoplastic/nodular; higher penetrance than PTCH1;
         Fewer BCCs than PTCH1; AVOID RADIATION ABSOLUTELY;
         seed SEED_BASE+2) ·
PTEN   (Phosphatase and Tensin Homolog; 403aa; 10q23.31; AD LOF;
         Cowden Syndrome / PHTS;
         Lhermitte-Duclos disease — cerebellar dysplastic gangliocytoma PATHOGNOMONIC;
         Macrocephaly PATHOGNOMONIC (HC >97th centile); Breast/thyroid/endometrial cancer;
         Everolimus mTOR inhibition;
         seed SEED_BASE+3) ·
APC    (Adenomatous Polyposis Coli; 2843aa; 5q22.2; AD LOF;
         FAP / Turcot Syndrome type 2;
         WNT-activated medulloblastoma — desmoplastic WNT subtype; CHRPE PATHOGNOMONIC in FAP;
         Gardner syndrome: desmoid tumours; osteomas; epidermoid cysts;
         Prophylactic colectomy; annual brain MRI from diagnosis;
         seed SEED_BASE+4) ·
VHL    (Von Hippel-Lindau; 213aa; 3p25.3; AD LOF;
         VHL Disease;
         CNS hemangioblastoma PATHOGNOMONIC (cerebellum/brainstem/spinal cord);
         Retinal hemangioblastoma PATHOGNOMONIC — annual ophthalmology from age 1;
         ccRCC; phaeochromocytoma; Belzutifan HIF-2α inhibitor FDA 2021;
         seed SEED_BASE+5) ·
SMARCB1 (SWI/SNF Related Matrix Associated Actin Dependent Regulator of Chromatin B1; 385aa; 22q11.23; AD LOF;
         Rhabdoid Tumour Predisposition Syndrome type 2 (RTPS2);
         AT/RT under age 3 years PATHOGNOMONIC — INI1 IHC loss PATHOGNOMONIC;
         Rhabdoid meningioma; Schwannomatosis-2;
         SIBLING surveillance mandatory — risk recurrence in family;
         seed SEED_BASE+6) ·
PMS2   (PMS1 Homolog 2 Mismatch Repair; 862aa; 7p22.2; AR biallelic/AD monoallelic;
         Constitutional Mismatch Repair Deficiency (CMMRD) biallelic;
         Glioblastoma / medulloblastoma in children — hypermutation PATHOGNOMONIC (>100 mut/Mb);
         Café au lait macules + brain tumour in child = CMMRD until excluded;
         PD-1 immunotherapy — hypermutated CNS tumours respond;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3150-3157)
"""
import random

SEED_BASE = 3150

ATLAS_GENES = [
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "Tumour-Protein-p53-Guardian-Genome-43kDa-"
            "Li-Fraumeni-Syndrome-CPC-Age5-PATHOGNOMONIC-Glioblastoma-Medulloblastoma-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 17p13.1 TP53 encodes p53 (tumour protein p53 / p53 / TP53): "
            "STRUCTURE: "
            "  N-terminal transactivation domain 1 (TAD1, aa 1-40): MDM2-interaction site; "
            "  Transactivation domain 2 (TAD2, aa 40-67): second activation subdomain; "
            "  Proline-rich region (aa 67-98): PXXP motifs; apoptosis regulation; "
            "  Central sequence-specific DNA-binding domain (DBD, aa 102-292): "
            "    Most hotspot mutations cluster here (R175H, G245S, R248Q, R248W, R249S, R273H, R273C, R282W); "
            "  Tetramerisation domain (TD, aa 323-356): p53 forms functional homotetramers; "
            "  C-terminal regulatory domain (aa 356-393): lysine acetylation/ubiquitination; "
            "FUNCTION: "
            "  Transcription factor activated by genotoxic/oncogenic stress; "
            "  Transactivates CDKN1A (p21), MDM2, GADD45, PUMA, NOXA, BAX; "
            "  Cell-cycle arrest (G1/S via p21), apoptosis, senescence; "
            "  Tumour suppressor — loss-of-function dominant negative or haploinsufficiency; "
            "LFS AND CNS TUMOURS: "
            "  Li-Fraumeni Syndrome (LFS): germline TP53 heterozygous pathogenic variant; "
            "  CNS tumour types in LFS: glioblastoma (adult), medulloblastoma, ependymoma, choroid plexus tumours; "
            "  Choroid Plexus Carcinoma (CPC) in child <5yr: PATHOGNOMONIC for germline TP53 (>50% CPC harbour germline TP53); "
            "  Choroid Plexus Papilloma (CPP) in infant: test TP53 germline; "
            "  Glioblastoma <30yr or multiple primaries: test TP53 germline; "
            "  Medulloblastoma LFS component: non-WNT non-SHH (Group 3/4) — rarer but documented; "
            "RADIATION — ABSOLUTE CONTRAINDICATION: "
            "  Standard-dose radiotherapy in germline TP53 carriers: "
            "    Massive radiation-induced sarcoma risk in RT field; "
            "    Multiple case reports of RT-induced malignancy in LFS within 5-10yr; "
            "    Principle: ionising radiation is a direct mutagen — TP53-null repair → carcinogenesis; "
            "    AVOID ALL THORACIC, WHOLE-BRAIN, SPINAL RT — standard + high dose alike; "
            "  Proton therapy or photon SBRT NOT safe for LFS: still ionising radiation; "
            "  Exception: emergency brain herniation where RT is sole option — discuss risk with family explicitly; "
            "  Re-irradiation: NEVER in LFS; "
            "SURVEILLANCE — TORONTO WBMRI PROTOCOL: "
            "  WBMRI annually from diagnosis — whole body MRI (brain + spine + body); "
            "  Toronto protocol (Kim et al. 2010, updated 2021): "
            "    Children <18yr: WBMRI + brain MRI every 12 months; "
            "    Adults: WBMRI annually; abdominal USS 6-monthly (adrenal/ACC); "
            "    Brain MRI 6-monthly in children (primary CNS site); "
            "    Breast MRI annually from age 20-25; "
            "  Choroid plexus tumour under 5yr → IMMEDIATE germline TP53 + whole CNS staging"
        ),
        "inheritance": (
            "AD LOF 17p13.1 — TP53. Germline TP53 pathogenic variant: LFS. "
            "Penetrance: >90% lifetime cancer risk (any cancer). "
            "De novo germline TP53 in ~7-20% of LFS probands (not inherited). "
            "Dominant negative mechanism: mutant p53 oligomerises with WT p53 — inactivates tetramers. "
            "Haploinsufficiency also contributes (LOH common in tumours). "
            "Prevalence: 1 per 5,000-20,000 (LFS); exact figure uncertain due to incomplete penetrance."
        ),
        "surveillance_key": (
            "WBMRI annually (Toronto protocol); brain MRI 6-monthly in children; "
            "abdominal USS 6-monthly (ACC); breast MRI age 20-25; "
            "AVOID ALL IONISING RADIATION; choroid plexus tumour <5yr → immediate germline TP53"
        ),
        "pathognomonic": (
            "Choroid Plexus Carcinoma age <5yr = PATHOGNOMONIC LFS (>50% CPC = germline TP53); "
            "ACC under age 5yr = PATHOGNOMONIC LFS component; "
            "Multiple primary cancers + family history = LFS pattern"
        ),
    },
    {
        "gene": "PTCH1",
        "protein": (
            "PTCH1 -- 9q22.32 Autosomal-Dominant-LOF -- 1447aa -- "
            "Patched-1-Hedgehog-Receptor-160kDa-"
            "Gorlin-Syndrome-NBCCS-Desmoplastic-Medulloblastoma-SHH-PATHOGNOMONIC-"
            "Calcified-Falx-PATHOGNOMONIC-AVOID-RADIATION-BCCs-Hundreds-Vismodegib-OMIM-601309"
        ),
        "locus": "9q22.32",
        "protein_size": (
            "1447 aa / 9q22.32 PTCH1 encodes Patched-1 (Patched homolog 1): "
            "STRUCTURE: "
            "  12-pass transmembrane receptor (sterol-sensing domain SSD, aa 576-798); "
            "  Two large extracellular loops (EL1, EL2): Hedgehog (Hh) ligand binding; "
            "  C-terminal intracellular tail: signal transduction; "
            "  Patched protein pumps cholesterol-modified Hh away from SMO; "
            "HEDGEHOG PATHWAY: "
            "  Unliganded: PTCH1 inhibits Smoothened (SMO) → GLI1/2 transcription REPRESSED; "
            "  Liganded (SHH/IHH/DHH binds PTCH1): PTCH1 no longer inhibits SMO → SMO activates → GLI1/2 → target genes; "
            "  Target genes: PTCH1 itself (feedback), GLI1, CCND1, MYC; "
            "  SHH pathway drives ~25-30% of all medulloblastomas (SHH subgroup); "
            "GORLIN SYNDROME / NBCCS: "
            "  Naevoid Basal Cell Carcinoma Syndrome (NBCCS / Gorlin-Goltz syndrome); "
            "  Germline PTCH1 loss-of-function: autosomal dominant, >95% penetrance for BCCs by 40yr; "
            "  MEDULLOBLASTOMA: "
            "    Desmoplastic/nodular medulloblastoma, SHH-activated, infratentorial — PATHOGNOMONIC in PTCH1/Gorlin; "
            "    Age: childhood peak 1-5yr (earlier than sporadic SHH-MB 3-15yr); "
            "    PTCH1-associated MB: typically desmoplastic/nodular — better prognosis than WNT/non-SHH; "
            "    Annual brain MRI from Gorlin diagnosis (childhood); "
            "    Lifetime MB risk in Gorlin: ~2-5%; "
            "  BASAL CELL CARCINOMAS: "
            "    Hundreds of BCCs — onset as early as puberty; "
            "    Jaw odontogenic keratocysts (OKCs): multiple jaw OKCs PATHOGNOMONIC; "
            "    Calcified falx cerebri on imaging: PATHOGNOMONIC for Gorlin/PTCH1; "
            "    Bifid ribs: characteristic skeletal anomaly; "
            "    Ovarian fibromas (bilateral in Gorlin: PATHOGNOMONIC); "
            "  AVOID RADIATION ABSOLUTELY: "
            "    RT-induced BCC field: thousands of BCCs in irradiated field within months-years; "
            "    Gorlin + RT = catastrophic BCC burden → disfigurement, death; "
            "    Standard WBRT for MB in Gorlin child: ABSOLUTELY CONTRAINDICATED; "
            "    Use chemotherapy-only protocol (e.g. Baby Brain protocol — avoid RT under 5yr); "
            "    AVOID dental X-rays in bulk; use shielding; minimise jaw radiation; "
            "  VISMODEGIB / SONIDEGIB (SMO inhibitors): "
            "    FDA 2012 vismodegib (Erivedge) for locally advanced/metastatic BCC; "
            "    SMO inhibitor: blocks SMO activity (downstream of PTCH1 loss); "
            "    Reduces BCC burden in Gorlin; cisplatin/vincristine medulloblastoma first (avoid RT); "
            "    Resistance: SMO W535L mutation; "
            "SURVEILLANCE: "
            "  Dermatology every 6 months (BCCs); "
            "  Brain MRI annually from diagnosis through age 10; "
            "  Panoramic jaw X-ray every 12-18 months (OKC); "
            "  Pelvic USS (ovarian fibromas); "
            "  Ophthalmology (colobomata, strabismus)"
        ),
        "inheritance": (
            "AD LOF 9q22.32 — PTCH1. Gorlin syndrome. "
            "Penetrance: >95% BCCs by age 40 in carriers. MB risk ~2-5%. "
            "De novo rate ~20-30% of Gorlin probands. "
            "Second hit (LOH) required for tumorigenesis (two-hit Knudson model). "
            "Prevalence: 1 per 31,000-40,000. "
            "Genotype-phenotype: truncating variants → higher BCC burden; missense → milder."
        ),
        "surveillance_key": (
            "Dermatology 6-monthly (BCCs); brain MRI annually through age 10 (MB); "
            "jaw panorex 12-18 monthly (OKC); AVOID ALL RADIATION ABSOLUTELY; "
            "vismodegib/sonidegib for BCC burden; Baby Brain protocol chemo-only for MB"
        ),
        "pathognomonic": (
            "Calcified falx cerebri on CT/MRI = PATHOGNOMONIC Gorlin/PTCH1; "
            "Desmoplastic/nodular medulloblastoma SHH-activated in child = PTCH1/Gorlin pattern; "
            "Multiple jaw OKCs = PATHOGNOMONIC Gorlin; "
            "Bilateral ovarian fibromas in young woman = PATHOGNOMONIC Gorlin"
        ),
    },
    {
        "gene": "SUFU",
        "protein": (
            "SUFU -- 10q24.32 Autosomal-Dominant-LOF -- 484aa -- "
            "Suppressor-of-Fused-Hedgehog-Pathway-60kDa-"
            "BCNS2-Gorlin-Type2-Childhood-Medulloblastoma-SHH-Higher-Penetrance-PTCH1-"
            "Fewer-BCCs-AVOID-RADIATION-ABSOLUTELY-OMIM-607035"
        ),
        "locus": "10q24.32",
        "protein_size": (
            "484 aa / 10q24.32 SUFU encodes Suppressor of Fused (SUFU): "
            "STRUCTURE: "
            "  N-terminal domain (aa 1-125): GLI-binding; "
            "  Middle linker (aa 126-268); "
            "  C-terminal domain (aa 269-484): SPOP-binding, degradation signalling; "
            "  SUFU forms cytoplasmic complex with GLI proteins; "
            "FUNCTION IN HEDGEHOG PATHWAY: "
            "  Acts downstream of SMO — directly inhibits GLI1 and GLI2 transcription factors; "
            "  Sequesters GLI in cytoplasm → prevents nuclear translocation → represses Hh targets; "
            "  Two-hit Knudson tumour suppressor mechanism; "
            "  Distinct from PTCH1: SUFU loss constitutively activates Hh signalling regardless of SMO; "
            "SUFU-RELATED GORLIN (BCNS TYPE 2): "
            "  Germline SUFU heterozygous pathogenic variant — phenotypically overlaps Gorlin/NBCCS; "
            "  MEDULLOBLASTOMA — HIGHEST PENETRANCE: "
            "    SHH-activated desmoplastic/nodular or MBEN (MB with extensive nodularity) — PATHOGNOMONIC pattern; "
            "    SUFU-MB penetrance for MB: ~15-20% (HIGHER than PTCH1 ~2-5%); "
            "    Age: infants/toddlers <3yr — earlier than PTCH1-MB; "
            "    MBEN histology (MB with extensive nodularity): almost exclusively SUFU or PTCH1 germline; "
            "    Excellent prognosis with chemo-only (no RT under 5yr); "
            "  BCCs: FEWER than PTCH1 (SUFU BCCs develop later, less florid); "
            "  OKCs: jaw OKCs present but fewer; "
            "  Calcified falx: less common than PTCH1; "
            "SURVEILLANCE: "
            "  MRI brain and spine at diagnosis; "
            "  Brain MRI 6-monthly for first 5yr of life (highest MB risk infancy-5yr); "
            "  Brain MRI annually thereafter to age 10; "
            "  Dermatology annually for BCC screening; "
            "  Jaw panorex 2-yearly (OKC screening); "
            "  AVOID ALL RADIATION ABSOLUTELY — same rationale as PTCH1/Gorlin; "
            "TREATMENT: "
            "  Medulloblastoma: chemotherapy-only protocol for infants/toddlers (AVOID RT); "
            "  Vismodegib/sonidegib: potentially active (SMO downstream of SUFU); "
            "  MBEN in SUFU child: excellent chemo-only outcomes (5yr OS >90%)"
        ),
        "inheritance": (
            "AD LOF 10q24.32 — SUFU. BCNS2 / SUFU-related Gorlin. "
            "MB penetrance ~15-20% (higher than PTCH1 ~2-5%). "
            "De novo rate not well characterised — many families with isolated MB index case. "
            "Second hit (LOH at 10q24) in medulloblastoma confirms Knudson model. "
            "Prevalence: rarer than PTCH1 Gorlin; exact figure unknown (likely 1:100,000+)."
        ),
        "surveillance_key": (
            "Brain MRI 6-monthly to age 5 (peak MB risk infancy); brain MRI annually to age 10; "
            "dermatology annually (BCCs); jaw panorex 2-yearly (OKC); "
            "AVOID ALL RADIATION ABSOLUTELY; chemo-only MB protocol"
        ),
        "pathognomonic": (
            "MBEN (medulloblastoma with extensive nodularity) in infant = PATHOGNOMONIC SUFU/PTCH1 germline; "
            "SHH-activated MB in child <2yr = SUFU/PTCH1 germline testing mandatory; "
            "SUFU: higher MB penetrance than PTCH1 — MB may be first presenting feature"
        ),
    },
    {
        "gene": "PTEN",
        "protein": (
            "PTEN -- 10q23.31 Autosomal-Dominant-LOF -- 403aa -- "
            "Phosphatase-and-Tensin-Homolog-47kDa-PI3K-Phosphatase-"
            "Cowden-PHTS-Lhermitte-Duclos-Cerebellar-Gangliocytoma-PATHOGNOMONIC-"
            "Macrocephaly-PATHOGNOMONIC-Everolimus-mTOR-OMIM-601728"
        ),
        "locus": "10q23.31",
        "protein_size": (
            "403 aa / 10q23.31 PTEN encodes PTEN (phosphatase and tensin homolog): "
            "STRUCTURE: "
            "  N-terminal PIP2-binding module (aa 1-12); "
            "  Phosphatase domain (aa 14-185): catalytic cysteine C124; lipid and protein phosphatase activity; "
            "  C2 domain (aa 186-351): membrane association; "
            "  C-terminal regulatory tail (aa 352-403): PDZ-binding, phosphorylation, ubiquitination; "
            "FUNCTION: "
            "  Primary: dephosphorylates PIP3 → PIP2; antagonises PI3K → inhibits AKT-mTOR pathway; "
            "  PTEN loss → constitutive PI3K/AKT/mTOR signalling → proliferation, survival, angiogenesis; "
            "  Nuclear PTEN: maintains chromosomal stability independently of phosphatase activity; "
            "PTEN HAMARTOMA TUMOUR SYNDROME (PHTS / COWDEN): "
            "  Germline PTEN pathogenic variant: Cowden syndrome, Bannayan-Riley-Ruvalcaba, Proteus-like; "
            "  CNS HALLMARK — LHERMITTE-DUCLOS DISEASE: "
            "    Dysplastic cerebellar gangliocytoma: diffuse hamartomatous overgrowth of cerebellar cortex; "
            "    MRI: striated/tigroid pattern of cerebellar cortex — PATHOGNOMONIC radiological appearance; "
            "    Lhermitte-Duclos = PATHOGNOMONIC for adult-onset Cowden/PHTS (adult onset distinguishes from childhood cerebellar tumours); "
            "    Presents: progressive cerebellar ataxia, raised ICP, hydrocephalus; "
            "    Not a malignant neoplasm — benign hamartoma; but local mass effect can be fatal; "
            "    Surgical resection if symptomatic; recurrence possible; "
            "  MACROCEPHALY: "
            "    HC >97th centile (>2SD above mean): PATHOGNOMONIC screening criterion for PHTS; "
            "    Macrocephaly + breast cancer at any age → PTEN germline testing; "
            "    Macrocephaly + autism spectrum disorder → PTEN germline testing (PTEN ASD link); "
            "  BREAST CANCER: lifetime risk 77-85% in female PTEN carriers; "
            "  THYROID CANCER: follicular thyroid carcinoma 35-38% lifetime; "
            "  ENDOMETRIAL CANCER: 28-44% lifetime; "
            "  EVEROLIMUS / mTOR THERAPY: "
            "    mTOR activation central to PTEN-loss biology; "
            "    Everolimus (Afinitor) FDA: PTEN-mutant breast Ca, RCC, SEGA (TSC/PTEN); "
            "    Ongoing trials: everolimus for PTEN-driven brain hamartomas, LDD; "
            "SURVEILLANCE: "
            "  Brain MRI at initial evaluation (Lhermitte-Duclos screen); "
            "  Repeat if new cerebellar symptoms; "
            "  Annual breast MRI from age 25-30; "
            "  Annual thyroid USS; "
            "  Annual endometrial USS + endometrial sampling 35yr; "
            "  Annual dermatology (tricholemmomas, papules)"
        ),
        "inheritance": (
            "AD LOF 10q23.31 — PTEN. PHTS (PTEN Hamartoma Tumour Syndrome). "
            "Penetrance: lifetime cancer risk >85% (breast, thyroid, endometrial). "
            "Lhermitte-Duclos penetrance unknown — likely <25% of PHTS carriers develop symptomatic LDD. "
            "De novo rate ~10-15% of PHTS probands. "
            "Prevalence: 1 per 200,000-250,000 (Cowden syndrome phenotype). "
            "Genotype-phenotype: promoter variants + large deletions → highest risk."
        ),
        "surveillance_key": (
            "Brain MRI at diagnosis (LDD screen); annual breast MRI from age 25; "
            "annual thyroid USS; endometrial sampling from age 35; "
            "Lhermitte-Duclos = cerebellar ataxia + striped MRI pattern PATHOGNOMONIC; "
            "macrocephaly (>97th centile) = first screening criterion"
        ),
        "pathognomonic": (
            "Lhermitte-Duclos (dysplastic cerebellar gangliocytoma): striated/tigroid MRI pattern = PATHOGNOMONIC adult PHTS; "
            "Macrocephaly HC >97th centile = PATHOGNOMONIC screening threshold for PTEN germline; "
            "Tricholemmomas (face) + papillomatous papules (mucosal) = PATHOGNOMONIC Cowden"
        ),
    },
    {
        "gene": "APC",
        "protein": (
            "APC -- 5q22.2 Autosomal-Dominant-LOF -- 2843aa -- "
            "Adenomatous-Polyposis-Coli-310kDa-WNT-Pathway-Gatekeeper-"
            "FAP-Turcot-Type2-WNT-Medulloblastoma-CHRPE-PATHOGNOMONIC-"
            "Gardner-Desmoid-Prophylactic-Colectomy-Annual-Brain-MRI-OMIM-611731"
        ),
        "locus": "5q22.2",
        "protein_size": (
            "2843 aa / 5q22.2 APC encodes APC (adenomatous polyposis coli protein): "
            "STRUCTURE: "
            "  Oligomerisation domain (aa 1-58): APC homodimerisation; "
            "  Armadillo repeats (aa 453-767): protein-protein interaction; "
            "  β-catenin binding regions (15-aa and 20-aa repeats, aa 1020-2130); "
            "  SAMP repeats (aa 1440-2130): Axin interaction; "
            "  C-terminal basic domain (aa 2200-2843): microtubule, EB1 interaction; "
            "FUNCTION: "
            "  APC is a scaffold in the β-catenin destruction complex (with Axin, GSK3β, CK1); "
            "  APC sequesters β-catenin → phosphorylation → ubiquitination → proteasomal degradation; "
            "  APC loss → β-catenin accumulates → nuclear translocation → TCF/LEF → WNT target genes; "
            "  WNT targets: CCND1, MYC, AXIN2, LGR5 — proliferation/stem cell; "
            "FAP AND TURCOT SYNDROME TYPE 2 (BRAIN): "
            "  Familial Adenomatous Polyposis (FAP): thousands of colonic polyps by age 20-30; "
            "  Turcot syndrome type 2: FAP + CNS tumour (classically medulloblastoma); "
            "  MEDULLOBLASTOMA — WNT SUBTYPE: "
            "    WNT-activated medulloblastoma: most favourable prognosis of all MB subtypes (5yr OS >90%); "
            "    APC germline in ~10-15% of WNT-MB patients; "
            "    β-catenin nuclear IHC staining: PATHOGNOMONIC for WNT-MB (monosomy 6 also characteristic); "
            "    Annual brain MRI from FAP diagnosis (particularly in children/teens); "
            "    MB in FAP child may be first presenting cancer (before colonic polyps manifest); "
            "  CHRPE (Congenital Hypertrophy of Retinal Pigment Epithelium): "
            "    Multiple bilateral CHRPE lesions: PATHOGNOMONIC for FAP/APC germline; "
            "    Ophthalmoscopy: hyperpigmented oval retinal lesions; "
            "    FAP-CHRPE: >4 lesions bilateral, or characteristic fishhook/bear-track pattern; "
            "    CHRPE requires NO treatment — purely diagnostic marker; "
            "  GARDNER SYNDROME (APC variant): "
            "    FAP + desmoid tumours (aggressive fibromatosis) + osteomas + epidermoid cysts; "
            "    Desmoid tumours: intra-abdominal — complication of colectomy; can be fatal; "
            "    COX-2 inhibitor celecoxib: modestly reduces desmoid growth; sulindac reduces polyps; "
            "SURVEILLANCE: "
            "  Annual colonoscopy from age 10-12 or diagnosis; "
            "  Prophylactic colectomy before age 20-25 (proctocolectomy or IPAA); "
            "  Annual upper endoscopy (duodenal adenomas — Spigelman stage); "
            "  Annual brain MRI from diagnosis through adulthood (MB risk, though rare); "
            "  CHRPE ophthalmoscopy (diagnostic, no treatment); "
            "  Annual thyroid USS (papillary thyroid carcinoma 2-3× in FAP)"
        ),
        "inheritance": (
            "AD LOF 5q22.2 — APC. FAP / Gardner / Turcot type 2. "
            "Penetrance: >95% develop colonic polyps; colorectal cancer risk ~100% without colectomy. "
            "Brain tumour (MB) risk: ~1% of FAP patients. "
            "De novo rate ~25-30% of FAP probands (no family history). "
            "Prevalence: 1 per 8,000-10,000 (FAP). "
            "Genotype-phenotype: codons 1250-1464 → profuse polyposis + desmoid risk; "
            "codon 1309 AAAG deletion → most severe (earliest onset, >5,000 polyps)."
        ),
        "surveillance_key": (
            "Annual colonoscopy from age 10-12; prophylactic colectomy before age 25; "
            "upper endoscopy (duodenal adenomas); annual brain MRI from diagnosis (MB/WNT); "
            "annual thyroid USS; CHRPE ophthalmoscopy (diagnostic)"
        ),
        "pathognomonic": (
            "Multiple bilateral CHRPE lesions (>4) = PATHOGNOMONIC FAP/APC germline; "
            "β-catenin nuclear IHC in medulloblastoma = PATHOGNOMONIC WNT-MB (APC/CTNNB1); "
            "FAP + medulloblastoma = Turcot syndrome type 2 (APC pathway)"
        ),
    },
    {
        "gene": "VHL",
        "protein": (
            "VHL -- 3p25.3 Autosomal-Dominant-LOF -- 213aa -- "
            "Von-Hippel-Lindau-pVHL-24kDa-HIF-E3-Ubiquitin-Adaptor-"
            "VHL-Disease-CNS-Hemangioblastoma-PATHOGNOMONIC-Retinal-Annual-Age1-"
            "ccRCC-Phaeochromocytoma-Belzutifan-HIF2alpha-FDA2021-OMIM-608537"
        ),
        "locus": "3p25.3",
        "protein_size": (
            "213 aa / 3p25.3 VHL encodes pVHL (Von Hippel-Lindau tumour suppressor protein): "
            "STRUCTURE: "
            "  α domain (α-helices, aa 155-213): binds Elongin B/C; "
            "  β domain (β-barrel, aa 63-154): binds HIF-1α/2α oxygen-dependent degradation domain (ODD); "
            "  N-terminal disordered region (aa 1-62); "
            "  pVHL forms VBC complex with Elongin B + Elongin C + Cullin-2 + RBX1 (E3 ubiquitin ligase); "
            "FUNCTION: "
            "  pVHL = substrate recognition subunit of E3 ubiquitin ligase; "
            "  Under normoxia: PHD hydroxylates HIF-α Pro402/Pro564 → pVHL binds → ubiquitination → proteasomal degradation; "
            "  VHL loss: HIF-1α and HIF-2α stabilised → nuclear accumulation → HIF target genes; "
            "  HIF targets: VEGF, PDGF, EPO, GLUT1, SLC2A1 → angiogenesis, erythropoiesis, glycolysis; "
            "VHL DISEASE — CNS MANIFESTATIONS: "
            "  TYPE 1 VHL (truncating/missense — no Phe motif): "
            "    Hemangioblastoma CNS + retina; ccRCC; pancreatic cysts; NO phaeochromocytoma; "
            "  TYPE 2 VHL (missense with intact Phe motif): "
            "    Type 2A: hemangioblastoma + phaeochromocytoma; low RCC; "
            "    Type 2B: hemangioblastoma + phaeochromocytoma + ccRCC; "
            "    Type 2C: phaeochromocytoma ONLY; "
            "  CNS HEMANGIOBLASTOMA — PATHOGNOMONIC: "
            "    Multiple CNS hemangioblastomas (cerebellum, brainstem, spinal cord, supratentorial): PATHOGNOMONIC VHL; "
            "    Sporadic single hemangioblastoma: test VHL germline; "
            "    Multiple or recurrent CNS hemangioblastomas: VHL germline until excluded; "
            "    MRI: enhancing cystic + solid mural nodule — rich vascular supply; "
            "    Management: observe small (<3cm asymptomatic); surgery or SRS for symptomatic/growing; "
            "  RETINAL HEMANGIOBLASTOMA — PATHOGNOMONIC: "
            "    Retinal capillary hemangioblastoma (retinal angioma): bilateral/multiple = PATHOGNOMONIC VHL; "
            "    Annual ophthalmoscopy from age 1 year: "
            "    WHY age 1: retinal hemangioblastoma can present in infants → risk of retinal detachment → blindness; "
            "    Laser photocoagulation / intravitreal anti-VEGF for small lesions; "
            "    Early detection: prevents vision-threatening complications; "
            "BELZUTIFAN (HIF-2α inhibitor): "
            "  Mechanism: belzutifan (Welireg, Merck) inhibits HIF-2α — blocks HIF-2α/ARNT dimerisation; "
            "  FDA approval 2021: VHL disease-associated RCC, CNS hemangioblastoma, pancreatic neuroendocrine tumour; "
            "  Phase 3 LITESPARK-004: belzutifan for VHL-associated hemangioblastoma — ORR 49% CNS hemangioblastoma; "
            "  First systemic therapy for VHL disease (previously surgical only); "
            "  Dose: 120mg once daily; adverse effects: anaemia, fatigue, dizziness; "
            "SURVEILLANCE: "
            "  Annual retinal ophthalmoscopy from age 1; "
            "  Annual brain+spine MRI from age 11 (or diagnosis); "
            "  Annual abdominal MRI/USS (ccRCC, pancreatic lesions, phaeochromocytoma); "
            "  Annual 24hr urine catecholamines/plasma metanephrines (phaeochromocytoma); "
            "  Annual audiogram (endolymphatic sac tumours — ELST)"
        ),
        "inheritance": (
            "AD LOF 3p25.3 — VHL. VHL disease. "
            "Penetrance: >95% for at least one VHL manifestation by age 60. "
            "De novo rate ~20% of VHL disease probands. "
            "Second hit (LOH or somatic point mutation at 3p25) required for tumorigenesis. "
            "Prevalence: 1 per 36,000. "
            "Genotype-phenotype: missense C162F → type 2B (highest phaeochromocytoma + RCC risk); "
            "truncating variants → type 1 (no phaeochromocytoma but high RCC + CNS hemangioblastoma)."
        ),
        "surveillance_key": (
            "Annual retinal ophthalmoscopy from age 1 (retinal hemangioblastoma); "
            "annual brain+spine MRI from age 11; annual abdominal MRI; "
            "annual plasma metanephrines; belzutifan for VHL-associated RCC + CNS hemangioblastoma"
        ),
        "pathognomonic": (
            "Multiple CNS hemangioblastomas (cerebellum/brainstem/spinal cord) = PATHOGNOMONIC VHL; "
            "Bilateral/multiple retinal hemangioblastomas = PATHOGNOMONIC VHL; "
            "VHL triad: hemangioblastoma + ccRCC + phaeochromocytoma = diagnostic"
        ),
    },
    {
        "gene": "SMARCB1",
        "protein": (
            "SMARCB1 -- 22q11.23 Autosomal-Dominant-LOF -- 385aa -- "
            "INI1-hSNF5-SWI-SNF-Chromatin-Remodeller-45kDa-"
            "RTPS2-AT-RT-Under-Age3-PATHOGNOMONIC-INI1-IHC-Loss-PATHOGNOMONIC-"
            "Schwannomatosis2-Rhabdoid-Meningioma-Sibling-Surveillance-OMIM-601607"
        ),
        "locus": "22q11.23",
        "protein_size": (
            "385 aa / 22q11.23 SMARCB1 encodes SMARCB1 (INI1 / hSNF5 / BAF47): "
            "STRUCTURE: "
            "  RVH domain (aa 1-30): N-terminal; "
            "  SWIB/MDM2 domain (aa 57-156): SWI/SNF complex interaction; "
            "  Coiled-coil domain (aa 290-385): multimerisation; "
            "FUNCTION: "
            "  Core subunit of SWI/SNF (BAF) chromatin remodelling complex; "
            "  SWI/SNF complex: ATP-dependent chromatin remodelling → nucleosome repositioning → gene activation; "
            "  SMARCB1/INI1 required for SWI/SNF complex assembly and function; "
            "  SMARCB1 loss: epigenetic reprogramming → dedifferentiation → MYC/PRC2-driven oncogenesis; "
            "  EZH2 (PRC2) becomes unopposed in SMARCB1-null cells → tazemetostat sensitivity; "
            "RHABDOID TUMOUR PREDISPOSITION SYNDROME TYPE 2 (RTPS2): "
            "  Germline SMARCB1 heterozygous variant → RTPS2 (constitutional); "
            "  ATYPICAL TERATOID/RHABDOID TUMOUR (AT/RT) — PATHOGNOMONIC: "
            "    AT/RT in child <3yr (especially <18 months): PATHOGNOMONIC for germline SMARCB1; "
            "    AT/RT is the most aggressive CNS embryonal tumour — median survival <12 months historically; "
            "    Modern intensive multimodal therapy: median OS improving (18-24 months); "
            "    AT/RT location: posterior fossa (infants) or supratentorial (older children); "
            "    INI1 IHC LOSS = PATHOGNOMONIC AT/RT: INI1 absent in tumour cells, retained in vessels/stroma; "
            "    Molecular diagnosis: SMARCB1 deletion or mutation (somatic, germline, or mosaicism); "
            "  RHABDOID MENINGIOMA: "
            "    SMARCB1-deficient meningioma — high grade, aggressive; INI1 loss IHC; "
            "    SMARCB1 germline in subset of rhabdoid meningioma patients; "
            "  SCHWANNOMATOSIS TYPE 2: "
            "    Multiple painful schwannomas (spinal, peripheral) without bilateral VS (≠ NF2); "
            "    SMARCB1 germline + somatic hits on 22q → schwannomatosis; "
            "  SIBLING SURVEILLANCE: "
            "    AT/RT child: siblings must be tested for germline SMARCB1 (MANDATORY); "
            "    RTPS2 families: 50% risk each child; "
            "    Germline SMARCB1 carrier infant: brain+spine MRI 3-6 monthly first 5yr; "
            "  TAZEMETOSTAT (EZH2 inhibitor): "
            "    PRC2/EZH2 becomes essential oncogenic driver in SMARCB1-null — synthetic lethality; "
            "    Tazemetostat FDA 2020: epithelioid sarcoma (SMARCB1-null); clinical trials AT/RT; "
            "  INI1 IHC: "
            "    Standard immunostaining — absent (null) nuclear staining in SMARCB1-null tumours; "
            "    PATHOGNOMONIC pattern: tumour cell nuclei negative, endothelial/inflammatory cells positive; "
            "    All CNS tumours in children <3yr: INI1 IHC mandatory"
        ),
        "inheritance": (
            "AD LOF 22q11.23 — SMARCB1. RTPS2 / Schwannomatosis-2. "
            "AT/RT penetrance in germline SMARCB1: ~15-25% lifetime CNS rhabdoid tumour risk. "
            "De novo germline SMARCB1 common in AT/RT (~35-50% of SMARCB1-mutant AT/RT are de novo); "
            "Somatic mosaicism: ~20% of at-risk families — both parents may test negative but child has germline variant. "
            "Prevalence AT/RT: 1-2% of all childhood CNS tumours; germline SMARCB1 in ~35-40% of AT/RT. "
            "Genotype-phenotype: SMARCB1 whole-gene deletion → Schwannomatosis-2 dominant; "
            "truncating variants → AT/RT predominant."
        ),
        "surveillance_key": (
            "Germline SMARCB1: brain+spine MRI 3-6 monthly first 5yr; "
            "siblings tested immediately (MANDATORY); "
            "AT/RT <3yr → germline SMARCB1 testing + sibling surveillance; "
            "INI1 IHC on all CNS embryonal tumours in children; tazemetostat clinical trials"
        ),
        "pathognomonic": (
            "AT/RT in child <18 months = PATHOGNOMONIC for germline SMARCB1 — sibling testing mandatory; "
            "INI1 IHC loss (absent tumour nuclei) = PATHOGNOMONIC SMARCB1-deficient tumour; "
            "AT/RT + sibling with AT/RT = RTPS2 (SMARCB1 germline) pattern"
        ),
    },
    {
        "gene": "PMS2",
        "protein": (
            "PMS2 -- 7p22.2 AR-Biallelic-CMMRD-AD-Monoallelic-Lynch -- 862aa -- "
            "PMS1-Homolog2-Mismatch-Repair-MutL-96kDa-"
            "CMMRD-Glioblastoma-Medulloblastoma-Childhood-Hypermutation->100mut/Mb-"
            "CafeAuLait-Brain-Tumour-CMMRD-Excluded-PD1-Immunotherapy-OMIM-600259"
        ),
        "locus": "7p22.2",
        "protein_size": (
            "862 aa / 7p22.2 PMS2 encodes PMS2 (PMS1 homolog 2, mismatch repair system component): "
            "STRUCTURE: "
            "  N-terminal ATPase domain (aa 1-365): ATP binding/hydrolysis; "
            "  Endonuclease domain (DCHE motif, aa 390-439): metal-dependent nuclease; "
            "  C-terminal MLH1-interaction domain (aa 675-862): MutL heterodimer interface; "
            "FUNCTION: "
            "  PMS2 heterodimerises with MLH1 → MutLα complex; "
            "  MutLα: strand discrimination + nicking during MMR after MutS mismatch recognition; "
            "  PMS2 endonuclease: introduces nicks in nascent strand → exonuclease excision → re-synthesis; "
            "  PMS2 loss: base-pair mismatches and insertions/deletions accumulate → microsatellite instability; "
            "CONSTITUTIONAL MMR DEFICIENCY (CMMRD) — BIALLELIC PMS2: "
            "  CMMRD: biallelic germline MMR gene mutation (PMS2 is MOST COMMON CMMRD gene); "
            "  CNS TUMOURS — CHILDHOOD GLIOBLASTOMA / MEDULLOBLASTOMA: "
            "    CMMRD is a major cause of childhood glioblastoma (GBM) and medulloblastoma: "
            "    GBM in child <10yr: CMMRD must be excluded — rare in general population, enriched CMMRD; "
            "    CMMRD-GBM: ultra-hypermutated (>100 mut/Mb) — distinct from somatic GBM (IDH-WT); "
            "    Hypermutation PATHOGNOMONIC: CMMRD CNS tumours have TMB >100 mut/Mb (somatic GBM ~5 mut/Mb); "
            "  CAFÉ AU LAIT MACULES (CALM): "
            "    Multiple CALMs (typically ≥6, >5mm prepubertal) in child with brain tumour: CMMRD until excluded; "
            "    CMMRD-CALMs: present from birth, often Neurofibromatosis-1-like pattern; "
            "    KEY DDX: CMMRD vs NF1 — CMMRD CALMs lack freckling, Lisch nodules, neurofibromas; "
            "    Constitutional Lynch: NBS1/Lynch features overlap in CMMRD; "
            "  PD-1 IMMUNOTHERAPY — HYPERMUTATED CNS TUMOURS: "
            "    CMMRD-GBM TMB >100 mut/Mb → neoantigen burden → immune recognition; "
            "    PD-1 blockade (pembrolizumab/nivolumab): durable responses in hypermutated CMMRD-GBM; "
            "    Case series and KEYNOTE data: >50% response rate in TMB-high CMMRD CNS tumours; "
            "    CheckMate 143 / KEYNOTE-158: MSI-H/dMMR solid tumours — landmark hypermutation response; "
            "    CMMRD-MB also responds to PD-1 if hypermutated (SHH-MB with CMMRD); "
            "  MONOALLELIC PMS2 (LYNCH SYNDROME): "
            "    Monoallelic PMS2 germline: Lynch syndrome — lower penetrance than MLH1/MSH2; "
            "    Lynch-PMS2: colorectal Ca 10-20% lifetime; endometrial Ca 12-20%; "
            "    Brain tumour in Lynch-PMS2: rare (Turcot type 1 — glioblastoma); "
            "SURVEILLANCE — CMMRD: "
            "  Brain+spine MRI every 6 months (highest CNS risk in childhood); "
            "  Annual colonoscopy from age 6 (CRC risk extreme early onset); "
            "  Annual lymphoma screen (blood + abdominal USS — NHL/Hodgkin in CMMRD); "
            "  Immunogenetics referral — PD-1 therapy consideration for any CNS recurrence; "
            "  Dermatology: CALMs surveillance"
        ),
        "inheritance": (
            "Biallelic AR (CMMRD) / Monoallelic AD (Lynch) 7p22.2 — PMS2. "
            "CMMRD: two pathogenic PMS2 alleles; parents usually Lynch carriers (monoallelic). "
            "PMS2 most common CMMRD gene (~55% of CMMRD cases). "
            "Lynch (monoallelic PMS2): ~15-20% lifetime CRC; ~12-20% endometrial. "
            "CMMRD penetrance for cancer in childhood: >95% — most CMMRD patients develop cancer <18yr. "
            "Prevalence CMMRD: ~1:1,000,000 (rare; underdiagnosed). "
            "PMS2 pseudogene PMS2CL on 7p22 complicates Sanger sequencing — MLPA + long-range PCR required."
        ),
        "surveillance_key": (
            "CMMRD: brain+spine MRI 6-monthly; annual colonoscopy from age 6; "
            "annual lymphoma screen; PD-1 immunotherapy for hypermutated CMMRD-GBM recurrence; "
            "TMB testing on all childhood GBM (CMMRD = TMB >100 mut/Mb); "
            "Lynch (monoallelic): colonoscopy from age 25-35; endometrial sampling from age 35"
        ),
        "pathognomonic": (
            "CMMRD: childhood GBM with TMB >100 mut/Mb = PATHOGNOMONIC constitutional MMR deficiency; "
            "Multiple CALMs + childhood brain tumour = CMMRD until excluded (test PMS2/MLH1/MSH2/MSH6); "
            "PMS2 pseudogene: PMS2CL on 7p22 requires MLPA + long-range PCR — Sanger MISSES deletions"
        ),
    },
]


def _gene_stats(seed: int, gene_cfg: dict) -> dict:
    rng = random.Random(seed)
    gene = gene_cfg["gene"]

    base = {
        "TP53":    dict(age_mu=28, age_sd=12, brain_pct=(30, 45), sarcoma_pct=(25, 38), cpc_pct=(10, 20), radiation_avoid_pct=(90, 98)),
        "PTCH1":   dict(age_mu=9,  age_sd=5,  mb_pct=(20, 35),   bcc_pct=(82, 96),     okc_pct=(65, 82), calcified_falx_pct=(55, 78)),
        "SUFU":    dict(age_mu=2,  age_sd=1,  mb_pct=(55, 72),   bcc_pct=(30, 55),     mben_pct=(38, 55), radiation_avoid_pct=(88, 97)),
        "PTEN":    dict(age_mu=38, age_sd=12, ldd_pct=(18, 32),  breast_pct=(62, 80),  thyroid_pct=(28, 40), macrocephaly_pct=(78, 92)),
        "APC":     dict(age_mu=12, age_sd=6,  mb_pct=(8, 16),    polyp_pct=(95, 100),  chrpe_pct=(72, 88), desmoid_pct=(15, 28)),
        "VHL":     dict(age_mu=33, age_sd=10, cns_hb_pct=(60, 80), ret_hb_pct=(45, 65), rcc_pct=(28, 45), pheo_pct=(18, 35)),
        "SMARCB1": dict(age_mu=2,  age_sd=1,  atrt_pct=(45, 65), schwann_pct=(25, 42), ini1_loss_pct=(92, 100), sibling_risk_pct=(42, 52)),
        "PMS2":    dict(age_mu=10, age_sd=5,  gbm_pct=(35, 55),  calm_pct=(72, 90),    tmb_high_pct=(82, 95), pd1_response_pct=(40, 62)),
    }
    b = base.get(gene, dict(age_mu=20, age_sd=10))

    def pct(lo, hi): return round(rng.uniform(lo, hi), 1)
    def age(): return max(0.5, round(rng.gauss(b["age_mu"], b["age_sd"]), 1))

    ages = [age() for _ in range(40)]
    mean_age = round(sum(ages) / len(ages), 1)

    stats = dict(gene=gene, n=40, mean_age_diagnosis=mean_age)
    for key, val in b.items():
        if key.endswith("_pct") and isinstance(val, tuple):
            stats[key] = pct(val[0], val[1])

    stats["locus"] = gene_cfg["locus"]
    stats["inheritance"] = gene_cfg["inheritance"]
    stats["surveillance_key"] = gene_cfg["surveillance_key"]
    stats["pathognomonic"] = gene_cfg["pathognomonic"]
    return stats


def generate_overview() -> dict:
    return {
        "atlas":          "Hereditary-Brain-CNS-Tumour-Predisposition-Atlas",
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_genes":    len(ATLAS_GENES),
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "inheritance_modes": {
            "TP53": (
                "AD LOF 17p13.1 (p53 guardian genome; 393aa; Li-Fraumeni Syndrome; "
                "Choroid Plexus Carcinoma <5yr = PATHOGNOMONIC; Glioblastoma; Medulloblastoma; "
                "AVOID ALL RADIATION ABSOLUTELY; WBMRI Toronto annually)"
            ),
            "PTCH1": (
                "AD LOF 9q22.32 (Patched-1 Hh receptor; 1447aa; Gorlin/NBCCS; "
                "Desmoplastic medulloblastoma SHH infratentorial = PATHOGNOMONIC; "
                "Calcified falx PATHOGNOMONIC; OKCs; BCCs hundreds; AVOID RADIATION ABSOLUTELY; Vismodegib)"
            ),
            "SUFU": (
                "AD LOF 10q24.32 (Suppressor of Fused Hh inhibitor; 484aa; BCNS type 2; "
                "Childhood MB highest penetrance ~15-20%; MBEN = PATHOGNOMONIC in infant; "
                "Fewer BCCs than PTCH1; AVOID RADIATION ABSOLUTELY; chemo-only infant protocol)"
            ),
            "PTEN": (
                "AD LOF 10q23.31 (PTEN phosphatase PI3K; 403aa; Cowden/PHTS; "
                "Lhermitte-Duclos cerebellar dysplastic gangliocytoma = PATHOGNOMONIC adult; "
                "Macrocephaly HC >97th centile = PATHOGNOMONIC screening criterion; "
                "breast/thyroid/endometrial cancer; Everolimus mTOR)"
            ),
            "APC": (
                "AD LOF 5q22.2 (APC WNT gatekeeper; 2843aa; FAP/Turcot type 2; "
                "WNT-activated medulloblastoma best prognosis; CHRPE = PATHOGNOMONIC FAP; "
                "β-catenin nuclear IHC = PATHOGNOMONIC WNT-MB; prophylactic colectomy; Gardner desmoids)"
            ),
            "VHL": (
                "AD LOF 3p25.3 (pVHL HIF E3 adaptor; 213aa; VHL disease; "
                "CNS hemangioblastoma = PATHOGNOMONIC; retinal hemangioblastoma = PATHOGNOMONIC; "
                "annual ophthalmology from age 1; ccRCC; phaeochromocytoma; "
                "Belzutifan HIF-2α inhibitor FDA 2021)"
            ),
            "SMARCB1": (
                "AD LOF 22q11.23 (INI1/hSNF5 SWI/SNF; 385aa; RTPS2; "
                "AT/RT under age 3yr = PATHOGNOMONIC germline SMARCB1; "
                "INI1 IHC loss = PATHOGNOMONIC AT/RT; sibling surveillance MANDATORY; "
                "rhabdoid meningioma; Schwannomatosis-2; tazemetostat EZH2)"
            ),
            "PMS2": (
                "Biallelic AR CMMRD / Monoallelic AD Lynch 7p22.2 (PMS2 MutLα; 862aa; "
                "CMMRD: childhood GBM/MB hypermutated TMB >100 mut/Mb = PATHOGNOMONIC; "
                "CALMs + childhood brain tumour = CMMRD excluded; "
                "PD-1 immunotherapy — hypermutated CMMRD-GBM responds)"
            ),
        },
        "key_clinical_rules": [
            "TP53 germline (LFS): Choroid Plexus Carcinoma in child <5yr = PATHOGNOMONIC — >50% CPC harbour germline TP53; test immediately",
            "TP53 germline: AVOID ALL IONISING RADIATION ABSOLUTELY — standard RT induces radiation-field sarcoma; proton does NOT exempt from this rule",
            "TP53 surveillance: WBMRI annually (Toronto protocol) + brain MRI 6-monthly in children; abdominal USS 6-monthly for ACC",
            "PTCH1/Gorlin: calcified falx cerebri on CT = PATHOGNOMONIC — diagnose before any tumour; AVOID ALL RT — field cancerisation causes hundreds of BCCs in irradiated skin",
            "PTCH1: desmoplastic/nodular medulloblastoma SHH-activated in child = PTCH1/Gorlin germline testing mandatory; use Baby Brain chemo-only protocol (NO WBRT)",
            "SUFU: higher MB penetrance (~15-20%) than PTCH1 (~2-5%); infant <2yr + SHH-activated MB → SUFU germline testing; MBEN histology = PATHOGNOMONIC SUFU/PTCH1 germline",
            "PTEN/Cowden: Lhermitte-Duclos disease (dysplastic cerebellar gangliocytoma, striated MRI) = PATHOGNOMONIC adult PHTS — test PTEN germline in every LDD patient",
            "PTEN: macrocephaly (HC >97th centile) = PATHOGNOMONIC screening criterion; macrocephaly + breast Ca at any age → PTEN germline testing",
            "APC/Turcot type 2: WNT-activated medulloblastoma = best prognosis MB (5yr OS >90%); CHRPE (multiple bilateral) = PATHOGNOMONIC FAP; prophylactic colectomy before age 25",
            "APC: β-catenin nuclear IHC in MB = PATHOGNOMONIC WNT subtype — APC/CTNNB1 test; MB in FAP child may precede colorectal polyps",
            "VHL: CNS hemangioblastoma = PATHOGNOMONIC VHL; annual retinal ophthalmoscopy from age 1 (retinal hemangioblastoma → blindness if missed); Belzutifan FDA 2021 for VHL-associated RCC + CNS hemangioblastoma",
            "VHL genotype-phenotype: type 1 (truncating) → hemangioblastoma + RCC, NO phaeochromocytoma; type 2A/2B (missense Phe motif) → phaeochromocytoma + hemangioblastoma ± RCC",
            "SMARCB1/RTPS2: AT/RT in child <3yr (especially <18 months) = PATHOGNOMONIC germline SMARCB1 — sibling testing MANDATORY immediately",
            "SMARCB1: INI1 IHC loss = PATHOGNOMONIC SMARCB1-null tumour — ALL CNS embryonal tumours in children <3yr must have INI1 IHC; tazemetostat (EZH2) clinical trials for AT/RT",
            "PMS2 biallelic (CMMRD): childhood GBM with TMB >100 mut/Mb = PATHOGNOMONIC constitutional MMR deficiency; multiple CALMs + childhood brain tumour → CMMRD excluded (test all 4 MMR genes)",
            "PMS2/CMMRD: PD-1 immunotherapy (pembrolizumab) — durable responses in hypermutated CMMRD-GBM; check TMB on ALL recurrent childhood brain tumours",
            "ALL HEREDITARY CNS TUMOUR PANEL: germline TP53 + PTCH1 + SUFU + PTEN + APC + VHL + SMARCB1 + PMS2 — indicated for: childhood brain tumour <5yr, multiple CNS tumours, family history brain tumour, MB (SHH/WNT), AT/RT any age, CPC any age, hemangioblastoma, LDD",
        ],
        "gene_panel_note": (
            "Hereditary Brain & CNS Tumour Germline Panel (clinical 2024): "
            "TP53 PATHWAY (LFS — CPC PATHOGNOMONIC, AVOID RADIATION ABSOLUTELY): "
            "  TP53: LFS; CPC <5yr PATHOGNOMONIC; GBM; MB; WBMRI Toronto; ACC <5yr PATHOGNOMONIC; "
            "HEDGEHOG PATHWAY (PTCH1/SUFU — SHH medulloblastoma, AVOID RADIATION ABSOLUTELY): "
            "  PTCH1: Gorlin/NBCCS; desmoplastic MB SHH; calcified falx PATHOGNOMONIC; BCCs; OKC; Vismodegib; "
            "  SUFU: BCNS2; infant MB higher penetrance; MBEN PATHOGNOMONIC; fewer BCCs; "
            "PI3K/AKT/mTOR PATHWAY (PTEN — Lhermitte-Duclos PATHOGNOMONIC): "
            "  PTEN: Cowden/PHTS; LDD cerebellar PATHOGNOMONIC; macrocephaly PATHOGNOMONIC; breast/thyroid/endometrial; Everolimus; "
            "WNT PATHWAY (APC/Turcot type 2 — WNT medulloblastoma BEST PROGNOSIS): "
            "  APC: FAP; WNT-MB; CHRPE PATHOGNOMONIC; β-catenin nuclear IHC; Gardner desmoids; "
            "VHL HIF-2α PATHWAY (CNS hemangioblastoma PATHOGNOMONIC): "
            "  VHL: hemangioblastoma CNS/retinal PATHOGNOMONIC; annual retinal age 1; Belzutifan HIF2α FDA2021; ccRCC; phaeochromocytoma; "
            "SWI/SNF CHROMATIN REMODELLING (SMARCB1 — AT/RT PATHOGNOMONIC age <3): "
            "  SMARCB1: RTPS2; AT/RT <3yr PATHOGNOMONIC; INI1 IHC loss PATHOGNOMONIC; sibling test MANDATORY; tazemetostat EZH2; "
            "MISMATCH REPAIR PATHWAY (PMS2 — CMMRD hypermutation PATHOGNOMONIC): "
            "  PMS2 biallelic CMMRD: childhood GBM/MB; TMB >100 PATHOGNOMONIC; CALMs + brain tumour → CMMRD excluded; PD-1 active; "
            "  PMS2 monoallelic Lynch: CRC/endometrial lower penetrance vs MLH1/MSH2; "
            "UNIVERSAL TESTING CRITERIA: "
            "  ALL CPC (any age): TP53 germline mandatory; "
            "  ALL AT/RT (any age): SMARCB1 germline mandatory + sibling testing; "
            "  ALL childhood GBM <10yr: PMS2 germline + TMB testing; "
            "  MB SHH subtype child <5yr: PTCH1 + SUFU germline; "
            "  MB WNT subtype: APC germline; "
            "  CNS hemangioblastoma (single): VHL germline; "
            "  Cerebellar mass + striated MRI: PTEN germline (Lhermitte-Duclos)"
        ),
    }


def generate_breakdown() -> dict:
    genes_data = []
    for i, gene_cfg in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        stats = _gene_stats(seed, gene_cfg)
        stats["protein_summary"] = gene_cfg["protein"]
        stats["locus"] = gene_cfg["locus"]
        stats["inheritance"] = gene_cfg["inheritance"]
        stats["surveillance_key"] = gene_cfg["surveillance_key"]
        stats["pathognomonic"] = gene_cfg["pathognomonic"]
        genes_data.append(stats)
    return {
        "atlas":          "Hereditary-Brain-CNS-Tumour-Predisposition-Atlas",
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "n_genes":        len(genes_data),
        "total_patients": 320,
        "genes":          genes_data,
    }


def generate_definitions() -> dict:
    definitions = [
        {
            "term": "TP53-CPC-PATHOGNOMONIC-LFS-AVOID-RADIATION-ABSOLUTELY",
            "definition": (
                "CHOROID PLEXUS CARCINOMA (CPC) IN CHILD <5YR = PATHOGNOMONIC LI-FRAUMENI SYNDROME: "
                "CPC is a highly malignant intraventricular tumour arising from choroid plexus epithelium. "
                "CPC hallmark genetics: >50% of CPC harbour germline TP53 pathogenic variant (Li-Fraumeni syndrome). "
                "Sporadic CPC (no germline TP53) does occur but is the minority in children <5yr. "
                "CLINICAL RULE: ALL CPC regardless of age → germline TP53 mandatory. "
                "Presentation: ventriculomegaly, macrocephaly in infant, raised ICP. "
                "MRI: enhancing intraventricular frond-like mass, often in lateral ventricle. "
                "Treatment: resection (gross total) + chemotherapy (avoid RT); "
                "LFS-CPC: CPE protocol (carboplatin, vincristine, etoposide) — avoid RT absolutely. "
                "Choroid plexus papilloma (CPP, benign): lower TP53 association but still test germline. "
                "AVOID RADIATION ABSOLUTELY IN GERMLINE TP53 CARRIERS: "
                "RT induces malignant transformation in TP53-null cells within radiation field — "
                "osteosarcoma, secondary glioma, radiation-induced sarcoma documented in multiple LFS RT series; "
                "This is an absolute contraindication regardless of clinical urgency. "
                "WBMRI annually (Toronto protocol): whole-body MRI screens for LFS-associated tumour spectrum "
                "(brain + spine + chest + abdomen + pelvis + long bones + soft tissue); "
                "50% cancer risk by age 30, >90% lifetime — early detection saves lives in LFS."
            ),
        },
        {
            "term": "PTCH1-Gorlin-Calcified-Falx-PATHOGNOMONIC-Desmoplastic-MB-AVOID-RADIATION-Vismodegib",
            "definition": (
                "GORLIN SYNDROME (NBCCS) — PTCH1 GERMLINE: RADIATION CATASTROPHE RISK: "
                "CALCIFIED FALX CEREBRI = PATHOGNOMONIC GORLIN/PTCH1 GERMLINE: "
                "  CT/MRI: calcification of falx cerebri by age 20 (normal calcification is >30yr); "
                "  Early calcified falx is a non-tumour PATHOGNOMONIC structural finding of Gorlin; "
                "  Identified on routine CT — does not require treatment; diagnoses carrier status; "
                "DESMOPLASTIC/NODULAR MEDULLOBLASTOMA SHH-ACTIVATED = PATHOGNOMONIC GORLIN PATTERN: "
                "  SHH-subgroup MB in posterior fossa child 1-5yr: PTCH1 germline testing mandatory; "
                "  Desmoplastic/nodular histology: reticulin-free islands (pale islands) in fibrillary stroma; "
                "  MB in Gorlin: EXCELLENT prognosis with chemo-only — avoid WBRT absolutely; "
                "AVOID RADIATION ABSOLUTELY — FIELD CANCERISATION: "
                "  WBRT in Gorlin child → thousands BCCs in irradiated skin within 6-12 months; "
                "  Case reports: Gorlin child treated with WBRT → died of BCC burden; "
                "  Protocol: Baby Brain / SJMB96 infant protocol (chemo-only, reduce/eliminate RT); "
                "  Post-resection: intensive chemo; second-look surgery; avoid radiotherapy at all costs; "
                "VISMODEGIB (FDA 2012) / SONIDEGIB: "
                "  SMO inhibitor: blocks Hh pathway downstream of PTCH1 loss; "
                "  Vismodegib 150mg daily: ORR for locally advanced BCC ~60%; "
                "  Drug holiday protocol: 3-months-on/3-months-off reduces cumulative toxicity; "
                "  Muscle cramps, alopecia, dysgeusia, teratogenicity (embryofetal toxicity — contraception MANDATORY); "
                "  Role in Gorlin MB: under investigation (preclinical + compassionate use data)."
            ),
        },
        {
            "term": "SUFU-BCNS2-MBEN-Infant-Medulloblastoma-Highest-Penetrance-AVOID-RADIATION",
            "definition": (
                "SUFU-RELATED GORLIN (BCNS TYPE 2) — HIGHEST MEDULLOBLASTOMA PENETRANCE: "
                "SUFU acts downstream of SMO — directly inhibits GLI1/GLI2. "
                "SUFU loss → constitutive Hh signalling independent of ligand or PTCH1/SMO status. "
                "MEDULLOBLASTOMA WITH EXTENSIVE NODULARITY (MBEN): "
                "  MBEN = grapelike nodular MB exclusively in infants/toddlers; "
                "  MBEN = PATHOGNOMONIC pattern in SUFU (and PTCH1) germline MB; "
                "  MBEN has BEST prognosis of all MB subtypes — chemo-only: 5yr OS >90%; "
                "  MBEN MRI: posterior fossa enhancing mass, grape-cluster nodules; "
                "MB PENETRANCE SUFU > PTCH1: "
                "  SUFU: ~15-20% lifetime MB risk (vs PTCH1 ~2-5%); "
                "  SUFU MB onset: infants (median <2yr); "
                "  Brain MRI 6-monthly from birth in SUFU carrier infants; "
                "AVOID RADIATION ABSOLUTELY — same rationale as PTCH1/Gorlin: "
                "  Hh pathway active in skin → RT-induced BCC field cancerisation; "
                "  All SUFU/PTCH1 MB treated with chemo-only (infant protocols); "
                "FEWER BCCs THAN PTCH1: SUFU BCCs develop later in adulthood (3rd-4th decade), "
                "less florid, but BCC surveillance still required."
            ),
        },
        {
            "term": "PTEN-Lhermitte-Duclos-PATHOGNOMONIC-Macrocephaly-Cowden-Everolimus",
            "definition": (
                "LHERMITTE-DUCLOS DISEASE (LDD) = PATHOGNOMONIC PTEN/COWDEN HAMARTOMA SYNDROME: "
                "LDD = DYSPLASTIC CEREBELLAR GANGLIOCYTOMA: "
                "  Benign hamartomatous overgrowth of cerebellar granule cell layer → cerebellum mass; "
                "  NOT a malignant neoplasm — no metastatic potential; but local mass effect → death; "
                "MRI SIGNATURE = PATHOGNOMONIC: "
                "  Striated/tigroid pattern of cerebellar cortex on T2/FLAIR: alternating bright and dark bands; "
                "  This radiological appearance is PATHOGNOMONIC for LDD — no other CNS lesion shows this pattern; "
                "  Slowly progressive; may be present for years before symptoms; "
                "  Presentation: progressive cerebellar ataxia, headache, raised ICP, hydrocephalus; "
                "CLINICAL RULE: EVERY LDD PATIENT → PTEN GERMLINE TESTING: "
                "  Adult-onset LDD: >90% are Cowden/PHTS — germline PTEN; "
                "  Childhood-onset LDD (rare) → also test PTEN; "
                "MACROCEPHALY = PATHOGNOMONIC PTEN SCREENING CRITERION: "
                "  HC >97th centile (>+2SD): single most sensitive PHTS clinical feature; "
                "  Macrocephaly + breast Ca (any age) → PTEN germline (NCCN criteria); "
                "  Macrocephaly + ASD: ~17% of ASD+macrocephaly have germline PTEN; "
                "EVEROLIMUS (mTOR INHIBITOR): "
                "  PTEN loss → constitutive PI3K/AKT/mTOR; "
                "  Everolimus: used in PTEN-driven RCC, breast, SEGA (TSC/PTEN); "
                "  LDD: everolimus trials ongoing; surgical resection remains standard for symptomatic LDD."
            ),
        },
        {
            "term": "APC-Turcot2-WNT-Medulloblastoma-CHRPE-PATHOGNOMONIC-Gardner-Desmoid",
            "definition": (
                "TURCOT SYNDROME TYPE 2 (APC): FAP + WNT-MEDULLOBLASTOMA: "
                "WNT-ACTIVATED MEDULLOBLASTOMA = BEST PROGNOSIS MB: "
                "  WNT-MB: 5yr OS >90% with standard therapy; often curable; "
                "  APC germline in ~10-15% of WNT-MB; CTNNB1 somatic mutation in remainder; "
                "  β-catenin nuclear IHC = PATHOGNOMONIC WNT-MB (nuclear accumulation of β-catenin); "
                "  Characteristic: monosomy 6 in WNT-MB (80%); "
                "  WNT-MB treatment: de-escalation trials underway (reduce RT dose given excellent prognosis); "
                "CHRPE = PATHOGNOMONIC FAP/APC GERMLINE: "
                "  Multiple bilateral congenital hypertrophy of retinal pigment epithelium (CHRPE): "
                "  >4 lesions bilaterally or characteristic 'fishhook' / 'bear-track' pattern = PATHOGNOMONIC; "
                "  CHRPE require NO treatment — purely diagnostic marker; found on ophthalmoscopy; "
                "  FAP-CHRPE diagnostic sensitivity ~80%; specificity ~99% for APC germline; "
                "GARDNER SYNDROME (APC VARIANT): "
                "  FAP + desmoid tumours (aggressive fibromatosis) + osteomas + epidermoid cysts; "
                "  Desmoid tumours: mesenteric/intra-abdominal → can compress bowel post-colectomy → fatal; "
                "  Celecoxib (COX-2 inhibitor): modest desmoid regression; sulindac reduces polyps; "
                "  Prophylactic colectomy: total proctocolectomy + IPAA (ileo-pouch anal anastomosis) by age 20-25; "
                "  Without colectomy: 100% CRC risk by age 40."
            ),
        },
        {
            "term": "VHL-CNS-Hemangioblastoma-PATHOGNOMONIC-Retinal-Annual-Age1-Belzutifan-HIF2alpha",
            "definition": (
                "VHL DISEASE — CNS HEMANGIOBLASTOMA + RETINAL HEMANGIOBLASTOMA PATHOGNOMONIC: "
                "CNS HEMANGIOBLASTOMA = PATHOGNOMONIC VHL: "
                "  Multiple or recurrent CNS hemangioblastomas = VHL germline until excluded; "
                "  Even single cerebellar hemangioblastoma: VHL germline testing recommended; "
                "  MRI: enhancing cystic mass with solid vascular mural nodule; "
                "  Locations: cerebellum (60%), spinal cord (44%), brainstem (18%), supratentorial (rare); "
                "  Observe small asymptomatic (<3cm, no growth, no symptoms); "
                "  Surgery or SRS: growing lesions or symptomatic (cyst expansion, oedema); "
                "RETINAL HEMANGIOBLASTOMA = PATHOGNOMONIC VHL — ANNUAL OPHTHALMOLOGY FROM AGE 1: "
                "  Retinal capillary hemangioblastoma (Von Hippel tumour): "
                "    Bilateral or multiple = PATHOGNOMONIC VHL; even unilateral → test germline; "
                "  Annual ophthalmoscopy from age 1 year: "
                "    Rationale: retinal hemangioblastoma can present in infants → retinal detachment → permanent blindness; "
                "    Early detection: laser photocoagulation / intravitreal anti-VEGF → prevents vision loss; "
                "    This is the MOST time-sensitive VHL surveillance item; "
                "BELZUTIFAN (HIF-2α INHIBITOR) FDA 2021: "
                "  Mechanism: belzutifan blocks HIF-2α/ARNT dimerisation → suppresses VEGF, PDGF, EPO; "
                "  FDA 2021: VHL disease-associated RCC, CNS hemangioblastoma, pancreatic neuroendocrine tumour; "
                "  LITESPARK-004: ORR 49% CNS hemangioblastoma; 64% RCC; "
                "  Dose 120mg daily; key AEs: anaemia (Hb monitoring), fatigue, dizziness; "
                "  Transforms VHL disease from purely surgical to medical management option."
            ),
        },
        {
            "term": "SMARCB1-RTPS2-ATRT-Under3-PATHOGNOMONIC-INI1-IHC-Sibling-Surveillance-Tazemetostat",
            "definition": (
                "RHABDOID TUMOUR PREDISPOSITION SYNDROME TYPE 2 (RTPS2) — SMARCB1: "
                "AT/RT IN CHILD <3YR (ESPECIALLY <18 MONTHS) = PATHOGNOMONIC GERMLINE SMARCB1: "
                "  AT/RT is the most aggressive paediatric CNS tumour — median age at diagnosis <2yr; "
                "  Germline SMARCB1 in ~35-40% of AT/RT; de novo germline ~50% of these; "
                "  CLINICAL RULE: ALL AT/RT → GERMLINE SMARCB1 MANDATORY + SIBLING TESTING IMMEDIATELY; "
                "INI1 IHC LOSS = PATHOGNOMONIC AT/RT / SMARCB1-DEFICIENT TUMOUR: "
                "  INI1 (SMARCB1 protein) IHC: absent nuclear staining in tumour cells; "
                "  Vessels and stromal cells retain INI1 staining (internal positive control); "
                "  ALL CNS embryonal tumours in children <3yr → INI1 IHC mandatory; "
                "  INI1 null on IHC + SMARCB1 deletion/mutation on molecular → diagnostic AT/RT; "
                "SIBLING SURVEILLANCE — MANDATORY: "
                "  RTPS2 family: 50% sibling risk; test siblings for germline SMARCB1; "
                "  SMARCB1-positive sibling: brain + spine MRI every 3-6 months first 5yr; "
                "  Somatic mosaicism: parents may test negative → offspring screening still required; "
                "TAZEMETOSTAT (EZH2 INHIBITOR): "
                "  SMARCB1 loss → unopposed PRC2/EZH2 → H3K27me3 accumulation → silencing of tumour suppressors; "
                "  Tazemetostat blocks EZH2 → reverses PRC2-driven oncogenesis (synthetic lethality); "
                "  FDA 2020: epithelioid sarcoma (SMARCB1-null); clinical trials AT/RT ongoing; "
                "  AT/RT trials: tazemetostat combinations with DNA damage agents."
            ),
        },
        {
            "term": "PMS2-CMMRD-GBM-TMB100-PATHOGNOMONIC-CALMs-PD1-Immunotherapy",
            "definition": (
                "CONSTITUTIONAL MMR DEFICIENCY (CMMRD) — BIALLELIC PMS2: "
                "CHILDHOOD GBM WITH TMB >100 MUT/MB = PATHOGNOMONIC CMMRD: "
                "  CMMRD: biallelic germline MMR mutation (PMS2 most common ~55% of CMMRD); "
                "  GBM in child <10yr: extremely rare in general population → CMMRD must be excluded; "
                "  CMMRD-GBM: TMB >100 mut/Mb (often 200-500 mut/Mb) vs somatic GBM ~5 mut/Mb; "
                "  Hypermutation PATHOGNOMONIC: TMB >100 in childhood GBM = CMMRD constitutional until proven otherwise; "
                "  MMR IHC on all paediatric GBM: absent PMS2 or MLH1 staining → germline MMR testing; "
                "CAFÉ AU LAIT MACULES (CALMs) + CHILDHOOD BRAIN TUMOUR = CMMRD EXCLUDED: "
                "  CMMRD-CALMs: NF1-like pattern (multiple large pigmented macules); "
                "  KEY DDX NF1 vs CMMRD: "
                "    NF1: Lisch nodules, axillary freckling, plexiform neurofibromas; "
                "    CMMRD: NO Lisch nodules, NO neurofibromas; CALMs alone; brain tumour often precedes CRC; "
                "  CLINICAL RULE: CALMs ≥6 (>5mm prepubertal) + CNS tumour → test all 4 MMR genes germline; "
                "PD-1 IMMUNOTHERAPY — HYPERMUTATED CMMRD BRAIN TUMOURS: "
                "  Neoantigen burden: TMB >100 → hundreds of neoantigens → T-cell recognition; "
                "  PD-1 blockade (pembrolizumab / nivolumab): durable responses documented in CMMRD-GBM; "
                "  Case series: >50% response rate; some complete remissions in previously fatal paediatric GBM; "
                "  CMMRD-MB also responds if hypermutated; "
                "  EVERY RECURRENT PAEDIATRIC CNS TUMOUR → TMB testing → PD-1 consideration; "
                "PMS2 PSEUDOGENE PITFALL: "
                "  PMS2CL on 7p22 → Sanger sequencing MISSES exon 11-15 deletions; "
                "  MLPA + long-range PCR required for complete PMS2 germline analysis."
            ),
        },
    ]
    return {
        "atlas":       "Hereditary-Brain-CNS-Tumour-Predisposition-Atlas",
        "seed_range":  f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2))
    print(json.dumps(generate_breakdown(), indent=2))
    print(json.dumps(generate_definitions(), indent=2))
