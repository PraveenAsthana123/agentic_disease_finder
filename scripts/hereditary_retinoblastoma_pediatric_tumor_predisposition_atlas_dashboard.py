#!/usr/bin/env python3
"""Hereditary-Retinoblastoma-Pediatric-Solid-Tumor-Predisposition-Atlas -- Complete 8-Gene Reference
Inherited pediatric solid tumor predispositions: retinoblastoma, DICER1 syndrome,
rhabdoid tumor predisposition, Gorlin/NBCCS, medulloblastoma-SHH, SGB, and familial neuroblastoma.

RB1     (Retinoblastoma protein; 928aa; 13q14.2; AD LOF;
          Hereditary retinoblastoma; bilateral Rb PATHOGNOMONIC;
          45% bilateral = germline; trilateral Rb (pinealoblastoma) PATHOGNOMONIC;
          Second primary tumor risk 30-35% lifetime;
          seed SEED_BASE+0).
DICER1  (DICER1 ribonuclease; 1922aa; 14q32.13; AD LOF;
          DICER1 syndrome; PPB PATHOGNOMONIC <8yr;
          Thyroid nodule/DTC in adolescence; Ovarian SLCT;
          2nd-hit hotspot somatic metal-binding exons 24-25;
          seed SEED_BASE+1).
SMARCB1 (SWI/SNF chromatin remodeling; 385aa; 22q11.23; AD LOF;
          Rhabdoid tumor predisposition syndrome-1 (RTPS1);
          AT/RT brain + RTK kidney; IHC SMARCB1/BAF47 loss PATHOGNOMONIC;
          Radiation CI <36 months;
          seed SEED_BASE+2).
SMARCA4 (SWI/SNF ATPase subunit; 1647aa; 19p13.2; AD LOF;
          RTPS2; SCCOHT PATHOGNOMONIC young women;
          Hypercalcemia PATHOGNOMONIC SCCOHT;
          Coffin-Siris syndrome;
          seed SEED_BASE+3).
PTCH1   (Patched-1 receptor; 1447aa; 9q22.32; AD LOF;
          Gorlin syndrome / NBCCS;
          Multiple BCC PATHOGNOMONIC; OKC jaw cysts PATHOGNOMONIC; bifid ribs PATHOGNOMONIC;
          Radiation CONTRAINDICATED ABSOLUTELY -- massive BCC induction;
          seed SEED_BASE+4).
SUFU    (Suppressor of fused; 484aa; 10q24.32; AD LOF;
          SHH-medulloblastoma / Gorlin-like without jaw cysts;
          Desmoplastic/nodular medulloblastoma;
          Vismodegib NOT effective SUFU germline;
          seed SEED_BASE+5).
GPC3    (Glypican-3; 580aa; Xq26.2; XLR LOF;
          Simpson-Golabi-Behmel syndrome;
          Overgrowth PATHOGNOMONIC; embryonal tumor risk (Wilms, hepatoblastoma, NB);
          AFP + abdominal USS surveillance every 3 months first 8yr;
          seed SEED_BASE+6).
ALK     (Anaplastic Lymphoma Kinase; 1620aa; 2p23.2; AD GOF;
          Familial neuroblastoma; F1174L/R1275Q germline;
          75% penetrance germline ALK; crizotinib/alectinib active;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3702-3709)
"""
import random

SEED_BASE = 3702

ATLAS_GENES = [
    {
        "gene": "RB1",
        "protein": (
            "RB1 -- 13q14.2 AD LOF -- 928aa -- "
            "Retinoblastoma-protein-pRb-bilateral-Rb-45pct-germline-PATHOGNOMONIC-"
            "Trilateral-pinealoblastoma-PATHOGNOMONIC-"
            "Second-primary-tumor-30-35pct-lifetime-"
            "RB1-OMIM-614041"
        ),
        "locus": "13q14.2",
        "protein_size": (
            "928 aa / ~110 kDa / 13q14.2 RB1 Retinoblastoma Protein molecular context: "
            "STRUCTURE: "
            "  928 aa / ~110 kDa; tumour suppressor; pocket protein family (RB1/p107/p130); "
            "  Three functional domains: N-terminal, pocket (A+B box), C-terminal; "
            "  Pocket domain binds E2F transcription factors → represses E2F target genes → "
            "    G1 cell cycle arrest (G1/S checkpoint guardian); "
            "  CDK4/6-cyclin D → phosphorylates pRb → releases E2F → S phase entry; "
            "  LOF: E2F constitutively active → uncontrolled proliferation; "
            "KNUDSON TWO-HIT HYPOTHESIS (retinoblastoma model): "
            "  Hereditary Rb: 1st hit = germline RB1 mutation (heterozygous); "
            "    2nd hit = somatic LOH of remaining allele → retinoblastoma; "
            "  BILATERAL Rb: PATHOGNOMONIC for germline RB1 mutation; "
            "    ~45% bilateral retinoblastoma = germline; bilateral virtually 100% germline; "
            "  UNILATERAL Rb: 85% somatic; ~15% germline (may be unifocal presentation); "
            "    Test ALL retinoblastoma patients for germline RB1 (even unilateral); "
            "RETINOBLASTOMA CLINICAL FEATURES: "
            "  Median age diagnosis: 18 months (bilateral germline); 24 months (unilateral somatic); "
            "  Leukocoria (white pupillary reflex): presenting sign in 60%; PATHOGNOMONIC; "
            "  Strabismus: 2nd most common presenting sign; "
            "  Trilateral Rb: bilateral Rb + pineoblastoma (pineal gland); PATHOGNOMONIC germline; "
            "    Incidence ~5% germline Rb; median age 2-3 years; poor prognosis; "
            "    Screen with annual MRI brain until age 5 in all germline Rb; "
            "TREATMENT (retinoblastoma): "
            "  Intra-arterial chemotherapy (IAC, ophthalmic artery): standard for advanced intraocular Rb; "
            "  Intravitreal chemotherapy (melphalan): vitreous seeds; "
            "  Systemic chemotherapy + focal treatments: bilateral disease; "
            "  Enucleation: severe/extraocular disease; "
            "  Laser/cryotherapy: small focal tumors; "
            "SECOND PRIMARY TUMOR RISK (hereditary Rb): "
            "  30-35% lifetime risk of second primary cancers; "
            "  Osteosarcoma: most common (long bones, jaw); ~500x increased vs general population; "
            "  Soft tissue sarcoma, melanoma, breast cancer, bladder cancer, lung adenocarcinoma; "
            "  RADIATION-RELATED: prior radiation field cancers (fibrosarcoma, osteosarcoma); "
            "    Risk dramatically increases with radiation; avoid if possible in germline Rb; "
            "  CDK4/6i: pRb-deficient tumors may paradoxically not respond (no pRb target); "
            "    CDK4/6i (palbociclib/ribociclib) INACTIVE in Rb-null tumors; "
            "SURVEILLANCE (germline RB1): "
            "  Ophthalmologic exam under anesthesia (EUA): every 3-4 weeks until age 2 then 3 months until 5; "
            "  Annual MRI brain (ages 1-5): pinealoblastoma screening; "
            "  Annual whole-body MRI (age 6 onwards): second primary tumor surveillance; "
            "  Genetic counseling all first-degree relatives (AD)"
        ),
        "syndrome": "Hereditary-Retinoblastoma / AD-LOF / Bilateral-Rb-PATHOGNOMONIC / Trilateral-Rb-PATHOGNOMONIC / Second-Primary-Osteosarcoma-30-35pct",
        "inheritance": "AD LOF (haploinsufficiency second hit somatic); ~45% bilateral = germline; ~15% unilateral = germline; de novo mutation ~50% germline cases",
        "bmf_risk": "No BMF; retinal tumors; second primary tumor risk 30-35% lifetime (osteosarcoma, sarcoma, melanoma, breast); radiation-field sarcomas",
        "pathognomonic": "Bilateral retinoblastoma = germline RB1 PATHOGNOMONIC; leukocoria (white pupillary reflex) presenting sign; trilateral Rb (pinealoblastoma) = germline PATHOGNOMONIC; CDK4/6i INACTIVE in Rb-null tumors",
        "key_rule": "All bilateral retinoblastoma patients = assume germline RB1; ophthalmologic exam under anesthesia every 3-4 weeks until age 2 then every 3 months until age 5; annual MRI brain until age 5 for trilateral Rb screening",
        "key_avoid": "CDK4/6i in Rb-null second primary tumors (INACTIVE — pRb is the CDK4/6i target); unnecessary radiation in hereditary Rb (radiation-field sarcoma risk); delaying genetic testing in unilateral Rb",
        "surveillance": "EUA every 3-4 weeks (infant), 3 months (ages 2-5); MRI brain annual (ages 1-5); whole-body MRI annual (age 6+); osteosarcoma surveillance; cascade family testing",
        "targeted_rx": "Intra-arterial chemotherapy (IAC, ophthalmic artery); intravitreal melphalan; systemic carboplatin/vincristine/etoposide; enucleation (severe); laser/cryo (focal); no CDK4/6i in Rb-null",
    },
    {
        "gene": "DICER1",
        "protein": (
            "DICER1 -- 14q32.13 AD LOF -- 1922aa -- "
            "DICER1-syndrome-PPB-PATHOGNOMONIC-<8yr-"
            "Thyroid-nodule-DTC-adolescence-"
            "Ovarian-SLCT-Sertoli-Leydig-"
            "2nd-hit-somatic-hotspot-metal-binding-exons-24-25-"
            "DICER1-OMIM-606241"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "1922 aa / ~218 kDa / 14q32.13 DICER1 Endoribonuclease DICER1 molecular context: "
            "STRUCTURE: "
            "  1922 aa / ~218 kDa; RNase III enzyme; "
            "  Processes pre-miRNA hairpins → mature miRNA; "
            "  Key domains: PAZ domain (dsRNA binding), two RNase III domains (RIIIa + RIIIb), "
            "    metal-binding sites in RIIIb (E1705, E1707, D1709, E1813); "
            "  Metal-binding domain: processes 5p strand of miRNA; "
            "  LOF → loss of tumor-suppressive miRNA → derepression of oncogenes; "
            "TWO-HIT MECHANISM — UNIQUE PATTERN: "
            "  Germline 1st hit: TYPICALLY loss-of-function (frameshift/nonsense) heterozygous; "
            "  Somatic 2nd hit: HOTSPOT missense in metal-binding domain (RIIIb) exons 24-25; "
            "    E1705K, E1707K, D1709N, E1813K, G1809R most common somatic hotspots; "
            "  Hotspot 2nd hit = PARTIAL loss (RNase-IIIb deficient but not null); "
            "    This partial loss is SPECIFIC to DICER1 syndrome (unlike full LOH in RB1); "
            "    Germline loss-of-function alone is INSUFFICIENT for tumor development; "
            "DICER1 SYNDROME TUMOR SPECTRUM: "
            "  Pleuropulmonary Blastoma (PPB): "
            "    PATHOGNOMONIC: PPB in child <8 years → test DICER1; "
            "    Type I (cystic): age <2yr; favorable; may regress; "
            "    Type II/III: age 2-6yr; solid component; sarcomatous; poor prognosis; "
            "    ~70% PPB patients carry DICER1 germline mutation; "
            "    PPB requires systemic chemotherapy + resection; "
            "  Thyroid: "
            "    Multinodular goiter: adolescent girls; most common DICER1 manifestation; "
            "    Differentiated thyroid carcinoma (DTC): DICER1-associated; surveillance thyroid USS; "
            "  Ovarian: "
            "    Sertoli-Leydig Cell Tumor (SLCT): DICER1 germline; adolescent-young adult; "
            "    Androgenic symptoms (virilization): hair growth, clitoromegaly; "
            "    DICER1 hotspot somatic 2nd hit confirmed in majority of SLCT; "
            "  Other: ciliary body medulloepithelioma (eye), cystic nephroma, pineoblastoma, "
            "    embryonal rhabdomyosarcoma (cervical/uterine), nasal chondromesenchymal hamartoma; "
            "SURVEILLANCE: "
            "  Annual chest CT/CXR (ages 0-8): PPB surveillance; "
            "  Annual thyroid USS (from age 8): thyroid surveillance; "
            "  Annual pelvic USS (female, from age 8): SLCT; "
            "  Ophthalmologic exam (ages 0-10): ciliary body medulloepithelioma; "
            "  Full DICER1 syndrome surveillance protocol (PPLC consortium guidelines)"
        ),
        "syndrome": "DICER1-Syndrome / AD-LOF / PPB-PATHOGNOMONIC-Child / Thyroid-Nodule-Adolescent / Ovarian-SLCT / 2nd-Hit-Somatic-Hotspot-Metal-Binding",
        "inheritance": "AD LOF (germline) with somatic hotspot 2nd hit (NOT full LOH); ~70% PPB = germline DICER1; de novo mutations common; incomplete penetrance",
        "bmf_risk": "No BMF; multi-organ tumor predisposition; PPB (lung), thyroid, ovary, eye, kidney, brain, cervix; non-hematologic",
        "pathognomonic": "PPB in child <8 years = test DICER1 PATHOGNOMONIC; thyroid multinodular goiter in adolescent girl; ovarian SLCT with androgenic symptoms; somatic hotspot confined to metal-binding domain (E1705/E1707/D1709/E1813)",
        "key_rule": "PPB in any child <8 years demands germline DICER1 testing — ~70% of PPB carry germline DICER1; annual chest CT from birth to age 8 for DICER1 carriers; 2nd hit is always somatic hotspot (test tumor tissue for metal-binding domain hotspot)",
        "key_avoid": "Assuming DICER1 germline alone causes tumor (2nd somatic hit required — single germline mutation is insufficient); missing thyroid/ovarian surveillance in adolescent females",
        "surveillance": "Annual chest CT ages 0-8; thyroid USS annual from age 8; pelvic USS annual (females) from age 8; ophthalmologic exam ages 0-10; MRI brain for pineoblastoma risk",
        "targeted_rx": "PPB: sarcoma-type chemotherapy (IFO/ETO/VCR) + resection; thyroid: thyroidectomy (carcinoma); ovarian SLCT: surgery ± BEP chemotherapy; specific targets investigational",
    },
    {
        "gene": "SMARCB1",
        "protein": (
            "SMARCB1 -- 22q11.23 AD LOF -- 385aa -- "
            "SWI/SNF-BAF47-INI1-chromatin-remodeling-"
            "Rhabdoid-Tumor-Predisposition-Syndrome-1-RTPS1-"
            "AT/RT-brain-RTK-kidney-PATHOGNOMONIC-"
            "IHC-SMARCB1-loss-PATHOGNOMONIC-"
            "Radiation-CI-<36-months-neurological-catastrophe-"
            "SMARCB1-OMIM-601607"
        ),
        "locus": "22q11.23",
        "protein_size": (
            "385 aa / ~44 kDa / 22q11.23 SMARCB1 SWI/SNF Related Matrix Associated Actin-Dependent Regulator of Chromatin Subfamily B Member 1 molecular context: "
            "STRUCTURE: "
            "  385 aa / ~44 kDa; also named BAF47, INI1, SNF5; "
            "  Core subunit of BAF (SWI/SNF) chromatin remodeling complex; "
            "  Two RVH (Rpt) domains flank central domain; C-terminal coiled-coil; "
            "  SMARCB1 is the defining tumor suppressor of the BAF complex; "
            "  BAF complex: epigenetic regulator — opens chromatin, activates tumor suppressors; "
            "  SMARCB1 LOF → BAF complex non-functional → EZH2 unopposed → H3K27me3 silencing of "
            "    tumor suppressors → rhabdoid tumors; "
            "  EZH2 inhibitor (tazemetostat) rationale: SMARCB1-null tumors depend on EZH2; "
            "RHABDOID TUMOR PREDISPOSITION SYNDROME 1 (RTPS1): "
            "  Germline heterozygous SMARCB1 LOF → predisposition to rhabdoid tumors; "
            "  Tumors require somatic 2nd hit (LOH 22q / SMARCB1 deletion); "
            "  ATYPICAL TERATOID/RHABDOID TUMOR (AT/RT): "
            "    CNS malignancy; most common in children <3 years (median age 15 months); "
            "    ~50-70% AT/RT harbor germline SMARCB1 mutation (RTPS1); "
            "    Especially: infants <18 months; bilateral or multifocal AT/RT → virtually all germline; "
            "    SMARCB1/BAF47 IHC: loss of nuclear staining PATHOGNOMONIC for rhabdoid tumors; "
            "  RHABDOID TUMOR OF KIDNEY (RTK): "
            "    Malignant rhabdoid tumor of kidney (MRT/RTK): ~25-35% germline SMARCB1; "
            "    Often infants; aggressive; SMARCB1 IHC loss; "
            "  EXTRA-CNS/EXTRA-RENAL RHABDOID TUMORS: liver, soft tissue, rare locations; "
            "    All rhabdoid tumors: test germline SMARCB1 (and SMARCA4 if SMARCB1 negative); "
            "DE NOVO vs INHERITED: "
            "  ~40% RTPS1 germline = DE NOVO (not familial); "
            "  Familial cases: AD; penetrance high (most germline carriers develop tumor); "
            "  22q11 deletions: some RTPS1 due to 22q11 microdeletion (MLPA mandatory); "
            "TREATMENT CHALLENGES: "
            "  No standard curative treatment; aggressive multimodal therapy; "
            "  Radiation in infants <36 months → CATASTROPHIC neurological outcomes; "
            "    Cognitive/motor damage from cranial radiation in infancy; avoid if possible; "
            "  Current protocols try to delay/avoid radiation in <36 months; "
            "  Tazemetostat (EZH2 inhibitor): rationale in SMARCB1-null; trials ongoing; "
            "  Alisertib (Aurora A kinase inhibitor): investigational; "
            "  ICB (pembrolizumab): investigational in rhabdoid tumors; "
            "DIAGNOSIS: "
            "  SMARCB1/BAF47 IHC: complete loss of nuclear staining = PATHOGNOMONIC; "
            "  Molecular: SMARCB1 sequencing + MLPA (deletions common); "
            "  Germline testing: all rhabdoid tumors, especially age <3 years"
        ),
        "syndrome": "Rhabdoid-Tumor-Predisposition-Syndrome-1 / AD-LOF / AT/RT-Brain / RTK-Kidney / SMARCB1-IHC-Loss-PATHOGNOMONIC / Radiation-CI-<36-months",
        "inheritance": "AD LOF; ~40% de novo; familial AD; 22q11 deletions (MLPA mandatory); somatic 2nd hit LOH 22q; ~50-70% AT/RT in infants = germline",
        "bmf_risk": "No BMF; rhabdoid tumors brain (AT/RT), kidney (RTK), soft tissue; virtually all fatal without treatment; high-risk pediatric oncology",
        "pathognomonic": "AT/RT in infant <18 months = assume germline SMARCB1; SMARCB1/BAF47 IHC complete nuclear loss PATHOGNOMONIC rhabdoid tumor; bilateral/multifocal AT/RT = germline virtually certain",
        "key_rule": "Radiation CONTRAINDICATED in infants <36 months with AT/RT — catastrophic cognitive/motor developmental damage; SMARCB1 IHC is mandatory on all rhabdoid histology; germline test all AT/RT and RTK regardless of age",
        "key_avoid": "Cranial radiation <36 months (catastrophic neurotoxicity); missing germline testing in any infant with rhabdoid tumor; confusing with Wilms tumor (RTK is SMARCB1-null, Wilms is NOT)",
        "surveillance": "MRI brain + spine every 3 months after treatment (relapse monitoring); renal USS every 6 months (RTK risk); cascade testing first-degree relatives; pelvic surveillance in females",
        "targeted_rx": "Multimodal (carboplatin/vincristine/cyclophosphamide/etoposide); tazemetostat (EZH2 inhibitor, SMARCB1-null rationale); high-dose chemotherapy + stem cell rescue; avoid radiation <36 months",
    },
    {
        "gene": "SMARCA4",
        "protein": (
            "SMARCA4 -- 19p13.2 AD LOF -- 1647aa -- "
            "SWI/SNF-BRG1-BAF190-ATPase-"
            "Rhabdoid-Tumor-Predisposition-Syndrome-2-RTPS2-"
            "SCCOHT-Small-Cell-Carcinoma-Ovary-Hypercalcemic-Type-PATHOGNOMONIC-young-women-"
            "Hypercalcemia-PATHOGNOMONIC-SCCOHT-"
            "Coffin-Siris-syndrome-"
            "SMARCA4-OMIM-603254"
        ),
        "locus": "19p13.2",
        "protein_size": (
            "1647 aa / ~185 kDa / 19p13.2 SMARCA4 SWI/SNF Related Matrix Associated Actin-Dependent Regulator of Chromatin Subfamily A Member 4 molecular context: "
            "STRUCTURE: "
            "  1647 aa / ~185 kDa; also named BRG1 (Brahma-related gene 1), BAF190; "
            "  Catalytic ATPase subunit of BAF (SWI/SNF) chromatin remodeling complex; "
            "  Helicase-SANT (HSA) domain, DEXDc (helicase), BROMO domain (acetyl-lysine binding); "
            "  ATPase activity drives nucleosome repositioning → opens chromatin; "
            "  SMARCA4/BRG1 LOF → BAF complex non-functional → same EZH2-dependent pathway as SMARCB1 LOF; "
            "SMARCA4 LOF TUMOR SPECTRUM: "
            "  SMALL CELL CARCINOMA OF THE OVARY HYPERCALCEMIC TYPE (SCCOHT): "
            "    PATHOGNOMONIC DIAG: SCCOHT is ~100% SMARCA4-deficient (IHC loss); "
            "    Extremely rare but most common ovarian malignancy in women under 40; "
            "    SMARCA4 germline: ~43% SCCOHT patients carry germline SMARCA4 mutation; "
            "    Median age: 24 years (range: infancy to 50s); "
            "    Hypercalcemia: parathyroid hormone-related protein (PTHrP) secretion → "
            "      hypercalcemia in ~60% = PATHOGNOMONIC association with SCCOHT; "
            "    Diagnosis: SMARCA4/BRG1 IHC loss on ovarian tumor PATHOGNOMONIC; "
            "    Prognosis: extremely poor (5-yr OS ~10-15% advanced stage); "
            "  AT/RT (CNS RHABDOID): SMARCA4 LOF → AT/RT (less common than SMARCB1); "
            "    ~5-10% AT/RT = SMARCA4-deficient (SMARCB1-intact); "
            "    SMARCA4-IHC: loss of nuclear BRG1 staining; "
            "  THORACIC SARCOMA (SMARCA4-deficient undifferentiated thoracic sarcoma): "
            "    Aggressive; young men; mediastinal/pulmonary; SMARCA4 IHC loss; "
            "  COFFIN-SIRIS SYNDROME (germline heterozygous): "
            "    Intellectual disability, absent/hypoplastic 5th fingernail, coarse facies; "
            "    Predominantly non-cancer manifestation (different mutation class vs SCCOHT); "
            "TREATMENT (SCCOHT): "
            "  High-dose chemotherapy with autologous SCT: best current outcomes in young patients; "
            "  Whole abdominal radiation: historically used; significant toxicity; "
            "  EZH2 inhibitors (tazemetostat): phase 2 trials; rationale = SMARCA4-null EZH2 dependence; "
            "  Immunotherapy (pembrolizumab): case reports + early trials; "
            "  SMARCA4-null specifically: BET bromodomain inhibitors, CDK4/6i investigated; "
            "GENETICS: "
            "  SMARCA4 germline: AD LOF; ~43% SCCOHT = germline; "
            "  Family history: recommend SMARCA4 germline testing in all SCCOHT; "
            "  Sibling/maternal relatives: germline risk; prophylactic BSO in high-risk females after family planning complete; "
            "DIAGNOSIS: "
            "  SMARCA4/BRG1 IHC: complete nuclear loss PATHOGNOMONIC; "
            "  Molecular: SMARCA4 sequencing (and SMARCA2/BRM for thoracic sarcoma); "
            "  SCCOHT panels include SMARCA4 now"
        ),
        "syndrome": "Rhabdoid-Tumor-Predisposition-Syndrome-2 / SCCOHT-PATHOGNOMONIC-Young-Women / AD-LOF / Hypercalcemia-PATHOGNOMONIC / AT/RT / Coffin-Siris-Syndrome",
        "inheritance": "AD LOF; ~43% SCCOHT = germline SMARCA4; Coffin-Siris = de novo germline common; SMARCA4 IHC loss in ~100% SCCOHT and thoracic sarcoma",
        "bmf_risk": "No BMF; SCCOHT (ovary), AT/RT (brain), thoracic sarcoma; extremely aggressive; non-hematologic",
        "pathognomonic": "SCCOHT + hypercalcemia in woman <40 = SMARCA4 loss PATHOGNOMONIC; SMARCA4/BRG1 IHC complete nuclear loss diagnostic for SCCOHT and thoracic sarcoma; absent 5th fingernail in Coffin-Siris",
        "key_rule": "Test SMARCA4 germline in ALL SCCOHT — ~43% germline; offer prophylactic BSO after family planning if germline confirmed; test SMARCA4 IHC on ALL ovarian tumors in women under 40 with hypercalcemia",
        "key_avoid": "Missing SMARCA4 IHC in young women with undifferentiated ovarian tumors + hypercalcemia; failing to test germline in SCCOHT; assuming Coffin-Siris is never cancer-predisposing (rare overlap with RTPS2)",
        "surveillance": "Annual pelvic USS + CA-125 (SMARCA4 germline females); SMARCA4 family cascade testing; BRG1 IHC on any new thoracic sarcoma; brain MRI if neurodevelopmental concern (Coffin-Siris + AT/RT overlap)",
        "targeted_rx": "SCCOHT: high-dose chemo + auto-SCT; tazemetostat (EZH2 inhibitor, investigational); pembrolizumab (phase 2); BET inhibitors investigational; prophylactic BSO post-childbearing (germline carriers)",
    },
    {
        "gene": "PTCH1",
        "protein": (
            "PTCH1 -- 9q22.32 AD LOF -- 1447aa -- "
            "Patched-1-receptor-SHH-pathway-"
            "Gorlin-syndrome-NBCCS-"
            "Multiple-BCC-PATHOGNOMONIC-"
            "OKC-jaw-keratocysts-PATHOGNOMONIC-"
            "Bifid-ribs-PATHOGNOMONIC-"
            "Radiation-CONTRAINDICATED-ABSOLUTELY-massive-BCC-induction-"
            "PTCH1-OMIM-601309"
        ),
        "locus": "9q22.32",
        "protein_size": (
            "1447 aa / ~162 kDa / 9q22.32 PTCH1 Patched-1 Receptor molecular context: "
            "STRUCTURE: "
            "  1447 aa / ~162 kDa; 12-transmembrane sterol-sensing domain (SSD) protein; "
            "  PTCH1 = receptor for SHH (Sonic Hedgehog) ligand; "
            "  Without SHH: PTCH1 suppresses SMO (Smoothened) → no pathway activation; "
            "  With SHH: SHH binds PTCH1 → PTCH1 inhibition lifted → SMO active → GLI1/2 nuclear; "
            "  PTCH1 LOF → SMO constitutively active → GLI1/2 → SHH pathway uncontrolled → "
            "    proliferation/survival → basal cell carcinoma, medulloblastoma, etc.; "
            "GORLIN SYNDROME / NEVOID BASAL CELL CARCINOMA SYNDROME (NBCCS): "
            "  PATHOGNOMONIC diagnostic criteria (all require only ONE): "
            "    Multiple BCC (≥5 lifetime, or 1 BCC <30 years): "
            "      BCC begin in early 20s (sun-exposed skin); "
            "      Hundreds to thousands of BCCs over lifetime; "
            "    Odontogenic keratocysts (OKC) / jaw cysts: "
            "      PATHOGNOMONIC: first OKC often appears age 7-12 years; "
            "      Multiple, recurrent jaw cysts; require surgical management; "
            "      Panoramic dental X-ray: multiple OKC PATHOGNOMONIC; "
            "    Calcification of falx cerebri (on skull X-ray/CT): "
            "      Lamellar calcification; PATHOGNOMONIC; often asymptomatic; "
            "    Bifid/fused ribs; "
            "    1st-degree relative with PTCH1 mutation; "
            "  MEDULLOBLASTOMA (GORLIN): "
            "    ~5% Gorlin patients develop medulloblastoma; "
            "    Desmoplastic/nodular medulloblastoma subtype; "
            "    Younger age (infancy/toddler); "
            "    SHH-pathway medulloblastoma; responds to vismodegib/sonidegib (SMO inhibitors); "
            "RADIATION — ABSOLUTELY CONTRAINDICATED: "
            "  Historical cases: cranial radiation for medulloblastoma → "
            "    HUNDREDS of BCCs developed in irradiated skin field; "
            "  Mechanism: radiation → DNA damage → 2nd hit somatic PTCH1 LOH in SHH-primed skin; "
            "  NEVER irradiate Gorlin patients without extreme necessity; "
            "  Alternative medulloblastoma treatment: surgery + chemotherapy AVOIDING radiation; "
            "  Avoid unnecessary CT scans (minimize radiation exposure); "
            "MANAGEMENT (BCCs): "
            "  Vismodegib (Erivedge, FDA2012): SMO inhibitor; reduces BCC burden; "
            "    Active in Gorlin syndrome BCCs; reduces OKC also; "
            "    Side effects: muscle cramps, alopecia, ageusia (taste loss), teratogenic (contraception MANDATORY); "
            "  Sonidegib (Odomzo): 2nd SMO inhibitor; similar; "
            "  Topical treatments: 5-FU, imiquimod, photodynamic therapy (visible light); "
            "  Surgical: excision for concerning BCCs; Mohs for facial; "
            "  Photoprotection: strict sun avoidance (UV trigger); "
            "SURVEILLANCE: "
            "  Annual dermatologic exam (from diagnosis); "
            "  Annual MRI brain (ages 0-5 for medulloblastoma screening); "
            "  Panoramic dental X-ray every 12-18 months from age 7 (OKC surveillance); "
            "  Ophthalmologic exam (coloboma, cataract in some Gorlin); "
            "  Echocardiogram (cardiac fibromas)"
        ),
        "syndrome": "Gorlin-Syndrome / NBCCS / AD-LOF / Multiple-BCC-PATHOGNOMONIC / OKC-Jaw-Keratocysts-PATHOGNOMONIC / Falx-Calcification-PATHOGNOMONIC / Bifid-Ribs / Radiation-CONTRAINDICATED-ABSOLUTELY",
        "inheritance": "AD LOF (haploinsufficiency); ~1/30,000-60,000; ~50% de novo PTCH1 mutations; SMO constitutively active without PTCH1",
        "bmf_risk": "No BMF; BCC (skin), medulloblastoma (brain, ~5%), jaw OKC, ovarian fibromas, cardiac fibromas; non-hematologic",
        "pathognomonic": "Multiple BCC before age 30 = Gorlin PATHOGNOMONIC; jaw OKC (odontogenic keratocysts) first sign in children PATHOGNOMONIC; falx cerebri calcification on imaging PATHOGNOMONIC; bifid ribs on X-ray",
        "key_rule": "Radiation ABSOLUTELY CONTRAINDICATED in Gorlin syndrome — irradiation causes massive BCC induction in radiation field (hundreds of BCCs reported); always ask about Gorlin diagnosis before radiation for medulloblastoma",
        "key_avoid": "Radiation in Gorlin patients (ABSOLUTELY CI — hundreds BCCs); vismodegib without contraception counseling (severe teratogen — male AND female patients); unnecessary CT scans (minimize ionizing radiation exposure)",
        "surveillance": "Annual dermatology; panoramic dental X-ray every 12-18 months from age 7; MRI brain annually ages 0-5; echocardiogram (cardiac fibromas); ophthalmologic exam; pelvic USS females (ovarian fibromas)",
        "targeted_rx": "Vismodegib/sonidegib (SMO inhibitors — reduce BCC burden, active in Gorlin); topical 5-FU/imiquimod; photodynamic therapy; surgery/Mohs; strict sun protection; avoid radiation",
    },
    {
        "gene": "SUFU",
        "protein": (
            "SUFU -- 10q24.32 AD LOF -- 484aa -- "
            "Suppressor-of-fused-SHH-pathway-"
            "SHH-Medulloblastoma-Gorlin-like-fewer-BCC-jaw-cysts-"
            "Desmoplastic-nodular-medulloblastoma-"
            "Vismodegib-NOT-effective-SUFU-germline-medulloblastoma-"
            "SUFU-OMIM-607035"
        ),
        "locus": "10q24.32",
        "protein_size": (
            "484 aa / ~55 kDa / 10q24.32 SUFU Suppressor of Fused Homolog molecular context: "
            "STRUCTURE: "
            "  484 aa / ~55 kDa; negative regulator of SHH signaling DOWNSTREAM of SMO; "
            "  SUFU binds GLI proteins (GLI1, GLI2, GLI3) in cytoplasm → sequesters them; "
            "  Prevents GLI nuclear translocation → no SHH target gene activation; "
            "  SUFU LOF → GLI1/2 constitutively nuclear → SHH pathway uncontrolled; "
            "  KEY DISTINCTION from PTCH1: "
            "    PTCH1 upstream of SMO; "
            "    SUFU downstream of SMO → vismodegib (SMO inhibitor) does NOT work for SUFU LOF; "
            "    Vismodegib blocks SMO → irrelevant when bypass is downstream at GLI level; "
            "SUFU GERMLINE LOF — MEDULLOBLASTOMA PREDISPOSITION: "
            "  SHH-subgroup medulloblastoma: ~10-15% carry germline SUFU mutation; "
            "  Germline SUFU medulloblastoma: "
            "    Age: typically infants and young children (median ~1-2 years); "
            "    Histology: desmoplastic/nodular medulloblastoma (not classic MB); "
            "    Location: cerebellar; "
            "    Prognosis: SUFU-germline SHH-MB GENERALLY favorable vs somatic SHH-MB in adults; "
            "SUFU vs PTCH1 GORLIN-LIKE: "
            "  SUFU germline: presents similarly to Gorlin (PTCH1) but FEWER skin BCCs; "
            "  OKC jaw cysts: LESS COMMON in SUFU than PTCH1 Gorlin; "
            "  Falx calcification: present in some SUFU; "
            "  Medulloblastoma: ~25% SUFU carriers (vs ~5% PTCH1 Gorlin) — HIGHER MB risk; "
            "  Ovarian fibromas: present in SUFU as in PTCH1; "
            "  Critical distinction: SUFU and PTCH1 carriers may BOTH satisfy Gorlin criteria; "
            "    Molecular testing required to distinguish → guides treatment (vismodegib choice); "
            "TREATMENT — CRITICAL DISTINCTION: "
            "  PTCH1-Gorlin medulloblastoma → vismodegib/sonidegib (SMO inhibitors) active; "
            "  SUFU-germline medulloblastoma → vismodegib NOT effective (downstream of SMO); "
            "    Use conventional MB treatment (craniospinal radiation + chemotherapy in older children); "
            "  Radiation: SUFU carriers may also be more sensitive to radiation BCC induction; "
            "    Use caution; avoid cranial radiation in infants <36 months if possible; "
            "SURVEILLANCE: "
            "  Annual MRI brain (ages 0-10): medulloblastoma (higher risk than PTCH1); "
            "  Annual dermatologic exam (BCCs — less common but present); "
            "  Panoramic X-ray every 12-18 months from age 8 (OKC — less common); "
            "  Pelvic USS females: ovarian fibromas; "
            "  Echocardiogram: cardiac fibromas; "
            "MOLECULAR TESTING: "
            "  SUFU sequencing + MLPA (deletions); "
            "  SHH-MB tumor molecular profiling (DNA methylation): identifies SHH subgroup; "
            "  Germline test ALL desmoplastic/nodular MB in children under 3 years for SUFU + PTCH1"
        ),
        "syndrome": "SUFU-Medulloblastoma-SHH / AD-LOF / Desmoplastic-Nodular-MB / Gorlin-like-Fewer-BCC / Vismodegib-NOT-Effective-SUFU / Higher-MB-Risk-Than-PTCH1-Gorlin",
        "inheritance": "AD LOF; ~10-15% SHH-MB = germline SUFU; de novo mutations common; penetrance high for MB (~25% carriers)",
        "bmf_risk": "No BMF; medulloblastoma (high risk ~25%), BCC (less than PTCH1), ovarian fibromas, jaw OKC (rare); non-hematologic",
        "pathognomonic": "Desmoplastic/nodular medulloblastoma in infant + Gorlin-like features = test SUFU (not just PTCH1); SUFU carriers have ~25% MB risk (higher than PTCH1 5%); vismodegib inactive in SUFU-germline MB",
        "key_rule": "Vismodegib (SMO inhibitor) is NOT effective in SUFU-germline SHH-MB — SUFU acts downstream of SMO; always test both PTCH1 AND SUFU in Gorlin-like patients with medulloblastoma to guide treatment choice",
        "key_avoid": "Prescribing vismodegib for SUFU-germline medulloblastoma (ineffective — downstream of drug target SMO); missing SUFU testing in infant with desmoplastic/nodular MB; confusing SUFU-Gorlin with PTCH1-Gorlin on clinical grounds alone",
        "surveillance": "Annual MRI brain ages 0-10 (MB risk ~25%); annual dermatology (BCCs less common); panoramic X-ray from age 8; pelvic USS females; echocardiogram; molecular profiling all SHH-MB for germline",
        "targeted_rx": "SHH-MB: surgery + craniospinal radiation + chemotherapy (in older children/adults); SUFU-germline: avoid radiation in infants; NO vismodegib; investigational GLI inhibitors",
    },
    {
        "gene": "GPC3",
        "protein": (
            "GPC3 -- Xq26.2 XLR LOF -- 580aa -- "
            "Glypican-3-heparan-sulfate-proteoglycan-"
            "Simpson-Golabi-Behmel-syndrome-SGB-"
            "Overgrowth-PATHOGNOMONIC-coarse-facies-macrocephaly-polydactyly-"
            "Embryonal-tumor-risk-Wilms-hepatoblastoma-neuroblastoma-"
            "AFP-surveillance-every-3-months-first-8yr-"
            "GPC3-OMIM-300037"
        ),
        "locus": "Xq26.2",
        "protein_size": (
            "580 aa / ~70 kDa (preprotein) / Xq26.2 GPC3 Glypican-3 molecular context: "
            "STRUCTURE: "
            "  580 aa / ~70 kDa; GPI-anchored heparan sulfate proteoglycan; "
            "  N-terminal cysteine-rich domain; central core protein; C-terminal GPI anchor signal; "
            "  Binds growth factors (IGF2, HGF, BMP4, Wnt ligands) and modulates their signaling; "
            "  GPC3 normally inhibits IGF2/Wnt/Hedgehog signaling → growth restraint; "
            "  GPC3 LOF → IGF2/Wnt/Hedgehog unrestrained → overgrowth + embryonal tumor predisposition; "
            "  GPC3 is also a DIAGNOSTIC MARKER: hepatocellular carcinoma IHC positive (GPC3 re-expressed); "
            "SIMPSON-GOLABI-BEHMEL SYNDROME (SGB, SGBS): "
            "  X-linked recessive; males severely affected; females: variable phenotype (mosaic); "
            "  OVERGROWTH PHENOTYPE (PATHOGNOMONIC): "
            "    Macrosomia (increased birth weight, >97th percentile); "
            "    Macrocephaly; coarse facial features (broad/flat nasal bridge, large jaw, "
            "      macroglossia, wide mouth); "
            "    Tall stature post-natally; "
            "    Polydactyly (extra digits, especially post-axial); "
            "    Midline defects: cleft palate, bifid sternum, diastasis recti, umbilical hernia; "
            "    Organomegaly: hepatomegaly, splenomegaly, nephromegaly, cardiomegaly; "
            "    Mild-moderate intellectual disability (variable); "
            "  EMBRYONAL TUMOR PREDISPOSITION (~10% lifetime): "
            "    Wilms tumor (nephroblastoma): most common embryonal tumor in SGB; "
            "      Annual abdominal USS from birth to age 8; AFP if hepatoblastoma risk; "
            "    Hepatoblastoma: elevated AFP → surveillance; "
            "    Neuroblastoma: rare; catecholamine/urine VMA/HVA surveillance; "
            "    Adrenal cortical carcinoma: rare; "
            "CARDIAC: "
            "  Congenital heart defects: ~30-40% SGB; VSD, ASD, dilated cardiomyopathy; "
            "  Cardiac overgrowth → neonatal cardiomegaly; "
            "  Echocardiogram at diagnosis mandatory; "
            "NEONATAL COMPLICATIONS: "
            "  Neonatal hypoglycemia: macrosomia → Beckwith-Wiedemann overlap clinically; "
            "  Respiratory difficulties: macroglossia → airway; "
            "  Distinguish from Beckwith-Wiedemann: genetic test (GPC3 vs 11p15 region); "
            "GPC3 AS TUMOR MARKER (hepatocellular carcinoma): "
            "  GPC3 re-expressed in ~70% HCC (not germline, tumor somatic re-expression); "
            "  GPC3 IHC used to distinguish HCC from other hepatic tumors; "
            "  GPC3-targeted CAR-T cell therapy: phase 1/2 trials for GPC3+ HCC; "
            "DIAGNOSIS: "
            "  Clinical features + GPC3 gene sequencing; "
            "  Females: GPC3 testing (XLR — carrier females may have mild manifestations); "
            "  Distinguish from Beckwith-Wiedemann (11p15 methylation), Sotos (NSD1)"
        ),
        "syndrome": "Simpson-Golabi-Behmel-Syndrome / XLR-LOF / Overgrowth-PATHOGNOMONIC / Embryonal-Tumor-Risk-10pct / Wilms-Hepatoblastoma-NB / AFP-USS-Surveillance-Every-3-Months-First-8yr",
        "inheritance": "X-linked recessive; males severely affected; carrier females variable (mosaic X-inactivation); GPC3 deletions common (MLPA mandatory); point mutations",
        "bmf_risk": "No BMF; overgrowth syndrome; embryonal tumors (Wilms 5%, hepatoblastoma, NB); cardiac defects (~30-40%); non-hematologic",
        "pathognomonic": "Macrosomia + coarse facies + macroglossia + polydactyly in male infant = SGB PATHOGNOMONIC; nephromegaly + elevated AFP = hepatoblastoma/Wilms surveillance trigger; organomegaly in male overgrowth infant",
        "key_rule": "AFP + abdominal USS every 3 months for first 8 years in ALL GPC3 germline males — Wilms and hepatoblastoma risk up to age 8; cardiac echo at diagnosis (30-40% CHD); distinguish from Beckwith-Wiedemann (11p15 region, not GPC3)",
        "key_avoid": "Missing abdominal USS/AFP surveillance (embryonal tumor catches at early stage vs late); confusing SGB with Beckwith-Wiedemann without molecular testing; missing cardiac evaluation at diagnosis",
        "surveillance": "Abdominal USS + AFP every 3 months ages 0-8; echocardiogram at diagnosis; renal function; urine catecholamines (neuroblastoma surveillance); developmental assessment; cascade female carrier testing",
        "targeted_rx": "Wilms: nephrectomy + chemotherapy (vincristine/actinomycin/doxorubicin per NWTS/SIOP); hepatoblastoma: cisplatin + surgery; neuroblastoma: risk-stratified; GPC3-targeted CAR-T investigational for adult GPC3+ HCC",
    },
    {
        "gene": "ALK",
        "protein": (
            "ALK -- 2p23.2 AD GOF -- 1620aa -- "
            "Anaplastic-Lymphoma-Kinase-"
            "Familial-Neuroblastoma-"
            "F1174L-R1275Q-most-common-germline-GOF-"
            "75pct-penetrance-germline-ALK-"
            "Crizotinib-alectinib-lorlatinib-ALK-inhibitors-active-"
            "ALK-OMIM-105590"
        ),
        "locus": "2p23.2",
        "protein_size": (
            "1620 aa / ~177 kDa / 2p23.2 ALK Anaplastic Lymphoma Kinase molecular context: "
            "STRUCTURE: "
            "  1620 aa / ~177 kDa; receptor tyrosine kinase (RTK); LTK subfamily; "
            "  Extracellular: MAM (meprin/A5/mu phosphatase) domains + LDLa domain + EGF-like; "
            "  Single transmembrane helix; intracellular kinase domain (aa 1122-1376); "
            "  Normal expression: neural tissue during development (Schwann cells, sympathoadrenal); "
            "  Downregulated postnatally; re-activated in neuroblastoma; "
            "  ALK activates JAK/STAT, RAS/MAPK, PI3K/AKT pathways; "
            "FAMILIAL NEUROBLASTOMA — ALK GERMLINE GOF: "
            "  Germline ALK mutations = major cause of hereditary neuroblastoma; "
            "  ~3-5% neuroblastoma overall have germline ALK mutations; "
            "  ~50% of familial neuroblastoma (NB with affected family member) = ALK germline; "
            "  Penetrance: ~75% (incomplete — not all germline carriers develop NB); "
            "KEY GERMLINE GOF MUTATIONS: "
            "  F1174L (c.3522C>T): most common germline + somatic ALK mutation in NB; "
            "    Highly activating; associated with high-risk NB; poor prognosis somatic; "
            "    Germline F1174L: variable age at onset; "
            "  R1275Q (c.3824G>A): 2nd most common; germline + somatic; moderate activation; "
            "  Other: F1245V, F1245C, K1062M, I1171N — all GOF activating mutations; "
            "CLINICAL SPECTRUM: "
            "  Neuroblastoma (NB): embryonal tumor of sympathetic nervous system; "
            "    Sites: adrenal medulla (most common), abdominal para-aortic ganglia, posterior mediastinum, "
            "      cervical/pelvic ganglia; "
            "    ALK-germline NB: often presents early childhood (infants to age 5); "
            "    May be bilateral adrenal tumors (PATHOGNOMONIC for hereditary/germline NB); "
            "    MYCN amplification frequently co-occurs with F1174L somatic (highest risk); "
            "  Age range germline: infancy to adolescence; mean ~2 years; "
            "TREATMENT — ALK INHIBITORS: "
            "  ALK inhibitors active in ALK-mutant NB: "
            "    Crizotinib (1st gen): active in ALK-amp/mut NB; FDA-approved NSCLC (repurposed for NB); "
            "    Alectinib (2nd gen): CNS penetration; trials in NB; "
            "    Lorlatinib (3rd gen): CNS penetrant; F1174L activity; NBLAST phase 1/2; "
            "  ALK-F1174L: crizotinib PARTIALLY active; F1174L is slightly crizotinib-resistant; "
            "    lorlatinib better for F1174L; "
            "  Dinutuximab (anti-GD2 MAb): standard high-risk NB; FDA2015; "
            "  MIBG therapy: targeted radiotherapy for MIBG-avid NB; "
            "SURVEILLANCE (germline ALK carriers): "
            "  Annual catecholamine urine (VMA/HVA): NB biomarkers; "
            "  Annual MIBG scan or abdominal USS/MRI: surveillance for asymptomatic NB; "
            "  Family cascade testing (AD inheritance); "
            "  Siblings of NB patients: test for ALK germline; "
            "PHOX2B (CCHS/autonomic NB predisposition): "
            "  ALK is the major hereditary NB gene; PHOX2B germline also predisposes to NB; "
            "  Test BOTH ALK and PHOX2B in familial/bilateral NB; "
            "  PHOX2B: also Hirschsprung disease and congenital central hypoventilation (CCHS) phenotype"
        ),
        "syndrome": "Familial-Neuroblastoma / AD-GOF / ALK-F1174L-R1275Q / 75pct-Penetrance / Bilateral-Adrenal-NB-PATHOGNOMONIC / Crizotinib-Alectinib-Lorlatinib-Active",
        "inheritance": "AD GOF (gain-of-function RTK); ~50% familial NB = germline ALK; ~3-5% all NB; incomplete penetrance ~75%; F1174L and R1275Q most common",
        "bmf_risk": "No BMF; neuroblastoma (sympathetic nervous system); bilateral adrenal NB in hereditary; MYCN co-amplification high-risk; non-hematologic",
        "pathognomonic": "Bilateral adrenal neuroblastoma = germline predisposition PATHOGNOMONIC (test ALK + PHOX2B); familial NB (≥2 affected relatives) = ALK germline in ~50%; elevated urine VMA/HVA + MIBG-avid mass",
        "key_rule": "Test ALK germline in ALL familial neuroblastoma and ALL bilateral adrenal NB; lorlatinib (3rd gen ALK inhibitor) has best F1174L activity — use over crizotinib for F1174L-mutant NB; test both ALK and PHOX2B in familial NB",
        "key_avoid": "Crizotinib monotherapy for F1174L ALK (partially resistant — prefer lorlatinib); missing bilateral adrenal NB as hereditary presentation; failing to test siblings/parents when germline ALK identified",
        "surveillance": "Annual urine VMA/HVA (catecholamine biomarkers); annual abdominal MRI or MIBG scan; cascade family testing (AD); test PHOX2B concurrently in familial NB; cardiac monitoring (ALK inhibitors)",
        "targeted_rx": "ALK inhibitors (crizotinib, alectinib, lorlatinib — lorlatinib preferred F1174L); dinutuximab (anti-GD2, standard high-risk NB); MIBG therapy; isotretinoin (maintenance); HSCT (consolidation high-risk); topotecan/cyclophosphamide",
    },
]

DEFINITIONS = [
    {"term": "Bilateral retinoblastoma", "definition": "Retinoblastoma in BOTH eyes; ~45% = germline RB1 mutation; ALL bilateral Rb patients should be assumed germline until molecular testing complete; triggers trilateral Rb screening (pinealoblastoma)."},
    {"term": "Trilateral retinoblastoma", "definition": "Bilateral Rb + pineoblastoma (pineal gland or suprasellar region); PATHOGNOMONIC for germline RB1; ~5% germline Rb; annual MRI brain until age 5 mandatory for all germline Rb carriers."},
    {"term": "CDK4/6 inhibitor ineffectiveness in Rb-null tumors", "definition": "pRb is the target of CDK4/6 inhibition (CDK4/6 phosphorylates pRb → E2F release); Rb-null tumors have no pRb to phosphorylate → CDK4/6i (palbociclib/ribociclib) are INACTIVE in RB1-null second primary tumors."},
    {"term": "DICER1 two-hit mechanism", "definition": "Unique: germline 1st hit = loss-of-function (frameshift/nonsense); somatic 2nd hit = HOTSPOT missense in metal-binding domain (RIIIb, exons 24-25: E1705/E1707/D1709/E1813); germline loss-of-function ALONE is insufficient — tumor requires BOTH hits."},
    {"term": "Pleuropulmonary Blastoma (PPB)", "definition": "Rare pediatric lung tumor; Type I (cystic, <2yr), II (mixed), III (solid); ~70% PPB = DICER1 germline mutation; PPB in any child <8 years = mandatory germline DICER1 testing PATHOGNOMONIC association."},
    {"term": "SMARCB1/BAF47 IHC loss", "definition": "Complete loss of nuclear SMARCB1/BAF47 immunohistochemistry staining; PATHOGNOMONIC for rhabdoid tumors (AT/RT, RTK); must be performed on all rhabdoid histology; retained SMARCB1 but absent SMARCA4 = RTPS2."},
    {"term": "SCCOHT (Small Cell Carcinoma Ovary Hypercalcemic Type)", "definition": "Most aggressive ovarian malignancy in women under 40; ~100% SMARCA4-deficient on IHC; ~43% = germline SMARCA4 mutation; PTHrP-mediated hypercalcemia in ~60% PATHOGNOMONIC; 5-yr OS ~10-15% advanced."},
    {"term": "Gorlin syndrome radiation CI", "definition": "Radiation is ABSOLUTELY CONTRAINDICATED in PTCH1 Gorlin syndrome — historical cases of cranial radiation for medulloblastoma led to hundreds of BCCs in the irradiated field; mechanism: DNA damage → somatic 2nd hit PTCH1 in SHH-primed skin."},
    {"term": "Odontogenic Keratocysts (OKC)", "definition": "Jaw cysts; PATHOGNOMONIC for Gorlin syndrome; first appear age 7-12 years; recurrent; visible on panoramic dental X-ray; require surgical management; surveillance panoramic X-ray every 12-18 months from age 7 in Gorlin/PTCH1 carriers."},
    {"term": "SUFU vs PTCH1 medulloblastoma treatment", "definition": "PTCH1-Gorlin MB → vismodegib/sonidegib (SMO inhibitors) ACTIVE; SUFU-germline MB → vismodegib NOT effective (SUFU acts downstream of SMO — drug target irrelevant); SUFU carriers have ~25% MB risk (higher than PTCH1 ~5%)."},
    {"term": "Simpson-Golabi-Behmel syndrome (SGB)", "definition": "GPC3 XLR LOF; overgrowth syndrome with macrosomia, coarse facies, macroglossia, polydactyly, organomegaly; ~10% embryonal tumor risk (Wilms, hepatoblastoma, NB); AFP + abdominal USS every 3 months first 8 years MANDATORY."},
    {"term": "Familial neuroblastoma ALK", "definition": "ALK germline GOF (F1174L, R1275Q); ~50% familial NB; ~75% penetrance; bilateral adrenal NB PATHOGNOMONIC hereditary; lorlatinib (3rd gen) preferred for F1174L (crizotinib partially resistant); test ALK + PHOX2B in all familial/bilateral NB."},
    {"term": "Desmoplastic/nodular medulloblastoma", "definition": "Histologic subtype of medulloblastoma; prominent desmoplastic (fibrotic) stroma with nodular pale islands; characteristic of SHH-pathway medulloblastoma (PTCH1, SUFU germline); most common in infants <3yr and young children in germline setting."},
]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def generate_overview() -> dict:
    cohort_stats = []
    for i, g in enumerate(ATLAS_GENES):
        rng = _rng(SEED_BASE + i)
        n = 40
        mean_age = round(rng.gauss(8, 5), 1)
        pct_female = round(rng.uniform(40, 60) if g["gene"] != "GPC3" else rng.uniform(10, 20), 1)
        cohort_stats.append({
            "gene": g["gene"],
            "n": n,
            "mean_age": max(0.5, mean_age),
            "pct_female": pct_female,
            "syndrome": g["syndrome"].split("/")[0].strip(),
            "inheritance": g["inheritance"].split(";")[0].strip(),
        })

    return {
        "atlas": "Hereditary-Retinoblastoma-Pediatric-Solid-Tumor-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Reference — Rb / DICER1 / Rhabdoid / Gorlin / SGB / Familial-NB",
        "total_patients": 320,
        "gene_count": 8,
        "seeds": "3702-3709",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "cohort_stats": cohort_stats,
        "clinical_categories": {
            "retinoblastoma": ["RB1"],
            "dicer1_syndrome": ["DICER1"],
            "rhabdoid_tumors": ["SMARCB1", "SMARCA4"],
            "hedgehog_pathway": ["PTCH1", "SUFU"],
            "overgrowth_embryonal": ["GPC3"],
            "familial_neuroblastoma": ["ALK"],
        },
        "key_clinical_rules": [
            "RB1: bilateral Rb = assume germline; EUA every 3-4 weeks infant; trilateral Rb screen (MRI brain annual)",
            "DICER1: PPB <8yr = test germline; 2nd hit somatic hotspot metal-binding domain; annual chest CT 0-8yr",
            "SMARCB1: radiation CI <36 months (catastrophic neurotoxicity); BAF47 IHC loss PATHOGNOMONIC",
            "SMARCA4: SCCOHT young women + hypercalcemia = SMARCA4 IHC; ~43% germline",
            "PTCH1: radiation ABSOLUTELY CI (hundreds BCCs); vismodegib/sonidegib (SMO inhibitors) active",
            "SUFU: vismodegib NOT effective (downstream of SMO); ~25% MB risk (higher than PTCH1)",
            "GPC3: AFP + USS every 3 months first 8yr; cardiac echo at diagnosis",
            "ALK: lorlatinib preferred F1174L (crizotinib partially resistant); bilateral adrenal NB = germline",
        ],
    }


def generate_breakdown() -> dict:
    gene_data = {}
    for i, g in enumerate(ATLAS_GENES):
        rng = _rng(SEED_BASE + i)
        n = 40
        mean_age = round(max(0.5, rng.gauss(8, 5)), 1)
        pct_f = round(rng.uniform(40, 60) if g["gene"] != "GPC3" else rng.uniform(10, 20), 1)
        bilateral_n = rng.randint(15, 35) if g["gene"] == "RB1" else 0
        met_n = rng.randint(5, 25)
        second_primary_n = rng.randint(3, 15) if g["gene"] == "RB1" else rng.randint(1, 8)
        gene_data[g["gene"]] = {
            "n": n,
            "mean_age": mean_age,
            "pct_female": pct_f,
            "bilateral_n": bilateral_n,
            "metastatic_n": met_n,
            "second_primary_n": second_primary_n,
            "syndrome": g["syndrome"].split("/")[0].strip(),
            "inheritance": g["inheritance"].split(";")[0].strip(),
        }

    return {
        "atlas": "Hereditary-Retinoblastoma-Pediatric-Solid-Tumor-Predisposition-Atlas",
        "total_patients": 320,
        "seeds": "3702-3709",
        "gene_data": gene_data,
        "atlas_genes": ATLAS_GENES,
        "summary": {
            "retinoblastoma_patients": 40,
            "dicer1_patients": 40,
            "rhabdoid_patients": 80,
            "gorlin_hedgehog_patients": 80,
            "overgrowth_embryonal_patients": 40,
            "familial_nb_patients": 40,
            "absolute_radiation_ci_genes": ["PTCH1"],
            "radiation_caution_genes": ["SMARCB1", "SUFU", "RB1"],
            "targeted_therapy_available": ["RB1-no-cdk46i", "PTCH1-vismodegib", "ALK-lorlatinib", "SMARCB1-tazemetostat"],
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Retinoblastoma-Pediatric-Solid-Tumor-Predisposition-Atlas",
        "definitions": DEFINITIONS,
        "critical_clinical_rules": {
            "RB1-BILATERAL-ALWAYS-GERMLINE": "Bilateral Rb = assume germline until proven otherwise — EUA every 3-4 weeks; annual MRI brain ages 1-5 for pinealoblastoma",
            "RB1-CDK46I-INACTIVE": "CDK4/6i (palbociclib/ribociclib) INACTIVE in RB1-null tumors — pRb is the drug target",
            "DICER1-PPB-TEST": "PPB in any child <8 years = MANDATORY germline DICER1 testing; annual chest CT ages 0-8 for DICER1 carriers",
            "SMARCB1-RADIATION-CI-36M": "Cranial radiation CONTRAINDICATED in infants <36 months with AT/RT — catastrophic neurological outcomes",
            "SMARCA4-SCCOHT-IHC": "SMARCA4/BRG1 IHC loss in young woman with ovarian tumor + hypercalcemia = SCCOHT PATHOGNOMONIC; test germline in ALL SCCOHT",
            "PTCH1-RADIATION-ABSOLUTELY-CI": "Radiation ABSOLUTELY CONTRAINDICATED in Gorlin/NBCCS (PTCH1) — massive BCC induction in irradiated field; multiple historical catastrophes",
            "SUFU-NO-VISMODEGIB": "Vismodegib (SMO inhibitor) NOT effective in SUFU-germline MB — SUFU downstream of SMO target; use conventional MB therapy",
            "GPC3-AFP-USS-SURVEILLANCE": "AFP + abdominal USS every 3 months for first 8 years in ALL GPC3 germline males — Wilms and hepatoblastoma early detection",
            "ALK-F1174L-LORLATINIB": "F1174L ALK is partially crizotinib-resistant — use lorlatinib (3rd gen) for F1174L-mutant neuroblastoma",
            "ALK-BILATERAL-NB-GERMLINE": "Bilateral adrenal neuroblastoma = assume germline predisposition; test ALK AND PHOX2B; ~75% penetrance",
        },
    }
