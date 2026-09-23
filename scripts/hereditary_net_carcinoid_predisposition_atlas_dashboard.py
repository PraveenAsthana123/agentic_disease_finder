#!/usr/bin/env python3
"""Hereditary-Neuroendocrine-Tumor-NET-Carcinoid-Predisposition-Atlas -- Complete 8-Gene Reference
MEN1    (Menin nuclear scaffold; 610aa; 11q13.1; AD LOF;
         MEN1 syndrome — pNETs 40-70%; gastrinoma/insulinoma most common;
         Zollinger-Ellison PATHOGNOMONIC for MEN1-gastrinoma;
         concurrent HPT 95%+ and pituitary 30-50%;
         everolimus NOT first-line for MEN1; somatostatin analogues Level A;
         seed SEED_BASE+0) .
VHL     (HIF-1alpha/HIF-2alpha substrate adaptor; 213aa; 3p25.3; AD LOF;
         VHL disease — pNETs 15-17% EXCLUSIVELY clear cell cytoplasm;
         hemangioblastoma cerebellum/retina PATHOGNOMONIC;
         ccRCC 70%; belzutifan HIF-2alpha inhibitor FDA 2021;
         pNETs VHL = NON-FUNCTIONAL low grade; pheo/PGL 10-20%;
         seed SEED_BASE+1) .
RET     (Receptor tyrosine kinase; 1114aa; 10q11.21; AD GOF;
         MEN2A — MTC 100% penetrance; pheo 50% bilateral;
         codon-based thyroidectomy: M918T 6 months, C634F/R 5yr;
         selpercatinib FDA2020; vandetanib/cabozantinib MTC;
         seed SEED_BASE+2) .
NF1     (Neurofibromin RAS-GAP; 2839aa; 17q11.2; AD LOF;
         NF1 syndrome — DUODENAL SOMATOSTATINOMAS PATHOGNOMONIC periampullary;
         GIST 7%; MPNST 8-13% PATHOGNOMONIC; café-au-lait 6+ PATHOGNOMONIC;
         selumetinib FDA 2020; NETs in NF1 rarely pNETs (duodenal primarily);
         psammoma bodies in somatostatinoma PATHOGNOMONIC;
         seed SEED_BASE+3) .
TSC2    (Tuberin mTOR-regulatory GAP; 1807aa; 16p13.3; AD LOF;
         TSC — pulmonary carcinoids 1-3%; pNETs rare;
         everolimus FDA 2016 TSC-pNETs/angiomyolipoma;
         renal AML 80% bilateral; SEGA PATHOGNOMONIC;
         cortical tubers PATHOGNOMONIC; cardiac rhabdomyoma fetal PATHOGNOMONIC;
         mTOR pathway DIRECTLY therapeutic target;
         seed SEED_BASE+4) .
CDKN1B  (p27/KIP1 cell-cycle inhibitor G1/S; 196aa; 12p13.1; AD LOF;
         MEN4 — pituitary + parathyroid + pNETs rare;
         phenotypically overlaps MEN1 but genetically distinct;
         MEN4 = diagnosis of exclusion after MEN1 negative;
         prevalence very low ~3% of MEN1-negative MEN-like families;
         seed SEED_BASE+5) .
PRKAR1A (Regulatory subunit type 1A of PKA; 381aa; 17q24.2; AD LOF;
         Carney complex — PPNAD PATHOGNOMONIC subclinical Cushing;
         cardiac myxoma LIFE-THREATENING annual echo MANDATORY;
         somatotropinoma acromegaly; lentigines PATHOGNOMONIC;
         testicular LCCSCT PATHOGNOMONIC males;
         psammomatous melanotic schwannoma PATHOGNOMONIC;
         seed SEED_BASE+6) .
SDHB    (Succinate dehydrogenase subunit B; 280aa; 1p36.13; AR LOF for full SDH loss;
         SDHB-IHC negative = global SDHx marker PATHOGNOMONIC;
         pNETs in SDHx families 10-15%; MALIGNANT PGL 35-40% HIGHEST SDHx gene;
         extra-adrenal retroperitoneal PGL; SSTR-PET MANDATORY;
         succinate accumulation pseudohypoxia;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3406-3413)
"""
import random

SEED_BASE = 3406

ATLAS_GENES = [
    {
        "gene": "MEN1",
        "protein": (
            "MEN1 -- 11q13.1 Autosomal-Dominant-LOF -- 610aa -- "
            "Menin-Nuclear-Scaffold-67kDa-Histone-H3K4-Methylation-"
            "MEN1-Syndrome-pNETs-40-70pct-"
            "Gastrinoma-Zollinger-Ellison-PATHOGNOMONIC-"
            "HPT-95pct-Concurrent-Pituitary-30-50pct-"
            "Somatostatin-Analogues-Level-A-Everolimus-NOT-First-Line-OMIM-131100"
        ),
        "locus": "11q13.1",
        "protein_size": (
            "610 aa / 67 kDa / 11q13.1 MEN1 neuroendocrine molecular context: "
            "STRUCTURE: "
            "  610 aa / 67 kDa; Menin nuclear scaffold — no enzymatic domain; "
            "  Interacts with MLL1/2 histone methyltransferases: H3K4me3 at active genes; "
            "  JunD binding domain: suppresses AP-1 transcription; "
            "  Two nuclear localisation signals (NLS1: aa 479-497; NLS2: aa 588-608); "
            "CANCER RISKS (NET FOCUS): "
            "  pNETs: 40-70% lifetime — gastrinoma (Zollinger-Ellison) + insulinoma most common; "
            "  GASTRINOMA: Zollinger-Ellison syndrome PATHOGNOMONIC for MEN1-gastrinoma — refractory peptic ulcers; "
            "  HYPERPARATHYROIDISM: 95%+ concurrent — screen Ca2+/PTH at every visit; "
            "  PITUITARY: prolactinoma 30-50% — MOST COMMON MEN1 pituitary tumour; "
            "  Foregut NETs: bronchial + thymic carcinoids; "
            "KEY MANAGEMENT (NET): "
            "  Somatostatin analogues (octreotide/lanreotide): Level A evidence — antisecretory + antiproliferative; "
            "  Everolimus: NOT first-line for MEN1-pNETs (mTOR not primary MEN1 driver) — reserve for sporadic pNETs; "
            "  Loss of heterozygosity (LOH): second hit drives MEN1 tumourigenesis — somatic LOH 11q13; "
            "  Surgery: insulinoma localised — enucleation; gastrinoma multifocal — PPI + SSA; "
            "  Annual biochemical screen: Ca2+, PTH, prolactin, fasting gastrin, IGF-1, chromogranin A"
        ),
        "syndrome": "Multiple Endocrine Neoplasia type 1 (MEN1) — HPT + pNET + pituitary triad",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); LOH drives tumour in heterozygotes",
        "net_risk": "pNETs 40-70%; gastrinoma (ZES) + insulinoma most common; bronchial + thymic carcinoids",
        "pathognomonic": "Zollinger-Ellison syndrome (gastrinoma) + concurrent hyperparathyroidism in MEN1; MEN1 triad HPT+pNET+pituitary",
        "key_avoid": "Do NOT use everolimus first-line for MEN1-pNETs — mTOR is NOT the primary MEN1 driver; reserve everolimus for sporadic/refractory pNETs",
        "key_rule": "MEN1: always screen Ca2+/PTH (HPT 95%) + prolactin + fasting gastrin + chromogranin A annually. Somatostatin analogues Level A. LOH 11q13 confirms somatic second hit.",
        "surveillance": "Annual Ca2+/PTH + prolactin + fasting gastrin + insulin + IGF-1 + chromogranin A; pancreatic MRI every 1-2yr; pituitary MRI every 3-5yr",
        "targeted_rx": "Octreotide/lanreotide Level A SSA; everolimus reserved refractory/sporadic; streptozocin-based chemo pNET; everolimus RADIANT-3 sporadic pNET only",
    },
    {
        "gene": "VHL",
        "protein": (
            "VHL -- 3p25.3 Autosomal-Dominant-LOF -- 213aa -- "
            "HIF-1alpha-HIF-2alpha-Substrate-Adaptor-24kDa-E3-Ubiquitin-Ligase-"
            "VHL-Disease-pNETs-15-17pct-EXCLUSIVELY-Clear-Cell-Cytoplasm-"
            "Hemangioblastoma-Cerebellum-Retina-PATHOGNOMONIC-"
            "ccRCC-70pct-Belzutifan-HIF-2alpha-Inhibitor-FDA-2021-"
            "pNETs-VHL-NON-FUNCTIONAL-Low-Grade-Pheo-PGL-10-20pct-OMIM-608537"
        ),
        "locus": "3p25.3",
        "protein_size": (
            "213 aa / 24 kDa / 3p25.3 VHL neuroendocrine molecular context: "
            "STRUCTURE: "
            "  213 aa / 24 kDa; α domain (aa 63-154): HIF binding; β domain (aa 1-62 + 155-213): ElonginC/B; "
            "  E3 ubiquitin ligase complex: VHL-ElonginB-ElonginC-Cullin2-RBX1; "
            "  Targets HIF-1alpha/HIF-2alpha for ubiquitin-proteasomal degradation in normoxia; "
            "  VHL loss → HIF stabilisation → VEGF, PDGF, EPO, GLUT1 upregulation; "
            "CANCER RISKS (NET FOCUS): "
            "  pNETs: 15-17% VHL — EXCLUSIVELY clear cell cytoplasm (VHL pNET histology DISTINCTIVE); "
            "  VHL pNETs: NON-FUNCTIONAL low grade — rarely cause hormonal syndromes; "
            "  ccRCC: 70% lifetime — most common VHL malignancy; clear cell histology; "
            "  Hemangioblastoma: cerebellum + retina + spinal cord — PATHOGNOMONIC VHL; "
            "  Pheo/PGL: 10-20% — SDHx-like pseudohypoxia HIF pathway; "
            "  Endolymphatic sac tumour (ELST): PATHOGNOMONIC VHL — hearing loss; "
            "KEY MANAGEMENT (NET): "
            "  Belzutifan (HIF-2alpha inhibitor): FDA 2021 VHL-ccRCC/pNET/hemangioblastoma — pNET indication included; "
            "  VHL pNETs: watch-and-wait if <3cm non-functional; surgery if >3cm or growth; "
            "  Somatostatin analogues: symptomatic control (less indication than MEN1 — non-functional); "
            "  Annual ophthalmology: retinal hemangioblastoma surveillance; "
            "  PATHOGNOMONIC RULE: clear cell cytoplasm in pNET = VHL until proven otherwise"
        ),
        "syndrome": "VHL disease (Von Hippel-Lindau) — hemangioblastoma + ccRCC + pNET + pheo triad",
        "inheritance": "AD LOF (autosomal dominant); somatic second hit LOH 3p25.3",
        "net_risk": "pNETs 15-17% EXCLUSIVELY clear cell; NON-FUNCTIONAL low grade; pheo/PGL 10-20%",
        "pathognomonic": "Clear cell cytoplasm in pNET PATHOGNOMONIC for VHL; retinal + cerebellar hemangioblastoma PATHOGNOMONIC VHL; ELST hearing loss PATHOGNOMONIC",
        "key_avoid": "Do NOT assume VHL pNETs are functional/secretory — they are almost exclusively NON-FUNCTIONAL; Do NOT miss retinal hemangioblastoma (annual ophthalmology MANDATORY)",
        "key_rule": "VHL pNET = clear cell NON-FUNCTIONAL. Belzutifan FDA 2021 covers pNET + ccRCC + hemangioblastoma. Annual retinal exam MANDATORY. <3cm pNET = watch; >3cm = surgery.",
        "surveillance": "Annual ophthalmology (retinal hemangioblastoma); annual MRI brain/spine + abdomen; annual urine catecholamines (pheo); audiology (ELST)",
        "targeted_rx": "Belzutifan HIF-2alpha inhibitor FDA 2021 (VHL-pNET + ccRCC + hemangioblastoma); sunitinib/everolimus refractory pNET; surgery <3cm watch >3cm resect",
    },
    {
        "gene": "RET",
        "protein": (
            "RET -- 10q11.21 Autosomal-Dominant-GOF -- 1114aa -- "
            "Receptor-Tyrosine-Kinase-124kDa-GDNF-Family-Ligand-Receptor-"
            "MEN2A-MTC-100pct-Penetrance-Pheo-50pct-Bilateral-"
            "Codon-Based-Thyroidectomy-M918T-6mo-C634FR-5yr-Other-MEN2A-5-10yr-"
            "Selpercatinib-FDA2020-RET-Fusion-Mutation-"
            "Vandetanib-Cabozantinib-MTC-OMIM-164761"
        ),
        "locus": "10q11.21",
        "protein_size": (
            "1114 aa / 124 kDa / 10q11.21 RET neuroendocrine molecular context: "
            "STRUCTURE: "
            "  1114 aa / 124 kDa; Cadherin-like domain (aa 1-635): GDNF binding; "
            "  TM domain (aa 636-658); intracellular kinase domain (aa 724-1013); "
            "  MEN2A mutations: extracellular cysteine domain (codons 609-634) → disulfide bond disruption → constitutive dimerisation; "
            "  MEN2B M918T: kinase domain shift → substrate specificity change; "
            "CANCER RISKS (NET FOCUS): "
            "  MTC: 100% penetrance in MEN2 — medullary thyroid carcinoma is C-cell NET; "
            "  Pheo: 50% in MEN2A — frequently bilateral; always screen before thyroid surgery; "
            "  Parathyroid: 20-30% in MEN2A (unlike MEN2B); "
            "  MEN2B (M918T): MTC earliest onset + marfanoid habitus + mucosal neuromas PATHOGNOMONIC; "
            "CODON-BASED THYROIDECTOMY TIMING (ATA risk stratification): "
            "  M918T (MEN2B): thyroidectomy by 6 months of life — HIGHEST RISK; "
            "  C634F/R + C609/611/618/620 (MEN2A highest): thyroidectomy by age 5yr; "
            "  Other MEN2A codon mutations: thyroidectomy by 5-10yr or when calcitonin elevated; "
            "KEY MANAGEMENT: "
            "  Selpercatinib (RET inhibitor): FDA 2020 RET-fusion NSCLC + RET-mutant MTC (RET V804L/M, M918T); "
            "  Vandetanib + cabozantinib: approved metastatic MTC (multi-kinase including RET); "
            "  Biochemical: calcitonin + CEA annual — doubling time <6 months = urgent; "
            "  Screen pheo BEFORE thyroidectomy: bilateral adrenalectomy risk in MEN2"
        ),
        "syndrome": "Multiple Endocrine Neoplasia type 2A (MEN2A) / 2B (M918T) / FMTC",
        "inheritance": "AD GOF (autosomal dominant gain-of-function); codon-specific phenotype",
        "net_risk": "MTC 100% penetrance; pheo 50% bilateral MEN2A; C-cell NET (parafollicular)",
        "pathognomonic": "MEN2B M918T: marfanoid + mucosal neuromas + corneal nerve thickening PATHOGNOMONIC; bilateral pheo in MEN2; MTC family + pheo = MEN2",
        "key_avoid": "NEVER do thyroidectomy in MEN2 without FIRST screening for pheo — untreated pheo during thyroid surgery = hypertensive crisis FATAL; Do NOT delay M918T thyroidectomy beyond 6 months",
        "key_rule": "RET codon-based thyroidectomy timing is MANDATORY risk-stratification. Screen pheo FIRST. Selpercatinib FDA2020 RET-mutant MTC. Calcitonin doubling time <6mo = urgent.",
        "surveillance": "Annual calcitonin + CEA; annual urine metanephrines (pheo); annual Ca2+/PTH (MEN2A); neck USS annually; codon-stratified thyroidectomy timing",
        "targeted_rx": "Selpercatinib FDA2020 RET-mutant MTC; vandetanib FDA2011 metastatic MTC; cabozantinib FDA2012 metastatic MTC; pralsetinib RET-mutant MTC",
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-RAS-GAP-319kDa-RAS-GTPase-Activating-Protein-"
            "NF1-Syndrome-Duodenal-Somatostatinomas-PATHOGNOMONIC-Periampullary-"
            "GIST-7pct-MPNST-8-13pct-PATHOGNOMONIC-"
            "Cafe-Au-Lait-6plus-PATHOGNOMONIC-"
            "Selumetinib-FDA-2020-Psammoma-Bodies-Somatostatinoma-PATHOGNOMONIC-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 neuroendocrine molecular context: "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; GRD domain (aa 1175-1530): RAS-GAP catalytic; "
            "  SecPH domain: lipid binding; CSRD + PH-like domains; "
            "  NF1 loss → RAS-GTP accumulation → MAPK/ERK + PI3K/mTOR constitutive activation; "
            "CANCER RISKS (NET FOCUS): "
            "  DUODENAL SOMATOSTATINOMAS: PATHOGNOMONIC for NF1 — periampullary location; "
            "  NF1 somatostatinoma: psammoma bodies (concentric calcifications) PATHOGNOMONIC on histology; "
            "  NET in NF1 = primarily DUODENAL (not pNETs) — critical distinction from MEN1; "
            "  GIST: 7% gastrointestinal stromal tumours — NF1-GIST small intestine IMATINIB RESISTANT (not KIT/PDGFRA); "
            "  MPNST: 8-13% malignant peripheral nerve sheath tumour PATHOGNOMONIC NF1 context; "
            "  Pheo: 1-5% NF1 (lower penetrance); "
            "KEY MANAGEMENT (NET): "
            "  NF1-GIST: imatinib INEFFECTIVE (no KIT/PDGFRA mutation) — surgery primary; "
            "  Selumetinib (MEK inhibitor): FDA 2020 paediatric NF1 plexiform neurofibroma — MEK/ERK pathway; "
            "  Duodenal somatostatinoma surveillance: duodenoscopy + EUS from 25yr NF1; "
            "  Café-au-lait macules 6+: PATHOGNOMONIC NF1 diagnostic criterion; Lisch nodules (iris hamartomas) PATHOGNOMONIC; "
            "KEY DISTINCTION: NF1-NETs = DUODENAL periampullary, not pancreatic (unlike MEN1/VHL/TSC)"
        ),
        "syndrome": "Neurofibromatosis type 1 (NF1) — café-au-lait + neurofibroma + Lisch nodule triad",
        "inheritance": "AD LOF (autosomal dominant loss-of-function); RAS-GAP function lost",
        "net_risk": "Duodenal somatostatinomas PATHOGNOMONIC periampullary; GIST 7%; pheo 1-5%; NOT primarily pNETs",
        "pathognomonic": "Duodenal periampullary somatostatinoma + psammoma bodies in NF1; café-au-lait 6+ PATHOGNOMONIC; MPNST PATHOGNOMONIC NF1 context; Lisch nodules iris",
        "key_avoid": "Do NOT use imatinib for NF1-GIST — NF1-GIST lacks KIT/PDGFRA mutations (imatinib target); surgery is primary. Do NOT assume NF1-NETs are pancreatic — they are duodenal.",
        "key_rule": "NF1 NETs = periampullary DUODENAL somatostatinoma with psammoma bodies. GIST imatinib-resistant. Selumetinib MEK inhibitor for plexiform neurofibroma. Annual full-body exam.",
        "surveillance": "Annual clinical exam (skin + neuro); annual ophthalmology (Lisch nodules); duodenoscopy + EUS from 25yr (somatostatinoma); annual urine catecholamines (pheo); MRI brain/spine if neurological symptoms",
        "targeted_rx": "Selumetinib FDA2020 MEK inhibitor NF1 plexiform; binimetinib MEK adult trials; surgery primary NF1-GIST; somatostatin analogues NF1 somatostatinoma symptomatic",
    },
    {
        "gene": "TSC2",
        "protein": (
            "TSC2 -- 16p13.3 Autosomal-Dominant-LOF -- 1807aa -- "
            "Tuberin-mTOR-Regulatory-GAP-200kDa-Rheb-GTP-Inhibitor-"
            "TSC-Pulmonary-Carcinoids-1-3pct-"
            "Everolimus-FDA-2016-TSC-pNETs-Angiomyolipoma-"
            "Renal-AML-80pct-Bilateral-SEGA-PATHOGNOMONIC-"
            "Cortical-Tubers-PATHOGNOMONIC-Cardiac-Rhabdomyoma-Fetal-PATHOGNOMONIC-"
            "mTOR-DIRECTLY-Therapeutic-Target-OMIM-191092"
        ),
        "locus": "16p13.3",
        "protein_size": (
            "1807 aa / 200 kDa / 16p13.3 TSC2 neuroendocrine molecular context: "
            "STRUCTURE: "
            "  1807 aa / 200 kDa; Rheb-GAP domain (aa 1525-1742): inhibits Rheb-GTP → mTORC1 off; "
            "  Hamartin (TSC1) binding domain: forms TSC1-TSC2 heterodimer; "
            "  TSC2 loss → Rheb-GTP accumulates → mTORC1 constitutive activation → S6K/4EBP1 hyperphosphorylation; "
            "CANCER RISKS (NET FOCUS): "
            "  Pulmonary carcinoids: 1-3% (rare but recognised TSC feature); "
            "  pNETs: rare in TSC — case series only; non-functional; "
            "  Renal AML (angiomyolipoma): 80% bilateral — everolimus shrinks AML + prevents haemorrhage; "
            "  SEGA (subependymal giant cell astrocytoma): PATHOGNOMONIC TSC — everolimus first-line; "
            "  Cortical tubers: PATHOGNOMONIC TSC — epilepsy; "
            "  LAM (lymphangioleiomyomatosis): lung in adult females — everolimus slows FEV1 decline; "
            "  Cardiac rhabdomyoma in fetal life: PATHOGNOMONIC TSC — usually regresses; "
            "KEY MANAGEMENT (NET/mTOR): "
            "  Everolimus: FDA 2016 TSC-pNETs + angiomyolipoma + SEGA + LAM — DIRECT mTOR pathway target; "
            "  mTOR = DIRECTLY THERAPEUTIC for TSC (unlike MEN1 where mTOR not primary driver); "
            "  Pulmonary carcinoid in TSC: consider everolimus; standard SSA if symptomatic; "
            "  Sirolimus: alternative mTORC1 inhibitor (less common than everolimus); "
            "KEY DISTINCTION: TSC mTOR is DIRECTLY the therapeutic target (constitutive mTORC1) vs MEN1 (LOH menin — mTOR secondary)"
        ),
        "syndrome": "Tuberous Sclerosis Complex (TSC1/TSC2) — cortical tubers + AML + SEGA + cardiac rhabdomyoma",
        "inheritance": "AD LOF (autosomal dominant); de novo mutations 60-70%; TSC1/TSC2 both cause TSC",
        "net_risk": "Pulmonary carcinoids 1-3% (rare); pNETs rare; renal AML 80%; SEGA common; LAM females",
        "pathognomonic": "SEGA PATHOGNOMONIC TSC; cortical tubers PATHOGNOMONIC; cardiac rhabdomyoma fetal PATHOGNOMONIC; subependymal nodules (candle drippings); shagreen patch + ash leaf macules",
        "key_avoid": "Do NOT conflate TSC everolimus with MEN1 — everolimus works in TSC because mTOR IS the primary driver; in MEN1 menin loss is primary and mTOR is secondary. Do NOT ignore cardiac rhabdomyoma in neonates.",
        "key_rule": "TSC: mTOR is DIRECTLY therapeutic — everolimus FDA approved for SEGA + AML + LAM + pNETs. Everolimus first-line SEGA. Annual renal USS + brain MRI. Cardiac echo fetal/neonatal.",
        "surveillance": "Annual renal USS (AML); annual brain MRI (SEGA/tubers); annual pulmonary function + chest CT females (LAM); annual ophthalmology (retinal astrocytoma); annual dermatology; echocardiogram neonatal",
        "targeted_rx": "Everolimus FDA2016 TSC (SEGA + AML + LAM + pNETs); sirolimus alternative; surgical resection focal TSC lesions; vigabatrin/ACTH infantile spasms TSC",
    },
    {
        "gene": "CDKN1B",
        "protein": (
            "CDKN1B -- 12p13.1 Autosomal-Dominant-LOF -- 196aa -- "
            "p27-KIP1-Cell-Cycle-Inhibitor-22kDa-G1-S-Checkpoint-CDK2-CDK4-"
            "MEN4-MEN-Type-4-Pituitary-Parathyroid-pNETs-Rare-"
            "MEN4-Phenotypically-Overlaps-MEN1-Genetically-Distinct-"
            "MEN4-Diagnosis-Exclusion-After-MEN1-Negative-"
            "Prevalence-Very-Low-3pct-MEN1-Negative-Families-OMIM-600778"
        ),
        "locus": "12p13.1",
        "protein_size": (
            "196 aa / 22 kDa / 12p13.1 CDKN1B neuroendocrine molecular context: "
            "STRUCTURE: "
            "  196 aa / 22 kDa; CDK-binding domain (aa 25-93): inhibits CDK2+cyclinE and CDK4+cyclinD; "
            "  NLS (aa 151-183): nuclear localisation; Myristoylation (aa 183-196) anchors cytoplasmic form; "
            "  p27 inhibits G1-to-S phase progression by blocking CDK activity; "
            "  CDKN1B loss → unrestrained cell cycle entry; "
            "CANCER RISKS (NET FOCUS): "
            "  MEN4: pituitary adenoma + parathyroid hyperplasia/adenoma + pNETs (rare); "
            "  MEN4 spectrum: phenotypically identical to MEN1 but MEN1 gene sequencing NEGATIVE; "
            "  Prevalence: very low — ~3% of MEN1-negative MEN-like families; "
            "  pNETs in MEN4: rare, lower penetrance than MEN1; "
            "  Other cancers: gastric + renal + cervical in CDKN1B mouse models (clinical significance uncertain); "
            "KEY MANAGEMENT: "
            "  MEN4 = DIAGNOSIS OF EXCLUSION: MEN1 gene full sequencing + deletion/duplication NEGATIVE first; "
            "  Surveillance: SAME as MEN1 protocol (Ca2+/PTH + prolactin + gastrin + chromogranin A annually); "
            "  No specific targeted therapy for CDKN1B beyond standard SSA for pNETs; "
            "  CDK4/6 inhibitors (palbociclib): THEORETICALLY relevant (CDK4/6 are CDKN1B substrates) — no approved indication yet; "
            "KEY DISTINCTION: MEN4 is MEN1-phenotype with MEN1-negative germline → test CDKN1B next in MEN-like families after MEN1 exclusion"
        ),
        "syndrome": "Multiple Endocrine Neoplasia type 4 (MEN4) — MEN1-phenotype with MEN1-negative germline",
        "inheritance": "AD LOF (autosomal dominant); rare — ~3% MEN1-negative MEN-like families",
        "net_risk": "pNETs rare (lower penetrance than MEN1); pituitary adenoma + parathyroid primary features; MEN4 spectrum",
        "pathognomonic": "MEN-phenotype (HPT + pNET + pituitary) + MEN1-germline-negative = CDKN1B/MEN4 diagnosis; p27 loss IHC NET tumours",
        "key_avoid": "Do NOT diagnose MEN4 without first excluding MEN1 by full sequencing including deletion/duplication analysis (MLPA); Do NOT assume low MEN4 prevalence means low risk in MEN1-negative MEN families",
        "key_rule": "MEN4 = MEN1-negative by full gene test + CDKN1B pathogenic variant. Surveillance mirrors MEN1. No specific targeted therapy beyond SSA. CDK4/6 inhibitors theoretical but not approved.",
        "surveillance": "Same as MEN1: annual Ca2+/PTH + prolactin + fasting gastrin + chromogranin A; pancreatic MRI every 1-2yr; pituitary MRI every 3-5yr; first-degree relatives CDKN1B testing",
        "targeted_rx": "Somatostatin analogues (octreotide/lanreotide) for functional pNETs; CDK4/6 inhibitors theoretical; standard pNET regimens (streptozocin, everolimus) if advanced; parathyroidectomy HPT",
    },
    {
        "gene": "PRKAR1A",
        "protein": (
            "PRKAR1A -- 17q24.2 Autosomal-Dominant-LOF -- 381aa -- "
            "PKA-Regulatory-Subunit-Type-1A-43kDa-cAMP-Binding-"
            "Carney-Complex-CNC-PPNAD-PATHOGNOMONIC-Subclinical-Cushing-"
            "Cardiac-Myxoma-LIFE-THREATENING-Annual-Echo-MANDATORY-"
            "Somatotropinoma-Acromegaly-Lentigines-PATHOGNOMONIC-"
            "Testicular-LCCSCT-PATHOGNOMONIC-Males-"
            "Psammomatous-Melanotic-Schwannoma-PATHOGNOMONIC-OMIM-160980"
        ),
        "locus": "17q24.2",
        "protein_size": (
            "381 aa / 43 kDa / 17q24.2 PRKAR1A neuroendocrine molecular context: "
            "STRUCTURE: "
            "  381 aa / 43 kDa; dimerisation/docking domain (DD; aa 1-44); "
            "  Inhibitory sequence (IS; aa 92-99): blocks PKA catalytic subunit; "
            "  Two cAMP-binding domains (CNBA aa 139-258; CNBB aa 259-381); "
            "  PRKAR1A inhibits PKAc at rest; cAMP binding → PKAc release → phosphorylation cascade; "
            "  PRKAR1A LOF → constitutive PKA activity → CREB phosphorylation → ACTH-independent cortisol; "
            "CANCER RISKS (NET/ENDOCRINE FOCUS): "
            "  PPNAD (primary pigmented nodular adrenocortical disease): PATHOGNOMONIC Carney complex — subclinical Cushing, ACTH-independent, bilateral micronodular adrenals; "
            "  Cardiac myxoma: 50-60% Carney complex — atria predominantly; LIFE-THREATENING embolic risk; annual echo MANDATORY; "
            "  Somatotropinoma: 10-20% — GH excess → acromegaly (IGF-1 elevated); "
            "  Lentigines: centrofacial + lip + conjunctival PATHOGNOMONIC Carney complex; "
            "  Testicular LCCSCT (large-cell calcifying Sertoli cell tumour): PATHOGNOMONIC males with Carney complex; "
            "  Psammomatous melanotic schwannoma (PMS): PATHOGNOMONIC — peripheral nerve melanin + psammoma bodies; "
            "KEY MANAGEMENT: "
            "  Cardiac myxoma: surgical resection + annual echo for recurrence MANDATORY; "
            "  PPNAD: bilateral laparoscopic adrenalectomy for frank Cushing; "
            "  Acromegaly: somatostatin analogues first-line + pegvisomant if refractory; "
            "  Lentigines: no treatment — cosmetic; diagnostic marker only"
        ),
        "syndrome": "Carney Complex (CNC) — cardiac myxoma + PPNAD + lentigines + schwannoma triad",
        "inheritance": "AD LOF (autosomal dominant); PRKAR1A mutations 70% Carney complex",
        "net_risk": "PPNAD subclinical Cushing PATHOGNOMONIC; somatotropinoma 10-20%; thyroid nodules common; pNETs rare",
        "pathognomonic": "PPNAD + Carney complex; cardiac myxoma bilateral/multifocal; lentigines centrofacial PATHOGNOMONIC; testicular LCCSCT males PATHOGNOMONIC; psammomatous melanotic schwannoma PATHOGNOMONIC",
        "key_avoid": "NEVER omit annual echocardiogram in PRKAR1A/Carney complex — cardiac myxoma embolic risk is life-threatening; Do NOT confuse PPNAD Cushing (ACTH-independent, bilateral micronodular) with pituitary Cushing (ACTH-dependent)",
        "key_rule": "Carney complex: annual cardiac echo MANDATORY (myxoma embolic risk). PPNAD = ACTH-independent Cushing. Testicular USS annual males (LCCSCT). IGF-1 annual (acromegaly). Lentigines = diagnostic marker.",
        "surveillance": "Annual echocardiogram (cardiac myxoma MANDATORY); annual cortisol/ACTH (PPNAD); annual IGF-1/GH (acromegaly); annual testicular USS males (LCCSCT); annual thyroid USS; skin exam lentigines",
        "targeted_rx": "Cardiac myxoma: surgical resection; PPNAD bilateral adrenalectomy frank Cushing; somatostatin analogues + pegvisomant acromegaly; ketoconazole/metyrapone Cushing bridging",
    },
    {
        "gene": "SDHB",
        "protein": (
            "SDHB -- 1p36.13 AR-LOF-for-Full-SDH-Loss -- 280aa -- "
            "Succinate-Dehydrogenase-Subunit-B-32kDa-Iron-Sulfur-Cluster-"
            "SDHB-Loss-Global-SDHx-IHC-Marker-PATHOGNOMONIC-"
            "pNETs-SDHx-10-15pct-MALIGNANT-PGL-35-40pct-HIGHEST-SDHx-Gene-"
            "Extra-Adrenal-Retroperitoneal-PGL-SSTR-PET-MANDATORY-"
            "Succinate-Accumulation-Pseudohypoxia-"
            "SDHB-IHC-Negative-All-SDHx-Members-OMIM-185470"
        ),
        "locus": "1p36.13",
        "protein_size": (
            "280 aa / 32 kDa / 1p36.13 SDHB neuroendocrine molecular context: "
            "STRUCTURE: "
            "  280 aa / 32 kDa; N-terminal mitochondrial targeting sequence; "
            "  Three iron-sulfur clusters ([2Fe-2S], [4Fe-4S], [3Fe-4S]): electron transfer chain II; "
            "  SDHB bridges SDHA (flavoprotein) to SDHC/D (membrane anchor); "
            "  SDH complex: succinate → fumarate (complex II electron transfer); "
            "  SDHB LOF → succinate accumulates → HIF-1alpha/HIF-2alpha pseudohypoxia stabilisation; "
            "  Succinate inhibits prolyl hydroxylases → HIF-pathway activation (oncometabolite mechanism); "
            "CANCER RISKS (NET FOCUS): "
            "  Malignant PGL: 35-40% SDHB — HIGHEST malignancy rate of all SDHx genes; "
            "  Extra-adrenal retroperitoneal PGL: SDHB enriched location (vs adrenal pheo enriched in MEN2/VHL); "
            "  pNETs in SDHx families: 10-15% lifetime; "
            "  Gastric GISTs: SDHB-deficient GIST (Carney-Stratakis syndrome) — KIT/PDGFRA wildtype; "
            "  ccRCC: emerging risk in SDHB carriers; "
            "KEY MANAGEMENT: "
            "  SDHB-IHC: PATHOGNOMONIC tool — SDHB antibody stains ALL SDHx-deficient tumours (SDHB/C/D/A loss) by blocking whole complex assembly; "
            "  SSTR-PET (68Ga-DOTATATE): MANDATORY staging in all SDHx-related NETs — superior to CT/MRI for occult lesions; "
            "  Somatic SDHx: 30% sporadic PGL have somatic (not germline) SDHx mutations; "
            "  Annual surveillance: urine/plasma metanephrines + normetanephrines + 3-methoxytyramine (dopaminergic SDHB); "
            "KEY RULE: SDHB = malignant PGL 35-40% — most aggressive SDHx gene; sunitinib/temozolomide for malignant PGL"
        ),
        "syndrome": "Hereditary PGL/Pheo syndrome type 4 (SDHB) — malignant PGL enriched; Carney-Stratakis (SDHB-GIST+PGL)",
        "inheritance": "AR LOF for full SDH complex loss; monoallelic germline + somatic second hit = LOH 1p36",
        "net_risk": "pNETs 10-15%; malignant PGL 35-40% HIGHEST SDHx; extra-adrenal retroperitoneal PGL; GIST SDHx-deficient",
        "pathognomonic": "SDHB-IHC negative = SDHx deficiency PATHOGNOMONIC (covers all 4 SDH subunits); extra-adrenal retroperitoneal PGL SDHB; malignant PGL 35-40% HIGHEST SDHx",
        "key_avoid": "Do NOT rely on CT/MRI alone for SDHB staging — SSTR-PET MANDATORY (68Ga-DOTATATE superior); Do NOT assume sporadic PGL has no germline — 30% sporadic PGL have somatic SDHx; Do NOT conflate SDHB-GIST (KIT wildtype) with standard GIST (imatinib works)",
        "key_rule": "SDHB = 35-40% malignant PGL — most aggressive SDHx gene. SDHB-IHC negative = all SDHx marker. SSTR-PET MANDATORY. Annual 3-methoxytyramine (dopaminergic PGL marker). Somatic SDHx 30% sporadic PGL.",
        "surveillance": "Annual plasma/urine metanephrines + normetanephrines + 3-methoxytyramine; annual SSTR-PET (or alternating MRI); annual abdominal MRI; cascade first-degree SDHB testing; SDHB-IHC on all NET/PGL specimens",
        "targeted_rx": "Sunitinib malignant PGL (SDHB); temozolomide malignant PGL; PRRT (177Lu-DOTATATE) SSTR-positive pNET/PGL; somatostatin analogues functional NETs; everolimus pNET metastatic",
    },
]

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]

_TUMOUR_TYPES = {
    "MEN1":    ["Gastrinoma (Zollinger-Ellison)", "Insulinoma MEN1", "Non-functional pNET MEN1", "Bronchial carcinoid MEN1", "Thymic carcinoid MEN1"],
    "VHL":     ["pNET clear cell non-functional VHL", "ccRCC VHL", "Cerebellar hemangioblastoma VHL", "Retinal hemangioblastoma VHL", "Pheo/PGL VHL"],
    "RET":     ["Medullary thyroid carcinoma MEN2A", "Bilateral pheo MEN2A", "MEN2B MTC M918T", "Parathyroid adenoma MEN2A", "FMTC familial MTC"],
    "NF1":     ["Duodenal somatostatinoma NF1", "GIST NF1 (KIT wildtype)", "MPNST NF1", "Periampullary NET NF1", "Pheo NF1"],
    "TSC2":    ["Pulmonary carcinoid TSC", "Renal angiomyolipoma TSC", "SEGA TSC (subependymal giant cell)", "pNET TSC (rare)", "LAM lung TSC females"],
    "CDKN1B":  ["pNET MEN4 non-functional", "Pituitary prolactinoma MEN4", "Parathyroid adenoma MEN4", "Gastric carcinoid MEN4", "Cervical NET CDKN1B"],
    "PRKAR1A": ["PPNAD adrenal (subclinical Cushing)", "Cardiac myxoma Carney complex", "Somatotropinoma acromegaly CNC", "Testicular LCCSCT PRKAR1A", "Psammomatous melanotic schwannoma CNC"],
    "SDHB":    ["Malignant extra-adrenal PGL SDHB", "pNET SDHx-deficient", "Adrenal pheo SDHB", "GIST SDHx-deficient (KIT wildtype)", "Metastatic paraganglioma SDHB"],
}

_VARIANTS_BY_GENE = {
    "MEN1":    ["c.1546_1547insC (p.Leu516Profs)", "c.292C>T (p.Arg98Ter)", "c.784-9G>A splice MEN1", "Large exon deletion MEN1 exons 2-9", "c.1399C>T (p.Arg467Ter)"],
    "VHL":     ["c.194T>A (p.Val65Glu)", "c.482G>T (p.Cys161Phe)", "c.376G>A (p.Asp126Asn)", "Large deletion VHL exon 1-2", "c.208G>T (p.Glu70Ter)"],
    "RET":     ["c.1900T>C (p.Cys634Arg — MEN2A highest risk)", "c.2753T>G (p.Met918Thr — MEN2B)", "c.1826G>A (p.Cys609Tyr MEN2A)", "c.1832G>A (p.Cys611Trp MEN2A)", "c.2410G>A (p.Val804Met RET)"],
    "NF1":     ["c.2033_2046del14 (p.Met678Argfs)", "c.6579C>A (p.Tyr2193Ter)", "Large deletion NF1 17q11.2", "c.2407C>T (p.Arg803Ter)", "c.4006C>T (p.Arg1336Trp NF1)"],
    "TSC2":    ["c.1444C>T (p.Arg482Trp)", "c.3113G>A (p.Cys1038Tyr)", "Large deletion TSC2 exons 1-10", "c.4375_4376insA (p.Met1459Asnfs)", "c.5024G>A (p.Arg1675Gln)"],
    "CDKN1B":  ["c.326_327delCT (p.Thr109Metfs)", "c.508C>T (p.Arg170Cys)", "c.598_599delAG (p.Ser200Ter)", "c.326T>A (p.Val109Asp CDKN1B)", "Exon 1 frameshift CDKN1B"],
    "PRKAR1A": ["c.491_492delTG (p.Val164Glufs)", "c.709C>T (p.Arg237Ter)", "c.1A>G (p.Met1Val — start codon)", "Large deletion PRKAR1A exons 4-6", "c.578T>G (p.Leu193Arg PRKAR1A)"],
    "SDHB":    ["c.136C>T (p.Pro46Ser SDHB)", "c.343delA (p.Ile115Phefs)", "c.725G>A (p.Arg242His SDHB)", "Large deletion SDHB exon 1", "c.642+1G>T splice SDHB"],
}

TREATMENT_PROTOCOLS_BY_GENE = {
    "MEN1":    ["Octreotide/lanreotide SSA Level A antisecretory + antiproliferative", "Everolimus RADIANT-3 sporadic/refractory pNET only (NOT first-line MEN1)", "Streptozocin-based chemo advanced pNET", "PRRT 177Lu-DOTATATE SSTR-positive pNET", "Surgery: insulinoma enucleation; ZES PPI+SSA first"],
    "VHL":     ["Belzutifan HIF-2alpha inhibitor FDA2021 VHL-pNET+ccRCC+hemangioblastoma", "Sunitinib/everolimus refractory pNET VHL", "Surgery pNET >3cm or growing", "Cabozantinib ccRCC VHL", "Watch-and-wait pNET <3cm non-functional"],
    "RET":     ["Selpercatinib FDA2020 RET-mutant MTC and RET-fusion", "Vandetanib FDA2011 metastatic MTC multi-kinase", "Cabozantinib FDA2012 metastatic MTC", "Pralsetinib RET-mutant MTC", "Thyroidectomy codon-stratified timing MANDATORY"],
    "NF1":     ["Selumetinib FDA2020 MEK inhibitor NF1 plexiform neurofibroma", "Binimetinib MEK inhibitor adult NF1 trials", "Surgery primary NF1-GIST (imatinib INEFFECTIVE)", "Somatostatin analogues NF1 somatostatinoma symptomatic", "MPNST: doxorubicin+ifosfamide standard"],
    "TSC2":    ["Everolimus FDA2016 TSC-SEGA + AML + LAM + pNETs", "Sirolimus alternative mTORC1 inhibitor TSC", "Surgical resection SEGA if obstructive hydrocephalus", "Vigabatrin/ACTH infantile spasms TSC-epilepsy", "Pulmonary carcinoid: SSA symptomatic + everolimus"],
    "CDKN1B":  ["Somatostatin analogues pNET MEN4 functional", "Standard pNET: streptozocin-based or everolimus advanced", "Parathyroidectomy HPT MEN4", "Dopamine agonist (cabergoline) prolactinoma MEN4", "CDK4/6 inhibitors theoretical — no approved NET indication"],
    "PRKAR1A": ["Cardiac myxoma surgical resection + annual echo surveillance", "Bilateral laparoscopic adrenalectomy PPNAD frank Cushing", "SSA (octreotide/lanreotide) + pegvisomant GH excess acromegaly", "Ketoconazole/metyrapone Cushing bridging pre-surgery", "No specific NET targeted therapy — standard SSA"],
    "SDHB":    ["Sunitinib malignant PGL/pheo SDHB (FIRSTMAPPP trial)", "Temozolomide malignant PGL SDHB", "PRRT 177Lu-DOTATATE SSTR-positive PGL/pNET SDHx", "Cabozantinib PGL emerging data", "SSTR-PET 68Ga-DOTATATE staging MANDATORY"],
}

SURVEILLANCE_BY_GENE = {
    "MEN1":    ["Annual Ca2+/PTH + prolactin + fasting gastrin + insulin + IGF-1 + chromogranin A", "Pancreatic MRI every 1-2yr (pNET surveillance)", "Pituitary MRI every 3-5yr", "Annual parathyroid Ca2+/PTH (HPT 95%)", "First-degree relatives: germline MEN1 testing + annual biochemical screen"],
    "VHL":     ["Annual ophthalmology (retinal hemangioblastoma MANDATORY)", "Annual MRI brain + spine + abdomen (hemangioblastoma + ccRCC + pNET)", "Annual urine catecholamines/metanephrines (pheo 10-20%)", "Audiology annual (ELST hearing loss)", "Cascade first-degree VHL germline testing"],
    "RET":     ["Annual calcitonin + CEA (MTC surveillance; doubling time <6mo = urgent)", "Annual urine/plasma metanephrines (pheo screen BEFORE surgery)", "Annual Ca2+/PTH MEN2A (parathyroid 20-30%)", "Annual neck USS post-thyroidectomy", "Codon-stratified thyroidectomy timing cascade relatives"],
    "NF1":     ["Annual full-body clinical exam (skin + neuro + ophthalmology)", "Annual ophthalmology (Lisch nodules + optic glioma)", "Duodenoscopy + EUS from 25yr (somatostatinoma surveillance)", "Annual urine catecholamines (pheo NF1 1-5%)", "MRI brain/spine if neurological symptoms (MPNST)"],
    "TSC2":    ["Annual renal USS + MRI (AML 80% bilateral)", "Annual brain MRI (SEGA + cortical tubers)", "Annual pulmonary function + CT chest females (LAM)", "Annual ophthalmology (retinal astrocytoma)", "Echocardiogram neonatal/fetal (cardiac rhabdomyoma)"],
    "CDKN1B":  ["Annual Ca2+/PTH + prolactin + fasting gastrin + chromogranin A (same as MEN1)", "Pancreatic MRI every 1-2yr", "Pituitary MRI every 3-5yr", "First-degree relatives CDKN1B germline testing after MEN1 excluded", "Annual clinical exam for MEN4 features"],
    "PRKAR1A": ["Annual echocardiogram (cardiac myxoma LIFE-THREATENING embolic risk MANDATORY)", "Annual cortisol + ACTH (PPNAD Cushing screen)", "Annual IGF-1 + GH (acromegaly screen)", "Annual testicular USS males (LCCSCT PATHOGNOMONIC)", "Annual dermatology (lentigines + melanoma risk)"],
    "SDHB":    ["Annual plasma/urine metanephrines + normetanephrines + 3-methoxytyramine", "Annual SSTR-PET (68Ga-DOTATATE) or alternating MRI abdomen/pelvis", "Annual abdominal MRI (extra-adrenal retroperitoneal PGL)", "Cascade first-degree SDHB germline testing", "SDHB-IHC on all NET/PGL/GIST specimens"],
}


def _make_patients(gene: str, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    g = next(g for g in ATLAS_GENES if g["gene"] == gene)
    tumours = _TUMOUR_TYPES.get(gene, ["NET NOS"])
    variants = _VARIANTS_BY_GENE.get(gene, ["Pathogenic variant"])
    pts = []
    for i in range(n):
        age = rng.randint(35, 80) if gene != "SDHB" else rng.randint(35, 80)
        pts.append({
            "patient_id": f"{gene[:4]}-HNET-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_dx": age,
            "tumour_type": rng.choice(tumours),
            "variant": rng.choice(variants),
            "stage": rng.choice(["Localised", "Locally Advanced", "Metastatic"]),
            "somatostatin_eligible": rng.random() < (
                0.85 if gene == "MEN1" else
                0.60 if gene in ("VHL", "SDHB") else
                0.30 if gene in ("RET", "NF1") else
                0.50 if gene == "TSC2" else
                0.70 if gene == "CDKN1B" else
                0.40
            ),
            "mtor_eligible": rng.random() < (
                0.80 if gene == "TSC2" else
                0.25 if gene in ("MEN1", "VHL", "SDHB") else
                0.10
            ),
            "ret_targeted": rng.random() < (
                0.85 if gene == "RET" else 0.05
            ),
            "malignant_pgl": rng.random() < (
                0.38 if gene == "SDHB" else
                0.12 if gene in ("VHL", "RET") else
                0.04
            ),
            "relapse": rng.random() < 0.35,
            "sstr_pet_done": rng.random() < (
                0.90 if gene == "SDHB" else
                0.65 if gene in ("MEN1", "VHL", "NF1") else
                0.45
            ),
        })
    return pts


def generate_overview() -> dict:
    cohorts = {}
    for i, g in enumerate(ATLAS_GENES):
        gene = g["gene"]
        pts = _make_patients(gene, SEED_BASE + i)
        cohorts[gene] = pts

    total = sum(len(v) for v in cohorts.values())
    gene_counts = {g: len(pts) for g, pts in cohorts.items()}

    somatostatin_rate = round(
        100 * sum(p["somatostatin_eligible"] for pts in cohorts.values() for p in pts) / total, 1
    )
    mtor_rate = round(
        100 * sum(p["mtor_eligible"] for pts in cohorts.values() for p in pts) / total, 1
    )
    ret_targeted_rate = round(
        100 * sum(p["ret_targeted"] for pts in cohorts.values() for p in pts) / total, 1
    )
    malignant_pgl_rate = round(
        100 * sum(p["malignant_pgl"] for pts in cohorts.values() for p in pts) / total, 1
    )
    mean_age = round(
        sum(p["age_at_dx"] for pts in cohorts.values() for p in pts) / total, 1
    )

    return {
        "atlas": "Hereditary-Neuroendocrine-Tumor-NET-Carcinoid-Predisposition-Atlas",
        "genes": _GENE_LIST,
        "total_patients": total,
        "gene_counts": gene_counts,
        "seed_range": f"{SEED_BASE}-{SEED_BASE+7}",
        "somatostatin_eligible_rate_pct": somatostatin_rate,
        "mtor_eligible_rate_pct": mtor_rate,
        "ret_targeted_rate_pct": ret_targeted_rate,
        "malignant_pgl_rate_pct": malignant_pgl_rate,
        "mean_age_at_dx": mean_age,
        "key_facts": [
            "MEN1: pNETs 40-70%; gastrinoma ZES PATHOGNOMONIC; HPT 95%+ concurrent; everolimus NOT first-line MEN1; somatostatin analogues Level A",
            "VHL pNET: clear cell cytoplasm EXCLUSIVELY — NON-FUNCTIONAL low grade; belzutifan HIF-2alpha inhibitor FDA 2021 covers pNET+ccRCC+hemangioblastoma",
            "RET: MTC 100% penetrance MEN2; codon-based thyroidectomy timing MANDATORY (M918T by 6 months); screen pheo BEFORE thyroid surgery",
            "NF1: NETs = DUODENAL periampullary somatostatinoma PATHOGNOMONIC + psammoma bodies; NF1-GIST imatinib-RESISTANT (not KIT/PDGFRA); NOT primarily pNETs",
            "TSC2: mTOR is DIRECTLY therapeutic (constitutive mTORC1); everolimus FDA 2016 TSC-pNET+SEGA+AML+LAM; cardiac rhabdomyoma fetal PATHOGNOMONIC",
            "CDKN1B/MEN4: MEN1-phenotype + MEN1-germline-NEGATIVE = test CDKN1B; MEN4 is diagnosis of exclusion; ~3% MEN1-negative MEN-like families; same surveillance as MEN1",
            "PRKAR1A/Carney complex: cardiac myxoma LIFE-THREATENING embolic risk — annual echo MANDATORY; PPNAD ACTH-independent Cushing PATHOGNOMONIC; testicular LCCSCT males PATHOGNOMONIC",
            "SDHB: malignant PGL 35-40% HIGHEST SDHx gene; SDHB-IHC negative = global SDHx marker PATHOGNOMONIC; SSTR-PET MANDATORY; extra-adrenal retroperitoneal PGL enriched",
        ],
    }


def generate_breakdown() -> dict:
    breakdown = {}
    from collections import Counter
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
                "net_risk": g["net_risk"],
                "pathognomonic": g["pathognomonic"],
                "key_avoid": g["key_avoid"],
                "key_rule": g["key_rule"],
                "surveillance": g["surveillance"],
                "targeted_rx": g["targeted_rx"],
            },
            "n": len(pts),
            "somatostatin_eligible_pct": round(100 * sum(1 for p in pts if p["somatostatin_eligible"]) / len(pts), 1),
            "mtor_eligible_pct": round(100 * sum(1 for p in pts if p["mtor_eligible"]) / len(pts), 1),
            "ret_targeted_pct": round(100 * sum(1 for p in pts if p["ret_targeted"]) / len(pts), 1),
            "malignant_pgl_pct": round(100 * sum(1 for p in pts if p["malignant_pgl"]) / len(pts), 1),
            "relapse_pct": round(100 * sum(1 for p in pts if p["relapse"]) / len(pts), 1),
            "sstr_pet_done_pct": round(100 * sum(1 for p in pts if p["sstr_pet_done"]) / len(pts), 1),
            "mean_age": round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "top_tumour_types": [{"type": t, "count": c} for t, c in top_tumours],
            "top_variants": [{"variant": v, "count": c} for v, c in top_variants],
            "treatment_protocols": TREATMENT_PROTOCOLS_BY_GENE[gene],
            "surveillance_protocols": SURVEILLANCE_BY_GENE[gene],
        }
    return {"breakdown": breakdown, "genes": _GENE_LIST}


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Neuroendocrine-Tumor-NET-Carcinoid-Predisposition-Atlas",
        "definitions": {
            "men1_gastrinoma_zes": (
                "MEN1 pNET: Menin nuclear scaffold 610aa LOF — pNETs 40-70% lifetime; "
                "GASTRINOMA / ZOLLINGER-ELLISON: PATHOGNOMONIC for MEN1-gastrinoma — gastric acid hypersecretion + refractory peptic ulcers; "
                "CONCURRENT HPT 95%+: Ca2+/PTH ALWAYS elevated in MEN1 — screen every visit; "
                "PITUITARY 30-50%: prolactinoma most common; "
                "EVEROLIMUS NOT FIRST-LINE: mTOR is secondary in MEN1 (primary = menin-LOH); reserve for sporadic/refractory; "
                "SOMATOSTATIN ANALOGUES Level A: octreotide/lanreotide antisecretory + antiproliferative MEN1-pNET"
            ),
            "vhl_pnet_clear_cell": (
                "VHL pNET: HIF-alpha substrate adaptor 213aa LOF — pNETs 15-17% EXCLUSIVELY clear cell cytoplasm; "
                "NON-FUNCTIONAL: VHL pNETs rarely cause hormonal syndromes (unlike MEN1 gastrinoma/insulinoma); "
                "BELZUTIFAN FDA 2021: HIF-2alpha inhibitor — single agent VHL-ccRCC + VHL-pNET + VHL-hemangioblastoma; "
                "HEMANGIOBLASTOMA: cerebellum + retina + spinal cord PATHOGNOMONIC VHL — annual ophthalmology; "
                "PATHOGNOMONIC RULE: clear cell cytoplasm pNET = VHL until proven otherwise; "
                "WATCH <3cm: non-functional pNET <3cm = surveillance; >3cm or growing = surgery"
            ),
            "ret_codon_thyroidectomy": (
                "RET MTC: receptor tyrosine kinase GOF 1114aa — MTC 100% penetrance MEN2; "
                "CODON-BASED THYROIDECTOMY TIMING: M918T (MEN2B) thyroidectomy BY 6 MONTHS; C634F/R by 5yr; other MEN2A by 5-10yr; "
                "PHEO SCREEN FIRST: ALWAYS screen pheo before thyroid surgery in MEN2 — untreated pheo = hypertensive crisis fatal; "
                "SELPERCATINIB FDA 2020: RET-selective kinase inhibitor — RET-mutant MTC + RET-fusion tumours; "
                "VANDETANIB + CABOZANTINIB: approved metastatic MTC; vandetanib also inhibits VEGFR/EGFR; "
                "BILATERAL PHEO 50%: MEN2A pheo bilateral — bilateral laparoscopic adrenalectomy"
            ),
            "nf1_duodenal_somatostatinoma": (
                "NF1 NETs: neurofibromin RAS-GAP 2839aa LOF — DUODENAL PERIAMPULLARY somatostatinomas PATHOGNOMONIC; "
                "PSAMMOMA BODIES: concentric calcifications in NF1 somatostatinoma on histology PATHOGNOMONIC; "
                "NOT pNETs: NF1-NETs are DUODENAL not pancreatic (distinct from MEN1/VHL/TSC); "
                "NF1-GIST: KIT wildtype/PDGFRA wildtype — imatinib INEFFECTIVE; surgery primary; "
                "SELUMETINIB FDA 2020: MEK inhibitor — paediatric NF1 plexiform neurofibroma; "
                "CAFÉ-AU-LAIT 6+: PATHOGNOMONIC NF1 diagnostic criterion; Lisch nodules iris PATHOGNOMONIC"
            ),
            "tsc2_mtor_direct": (
                "TSC2 NETs: tuberin mTOR-regulatory GAP 1807aa LOF — mTOR IS DIRECTLY THE THERAPEUTIC TARGET; "
                "CONSTITUTIVE mTORC1: TSC2 loss → Rheb-GTP uninhibited → mTORC1 hyperactive — everolimus restores inhibition; "
                "EVEROLIMUS FDA 2016: TSC-SEGA + AML + LAM + pNETs — covers ALL major TSC manifestations; "
                "SEGA PATHOGNOMONIC: subependymal giant cell astrocytoma — everolimus first-line medical; "
                "CARDIAC RHABDOMYOMA: fetal PATHOGNOMONIC TSC — usually regresses; echo neonatal; "
                "KEY DISTINCTION: TSC mTOR is primary driver; MEN1 mTOR is secondary — everolimus NOT first-line MEN1 but IS first-line TSC"
            ),
            "cdkn1b_men4_exclusion": (
                "CDKN1B MEN4: p27/KIP1 cell-cycle inhibitor 196aa LOF — MEN1-phenotype with MEN1-germline-NEGATIVE; "
                "DIAGNOSIS OF EXCLUSION: MEN4 requires MEN1 full sequencing + deletion/duplication (MLPA) NEGATIVE first; "
                "PREVALENCE ~3%: of MEN1-negative MEN-like families — rare but real; "
                "SAME SURVEILLANCE AS MEN1: Ca2+/PTH + prolactin + gastrin + chromogranin A annually; pancreatic MRI + pituitary MRI; "
                "CDK4/6 INHIBITORS: theoretically relevant (CDK2/CDK4 are CDKN1B substrates) — no approved NET indication; "
                "PRACTICAL RULE: every MEN1-negative MEN-phenotype family should be offered CDKN1B testing after MEN1 exclusion"
            ),
            "prkar1a_carney_myxoma": (
                "PRKAR1A Carney complex: PKA regulatory subunit 1A 381aa LOF — constitutive PKA → CREB → ACTH-independent cortisol; "
                "CARDIAC MYXOMA LIFE-THREATENING: 50-60% Carney complex — embolic stroke/MI if untreated; annual echo MANDATORY; "
                "PPNAD PATHOGNOMONIC: primary pigmented nodular adrenocortical disease — subclinical Cushing ACTH-independent bilateral micronodular; "
                "TESTICULAR LCCSCT PATHOGNOMONIC: large-cell calcifying Sertoli cell tumour in males — annual testicular USS; "
                "PSAMMOMATOUS MELANOTIC SCHWANNOMA PATHOGNOMONIC: peripheral nerve melanin + psammoma bodies; "
                "LENTIGINES CENTROFACIAL PATHOGNOMONIC: lip + conjunctival + skin perioral — diagnostic marker"
            ),
            "sdhb_malignant_pgl": (
                "SDHB PGL/pNET: succinate dehydrogenase subunit B 280aa LOF — MALIGNANT PGL 35-40% HIGHEST SDHx gene; "
                "SDHB-IHC PATHOGNOMONIC: SDHB antibody negative = SDHx-deficient (covers SDHB/C/D/A loss) — diagnostic tool for all SDHx; "
                "EXTRA-ADRENAL RETROPERITONEAL PGL: SDHB enriched location (vs adrenal enriched in MEN2/VHL); "
                "SSTR-PET MANDATORY: 68Ga-DOTATATE superior to CT/MRI for occult SDHx PGL staging; "
                "SUCCINATE PSEUDOHYPOXIA: succinate accumulates → inhibits PHDs → HIF stabilisation (oncometabolite); "
                "SOMATIC SDHx 30%: sporadic PGL have somatic SDHx — always test tumour SDHx/SDHB-IHC even without family history; "
                "SUNITINIB MALIGNANT PGL: FDA-approved-class; temozolomide alternative; PRRT 177Lu-DOTATATE SSTR-positive",
            ),
            "cascade_testing_hnet": (
                "CASCADE TESTING Hereditary NET/Carcinoid Predisposition: "
                "MEN1: all first-degree — annual Ca2+/PTH + gastrin + prolactin + IGF-1; pancreatic MRI; "
                "VHL: first-degree — annual ophthalmology + MRI brain/spine/abdomen + catecholamines; "
                "RET: codon-stratified thyroidectomy timing; annual calcitonin + catecholamines; "
                "NF1: clinical exam; duodenoscopy from 25yr; catecholamines; "
                "TSC2: renal USS + brain MRI + echo; everolimus when lesions symptomatic; "
                "CDKN1B: same as MEN1 after MEN1 excluded; "
                "PRKAR1A: annual echo + cortisol + IGF-1 + testicular USS males; "
                "SDHB: annual metanephrines + SSTR-PET + SDHB-IHC on all specimens"
            ),
        },
        "key_clinical_distinctions": [
            "MEN1 everolimus NOT first-line: mTOR secondary in MEN1 (menin-LOH primary); TSC2 everolimus IS first-line (mTOR directly constitutive)",
            "VHL pNET = clear cell NON-FUNCTIONAL: belzutifan FDA 2021 covers pNET+ccRCC+hemangioblastoma in ONE agent",
            "RET codon-based thyroidectomy: M918T (MEN2B) by 6 months; screen pheo BEFORE thyroid surgery — hypertensive crisis risk",
            "NF1 NETs = DUODENAL periampullary somatostatinoma + psammoma bodies PATHOGNOMONIC; NOT pNETs; NF1-GIST imatinib-resistant",
            "CDKN1B/MEN4: MEN1-phenotype + MEN1-germline-negative = test CDKN1B; diagnosis of exclusion; ~3% MEN1-negative families",
            "PRKAR1A: cardiac myxoma annual echo MANDATORY (embolic LIFE-THREATENING); PPNAD ACTH-independent Cushing; testicular LCCSCT PATHOGNOMONIC males",
            "SDHB: 35-40% malignant PGL — most aggressive SDHx gene; SDHB-IHC negative = global SDHx marker; SSTR-PET MANDATORY staging",
            "Universal rule: SDHB-IHC on every NET/PGL/GIST specimen regardless of family history (somatic SDHx 30% sporadic PGL)",
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2))
