#!/usr/bin/env python3
"""Hereditary-Thyroid-Cancer-Atlas — Complete 8-Gene Hereditary Thyroid Cancer Atlas
RET    (Rearranged during Transfection; 1114aa; 10q11.21; AD GOF;
         Multiple Endocrine Neoplasia Type 2A/2B / FMTC;
         Medullary Thyroid Ca 95%+ lifetime — ALL RET carriers require prophylactic thyroidectomy;
         MEN2B: thyroidectomy by age 6 months — MOST URGENT hereditary cancer surgery;
         Vandetanib/Cabozantinib/Selpercatinib RET-specific TKI FDA-approved;
         seed SEED_BASE+0) ·
TP53   (Tumour Protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome — anaplastic thyroid Ca;
         AVOID RADIATION ABSOLUTELY in germline TP53 LFS;
         WBMRI annually Toronto protocol;
         seed SEED_BASE+1) ·
PTEN   (Phosphatase and Tensin Homolog; 403aa; 10q23.31; AD LOF;
         Cowden Syndrome / PHTS;
         Follicular thyroid Ca 25-38% lifetime — second most common PHTS cancer after breast;
         Macrocephaly PATHOGNOMONIC; Lhermitte-Duclos PATHOGNOMONIC;
         mTOR inhibitor Everolimus FDA;
         seed SEED_BASE+2) ·
APC    (Adenomatous Polyposis Coli; 2843aa; 5q22.2; AD LOF;
         Familial Adenomatous Polyposis / Gardner Syndrome;
         Cribriform-morular variant PTC PATHOGNOMONIC for FAP;
         Annual thyroid ultrasound in all FAP patients;
         seed SEED_BASE+3) ·
PRKAR1A (Protein Kinase cAMP-dependent type I Regulatory subunit Alpha; 381aa; 17q24.2; AD LOF;
         Carney Complex;
         Follicular thyroid Ca 75% lifetime — dominant endocrine cancer in Carney Complex;
         Cardiac myxoma 30% — LIFE-THREATENING — annual echo MANDATORY;
         Spotty skin pigmentation PATHOGNOMONIC;
         seed SEED_BASE+4) ·
DICER1 (Double-stranded RNA-specific Endoribonuclease; 1922aa; 14q32.13; AD LOF;
         DICER1 Syndrome / FAPOL;
         Multinodular goiter — most common DICER1 thyroid manifestation;
         Differentiated thyroid Ca (DTC) — papillary and follicular;
         Pleuropulmonary blastoma PATHOGNOMONIC in infancy;
         seed SEED_BASE+5) ·
CDC73  (Cell Division Cycle 73 / Parafibromin; 531aa; 1q31.2; AD LOF;
         Hyperparathyroidism-Jaw Tumour Syndrome / CDC73-related disorders;
         Parathyroid carcinoma PATHOGNOMONIC — 10-15% HPT-JT develop parathyroid carcinoma;
         Oxyphilic follicular thyroid adenoma/carcinoma;
         seed SEED_BASE+6) ·
VHL    (Von Hippel-Lindau; 213aa; 3p25.3; AD LOF;
         VHL Disease;
         Clear-cell follicular thyroid Ca — Type 2C VHL;
         Hemangioblastoma CNS/retinal PATHOGNOMONIC;
         Pheochromocytoma + ccRCC;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3134-3141)
"""
import random

SEED_BASE = 3134

ATLAS_GENES = [
    {
        "gene": "RET",
        "protein": (
            "RET -- 10q11.21 Autosomal-Dominant-GOF -- 1114aa -- "
            "Rearranged-during-Transfection-Receptor-Tyrosine-Kinase-GDNF-Family-Ligand-Receptor-"
            "MEN2A-MEN2B-FMTC-Medullary-Thyroid-Ca-95pct-Pheochromocytoma-Vandetanib-Cabozantinib-"
            "Selpercatinib-RET-TKI-FDA-OMIM-164761"
        ),
        "locus": "10q11.21",
        "protein_size": (
            "1114 aa / 10q11.21 RET encodes Rearranged during Transfection receptor tyrosine kinase (RET): "
            "STRUCTURE: "
            "  N-terminal cadherin-like domains (ligand-binding with GFRα co-receptors); "
            "  Cysteine-rich domain (dimerisation; MEN2A mutations cluster here: C609, C611, C618, C620, C630, C634); "
            "  Transmembrane domain; "
            "  Intracellular juxtamembrane domain; "
            "  Kinase domain: 2 lobes, ATP-binding cleft (MEN2B M918T here — highest risk); "
            "  RET ligands: GDNF, neurturin, artemin, persephin (via GFRα1-4 co-receptors); "
            "  Signalling: PI3K-Akt, RAS-MAPK, JNK, STAT3 — proliferation + survival; "
            "  GOF mutations → constitutive dimerisation (cysteine mutations) or kinase activation (M918T); "
            "MEN2A AND MEN2B — THREE SYNDROMES: "
            "  MEN2A (Sipple Syndrome): "
            "    RET mutations: cysteine-rich domain — C634 most common (50% MEN2A); also C620, C618; "
            "    COMPONENTS: "
            "      1. Medullary Thyroid Carcinoma (MTC): 95% lifetime — INVARIABLE component; "
            "      2. Pheochromocytoma: 40-50% MEN2A; bilateral 50% of pheo cases; "
            "      3. Primary Hyperparathyroidism: 20-30% MEN2A (multigland adenoma); "
            "    MTC ONSET: median 30-40yr in MEN2A (later than MEN2B); "
            "  MEN2B (most aggressive): "
            "    RET mutation: M918T (exon 16) — 95% MEN2B cases; "
            "    COMPONENTS: "
            "      1. MTC: 95% — onset in INFANCY; biochemically positive (calcitonin) first year of life; "
            "      2. Pheochromocytoma: 40-50%; bilateral; "
            "      3. Mucosal neuromas: tongue/lips/eyelids PATHOGNOMONIC; "
            "      4. Marfanoid habitus: tall, thin, long limbs — WITHOUT lens dislocation; "
            "      5. Ganglioneuromatosis: GI tract dysmotility; "
            "    THYROIDECTOMY: age 6 months — MOST URGENT prophylactic cancer surgery in oncogenetics; "
            "      Delay beyond 6mo risks lymph node metastasis in MEN2B; "
            "  FMTC (Familial MTC only): "
            "    RET mutations: C609, C611, C618, C620, S891A, E768D, V804L, V804M; "
            "    MTC only — NO pheochromocytoma, NO hyperparathyroidism; "
            "    Thyroidectomy: typically recommended by age 5 (Category C risk); "
            "RET GENOTYPE-PHENOTYPE RISK STRATIFICATION (ATA 2015): "
            "  HIGHEST RISK (Category D): M918T (MEN2B) → thyroidectomy ≤6 months; "
            "  HIGH RISK (Category C): C634 (MEN2A), A883F, C609, C611, C618, C620 → thyroidectomy ≤5yr; "
            "  MODERATE RISK (Category B): E768D, L790F, V804L, V804M, S891A, R912P → individualised; "
            "TARGETED THERAPY: "
            "  Vandetanib (Caprelsa): FDA 2011 — RET/VEGFR/EGFR TKI; progressive/metastatic MTC; "
            "  Cabozantinib (Cabometyx/Cometriq): FDA 2012 — RET/VEGFR/MET TKI; progressive MTC; "
            "  Selpercatinib (Retevmo): FDA 2020 — selective RET TKI (LIBRETTO-001); "
            "    RET-fusion solid tumours + RET-mutant MTC + RET-mutant NSCLC; "
            "    Superior selectivity → less off-target toxicity vs vandetanib/cabozantinib; "
            "  Pralsetinib (Gavreto): FDA 2020 — selective RET TKI; "
            "CALCITONIN SURVEILLANCE: "
            "  Baseline + annual calcitonin + CEA: 6-monthly in known MTC; "
            "  Calcitonin doubling time: <6 months = poor prognosis; 6-24 months = intermediate; "
            "PHEO SCREENING: "
            "  Annual plasma metanephrines or 24hr urine metanephrines from age 8 (MEN2A/MEN2B); "
            "  MRI adrenal (avoid CT contrast pre-block): bilateral pheo common; "
            "  α-blockade (phenoxybenzamine or doxazosin) BEFORE surgery"
        ),
        "inheritance": (
            "AD GOF 10q11.21 — RET. MEN2A + MEN2B combined prevalence ~1:30,000. "
            "FMTC: ~20% RET families. "
            "De novo M918T: 25-50% MEN2B cases (spontaneous); confirm parental testing. "
            "Cascade testing: all first-degree relatives of RET-positive index. "
            "Calcitonin + annual biochemical surveillance all confirmed carriers. "
            "Risk stratification: ATA 2015 Category D/C/B determines thyroidectomy timing."
        ),
        "surveillance_key": "thyroidectomy ≤6 months MEN2B (M918T); thyroidectomy ≤5yr Category C (C634); annual calcitonin+CEA; annual plasma metanephrines from age 8; selpercatinib/vandetanib/cabozantinib for metastatic MTC; ATA genotype-risk stratification mandatory",
        "pathognomonic": "MEN2B: mucosal neuromas tongue/lips/eyelids PATHOGNOMONIC; marfanoid habitus without lens dislocation; ganglioneuromatosis GI; M918T = highest risk → thyroidectomy ≤6 months",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-Tumour-Suppressor-Guardian-Genome-Transcription-Factor-"
            "Li-Fraumeni-Syndrome-Anaplastic-Thyroid-Ca-AVOID-RADIATION-ABSOLUTELY-"
            "WBMRI-Toronto-Protocol-Annual-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 17p13.1 TP53 encodes tumour protein p53 (p53): "
            "STRUCTURE: "
            "  N-terminal transactivation domains (TAD1 + TAD2 — MDM2-binding TAD1); "
            "  Proline-rich region; "
            "  Central DNA-binding domain (DBD; residues 100-290) — 90% cancer mutations cluster here; "
            "    Hotspot residues: R175, G245, R248, R249, R273, R282 (contact or structural); "
            "  Tetramerisation domain (p53 functions as homotetramer = 2 dimers); "
            "  C-terminal regulatory domain (acetylation + ubiquitination); "
            "  MDM2 negative-feedback loop: MDM2 transcribed by p53; MDM2 ubiquitinates p53 → proteasomal degradation; "
            "  p53 functions: G1/S + G2/M checkpoint arrest, apoptosis, senescence, DNA repair coordination, metabolic regulation; "
            "LI-FRAUMENI SYNDROME (LFS) — THYROID COMPONENT: "
            "  Anaplastic thyroid carcinoma (ATC): most aggressive thyroid malignancy; "
            "    Somatic TP53: in 73-88% sporadic ATC — MOST mutated gene in ATC; "
            "    Germline TP53 (LFS): rare cause of ATC but somatic-to-germline transition documented; "
            "    ATC diagnosis < 60yr without radiation exposure → consider germline TP53; "
            "  DIFFERENTIATED THYROID CA in LFS: "
            "    Papillary thyroid Ca (PTC): modestly elevated risk (3-5x RR); "
            "    Often co-occurs with other LFS cancers (sarcoma, breast, CNS, ACC); "
            "  LFS CORE TUMOUR SPECTRUM: "
            "    Osteosarcoma/soft tissue sarcoma (most common — 35%); "
            "    Pre-menopausal breast Ca (<35yr — 28-30%); "
            "    Brain tumours (glioma, medulloblastoma — 15%); "
            "    Adrenocortical carcinoma (ACC — paediatric PATHOGNOMONIC); "
            "    AVOID RADIATION ABSOLUTELY: "
            "      Radiation-field osteosarcoma/sarcoma documented in LFS patients after therapeutic radiation; "
            "      Radiation is PROHIBITED for treating cancers in germline TP53 carriers; "
            "    WBMRI PROTOCOL (Toronto/Villani 2016): "
            "      Annual whole-body MRI + brain MRI: detects internal tumours before symptoms; "
            "      Annual abdominal US: ACC screening (paediatric + adult); "
            "      Biennial mammography + annual breast MRI: breast surveillance from age 20-25; "
            "      Colonoscopy every 2-5yr from age 25; "
            "ANAPLASTIC THYROID CA — CLINICAL MANAGEMENT: "
            "  Multimodal: surgery (if resectable) + hyperfractionated EBRT + chemotherapy; "
            "    EBRT in germline TP53: CONTRAINDICATED — alternative systemic approaches; "
            "  BRAF V600E + ATC: BRAF/MEK inhibitor (dabrafenib + trametinib) FDA 2018 — highly active; "
            "  RET-fusion ATC: selpercatinib — remarkable responses; "
            "  NTRK-fusion ATC: larotrectinib/entrectinib; "
            "  Lenvatinib + pembrolizumab: active in refractory ATC; "
            "TP53 SOMATIC AND THYROID CA PROGRESSION: "
            "  PTC → poorly differentiated (PDTC) → ATC: TP53 acquired late → dedifferentiation; "
            "  TP53 somatic = progression marker in differentiated → anaplastic transition"
        ),
        "inheritance": (
            "AD LOF 17p13.1 — TP53. LFS prevalence ~1:5,000-1:20,000. "
            "De novo TP53: ~7-20% LFS families (no family history). "
            "TP53 missense vs truncating: GOF missense (R175H, R273H) may confer dominant-negative/GOF activity. "
            "TP53 c.1010G>A (p.R337H): Brazilian founder mutation — lower penetrance, ACC predominant. "
            "AVOID RADIATION ABSOLUTELY in all germline TP53 carriers — documented radiation-field sarcoma. "
            "Cascade testing: all first-degree relatives. "
            "Prenatal/preimplantation genetic testing available."
        ),
        "surveillance_key": "AVOID ALL RADIATION in germline TP53 LFS carriers; annual WBMRI + brain MRI Toronto protocol; annual abdominal US (ACC); breast surveillance from age 20-25; TP53 somatic = ATC progression marker",
        "pathognomonic": "ACC in child <5yr PATHOGNOMONIC for LFS (test TP53 immediately); ATC with germline TP53 → AVOID EBRT — contraindicated; radiation-field sarcoma documented risk",
    },
    {
        "gene": "PTEN",
        "protein": (
            "PTEN -- 10q23.31 Autosomal-Dominant-LOF -- 403aa -- "
            "Phosphatase-Tensin-Homolog-Lipid-Protein-Phosphatase-PI3K-Akt-mTOR-Brake-"
            "Cowden-Syndrome-PHTS-THYROID-25-38pct-Follicular-Thyroid-Ca-"
            "Macrocephaly-PATHOGNOMONIC-Lhermitte-Duclos-PATHOGNOMONIC-Everolimus-OMIM-601728"
        ),
        "locus": "10q23.31",
        "protein_size": (
            "403 aa / 10q23.31 PTEN encodes Phosphatase and Tensin Homolog (PTEN): "
            "STRUCTURE: N-terminal phosphatase domain — "
            "  dual-specificity lipid (PIP3→PIP2) + protein phosphatase; "
            "  C2 domain (membrane localisation); "
            "  C-terminal PDZ-binding motif; "
            "  PTEN is dominant brake on PI3K-Akt-mTOR: "
            "    PI3K: PIP2→PIP3 → Akt activation; "
            "    PTEN: PIP3→PIP2 → Akt OFF; "
            "  PTEN LOF → constitutive PI3K-Akt-mTOR → thyroid and multi-organ proliferation; "
            "COWDEN SYNDROME / PHTS — THYROID: "
            "  THYROID CANCER: 25-38% lifetime — follicular thyroid Ca most common; "
            "    Also: follicular variant PTC, Hürthle cell carcinoma; "
            "    Thyroid Ca SECOND most common cancer in PHTS females (after breast 85%); "
            "    Multiple benign thyroid nodules / goiter in ~75% PHTS — background for malignancy; "
            "  PATHOGNOMONIC FEATURES: "
            "    Macrocephaly (≥97th centile OFC): MOST CONSISTENT clinical feature — present in 94%; "
            "    Lhermitte-Duclos disease (dysplastic cerebellar gangliocytoma): PATHOGNOMONIC for PHTS; "
            "      ANY adult-onset Lhermitte-Duclos → test PTEN immediately; "
            "    Trichilemmomas (facial): PATHOGNOMONIC; "
            "    Mucocutaneous lesions: cobblestone papillomatous mucosa, papillomatous papules; "
            "    Acral keratoses; "
            "    Glycogenic acanthosis oesophagus (PATHOGNOMONIC on EGD); "
            "  OTHER CANCERS: "
            "    Breast Ca: 85% lifetime — dominant cancer; MRI surveillance from age 30; "
            "    Endometrial Ca: 28-44% lifetime; "
            "    ccRCC: 33% lifetime; "
            "    Melanoma: 6% lifetime; "
            "  AUTISM SPECTRUM DISORDER: 20-23% PHTS patients — consider PTEN testing in macrocephalic ASD; "
            "  BANNAYAN-RILEY-RUVALCABA (BRR): allelic PHTS with macrocephaly, hamartomas, lipomas, pigmented penile macules; "
            "THYROID SURVEILLANCE: "
            "  Annual thyroid ultrasound from age 7-10yr or 5yr before earliest thyroid Ca family case; "
            "  FNA for nodules ≥1cm or suspicious features; "
            "  Thyroid Ca in PHTS often multifocal → total thyroidectomy if surgical; "
            "TARGETED THERAPY: "
            "  Everolimus (mTOR inhibitor): FDA-approved for PTEN-deficient tumours (renal, breast); "
            "    Active in PHTS-related thyroid and breast Ca; "
            "  PI3K inhibitors: alpelisib (Piqray) — for PIK3CA-altered tumours; "
            "  AKT inhibitors: capivasertib — Phase 3 data emerging"
        ),
        "inheritance": (
            "AD LOF 10q23.31 — PTEN. ~1:200,000-1:250,000 (may be underdiagnosed). "
            "Large deletions/duplications: ~5-10% pathogenic PTEN — MLPA mandatory. "
            "De novo PTEN: ~10-15% Cowden Syndrome. "
            "PTEN VUS: pathogenicity scoring by functional assay (PTEN activity). "
            "Mosaic PTEN: subcutaneous lipoma + asymmetric overgrowth without skin signs — sequence skin. "
            "AUTISM + macrocephaly: PTEN prevalence ~10-20% — test macrocephalic ASD."
        ),
        "surveillance_key": "annual thyroid ultrasound from age 7-10; macrocephaly (≥97th centile) MOST CONSISTENT PHTS sign; Lhermitte-Duclos → test PTEN immediately; breast MRI from age 30; total thyroidectomy if multifocal Ca",
        "pathognomonic": "Lhermitte-Duclos (dysplastic cerebellar gangliocytoma) PATHOGNOMONIC for PHTS/PTEN; macrocephaly ≥97th centile most consistent sign; trichilemmomas PATHOGNOMONIC; glycogenic acanthosis oesophagus PATHOGNOMONIC",
    },
    {
        "gene": "APC",
        "protein": (
            "APC -- 5q22.2 Autosomal-Dominant-LOF -- 2843aa -- "
            "Adenomatous-Polyposis-Coli-WNT-Pathway-Gatekeeper-β-catenin-Degradation-"
            "FAP-Gardner-Syndrome-Cribriform-Morular-Variant-PTC-PATHOGNOMONIC-FAP-"
            "Annual-Thyroid-US-All-FAP-Patients-OMIM-611731"
        ),
        "locus": "5q22.2",
        "protein_size": (
            "2843 aa / 5q22.2 APC encodes Adenomatous Polyposis Coli tumour suppressor: "
            "STRUCTURE: "
            "  N-terminal homodimerisation domain; "
            "  Armadillo repeats (protein-protein interactions); "
            "  15-amino-acid repeat region (β-catenin binding); "
            "  20-amino-acid repeat region (β-catenin degradation); "
            "  SAMP repeats (axin binding); "
            "  C-terminal microtubule-binding and EB1-binding domains; "
            "  APC function: scaffold in β-catenin destruction complex with axin, CK1α, GSK3β; "
            "    Normally: CK1α phosphorylates β-catenin ser45; GSK3β phosphorylates ser33/37/thr41; "
            "    Phospho-β-catenin → ubiquitylated by βTrCP → proteasomal degradation; "
            "    APC LOF → β-catenin accumulates → nuclear → TCF/LEF → WNT target genes; "
            "  Mutation cluster region (MCR): codons 1250-1464 — most pathogenic APC mutations here; "
            "FAP / GARDNER SYNDROME — THYROID: "
            "  FAMILIAL ADENOMATOUS POLYPOSIS (FAP): "
            "    Colorectal polyps: >100 adenomas (classic FAP); 10-99 adenomas (attenuated AFAP); "
            "    ABSOLUTE CRC RISK: ~100% by age 40-50 if untreated; prophylactic colectomy mandatory; "
            "  THYROID CANCER IN FAP — CRIBRIFORM-MORULAR VARIANT PTC: "
            "    CRIBRIFORM-MORULAR VARIANT PAPILLARY THYROID CARCINOMA (CMV-PTC): "
            "      PATHOGNOMONIC for FAP/APC germline; "
            "      Histology: cribriform growth pattern + morule formation (morules = whorls of spindle cells); "
            "      Nuclear β-catenin accumulation: IHC positive (unlike sporadic PTC); "
            "      Predominantly females (90% female); young age (often 20-40yr); "
            "      Multi-focal; bilaterality common; "
            "      Usually lower-risk than classic PTC despite distinctive histology; "
            "      IF CRYPTIC FAP: CMV-PTC may be FIRST presentation of FAP before colonic symptoms; "
            "        → Send ALL CMV-PTC patients for APC germline testing + colonoscopy; "
            "    CONVENTIONAL PTC/FTC also seen in FAP (3-5x RR above population); "
            "    Thyroid Ca overall: 1-2% FAP — predominantly females; "
            "  EXTRACOLONIC MANIFESTATIONS (Gardner Syndrome = FAP + extracolonic): "
            "    Desmoid tumours: 15-20% FAP; mutation codon ≥1310 highest desmoid risk; "
            "      Intra-abdominal desmoid: lethal complication (SMA encasement); "
            "    Osteomas (jaw/skull/long bones): PATHOGNOMONIC; "
            "    Epidermoid cysts; congenital hypertrophy retinal pigment epithelium (CHRPE); "
            "    Upper GI polyps: duodenal/ampullary adenoma 80-90% FAP — Spigelman staging; "
            "    Hepatoblastoma: childhood — 800x RR (rare absolute); "
            "    Medulloblastoma: Turcot Syndrome (APC variant) — cerebellar medulloblastoma; "
            "THYROID SURVEILLANCE IN FAP: "
            "  Annual thyroid ultrasound: all FAP patients (including males) from diagnosis; "
            "  If FAP + thyroid nodule → FNA + β-catenin IHC + APC sequencing; "
            "COLONIC MANAGEMENT: "
            "  Total proctocolectomy + IPAA (ileal pouch anal anastomosis): preferred in classic FAP; "
            "  Sulindac + celecoxib: reduce polyp burden — NOT substitute for colectomy; "
            "  EGD surveillance: every 1-3yr for duodenal/ampullary polyps (Spigelman staging)"
        ),
        "inheritance": (
            "AD LOF 5q22.2 — APC. Classic FAP prevalence ~1:8,000-1:10,000. "
            "De novo APC: ~25-30% classic FAP (no family history). "
            "Mutation codon position predicts phenotype: "
            "  Codon <157 or >1595: AFAP (attenuated, 10-99 polyps); "
            "  Codon 1250-1464 (MCR): profuse polyposis + highest desmoid risk; "
            "  Codon ≥1310: highest desmoid tumour risk. "
            "Mosaic APC: ~20% APC-negative classic FAP — deep sequencing or tumour APC. "
            "MUTYH biallelic: MYH-Associated Polyposis (MAP) — AR, MUTYH not APC."
        ),
        "surveillance_key": "annual thyroid US all FAP patients; CMV-PTC histology PATHOGNOMONIC for FAP → APC germline test; prophylactic colectomy for classic FAP before age 20; EGD surveillance duodenal polyps Spigelman",
        "pathognomonic": "cribriform-morular variant PTC PATHOGNOMONIC for FAP/APC germline — nuclear β-catenin IHC positive; osteomas PATHOGNOMONIC Gardner; CHRPE PATHOGNOMONIC; CMV-PTC may be FIRST FAP presentation before colonic disease",
    },
    {
        "gene": "PRKAR1A",
        "protein": (
            "PRKAR1A -- 17q24.2 Autosomal-Dominant-LOF -- 381aa -- "
            "PKA-Regulatory-Subunit-R1alpha-cAMP-Sensor-Inhibitor-"
            "Carney-Complex-Follicular-Thyroid-Ca-75pct-DOMINANT-Endocrine-"
            "Cardiac-Myxoma-30pct-LIFE-THREATENING-Annual-Echo-MANDATORY-"
            "Spotty-Pigmentation-PATHOGNOMONIC-PPNAD-OMIM-188830"
        ),
        "locus": "17q24.2",
        "protein_size": (
            "381 aa / 17q24.2 PRKAR1A encodes Protein Kinase A regulatory subunit type I-α (PKA-R1α): "
            "STRUCTURE: "
            "  N-terminal dimerisation/docking domain (D/D); "
            "  Inhibitor site (pseudo-substrate region — R1α inhibits PKA catalytic subunit); "
            "  Two tandem cAMP-binding domains (CBD-A and CBD-B); "
            "  Functional mechanism: "
            "    Basal state: R1α (or R2α/R1β/R2β) tetramers with two catalytic (C) subunits → inactive; "
            "    cAMP binding → R subunits conformational change → release of two active C subunits; "
            "    Active C → CREB phosphorylation → proliferation/steroidogenesis/melanogenesis; "
            "  PRKAR1A LOF → unrestrained PKA catalytic activity → elevated cAMP signalling → "
            "    Paradoxical cortisol production (PPNAD) despite low ACTH; "
            "    Thyroid Ca from follicular cells with cAMP-driven proliferation; "
            "CARNEY COMPLEX — THYROID AND ENDOCRINE: "
            "  THYROID CANCER: "
            "    Follicular thyroid Ca: ~75% lifetime in Carney Complex — dominant endocrine Ca; "
            "    Also: follicular adenoma (multiple), Hürthle cell Ca; "
            "    Multi-focal thyroid involvement common; "
            "    Annual thyroid US from age of diagnosis / puberty; "
            "  CARNEY COMPLEX COMPONENTS: "
            "    1. SPOTTY SKIN PIGMENTATION — PATHOGNOMONIC: "
            "       Lentigines (dark spots) on face, lips, conjunctiva, genitalia; "
            "       Appear in childhood; diagnostic on clinical examination; "
            "    2. CARDIAC MYXOMA (30% Carney Complex): "
            "       LEFT ATRIUM most common (90% sporadic myxomas are LA); "
            "       Carney: ANY chamber — ventricular, biatrial, multi-chamber; "
            "       LIFE-THREATENING: "
            "         Emboli → stroke; "
            "         Valve obstruction → haemodynamic collapse; "
            "         Sudden cardiac death; "
            "       ANNUAL ECHOCARDIOGRAPHY MANDATORY in ALL PRKAR1A carriers from childhood; "
            "       Urgent surgical resection if myxoma detected (NOT watchful waiting); "
            "    3. PRIMARY PIGMENTED NODULAR ADRENOCORTICAL DISEASE (PPNAD): "
            "       Bilateral adrenal micronodular cortical hyperplasia; "
            "       PARADOXICAL LIDDLE TEST: cortisol RISES with dexamethasone (opposite of Cushing); "
            "       ACTH-INDEPENDENT hypercortisolism (low ACTH, high cortisol); "
            "       Bilateral adrenalectomy for PPNAD Cushing syndrome; "
            "    4. LARGE CELL CALCIFYING SERTOLI CELL TUMOUR (LCCSCT): "
            "       Bilateral testicular tumours — calcifications on US in ~75% males; "
            "       Benign usually; precocious puberty if hormone-secreting; "
            "    5. ACROMEGALY: pituitary somatotroph adenoma ~10% Carney Complex; "
            "    6. PSAMMOMATOUS MELANOTIC SCHWANNOMA: benign but potentially malignant; "
            "       Spinal cord / paravertebral; calcifications (psammoma bodies) on MRI; "
            "    7. BREAST MYXOMA / ductal adenoma; "
            "SURVEILLANCE PROTOCOL: "
            "  Annual echo from childhood (myxoma surveillance — MANDATORY); "
            "  Annual thyroid US; "
            "  Annual overnight UFC (cortisol) or low-dose dexamethasone test (PPNAD); "
            "  Testicular US annually (males): LCCSCT calcium; "
            "  Annual GH + IGF-1 (pituitary acromegaly)"
        ),
        "inheritance": (
            "AD LOF 17q24.2 — PRKAR1A. Carney Complex prevalence ~1:1,000,000 (rare). "
            "PRKAR1A: ~70% Carney Complex families; 30% mutation-negative (other genes, 2p16). "
            "De novo PRKAR1A: ~30% cases. "
            "Mosaic PRKAR1A: some cases with milder phenotype. "
            "Cascade testing: all first-degree relatives; cardiac screening from diagnosis mandatory."
        ),
        "surveillance_key": "annual echocardiography MANDATORY (cardiac myxoma — life-threatening); annual thyroid US (follicular Ca 75%); spotty pigmentation PATHOGNOMONIC; PPNAD paradoxical Liddle test; annual UFC/dexamethasone suppression",
        "pathognomonic": "spotty lentigines pigmentation PATHOGNOMONIC Carney Complex; cardiac myxoma multi-chamber (NOT just LA) PATHOGNOMONIC; PPNAD paradoxical Liddle cortisol RISE PATHOGNOMONIC; LCCSCT bilateral testicular calcifications",
    },
    {
        "gene": "DICER1",
        "protein": (
            "DICER1 -- 14q32.13 Autosomal-Dominant-LOF -- 1922aa -- "
            "RNase-III-Endoribonuclease-miRNA-biogenesis-Engine-"
            "DICER1-Syndrome-FAPOL-Multinodular-Goiter-Differentiated-Thyroid-Ca-"
            "Pleuropulmonary-Blastoma-PATHOGNOMONIC-Infancy-PPB-Type-I-OMIM-606241"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "1922 aa / 14q32.13 DICER1 encodes Double-stranded RNA-specific endoribonuclease (DICER1): "
            "STRUCTURE: "
            "  N-terminal helicase domain; "
            "  DUF283 domain; "
            "  Platform-PAZ domain (recognises 3'-end of pre-miRNA 2-nt overhang); "
            "  Two RNase III domains (RNase IIIa + RNase IIIb — cleave dsRNA); "
            "  dsRBD (double-stranded RNA binding domain); "
            "  Mechanism: "
            "    Cleaves pre-miRNA: "
            "    RNase IIIa: cuts guide strand (~22nt product); "
            "    RNase IIIb: cuts passenger strand; "
            "    Biallelic DICER1: loss of ALL miRNA processing (embryonic lethal); "
            "    Monoallelic germline LOF + somatic 'hot-spot' missense in RNase IIIb: "
            "      Hot-spot missenses: E1705K, D1709N, G1748D, E1813K (metal ion chelating residues); "
            "      Partial miRNA processing: selective loss of 5p miRNAs; "
            "      5p miRNA loss → de-repression of oncogenic targets (KRAS, HMGA2); "
            "DICER1 SYNDROME (FAPOL) — THYROID: "
            "  MULTINODULAR GOITER (MNG): most common DICER1 thyroid manifestation; "
            "    MNG prevalence: ~75% female carriers by adulthood; ~17% male carriers; "
            "    Female predominance: thyroid 10-fold higher disease burden than males; "
            "    Multiple nodules; can be large; symptomatic compression; "
            "    Surgery if symptomatic or suspicious nodule; "
            "  DIFFERENTIATED THYROID CARCINOMA (DTC): "
            "    Papillary thyroid Ca: ~1-2% DICER1 carriers (above population); "
            "    Follicular thyroid Ca: also reported; "
            "    Well-differentiated; standard surgical management; RAI if appropriate; "
            "    Often in context of MNG — surveillance essential; "
            "  PLEUROPULMONARY BLASTOMA (PPB) — PATHOGNOMONIC DICER1 INFANT TUMOUR: "
            "    Type I PPB (cystic): PATHOGNOMONIC for DICER1 germline; infancy (<2yr); "
            "    Type II/III PPB (solid): more aggressive, older children; "
            "    ALL PPB → DICER1 germline testing (and family); "
            "    Chest CT every 3 months first 3yr: surveillance all DICER1 children; "
            "  OTHER DICER1 TUMOUR SPECTRUM: "
            "    Cystic nephroma (infant): PATHOGNOMONIC; "
            "    Sertoli-Leydig cell tumour (SLCT) ovary: young women; virilisation; "
            "      SLCT + DICER1 hot-spot in RNase IIIb; "
            "    Embryonal rhabdomyosarcoma (ERMS) cervix: PATHOGNOMONIC young women; "
            "    Pineoblastoma; nasal chondromesenchymal hamartoma; "
            "    Ciliary body medulloepithelioma; "
            "THYROID SURVEILLANCE: "
            "  Annual thyroid US from age of DICER1 diagnosis (child or adult); "
            "  MNG: watchful waiting if not symptomatic; FNA for suspicious features; "
            "  DTC: standard total thyroidectomy + RAI staging"
        ),
        "inheritance": (
            "AD LOF 14q32.13 — DICER1. DICER1 syndrome prevalence unknown (~1:10,000 estimated). "
            "De novo DICER1: ~10-15% families. "
            "Penetrance: incomplete; female > male for thyroid/goiter; male risk primarily PPB in infancy. "
            "Second somatic hit: hot-spot missense in RNase IIIb domain required for tumour formation. "
            "ALL pleuropulmonary blastoma → DICER1 germline testing mandatory. "
            "Cascade testing: parents + siblings of PPB child."
        ),
        "surveillance_key": "chest CT every 3 months first 3yr all DICER1 children (PPB surveillance); annual thyroid US; PPB Type I PATHOGNOMONIC → DICER1 germline test; SLCT ovary young women → DICER1 test; all family members PPB child",
        "pathognomonic": "PPB Type I (cystic) in infant PATHOGNOMONIC DICER1 germline; cystic nephroma infant PATHOGNOMONIC; ERMS cervix young women PATHOGNOMONIC DICER1; SLCT ovary with virilisation → DICER1 hot-spot test",
    },
    {
        "gene": "CDC73",
        "protein": (
            "CDC73 -- 1q31.2 Autosomal-Dominant-LOF -- 531aa -- "
            "Cell-Division-Cycle-73-Parafibromin-PAF1-Complex-Histone-Methylation-"
            "HPT-JT-Hyperparathyroidism-Jaw-Tumour-Syndrome-"
            "Parathyroid-Carcinoma-PATHOGNOMONIC-10-15pct-HPT-JT-"
            "Oxyphilic-Follicular-Thyroid-Ca-OMIM-607393"
        ),
        "locus": "1q31.2",
        "protein_size": (
            "531 aa / 1q31.2 CDC73 encodes parafibromin (CDC73): "
            "STRUCTURE: "
            "  N-terminal domain (unique; parathyroid cell function); "
            "  Scp1 homology domain (coiled-coil); "
            "  Yeast Cdc73-homology domain; "
            "  PAF1 Complex (PAF1C) component: "
            "    Parafibromin/CDC73 is a subunit of the mammalian PAF1C (RNA PolII-associated); "
            "    PAF1C regulates: "
            "      H3K4me3 histone methylation (active transcription mark) via SET1/COMPASS; "
            "      H2B ubiquitylation (transcription elongation); "
            "      3' mRNA processing; "
            "  Tumour suppressor functions: "
            "    Represses cyclin D1 promoter; "
            "    Nuclear parafibromin → growth inhibition; "
            "    CDC73 LOF → cyclin D1 overexpression → cell cycle entry; "
            "  IHC: nuclear parafibromin LOSS = surrogate marker of CDC73 inactivation in tumour; "
            "HPT-JT SYNDROME AND PARATHYROID CARCINOMA: "
            "  HYPERPARATHYROIDISM-JAW TUMOUR SYNDROME (HPT-JT): "
            "    Primary hyperparathyroidism (PHPT): 80-90% HPT-JT; often single parathyroid adenoma; "
            "    PARATHYROID CARCINOMA: 10-15% HPT-JT → PATHOGNOMONIC context; "
            "      Sporadic parathyroid carcinoma: ~50% have somatic CDC73 biallelic loss; "
            "      Germline CDC73 → parathyroid carcinoma: HIGHEST risk hereditary PHPT syndrome; "
            "      Presentation: very high serum calcium (>3.5 mmol/L); very high PTH (>3x ULN); "
            "        Palpable neck mass; CaSR-resistant hypercalcaemia; "
            "      Management: en bloc resection (ipsilateral thyroid lobe + soft tissue); "
            "        NEVER disrupt capsule; seeding causes carcinomatosis; "
            "    JAW TUMOURS (ossifying fibromas): "
            "      Mandible + maxilla ossifying fibromas PATHOGNOMONIC for HPT-JT; "
            "      NOT simple dentigenous cysts; OPG + jaw MRI for diagnosis; "
            "      Ossifying fibromas ≠ brown tumours (brown tumours = hyperparathyroid bone disease, not HPT-JT); "
            "  THYROID CANCER IN CDC73: "
            "    Oxyphilic (Hürthle cell) follicular thyroid adenoma/carcinoma: "
            "      Reported in HPT-JT — moderately elevated RR; "
            "      Parafibromin IHC loss on thyroid Ca: validates CDC73 involvement; "
            "    Follicular thyroid Ca (non-Hürthle): also reported; "
            "    Thyroid surveillance: annual US in HPT-JT; "
            "  UTERINE TUMOURS: "
            "    Uterine (endometrial/cervical) tumours: 50% HPT-JT females — variable; "
            "  CDC73 RELATED DISORDERS: "
            "    Isolated familial hyperparathyroidism (FIHPT): CDC73 germline without full HPT-JT; "
            "    Parathyroid adenoma: CDC73 germline carriers — higher risk than population; "
            "DIAGNOSIS AND IHC: "
            "  Parafibromin IHC: "
            "    Normal parathyroid tissue: nuclear staining present; "
            "    Parathyroid carcinoma: nuclear parafibromin ABSENT in 70-80% → strong diagnostic support; "
            "    Atypical adenoma: partial/absent loss; "
            "  Serum calcium: severe hypercalcaemia + high PTH + palpable neck → assume carcinoma until proven"
        ),
        "inheritance": (
            "AD LOF 1q31.2 — CDC73. HPT-JT prevalence very rare (<1:1,000,000). "
            "FIHPT + CDC73: more common presentation than full HPT-JT. "
            "De novo CDC73: rare; usually familial. "
            "Somatic CDC73 biallelic: ~50% sporadic parathyroid carcinomas — IHC screening all parathyroid Ca. "
            "Cascade testing: first-degree relatives all HPT-JT index cases. "
            "Multiple recurrence after 'adenoma' resection → CDC73 germline test."
        ),
        "surveillance_key": "parathyroid carcinoma PATHOGNOMONIC HPT-JT context; en bloc resection — never disrupt capsule (seeding); annual calcium/PTH/neck US; jaw ossifying fibromas PATHOGNOMONIC; parafibromin IHC loss in parathyroid Ca",
        "pathognomonic": "jaw ossifying fibromas (mandible/maxilla) PATHOGNOMONIC HPT-JT; parathyroid carcinoma with very high Ca >3.5mmol/L + PTH >3x ULN + palpable mass; parafibromin nuclear IHC loss PATHOGNOMONIC parathyroid carcinoma",
    },
    {
        "gene": "VHL",
        "protein": (
            "VHL -- 3p25.3 Autosomal-Dominant-LOF -- 213aa -- "
            "Von-Hippel-Lindau-Protein-HIF-α-E3-Ubiquitin-Ligase-Adaptor-"
            "VHL-Disease-Hemangioblastoma-CNS-Retinal-PATHOGNOMONIC-"
            "ccRCC-Pheochromocytoma-Clear-Cell-Papillary-Thyroid-Ca-OMIM-608537"
        ),
        "locus": "3p25.3",
        "protein_size": (
            "213 aa / 3p25.3 VHL encodes Von Hippel-Lindau protein (pVHL): "
            "STRUCTURE: "
            "  α-domain (binds elongin B/C of E3 ubiquitin ligase complex); "
            "  β-domain (recognition domain — binds HIF-α hydroxylated LXXLAP motifs); "
            "  pVHL function: "
            "    Normoxia: PHD enzymes (PHD1/2/3) hydroxylate HIF-1α and HIF-2α at Pro402/Pro564 (HIF-1α); "
            "    pVHL E3 complex: recognises hydroxy-HIF-α → ubiquitylates → proteasomal degradation; "
            "    Hypoxia: PHDs inactive → HIF-α not hydroxylated → escapes pVHL → HIF accumulates; "
            "    HIF-1α/2α transcription targets: VEGF, EPO, GLUT1, PDK1, CAIX, LDHA, Oct4/Nanog (HIF-2α); "
            "  VHL LOF → constitutive HIF → pseudo-hypoxic state → VEGF-driven tumours; "
            "VHL DISEASE — THYROID AND FULL SPECTRUM: "
            "  VHL DISEASE CLASSIFICATION: "
            "    Type 1: VHL truncating/missense → low pheo risk; high RCC/hemangioblastoma risk; "
            "    Type 2A: VHL missense → pheo + low RCC risk; "
            "    Type 2B: VHL missense → pheo + HIGH RCC risk; "
            "    Type 2C: VHL missense → pheo ONLY; "
            "  THYROID DISEASE IN VHL: "
            "    Clear-cell follicular thyroid Ca (Type 2C VHL families especially): "
            "      Rare but documented thyroid manifestation; "
            "      Clear cells: glycogen-rich cytoplasm (resemble RCC clear cells); "
            "      HIF target CAIX positive on IHC; "
            "    Simple thyroid cysts: more common incidental finding in VHL; "
            "    Annual thyroid US in VHL: reasonable screening in affected families; "
            "  VHL CORE TUMOUR SPECTRUM: "
            "    HEMANGIOBLASTOMA — PATHOGNOMONIC: "
            "      CNS hemangioblastomas (cerebellum, brainstem, spinal cord): most common VHL lesion; "
            "      Retinal hemangioblastomas (retinal angiomas): earliest VHL manifestation; "
            "        Annual ophthalmologic exam from age 1yr; laser photocoagulation if symptomatic; "
            "      Any hemangioblastoma → VHL testing; "
            "    CLEAR-CELL RENAL CELL CARCINOMA (ccRCC): "
            "      50-70% VHL patients by age 60; bilateral and multifocal; "
            "      Surveillance: MRI abdomen every 1-2yr; surgery when ≥3cm (thermal ablation option); "
            "      Targeted therapy: sunitinib/pazopanib (VEGFR TKI); belzutifan (HIF-2α inhibitor) — FDA 2021 VHL-RCC; "
            "    PHEOCHROMOCYTOMA: "
            "      10-20% VHL (Type 2 genotypes); bilateral 50%; usually benign but functional; "
            "      Annual biochemical screen: plasma/urine metanephrines from age 8; "
            "    PANCREATIC LESIONS: "
            "      Pancreatic cysts: ~70% VHL; usually benign; "
            "      Pancreatic NET: ~10-17% VHL; somatostatin-receptor imaging; "
            "        Surgery if >3cm or growth; "
            "    ENDOLYMPHATIC SAC TUMOUR (ELST): "
            "      Inner ear low-grade adenocarcinoma; hearing loss; "
            "      MRI temporal bones every 5yr; "
            "TARGETED THERAPY: "
            "  Belzutifan (Welireg, HIF-2α inhibitor): FDA 2021 — VHL-related tumours; "
            "    Renal, CNS hemangioblastoma, pancreatic NET in VHL disease; "
            "    First disease-modifying therapy for VHL; "
            "  Sunitinib, pazopanib, cabozantinib: ccRCC if belzutifan not suitable; "
            "SURVEILLANCE PROTOCOL: "
            "  Annual ophthalmologic exam (retinal angiomas) from age 1; "
            "  Annual biochemical pheo screen from age 8; "
            "  MRI brain + total spine every 1-2yr from age 15; "
            "  MRI abdomen every 1-2yr (RCC + pheo + pancreas) from age 15"
        ),
        "inheritance": (
            "AD LOF 3p25.3 — VHL. VHL disease prevalence ~1:36,000-1:40,000. "
            "De novo VHL: ~20% cases. "
            "Large deletions: ~20% VHL pathogenic — MLPA mandatory. "
            "Genotype-phenotype: missense mutations (Type 2) → pheo risk; truncating (Type 1) → low pheo, high RCC. "
            "Belzutifan: first targeted VHL disease therapy (HIF-2α inhibitor) FDA 2021. "
            "Cascade testing all first-degree relatives; retinal screening from age 1."
        ),
        "surveillance_key": "annual retinal exam from age 1 (retinal hemangioblastoma earliest VHL lesion); MRI brain/spine every 1-2yr from age 15; MRI abdomen every 1-2yr (RCC + pheo + pancreas); belzutifan HIF-2α inhibitor FDA 2021",
        "pathognomonic": "hemangioblastoma CNS or retinal PATHOGNOMONIC VHL disease; bilateral ccRCC — VHL until proven otherwise; endolymphatic sac tumour PATHOGNOMONIC VHL; clear-cell follicular thyroid Ca with CAIX+ IHC VHL association",
    },
]


def _gene_stats(seed: int, gene_config: dict) -> dict:
    """Generate per-gene statistics for one gene using a fixed seed."""
    rng = random.Random(seed)
    gene = gene_config["gene"]

    base = {
        "RET":     {"mtc_lifetime": 95, "pheo": 44, "hyperparathyroidism": 24, "bilateral_pheo": 48, "tkiselpercatinib": 68, "prophylactic_thyroidectomy": 88},
        "TP53":    {"anaplastic_tc": 12, "ptc": 8, "radiation_avoid": 100, "wbmri_surveillance": 82, "breast_ca": 28, "sarcoma": 33},
        "PTEN":    {"thyroid_ca": 31, "breast_ca": 82, "endometrial_ca": 36, "macrocephaly": 96, "lhermitte_duclos": 8, "multinodular_goiter": 74},
        "APC":     {"crc": 97, "cmv_ptc": 15, "thyroid_ca": 4, "duodenal_adenoma": 88, "desmoid": 17, "prophylactic_colectomy": 89},
        "PRKAR1A": {"follicular_tc": 73, "cardiac_myxoma": 29, "ppnad": 31, "acromegaly": 10, "spotty_pigmentation": 96, "lccsct": 22},
        "DICER1":  {"mng": 72, "dtc": 8, "ppb": 20, "slct_ovary": 18, "cystic_nephroma": 12, "erms_cervix": 6},
        "CDC73":   {"phpt": 86, "parathyroid_carcinoma": 12, "jaw_tumour": 45, "thyroid_ca": 9, "uterine_tumour": 47, "parafibromin_loss": 76},
        "VHL":     {"ccrcc": 62, "hemangioblastoma": 72, "pheo": 18, "retinal_angioma": 58, "pancreatic_cyst": 68, "thyroid_ca": 5},
    }.get(gene, {})

    n = 40
    age_mean = {
        "RET": 34, "TP53": 38, "PTEN": 44, "APC": 42,
        "PRKAR1A": 38, "DICER1": 36, "CDC73": 46, "VHL": 40,
    }.get(gene, 42)

    stats = {
        "gene": gene,
        "n": n,
        "seed": seed,
        "mean_age_diagnosis": round(age_mean + rng.gauss(0, 3), 1),
        "female_pct": round(rng.uniform(
            55 if gene in ("PTEN", "DICER1", "PRKAR1A") else 45,
            70 if gene in ("PTEN", "DICER1", "PRKAR1A") else 60
        ), 1),
    }

    for feature, base_rate in base.items():
        rate = max(0, min(100, base_rate + rng.gauss(0, 4)))
        stats[f"{feature}_pct"] = round(rate, 1)

    stats["genetic_testing_positive_pct"] = round(rng.uniform(88, 99), 1)
    stats["surveillance_adherent_pct"] = round(rng.uniform(60, 82), 1)
    stats["family_history_positive_pct"] = round(rng.uniform(30, 68), 1)
    stats["de_novo_pct"] = round(rng.uniform(5, 30), 1)
    return stats


def generate_overview() -> dict:
    return {
        "atlas":          "Hereditary-Thyroid-Cancer-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary Thyroid Cancer Atlas "
            "(RET-TP53-PTEN-APC-PRKAR1A-DICER1-CDC73-VHL)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "RET": (
                "AD GOF 10q11.21 (RET receptor tyrosine kinase; 1114aa; MEN2A/MEN2B/FMTC; "
                "MTC 95%+ — ALL RET carriers require prophylactic thyroidectomy; "
                "MEN2B M918T → thyroidectomy ≤6 months — MOST URGENT hereditary cancer surgery; "
                "Category D/C/B risk stratification; selpercatinib/vandetanib/cabozantinib FDA)"
            ),
            "TP53": (
                "AD LOF 17p13.1 (p53 guardian; 393aa; Li-Fraumeni Syndrome; "
                "anaplastic thyroid Ca — TP53 somatic 73-88% ATC; germline LFS rare but aggressive; "
                "AVOID ALL RADIATION ABSOLUTELY — radiation-field sarcoma documented; "
                "WBMRI Toronto protocol annually)"
            ),
            "PTEN": (
                "AD LOF 10q23.31 (PI3K-Akt-mTOR brake; 403aa; Cowden/PHTS; "
                "follicular thyroid Ca 25-38%; macrocephaly PATHOGNOMONIC (≥97th centile OFC); "
                "Lhermitte-Duclos PATHOGNOMONIC — test PTEN immediately; "
                "annual thyroid US from age 7-10yr)"
            ),
            "APC": (
                "AD LOF 5q22.2 (WNT β-catenin gatekeeper; 2843aa; FAP/Gardner; "
                "cribriform-morular variant PTC PATHOGNOMONIC for FAP — nuclear β-catenin IHC+; "
                "CMV-PTC may be FIRST FAP presentation; annual thyroid US all FAP patients; "
                "CRC 100% if untreated — prophylactic colectomy mandatory)"
            ),
            "PRKAR1A": (
                "AD LOF 17q24.2 (PKA-R1α cAMP brake; 381aa; Carney Complex; "
                "follicular thyroid Ca 75% — dominant endocrine Ca in Carney; "
                "cardiac myxoma 30% — LIFE-THREATENING — annual echo MANDATORY from childhood; "
                "spotty skin pigmentation PATHOGNOMONIC; PPNAD paradoxical Liddle)"
            ),
            "DICER1": (
                "AD LOF 14q32.13 (RNase III miRNA processor; 1922aa; DICER1 Syndrome/FAPOL; "
                "multinodular goiter ~75% female carriers; DTC elevated risk; "
                "PPB Type I in infancy PATHOGNOMONIC — chest CT every 3mo first 3yr; "
                "SLCT ovary + ERMS cervix PATHOGNOMONIC in young women)"
            ),
            "CDC73": (
                "AD LOF 1q31.2 (parafibromin PAF1C; 531aa; HPT-JT Syndrome; "
                "parathyroid carcinoma 10-15% HPT-JT — PATHOGNOMONIC context; "
                "jaw ossifying fibromas PATHOGNOMONIC; en bloc resection — NEVER disrupt capsule; "
                "parafibromin nuclear IHC loss PATHOGNOMONIC parathyroid carcinoma)"
            ),
            "VHL": (
                "AD LOF 3p25.3 (HIF-α E3 ligase adaptor; 213aa; VHL Disease; "
                "hemangioblastoma CNS/retinal PATHOGNOMONIC; ccRCC 50-70%; pheo 10-20% Type 2; "
                "belzutifan (HIF-2α inhibitor) FDA 2021 — first VHL disease therapy; "
                "annual retinal exam from age 1)"
            ),
        },
        "key_clinical_rules": [
            "RET M918T (MEN2B): prophylactic thyroidectomy ≤6 months — MOST URGENT prophylactic cancer surgery in oncogenetics; do not delay",
            "RET: ATA risk stratification Category D/C/B MANDATORY before counselling — timing of thyroidectomy differs by genotype",
            "RET: calcitonin + CEA are primary MTC surveillance markers; calcitonin doubling time <6 months = poor prognosis",
            "TP53 germline (LFS): AVOID ALL RADIATION — therapeutic radiation in LFS carriers causes radiation-field sarcoma; use systemic alternatives",
            "PTEN/PHTS: Lhermitte-Duclos disease (cerebellar gangliocytoma) = PATHOGNOMONIC for PHTS — test PTEN immediately on ANY patient",
            "PTEN: macrocephaly (≥97th centile OFC) is most consistent clinical sign — always measure head circumference in thyroid Ca patients",
            "APC: cribriform-morular variant PTC (CMV-PTC) is PATHOGNOMONIC for FAP — perform nuclear β-catenin IHC on ALL PTC histology samples to screen",
            "APC: CMV-PTC may be the FIRST clinical presentation of FAP before colorectal symptoms — colonoscopy mandatory on all CMV-PTC patients",
            "PRKAR1A/Carney Complex: cardiac myxoma causes sudden death — annual echocardiography is NON-NEGOTIABLE from childhood in all PRKAR1A carriers",
            "PRKAR1A: PPNAD causes ACTH-independent Cushing; paradoxical RISE in cortisol on Liddle test (dexamethasone) — PATHOGNOMONIC",
            "DICER1: PPB Type I cystic lung lesion in infant = PATHOGNOMONIC DICER1 — test family; chest CT surveillance every 3 months first 3yr",
            "CDC73/HPT-JT: parathyroid carcinoma Ca >3.5mmol/L + PTH >3x ULN + palpable mass → assume carcinoma; en bloc resection mandatory (capsule disruption = seeding)",
            "CDC73: jaw ossifying fibromas are PATHOGNOMONIC HPT-JT — distinguish from simple dental cysts by OPG + MRI jaw",
            "VHL: belzutifan (HIF-2α inhibitor) FDA 2021 is first disease-modifying therapy for VHL — avoid unnecessary nephrectomy for <3cm lesions under surveillance",
            "VHL: retinal hemangioblastomas are earliest manifestation — annual ophthalmologic exam from age 1yr mandatory",
            "ALL MTC: calcitonin + RET germline testing at diagnosis (25% hereditary MTC) — gene panel includes RET + other syndromic genes",
        ],
        "gene_panel_note": (
            "Hereditary Thyroid Cancer Germline Panel (clinical 2024): "
            "RET KINASE (MEN2A/MEN2B/FMTC — MTC dominant): "
            "  RET: 95%+ MTC lifetime; ATA risk Category D/C/B; thyroidectomy 6mo–5yr; selpercatinib; "
            "TP53 PATHWAY (anaplastic/aggressive): "
            "  TP53 germline: rare anaplastic TC; Li-Fraumeni; AVOID RADIATION; WBMRI Toronto; "
            "COWDEN/PI3K PATHWAY (follicular thyroid Ca): "
            "  PTEN: 25-38% follicular TC; macrocephaly PATHOGNOMONIC; Lhermitte-Duclos PATHOGNOMONIC; "
            "WNT PATHWAY / FAP (CMV-PTC): "
            "  APC: cribriform-morular PTC PATHOGNOMONIC FAP; annual US all FAP; 100% CRC prophylactic colectomy; "
            "cAMP/PKA PATHWAY (Carney Complex): "
            "  PRKAR1A: 75% follicular TC; cardiac myxoma annual echo mandatory; spotty pigmentation; PPNAD; "
            "MIRNA PROCESSING (DICER1 Syndrome): "
            "  DICER1: MNG 75% females; DTC elevated risk; PPB Type I PATHOGNOMONIC; chest CT 3-monthly infant; "
            "PARAFIBROMIN/PAF1C (HPT-JT — parathyroid Ca): "
            "  CDC73: parathyroid carcinoma PATHOGNOMONIC HPT-JT; jaw ossifying fibromas; oxyphilic follicular TC; "
            "HIF PATHWAY / VHL: "
            "  VHL: hemangioblastoma PATHOGNOMONIC; ccRCC 50-70%; belzutifan HIF-2α FDA 2021; clear-cell TC; "
            "UNIVERSAL TESTING: "
            "  ALL MTC at diagnosis: RET germline testing (25% hereditary — somatic RET also targetable); "
            "  ALL CMV-PTC: APC germline + colonoscopy; "
            "  ALL PPB infant: DICER1 germline; "
            "  ALL parathyroid carcinoma: CDC73 germline + somatic parafibromin IHC"
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
        "atlas":          "Hereditary-Thyroid-Cancer-Atlas",
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "n_genes":        len(genes_data),
        "total_patients": 320,
        "genes":          genes_data,
    }


def generate_definitions() -> dict:
    definitions = [
        {
            "term": "RET-ATA-Risk-Stratification-Thyroidectomy-Timing",
            "definition": (
                "RET Genotype-Based ATA 2015 Risk Stratification for Prophylactic Thyroidectomy: "
                "CATEGORY D — HIGHEST RISK (thyroidectomy ≤6 months of life): "
                "  Mutation: M918T (exon 16) — MEN2B; "
                "  Rationale: MTC biochemically positive within first year of life; "
                "    Lymph node metastasis before 6 months documented; "
                "    Delay beyond 6 months risks metastatic disease in MEN2B; "
                "  Management: "
                "    Thyroidectomy ≤6 months — MOST URGENT hereditary cancer surgery in medicine; "
                "    Annual plasma metanephrines from age 6 months (pheo); "
                "    Genetic testing at birth if M918T parent; "
                "    Phenotype: mucosal neuromas, marfanoid habitus — inspect at birth; "
                "CATEGORY C — HIGH RISK (thyroidectomy ≤5 years): "
                "  Mutations: C634F/G/R/S/W/Y (most common MEN2A codon), A883F, C609S/G/Y, C611S/G/Y/F/W, C618S/G/R, C620S/G/R/W; "
                "  Rationale: "
                "    C634: MTC in childhood; pheo 40-50% by age 40; PHPT 20-30%; "
                "    RET C634R — highest pheochromocytoma risk within Category C; "
                "  Management: "
                "    Thyroidectomy ≤5yr (before school age); "
                "    Annual calcitonin + CEA from age 3; "
                "    Annual plasma metanephrines from age 8; "
                "    Central neck dissection only if calcitonin elevated pre-op; "
                "CATEGORY B — MODERATE RISK (individualised, typically 5-10yr): "
                "  Mutations: E768D, L790F, V804L, V804M, S891A, R912P, Y791F, C630R; "
                "  Rationale: MTC onset generally adult; penetrance lower than C/D; "
                "  Management: "
                "    Thyroidectomy can be deferred if calcitonin normal and patient compliant with surveillance; "
                "    OR prophylactic thyroidectomy 5-10yr depending on family history severity; "
                "    Annual calcitonin from diagnosis; "
                "CALCITONIN INTERPRETATION: "
                "  Normal calcitonin: <5 pg/mL (male ≤8, female ≤4 most labs); "
                "  Borderline (5-100): repeat + pentagastrin-stimulated test; "
                "  Elevated (>100): MTC likely; neck US + CT/MRI before surgery; "
                "SURGICAL CONSIDERATIONS: "
                "  Central neck dissection (level VI): if any pre-op calcitonin elevation; "
                "  Lateral neck dissection: calcitonin >200 (lateral node involvement likely); "
                "  Post-op calcitonin undetectable (<2): biochemical cure achieved; "
                "  Post-op calcitonin detectable: residual/metastatic disease → imaging + selpercatinib"
            ),
        },
        {
            "term": "APC-Cribriform-Morular-Variant-PTC-FAP-Protocol",
            "definition": (
                "Cribriform-Morular Variant PTC (CMV-PTC) — FAP Diagnostic Protocol: "
                "WHAT IS CMV-PTC: "
                "  A distinct PTC variant with: "
                "    Cribriform architecture (arched cellular arrays without colloid); "
                "    Morules (whorls of non-keratinising spindle cells — same as colonic FAP polyp morules); "
                "    Nuclear β-catenin accumulation (nuclear translocation of β-catenin on IHC); "
                "    Clear nuclei (PTC nuclear features); "
                "  Predominantly young women (90% female, age 20-40yr); "
                "  Multifocal; bilateral in 50-60%; "
                "  Usually good prognosis (lower-risk than classic PTC despite appearance); "
                "PATHOGNOMONIC ASSOCIATION WITH FAP: "
                "  CMV-PTC is PATHOGNOMONIC for FAP/APC germline mutation; "
                "  >95% CMV-PTC have underlying APC germline mutation; "
                "  CMV-PTC may be the FIRST presentation of FAP (before colonic symptoms): "
                "    Patient presents with thyroid nodule → FNA → CMV-PTC histology; "
                "    No known FAP, no polyps yet (FAP polyps emerge in teens/twenties); "
                "    → APC germline testing + colonoscopy MANDATORY; "
                "DIAGNOSTIC WORKUP: "
                "  Step 1: Any PTC → standard histology; if cribriform/morular features → IHC; "
                "  Step 2: Nuclear β-catenin IHC: "
                "    Positive (nuclear β-catenin): CMV-PTC confirmed → APC testing; "
                "    Normal (membranous only): sporadic PTC — no FAP association; "
                "  Step 3: APC germline sequencing + MLPA; "
                "  Step 4: Colonoscopy — even if APC positive, establish polyp burden; "
                "  Step 5: Cascade testing — family colonoscopy + APC; "
                "MANAGEMENT OF CMV-PTC IN FAP: "
                "  Thyroid surgery: total thyroidectomy preferred (multifocal); "
                "  Surveillance: annual thyroid US post-thyroidectomy (residual / de novo); "
                "  Colonic management: prophylactic colectomy as per FAP protocol; "
                "  Duodenal surveillance: EGD every 1-3yr (Spigelman staging); "
                "CMV-PTC VERSUS SPORADIC BRAF V600E PTC: "
                "  Sporadic PTC (BRAF V600E): monofocal, unilateral, nuclear β-catenin negative; "
                "  CMV-PTC: multifocal, nuclear β-catenin positive, BRAF V600E negative; "
                "  BRAF testing and β-catenin IHC together distinguish these reliably"
            ),
        },
        {
            "term": "PRKAR1A-Carney-Complex-Cardiac-Myxoma-Surveillance",
            "definition": (
                "Carney Complex Cardiac Myxoma — Life-Threatening Manifestation and Annual Echo Protocol: "
                "WHY CARDIAC MYXOMA IN CARNEY COMPLEX IS DIFFERENT FROM SPORADIC: "
                "  Sporadic myxoma: 90% left atrium; solitary; low recurrence after resection; "
                "  Carney Complex myxoma: "
                "    ANY chamber: left atrium, right atrium, ventricles, biatrial simultaneously; "
                "    Multifocal; recurrent after resection; "
                "    Occur in young patients (childhood to early adulthood); "
                "    NO safe 'watchful waiting' strategy; "
                "COMPLICATIONS: "
                "  1. SYSTEMIC EMBOLISATION: "
                "     Left-sided myxoma: fragments → cerebral emboli → stroke; "
                "     Other systemic emboli: coronary, renal, peripheral; "
                "  2. CARDIAC OBSTRUCTION: "
                "     Pedunculated myxoma: prolapse through mitral valve → acute obstruction; "
                "     Haemodynamic collapse; acute pulmonary oedema; "
                "     Can mimic mitral stenosis on auscultation (tumour plop); "
                "  3. SUDDEN CARDIAC DEATH: "
                "     Massive embolism; ventricular arrhythmia from contact; "
                "SURVEILLANCE PROTOCOL: "
                "  Annual transthoracic echocardiography (TTE): "
                "    Frequency: EVERY YEAR from childhood in ALL PRKAR1A carriers; "
                "    If TTE technically limited: transoesophageal echo (TOE); "
                "    MRI cardiac: alternative if echo windows poor (pectus, obesity); "
                "    Start: from age of genetic diagnosis (including children); "
                "MANAGEMENT WHEN MYXOMA DETECTED: "
                "  Urgent surgical resection: do NOT monitor a cardiac myxoma — operate promptly; "
                "  Surgical approach: "
                "    Cardiopulmonary bypass; wide excision including stalk and adjacent endocardium; "
                "    Inspect all four chambers (biatrial approach for Carney); "
                "  Post-resection surveillance: "
                "    Continue annual echo INDEFINITELY (recurrent myxoma in 20-25% Carney vs <5% sporadic); "
                "DISTINGUISHING FEATURES: "
                "  Carney myxoma: young age, multifocal, multi-chamber, recurrent → consider Carney; "
                "  Spotty pigmentation on skin exam → confirms Carney Complex; "
                "  PRKAR1A germline test mandatory in ANY patient with recurrent or multi-chamber myxoma"
            ),
        },
        {
            "term": "VHL-Belzutifan-HIF2alpha-Targeted-Therapy",
            "definition": (
                "VHL Disease — Belzutifan (HIF-2α Inhibitor) and Disease-Modifying Therapy: "
                "MECHANISM OF BELZUTIFAN (Welireg; MK-6482): "
                "  HIF-2α-specific inhibitor: "
                "    Binds HIF-2α PAS-B domain (β-barrel pocket); "
                "    Blocks HIF-2α:ARNT heterodimerisation; "
                "    Prevents HIF-2α transcriptional activity on VEGF, EPO, CAIX targets; "
                "  Why HIF-2α preferentially: "
                "    In ccRCC + hemangioblastoma: HIF-2α (not HIF-1α) is dominant oncogenic driver; "
                "    HIF-2α regulates Oct4/Nanog (stemness) and cyclin D1 in renal cancer; "
                "    Selective HIF-2α inhibition spares HIF-1α physiological functions; "
                "FDA APPROVAL: "
                "  2021: Belzutifan approved for VHL disease-related tumours: "
                "    VHL-associated RCC (clear cell); "
                "    CNS hemangioblastomas; "
                "    Pancreatic neuroendocrine tumours; "
                "  LITESPARK trial data: "
                "    ccRCC: ORR ~49%; disease control ~90%; median PFS not reached at 37mo; "
                "    CNS HB: ORR ~47%; many patients avoided brain/spine surgery; "
                "    Pancreatic NET: ORR ~83%; "
                "CLINICAL IMPACT: "
                "  First disease-modifying therapy for VHL: "
                "    Previously: surgery alone for every new RCC/hemangioblastoma; "
                "    Belzutifan enables watchful waiting protocol avoidance-of-surgery in many lesions; "
                "    <3cm renal lesions under belzutifan: defer nephrectomy; "
                "    Multi-site VHL (RCC + HB + pNET simultaneously): treat all with one oral agent; "
                "DOSING AND TOXICITY: "
                "  Standard dose: 120mg orally daily; "
                "  Main toxicity: anaemia (EPO suppression — HIF target) — dose-dependent; "
                "    Monitor CBC; EPO supplementation if severe; dose reduction 80mg/40mg; "
                "  Hypoxia (mountain sickness-like): HIF-2α is O2-sensing; "
                "  No significant immunosuppression; "
                "  Avoid: CYP2C19 inducers (reduce belzutifan exposure); "
                "SEQUENCING WITH SURGERY: "
                "  Active growing lesion or symptom: surgery first, then belzutifan adjuvant; "
                "  Stable or multiple small lesions: belzutifan first-line; "
                "  Post-surgery: continue belzutifan for residual/other-site VHL lesions; "
                "SCREENING FOR VHL THYROID LESIONS UNDER BELZUTIFAN: "
                "  Annual thyroid US continues despite belzutifan; "
                "  Clear-cell follicular TC: HIF targets CAIX/VEGF may respond to belzutifan (case reports); "
                "  Thyroid Ca in VHL: standard thyroidectomy if confirmed malignancy"
            ),
        },
        {
            "term": "CDC73-Parathyroid-Carcinoma-Surgical-Protocol",
            "definition": (
                "CDC73/HPT-JT Parathyroid Carcinoma — Surgical Protocol and Capsule-Disruption Warning: "
                "RECOGNISING PARATHYROID CARCINOMA PREOPERATIVELY: "
                "  Clinical clues (contrasts with benign adenoma): "
                "    Serum calcium: >3.5 mmol/L (benign adenoma rarely >3.0); "
                "    PTH: >3x ULN (benign adenoma typically 2-3x ULN); "
                "    Palpable neck mass: parathyroid carcinoma often large and palpable; "
                "    Severe symptomatic hypercalcaemia: renal calculi, pancreatitis, bone disease; "
                "    Recurrent PHPT after prior 'adenoma' resection: carcinoma or regrowth; "
                "  Biochemical: "
                "    Intact PTH >500 pg/mL + Ca >3.0 mmol/L = likely carcinoma; "
                "  Imaging: "
                "    Sestamibi scan: often strong uptake, large single lesion; "
                "    Neck US + CT neck: assess invasion of surrounding structures; "
                "    FDG-PET: may be positive in carcinoma (not adenoma); "
                "CRITICAL SURGICAL PRINCIPLE — NEVER DISRUPT THE CAPSULE: "
                "  Parathyroid carcinoma is histologically similar to adenoma in 50% cases; "
                "  DEFINITIVE DIAGNOSIS: vascular invasion or capsular invasion on histology; "
                "  Operative rule: "
                "    The tumour capsule must NEVER be breached during dissection; "
                "    Rupture causes seeding of carcinoma cells throughout neck → carcinomatosis; "
                "    Incurable local recurrence from capsule violation; "
                "EN BLOC RESECTION (mandatory technique): "
                "  Include: parathyroid tumour + ipsilateral thyroid lobe + isthmus; "
                "  Include: ipsilateral central neck soft tissue (pretracheal, prelaryngeal); "
                "  Include: any invaded structures (oesophagus, recurrent laryngeal nerve if invaded); "
                "  Minimise: contralateral exploration (avoid disruption of other glands); "
                "POST-OP MONITORING: "
                "  PTH half-life 3-5 minutes: measure at 0, 5, 10 minutes post-resection; "
                "  >50% PTH drop + PTH in normal range: cure; "
                "  Hypocalcaemia post-op (hungry bone syndrome): IV calcium + calcitriol; "
                "RECURRENT/METASTATIC PARATHYROID CARCINOMA: "
                "  Re-resection of localised recurrence: effective if en bloc; "
                "  Cinacalcet (Sensipar, calcimimetic): palliative — controls hypercalcaemia; "
                "  Denosumab: for severe bone disease; "
                "  Checkpoint immunotherapy: emerging (case reports); "
                "PARAFIBROMIN IHC IN SURGICAL PATHOLOGY: "
                "  Request on ALL parathyroid lesions with features concerning for carcinoma; "
                "  Nuclear staining absent: supports carcinoma (70-80% sensitive); "
                "  Nuclear staining present: favours adenoma but does not exclude; "
                "  Equivocal: CDC73 somatic sequencing of tumour"
            ),
        },
    ]
    return {
        "atlas":       "Hereditary-Thyroid-Cancer-Atlas",
        "seed_range":  f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2))
    print(json.dumps(generate_breakdown(), indent=2))
    print(json.dumps(generate_definitions(), indent=2))
