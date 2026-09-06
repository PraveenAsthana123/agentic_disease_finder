#!/usr/bin/env python3
"""Hereditary-Renal-Cancer-Atlas — Complete 8-Gene Hereditary Renal Cell Carcinoma Atlas
VHL     (Von Hippel-Lindau tumour suppressor; 213 aa; 3p25.3; AD;
         HRCC-VHL1 — 95% penetrance clear cell RCC; Belzutifan (HIF-2α inhibitor) FDA 2021;
         hemangioblastoma + pheochromocytoma + endolymphatic sac tumour; E3 ubiquitin ligase;
         seed SEED_BASE+0) ·
SDHB    (Succinate dehydrogenase subunit B; 280 aa; 1p36.13; AD;
         HRCC-SDH4 — SDH-associated RCC type 4; most aggressive SDH-RCC; papillary pattern;
         SDHB-IHC loss diagnostic; MIBG/DOTATATE for pheo/para; sunitinib preferred;
         seed SEED_BASE+1) ·
FH      (Fumarate hydratase; 510 aa; 1q43; AD;
         HRCC-HLRCC — Hereditary Leiomyomatosis and RCC; Type 2C RCC most aggressive;
         ANY SIZE warrants surgery; bevacizumab+erlotinib; 2SC-IHC surrogate;
         uterine leiomyoma 95% women; seed SEED_BASE+2) ·
FLCN    (Folliculin; 579 aa; 17p11.2; AD;
         HRCC-BHD — Birt-Hogg-Dubé syndrome; hybrid chromophobe/oncocytoma 55-67%;
         fibrofolliculomas PATHOGNOMONIC; spontaneous pneumothorax 25-33%;
         mTORC1 pathway; everolimus off-label; seed SEED_BASE+3) ·
MET     (MET proto-oncogene; 1390 aa; 7q31.2; AD;
         HRCC-HPRCC1 — Hereditary Papillary Renal Carcinoma Type 1; pure renal phenotype;
         GOF activating mutations; cabozantinib (MET/VEGFR2/AXL); trisomy 7;
         seed SEED_BASE+4) ·
BAP1    (BRCA1-associated protein-1; 729 aa; 3p21.1; AD;
         HRCC-BAP1-TPDS — BAP1 Tumour Predisposition Syndrome; clear cell RCC + uveal melanoma + mesothelioma;
         ANNUAL OPHTHALMOLOGY MANDATORY; deubiquitylase H2A; pembrolizumab;
         seed SEED_BASE+5) ·
SDHA    (Succinate dehydrogenase subunit A; 664 aa; 5p15.33; AD/AR;
         HRCC-SDH5 — SDH-RCC catalytic subunit; pituitary adenoma UNIQUE; rarest SDH-RCC;
         SDHA-IHC-specific loss; 1% population carrier; childhood AR severe;
         seed SEED_BASE+6) ·
PTEN    (Phosphatase and tensin homologue; 403 aa; 10q23.31; AD;
         HRCC-CS — Cowden syndrome / PTEN Hamartoma Tumour Syndrome; 34% lifetime renal cancer risk;
         everolimus FDA RECORD-1; macrocephaly + trichilemmomas PATHOGNOMONIC; multi-organ;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1654–1661)
"""

import random

SEED_BASE = 1654

RENAL_GENES = [
    # ── VHL — HRCC-VHL1 ──────────────────────────────────────────────────────
    {
        "gene": "VHL",
        "protein": "VHL — HRCC-VHL1 AD — EAB2/TCEB2 E3-Ubiquitin-Ligase — 95% Clear Cell RCC Penetrance — Belzutifan HIF-2α FDA 2021 — Hemangioblastoma+Pheo+ELST",
        "alias": (
            "VHL; OMIM gene 608537; Von Hippel-Lindau disease OMIM 193300. "
            "3p25.3; 213 aa; ~24 kDa; AD with loss of heterozygosity (tumour suppressor). "
            "VHL is the most common and best-characterised hereditary renal cell carcinoma gene: "
            "~1 in 36,000 births; estimated 60,000–70,000 VHL disease patients worldwide. "
            "FUNCTION: pVHL is the substrate-recognition subunit of the ELO-B/C–CUL2–RBX1 E3 ubiquitin "
            "ligase complex (VCB-CRL2 complex). Key substrate: HIF-α subunits (HIF-1α, HIF-2α). "
            "Under normoxia: PHD prolyl-hydroxylases hydroxylate HIF-α Pro402/Pro564 → "
            "VHL β-domain binds pOH-HIF-α → ubiquitylation → 26S proteasome degradation. "
            "Under VHL loss: HIF-α escapes degradation → constitutive transcription of VEGF, PDGF, "
            "EPO, GLUT1, Carbonic anhydrase IX (CAIX) → pseudohypoxia → clear cell RCC. "
            "HIF-2α (EPAS1) is the dominant oncogenic driver in ccRCC over HIF-1α — "
            "basis for Belzutifan (HIF-2α selective inhibitor) therapy. "
            "VHL DOMAIN STRUCTURE: α-domain (binds ELO-B/C scaffold); β-domain (binds HIF-OHPro). "
            "Mutation classes and clinical phenotype (Zbar classification): "
            "Type 1 (missense/truncating, loss of HIF binding) → low pheochromocytoma risk, high ccRCC risk. "
            "Type 2A (missense, partial HIF regulation preserved) → high pheo, low ccRCC. "
            "Type 2B (missense) → high pheo + high ccRCC. "
            "Type 2C (missense, retains HIF binding) → pheo only (Chuvash polycythaemia variant). "
            "PENETRANCE: ~95% of VHL carriers develop RCC by age 60; "
            "mean age of RCC diagnosis: ~37 years (range 16–67); "
            "multifocal, bilateral ccRCC is the hallmark — average 1100 tumours/lifetime (micro+macro). "
            "VHL DISEASE MANIFESTATIONS: "
            "ccRCC (95% penetrance); Cerebellar/spinal hemangioblastomas (72%); "
            "Retinal hemangioblastomas (50%) — earliest presentation, treatable if found early; "
            "Pheochromocytoma (10–20%, type 2 VHL only); "
            "Pancreatic cysts/neuroendocrine tumours (50–70%); "
            "Endolymphatic sac tumours (ELST, 10–16%) → sensorineural hearing loss; "
            "Epididymal cystadenomas (males). "
            "SURVEILLANCE PROTOCOL (VHL Alliance, European guidelines): "
            "Annual ophthalmology (retinal hemangioblastomas); "
            "Annual MRI brain+spine (hemangioblastomas); "
            "Annual MRI abdomen (RCC, pancreatic cysts/pNET); "
            "Annual urine catecholamines + plasma metanephrines (pheo); "
            "Audiogram biennial (ELST). "
            "RENAL INTERVENTION THRESHOLD: Treat when largest renal lesion reaches 3 cm "
            "(active surveillance for <3 cm lesions); multiple partial nephrectomies preferred; "
            "thermal ablation for small lesions; avoid total nephrectomy (bilateral multifocal disease). "
            "BELZUTIFAN (MK-6482): First-in-class HIF-2α inhibitor FDA-approved August 2021. "
            "LITESPARK-004 trial (Jonasch 2021, NEJM): VHL-associated ccRCC (n=61); "
            "ORR 49.2% (all partial responses); DCR 98.4%; median DOR not reached (follow-up 21.8 months); "
            "LITESPARK-005: Advanced non-VHL ccRCC vs everolimus (Phase III) — Belzutifan superior OS. "
            "Mechanism: Belzutifan binds HIF-2α PAS-B domain → prevents ARNT dimerisation → blocks HIF-2α "
            "transcriptional activity → reduced VEGF/PDGF/EPO production. "
            "Adverse effects: anaemia (52%), fatigue (36%), HIF-EPO suppression — manage with EPO/transfusion. "
            "GENETIC TESTING: Germline sequencing + MLPA (large deletions in ~7% of VHL); "
            "somatic VHL loss in 70–90% of all sporadic ccRCC — germline vs somatic distinction critical."
        ),
        "locus": "3p25.3",
        "aa": 213,
        "kDa": 24,
        "omim_gene": "608537",
        "omim_disease": "Von Hippel-Lindau Disease (OMIM 193300); Chuvash Polycythaemia (biallelic p.Arg200Trp, OMIM 263400)",
        "inheritance": "AD tumour suppressor; somatic second-hit LOH in tumour; biallelic Chuvash polycythaemia",
        "gene_class": "HIF ubiquitin E3 ligase — VCB-CRL2 substrate recognition subunit; pseudohypoxia suppressor",
        "key_alerts": [
            "VHL-BELZUTIFAN-FDA2021: Belzutifan (HIF-2α inhibitor) FDA approved for VHL-associated ccRCC — first-in-class; ORR 49.2% (LITESPARK-004)",
            "VHL-3CM-THRESHOLD: Treat renal lesions when largest tumour reaches 3 cm; active surveillance below threshold preserves renal function in bilateral multifocal disease",
            "VHL-ANNUAL-OPHTHALMOLOGY: Retinal hemangioblastomas present in 50% — earliest VHL manifestation; laser photocoagulation when small, enucleation risk with large untreated lesions",
            "VHL-TYPE2-PHEOCHROMOCYTOMA: Type 2 VHL mutations (missense, especially exon 2 cluster) — high pheochromocytoma risk; annual metanephrines mandatory",
            "VHL-ELST-HEARING: Endolymphatic sac tumours → progressive sensorineural hearing loss; biennial audiogram + MRI internal auditory canal",
        ],
        "etiologies": [
            "Germline VHL LOF → pVHL absent → HIF-2α escapes proteasomal degradation → VEGF/PDGF/EPO transcription → ccRCC",
            "Somatic second-hit (LOH 3p, somatic point mutation) → biallelic VHL loss in tumour cell",
            "HIF-2α pseudohypoxic drive: CAIX overexpression (IHC marker), VEGF-driven angiogenesis, EPO → polycythaemia",
            "Mutation type → phenotype: missense exon 2 (type 2B) = highest combined ccRCC+pheo risk; truncating/large deletion (type 1) = high ccRCC, low pheo",
        ],
        "stats": {
            "mean_dx_age": 37,
            "mean_dx_delay_months": 6,
            "renal_rr": "100× (near universal)",
            "lifetime_risk_pct": 95,
            "belzutifan_approved": True,
            "surveillance_mri_annual": True,
            "pheo_risk_pct": 15,
        },
        "dx_delay_distribution": "6–18 months (surveillance-detected in known VHL carriers; longer delay in de-novo/undiagnosed patients)",
    },
    # ── SDHB — HRCC-SDH4 ─────────────────────────────────────────────────────
    {
        "gene": "SDHB",
        "protein": "SDHB — HRCC-SDH4 AD — Iron-Sulfur Cluster Subunit — Paraganglioma-Pheochromocytoma + Aggressive RCC — SDHB-IHC Loss Diagnostic — MIBG/DOTATATE Mandatory",
        "alias": (
            "SDHB; OMIM gene 185470; Hereditary Paraganglioma-Pheochromocytoma Syndrome Type 4 "
            "(PGL4/Paraganglioma 4) OMIM 115310; also SDH-associated RCC. "
            "1p36.13; 280 aa; ~32 kDa; AD. "
            "SDHB encodes the iron-sulfur subunit of mitochondrial complex II (succinate:ubiquinone "
            "oxidoreductase), which catalyses oxidation of succinate to fumarate in the TCA cycle "
            "AND transfers electrons to ubiquinone in the respiratory chain. "
            "FUNCTION: SDHB contains three Fe-S clusters ([2Fe-2S], [4Fe-2S], [4Fe-4S]) that "
            "form the electron transport chain from SDHA (catalytic) to ubiquinone via SDHC/D (membrane anchor). "
            "SDH COMPLEX LOSS → SUCCINATE ACCUMULATION: succinate inhibits prolyl hydroxylases "
            "(PHD1/2/3) → HIF-α cannot be hydroxylated → VHL-independent HIF stabilisation → pseudohypoxia; "
            "succinate also inhibits TET methylcytosine dioxygenases and histone KDMs → epigenetic reprogramming; "
            "2-oxoglutarate-dependent dioxygenase inhibition → hypermethylator phenotype (similar to IDH mutations). "
            "CLINICAL MANIFESTATIONS (SDHB — Type 4 PGL): "
            "Paragangliomas (PGL): extra-adrenal sympathetic (abdomen/pelvis/thorax), parasympathetic (head/neck); "
            "SDHB-PGL is highest-risk for malignancy: ~40% of SDHB-PGL are malignant (metastatic); "
            "Pheochromocytoma: 25–30% of SDHB carriers develop pheo; "
            "Renal cell carcinoma: 10–15% lifetime risk; SDH-deficient RCC (WHO 2022 entity); "
            "histology: oncocytic cells with eosinophilic cytoplasmic inclusions (mitochondrial inclusions PATHOGNOMONIC); "
            "type 4 (papillary-like pattern) — aggressive, metastasize early even when small. "
            "SDHB-IHC: Loss of SDHB immunoreactivity in tumour (any SDH subunit mutation → SDHB protein destabilises); "
            "SDHB-IHC loss is sensitive but not specific for SDHB mutation — must correlate with germline; "
            "SDHA-IHC: additional SDHA staining needed to distinguish SDHA-specific loss. "
            "BIOCHEMICAL TESTING: Plasma metanephrines (normetanephrine elevated in SDHB pheo — dopaminergic); "
            "24h urine catecholamines; "
            "IMAGING: MIBG scintigraphy or DOTATATE-PET (68Ga-SST) for locoregional/metastatic staging; "
            "DOTATATE-PET superior to MIBG for SDHB-PGL (higher sensitivity). "
            "TREATMENT: Surgery for localised disease (primary); "
            "Sunitinib preferred for SDH-associated RCC (VEGFR pathway upregulated via HIF); "
            "Metastatic PGL: 177Lu-DOTATATE (LUTATHERA) for SSTR-positive disease; "
            "Cabozantinib/sunitinib for metastatic SDH-RCC; "
            "Temozolomide + capecitabine for SDH-PGL (off-label, anecdotal). "
            "SURVEILLANCE: Annual plasma metanephrines; biennial MRI total body; "
            "DOTATATE-PET at diagnosis and after surgery; "
            "Renal surveillance: annual MRI abdomen from age 20."
        ),
        "locus": "1p36.13",
        "aa": 280,
        "kDa": 32,
        "omim_gene": "185470",
        "omim_disease": "Paraganglioma-Pheochromocytoma Type 4 (OMIM 115310); SDH-Deficient Renal Cell Carcinoma (WHO 2022)",
        "inheritance": "AD; maternal imprinting debate (maternal SDHB silencing incomplete); biallelic somatic loss in tumour",
        "gene_class": "Mitochondrial complex II iron-sulfur subunit; TCA cycle electron transfer; oncometabolite (succinate) suppressor",
        "key_alerts": [
            "SDHB-MALIGNANT-PGL-40PCT: 40% of SDHB-associated PGL are malignant/metastatic — highest malignancy rate of all SDH genes; aggressive surveillance and staging mandatory",
            "SDHB-IHC-LOSS-DIAGNOSTIC: SDHB immunostaining loss in any tumour = SDH mutation (any subunit); correlate with SDHA staining to distinguish SDHB from SDHA-specific loss",
            "SDHB-DOTATATE-PET-PREFERRED: DOTATATE-PET superior to MIBG for SDHB-PGL staging (higher somatostatin receptor expression); use DOTATATE at diagnosis and surveillance",
            "SDHB-RCC-AGGRESSIVE: SDH-deficient RCC is aggressive even at small size; cytoplasmic eosinophilic inclusions PATHOGNOMONIC; annual MRI abdomen surveillance from age 20",
            "SDHB-SUCCINATE-PSEUDOHYPOXIA: Succinate accumulation → PHD inhibition → HIF-α stabilisation (VHL-independent); targeted by sunitinib/cabozantinib (anti-VEGFR downstream of HIF)",
        ],
        "etiologies": [
            "Germline SDHB LOF → Fe-S cluster electron transfer chain failure → complex II destabilisation → succinate accumulation",
            "Succinate → PHD inhibition → HIF-α pseudohypoxia → VEGF/PDGF upregulation → tumour angiogenesis",
            "Succinate → TET/KDM inhibition → hypermethylator epigenetic phenotype → gene silencing (similar to IDH/FH)",
            "Somatic LOH at 1p36 → biallelic SDHB loss in PGL/pheo/RCC → complete complex II abrogation",
        ],
        "stats": {
            "mean_dx_age": 38,
            "mean_dx_delay_months": 12,
            "renal_rr": "10–15× elevated risk",
            "lifetime_risk_pct": 13,
            "sunitinib_preferred": True,
            "dotatate_pet_mandatory": True,
            "malignant_pgl_pct": 40,
        },
        "dx_delay_distribution": "12–24 months (aggressive tumours, systemic symptoms; PGL/pheo may precede RCC by years)",
    },
    # ── FH — HRCC-HLRCC ──────────────────────────────────────────────────────
    {
        "gene": "FH",
        "protein": "FH — HRCC-HLRCC AD — TCA Fumarase — Type 2C RCC Most Aggressive — ANY-SIZE Warrants Surgery — Bevacizumab+Erlotinib — 2SC-IHC Surrogate — Uterine Leiomyoma 95% Women",
        "alias": (
            "FH; OMIM gene 136850; Hereditary Leiomyomatosis and Renal Cell Carcinoma (HLRCC) OMIM 150800; "
            "also: Fumarate hydratase deficiency (biallelic, OMIM 606812). "
            "1q43; 510 aa; ~55 kDa; AD (germline heterozygous → HLRCC); biallelic → encephalopathy/early death. "
            "FH encodes fumarate hydratase (fumarase), the TCA cycle enzyme converting fumarate to malate. "
            "FUNCTION: FH is a homotetramer (mitochondrial + cytoplasmic isoforms from single transcript); "
            "loss of FH → fumarate accumulation. "
            "ONCOMETABOLIC MECHANISMS OF FUMARATE: "
            "(1) PHD inhibition: fumarate is a competitive inhibitor of prolyl hydroxylases → "
            "HIF-1α stabilisation (VHL-independent pseudohypoxia); "
            "(2) 2-Succinylcysteine (2SC): fumarate nonenzymatically succinate-alkylates cysteine residues "
            "(protein succinylation) → 2SC-IHC is a surrogate marker for HLRCC tumours; "
            "(3) NRF2 activation: succination of KEAP1 Cys273/288 → NRF2 constitutive activation → "
            "antioxidant response + drug resistance; "
            "(4) TET/KDM2A/B inhibition → hypermethylator epigenetic phenotype + H3K9/36me2 accumulation. "
            "HLRCC — CLINICAL SYNDROME: "
            "Uterine leiomyomas (fibroids): ~95% of female HLRCC carriers; onset age 20–25; "
            "multiple, large, symptomatic; hysterectomy often required before age 30 in severely affected women; "
            "cutaneous leiomyomas (piloleiomyomas): multiple painful nodules, usually trunk/extremities, "
            "in 75% of carriers; PATHOGNOMONIC combination with uterine leiomyomas. "
            "HLRCC-ASSOCIATED RCC: Type 2C papillary RCC (WHO classification: collecting duct-like, "
            "papillary with prominent eosinophilic nucleoli with perinucleolar halos, high-grade); "
            "MOST AGGRESSIVE hereditary RCC — stage I tumours can already harbour micrometastases; "
            "even 1 cm HLRCC-associated RCC warrants surgical resection (unlike VHL 3 cm threshold); "
            "bilateral tumours possible but less common than VHL; "
            "5-year survival for advanced HLRCC-RCC: <40% (worse than sporadic ccRCC). "
            "DIAGNOSIS OF HLRCC-RCC: "
            "FH IHC (loss of FH staining); 2SC IHC (positive cytoplasmic succinylcysteine staining = HLRCC); "
            "germline FH sequencing + MLPA; RNA sequencing for deep-intronic variants. "
            "TREATMENT — HLRCC-RCC: "
            "Bevacizumab (anti-VEGF) + erlotinib (EGFR-TKI): MDACC (Srinivasan 2018) "
            "phase II — ORR 65%, PFS 21 months in HLRCC-RCC (best results of any hereditary RCC); "
            "rationale: fumarate → NRF2 → HER1/EGFR upregulation + HIF-1α → VEGF; dual blockade; "
            "sunitinib/pazopanib — lesser evidence in HLRCC vs sporadic ccRCC; "
            "mTOR inhibitors — mechanistically suboptimal (NRF2-driven, not mTOR-driven). "
            "SURVEILLANCE: MRI total body annually from diagnosis; "
            "annual renal MRI (abdomen/pelvis) from age 8 in confirmed carriers (unlike VHL at 15); "
            "uterine MRI annually for female carriers from age 18; "
            "skin examination annually (cutaneous leiomyomas for diagnosis). "
            "ANY SIZE RCC IN HLRCC = TREAT: "
            "Expert consensus (Linehan 2018, NEJM; HLRCC Consortium): "
            "No active surveillance for HLRCC-associated RCC regardless of size; "
            "immediate partial nephrectomy or ablation at detection; "
            "this distinguishes HLRCC from VHL (3 cm threshold) and BHD."
        ),
        "locus": "1q43",
        "aa": 510,
        "kDa": 55,
        "omim_gene": "136850",
        "omim_disease": "Hereditary Leiomyomatosis and Renal Cell Carcinoma HLRCC (OMIM 150800); Fumarate Hydratase Deficiency (biallelic, OMIM 606812)",
        "inheritance": "AD (HLRCC); biallelic → fumarate hydratase deficiency with severe neonatal encephalopathy",
        "gene_class": "TCA cycle — fumarate hydratase (fumarase); oncometabolic (fumarate) suppressor; NRF2/KEAP1/PHD regulator",
        "key_alerts": [
            "FH-ANY-SIZE-SURGERY: HLRCC-RCC warrants immediate surgical resection regardless of size — no active surveillance threshold (contrast: VHL 3 cm, BHD 3 cm); micrometastases at stage I are documented",
            "FH-BEVACIZUMAB-ERLOTINIB: Bevacizumab+erlotinib combination achieves 65% ORR in HLRCC-RCC (MDACC phase II) — rationale: fumarate → NRF2 → EGFR upregulation + HIF-VEGF axis",
            "FH-2SC-IHC-SURROGATE: 2-Succinylcysteine (2SC) IHC positive staining = HLRCC tumour marker; FH-IHC loss confirms diagnosis — use both in undiagnosed aggressive papillary RCC",
            "FH-UTERINE-LEIOMYOMA-95PCT: 95% of female HLRCC carriers develop symptomatic uterine leiomyomas age 20s; hysterectomy often before age 30; painful cutaneous piloleiomyomas PATHOGNOMONIC",
            "FH-BIALLELIC-LETHAL: Biallelic FH loss → fumarate hydratase deficiency with severe neonatal encephalopathy + leukoencephalopathy + early death — do NOT confuse with HLRCC heterozygous phenotype",
        ],
        "etiologies": [
            "Germline FH haploinsufficiency → fumarate accumulation → PHD inhibition → HIF-1α pseudohypoxia → VEGF/EGFR-driven RCC",
            "Fumarate → KEAP1 succination → NRF2 constitutive activation → antioxidant programme + EGFR upregulation → bevacizumab+erlotinib sensitivity",
            "2SC protein modification → altered proteome function; 2SC-IHC detects HLRCC tumours non-invasively",
            "Fumarate → TET2/KDM2A inhibition → epigenetic hypermethylator phenotype + H3K9me2 accumulation → gene silencing in HLRCC-RCC",
        ],
        "stats": {
            "mean_dx_age": 44,
            "mean_dx_delay_months": 14,
            "renal_rr": "15–20× (most aggressive)",
            "lifetime_risk_pct": 20,
            "bevacizumab_erlotinib_preferred": True,
            "any_size_treat": True,
            "uterine_leiomyoma_pct": 95,
        },
        "dx_delay_distribution": "14–30 months (often misdiagnosed as sporadic RCC; leiomyomas diagnosed separately; FH germline not tested until suspicious histology)",
    },
    # ── FLCN — HRCC-BHD ──────────────────────────────────────────────────────
    {
        "gene": "FLCN",
        "protein": "FLCN — HRCC-BHD AD — Folliculin DENN-Domain mTORC1 Regulator — Birt-Hogg-Dubé — Hybrid Chromophobe/Oncocytoma 55-67% — Fibrofolliculomas PATHOGNOMONIC — Pneumothorax 25-33%",
        "alias": (
            "FLCN; OMIM gene 607273; Birt-Hogg-Dubé syndrome (BHD) OMIM 135150. "
            "17p11.2; 579 aa; ~64 kDa; AD haploinsufficiency. "
            "FLCN encodes folliculin, a tumour suppressor with a DENN (differentially expressed in normal and "
            "neoplastic cells)-like domain (N-terminal) and DENN-fold domain (C-terminal) that functions as "
            "a GAP (GTPase-activating protein) for Rab7a and a GEF for TFEB/TFE3 lysosomal translocation. "
            "KEY FUNCTION: FLCN forms complex with FNIP1/FNIP2 (folliculin-interacting proteins); "
            "FLCN-FNIP1/2 → activates AMPK → inhibits mTORC1 (via TSC1/2); "
            "additionally: FLCN-FNIP complex inhibits lysosomal TFEB/TFE3 nuclear translocation → "
            "suppresses MiT/TFE transcription factor activity. "
            "BHD SYNDROME MANIFESTATIONS: "
            "SKIN: Fibrofolliculomas (hair follicle hamartomas) — white papules on face, neck, trunk; "
            "PATHOGNOMONIC for BHD; onset 3rd–4th decade; histology: mantle zone hair follicle dilation "
            "with fibrous stroma; also trichodiscomas, acrochordons. "
            "LUNG: Pulmonary cysts — basilar/subpleural distribution; "
            "spontaneous pneumothorax in 25–33% of BHD carriers; "
            "mean age at first pneumothorax: 38 years (range 7–70); "
            "recurrent pneumothorax → pleurodesis; no malignant transformation of lung cysts. "
            "RENAL: RCC in 30–34% of BHD carriers by age 70; "
            "HISTOLOGY (BHD-RCC distinctive): "
            "Hybrid oncocytic/chromophobe RCC (55–67%): features of both chromophobe + oncocytoma; "
            "Chromophobe RCC (22%): raisinoid nuclei, pale cytoplasm, Hale colloidal iron positive; "
            "Clear cell RCC (9%): more aggressive than hybrid; "
            "Oncocytoma (3%): benign; "
            "BHD-RCC onset: mean age 50 (range 31–72); "
            "multifocal bilateral in ~67% of cases; low-grade, slow-growing. "
            "SURVEILLANCE: Annual MRI abdomen from age 20–25; "
            "CT chest at diagnosis + every 2–3 years (lung cysts, pneumothorax risk); "
            "annual skin examination; ophthalmology at baseline. "
            "TREATMENT — BHD-RCC: "
            "Surveillance threshold: 3 cm (conservative management below); partial nephrectomy preferred; "
            "mTORC1 inhibitors (everolimus, temsirolimus) — off-label rationale (FLCN → mTORC1 loss); "
            "limited prospective trial data for BHD-specific RCC; "
            "sunitinib/pazopanib for advanced BHD-RCC (VEGFR pathway activated downstream of mTORC1). "
            "GENETIC TESTING: FLCN germline sequencing + MLPA (exonic deletions/duplications in ~5%); "
            "hotspot mutation: c.1285delC (exon 11, European founder, ~40% of BHD families); "
            "truncating mutations in exon 11 = most common; frameshift/nonsense predominant."
        ),
        "locus": "17p11.2",
        "aa": 579,
        "kDa": 64,
        "omim_gene": "607273",
        "omim_disease": "Birt-Hogg-Dubé Syndrome (OMIM 135150)",
        "inheritance": "AD haploinsufficiency; exon 11 frameshift/truncating most common",
        "gene_class": "DENN-domain mTORC1 suppressor; AMPK activator; FNIP1/2 complex; lysosomal TFEB/TFE3 regulator",
        "key_alerts": [
            "FLCN-FIBROFOLLICULOMAS-PATHOGNOMONIC: White dome-shaped facial/neck/trunk papules (fibrofolliculomas) = BHD diagnosis; biopsy one lesion for histological confirmation before germline testing if suspicious",
            "FLCN-PNEUMOTHORAX-25PCT: Spontaneous pneumothorax in 25–33% of BHD carriers; recurrent → pleurodesis; avoid high-altitude activities + scuba diving pre-pleurodesis; family alert for emergency presentation",
            "FLCN-HYBRID-CHROMOPHOBE-ONCOCYTOMA: BHD-RCC histology is distinctive (55–67% hybrid chromophobe/oncocytoma); pathologists may diagnose as oncocytoma (benign) — submit genetic context for all bilateral multifocal oncocytic RCC",
            "FLCN-3CM-SURVEILLANCE: Unlike HLRCC (treat any size), BHD-RCC follows 3 cm threshold for intervention; multifocal bilateral → nephron-sparing partial nephrectomy prioritised",
            "FLCN-MTORC1-RATIONALE: FLCN loss → mTORC1 constitutive activation → everolimus/temsirolimus mechanistically rational (off-label); sunitinib for VEGFR-axis downstream activation",
        ],
        "etiologies": [
            "Germline FLCN LOF → loss of FLCN-FNIP1/2 complex → AMPK under-activation → mTORC1 constitutive activation → hybrid chromophobe/oncocytic RCC",
            "FLCN loss → TFEB/TFE3 nuclear translocation → MiT/TFE transcription → lysosomal biogenesis + cellular metabolism reprogramming → RCC",
            "Pulmonary cyst formation: FLCN loss in alveolar type II pneumocytes → mTORC1/AMPK dysregulation → structural cyst development → pneumothorax",
            "Somatic second-hit LOH at 17p11.2 in RCC tumour → biallelic FLCN loss confirms cancer pathway",
        ],
        "stats": {
            "mean_dx_age": 50,
            "mean_dx_delay_months": 18,
            "renal_rr": "7× (hybrid chromophobe/oncocytoma pattern)",
            "lifetime_risk_pct": 34,
            "everolimus_off_label": True,
            "pneumothorax_pct": 28,
            "bilateral_multifocal_pct": 67,
        },
        "dx_delay_distribution": "18–36 months (fibrofolliculomas often missed; pneumothorax may be first presentation; RCC found incidentally on CT chest)",
    },
    # ── MET — HRCC-HPRCC1 ────────────────────────────────────────────────────
    {
        "gene": "MET",
        "protein": "MET — HRCC-HPRCC1 AD — RTK HGF Receptor — Hereditary Papillary RCC Type 1 — Pure Renal Phenotype — GOF Activating Mutations — Cabozantinib MET/VEGFR2/AXL — Trisomy 7",
        "alias": (
            "MET; OMIM gene 164860; Hereditary Papillary Renal Cell Carcinoma Type 1 (HPRCC) OMIM 605074. "
            "7q31.2; 1390 aa; ~170 kDa (mature receptor); AD gain-of-function. "
            "MET encodes the MET receptor tyrosine kinase (RTK), the high-affinity receptor for "
            "Hepatocyte Growth Factor / Scatter Factor (HGF/SF). "
            "FUNCTION: MET-HGF signalling activates RAS/MAPK, PI3K/AKT/mTOR, STAT3, and β-catenin pathways "
            "→ cell proliferation, survival, motility, invasion, angiogenesis; "
            "essential in embryogenesis (hepatocyte, myoblast, renal tubule development); "
            "MET activation in kidney: proximal tubule cell regeneration after injury. "
            "HPRCC GAIN-OF-FUNCTION MUTATIONS: "
            "Activating missense mutations in exons 16–19 (kinase activation loop/P+1 loop): "
            "pThr1010Ile, pVal1110Ile, pMet1268Thr, pAsp1246Asn (most common); "
            "mutations mimic HGF-bound state → constitutive kinase activation without ligand; "
            "penetrance: ~80–90% of HPRCC carriers develop RCC; "
            "tumours are bilateral, multifocal (hundreds of micro-tumours throughout both kidneys). "
            "HISTOLOGY — TYPE 1 PAPILLARY RCC: "
            "Papillary architecture with basophilic/amphophilic small cells; "
            "low nuclear grade (Fuhrman 1–2 / ISUP 1–2); "
            "foamy macrophages within papillary cores; "
            "psammoma bodies occasional; "
            "trisomy 7 (7q31 MET amplification/gain) in 70–80% of papillary RCC — somatic; "
            "HPRCC type 1 papillary RCC has better prognosis than clear cell or FH-associated RCC. "
            "PURE RENAL PHENOTYPE: "
            "HPRCC does NOT feature extra-renal manifestations (unlike VHL, BHD, HLRCC, BAP1-TPDS); "
            "no skin lesions, no pheochromocytoma, no lung pathology; "
            "RENAL-ONLY SYNDROME — bilateral multifocal papillary RCC is the sole manifestation. "
            "SURVEILLANCE: Annual MRI or CT abdomen from age 30–35 "
            "(sooner if family history of early onset); "
            "bilateral multifocal tumours → renal function preservation critical. "
            "TREATMENT: "
            "Threshold: 3 cm (active surveillance below); partial nephrectomy preferred; "
            "radiofrequency ablation/cryoablation for small lesions in bilateral disease; "
            "Cabozantinib (MET/VEGFR2/AXL inhibitor) — FDA approved for advanced RCC generally; "
            "highest MET activity in HPRCC (GOF mutations); "
            "Foretinib (MET+VEGFR2): NCI phase II — 13.5% ORR in HPRCC (Choueiri 2013); "
            "Savolitinib (selective MET inhibitor): phase II — ORR 27% in MET-driven papillary RCC; "
            "Tepotinib + everolimus (MET-selective + mTOR): early-phase data. "
            "SOMATIC MET IN RCC: MET amplification/overexpression in 7–10% sporadic ccRCC → "
            "cabozantinib activity independent of hereditary status."
        ),
        "locus": "7q31.2",
        "aa": 1390,
        "kDa": 170,
        "omim_gene": "164860",
        "omim_disease": "Hereditary Papillary Renal Cell Carcinoma Type 1 (OMIM 605074)",
        "inheritance": "AD gain-of-function; activating missense in exons 16–19; penetrance ~80–90%",
        "gene_class": "Receptor tyrosine kinase — HGF receptor; RAS/PI3K/STAT3 activator; GOF hereditary cancer gene",
        "key_alerts": [
            "MET-PURE-RENAL-PHENOTYPE: HPRCC has NO extra-renal features (no skin, lung, pheo manifestations); bilateral multifocal papillary RCC is the sole clinical presentation — do not search for systemic features",
            "MET-GOF-ACTIVATING: Gain-of-function exon 16–19 missense mutations (not LOF like most hereditary cancer genes) → constitutive RTK activation; this makes MET inhibitors (cabozantinib, savolitinib) mechanistically first-choice",
            "MET-TRISOMY-7-SOMATIC: Somatic trisomy 7 (MET locus amplification) in 70–80% of sporadic papillary RCC type 1 — germline HPRCC and somatic MET amplification both converge on MET pathway",
            "MET-CABOZANTINIB-PREFERRED: Cabozantinib (MET/VEGFR2/AXL) FDA approved advanced RCC; highest biological rationale in HPRCC; savolitinib (selective MET) shows 27% ORR in MET-driven papillary RCC",
            "MET-BILATERAL-MULTIFOCAL: Hundreds of micro-tumours throughout both kidneys at pathology; renal function preservation (nephron-sparing surgery) critical; bilateral nephrectomy leads to dialysis dependence",
        ],
        "etiologies": [
            "Germline MET GOF missense → constitutive RTK kinase activation (HGF-independent) → RAS/MAPK+PI3K/AKT+STAT3 proliferation → bilateral papillary RCC",
            "MET activation → HIF-1α transcription (independent pathway) → VEGF/PDGF → angiogenesis in type 1 papillary RCC",
            "Somatic trisomy 7 (acquired) amplifies GOF MET allele → further MET overexpression in established tumour",
            "MET-driven PI3K/mTOR: downstream mTOR activation → everolimus+MET inhibitor rational combination (dual blockade)",
        ],
        "stats": {
            "mean_dx_age": 55,
            "mean_dx_delay_months": 8,
            "renal_rr": "90× (pure renal phenotype)",
            "lifetime_risk_pct": 85,
            "cabozantinib_preferred": True,
            "bilateral_pct": 90,
            "extra_renal_features": False,
        },
        "dx_delay_distribution": "8–18 months (incidental detection on imaging; may be found late if asymptomatic; surveillance reduces delay in known HPRCC families)",
    },
    # ── BAP1 — HRCC-BAP1-TPDS ────────────────────────────────────────────────
    {
        "gene": "BAP1",
        "protein": "BAP1 — HRCC-BAP1-TPDS AD — H2A Deubiquitylase BRCT-Domain — Clear Cell RCC + Uveal Melanoma + Mesothelioma — ANNUAL OPHTHALMOLOGY MANDATORY — 3p21.1 Near VHL",
        "alias": (
            "BAP1; OMIM gene 603089; BAP1 Tumour Predisposition Syndrome (BAP1-TPDS) OMIM 614327. "
            "3p21.1; 729 aa; ~80 kDa; AD haploinsufficiency (tumour suppressor). "
            "BAP1 encodes BRCA1-associated protein-1, a nuclear deubiquitylase (DUB) and tumour suppressor. "
            "STRUCTURE AND FUNCTION: "
            "N-terminal UCH (ubiquitin C-terminal hydrolase) catalytic domain → deubiquitylase activity; "
            "C-terminal BRCT (BRCA1 C-Terminal) domain → protein-protein interaction scaffold; "
            "BAP1 deubiquitylates H2AK119ub1 (monoubiquitinated histone H2A Lys119) → "
            "this opposes PRC1 (Polycomb Repressive Complex 1) H2A ubiquitylation → gene activation; "
            "BAP1 also binds ASXL1/2/3 (PR-DUB Polycomb Repressive DeUBiquitylase complex) → "
            "chromatin remodelling and transcriptional regulation; "
            "nuclear localisation signal (NLS) in BRCT domain; "
            "cytoplasmic BAP1 sequestration (when mutant) → mislocalized, non-functional. "
            "BAP1-TPDS — CLINICAL SYNDROME: "
            "Multi-tumour syndrome with overlapping cancer predisposition: "
            "Uveal melanoma (UM): 25–50% lifetime risk; "
            "ANNUAL OPHTHALMOLOGY WITH DILATED FUNDOSCOPY MANDATORY from diagnosis; "
            "UM is often the first BAP1-TPDS manifestation (mean age 40–50); "
            "BAP1-associated UM: class 2 high-risk, metastasising to liver in ~50% within 5 years. "
            "Mesothelioma: 5–10% lifetime risk; predominantly pleural; "
            "younger onset than sporadic mesothelioma; asbestos co-exposure dramatically increases risk; "
            "AVOID ASBESTOS EXPOSURE in BAP1 carriers (absolute). "
            "Clear cell RCC: 10–15% lifetime risk; mean age 55–60; conventional ccRCC histology; "
            "may be difficult to distinguish from VHL-associated ccRCC on pathology (3p21.1 near 3p25 VHL). "
            "Cutaneous melanoma: elevated risk (exact penetrance undefined); "
            "atypical Spitz tumours (BAP1-inactivated melanocytomas, BIMI) — benign pigmented papules "
            "with characteristic BAP1-loss IHC — not malignant but diagnostic skin marker. "
            "Additional risks: cholangiocarcinoma, basal cell carcinoma, thyroid cancer. "
            "DIAGNOSIS: "
            "BAP1 germline sequencing + MLPA; "
            "BAP1-IHC loss in tumour (nuclear BAP1 absent) — validates somatic second hit; "
            "atypical Spitz tumour biopsy with BAP1-IHC loss → consider germline testing. "
            "TREATMENT — BAP1-TPDS: "
            "ccRCC: sunitinib/pazopanib standard; immune checkpoint inhibitors (pembrolizumab/nivolumab) "
            "with rationale (BAP1-loss tumours have increased mutational burden); "
            "UM metastatic: tebentafusp (IMCgp100, FDA 2022) for HLA-A*02:01+ uveal melanoma — "
            "first approved therapy specific to UM (immune redirecting); "
            "selumetinib (MEK inhibitor) — limited activity; ipilimumab/nivolumab for metastatic UM. "
            "Mesothelioma: platinum/pemetrexed ± bevacizumab; nivolumab+ipilimumab (CheckMate 743 OS benefit). "
            "SURVEILLANCE: Annual ophthalmology (UM); annual MRI abdomen (RCC); "
            "annual CT chest (mesothelioma screening — no validated protocol); "
            "annual skin exam; avoid asbestos."
        ),
        "locus": "3p21.1",
        "aa": 729,
        "kDa": 80,
        "omim_gene": "603089",
        "omim_disease": "BAP1 Tumour Predisposition Syndrome BAP1-TPDS (OMIM 614327)",
        "inheritance": "AD haploinsufficiency; somatic LOH at 3p21.1 in tumour; biallelic somatic in sporadic cases",
        "gene_class": "Deubiquitylase (DUB) — H2AK119ub1 eraser; BRCT scaffold; PR-DUB Polycomb complex; chromatin remodelling tumour suppressor",
        "key_alerts": [
            "BAP1-UVEAL-MELANOMA-ANNUAL-OPHTHALMOLOGY-MANDATORY: 25–50% lifetime uveal melanoma risk; annual dilated fundoscopy from diagnosis; BAP1-UM is class 2 high-risk, liver-metastasising — delay in surveillance risks metastatic disease",
            "BAP1-MESOTHELIOMA-ASBESTOS-ABSOLUTE-CI: Asbestos co-exposure dramatically multiplies BAP1 mesothelioma risk; lifelong asbestos avoidance mandatory for all BAP1 carriers and at-risk family members",
            "BAP1-TPDS-TRIPLE-CANCER: Clear cell RCC + uveal melanoma + mesothelioma in one patient/family = BAP1-TPDS until proven otherwise; single-gene testing insufficient — germline panel required",
            "BAP1-IHC-BIMI-SKIN: Atypical Spitz tumour (BAP1-inactivated melanocytoma) skin biopsy with BAP1-IHC loss → germline BAP1 testing trigger; benign but diagnostic marker of BAP1-TPDS",
            "BAP1-TEBENTAFUSP-UM: Tebentafusp (IMCgp100) FDA 2022 — only HLA-A*02:01+ uveal melanoma; test HLA type in all BAP1-UM; first therapy with OS benefit in metastatic UM (CheckMate ORR: 9% but OS +)",
        ],
        "etiologies": [
            "Germline BAP1 LOF → H2AK119ub1 deubiquitylation impaired → PRC1 H2A ubiquitylation unopposed → transcriptional silencing of tumour suppressors → multi-cancer predisposition",
            "BAP1 loss → ASXL1-PR-DUB complex disruption → epigenetic dysregulation of developmental/proliferation gene networks",
            "3p21.1 LOH (somatic second hit) in ccRCC: somatic loss near VHL (3p25) often co-occurs → dual 3p hits in some BAP1-associated RCC",
            "Uveal melanoma: BAP1 somatic loss in class 2 tumours (biallelic) → highest metastatic risk (liver-tropic); germline BAP1 carriers start with one hit already in every cell",
        ],
        "stats": {
            "mean_dx_age": 55,
            "mean_dx_delay_months": 10,
            "renal_rr": "8× elevated risk",
            "lifetime_risk_pct": 13,
            "uveal_melanoma_risk_pct": 35,
            "mesothelioma_risk_pct": 8,
            "annual_ophthalmology_mandatory": True,
        },
        "dx_delay_distribution": "10–24 months (uveal melanoma often first; RCC found at workup; multi-speciality syndrome recognition slow)",
    },
    # ── SDHA — HRCC-SDH5 ─────────────────────────────────────────────────────
    {
        "gene": "SDHA",
        "protein": "SDHA — HRCC-SDH5 AD/AR — Flavoprotein Catalytic Subunit — Pituitary Adenoma UNIQUE — 1% Population Carrier — SDHA-IHC-Specific Loss — Rarest SDH-RCC",
        "alias": (
            "SDHA; OMIM gene 600857; Pituitary Adenoma 5 (multiple types) OMIM 617542; "
            "SDH-deficient GIST; SDH-deficient RCC (WHO 2022); Leigh syndrome / Complex II deficiency (biallelic). "
            "5p15.33; 664 aa; ~73 kDa; AD (paraganglioma/pituitary) or AR (Leigh syndrome). "
            "SDHA encodes the flavoprotein (Fp) catalytic subunit of complex II "
            "(succinate:ubiquinone oxidoreductase), the only TCA cycle enzyme also integral to "
            "the electron transport chain. "
            "STRUCTURE: SDHA contains: FAD cofactor-binding domain (flavoprotein subdomain A, B, C) → "
            "FAD-dependent oxidation of succinate to fumarate; "
            "SDHA non-covalently binds SDHB, and SDHA+SDHB = catalytic dimer (peripheral arm); "
            "SDHC+SDHD = membrane anchor; all four subunits required for complex II integrity. "
            "UNIQUE SDHA FEATURES DISTINGUISHING FROM SDHB/C/D: "
            "(1) PITUITARY ADENOMA: SDHA germline mutations are the ONLY SDH gene associated with "
            "functioning and non-functioning pituitary adenomas (acromegaly, Cushing's, prolactinoma, null-cell); "
            "not seen with SDHB/C/D; pituitary adenoma surveillance recommended annually from age 20. "
            "(2) IHC SPECIFICITY: SDHA-IHC loss specifically indicates SDHA mutation "
            "(other SDH mutations → SDHB-IHC loss only, SDHA preserved); "
            "SDHA + SDHB dual IHC staining identifies SDHA specifically; "
            "any tumour with BOTH SDHA and SDHB IHC loss → SDHA germline testing mandatory. "
            "(3) POPULATION PREVALENCE: SDHA pathogenic variants found in ~1% of general population "
            "(highest germline carrier rate of all SDH subunits); "
            "but penetrance for paraganglioma/RCC is lower than SDHB (~5–10% lifetime). "
            "(4) BIALLELIC (AR): Leigh syndrome with isolated complex II deficiency — "
            "neonatal/infantile severe encephalopathy + lactic acidosis + basal ganglia lesions; "
            "separate from heterozygous carrier phenotype. "
            "SDHA-ASSOCIATED NEOPLASMS (AD heterozygous): "
            "Paraganglioma/pheochromocytoma (lower malignancy rate than SDHB, ~3–5%); "
            "GIST (gastrointestinal stromal tumours) — Carney triad/Carney-Stratakis; "
            "RCC: SDH-deficient RCC; rarest SDH-gene-associated RCC; "
            "characteristic oncocytic cytoplasmic inclusions on H&E (same as SDHB-RCC); "
            "Pituitary adenomas (SDHA-unique). "
            "TREATMENT: Surgery for localised lesions (PGL, GIST, RCC, pituitary); "
            "sunitinib for advanced SDH-deficient RCC; "
            "imatinib-resistant SDHA-GIST → sunitinib/regorafenib; "
            "pituitary: octreotide/lanreotide for functioning (GH, ACTH); dopamine agonist for prolactin; "
            "surgery (transsphenoidal) for non-functioning macro-adenomas."
        ),
        "locus": "5p15.33",
        "aa": 664,
        "kDa": 73,
        "omim_gene": "600857",
        "omim_disease": "Pituitary Adenoma Predisposition (OMIM 617542); SDH-Deficient GIST; SDH-Deficient RCC (WHO 2022); Leigh Syndrome Complex II Deficiency (biallelic, OMIM 252011)",
        "inheritance": "AD (paraganglioma/pituitary/RCC — heterozygous); AR (Leigh syndrome — biallelic homozygous/compound heterozygous)",
        "gene_class": "Mitochondrial complex II catalytic subunit; FAD-dependent succinate dehydrogenase; TCA cycle + electron transport chain",
        "key_alerts": [
            "SDHA-PITUITARY-UNIQUE: Pituitary adenoma predisposition is UNIQUE to SDHA among all SDH subunits — annual pituitary MRI from age 20 in SDHA carriers; screen for acromegaly (GH/IGF-1), Cushing's (UFC/LNSC), prolactin",
            "SDHA-IHC-DUAL-SPECIFIC: SDHA + SDHB dual IHC loss identifies SDHA-specific mutation (SDHB loss alone does NOT distinguish SDHA from SDHB/C/D); always run SDHA IHC alongside SDHB in suspected SDH-deficient tumours",
            "SDHA-1PCT-CARRIER: ~1% population SDHA pathogenic variant prevalence — highest SDH carrier rate; yet RCC/PGL penetrance lower than SDHB; many carriers undiagnosed — SDHA testing in unexplained GIST/PGL essential",
            "SDHA-BIALLELIC-LEIGH: Homozygous/compound heterozygous SDHA → Leigh syndrome (severe neonatal complex II deficiency); reproductive counselling for consanguineous SDHA carrier couples; distinct from heterozygous cancer predisposition",
            "SDHA-GIST-IMATINIB-RESISTANT: SDHA-driven GIST is KIT/PDGFRA-wild-type → primary imatinib resistance; sunitinib or regorafenib as primary treatment; do NOT offer imatinib as first-line",
        ],
        "etiologies": [
            "Germline SDHA LOF → complex II catalytic dysfunction → succinate accumulation → PHD/TET/KDM inhibition (same oncometabolic pathway as SDHB/FH) → HIF pseudohypoxia → RCC/PGL/GIST",
            "SDHA-FNIP2 interaction impaired → mTORC1 upregulation → cellular metabolic reprogramming in pituitary",
            "Somatic LOH at 5p15 + germline hit → biallelic SDHA loss in tumour → complex II complete abrogation",
            "Succinate hypermethylator phenotype: TET2/KDM inhibition → global DNA hypermethylation → tumour suppressor silencing in GIST/pituitary",
        ],
        "stats": {
            "mean_dx_age": 42,
            "mean_dx_delay_months": 24,
            "renal_rr": "4–6× (rarest SDH-RCC)",
            "lifetime_risk_pct": 7,
            "pituitary_adenoma_unique": True,
            "carrier_prevalence_pct": 1,
            "imatinib_resistant_gist": True,
        },
        "dx_delay_distribution": "24–48 months (rarest SDH gene, least recognised; pituitary diagnosis often precedes RCC; multi-tumour presentation confusing)",
    },
    # ── PTEN — HRCC-CS ────────────────────────────────────────────────────────
    {
        "gene": "PTEN",
        "protein": "PTEN — HRCC-CS AD — PI3K Phosphatase Master Regulator — Cowden Syndrome PHTS — 34% Lifetime Renal Cancer Risk — Everolimus FDA RECORD-1 — Macrocephaly+Trichilemmomas PATHOGNOMONIC",
        "alias": (
            "PTEN; OMIM gene 601728; Cowden Syndrome 1 (CS1/PHTS) OMIM 158350; "
            "also: Bannayan-Riley-Ruvalcaba syndrome (BRRS); Proteus syndrome (mosaic PTEN); PTEN-related Proteus-like syndrome. "
            "10q23.31; 403 aa; ~47 kDa; AD haploinsufficiency. "
            "PTEN (Phosphatase and Tensin Homologue deleted on chromosome Ten) is the most frequently "
            "inactivated tumour suppressor in human cancer (second only to TP53). "
            "FUNCTION: PTEN is a dual-specificity phosphatase with: "
            "(1) lipid phosphatase activity: dephosphorylates PIP3 → PIP2 → "
            "counteracts PI3K → abrogates AKT activation → mTORC1 inhibition; "
            "(2) protein phosphatase activity: dephosphorylates FAK, PDGFR → cytoskeletal regulation. "
            "PTEN LOSS → AKT/mTOR constitutive activation: "
            "PI3K → PIP3 unopposed → PDK1/AKT phosphorylation → TSC1/2 inactivation → "
            "mTORC1 (Raptor complex) constitutive activity → S6K1/4EBP1 → protein synthesis, "
            "cell growth, cell cycle, angiogenesis (HIF → VEGF under mTOR). "
            "COWDEN SYNDROME / PHTS — CLINICAL FEATURES: "
            "Diagnostic criteria (NCCN CS guidelines): "
            "Pathognomonic: trichilemmomas (benign hair follicle tumours, face); "
            "oral mucosal papillomas (cobblestone appearance, gums/buccal mucosa); "
            "macrocephaly (OFC >97th percentile, ~84% of CS); "
            "Lhermitte-Duclos disease (dysplastic gangliocytoma cerebellum, adults). "
            "CANCER RISKS IN PHTS/CS: "
            "Breast: 85% lifetime (female carriers) — highest PTEN-associated risk; "
            "Thyroid: 38% lifetime (predominantly follicular thyroid carcinoma, NOT papillary); "
            "Endometrial: 28% lifetime; "
            "Colorectal: 9% lifetime; "
            "RENAL CELL CARCINOMA: 34% lifetime risk (predominantly papillary type I + ccRCC); "
            "younger onset vs sporadic RCC (mean age 52); bilateral and multifocal possible. "
            "PTEN RENAL CANCER — TREATMENT: "
            "Everolimus (mTOR inhibitor): FDA approved for advanced renal cell carcinoma "
            "(RECORD-1 trial — PTEN-loss cohort subanalysis supports everolimus efficacy); "
            "RECORD-1 (Motzer 2010, Lancet): everolimus vs placebo post-sunitinib; "
            "PFS 4.9 vs 1.9 months; HR 0.33; FDA approved 2009 for advanced RCC second-line. "
            "Temsirolimus (mTOR inhibitor): FDA approved for advanced RCC (poor prognosis, first-line); "
            "VEGFR TKIs (sunitinib, pazopanib) — first-line advanced PTEN-RCC; "
            "nivolumab+ipilimumab for high-risk clear cell (PTEN-loss does not confer specific ICI sensitivity). "
            "GENETIC TESTING: PTEN germline sequencing (point mutations, small indels 80%); "
            "MLPA for large deletions (5–10%); "
            "PTEN promoter methylation testing; "
            "PTEN 5' UTR and deep-intronic variant testing for unexplained clinical CS. "
            "SURVEILLANCE: Annual breast MRI + mammography from age 30–35; "
            "annual thyroid ultrasound from age 18; "
            "annual endometrial ultrasound ± biopsy from age 35; "
            "colonoscopy every 5 years from age 35 (or 40); "
            "annual MRI abdomen (renal) from age 40. "
            "NEURODEVELOPMENTAL: 20–30% of PHTS patients have autism spectrum disorder or "
            "macrocephaly + developmental delay — PTEN is a critical neurodevelopmental gene; "
            "autism with macrocephaly → PTEN germline testing recommended (NCCN)."
        ),
        "locus": "10q23.31",
        "aa": 403,
        "kDa": 47,
        "omim_gene": "601728",
        "omim_disease": "Cowden Syndrome 1 / PTEN Hamartoma Tumour Syndrome PHTS (OMIM 158350); Bannayan-Riley-Ruvalcaba Syndrome (OMIM 153480)",
        "inheritance": "AD haploinsufficiency; germline missense/nonsense/frameshift/large deletion; mosaic Proteus syndrome rare",
        "gene_class": "PI3K lipid phosphatase — PIP3 → PIP2 conversion; AKT/mTOR master suppressor; most frequently altered tumour suppressor in human cancer",
        "key_alerts": [
            "PTEN-BREAST-85PCT-PRIORITISE: 85% lifetime breast cancer risk (female carriers) is the most penetrant PTEN-associated cancer — MRI breast annually from age 30; PTEN-RCC surveillance secondary but still 34% lifetime risk",
            "PTEN-EVEROLIMUS-FDA-RECORD1: Everolimus (mTOR inhibitor) FDA approved advanced RCC; PTEN-loss mechanistically driven by mTORC1 constitutive activation — strongest pharmacological rationale for mTOR inhibition in PTEN-RCC",
            "PTEN-MACROCEPHALY-PATHOGNOMONIC: Macrocephaly (OFC >97th percentile) + trichilemmomas (facial papules) + oral mucosal papillomas = Cowden syndrome diagnosis; germline PTEN testing mandatory in macrocephalic autism (NCCN)",
            "PTEN-THYROID-FOLLICULAR-NOT-PAPILLARY: PTEN-associated thyroid cancer is predominantly FOLLICULAR thyroid carcinoma (not papillary); pathologist alert — follicular adenoma in CS = borderline malignant; annual thyroid US from age 18",
            "PTEN-ASD-MACROCEPHALY-TESTING: 20–30% of PHTS/CS have autism or developmental delay; paediatric/adult macrocephalic ASD → PTEN germline testing (NCCN category 2A); neurodevelopmental PTEN presentation is not cancer but warrants full surveillance",
        ],
        "etiologies": [
            "Germline PTEN LOF → PIP3 accumulates → AKT constitutive phosphorylation → mTORC1 activation → S6K1/4EBP1 → protein synthesis + HIF-VEGF → renal cell carcinoma",
            "PTEN haploinsufficiency → insufficient PI3K counter-regulation → cell survival + proliferation signalling unopposed → multiorgan tumour predisposition",
            "Somatic second-hit LOH at 10q23 in tumour → biallelic PTEN loss → complete PI3K/AKT/mTOR derepression → cancer",
            "PTEN-loss epigenetic: promoter hypermethylation (somatic) as alternative mechanism in non-germline PTEN-deficient tumours → IHC loss guides identification",
        ],
        "stats": {
            "mean_dx_age": 52,
            "mean_dx_delay_months": 12,
            "renal_rr": "8× (papillary + clear cell)",
            "lifetime_risk_pct": 34,
            "everolimus_approved": True,
            "breast_risk_pct": 85,
            "thyroid_risk_pct": 38,
        },
        "dx_delay_distribution": "12–24 months (Cowden syndrome under-recognised; trichilemmomas diagnosed as skin tags; RCC found at multi-cancer workup)",
    },
]


def _make_patients(gene_info: dict, seed: int):
    rng = random.Random(seed)
    pts = []
    mean_age = gene_info["stats"]["mean_dx_age"]
    mean_delay = gene_info["stats"]["mean_dx_delay_months"]
    gene = gene_info["gene"]
    for i in range(40):
        age = max(18, min(80, int(rng.gauss(mean_age, 8))))
        delay = max(1, int(rng.gauss(mean_delay, mean_delay * 0.4)))
        sex = rng.choice(["M", "F"])
        pts.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_dx": age,
            "sex": sex,
            "dx_delay_months": delay,
            "seed": seed,
        })
    return pts


# pre-generate all patient cohorts
for _idx, _g in enumerate(RENAL_GENES):
    _g["patients"] = _make_patients(_g, SEED_BASE + _idx)


def get_overview():
    genes = []
    all_ages = []
    all_delays = []
    for g in RENAL_GENES:
        ages = [p["age_at_dx"] for p in g["patients"]]
        delays = [p["dx_delay_months"] for p in g["patients"]]
        all_ages.extend(ages)
        all_delays.extend(delays)
        genes.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "lifetime_risk_pct": g["stats"]["lifetime_risk_pct"],
            "renal_rr": g["stats"]["renal_rr"],
            "mean_dx_age": round(sum(ages) / len(ages), 1),
            "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
            "key_alerts": g["key_alerts"],
            "n_patients": len(g["patients"]),
        })
    return {
        "atlas": "Hereditary-Renal-Cancer-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Renal Cell Carcinoma Atlas",
        "seed_range": f"{SEED_BASE}–{SEED_BASE+7}",
        "total_patients": sum(len(g["patients"]) for g in RENAL_GENES),
        "aggregate_stats": {
            "mean_dx_age": round(sum(all_ages) / len(all_ages), 1),
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
            "genes_covered": len(RENAL_GENES),
            "patients_per_gene": 40,
        },
        "genes": genes,
        "top_alerts": [
            "VHL-BELZUTIFAN-FDA2021: First-in-class HIF-2α inhibitor FDA 2021 for VHL-associated ccRCC — 49.2% ORR (LITESPARK-004); LITESPARK-005 demonstrates OS benefit vs everolimus in advanced RCC",
            "FH-ANY-SIZE-SURGERY: HLRCC-RCC warrants immediate resection at ANY SIZE — no 3 cm threshold; stage I HLRCC-RCC already harbours micrometastases; bevacizumab+erlotinib 65% ORR",
            "BAP1-ANNUAL-OPHTHALMOLOGY-MANDATORY: 25–50% lifetime uveal melanoma risk in BAP1-TPDS; annual dilated fundoscopy from diagnosis; tebentafusp FDA 2022 for HLA-A*02:01+ metastatic UM",
            "SDHB-MALIGNANT-PGL-40PCT: 40% of SDHB-PGL are malignant (metastatic) — highest SDH malignancy rate; DOTATATE-PET staging mandatory; 177Lu-DOTATATE (LUTATHERA) for SSTR+ disease",
            "FLCN-FIBROFOLLICULOMAS-PNEUMOTHORAX: Fibrofolliculomas PATHOGNOMONIC for BHD; 25–33% spontaneous pneumothorax — emergency presentation; pleurodesis for recurrent events",
            "MET-GOF-PURE-RENAL-CABOZANTINIB: HPRCC type 1 papillary RCC is a pure renal phenotype (GOF MET exons 16–19); cabozantinib/savolitinib preferred; bilateral multifocal — nephron-sparing mandatory",
            "SDHA-PITUITARY-UNIQUE-SDHA-IHC: Pituitary adenoma predisposition is SDHA-specific (not SDHB/C/D); SDHA+SDHB dual IHC loss identifies SDHA mutation specifically; 1% population carrier rate",
            "PTEN-BREAST-85PCT-EVEROLIMUS: PTEN Cowden syndrome — 85% lifetime breast cancer (primary surveillance priority); 34% lifetime RCC — everolimus FDA-approved (RECORD-1); macrocephaly+trichilemmomas diagnostic",
        ],
    }


def get_breakdown():
    result = []
    for idx, g in enumerate(RENAL_GENES):
        ages = [p["age_at_dx"] for p in g["patients"]]
        delays = [p["dx_delay_months"] for p in g["patients"]]
        result.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "omim_disease": g["omim_disease"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "key_alerts": g["key_alerts"],
            "etiologies": g["etiologies"],
            "stats": g["stats"],
            "dx_delay_distribution": g["dx_delay_distribution"],
            "computed": {
                "mean_dx_age": round(sum(ages) / len(ages), 1),
                "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
                "n_patients": len(g["patients"]),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": g["patients"][:10],
        })
    return result


def get_definitions():
    return {
        "concepts": {
            "VHL — HIF Pathway and Belzutifan: Mechanism and Clinical Targets": (
                "VHL disease arises from loss of the VHL E3 ubiquitin ligase → HIF-2α accumulation → pseudohypoxia → ccRCC. "
                "HIF-2α (EPAS1) is the dominant oncogenic driver in clear cell RCC over HIF-1α: "
                "HIF-2α drives VEGF, PDGF, CCND1, TGF-α, EPO (polycythaemia), Oct4 (stem cell renewal), and Cyclin E. "
                "HIF-1α is pro-apoptotic in some contexts — net balance shifted by VHL loss. "
                "BELZUTIFAN MECHANISM: Binds HIF-2α PAS-B domain (oxygen sensor domain) → "
                "disrupts HIF-2α/ARNT (HIF-1β) dimerisation → prevents HIF-2α complex from binding HRE "
                "(hypoxia response element) DNA → reduced VEGF/PDGF/EPO transcription. "
                "LITESPARK-004 (Jonasch 2021, NEJM): VHL-associated ccRCC (n=61); ORR 49.2%; DCR 98.4%; "
                "all objective responses were partial — complete responses uncommon in VHL (multifocal); "
                "DOR not reached at 21.8 months median follow-up; "
                "Anaemia most common AE (grade 3: 7.4%) — HIF-EPO suppression; monitor haemoglobin; "
                "Fatigue, dizziness (HIF-driven EPO withdrawal symptoms). "
                "LITESPARK-005 (Choueiri 2023, NEJM): Advanced ccRCC (non-VHL) vs everolimus post-TKI; "
                "Belzutifan superior OS (not just PFS) — HR 0.74; median OS 21.4 vs 17.3 months. "
                "VHL SURVEILLANCE PRECISION: 3 cm threshold is evidence-based — "
                "NCI long-term follow-up data: tumours <3 cm have ~3% metastasis rate; "
                ">3 cm: metastasis risk increases substantially; "
                "manage bilateral disease with multiple nephron-sparing procedures; "
                "total nephrectomy last resort (triggers CKD + dialysis in bilateral disease). "
                "RETINAL HEMANGIOBLASTOMA: laser photocoagulation for small retinal lesions; "
                "PDT (photodynamic therapy) or anti-VEGF (bevacizumab intravitreal) for juxtafoveal lesions; "
                "NOT accessible by belzutifan (blood-retinal barrier); ophthalmology separately manages. "
                "HEMANGIOBLASTOMA CNS: Observe stable; resect symptomatic/growing; "
                "belzutifan shows responses in CNS hemangioblastoma (NCI ongoing data)."
            ),
            "SDH-Deficient RCC — Oncometabolic Mechanism and IHC Diagnosis": (
                "SDH-deficient RCC (WHO 2022) is defined by biallelic SDH subunit gene inactivation → "
                "complete complex II loss → oncometabolic succinate accumulation. "
                "SUCCINATE ONCOMETABOLIC EFFECTS (applies to SDHB, SDHC, SDHD, SDHA, FH-via-fumarate): "
                "(1) PHD inhibition: succinate competitively inhibits α-KG-dependent PHD1/2/3 → "
                "HIF-1α/HIF-2α stabilisation independent of VHL → pseudohypoxia → VEGF/PDGF upregulation; "
                "(2) TET2 inhibition: succinate competes with α-KG for TET2 → "
                "5mC cannot be converted to 5hmC → global DNA hypermethylation (CpG island methylator phenotype); "
                "(3) KDM inhibition: Jumonji-domain histone demethylases require α-KG → succinate inhibition → "
                "H3K9me2, H3K27me3 accumulation → gene silencing (EZH2-independent). "
                "MORPHOLOGY OF SDH-DEFICIENT RCC: "
                "Oncocytic cells (abundant mitochondria as compensatory response to complex II dysfunction); "
                "eosinophilic cytoplasmic inclusions (autophagy vacuoles — mitochondria trapped in autophagosomes) PATHOGNOMONIC; "
                "WHO 2022: distinct RCC entity recognised (not papillary or clear cell). "
                "IHC ALGORITHM FOR SDH-RCC: "
                "Step 1: SDHB IHC on all oncocytic/unclassified RCC → if SDHB lost: proceed; "
                "Step 2: SDHA IHC → if SDHA also lost: SDHA mutation likely; if SDHA preserved: SDHB/C/D mutation; "
                "Step 3: Germline SDH panel (SDHB, SDHC, SDHD, SDHA); "
                "Step 4: Somatic SDH if germline negative (rare). "
                "TREATMENT OF SDH-DEFICIENT RCC: "
                "Sunitinib (VEGFR2/1/3, PDGFR, KIT, RET, FLT3) — preferred (anti-HIF downstream VEGF); "
                "Cabozantinib (MET/VEGFR2/AXL) — second-line or if sunitinib-refractory; "
                "Everolimus — less evidence; mTOR not primary driver vs pseudohypoxia VEGF; "
                "mTOR + VEGFR combination may provide additive benefit (limited data)."
            ),
            "FH-HLRCC: Why Every Millimetre Matters": (
                "HLRCC-associated RCC (Type 2C, WHO papillary or collecting duct-like) is categorically "
                "different from VHL-ccRCC or BHD-hybrid in its aggressiveness: "
                "KEY EVIDENCE FOR ANY-SIZE SURGERY: "
                "(1) Linehan 2018 NEJM case series: HLRCC-RCC nodal/systemic metastases documented at stage I (<3 cm) lesions; "
                "(2) Somatic mutations in HLRCC-RCC: additional TP53 mutation, MET amplification, CDH1 loss — aggressive biology; "
                "(3) Median OS from RCC diagnosis (advanced): <2 years (worse than any other hereditary RCC subtype). "
                "BEVACIZUMAB + ERLOTINIB MECHANISM IN HLRCC: "
                "FH loss → fumarate → KEAP1 succination → NRF2 constitutive activation → "
                "HER1/EGFR upregulation (NRF2 target) + HIF-1α (PHD inhibition) → VEGF upregulation; "
                "DUAL BLOCKADE: Bevacizumab (anti-VEGF-A, VEGFR-independent) + erlotinib (EGFR-TKI); "
                "MDACC Phase II (Srinivasan 2018, Lancet Oncol): ORR 65%, PFS 21 months — "
                "best reported outcomes for any hereditary RCC subtype in the advanced setting; "
                "VEGFR TKIs alone (sunitinib): ORR ~20–30% in HLRCC (inferior to bev+erl); "
                "Cabozantinib (MET — secondarily upregulated in HLRCC via HIF-1α): limited HLRCC-specific data. "
                "SURVEILLANCE TIMING FOR HLRCC: "
                "Annual MRI total body from age 8–10 (unlike VHL age 15, BHD age 20–25); "
                "Earlier onset than other hereditary RCC (paediatric cases documented); "
                "Female carriers: annual uterine MRI from age 18 (fibroid onset age 20s)."
            ),
            "BAP1-TPDS and Uveal Melanoma: Tebentafusp and Annual Ophthalmology": (
                "BAP1-TPDS is the paradigmatic multi-cancer predisposition syndrome requiring multi-speciality co-ordination: "
                "UVEAL MELANOMA IN BAP1-TPDS: "
                "UM represents the highest-penetrance BAP1-TPDS cancer (25–50% vs 15% RCC); "
                "BAP1-associated UM: class 2 by gene expression profiling (high metastatic risk); "
                "liver-tropic metastases in ~50% within 5 years of primary UM; "
                "SURVEILLANCE: Annual dilated fundoscopy (ophthalmology) + anterior segment photography; "
                "OCT (optical coherence tomography) for posterior pole lesions; "
                "no circulating UM ctDNA validated for screening. "
                "TEBENTAFUSP (IMCgp100): "
                "FDA approved January 2022 for HLA-A*02:01+ unresectable/metastatic uveal melanoma; "
                "mechanism: bispecific T-cell receptor/anti-CD3 molecule (ImmTAC); "
                "gp100 peptide (tumour antigen) + anti-CD3 → redirects T cells to kill gp100+ UM cells; "
                "HLA-A*02:01 prevalence: ~50% of European-ancestry patients (test ALL BAP1-UM patients at diagnosis); "
                "PIVOT-002 trial (Nathan 2021, NEJM): 1L metastatic UM (n=378); "
                "1-year OS 73% (tebentafusp) vs 59% (investigator choice); HR 0.51; p<0.001; "
                "First OS benefit in metastatic UM — historically median OS 6–12 months; "
                "ADVERSE EVENTS: cytokine release syndrome (73%), skin reactions (rash, pruritus 84%), "
                "liver toxicity — manage with corticosteroids; pre-medicate each infusion. "
                "RCC IN BAP1-TPDS: ccRCC treatment with sunitinib/pazopanib standard; "
                "immune checkpoint inhibitors (nivolumab+ipilimumab) rationale: "
                "BAP1-mutated RCC may have elevated tumour mutational burden vs VHL-mutated RCC; "
                "limited BAP1-specific RCC trial data. "
                "MESOTHELIOMA IN BAP1-TPDS: "
                "CheckMate 743 (Baas 2021, Lancet): nivolumab+ipilimumab vs platinum/pemetrexed (1L mesothelioma); "
                "OS benefit in non-epithelioid histology (sarcomatoid/biphasic) — BAP1-mutated often non-epithelioid; "
                "BAP1-mesothelioma CT screening protocol not established — no recommendation beyond standard clinical triggers."
            ),
            "PTEN/Cowden Syndrome — mTOR Pharmacology and Multi-Organ Surveillance": (
                "PTEN-loss creates constitutive mTORC1 activation — the strongest pharmacological rationale for "
                "mTOR inhibitors in any hereditary cancer syndrome: "
                "mTOR INHIBITOR CLASSES: "
                "Rapalogs (allosteric mTORC1 inhibitors): everolimus (RECORD-1, BOLERO-2), temsirolimus (ARCC trial); "
                "bind FKBP12 → complex inhibits mTORC1 Raptor-FRB domain; "
                "incomplete mTORC2 inhibition (AKT S473 escapes → AKT rebound → resistance); "
                "Next-generation: mTOR kinase inhibitors (MLN0128, AZD8055) — inhibit both mTORC1+mTORC2 → "
                "no AKT rebound; Phase I/II in PTEN-solid tumours. "
                "RECORD-1 TRIAL (Motzer 2010, Lancet): everolimus vs placebo post-sunitinib/sorafenib failure; "
                "PFS 4.9 vs 1.9 months; HR 0.33; p<0.001; FDA approved March 2009; "
                "PTEN-loss subset: higher everolimus benefit (exploratory subanalysis). "
                "PTEN RENAL CANCER — HISTOLOGY: "
                "Predominantly papillary type 1 and conventional ccRCC; "
                "oncocytic tumours less common; "
                "PTEN-loss IHC in tumour validates somatic second hit. "
                "COWDEN SYNDROME DIAGNOSTIC CRITERIA (NCCN v2.2024): "
                "Pathognomonic criteria: adult Lhermitte-Duclos (DGC); mucocutaneous lesions alone if "
                "≥6 facial papules (≥3 trichilemmomas); ≥3 trichilemmomas; ≥2 major cutaneous criteria; "
                "1 major + ≥3 minor criteria; ≥4 minor criteria. "
                "MAJOR CRITERIA: breast ca; follicular thyroid ca; macrocephaly (OFC >97th percentile); "
                "mismatch repair tumours; endometrial carcinoma. "
                "MINOR CRITERIA: autism/intellectual disability; colorectal carcinoma; "
                "oesophageal glycogenic acanthosis (≥3); lipomas (≥3); renal cell carcinoma; "
                "testicular lipomatosis; thyroid structural lesions; vascular abnormalities. "
                "AUTISTIC MACROCEPHALY TESTING RULE (NCCN): "
                "Any child with autism + macrocephaly (OFC >97th) → PTEN germline testing; "
                "PTEN mutations found in ~7–17% of autistic children with macrocephaly vs <1% general ASD."
            ),
        },
        "pharmacological_distinctions": [
            "Belzutifan (VHL/HIF-2α) vs sunitinib (VEGFR/PDGFR): Belzutifan is upstream (directly blocks HIF-2α transcription factor) while sunitinib is downstream (blocks VEGFR-mediated angiogenesis); Belzutifan for VHL-associated disease (FDA-approved); sunitinib for SDH-deficient, MET-driven, or PTEN-associated RCC where HIF-2α is not the sole driver; do NOT substitute sunitinib for belzutifan in VHL-ccRCC (inferior efficacy for VHL-specific disease)",
            "Bevacizumab+erlotinib (FH-HLRCC) vs VEGFR-TKIs (VHL, SDH-RCC): HLRCC-specific combination exploits dual FH-loss → NRF2-EGFR + HIF-VEGF axes; VEGFR-TKIs alone achieve only 20–30% ORR in HLRCC vs 65% for bev+erl; mechanism-matched therapy; do NOT use sunitinib as first-line for HLRCC-RCC without bevacizumab+erlotinib consideration",
            "Everolimus/temsirolimus (PTEN/FLCN mTORC1) vs sunitinib (SDHB/FH anti-VEGF): mTOR inhibitors rational in PTEN-loss (mTORC1 constitutive, FDA RECORD-1) and FLCN-loss (mTORC1 via AMPK-FLCN axis); sunitinib rational in SDH-deficient RCC (succinate → HIF → VEGF downstream); cross-applying mTOR inhibitors to SDH-RCC or VEGFR-TKIs to PTEN-RCC is mechanistically suboptimal but used empirically",
            "Cabozantinib/savolitinib (MET-HPRCC) vs anti-HIF pathway agents (VHL-ccRCC): HPRCC type 1 is a GOF-MET-driven papillary RCC; MET kinase inhibitors are mechanistically first-choice; cabozantinib (multi-kinase: MET+VEGFR2+AXL) FDA-approved for advanced RCC; savolitinib (selective MET) shows 27% ORR in MET-driven papillary RCC (SAVOIR trial); belzutifan is NOT indicated for HPRCC (papillary RCC — HIF-2α not the primary driver)",
            "Tebentafusp (BAP1-uveal melanoma) vs nivolumab+ipilimumab (BAP1-mesothelioma): tebentafusp is ONLY for uveal melanoma in HLA-A*02:01+ patients (FDA 2022); NOT applicable to BAP1-ccRCC or mesothelioma; nivolumab+ipilimumab (CheckMate 743) is the preferred first-line for non-epithelioid mesothelioma (BAP1-associated); pembrolizumab for ccRCC regardless of BAP1 status (CheckMate 214 extrapolation); match immunotherapy to the specific tumour type, not the gene alone",
        ],
        "key_standards": [
            "NCCN Kidney Cancer (v2.2024): germline testing recommended for bilateral/multifocal RCC, early-onset (<46 years), family history, or histology consistent with hereditary syndrome (papillary, chromophobe, oncocytic); VHL testing for ccRCC with hemangioblastoma/polycythaemia",
            "FDA Belzutifan (MK-6482) August 2021: for VHL-associated ccRCC not requiring immediate surgery; also approved in LITESPARK-005 context for advanced ccRCC post-TKI/IO progression",
            "FDA Tebentafusp (IMCgp100) January 2022: for HLA-A*02:01+ unresectable/metastatic uveal melanoma; first-ever OS benefit in metastatic UM; BAP1-TPDS patients require HLA typing at UM diagnosis",
            "WHO 2022 Classification of Urinary and Male Genital Tumours: SDH-deficient RCC recognised as distinct entity (biallelic SDH loss); FH-deficient RCC (HLRCC-associated); BAP1-inactivated RCC; MiT/TFE translocation RCC",
            "VHL Alliance Surveillance Guidelines (2020): annual MRI brain+spine+abdomen; annual ophthalmology; annual biochemistry (metanephrines); audiogram biennial; Belzutifan as systemic option for unresectable/progressive VHL-RCC",
        ],
    }
