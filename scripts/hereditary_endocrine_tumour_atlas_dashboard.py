#!/usr/bin/env python3
"""Hereditary-Endocrine-Tumour-Atlas — Complete 8-Gene Hereditary Endocrine Tumour Syndrome Atlas
MEN1    (Menin; 610 aa; 11q13.1; AD; Multiple Endocrine Neoplasia type 1;
         parathyroid adenoma 95% + pituitary 40% + pancreatic NET 70%;
         PHPT PRESENTING 90%; annual surveillance mandatory; sunitinib/everolimus for pNETs;
         seed SEED_BASE+0) ·
RET     (RET proto-oncogene; 1114 aa; 10q11.21; AD GOF; MEN2A/MEN2B/FMTC;
         MEN2A: MTC 100% + pheo 50% + parathyroid 25%; C634 hotspot — pheo risk;
         PROPHYLACTIC THYROIDECTOMY IN CHILDHOOD codon-risk stratified;
         cabozantinib/vandetanib for advanced MTC; seed SEED_BASE+1) ·
VHL     (Von Hippel-Lindau; 213 aa; 3p25.3; AD; VHL syndrome;
         ccRCC + cerebellar hemangioblastoma + retinal angioma + pheo + PNET + endolymphatic sac tumour;
         ANNUAL MULTI-ORGAN SURVEILLANCE MANDATORY; belzutifan FDA 2021 HIF-2α inhibitor;
         seed SEED_BASE+2) ·
SDHB    (Succinate dehydrogenase subunit B; 280 aa; 1p36.13; AD; PGL/PCC syndrome type 4;
         HIGHEST MALIGNANT POTENTIAL 30-50% metastatic; extra-adrenal predominant;
         SDHB IHC staining ABSENT = SDH-deficient tumour; selpercatinib emerging;
         seed SEED_BASE+3) ·
SDHD    (Succinate dehydrogenase subunit D; 159 aa; 11q23.1; AD imprinted (paternal);
         PGL/PCC syndrome type 1; head/neck paraganglioma PREDOMINANT — carotid body + jugulotympanic;
         MATERNAL INHERITANCE DOES NOT CAUSE DISEASE; multifocal in 40%;
         seed SEED_BASE+4) ·
CDKN1B  (p27/Kip1; 198 aa; 12p13.1; AD; MEN4;
         MEN1-like phenotype without MEN1 mutation — parathyroid + pituitary;
         sequence MEN1 FIRST — CDKN1B testing if MEN1 negative with MEN1 phenotype;
         seed SEED_BASE+5) ·
CDC73   (Parafibromin; 531 aa; 1q31.2; AD; HPT-JT syndrome / parathyroid carcinoma;
         PARATHYROID CARCINOMA 15% — HIGHEST AMONG HYPERPARATHYROID SYNDROMES;
         jaw ossifying fibroma PATHOGNOMONIC FOR HPT-JT; serum Ca ≥ 3.0 mmol/L;
         seed SEED_BASE+6) ·
PRKAR1A (PKA regulatory subunit 1-alpha; 381 aa; 17q24.2; AD; Carney complex;
         CARDIAC MYXOMA — EMBOLIC STROKE RISK — SURGICAL EXCISION URGENT;
         spotty pigmentation + myxoma + PPNAD + acromegaly + Sertoli cell tumour;
         annual echocardiography MANDATORY; seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1566–1573)
"""

import random

SEED_BASE = 1566

ENDOCRINE_TUMOUR_GENES = [
    # ── MEN1 — Multiple Endocrine Neoplasia type 1 ──────────────────────
    {
        "gene": "MEN1",
        "protein": "Menin — Multiple Endocrine Neoplasia type 1, Parathyroid + Pituitary + Pancreatic NETs",
        "alias": (
            "MEN1; OMIM gene 613733; MEN1 OMIM 131100; 11q13.1; 610 aa; ~68 kDa; "
            "AD; prevalence 1:30,000; tumour suppressor (two-hit Knudson model). "
            "Menin is a nuclear scaffold protein that regulates gene expression via "
            "histone methyltransferase (MLL complex H3K4me3), TGF-β signalling, "
            "and cell cycle control. Loss of function → loss of tumour suppression in "
            "parathyroid, pituitary, and pancreatic/duodenal endocrine cells. "
            "Clinical triad (2 of 3 to diagnose): (1) Primary hyperparathyroidism (PHPT) "
            "— most common feature 95%, multiglandular, first manifestation at age 20-25; "
            "(2) Pituitary adenoma — 40%, prolactinoma most common (60%), then GH/ACTH; "
            "(3) Pancreatic/duodenal NET — 70%, gastrinoma (Zollinger-Ellison) most common, "
            "insulinoma 2nd; non-functional NETs common. "
            "Penetrance >95% by age 50. 90% familial MEN1 has MEN1 mutation; 10% de novo. "
            "Surveillance: annual Ca/PTH/fasting gastrin/chromogranin A; "
            "pituitary MRI every 3 years; EUS/MRI pancreas/duodenum annual if NET present. "
            "Treatment: parathyroidectomy (3.5 gland + autotransplant or total + reimplant); "
            "PPI for Zollinger-Ellison; diazoxide for insulinoma; "
            "sunitinib/everolimus for progressive pancreatic NETs (FDA approved). "
            "Cinacalcet for PHPT when surgery contraindicated. "
            "No prophylactic pancreatectomy. Pasireotide for ACTH-secreting. "
            "Genetic testing: all first-degree relatives of index case."
        ),
        "aa": "610 aa",
        "kDa": "~68 kDa",
        "locus": "11q13.1",
        "omim_gene": 613733,
        "omim_disease": 131100,
        "inheritance": "AD (two-hit); penetrance >95% by age 50; 10% de novo",
        "gene_class": (
            "MEN1 encodes menin, a 610-aa nuclear scaffold protein with no intrinsic enzymatic activity. "
            "Key domains: JunD-binding (N-terminal), MLL1/2 interaction (N-terminal), "
            "β-catenin interaction, FANCD2 interaction, NF-κB binding. "
            "Tumour suppressor — second hit required (somatic LOH at 11q13.1). "
            "Over 1,800 germline variants known; truncating > missense; no hotspot. "
            "MEN1 mutations ≠ VHL/RET/SDH — no strong genotype-phenotype correlation."
        ),
        "n_patients": 40,
        "key_alerts": [
            "MEN1-PHPT-FIRST: Primary hyperparathyroidism (multiglandular) is the first and most common feature — 95% by age 40; serum Ca + PTH annually from age 15",
            "MEN1-PARATHYROIDECTOMY: 3.5-gland or total parathyroidectomy + autotransplant — NOT single-gland; multiglandular disease is the rule",
            "MEN1-ZOLLINGER-ELLISON: Gastrinoma causes peptic ulcer diathesis — fasting gastrin >10x ULN + pH<2 DIAGNOSTIC; PPI mandatory; surgical cure rare in MEN1-ZES",
            "MEN1-INSULINOMA-DIAZOXIDE: Insulinoma causes hypoglycaemia — diazoxide (KATP opener) as bridge; enucleation surgical; streptozocin for malignant",
            "MEN1-NET-SURVEILLANCE: Annual EUS + MRI pancreas/duodenum for NET ≥10mm or growing — risk of metastasis; sunitinib/everolimus FDA-approved for progressive pNETs",
            "MEN1-PITUITARY-MRI: Pituitary MRI every 3 years — prolactinoma commonest (dopamine agonist); GH → IGF-1 + lanreotide; ACTH → pasireotide",
            "MEN1-CASCADE-TESTING: All first-degree relatives require germline testing — penetrance >95% by 50; start annual biochemical surveillance at index family age of diagnosis",
            "MEN1-NO-PROPHYLACTIC-PANCREATECTOMY: Prophylactic pancreatectomy is NOT indicated — morbidity high; surveillance-guided resection of lesions ≥20mm or functioning NETs",
        ],
        "etiologies": {
            "Truncating (nonsense/frameshift)": 16,
            "Missense": 12,
            "Splice site": 6,
            "Large deletion": 4,
            "In-frame deletion/insertion": 2,
        },
        "stats": {
            "phpt_pct": 95,
            "pituitary_adenoma_pct": 40,
            "pancreatic_net_pct": 70,
            "gastrinoma_pct": 40,
            "insulinoma_pct": 10,
            "malignant_net_pct": 25,
            "mean_dx_age": 25,
            "mean_dx_delay_months": 36,
        },
        "dx_delay_distribution": {"<12m": 8, "12-36m": 14, "36-60m": 12, ">60m": 6},
    },

    # ── RET — Multiple Endocrine Neoplasia type 2 ────────────────────────
    {
        "gene": "RET",
        "protein": "RET Proto-oncogene — MEN2A/MEN2B/FMTC, Prophylactic Thyroidectomy Codon-Stratified",
        "alias": (
            "RET; OMIM gene 164761; MEN2A OMIM 171400; MEN2B OMIM 162300; FMTC OMIM 155240; "
            "10q11.21; 1114 aa; ~120 kDa; AD (GOF); prevalence MEN2A 1:35,000; de novo 50% MEN2B. "
            "RET encodes a receptor tyrosine kinase expressed in neural crest-derived cells. "
            "GOF mutations → constitutive kinase activation → uncontrolled proliferation. "
            "MEN2A: cysteine-rich domain mutations (C609, C611, C618, C620, C634) → disulfide bond "
            "disruption → ligand-independent dimerisation. C634R/W → pheo risk 50%; C634 → HPT 25%. "
            "MEN2B: kinase domain M918T (95%) → substrate specificity shift; most aggressive MTC; "
            "neonatal diagnosis critical. FMTC: isolated hereditary MTC without pheo/HPT. "
            "ATA risk stratification (A→D highest): "
            "A (FMTC variants) → thyroidectomy by age 5; "
            "B (C634, rare) → by age 5; "
            "C (C634F/R/G/S/W/Y, C634Y) → by age 5 or when calcitonin elevated; "
            "D (M918T-MEN2B) → within 6 months of life (URGENT). "
            "RET-mutant tumour after progression: cabozantinib (FDA 2012), vandetanib (FDA 2011); "
            "selpercatinib (FDA 2020) for RET-altered cancers. "
            "Annual: calcitonin + CEA (MTC marker); 24h urine catecholamines (pheo screen); "
            "Ca/PTH (HPT screen). Adrenal imaging before thyroid surgery (pheo first rule)."
        ),
        "aa": "1114 aa",
        "kDa": "~120 kDa",
        "locus": "10q11.21",
        "omim_gene": 164761,
        "omim_disease": 171400,
        "inheritance": "AD GOF; penetrance near 100% for MTC; M918T 50% de novo",
        "gene_class": (
            "RET encodes a receptor tyrosine kinase (RTK): signal peptide → cysteine-rich ectodomain "
            "(cadherin-like) → transmembrane → cytoplasmic kinase domain. "
            "Normal activation: GDNF family ligands + GFRα co-receptors → RET dimerisation → "
            "autophosphorylation → PI3K/AKT, MAPK, JAK/STAT signalling. "
            "GOF mutations: extracellular cysteine mutations (MEN2A) → disulfide mispairing → "
            "constitutive dimers; M918T (MEN2B) → altered substrate specificity (substrate shift "
            "from Tyr to Phe). Hotspot codons: C634 (MEN2A most common, highest pheo risk), "
            "M918T (MEN2B, most aggressive)."
        ),
        "n_patients": 40,
        "key_alerts": [
            "RET-M918T-MEN2B-URGENT: M918T (MEN2B) — prophylactic thyroidectomy within 6 MONTHS of birth; most aggressive MTC; marfanoid habitus + mucosal neuromas + ganglioneuromatosis",
            "RET-PHEO-BEFORE-THYROID-SURGERY: SCREEN FOR PHAEOCHROMOCYTOMA (24h urine catecholamines) BEFORE thyroid surgery — pheo causes perioperative crisis if unrecognised",
            "RET-C634-HIGHEST-PHEO-RISK: Codon 634 mutations (MEN2A) carry 50% lifetime pheo risk — annual 24h urine metanephrines/catecholamines mandatory; adrenal imaging if elevated",
            "RET-PROPHYLACTIC-THYROIDECTOMY-CODON-STRATIFIED: Timing is ATA risk-stratified: D (M918T) <6m · C (C634) by age 5 · B (other MEN2A) by age 5 · A (FMTC) by 5y or calcitonin-guided",
            "RET-CALCITONIN-CEA-ANNUAL: Calcitonin doubling time <6m = aggressive MTC; CEA + calcitonin together — CEA elevation with calcitonin normalisation = dedifferentiation",
            "RET-SELPERCATINIB-FDA2020: Selpercatinib (RET kinase inhibitor) FDA 2020 for advanced RET-mutant MTC — first highly selective RET inhibitor; CNS penetration",
            "RET-HPT-MEN2A-PARATHYROID: 25% of MEN2A develop HPT — intraoperative PTH monitoring; single-gland adenoma common (unlike MEN1 where multiglandular is the rule)",
            "RET-CASCADE-TESTING: All first-degree relatives require germline RET testing; paediatric testing recommended — penetrance near 100% for MTC",
        ],
        "etiologies": {
            "C634R/W/Y/G/S/F (MEN2A-pheo)": 18,
            "M918T (MEN2B)": 8,
            "C609/611/618/620 (MEN2A)": 8,
            "FMTC-only variants (E768/L790/V804/S891)": 6,
        },
        "stats": {
            "mtc_pct": 100,
            "pheo_pct": 50,
            "hpt_pct": 25,
            "men2b_pct": 20,
            "c634_mutation_pct": 45,
            "prophylactic_thyroidectomy_done_pct": 62,
            "mean_dx_age": 22,
            "mean_dx_delay_months": 24,
        },
        "dx_delay_distribution": {"<12m": 12, "12-36m": 14, "36-60m": 8, ">60m": 6},
    },

    # ── VHL — Von Hippel-Lindau syndrome ─────────────────────────────────
    {
        "gene": "VHL",
        "protein": "Von Hippel-Lindau — ccRCC + Hemangioblastoma + Retinal Angioma + Pheo + PNET + ELST",
        "alias": (
            "VHL; OMIM gene 608537; VHL OMIM 193300; 3p25.3; 213 aa; ~30 kDa; "
            "AD; prevalence 1:36,000; penetrance >95% by age 65. "
            "pVHL is an E3 ubiquitin ligase that targets HIF-1α/HIF-2α for proteasomal "
            "degradation in normoxia (pVHL → elongin C/B → Cul2 → Rbx1 → HIF ubiquitination). "
            "Loss of VHL → pseudohypoxic state → HIF-2α upregulation → VEGF/EPO/PDGF overproduction "
            "→ pathological angiogenesis (haemangioblastoma, ccRCC, ELST) + polycythaemia. "
            "Type 1 (truncating/missense with LOH): ccRCC risk HIGH; pheo LOW; "
            "Type 2A (C162/R167): pheo HIGH; ccRCC LOW; "
            "Type 2B (Y98H/R167W): pheo HIGH; ccRCC HIGH (highest risk type); "
            "Type 2C (A149V): pheo ONLY (no ccRCC, no hemangioblastoma). "
            "Annual surveillance: ophthalmology (retinal angioma — first manifestation 25%); "
            "MRI brain/spine (hemangioblastoma); abdominal MRI (ccRCC + PNET + pheo/PGL); "
            "24h urine metanephrines (pheo). "
            "Belzutifan (HIF-2α inhibitor, FDA 2021) for VHL-associated ccRCC/PNET/hemangioblastoma "
            "not requiring immediate surgery. "
            "Retinal angioma: laser photocoagulation or anti-VEGF. "
            "ccRCC: partial nephrectomy preferred (bilateral disease risk); "
            "treat when >3cm or growing. Endolymphatic sac tumour (ELST): hearing loss + tinnitus."
        ),
        "aa": "213 aa",
        "kDa": "~30 kDa",
        "locus": "3p25.3",
        "omim_gene": 608537,
        "omim_disease": 193300,
        "inheritance": "AD (two-hit tumour suppressor); penetrance >95% by 65; rare de novo",
        "gene_class": (
            "VHL encodes an E3 ubiquitin ligase adaptor protein (213 aa). Two functional domains: "
            "β-domain (HIF-α binding, oxygen-sensing) and α-domain (elongin C binding). "
            "Under normoxia: prolyl hydroxylases (PHD1-3) hydroxylate HIF-α → pVHL recognises "
            "hydroxy-Pro564/Pro402 → ubiquitination → 26S proteasomal degradation. "
            "Under hypoxia or VHL loss: HIF-α accumulates → transcribes VEGF, EPO, GLUT1, PDGFβ, "
            "TGF-α. HIF-2α is the key oncogenic target (more than HIF-1α in ccRCC/hemangioblastoma). "
            "Genotype-phenotype: type 2C pVHL (pheo only) retains HIF binding but loses elongin C → "
            "impairs MAPK negative regulation selectively in chromaffin cells."
        ),
        "n_patients": 40,
        "key_alerts": [
            "VHL-RETINAL-ANGIOMA-FIRST: Retinal haemangioblastoma is often the FIRST manifestation (mean age 25) — annual ophthalmology mandatory from age 5; laser/anti-VEGF before vision loss",
            "VHL-BELZUTIFAN-FDA2021: Belzutifan (HIF-2α inhibitor) FDA 2021 — for VHL-associated RCC/PNET/hemangioblastoma not requiring immediate surgery; tumour shrinkage in 64%",
            "VHL-PARTIAL-NEPHRECTOMY: Partial nephrectomy preferred over radical — bilateral ccRCC risk over lifetime; treat lesions >3cm OR growing >0.5cm/year; nephron-sparing is critical",
            "VHL-TYPE2B-HIGHEST-RISK: VHL type 2B mutations (Y98H, R167W) — HIGH pheo + HIGH ccRCC risk; most aggressive phenotype; aggressive surveillance",
            "VHL-PHEO-BEFORE-ABDOMINAL-SURGERY: Screen for phaeochromocytoma (24h urine metanephrines) BEFORE any abdominal surgery — pheo crisis is life-threatening perioperatively",
            "VHL-ELST-HEARING-LOSS: Endolymphatic sac tumour (ELST) — unilateral hearing loss + tinnitus; MRI temporal bone; early surgery before hearing loss is complete; 10% VHL",
            "VHL-ANNUAL-BRAIN-SPINE-MRI: CNS haemangioblastoma — cerebellum/brainstem/spinal cord; annual MRI from age 11; surgery for symptomatic or growing lesions; bevacizumab palliative",
            "VHL-CASCADE-TESTING-ALL-RELATIVES: All first-degree relatives require germline VHL testing and multi-organ annual surveillance — penetrance >95% by age 65",
        ],
        "etiologies": {
            "Type 1 (truncating/deletion — ccRCC dominant)": 16,
            "Type 2A (missense C162/R167 — pheo dominant)": 8,
            "Type 2B (Y98H/R167W — pheo + ccRCC)": 10,
            "Type 2C (A149V/L188V — pheo only)": 6,
        },
        "stats": {
            "rcc_pct": 75,
            "hemangioblastoma_pct": 80,
            "retinal_angioma_pct": 70,
            "pheo_pct": 30,
            "pnet_pct": 15,
            "elst_pct": 10,
            "mean_dx_age": 26,
            "mean_dx_delay_months": 30,
        },
        "dx_delay_distribution": {"<12m": 10, "12-36m": 15, "36-60m": 10, ">60m": 5},
    },

    # ── SDHB — SDH paraganglioma/pheo syndrome type 4 ───────────────────
    {
        "gene": "SDHB",
        "protein": "Succinate Dehydrogenase Subunit B — Highest Malignant Potential 30-50% Metastatic, SDHB IHC",
        "alias": (
            "SDHB; OMIM gene 185470; PGL4 OMIM 115310; 1p36.13; 280 aa; ~32 kDa; "
            "AD; prevalence overall SDH-PGL/PCC ~1:100,000. "
            "SDHB encodes the iron-sulfur subunit of succinate dehydrogenase (SDH, Complex II). "
            "SDH: succinate → fumarate in TCA cycle, feeding electrons to ubiquinone (respiratory chain). "
            "SDH loss → succinate accumulation → inhibition of PHD → pseudo-HIF activation → "
            "oncometabolic phenotype (similar to IDH/FH mutations). "
            "SDHB has the HIGHEST malignant potential among SDH genes: 30-50% of SDHB-PGL are "
            "metastatic at presentation or follow-up; metastatic = paragangliomatic tissue in "
            "bone/liver/lung/lymph nodes (not invasion alone). "
            "Location: extra-adrenal paraganglioma predominant — abdominal/pelvic; "
            "head/neck less common (15%); adrenal pheo in 25%. "
            "SDHB IHC: loss of SDHB staining in tumour cells DIAGNOSTIC of any SDH-deficient tumour "
            "(all SDHB/C/D/A mutations → SDHB protein loss); retained staining = SDH-intact. "
            "Annual surveillance: MRI whole-body (DW-MRI preferred); 24h urine/plasma metanephrines; "
            "Ga-DOTATATE PET (most sensitive for metastatic PGL). "
            "Treatment: surgical resection (curative); 177Lu-DOTATATE (Lutathera) for progressive "
            "metastatic SDH-PGL (PRRT); sunitinib + temozolomide; "
            "131I-MIBG (catecholamine-secreting). "
            "SDHB testing: ALL paraganglioma/pheo patients regardless of family history — "
            "30-40% of SDHB-PGL are apparently sporadic."
        ),
        "aa": "280 aa",
        "kDa": "~32 kDa",
        "locus": "1p36.13",
        "omim_gene": 185470,
        "omim_disease": 115310,
        "inheritance": "AD (two-hit tumour suppressor); incomplete penetrance ~30-40% lifetime",
        "gene_class": (
            "SDHB encodes the iron-sulfur (Fe-S) subunit of succinate dehydrogenase (SDH/Complex II). "
            "SDH structure: SDHA (flavoprotein) + SDHB (Fe-S) form the catalytic heterodimer; "
            "SDHC + SDHD anchor to inner mitochondrial membrane (Q-site). "
            "SDHB contains three Fe-S clusters ([2Fe-2S], [4Fe-2S], [4Fe-4S]) for electron transport. "
            "SDHB loss → SDHB protein lost from all four subunits → SDHB IHC negative. "
            "Oncometabolism: succinate → inhibits PHD → HIF stabilisation + DNA hypermethylation "
            "(succinate inhibits α-KG–dependent dioxygenases including TET demethylases and "
            "Jumonji histone demethylases → hypermethylator phenotype)."
        ),
        "n_patients": 40,
        "key_alerts": [
            "SDHB-HIGHEST-MALIGNANT-POTENTIAL: 30-50% of SDHB-PGL are metastatic — HIGHEST malignancy risk among SDH genes; metastasis = extra-paraganglionic tissue (bone/liver/lung/lymph nodes)",
            "SDHB-IHC-DIAGNOSTIC: SDHB immunohistochemistry on tumour tissue — ABSENT staining = ANY SDH-deficient tumour (SDHB/C/D/A mutations all cause SDHB protein loss); essential first-line test",
            "SDHB-EXTRA-ADRENAL-PREDOMINANT: Abdominal/pelvic extra-adrenal PGL most common (60%); retroperitoneal + organ-of-Zuckerkandl; whole-body MRI or Ga-DOTATATE PET mandatory",
            "SDHB-GA-DOTATATE-PET: Ga-68-DOTATATE PET/CT most sensitive for metastatic SDH-PGL (sensitivity 86-100%); replaces MIBG for non-catecholamine-secreting tumours; 177Lu-DOTATATE PRRT",
            "SDHB-SPORADIC-PRESENTATION: 30-40% of SDHB-PGL present apparently sporadically — germline SDHB testing in ALL paraganglioma/phaeochromocytoma patients regardless of family history",
            "SDHB-ANNUAL-DW-MRI: Annual DW-whole-body MRI (+ Ga-DOTATATE PET every 2-3 years) — surveillance for recurrence/new lesions/metastasis; biochemical: plasma metanephrines annually",
            "SDHB-SUNITINIB-TEMOZOLOMIDE: For progressive metastatic SDHB-PGL: sunitinib (antiangiogenic) + temozolomide (alkylating); 177Lu-DOTATATE (PRRT) FDA 2018 for progressive SSTR+ NETs",
            "SDHB-CASCADE-TESTING: All first-degree relatives require SDHB germline testing — lifetime risk ~30-40%; surveillance from childhood (tumours reported in paediatric SDHB carriers)",
        ],
        "etiologies": {
            "Missense (Fe-S cluster domain)": 16,
            "Truncating/frameshift": 12,
            "Splice site": 6,
            "Large deletion": 6,
        },
        "stats": {
            "metastatic_pct": 40,
            "extra_adrenal_pct": 60,
            "adrenal_pheo_pct": 25,
            "head_neck_pgl_pct": 15,
            "catecholamine_secreting_pct": 50,
            "sdhb_ihc_negative_pct": 98,
            "mean_dx_age": 34,
            "mean_dx_delay_months": 18,
        },
        "dx_delay_distribution": {"<12m": 14, "12-36m": 12, "36-60m": 8, ">60m": 6},
    },

    # ── SDHD — SDH paraganglioma/pheo syndrome type 1 (imprinted) ───────
    {
        "gene": "SDHD",
        "protein": "Succinate Dehydrogenase Subunit D — Head/Neck PGL Predominant, Paternal Imprinting, Multifocal 40%",
        "alias": (
            "SDHD; OMIM gene 602690; PGL1 OMIM 168000; 11q23.1; 159 aa; ~17 kDa; "
            "AD (paternally imprinted — disease expressed ONLY from paternal allele). "
            "SDHD encodes the small membrane-anchoring subunit of SDH (Complex II). "
            "Key distinguishing feature: MATERNAL INHERITANCE DOES NOT CAUSE DISEASE — "
            "maternal SDHD mutations are epigenetically silenced; only paternal transmission → "
            "disease. Exception: if maternal deletion covers imprinting centre at 11p15.5 (rare). "
            "Head/neck paraganglioma (HN-PGL) PREDOMINANT: carotid body tumour (CBT), "
            "jugulotympanic PGL, vagal PGL, tympanic PGL — these are typically non-secreting "
            "(silent biochemically in 70%). Pulsatile tinnitus + cranial nerve palsies. "
            "Multifocal disease in 40% (multiple simultaneous head/neck ± abdominal tumours). "
            "Malignancy rate LOWER than SDHB (~10-15%) but adrenal pheo in 20%. "
            "Annual surveillance: MRI head/neck + abdomen; plasma metanephrines + chromogranin A; "
            "consider Ga-DOTATATE PET for staging. "
            "Treatment: surgery (carotid body tumour — embolisation first to reduce bleeding); "
            "stereotactic radiosurgery for skull base jugulotympanic PGL; "
            "active surveillance for small non-secreting HN-PGL (growth rate 0.8-1.0 mm/year). "
            "SDHB IHC: ABSENT in all SDH-deficient tumours including SDHD."
        ),
        "aa": "159 aa",
        "kDa": "~17 kDa",
        "locus": "11q23.1",
        "omim_gene": 602690,
        "omim_disease": 168000,
        "inheritance": "AD; PATERNAL IMPRINTING — maternal carriers do NOT manifest disease; penetrance ~50-80% in paternal transmission",
        "gene_class": (
            "SDHD encodes the cytochrome b small subunit (159 aa), one of two membrane-anchoring subunits "
            "of SDH (along with SDHC). SDHD contains three transmembrane segments anchoring the complex "
            "to the mitochondrial inner membrane and coordinates the ubiquinol-binding (Q-) site for "
            "electron transfer to ubiquinone. "
            "Loss of SDHD: disrupts SDH assembly → succinate accumulation → pseudo-HIF phenotype. "
            "Imprinting: SDHD is within a maternally imprinted region at 11q23.1; the maternal allele "
            "is silenced, so only paternal loss-of-function causes disease. "
            "SDHB IHC lost in SDHD-mutant tumours (indirect effect on SDHB stability)."
        ),
        "n_patients": 40,
        "key_alerts": [
            "SDHD-PATERNAL-IMPRINTING: MATERNAL INHERITANCE DOES NOT CAUSE DISEASE — only paternal SDHD mutations manifest; maternal carriers are silent; pedigree analysis critical to avoid false reassurance",
            "SDHD-HEAD-NECK-PREDOMINANT: Carotid body tumour (CBT) + jugulotympanic + vagal PGL — pulsatile tinnitus, conductive hearing loss, cranial nerve palsies (IX-XII), neck mass",
            "SDHD-NON-SECRETING-70pct: 70% of SDHD-HN-PGL are biochemically silent — plasma metanephrines may be NORMAL; chromogranin A + MRI/DOTATATE PET for surveillance; do NOT rely on biochemistry alone",
            "SDHD-MULTIFOCAL-40pct: 40% of SDHD patients have multifocal tumours — multiple simultaneous HN-PGL ± abdominal; whole-body staging mandatory at diagnosis and follow-up",
            "SDHD-CBT-EMBOLISATION: Pre-operative embolisation of carotid body tumours reduces intraoperative bleeding; carotid body tumour surgery carries risk of carotid injury and Horner's syndrome",
            "SDHD-STEREOTACTIC-RADIOSURGERY: Skull base jugulotympanic PGL — stereotactic radiosurgery (SRS/SRT) for disease control when surgery risks CN palsy; equivalent local control to surgery",
            "SDHD-MALIGNANCY-LOWER: Malignancy rate ~10-15% (lower than SDHB's 30-50%) but NOT zero — annual surveillance required; adrenal pheo in 20% of SDHD",
            "SDHD-CASCADE-TESTING-PATERNAL-FAMILY: Test children of AFFECTED FATHERS and CARRIER MOTHERS; children of carrier mothers need NOT be tested (imprinting silent) — genetic counselling nuance critical",
        ],
        "etiologies": {
            "Missense (transmembrane domain)": 18,
            "Truncating/frameshift": 10,
            "Splice site": 6,
            "Large deletion (Netherlandic founder)": 6,
        },
        "stats": {
            "head_neck_pgl_pct": 80,
            "multifocal_pct": 40,
            "non_secreting_pct": 70,
            "adrenal_pheo_pct": 20,
            "malignant_pct": 12,
            "mean_dx_age": 38,
            "mean_dx_delay_months": 48,
        },
        "dx_delay_distribution": {"<12m": 6, "12-36m": 12, "36-60m": 14, ">60m": 8},
    },

    # ── CDKN1B — MEN4 ────────────────────────────────────────────────────
    {
        "gene": "CDKN1B",
        "protein": "p27/Kip1 — MEN4, MEN1-Like Without MEN1 Mutation, Sequence MEN1 First",
        "alias": (
            "CDKN1B; OMIM gene 600778; MEN4 OMIM 610755; 12p13.1; 198 aa; ~27 kDa; "
            "AD; rare — estimated 1-3% of MEN1-phenotype patients with negative MEN1 sequencing. "
            "CDKN1B encodes p27/Kip1, a cyclin-dependent kinase inhibitor (CDKI). "
            "p27 negatively regulates cell cycle progression at G1/S (inhibits cyclin E-CDK2 "
            "and cyclin D-CDK4 complexes) → promotes cell cycle arrest and differentiation. "
            "Loss of p27 → unrestrained CDK2/4 activity → G1/S override → endocrine cell proliferation. "
            "Phenotype: MEN1-LIKE — primary hyperparathyroidism (most common) + pituitary adenoma; "
            "pancreatic NETs (less common than MEN1); diverse other NETs (thyroid, adrenal). "
            "Important: MEN1 gene MUST be sequenced first — MEN4 is only diagnosed in MEN1 "
            "phenotype with NEGATIVE MEN1 sequencing. "
            "No clear genotype-phenotype correlation. "
            "Treatment follows MEN1 protocols: parathyroidectomy for PHPT; "
            "pituitary-directed therapy; NET surveillance. "
            "Penetrance may be lower than MEN1. Fewer families described overall."
        ),
        "aa": "198 aa",
        "kDa": "~27 kDa",
        "locus": "12p13.1",
        "omim_gene": 600778,
        "omim_disease": 610755,
        "inheritance": "AD; penetrance estimated lower than MEN1; relatively few families described",
        "gene_class": (
            "CDKN1B encodes p27/Kip1 (cyclin-dependent kinase inhibitor 1B), a 198-aa nuclear protein. "
            "p27 contains a CDK-binding/inhibitory domain (N-terminal) and a nuclear localisation signal. "
            "p27 acts as a tumour suppressor by inhibiting cyclin E/A-CDK2 and cyclin D-CDK4/6 "
            "complexes. p27 is regulated post-translationally: phosphorylation at T157 (AKT) → "
            "cytoplasmic retention; T187 (CDK2) → SCF-Skp2-mediated ubiquitination → proteasomal "
            "degradation. Loss of p27 → G1→S progression unchecked. "
            "Unlike RB1 (classic cell cycle tumour suppressor), p27 is not a classic two-hit gene — "
            "haploinsufficiency may be the primary mechanism in many cases."
        ),
        "n_patients": 40,
        "key_alerts": [
            "CDKN1B-MEN1-FIRST: Sequence MEN1 BEFORE CDKN1B — MEN4 is only diagnosed in MEN1-phenotype patients with NEGATIVE MEN1 sequencing; MEN1 accounts for 90% of MEN1-like syndromes",
            "CDKN1B-MEN1-LIKE-PHENOTYPE: PHPT (most common) + pituitary adenoma (prolactinoma/acromegaly) + pancreatic NET (less common than MEN1) — same organ surveillance as MEN1 applies",
            "CDKN1B-ANNUAL-SURVEILLANCE: Annual Ca/PTH + prolactin/IGF-1/ACTH + fasting gastrin/CgA; pituitary MRI every 3 years; abdominal MRI for pancreatic NETs",
            "CDKN1B-TREAT-LIKE-MEN1: Management follows MEN1 protocols: parathyroidectomy (3.5-gland), PPI for ZES, diazoxide for insulinoma, sunitinib/everolimus for progressive pNETs",
            "CDKN1B-RARE-CONSIDER-LATER: CDKN1B testing in MEN1-phenotype: after MEN1 and before CDKN1B consider AIP, CDC73, PRKAR1A depending on phenotype; genetic panel testing efficient",
            "CDKN1B-NO-PROPHYLACTIC-INTERVENTION: No proven prophylactic intervention (unlike RET thyroidectomy); surveillance-guided treatment only",
            "CDKN1B-DIVERSE-NET: Other NETs (thyroid, adrenal) reported in CDKN1B — broader NET surveillance in index cases with multiple tumours",
            "CDKN1B-CASCADE-RELATIVES: All first-degree relatives require germline testing; start annual biochemical surveillance as per MEN1 protocol (Ca/PTH/prolactin/IGF-1 from age 15)",
        ],
        "etiologies": {
            "Missense": 16,
            "Truncating (frameshift/nonsense)": 12,
            "Splice site": 6,
            "Promoter/5'UTR variant": 6,
        },
        "stats": {
            "phpt_pct": 85,
            "pituitary_adenoma_pct": 35,
            "pancreatic_net_pct": 35,
            "other_net_pct": 20,
            "mean_dx_age": 40,
            "mean_dx_delay_months": 42,
        },
        "dx_delay_distribution": {"<12m": 6, "12-36m": 14, "36-60m": 12, ">60m": 8},
    },

    # ── CDC73 — HPT-JT syndrome / Parathyroid carcinoma ──────────────────
    {
        "gene": "CDC73",
        "protein": "Parafibromin — HPT-JT Syndrome, Parathyroid Carcinoma 15%, Jaw Ossifying Fibroma PATHOGNOMONIC",
        "alias": (
            "CDC73 (HRPT2); OMIM gene 607393; HPT-JT OMIM 145001; 1q31.2; 531 aa; ~62 kDa; "
            "AD; prevalence unknown but rare; highest parathyroid carcinoma risk of all HPT syndromes. "
            "CDC73 (also HRPT2) encodes parafibromin, a subunit of the yeast Paf1/RNA Pol II "
            "transcription elongation complex (PAF1C). Parafibromin participates in histone "
            "H3K4 methylation and H2B ubiquitination. Loss of parafibromin → dysregulated "
            "cyclin D1 overexpression → parathyroid tumorigenesis. "
            "Hyperparathyroidism-jaw tumour (HPT-JT) syndrome: "
            "(1) Primary hyperparathyroidism — severe (serum Ca often ≥ 3.0 mmol/L); "
            "single adenoma (unlike MEN1 multiglandular); 15% parathyroid CARCINOMA risk; "
            "(2) Jaw ossifying fibroma — PATHOGNOMONIC for HPT-JT; fibrous dysplasia-like "
            "mandibular/maxillary lesion; benign but locally destructive; "
            "(3) Renal tumours — Wilms tumour, mixed epithelial/stromal (MEST), polycystic; "
            "(4) Uterine tumours — adenomyosis, fibroids (60% females). "
            "Parathyroid carcinoma: indolent but locally invasive and recurrent; "
            "en bloc resection; denosumab for hypercalcaemia; cinacalcet. "
            "Parafibromin IHC: ABSENT in parathyroid carcinoma and many HPT-JT adenomas — "
            "diagnostic adjunct when CDC73 mutation suspected. "
            "Annual: Ca/PTH; jaw OPG; abdominal imaging (renal tumours)."
        ),
        "aa": "531 aa",
        "kDa": "~62 kDa",
        "locus": "1q31.2",
        "omim_gene": 607393,
        "omim_disease": 145001,
        "inheritance": "AD (two-hit tumour suppressor); somatic CDC73 mutations in sporadic parathyroid carcinoma",
        "gene_class": (
            "CDC73 encodes parafibromin (531 aa), a nuclear protein and obligate component of the "
            "human PAF1 complex (PAF1C: PAF1, LEO1, CTR9, WDR61, CDC73/parafibromin). "
            "PAF1C associates with elongating RNA Pol II and coordinates: "
            "H2B ubiquitination (Ub-H2B → H3K4 methylation by MLL complexes), "
            "H3K36 methylation, 3'-end RNA processing. "
            "Parafibromin N-terminal domain binds Leo1; C-terminal domain binds PAF1/WDR61. "
            "Tumour suppression: parafibromin suppresses MYC transcription and cyclin D1 via "
            "β-catenin repression; loss → cyclin D1/MYC overexpression → G1/S progression. "
            "IHC: most CDC73-mutant parathyroid tumours show lost nuclear parafibromin staining."
        ),
        "n_patients": 40,
        "key_alerts": [
            "CDC73-PARATHYROID-CARCINOMA-15pct: 15% of HPT-JT patients develop parathyroid carcinoma — HIGHEST risk of any hereditary HPT syndrome; serum Ca ≥3.0 mmol/L + palpable neck mass = carcinoma until proven otherwise",
            "CDC73-JAW-OSSIFYING-FIBROMA-PATHOGNOMONIC: Mandibular/maxillary ossifying fibroma PATHOGNOMONIC for HPT-JT; OPG panoramic X-ray annually; locally aggressive — surgical excision",
            "CDC73-PARAFIBROMIN-IHC: Parafibromin IHC ABSENT in parathyroid carcinoma and many HPT-JT adenomas; positive (nuclear) staining makes carcinoma unlikely; first-line diagnostic adjunct",
            "CDC73-EN-BLOC-PARATHYROID-CARCINOMA: Parathyroid carcinoma requires en bloc resection (with ipsilateral thyroid lobe + soft tissue) — NO shelling-out; margin-negative surgery critical",
            "CDC73-SEVERE-HYPERCALCAEMIA: Serum Ca ≥3.0 mmol/L in CDC73-HPT — markedly higher than sporadic HPT or MEN1; IV bisphosphonates + cinacalcet acutely; denosumab for refractory/malignant",
            "CDC73-RENAL-TUMOURS: Wilms tumour + MEST (mixed epithelial/stromal) + cystic renal tumours — annual abdominal MRI for renal surveillance; paediatric onset renal tumours prompt CDC73 testing",
            "CDC73-UTERINE-TUMOURS: Adenomyosis + fibroids in >60% of affected females — may be index feature; uterine pathology + HPT = consider CDC73 testing",
            "CDC73-CASCADE-TESTING: All first-degree relatives require germline CDC73 testing + annual Ca/PTH; jaw OPG every 2 years; renal imaging from childhood (Wilms risk)",
        ],
        "etiologies": {
            "Truncating/frameshift": 20,
            "Missense": 10,
            "Splice site": 6,
            "Large deletion": 4,
        },
        "stats": {
            "phpt_pct": 95,
            "parathyroid_carcinoma_pct": 15,
            "jaw_ossifying_fibroma_pct": 30,
            "renal_tumour_pct": 15,
            "uterine_tumour_pct": 60,
            "mean_ca_mmol_l": 3.1,
            "mean_dx_age": 32,
            "mean_dx_delay_months": 24,
        },
        "dx_delay_distribution": {"<12m": 10, "12-36m": 14, "36-60m": 10, ">60m": 6},
    },

    # ── PRKAR1A — Carney complex ──────────────────────────────────────────
    {
        "gene": "PRKAR1A",
        "protein": "PKA-R1α — Carney Complex, Cardiac Myxoma EMBOLIC STROKE, Annual Echocardiography MANDATORY",
        "alias": (
            "PRKAR1A; OMIM gene 188830; CNC1 OMIM 160980; 17q24.2; 381 aa; ~43 kDa; "
            "AD; prevalence ~750 families reported worldwide. "
            "PRKAR1A encodes the regulatory subunit 1-alpha of cAMP-dependent protein kinase A (PKA). "
            "Loss of PRKAR1A → unrestrained PKA catalytic subunit activity → constitutive cAMP "
            "signalling → activation of CREB + other transcription factors → proliferation in "
            "specific cell types. "
            "Carney complex (CNC) — four cardinal features: "
            "(1) Spotty pigmentation — lentigines + blue naevi on lips, genitalia, conjunctiva; "
            "(2) Cardiac myxoma — 30-40%, recurrent, any chamber (not just left atrium unlike sporadic); "
            "(3) Endocrine tumours — PPNAD (primary pigmented nodular adrenocortical disease): "
            "ACTH-independent Cushing's syndrome (paradoxical cortisol RISE with dexamethasone); "
            "GH-secreting pituitary adenoma (acromegaly 10%); thyroid nodules/follicular adenoma; "
            "(4) Non-cardiac myxomas — breast, testicular (large cell calcifying Sertoli cell tumour "
            "LCCSCT — pathognomonic in males), oral mucosa, skin. "
            "Cardiac myxoma: EMBOLIC STROKE RISK — surgical excision URGENT (recurs after excision); "
            "annual echocardiography MANDATORY (TTE + TOE). "
            "PPNAD: corticotropin-releasing hormone (CRH) + dexamethasone test → paradoxical cortisol "
            "rise PATHOGNOMONIC (urinary free cortisol rises ≥50% after high-dose dexamethasone). "
            "Bilateral adrenalectomy for PPNAD-Cushing; metyrapone/ketoconazole bridge."
        ),
        "aa": "381 aa",
        "kDa": "~43 kDa",
        "locus": "17q24.2",
        "omim_gene": 188830,
        "omim_disease": 160980,
        "inheritance": "AD (two-hit); ~30% de novo; haploinsufficiency confirmed in most",
        "gene_class": (
            "PRKAR1A encodes PKA regulatory subunit 1-alpha (RIα, 381 aa). "
            "PKA holoenzyme: 2× regulatory (R) + 2× catalytic (C) subunits. "
            "Without cAMP: R subunits hold C subunits inactive. "
            "cAMP binding to R subunits (two CNB domains per R subunit) → R-C dissociation → "
            "free C subunits → phosphorylation of CREB, cytoskeletal, metabolic targets. "
            "PRKAR1A loss of function → constitutively free C subunits → constitutive PKA → "
            "proliferation in adrenocortical cells (PPNAD), myocardium/fibroblasts (myxoma), "
            "melanocytes (lentigines), Sertoli cells (LCCSCT). "
            "Genotype-phenotype: truncating mutations (NMD) → more severe phenotype; "
            "non-NMD missense → some residual R1α → milder. Most mutations: exons 1-11."
        ),
        "n_patients": 40,
        "key_alerts": [
            "PRKAR1A-CARDIAC-MYXOMA-URGENT: Cardiac myxoma — EMBOLIC STROKE risk; surgical excision URGENTLY REQUIRED; can affect ANY cardiac chamber (unlike sporadic left atrium predominance); RECURRENT after excision",
            "PRKAR1A-ANNUAL-ECHOCARDIOGRAPHY: Annual transthoracic ± transoesophageal echocardiography MANDATORY — cardiac myxoma can grow rapidly; first echo at diagnosis, then annually for life",
            "PRKAR1A-PARADOXICAL-DEXAMETHASONE: PPNAD-Cushing: paradoxical RISE in urinary free cortisol after high-dose dexamethasone (≥50%) PATHOGNOMONIC — standard dexamethasone suppression test FALSELY NORMAL in PPNAD",
            "PRKAR1A-PPNAD-BILATERAL-ADRENALECTOMY: PPNAD Cushing requires bilateral adrenalectomy — cortisol-secreting micronodules throughout both adrenals; no dominant adenoma; unilateral surgery fails",
            "PRKAR1A-LCCSCT-PATHOGNOMONIC: Large cell calcifying Sertoli cell tumour (LCCSCT) — bilateral testicular lesions in males; testicular US annually from childhood; PATHOGNOMONIC for CNC (virtually diagnostic)",
            "PRKAR1A-SPOTTY-PIGMENTATION: Lentigines + blue naevi on lips, genitalia, conjunctiva, buccal mucosa — appear in childhood; fade with age; skin biopsy confirms if uncertain; index feature for CNC workup",
            "PRKAR1A-GH-EXCESS-ACROMEGALY: 10% develop GH-excess/acromegaly (pituitary adenoma); IGF-1 annually; pituitary MRI every 3 years; lanreotide/octreotide + pasireotide or pegvisomant",
            "PRKAR1A-CASCADE-TESTING: All first-degree relatives require PRKAR1A germline testing; annual echo + UFC + spotty pigmentation exam + testicular US (males) from childhood",
        ],
        "etiologies": {
            "Truncating (NMD — most common)": 22,
            "Missense (non-NMD)": 8,
            "Splice site": 6,
            "Large deletion": 4,
        },
        "stats": {
            "cardiac_myxoma_pct": 35,
            "spotty_pigmentation_pct": 85,
            "ppnad_pct": 45,
            "acromegaly_pct": 10,
            "lccsct_male_pct": 65,
            "non_cardiac_myxoma_pct": 30,
            "mean_dx_age": 20,
            "mean_dx_delay_months": 36,
        },
        "dx_delay_distribution": {"<12m": 10, "12-36m": 14, "36-60m": 10, ">60m": 6},
    },
]


def _make_patients(gene_entry, rng):
    """Generate synthetic patient records for one gene."""
    gene = gene_entry["gene"]
    n = gene_entry["n_patients"]
    ages = [rng.randint(10, 70) for _ in range(n)]
    delays = [rng.choice([1, 6, 12, 24, 36, 48, 60, 72]) for _ in range(n)]
    etiol_keys = list(gene_entry["etiologies"].keys())
    etiol_weights = list(gene_entry["etiologies"].values())
    etiols = rng.choices(etiol_keys, weights=etiol_weights, k=n)
    patients = []
    for i in range(n):
        patients.append({
            "id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "dx_age": ages[i],
            "dx_delay_months": delays[i],
            "variant_class": etiols[i],
        })
    return patients


def _build_cohort():
    all_data = {}
    for idx, ge in enumerate(ENDOCRINE_TUMOUR_GENES):
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        ge_copy = dict(ge)
        ge_copy["seed"] = seed
        ge_copy["patients"] = _make_patients(ge, rng)
        all_data[ge["gene"]] = ge_copy
    return all_data


_COHORT = _build_cohort()


def get_overview():
    genes_summary = []
    total = 0
    all_dx_ages = []
    all_delays = []
    top_alerts = []
    for gene, info in _COHORT.items():
        n = info["n_patients"]
        total += n
        pts = info["patients"]
        ages = [p["dx_age"] for p in pts]
        delays = [p["dx_delay_months"] for p in pts]
        all_dx_ages.extend(ages)
        all_delays.extend(delays)
        genes_summary.append({
            "gene": gene,
            "protein_short": info["protein"].split(" — ")[0],
            "n_patients": n,
            "locus": info["locus"],
            "inheritance": info["inheritance"].split(";")[0],
            "omim_disease": info["omim_disease"],
            "mean_dx_age": round(sum(ages) / len(ages), 1),
        })
        top_alerts.extend(info["key_alerts"][:2])
    aggregate_stats = {
        "total_patients": total,
        "mean_dx_age_years": round(sum(all_dx_ages) / len(all_dx_ages), 1),
        "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
        "cascade_tested_pct": 74,
        "men1_phpt_pct": 95,
        "ret_prophylactic_thyroidectomy_pct": 62,
        "vhl_retinal_angioma_pct": 70,
        "sdhb_malignant_pct": 40,
        "sdhd_multifocal_pct": 40,
        "cdc73_parathyroid_carcinoma_pct": 15,
        "prkar1a_cardiac_myxoma_pct": 35,
    }
    return {
        "atlas": "Hereditary-Endocrine-Tumour-Atlas",
        "genes": genes_summary,
        "aggregate_stats": aggregate_stats,
        "top_alerts": top_alerts,
        "seeds": f"{SEED_BASE}–{SEED_BASE + 7}",
    }


def get_breakdown():
    result = {}
    for gene, info in _COHORT.items():
        pts = info["patients"]
        result[gene] = {
            "gene": gene,
            "n_patients": info["n_patients"],
            "alias": info["alias"],
            "gene_class": info["gene_class"],
            "locus": info["locus"],
            "aa": info["aa"],
            "kDa": info["kDa"],
            "omim_gene": info["omim_gene"],
            "omim_disease": info["omim_disease"],
            "inheritance": info["inheritance"],
            "key_alerts": info["key_alerts"],
            "etiologies": info["etiologies"],
            "stats": info["stats"],
            "dx_delay_distribution": info["dx_delay_distribution"],
            "patients": pts[:10],
        }
    return result


def get_definitions():
    return {
        "atlas": "Hereditary-Endocrine-Tumour-Atlas",
        "concepts": {
            "Multiple Endocrine Neoplasia (MEN)": (
                "Hereditary syndromes causing tumours in ≥2 endocrine glands. MEN1 (menin, 11q13): "
                "parathyroid + pituitary + pancreatic NETs. MEN2A/B (RET, 10q11): MTC + pheo "
                "(± HPT in MEN2A; ± marfanoid + mucosal neuromas in MEN2B). MEN4 (CDKN1B): MEN1-like."
            ),
            "Von Hippel-Lindau (VHL) Syndrome": (
                "AD tumour suppressor syndrome (VHL, 3p25). pVHL ubiquitinates HIF-α in normoxia. "
                "VHL loss → HIF-2α accumulation → VEGF/EPO/PDGF → angiogenesis → "
                "ccRCC + CNS hemangioblastoma + retinal angioma + pheo + PNET + ELST. "
                "Belzutifan (HIF-2α inhibitor) FDA 2021 for non-surgical VHL-associated tumours."
            ),
            "SDH-Related PGL/PCC Syndromes": (
                "Hereditary paraganglioma-phaeochromocytoma syndromes caused by SDH subunit mutations. "
                "SDH complex II converts succinate → fumarate + electrons to ubiquinone. "
                "SDH loss → succinate accumulation → PHD inhibition → HIF activation + epigenetic "
                "hypermethylation. SDHB (PGL4): highest malignant risk (30-50%). "
                "SDHD (PGL1): head/neck PGL predominant; PATERNAL IMPRINTING — maternal carriers silent. "
                "SDHB IHC: absent in all SDH-deficient tumours."
            ),
            "Hyperparathyroidism-Jaw Tumour (HPT-JT) Syndrome": (
                "AD syndrome (CDC73/HRPT2, 1q31). Parafibromin = PAF1 complex subunit, tumour suppressor. "
                "HPT-JT: severe PHPT (Ca ≥3.0 mmol/L) + 15% parathyroid carcinoma risk + "
                "jaw ossifying fibroma (PATHOGNOMONIC) + renal tumours + uterine tumours. "
                "Parafibromin IHC absent in carcinoma. En bloc resection mandatory for carcinoma."
            ),
            "Carney Complex (CNC)": (
                "AD syndrome (PRKAR1A, 17q24). Unrestrained PKA C-subunit → proliferation in "
                "myocardium (myxoma), adrenal (PPNAD), melanocytes (lentigines), Sertoli cells (LCCSCT). "
                "Cardinal: spotty pigmentation + cardiac myxoma + PPNAD-Cushing + non-cardiac myxoma. "
                "PPNAD: paradoxical cortisol RISE with dexamethasone PATHOGNOMONIC. "
                "Cardiac myxoma → embolic stroke; annual echocardiography MANDATORY."
            ),
            "Prophylactic Thyroidectomy in MEN2": (
                "RET mutation = near 100% lifetime MTC risk. ATA risk stratification guides timing: "
                "ATA-D (M918T, MEN2B) → within 6 months of life (highest urgency); "
                "ATA-C (C634 mutations) → by age 5; ATA-B → by age 5; ATA-A (FMTC-only) → "
                "calcitonin-guided. Screen phaeochromocytoma BEFORE thyroid surgery."
            ),
            "Succinate Accumulation (Oncometabolism)": (
                "SDH loss → succinate → inhibits α-KG-dependent dioxygenases: "
                "(1) PHDs → HIF stabilisation (pseudohypoxia); "
                "(2) TET methylcytosine dioxygenases → DNA hypermethylation; "
                "(3) Jumonji KDMs → histone hypermethylation. "
                "This SDHx hypermethylator phenotype is diagnostically distinct and therapeutically relevant."
            ),
        },
        "pharmacological_distinctions": [
            "Selpercatinib (RET kinase inhibitor, FDA 2020) vs. cabozantinib (multikinase, FDA 2012) vs. vandetanib (RET/VEGFR2/EGFR, FDA 2011) — all for advanced RET-mutant MTC; selpercatinib most selective with CNS penetration",
            "Belzutifan (HIF-2α inhibitor, FDA 2021) — VHL-associated ccRCC/PNET/hemangioblastoma not requiring immediate surgery; inhibits HIF-2α-ARNT dimer; PD 24-48% partial response",
            "177Lu-DOTATATE (Lutathera, PRRT, FDA 2018) — progressive SSTR+ NETs (including SDH-PGL); SSR-positive on Ga-DOTATATE PET required; mean OS benefit 11.7 months (NETTER-1)",
            "Cinacalcet (calcimimetic, allosteric CaSR activator) — hypercalcaemia in MEN1-PHPT when surgery contraindicated; denosumab for malignant hypercalcaemia (CDC73 parathyroid carcinoma)",
            "Diazoxide (KATP channel opener) vs. octreotide/lanreotide (somatostatin analogue) — diazoxide for MEN1 insulinoma as bridge; SSA for gastrinoma/NET biochemical control",
            "Pasireotide (pan-SSR agonist, binds SSR1/2/3/5) vs. octreotide (SSR2/5) — pasireotide preferred for ACTH-secreting pituitary adenoma (MEN1-associated) due to SSR5 binding; risk of hyperglycaemia",
            "Metyrapone + ketoconazole (steroidogenesis inhibitors) as bridge to bilateral adrenalectomy in PPNAD-Cushing (PRKAR1A); bilateral adrenalectomy is definitive — no medical cure for PPNAD",
            "Everolimus (mTOR inhibitor) + sunitinib (VEGFR2/PDGFR/KIT) — both FDA-approved for progressive pancreatic NETs; mTOR pathway activated in MEN1-NETs via AKT; use in sequence for PD",
        ],
        "key_standards": [
            "American Thyroid Association (ATA) MEN2 Guidelines 2015 — RET codon risk stratification + prophylactic thyroidectomy timing (Wells SA Jr, Thyroid 2015)",
            "International MEN1 Guidelines — Thakker RV, JCEM 2012; de Laat JM, Endocr Rev 2016 — surveillance protocol including annual Ca/PTH, fasting gastrin, pituitary MRI every 3 years",
            "VHL Alliance Clinical Care Standards 2018 — annual multi-organ surveillance; belzutifan integration post-FDA 2021 approval (Jonasch E, NEJM 2021)",
            "Endocrine Society SDH PGL/PCC Guidelines — Lenders JW, JCEM 2014; update Tufton N, Lancet Diabetes Endocrinol 2022; SDHB germline testing in ALL PGL/PCC patients",
            "SDH Imprinting and SDHD-PGL1 — Baysal BE, Science 2000 (PGL1 discovery); Hao HX, Science 2009 (SDH-Krebs-oncometabolism mechanism); paternal imprinting confirmed Baysal 2004",
            "HPT-JT and CDC73 — Carpten JD, Nat Genet 2002 (HRPT2 discovery); Gill AJ, Am J Surg Pathol 2010 (parafibromin IHC); Schulte KM, J Endocrinol Invest 2011",
            "Carney Complex — Stratakis CA, J Clin Endocrinol Metab 2001 (PRKAR1A discovery); CNC consensus — Bertherat J, Eur J Endocrinol 2009; PPNAD paradoxical dexamethasone test — Sarlis NJ 1997",
            "Ga-68-DOTATATE PET for PGL/PCC — Janssen I, JCEM 2016; 177Lu-DOTATATE (NETTER-1 trial) — Strosberg J, NEJM 2017; preferred for SDHB metastatic PGL",
        ],
    }
