#!/usr/bin/env python3
"""Hereditary-Multiple-Endocrine-Neoplasia-Atlas — Complete 8-Gene MEN Atlas
(MEN1 · RET · CDKN1B · AIP · PRKAR1A · VHL · SDHB · MAX).

MEN1     (Menin; 610 aa; 67 kDa; 11q13.1; AD LOF;
          MEN1 syndrome — 3P triad: Parathyroid adenoma (>90%) +
          Pituitary adenoma (40%) + Pancreatic/duodenal NET (40-70%);
          MULTIGLANDULAR HPT PATHOGNOMONIC; gastrinoma ZES most common PNET;
          subtotal parathyroidectomy + thymectomy MANDATORY same operation;
          seed SEED_BASE+0).
RET      (Rearranged during Transfection; 1114 aa; 124 kDa; 10q11.21; AD GOF;
          MEN2A: MTC + Pheo + HPT; MEN2B: MTC + Pheo + mucosal neuromas + Marfanoid;
          ATA risk D (M918T) thyroidectomy within 6 months of life;
          CHECK PHEO BEFORE THYROID SURGERY — hypertensive crisis;
          seed SEED_BASE+1).
CDKN1B   (p27/KIP1; 198 aa; 22 kDa; 12p13.1; AD LOF;
          MEN4 syndrome — MEN1-like: HPT + pituitary adenoma + rare PNET;
          MEN1-phenotype with NEGATIVE MEN1 sequencing;
          seed SEED_BASE+2).
AIP      (Aryl hydrocarbon receptor Interacting Protein; 330 aa; 37 kDa; 11q13.2; AD LOF;
          FIPA — GH-secreting somatotrophinoma; gigantism/acromegaly; young onset (<30y);
          SSA-resistant — pegvisomant preferred;
          seed SEED_BASE+3).
PRKAR1A  (cAMP-dependent protein kinase type I regulatory subunit alpha; 381 aa; 43 kDa; 17q24.2; AD LOF;
          Carney complex — cardiac myxoma + spotty pigmentation + PPNAD Cushing +
          PARADOXICAL CORTISOL RISE ON DEXAMETHASONE (Liddle test) PATHOGNOMONIC;
          seed SEED_BASE+4).
VHL      (pVHL; 213 aa; 24 kDa; 3p25.3; AD LOF;
          VHL syndrome — Hemangioblastoma + Clear cell RCC + Pheo + PNET + ELST;
          belzutifan (HIF-2α inhibitor) FDA approved 2021;
          seed SEED_BASE+5).
SDHB     (Succinate dehydrogenase iron-sulfur subunit; 280 aa; 32 kDa; 1p36.13; AD LOF;
          PGL4 — Extra-adrenal paraganglioma predominant; HIGHEST MALIGNANCY 30-50%;
          Carney-Stratakis dyad (GIST + PGL); DOTATATE PET staging;
          seed SEED_BASE+6).
MAX      (MYC-associated factor X; 160 aa; 18 kDa; 14q23.3; AD LOF, paternal imprinting;
          PGL5 — Bilateral adrenal pheo; PATERNAL IMPRINTING — maternal carriers NOT at risk;
          adrenaline-secreting; cortical-sparing adrenalectomy preferred;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2518-2525).
"""

import random

SEED_BASE = 2518

MEN_GENES = [
    # -- MEN1 -- MEN1 syndrome 3P triad -------------------------------------------------------
    {
        "gene": "MEN1",
        "alt_name": (
            "MEN1 (MEN1-610aa-11q13.1 / AD-LOF -- "
            "MEN1-SYNDROME-3P-TRIAD-PARATHYROID+PITUITARY+PANCREATIC-NET -- "
            "MULTIGLANDULAR-PRIMARY-HPT-ALL-4-GLANDS-PATHOGNOMONIC -- "
            "GASTRINOMA-ZES-SECRETIN-STIMULATION->120pgmL-PATHOGNOMONIC -- "
            "SUBTOTAL-PARATHYROIDECTOMY-3.5GLANDS+THYMECTOMY-SAME-OPERATION-MANDATORY)"
        ),
        "protein": (
            "MEN1 -- 11q13.1 AD LOF -- MEN1-610aa -- "
            "Menin-67kDa-Nuclear-Scaffold-Protein-Tumour-Suppressor -- "
            "Histone-H3K4-Methyltransferase-Complex-Component-MLL1-MLL2-Trithorax -- "
            "Interacts-JunD-SMAD3-NF-kappaB-Pem-Transcription-Factors -- "
            "Growth-Suppressor-In-Endocrine-Pancreas-Parathyroid-Pituitary -- "
            "Two-Hit-Knudson-Model-Germline-LOF-Plus-Somatic-Second-Hit-in-Tumour -- "
            "Founder-Variants-W341X-R460X-frameshift-Most-Common-European -- "
            "OMIM-Gene-613733-Disease-MEN1-131100"
        ),
        "locus": "11q13.1",
        "protein_size": "610 aa / 67 kDa",
        "inheritance": (
            "AD (autosomal dominant, LOF, two-hit tumour suppressor); "
            "penetrance >95% by age 50; "
            "MULTIGLANDULAR parathyroid involvement distinguishes MEN1 from sporadic single-gland HPT; "
            "gastrinoma most common cause of death; "
            "thymectomy mandatory at time of parathyroidectomy (thymic NET — rare but lethal in MEN1); "
            "cascade genetic testing recommended from age 5 in gene-positive families; "
            "CDKN1B (MEN4) phenocopies MEN1 when MEN1 sequencing negative"
        ),
        "disease_category": (
            "MEN1 syndrome — 3P triad: Parathyroid adenoma (>90%, multiglandular) + "
            "Pituitary adenoma (40%, prolactinoma most common) + "
            "Pancreatic/duodenal NET (40-70%, gastrinoma/Zollinger-Ellison most common)"
        ),
        "disease_pathway": (
            "Menin is a nuclear scaffold protein functioning as a tumour suppressor in endocrine tissues. "
            "It is a core component of the MLL1/MLL2 histone methyltransferase complex, regulating H3K4me3 marks "
            "on promoters of CDK inhibitor genes (p27, p18, p21). Loss of menin → reduced CDK inhibitor expression "
            "→ cell cycle deregulation → endocrine cell hyperplasia and tumour formation. "
            "MEN1 syndrome follows Knudson two-hit model: germline MEN1 LOF (first hit) + somatic second hit "
            "(deletion of 11q13 region or point mutation) in endocrine progenitor cells → tumour initiation. "
            "PARATHYROID: multiglandular hyperplasia/adenomas (all 4 glands) → PTH excess → hypercalcaemia. "
            "PANCREATIC NET: gastrinoma (G cells, duodenal wall/pancreas) → uncontrolled gastrin secretion → "
            "Zollinger-Ellison syndrome (ZES) — peptic ulceration, diarrhoea, oesophageal reflux; "
            "insulinoma second most common; glucagonoma, VIPoma, non-functional PNET also occur. "
            "PITUITARY: prolactinoma most common; also GH-secreting (acromegaly), ACTH-secreting (Cushing), "
            "non-functional; usually benign but may cause local mass effects. "
            "THYMIC NET: uncommon but carries highest mortality in MEN1 — prophylactic thymectomy at "
            "parathyroidectomy is established standard of care."
        ),
        "pathognomonic": (
            "MULTIGLANDULAR PRIMARY HPT — all 4 parathyroid glands enlarged (unlike sporadic single adenoma) "
            "PATHOGNOMONIC for MEN1; GASTRINOMA: secretin stimulation test gastrin rise >120 pg/mL "
            "PATHOGNOMONIC for ZES; duodenal wall gastrinoma (triangle of Whipple); "
            "subtotal parathyroidectomy (3.5 glands) + thymectomy MANDATORY at same operation "
            "(thymic NET lethal in MEN1)"
        ),
        "hormone_profile": (
            "Elevated PTH + hypercalcaemia (HPT); elevated gastrin (ZES); "
            "elevated prolactin or IGF-1 (pituitary); elevated fasting glucose or insulin (insulinoma)"
        ),
        "severity_sds": "N/A — endocrine tumor syndrome; penetrance >95% by age 50",
        "treatment": (
            "Annual biochemical surveillance from age 5 (calcium/PTH/gastrin/prolactin/IGF-1); "
            "EUS for PNET; subtotal parathyroidectomy + thymectomy for HPT; "
            "PPIs for ZES; somatostatin analogues (octreotide/lanreotide) for functional PNET; "
            "MRI pituitary annually; thymic NET surveillance CT annually"
        ),
        "key_features": [
            "3P triad: Parathyroid + Pituitary + Pancreatic NET",
            "Multiglandular parathyroid disease (all 4 glands) — distinguishes from sporadic",
            "Gastrinoma most common PNET and most common cause of death in MEN1",
            "Secretin stimulation test: gastrin rise >120 pg/mL confirms ZES",
            "Subtotal parathyroidectomy (3.5 glands) + thymectomy in same operation",
            "Annual biochemical screening from age 5 in gene-positive relatives",
            "Thymic NET rare but lethal — prophylactic thymectomy lifesaving",
        ],
        "key_ddx": [
            "Sporadic primary HPT (single gland, MEN1-negative)",
            "Sporadic non-functional PNET (MEN1 negative)",
            "Sporadic prolactinoma",
            "MEN4 (CDKN1B) — MEN1-like but MEN1 negative",
            "MEN2A (RET) — no pituitary/PNET; has MTC",
        ],
        "onset_age": "Variable; HPT usually 2nd-3rd decade; PNET 3rd-4th decade; pituitary variable",
        "gh_deficiency": False,
        "mecasermin_indicated": False,
        "autosomal_recessive_risk": False,
    },

    # -- RET -- MEN2A / MEN2B / FMTC --------------------------------------------------------
    {
        "gene": "RET",
        "alt_name": (
            "RET (RET-1114aa-10q11.21 / AD-GOF -- "
            "MEN2A-Cys634Arg/Tyr-MTC+PHEO+HPT -- "
            "MEN2B-Met918Thr-MTC+PHEO+MUCOSAL-NEUROMAS+MARFANOID -- "
            "ATA-RISK-D-M918T-THYROIDECTOMY-WITHIN-6-MONTHS -- "
            "CHECK-PHEO-BEFORE-THYROID-SURGERY-HYPERTENSIVE-CRISIS)"
        ),
        "protein": (
            "RET -- 10q11.21 AD GOF -- RET-1114aa -- "
            "Rearranged-During-Transfection-RET-RTK-124kDa-Receptor-Tyrosine-Kinase -- "
            "GDNF-Family-Ligand-Receptor-Cadherin-Like-Domain-CRD-TM-Intracellular-TK-Domain -- "
            "Cys634Arg/Tyr-Exon11-Extracellular-CRD-Constitutive-Dimerisation-MEN2A -- "
            "Met918Thr-Exon16-Intracellular-TK-Domain-Constitutive-Activation-MEN2B-Most-Aggressive -- "
            "Codon-Specific-Phenotype-Genotype-Exon10-11-MEN2A-Exon16-MEN2B-Exon13-14-FMTC -- "
            "ATA-Risk-Stratification-A-B-C-D-D=Met918Thr-Highest-Risk -- "
            "OMIM-Gene-164761-Disease-MEN2A-171400-MEN2B-162300"
        ),
        "locus": "10q11.21",
        "protein_size": "1114 aa / 124 kDa",
        "inheritance": (
            "AD (autosomal dominant, GOF); "
            "MEN2A (Cys634Arg/Tyr, Cys611): MTC >90% + pheo 50% + HPT 20-30%; "
            "MEN2B (Met918Thr): MTC most aggressive (infancy onset) + pheo + mucosal neuromas + Marfanoid habitus; no HPT; "
            "FMTC (familial medullary thyroid carcinoma): MTC only, lower penetrance; "
            "ATA risk stratification: D=Met918Thr (thyroidectomy <6 months), C=Cys634 (before age 5), "
            "B=Cys611/618/620 (before age 5-10), A=others; "
            "CHECK PHEO BEFORE ANY SURGERY — alpha-blockade mandatory; "
            "cascade testing all first-degree relatives"
        ),
        "disease_category": (
            "MEN2A (RET Cys634Arg/Arg) — MTC (>90%) + Pheo (50%) + Primary HPT (20-30%); "
            "MEN2B (Met918Thr) — MTC (most aggressive, infancy onset) + Pheo + Mucosal neuromas + Marfanoid; "
            "FMTC (same mutations) — MTC only"
        ),
        "disease_pathway": (
            "RET (Rearranged during Transfection) is a receptor tyrosine kinase critical for neural crest cell migration "
            "and development of the kidney, enteric nervous system, and C cells of the thyroid. "
            "GOF mutations cause constitutive (ligand-independent) RET kinase activity → "
            "downstream RAS-MAPK, PI3K-AKT, and PLC-γ signalling → proliferation and survival of "
            "thyroid C cells, adrenal chromaffin cells, and parathyroid cells. "
            "MEN2A (exon 10/11 cysteine mutations): extracellular cysteine → free cysteine → "
            "intermolecular disulfide bonds → constitutive dimerisation → kinase activation. "
            "MEN2B (Met918Thr, exon 16): intracellular kinase domain → alters substrate specificity + "
            "constitutive activation; most aggressive — MTC occurs in infancy; "
            "neural crest-related features: mucosal neuromas (tongue, lips), intestinal ganglioneuromatosis, Marfanoid. "
            "MTC: calcitonin is the biomarker; doubling time of calcitonin and CEA predicts prognosis. "
            "Pheo: adrenal chromaffin cell GOF → catecholamine excess (predominantly adrenaline in adrenal pheo); "
            "CRITICAL SAFETY: undetected pheo → hypertensive crisis during thyroid surgery → fatal; "
            "always measure plasma metanephrines BEFORE thyroid surgery."
        ),
        "pathognomonic": (
            "MEDULLARY THYROID CARCINOMA + PHEOCHROMOCYTOMA in same patient = MEN2A PATHOGNOMONIC; "
            "MUCOSAL NEUROMAS (tongue/lips) + Marfanoid habitus + MTC = MEN2B PATHOGNOMONIC; "
            "calcitonin elevation (basal >100 pg/mL or pentagastrin-stimulated) confirms MTC; "
            "prophylactic thyroidectomy timing by ATA risk category: D (Met918Thr) within 6 months of life; "
            "C (Cys634) before age 5; check pheo BEFORE thyroid surgery (hypertensive crisis if undetected)"
        ),
        "hormone_profile": (
            "Elevated calcitonin (MTC biomarker); elevated metanephrines/normetanephrines (pheo); "
            "elevated PTH/calcium (HPT in MEN2A only)"
        ),
        "severity_sds": "N/A — hereditary cancer syndrome; MTC penetrance >90% lifetime",
        "treatment": (
            "Prophylactic thyroidectomy by ATA risk category; "
            "pheo: adrenalectomy with alpha-blockade FIRST (phenoxybenzamine) before ANY surgery; "
            "RET inhibitors (vandetanib/cabozantinib) for metastatic MTC; "
            "annual calcitonin + CEA monitoring post-thyroidectomy; "
            "plasma metanephrines annually; genetic cascade testing for at-risk family"
        ),
        "key_features": [
            "MTC is first manifestation in most patients",
            "Pheo must be excluded BEFORE thyroid surgery (hypertensive crisis risk)",
            "ATA risk category D (M918T) = prophylactic thyroidectomy within 6 months",
            "Alpha-blockade (phenoxybenzamine) mandatory before pheo resection",
            "MEN2B: no parathyroid disease; MEN2A: parathyroid 20-30%",
            "CEA + calcitonin doubling time predicts prognosis",
            "RET exon 10/11 codon-specific phenotype-genotype correlation",
        ],
        "key_ddx": [
            "MEN1 (no MTC, parathyroid multiglandular)",
            "VHL (pheo + RCC + hemangioblastoma, no MTC)",
            "Sporadic MTC (bilateral/multifocal → test RET)",
            "SDHx pheo/PGL (extra-adrenal predominant)",
        ],
        "onset_age": "MTC from infancy (MEN2B M918T); MEN2A 2nd-3rd decade; pheo 3rd-4th decade",
        "gh_deficiency": False,
        "mecasermin_indicated": False,
        "autosomal_recessive_risk": False,
    },

    # -- CDKN1B -- MEN4 syndrome -------------------------------------------------------------
    {
        "gene": "CDKN1B",
        "alt_name": (
            "CDKN1B (CDKN1B-198aa-12p13.1 / AD-LOF -- "
            "MEN4-SYNDROME-MEN1-PHENOTYPE-MEN1-NEGATIVE -- "
            "p27-KIP1-CYCLIN-DEPENDENT-KINASE-INHIBITOR -- "
            "HPT+PITUITARY-ADENOMA+RARE-PNET -- "
            "ACTH-ADENOMA-CUSHING-MORE-COMMON-THAN-MEN1)"
        ),
        "protein": (
            "CDKN1B -- 12p13.1 AD LOF -- CDKN1B-198aa -- "
            "p27-KIP1-22kDa-Cyclin-Dependent-Kinase-Inhibitor-1B -- "
            "Binds-Inhibits-CDK2-Cyclin-E-CDK4-Cyclin-D-Complexes -- "
            "G1-S-Phase-Transition-Brake-Tumour-Suppressor-Cell-Cycle -- "
            "Nuclear-Localisation-Ubiquitin-SKP2-Mediated-Degradation -- "
            "MEN4-First-Identified-Rats-Heterozygous-p27-Develop-MEN-Like-Syndrome -- "
            "OMIM-Gene-600778-Disease-MEN4-610755"
        ),
        "locus": "12p13.1",
        "protein_size": "198 aa / 22 kDa",
        "inheritance": (
            "AD (autosomal dominant, LOF); "
            "MEN1-like phenotype with NEGATIVE MEN1 sequencing — test CDKN1B next; "
            "lower penetrance than MEN1; variable expressivity; "
            "pancreatic NET much rarer than MEN1; "
            "ACTH-secreting pituitary adenomas (Cushing disease) more common in MEN4 than MEN1; "
            "same surveillance protocol as MEN1 once CDKN1B positive"
        ),
        "disease_category": (
            "MEN4 syndrome — MEN1-like: Parathyroid adenoma + Pituitary adenoma "
            "(ACTH/GH/prolactin) + rare PNET; MEN1 gene-negative MEN1-phenotype"
        ),
        "disease_pathway": (
            "p27 (KIP1) is a cyclin-dependent kinase inhibitor that acts as a cell cycle brake at G1/S transition. "
            "p27 binds and inhibits CDK2/cyclin E and CDK4/cyclin D complexes, preventing S-phase entry. "
            "LOF of CDKN1B → insufficient CDK inhibition → inappropriate cell cycle progression "
            "in endocrine cells → hyperplasia and adenoma formation. "
            "The phenotypic similarity to MEN1 reflects shared downstream targets: "
            "menin (MEN1) regulates p27 transcription (H3K4 methylation at CDKN1B promoter); "
            "MEN1 LOF → reduced p27 expression → cell cycle dysregulation; "
            "CDKN1B LOF → same downstream endpoint via a different mechanism. "
            "First described in rats: heterozygous p27-null rats spontaneously develop pituitary tumours "
            "and parathyroid hyperplasia — mirroring the human MEN4 phenotype. "
            "Human MEN4: smaller cohort than MEN1; reported pituitary adenomas include ACTH (Cushing disease), "
            "GH, prolactin, and non-functional; parathyroid HPT clinically similar to MEN1 but single or multiglandular. "
            "Pancreatic NETs are rare in MEN4 (unlike MEN1 where they dominate morbidity)."
        ),
        "pathognomonic": (
            "MEN1-LIKE PHENOTYPE WITH NEGATIVE MEN1 SEQUENCING = MEN4 (CDKN1B) UNTIL PROVEN OTHERWISE; "
            "HPT often first presentation; pituitary adenoma especially ACTH-secreting (Cushing disease); "
            "pancreatic NET rare compared to MEN1; lower penetrance than MEN1"
        ),
        "hormone_profile": (
            "Elevated PTH + hypercalcaemia; elevated ACTH or cortisol (Cushing); "
            "pituitary hormone excess by adenoma type"
        ),
        "severity_sds": "N/A — lower penetrance than MEN1; variable expressivity",
        "treatment": (
            "Same surveillance protocol as MEN1 (annual biochemical) once CDKN1B positive; "
            "parathyroidectomy for HPT; pituitary-directed therapy; genetic counselling"
        ),
        "key_features": [
            "MEN1-phenotype with negative MEN1 gene testing — test CDKN1B",
            "Lower penetrance than MEN1; pancreatic NET much rarer",
            "ACTH-secreting pituitary adenomas more common than in MEN1",
            "p27/KIP1 is a cyclin-dependent kinase inhibitor — cell cycle brake",
        ],
        "key_ddx": [
            "MEN1 (MEN1 positive, multiglandular HPT, more PNET)",
            "Sporadic HPT + sporadic pituitary adenoma (coincidence)",
            "AIP FIPA (GH-adenoma-predominant, no HPT)",
        ],
        "onset_age": "3rd-5th decade typically",
        "gh_deficiency": False,
        "mecasermin_indicated": False,
        "autosomal_recessive_risk": False,
    },

    # -- AIP -- FIPA / GH-secreting somatotrophinoma -----------------------------------------
    {
        "gene": "AIP",
        "alt_name": (
            "AIP (AIP-330aa-11q13.2 / AD-LOF -- "
            "FIPA-FAMILIAL-ISOLATED-PITUITARY-ADENOMA -- "
            "GH-SECRETING-SOMATOTROPHINOMA-YOUNG-ONSET-<30YEARS -- "
            "GIGANTISM-IN-CHILD-AIP-UNTIL-PROVEN-OTHERWISE -- "
            "SSA-RESISTANT-PEGVISOMANT-GH-RECEPTOR-ANTAGONIST-PREFERRED)"
        ),
        "protein": (
            "AIP -- 11q13.2 AD LOF -- AIP-330aa -- "
            "Aryl-Hydrocarbon-Receptor-Interacting-Protein-37kDa-Immunophilin-Like-Cochaperone -- "
            "TPR-Tetratricopeptide-Repeat-Domains-Binds-HSP90-AhR-PDE4A5 -- "
            "Chaperone-Function-AhR-Stability-PDE4A5-Regulation -- "
            "AIP-LOF-PDE4A5-Dysregulated-cAMP-PKA-Pathway-Somatotroph-Proliferation -- "
            "AIP-Negative-FIPA-Families-Exist-Unidentified-Gene -- "
            "OMIM-Gene-605555-Disease-FIPA-102200"
        ),
        "locus": "11q13.2",
        "protein_size": "330 aa / 37 kDa",
        "inheritance": (
            "AD (autosomal dominant, LOF); "
            "AIP accounts for ~20% of FIPA families; "
            "GH-secreting macroadenoma at young age (<30 years) — hallmark; "
            "penetrance incomplete (~20-30% in affected families); "
            "SSA (octreotide/lanreotide) typically resistant — pegvisomant more effective; "
            "annual MRI pituitary + GH/IGF-1 in gene-positive family members from adolescence; "
            "AIP-negative FIPA families exist (other loci not yet identified)"
        ),
        "disease_category": (
            "FIPA (Familial Isolated Pituitary Adenoma) — GH-secreting somatotrophinoma predominant (>70%); "
            "acromegaly/gigantism; young onset (<30 years); "
            "larger + more aggressive + SSA-resistant vs sporadic"
        ),
        "disease_pathway": (
            "AIP (aryl hydrocarbon receptor interacting protein) is a co-chaperone protein with TPR domains "
            "that interacts with HSP90, the aryl hydrocarbon receptor (AhR), and PDE4A5 (phosphodiesterase 4A5). "
            "AIP chaperones AhR stability and regulates cAMP signalling via PDE4A5. "
            "AIP LOF → loss of chaperone function → PDE4A5 dysregulation → elevated intracellular cAMP → "
            "PKA activation → somatotroph cell proliferation and GH hypersecretion. "
            "The predominance of GH-secreting adenomas (>70%) in FIPA/AIP is unexplained but may reflect "
            "specific cAMP/PKA sensitivity of somatotrophs compared to other pituitary cell lineages. "
            "Tumours tend to be large macroadenomas at diagnosis — likely due to young age at onset "
            "combined with incomplete/delayed clinical recognition (gigantism may not be diagnosed until extremes). "
            "SSA resistance: octreotide/lanreotide act via somatostatin receptor type 2 (SSTR2); "
            "AIP-mutant somatotrophinomas often have reduced SSTR2 expression → SSA cannot suppress GH effectively. "
            "Pegvisomant (GH receptor antagonist) blocks GH action peripherally → normalises IGF-1 "
            "regardless of pituitary GH output — preferred second-line. "
            "Pasireotide (SSTR1/2/3/5 agonist) may overcome partial SSTR2 resistance."
        ),
        "pathognomonic": (
            "GIGANTISM IN A CHILD/TEENAGER = AIP FIPA UNTIL PROVEN OTHERWISE; "
            "GH-secreting macroadenoma at young age; "
            "POOR RESPONSE TO SOMATOSTATIN ANALOGUES (octreotide/lanreotide) PATHOGNOMONIC for AIP vs sporadic GH adenoma; "
            "pegvisomant + surgery required; AIP-negative FIPA families exist (unidentified gene)"
        ),
        "hormone_profile": (
            "Markedly elevated GH (fails to suppress on OGTT) + elevated IGF-1 + enlarged sella turcica"
        ),
        "severity_sds": "+SDS (tall in childhood/gigantism); final adult height above +3 SDS if untreated",
        "treatment": (
            "Trans-sphenoidal surgery (TSS) first-line — often incomplete due to macroadenoma size; "
            "post-TSS: GH/IGF-1 monitoring; SSA (octreotide/lanreotide) poor response in AIP; "
            "pegvisomant (GH receptor antagonist) more effective; "
            "radiotherapy if persistent; pasireotide (somatostatin type-5 receptor) alternative"
        ),
        "key_features": [
            "Young-onset GH-secreting pituitary macroadenoma — hallmark",
            "AIP accounts for ~20% of FIPA families",
            "Gigantism if pre-pubertal; acromegaly if post-pubertal",
            "SSA-resistant — pegvisomant preferred second line",
            "Annual MRI pituitary + GH/IGF-1 in gene-positive family members",
            "AIP LOF → chaperone function lost → PDE4A5 dysregulation → cAMP/PKA → proliferation",
        ],
        "key_ddx": [
            "MEN1 pituitary adenoma (associated HPT + PNET)",
            "Carney complex PRKAR1A (spotty pigmentation + cardiac myxoma)",
            "McCune-Albright (GNAS somatic mosaic)",
            "Sporadic somatotrophinoma (older, smaller, SSA-responsive)",
        ],
        "onset_age": "Childhood to early 3rd decade",
        "gh_deficiency": False,
        "mecasermin_indicated": False,
        "autosomal_recessive_risk": False,
    },

    # -- PRKAR1A -- Carney complex ------------------------------------------------------------
    {
        "gene": "PRKAR1A",
        "alt_name": (
            "PRKAR1A (PRKAR1A-381aa-17q24.2 / AD-LOF -- "
            "CARNEY-COMPLEX-CNC -- "
            "PPNAD-PARADOXICAL-CORTISOL-RISE-DEXAMETHASONE-PATHOGNOMONIC -- "
            "CARDIAC-MYXOMA-ANY-CHAMBER-RECURS-ANNUAL-ECHO-MANDATORY -- "
            "SPOTTY-SKIN-PIGMENTATION-LIPS-CONJUNCTIVA-GENITAL)"
        ),
        "protein": (
            "PRKAR1A -- 17q24.2 AD LOF -- PRKAR1A-381aa -- "
            "PKA-Regulatory-Subunit-Type-1-Alpha-43kDa-cAMP-Sensing-Subunit -- "
            "R-Subunit-Normally-Sequesters-and-Inhibits-C-Subunit-Catalytic -- "
            "cAMP-Binds-R-Subunit-CNB-A-B-Domains-Releases-C-Subunit-Active -- "
            "PRKAR1A-LOF-Constitutive-PKA-C-Subunit-Activity-cAMP-Independent -- "
            "Adrenal-Cortex-Pituitary-Cardiac-Fibroblast-Tumour-Suppressor -- "
            "OMIM-Gene-188830-Disease-Carney-Complex-160980"
        ),
        "locus": "17q24.2",
        "protein_size": "381 aa / 43 kDa",
        "inheritance": (
            "AD (autosomal dominant, LOF); "
            "cardiac myxoma may occur at any age and any cardiac chamber (left atrium most common but NOT exclusive); "
            "PPNAD (primary pigmented nodular adrenocortical disease): multiple bilateral pigmented adrenal nodules → "
            "ACTH-independent Cushing; PARADOXICAL CORTISOL RISE ON EXTENDED DEXAMETHASONE (Liddle 8-day test); "
            "annual echocardiography mandatory (myxoma may embolise); "
            "spotty skin pigmentation on lips/oral mucosa/conjunctiva/genital skin = lentigines (not café-au-lait)"
        ),
        "disease_category": (
            "Carney complex (CNC) — Cardiac myxoma (30-60%) + Spotty skin pigmentation (lentigines) + "
            "Multiple endocrine tumors: PPNAD (Cushing) + GH excess (20%) + testicular LCCSCT + thyroid follicular adenoma"
        ),
        "disease_pathway": (
            "PRKAR1A encodes the regulatory subunit type 1-alpha (R1α) of cAMP-dependent protein kinase A (PKA). "
            "In normal cells: cAMP binds R1α → releases catalytic (C) subunit → PKA activation (controlled). "
            "PRKAR1A LOF: reduced R1α → constitutive/unconstrained PKA catalytic subunit activity → "
            "downstream phosphorylation of CREB, hormone synthesis enzymes, and cell cycle regulators. "
            "In adrenal cortex: excess PKA activity → constitutive steroidogenesis (cortisol) independent of ACTH → "
            "PPNAD (primary pigmented nodular adrenocortical disease); multiple bilateral small pigmented nodules "
            "(melapnosis — black/brown melanin deposition); "
            "standard dexamethasone suppresses ACTH but adrenal cortisol secretion paradoxically INCREASES "
            "(extended Liddle test phenomenon — proposed mechanism: dexamethasone → GIP/other receptor induction). "
            "In cardiac fibroblasts: PKA hyperactivity → myxoma formation; "
            "myxoma recurs after excision because germline mutation is present in all fibroblasts (not cured by surgery). "
            "Spotty pigmentation: lentigines on sun-exposed and non-sun-exposed areas (lips, genital skin, conjunctiva) — "
            "melanocyte PKA activation; distinct from café-au-lait spots (NF1, McCune-Albright)."
        ),
        "pathognomonic": (
            "PARADOXICAL CORTISOL RISE ON DEXAMETHASONE (Liddle test / extended low-dose dexamethasone) = "
            "PPNAD / Carney complex PATHOGNOMONIC; "
            "normal 24h cortisol on standard dexamethasone but RISE (paradoxical) on 8-day Liddle test; "
            "CARDIAC MYXOMA recurs after excision and occurs in unusual locations (any chamber, not just left atrium); "
            "spotty skin pigmentation on lips/oral mucosa/conjunctiva/genital skin = lentigines (not café-au-lait)"
        ),
        "hormone_profile": (
            "ACTH-independent Cushing (low ACTH + elevated cortisol); "
            "elevated IGF-1/GH (pituitary); elevated estrogen/testosterone (testicular tumor)"
        ),
        "severity_sds": "N/A — endocrine tumor syndrome; variable expressivity",
        "treatment": (
            "Annual echocardiography (cardiac myxoma may embolise/obstruct); "
            "bilateral adrenalectomy for PPNAD-Cushing; "
            "pituitary surgery/SSA for GH excess; testicular US annually; thyroid US; "
            "skin surveillance; hormone replacement post-adrenalectomy"
        ),
        "key_features": [
            "Cardiac myxoma: any chamber (not just LA), recurs — annual echo mandatory",
            "PPNAD: multiple bilateral pigmented adrenal nodules — ACTH-independent Cushing",
            "Paradoxical cortisol rise on extended dexamethasone = PPNAD pathognomonic",
            "Spotty pigmentation: lips, conjunctiva, oral mucosa, genitalia",
            "LCCSCT: testicular tumor on US; calcifications; estrogen excess in boys",
            "PKA pathway: PRKAR1A LOF → unconstrained PKA-catalytic subunit activity",
        ],
        "key_ddx": [
            "ACTH-dependent Cushing (pituitary or ectopic — ACTH elevated)",
            "Isolated cardiac myxoma (sporadic, single, left atrium)",
            "PRKACA (CNC type 2 — bilateral adrenal hyperplasia Cushing, no myxoma)",
        ],
        "onset_age": "Variable; myxoma any age; Cushing typically 2nd-3rd decade",
        "gh_deficiency": False,
        "mecasermin_indicated": False,
        "autosomal_recessive_risk": False,
    },

    # -- VHL -- VHL syndrome ------------------------------------------------------------------
    {
        "gene": "VHL",
        "alt_name": (
            "VHL (VHL-213aa-3p25.3 / AD-LOF -- "
            "VHL-SYNDROME-HEMANGIOBLASTOMA+CLEAR-CELL-RCC+PHEO-TRIAD -- "
            "BELZUTIFAN-HIF-2ALPHA-INHIBITOR-FDA-APPROVED-2021 -- "
            "RETINAL-ANGIOMA-FIRST-MANIFESTATION -- "
            "TYPE-2C-PHEO-ONLY-HIGH-RISK)"
        ),
        "protein": (
            "VHL -- 3p25.3 AD LOF -- VHL-213aa -- "
            "pVHL-Von-Hippel-Lindau-Protein-24kDa-E3-Ubiquitin-Ligase-Complex-Component -- "
            "Elongin-B-C-CUL2-RBX1-VHL-Complex-E3-Ligase -- "
            "HIF-1alpha-2alpha-Substrate-Hydroxylated-Pro402-Pro564-Binds-pVHL-Polyubiquitinated-Degraded -- "
            "VHL-LOF-HIF-1alpha-2alpha-Constitutively-Active-Normoxia -- "
            "HIF-Target-Genes-VEGF-EPO-PDGF-GLUT1-Angiogenesis-Erythropoiesis -- "
            "Belzutifan-MK-6482-HIF-2alpha-Selective-Inhibitor-FDA-2021-VHL-RCC-Hemangioblastoma -- "
            "OMIM-Gene-608537-Disease-VHL-193300"
        ),
        "locus": "3p25.3",
        "protein_size": "213 aa / 24 kDa",
        "inheritance": (
            "AD (autosomal dominant, LOF, two-hit tumour suppressor); "
            "Type 1 (truncating/deletion): high hemangioblastoma + RCC risk, low pheo; "
            "Type 2A (missense): high pheo + hemangioblastoma, lower RCC; "
            "Type 2B (missense): high pheo + high RCC + hemangioblastoma — all three; "
            "Type 2C (missense): pheo ONLY — no hemangioblastoma or RCC; "
            "belzutifan (HIF-2α inhibitor) FDA approved 2021 for VHL-associated RCC/hemangioblastoma/PNET; "
            "retinal angioma may be first presentation in childhood — ophthalmology surveillance essential"
        ),
        "disease_category": (
            "VHL syndrome — Hemangioblastoma (cerebellum/retina/spinal cord) + "
            "Clear cell RCC (most common cause of death) + Pheochromocytoma (type 2) + "
            "Pancreatic NETs + Endolymphatic sac tumor (ELST)"
        ),
        "disease_pathway": (
            "pVHL is the substrate recognition component of an E3 ubiquitin ligase complex (VHL-Elongin B/C-CUL2-RBX1). "
            "Under normoxia: HIFα subunits (HIF-1α, HIF-2α) are hydroxylated on proline residues (Pro402/Pro564) by "
            "PHD enzymes → recognised by VHL → polyubiquitination → proteasomal degradation. "
            "Under hypoxia: PHDs inactive → HIFα accumulates → dimerises with ARNT (HIF-1β) → "
            "transcribes VEGF, EPO, GLUT1, PDGF, LDHA → angiogenesis, erythropoiesis, glycolysis. "
            "VHL LOF: HIFα constitutively active in normoxia → pseudo-hypoxic signalling → "
            "VEGF drives hemangioblastoma formation (highly vascular tumours); "
            "EPO drives paraneoplastic erythrocytosis (elevated haemoglobin) in some RCC patients; "
            "clear cell RCC is the principal cancer with VHL loss in renal tubular epithelial cells. "
            "HIF-2α is the primary oncogenic HIFα isoform in VHL-mutated clear cell RCC → "
            "belzutifan selectively inhibits HIF-2α dimerisation with ARNT → "
            "reduces VEGF and downstream proliferation. "
            "Pheo in VHL: predominantly norepinephrine-secreting (normetanephrine elevated); "
            "adrenal chromaffin cells rely on HIF-2α for catecholamine synthesis regulation. "
            "ELST (endolymphatic sac tumour): aggressive locally; tinnitus + hearing loss + VHL = ELST until excluded."
        ),
        "pathognomonic": (
            "HEMANGIOBLASTOMA (cerebellar/spinal/retinal) + CLEAR CELL RCC + PHEOCHROMOCYTOMA TRIAD = VHL PATHOGNOMONIC; "
            "retinal angioma may be first presentation (ophthalmology finding); "
            "pheo in VHL: norepinephrine-secreting (normetanephrine elevated); "
            "ELST: tinnitus + hearing loss + VHL = endolymphatic sac tumor; "
            "type 2C: pheo only (high pheo risk, NO hemangioblastoma)"
        ),
        "hormone_profile": (
            "Elevated normetanephrine (pheo); RCC: paraneoplastic erythrocytosis (EPO); "
            "PNET: usually non-functional"
        ),
        "severity_sds": "N/A — hereditary cancer syndrome",
        "treatment": (
            "Annual MRI brain/spine/abdomen from age 15; fundoscopy annually from childhood; "
            "nephron-sparing surgery for RCC <3cm; laser/cryo for retinal angiomas; "
            "pheo: alpha-blockade then adrenalectomy; ELST: audiometry + MRI temporal bone; "
            "belzutifan (HIF-2α inhibitor) for VHL-associated tumors (FDA 2021)"
        ),
        "key_features": [
            "Retinal angioma may be first manifestation — ophthalmology essential",
            "RCC: nephron-sparing surgery preferred (<3cm); radical only if large",
            "Belzutifan (HIF-2α inhibitor) FDA-approved 2021 for VHL RCC/hemangioblastoma/PNET",
            "Type 1 (LOF): high hemangioblastoma/RCC, low pheo; Type 2 (missense): high pheo",
            "pVHL is E3 ubiquitin ligase for HIF-1α/2α — LOF → HIF constitutively active → VEGF/EPO/PDGF",
        ],
        "key_ddx": [
            "Sporadic hemangioblastoma (unilateral, older)",
            "Sporadic clear cell RCC (unilateral, no pheo)",
            "SDHB pheo (extra-adrenal, norepinephrine, high malignancy)",
            "MEN2 pheo (MTC + parathyroid; norepinephrine predominant)",
        ],
        "onset_age": "Retinal angioma childhood-adolescence; cerebellar hemangioblastoma 3rd-4th decade; RCC 4th-5th decade",
        "gh_deficiency": False,
        "mecasermin_indicated": False,
        "autosomal_recessive_risk": False,
    },

    # -- SDHB -- Hereditary Pheo/PGL type 4 (PGL4) -------------------------------------------
    {
        "gene": "SDHB",
        "alt_name": (
            "SDHB (SDHB-280aa-1p36.13 / AD-LOF -- "
            "PGL4-EXTRA-ADRENAL-PARAGANGLIOMA-PREDOMINANT -- "
            "HIGHEST-MALIGNANCY-30-50pct-ALL-HEREDITARY-PHEO -- "
            "CARNEY-STRATAKIS-DYAD-GIST+PGL-PATHOGNOMONIC -- "
            "METHOXYTYRAMINE-DOPAMINE-METABOLITE-DOTATATE-PET-CT-STAGING)"
        ),
        "protein": (
            "SDHB -- 1p36.13 AD LOF -- SDHB-280aa -- "
            "Succinate-Dehydrogenase-Iron-Sulfur-Subunit-32kDa-Complex-II-Subunit -- "
            "Mitochondrial-Respiratory-Chain-Complex-II-TCA-Cycle-SDH-Electron-Transport -- "
            "SDH-Complex-SDHA-SDHB-SDHC-SDHD-Four-Subunits-Succinate-Fumarate -- "
            "SDHB-Iron-Sulfur-Clusters-3Fe-4S-4Fe-4S-Electron-Shuttling -- "
            "SDH-LOF-Succinate-Accumulates-Inhibits-PHDs-Pseudo-Hypoxia-HIF-Activation -- "
            "SDHB-IHC-Loss-of-Staining-Confirms-SDHx-Mutation-In-Tumour -- "
            "OMIM-Gene-185470-Disease-PGL4-115310"
        ),
        "locus": "1p36.13",
        "protein_size": "280 aa / 32 kDa",
        "inheritance": (
            "AD (autosomal dominant, LOF, two-hit); "
            "extra-adrenal paraganglioma (PGL) predominant — retroperitoneal, thoracic, head/neck; "
            "adrenal pheo also occurs but less common than SDHD; "
            "HIGHEST malignancy risk of all hereditary pheo syndromes (30-50% lifetime); "
            "Carney-Stratakis dyad: PGL + GIST (without NF1) = SDHx mutation; "
            "DOTATATE PET/CT superior to FDG-PET for SDH-mutated tumours (somatostatin receptor expression); "
            "annual surveillance from age 6 in gene-positive relatives"
        ),
        "disease_category": (
            "Hereditary Pheo/PGL type 4 (PGL4) — Extra-adrenal paraganglioma (predominant) + "
            "Adrenal pheo; HIGHEST MALIGNANCY RISK 30-50%; Carney-Stratakis dyad (GIST + PGL); "
            "clear cell RCC association; dopamine-secreting"
        ),
        "disease_pathway": (
            "Succinate dehydrogenase (SDH, Complex II) catalyses oxidation of succinate to fumarate "
            "in the TCA cycle and feeds electrons into the mitochondrial respiratory chain. "
            "SDH complex: SDHA (catalytic FAD-binding), SDHB (iron-sulfur subunit, electron shuttle), "
            "SDHC, SDHD (anchoring subunits). "
            "SDHB LOF → SDH complex unstable/non-functional → succinate accumulates in mitochondria and cytoplasm → "
            "succinate inhibits prolyl hydroxylase domain (PHD) enzymes → HIF-1α/2α not hydroxylated → "
            "not degraded by VHL → constitutively active → pseudo-hypoxic signalling → "
            "VEGF, catecholamine synthesis genes, and chromaffin/glomus cell proliferation. "
            "SDHB IHC: anti-SDHB antibody stains mitochondria in normal tissue; "
            "loss of SDHB staining in a tumour (even with non-SDHB SDHx mutation) indicates "
            "SDH complex destabilisation — functional diagnostic marker of any SDHx germline variant. "
            "Dopaminergic secretion: extra-adrenal PGL cells lack phenylethanolamine-N-methyltransferase (PNMT, "
            "which converts norepinephrine → adrenaline); SDHx-mutated extra-adrenal PGLs secrete "
            "predominantly dopamine → methoxytyramine elevated; may be clinically 'silent' (no hypertension). "
            "GIST connection: Carney-Stratakis syndrome — GIST with loss of SDHB IHC + PGL; "
            "SDH-deficient GIST lacks KIT/PDGFRA mutations (separate from standard GIST pathway)."
        ),
        "pathognomonic": (
            "EXTRA-ADRENAL PARAGANGLIOMA + GIST IN SAME PATIENT = CARNEY-STRATAKIS DYAD = "
            "SDHB/SDHC/SDHD PATHOGNOMONIC; "
            "SDHB immunohistochemistry: loss of SDHB staining in any SDHx-mutated tumor PATHOGNOMONIC; "
            "methoxytyramine (plasma) elevated = dopaminergic secretion = extra-adrenal origin; "
            "HIGHEST malignancy risk of all hereditary pheo (30-50% lifetime)"
        ),
        "hormone_profile": (
            "Elevated plasma methoxytyramine (dopamine metabolite) — dopaminergic secretion; "
            "normetanephrine may also be elevated; often asymptomatic despite large tumor (dopamine not hypertensogenic)"
        ),
        "severity_sds": "N/A — endocrine tumor syndrome; high malignancy risk",
        "treatment": (
            "Annual plasma metanephrines/methoxytyramine + MRI whole body (neck to pelvis) from age 6; "
            "DOTATATE PET/CT for metastatic staging (superior to FDG for SDH-mutated); "
            "high-dose 131I-MIBG or DOTATATE-PRRT for metastatic; "
            "surgical debulking; sunitinib/temozolomide for progressive metastatic disease"
        ),
        "key_features": [
            "Highest malignancy risk of all hereditary pheo syndromes (30-50%)",
            "Extra-adrenal paraganglioma predominant (retroperitoneal, thoracic, head/neck)",
            "Dopamine-secreting: methoxytyramine elevated; may lack hypertension despite large tumor",
            "SDHB IHC: loss of staining confirms SDHx mutation (functional)",
            "Carney-Stratakis dyad: PGL + GIST without NF1",
            "DOTATATE PET/CT preferred for staging and surveillance",
            "Annual surveillance from age 6 in gene-positive relatives",
        ],
        "key_ddx": [
            "VHL pheo (normetanephrine dominant; RCC + hemangioblastoma)",
            "RET pheo (MTC + parathyroid + adrenal; metanephrine-dominant)",
            "NF1 pheo (neurofibromas + café-au-lait; adrenal, bilateral)",
            "MAX pheo (bilateral adrenal; adrenaline-secreting; paternal imprinting)",
        ],
        "onset_age": "2nd-3rd decade typical (can be childhood)",
        "gh_deficiency": False,
        "mecasermin_indicated": False,
        "autosomal_recessive_risk": False,
    },

    # -- MAX -- Hereditary Pheo/PGL type 5 (PGL5) / Paternal imprinting ----------------------
    {
        "gene": "MAX",
        "alt_name": (
            "MAX (MAX-160aa-14q23.3 / AD-LOF-PATERNAL-IMPRINTING -- "
            "PGL5-BILATERAL-ADRENAL-PHEO-PREDOMINANT -- "
            "MATERNAL-CARRIERS-NOT-AT-RISK-PATERNAL-IMPRINTING -- "
            "METANEPHRINE-ELEVATED-ADRENALINE-ADRENAL-ORIGIN -- "
            "CORTICAL-SPARING-ADRENALECTOMY-PREFERRED-BILATERAL)"
        ),
        "protein": (
            "MAX -- 14q23.3 AD LOF-Paternal-Imprinting -- MAX-160aa -- "
            "MYC-Associated-Factor-X-18kDa-HLH-LZ-Transcription-Factor -- "
            "Obligate-Heterodimerisation-Partner-MYC-MYCN-MYCL-MNT-MXD -- "
            "MYC-MAX-Dimer-Activates-E-Box-Target-Genes-Proliferation-Metabolism -- "
            "MAX-MAX-Homodimer-Represses-MYC-Targets -- "
            "MAX-LOF-MYC-Heterodimerisation-Lost-MYC-Hyperactive-Proliferation -- "
            "Paternal-Imprinting-Maternally-Expressed-MAX-Not-Expressed-in-Adrenal-from-Maternal-Allele -- "
            "OMIM-Gene-154950-Disease-PGL5-614165"
        ),
        "locus": "14q23.3",
        "protein_size": "160 aa / 18 kDa",
        "inheritance": (
            "AD LOF with PATERNAL IMPRINTING — the maternal MAX allele is imprinted (silenced) in adrenal chromaffin cells; "
            "only the paternal MAX allele is expressed in adrenal; "
            "therefore maternally-inherited MAX LOF variants do NOT cause pheo (imprinted allele already silent); "
            "ONLY paternally-inherited MAX LOF causes pheo (paternal expressed allele lost); "
            "bilateral adrenal pheo — hallmark; "
            "cortical-sparing adrenalectomy preferred to preserve glucocorticoid production; "
            "lower malignancy risk than SDHB; "
            "genetic counselling: critical to establish parent-of-origin before advising family members"
        ),
        "disease_category": (
            "Hereditary Pheo/PGL type 5 (PGL5) — Bilateral adrenal pheochromocytoma (predominant); "
            "PATERNAL IMPRINTING: maternally inherited MAX does NOT cause pheo; MYC pathway dysregulation"
        ),
        "disease_pathway": (
            "MAX (MYC-associated factor X) is an obligate helix-loop-helix leucine zipper (HLH-LZ) transcription factor "
            "that dimerises with MYC oncoproteins (cMYC, MYCN, MYCL) to activate E-box target genes "
            "promoting proliferation, growth, and metabolism. "
            "MAX also forms homodimers (MAX-MAX) that compete with MYC-MAX and repress MYC target genes. "
            "MAX LOF: MYC cannot dimerize efficiently with MAX → loss of both activation (MYC-MAX) and "
            "repression (MAX-MAX) — net effect is MYC hyperactivity due to disinhibition of alternative "
            "MYC partners (MXD, MNT relieved). "
            "Paternal imprinting mechanism: the maternal MAX allele is epigenetically silenced (imprinted) "
            "specifically in adrenal chromaffin cells; only paternal MAX is expressed; "
            "therefore a maternal germline LOF variant is silent in adrenal tissue (imprinted allele already off); "
            "paternal germline LOF → loss of the ONLY expressed allele → biallelic effective loss in chromaffin cells → "
            "MYC hyperactivity → pheo. "
            "Adrenal origin: chromaffin cells contain PNMT → convert norepinephrine → adrenaline → "
            "metanephrine is the predominant elevated metabolite (distinguishes from SDHB/VHL extra-adrenal which secrete dopamine/norepinephrine). "
            "Bilateral: both adrenal glands carry the germline LOF; both adrenal chromaffin populations vulnerable; "
            "cortical-sparing bilateral adrenalectomy preserves adrenocortical function → avoids lifelong glucocorticoid dependence."
        ),
        "pathognomonic": (
            "BILATERAL ADRENAL PHEO IN YOUNG ADULT WITH NEGATIVE RET/VHL/SDHB TESTING = MAX UNTIL PROVEN OTHERWISE; "
            "PATERNAL IMPRINTING: only paternally-inherited MAX allele expressed in adrenal chromaffin cells — "
            "maternal MAX mutations do NOT cause pheo; "
            "adrenaline-secreting (metanephrine elevated) = adrenal origin; "
            "bilateral simultaneous or metachronous pheo"
        ),
        "hormone_profile": (
            "Elevated plasma metanephrine (adrenaline metabolite) — bilateral adrenal origin; "
            "adrenaline-dominant secretion"
        ),
        "severity_sds": "N/A — endocrine tumor syndrome; lower malignancy risk than SDHB",
        "treatment": (
            "Plasma metanephrines annually; MRI adrenals annually; "
            "cortical-sparing adrenalectomy if possible (bilateral → adrenal insufficiency risk); "
            "glucocorticoid replacement if bilateral adrenalectomy; "
            "alpha-blockade pre-operatively; "
            "genetic counselling: paternal imprinting — maternal carriers not at risk"
        ),
        "key_features": [
            "Bilateral adrenal pheo — hallmark of MAX (and RET MEN2)",
            "PATERNAL IMPRINTING: test only if inherited from father; maternal carriers not at risk",
            "MYC-MAX pathway: MAX normally suppresses MYC targets; LOF → MYC hyperactive → proliferation",
            "Adrenaline-secreting (adrenal origin) — metanephrine elevated",
            "Cortical-sparing adrenalectomy preferred to preserve cortisol production",
            "Lower malignancy risk than SDHB; still higher than sporadic",
        ],
        "key_ddx": [
            "RET MEN2 bilateral pheo (MTC + parathyroid; bilateral)",
            "VHL pheo (normetanephrine; RCC + hemangioblastoma)",
            "NF1 pheo (unilateral adrenal; café-au-lait; neurofibromas)",
            "SDHB pheo (extra-adrenal predominant; methoxytyramine)",
        ],
        "onset_age": "3rd-4th decade typical",
        "gh_deficiency": False,
        "mecasermin_indicated": False,
        "autosomal_recessive_risk": False,
    },
]


# ---------------------------------------------------------------------------
# Cohort simulation rates
# ---------------------------------------------------------------------------
_RATES = {
    "MEN1":    {"surgery": 0.75, "pheo": 0.0,  "malignant": 0.25, "surveillance": 0.98, "bilateral": 0.95},
    "RET":     {"surgery": 0.90, "pheo": 0.45, "malignant": 0.20, "surveillance": 0.95, "bilateral": 0.30},
    "CDKN1B":  {"surgery": 0.50, "pheo": 0.0,  "malignant": 0.10, "surveillance": 0.90, "bilateral": 0.40},
    "AIP":     {"surgery": 0.85, "pheo": 0.0,  "malignant": 0.05, "surveillance": 0.95, "bilateral": 0.0},
    "PRKAR1A": {"surgery": 0.70, "pheo": 0.0,  "malignant": 0.05, "surveillance": 0.98, "bilateral": 0.60},
    "VHL":     {"surgery": 0.65, "pheo": 0.30, "malignant": 0.30, "surveillance": 0.98, "bilateral": 0.40},
    "SDHB":    {"surgery": 0.60, "pheo": 0.90, "malignant": 0.40, "surveillance": 0.95, "bilateral": 0.15},
    "MAX":     {"surgery": 0.70, "pheo": 1.0,  "malignant": 0.10, "surveillance": 0.95, "bilateral": 0.80},
}

_AGE_RANGES = {
    "MEN1":    (25, 50),
    "RET":     (5,  45),
    "CDKN1B":  (30, 55),
    "AIP":     (10, 30),
    "PRKAR1A": (15, 50),
    "VHL":     (15, 45),
    "SDHB":    (18, 45),
    "MAX":     (20, 45),
}

_TYPE_VARIANTS = {
    "MEN1":    ["HPT+Gastrinoma", "HPT+Prolactinoma", "HPT+Insulinoma", "HPT+Non-functional PNET"],
    "RET":     ["MEN2A Cys634Arg", "MEN2A Cys634Tyr", "MEN2B Met918Thr", "FMTC Cys611Tyr"],
    "CDKN1B":  ["HPT+ACTH adenoma", "HPT+Prolactinoma", "HPT only", "HPT+GH adenoma"],
    "AIP":     ["Somatotrophinoma (GH)", "Somatotrophinoma (GH)", "Mixed GH+PRL", "Non-functional"],
    "PRKAR1A": ["PPNAD Cushing", "PPNAD+Myxoma", "PPNAD+GH excess", "Myxoma only"],
    "VHL":     ["Type 1 Hemangioblastoma+RCC", "Type 2A Pheo+Hemangioblastoma", "Type 2B Pheo+RCC+Hemangioblastoma", "Type 2C Pheo only"],
    "SDHB":    ["Retroperitoneal PGL", "Thoracic PGL", "Adrenal pheo", "Head-neck PGL"],
    "MAX":     ["Bilateral adrenal pheo", "Bilateral adrenal pheo", "Unilateral adrenal pheo", "Bilateral+recurrent"],
}


def _make_cohort(entry: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene = entry["gene"]
    r = _RATES[gene]
    age_min, age_max = _AGE_RANGES[gene]
    variants = _TYPE_VARIANTS[gene]

    cohort = []
    sexes = ["M", "F"]
    for i in range(n):
        sex = rng.choice(sexes)
        age_dx = round(rng.uniform(age_min, age_max), 1)
        type_variant = rng.choice(variants)
        had_surgery = rng.random() < r["surgery"]
        pheo_present = rng.random() < r["pheo"]
        malignant_disease = rng.random() < r["malignant"]
        surveillance_active = rng.random() < r["surveillance"]
        bilateral_disease = rng.random() < r["bilateral"]

        patient = {
            "patient_id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "sex": sex,
            "age_at_diagnosis_years": age_dx,
            "type_variant": type_variant,
            "had_surgery": had_surgery,
            "pheo_present": pheo_present,
            "malignant_disease": malignant_disease,
            "surveillance_active": surveillance_active,
            "bilateral_disease": bilateral_disease,
        }
        cohort.append(patient)
    return cohort


def generate_overview():
    all_patients = []
    for idx, entry in enumerate(MEN_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(entry, seed))

    total = len(all_patients)
    surgery_count    = sum(1 for p in all_patients if p["had_surgery"])
    pheo_count       = sum(1 for p in all_patients if p["pheo_present"])
    malignant_count  = sum(1 for p in all_patients if p["malignant_disease"])
    surveillance_count = sum(1 for p in all_patients if p["surveillance_active"])
    bilateral_count  = sum(1 for p in all_patients if p["bilateral_disease"])

    gene_summary = {}
    for idx, entry in enumerate(MEN_GENES):
        gene = entry["gene"]
        cohort = _make_cohort(entry, SEED_BASE + idx)
        gene_summary[gene] = {
            "gene": gene,
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_category": entry["disease_category"],
            "pathognomonic": entry["pathognomonic"][:300],
            "hormone_profile": entry["hormone_profile"],
            "n_patients": len(cohort),
            "surgery_pct": round(100 * sum(1 for p in cohort if p["had_surgery"]) / len(cohort), 1),
            "pheo_pct": round(100 * sum(1 for p in cohort if p["pheo_present"]) / len(cohort), 1),
            "malignant_pct": round(100 * sum(1 for p in cohort if p["malignant_disease"]) / len(cohort), 1),
            "surveillance_pct": round(100 * sum(1 for p in cohort if p["surveillance_active"]) / len(cohort), 1),
            "bilateral_pct": round(100 * sum(1 for p in cohort if p["bilateral_disease"]) / len(cohort), 1),
            "avg_age_dx_years": round(sum(p["age_at_diagnosis_years"] for p in cohort) / len(cohort), 1),
        }

    return {
        "atlas": "Hereditary-Multiple-Endocrine-Neoplasia-Atlas",
        "subtitle": "Complete 8-Gene MEN Reference -- MEN1/RET/CDKN1B/AIP/PRKAR1A/VHL/SDHB/MAX",
        "genes_covered": [e["gene"] for e in MEN_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "surgery_pct": round(100 * surgery_count / total, 1),
            "pheo_pct": round(100 * pheo_count / total, 1),
            "malignant_pct": round(100 * malignant_count / total, 1),
            "surveillance_pct": round(100 * surveillance_count / total, 1),
            "bilateral_pct": round(100 * bilateral_count / total, 1),
        },
        "gene_summary": gene_summary,
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(MEN_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(entry, seed)
        breakdown.append({
            "gene": entry["gene"],
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"],
            "disease_category": entry["disease_category"],
            "disease_pathway": entry["disease_pathway"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "key_features": entry["key_features"],
            "key_ddx": entry["key_ddx"],
            "hormone_profile": entry["hormone_profile"],
            "onset_age": entry["onset_age"],
            "severity_sds": entry["severity_sds"],
            "gh_deficiency": entry["gh_deficiency"],
            "mecasermin_indicated": entry["mecasermin_indicated"],
            "autosomal_recessive_risk": entry["autosomal_recessive_risk"],
            "n_patients": len(cohort),
            "surgery_pct": round(100 * sum(1 for p in cohort if p["had_surgery"]) / len(cohort), 1),
            "pheo_pct": round(100 * sum(1 for p in cohort if p["pheo_present"]) / len(cohort), 1),
            "malignant_pct": round(100 * sum(1 for p in cohort if p["malignant_disease"]) / len(cohort), 1),
            "surveillance_pct": round(100 * sum(1 for p in cohort if p["surveillance_active"]) / len(cohort), 1),
            "bilateral_pct": round(100 * sum(1 for p in cohort if p["bilateral_disease"]) / len(cohort), 1),
            "avg_age_dx_years": round(sum(p["age_at_diagnosis_years"] for p in cohort) / len(cohort), 1),
            "sample_patients": cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["protein"].split(" --")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["disease_category"],
                "disease_pathway": entry["disease_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:400],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "hormone_profile": entry["hormone_profile"],
                "onset_age": entry["onset_age"],
                "severity_sds": entry["severity_sds"],
                "gh_deficiency": entry["gh_deficiency"],
                "mecasermin_indicated": entry["mecasermin_indicated"],
            }
            for entry in MEN_GENES
        },
        "men_glossary": {
            "Multiple Endocrine Neoplasia — Classification and Genetics": (
                "Multiple endocrine neoplasia (MEN) syndromes are hereditary disorders causing synchronous or "
                "metachronous tumours in two or more endocrine glands. "
                "MEN1 (Menin, 11q13.1, AD LOF): 3P triad — Parathyroid (>90%) + Pituitary (40%) + Pancreatic NET (40-70%); "
                "gastrinoma most common PNET and leading cause of death; thymectomy mandatory at parathyroidectomy. "
                "MEN2A (RET GOF, 10q11.21): MTC (>90%) + Pheo (50%) + HPT (20-30%); "
                "Cys634Arg most common; pheo MUST be excluded before thyroid surgery. "
                "MEN2B (RET Met918Thr, exon 16): MTC (most aggressive, infancy) + Pheo + mucosal neuromas + Marfanoid; "
                "ATA risk D — prophylactic thyroidectomy within 6 months of birth. "
                "MEN4 (CDKN1B/p27, 12p13.1, AD LOF): MEN1-phenotype with negative MEN1 sequencing; "
                "HPT + pituitary adenoma (ACTH-predominant); PNET rare; lower penetrance. "
                "Related syndromes in this atlas: FIPA (AIP), Carney complex (PRKAR1A), VHL, PGL4 (SDHB), PGL5 (MAX)."
            ),
            "MEN1 3P Triad — Surveillance Protocol": (
                "MEN1 surveillance begins at age 5 in confirmed gene carriers. "
                "Annual biochemical panel: serum calcium, PTH (parathyroid); fasting gastrin + secretin stimulation "
                "test if gastrin borderline (gastric NET/ZES); prolactin, IGF-1 (pituitary); fasting glucose, insulin (insulinoma). "
                "Imaging: MRI pituitary annually (adenoma detection); EUS (endoscopic ultrasound) every 1-2 years "
                "for PNET (CT has limited sensitivity for small duodenal gastrinomas); "
                "CT chest annually for thymic NET surveillance. "
                "SURGICAL RULE — SUBTOTAL PARATHYROIDECTOMY: remove 3.5 of 4 glands (not single-gland) + "
                "THYMECTOMY IN SAME OPERATION (prophylactic — thymic NET rare but highest MEN1 mortality). "
                "ZES management: high-dose PPI (omeprazole 80-160mg/day) controls acid in >90%; "
                "somatostatin analogues (octreotide LAR) for functional NETs; PNET >2cm or enlarging → surgery. "
                "INSULINOMA: fasting hypoglycaemia test + 72h fast; diazoxide, everolimus, or surgical enucleation. "
                "Prolactinoma: dopamine agonists (cabergoline) first-line; surgery if macroadenoma with mass effect."
            ),
            "Medullary Thyroid Carcinoma (MTC) — ATA Risk Stratification and Prophylactic Thyroidectomy Timing": (
                "The American Thyroid Association (ATA) 2015 guidelines stratify RET mutations by phaeochromocytoma "
                "and MTC penetrance/aggressiveness into four risk categories (A-D), determining timing of prophylactic thyroidectomy. "
                "RISK D (highest): Met918Thr (exon 16, MEN2B) — MTC penetrance near 100%, may occur in infancy; "
                "thyroidectomy WITHIN 6 MONTHS OF BIRTH or as soon as possible after diagnosis. "
                "RISK C: Cys634Phe/Gly/Arg/Ser/Tyr/Trp (exon 11) — thyroidectomy before age 5. "
                "RISK B: Cys609/611/618/620 (exon 10), Cys630/634 — thyroidectomy before age 5-10; "
                "calcitonin monitoring guides timing. "
                "RISK A: other mutations — calcitonin-guided, may defer to puberty if calcitonin normal. "
                "POST-THYROIDECTOMY MONITORING: calcitonin + CEA every 6 months; "
                "doubling time <6 months = poor prognosis; vandetanib or cabozantinib for metastatic MTC. "
                "CARDINAL RULE — PHEO BEFORE THYROID SURGERY: "
                "always measure plasma metanephrines BEFORE thyroid surgery; "
                "undetected pheo → adrenergic storm during induction anaesthesia → fatal hypertensive crisis."
            ),
            "Pheochromocytoma — Alpha-Blockade Before Surgery Mandatory": (
                "Pheochromocytoma (adrenal) and paraganglioma (extra-adrenal) require mandatory alpha-adrenergic blockade "
                "before surgical resection to prevent perioperative hypertensive crisis. "
                "MECHANISM: catecholamine release during tumour manipulation → massive vasoconstriction → "
                "hypertensive emergency; without blockade, mortality from perioperative crisis can reach 25-50%. "
                "PHENOXYBENZAMINE: non-selective, irreversible alpha-blocker; gold standard; "
                "start 2-4 weeks pre-operatively; dose titrate until postural hypotension; "
                "allows vascular volume expansion (catecholamines cause chronic vasoconstriction → reduced circulating volume). "
                "ALTERNATIVE: doxazosin or prazosin (selective alpha-1, reversible) — shorter duration, more flexible. "
                "BETA-BLOCKER: add only AFTER alpha-blockade established (never first — unopposed alpha-stimulation worsens hypertension); "
                "used for tachycardia/arrhythmia control pre-operatively. "
                "HIGH-SALT DIET + FLUID: pre-operative volume loading prevents profound hypotension post-tumour removal "
                "(catecholamine withdrawal → vasodilation → crash). "
                "GENETIC TESTING ORDER OF PRIORITY: RET → VHL → SDHB/SDHD → MAX → NF1 → TMEM127. "
                "BIOCHEMISTRY: plasma metanephrines (sensitivity >97%) preferred over urinary catecholamines."
            ),
            "VHL Disease — Belzutifan and HIF Pathway": (
                "Belzutifan (Welireg, MK-6482) is the first approved HIF-2α selective inhibitor, "
                "FDA approved August 2021 for adults with VHL disease-associated renal cell carcinoma (RCC), "
                "CNS hemangioblastomas, or pancreatic NETs not requiring immediate surgery. "
                "MECHANISM: pVHL normally degrades HIF-1α and HIF-2α; VHL LOF → HIF-2α constitutively active → "
                "VEGF/EPO/PDGF → tumour angiogenesis; belzutifan binds HIF-2α directly → "
                "blocks HIF-2α/ARNT (HIF-1β) dimerisation → reduces VEGF transcription. "
                "CLINICAL EVIDENCE: LITESPARK-004 trial — 61% response rate in VHL-associated RCC; "
                "22% response in hemangioblastoma; 91% response in PNET. "
                "SIDE EFFECTS: anaemia (EPO reduction — HIF-2α drives EPO), hypoxia (dose-related); "
                "monitor haemoglobin monthly; dose reduce if Hb <9g/dL; "
                "headache, dizziness (EPO-mediated); potential teratogenicity. "
                "LIMITATIONS: not active in non-VHL clear cell RCC; VHL-specific mechanism; "
                "HIF-1α-dominant tumours may not respond as well. "
                "COMPLEMENTARY ROLE: belzutifan does not replace surgery for large RCC (>3cm) or "
                "symptomatic/growing hemangioblastoma — used to reduce tumour burden or delay surgery."
            ),
            "SDHx Immunohistochemistry — Functional Confirmation of Germline Variants": (
                "SDHB immunohistochemistry (IHC) is a validated surrogate marker of SDH complex functional loss "
                "in tumour tissue, applicable regardless of which SDHx subunit (SDHA/SDHB/SDHC/SDHD) carries the mutation. "
                "MECHANISM: SDH complex requires all four subunits for stability; "
                "if any subunit (SDHA/B/C/D) is non-functional → entire complex degrades → "
                "SDHB protein (detected by anti-SDHB antibody) is absent in tumour cells. "
                "INTERPRETATION: normal staining (mitochondrial granular pattern) = SDH complex intact; "
                "loss of cytoplasmic granular SDHB staining (with positive internal control in stromal cells) = "
                "SDH complex non-functional = SDHx germline mutation confirmed functionally. "
                "SDHA IHC EXTRA STEP: if SDHA mutation suspected, SDHA IHC performed first; "
                "SDHA LOF → loss of both SDHA AND SDHB staining. "
                "CLINICAL UTILITY: confirms pathogenicity of VUS (variants of uncertain significance) in SDHx; "
                "screens GIST specimens for Carney-Stratakis syndrome (SDHx-deficient GIST has distinct biology); "
                "guides germline testing in sporadic pheo/PGL. "
                "DOTATATE PET/CT: SDHx-mutated pheo/PGL express somatostatin receptors (SSTR2/5) → "
                "DOTATATE PET superior to FDG-PET for staging and surveillance of SDHx-associated disease; "
                "use FDG-PET for SDHB (higher grade/malignant tumours express less SSTR, more glucose uptake)."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (gene 0 only) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["gene_breakdowns"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (first key) ===")
    defn = generate_definitions()
    first_key = next(iter(defn["gene_entries"]))
    print(json.dumps(defn["gene_entries"][first_key], indent=2)[:1500])
