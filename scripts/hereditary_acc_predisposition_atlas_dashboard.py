#!/usr/bin/env python3
"""Hereditary-Adrenocortical-Carcinoma-Predisposition-Atlas -- Complete 8-Gene Reference
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome;
         pediatric ACC 50-80% carry germline TP53 mutation;
         anaplastic ACC PATHOGNOMONIC in LFS;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually;
         seed SEED_BASE+0) .
CTNNB1  (Beta-catenin; 781aa; 3p22.1; AD GOF somatic dominant;
         WNT/beta-catenin pathway;
         somatic GOF exon 3 in-frame del 15-25% sporadic ACC;
         nuclear beta-catenin IHC staining PATHOGNOMONIC aggressive ACC;
         germline GOF extremely rare;
         seed SEED_BASE+1) .
CDKN2A  (p16-INK4A / ARF; 156aa; 9p21.3; AD LOF;
         FAMMM syndrome;
         p16 IHC loss PATHOGNOMONIC aggressive ACC;
         CDK4/6 inhibitor palbociclib investigational;
         pancreatic cancer 20x concurrent risk;
         9p21 homozygous deletion 25-30% ACC;
         seed SEED_BASE+2) .
NF1     (Neurofibromin; 2839aa; 17q11.2; AD LOF;
         Neurofibromatosis type 1;
         cafe-au-lait macules >=6 PATHOGNOMONIC;
         adrenocortical tumors 1-3x elevated;
         MPNST 8-13% lifetime; selumetinib FDA 2020;
         seed SEED_BASE+3) .
MEN1    (Menin; 610aa; 11q13.1; AD LOF;
         MEN1 syndrome;
         adrenocortical adenoma 20-40% non-functional;
         parathyroid 95% EARLIEST; pancreatic NET 40-80%;
         MEN1 triad: parathyroid + pituitary + pancreatic;
         seed SEED_BASE+4) .
PRKAR1A (Protein kinase cAMP-dependent regulatory I alpha; 381aa; 17q24.2; AD LOF;
         Carney complex (CNC);
         PPNAD PATHOGNOMONIC; bilateral micronodular hyperplasia PATHOGNOMONIC;
         Cushing syndrome cyclic; annual cortisol MANDATORY;
         cardiac myxoma LIFE-THREATENING;
         spotty perioral pigmentation PATHOGNOMONIC;
         seed SEED_BASE+5) .
ARMC5   (Armadillo repeat containing 5; 1059aa; 2p13.3; AD LOF;
         Primary bilateral macronodular adrenocortical hyperplasia (PBMAH);
         bilateral adrenal masses PATHOGNOMONIC;
         ACTH-independent Cushing 40-80%; meningioma co-risk 25%;
         annual adrenal MRI MANDATORY;
         seed SEED_BASE+6) .
MAX     (MYC-associated factor X; 236aa; 14q23.3; AD LOF;
         Hereditary pheochromocytoma/ACC overlap;
         bilateral pheochromocytoma 2-3x; SDHx pathway overlap;
         MAX IHC nuclear loss in pheo PATHOGNOMONIC;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3334-3341)
"""
import random

SEED_BASE = 3334

ATLAS_GENES = [
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-"
            "Pediatric-ACC-50-80pct-Germline-TP53-"
            "Anaplastic-ACC-PATHOGNOMONIC-LFS-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-Annual-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes tumour protein p53: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; homotetrameric transcription factor; genome guardian; "
            "  N-terminal transactivation domain (aa 1-42): MDM2 binding (E3 ubiquitin ligase); "
            "  Proline-rich region (aa 40-90): apoptosis regulation; "
            "  DNA-binding domain (aa 94-292): hotspot mutations R175H, G245S, R248W, R273H; "
            "  Tetramerisation domain (aa 323-356): functional tetramer assembly; "
            "  C-terminal regulatory domain (aa 356-393): post-translational modification; "
            "  TP53 activates CDKN1A (p21) → G1/S arrest; PUMA/NOXA → apoptosis; "
            "LI-FRAUMENI SYNDROME (LFS) AND ACC: "
            "  50-80% of pediatric ACC carry germline TP53 mutation — DOMINANT cause; "
            "  Pediatric ACC (age <15yr): TP53 germline PATHOGNOMONIC LFS context; "
            "  R337H founder mutation (Brazilian pediatric ACC cohort): 35x elevated adrenal risk; "
            "  Adult ACC in LFS: 3-7% of LFS-associated cancers; "
            "  Sarcoma 50-60% (dominant LFS cancer); breast cancer <45yr 30%; brain tumour 15%; "
            "  Adrenocortical carcinoma child: PATHOGNOMONIC of LFS — test TP53 every pediatric ACC; "
            "ACC HISTOPATHOLOGY TP53: "
            "  Anaplastic/poorly differentiated ACC predominant histology in LFS; "
            "  Weiss score ≥3 in all germline TP53-associated ACC; "
            "  Somatic TP53 LOF in 20-30% sporadic adult ACC (secondary event); "
            "  Mitotane: first-line adrenocortical carcinoma adjuvant; "
            "AVOID RADIATION ABSOLUTELY in germline TP53: "
            "  Ionising radiation → secondary sarcoma/carcinoma in RT field; "
            "  Surgery preferred over RT consolidation always; "
            "  EBRT to adrenal bed: CONTRAINDICATED in germline TP53; "
            "SURVEILLANCE LFS: "
            "  WBMRI: annually (Toronto protocol); brain MRI annually; "
            "  Abdominopelvic USS every 3-4 months <18yr (ACC + liver surveillance); "
            "  Annual adrenal CT/MRI — ACC early detection critical; "
            "  Breast MRI annually from age 20-25yr; annual CBC; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Li-Fraumeni Syndrome (LFS)",
        "acc_risk": "Pediatric ACC 50-80% carry germline TP53 — DOMINANT cause; adult ACC 3-7% LFS spectrum",
        "pathognomonic": "Pediatric ACC PATHOGNOMONIC LFS; anaplastic ACC child; R337H Brazilian founder hotspot",
        "key_avoid": "RADIATION — AVOID RADIATION ABSOLUTELY in germline TP53; surgery preferred over RT always",
        "surveillance": "WBMRI annually + brain MRI + abdominopelvic USS q3-4mo <18yr; annual adrenal CT/MRI",
        "targeted_rx": "Mitotane adjuvant ACC; EDP-M (etoposide-doxorubicin-cisplatin-mitotane) advanced ACC",
        "key_rule": "AVOID RADIATION ABSOLUTELY — TP53 germline + radiation → secondary sarcoma/carcinoma in RT field",
    },
    {
        "gene": "CTNNB1",
        "protein": (
            "CTNNB1 -- 3p22.1 Autosomal-Dominant-GOF-Somatic-Dominant -- 781aa -- "
            "BetaCatenin-85kDa-WNT-Armadillo-Repeat-"
            "Somatic-GOF-Exon3-InFrame-Del-15-25pct-Sporadic-ACC-"
            "Nuclear-BetaCatenin-IHC-PATHOGNOMONIC-Aggressive-ACC-"
            "Germline-GOF-Familial-ACC-Extremely-Rare-OMIM-116806"
        ),
        "locus": "3p22.1",
        "protein_size": (
            "781 aa / 85 kDa / 3p22.1 CTNNB1 encodes beta-catenin: "
            "STRUCTURE: "
            "  781 aa / 85 kDa; scaffolding protein and transcriptional co-activator; "
            "  N-terminal regulatory domain (aa 1-140): GSK3B/CK1 phosphorylation sites S33/S37/T41/S45; "
            "  Armadillo repeat domain (aa 141-664): 12 Arm repeats; protein-protein interactions; "
            "  C-terminal transactivation domain (aa 665-781): TCF/LEF co-activation; "
            "  Exon 3 encodes phosphodegron (S33/S37/T41/S45) — GOF in-frame del destroys phosphodegron; "
            "  CTNNB1 GOF → beta-catenin escapes APC/AXIN destruction complex → nuclear WNT target activation; "
            "WNT/BETA-CATENIN IN ACC: "
            "  Somatic CTNNB1 GOF: 15-25% sporadic ACC (exon 3 in-frame deletion MOST COMMON); "
            "  Nuclear beta-catenin IHC staining PATHOGNOMONIC of aggressive, dedifferentiated ACC; "
            "  CTNNB1 GOF correlates with high Weiss score and adverse prognosis; "
            "  Loss of membranous beta-catenin + nuclear accumulation = IHC signature; "
            "  CTNNB1 somatic variants activate SF-1 + WNT targets synergistically in ACC; "
            "GERMLINE CTNNB1 AND ACC: "
            "  Germline CTNNB1 GOF extremely rare (<50 families worldwide); "
            "  Familial ACC clusters: some linked to CTNNB1 germline GOF (exon 3); "
            "  Associated phenotype: medulloblastoma, pilomatricoma, polycystic kidneys; "
            "  APC pathway overlap: distinguishing germline FAP vs CTNNB1 GOF requires sequencing; "
            "AFP NORMALISATION: "
            "  AFP monitoring not applicable to ACC (ACC is AFP-negative); "
            "  Cortisol/DHEA-S biochemical monitoring primary; "
            "  Mitotane remains first-line adjuvant; "
            "  Investigational: tankyrase/porcupine inhibitors (WNT pathway); "
        ),
        "inheritance": "AD GOF (somatic dominant)",
        "syndrome": "Somatic WNT/beta-catenin ACC; germline GOF familial ACC (extremely rare)",
        "acc_risk": "Somatic GOF exon 3 in 15-25% sporadic ACC; germline GOF familial ACC clusters",
        "pathognomonic": "Nuclear beta-catenin IHC staining PATHOGNOMONIC aggressive ACC; exon 3 in-frame del",
        "key_avoid": "MISS NUCLEAR BETA-CATENIN IHC — always stain ACC for nuclear beta-catenin to guide prognosis",
        "surveillance": "Annual adrenal CT/MRI in germline GOF carriers; cortisol/DHEA-S biochemistry annually",
        "targeted_rx": "Mitotane adjuvant; EDP-M advanced ACC; WNT inhibitors investigational (tankyrase/porcupine)",
        "key_rule": "NUCLEAR BETA-CATENIN IHC — PATHOGNOMONIC aggressive ACC; exon 3 sequencing mandatory for somatic profiling",
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 Autosomal-Dominant-LOF -- 156aa -- "
            "p16-INK4A-17kDa-CDK4-6-Inhibitor-FAMMM-"
            "p16-IHC-Loss-PATHOGNOMONIC-Aggressive-ACC-"
            "CDK4-6i-Palbociclib-Investigational-"
            "9p21-Homozygous-Del-25-30pct-ACC-Pancreatic-20x-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / 17 kDa / 9p21.3 CDKN2A encodes p16-INK4A (and p14-ARF alternate reading frame): "
            "STRUCTURE: "
            "  156 aa / 17 kDa (p16-INK4A); 132 aa (p14-ARF shares exons 2-3 in alternate frame); "
            "  Four ankyrin repeats (ANK1-4): CDK4/CDK6 binding interface; "
            "  p16-INK4A binds CDK4/CDK6 → blocks cyclin D1 binding → CDK4/6 inhibited → RB1 hypophosphorylation → G1 arrest; "
            "  p14-ARF binds MDM2 → stabilises TP53 (indirect TP53 pathway); "
            "  CDKN2A LOF → CDK4/6 hyperactive → RB1 hyperphosphorylated → E2F released → S-phase entry; "
            "FAMMM SYNDROME AND ACC: "
            "  Familial Atypical Multiple Mole Melanoma (FAMMM): CDKN2A germline LOF; "
            "  Melanoma 30-50% lifetime; pancreatic cancer 20x concurrent risk; "
            "  Adrenocortical carcinoma: elevated risk in CDKN2A LOF carriers (reported series); "
            "  p16 IHC loss in ACC biopsy PATHOGNOMONIC of 9p21 deletion / CDKN2A LOF; "
            "9P21 DELETION IN SPORADIC ACC: "
            "  9p21 homozygous deletion: 25-30% of all sporadic ACC; "
            "  CDKN2A/B co-deleted (CDKN2A/p16 + CDKN2B/p15 locus); "
            "  Correlates with aggressive phenotype, large tumour, and adverse prognosis; "
            "  FISH or MLPA at 9p21 recommended for ACC genomic profiling; "
            "CDK4/6 INHIBITOR IN ACC: "
            "  Palbociclib (CDK4/6 inhibitor): investigational ACC (CDKN2A-deleted tumours); "
            "  Rationale: CDKN2A LOF → CDK4/6 hyperactive → palbociclib restores RB1 suppression; "
            "  Clinical trials ongoing; not yet FDA approved for ACC; "
            "PANCREATIC CANCER RISK CDKN2A: "
            "  20x elevated pancreatic ductal adenocarcinoma in CDKN2A germline LOF; "
            "  Annual MRI pancreas from age 40 in CDKN2A carriers; "
            "  Combined ACC + pancreatic risk surveillance mandatory; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Familial Atypical Multiple Mole Melanoma (FAMMM) / Hereditary Melanoma-Pancreatic Cancer",
        "acc_risk": "ACC elevated in CDKN2A LOF; 9p21 homozygous deletion in 25-30% sporadic ACC",
        "pathognomonic": "p16 IHC loss PATHOGNOMONIC aggressive ACC; 9p21 homozygous deletion; FAMMM family with ACC",
        "key_avoid": "MISS 9P21 CDKN2A DELETION — always test 9p21 FISH/MLPA in ACC for prognosis and CDK4/6i eligibility",
        "surveillance": "Annual adrenal CT/MRI; annual MRI pancreas from age 40; annual dermatology (melanoma)",
        "targeted_rx": "Palbociclib CDK4/6i investigational for CDKN2A-deleted ACC; mitotane adjuvant standard",
        "key_rule": "P16 IHC LOSS PATHOGNOMONIC — 9p21/CDKN2A deletion drives aggressive ACC; palbociclib investigational target",
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RAS-GAP-NF1-"
            "Adrenocortical-Tumors-1-3x-Elevated-"
            "Cafe-au-Lait-PATHOGNOMONIC-MPNST-8-13pct-"
            "Selumetinib-FDA2020-Paediatric-Plexiform-OMIM-613113"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 encodes neurofibromin: "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; GTPase activating protein (GAP) for RAS; "
            "  Central GRD (GTPase-activating related domain, aa 1175-1551): RAS-GTP hydrolysis; "
            "  NF1 LOF → reduced RAS-GAP activity → elevated RAS-GTP → constitutive RAS/MAPK; "
            "  SEC14 domain: lipid binding; CSRD and CTD domains: scaffold functions; "
            "  Largest known tumour suppressor gene (350 kb genomic span); "
            "  Highest spontaneous germline mutation rate of any known tumour suppressor; "
            "NF1 AND ADRENOCORTICAL TUMOURS: "
            "  Adrenocortical adenoma and carcinoma: 1-3x elevated vs general population; "
            "  Pheochromocytoma: 5% of NF1 patients (dominant adrenal manifestation); "
            "  Adrenocortical tumours in NF1 usually non-functional; incidentally detected on MRI; "
            "  Annual adrenal imaging from age 20 recommended in NF1 surveillance; "
            "  MPNST (malignant peripheral nerve sheath tumour): dominant cancer 8-13%; "
            "NF1 NIH DIAGNOSTIC CRITERIA (≥2 of 8): "
            "  ≥6 cafe-au-lait macules (>5mm prepubertal / >15mm postpubertal) PATHOGNOMONIC; "
            "  ≥2 neurofibromas OR ≥1 plexiform neurofibroma; "
            "  Axillary OR inguinal freckling PATHOGNOMONIC (Crowe sign); "
            "  Optic pathway glioma; ≥2 Lisch nodules; "
            "  Distinctive osseous lesion; first-degree relative with NF1; "
            "NF1 ONCOLOGICAL BURDEN: "
            "  MPNST: 8-13% lifetime; dominant cancer risk in NF1; "
            "  Optic pathway glioma: 15-20% children; selumetinib FDA 2020; "
            "  JMML: 200x elevated in NF1 children; "
            "  Breast cancer: ~5x elevated; GI stromal tumour (GIST): elevated; "
            "TREATMENT NF1: "
            "  Selumetinib (MEK inhibitor): FDA 2020 paediatric inoperable plexiform neurofibromas; "
            "  MPNST: complete resection (R0) first-line; poor chemo response; "
            "  Mitotane adjuvant for NF1-associated ACC; annual BP + urine catecholamines; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Neurofibromatosis Type 1 (NF1)",
        "acc_risk": "Adrenocortical tumours 1-3x elevated; pheochromocytoma 5% dominant adrenal manifestation",
        "pathognomonic": "≥6 cafe-au-lait macules PATHOGNOMONIC; axillary freckling PATHOGNOMONIC; Lisch nodules PATHOGNOMONIC",
        "key_avoid": "AVOID RADIATION — radiation risk in NF1 significant; MPNST acceleration documented with RT",
        "surveillance": "Annual NF specialist exam; annual adrenal CT/MRI from age 20; annual BP + urine catecholamines",
        "targeted_rx": "Selumetinib FDA2020 paediatric plexiform NF; mitotane adjuvant NF1-associated ACC",
        "key_rule": "CAFE-AU-LAIT ≥6 MACULES PATHOGNOMONIC — NF1 diagnosis mandates annual adrenal surveillance + pheo screen",
    },
    {
        "gene": "MEN1",
        "protein": (
            "MEN1 -- 11q13.1 Autosomal-Dominant-LOF -- 610aa -- "
            "Menin-68kDa-H3K4-Methyltransferase-Scaffold-MEN1-Triad-"
            "Adrenocortical-Adenoma-20-40pct-Non-Functional-"
            "Parathyroid-95pct-Earliest-PancNET-40-80pct-"
            "3.5-Gland-Parathyroid-Resection-MANDATORY-OMIM-131100"
        ),
        "locus": "11q13.1",
        "protein_size": (
            "610 aa / 68 kDa / 11q13.1 MEN1 encodes menin: "
            "STRUCTURE: "
            "  610 aa / 68 kDa; nuclear scaffold protein; no intrinsic enzymatic activity; "
            "  Interacts with MLL1/MLL2 histone H3K4 methyltransferase complex (KAT3A/KAT3B); "
            "  Regulates CDK inhibitors: CDKN1B (p27), CDKN2C (p18) — MEN1 LOF → CDK inhibitor loss; "
            "  JunD binding domain (aa 1-40): JunD transcriptional repressor interaction; "
            "  C-terminal nuclear export signal; "
            "  MEN1 LOF → reduced H3K4 methylation at tumour suppressor loci → proliferation; "
            "MULTIPLE ENDOCRINE NEOPLASIA TYPE 1 (MEN1): "
            "  Classic Triad: Parathyroid 95% (EARLIEST) + Pituitary 40% + Pancreatic/duodenal NET 70%; "
            "  Most common hereditary endocrine tumour syndrome (1 in 30,000); "
            "  Annual calcium/PTH from age 5-8 (first manifestation is hyperparathyroidism); "
            "  Gastrinoma / Zollinger-Ellison: 25-40%; ulcers; PPI lifelong; "
            "MEN1 ADRENAL CORTEX: "
            "  Adrenocortical adenoma: 20-40% of MEN1 patients (predominantly non-functional); "
            "  Bilateral non-functional adrenocortical lesions on imaging common in MEN1; "
            "  ACC in MEN1: rare but documented; annual adrenal CT/MRI mandatory in MEN1; "
            "  ACTH-independent Cushing rare in MEN1 adrenal tumours; "
            "  Annual cortisol / DHEA-S to detect functional shift in known MEN1 adrenal adenoma; "
            "  Adrenalectomy threshold: >4 cm, rapid growth, or biochemical evidence of function; "
            "PANCREATIC NET TREATMENT: "
            "  <2 cm: watch + annual MRI; ≥2 cm: surgery (Whipple or distal pancreatectomy); "
            "  Everolimus (mTOR inhibitor): FDA approved advanced pancreatic NET; "
            "  Sunitinib: FDA approved advanced pancreatic NET; "
            "  Lutetium-177 DOTATATE (PRRT): FDA 2018 advanced somatostatin-positive NETs; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Multiple Endocrine Neoplasia Type 1 (MEN1)",
        "acc_risk": "Adrenocortical adenoma 20-40% (predominantly non-functional); ACC rare but documented in MEN1",
        "pathognomonic": "Multiglandular parathyroid + pancreatic NET + pituitary adenoma = MEN1 TRIAD PATHOGNOMONIC",
        "key_avoid": "SINGLE GLAND PARATHYROID SURGERY — MEN1 requires 3.5-gland resection; single adenomectomy recurs",
        "surveillance": "Annual Ca2+/PTH from age 8; annual adrenal CT/MRI; annual cortisol/DHEA-S; 3-yearly pituitary MRI",
        "targeted_rx": "Everolimus / sunitinib / lutetium-177 DOTATATE FDA approved pancreatic NET; mitotane adjuvant ACC",
        "key_rule": "3.5-GLAND PARATHYROID RESECTION — MEN1 hyperparathyroidism is multiglandular; single-gland surgery fails",
    },
    {
        "gene": "PRKAR1A",
        "protein": (
            "PRKAR1A -- 17q24.2 Autosomal-Dominant-LOF -- 381aa -- "
            "PKA-R1alpha-43kDa-cAMP-Regulatory-Subunit-Carney-Complex-"
            "PPNAD-PATHOGNOMONIC-Bilateral-Micronodular-Hyperplasia-PATHOGNOMONIC-"
            "Cushing-Syndrome-Cyclic-Annual-Cortisol-MANDATORY-"
            "Cardiac-Myxoma-LIFE-THREATENING-Spotty-Perioral-Pigmentation-PATHOGNOMONIC-OMIM-188830"
        ),
        "locus": "17q24.2",
        "protein_size": (
            "381 aa / 43 kDa / 17q24.2 PRKAR1A encodes protein kinase cAMP-dependent regulatory I alpha: "
            "STRUCTURE: "
            "  381 aa / 43 kDa; regulatory subunit of Protein Kinase A (PKA); "
            "  Dimerisation/docking domain (aa 1-45): AKAP anchoring; "
            "  Inhibitory domain (aa 94-100): pseudosubstrate; "
            "  cAMP-binding domain A (aa 149-252): first cAMP binding pocket; "
            "  cAMP-binding domain B (aa 253-376): second cAMP binding pocket; "
            "  PRKAR1A LOF → disinhibited PKA catalytic subunit → constitutive cAMP/PKA signalling → cortisol overproduction; "
            "PRIMARY PIGMENTED NODULAR ADRENOCORTICAL DISEASE (PPNAD): "
            "  PPNAD: bilateral micronodular adrenocortical hyperplasia PATHOGNOMONIC for Carney Complex; "
            "  Bilateral small black/dark-brown pigmented cortical nodules on adrenal imaging; "
            "  ACTH-independent Cushing syndrome: cyclic and periodic cortisol excess; "
            "  Paradoxical cortisol RISE on Liddle dexamethasone suppression test PATHOGNOMONIC; "
            "  Cushing phenotype often cyclical — may be intermittently normal; "
            "  Annual midnight salivary cortisol + 24h UFC + Liddle test MANDATORY in all PRKAR1A carriers; "
            "  Bilateral adrenalectomy: definitive treatment for PPNAD Cushing; requires lifelong steroid replacement; "
            "CARDIAC MYXOMA — LIFE-THREATENING: "
            "  30-40% of Carney Complex patients; can be bilateral (both ventricles) and valvular; "
            "  Embolic stroke risk if undetected; sudden death documented; "
            "  ANNUAL ECHOCARDIOGRAM MANDATORY from diagnosis — cannot be omitted; "
            "  Can recur in any cardiac chamber after resection; "
            "CARNEY COMPLEX (CNC) OTHER FEATURES: "
            "  Spotty skin pigmentation: lentigines (perioral, periocular, conjunctival, genital mucosal) PATHOGNOMONIC; "
            "  GH-secreting pituitary adenoma: 10% (acromegaly); "
            "  LCCSCT (large-cell calcifying Sertoli cell tumour): 30-40% males PATHOGNOMONIC; "
            "  Thyroid follicular adenoma near universal (>70% on USS); "
            "  Psammomatous melanotic schwannoma; breast ductal adenoma; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Carney Complex (CNC)",
        "acc_risk": "PPNAD PATHOGNOMONIC; bilateral micronodular adrenocortical hyperplasia with ACTH-independent Cushing",
        "pathognomonic": "PPNAD bilateral micronodular PATHOGNOMONIC; paradoxical Liddle test PATHOGNOMONIC; spotty perioral lentigines",
        "key_avoid": "MISS CARDIAC MYXOMA — annual echo MANDATORY; myxoma causes fatal embolism if undetected",
        "surveillance": "ANNUAL ECHOCARDIOGRAM MANDATORY; annual midnight cortisol + Liddle test; annual adrenal imaging",
        "targeted_rx": "Bilateral adrenalectomy PPNAD Cushing; surgical resection cardiac myxoma; steroid replacement lifelong",
        "key_rule": "ANNUAL ECHOCARDIOGRAM MANDATORY — cardiac myxoma causes fatal stroke/sudden death if missed; Liddle test annually",
    },
    {
        "gene": "ARMC5",
        "protein": (
            "ARMC5 -- 2p13.3 Autosomal-Dominant-LOF -- 1059aa -- "
            "ARMC5-120kDa-Armadillo-Repeat-PBMAH-"
            "Bilateral-Adrenal-Masses-PATHOGNOMONIC-"
            "ACTH-Independent-Cushing-40-80pct-Meningioma-Co-Risk-25pct-"
            "Annual-Adrenal-MRI-MANDATORY-OMIM-615954"
        ),
        "locus": "2p13.3",
        "protein_size": (
            "1059 aa / 120 kDa / 2p13.3 ARMC5 encodes armadillo repeat containing protein 5: "
            "STRUCTURE: "
            "  1059 aa / 120 kDa; scaffold protein with armadillo repeat units; "
            "  N-terminal BTB/POZ domain: protein-protein interaction (aa 1-120); "
            "  Central armadillo repeat domain: multiple ARM repeats mediating protein interactions; "
            "  C-terminal domain: nuclear localisation signals; "
            "  ARMC5 LOF → impaired adrenocortical cell apoptosis → bilateral hyperplasia; "
            "  Biallelic inactivation required (germline LOF + somatic second hit) → PBMAH; "
            "PRIMARY BILATERAL MACRONODULAR ADRENOCORTICAL HYPERPLASIA (PBMAH): "
            "  Bilateral adrenal masses (typically large, >5 cm bilateral): PATHOGNOMONIC presentation; "
            "  ACTH-independent Cushing syndrome: 40-80% of ARMC5 carriers develop biochemical Cushing; "
            "  Often subclinical Cushing (ACTH suppression without full phenotype); "
            "  Bilateral macronodular morphology on CT/MRI: hallmark finding; "
            "  ARMC5 germline mutation frequency: ~25% of PBMAH families; most common PBMAH gene; "
            "  Meningioma co-risk: 25% of ARMC5 PBMAH patients; annual brain MRI recommended; "
            "ANNUAL ADRENAL MRI MANDATORY: "
            "  Adrenal growth monitoring: bilateral enlargement precedes biochemical Cushing; "
            "  Annual cortisol / DHEA-S / ACTH to detect functional shift; "
            "  Annual low-dose dexamethasone suppression test in all carriers; "
            "  Cascade testing of first-degree relatives: ARMC5 carrier rate up to 50% per family; "
            "TREATMENT PBMAH-ARMC5: "
            "  Unilateral adrenalectomy (dominant adrenal): partial cortisol normalisation; "
            "  Bilateral adrenalectomy: definitive Cushing cure; lifelong steroid replacement required; "
            "  Metyrapone / osilodrostat: cortisol synthesis inhibitors as bridge to surgery; "
            "  Mifepristone (glucocorticoid receptor antagonist): FDA approved Cushing syndrome; "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Primary Bilateral Macronodular Adrenocortical Hyperplasia (PBMAH)",
        "acc_risk": "PBMAH with bilateral macronodular lesions; ACTH-independent Cushing 40-80%; ACC rare secondary event",
        "pathognomonic": "Bilateral macronodular adrenal masses PATHOGNOMONIC ARMC5; ACTH-independent Cushing without adrenal mass asymmetry",
        "key_avoid": "MISS SUBCLINICAL CUSHING — annual cortisol/ACTH screen mandatory even without clinical phenotype",
        "surveillance": "Annual adrenal MRI MANDATORY; annual cortisol/DHEA-S/ACTH; annual brain MRI (meningioma 25%)",
        "targeted_rx": "Bilateral/unilateral adrenalectomy; osilodrostat/metyrapone bridge; mifepristone FDA Cushing",
        "key_rule": "ANNUAL ADRENAL MRI MANDATORY — bilateral macronodular growth precedes Cushing by years; cascade test family",
    },
    {
        "gene": "MAX",
        "protein": (
            "MAX -- 14q23.3 Autosomal-Dominant-LOF -- 236aa -- "
            "MAX-18kDa-MYC-Associated-Factor-X-bHLH-LZ-"
            "Hereditary-Pheo-ACC-Overlap-"
            "Bilateral-Pheo-2-3x-SDHx-Pathway-Overlap-"
            "MAX-IHC-Nuclear-Loss-Pheo-PATHOGNOMONIC-OMIM-154950"
        ),
        "locus": "14q23.3",
        "protein_size": (
            "236 aa / 18 kDa / 14q23.3 MAX encodes MYC-associated factor X: "
            "STRUCTURE: "
            "  236 aa / 18 kDa; obligate heterodimerisation partner for MYC family proteins; "
            "  Basic helix-loop-helix leucine zipper (bHLH-LZ) domain: DNA binding + dimerisation; "
            "  N-terminal basic region (aa 22-35): E-box (CANNTG) DNA binding; "
            "  HLH domain (aa 36-76): helix-loop-helix dimerisation; "
            "  Leucine zipper (aa 77-98): homodimerisation and heterodimerisation; "
            "  MAX forms obligate heterodimer with MYC (c-MYC, N-MYC, L-MYC) for transcriptional activation; "
            "  MAX LOF → altered MYC/MAX stoichiometry → unbalanced MYC oncogenic transcription; "
            "MAX AND HEREDITARY PHAEOCHROMOCYTOMA/PARAGANGLIOMA: "
            "  MAX germline LOF: rare hereditary phaeochromocytoma predisposition gene; "
            "  Bilateral pheochromocytoma: 2-3x elevated vs SDHx germline; bilateral pheo predominant; "
            "  Predominantly adrenal (pheochromocytoma) rather than extra-adrenal (paraganglioma); "
            "  MAX IHC: nuclear loss in MAX-mutant pheochromocytoma PATHOGNOMONIC; "
            "  MAX-mutant tumours: predominantly noradrenergic biochemical phenotype; "
            "MAX AND ACC OVERLAP: "
            "  SDHx pathway overlap: like SDHB/SDHD carriers, MAX patients can develop ACC; "
            "  ACC in MAX germline: reported in case series; adrenal cortex involvement documented; "
            "  Annual adrenal imaging mandatory to detect early cortical lesions in MAX carriers; "
            "  N-MYC amplification in high-grade pheo: analogous pathway to MAX LOF; "
            "  Biochemical screening: annual plasma/urine metanephrines + normetanephrines; "
            "SURVEILLANCE MAX: "
            "  Annual plasma metanephrines + normetanephrines (pheochromocytoma surveillance); "
            "  Annual adrenal CT/MRI (pheochromocytoma + ACC); "
            "  Annual 24h urine catecholamines; "
            "  Cascade testing of first-degree relatives (AD inheritance); "
            "TREATMENT MAX PHEO: "
            "  Alpha-blockade (phenoxybenzamine or doxazosin) mandatory pre-operatively; "
            "  Laparoscopic adrenalectomy; bilateral staged approach if both adrenals affected; "
            "  Sunitinib / cabozantinib: advanced/metastatic pheo (clinical trials); "
        ),
        "inheritance": "AD LOF",
        "syndrome": "Hereditary Pheochromocytoma / ACC Overlap (MAX)",
        "acc_risk": "Pheochromocytoma dominant; ACC overlap via SDHx-like pathway; bilateral pheo 2-3x elevated",
        "pathognomonic": "MAX IHC nuclear loss in pheochromocytoma PATHOGNOMONIC; bilateral adrenal pheo in young patient",
        "key_avoid": "MISS BILATERAL PHEO — alpha-blockade MANDATORY before any surgical intervention; hypertensive crisis fatal",
        "surveillance": "Annual plasma metanephrines + normetanephrines; annual adrenal CT/MRI; annual 24h urine catecholamines",
        "targeted_rx": "Alpha-blockade pre-op mandatory; laparoscopic adrenalectomy; sunitinib/cabozantinib advanced pheo",
        "key_rule": "ALPHA-BLOCKADE MANDATORY PRE-OP — MAX pheo bilateral; phenoxybenzamine/doxazosin before any surgery",
    },
]

# Tumour types per gene
TUMOUR_TYPES_BY_GENE: dict = {
    "TP53":    ["Adrenocortical Carcinoma (ACC) Pediatric", "ACC Adult LFS", "Anaplastic ACC (high-grade)", "ACC Stage III LFS", "ACC Stage IV with metastases", "ACC + Sarcoma (synchronous LFS)"],
    "CTNNB1":  ["ACC Somatic CTNNB1 GOF", "ACC Nuclear Beta-catenin+", "Poorly Differentiated ACC", "ACC Stage III CTNNB1", "Adrenocortical Adenoma Functional", "ACC Recurrent CTNNB1"],
    "CDKN2A":  ["ACC 9p21-deleted", "ACC p16-IHC-loss", "ACC CDKN2A LOF", "ACC Adrenal Incidentaloma Progressive", "ACC + Melanoma (synchronous)", "ACC Bilateral CDKN2A"],
    "NF1":     ["Adrenocortical Adenoma NF1", "ACC NF1 incidental", "Pheochromocytoma NF1", "Adrenocortical Carcinoma NF1", "Adrenal Myelolipoma NF1"],
    "MEN1":    ["Adrenocortical Adenoma Non-functional", "Adrenocortical Adenoma Bilateral MEN1", "ACC MEN1 (rare)", "Adrenocortical Hyperplasia MEN1", "Adrenal Incidentaloma MEN1"],
    "PRKAR1A": ["PPNAD Bilateral Micronodular", "PPNAD + Cushing Syndrome", "PPNAD Cyclic Cushing", "Adrenocortical Adenoma CNC", "ACC Carney Complex (rare)"],
    "ARMC5":   ["PBMAH Bilateral Macronodular", "PBMAH + ACTH-independent Cushing", "PBMAH Subclinical Cushing", "PBMAH Large Bilateral Masses", "PBMAH + Meningioma (concurrent)"],
    "MAX":     ["Pheochromocytoma Unilateral MAX", "Pheochromocytoma Bilateral MAX", "ACC MAX Overlap", "Paraganglioma MAX (rare)", "Pheo + ACC Concurrent"],
}

# Pathogenic variants per gene
VARIANTS_BY_GENE: dict = {
    "TP53":    ["p.R337H (Brazilian founder pediatric ACC)", "p.R175H (DNA-binding domain hotspot)", "p.R248W (hotspot)", "p.G245S (hotspot)", "p.R273H (hotspot)", "Splice donor IVS4+1", "p.C176F (ZnF)"],
    "CTNNB1":  ["p.S45del (exon 3 in-frame del PATHOGNOMONIC)", "p.T41A (exon 3 GOF)", "p.S37F (exon 3 GOF)", "p.S33C (exon 3)", "p.D32G (exon 3)", "p.G34E (exon 3 sporadic)", "Exon 3 multi-codon deletion"],
    "CDKN2A":  ["p.R58Ter (frameshift FAMMM)", "9p21 homozygous deletion (FISH)", "p.G101W (common missense LOF)", "p.R24P (ankyrin repeat)", "p.A148T (variant of uncertain significance)", "Exon 2 deletion MLPA", "p.L32P (ANK1)"],
    "NF1":     ["p.R1947Ter (common NF1)", "p.Q1966Ter", "Exon 22 skipping (splice)", "17q11.2 microdeletion (MLPA 5%)", "p.R1276Q (GRD missense)", "Large segmental deletion NF1"],
    "MEN1":    ["p.R460Ter (frameshift region)", "p.W341Ter (common)", "p.V184E (missense)", "p.L22R (JunD binding)", "11q13 deletion (MLPA)", "IVS2+1G>A splice"],
    "PRKAR1A": ["p.R74Ter (common CNC)", "p.L206R (cAMP-BD A)", "p.S9Ter (N-terminal)", "17q24.2 large deletion (MLPA 30%)", "c.708+1G>A splice", "p.E143Val"],
    "ARMC5":   ["p.R945Ter (PBMAH common)", "p.Q586Ter (frameshift)", "p.E266Ter (ARM domain)", "p.R267Ter (ARM domain)", "p.H913Y (missense PBMAH)", "2p13.3 deletion MLPA", "p.R876H (pathogenic PBMAH)"],
    "MAX":     ["p.R60Q (bHLH domain)", "p.P20H (N-terminal)", "p.R35Ter (basic region)", "p.L80P (leucine zipper)", "14q23.3 deletion", "p.R36W (basic region)", "p.H28R (bHLH)"],
}

# Treatment protocols per gene
TREATMENT_PROTOCOLS_BY_GENE: dict = {
    "TP53":    ["Surgery preferred — AVOID RADIATION ABSOLUTELY in germline TP53", "Mitotane adjuvant (standard ACC adjuvant)", "EDP-M (etoposide-doxorubicin-cisplatin-mitotane) advanced ACC", "WBMRI Toronto annually (LFS surveillance)", "Abdominopelvic USS q3-4mo <18yr", "MDM2 inhibitors investigational in TP53-null ACC"],
    "CTNNB1":  ["Adrenalectomy (laparoscopic) stage I-II ACC", "Mitotane adjuvant ACC", "EDP-M advanced/metastatic ACC", "Nuclear beta-catenin IHC profiling mandatory at diagnosis", "WNT/porcupine inhibitors investigational", "Annual adrenal CT/MRI germline GOF carriers"],
    "CDKN2A":  ["Adrenalectomy laparoscopic stage I-II", "Mitotane adjuvant standard", "EDP-M advanced ACC", "Palbociclib (CDK4/6i) investigational CDKN2A-deleted ACC", "Annual MRI pancreas from age 40 (pancreatic 20x risk)", "Annual dermatology melanoma screening"],
    "NF1":     ["Laparoscopic adrenalectomy NF1 adrenal tumour", "Mitotane adjuvant if ACC", "Selumetinib (MEK) FDA 2020 paediatric plexiform NF", "MPNST: surgery R0 first-line", "Annual BP + urine catecholamines (pheo screen)", "Annual adrenal CT/MRI from age 20"],
    "MEN1":    ["3.5-gland parathyroid resection (multiglandular MEN1)", "Adrenalectomy if ACC or growing >4 cm adrenal lesion", "Mitotane adjuvant if ACC", "Everolimus (mTOR) advanced pancreatic NET", "Sunitinib advanced pancreatic NET", "Lutetium-177 DOTATATE (PRRT) advanced NET"],
    "PRKAR1A": ["ANNUAL ECHOCARDIOGRAM — cardiac myxoma resection", "Bilateral adrenalectomy PPNAD Cushing", "Annual midnight salivary cortisol + Liddle test", "Steroid replacement post-adrenalectomy (lifelong)", "Annual testicular USS males (LCCSCT)", "Annual pituitary MRI (GH adenoma)"],
    "ARMC5":   ["Annual adrenal MRI MANDATORY (PBMAH surveillance)", "Bilateral adrenalectomy definitive Cushing PBMAH", "Unilateral adrenalectomy (dominant adrenal) partial control", "Osilodrostat / metyrapone bridge to surgery", "Mifepristone (glucocorticoid receptor antagonist) FDA approved", "Annual brain MRI (meningioma 25%)"],
    "MAX":     ["Alpha-blockade MANDATORY pre-op (phenoxybenzamine/doxazosin)", "Laparoscopic adrenalectomy — bilateral staged", "Annual plasma metanephrines + normetanephrines", "Annual adrenal CT/MRI (pheo + ACC)", "Sunitinib / cabozantinib advanced/metastatic pheo", "Cascade testing first-degree relatives"],
}

# Surveillance protocols per gene
SURVEILLANCE_BY_GENE: dict = {
    "TP53":    ["WBMRI Toronto annually", "Annual brain MRI", "Annual adrenal CT/MRI (ACC early detection)", "Abdominopelvic USS every 3-4 months <18yr", "Annual breast MRI from age 20-25yr (women)", "Annual CBC (leukaemia surveillance)"],
    "CTNNB1":  ["Annual adrenal CT/MRI (germline GOF carriers)", "Annual cortisol/DHEA-S biochemistry", "Annual abdominal imaging (hepatic lesions WNT)", "Nuclear beta-catenin IHC all ACC biopsies", "Annual pelvic USS (endometrial WNT)", "3-yearly whole exome if cascade-positive family"],
    "CDKN2A":  ["Annual adrenal CT/MRI", "Annual MRI pancreas from age 40 (pancreatic 20x)", "Annual dermatology melanoma exam", "Annual colonoscopy from age 40 (CRC CDKN2A)", "FISH/MLPA 9p21 at ACC diagnosis", "Annual ophthalmology (uveal melanoma CDKN2A)"],
    "NF1":     ["Annual clinical NF specialist exam", "Annual adrenal CT/MRI from age 20yr", "Annual BP + urine catecholamines (pheo/adrenal)", "MRI spine/CNS if new neurological symptoms", "Annual ophthalmology (optic pathway glioma)", "Selumetinib for symptomatic paediatric plexiform NF"],
    "MEN1":    ["Annual Ca2+/PTH from age 5-8yr (parathyroid)", "Annual fasting gastrin + glucagon + PP (pancreatic NET)", "Annual adrenal CT/MRI + cortisol/DHEA-S (adrenal)", "3-yearly pituitary MRI (pituitary adenoma)", "Annual prolactin + IGF-1 (pituitary)", "Annual thyroid USS"],
    "PRKAR1A": ["ANNUAL ECHOCARDIOGRAM MANDATORY (cardiac myxoma)", "Annual midnight salivary cortisol + 24h UFC + Liddle test (PPNAD)", "Annual adrenal imaging (PPNAD, bilateral)", "Annual testicular USS from puberty males (LCCSCT)", "Annual pituitary MRI + IGF-1 (GH adenoma)", "Annual dermatology (lentigines, blue nevi)"],
    "ARMC5":   ["Annual adrenal MRI MANDATORY (PBMAH bilateral masses)", "Annual low-dose dexamethasone suppression test", "Annual cortisol/DHEA-S/ACTH (functional shift)", "Annual brain MRI from diagnosis (meningioma 25%)", "Cascade testing all first-degree relatives", "Annual 24h urinary free cortisol"],
    "MAX":     ["Annual plasma metanephrines + normetanephrines (pheo)", "Annual adrenal CT/MRI (pheo + ACC)", "Annual 24h urine catecholamines", "Annual BP monitoring (pheo hypertensive episodes)", "Cascade testing first-degree relatives (AD)", "MIBG scan if biochemistry positive"],
}

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]

# Per-gene clinical parameters for realistic seed-driven data
_GENE_CLINICAL_PARAMS: dict = {
    "TP53":    {"mean_age": 4.0,  "cr": 0.55, "relapse": 0.40, "radiation": 0.00},
    "CTNNB1":  {"mean_age": 45.0, "cr": 0.40, "relapse": 0.55, "radiation": 0.25},
    "CDKN2A":  {"mean_age": 48.0, "cr": 0.45, "relapse": 0.50, "radiation": 0.30},
    "NF1":     {"mean_age": 38.0, "cr": 0.50, "relapse": 0.45, "radiation": 0.20},
    "MEN1":    {"mean_age": 42.0, "cr": 0.75, "relapse": 0.20, "radiation": 0.10},
    "PRKAR1A": {"mean_age": 28.0, "cr": 0.80, "relapse": 0.15, "radiation": 0.05},
    "ARMC5":   {"mean_age": 50.0, "cr": 0.70, "relapse": 0.25, "radiation": 0.08},
    "MAX":     {"mean_age": 35.0, "cr": 0.60, "relapse": 0.35, "radiation": 0.15},
}


def _make_patient(gene_idx: int, patient_idx: int) -> dict:
    seed = SEED_BASE + gene_idx + patient_idx * len(ATLAS_GENES)
    rng = random.Random(seed)
    gene = _GENE_LIST[gene_idx]
    gene_info = ATLAS_GENES[gene_idx]
    params = _GENE_CLINICAL_PARAMS[gene]
    tumour_types = TUMOUR_TYPES_BY_GENE[gene]
    variants = VARIANTS_BY_GENE[gene]
    treatments = TREATMENT_PROTOCOLS_BY_GENE[gene]

    mean_age = params["mean_age"]
    if gene == "TP53":
        age_at_dx = max(1, int(rng.gauss(mean_age, 2.0)))
    else:
        age_at_dx = max(12, int(rng.gauss(mean_age, 8.0)))

    tumour_type = rng.choice(tumour_types)
    variant = rng.choice(variants)
    treatment = rng.choice(treatments)
    cr = rng.random() < params["cr"]
    radiation = rng.random() < params["radiation"]
    relapse_p = params["relapse"] * 0.5 if cr else params["relapse"] * 1.3
    relapse = rng.random() < min(relapse_p, 0.95)

    return {
        "patient_id": f"HACC-{gene}-{patient_idx:03d}",
        "gene": gene,
        "syndrome": gene_info["syndrome"],
        "age_at_dx": age_at_dx,
        "tumour_type": tumour_type,
        "variant": variant,
        "treatment": treatment,
        "cr": cr,
        "radiation": radiation,
        "relapse": relapse,
    }


def _generate_cohort() -> list:
    patients = []
    for gi in range(len(ATLAS_GENES)):
        for pi in range(40):
            patients.append(_make_patient(gi, pi))
    return patients


def generate_overview() -> dict:
    cohort = _generate_cohort()
    n = len(cohort)
    cr_n = sum(1 for p in cohort if p["cr"])
    rad_n = sum(1 for p in cohort if p["radiation"])
    relapse_n = sum(1 for p in cohort if p["relapse"])
    mean_age = round(sum(p["age_at_dx"] for p in cohort) / n, 1)

    gene_summary = {}
    for g in _GENE_LIST:
        pts = [p for p in cohort if p["gene"] == g]
        gene_summary[g] = {
            "n": len(pts),
            "cr_pct": round(100 * sum(1 for p in pts if p["cr"]) / len(pts), 1),
            "radiation_pct": round(100 * sum(1 for p in pts if p["radiation"]) / len(pts), 1),
            "relapse_pct": round(100 * sum(1 for p in pts if p["relapse"]) / len(pts), 1),
            "mean_age": round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
        }

    return {
        "atlas": "Hereditary-Adrenocortical-Carcinoma-Predisposition-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Adrenocortical Carcinoma Predisposition Reference — "
            "TP53-CTNNB1-CDKN2A-NF1-MEN1-PRKAR1A-ARMC5-MAX"
        ),
        "seeds": f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
        "total_patients": n,
        "cr_pct": round(100 * cr_n / n, 1),
        "radiation_pct": round(100 * rad_n / n, 1),
        "relapse_pct": round(100 * relapse_n / n, 1),
        "mean_age_at_dx": mean_age,
        "genes": _GENE_LIST,
        "gene_summary": gene_summary,
        "gene_colors": {
            "TP53":    "#e74c3c",
            "CTNNB1":  "#e67e22",
            "CDKN2A":  "#f39c12",
            "NF1":     "#27ae60",
            "MEN1":    "#16a085",
            "PRKAR1A": "#2980b9",
            "ARMC5":   "#8e44ad",
            "MAX":     "#2c3e50",
        },
        "key_rules": [
            "AVOID RADIATION ABSOLUTELY in TP53 — secondary malignancy acceleration; surgery preferred over RT for ACC in LFS",
            "PEDIATRIC ACC = GERMLINE TP53 UNTIL DISPROVEN — 50-80% of pediatric ACC carry germline TP53; test every child",
            "NUCLEAR BETA-CATENIN IHC PATHOGNOMONIC aggressive ACC — CTNNB1 exon 3 GOF; always profile at diagnosis",
            "P16 IHC LOSS PATHOGNOMONIC aggressive ACC — 9p21/CDKN2A deletion; palbociclib investigational CDK4/6i target",
            "ANNUAL ECHOCARDIOGRAM MANDATORY in PRKAR1A — cardiac myxoma causes fatal embolism/sudden death if missed",
            "ANNUAL ADRENAL MRI MANDATORY in ARMC5 — bilateral macronodular growth precedes Cushing by years; cascade test family",
            "ALPHA-BLOCKADE MANDATORY PRE-OP in MAX — bilateral pheo; phenoxybenzamine/doxazosin before any adrenal surgery",
            "3.5-GLAND PARATHYROID RESECTION in MEN1 — multiglandular hyperplasia; single adenomectomy fails inevitably",
        ],
    }


def generate_breakdown() -> dict:
    cohort = _generate_cohort()
    breakdown = {}
    for gi, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        pts = [p for p in cohort if p["gene"] == gene]
        tumour_counts: dict = {}
        for p in pts:
            tumour_counts[p["tumour_type"]] = tumour_counts.get(p["tumour_type"], 0) + 1
        top_tumours = sorted(tumour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        variant_counts: dict = {}
        for p in pts:
            variant_counts[p["variant"]] = variant_counts.get(p["variant"], 0) + 1
        top_variants = sorted(variant_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        breakdown[gene] = {
            "gene": gene,
            "protein": gene_info["protein"],
            "locus": gene_info["locus"],
            "syndrome": gene_info["syndrome"],
            "inheritance": gene_info["inheritance"],
            "acc_risk": gene_info["acc_risk"],
            "pathognomonic": gene_info["pathognomonic"],
            "key_avoid": gene_info["key_avoid"],
            "key_rule": gene_info["key_rule"],
            "surveillance": gene_info["surveillance"],
            "targeted_rx": gene_info["targeted_rx"],
            "n_patients": len(pts),
            "cr_pct": round(100 * sum(1 for p in pts if p["cr"]) / len(pts), 1),
            "radiation_pct": round(100 * sum(1 for p in pts if p["radiation"]) / len(pts), 1),
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
        "atlas": "Hereditary-Adrenocortical-Carcinoma-Predisposition-Atlas",
        "definitions": {
            "lfs_tp53_acc": (
                "Li-Fraumeni Syndrome (LFS) / TP53 germline LOF and ACC: "
                "50-80% of pediatric ACC carry germline TP53 mutation — dominant genetic cause; "
                "R337H founder mutation (Brazil): 35x adrenal risk; "
                "Anaplastic ACC in LFS context PATHOGNOMONIC; "
                "AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually; "
                "EDP-M mitotane for advanced ACC; surgery preferred over RT."
            ),
            "ctnnb1_acc": (
                "CTNNB1 somatic GOF / WNT pathway and ACC: "
                "Somatic GOF exon 3 in-frame deletion in 15-25% sporadic ACC; "
                "Nuclear beta-catenin IHC staining PATHOGNOMONIC aggressive ACC; "
                "Germline GOF extremely rare — familial ACC clusters; "
                "WNT inhibitors (porcupine/tankyrase) investigational; "
                "Mitotane adjuvant standard; EDP-M advanced."
            ),
            "cdkn2a_fammm_acc": (
                "CDKN2A LOF / FAMMM Syndrome and ACC: "
                "p16 IHC loss PATHOGNOMONIC aggressive ACC; "
                "9p21 homozygous deletion in 25-30% sporadic ACC; "
                "Pancreatic cancer 20x concurrent risk — annual MRI pancreas from age 40; "
                "CDK4/6 inhibitor palbociclib investigational for CDKN2A-deleted ACC; "
                "Annual dermatology for melanoma surveillance."
            ),
            "nf1_adrenal": (
                "NF1 (Neurofibromatosis Type 1) and adrenal: "
                "Adrenocortical tumours 1-3x elevated; pheochromocytoma 5% dominant adrenal; "
                "≥6 cafe-au-lait macules PATHOGNOMONIC; axillary freckling PATHOGNOMONIC; "
                "MPNST 8-13% lifetime dominant cancer risk; "
                "Selumetinib FDA 2020 paediatric plexiform NF; annual adrenal CT/MRI from age 20."
            ),
            "men1_adrenal": (
                "MEN1: MEN1 germline LOF; adrenocortical adenoma 20-40% (non-functional, bilateral); "
                "Classic triad: parathyroid 95% (EARLIEST) + pituitary 40% + pancreatic NET 70%; "
                "ACC rare in MEN1 but documented; annual adrenal CT/MRI + cortisol/DHEA-S mandatory; "
                "3.5-gland parathyroid resection mandatory (multiglandular); "
                "Everolimus/sunitinib/lutetium-177 DOTATATE pancreatic NET."
            ),
            "carney_complex_ppnad": (
                "Carney Complex (CNC) / PRKAR1A LOF and PPNAD: "
                "PPNAD (Primary Pigmented Nodular Adrenocortical Disease): bilateral micronodular hyperplasia PATHOGNOMONIC; "
                "Paradoxical cortisol RISE on Liddle dexamethasone test PATHOGNOMONIC; "
                "Cyclic ACTH-independent Cushing; annual midnight cortisol + Liddle test MANDATORY; "
                "Cardiac myxoma 30-40% LIFE-THREATENING — annual echo MANDATORY; "
                "Spotty perioral/genital lentigines PATHOGNOMONIC; bilateral adrenalectomy definitive."
            ),
            "armc5_pbmah": (
                "ARMC5 LOF / Primary Bilateral Macronodular Adrenocortical Hyperplasia (PBMAH): "
                "Bilateral adrenal macronodular masses PATHOGNOMONIC; "
                "ACTH-independent Cushing 40-80%; subclinical Cushing common; "
                "Meningioma co-risk 25% — annual brain MRI; "
                "Annual adrenal MRI MANDATORY; cascade testing family; "
                "Bilateral/unilateral adrenalectomy; osilodrostat/mifepristone bridge."
            ),
            "max_pheo_acc": (
                "MAX LOF / Hereditary Pheochromocytoma-ACC Overlap: "
                "Bilateral pheochromocytoma 2-3x elevated vs SDHx germline; "
                "MAX IHC nuclear loss in pheochromocytoma PATHOGNOMONIC; "
                "SDHx pathway overlap — ACC documented in MAX germline carriers; "
                "Alpha-blockade MANDATORY pre-op; annual plasma metanephrines; "
                "Sunitinib/cabozantinib advanced/metastatic pheo."
            ),
            "cascade_testing": (
                "CASCADE TESTING — Hereditary ACC Predisposition: "
                "1. TP53: every pediatric ACC must be germline-tested; WBMRI; AVOID radiation; "
                "2. CTNNB1: nuclear beta-catenin IHC all ACC; germline test familial ACC clusters; "
                "3. CDKN2A: 9p21 FISH/MLPA all ACC; annual MRI pancreas + dermatology; "
                "4. NF1: annual adrenal CT/MRI from 20; pheo screen annually; selumetinib plexiform; "
                "5. MEN1: annual Ca2+/PTH + adrenal CT/MRI + cortisol; 3.5-gland parathyroid; "
                "6. PRKAR1A: ANNUAL ECHO MANDATORY; Liddle test; bilateral adrenalectomy PPNAD; "
                "7. ARMC5: ANNUAL ADRENAL MRI; cascade family; osilodrostat/adrenalectomy; "
                "8. MAX: annual metanephrines; alpha-blockade pre-op; bilateral adrenal staged."
            ),
        },
        "key_clinical_rules": [
            {
                "rule": "AVOID RADIATION ABSOLUTELY (TP53)",
                "gene": "TP53",
                "rationale": "TP53 LOF impairs G1/S checkpoint; ionising radiation causes secondary sarcoma/carcinoma in field",
                "consequence": "Secondary malignancy in RT field within 5-10yr; accelerated carcinogenesis in LFS; EBRT to adrenal bed contraindicated",
            },
            {
                "rule": "EVERY PEDIATRIC ACC → TEST GERMLINE TP53 (TP53)",
                "gene": "TP53",
                "rationale": "50-80% of pediatric ACC carry germline TP53 mutation; R337H founder mutation has 35x adrenal risk",
                "consequence": "Missed germline TP53 → no LFS surveillance; sibling/parent cascade testing omitted; radiation delivered to germline TP53 patient",
            },
            {
                "rule": "NUCLEAR BETA-CATENIN IHC MANDATORY ALL ACC (CTNNB1)",
                "gene": "CTNNB1",
                "rationale": "Nuclear beta-catenin IHC positive = aggressive ACC with WNT GOF; prognostic and predictive for WNT inhibitors",
                "consequence": "Missed IHC → incorrect risk stratification; WNT inhibitor trial eligibility missed",
            },
            {
                "rule": "ANNUAL ECHOCARDIOGRAM MANDATORY (PRKAR1A)",
                "gene": "PRKAR1A",
                "rationale": "Cardiac myxoma in Carney Complex causes fatal embolic stroke or sudden death if undetected",
                "consequence": "Undetected myxoma fragment embolises → fatal stroke; bilateral/valvular myxoma recurs after partial resection",
            },
            {
                "rule": "ANNUAL ADRENAL MRI MANDATORY (ARMC5)",
                "gene": "ARMC5",
                "rationale": "PBMAH bilateral adrenal growth precedes clinical Cushing by years; early detection enables less radical surgery",
                "consequence": "Delayed diagnosis → bilateral large masses; full bilateral adrenalectomy required vs early unilateral option",
            },
            {
                "rule": "ALPHA-BLOCKADE MANDATORY PRE-OP (MAX)",
                "gene": "MAX",
                "rationale": "MAX pheo bilateral; unblocked catecholamine release during anaesthesia induction → hypertensive crisis, cardiac arrest",
                "consequence": "Intra-operative hypertensive crisis → cardiac arrest; phenoxybenzamine or doxazosin 10-14 days pre-op mandatory",
            },
            {
                "rule": "3.5-GLAND PARATHYROID RESECTION (MEN1)",
                "gene": "MEN1",
                "rationale": "MEN1 hyperparathyroidism is multiglandular hyperplasia; single adenomectomy inevitably recurs",
                "consequence": "Single adenomectomy recurrence >80% at 10yr; multiglandular resection + cryopreservation is curative approach",
            },
            {
                "rule": "P16 IHC LOSS + 9P21 FISH/MLPA IN ALL ACC (CDKN2A)",
                "gene": "CDKN2A",
                "rationale": "9p21 homozygous deletion in 25-30% sporadic ACC; CDK4/6 inhibitor palbociclib trial eligibility depends on deletion status",
                "consequence": "Missed 9p21 deletion → palbociclib trial exclusion; incorrect prognosis; pancreatic cancer 20x risk surveillance omitted",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(f"Total patients: {ov['total_patients']}")
    print(f"Genes: {ov['genes']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"CR%: {ov['cr_pct']}")
    print(f"Mean age dx: {ov['mean_age_at_dx']}")
    print("\n=== BREAKDOWN KEYS ===")
    bd = generate_breakdown()
    for g in bd["genes"]:
        print(f"  {g}: n={bd['breakdown'][g]['n_patients']}, CR={bd['breakdown'][g]['cr_pct']}%, mean_age={bd['breakdown'][g]['mean_age']}")
    print("\n=== DEFINITIONS ===")
    df = generate_definitions()
    for k in list(df["definitions"].keys())[:3]:
        print(f"  {k}: {df['definitions'][k][:60]}...")
