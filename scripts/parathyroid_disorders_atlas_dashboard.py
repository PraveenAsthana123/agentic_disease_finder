#!/usr/bin/env python3
"""Parathyroid-Disorders-Atlas — Complete 8-Gene Hereditary Parathyroid & Calcium Metabolism Disorders Atlas
CASR    (Calcium-Sensing Receptor; 1078 aa; 3q21.1; OMIM gene 601199;
         FHH1 (AD LOF) — Familial Hypocalciuric Hypercalcemia type 1; FECa<1%; mild/benign;
         ADH1 (AD GOF) — Autosomal Dominant Hypocalcemia type 1; low PTH + low Ca;
         NSHPT (AR biallelic LOF) — Neonatal Severe Hyperparathyroidism; emergency;
         Calcimimetics (cinacalcet) for symptomatic FHH1; THIAZIDES CONTRAINDICATED) ·
GNA11   (G-protein alpha-11 subunit; 359 aa; 19p13.3; OMIM gene 139313;
         FHH2 (AD LOF) — Familial Hypocalciuric Hypercalcemia type 2; CASR-negative FHH;
         ADH2 (AD GOF) — Autosomal Dominant Hypocalcemia type 2; low Ca, low PTH;
         Gα11 couples CASR → PLC → IP3; gain/loss phenotypes mirror CASR GOF/LOF) ·
AP2S1   (Adaptor Protein 2 Sigma-1 subunit; 142 aa; 19q13.3; OMIM gene 602242;
         FHH3 (AD LOF) — Familial Hypocalciuric Hypercalcemia type 3; Arg15 hotspot;
         Often MORE SYMPTOMATIC than FHH1/2; osmoregulation abnormalities;
         AP2 complex mediates CASR endocytosis — LOF → CASR stays at membrane → hypersensitive) ·
MEN1    (Menin tumour suppressor; 610 aa; 11q13.1; OMIM gene 613733;
         Multiple Endocrine Neoplasia type 1 (MEN1) — AD; 4-gland parathyroid hyperplasia 90%;
         Pancreatic NETs (ZES/gastrinoma, insulinoma); Pituitary adenomas (prolactinoma most common);
         Annual biochemical screen from age 8; HYPERPARATHYROIDISM FIRST manifestation in 90%) ·
CDC73   (Cell Division Cycle 73 / Parafibromin; 531 aa; 1q25.3; OMIM gene 607393;
         Hyperparathyroidism-Jaw Tumour syndrome (HPT-JT) — AD; PARATHYROID CARCINOMA 15-20%;
         Jaw ossifying fibroma; Renal hamartomas/cysts/Wilms tumour;
         Parafibromin IHC: ABSENT in carcinoma — diagnostic; En-bloc resection for carcinoma) ·
CDKN1B  (Cyclin-Dependent Kinase Inhibitor 1B / p27Kip1; 198 aa; 12p13.1; OMIM gene 600778;
         MEN4 — AD; SAME phenotype as MEN1 but MEN1-negative; CDKN1B is the key DDx gene;
         Parathyroid hyperplasia; Pituitary adenomas; GH-producing tumours;
         Order CDKN1B after MEN1 WES is negative in MEN1 phenotype) ·
GCM2    (Glial Cells Missing homolog 2 / GCMB; 502 aa; 6p24.2; OMIM gene 603716;
         Familial Isolated Hypoparathyroidism (FIH) — AR LOF biallelic; low PTH, low Ca, high PO4;
         Familial Isolated Hyperparathyroidism (FIHP) — AD GOF variants also described;
         GCM2 is the master regulator of parathyroid gland development;
         Hypoparathyroidism: Mg FIRST before Ca supplementation — hypoMg blocks PTH secretion) ·
PTH     (Parathyroid Hormone; 115 aa; 11p15.3; OMIM gene 168450;
         Isolated Familial Hypoparathyroidism — AD/AR; PTH gene variants → defective preproPTH;
         Low Ca + High PO4 + UNDETECTABLE PTH — differs from pseudohypoparathyroidism;
         AVOID OVER-TREATMENT: goal Ca low-normal (2.0-2.1 mmol/L) — nephrocalcinosis risk;
         Recombinant PTH (teriparatide) for severe cases; thiazides reduce urinary Ca loss)
320-patient aggregate cohort (8 x 40, seeds 1342-1349)
"""

import random

SEED_BASE = 1342

PARATHYROID_GENES = [
    # ── CASR — FHH1 / ADH1 / NSHPT ──
    {
        "gene": "CASR",
        "protein": "Calcium-Sensing Receptor",
        "alias": (
            "CASR; OMIM gene 601199; FHH1 #145980 (AD LOF) / ADH1 #146200 (AD GOF) / NSHPT #239200 (AR biallelic LOF); "
            "3q21.1; 1078 aa; ~121 kDa; Class C GPCR; homodimer; "
            "expressed on parathyroid chief cells, kidney tubular cells, thyroid C-cells; "
            "senses extracellular Ca²⁺ → inhibits PTH secretion (inverse set-point); "
            "also senses Mg²⁺, amino acids (phenylalanine), polyamines; "
            "LOF → set-point shifted rightward → hypercalcaemia with inappropriately low/normal PTH; "
            "GOF → set-point shifted leftward → hypocalcaemia with inappropriately low PTH"
        ),
        "aa": "1078 aa",
        "kDa": "~121 kDa",
        "locus": "3q21.1",
        "omim_gene": 601199,
        "omim_disease": 145980,
        "inheritance": (
            "AD LOF → FHH1 (autosomal dominant, high penetrance, usually benign); "
            "AD GOF → ADH1 (autosomal dominant, variable penetrance, hypocalcaemia with inappropriately low PTH); "
            "AR biallelic LOF → NSHPT (neonatal severe hyperparathyroidism — medical emergency); "
            "De novo GOF variants cause ADH1 in 30-50%; "
            "Consanguinity required for NSHPT; heterozygous LOF carriers in NSHPT pedigrees have FHH1"
        ),
        "gene_class": (
            "CASR encodes the calcium-sensing receptor, a Class C GPCR that functions as a homodimer "
            "sensing extracellular Ca²⁺ concentration via its large extracellular venus-flytrap domain. "
            "FUNCTION: Elevated Ca²⁺ → CASR activation → Gαq/11 → PLC → IP3 → intracellular Ca²⁺ release → "
            "PTH secretion SUPPRESSED; also → Gαi → cAMP reduction → PTH gene transcription reduced. "
            "In kidney: CASR activation → reduces Ca²⁺ reabsorption in thick ascending limb → calciuria. "
            "PATHOPHYSIOLOGY LOF (FHH1): Heterozygous LOF → receptor less sensitive → "
            "higher Ca²⁺ required to suppress PTH → set-point shifted right → "
            "mild-moderate hypercalcaemia + INAPPROPRIATELY LOW/NORMAL PTH + LOW URINARY Ca EXCRETION. "
            "KEY DIAGNOSTIC: Fractional excretion of calcium (FECa) < 1% — "
            "distinguishes FHH (benign, kidney retains Ca) from primary hyperparathyroidism (FECa >1-2%). "
            "CALCIUM:CREATININE CLEARANCE RATIO (CCCR) < 0.01 = FHH diagnostic threshold. "
            "THIAZIDE DIURETICS: ABSOLUTELY CONTRAINDICATED in FHH — further reduce urinary Ca → "
            "severe hypercalcaemia crisis. "
            "PATHOPHYSIOLOGY GOF (ADH1): Heterozygous GOF → receptor hypersensitive → "
            "normal Ca²⁺ suppresses PTH → low Ca²⁺ + low/low-normal PTH + excessive calciuria → "
            "risk of nephrocalcinosis if calcitriol/Ca²⁺ supplementation given without CASR inhibitor. "
            "CALCIMIMETICS (cinacalcet): positive allosteric modulator; lowers Ca²⁺ set-point; "
            "used for SYMPTOMATIC FHH1 or MEN1/parathyroid carcinoma; "
            "CASR ANTAGONISTS (calcilytics): in development for ADH1. "
            "NSHPT (AR biallelic): Severe PTH excess → marked hypercalcaemia → life-threatening; "
            "total parathyroidectomy required as emergency within weeks of birth."
        ),
        "phenotype": (
            "FHH1 (AD LOF): Mild-moderate hypercalcaemia (usually 2.6-3.0 mmol/L); "
            "PTH: normal or mildly elevated (paradoxically non-suppressed); "
            "Urine Ca²⁺: LOW (FECa <1%, CCCR <0.01) — KEY DDx from primary hyperparathyroidism; "
            "Usually ASYMPTOMATIC: incidental hypercalcaemia; "
            "Rarely: fatigue, polyuria, mild pancreatitis; "
            "NO nephrolithiasis, NO osteitis fibrosa; "
            "FAMILY HISTORY: autosomal dominant pattern; hypercalcaemia in relatives; "
            "ADH1 (AD GOF): Mild-moderate hypocalcaemia; tetany/seizures in neonates; "
            "Basal ganglia calcifications; nephrocalcinosis with calcitriol treatment; "
            "NSHPT (AR biallelic): Neonatal hypercalcaemia crisis (Ca²⁺ >4 mmol/L); "
            "respiratory distress, failure to thrive, skeletal demineralization; emergency parathyroidectomy"
        ),
        "key_hallmarks": [
            "FHH1: CCCR <0.01 (urine Ca/Cr clearance ratio) — single most discriminating test vs primary hyperparathyroidism",
            "THIAZIDE DIURETICS CONTRAINDICATED in FHH — reduce urinary Ca → severe hypercalcaemia crisis",
            "FHH1 is BENIGN: surgery NOT indicated; parathyroidectomy fails (all 4 glands hypersensitive) and causes hypoparathyroidism",
            "ADH1 goal: Ca low-NORMAL (2.0-2.1 mmol/L) — higher Ca → nephrocalcinosis (calciuria continues)",
            "NSHPT: emergency total parathyroidectomy; cinacalcet as bridge in neonates pre-surgery",
        ],
        "treatment_alerts": [
            "FHH1: NO parathyroidectomy (surgery fails — all 4 glands reset at wrong Ca²⁺ threshold)",
            "FHH1: NO thiazides — fatal hypercalcaemia; NO lithium — also raises Ca set-point (additive)",
            "ADH1: Ca + calcitriol supplementation WITHOUT CASR monitoring → nephrocalcinosis — target Ca 2.0-2.1 mmol/L only",
            "NSHPT: IV bisphosphonates as emergency bridge; cinacalcet off-label in neonates; total parathyroidectomy definitive",
            "Symptomatic FHH1/MEN1 hypercalcaemia: cinacalcet (calcimimetic) — positive allosteric modulator lowers Ca set-point",
            "Genetic counselling: CASR variant in family → test ALL first-degree relatives before any surgery planned",
        ],
        "ddx": [
            "Primary hyperparathyroidism (PHPT): FECa >1-2%, CCCR >0.02, PTH elevated, single adenoma on sestamibi; responds to parathyroidectomy",
            "GNA11 FHH2: identical FHH phenotype, CASR-negative; screen GNA11 second in FHH panel",
            "AP2S1 FHH3: CASR/GNA11-negative FHH; AP2S1 Arg15 hotspot; may be more symptomatic",
            "Familial Isolated Hyperparathyroidism (FIHP): MEN1/CDC73/GCM2 GOF panel; no extra-parathyroid features",
            "Vitamin D deficiency: low Ca, high PTH, low 25-OH-D; responds to vitamin D; not genetic",
            "Lithium-induced hypercalcaemia: mimics FHH (lithium raises Ca set-point); drug history essential",
        ],
        "seed": SEED_BASE + 0,
        "n_patients": 40,
        "age_range": (1, 70),
        "female_pct": 55,
    },
    # ── GNA11 — FHH2 / ADH2 ──
    {
        "gene": "GNA11",
        "protein": "Guanine nucleotide-binding protein G(11) alpha subunit",
        "alias": (
            "GNA11; OMIM gene 139313; FHH2 #145981 (AD LOF) / ADH2 #615361 (AD GOF); "
            "19p13.3; 359 aa; ~42 kDa; Gαq/11 family; "
            "couples CASR to phospholipase C-beta (PLCβ) → IP3/DAG → intracellular Ca²⁺ release; "
            "downstream effector of CASR (CASR → Gα11 → PLCβ → IP3 → [Ca²⁺]i → PTH inhibition); "
            "GNA11 LOF phenocopies CASR LOF (FHH); GNA11 GOF phenocopies CASR GOF (ADH)"
        ),
        "aa": "359 aa",
        "kDa": "~42 kDa",
        "locus": "19p13.3",
        "omim_gene": 139313,
        "omim_disease": 145981,
        "inheritance": (
            "Autosomal dominant; LOF → FHH2; GOF → ADH2; "
            "de novo GOF variants common in ADH2; "
            "50% risk to offspring; penetrance high but variable expressivity; "
            "FHH2 estimated to account for ~5-10% of CASR-negative FHH families"
        ),
        "gene_class": (
            "GNA11 encodes the Gα11 (Gαq/11 family) subunit, the primary downstream signalling partner of CASR. "
            "SIGNALLING PATHWAY: Ca²⁺ → CASR → conformational change → Gα11 activation → "
            "PLCβ3 stimulation → PIP2 hydrolysis → IP3 (Ca²⁺ release from ER) + DAG (PKC activation); "
            "collectively → intracellular Ca²⁺ rise → inhibition of PTH secretion + PTH gene transcription. "
            "ALSO activates MAPK/ERK pathway through CASR-Gα11 → modulates parathyroid cell proliferation. "
            "PATHOPHYSIOLOGY LOF (FHH2): GNA11 LOF → CASR signalling impaired downstream of CASR → "
            "same functional outcome as CASR LOF → parathyroid set-point shifted right → "
            "hypercalcaemia + low urinary Ca (FECa <1%) + PTH inappropriately normal/elevated. "
            "IDENTICAL PHENOTYPE to FHH1 (CASR) — cannot distinguish clinically; genetic panel required. "
            "PATHOPHYSIOLOGY GOF (ADH2): GNA11 GOF → constitutive/hypersensitive Gα11 signalling → "
            "excessive PTH suppression at normal Ca²⁺ → hypocalcaemia + low PTH → "
            "same manifestations as ADH1 (tetany, seizures, basal ganglia calcifications, nephrocalcinosis). "
            "SOMATIC GNA11 GOF: Also found in uveal melanoma (unrelated to calcium homeostasis — "
            "drives tumour cell proliferation via MAPK; important not to confuse with germline)."
        ),
        "phenotype": (
            "FHH2 (AD LOF): Identical to FHH1 — mild-moderate hypercalcaemia; "
            "PTH normal/mildly elevated; FECa <1%; CCCR <0.01; asymptomatic or minimal symptoms; "
            "Distinguishable from FHH1 only by genetic testing (CASR normal in FHH2); "
            "ADH2 (AD GOF): Identical to ADH1 — mild-moderate hypocalcaemia; "
            "Low PTH; Tetany, perioral paraesthesiae, carpopedal spasm; "
            "Seizures in neonates/infants; Basal ganglia calcifications on CT; "
            "Nephrocalcinosis risk if Ca²⁺ supplemented without close monitoring"
        ),
        "key_hallmarks": [
            "FHH2 is CASR-negative FHH — sequence GNA11 (and AP2S1) when CASR panel is normal in FHH phenotype",
            "FHH2 phenotype IDENTICAL to FHH1: same CCCR <0.01, same low FECa, same benign natural history",
            "ADH2: low Ca + LOW PTH distinguishes from vitamin D deficiency (high PTH) and pseudohypoparathyroidism (normal/high PTH)",
            "GNA11 somatic GOF in uveal melanoma — unrelated to calcium disorders; distinguish germline vs somatic",
            "FHH2 family members: same benign prognosis as FHH1; NO surgery; genetic cascade testing",
        ],
        "treatment_alerts": [
            "FHH2: NO surgery — same principle as FHH1; parathyroidectomy fails and causes permanent hypoparathyroidism",
            "ADH2: Ca + calcitriol supplementation ONLY to low-normal Ca target (2.0-2.1 mmol/L) — nephrocalcinosis risk",
            "ADH2: renal USS annually — calciuria-driven nephrocalcinosis even at therapeutic Ca levels",
            "Hydrochlorothiazide: may reduce calciuria in ADH (thiazide reduces urinary Ca) — useful adjunct to lower nephrocalcinosis risk",
            "Recombinant PTH 1-34 (teriparatide): considered for severe ADH2 with poor Ca control",
            "Cascading: CASR-negative FHH → GNA11 → AP2S1 in sequence; clinical management same regardless of which gene",
        ],
        "ddx": [
            "FHH1 (CASR LOF): Identical phenotype; genetic panel distinguishes; FHH1 more common (~65% of FHH)",
            "AP2S1 FHH3: third in FHH cascade; Arg15 hotspot; may be more symptomatic; osmolality abnormalities",
            "Primary hyperparathyroidism: CCCR >0.02, FECa >1-2%, single adenoma; responds to parathyroidectomy",
            "ADH1 (CASR GOF): Identical ADH phenotype; CASR sequencing first in ADH",
            "Autoimmune hypoparathyroidism (anti-CASR antibodies): acquired; distinguishable by antibody testing; not genetic",
        ],
        "seed": SEED_BASE + 1,
        "n_patients": 40,
        "age_range": (1, 65),
        "female_pct": 53,
    },
    # ── AP2S1 — FHH3 ──
    {
        "gene": "AP2S1",
        "protein": "Adaptor Protein complex 2 Sigma-1 subunit",
        "alias": (
            "AP2S1; OMIM gene 602242; FHH3 #600740 (AD LOF); "
            "19q13.3; 142 aa; ~17 kDa; sigma-1 subunit of AP-2 clathrin adaptor complex; "
            "AP-2 mediates clathrin-coated vesicle (CCV) endocytosis of plasma membrane GPCRs; "
            "AP2S1 variants: Arg15His/Cys/Leu — hotspot; prevent AP-2 binding to CASR di-leucine motif; "
            "LOF → CASR trapped at membrane → prolonged signalling → paradoxically raises Ca set-point; "
            "FHH3 often MORE SYMPTOMATIC than FHH1/2 (neuropsychiatric + osmolality effects)"
        ),
        "aa": "142 aa",
        "kDa": "~17 kDa",
        "locus": "19q13.3",
        "omim_gene": 602242,
        "omim_disease": 600740,
        "inheritance": (
            "Autosomal dominant; LOF variants; Arg15 codon is a mutational hotspot (>95% of FHH3 cases); "
            "50% risk to offspring; penetrance high; "
            "de novo variants occur; may present without obvious family history; "
            "FHH3 estimated at ~5% of CASR/GNA11-negative FHH"
        ),
        "gene_class": (
            "AP2S1 encodes the sigma-1 (σ2) subunit of the AP-2 clathrin adaptor complex. "
            "FUNCTION: AP-2 complex (α, β2, μ2, σ2 subunits) recognizes dileucine (D/ExxxLL) and "
            "tyrosine-based (Yxxφ) motifs on cytoplasmic tails of plasma membrane receptors → "
            "recruits clathrin → clathrin-coated vesicle (CCV) formation → receptor internalization → "
            "endolysosomal degradation or recycling. "
            "CASR INTERNALIZATION: CASR cytoplasmic tail contains a dileucine motif (780-LL) "
            "recognized by AP2S1 (σ2 subunit); CASR activation → AP-2 binding → CASR endocytosis → "
            "signal termination. "
            "PATHOPHYSIOLOGY (FHH3 LOF): AP2S1 Arg15 variants (Arg15His, Arg15Cys, Arg15Leu) → "
            "σ2 subunit cannot bind CASR dileucine motif → CASR NOT internalized → "
            "CASR accumulates at plasma membrane → prolonged/excessive signal at any Ca²⁺ level → "
            "paradoxically, the net effect on the parathyroid is REDUCED sensitivity to Ca²⁺ "
            "(counter-intuitive: endocytosis is required for receptor desensitisation → "
            "without endocytosis, CASR-mediated inhibitory signalling may be disrupted by chronic activation → "
            "parathyroid Ca set-point shifted right → FHH phenotype). "
            "ADDITIONAL FEATURES: AP2S1 LOF affects endocytosis of other receptors beyond CASR → "
            "neuropsychiatric features (cognitive changes, depression) and "
            "impaired osmoregulation (altered AVP receptor endocytosis) — "
            "making FHH3 potentially more symptomatic than FHH1/2. "
            "HOTSPOT: Arg15 in AP2S1 is a recurrent pathogenic hotspot; "
            "virtually all FHH3 cases carry Arg15His, Arg15Cys, or Arg15Leu."
        ),
        "phenotype": (
            "FHH3: Hypercalcaemia similar to FHH1/2 (Ca²⁺ 2.6-3.2 mmol/L); "
            "PTH: normal or mildly elevated; FECa <1%; CCCR <0.01; "
            "More likely to be SYMPTOMATIC than FHH1/2: fatigue, cognitive difficulties, depression, polyuria, polydipsia; "
            "Osmoregulatory abnormalities: relative hypotonic hyponatraemia reported (impaired AVP receptor internalization); "
            "Nephrolithiasis: less common than PHPT but more than FHH1/2; "
            "Pancreatitis: rare reports; "
            "Neuropsychiatric features may dominate clinical presentation"
        ),
        "key_hallmarks": [
            "FHH3 is CASR+GNA11-negative FHH — AP2S1 Arg15 hotspot (>95% of FHH3); direct sequencing of Arg15 codon",
            "AP2S1 FHH3 MAY BE SYMPTOMATIC — unlike benign FHH1/2; neuropsychiatric and osmolality features",
            "CCCR <0.01 still holds for FHH3 — remains the diagnostic anchor",
            "Osmoregulatory dysfunction: impaired AVP-R2 endocytosis → abnormal water handling → check serum Na/osmolality",
            "FHH3 management: same as FHH1/2 (no surgery); if symptomatic hypercalcaemia, cinacalcet may be considered",
        ],
        "treatment_alerts": [
            "FHH3: NO parathyroidectomy — surgery fails as in FHH1/2; all 4 glands affected",
            "Cinacalcet: may reduce Ca²⁺ and improve neuropsychiatric symptoms in symptomatic FHH3",
            "Monitor serum Na and osmolality: osmoregulatory impairment may require fluid management",
            "Avoid thiazide diuretics: same rationale as all FHH types — worsen hypercalcaemia",
            "Cascade test all first-degree relatives: identify carriers before inadvertent parathyroidectomy",
        ],
        "ddx": [
            "FHH1 (CASR LOF): Most common FHH; CASR panel negative in FHH3; same CCCR criterion",
            "FHH2 (GNA11 LOF): Second in panel; CASR + GNA11 negative in FHH3",
            "PHPT: CCCR >0.02; FECa >1%; single adenoma; parathyroidectomy curative",
            "Lithium-induced hypercalcaemia: drug history; CCCR may be <0.01 (lithium acts like LOF CASR)",
            "Hypercalcaemia of malignancy: PTHrP elevated; PTH suppressed; no family history",
        ],
        "seed": SEED_BASE + 2,
        "n_patients": 40,
        "age_range": (10, 75),
        "female_pct": 57,
    },
    # ── MEN1 — Multiple Endocrine Neoplasia 1 ──
    {
        "gene": "MEN1",
        "protein": "Menin tumour suppressor protein",
        "alias": (
            "MEN1; OMIM gene 613733; Multiple Endocrine Neoplasia type 1 (MEN1) #131100; autosomal dominant; 11q13.1; "
            "610 aa; ~67 kDa; nuclear scaffold protein; no enzymatic activity; "
            "interacts with >40 proteins: MLL/KMT2A histone H3K4 methyltransferase complex (epigenetic); "
            "JunD (AP-1 TF repressor); RPA2 (DNA repair); FANCD2 (Fanconi anaemia pathway); "
            "LOF → loss of tumour suppression in parathyroid, pancreas, pituitary → MEN1 triad; "
            "HYPERPARATHYROIDISM is FIRST and MOST COMMON manifestation (90% by age 50)"
        ),
        "aa": "610 aa",
        "kDa": "~67 kDa",
        "locus": "11q13.1",
        "omim_gene": 613733,
        "omim_disease": 131100,
        "inheritance": (
            "Autosomal dominant; germline LOF + somatic second hit (Knudson two-hit); "
            "50% risk to offspring; penetrance >95% by age 50; "
            "de novo variants in ~5-10% of cases; "
            "Mutation types: frameshift, nonsense, splice-site, missense (throughout gene — no hotspot); "
            "Prophylactic parathyroidectomy timing debated but most centers screen from age 8 annually"
        ),
        "gene_class": (
            "MEN1 encodes menin, a scaffold protein that acts as a tumour suppressor in endocrine tissues. "
            "MOLECULAR FUNCTION: Menin forms a complex with MLL1/MLL2 (KMT2A/KMT2B) histone H3K4 "
            "methyltransferases → epigenetic activation of gene expression (cyclin-dependent kinase "
            "inhibitors p27Kip1, p18Ink4c); "
            "Also interacts with JunD (AP-1 transcription factor) → menin represses JunD → "
            "LOF → JunD constitutively active → proliferation. "
            "TUMOUR SUPPRESSOR MECHANISM: Two-hit model (Knudson); germline LOF allele + "
            "somatic LOH at 11q13 in tumour cells → complete menin loss → tumour development. "
            "CLINICAL TRIAD: "
            "(1) PRIMARY HYPERPARATHYROIDISM (pHPT): 90% by age 50; 4-gland hyperplasia (not adenoma); "
            "multigland disease; high recurrence after subtotal parathyroidectomy; "
            "cinacalcet bridge while awaiting surgery. "
            "(2) PANCREATIC NETs: Gastrinoma (ZES — Zollinger-Ellison syndrome, MEN1 accounts for 25% of ZES); "
            "Insulinoma (hypoglycaemia); VIPoma; Glucagonoma; Non-functioning pNETs. "
            "(3) PITUITARY ADENOMAS: Prolactinoma most common (60%); GH-secreting (acromegaly); "
            "ACTH-secreting (Cushing's); non-functioning. "
            "ANNUAL SCREENING PROTOCOL (from age 8): "
            "Ca²⁺, PTH, fasting glucose, insulin, gastrin, glucagon, VIP, chromogranin A; "
            "MRI pituitary every 3-5 years; CT/MRI pancreas every 1-2 years (when pNET suspected)."
        ),
        "phenotype": (
            "Primary hyperparathyroidism: Usually mild hypercalcaemia (2.6-3.0 mmol/L); "
            "PTH elevated; 4-gland hyperplasia (NOT single adenoma); recurrent after subtotal parathyroidectomy; "
            "Nephrolithiasis more common than sporadic pHPT; osteoporosis; "
            "Pancreatic NETs: gastrinoma → severe peptic ulceration (ZES), diarrhoea; "
            "Insulinoma → fasting hypoglycaemia; "
            "Pituitary adenoma: prolactinoma → amenorrhoea/galactorrhoea/ED; "
            "GH adenoma → acromegaly (hands/feet/jaw enlargement, sleep apnoea); "
            "Full MEN1 triad occurs in ~60% by age 50; "
            "Carcinoid tumours (thymic, bronchial, gastric); Lipomas; Facial angiofibromas"
        ),
        "key_hallmarks": [
            "MEN1 HYPERPARATHYROIDISM: 4-GLAND hyperplasia — NOT single adenoma; sestamibi often FALSE NEGATIVE",
            "Multigland disease → SUBTOTAL parathyroidectomy (3.5 glands) or total + autotransplantation",
            "Annual biochemical screen from age 8: Ca, PTH, gastrin, glucose, insulin, chromogranin A; prolactin",
            "ZES (gastrinoma): PROTON PUMP INHIBITOR titrated to pH >4 — high-dose PPI FIRST before resection decision",
            "INSULINOMA: fasting test + 72-hour fast for diagnosis; surgical resection for localised disease",
        ],
        "treatment_alerts": [
            "MEN1 pHPT: SUBTOTAL parathyroidectomy (3.5 glands) + cervical thymectomy (thymic carcinoids); NOT single-gland surgery",
            "Cinacalcet: bridge to surgery or for mild/elderly patients with MEN1 pHPT; does not cure",
            "ZES: HIGH-DOSE PPI (omeprazole 40-120 mg/day) mandatory; octreotide for refractory; resection for localized",
            "Insulinoma: diazoxide pre-op; laparoscopic resection curative for single lesion; glucagon kit for emergencies",
            "Prolactinoma: cabergoline first-line; dopamine agonists usually effective (larger dose than sporadic); surgery rarely needed",
            "GENETIC CASCADE: ALL first-degree relatives offered MEN1 testing; positive → annual screening from age 8",
        ],
        "ddx": [
            "Sporadic primary hyperparathyroidism: single adenoma; FECa >1-2%; no family history; sestamibi positive single gland",
            "MEN2A (RET gene): pHPT + medullary thyroid cancer + phaeochromocytoma; CASR/MEN1 negative",
            "MEN4 (CDKN1B): same triad as MEN1; MEN1 gene normal; CDKN1B testing required",
            "HPT-JT (CDC73): pHPT + jaw ossifying fibroma; parathyroid carcinoma risk; single adenoma/carcinoma (NOT hyperplasia)",
            "Familial Isolated Hyperparathyroidism (FIHP): pHPT without extra-parathyroid features; may be MEN1 variant or CASR/GCM2",
        ],
        "seed": SEED_BASE + 3,
        "n_patients": 40,
        "age_range": (8, 70),
        "female_pct": 52,
    },
    # ── CDC73 — HPT-JT / Parathyroid Carcinoma ──
    {
        "gene": "CDC73",
        "protein": "Cell Division Cycle protein 73 (Parafibromin)",
        "alias": (
            "CDC73 (formerly HRPT2); OMIM gene 607393; "
            "Hyperparathyroidism-Jaw Tumour syndrome (HPT-JT) #145001; autosomal dominant; 1q25.3; "
            "531 aa; ~60 kDa; PAF1/RNA Pol II elongation complex component; nuclear tumour suppressor; "
            "Parafibromin IHC: ABSENT in parathyroid carcinoma — KEY DIAGNOSTIC MARKER; "
            "PARATHYROID CARCINOMA in 15-20% of HPT-JT (vs <1% in sporadic pHPT); "
            "Jaw ossifying fibroma (NOT osteitis fibrosa cystica — radiological distinction critical)"
        ),
        "aa": "531 aa",
        "kDa": "~60 kDa",
        "locus": "1q25.3",
        "omim_gene": 607393,
        "omim_disease": 145001,
        "inheritance": (
            "Autosomal dominant; germline LOF + somatic second hit; "
            "50% risk to offspring; penetrance ~75-80% by age 70; "
            "de novo variants occur; variable expressivity; "
            "HPT-JT families: some members have only hyperparathyroidism (no jaw tumour); "
            "Sporadic parathyroid carcinoma: ~70% have SOMATIC CDC73 biallelic LOF (check germline in all PC)"
        ),
        "gene_class": (
            "CDC73 encodes parafibromin, a component of the PAF1 transcriptional elongation complex. "
            "MOLECULAR FUNCTION: Parafibromin is a nuclear protein that joins the PAF1 complex "
            "(PAF1, CTR9, LEO1, RTF1, CDC73) — coupled to RNA polymerase II C-terminal domain; "
            "regulates H3K4 trimethylation (active transcription mark) and H3K36 methylation; "
            "TUMOUR SUPPRESSOR MECHANISM: Parafibromin interacts with β-catenin (Wnt pathway) → "
            "sequesters β-catenin in nucleus → represses cyclin D1 transcription → cell cycle arrest; "
            "parafibromin also activates p27 (CDKN1B), p21 (CDKN1A) → G1 arrest; "
            "LOF → Wnt/cyclin D1 derepression + loss of cell cycle brake → parathyroid neoplasia. "
            "DIAGNOSIS: Parafibromin IHC on parathyroid tumour tissue — "
            "ABSENT nuclear staining = CDC73 LOF = carcinoma or high-risk adenoma; "
            "IMPORTANT: Parafibromin loss is the most sensitive/specific IHC marker of parathyroid carcinoma (>95%); "
            "focal loss or reduced staining may indicate carcinoma. "
            "HPT-JT FEATURES: "
            "PARATHYROID: Usually single large gland (adenoma or carcinoma, NOT multigland hyperplasia); "
            "Ca²⁺ often markedly elevated (>3.0-3.5 mmol/L); PTH markedly elevated; "
            "JAW TUMOURS: Ossifying fibroma of mandible or maxilla (NOT osteitis fibrosa cystica — "
            "different histology); slow-growing; jaw X-ray MANDATORY in all HPT-JT families; "
            "RENAL: Hamartomas, cysts, Wilms tumour (rare); renal USS at diagnosis; "
            "UTERINE: Fibroma/carcinosarcoma reported in female patients."
        ),
        "phenotype": (
            "Primary hyperparathyroidism: often SEVERE (Ca²⁺ >3.0 mmol/L, PTH markedly elevated); "
            "Usually SINGLE large parathyroid gland (adenoma or carcinoma); "
            "Parathyroid carcinoma: 15-20% of HPT-JT cases; "
            "SYMPTOMS of severe hypercalcaemia: nephrolithiasis, bone pain, fatigue, cognitive impairment, GI symptoms; "
            "Jaw ossifying fibroma: mandible or maxilla; distinct from brown tumour of osteitis fibrosa; "
            "Renal cysts/hamartomas: usually asymptomatic; Wilms tumour (pediatric, rare); "
            "Uterine pathology: fibroid/carcinosarcoma in some female carriers"
        ),
        "key_hallmarks": [
            "CDC73/HPT-JT: PARATHYROID CARCINOMA risk 15-20% — En-bloc resection (parathyroid + ipsilateral thyroid lobe) mandatory for carcinoma",
            "Parafibromin IHC ABSENT = carcinoma or high-risk adenoma — MOST SENSITIVE marker of parathyroid carcinoma",
            "Jaw ossifying fibroma (NOT osteitis fibrosa): jaw X-ray mandatory in all HPT-JT family members",
            "SEVERE hypercalcaemia (Ca >3.0 mmol/L) in single-gland pHPT → CDC73 germline testing mandatory",
            "ALL sporadic parathyroid carcinomas should have germline CDC73 testing — 70% have somatic biallelic LOF",
        ],
        "treatment_alerts": [
            "PARATHYROID CARCINOMA: En-bloc resection (parathyroid + ipsilateral thyroid lobe + isthmus + neck nodes if involved)",
            "DO NOT rupture capsule during surgery — local recurrence very high after capsule violation",
            "Denosumab: for severe hypercalcaemia-of-malignancy in parathyroid carcinoma; bridges to surgery",
            "Anti-PD-1 immunotherapy: experimental; some responses in recurrent parathyroid carcinoma",
            "Cinacalcet: for inoperable/recurrent parathyroid carcinoma — palliative Ca control",
            "Annual biochemical + imaging surveillance: Ca, PTH, neck USS, CT chest/abdomen in HPT-JT families",
        ],
        "ddx": [
            "MEN1 pHPT: 4-gland hyperplasia (NOT single gland); no jaw tumour; no carcinoma risk; menin IHC retained",
            "Sporadic primary hyperparathyroidism: single adenoma; no family history; CDC73 germline negative; parafibromin IHC usually positive",
            "Sporadic parathyroid carcinoma: may lack family history but 70% somatic CDC73 biallelic; test germline",
            "Familial Isolated Hyperparathyroidism (FIHP): pHPT without extra-parathyroid features; may be CDC73 without jaw tumour",
            "Jaw ossifying fibroma vs osteitis fibrosa cystica: distinct histology; OFC is complication of severe/prolonged pHPT (any cause)",
        ],
        "seed": SEED_BASE + 4,
        "n_patients": 40,
        "age_range": (15, 75),
        "female_pct": 54,
    },
    # ── CDKN1B — MEN4 ──
    {
        "gene": "CDKN1B",
        "protein": "Cyclin-Dependent Kinase Inhibitor 1B (p27/Kip1)",
        "alias": (
            "CDKN1B; OMIM gene 600778; Multiple Endocrine Neoplasia type 4 (MEN4) #610755; autosomal dominant; 12p13.1; "
            "198 aa; ~22 kDa (functional form); CDK2/CDK4 inhibitor; G1/S phase cell cycle checkpoint; "
            "MEN4: MEN1-phenotype (pHPT + pituitary + pNETs) with NORMAL MEN1 gene; "
            "Order CDKN1B after MEN1 panel is NEGATIVE in MEN1 phenotype; "
            "Rarer than MEN1; estimated 1-5% of MEN1-phenotype cases"
        ),
        "aa": "198 aa",
        "kDa": "~22 kDa",
        "locus": "12p13.1",
        "omim_gene": 600778,
        "omim_disease": 610755,
        "inheritance": (
            "Autosomal dominant; germline LOF variants; "
            "50% risk to offspring; penetrance incomplete and variable; "
            "Also found in sporadic pituitary adenomas (somatic LOH at 12p13); "
            "p27 abundance is regulated post-translationally (ubiquitin-mediated) — "
            "pathogenic variants may disrupt protein stability or nuclear localisation"
        ),
        "gene_class": (
            "CDKN1B encodes p27Kip1, a member of the Cip/Kip family of cyclin-dependent kinase inhibitors. "
            "MOLECULAR FUNCTION: p27 binds CDK2-cyclin E and CDK2-cyclin A complexes → "
            "blocks cell cycle G1-to-S transition; also binds CDK4-cyclin D (weaker); "
            "p27 nuclear import requires Importin-α1 (KPNA2) — cytoplasmic p27 is pro-tumourigenic; "
            "p27 abundance is tightly regulated: synthesised in G0/G1, degraded in S phase via "
            "Skp2-mediated ubiquitination (F-box protein SCFSkp2 E3 ligase). "
            "MENIN CONNECTION: Menin activates CDKN1B transcription → "
            "MEN1 LOF → reduced p27 → same downstream cell cycle derepression; "
            "CDKN1B LOF → direct p27 loss → same outcome = parallel pathway to MEN1 LOF. "
            "MEN4 PHENOTYPE: Clinically resembles MEN1: "
            "(1) Primary hyperparathyroidism (~75% of MEN4 cases); "
            "(2) Pituitary adenomas (30-50%: GH-secreting, ACTH, prolactinoma); "
            "(3) Pancreatic NETs (less common than MEN1, ~10-15%); "
            "(4) Other: adrenal, renal, ovarian/testicular tumours. "
            "KEY DISTINCTION from MEN1: CDKN1B pNETs less frequent and less aggressive than MEN1 pNETs; "
            "gastrinoma rare in MEN4 (common in MEN1); "
            "overall prognosis better in MEN4 than MEN1."
        ),
        "phenotype": (
            "PRIMARY HYPERPARATHYROIDISM: Most common (75%); Ca²⁺ mild-moderate elevation; "
            "PTH elevated; 4-gland hyperplasia or adenoma; "
            "Pituitary adenomas: GH-secreting (acromegaly) more common in MEN4 vs MEN1; "
            "Pancreatic NETs: less common than MEN1; gastrinoma rare; "
            "Adrenal, gonadal tumours: infrequent; "
            "Overall MILDER PHENOTYPE than MEN1 in most series; "
            "MEN1 gene testing NEGATIVE in all MEN4 patients — prerequisite for diagnosis"
        ),
        "key_hallmarks": [
            "MEN4 = MEN1 phenotype with NORMAL MEN1 gene — always sequence CDKN1B next in MEN1-negative MEN phenotype",
            "MEN4 pNETs MILDER than MEN1: fewer aggressive gastrinomas; overall better prognosis",
            "Same annual screening protocol as MEN1: Ca, PTH, gastrin, glucose, pituitary MRI",
            "p27 is post-translationally regulated — WES may underdetect splice/regulatory CDKN1B variants; "
            "clinical suspicion drives genetic counselling even if sequencing ambiguous",
            "CDKN1B also implicated in sporadic pituitary adenomas (somatic) — distinguish germline vs somatic",
        ],
        "treatment_alerts": [
            "MEN4 pHPT: Same as MEN1 — subtotal parathyroidectomy (multigland disease); cinacalcet bridge",
            "Acromegaly (GH adenoma): somatostatin analogues (octreotide/lanreotide) first-line; surgery for large/refractory",
            "Annual biochemical surveillance: Ca, PTH, fasting glucose, IGF-1, prolactin, ACTH — same as MEN1 protocol",
            "Cascade testing: all first-degree relatives; same surveillance schedule; genetic counselling",
            "MEN4 pNETs: less aggressive — surgery reserved for symptomatic or growing lesions >2 cm",
        ],
        "ddx": [
            "MEN1: identical triad; MEN1 gene positive; gastrinoma more common; aggressive pNETs; MEN1-negative excludes",
            "MEN2A (RET): pHPT + medullary thyroid carcinoma + phaeochromocytoma; no pituitary; different gene",
            "Sporadic MEN-like: no family history; somatic variants; germline panel negative",
            "FIHP (Familial Isolated Hyperparathyroidism): only pHPT without pituitary/pNET; may be MEN1/CDC73/GCM2",
        ],
        "seed": SEED_BASE + 5,
        "n_patients": 40,
        "age_range": (15, 70),
        "female_pct": 51,
    },
    # ── GCM2 — Familial Isolated Hypoparathyroidism ──
    {
        "gene": "GCM2",
        "protein": "Glial Cells Missing transcription factor 2 (GCMB)",
        "alias": (
            "GCM2 (GCMB); OMIM gene 603716; Familial Isolated Hypoparathyroidism (FIH) #146200 (AD/AR LOF); "
            "Familial Isolated Hyperparathyroidism (FIHP) (AD GOF); 6p24.2; "
            "502 aa; ~58 kDa; GCM-domain TF; essential for parathyroid gland development; "
            "GCM2 is the master regulator of parathyroid gland specification from pharyngeal pouches; "
            "LOF → parathyroid gland aplasia/hypoplasia → hypoparathyroidism; "
            "GOF → parathyroid hyperplasia → hyperparathyroidism"
        ),
        "aa": "502 aa",
        "kDa": "~58 kDa",
        "locus": "6p24.2",
        "omim_gene": 603716,
        "omim_disease": 146200,
        "inheritance": (
            "AD LOF → Familial Isolated Hypoparathyroidism (autosomal dominant); "
            "AR LOF → Familial Isolated Hypoparathyroidism (autosomal recessive, more severe); "
            "AD GOF → Familial Isolated Hyperparathyroidism (FIHP) — rare; "
            "Penetrance: AD form variable; AR form high penetrance; "
            "De novo variants occur; consanguinity associated with AR form"
        ),
        "gene_class": (
            "GCM2 (also called GCMB in humans) encodes a transcription factor with a unique GCM DNA-binding domain "
            "(recognises GCM motif: 5'-ATGCGGGT-3'). "
            "DEVELOPMENTAL FUNCTION: GCM2 is the master transcription factor required for parathyroid gland "
            "specification from the 3rd and 4th pharyngeal pouches; "
            "expressed from embryonic day E9.5 in pharyngeal pouch endoderm; "
            "activates PTH gene transcription + CASR expression in parathyroid chief cells; "
            "GCM2 null mice: complete absence of parathyroid glands → postnatal hypocalcaemic death. "
            "PARATHYROID vs THYMUS: GCM2 expression marks parathyroid progenitors; "
            "TBX1 expression marks thymic progenitors; they share the pharyngeal pouch endoderm — "
            "GCM2 LOF → parathyroid aplasia WITHOUT thymic aplasia (different from DiGeorge/TBX1). "
            "PATHOPHYSIOLOGY LOF: GCM2 haploinsufficiency (AD) or biallelic LOF (AR) → "
            "reduced parathyroid cell mass → insufficient PTH secretion → "
            "hypoparathyroidism: low Ca²⁺, high PO4, undetectable/low PTH. "
            "PATHOPHYSIOLOGY GOF: GCM2 GOF variants → parathyroid hyperplasia → "
            "autonomous PTH secretion → primary hyperparathyroidism (FIHP). "
            "MAGNESIUM: Hypomagnesaemia impairs PTH secretion AND PTH end-organ action → "
            "ALWAYS CHECK AND CORRECT Mg²⁺ FIRST in hypoparathyroidism — "
            "Mg deficiency can cause PTH-resistant hypocalcaemia that will not respond to Ca supplements alone."
        ),
        "phenotype": (
            "Familial Isolated Hypoparathyroidism (FIH): "
            "Low serum Ca²⁺ (<2.0 mmol/L) with high PO4 and low/undetectable PTH; "
            "Symptoms: Tetany, carpopedal spasm, perioral paraesthesiae, laryngospasm (emergency), seizures; "
            "Basal ganglia calcifications (CT: Fahr disease pattern) — long-standing hypoparathyroidism; "
            "High bone density (PTH deficiency → osteoclast suppression → cortical bone increases); "
            "Cataracts (chronic hypocalcaemia → lens opacification); "
            "GCM2 FIH: No thymic aplasia, no cardiac defects — differentiates from DiGeorge; "
            "FIHP (GCM2 GOF): Primary hyperparathyroidism without extra-parathyroid features"
        ),
        "key_hallmarks": [
            "GCM2 FIH: No thymic/cardiac anomalies — differentiates from DiGeorge (22q11.2 deletion) and GATA3 mutation (HDR syndrome)",
            "MAGNESIUM FIRST: Correct hypoMg before Ca + calcitriol — hypoMg causes PTH resistance AND impairs PTH release",
            "Hypoparathyroidism GOAL: Serum Ca 2.0-2.1 mmol/L (low-NORMAL) — avoid over-treatment → nephrocalcinosis",
            "Thiazide diuretics: reduce urinary Ca excretion → useful adjunct to reduce Ca supplement dose and calciuria",
            "Recombinant PTH (teriparatide 20 µg SC BD): for brittle hypoparathyroidism not controlled on Ca/calcitriol",
        ],
        "treatment_alerts": [
            "ACUTE HYPOCALCAEMIA: IV calcium gluconate 10% (10 mL over 10 min, then infusion) — emergency",
            "CHRONIC: Calcitriol 0.25-2 µg/day + Ca carbonate/citrate; titrate to Ca 2.0-2.1 mmol/L",
            "MAGNESIUM MANDATORY: Check Mg²⁺ in all hypoparathyroidism; correct to >0.8 mmol/L before Ca/calcitriol",
            "Thiazide diuretics (hydrochlorothiazide 12.5-25 mg): reduce calciuria → allow lower Ca supplement dose",
            "Renal monitoring: annual renal USS + urine Ca/Cr ratio (nephrocalcinosis risk even with 'therapeutic' Ca levels)",
            "Recombinant PTH 1-84 (Natpara): licensed for chronic hypoparathyroidism in some regions; reduces calcitriol/Ca dose",
        ],
        "ddx": [
            "DiGeorge syndrome (22q11.2 deletion): hypoparathyroidism + thymic aplasia + conotruncal cardiac defects + learning difficulty; TBX1 region",
            "HDR syndrome (GATA3 LOF): Hypoparathyroidism + Deafness + Renal anomalies; GCM2 normal",
            "KSS/MELAS (mitochondrial): hypoparathyroidism + other mitochondrial features; mtDNA/nuclear mt genes",
            "Autoimmune hypoparathyroidism: APS-1 (AIRE gene); anti-CASR antibodies; acquired; often with other autoimmune",
            "Pseudohypoparathyroidism (GNAS): PTH high (PTH resistance), not low; Albright hereditary osteodystrophy features",
        ],
        "seed": SEED_BASE + 6,
        "n_patients": 40,
        "age_range": (0, 60),
        "female_pct": 56,
    },
    # ── PTH — Isolated Familial Hypoparathyroidism ──
    {
        "gene": "PTH",
        "protein": "Parathyroid Hormone preproprotein",
        "alias": (
            "PTH; OMIM gene 168450; Familial Isolated Hypoparathyroidism (IHP) #146200 (AD/AR); "
            "11p15.3; 115 aa (preproPTH) → 84 aa mature secreted hormone; ~10 kDa (mature); "
            "prepro sequence: 25 aa signal + 6 aa pro-sequence → cleaved to 84 aa PTH(1-84); "
            "PTH(1-34) N-terminal biologically active domain; PTH(1-84) secreted; "
            "AD variants → dominant negative misfolded PTH → ER stress → chief cell death; "
            "AR variants → complete PTH deficiency; Low Ca + Low/undetectable PTH + High PO4"
        ),
        "aa": "115 aa (preproPTH)",
        "kDa": "~10 kDa (mature PTH 1-84)",
        "locus": "11p15.3",
        "omim_gene": 168450,
        "omim_disease": 146200,
        "inheritance": (
            "AD: signal peptide variants → misfolded preproPTH → dominant negative ER stress → chief cell apoptosis; "
            "AR: biallelic LOF → complete PTH deficiency; "
            "AD penetrance: variable (signal peptide Cys18Arg is a common AD variant); "
            "AR penetrance: high (biallelic null); "
            "Rare: X-linked hypoparathyroidism maps to Xq27 (different locus); "
            "PTH gene variants are less common cause of FIH than GCM2 — always panel both"
        ),
        "gene_class": (
            "PTH encodes preproPTH (115 amino acids): "
            "25 aa signal peptide (ER entry) + 6 aa pro-sequence (Golgi cleavage) + 84 aa mature PTH. "
            "BIOSYNTHESIS: preproPTH synthesized on ribosomes → signal peptide cleaved in ER lumen "
            "(by signal peptidase) → proPTH transported to Golgi → pro-sequence cleaved (by furin) → "
            "PTH(1-84) stored in secretory granules of parathyroid chief cells; "
            "PTH(1-84) secreted in response to low Ca²⁺ (detected by CASR) → bloodstream → "
            "PTH(1-34) is the biologically active N-terminal fragment. "
            "MECHANISM OF ACTION: PTH → PTH1R (GPCR on osteoblasts and kidney tubular cells) → "
            "Gαs → cAMP → PKA → "
            "(1) Kidney: stimulates Ca²⁺ reabsorption (distal tubule), inhibits PO4 reabsorption (proximal); "
            "stimulates 25-OH-D → 1,25-OH2-D (calcitriol) activation; "
            "(2) Bone: stimulates osteoclast activation via RANKL on osteoblasts → bone resorption → Ca release. "
            "PATHOPHYSIOLOGY AD LOF (dominant negative): Signal peptide variants → "
            "misfolded preproPTH trapped in ER → activates unfolded protein response (UPR) → "
            "ER stress → chief cell apoptosis → reduced PTH secretory capacity; "
            "dominant over normal allele because misfolded protein oligomerises with wild-type. "
            "PATHOPHYSIOLOGY AR LOF: Biallelic null → complete absence of PTH secretion → "
            "profound hypocalcaemia from birth (neonatal tetany)."
        ),
        "phenotype": (
            "Familial Isolated Hypoparathyroidism (PTH gene): "
            "Low Ca²⁺ (<2.0 mmol/L); High PO4 (>1.6 mmol/L); PTH UNDETECTABLE or very low; "
            "AD (signal peptide variants): onset childhood or adult; variable severity; "
            "AR (biallelic): neonatal onset; severe; "
            "Tetany: carpopedal spasm, perioral paraesthesiae; Trousseau sign positive; "
            "Chvostek sign positive (tap facial nerve → facial twitch); "
            "Seizures in severe/acute hypocalcaemia; "
            "Basal ganglia calcifications (chronic); Cataracts (chronic); "
            "High bone density (cortical — PTH deficiency suppresses bone resorption); "
            "NO thymic/cardiac anomalies (distinguishes from DiGeorge)"
        ),
        "key_hallmarks": [
            "PTH gene variants cause IHP — but GCM2 more common; always panel BOTH PTH + GCM2 in FIH",
            "Undetectable PTH differentiates IHP from pseudohypoparathyroidism (PHPIa/Ib: PTH HIGH due to resistance)",
            "BONE DENSITY HIGH in hypoparathyroidism: PTH deficiency → suppressed bone resorption → cortical bone dense",
            "AVOID OVER-TREATMENT: Target Ca 2.0-2.1 mmol/L — goal is symptom prevention NOT normocalcaemia",
            "Thiazide diuretics reduce calciuria in IHP — useful adjunct alongside Ca/calcitriol to reduce nephrocalcinosis",
        ],
        "treatment_alerts": [
            "ACUTE: IV calcium gluconate 10% bolus + infusion; cardiac monitoring (QTc); ECG during IV Ca",
            "CHRONIC: Calcitriol (0.25-2 µg/day) + Ca carbonate 1-2 g/day; titrate to Ca 2.0-2.1 mmol/L",
            "MAGNESIUM CHECK mandatory: Mg deficiency → PTH secretion impaired + PTH resistance → hypocalcaemia refractory",
            "NEPHROCALCINOSIS MONITORING: Annual renal USS + spot urine Ca/creatinine ratio; adjust calcitriol if rising",
            "Recombinant PTH 1-34 (teriparatide) or 1-84 (Natpara): for brittle IHP; reduces calciuria vs calcitriol",
            "Pregnancy: hypoparathyroidism often IMPROVES (placental PTHrP); reduce calcitriol dose; neonatal monitoring",
        ],
        "ddx": [
            "GCM2 FIH: Same phenotype; GCM2 gene positive vs PTH gene positive; panel both simultaneously",
            "Pseudohypoparathyroidism type 1a/1b (GNAS): PTH HIGH (resistance); Albright osteodystrophy in type 1a; short metacarpal",
            "DiGeorge syndrome (22q11.2): thymic aplasia + conotruncal cardiac + learning difficulty; chromosomal deletion",
            "Autoimmune hypoparathyroidism (APS-1): AIRE mutations; acquired polyglandular autoimmune; anti-CASR antibodies",
            "Hypomagnesaemia-induced hypoparathyroidism: Mg deficiency → impaired PTH release; correct Mg first (and only)",
        ],
        "seed": SEED_BASE + 7,
        "n_patients": 40,
        "age_range": (0, 60),
        "female_pct": 55,
    },
]

# ── Patient Cohort Generator ──────────────────────────────────────────────────
def _generate_cohort(gd: dict) -> list:
    rng = random.Random(gd["seed"])
    patients = []
    gene = gd["gene"]
    lo, hi = gd["age_range"]
    female_pct = gd["female_pct"]

    for i in range(gd["n_patients"]):
        pid = f"{gene}-{gd['seed']}-{i+1:03d}"
        sex = "F" if rng.random() * 100 < female_pct else "M"
        age_dx = rng.randint(lo, hi)
        dx_delay = round(rng.uniform(0.5, 8.0), 1)

        # Gene-specific fields
        p = {
            "patient_id": pid,
            "gene": gene,
            "sex": sex,
            "age_at_dx": age_dx,
            "dx_delay_years": dx_delay,
        }

        if gene == "CASR":
            phenotype_roll = rng.random()
            if phenotype_roll < 0.65:
                p["phenotype"] = "FHH1"
                p["serum_ca_mmol"] = round(rng.uniform(2.6, 3.1), 2)
                p["pth_pmol"] = round(rng.uniform(3.5, 8.0), 1)
                p["cccr"] = round(rng.uniform(0.005, 0.009), 4)
                p["feca_pct"] = round(rng.uniform(0.3, 0.9), 2)
                p["symptomatic"] = rng.random() < 0.2
                p["surgery_done_incorrectly"] = rng.random() < 0.08
                p["thiazide_given_incorrectly"] = rng.random() < 0.05
            elif phenotype_roll < 0.85:
                p["phenotype"] = "ADH1"
                p["serum_ca_mmol"] = round(rng.uniform(1.55, 1.95), 2)
                p["pth_pmol"] = round(rng.uniform(0.5, 2.0), 1)
                p["nephrocalcinosis"] = rng.random() < 0.30
                p["basal_ganglia_calcification"] = rng.random() < 0.25
                p["calcitriol_given"] = True
                p["target_ca_achieved"] = rng.random() < 0.65
            else:
                p["phenotype"] = "NSHPT"
                p["serum_ca_mmol"] = round(rng.uniform(3.5, 5.0), 2)
                p["pth_pmol"] = round(rng.uniform(30, 120), 1)
                p["parathyroidectomy_done"] = True
                p["age_at_dx"] = rng.randint(0, 1)  # neonatal

        elif gene == "GNA11":
            phenotype_roll = rng.random()
            if phenotype_roll < 0.70:
                p["phenotype"] = "FHH2"
                p["serum_ca_mmol"] = round(rng.uniform(2.6, 3.1), 2)
                p["cccr"] = round(rng.uniform(0.004, 0.009), 4)
                p["casr_gene_normal"] = True
            else:
                p["phenotype"] = "ADH2"
                p["serum_ca_mmol"] = round(rng.uniform(1.55, 1.95), 2)
                p["nephrocalcinosis"] = rng.random() < 0.28
                p["de_novo_variant"] = rng.random() < 0.45
            p["symptomatic"] = rng.random() < 0.25

        elif gene == "AP2S1":
            p["phenotype"] = "FHH3"
            p["serum_ca_mmol"] = round(rng.uniform(2.7, 3.2), 2)
            p["cccr"] = round(rng.uniform(0.004, 0.009), 4)
            p["arg15_variant"] = rng.choice(["Arg15His", "Arg15Cys", "Arg15Leu"])
            p["symptomatic"] = rng.random() < 0.40  # more symptomatic than FHH1/2
            p["neuropsychiatric_features"] = rng.random() < 0.30
            p["osmolality_abnormal"] = rng.random() < 0.20
            p["surgery_done_incorrectly"] = rng.random() < 0.10

        elif gene == "MEN1":
            p["pHPT"] = True
            p["serum_ca_mmol"] = round(rng.uniform(2.7, 3.3), 2)
            p["pth_pmol"] = round(rng.uniform(8.0, 35.0), 1)
            p["pancreatic_net"] = rng.random() < 0.45
            p["gastrinoma_zes"] = rng.random() < 0.25 if p["pancreatic_net"] else False
            p["insulinoma"] = rng.random() < 0.15 if p["pancreatic_net"] else False
            p["pituitary_adenoma"] = rng.random() < 0.38
            p["pituitary_type"] = rng.choice(["prolactinoma", "GH", "ACTH", "non-functioning"]) if p["pituitary_adenoma"] else None
            p["4gland_hyperplasia"] = True
            p["subtotal_ptx_done"] = rng.random() < 0.55
            p["recurrence_post_ptx"] = rng.random() < 0.30 if p["subtotal_ptx_done"] else False
            p["annual_screen_compliant"] = rng.random() < 0.72
            p["nephrolithiasis"] = rng.random() < 0.35

        elif gene == "CDC73":
            p["serum_ca_mmol"] = round(rng.uniform(2.9, 3.8), 2)
            p["pth_pmol"] = round(rng.uniform(15.0, 80.0), 1)
            p["parathyroid_carcinoma"] = rng.random() < 0.17
            p["jaw_ossifying_fibroma"] = rng.random() < 0.50
            p["renal_cysts_hamartomas"] = rng.random() < 0.25
            p["parafibromin_ihc_absent"] = p["parathyroid_carcinoma"]
            p["en_bloc_resection_done"] = p["parathyroid_carcinoma"] and rng.random() < 0.75
            p["capsule_violated_intraop"] = p["parathyroid_carcinoma"] and rng.random() < 0.20
            p["recurrence"] = p["parathyroid_carcinoma"] and (p["capsule_violated_intraop"] or rng.random() < 0.25)
            p["cinacalcet_used"] = rng.random() < 0.30

        elif gene == "CDKN1B":
            p["pHPT"] = True
            p["serum_ca_mmol"] = round(rng.uniform(2.7, 3.1), 2)
            p["pth_pmol"] = round(rng.uniform(7.0, 28.0), 1)
            p["pituitary_adenoma"] = rng.random() < 0.38
            p["pituitary_type"] = rng.choice(["GH/acromegaly", "prolactinoma", "non-functioning"]) if p["pituitary_adenoma"] else None
            p["pancreatic_net"] = rng.random() < 0.12
            p["men1_gene_normal"] = True  # prerequisite for MEN4 diagnosis
            p["annual_screen_compliant"] = rng.random() < 0.68

        elif gene == "GCM2":
            p_type = "FIH" if rng.random() < 0.80 else "FIHP"
            p["phenotype"] = p_type
            if p_type == "FIH":
                p["serum_ca_mmol"] = round(rng.uniform(1.45, 1.95), 2)
                p["serum_phosphate_mmol"] = round(rng.uniform(1.6, 2.4), 2)
                p["pth_pmol"] = round(rng.uniform(0.3, 1.5), 2)
                p["tetany_episodes"] = rng.randint(0, 8)
                p["seizures"] = rng.random() < 0.20
                p["basal_ganglia_calcification"] = rng.random() < 0.28
                p["nephrocalcinosis"] = rng.random() < 0.22
                p["calcitriol_on_treatment"] = True
                p["mg_checked_corrected"] = rng.random() < 0.60
            else:
                p["serum_ca_mmol"] = round(rng.uniform(2.7, 3.1), 2)
                p["pth_pmol"] = round(rng.uniform(7.0, 20.0), 1)
                p["extra_parathyroid_features"] = False
            p["inheritance"] = rng.choice(["AD", "AR"]) if p_type == "FIH" else "AD_GOF"

        elif gene == "PTH":
            inh = "AD" if rng.random() < 0.55 else "AR"
            p["inheritance_type"] = inh
            p["serum_ca_mmol"] = round(rng.uniform(1.4, 1.9), 2)
            p["serum_phosphate_mmol"] = round(rng.uniform(1.6, 2.5), 2)
            p["pth_pmol"] = round(rng.uniform(0.1, 1.2), 2)
            p["tetany_episodes"] = rng.randint(0, 10)
            p["seizures"] = rng.random() < 0.22
            p["basal_ganglia_calcification"] = rng.random() < 0.25
            p["nephrocalcinosis"] = rng.random() < 0.20
            p["mg_checked"] = rng.random() < 0.55
            p["mg_corrected"] = p["mg_checked"] and rng.random() < 0.80
            p["calcitriol_on_treatment"] = True
            p["variant_location"] = rng.choice(["signal_peptide", "pro_sequence", "mature_PTH"])
            p["er_stress_mechanism"] = p["inheritance_type"] == "AD" and p["variant_location"] == "signal_peptide"
            if inh == "AR":
                p["age_at_dx"] = rng.randint(0, 2)  # neonatal/infantile for AR

        patients.append(p)
    return patients


_ALL_COHORTS = {gd["gene"]: _generate_cohort(gd) for gd in PARATHYROID_GENES}


# ── Public API Functions ───────────────────────────────────────────────────────
def get_overview() -> dict:
    n = sum(len(c) for c in _ALL_COHORTS.values())
    # Gene-specific summary stats
    casr_fhh = sum(1 for p in _ALL_COHORTS["CASR"] if p.get("phenotype") == "FHH1")
    casr_adh = sum(1 for p in _ALL_COHORTS["CASR"] if p.get("phenotype") == "ADH1")
    casr_nshpt = sum(1 for p in _ALL_COHORTS["CASR"] if p.get("phenotype") == "NSHPT")
    casr_surgery_incorrect = sum(1 for p in _ALL_COHORTS["CASR"] if p.get("surgery_done_incorrectly"))
    casr_thiazide_incorrect = sum(1 for p in _ALL_COHORTS["CASR"] if p.get("thiazide_given_incorrectly"))
    men1_pituitary = sum(1 for p in _ALL_COHORTS["MEN1"] if p.get("pituitary_adenoma"))
    men1_pnet = sum(1 for p in _ALL_COHORTS["MEN1"] if p.get("pancreatic_net"))
    men1_recurrence = sum(1 for p in _ALL_COHORTS["MEN1"] if p.get("recurrence_post_ptx"))
    cdc73_carcinoma = sum(1 for p in _ALL_COHORTS["CDC73"] if p.get("parathyroid_carcinoma"))
    cdc73_jaw = sum(1 for p in _ALL_COHORTS["CDC73"] if p.get("jaw_ossifying_fibroma"))
    cdc73_capsule_violation = sum(1 for p in _ALL_COHORTS["CDC73"] if p.get("capsule_violated_intraop"))
    gcm2_fih = sum(1 for p in _ALL_COHORTS["GCM2"] if p.get("phenotype") == "FIH")
    pth_seizures = sum(1 for p in _ALL_COHORTS["PTH"] if p.get("seizures"))
    pth_mg_not_checked = sum(1 for p in _ALL_COHORTS["PTH"] if not p.get("mg_checked"))
    ap2s1_symptomatic = sum(1 for p in _ALL_COHORTS["AP2S1"] if p.get("symptomatic"))
    mean_dx_delay = round(
        sum(p["dx_delay_years"] for c in _ALL_COHORTS.values() for p in c) / n, 1
    )

    return {
        "atlas_name": "Parathyroid-Disorders-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Parathyroid & Calcium Metabolism Disorders Atlas · "
            "CASR-1078aa-3q21.1-AD-FHH1/ADH1/AR-NSHPT-CCCR<0.01-Thiazide-CI · "
            "GNA11-359aa-19p13.3-AD-FHH2/ADH2-CASR-Negative-FHH · "
            "AP2S1-142aa-19q13.3-AD-FHH3-Arg15-Hotspot-More-Symptomatic · "
            "MEN1-610aa-11q13.1-AD-4-Gland-Hyperplasia-90pct-Annual-Screen-Age-8 · "
            "CDC73-531aa-1q25.3-AD-HPT-JT-Carcinoma-15pct-Parafibromin-IHC-Absent · "
            "CDKN1B-198aa-12p13.1-AD-MEN4-MEN1-Negative-Same-Triad · "
            "GCM2-502aa-6p24.2-AD-AR-FIH-Mg-First-Before-Ca · "
            "PTH-115aa-11p15.3-AD-AR-IHP-Undetectable-PTH-Avoid-Over-Treatment"
        ),
        "n_patients": n,
        "gene_count": len(PARATHYROID_GENES),
        "seeds": "1342-1349",
        "disease_category": "Hereditary Parathyroid & Calcium Metabolism Disorders",
        "casr_phenotypes": {"FHH1": casr_fhh, "ADH1": casr_adh, "NSHPT": casr_nshpt},
        "casr_surgery_incorrect": casr_surgery_incorrect,
        "casr_thiazide_incorrect": casr_thiazide_incorrect,
        "men1_pituitary_adenomas": men1_pituitary,
        "men1_pancreatic_nets": men1_pnet,
        "men1_pHPT_recurrence": men1_recurrence,
        "cdc73_parathyroid_carcinomas": cdc73_carcinoma,
        "cdc73_jaw_tumours": cdc73_jaw,
        "cdc73_capsule_violations": cdc73_capsule_violation,
        "gcm2_fih_cases": gcm2_fih,
        "pth_seizures": pth_seizures,
        "pth_mg_not_checked": pth_mg_not_checked,
        "ap2s1_symptomatic": ap2s1_symptomatic,
        "mean_diagnosis_delay_years": mean_dx_delay,
        "critical_drug_alerts": {
            "CASR_FHH_THIAZIDE_CI": "THIAZIDES ABSOLUTELY CONTRAINDICATED in FHH — reduce urinary Ca → severe hypercalcaemia crisis",
            "CASR_FHH_NO_SURGERY": "FHH: PARATHYROIDECTOMY FAILS — all 4 glands reset at wrong Ca threshold; causes permanent hypoparathyroidism",
            "ADH1_2_TARGET_LOW_NORMAL_CA": "ADH1/ADH2: Ca target LOW-NORMAL 2.0-2.1 mmol/L ONLY — higher target → nephrocalcinosis (calciuria continues)",
            "HYPOPARATHYROIDISM_MG_FIRST": "Hypoparathyroidism: CORRECT Mg²⁺ FIRST — hypoMg blocks PTH secretion AND causes PTH resistance",
            "HYPOPARATHYROIDISM_AVOID_OVERCORRECT": "Hypoparathyroidism: NEVER target normocalcaemia — nephrocalcinosis develops even at 2.3 mmol/L",
            "CDC73_EN_BLOC": "CDC73 parathyroid carcinoma: EN-BLOC resection mandatory — capsule rupture → near-certain local recurrence",
        },
        "kpis": [
            {"label": "Total Patients", "value": str(n)},
            {"label": "Genes Covered", "value": str(len(PARATHYROID_GENES))},
            {"label": "CDC73 Carcinomas", "value": f"{cdc73_carcinoma}/40 ({round(cdc73_carcinoma/40*100,1)}%)"},
            {"label": "MEN1 Pituitary Adenomas", "value": f"{men1_pituitary}/40"},
            {"label": "PTH Mg Not Checked", "value": f"{pth_mg_not_checked}/40"},
            {"label": "FHH3 (AP2S1) Symptomatic", "value": f"{ap2s1_symptomatic}/40"},
            {"label": "CASR Thiazide Incorrect", "value": f"{casr_thiazide_incorrect}/40"},
            {"label": "Mean Dx Delay", "value": f"{mean_dx_delay} yrs"},
        ],
        "drug_alerts": [
            "FHH (CASR/GNA11/AP2S1): THIAZIDE DIURETICS CONTRAINDICATED — severe hypercalcaemia crisis",
            "FHH (all types): NO PARATHYROIDECTOMY — surgery fails; all 4 glands affected; permanent hypoparathyroidism result",
            "ADH1/ADH2: Ca supplement target 2.0-2.1 mmol/L ONLY — higher → nephrocalcinosis",
            "Hypoparathyroidism (GCM2/PTH): Check Mg FIRST every time — hypoMg refractory to Ca/calcitriol without Mg correction",
            "CDC73 parathyroid carcinoma: En-bloc resection — capsule rupture → recurrence near-certain",
            "MEN1 pHPT: 4-gland surgery (subtotal parathyroidectomy) — single-gland surgery will fail (multigland disease)",
        ],
        "critical_rules": [
            "CCCR <0.01 in hypercalcaemia = FHH (CASR/GNA11/AP2S1) — NOT primary hyperparathyroidism; NO surgery",
            "FHH diagnostic cascade: CASR → GNA11 → AP2S1 (in sequence if prior negative)",
            "MEN1: Annual biochemical screen from age 8; 4-gland hyperplasia surgery; cascade test all first-degree",
            "CDC73: Parafibromin IHC on every parathyroid tumour specimen — absent = carcinoma; en-bloc mandatory",
            "Hypoparathyroidism Ca target 2.0-2.1 mmol/L (low-normal): annual renal USS for nephrocalcinosis",
            "Mg deficiency corrected BEFORE Ca/calcitriol in hypoparathyroidism — or treatment will fail",
            "All sporadic parathyroid carcinoma: germline CDC73 testing mandatory (70% have somatic biallelic loss)",
            "MEN4 (CDKN1B): Same triad as MEN1 with normal MEN1 gene — order CDKN1B after MEN1 negative WES",
        ],
    }


def get_breakdown() -> list:
    rows = []
    for gd in PARATHYROID_GENES:
        cohort = _ALL_COHORTS[gd["gene"]]
        n = len(cohort)
        female_n = sum(1 for p in cohort if p["sex"] == "F")
        mean_age = round(sum(p["age_at_dx"] for p in cohort) / n, 1)
        mean_delay = round(sum(p["dx_delay_years"] for p in cohort) / n, 1)
        gene = gd["gene"]
        special_features = {}

        if gene == "CASR":
            special_features["fhh1"] = sum(1 for p in cohort if p.get("phenotype") == "FHH1")
            special_features["adh1"] = sum(1 for p in cohort if p.get("phenotype") == "ADH1")
            special_features["nshpt"] = sum(1 for p in cohort if p.get("phenotype") == "NSHPT")
            special_features["surgery_incorrect"] = sum(1 for p in cohort if p.get("surgery_done_incorrectly"))
            special_features["thiazide_incorrect"] = sum(1 for p in cohort if p.get("thiazide_given_incorrectly"))
        elif gene == "GNA11":
            special_features["fhh2"] = sum(1 for p in cohort if p.get("phenotype") == "FHH2")
            special_features["adh2"] = sum(1 for p in cohort if p.get("phenotype") == "ADH2")
            special_features["symptomatic"] = sum(1 for p in cohort if p.get("symptomatic"))
        elif gene == "AP2S1":
            special_features["symptomatic"] = sum(1 for p in cohort if p.get("symptomatic"))
            special_features["neuropsychiatric"] = sum(1 for p in cohort if p.get("neuropsychiatric_features"))
            special_features["surgery_incorrect"] = sum(1 for p in cohort if p.get("surgery_done_incorrectly"))
            arg_dist = {}
            for p in cohort:
                v = p.get("arg15_variant", "unknown")
                arg_dist[v] = arg_dist.get(v, 0) + 1
            special_features["arg15_distribution"] = arg_dist
        elif gene == "MEN1":
            special_features["pituitary_adenoma"] = sum(1 for p in cohort if p.get("pituitary_adenoma"))
            special_features["pancreatic_net"] = sum(1 for p in cohort if p.get("pancreatic_net"))
            special_features["gastrinoma"] = sum(1 for p in cohort if p.get("gastrinoma_zes"))
            special_features["recurrence_post_ptx"] = sum(1 for p in cohort if p.get("recurrence_post_ptx"))
            special_features["nephrolithiasis"] = sum(1 for p in cohort if p.get("nephrolithiasis"))
        elif gene == "CDC73":
            special_features["parathyroid_carcinoma"] = sum(1 for p in cohort if p.get("parathyroid_carcinoma"))
            special_features["jaw_fibroma"] = sum(1 for p in cohort if p.get("jaw_ossifying_fibroma"))
            special_features["renal_cysts"] = sum(1 for p in cohort if p.get("renal_cysts_hamartomas"))
            special_features["capsule_violated"] = sum(1 for p in cohort if p.get("capsule_violated_intraop"))
            special_features["recurrence"] = sum(1 for p in cohort if p.get("recurrence"))
        elif gene == "CDKN1B":
            special_features["pituitary_adenoma"] = sum(1 for p in cohort if p.get("pituitary_adenoma"))
            special_features["pancreatic_net"] = sum(1 for p in cohort if p.get("pancreatic_net"))
            special_features["annual_screen_compliant"] = sum(1 for p in cohort if p.get("annual_screen_compliant"))
        elif gene == "GCM2":
            special_features["fih_cases"] = sum(1 for p in cohort if p.get("phenotype") == "FIH")
            special_features["mg_not_checked"] = sum(1 for p in cohort if p.get("phenotype") == "FIH" and not p.get("mg_checked_corrected"))
            special_features["nephrocalcinosis"] = sum(1 for p in cohort if p.get("nephrocalcinosis"))
            special_features["seizures"] = sum(1 for p in cohort if p.get("seizures"))
        elif gene == "PTH":
            special_features["seizures"] = sum(1 for p in cohort if p.get("seizures"))
            special_features["mg_not_checked"] = sum(1 for p in cohort if not p.get("mg_checked"))
            special_features["nephrocalcinosis"] = sum(1 for p in cohort if p.get("nephrocalcinosis"))
            special_features["er_stress_ad"] = sum(1 for p in cohort if p.get("er_stress_mechanism"))

        rows.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias_summary": gd["alias"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "gene_class": gd["gene_class"],
            "phenotype": gd["phenotype"],
            "key_hallmarks": gd["key_hallmarks"],
            "treatment_alerts": gd["treatment_alerts"],
            "ddx": gd["ddx"],
            "cohort_stats": {
                "n": n,
                "seed": gd["seed"],
                "female_pct": round(female_n / n * 100, 1),
                "age_mean": mean_age,
                "mean_diagnosis_delay_yrs": mean_delay,
                "special_features": special_features,
            },
            "patients": [
                {
                    "patient_id": p["patient_id"],
                    "sex": p["sex"],
                    "age_at_dx": p["age_at_dx"],
                    "dx_delay_years": p["dx_delay_years"],
                    **{k: v for k, v in p.items() if k not in ("patient_id", "sex", "age_at_dx", "dx_delay_years", "gene")},
                }
                for p in cohort
            ],
        })
    return rows


def get_definitions() -> dict:
    return {
        "atlas": "Parathyroid-Disorders-Atlas",
        "definitions": [
            {
                "term": "CCCR (Calcium:Creatinine Clearance Ratio) — The FHH Diagnostic Anchor",
                "short": "CCCR <0.01 = FHH (CASR/GNA11/AP2S1); CCCR >0.02 = primary hyperparathyroidism; thiazides CONTRAINDICATED in FHH",
                "detail": (
                    "The Calcium:Creatinine Clearance Ratio (CCCR) is calculated as: "
                    "(urine Ca × serum Cr) / (serum Ca × urine Cr). "
                    "Normal range: 0.01-0.02; PHPT: usually >0.02; FHH: <0.01 (often 0.004-0.008). "
                    "MECHANISM: In FHH (CASR/GNA11/AP2S1 LOF), CASR in kidney tubules is also insensitive → "
                    "kidney inappropriately reabsorbs Ca even when serum Ca is elevated → "
                    "urine Ca stays low despite hypercalcaemia → CCCR <0.01. "
                    "In PHPT: PTH stimulates Ca reabsorption but filtered load eventually overwhelms → "
                    "FECa > 1-2%, CCCR > 0.02. "
                    "THIAZIDE CONTRAINDICATION: Thiazides reduce urinary Ca excretion → "
                    "in FHH (already very low urinary Ca), thiazides push serum Ca further up → "
                    "severe hypercalcaemia crisis; potentially life-threatening. "
                    "CONFOUNDERS: Vitamin D deficiency, bisphosphonates, lithium all affect CCCR — "
                    "ensure 25-OH-D >50 nmol/L before measuring CCCR; stop bisphosphonates 3 months prior if possible."
                ),
                "clinical_rule": "Hypercalcaemia workup: ALWAYS calculate CCCR before considering parathyroidectomy; CCCR <0.01 = FHH = NO surgery; NO thiazides",
            },
            {
                "term": "FHH Genetic Cascade — CASR → GNA11 → AP2S1",
                "short": "Test CASR first (65% of FHH); GNA11 if CASR negative (5-10%); AP2S1 if both negative (5%); AP2S1 FHH3 more symptomatic; all three = benign, no surgery",
                "detail": (
                    "Familial Hypocalciuric Hypercalcaemia (FHH) is caused by LOF variants in three genes encoding "
                    "the CASR signalling axis: CASR (receptor), GNA11 (Gα11 signal transducer), AP2S1 (CASR endocytosis). "
                    "TESTING SEQUENCE: "
                    "(1) CASR: >65% of genetically confirmed FHH; test first; "
                    "(2) GNA11: ~5-10% of CASR-negative FHH; "
                    "(3) AP2S1: ~5% of CASR/GNA11-negative FHH; Arg15 hotspot (His/Cys/Leu at position 15); "
                    "direct Arg15 Sanger sequencing is rapid if AP2S1 suspected. "
                    "CLINICAL MANAGEMENT regardless of gene: ALL three forms are BENIGN; NO parathyroidectomy; "
                    "NO thiazide diuretics; only symptomatic management (cinacalcet for Ca-related symptoms). "
                    "FHH3 (AP2S1) EXCEPTION: More likely to be symptomatic than FHH1/2 (neuropsychiatric, osmolality); "
                    "trial of cinacalcet for troublesome symptoms; still no surgery. "
                    "DIAGNOSTIC PITFALL: Some physicians operate on FHH (especially if CCCR borderline or FHH3 with symptoms) → "
                    "parathyroidectomy fails (all 4 glands reset) + causes permanent hypoparathyroidism → "
                    "genetic testing BEFORE any operation is mandatory. "
                    "FAMILY TESTING: Once proband confirmed, test all first-degree relatives; "
                    "prevent inadvertent surgery in asymptomating carriers."
                ),
                "clinical_rule": "FHH panel: CASR → GNA11 → AP2S1 in sequence. CCCR <0.01 = FHH = NO surgery + NO thiazides. Cinacalcet for symptoms only",
            },
            {
                "term": "MEN1 Hyperparathyroidism — 4-Gland Disease vs Sporadic Single Adenoma",
                "short": "MEN1 pHPT = 4-gland hyperplasia; sestamibi false-negative; subtotal (3.5-gland) parathyroidectomy + cervical thymectomy; annual screen from age 8",
                "detail": (
                    "MEN1-associated primary hyperparathyroidism (pHPT) is FUNDAMENTALLY DIFFERENT from sporadic pHPT: "
                    "SPORADIC pHPT: ~80% single adenoma (one gland); sestamibi scan positive single gland; "
                    "focused parathyroidectomy (minimally invasive) is curative. "
                    "MEN1 pHPT: 4-gland hyperplasia in nearly all cases; "
                    "sestamibi is OFTEN FALSE NEGATIVE or misleadingly shows 'dominant' gland — "
                    "DO NOT perform focused parathyroidectomy based on sestamibi in MEN1 → will fail and recur. "
                    "SURGICAL APPROACH in MEN1: "
                    "Subtotal parathyroidectomy (3.5 glands removed, 0.5 gland left as vascularised remnant) OR "
                    "Total parathyroidectomy + forearm autotransplantation; "
                    "CERVICAL THYMECTOMY: MANDATORY at time of parathyroid surgery — "
                    "thymic carcinoids (aggressive) can arise in neck/mediastinum; removing cervical thymus "
                    "also captures ectopic parathyroid glands (supernumerary 5th gland in thymus). "
                    "RECURRENCE: 50-70% within 10 years of subtotal parathyroidectomy — "
                    "annual biochemical monitoring (Ca, PTH) is lifelong. "
                    "ZES IN MEN1: Zollinger-Ellison syndrome (gastrinoma) in MEN1 — "
                    "HIGH-DOSE PPI is MANDATORY (target gastric pH >4); "
                    "acid hypersecretion corrected before other gastrinoma decisions."
                ),
                "clinical_rule": "MEN1 pHPT: 4-gland surgery (subtotal) + cervical thymectomy; sestamibi NOT reliable; annual Ca/PTH lifelong; annual screen all first-degree from age 8",
            },
            {
                "term": "Parathyroid Carcinoma (CDC73) — En-Bloc Resection and Parafibromin IHC",
                "short": "CDC73 LOF → HPT-JT; parathyroid carcinoma 15-20%; parafibromin IHC absent = carcinoma; en-bloc resection mandatory; jaw ossifying fibroma DDx from osteitis fibrosa",
                "detail": (
                    "Parathyroid carcinoma is rare (<1% of all pHPT) but highly overrepresented in CDC73 germline LOF carriers "
                    "(15-20% lifetime risk). "
                    "CLINICAL CLUES to carcinoma: Very high Ca²⁺ (>3.2 mmol/L); very high PTH (>5x ULN); "
                    "palpable neck mass; recurrent laryngeal nerve palsy; adherent gland intraoperatively. "
                    "PARAFIBROMIN IHC: The most sensitive and specific marker. "
                    "Normal parathyroid → strong nuclear parafibromin staining; "
                    "Parathyroid carcinoma → ABSENT nuclear parafibromin staining (CDC73 biallelic loss → no parafibromin protein); "
                    ">95% sensitivity + specificity for carcinoma. "
                    "Always request parafibromin IHC on ANY parathyroid tumour specimen — not just suspected carcinoma. "
                    "SURGICAL APPROACH: En-bloc resection = parathyroid tumour + ipsilateral thyroid lobe + isthmus; "
                    "CRITICAL: Do NOT violate capsule — capsule violation almost guarantees local recurrence. "
                    "RECURRENCE: Local recurrence and distant metastases (lung, bone, liver) are common; "
                    "cinacalcet palliates hypercalcaemia; denosumab for bone mets. "
                    "JAW TUMOUR DISTINCTION: HPT-JT jaw tumour = ossifying fibroma (whorled fibroblasts + calcification); "
                    "Osteitis fibrosa cystica (brown tumour) = complication of severe pHPT (any cause) — "
                    "these are histologically and radiologically distinct; jaw X-ray mandatory in HPT-JT families. "
                    "ALL SPORADIC PARATHYROID CARCINOMA: Germline CDC73 testing mandatory — ~70% carry somatic CDC73 biallelic LOF "
                    "but ~15-20% carry germline (may be index case of unsuspected HPT-JT family)."
                ),
                "clinical_rule": "Parathyroid carcinoma: parafibromin IHC on every specimen; en-bloc resection; no capsule rupture; CDC73 germline test ALL parathyroid carcinoma cases",
            },
            {
                "term": "Hypoparathyroidism — Magnesium First Rule",
                "short": "Low Ca + Low PTH + High PO4 = hypoparathyroidism (GCM2/PTH); ALWAYS check and correct Mg²⁺ FIRST; Ca target 2.0-2.1 mmol/L only; renal USS annually",
                "detail": (
                    "Hypoparathyroidism (low PTH → low Ca²⁺ + high PO4) arises from GCM2 LOF (parathyroid aplasia), "
                    "PTH gene variants (defective preproPTH), and acquired causes (autoimmune, post-surgical, infiltrative). "
                    "THE MAGNESIUM RULE: "
                    "MAGNESIUM DEFICIENCY causes BOTH impaired PTH secretion AND peripheral PTH resistance. "
                    "(1) HypoMg → blocks Ca²⁺-sensing by parathyroid chief cells → reduces PTH release; "
                    "(2) HypoMg → impairs adenylyl cyclase coupling to PTH1R → PTH cannot signal effectively. "
                    "CONSEQUENCE: HypoMg-induced hypoparathyroidism will NOT RESPOND to Ca + calcitriol "
                    "until Mg is corrected — the hypocalcaemia is refractory until Mg replenished. "
                    "CHECK MG IN EVERY HYPOPARATHYROIDISM PATIENT: "
                    "Serum Mg normal range 0.7-1.0 mmol/L; correct to >0.8 mmol/L. "
                    "TREATMENT TARGET: "
                    "TARGET Ca 2.0-2.1 mmol/L (low-NORMAL), NOT 2.2-2.4 mmol/L (normal); "
                    "REASON: Despite low PTH, calcitriol still drives intestinal Ca absorption AND "
                    "calcitriol + Ca²⁺ → calciuria → nephrocalcinosis even at 'therapeutic' Ca levels; "
                    "targeting high-normal Ca → markedly increased nephrocalcinosis risk. "
                    "ANNUAL RENAL USS + SPOT URINE Ca/Cr RATIO: mandatory to detect early nephrocalcinosis. "
                    "THIAZIDE DIURETICS (hydrochlorothiazide 12.5-25 mg): Reduce calciuria → allow lower Ca/calcitriol dose → "
                    "reduce nephrocalcinosis risk — useful in chronic hypoparathyroidism. "
                    "RECOMBINANT PTH: teriparatide (PTH1-34) or PTH1-84 (Natpara) → "
                    "restores PTH physiology → reduces calciuria compared to calcitriol alone."
                ),
                "clinical_rule": "Hypoparathyroidism: CHECK Mg FIRST (every time); Ca target 2.0-2.1 mmol/L only; annual renal USS; thiazide reduces calciuria; rPTH for brittle cases",
            },
            {
                "term": "ADH (Autosomal Dominant Hypocalcaemia) — Nephrocalcinosis Risk with Treatment",
                "short": "ADH1 (CASR GOF) / ADH2 (GNA11 GOF): low Ca + LOW PTH; goal Ca 2.0-2.1 mmol/L; calcitriol + excess Ca → nephrocalcinosis (calciuria continues despite low serum Ca)",
                "detail": (
                    "ADH1 (CASR GOF) and ADH2 (GNA11 GOF) cause hypocalcaemia with paradoxically suppressed PTH — "
                    "unlike most hypocalcaemic causes where PTH appropriately rises. "
                    "MECHANISM: GOF → CASR/Gα11 hypersensitive → normal Ca²⁺ suppresses PTH → "
                    "low PTH → low Ca²⁺ + HIGH calciuria (CASR GOF in kidney → excessive Ca wasting). "
                    "TREATMENT PARADOX: "
                    "Standard treatment for hypoparathyroidism (calcitriol + Ca supplements) → "
                    "raises serum Ca → further CASR activation in kidney → EVEN MORE calciuria → "
                    "nephrocalcinosis develops rapidly even at Ca levels that are low-normal. "
                    "This means the treatment goal is to bring Ca to 2.0-2.1 mmol/L ONLY — "
                    "do NOT target normocalcaemia (2.2-2.6 mmol/L) in ADH. "
                    "MONITORING: Annual renal USS + 24-hour urine Ca (or spot urine Ca/Cr) — "
                    "24-hr urinary Ca should be <5-7.5 mmol/day; higher → reduce calcitriol dose. "
                    "EMERGING TREATMENT: Calcilytics (CASR antagonists, e.g. NPSP558/encaleret) — "
                    "specifically antagonise the hyperactive CASR → restore PTH secretion → "
                    "more physiological Ca handling → less calciuria; in clinical trials for ADH. "
                    "THIAZIDES: Can reduce calciuria in ADH — useful to add alongside minimal Ca/calcitriol."
                ),
                "clinical_rule": "ADH1/2: Ca target 2.0-2.1 mmol/L ONLY; monitor 24-hr urinary Ca; annual renal USS; calcilytics emerging for ADH1; thiazides reduce calciuria",
            },
            {
                "term": "Pseudohypoparathyroidism (PHP) vs True Hypoparathyroidism — PTH Level is Key",
                "short": "PHP: high PTH + low Ca (PTH resistance via GNAS LOF); True hypoPTH (GCM2/PTH): low/undetectable PTH + low Ca; Albright osteodystrophy in PHP type 1a",
                "detail": (
                    "A critical diagnostic distinction in hypocalcaemia is whether PTH is LOW or HIGH: "
                    "TRUE HYPOPARATHYROIDISM (GCM2 LOF, PTH gene variants, post-surgical, autoimmune): "
                    "LOW or UNDETECTABLE PTH + LOW Ca²⁺ + HIGH PO4. "
                    "PTH secretion is deficient — parathyroid glands are absent, aplastic, or destroyed. "
                    "PSEUDOHYPOPARATHYROIDISM (PHP) — GNAS LOF: "
                    "HIGH PTH + LOW Ca²⁺ + HIGH PO4. "
                    "PTH is produced in excess (appropriate compensatory response to low Ca) — "
                    "but end-organs (kidney tubule, bone) cannot respond to PTH (PTH1R/GNAS signalling defect). "
                    "PHP1a: GNAS maternal LOF; Albright hereditary osteodystrophy (AHO) features: "
                    "short stature, round face, short 4th/5th metacarpals, obesity, cognitive impairment. "
                    "PHP1b: GNAS imprinting defect; no AHO features; isolated PTH resistance. "
                    "TREATMENT DIFFERENCE: Both true hypoPTH and PHP treated with Ca + calcitriol; "
                    "but in PHP, recombinant PTH is NOT needed (PTH already elevated); "
                    "in PHP1a, GH deficiency and hypothyroidism must be screened (multi-hormone resistance). "
                    "CLINICAL PEARL: Chvostek and Trousseau signs are positive in BOTH — "
                    "only PTH measurement distinguishes them at bedside."
                ),
                "clinical_rule": "Hypocalcaemia: PTH undetectable → true hypoparathyroidism (GCM2/PTH/surgical/autoimmune); PTH high → PHP (GNAS) or vitamin D deficiency",
            },
            {
                "term": "Cascade Testing — Hereditary Parathyroid Disorders",
                "short": "FHH (AD LOF): 50% risk; no surgery needed but CCCR <0.01 must be documented. MEN1/CDC73/CDKN1B (AD): 50% risk; annual surveillance from age 8. GCM2/PTH hypoparathyroidism: 25-50% risk depending on AR/AD",
                "detail": (
                    "Cascade genetic testing in hereditary parathyroid disorders: "
                    "FHH (CASR/GNA11/AP2S1 — AD LOF): 50% risk per child; "
                    "test parents and siblings; "
                    "PURPOSE: identify carriers BEFORE inadvertent parathyroidectomy; "
                    "document CCCR <0.01 in carrier → no surgery EVER warranted; "
                    "management: same for all carriers (no surgery, no thiazides, observe). "
                    "MEN1 (AD LOF): 50% risk per child; "
                    "ALL first-degree relatives must be offered testing; "
                    "positive carrier → annual Ca/PTH/gastrin/glucose/prolactin/IGF-1 from age 8; "
                    "MRI pituitary every 3-5 years; pancreas MRI every 1-2 years from age ~20. "
                    "CDC73 (AD LOF): 50% risk; "
                    "cascade testing critical — parathyroid carcinoma in undetected carriers; "
                    "jaw X-ray mandatory in all family members; renal USS; Ca/PTH annually. "
                    "CDKN1B MEN4 (AD LOF): same as MEN1 cascade protocol once identified. "
                    "GCM2 FIH (AR): 25% sib risk; carrier parents asymptomatic (AD form: 50%); "
                    "PTH gene variants (AR/AD): same sib risk calculation. "
                    "PRACTICAL PRIORITY: CDC73 cascade most urgent (carcinoma risk); "
                    "MEN1 second (annual surveillance changes outcomes); "
                    "FHH cascade: prevention of surgery is the primary goal."
                ),
                "clinical_rule": "Hereditary pHPT: CASCADE TEST ALL FIRST-DEGREE RELATIVES immediately. FHH: prevent surgery. MEN1/CDC73: annual surveillance saves lives. GCM2/PTH: treat and educate family",
            },
        ],
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    bd = get_breakdown()
    df = get_definitions()
    print(f"Overview: atlas={ov['atlas_name']}, n={ov['n_patients']}, genes={ov['gene_count']}")
    print(f"Breakdown: {len(bd)} genes")
    print(f"Definitions: {len(df['definitions'])} terms")
    for g in bd:
        print(f"  {g['gene']}: {g['cohort_stats']['n']} patients, seed {g['cohort_stats']['seed']}, "
              f"special={g['cohort_stats']['special_features']}")
