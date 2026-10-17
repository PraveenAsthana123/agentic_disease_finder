#!/usr/bin/env python3
"""Hereditary-Cerebrovascular-Disease-Atlas -- Complete 8-Gene Reference
Hereditary cerebrovascular diseases (CADASIL, COL4A1/COL4A2 angiopathy,
Moyamoya ACTA2, CARASIL HTRA1, AGS TREX1, HCHWA-I CST3, HCHWA-D APP):

NOTCH3 (CADASIL; 2321aa; 19p13.2; AD GOF cysteine-altering;
         Most-common hereditary stroke; GOM-deposits skin-biopsy PATHOGNOMONIC;
         Lacunar infarcts 40-50yr; migraine-with-aura onset 30yr; dementia;
         seed SEED_BASE+0).
COL4A1 (Type-IV collagen alpha-1; 1454aa; 13q34; AD LOF dominant-negative;
         Porencephaly PATHOGNOMONIC; ICH young adults; renal cysts; cataracts;
         AVOID-anticoagulation + AVOID-excessive-exertion;
         seed SEED_BASE+1).
COL4A2 (Type-IV collagen alpha-2; 1712aa; 13q34; AD LOF;
         Same-locus-COL4A1; ICH-dominant; less-porencephaly;
         COL4A1+COL4A2-genotype-panel-together;
         seed SEED_BASE+2).
ACTA2  (alpha-SMA; 377aa; 10q23.31; AD GOF;
         Moyamoya-PATHOGNOMONIC + iris-flocculi + livedo-reticularis TRIAD;
         Cerebral-MRA-MANDATORY; TAAD-overlap;
         seed SEED_BASE+3).
HTRA1  (HtrA serine peptidase 1; 480aa; 10q26.13; AR biallelic LOF;
         CARASIL; alopecia PATHOGNOMONIC; spondylosis-before-stroke;
         Japanese/Asian founder; AD monoallelic = CADASIL-like-milder;
         seed SEED_BASE+4).
TREX1  (3-prime repair exonuclease 1; 314aa; 3p21.31; AR/AD LOF;
         Aicardi-Goutieres-syndrome-type-1; cerebral-vasculopathy;
         JAK-inhibitor-baricitinib-therapy; interferon-signature-elevated;
         seed SEED_BASE+5).
CST3   (Cystatin-C; 146aa; 20p11.21; AD GOF amyloidogenic;
         HCHWA-I Icelandic; L68Q founder; cerebral-amyloid-angiopathy;
         ANTICOAGULATION-ABSOLUTELY-CONTRAINDICATED; ANTIPLATELET-AVOID;
         seed SEED_BASE+6).
APP    (Amyloid precursor protein; 770aa; 21q21.3; AD GOF;
         HCHWA-D Dutch-A692G + Iowa-D694N; cerebral-amyloid-angiopathy;
         ANTICOAGULATION-ABSOLUTELY-CONTRAINDICATED; early-Alzheimer-overlap;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3774-3781)
"""
import random

SEED_BASE = 3774

ATLAS_GENES = [
    {
        "gene": "NOTCH3",
        "protein": (
            "NOTCH3 -- 19p13.2 AD GOF -- 2321aa -- "
            "CADASIL-EGF-repeat-cysteine-mutation-arteriolar-smooth-muscle-"
            "GOM-deposits-skin-biopsy-PATHOGNOMONIC-lacunar-infarcts-40-50yr-"
            "migraine-aura-30yr-dementia-NOTCH3-OMIM-125310"
        ),
        "locus": "19p13.2",
        "protein_size": (
            "2321 aa / ~280 kDa / 19p13.2 NOTCH3 transmembrane receptor molecular context: "
            "STRUCTURE: "
            "  2321 aa / ~280 kDa; type I transmembrane receptor; "
            "  34 EGF-like repeat domains (N-terminal extracellular); "
            "  3 LNR (Lin/Notch Repeats) + transmembrane + RAM + ANK + PEST (intracellular); "
            "  Pathogenic variants = cysteine-altering missense in EGF repeats 1-34; "
            "  Odd-numbered cysteines lost/odd cysteine added -> conformational change -> "
            "  extracellular domain accumulates as GOM deposits in arteriolar walls; "
            "CADASIL (Cerebral Autosomal Dominant Arteriopathy with Subcortical Infarcts and Leukoencephalopathy): "
            "  Most common hereditary stroke syndrome; AD GOF; >500 pathogenic variants; "
            "  95% are cysteine-altering substitutions in EGF-like repeats; "
            "  GOM (granular osmiophilic material) deposits in arteriolar BM = PATHOGNOMONIC; "
            "  Skin biopsy electron microscopy detects GOM = diagnostic; "
            "  MRI: periventricular + anterior temporal pole WM lesions (PATHOGNOMONIC territory); "
            "  Anterior temporal pole WM PATHOGNOMONIC (CADASIL vs. other SVD); "
            "CLINICAL TIMELINE: "
            "  Age 20-30yr: migraine with aura (onset symptom, 35-75%); "
            "  Age 40-50yr: lacunar stroke/TIA (most common; recurrent); "
            "  Age 50-60yr: cognitive decline -> vascular dementia; "
            "  Age 60-70yr: severe disability; walking/swallowing impairment; "
            "TREATMENT: "
            "  No disease-modifying therapy approved 2026; "
            "  Antiplatelet (aspirin 100mg): standard post-stroke; "
            "  Anticoagulation: AVOID unless proven cardioembolic source (AF) -- ICH risk ELEVATED; "
            "  Migraine: triptans OK (no specific CI in CADASIL); beta-blocker prophylaxis; "
            "  BP control: <130/80; "
            "  Statins: for CV risk reduction (no CADASIL-specific benefit proven); "
            "PATHOGNOMONIC FEATURES: "
            "  GOM deposits on skin biopsy EM -- CONFIRMATORY; "
            "  Anterior temporal pole WM hyperintensities on MRI; "
            "  External capsule involvement on MRI; "
            "CASCADE TESTING: all 1st-degree relatives; "
            "ANTICOAGULATION-WARNING: AVOID in CADASIL -- ICH rate elevated even without AF"
        ),
        "inheritance": "AD GOF (cysteine-altering EGF repeat mutations)",
        "syndrome": "CADASIL / Most-Common-Hereditary-Stroke / Lacunar-Infarcts-40-50yr / Migraine-Aura-30yr / GOM-PATHOGNOMONIC",
        "key_drugs_ci": [
            "Anticoagulation-AVOID-ICH-Risk (unless proven AF cardioembolic)",
            "No-disease-modifying-therapy-approved-2026",
        ],
        "pathognomonic": "GOM deposits on skin biopsy EM + anterior temporal pole WM on MRI",
        "mean_age_dx": 42,
        "pct_female": 48,
        "stroke_n_per40": 31,
        "mri_lesion_n_per40": 40,
        "dementia_n_per40": 22,
        "anticoag_contraindicated": True,
    },
    {
        "gene": "COL4A1",
        "protein": (
            "COL4A1 -- 13q34 AD LOF dominant-negative -- 1454aa -- "
            "type-IV-collagen-alpha1-chain-basement-membrane-structural-"
            "porencephaly-PATHOGNOMONIC-ICH-young-renal-cysts-cataracts-"
            "AVOID-anticoagulation-AVOID-excessive-exertion-COL4A1-OMIM-120130"
        ),
        "locus": "13q34",
        "protein_size": (
            "1454 aa / ~160 kDa / 13q34 COL4A1 type-IV-collagen molecular context: "
            "STRUCTURE: "
            "  1454 aa / ~160 kDa; structural component of basement membranes; "
            "  N-terminal 7S domain, central collagenous triple-helix (Gly-X-Y repeats), "
            "  C-terminal NC1 domain; "
            "  COL4A1 + COL4A2 form obligate heterotrimers -> BM structural integrity; "
            "  Pathogenic missense in Gly-X-Y -> misfolding -> BM weakness -> hemorrhage; "
            "COL4A1 ANGIOPATHY: "
            "  Porencephaly: fluid-filled cavity in cerebral hemisphere = PATHOGNOMONIC; "
            "  Often detected in childhood (seizures) or prenatal; "
            "  ICH in young adults (20-50yr): lobar pattern; "
            "  Periventricular leukoencephalopathy; "
            "  Lacunar infarcts (SVD pattern); "
            "SYSTEMIC FEATURES: "
            "  Renal cysts (30-40%): monitor renal function; "
            "  Cataracts (20-30%): congenital or early-onset; "
            "  Retinal arteriolar tortuosity: fundoscopy clue; "
            "  Muscle cramps/elevated CK (rare); "
            "  Raynaud phenomenon (rare); "
            "TREATMENT / MANAGEMENT: "
            "  AVOID anticoagulation -- ICH risk; warfarin/DOAC CONTRAINDICATED unless absolute necessity; "
            "  AVOID excessive physical exertion / contact sports -- hemorrhagic risk; "
            "  Antiplatelets: use with caution (may increase ICH); "
            "  BP tight control: <130/80; "
            "  Seizure management: standard AEDs; "
            "  Renal monitoring annually; "
            "CASCADE TESTING: all 1st-degree relatives; "
            "PATHOGNOMONIC: porencephaly on MRI (fluid cavity communicating with ventricle)"
        ),
        "inheritance": "AD LOF dominant-negative (Gly-X-Y missense)",
        "syndrome": "COL4A1-Angiopathy / Porencephaly-PATHOGNOMONIC / ICH-Young / Renal-Cysts / Cataracts",
        "key_drugs_ci": [
            "Anticoagulation-CONTRAINDICATED-ICH",
            "Contact-sports-AVOID",
            "Excessive-exertion-AVOID",
        ],
        "pathognomonic": "Porencephaly on MRI (fluid-filled cerebral cavity) + ICH young adult",
        "mean_age_dx": 35,
        "pct_female": 50,
        "stroke_n_per40": 26,
        "mri_lesion_n_per40": 38,
        "dementia_n_per40": 9,
        "anticoag_contraindicated": True,
    },
    {
        "gene": "COL4A2",
        "protein": (
            "COL4A2 -- 13q34 AD LOF -- 1712aa -- "
            "type-IV-collagen-alpha2-chain-obligate-COL4A1-heterotrimer-"
            "ICH-dominant-less-porencephaly-than-COL4A1-same-locus-panel-together-"
            "COL4A2-OMIM-120090"
        ),
        "locus": "13q34",
        "protein_size": (
            "1712 aa / ~167 kDa / 13q34 COL4A2 type-IV-collagen molecular context: "
            "STRUCTURE: "
            "  1712 aa / ~167 kDa; obligate heterotrimer partner with COL4A1; "
            "  Same 7S-collagenous-NC1 domain architecture; "
            "  COL4A1+COL4A2 genes at 13q34 in tandem head-to-head arrangement; "
            "  Variants in either gene disrupt the shared heterotrimer; "
            "COL4A2 ANGIOPATHY vs COL4A1: "
            "  ICH-dominant phenotype: intracerebral hemorrhage main presentation; "
            "  Less porencephaly than COL4A1; "
            "  Periventricular leukoencephalopathy similar; "
            "  Renal + cataracts: less common than COL4A1; "
            "  Retinal arteriolar disease: similar; "
            "DIAGNOSIS: "
            "  COL4A1 + COL4A2 gene panel must be tested together (same locus); "
            "  WES/NGS often misses large deletions -- MLPA/CNV required; "
            "  MRI: deep ICH in young + leukoencephalopathy -> COL4A1/COL4A2 panel; "
            "MANAGEMENT (same as COL4A1): "
            "  AVOID anticoagulation; "
            "  AVOID contact sports; "
            "  Tight BP control; "
            "  CASCADE TESTING all 1st-degree relatives; "
        ),
        "inheritance": "AD LOF (Gly-X-Y missense; same locus as COL4A1)",
        "syndrome": "COL4A2-Angiopathy / ICH-Dominant / Less-Porencephaly / Panel-COL4A1-COL4A2-Together",
        "key_drugs_ci": [
            "Anticoagulation-CONTRAINDICATED-ICH",
            "Contact-sports-AVOID",
        ],
        "pathognomonic": "ICH young adult + leukoencephalopathy; COL4A1+COL4A2 panel together",
        "mean_age_dx": 40,
        "pct_female": 51,
        "stroke_n_per40": 28,
        "mri_lesion_n_per40": 36,
        "dementia_n_per40": 10,
        "anticoag_contraindicated": True,
    },
    {
        "gene": "ACTA2",
        "protein": (
            "ACTA2 -- 10q23.31 AD GOF -- 377aa -- "
            "alpha-smooth-muscle-actin-TAAD2-Moyamoya-PATHOGNOMONIC-"
            "iris-flocculi-livedo-reticularis-cerebral-MRA-MANDATORY-"
            "ACTA2-OMIM-102620"
        ),
        "locus": "10q23.31",
        "protein_size": (
            "377 aa / ~42 kDa / 10q23.31 ACTA2 alpha-SMA molecular context: "
            "STRUCTURE: "
            "  377 aa / ~42 kDa; alpha-smooth muscle actin; "
            "  Cytoskeletal protein in smooth muscle cells; "
            "  R179H/R179C most common cerebrovascular mutations (most severe); "
            "  R258C/L96P/other: TAAD-dominant, cerebrovascular less severe; "
            "PATHOGNOMONIC TRIAD (R179H/R179C): "
            "  Moyamoya disease (cerebral artery stenosis/occlusion + puff-of-smoke collaterals); "
            "  Iris flocculi (white patches on posterior iris pigment epithelium): PATHOGNOMONIC; "
            "  Livedo reticularis (mottled skin pattern); "
            "CEREBROVASCULAR FEATURES: "
            "  Moyamoya: bilateral internal carotid occlusive disease; "
            "  Stroke + TIA in young patients; "
            "  Cerebral MRA MANDATORY at diagnosis + every 2yr; "
            "  Bypass surgery (EC-IC) for symptomatic Moyamoya; "
            "OVERLAP: "
            "  TAAD (ascending aortic aneurysm): also present (see TAAD atlas); "
            "  PDA (patent ductus arteriosus): TAAD4 gene in cardiac series; "
            "CASCADE TESTING: cerebral MRA mandatory in all 1st-degree relatives"
        ),
        "inheritance": "AD GOF (smooth muscle contraction pathway hyperactivation)",
        "syndrome": "ACTA2-Moyamoya-PATHOGNOMONIC / Iris-Flocculi-Livedo-Reticularis-TRIAD / TAAD-Overlap / Cerebral-MRA-MANDATORY",
        "key_drugs_ci": [
            "Cerebral-MRA-MANDATORY-diagnosis+2yr",
            "EC-IC-bypass-symptomatic-Moyamoya",
        ],
        "pathognomonic": "Moyamoya + iris flocculi + livedo reticularis PATHOGNOMONIC triad (R179H/R179C)",
        "mean_age_dx": 28,
        "pct_female": 52,
        "stroke_n_per40": 24,
        "mri_lesion_n_per40": 35,
        "dementia_n_per40": 5,
        "anticoag_contraindicated": False,
    },
    {
        "gene": "HTRA1",
        "protein": (
            "HTRA1 -- 10q26.13 AR biallelic LOF -- 480aa -- "
            "HtrA-serine-peptidase-1-CARASIL-alopecia-PATHOGNOMONIC-"
            "spondylosis-before-stroke-Japanese-Asian-founder-"
            "AD-monoallelic-CADASIL-like-milder-HTRA1-OMIM-602194"
        ),
        "locus": "10q26.13",
        "protein_size": (
            "480 aa / ~51 kDa / 10q26.13 HTRA1 serine protease molecular context: "
            "STRUCTURE: "
            "  480 aa / ~51 kDa; secreted serine protease; "
            "  N-terminal IGFBP domain + Kazal-type inhibitor domain + trypsin-like protease domain + PDZ domain; "
            "  Degrades TGF-beta family members -> regulates BM remodeling; "
            "  Biallelic LOF -> accumulation TGF-beta -> SVD; "
            "CARASIL (Cerebral Autosomal Recessive Arteriopathy with Subcortical Infarcts and Leukoencephalopathy): "
            "  AR biallelic LOF (homozygous or compound heterozygous); "
            "  Japanese/Asian enriched (Yamamoto R302H founder); European cases rare; "
            "PATHOGNOMONIC FEATURES: "
            "  Alopecia: hair loss onset 20-30yr, before stroke PATHOGNOMONIC (unique among SVD); "
            "  Spondylosis: severe early-onset lumbar/cervical spondylosis + disc herniation; "
            "  Stroke: lacunar infarcts 30-40yr; "
            "  Cognitive decline + dementia follows stroke; "
            "CLINICAL SEQUENCE (CARASIL): Alopecia -> Spondylosis -> Stroke -> Dementia; "
            "AD MONOALLELIC HTRA1: "
            "  AD HTRA1 (heterozygous loss-of-function): "
            "  CADASIL-like phenotype but milder; less severe alopecia; "
            "  Penetrance ~60-70% by age 70yr; "
            "TREATMENT: no disease-modifying therapy; "
            "  Antiplatelet post-stroke; "
            "  BP control; "
            "  Physical therapy for spondylosis; "
            "CASCADE TESTING: AR = both parents carriers; test siblings; "
            "DIAGNOSIS CLUE: young adult + alopecia + lumbar spondylosis + stroke = HTRA1 immediately"
        ),
        "inheritance": "AR biallelic LOF (CARASIL) / AD monoallelic LOF (CADASIL-like milder)",
        "syndrome": "CARASIL / Alopecia-PATHOGNOMONIC / Spondylosis-Before-Stroke / Japanese-Asian-Founder / AD-Monoallelic-Milder",
        "key_drugs_ci": [
            "No-disease-modifying-therapy-CARASIL",
            "Alopecia-Spondylosis-Stroke-sequence-PATHOGNOMONIC",
        ],
        "pathognomonic": "Alopecia onset 20-30yr BEFORE stroke + early severe spondylosis (unique among hereditary SVD)",
        "mean_age_dx": 33,
        "pct_female": 47,
        "stroke_n_per40": 29,
        "mri_lesion_n_per40": 38,
        "dementia_n_per40": 18,
        "anticoag_contraindicated": False,
    },
    {
        "gene": "TREX1",
        "protein": (
            "TREX1 -- 3p21.31 AR/AD LOF -- 314aa -- "
            "3-prime-repair-exonuclease-1-Aicardi-Goutieres-syndrome-type-1-"
            "cerebral-vasculopathy-interferon-signature-elevated-PATHOGNOMONIC-"
            "JAK-inhibitor-baricitinib-therapy-TREX1-OMIM-606609"
        ),
        "locus": "3p21.31",
        "protein_size": (
            "314 aa / ~33 kDa / 3p21.31 TREX1 exonuclease molecular context: "
            "STRUCTURE: "
            "  314 aa / ~33 kDa; major mammalian 3' to 5' DNA exonuclease; "
            "  N-terminal DEDDh exonuclease domain + C-terminal ER-tethering domain; "
            "  Degrades cytosolic single-stranded DNA from endogenous retroelements; "
            "  LOF -> cytosolic ssDNA accumulates -> cGAS-STING activation -> type-I IFN surge; "
            "AICARDI-GOUTIERES SYNDROME TYPE 1 (AGS1): "
            "  AR biallelic LOF: classic severe infantile encephalopathy; "
            "  Pseudo-TORCH: calcifications (basal ganglia) + leukoencephalopathy + CSF lymphocytosis; "
            "  Interferon signature elevated in blood/CSF PATHOGNOMONIC; "
            "  Cerebral vasculopathy: small vessel disease; "
            "  Chilblain lesions: acral erythematous plaques (cold-induced); "
            "AD TREX1 LOF (heterozygous): "
            "  Familial chilblain lupus: mild cutaneous SLE-like; "
            "  Retinal vasculopathy with cerebral leukoencephalopathy (RVCL): progressive; "
            "  RVCL: progressive vision loss + stroke + psychiatric + dementia; "
            "TREATMENT: "
            "  JAK inhibitor baricitinib: FDA-approved for AGS 2023 (pediatric/adult); "
            "  Reduces interferon-stimulated gene expression; "
            "  IVIG + corticosteroids: adjunct; "
            "  Reverse transcriptase inhibitors (antiretroviral): investigational; "
            "DIAGNOSIS: "
            "  Interferon score (ISG15/IFIT1/IFIT2/IFIT3/RSAD2/SIGLEC1): elevated; "
            "  CSF: lymphocytic pleocytosis + elevated IFN-alpha; "
            "  NGS: TREX1 biallelic variants; "
            "  Brain MRI: bilateral symmetrical calcifications + WM disease; "
        ),
        "inheritance": "AR biallelic LOF (AGS1) / AD LOF (chilblain lupus / RVCL)",
        "syndrome": "AGS1-Cerebral-Vasculopathy / Interferon-Signature-PATHOGNOMONIC / Baricitinib-JAKi-FDA2023 / RVCL-AD",
        "key_drugs_ci": [
            "Baricitinib-JAKi-FDA2023-AGS-treatment",
            "Interferon-score-DIAGNOSTIC-PATHOGNOMONIC",
        ],
        "pathognomonic": "Elevated interferon signature + basal ganglia calcifications + WM disease",
        "mean_age_dx": 6,
        "pct_female": 49,
        "stroke_n_per40": 18,
        "mri_lesion_n_per40": 40,
        "dementia_n_per40": 24,
        "anticoag_contraindicated": False,
    },
    {
        "gene": "CST3",
        "protein": (
            "CST3 -- 20p11.21 AD GOF amyloidogenic -- 146aa -- "
            "Cystatin-C-HCHWA-I-Icelandic-L68Q-founder-"
            "cerebral-amyloid-angiopathy-hemorrhage-20-30yr-"
            "ANTICOAGULATION-ABSOLUTELY-CONTRAINDICATED-ANTIPLATELET-AVOID-CST3-OMIM-604312"
        ),
        "locus": "20p11.21",
        "protein_size": (
            "146 aa / ~13 kDa / 20p11.21 CST3 cystatin-C molecular context: "
            "STRUCTURE: "
            "  146 aa / ~13 kDa; secreted cysteine protease inhibitor; "
            "  Ubiquitously expressed, highest in brain/CSF; "
            "  Signal peptide (26aa) + mature protein (120aa); "
            "  L68Q variant: single substitution -> misfolding -> amyloid fibrils in arteriolar walls; "
            "  Dimerization via 3D domain swapping -> accelerated amyloid formation; "
            "HCHWA-I (Hereditary Cerebral Hemorrhage with Amyloidosis - Icelandic): "
            "  Autosomal dominant GOF amyloidogenic variant; "
            "  L68Q founder mutation: highly enriched in Iceland (carrier freq ~1/25 in affected families); "
            "  Cerebral amyloid angiopathy (CAA): amyloid deposits in leptomeningeal + cortical arterioles; "
            "  Recurrent lobar ICH onset 20-30yr (VERY YOUNG -- DDx essential); "
            "  Death within 5yr of first hemorrhage typical; "
            "TREATMENT (none disease-modifying): "
            "  ANTICOAGULATION: ABSOLUTELY CONTRAINDICATED -- amyloid angiopathy ICH risk massive; "
            "  ANTIPLATELETS: AVOID -- ICH risk; "
            "  BP control strict <120/80; "
            "  No surgery beyond evacuation of life-threatening hematoma; "
            "CSF CYSTATIN-C: reduced (amyloid deposits sequester it); diagnostic clue; "
            "DIFFERENTIAL vs HCHWA-D (APP): both CAA-hemorrhage -- test CST3 + APP gene panel; "
            "PATHOGNOMONIC: "
            "  Recurrent lobar ICH onset 20-30yr in Icelandic family; "
            "  CAA on MRI/SWI (cortical superficial siderosis); "
            "CASCADE TESTING: AD -- 50% risk; all 1st-degree relatives"
        ),
        "inheritance": "AD GOF amyloidogenic (L68Q cystatin-C amyloid)",
        "syndrome": "HCHWA-I-Icelandic / Recurrent-Lobar-ICH-20-30yr / CAA / L68Q-Founder / ANTICOAGULATION-ABSOLUTELY-CI",
        "key_drugs_ci": [
            "Anticoagulation-ABSOLUTELY-CONTRAINDICATED",
            "Antiplatelets-AVOID",
            "BP-strict-120/80-maximum",
        ],
        "pathognomonic": "Recurrent lobar ICH onset 20-30yr + Icelandic family + CSF cystatin-C low",
        "mean_age_dx": 26,
        "pct_female": 50,
        "stroke_n_per40": 40,
        "mri_lesion_n_per40": 40,
        "dementia_n_per40": 28,
        "anticoag_contraindicated": True,
    },
    {
        "gene": "APP",
        "protein": (
            "APP -- 21q21.3 AD GOF -- 770aa -- "
            "amyloid-precursor-protein-HCHWA-D-Dutch-A692G-Iowa-D694N-"
            "cerebral-amyloid-angiopathy-ICH-early-Alzheimer-"
            "ANTICOAGULATION-ABSOLUTELY-CONTRAINDICATED-APP-OMIM-104760"
        ),
        "locus": "21q21.3",
        "protein_size": (
            "770 aa / ~87 kDa / 21q21.3 APP amyloid precursor protein molecular context: "
            "STRUCTURE: "
            "  770 aa / ~87 kDa; type I transmembrane glycoprotein; "
            "  Extracellular E1 + E2 + Abeta domain + transmembrane + C-terminal; "
            "  Cleaved by alpha/beta/gamma secretases -> Abeta fragments; "
            "  HCHWA-D variant A692G (Dutch): within Abeta domain; "
            "  Abeta A21G (Dutch numbering) -> Abeta aggregation in vessel walls (not plaques); "
            "  Iowa variant D694N: similar cerebrovascular amyloid; "
            "HCHWA-D (Hereditary Cerebral Hemorrhage with Amyloidosis - Dutch): "
            "  AD GOF amyloidogenic; "
            "  Cerebral amyloid angiopathy: Abeta accumulates in leptomeningeal + cortical vessels; "
            "  Recurrent lobar ICH: onset 40-60yr (older than CST3/HCHWA-I); "
            "  Dementia: progressive Alzheimer-like; "
            "  Unlike classic AD: hemorrhage predominates over plaque burden; "
            "IOWA VARIANT: "
            "  APP D694N (Asp694Asn): similar CAA + ICH + dementia phenotype; "
            "  Hemorrhagic > amnestic presentation; "
            "ANTICOAGULATION ABSOLUTELY CONTRAINDICATED: "
            "  CAA with hemorrhagic phenotype = anticoagulants massively increase ICH risk; "
            "  If AF coexists: discuss risks explicitly -- NOACs may still be lower risk than warfarin; "
            "  MRI SWI: microhemorrhages + cortical superficial siderosis; "
            "CST3 vs APP: "
            "  CST3 HCHWA-I: ICH onset 20-30yr (younger) -- Icelandic; "
            "  APP HCHWA-D: ICH onset 40-60yr (older) -- Dutch/Iowa; "
            "  Test both genes when hereditary CAA suspected; "
            "RELATIONSHIP TO AD: "
            "  APP duplication (trisomy 21/Down syndrome): early-onset AD; "
            "  APP KM670/671NL (Swedish): AD not hemorrhagic CAA; "
            "  Dutch/Iowa variants specifically target vascular amyloid; "
            "CASCADE TESTING: AD -- 50% risk"
        ),
        "inheritance": "AD GOF amyloidogenic (Abeta vascular deposition; A692G Dutch / D694N Iowa)",
        "syndrome": "HCHWA-D-Dutch / HCHWA-Iowa / CAA-Hemorrhagic / Early-Alzheimer-Overlap / ANTICOAGULATION-ABSOLUTELY-CI",
        "key_drugs_ci": [
            "Anticoagulation-ABSOLUTELY-CONTRAINDICATED",
            "Warfarin-higher-ICH-risk-than-NOAC-if-AF-coexists",
            "BP-strict-120/80-maximum",
        ],
        "pathognomonic": "Recurrent lobar ICH + Dutch/Iowa family history + Abeta CAA on MRI-SWI",
        "mean_age_dx": 48,
        "pct_female": 53,
        "stroke_n_per40": 38,
        "mri_lesion_n_per40": 40,
        "dementia_n_per40": 30,
        "anticoag_contraindicated": True,
    },
]

DEFINITIONS = {
    "CADASIL": (
        "Cerebral Autosomal Dominant Arteriopathy with Subcortical Infarcts and Leukoencephalopathy; "
        "NOTCH3 cysteine-altering EGF-repeat mutations; arteriolar smooth muscle cell degeneration; "
        "GOM deposits PATHOGNOMONIC on skin biopsy EM; anterior temporal pole WM PATHOGNOMONIC on MRI"
    ),
    "CARASIL": (
        "Cerebral Autosomal Recessive Arteriopathy with Subcortical Infarcts and Leukoencephalopathy; "
        "HTRA1 biallelic LOF; alopecia onset 20-30yr PATHOGNOMONIC; "
        "early-onset spondylosis before stroke; Japanese/Asian enriched"
    ),
    "GOM_deposits": (
        "Granular osmiophilic material; electron-dense deposits in arteriolar BM adjacent to SMC; "
        "PATHOGNOMONIC for CADASIL (NOTCH3); detected by electron microscopy on skin biopsy"
    ),
    "Cerebral_amyloid_angiopathy_CAA": (
        "Amyloid deposition in cerebral vessel walls; "
        "CST3 (cystatin-C amyloid, HCHWA-I) + APP (Abeta amyloid, HCHWA-D): hereditary CAA; "
        "Sporadic CAA: common in elderly; "
        "ICH risk: ANTICOAGULATION ABSOLUTELY CONTRAINDICATED in confirmed CAA"
    ),
    "Moyamoya_disease": (
        "Progressive bilateral stenosis/occlusion of internal carotid arteries and proximal branches; "
        "Puff-of-smoke collateral vessels on angiography; "
        "ACTA2 R179H/R179C: PATHOGNOMONIC for hereditary Moyamoya + iris flocculi + livedo reticularis; "
        "RNF213 p.R4810K: Asian sporadic/familial Moyamoya; "
        "EC-IC bypass surgery for symptomatic Moyamoya"
    ),
    "Porencephaly": (
        "Fluid-filled cavity in cerebral hemisphere communicating with ventricle or subarachnoid space; "
        "COL4A1 mutation PATHOGNOMONIC for hereditary porencephaly; "
        "Presents with seizures/hemiparesis in childhood; "
        "Distinguish from schizencephaly (cortical cleft)"
    ),
    "Interferon_signature": (
        "Elevated expression of interferon-stimulated genes (ISG15/IFIT1/IFIT2/IFIT3/RSAD2/SIGLEC1); "
        "PATHOGNOMONIC for Aicardi-Goutières syndrome (AGS, TREX1/RNASEH2A-C/SAMHD1/ADAR/IFIH1); "
        "Measured by IFN score in blood or CSF IFN-alpha; "
        "Target for JAK inhibitor (baricitinib) therapy"
    ),
    "HCHWA_I": (
        "Hereditary Cerebral Hemorrhage with Amyloidosis - Icelandic; "
        "CST3 L68Q cystatin-C amyloid; recurrent lobar ICH onset 20-30yr; "
        "Icelandic founder; ANTICOAGULATION + ANTIPLATELETS CONTRAINDICATED"
    ),
    "HCHWA_D": (
        "Hereditary Cerebral Hemorrhage with Amyloidosis - Dutch; "
        "APP A692G (Dutch) / D694N (Iowa); Abeta cerebrovascular amyloid; "
        "Recurrent lobar ICH onset 40-60yr; Alzheimer-like dementia; "
        "ANTICOAGULATION ABSOLUTELY CONTRAINDICATED"
    ),
    "SVD_hereditary": (
        "Hereditary small vessel disease: NOTCH3 (CADASIL) + HTRA1 (CARASIL) + COL4A1/COL4A2; "
        "MRI: periventricular WM hyperintensities + lacunar infarcts; "
        "Young onset (before age 50yr) or family history -> gene panel"
    ),
    "Cortical_superficial_siderosis": (
        "Hemosiderin deposits in cerebral cortical sulci on MRI SWI/T2*; "
        "PATHOGNOMONIC for cerebral amyloid angiopathy (CAA); "
        "CST3 / APP hereditary CAA: ANTICOAGULATION ABSOLUTELY CONTRAINDICATED"
    ),
    "Baricitinib_AGS": (
        "JAK1/JAK2 inhibitor; FDA-approved for Aicardi-Goutières syndrome (2023); "
        "Inhibits JAK-STAT pathway downstream of type-I interferon receptor; "
        "Reduces ISG expression; stabilizes neurological function; "
        "TREX1 and other AGS genes respond"
    ),
    "EC_IC_bypass": (
        "Extracranial-intracranial bypass surgery; "
        "Anastomoses superficial temporal artery to middle cerebral artery; "
        "Indicated for symptomatic Moyamoya (ACTA2) with hemodynamic compromise; "
        "Staged bilateral for bilateral Moyamoya; "
        "Aspirin post-bypass; AVOID hypotension"
    ),
    "Anterior_temporal_WM": (
        "Anterior temporal lobe white matter hyperintensity on MRI FLAIR; "
        "Highly specific for CADASIL (NOTCH3) -- not seen in sporadic SVD; "
        "External capsule involvement also characteristic; "
        "Score: ARWMC + anterior temporal pole involvement -> suspect NOTCH3"
    ),
}


def _make_cohort(gene_entry: dict, seed: int) -> dict:
    rng = random.Random(seed)
    n = 40
    gene = gene_entry["gene"]
    base_age = gene_entry["mean_age_dx"]
    base_female = gene_entry["pct_female"]
    stroke_n = gene_entry["stroke_n_per40"]
    mri_n = gene_entry["mri_lesion_n_per40"]
    dementia_n = gene_entry["dementia_n_per40"]

    ages = [max(10, int(rng.gauss(base_age, 8))) for _ in range(n)]
    females = rng.sample(range(n), int(n * base_female / 100))

    stroke_pts = rng.sample(range(n), stroke_n)
    mri_pts = rng.sample(range(n), mri_n)
    dementia_pts = rng.sample(range(n), dementia_n)

    anticoag_pts = rng.sample(range(n), rng.randint(2, 8))
    antiplatelet_pts = rng.sample(range(n), rng.randint(8, 22))
    bp_pts = rng.sample(range(n), rng.randint(20, 32))

    return {
        "gene": gene,
        "n": n,
        "seed": seed,
        "syndrome": gene_entry["syndrome"],
        "inheritance": gene_entry["inheritance"],
        "locus": gene_entry["locus"],
        "mean_age_dx": round(sum(ages) / len(ages), 1),
        "pct_female": round(len(females) / n * 100, 1),
        "stroke_n": stroke_n,
        "mri_lesion_n": mri_n,
        "dementia_n": dementia_n,
        "on_anticoagulation_n": len(anticoag_pts),
        "on_antiplatelet_n": len(antiplatelet_pts),
        "bp_controlled_n": len(bp_pts),
        "pathognomonic": gene_entry.get("pathognomonic", ""),
        "anticoag_contraindicated": gene_entry.get("anticoag_contraindicated", False),
    }


def generate_overview() -> dict:
    genes = [g["gene"] for g in ATLAS_GENES]
    cohort_stats = []
    for i, g in enumerate(ATLAS_GENES):
        cohort_stats.append({
            "gene": g["gene"],
            "syndrome": g["syndrome"].split(" / ")[0],
            "locus": g["locus"],
            "inheritance": g["inheritance"].split(" ")[0],
            "mean_age": g["mean_age_dx"],
            "pct_female": g["pct_female"],
            "anticoag_ci": g.get("anticoag_contraindicated", False),
        })

    return {
        "atlas": "Hereditary-Cerebrovascular-Disease-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Cerebrovascular Disease Reference "
            "(NOTCH3-COL4A1-COL4A2-ACTA2-HTRA1-TREX1-CST3-APP; "
            "seeds 3774-3781; 320-patient aggregate)"
        ),
        "total_patients": 320,
        "gene_count": len(genes),
        "genes": genes,
        "seeds": "3774-3781",
        "cohort_stats": cohort_stats,
        "clinical_categories": {
            "hereditary_svd_cadasil_carasil": ["NOTCH3", "HTRA1", "COL4A1", "COL4A2"],
            "cerebral_amyloid_angiopathy_hemorrhagic": ["CST3", "APP"],
            "moyamoya_vasculopathy": ["ACTA2"],
            "autoinflammatory_interferon_mediated": ["TREX1"],
            "anticoagulation_absolutely_contraindicated": ["CST3", "APP", "COL4A1", "COL4A2", "NOTCH3"],
            "disease_modifying_therapy_exists": ["TREX1"],
            "genetic_panel_mandatory_together": ["COL4A1", "COL4A2"],
            "cerebral_mra_mandatory": ["ACTA2"],
        },
        "key_clinical_rules": [
            "NOTCH3-GOM-PATHOGNOMONIC: GOM deposits on skin biopsy EM = PATHOGNOMONIC for CADASIL -- AVOID anticoagulation in CADASIL (ICH risk elevated); anterior temporal pole WM = MRI clue",
            "COL4A1-PORENCEPHALY-PATHOGNOMONIC: Porencephaly = COL4A1 PATHOGNOMONIC -- AVOID anticoagulation + contact sports; test COL4A1+COL4A2 together (same locus 13q34)",
            "CST3-APP-ANTICOAG-ABSOLUTELY-CI: HCHWA-I (CST3) + HCHWA-D (APP) = ANTICOAGULATION ABSOLUTELY CONTRAINDICATED -- CAA hemorrhagic risk; BP <120/80 strictly",
            "ACTA2-MOYAMOYA-TRIAD: ACTA2 R179H/R179C = Moyamoya + iris flocculi + livedo reticularis PATHOGNOMONIC triad -- cerebral MRA mandatory at diagnosis",
            "HTRA1-ALOPECIA-PATHOGNOMONIC: HTRA1 CARASIL = alopecia before stroke PATHOGNOMONIC (unique among hereditary SVD) -- test biallelic HTRA1 in young patient + alopecia + spondylosis",
            "TREX1-BARICITINIB: TREX1 AGS = elevated interferon signature PATHOGNOMONIC -- baricitinib JAK inhibitor FDA-approved 2023; IFN score guides treatment",
            "HEREDITARY-CVD-YOUNG-RULE: Any young patient (<50yr) + stroke/ICH + family history -> hereditary cerebrovascular gene panel: NOTCH3/COL4A1/COL4A2/ACTA2/HTRA1/TREX1/CST3/APP",
            "CAA-ANTICOAG-RULE: Confirmed CAA (CST3/APP/sporadic) + any ICH or cortical superficial siderosis -> ANTICOAGULATION ABSOLUTELY CONTRAINDICATED; if AF coexists -- NOAC lower risk than warfarin but still high risk; multidisciplinary decision required",
        ],
    }


def generate_breakdown() -> dict:
    gene_data = {}
    for i, g in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        cohort = _make_cohort(g, seed)
        gene_data[g["gene"]] = cohort

    return {
        "atlas": "Hereditary-Cerebrovascular-Disease-Atlas",
        "total_patients": 320,
        "seeds": "3774-3781",
        "gene_data": gene_data,
        "atlas_genes": ATLAS_GENES,
        "summary": {
            "cadasil_carasil_svd_genes": ["NOTCH3", "HTRA1", "COL4A1", "COL4A2"],
            "caa_hemorrhagic_genes": ["CST3", "APP"],
            "moyamoya_gene": ["ACTA2"],
            "autoinflammatory_gene": ["TREX1"],
            "anticoagulation_absolutely_contraindicated": ["CST3", "APP", "COL4A1", "COL4A2", "NOTCH3"],
            "disease_modifying_therapy_baricitinib": ["TREX1"],
            "test_together_same_locus_13q34": ["COL4A1", "COL4A2"],
            "youngest_stroke_onset": ["CST3"],
            "cerebral_mra_mandatory_at_diagnosis": ["ACTA2"],
            "pathognomonic_alopecia": ["HTRA1"],
            "pathognomonic_gom_deposits": ["NOTCH3"],
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Cerebrovascular-Disease-Atlas",
        "definitions": DEFINITIONS,
        "critical_clinical_rules": {
            "NOTCH3-CADASIL-GOM-PATHOGNOMONIC": (
                "NOTCH3 cysteine-altering EGF repeat mutation = CADASIL; "
                "GOM deposits on skin biopsy EM = CONFIRMATORY PATHOGNOMONIC; "
                "AVOID anticoagulation (ICH risk elevated in CADASIL even without AF); "
                "Antiplatelet OK post-lacunar stroke; BP <130/80; "
                "No disease-modifying therapy 2026; anterior temporal pole WM = MRI clue"
            ),
            "COL4A1-PORENCEPHALY-ANTICOAG-CI": (
                "COL4A1 = porencephaly PATHOGNOMONIC (fluid-filled cerebral cavity); "
                "ANTICOAGULATION CONTRAINDICATED -- ICH risk; "
                "AVOID contact sports + excessive exertion; "
                "Test COL4A1+COL4A2 together (13q34 tandem locus); "
                "MLPA/CNV required -- WES misses large deletions"
            ),
            "COL4A1-COL4A2-PANEL-TOGETHER": (
                "COL4A1 + COL4A2 at 13q34 -- always test as panel; "
                "COL4A2 = ICH-dominant, less porencephaly; "
                "Indistinguishable clinically -- requires genotyping both; "
                "Variants in either disrupt obligate COL4A1/COL4A2 heterotrimer"
            ),
            "CST3-ANTICOAGULATION-ABSOLUTELY-CI": (
                "CST3 L68Q HCHWA-I = recurrent lobar ICH 20-30yr; "
                "ANTICOAGULATION ABSOLUTELY CONTRAINDICATED; "
                "ANTIPLATELETS AVOID; "
                "BP strict <120/80; "
                "Reduced CSF cystatin-C diagnostic; "
                "CAA on MRI-SWI (cortical superficial siderosis)"
            ),
            "APP-HCHWA-D-ANTICOAG-ABSOLUTELY-CI": (
                "APP A692G Dutch / D694N Iowa = HCHWA-D; "
                "Recurrent lobar ICH 40-60yr + Alzheimer-like dementia; "
                "ANTICOAGULATION ABSOLUTELY CONTRAINDICATED; "
                "Distinguish: CST3 younger (20-30yr) + Icelandic; APP older (40-60yr) + Dutch/Iowa; "
                "If AF coexists: NOAC may be less harmful than warfarin -- multidisciplinary only"
            ),
            "ACTA2-MOYAMOYA-TRIAD-MRA-MANDATORY": (
                "ACTA2 R179H/R179C = Moyamoya + iris flocculi + livedo reticularis PATHOGNOMONIC TRIAD; "
                "Cerebral MRA MANDATORY at diagnosis + every 2yr; "
                "EC-IC bypass for symptomatic hemodynamic compromise; "
                "Also has TAAD -- aortic imaging mandatory concurrently"
            ),
            "HTRA1-CARASIL-ALOPECIA-FIRST": (
                "HTRA1 CARASIL: alopecia onset 20-30yr BEFORE stroke = PATHOGNOMONIC; "
                "Sequence: alopecia -> spondylosis -> stroke -> dementia; "
                "Japanese/Asian enriched (Yamamoto R302H); "
                "AD monoallelic = milder CADASIL-like; "
                "Young adult + alopecia + lumbar spondylosis + stroke = test HTRA1 immediately"
            ),
            "TREX1-BARICITINIB-JAKi": (
                "TREX1 AGS: elevated interferon score (ISG15/IFIT1/RSAD2/SIGLEC1) PATHOGNOMONIC; "
                "Baricitinib (JAK1/JAK2 inhibitor): FDA-approved AGS 2023; "
                "Reduces ISG expression; stabilizes neurological function; "
                "CSF: lymphocytic pleocytosis + IFN-alpha elevated; "
                "MRI: bilateral basal ganglia calcifications + WM disease"
            ),
            "HEREDITARY-CVD-YOUNG-STROKE-PANEL": (
                "Young stroke/ICH (<50yr) + family history -> hereditary cerebrovascular panel: "
                "NOTCH3 (CADASIL) + COL4A1/COL4A2 (SVD/ICH) + ACTA2 (Moyamoya) + "
                "HTRA1 (CARASIL) + TREX1 (AGS) + CST3 (HCHWA-I) + APP (HCHWA-D); "
                "MRI clues: anterior temporal pole (NOTCH3) / porencephaly (COL4A1) / "
                "CAA-SWI (CST3/APP) / basal ganglia Ca2+ (TREX1)"
            ),
            "CASCADE-TESTING-HEREDITARY-CVD": (
                "All 1st-degree relatives of hereditary cerebrovascular disease index case; "
                "AD (NOTCH3/COL4A1/COL4A2/ACTA2/CST3/APP): 50% inheritance risk; "
                "AR (HTRA1 CARASIL): parents carriers; siblings 25% risk; "
                "AR/AD (TREX1): biallelic severe vs. monoallelic mild; "
                "Pre-symptomatic identification enables BP control, lifestyle modification, AVOID anticoagulants"
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== Hereditary Cerebrovascular Disease Atlas ===")
    ov = generate_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print()
    br = generate_breakdown()
    for gene, d in br["gene_data"].items():
        print(f"  {gene}: n={d['n']}, age_dx={d['mean_age_dx']}yr, stroke={d['stroke_n']}/40, dementia={d['dementia_n']}/40, anticoag_ci={d['anticoag_contraindicated']}")
    print()
    df = generate_definitions()
    print(f"Definitions: {len(df['definitions'])} terms")
    print(f"Clinical rules: {len(df['critical_clinical_rules'])} rules")
    print("SCRIPT OK")
