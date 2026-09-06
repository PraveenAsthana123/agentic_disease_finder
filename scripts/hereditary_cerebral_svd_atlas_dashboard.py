#!/usr/bin/env python3
"""Hereditary-Cerebral-SVD-Atlas — Complete 8-Gene Hereditary Cerebral Small Vessel Disease Atlas
NOTCH3  (Notch Receptor 3; 2321 aa; 19p13.12; AD;
          CADASIL — Cerebral Autosomal Dominant Arteriopathy with Subcortical Infarcts and Leukoencephalopathy;
          Most common hereditary SVD; EGF-like domain cysteine mutations → GOM deposits;
          SKIN BIOPSY (EM GOM) DIAGNOSTIC; NO tPA; NO OCP; migraine aura precedes stroke 15–20 years) ·
HTRA1   (HtrA Serine Peptidase 1; 480 aa; 10q26.13; AR (CARASIL) + AD (HTRA1-AD);
          CARASIL — AR: alopecia + lumbar spondylosis PATHOGNOMONIC before stroke; onset 20s–40s;
          HTRA1-AD — heterozygous: milder, onset 50s–60s; TGFβ ligand cleavage protease) ·
COL4A1  (Collagen Type IV Alpha 1; 1669 aa; 13q34; AD;
          COL4A1 angiopathy; porencephaly = COL4A1/COL4A2 until proven otherwise;
          NO anticoagulants; NO contact sports; retinal arterial tortuosity; test COL4A2 simultaneously) ·
COL4A2  (Collagen Type IV Alpha 2; 1712 aa; 13q34; AD;
          COL4A2 angiopathy — same locus as COL4A1; test BOTH simultaneously;
          ICH + porencephaly; milder average than COL4A1; NO anticoagulants) ·
TREX1   (Three Prime Repair Exonuclease 1; 314 aa; 3p21.31; AD;
          RVCL-S — Retinal Vasculopathy with Cerebral Leukoencephalopathy and Systemic manifestations;
          Retinal vasculopathy ALWAYS first; fluorescein angiogram MANDATORY;
          Punctate gadolinium-enhancing mass-like lesions PATHOGNOMONIC — NOT tumour/demyelination) ·
GLA     (Galactosidase Alpha; 429 aa; Xq22.1; XLD;
          Fabry disease — MOST COMMON TREATABLE hereditary SVD;
          Posterior circulation lacunar strokes; pulvinar sign 25–30% males PATHOGNOMONIC;
          ERT: agalsidase; migalastat for amenable missense; lyso-GL3 diagnostic in females) ·
ADA2    (Adenosine Deaminase 2; 511 aa; 22q11.1; AR;
          DADA2 — Deficiency of Adenosine Deaminase 2;
          Livedoid rash + childhood stroke PATHOGNOMONIC; ADA2 enzyme activity DIAGNOSTIC;
          TNF blockade CURATIVE/disease-modifying — do NOT use steroids alone) ·
CST3    (Cystatin C; 146 aa; 20p11.21; AD;
          HCCAA — Hereditary Cystatin C Amyloid Angiopathy (Icelandic type);
          L68Q mutation — plasma cystatin C paradoxically LOW;
          ~100% ICH by age 30 homozygotes; heterozygotes mean age 50; NO anticoagulants)
320-patient aggregate cohort (8 × 40, seeds 1470–1477)
"""

import random

SEED_BASE = 1470

CEREBRAL_SVD_GENES = [
    # ── NOTCH3 — CADASIL ──
    {
        "gene": "NOTCH3",
        "protein": "Notch Receptor 3 — EGF-Like Domain Transmembrane Signalling Receptor",
        "alias": (
            "NOTCH3; OMIM gene 600276; CADASIL OMIM 125310; 19p13.12; 2321 aa; ~260 kDa; "
            "Notch Receptor 3 — type I transmembrane receptor; EGF-like domains 1–34 harbour "
            "virtually all pathogenic missense mutations; all pathogenic mutations alter the "
            "number of cysteine residues within an EGF-like domain (odd → even or even → odd), "
            "causing misfolded NOTCH3 ECD accumulation and GOM (granular osmiophilic material) "
            "deposits in vessel walls; CADASIL: most common hereditary cerebral SVD globally "
            "(~1:15000–50000); migraine with aura precedes stroke by 15–20 years; "
            "subcortical ischaemic strokes → progressive dementia; white matter lesions "
            "anterior temporal poles + external capsule PATHOGNOMONIC on MRI; "
            "SKIN BIOPSY (EM for GOM) or anti-NOTCH3 ectodomain immunostaining DIAGNOSTIC — "
            "avoids invasive brain biopsy; NO tPA (small vessel mechanism, no large vessel occlusion); "
            "NO oral contraceptives (migraine + aura + stroke risk); "
            "NOTCH3-antibody (anti-NOTCH3 ECD) immunoassay in plasma emerging biomarker; "
            "EGF domains 1–6 mutations most severe; EGF >34 atypical/uncertain pathogenicity; "
            "No curative therapy; antiplatelet; vascular risk factor control; avoid smoking"
        ),
        "aa": "2321 aa",
        "kDa": "~260 kDa",
        "locus": "19p13.12",
        "omim_gene": 600276,
        "omim_disease": 125310,
        "inheritance": "AD — EGF-like domain cysteine-altering missense mutations",
        "gene_class": (
            "NOTCH3 encodes the Notch 3 receptor, a single-pass transmembrane protein expressed "
            "predominantly in vascular smooth muscle cells (vSMCs) and pericytes. The extracellular "
            "domain contains 34 EGF-like repeats; virtually all CADASIL-causing mutations lie within "
            "EGF domains 1–34 and result in an odd number of cysteine residues (gain or loss of one "
            "cysteine), creating an unpaired thiol that drives ectodomain misfolding and progressive "
            "GOM accumulation in the tunica media of small penetrating arteries. "
            "This leads to vSMC degeneration, vessel wall thickening, luminal narrowing, and "
            "ultimately ischaemic lacunar infarcts and diffuse leukoencephalopathy. "
            "CADASIL manifests in the 3rd–5th decade with the clinical tetrad: migraine with aura "
            "(35–75%), recurrent subcortical ischaemic strokes, cognitive decline, and psychiatric "
            "disturbance. MRI hallmarks are bilateral confluent white matter lesions with "
            "PATHOGNOMONIC anterior temporal pole and external capsule involvement. "
            "Skin biopsy electron microscopy (GOM deposits) or anti-NOTCH3 ECD immunostaining "
            "are the diagnostic workhorses — avoiding brain biopsy. "
            "No disease-modifying therapy yet; antiplatelet agents used, but tPA is contraindicated "
            "because CADASIL strokes are lacunar (perforator occlusion), not embolic."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("CADASIL — EGF domain 1-6 (most severe)", 0.48),
            ("CADASIL — EGF domain 7-34 (moderate)", 0.37),
            ("CADASIL — EGF domain >34 (atypical/uncertain)", 0.08),
            ("CADASIL-like (clinical, GOM-confirmed, sequencing pending)", 0.07),
        ],
        "onset_range": (25, 55),
        "key_alerts": [
            "NOTCH3-CADASIL: EGF-domain cysteine count — odd number within EGF 1-34 = pathogenic rule",
            "CADASIL-MRI: Anterior temporal pole WML + external capsule involvement PATHOGNOMONIC",
            "NO-tPA: CADASIL strokes are lacunar/perforator — thrombolysis contraindicated",
            "NO-OCP: Migraine with aura + CADASIL = absolute OCP contraindication (stroke risk)",
            "SKIN-BIOPSY: EM for GOM deposits or anti-NOTCH3 ECD immunostaining DIAGNOSTIC",
            "MIGRAINE-FIRST: Migraine with aura precedes stroke by 15-20 years — screen family",
            "NOTCH3-ECD-ANTIBODY: Emerging plasma biomarker — correlates with GOM burden",
            "CASCADE-Testing: All first-degree relatives — presymptomatic MRI + genetic counselling",
        ],
    },
    # ── HTRA1 — CARASIL / HTRA1-AD ──
    {
        "gene": "HTRA1",
        "protein": "HtrA Serine Peptidase 1 — TGFβ Ligand-Cleaving Secreted Protease",
        "alias": (
            "HTRA1; OMIM gene 602194; CARASIL OMIM 600142; HTRA1-AD OMIM 616779; 10q26.13; 480 aa; ~51 kDa; "
            "HtrA Serine Peptidase 1 — secreted serine protease; cleaves TGFβ family ligands "
            "(TGFβ1, TGFβ2, GDF5) to downregulate TGFβ signalling in blood vessels; "
            "biallelic loss-of-function → CARASIL (AR): "
            "alopecia (premature diffuse) + lumbar/cervical spondylosis PATHOGNOMONIC combination "
            "before stroke onset; early-onset ischaemic strokes 20s–40s; diffuse WML; "
            "NO conventional vascular risk factors (normotensive, non-diabetic); "
            "heterozygous LOF → HTRA1-AD: milder adult-onset (~50s–60s); "
            "some kindreds diagnosed as 'atypical CADASIL' before HTRA1 testing; "
            "CARASIL brain biopsy: concentric intimal thickening, GOM ABSENT (distinguishes from CADASIL); "
            "TGFβ excess (unopposed) → vessel wall fibrosis and arteriolar occlusion; "
            "No curative therapy; vascular risk factor control; physiotherapy for spondylosis"
        ),
        "aa": "480 aa",
        "kDa": "~51 kDa",
        "locus": "10q26.13",
        "omim_gene": 602194,
        "omim_disease": 600142,
        "inheritance": "AR (CARASIL biallelic) · AD (HTRA1-AD heterozygous)",
        "gene_class": (
            "HTRA1 encodes a disulfide-linked homotrimeric serine protease secreted into the "
            "extracellular matrix. Its primary substrates include TGFβ1, TGFβ2, LTBP-1, and GDF5. "
            "In blood vessels, HTRA1-mediated TGFβ ligand cleavage suppresses pro-fibrotic signalling; "
            "loss of this function leads to unopposed TGFβ-driven concentric intimal hyalinosis, "
            "arteriolar occlusion, and progressive leukoencephalopathy. "
            "Biallelic HTRA1 mutations cause CARASIL — an AR hereditary SVD distinct from CADASIL "
            "by: (1) absence of GOM on EM, (2) presence of premature alopecia and lumbar spondylosis "
            "(both PATHOGNOMONIC before neurological onset), and (3) recessive inheritance. "
            "Heterozygous loss-of-function (HTRA1-AD) causes a milder, later-onset SVD phenotype. "
            "HTRA1 should be sequenced in all GOM-negative hereditary SVD cases, especially those "
            "with the alopecia-spondylosis-stroke triad."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("CARASIL — biallelic LOF (AR)", 0.52),
            ("HTRA1-AD — heterozygous LOF (AD)", 0.38),
            ("HTRA1 VUS under evaluation", 0.10),
        ],
        "onset_range": (20, 65),
        "key_alerts": [
            "CARASIL-TRIAD: Alopecia + Lumbar spondylosis + Stroke (20s-40s) PATHOGNOMONIC — no RF",
            "GOM-ABSENT: CARASIL has NO GOM on EM (distinguishes from CADASIL) — skin biopsy still useful",
            "HTRA1-AD: Heterozygous — milder onset 50s-60s — sequencing in GOM-negative hereditary SVD",
            "CARASIL-No-RF: Young stroke + NO hypertension/DM/smoking → mandatory HTRA1 testing",
            "ALOPECIA-FIRST: Premature diffuse alopecia may precede stroke by years — screen family",
            "SPONDYLOSIS-EARLY: Multilevel spondylosis before age 40 + WML = CARASIL until proven otherwise",
            "TGFbeta-EXCESS: Theoretical caution with exogenous TGFβ pathway activators (no proven harm)",
            "CASCADE-Testing: All first-degree relatives of AR proband — carrier testing + counselling",
        ],
    },
    # ── COL4A1 — COL4A1 angiopathy ──
    {
        "gene": "COL4A1",
        "protein": "Collagen Type IV Alpha 1 — Basement Membrane Network-Forming Collagen",
        "alias": (
            "COL4A1; OMIM gene 120130; COL4A1 OMIM 607595; 13q34; 1669 aa; ~185 kDa; "
            "Collagen Type IV Alpha 1 — major structural component of all basement membranes; "
            "COL4A1 and COL4A2 co-assemble as [COL4A1]2[COL4A2] heterotrimers; "
            "pathogenic missense mutations disrupt Gly-X-Y triple helix motif → BM structural instability; "
            "COL4A1 angiopathy: porencephaly (usually prenatal/perinatal ICH → cystic cavity), "
            "childhood or adult ICH, retinal arterial tortuosity (fundoscopy hallmark), "
            "leukoencephalopathy; same gene → broad phenotype: infantile hemiplegia to adult SVD; "
            "PORENCEPHALY = COL4A1/COL4A2 until proven otherwise — ALWAYS test both simultaneously; "
            "NO anticoagulants (fragile BM → haemorrhagic transformation); "
            "NO contact sports / high-intensity Valsalva (trauma → ICH); "
            "renal cysts and ESRD rare; muscle cramps (HANAC syndrome: COL4A1 with nephrotic syndrome); "
            "prenatal MRI if fetal ICH suspected"
        ),
        "aa": "1669 aa",
        "kDa": "~185 kDa",
        "locus": "13q34",
        "omim_gene": 120130,
        "omim_disease": 607595,
        "inheritance": "AD — Gly-X-Y missense disrupting triple helix",
        "gene_class": (
            "COL4A1 encodes the alpha-1 chain of type IV collagen, which forms the core scaffold of "
            "virtually all basement membranes (BMs). Two COL4A1 chains and one COL4A2 chain "
            "co-assemble into a triple helix, then cross-link into a polymer network providing "
            "structural integrity and filtration function to BMs in brain, kidney, eye, and muscle. "
            "Pathogenic mutations cluster in glycine residues of the Gly-X-Y collagenous repeats; "
            "a glycine substitution disrupts the triple helix, causing intracellular retention, "
            "ER stress, and secretion of abnormal collagen that destabilises vascular BMs. "
            "Vessel wall fragility leads to haemorrhagic events: perinatal intracerebral haemorrhage "
            "→ porencephaly (the commonest genetic cause), and adult spontaneous or traumatic ICH. "
            "COL4A1 and COL4A2 share the same chromosomal locus (13q34) and must always be "
            "sequenced simultaneously. Anticoagulants are absolutely contraindicated."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("Porencephaly — prenatal/perinatal ICH", 0.42),
            ("Adult spontaneous ICH", 0.32),
            ("Leukoencephalopathy + lacunar strokes", 0.16),
            ("HANAC syndrome (nephropathy + muscle cramps)", 0.10),
        ],
        "onset_range": (0, 60),
        "key_alerts": [
            "PORENCEPHALY: COL4A1 (and COL4A2) mutation until proven otherwise — test BOTH simultaneously",
            "NO-ANTICOAGULANTS: Basement membrane fragility — anticoagulation causes catastrophic ICH",
            "NO-CONTACT-SPORTS: Trauma → ICH in COL4A1 — written sports restriction at every visit",
            "RETINAL-TORTUOSITY: Fundoscopy mandatory — retinal arterial tortuosity in >50% carriers",
            "PRENATAL-ICH: Fetal/neonatal ICH → porencephaly → COL4A1/COL4A2 testing + family screen",
            "COL4A2-SIMULTANEOUSLY: Same 13q34 locus — never test COL4A1 without COL4A2",
            "HANAC: Muscle cramps + haematuria + nephropathy + aneurysm + COL4A1 = HANAC variant",
            "CASCADE-Testing: All first-degree relatives — fundoscopy + brain MRI + genetic testing",
        ],
    },
    # ── COL4A2 — COL4A2 angiopathy ──
    {
        "gene": "COL4A2",
        "protein": "Collagen Type IV Alpha 2 — Basement Membrane Heterotrimer Partner",
        "alias": (
            "COL4A2; OMIM gene 120090; COL4A2 OMIM 614519; 13q34; 1712 aa; ~187 kDa; "
            "Collagen Type IV Alpha 2 — one COL4A2 per [COL4A1]2[COL4A2] heterotrimer; "
            "same 13q34 locus as COL4A1 — ALWAYS test BOTH simultaneously; "
            "pathogenic Gly-X-Y missense mutations: same triple-helix disruption mechanism as COL4A1; "
            "COL4A2 angiopathy: ICH, porencephaly, leukoencephalopathy — milder average than COL4A1 "
            "but wide phenotypic range; haematuria and renal cysts less common than COL4A1; "
            "retinal arterial tortuosity present but less penetrant than COL4A1; "
            "NO anticoagulants; NO contact sports; "
            "COL4A2 G702D commonest UK Biobank variant — may represent hypomorphic allele; "
            "diagnosis often delayed due to phenotypic overlap with idiopathic ICH; "
            "family history of porencephaly/ICH → mandatory 13q34 panel"
        ),
        "aa": "1712 aa",
        "kDa": "~187 kDa",
        "locus": "13q34",
        "omim_gene": 120090,
        "omim_disease": 614519,
        "inheritance": "AD — Gly-X-Y missense",
        "gene_class": (
            "COL4A2 encodes the alpha-2 chain of type IV collagen. Each [COL4A1]2[COL4A2] protomer "
            "has one COL4A2 subunit; pathogenic missense mutations in COL4A2 disrupt heterotrimer "
            "formation identically to COL4A1 mutations. The clinical manifestation is largely "
            "indistinguishable from COL4A1 angiopathy: porencephaly, spontaneous ICH, and "
            "leukoencephalopathy. On average, COL4A2 carriers have a slightly milder course, "
            "but overlap is substantial. The shared 13q34 locus means that comprehensive sequencing "
            "must always cover both genes. Haematuria is present in some COL4A2 carriers but "
            "HANAC-like syndrome is rarer than with COL4A1 mutations. Standard clinical panels often "
            "miss COL4A2 if ordered as 'COL4A1 only'; the correct order is always a 13q34 two-gene "
            "simultaneous panel."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("Spontaneous ICH — adult onset", 0.45),
            ("Porencephaly — perinatal ICH", 0.30),
            ("Leukoencephalopathy — progressive", 0.18),
            ("Subclinical / imaging-only (family screen)", 0.07),
        ],
        "onset_range": (0, 65),
        "key_alerts": [
            "COL4A1-SIMULTANEOUSLY: Never test COL4A2 alone — always order 13q34 two-gene panel",
            "NO-ANTICOAGULANTS: Haemorrhagic angiopathy — anticoagulation absolutely contraindicated",
            "NO-CONTACT-SPORTS: Trauma risk for ICH — document restriction in every clinic letter",
            "PORENCEPHALY: Perinatal ICH → porencephaly → COL4A2 (and COL4A1) testing mandatory",
            "ICH-YOUNG-ADULT: Spontaneous ICH <60 without hypertension → COL4A1/COL4A2 panel",
            "MILDER-THAN-COL4A1: Average milder but wide range — full neurological surveillance",
            "HAEMATURIA-SCREEN: Renal involvement rarer but present — urinalysis at diagnosis",
            "CASCADE-Testing: All first-degree relatives — MRI + fundoscopy + genetic panel",
        ],
    },
    # ── TREX1 — RVCL-S ──
    {
        "gene": "TREX1",
        "protein": "Three Prime Repair Exonuclease 1 — ER-Anchored 3'-5' DNase",
        "alias": (
            "TREX1; OMIM gene 606609; RVCL-S OMIM 192315; 3p21.31; 314 aa; ~33 kDa; "
            "Three Prime Repair Exonuclease 1 — major mammalian 3'-5' DNA exonuclease; "
            "ER-anchored via C-terminal transmembrane domain; degrades cytosolic ssDNA and dsDNA "
            "to prevent innate immune activation (cGAS-STING axis); "
            "heterozygous C-terminal frameshift/truncating mutations → RVCL-S: "
            "Retinal Vasculopathy with Cerebral Leukoencephalopathy and Systemic manifestations; "
            "retinal vasculopathy ALWAYS first (fluorescein angiogram MANDATORY at diagnosis); "
            "punctate/ring gadolinium-enhancing mass-like white matter lesions PATHOGNOMONIC; "
            "systemic: Raynaud phenomenon, hepatopathy, nephropathy, normocytic anaemia; "
            "same gene: missense → lupus/SLE/AGS (different mechanism — gain of immune activation); "
            "NO curative therapy; immunosuppression NOT effective for mass lesions; "
            "supportive care; transplant evaluation for renal/hepatic failure"
        ),
        "aa": "314 aa",
        "kDa": "~33 kDa",
        "locus": "3p21.31",
        "omim_gene": 606609,
        "omim_disease": 192315,
        "inheritance": "AD — heterozygous C-terminal truncating mutations",
        "gene_class": (
            "TREX1 encodes the principal cytosolic 3'-5' DNA exonuclease in mammalian cells. "
            "Its N-terminal catalytic domain degrades misincorporated ribonucleotides, "
            "Okazaki fragment primers, and cytosolic DNA generated during retrotransposon activity "
            "and genome replication stress. The C-terminus anchors TREX1 to the ER outer membrane. "
            "Heterozygous frameshift mutations in the C-terminal transmembrane anchor (exon 7) "
            "cause RVCL-S by a dominant-negative mechanism: the truncated protein mislocalises and "
            "fails to degrade cytosolic DNA, activating the cGAS-STING pathway and causing "
            "endothelial cell dysfunction and vessel wall inflammation. Clinically, retinal "
            "arteriolar occlusions appear first (fluorescein angiography critical), followed by "
            "characteristic punctate, ring, or mass-like gadolinium-enhancing white matter lesions "
            "that mimic brain tumours and demyelination — the imaging PATHOGNOMONIC signature. "
            "Distinct from TREX1 missense mutations (AGS/SLE via loss of nuclease activity). "
            "Immunosuppression does not halt progression."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("RVCL-S — full triad (retina + brain + systemic)", 0.55),
            ("RVCL-S — retina + brain (no systemic)", 0.28),
            ("RVCL-S — retina only (early / presymptomatic)", 0.17),
        ],
        "onset_range": (35, 60),
        "key_alerts": [
            "RVCL-S-RETINA-FIRST: Fluorescein angiogram MANDATORY at diagnosis — retinal vasculopathy always precedes brain",
            "RVCL-S-MRI-PATHOGNOMONIC: Punctate/ring gadolinium-enhancing mass lesions — NOT tumour, NOT demyelination",
            "TREX1-C-TERMINAL: RVCL-S mutations = frameshift in C-terminal anchor exon — distinguish from missense (SLE/AGS)",
            "NO-IMMUNOSUPPRESSION: Immunosuppressants NOT effective for RVCL-S white matter lesions",
            "SYSTEMIC-SCREEN: Raynaud + hepatopathy + nephropathy + anaemia — annual multi-organ screen",
            "BRAIN-BIOPSY-AVOID: Mass lesions on MRI = RVCL-S — biopsy risks before TREX1 testing",
            "TRANSPLANT-EVALUATION: Renal/hepatic failure may require organ transplant — multidisciplinary",
            "CASCADE-Testing: All first-degree relatives — ophthalmology + MRI + TREX1 C-terminal sequencing",
        ],
    },
    # ── GLA — Fabry disease ──
    {
        "gene": "GLA",
        "protein": "Galactosidase Alpha — Lysosomal Alpha-Galactosidase A",
        "alias": (
            "GLA; OMIM gene 300644; Fabry disease OMIM 301500; Xq22.1; 429 aa; ~46 kDa; "
            "Galactosidase Alpha / Alpha-Galactosidase A — lysosomal hydrolase; "
            "cleaves terminal alpha-galactosyl moieties from glycosphingolipids (Gb3, lyso-Gb3); "
            "XLD: males affected (classic and late-onset/cardiac variants); "
            "females may be significantly affected (X-inactivation); "
            "MOST COMMON TREATABLE hereditary cerebral SVD — TREATABLE with ERT + chaperone; "
            "posterior circulation lacunar stroke hallmark (DO NOT MISS in young cryptogenic stroke); "
            "pulvinar sign (T1 MRI hyperintensity bilateral pulvinar) 25–30% affected males PATHOGNOMONIC; "
            "enzyme activity: males <1% classic; females can be NORMAL — lyso-Gb3 (plasma) DIAGNOSTIC; "
            "ERT: agalsidase beta 1 mg/kg q2w (IV) or agalsidase alfa 0.2 mg/kg q2w; "
            "migalastat (Galafold) oral — ONLY for amenable missense variants (use Fabry-specific predictor); "
            "annual: echocardiography (LVH/cardiomyopathy), eGFR, urine protein, cornea whorling (slit lamp)"
        ),
        "aa": "429 aa",
        "kDa": "~46 kDa",
        "locus": "Xq22.1",
        "omim_gene": 300644,
        "omim_disease": 301500,
        "inheritance": "XLD — X-linked dominant; males classic; females variable (X-inactivation)",
        "gene_class": (
            "GLA encodes alpha-galactosidase A (α-GAL A), a lysosomal acid hydrolase that "
            "cleaves terminal alpha-galactosyl residues from glycosphingolipids, primarily "
            "globotriaosylceramide (Gb3/GL-3). α-GAL A deficiency causes progressive Gb3 "
            "accumulation in vascular endothelium, smooth muscle, cardiomyocytes, podocytes, "
            "and dorsal root ganglia neurons. Cerebrovascular involvement manifests as "
            "posterior circulation lacunar strokes (VA/BA territory) from endothelial Gb3 "
            "deposition causing hypercoagulability and vessel wall thickening. "
            "The pulvinar sign (bilateral T1 MRI hyperintensity in the thalamic pulvinar) "
            "is a PATHOGNOMONIC MRI feature in 25–30% of affected males and results from "
            "calcification of Gb3-laden vessels. Fabry disease is the most common treatable "
            "hereditary SVD: ERT clears vascular Gb3 and, if started before irreversible organ "
            "damage, prevents progressive cardiomyopathy and renal failure. Lyso-Gb3 in plasma "
            "is the most sensitive biomarker for females, in whom enzyme activity may be normal."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("Classic Fabry — male, <1% enzyme activity", 0.40),
            ("Late-onset cardiac variant — male, residual activity", 0.25),
            ("Female heterozygote — symptomatic", 0.28),
            ("Fabry-associated cryptogenic stroke — undiagnosed prior", 0.07),
        ],
        "onset_range": (18, 65),
        "key_alerts": [
            "FABRY-TREATABLE: Most common TREATABLE hereditary SVD — ERT + migalastat; do not miss",
            "POSTERIOR-CIRCULATION: VA/BA territory lacunar strokes in young adult → GLA enzyme activity mandatory",
            "PULVINAR-SIGN: Bilateral T1-hyperintensity pulvinar on MRI — 25-30% males PATHOGNOMONIC",
            "FEMALE-LYSO-Gb3: Enzyme activity can be NORMAL in females — plasma lyso-Gb3 is the diagnostic test",
            "MIGALASTAT: Only for amenable missense variants — use Fabry-variant predictor, not for all missense",
            "ANNUAL-ECHO: LVH → cardiomyopathy → sudden death — echocardiography mandatory every year",
            "ANNUAL-eGFR-PROTEIN: Podocyte Gb3 → FSGS → ESRD — nephrology co-management",
            "CORNEA-WHORLING: Cornea verticillata (slit-lamp) — present in >70% males; females variable",
        ],
    },
    # ── ADA2 — DADA2 ──
    {
        "gene": "ADA2",
        "protein": "Adenosine Deaminase 2 — Secreted Growth Factor-Like Adenosine Deaminase",
        "alias": (
            "ADA2 (formerly CECR1); OMIM gene 607575; DADA2 OMIM 615688; 22q11.1; 511 aa; ~59 kDa; "
            "Adenosine Deaminase 2 — secreted glycoprotein; distinct from ADA1 (housekeeping enzyme); "
            "promotes macrophage differentiation towards M2 phenotype; maintains endothelial integrity; "
            "biallelic LOF mutations → DADA2: Deficiency of Adenosine Deaminase 2; "
            "CHILDHOOD vasculopathy: ischaemic AND haemorrhagic strokes in the same patient; "
            "livedoid rash (livedo racemosa) PATHOGNOMONIC + stroke in child/young adult; "
            "ADA2 ENZYME ACTIVITY ASSAY (plasma) DIAGNOSTIC — not genetic alone; "
            "TNF blockade (etanercept 0.4 mg/kg biweekly SC, or adalimumab) CURATIVE/disease-modifying: "
            "prevents >95% of strokes if started early; "
            "DO NOT use steroids alone (short-term help, no prevention); "
            "bone marrow failure: pure red cell aplasia, neutropenia, hypogammaglobulinaemia — HSCT curative; "
            "22q11.1 deletion may be missed by CMA → sequencing MANDATORY; "
            "avoid live vaccines before immunosuppression started"
        ),
        "aa": "511 aa",
        "kDa": "~59 kDa",
        "locus": "22q11.1",
        "omim_gene": 607575,
        "omim_disease": 615688,
        "inheritance": "AR — biallelic loss-of-function mutations",
        "gene_class": (
            "ADA2 (previously CECR1) encodes adenosine deaminase 2, a secreted enzyme structurally "
            "related to adenosine deaminase but primarily acting as a growth factor. ADA2 is expressed "
            "predominantly in monocytes/macrophages and promotes differentiation towards an M2 "
            "(anti-inflammatory, pro-healing) phenotype. It also maintains vascular endothelial "
            "integrity via adenosine-dependent and -independent signalling. "
            "Biallelic loss-of-function causes DADA2, characterised by a systemic vasculopathy "
            "with markedly elevated TNF-α. The mechanism involves dysfunctional M1-polarised "
            "macrophages damaging small and medium vessel walls, causing both ischaemic and "
            "haemorrhagic strokes — an unusual combination that is a diagnostic clue. "
            "Livedoid rash (livedo racemosa) accompanies strokes in >50% of paediatric cases. "
            "TNF blockade is transformative: early treatment prevents recurrent strokes in >95% of cases. "
            "Bone marrow failure (pure red cell aplasia, neutropenia) in a subset is curable by HSCT. "
            "ADA2 enzyme activity assay in plasma is the primary diagnostic test."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("DADA2 — childhood ischaemic + haemorrhagic strokes", 0.48),
            ("DADA2 — vasculopathy + bone marrow failure", 0.25),
            ("DADA2 — livedo racemosa + vasculopathy alone", 0.17),
            ("DADA2 — late-onset / mild alleles", 0.10),
        ],
        "onset_range": (2, 45),
        "key_alerts": [
            "DADA2-LIVEDO: Livedoid rash (livedo racemosa) + childhood stroke = DADA2 until proven otherwise",
            "ADA2-ENZYME-ASSAY: Plasma ADA2 enzyme activity DIAGNOSTIC — genetic testing confirms but enzyme first",
            "TNF-BLOCKADE-CURATIVE: Etanercept 0.4mg/kg biweekly prevents >95% strokes — start early",
            "NO-STEROIDS-ALONE: Steroids give short-term benefit but do NOT prevent recurrent strokes",
            "MIXED-STROKES: Ischaemic AND haemorrhagic strokes in same patient = DADA2 diagnostic clue",
            "BMF-MONITOR: Pure red cell aplasia/neutropenia — CBC monthly; HSCT curative for BMF",
            "22q11-CMA-MISS: 22q11.1 locus may not be captured by standard CMA — sequence MANDATORY",
            "NO-LIVE-VACCINES: Before diagnosis/immunotherapy established — avoid all live vaccines",
        ],
    },
    # ── CST3 — HCCAA ──
    {
        "gene": "CST3",
        "protein": "Cystatin C — Secreted Cysteine Protease Inhibitor",
        "alias": (
            "CST3; OMIM gene 604312; HCCAA OMIM 105150; 20p11.21; 146 aa; ~13 kDa; "
            "Cystatin C — secreted cysteine protease inhibitor; inhibits cathepsins B, H, L, S; "
            "Icelandic founder mutation L68Q (c.203T>A) — heterozygotes + rare homozygotes; "
            "HCCAA — Hereditary Cystatin C Amyloid Angiopathy (Icelandic type): "
            "L68Q variant — mutant cystatin C forms amyloid fibrils depositing in cerebral vessel walls; "
            "plasma cystatin C paradoxically LOW (sequestered in amyloid — opposite of renal failure marker); "
            "~100% ICH penetrance in homozygotes by age 30 (mean 30 years); "
            "heterozygotes: mean ICH age 50 years; "
            "NO anticoagulants (cerebral amyloid angiopathy — catastrophic ICH risk); "
            "NO antiplatelet agents acutely in active ICH; "
            "CAUTION: serum cystatin C (standard renal biomarker) ≠ CST3 genetic diagnosis — do not confuse; "
            "Iceland/Scandinavia highest prevalence; international cases reported"
        ),
        "aa": "146 aa",
        "kDa": "~13 kDa",
        "locus": "20p11.21",
        "omim_gene": 604312,
        "omim_disease": 105150,
        "inheritance": "AD — L68Q (c.203T>A) heterozygous (classic) or homozygous (very rare, severe)",
        "gene_class": (
            "CST3 encodes cystatin C, a 13 kDa secreted cysteine protease inhibitor abundantly "
            "expressed in the choroid plexus and all nucleated cells, excreted into CSF, plasma, "
            "and urine. Cystatin C inhibits lysosomal cathepsins and extracellular cysteine proteases. "
            "The Icelandic L68Q (c.203T>A) variant causes an Ala68Leu substitution that destabilises "
            "the fold and promotes amyloid fibril formation. These fibrils deposit in the tunica media "
            "and adventitia of leptomeningeal and cortical arteries, causing progressive vessel wall "
            "weakening and fatal ICH. Unlike renal disease (where serum cystatin C is elevated), "
            "HCCAA patients have paradoxically LOW plasma cystatin C because the mutant protein "
            "is rapidly incorporated into amyloid deposits. This LOW plasma cystatin C is a "
            "diagnostic biomarker clue. Homozygotes have near-complete penetrance by age 30; "
            "heterozygotes develop ICH at a mean of 50 years. No disease-modifying therapy exists; "
            "anticoagulants are absolutely contraindicated."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("HCCAA — L68Q heterozygote (typical)", 0.68),
            ("HCCAA — L68Q homozygote (severe, early)", 0.08),
            ("HCCAA — recurrent ICH, known L68Q carrier", 0.18),
            ("HCCAA — presymptomatic family screen", 0.06),
        ],
        "onset_range": (20, 70),
        "key_alerts": [
            "HCCAA-L68Q: Icelandic/Scandinavian ICH + family history → CST3 L68Q (c.203T>A) targeted test",
            "PLASMA-CYSTATIN-C-LOW: Paradoxically LOW (sequestered in amyloid) — diagnostic biomarker clue",
            "NO-ANTICOAGULANTS: Cerebral amyloid angiopathy — absolute contraindication in any setting",
            "NO-ANTIPLATELET-ACUTELY: Active ICH — hold all antithrombotics during acute haemorrhage",
            "HOMOZYGOTE-SEVERE: ~100% ICH penetrance by age 30 — presymptomatic monitoring from teen years",
            "HETEROZYGOTE-AGE-50: Mean ICH age 50 — surveillance MRI for microbleeds from age 40",
            "RENAL-CYSTATIN-CONFUSION: Serum cystatin C renal biomarker ≠ CST3 genetic diagnosis — document clearly",
            "CASCADE-Testing: All first-degree relatives — L68Q targeted sequencing + plasma cystatin C",
        ],
    },
]


def _make_patients(gene_def: dict) -> list[dict]:
    rng = random.Random(gene_def["seed"])
    gene = gene_def["gene"]
    patients = []
    for i in range(gene_def["n_patients"]):
        etiology, _ = rng.choices(
            gene_def["etiologies"],
            weights=[w for _, w in gene_def["etiologies"]],
            k=1,
        )[0], None
        etiology = rng.choices(
            [e for e, _ in gene_def["etiologies"]],
            weights=[w for _, w in gene_def["etiologies"]],
        )[0]
        age = rng.randint(*gene_def["onset_range"])

        # gene-specific patient features
        if gene == "NOTCH3":
            migraine = rng.random() < 0.55
            stroke_count = rng.randint(1, 4)
            gom_confirmed = rng.random() < 0.72
            dementia = age > 45 and rng.random() < 0.40
            p = {
                "patient_id": f"NOTCH3-{gene_def['seed']:04d}-{i+1:03d}",
                "gene": gene, "age_at_onset": age, "etiology": etiology,
                "migraine_with_aura": migraine,
                "stroke_count": stroke_count,
                "gom_confirmed_skin_biopsy": gom_confirmed,
                "dementia_present": dementia,
                "anterior_temporal_wml": rng.random() < 0.82,
                "external_capsule_wml": rng.random() < 0.78,
                "tpa_contraindicated": True,
                "ocp_contraindicated": migraine,
                "ecd_domain": rng.choices(
                    ["EGF1-6 (severe)", "EGF7-34 (moderate)", "EGF>34 (atypical)"],
                    weights=[0.48, 0.37, 0.15],
                )[0],
                "key_alert": gene_def["key_alerts"][rng.randint(0, len(gene_def["key_alerts"]) - 1)],
            }
        elif gene == "HTRA1":
            alopecia = rng.random() < 0.68
            spondylosis = rng.random() < 0.75
            biallelic = "AR" in etiology
            p = {
                "patient_id": f"HTRA1-{gene_def['seed']:04d}-{i+1:03d}",
                "gene": gene, "age_at_onset": age, "etiology": etiology,
                "premature_alopecia": alopecia,
                "lumbar_spondylosis": spondylosis,
                "biallelic_carasil": biallelic,
                "gom_on_biopsy": False,
                "wml_diffuse": rng.random() < 0.85,
                "no_conventional_risk_factors": rng.random() < 0.78,
                "key_alert": gene_def["key_alerts"][rng.randint(0, len(gene_def["key_alerts"]) - 1)],
            }
        elif gene == "COL4A1":
            porencephaly = "porencephaly" in etiology.lower() or "prenatal" in etiology.lower()
            anticoag_risk = True
            p = {
                "patient_id": f"COL4A1-{gene_def['seed']:04d}-{i+1:03d}",
                "gene": gene, "age_at_onset": age, "etiology": etiology,
                "porencephaly": porencephaly,
                "spontaneous_ich": "ICH" in etiology or rng.random() < 0.35,
                "retinal_arterial_tortuosity": rng.random() < 0.52,
                "anticoagulants_contraindicated": anticoag_risk,
                "contact_sports_restricted": True,
                "hanac_features": "HANAC" in etiology,
                "col4a2_tested_simultaneously": rng.random() < 0.81,
                "key_alert": gene_def["key_alerts"][rng.randint(0, len(gene_def["key_alerts"]) - 1)],
            }
        elif gene == "COL4A2":
            porencephaly = "porencephaly" in etiology.lower() or "perinatal" in etiology.lower()
            p = {
                "patient_id": f"COL4A2-{gene_def['seed']:04d}-{i+1:03d}",
                "gene": gene, "age_at_onset": age, "etiology": etiology,
                "porencephaly": porencephaly,
                "spontaneous_ich": "ICH" in etiology or rng.random() < 0.38,
                "anticoagulants_contraindicated": True,
                "haematuria_screen_done": rng.random() < 0.65,
                "col4a1_tested_simultaneously": rng.random() < 0.83,
                "subclinical_mri_only": "subclinical" in etiology.lower(),
                "key_alert": gene_def["key_alerts"][rng.randint(0, len(gene_def["key_alerts"]) - 1)],
            }
        elif gene == "TREX1":
            retinal_fa = rng.random() < 0.92
            mass_lesion = rng.random() < 0.72
            p = {
                "patient_id": f"TREX1-{gene_def['seed']:04d}-{i+1:03d}",
                "gene": gene, "age_at_onset": age, "etiology": etiology,
                "retinal_vasculopathy_confirmed_fa": retinal_fa,
                "gadolinium_mass_lesions": mass_lesion,
                "raynaud_phenomenon": rng.random() < 0.48,
                "hepatopathy": rng.random() < 0.35,
                "nephropathy": rng.random() < 0.32,
                "anaemia": rng.random() < 0.40,
                "immunosuppression_given_ineffective": rng.random() < 0.38,
                "c_terminal_frameshift": True,
                "key_alert": gene_def["key_alerts"][rng.randint(0, len(gene_def["key_alerts"]) - 1)],
            }
        elif gene == "GLA":
            classic_male = "classic" in etiology.lower() and "male" in etiology.lower()
            posterior_stroke = rng.random() < 0.62
            pulvinar = classic_male and rng.random() < 0.28
            p = {
                "patient_id": f"GLA-{gene_def['seed']:04d}-{i+1:03d}",
                "gene": gene, "age_at_onset": age, "etiology": etiology,
                "posterior_circulation_stroke": posterior_stroke,
                "pulvinar_sign_mri": pulvinar,
                "lvh_echo": rng.random() < 0.55,
                "cornea_verticillata": rng.random() < 0.68,
                "lyso_gb3_elevated": rng.random() < 0.88,
                "enzyme_activity_low": classic_male,
                "ert_started": rng.random() < 0.72,
                "migalastat_eligible": rng.random() < 0.28,
                "key_alert": gene_def["key_alerts"][rng.randint(0, len(gene_def["key_alerts"]) - 1)],
            }
        elif gene == "ADA2":
            livedo = rng.random() < 0.58
            ischemic_stroke = rng.random() < 0.75
            haemorrhagic_stroke = rng.random() < 0.42
            tnf_blockade = rng.random() < 0.65
            p = {
                "patient_id": f"ADA2-{gene_def['seed']:04d}-{i+1:03d}",
                "gene": gene, "age_at_onset": age, "etiology": etiology,
                "livedo_racemosa": livedo,
                "ischaemic_stroke": ischemic_stroke,
                "haemorrhagic_stroke": haemorrhagic_stroke,
                "ada2_enzyme_activity_confirmed": rng.random() < 0.88,
                "tnf_blockade_started": tnf_blockade,
                "bone_marrow_failure": "BMF" in etiology or rng.random() < 0.22,
                "steroids_used_alone_ineffective": rng.random() < 0.30,
                "key_alert": gene_def["key_alerts"][rng.randint(0, len(gene_def["key_alerts"]) - 1)],
            }
        else:  # CST3
            homozygote = "homozygote" in etiology.lower()
            cystatin_low = rng.random() < 0.78
            ich = "ICH" in etiology or rng.random() < 0.68
            p = {
                "patient_id": f"CST3-{gene_def['seed']:04d}-{i+1:03d}",
                "gene": gene, "age_at_onset": age, "etiology": etiology,
                "l68q_confirmed": True,
                "homozygote": homozygote,
                "ich_event": ich,
                "plasma_cystatin_c_low": cystatin_low,
                "anticoagulants_contraindicated": True,
                "microbleeds_mri": rng.random() < 0.55,
                "family_screen_done": rng.random() < 0.72,
                "key_alert": gene_def["key_alerts"][rng.randint(0, len(gene_def["key_alerts"]) - 1)],
            }
        patients.append(p)
    return patients


def _build_all_patients() -> list[dict]:
    all_pts: list[dict] = []
    for gd in CEREBRAL_SVD_GENES:
        all_pts.extend(_make_patients(gd))
    return all_pts


_PATIENTS: list[dict] | None = None


def _patients() -> list[dict]:
    global _PATIENTS
    if _PATIENTS is None:
        _PATIENTS = _build_all_patients()
    return _PATIENTS


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_overview() -> dict:
    pts = _patients()
    n = len(pts)  # 320

    notch3 = [p for p in pts if p["gene"] == "NOTCH3"]
    htra1  = [p for p in pts if p["gene"] == "HTRA1"]
    col4a1 = [p for p in pts if p["gene"] == "COL4A1"]
    col4a2 = [p for p in pts if p["gene"] == "COL4A2"]
    trex1  = [p for p in pts if p["gene"] == "TREX1"]
    gla    = [p for p in pts if p["gene"] == "GLA"]
    ada2   = [p for p in pts if p["gene"] == "ADA2"]
    cst3   = [p for p in pts if p["gene"] == "CST3"]

    def pct(lst, key, val=True):
        if not lst:
            return 0
        return round(100 * sum(1 for p in lst if p.get(key) == val) / len(lst), 1)

    def any_pct(all_pts, key, val=True):
        return round(100 * sum(1 for p in all_pts if p.get(key) == val) / len(all_pts), 1) if all_pts else 0

    aggregate_stats = {
        "total_patients": n,
        "n_genes": 8,
        "seeds": "1470-1477",
        "cohort_label": "320 patients (8×40, seeds 1470–1477)",
        # per-gene counts
        "notch3_cadasil_n": len(notch3),
        "htra1_carasil_ad_n": len(htra1),
        "col4a1_angiopathy_n": len(col4a1),
        "col4a2_angiopathy_n": len(col4a2),
        "trex1_rvcl_s_n": len(trex1),
        "gla_fabry_n": len(gla),
        "ada2_dada2_n": len(ada2),
        "cst3_hccaa_n": len(cst3),
        # NOTCH3 stats
        "notch3_gom_confirmed_pct": pct(notch3, "gom_confirmed_skin_biopsy"),
        "notch3_migraine_aura_pct": pct(notch3, "migraine_with_aura"),
        "notch3_anterior_temporal_wml_pct": pct(notch3, "anterior_temporal_wml"),
        "notch3_tpa_contraindicated_pct": 100.0,
        # HTRA1 stats
        "htra1_alopecia_pct": pct(htra1, "premature_alopecia"),
        "htra1_spondylosis_pct": pct(htra1, "lumbar_spondylosis"),
        "htra1_carasil_biallelic_pct": pct(htra1, "biallelic_carasil"),
        "htra1_gom_absent_pct": 100.0,
        # COL4A1 stats
        "col4a1_porencephaly_pct": pct(col4a1, "porencephaly"),
        "col4a1_retinal_tortuosity_pct": pct(col4a1, "retinal_arterial_tortuosity"),
        "col4a1_col4a2_tested_simultaneously_pct": pct(col4a1, "col4a2_tested_simultaneously"),
        "col4a1_anticoag_contraindicated_pct": 100.0,
        # COL4A2 stats
        "col4a2_porencephaly_pct": pct(col4a2, "porencephaly"),
        "col4a2_col4a1_tested_pct": pct(col4a2, "col4a1_tested_simultaneously"),
        "col4a2_anticoag_ci_pct": 100.0,
        # TREX1 stats
        "trex1_retinal_fa_confirmed_pct": pct(trex1, "retinal_vasculopathy_confirmed_fa"),
        "trex1_gadolinium_mass_lesions_pct": pct(trex1, "gadolinium_mass_lesions"),
        "trex1_raynaud_pct": pct(trex1, "raynaud_phenomenon"),
        "trex1_immunosuppression_failed_pct": pct(trex1, "immunosuppression_given_ineffective"),
        # GLA stats
        "gla_posterior_stroke_pct": pct(gla, "posterior_circulation_stroke"),
        "gla_pulvinar_sign_pct": pct(gla, "pulvinar_sign_mri"),
        "gla_ert_started_pct": pct(gla, "ert_started"),
        "gla_cornea_verticillata_pct": pct(gla, "cornea_verticillata"),
        "gla_lyso_gb3_elevated_pct": pct(gla, "lyso_gb3_elevated"),
        # ADA2 stats
        "ada2_livedo_racemosa_pct": pct(ada2, "livedo_racemosa"),
        "ada2_mixed_strokes_pct": round(
            100 * sum(1 for p in ada2 if p.get("ischaemic_stroke") and p.get("haemorrhagic_stroke")) / len(ada2), 1
        ) if ada2 else 0,
        "ada2_tnf_blockade_pct": pct(ada2, "tnf_blockade_started"),
        "ada2_bmf_pct": pct(ada2, "bone_marrow_failure"),
        # CST3 stats
        "cst3_ich_pct": pct(cst3, "ich_event"),
        "cst3_low_cystatin_c_pct": pct(cst3, "plasma_cystatin_c_low"),
        "cst3_anticoag_ci_pct": 100.0,
        "cst3_family_screen_pct": pct(cst3, "family_screen_done"),
        # cross-gene
        "anticoag_contraindicated_any_pct": round(
            100 * sum(1 for p in pts if p.get("anticoagulants_contraindicated")) / n, 1
        ),
    }

    top_alerts = [
        "NOTCH3-CADASIL: EGF-domain cysteine rule — odd-count within EGF 1-34 = pathogenic; NO tPA; NO OCP",
        "CARASIL-TRIAD: Alopecia + Spondylosis + Young Stroke (no RF) = HTRA1 biallelic until proven otherwise",
        "COL4A1/COL4A2-13q34: PORENCEPHALY = test BOTH genes simultaneously; NO anticoagulants; NO contact sports",
        "RVCL-S-TREX1: Retinal vasculopathy ALWAYS first; punctate Gd-enhancing lesions = NOT tumour/demyelination",
        "FABRY-GLA-TREATABLE: Most common treatable hereditary SVD; posterior stroke + pulvinar sign → GLA enzyme",
        "DADA2-ADA2: Livedo + childhood mixed strokes → enzyme activity assay; TNF blockade CURATIVE not steroids",
        "HCCAA-CST3: L68Q ICH; plasma cystatin C LOW (paradoxical); NO anticoagulants; family L68Q screen",
        "COL4A1/COL4A2-NO-ANTICOAGULANTS: Haemorrhagic angiopathy — anticoagulation causes catastrophic ICH",
    ]

    genes = [
        {
            "gene": gd["gene"],
            "protein": gd["protein"],
            "locus": gd["locus"],
            "aa": gd["aa"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "seed": gd["seed"],
        }
        for gd in CEREBRAL_SVD_GENES
    ]

    return {
        "atlas": "Hereditary-Cerebral-SVD-Atlas",
        "title": "Complete 8-Gene Hereditary Cerebral Small Vessel Disease Atlas",
        "aggregate_stats": aggregate_stats,
        "top_alerts": top_alerts,
        "genes": genes,
    }


def get_breakdown() -> dict:
    pts = _patients()
    breakdown: dict[str, dict] = {}

    for gd in CEREBRAL_SVD_GENES:
        gene = gd["gene"]
        gene_pts = [p for p in pts if p["gene"] == gene]
        n = len(gene_pts)

        # etiology distribution
        from collections import Counter
        etio_counts = Counter(p["etiology"] for p in gene_pts)

        breakdown[gene] = {
            "gene": gene,
            "protein": gd["protein"],
            "alias": gd["alias"],
            "locus": gd["locus"],
            "aa": gd["aa"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": n,
            "seed": gd["seed"],
            "etiologies": dict(etio_counts),
            "key_alerts": gd["key_alerts"],
            "gene_class": gd["gene_class"],
            "sample_patients": gene_pts[:6],
        }

    return {"atlas": "Hereditary-Cerebral-SVD-Atlas", "breakdown": breakdown}


def get_definitions() -> dict:
    return {
        "atlas": "Hereditary-Cerebral-SVD-Atlas",
        "definitions": [
            {
                "term": "Hereditary Cerebral SVD",
                "definition": (
                    "A group of single-gene disorders causing progressive disease of small cerebral arteries "
                    "and arterioles (diameter <300 µm), manifesting as ischaemic lacunar infarcts, "
                    "white matter lesions, cerebral microbleeds, and/or intracerebral haemorrhage. "
                    "Unlike sporadic SVD (driven by hypertension and diabetes), hereditary SVD presents "
                    "in younger individuals with a positive family history and requires genetic diagnosis."
                ),
            },
            {
                "term": "CADASIL (NOTCH3)",
                "definition": (
                    "Cerebral Autosomal Dominant Arteriopathy with Subcortical Infarcts and "
                    "Leukoencephalopathy. Most common hereditary SVD (~1:15000–50000). Caused by "
                    "EGF-like domain cysteine-altering mutations in NOTCH3. GOM deposits in vascular "
                    "SMCs are PATHOGNOMONIC. Clinical tetrad: migraine + aura, subcortical strokes, "
                    "cognitive decline, psychiatric disturbance. Anterior temporal WML + external capsule "
                    "involvement PATHOGNOMONIC on MRI. NO tPA; NO OCP."
                ),
            },
            {
                "term": "CARASIL (HTRA1 biallelic)",
                "definition": (
                    "Cerebral Autosomal Recessive Arteriopathy with Subcortical Infarcts and "
                    "Leukoencephalopathy. AR HTRA1 biallelic LOF. GOM ABSENT on EM. "
                    "PATHOGNOMONIC triad: premature alopecia + lumbar/cervical spondylosis + young-onset "
                    "stroke (20s–40s) WITHOUT conventional vascular risk factors. "
                    "Distinguished from CADASIL by: recessive inheritance, absence of GOM, alopecia, "
                    "and spondylosis."
                ),
            },
            {
                "term": "COL4A1/COL4A2 Angiopathy",
                "definition": (
                    "AD vascular basement membrane disorders caused by Gly-X-Y triple-helix missense "
                    "mutations. COL4A1 and COL4A2 share the same 13q34 locus; ALWAYS test both "
                    "simultaneously. Porencephaly (perinatal/prenatal ICH causing cerebral cyst) is "
                    "the commonest genetic cause. Anticoagulants are ABSOLUTELY CONTRAINDICATED. "
                    "Retinal arterial tortuosity in >50% COL4A1 carriers."
                ),
            },
            {
                "term": "RVCL-S (TREX1)",
                "definition": (
                    "Retinal Vasculopathy with Cerebral Leukoencephalopathy and Systemic manifestations. "
                    "AD TREX1 C-terminal truncating mutations. Retinal vasculopathy ALWAYS precedes "
                    "brain involvement. PATHOGNOMONIC MRI: punctate/ring gadolinium-enhancing mass-like "
                    "white matter lesions — often misdiagnosed as tumour or demyelination. "
                    "Systemic features: Raynaud, hepatopathy, nephropathy. "
                    "No curative therapy; immunosuppression ineffective."
                ),
            },
            {
                "term": "Fabry Disease (GLA) — Cerebrovascular",
                "definition": (
                    "XLD lysosomal storage disorder due to alpha-galactosidase A deficiency. "
                    "The most common TREATABLE hereditary SVD. Posterior circulation lacunar strokes. "
                    "Pulvinar sign (bilateral T1 MRI hyperintensity) PATHOGNOMONIC in 25–30% males. "
                    "ERT (agalsidase) or migalastat (amenable missense only) is disease-modifying. "
                    "Lyso-Gb3 in plasma is the diagnostic biomarker in females (enzyme may be normal)."
                ),
            },
            {
                "term": "DADA2 (ADA2)",
                "definition": (
                    "Deficiency of Adenosine Deaminase 2. AR ADA2 biallelic LOF. Childhood systemic "
                    "vasculopathy with BOTH ischaemic AND haemorrhagic strokes in the same patient — "
                    "diagnostic clue. Livedoid rash (livedo racemosa) PATHOGNOMONIC + stroke. "
                    "ADA2 enzyme activity assay DIAGNOSTIC. TNF blockade CURATIVE (prevents >95% strokes). "
                    "DO NOT use steroids alone. Bone marrow failure subset: HSCT curative."
                ),
            },
            {
                "term": "HCCAA (CST3)",
                "definition": (
                    "Hereditary Cystatin C Amyloid Angiopathy — Icelandic type (L68Q). AD CST3 mutation. "
                    "Mutant cystatin C forms amyloid in cerebral vessel walls → recurrent ICH. "
                    "Plasma cystatin C paradoxically LOW (sequestered in amyloid). "
                    "~100% ICH penetrance by age 30 in homozygotes; mean age 50 in heterozygotes. "
                    "NO anticoagulants (cerebral amyloid angiopathy). L68Q targeted sequencing in "
                    "Icelandic/Scandinavian families with recurrent ICH."
                ),
            },
            {
                "term": "GOM (Granular Osmiophilic Material)",
                "definition": (
                    "Electron-dense granular deposits in the basal lamina of vascular smooth muscle cells, "
                    "PATHOGNOMONIC for CADASIL. Detected by electron microscopy on skin biopsy (arterioles "
                    "in dermis). Anti-NOTCH3 ECD immunostaining is an alternative. GOM is ABSENT in "
                    "CARASIL — this distinction is diagnostically critical."
                ),
            },
            {
                "term": "Pulvinar Sign",
                "definition": (
                    "Bilateral T1-weighted MRI hyperintensity in the posterior thalamus (pulvinar nuclei), "
                    "seen in 25–30% of males with Fabry disease (GLA deficiency). Caused by Gb3 "
                    "accumulation and/or calcium deposition in thalamic vessels. PATHOGNOMONIC for Fabry "
                    "disease when present in the correct clinical context (young stroke, cryptogenic)."
                ),
            },
            {
                "term": "13q34 Two-Gene Panel",
                "definition": (
                    "COL4A1 and COL4A2 share the chromosomal locus 13q34 and must always be sequenced "
                    "as a pair. Ordering COL4A1-only sequencing is clinically insufficient. Porencephaly, "
                    "spontaneous ICH, and hereditary SVD workup in any patient must include both genes."
                ),
            },
            {
                "term": "Posterior Circulation Lacunar Stroke — Fabry",
                "definition": (
                    "Vertebrobasilar territory lacunar infarcts are the hallmark cerebrovascular "
                    "manifestation of Fabry disease, contrasting with the anterior circulation preference "
                    "of hypertensive SVD. Any posterior circulation lacunar stroke in a patient <65, "
                    "especially with cryptogenic workup, should trigger GLA enzyme activity (males) "
                    "or lyso-Gb3 (females) testing."
                ),
            },
            {
                "term": "Anticoagulation Contraindication in Hereditary Haemorrhagic SVD",
                "definition": (
                    "COL4A1, COL4A2, and CST3 (HCCAA) cause haemorrhagic cerebral vasculopathy. "
                    "Anticoagulation (warfarin, DOACs, heparin) is ABSOLUTELY CONTRAINDICATED in all "
                    "three. Even for concomitant atrial fibrillation, risks must be weighed carefully "
                    "and anticoagulation avoided if possible. Antiplatelet agents may be used cautiously "
                    "outside active ICH."
                ),
            },
            {
                "term": "Cascade Testing — Hereditary Cerebral SVD",
                "definition": (
                    "All first-degree relatives of an index case with any hereditary SVD diagnosis should "
                    "be offered: genetic counselling, targeted gene sequencing (or full panel if proband "
                    "mutation confirmed), brain MRI (WML/microbleeds), and disease-specific surveillance "
                    "(fundoscopy for COL4A1/GLA, enzyme activity for GLA/ADA2, etc.). Presymptomatic "
                    "diagnosis enables preventive treatment (TNF blockade in DADA2, ERT in Fabry) "
                    "before irreversible organ damage."
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print(json.dumps(ov["aggregate_stats"], indent=2))
    print(f"\nTop alerts ({len(ov['top_alerts'])}):")
    for a in ov["top_alerts"]:
        print(f"  • {a}")
