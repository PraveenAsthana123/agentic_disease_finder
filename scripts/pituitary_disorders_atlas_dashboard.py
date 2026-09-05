#!/usr/bin/env python3
"""Pituitary-Disorders-Atlas — Complete 8-Gene Hereditary Pituitary Disorders Atlas
AIP     (Aryl hydrocarbon receptor Interacting Protein; 330 aa; 11q13.2; OMIM gene 605555;
         FIPA — Familial Isolated Pituitary Adenoma; AD haploinsufficiency; predominantly GH-secreting;
         young-onset acromegaly/gigantism (<30 years); macroadenoma at diagnosis; somatostatin analogs
         LESS effective than sporadic; pegvisomant GH-receptor blocker preferred for resistant cases;
         screen first-degree relatives with MRI from age 10) ·
PRKAR1A (Protein kinase A regulatory subunit 1α; 381 aa; 17q24.2; OMIM gene 188830;
         Carney Complex #160980; AD LOF; lentigines + CARDIAC MYXOMA + GH adenoma + PPNAD Cushing's +
         large cell calcifying Sertoli tumors + schwannomas; Cardiac myxoma can occur in ANY chamber —
         echocardiography MANDATORY annually; PPNAD = ACTH-independent Cushing's — dexamethasone
         paradoxical cortisol RISE; CNC diagnosis = any 2 major criteria) ·
PROP1   (Paired-like homeobox 1; 226 aa; 5q35.3; OMIM gene 601538; CPHD2 #262600; AR biallelic;
         most common AR cause of CPHD worldwide; LOF → GH + TSH + LH/FSH + PRL deficiency;
         ACTH deficiency develops LATE — screen annually; unique: pituitary HYPERPLASIA early on MRI
         (do NOT biopsy — it involutes); start thyroxine + GH + sex steroids at appropriate age) ·
POU1F1  (Pit-1 transcription factor; 291 aa; 3p11.2; OMIM gene 173110; CPHD1 #613038; AD or AR;
         CPHD1 = GH + TSH + PRL deficiency ONLY — NOT gonadotrophs, NOT ACTH; start THYROXINE FIRST
         before GH to avoid unmasking of subclinical adrenal insufficiency) ·
LHX3    (LIM homeobox 3; 397 aa; 9q34.3; OMIM gene 600577; CPHD3 #221750; AR biallelic;
         CPHD + RIGID CERVICAL SPINE pathognomonic — rotation restricted to <90°;
         sensorineural hearing loss in subset; DO NOT manipulate cervical spine;
         X-ray cervical spine before MRI; GH + TSH + LH/FSH + PRL deficiency) ·
HESX1   (Homeobox gene expressed in ES cells; 185 aa; 3p14.3; OMIM gene 601802; SOD/CPHD #182230;
         AD haploinsufficiency or AR biallelic; septo-optic dysplasia (SOD) — absent septum pellucidum
         + optic nerve hypoplasia + CPHD; optic hypoplasia in neonate → pendular nystagmus →
         screen pituitary immediately; absent septum pellucidum alone does NOT confirm SOD) ·
GLI2    (GLI family zinc finger 2; 1586 aa; 2q14.2; OMIM gene 165230; HPE9/CPHD #610829;
         AD haploinsufficiency; hedgehog signaling TF; SINGLE CENTRAL MAXILLARY INCISOR (SCMI)
         PATHOGNOMONIC — even without HPE; SCMI in child → brain MRI + full pituitary hormone testing;
         HPE spectrum from alobar to microform) ·
CABLES1 (CDK5 and ABL1 enzyme substrate 1; 509 aa; 18q11.2; OMIM gene 607516;
         familial corticotropinoma — ACTH-secreting pituitary adenoma — Cushing's disease;
         AD haploinsufficiency; if transsphenoidal surgery fails → bilateral adrenalectomy;
         MANDATORY Nelson syndrome surveillance post-adrenalectomy — ACTH-driven rapid tumor growth)
320-patient aggregate cohort (8 x 40, seeds 1310-1317)
"""

import random

SEED_BASE = 1310

PITUITARY_GENES = [
    # ── AIP — Familial Isolated Pituitary Adenoma (FIPA) ──
    {
        "gene": "AIP",
        "protein": "Aryl Hydrocarbon Receptor Interacting Protein (AIP)",
        "alias": (
            "AIP; OMIM gene 605555; FIPA (Familial Isolated Pituitary Adenoma) AD-haploinsufficiency; 11q13.2; "
            "330 aa; ~37 kDa; co-chaperone that interacts with Hsp90 and PDE4A5; "
            "negative regulator of GH secretion via cAMP phosphodiesterase stabilization; "
            "LOF → loss of negative regulation → GH-secreting somatotroph hyperproliferation → "
            "gigantism in childhood / acromegaly in adults; "
            "AIP-mutant adenomas: younger onset (<30 yr), LARGER at diagnosis (macroadenoma ≥10 mm), "
            "LESS responsive to somatostatin analogs (SSA) than sporadic; "
            "pegvisomant (GH-receptor antagonist) preferred for SSA-resistant disease; "
            "prevalence: AIP mutation in ~15-20% of FIPA families, ~3% of all pituitary adenomas <35 yr"
        ),
        "aa": "330 aa",
        "kDa": "~37 kDa",
        "locus": "11q13.2",
        "omim_gene": 605555,
        "omim_disease": 605555,
        "inheritance": (
            "AD haploinsufficiency; dominant negative variants also described; "
            "penetrance ~23% in families (many carriers never develop adenoma); "
            "de novo variants in ~50% of pediatric gigantism cases; "
            "screen all first-degree relatives from age 10 — MRI + IGF-1 annually"
        ),
        "gene_class": (
            "AIP encodes the aryl hydrocarbon receptor-interacting protein, a co-chaperone of Hsp90. "
            "In somatotrophs, AIP stabilizes PDE4A5 (phosphodiesterase) which degrades cAMP. "
            "LOF → loss of PDE4A5 stabilization → elevated cAMP → PKA activation → "
            "somatotroph proliferation + excess GH secretion. "
            "PATHOPHYSIOLOGY: AIP-mutant tumors suppress Gαi2 signaling and overexpress ZAC1 and FGFR4, "
            "providing partial mechanistic explanation for SSA resistance. "
            "CLINICAL IMPLICATION: start with SSA, but escalate to PEGVISOMANT early if IGF-1 does not "
            "normalize within 6 months (IGF-1 normalization rate with SSA only ~30% vs 50% sporadic). "
            "SURVEILLANCE: family members with germline AIP variant: annual IGF-1 + MRI every 2 years "
            "from age 10; earlier if symptomatic (headache, visual field change)."
        ),
        "phenotype": (
            "Gigantism (pre-pubertal onset) or acromegaly (post-pubertal): tall stature, coarse facial features, "
            "enlarged hands/feet, macroadenoma ≥10 mm at diagnosis in >80%; "
            "headache (tumor mass effect), bitemporal hemianopia if suprasellar extension; "
            "elevated GH (not suppressed to <1 μg/L on OGTT), elevated IGF-1 (>2 SD for age/sex); "
            "hyperprolactinemia from pituitary stalk compression; "
            "ACTH and TSH deficiencies can occur from mass effect (assess all axes at diagnosis); "
            "Carney Complex NOT present (no lentigines, no myxoma — AIP ≠ PRKAR1A)"
        ),
        "key_hallmarks": [
            "AIP-mutant GH adenoma: younger (<30 yr), BIGGER (macroadenoma), MORE SSA-resistant",
            "First-degree relatives: MRI + IGF-1 from age 10 annually",
            "Pegvisomant preferred over SSA if IGF-1 not normalized at 6 months",
            "Trans-sphenoidal surgery first-line regardless — SSA does NOT shrink AIP tumors reliably",
            "Do NOT delay surgery hoping SSA will reduce tumor — it usually will not",
        ],
        "treatment_alerts": [
            "SSA (octreotide/lanreotide) LESS effective in AIP — switch to pegvisomant if IGF-1 persists",
            "Surgery first-line: trans-sphenoidal resection; adjuvant radiotherapy for residual",
            "Cabergoline adjuvant if mild hyperprolactinemia component",
            "Assess ALL pituitary axes at diagnosis — mass effect can cause panhypopituitarism",
        ],
        "ddx": [
            "Sporadic GH adenoma: later onset, smaller, better SSA response — AIP germline testing if <35 yr",
            "MEN1: also GH-secreting but also pNET + pHPT + pituitary (different syndrome)",
            "PRKAR1A/Carney Complex: also GH adenoma but lentigines + cardiac myxoma present",
            "McCune-Albright/GNAS: somatic, fibrous dysplasia + cafe-au-lait + polyostotic — no germline",
        ],
        "seed": SEED_BASE + 0,
        "n_patients": 40,
        "age_range": (7, 38),
        "female_pct": 42,
    },
    # ── PRKAR1A — Carney Complex ──
    {
        "gene": "PRKAR1A",
        "protein": "Protein Kinase cAMP-dependent Type I Regulatory Subunit Alpha",
        "alias": (
            "PRKAR1A; OMIM gene 188830; Carney Complex #160980; AD LOF; 17q24.2; 381 aa; ~43 kDa; "
            "regulatory subunit of protein kinase A (PKA); inhibits PKA catalytic subunits in absence of cAMP; "
            "LOF → constitutive PKA activation → endocrine and non-endocrine tumor formation; "
            "Carney Complex (CNC) = lentigines + cardiac myxoma + pituitary GH adenoma + "
            "PPNAD (primary pigmented nodular adrenocortical disease) + "
            "large cell calcifying Sertoli cell tumors (LCCSCT) + psammomatous melanotic schwannoma; "
            "Cardiac myxoma: ANY cardiac chamber, recur after surgery, risk of embolism/sudden death → "
            "ECHOCARDIOGRAPHY MANDATORY ANNUALLY"
        ),
        "aa": "381 aa",
        "kDa": "~43 kDa",
        "locus": "17q24.2",
        "omim_gene": 188830,
        "omim_disease": 160980,
        "inheritance": (
            "AD LOF (haploinsufficiency); ~80% penetrance by age 50; "
            "de novo in ~30%; familial with variable expressivity; "
            "diagnosis: 2 of 11 major criteria OR 1 major + confirmed PRKAR1A variant OR "
            "1 major + affected first-degree relative"
        ),
        "gene_class": (
            "PRKAR1A encodes the R1α regulatory subunit of PKA. Under basal conditions, "
            "R1α binds and inhibits the catalytic subunits (PKA-C). "
            "When cAMP rises (e.g., ACTH stimulation), cAMP binds R1α → releases PKA-C → downstream signaling. "
            "LOF: R1α absent → PKA-C constitutively active even without cAMP → "
            "autonomous cortisol production (PPNAD), GH excess, proliferation in multiple cell types. "
            "PPNAD PATHOGNOMONIC FEATURE: Liddle maneuver — dexamethasone PARADOXICALLY INCREASES cortisol "
            "(mechanism: suppressing ACTH removes remaining cAMP input, unmasking constitutive PKA; "
            "also: PPNAD micro-nodules have PRKAR1A LOH, highly sensitive to any cAMP change). "
            "CLINICAL: Cushing's syndrome from PPNAD is ACTH-independent, low-dose AND high-dose "
            "dexamethasone suppression tests BOTH FAIL to suppress cortisol. "
            "CARDIAC MYXOMA: can occur in any cardiac chamber, multiple, recurrent after resection — "
            "annual echocardiography is MANDATORY regardless of prior negative echo."
        ),
        "phenotype": (
            "Lentigines (spotty skin pigmentation): small, dark, perioral/periorbital/lips — unlike café-au-lait; "
            "Cardiac myxoma: left/right atrium or ventricle, polypoid, pedunculated, risk of embolism and sudden death; "
            "PPNAD: bilateral adrenal micronodular disease → ACTH-independent Cushing's, "
            "low normal to mildly elevated 24h UFC, atypical Cushing's features (young, thin, bruising); "
            "GH-secreting pituitary macroadenoma (acromegaly); "
            "LCCSCT in males (intratubular, calcifying, Reinke crystal-free); "
            "Psammomatous melanotic schwannomas (paraspinal, GI); "
            "Blue nevi, breast myxoid fibroadenoma"
        ),
        "key_hallmarks": [
            "Cardiac myxoma in ANY chamber + lentigines = Carney Complex until proven otherwise",
            "ECHOCARDIOGRAPHY MANDATORY annually — myxomas recur after resection",
            "PPNAD: dexamethasone paradoxically INCREASES cortisol (Liddle test positive)",
            "ACTH-independent Cushing's in young patient → PRKAR1A",
            "Screen all first-degree relatives for CNC criteria",
        ],
        "treatment_alerts": [
            "Cardiac myxoma resection: surgery URGENT (embolism risk — do NOT delay for genetics)",
            "Annual echocardiography MANDATORY — myxomas recur in up to 15% after surgery",
            "PPNAD Cushing's: bilateral adrenalectomy curative; hydrocortisone replacement lifelong",
            "GH adenoma: trans-sphenoidal surgery; SSA may be used adjuvantly",
            "Annual surveillance: echo + adrenal function + IGF-1 + testicular USS in males",
        ],
        "ddx": [
            "AIP: GH adenoma without lentigines/myxoma/Cushing's — no PRKAR1A",
            "MEN1: pituitary + pNET + pHPT (no lentigines, no cardiac myxoma)",
            "Sporadic cardiac myxoma: isolated, no lentigines, no CNC criteria",
            "Isolated PPNAD: some cases have PRKAR1A variant but no other CNC features",
        ],
        "seed": SEED_BASE + 1,
        "n_patients": 40,
        "age_range": (8, 55),
        "female_pct": 58,
    },
    # ── PROP1 — CPHD type 2 ──
    {
        "gene": "PROP1",
        "protein": "Paired-Like Homeobox 1 (Prop1 / Prophet of Pit-1)",
        "alias": (
            "PROP1; OMIM gene 601538; CPHD2 #262600; AR biallelic; 5q35.3; 226 aa; ~27 kDa; "
            "paired-type homeobox transcription factor; required for Pit-1 lineage specification "
            "(somatotrophs, thyrotrophs, lactotrophs) AND gonadotroph development; "
            "most common AR cause of CPHD worldwide; LOF → GH + TSH + LH/FSH + PRL deficiency; "
            "ACTH deficiency in subset (variable, often late onset); "
            "UNIQUE: early MRI shows pituitary HYPERPLASIA (mimics adenoma) → involution later; "
            "DO NOT biopsy — the mass will resolve; GH deficiency most severe at presentation"
        ),
        "aa": "226 aa",
        "kDa": "~27 kDa",
        "locus": "5q35.3",
        "omim_gene": 601538,
        "omim_disease": 262600,
        "inheritance": (
            "AR biallelic; most common variant: 301-302delAG (c.2delAG) in exon 2; "
            "also: R120C, F117I, S167T common missense; "
            "heterozygous carriers usually unaffected; "
            "consanguinity increases likelihood; "
            "prevalence: PROP1 is the most prevalent known cause of AR combined hormone deficiency"
        ),
        "gene_class": (
            "PROP1 (Prophet of Pit-1) encodes a paired-like homeodomain TF essential for two sequential "
            "steps in pituitary development: (1) specification of the Pit-1 cell lineage "
            "(somatotrophs/GH, thyrotrophs/TSH, lactotrophs/PRL) via regulating POU1F1 expression; "
            "(2) later specification of gonadotrophs (LH/FSH-secreting cells). "
            "PROP1 is also required for clearance of Rathke's cleft cells — failure to clear them "
            "contributes to the transient PITUITARY HYPERPLASIA seen on MRI in neonates/infants. "
            "HORMONAL SEQUENCE: GH deficiency appears first → causes growth failure; "
            "TSH deficiency → hypothyroidism (often partial); LH/FSH → absent/delayed puberty; "
            "PRL usually low; ACTH deficiency variable (may appear decades later — screen annually). "
            "MANAGEMENT: Replace GH early (critical window for brain development); "
            "start THYROXINE early; sex steroids at appropriate age; "
            "monitor for late-onset ACTH deficiency with annual testing."
        ),
        "phenotype": (
            "Severe GH deficiency: short stature (<-2.5 SD) + delayed bone age; "
            "hypothyroidism: fatigue, constipation, low T4/high TSH initially (then low TSH with low T4 as "
            "thyrotroph fails); hypogonadotropic hypogonadism: absent puberty, primary amenorrhea/azoospermia; "
            "PRL deficiency: failure of lactation in females; "
            "late ACTH deficiency (10-30%): fatigue, hypoglycaemia, hypotension, hyponatraemia; "
            "MRI: pituitary hyperplasia in childhood → normal or small gland in adulthood"
        ),
        "key_hallmarks": [
            "PROP1 = most common AR cause of combined pituitary hormone deficiency worldwide",
            "MRI pituitary hyperplasia early → involution: DO NOT BIOPSY",
            "Sequence: GH → TSH → LH/FSH → PRL deficiency; ACTH late (screen annually)",
            "Replace GH + thyroxine first; add sex steroids at pubertal age",
            "Annual ACTH/cortisol testing lifelong — even if normal initially",
        ],
        "treatment_alerts": [
            "Do NOT biopsy pituitary hyperplasia in PROP1 patient — it involutes spontaneously",
            "Start GH replacement early (critical for brain maturation and linear growth)",
            "Annual morning cortisol or insulin tolerance test — ACTH deficiency can emerge late",
            "Sex hormone replacement at age 11-12 for pubertal induction",
            "Fertility: FSH/LH replacement (gonadotropin therapy) required for pregnancy",
        ],
        "ddx": [
            "POU1F1/CPHD1: GH+TSH+PRL only (NOT gonadotrophs) — similar but distinct",
            "LHX3/CPHD3: adds rigid cervical spine pathognomonic",
            "HESX1: adds optic nerve hypoplasia + absent septum pellucidum",
            "Craniopharyngioma: acquired CPHD, MRI mass with calcification, no family history",
        ],
        "seed": SEED_BASE + 2,
        "n_patients": 40,
        "age_range": (1, 30),
        "female_pct": 50,
    },
    # ── POU1F1 — CPHD type 1 ──
    {
        "gene": "POU1F1",
        "protein": "POU Class 1 Homeobox 1 (Pit-1)",
        "alias": (
            "POU1F1 (Pit-1); OMIM gene 173110; CPHD1 #613038; AD (dominant negative) or AR biallelic; "
            "3p11.2; 291 aa; ~33 kDa; POU-domain homeodomain transcription factor; "
            "directly activates GH, TSH, PRL gene promoters; essential for somatotroph, thyrotroph, "
            "lactotroph terminal differentiation; LOF → CPHD1: GH + TSH + PRL deficiency SPECIFICALLY; "
            "gonadotrophs (LH/FSH) and corticotrophs (ACTH) are SPARED; "
            "AD dominant negative: heterozygous p.R271W or p.R172Q produces misfolded Pit-1 that sequesters "
            "wild-type → dominant negative haploinsufficiency"
        ),
        "aa": "291 aa",
        "kDa": "~33 kDa",
        "locus": "3p11.2",
        "omim_gene": 173110,
        "omim_disease": 613038,
        "inheritance": (
            "AD (dominant negative; p.R271W most common dominant-negative variant) or AR biallelic; "
            "AD cases: heterozygous, variable penetrance, de novo or familial; "
            "AR cases: consanguineous families, more severe; "
            "carrier parents of AR usually normal or mildly affected"
        ),
        "gene_class": (
            "POU1F1 (Pit-1) encodes a POU-homeodomain TF expressed in somatotrophs, thyrotrophs, and "
            "lactotrophs from embryonic day E17.5 in rodents (equivalent human timing). "
            "Pit-1 directly binds and transactivates GH1, TSHB, and PRL gene promoters. "
            "It is also essential for the expansion of these cell types after specification "
            "(Pit-1 mutations cause not just deficient hormone gene expression but also "
            "hypoplasia of these cell types). "
            "DOMINANT NEGATIVE MECHANISM: p.R271W (Arg271Trp) in the homeodomain — "
            "mutant Pit-1 protein can still dimerize with normal Pit-1 via POU domain "
            "but cannot bind DNA → dominant inhibition; single allele sufficient to cause disease. "
            "CRITICAL MANAGEMENT RULE: In POU1F1 CPHD1, ACTH axis is INTACT. "
            "However, thyroid hormone deficiency can mask relative cortisol demand. "
            "Starting GH before adequate thyroxine replacement can accelerate cortisol clearance "
            "and precipitate adrenal insufficiency crisis. "
            "RULE: THYROXINE FIRST (normalize T4), THEN ADD GH."
        ),
        "phenotype": (
            "GH deficiency: growth failure from infancy, short stature; "
            "TSH deficiency: central hypothyroidism — T4 low, TSH inappropriately low/normal; "
            "PRL deficiency: undetectable basal PRL, failure of postpartum lactation; "
            "PUBERTY NORMAL (LH/FSH intact — distinguishes from PROP1); "
            "ACTH/cortisol NORMAL (corticotrophs intact); "
            "anterior pituitary: hypoplastic on MRI (reduced volume of somatotroph/thyrotroph/lactotroph mass); "
            "ectopic posterior pituitary (bright spot) may be seen"
        ),
        "key_hallmarks": [
            "POU1F1: GH + TSH + PRL deficiency ONLY — puberty NORMAL, ACTH NORMAL",
            "Thyroxine FIRST before GH to avoid cortisol crisis (accelerated clearance by GH)",
            "Dominant negative p.R271W: single heterozygous variant sufficient to cause CPHD1",
            "Undetectable PRL is a diagnostic clue (most other causes have some PRL)",
            "Anterior pituitary hypoplasia on MRI — no mass, no cyst",
        ],
        "treatment_alerts": [
            "THYROXINE FIRST — then add GH replacement; do NOT start GH in hypothyroid state",
            "GH replacement: standard dosing (weight-based in children)",
            "No need for sex hormone replacement (LH/FSH axis intact)",
            "No need for hydrocortisone (ACTH axis intact) unless separately confirmed",
            "Lifelong surveillance: annual IGF-1, T4/fT4, PRL levels",
        ],
        "ddx": [
            "PROP1: also TSH+GH+PRL but ADDS LH/FSH deficiency and late ACTH — pituitary hyperplasia",
            "HESX1: adds optic nerve hypoplasia + septal defect",
            "Craniopharyngioma: acquired, MRI mass + calcification",
            "Isolated GH deficiency (GHR, GH1): normal TSH and PRL — POU1F1 affects all three",
        ],
        "seed": SEED_BASE + 3,
        "n_patients": 40,
        "age_range": (0, 25),
        "female_pct": 52,
    },
    # ── LHX3 — CPHD type 3 ──
    {
        "gene": "LHX3",
        "protein": "LIM Homeobox 3",
        "alias": (
            "LHX3; OMIM gene 600577; CPHD3 #221750; AR biallelic; 9q34.3; 397 aa; ~44 kDa; "
            "LIM-homeodomain transcription factor expressed in anterior pituitary + motor neurons + inner ear; "
            "LOF → CPHD3: GH + TSH + LH/FSH + PRL deficiency (ACTH usually spared, same as PROP1); "
            "PATHOGNOMONIC: RIGID CERVICAL SPINE — cervical rotation restricted to <90° "
            "(due to anomalous atlas/axis/C-spine bony fusion or ligamentous rigidity); "
            "sensorineural hearing loss in ~25% (inner ear expression); "
            "X-ray cervical spine BEFORE MRI (to protect rigid spine from manipulation)"
        ),
        "aa": "397 aa",
        "kDa": "~44 kDa",
        "locus": "9q34.3",
        "omim_gene": 600577,
        "omim_disease": 221750,
        "inheritance": (
            "AR biallelic; both parents carriers (usually unaffected); "
            "consanguinity increases prevalence; "
            "common variants include deletions and missense in LIM or homeodomain; "
            "rare AD dominant negative variants also reported but AR is the rule"
        ),
        "gene_class": (
            "LHX3 encodes a LIM-homeodomain TF with dual expression: (1) Rathke's pouch and anterior pituitary "
            "(required for Pit-1 lineage expansion and gonadotroph development, similar to PROP1 but independent); "
            "(2) motor neurons of spinal cord and brainstem (mutations may contribute to tone/motor issues); "
            "(3) inner ear (cochlear development → SNHL). "
            "RIGID CERVICAL SPINE MECHANISM: LHX3 is expressed in paraxial mesoderm derivatives; "
            "LOF during vertebral segmentation → abnormal atlantoaxial or subaxial joint development → "
            "restricted rotation <90° (normally >90°). This is the most consistent and "
            "most PATHOGNOMONIC extrasellar feature. "
            "CLINICAL RULE: Before performing any cervical MRI, obtain plain X-ray to document C-spine anatomy. "
            "Do NOT perform cervical manipulation or Dix-Hallpike maneuver without checking for "
            "atlantoaxial instability. "
            "HORMONE DEFICIENCY: same pattern as PROP1 (GH + TSH + LH/FSH + PRL), with ACTH typically intact. "
            "Early hormone replacement is critical to prevent intellectual disability and growth failure."
        ),
        "phenotype": (
            "Severe combined pituitary hormone deficiency: growth failure + hypothyroidism + absent puberty; "
            "Rigid cervical spine: rotation limited to <90° (normal >90°), stiff neck from birth or early childhood; "
            "Sensorineural hearing loss: present in ~25%, variable severity, congenital or progressive; "
            "MRI pituitary: hypoplastic anterior lobe, absent or ectopic posterior pituitary; "
            "Possible atlantoaxial instability: cord compression risk with neck manipulation; "
            "Normal intelligence in uncomplicated cases; "
            "Possible scoliosis (LHX3 motor neuron expression)"
        ),
        "key_hallmarks": [
            "RIGID CERVICAL SPINE + CPHD = LHX3 until proven otherwise — rotation <90° pathognomonic",
            "X-ray cervical spine BEFORE MRI — do NOT manipulate rigid spine (instability risk)",
            "DO NOT perform cervical manipulation, Dix-Hallpike, or chiropractic maneuvers",
            "Screen for sensorineural hearing loss (25%) at diagnosis",
            "Hormone deficiency: GH + TSH + LH/FSH + PRL (ACTH usually spared)",
        ],
        "treatment_alerts": [
            "Cervical spine X-ray first — establish anatomy before any neck procedure or imaging",
            "Avoid cervical manipulation and high-impact sports until C-spine stability confirmed",
            "Start GH + thyroxine replacement early (critical for growth/brain development)",
            "Sex hormone replacement at pubertal age (LH/FSH deficient)",
            "Annual ACTH testing (usually spared but not always)",
        ],
        "ddx": [
            "PROP1: similar hormone deficiency pattern but NO rigid cervical spine",
            "POU1F1: GH+TSH+PRL only, puberty normal — no neck rigidity",
            "Klippel-Feil: rigid neck from vertebral fusion but NORMAL pituitary (no CPHD)",
            "HESX1: optic hypoplasia + absent septum pellucidum — no rigid neck",
        ],
        "seed": SEED_BASE + 4,
        "n_patients": 40,
        "age_range": (0, 20),
        "female_pct": 50,
    },
    # ── HESX1 — Septo-Optic Dysplasia / CPHD ──
    {
        "gene": "HESX1",
        "protein": "Homeobox Gene Expressed in ES Cells (HESX1)",
        "alias": (
            "HESX1; OMIM gene 601802; SOD/CPHD #182230; AD haploinsufficiency or AR biallelic; "
            "3p14.3; 185 aa; ~21 kDa; paired-like homeobox TF — EARLIEST pituitary transcription factor; "
            "expressed in anterior neural ridge E5.5 → represses forebrain/pituitary differentiation prematurely; "
            "later downregulated to allow differentiation; LOF → septo-optic dysplasia (SOD) spectrum: "
            "absent septum pellucidum + optic nerve hypoplasia + pituitary defects; "
            "De Morsier syndrome = all 3 (full triad); "
            "optic nerve hypoplasia → pendular nystagmus in neonate = SCREEN PITUITARY IMMEDIATELY"
        ),
        "aa": "185 aa",
        "kDa": "~21 kDa",
        "locus": "3p14.3",
        "omim_gene": 601802,
        "omim_disease": 182230,
        "inheritance": (
            "AD haploinsufficiency: single heterozygous variant (Ala57Val most studied); "
            "AR biallelic: more severe, bilateral optic nerve hypoplasia + complete CPHD; "
            "de novo variants common; familial cases with variable expressivity; "
            "penetrance incomplete — some carriers of AD variants unaffected"
        ),
        "gene_class": (
            "HESX1 encodes the earliest pituitary TF, expressed transiently in anterior neural ridge "
            "and Rathke's pouch before E8.5. It acts as a transcriptional repressor "
            "(via Groucho co-repressors) that must be downregulated at the right time to allow "
            "Pit-1 and PROP1 to drive pituitary cell differentiation. "
            "LOF: premature transcriptional permissiveness → aberrant forebrain/pituitary development → "
            "three interdependent midline structures fail: "
            "(1) optic nerves (hypoplasia, thin optic nerve, small disc); "
            "(2) septum pellucidum (absent or thinned — transparent partition between lateral ventricles); "
            "(3) pituitary (anterior hypoplasia, ectopic posterior bright spot, CPHD of variable severity). "
            "NOTE: SOD is clinically heterogeneous — HESX1 explains only ~1% of SOD cases "
            "(most SOD is multifactorial, sporadic). "
            "KEY DIAGNOSTIC TRIGGER: Pendular nystagmus in a neonate = optic nerve hypoplasia until proven → "
            "ophthalmology + URGENT pituitary function testing (especially ACTH/cortisol — "
            "adrenal crisis can be first presentation if ACTH deficient)."
        ),
        "phenotype": (
            "Optic nerve hypoplasia (ONH): bilateral or unilateral, small pale optic disc, "
            "pendular nystagmus from birth, poor visual acuity (variable severity); "
            "Absent septum pellucidum: identified on MRI — usually asymptomatic per se but signals SOD; "
            "CPHD (variable): GH most common → growth failure; ACTH deficiency → adrenal crisis risk; "
            "TSH, LH/FSH, ADH (diabetes insipidus) may also be deficient; "
            "Cortical visual impairment in severe cases; "
            "Schizencephaly, cortical migration anomalies in ~30% with severe HESX1 mutations"
        ),
        "key_hallmarks": [
            "Optic nerve hypoplasia (pendular nystagmus) in neonate → URGENT pituitary screen",
            "Absent septum pellucidum + optic hypoplasia + CPHD = De Morsier syndrome / SOD",
            "Absent septum pellucidum ALONE does NOT confirm SOD (can be isolated variant)",
            "ACTH deficiency → adrenal crisis can be the first life-threatening presentation",
            "HESX1 explains ~1% of SOD — most SOD is sporadic multifactorial",
        ],
        "treatment_alerts": [
            "URGENT: check morning cortisol if neonatal hypoglycaemia or shock — adrenal crisis risk",
            "Stress-dose hydrocortisone protocol if ACTH-deficient",
            "Ophthalmology referral immediately — visual aids, patching, low vision support",
            "MRI brain + pituitary with gadolinium: look for ectopic posterior pituitary",
            "Replace all deficient hormones; monitor for diabetes insipidus (ADH deficiency)",
        ],
        "ddx": [
            "Isolated absent septum pellucidum: normal pituitary, normal optic nerves — NOT SOD",
            "PROP1/CPHD2: no optic hypoplasia, no septal anomaly",
            "LHX3/CPHD3: rigid cervical spine, no optic anomaly",
            "Isolated ONH: no septal/pituitary abnormality — still screen pituitary once",
        ],
        "seed": SEED_BASE + 5,
        "n_patients": 40,
        "age_range": (0, 15),
        "female_pct": 48,
    },
    # ── GLI2 — Holoprosencephaly type 9 + CPHD ──
    {
        "gene": "GLI2",
        "protein": "GLI Family Zinc Finger 2",
        "alias": (
            "GLI2; OMIM gene 165230; HPE9/CPHD #610829; AD haploinsufficiency; 2q14.2; 1586 aa; ~168 kDa; "
            "zinc finger transcription factor in hedgehog (SHH) signaling cascade; "
            "primary activator of SHH target genes; essential for forebrain/pituitary/midline structure development; "
            "LOF → variable HPE spectrum + CPHD; "
            "PATHOGNOMONIC: SINGLE CENTRAL MAXILLARY INCISOR (SCMI) — most common midline defect "
            "even without visible HPE; a child with SCMI → brain MRI + FULL pituitary hormone testing; "
            "HPE spectrum: alobar (fully fused hemispheres, lethal) → lobar → microform (SCMI only)"
        ),
        "aa": "1586 aa",
        "kDa": "~168 kDa",
        "locus": "2q14.2",
        "omim_gene": 165230,
        "omim_disease": 610829,
        "inheritance": (
            "AD haploinsufficiency; variable penetrance and expressivity (intrafamilial variation extreme — "
            "parent with SCMI may have child with alobar HPE); "
            "de novo in ~30%; familial with AD transmission; "
            "genotype-phenotype correlation poor — same variant → HPE in one family member, "
            "SCMI only in another"
        ),
        "gene_class": (
            "GLI2 encodes a zinc-finger TF that is the primary transcriptional activator of SHH target genes. "
            "In the absence of SHH signaling, GLI2 is processed by proteolysis into a transcriptional repressor. "
            "SHH ligand → smoothened (SMO) activation → full-length GLI2 accumulates → "
            "activates targets (PTCH1, GLI1, forebrain/pituitary identity genes). "
            "LOF: SHH signaling is deficient → midline structures fail to form/pattern correctly. "
            "MIDLINE STRUCTURES AFFECTED: forebrain hemispheres (HPE spectrum), "
            "pituitary (reduced Rathke's pouch expansion → hypoplastic pituitary → CPHD), "
            "dental midline (single central maxillary incisor — a common midline defect), "
            "corpus callosum (thinning/absence in severe cases). "
            "CLINICAL PEARL: The SINGLE CENTRAL MAXILLARY INCISOR is the most easily missed GLI2 sign. "
            "A child with one central upper tooth instead of two → PATHOGNOMONIC midline marker → "
            "do not dismiss as dental anomaly; obtain brain MRI + pituitary hormone panel. "
            "CPHD in GLI2: typically GH + ACTH + TSH deficiency; severity correlates with HPE severity."
        ),
        "phenotype": (
            "HPE spectrum (most severe): alobar HPE — fused hemispheres, cyclopia/proboscis (lethal); "
            "lobar/semilobar HPE: partial separation, intellectual disability, spastic quadriplegia; "
            "Microform HPE: single central maxillary incisor (SCMI), hypotelorism, flat nasal bridge; "
            "CPHD: GH + ACTH + TSH most common combination; GH deficiency → growth failure; "
            "ACTH deficiency → adrenal crisis (life-threatening first presentation); "
            "Ectopic posterior pituitary or absent on MRI; "
            "Variable intellectual disability depending on HPE severity"
        ),
        "key_hallmarks": [
            "SINGLE CENTRAL MAXILLARY INCISOR (SCMI) = GLI2 midline marker — check pituitary immediately",
            "SCMI in child without visible HPE → full brain MRI + GH/ACTH/TSH/IGF-1 panel",
            "HPE severity does NOT predict CPHD severity — variable expressivity within families",
            "ACTH deficiency → adrenal crisis can be first presentation (life-threatening)",
            "Same GLI2 variant → alobar HPE (lethal) in one sibling, SCMI only in parent",
        ],
        "treatment_alerts": [
            "URGENT: morning cortisol + ACTH in any GLI2 patient — adrenal crisis risk",
            "GH replacement if deficient (critical for brain growth and stature)",
            "Thyroxine replacement if TSH-deficient",
            "Dental referral for SCMI — orthodontic management required",
            "Genetic counseling: extreme intrafamilial variability — offspring risk 50%",
        ],
        "ddx": [
            "SHH: alobar HPE with cyclopia — most severe HPE gene, no SCMI, no CPHD",
            "ZIC2: HPE without SCMI, microform HPE, no pituitary deficiency",
            "HESX1: optic hypoplasia + absent septum, NOT HPE spectrum",
            "Isolated SCMI (sporadic): can occur without GLI2 mutation — still screen pituitary",
        ],
        "seed": SEED_BASE + 6,
        "n_patients": 40,
        "age_range": (0, 20),
        "female_pct": 50,
    },
    # ── CABLES1 — Familial Corticotropinoma ──
    {
        "gene": "CABLES1",
        "protein": "CDK5 and ABL1 Enzyme Substrate 1",
        "alias": (
            "CABLES1; OMIM gene 607516; familial corticotroph adenoma (Cushing's disease) predisposition; "
            "18q11.2; 509 aa; ~58 kDa; CDK5 substrate and ABL1 interacting protein; "
            "LOF → reduced cell-cycle checkpoint in corticotrophs → increased corticotroph proliferation; "
            "germline CABLES1 LOF variants found in ~5-10% of familial corticotropinoma kindreds; "
            "presentation: classical ACTH-dependent Cushing's disease; "
            "if trans-sphenoidal surgery fails → bilateral adrenalectomy → "
            "MANDATORY Nelson syndrome surveillance post-adrenalectomy"
        ),
        "aa": "509 aa",
        "kDa": "~58 kDa",
        "locus": "18q11.2",
        "omim_gene": 607516,
        "omim_disease": 607516,
        "inheritance": (
            "AD haploinsufficiency; germline variants in CABLES1 exons 2-6 (truncating or missense); "
            "incomplete penetrance; familial clustering of Cushing's disease or corticotropinoma; "
            "genetic testing indicated in: familial Cushing's disease, young-onset Cushing's <25 yr, "
            "multiple family members with pituitary adenoma"
        ),
        "gene_class": (
            "CABLES1 (CDK5 and ABL1 enzyme substrate 1) is a regulatory protein that interacts with "
            "CDK3/CDK5 and ABL1 to modulate cell-cycle progression. "
            "In corticotrophs, CABLES1 acts as a tumor suppressor by upregulating p21 (CDKN1A) "
            "in response to apoptotic signals and restraining CDK-mediated cell-cycle entry. "
            "LOF → reduced p21 induction → corticotrophs escape apoptosis + proliferate → "
            "ACTH-secreting adenoma → Cushing's disease. "
            "CUSHING'S DISEASE REMINDER: ACTH-dependent hypercortisolism → "
            "classic features: central obesity, moon face, buffalo hump, purple striae, "
            "hypertension, diabetes, osteoporosis, proximal myopathy. "
            "DIAGNOSIS: 24h UFC ↑ + late-night salivary cortisol ↑ + 1 mg DST failure → "
            "ACTH-dependent confirmed → MRI pituitary; if MRI negative, bilateral inferior petrosal "
            "sinus sampling (BIPSS) to confirm pituitary source. "
            "CABLES1-specific: no other CNC criteria, no skin lesions, no cardiac myxoma — "
            "distinguished from PRKAR1A/Carney Complex."
        ),
        "phenotype": (
            "ACTH-dependent Cushing's disease: central obesity, moon face, dorsal fat pad (buffalo hump), "
            "proximal myopathy, purple abdominal/thigh striae >1 cm, easy bruising, thin skin; "
            "hypertension, hyperglycaemia/diabetes, hypokalaemia; "
            "osteoporosis + vertebral fractures; growth failure in children; "
            "psychiatric: depression, cognitive impairment, psychosis; "
            "MRI: corticotroph microadenoma (<6 mm in 60%); "
            "family history: first-degree relative with Cushing's disease or corticotropinoma"
        ),
        "key_hallmarks": [
            "CABLES1 = familial corticotropinoma — Cushing's disease + family history → screen CABLES1",
            "Bilateral adrenalectomy if TSS fails → Nelson syndrome MANDATORY surveillance",
            "Nelson syndrome: rapid ACTH-driven corticotroph tumor growth post-adrenalectomy",
            "BIPSS mandatory if MRI negative but ACTH-dependent Cushing's confirmed",
            "Young-onset Cushing's (<25 yr) or familial clustering → germline testing",
        ],
        "treatment_alerts": [
            "Trans-sphenoidal surgery (TSS) first-line: remission 65-80%",
            "If TSS fails: bilateral adrenalectomy curative for hypercortisolism BUT → Nelson syndrome",
            "Nelson syndrome post-adrenalectomy: skin hyperpigmentation + aggressive ACTH tumor → MRI annually",
            "Post-TSS: hydrocortisone replacement mandatory (HPA axis suppressed); wean over 6-18 months",
            "Pasireotide or cabergoline for residual/recurrent disease",
        ],
        "ddx": [
            "PRKAR1A/Carney Complex: also Cushing's (from PPNAD) but ACTH-INDEPENDENT + lentigines + cardiac myxoma",
            "MEN1: pituitary adenoma but not specifically corticotroph; pNET + pHPT present",
            "Sporadic corticotropinoma: no family history, no germline variant",
            "Ectopic ACTH syndrome: ACTH-dependent Cushing's from lung/carcinoid — BIPSS shows peripheral ACTH",
        ],
        "seed": SEED_BASE + 7,
        "n_patients": 40,
        "age_range": (15, 60),
        "female_pct": 68,
    },
]


# ─── Patient cohort generation ─────────────────────────────────────────────

def _make_cohort(gene_def):
    rng = random.Random(gene_def["seed"])
    gene = gene_def["gene"]
    n = gene_def["n_patients"]
    age_lo, age_hi = gene_def["age_range"]
    female_pct = gene_def["female_pct"]
    patients = []
    for i in range(n):
        age = rng.randint(age_lo, age_hi)
        sex = "F" if rng.randint(1, 100) <= female_pct else "M"

        # Gene-specific clinical parameters
        if gene == "AIP":
            gh_level = round(rng.uniform(8, 80), 1)  # elevated GH (ng/mL)
            igf1_sds = round(rng.uniform(2.5, 6.0), 1)  # SD above normal
            tumor_size_mm = rng.randint(10, 35)  # macroadenoma predominant
            ssa_response = rng.choice(["partial", "resistant", "resistant", "resistant"])
            surgery_done = rng.choice([True, True, True, False])
            p = {
                "patient_id": f"AIP-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "gh_level_ng_mL": gh_level,
                "igf1_sds": igf1_sds,
                "tumor_size_mm": tumor_size_mm,
                "tumor_type": "macroadenoma" if tumor_size_mm >= 10 else "microadenoma",
                "ssa_response": ssa_response,
                "surgery_done": surgery_done,
                "pegvisomant_used": ssa_response == "resistant",
                "family_screen_done": rng.choice([True, False]),
            }
        elif gene == "PRKAR1A":
            has_myxoma = rng.random() < 0.70
            has_ppnad = rng.random() < 0.55
            has_gh_adenoma = rng.random() < 0.35
            has_lentigines = rng.random() < 0.85
            cortisol_dex_paradox = has_ppnad
            p = {
                "patient_id": f"PRKAR1A-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "cardiac_myxoma": has_myxoma,
                "ppnad_cushing": has_ppnad,
                "gh_adenoma": has_gh_adenoma,
                "lentigines": has_lentigines,
                "cortisol_dex_paradox_positive": cortisol_dex_paradox,
                "echo_surveillance_enrolled": rng.choice([True, True, False]),
                "bilateral_adrenalectomy": has_ppnad and rng.random() < 0.6,
            }
        elif gene == "PROP1":
            acth_deficient = rng.random() < 0.25  # late onset
            pituitary_hyperplasia_on_mri = age < 5 and rng.random() < 0.70
            p = {
                "patient_id": f"PROP1-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "gh_deficient": True,
                "tsh_deficient": True,
                "lhfsh_deficient": True,
                "prl_deficient": True,
                "acth_deficient": acth_deficient,
                "pituitary_hyperplasia_on_mri": pituitary_hyperplasia_on_mri,
                "gh_replacement": rng.choice([True, True, False]),
                "thyroxine_replacement": True,
                "sex_hormone_replacement": age >= 11,
            }
        elif gene == "POU1F1":
            dominant_negative = rng.random() < 0.55
            p = {
                "patient_id": f"POU1F1-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "gh_deficient": True,
                "tsh_deficient": True,
                "lhfsh_deficient": False,   # spared in POU1F1
                "prl_deficient": True,
                "acth_deficient": False,    # spared
                "dominant_negative_variant": dominant_negative,
                "thyroxine_started_before_gh": rng.choice([True, True, False]),
                "puberty_normal": True,
            }
        elif gene == "LHX3":
            snhl = rng.random() < 0.25
            rigid_spine = True  # pathognomonic
            atlantoaxial_instability = rng.random() < 0.20
            p = {
                "patient_id": f"LHX3-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "gh_deficient": True,
                "tsh_deficient": True,
                "lhfsh_deficient": True,
                "prl_deficient": True,
                "acth_deficient": False,
                "rigid_cervical_spine": rigid_spine,
                "cervical_rotation_degrees": rng.randint(20, 75),
                "snhl_present": snhl,
                "cspine_xray_done": rng.choice([True, True, False]),
                "atlantoaxial_instability": atlantoaxial_instability,
            }
        elif gene == "HESX1":
            onh_bilateral = rng.random() < 0.65
            absent_septum = rng.random() < 0.70
            acth_deficient = rng.random() < 0.40
            adrenal_crisis_first_presentation = acth_deficient and rng.random() < 0.35
            p = {
                "patient_id": f"HESX1-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "optic_nerve_hypoplasia": True,
                "bilateral_onh": onh_bilateral,
                "absent_septum_pellucidum": absent_septum,
                "pendular_nystagmus": True,
                "acth_deficient": acth_deficient,
                "gh_deficient": rng.random() < 0.70,
                "tsh_deficient": rng.random() < 0.50,
                "adh_deficient": rng.random() < 0.30,
                "adrenal_crisis_first_presentation": adrenal_crisis_first_presentation,
                "full_sod_triad": onh_bilateral and absent_septum and acth_deficient,
            }
        elif gene == "GLI2":
            scmi = rng.random() < 0.55
            hpe_severity = rng.choice(["microform", "microform", "lobar", "semilobar", "alobar"])
            acth_deficient = rng.random() < 0.50
            adrenal_crisis = acth_deficient and rng.random() < 0.40
            p = {
                "patient_id": f"GLI2-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "hpe_severity": hpe_severity,
                "scmi_present": scmi or hpe_severity == "microform",
                "acth_deficient": acth_deficient,
                "gh_deficient": rng.random() < 0.60,
                "tsh_deficient": rng.random() < 0.40,
                "adrenal_crisis_presentation": adrenal_crisis,
                "mri_done": True,
                "pituitary_hormone_tested_after_scmi": rng.choice([True, False]),
            }
        elif gene == "CABLES1":
            tss_remission = rng.random() < 0.72
            bilateral_adrenalectomy = not tss_remission and rng.random() < 0.70
            nelson_syndrome = bilateral_adrenalectomy and rng.random() < 0.25
            ufc_multiple_uln = round(rng.uniform(3, 30), 1)
            p = {
                "patient_id": f"CABLES1-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "ufc_x_uln": ufc_multiple_uln,
                "late_night_salivary_cortisol_elevated": True,
                "acth_dependent_cushing": True,
                "tss_done": True,
                "tss_remission": tss_remission,
                "bilateral_adrenalectomy": bilateral_adrenalectomy,
                "nelson_syndrome": nelson_syndrome,
                "mri_annual_nelson_surveillance": nelson_syndrome,
                "family_history_cushing": rng.random() < 0.60,
            }
        else:
            p = {"patient_id": f"{gene}-{i+1:03d}", "gene": gene, "age_at_dx": age, "sex": sex}

        patients.append(p)
    return patients


_ALL_COHORTS = {g["gene"]: _make_cohort(g) for g in PITUITARY_GENES}


# ─── API response builders ─────────────────────────────────────────────────

def _pct(cohort, key):
    n = len(cohort)
    if n == 0:
        return 0
    return round(100 * sum(1 for p in cohort if p.get(key)) / n, 1)


def get_overview():
    n = sum(len(v) for v in _ALL_COHORTS.values())

    # Aggregate key metrics
    gh_deficient_genes = ["PROP1", "POU1F1", "LHX3", "HESX1", "GLI2"]
    acth_deficient_genes = ["HESX1", "GLI2", "CABLES1"]

    gh_deficiency_rate = round(
        sum(len(_ALL_COHORTS[g]) for g in gh_deficient_genes) / n * 100, 1
    )

    return {
        "atlas_name": "Pituitary-Disorders-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Pituitary Disorders Atlas",
        "n_patients": n,
        "gene_count": len(PITUITARY_GENES),
        "genes": [g["gene"] for g in PITUITARY_GENES],
        "seeds": "1310–1317",
        "registered": "2026-09-05",
        "atlas_version": "1.0",
        "gene_summary": [
            {
                "gene": "AIP",
                "protein": "Aryl Hydrocarbon Receptor Interacting Protein",
                "aa": "330 aa",
                "locus": "11q13.2",
                "inheritance": "AD haploinsufficiency",
                "phenotype_short": "FIPA: familial isolated GH-secreting pituitary adenoma, young-onset gigantism/acromegaly",
                "hallmark_short": "AIP: younger + bigger + MORE SSA-resistant — use pegvisomant if SSA fails at 6 months",
            },
            {
                "gene": "PRKAR1A",
                "protein": "PKA Regulatory Subunit 1α",
                "aa": "381 aa",
                "locus": "17q24.2",
                "inheritance": "AD LOF",
                "phenotype_short": "Carney Complex: lentigines + cardiac myxoma + GH adenoma + PPNAD Cushing's",
                "hallmark_short": "Cardiac myxoma ANY chamber + lentigines = Carney Complex — echocardiography MANDATORY annually",
            },
            {
                "gene": "PROP1",
                "protein": "Paired-Like Homeobox 1 (Prophet of Pit-1)",
                "aa": "226 aa",
                "locus": "5q35.3",
                "inheritance": "AR biallelic",
                "phenotype_short": "CPHD2: GH + TSH + LH/FSH + PRL deficiency; pituitary hyperplasia early → involution",
                "hallmark_short": "Pituitary hyperplasia on MRI early: DO NOT BIOPSY — it involutes; ACTH deficiency emerges late",
            },
            {
                "gene": "POU1F1",
                "protein": "POU Class 1 Homeobox 1 (Pit-1)",
                "aa": "291 aa",
                "locus": "3p11.2",
                "inheritance": "AD dominant-negative or AR biallelic",
                "phenotype_short": "CPHD1: GH + TSH + PRL only — puberty NORMAL, ACTH NORMAL",
                "hallmark_short": "Start THYROXINE FIRST before GH — GH in hypothyroid state risks adrenal crisis",
            },
            {
                "gene": "LHX3",
                "protein": "LIM Homeobox 3",
                "aa": "397 aa",
                "locus": "9q34.3",
                "inheritance": "AR biallelic",
                "phenotype_short": "CPHD3: GH + TSH + LH/FSH + PRL deficiency + RIGID CERVICAL SPINE",
                "hallmark_short": "Rigid cervical spine + CPHD = LHX3 — X-ray C-spine BEFORE MRI; NO cervical manipulation",
            },
            {
                "gene": "HESX1",
                "protein": "Homeobox Gene Expressed in ES Cells",
                "aa": "185 aa",
                "locus": "3p14.3",
                "inheritance": "AD haploinsufficiency or AR biallelic",
                "phenotype_short": "SOD: optic nerve hypoplasia + absent septum pellucidum + CPHD",
                "hallmark_short": "Pendular nystagmus in neonate = optic hypoplasia → URGENT pituitary screen (ACTH crisis risk)",
            },
            {
                "gene": "GLI2",
                "protein": "GLI Family Zinc Finger 2",
                "aa": "1586 aa",
                "locus": "2q14.2",
                "inheritance": "AD haploinsufficiency",
                "phenotype_short": "HPE9/CPHD: single central maxillary incisor + CPHD ± holoprosencephaly",
                "hallmark_short": "Single central maxillary incisor = GLI2 midline marker — brain MRI + pituitary hormones immediately",
            },
            {
                "gene": "CABLES1",
                "protein": "CDK5 and ABL1 Enzyme Substrate 1",
                "aa": "509 aa",
                "locus": "18q11.2",
                "inheritance": "AD haploinsufficiency",
                "phenotype_short": "Familial corticotropinoma: ACTH-dependent Cushing's disease, family history",
                "hallmark_short": "Bilateral adrenalectomy → Nelson syndrome MANDATORY annual MRI surveillance",
            },
        ],
        "aggregate_clinical": {
            "combined_hormone_deficiency_pct": round(
                (len(_ALL_COHORTS["PROP1"]) + len(_ALL_COHORTS["POU1F1"]) +
                 len(_ALL_COHORTS["LHX3"]) + len(_ALL_COHORTS["HESX1"]) +
                 len(_ALL_COHORTS["GLI2"])) / n * 100, 1
            ),
            "gh_deficiency_pct": gh_deficiency_rate,
            "acth_crisis_risk_pct": round(
                sum(len(_ALL_COHORTS[g]) for g in acth_deficient_genes) / n * 100, 1
            ),
            "tumour_predisposition_pct": round(
                (len(_ALL_COHORTS["AIP"]) + len(_ALL_COHORTS["PRKAR1A"]) +
                 len(_ALL_COHORTS["CABLES1"])) / n * 100, 1
            ),
            "cardiac_surveillance_required_pct": round(
                _pct(_ALL_COHORTS["PRKAR1A"], "cardiac_myxoma") *
                len(_ALL_COHORTS["PRKAR1A"]) / n, 1
            ),
            "rigid_spine_pct": round(
                100 * len(_ALL_COHORTS["LHX3"]) / n, 1
            ),
        },
        "key_clinical_rules": [
            "AIP: GH adenoma in patient <30 yr → germline AIP testing; SSA resistance → pegvisomant",
            "PRKAR1A: cardiac myxoma ANY chamber → echocardiography ANNUALLY; Carney Complex = any 2 of 11 criteria",
            "PROP1: pituitary hyperplasia on MRI → DO NOT BIOPSY (it involutes); ACTH deficiency emerges late",
            "POU1F1: THYROXINE FIRST before GH — GH in hypothyroid state unmasks adrenal insufficiency",
            "LHX3: rigid cervical spine + CPHD = LHX3; X-ray C-spine BEFORE MRI; NO manipulation",
            "HESX1: pendular nystagmus in neonate → URGENT ACTH/cortisol (adrenal crisis risk)",
            "GLI2: single central maxillary incisor → brain MRI + pituitary panel (ACTH deficiency risk)",
            "CABLES1: bilateral adrenalectomy → Nelson syndrome → annual MRI surveillance mandatory",
        ],
    }


def get_breakdown():
    result = []
    for g in PITUITARY_GENES:
        gene = g["gene"]
        cohort = _ALL_COHORTS[gene]
        n = len(cohort)

        # Base entry
        entry = {
            "gene": gene,
            "protein": g["protein"],
            "aa": g["aa"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "omim_disease": g["omim_disease"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "phenotype": g["phenotype"],
            "key_hallmarks": g["key_hallmarks"],
            "treatment_alerts": g["treatment_alerts"],
            "ddx": g["ddx"],
            "n_patients": n,
            "cohort_stats": {},
        }

        # Per-gene cohort statistics
        if gene == "AIP":
            macro_pct = round(100 * sum(1 for p in cohort if p.get("tumor_type") == "macroadenoma") / n, 1)
            resistant_pct = _pct(cohort, "pegvisomant_used")
            entry["cohort_stats"] = {
                "macroadenoma_pct": macro_pct,
                "somatostatin_resistant_pct": resistant_pct,
                "mean_tumor_size_mm": round(sum(p["tumor_size_mm"] for p in cohort) / n, 1),
                "mean_igf1_sds": round(sum(p["igf1_sds"] for p in cohort) / n, 1),
                "family_screen_done_pct": _pct(cohort, "family_screen_done"),
            }
        elif gene == "PRKAR1A":
            entry["cohort_stats"] = {
                "cardiac_myxoma_pct": _pct(cohort, "cardiac_myxoma"),
                "ppnad_cushing_pct": _pct(cohort, "ppnad_cushing"),
                "gh_adenoma_pct": _pct(cohort, "gh_adenoma"),
                "lentigines_pct": _pct(cohort, "lentigines"),
                "dex_paradox_positive_pct": _pct(cohort, "cortisol_dex_paradox_positive"),
                "echo_surveillance_enrolled_pct": _pct(cohort, "echo_surveillance_enrolled"),
            }
        elif gene == "PROP1":
            entry["cohort_stats"] = {
                "acth_deficient_pct": _pct(cohort, "acth_deficient"),
                "pituitary_hyperplasia_on_mri_pct": _pct(cohort, "pituitary_hyperplasia_on_mri"),
                "gh_replacement_pct": _pct(cohort, "gh_replacement"),
                "thyroxine_replacement_pct": _pct(cohort, "thyroxine_replacement"),
            }
        elif gene == "POU1F1":
            entry["cohort_stats"] = {
                "dominant_negative_variant_pct": _pct(cohort, "dominant_negative_variant"),
                "thyroxine_before_gh_pct": _pct(cohort, "thyroxine_started_before_gh"),
                "puberty_normal_pct": _pct(cohort, "puberty_normal"),
            }
        elif gene == "LHX3":
            entry["cohort_stats"] = {
                "rigid_spine_pct": 100.0,
                "mean_cervical_rotation": round(sum(p["cervical_rotation_degrees"] for p in cohort) / n, 1),
                "snhl_pct": _pct(cohort, "snhl_present"),
                "atlantoaxial_instability_pct": _pct(cohort, "atlantoaxial_instability"),
                "cspine_xray_done_pct": _pct(cohort, "cspine_xray_done"),
            }
        elif gene == "HESX1":
            entry["cohort_stats"] = {
                "bilateral_onh_pct": _pct(cohort, "bilateral_onh"),
                "absent_septum_pct": _pct(cohort, "absent_septum_pellucidum"),
                "acth_deficient_pct": _pct(cohort, "acth_deficient"),
                "adrenal_crisis_first_pct": _pct(cohort, "adrenal_crisis_first_presentation"),
                "full_sod_triad_pct": _pct(cohort, "full_sod_triad"),
            }
        elif gene == "GLI2":
            entry["cohort_stats"] = {
                "scmi_present_pct": _pct(cohort, "scmi_present"),
                "acth_deficient_pct": _pct(cohort, "acth_deficient"),
                "adrenal_crisis_pct": _pct(cohort, "adrenal_crisis_presentation"),
                "hpe_severity_breakdown": {
                    sev: round(100 * sum(1 for p in cohort if p.get("hpe_severity") == sev) / n, 1)
                    for sev in ["microform", "lobar", "semilobar", "alobar"]
                },
                "pituitary_tested_after_scmi_pct": _pct(cohort, "pituitary_hormone_tested_after_scmi"),
            }
        elif gene == "CABLES1":
            entry["cohort_stats"] = {
                "tss_remission_pct": _pct(cohort, "tss_remission"),
                "bilateral_adrenalectomy_pct": _pct(cohort, "bilateral_adrenalectomy"),
                "nelson_syndrome_pct": _pct(cohort, "nelson_syndrome"),
                "family_history_pct": _pct(cohort, "family_history_cushing"),
                "mean_ufc_x_uln": round(sum(p["ufc_x_uln"] for p in cohort) / n, 1),
            }

        result.append(entry)
    return result


def get_definitions():
    return {
        "atlas": "Pituitary-Disorders-Atlas",
        "total_definitions": 12,
        "definitions": [
            {
                "term": "Familial Isolated Pituitary Adenoma (FIPA)",
                "short": "Hereditary pituitary adenoma without other syndromic features; ~20% have AIP mutation",
                "detail": (
                    "FIPA: ≥2 relatives with pituitary adenoma in absence of MEN1 or Carney Complex. "
                    "AIP mutation found in ~15-20% of FIPA kindreds, predominantly GH-secreting subtypes. "
                    "AIP screening indications: pituitary adenoma <35 yr, familial clustering, "
                    "somatostatin analog resistance. AIP-mutant adenomas: earlier onset, larger, more SSA-resistant. "
                    "First-degree relatives of AIP mutation carriers: annual IGF-1 + MRI every 2 years from age 10."
                ),
                "clinical_rule": "GH adenoma in patient under 35 → AIP germline testing regardless of family history",
            },
            {
                "term": "Carney Complex (CNC) — PRKAR1A",
                "short": "Multi-endocrine tumor syndrome: lentigines + cardiac myxoma + PPNAD + GH adenoma + schwannomas",
                "detail": (
                    "Carney Complex diagnostic criteria (any 2 of 11 major): spotty skin pigmentation; "
                    "cardiac myxoma; cutaneous myxoma; breast myxoid fibroadenoma; PPNAD; GH-secreting pituitary adenoma; "
                    "large cell calcifying Sertoli cell tumor; thyroid carcinoma; malignant psammomatous melanotic schwannoma; "
                    "blue nevus; ductal adenoma of breast. "
                    "CARDIAC MYXOMA: can occur in any cardiac chamber (NOT just left atrium as in sporadic). "
                    "Annual echocardiography is MANDATORY — myxomas recur in ~15% after resection and cause "
                    "embolism (stroke) or sudden cardiac death. "
                    "PPNAD Cushing's: ACTH-independent + paradoxical cortisol RISE after dexamethasone (Liddle maneuver)."
                ),
                "clinical_rule": "Cardiac myxoma in young patient + any skin lesion → Carney Complex screening; echocardiography ANNUALLY",
            },
            {
                "term": "Combined Pituitary Hormone Deficiency (CPHD) — Types 1-3",
                "short": "Hereditary deficiency of multiple pituitary hormones based on TF class (POU1F1/PROP1/LHX3)",
                "detail": (
                    "CPHD types by gene: "
                    "CPHD1 (POU1F1/Pit-1): GH + TSH + PRL — puberty NORMAL, ACTH NORMAL; "
                    "CPHD2 (PROP1): GH + TSH + LH/FSH + PRL + LATE ACTH; "
                    "CPHD3 (LHX3): GH + TSH + LH/FSH + PRL + RIGID NECK; "
                    "CPHD4 (LHX4): GH + TSH + ACTH (cerebellar/sella anomalies); "
                    "CPHD5 (HESX1): SOD spectrum. "
                    "HORMONAL REPLACEMENT SEQUENCE: thyroxine first, then GH (to avoid adrenal crisis unmask), "
                    "then sex hormones at pubertal age. Annual ACTH testing for all PROP1 patients."
                ),
                "clinical_rule": "Replace thyroxine before GH in any CPHD — GH in hypothyroid state risks adrenal crisis",
            },
            {
                "term": "PROP1 Pituitary Hyperplasia — Do Not Biopsy",
                "short": "PROP1 mutation causes early pituitary hyperplasia on MRI that involutes over time",
                "detail": (
                    "PROP1 LOF fails to clear Rathke's cleft precursor cells → transient hyperplasia of "
                    "undifferentiated pituitary cells visible on MRI as an enlarged pituitary mass in infancy/childhood. "
                    "The mass can be confused for a craniopharyngioma or adenoma and trigger surgical referral. "
                    "KEY: this hyperplasia involutes spontaneously over years → DO NOT BIOPSY. "
                    "Confirm PROP1 germline variant + multi-hormone deficiency + age-appropriate presentation → "
                    "conservative management. MRI follow-up every 6-12 months confirms involution."
                ),
                "clinical_rule": "Pituitary hyperplasia in young child with GH+TSH+LH/FSH+PRL deficiency → PROP1 testing; DO NOT BIOPSY",
            },
            {
                "term": "LHX3 Rigid Cervical Spine — Protocol",
                "short": "LHX3 mutation causes pathognomonic restricted cervical rotation (<90°) requiring X-ray before MRI",
                "detail": (
                    "LHX3 is expressed in paraxial mesoderm during vertebral segmentation — LOF → "
                    "abnormal atlantoaxial or subaxial bony/ligamentous development → cervical rotation <90° "
                    "(normally >90°). This is detected clinically by asking the child to look over each shoulder. "
                    "PROTOCOL: (1) Before performing cervical MRI or any neck procedure: plain AP+lateral C-spine X-ray "
                    "to document anatomy and check for atlantoaxial instability. "
                    "(2) If atlantoaxial instability: neurosurgical referral; no manipulation/contact sports. "
                    "(3) Do NOT perform Dix-Hallpike or any cervical chiropractic maneuver. "
                    "(4) Alert all care providers (dental, ENT, anesthesia) about rigid C-spine risk."
                ),
                "clinical_rule": "LHX3/CPHD3: X-ray C-spine before MRI; no cervical manipulation; alert anesthesia before any procedure",
            },
            {
                "term": "Septo-Optic Dysplasia (SOD) / De Morsier Syndrome — HESX1",
                "short": "SOD triad: optic nerve hypoplasia + absent septum pellucidum + CPHD; HESX1 in ~1%",
                "detail": (
                    "De Morsier syndrome (full SOD triad): absent septum pellucidum + bilateral optic nerve hypoplasia + "
                    "combined pituitary hormone deficiency. HESX1 explains ~1% of SOD (most SOD is sporadic). "
                    "CLINICAL TRIGGER: pendular nystagmus in neonate → optic nerve hypoplasia until proven → "
                    "ophthalmology + MRI brain/pituitary + URGENT pituitary hormone testing. "
                    "Priority test: morning cortisol — ACTH deficiency can cause life-threatening adrenal crisis "
                    "as first presentation of SOD. Absent septum pellucidum ALONE (without ONH or CPHD) "
                    "does NOT confirm SOD and does NOT require HESX1 testing."
                ),
                "clinical_rule": "Neonatal nystagmus → optic hypoplasia → screen ACTH immediately; adrenal crisis can be first presentation",
            },
            {
                "term": "Single Central Maxillary Incisor (SCMI) — GLI2 Midline Marker",
                "short": "SCMI = pathognomonic GLI2 midline defect; requires brain MRI + pituitary hormone panel",
                "detail": (
                    "A single central maxillary incisor (one upper central tooth instead of two, perfectly midline) "
                    "is the most common microform midline defect in GLI2 mutations. "
                    "It may be the only visible manifestation of an otherwise invisible holoprosencephaly spectrum. "
                    "ACTION: SCMI in any child → (1) brain MRI to evaluate for HPE spectrum; "
                    "(2) full pituitary hormone panel: GH (IGF-1), ACTH/cortisol, TSH/T4; "
                    "(3) first-degree family screening — extreme intrafamilial variability means "
                    "a parent may have SCMI only while a sibling has lobar HPE. "
                    "Dental note: SCMI cannot be fixed orthodontically until pituitary is assessed — "
                    "urgency is the endocrine emergency (ACTH deficiency), not the dental appearance."
                ),
                "clinical_rule": "Single central maxillary incisor in child → brain MRI + ACTH/cortisol panel immediately (adrenal crisis risk)",
            },
            {
                "term": "Nelson Syndrome — Post-Adrenalectomy ACTH Tumor",
                "short": "After bilateral adrenalectomy for Cushing's disease, rapid ACTH-driven corticotroph tumor growth",
                "detail": (
                    "Nelson syndrome: following bilateral adrenalectomy for ACTH-dependent Cushing's disease, "
                    "removal of cortisol negative feedback → sustained ACTH hypersecretion → "
                    "rapid growth of the pre-existing corticotroph adenoma (which was the source of excess ACTH). "
                    "Features: progressive skin and mucosal hyperpigmentation (MSH co-secretion), "
                    "headache + visual field defects (tumor mass effect), rising plasma ACTH. "
                    "Incidence: ~15-30% of post-adrenalectomy Cushing's patients. "
                    "MANDATORY SURVEILLANCE: annual pituitary MRI + plasma ACTH in ALL patients after "
                    "bilateral adrenalectomy for Cushing's disease. "
                    "Treatment of Nelson: pituitary-directed radiotherapy or reoperation; temozolomide for aggressive cases."
                ),
                "clinical_rule": "Bilateral adrenalectomy → annual MRI + ACTH mandatory lifelong; rising ACTH + hyperpigmentation = Nelson syndrome",
            },
            {
                "term": "PPNAD — Primary Pigmented Nodular Adrenocortical Disease",
                "short": "PRKAR1A LOF → bilateral adrenal micronodular Cushing's, ACTH-independent, dexamethasone paradox",
                "detail": (
                    "PPNAD: bilateral micronodular adrenal disease (nodules 1-3 mm, pigmented, cortisol-secreting). "
                    "Pathophysiology: PRKAR1A LOF → constitutive PKA activation in adrenal cells → autonomous cortisol. "
                    "ACTH INDEPENDENT: bilateral adrenal 24h UFC elevation despite suppressed ACTH. "
                    "DEXAMETHASONE PARADOX (Liddle maneuver): after low-dose dexamethasone, cortisol RISES "
                    "paradoxically by ≥50% — pathognomonic for PPNAD. "
                    "Diagnosis: CBC/serum 17-OHC urine collection, adrenal CT (bilateral micro-nodules), "
                    "cortisol post-Liddle test. "
                    "Treatment: bilateral adrenalectomy → permanent hydrocortisone/fludrocortisone replacement lifelong."
                ),
                "clinical_rule": "ACTH-independent Cushing's + dexamethasone paradox (cortisol RISES) = PPNAD / Carney Complex",
            },
            {
                "term": "Holoprosencephaly (HPE) Spectrum — GLI2/SHH Signaling",
                "short": "Failure of forebrain division; ranges from alobar (lethal) to microform (SCMI only)",
                "detail": (
                    "HPE spectrum: "
                    "Alobar: complete failure of forebrain division, cyclopia/proboscis/arhinencephaly (incompatible with life); "
                    "Semilobar: partial posterior separation; "
                    "Lobar: near-complete separation with midline defects; "
                    "Microform: SCMI + hypotelorism ± anosmia, NO brain MRI HPE visible. "
                    "GLI2 causes HPE via failure of SHH transcriptional activation of ventral forebrain identity genes. "
                    "KEY CLINICAL POINT: intrafamilial variability is extreme — "
                    "the same GLI2 variant causes microform in one family member and alobar HPE in another. "
                    "CPHD severity does NOT reliably correlate with HPE severity in GLI2 patients."
                ),
                "clinical_rule": "HPE in neonate: urgent pituitary function testing; ACTH deficiency risk; family variant testing",
            },
            {
                "term": "Somatostatin Analogs (SSA) — AIP-Mutant Resistance",
                "short": "AIP-mutant GH adenomas are less responsive to SSA (octreotide/lanreotide) — prefer pegvisomant",
                "detail": (
                    "Somatostatin analogs (octreotide LAR, lanreotide autogel): first-line medical therapy for GH adenomas. "
                    "Mechanism: bind SSTR2/SSTR5 → inhibit cAMP → reduce GH secretion + tumor growth. "
                    "AIP-MUTANT RESISTANCE: AIP interacts with PDE4A5 to modulate cAMP degradation. "
                    "AIP LOF disrupts cAMP signaling scaffold → reduced SSTR2 expression + altered cAMP response → "
                    "IGF-1 normalization rate with SSA ~30% (vs ~50% in sporadic). "
                    "MANAGEMENT: if IGF-1 not normalized after 6 months of SSA → ADD or SWITCH to pegvisomant "
                    "(GH-receptor antagonist, normalizes IGF-1 in >90% regardless of AIP status). "
                    "Surgery remains first-line regardless — SSA does NOT shrink AIP adenomas reliably."
                ),
                "clinical_rule": "AIP-mutant GH adenoma: surgery first; if SSA fails at 6 months → pegvisomant; do NOT wait beyond 12 months",
            },
            {
                "term": "Cascade Testing — Pituitary Disorders",
                "short": "All first-degree relatives of AIP/PRKAR1A/hereditary pituitary patients require germline testing",
                "detail": (
                    "Cascade testing principles for hereditary pituitary disorders: "
                    "AIP mutation carriers: offer germline testing to all first-degree relatives; "
                    "positive relatives: MRI pituitary + IGF-1 annually from age 10. "
                    "PRKAR1A/Carney Complex: offer testing to all first-degree relatives; "
                    "positive relatives: echocardiography + adrenal function + IGF-1 annually. "
                    "PROP1/POU1F1/LHX3: AR inheritance → siblings at 25% risk; parents carriers (test); "
                    "HESX1 AD: 50% risk per child. "
                    "GLI2: AD with extreme variability — siblings and parents need brain MRI + pituitary screen. "
                    "CABLES1: AD; family members with pituitary adenoma history → test."
                ),
                "clinical_rule": "Hereditary pituitary disorder confirmed → test all first-degree relatives before their adenoma becomes symptomatic",
            },
        ]
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:1000])
    print("\n=== BREAKDOWN (gene 1) ===")
    bd = get_breakdown()
    print(json.dumps(bd[0], indent=2)[:800])
    print(f"\nTotal patients: {sum(len(v) for v in _ALL_COHORTS.values())}")
