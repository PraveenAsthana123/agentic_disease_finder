#!/usr/bin/env python3
"""Adrenal Disorders Atlas — Complete 8-Gene Hereditary Adrenal / Steroidogenesis Disorders Atlas
CYP21A2 (CAH21 — 21-Hydroxylase Deficiency; most common CAH; 90-95%; 495 aa; 21q22.13; AR;
         salt-wasting (SW) + simple-virilizing (SV); elevated 17-OHP PATHOGNOMONIC; NBS standard;
         fludrocortisone + hydrocortisone; prenatal dexamethasone; never mineralocorticoid alone) ·
CYP11B1 (CAH-11β — 11β-Hydroxylase Deficiency; 2nd most common CAH; 5-8%; 503 aa; 8q24.3; AR;
         HYPERTENSION PATHOGNOMONIC — elevated 11-deoxycortisol + DOC; virilisation + HTN;
         spironolactone CI acute; dexamethasone suppression; no SW — mineralocorticoids NORMAL) ·
CYP11B2 (CMO-I/II — Aldosterone Synthase Deficiency; 503 aa; 8q24.3; AR;
         salt-wasting with LOW aldosterone + ELEVATED plasma renin; 18-OH-corticosterone/aldosterone ratio;
         fludrocortisone curative; no virilisation; no hypertension DDx CYP11B1) ·
CYP17A1 (17α-Hydroxylase/17,20-Lyase Deficiency; 508 aa; 10q24.32; AR;
         HYPERTENSION + SEXUAL INFANTILISM/AMBIGUITY + HYPOKALEMIA; 46,XX: primary amenorrhoea + no puberty;
         46,XY: phenotypically female at birth; elevated ACTH suppressed by dexamethasone) ·
STAR   (Congenital Lipoid Adrenal Hyperplasia; StAR Steroidogenic Acute Regulatory Protein;
         285 aa; 8p11.23; AR; MOST SEVERE adrenal insufficiency — all steroid classes absent;
         lipid droplet accumulation in adrenal on MRI PATHOGNOMONIC; 46,XY phenotypically female;
         immediate stress steroids LIFE-SAVING; AVOID LIPID-LOWERING — worsens StAR substrate deprivation) ·
NR0B1  (Adrenal Hypoplasia Congenita AHC; DAX1; 470 aa; Xp21.2; X-linked;
         X-linked AHC + Hypogonadotropic Hypogonadism (HH); primary adrenal failure + GnRH deficiency;
         testosterone + gonadotropins REQUIRED; CONTIGUOUS GENE DELETION — check for DMD, GKD on Xp21) ·
MC2R   (Familial Glucocorticoid Deficiency Type 1 FGD1; ACTH Receptor; 297 aa; 18p11.21; AR;
         ISOLATED glucocorticoid deficiency — mineralocorticoids NORMAL; tall stature + hyperpigmentation;
         ACTH markedly elevated; renin NORMAL DDx CAH; no salt-wasting; hydrocortisone CURATIVE) ·
AAAS   (Triple-A Syndrome / Allgrove Syndrome; Aladin; 547 aa; 12q13.13; AR;
         TRIAD: Alacrima (first symptom) + Achalasia + ACTH-resistant adrenal insufficiency;
         autonomic neuropathy + progressive neurological; ALACRIMA IS EARLIEST SIGN — appears in childhood
         before adrenal crisis; pilocarpine eye drops + lubricants; achalasia — pneumatic dilation/Heller)
320-patient aggregate cohort (8 × 40, seeds 1254–1261)
"""

import random

SEED_BASE = 1254

ADRENAL_GENES = [
    # ── CYP21A2 — 21-Hydroxylase Deficiency (Classic CAH) ──────────────────
    {
        "gene": "CYP21A2",
        "protein": "21-Hydroxylase (CYP21A2 / P450c21)",
        "alias": (
            "CYP21A2; OMIM gene 201910; Congenital Adrenal Hyperplasia due to 21-Hydroxylase Deficiency #201910; "
            "21q22.13; 495 aa; ~55 kDa; AR (biallelic); most common CAH (90–95% of all CAH); "
            "prevalence 1 in 15,000–16,000 births (classic); two classic forms: salt-wasting (SW, ~75%) and "
            "simple-virilizing (SV, ~25%); non-classic (NC) CAH is commonest AR disorder in humans "
            "(1 in 50–1000 depending on ethnicity); pseudogene CYP21A1P on same chromosome — MLPA mandatory"
        ),
        "aa": "495 aa",
        "kDa": "~55 kDa",
        "locus": "21q22.13",
        "omim_gene": 201910,
        "omim_disease": 201910,
        "inheritance": "AR (biallelic); carrier frequency 1 in 60 in general population",
        "gene_class": (
            "Microsomal cytochrome P450 enzyme (CYP21A2 / P450c21); "
            "catalyses hydroxylation at C-21: progesterone → DOC (11-deoxycorticosterone) and "
            "17-hydroxyprogesterone (17-OHP) → 11-deoxycortisol; "
            "pathway: cholesterol → pregnenolone → 17-OHP [→ CYP21A2 block →] 11-deoxycortisol → cortisol; "
            "and: progesterone → DOC → corticosterone → aldosterone; "
            "LOF → cortisol deficiency → compensatory ACTH elevation → androgen precursor accumulation "
            "(17-OHP, androstenedione, testosterone); "
            "SW: severe LOF (null alleles) → aldosterone also deficient → life-threatening salt loss in neonatal period; "
            "SV: partial LOF → cortisol reduced + androgens elevated, aldosterone sufficient; "
            "NC: mild LOF → only mild androgen excess (hirsutism, oligomenorrhoea, premature puberty); "
            "gene duplication/deletions: CYP21A2 adjacent to pseudogene CYP21A1P on 6p21.3 (RP-C4-CYP21-TNX module); "
            "gene conversion from pseudogene: most common mutation source (~75% alleles); "
            "MLPA + sequencing BOTH required to detect deletions and gene conversions; "
            "17-OHP stimulated > 1500 nmol/L (60-minute Synacthen 250 μg IM): DIAGNOSTIC for classic CAH; "
            "NBS: whole-blood 17-OHP (day 3); SW form identified before salt-losing crisis in most countries"
        ),
        "phenotype": (
            "SALT-WASTING (SW) — severe (75% of classic): "
            "Neonatal presentation (day 7–14): vomiting, poor feeding, weight loss, hyponatraemia, hyperkalaemia, "
            "shock; LIFE-THREATENING adrenal crisis if unrecognised; "
            "46,XX: VIRILISED genitalia at birth (Prader I–V); uterus and ovaries present; "
            "46,XY: normal male external genitalia; "
            "SIMPLE-VIRILIZING (SV) — moderate (25% of classic): "
            "46,XX: virilised genitalia at birth; no SW crisis; "
            "46,XY: presents at age 2–4 with precocious puberty (pseudo-precocious puberty); "
            "NON-CLASSIC (NC): "
            "Children: premature puberty/adrenarche, rapid growth, advanced bone age; "
            "Adolescent/adult females: hirsutism, acne, oligomenorrhoea, infertility; "
            "Biochemistry all forms: 17-OHP markedly elevated; androstenedione elevated; renin elevated (SW); "
            "Mineralocorticoid status: SW — aldosterone deficient; SV/NC — aldosterone normal"
        ),
        "hallmark": (
            "17-OHP MARKEDLY ELEVATED is PATHOGNOMONIC for CAH21 — 17-OHP is the diagnostic biomarker; "
            "NEONATAL CRISIS (SW): hyponatraemia + hyperkalaemia + poor feeding + shock day 7–14 — LIFE-THREATENING; "
            "DO NOT WAIT FOR 17-OHP before starting steroids in sick neonate with these electrolytes; "
            "46,XX VIRILISED AT BIRTH: diagnose CAH21 urgently; never assign sex on external appearance alone; "
            "MLPA MANDATORY: CYP21A2 deletions and gene conversions (from pseudogene CYP21A1P) account for >75% of mutations; "
            "sequencing alone MISSES large deletions/conversions; order MLPA + sequencing TOGETHER; "
            "STRESS DOSE STEROIDS: sick day rule — any fever/vomiting/surgery → 3× hydrocortisone dose; "
            "FLUDROCORTISONE for SW and SV forms (even SV may have subclinical mineralocorticoid insufficiency); "
            "NBS 17-OHP: premature infants have false-positive elevation — recheck at expected term age; "
            "PRENATAL TREATMENT: dexamethasone 20 μg/kg/day from 5–6 weeks gestation for at-risk 46,XX fetuses "
            "(controversial; requires rapid genetic confirmation; evidence level B)"
        ),
        "treatment_alert": (
            "HYDROCORTISONE: glucocorticoid of choice (10–15 mg/m²/day in 3 doses); "
            "avoid prednisolone/dexamethasone in children (growth suppression); "
            "FLUDROCORTISONE: mineralocorticoid replacement SW + SV (0.05–0.2 mg/day); "
            "SALT SUPPLEMENTATION: 1–2 g NaCl/day in infants with SW (first 2 years); "
            "ACUTE CRISIS: IV hydrocortisone 100 mg/m²/day + IV saline + dextrose; "
            "STRESS DOSES: any fever/vomiting/surgery → 3× normal HC dose; IM hydrocortisone kit at home; "
            "EMERGENCY STEROID CARD: wear/carry at all times; "
            "GENDER SURGERY (46,XX virilised): multidisciplinary team + ethics; deferred if possible; "
            "MONITORING: 17-OHP, androstenedione, ACTH, bone age, growth velocity, BP, weight; "
            "OVER-TREATMENT risk: linear growth impairment, obesity, Cushingoid; "
            "UNDER-TREATMENT risk: advanced bone age, early epiphyseal fusion, short adult stature; "
            "FERTILITY: females with classic CAH can conceive with good control; "
            "PREGNANCY: hydrocortisone and fludrocortisone SAFE; no dose change usually needed in 1st trimester; "
            "MLPA + sequencing for family planning and prenatal diagnosis"
        ),
        "key_ddx": (
            "CYP11B1 (CAH-11β): 17-OHP not elevated; 11-deoxycortisol elevated; HYPERTENSION (not SW); virilisation ✓; "
            "CYP17A1: 17-OHP normal; no virilisation in 46,XX; HYPERTENSION + sexual infantilism; "
            "STAR/NR0B1: all steroids absent; severe AI without virilisation in 46,XX; "
            "MC2R (FGD1): isolated cortisol deficiency, no mineralocorticoid problem, no virilisation; renin NORMAL; "
            "AAAS (Triple-A): alacrima + achalasia = key distinguisher; no virilisation; "
            "Adrenal haemorrhage (neonatal): biochemistry normal initially; imaging differentiates"
        ),
        "mineralocorticoid_status": "SW: deficient → salt-wasting; SV/NC: normal → no salt-wasting",
        "glucocorticoid_status": "Deficient (all classic); mild reduction (NC)",
        "androgen_status": "Markedly elevated (all classic + NC); 17-OHP is pathognomonic biomarker",
        "severity_weights": [0.20, 0.50, 0.30],  # mild(NC)/moderate(SV)/severe(SW)
    },

    # ── CYP11B1 — 11β-Hydroxylase Deficiency ────────────────────────────────
    {
        "gene": "CYP11B1",
        "protein": "11β-Hydroxylase (CYP11B1 / P450c11β)",
        "alias": (
            "CYP11B1; OMIM gene 202010; CAH due to 11β-Hydroxylase Deficiency #202010; "
            "8q24.3; 503 aa; ~56 kDa; AR; 2nd most common CAH (5–8% of classic CAH); "
            "prevalence 1 in 100,000; more common in Moroccan Jews (1 in 5,000–7,000); "
            "p.Arg448His (c.1343G>A) common in North African Jewish patients"
        ),
        "aa": "503 aa",
        "kDa": "~56 kDa",
        "locus": "8q24.3",
        "omim_gene": 202010,
        "omim_disease": 202010,
        "inheritance": "AR (biallelic); higher prevalence in consanguineous populations",
        "gene_class": (
            "Mitochondrial cytochrome P450 enzyme (CYP11B1); "
            "catalyses 11β-hydroxylation: 11-deoxycortisol → cortisol; and 11-deoxycorticosterone (DOC) → corticosterone; "
            "CYP11B1 (P450c11β) vs CYP11B2 (P450c11AS / aldosterone synthase): "
            "same chromosomal region (8q24.3), 93% amino acid identity, BUT different subcellular function; "
            "CYP11B1: expressed in zona fasciculata → cortisol pathway; "
            "CYP11B2: expressed in zona glomerulosa → aldosterone pathway; "
            "LOF CYP11B1 → cortisol deficiency → ACTH elevation → DOC accumulation + androgen excess; "
            "DOC (11-deoxycorticosterone) accumulates → MINERALOCORTICOID EXCESS → HYPERTENSION + HYPOKALAEMIA; "
            "ALDOSTERONE itself is SUPPRESSED (because DOC acts as mineralocorticoid suppressing renin → aldosterone); "
            "KEY DDx from CYP21A2: NO SALT-WASTING (DOC provides mineralocorticoid activity); "
            "virilisation ≡ CYP21A2 (both ↑ androgen); HYPERTENSION distinguishes CYP11B1; "
            "11-deoxycortisol (compound S) and DOC markedly elevated — DIAGNOSTIC biomarkers; "
            "17-OHP mildly elevated (substrate crossover via adrenal androgen pathway) — "
            "NBS may falsely flag 17-OHP but much lower than CYP21A2; "
            "Moroccan Jewish founder: p.Arg448His (c.1343G>A) in CYP11B1"
        ),
        "phenotype": (
            "KEY FEATURES: "
            "Glucocorticoid deficiency (cortisol) with compensatory ACTH excess; "
            "MINERALOCORTICOID EXCESS (DOC accumulation, NOT aldosterone): "
            "  → HYPERTENSION (may be severe; diastolic HTN; appears in childhood/adolescence); "
            "  → HYPOKALAEMIA (DOC mineralocorticoid effect); "
            "  → LOW RENIN (suppressed by DOC mineralocorticoid effect); "
            "  → SUPPRESSED ALDOSTERONE (low — DDx from hyperaldosteronism); "
            "ANDROGEN EXCESS (ACTH-driven): "
            "  → 46,XX: virilised genitalia at birth (Prader stage I–III typically less severe than CYP21A2); "
            "  → Both sexes: postnatal androgen excess → advanced bone age, precocious pseudo-puberty; "
            "ABSENCE OF SALT-WASTING (mineralocorticoid excess prevents SW); "
            "Biochemistry: 11-deoxycortisol elevated (>100 nmol/L after Synacthen); DOC elevated; "
            "17-OHP mildly elevated (can cause NBS false-positive, but much lower than CYP21A2); "
            "Plasma renin: suppressed; Aldosterone: suppressed/low"
        ),
        "hallmark": (
            "HYPERTENSION IN CAH = CYP11B1 UNTIL PROVEN OTHERWISE; "
            "combination VIRILISATION + HYPERTENSION + NO SALT-WASTING = CYP11B1 clinical fingerprint; "
            "11-DEOXYCORTISOL MARKEDLY ELEVATED: PATHOGNOMONIC — order 11-deoxycortisol (compound S) as first test; "
            "DOC ELEVATED: explains hypertension (mineralocorticoid excess); "
            "RENIN SUPPRESSED (not elevated as in CYP21A2-SW): key DDx from CYP21A2; "
            "NBS: 17-OHP may be mildly elevated causing false-positive — confirm with 11-deoxycortisol + DOC; "
            "SPIRONOLACTONE RISKS in acute phase: may precipitate crisis by competing with DOC; "
            "TREATMENT IS HYDROCORTISONE (suppresses ACTH → lowers DOC → BP normalises); "
            "DO NOT use antihypertensives alone — ACTH suppression by HC is the definitive BP treatment; "
            "Moroccan Jewish ethnicity → p.Arg448His founder: targeted sequencing first; "
            "MLPA: less critical than CYP21A2 (fewer deletions) but still prudent"
        ),
        "treatment_alert": (
            "HYDROCORTISONE: ACTH suppression lowers DOC → treats hypertension without antihypertensives; "
            "start glucocorticoid FIRST — BP will normalise with adequate cortisol replacement; "
            "ANTIHYPERTENSIVES: may be needed early but hydrocortisone is definitive; "
            "avoid spironolactone acutely (competes with DOC at MR receptor — may unmask AI); "
            "FLUDROCORTISONE: NOT needed (DOC provides mineralocorticoid activity; may worsen HTN); "
            "STRESS DOSES: same as CYP21A2 (3× HC for illness/surgery/fever); "
            "MONITORING: 11-deoxycortisol, 17-OHP, androstenedione, ACTH, renin, aldosterone, BP, K+, bone age; "
            "46,XX GENITALIA: surgical discussion; DSD multidisciplinary team; "
            "FERTILITY: possible with good control; "
            "GENETIC COUNSELLING: AR; target Moroccan Jewish community testing"
        ),
        "key_ddx": (
            "CYP21A2 (most common CAH): SALT-WASTING (not HTN); 17-OHP markedly elevated; renin elevated; "
            "CYP17A1: HTN + sexual infantilism + NO virilisation; 17-OHP low; "
            "Primary hyperaldosteronism (Conn adenoma): renin suppressed + K+ low + HTN, BUT aldosterone ELEVATED; "
            "in CYP11B1: aldosterone SUPPRESSED (DOC is the mineralocorticoid, not aldosterone); "
            "Cushing syndrome: cortisol elevated (in CYP11B1: cortisol LOW/deficient); "
            "Phaeochromocytoma: catecholamine crisis; biochemistry different; "
            "Apparent mineralocorticoid excess (AME — HSD11B2): 11-deoxycortisol normal"
        ),
        "mineralocorticoid_status": "Excess (DOC accumulates) → HTN + hypokalaemia; aldosterone itself suppressed",
        "glucocorticoid_status": "Deficient (cortisol)",
        "androgen_status": "Elevated (ACTH-driven); virilisation in 46,XX",
        "severity_weights": [0.25, 0.50, 0.25],
    },

    # ── CYP11B2 — Aldosterone Synthase Deficiency ──────────────────────────
    {
        "gene": "CYP11B2",
        "protein": "Aldosterone Synthase (CYP11B2 / P450c11AS)",
        "alias": (
            "CYP11B2; OMIM gene 124080; Aldosterone Synthase Deficiency (CMO I/II) #203400; "
            "8q24.3; 503 aa; ~56 kDa; AR; "
            "also: OMIM gene 124080 linked to Familial Hyperaldosteronism Type I (FH-I) by hybrid gene with CYP11B1; "
            "CMO-I: enzyme blocked at 18-hydroxylation step; CMO-II: enzyme blocked at 18-oxidation step; "
            "clinical distinction is biochemical; both present with salt-wasting + low aldosterone"
        ),
        "aa": "503 aa",
        "kDa": "~56 kDa",
        "locus": "8q24.3",
        "omim_gene": 124080,
        "omim_disease": 203400,
        "inheritance": "AR (biallelic); founder mutations in Iranian Jews and Arab populations",
        "gene_class": (
            "Mitochondrial cytochrome P450 enzyme (CYP11B2 / P450c11AS / aldosterone synthase); "
            "expressed EXCLUSIVELY in zona glomerulosa (unlike CYP11B1 which is in zona fasciculata); "
            "catalyses three sequential reactions on corticosterone: "
            "  (1) corticosterone → 18-hydroxycorticosterone [18-hydroxylase]; "
            "  (2) 18-hydroxycorticosterone → aldosterone [18-oxidase]; "
            "LOF → impaired or absent aldosterone synthesis → "
            "  plasma renin activity markedly elevated (angiotensin II not able to upregulate aldosterone); "
            "  aldosterone deficient → renal Na+ wasting → salt-wasting + hyperkalaemia; "
            "  17-OHP NORMAL (cortisol pathway via CYP11B1 is INTACT) — KEY DDx from CYP21A2; "
            "  androgen precursors NORMAL — NO virilisation — KEY DDx from CYP21A2 and CYP11B1; "
            "CMO-I vs CMO-II: biochemical distinction (18-OH-corticosterone/aldosterone ratio); "
            "both respond to fludrocortisone replacement; "
            "HYBRID GENE (CYP11B2/CYP11B1 fusion): unequal crossing-over between CYP11B1 and CYP11B2 → "
            "glucocorticoid-remediable aldosteronism (GRA/FH-I): ACTH-driven aldosterone excess → HTN; "
            "distinct from LOF CMO described here"
        ),
        "phenotype": (
            "PRESENTATION: "
            "Neonatal/infantile salt-wasting: vomiting, poor feeding, dehydration, hyponatraemia, hyperkalaemia; "
            "Failure to thrive; polyuria; "
            "DISTINGUISHING FEATURES FROM CYP21A2: "
            "  NO VIRILISATION (androgens normal — only mineralocorticoid pathway is affected); "
            "  17-OHP NORMAL; "
            "  No neonatal adrenal crisis beyond electrolyte imbalance; "
            "Biochemistry: aldosterone VERY LOW or undetectable; "
            "Plasma renin activity: MARKEDLY ELEVATED; "
            "18-Hydroxycorticosterone: variable depending on CMO type; "
            "Corticosterone: elevated (accumulates proximal to block); "
            "Cortisol: NORMAL; ACTH: NORMAL; "
            "With age: SW tendency IMPROVES (many children tolerate salt-loading better by adolescence); "
            "Salt-wasting ameliorates spontaneously in some after puberty (mechanism unclear: other aldosterone-independent mechanisms mature)"
        ),
        "hallmark": (
            "SALT-WASTING + NO VIRILISATION + 17-OHP NORMAL + ALDOSTERONE LOW + RENIN HIGH = CMO/CYP11B2; "
            "CORTISOL NORMAL is KEY — pure mineralocorticoid deficiency with intact glucocorticoid; "
            "HYDROCORTISONE NOT NEEDED (cortisol is normal — only fludrocortisone required); "
            "giving HC is an error — only mineralocorticoid replacement is indicated; "
            "SW IMPROVES WITH AGE: many adolescents/adults can reduce or stop fludrocortisone; "
            "DDx CYP21A2: same SW phenotype but CYP21A2 has 17-OHP elevated + virilisation; "
            "RENIN-GUIDED DOSING: target normal plasma renin activity on fludrocortisone; "
            "EXCESS FLUDROCORTISONE: hypertension + hypokalaemia (suppress renin to below normal); "
            "Founder mutations: Iranian Jews (c.788A→G p.Gln263Arg) and Arab populations"
        ),
        "treatment_alert": (
            "FLUDROCORTISONE: mineralocorticoid replacement CURATIVE (0.05–0.2 mg/day); "
            "SALT SUPPLEMENTATION: 1–3 g NaCl/day in infants; "
            "HYDROCORTISONE: NOT needed (cortisol is normal) — avoid over-treatment; "
            "MONITOR: plasma renin activity (target normal range on treatment); electrolytes; BP; "
            "OVER-TREATMENT: HTN, hypokalaemia — reduce fludrocortisone; "
            "UNDER-TREATMENT: continued SW — increase fludrocortisone + salt; "
            "NATURAL HISTORY: SW improves in adolescence; trial of dose reduction appropriate; "
            "STRESS DOSE: NOT required (glucocorticoid axis intact); "
            "EMERGENCY STEROID CARD: NOT required (no glucocorticoid deficiency); "
            "GENETIC COUNSELLING: AR; ethnic-specific founder mutation testing (Iranian Jewish, Arab)"
        ),
        "key_ddx": (
            "CYP21A2 (most common CAH-SW): virilisation ✓ + 17-OHP elevated + cortisol low; "
            "Pseudohypoaldosteronism type 1 (MLR/NCC): aldosterone elevated (not low) + renin elevated; "
            "Pseudohypoaldosteronism type 2 (WNK1/4, KLHL3): renin + aldosterone suppressed; hyperKalaemia + HTN; "
            "Glucocorticoid-remediable aldosteronism (GRA/FH-I — hybrid gene): aldosterone elevated + ACTH-regulated; "
            "Adrenal insufficiency (ACTH-dependent): cortisol also deficient; ACTH elevated; "
            "NR0B1 (AHC): all adrenal steroids deficient; X-linked; associated HH in males"
        ),
        "mineralocorticoid_status": "Deficient (aldosterone absent/low); renin markedly elevated",
        "glucocorticoid_status": "Normal (CYP11B1 and cortisol pathway intact)",
        "androgen_status": "Normal (no virilisation — androgen pathway not affected)",
        "severity_weights": [0.30, 0.50, 0.20],
    },

    # ── CYP17A1 — 17α-Hydroxylase/17,20-Lyase Deficiency ──────────────────
    {
        "gene": "CYP17A1",
        "protein": "17α-Hydroxylase / 17,20-Lyase (CYP17A1 / P450c17)",
        "alias": (
            "CYP17A1; OMIM gene 202110; 17α-Hydroxylase/17,20-Lyase Deficiency #202110; "
            "10q24.32; 508 aa; ~57 kDa; AR; "
            "distinctive presentation: combined glucocorticoid deficiency + sex steroid deficiency + "
            "mineralocorticoid EXCESS; classically presents at puberty with primary amenorrhoea (46,XX) "
            "or complete male-to-female sex reversal (46,XY DSD); "
            "Brazilian founder: p.Trp406Arg common in Brazil"
        ),
        "aa": "508 aa",
        "kDa": "~57 kDa",
        "locus": "10q24.32",
        "omim_gene": 202110,
        "omim_disease": 202110,
        "inheritance": "AR (biallelic); Brazilian founder common; some Turkish/Chinese founders",
        "gene_class": (
            "Microsomal cytochrome P450 (CYP17A1 / P450c17); "
            "bifunctional enzyme: "
            "  (1) 17α-hydroxylase activity: pregnenolone → 17-OH-pregnenolone; progesterone → 17-OHP; "
            "  (2) 17,20-lyase activity: 17-OH-pregnenolone → DHEA (adrenal androgen precursor); "
            "LOF → BLOCKS cortisol synthesis (at 17-OHP step) → ACTH elevation; "
            "LOF → BLOCKS androgen and oestrogen synthesis (at 17,20-lyase step) → "
            "  sex steroids absent (DHEA, androstenedione, testosterone, oestradiol all deficient); "
            "LOF → SHIFTS steroidogenesis toward mineralocorticoid pathway: "
            "  pregnenolone → progesterone → DOC → corticosterone accumulate; "
            "  DOC ACCUMULATION → HYPERTENSION + HYPOKALAEMIA (mineralocorticoid excess); "
            "  ALDOSTERONE itself is LOW (renin suppressed by DOC); "
            "46,XX PHENOTYPE: genetically female; ovaries present; NO virilisation (sex steroids absent); "
            "primary amenorrhoea; no breast development; no axillary/pubic hair (adrenal androgens absent); "
            "46,XY PHENOTYPE: testes present (may be cryptorchid); "
            "FEMALE external genitalia at birth (testosterone absent during fetal development); "
            "female social gender assignment often; gender dysphoria issues at puberty"
        ),
        "phenotype": (
            "TRIAD: HYPERTENSION + SEXUAL INFANTILISM/AMBIGUITY + HYPOKALAEMIA; "
            "46,XX: "
            "  Primary amenorrhoea; no secondary sexual characteristics (no breasts, no pubic/axillary hair); "
            "  Female genitalia; uterus present but atrophic; "
            "  Height TALL (no sex steroid-induced epiphyseal fusion, open growth plates); "
            "46,XY (DSD): "
            "  Phenotypically female at birth; "
            "  Testes (inguinal or intra-abdominal, not descended); "
            "  Female social gender often assigned; "
            "  Gender identity complex — multidisciplinary DSD team MANDATORY; "
            "BOTH SEXES: "
            "  HYPERTENSION (DOC mineralocorticoid excess); "
            "  HYPOKALAEMIA; "
            "  LOW RENIN (suppressed by DOC); "
            "  ELEVATED ACTH; "
            "  GLUCOCORTICOID DEFICIENCY (cortisol low) — risk of adrenal crisis with stress; "
            "Biochemistry: 17-OHP LOW (paradoxically — not elevated as in CYP21A2); "
            "DOC, progesterone, corticosterone elevated; DHEA very low; testosterone very low"
        ),
        "hallmark": (
            "HYPERTENSION + PRIMARY AMENORRHOEA + NO PUBIC HAIR + TALL STATURE = CYP17A1 until proven otherwise; "
            "17-OHP IS LOW (NOT elevated) — critical DDx from CYP21A2; "
            "MISSED AT NBS: 17-OHP screening misses CYP17A1 (presents at puberty not neonatally); "
            "46,XY DSD: testes present but phenotypically female — gonadectomy risk of malignancy; "
            "GLUCOCORTICOID CRISIS POSSIBLE with stress — even though presentation is at puberty; "
            "STRESS DOSE STEROIDS MANDATORY — educate patient/family; "
            "HYPERTENSION TREATED BY HYDROCORTISONE (ACTH suppression → DOC lowers → BP normalises); "
            "SEX STEROID REPLACEMENT: oestrogen for 46,XX after adolescence; "
            "46,XY: DSD team; gender identity discussions; gonadectomy decision (testes in situ: "
            "gonadoblastoma risk ~10–30% — timing and decision context-dependent); "
            "ALDOSTERONE: suppressed (not elevated) — use aldosterone level to DDx from Conn adenoma"
        ),
        "treatment_alert": (
            "HYDROCORTISONE: ACTH suppression lowers DOC → normalises BP + K+; give first; "
            "do not treat HTN with antihypertensives alone; "
            "STRESS DOSE STEROIDS: mandatory (3× HC for illness/surgery) despite puberty presentation; "
            "SEX STEROID REPLACEMENT: "
            "  46,XX: oestrogen (transdermal preferred) to induce puberty; progesterone cyclically; "
            "  46,XY: gender-affirming hormones based on gender identity decision; "
            "FLUDROCORTISONE: not needed (DOC is mineralocorticoid — excess, not deficiency); "
            "GONADECTOMY (46,XY): multidisciplinary decision (gonadoblastoma risk); timing varies; "
            "FERTILITY: 46,XX — potential if oestrogen-induced ovulation possible (rare); "
            "46,XY — no fertility (no spermatogenesis); "
            "GENETIC COUNSELLING: AR; Brazilian/Turkish/Chinese ethnicity — founder testing"
        ),
        "key_ddx": (
            "CYP21A2 (most common CAH): 17-OHP ELEVATED (in CYP17A1 it is LOW); virilisation ✓; SW ✓; "
            "CYP11B1: virilisation ✓; 11-deoxycortisol elevated; 17-OHP mildly up; "
            "Turner syndrome (45,X): primary amenorrhoea + sexual infantilism BUT no HTN + karyotype 45,X; "
            "CAIS (complete androgen insensitivity — AR gene): 46,XY; female phenotype; testes; "
            "BUT testosterone is ELEVATED in CAIS (not deficient); karyotype + testosterone distinguish; "
            "Mayer-Rokitansky syndrome: absent uterus/vagina; normal oestrogen; no HTN; "
            "Primary hyperaldosteronism: aldosterone ELEVATED (in CYP17A1 aldosterone is LOW)"
        ),
        "mineralocorticoid_status": "Excess (DOC accumulates) → HTN + hypokalaemia; aldosterone suppressed",
        "glucocorticoid_status": "Deficient (17α-hydroxylase block prevents cortisol synthesis)",
        "androgen_status": "Absent (17,20-lyase block prevents all sex steroids); no virilisation",
        "severity_weights": [0.20, 0.45, 0.35],
    },

    # ── StAR — Congenital Lipoid Adrenal Hyperplasia ─────────────────────
    {
        "gene": "STAR",
        "protein": "StAR (Steroidogenic Acute Regulatory Protein)",
        "alias": (
            "STAR; OMIM gene 600617; Congenital Lipoid Adrenal Hyperplasia (CLAH) #201710; "
            "8p11.23; 285 aa; ~37 kDa; AR; "
            "MOST SEVERE form of CAH — all steroid classes absent (glucocorticoids + mineralocorticoids + sex steroids); "
            "prevalence rare (~100 families reported); most common in East Asian (Korean/Japanese): "
            "p.Gln258Ter (Q258X) founder in Korean/Japanese population; "
            "lipid droplet accumulation in adrenal gland seen on MRI — PATHOGNOMONIC"
        ),
        "aa": "285 aa",
        "kDa": "~37 kDa",
        "locus": "8p11.23",
        "omim_gene": 600617,
        "omim_disease": 201710,
        "inheritance": "AR (biallelic); East Asian (Korean/Japanese) founder Q258X",
        "gene_class": (
            "Mitochondrial outer-inner membrane transfer protein (StAR — Steroidogenic Acute Regulatory Protein); "
            "NOT an enzyme — acts as transport protein/regulator; "
            "function: facilitates transfer of cholesterol from outer mitochondrial membrane → inner mitochondrial membrane; "
            "rate-limiting step in ALL steroidogenesis (cholesterol must reach CYP11A1/SCC on inner membrane); "
            "LOF → cholesterol CANNOT enter mitochondria → ALL downstream steroid synthesis ZERO; "
            "consequence: "
            "  cholesterol esters accumulate in adrenal cortex and gonadal cells → LIPID DROPLETS (visible on MRI); "
            "  physical lipid accumulation further damages cells (two-hit mechanism); "
            "ALL THREE zones of adrenal cortex affected: "
            "  zona glomerulosa → aldosterone ZERO → severe SW crisis; "
            "  zona fasciculata → cortisol ZERO → severe AI; "
            "  zona reticularis → DHEA/androgens ZERO → sex steroid deficiency; "
            "Gonads also affected: "
            "  46,XX: ovaries produce NO sex steroids before puberty (ovary is largely quiescent — lipid damage delayed); "
            "  46,XY: testes severely damaged — phenotypically FEMALE at birth (no testosterone during fetal development); "
            "testicular feminisation-like phenotype but mechanism is steroid synthesis failure, not androgen resistance"
        ),
        "phenotype": (
            "PRESENTATION — NEONATAL ADRENAL CRISIS (MOST SEVERE): "
            "Day 1–4 of life: severe SW crisis (vomiting, dehydration, hyponatraemia, hyperkalaemia, shock); "
            "WITHOUT TREATMENT: fatal; "
            "BIOCHEMISTRY: ALL steroids absent or near-zero (cortisol, aldosterone, DHEA, androstenedione); "
            "ACTH: massively elevated; renin: massively elevated; "
            "17-OHP: very low (no substrate produced even proximally); "
            "46,XX: normal female genitalia; uterus + ovaries present; "
            "46,XY (DSD): PHENOTYPICALLY FEMALE at birth (no testosterone during fetal life — female differentiation default); "
            "testes may be intra-abdominal or inguinal; "
            "IMAGING: adrenal glands enlarged with characteristic lipid accumulation on MRI/CT — PATHOGNOMONIC; "
            "GONADAL FEATURES: "
            "  46,XX — ovaries SPARED initially (ovary is relatively inactive in fetal life); "
            "  may produce oestrogen at puberty (delayed lipid damage in ovary); "
            "  spontaneous puberty reported in some 46,XX patients; "
            "  46,XY — testes massively lipid-laden; no puberty; infertility universal"
        ),
        "hallmark": (
            "ALL STEROIDS ABSENT in neonatal crisis = StAR / STAR mutation most likely; "
            "LIPID DROPLET ACCUMULATION IN ADRENAL ON MRI: PATHOGNOMONIC for CLAH; "
            "enlarged adrenals with lipid signal — request specific MRI adrenal protocol; "
            "MOST SEVERE CAH — higher mortality without rapid treatment than CYP21A2-SW; "
            "46,XY PHENOTYPICALLY FEMALE: multidisciplinary DSD; gender assignment decision complex; "
            "CORTISOL ZERO: massive ACTH; ALDOSTERONE ZERO: massive renin; both simultaneously; "
            "AVOID LIPID-LOWERING MEDICATIONS (statins, etc.): "
            "  cholesterol substrate is needed for any residual steroidogenesis; "
            "  theoretical concern (clinical evidence limited but mechanistically rational); "
            "SUPPLEMENTAL OESTROGEN NEEDED at puberty for 46,XX (ovaries may partially recover but unreliable); "
            "East Asian ethnicity → p.Gln258Ter founder: targeted sequencing first in Korean/Japanese patients"
        ),
        "treatment_alert": (
            "IMMEDIATE STRESS STEROIDS: IV hydrocortisone + IV saline + dextrose at diagnosis; "
            "do not wait for biochemistry before starting in sick neonate with suspected AI; "
            "HYDROCORTISONE: 10–15 mg/m²/day maintenance in 3 doses; "
            "FLUDROCORTISONE: 0.05–0.2 mg/day (mineralocorticoid replacement); "
            "SALT: 1–2 g NaCl/day in infancy; "
            "EMERGENCY STEROID CARD + IM hydrocortisone kit; "
            "STRESS DOSES: 3× HC for fever/vomiting/surgery; "
            "SEX STEROID REPLACEMENT: oestrogen for 46,XX at puberty (11–12 years); "
            "progesterone cyclically for uterine health in 46,XX; "
            "46,XY: gender-affirming hormones based on gender identity; "
            "LIPID-LOWERING AVOID: statins/fibrates may reduce cholesterol substrate; "
            "MONITORING: all steroid levels essentially undetectable; use ACTH and renin as surrogates; "
            "GONADECTOMY (46,XY): gonadoblastoma risk in intra-abdominal testes; DSD team decision"
        ),
        "key_ddx": (
            "CYP21A2 (CAH21): 17-OHP elevated; aldosterone may be partially present; virilisation in 46,XX; "
            "NR0B1 (AHC X-linked): X-linked (males only severely affected); no lipid droplet imaging; "
            "MC2R (FGD1): isolated cortisol deficiency only — aldosterone NORMAL; less severe; "
            "AAAS: alacrima + achalasia first; milder AI; "
            "NNS (Neonatal adrenal haemorrhage): imaging shows haemorrhage not lipid; biochemistry normalises; "
            "Wolman disease (LIPA): also lipid accumulation; but hepatosplenomegaly + calcified adrenals; "
            "distinct biochemistry (lysosomal acid lipase deficiency)"
        ),
        "mineralocorticoid_status": "Absent (zero aldosterone) — most severe SW",
        "glucocorticoid_status": "Absent (zero cortisol) — life-threatening AI",
        "androgen_status": "Absent (zero sex steroids) — 46,XY phenotypically female",
        "severity_weights": [0.05, 0.15, 0.80],
    },

    # ── NR0B1 — Adrenal Hypoplasia Congenita (X-linked) ──────────────────
    {
        "gene": "NR0B1",
        "protein": "DAX-1 (Dosage-Sensitive Sex Reversal — Adrenal Hypoplasia Congenita Nuclear Receptor)",
        "alias": (
            "NR0B1 (previously DAX1); OMIM gene 300473; Adrenal Hypoplasia Congenita, X-linked (AHC) #300200; "
            "Xp21.2; 470 aa; ~54 kDa; X-linked (males severely affected, females generally carriers); "
            "presents with primary adrenal failure in infancy/childhood + hypogonadotropic hypogonadism (HH) in adolescence; "
            "contiguous gene deletion syndrome: Xp21 deletion may involve DMD (Duchenne MD) + GK gene (glycerol kinase deficiency)"
        ),
        "aa": "470 aa",
        "kDa": "~54 kDa",
        "locus": "Xp21.2",
        "omim_gene": 300473,
        "omim_disease": 300200,
        "inheritance": "X-linked (XL); males severely affected; females carriers (may have partial HH rarely)",
        "gene_class": (
            "Nuclear receptor superfamily protein (NR0B1 / DAX-1); "
            "atypical nuclear receptor: lacks conventional DNA-binding zinc-finger domain; "
            "acts as repressor of steroidogenic gene transcription (via interaction with SF-1/NR5A1); "
            "essential roles: "
            "  (1) adrenal cortex development and maintenance (all three zones); "
            "  (2) gonadal development (gonads and hypothalamic-pituitary axis); "
            "LOF → adrenal cortex fails to develop/maintain → primary adrenal insufficiency; "
            "LOF → impaired GnRH secretion from hypothalamus → impaired LH/FSH secretion from pituitary → "
            "  hypogonadotropic hypogonadism (HH) — bilateral combination unique to NR0B1; "
            "CONTIGUOUS GENE DELETION at Xp21: NR0B1 (AHC) + GK (glycerol kinase — elevated plasma glycerol) + "
            "  DMD (Duchenne muscular dystrophy — proximal muscle weakness + elevated CK); "
            "Check CK and glycerol in males with AHC — order multiplex ligation or aCGH to detect Xp21 deletions; "
            "Female carriers: usually phenotypically normal; rarely delayed puberty; "
            "SF-1 (NR5A1) LOF: different gene but overlapping adrenal/gonadal phenotype (autosomal)"
        ),
        "phenotype": (
            "PRIMARY PRESENTATION (INFANCY/EARLY CHILDHOOD): "
            "Adrenal crisis: vomiting, dehydration, hyponatraemia, hyperkalaemia (mineralocorticoid + glucocorticoid deficiency); "
            "can present as late as early childhood (not always neonatal unlike CYP21A2-SW or STAR); "
            "BIOCHEMISTRY (ADRENAL FAILURE): "
            "  Cortisol: LOW; ACTH: elevated; "
            "  Aldosterone: LOW; renin: elevated; "
            "  ALL steroid classes deficient (similar to STAR but different mechanism); "
            "  17-OHP: LOW (hypoplastic adrenal produces nothing); "
            "SECONDARY PRESENTATION (ADOLESCENCE): "
            "  Hypogonadotropic hypogonadism: absent/delayed puberty in males; "
            "  LH/FSH: LOW despite absent puberty; testosterone: very LOW; "
            "  Absent testicular enlargement; azoospermia; infertility; "
            "CONTIGUOUS GENE DELETION FEATURES (if Xp21 deletion): "
            "  Elevated CK + proximal weakness (DMD); elevated plasma glycerol (GK deficiency); "
            "Adrenal gland: hypoplastic on imaging (small adrenals) DDx from STAR (enlarged with lipid)"
        ),
        "hallmark": (
            "PRIMARY ADRENAL FAILURE IN A MALE INFANT + DELAYED PUBERTY LATER = NR0B1/DAX1; "
            "COMBINED AI + HH is the defining feature — both occur in the SAME patient; "
            "HH typically manifests in adolescence even if AI presented in infancy; "
            "SMALL ADRENALS ON IMAGING: adrenal hypoplasia (STAR has enlarged lipid-laden adrenals — key DDx); "
            "XLINKED: only males severely affected; test mother as obligate carrier; "
            "CONTIGUOUS GENE DELETION SCREEN: always check CK (DMD) and glycerol (GK) in males with AHC; "
            "order Xp21 multiplex/MLPA or aCGH to detect large deletions; point mutations in NR0B1 also exist; "
            "FERTILITY: profound HH — gonadotropin therapy can achieve spermatogenesis in some; "
            "  early pulsatile GnRH or gonadotropin treatment pre-puberty may preserve sperm production potential; "
            "HH TREATMENT PRIORITY: testosterone for virilisation; but if fertility desired → gonadotropins/GnRH pump first; "
            "STEROID EMERGENCY CARD: mandatory (full adrenal insufficiency)"
        ),
        "treatment_alert": (
            "HYDROCORTISONE: 10–15 mg/m²/day in 3 doses (glucocorticoid replacement); "
            "FLUDROCORTISONE: 0.05–0.15 mg/day (mineralocorticoid replacement); "
            "SALT in infancy: 1–2 g/day NaCl; "
            "STRESS DOSES: 3× HC for fever/vomiting/surgery; IM kit; emergency card; "
            "HYPOGONADOTROPIC HH TREATMENT: "
            "  Testosterone replacement: virilisation (puberty induction) — IM testosterone or transdermal; "
            "  FERTILITY: gonadotropin therapy (hCG + FSH/menotropins) or pulsatile GnRH pump; "
            "  initiate gonadotropins BEFORE or at start of testosterone (to prime gonads); "
            "MONITORING: cortisol, ACTH, renin, aldosterone, testosterone, LH, FSH, bone age; "
            "Xp21 DELETION WORKUP: CK (DMD), plasma glycerol (GK deficiency), aCGH/MLPA; "
            "GENETIC COUNSELLING: X-linked; carrier females usually normal; "
            "  test female relatives (obligate carriers); prenatal diagnosis available"
        ),
        "key_ddx": (
            "STAR (CLAH): similar full adrenal failure + 46,XY female phenotype; BUT STAR has LIPID-ENLARGED adrenals; "
            "CYP21A2 (most common CAH): 17-OHP elevated; no HH; adrenals enlarged (not hypoplastic); virilised 46,XX; "
            "SF-1/NR5A1: autosomal dominant; adrenal failure + gonadal dysgenesis; both sexes affected; "
            "Kallmann syndrome (KAL1/ANOS1, FGFR1): HH + anosmia; no adrenal involvement; "
            "X-linked adrenoleukodystrophy (ABCD1): different mechanism; VLCFA elevated; white matter changes"
        ),
        "mineralocorticoid_status": "Deficient (adrenal hypoplasia — aldosterone absent)",
        "glucocorticoid_status": "Deficient (adrenal hypoplasia — cortisol absent)",
        "androgen_status": "Deficient at adrenal + testicular level; HH causes low testosterone at puberty",
        "severity_weights": [0.10, 0.35, 0.55],
    },

    # ── MC2R — Familial Glucocorticoid Deficiency Type 1 ──────────────────
    {
        "gene": "MC2R",
        "protein": "MC2R (Melanocortin 2 Receptor / ACTH Receptor)",
        "alias": (
            "MC2R; OMIM gene 607397; Familial Glucocorticoid Deficiency Type 1 (FGD1) #202200; "
            "18p11.21; 297 aa; ~35 kDa; AR; "
            "ACTH receptor on adrenal cortex zona fasciculata; "
            "LOF → adrenal cortex cannot respond to ACTH → isolated glucocorticoid deficiency; "
            "MINERALOCORTICOIDS NORMAL (zona glomerulosa responds to angiotensin II, not ACTH); "
            "DISTINCTIVE: tall stature + hyperpigmentation in a child with no salt-wasting"
        ),
        "aa": "297 aa",
        "kDa": "~35 kDa",
        "locus": "18p11.21",
        "omim_gene": 607397,
        "omim_disease": 202200,
        "inheritance": "AR (biallelic); various founder mutations; West African populations more common",
        "gene_class": (
            "G-protein-coupled receptor (GPCR) — Melanocortin 2 receptor (MC2R); "
            "ACTH binds MC2R → Gs → cAMP → PKA → steroidogenesis (StAR induction → cortisol); "
            "MC2R expressed on zona fasciculata (glucocorticoid zone) — regulated by ACTH; "
            "IMPORTANT: zona glomerulosa (aldosterone) is regulated primarily by angiotensin II + K+, NOT ACTH; "
            "LOF MC2R → zona fasciculata cannot be stimulated by ACTH → cortisol deficient; "
            "→ feedback failure → ACTH MARKEDLY ELEVATED (no cortisol feedback); "
            "→ mineralocorticoids produced normally via angiotensin II on zona glomerulosa; "
            "→ RENIN NORMAL (no mineralocorticoid deficiency); "
            "ACTH ELEVATION EXPLAINS: "
            "  hyperpigmentation (ACTH stimulates MC1R on melanocytes); "
            "  TALL STATURE (ACTH has mild growth-promoting effect via IGF-1 interactions); "
            "MRAP (melanocortin-2 receptor accessory protein): essential co-receptor for MC2R trafficking; "
            "MRAP mutations → FGD2 (clinically identical to FGD1); "
            "50% of FGD cases: MC2R + MRAP; remaining: unknown mutations or AAAS (Triple-A)"
        ),
        "phenotype": (
            "PRESENTATION (INFANCY TO CHILDHOOD): "
            "Hypoglycaemia: cortisol deficiency → impaired gluconeogenesis → hypoglycaemia (often presenting symptom); "
            "Neonatal hypoglycaemia or prolonged jaundice; "
            "Recurrent hypoglycaemia + seizures in infancy; "
            "NO SALT-WASTING (mineralocorticoids normal) — distinguishes from most CAH; "
            "NO ADRENAL CRISIS in same pattern as CAH21-SW (but cortisol crises DO occur with stress); "
            "PHYSICAL FEATURES: "
            "  TALL STATURE: ACTH has growth-promoting properties; consistently tall for age; "
            "  HYPERPIGMENTATION: generalised (MC1R stimulation by excess ACTH); "
            "  facial freckles; buccal mucosa pigmentation; "
            "BIOCHEMISTRY: "
            "  Cortisol: LOW or absent; ACTH: MARKEDLY ELEVATED (often >1000 ng/L); "
            "  Aldosterone: NORMAL; Renin: NORMAL (KEY DDx from mineralocorticoid-deficient forms); "
            "  Electrolytes: NORMAL Na+, K+ (NO SW); "
            "  Blood glucose: LOW (cortisol deficiency)"
        ),
        "hallmark": (
            "TALL + HYPERPIGMENTED CHILD WITH HYPOGLYCAEMIA AND NO SW = FGD1 (MC2R); "
            "RENIN NORMAL is KEY — mineralocorticoids are fine (no SW); "
            "ACTH MASSIVELY ELEVATED: driving hyperpigmentation and growth; "
            "ISOLATED GLUCOCORTICOID DEFICIENCY: treat with HC only — no fludrocortisone needed; "
            "FLUDROCORTISONE IS AN ERROR: mineralocorticoid axis is intact; "
            "NO EMERGENCY STEROID CARD FOR SALT — only for cortisol stress doses; "
            "STRESS DOSES MANDATORY: cortisol is deficient — any illness/surgery/fever → 3× HC; "
            "TALL STATURE: not pathological in FGD — ACTH-mediated; no intervention needed; "
            "HYPERPIGMENTATION resolves or fades with adequate HC (as ACTH suppresses with treatment); "
            "MRAP gene: if MC2R sequence negative → test MRAP (FGD2 — identical phenotype); "
            "SYNACTHEN TEST: ABSENT cortisol response (confirms ACTH-receptor dysfunction)"
        ),
        "treatment_alert": (
            "HYDROCORTISONE: glucocorticoid replacement (10–15 mg/m²/day in 3 doses); "
            "FLUDROCORTISONE: NOT needed (mineralocorticoids normal); prescribing it is a clinical error; "
            "STRESS DOSES: 3× HC for fever/vomiting/surgery; IM hydrocortisone emergency kit; "
            "EMERGENCY STEROID CARD: YES (for cortisol stress doses — not for salt loss); "
            "HYPOGLYCAEMIA PREVENTION: regular feeding; no prolonged fasting; dextrose if vomiting; "
            "MONITORING: cortisol, ACTH, growth velocity, bone age, electrolytes (for over-treatment); "
            "OVER-TREATMENT (excessive HC): growth suppression + Cushingoid; titrate to minimum effective dose; "
            "TALL STATURE: observe; avoid GH — ACTH already driving growth; "
            "GENETIC COUNSELLING: AR; test MRAP if MC2R negative"
        ),
        "key_ddx": (
            "CYP21A2 (most common CAH): 17-OHP elevated; SW + virilisation; renin elevated; "
            "AAAS (Triple-A): alacrima + achalasia distinguishes; same isolated glucocorticoid deficiency; "
            "Addison disease (autoimmune): anti-21OH antibodies; mineralocorticoids also deficient (renin elevated); "
            "ACTH deficiency (secondary AI): ACTH LOW (not high) + no hyperpigmentation; "
            "NR0B1 (AHC): X-linked; mineralocorticoids also deficient; HH in adolescence; "
            "STAR: all steroids absent; 46,XY phenotypically female; "
            "Glucocorticoid resistance (NR3C1): ACTH high + hyperpigmentation BUT cortisol elevated (not deficient)"
        ),
        "mineralocorticoid_status": "Normal (ACTH does not regulate aldosterone; zona glomerulosa intact)",
        "glucocorticoid_status": "Absent/markedly deficient (cortisol); ACTH massively elevated",
        "androgen_status": "Mildly elevated (ACTH drives adrenal androgen excess via intact 17,20-lyase)",
        "severity_weights": [0.15, 0.55, 0.30],
    },

    # ── AAAS — Triple-A Syndrome / Allgrove Syndrome ──────────────────────
    {
        "gene": "AAAS",
        "protein": "Aladin (WD-Repeat Nucleoporin / Triple-A Syndrome Protein)",
        "alias": (
            "AAAS; OMIM gene 605378; Allgrove Syndrome / Triple-A Syndrome #231550; "
            "12q13.13; 547 aa; ~60 kDa; AR; "
            "TRIAD: Alacrima (absent tearing — earliest and most consistent feature) + "
            "Achalasia (oesophageal smooth muscle failure) + ACTH-resistant Adrenal Insufficiency; "
            "ALACRIMA IS THE FIRST SIGN — present from infancy, before achalasia or AI develops; "
            "progressive autonomic neuropathy + sensorimotor neuropathy develop with time; "
            "Allgrove (1978) original description; 'Triple-A' or 'Allgrove syndrome'"
        ),
        "aa": "547 aa",
        "kDa": "~60 kDa",
        "locus": "12q13.13",
        "omim_gene": 605378,
        "omim_disease": 231550,
        "inheritance": "AR (biallelic); numerous private mutations; consanguineous families",
        "gene_class": (
            "WD-repeat containing nucleoporin (Aladin); "
            "component of nuclear pore complex (NPC) — specifically the cytoplasmic face; "
            "molecular function: regulates nuclear transport of proteins involved in oxidative stress defence; "
            "LOF → impaired nuclear import of DNA repair/antioxidant proteins → oxidative damage accumulates in: "
            "  adrenal cortex → ACTH-resistant glucocorticoid deficiency; "
            "  oesophageal smooth muscle → achalasia; "
            "  lacrimal glands → alacrima (absent tears); "
            "  autonomic nervous system → progressive autonomic neuropathy; "
            "  sensorimotor peripheral nerves → progressive neuropathy; "
            "ACTH RESISTANCE: zona fasciculata damaged by oxidative stress → cannot respond to ACTH; "
            "SIMILAR TO MC2R/FGD: isolated glucocorticoid deficiency (mineralocorticoids usually NORMAL initially); "
            "but unlike MC2R: progressive neurological involvement + autonomic neuropathy; "
            "SOME PATIENTS: mineralocorticoid deficiency develops later (disease progression); "
            "ALACRIMA: complete absence of reflex tearing from infancy — use Schirmer test; "
            "ACHALASIA: failure of lower oesophageal sphincter relaxation → dysphagia + regurgitation + weight loss"
        ),
        "phenotype": (
            "TRIAD (not always present simultaneously): "
            "1. ALACRIMA (100% — first sign, from birth/infancy): "
            "   absent/markedly reduced reflex and psychic tearing; "
            "   dry eyes; recurrent corneal erosions; photophobia; eye infections; "
            "   Schirmer test: 0–3 mm (severely reduced); "
            "2. ACHALASIA (75–85%, childhood/adolescence): "
            "   dysphagia (solids > liquids initially); regurgitation; "
            "   weight loss; recurrent aspiration; oesophageal dilatation on barium swallow; "
            "   oesophageal manometry: absent peristalsis + high LOS pressure + impaired relaxation; "
            "3. ADRENAL INSUFFICIENCY (60–80%, childhood): "
            "   glucocorticoid deficiency (ACTH-resistant); "
            "   hyperpigmentation; fatigue; hypoglycaemia; "
            "   mineralocorticoids: usually preserved early (can deplete over years); "
            "PROGRESSIVE NEUROLOGICAL (develops later, variable): "
            "   autonomic neuropathy: postural hypotension, bladder dysfunction, sweating abnormalities; "
            "   sensorimotor peripheral neuropathy; "
            "   upper motor neuron signs; bulbar weakness; "
            "   cognitive impairment (reported in minority); "
            "BIOCHEMISTRY: ACTH markedly elevated; cortisol low; renin often normal (early)"
        ),
        "hallmark": (
            "DRY EYES FROM INFANCY IS THE EARLIEST SIGN — Schirmer test on any child presenting with adrenal crisis; "
            "ALACRIMA PRECEDES ACHALASIA AND AI — children may have alacrima for years before AI/achalasia appears; "
            "SWALLOWING DIFFICULTY + ADRENAL CRISIS + DRY EYES = AAAS (consider even if triad not complete); "
            "ACTH-RESISTANT AI: adrenal gland cannot respond even with high ACTH; "
            "SYNACTHEN TEST: cortisol ABSENT/subnormal despite high baseline ACTH; "
            "MINERALOCORTICOIDS INITIALLY PRESERVED: may not need fludrocortisone early; "
            "monitor renin/aldosterone yearly — can develop mineralocorticoid deficiency; "
            "OPHTHALMOLOGY: lubricating eye drops ESSENTIAL to prevent corneal damage; "
            "punctal plugs if severe; scleral lenses; "
            "ACHALASIA MANAGEMENT: pneumatic balloon dilation (preferred first-line); Heller myotomy; "
            "  AVOID prolonged achalasia — aspiration risk + nutrition impairment; "
            "NEUROLOGICAL: no DMT; physiotherapy; autonomic management (stockings, fludrocortisone may help OH); "
            "PROGRESSIVE DISEASE: neurological worsening despite good endocrine control"
        ),
        "treatment_alert": (
            "HYDROCORTISONE: glucocorticoid replacement (10–15 mg/m²/day in 3 doses); "
            "FLUDROCORTISONE: may or may not be needed — check renin/aldosterone annually; "
            "STRESS DOSES: 3× HC for fever/vomiting/surgery; IM hydrocortisone kit mandatory; "
            "EMERGENCY STEROID CARD: mandatory (cortisol deficiency); "
            "EYE CARE: preservative-free lubricating eye drops HOURLY during waking hours (dry eye disease); "
            "NIGHT OINTMENT: prevent nocturnal corneal exposure; "
            "ACHALASIA: "
            "  Pneumatic balloon dilation: first-line (75% effective; repeat as needed); "
            "  Heller myotomy + fundoplication: surgical option; "
            "  Peroral endoscopic myotomy (POEM): emerging technique; "
            "  NG tube nutrition if severe weight loss pre-treatment; "
            "AUTONOMIC NEUROPATHY: compression stockings; fludrocortisone may help orthostatic hypotension; "
            "NEUROLOGICAL: no disease-modifying therapy; rehabilitation; physiotherapy; "
            "MONITORING: ACTH, cortisol, renin, aldosterone, Schirmer test, oesophageal dilatation, neurological exam"
        ),
        "key_ddx": (
            "MC2R (FGD1/FGD2): isolated GC deficiency + hyperpigmentation + tall stature; "
            "NO alacrima; NO achalasia; NO progressive neuropathy; MRAP if MC2R negative; "
            "Addison disease (autoimmune): anti-21-OH antibodies; mineralocorticoids also deficient; "
            "ACTH deficiency (pituitary): ACTH LOW (not high); no hyperpigmentation; no alacrima/achalasia; "
            "Sjögren syndrome: dry eyes (alacrima) but autoimmune + primary + adult; anti-Ro/La antibodies; "
            "Achalasia alone: isolated; no adrenal or lacrimal involvement; "
            "NR0B1 (AHC): X-linked; no alacrima/achalasia; HH in adolescence; "
            "Neurodegeneration with adrenal insufficiency: several ultra-rare conditions exist; AAAS has NPC biology"
        ),
        "mineralocorticoid_status": "Initially normal; may progress to deficiency — monitor renin annually",
        "glucocorticoid_status": "ACTH-resistant deficiency; cortisol absent despite high ACTH",
        "androgen_status": "Mild excess (ACTH elevation drives adrenal androgen production slightly)",
        "severity_weights": [0.20, 0.50, 0.30],
    },
]


def _make_cohort(gene_data, seed):
    """Generate a 40-patient synthetic cohort for one gene."""
    rng = random.Random(seed)

    pts = []
    sev_labels = ["mild", "moderate", "severe"]
    for i in range(40):
        sev = rng.choices(sev_labels, weights=gene_data["severity_weights"])[0]

        pt = {
            "patient_id": f"{gene_data['gene']}-{seed:04d}-{i+1:02d}",
            "gene": gene_data["gene"],
            "seed": seed,
            "severity": sev,
            "age_at_diagnosis_years": round(rng.uniform(0, 25), 1),
            "sex": rng.choice(["M", "F"]),
            "adrenal_crisis_at_presentation": rng.random() < (0.8 if sev == "severe" else 0.4 if sev == "moderate" else 0.1),
            "salt_wasting": rng.random() < (0.75 if gene_data["gene"] in ("CYP21A2", "STAR", "NR0B1") else 0.0 if gene_data["gene"] in ("CYP11B1", "CYP17A1", "MC2R") else 0.5 if gene_data["gene"] == "CYP11B2" else 0.2),
            "hypertension": rng.random() < (0.85 if gene_data["gene"] in ("CYP11B1", "CYP17A1") else 0.05),
            "virilisation_46xx": rng.random() < (0.90 if gene_data["gene"] in ("CYP21A2", "CYP11B1") else 0.0),
            "sexual_infantilism": rng.random() < (0.95 if gene_data["gene"] in ("CYP17A1", "NR0B1") else 0.05),
            "hyperpigmentation": rng.random() < (0.85 if gene_data["gene"] in ("MC2R", "AAAS", "STAR", "NR0B1") else 0.30),
            "tall_stature": rng.random() < (0.75 if gene_data["gene"] == "MC2R" else 0.05),
            "alacrima": rng.random() < (0.98 if gene_data["gene"] == "AAAS" else 0.0),
            "achalasia": rng.random() < (0.80 if gene_data["gene"] == "AAAS" else 0.0),
            "hypogonadotropic_hh": rng.random() < (0.90 if gene_data["gene"] == "NR0B1" else 0.03),
            "lipid_adrenal_accumulation": rng.random() < (0.95 if gene_data["gene"] == "STAR" else 0.0),
            "hypoglycaemia": rng.random() < (0.80 if gene_data["gene"] in ("MC2R", "AAAS", "STAR", "NR0B1") else 0.35 if gene_data["gene"] == "CYP21A2" else 0.10),
            "on_hydrocortisone": rng.random() < (0.95 if sev != "mild" else 0.30),
            "on_fludrocortisone": rng.random() < (0.90 if gene_data["gene"] in ("CYP21A2", "STAR", "NR0B1") else 0.60 if gene_data["gene"] == "CYP11B2" else 0.05),
            "acth_elevated": rng.random() < (0.95 if gene_data["gene"] in ("MC2R", "AAAS", "STAR", "NR0B1") else 0.70),
        }
        pts.append(pt)
    return pts


# Pre-build cohorts at import time
_ALL_COHORTS = {}
for _idx, _gd in enumerate(ADRENAL_GENES):
    _seed = SEED_BASE + _idx
    _ALL_COHORTS[_gd["gene"]] = _make_cohort(_gd, _seed)


def _pct(pts, key):
    return round(100 * sum(1 for p in pts if p.get(key)) / max(len(pts), 1))


def get_overview():
    all_pts = [p for pts in _ALL_COHORTS.values() for p in pts]
    genes = [g["gene"] for g in ADRENAL_GENES]
    return {
        "atlas_name": "Adrenal Disorders Atlas",
        "atlas_subtitle": (
            "Complete 8-Gene Hereditary Adrenal & Steroidogenesis Disorders Reference — "
            "CYP21A2 · CYP11B1 · CYP11B2 · CYP17A1 · STAR · NR0B1 · MC2R · AAAS"
        ),
        "n_genes": 8,
        "n_patients": len(all_pts),
        "seeds": "1254–1261",
        "genes": genes,
        "description": (
            "This atlas covers the eight primary hereditary adrenal insufficiency and steroidogenesis "
            "disorders in clinical genetics: 21-hydroxylase deficiency (CYP21A2; most common CAH, 90–95%), "
            "11β-hydroxylase deficiency (CYP11B1; hypertension-CAH), "
            "aldosterone synthase deficiency (CYP11B2; isolated mineralocorticoid deficiency), "
            "17α-hydroxylase/17,20-lyase deficiency (CYP17A1; hypertension + sexual infantilism), "
            "congenital lipoid adrenal hyperplasia (STAR; most severe — all steroids absent), "
            "X-linked adrenal hypoplasia congenita (NR0B1/DAX1; adrenal failure + HH), "
            "familial glucocorticoid deficiency type 1 (MC2R; isolated cortisol with tall stature + hyperpigmentation), "
            "and Triple-A/Allgrove syndrome (AAAS; alacrima + achalasia + ACTH-resistant AI). "
            "Critical drug alerts: fludrocortisone is NOT needed in CYP17A1/CYP11B1 (mineralocorticoid excess — "
            "would worsen hypertension); stress dose steroids mandatory in all glucocorticoid-deficient forms; "
            "MLPA mandatory for CYP21A2 (pseudogene conversions missed by sequencing alone)."
        ),
        "aggregate_clinical": {
            "salt_wasting_pct": _pct(all_pts, "salt_wasting"),
            "hypertension_pct": _pct(all_pts, "hypertension"),
            "virilisation_pct": _pct(all_pts, "virilisation_46xx"),
            "hyperpigmentation_pct": _pct(all_pts, "hyperpigmentation"),
            "hypoglycaemia_pct": _pct(all_pts, "hypoglycaemia"),
            "adrenal_crisis_pct": _pct(all_pts, "adrenal_crisis_at_presentation"),
            "alacrima_pct": _pct(all_pts, "alacrima"),
            "achalasia_pct": _pct(all_pts, "achalasia"),
            "hh_pct": _pct(all_pts, "hypogonadotropic_hh"),
            "on_fludrocortisone_pct": _pct(all_pts, "on_fludrocortisone"),
        },
        "drug_alerts": [
            {
                "title": "CYP21A2 — MLPA MANDATORY (pseudogene conversions missed by sequencing alone)",
                "body": (
                    "CYP21A2 is adjacent to the pseudogene CYP21A1P. Gene conversions from the pseudogene "
                    "account for >75% of pathogenic alleles. Standard sequencing MISSES large deletions and gene "
                    "conversions. Order MLPA + sequencing together. A 'negative' CYP21A2 sequencing result is "
                    "insufficient — always confirm with MLPA."
                ),
            },
            {
                "title": "CYP11B1 & CYP17A1 — FLUDROCORTISONE CONTRAINDICATED (mineralocorticoid excess)",
                "body": (
                    "Both CYP11B1 and CYP17A1 cause mineralocorticoid EXCESS (via DOC accumulation), "
                    "presenting with hypertension and hypokalaemia. Adding fludrocortisone would dangerously "
                    "worsen hypertension. Treat with hydrocortisone (ACTH suppression lowers DOC) — "
                    "do NOT prescribe fludrocortisone in these conditions."
                ),
            },
            {
                "title": "STAR (Lipoid CAH) — AVOID LIPID-LOWERING AGENTS (statins/fibrates)",
                "body": (
                    "StAR protein facilitates cholesterol transport into mitochondria for all steroidogenesis. "
                    "Reducing cholesterol substrate with statins/fibrates may further impair any residual "
                    "steroidogenic capacity. Avoid lipid-lowering agents in CLAH patients."
                ),
            },
            {
                "title": "AAAS (Triple-A) — ALACRIMA IS THE EARLIEST SIGN (before AI or achalasia)",
                "body": (
                    "Alacrima (absent tearing) is present from birth or early infancy — years before adrenal "
                    "insufficiency or achalasia develops. Perform Schirmer test in every child presenting with "
                    "adrenal crisis to detect Triple-A syndrome. Lubricating eye drops must be started "
                    "immediately to prevent corneal erosion and blindness."
                ),
            },
            {
                "title": "ALL FORMS — STRESS DOSE STEROIDS + EMERGENCY STEROID CARD MANDATORY",
                "body": (
                    "Every patient with glucocorticoid deficiency (CYP21A2, CYP11B1, CYP17A1, STAR, NR0B1, "
                    "MC2R, AAAS) must carry an emergency steroid card and IM hydrocortisone kit. "
                    "Sick-day rule: 3× normal HC dose for fever/vomiting; IV HC 100 mg/m²/day for surgery. "
                    "CYP11B2 (isolated aldosterone deficiency) does NOT need stress glucocorticoid cover "
                    "as cortisol axis is intact."
                ),
            },
            {
                "title": "NR0B1 (AHC) — CONTIGUOUS GENE DELETION: check CK + glycerol for DMD + GK deficiency",
                "body": (
                    "NR0B1 at Xp21.2 may be deleted as part of a larger Xp21 contiguous gene deletion "
                    "involving DMD (Duchenne muscular dystrophy, CK elevated) and GK (glycerol kinase "
                    "deficiency, plasma glycerol elevated). Always check CK and plasma glycerol in males "
                    "with X-linked adrenal hypoplasia congenita. Order aCGH or Xp21 MLPA."
                ),
            },
        ],
        "clinical_pearls": [
            "CYP21A2: 17-OHP is PATHOGNOMONIC — all other CAH forms have normal or low 17-OHP.",
            "CYP11B1: HYPERTENSION + virilisation + NO salt-wasting = 11β-hydroxylase deficiency.",
            "CYP11B2: salt-wasting + NO virilisation + 17-OHP normal = aldosterone synthase deficiency.",
            "CYP17A1: 17-OHP LOW (not elevated) + HTN + no puberty = CYP17A1 — missed by NBS (17-OHP screening).",
            "STAR: ALL steroids absent simultaneously (cortisol + aldosterone + sex steroids = zero) + lipid adrenals on MRI.",
            "NR0B1: AI in infancy → HH at puberty — both in same patient defines NR0B1 (DAX1).",
            "MC2R: tall + hyperpigmented + hypoglycaemic child with NO salt-wasting + NORMAL renin.",
            "AAAS: DRY EYES FROM BIRTH — Schirmer test in every unexplained adrenal crisis in a child.",
            "STRESS STEROIDS: sick-day rule applies to ALL glucocorticoid-deficient forms (not CYP11B2).",
            "MLPA for CYP21A2: gene conversion from pseudogene missed by sequencing in >75% of alleles.",
        ],
    }


def get_breakdown():
    out = {}
    for gd in ADRENAL_GENES:
        pts = _ALL_COHORTS[gd["gene"]]
        out[gd["gene"]] = {
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "gene_class": gd["gene_class"],
            "phenotype": gd["phenotype"],
            "hallmark": gd["hallmark"],
            "treatment_alert": gd["treatment_alert"],
            "key_ddx": gd["key_ddx"],
            "mineralocorticoid_status": gd["mineralocorticoid_status"],
            "glucocorticoid_status": gd["glucocorticoid_status"],
            "androgen_status": gd["androgen_status"],
            "cohort_n": len(pts),
            "stats": {
                "salt_wasting_pct": _pct(pts, "salt_wasting"),
                "hypertension_pct": _pct(pts, "hypertension"),
                "virilisation_pct": _pct(pts, "virilisation_46xx"),
                "hyperpigmentation_pct": _pct(pts, "hyperpigmentation"),
                "hypoglycaemia_pct": _pct(pts, "hypoglycaemia"),
                "adrenal_crisis_pct": _pct(pts, "adrenal_crisis_at_presentation"),
                "alacrima_pct": _pct(pts, "alacrima"),
                "achalasia_pct": _pct(pts, "achalasia"),
                "hh_pct": _pct(pts, "hypogonadotropic_hh"),
                "lipid_adrenal_pct": _pct(pts, "lipid_adrenal_accumulation"),
                "on_fludrocortisone_pct": _pct(pts, "on_fludrocortisone"),
                "on_hydrocortisone_pct": _pct(pts, "on_hydrocortisone"),
                "severity_severe_pct": _pct(pts, "severity") if False else round(100 * sum(1 for p in pts if p["severity"] == "severe") / 40),
                "severity_moderate_pct": round(100 * sum(1 for p in pts if p["severity"] == "moderate") / 40),
            },
        }
    return out


def get_definitions():
    return {
        "terms": [
            {
                "term": "21-Hydroxylase Deficiency (CYP21A2 CAH) — 17-OHP PATHOGNOMONIC",
                "definition": (
                    "Most common cause of CAH (90–95%). CYP21A2 catalyses conversion of 17-OHP → 11-deoxycortisol. "
                    "LOF → 17-hydroxyprogesterone (17-OHP) accumulates. 17-OHP is the diagnostic biomarker and "
                    "is MARKEDLY ELEVATED (>1500 nmol/L post-Synacthen in classic CAH). NBS uses whole-blood "
                    "17-OHP (day 3 of life). MLPA MANDATORY alongside sequencing — pseudogene (CYP21A1P) gene "
                    "conversions account for >75% of alleles and are missed by sequencing alone. "
                    "Forms: Salt-wasting (SW, ~75%), Simple-virilizing (SV, ~25%), Non-classic (NC, most common AR disorder). "
                    "Treatment: hydrocortisone + fludrocortisone (SW/SV) + stress dose sick-day rule."
                ),
            },
            {
                "term": "11β-Hydroxylase Deficiency (CYP11B1) — Hypertension-CAH",
                "definition": (
                    "2nd most common CAH (5–8%). CYP11B1 catalyses 11-deoxycortisol → cortisol. "
                    "LOF → 11-deoxycortisol and 11-deoxycorticosterone (DOC) accumulate. DOC has mineralocorticoid "
                    "activity → HYPERTENSION + HYPOKALAEMIA + SUPPRESSED RENIN. Aldosterone itself is LOW/suppressed "
                    "(renin suppressed by DOC). KEY DDx from CYP21A2: no salt-wasting (DOC provides mineralocorticoid); "
                    "hypertension instead. Treatment: hydrocortisone (ACTH suppression → lowers DOC → normalises BP). "
                    "FLUDROCORTISONE CONTRAINDICATED (mineralocorticoid excess already present). "
                    "Moroccan Jewish founder: p.Arg448His."
                ),
            },
            {
                "term": "Aldosterone Synthase Deficiency (CYP11B2 / CMO I/II) — Isolated Mineralocorticoid Deficiency",
                "definition": (
                    "CYP11B2 (aldosterone synthase) expressed exclusively in zona glomerulosa catalyses "
                    "corticosterone → aldosterone. LOF → isolated mineralocorticoid deficiency (aldosterone absent/low). "
                    "Renin markedly elevated. Salt-wasting + hyperkalaemia. CORTISOL NORMAL (CYP11B1 intact). "
                    "NO virilisation (androgen pathway intact). KEY DDx from CYP21A2: 17-OHP NORMAL; no virilisation. "
                    "Treatment: fludrocortisone + salt (infancy). HYDROCORTISONE NOT NEEDED. "
                    "SW improves with age in some patients (reduce fludrocortisone dose in adolescence/adulthood). "
                    "Founder mutations: Iranian Jewish (c.788A→G p.Gln263Arg)."
                ),
            },
            {
                "term": "17α-Hydroxylase/17,20-Lyase Deficiency (CYP17A1) — HTN + Sexual Infantilism",
                "definition": (
                    "Rare CAH. CYP17A1 (bifunctional enzyme): 17α-hydroxylase (cortisol pathway) + 17,20-lyase "
                    "(sex steroid pathway). LOF → cortisol deficient + sex steroids absent + DOC accumulates. "
                    "TRIAD: HYPERTENSION + SEXUAL INFANTILISM/AMBIGUITY + HYPOKALAEMIA. "
                    "17-OHP IS LOW (paradoxical — not elevated as in CYP21A2). NBS misses this disorder. "
                    "46,XX: presents at puberty with primary amenorrhoea + no secondary sexual characteristics + tall stature. "
                    "46,XY DSD: phenotypically female at birth (no fetal testosterone). "
                    "Treatment: hydrocortisone (suppresses DOC → treats HTN) + sex steroid replacement at puberty. "
                    "FLUDROCORTISONE CONTRAINDICATED (mineralocorticoid excess). "
                    "Brazilian founder: p.Trp406Arg."
                ),
            },
            {
                "term": "Congenital Lipoid Adrenal Hyperplasia (STAR) — Most Severe; All Steroids Absent",
                "definition": (
                    "STAR protein facilitates cholesterol transport across the outer mitochondrial membrane to CYP11A1. "
                    "LOF → ALL steroidogenesis absent (cortisol + aldosterone + sex steroids = zero). "
                    "MOST SEVERE form of adrenal insufficiency. Presents as neonatal crisis (day 1–4). "
                    "PATHOGNOMONIC: lipid droplet accumulation in adrenal cortex visible on MRI (enlarged adrenals with lipid signal). "
                    "46,XY phenotypically female at birth (no fetal testosterone → female differentiation default). "
                    "Two-hit hypothesis: StAR LOF → initial steroid failure; accumulated lipid → secondary gonadal destruction. "
                    "46,XX ovaries partially spared initially (ovary relatively quiescent in fetal life). "
                    "Treatment: full adrenal replacement (HC + fludrocortisone) + sex steroids at puberty. "
                    "AVOID lipid-lowering agents. East Asian (Korean/Japanese) founder: p.Gln258Ter."
                ),
            },
            {
                "term": "X-linked Adrenal Hypoplasia Congenita (NR0B1/DAX1) — AI + Hypogonadotropic HH",
                "definition": (
                    "NR0B1 (DAX1) is an atypical nuclear receptor essential for adrenal and hypothalamic-pituitary-gonadal "
                    "development. X-linked — males severely affected. "
                    "COMBINATION DEFINING FEATURE: primary adrenal failure (infancy/childhood) + hypogonadotropic "
                    "hypogonadism (adolescence) in the SAME male patient. "
                    "Adrenals are SMALL/HYPOPLASTIC on imaging (DDx from STAR: enlarged lipid-filled adrenals). "
                    "Contiguous Xp21 deletion: check CK (DMD) + plasma glycerol (glycerol kinase deficiency) + "
                    "order aCGH/MLPA. Treatment: HC + fludrocortisone + testosterone ± gonadotropins for fertility. "
                    "Pulsatile GnRH pump or gonadotropin therapy pre-puberty optimises spermatogenesis potential."
                ),
            },
            {
                "term": "Familial Glucocorticoid Deficiency Type 1 (MC2R/FGD1) — Isolated Cortisol Deficiency + Tall Stature",
                "definition": (
                    "MC2R is the ACTH receptor on zona fasciculata. LOF → adrenal cortex cannot respond to ACTH → "
                    "isolated glucocorticoid deficiency. MINERALOCORTICOIDS NORMAL (zona glomerulosa responds to "
                    "angiotensin II, not ACTH). RENIN NORMAL — key DDx from all mineralocorticoid-deficient forms. "
                    "TRIAD: hyperpigmentation + tall stature + hypoglycaemia with NO salt-wasting. "
                    "ACTH massively elevated → drives hyperpigmentation (via MC1R) and tall stature. "
                    "Treatment: hydrocortisone ONLY. FLUDROCORTISONE NOT NEEDED (and is an error). "
                    "No emergency salt cover needed (only cortisol stress doses). "
                    "MRAP mutations → FGD2 (identical phenotype): test MRAP if MC2R sequencing negative."
                ),
            },
            {
                "term": "Triple-A / Allgrove Syndrome (AAAS) — Alacrima + Achalasia + ACTH-Resistant AI",
                "definition": (
                    "Aladin (AAAS protein) is a nuclear pore complex component. LOF → impaired nuclear import of "
                    "oxidative stress defence proteins → oxidative damage in adrenal cortex, oesophageal smooth muscle, "
                    "lacrimal glands, and peripheral/autonomic nerves. "
                    "TRIAD: ALACRIMA (from birth — earliest and most consistent feature) + ACHALASIA (childhood/adolescence) "
                    "+ ACTH-resistant adrenal insufficiency. ALACRIMA APPEARS YEARS BEFORE AI/ACHALASIA. "
                    "Schirmer test (0–3 mm): PATHOGNOMONIC if dry from infancy. "
                    "Progressive autonomic + sensorimotor neuropathy develops with time. "
                    "Mineralocorticoids initially preserved (monitor renin annually — can deplete). "
                    "Treatment: HC + eye lubricants hourly + pneumatic dilation or Heller myotomy for achalasia. "
                    "No disease-modifying neurological treatment."
                ),
            },
            {
                "term": "Synacthen (ACTH Stimulation) Test in Adrenal Disorders",
                "definition": (
                    "Short Synacthen test (SST): 250 μg tetracosactide (ACTH1-24) IM/IV; cortisol at 0 and 60 min. "
                    "Normal response: cortisol ≥ 500 nmol/L at 60 min. Subnormal = primary or secondary AI. "
                    "In CYP21A2: 17-OHP stimulated >1500 nmol/L at 60 min (diagnostic for classic CAH). "
                    "In MC2R (FGD1): cortisol ABSENT despite high baseline ACTH — ACTH receptor dysfunction. "
                    "In AAAS: cortisol subnormal (ACTH-resistant pattern). "
                    "In CYP11B2: cortisol NORMAL (glucocorticoid axis intact). "
                    "Long Synacthen test (1 mg IM, 8-hour cortisol sampling): used for definitive primary AI workup "
                    "when short SST equivocal."
                ),
            },
            {
                "term": "Stress Dose Steroids — Sick-Day Rule for Adrenal Insufficiency",
                "definition": (
                    "ALL patients with glucocorticoid deficiency (CYP21A2, CYP11B1, CYP17A1, STAR, NR0B1, MC2R, AAAS) "
                    "MUST follow the sick-day rule: "
                    "MILD ILLNESS (fever, vomiting × 1): double or triple oral HC dose; "
                    "VOMITING (unable to take oral): IM hydrocortisone 50–100 mg/m² IMMEDIATELY → hospital; "
                    "SURGERY/MAJOR STRESS: IV HC 100 mg/m²/day perioperatively. "
                    "Every patient carries: (1) emergency steroid card; (2) IM hydrocortisone kit at home/school. "
                    "EXCEPTION: CYP11B2 (isolated aldosterone deficiency) — cortisol axis intact, no stress HC needed. "
                    "Signs of adrenal crisis: vomiting, abdominal pain, hypotension, confusion, electrolyte disturbance."
                ),
            },
            {
                "term": "MLPA for CYP21A2 — Pseudogene Gene Conversions",
                "definition": (
                    "CYP21A2 gene is located within the RP-C4-CYP21-TNX (RCCX) module on chromosome 6p21.3, "
                    "adjacent to the highly homologous pseudogene CYP21A1P (97% nucleotide identity in exons). "
                    "Unequal crossover and gene conversion events between CYP21A2 and CYP21A1P during meiosis "
                    "are the most common mechanism of CYP21A2 inactivation (>75% of pathogenic alleles). "
                    "Standard sequencing CANNOT reliably distinguish the active gene from the pseudogene. "
                    "MLPA (Multiplex Ligation-dependent Probe Amplification) is MANDATORY alongside sequencing to: "
                    "(1) detect large deletions; (2) detect gene conversions; (3) quantify copy number. "
                    "A 'normal' CYP21A2 sequence result WITHOUT MLPA is NOT sufficient to rule out CAH."
                ),
            },
            {
                "term": "46,XY Disorders of Sexual Development (DSD) in Adrenal Disorders",
                "definition": (
                    "Three adrenal enzyme defects cause 46,XY DSD (phenotypically female males): "
                    "CYP17A1: 17,20-lyase block → no testosterone in fetal life → female external genitalia; "
                    "  testes present (inguinal/intra-abdominal); gonadoblastoma risk (~10–30%); "
                    "STAR (CLAH): all steroids absent → no testosterone → female external genitalia; "
                    "  massive lipid accumulation destroys testes; "
                    "NR0B1 (AHC): hypoplastic gonads but external genitalia may be male (testosterone sometimes produced early). "
                    "All require multidisciplinary DSD team (endocrinology + genetics + urology + psychology + ethics). "
                    "Gender identity counselling essential. Gonadectomy decision: timing context-dependent "
                    "(intra-abdominal testes → gonadoblastoma risk warrants discussion)."
                ),
            },
            {
                "term": "Hypogonadotropic Hypogonadism (HH) in Adrenal Disorders",
                "definition": (
                    "NR0B1 (DAX1) is the primary adrenal gene causing HH. DAX1 is required for normal "
                    "GnRH neuron development/function in hypothalamus AND for gonadotrope development in pituitary. "
                    "LOF → impaired GnRH secretion → low LH/FSH → low testosterone/oestrogen in adolescence. "
                    "Distinguishes from Kallmann syndrome (ANOS1/KAL1, FGFR1): Kallmann has anosmia; NR0B1 does not. "
                    "Fertility treatment: gonadotropin therapy (hCG + FSH) can stimulate spermatogenesis; "
                    "pulsatile GnRH pump (if hypothalamic). Start EARLY (pre-puberty) for best sperm outcome. "
                    "Testosterone alone is NOT sufficient for fertility — it suppresses gonadotropin drive further."
                ),
            },
        ]
    }
