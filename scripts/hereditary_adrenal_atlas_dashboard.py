#!/usr/bin/env python3
"""Hereditary-Adrenal-Disorders-Atlas — Complete 8-Gene Hereditary Adrenal Atlas
CYP21A2 (21-hydroxylase; 495 aa; 6p21.33; AR;
         Congenital Adrenal Hyperplasia 21-OHD — most common CAH 90-95%;
         Salt-Wasting / Simple Virilizing / Non-Classic subtypes;
         17-OHP newborn screen PATHOGNOMONIC; Hydrocortisone + Fludrocortisone; seed SEED_BASE+0) ·
CYP11B1 (11β-hydroxylase; 503 aa; 8q24.3; AR;
         CAH-11βOHD — 5-8% of CAH; HYPERTENSION unique among CAH;
         ↑ DOC → HTN + virilization; plasma renin LOW; no fludrocortisone needed; seed SEED_BASE+1) ·
CYP17A1 (17α-hydroxylase / 17,20-lyase; 508 aa; 10q24.32; AR;
         Combined 17α-OHD — HYPERTENSION + PRIMARY AMENORRHOEA + NO virilization;
         46XY female phenotype; gonadectomy mandatory; HRT both sexes; seed SEED_BASE+2) ·
STAR    (steroidogenic acute regulatory protein; 285 aa; 8p11.23; AR;
         Lipoid CAH — most severe; ALL steroidogenesis abolished; 46XY female phenotype;
         bilateral large lipid-laden adrenals on imaging; PATHOGNOMONIC; seed SEED_BASE+3) ·
NR0B1   (DAX1; 470 aa; Xp21.2; XLR;
         Adrenal Hypoplasia Congenita + Hypogonadotropic Hypogonadism;
         adrenal SMALL (not large); X-linked males; contiguous gene deletion DMD+AHC; seed SEED_BASE+4) ·
MC2R    (ACTH receptor; 297 aa; 18p11.21; AR;
         Familial Glucocorticoid Deficiency type 1 — isolated glucocorticoid deficiency;
         aldosterone NORMAL (no salt-wasting); ACTH extremely HIGH → hyperpigmentation; seed SEED_BASE+5) ·
AAAS    (ALADIN; 546 aa; 12q13.13; AR;
         Triple-A / Allgrove Syndrome — Alacrima + Achalasia + ACTH-resistant adrenal insufficiency;
         alacrima IS THE FIRST SIGN from birth; Schirmer test 0 mm PATHOGNOMONIC; seed SEED_BASE+6) ·
ABCD1   (ALDP peroxisomal transporter; 745 aa; Xq28; XLR;
         X-linked Adrenoleukodystrophy — VLCFA accumulation; CCALD / AMN / Addison-only;
         newborn screen C26:0-LPC; CCALD MRI every 6 months age 4-12; Loes <9 → HSCT; seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1766–1773)
"""

import random

SEED_BASE = 1766

ADRENAL_GENES = [
    # ── CYP21A2 — Congenital Adrenal Hyperplasia (21-OHD) ────────────────────
    {
        "gene": "CYP21A2",
        "protein": (
            "CYP21A2 — 6p21.33 AR — 21-Hydroxylase-495aa — "
            "Congenital-Adrenal-Hyperplasia-CAH-21OHD-Most-Common-90-95pct-of-all-CAH — "
            "Three-Subtypes: Salt-Wasting-75pct / Simple-Virilizing-25pct / Non-Classic-1:50-Ashkenazi — "
            "17-OHP-Newborn-Screen-PATHOGNOMONIC — "
            "Hydrocortisone-10-15-mg/m2/day + Fludrocortisone-0.05-0.2mg (SW) — "
            "Stress-Dosing-2x-3x-Sick-Day-Rule-MANDATORY"
        ),
        "alias": (
            "CYP21A2 (cytochrome P450 21A2; steroid 21-hydroxylase); OMIM gene 613815; "
            "Congenital Adrenal Hyperplasia due to 21-Hydroxylase Deficiency (CAH-21OHD) OMIM 201910. "
            "6p21.33; 495 aa; ~55 kDa; autosomal recessive; most common inborn error of steroid synthesis. "
            "FUNCTION: CYP21A2 is a microsomal cytochrome P450 enzyme with two critical substrates: "
            "(1) progesterone → 11-deoxycorticosterone (DOC) — the mineralocorticoid biosynthesis arm; "
            "(2) 17-hydroxyprogesterone (17-OHP) → 11-deoxycortisol — the glucocorticoid biosynthesis arm. "
            "Loss of enzyme activity → failure of both cortisol AND aldosterone synthesis → "
            "↑ ACTH (loss of cortisol feedback) → adrenal androgen excess (DHEA, androstenedione) → "
            "virilization in 46XX females and precocious pseudopuberty in males. "
            "THREE CLINICAL SUBTYPES: "
            "SALT-WASTING (SW, ~75% of classic): <1% residual enzyme activity; "
            "no aldosterone → sodium wasting → hyponatraemia + hyperkalaemia + hypovolaemia; "
            "life-threatening adrenal crisis at ~2 weeks of life (before NBS return); "
            "also complete failure of cortisol → hypoglycaemia; "
            "46XX neonates: ambiguous genitalia (Prader scale I–V) at birth; "
            "46XY neonates: appear normal at birth — SW crisis is presentation. "
            "SIMPLE VIRILIZING (SV, ~25% of classic): 1-2% residual enzyme; "
            "sufficient mineralocorticoid synthesis to prevent salt-wasting; "
            "enough cortisol for most physiological demands; "
            "46XX: virilized genitalia at birth; "
            "46XY: precocious pseudopuberty (pubic hair, penile enlargement) by 2-4y; "
            "advanced bone age; without treatment → premature epiphyseal closure → short adult stature. "
            "NON-CLASSIC (NCAH): mild enzyme deficiency; "
            "most common autosomal recessive disorder in humans (1:50-1:100 Ashkenazi Jews globally); "
            "late-onset: hirsutism, acne, oligomenorrhoea, infertility in females; "
            "males: often asymptomatic or oligospermia; "
            "17-OHP stimulation test (Synacthen/ACTH): 17-OHP >300 nmol/L (at 60 min) = diagnostic. "
            "DIAGNOSTIC BIOCHEMISTRY: "
            "17-OHP (17-hydroxyprogesterone): the DIAGNOSTIC marker; "
            "newborn screen (heel prick 24-72h): 17-OHP elevation; "
            "cut-off varies by gestational age (preterm falsely elevated); "
            "SW: 17-OHP often >300 nmol/L (normal <3 nmol/L at term); "
            "SV: 17-OHP elevated but lower than SW; "
            "NCAH: basal 17-OHP 3-30 nmol/L; Synacthen-stimulated >300 nmol/L; "
            "Electrolytes: SW → ↓Na, ↑K, ↑renin, ↓aldosterone; "
            "Androgens: DHEA-S, androstenedione, testosterone elevated; "
            "Bone age X-ray (wrist): advanced in classic forms. "
            "TREATMENT — GLUCOCORTICOID REPLACEMENT: "
            "Hydrocortisone (HC) 10-15 mg/m²/day divided 3 doses (preferred in children — "
            "shorter acting, less growth suppression than prednisolone or dexamethasone); "
            "Monitor: 17-OHP, androstenedione, growth velocity, bone age (annual), DEXA (every 3-5y adult); "
            "adults may use prednisolone or dexamethasone for convenience; "
            "MINERALOCORTICOID REPLACEMENT (SW): "
            "Fludrocortisone 0.05-0.2 mg/day orally; "
            "infants: sodium chloride supplement 1-2 g/day (infant formula low in Na); "
            "monitor plasma renin activity (target upper normal range) + blood pressure; "
            "STRESS DOSING (2x-3x SICK DAY RULE — ALL patients, MANDATORY): "
            "Minor illness (fever >38°C, vomiting without diarrhoea): double HC dose; "
            "Major illness or vomiting/diarrhoea (cannot absorb oral): "
            "IM/IV hydrocortisone 25 mg/m² bolus → 25-50 mg/m²/day infusion; "
            "CARRY EMERGENCY HYDROCORTISONE KIT (Solu-Cortef IM 100 mg): every patient + carer trained; "
            "PREGNANCY AND PRENATAL TREATMENT: "
            "Prenatal diagnosis: CVS/amniocentesis → karyotype + CYP21A2 genotype; "
            "Prenatal dexamethasone (dex) for affected 46XX fetus: "
            "given before diagnosis (before CVS) to all at-risk pregnancies to reduce virilization; "
            "CONTROVERSIAL: dex given to 7/8 fetuses that are not affected females; "
            "cognitive/metabolic long-term effects debated — not universally recommended; "
            "SURGICAL: "
            "Feminising genitoplasty for severe (Prader IV-V) virilization in 46XX: "
            "timing, extent, and indication are debated (patient-centred approach); "
            "FERTILITY: "
            "46XX SW: fertility possible with good hormonal control; "
            "adrenal rest tumours in testes (TART — ectopic adrenal tissue in testis) → "
            "evaluate with testicular ultrasound in males; "
            "LONG-TERM COMPLICATIONS: "
            "Adrenal crisis (lifetime risk); "
            "Adrenal rest tumours (TART in males; ovarian adrenal rest in females); "
            "Metabolic syndrome (cortisol over-replacement); "
            "Osteoporosis (over-replacement); "
            "Short stature (if undertreated → advanced bone age, OR overtreated → GC-induced growth failure)."
        ),
        "locus": "6p21.33",
        "aa": 495,
        "kDa": 55,
        "omim_gene": "613815",
        "omim_disease": "201910",
        "inheritance": "AR; most common CAH (90-95% of all cases); biallelic CYP21A2 loss-of-function",
        "gene_class": "Adrenal Steroidogenesis — Microsomal CYP450 — 21-Hydroxylase — Cortisol + Aldosterone Pathway",
        "key_alerts": [
            "CYP21A2-17OHP-NEWBORN-SCREEN-PATHOGNOMONIC: 17-OHP at 72h newborn heel-prick >30 nmol/L in salt-wasting CAH is the diagnostic hallmark; screen result must be followed immediately with electrolytes (Na/K) and blood gas — SW crisis peaks at 7-14 days of life; a male neonate with hyponatraemia + hyperkalaemia + adrenal crisis IS CYP21A2 SW until proven otherwise",
            "CYP21A2-STRESS-DOSING-MANDATORY-SICK-DAY-RULE: ALL CYP21A2 patients MUST carry an emergency hydrocortisone kit (Solu-Cortef 100 mg IM) and their carers trained to administer; double oral HC for fever; IM/IV HC 25-50 mg/m2/day for vomiting/diarrhoea — failure = adrenal crisis = potentially fatal; never withhold HC because 'too much cortisol'; parenteral HC BEFORE hospital assessment if in doubt",
            "CYP21A2-NCAH-MOST-COMMON-AR-DISORDER: Non-Classic CAH (mild CYP21A2) affects 1:50-1:100 Ashkenazi Jews and 1:1000 general population — MOST COMMON autosomal recessive disorder in humans; presents in adult females as hirsutism, oligomenorrhoea, and infertility; basal 17-OHP 3-30 nmol/L; Synacthen stimulation confirms; low-dose HC or prednisolone at night (suppress ACTH rhythm) treats symptoms",
            "CYP21A2-FLUDROCORTISONE-SW-MINERALOCORTICOID: Salt-Wasting CAH requires fludrocortisone 0.05-0.2 mg/day PLUS sodium chloride supplementation in infancy; monitor plasma renin activity (aim upper normal); do NOT omit fludrocortisone even if blood pressure normal — renin activity is the monitoring target; excess fludrocortisone → hypertension + hypokalemia",
        ],
        "etiologies": {
            "Salt_Wasting_CAH_21OHD": {"pct": 75, "phenotype": "no aldosterone + no cortisol → adrenal crisis week 2, hyponatraemia, hyperkalaemia, 46XX virilization"},
            "Simple_Virilizing_CAH_21OHD": {"pct": 25, "phenotype": "sufficient mineralocorticoid; virilization only; precocious pseudopuberty in boys"},
            "Non_Classic_CAH_21OHD": {"pct": 100, "phenotype": "mild; adult females — hirsutism, oligomenorrhoea; 1:50-1:100 Ashkenazi; often missed"},
            "Adrenal_Rest_Tumours_TART": {"pct": 40, "phenotype": "ectopic adrenal tissue in testes/ovaries; benign; intensive HC suppresses; ultrasound surveillance"},
        },
        "stats": {
            "mean_onset_age_y": 0.1,
            "mean_dx_delay_months": 0.5,
            "salt_wasting_crisis_rate_pct": 75,
            "ncah_female_infertility_pct": 30,
        },
        "dx_delay_distribution": [
            {"bucket": "<1mo", "pct": 78, "desc": "newborn screen detection"},
            {"bucket": "1-12mo", "pct": 12, "desc": "clinical crisis presentation"},
            {"bucket": ">12mo", "pct": 10, "desc": "NCAH / SV missed on NBS"},
        ],
        "patients": [],
    },
    # ── CYP11B1 — CAH 11β-Hydroxylase Deficiency ─────────────────────────────
    {
        "gene": "CYP11B1",
        "protein": (
            "CYP11B1 — 8q24.3 AR — 11β-Hydroxylase-503aa — "
            "CAH-11βOHD-5-8pct-of-all-CAH — "
            "HYPERTENSION-UNIQUE-AMONG-CAH — "
            "DOC-Excess-Mineralocorticoid-Precursor — "
            "Plasma-Renin-LOW-NOT-HIGH — "
            "NO-Fludrocortisone-Needed — "
            "11-Deoxycortisol-Plus-17OHP-Both-Elevated"
        ),
        "alias": (
            "CYP11B1 (cytochrome P450 11B1; steroid 11β-hydroxylase); OMIM gene 610613; "
            "CAH due to 11β-Hydroxylase Deficiency (CAH-11βOHD) OMIM 202010. "
            "8q24.3; 503 aa; ~57 kDa; autosomal recessive; "
            "second most common form of CAH (5-8% of cases). "
            "FUNCTION: CYP11B1 is a mitochondrial cytochrome P450 enzyme catalysing: "
            "(1) 11-deoxycortisol → cortisol (glucocorticoid arm); "
            "(2) 11-deoxycorticosterone (DOC) → corticosterone (mineralocorticoid arm). "
            "CYP11B1 loss → failure of cortisol synthesis → ↑ ACTH → "
            "accumulation of 11-deoxycortisol and DOC (the immediate precursors). "
            "CRITICAL DISTINCTION: DOC is a potent mineralocorticoid — "
            "its accumulation MIMICS aldosterone action → SODIUM RETENTION, POTASSIUM LOSS, "
            "and HYPERTENSION. This is the opposite to SW-CAH where there is aldosterone deficiency. "
            "BIOCHEMICAL PROFILE: "
            "↑ 11-deoxycortisol (11-S, compound S) — the key biomarker; "
            "↑ DOC (11-deoxycorticosterone); "
            "↑ 17-OHP (because CYP11B1 is downstream of CYP21A2 — 17-OHP to 11-deoxycortisol is unblocked); "
            "cortisol LOW or undetectable; "
            "aldosterone LOW (DOC suppresses renin-angiotensin → aldosterone not stimulated); "
            "plasma renin LOW (suppressed by DOC mineralocorticoid effect) — "
            "KEY DDx from SW-CAH where renin is HIGH; "
            "androgens: ↑ DHEA, androstenedione, testosterone (same as CYP21A2 — virilization occurs). "
            "CLINICAL FEATURES: "
            "HYPERTENSION: the hallmark of CYP11B1 deficiency; "
            "present in 2/3 of patients; severity correlates with DOC levels; "
            "can cause hypertensive encephalopathy and CVD if undiagnosed; "
            "VIRILIZATION: 46XX neonates have ambiguous genitalia (same as CYP21A2-SW, SV); "
            "46XY: precocious pseudopuberty; "
            "CORTISOL DEFICIENCY: mild-moderate; full SW crisis is RARE (because DOC supports some Na retention); "
            "hypoglycaemia less prominent than CYP21A2-SW. "
            "KEY DISTINGUISHING FEATURES FROM CYP21A2: "
            "1. HYPERTENSION (CYP11B1) vs normal or low BP (CYP21A2-SW); "
            "2. Plasma renin LOW (CYP11B1) vs HIGH (CYP21A2-SW); "
            "3. DOC elevated (CYP11B1) vs DOC LOW (CYP21A2-SW); "
            "4. 11-deoxycortisol elevated (CYP11B1) — this is the specific test; "
            "5. Both share elevated 17-OHP — 17-OHP alone cannot distinguish the two. "
            "TREATMENT: "
            "Hydrocortisone (HC) 10-15 mg/m²/day in 3 doses: "
            "suppresses ACTH → ↓ DOC → blood pressure normalises in most; "
            "NO fludrocortisone needed (DOC provides mineralocorticoid activity — adding fludrocortisone → severe HTN); "
            "Antihypertensives (calcium channel blockers) may be needed transiently before HC control achieved; "
            "Monitor: 11-deoxycortisol, 17-OHP, plasma renin, blood pressure; "
            "stress dosing as for CYP21A2 (cortisol deficiency still present); "
            "surgery for severe virilization in 46XX (same considerations as CYP21A2); "
            "MUTATION SPECTRUM: "
            "CYP11B1 is 93% homologous with CYP11B2 (aldosterone synthase) on 8q24.3 — "
            "chimeric genes can arise from unequal crossing-over (also cause glucocorticoid-remediable aldosteronism); "
            "common mutations in Middle Eastern, Moroccan Jewish, and Arab populations."
        ),
        "locus": "8q24.3",
        "aa": 503,
        "kDa": 57,
        "omim_gene": "610613",
        "omim_disease": "202010",
        "inheritance": "AR; mitochondrial CYP450; 5-8% of all CAH; common in Middle Eastern/Moroccan Jewish",
        "gene_class": "Adrenal Steroidogenesis — Mitochondrial CYP450 — 11β-Hydroxylase — DOC/Cortisol Pathway",
        "key_alerts": [
            "CYP11B1-HYPERTENSION-IN-CAH-DIAGNOSTIC: Hypertension in any patient with CAH-like features (virilization, elevated 17-OHP) is CYP11B1 deficiency until proven otherwise — 2/3 of CYP11B1 patients develop HTN; DOC accumulation acts like mineralocorticoid; plasma renin is SUPPRESSED (low) — opposite to Salt-Wasting CYP21A2 where renin is HIGH",
            "CYP11B1-NO-FLUDROCORTISONE-CONTRAINDICATED: Do NOT give fludrocortisone to CYP11B1 CAH — DOC excess already causes mineralocorticoid hypertension; adding fludrocortisone causes severe refractory hypertension and hypokalaemia; hydrocortisone alone suppresses ACTH → ↓ DOC → BP normalises",
            "CYP11B1-11-DEOXYCORTISOL-BIOMARKER: Plasma 11-deoxycortisol (compound S) is the specific diagnostic marker for CYP11B1 deficiency — it is elevated (unlike CYP21A2 where deoxycortisol is low); 17-OHP is also elevated in both — 17-OHP alone cannot distinguish CYP11B1 from CYP21A2",
            "CYP11B1-VIRILIZATION-SAME-AS-21OHD: Androgen excess and 46XX virilization are present in CYP11B1 — same ACTH-driven adrenal androgen accumulation as CYP21A2; genital surgery considerations identical; however adrenal crisis is less acute at birth (DOC provides some mineralocorticoid effect)",
        ],
        "etiologies": {
            "Classic_CYP11B1_HTN_Virilization": {"pct": 67, "phenotype": "hypertension + virilization + elevated DOC + low renin — the full phenotype"},
            "CYP11B1_Virilization_Only": {"pct": 33, "phenotype": "virilization without clinical hypertension; DOC elevated biochemically"},
            "CYP11B1_Hypertensive_Encephalopathy": {"pct": 8, "phenotype": "severe uncontrolled HTN; diagnostic emergency; HC → rapid BP normalisation"},
        },
        "stats": {
            "mean_onset_age_y": 0.5,
            "mean_dx_delay_months": 8.4,
            "hypertension_rate_pct": 67,
            "virilization_46xx_pct": 100,
        },
        "dx_delay_distribution": [
            {"bucket": "<3mo", "pct": 45, "desc": "virilization in 46XX triggers early evaluation"},
            {"bucket": "3-24mo", "pct": 35, "desc": "hypertension detected on routine check"},
            {"bucket": ">24mo", "pct": 20, "desc": "NCAH-like mild forms; late HTN presentation"},
        ],
        "patients": [],
    },
    # ── CYP17A1 — Combined 17α-Hydroxylase/17,20-Lyase Deficiency ────────────
    {
        "gene": "CYP17A1",
        "protein": (
            "CYP17A1 — 10q24.32 AR — 17α-Hydroxylase-17,20-Lyase-508aa — "
            "Combined-17α-OHD — "
            "HYPERTENSION-Plus-PRIMARY-AMENORRHOEA-Plus-ABSENT-SECONDARY-SEX-CHARS — "
            "NO-Virilization-UNLIKE-CYP11B1 — "
            "46XY-Female-External-Genitalia-Gonadectomy-MANDATORY — "
            "HRT-Both-Sexes-Required"
        ),
        "alias": (
            "CYP17A1 (cytochrome P450 17A1; 17α-hydroxylase/17,20-lyase); OMIM gene 609300; "
            "Combined 17α-Hydroxylase/17,20-Lyase Deficiency OMIM 202110. "
            "10q24.32; 508 aa; ~57 kDa; autosomal recessive. "
            "ENZYME DUAL FUNCTION: "
            "CYP17A1 is a single protein with two enzymatic activities: "
            "(1) 17α-HYDROXYLASE: "
            "pregnenolone → 17-OH-pregnenolone (in Δ5 pathway); "
            "progesterone → 17-OH-progesterone (17-OHP) (in Δ4 pathway); "
            "required for cortisol and sex steroid synthesis. "
            "(2) 17,20-LYASE: "
            "17-OH-pregnenolone → DHEA (Δ5 androgen precursor); "
            "17-OH-progesterone → androstenedione (Δ4 androgen); "
            "required for sex steroid synthesis (both male and female). "
            "CYP17A1 COMPLETE LOSS → "
            "NO cortisol synthesis; "
            "NO sex steroids (neither androgens nor oestrogens) in adrenal or gonads; "
            "DOC and corticosterone accumulate (these are made without CYP17A1) → "
            "MINERALOCORTICOID EXCESS → HYPERTENSION + HYPOKALAEMIA. "
            "CLINICAL FEATURES: "
            "HYPERTENSION + HYPOKALAEMIA (from DOC/corticosterone excess): "
            "hallmarks; present from childhood; can be severe; "
            "plasma renin suppressed (same mechanism as CYP11B1); "
            "aldosterone LOW (suppressed by DOC). "
            "SEX DEVELOPMENT — THE CRITICAL DISTINCTION: "
            "46XY patients: NO testosterone production in utero → "
            "complete female external genitalia (anti-Müllerian hormone still produced by Sertoli cells → "
            "no uterus/fallopian tubes, but female external genitalia) → "
            "inguinal testes (undescended) or labial testes; "
            "AT PUBERTY: no virilization, no pubic hair, no axillary hair, no breast development; "
            "→ present as 'primary amenorrhoea in apparent female' (actually 46XY DSD); "
            "GONADECTOMY MANDATORY (gonadal malignancy — gonadoblastoma/dysgerminoma — risk ~15-30% untreated). "
            "46XX patients: normal female external genitalia (oestrogen not required for female differentiation); "
            "NO virilization (unlike CYP21A2, CYP11B1 — no androgen excess in CYP17A1 deficiency); "
            "AT PUBERTY: primary amenorrhoea; absent secondary sex characteristics (no oestrogen); "
            "ovarian function abolished. "
            "DIAGNOSTIC PATTERN: "
            "↓ cortisol; ↑ ACTH; ↑ DOC; ↑ corticosterone; ↑ progesterone; "
            "17-OHP LOW (not elevated — because the 17-hydroxylation is blocked, 17-OHP cannot accumulate); "
            "KEY DDx from CYP21A2: 17-OHP LOW in CYP17A1 vs HIGH in CYP21A2; "
            "sex steroids (testosterone, DHEA-S, oestradiol): ALL very low/undetectable; "
            "renin LOW; aldosterone LOW; karyotype ESSENTIAL (46XY in phenotypic female). "
            "TREATMENT: "
            "Hydrocortisone / dexamethasone: suppresses ACTH → ↓ DOC → BP normalises; "
            "NO fludrocortisone (DOC excess causes mineralocorticoid HTN — spironolactone may be needed); "
            "HRT (oestrogen + progesterone cycling in 46XX; oestrogen alone in gonadectomised 46XY): "
            "essential for secondary sex characteristics, bone health, cardiovascular protection; "
            "46XY: full DSD team — psychological support, gonadectomy planning; "
            "fertility: generally not possible (no sex steroid synthesis); "
            "stress dosing as for other CAH forms."
        ),
        "locus": "10q24.32",
        "aa": 508,
        "kDa": 57,
        "omim_gene": "609300",
        "omim_disease": "202110",
        "inheritance": "AR; dual enzyme; rare; higher prevalence in Dutch, Japanese, Brazilian populations",
        "gene_class": "Adrenal Steroidogenesis — CYP17A1 Dual Enzyme — Cortisol + Sex Steroid Pathway",
        "key_alerts": [
            "CYP17A1-HYPERTENSION-AMENORRHOEA-NO-VIRILIZATION-TRIAD: Hypertension + primary amenorrhoea + complete absence of secondary sex characteristics with NO virilization = CYP17A1 deficiency until proven otherwise; the KEY distinction from CYP11B1 (which causes virilization) — no androgens are produced in CYP17A1 deficiency",
            "CYP17A1-46XY-FEMALE-PHENOTYPE-GONADECTOMY: 46XY patients present with complete female external genitalia; karyotype is MANDATORY in all cases of primary amenorrhoea with absent secondary sex chars; gonadectomy is mandatory (gonadoblastoma risk 15-30%); DSD multidisciplinary team; lifelong oestrogen HRT post-gonadectomy",
            "CYP17A1-17OHP-LOW-NOT-HIGH: 17-OHP is characteristically LOW in CYP17A1 deficiency (cannot be synthesized without 17α-hydroxylase); this is the OPPOSITE of CYP21A2 and CYP11B1 where 17-OHP is high — use 17-OHP levels to distinguish CAH subtypes; progesterone is markedly elevated",
            "CYP17A1-NO-FLUDROCORTISONE-SPIRONOLACTONE: DOC accumulation causes mineralocorticoid excess (same mechanism as CYP11B1); do not add fludrocortisone; hydrocortisone suppresses ACTH → ↓ DOC; if HTN persists, spironolactone (mineralocorticoid antagonist) may be needed; target normal renin activity",
        ],
        "etiologies": {
            "Classic_CYP17A1_46XX": {"pct": 50, "phenotype": "normal female external genitalia; primary amenorrhoea; absent breasts; hypertension"},
            "Classic_CYP17A1_46XY_DSD": {"pct": 50, "phenotype": "female external genitalia; inguinal testes; primary amenorrhoea; gonadectomy mandatory"},
            "Isolated_17_20_Lyase_Deficiency": {"pct": 5, "phenotype": "rare; sex steroid deficiency only; no mineralocorticoid excess; selective 17,20-lyase mutations"},
        },
        "stats": {
            "mean_onset_age_y": 14.5,
            "mean_dx_delay_months": 24.3,
            "hypertension_rate_pct": 85,
            "gonadectomy_46xy_pct": 100,
        },
        "dx_delay_distribution": [
            {"bucket": "<12mo", "pct": 15, "desc": "neonatal hypertension or DSD evaluation"},
            {"bucket": "12-60mo", "pct": 20, "desc": "childhood hypertension investigation"},
            {"bucket": ">60mo", "pct": 65, "desc": "pubertal failure / primary amenorrhoea — typical presentation"},
        ],
        "patients": [],
    },
    # ── STAR — Lipoid Congenital Adrenal Hyperplasia ──────────────────────────
    {
        "gene": "STAR",
        "protein": (
            "STAR — 8p11.23 AR — StAR-Steroidogenic-Acute-Regulatory-Protein-285aa — "
            "Lipoid-CAH-MOST-SEVERE-Form — "
            "ALL-Steroidogenesis-Abolished-Adrenal-AND-Gonads — "
            "46XY-Female-External-Genitalia-PATHOGNOMONIC — "
            "Bilateral-Large-Lipid-Laden-Adrenals-on-Imaging — "
            "Hydrocortisone-PLUS-Fludrocortisone-Lifelong — "
            "46XY-Gonadectomy-MANDATORY"
        ),
        "alias": (
            "STAR (steroidogenic acute regulatory protein); OMIM gene 600617; "
            "Lipoid Congenital Adrenal Hyperplasia (Lipoid CAH) OMIM 201710. "
            "8p11.23; 285 aa; ~37 kDa; autosomal recessive; "
            "the most severe form of congenital adrenal hyperplasia. "
            "FUNCTION: StAR protein is located on the outer mitochondrial membrane and "
            "facilitates the rate-limiting step of ALL steroidogenesis: "
            "transport of cholesterol from the outer to the inner mitochondrial membrane, "
            "where CYP11A1 (cholesterol side-chain cleavage enzyme) converts it to pregnenolone. "
            "Without StAR: NO pregnenolone → NO cortisol, NO aldosterone, NO sex steroids "
            "in the adrenal cortex OR gonads. "
            "THE 'TWO-HIT' MECHANISM: "
            "Initially, the adrenal produces very small amounts of steroid via StAR-independent transfer; "
            "but ACTH-driven stimulation → massive cholesterol accumulation in adrenocortical cells → "
            "'second hit': cholesterol esters and oxysterols destroy remaining adrenocortical cells → "
            "adrenal becomes massively enlarged with lipid-laden appearance ('lipoid' = lipid filled) → "
            "BILATERAL LARGE ADRENALS on ultrasound/CT. "
            "GONADAL CONSEQUENCE — SEX DETERMINATION: "
            "46XY: fetal Leydig cells cannot make testosterone → NO masculinisation of external genitalia → "
            "COMPLETE FEMALE EXTERNAL GENITALIA in 46XY (most severe DSD); "
            "anti-Müllerian hormone still produced by Sertoli cells (different mechanism, StAR-independent) → "
            "no uterus, no fallopian tubes; inguinal testes (undescended); "
            "46XX: PROTECTED — the ovary in fetal life does not require gonadotrophins (FSH) for initial development; "
            "ovarian cells initially spare (minimal cholesterol accumulation in fetal life); "
            "AT PUBERTY: FSH drives cholesterol accumulation in ovarian cells → second-hit destroys follicles; "
            "clinical: SPONTANEOUS THELARCHE in 46XX (initial oestrogen from follicles) → then ovarian failure; "
            "secondary amenorrhoea after initial puberty. "
            "CLINICAL PRESENTATION: "
            "Neonatal adrenal crisis (week 1-2): "
            "hyponatraemia + hyperkalaemia + hypoglycaemia + hypotension → shock → death if untreated; "
            "ALL cortisol, aldosterone, and androgens absent; "
            "46XX neonates: normal female genitalia (may appear completely normal) — "
            "diagnosis may be delayed without NBS; "
            "46XY neonates: female external genitalia → sex assignment challenge. "
            "IMAGING: "
            "Bilateral large lipid-filled adrenal glands on ultrasound/CT — pathognomonic finding; "
            "cholesterol accumulates → adrenal enlargement + bilateral echogenic masses on US. "
            "DIAGNOSIS: "
            "Severe cortisol + aldosterone deficiency; ACTH markedly elevated; "
            "ALL sex steroids undetectable; "
            "ACTH stimulation test: cortisol response absent (most severe form); "
            "STAR gene sequencing: confirms; common mutations: Q258X (pan-Asian); "
            "karyotype essential. "
            "TREATMENT: "
            "Hydrocortisone 10-15 mg/m²/day (3 doses) — lifelong; "
            "Fludrocortisone 0.05-0.15 mg/day — lifelong; "
            "Sodium chloride supplement in infancy; "
            "Stress dosing as for CYP21A2 (full adrenal insufficiency); "
            "46XY: gonadectomy (malignancy risk from intra-abdominal testes; dysgenetic gonads) + oestrogen HRT; "
            "46XX: oestrogen HRT for ovarian failure + fertility referral (premature ovarian insufficiency); "
            "PROGNOSIS: good with early treatment; adrenal crises are the main risk."
        ),
        "locus": "8p11.23",
        "aa": 285,
        "kDa": 37,
        "omim_gene": "600617",
        "omim_disease": "201710",
        "inheritance": "AR; most severe CAH; Q258X mutation common in Japanese/Korean; biallelic STAR loss",
        "gene_class": "Adrenal Steroidogenesis — StAR Mitochondrial Cholesterol Transport — Rate-Limiting Step ALL Steroidogenesis",
        "key_alerts": [
            "STAR-LIPOID-CAH-MOST-SEVERE: Lipoid CAH (STAR) abolishes ALL steroidogenesis — cortisol, aldosterone, AND sex steroids are absent; presents as severe neonatal adrenal crisis in both sexes; bilateral large lipid-laden adrenals on imaging is pathognomonic; requires BOTH hydrocortisone AND fludrocortisone lifelong",
            "STAR-46XY-FEMALE-PHENOTYPE: 46XY individuals with STAR mutations have complete female external genitalia (no fetal testosterone) — the most severe form of 46XY DSD; karyotype is essential; gonadectomy mandatory (intra-abdominal dysgenetic gonads with malignancy risk); oestrogen HRT from puberty",
            "STAR-BILATERAL-LARGE-ADRENALS-IMAGING: Bilateral enlarged, lipid-laden (echogenic on US, low-density on CT) adrenal glands are a pathognomonic imaging finding in Lipoid CAH — distinguish from other adrenal disorders; the 'lipoid' appearance is cholesterol ester accumulation secondary to ACTH-driven stimulation without output",
            "STAR-46XX-SPONTANEOUS-PUBERTY-THEN-FAILURE: 46XX girls with STAR mutations can undergo spontaneous breast development at puberty (initial oestrogen from ovarian follicles before second-hit destruction) — this should NOT be reassuring without diagnosis; ovarian failure follows; HRT is required for fertility preservation referral; POI management mandatory",
        ],
        "etiologies": {
            "Lipoid_CAH_46XY_Female_Phenotype": {"pct": 50, "phenotype": "complete female external genitalia; inguinal testes; adrenal crisis neonate; gonadectomy"},
            "Lipoid_CAH_46XX": {"pct": 50, "phenotype": "normal female genitalia at birth; severe adrenal crisis; spontaneous puberty then ovarian failure"},
            "Partial_STAR_Deficiency": {"pct": 10, "phenotype": "milder; some residual steroidogenesis; later presentation; less severe DSD"},
        },
        "stats": {
            "mean_onset_age_y": 0.05,
            "mean_dx_delay_months": 0.5,
            "adrenal_crisis_neonatal_pct": 95,
            "gonadectomy_46xy_pct": 100,
        },
        "dx_delay_distribution": [
            {"bucket": "<1mo", "pct": 90, "desc": "neonatal adrenal crisis — diagnosis emergency"},
            {"bucket": "1-12mo", "pct": 8, "desc": "partial deficiency or 46XX missed on presentation"},
            {"bucket": ">12mo", "pct": 2, "desc": "very partial deficiency; pubertal onset"},
        ],
        "patients": [],
    },
    # ── NR0B1 (DAX1) — Adrenal Hypoplasia Congenita ──────────────────────────
    {
        "gene": "NR0B1",
        "protein": (
            "NR0B1-DAX1 — Xp21.2 XLR — DAX1-Nuclear-Receptor-470aa — "
            "Adrenal-Hypoplasia-Congenita-PLUS-Hypogonadotropic-Hypogonadism — "
            "X-LINKED-Males-Primarily-Affected — "
            "Adrenal-SMALL-NOT-LARGE — "
            "Contiguous-Gene-Deletion-DMD-Plus-Glycerol-Kinase-Plus-AHC — "
            "Mineralocorticoid-Deficiency-Fludrocortisone-Needed"
        ),
        "alias": (
            "NR0B1 (nuclear receptor subfamily 0, group B, member 1; DAX1 — dosage-sensitive sex reversal, "
            "adrenal hypoplasia critical region, chromosome X, gene 1); OMIM gene 300473; "
            "Adrenal Hypoplasia Congenita (AHC) + Hypogonadotropic Hypogonadism OMIM 300200. "
            "Xp21.2; 470 aa; ~51 kDa; X-linked recessive (XLR); "
            "primarily affects males (females may be carriers with mild features). "
            "PROTEIN FUNCTION: "
            "DAX1 is an atypical nuclear receptor (orphan receptor — no known ligand); "
            "it lacks the conventional DNA-binding domain (zinc finger) of classical nuclear receptors; "
            "instead, it acts as a TRANSCRIPTIONAL REPRESSOR via protein-protein interaction with: "
            "(1) SF1 (steroidogenic factor 1 / NR5A1) — represses SF1 target genes in adrenal cortex; "
            "(2) GnRH receptor signalling pathway — represses gonadotroph differentiation and LH/FSH secretion. "
            "CONSEQUENCE OF LOSS: "
            "Adrenal hypoplasia: adrenal glands fail to develop normally → small, cytomegalic adrenal cortex; "
            "BOTH mineralocorticoid (aldosterone) AND glucocorticoid (cortisol) deficient; "
            "Hypogonadotropic Hypogonadism (HH): "
            "LH and FSH not secreted (hypothalamic + pituitary levels both affected); "
            "males → delayed/absent puberty + azoospermia (irreversible). "
            "CONTIGUOUS GENE SYNDROME — CRITICAL: "
            "NR0B1 is at Xp21.2; adjacent genes include: "
            "GK (glycerol kinase deficiency) — Xp21.3; "
            "DMD (Duchenne muscular dystrophy) — Xp21.2; "
            "Xp21 deletion → AHC + GKD + DMD (contiguous gene syndrome); "
            "ALL males with AHC should have DMD (CK level) and GK screening; "
            "the extent of deletion determines the clinical triad. "
            "CLINICAL PRESENTATION — TWO PATTERNS: "
            "INFANTILE (most common): "
            "salt-wasting adrenal crisis at 1-8 weeks of life (mimics CYP21A2-SW); "
            "hyponatraemia + hyperkalaemia + hypoglycaemia + shock; "
            "ADOLESCENT (delayed puberty pattern): "
            "cryptorchidism; absent or arrested puberty; small testes; azoospermia; "
            "incidentally identified (less severe hypocortisolism). "
            "KEY IMAGING DDx: "
            "NR0B1-AHC: adrenal glands SMALL or absent on imaging; "
            "Lipoid CAH (STAR): adrenal glands LARGE (lipid-filled); "
            "this imaging distinction is clinically important. "
            "DIAGNOSIS: "
            "Low cortisol + low aldosterone + high ACTH + high renin in male neonate; "
            "NR0B1 gene sequencing (deletions, point mutations, frameshifts); "
            "XLR inheritance: maternal carrier females usually unaffected; "
            "GnRH stimulation test: absent or severely blunted LH/FSH response. "
            "TREATMENT: "
            "Hydrocortisone 10-15 mg/m²/day (glucocorticoid replacement); "
            "Fludrocortisone 0.05-0.15 mg/day (mineralocorticoid — BOTH are deficient unlike FGD/MC2R); "
            "Stress dosing mandatory; "
            "Gonadotrophin therapy (hCG + FSH) for fertility in adolescence/adulthood: "
            "variable response — usually incomplete spermatogenesis; "
            "Testosterone HRT for virilisation of puberty if gonadotrophins inadequate; "
            "Genetic counselling: X-linked; carrier testing for maternal relatives."
        ),
        "locus": "Xp21.2",
        "aa": 470,
        "kDa": 51,
        "omim_gene": "300473",
        "omim_disease": "300200",
        "inheritance": "XLR; males affected; female carriers mostly unaffected; contiguous deletion → AHC+DMD+GKD",
        "gene_class": "Adrenal Development — Nuclear Receptor Repressor — SF1 Target — HH Pathway",
        "key_alerts": [
            "NR0B1-ADRENAL-SMALL-NOT-LARGE: Adrenal glands in NR0B1-AHC are SMALL or hypoplastic on imaging — this distinguishes from Lipoid CAH (STAR) where adrenals are LARGE and lipid-laden; both present as neonatal salt-wasting crisis in males; imaging is a key DDx step",
            "NR0B1-CONTIGUOUS-XPDELETION-DMD-SCREEN: All males with NR0B1-AHC MUST be screened for DMD (serum CK, dystrophin gene) and glycerol kinase deficiency — contiguous Xp21 deletion (AHC+GKD+DMD) is a recognised syndrome; missing DMD diagnosis delays appropriate management and genetic counselling",
            "NR0B1-HYPOGONADOTROPIC-HYPOGONADISM: Males with NR0B1 mutations develop HH — absent puberty, azoospermia, small testes; LH/FSH are undetectable on GnRH stimulation; gonadotrophin (hCG+FSH) therapy may partially restore spermatogenesis; lifelong testosterone HRT required for secondary sex characteristics if gonadotrophins insufficient",
            "NR0B1-BOTH-MINERALOCORTICOID-AND-GLUCOCORTICOID-DEFICIENCY: Unlike MC2R (FGD) where ONLY glucocorticoids are deficient, NR0B1-AHC causes BOTH cortisol AND aldosterone deficiency (full adrenal insufficiency); both hydrocortisone AND fludrocortisone are required; failure to replace mineralocorticoid → salt-wasting crisis",
        ],
        "etiologies": {
            "NR0B1_Infantile_Salt_Wasting_Crisis": {"pct": 60, "phenotype": "neonatal adrenal crisis weeks 1-8; hyponatraemia + hyperkalaemia in male"},
            "NR0B1_Delayed_Puberty_Adolescent": {"pct": 40, "phenotype": "male adolescent with cryptorchidism + absent puberty; HH discovered"},
            "NR0B1_Contiguous_Xp21_Deletion": {"pct": 20, "phenotype": "AHC + DMD + glycerol kinase deficiency triad — large deletion"},
        },
        "stats": {
            "mean_onset_age_y": 3.2,
            "mean_dx_delay_months": 14.5,
            "hh_rate_pct": 100,
            "dmdd_coexistence_pct": 20,
        },
        "dx_delay_distribution": [
            {"bucket": "<3mo", "pct": 55, "desc": "neonatal crisis in males triggers evaluation"},
            {"bucket": "3mo-2y", "pct": 15, "desc": "recurrent crises; CK screening reveals DMD contiguous deletion"},
            {"bucket": ">2y", "pct": 30, "desc": "adolescent presentation with delayed puberty"},
        ],
        "patients": [],
    },
    # ── MC2R — Familial Glucocorticoid Deficiency type 1 ─────────────────────
    {
        "gene": "MC2R",
        "protein": (
            "MC2R — 18p11.21 AR — ACTH-Receptor-Melanocortin-2-Receptor-297aa — "
            "Familial-Glucocorticoid-Deficiency-FGD1 — "
            "ISOLATED-Glucocorticoid-Deficiency — "
            "Aldosterone-NORMAL-No-Salt-Wasting-No-Fludrocortisone — "
            "ACTH-Extremely-HIGH-Hyperpigmentation-MC1R-PATHOGNOMONIC — "
            "Tall-Stature-CLUE — "
            "FGD2-is-MRAP-Cofactor-Mutation"
        ),
        "alias": (
            "MC2R (melanocortin 2 receptor; ACTH receptor); OMIM gene 607397; "
            "Familial Glucocorticoid Deficiency type 1 (FGD1) OMIM 202200. "
            "18p11.21; 297 aa; ~33 kDa; autosomal recessive. "
            "FGD SPECTRUM — GENETIC HETEROGENEITY: "
            "FGD1: MC2R mutation (40-50% of FGD); "
            "FGD2: MRAP mutation (15-20% of FGD); "
            "MRAP is the melanocortin receptor accessory protein — required for MC2R trafficking to plasma membrane; "
            "FGD3/4/5: TXNRD2, STAR (partial), NNT, MCM4 mutations; "
            "all share the FGD phenotype — isolated glucocorticoid deficiency. "
            "MECHANISM OF MC2R: "
            "ACTH binds MC2R on zona fasciculata cells → Gs-cAMP-PKA cascade → "
            "StAR phosphorylation → cholesterol transport → cortisol synthesis. "
            "ANATOMICAL SPECIFICITY: "
            "MC2R is expressed ONLY on zona fasciculata (glucocorticoid zone); "
            "zona glomerulosa (mineralocorticoid) is regulated by the renin-angiotensin-aldosterone system (RAAS) "
            "and potassium — NOT by ACTH; "
            "THEREFORE: MC2R loss → ONLY glucocorticoid deficiency; "
            "aldosterone synthesis is COMPLETELY UNAFFECTED → no salt-wasting, no fludrocortisone needed. "
            "CONSEQUENCE OF MC2R LOSS: "
            "No cortisol despite extremely high ACTH → "
            "(1) Cortisol deficiency: hypoglycaemia, fatigue, collapse; "
            "(2) ACTH unchecked: ACTH is derived from POMC; "
            "ACTH can stimulate MC1R (melanocortin 1 receptor) on melanocytes → "
            "↑ EUMELANIN → HYPERPIGMENTATION — the clinical hallmark; "
            "(3) Adrenal androgens: ACTH drives zona reticularis → ↑ DHEA-S; "
            "no oestrogen/testosterone production (adrenal androgens only, not sex steroids per se); "
            "boys: tall stature (cortisol deficiency → growth hormone axis activated, + adrenal androgens → "
            "growth acceleration → tall stature in children — a diagnostic clue NOT seen in primary adrenal insufficiency). "
            "CLINICAL FEATURES: "
            "HYPERPIGMENTATION: bronze/dark skin, mucous membranes, palmar creases, scars — "
            "present from infancy, often the FIRST NOTICED sign by family; "
            "HYPOGLYCAEMIA: symptomatic, often recurrent; "
            "seizures and coma if severe; may present in neonate or early childhood; "
            "TALL STATURE: paradoxical tall stature in children with adrenal insufficiency → FGD clue "
            "(cortisol deficiency → ↑ IGF-1 axis + adrenal androgen → growth acceleration); "
            "PRESERVED SALT HOMEOSTASIS: no salt-craving, no hyponatraemia, normal BP; "
            "DIAGNOSTIC ALGORITHM: "
            "Morning cortisol LOW + ACTH MARKEDLY elevated (often >1000 ng/L) + aldosterone NORMAL; "
            "ACTH stimulation (Synacthen) test: absent cortisol response; aldosterone rises normally; "
            "this pattern = isolated glucocorticoid deficiency = FGD; "
            "MC2R sequencing (and MRAP if negative); "
            "TREATMENT: "
            "Hydrocortisone 10-15 mg/m²/day (all physiological replacement); "
            "NO fludrocortisone; "
            "stress dosing (adrenal crisis possible if missed); "
            "hyperpigmentation fades slowly with adequate HC (ACTH suppressed); "
            "PROGNOSIS: excellent with HC replacement; "
            "growth normalises; hyperpigmentation resolves; hypoglycaemia prevented."
        ),
        "locus": "18p11.21",
        "aa": 297,
        "kDa": 33,
        "omim_gene": "607397",
        "omim_disease": "202200",
        "inheritance": "AR; FGD1 (MC2R) 40-50% of FGD spectrum; FGD2 (MRAP) 15-20%",
        "gene_class": "Adrenal ACTH Signal — Melanocortin-2 Receptor — Zona Fasciculata Specific",
        "key_alerts": [
            "MC2R-ISOLATED-GLUCOCORTICOID-DEFICIENCY-NO-FLUDROCORTISONE: MC2R (FGD1) causes ONLY glucocorticoid deficiency — aldosterone is NORMAL because zona glomerulosa uses RAAS not ACTH; never give fludrocortisone (salt not wasted); distinguish from all other forms of adrenal insufficiency where mineralocorticoid may be needed",
            "MC2R-ACTH-HYPERPIGMENTATION-PATHOGNOMONIC: ACTH is extremely elevated (often >1000 ng/L, sometimes >10000) and stimulates MC1R on melanocytes → bronze hyperpigmentation of skin, mucous membranes, palmar creases; this is pathognomonic of ACTH-driven primary adrenal insufficiency; fades with adequate hydrocortisone replacement",
            "MC2R-TALL-STATURE-CLUE: Children with FGD/MC2R are characteristically tall for their age (unlike most causes of hypoadrenalism); cortisol deficiency + high adrenal androgens → growth acceleration; this is a clinical CLUE to FGD vs primary adrenal insufficiency with normal height",
            "MC2R-HYPOGLYCAEMIA-SEIZURES-NEONATAL: Cortisol deficiency causes recurrent hypoglycaemia from infancy — presenting as seizures, irritability, or coma; a hypoglycaemic seizure with hyperpigmentation in a child is FGD until proven otherwise; morning cortisol + paired ACTH is the key initial test",
        ],
        "etiologies": {
            "FGD1_MC2R_Classic": {"pct": 45, "phenotype": "isolated glucocorticoid deficiency; hyperpigmentation; hypoglycaemia; tall stature"},
            "FGD2_MRAP": {"pct": 18, "phenotype": "MRAP cofactor deficiency; phenotypically identical to FGD1; MRAP required for MC2R surface trafficking"},
            "FGD_Severe_Neonatal": {"pct": 20, "phenotype": "neonatal hypoglycaemia + seizures; hyperpigmentation present from birth"},
        },
        "stats": {
            "mean_onset_age_y": 1.8,
            "mean_dx_delay_months": 28.6,
            "hyperpigmentation_rate_pct": 98,
            "hypoglycaemia_seizures_pct": 55,
        },
        "dx_delay_distribution": [
            {"bucket": "<6mo", "pct": 35, "desc": "neonatal hypoglycaemic seizures trigger evaluation"},
            {"bucket": "6mo-2y", "pct": 30, "desc": "recurrent hypoglycaemia + hyperpigmentation noted"},
            {"bucket": ">2y", "pct": 35, "desc": "hyperpigmentation attributed to other causes; delayed recognition"},
        ],
        "patients": [],
    },
    # ── AAAS — Triple-A Syndrome (Allgrove Syndrome) ──────────────────────────
    {
        "gene": "AAAS",
        "protein": (
            "AAAS — 12q13.13 AR — ALADIN-Nuclear-Pore-Complex-Protein-546aa — "
            "Triple-A-Allgrove-Syndrome — "
            "Alacrima-FIRST-SIGN-From-Birth-Schirmer-Test-0mm-PATHOGNOMONIC — "
            "Achalasia-Dysphagia-Aspiration — "
            "ACTH-Resistant-Adrenal-Insufficiency — "
            "Progressive-Autonomic-Peripheral-Neuropathy-4th-A — "
            "Ophthalmology-MANDATORY-Optic-Atrophy"
        ),
        "alias": (
            "AAAS (achalasia-adrenocortical insufficiency-alacrimia syndrome gene); OMIM gene 605378; "
            "Triple-A Syndrome / Allgrove Syndrome OMIM 231550. "
            "12q13.13; 546 aa; ~60 kDa; autosomal recessive. "
            "PROTEIN FUNCTION: "
            "ALADIN (alacrima-achalasia-adrenal insufficiency neurologic disorder protein) is a "
            "WD-repeat protein that is a structural component of the nuclear pore complex (NPC). "
            "ALADIN localises to the nuclear pore basket and is required for: "
            "(1) DNA damage response — nuclear import of DNA repair proteins (FEN1, APE1); "
            "(2) Nucleocytoplasmic transport in steroidogenic cells; "
            "(3) Protection against oxidative stress in adrenal cortical cells. "
            "MECHANISM OF DISEASE: "
            "ALADIN loss → defective nuclear import of repair proteins → "
            "accumulation of DNA damage in steroidogenic cells → adrenal cortical dysfunction; "
            "Also affects autonomic neurons, oesophageal myenteric plexus, and lacrimal glands. "
            "THE FOUR As (clinical features): "
            "A1 — ALACRIMA (absent/severely reduced tear secretion): "
            "PRESENT FROM BIRTH — the earliest sign; "
            "parents notice absent crying tears, eye dryness, frequent blinking; "
            "Schirmer test (filter paper in lower conjunctival sac 5 min): 0 mm = PATHOGNOMONIC; "
            "corneal damage (punctate keratopathy, corneal scarring) develops without treatment; "
            "MUST be identified before corneal damage occurs — eye drops from birth/diagnosis. "
            "A2 — ACHALASIA (oesophageal): "
            "failure of lower oesophageal sphincter relaxation → progressive dysphagia; "
            "solids first, then liquids; regurgitation; aspiration pneumonia risk; "
            "oesophageal manometry: absent peristalsis + incomplete LES relaxation — DIAGNOSTIC; "
            "barium swallow: 'bird's beak' narrowing at gastro-oesophageal junction; "
            "treatment: pneumatic dilatation or laparoscopic Heller myotomy + fundoplication. "
            "A3 — ACTH-RESISTANT ADRENAL INSUFFICIENCY: "
            "glucocorticoid deficiency due to ACTH resistance (similar to FGD); "
            "mineralocorticoid may be initially spared in many patients but CAN develop later; "
            "ACTH extremely elevated; cortisol low/absent on stimulation; "
            "treatment: hydrocortisone (+ fludrocortisone if mineralocorticoid deficient). "
            "A4 — AUTONOMIC AND NEUROLOGICAL PROGRESSION: "
            "Motor neuropathy: distal weakness, areflexia, foot drop; "
            "Autonomic neuropathy: orthostatic hypotension, bladder dysfunction, anhidrosis; "
            "Bulbar dysfunction: dysarthria, dysphagia (added to achalasia); "
            "Optic atrophy: visual failure — REGULAR OPHTHALMOLOGY ESSENTIAL; "
            "cerebral demyelination (rare); "
            "neurological progression is the most DISABLING long-term complication — no proven treatment. "
            "DIAGNOSIS — TRIPLE RECOGNITION REQUIRED: "
            "No single test is diagnostic — ALL THREE of the first 3 As must be identified: "
            "Alacrima: Schirmer test (0-5 mm); "
            "Achalasia: oesophageal manometry + barium swallow; "
            "Adrenal insufficiency: ACTH stimulation test (low cortisol + high ACTH); "
            "AAAS gene sequencing: confirms; "
            "common mutations: exon 16 deletions (patients of Middle Eastern origin); "
            "p.R478C frequent in European/Turkish kindreds. "
            "TREATMENT: "
            "Alacrima: artificial tears + lubricating ointment (protect cornea from first diagnosis); "
            "Achalasia: pneumatic dilatation or surgical myotomy; "
            "Adrenal insufficiency: HC + fludrocortisone if mineralocorticoid also deficient; "
            "Neurological: physiotherapy, AFOs, wheelchair when needed; "
            "Ophthalmology: visual fields, OCT, ERG annually; anti-VEGF if neovascularisation."
        ),
        "locus": "12q13.13",
        "aa": 546,
        "kDa": 60,
        "omim_gene": "605378",
        "omim_disease": "231550",
        "inheritance": "AR; ALADIN nuclear pore protein; Middle Eastern and European founder mutations",
        "gene_class": "Adrenal Development + Nuclear Pore — ALADIN WD-Repeat Protein — Oxidative Stress Response",
        "key_alerts": [
            "AAAS-ALACRIMA-FIRST-SIGN-BIRTH-SCHIRMER-ZERO: Absent lacrimation (alacrima) is present from birth and is the FIRST feature of Triple-A syndrome; parents report no tears with crying; Schirmer test 0 mm is pathognomonic; artificial tears must be started immediately to prevent corneal scarring; any child with unexplained alacrima needs AAAS evaluation",
            "AAAS-DIAGNOSTIC-TRIAD-ALL-THREE-REQUIRED: No single test diagnoses Triple-A syndrome — ALL THREE As must be actively looked for: Alacrima (Schirmer), Achalasia (manometry), and ACTH-resistant adrenal insufficiency (Synacthen test); presenting with any one A requires systematic evaluation for the others; neurological (4th A) develops progressively",
            "AAAS-ACHALASIA-ASPIRATION-MANOMETRY-MANDATORY: Oesophageal manometry is mandatory in all AAAS patients — achalasia causes progressive dysphagia and aspiration pneumonia; pneumatic dilatation or Heller myotomy is the treatment; nutritional status must be monitored; gastrostomy may eventually be required",
            "AAAS-NEUROLOGICAL-PROGRESSION-MOST-DISABLING: The autonomic and peripheral neuropathy (the 4th A) is the most severe long-term complication — optic atrophy, foot drop, orthostatic hypotension, bladder dysfunction; annual ophthalmology + neurophysiology; no disease-modifying treatment proven; antioxidant trials ongoing",
        ],
        "etiologies": {
            "Triple_A_Full_Phenotype": {"pct": 60, "phenotype": "all 3 As present at diagnosis + neurological progression in 50%"},
            "Triple_A_Alacrima_Plus_Adrenal": {"pct": 25, "phenotype": "alacrima + adrenal insufficiency without prominent achalasia initially"},
            "Triple_A_Predominantly_Neurological": {"pct": 15, "phenotype": "motor neuropathy dominant; triple-A identified in neurological workup"},
        },
        "stats": {
            "mean_onset_age_y": 6.2,
            "mean_dx_delay_months": 48.0,
            "neurological_complication_pct": 50,
            "corneal_damage_without_treatment_pct": 70,
        },
        "dx_delay_distribution": [
            {"bucket": "<12mo", "pct": 20, "desc": "alacrima noted at birth; early referral"},
            {"bucket": "1-5y", "pct": 35, "desc": "achalasia presents; triad recognised"},
            {"bucket": ">5y", "pct": 45, "desc": "neurological presentation; retrospective triple-A identification"},
        ],
        "patients": [],
    },
    # ── ABCD1 — X-linked Adrenoleukodystrophy ─────────────────────────────────
    {
        "gene": "ABCD1",
        "protein": (
            "ABCD1 — Xq28 XLR — ALDP-Peroxisomal-ABC-Transporter-745aa — "
            "X-linked-Adrenoleukodystrophy-VLCFA-Accumulation — "
            "THREE-Phenotypes: CCALD-35-40pct-Most-Urgent / AMN-Adult-Myelopathy / Addison-Only-15-20pct — "
            "Newborn-Screen-C26:0-LPC-NOW-INCLUDED — "
            "CCALD-Loes-Score-<9-Gadolinium-LESION-HSCT-NOW — "
            "Loes->12-Or-Symptomatic-HSCT-NOT-Effective"
        ),
        "alias": (
            "ABCD1 (ATP-binding cassette, sub-family D, member 1; adrenoleukodystrophy protein ALDP); "
            "OMIM gene 300371; X-linked Adrenoleukodystrophy (X-ALD) OMIM 300100. "
            "Xq28; 745 aa; ~84 kDa; X-linked recessive (XLR). "
            "PROTEIN FUNCTION: "
            "ALDP is a half-transporter of the ABCD subfamily localised to the peroxisomal membrane; "
            "it forms homodimers and transports very-long-chain fatty acid (VLCFA) -CoA esters "
            "into the peroxisome for beta-oxidation. "
            "VLCFA (C24:0, C25:0, C26:0 — saturated fatty acids with 24-26 carbons) cannot enter the "
            "mitochondria (normal LCFA beta-oxidation route) and MUST use peroxisomal beta-oxidation. "
            "ABCD1 loss → VLCFA cannot enter peroxisome → accumulate in plasma, adrenal cortex, "
            "and central/peripheral nervous system myelin. "
            "VLCFA TOXICITY: "
            "Adrenal: VLCFA intercalate into cell membranes → adrenocortical cell death → "
            "Addison disease (primary adrenal insufficiency); "
            "CNS: VLCFA disrupt myelin → neuroinflammation → demyelination; "
            "the precise trigger for the CCALD vs AMN phenotype is unknown (not genotype-driven). "
            "THREE CLINICAL PHENOTYPES (in affected males): "
            "1. CHILDHOOD CEREBRAL ALD (CCALD) — 35-40%: "
            "onset 4-10 years; "
            "INITIAL: behavioural change (distractibility, aggression, school decline); "
            "then: cognitive decline, visual/auditory deficits, spasticity, seizures; "
            "MRI: white matter hyperintensity (T2/FLAIR) — typically starts posterior parieto-occipital; "
            "gadolinium enhancement at the advancing edge of demyelination = active neuroinflammation; "
            "LOES SCORE: quantifies MRI severity (0 = normal, 34 = most severe); "
            "Loes <9 + Gd+ = window for HSCT (transplant halts progression if pre-symptomatic or early); "
            "Loes ≥9-12 or symptomatic → HSCT no longer effective (damage irreversible); "
            "WITHOUT HSCT: vegetative state or death within 2-5 years of cerebral onset. "
            "2. ADRENOMYELONEUROPATHY (AMN) — most common adult phenotype: "
            "onset 20-40 years; "
            "progressive spastic paraparesis (stiff legs, scissor gait); "
            "peripheral neuropathy; "
            "bladder and bowel dysfunction; "
            "most have adrenal insufficiency (50-70%); "
            "MRI: spinal cord atrophy ± mild cerebral white matter changes; "
            "NO effective disease-modifying therapy for AMN myelopathy; "
            "HSCT does NOT benefit AMN spinal cord disease. "
            "3. ADDISON-ONLY — 15-20%: "
            "primary adrenal insufficiency without neurological involvement at time of diagnosis; "
            "may be pre-symptomatic CCALD or AMN variant; "
            "ALL males with Addison disease of unknown cause should have VLCFA measured. "
            "FEMALE CARRIERS: "
            "Never get CCALD; "
            "~50% develop AMN-like symptoms after age 40-50 (milder than males); "
            "adrenal insufficiency rare but occurs; "
            "VLCFA mildly elevated. "
            "NEWBORN SCREENING — CRITICAL UPDATE: "
            "X-ALD is now included in newborn screening panels in the USA, EU, and many countries; "
            "VLCFA C26:0-lysophosphatidylcholine (C26:0-LPC) by MS/MS on dried blood spot; "
            "enables pre-symptomatic identification of males → enrol in MRI surveillance programme; "
            "NBS does not predict phenotype (CCALD vs AMN cannot be predicted from genotype). "
            "TREATMENT: "
            "CCALD (pre-symptomatic or early, Loes <9, Gd+): "
            "HSCT: allogeneic HLA-matched HSCT halts cerebral progression if performed early enough; "
            "lentiviral gene therapy (Skysona / elivaldogene autotemcel) — FDA 2022: "
            "autologous HSC transduced with functional ABCD1; approved for early active CCALD; "
            "avoids GvHD (no donor needed); "
            "Lorenzo's Oil (4:1 erucic:oleic acid): reduces plasma VLCFA but does NOT halt neurological disease; "
            "may delay CCALD onset in pre-symptomatic males (NBS programme + LO surveillance); "
            "ADRENAL INSUFFICIENCY: hydrocortisone + fludrocortisone in all affected males with adrenal failure; "
            "does not affect neurological disease; "
            "AMN: symptomatic — antispastics (baclofen, tizanidine), bladder management, physiotherapy; "
            "MONITORING ALL AFFECTED MALES: "
            "Annual MRI brain (T2/FLAIR + gadolinium) from age 4 to 12; "
            "Adrenal function test (ACTH stimulation) annually; "
            "family cascade testing (maternal relatives, sisters — check carrier status)."
        ),
        "locus": "Xq28",
        "aa": 745,
        "kDa": 84,
        "omim_gene": "300371",
        "omim_disease": "300100",
        "inheritance": "XLR; ABCD1 at Xq28; all males with pathogenic ABCD1 variants are affected (100% penetrance for VLCFA elevation); phenotype unpredictable",
        "gene_class": "Peroxisomal VLCFA Transport — ABCD Subfamily ABC Transporter — Adrenal + CNS White Matter",
        "key_alerts": [
            "ABCD1-NEWBORN-SCREEN-NBS-CCALD-URGENCY: X-ALD is now on newborn screening (C26:0-LPC by MS/MS); a positive NBS triggers brain MRI surveillance from age 4 every 6 months; Loes score <9 + gadolinium enhancement = HSCT or gene therapy NOW — this is the only window to prevent irreversible cerebral damage; Loes >12 or symptomatic → transplant no longer effective",
            "ABCD1-LOES-SCORE-CRITICAL-THRESHOLD: Loes score is the MRI severity index (0-34); Loes <9 + active gadolinium enhancement = active neuroinflammation = HSCT/gene therapy window; Loes 9-12 = borderline — urgent specialist decision; Loes >12 or already symptomatic = HSCT cannot reverse damage; Lorenzo's Oil does not halt CCALD once started",
            "ABCD1-ADDISONS-UNKNOWN-CAUSE-VLCFA: ALL males with Addison disease of unknown cause MUST have plasma VLCFA measured (C26:0, C24:0/C22:0 ratio); 15-20% of X-ALD males present initially as isolated adrenal insufficiency; missing the diagnosis delays MRI surveillance and HSCT opportunity for CCALD prevention",
            "ABCD1-GENE-THERAPY-SKYSONA-FDA2022: Elivaldogene autotemcel (Skysona, bluebird bio) is an autologous lentiviral gene therapy FDA-approved 2022 for early active CCALD (Loes 0.5-9 with Gd+ lesion); avoids GvHD risk of allogeneic HSCT; requires functional ABCD1 ex vivo transduction into autologous HSCs; does not benefit AMN or late-stage CCALD",
        ],
        "etiologies": {
            "CCALD_Childhood_Cerebral": {"pct": 37, "phenotype": "rapid cerebral demyelination 4-10y; behavioural then cognitive then neurological decline; HSCT window critical"},
            "AMN_Adrenomyeloneuropathy": {"pct": 45, "phenotype": "adult-onset spastic paraparesis + peripheral neuropathy + bladder; no HSCT benefit for spinal cord"},
            "Addison_Only_X_ALD": {"pct": 18, "phenotype": "isolated adrenal insufficiency; NBS or VLCFA workup reveals X-ALD; MRI surveillance mandatory"},
            "Female_Carrier_AMN_Like": {"pct": 50, "phenotype": "~50% of female ABCD1 carriers develop AMN symptoms after age 40; adrenal AI rare; no CCALD"},
        },
        "stats": {
            "mean_onset_age_y": 8.5,
            "mean_dx_delay_months": 18.0,
            "adrenal_insufficiency_pct": 70,
            "ccald_hsct_eligible_pct": 35,
        },
        "dx_delay_distribution": [
            {"bucket": "<6mo", "pct": 30, "desc": "NBS positive → early identification"},
            {"bucket": "6mo-2y", "pct": 20, "desc": "adrenal crisis workup reveals VLCFA"},
            {"bucket": ">2y", "pct": 50, "desc": "CCALD or AMN symptom onset triggers evaluation"},
        ],
        "patients": [],
    },
]


def _generate_patients():
    for idx, gene_data in enumerate(ADRENAL_GENES):
        gene = gene_data["gene"]
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        patients = []
        for i in range(40):
            if gene == "CYP21A2":
                subtype = rng.choices(
                    ["salt_wasting", "simple_virilizing", "non_classic"],
                    weights=[75, 25, 0],  # only classic for this cohort
                )[0]
                # mix in some NCAH
                if i >= 30:
                    subtype = "non_classic"
                onset_age = (
                    rng.uniform(0.02, 0.25) if subtype == "salt_wasting"
                    else rng.uniform(0.5, 4) if subtype == "simple_virilizing"
                    else rng.randint(15, 35)
                )
                dx_delay = (
                    rng.randint(0, 2) if subtype == "salt_wasting"
                    else rng.randint(2, 12) if subtype == "simple_virilizing"
                    else rng.randint(12, 72)
                )
                patients.append({
                    "patient_id": f"CYP21A2-{i+1:03d}",
                    "onset_age": round(onset_age, 2),
                    "dx_delay_months": dx_delay,
                    "phenotype": subtype,
                    "sex_karyotype": rng.choice(["46XX", "46XY"]),
                    "gene": gene, "seed": seed,
                })
            elif gene == "CYP11B1":
                hypertension = rng.random() < 0.67
                onset_age = rng.uniform(0.1, 5.0)
                dx_delay = rng.randint(3, 30)
                patients.append({
                    "patient_id": f"CYP11B1-{i+1:03d}",
                    "onset_age": round(onset_age, 2),
                    "dx_delay_months": dx_delay,
                    "phenotype": "cah_11bhod_hypertensive" if hypertension else "cah_11bhod_virilizing",
                    "hypertension": hypertension,
                    "sex_karyotype": rng.choice(["46XX", "46XY"]),
                    "gene": gene, "seed": seed,
                })
            elif gene == "CYP17A1":
                karyotype = rng.choice(["46XX", "46XY"])
                onset_age = rng.randint(12, 20)
                dx_delay = rng.randint(6, 60)
                gonadectomy = karyotype == "46XY"
                patients.append({
                    "patient_id": f"CYP17A1-{i+1:03d}",
                    "onset_age": onset_age,
                    "dx_delay_months": dx_delay,
                    "phenotype": "17ohd_primary_amenorrhoea",
                    "karyotype": karyotype,
                    "hypertension": True,
                    "gonadectomy": gonadectomy,
                    "gene": gene, "seed": seed,
                })
            elif gene == "STAR":
                karyotype = rng.choice(["46XX", "46XY"])
                onset_age = rng.uniform(0.01, 0.15)
                dx_delay = rng.randint(0, 4)
                gonadectomy = karyotype == "46XY"
                patients.append({
                    "patient_id": f"STAR-{i+1:03d}",
                    "onset_age": round(onset_age, 3),
                    "dx_delay_months": dx_delay,
                    "phenotype": "lipoid_cah_neonatal_crisis",
                    "karyotype": karyotype,
                    "adrenal_large_imaging": True,
                    "gonadectomy": gonadectomy,
                    "gene": gene, "seed": seed,
                })
            elif gene == "NR0B1":
                pattern = rng.choices(["infantile", "adolescent", "contiguous"], weights=[60, 30, 10])[0]
                onset_age = (
                    rng.uniform(0.02, 0.5) if pattern == "infantile"
                    else rng.randint(13, 18)
                )
                dx_delay = rng.randint(0, 18) if pattern == "infantile" else rng.randint(6, 36)
                dmd_coexistence = pattern == "contiguous"
                patients.append({
                    "patient_id": f"NR0B1-{i+1:03d}",
                    "onset_age": round(onset_age, 2),
                    "dx_delay_months": dx_delay,
                    "phenotype": pattern,
                    "dmd_coexistence": dmd_coexistence,
                    "hh_present": True,
                    "gene": gene, "seed": seed,
                })
            elif gene == "MC2R":
                onset_age = rng.uniform(0.1, 5.0)
                dx_delay = rng.randint(6, 60)
                hyperpig = rng.random() < 0.98
                hypoglycaemia = rng.random() < 0.75
                tall_stature = rng.random() < 0.60
                patients.append({
                    "patient_id": f"MC2R-{i+1:03d}",
                    "onset_age": round(onset_age, 2),
                    "dx_delay_months": dx_delay,
                    "phenotype": "fgd1_isolated_glucocorticoid_deficiency",
                    "hyperpigmentation": hyperpig,
                    "hypoglycaemia": hypoglycaemia,
                    "tall_stature": tall_stature,
                    "gene": gene, "seed": seed,
                })
            elif gene == "AAAS":
                onset_age = rng.randint(2, 15)
                dx_delay = rng.randint(12, 96)
                alacrima = True
                achalasia = rng.random() < 0.85
                neuro = rng.random() < 0.50
                patients.append({
                    "patient_id": f"AAAS-{i+1:03d}",
                    "onset_age": onset_age,
                    "dx_delay_months": dx_delay,
                    "phenotype": "triple_a_full" if (alacrima and achalasia and neuro) else "triple_a_partial",
                    "alacrima": alacrima,
                    "achalasia": achalasia,
                    "neurological": neuro,
                    "gene": gene, "seed": seed,
                })
            else:  # ABCD1
                phenotype = rng.choices(
                    ["ccald", "amn", "addison_only"],
                    weights=[37, 45, 18]
                )[0]
                onset_age = (
                    rng.randint(4, 10) if phenotype == "ccald"
                    else rng.randint(20, 45) if phenotype == "amn"
                    else rng.randint(10, 40)
                )
                dx_delay = rng.randint(2, 36)
                loes_score = (
                    rng.randint(0, 12) if phenotype == "ccald"
                    else rng.randint(0, 4) if phenotype == "amn"
                    else 0
                )
                hsct_eligible = phenotype == "ccald" and loes_score < 9
                patients.append({
                    "patient_id": f"ABCD1-{i+1:03d}",
                    "onset_age": onset_age,
                    "dx_delay_months": dx_delay,
                    "phenotype": phenotype,
                    "loes_score": loes_score,
                    "hsct_eligible": hsct_eligible,
                    "adrenal_insufficiency": rng.random() < 0.70,
                    "gene": gene, "seed": seed,
                })
        gene_data["patients"] = patients


_generate_patients()


def overview():
    all_delays = [
        p.get("dx_delay_months", 0)
        for g in ADRENAL_GENES for p in g["patients"]
    ]
    all_ages = [
        p.get("onset_age", 0)
        for g in ADRENAL_GENES for p in g["patients"]
    ]
    total = sum(len(g["patients"]) for g in ADRENAL_GENES)
    return {
        "atlas": "Hereditary Adrenal Disorders Atlas — Complete 8-Gene Reference",
        "subtitle": (
            "CYP21A2 (CAH-21OHD) · CYP11B1 (CAH-11βOHD) · CYP17A1 (17α-OHD) · STAR (Lipoid CAH) · "
            "NR0B1 (AHC+HH) · MC2R (FGD1) · AAAS (Triple-A) · ABCD1 (X-ALD) — "
            "320 Patients (8×40, Seeds 1766–1773)"
        ),
        "total_patients": total,
        "seed_range": f"{SEED_BASE}–{SEED_BASE + 7}",
        "aggregate_stats": {
            "genes_covered": 8,
            "patients_per_gene": 40,
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
            "mean_dx_age": round(sum(all_ages) / len(all_ages), 1),
        },
        "genes": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "aa": g["aa"],
                "kDa": g["kDa"],
                "n_patients": len(g["patients"]),
                "mean_dx_age": round(
                    sum(p.get("onset_age", 0) for p in g["patients"]) / len(g["patients"]), 1
                ),
                "mean_dx_delay_months": round(
                    sum(p.get("dx_delay_months", 0) for p in g["patients"]) / len(g["patients"]), 1
                ),
            }
            for g in ADRENAL_GENES
        ],
        "top_alerts": [
            "CYP21A2-STRESS-DOSING-MANDATORY-SICK-DAY-RULE: ALL CAH-21OHD patients carry an emergency HC kit; double oral dose for fever; IM/IV HC 25-50 mg/m2/day for vomiting or surgery — adrenal crisis is preventable but fatal if missed; 17-OHP on newborn screen >30 nmol/L is pathognomonic for SW-CAH",
            "CYP11B1-HYPERTENSION-IN-CAH-DIAGNOSTIC: Hypertension + virilization + low plasma renin = CYP11B1 deficiency; DOC accumulation causes mineralocorticoid HTN; do NOT give fludrocortisone; HC suppresses ACTH → ↓ DOC → BP normalises",
            "CYP17A1-HYPERTENSION-AMENORRHOEA-NO-VIRILIZATION: HTN + primary amenorrhoea + absent secondary sex chars + NO virilization = CYP17A1; 46XY presents as phenotypic female; karyotype mandatory; gonadectomy mandatory; 17-OHP is LOW (not high)",
            "STAR-LIPOID-CAH-MOST-SEVERE: All steroidogenesis abolished; bilateral large lipid-laden adrenals; 46XY = complete female phenotype; neonatal crisis within days; HC + fludrocortisone lifelong; gonadectomy 46XY",
            "NR0B1-CONTIGUOUS-DELETION-DMD-SCREEN: NR0B1-AHC males MUST have CK + DMD gene testing; Xp21 contiguous deletion (AHC + GKD + DMD) is a recognised triad; adrenal glands SMALL on imaging (unlike STAR where large)",
            "MC2R-ISOLATED-GLUCOCORTICOID-NO-FLUDROCORTISONE: FGD1 = isolated glucocorticoid deficiency; aldosterone NORMAL; ACTH >1000 ng/L → hyperpigmentation PATHOGNOMONIC; tall stature clue; HC only — no fludrocortisone",
            "AAAS-ALACRIMA-SCHIRMER-ZERO-FROM-BIRTH: Alacrima present from birth — first sign of Triple-A; Schirmer test 0 mm = pathognomonic; artificial tears immediately to prevent corneal scarring; triad (alacrima + achalasia + adrenal insufficiency) must ALL be identified",
            "ABCD1-CCALD-LOES-LESS-THAN-9-HSCT-NOW: CCALD in a boy — if Loes <9 + gadolinium enhancement → HSCT or gene therapy (Skysona) is the ONLY intervention that halts progression; surveillance MRI every 6 months age 4-12 is mandatory for all affected males; NBS C26:0-LPC enables pre-symptomatic detection",
        ],
    }


def breakdown():
    result = []
    for idx, g in enumerate(ADRENAL_GENES):
        delays = [p.get("dx_delay_months", 0) for p in g["patients"]]
        ages = [p.get("onset_age", 0) for p in g["patients"]]
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
                "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
                "mean_dx_age": round(sum(ages) / len(ages), 1),
                "n_patients": len(g["patients"]),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": g["patients"][:10],
        })
    return result


def definitions():
    return {
        "concepts": {
            "Adrenal Steroidogenesis — The Biosynthetic Cascade": (
                "The adrenal cortex produces three classes of steroids from cholesterol via a "
                "compartmentalised cascade: "
                "STEP 1 (RATE-LIMITING): Cholesterol → Pregnenolone. "
                "Requires STAR protein (mitochondrial cholesterol transport) + CYP11A1 (cholesterol side-chain cleavage). "
                "StAR is the gate — without it (Lipoid CAH), ALL downstream steroidogenesis fails. "
                "STEP 2: Pregnenolone → DHEA (Δ5 pathway) or Progesterone (Δ4 pathway). "
                "CYP17A1 17α-hydroxylase + 17,20-lyase required for sex steroid and cortisol arms. "
                "Without CYP17A1: no cortisol, no sex steroids; DOC and corticosterone accumulate → "
                "mineralocorticoid excess (hypertension). "
                "CORTISOL ARM: Progesterone → 17-OHP (CYP17A1) → 11-deoxycortisol (CYP21A2) → "
                "Cortisol (CYP11B1). "
                "ALDOSTERONE ARM: Progesterone → DOC (CYP21A2) → Corticosterone (CYP11B1) → "
                "Aldosterone (CYP11B2). "
                "ANDROGEN ARM: DHEA → Androstenedione (CYP17A1 lyase) → Testosterone → DHT. "
                "KEY ENZYME BLOCKS: "
                "CYP21A2 block → ↑ 17-OHP; ↓ cortisol + aldosterone; ↑ androgens; "
                "CYP11B1 block → ↑ 11-deoxycortisol + DOC; ↓ cortisol; DOC → HTN; ↑ androgens; "
                "CYP17A1 block → ↑ DOC + corticosterone; ↓ cortisol; ↓↓ sex steroids; HTN; "
                "STAR block → ALL steroids absent; cholesterol accumulates."
            ),
            "The HPA Axis — ACTH Feedback and Its Disruption": (
                "The hypothalamic-pituitary-adrenal (HPA) axis: "
                "Hypothalamus → CRH → Pituitary → ACTH → Adrenal cortex → Cortisol → "
                "NEGATIVE FEEDBACK to hypothalamus and pituitary. "
                "In CAH and adrenal insufficiency: low cortisol → no feedback → ↑ CRH → ↑ ACTH. "
                "CONSEQUENCES OF CHRONIC ↑ ACTH: "
                "(1) Adrenal androgen excess (in CAH): ACTH drives zona reticularis → "
                "virilization, precocious puberty, primary amenorrhoea; "
                "(2) Hyperpigmentation: ACTH derived from POMC; "
                "ACTH stimulates MC1R on melanocytes → ↑ melanin → bronze pigmentation "
                "(MC2R/FGD1, Addison disease, X-ALD adrenal insufficiency); "
                "(3) Adrenal enlargement: ACTH trophic effect → bilateral adrenal hyperplasia; "
                "in CYP21A2-SW: adrenal hyperplasia because cortisol not made → ACTH unchecked → "
                "adrenal grows trying to make cortisol; "
                "in Lipoid CAH: STAR loss → cholesterol cannot enter mitochondria → "
                "ACTH drives massive cholesterol accumulation → lipoid appearance; "
                "(4) Adrenal rest tumours (TART/OART): ectopic adrenal tissue in gonads responds to ACTH → "
                "enlarges → impairs fertility; suppression with HC shrinks them."
            ),
            "Adrenal Crisis — Recognition, Prevention, and Emergency Treatment": (
                "Adrenal crisis (acute adrenal insufficiency) is life-threatening and preventable. "
                "CAUSES IN HEREDITARY ADRENAL DISORDERS: "
                "Missed diagnosis; intercurrent illness without stress dosing; vomiting preventing oral HC; "
                "surgery/anaesthesia without perioperative HC cover; drug interactions. "
                "CLINICAL FEATURES: "
                "Hypotension (vascular collapse — no cortisol → vasopressor response lost); "
                "Hyponatraemia + hyperkalaemia (mineralocorticoid deficient forms: CYP21A2-SW, STAR, NR0B1-AHC); "
                "Hypoglycaemia (cortisol maintains gluconeogenesis); "
                "Nausea/vomiting/abdominal pain; confusion → coma; "
                "Fever (often the precipitating infection). "
                "EMERGENCY TREATMENT: "
                "1. IV/IM hydrocortisone 100 mg (adult) / 25-50 mg (child) IMMEDIATELY — before labs; "
                "2. IV normal saline 0.9% bolus (1-2 L adult / 20 mL/kg child) for hypotension; "
                "3. IV dextrose for hypoglycaemia (10% glucose infusion); "
                "4. Identify + treat precipitating cause; "
                "5. Continue IV/IM HC q4-6h until able to tolerate oral; "
                "6. Increase fludrocortisone dose if SW crisis. "
                "PREVENTION (THE SICK DAY RULE): "
                "Fever >38°C / minor illness: DOUBLE oral HC dose; "
                "Vomiting or diarrhoea: CANNOT rely on oral; IM HC kit at home; "
                "ALL patients carry a 'hydrocortisone emergency injection kit' (Solu-Cortef 100 mg IM); "
                "Medical alert bracelet / Steroid Emergency Card; "
                "Perioperative: HC 25 mg/m² IV at induction + infusion perioperatively."
            ),
            "Newborn Screening for Adrenal Disorders — What to Screen For": (
                "HEREDITARY ADRENAL DISORDERS ON NEWBORN SCREEN (NBS): "
                "CYP21A2 (CAH-21OHD): 17-OHP elevated on dried blood spot (DBS) — universal in most countries; "
                "gestational age correction required (premature infants have higher 17-OHP falsely); "
                "positive screen → urgently check electrolytes + plasma 17-OHP + ACTH; "
                "X-ALD (ABCD1): C26:0-lysophosphatidylcholine (C26:0-LPC) by MS/MS on DBS; "
                "added to RUSP (US Recommended Universal Screening Panel) 2016; "
                "now active in most US states + many EU countries; "
                "positive NBS → confirmatory plasma VLCFA + ABCD1 gene sequencing → "
                "enrol in MRI surveillance (annual MRI from age 4, 6-monthly if any white matter signal); "
                "NBS CANNOT PREDICT phenotype (CCALD vs AMN vs Addison). "
                "NOT YET UNIVERSALLY ON NBS: "
                "CYP11B1 (CAH-11βOHD): some programmes check 11-deoxycortisol on DBS; "
                "FGD/MC2R: no routine screen; "
                "Triple-A/AAAS: no routine screen; "
                "Lipoid CAH/STAR: no specific NBS test — detected when NBS 17-OHP pattern suggests severe CAH; "
                "NR0B1-AHC: no routine screen; diagnosed on clinical presentation."
            ),
            "VLCFA and X-ALD — Biochemistry and Newborn Screen Interpretation": (
                "Very-long-chain fatty acids (VLCFA) are saturated fatty acids with ≥22 carbons. "
                "KEY VLCFA SPECIES IN X-ALD: "
                "C26:0 (hexacosanoic acid) — the primary diagnostic marker; "
                "C26:0/C22:0 ratio — elevated in X-ALD; "
                "C24:0/C22:0 ratio — supportive. "
                "PLASMA VLCFA (diagnostic): "
                "C26:0 elevated in ALL affected males (100% sensitivity); "
                "C26:0-LPC elevated in NBS DBS (MS/MS method); "
                "Female carriers: VLCFA mildly elevated (sensitivity ~80-85% — some carriers have normal VLCFA; "
                "ABCD1 gene sequencing is the gold standard for carrier testing). "
                "PATHOPHYSIOLOGY: "
                "VLCFA cannot be beta-oxidised in mitochondria (too long); "
                "MUST enter peroxisome via ABCD1 transport; "
                "ABCD1 loss → VLCFA accumulate → intercalate into adrenal cell membranes → cell death → Addison; "
                "in CNS: VLCFA disrupt myelin stability + trigger neuroinflammatory cascade (mechanism not fully understood); "
                "Lorenzo's Oil (4:1 erucic acid:oleic acid): "
                "competitive inhibition of VLCFA elongase → reduces C26:0 in plasma; "
                "normalises plasma VLCFA; "
                "DOES NOT reduce VLCFA in CNS or adrenal; "
                "Asymptomatic pre-symptomatic males (NBS): Lorenzo's Oil may delay CCALD onset "
                "(ALD Connect trial data — observational); "
                "does NOT halt CCALD once demyelination starts."
            ),
        },
        "pharmacological_distinctions": [
            "Hydrocortisone (HC) 10-15 mg/m²/day in 3 doses — physiological glucocorticoid replacement in ALL hereditary adrenal disorders; preferred in children (shorter half-life, least growth suppression vs prednisolone or dexamethasone); stress dosing 2-3× mandatory for illness/surgery; perioperative IV HC 25 mg/m² at induction",
            "Fludrocortisone 0.05-0.2 mg/day — mineralocorticoid replacement ONLY in disorders with aldosterone deficiency: CYP21A2-SW, STAR, NR0B1-AHC; NOT needed in CYP11B1 (DOC excess acts as mineralocorticoid), CYP17A1 (DOC excess), MC2R/FGD (zona glomerulosa intact), AAAS (mineralocorticoid may be initially spared); monitor plasma renin activity (target upper normal)",
            "Dexamethasone — used in adults with CAH for overnight ACTH suppression (longer-acting, more potent); NOT preferred in growing children (causes growth suppression, Cushingoid features at lower doses); also used as prenatal dexamethasone (experimental/controversial) for at-risk 46XX fetuses with CYP21A2 mutations",
            "Hydrocortisone sodium succinate (Solu-Cortef) IM emergency kit — ALL patients with adrenal insufficiency MUST carry; 100 mg IM administered by trained carer before hospital assessment in crisis; critical for isolated patients, travel, and paediatric school/nursery management; Steroid Emergency Card issued alongside",
            "Testosterone HRT / gonadotrophin therapy (hCG + rFSH) — for NR0B1-AHC males with HH; testosterone alone for virilisation; hCG + rFSH for fertility induction (spermatogenesis) — variable response, often incomplete; long-term testosterone required if gonadotrophins insufficient",
            "Oestrogen HRT (± progesterone cycling in 46XX) — for CYP17A1 and STAR 46XX patients with ovarian failure, and 46XY patients post-gonadectomy; essential for secondary sex characteristics, bone health, cardiovascular protection; transdermal preferred; cyclic progesterone for those with uterus (46XX CYP17A1)",
            "Lorenzo's Oil (erucic acid 20% + oleic acid 80%) — reduces plasma VLCFA C26:0 in X-ALD; used in pre-symptomatic males identified by NBS to potentially delay CCALD onset; DOES NOT halt neurological disease once demyelination starts; requires fat-modified diet; platelet monitoring (thrombocytopenia risk); monitoring plasma VLCFA monthly initially",
            "Elivaldogene autotemcel (Skysona, bluebird bio) — lentiviral gene therapy FDA-approved 2022 for early CCALD (Loes 0.5-9, gadolinium+); autologous HSCs transduced with functional ABCD1 ex vivo; avoids allogeneic GvHD; conditioning with busulfan/cyclophosphamide required; does not benefit AMN or late-stage CCALD; long-term safety follow-up ongoing",
            "Pneumatic dilatation / Heller myotomy — treatment for oesophageal achalasia in Triple-A/AAAS syndrome; pneumatic dilatation (endoscopic balloon) is first-line in most centres; laparoscopic Heller myotomy + Dor fundoplication for failures; nutritional support and swallowing therapy adjunctive; per-oral endoscopic myotomy (POEM) is a newer option",
            "Spironolactone — mineralocorticoid receptor antagonist; used in CYP17A1 and CYP11B1 when hypertension from DOC excess persists despite HC; helps counteract aldosterone/DOC-mediated potassium wasting; NOT a substitute for HC in glucocorticoid deficiency; monitor potassium + renal function",
        ],
        "key_standards": [
            "Endocrine Society CAH Clinical Practice Guideline (2018): universal NBS with 17-OHP for CAH-21OHD; confirms NBS positives with second-tier steroid profiling by LC-MS/MS; DNA diagnosis mandatory; surgical decision-making for virilized 46XX must be patient/family centred; monitoring 17-OHP + androstenedione + growth + bone age annually in children",
            "ESPE/LWPES Consensus Statement on CAH Management: aim for normal growth velocity and bone age advancement <1 SD; over-treatment with HC causes growth failure and obesity — monitor carefully; adrenal crisis prevention including school protocols, travel kits, perioperative management",
            "X-ALD RUSP Inclusion (2016) and NBS Implementation: C26:0-LPC by MS/MS on DBS; positive NBS → plasma VLCFA confirmation → ABCD1 sequencing → MRI from age 4 (6-monthly); Lorenzo's Oil for pre-symptomatic males; Loes <9 + Gd+ → refer to HSCT/gene therapy centre within 4-8 weeks",
            "ALD Connect / Global ALD Registry: all X-ALD patients enrolled; natural history data; HSCT/gene therapy outcomes; carrier testing programme; available at www.aldconnect.org",
            "Endocrine Society Adrenal Insufficiency Clinical Practice Guideline (2016): sick day rules education mandatory at every visit; Steroid Emergency Card + IM HC kit prescribed at diagnosis; perioperative steroid protocols; stress dosing written action plan",
            "Triple-A Syndrome Diagnostic Standard: Schirmer test at diagnosis AND at each annual review; oesophageal manometry if dysphagia present; ACTH stimulation test; neurophysiology (nerve conduction studies) every 2 years; ophthalmology (visual fields, OCT) annually; AAAS gene panel",
            "CAH Newborn Emergency Protocol: any male neonate with hyponatraemia + hyperkalaemia + hypoglycaemia = assume adrenal crisis until proven otherwise; IV HC 25 mg/m² bolus + NS bolus + glucose infusion BEFORE awaiting labs; rapid 17-OHP + electrolytes + blood gas; family cascade testing after index case identified",
            "FGD (MC2R/MRAP) Diagnosis Standard: isolated primary adrenal insufficiency with ACTH >2× ULN and aldosterone NORMAL on ACTH stimulation test = FGD pattern; test MC2R and MRAP sequencing; ACTH >1000 ng/L strongly supports FGD; no fludrocortisone; hyperpigmentation as monitoring marker (should fade with adequate HC)",
        ],
    }
