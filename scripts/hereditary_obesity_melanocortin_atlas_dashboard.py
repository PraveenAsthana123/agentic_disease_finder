#!/usr/bin/env python3
"""Hereditary-Obesity-Melanocortin-Atlas — Complete 8-Gene Leptin-Melanocortin Pathway Atlas
MC4R    (Melanocortin 4 receptor; 332 aa; 18q21.32; AD;
         Most common monogenic obesity — 1-2% of severe obesity;
         Haploinsufficiency → impaired satiety signalling;
         Setmelanotide partial benefit; hyperinsulinaemia early marker;
         seed SEED_BASE+0)
LEPR    (Leptin receptor; 1165 aa; 1p31.3; AR;
         Severe hyperphagia from birth; undetectable serum leptin response;
         Setmelanotide (Imcivree) FDA-approved 2020 — FIRST targeted therapy;
         Hypogonadotrophic hypogonadism + GH deficiency;
         seed SEED_BASE+1)
LEP     (Leptin; 167 aa; 7q32.1; AR;
         Undetectable serum leptin — PATHOGNOMONIC diagnostic;
         Metreleptin (recombinant leptin) therapy CURATIVE — not setmelanotide;
         Hypogonadism reversal with treatment; immune dysfunction;
         seed SEED_BASE+2)
PCSK1   (Proprotein convertase subtilisin/kexin type 1/3; 753 aa; 5q15; AR;
         Obesity + neonatal malabsorptive diarrhoea + hypoglycaemia + hypogonadism;
         Setmelanotide FDA-approved 2021 for PCSK1-deficiency obesity;
         Multiple endocrine dysfunction (TSH, ACTH, GH, FSH/LH) — panlopituitarism;
         seed SEED_BASE+3)
POMC    (Proopiomelanocortin; 267 aa; 2p23.3; AR;
         Obesity + RED HAIR (absent α-MSH → no MC1R activation) + hypocortisolism;
         Adrenal crisis risk (ACTH deficiency) — cortisol replacement MANDATORY;
         Setmelanotide FDA-approved 2020 for POMC-deficiency obesity;
         seed SEED_BASE+4)
SH2B1   (SH2B adaptor protein 1; 613 aa; 16p11.2; AD;
         Severe obesity + insulin resistance + behavioural problems (aggression, ADHD);
         16p11.2 deletion syndrome overlaps; leptin signal amplifier → deficit → hyperphagia;
         seed SEED_BASE+5)
KSR2    (Kinase suppressor of Ras 2; 950 aa; 12q24.22; AR/compound-het;
         Severe early-onset obesity + reduced resting heart rate + insulin resistance;
         Metformin dramatically effective (UNIQUE) — KSR2 regulates AMPK-MAPK-energy sensing;
         seed SEED_BASE+6)
SIM1    (Single-minded homolog 1; 786 aa; 6q16.3; AD;
         Haploinsufficiency → severe hyperphagia + Prader-Willi-like + aggressive behaviour;
         Hypothalamic paraventricular nucleus (PVN) development critical;
         Oxytocin trials ongoing (PVN-OXT axis);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 2006-2013)
"""

import random

SEED_BASE = 2006

OB_GENES = [
    # -- MC4R -- Most common monogenic obesity -----------------------------------------
    {
        "gene": "MC4R",
        "alt_name": "MC4R (Melanocortin 4 Receptor / Most Common Monogenic Obesity — 1-2% Severe Obesity)",
        "protein": (
            "MC4R -- 18q21.32 AD -- MC4R-332aa -- "
            "Most-Common-Monogenic-Obesity-1-2pct-Severe-Obesity -- "
            "Haploinsufficiency-Impaired-Satiety-Signalling -- "
            "Hyperinsulinaemia-Early-Marker-Before-Obesity-Manifests -- "
            "Setmelanotide-Partial-Benefit-Functional-Assessment-Needed"
        ),
        "locus": "18q21.32",
        "protein_size": "332 aa",
        "inheritance": "AD (autosomal dominant) — haploinsufficiency; some compound heterozygotes",
        "age_of_onset": (
            "Early childhood: hyperphagia onset typically 1-5 years; "
            "Rapid weight gain exceeds centiles from infancy; "
            "Increased linear growth velocity (tall stature) in childhood — DISTINCTIVE in MC4R; "
            "Hyperinsulinaemia appears early — before frank obesity on centile charts; "
            "Penetrance variable — heterozygous carriers range from obese to normal weight; "
            "Prevalence: ~1-2% of patients with BMI >35; ~5% of patients with BMI >40 + early onset; "
            "Puberty: normal (distinguishes from POMC/LEPR — hypogonadism absent in MC4R); "
            "Adult: BMI often 35-60; metabolic syndrome common"
        ),
        "key_biomarker": (
            "Serum leptin: ELEVATED (paradoxically — leptin resistance in hypothalamus); "
            "Fasting insulin + HOMA-IR: markedly elevated — hyperinsulinaemia DISTINCTIVE early feature; "
            "IGF-1: elevated (tall stature — MC4R suppresses GH release; paradoxically GH normal/elevated via alternate); "
            "HbA1c: elevated with time — T2D risk high; "
            "Lipid panel: dyslipidaemia — elevated TG, low HDL; "
            "Genetic: MC4R sequencing — identify pathogenic variant; assess functional impact (GOF partial vs LOF); "
            "Functional cAMP assay: determines whether MC4R variant retains partial function (predictive for setmelanotide benefit); "
            "Bone density: elevated (MC4R regulates bone mass via SNS)"
        ),
        "pathognomonic": (
            "Severe early-onset obesity + hyperinsulinaemia + tall stature + NORMAL puberty = MC4R first; "
            "DISTINGUISH from POMC: MC4R has normal cortisol + normal hair colour; "
            "DISTINGUISH from LEPR: MC4R leptin is elevated (leptin resistance), LEPR is low due to receptor loss; "
            "DISTINGUISH from hypothyroidism: thyroid function normal in MC4R; "
            "DISTINGUISH from genetic syndromes (PWS, Albright): specific features absent; "
            "Tall stature: MC4R haploinsufficiency → reduced sympathetic tone → increased IGF-1 → tall in childhood; "
            "Blood pressure: often lower than expected for BMI (MC4R regulates SNS tone); "
            "Hyperphagia assessment: Dykens Hyperphagia Questionnaire score >11 total in clinical trials"
        ),
        "treatment": (
            "Setmelanotide (Imcivree): MC4R agonist — FDA approved for POMC/LEPR/PCSK1 deficiency; "
            "MC4R deficiency ITSELF: setmelanotide under study — benefit in those with partial-function variants; "
            "GLP-1 receptor agonists: liraglutide 3mg (Saxenda), semaglutide 2.4mg (Wegovy) — FDA approved for obesity; "
            "Evidence: MC4R pathogenic variants may have REDUCED response to GLP-1RA (downstream pathway impaired); "
            "Bariatric surgery: sleeve gastrectomy / RYGB — effective but results variable; "
            "MC4R GOF (gain of function) variants: protective against obesity — therapeutic concept; "
            "Dietary: structured very-low-calorie diet; behavioural hyperphagia management; "
            "Metabolic: metformin for insulin resistance; statin for dyslipidaemia; "
            "Genetic counselling: AD — 50% inheritance; variable penetrance; "
            "Screen family: siblings + parents + children of index case for metabolic disease"
        ),
        "critical_flags": [
            "MC4R-MOST-COMMON-MONOGENIC-OBESITY-1-2pct-SEVERE",
            "MC4R-HYPERINSULINAEMIA-EARLY-MARKER-BEFORE-OBESITY-PEAKS",
            "MC4R-TALL-STATURE-CHILDHOOD-DISTINCTIVE",
            "MC4R-SETMELANOTIDE-BENEFIT-DEPENDS-ON-VARIANT-FUNCTION",
            "MC4R-GLP1RA-REDUCED-RESPONSE-DOWNSTREAM-IMPAIRED",
            "MC4R-NORMAL-PUBERTY-DISTINGUISHES-FROM-POMC-LEPR",
            "MC4R-LEPTIN-ELEVATED-RESISTANCE-NOT-DEFICIENCY",
        ],
        "seed": SEED_BASE + 0,
    },
    # -- LEPR -- Leptin receptor deficiency (AR) ----------------------------------------
    {
        "gene": "LEPR",
        "alt_name": "LEPR (Leptin Receptor / AR Deficiency — Setmelanotide FDA-Approved 2020 — Hypogonadism + GH Deficiency)",
        "protein": (
            "LEPR -- 1p31.3 AR -- LEPR-1165aa -- "
            "Severe-Hyperphagia-From-Birth-Undetectable-Leptin-Response -- "
            "Setmelanotide-Imcivree-FDA-Approved-2020-FIRST-Targeted-Therapy -- "
            "Hypogonadotrophic-Hypogonadism-GH-Deficiency -- "
            "Biallelic-LOF-Biallelic-LOF-LEPR-Defines-LEPRD"
        ),
        "locus": "1p31.3",
        "protein_size": "1165 aa",
        "inheritance": "AR (autosomal recessive) — biallelic LOF",
        "age_of_onset": (
            "Neonatal/infancy: hyperphagia from first weeks of life; "
            "Normal birth weight — obesity develops rapidly within months; "
            "Serum leptin: VERY HIGH (produced normally; receptor absent — leptin cannot signal); "
            "Immune dysfunction: recurrent infections — leptin normally activates immune cells; "
            "Hypogonadotrophic hypogonadism: LH/FSH low; pubertal failure; amenorrhoea in females; "
            "Growth hormone deficiency: short stature trajectory without GH; "
            "Hypothalamic hypothyroidism: TSH may be blunted; free T4 low-normal; "
            "Adrenal function: usually preserved (distinguishes from POMC); "
            "BMI: typically 40-70 without treatment; adiposity predominantly truncal"
        ),
        "key_biomarker": (
            "Serum leptin: MARKEDLY ELEVATED (paradox — receptor absent, leptin accumulates); "
            "LH/FSH: low/undetectable — hypogonadotrophic hypogonadism; "
            "IGF-1: low — GH deficiency secondary to absent leptin signalling; "
            "GH stimulation test: subnormal GH response; "
            "Free T4 + TSH: central hypothyroidism screen; "
            "Cortisol: usually normal (unlike POMC); "
            "HbA1c + fasting insulin: hyperinsulinaemia + T2D progression; "
            "Molecular: LEPR biallelic sequencing — exons 1-20; large deletions by MLPA; "
            "Functional: leptin-stimulated JAK2-STAT3 signalling assay (research)"
        ),
        "pathognomonic": (
            "Severe infantile hyperphagia + obesity + VERY HIGH serum leptin + failed puberty = LEPR; "
            "DISTINGUISH from LEP (leptin deficiency): LEPR has VERY HIGH leptin; LEP has UNDETECTABLE leptin; "
            "DISTINGUISH from POMC: LEPR has normal hair colour + normal cortisol; "
            "DISTINGUISH from PWS: LEPR lacks characteristic facial features + normal chromosome 15q; "
            "Immune dysfunction (recurrent sinopulmonary infections): LEPR deficiency impairs T-cell function; "
            "GH deficiency: responds to GH replacement but obesity-driven resistance limits height gain; "
            "Hyperleptinaemia + obesity + pubertal failure = BIALLELIC LEPR until proven otherwise"
        ),
        "treatment": (
            "Setmelanotide (Imcivree): Rhythm Pharmaceuticals — subcutaneous daily injection; "
            "FDA approved November 2020 for LEPR-deficiency obesity (CALIB trial — 13.9kg mean weight loss); "
            "Mechanism: bypasses non-functional LEPR → directly activates MC4R in hypothalamus; "
            "Hyperphagia score: significant reduction in hunger (VAS); "
            "GH replacement: for confirmed GH deficiency + short stature; "
            "Oestrogen/testosterone: for hypogonadism (pubertal induction); "
            "Thyroxine: if central hypothyroidism confirmed; "
            "AVOID: exogenous leptin (metreleptin) — receptor absent, no benefit (opposite to LEP deficiency); "
            "Bariatric surgery: modest effect due to downstream MC4R pathway still dysfunctional; "
            "Genetic counselling: AR — 25% recurrence; heterozygous parents — screen for metabolic disease"
        ),
        "critical_flags": [
            "LEPR-SETMELANOTIDE-FDA-2020-FIRST-TARGETED-THERAPY",
            "LEPR-SERUM-LEPTIN-VERY-HIGH-PARADOX-RECEPTOR-ABSENT",
            "LEPR-METRELEPTIN-NOT-USEFUL-RECEPTOR-ABSENT",
            "LEPR-HYPOGONADOTROPHIC-HYPOGONADISM-PUBERTAL-FAILURE",
            "LEPR-GH-DEFICIENCY-SECONDARY-TO-LEPTIN-SIGNAL-LOSS",
            "LEPR-IMMUNE-DYSFUNCTION-RECURRENT-INFECTIONS",
            "LEPR-DISTINGUISH-FROM-LEP-LEPTIN-LEVEL-KEY-DIFFERENCE",
        ],
        "seed": SEED_BASE + 1,
    },
    # -- LEP -- Leptin deficiency (AR) --------------------------------------------------
    {
        "gene": "LEP",
        "alt_name": "LEP (Leptin / AR Deficiency — UNDETECTABLE Serum Leptin — Metreleptin CURATIVE — Not Setmelanotide)",
        "protein": (
            "LEP -- 7q32.1 AR -- LEP-167aa -- "
            "Undetectable-Serum-Leptin-PATHOGNOMONIC-Diagnostic -- "
            "Metreleptin-Recombinant-Leptin-Therapy-CURATIVE -- "
            "Setmelanotide-NOT-Indicated-LEPR-Intact-Leptin-Absent -- "
            "Biallelic-LOF-Adipokine-Absent-from-Birth"
        ),
        "locus": "7q32.1",
        "protein_size": "167 aa",
        "inheritance": "AR (autosomal recessive) — biallelic LOF",
        "age_of_onset": (
            "Neonatal/infancy: hyperphagia from birth; normal birth weight; "
            "Serum leptin: UNDETECTABLE (< 0.5 ng/mL) — PATHOGNOMONIC; "
            "Rapid-onset severe obesity in first year of life; "
            "Immune dysfunction: severe — recurrent infections; impaired T-cell + NK-cell function; "
            "Hypogonadotrophic hypogonadism: no spontaneous puberty (leptin is permissive for GnRH pulse); "
            "Normal adrenal function (unlike POMC — cortisol normal); "
            "Normal hair colour (unlike POMC — MC1R function intact, α-MSH from POMC intact, just no leptin); "
            "Thyroid: central hypothyroidism sometimes (mild); "
            "BMI: 50-80 in untreated adults; adiposity generalised"
        ),
        "key_biomarker": (
            "Serum leptin: UNDETECTABLE (< 0.5 ng/mL) — diagnostic gold standard; "
            "DISTINGUISH from LEPR: LEPR has VERY HIGH leptin; LEP has ABSENT leptin; "
            "LH/FSH: low — hypogonadotrophic hypogonadism; "
            "Cortisol: NORMAL (critical DDx from POMC — adrenal crisis NOT a risk in LEP); "
            "Free T4 + TSH: mild central hypothyroidism possible; "
            "HbA1c + insulin: severe hyperinsulinaemia; T2D by early adulthood; "
            "Molecular: LEP biallelic sequencing; founder variants in Pakistani/Turkish populations (ΔG133, L72S); "
            "Immunological: lymphocyte subset analysis — CD4+ T-cell function impaired"
        ),
        "pathognomonic": (
            "Undetectable serum leptin (<0.5 ng/mL) + severe infantile obesity = LEP until proven otherwise; "
            "DISTINGUISH from LEPR: leptin level is the single most important test (undetectable vs very high); "
            "DISTINGUISH from POMC: LEP has normal hair colour + normal cortisol + no adrenal crisis risk; "
            "Pakistan/Turkey: ΔG133 frameshift founder variant — 1 in 200 carriers in some communities; "
            "Dramatic response to metreleptin: hyperphagia resolves within days; puberty initiates in adolescents; "
            "Immune dysfunction: first presentation sometimes as recurrent/severe infection before obesity diagnosis; "
            "Diagnostic trap: very low (not absent) leptin may be MCL missense — full sequencing needed"
        ),
        "treatment": (
            "Metreleptin (Myalept): recombinant methionyl-leptin — subcutaneous daily injection; "
            "CURATIVE: dramatic reduction in hyperphagia (within days); weight loss; puberty induction; "
            "FDA approved for GENERALISED LIPODYSTROPHY (2014) — used off-label for congenital LEP deficiency; "
            "EMA: approved for lipodystrophy; compassionate use for LEP deficiency in EU; "
            "Setmelanotide: NOT indicated — LEPR is intact; leptin replacement is the correct approach; "
            "GH replacement: if GH deficiency documented; "
            "Sex hormone: oestrogen/testosterone for hypogonadism until metreleptin initiates puberty; "
            "Thyroxine: if central hypothyroidism confirmed; "
            "Genetic counselling: AR — 25% recurrence; "
            "Screen extended family in high-prevalence communities (Pakistan, Turkey)"
        ),
        "critical_flags": [
            "LEP-UNDETECTABLE-SERUM-LEPTIN-PATHOGNOMONIC-DIAGNOSTIC",
            "LEP-METRELEPTIN-CURATIVE-WITHIN-DAYS",
            "LEP-SETMELANOTIDE-NOT-INDICATED-LEPR-INTACT",
            "LEP-NORMAL-CORTISOL-NO-ADRENAL-CRISIS-RISK-UNLIKE-POMC",
            "LEP-NORMAL-HAIR-COLOUR-DISTINGUISHES-FROM-POMC",
            "LEP-PAKISTAN-TURKEY-FOUNDER-DELTA-G133-VARIANT",
            "LEP-DISTINGUISH-FROM-LEPR-LEPTIN-LEVEL-KEY",
        ],
        "seed": SEED_BASE + 2,
    },
    # -- PCSK1 -- Proprotein convertase 1/3 deficiency (AR) ----------------------------
    {
        "gene": "PCSK1",
        "alt_name": "PCSK1 (PC1/3 / AR Deficiency — Obesity + Neonatal Diarrhoea + Panhypopituitarism — Setmelanotide FDA-2021)",
        "protein": (
            "PCSK1 -- 5q15 AR -- PCSK1-753aa -- "
            "Obesity-Plus-Neonatal-Malabsorptive-Diarrhoea-Plus-Hypoglycaemia -- "
            "Panhypopituitarism-TSH-ACTH-GH-FSH-LH-ALL-Deficient -- "
            "Setmelanotide-FDA-Approved-2021-for-PCSK1-Deficiency-Obesity -- "
            "PCSK1-Converts-ProHormones-POMC-ProInsulin-ProGlucagon-In-Pituitary-Gut"
        ),
        "locus": "5q15",
        "protein_size": "753 aa",
        "inheritance": "AR (autosomal recessive) — biallelic LOF",
        "age_of_onset": (
            "Neonatal: malabsorptive diarrhoea — often first presentation; "
            "PC1/3 cleaves proglucagon in gut (GLP-1 production fails) → osmotic diarrhoea; "
            "Neonatal hypoglycaemia: proinsulin not converted → hyperpro-insulinaemia (high proinsulin, low insulin); "
            "Obesity onset: early childhood once diarrhoea managed; "
            "Panhypopituitarism: ACTH deficiency → cortisol insufficiency (adrenal crisis risk); "
            "TSH deficiency → central hypothyroidism; GH deficiency → short stature; "
            "LH/FSH deficiency → hypogonadotrophic hypogonadism; "
            "Diagnostic challenge: diarrhoea + failure-to-thrive in neonatal period precedes obesity; "
            "Later obesity BMI: 35-65 without treatment"
        ),
        "key_biomarker": (
            "Proinsulin:insulin ratio: MARKEDLY ELEVATED (proinsulin not converted — diagnostic fingerprint); "
            "Cortisol stimulation test: subnormal → ACTH deficiency (adrenal crisis risk); "
            "Free T4 + TSH: central hypothyroidism (TSH may be low-normal or blunted); "
            "IGF-1: low → GH deficiency; GH stimulation test: subnormal; "
            "LH/FSH: low → hypogonadotrophic hypogonadism; "
            "Faecal elastase: malabsorption markers; "
            "HbA1c + glucose: paradoxical hyperglycaemia + hypoglycaemia pattern; "
            "Molecular: PCSK1 biallelic sequencing — confirm pathogenic variants; "
            "Serum leptin: high (receptor normal; leptin production normal — obesity drives elevation)"
        ),
        "pathognomonic": (
            "Neonatal malabsorptive diarrhoea + neonatal hypoglycaemia + panhypopituitarism + childhood obesity = PCSK1; "
            "DISTINGUISH from POMC: PCSK1 has panhypopituitarism (POMC has isolated ACTH deficiency); "
            "DISTINGUISH from LEPR: PCSK1 has neonatal diarrhoea + panhypopit; LEPR has pure obesity + immune; "
            "Proinsulin:insulin ratio elevated: PATHOGNOMONIC — proinsulin accumulates when PC1/3 absent; "
            "Adrenal crisis: ACTH deficiency → cortisol insufficiency → stress-induced adrenal crisis RISK; "
            "Stress dosing hydrocortisone: MANDATORY during illness, surgery, trauma; "
            "DISTINGUISH from MODY: proinsulin ratio normal in MODY; PCSK1 has multi-endocrine deficit"
        ),
        "treatment": (
            "Setmelanotide (Imcivree): FDA approved June 2021 for PCSK1-deficiency obesity; "
            "Mechanism: directly activates MC4R (bypasses absent PC1/3 → POMC processing defect); "
            "Hydrocortisone: MANDATORY replacement for ACTH deficiency; stress dosing protocol; "
            "Levothyroxine: for central hypothyroidism (free T4 guided, not TSH); "
            "GH replacement: for confirmed GH deficiency; "
            "Sex hormone: pubertal induction with oestrogen/testosterone; "
            "GLP-1 levels: paradoxically low (proglucagon not cleaved); GLP-1RA drugs: uncertain benefit; "
            "Neonatal diarrhoea management: pancreatic enzyme replacement + dietary adjustment; "
            "Genetic counselling: AR — 25% recurrence; "
            "Emergency card: ADRENAL CRISIS RISK — carry cortisol emergency card"
        ),
        "critical_flags": [
            "PCSK1-SETMELANOTIDE-FDA-2021-APPROVED",
            "PCSK1-ACTH-DEFICIENCY-ADRENAL-CRISIS-RISK-STRESS-DOSING-MANDATORY",
            "PCSK1-PROINSULIN-RATIO-ELEVATED-PATHOGNOMONIC",
            "PCSK1-NEONATAL-DIARRHOEA-FIRST-PRESENTATION-BEFORE-OBESITY",
            "PCSK1-PANHYPOPITUITARISM-ALL-AXES-AFFECTED",
            "PCSK1-TSH-FREE-T4-GUIDED-NOT-TSH-CENTRAL-HYPOTHYROIDISM",
            "PCSK1-EMERGENCY-HYDROCORTISONE-SICK-DAY-RULES-MANDATORY",
        ],
        "seed": SEED_BASE + 3,
    },
    # -- POMC -- Proopiomelanocortin deficiency (AR) ------------------------------------
    {
        "gene": "POMC",
        "alt_name": "POMC (Proopiomelanocortin / AR Deficiency — Obesity + RED HAIR + Hypocortisolism — Setmelanotide FDA-2020)",
        "protein": (
            "POMC -- 2p23.3 AR -- POMC-267aa -- "
            "Obesity-RED-HAIR-Hypocortisolism-TRIAD-PATHOGNOMONIC -- "
            "Adrenal-Crisis-ACTH-Deficiency-Cortisol-Replacement-MANDATORY -- "
            "Setmelanotide-Imcivree-FDA-Approved-2020-for-POMC-Deficiency -- "
            "MC4R-MC2R-MC1R-All-Downstream-Receptors-Deprived"
        ),
        "locus": "2p23.3",
        "protein_size": "267 aa",
        "inheritance": "AR (autosomal recessive) — biallelic LOF",
        "age_of_onset": (
            "Neonatal: adrenal crisis risk from birth (ACTH absent → neonatal hypocortisolism); "
            "Neonatal hypoglycaemia: cortisol deficiency → counter-regulatory failure; "
            "Hyperphagia onset: weeks to months of life; "
            "Red/auburn hair: absent α-MSH → MC1R not activated → loss of eumelanin → red/auburn hair; "
            "Fair/pale skin: absent MC1R activation → reduced melanin; "
            "BMI: 40-70 in childhood; severe early-onset obesity; "
            "Puberty: may fail (LH/FSH partially dependent on leptin-POMC axis); "
            "Hypothyroidism: absent TSH from thyrotrophs (POMC → MSH → TRH signalling impaired)"
        ),
        "key_biomarker": (
            "ACTH: undetectable or very low (POMC → ACTH not generated); "
            "Cortisol: very low — primary failure to synthesise cortisol; "
            "Cortisol stimulation test (Synacthen): subnormal — adrenal is intact but no ACTH drive; "
            "ACTH stimulation: subnormal — central ACTH deficiency; "
            "β-endorphin: very low (POMC → β-endorphin also absent); "
            "Serum leptin: markedly elevated (adiposity driven; LEPR intact; MC4R dysfunctional); "
            "Prolactin: may be elevated (loss of MSH-mediated inhibition); "
            "Molecular: POMC biallelic sequencing; Dutch/German: exon 3 frameshift founder; "
            "Skin: absence of tanning response to UV (MC1R not activated by α-MSH)"
        ),
        "pathognomonic": (
            "Severe infantile obesity + RED/AUBURN HAIR + neonatal adrenal crisis = POMC PATHOGNOMONIC; "
            "TRIAD: obesity + red hair + hypocortisolism — present in almost all biallelic POMC; "
            "DISTINGUISH from LEPR: POMC has red hair + low cortisol; LEPR has normal hair + normal cortisol; "
            "DISTINGUISH from LEP: LEP has normal hair + normal cortisol; POMC has the triad; "
            "Red hair in darkly-pigmented ethnicity: even stronger signal for POMC (MC1R loss expresses most in dark-hair individuals); "
            "Adrenal crisis trigger: infection, surgery, fasting → acute cortisol deficiency → cardiovascular collapse; "
            "POMC heterozygotes (carriers): slight increase in obesity risk — incomplete penetrance"
        ),
        "treatment": (
            "Setmelanotide (Imcivree): FDA approved June 2020 for POMC-deficiency obesity; "
            "Mechanism: direct MC4R agonist — bypasses absent POMC-derived α-MSH; "
            "Clinical trial: >90% patients had ≥10% body weight reduction; hyperphagia scores normalised; "
            "Hydrocortisone: MANDATORY replacement (ACTH deficiency); weight-based dosing; "
            "Stress dosing: MANDATORY protocol — illness, surgery, vomiting → parenteral hydrocortisone; "
            "Levothyroxine: if central hypothyroidism confirmed (free T4 guided); "
            "AVOID: high-dose UV exposure (MC1R absent → sunburn risk without tanning protection); "
            "Sunscreen: high-factor mandatory (melanin synthesis absent); "
            "GH replacement: if GH deficiency documented; "
            "Genetic counselling: AR — 25% recurrence; heterozygous parents may have slightly increased obesity risk"
        ),
        "critical_flags": [
            "POMC-SETMELANOTIDE-FDA-2020-FIRST-APPROVED",
            "POMC-RED-HAIR-OBESITY-HYPOCORTISOLISM-TRIAD-PATHOGNOMONIC",
            "POMC-ADRENAL-CRISIS-ACTH-DEFICIENCY-NEONATAL-RISK",
            "POMC-HYDROCORTISONE-REPLACEMENT-MANDATORY-FROM-BIRTH",
            "POMC-STRESS-DOSING-EMERGENCY-CARD-MANDATORY",
            "POMC-SUNSCREEN-HIGH-FACTOR-MC1R-ABSENT-BURNS-NOT-TANS",
            "POMC-NEONATAL-HYPOGLYCAEMIA-CORTISOL-DEFICIENCY-COUNTER-REGULATORY",
        ],
        "seed": SEED_BASE + 4,
    },
    # -- SH2B1 -- SH2B adaptor protein 1 deficiency (AD) --------------------------------
    {
        "gene": "SH2B1",
        "alt_name": "SH2B1 (SH2B Adaptor 1 / AD — Severe Obesity + Insulin Resistance + Behavioural Problems — 16p11.2 Deletion Overlap)",
        "protein": (
            "SH2B1 -- 16p11.2 AD -- SH2B1-613aa -- "
            "Severe-Obesity-Insulin-Resistance-Behavioural-Problems-Aggression-ADHD -- "
            "16p11.2-Deletion-Syndrome-Overlap-Autism-ASD-Component -- "
            "Leptin-Signal-Amplifier-SH2B1-Enhances-JAK2-STAT3-Pathway -- "
            "Haploinsufficiency-Impairs-Leptin-Insulin-Neural-Signals"
        ),
        "locus": "16p11.2",
        "protein_size": "613 aa",
        "inheritance": "AD (autosomal dominant) — haploinsufficiency; 16p11.2 deletion (500kb) frequently de novo",
        "age_of_onset": (
            "Early childhood: hyperphagia + rapid weight gain by age 2-5; "
            "Behavioural abnormalities: aggression, emotional dysregulation, ADHD — DISTINCTIVE; "
            "Social difficulties: may overlap with ASD features (16p11.2 microdeletion spectrum); "
            "Cognitive: borderline to mild intellectual disability in some; "
            "Insulin resistance: severe — disproportionate to obesity; "
            "Puberty: usually normal; "
            "16p11.2 deletion (500kb): detected by chromosomal microarray — includes SH2B1 + nearby genes; "
            "Point variants in SH2B1 alone: obesity + insulin resistance WITHOUT ASD features (purer phenotype); "
            "BMI: 35-65; adiposity truncal"
        ),
        "key_biomarker": (
            "Fasting insulin + HOMA-IR: severely elevated — insulin resistance out of proportion to BMI; "
            "Chromosomal microarray: detects 16p11.2 deletion (500kb de novo or inherited); "
            "SH2B1 sequencing: for point variants (negative microarray but clinical suspicion); "
            "Leptin: elevated (resistance); "
            "HbA1c: T2D risk elevated; "
            "Behavioural assessment: ADHD/autism screening — Conners, ADOS; "
            "Lipid panel: dyslipidaemia; "
            "Echo + EKG: cardiac screening for 16p11.2 deletion spectrum"
        ),
        "pathognomonic": (
            "Severe early-onset obesity + aggression/emotional dysregulation + severe insulin resistance = SH2B1/16p11.2; "
            "Behavioural component: key distinguishing feature — MC4R and LEP/LEPR lack behavioural phenotype; "
            "16p11.2 deletion: de novo most common; ASD-spectrum + obesity package; "
            "DISTINGUISH from MC4R: SH2B1 has behavioural problems + more severe insulin resistance; "
            "DISTINGUISH from BBS (Bardet-Biedl): SH2B1 lacks polydactyly + retinal dystrophy; "
            "Insulin resistance marker: often more severe than expected for degree of adiposity; "
            "Mechanism: SH2B1 amplifies both insulin receptor and leptin receptor (JAK2) signalling; "
            "Double deficiency effect: impaired leptin AND insulin signalling simultaneously"
        ),
        "treatment": (
            "No specifically approved targeted therapy (setmelanotide not approved for SH2B1); "
            "GLP-1 receptor agonists: semaglutide/liraglutide — may benefit (insulin resistance component responsive); "
            "Metformin: first-line for insulin resistance; "
            "Bariatric surgery: effective; RYG bypass preferred for severe insulin resistance; "
            "Behavioural therapy: structured ABA for behavioural dysregulation; methylphenidate for ADHD; "
            "Educational support: IEP/special education if intellectual disability; "
            "Genetic counselling: AD haploinsufficiency; de novo 16p11.2 deletion — low recurrence risk; "
            "Inherited point variant: 50% recurrence; "
            "Screen siblings and parents: microarray + clinical assessment; "
            "Cardiometabolic monitoring: annual fasting glucose, HbA1c, lipids, BP"
        ),
        "critical_flags": [
            "SH2B1-BEHAVIOURAL-PROBLEMS-AGGRESSION-ADHD-DISTINCTIVE",
            "SH2B1-16p11.2-DELETION-MICROARRAY-MANDATORY-NOT-SEQUENCING-ALONE",
            "SH2B1-INSULIN-RESISTANCE-DISPROPORTIONATE-TO-OBESITY",
            "SH2B1-SETMELANOTIDE-NOT-APPROVED-FOR-SH2B1",
            "SH2B1-ASD-FEATURES-16P11.2-DELETION-SPECTRUM",
            "SH2B1-METFORMIN-GLP1RA-MAINSTAY-TREATMENT",
            "SH2B1-DE-NOVO-16P11.2-DELETION-FREQUENTLY-SPORADIC",
        ],
        "seed": SEED_BASE + 5,
    },
    # -- KSR2 -- Kinase suppressor of Ras 2 (AR/compound-het) --------------------------
    {
        "gene": "KSR2",
        "alt_name": "KSR2 (Kinase Suppressor of Ras 2 / AR — Severe Obesity + Low Heart Rate — Metformin DRAMATICALLY Effective — Unique)",
        "protein": (
            "KSR2 -- 12q24.22 AR -- KSR2-950aa -- "
            "Severe-Early-Onset-Obesity-Reduced-Resting-Heart-Rate-DISTINCTIVE -- "
            "Metformin-Dramatically-Effective-UNIQUE-AMPK-MAPK-Energy-Sensing -- "
            "Insulin-Resistance-Severe-KSR2-Scaffold-AMPK-RAS-ERK-MEK -- "
            "Compound-Heterozygous-or-Homozygous-Biallelic-LOF"
        ),
        "locus": "12q24.22",
        "protein_size": "950 aa",
        "inheritance": "AR (autosomal recessive) — biallelic LOF or compound heterozygous",
        "age_of_onset": (
            "Early childhood: severe hyperphagia and obesity from age 1-5; "
            "Reduced resting heart rate: bradycardia-tendency DISTINCTIVE (not typical for monogenic obesity); "
            "Mechanism: KSR2 scaffolds AMPK → energy sensing defect → reduced fatty acid oxidation; "
            "Cellular energy inefficiency: reduced mitochondrial fat oxidation → weight gain; "
            "Insulin resistance: severe — KSR2 also scaffolds insulin receptor-MAPK cascade; "
            "Glucose metabolism: impaired; T2D risk high from early adulthood; "
            "Puberty: generally normal; "
            "IQ: usually normal; "
            "BMI: 35-70; adiposity generalised; "
            "Prevalence: rare; compound heterozygous most common configuration"
        ),
        "key_biomarker": (
            "Resting heart rate: reduced (bradycardia-tendency) at baseline — DISTINCTIVE; "
            "Fasting insulin + HOMA-IR: markedly elevated; "
            "HbA1c: elevated; T2D progression; "
            "Indirect calorimetry: reduced resting metabolic rate and fatty acid oxidation; "
            "Molecular: KSR2 biallelic sequencing — exons 1-22; "
            "AMPK activation assay: reduced AICAR-stimulated AMPK activity (research); "
            "Metformin trial: dramatic response (weight loss + improved insulin sensitivity) supports diagnosis; "
            "Lipid panel: dyslipidaemia; elevated TG; "
            "Leptin: elevated (adiposity driven)"
        ),
        "pathognomonic": (
            "Severe childhood obesity + REDUCED RESTING HEART RATE + severe insulin resistance = KSR2; "
            "Reduced heart rate: DISTINCTIVE — most monogenic obesity has normal or elevated HR; "
            "Metformin DRAMATIC response: unique to KSR2 — metformin activates AMPK (KSR2-independent pathway); "
            "DISTINGUISH from MC4R: KSR2 has low HR + normal puberty + normal leptin receptor; "
            "DISTINGUISH from hypothyroidism: KSR2 has normal thyroid function; "
            "DISTINGUISH from primary mitochondrial disease: KSR2 lacks ragged-red fibres, elevated lactate; "
            "Mechanism insight: KSR2 normally binds AMPK catalytic subunit directly — loss = AMPK scaffolding failure; "
            "Compound heterozygosity: most cases in Farooqi discovery cohort (2013)"
        ),
        "treatment": (
            "Metformin: DRAMATICALLY effective — unique to KSR2; "
            "Mechanism: metformin activates AMPK via AMPK kinase (LKB1-AMPKK) pathway, bypassing absent KSR2 scaffold; "
            "Weight loss: significant on metformin even before lifestyle changes; "
            "Dosing: standard metformin titration (500mg BD, increase to 1g BD/2g/day); "
            "GLP-1 receptor agonists: adjunctive for insulin resistance; "
            "Bariatric surgery: effective as adjunct; "
            "NO setmelanotide: MC4R pathway intact — setmelanotide not indicated; "
            "Cardiological evaluation: resting HR monitoring; EKG for conduction; "
            "Dietary: reduced calorie + structured fat content (FAO impaired — avoid very high fat); "
            "Genetic counselling: AR — 25% recurrence; screen siblings"
        ),
        "critical_flags": [
            "KSR2-METFORMIN-DRAMATICALLY-EFFECTIVE-UNIQUE-MECHANISM",
            "KSR2-REDUCED-RESTING-HEART-RATE-DISTINCTIVE",
            "KSR2-SETMELANOTIDE-NOT-INDICATED-MC4R-INTACT",
            "KSR2-AMPK-SCAFFOLD-LOSS-ENERGY-SENSING-DEFECT",
            "KSR2-REDUCED-FATTY-ACID-OXIDATION-INDIRECT-CALORIMETRY",
            "KSR2-COMPOUND-HETEROZYGOUS-MOST-COMMON",
            "KSR2-FAROOQI-2013-NEJM-DISCOVERY-COHORT",
        ],
        "seed": SEED_BASE + 6,
    },
    # -- SIM1 -- Single-minded homolog 1 (AD) -------------------------------------------
    {
        "gene": "SIM1",
        "alt_name": "SIM1 (Single-Minded Homolog 1 / AD — Prader-Willi-Like Obesity + Hyperphagia + Aggression — PVN-OXT Axis)",
        "protein": (
            "SIM1 -- 6q16.3 AD -- SIM1-786aa -- "
            "Haploinsufficiency-Severe-Hyperphagia-Prader-Willi-Like-Phenotype -- "
            "Hypothalamic-PVN-Development-Critical-SIM1-ARNT-Transcription-Factor -- "
            "Oxytocin-Pathway-Defect-Hypersociality-Hyperphagia-Behavioural-Dysregulation -- "
            "Oxytocin-Trials-PVN-OXT-Axis-Ongoing"
        ),
        "locus": "6q16.3",
        "protein_size": "786 aa",
        "inheritance": "AD (autosomal dominant) — haploinsufficiency; de novo or familial",
        "age_of_onset": (
            "Infancy/early childhood: severe hyperphagia and rapid weight gain; "
            "Prader-Willi-like: hypotonia in infancy + poor feeding → rapid reversal to hyperphagia by age 2-4; "
            "Aggressive behaviour: food-seeking aggression + general behavioural dysregulation — DISTINCTIVE; "
            "Hypersociality: paradoxically increased social drive (unlike autism); "
            "Intellectual disability: mild-moderate in some; "
            "Tall stature: sometimes (as in MC4R); "
            "Puberty: usually normal; "
            "Hypothalamic PVN: SIM1 required for development of oxytocin/AVP-producing neurons; "
            "BMI: 35-70; adiposity truncal; "
            "Chromosome 6q16.3 deletion: overlaps with SIM1; detected by microarray"
        ),
        "key_biomarker": (
            "Chromosomal microarray: 6q16.3 deletion encompasses SIM1 + flanking genes; "
            "SIM1 sequencing: for point variants (microarray negative); "
            "Oxytocin levels: low plasma oxytocin (research biomarker — PVN-OXT neuron loss); "
            "Leptin: elevated (adiposity driven); "
            "Fasting insulin + HOMA-IR: elevated; "
            "Behavioural assessment: hyperphagia questionnaire + aggression scale; "
            "Chromosome 15q: NORMAL (PWS excluded) — key distinction; "
            "FISH/MLPA for 15q11-q13: negative (SIM1-like phenotype, not Prader-Willi); "
            "Prolactin: may be elevated (hypothalamic dysregulation); "
            "LH/FSH: usually normal (puberty preserved)"
        ),
        "pathognomonic": (
            "Prader-Willi-like phenotype + NORMAL chromosome 15q11-q13 methylation = SIM1 first line; "
            "Infantile hypotonia → hyperphagia + severe obesity by age 3-4 + aggressive behaviour; "
            "Aggressive food-seeking behaviour: DISTINCTIVE — more severe aggression than MC4R or LEPR; "
            "DISTINGUISH from Prader-Willi Syndrome: normal 15q11-q13; normal methylation PCR; "
            "DISTINGUISH from Bardet-Biedl: no polydactyly, no retinal dystrophy, no renal anomalies; "
            "Oxytocin pathway: SIM1 drives OXT neuron survival in PVN — absent OXT → hyperphagia + social drive; "
            "6q16.3 deletion: de novo most common; larger deletion → more severe phenotype with additional features; "
            "Mechanism: SIM1/ARNT2 heterodimer activates OXT + AVP gene expression in PVN"
        ),
        "treatment": (
            "No FDA-approved targeted therapy for SIM1-obesity (setmelanotide not approved); "
            "Oxytocin: intranasal OXT trials ongoing — targets lost PVN-OXT output; "
            "Setmelanotide: pilot data showing benefit in some; not yet approved; "
            "GLP-1 receptor agonists: semaglutide/liraglutide — benefit for appetite/weight; "
            "Behavioural therapy: structured ABA + food access restriction + food-seeking behaviour protocols; "
            "Pharmacological: naltrexone + bupropion (Contrave) — some benefit for food-reward pathways; "
            "Methylphenidate: for ADHD-like dysregulation if present; "
            "Bariatric surgery: RYGB effective; benefit sustained; "
            "Food environment: locked food storage; food-structured household protocols (as in PWS management); "
            "Genetic counselling: AD — 50% recurrence if inherited; de novo: low recurrence"
        ),
        "critical_flags": [
            "SIM1-PRADER-WILLI-LIKE-EXCLUDE-PWS-15Q-METHYLATION-FIRST",
            "SIM1-6Q16.3-DELETION-MICROARRAY-MANDATORY",
            "SIM1-AGGRESSIVE-FOOD-SEEKING-BEHAVIOUR-DISTINCTIVE",
            "SIM1-OXYTOCIN-TRIALS-PVN-OXT-AXIS",
            "SIM1-SETMELANOTIDE-NOT-YET-APPROVED-PILOT-DATA-ONLY",
            "SIM1-FOOD-SECURITY-PROTOCOLS-SAME-AS-PWS-MANAGEMENT",
            "SIM1-NORMAL-15Q11-Q13-KEY-DISTINGUISHING-FROM-PWS",
        ],
        "seed": SEED_BASE + 7,
    },
]

# ── COHORT GENERATION ─────────────────────────────────────────────────────────

def _generate_cohort(gene_entry: dict, n: int = 40) -> list:
    rng = random.Random(gene_entry["seed"])
    gene = gene_entry["gene"]
    cohort = []
    for i in range(n):
        age = rng.randint(2, 65)
        sex = rng.choice(["M", "F"])

        # Gene-specific clinical probabilities
        if gene == "MC4R":
            hyperphagia           = rng.random() < 0.95
            severe_obesity        = rng.random() < 0.92
            hyperinsulinaemia     = rng.random() < 0.85
            tall_stature          = rng.random() < 0.55  # childhood tall
            red_hair              = False
            adrenal_insufficiency = False
            hypogonadism          = rng.random() < 0.10  # mild
            neonatal_diarrhoea    = False
            behavioural_issues    = rng.random() < 0.15
            low_heart_rate        = False
            pwl_phenotype         = False
            setmelanotide_eligible= rng.random() < 0.40  # partial function variants
            t2d                   = rng.random() < 0.55
            bariatric_surgery     = rng.random() < 0.25
            low_cortisol          = False
            immune_dysfunction    = False

        elif gene == "LEPR":
            hyperphagia           = rng.random() < 1.00
            severe_obesity        = rng.random() < 0.98
            hyperinsulinaemia     = rng.random() < 0.90
            tall_stature          = rng.random() < 0.10
            red_hair              = False
            adrenal_insufficiency = False
            hypogonadism          = rng.random() < 0.90  # hypogonadotrophic
            neonatal_diarrhoea    = False
            behavioural_issues    = rng.random() < 0.10
            low_heart_rate        = False
            pwl_phenotype         = False
            setmelanotide_eligible= True
            t2d                   = rng.random() < 0.65
            bariatric_surgery     = rng.random() < 0.15
            low_cortisol          = False
            immune_dysfunction    = rng.random() < 0.70

        elif gene == "LEP":
            hyperphagia           = rng.random() < 1.00
            severe_obesity        = rng.random() < 0.97
            hyperinsulinaemia     = rng.random() < 0.90
            tall_stature          = rng.random() < 0.05
            red_hair              = False
            adrenal_insufficiency = False
            hypogonadism          = rng.random() < 0.85
            neonatal_diarrhoea    = False
            behavioural_issues    = rng.random() < 0.05
            low_heart_rate        = False
            pwl_phenotype         = False
            setmelanotide_eligible= False  # metreleptin is correct, not setmelanotide
            t2d                   = rng.random() < 0.60
            bariatric_surgery     = rng.random() < 0.10
            low_cortisol          = False
            immune_dysfunction    = rng.random() < 0.80

        elif gene == "PCSK1":
            hyperphagia           = rng.random() < 0.95
            severe_obesity        = rng.random() < 0.93
            hyperinsulinaemia     = rng.random() < 0.80
            tall_stature          = rng.random() < 0.05
            red_hair              = False
            adrenal_insufficiency = rng.random() < 0.90  # ACTH deficiency
            hypogonadism          = rng.random() < 0.85
            neonatal_diarrhoea    = rng.random() < 0.85
            behavioural_issues    = rng.random() < 0.10
            low_heart_rate        = False
            pwl_phenotype         = False
            setmelanotide_eligible= True
            t2d                   = rng.random() < 0.50
            bariatric_surgery     = rng.random() < 0.15
            low_cortisol          = rng.random() < 0.90
            immune_dysfunction    = rng.random() < 0.10

        elif gene == "POMC":
            hyperphagia           = rng.random() < 0.99
            severe_obesity        = rng.random() < 0.97
            hyperinsulinaemia     = rng.random() < 0.85
            tall_stature          = rng.random() < 0.05
            red_hair              = rng.random() < 0.90  # KEY FEATURE
            adrenal_insufficiency = rng.random() < 0.95  # ACTH deficiency
            hypogonadism          = rng.random() < 0.50
            neonatal_diarrhoea    = False
            behavioural_issues    = rng.random() < 0.10
            low_heart_rate        = False
            pwl_phenotype         = False
            setmelanotide_eligible= True
            t2d                   = rng.random() < 0.55
            bariatric_surgery     = rng.random() < 0.15
            low_cortisol          = rng.random() < 0.95
            immune_dysfunction    = rng.random() < 0.15

        elif gene == "SH2B1":
            hyperphagia           = rng.random() < 0.92
            severe_obesity        = rng.random() < 0.90
            hyperinsulinaemia     = rng.random() < 0.92
            tall_stature          = rng.random() < 0.10
            red_hair              = False
            adrenal_insufficiency = False
            hypogonadism          = rng.random() < 0.10
            neonatal_diarrhoea    = False
            behavioural_issues    = rng.random() < 0.85  # DISTINCTIVE
            low_heart_rate        = False
            pwl_phenotype         = rng.random() < 0.20
            setmelanotide_eligible= False
            t2d                   = rng.random() < 0.60
            bariatric_surgery     = rng.random() < 0.20
            low_cortisol          = False
            immune_dysfunction    = rng.random() < 0.10

        elif gene == "KSR2":
            hyperphagia           = rng.random() < 0.95
            severe_obesity        = rng.random() < 0.93
            hyperinsulinaemia     = rng.random() < 0.95
            tall_stature          = rng.random() < 0.05
            red_hair              = False
            adrenal_insufficiency = False
            hypogonadism          = rng.random() < 0.05
            neonatal_diarrhoea    = False
            behavioural_issues    = rng.random() < 0.10
            low_heart_rate        = rng.random() < 0.85  # DISTINCTIVE
            pwl_phenotype         = False
            setmelanotide_eligible= False
            t2d                   = rng.random() < 0.70
            bariatric_surgery     = rng.random() < 0.15
            low_cortisol          = False
            immune_dysfunction    = False

        else:  # SIM1
            hyperphagia           = rng.random() < 0.97
            severe_obesity        = rng.random() < 0.93
            hyperinsulinaemia     = rng.random() < 0.80
            tall_stature          = rng.random() < 0.35
            red_hair              = False
            adrenal_insufficiency = False
            hypogonadism          = rng.random() < 0.10
            neonatal_diarrhoea    = False
            behavioural_issues    = rng.random() < 0.90  # DISTINCTIVE
            low_heart_rate        = False
            pwl_phenotype         = rng.random() < 0.80  # PWL-like
            setmelanotide_eligible= rng.random() < 0.20  # pilot only
            t2d                   = rng.random() < 0.50
            bariatric_surgery     = rng.random() < 0.20
            low_cortisol          = False
            immune_dysfunction    = False

        cohort.append({
            "patient_id":          f"{gene}-{i+1:03d}",
            "age":                 age,
            "sex":                 sex,
            "gene":                gene,
            "hyperphagia":         hyperphagia,
            "severe_obesity":      severe_obesity,
            "hyperinsulinaemia":   hyperinsulinaemia,
            "tall_stature":        tall_stature,
            "red_hair":            red_hair,
            "adrenal_insufficiency": adrenal_insufficiency,
            "hypogonadism":        hypogonadism,
            "neonatal_diarrhoea":  neonatal_diarrhoea,
            "behavioural_issues":  behavioural_issues,
            "low_heart_rate":      low_heart_rate,
            "pwl_phenotype":       pwl_phenotype,
            "setmelanotide_eligible": setmelanotide_eligible,
            "t2d":                 t2d,
            "bariatric_surgery":   bariatric_surgery,
            "low_cortisol":        low_cortisol,
            "immune_dysfunction":  immune_dysfunction,
        })
    return cohort


def overview() -> dict:
    all_cohorts = [_generate_cohort(g) for g in OB_GENES]
    total = sum(len(c) for c in all_cohorts)
    all_pts = [p for c in all_cohorts for p in c]

    def N(key): return sum(1 for p in all_pts if p[key])

    return {
        "atlas": "Hereditary-Obesity-Melanocortin-Atlas",
        "subtitle": (
            "Complete 8-Gene Leptin-Melanocortin Pathway Atlas: "
            "MC4R (most common), LEPR, LEP, PCSK1, POMC (setmelanotide-eligible) + "
            "SH2B1 (16p11.2), KSR2 (metformin-responsive), SIM1 (PVN-OXT)"
        ),
        "genes": [g["gene"] for g in OB_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "hyperphagia_patients":           N("hyperphagia"),
        "severe_obesity_patients":        N("severe_obesity"),
        "hyperinsulinaemia_patients":     N("hyperinsulinaemia"),
        "red_hair_patients":              N("red_hair"),
        "adrenal_insufficiency_patients": N("adrenal_insufficiency"),
        "hypogonadism_patients":          N("hypogonadism"),
        "neonatal_diarrhoea_patients":    N("neonatal_diarrhoea"),
        "behavioural_issues_patients":    N("behavioural_issues"),
        "low_heart_rate_patients":        N("low_heart_rate"),
        "pwl_phenotype_patients":         N("pwl_phenotype"),
        "setmelanotide_eligible_patients":N("setmelanotide_eligible"),
        "t2d_patients":                   N("t2d"),
        "immune_dysfunction_patients":    N("immune_dysfunction"),
        "gene_patient_counts": {g["gene"]: len(_generate_cohort(g)) for g in OB_GENES},
        "pathway": (
            "Leptin → LEPR (JAK2-STAT3) → POMC neurons in ARC → α-MSH → MC4R in PVN → "
            "anorexigenic signalling → satiety; "
            "PCSK1 cleaves POMC → ACTH + α-MSH; "
            "SH2B1 amplifies leptin-JAK2 and insulin receptor signals; "
            "KSR2 scaffolds AMPK → energy sensing + FAO; "
            "SIM1/ARNT2 drives OXT/AVP PVN neurons (MC4R-OXT-PVN circuit); "
            "MC4R is the convergence point — downstream of LEPR, LEP, POMC, PCSK1, SIM1."
        ),
        "key_clinical_insight": (
            "MC4R: most common (1-2% severe obesity); hyperinsulinaemia + tall stature; setmelanotide partial benefit. "
            "LEPR: very high leptin (paradox); setmelanotide FDA-2020; hypogonadism + immune dysfunction; NOT metreleptin. "
            "LEP: UNDETECTABLE leptin; metreleptin CURATIVE; NOT setmelanotide; normal cortisol + hair. "
            "PCSK1: neonatal diarrhoea + panhypopituitarism; setmelanotide FDA-2021; ADRENAL CRISIS RISK. "
            "POMC: RED HAIR + obesity + hypocortisolism TRIAD; setmelanotide FDA-2020; ADRENAL CRISIS MANDATORY cortisol. "
            "SH2B1: 16p11.2 deletion; behavioural dysregulation + severe insulin resistance; microarray mandatory. "
            "KSR2: low resting HR + severe insulin resistance; metformin DRAMATICALLY effective — UNIQUE. "
            "SIM1: PWS-like phenotype; normal 15q11-q13; aggressive food-seeking; OXT trials ongoing."
        ),
    }


def breakdown() -> dict:
    result = {}
    for gene_entry in OB_GENES:
        cohort = _generate_cohort(gene_entry)
        gene = gene_entry["gene"]

        def pct(key):
            return round(100 * sum(1 for p in cohort if p[key]) / len(cohort))

        result[gene] = {
            "gene": gene,
            "alt_name": gene_entry["alt_name"],
            "locus": gene_entry["locus"],
            "protein_size": gene_entry["protein_size"],
            "inheritance": gene_entry["inheritance"],
            "n_patients": len(cohort),
            "hyperphagia_pct":           pct("hyperphagia"),
            "severe_obesity_pct":        pct("severe_obesity"),
            "hyperinsulinaemia_pct":     pct("hyperinsulinaemia"),
            "tall_stature_pct":          pct("tall_stature"),
            "red_hair_pct":              pct("red_hair"),
            "adrenal_insufficiency_pct": pct("adrenal_insufficiency"),
            "hypogonadism_pct":          pct("hypogonadism"),
            "neonatal_diarrhoea_pct":    pct("neonatal_diarrhoea"),
            "behavioural_issues_pct":    pct("behavioural_issues"),
            "low_heart_rate_pct":        pct("low_heart_rate"),
            "pwl_phenotype_pct":         pct("pwl_phenotype"),
            "setmelanotide_eligible_pct":pct("setmelanotide_eligible"),
            "t2d_pct":                   pct("t2d"),
            "immune_dysfunction_pct":    pct("immune_dysfunction"),
            "low_cortisol_pct":          pct("low_cortisol"),
            "age_of_onset":   gene_entry["age_of_onset"],
            "key_biomarker":  gene_entry["key_biomarker"],
            "pathognomonic":  gene_entry["pathognomonic"],
            "treatment":      gene_entry["treatment"],
            "critical_flags": gene_entry["critical_flags"],
            "seed":           gene_entry["seed"],
            "cohort_preview": cohort[:5],
        }
    return result


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Obesity-Melanocortin-Atlas",
        "pathway": "Leptin-Melanocortin-MC4R Axis / Hypothalamic Energy Balance",
        "shared_mechanism": (
            "The leptin-melanocortin pathway is the central hypothalamic circuit controlling food intake and energy balance: "
            "adipose tissue secretes leptin → LEPR on arcuate nucleus (ARC) POMC neurons → "
            "JAK2-STAT3 signalling → POMC gene expression → PC1/3 (PCSK1) cleaves POMC → α-MSH + β-endorphin + ACTH → "
            "α-MSH binds MC4R in paraventricular nucleus (PVN) → anorexigenic signalling → satiety. "
            "Loss of any node (LEP, LEPR, PCSK1, POMC, MC4R) → unopposed AgRP/NPY orexigenic drive → hyperphagia + obesity. "
            "SH2B1 amplifies the LEPR-JAK2 signal; KSR2 links AMPK energy sensing to this axis; "
            "SIM1 drives OXT/AVP PVN neurons that relay MC4R signals downstream."
        ),
        "genes": {
            g["gene"]: {
                "full_name": g["alt_name"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "critical_flags": g["critical_flags"],
                "pathognomonic": g["pathognomonic"],
                "treatment_summary": g["treatment"],
            }
            for g in OB_GENES
        },
        "glossary": {
            "MC4R (melanocortin 4 receptor)": "G-protein coupled receptor in PVN; activated by α-MSH (from POMC) → anorexigenic signalling; most common monogenic obesity gene (1-2% severe obesity)",
            "LEPR (leptin receptor)": "Long-form ObRb on ARC POMC neurons; JAK2-STAT3 intracellular signalling; loss → leptin resistance phenotype with VERY HIGH serum leptin",
            "LEP (leptin)": "Adipokine from white adipose; proportional to fat mass; signals satiety via LEPR; absent = UNDETECTABLE serum leptin; metreleptin (recombinant) replaces it",
            "PCSK1 (proprotein convertase 1/3)": "Serine endoprotease in regulated secretory pathway; cleaves POMC → ACTH + α-MSH; loss = panhypopituitarism + neonatal malabsorptive diarrhoea (proglucagon not cleaved)",
            "POMC (proopiomelanocortin)": "Precursor polypeptide: PC1/3 cleaves → ACTH (adrenal cortisol drive) + α-MSH (MC4R satiety) + β-endorphin; absent → obesity + red hair + hypocortisolism TRIAD",
            "Setmelanotide (Imcivree)": "MC4R agonist; Rhythm Pharmaceuticals; FDA approved 2020 (POMC, LEPR), 2021 (PCSK1), 2022 (BBS); bypasses upstream pathway defects; NOT for LEP or KSR2",
            "Metreleptin (Myalept)": "Recombinant methionyl-leptin; FDA approved for GENERALISED lipodystrophy; used off-label for congenital LEP deficiency (CURATIVE); useless in LEPR (receptor absent)",
            "α-MSH (alpha-melanocyte stimulating hormone)": "POMC-derived peptide: activates MC1R (melanin/pigmentation) + MC4R (satiety) + MC3R; absence → obesity (MC4R) + red/pale hair (MC1R)",
            "ACTH (adrenocorticotrophic hormone)": "POMC-derived peptide: stimulates adrenal cortex to produce cortisol; ACTH deficiency (POMC, PCSK1) → adrenal insufficiency → adrenal crisis",
            "β-endorphin": "POMC-derived opioid peptide; absent in POMC deficiency; contributes to pain modulation and mood",
            "AgRP/NPY neurons": "ARC orexigenic neurons; antagonise POMC/α-MSH; when LEPR/MC4R signalling fails, AgRP/NPY unopposed → hyperphagia",
            "Paraventricular nucleus (PVN)": "Hypothalamic nucleus; site of MC4R expression; SIM1 drives OXT/AVP neurons here; integrates energy signals",
            "Arcuate nucleus (ARC)": "Hypothalamic nucleus at base of 3rd ventricle; POMC neurons (anorexigenic) + AgRP/NPY neurons (orexigenic) — key energy balance centre",
            "SH2B1": "Adaptor protein; amplifies JAK2 signal downstream of both LEPR and insulin receptor; 16p11.2 chromosomal locus — deletion causes autism-obesity syndrome",
            "KSR2": "Kinase suppressor of Ras 2; scaffolds AMPK catalytic subunit; regulates cellular energy sensing + FAO; loss → reduced resting HR + reduced FAO + severe insulin resistance; metformin bypasses KSR2 via LKB1-AMPK",
            "SIM1/ARNT2": "bHLH-PAS transcription factor heterodimer; drives OXT + AVP gene expression in PVN neurons; haploinsufficiency → reduced PVN neuron number → Prader-Willi-like phenotype",
            "Oxytocin (OXT)": "PVN neuropeptide; relays MC4R anorexigenic signal; reduced in SIM1 deficiency; intranasal OXT trials underway for SIM1-obesity",
            "16p11.2 deletion syndrome": "500kb chromosomal deletion including SH2B1 + MAPK3 + other genes; causes obesity + autism spectrum disorder + intellectual disability; microarray diagnostic",
            "Hypocortisolism / adrenal insufficiency": "Cortisol deficiency due to absent ACTH drive (POMC or PCSK1 deficiency); adrenal gland intact but not stimulated; Synacthen test blunted",
            "Adrenal crisis": "Acute cortisol deficiency during physiological stress (infection, surgery, vomiting); cardiovascular collapse; prevention: stress dosing hydrocortisone + emergency card",
            "Prader-Willi-like phenotype": "Infantile hypotonia → hyperphagia + obesity + behavioural dysregulation — but NORMAL chromosome 15q11-q13; caused by SIM1 haploinsufficiency or other PVN defects",
            "Proinsulin:insulin ratio": "Elevated in PCSK1 deficiency — proinsulin not cleaved to insulin; diagnostic fingerprint; normal in all other monogenic obesity genes",
            "Hyperphagia": "Pathological hunger — Dykens Hyperphagia Questionnaire (DHQ) used in clinical trials; score >11 = severe; core feature of ALL 8 genes in this atlas",
            "BMI centile": "Severe obesity = BMI >35 in adults; >99.6th centile in children; monogenic obesity typically causes BMI 40-80 in untreated adults",
            "HOMA-IR": "Homeostatic Model Assessment of Insulin Resistance = fasting insulin × fasting glucose / 22.5; elevated in MC4R, LEPR, LEP, SH2B1, KSR2",
        },
        "surveillance_protocols": {
            "MC4R": "Annual: HbA1c, fasting insulin, lipids, BP; setmelanotide functional variant assessment before prescribing; GLP-1RA trial if setmelanotide not indicated",
            "LEPR": "Annual: leptin, LH/FSH, IGF-1, GH stim (paeds), free T4; setmelanotide monitoring (weight, hyperphagia VAS); immune function if recurrent infections",
            "LEP": "Annual: serum leptin (confirm <0.5 on treatment); metreleptin dose adjustment; LH/FSH (puberty monitoring); immune function; HbA1c",
            "PCSK1": "Annual: cortisol (9am), ACTH stim, free T4+TSH, IGF-1, LH/FSH; emergency cortisol card; stress dosing protocol; setmelanotide monitoring",
            "POMC": "Annual: cortisol (9am), ACTH stim, free T4, prolactin; skin check (sunburn risk); setmelanotide monitoring; stress dosing mandatory; adrenal crisis plan",
            "SH2B1": "Annual: HbA1c, fasting insulin, lipids, BP; behavioural/psychiatric review; microarray if not done; cardiometabolic screen",
            "KSR2": "Annual: HbA1c, fasting insulin, resting HR, ECG; metformin dose optimisation; indirect calorimetry; lipids",
            "SIM1": "Annual: hyperphagia score; weight + BMI; behavioural assessment; 15q methylation (if not done); oxytocin levels (research); food security review",
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Hyperphagia pts: {ov['hyperphagia_patients']}")
    print(f"Red hair pts: {ov['red_hair_patients']}")
    print(f"Adrenal insufficiency pts: {ov['adrenal_insufficiency_patients']}")
    print(f"Setmelanotide eligible: {ov['setmelanotide_eligible_patients']}")
    print(f"T2D pts: {ov['t2d_patients']}")
    print("Breakdown gene keys:", list(breakdown().keys()))
