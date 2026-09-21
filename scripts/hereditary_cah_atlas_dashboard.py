#!/usr/bin/env python3
"""Hereditary-CAH-Atlas — Complete 8-Gene Congenital Adrenal Hyperplasia Steroidogenesis Atlas
CYP21A2 (21-hydroxylase; 495 aa; 6p21.33; AR;
         Most common CAH ~95%; 17-OHP accumulates; classic salt-wasting/simple-virilising/non-classic;
         NBS heel-prick 17-OHP; CYP21A1P pseudogene gene-conversion mechanism; fludrocortisone SW;
         seed SEED_BASE+0) ·
CYP11B1 (11β-hydroxylase; 503 aa; 8q24.3; AR;
         2nd most common CAH ~5%; 11-DOC + 11-deoxycortisol accumulate; HYPERTENSION + hypokalemia;
         NO fludrocortisone — DOC acts as mineralocorticoid; compound-S elevated PATHOGNOMONIC;
         seed SEED_BASE+1) ·
HSD3B2  (3β-HSD type 2; 372 aa; 1p12; AR;
         Rare; all three steroidogenic zones affected; DHEA accumulates (delta-5);
         PARADOX: 46XX mildly virilised; 46XY undervirilised; elevated delta-5:delta-4 ratio;
         seed SEED_BASE+2) ·
CYP17A1 (17α-hydroxylase/17,20-lyase; 508 aa; 10q24.32; AR;
         Rare; absent cortisol AND absent sex steroids; DOC/corticosterone excess → HTN;
         46XY complete sex reversal (female external phenotype); absent puberty both sexes;
         seed SEED_BASE+3) ·
STAR    (StAR steroidogenic acute regulatory protein; 285 aa; 8p11.23; AR;
         Lipoid CAH — most severe; ALL steroids absent; CT/MRI enlarged lipid-laden adrenals PATHOGNOMONIC;
         46XY female external phenotype; neonatal adrenal crisis; fludrocortisone + HC lifelong;
         seed SEED_BASE+4) ·
CYP11A1 (P450scc cholesterol side-chain cleavage; 521 aa; 15q24.1; AR;
         Similar to StAR — first enzymatic step; complete LOF = lipoid-like; partial = late-onset;
         molecular panel distinguishes StAR vs CYP11A1; variable severity;
         seed SEED_BASE+5) ·
POR     (P450 oxidoreductase; 680 aa; 7q11.23; AR;
         Electron donor for ALL microsomal CYPs (CYP21A2, CYP17A1, CYP19A1);
         Antley-Bixler craniosynostosis + radiohumeral synostosis PATHOGNOMONIC; maternal virilisation;
         combined enzymatic block; mixed steroid profile;
         seed SEED_BASE+6) ·
CYP11B2 (Aldosterone synthase; 503 aa; 8q24.3; AR;
         Isolated aldosterone deficiency (CMO I/II); pure salt-wasting NO virilisation NO HTN;
         normal cortisol + 17-OHP; 18-OH-corticosterone elevated (CMO II); fludrocortisone ONLY;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 2942–2949)
"""
import random

SEED_BASE = 2942

ATLAS_GENES = [
    {
        "gene": "CYP21A2",
        "protein": (
            "CYP21A2 -- 6p21.33 AR -- 495aa -- 21-Hydroxylase-"
            "55kDa-Most-Common-CAH-95pct-17OHP-Elevated-PATHOGNOMONIC-"
            "Classic-Salt-Wasting-Simple-Virilising-Non-Classic-"
            "CYP21A1P-Pseudogene-Gene-Conversion-Mechanism-"
            "OMIM-Gene-613815-Disease-OMIM-201910"
        ),
        "locus": "6p21.33",
        "protein_size": (
            "495 aa / 55 kDa (CYP21A2 — cytochrome P450 family 21 subfamily A member 2; "
            "microsomal 21-hydroxylase; MHC class III region 6p21.33; "
            "FUNCTION: converts 17-OHP → 11-deoxycortisol (cortisol pathway) + "
            "  progesterone → 11-deoxycorticosterone (aldosterone pathway); "
            "  Rate-limiting step for both cortisol and aldosterone biosynthesis; "
            "LOF CONSEQUENCE: "
            "  17-OHP accumulates (measurable by NBS heel-prick and serum); "
            "  Cortisol deficiency → ACTH rises markedly (CRH-ACTH feedback uninhibited); "
            "  ACTH drives adrenal hyperplasia → DHEA/androstenedione excess → peripheral testosterone; "
            "  Mineralocorticoid deficiency (SW form): profound Na+ loss, K+ retention, hypovolaemic shock; "
            "GENE ARCHITECTURE: "
            "  CYP21A2 (active) lies adjacent to CYP21A1P (pseudogene) in tandem repeat with TNXA/TNXB; "
            "  80% of mutations = gene conversion (pseudogene sequence replaces active gene) or deletion; "
            "  20% = point mutations; standard sequencing may miss large deletions → MLPA mandatory; "
            "encoded 6p21.33 (HLA-B region)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) biallelic — CYP21A2 CAH: "
            "  PHENOTYPIC SPECTRUM (correlates with residual enzyme activity): "
            "  1. Classic salt-wasting (SW): <1% enzyme activity; profound Na+ loss; "
            "     Neonatal adrenal crisis (hyponatremia + hyperkalemia + hypoglycaemia + shock); "
            "     46XX: ambiguous genitalia at birth (Prader 2-5 clitoromegaly/labioscrotal fusion); "
            "     46XY: normal genitalia but adrenal crisis; "
            "  2. Simple virilising (SV): 1-2% activity; cortisol-deficient, aldosterone borderline; "
            "     46XX: virilisation ± late presentation; "
            "  3. Non-classic (NC): 20-50% activity; mild androgen excess only (PCOS-like); "
            "     females: hirsuitism, acne, irregular menses, subfertility; often diagnosed in adulthood; "
            "DIAGNOSIS: "
            "    Classic: serum 17-OHP >100 nmol/L (30 ng/mL) random; "
            "    Non-classic: post-Synacthen 60-min 17-OHP >30 nmol/L (>10 ng/mL); "
            "    NBS: 17-OHP heel-prick (false-positives in premature infants — repeat required); "
            "    Genotype: MLPA (gene dosage) + sequencing (gene conversion not seen by exon-seq alone); "
            "TREATMENT: "
            "    HC (hydrocortisone) 10-15 mg/m²/day divided 3×/day (cortisol replacement); "
            "    Fludrocortisone (SW and SV): 50-200 μg/day + NaCl supplement in infancy; "
            "    Stress dosing (illness/surgery): 3× hydrocortisone ('sick day rules'); "
            "    Adrenal crisis kit (IM hydrocortisone 50-100 mg) for all classic patients; "
            "    46XX virilised genitalia: genital surgery (parental choice + multidisciplinary team)"
        ),
        "disease_category": (
            "CLASSIC 21-HYDROXYLASE CAH — SALT-WASTING / SIMPLE VIRILISING / NON-CLASSIC: "
            "  MOST IMPORTANT CAH GENE — ~95% of all CAH cases globally; "
            "  NEONATAL EMERGENCY: SW form → life-threatening crisis day 7-14 of life; "
            "    Males often missed on NBS (no external genitalia abnormality); "
            "    Females: ambiguous genitalia → earlier diagnosis; "
            "  17-OHP PATHWAY CONCEPT: "
            "    ACTH → cholesterol → pregnenolone → 17-OHP → 11-deoxycortisol → cortisol (BLOCKED); "
            "    Excess 17-OHP → diverted via 21-hydroxylase-independent pathways → "
            "      androgens (androstenedione → testosterone); "
            "  GENE CONVERSION PITFALL: "
            "    Most labs report 'no pathogenic variant found' on standard sequencing in carriers; "
            "    MLPA (multiplex ligation-dependent probe amplification) mandatory to detect deletions; "
            "  TREATMENT MONITORING: "
            "    17-OHP poorly predicts over-treatment (falsely high in AM, falsely low if HC taken before blood); "
            "    Androstenedione + renin: better treatment markers; "
            "    Over-treatment → Cushing + growth failure; under-treatment → accelerated bone age"
        ),
        "disease_pathway": (
            "CYP21A2 LOF → 17-OHP ACCUMULATION → ACTH-DRIVEN ADRENAL ANDROGEN EXCESS: "
            "  Normal adrenal steroidogenesis: "
            "    Zona fasciculata: progesterone → 17-OHP → 11-deoxycortisol → cortisol; "
            "    Zona glomerulosa: progesterone → DOC → corticosterone → 18-OHB → aldosterone; "
            "  CYP21A2 block: "
            "    17-OHP cannot proceed → accumulates (adrenal + serum + amniotic fluid); "
            "    Cortisol falls → ACTH rises (loss of negative feedback); "
            "    ACTH hyperdrives adrenal → adrenal hyperplasia + excess 17-OHP; "
            "    Excess 17-OHP → 17,20-lyase (CYP17A1) + HSD3B2 → androstenedione; "
            "    Androstenedione → testosterone (peripheral 17β-HSD1) → virilisation; "
            "  Salt-wasting: aldosterone pathway also blocked (no 11-DOC) → Na+ loss; "
            "  Treatment rationale: HC → replaces cortisol → suppresses ACTH → reduces 17-OHP + androgens; "
            "    Fludrocortisone → replaces aldosterone → prevents salt-wasting crisis"
        ),
    },
    {
        "gene": "CYP11B1",
        "protein": (
            "CYP11B1 -- 8q24.3 AR -- 503aa -- 11beta-Hydroxylase-"
            "56kDa-2nd-Most-Common-CAH-5pct-HYPERTENSION-Hypokalemia-"
            "11-Deoxycortisol-Compound-S-PATHOGNOMONIC-"
            "NO-Fludrocortisone-DOC-Mineralocorticoid-"
            "OMIM-Gene-610613-Disease-OMIM-202010"
        ),
        "locus": "8q24.3",
        "protein_size": (
            "503 aa / 56 kDa (CYP11B1 — cytochrome P450 family 11 subfamily B member 1; "
            "mitochondrial 11β-hydroxylase; zona fasciculata; "
            "FUNCTION: 11-deoxycortisol → cortisol (final step in cortisol pathway); "
            "  also: 11-DOC → corticosterone (mineralocorticoid precursor); "
            "  NOTE: CYP11B2 (aldosterone synthase) and CYP11B1 are adjacent paralogues at 8q24.3 (96% identity); "
            "LOF CONSEQUENCE: "
            "  11-deoxycortisol (compound S) accumulates → PATHOGNOMONIC marker; "
            "  11-deoxycorticosterone (DOC) accumulates → acts as weak mineralocorticoid → "
            "    Na+ retention + K+ wasting + HTN + aldosterone suppressed (feedback); "
            "  VIRILISATION: same as CYP21A2 (ACTH-driven androgen excess); "
            "  HYPERTENSION — KEY DISTINGUISHER from CYP21A2 (SW form has LOW BP); "
            "LABORATORY: "
            "  11-deoxycortisol (compound S) elevated (>10× normal); "
            "  DOC elevated; corticosterone elevated; "
            "  ACTH elevated; cortisol low; "
            "  Aldosterone low/suppressed (DOC-mediated volume expansion suppresses RAAS); "
            "encoded 8q24.3 (adjacent to CYP11B2)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — CYP11B1 CAH: "
            "  Missense mutations: most common; R448H most frequent in North African/Middle Eastern populations; "
            "  Founder effect: Moroccan Jews (p.Arg448His ~90% of alleles); "
            "CLINICAL FEATURES: "
            "    Cortisol deficiency → ACTH elevated → adrenal hyperplasia + androgen excess; "
            "    46XX: ambiguous genitalia (virilisation similar to CYP21A2); "
            "    46XY: normal genitalia but HTN + hypokalaemia; "
            "    HYPERTENSION: present in ~2/3 of patients; can be severe (LVH, stroke risk); "
            "    Hypokalemia: from DOC mineralocorticoid effect; "
            "    NO SALT-WASTING (contrast CYP21A2 SW): DOC preserves Na+ → NO neonatal crisis in males; "
            "    Females: ambiguous genitalia may still trigger early diagnosis; "
            "TREATMENT: "
            "    Hydrocortisone: replaces cortisol → suppresses ACTH → DOC falls → HTN resolves; "
            "    NO fludrocortisone (DOC excess, not deficiency); "
            "    Antihypertensives: occasionally needed during initial management; "
            "    HTN monitoring: BP reverses over weeks-months with HC suppression; "
            "    Stress dosing: same as CYP21A2 (adrenal crisis possible with cortisol deficiency); "
            "DIAGNOSTIC PITFALL: "
            "    Males missed on NBS (normal 17-OHP — different pathway block); "
            "    Present in childhood with HTN + hypokalaemia + short stature (advanced bone age)"
        ),
        "disease_category": (
            "CYP11B1 CAH — HYPERTENSIVE VIRILISING CAH — DOC MINERALOCORTICOID EXCESS: "
            "  FUNDAMENTAL DIFFERENCE FROM CYP21A2: "
            "    CYP21A2: salt-wasting (mineralocorticoid + glucocorticoid deficient); "
            "    CYP11B1: hypertension (DOC acts as mineralocorticoid, no deficit in aldosterone pathway); "
            "  DOC PARADOX: "
            "    CYP11B1 block → DOC cannot become corticosterone; "
            "    DOC accumulates → binds MR (weaker than aldosterone) → Na+ retention; "
            "    Volume expansion → renin ↓ → aldosterone ↓; "
            "  TREATMENT REVERSAL OF HTN: "
            "    HC suppresses ACTH → steroidogenesis reduced → DOC falls → Na+ retention reverses; "
            "    BP normalises over weeks; "
            "  FOUNDER MUTATION CONTEXT: "
            "    R448H (CYP11B1): most common variant in North African/Middle Eastern ancestry; "
            "    Consider CYP11B1 in patients of Moroccan/Sephardic Jewish ancestry with virilising HTN"
        ),
        "disease_pathway": (
            "CYP11B1 LOF → COMPOUND S + DOC ACCUMULATION → HTN VIRILISING CAH: "
            "  Steroidogenesis pathway: "
            "    17-OHP → CYP21A2 → 11-deoxycortisol (compound S) → CYP11B1 → cortisol (BLOCKED); "
            "    Progesterone → CYP21A2 → 11-DOC → CYP11B1 → corticosterone (BLOCKED); "
            "  CYP11B1 block: "
            "    11-deoxycortisol accumulates → NBS 17-OHP: may be mildly elevated (shared pathway); "
            "    Cortisol falls → ACTH rises → adrenal hyperplasia; "
            "    11-DOC accumulates → MR binding (weak) → Na+ retention → volume ↑; "
            "    Volume expansion → renin-angiotensin suppressed → aldosterone low; "
            "    ACTH excess → androstenedione + testosterone overproduction → virilisation; "
            "  Therapy: "
            "    HC: cortisol replacement → ACTH falls → all upstream intermediates (DOC, compound S) fall; "
            "    HTN resolves within weeks as DOC cleared"
        ),
    },
    {
        "gene": "HSD3B2",
        "protein": (
            "HSD3B2 -- 1p12 AR -- 372aa -- 3beta-Hydroxysteroid-Dehydrogenase-Type-2-"
            "42kDa-Rare-CAH-All-Three-Zones-Affected-DHEA-Accumulates-Delta5-"
            "PARADOX-46XX-Virilised-46XY-Undervirilised-"
            "Elevated-Delta5-Delta4-Ratio-"
            "OMIM-Gene-109715-Disease-OMIM-201810"
        ),
        "locus": "1p12",
        "protein_size": (
            "372 aa / 42 kDa (HSD3B2 — hydroxy-delta-5-steroid dehydrogenase 3 beta and steroid delta-isomerase 2; "
            "mitochondrial + ER enzyme; adrenal + gonad specific (type 2); "
            "FUNCTION: "
            "  Converts all delta-5 steroids → delta-4 steroids: "
            "    Pregnenolone → progesterone (zona glomerulosa → aldosterone pathway); "
            "    17-OH-pregnenolone → 17-OHP (zona fasciculata → cortisol pathway); "
            "    DHEA → androstenedione (zona reticularis + gonad → sex steroid pathway); "
            "  Required for glucocorticoid, mineralocorticoid, AND sex steroid biosynthesis; "
            "LOF CONSEQUENCE: "
            "  All delta-5 steroids accumulate; delta-4 steroids reduced; "
            "  DHEA accumulates markedly (C19 steroid, weak androgen); "
            "  Cortisol and aldosterone both deficient (salt-wasting + cortisol deficiency); "
            "  Paradox: insufficient potent androgens (no androstenedione/testosterone) → "
            "    46XY: undervirilised (incomplete masculinisation) at birth; "
            "    46XX: DHEA (weak androgen) → mild peripheral virilisation via HSD3B1 (extra-adrenal); "
            "encoded 1p12 (note: HSD3B1 = type 1 isoform, different gene, placenta + peripheral; "
            "  only type 2 expressed in adrenal/gonad; HSD3B1 partially compensates peripherally)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — HSD3B2 CAH: "
            "  Missense or frameshift mutations in HSD3B2 gene; "
            "  Spectrum: complete LOF (severe) to partial LOF (mild, rare non-classic form); "
            "CLINICAL FEATURES: "
            "  SALT-WASTING (like CYP21A2 SW): both aldosterone and cortisol deficient; "
            "    Neonatal salt-wasting crisis; "
            "  PARADOXICAL VIRILISATION: "
            "    46XX: mildly virilised genitalia (ambiguous) despite being a female — "
            "      DHEA → HSD3B1 (extraadrenal) → androstenedione → testosterone peripherally; "
            "      Prader 1-2 clitoromegaly; less severe than CYP21A2; "
            "    46XY: undervirilised genitalia (ambiguous, hypospadias, cryptorchidism) — "
            "      insufficient testosterone (no androstenedione from adrenal); "
            "      Testicular HSD3B2 also absent → no testosterone during virilisation window; "
            "DIAGNOSIS: "
            "    NBS: 17-OHP may be ELEVATED (17-OH-pregnenolone → 17-OHP via extra-adrenal HSD3B1); "
            "    Serum: DHEA/DHEAS elevated; delta-5:delta-4 ratio elevated; "
            "    17-OH-pregnenolone markedly elevated (most specific); "
            "    Urine steroid profile: delta-5 steroids dominant; "
            "TREATMENT: "
            "    HC + fludrocortisone (same as CYP21A2 SW); "
            "    Sex hormone replacement at puberty; "
            "    DSD team for genital ambiguity (46XX and 46XY both ambiguous at birth)"
        ),
        "disease_category": (
            "HSD3B2 CAH — ALL-ZONE DEFECT — DHEA ACCUMULATION — PARADOXICAL VIRILISATION/UNDERVIRILISATION: "
            "  UNIQUE CLINICAL PARADOX: "
            "    46XX: mild virilisation (DHEA → peripheral HSD3B1) despite deficient potent androgens; "
            "    46XY: undervirilisation (no testosterone → incomplete masculinisation); "
            "    BOTH: ambiguous genitalia at birth — DSD evaluation mandatory; "
            "  DIAGNOSTIC CLUE: "
            "    Delta-5 steroid accumulation (DHEA + 17-OH-pregnenolone) — distinct from CYP21A2; "
            "    17-OHP may be elevated (NBS) → potential confusion with CYP21A2 → steroid profile needed; "
            "  EXTRA-ADRENAL RESCUE: "
            "    HSD3B1 (type 1, expressed peripherally) partially converts DHEA → androstenedione; "
            "    Explains mild female virilisation despite HSD3B2 absence; "
            "    Does NOT rescue glucocorticoid/mineralocorticoid synthesis (adrenal-only function)"
        ),
        "disease_pathway": (
            "HSD3B2 LOF → DELTA-5 STEROID ACCUMULATION → SALT-WASTING + PARADOXICAL DSD: "
            "  Normal pathway: "
            "    Cholesterol → pregnenolone (StAR/CYP11A1) → progesterone (HSD3B2) → DOC → corticosterone → aldosterone; "
            "    17-OH-pregnenolone (HSD3B2) → 17-OHP → 11-deoxycortisol → cortisol; "
            "    17-OH-pregnenolone (HSD3B2 + CYP17A1-lyase) → DHEA → androstenedione (HSD3B2); "
            "  HSD3B2 block: "
            "    All delta-5→delta-4 conversions FAIL in adrenal and gonad; "
            "    Pregnenolone, 17-OH-pregnenolone, DHEA accumulate; "
            "    Cortisol absent → ACTH rises → more DHEA (marked elevation); "
            "    Aldosterone absent → Na+ loss → salt-wasting crisis; "
            "  Peripheral rescue: "
            "    DHEA → HSD3B1 → androstenedione → 17β-HSD → testosterone (PERIPHERAL); "
            "    This weak peripheral conversion causes 46XX mild virilisation; "
            "    46XY testes: no HSD3B2 → no testicular testosterone → undervirilised masculinisation"
        ),
    },
    {
        "gene": "CYP17A1",
        "protein": (
            "CYP17A1 -- 10q24.32 AR -- 508aa -- 17alpha-Hydroxylase-17-20-Lyase-"
            "57kDa-HYPERTENSION-Absent-Puberty-Both-Sexes-"
            "46XY-Female-External-Phenotype-Sex-Reversal-"
            "DOC-Corticosterone-Excess-Aldosterone-Suppressed-"
            "OMIM-Gene-609300-Disease-OMIM-202110"
        ),
        "locus": "10q24.32",
        "protein_size": (
            "508 aa / 57 kDa (CYP17A1 — cytochrome P450 family 17 subfamily A member 1; "
            "ER-bound; bifunctional enzyme: 17α-hydroxylase activity + 17,20-lyase activity; "
            "FUNCTION: "
            "  17α-HYDROXYLASE: pregnenolone → 17-OH-pregnenolone; progesterone → 17-OHP; "
            "    Required for cortisol (zona fasciculata) and sex steroids (zona reticularis + gonad); "
            "  17,20-LYASE: 17-OH-pregnenolone → DHEA; 17-OHP → androstenedione; "
            "    Required for ALL sex steroids (androgens + estrogens); "
            "  NOT EXPRESSED in zona glomerulosa (aldosterone pathway does not use CYP17A1); "
            "LOF CONSEQUENCE: "
            "  Cortisol absent (17α-OH activity blocked) → ACTH elevated; "
            "  Sex steroids absent (17,20-lyase blocked) — affects both adrenal AND gonads; "
            "  ACTH-driven: progesterone/DOC/corticosterone accumulate in zona fasciculata → "
            "    DOC + corticosterone → mineralocorticoid activity → Na+ retention → HTN; "
            "    Aldosterone SUPPRESSED (renin suppressed by DOC-mediated volume expansion); "
            "encoded 10q24.32"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — CYP17A1 CAH: "
            "  Missense, frameshift, or combined 17α-OH + 17,20-lyase deficiency; "
            "  Rare: isolated 17,20-lyase deficiency (POR mutations can mimic); "
            "CLINICAL FEATURES — HALLMARK TRIAD: "
            "  1. HYPERTENSION + HYPOKALEMIA: DOC/corticosterone excess → volume expansion; "
            "     HTN can be severe; normal or suppressed aldosterone (paradox); "
            "  2. ABSENT PUBERTY IN BOTH SEXES: no sex steroids from adrenal OR gonad; "
            "     46XX: streak ovaries → no estrogen → no breast development; primary amenorrhoea; "
            "     46XY: testes present (usually intra-abdominal) → no testosterone → "
            "       female external genitalia (complete sex reversal); cryptorchidism; "
            "  3. CORTISOL DEFICIENCY: compensated by corticosterone (weak GC) in some; "
            "     Adrenal crisis less common than CYP21A2 (corticosterone partially compensates); "
            "DIAGNOSIS: "
            "    Low cortisol; elevated ACTH; elevated DOC + corticosterone + progesterone; "
            "    Low/undetectable: DHEA, androstenedione, testosterone, estradiol; "
            "    Aldosterone: low (RAAS suppressed); renin: low; "
            "    46XY: karyotype shows XY with female phenotype → critical clue; "
            "TREATMENT: "
            "    HC (replaces cortisol, suppresses ACTH → DOC falls → HTN resolves); "
            "    Sex steroid replacement: "
            "      46XX: ethinyl estradiol for feminisation + menstrual cycle induction; "
            "      46XY: gonadectomy (gonadal tumour risk) then estrogen; "
            "    NO fludrocortisone (DOC excess is the problem — not deficiency)"
        ),
        "disease_category": (
            "CYP17A1 CAH — HYPERTENSION + ABSENT PUBERTY + 46XY COMPLETE SEX REVERSAL: "
            "  UNIQUE PRESENTATION: "
            "    HTN + ABSENT PUBERTY — consider CYP17A1 in any adolescent with primary amenorrhoea + HTN; "
            "    46XY: female phenotype at birth → complete male-to-female sex reversal; "
            "      Diagnosed when puberty absent or karyotype obtained for another reason; "
            "  GONADAL TUMOUR RISK: "
            "    46XY streak/dysgenetic testes: gonadoblastoma risk → gonadectomy recommended; "
            "    46XX: streak ovaries (low functional reserve) → HRT mandatory; "
            "  DOC MECHANISM: "
            "    DOC accumulates (17α-OH blocked → cannot proceed to cortisol, cannot be removed); "
            "    DOC acts as weak mineralocorticoid → Na+ retention; "
            "    Aldosterone paradoxically LOW (renin suppressed by DOC-driven volume); "
            "  CORTISOL PARTIALLY COMPENSATED: "
            "    Corticosterone (without 17α-hydroxylation) acts as weak glucocorticoid; "
            "    Adrenal crisis less common but still occurs; "
            "  TREATMENT: HC NORMALISES HTN (DOC-mediated): "
            "    HC → ACTH falls → progesterone/DOC falls → Na+ retention reverses → BP normalises"
        ),
        "disease_pathway": (
            "CYP17A1 LOF → ABSENT CORTISOL + ABSENT SEX STEROIDS + DOC EXCESS → CLINICAL TRIAD: "
            "  Steroidogenesis without CYP17A1: "
            "    Zona glomerulosa (CYP17A1-independent): aldosterone pathway intact in substrate; "
            "    Zona fasciculata: pregnenolone → progesterone → (no 17α-OH) → DOC → corticosterone; "
            "    BUT: cortisol requires 17α-OH → BLOCKED; "
            "    ACTH rises: more progesterone → more DOC + corticosterone → mineralocorticoid; "
            "  Sex steroid pathway: "
            "    17-OH-pregnenolone → DHEA → androstenedione → testosterone/estrogen: ALL BLOCKED; "
            "    Gonads: same CYP17A1 deficiency → no testicular testosterone (46XY) → female phenotype; "
            "    No estrogen (46XX) → streak ovaries + primary amenorrhoea; "
            "  Therapy: "
            "    HC: ACTH suppressed → DOC/corticosterone reduced → HTN resolves; "
            "    Estrogen (both sexes): restores secondary sex characteristics (HC does not)"
        ),
    },
    {
        "gene": "STAR",
        "protein": (
            "STAR -- 8p11.23 AR -- 285aa -- Steroidogenic-Acute-Regulatory-Protein-"
            "32kDa-Lipoid-CAH-Most-Severe-ALL-Steroids-Absent-"
            "CT-Enlarged-Lipid-Laden-Adrenals-PATHOGNOMONIC-"
            "46XY-Female-External-Phenotype-Neonatal-Crisis-"
            "OMIM-Gene-600617-Disease-OMIM-201710"
        ),
        "locus": "8p11.23",
        "protein_size": (
            "285 aa / 32 kDa (STAR — steroidogenic acute regulatory protein; "
            "also 30 kDa mitochondrial processed form; "
            "FUNCTION: "
            "  Rate-limiting step for ALL steroidogenesis: transfers cholesterol from outer to inner "
            "  mitochondrial membrane (OMM→IMM); "
            "  Without StAR, cholesterol cannot reach CYP11A1 (P450scc) on IMM; "
            "  Required in adrenal (all three zones) + gonad (Leydig cells + granulosa cells); "
            "  NOT required for some StAR-independent pathways (placental progesterone, brain neurosteroids); "
            "LOF CONSEQUENCE: "
            "  ALL adrenal steroids absent (glucocorticoid + mineralocorticoid + sex steroids); "
            "  Cholesterol accumulates in lipid droplets in adrenal cells → "
            "    lipid-laden enlarged adrenals; "
            "    PATHOGNOMONIC: CT/MRI shows massively enlarged, lipid-filled adrenals; "
            "  Gonadal steroidogenesis absent: "
            "    46XY Leydig cells: no testosterone → female external genitalia; "
            "    46XX: ovaries have StAR-independent estrogen production pathway → "
            "      female puberty occurs normally (follicular estrogen); "
            "encoded 8p11.23"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — StAR Lipoid CAH: "
            "  Biallelic LOF mutations; Q258X most common (Japanese + Korean ancestry); "
            "  R182L, R182H: Ashkenazi Jewish founder; "
            "CLINICAL FEATURES: "
            "  MOST SEVERE CAH: COMPLETE absence of all adrenal and gonadal steroids; "
            "  NEONATAL ADRENAL CRISIS: "
            "    Day 1-4 of life: profound Na+ loss + K+ elevation + hypoglycaemia + shock; "
            "    No cortisol (no GC), no aldosterone (no MC) → life-threatening; "
            "    All 46XY: female external genitalia at birth (no testosterone during development); "
            "    All 46XX: phenotypically normal female newborn; "
            "  IMAGING PATHOGNOMONIC: "
            "    Adrenal CT/MRI: bilateral massive adrenal enlargement with lipid density; "
            "    If small/atrophic adrenals → lipid has depleted → later stage; "
            "  46XY SEX REVERSAL: "
            "    Cryptorchidism; female phenotype; testes (produce no testosterone); "
            "    Diagnosis often at birth (ambiguous) or when puberty absent + 46XY karyotype; "
            "  46XX PUBERTY: "
            "    Normal female puberty (StAR-independent ovarian estrogen); "
            "    BUT: hypergonadotropic hypogonadism develops in 20s (follicular depletion); "
            "TREATMENT: "
            "    HC + fludrocortisone lifelong; stress dosing; adrenal crisis kit; "
            "    46XY: gonadectomy at adolescence (remove non-functional testes, prevent tumour risk) + estrogen; "
            "    46XX: estrogen + progestogen for cycle maintenance after natural puberty"
        ),
        "disease_category": (
            "LIPOID CAH — STAR — MOST SEVERE ALL-STEROID-ABSENT FORM: "
            "  CRITICAL CLINICAL RULE: NEONATAL EMERGENCY; "
            "    Day 1-4 adrenal crisis = complete steroid failure → IV hydrocortisone + fluids + NaCl IMMEDIATELY; "
            "  IMAGING RULE: "
            "    CT/MRI enlarged lipid-laden adrenals PATHOGNOMONIC for StAR deficiency; "
            "    Distinguish: CYP11A1 deficiency (similar but adrenals less lipid-laden; "
            "      CYP11A1 deficiency = first enzymatic step missing, not cholesterol transport); "
            "  46XY: all present as phenotypically female — NEVER male genitalia in complete STAR LOF; "
            "    Critical: karyotype ALL ambiguous/female neonates with adrenal crisis; "
            "  46XX: present as normal female → often diagnosed LATE (neonatal crisis may be less severe); "
            "    Adrenal crisis in 46XX still life-threatening → high index of suspicion; "
            "  TREATMENT: "
            "    HC + fludrocortisone: immediate initiation; "
            "    Salt supplement: first months of life (10-30 mmol NaCl/day); "
            "    ADRENAL CRISIS KIT: IM hydrocortisone — teach parents before discharge"
        ),
        "disease_pathway": (
            "STAR LOF → CHOLESTEROL TRANSPORT BLOCKED → LIPID DROPLETS → ALL STEROID FAILURE: "
            "  Normal StAR function: "
            "    ACTH → cAMP → StAR protein synthesised → phosphorylated (Ser194) → "
            "      translocates to OMM → creates contact sites → cholesterol flux to IMM; "
            "    CYP11A1 (P450scc) on IMM: cholesterol → pregnenolone (first step); "
            "    Without StAR: cholesterol delivery to CYP11A1 → negligible (StAR-independent flux ~14%); "
            "  StAR LOF: "
            "    ACTH rises (no cortisol feedback) → stimulates adrenal → MORE cholesterol synthesised/stored; "
            "    Cholesterol cannot be transported → accumulates in lipid droplets (esterified cholesterol); "
            "    Lipid droplets expand → cytotoxic → adrenal cell damage; "
            "    Over time: adrenal cells die → eventual adrenal destruction (hence atrophic in older patients); "
            "    Imaging: enlarged lipid-filled adrenals early; depleted/fibrotic late; "
            "  Gonadal: "
            "    Leydig cells: same StAR deficiency → no testosterone → female phenotype in 46XY; "
            "    Granulosa/theca: StAR-independent pathway (CYP11A1 on inner membrane, partial access) → "
            "      some estrogen in 46XX → normal puberty; "
            "      Eventually: follicular pool depletes → hypergonadotropic hypogonadism in 3rd decade"
        ),
    },
    {
        "gene": "CYP11A1",
        "protein": (
            "CYP11A1 -- 15q24.1 AR -- 521aa -- P450scc-Cholesterol-Side-Chain-Cleavage-"
            "60kDa-First-Enzymatic-Step-Steroidogenesis-"
            "Similar-to-STAR-Milder-Adrenals-Less-Lipid-Laden-"
            "Variable-Severity-Partial-LOF-Late-Onset-"
            "OMIM-Gene-118485-Disease-OMIM-613743"
        ),
        "locus": "15q24.1",
        "protein_size": (
            "521 aa / 60 kDa (CYP11A1 — cytochrome P450 family 11 subfamily A member 1; "
            "mitochondrial inner membrane; adrenocortical + gonadal + placental expression; "
            "FUNCTION: "
            "  P450scc = cholesterol side-chain cleavage enzyme; "
            "  Converts cholesterol → pregnenolone (first and committed step of steroidogenesis); "
            "  3-step oxidation: "
            "    20α-hydroxycholesterol → 22R-hydroxycholesterol → 20,22R-dihydroxycholesterol → pregnenolone + isocaproaldehyde; "
            "  Requires: ferredoxin reductase (FDXR) + ferredoxin (FDX1) electron transfer chain; "
            "  Expressed in adrenal, gonad, placenta, brain; "
            "LOF CONSEQUENCE: "
            "  No pregnenolone → ALL downstream steroids absent (same as StAR); "
            "  DIFFERENCE FROM STAR: "
            "    CYP11A1 block: enzyme absent → cholesterol CANNOT be converted; "
            "    Cholesterol still transported to IMM by StAR (StAR functional); "
            "    Adrenals less lipid-laden than StAR (cholesterol not as massively accumulated); "
            "    Clinical severity: similar to StAR deficiency for complete LOF; "
            "    Partial LOF: late-onset adrenal insufficiency (cortisol/aldosterone deficiency in adulthood); "
            "encoded 15q24.1 (note: chromosome 15, not 8 like STAR)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — CYP11A1 P450scc deficiency: "
            "  Biallelic mutations: complete LOF = severe neonatal-onset; "
            "  Partial LOF (missense reducing enzyme activity): late-onset, variable; "
            "  Very rare: few hundred cases described globally; "
            "CLINICAL FEATURES: "
            "  COMPLETE LOF: identical to StAR — all steroids absent; neonatal crisis; 46XY sex reversal; "
            "    DIFFERENTIATION from StAR: adrenal imaging less lipid-laden (cholesterol accumulation absent since StAR works but CYP11A1 missing); "
            "    Definitive distinction: gene sequencing panel (StAR + CYP11A1 both tested); "
            "  PARTIAL LOF (more common than complete): "
            "    Late-onset adrenal insufficiency (childhood, adolescence, early adulthood); "
            "    May present with Addisonian crisis under stress; "
            "    Basal cortisol low-normal; poor Synacthen response; "
            "    46XY: may have partial virilisation (residual testosterone); "
            "    46XX: normal female puberty (ovarian pathway partially StAR-independent + some P450scc activity); "
            "TREATMENT: "
            "    HC + fludrocortisone: same as StAR; "
            "    Severity monitoring: variable — some partial LOF: cortisol only deficient; "
            "    Adrenal crisis kit mandatory for all forms"
        ),
        "disease_category": (
            "P450SCC (CYP11A1) DEFICIENCY — FIRST ENZYMATIC STEP ABSENT — STAR DIFFERENTIAL: "
            "  KEY CLINICAL DISTINCTION from StAR: "
            "    StAR: cholesterol transport blocked → accumulates → lipid-laden adrenals on imaging; "
            "    CYP11A1: enzyme absent → pregnenolone cannot form → adrenals NOT markedly lipid-laden; "
            "    Clinically: identical — GENE PANEL distinguishes; "
            "  PARTIAL LOF SPECTRUM: "
            "    Partial activity → late-onset presentations; "
            "    May be missed for years until adrenal crisis under stress (surgery, illness, pregnancy); "
            "    Consider CYP11A1 in unexplained adrenal insufficiency at any age (rare but treatable); "
            "  GONADAL INVOLVEMENT: "
            "    Same as StAR: 46XY sex reversal (complete LOF); "
            "    46XX: ovarian function initially preserved (StAR-independent pathway) then declines; "
            "  MANAGEMENT: "
            "    Glucocorticoid + mineralocorticoid replacement (same protocol as primary adrenal insufficiency); "
            "    Annual cortisol day curve to monitor adequacy; "
            "    Hydration + electrolytes: education for all patients + family"
        ),
        "disease_pathway": (
            "CYP11A1 LOF → PREGNENOLONE ABSENT → ALL STEROID PATHWAYS BLOCKED — SIMILAR TO STAR: "
            "  Electron transfer chain required: "
            "    NADPH → FDXR → FDX1 → CYP11A1 → cholesterol side-chain cleavage; "
            "  Complete CYP11A1 LOF: "
            "    Pregnenolone not produced → all downstream steroids absent (progesterone, 17-OHP, DHEA, etc.); "
            "    Adrenals: ACTH elevated → stimulates steroidogenesis → but no pregnenolone; "
            "    Unlike StAR: cholesterol successfully transported to IMM by StAR but cannot be enzymatically cleaved; "
            "    Adrenal cells: less lipid-laden (cholesterol enters mitochondria → not deposited as droplets in cytoplasm); "
            "  Partial LOF: "
            "    Residual enzyme activity → some pregnenolone → some cortisol; "
            "    Basal may be adequate, stress-response insufficient → Synacthen test critical; "
            "    46XY: variable masculinisation (depends on residual testosterone from partial CYP11A1 activity in testes)"
        ),
    },
    {
        "gene": "POR",
        "protein": (
            "POR -- 7q11.23 AR -- 680aa -- P450-Oxidoreductase-"
            "77kDa-Electron-Donor-ALL-Microsomal-CYPs-"
            "Antley-Bixler-Craniosynostosis-Radiohumeral-Synostosis-PATHOGNOMONIC-"
            "Maternal-Virilisation-CYP19A1-Aromatase-Blocked-"
            "Combined-CYP21A2-CYP17A1-Enzymatic-Block-"
            "OMIM-Gene-124015-Disease-OMIM-201750"
        ),
        "locus": "7q11.23",
        "protein_size": (
            "680 aa / 77 kDa (POR — cytochrome P450 oxidoreductase; "
            "ER membrane-bound; ubiquitous electron donor; "
            "FUNCTION: "
            "  Sole electron donor for ALL microsomal (ER-bound) cytochrome P450 enzymes: "
            "    CYP21A2 (21-hydroxylase), CYP17A1 (17α-hydroxylase), CYP19A1 (aromatase), "
            "    CYP51A1 (lanosterol 14α-demethylase — cholesterol synthesis), "
            "    CYP26 (retinoic acid metabolism), and others; "
            "  POR transfers electrons from NADPH to all these CYPs; "
            "  Note: mitochondrial CYPs (CYP11A1, CYP11B1, CYP11B2) use FDXR/FDX1, NOT POR; "
            "LOF CONSEQUENCE: "
            "  COMBINED enzymatic block of ALL POR-dependent CYPs: "
            "    CYP21A2 block → some 17-OHP accumulation + partial virilisation; "
            "    CYP17A1 block → absent sex steroids + some DOC accumulation; "
            "    CYP19A1 (aromatase) block → aromatase absent → fetal androgens not converted to estrogen; "
            "      MATERNAL VIRILISATION: fetal androgens escape to mother → maternal acne/hirsutism in pregnancy; "
            "      This is PATHOGNOMONIC clue — maternal virilisation in pregnancy; "
            "    CYP51A1 block → disrupted cholesterol synthesis → skeletal malformations; "
            "encoded 7q11.23 (Williams-Beuren region)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — POR deficiency: "
            "  A287P (Ala287Pro) most common in European ancestry; R457H in Asian ancestry; "
            "  Spectrum from mild to severe depending on residual POR activity; "
            "CLINICAL FEATURES — ANTLEY-BIXLER SYNDROME PLUS: "
            "  SKELETAL: craniosynostosis (multiple cranial sutures); radiohumeral synostosis (elbow fixed); "
            "    choanal atresia; femoral bowing; arachnodactyly; "
            "  STEROID: combined CYP21A2 + CYP17A1 block → mixed profile; "
            "    46XX: ambiguous genitalia (mild virilisation from partial CYP21A2-like effect + no aromatase); "
            "    46XY: undervirilised (partial CYP17A1-like block); "
            "    Salt-wasting: uncommon (partial CYP21A2 activity); "
            "  MATERNAL VIRILISATION: "
            "    Fetal androgens cannot be aromatised (no CYP19A1 activity) → "
            "    pass to maternal compartment → maternal acne, hirsuitism in 3rd trimester; "
            "    PATHOGNOMONIC CLUE; "
            "  CORTISOL: variably deficient (partial CYP21A2 block); may have adrenal insufficiency; "
            "  URINE STEROID PROFILE: "
            "    Complex — both 17-OHP elevated AND 17-OH-pregnenolone elevated + DOC + corticosterone; "
            "    Metabolomics/comprehensive steroid profile required for diagnosis; "
            "TREATMENT: "
            "    HC if adrenal insufficiency confirmed (Synacthen test); "
            "    Surgical: craniosynostosis + choanal atresia (neonatal emergency if airways compromised); "
            "    DSD team: genital ambiguity in both 46XX and 46XY; "
            "    No single treatment rule: patient-specific based on steroid profile"
        ),
        "disease_category": (
            "POR DEFICIENCY — COMBINED MICROSOMAL CYP BLOCK — ANTLEY-BIXLER + MATERNAL VIRILISATION: "
            "  UNIQUE DIAGNOSTIC CLUE: "
            "    Maternal virilisation in pregnancy (maternal acne/hirsutism 3rd trimester); "
            "    Caused by fetal CYP19A1 (aromatase) absence → androgens escape to mother; "
            "    Not seen in any other CAH form; "
            "  SKELETAL PATHOGNOMONIC: Antley-Bixler syndrome features: "
            "    Craniosynostosis: surgery in first weeks of life (intracranial pressure risk); "
            "    Radiohumeral synostosis: elbow fixed → occupational therapy; "
            "  STEROID PROFILE COMPLEXITY: "
            "    NOT a simple single-enzyme block; multiple CYPs affected; "
            "    Standard CAH markers (17-OHP, compound S) elevated to variable degrees; "
            "    Comprehensive urinary steroid metabolomics essential; "
            "  ALLELIC HETEROGENEITY: "
            "    A287P (European): partial POR → moderate phenotype; "
            "    R457H (East Asian): partial POR → similar; "
            "    Null mutations: severe; craniosynostosis + complete DSD"
        ),
        "disease_pathway": (
            "POR LOF → ALL MICROSOMAL CYP ACTIVITIES REDUCED → MULTI-ENZYME STEROIDOGENESIS BLOCK: "
            "  POR electron delivery chain: "
            "    NADPH → POR (FMN + FAD domains) → electrons → CYP active site (haem-iron); "
            "    CYP reduced → reacts with molecular oxygen → substrate hydroxylation; "
            "  POR LOF: ALL microsomal CYPs receive fewer electrons → all have reduced activity: "
            "    CYP21A2: reduced → 17-OHP partially elevated; "
            "    CYP17A1: reduced → less 17α-OH + less 17,20-lyase; some DOC accumulation; "
            "    CYP19A1: reduced or absent → aromatase inactive → androgens not converted to estrogen; "
            "      In fetoplacental unit: fetal adrenal androgens (DHEA) → placenta aromatase (CYP19A1) → estrogen; "
            "      POR deficiency: CYP19A1 in placenta also affected → androgens pass to mother; "
            "    CYP51A1: reduced → lanosterol accumulates → disrupted cholesterol biosynthesis → skeletal malformations; "
            "  Steroid profile: complex mixture reflecting all blocked steps; "
            "  Treatment: HC (Synacthen-confirmed adrenal insufficiency); neonatal airway if choanal atresia"
        ),
    },
    {
        "gene": "CYP11B2",
        "protein": (
            "CYP11B2 -- 8q24.3 AR -- 503aa -- Aldosterone-Synthase-CMO-I-CMO-II-"
            "56kDa-ISOLATED-Aldosterone-Deficiency-NO-Virilisation-NO-Cortisol-Deficiency-"
            "Normal-17OHP-Normal-Cortisol-PURE-Salt-Wasting-"
            "18-OH-Corticosterone-Elevated-CMO-II-PATHOGNOMONIC-"
            "Fludrocortisone-ONLY-No-HC-Needed-"
            "OMIM-Gene-124080-Disease-OMIM-203400"
        ),
        "locus": "8q24.3",
        "protein_size": (
            "503 aa / 56 kDa (CYP11B2 — cytochrome P450 family 11 subfamily B member 2; "
            "aldosterone synthase; mitochondrial; zona glomerulosa exclusive; "
            "FUNCTION: "
            "  Three sequential oxidations: "
            "    11-deoxycorticosterone (DOC) → corticosterone (11β-hydroxylation); "
            "    Corticosterone → 18-OH-corticosterone (18-hydroxylation); "
            "    18-OH-corticosterone → aldosterone (18-oxidation/CMO II activity); "
            "  Expressed ONLY in zona glomerulosa; "
            "  Regulated by angiotensin II + K+ (not ACTH); "
            "  DISTINCT from CYP11B1 (11β-hydroxylase, zona fasciculata): "
            "    96% amino acid identity; different zone; different regulation; different function; "
            "LOF CONSEQUENCE: "
            "  Aldosterone absent or severely reduced; "
            "  Corticosterone elevated (11β-hydroxylation intact → DOC → corticosterone; "
            "    but 18-hydroxylation/18-oxidation blocked → 18-OH-B elevated in CMO II); "
            "  Cortisol: NORMAL (zona fasciculata/CYP11B1 intact); "
            "  17-OHP: NORMAL (not a block in 17-hydroxylase pathway); "
            "  PURE MINERALOCORTICOID DEFICIENCY: Na+ loss + K+ elevation + dehydration; "
            "encoded 8q24.3 (adjacent to CYP11B1; 96% identity → gene conversion can cause hybrid alleles)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — CYP11B2 CMO I / CMO II deficiency: "
            "  CMO I: 11β-hydroxylation AND 18-hydroxylation blocked → no corticosterone, no 18-OHB, no aldosterone; "
            "  CMO II: 18-oxidation blocked → corticosterone → 18-OHB → ACCUMULATES (cannot convert to aldosterone); "
            "    CMO II: 18-OH-corticosterone/aldosterone ratio >100 (PATHOGNOMONIC); "
            "  Founder mutations: CMO II R181W (Ashkenazi Jewish); CMO I in Middle East; "
            "CLINICAL FEATURES: "
            "  PURE SALT-WASTING WITHOUT VIRILISATION: "
            "    Na+ loss + hyponatremia + hyperkalemia + metabolic acidosis; "
            "    Failure to thrive; poor feeding; recurrent vomiting (infant); "
            "    NO virilisation (no androgen excess; cortisol pathway normal); "
            "    NO HTN (no DOC excess — DOC is also deficient since CYP11B2 also reduces DOC); "
            "    Normal secondary sex development; normal puberty; "
            "  DIFFERENTIAL DIAGNOSIS: "
            "    CYP21A2 SW: 17-OHP elevated + virilisation in 46XX → excludes CYP11B2; "
            "    CYP11B2: 17-OHP NORMAL + no virilisation → isolated aldosterone pathway; "
            "    Pseudohypoaldosteronism type 1 (MR/ENaC defects): aldosterone HIGH (not low); "
            "DIAGNOSIS: "
            "    Aldosterone: low or undetectable; renin: markedly elevated (RAAS stimulated); "
            "    CMO II: 18-OH-corticosterone elevated; 18-OHB:aldosterone ratio >100; "
            "    Corticosterone: high in CMO II; normal in CMO I; "
            "    Cortisol: NORMAL; 17-OHP: NORMAL; "
            "TREATMENT: "
            "    Fludrocortisone 0.1-0.2 mg/day: replaces aldosterone; "
            "    NO hydrocortisone needed (cortisol normal); "
            "    NaCl supplement in infancy; "
            "    NBS: NOT detected by 17-OHP (17-OHP normal in CYP11B2 deficiency)"
        ),
        "disease_category": (
            "CYP11B2 ALDOSTERONE SYNTHASE DEFICIENCY — ISOLATED MINERALOCORTICOID DEFICIENCY: "
            "  CRITICAL DISTINGUISHING FEATURE: "
            "    Normal cortisol + normal 17-OHP + no virilisation + pure salt-wasting: "
            "    = Isolated aldosterone deficiency (CYP11B2); "
            "    Excludes all other CAH genes; "
            "  NBS PITFALL: NOT detected on 17-OHP NBS — presents in infancy with FTT + salt-wasting; "
            "    Any infant with salt-wasting + normal 17-OHP → CYP11B2 (and MR/ENaC genes) in differential; "
            "  CMO I vs CMO II BIOCHEMISTRY: "
            "    CMO I: all three CYP11B2 activities blocked → no corticosterone/18-OHB/aldosterone; "
            "    CMO II: only 18-oxidase blocked → 18-OH-corticosterone ELEVATED; "
            "      18-OHB:aldosterone ratio >100 PATHOGNOMONIC for CMO II; "
            "  TREATMENT: FLUDROCORTISONE ONLY — simple, effective, lifelong; "
            "    No need for HC (different from all other CAH forms); "
            "    Dose titrates to renin (keep mid-normal); "
            "  ADULT SELF-SUFFICIENCY: "
            "    Many patients can reduce fludrocortisone dose in adulthood as renal Na+ handling improves; "
            "    Monitor renin + electrolytes annually"
        ),
        "disease_pathway": (
            "CYP11B2 LOF → ALDOSTERONE ABSENT → PURE MINERALOCORTICOID DEFICIENCY — NO GC DEFECT: "
            "  Aldosterone biosynthesis in zona glomerulosa: "
            "    Angiotensin II (from RAAS) + K+ → CYP11B2 expression in glomerulosa; "
            "    DOC → corticosterone (CYP11B2 step 1: 11β-hydroxylation); "
            "    Corticosterone → 18-OH-corticosterone (CYP11B2 step 2: 18-hydroxylation); "
            "    18-OH-corticosterone → aldosterone (CYP11B2 step 3: 18-oxidation/CMO II); "
            "  CYP11B2 LOF: "
            "    Steps blocked depending on which activities lost (CMO I = all; CMO II = last step); "
            "    Aldosterone absent → MR in collecting duct unstimulated → Na+ lost → volume depleted; "
            "    Volume depletion → renin markedly elevated; "
            "    Na+ loss → hyponatremia; K+ retention → hyperkalemia; "
            "  Cortisol: "
            "    CYP11B1 (separate gene) intact → cortisol synthesis unaffected; "
            "    ACTH normal (cortisol feedback normal); "
            "    17-OHP: normal (no block in 17-hydroxylase pathway); "
            "  Treatment: fludrocortisone → activates MR → Na+ reabsorption → crisis prevented"
        ),
    },
]


# ── Patient simulation ────────────────────────────────────────────────────────
def _make_patients(seed: int, gene: str, n: int = 40) -> list:
    rng = random.Random(seed)

    genders = ["M", "F"]
    gene_params = {
        "CYP21A2": {
            "onset": (0, 14),   "sbp": (100, 135),  "k_range": (4.5, 7.5),
            "na_range": (115, 135), "htn": False, "virilised_46xx": True,
            "salt_wasting_pct": 0.67, "htn_pct": 0.0, "17ohp_range": (150, 1000),
        },
        "CYP11B1": {
            "onset": (0, 180),  "sbp": (145, 200),  "k_range": (2.5, 3.8),
            "na_range": (135, 148), "htn": True, "virilised_46xx": True,
            "salt_wasting_pct": 0.0, "htn_pct": 0.70, "17ohp_range": (15, 60),
        },
        "HSD3B2": {
            "onset": (0, 21),   "sbp": (95, 130),   "k_range": (5.0, 8.0),
            "na_range": (110, 132), "htn": False, "virilised_46xx": True,
            "salt_wasting_pct": 0.80, "htn_pct": 0.0, "17ohp_range": (20, 80),
        },
        "CYP17A1": {
            "onset": (120, 216), "sbp": (145, 195), "k_range": (2.5, 3.5),
            "na_range": (135, 148), "htn": True, "virilised_46xx": False,
            "salt_wasting_pct": 0.0, "htn_pct": 0.80, "17ohp_range": (0, 5),
        },
        "STAR": {
            "onset": (0, 7),    "sbp": (75, 115),   "k_range": (6.0, 9.0),
            "na_range": (105, 128), "htn": False, "virilised_46xx": False,
            "salt_wasting_pct": 1.0, "htn_pct": 0.0, "17ohp_range": (0, 5),
        },
        "CYP11A1": {
            "onset": (0, 365),  "sbp": (85, 130),   "k_range": (5.0, 8.5),
            "na_range": (110, 135), "htn": False, "virilised_46xx": False,
            "salt_wasting_pct": 0.75, "htn_pct": 0.0, "17ohp_range": (0, 5),
        },
        "POR": {
            "onset": (0, 30),   "sbp": (100, 145),  "k_range": (3.5, 6.0),
            "na_range": (125, 140), "htn": False, "virilised_46xx": True,
            "salt_wasting_pct": 0.25, "htn_pct": 0.15, "17ohp_range": (30, 120),
        },
        "CYP11B2": {
            "onset": (3, 60),   "sbp": (85, 115),   "k_range": (5.5, 8.0),
            "na_range": (118, 132), "htn": False, "virilised_46xx": False,
            "salt_wasting_pct": 1.0, "htn_pct": 0.0, "17ohp_range": (5, 20),
        },
    }
    p = gene_params.get(gene, gene_params["CYP21A2"])

    def treatment_choice():
        r = rng.random()
        if gene == "CYP21A2":
            if r < 0.55: return "Hydrocortisone + Fludrocortisone"
            if r < 0.75: return "Hydrocortisone only (NC/SV)"
            if r < 0.88: return "HC + Fludrocortisone + NaCl supplement"
            return "HC + Fludrocortisone + genital surgery"
        elif gene == "CYP11B1":
            if r < 0.70: return "Hydrocortisone (no fludrocortisone)"
            if r < 0.85: return "HC + antihypertensive (transition)"
            return "HC + calcium channel blocker"
        elif gene == "HSD3B2":
            if r < 0.65: return "HC + Fludrocortisone"
            if r < 0.85: return "HC + Fludrocortisone + NaCl"
            return "HC + Fludrocortisone + sex hormones (pubertal)"
        elif gene == "CYP17A1":
            if r < 0.55: return "HC + sex hormone replacement"
            if r < 0.75: return "HC (HTN resolved on HC)"
            if r < 0.88: return "HC + estrogen (46XX)"
            return "HC + gonadectomy + estrogen (46XY)"
        elif gene == "STAR":
            if r < 0.60: return "HC + Fludrocortisone + NaCl"
            if r < 0.80: return "HC + Fludrocortisone + gonadectomy (46XY)"
            return "IV hydrocortisone (neonatal crisis) → lifelong oral HC+Fludro"
        elif gene == "CYP11A1":
            if r < 0.55: return "HC + Fludrocortisone"
            if r < 0.78: return "HC only (partial LOF)"
            return "HC + Fludrocortisone + sex hormones (46XY)"
        elif gene == "POR":
            if r < 0.45: return "HC (adrenal insufficiency confirmed)"
            if r < 0.65: return "HC + craniosynostosis surgery"
            if r < 0.80: return "Craniosynostosis surgery alone (mild adrenal)"
            return "HC + choanal atresia repair + DSD management"
        else:  # CYP11B2
            if r < 0.75: return "Fludrocortisone only"
            if r < 0.90: return "Fludrocortisone + NaCl supplement (infant)"
            return "Fludrocortisone + dietary NaCl"

    patients = []
    for i in range(n):
        onset_days = rng.randint(*p["onset"])
        age_dx     = onset_days + rng.randint(1, 30)
        sbp        = rng.randint(*p["sbp"])
        dbp        = sbp - rng.randint(20, 40)
        k_plus     = round(rng.uniform(*p["k_range"]), 2)
        na_serum   = round(rng.uniform(*p["na_range"]), 1)
        ohp_17     = round(rng.uniform(*p["17ohp_range"]), 1)
        gender     = rng.choice(genders)
        sw_crisis  = rng.random() < p["salt_wasting_pct"]
        htn        = p["htn"] and (rng.random() < p["htn_pct"])
        virilised  = p["virilised_46xx"] and (gender == "F") and rng.random() < 0.80
        sex_reversal = (gene in ("STAR", "CYP17A1", "CYP11A1", "POR")) and (gender == "M") and rng.random() < 0.80
        tx         = treatment_choice()

        patients.append({
            "id":                f"{gene}-{i+1:02d}",
            "gene":              gene,
            "gender":            gender,
            "onset_days":        onset_days,
            "age_at_dx_days":    age_dx,
            "sbp_mmhg":          sbp,
            "dbp_mmhg":          dbp,
            "serum_k_mmol":      k_plus,
            "serum_na_mmol":     na_serum,
            "serum_17ohp_nmol":  ohp_17,
            "salt_wasting_crisis": sw_crisis,
            "hypertension":      htn,
            "virilisation_46xx": virilised,
            "sex_reversal_46xy": sex_reversal,
            "treatment":         tx,
        })
    return patients


# ── API surface ───────────────────────────────────────────────────────────────
def generate_overview() -> dict:
    """Atlas overview — aggregate stats across all 8 CAH steroidogenesis genes."""
    all_patients = []
    for idx, g in enumerate(ATLAS_GENES):
        all_patients.extend(_make_patients(SEED_BASE + idx, g["gene"]))

    n               = len(all_patients)
    n_sw            = sum(1 for p in all_patients if p["salt_wasting_crisis"])
    n_htn           = sum(1 for p in all_patients if p["hypertension"])
    n_virilised     = sum(1 for p in all_patients if p["virilisation_46xx"])
    n_sex_rev       = sum(1 for p in all_patients if p["sex_reversal_46xy"])
    n_hi_17ohp      = sum(1 for p in all_patients if p["serum_17ohp_nmol"] > 30)
    mean_sbp        = round(sum(p["sbp_mmhg"]   for p in all_patients) / n, 1)
    mean_k          = round(sum(p["serum_k_mmol"] for p in all_patients) / n, 2)
    mean_na         = round(sum(p["serum_na_mmol"] for p in all_patients) / n, 1)

    gene_summary = []
    for idx, g in enumerate(ATLAS_GENES):
        pts = _make_patients(SEED_BASE + idx, g["gene"])
        gene_summary.append({
            "gene":          g["gene"],
            "locus":         g["locus"],
            "n_patients":    len(pts),
            "mean_sbp":      round(sum(p["sbp_mmhg"]     for p in pts) / len(pts), 1),
            "mean_k":        round(sum(p["serum_k_mmol"]  for p in pts) / len(pts), 2),
            "mean_na":       round(sum(p["serum_na_mmol"] for p in pts) / len(pts), 1),
            "sw_crisis_pct": round(100 * sum(1 for p in pts if p["salt_wasting_crisis"]) / len(pts), 1),
            "htn_pct":       round(100 * sum(1 for p in pts if p["hypertension"])        / len(pts), 1),
            "viril_46xx_pct":round(100 * sum(1 for p in pts if p["virilisation_46xx"])  / len(pts), 1),
            "sex_rev_pct":   round(100 * sum(1 for p in pts if p["sex_reversal_46xy"])  / len(pts), 1),
            "hi_17ohp_pct":  round(100 * sum(1 for p in pts if p["serum_17ohp_nmol"] > 30) / len(pts), 1),
            "syndrome": (
                "Classic-SW/SV/NC"  if g["gene"] == "CYP21A2" else
                "Hypertensive-CAH"  if g["gene"] in ("CYP11B1", "CYP17A1") else
                "All-Zone-CAH"      if g["gene"] == "HSD3B2" else
                "Lipoid-CAH"        if g["gene"] in ("STAR", "CYP11A1") else
                "Combined-Block"    if g["gene"] == "POR" else
                "Isolated-MC-Def"
            ),
        })

    return {
        "atlas":          "Hereditary-CAH-Atlas",
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "n_genes":        len(ATLAS_GENES),
        "n_patients":     n,
        "seeds":          f"{SEED_BASE}–{SEED_BASE + len(ATLAS_GENES) - 1}",
        "syndromes":      [
            "Classic 21-OH CAH (CYP21A2)",
            "Hypertensive virilising CAH (CYP11B1)",
            "All-zone CAH paradoxical DSD (HSD3B2)",
            "Hypertensive absent-puberty CAH (CYP17A1)",
            "Lipoid CAH most-severe (STAR)",
            "P450scc deficiency variable (CYP11A1)",
            "Combined-CYP block Antley-Bixler (POR)",
            "Isolated aldosterone deficiency (CYP11B2)",
        ],
        "aggregate_metrics": {
            "mean_sbp_mmhg":         mean_sbp,
            "mean_serum_k_mmol":     mean_k,
            "mean_serum_na_mmol":    mean_na,
            "salt_wasting_crisis_pct": round(100 * n_sw      / n, 1),
            "hypertensive_cah_pct":    round(100 * n_htn     / n, 1),
            "virilisation_46xx_pct":   round(100 * n_virilised / n, 1),
            "sex_reversal_46xy_pct":   round(100 * n_sex_rev  / n, 1),
            "elevated_17ohp_pct":      round(100 * n_hi_17ohp / n, 1),
        },
        "gene_summary": gene_summary,
        "key_clinical_rules": [
            "CYP21A2 (~95%): 17-OHP >100 nmol/L PATHOGNOMONIC classic; NBS heel-prick 17-OHP; MLPA mandatory (gene conversion missed by sequencing alone)",
            "CYP11B1 (~5%): compound S (11-deoxycortisol) elevated + HYPERTENSION + hypokalemia; NO fludrocortisone — DOC acts as mineralocorticoid",
            "CYP21A2 SW vs CYP11B1: CYP21A2 = salt-wasting + low BP; CYP11B1 = hypertension + normal/high Na",
            "CYP17A1: HTN + absent puberty + 46XY female phenotype — karyotype all adolescents with primary amenorrhoea + HTN",
            "STAR lipoid CAH: CT/MRI enlarged lipid-laden adrenals PATHOGNOMONIC; 46XY = female external genitalia; neonatal crisis day 1-4",
            "CYP11B2: normal 17-OHP + normal cortisol + pure salt-wasting = isolated aldosterone deficiency; fludrocortisone ONLY (no HC)",
            "POR: maternal virilisation in pregnancy (CYP19A1 aromatase blocked) + Antley-Bixler skeletal PATHOGNOMONIC",
            "HSD3B2 paradox: 46XX mildly virilised; 46XY undervirilised; delta-5 steroids (DHEA) elevated",
        ],
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for all 8 CAH genes."""
    genes_data = []
    for idx, g in enumerate(ATLAS_GENES):
        pts = _make_patients(SEED_BASE + idx, g["gene"])
        treatments = {}
        for p in pts:
            treatments[p["treatment"]] = treatments.get(p["treatment"], 0) + 1
        genes_data.append({
            "gene":            g["gene"],
            "locus":           g["locus"],
            "protein":         g["protein"],
            "protein_size":    g["protein_size"],
            "inheritance":     g["inheritance"],
            "disease_category": g["disease_category"],
            "disease_pathway": g["disease_pathway"],
            "n_patients":      len(pts),
            "mean_sbp":        round(sum(p["sbp_mmhg"]       for p in pts) / len(pts), 1),
            "mean_k":          round(sum(p["serum_k_mmol"]    for p in pts) / len(pts), 2),
            "mean_na":         round(sum(p["serum_na_mmol"]   for p in pts) / len(pts), 1),
            "mean_17ohp":      round(sum(p["serum_17ohp_nmol"] for p in pts) / len(pts), 1),
            "sw_crisis_pct":   round(100 * sum(1 for p in pts if p["salt_wasting_crisis"]) / len(pts), 1),
            "htn_pct":         round(100 * sum(1 for p in pts if p["hypertension"])         / len(pts), 1),
            "viril_pct":       round(100 * sum(1 for p in pts if p["virilisation_46xx"])   / len(pts), 1),
            "sex_rev_pct":     round(100 * sum(1 for p in pts if p["sex_reversal_46xy"])   / len(pts), 1),
            "treatment_distribution": treatments,
            "patients":        pts[:10],
        })
    return {"count": len(genes_data), "genes": genes_data}


def generate_definitions() -> dict:
    """Clinical glossary for hereditary CAH steroidogenesis defects."""
    return {
        "count": 8,
        "terms": [
            {
                "term": "CYP21A2 — 21-Hydroxylase Deficiency — 17-OHP >100 nmol/L PATHOGNOMONIC + MLPA Mandatory",
                "definition": (
                    "Most common CAH (~95% globally). 17-OHP accumulates → ACTH-driven androgen excess. "
                    "CLASSIC SALT-WASTING (SW): <1% enzyme activity; neonatal Na+ crisis day 7-14; "
                    "  46XX: ambiguous genitalia; 46XY: normal genitalia but crisis (often missed on NBS). "
                    "SIMPLE VIRILISING (SV): 1-2% activity; cortisol deficient, aldosterone borderline; "
                    "  virilisation without mineralocorticoid deficiency. "
                    "NON-CLASSIC (NC): 20-50% activity; mild androgen excess; PCOS-like in women. "
                    "17-OHP DIAGNOSTIC THRESHOLD: >100 nmol/L (30 ng/mL) random = classic; "
                    "  post-Synacthen 60-min >30 nmol/L (>10 ng/mL) = non-classic. "
                    "MLPA MANDATORY: 80% of mutations = gene conversion from CYP21A1P pseudogene; "
                    "  exon sequencing alone misses deletions/conversions — always add MLPA. "
                    "TREATMENT: HC 10-15 mg/m²/day ÷3; fludrocortisone (SW/SV); NaCl supplement (infants). "
                    "STRESS DOSING: illness/surgery → 3× HC; IM hydrocortisone 50-100 mg emergency kit."
                ),
            },
            {
                "term": "CYP11B1 — 11β-Hydroxylase Deficiency — Hypertensive Virilising CAH — Compound S Elevated — No Fludrocortisone",
                "definition": (
                    "2nd most common CAH (~5%). Compound S (11-deoxycortisol) accumulates — PATHOGNOMONIC marker. "
                    "DOC (11-deoxycorticosterone) accumulates → weak mineralocorticoid → Na+ retention → HYPERTENSION. "
                    "CRITICAL: NO salt-wasting (DOC preserves Na+); males not diagnosed on NBS (17-OHP NORMAL). "
                    "TREATMENT: HC only — suppresses ACTH → DOC falls → HTN resolves; "
                    "  NEVER fludrocortisone (DOC excess, not MC deficiency). "
                    "FOUNDER MUTATION: CYP11B1 R448H prevalent in North African/Sephardic Jewish ancestry. "
                    "VIRILISATION: same as CYP21A2 (ACTH-driven androgen excess); 46XX: ambiguous genitalia. "
                    "HTN REVERSAL: BP normalises over weeks-months once HC suppresses ACTH + DOC."
                ),
            },
            {
                "term": "HSD3B2 — 3β-HSD2 Deficiency — Delta-5 Steroid Accumulation — Paradoxical Virilisation/Undervirilisation",
                "definition": (
                    "Rare CAH; all three steroidogenic zones (mineralocorticoid + glucocorticoid + sex steroids) affected. "
                    "PARADOX: "
                    "  46XX: MILD virilisation (DHEA → peripheral HSD3B1 → androstenedione → testosterone); "
                    "  46XY: UNDERVIRILISED (no testicular testosterone — HSD3B2 also absent in testes); "
                    "  BOTH sexes: ambiguous genitalia at birth → DSD team mandatory. "
                    "DIAGNOSTIC CLUE: elevated DHEA/DHEAS + elevated 17-OH-pregnenolone (delta-5 steroids). "
                    "NBS: 17-OHP may be MILDLY elevated (cross-reacts) → steroid profile required to distinguish. "
                    "TREATMENT: HC + fludrocortisone (same as CYP21A2 SW) + sex hormone replacement at puberty."
                ),
            },
            {
                "term": "CYP17A1 — 17α-Hydroxylase/17,20-Lyase Deficiency — HTN + Absent Puberty + 46XY Sex Reversal",
                "definition": (
                    "HALLMARK TRIAD: hypertension + hypokalemia + absent puberty (both sexes). "
                    "ABSENT SEX STEROIDS: CYP17A1 required for sex steroid biosynthesis (adrenal + gonad). "
                    "46XY SEX REVERSAL: no testosterone → female external genitalia + cryptorchidism. "
                    "  Gonadal tumour risk → gonadectomy recommended (gonadoblastoma in dysgenetic testes). "
                    "46XX: streak ovaries + primary amenorrhoea + no breast development (no estrogen). "
                    "HTN MECHANISM: DOC + corticosterone accumulate (zona fasciculata diverted without 17α-OH); "
                    "  ALDOSTERONE SUPPRESSED (renin suppressed by DOC volume) — paradox. "
                    "CORTISOL: partially compensated by corticosterone (weak GC) → adrenal crisis less frequent. "
                    "TREATMENT: HC (HTN resolves) + sex hormone replacement; NO fludrocortisone."
                ),
            },
            {
                "term": "StAR Lipoid CAH — Most Severe — Enlarged Lipid-Laden Adrenals CT/MRI PATHOGNOMONIC — 46XY Female Phenotype",
                "definition": (
                    "Most severe CAH: all steroids absent (cortisol + aldosterone + sex steroids). "
                    "PATHOGNOMONIC IMAGING: CT/MRI bilateral enlarged lipid-filled adrenals. "
                    "  Cholesterol accumulates (cannot be transported to IMM by StAR) → lipid droplets. "
                    "  StAR vs CYP11A1: StAR = lipid-laden adrenals; CYP11A1 = less lipid (cholesterol transported but cannot be cleaved). "
                    "NEONATAL CRISIS: day 1-4; IV hydrocortisone + saline + glucose IMMEDIATELY. "
                    "46XY: ALL have female external genitalia (no testosterone during fetal virilisation); "
                    "  RULE: karyotype ALL neonates with adrenal crisis + female external genitalia. "
                    "46XX: normal female puberty (StAR-independent ovarian pathway) → but eventual POI in 20s. "
                    "FOUNDER MUTATIONS: Q258X (Japanese/Korean); R182L (Ashkenazi Jewish). "
                    "TREATMENT: HC + fludrocortisone lifelong; IM HC emergency kit; sex steroids (46XY)."
                ),
            },
            {
                "term": "CYP11A1 — P450scc Deficiency — Similar to StAR — Variable Severity — Partial LOF = Late-Onset Addison",
                "definition": (
                    "First enzymatic step in steroidogenesis (cholesterol → pregnenolone). "
                    "COMPLETE LOF: identical to StAR clinically (all steroids absent; neonatal crisis; 46XY sex reversal). "
                    "DISTINCTION from StAR: "
                    "  CYP11A1: enzyme absent → cholesterol enters mitochondria (StAR works) but not cleaved; "
                    "  ADRENALS LESS LIPID-LADEN than StAR (less cytoplasmic cholesterol accumulation). "
                    "  Definitive distinction by gene sequencing panel (both tested together). "
                    "PARTIAL LOF (more clinically common): "
                    "  Late-onset adrenal insufficiency (childhood to adulthood); "
                    "  May present as unexplained Addisonian crisis under stress; "
                    "  Basal cortisol normal → but poor Synacthen response → TEST required. "
                    "TREATMENT: HC + fludrocortisone; emergency kit; adrenal crisis education."
                ),
            },
            {
                "term": "POR — P450 Oxidoreductase Deficiency — Antley-Bixler Syndrome — Maternal Virilisation PATHOGNOMONIC",
                "definition": (
                    "POR electron-donates to ALL microsomal CYPs → combined CYP21A2 + CYP17A1 + CYP19A1 (aromatase) block. "
                    "MATERNAL VIRILISATION (3rd trimester) — PATHOGNOMONIC: "
                    "  Fetal CYP19A1 (placental aromatase) absent → fetal androgens not converted to estrogen; "
                    "  Androgens pass to mother → maternal acne/hirsutism; resolves post-delivery. "
                    "  ONLY CAH form causing maternal virilisation — diagnostic clue. "
                    "ANTLEY-BIXLER FEATURES: craniosynostosis + radiohumeral synostosis + choanal atresia; "
                    "  CYP51A1 (lanosterol demethylase) block → disrupted cholesterol synthesis → skeletal malformations. "
                    "NEONATAL AIRWAY: choanal atresia → neonatal emergency (surgical repair). "
                    "STEROID PROFILE: complex mixture (mixed CYP21A2 + CYP17A1 block); "
                    "  urine metabolomics required; not interpretable from single marker. "
                    "TREATMENT: HC if adrenal insufficiency confirmed; craniosynostosis surgery."
                ),
            },
            {
                "term": "CYP11B2 — Aldosterone Synthase Deficiency (CMO I/II) — Normal 17-OHP + Normal Cortisol — Pure Salt-Wasting — Fludrocortisone ONLY",
                "definition": (
                    "ISOLATED mineralocorticoid deficiency — zona glomerulosa only affected. "
                    "CRITICAL DISTINGUISHER: Normal cortisol + Normal 17-OHP + NO virilisation + pure salt-wasting. "
                    "  = CYP11B2 (or MR/ENaC defect): fludrocortisone-responsive. "
                    "NBS PITFALL: NOT detected on 17-OHP heel-prick; presents infant FTT + salt-wasting. "
                    "CMO I vs CMO II: "
                    "  CMO I: all CYP11B2 activities blocked → no corticosterone/18-OHB/aldosterone; "
                    "  CMO II: 18-oxidation blocked → 18-OH-corticosterone elevated; "
                    "    18-OHB:aldosterone ratio >100 = CMO II PATHOGNOMONIC. "
                    "TREATMENT: fludrocortisone ONLY — no HC (glucocorticoid axis normal); "
                    "  dose titrated to renin (aim: mid-normal renin range). "
                    "ADULT DOSE REDUCTION: many patients reduce fludrocortisone dose in adulthood. "
                    "ADJACENT GENE: CYP11B2 and CYP11B1 at 8q24.3 — 96% identical; "
                    "  hybrid CYP11B1/CYP11B2 genes → familial hyperaldosteronism type I (REVERSE phenotype)."
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:500])
    print("\n=== BREAKDOWN (count) ===")
    bd = generate_breakdown()
    print(f"Genes: {bd['count']}")
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Terms: {df['count']}")
