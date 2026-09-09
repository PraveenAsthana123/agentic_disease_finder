#!/usr/bin/env python3
"""Hereditary-Hypopituitarism-Atlas — Complete 8-Gene Atlas
(POU1F1 · PROP1 · HESX1 · OTX2 · SOX3 · LHX3 · LHX4 · GLI2).

POU1F1  (PIT-1 / Pituitary-Specific Transcription Factor 1; 291 aa; ~33 kDa; 3p11.2;
          AR (classic) or AD (dominant-negative);
          CPHD1 — GH + Prolactin + TSH triple deficiency;
          DOES NOT affect LH/FSH/ACTH/ADH;
          seed SEED_BASE+0).
PROP1   (Prophet of PIT-1; 226 aa; ~26 kDa; 5q35.3; AR;
          CPHD2 — MOST COMMON cause of CPHD globally;
          GH + TSH + PRL + LH + FSH; ACTH evolves LATER;
          transient pituitary mass then involution;
          seed SEED_BASE+1).
HESX1   (Homeobox Expressed in ES Cells 1; 185 aa; ~21 kDa; 3p14.3; AR/AD;
          Septo-optic dysplasia (SOD / de Morsier syndrome);
          optic nerve hypoplasia + absent septum pellucidum + pituitary hypoplasia;
          seed SEED_BASE+2).
OTX2    (Orthodenticle Homeobox 2; 289 aa; ~32 kDa; 14q22.3; AD;
          pituitary hypoplasia with severe eye anomalies;
          anophthalmia, microphthalmia, coloboma prominent;
          seed SEED_BASE+3).
SOX3    (SRY-Box Transcription Factor 3; 446 aa; ~50 kDa; Xq27.1; XLR;
          X-linked hypopituitarism + intellectual disability;
          infundibular hypoplasia on MRI;
          seed SEED_BASE+4).
LHX3    (LIM Homeobox 3; 397 aa; ~45 kDa; 9q34.3; AR;
          CPHD3 — GH+TSH+PRL+LH/FSH + RIGID CERVICAL SPINE PATHOGNOMONIC;
          cannot rotate neck — unique physical finding;
          seed SEED_BASE+5).
LHX4    (LIM Homeobox 4; 390 aa; ~45 kDa; 1q25.2; AD;
          CPHD4 — GH+TSH+ACTH; ACTH deficiency early (unlike PROP1);
          Arnold-Chiari + ectopic posterior pituitary;
          seed SEED_BASE+6).
GLI2    (GLI Family Zinc Finger 2; 1586 aa; ~174 kDa; 2q14.2; AD;
          HPE9 — holoprosencephaly spectrum + hypopituitarism;
          MOST VARIABLE EXPRESSIVITY; single central incisor;
          pituitary stalk interruption syndrome (PSIS);
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2486-2493).
"""

import random

SEED_BASE = 2486

HYPOPITUITARY_GENES = [
    # -- POU1F1 -- PIT-1 / CPHD1 ------------------------------------------------------------------
    {
        "gene": "POU1F1",
        "alt_name": (
            "POU1F1 (POU1F1-291aa-3p11.2 / AR-classic-AD-dominant-negative -- "
            "PIT-1-Pituitary-Specific-Transcription-Factor-1 -- "
            "CPHD1-Triple-GH+PRL+TSH-Deficiency -- "
            "SPARED-LH/FSH/ACTH/ADH-PATHOGNOMONIC -- "
            "Anterior-Pituitary-Hypoplasia)"
        ),
        "protein": (
            "POU1F1 -- 3p11.2 AR/AD-DN -- POU1F1-291aa -- "
            "PIT-1-POU-Domain-Transcription-Factor-33kDa-Nuclear -- "
            "Regulates-GH1-PRL-TSHB-Pit1-Gene-Expression-Somatotrophs-Lactotrophs-Thyrotrophs -- "
            "OMIM-Gene-173110-Disease-CPHD1-613038"
        ),
        "locus": "3p11.2",
        "protein_size": "291 aa / ~33 kDa",
        "inheritance": (
            "Autosomal recessive (classic biallelic LOF) or autosomal dominant (dominant-negative). "
            "POU1F1 encodes PIT-1, a POU-domain transcription factor expressed exclusively in the "
            "anterior pituitary. It activates transcription of GH1 (growth hormone), PRL (prolactin), "
            "and TSHB (TSH beta subunit), and is required for the differentiation and survival of "
            "somatotrophs, lactotrophs, and thyrotrophs. "
            "MECHANISM: POU1F1 binds specific promoter elements on GH1, PRL, and TSHB genes; "
            "LOF → these three cell lineages fail to develop or survive → triple hormone deficiency. "
            "CRITICAL SPARING: Gonadotrophs (LH/FSH) and corticotrophs (ACTH) do NOT require POU1F1 "
            "for differentiation → LH, FSH, ACTH, and ADH are NORMAL in POU1F1 deficiency. "
            "This is the defining DDx from PROP1 (which also loses gonadotropins). "
            "DOMINANT-NEGATIVE MUTATIONS: Certain missense variants (e.g., p.Arg271Trp) produce "
            "a mutant PIT-1 that actively represses POU1F1 target genes in heterozygotes — "
            "causing AD CPHD that may appear to be de novo or 'dominant' within a pedigree; "
            "recognizing the AD form prevents unnecessary parental testing for biallelic disease. "
            "IMAGING: Anterior pituitary hypoplasia (small adenohypophysis); neurohypophysis (posterior "
            "pituitary) and pituitary stalk are typically NORMAL (contrast with HESX1, OTX2, LHX4, GLI2 "
            "where ectopic posterior pituitary / stalk interruption is common). "
            "MRI bright spot (posterior pituitary T1 signal) is present and normally positioned."
        ),
        "disease_category": (
            "Combined Pituitary Hormone Deficiency type 1 (CPHD1, OMIM #613038); "
            "AR or AD-dominant-negative; incidence ~1/4,000-8,000; "
            "triple deficiency: GH + PRL + TSH; LH/FSH/ACTH/ADH SPARED; "
            "anterior pituitary hypoplasia; normal posterior pituitary"
        ),
        "disease_pathway": (
            "POU1F1 PITUITARY TRANSCRIPTION FACTOR CASCADE: "
            "POU1F1 activates: GH1 → somatotroph → Growth Hormone → IGF-1 → linear growth. "
            "POU1F1 activates: PRL → lactotroph → Prolactin → lactation. "
            "POU1F1 activates: TSHB → thyrotroph → TSH → thyroid T3/T4. "
            "LOF CONSEQUENCE: All three cell types hypoplastic/absent → "
            "GH ABSENT → growth failure, hypoglycaemia, delayed bone age. "
            "PRL ABSENT → no lactation (detected in females post-partum). "
            "TSH ABSENT → central hypothyroidism → delayed myelination, poor cognition. "
            "PRESERVED AXES: LH/FSH (gonadotrophs survive) → puberty NORMAL; "
            "ACTH (corticotrophs survive) → cortisol NORMAL → NO adrenal crisis; "
            "ADH (hypothalamic, not pituitary) → water balance NORMAL. "
            "BIOCHEMISTRY: GH LOW/absent (GH stimulation test <10 mU/L); "
            "IGF-1 LOW for age; PRL LOW; TSH LOW/inappropriately normal with LOW FT4; "
            "LH/FSH NORMAL; ACTH/cortisol NORMAL; ADH NORMAL (no DI)."
        ),
        "pathognomonic": (
            "POU1F1 CLINICAL PEARLS: "
            "PATHOGNOMONIC TRIPLE: GH + PRL + TSH deficiency with NORMAL LH/FSH/ACTH — "
            "if you see this combination, POU1F1 (or PROP1 early phase) is the cause. "
            "NEONATAL HYPOGLYCAEMIA: GH deficiency + central hypothyroidism → "
            "neonatal hypoglycaemia (GH needed for gluconeogenesis) without adrenal crisis "
            "(ACTH intact) → MRI pituitary MANDATORY in unexplained neonatal hypoglycaemia. "
            "GROWTH FAILURE is often the presenting feature in childhood. "
            "CENTRAL HYPOTHYROIDISM: TSH LOW (or inappropriately normal) + FREE T4 LOW → "
            "treat with levothyroxine (NOT based on TSH monitoring — TSH unreliable in central hypothyroidism; "
            "monitor by FT4 levels and clinical parameters). "
            "PROLACTIN: Low PRL in a lactating female is a clue to POU1F1 or PROP1. "
            "POSTERIOR PITUITARY NORMAL: T1 bright spot normally positioned — helps distinguish "
            "from GLI2/LHX4/HESX1 where ectopic posterior pituitary is common. "
            "FOUNDER MUTATIONS: p.Arg271Trp (dominant-negative, European); "
            "p.Trp261Cys, p.Pro76Arg (AR, recessive). "
            "OMIM Gene: 173110, Disease: CPHD1 — 613038."
        ),
        "treatment": (
            "GH REPLACEMENT: Recombinant human GH (rhGH) 0.025-0.05 mg/kg/day SC; "
            "monitor IGF-1 (target mid-normal range for age/sex); "
            "continue until near-adult height then reassess GH axis in adulthood. "
            "LEVOTHYROXINE: T4 replacement for central hypothyroidism; "
            "start BEFORE GH (GH accelerates T4 clearance; untreated hypothyroidism impairs GH response); "
            "dose guide by FT4 (NOT TSH — TSH unreliable in central hypothyroidism). "
            "NO FLUDROCORTISONE: ACTH/cortisol intact — fludrocortisone is NEVER needed. "
            "NO ROUTINE SEX HORMONE REPLACEMENT: LH/FSH intact → puberty will occur normally; "
            "ONLY needed if secondary gonadotropin failure develops (rare in POU1F1). "
            "NO DESMOPRESSIN: ADH intact — no diabetes insipidus. "
            "MONITORING: Auxology quarterly; IGF-1 + FT4 6-monthly; GH provocation test to confirm. "
            "GENETIC COUNSELLING: AR form: 25% recurrence; AD-DN: may appear de novo or AD."
        ),
        "key_features": [
            "CPHD1: GH + PRL + TSH triple deficiency (somatotrophs + lactotrophs + thyrotrophs lost)",
            "SPARED: LH/FSH/ACTH/ADH NORMAL — PATHOGNOMONIC of POU1F1 (DDx PROP1 which loses gonadotropins)",
            "Anterior pituitary hypoplasia; NORMAL posterior pituitary position (T1 bright spot in situ)",
            "Neonatal hypoglycaemia (GH + TSH absent) WITHOUT adrenal crisis (ACTH intact)",
            "Central hypothyroidism: FT4 LOW + TSH LOW/inappropriately normal — monitor FT4 NOT TSH",
            "Treatment: rhGH + levothyroxine ONLY (no fludrocortisone, no sex hormones, no desmopressin)",
            "Dominant-negative mutations (p.Arg271Trp) → AD phenotype; recessive: biallelic LOF",
            "Puberty NORMAL (gonadotropins intact); fertility possible",
        ],
        "key_ddx": (
            "POU1F1 vs PROP1: PROP1 loses LH/FSH too (delayed/absent puberty); POU1F1 spares gonadotropins; "
            "PROP1 can develop evolving ACTH loss in adulthood; POU1F1 ACTH always intact; "
            "POU1F1 vs HESX1: HESX1 has SOD (optic nerve hypoplasia + absent septum pellucidum); "
            "HESX1 can be ACTH deficient (DANGEROUS); POU1F1 ACTH intact; "
            "POU1F1 vs LHX3: LHX3 has RIGID NECK (cannot rotate) PATHOGNOMONIC; "
            "POU1F1 vs LHX4: LHX4 has ACTH deficiency early + Chiari malformation; "
            "POU1F1 vs GLI2: GLI2 has HPE spectrum + single central incisor; variable expressivity"
        ),
        "systemic_involvement": {
            "pituitary": "Anterior hypoplasia (somatotrophs/lactotrophs/thyrotrophs absent); posterior normal",
            "thyroid": "Central hypothyroidism (TSH deficient) — levothyroxine required",
            "growth": "GH deficiency → growth failure, hypoglycaemia, delayed bone age",
            "reproductive": "NORMAL — LH/FSH intact; puberty occurs; fertility possible",
            "adrenal": "NORMAL — ACTH intact; no adrenal crisis risk",
        },
        "cascade_testing": (
            "AR form: siblings 25% affected; parents obligate carriers; molecular POU1F1 sequencing; "
            "AD-DN form: parent testing (may appear de novo — p.Arg271Trp is recurrent); "
            "prenatal: molecular diagnosis available if family variant known; "
            "newborn screening does not detect CPHD — clinical vigilance for neonatal hypoglycaemia required"
        ),
        "emergency_protocol": (
            "NEONATAL HYPOGLYCAEMIA (GH + TSH absent): "
            "IV glucose 10% 2 mL/kg bolus + infusion; GH level (LOW) + TSH + FT4 (LOW); "
            "pituitary MRI mandatory; do NOT give hydrocortisone empirically "
            "(ACTH intact in POU1F1 — unlike HESX1/LHX4 where ACTH can be absent); "
            "CONFIRM ACTH/cortisol NORMAL before withholding hydrocortisone; "
            "start levothyroxine FIRST (before rhGH to prevent accelerated T4 clearance)."
        ),
    },
    # -- PROP1 -- Prophet of PIT-1 -- CPHD2 -------------------------------------------------------
    {
        "gene": "PROP1",
        "alt_name": (
            "PROP1 (PROP1-226aa-5q35.3 / AR -- "
            "Prophet-of-PIT-1 -- "
            "CPHD2-MOST-COMMON-Cause-of-CPHD-Globally -- "
            "GH+TSH+PRL+LH+FSH-ALL-DEFICIENT -- "
            "ACTH-Evolves-LATER-Adulthood-Adrenal-Crisis-Risk -- "
            "Transient-Pituitary-Mass-Then-Involution-Characteristic-MRI)"
        ),
        "protein": (
            "PROP1 -- 5q35.3 AR -- PROP1-226aa -- "
            "Prophet-of-PIT-1-Paired-Like-Homeodomain-TF-26kDa-Nuclear -- "
            "Required-for-POU1F1-Expression-AND-Gonadotroph-Differentiation -- "
            "OMIM-Gene-601538-Disease-CPHD2-262600"
        ),
        "locus": "5q35.3",
        "protein_size": "226 aa / ~26 kDa",
        "inheritance": (
            "Autosomal recessive. PROP1 encodes Prophet of PIT-1, a paired-like homeodomain "
            "transcription factor that is upstream of POU1F1 in the pituitary development cascade. "
            "PROP1 is required for: (1) activation of POU1F1 transcription → differentiation of "
            "somatotrophs, lactotrophs, thyrotrophs (GH/PRL/TSH); and (2) differentiation of "
            "gonadotrophs (LH/FSH). "
            "BROADER DEFICIENCY THAN POU1F1: Because PROP1 is upstream of POU1F1, PROP1 mutations "
            "affect EVERYTHING POU1F1 affects PLUS gonadotropins → GH + TSH + PRL + LH + FSH all deficient. "
            "ACTH/CORTISOL: Variable and LATE-ONSET — corticotroph differentiation is partly PROP1-dependent "
            "but less so; ACTH deficiency develops in many patients during adulthood (age 20-50y), "
            "making PROP1 a progressive disease; adrenal crisis risk increases over time. "
            "PITUITARY MASS: Characteristic MRI finding — transient ENLARGEMENT of the pituitary "
            "in childhood/adolescence (appears as mass lesion, can raise concern for pituitary tumour), "
            "followed by INVOLUTION over years; the mass is composed of apoptotic/hyperplastic remnant "
            "pituitary tissue. This evolution is PATHOGNOMONIC of PROP1. "
            "FOUNDER MUTATIONS: del301-302 (intronic deletion, most common worldwide ~53%); "
            "p.Arg120Cys (European populations); p.Ser167Ile (isolated populations). "
            "MOST COMMON CAUSE OF CPHD globally — accounts for 50% of familial CPHD cases."
        ),
        "disease_category": (
            "Combined Pituitary Hormone Deficiency type 2 (CPHD2, OMIM #262600); "
            "AR; MOST COMMON genetic cause of CPHD; incidence ~1/8,000-10,000; "
            "GH + TSH + PRL + LH + FSH deficient from early childhood; "
            "ACTH deficiency evolves in adulthood (50-75% of patients by age 40); "
            "characteristic transient pituitary enlargement then involution on MRI"
        ),
        "disease_pathway": (
            "PROP1 CASCADE FAILURE: "
            "PROP1 activates POU1F1 → somatotroph/lactotroph/thyrotroph differentiation LOST. "
            "PROP1 directly promotes gonadotroph differentiation → LH/FSH cells also LOST. "
            "GH AXIS: GH absent → IGF-1 low → growth failure from infancy/childhood. "
            "THYROID AXIS: TSH absent → FT4 low → central hypothyroidism (same as POU1F1). "
            "GONADOTROPIN AXIS: LH + FSH absent → delayed/absent puberty; "
            "females: primary amenorrhoea or no breast development; "
            "males: micropenis at birth (LH-dependent fetal testosterone), undescended testes, no virilisation. "
            "PROLACTIN: Absent (same as POU1F1). "
            "PROGRESSIVE ACTH LOSS: Corticotroph function declines in adulthood → "
            "cortisol insufficient → adrenal crisis during illness/surgery — "
            "CRITICAL: PROP1 patients need regular cortisol axis testing from adolescence onwards. "
            "PITUITARY MASS EVOLUTION: Childhood MRI shows enlarged pituitary (looks like adenoma); "
            "involution occurs spontaneously; end-stage: small or empty sella."
        ),
        "pathognomonic": (
            "PROP1 CLINICAL PEARLS: "
            "MOST COMMON CPHD GENE GLOBALLY — del301-302 is the most common worldwide mutation (~53%). "
            "EVOLVING ACTH LOSS: The CRITICAL management issue — begin cortisol axis testing from age 15-20y; "
            "stimulated cortisol <500 nmol/L after Synacthen → start hydrocortisone; "
            "adrenal crisis can be FATAL if ACTH loss is not anticipated; "
            "every PROP1 patient must carry IM hydrocortisone kit once ACTH status becomes uncertain. "
            "PITUITARY MASS: Do NOT biopsy — it involutes spontaneously; "
            "distinguish from pituitary adenoma by: history of panhypopituitarism + PROP1 mutation + "
            "longitudinal imaging showing involution. "
            "DELAYED PUBERTY: LH/FSH absent → puberty induction mandatory (oestrogen in females from ~13y, "
            "testosterone in males from ~13y) — if not treated, osteoporosis and psychosocial impact. "
            "FOUNDER MUTATIONS enable specific PCR-based rapid testing in high-prevalence populations. "
            "OMIM Gene: 601538, Disease: CPHD2 — 262600."
        ),
        "treatment": (
            "GH REPLACEMENT: rhGH 0.025-0.05 mg/kg/day SC; IGF-1 monitoring. "
            "LEVOTHYROXINE: For central hypothyroidism; start before GH; guide by FT4. "
            "SEX HORMONE INDUCTION: Oestradiol (females, ~13y, incremental doses); "
            "testosterone (males, ~13y, incremental doses); combined HRT lifelong. "
            "HYDROCORTISONE: When ACTH axis fails (monitor annually from adolescence); "
            "10-15 mg/m²/day in divided doses; stress dosing MANDATORY once ACTH deficient; "
            "IM hydrocortisone kit + sick-day rules + medical alert bracelet. "
            "DESMOPRESSIN: NOT needed (ADH/posterior pituitary typically intact). "
            "MONITORING SCHEDULE: Annual cortisol Synacthen test from age 15-20y; "
            "if stimulated cortisol declining → pre-empt with hydrocortisone; "
            "annual MRI (first 5-10 years to document involution of mass). "
            "FERTILITY: GnRH pump or gonadotropin injections for fertility if desired."
        ),
        "key_features": [
            "MOST COMMON genetic CPHD globally — del301-302 founder mutation ~53% of alleles worldwide",
            "GH + TSH + PRL + LH + FSH ALL deficient; ACTH loss is PROGRESSIVE in adulthood",
            "Transient pituitary ENLARGEMENT in childhood then INVOLUTION — characteristic MRI PATHOGNOMONIC",
            "Evolving ACTH deficiency: annual Synacthen test mandatory from adolescence — adrenal crisis risk",
            "LH/FSH absent → delayed puberty → MANDATORY sex hormone induction",
            "del301-302 / p.Arg120Cys founder mutations allow rapid targeted PCR testing",
            "Treatment: GH + levothyroxine + sex hormones + hydrocortisone (when ACTH fails)",
            "Pituitary mass: do NOT biopsy — involutes spontaneously; diagnose by genetics + evolution",
        ],
        "key_ddx": (
            "PROP1 vs POU1F1: POU1F1 spares LH/FSH + ACTH always intact; PROP1 loses LH/FSH + ACTH evolves; "
            "PROP1 vs LHX3: LHX3 has RIGID NECK; both lose GH/TSH/LH/FSH; LHX3 no evolving ACTH; "
            "PROP1 vs LHX4: LHX4 has ACTH early (not late) + Chiari + ectopic PP; "
            "PROP1 pituitary mass vs adenoma: genetics + longitudinal involution; "
            "PROP1 vs multiple other CPHD: molecular panel required"
        ),
        "systemic_involvement": {
            "pituitary": "Transient enlargement then involution; end-stage small/empty sella",
            "thyroid": "Central hypothyroidism (TSH deficient) — levothyroxine mandatory",
            "growth": "Severe GH deficiency; growth failure if untreated",
            "reproductive": "LH/FSH absent — delayed/absent puberty; osteoporosis risk; fertility requires gonadotropins",
            "adrenal": "Evolving ACTH deficiency in adulthood → adrenal crisis if not monitored/treated",
        },
        "cascade_testing": (
            "Siblings: 25% affected (AR); del301-302 founder testing in European/middle Eastern populations; "
            "p.Arg120Cys in European families; "
            "all first-degree relatives need PROP1 sequencing if proband found; "
            "IMPORTANT: Even carriers are at risk for partial hypopituitarism (variable penetrance in some families)"
        ),
        "emergency_protocol": (
            "ADRENAL CRISIS (once ACTH loss develops): "
            "IV hydrocortisone 50-100 mg/m² IV IMMEDIATELY (do not wait for cortisol result); "
            "0.9% NaCl + 10% glucose if hypoglycaemia; "
            "PRECAUTION: ALL PROP1 patients with unknown ACTH status undergoing surgery/illness "
            "should receive empiric stress-dose hydrocortisone until axis confirmed intact; "
            "IM hydrocortisone kit at home from late adolescence; "
            "CHECK ACTH STATUS: Annual Synacthen test from age 15-20y."
        ),
    },
    # -- HESX1 -- Septo-Optic Dysplasia -----------------------------------------------------------
    {
        "gene": "HESX1",
        "alt_name": (
            "HESX1 (HESX1-185aa-3p14.3 / AR-severe-AD-milder -- "
            "Homeobox-Expressed-in-ES-Cells-1 -- "
            "Septo-Optic-Dysplasia-SOD-de-Morsier-Syndrome -- "
            "CLASSIC-TRIAD-Optic-Nerve-Hypoplasia+Absent-Septum+Pituitary-Hypoplasia -- "
            "Pendular-Nystagmus-First-Clinical-Sign)"
        ),
        "protein": (
            "HESX1 -- 3p14.3 AR/AD -- HESX1-185aa -- "
            "Paired-Like-Homeodomain-Transcription-Factor-21kDa-Nuclear-Early-Pituitary-Eye-Brain-Development -- "
            "Represses-Anterior-Visceral-Endoderm-Marker-Genes-Permits-Anterior-Neural-Development -- "
            "OMIM-Gene-601802-Disease-SOD-182230"
        ),
        "locus": "3p14.3",
        "protein_size": "185 aa / ~21 kDa",
        "inheritance": (
            "Autosomal recessive (homozygous mutations → severe, classic SOD) or "
            "autosomal dominant (heterozygous mutations → variable, often milder SOD). "
            "HESX1 encodes a paired-like homeodomain transcription factor essential for early "
            "development of anterior pituitary, optic vesicles, and the forebrain midline structures. "
            "It acts as a transcriptional repressor during early embryogenesis, particularly "
            "in the anterior visceral endoderm and the prospective pituitary placode. "
            "SEPTO-OPTIC DYSPLASIA (SOD/de Morsier syndrome, OMIM #182230): "
            "The clinical entity is defined by the classic TRIAD: "
            "(1) Optic nerve hypoplasia (ONH) — bilateral or unilateral small optic nerves; "
            "(2) Absent septum pellucidum — midline brain structure absent on MRI; "
            "(3) Pituitary hypoplasia — with variable anterior pituitary hormone deficiencies. "
            "IMPORTANT: ALL THREE FEATURES are present in only ~30% of SOD patients; "
            "presence of TWO features qualifies for SOD diagnosis; "
            "HESX1 mutations explain only ~1% of SOD cases (most SOD is environmental/multifactorial). "
            "VARIABLE EXPRESSIVITY: Even within the same family, expressivity varies; "
            "AR mutations produce more complete/severe SOD; AD mutations may produce only one feature. "
            "ECTOPIC POSTERIOR PITUITARY (EPP): T1 bright spot in ectopic position along the "
            "pituitary stalk or at the median eminence — associated with pituitary stalk interruption."
        ),
        "disease_category": (
            "Septo-optic dysplasia / de Morsier syndrome (OMIM #182230); AR or AD; "
            "incidence SOD overall ~1/10,000; HESX1 mutations in ~1% of SOD; "
            "GH most commonly deficient; ACTH deficiency occurs and is DANGEROUS (can be fatal); "
            "variable combination of anterior pituitary deficiencies"
        ),
        "disease_pathway": (
            "HESX1 DEVELOPMENTAL PATHWAY: "
            "HESX1 is expressed in the anterior visceral endoderm and the pituitary placode "
            "during early embryogenesis; it represses Hex gene and permits anterior neural "
            "plate identity to form properly. "
            "OPTIC VESICLE: HESX1 required for optic cup and optic nerve development → "
            "LOF → optic nerve hypoplasia → reduced visual acuity, pendular nystagmus, "
            "colour vision defects; visual prognosis variable. "
            "MIDLINE FOREBRAIN: HESX1 needed for septum pellucidum and other midline structures → "
            "LOF → absent septum pellucidum on MRI (space between frontal horns is continuous). "
            "PITUITARY PLACODE: HESX1 promotes proper Rathke's pouch development → "
            "LOF → anterior pituitary hypoplasia → variable hormone deficiencies. "
            "GH most commonly affected; then TSH, ACTH, LH/FSH in varying combinations. "
            "CORTISOL AXIS: ACTH deficiency occurs in HESX1 — this is the MOST DANGEROUS axis "
            "because central adrenal insufficiency is life-threatening and can be missed "
            "(no biochemical alarm unless specifically tested or until crisis). "
            "MRI FINDINGS: Absent septum pellucidum, small/hypoplastic optic nerves/chiasm, "
            "small anterior pituitary, ectopic posterior pituitary (EPP), hypoplastic infundibulum."
        ),
        "pathognomonic": (
            "HESX1/SOD CLINICAL PEARLS: "
            "NYSTAGMUS IN A NEWBORN: Pendular nystagmus from birth is often the FIRST clinical sign "
            "(optic nerve hypoplasia → poor visual fixation → nystagmus); "
            "triggers ophthalmic + pituitary workup; do NOT diagnose as 'benign nystagmus' without MRI. "
            "CLASSIC SOD TRIAD on MRI: Optic nerve hypoplasia + absent septum pellucidum + pituitary hypoplasia; "
            "EPP (ectopic bright spot off the stalk or median eminence) is highly suggestive. "
            "ACTH DEFICIENCY IS MOST DANGEROUS: Central adrenal insufficiency in SOD patients "
            "can be missed — morning cortisol >450 nmol/L is reassuring but does not exclude partial "
            "deficiency under stress; Synacthen test is MANDATORY in ALL SOD patients; "
            "ALL SOD/HESX1 patients should carry stress dosing protocols and IM hydrocortisone. "
            "SCHOOL DIFFICULTIES: Absent septum pellucidum + visual impairment → learning difficulties; "
            "early ophthalmology, neurology, and educational support needed. "
            "LOW-INCIDENCE GENETIC CAUSE: Most SOD (>99%) is not HESX1 genetic; "
            "HESX1 molecular testing warranted if family history or bilateral symmetric presentation. "
            "OMIM Gene: 601802, Disease: SOD — 182230."
        ),
        "treatment": (
            "GH REPLACEMENT: rhGH if GH-deficient; dose as per standard CPHD; "
            "growth response is variable given associated brain anomalies. "
            "LEVOTHYROXINE: If TSH deficient; guide by FT4; start before GH. "
            "HYDROCORTISONE: MANDATORY workup for ACTH deficiency; "
            "if Synacthen-stimulated cortisol <500 nmol/L → start HC 10-15 mg/m²/day; "
            "stress dosing + IM kit MANDATORY; medical alert bracelet. "
            "SEX HORMONES: If LH/FSH deficient; puberty induction as needed. "
            "DESMOPRESSIN: If diabetes insipidus (central DI occurs in SOD — assess ADH axis). "
            "OPHTHALMOLOGY: Visual monitoring, glasses, possibly low-vision aids; "
            "patching if amblyopia risk; regular visual acuity testing. "
            "NEURODEVELOPMENTAL SUPPORT: Learning support; speech therapy if needed."
        ),
        "key_features": [
            "SOD/de Morsier syndrome: optic nerve hypoplasia + absent septum pellucidum + pituitary hypoplasia",
            "Pendular nystagmus at birth — often FIRST clinical sign (optic nerve hypoplasia)",
            "ALL THREE SOD features in only ~30% — TWO features suffices for SOD diagnosis",
            "ACTH deficiency MOST DANGEROUS — Synacthen test MANDATORY in all SOD patients",
            "Ectopic posterior pituitary (EPP) on MRI — T1 bright spot off-stalk or at median eminence",
            "HESX1 mutations explain ~1% of SOD — most SOD is environmental/multifactorial",
            "GH most commonly deficient; variable ACTH, TSH, LH/FSH, ADH combination",
            "ALL SOD patients: IM hydrocortisone kit + sick-day rules regardless of ACTH baseline",
        ],
        "key_ddx": (
            "HESX1-SOD vs OTX2: OTX2 has SEVERE eye structural anomalies (anophthalmia/microphthalmia) not just ONH; "
            "OTX2 septum pellucidum usually present; "
            "HESX1-SOD vs POU1F1: POU1F1 has NO eye anomalies, NO midline brain anomalies; "
            "HESX1 vs environmental SOD: family history + bilateral + symmetric ONH → test HESX1; "
            "SOD vs septo-optic-pituitary anomaly: full SOD triad required; "
            "HESX1 ACTH-deficient vs POU1F1: POU1F1 ACTH always intact"
        ),
        "systemic_involvement": {
            "visual": "Optic nerve hypoplasia → reduced visual acuity, colour vision loss, nystagmus",
            "brain": "Absent septum pellucidum; variable cortical anomalies; midline defects",
            "pituitary": "Variable hormone deficiencies: GH most common; ACTH critical; EPP MRI",
            "adrenal": "ACTH deficiency → central adrenal insufficiency — life-threatening",
            "water_balance": "Central DI can occur (ADH axis may be affected in some cases)",
        },
        "cascade_testing": (
            "AR form (symmetric, bilateral ONH, complete triad): siblings 25% risk; parental carrier testing; "
            "AD form (milder, variable): first-degree relatives; "
            "NOTE: most SOD is NOT genetic — prenatal alcohol, cytomegalovirus, "
            "young maternal age are known environmental triggers; "
            "HESX1 panel in familial/bilateral/severe cases"
        ),
        "emergency_protocol": (
            "ADRENAL CRISIS (central adrenal insufficiency — ALL SOD patients at risk): "
            "IV hydrocortisone 50-100 mg/m² IV IMMEDIATELY (do NOT wait for cortisol result); "
            "0.9% NaCl 20 mL/kg IV bolus; 10% glucose for hypoglycaemia; "
            "IMPORTANT: ALL SOD patients regardless of baseline ACTH status should be treated "
            "as potentially ACTH-insufficient during major illness/surgery; "
            "stress-dose protocol and IM hydrocortisone kit at home are MANDATORY for ALL SOD patients."
        ),
    },
    # -- OTX2 -- Pituitary Hypoplasia + Severe Eye Anomalies ------------------------------------
    {
        "gene": "OTX2",
        "alt_name": (
            "OTX2 (OTX2-289aa-14q22.3 / AD-haploinsufficiency -- "
            "Orthodenticle-Homeobox-2 -- "
            "Pituitary-Hypoplasia-With-Severe-Eye-Anomalies -- "
            "Anophthalmia-Microphthalmia-Coloboma-Retinal-Dystrophy -- "
            "Eye-Disease-OFTEN-More-Severe-Than-Pituitary)"
        ),
        "protein": (
            "OTX2 -- 14q22.3 AD-haploinsufficiency -- OTX2-289aa -- "
            "Orthodenticle-Homeobox-2-32kDa-Bicoid-Class-Homeodomain-TF-Nuclear -- "
            "Required-Eye-Brain-Pituitary-Retinal-Development-Early-Embryogenesis -- "
            "OMIM-Gene-600037-Disease-PHPX-MCOPCB2-610125"
        ),
        "locus": "14q22.3",
        "protein_size": "289 aa / ~32 kDa",
        "inheritance": (
            "Autosomal dominant (haploinsufficiency); heterozygous loss-of-function mutations cause disease. "
            "OTX2 encodes Orthodenticle Homeobox 2, a bicoid-class homeodomain transcription factor "
            "expressed in the anterior brain, eye cup, retinal pigment epithelium, pineal gland, "
            "and anterior pituitary during embryogenesis. "
            "EYE DEVELOPMENT: OTX2 is critical for optic cup morphogenesis, retinal differentiation, "
            "and formation of the RPE (retinal pigment epithelium). "
            "Heterozygous LOF → anophthalmia (absent eye), microphthalmia (small eye), coloboma "
            "(failure of optic fissure closure), retinal dystrophy. "
            "The eye anomalies are often the MOST CLINICALLY PROMINENT feature — diagnosed at birth "
            "or prenatally; may overshadow the pituitary disease. "
            "PITUITARY: OTX2 regulates pituitary development at multiple steps; "
            "GH deficiency is most common (and often the first pituitary hormone to be tested); "
            "variable TSH, LH/FSH deficiency; ectopic posterior pituitary on MRI. "
            "BRAIN: Intellectual disability occurs in some patients (OTX2 role in cerebellar/cortical development); "
            "cerebellar hypoplasia in severe cases. "
            "GENOTYPE-PHENOTYPE: Missense vs null alleles show some correlation; "
            "even within the same family with identical mutations, phenotype varies (variable expressivity)."
        ),
        "disease_category": (
            "Pituitary hormone deficiency with ocular anomalies (OMIM #610125, PHPX/MCOPCB2); "
            "AD haploinsufficiency; structural eye anomalies dominant feature at birth; "
            "GH deficiency most common pituitary defect; variable TSH/gonadotropin deficiency; "
            "ectopic posterior pituitary common"
        ),
        "disease_pathway": (
            "OTX2 DUAL ROLE — EYE AND PITUITARY: "
            "EYE: OTX2 drives optic cup differentiation from neuroepithelium → "
            "haploinsufficiency → anophthalmia/microphthalmia/coloboma at birth; "
            "retinal pigment epithelium fails → retinal dystrophy; "
            "photoreceptor differentiation impaired (OTX2 drives CRX expression in rods/cones). "
            "PITUITARY: OTX2 activates HESX1 and other pituitary development genes → "
            "anterior pituitary placode specification impaired → "
            "pituitary hypoplasia → GH +/- TSH +/- LH/FSH deficiency. "
            "MRI FINDINGS: Anterior pituitary hypoplasia; ectopic posterior pituitary (EPP); "
            "hypoplastic infundibulum; in severe cases cerebellar hypoplasia. "
            "RETINAL DYSTROPHY: Photoreceptor loss over time (rod-cone dystrophy pattern) → "
            "progressive visual loss even in the non-anophthalmic eye. "
            "INTELLECTUAL DISABILITY: When present, reflects OTX2 role in cerebellar granule cell "
            "migration and cortical maturation."
        ),
        "pathognomonic": (
            "OTX2 CLINICAL PEARLS: "
            "EYE ANOMALY AT BIRTH IS THE ENTRY POINT: "
            "Anophthalmia/microphthalmia/coloboma detected at birth or on prenatal ultrasound → "
            "ALWAYS test for OTX2 (and other eye genes: SOX2, PAX6, CHX10) + pituitary function. "
            "PITUITARY DISEASE SECONDARY: GH deficiency may be missed because clinicians focus on "
            "eye management; MANDATORY to check full pituitary function in any child with OTX2 mutation. "
            "RETINAL DYSTROPHY PROGRESSION: Even if eye structurally present, retinal dystrophy "
            "may cause progressive visual loss — annual ophthalmology review. "
            "VISUAL REHABILITATION: Low vision aids, orientation and mobility training are the primary "
            "morbidity management (visual impairment is the dominant clinical challenge). "
            "INTELLECTUAL DISABILITY: Screen at diagnosis; early intervention improves outcomes. "
            "ECTOPIC POSTERIOR PITUITARY on MRI → expect multiple anterior pituitary deficiencies; "
            "test all axes (GH, TSH, LH/FSH, ACTH, ADH). "
            "OMIM Gene: 600037, Disease: PHPX — 610125."
        ),
        "treatment": (
            "GH REPLACEMENT: rhGH if GH-deficient; monitor IGF-1; benefits linear growth. "
            "LEVOTHYROXINE: If TSH deficient; FT4-guided. "
            "HYDROCORTISONE: If ACTH deficient (less common than in HESX1 but must be checked); "
            "Synacthen test in all OTX2 patients. "
            "SEX HORMONES: If LH/FSH deficient; puberty induction as needed. "
            "OPHTHALMOLOGY: Primary focus — prosthetic eye (if anophthalmic), low-vision aids, "
            "retinal dystrophy monitoring (ERG), refractive correction; "
            "annual retinal review for dystrophy progression; "
            "genetic counselling re: retinal gene therapy prospects (OTX2 not yet amenable). "
            "NEURODEVELOPMENTAL: If ID present — early intervention, educational support."
        ),
        "key_features": [
            "Severe structural eye anomalies (anophthalmia/microphthalmia/coloboma/retinal dystrophy) often DOMINANT feature",
            "AD haploinsufficiency — heterozygous LOF sufficient for severe phenotype",
            "GH deficiency most common pituitary defect; variable TSH/LH/FSH; ACTH less common",
            "MANDATORY: full pituitary function testing in ALL patients with OTX2 mutations",
            "Ectopic posterior pituitary (EPP) on MRI — T1 bright spot at ectopic position",
            "Intellectual disability can occur (OTX2 role in cerebellar/cortical development)",
            "Variable expressivity — even within families; same mutation can give anophthalmia vs coloboma",
            "Ophthalmology is primary morbidity — annual retinal review mandatory",
        ],
        "key_ddx": (
            "OTX2 vs HESX1: HESX1 has ONH (small but present optic nerves); absent septum pellucidum; "
            "OTX2 has SEVERE structural eye anomalies (anophthalmia/microphthalmia) not typical of HESX1; "
            "OTX2 vs SOX2: SOX2 also causes anophthalmia + GH deficiency; molecular panel required; "
            "SOX2 mutations cause more severe bilateral anophthalmia; both AD; "
            "OTX2 isolated GH deficiency vs IGHD1 (GH1): OTX2 has EYE ANOMALY ALWAYS; "
            "OTX2 vs POU1F1: POU1F1 has no eye anomalies; normal posterior pituitary"
        ),
        "systemic_involvement": {
            "visual": "Anophthalmia/microphthalmia/coloboma at birth; retinal dystrophy (progressive)",
            "pituitary": "GH most common; variable TSH/LH/FSH; ectopic PP; hypoplastic infundibulum",
            "brain": "Intellectual disability in subset; cerebellar hypoplasia in severe cases",
            "adrenal": "ACTH deficiency less common but must be excluded; Synacthen test required",
            "growth": "GH deficiency → growth failure if untreated",
        },
        "cascade_testing": (
            "AD — first-degree relatives; parent testing (de novo mutations common ~50%); "
            "OTX2 sequencing + MLPA (deletions of 14q22-23 region); "
            "eye examination of parents (variable expressivity — parent may have subtle coloboma); "
            "prenatal: molecular diagnosis available if family variant known; "
            "ultrasound 20 weeks: microphthalmia may be detectable"
        ),
        "emergency_protocol": (
            "HYPOGLYCAEMIA (GH + possible ACTH deficiency): "
            "IV glucose 10% 2 mL/kg bolus; "
            "give empiric stress-dose hydrocortisone if ACTH status unknown; "
            "CHECK: if Synacthen test has been performed and cortisol is adequate, "
            "hydrocortisone is only needed for stress doses; "
            "SURGICAL PRECAUTION: Cover all surgical procedures with stress-dose HC "
            "until full ACTH axis characterisation is documented."
        ),
    },
    # -- SOX3 -- X-Linked Hypopituitarism + Intellectual Disability ----------------------------
    {
        "gene": "SOX3",
        "alt_name": (
            "SOX3 (SOX3-446aa-Xq27.1 / XLR -- "
            "SRY-Box-Transcription-Factor-3 -- "
            "X-Linked-Hypopituitarism-Intellectual-Disability -- "
            "MRXHF1-OMIM-300123 -- "
            "Infundibular-Hypoplasia-on-MRI -- "
            "Poly-Alanine-Expansion-Contraction-Mutations)"
        ),
        "protein": (
            "SOX3 -- Xq27.1 XLR -- SOX3-446aa -- "
            "SRY-Box-B-HMG-Domain-TF-50kDa-Nuclear-X-Chromosome -- "
            "Expressed-Hypothalamo-Pituitary-Axis-Infundibulum-Anterior-Pituitary -- "
            "OMIM-Gene-313430-Disease-MRXHF1-300123"
        ),
        "locus": "Xq27.1",
        "protein_size": "446 aa / ~50 kDa",
        "inheritance": (
            "X-linked recessive (XLR). Males are primarily affected (hemizygous loss-of-function); "
            "females are typically carriers with variable (often mild or absent) expression. "
            "SOX3 encodes SRY-Box Transcription Factor 3, a member of the SOX (SRY-related HMG-box) "
            "family expressed in the developing hypothalamus, infundibulum, and anterior pituitary. "
            "SOX3 is required for: (1) proper infundibular (pituitary stalk) development; "
            "(2) differentiation/maintenance of anterior pituitary cell lineages; "
            "(3) hypothalamic neuronal development related to pituitary regulation. "
            "MUTATIONS: "
            "(a) Poly-alanine expansion mutations: expansion of poly-Ala tract within SOX3 "
            "leads to protein misfolding/dysfunction; similar mechanism to HOXD13, PHOX2B, RUNX2 expansions; "
            "(b) Poly-alanine contraction: paradoxically also causes disease (haploinsufficiency); "
            "(c) Point mutations disrupting HMG domain. "
            "INTELLECTUAL DISABILITY: A key feature — cognitive impairment in males (IQ typically 50-80, "
            "moderate range) alongside the endocrine phenotype; "
            "SOX3 role in hypothalamic-pituitary neuronal wiring explains the combined endocrine + ID phenotype. "
            "INFUNDIBULAR HYPOPLASIA: Characteristic MRI finding — hypoplastic pituitary stalk "
            "(infundibulum narrowed/absent); pituitary may be small."
        ),
        "disease_category": (
            "X-linked hypopituitarism with intellectual disability (MRXHF1, OMIM #300123); "
            "XLR; males affected; GH deficiency most common; variable other pituitary deficiencies; "
            "intellectual disability common in affected males; infundibular hypoplasia on MRI"
        ),
        "disease_pathway": (
            "SOX3 INFUNDIBULAR PATHWAY: "
            "SOX3 expressed in infundibular stalk precursors during embryogenesis → "
            "LOF → infundibular hypoplasia → impaired hypothalamic-pituitary signalling → "
            "GH deficiency (most common); variable TSH, LH/FSH loss. "
            "COGNITIVE PATHWAY: SOX3 expressed in hypothalamic neurons and cortical precursors → "
            "LOF → moderate intellectual disability in males; "
            "females: X-inactivation may protect, but some carrier females have mild ID. "
            "HORMONAL DEFICIENCIES: GH deficiency most common (~80% of affected males); "
            "TSH deficiency variable; LH/FSH deficiency common (delayed puberty in males); "
            "ACTH: may be mildly impaired but frank adrenal crisis less typical than HESX1/LHX4. "
            "MRI FINDINGS: Hypoplastic pituitary stalk (infundibular hypoplasia) + "
            "small anterior pituitary +/- ectopic posterior pituitary."
        ),
        "pathognomonic": (
            "SOX3 CLINICAL PEARLS: "
            "X-LINKED INHERITANCE: Son-to-son transmission NEVER occurs (X-linked); "
            "affected males in maternal lineage; female carriers rarely affected. "
            "INTELLECTUAL DISABILITY PLUS ENDOCRINE: The combination of moderate ID + GH deficiency "
            "in a male from maternal lineage → think SOX3 FIRST (before other CPHD genes). "
            "INFUNDIBULAR HYPOPLASIA: MRI finding characteristic — look at the pituitary stalk "
            "specifically; narrow stalk in male with ID + GH deficiency = SOX3 until proven otherwise. "
            "POLY-ALANINE EXPANSION/CONTRACTION: Standard Sanger sequencing of coding region may miss "
            "poly-Ala length changes — ensure lab specifically reports poly-Ala tract length; "
            "molecular combing or Southern blot may be needed for large expansions. "
            "FEMALE CARRIERS: Generally unaffected but may have subtle learning difficulties; "
            "check X-inactivation pattern in carrier females with symptoms. "
            "OMIM Gene: 313430, Disease: MRXHF1 — 300123."
        ),
        "treatment": (
            "GH REPLACEMENT: rhGH for GH deficiency; critical for linear growth; "
            "cognitive benefits of treating GH deficiency in ID patients are additional. "
            "LEVOTHYROXINE: If TSH deficient; FT4-guided. "
            "SEX HORMONE REPLACEMENT: Testosterone in males with LH/FSH deficiency from early puberty. "
            "HYDROCORTISONE: If ACTH deficiency confirmed (less common but test all axes). "
            "NEURODEVELOPMENTAL SUPPORT: Formal intellectual assessment; "
            "educational support (special education setting typical); "
            "speech and language therapy; occupational therapy; "
            "long-term supported living planning. "
            "MONITORING: Annual pituitary function tests; auxology; "
            "GH adequacy as adult is important for body composition/bone density in ID patients."
        ),
        "key_features": [
            "XLR — males primarily affected; maternal lineage pattern; no son-to-son transmission",
            "GH deficiency most common; variable TSH, LH/FSH deficiency",
            "Intellectual disability COMMON in males — combination of ID + GH deficiency = SOX3 until proven otherwise",
            "Infundibular (pituitary stalk) hypoplasia on MRI — CHARACTERISTIC finding",
            "Poly-alanine expansion/contraction mutations — standard Sanger may miss; specific poly-Ala testing needed",
            "Female carriers: usually unaffected; subtle learning difficulties in some (check X-inactivation)",
            "Treatment: rhGH + levothyroxine + testosterone (if LH/FSH deficient) + neurodevelopmental support",
            "ACTH deficiency less common but Synacthen test required",
        ],
        "key_ddx": (
            "SOX3 vs other X-linked ID: SOX3 has PITUITARY HORMONE DEFICIENCY (GH most common); "
            "infundibular hypoplasia on MRI distinguishes; "
            "SOX3 vs PROP1: PROP1 is AR; no intellectual disability; gonadotropin deficiency prominent; "
            "SOX3 vs GLI2: GLI2 has AD + HPE spectrum + single central incisor; no XLR pattern; "
            "SOX3 vs SOX2: SOX2 causes severe anophthalmia + GH deficiency (AD); no XLR"
        ),
        "systemic_involvement": {
            "cognitive": "Intellectual disability (moderate, IQ ~50-80) in majority of affected males",
            "pituitary": "GH deficiency; variable TSH, LH/FSH; infundibular hypoplasia",
            "growth": "Severe GH deficiency without treatment; short stature is prominent",
            "reproductive": "LH/FSH deficiency → delayed puberty in males; testosterone needed",
            "adrenal": "ACTH relatively spared but should be tested; frank crisis less common",
        },
        "cascade_testing": (
            "XLR — all maternal uncles of affected male should be tested; "
            "maternal female relatives are obligate/possible carriers; "
            "X-inactivation in female carriers; "
            "poly-Ala expansion: ensure test covers repeat length measurement; "
            "prenatal: molecular diagnosis possible if family mutation known (X-linked so male fetus at 50% risk)"
        ),
        "emergency_protocol": (
            "HYPOGLYCAEMIA (GH deficiency predominant): "
            "IV glucose 10% 2 mL/kg bolus; oral glucose if mild; "
            "GH deficiency → hypoglycaemia especially fasting/overnight in children; "
            "ACTH generally intact: withhold hydrocortisone unless ACTH clearly deficient; "
            "educate carers/school re: hypoglycaemia signs in ID patients who cannot self-report; "
            "regular snacks; rhGH therapy reduces fasting hypoglycaemia risk."
        ),
    },
    # -- LHX3 -- CPHD3 -- RIGID CERVICAL SPINE ---------------------------------------------------
    {
        "gene": "LHX3",
        "alt_name": (
            "LHX3 (LHX3-397aa-9q34.3 / AR -- "
            "LIM-Homeobox-3 -- "
            "CPHD3-GH+TSH+PRL+LH/FSH-ALL-DEFICIENT -- "
            "RIGID-CERVICAL-SPINE-PATHOGNOMONIC-Cannot-Rotate-Neck -- "
            "Short-Neck-Sensorineural-Hearing-Loss-Some -- "
            "OMIM-CPHD3-221750)"
        ),
        "protein": (
            "LHX3 -- 9q34.3 AR -- LHX3-397aa -- "
            "LIM-Homeobox-TF-45kDa-Two-LIM-Domains-Homeodomain-Nuclear -- "
            "Regulates-Pituitary-Cell-Lineage-Differentiation-Alpha-GSU-TSHB-PRL-GH-LH-FSH -- "
            "OMIM-Gene-600577-Disease-CPHD3-221750"
        ),
        "locus": "9q34.3",
        "protein_size": "397 aa / ~45 kDa",
        "inheritance": (
            "Autosomal recessive. LHX3 encodes LIM Homeobox 3, a dual-LIM-domain homeodomain "
            "transcription factor expressed in Rathke's pouch and the developing anterior pituitary. "
            "LHX3 is required for differentiation and survival of somatotrophs, thyrotrophs, lactotrophs, "
            "and gonadotrophs — a similar but broader role compared to PROP1. "
            "HORMONES AFFECTED: GH + TSH + PRL + LH + FSH all deficient (same as PROP1/CPHD2). "
            "ACTH: Usually SPARED (corticotrophs are relatively LHX3-independent). "
            "UNIQUE FEATURE — RIGID CERVICAL SPINE: "
            "This is the MOST DIAGNOSTICALLY IMPORTANT feature of LHX3 — "
            "patients CANNOT rotate their neck laterally (limited cervical rotation); "
            "may have short, thickened neck; C1-C3 vertebral fusion or cervical spine rigidity; "
            "this is NOT due to pituitary disease — it reflects an independent role of LHX3 in "
            "embryonic neck/craniocervical development. "
            "NO OTHER CPHD gene causes cervical spine rigidity — this is pathognomonic. "
            "SENSORINEURAL HEARING LOSS: Present in ~50% of some LHX3 alleles; variable. "
            "MRI: Anterior pituitary hypoplasia; ectopic posterior pituitary common; "
            "hypoplastic stalk."
        ),
        "disease_category": (
            "Combined Pituitary Hormone Deficiency type 3 (CPHD3, OMIM #221750); "
            "AR; GH + TSH + PRL + LH + FSH deficient; ACTH typically spared; "
            "RIGID CERVICAL SPINE unique physical finding — pathognomonic; "
            "anterior pituitary hypoplasia + ectopic posterior pituitary; "
            "sensorineural hearing loss in ~50%"
        ),
        "disease_pathway": (
            "LHX3 PITUITARY CELL SPECIFICATION: "
            "LHX3 required for: somatotrophs (GH), thyrotrophs (TSH), lactotrophs (PRL), "
            "gonadotrophs (LH/FSH) — same as PROP1 but via LIM domain interactions with different co-factors. "
            "LOF → absence of all above cell types → panhypopituitarism (minus ACTH). "
            "GH ABSENT: Growth failure, neonatal/infantile hypoglycaemia. "
            "TSH ABSENT: Central hypothyroidism; levothyroxine required. "
            "LH/FSH ABSENT: Delayed/absent puberty; testosterone/oestrogen induction required. "
            "PRL ABSENT: Post-partum lactation failure (females). "
            "CERVICAL SPINE MECHANISM: LHX3 expressed in craniocervical somitic mesoderm → "
            "LOF → cervical vertebral anomalies (fusion, thickening) → rigid cervical spine; "
            "degree of rigidity correlates with allele severity; "
            "orthopedic/physiotherapy management. "
            "HEARING LOSS MECHANISM: LHX3 expressed in cochlear development; "
            "sensorineural loss in ~50% — audiology testing mandatory."
        ),
        "pathognomonic": (
            "LHX3 CLINICAL PEARLS: "
            "RIGID NECK IS PATHOGNOMONIC: In a child with CPHD (GH + TSH + LH/FSH low), "
            "examine the neck ROTATION — LHX3 is the ONLY CPHD gene causing this; "
            "limited lateral rotation of neck + short neck → LHX3 FIRST before molecular testing. "
            "DDx from TORTICOLLIS: Cervical rigidity in LHX3 is structural (vertebral fusion), "
            "not muscular; passive rotation also limited; X-ray cervical spine shows vertebral anomaly. "
            "HEARING LOSS: Test audiometry in all LHX3 patients — SNHL in ~50%; "
            "cochlear implant or hearing aids if significant loss. "
            "ACTH INTACT (USUALLY): Unlike PROP1, ACTH deficiency is NOT a major evolving concern "
            "in LHX3; however, formal Synacthen test is still recommended. "
            "PHYSIOTHERAPY: Cervical spine rigidity may impair quality of life; "
            "physiotherapy and/or orthopaedic review; neck brace for contact sports. "
            "OMIM Gene: 600577, Disease: CPHD3 — 221750."
        ),
        "treatment": (
            "GH REPLACEMENT: rhGH 0.025-0.05 mg/kg/day SC; growth monitoring. "
            "LEVOTHYROXINE: Central hypothyroidism; FT4-guided; start before GH. "
            "SEX HORMONE INDUCTION: LH/FSH absent; oestrogen (females) or testosterone (males) "
            "from puberty (~13y); combined HRT lifelong. "
            "HYDROCORTISONE: Synacthen test recommended; usually NOT needed (ACTH typically intact). "
            "CERVICAL SPINE: Orthopaedic review; physiotherapy; "
            "neck brace for sports; avoid high-impact head/neck activities; "
            "cervical spine X-ray and MRI to characterise vertebral anomaly. "
            "AUDIOLOGICAL MANAGEMENT: Formal audiometry; hearing aids if SNHL present; "
            "cochlear implant if severe SNHL."
        ),
        "key_features": [
            "CPHD3: GH + TSH + PRL + LH/FSH ALL deficient; ACTH typically SPARED",
            "RIGID CERVICAL SPINE — cannot rotate neck — PATHOGNOMONIC: the ONLY CPHD gene with this finding",
            "Short/thickened neck; cervical vertebral fusion on X-ray",
            "Sensorineural hearing loss in ~50% — audiology testing mandatory in all LHX3 patients",
            "Anterior pituitary hypoplasia + ectopic posterior pituitary on MRI",
            "ACTH usually intact — evolving ACTH loss (as in PROP1) not expected, but test formally",
            "Treatment: GH + levothyroxine + sex hormones; orthopaedic/physio for neck; audiology",
            "LHX3 mutations: compound heterozygous or homozygous LOF; AR inheritance",
        ],
        "key_ddx": (
            "LHX3 vs PROP1: PROP1 has NO neck rigidity; PROP1 has evolving ACTH loss; both lose LH/FSH; "
            "LHX3 rigid neck is ABSOLUTELY pathognomonic — no other CPHD gene has it; "
            "LHX3 vs LHX4: LHX4 is AD (not AR); LHX4 has Chiari malformation; LHX4 has ACTH deficiency; "
            "LHX3 cervical rigidity vs Klippel-Feil: Klippel-Feil is isolated skeletal; no endocrine; "
            "LHX3 vs POU1F1: POU1F1 spares LH/FSH; POU1F1 no neck anomaly"
        ),
        "systemic_involvement": {
            "cervical_spine": "Rigid cervical spine (limited rotation) — pathognomonic structural anomaly",
            "hearing": "SNHL in ~50% — cochlear LHX3 expression; audiology mandatory",
            "pituitary": "Anterior hypoplasia; ectopic PP; GH/TSH/PRL/LH/FSH all deficient",
            "growth": "Severe GH deficiency; short stature if untreated",
            "reproductive": "LH/FSH absent → delayed puberty; testosterone/oestrogen induction required",
        },
        "cascade_testing": (
            "AR — siblings: 25% risk; parental carrier testing (LHX3 sequencing); "
            "hearing test in all identified carriers (may have subclinical SNHL); "
            "neck rotation examination in all at-risk siblings; "
            "prenatal: molecular diagnosis available if family variant known"
        ),
        "emergency_protocol": (
            "HYPOGLYCAEMIA (GH + TSH absent, ACTH usually intact): "
            "IV glucose 10% 2 mL/kg bolus; check blood glucose; "
            "hydrocortisone NOT routinely needed (ACTH intact) but give empirically if "
            "ACTH status unknown and patient seriously ill; "
            "CERVICAL SPINE PRECAUTION: General anaesthetic/intubation — alert anaesthetist to "
            "rigid cervical spine (difficult airway management); "
            "awake fiberoptic intubation should be considered in elective surgery."
        ),
    },
    # -- LHX4 -- CPHD4 -- Chiari + ACTH Early -----------------------------------------------
    {
        "gene": "LHX4",
        "alt_name": (
            "LHX4 (LHX4-390aa-1q25.2 / AD-haploinsufficiency -- "
            "LIM-Homeobox-4 -- "
            "CPHD4-GH+TSH+ACTH-Deficiency -- "
            "ACTH-Deficiency-EARLY-Unlike-PROP1 -- "
            "Ectopic-Posterior-Pituitary-EPP -- "
            "Arnold-Chiari-Malformation-Associated -- "
            "OMIM-CPHD4-262700)"
        ),
        "protein": (
            "LHX4 -- 1q25.2 AD-haploinsufficiency -- LHX4-390aa -- "
            "LIM-Homeobox-4-45kDa-Two-LIM-Domains-Homeodomain-Nuclear -- "
            "Regulates-Pituitary-Cell-Differentiation-Somatotrophs-Thyrotrophs-Corticotrophs -- "
            "OMIM-Gene-602146-Disease-CPHD4-262700"
        ),
        "locus": "1q25.2",
        "protein_size": "390 aa / ~45 kDa",
        "inheritance": (
            "Autosomal dominant (haploinsufficiency). Heterozygous LOF mutations in LHX4 cause "
            "CPHD4. Unlike the AR CPHD genes (LHX3, PROP1, POU1F1 in recessive form), "
            "a single mutant LHX4 allele is sufficient to cause disease (haploinsufficiency). "
            "This means: (a) variable penetrance within families; (b) de novo mutations can occur; "
            "(c) the disease may appear sporadic or vertical in pedigrees. "
            "LHX4 encodes LIM Homeobox 4, a closely related paralog of LHX3 expressed in "
            "Rathke's pouch and anterior pituitary. "
            "HORMONES AFFECTED: GH + TSH + ACTH deficient. "
            "CRITICAL DIFFERENCE FROM PROP1: In PROP1, ACTH loss is a LATE complication (adulthood); "
            "in LHX4, ACTH deficiency is PRESENT from diagnosis (childhood/neonatal) — "
            "this means LHX4 patients are at immediate adrenal crisis risk from presentation. "
            "LH/FSH: Variable — some patients develop gonadotropin deficiency, others do not. "
            "ARNOLD-CHIARI MALFORMATION: Cerebellar tonsillar herniation into foramen magnum "
            "associated with LHX4 — a key physical/imaging feature; "
            "may cause obstructive hydrocephalus or cervical cord compression. "
            "MRI FEATURES: Ectopic posterior pituitary (EPP); hypoplastic infundibulum; "
            "small anterior pituitary; cerebellar/brainstem anomalies in some cases."
        ),
        "disease_category": (
            "Combined Pituitary Hormone Deficiency type 4 (CPHD4, OMIM #262700); "
            "AD haploinsufficiency with variable penetrance; "
            "GH + TSH + ACTH deficient — ACTH early (present from diagnosis, unlike PROP1); "
            "Arnold-Chiari malformation associated; ectopic posterior pituitary on MRI; "
            "variable LH/FSH deficiency"
        ),
        "disease_pathway": (
            "LHX4 TRIPLE-AXIS FAILURE (GH + TSH + ACTH): "
            "LHX4 activates differentiation of somatotrophs (GH), thyrotrophs (TSH), "
            "and corticotrophs (ACTH) in Rathke's pouch. "
            "EARLY ACTH LOSS: Corticotrophs are among the EARLIEST pituitary cell types to form; "
            "LHX4 haploinsufficiency impairs corticotroph differentiation from the start; "
            "cortisol deficiency is present at diagnosis — risk of adrenal crisis in infancy/childhood. "
            "GH ABSENT: Neonatal hypoglycaemia (both GH and cortisol low → compounded hypoglycaemia risk). "
            "TSH ABSENT: Central hypothyroidism. "
            "ARNOLD-CHIARI: LHX4 expressed in posterior fossa/hindbrain development → "
            "haploinsufficiency → cerebellar tonsillar herniation; "
            "may require posterior fossa decompression surgery if symptomatic. "
            "EPP MRI: T1 bright spot ectopically positioned → confirms pituitary stalk disruption. "
            "VARIABLE PENETRANCE: Some heterozygotes are clinically unaffected."
        ),
        "pathognomonic": (
            "LHX4 CLINICAL PEARLS: "
            "ACTH DEFICIENCY IS PRESENT FROM DIAGNOSIS: Unlike PROP1 where ACTH loss evolves in adulthood, "
            "LHX4 patients have ACTH deficiency from birth/early childhood — "
            "adrenal crisis risk is IMMEDIATE; ALL LHX4 patients must start hydrocortisone "
            "AT DIAGNOSIS regardless of whether cortisol is borderline or clearly low. "
            "NEONATAL PRESENTATION: GH + ACTH both absent → prolonged neonatal hypoglycaemia + "
            "potential adrenal crisis (hyponatraemia, hyperkalaemia, shock) — "
            "empiric HC + glucose at presentation before labs confirmed. "
            "ARNOLD-CHIARI MRI: Posterior fossa imaging is part of standard LHX4 work-up; "
            "symptomatic Chiari (headache on Valsalva, myelopathy) → neurosurgical decompression; "
            "asymptomatic Chiari → annual monitoring. "
            "VARIABLE PENETRANCE: An LHX4 mutation in a parent may cause no symptoms; "
            "do not reassure a child based on unaffected parent phenotype. "
            "OMIM Gene: 602146, Disease: CPHD4 — 262700."
        ),
        "treatment": (
            "HYDROCORTISONE: START AT DIAGNOSIS — ACTH deficient from birth/early childhood; "
            "10-15 mg/m²/day in 3 divided doses; stress dosing MANDATORY; "
            "IM hydrocortisone kit; medical alert bracelet. "
            "GH REPLACEMENT: rhGH for GH deficiency; after cortisol axis secured. "
            "LEVOTHYROXINE: For central hypothyroidism; start before GH. "
            "SEX HORMONES: If LH/FSH deficient; puberty induction from ~13y. "
            "ARNOLD-CHIARI MANAGEMENT: Neurosurgery review; "
            "posterior fossa decompression if symptomatic (headache, myelopathy, obstructive hydrocephalus). "
            "MONITORING: Regular cortisol compliance; growth; FT4; MRI every 2-5 years."
        ),
        "key_features": [
            "CPHD4: GH + TSH + ACTH deficient — ACTH deficiency PRESENT FROM DIAGNOSIS (unlike PROP1 where it's late)",
            "AD haploinsufficiency — single allele mutation sufficient; variable penetrance",
            "Ectopic posterior pituitary (EPP) on MRI — T1 bright spot at ectopic position",
            "Arnold-Chiari malformation — cerebellar tonsillar herniation — posterior fossa MRI mandatory",
            "Adrenal crisis risk from birth — hydrocortisone MUST start at diagnosis",
            "Variable LH/FSH involvement — test gonadotropin axis at puberty",
            "De novo mutations occur — may appear sporadic; sequence LHX4 in unexplained CPHD + Chiari",
            "Variable penetrance — unaffected parent does not exclude mutation transmission",
        ],
        "key_ddx": (
            "LHX4 vs PROP1: PROP1 ACTH is late (adulthood); LHX4 ACTH is EARLY (diagnosis); "
            "PROP1 is AR; LHX4 is AD; PROP1 has pituitary mass/involution; LHX4 has Chiari; "
            "LHX4 vs LHX3: LHX3 is AR; LHX3 has rigid neck; LHX3 ACTH usually intact; "
            "LHX4 has Chiari; LHX3 has hearing loss; "
            "LHX4 vs POU1F1: POU1F1 ACTH ALWAYS intact; POU1F1 no Chiari; "
            "LHX4 vs HESX1: HESX1 has SOD triad + optic nerve hypoplasia"
        ),
        "systemic_involvement": {
            "adrenal": "ACTH deficiency from diagnosis — immediate adrenal crisis risk",
            "brain": "Arnold-Chiari malformation (cerebellar tonsillar herniation); cerebellar/brainstem anomalies",
            "pituitary": "Anterior hypoplasia; ectopic PP; hypoplastic stalk",
            "thyroid": "Central hypothyroidism — levothyroxine required",
            "growth": "GH deficiency; short stature if untreated; compounded by cortisol deficiency",
        },
        "cascade_testing": (
            "AD — first-degree relatives; parent sequencing (variable penetrance may mean parent is mutation carrier); "
            "examine family members for subtle CPHD signs; "
            "MRI of at-risk family members; "
            "prenatal: molecular diagnosis if family variant known; "
            "Synacthen test at diagnosis to characterise ACTH status in all relatives"
        ),
        "emergency_protocol": (
            "ADRENAL CRISIS (ACTH deficient from birth): "
            "IV hydrocortisone 50-100 mg/m² IV IMMEDIATELY; "
            "0.9% NaCl 20 mL/kg bolus; 10% glucose for hypoglycaemia; "
            "TREAT EMPIRICALLY in any LHX4 patient with illness, vomiting, surgery, or anaesthetic — "
            "do NOT wait for cortisol result; "
            "CHIARI PRECAUTION: If Arnold-Chiari present, emergency intubation may be difficult; "
            "alert anaesthetist; avoid neck hyperextension."
        ),
    },
    # -- GLI2 -- HPE9 -- Most Variable CPHD Gene -----------------------------------------------
    {
        "gene": "GLI2",
        "alt_name": (
            "GLI2 (GLI2-1586aa-2q14.2 / AD-haploinsufficiency-variable-expressivity -- "
            "GLI-Family-Zinc-Finger-2 -- "
            "HPE9-Holoprosencephaly-Spectrum-Plus-Hypopituitarism -- "
            "MOST-VARIABLE-EXPRESSIVITY-Any-Hypopituitarism-Gene -- "
            "PSIS-Pituitary-Stalk-Interruption-Syndrome -- "
            "Single-Central-Incisor-Midline-Marker -- "
            "OMIM-HPE9-610829)"
        ),
        "protein": (
            "GLI2 -- 2q14.2 AD-haploinsufficiency -- GLI2-1586aa -- "
            "GLI-Zinc-Finger-TF-174kDa-Nuclear-Sonic-Hedgehog-Signal-Transducer -- "
            "Activates-Pituitary-Development-Downstream-SHH-Signalling -- "
            "OMIM-Gene-165230-Disease-HPE9-610829"
        ),
        "locus": "2q14.2",
        "protein_size": "1586 aa / ~174 kDa",
        "inheritance": (
            "Autosomal dominant (haploinsufficiency with HIGHLY VARIABLE expressivity). "
            "GLI2 is the downstream nuclear effector of the Sonic Hedgehog (SHH) signalling pathway. "
            "It encodes a zinc-finger transcription factor that, when activated by SHH signalling, "
            "drives proliferation and patterning of ventral forebrain structures including the "
            "pituitary gland and midline brain structures. "
            "MOST VARIABLE EXPRESSIVITY OF ANY HYPOPITUITARISM GENE: "
            "The same GLI2 mutation in one family member can produce isolated GH deficiency; "
            "in another family member: holoprosencephaly (HPE); in another: asymptomatic. "
            "This variability is striking even compared to other AD CPHD genes. "
            "HOLOPROSENCEPHALY SPECTRUM: From mild (microform — single central incisor, "
            "hypotelorism) to moderate (lobar HPE) to severe (alobar HPE — lethal). "
            "PITUITARY STALK INTERRUPTION SYNDROME (PSIS): Absent or hypoplastic pituitary stalk + "
            "ectopic posterior pituitary + anterior pituitary hypoplasia — GLI2 is a common "
            "monogenic cause of PSIS. "
            "SINGLE CENTRAL INCISOR (SCI): A MIDLINE MARKER — single midline upper incisor "
            "(versus normal two central incisors) indicates failed midline separation; "
            "SCI + GH deficiency → test GLI2. "
            "FACIAL MIDLINE: Hypotelorism (close-set eyes), flat nasal bridge, cleft lip/palate "
            "in severe HPE forms."
        ),
        "disease_category": (
            "Holoprosencephaly type 9 (HPE9, OMIM #610829); AD haploinsufficiency, high variable expressivity; "
            "spectrum: isolated GH deficiency + PSIS ↔ full alobar HPE; "
            "single central incisor is midline microform marker; "
            "pituitary stalk interruption syndrome (PSIS) most endocrine presentation"
        ),
        "disease_pathway": (
            "SHH → GLI2 PATHWAY IN PITUITARY AND FOREBRAIN: "
            "Sonic Hedgehog (SHH) from floor plate activates PTCH1 → SMO → GLI2 nuclear translocation → "
            "GLI2 activates ventral forebrain identity genes, pituitary proliferation, and midline fusion. "
            "PITUITARY: GLI2 activates pituitary progenitor proliferation and differentiation; "
            "haploinsufficiency → inadequate pituitary progenitor pool → "
            "pituitary stalk fails to form (PSIS); ectopic posterior pituitary; "
            "anterior pituitary hypoplastic → GH most commonly deficient; "
            "variable TSH, ACTH, LH/FSH loss depending on severity. "
            "FOREBRAIN: SHH drives telencephalic vesicle separation into two hemispheres; "
            "GLI2 LOF → failed separation → HPE spectrum (alobar most severe → lethal; "
            "lobar: partial separation; microform: only midline face anomalies including SCI). "
            "VARIABLE EXPRESSIVITY: The degree of SHH pathway impairment varies with allele severity, "
            "genetic modifiers, and stochastic developmental noise → accounts for enormous variability."
        ),
        "pathognomonic": (
            "GLI2 CLINICAL PEARLS: "
            "SINGLE CENTRAL INCISOR (SCI) IN A CHILD WITH SHORT STATURE: "
            "SCI is a midline face marker for impaired SHH midline signalling → "
            "check pituitary function AND brain MRI immediately; "
            "test GLI2 (and SHH, PTCH1, ZIC2 for HPE spectrum); "
            "SCI is PATHOGNOMONIC of midline signalling failure in this context. "
            "PSIS ON MRI: Absent/interrupted pituitary stalk + EPP + small anterior pituitary → "
            "test GLI2 in all unexplained PSIS. "
            "FAMILY VARIABILITY: Counsel families that the same mutation can cause very different "
            "outcomes — one sibling may have only short stature; another may have alobar HPE. "
            "FACIAL FEATURES: Hypotelorism, flat midface, any cleft → midline signalling defect; "
            "full HPE evaluation + pituitary testing. "
            "MULTIDISCIPLINARY: Neurology (HPE), endocrinology (hypopituitarism), "
            "genetics, craniofacial surgery if cleft. "
            "OMIM Gene: 165230, Disease: HPE9 — 610829."
        ),
        "treatment": (
            "GH REPLACEMENT: rhGH for GH deficiency (most common); IGF-1 monitoring. "
            "LEVOTHYROXINE: If TSH deficient; FT4-guided. "
            "HYDROCORTISONE: If ACTH deficient (common with PSIS) — start at diagnosis; "
            "stress dosing + IM kit + medical alert bracelet. "
            "SEX HORMONES: If LH/FSH deficient; puberty induction. "
            "HPE MANAGEMENT (if severe): "
            "Obstructive hydrocephalus → ventriculo-peritoneal shunt; "
            "seizures (common in HPE) → antiepileptic drugs (LEV preferred); "
            "temperature dysregulation (hypothalamic HPE involvement); "
            "swallowing difficulties → gastrostomy feed if needed. "
            "SINGLE CENTRAL INCISOR: Dental/orthodontic management; cosmetic implications. "
            "MONITORING: Annual pituitary function; MRI every 2-3 years; neurology follow-up."
        ),
        "key_features": [
            "MOST VARIABLE EXPRESSIVITY of any hypopituitarism gene — same mutation: isolated GH deficiency to alobar HPE",
            "AD haploinsufficiency with high variability; de novo and inherited forms",
            "Single central incisor (SCI) — midline marker — SCI + short stature = GLI2 until proven otherwise",
            "Pituitary stalk interruption syndrome (PSIS): absent stalk + EPP + anterior pituitary hypoplasia",
            "Holoprosencephaly spectrum: lobar/semilobar/alobar HPE in severe; microform in mild",
            "Hypotelorism, cleft lip/palate in HPE forms — midline face defects signal severity",
            "Multiple pituitary hormone deficiencies common with PSIS (GH + ACTH + TSH ± LH/FSH)",
            "Treatment: GH + HC (if ACTH low, present early) + levothyroxine + sex hormones; HPE multidisciplinary",
        ],
        "key_ddx": (
            "GLI2 vs other HPE genes: ZIC2 (most common HPE gene, non-pituitary prominent); "
            "SHH (HPE3, facial anomalies); PTCH1 (Gorlin, different); "
            "GLI2 vs GLI3 (Greig/Pallister-Hall — hands/feet + HPE); "
            "GLI2 PSIS vs other PSIS: HESX1, LHX4 also cause PSIS — molecular panel; "
            "GLI2 SCI vs normal variant: SCI in any child requires GLI2 + midline brain MRI; "
            "GLI2 isolated GHD vs idiopathic GHD: SCI or EPP + family HPE history → sequence GLI2"
        ),
        "systemic_involvement": {
            "brain": "HPE spectrum: alobar to microform; seizures common; temperature dysregulation",
            "pituitary": "PSIS; anterior hypoplasia; EPP; GH most common; variable ACTH/TSH/LH/FSH",
            "face": "Single central incisor (midline marker); hypotelorism; cleft lip/palate in severe HPE",
            "adrenal": "ACTH deficiency common with PSIS — immediate adrenal crisis risk",
            "swallowing": "Dysphagia in severe HPE — gastrostomy feeding may be required",
        },
        "cascade_testing": (
            "AD — first-degree relatives; parent sequencing (may be phenotypically mild carrier); "
            "MRI of at-risk relatives (PSIS/HPE may be sub-clinical); "
            "facial examination (SCI, hypotelorism in parents); "
            "variable penetrance counselling: unaffected parent with mutation can have severely affected child; "
            "prenatal: ultrasound HPE detectable from 16-20 weeks; molecular testing available"
        ),
        "emergency_protocol": (
            "ADRENAL CRISIS (ACTH deficiency with PSIS): "
            "IV hydrocortisone 50-100 mg/m² IV IMMEDIATELY; "
            "0.9% NaCl 20 mL/kg; 10% glucose; "
            "HPE EMERGENCIES: Seizures → IV benzodiazepine (lorazepam 0.1 mg/kg); "
            "obstructive hydrocephalus → emergency neurosurgery for VP shunt if acute; "
            "temperature dysregulation (poikilothermia) → active temperature management in NICU; "
            "simultaneous endocrine + neurological emergency management in severe HPE."
        ),
    },
]


def _make_cohort(gene_data: dict, seed: int, n: int = 40) -> list:
    r = random.Random(seed)
    gene = gene_data["gene"]

    pres_map = {
        "POU1F1": [
            "neonatal_hypoglycaemia_GH_TSH_PRL_absent",
            "growth_failure_triple_deficiency",
            "central_hypothyroidism_childhood",
            "delayed_growth_normal_puberty",
            "dominant_negative_short_stature_isolated",
        ],
        "PROP1": [
            "growth_failure_LH_FSH_absent_delayed_puberty",
            "pituitary_mass_childhood_involution_adolescence",
            "adrenal_crisis_adulthood_evolving_ACTH_loss",
            "primary_amenorrhoea_central_hypothyroidism",
            "del301_302_founder_panhypopituitarism",
        ],
        "HESX1": [
            "nystagmus_birth_optic_nerve_hypoplasia",
            "SOD_triad_MRI_absent_septum_pellucidum",
            "adrenal_crisis_ACTH_deficiency_SOD",
            "GH_deficiency_visual_impairment",
            "single_feature_optic_nerve_hypoplasia_only",
        ],
        "OTX2": [
            "anophthalmia_microphthalmia_birth",
            "coloboma_GH_deficiency",
            "retinal_dystrophy_EPP_on_MRI",
            "intellectual_disability_pituitary_hypoplasia",
            "severe_bilateral_microphthalmia_CPHD",
        ],
        "SOX3": [
            "X_linked_ID_GH_deficiency_males",
            "infundibular_hypoplasia_MRI_delayed_puberty",
            "moderate_ID_short_stature_hemizygous_male",
            "growth_failure_no_family_history_de_novo_expansion",
            "central_hypothyroidism_cognitive_impairment_male",
        ],
        "LHX3": [
            "CPHD3_rigid_neck_panhypopituitarism",
            "short_neck_cannot_rotate_childhood_GH_failure",
            "SNHL_GH_TSH_absent_rigid_cervical_spine",
            "delayed_puberty_LH_FSH_absent_neck_rigidity",
            "compound_heterozygous_LHX3_panhypopituitarism",
        ],
        "LHX4": [
            "neonatal_adrenal_crisis_ACTH_early_GH_TSH_absent",
            "Chiari_malformation_CPHD4_childhood",
            "EPP_MRI_triple_deficiency_GH_TSH_ACTH",
            "variable_penetrance_parent_unaffected_child_severe",
            "de_novo_LHX4_PSIS_neonatal_hypoglycaemia_hypotension",
        ],
        "GLI2": [
            "single_central_incisor_short_stature_PSIS",
            "HPE_lobar_panhypopituitarism_seizures",
            "PSIS_GH_ACTH_TSH_absent_EPP_MRI",
            "microform_HPE_hypotelorism_isolated_GHD",
            "family_HPE_alobar_index_case_GHD_sibling",
        ],
    }

    mgmt_map = {
        "POU1F1": [
            "rhGH_levothyroxine_only_no_HC_no_sex_hormones",
            "rhGH_FT4_monitoring_puberty_normal",
            "levothyroxine_first_then_GH",
            "IGF1_monitoring_FT4_monitoring",
        ],
        "PROP1": [
            "rhGH_levothyroxine_sex_hormones_HC_when_ACTH_fails",
            "annual_Synacthen_test_monitoring_ACTH_status",
            "del301_302_targeted_testing_panhypopituitary_replacement",
            "puberty_induction_GnRH_pump_fertility_if_desired",
        ],
        "HESX1": [
            "rhGH_levothyroxine_HC_MANDATORY_stress_dosing",
            "ophthalmology_visual_rehab_GH_TSH_replacement",
            "stress_HC_Synacthen_annual_ALL_SOD",
            "desmopressin_if_central_DI_SOD",
        ],
        "OTX2": [
            "rhGH_ophthalmology_levothyroxine_HC_if_ACTH_low",
            "prosthetic_eye_low_vision_aids_GH_replacement",
            "annual_retinal_review_ERG_pituitary_hormones",
            "neurodevelopmental_support_GH_levothyroxine",
        ],
        "SOX3": [
            "rhGH_levothyroxine_testosterone_ID_support",
            "educational_support_GH_infundibular_hypoplasia",
            "supported_living_rhGH_testosterone_monitoring",
            "poly_ala_molecular_confirmation_rhGH_replacement",
        ],
        "LHX3": [
            "rhGH_levothyroxine_sex_hormones_orthopaedic_neck",
            "physiotherapy_cervical_spine_GH_TSH_replacement",
            "audiology_hearing_aids_pituitary_replacement",
            "awake_fiberoptic_intubation_aware_anaesthetist",
        ],
        "LHX4": [
            "HC_IMMEDIATELY_rhGH_levothyroxine_Chiari_monitoring",
            "posterior_fossa_decompression_pituitary_replacement",
            "stress_dosing_IM_kit_medical_alert_bracelet_LHX4",
            "sex_hormones_when_LH_FSH_deficit_confirmed",
        ],
        "GLI2": [
            "rhGH_HC_levothyroxine_HPE_multidisciplinary",
            "VP_shunt_hydrocephalus_pituitary_replacement",
            "LEV_seizures_HPE_HC_IM_kit_SCI_dental",
            "single_central_incisor_orthodontic_GH_replacement",
        ],
    }

    age_ranges = {
        "POU1F1": (0, 5),
        "PROP1":  (1, 15),
        "HESX1":  (0, 3),
        "OTX2":   (0, 2),
        "SOX3":   (1, 8),
        "LHX3":   (1, 10),
        "LHX4":   (0, 5),
        "GLI2":   (0, 8),
    }

    pres_list = pres_map.get(gene, ["pituitary_hormone_deficiency", "CPHD_workup"])
    mgmt_list = mgmt_map.get(gene, ["pituitary_hormone_replacement", "monitoring"])
    age_min, age_max = age_ranges.get(gene, (0, 10))

    patients = []
    for i in range(n):
        age_dx = r.randint(age_min, age_max)
        age_curr = age_dx + r.randint(1, 20)
        patients.append({
            "id": f"{gene}-{seed}-{i+1:02d}",
            "gene": gene,
            "age_at_diagnosis": age_dx,
            "age_current": min(age_curr, 55),
            "presentation": r.choice(pres_list),
            "management": r.choice(mgmt_list),
            "outcome": r.choice([
                "stable_on_replacement",
                "ongoing_surveillance",
                "adrenal_crisis_averted",
                "MDT_review",
                "growth_normal_on_GH",
            ]),
        })
    return patients


def get_overview() -> dict:
    cohorts = []
    for i, g in enumerate(HYPOPITUITARY_GENES):
        seed = SEED_BASE + i
        cohort = _make_cohort(g, seed)
        avg_age_dx = round(sum(p["age_at_diagnosis"] for p in cohort) / len(cohort), 1)
        cohorts.append({
            "gene": g["gene"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "disease_summary": g["disease_category"][:120],
            "patients": len(cohort),
            "avg_age_at_dx": avg_age_dx,
        })

    total_patients = sum(c["patients"] for c in cohorts)
    return {
        "atlas": "Hereditary-Hypopituitarism-Atlas",
        "subtitle": (
            "Complete 8-Gene Reference — POU1F1 · PROP1 · HESX1 · OTX2 · SOX3 · LHX3 · LHX4 · GLI2"
        ),
        "total_patients": total_patients,
        "genes_covered": len(HYPOPITUITARY_GENES),
        "seed_range": f"{SEED_BASE}–{SEED_BASE + len(HYPOPITUITARY_GENES) - 1}",
        "cohort_size_per_gene": 40,
        "domain": (
            "Hereditary Combined Pituitary Hormone Deficiency (CPHD) & Syndromic Hypopituitarism — "
            "Transcription Factor Defects in Pituitary Development"
        ),
        "key_diagnostic_tests": [
            "GH provocation test (insulin tolerance / glucagon / GHRH-arginine) — confirm GH deficiency",
            "IGF-1 (age/sex matched) — GH axis screening",
            "TSH + Free T4 — central hypothyroidism (TSH unreliable; use FT4)",
            "Prolactin — low PRL confirms lactotroph failure (POU1F1/PROP1)",
            "LH + FSH (basal + GnRH-stimulated) — gonadotropin deficiency",
            "ACTH + Morning cortisol (+ Synacthen-stimulated cortisol) — MANDATORY in ALL patients",
            "ADH / paired serum-urine osmolality — central DI assessment (HESX1/GLI2)",
            "Pituitary MRI (1.5T-3T) — size, stalk, EPP T1 bright spot, HPE, Chiari, optic nerves",
            "Ophthalmology: visual acuity + visual fields + fundoscopy (HESX1/OTX2)",
            "Cervical spine X-ray + MRI — rigid spine assessment (LHX3)",
            "Audiometry — SNHL assessment (LHX3)",
            "Gene panel: POU1F1+PROP1+HESX1+OTX2+SOX3+LHX3+LHX4+GLI2+LHX4+FOXA2+SOX2+OTX2",
        ],
        "key_emergency_rules": [
            "LHX4: ACTH deficient FROM BIRTH — start hydrocortisone at diagnosis without delay",
            "HESX1/SOD: ALL SOD patients carry IM HC kit regardless of baseline ACTH status",
            "GLI2/PSIS: ACTH deficiency common — treat as adrenal insufficient until proven otherwise",
            "PROP1: Annual Synacthen test from adolescence — evolving ACTH loss kills; anticipate it",
            "LHX3: Alert anaesthetist to rigid cervical spine — awake fiberoptic intubation for elective surgery",
            "ALL CPHD: Stress dosing triple HC dose during fever/surgery; IM kit + medical alert bracelet",
            "OTX2/SOX3: Full pituitary axes must be tested — ACTH may be deficient even if eye disease is dominant",
        ],
        "cohort_breakdown": cohorts,
    }


def get_breakdown() -> dict:
    breakdown_by_gene = {}
    for i, g in enumerate(HYPOPITUITARY_GENES):
        seed = SEED_BASE + i
        cohort = _make_cohort(g, seed)
        presentations = {}
        managements = {}
        outcomes = {}
        for p in cohort:
            pres = p["presentation"]
            presentations[pres] = presentations.get(pres, 0) + 1
            mgmt = p["management"]
            managements[mgmt] = managements.get(mgmt, 0) + 1
            out = p["outcome"]
            outcomes[out] = outcomes.get(out, 0) + 1

        breakdown_by_gene[g["gene"]] = {
            "gene": g["gene"],
            "alt_name": g["alt_name"],
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "disease_category": g["disease_category"],
            "disease_pathway": g["disease_pathway"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "key_features": g["key_features"],
            "key_ddx": g["key_ddx"],
            "systemic_involvement": g["systemic_involvement"],
            "cascade_testing": g["cascade_testing"],
            "emergency_protocol": g["emergency_protocol"],
            "cohort_size": len(cohort),
            "presentation_distribution": presentations,
            "management_distribution": managements,
            "outcome_distribution": outcomes,
            "sample_patients": cohort[:5],
        }
    return {"breakdown_by_gene": breakdown_by_gene, "total_genes": len(HYPOPITUITARY_GENES)}


def get_definitions() -> dict:
    return {
        "atlas_domain": "Hereditary-Hypopituitarism-Atlas — Complete 8-Gene Pituitary Transcription Factor Reference",
        "key_definitions": {
            "Combined_Pituitary_Hormone_Deficiency_CPHD": (
                "Deficiency of 2 or more anterior pituitary hormones (GH, TSH, ACTH, LH/FSH, PRL) "
                "due to a single genetic cause; distinguished from isolated hormone deficiencies; "
                "most commonly caused by transcription factor mutations (POU1F1, PROP1, LHX3, LHX4, GLI2); "
                "CPHD1 = POU1F1; CPHD2 = PROP1; CPHD3 = LHX3; CPHD4 = LHX4."
            ),
            "GH_Deficiency_Diagnosis": (
                "Cannot diagnose GH deficiency by single random GH level (GH secretion is pulsatile); "
                "require GH PROVOCATION TEST: insulin tolerance test (ITT) gold standard "
                "(hypoglycaemia <2.2 mmol/L → GH should rise >6-9 mcg/L in adults; >15-20 mU/L in children); "
                "alternatives: glucagon test, GHRH-arginine test (less reliable with GH axis disruption); "
                "IGF-1 low for age/sex is supportive but not diagnostic alone; "
                "ITT is contraindicated in seizure disorder, ischaemic heart disease, severe ACTH deficiency."
            ),
            "Central_Hypothyroidism": (
                "TSH-deficient hypothyroidism from pituitary/hypothalamic disease; "
                "TSH can be LOW, inappropriately normal, or even mildly elevated (biologically inactive TSH); "
                "diagnosis requires FT4 LOW + clinical context; "
                "CRITICAL: Do NOT use TSH alone to monitor treatment in central hypothyroidism; "
                "use FT4 (target mid-to-upper normal range); "
                "start levothyroxine BEFORE initiating rhGH (GH increases T4 clearance; "
                "unrecognised central hypothyroidism blunts GH response)."
            ),
            "Ectopic_Posterior_Pituitary_EPP": (
                "T1-bright spot (posterior pituitary neurophysin/vasopressin-associated protein) "
                "normally located in sella turcica (in situ); "
                "EPP: bright spot found at ectopic position — along pituitary stalk, "
                "at median eminence, or at hypothalamus; "
                "EPP on MRI indicates pituitary stalk interruption or failed stalk development; "
                "associated with: LHX4, GLI2, OTX2, HESX1, SOX3 hypopituitarism genes; "
                "patients with EPP typically have multiple anterior pituitary deficiencies; "
                "ADH is often preserved (EPP still produces vasopressin even if not in sella)."
            ),
            "Pituitary_Stalk_Interruption_Syndrome_PSIS": (
                "MRI triad: absent/hypoplastic pituitary stalk + ectopic posterior pituitary + "
                "anterior pituitary hypoplasia; "
                "causes multiple anterior pituitary deficiencies (GH most common; ACTH, TSH, LH/FSH variable); "
                "GL2 is most common monogenic cause; also HESX1, LHX4, OTX2; "
                "non-genetic PSIS from birth trauma (forceps delivery) or perinatal hypoxia-ischaemia; "
                "diagnosis: 3T MRI with 2mm slices through pituitary; "
                "ACTH axis testing mandatory — adrenal crisis risk is high in PSIS."
            ),
            "Septo_Optic_Dysplasia_SOD": (
                "De Morsier syndrome; clinical triad: optic nerve hypoplasia (ONH) + "
                "absent septum pellucidum + pituitary hypoplasia; "
                "all three features in only ~30%; two features = SOD; "
                "ONH: bilateral small optic nerves on MRI; 'double ring' sign on fundoscopy; "
                "HESX1 mutations account for ~1% of SOD — most is environmental; "
                "ACTH deficiency MOST DANGEROUS complication — can cause death if missed; "
                "ALL SOD patients: mandatory Synacthen test + IM hydrocortisone kit."
            ),
            "Holoprosencephaly_HPE_Spectrum": (
                "Failure of forebrain to divide into two hemispheres; "
                "alobar: single monoventricle, no interhemispheric fissure (lethal); "
                "semilobar: partial separation posteriorly; "
                "lobar: near-complete separation except frontal; "
                "microform: only facial midline anomalies (single central incisor, hypotelorism, cleft); "
                "most common HPE gene: ZIC2 (no pituitary involvement); "
                "GLI2 (HPE9) — pituitary involvement prominent; "
                "SHH, PTCH1, GLI3 also cause HPE."
            ),
            "Central_Adrenal_Insufficiency_CAI": (
                "ACTH deficiency → reduced cortisol synthesis; "
                "unlike primary adrenal failure (Addison's): mineralocorticoids NORMAL "
                "(zona glomerulosa is RAAS-regulated, not ACTH-regulated); "
                "NO salt-wasting; hyperpigmentation ABSENT (ACTH/MSH low not high); "
                "DANGEROUS if unrecognised — cortisol essential for stress response; "
                "adrenal crisis during illness/surgery = life-threatening; "
                "treatment: hydrocortisone (not fludrocortisone) 10-15 mg/m²/day; "
                "stress dosing: triple HC dose during fever/illness; IM hydrocortisone for vomiting."
            ),
            "GH_Replacement_Monitoring": (
                "rhGH 0.025-0.05 mg/kg/day SC (children); 0.1-0.3 mg/day SC (adults, weight-based); "
                "monitor: IGF-1 (target age/sex-matched mid-normal range); "
                "annual auxology (height velocity); bone age X-ray; thyroid function; "
                "side effects of GH: pseudo-tumour cerebri (headache, papilloedema) — uncommon; "
                "fluid retention; slipped upper femoral epiphysis (limp in prepubertal children); "
                "do NOT start rhGH without first treating central hypothyroidism (if present) "
                "and ensuring ACTH axis is sufficient."
            ),
            "Puberty_Induction_Hypogonadotropic": (
                "Required when LH/FSH absent (PROP1, LHX3, SOX3); "
                "Females: ethinylestradiol 2 mcg/day (or conjugated oestrogens 0.3 mg/day) "
                "from ~12-13y; increase gradually over 2-3y; add progestogen after 2y or first bleed; "
                "continue as HRT until natural menopause age; "
                "Males: testosterone 25 mg IM monthly → escalate to adult dose over 2-3y; "
                "oral testosterone undecanoate or transdermal in older adolescents; "
                "BONE PROTECTION: Sex hormones are critical for bone mineralisation; "
                "DXA annually once replacement started; "
                "FERTILITY: GnRH pump (pulsatile) or combined FSH/LH injections; "
                "refer to reproductive endocrinologist."
            ),
            "Single_Central_Incisor_SCI": (
                "Midline dental anomaly: single unpaired central maxillary incisor "
                "(normally 2 central incisors); "
                "midline marker for impaired SHH signalling during facial midline development; "
                "SCI in a child with growth failure → MANDATORY pituitary MRI + GLI2 testing; "
                "SCI can occur without HPE (microform HPE); "
                "not to be confused with fusion of two incisors (gemination) — true SCI is single tooth."
            ),
        },
        "key_drug_contraindications": [
            "ALL CPHD: Do NOT start rhGH before treating central hypothyroidism — GH accelerates T4 clearance",
            "PROP1/LHX4/GLI2: Do NOT withhold hydrocortisone during illness — adrenal crisis is fatal",
            "LHX3: Anaesthesia with rigid cervical spine — ALERT anaesthetist; no blind nasal intubation",
            "HESX1/SOD: NEVER assume ACTH intact — Synacthen test is MANDATORY for every SOD patient",
            "LHX4/GLI2: Do NOT give dexamethasone chronically in children — use hydrocortisone only",
            "POU1F1: Do NOT give fludrocortisone — ACTH/aldosterone axis is INTACT",
            "SOX3: Standard Sanger sequencing MISSES poly-alanine expansion — ensure poly-Ala tract length tested",
            "PROP1: Do NOT biopsy pituitary mass — it involutes spontaneously; biopsy causes harm",
        ],
        "genes_in_atlas": [g["gene"] for g in HYPOPITUITARY_GENES],
        "seeds": list(range(SEED_BASE, SEED_BASE + len(HYPOPITUITARY_GENES))),
        "total_patients_modelled": 320,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (POU1F1) ===")
    bk = get_breakdown()
    print(json.dumps(bk["breakdown_by_gene"]["POU1F1"], indent=2)[:2000])
    print("\n=== DEFINITIONS (first 1000 chars) ===")
    df = get_definitions()
    print(json.dumps(df, indent=2)[:1000])
