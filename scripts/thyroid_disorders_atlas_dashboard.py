#!/usr/bin/env python3
"""Thyroid-Disorders-Atlas — Complete 8-Gene Hereditary Thyroid Disorders Atlas
TSHR    (TSH receptor; 764 aa; 14q31.1; OMIM gene 603372; AR LOF → Resistance to TSH RTSH #275200;
         AD GOF → familial non-autoimmune hyperthyroidism #609152; GPCR 7-transmembrane; TSH binding
         activates cAMP/PKA → thyroid growth + T4/T3 synthesis; LOF: TSH very high, T4 low, thyroid
         hypoplastic, TSH unresponsive to TRH; GOF: hyperthyroidism → radioiodine or hemithyroidectomy;
         LOF: levothyroxine lifelong from neonatal screening; Never assume maternal anti-TPO if TSH
         does not normalize by 3 years — sequence TSHR) ·
PAX8    (Paired box protein Pax-8; 450 aa; 2q14.1; OMIM gene 167415; OMIM disease #218700;
         AD haploinsufficiency 50% penetrance; thyroid dysgenesis — agenesis/hypoplasia/ectopic;
         sublingual ectopy most common; co-occurring renal anomalies; Müllerian aplasia in 46,XX;
         DO NOT remove ectopic gland without nuclear medicine scan confirming non-functional;
         levothyroxine lifelong) ·
TPO     (Thyroid peroxidase; 933 aa; 2p25.3; OMIM gene 606765; OMIM disease DH1 #274500; AR biallelic;
         catalyzes iodination of thyroglobulin + coupling of MIT/DIT; LOF → complete or partial
         organification defect; POSITIVE perchlorate discharge test; goiter; levothyroxine lifelong) ·
TG      (Thyroglobulin; 2768 aa; 8q24.22; OMIM gene 188450; OMIM disease DH3 #274700; AR biallelic;
         scaffold protein for thyroid hormone synthesis; LOF → goiter + VERY LOW serum Tg despite goiter;
         perchlorate discharge NEGATIVE; levothyroxine lifelong) ·
SLC5A5  (Sodium-iodide symporter NIS; 643 aa; 19p13.11; OMIM gene 601843; OMIM disease ITD #274400;
         AR biallelic; electrogenic 2Na+:1I- cotransporter; basolateral membrane; concentrates iodide
         20-40x; LOF → ABSENT radioiodine uptake WITH goiter; perchlorate negative; levothyroxine lifelong) ·
DUOX2   (Dual oxidase 2; 1548 aa; 15q21.1; OMIM gene 606759; OMIM disease DH6 #607200; biallelic AR
         severe, monoallelic milder/transient; produces H2O2 for TPO iodination; perchlorate POSITIVE;
         monoallelic = most common cause of TRANSIENT congenital hypothyroidism; re-evaluate at age 3) ·
SLC26A4 (Pendrin; 780 aa; 7q22.3; OMIM gene 605646; OMIM disease Pendred syndrome #274600; AR biallelic;
         Cl-/I- anion exchanger; thyroid + inner ear + kidney; LOF → SENSORINEURAL DEAFNESS + GOITER +
         Enlarged Vestibular Aqueduct; EVA on CT pathognomonic; cochlear implant for deafness) ·
FOXE1   (Forkhead box E1 TTF-2; 373 aa; 9q22.33; OMIM gene 602617; OMIM disease Bamforth-Lazarus
         syndrome #241850; AR biallelic; essential for thyroid migration + palate/choanae development;
         LOF → ATHYREOSIS + CLEFT PALATE + CHOANAL ATRESIA + SPIKY HAIR; choanal atresia = immediate
         airway emergency at birth; levothyroxine from day 1)
320-patient aggregate cohort (8 x 40, seeds 1302-1309)
"""

import random

SEED_BASE = 1302

THYROID_GENES = [
    # ── TSHR — TSH Receptor / Resistance to TSH (RTSH) & Non-autoimmune Hyperthyroidism ──
    {
        "gene": "TSHR",
        "protein": "TSH Receptor (TSHR)",
        "alias": (
            "TSHR; OMIM gene 603372; RTSH #275200 (AR LOF); FNAH #609152 (AD GOF); 14q31.1; 764 aa; ~87 kDa; "
            "GPCR superfamily; 7-transmembrane domain; extracellular leucine-rich repeat domain binds TSH; "
            "TSH binding → Gs-coupled adenylate cyclase → cAMP↑ → PKA → thyroid transcription factor activation "
            "→ thyroglobulin synthesis + iodide uptake + T4/T3 production + thyrocyte proliferation; "
            "LOF (AR biallelic): RTSH — receptor absent or unresponsive → thyroid hypoplastic despite very high TSH; "
            "GOF (AD activating): familial non-autoimmune hyperthyroidism — constitutive cAMP → autonomous growth + T4/T3"
        ),
        "aa": "764 aa",
        "kDa": "~87 kDa",
        "locus": "14q31.1",
        "omim_gene": 603372,
        "omim_disease": 275200,
        "inheritance": (
            "AR biallelic (RTSH — LOF); AD activating/GOF (familial non-autoimmune hyperthyroidism); "
            "de novo GOF causes sporadic toxic thyroid nodule; complete RTSH requires biallelic null variants; "
            "heterozygous LOF = partial TSH resistance with mild/borderline TSH elevation"
        ),
        "gene_class": (
            "TSHR encodes the TSH receptor, a Gs-protein-coupled receptor expressed exclusively on thyroid "
            "follicular cells. It is the primary regulator of thyroid growth, differentiation, and hormone synthesis. "
            "MECHANISM (LOF): biallelic inactivating variants (missense, nonsense, splice) → absent or non-functional "
            "receptor → thyrocytes cannot respond to TSH → thyroid remains hypoplastic despite markedly elevated TSH → "
            "congenital hypothyroidism with small thyroid gland; the TRH stimulation test is diagnostic — "
            "TSH rises normally after TRH (pituitary intact) but serum T4 does NOT rise (thyroid cannot respond). "
            "MECHANISM (GOF): activating variants in transmembrane or extracellular domain → constitutive cAMP "
            "production independent of TSH → autonomous thyroid growth + T4/T3 hypersecretion → "
            "familial non-autoimmune hyperthyroidism (TSH low, T4/T3 high, no autoantibodies). "
            "KEY DDx: RTSH vs central hypothyroidism — in RTSH, TSH is VERY HIGH (pituitary drives it up); "
            "in central hypothyroidism, TSH is LOW or inappropriately normal with low T4."
        ),
        "phenotype": (
            "LOF/RTSH: severe congenital hypothyroidism from birth; TSH 50-250 mU/L at neonatal screening; "
            "T4 very low (2-8 pmol/L); thyroid hypoplasia on ultrasound/scan (0.5-3 mL); "
            "developmental delay if untreated; growth retardation; myxedema; "
            "TSH does NOT respond to TRH stimulation (flat T4 response); "
            "GOF/FNAH: familial hyperthyroidism — palpitations, heat intolerance, weight loss, goiter; "
            "no TSI/TPO antibodies; family members affected across generations (AD); "
            "sporadic GOF: autonomous toxic nodule → unilateral hyperthyroidism"
        ),
        "hallmark": (
            "TSH VERY HIGH + T4 VERY LOW + THYROID HYPOPLASTIC + TSH UNRESPONSIVE TO TRH = RTSH until proven otherwise. "
            "The flat T4 response to TRH stimulation distinguishes TSHR LOF from all other causes of "
            "neonatal hypothyroidism (in all other forms the thyroid gland can respond if present). "
            "KEY RULE: Never assume maternal anti-TPO antibody explains persistent neonatal hypothyroidism "
            "if TSH does not normalize by age 3 years — sequence TSHR urgently. "
            "GOF hallmark: hyperthyroidism without autoantibodies + family history = TSHR GOF."
        ),
        "treatment_alerts": [
            "LOF/RTSH: Levothyroxine lifelong from neonatal screening; target T4 upper half of normal range in first 3 years.",
            "GOF/FNAH: methimazole bridge → radioiodine ablation or hemithyroidectomy (definitive).",
            "TSH UNRESPONSIVE TO TRH: perform TRH stimulation test to confirm RTSH before labeling as 'resistant'.",
            "DO NOT DELAY TREATMENT in RTSH: every week of untreated hypothyroidism risks irreversible neurodevelopmental harm.",
            "MATERNAL ANTI-TPO: transient neonatal hypothyroidism from maternal antibodies resolves by 3-6 months; "
            "persistent beyond this requires TSHR sequencing.",
            "FAMILY SCREENING: siblings of RTSH proband need TSHR sequencing; 25% recurrence risk (AR).",
            "TOXIC NODULE from GOF: exclude Graves disease (TSI negative in TSHR GOF) before radioiodine.",
        ],
        "key_ddx": (
            "Central (secondary) hypothyroidism: TSH LOW or inappropriately normal with low T4 — "
            "opposite biochemistry to RTSH where TSH is very high; "
            "Thyroid dysgenesis from PAX8/FOXE1: thyroid small/absent but TSHR intact (TSH responds to TRH); "
            "Dyshormonogenesis (TPO/TG/SLC5A5/DUOX2): thyroid large (goiter), not small; "
            "Transient neonatal hypothyroidism from maternal TSH-receptor blocking antibodies (TRBAb): "
            "TSH high, T4 low, but resolves within 3-6 months — TRBAb maternal titre diagnostic; "
            "Iodine deficiency: goiter, not hypoplasia; endemic; responds to iodine supplement."
        ),
    },
    # ── PAX8 — Thyroid Dysgenesis ─────────────────────────────────────────────
    {
        "gene": "PAX8",
        "protein": "Paired Box Protein Pax-8 (PAX8)",
        "alias": (
            "PAX8; OMIM gene 167415; thyroid dysgenesis #218700; 2q14.1; 450 aa; ~48 kDa; AD haploinsufficiency; "
            "paired box and homeodomain transcription factor; expressed in thyroid + kidney + Müllerian duct; "
            "PAX8 activates promoters of TG, TPO, NIS (SLC5A5) — master thyroid differentiation TF; "
            "haploinsufficiency (50% penetrance, variable expressivity): structural thyroid defects — "
            "agenesis (~20%), hypoplasia, ectopia (sublingual most common ~70%); "
            "renal anomalies in subset; MRKH overlap in 46,XX females; de novo in ~20% of cases"
        ),
        "aa": "450 aa",
        "kDa": "~48 kDa",
        "locus": "2q14.1",
        "omim_gene": 167415,
        "omim_disease": 218700,
        "inheritance": (
            "AD (haploinsufficiency); 50% penetrance; variable expressivity within families; "
            "de novo variants in ~20% of index cases; biallelic variants not reported (likely lethal); "
            "familial cases show AD transmission with variable thyroid phenotype"
        ),
        "gene_class": (
            "PAX8 encodes a paired-box domain transcription factor critical for thyroid organogenesis. "
            "During embryogenesis, PAX8 is expressed in the thyroid anlage at the foramen caecum from day 16 "
            "(embryonic equivalent) and is required for differentiation of thyroid precursor cells into "
            "functional follicular cells. "
            "MECHANISM: PAX8 directly activates transcription of TG, TPO, and NIS (SLC5A5) — without PAX8, "
            "thyroid progenitor cells fail to differentiate, leading to structural defects: complete agenesis "
            "(~20%), severe hypoplasia (small in situ gland), or ectopia (gland arrests migration → remains "
            "at base of tongue as sublingual mass or along the thyroglossal duct tract). "
            "ECTOPIA RULE: an ectopic sublingual thyroid may be the ONLY thyroid tissue present — "
            "removing it surgically without nuclear medicine scan confirmation of non-function will render "
            "the patient permanently athyreotic. Always confirm functionality before surgery. "
            "EXTRA-THYROIDAL: PAX8 is expressed in Müllerian ducts (uterus/fallopian tubes) and kidney — "
            "co-occurring renal anomalies and uterine agenesis (MRKH phenotype) reported in some families."
        ),
        "phenotype": (
            "Congenital hypothyroidism from thyroid dysgenesis: TSH elevated on NBS (may be borderline); "
            "T4 low; thyroid ultrasound shows absent, small, or ectopic gland; "
            "thyroid scan (99mTc or 123I): ectopic uptake at base of tongue or sublingual position; "
            "co-occurring renal anomalies (horseshoe kidney, renal agenesis) in ~20%; "
            "Müllerian aplasia (uterine agenesis, MRKH overlap) in some 46,XX females; "
            "phenotypic variability: same PAX8 variant may cause agenesis in one family member and borderline TSH in another"
        ),
        "hallmark": (
            "THYROID ECTOPIA (sublingual mass or uptake at base of tongue) on nuclear medicine scan = "
            "DO NOT SURGICALLY REMOVE until confirmed non-functional — it may be the only thyroid tissue. "
            "Variable penetrance means relatives may have only mild TSH elevation — family history may appear negative. "
            "Müllerian aplasia + congenital hypothyroidism in a 46,XX female → PAX8 sequencing. "
            "Renal anomalies + thyroid dysgenesis in same patient → PAX8 sequencing."
        ),
        "treatment_alerts": [
            "LEVOTHYROXINE LIFELONG: start from neonatal screening diagnosis; dose 10-15 mcg/kg/day in neonates.",
            "DO NOT REMOVE ECTOPIC GLAND without 99mTc/123I nuclear medicine scan confirming it is non-functional.",
            "THYROID ULTRASOUND + SCAN: mandatory work-up for thyroid dysgenesis morphology.",
            "RENAL ULTRASOUND: screen for renal anomalies in all PAX8 probands.",
            "GYNAECOLOGICAL ASSESSMENT: Müllerian anomaly screen in 46,XX females with PAX8 variants.",
            "FAMILY SCREENING: 50% offspring risk (AD, reduced penetrance); TSH + thyroid ultrasound in all first-degree relatives.",
            "NBS BORDERLINE TSH: do not dismiss; repeat with thyroid scan — ectopic gland may only partially compensate.",
        ],
        "key_ddx": (
            "FOXE1 Bamforth-Lazarus syndrome: also athyreosis but with choanal atresia + cleft palate + spiky hair (AR); "
            "TSHR RTSH: thyroid hypoplastic but TSHR intact — TSH does respond to TRH; "
            "NKX2-1 (TTF-1) haploinsufficiency: thyroid + lung + brain triad — also causes thyroid dysgenesis; "
            "Isolated thyroid ectopy without a gene: most thyroid ectopy is non-familial and genetically unknown; "
            "Dyshormonogenesis: goiter (large gland) — opposite of PAX8 dysgenesis."
        ),
    },
    # ── TPO — Dyshormonogenesis Type 1 / Complete Organification Defect ──────
    {
        "gene": "TPO",
        "protein": "Thyroid Peroxidase (TPO)",
        "alias": (
            "TPO; OMIM gene 606765; DH1 #274500; 2p25.3; 933 aa; ~103 kDa; AR biallelic; "
            "heme-containing glycoprotein embedded in thyroid follicular cell apical membrane; "
            "catalyzes two key steps: (1) iodination of tyrosine residues on thyroglobulin (organification); "
            "(2) coupling of iodotyrosines (MIT+DIT → T3; DIT+DIT → T4); "
            "requires H2O2 (provided by DUOX2/DUOXA2 system); "
            "LOF → complete organification defect (COD) or partial (POD); "
            "POSITIVE perchlorate discharge test PATHOGNOMONIC for organification defect; "
            "goiter from TSH-driven thyroid enlargement"
        ),
        "aa": "933 aa",
        "kDa": "~103 kDa",
        "locus": "2p25.3",
        "omim_gene": 606765,
        "omim_disease": 274500,
        "inheritance": (
            "AR (biallelic); more common in consanguineous families; most common genetic cause of "
            "dyshormonogenesis among TPO/TG/SLC5A5/DUOX2 group; European and Middle Eastern populations; "
            "heterozygous carriers have borderline high TSH in some studies but rarely clinically affected"
        ),
        "gene_class": (
            "TPO encodes thyroid peroxidase, the key enzyme in thyroid hormone biosynthesis. It is a "
            "heme-containing glycoprotein anchored to the apical membrane of thyroid follicular cells, facing "
            "the follicular lumen, where it acts on thyroglobulin. "
            "STEP 1 — Iodide organification: TPO oxidizes I- (using H2O2 from DUOX2) → I+ → "
            "iodinates tyrosine residues on thyroglobulin → monoiodotyrosine (MIT) and diiodotyrosine (DIT). "
            "STEP 2 — Hormone coupling: TPO couples MIT+DIT → T3; DIT+DIT → T4 (still within thyroglobulin). "
            "TPO LOF → iodide taken up normally by NIS but cannot be organified → iodide stays in inorganic form → "
            "PERCHLORATE DISCHARGE: perchlorate (ClO4-) competes with iodide at NIS, discharging non-organified iodide; "
            ">10% discharge of thyroid radioiodide = organification defect CONFIRMED. "
            "GOITER: thyroid enlarges under TSH stimulation (TSH elevated due to low T4) → large gland not small. "
            "GENETIC NOTE: DUOX2 LOF causes the identical perchlorate-positive picture — distinguish only by genetics."
        ),
        "phenotype": (
            "Congenital hypothyroidism with goiter: TSH 30-300 mU/L at presentation; T4 very low (2-9 pmol/L); "
            "thyroid enlarged on ultrasound (goiter); POSITIVE perchlorate discharge test (>10% discharge); "
            "iodide uptake normal or elevated (NIS intact); "
            "goiter usually regresses with adequate levothyroxine replacement (TSH suppression reduces drive); "
            "severe cases may have obstructive goiter requiring surgery; "
            "differentiation from DUOX2 requires genetic sequencing (identical clinical picture)"
        ),
        "hallmark": (
            "PERCHLORATE DISCHARGE TEST POSITIVE + GOITER = organification defect (TPO or DUOX2) until proven otherwise. "
            "The goiter distinguishes dyshormonogenesis (large gland) from dysgenesis (small/absent gland — PAX8/TSHR/FOXE1). "
            "TPO and DUOX2 are CLINICALLY IDENTICAL — genetic sequencing is the only differentiator. "
            "Goiter regression on levothyroxine confirms TSH-driven enlargement (supports dyshormonogenesis)."
        ),
        "treatment_alerts": [
            "LEVOTHYROXINE LIFELONG: start from neonatal screening; adequate dosing suppresses TSH → goiter regresses.",
            "PERCHLORATE DISCHARGE TEST: request specifically — not part of standard thyroid work-up; nuclear medicine.",
            "GOITER SURVEILLANCE: ultrasound annually; large obstructive goiter may require thyroidectomy.",
            "DO NOT confuse with Hashimoto thyroiditis: TPO antibodies are markers of autoimmune disease — "
            "genetic TPO LOF does not cause antibody production; TPO gene sequencing required.",
            "DUOX2 GENETICS: if perchlorate positive, sequence both TPO AND DUOX2 in same panel.",
            "FAMILY SCREENING: siblings 25% risk (AR); TSH + perchlorate test in symptomatic relatives.",
        ],
        "key_ddx": (
            "DUOX2 LOF: perchlorate positive + goiter — clinically identical; monoallelic DUOX2 = transient; "
            "SLC5A5/NIS ITD: goiter + absent radioiodine uptake — perchlorate negative (NIS absent, no iodide to discharge); "
            "TG LOF: goiter + perchlorate NEGATIVE + very low serum Tg; "
            "Hashimoto thyroiditis (autoimmune): positive anti-TPO antibodies, TPO gene intact; "
            "Iodine deficiency: endemic goiter — responds to iodine supplementation, not isolated to thyroid gene panel."
        ),
    },
    # ── TG — Dyshormonogenesis Type 3 ────────────────────────────────────────
    {
        "gene": "TG",
        "protein": "Thyroglobulin (TG)",
        "alias": (
            "TG; OMIM gene 188450; DH3 #274700; 8q24.22; 2768 aa; ~330 kDa monomer (~660 kDa homodimer); "
            "AR biallelic; the largest human secretory protein; the scaffold for thyroid hormone synthesis; "
            "stored as colloid in thyroid follicular lumen; provides tyrosine residues for iodination by TPO; "
            "LOF → dyshormonogenesis type 3: goiter (TSH-driven) + VERY LOW serum Tg despite goiter; "
            "perchlorate discharge NEGATIVE (iodide organification is intact — NIS and TPO functional); "
            "paradox: large gland, absent Tg scaffold, minimal T4/T3 production"
        ),
        "aa": "2768 aa",
        "kDa": "~330 kDa monomer",
        "locus": "8q24.22",
        "omim_gene": 188450,
        "omim_disease": 274700,
        "inheritance": (
            "AR (biallelic); founder variants in several European and Asian populations; "
            "Dutch, Afrikaner, Japanese reported; heterozygous carriers usually unaffected; "
            "biallelic null variants cause most severe phenotype"
        ),
        "gene_class": (
            "TG encodes thyroglobulin, the 660 kDa homodimeric glycoprotein that constitutes the principal "
            "component of thyroid follicular colloid and serves as the precursor for thyroid hormones. "
            "MECHANISM: follicular cells secrete TG into the follicular lumen → TPO iodinates TG tyrosine "
            "residues → MIT and DIT formed → TPO-catalyzed coupling produces T3 and T4 within TG → "
            "TG-T4/T3 complex re-endocytosed → lysosomal proteolysis releases free T4/T3 into blood. "
            "TG LOF → no scaffold for hormone storage or release → TSH drives thyroid enlargement (goiter) "
            "but despite the large gland, almost no T4/T3 can be stored or secreted. "
            "DIAGNOSTIC PARADOX: goiter (large gland, high TSH) BUT serum thyroglobulin is VERY LOW or "
            "undetectable — this is opposite of thyroid cancer monitoring where Tg is elevated; "
            "NIS and TPO are intact, so iodide is taken up and organified normally → "
            "perchlorate discharge test is NEGATIVE (distinguishes TG LOF from TPO/DUOX2). "
            "SERUM Tg: normally elevated in goiter (Tg released from enlarged gland); "
            "in TG LOF, serum Tg is paradoxically absent → diagnostic hallmark."
        ),
        "phenotype": (
            "Congenital hypothyroidism with goiter: TSH 25-200 mU/L; T4 low (3-10 pmol/L); "
            "large thyroid on ultrasound (goiter); NEGATIVE perchlorate discharge test; "
            "VERY LOW or undetectable serum thyroglobulin despite goiter (paradox); "
            "normal radioiodine uptake (NIS intact); "
            "goiter may be obstructive in severe cases (15% require surgery); "
            "levothyroxine suppresses TSH → partial goiter regression"
        ),
        "hallmark": (
            "GOITER + VERY LOW SERUM THYROGLOBULIN + PERCHLORATE NEGATIVE = TG LOF (Dyshormonogenesis Type 3). "
            "The paradox of a large TSH-driven goiter with absent serum Tg is pathognomonic — "
            "no other form of congenital hypothyroidism combines goiter with absent Tg. "
            "PERCHLORATE NEGATIVE distinguishes TG LOF from TPO/DUOX2 LOF (perchlorate positive). "
            "Serum Tg measurement requires calibration against the patient's TG variant to avoid false-negative "
            "on immunoassay due to variant Tg protein — consider mass spectrometry-based assay."
        ),
        "treatment_alerts": [
            "LEVOTHYROXINE LIFELONG: adequate TSH suppression is essential to reduce goiter drive.",
            "OBSTRUCTIVE GOITER: thyroidectomy if tracheal compression, dysphagia, or cosmetic severity; "
            "post-thyroidectomy levothyroxine dose adjustment mandatory.",
            "SERUM Tg MONITORING: cannot be used for post-thyroidectomy surveillance in TG LOF (Tg is constitutively absent).",
            "PERCHLORATE TEST NEGATIVE: confirm organification is intact (helps localize defect to TG not TPO/DUOX2).",
            "GENETIC PANEL: TG gene has 48 exons — comprehensive sequencing required; MLPA for large deletions.",
            "FAMILY SCREENING: AR — siblings 25% risk; maternal serum Tg in pregnancy may be low if carrier.",
        ],
        "key_ddx": (
            "TPO/DUOX2 LOF: goiter + perchlorate POSITIVE + serum Tg measurable; "
            "SLC5A5 NIS ITD: goiter + absent radioiodine uptake — Tg normal; "
            "PAX8 thyroid dysgenesis: small/ectopic gland, not goiter; Tg may be low but gland is small; "
            "Thyroid cancer: elevated serum Tg in context of thyroid nodule/mass — opposite; "
            "Anti-Tg antibodies: can interfere with Tg immunoassay — check anti-Tg Ab when interpreting Tg."
        ),
    },
    # ── SLC5A5 — Iodide Transport Defect (ITD) / NIS ─────────────────────────
    {
        "gene": "SLC5A5",
        "protein": "Sodium-Iodide Symporter (NIS / SLC5A5)",
        "alias": (
            "SLC5A5; OMIM gene 601843; iodide transport defect ITD #274400; 19p13.11; 643 aa; ~70 kDa; "
            "AR biallelic; electrogenic Na+/I- cotransporter (2 Na+ : 1 I- per cycle); "
            "expressed at basolateral membrane of thyroid follicular cells; concentrates iodide 20-40x vs plasma; "
            "also expressed in salivary glands, gastric mucosa, breast (lactating); "
            "NIS is the basis of radioiodine therapy for thyroid cancer/Graves disease; "
            "LOF → ABSENT thyroid radioiodine uptake WITH goiter (goiter distinguishes from agenesis); "
            "perchlorate discharge NEGATIVE (no iodide in gland to discharge)"
        ),
        "aa": "643 aa",
        "kDa": "~70 kDa",
        "locus": "19p13.11",
        "omim_gene": 601843,
        "omim_disease": 274400,
        "inheritance": (
            "AR (biallelic); rare; worldwide case reports; Japanese founder variants (p.V59E, p.T354P); "
            "consanguinity increases risk; heterozygous carriers have mildly reduced iodide uptake but "
            "typically asymptomatic"
        ),
        "gene_class": (
            "SLC5A5 encodes the sodium-iodide symporter (NIS), the key membrane transporter responsible for "
            "active iodide accumulation in the thyroid. NIS is a 13-transmembrane domain cotransporter that "
            "couples the inward movement of 2 Na+ ions (down their electrochemical gradient, maintained by "
            "Na+/K+-ATPase) with the inward transport of 1 I- ion, concentrating iodide to 20-40-fold above "
            "plasma levels. This active uptake is the first and rate-limiting step of thyroid hormone synthesis. "
            "NIS LOF → thyroid cannot accumulate iodide → despite intact TPO, TG, DUOX2, there is no substrate "
            "for iodination → T4/T3 synthesis fails → congenital hypothyroidism. "
            "IMAGING CHARACTERISTIC: radioiodine (123I) or 99mTc-pertechnetate scan shows ABSENT or greatly "
            "reduced thyroid uptake → CRITICAL diagnostic clue; goiter is present (TSH-driven enlargement) "
            "which distinguishes ITD from thyroid agenesis (agenesis also has absent uptake but no goiter). "
            "SALIVARY/GASTRIC NIS: also absent → low salivary iodide; "
            "IODIDE SUPPLEMENTATION: high-dose iodide can partially saturate residual NIS at supraphysiological "
            "concentrations → may partially correct in hypomorphic variants."
        ),
        "phenotype": (
            "Congenital hypothyroidism with goiter: TSH 40-350 mU/L; T4 very low (2-8 pmol/L); "
            "ABSENT thyroid radioiodine (123I) or 99mTc uptake on scan; "
            "perchlorate discharge test NEGATIVE (no iodide to discharge); "
            "goiter on ultrasound (80%); "
            "salivary and gastric NIS also affected — low salivary iodide concentration; "
            "iodide supplementation partially helps in ~40% (hypomorphic variants); "
            "levothyroxine required lifelong"
        ),
        "hallmark": (
            "ABSENT THYROID RADIOIODINE UPTAKE + GOITER = Iodide Transport Defect (NIS/SLC5A5 LOF) until proven otherwise. "
            "CRITICAL DISTINCTION from dysgenesis: agenesis/ectopy also has absent uptake BUT no goiter; "
            "ITD has absent uptake WITH visible enlarged thyroid gland on ultrasound. "
            "PERCHLORATE NEGATIVE: distinguishes ITD from TPO/DUOX2 (perchlorate positive). "
            "Thyroid scan + ultrasound together = diagnostic combination for ITD."
        ),
        "treatment_alerts": [
            "LEVOTHYROXINE LIFELONG: primary and definitive treatment; start from neonatal screening.",
            "HIGH-DOSE IODIDE SUPPLEMENT: may partially compensate in hypomorphic NIS — trial under endocrinology supervision.",
            "RADIOIODINE THERAPY: will NOT work in NIS LOF (no uptake) — record in chart to prevent future error.",
            "THYROID SCAN (99mTc/123I): mandatory for absent-uptake confirmation — request with ultrasound.",
            "SALIVARY IODIDE: low salivary-to-plasma iodide ratio (<20:1) supports NIS LOF.",
            "FAMILY SCREENING: AR — siblings 25% risk; thyroid function + scan in affected relatives.",
        ],
        "key_ddx": (
            "Thyroid agenesis (PAX8/FOXE1): absent radioiodine uptake + ABSENT gland on ultrasound (vs ITD: goiter present); "
            "TPO/DUOX2 LOF: goiter + POSITIVE perchlorate + normal radioiodine uptake (NIS intact); "
            "TG LOF: goiter + absent serum Tg + NEGATIVE perchlorate + radioiodine uptake normal; "
            "Pendred syndrome (SLC26A4): sensorineural deafness + goiter + partial perchlorate positive; "
            "Iodine deficiency: absent intake but responds to iodine; no gene mutation; endemic."
        ),
    },
    # ── DUOX2 — Dyshormonogenesis Type 6 / Dual Oxidase 2 ────────────────────
    {
        "gene": "DUOX2",
        "protein": "Dual Oxidase 2 (DUOX2)",
        "alias": (
            "DUOX2; OMIM gene 606759; DH6 #607200; 15q21.1; 1548 aa; ~175 kDa; "
            "biallelic AR → severe permanent; monoallelic → milder partial/transient; "
            "NADPH oxidase family; generates H2O2 at apical membrane of thyroid follicular cells; "
            "requires DUOXA2 (maturation factor) for membrane trafficking; "
            "H2O2 is the obligate oxidant for TPO-catalyzed iodination; "
            "DUOX2 null → no H2O2 → TPO cannot iodinate thyroglobulin → organification defect; "
            "POSITIVE perchlorate discharge test (same as TPO LOF); "
            "monoallelic DUOX2 = most common identified cause of TRANSIENT congenital hypothyroidism"
        ),
        "aa": "1548 aa",
        "kDa": "~175 kDa",
        "locus": "15q21.1",
        "omim_gene": 606759,
        "omim_disease": 607200,
        "inheritance": (
            "Biallelic AR → complete organification defect, permanent hypothyroidism; "
            "monoallelic (heterozygous) → partial H2O2 deficiency → transient or mild permanent hypothyroidism; "
            "monoallelic DUOX2 is the most frequently identified genetic cause of transient congenital hypothyroidism "
            "identified in NBS programmes; frequency ~1:30,000-50,000"
        ),
        "gene_class": (
            "DUOX2 encodes dual oxidase 2, a large NADPH oxidase expressed at the apical membrane of thyroid "
            "follicular cells. DUOX2 functions as the H2O2-generating system that provides the obligate "
            "oxidant for thyroid peroxidase (TPO) activity. "
            "MECHANISM: DUOX2 (with its maturation factor DUOXA2) oxidizes NADPH → produces H2O2 at the "
            "apical membrane of follicular cells facing the colloid lumen; H2O2 is then used by TPO as the "
            "electron acceptor for iodination of thyroglobulin tyrosines. Without H2O2, TPO is enzymatically "
            "inactive despite being structurally intact. "
            "BIALLELIC LOF → complete H2O2 deficiency → complete organification defect → perchlorate positive → "
            "severe congenital hypothyroidism with goiter (permanent, requires lifelong treatment). "
            "MONOALLELIC LOF → partial H2O2 deficiency → partial organification defect → "
            "typically transient neonatal hypothyroidism (TSH elevated at NBS, normalizes by 3 years in ~60%) — "
            "but some monoallelic carriers have persistent mild hypothyroidism requiring permanent treatment. "
            "DUOX2 RULE: in monoallelic DUOX2, attempt SUPERVISED levothyroxine cessation at age 3 under "
            "paediatric endocrinology — if TSH normalizes off treatment, hypothyroidism was transient."
        ),
        "phenotype": (
            "Biallelic: severe congenital hypothyroidism; TSH 50-200+ mU/L; goiter; perchlorate discharge positive; "
            "clinical identical to TPO LOF — genetic sequencing required to differentiate; "
            "Monoallelic: TSH 10-50 mU/L at NBS; may or may not have goiter; "
            "perchlorate discharge positive (85%); "
            "TRANSIENT in ~60% of monoallelic cases — TSH normalizes by age 3 without treatment or on minimal levothyroxine; "
            "PERSISTENT in ~40% monoallelic — lifelong mild hypothyroidism; "
            "Family history of transient neonatal hypothyroidism may be the only clue"
        ),
        "hallmark": (
            "MONOALLELIC DUOX2 = MOST COMMON CAUSE OF TRANSIENT CONGENITAL HYPOTHYROIDISM identified by NBS. "
            "RULE: attempt supervised levothyroxine CESSATION TRIAL at age 3 under paediatric endocrinology — "
            "if TSH remains normal after 4-6 weeks off levothyroxine, hypothyroidism was transient. "
            "Do NOT stop levothyroxine without specialist supervision — TSH rebound can be abrupt. "
            "DUOX2 biallelic: permanent, same severity as TPO LOF; "
            "PERCHLORATE POSITIVE in both biallelic and monoallelic DUOX2 — key organification defect marker."
        ),
        "treatment_alerts": [
            "BIALLELIC: levothyroxine lifelong; dose as for any congenital hypothyroidism.",
            "MONOALLELIC: levothyroxine during infancy/childhood; supervised cessation trial at age 3.",
            "CESSATION TRIAL: stop levothyroxine under endocrinology supervision at age 3; TSH at 4-6 weeks; "
            "if TSH >10 → restart; if TSH <5 → hypothyroidism was transient, no further treatment.",
            "DO NOT STOP levothyroxine abruptly without specialist input — especially in children.",
            "GOITER: regresses with adequate TSH suppression on levothyroxine.",
            "SEQUENCE BOTH TPO AND DUOX2: identical perchlorate-positive phenotype; genetic panel needed.",
            "DUOXA2 co-gene: DUOX2 maturation requires DUOXA2 — sequence DUOXA2 if DUOX2 negative but phenotype fits.",
        ],
        "key_ddx": (
            "TPO LOF DH1: perchlorate positive + goiter — clinically identical; only genetics differentiates; "
            "SLC26A4 Pendred: perchlorate positive + SENSORINEURAL DEAFNESS — deafness not in DUOX2; "
            "Iodine deficiency: transient neonatal hypothyroidism in iodine-deficient areas — "
            "DUOX2 monoallelic persists to some degree; iodine status should be tested; "
            "Maternal anti-TPO/TRBAb: transient neonatal hypothyroidism resolving by 3-6 months — "
            "DUOX2 transient resolves by age 3; maternal antibody titre diagnostic."
        ),
    },
    # ── SLC26A4 — Pendred Syndrome ────────────────────────────────────────────
    {
        "gene": "SLC26A4",
        "protein": "Pendrin (SLC26A4)",
        "alias": (
            "SLC26A4; OMIM gene 605646; Pendred syndrome #274600; 7q22.3; 780 aa; ~86 kDa; AR biallelic; "
            "SLC26 family multifunctional anion exchanger: Cl-/I-, Cl-/HCO3-, I-/HCO3-; "
            "expressed in thyroid apical membrane (iodide efflux into follicular lumen), "
            "inner ear endolymph (Cl-/HCO3- exchange for pH homeostasis), kidney intercalated cells; "
            "LOF → Pendred syndrome: SENSORINEURAL DEAFNESS + GOITER (± hypothyroidism) + "
            "Enlarged Vestibular Aqueduct (EVA); "
            "EVA on CT/MRI temporal bone = PATHOGNOMONIC for SLC26A4; "
            "non-syndromic DFNB4 deafness (deafness only) = monoallelic or biallelic SLC26A4 without thyroid phenotype"
        ),
        "aa": "780 aa",
        "kDa": "~86 kDa",
        "locus": "7q22.3",
        "omim_gene": 605646,
        "omim_disease": 274600,
        "inheritance": (
            "AR biallelic (Pendred syndrome); monoallelic SLC26A4 + another variant in CIB2, FOXI1, "
            "or KCNJ13 (digenic) also reported; DFNB4 non-syndromic deafness = biallelic SLC26A4 without "
            "thyroid phenotype (variable expressivity); p.V138F most common European variant"
        ),
        "gene_class": (
            "SLC26A4 encodes pendrin, a member of the SLC26 anion transporter family expressed in three "
            "main tissues: thyroid follicular cells, inner ear endolymph cells, and renal intercalated cells. "
            "THYROID FUNCTION: pendrin mediates iodide efflux from follicular cells into the follicular lumen "
            "(apical membrane); without pendrin, iodide accumulates inside the cell and cannot be "
            "organified by TPO in the colloid → partial organification defect → perchlorate discharge test "
            "partially positive (Wolff-Chaikoff effect disrupted). "
            "INNER EAR: pendrin in the endolymphatic sac and duct maintains endolymph pH and volume via "
            "Cl-/HCO3- exchange; pendrin LOF → endolymph acidosis → cochlear hair cell degeneration → "
            "sensorineural deafness (often prelingual, severe to profound); "
            "enlarged vestibular aqueduct (EVA) → progressive deafness with head trauma. "
            "EVA: the endolymphatic duct and sac enlarge (>1.5 mm at midpoint on CT) — "
            "this is the most consistent anatomical finding in SLC26A4 and is pathognomonic. "
            "GOITER: TSH-driven thyroid enlargement when iodide organification is impaired; "
            "many Pendred patients are euthyroid (partial defect)."
        ),
        "phenotype": (
            "PENDRED SYNDROME: bilateral sensorineural deafness (prelingual, severe to profound) + goiter (euthyroid 50%) + EVA on CT; "
            "deafness onset: birth to 24 months; goiter onset: childhood to adulthood (5-30 years); "
            "TSH: often normal (euthyroid, 50%) or mildly elevated (3-45 mU/L); T4 usually normal or low-normal; "
            "perchlorate discharge test: positive (75%) — partial organification defect; "
            "CT temporal bone: enlarged vestibular aqueduct (>1.5 mm) = pathognomonic (85%); "
            "Mondini cochlear malformation in some; "
            "cochlear implantation outcomes good (65%); "
            "progressive deafness — head trauma or pressure changes worsen"
        ),
        "hallmark": (
            "SENSORINEURAL DEAFNESS + GOITER = Pendred syndrome until proven otherwise. "
            "EVA (enlarged vestibular aqueduct >1.5 mm on CT temporal bone) is PATHOGNOMONIC for SLC26A4. "
            "Every child with bilateral sensorineural deafness and EVA needs: SLC26A4 sequencing + thyroid function + "
            "thyroid ultrasound (goiter may not be clinically obvious). "
            "TRAUMA RISK: EVA → progressive fluctuating deafness with minor head trauma — counsel families and "
            "restrict contact sports."
        ),
        "treatment_alerts": [
            "COCHLEAR IMPLANT: indicated for severe-profound deafness; outcomes in SLC26A4 are generally good.",
            "LEVOTHYROXINE: only if hypothyroid (TSH elevated); many Pendred patients are euthyroid.",
            "AVOID HEAD TRAUMA: EVA → progressive deafness worsened by trauma/pressure — no contact sports.",
            "CT TEMPORAL BONE: mandatory for bilateral sensorineural deafness diagnosis — confirms EVA.",
            "AUDIOLOGICAL MONITORING: annual audiogram; deafness may fluctuate and progress.",
            "THYROID ULTRASOUND: every 2-3 years for goiter surveillance even if euthyroid.",
            "FAMILY SCREENING: AR — siblings 25% risk; hearing screen + thyroid function in relatives.",
        ],
        "key_ddx": (
            "GJB2/Connexin 26 deafness: most common genetic deafness — NO thyroid and NO EVA; "
            "Usher syndrome: deafness + RETINITIS PIGMENTOSA (not goiter); "
            "Jervell-Lange-Nielsen: deafness + QTc PROLONGATION (cardiac) — no goiter; "
            "DFNB4 (non-syndromic SLC26A4): EVA + deafness but no thyroid phenotype — "
            "overlapping genotype, different expressivity; "
            "TPO/DUOX2: perchlorate positive + goiter but NO deafness and NO EVA."
        ),
    },
    # ── FOXE1 — Bamforth-Lazarus Syndrome ────────────────────────────────────
    {
        "gene": "FOXE1",
        "protein": "Forkhead Box Protein E1 / Thyroid Transcription Factor-2 (FOXE1/TTF-2)",
        "alias": (
            "FOXE1; OMIM gene 602617; Bamforth-Lazarus syndrome #241850; 9q22.33; 373 aa; ~42 kDa; "
            "AR biallelic; forkhead/winged-helix transcription factor; essential for thyroid anlage migration "
            "from foramen caecum → final cervical position (thyroid migration) AND for palatogenesis + choanal "
            "development; biallelic LOF → ATHYREOSIS (complete thyroid agenesis) + CLEFT PALATE + "
            "CHOANAL ATRESIA + SPIKY HAIR (bifid epiglottis in some); "
            "choanal atresia = life-threatening at birth — neonates are obligate nasal breathers; "
            "GWAS: common FOXE1 variants (polyalanine tract length) associated with sporadic thyroid cancer and "
            "non-syndromic thyroid dysgenesis"
        ),
        "aa": "373 aa",
        "kDa": "~42 kDa",
        "locus": "9q22.33",
        "omim_gene": 602617,
        "omim_disease": 241850,
        "inheritance": (
            "AR biallelic (Bamforth-Lazarus syndrome); extremely rare — <50 families worldwide; "
            "common polyalanine expansion variants are associated with thyroid cancer risk (not syndromic); "
            "heterozygous LOF: some reports of partial thyroid dysgenesis (reduced penetrance)"
        ),
        "gene_class": (
            "FOXE1 (formerly TTF-2) encodes a forkhead domain transcription factor expressed in the thyroid "
            "anlage, anterior pituitary, Rathke's pouch, palatal shelves, and embryonic epiglottis. "
            "THYROID MIGRATION: the thyroid anlage arises at the foramen caecum (base of tongue) at embryonic "
            "day E22 and migrates caudally to its final cervical position. FOXE1 is required for this migration "
            "— without FOXE1, the anlage fails to detach and migrate → thyroid arrested at base of tongue "
            "or athyreosis (complete failure of thyroid development). "
            "Unlike PAX8 ectopy where a functional remnant persists, FOXE1 biallelic LOF causes complete "
            "athyreosis in most cases — no functional thyroid tissue at any location. "
            "PALATE AND CHOANAE: FOXE1 is expressed in developing palatal shelves and choanae; "
            "biallelic LOF → bilateral choanal atresia (bony or membranous obstruction of nasal airways) "
            "AND cleft palate — these occur together in Bamforth-Lazarus syndrome. "
            "CHOANAL ATRESIA is a NEONATAL EMERGENCY: neonates are obligate nasal breathers; "
            "bilateral choanal atresia → severe respiratory distress at birth → immediate airway intervention. "
            "HAIR: spiky/bristle hair phenotype is characteristic; bifid epiglottis in a subset."
        ),
        "phenotype": (
            "BAMFORTH-LAZARUS SYNDROME: neonatal onset; "
            "SEVERE HYPOTHYROIDISM from day 1 (athyreosis — no thyroid tissue): TSH very high, T4 absent; "
            "BILATERAL CHOANAL ATRESIA (90%): respiratory distress, cyanosis, relief with crying (mouth breathing); "
            "CLEFT PALATE (80%): variable extent; "
            "SPIKY/BRISTLE HAIR (70%): characteristic from birth; "
            "BIFID EPIGLOTTIS in some cases; "
            "athyreosis confirmed by absent thyroid on ultrasound and scan (95%); "
            "immediate airway intervention required at birth (85%)"
        ),
        "hallmark": (
            "CHOANAL ATRESIA + HYPOTHYROIDISM + CLEFT PALATE IN NEONATE = Bamforth-Lazarus syndrome until proven otherwise. "
            "Bilateral choanal atresia is the presenting emergency — neonates are obligate nasal breathers; "
            "bilateral obstruction → cyanosis relieved only by crying (mouth breathing). "
            "ALL neonates with bilateral choanal atresia need immediate thyroid function testing. "
            "SPIKY HAIR is a specific clinical sign — photograph at birth for records. "
            "ATHYREOSIS: thyroid completely absent on scan — no ectopic tissue (unlike PAX8 ectopy)."
        ),
        "treatment_alerts": [
            "IMMEDIATE AIRWAY: McGovern nipple (oropharyngeal airway) or surgical choanal repair — "
            "this is a neonatal airway emergency; call ENT/NICU immediately.",
            "LEVOTHYROXINE FROM DAY 1: highest priority after airway stabilization; "
            "dose 10-15 mcg/kg/day; athyreosis → levothyroxine forever.",
            "CLEFT PALATE REPAIR: paediatric craniofacial surgery; timing per surgical team (usually 6-12 months).",
            "SPIKY HAIR: cosmetic only; no treatment needed; document as diagnostic sign.",
            "DO NOT DELAY THYROID TREATMENT waiting for genetics — treat severe hypothyroidism immediately.",
            "DEVELOPMENTAL FOLLOW-UP: monitor neurodevelopment; early thyroid treatment is critical for brain.",
            "FAMILY SCREENING: AR — siblings 25% risk; prenatal FOXE1 sequencing available in subsequent pregnancies.",
        ],
        "key_ddx": (
            "PAX8 thyroid dysgenesis: ectopic gland possible (sublingual), NO choanal atresia, NO cleft palate, "
            "spiky hair absent — PAX8 is AD, FOXE1 is AR; "
            "CHARGE syndrome (CHD7): choanal atresia + coloboma + heart defect + ear anomalies — no thyroid agenesis typically; "
            "22q11 deletion / DiGeorge: cleft palate + hypocalcaemia + cardiac — thyroid usually intact; "
            "Isolated cleft palate (non-syndromic): no choanal atresia, no thyroid involvement; "
            "TSHR RTSH: thyroid hypoplastic but NOT absent; no choanal atresia or cleft palate."
        ),
    },
]


def _make_cohort(gene_info: dict, seed: int) -> list:
    """Generate a 40-patient cohort for one gene."""
    rng = random.Random(seed)
    gene = gene_info["gene"]
    pts = []
    for i in range(40):
        age = rng.randint(0, 45)
        sex = rng.choice(["M", "F"])
        p: dict = {"id": i + 1, "age": age, "sex": sex}

        if gene == "TSHR":
            p["age_onset_months"] = rng.randint(0, 12)
            p["tsh_level"] = rng.randint(50, 250)
            p["t4_level"] = rng.randint(2, 8)
            p["thyroid_volume_ml"] = round(rng.uniform(0.5, 3.0), 1)
            p["levothyroxine_dose_mcg"] = rng.randint(25, 150)
            p["tsh_after_trh_unchanged"] = rng.random() < 0.90  # LOF: unchanged in RTSH
            p["levothyroxine_prescribed"] = rng.random() < 0.95
        elif gene == "PAX8":
            p["age_onset_months"] = rng.randint(0, 6)
            p["tsh_level"] = rng.randint(20, 120)
            p["t4_level"] = rng.randint(3, 10)
            p["thyroid_ectopic"] = rng.random() < 0.70
            p["gland_absent"] = (not p["thyroid_ectopic"]) and rng.random() < 0.20
            p["renal_anomaly"] = rng.random() < 0.20
            p["mullerian_aplasia"] = (sex == "F") and rng.random() < 0.15
            p["levothyroxine_prescribed"] = rng.random() < 0.95
        elif gene == "TPO":
            p["age_onset_months"] = rng.randint(0, 18)
            p["tsh_level"] = rng.randint(30, 300)
            p["t4_level"] = rng.randint(2, 9)
            p["goiter"] = rng.random() < 0.90
            p["perchlorate_discharge_positive"] = rng.random() < 0.95
            p["goiter_regression_on_levo"] = rng.random() < 0.75
            p["levothyroxine_prescribed"] = rng.random() < 0.95
        elif gene == "TG":
            p["age_onset_months"] = rng.randint(0, 24)
            p["tsh_level"] = rng.randint(25, 200)
            p["t4_level"] = rng.randint(3, 10)
            p["thyroglobulin_serum_low"] = rng.random() < 0.90
            p["goiter"] = rng.random() < 0.85
            p["perchlorate_discharge_positive"] = rng.random() < 0.05
            p["goiter_surgical"] = p["goiter"] and rng.random() < 0.15
            p["levothyroxine_prescribed"] = rng.random() < 0.95
        elif gene == "SLC5A5":
            p["age_onset_months"] = rng.randint(0, 24)
            p["tsh_level"] = rng.randint(40, 350)
            p["t4_level"] = rng.randint(2, 8)
            p["radioiodine_uptake_absent"] = rng.random() < 0.95
            p["goiter"] = rng.random() < 0.80
            p["perchlorate_negative"] = rng.random() < 0.95
            p["iodide_supplement_helps"] = rng.random() < 0.40
            p["levothyroxine_prescribed"] = rng.random() < 0.95
        elif gene == "DUOX2":
            p["age_onset_months"] = rng.randint(0, 12)
            p["tsh_level"] = rng.randint(10, 150)
            p["monoallelic"] = rng.random() < 0.50
            p["transient_hypo"] = p["monoallelic"] and rng.random() < 0.60
            p["perchlorate_positive"] = rng.random() < 0.85
            p["goiter"] = rng.random() < 0.70
            p["levothyroxine_prescribed"] = rng.random() < 0.90
            p["cessation_trial_at_3"] = p["monoallelic"] and rng.random() < 0.70
        elif gene == "SLC26A4":
            p["age_onset_deafness_months"] = rng.randint(0, 24)
            p["age_onset_goiter_years"] = rng.randint(5, 30)
            p["tsh_level"] = round(rng.uniform(3.0, 45.0), 1)
            p["t4_level"] = round(rng.uniform(8.0, 20.0), 1)
            p["eva_on_ct"] = rng.random() < 0.85
            p["cochlear_implant"] = rng.random() < 0.65
            p["euthyroid"] = rng.random() < 0.50
            p["perchlorate_positive"] = rng.random() < 0.75
            p["goiter"] = rng.random() < 0.80
            p["levothyroxine_prescribed"] = not p["euthyroid"] and rng.random() < 0.85
        elif gene == "FOXE1":
            p["age_onset_days"] = rng.randint(0, 7)
            p["severe_hypo"] = True
            p["choanal_atresia"] = rng.random() < 0.90
            p["cleft_palate"] = rng.random() < 0.80
            p["spiky_hair"] = rng.random() < 0.70
            p["athyreosis"] = rng.random() < 0.95
            p["immediate_airway_intervention"] = p["choanal_atresia"] and rng.random() < 0.95
            p["levothyroxine_prescribed"] = True
        pts.append(p)
    return pts


_ALL_COHORTS = {
    g["gene"]: _make_cohort(g, SEED_BASE + i)
    for i, g in enumerate(THYROID_GENES)
}


def _pct(pts: list, key: str) -> float:
    vals = [p[key] for p in pts if key in p and p[key] is not None]
    return round(sum(bool(v) for v in vals) / len(vals) * 100, 1) if vals else 0.0


def _avg(pts: list, key: str) -> float:
    vals = [p[key] for p in pts if key in p and p[key] is not None and isinstance(p[key], (int, float))]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def get_overview() -> dict:
    all_pts = [p for pts in _ALL_COHORTS.values() for p in pts]
    n = len(all_pts)

    # Atlas-level aggregate stats across all 320 patients
    all_tsh = []
    all_t4 = []
    for gene, pts in _ALL_COHORTS.items():
        for p in pts:
            if "tsh_level" in p and isinstance(p["tsh_level"], (int, float)):
                all_tsh.append(p["tsh_level"])
            if "t4_level" in p and isinstance(p["t4_level"], (int, float)):
                all_t4.append(p["t4_level"])

    median_tsh = round(sorted(all_tsh)[len(all_tsh) // 2], 1) if all_tsh else 0.0
    median_t4 = round(sorted(all_t4)[len(all_t4) // 2], 1) if all_t4 else 0.0

    levo_count = sum(_pct(_ALL_COHORTS[g["gene"]], "levothyroxine_prescribed") for g in THYROID_GENES) / 8
    goiter_count = sum(_pct(_ALL_COHORTS[g["gene"]], "goiter") for g in THYROID_GENES if any("goiter" in p for p in _ALL_COHORTS[g["gene"]])) / len([g for g in THYROID_GENES if any("goiter" in p for p in _ALL_COHORTS[g["gene"]])])

    return {
        "atlas_name": "Thyroid-Disorders-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Thyroid Disorders Atlas",
        "n_patients": n,
        "gene_count": len(THYROID_GENES),
        "genes": [g["gene"] for g in THYROID_GENES],
        "seeds": "1302–1309",
        "registered": "2026-09-05",
        "atlas_version": "1.0",
        "gene_summary": [
            {
                "gene": "TSHR",
                "protein": "TSH Receptor",
                "aa": "764 aa",
                "locus": "14q31.1",
                "inheritance": "AR LOF (RTSH) / AD GOF (FNAH)",
                "phenotype_short": "Resistance to TSH: congenital hypothyroidism, thyroid hypoplasia, TSH very high",
                "hallmark_short": "TSH high + T4 low + hypoplastic thyroid + TSH unresponsive to TRH = RTSH",
            },
            {
                "gene": "PAX8",
                "protein": "Paired Box Protein Pax-8",
                "aa": "450 aa",
                "locus": "2q14.1",
                "inheritance": "AD (haploinsufficiency, 50% penetrance)",
                "phenotype_short": "Thyroid dysgenesis: agenesis, hypoplasia, or ectopia (sublingual most common)",
                "hallmark_short": "Ectopic sublingual thyroid — DO NOT remove without nuclear medicine scan",
            },
            {
                "gene": "TPO",
                "protein": "Thyroid Peroxidase",
                "aa": "933 aa",
                "locus": "2p25.3",
                "inheritance": "AR biallelic",
                "phenotype_short": "Dyshormonogenesis DH1: congenital hypothyroidism, goiter, organification defect",
                "hallmark_short": "Perchlorate discharge test POSITIVE + goiter = organification defect",
            },
            {
                "gene": "TG",
                "protein": "Thyroglobulin",
                "aa": "2768 aa",
                "locus": "8q24.22",
                "inheritance": "AR biallelic",
                "phenotype_short": "Dyshormonogenesis DH3: goiter + very low serum Tg despite goiter",
                "hallmark_short": "Goiter + VERY LOW serum Tg + perchlorate NEGATIVE = TG LOF",
            },
            {
                "gene": "SLC5A5",
                "protein": "Sodium-Iodide Symporter (NIS)",
                "aa": "643 aa",
                "locus": "19p13.11",
                "inheritance": "AR biallelic",
                "phenotype_short": "Iodide transport defect: absent radioiodine uptake, goiter, hypothyroidism",
                "hallmark_short": "ABSENT radioiodine uptake WITH goiter = NIS/ITD — distinguishes from agenesis",
            },
            {
                "gene": "DUOX2",
                "protein": "Dual Oxidase 2",
                "aa": "1548 aa",
                "locus": "15q21.1",
                "inheritance": "AR biallelic (severe) / monoallelic (transient/mild)",
                "phenotype_short": "DH6: organification defect; monoallelic = most common transient congenital hypothyroidism",
                "hallmark_short": "Monoallelic DUOX2: attempt cessation trial at age 3 under endocrinology",
            },
            {
                "gene": "SLC26A4",
                "protein": "Pendrin",
                "aa": "780 aa",
                "locus": "7q22.3",
                "inheritance": "AR biallelic",
                "phenotype_short": "Pendred syndrome: sensorineural deafness + goiter + enlarged vestibular aqueduct",
                "hallmark_short": "EVA on CT temporal bone is pathognomonic; deafness + goiter = Pendred",
            },
            {
                "gene": "FOXE1",
                "protein": "Forkhead Box E1 (TTF-2)",
                "aa": "373 aa",
                "locus": "9q22.33",
                "inheritance": "AR biallelic",
                "phenotype_short": "Bamforth-Lazarus: athyreosis + choanal atresia + cleft palate + spiky hair",
                "hallmark_short": "Choanal atresia + hypothyroidism + cleft palate in neonate = Bamforth-Lazarus emergency",
            },
        ],
        "aggregate_clinical": {
            "neonatal_screening_detected_pct": round(
                sum(_pct(_ALL_COHORTS[g["gene"]], "levothyroxine_prescribed") for g in THYROID_GENES) / 8, 1
            ),
            "levothyroxine_prescribed_pct": round(levo_count, 1),
            "goiter_pct": round(
                sum(1 for g in THYROID_GENES for p in _ALL_COHORTS[g["gene"]] if p.get("goiter")) / n * 100, 1
            ),
            "deafness_pct": round(
                sum(1 for p in _ALL_COHORTS["SLC26A4"] if "age_onset_deafness_months" in p) / n * 100, 1
            ),
            "choanal_atresia_pct": round(
                sum(1 for p in _ALL_COHORTS["FOXE1"] if p.get("choanal_atresia")) / n * 100, 1
            ),
            "perchlorate_positive_pct": round(
                sum(1 for g in THYROID_GENES for p in _ALL_COHORTS[g["gene"]]
                    if p.get("perchlorate_discharge_positive") or p.get("perchlorate_positive")) / n * 100, 1
            ),
            "radioiodine_absent_pct": round(
                sum(1 for p in _ALL_COHORTS["SLC5A5"] if p.get("radioiodine_uptake_absent")) / n * 100, 1
            ),
            "median_tsh_at_diagnosis": median_tsh,
            "median_t4_at_diagnosis": median_t4,
        },
        "cascade_testing_note": (
            "All first-degree relatives of a proband with hereditary thyroid disorder should receive "
            "targeted gene panel testing plus thyroid function (TSH, free T4) and relevant imaging "
            "(thyroid ultrasound; CT temporal bone for SLC26A4). "
            "AR disorders (TSHR LOF, TPO, TG, SLC5A5, DUOX2 biallelic, SLC26A4, FOXE1): 25% recurrence risk in siblings. "
            "AD disorders (PAX8, TSHR GOF): 50% offspring risk with variable penetrance."
        ),
        "clinical_pearl": (
            "Congenital hypothyroidism from thyroid DYSGENESIS (PAX8, FOXE1, TSHR LOF) presents with a "
            "SMALL or ABSENT thyroid on imaging; from DYSHORMONOGENESIS (TPO, TG, SLC5A5, DUOX2) presents with "
            "a LARGE thyroid (goiter). This single imaging distinction narrows the genetic differential "
            "before sequencing. Pendred syndrome (SLC26A4) is the exception: goiter + deafness — "
            "bilateral sensorineural deafness in any child with thyroid disease mandates EVA workup."
        ),
    }


def get_breakdown() -> dict:
    out: dict = {}
    for ginfo in THYROID_GENES:
        gene = ginfo["gene"]
        pts = _ALL_COHORTS[gene]
        stats: dict = {
            "n": len(pts),
            "sex_m_pct": round(sum(1 for p in pts if p.get("sex") == "M") / len(pts) * 100, 1),
        }
        if gene == "TSHR":
            stats.update({
                "tsh_mean": _avg(pts, "tsh_level"),
                "t4_mean": _avg(pts, "t4_level"),
                "thyroid_volume_mean_ml": _avg(pts, "thyroid_volume_ml"),
                "tsh_unresponsive_to_trh_pct": _pct(pts, "tsh_after_trh_unchanged"),
                "levothyroxine_pct": _pct(pts, "levothyroxine_prescribed"),
            })
        elif gene == "PAX8":
            stats.update({
                "tsh_mean": _avg(pts, "tsh_level"),
                "t4_mean": _avg(pts, "t4_level"),
                "thyroid_ectopic_pct": _pct(pts, "thyroid_ectopic"),
                "gland_absent_pct": _pct(pts, "gland_absent"),
                "renal_anomaly_pct": _pct(pts, "renal_anomaly"),
                "levothyroxine_pct": _pct(pts, "levothyroxine_prescribed"),
            })
        elif gene == "TPO":
            stats.update({
                "tsh_mean": _avg(pts, "tsh_level"),
                "t4_mean": _avg(pts, "t4_level"),
                "goiter_pct": _pct(pts, "goiter"),
                "perchlorate_positive_pct": _pct(pts, "perchlorate_discharge_positive"),
                "goiter_regression_on_levo_pct": _pct(pts, "goiter_regression_on_levo"),
                "levothyroxine_pct": _pct(pts, "levothyroxine_prescribed"),
            })
        elif gene == "TG":
            stats.update({
                "tsh_mean": _avg(pts, "tsh_level"),
                "t4_mean": _avg(pts, "t4_level"),
                "thyroglobulin_low_pct": _pct(pts, "thyroglobulin_serum_low"),
                "goiter_pct": _pct(pts, "goiter"),
                "perchlorate_positive_pct": _pct(pts, "perchlorate_discharge_positive"),
                "goiter_surgical_pct": _pct(pts, "goiter_surgical"),
                "levothyroxine_pct": _pct(pts, "levothyroxine_prescribed"),
            })
        elif gene == "SLC5A5":
            stats.update({
                "tsh_mean": _avg(pts, "tsh_level"),
                "t4_mean": _avg(pts, "t4_level"),
                "radioiodine_absent_pct": _pct(pts, "radioiodine_uptake_absent"),
                "goiter_pct": _pct(pts, "goiter"),
                "perchlorate_negative_pct": _pct(pts, "perchlorate_negative"),
                "iodide_supplement_helps_pct": _pct(pts, "iodide_supplement_helps"),
                "levothyroxine_pct": _pct(pts, "levothyroxine_prescribed"),
            })
        elif gene == "DUOX2":
            stats.update({
                "tsh_mean": _avg(pts, "tsh_level"),
                "monoallelic_pct": _pct(pts, "monoallelic"),
                "transient_hypo_pct": _pct(pts, "transient_hypo"),
                "perchlorate_positive_pct": _pct(pts, "perchlorate_positive"),
                "goiter_pct": _pct(pts, "goiter"),
                "cessation_trial_at_3_pct": _pct(pts, "cessation_trial_at_3"),
                "levothyroxine_pct": _pct(pts, "levothyroxine_prescribed"),
            })
        elif gene == "SLC26A4":
            stats.update({
                "tsh_mean": _avg(pts, "tsh_level"),
                "t4_mean": _avg(pts, "t4_level"),
                "eva_on_ct_pct": _pct(pts, "eva_on_ct"),
                "cochlear_implant_pct": _pct(pts, "cochlear_implant"),
                "euthyroid_pct": _pct(pts, "euthyroid"),
                "perchlorate_positive_pct": _pct(pts, "perchlorate_positive"),
                "goiter_pct": _pct(pts, "goiter"),
                "levothyroxine_pct": _pct(pts, "levothyroxine_prescribed"),
            })
        elif gene == "FOXE1":
            stats.update({
                "severe_hypo_pct": _pct(pts, "severe_hypo"),
                "choanal_atresia_pct": _pct(pts, "choanal_atresia"),
                "cleft_palate_pct": _pct(pts, "cleft_palate"),
                "spiky_hair_pct": _pct(pts, "spiky_hair"),
                "athyreosis_pct": _pct(pts, "athyreosis"),
                "immediate_airway_intervention_pct": _pct(pts, "immediate_airway_intervention"),
                "levothyroxine_pct": _pct(pts, "levothyroxine_prescribed"),
            })
        out[gene] = {
            "gene": gene,
            "protein": ginfo["protein"],
            "aa": ginfo["aa"],
            "kDa": ginfo["kDa"],
            "locus": ginfo["locus"],
            "omim_gene": ginfo["omim_gene"],
            "omim_disease": ginfo["omim_disease"],
            "inheritance": ginfo["inheritance"],
            "gene_class": ginfo["gene_class"],
            "phenotype": ginfo["phenotype"],
            "hallmark": ginfo["hallmark"],
            "treatment_alerts": ginfo["treatment_alerts"],
            "key_ddx": ginfo["key_ddx"],
            "cohort_stats": stats,
        }
    return {"breakdown": out}


def get_definitions() -> dict:
    return {
        "definitions": {
            "Congenital_Hypothyroidism_Overview": (
                "Congenital hypothyroidism (CH) is one of the most common preventable causes of intellectual "
                "disability, affecting approximately 1 in 2,000-4,000 newborns worldwide. "
                "Neonatal screening (heel-prick TSH or T4 measurement) detects most cases before symptoms "
                "develop; early levothyroxine treatment (within 2 weeks of birth) normalizes neurodevelopmental "
                "outcomes in the majority of affected infants. "
                "Causes are broadly divided into thyroid dysgenesis (structural, ~85% of permanent CH: PAX8, "
                "FOXE1, TSHR LOF, NKX2-1) and dyshormonogenesis (functional, ~15%: TPO, TG, SLC5A5, DUOX2, SLC26A4). "
                "Permanent CH (all dysgenesis forms and biallelic dyshormonogenesis) requires lifelong "
                "levothyroxine; transient CH (monoallelic DUOX2, maternal antibodies, iodine excess) may "
                "resolve and warrants a supervised cessation trial at age 3 under paediatric endocrinology."
            ),
            "Thyroid_Dyshormonogenesis_vs_Dysgenesis": (
                "The single most useful initial distinction in hereditary congenital hypothyroidism is "
                "DYSGENESIS (structural thyroid defect) versus DYSHORMONOGENESIS (enzymatic thyroid defect). "
                "Dysgenesis: thyroid gland is ABSENT, SMALL, or ECTOPIC on ultrasound and nuclear medicine scan; "
                "genes: PAX8 (ectopia most common), FOXE1 (athyreosis + Bamforth-Lazarus syndrome), "
                "TSHR LOF (hypoplasia), NKX2-1. Radioiodine uptake absent or reduced; no goiter. "
                "Dyshormonogenesis: thyroid gland is ENLARGED (goiter) because TSH is high and drives "
                "gland growth, but the gland cannot make hormones properly; genes: TPO, TG, SLC5A5, DUOX2, "
                "SLC26A4 (partial). Radioiodine uptake usually present and may be elevated (NIS intact). "
                "Pendred syndrome (SLC26A4) occupies a middle ground — goiter with partial organification "
                "defect, but deafness is the dominant clinical feature. "
                "CLINICAL PEARL: imaging the thyroid first (ultrasound + 99mTc scan) narrows the differential "
                "before sequencing — small/absent gland → dysgenesis gene panel; goiter → dyshormonogenesis panel."
            ),
            "TSH_Receptor_Resistance_RTSH": (
                "Resistance to TSH (RTSH, #275200) is caused by biallelic inactivating variants in TSHR, "
                "the TSH receptor gene on chromosome 14q31.1. The TSH receptor is a Gs-protein-coupled receptor "
                "with a 7-transmembrane domain; TSH binding activates adenylate cyclase → cAMP → PKA → "
                "thyroid transcription factor activation → thyroglobulin/TPO/NIS expression → T4/T3 synthesis. "
                "RTSH hallmark biochemistry: TSH very high (50-250 mU/L), T4 very low, thyroid gland hypoplastic "
                "on imaging (0.5-3 mL instead of normal 3-5 mL in a neonate); the thyroid scan shows a small "
                "in-situ gland at the correct cervical location (distinguishes from PAX8 ectopy). "
                "TRH STIMULATION TEST: after TRH injection, TSH rises (pituitary is normal and responds) "
                "but T4 does NOT rise — the thyroid cannot transduce the TSH signal — this flat T4 response "
                "to TRH is pathognomonic for RTSH. "
                "CRITICAL RULE: maternal TSH-receptor blocking antibodies (TRBAb) cause a clinically identical "
                "but TRANSIENT picture resolving by 3-6 months; if TSH does not normalize by 3 years, "
                "maternal antibodies are excluded and TSHR sequencing is mandatory."
            ),
            "Perchlorate_Discharge_Test": (
                "The perchlorate discharge test is the key functional test for thyroid organification defects, "
                "caused by TPO, DUOX2, or SLC26A4 (partial) LOF. "
                "PRINCIPLE: the thyroid takes up radioiodine (123I) via NIS; organification by TPO converts "
                "free inorganic iodide into protein-bound iodotyrosines within thyroglobulin; "
                "once iodide is organified (bound to TG), it cannot be displaced by perchlorate; "
                "if iodide remains in the inorganic free form (organification defect), perchlorate (ClO4-) "
                "given 2 hours after radioiodine will compete at NIS and discharge the unbound iodide. "
                "INTERPRETATION: >10% discharge of thyroid radioiodide within 1 hour of perchlorate = "
                "organification defect CONFIRMED; >50% discharge = complete defect (biallelic TPO or DUOX2). "
                "NEGATIVE PERCHLORATE: expected in NIS/SLC5A5 LOF (no iodide to discharge), TG LOF "
                "(iodide organified normally but no scaffold for storage), dysgenesis (no gland). "
                "This test is not widely available — request from specialist nuclear medicine centre; "
                "genetic panel now often replaces it in comprehensive work-up."
            ),
            "PAX8_Thyroid_Ectopy_Rule": (
                "PAX8 haploinsufficiency causes thyroid dysgenesis with the most common structural abnormality "
                "being thyroid ectopy — the gland fails to complete its migration from the foramen caecum "
                "at the base of the tongue to its normal cervical position, leaving a residual thyroid remnant "
                "at the sublingual position, along the thyroglossal duct, or at the base of the tongue. "
                "The CRITICAL clinical rule is: DO NOT surgically remove a sublingual thyroid mass before "
                "confirming by nuclear medicine scan (99mTc or 123I) that it is non-functional and that "
                "normal cervical thyroid tissue is present — in many cases the ectopic sublingual gland is "
                "the ONLY thyroid tissue; removing it renders the patient permanently athyreotic. "
                "Clinical scenario: a child presents with a painless mass at the base of the tongue; "
                "thyroid function shows elevated TSH — this gland is working harder because it is the only "
                "functional tissue; request scan before ENT proceeds to surgery. "
                "MANAGEMENT: levothyroxine suppresses TSH → ectopic gland often shrinks without surgery; "
                "surgery (thyroglossal duct cyst excision) reserved for truly non-functional ectopic tissue "
                "only after thyroid imaging protocol confirms in-situ cervical gland is present."
            ),
            "NIS_Iodide_Transport_Defect": (
                "The iodide transport defect (ITD, #274400) is caused by biallelic inactivating variants in "
                "SLC5A5, the sodium-iodide symporter gene. NIS is expressed at the basolateral membrane of "
                "thyroid follicular cells and actively concentrates iodide 20-40 fold above plasma by coupling "
                "iodide uptake to the Na+ electrochemical gradient. "
                "Without NIS, the thyroid cannot accumulate iodide — the first and rate-limiting step of "
                "thyroid hormone synthesis — so despite intact TPO, TG, and DUOX2, T4/T3 cannot be made. "
                "IMAGING HALLMARK: thyroid scan (99mTc or 123I) shows ABSENT or severely reduced uptake; "
                "crucially the thyroid GLAND IS VISIBLE AND ENLARGED on ultrasound (goiter from TSH drive) — "
                "this combination of absent scan uptake with visible goiter is pathognomonic for ITD; "
                "thyroid agenesis also has absent uptake but NO gland is visible. "
                "NIS is also expressed in salivary glands and gastric mucosa — the salivary/plasma iodide "
                "concentration ratio is <10:1 in ITD (normally >20:1), providing a functional NIS assay. "
                "Treatment: levothyroxine lifelong; high-dose iodide supplement can partially compensate "
                "via passive diffusion in hypomorphic variants with residual NIS activity."
            ),
            "DUOX2_Transient_Hypothyroidism_Rule": (
                "DUOX2 monoallelic (heterozygous) pathogenic variants are the most common identified genetic "
                "cause of transient congenital hypothyroidism detected by neonatal screening programmes. "
                "In monoallelic DUOX2, H2O2 production by DUOX2 is reduced by ~50% — this partial deficit "
                "is sufficient to cause elevated TSH at NBS in the neonatal period when iodine demands are "
                "highest, but as thyroid reserve capacity matures and dietary iodine intake stabilizes, "
                "the remaining DUOX2 allele may provide sufficient H2O2 → TSH normalizes → the "
                "hypothyroidism is TRANSIENT (approximately 60% of monoallelic carriers in published series). "
                "THE RULE: all monoallelic DUOX2 patients should undergo a SUPERVISED CESSATION TRIAL at "
                "age 3 years under paediatric endocrinology — levothyroxine is stopped, TSH is measured at "
                "4-6 weeks; if TSH remains <10 mU/L, hypothyroidism was transient and treatment is not "
                "restarted; if TSH rises >10 mU/L, hypothyroidism is permanent and levothyroxine is resumed. "
                "DO NOT stop levothyroxine abruptly or outside specialist supervision — TSH can rebound "
                "rapidly in permanent hypothyroidism, causing harm. "
                "Biallelic DUOX2 LOF (complete H2O2 deficiency) is PERMANENT and requires lifelong treatment."
            ),
            "Pendred_Syndrome_EVA_Sign": (
                "Pendred syndrome (#274600) is caused by biallelic LOF variants in SLC26A4, encoding pendrin, "
                "a multifunctional anion exchanger expressed in the thyroid, inner ear, and kidney. "
                "The syndrome combines bilateral sensorineural deafness (typically prelingual, severe to "
                "profound), goiter (euthyroid in ~50%), and enlarged vestibular aqueduct (EVA) on CT/MRI of "
                "the temporal bone. "
                "EVA (defined as >1.5 mm diameter at the midpoint of the vestibular aqueduct) is the most "
                "consistent anatomical finding in SLC26A4 pathogenic variants and is considered pathognomonic "
                "— when EVA is identified in a child with sensorineural deafness, SLC26A4 sequencing is "
                "mandatory regardless of thyroid status. "
                "TRAUMA RULE: EVA predisposes to progressive fluctuating sensorineural hearing loss "
                "triggered by minor head trauma or pressure changes (barotrauma, Valsalva) — this is "
                "because the enlarged endolymphatic duct amplifies pressure waves into the cochlea. "
                "Families must be counselled to avoid contact sports; helmets should be worn for any activity "
                "with fall risk; this restriction is lifelong. "
                "Cochlear implantation outcomes in Pendred syndrome are generally good despite Mondini "
                "malformation coexisting in some patients — refer early to cochlear implant team."
            ),
            "Bamforth_Lazarus_Choanal_Emergency": (
                "Bamforth-Lazarus syndrome (#241850) is caused by biallelic inactivating variants in FOXE1 "
                "(forkhead box E1, formerly TTF-2) and represents one of the most severe syndromic forms "
                "of congenital hypothyroidism. FOXE1 is required for thyroid migration during embryogenesis "
                "and for normal development of the palate and choanae. "
                "ATHYREOSIS: most patients have complete absence of thyroid tissue on scan — "
                "not ectopic (as in PAX8) but truly absent; this means neonatal TSH is astronomically high "
                "and T4 is undetectable; levothyroxine must be started on day 1 of life. "
                "CHOANAL ATRESIA is the immediate life-threatening emergency: neonates are obligate nasal "
                "breathers; bilateral choanal atresia causes complete obstruction of the nasal airway → "
                "the neonate can only breathe when crying (crying opens the mouth); at rest → cyanosis; "
                "diagnosis is confirmed by inability to pass a 5 Fr catheter 3-4 cm through each nare. "
                "IMMEDIATE MANAGEMENT: insert McGovern nipple (large-hole oropharyngeal airway) to maintain "
                "oral airway while awaiting ENT; definitive surgical choanal repair within first days. "
                "ALL neonates with bilateral choanal atresia must have immediate thyroid function measured — "
                "Bamforth-Lazarus syndrome is the genetic diagnosis to exclude first. "
                "Spiky/bristle hair and cleft palate complete the syndrome."
            ),
            "Neonatal_Screening_Pitfalls": (
                "Neonatal blood-spot thyroid screening (heel-prick at 48-72 hours of life) detects most but "
                "not all cases of congenital hypothyroidism; several clinically important pitfalls exist. "
                "FALSE-NEGATIVE NBS (missed hypothyroidism): premature infants have immature HPT axis — "
                "TSH may not be elevated at NBS; PAX8 partial dysgenesis with borderline TSH may not "
                "exceed the cut-off; central/secondary hypothyroidism (low TSH + low T4) is missed by "
                "TSH-only screening programmes (approximately 1:20,000 births). "
                "FALSE-POSITIVE NBS (transient TSH elevation): sick/premature infants; iodine-exposed "
                "neonates (antiseptic use in NICU); maternal TRBAb (transient, resolves by 3-6 months); "
                "monoallelic DUOX2 (transient in ~60%); twin-to-twin transfusion. "
                "PERMANENT vs TRANSIENT: the distinction cannot be made at NBS diagnosis; all patients are "
                "treated with levothyroxine initially; at age 3, a supervised cessation trial identifies "
                "true transient cases (particularly monoallelic DUOX2 and maternal antibody cases). "
                "KEY RULE: if TSH does not normalize by age 3 years despite adequate levothyroxine AND "
                "cessation trial confirms persistent hypothyroidism → comprehensive thyroid gene panel "
                "including TSHR, PAX8, FOXE1, TPO, TG, SLC5A5, DUOX2/DUOXA2, SLC26A4 is indicated."
            ),
        }
    }


if __name__ == "__main__":
    import json

    print("=== OVERVIEW ===")
    ov = get_overview()
    print(f"Atlas: {ov['atlas_name']}")
    print(f"Subtitle: {ov['subtitle']}")
    print(f"N patients: {ov['n_patients']}")
    print(f"Genes ({ov['gene_count']}): {', '.join(ov['genes'])}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Registered: {ov['registered']}")
    print(f"Aggregate clinical: {json.dumps(ov['aggregate_clinical'], indent=2)}")

    print("\n=== BREAKDOWN (cohort stats per gene) ===")
    bd = get_breakdown()
    for g, info in bd["breakdown"].items():
        print(f"  {g}: {info['cohort_stats']}")

    print("\n=== DEFINITIONS (keys) ===")
    df = get_definitions()
    for k in df["definitions"]:
        print(f"  - {k}")

    print("\n=== SUMMARY ===")
    print(f"Total genes: {ov['gene_count']}")
    print(f"Total patients: {ov['n_patients']} (8 genes x 40 patients, seeds 1302-1309)")
    print(f"Levothyroxine prescribed (aggregate): {ov['aggregate_clinical']['levothyroxine_prescribed_pct']}%")
    print(f"Goiter across atlas: {ov['aggregate_clinical']['goiter_pct']}%")
    print(f"Perchlorate positive across atlas: {ov['aggregate_clinical']['perchlorate_positive_pct']}%")
    print(f"Median TSH at diagnosis: {ov['aggregate_clinical']['median_tsh_at_diagnosis']} mU/L")
    print(f"Median T4 at diagnosis: {ov['aggregate_clinical']['median_t4_at_diagnosis']} pmol/L")
    print(f"Choanal atresia (FOXE1): {ov['aggregate_clinical']['choanal_atresia_pct']}% of all atlas patients")
    print(f"Deafness (SLC26A4): {ov['aggregate_clinical']['deafness_pct']}% of all atlas patients")
