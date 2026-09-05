#!/usr/bin/env python3
"""Reproductive-Disorders-Atlas — Complete 8-Gene Hereditary Reproductive Disorders Atlas
ANOS1   (Anosmin-1; 680 aa; Xp22.31; OMIM gene 300836;
         Kallmann syndrome type 1 (KAL1) — X-linked; anosmia + hypogonadotropic hypogonadism;
         MIRROR MOVEMENTS (bimanual synkinesis) PATHOGNOMONIC — present in ~75% males;
         absent olfactory bulbs + sulci on MRI (pathognomonic imaging);
         ICVF (interhemispheric crossing via corpus callosum defect);
         testosterone + gonadotropins → fertility achievable in 70-80%) ·
FGFR1   (FGF receptor 1; 822 aa; 8p11.23; OMIM gene 136350;
         Kallmann syndrome type 2 + normosmic CHH (combined);
         AD with incomplete penetrance; craniofacial anomalies (cleft palate, dental agenesis);
         bimanual synkinesis LESS common than ANOS1;
         FGFR1 is a HH master regulator — variants span Kallmann to normosmic IHH to DP/fertile;
         reversal of HH (spontaneous recovery) occurs in ~10-20%) ·
GNRHR   (GnRH receptor; 328 aa; 4q13.2; OMIM gene 138850;
         normosmic IHH/CHH type 7 — NO anosmia, NO MRI abnormality;
         AR biallelic; PULSATILE GnRH therapy diagnostic + therapeutic (restores fertility);
         complete LOF → absent puberty; partial LOF → DP or fertile eunuch;
         GNRHR most common single AR gene in normosmic IHH) ·
KISS1R  (Kisspeptin receptor / GPR54; 398 aa; 19p13.3; OMIM gene 604161;
         normosmic IHH type 15 — AR biallelic; no olfactory deficit;
         kisspeptin-54 stimulation test diagnostic (LH surge absent in KISS1R LOF);
         upstream of GnRH → responds to pulsatile GnRH but NOT to kisspeptin challenge;
         KISS1R-IHH: puberty completely absent; testes prepubertal size) ·
FMR1    (Fragile X mental retardation protein; 632 aa; Xq27.3; OMIM gene 309550;
         FXPOI — fragile X-associated primary ovarian insufficiency;
         premutation carriers 55-200 CGG repeats (NOT full mutation >200);
         FSH elevated + irregular/absent menses before age 40 in ~20% of carriers;
         NO cognitive impairment in premutation women (distinguish from full mutation);
         FXTAS risk in carrier males (tremor/ataxia after 50);
         FMR1 FIRST in all POI workup — most common treatable cause) ·
FOXL2   (Forkhead box L2 TF; 376 aa; 3q22.3; OMIM gene 605597;
         BPES — blepharophimosis-ptosis-epicanthus inversus syndrome; AD;
         Type I BPES: ptosis + blepharophimosis + epicanthus inversus + POI;
         Type II BPES: same eye signs WITHOUT POI (different truncating variants);
         FOXL2 c.402C>G (p.Cys134Trp) HOT-SPOT SOMATIC MUTATION in adult granulosa cell tumors;
         surgical ptosis repair (age 3-4) — DO NOT DELAY, amblyopia risk;
         eyelid repair must precede gonadal hormone management) ·
BMP15   (Bone morphogenetic protein 15; 392 aa; Xp11.22; OMIM gene 300247;
         POI — premature ovarian insufficiency; X-linked dominant (XLD);
         oocyte-specific paracrine factor essential for folliculogenesis;
         hemizygous females (only one copy) → POI; XLD heterozygous carriers → reduced ovarian reserve;
         FMR1 premutation must be ruled out FIRST (more common);
         HRT mandatory from diagnosis to age 51 to prevent bone loss + cardiovascular risk) ·
PROKR2  (Prokineticin receptor 2; 384 aa; 20p12.3; OMIM gene 607123;
         Kallmann syndrome type 3 + normosmic IHH; AR (complete LOF) + AD digenic (partial);
         prokineticin-2 pathway essential for GnRH neuron migration + olfactory bulb morphogenesis;
         sleep disorder (hypersomnia) + obesity phenotype in some PROKR2 LOF patients;
         digenic inheritance with ANOS1/FGFR1 → compound HH phenotype)
320-patient aggregate cohort (8 x 40, seeds 1318-1325)
"""

import random

SEED_BASE = 1318

REPRODUCTIVE_GENES = [
    # ── ANOS1 — Kallmann Syndrome Type 1 (XLR) ──
    {
        "gene": "ANOS1",
        "protein": "Anosmin-1 (KAL1 protein)",
        "alias": (
            "ANOS1; OMIM gene 300836; Kallmann syndrome type 1 (KAL1) #308700; X-linked recessive; Xp22.31; "
            "680 aa; ~100 kDa secreted heparan-sulfate binding ECM glycoprotein; "
            "contains WAP/4DS motif + fibronectin type III domains; "
            "required for GnRH neuron migration from olfactory placode → hypothalamus along olfactory nerve; "
            "required for olfactory axon fasciculation → olfactory bulb formation; "
            "LOF: GnRH neurons stall in olfactory placode → absent olfactory bulb + absent/deficient hypothalamic GnRH → "
            "hypogonadotropic hypogonadism + anosmia (Kallmann syndrome); "
            "MIRROR MOVEMENTS (bimanual synkinesis): ipsilateral pyramidal tract fibers cross abnormally → "
            "contralateral movement during unilateral voluntary action; present ~75% affected males; "
            "IMAGING: absent olfactory bulbs + sulci on MRI olfactory sequences PATHOGNOMONIC; "
            "X-linked: female carriers mosaic phenotype (partial anosmia) — usually asymptomatic"
        ),
        "aa": "680 aa",
        "kDa": "~100 kDa",
        "locus": "Xp22.31",
        "omim_gene": 300836,
        "omim_disease": 308700,
        "inheritance": (
            "X-linked recessive; affected males hemizygous; "
            "carrier females rarely symptomatic but may have partial hyposmia; "
            "daughters of affected males are obligate carriers; "
            "sons of affected males are UNAFFECTED (Y-linked; not X); "
            "de novo variants in ~30% of sporadic KAL1"
        ),
        "gene_class": (
            "ANOS1 (formerly KAL1) encodes anosmin-1, a secreted ECM glycoprotein with a WAP domain "
            "and four fibronectin type III (FnIII) repeats. "
            "FUNCTION: anosmin-1 coats the olfactory nerve pathway and acts as a permissive cue for "
            "GnRH neuron migration from the olfactory placode to the hypothalamus. "
            "It also regulates olfactory axon fasciculation required to form the olfactory bulb. "
            "PATHOPHYSIOLOGY: ANOS1 LOF → GnRH neurons remain stranded in the cribriform plate/nose region → "
            "absent GnRH in hypothalamus → absent LH/FSH pulsatility → no puberty (micropenis, cryptorchidism, absent sex development). "
            "MRI finding: bilateral absence of olfactory bulbs and sulci = structural correlate of ANOS1 LOF. "
            "MIRROR MOVEMENTS: ANOS1 also guides development of corticospinal decussation; LOF → partial ipsilateral "
            "pyramidal fibers → bimanual synkinesia (pathognomonic for ANOS1 among the Kallmann genes). "
            "TREATMENT: Testosterone replacement for masculinization; pulsatile GnRH pump or FSH/LH injections for fertility. "
            "REVERSAL: Spontaneous return of gonadal axis ('reversal') documented in ~10% after cessation of TRT — "
            "likely partial GnRH neuronal recovery."
        ),
        "phenotype": (
            "Males: complete failure of puberty (micropenis ≤2.5 cm stretched, cryptorchidism, absent virilization); "
            "anosmia or severe hyposmia (smell test mandatory); "
            "mirror movements (75%); "
            "absent/underdeveloped secondary sexual characteristics; "
            "LH/FSH suppressed or undetectable; testosterone <100 ng/dL; "
            "testis volume prepubertal (1-2 mL); "
            "Females: primary amenorrhea, absent breast development, absent pubic/axillary hair, "
            "infantile uterus on ultrasound; anosmia; LH/FSH low; estradiol low. "
            "Associated: cleft palate (5-10%), renal agenesis (5%), hearing loss (rare), eye movement defects"
        ),
        "key_hallmarks": [
            "MIRROR MOVEMENTS (bimanual synkinesis) — present in ~75% ANOS1 males: PATHOGNOMONIC among Kallmann genes",
            "MRI: absent olfactory bulbs + olfactory sulci — order thin coronal T2 through olfactory region",
            "Anosmia + absent puberty in male = Kallmann syndrome → ANOS1 panel FIRST",
            "Testosterone replacement starts puberty; pulsatile GnRH pump → fertility 70-80%",
            "Daughters of affected males: ALL obligate carriers (X-linked; Y is passed to sons unaffected)",
        ],
        "treatment_alerts": [
            "Testosterone replacement (IM or transdermal): initiates virilization but does NOT restore fertility",
            "Pulsatile GnRH pump: stimulates endogenous LH/FSH pulsatility → testicular growth + spermatogenesis",
            "FSH + hCG injections: alternative fertility protocol if GnRH pump unavailable",
            "Cryptorchidism: orchidopexy before age 1 to preserve Sertoli cells for later fertility",
            "Smell screening (Sniffin' Sticks/UPSIT) in ALL first-degree males — anosmia = carrier testing next",
        ],
        "ddx": [
            "Constitutional delay of puberty (CDP): anosmia ABSENT; MRI normal; family history of late puberty",
            "FGFR1 Kallmann: AD, craniofacial anomalies, mirror movements LESS common, smell may be partially preserved",
            "GNRHR normosmic IHH: smell NORMAL; MRI olfactory normal; AR biallelic; responds to pulsatile GnRH",
            "Multiple pituitary hormone deficiency: other axes (TSH/ACTH/GH) also deficient — check full panel",
        ],
        "seed": SEED_BASE + 0,
        "n_patients": 40,
        "age_range": (14, 28),
        "female_pct": 15,  # X-linked: predominantly males
    },
    # ── FGFR1 — Kallmann Type 2 / Normosmic CHH ──
    {
        "gene": "FGFR1",
        "protein": "Fibroblast Growth Factor Receptor 1",
        "alias": (
            "FGFR1; OMIM gene 136350; Kallmann syndrome type 2 (KAL2) #147950 + normosmic IHH #146110; "
            "AD with incomplete penetrance (~40-70%); 8p11.23; 822 aa; ~92 kDa; "
            "type I transmembrane RTK; Ig-like domains I-III + TM + split TK domain; "
            "FGFR1 is the master regulator of GnRH neuron migration, olfactory axon fasciculation, "
            "and olfactory bulb morphogenesis; "
            "phenotypic spectrum: Kallmann (anosmia + HH) → normosmic IHH → delayed puberty → fertile; "
            "AD haploinsufficiency + dominant negative variants"
        ),
        "aa": "822 aa",
        "kDa": "~92 kDa",
        "locus": "8p11.23",
        "omim_gene": 136350,
        "omim_disease": 147950,
        "inheritance": (
            "AD haploinsufficiency with incomplete penetrance (~40-70%); "
            "same variant within a family can cause anosmia + HH in one member and only delayed puberty in another; "
            "digenic interactions with PROKR2, FGF8, NELF, CHD7 documented; "
            "de novo variants in ~40% of familial Kallmann"
        ),
        "gene_class": (
            "FGFR1 encodes FGF receptor 1, the principal RTK downstream of FGF8 in the olfactory epithelium. "
            "FUNCTION: FGF8/FGFR1 signaling is required for: "
            "(1) GnRH neuron specification and survival in the olfactory placode; "
            "(2) olfactory axon fasciculation through the cribriform plate; "
            "(3) olfactory bulb morphogenesis. "
            "SPECTRUM: FGFR1 spans from complete KAL2 (anosmia + absent puberty) to normosmic IHH (smell normal) "
            "to isolated delayed puberty to apparently normal fertile individuals — extreme intrafamilial variability. "
            "CRANIOFACIAL: FGFR1 also controls craniofacial development (FGF-Wnt signaling) → cleft palate (7-10%), "
            "dental agenesis (missing teeth, especially maxillary lateral incisors), high-arched palate. "
            "SYNKINESIS: mirror movements LESS common than ANOS1 (15-20%). "
            "REVERSAL: 10-20% experience spontaneous partial HH reversal — offer testosterone holiday annually."
        ),
        "phenotype": (
            "Absent or incomplete puberty (Kallmann spectrum); anosmia OR normosmia (spectrum); "
            "cleft lip/palate in 7-10%; missing teeth (especially maxillary lateral incisors); "
            "high-arched palate; bimanual synkinesis in 15-20%; "
            "LH/FSH low; testosterone/estradiol low; "
            "testis volume variable (may be slightly larger than ANOS1 if partial gonadotropin secretion); "
            "REVERSAL: ~15% develop spontaneous puberty after testosterone holiday"
        ),
        "key_hallmarks": [
            "FGFR1: phenotypic spectrum — Kallmann to normosmic IHH to delayed puberty (same variant, same family)",
            "Cleft palate + dental agenesis + HH → FGFR1 first (craniofacial + reproductive combination)",
            "Spontaneous reversal in ~15%: annual testosterone holiday trial after age 18",
            "Mirror movements LESS common than ANOS1 (15% vs 75%) — helpful DDx feature",
            "Digenic risk: FGFR1 + PROKR2 compound → more severe phenotype",
        ],
        "treatment_alerts": [
            "Testosterone replacement (puberty induction); FSH + hCG for fertility",
            "Annual testosterone holiday (3-6 months off TRT): test for spontaneous HH reversal",
            "Cleft palate: neonatal surgical repair team involvement from birth if affected",
            "Dental agenesis: orthodontic + prosthetic planning from childhood",
            "Cascade family testing: siblings + parents — penetrance ~50% in first-degree relatives",
        ],
        "ddx": [
            "ANOS1 Kallmann: XLR (males only severely affected), mirror movements 75%, absent olfactory bulbs MRI",
            "GNRHR normosmic IHH: smell NORMAL, no craniofacial anomalies, AR biallelic, responds to GnRH pulses",
            "CDP: delayed but spontaneous puberty, no anosmia, family history of delay, bone age delayed",
            "Panhypopituitarism: all axes (TSH/ACTH/GH) deficient; structural pituitary abnormality on MRI",
        ],
        "seed": SEED_BASE + 1,
        "n_patients": 40,
        "age_range": (14, 30),
        "female_pct": 40,  # AD — both sexes
    },
    # ── GNRHR — Normosmic IHH Type 7 (AR) ──
    {
        "gene": "GNRHR",
        "protein": "Gonadotropin-Releasing Hormone Receptor",
        "alias": (
            "GNRHR; OMIM gene 138850; normosmic IHH type 7 (nIHH7) #146110; AR biallelic; 4q13.2; "
            "328 aa; ~38 kDa; 7-TM GPCR (Gαq/11 coupling); "
            "LOF → GnRH cannot signal at pituitary → absent LH/FSH pulsatility despite normal GnRH neurons; "
            "most common single-gene AR cause of normosmic IHH; "
            "PULSATILE GnRH therapy is BOTH diagnostic (LH surge confirms pituitary integrity) AND therapeutic "
            "(restores gonadal axis and fertility in complete GNRHR LOF); "
            "partial LOF variants → fertile eunuch or delayed puberty"
        ),
        "aa": "328 aa",
        "kDa": "~38 kDa",
        "locus": "4q13.2",
        "omim_gene": 138850,
        "omim_disease": 146110,
        "inheritance": (
            "AR biallelic (complete LOF: absent puberty); "
            "compound heterozygous most common; "
            "partial LOF (hypomorphic) → fertile eunuch phenotype; "
            "GNRHR is the most common single AR gene in normosmic IHH (~16% of AR nIHH cases)"
        ),
        "gene_class": (
            "GNRHR encodes the GnRH receptor, a Gαq/11-coupled 7-TM GPCR on pituitary gonadotrophs. "
            "FUNCTION: GnRH (decapeptide from hypothalamic neurons) binds GNRHR → IP3/DAG cascade → "
            "LH + FSH synthesis and pulsatile release. "
            "PATHOPHYSIOLOGY: GNRHR LOF → normal GnRH neuron migration and hypothalamic GnRH synthesis "
            "but no pituitary LH/FSH response → normosmic IHH. "
            "DIAGNOSTIC KEY: pulsatile GnRH pump → LH surge = confirms pituitary responsiveness (GNRHR LOF "
            "is downstream of the receptor — pituitary gonadotrophs ARE intact). "
            "CONTRAST WITH ANOS1/FGFR1: those are upstream (GnRH neuron defects) → "
            "GnRH pump works, kisspeptin would NOT work. GNRHR → GnRH pump works but kisspeptin would not work. "
            "PARTIAL LOF: Arg262Gln / Gln106Arg compound → fertile eunuch (testosterone normal, FSH low, "
            "oligospermia only) — may present in adulthood for infertility. "
            "TREATMENT: pulsatile GnRH pump restores full gonadal axis + spermatogenesis."
        ),
        "phenotype": (
            "Normosmic IHH: absent puberty, normal smell; "
            "LH/FSH undetectable or very low; testosterone/estradiol prepubertal; "
            "normal olfactory bulbs on MRI; normal pituitary morphology; "
            "testis volume prepubertal; NO mirror movements; NO craniofacial anomalies; "
            "partial LOF: fertile eunuch (normal testosterone, low FSH, oligospermia, testicular volume slightly reduced)"
        ),
        "key_hallmarks": [
            "GNRHR — SMELL IS NORMAL: critical DDx from Kallmann (anosmia) — always test smell in HH workup",
            "Pulsatile GnRH pump: LH surge confirms pituitary responsiveness = diagnostic + therapeutic",
            "Most common AR single gene in normosmic IHH — include in first-tier panel",
            "Fertile eunuch phenotype: subtle hypomorphic GNRHR variants — may present as male infertility",
            "MRI: normal olfactory bulbs + normal pituitary (no structural cause)",
        ],
        "treatment_alerts": [
            "Pulsatile GnRH pump (90-min pulse intervals): restores LH/FSH pulsatility + spermatogenesis + fertility",
            "FSH + hCG injections: alternative to GnRH pump; slower but effective for fertility",
            "Testosterone alone: induces virilization but does NOT restore fertility (suppresses FSH)",
            "Males: if cryptorchidism — hCG trial first (orchidopexy if no descent after 6 months)",
            "Females: pulsatile GnRH → ovulation induction → conception achievable",
        ],
        "ddx": [
            "ANOS1 Kallmann: anosmia + XLR + mirror movements + absent olfactory bulbs MRI",
            "KISS1R normosmic IHH: smell normal, also AR, but kisspeptin stimulation test ABSENT (KISS1R LOF) vs GnRH responsive",
            "Functional HH (anorexia/illness/stress): reversible with weight restoration; no genetic variant",
            "CDP: delayed but spontaneous puberty; bone age delay; family history; normal GnRH stimulation test",
        ],
        "seed": SEED_BASE + 2,
        "n_patients": 40,
        "age_range": (14, 32),
        "female_pct": 45,
    },
    # ── KISS1R — Normosmic IHH Type 15 (AR) ──
    {
        "gene": "KISS1R",
        "protein": "Kisspeptin Receptor (GPR54)",
        "alias": (
            "KISS1R (GPR54); OMIM gene 604161; normosmic IHH type 15 #614880; AR biallelic; 19p13.3; "
            "398 aa; ~41 kDa; Gαq/11 GPCR; "
            "upstream of GnRH neurons in the hypothalamic KNDy network; "
            "kisspeptin-54 (KP54) binds KISS1R → GnRH neuron activation → pulsatile GnRH release; "
            "KISS1R LOF: kisspeptin cannot activate GnRH neurons → absent GnRH pulsatility → normosmic IHH; "
            "DIAGNOSTIC: kisspeptin-54 stimulation test → NO LH response (KISS1R LOF); pulsatile GnRH → LH SURGE (pituitary intact)"
        ),
        "aa": "398 aa",
        "kDa": "~41 kDa",
        "locus": "19p13.3",
        "omim_gene": 604161,
        "omim_disease": 614880,
        "inheritance": (
            "AR biallelic LOF; both sexes equally affected; "
            "heterozygous carriers normal — no haploinsufficiency; "
            "KISS1R LOF accounts for ~3-5% of normosmic IHH; "
            "de Roux 2003 Science: first KISS1R LOF human IHH"
        ),
        "gene_class": (
            "KISS1R (GPR54) is a Gαq/11-coupled GPCR on GnRH neurons in the arcuate (KNDy) and anteroventral periventricular nuclei. "
            "FUNCTION: kisspeptin-54 (N-terminally extended kisspeptin) from ARC neurons → binds KISS1R → "
            "IP3/DAG cascade → membrane depolarization of GnRH neurons → pulsatile GnRH burst release. "
            "KISSPEPTIN IS THE MASTER REGULATOR of GnRH pulsatility — integrates negative feedback (E2/T) + "
            "positive feedback (E2 surge for LH surge) + metabolic signals (leptin/NPY). "
            "PATHOPHYSIOLOGY: KISS1R biallelic LOF → GnRH neurons cannot be activated despite normal development and anatomy → "
            "absent GnRH pulsatility → absent LH/FSH → complete normosmic IHH. "
            "DIAGNOSTIC PROTOCOL: "
            "(1) kisspeptin-54 stimulation test: IV KP54 → absent LH rise in KISS1R LOF (pathognomonic); "
            "(2) pulsatile GnRH pump → LH surge PRESENT (confirms pituitary gonadotroph integrity). "
            "CONTRAST WITH GNRHR: GNRHR LOF → pulsatile GnRH works, KP54 also won't work (both upstream pathways dead); "
            "KISS1R LOF → GnRH pump works, KP54 doesn't (GnRH neurons intact but cannot be activated by kisspeptin)."
        ),
        "phenotype": (
            "Normosmic IHH: absent puberty in both sexes; "
            "smell NORMAL; olfactory MRI NORMAL; pituitary MRI NORMAL; "
            "LH/FSH undetectable; testosterone/estradiol prepubertal; "
            "NO mirror movements; NO craniofacial anomalies; "
            "gonadotropin response absent to kisspeptin-54 test (pathognomonic); "
            "normal LH response to pulsatile GnRH (pituitary responsive); "
            "testis/ovarian volume prepubertal at diagnosis"
        ),
        "key_hallmarks": [
            "KISS1R LOF: kisspeptin-54 test → NO LH rise (pathognomonic); pulsatile GnRH → LH rise present",
            "Smell is NORMAL: distinguishes from Kallmann (ANOS1/FGFR1/PROKR2 anosmia forms)",
            "Master upstream regulator of GnRH — KISS1R sits above GnRH neurons in the hierarchy",
            "Pulsatile GnRH pump restores fertility (GnRH neuron intact, receptor downstream OK)",
            "Cascade testing: AR — siblings at 25% risk; parents obligate carriers (often asymptomatic)",
        ],
        "treatment_alerts": [
            "Pulsatile GnRH pump: restores GnRH pulsatility + LH/FSH + fertility",
            "Testosterone/Estrogen replacement: initiates puberty but NO fertility",
            "Kisspeptin therapy investigational: NOT standard of care (patient has LOF receptor — won't respond)",
            "Annual HH reversal trial: testosterone holiday at age 18 — 5-10% show partial spontaneous axis recovery",
            "Bone density monitoring: DEXA at diagnosis and every 2 years — HH causes osteoporosis",
        ],
        "ddx": [
            "GNRHR normosmic IHH: clinically identical to KISS1R IHH — only kisspeptin test distinguishes (KP54 works if KISS1R intact)",
            "ANOS1 Kallmann: anosmia; XLR; mirror movements; absent olfactory bulbs MRI — easily distinguished",
            "Functional HH: reversible; normal kisspeptin test (KISS1R intact); associated with weight/exercise/stress",
            "Pituitary adenoma: structural cause; MRI abnormal; often other axes deficient",
        ],
        "seed": SEED_BASE + 3,
        "n_patients": 40,
        "age_range": (14, 30),
        "female_pct": 48,
    },
    # ── FMR1 — FXPOI (Fragile X Premutation Carrier) ──
    {
        "gene": "FMR1",
        "protein": "Fragile X Mental Retardation Protein (FMRP)",
        "alias": (
            "FMR1; OMIM gene 309550; FXPOI — fragile X-associated primary ovarian insufficiency; "
            "Xq27.3; 632 aa; ~71 kDa RNA-binding protein; CGG repeat expansion in 5'UTR; "
            "PREMUTATION 55-200 CGG (NOT full mutation >200 which causes FXS intellectual disability); "
            "FXPOI: premature ovarian insufficiency before age 40 in ~20% of female premutation carriers; "
            "FSH elevated + irregular/absent menses = biochemical POI; "
            "NO cognitive impairment in premutation females (unlike full mutation >200 CGG → FXS); "
            "FMR1 MUST BE TESTED FIRST in all POI workup — most common identifiable genetic cause"
        ),
        "aa": "632 aa",
        "kDa": "~71 kDa",
        "locus": "Xq27.3",
        "omim_gene": 309550,
        "omim_disease": 311360,
        "inheritance": (
            "X-linked; FMR1 premutation (55-200 CGG) is FXPOI risk; "
            "premutation repeat EXPANDS to full mutation (>200 CGG) in female carriers' children → "
            "risk of having affected son with FXS (intellectual disability, autism, macroorchidism); "
            "FXTAS: fragile X-associated tremor/ataxia syndrome in PREMUTATION MALES >50 years; "
            "prevalence: 1 in 150-300 females carry premutation"
        ),
        "gene_class": (
            "FMR1 encodes FMRP (fragile X mental retardation protein), an mRNA transport + translational suppressor. "
            "FULL MUTATION (>200 CGG): CGG hypermethylation → FMR1 silencing → absent FMRP → FXS (ASD + ID). "
            "PREMUTATION (55-200 CGG): FMR1 transcribed at INCREASED rate → toxic FMR1 mRNA accumulation → "
            "RNA-gain-of-function toxicity in ovarian granulosa cells → premature follicle depletion → FXPOI. "
            "MECHANISM: expanded CGG mRNA sequesters DROSHA/DGCR8 → miRNA dysregulation → granulosa cell apoptosis. "
            "CLINICAL: FXPOI presents as: secondary amenorrhea before age 40 + elevated FSH (>25 IU/L) + "
            "low AMH + antral follicle count reduction. "
            "IMPORTANT: ~30% of FXPOI women have RESIDUAL OVARIAN FUNCTION — spontaneous ovulation/conception possible. "
            "HRT: mandatory until age 51 (natural menopause age) → bone + cardiovascular + cognitive protection. "
            "GENETIC COUNSELING: premutation carrier woman risks: (1) FXPOI herself; (2) expansion in offspring → FXS sons."
        ),
        "phenotype": (
            "FXPOI: irregular or absent menses before age 40; "
            "elevated FSH (>25 IU/L on two measurements 4 weeks apart); "
            "low AMH (<1 ng/mL); reduced antral follicle count on ultrasound; "
            "hot flashes, night sweats, vaginal dryness (estrogen deficiency); "
            "normal or mild cognitive function (NOT intellectual disability — that is full mutation); "
            "family history: brothers/uncles with intellectual disability/autism → full mutation; "
            "FXTAS in carrier males: intention tremor, cerebellar ataxia after age 50; "
            "~20% of premutation carriers develop clinical FXPOI; 100% have reduced ovarian reserve"
        ),
        "key_hallmarks": [
            "FMR1 FIRST in all POI workup — most common identifiable genetic cause (~3% of sporadic, ~12% of familial POI)",
            "PREMUTATION (55-200 CGG) causes FXPOI — NOT the full mutation (>200 CGG which causes FXS/ID)",
            "~30% of FXPOI women retain residual ovarian function — spontaneous conception possible (advise contraception if not planning pregnancy)",
            "HRT mandatory from FXPOI diagnosis until age 51: bone, cardiovascular, cognitive protection",
            "Genetic counseling MANDATORY: premutation expands to full mutation in offspring → FXS sons risk",
        ],
        "treatment_alerts": [
            "HRT (estradiol + progesterone): mandatory from diagnosis until age ~51 — do NOT withhold (bone + cardiac risk)",
            "Spontaneous conception possible in 30%: counsel on contraception if pregnancy not desired",
            "Egg freezing / oocyte cryopreservation: offer BEFORE ovarian reserve depletes completely",
            "Genetic counseling: male offspring risk FXS (full mutation expansion); recommend FMR1 testing in family members",
            "DEXA scan: at diagnosis; repeat every 2 years — osteoporosis prevention priority",
        ],
        "ddx": [
            "FOXL2 BPES type I: blepharophimosis + ptosis + POI — clinical eye exam distinguishes; FOXL2 sequencing",
            "BMP15 POI: XLD; no eye signs; FMR1 should be tested first (more common than BMP15)",
            "Turner syndrome (45,X): karyotype distinguishes; streak gonads; short stature; cardiac anomalies",
            "Autoimmune POI: anti-TPO + ANA/anti-ovarian antibodies; FMR1 negative; associated with other autoimmune diseases",
        ],
        "seed": SEED_BASE + 4,
        "n_patients": 40,
        "age_range": (22, 42),
        "female_pct": 100,  # FXPOI — females only
    },
    # ── FOXL2 — BPES Type I (AD; Ptosis + POI) ──
    {
        "gene": "FOXL2",
        "protein": "Forkhead Box Protein L2",
        "alias": (
            "FOXL2; OMIM gene 605597; BPES — blepharophimosis-ptosis-epicanthus inversus syndrome #110100; "
            "AD haploinsufficiency + dominant negative; 3q22.3; 376 aa; ~44 kDa; "
            "forkhead TF + poly-alanine expansion domain; "
            "Type I BPES: blepharophimosis + ptosis + epicanthus inversus + POI (dominant negative/null variants); "
            "Type II BPES: same eye signs WITHOUT POI (haploinsufficiency/shorter AA changes); "
            "FOXL2 c.402C>G (p.Cys134Trp) SOMATIC HOT-SPOT in adult granulosa cell tumors (GCT); "
            "critical role in ovarian granulosa cell differentiation + folliculogenesis maintenance"
        ),
        "aa": "376 aa",
        "kDa": "~44 kDa",
        "locus": "3q22.3",
        "omim_gene": 605597,
        "omim_disease": 110100,
        "inheritance": (
            "AD haploinsufficiency (type I + II); "
            "dominant negative variants (larger in-frame deletions/poly-Ala expansions) → type I BPES (with POI); "
            "point mutations/smaller changes → type II BPES (without POI); "
            "penetrance ~100% for eye signs; POI penetrance variable (type I variants ~80%); "
            "de novo variants in ~30% of sporadic BPES"
        ),
        "gene_class": (
            "FOXL2 encodes a forkhead domain transcription factor expressed in ovarian granulosa cells and eyelid mesenchyme. "
            "EYE FUNCTION: FOXL2 is required for eyelid formation and levator palpebrae muscle development. "
            "LOF → blepharophimosis (horizontal eyelid narrowing), ptosis (levator failure), epicanthus inversus (medial fold). "
            "OVARIAN FUNCTION: FOXL2 maintains granulosa cell identity and prevents transdifferentiation to Sertoli cells. "
            "LOF → granulosa cells trans-differentiate toward Sertoli fate → follicle atresia → POI. "
            "TYPE DISTINCTION: "
            "Dominant negative large expansions → protein misfolds → sequesters wild-type → more severe (type I = POI). "
            "Haploinsufficiency point mutations → loss of one copy → less severe → type II (eyes only, ovaries spared). "
            "SOMATIC HOT-SPOT: FOXL2 c.402C>G (p.Cys134Trp) detected in >90% of adult ovarian granulosa cell tumors — "
            "somatic GOF (≠ germline BPES). Women with BPES need GCT surveillance. "
            "SURGICAL PRIORITY: ptosis repair at age 3-4 (BEFORE school age) to prevent stimulus deprivation amblyopia."
        ),
        "phenotype": (
            "Type I BPES: "
            "blepharophimosis (horizontal palpebral fissure <22 mm at birth); "
            "ptosis (levator palpebrae weakness); "
            "epicanthus inversus (skin fold from lower lid to medial canthus — INVERSE of normal epicanthus); "
            "telecanthus (increased medial canthal distance); "
            "PLUS: premature ovarian insufficiency (FSH elevated, amenorrhea/oligomenorrhea before age 40); "
            "Type II BPES: identical eye signs WITHOUT POI; "
            "Both types: fertility reduced in type I; type II generally fertile"
        ),
        "key_hallmarks": [
            "BPES type I: PTOSIS + blepharophimosis + epicanthus inversus + POI — FOXL2 germline panel",
            "Ptosis repair at age 3-4: DO NOT DELAY — amblyopia risk from stimulus deprivation (lid covers pupil)",
            "Type I vs II: POI distinguishes (type I has POI; type II does not) — guides genetic counseling",
            "FOXL2 p.Cys134Trp: SOMATIC HOT-SPOT in adult granulosa cell tumors (NOT germline BPES variant)",
            "Eyelid surgery PRECEDES hormone management timeline — surgical team first",
        ],
        "treatment_alerts": [
            "Ptosis repair: frontalis suspension (3-4 years) — ophthalmic surgery priority to prevent amblyopia",
            "Lateral canthoplasty: may be needed for blepharophimosis correction (staged with ptosis repair)",
            "HRT: mandatory for type I BPES + POI — bone, cardiovascular protection until age 51",
            "Fertility: egg freezing before ovarian reserve depletes; donor oocytes if reserve exhausted",
            "GCT surveillance: BPES patients may have slightly higher GCT risk — annual pelvic ultrasound",
        ],
        "ddx": [
            "FMR1 FXPOI: POI without blepharophimosis/ptosis — eye examination distinguishes immediately",
            "BMP15 POI: XLD; no eye signs; overlapping POI phenotype",
            "Turner syndrome: karyotype distinguishes; streak gonads; no eyelid anomaly",
            "Congenital ptosis (non-syndromic PTOS genes): ptosis alone without blepharophimosis/epicanthus inversus",
        ],
        "seed": SEED_BASE + 5,
        "n_patients": 40,
        "age_range": (0, 40),
        "female_pct": 65,  # AD — both sexes but POI component female-predominant presentation
    },
    # ── BMP15 — POI / XLD ──
    {
        "gene": "BMP15",
        "protein": "Bone Morphogenetic Protein 15",
        "alias": (
            "BMP15; OMIM gene 300247; POI (premature ovarian insufficiency) / POF4 #300511; "
            "X-linked dominant (XLD); Xp11.22; 392 aa; ~35 kDa mature form after propeptide cleavage; "
            "TGF-beta superfamily member; oocyte-specific secreted growth factor; "
            "BMP15 acts in PARACRINE manner on surrounding granulosa cells → promotes folliculogenesis; "
            "hemizygous females (single copy) → haploinsufficiency → reduced follicular survival → POI; "
            "BMP15 works in HETERODIMER with GDF9 (both oocyte factors) → amplified signaling; "
            "FMR1 testing MUST precede BMP15 in POI workup (FMR1 more common)"
        ),
        "aa": "392 aa",
        "kDa": "~35 kDa (mature)",
        "locus": "Xp11.22",
        "omim_gene": 300247,
        "omim_disease": 300511,
        "inheritance": (
            "X-linked dominant (XLD); "
            "hemizygous females (monosomy Xp11.22 or pathogenic variant on single X): more severe POI; "
            "heterozygous females: variable POI penetrance — ovarian reserve reduction; "
            "homozygous LOF: complete infertility; "
            "males hemizygous for LOF BMP15 variants: generally phenotypically normal (testes/spermatogenesis independent)"
        ),
        "gene_class": (
            "BMP15 (bone morphogenetic protein 15) is a TGF-beta superfamily growth factor expressed specifically "
            "by the oocyte throughout all stages of folliculogenesis. "
            "FUNCTION: BMP15 is secreted from the oocyte → binds BMPR1B/BMPR2 on surrounding granulosa cells → "
            "SMAD1/5/8 activation → promotes granulosa cell proliferation, FSH-receptor expression, anti-apoptotic signaling. "
            "BMP15 + GDF9 (another oocyte factor) form HETERODIMERS with 10-fold amplified granulosa signaling. "
            "PATHOPHYSIOLOGY: BMP15 LOF → granulosa cells fail to respond to oocyte signals → "
            "follicle atresia → reduced ovarian reserve → POI. "
            "X-LINKED DOMINANCE: BMP15 lies on X; females have two copies. Single-copy insufficiency → "
            "haploinsufficiency sufficient for ovarian phenotype (unlike autosomal genes requiring biallelic). "
            "CLINICAL: presents as secondary amenorrhea + elevated FSH + infertility in 20s-30s. "
            "CARRIER MALES: X-linked → males hemizygous LOF BMP15 — reported to have INCREASED twin rate "
            "(possible hyperovulation via BMP15-GDF9 heterodimer dosage effect in fertile female relatives). "
            "MANAGEMENT: identical to other POI — HRT + fertility counseling."
        ),
        "phenotype": (
            "Secondary amenorrhea or severe oligomenorrhea before age 40; "
            "elevated FSH (>25 IU/L); low AMH (<0.5 ng/mL); "
            "reduced antral follicle count; small ovaries on ultrasound; "
            "hot flashes, night sweats (estrogen deficiency); "
            "NO eye signs, NO cognitive abnormalities; "
            "variable severity: some women have near-complete ovarian failure by age 25, others oligomenorrhea into 30s; "
            "male family members: phenotypically normal"
        ),
        "key_hallmarks": [
            "BMP15 POI: XLD — single pathogenic allele sufficient for ovarian insufficiency in females",
            "FMR1 premutation must be excluded FIRST (more common than BMP15 in POI)",
            "BMP15 + GDF9 heterodimer: oocyte paracrine signaling — test GDF9 if BMP15 negative",
            "HRT mandatory from diagnosis until age 51: bone loss begins within 2 years of untreated POI",
            "Residual follicular activity: ~20% of BMP15 POI → fertility counseling; donor oocyte if exhausted",
        ],
        "treatment_alerts": [
            "HRT (estradiol + progesterone cyclically): mandatory from diagnosis — do NOT withhold",
            "DEXA scan at diagnosis + every 2 years: osteoporosis prevention",
            "Oocyte cryopreservation: offer early if any residual follicular activity remains",
            "Donor oocyte IVF: most effective fertility option when ovarian reserve exhausted",
            "Genetic counseling: daughters of BMP15 carriers have 50% risk; male carriers usually fertile",
        ],
        "ddx": [
            "FMR1 FXPOI: same POI phenotype — FMR1 testing first (more prevalent, manageable genetic risk in offspring)",
            "FOXL2 BPES type I: POI + blepharophimosis + ptosis — eye examination immediately distinguishes",
            "Turner syndrome 45,X or mosaic: karyotype; short stature; cardiac; no X-linked dominant pattern",
            "Autoimmune POI: positive anti-ovarian antibodies; associated with thyroiditis, adrenal insufficiency",
        ],
        "seed": SEED_BASE + 6,
        "n_patients": 40,
        "age_range": (18, 40),
        "female_pct": 100,  # XLD — primarily female phenotype (POI)
    },
    # ── PROKR2 — Kallmann Type 3 / Normosmic IHH ──
    {
        "gene": "PROKR2",
        "protein": "Prokineticin Receptor 2",
        "alias": (
            "PROKR2; OMIM gene 607123; Kallmann syndrome type 3 (KAL3) #244200 + normosmic IHH; "
            "AR (complete LOF) + AD digenic/monoallelic (partial); 20p12.3; 384 aa; ~41 kDa; "
            "7-TM GPCR; prokineticin-2 (PROK2) ligand; "
            "PROKR2/PROK2 pathway essential for: GnRH neuron migration + olfactory bulb morphogenesis; "
            "PROKR2 variants found in ~9% of Kallmann patients; "
            "sleep phenotype (hypersomnia) + obesity in some PROKR2 LOF; "
            "DIGENIC: PROKR2 + ANOS1 or FGFR1 compound heterozygosity → more severe HH"
        ),
        "aa": "384 aa",
        "kDa": "~41 kDa",
        "locus": "20p12.3",
        "omim_gene": 607123,
        "omim_disease": 244200,
        "inheritance": (
            "AR biallelic (complete LOF) → Kallmann or normosmic IHH; "
            "AD monoallelic (partial LOF or dominant negative) → milder phenotype or digenic contribution; "
            "digenic: PROKR2 + ANOS1 / FGFR1 / CHD7 → compound phenotype more severe than either alone; "
            "prevalence: 9% of Kallmann syndrome; more common than ANOS1 in some European cohorts"
        ),
        "gene_class": (
            "PROKR2 encodes prokineticin receptor 2, a Gαq/11 GPCR activated by PROK2 (prokineticin-2). "
            "PROKR2/PROK2 SIGNALING IN HH: "
            "(1) GnRH neuron migration: PROK2/PROKR2 pathway guides GnRH neuron translocation from olfactory placode "
            "into the hypothalamus (similar role to ANOS1 but different molecular mechanism). "
            "(2) Olfactory bulb morphogenesis: PROK2 regulates olfactory bulb neurogenesis from the subventricular zone. "
            "(3) Circadian regulation: PROKR2 expressed in SCN → LOF → sleep dysregulation (hypersomnia/sleep phase disorder). "
            "PHENOTYPIC VARIABILITY: biallelic LOF → Kallmann or normosmic IHH; monoallelic → delayed puberty or "
            "contribution in digenic setting. "
            "OBESITY: PROKR2 expressed in hypothalamic feeding centers → LOF → hyperphagia + obesity in some patients. "
            "DIGENIC MECHANISM: single PROKR2 variant (monoallelic partial LOF) may not cause HH alone but when "
            "combined with a second variant in ANOS1/FGFR1 → synergistic HH — a model of oligogenic inheritance. "
            "TREATMENT: identical to other Kallmann/IHH — testosterone/estrogen, pulsatile GnRH, FSH+hCG for fertility."
        ),
        "phenotype": (
            "Kallmann form: anosmia + HH (absent puberty, low LH/FSH, low sex steroids); "
            "normosmic IHH form: HH without smell deficit; "
            "sleep dysregulation (hypersomnia, altered sleep-wake cycle) in subset; "
            "obesity (BMI >30) in subset; "
            "DIGENIC patients: more severe or treatment-resistant HH; "
            "LH/FSH absent/very low; testosterone/estradiol prepubertal; "
            "testis volume prepubertal; olfactory MRI variable (may show reduced bulb volume)"
        ),
        "key_hallmarks": [
            "PROKR2 Kallmann: anosmia + HH + hypersomnia + obesity = PROKR2/PROK2 until proven otherwise",
            "Digenic inheritance: PROKR2 + ANOS1 or FGFR1 compound → more severe HH (oligogenic model)",
            "Sleep dysregulation (hypersomnia/circadian disorder) is a clue — other Kallmann genes do NOT cause this",
            "Treatment identical to ANOS1/FGFR1 Kallmann — pulsatile GnRH + FSH/hCG for fertility",
            "Include PROK2 in panel with PROKR2 — ligand + receptor both cause same phenotype",
        ],
        "treatment_alerts": [
            "Testosterone replacement (puberty induction and maintenance)",
            "Pulsatile GnRH pump or FSH + hCG for fertility — response similar to other Kallmann genes",
            "Sleep evaluation: if hypersomnia prominent → polysomnography; melatonin chronotherapy trial",
            "Obesity management: structured weight program + metabolic monitoring",
            "Family testing: AR probands → test siblings; AD/digenic → test first-degree relatives",
        ],
        "ddx": [
            "ANOS1 Kallmann: XLR; mirror movements 75%; absent olfactory bulbs MRI; no obesity phenotype",
            "FGFR1 Kallmann: AD; craniofacial anomalies; wider phenotypic spectrum; no sleep/obesity clue",
            "GNRHR normosmic IHH: smell normal; AR; no sleep phenotype; pulsatile GnRH diagnostic",
            "Hypothalamic obesity (non-genetic): acquired lesion on MRI; no anosmia; other hypothalamic deficits",
        ],
        "seed": SEED_BASE + 7,
        "n_patients": 40,
        "age_range": (14, 35),
        "female_pct": 38,
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

        if gene == "ANOS1":
            mirror_movements = rng.random() < 0.75
            cryptorchidism = sex == "M" and rng.random() < 0.65
            anosmia = True
            testosterone_started = rng.random() < 0.88
            gnrh_pump_fertility = sex == "M" and rng.random() < 0.70
            p = {
                "patient_id": f"ANOS1-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "anosmia": anosmia,
                "mirror_movements": mirror_movements,
                "cryptorchidism": cryptorchidism,
                "absent_olfactory_bulbs_mri": True,
                "lh_undetectable": True,
                "fsh_undetectable": True,
                "testosterone_ng_dL": round(rng.uniform(10, 80), 1) if sex == "M" else None,
                "testosterone_started": testosterone_started,
                "gnrh_pump_fertility_attempt": gnrh_pump_fertility,
                "successful_fertility": gnrh_pump_fertility and rng.random() < 0.72,
            }
        elif gene == "FGFR1":
            anosmia = rng.random() < 0.60  # spectrum: some normosmic
            cleft_palate = rng.random() < 0.08
            dental_agenesis = rng.random() < 0.15
            reversal = rng.random() < 0.15
            mirror_movements = rng.random() < 0.18
            p = {
                "patient_id": f"FGFR1-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "anosmia": anosmia,
                "normosmic_spectrum": not anosmia,
                "cleft_palate": cleft_palate,
                "dental_agenesis": dental_agenesis,
                "mirror_movements": mirror_movements,
                "spontaneous_reversal": reversal,
                "lh_undetectable": not reversal,
                "testosterone_or_estradiol_low": not reversal,
                "puberty_incomplete": not reversal,
            }
        elif gene == "GNRHR":
            partial_lof = rng.random() < 0.18  # fertile eunuch variant
            pulsatile_gnrh_given = rng.random() < 0.75
            p = {
                "patient_id": f"GNRHR-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "anosmia": False,
                "olfactory_mri_normal": True,
                "partial_lof_fertile_eunuch": partial_lof,
                "lh_undetectable": not partial_lof,
                "fsh_undetectable": not partial_lof,
                "pulsatile_gnrh_therapy": pulsatile_gnrh_given,
                "lh_surge_on_gnrh_pump": pulsatile_gnrh_given,
                "fertility_achieved": pulsatile_gnrh_given and rng.random() < 0.68,
                "testosterone_ng_dL": round(rng.uniform(15, 100), 1) if sex == "M" and partial_lof else None,
            }
        elif gene == "KISS1R":
            kp54_test_done = rng.random() < 0.62
            gnrh_pump_done = rng.random() < 0.70
            p = {
                "patient_id": f"KISS1R-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "anosmia": False,
                "olfactory_mri_normal": True,
                "lh_undetectable": True,
                "fsh_undetectable": True,
                "kp54_stimulation_test_done": kp54_test_done,
                "lh_response_to_kp54": False,  # pathognomonic
                "lh_response_to_gnrh_pump": gnrh_pump_done,
                "pulsatile_gnrh_therapy": gnrh_pump_done,
                "fertility_achieved": gnrh_pump_done and rng.random() < 0.60,
            }
        elif gene == "FMR1":
            cgg_repeats = rng.randint(57, 195)
            fsh_elevated = True
            amh_low = True
            hrt_started = rng.random() < 0.82
            oocyte_frozen = rng.random() < 0.30
            spontaneous_ovulation = rng.random() < 0.30
            p = {
                "patient_id": f"FMR1-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "cgg_repeat_count": cgg_repeats,
                "premutation_confirmed": True,
                "fsh_elevated": fsh_elevated,
                "fsh_iu_L": round(rng.uniform(28, 120), 1),
                "amh_low": amh_low,
                "amh_ng_mL": round(rng.uniform(0.01, 0.6), 2),
                "irregular_menses": True,
                "cognitive_normal": True,  # premutation — no FXS cognitive impairment
                "hrt_started": hrt_started,
                "oocyte_cryopreserved": oocyte_frozen,
                "spontaneous_conception_possible": spontaneous_ovulation,
                "fxtas_risk_male_relatives": True,
            }
        elif gene == "FOXL2":
            has_poi = rng.random() < 0.70  # type I predominant (with POI)
            bpes_type = "I" if has_poi else "II"
            ptosis_repair_done = rng.random() < 0.75
            hrt_started = has_poi and rng.random() < 0.78
            p = {
                "patient_id": f"FOXL2-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "bpes_type": bpes_type,
                "blepharophimosis": True,
                "ptosis": True,
                "epicanthus_inversus": True,
                "poi_present": has_poi,
                "fsh_elevated": has_poi,
                "fsh_iu_L": round(rng.uniform(30, 100), 1) if has_poi else None,
                "ptosis_surgery_done": ptosis_repair_done,
                "age_at_ptosis_repair": rng.randint(3, 7) if ptosis_repair_done else None,
                "hrt_started": hrt_started,
                "amblyopia_risk_managed": ptosis_repair_done,
            }
        elif gene == "BMP15":
            amh_very_low = True
            hrt_started = rng.random() < 0.85
            donor_oocyte = rng.random() < 0.35
            residual_follicular = rng.random() < 0.22
            p = {
                "patient_id": f"BMP15-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "amh_very_low": amh_very_low,
                "amh_ng_mL": round(rng.uniform(0.01, 0.4), 2),
                "fsh_elevated": True,
                "fsh_iu_L": round(rng.uniform(25, 110), 1),
                "secondary_amenorrhea": True,
                "no_eye_signs": True,
                "no_cognitive_anomaly": True,
                "hrt_started": hrt_started,
                "residual_follicular_activity": residual_follicular,
                "donor_oocyte_ivf": donor_oocyte,
                "dexa_done": rng.random() < 0.65,
            }
        elif gene == "PROKR2":
            anosmia = rng.random() < 0.50  # spectrum: Kallmann + normosmic IHH
            hypersomnia = rng.random() < 0.35
            obese = rng.random() < 0.30
            digenic = rng.random() < 0.25  # compound with ANOS1/FGFR1
            p = {
                "patient_id": f"PROKR2-{i+1:03d}",
                "gene": gene,
                "age_at_dx": age,
                "sex": sex,
                "anosmia": anosmia,
                "normosmic_spectrum": not anosmia,
                "hypersomnia": hypersomnia,
                "obesity_bmi_over_30": obese,
                "digenic_second_variant": digenic,
                "lh_undetectable": True,
                "fsh_undetectable": True,
                "testosterone_or_estradiol_low": True,
                "gnrh_pump_done": rng.random() < 0.60,
                "polysomnography_done": hypersomnia and rng.random() < 0.55,
            }
        else:
            p = {"patient_id": f"{gene}-{i+1:03d}", "gene": gene, "age_at_dx": age, "sex": sex}

        patients.append(p)
    return patients


_ALL_COHORTS = {g["gene"]: _make_cohort(g) for g in REPRODUCTIVE_GENES}


# ─── API response builders ─────────────────────────────────────────────────

def _pct(cohort, key):
    n = len(cohort)
    if n == 0:
        return 0
    return round(100 * sum(1 for p in cohort if p.get(key)) / n, 1)


def get_overview():
    n = sum(len(v) for v in _ALL_COHORTS.values())

    # Aggregate HH genes vs POI genes
    hh_genes = ["ANOS1", "FGFR1", "GNRHR", "KISS1R", "PROKR2"]
    poi_genes = ["FMR1", "FOXL2", "BMP15"]

    kallmann_genes = ["ANOS1", "FGFR1", "PROKR2"]  # anosmia forms
    anosmia_rate = round(
        sum(
            sum(1 for p in _ALL_COHORTS[g] if p.get("anosmia")) for g in kallmann_genes
        ) / n * 100, 1
    )
    female_n = sum(sum(1 for p in c if p.get("sex") == "F") for c in _ALL_COHORTS.values())
    poi_n = sum(len(_ALL_COHORTS[g]) for g in poi_genes)

    return {
        "atlas_name": "Reproductive-Disorders-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Reproductive Disorders Atlas",
        "n_patients": n,
        "gene_count": len(REPRODUCTIVE_GENES),
        "genes": [g["gene"] for g in REPRODUCTIVE_GENES],
        "seeds": "1318–1325",
        "registered": "2026-09-05",
        "atlas_version": "1.0",
        "gene_summary": [
            {
                "gene": "ANOS1",
                "protein": "Anosmin-1",
                "aa": "680 aa",
                "locus": "Xp22.31",
                "inheritance": "XLR",
                "phenotype_short": "Kallmann type 1 — anosmia + HH + MIRROR MOVEMENTS pathognomonic",
                "hallmark_short": "Mirror movements (75%) + absent olfactory bulbs MRI + anosmia = ANOS1 Kallmann",
            },
            {
                "gene": "FGFR1",
                "protein": "FGF Receptor 1",
                "aa": "822 aa",
                "locus": "8p11.23",
                "inheritance": "AD (incomplete penetrance)",
                "phenotype_short": "Kallmann type 2 + normosmic CHH spectrum — craniofacial anomalies",
                "hallmark_short": "Cleft palate + dental agenesis + HH → FGFR1; extreme intrafamilial variability",
            },
            {
                "gene": "GNRHR",
                "protein": "GnRH Receptor",
                "aa": "328 aa",
                "locus": "4q13.2",
                "inheritance": "AR biallelic",
                "phenotype_short": "Normosmic IHH type 7 — smell NORMAL, pulsatile GnRH diagnostic + therapeutic",
                "hallmark_short": "Smell normal + IHH: pulsatile GnRH pump → LH surge confirms GNRHR pituitary intact",
            },
            {
                "gene": "KISS1R",
                "protein": "Kisspeptin Receptor (GPR54)",
                "aa": "398 aa",
                "locus": "19p13.3",
                "inheritance": "AR biallelic",
                "phenotype_short": "Normosmic IHH type 15 — kisspeptin-54 test → NO LH rise pathognomonic",
                "hallmark_short": "KP54 stimulation → absent LH = KISS1R LOF; pulsatile GnRH still works",
            },
            {
                "gene": "FMR1",
                "protein": "Fragile X Mental Retardation Protein",
                "aa": "632 aa",
                "locus": "Xq27.3",
                "inheritance": "XL premutation (55-200 CGG)",
                "phenotype_short": "FXPOI — premature ovarian insufficiency; test FMR1 FIRST in all POI",
                "hallmark_short": "FMR1 FIRST in POI workup; premutation (NOT full mutation) causes FXPOI; HRT mandatory",
            },
            {
                "gene": "FOXL2",
                "protein": "Forkhead Box Protein L2",
                "aa": "376 aa",
                "locus": "3q22.3",
                "inheritance": "AD",
                "phenotype_short": "BPES type I: ptosis + blepharophimosis + epicanthus inversus + POI",
                "hallmark_short": "Ptosis repair at age 3-4 FIRST (amblyopia risk); FOXL2 c.402C>G = somatic GCT hot-spot",
            },
            {
                "gene": "BMP15",
                "protein": "Bone Morphogenetic Protein 15",
                "aa": "392 aa",
                "locus": "Xp11.22",
                "inheritance": "XLD",
                "phenotype_short": "POI — oocyte paracrine factor; XLD haploinsufficiency; FMR1 exclude first",
                "hallmark_short": "BMP15 POI: XLD; FMR1 first; HRT mandatory; donor oocyte IVF if reserve exhausted",
            },
            {
                "gene": "PROKR2",
                "protein": "Prokineticin Receptor 2",
                "aa": "384 aa",
                "locus": "20p12.3",
                "inheritance": "AR / AD digenic",
                "phenotype_short": "Kallmann type 3 + normosmic IHH — hypersomnia + obesity clue",
                "hallmark_short": "PROKR2: HH + hypersomnia + obesity → PROKR2; digenic with ANOS1/FGFR1 = compound HH",
            },
        ],
        "category_summary": {
            "hh_kallmann_normosmic_iHH_genes": hh_genes,
            "poi_premature_ovarian_insufficiency_genes": poi_genes,
            "anosmia_rate_atlas_wide_pct": anosmia_rate,
            "poi_patients_total": poi_n,
            "female_patients_total": female_n,
        },
        "aggregate_clinical": {
            "mirror_movements_anos1_pct": _pct(_ALL_COHORTS["ANOS1"], "mirror_movements"),
            "spontaneous_reversal_fgfr1_pct": _pct(_ALL_COHORTS["FGFR1"], "spontaneous_reversal"),
            "gnrh_pump_fertility_rate_pct": _pct(_ALL_COHORTS["GNRHR"], "fertility_achieved"),
            "fmr1_hrt_started_pct": _pct(_ALL_COHORTS["FMR1"], "hrt_started"),
            "foxl2_ptosis_repair_done_pct": _pct(_ALL_COHORTS["FOXL2"], "ptosis_surgery_done"),
            "bmp15_donor_oocyte_pct": _pct(_ALL_COHORTS["BMP15"], "donor_oocyte_ivf"),
        },
        "key_clinical_pearls": [
            "ANOSMIA + ABSENT PUBERTY → Kallmann syndrome panel (ANOS1, FGFR1, PROKR2, FGFR1)",
            "SMELL NORMAL + ABSENT PUBERTY → Normosmic IHH panel (GNRHR, KISS1R, PROKR2 normosmic)",
            "POI WORKUP ORDER: FMR1 FIRST → FOXL2 (eye signs?) → BMP15 (XLD) → karyotype",
            "MIRROR MOVEMENTS (bimanual synkinesis) in Kallmann → ANOS1 until proven otherwise (75% sensitivity)",
            "HH REVERSAL: trial testosterone holiday annually at age 18 — 10-20% FGFR1, 5-10% GNRHR/KISS1R",
            "HRT IN POI: mandatory from diagnosis until age 51 — bone, cardiovascular, cognitive protection",
        ],
    }


def get_breakdown():
    result = []
    for gd in REPRODUCTIVE_GENES:
        gene = gd["gene"]
        cohort = _ALL_COHORTS[gene]
        n = len(cohort)
        ages = [p["age_at_dx"] for p in cohort]
        females = sum(1 for p in cohort if p.get("sex") == "F")
        result.append({
            "gene": gene,
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
            "key_hallmarks": gd["key_hallmarks"],
            "treatment_alerts": gd["treatment_alerts"],
            "ddx": gd["ddx"],
            "cohort_stats": {
                "n": n,
                "seed": gd["seed"],
                "age_mean": round(sum(ages) / n, 1),
                "age_min": min(ages),
                "age_max": max(ages),
                "female_n": females,
                "male_n": n - females,
                "female_pct": round(females / n * 100, 1),
            },
            "patients": cohort,
        })
    return result


def get_definitions():
    return {
        "atlas": "Reproductive-Disorders-Atlas",
        "definitions": [
            {
                "term": "Kallmann Syndrome — Anosmia + Hypogonadotropic Hypogonadism",
                "short": "Deficient GnRH neuron migration → absent olfactory bulb + absent puberty",
                "detail": (
                    "Kallmann syndrome: the combination of anosmia (absent/severely reduced smell) + "
                    "hypogonadotropic hypogonadism (HH). "
                    "PATHOPHYSIOLOGY: GnRH neurons originate in the olfactory placode and migrate along "
                    "olfactory axons to the hypothalamus. In Kallmann syndrome, this migration fails → "
                    "GnRH neurons remain stranded in the nasal region → absent hypothalamic GnRH → "
                    "pituitary receives no GnRH → no LH/FSH secretion → no sex-hormone stimulation → absent puberty. "
                    "MRI: absent olfactory bulbs (bilateral) + olfactory sulci — structural correlate of failed olfactory development. "
                    "GENES: ANOS1 (XLR, mirror movements), FGFR1 (AD, craniofacial), PROKR2 (AR/digenic, sleep/obesity). "
                    "DIAGNOSIS: anosmia on formal testing (Sniffin' Sticks / UPSIT) + low LH/FSH/sex hormones + MRI olfactory sequences. "
                    "TREATMENT: testosterone/estrogen replacement + pulsatile GnRH pump or FSH+hCG for fertility."
                ),
                "clinical_rule": "Anosmia + absent puberty = Kallmann syndrome; smell test is MANDATORY in all delayed puberty workup",
            },
            {
                "term": "Normosmic IHH — Isolated Hypogonadotropic Hypogonadism without Anosmia",
                "short": "HH with NORMAL smell — GnRH neuron migration intact but GnRH signaling disrupted",
                "detail": (
                    "Normosmic IHH (nIHH): absence of puberty + low LH/FSH/sex steroids WITH normal sense of smell. "
                    "MRI: normal olfactory bulbs (GnRH neuron migration succeeded) + normal pituitary morphology. "
                    "GENES: GNRHR (AR, GnRH receptor LOF — most common single gene), KISS1R (AR, kisspeptin receptor LOF), "
                    "FGFR1 (AD spectrum — some normosmic), PROKR2 (AR/digenic normosmic form). "
                    "KEY DISTINCTION FROM KALLMANN: smell is normal (formal olfactory testing confirms). "
                    "PULSATILE GnRH PUMP: restores LH/FSH pulsatility in GNRHR LOF, KISS1R LOF — pituitary gonadotrophs are intact. "
                    "SPONTANEOUS REVERSAL: 10-20% of nIHH patients experience partial/complete return of gonadal axis → "
                    "offer annual testosterone holiday trial after age 18."
                ),
                "clinical_rule": "All HH workup requires formal smell test — normosmia rules out Kallmann, points to GNRHR/KISS1R/PROKR2 nIHH",
            },
            {
                "term": "Mirror Movements (Bimanual Synkinesis) — ANOS1 Hallmark",
                "short": "Involuntary contralateral limb movement during unilateral voluntary action — ANOS1 pathognomonic",
                "detail": (
                    "Mirror movements: when one hand performs a voluntary action (e.g., squeezing a ball), "
                    "the contralateral hand involuntarily mirrors the movement. "
                    "MECHANISM IN ANOS1: Anosmin-1 (ANOS1 protein) guides development of pyramidal tract decussation. "
                    "ANOS1 LOF → partial ipsilateral corticospinal fiber persistence → during unilateral cortical activation, "
                    "ipsilateral fibers activate ipsilateral AND contralateral motor neurons → mirror movement. "
                    "PREVALENCE: ~75% of ANOS1 hemizygous males; only ~15-20% of FGFR1 Kallmann (less common). "
                    "CLINICAL TEST: ask patient to rapidly tap fingers on one hand while observing the other hand. "
                    "PATHOGNOMONIC VALUE: among all Kallmann/IHH genes, prominent mirror movements strongly point to ANOS1. "
                    "No specific treatment needed for mild mirror movements; severe cases: occupational therapy."
                ),
                "clinical_rule": "Mirror movements in Kallmann patient → ANOS1 genetic panel FIRST (75% sensitivity for ANOS1)",
            },
            {
                "term": "Pulsatile GnRH Pump — Diagnostic and Therapeutic Protocol",
                "short": "IV/SC GnRH pulses every 90 min → restores pituitary LH/FSH pulsatility → fertility",
                "detail": (
                    "Pulsatile GnRH pump: an ambulatory device delivering subcutaneous or intravenous GnRH "
                    "in 90-min pulses (mimicking hypothalamic GnRH secretion). "
                    "DIAGNOSTIC USE: pulsatile GnRH → LH/FSH surge confirms pituitary gonadotroph integrity "
                    "(rules out pituitary disease; confirms hypothalamic/receptor origin of HH). "
                    "THERAPEUTIC USE: sustained pulsatile GnRH → progressive LH/FSH normalization → "
                    "testosterone/estrogen production → spermatogenesis (male) or ovulation (female). "
                    "FERTILITY OUTCOMES: testicular volume increases from prepubertal to adult over 12-24 months; "
                    "spermatogenesis in 70-80% of males with Kallmann/nIHH. "
                    "LIMITATIONS: requires ambulatory pump; not effective if pituitary gonadotrophs are destroyed (pituitary failure); "
                    "not needed if GNRHR complete LOF (receptor gone — GnRH won't signal). "
                    "CONTRAST: FSH + hCG injections can substitute but slower spermatogenic response."
                ),
                "clinical_rule": "GnRH pump: LH surge = pituitary intact (hypothalamic/receptor HH); absent LH surge = pituitary disease",
            },
            {
                "term": "Premature Ovarian Insufficiency (POI) — Diagnostic Criteria",
                "short": "FSH >25 IU/L × 2 + irregular/absent menses before age 40",
                "detail": (
                    "POI (premature ovarian insufficiency): "
                    "Diagnostic criteria: (1) oligo/amenorrhea ≥4 months; "
                    "(2) FSH >25 IU/L on two samples ≥4 weeks apart; "
                    "(3) age <40 years. "
                    "PREVIOUSLY CALLED: premature menopause / premature ovarian failure — now 'insufficiency' preferred "
                    "because ~5-10% of POI women ovulate intermittently and can conceive spontaneously. "
                    "GENETIC CAUSES (in workup order): "
                    "1. FMR1 premutation (most common identifiable cause — ~3% sporadic, ~12% familial); "
                    "2. FOXL2 (BPES type I — eye signs present); "
                    "3. BMP15 (XLD — no eye signs); "
                    "4. Karyotype (Turner 45,X or mosaic); "
                    "5. Autoimmune panel. "
                    "HRT: mandatory from diagnosis until age ~51 (natural menopause age) — osteoporosis + cardiovascular risk."
                ),
                "clinical_rule": "POI workup: FSH ×2 → FMR1 FIRST → eye exam (FOXL2) → BMP15 → karyotype → autoimmune screen",
            },
            {
                "term": "FXPOI — Fragile X-Associated Primary Ovarian Insufficiency",
                "short": "FMR1 premutation (55-200 CGG) → toxic mRNA → granulosa cell failure → POI",
                "detail": (
                    "FXPOI: POI in FMR1 premutation carriers (55-200 CGG repeats). "
                    "MECHANISM: premutation range → FMR1 transcribed at increased rate → "
                    "expanded CGG mRNA toxic to granulosa cells → sequestration of DROSHA/DGCR8 → "
                    "miRNA dysregulation → premature follicle atresia → POI. "
                    "CRITICAL DISTINCTION FROM FULL MUTATION: "
                    "Full mutation (>200 CGG): FMR1 silenced (methylated) → absent FMRP → FXS (intellectual disability, autism). "
                    "Premutation (55-200 CGG): FMR1 transcribed at excess → mRNA toxicity → FXPOI (no cognitive impairment). "
                    "PREVALENCE: 1 in 150-300 females are FMR1 premutation carriers; ~20% develop FXPOI. "
                    "GENETIC RISK: premutation expands to full mutation (>200 CGG) in offspring; "
                    "sons may have FXS; daughters may be premutation carriers. "
                    "FXTAS: premutation MALES >50 years → tremor + ataxia (repeat-associated non-ATG translation)."
                ),
                "clinical_rule": "FMR1 FIRST in ALL POI workup; premutation (not full mutation) = FXPOI; HRT mandatory; genetic counseling for offspring FXS risk",
            },
            {
                "term": "BPES — Blepharophimosis-Ptosis-Epicanthus Inversus Syndrome",
                "short": "FOXL2 AD — horizontal eyelid narrowing + ptosis + medial fold + POI (type I)",
                "detail": (
                    "BPES: autosomal dominant eyelid malformation caused by FOXL2 LOF. "
                    "FOUR FEATURES: "
                    "(1) Blepharophimosis: reduced horizontal palpebral fissure (<22 mm at birth); "
                    "(2) Ptosis: drooping upper eyelid (levator palpebrae weakness); "
                    "(3) Epicanthus inversus: skin fold running from lower lid to medial canthus; "
                    "(4) Telecanthus: increased medial canthal distance. "
                    "TYPE I vs TYPE II: "
                    "Type I: all four signs + POI (FSH elevated, menses irregular, fertility reduced). "
                    "Type II: all four signs WITHOUT POI (molecular distinction: dominant negative variants → type I; "
                    "haploinsufficiency → type II). "
                    "SURGICAL PRIORITY: ptosis repair at age 3-4 years MANDATORY → frontalis suspension; "
                    "delay beyond 4 years → stimulus-deprivation amblyopia (irreversible visual loss). "
                    "SOMATIC FOXL2 p.Cys134Trp: found in >90% adult granulosa cell tumors (not BPES)."
                ),
                "clinical_rule": "BPES: ptosis repair at age 3-4 (amblyopia prevention) before gonadal management; type I = add POI workup",
            },
            {
                "term": "HH Reversal — Spontaneous Recovery of Gonadal Axis",
                "short": "10-20% of IHH/Kallmann patients develop spontaneous partial/complete axis recovery after cessation of TRT",
                "detail": (
                    "HH reversal: the documented phenomenon of spontaneous return of pulsatile LH secretion and "
                    "gonadal steroid production in a subset (~10-20%) of IHH/Kallmann patients after "
                    "cessation of testosterone replacement therapy. "
                    "GENES MOST COMMONLY SHOWING REVERSAL: FGFR1 (~15-20%), GNRHR partial LOF, KISS1R hypomorphic. "
                    "MECHANISM: possibly incomplete penetrance / residual GnRH neuronal activity that is suppressed by exogenous testosterone. "
                    "CLINICAL PROTOCOL: after age 18, recommend testosterone holiday (cessation for 3-6 months); "
                    "monitor LH/FSH/testosterone at 0, 6, 12 weeks; "
                    "if LH >1 IU/L and rising → reversal confirmed; "
                    "continue monitoring annual (reversal can also re-revert). "
                    "IMPLICATION: IHH is NOT always a permanent diagnosis — avoid premature declaration of permanent infertility."
                ),
                "clinical_rule": "Annual testosterone holiday trial (3-6 months) at age 18+ in all IHH patients — 10-20% show reversal; do NOT tell patient fertility is impossible",
            },
            {
                "term": "Kisspeptin-54 Stimulation Test — KISS1R vs GNRHR Distinction",
                "short": "IV KP54 → LH surge: absent in KISS1R LOF; present in GNRHR LOF (GnRH neurons intact)",
                "detail": (
                    "Kisspeptin-54 (KP54) stimulation test: IV infusion of kisspeptin-54 → measure LH at 0, 30, 60, 90 min. "
                    "NORMAL RESPONSE: LH rises ≥0.5 IU/L above baseline (kisspeptin activates KISS1R on GnRH neurons → GnRH release → pituitary LH). "
                    "KISS1R LOF: absent LH response to KP54 (KISS1R receptor dysfunctional → cannot transduce kisspeptin signal). "
                    "GNRHR LOF: LH response to KP54 also ABSENT — because even if KISS1R is intact and GnRH is released, "
                    "pituitary GNRHR is dysfunctional → no LH response at pituitary level. "
                    "DIFFERENTIATING KEY: "
                    "KISS1R LOF → KP54 absent, pulsatile GnRH PRESENT (pituitary responsive); "
                    "GNRHR LOF → KP54 absent, pulsatile GnRH also ABSENT (pituitary receptor dysfunctional). "
                    "ANOS1/FGFR1 Kallmann: both KP54 and pulsatile GnRH → LH present (pituitary intact; defect is in GnRH neuron migration)."
                ),
                "clinical_rule": "KP54 absent + GnRH pump absent = GNRHR LOF; KP54 absent + GnRH pump present = KISS1R LOF; both present = Kallmann upstream defect",
            },
            {
                "term": "HRT in POI — Mandatory Until Natural Menopause Age",
                "short": "Estradiol + progesterone replacement mandatory from POI diagnosis until age ~51 — not optional",
                "detail": (
                    "POI causes estrogen deficiency decades earlier than natural menopause. "
                    "Consequences of untreated POI: "
                    "(1) Osteoporosis: bone density falls at 1-3% per year without HRT; fracture risk doubled by age 60; "
                    "(2) Cardiovascular: premature atherosclerosis, endothelial dysfunction; "
                    "(3) Cognitive: increased dementia risk in women with early untreated estrogen deficiency; "
                    "(4) Genitourinary: vaginal atrophy, dyspareunia, recurrent UTI. "
                    "HRT PRESCRIPTION: continuous estradiol (transdermal preferred: 75-100 mcg/24h patch or equivalent oral) + "
                    "cyclic or continuous progestin (mandatory if uterus present to prevent endometrial hyperplasia). "
                    "DURATION: continue until age ~51 (natural median menopause age) — not indefinitely like surgical menopause. "
                    "BREAST CANCER CONCERN: HRT in POI to age 51 does NOT increase breast cancer risk above baseline "
                    "(restores natural levels, not supraphysiologic)."
                ),
                "clinical_rule": "POI: start HRT IMMEDIATELY at diagnosis; do NOT withhold for fear of HRT risks — POI HRT restores normal physiology, not excess",
            },
            {
                "term": "Cascade Testing — Reproductive Disorders",
                "short": "First-degree relatives of Kallmann/IHH/POI patients require targeted genetic testing",
                "detail": (
                    "Cascade testing for hereditary reproductive disorders: "
                    "ANOS1 (XLR): test all brothers of affected males (50% carrier if mother is carrier); "
                    "test all sisters — obligate carriers if mother is carrier; "
                    "daughters of affected males: ALL obligate carriers. "
                    "FGFR1 (AD): 50% risk to each child; offer to all first-degree relatives; "
                    "incomplete penetrance (40-70%) means relatives may have subtle phenotype (hyposmia, delayed puberty). "
                    "GNRHR/KISS1R (AR): siblings at 25% risk; parents are obligate carriers; "
                    "test siblings → if carrier, no action needed (AR); if biallelic → refer. "
                    "FMR1 premutation: maternal uncles at risk for FXTAS; maternal female relatives for FXPOI; "
                    "children for full mutation (FXS) risk. "
                    "FOXL2 (AD): 50% risk to each child of affected parent. "
                    "BMP15 (XLD): daughters of affected females at 50% risk."
                ),
                "clinical_rule": "Hereditary reproductive disorder confirmed → cascade test all first-degree relatives before reproductive decisions are made",
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
    print("\n=== DEFINITIONS (first 2) ===")
    df = get_definitions()
    print(json.dumps(df["definitions"][:2], indent=2)[:800])
