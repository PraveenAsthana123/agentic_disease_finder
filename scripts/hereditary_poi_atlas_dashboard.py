#!/usr/bin/env python3
"""Hereditary-Primary-Ovarian-Insufficiency-Atlas — Complete 8-Gene Atlas
(FMR1 · FOXL2 · BMP15 · GDF9 · NR5A1 · NOBOX · FIGLA · MCM8).

FMR1     (FMRP; 632 aa; ~71 kDa; Xq27.3; X-linked dominant/premutation;
           Fragile X-associated Primary Ovarian Insufficiency (FXPOI);
           premutation 55-200 CGG repeats (NOT full mutation causes POI);
           20-28% of female premutation carriers; earliest menopause among identifiable causes;
           ALSO increases Parkinson-like tremor-ataxia (FXTAS) in older males/females;
           seed SEED_BASE+0).
FOXL2    (Forkhead Box L2; 376 aa; ~44 kDa; 3q22.3; AD;
           Blepharophimosis-Ptosis-Epicanthus-Inversus Syndrome (BPES) type I = POI;
           BPES type II = eyelid anomaly without POI; granulosa cell TF;
           FOXL2 c.402C>G (p.Cys134Trp) somatic = adult-type granulosa cell tumour;
           seed SEED_BASE+1).
BMP15    (Bone Morphogenetic Protein 15; 392 aa; ~46 kDa; Xp11.22; X-linked dominant/AR;
           oocyte-specific TGF-β superfamily ligand; paracrine signal to granulosa cells;
           heterozygous females: POI (XLD); homozygous females: more severe POI;
           heterozygous males: fertile (hemizygous); key trigger: FMR1 exclude first;
           seed SEED_BASE+2).
GDF9     (Growth Differentiation Factor 9; 454 aa; ~52 kDa; 5q31.1; AR/AD;
           oocyte-secreted TGF-β superfamily; BMP15-GDF9 heterodimer activates granulosa;
           AR biallelic: severe POI/primary amenorrhoea; heterozygous: premature menopause AD;
           seed SEED_BASE+3).
NR5A1    (Nuclear Receptor Subfamily 5, Group A, Member 1 / SF-1; 461 aa; ~52 kDa; 9q33.3; AD;
           Steroidogenic Factor 1; master regulator of adrenal, gonadal, pituitary development;
           heterozygous dominant-negative → 46,XX POI (± adrenal insufficiency risk);
           46,XY NR5A1 LOF → 46,XY DSD (female/ambiguous external genitalia);
           seed SEED_BASE+4).
NOBOX    (Newborn Ovary Homeobox; 672 aa; ~75 kDa; 7q35; AR;
           oocyte-specific homeobox transcription factor; primordial-to-primary follicle transition;
           biallelic LOF → primary amenorrhoea (no follicles at menarche);
           one of the most common AR POI genes in non-consanguineous populations;
           seed SEED_BASE+5).
FIGLA    (Factor In the Germline Alpha; 115 aa; ~13 kDa; 2p13.3; AR;
           basic helix-loop-helix (bHLH) TF; master regulator of primordial follicle assembly;
           oocyte-specific; biallelic LOF → complete primary amenorrhoea; very rare;
           seed SEED_BASE+6).
MCM8     (Minichromosome Maintenance 8 Helicase; 840 aa; ~93 kDa; 20p12.3; AR;
           DNA repair helicase; MCM8-MCM9 complex resolves DNA replication stress in meiosis;
           biallelic LOF → oocyte DNA repair failure → premature meiotic arrest → POI;
           associated with chromosomal instability; Lynch-overlap phenotype reported;
           seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2462-2469).
"""

import random

SEED_BASE = 2462

POI_GENES = [
    # -- FMR1 -- Fragile X-associated Primary Ovarian Insufficiency (FXPOI) -------------------------
    {
        "gene": "FMR1",
        "alt_name": (
            "FMR1 (FMR1-632aa-Xq27.3 / X-linked-dominant-premutation -- "
            "FXPOI-Fragile-X-Primary-Ovarian-Insufficiency -- "
            "PREMUTATION-55-200-CGG-NOT-FULL-MUTATION-CAUSES-POI -- "
            "20-28pct-FEMALE-PREMUTATION-CARRIERS-POI -- "
            "EARLIEST-MENOPAUSE-IDENTIFIABLE-GENETIC-CAUSE -- "
            "TEST-FIRST-IN-ALL-POI-FMR1-MOST-COMMON-IDENTIFIABLE)"
        ),
        "protein": (
            "FMR1 -- Xq27.3 XLD-premutation -- FMR1-632aa -- "
            "Fragile-X-Mental-Retardation-Protein-71kDa-RNA-Binding-Polyribosome-Translational-Regulator -- "
            "Premutation-mRNA-Toxic-Gain-of-Function-Elevated-FMR1-mRNA-Granulosa-Cell-Toxicity -- "
            "OMIM-Gene-309550-Disease-FXPOI-311360"
        ),
        "locus": "Xq27.3",
        "protein_size": "632 aa / ~71 kDa",
        "inheritance": (
            "X-linked dominant (premutation carrier; NOT full mutation >200 CGG). "
            "MECHANISM: Full mutation (>200 CGG) → gene silencing → Fragile X syndrome (ID, no POI). "
            "Premutation (55-200 CGG) → ELEVATED FMR1 mRNA (not silenced) → toxic mRNA accumulates in "
            "granulosa cells → RNA-protein sequestration → mitochondrial dysfunction → premature "
            "follicular atresia → POI. Incidence: 1/178 to 1/259 women carry the premutation. "
            "POI RISK: 20-28% of female premutation carriers develop POI (vs <1% population). "
            "Mean age of menopause in FXPOI carriers: 32-38 years (vs 51 years general population). "
            "FXTAS: In older premutation carriers (male and female) → Fragile X-associated tremor/"
            "ataxia syndrome (FXTAS) — intention tremor, cerebellar ataxia, parkinsonism. "
            "FAMILY: If patient has FMR1 premutation, offspring at 50% risk of premutation/full mutation; "
            "full mutation children (especially males) have Fragile X intellectual disability — "
            "GENETIC COUNSELLING MANDATORY before any fertility treatment. "
            "FERTILITY: IVF may succeed with own eggs if FSH not markedly elevated; egg donation if "
            "FSH >20 IU/L and AFC <5; pulsatile GnRH does not reverse POI (different mechanism to HH)."
        ),
        "disease_category": (
            "FXPOI (Fragile X-associated Primary Ovarian Insufficiency); most common identifiable genetic "
            "cause of POI (identifiable in ~5% of sporadic POI, 12-15% of familial POI); "
            "premutation 55-200 CGG; earlier age of menopause proportional to repeat length within 55-200 range; "
            "premutation diagnosed by PCR + Southern blot; standard FMR1 gene panel (not WES) required"
        ),
        "disease_pathway": (
            "FMR1 PREMUTATION PATHOGENESIS IN OVARY: "
            "Normal FMR1 mRNA levels; premutation → excess FMR1 mRNA (ELEVATED, not suppressed) → "
            "CGG-repeat RNA forms G-quadruplex hairpins → sequesters RNA-binding proteins (DROSHA, HNRNP) → "
            "dysregulates micro-RNA processing → mitochondrial dysfunction in granulosa cells → "
            "increased apoptosis → accelerated follicular atresia → reduced ovarian reserve. "
            "TIMELINE: Reduced AFC (antral follicle count) detectable by mid-20s in carriers; "
            "FSH rises earlier than non-carriers; AMH falls below 1 ng/mL earlier. "
            "REPEAT SIZE CORRELATION: Higher repeat count (closer to 200) → earlier FXPOI; "
            "size range 59-100 CGG carries highest POI risk (~42% in some series). "
            "NOT REVERSIBLE: No treatment restores follicular reserve; focus on fertility preservation "
            "(cryopreservation before POI is complete) and HRT. "
            "BRAIN: Same mRNA gain-of-function in Purkinje cells → FXTAS (tremor-ataxia, parkinsonism); "
            "screen premutation carriers neurologically over time."
        ),
        "pathognomonic": (
            "FMR1 / FXPOI CLINICAL PEARLS: "
            "PATHOGNOMONIC CLUE: Family history of Fragile X intellectual disability (Fragile X syndrome "
            "in a male relative) + POI = FXPOI until proven otherwise. "
            "TESTING: PCR for repeat sizing + Southern blot for full expansion; standard WES misses "
            "CGG repeat expansions — MUST use dedicated FMR1 PCR assay. "
            "PREMUTATION RANGE: 55-200 CGG; grey zone: 45-54 CGG (uncertain risk). "
            "FULL MUTATION: >200 CGG causes Fragile X syndrome (male), NOT POI; full mutation → "
            "gene silenced (methylation) → no toxic mRNA → no POI. "
            "FERTILITY COUNSELLING: All female premutation carriers should discuss fertility preservation "
            "in early 20s — egg freezing BEFORE FSH elevation is optimal. "
            "OFFSPRING RISK: Premutation → full mutation possible on maternal transmission (not paternal). "
            "FXTAS SCREEN: Annual neurological exam in premutation carriers aged >40y (both M/F). "
            "HRT MANDATORY: POI from any cause including FXPOI — oestrogen + progestogen until age 51."
        ),
        "treatment": (
            "HRT MANDATORY: Transdermal oestradiol (100 mcg/day) + cyclical progestogen until age 51 — "
            "bone, cardiovascular, cognitive, vasomotor protection. "
            "FERTILITY: Own eggs feasible if AFC ≥5 and FSH <20 IU/L; IVF with flare/antagonist protocol; "
            "egg donation if reserves depleted. "
            "FERTILITY PRESERVATION: Offer oocyte/embryo cryopreservation to ALL premutation carriers "
            "in their 20s before FSH rises — PROACTIVE not reactive. "
            "GENETIC COUNSELLING: Mandatory pre-pregnancy; offspring carry premutation/full mutation risk. "
            "FXTAS MANAGEMENT: No disease-modifying therapy; memantine trialled; physiotherapy; fall prevention. "
            "PSYCHOLOGICAL: POI diagnosis + Fragile X family implications — dual bereavement counselling."
        ),
        "key_features": [
            "FXPOI: 20-28% of premutation carriers (55-200 CGG repeats)",
            "TEST FIRST: FMR1 PCR is mandatory first-line in ALL POI — most common identifiable cause",
            "FULL MUTATION >200 CGG causes Fragile X syndrome (NOT POI) — gene silenced",
            "WES MISSES: CGG repeat expansions require dedicated FMR1 PCR + Southern blot",
            "FERTILITY PRESERVATION: offer egg freezing in 20s before FSH rises",
            "HRT MANDATORY: transdermal oestradiol + progestogen until age 51",
            "FXTAS: intention tremor + cerebellar ataxia in older premutation carriers",
            "OFFSPRING RISK: premutation can expand to full mutation on maternal transmission",
        ],
        "key_ddx": (
            "FMR1 vs FOXL2: FOXL2 has visible eyelid anomalies (blepharophimosis); FMR1 has FX family history; "
            "FMR1 vs Turner: karyotype 46,XX in FMR1 (not 45,X); "
            "FMR1 vs iatrogenic: exclude chemotherapy/radiation/autoimmune before genetic testing; "
            "Premutation vs full mutation: PCR + Southern blot mandatory to distinguish"
        ),
        "systemic_involvement": (
            "NEUROLOGICAL: FXTAS (tremor-ataxia-parkinsonism) in premutation carriers >50y; "
            "white matter lesions on MRI (middle cerebellar peduncle T2 hyperintensity — near-PATHOGNOMONIC FXTAS); "
            "REPRODUCTIVE: POI with reduced AFC, elevated FSH, low AMH; "
            "COGNITIVE: mild executive dysfunction in some premutation females (not full Fragile X ID); "
            "CARDIAC: mitral valve prolapse in ~50% of Fragile X (full mutation males); "
            "THYROID: autoimmune thyroiditis co-association; check annually"
        ),
        "onset_age": "Premature ovarian insufficiency typically diagnosed 25-38 years in carriers; FXTAS >50y",
        "surgical_urgency": "NOT a surgical emergency; URGENT fertility counselling in premutation carriers aged <35y",
        "gene_family": "FMR1 — mRNA-binding protein; CGG trinucleotide repeat expansion disease",
        "morphology": "Ovaries small on USS with reduced AFC; AMH very low or undetectable",
        "n_patients": 40,
    },
    # -- FOXL2 -- BPES type I + Primary Ovarian Insufficiency ------------------------------------
    {
        "gene": "FOXL2",
        "alt_name": (
            "FOXL2 (FOXL2-376aa-3q22.3 / AD -- "
            "BPES-Blepharophimosis-Ptosis-Epicanthus-Inversus-Syndrome -- "
            "BPES-TYPE-I-EYELID-ANOMALY-PLUS-POI -- "
            "BPES-TYPE-II-EYELID-ONLY-NO-POI -- "
            "PTOSIS-REPAIR-AGE-3-4-MANDATORY-AMBLYOPIA-RISK -- "
            "FOXL2-Cys134Trp-SOMATIC-ADULT-GRANULOSA-CELL-TUMOUR)"
        ),
        "protein": (
            "FOXL2 -- 3q22.3 AD -- FOXL2-376aa -- "
            "Forkhead-Box-L2-44kDa-Forkhead-Domain-TF-Granulosa-Cell-Identity -- "
            "Master-Regulator-Granulosa-Cell-Fate-Represses-SOX9-Testis-Determination -- "
            "OMIM-Gene-605597-Disease-BPES-110100"
        ),
        "locus": "3q22.3",
        "protein_size": "376 aa / ~44 kDa",
        "inheritance": (
            "Autosomal dominant. FOXL2 is the MASTER transcription factor for granulosa cell identity. "
            "In the ovary, FOXL2 maintains granulosa cell fate throughout reproductive life by: "
            "(1) suppressing SOX9 and other testis-determination genes; "
            "(2) maintaining FSH receptor expression; "
            "(3) regulating oestrogen biosynthesis (aromatase/CYP19A1 expression). "
            "LOF MECHANISM: FOXL2 haploinsufficiency → granulosa cells transdifferentiate toward "
            "Sertoli-like cells → follicular dysfunction → accelerated atresia → POI. "
            "BPES type I (majority): eyelid anomaly + POI; BPES type II: eyelid anomaly ALONE (no POI). "
            "GENOTYPE-PHENOTYPE: polyalanine tract expansions (most common) → BPES type II; "
            "truncating/missense outside polyalanine tract → more often BPES type I + POI. "
            "SOMATIC FOXL2 p.Cys134Trp: NOT germline — this somatic mutation causes ADULT-TYPE granulosa "
            "cell tumour (AGCT); diagnostic marker for AGCT on tumour biopsy; OPPOSITE clinical context."
        ),
        "disease_category": (
            "BPES type I (Blepharophimosis-Ptosis-Epicanthus Inversus Syndrome + POI): AD; "
            "characteristic eyelid triad VISIBLE AT BIRTH: blepharophimosis (reduced horizontal palpebral fissure), "
            "ptosis (drooping upper lid), and epicanthus inversus (skin fold at inner canthus pointing upward); "
            "BPES type I includes POI (premature ovarian insufficiency); "
            "BPES type II: same eyelid triad WITHOUT POI"
        ),
        "disease_pathway": (
            "FOXL2 IN GRANULOSA CELL IDENTITY: "
            "Normal FOXL2 continuously suppresses SOX9/SOX8 expression in granulosa cells throughout adult life. "
            "FOXL2 binds CYP19A1 promoter → activates aromatase → oestradiol production. "
            "FOXL2 LOF → de-repression of SOX9 → granulosa cells begin to express Sertoli markers → "
            "granulosa-to-Sertoli transdifferentiation → follicle dysfunction → atresia → POI. "
            "EYELID: FOXL2 is expressed in periocular mesenchyme during embryogenesis; "
            "haploinsufficiency → failure of eyelid opening/morphogenesis → blepharophimosis-ptosis. "
            "EPICANTHUS INVERSUS: medial canthus skin fold displaced upward vs downward in typical epicanthus; "
            "results from same periocular mesenchymal FOXL2 deficiency. "
            "GRANULOSA CELL TUMOUR: Somatic FOXL2 p.Cys134Trp (acquired, not germline) → "
            "dominant-negative disruption of FOXL2 dimer → de-repression of steroidogenesis → "
            "granulosa cell proliferation → adult granulosa cell tumour — SCREEN germline BPES patients."
        ),
        "pathognomonic": (
            "FOXL2 / BPES PATHOGNOMONIC FEATURES: "
            "VISIBLE AT BIRTH: triad of blepharophimosis + ptosis + epicanthus inversus — HIGHLY DISTINCTIVE. "
            "PTOSIS REPAIR TIMING: must repair ptosis by age 3-4 years — amblyopia risk; "
            "if ptosis blocks visual axis → deprivational amblyopia develops PERMANENTLY. "
            "BLEPHAROPHIMOSIS REPAIR: medial and lateral canthal tendon reconstruction (5-7 years); "
            "two-stage approach: epicanthus and blepharophimosis correction (age 3-4) → "
            "ptosis repair after canthal correction. "
            "POI SURVEILLANCE: All BPES type I girls — annual FSH/AMH/AFC from age 18y; "
            "fertility counselling at diagnosis. "
            "SOMATIC FOXL2 Cys134Trp: if postmenopausal woman with unexplained oestrogen production or "
            "granulosa cell tumour → send tumour biopsy for FOXL2 Cys134Trp (diagnostic). "
            "GERMLINE AGCT RISK: Germline FOXL2 LOF carriers do NOT have elevated granulosa cell tumour "
            "risk — the somatic Cys134Trp is a gain-of-function driver, different mechanism. "
            "TELOMERE LENGTH: FOXL2 also regulates telomere length in granulosa cells — POI partly "
            "explained by accelerated telomere shortening."
        ),
        "treatment": (
            "EYELID SURGERY: Ptosis repair age 3-4y (URGENT — amblyopia prevention); "
            "blepharophimosis + epicanthus inversus correction (age 5-7y); ophthalmology follow-up lifelong. "
            "HRT: Transdermal oestradiol + cyclical progestogen from POI diagnosis until age 51; "
            "bone densitometry at diagnosis and every 5 years. "
            "FERTILITY: IVF with own eggs if AFC ≥5 and FSH <20 IU/L; egg donation if reserves exhausted; "
            "oocyte cryopreservation offered proactively in adolescence. "
            "AMBLYOPIA: Patching therapy + glasses if detected; visual acuity monitoring. "
            "SCREENING: Annual FSH + AMH + AFC from puberty in BPES type I; luteal phase progesterone "
            "if irregular cycles."
        ),
        "key_features": [
            "BPES type I: blepharophimosis + ptosis + epicanthus inversus + POI (visible at birth)",
            "PTOSIS REPAIR MANDATORY age 3-4 years — amblyopia risk is IRREVERSIBLE if delayed",
            "BPES type II: same eyelid triad WITHOUT POI (different FOXL2 variant class)",
            "Somatic FOXL2 Cys134Trp (NOT germline) → adult-type granulosa cell tumour",
            "FOXL2 maintains granulosa cell identity — LOF → transdifferentiation toward Sertoli-like",
            "HRT mandatory from POI diagnosis until age 51",
            "Fertility counselling at diagnosis — oocyte cryopreservation in adolescence",
            "Annual FSH/AMH/AFC surveillance from puberty in all BPES type I",
        ],
        "key_ddx": (
            "BPES vs simple ptosis: BPES has blepharophimosis + epicanthus inversus NOT present in isolated ptosis; "
            "FOXL2 BPES vs Turner: karyotype 46,XX; visible eyelid triad absent in Turner; "
            "BPES type I vs II: POI present = type I; absent = type II; genotype-phenotype overlap; "
            "Granulosa cell tumour: somatic Cys134Trp (on tumour biopsy) vs germline LOF (BPES)"
        ),
        "systemic_involvement": (
            "OPHTHALMOLOGICAL: blepharophimosis, ptosis, epicanthus inversus PATHOGNOMONIC; "
            "amblyopia risk if ptosis not repaired early; strabismus; refractive error; "
            "REPRODUCTIVE: POI in BPES type I; granulosa cell tumour risk (monitor with pelvic USS); "
            "SKELETAL: none specific; "
            "NEUROLOGICAL: none specific (unlike FMR1)"
        ),
        "onset_age": "Eyelid anomalies present at birth; POI typically age 20-35 years",
        "surgical_urgency": "URGENT: ptosis repair by age 3-4 years (amblyopia prevention) — do NOT delay",
        "gene_family": "FOXL2 — Forkhead box transcription factor; granulosa cell identity master regulator",
        "morphology": "Ovaries with reduced follicular reserve on USS; eyelid anomalies visible externally",
        "n_patients": 40,
    },
    # -- BMP15 -- X-linked POI (Bone Morphogenetic Protein 15) -----------------------------------
    {
        "gene": "BMP15",
        "alt_name": (
            "BMP15 (BMP15-392aa-Xp11.22 / X-linked-dominant -- "
            "OOCYTE-SPECIFIC-TGFbeta-SUPERFAMILY-LIGAND -- "
            "HETEROZYGOUS-FEMALES-POI-X-LINKED-DOMINANT -- "
            "HEMIZYGOUS-MALES-UNAFFECTED-FERTILE -- "
            "FMR1-EXCLUDE-FIRST-THEN-BMP15 -- "
            "BMP15-GDF9-HETERODIMER-SYNERGISTIC-GRANULOSA-SIGNALLING)"
        ),
        "protein": (
            "BMP15 -- Xp11.22 XLD -- BMP15-392aa -- "
            "Bone-Morphogenetic-Protein-15-46kDa-TGFbeta-Superfamily-Oocyte-Specific-Secreted-Ligand -- "
            "Paracrine-Signal-Granulosa-SMAD1-5-8-PI3K-Proliferation-Survival -- "
            "OMIM-Gene-300247-Disease-POI4-300510"
        ),
        "locus": "Xp11.22",
        "protein_size": "392 aa / ~46 kDa",
        "inheritance": (
            "X-linked dominant. BMP15 is an oocyte-specific member of the TGF-beta superfamily. "
            "MECHANISM: BMP15 is secreted by the oocyte → signals to granulosa cells via BMPR2 + ALK6 → "
            "SMAD1/5/8 phosphorylation → promotes granulosa cell proliferation and survival → "
            "normal folliculogenesis. LOF → granulosa cell apoptosis → follicular atresia → POI. "
            "X-LINKED DOMINANT: Heterozygous females have ONE functional copy = insufficient for normal "
            "folliculogenesis → POI. Hemizygous males (ONE copy only) → fertile (male testes do not require BMP15). "
            "SEVERITY: Heterozygous females range from subfertility to complete POI; "
            "homozygous females (very rare, consanguineous) → more severe POI from puberty. "
            "BMP15-GDF9 HETERODIMER: BMP15 and GDF9 (oocyte-secreted) form heterodimers (cumulin) that "
            "are more potent than either homodimer alone → ovarian hyperstimulation sensitivity; "
            "BMP15 heterozygous mutations reduce cumulin production → poor follicular response."
        ),
        "disease_category": (
            "X-linked dominant POI (POI type 4); heterozygous females → POI; "
            "mean age at POI: 28-34 years in carriers; typically identified via family history of "
            "early menopause in female relatives + X-linked pedigree (no male-to-male transmission); "
            "BMP15 is the SECOND most common X-linked POI gene after FMR1"
        ),
        "disease_pathway": (
            "BMP15 OOCYTE-GRANULOSA PARACRINE AXIS: "
            "Oocyte secretes BMP15 → binds BMPRII / ALK6 heterodimer on granulosa membrane → "
            "SMAD1/5/8 phosphorylation + SMAD4 co-activation → nucleus → transcription of: "
            "Kit ligand (KITLG), Follistatin (FST), LH receptor (LHCGR), granulosa survival genes → "
            "normal antral follicle growth. BMP15 LOF → reduced KITLG → oocyte survival compromised; "
            "reduced FST → activin unopposed → FSH receptor downregulated → poor FSH response. "
            "OVARIAN HYPERSTIMULATION: BMP15 partial LOF variants have PARADOXICAL association with "
            "ovarian hyperstimulation syndrome (OHSS) risk in IVF — reduced BMP15 → impaired "
            "desensitisation of granulosa to FSH → exaggerated FSH response in residual follicles. "
            "FERTILITY TREATMENT IMPLICATION: Low-dose FSH protocol mandatory to avoid OHSS in BMP15 carriers."
        ),
        "pathognomonic": (
            "BMP15 / X-LINKED POI CLINICAL PEARLS: "
            "PEDIGREE CLUE: Multiple maternal female relatives with early menopause (25-35y) + "
            "NO AFFECTED MALES (X-linked inheritance) → BMP15 or FMR1. "
            "FMR1 FIRST: Always exclude FMR1 premutation (PCR) before BMP15 testing. "
            "OHSS RISK: BMP15 partial LOF variants → paradoxically higher OHSS risk in IVF; "
            "use LOW DOSE FSH with careful monitoring; antagonist protocol preferred. "
            "TESTING: BMP15 by gene sequencing (NGS) + MLPA for large deletions (X-linked CNV). "
            "RECURRENCE RISK: Heterozygous carrier mother → 50% daughters affected; sons unaffected. "
            "AMH: Very low or undetectable even before age 30 in BMP15 heterozygotes with POI. "
            "CUMULIN: BMP15-GDF9 heterodimer — mutations in either BMP15 or GDF9 reduce cumulin; "
            "test BOTH genes if one is negative and phenotype is severe."
        ),
        "treatment": (
            "HRT: Transdermal oestradiol + cyclical progestogen until age 51; bone DEXA at diagnosis. "
            "FERTILITY: IVF feasible if AFC ≥4-5 (low-dose FSH; antagonist protocol); "
            "OHSS PREVENTION: trigger with GnRH agonist (not hCG) if leading follicles >3; "
            "egg donation if reserves exhausted. "
            "FERTILITY PRESERVATION: Oocyte cryopreservation in early 20s before AFC falls. "
            "RECURRENCE COUNSELLING: 50% daughters at risk; prenatal testing available. "
            "BONE HEALTH: Calcium + vitamin D; DEXA every 5 years while on HRT."
        ),
        "key_features": [
            "X-linked dominant POI: heterozygous females affected; hemizygous males fertile (unaffected)",
            "Oocyte-specific BMP15 signals to granulosa via BMPR2/ALK6 → SMAD1/5/8",
            "FMR1 must be excluded first — BMP15 is second-line X-linked POI gene",
            "OHSS RISK in IVF: low-dose FSH + GnRH agonist trigger mandatory",
            "BMP15-GDF9 heterodimer (cumulin) — 10x more potent than homodimers",
            "HRT mandatory until age 51; fertility preservation in early 20s",
            "MLPA + sequencing: large Xp deletions cause BMP15 + nearby gene losses",
            "No affected males in pedigree — key X-linked inheritance clue",
        ],
        "key_ddx": (
            "BMP15 vs FMR1: FMR1 has family history of Fragile X syndrome; BMP15 no neurological features; "
            "BMP15 vs Turner: karyotype 46,XX; BMP15 no somatic features; "
            "BMP15 vs GDF9: both oocyte TGF-β ligands; BMP15 X-linked; GDF9 chromosome 5 AR/AD; "
            "BMP15 LOF vs OHSS risk: partial LOF → increased OHSS; null LOF → severe POI"
        ),
        "systemic_involvement": (
            "REPRODUCTIVE: POI with reduced AFC, elevated FSH, low AMH; "
            "OHSS risk paradox in IVF; "
            "NEUROLOGICAL: none (unlike FMR1); "
            "SKELETAL: osteoporosis secondary to hypoestrogenism if HRT not given; "
            "No somatic dysmorphic features (unlike FOXL2/Turner)"
        ),
        "onset_age": "POI typically 25-35 years; occasionally primary amenorrhoea in severe homozygous",
        "surgical_urgency": "No surgical emergency; urgent fertility counselling in early 20s",
        "gene_family": "BMP15 — TGF-β superfamily (BMP subgroup); oocyte-secreted paracrine factor",
        "morphology": "Ovaries small with low AFC; USS normal morphology but reduced follicular pool",
        "n_patients": 40,
    },
    # -- GDF9 -- Growth Differentiation Factor 9 (AR/AD POI) ------------------------------------
    {
        "gene": "GDF9",
        "alt_name": (
            "GDF9 (GDF9-454aa-5q31.1 / AR-biallelic-severe-AD-heterozygous-premature-menopause -- "
            "OOCYTE-SECRETED-TGFbeta-SUPERFAMILY -- "
            "BMP15-GDF9-CUMULIN-HETERODIMER-10x-POTENCY -- "
            "AR-BIALLELIC-PRIMARY-AMENORRHOEA -- "
            "AD-HETEROZYGOUS-PREMATURE-MENOPAUSE-30s -- "
            "DIZYGOTIC-TWINNING-ASSOCIATION-GDF9-POLYMORPHISMS)"
        ),
        "protein": (
            "GDF9 -- 5q31.1 AR/AD -- GDF9-454aa -- "
            "Growth-Differentiation-Factor-9-52kDa-TGFbeta-Oocyte-Secreted-Granulosa-Paracrine -- "
            "SMAD2-3-Pathway-Cumulus-Expansion-LH-Receptor-Induction -- "
            "OMIM-Gene-601918-Disease-POI14-618014"
        ),
        "locus": "5q31.1",
        "protein_size": "454 aa / ~52 kDa",
        "inheritance": (
            "Both AR (biallelic) and AD (heterozygous) inheritance described — unusual dual-mode. "
            "GDF9 is oocyte-specific, secreted from primary follicle stage onward. "
            "FUNCTION: GDF9 signals to granulosa cells via BMPRII + ALK5 → SMAD2/3 phosphorylation → "
            "promotes granulosa cell proliferation, cumulus expansion, LH receptor expression → "
            "oocyte competence and follicular growth. "
            "AR PHENOTYPE: Biallelic null alleles → primary amenorrhoea (no folliculogenesis beyond "
            "primary stage); very rare; consanguineous families. "
            "AD PHENOTYPE (more common): Heterozygous missense variants (especially in prodomain and "
            "mature domain) → haploinsufficiency → premature menopause (35-42 years typically). "
            "BMP15-GDF9 HETERODIMER (CUMULIN): GDF9 + BMP15 form a heterodimer ~10x more potent than "
            "either homodimer; heterozygous GDF9 LOF reduces cumulin → impaired cumulus expansion → "
            "reduced fertilisation rate in IVF. "
            "DIZYGOTIC TWINNING: GDF9 gain-of-function polymorphisms (e.g., rs254286) increase dizygotic "
            "twinning rate → maternal hyperovulation — opposite end of GDF9 activity spectrum to POI."
        ),
        "disease_category": (
            "GDF9-related POI (POI14); AR biallelic → severe primary amenorrhoea; "
            "AD heterozygous → premature menopause (variable penetrance); "
            "most GDF9 POI detected on expanded gene panels; prevalence underestimated; "
            "GDF9 + BMP15 should be tested together given cumulin interaction"
        ),
        "disease_pathway": (
            "GDF9-SMAD2/3 PATHWAY: "
            "GDF9 binds BMPRII (required) → recruits ALK5 (type I receptor) → SMAD2 + SMAD3 phosphorylated → "
            "SMAD4 co-activation → nucleus → transcription: "
            "- Cumulus expansion genes (HAS2, PTGS2, TNFAIP6) → cumulus matrix formation → "
            "oocyte-cumulus complex (COC) competence for fertilisation; "
            "- LH receptor (LHCGR) upregulation in granulosa → LH responsiveness; "
            "- Anti-apoptotic genes → granulosa survival. "
            "GDF9 LOF → reduced SMAD2/3 activity → cumulus expansion failure → fertilisation failure → "
            "accelerated atresia → POI. "
            "CUMULIN HETERODIMER: BMP15 (oocyte) + GDF9 (oocyte) form cumulin → binds BMPRII + ALK6 → "
            "activates SMAD1/5/8 AND SMAD2/3 simultaneously → cross-pathway amplification → "
            "this is the most potent oocyte signal to granulosa; loss of either BMP15 or GDF9 "
            "disrupts cumulin formation."
        ),
        "pathognomonic": (
            "GDF9 / OOCYTE TGF-β POI CLINICAL PEARLS: "
            "DUAL INHERITANCE: Biallelic → primary amenorrhoea (severe); heterozygous → premature menopause (mild-moderate). "
            "TEST BOTH: GDF9 AND BMP15 should always be sequenced together — cumulin interaction. "
            "IVF IMPLICATION: GDF9 LOF reduces cumulus expansion → lower fertilisation rate → "
            "ICSI preferred over conventional IVF in affected carriers. "
            "FERTILISATION RATE: Monitor post-retrieval; if poor cumulus expansion → ICSI all MII oocytes. "
            "DIZYGOTIC TWINNING: GDF9 gain-of-function → hyperovulation → twin pregnancies; "
            "not associated with POI (opposite spectrum). "
            "CONSANGUINITY: Biallelic GDF9 → primary amenorrhoea — suspect in consanguineous pedigrees "
            "with primary amenorrhoea and 46,XX karyotype. "
            "CUMULIN CLINICAL TARGET: recombinant cumulin (BMP15-GDF9 heterodimer) being investigated "
            "as a therapeutic to improve COC quality in IVF."
        ),
        "treatment": (
            "AR PRIMARY AMENORRHOEA: Oestrogen + progestogen induction of puberty; long-term HRT until 51; "
            "egg donation for fertility (own eggs non-functional in severe AR). "
            "AD PREMATURE MENOPAUSE: HRT transdermal oestradiol + progestogen until 51; "
            "IVF with ICSI (preferred over conventional IVF due to poor cumulus expansion); "
            "egg donation if AFC depleted. "
            "FERTILITY PRESERVATION: Oocyte cryopreservation before AFC falls in heterozygotes. "
            "GENETIC COUNSELLING: AR = 25% recurrence for siblings; AD = 50% daughters."
        ),
        "key_features": [
            "GDF9 has DUAL inheritance: AR biallelic → primary amenorrhoea; AD heterozygous → premature menopause",
            "Test BMP15 AND GDF9 together — both form cumulin heterodimer (10x more potent than homodimers)",
            "GDF9 signals via SMAD2/3 → cumulus expansion, LH receptor induction",
            "ICSI preferred over conventional IVF — GDF9 LOF reduces cumulus expansion",
            "GDF9 gain-of-function polymorphisms → dizygotic twinning (opposite end of spectrum)",
            "AR primary amenorrhoea in consanguineous families: test GDF9 + BMP15 + NOBOX + FIGLA panel",
            "HRT mandatory from POI diagnosis until age 51",
            "Recombinant cumulin (BMP15-GDF9 heterodimer) under investigation as IVF adjunct",
        ],
        "key_ddx": (
            "GDF9 vs BMP15: both oocyte TGF-β; BMP15 X-linked (no male transmission); GDF9 chromosome 5 (AR/AD); "
            "GDF9 AR vs Turner: 46,XX; no somatic features; "
            "GDF9 AD vs FMR1: FMR1 has CGG repeat expansion; neurological features in premutation carriers; "
            "GDF9 vs NOBOX: NOBOX is a TF (not a ligand); NOBOX more severe (primary amenorrhoea AR)"
        ),
        "systemic_involvement": (
            "REPRODUCTIVE: POI with reduced AFC; poor cumulus expansion in IVF; "
            "SKELETAL: osteoporosis risk if HRT withheld; "
            "NEUROLOGICAL: none; "
            "No somatic dysmorphic features; no extra-gonadal manifestations"
        ),
        "onset_age": "Primary amenorrhoea (AR biallelic) or premature menopause age 30-42 (AD heterozygous)",
        "surgical_urgency": "No surgical emergency; proactive fertility preservation counselling",
        "gene_family": "GDF9 — TGF-β superfamily (GDF subgroup); oocyte-secreted growth factor",
        "morphology": "Ovaries small with very low AFC; poor cumulus-oocyte complex quality in IVF",
        "n_patients": 40,
    },
    # -- NR5A1 -- SF-1 / Steroidogenic Factor 1 (46,XX POI + potential adrenal involvement) ----
    {
        "gene": "NR5A1",
        "alt_name": (
            "NR5A1 (NR5A1-461aa-9q33.3 / AD-dominant-negative -- "
            "SF1-STEROIDOGENIC-FACTOR-1-MASTER-ADRENAL-GONADAL-REGULATOR -- "
            "46XX-POI-PLUS-ADRENAL-INSUFFICIENCY-RISK -- "
            "46XY-NR5A1-LOF-46XY-DSD-FEMALE-AMBIGUOUS-EXTERNAL -- "
            "DOMINANT-NEGATIVE-MECHANISM-MOST-COMMON -- "
            "INHIBIN-B-LOW-BEFORE-FSH-RISES-EARLIEST-MARKER)"
        ),
        "protein": (
            "NR5A1 -- 9q33.3 AD -- NR5A1-461aa -- "
            "Steroidogenic-Factor-1-SF1-52kDa-Orphan-Nuclear-Receptor-Zinc-Finger-DBD-Ligand-Binding -- "
            "Master-Transcriptional-Regulator-CYP11A1-StAR-HSD3B2-CYP17A1-CYP19A1-STAR-Steroidogenesis -- "
            "OMIM-Gene-184757-Disease-POI7-612964"
        ),
        "locus": "9q33.3",
        "protein_size": "461 aa / ~52 kDa",
        "inheritance": (
            "Autosomal dominant with dominant-negative mechanism (most common). "
            "NR5A1 encodes Steroidogenic Factor 1 (SF-1), an orphan nuclear receptor essential for: "
            "(1) Adrenal cortex development; (2) Gonadal development and function; "
            "(3) Pituitary gonadotroph function; (4) Hypothalamic GnRH neuron maturation. "
            "MECHANISM: NR5A1 activates CYP11A1, StAR, HSD3B2, CYP17A1, CYP19A1 (aromatase) → "
            "controls the complete steroidogenesis cascade in both adrenals and gonads. "
            "46,XX FEMALES + NR5A1 LOF: ovarian insufficiency (POI type 7); adrenal function typically "
            "PRESERVED in isolated POI (different threshold for adrenal vs gonadal NR5A1 requirement). "
            "HOWEVER: Risk of subclinical adrenal insufficiency — cortisol response testing mandatory. "
            "46,XY + NR5A1 LOF: 46,XY DSD — Sertoli/Leydig cell dysgenesis → female or ambiguous "
            "external genitalia despite 46,XY karyotype — NR5A1 required for Leydig cell testosterone. "
            "POINT MUTATIONS: p.Arg92Trp is the most common mutation in 46,XY DSD; "
            "different mutations → 46,XX POI with adrenal risk or 46,XY DSD."
        ),
        "disease_category": (
            "NR5A1-related POI (POI7); AD dominant-negative; 46,XX females → POI ± subclinical adrenal insufficiency; "
            "46,XY males with NR5A1 LOF → 46,XY DSD (separate clinical entity); "
            "NR5A1 found in ~3-5% of women with unexplained POI on gene panels; "
            "earliest marker: low inhibin B before FSH rises"
        ),
        "disease_pathway": (
            "NR5A1 STEROIDOGENESIS REGULATION: "
            "NR5A1 binds SF-1 response elements (AGGTCA half-sites) → activates: "
            "StAR (cholesterol import into mitochondria → rate-limiting step), "
            "CYP11A1 (side-chain cleavage: cholesterol → pregnenolone), "
            "HSD3B2 (pregnenolone → progesterone), "
            "CYP17A1 (17-hydroxylase: progesterone → 17OH-progesterone → androstenedione), "
            "CYP19A1/aromatase (androstenedione → oestrone; testosterone → oestradiol). "
            "NR5A1 LOF → reduced aromatase → low oestradiol despite rising FSH (compensatory). "
            "GRANULOSA CELL FUNCTION: NR5A1 required for FSH receptor expression + granulosa survival; "
            "LOF → accelerated follicular atresia. "
            "ADRENAL THRESHOLD: Adrenal NR5A1 requirement is higher (more copies/activity needed) → "
            "NR5A1 heterozygous often has intact adrenal but stressed reserve → "
            "cortisol stimulation test (Synacthen) mandatory — watch for adrenal crisis at illness."
        ),
        "pathognomonic": (
            "NR5A1 / SF-1 POI CLINICAL PEARLS: "
            "ADRENAL RISK: ALL 46,XX women with NR5A1 POI must have Synacthen stimulation test — "
            "even if basal cortisol normal; adrenal crisis possible with major illness/surgery. "
            "CARRY HYDROCORTISONE: If Synacthen test borderline or abnormal → sick-day rule + "
            "emergency hydrocortisone 100 mg IM kit. "
            "INHIBIN B: Falls BEFORE FSH rises in NR5A1 POI — earliest marker of ovarian reserve decline; "
            "monitor annually from puberty in NR5A1 carriers. "
            "46,XY DSD: NR5A1 LOF in 46,XY → DSD workup (karyotype, HCG stimulation test, gonads); "
            "gonads may be dysgenetic — gonadectomy if gonadoblastoma risk (discuss with MDT). "
            "TESTOSTERONE: Low testosterone in 46,XX NR5A1 POI — NR5A1 also regulates adrenal androgen. "
            "GLUCOCORTICOID INTERACTIONS: NR5A1 variants interact with dexamethasone sensitivity (rare). "
            "GENE PANEL: NR5A1 must be included in all POI panels AND in all 46,XY DSD panels — "
            "same gene, very different phenotypes by sex chromosome."
        ),
        "treatment": (
            "HRT: Transdermal oestradiol + progestogen from POI diagnosis until age 51; "
            "DHEA supplementation may help low androgen symptoms (libido, fatigue). "
            "ADRENAL: Synacthen test at diagnosis; if borderline → hydrocortisone replacement + sick-day rule; "
            "annual morning cortisol monitoring. "
            "FERTILITY: IVF if AFC ≥4-5; egg donation if reserves exhausted; ICSI preferred. "
            "46,XY DSD: Karyotype all NR5A1 patients; if 46,XY → multidisciplinary DSD team; "
            "gonadectomy decision depends on phenotype and gonadoblastoma risk. "
            "BONE: DEXA at diagnosis; calcium + vitamin D; bisphosphonates if T-score <-2.5."
        ),
        "key_features": [
            "NR5A1/SF-1: master regulator of adrenal + gonadal + pituitary steroidogenesis",
            "46,XX NR5A1 LOF → POI ± subclinical adrenal insufficiency (Synacthen test MANDATORY)",
            "46,XY NR5A1 LOF → 46,XY DSD (female/ambiguous external genitalia) — same gene, different sex",
            "ADRENAL CRISIS RISK: give emergency hydrocortisone kit if Synacthen borderline",
            "Inhibin B falls BEFORE FSH rises — earliest marker of NR5A1 ovarian reserve decline",
            "NR5A1 activates StAR, CYP11A1, HSD3B2, CYP17A1, CYP19A1 (full steroidogenesis cascade)",
            "HRT mandatory; DHEA for androgen deficiency symptoms",
            "NR5A1 must be on ALL POI panels AND all 46,XY DSD panels",
        ],
        "key_ddx": (
            "NR5A1 vs Addison (autoimmune): NR5A1 POI has no anti-21-hydroxylase antibodies; "
            "NR5A1 vs CAH: NR5A1 reduces all steroids; CAH shunts (elevated 17OHP in CYP21A2); "
            "NR5A1 46,XX vs Turner: 46,XX karyotype; no somatic features of Turner; "
            "NR5A1 46,XY DSD vs complete androgen insensitivity: AR receptor testing; "
            "different androgen/oestrogen profile"
        ),
        "systemic_involvement": (
            "ADRENAL: Subclinical AI risk (Synacthen test); adrenal crisis risk with illness/surgery; "
            "REPRODUCTIVE: POI with reduced AFC; low inhibin B; low oestradiol; "
            "ANDROGEN: Low DHEA-S, testosterone in 46,XX NR5A1; "
            "46,XY: Leydig/Sertoli dysgenesis → 46,XY DSD; "
            "PITUITARY: NR5A1 expressed in gonadotrophs — rarely pituitary dysfunction"
        ),
        "onset_age": "POI typically 20-38 years; primary amenorrhoea if severe; adrenal crisis any age",
        "surgical_urgency": "URGENT: Synacthen test at diagnosis; emergency hydrocortisone kit if borderline",
        "gene_family": "NR5A1 — orphan nuclear receptor; steroidogenic factor family (SF-1/LRH-1)",
        "morphology": "Ovaries small with low AFC; adrenals typically normal on CT unless major LOF",
        "n_patients": 40,
    },
    # -- NOBOX -- Newborn Ovary Homeobox (AR POI / primary amenorrhoea) --------------------------
    {
        "gene": "NOBOX",
        "alt_name": (
            "NOBOX (NOBOX-672aa-7q35 / AR -- "
            "NEWBORN-OVARY-HOMEOBOX-TRANSCRIPTION-FACTOR -- "
            "PRIMORDIAL-TO-PRIMARY-FOLLICLE-TRANSITION-BLOCK -- "
            "PRIMARY-AMENORRHOEA-AR-BIALLELIC -- "
            "MOST-COMMON-IDENTIFIED-AR-POI-GENE-NON-CONSANGUINEOUS -- "
            "OOCYTE-HOMEOBOX-TF-REGULATES-GDF9-BMP15-FIGLA)"
        ),
        "protein": (
            "NOBOX -- 7q35 AR -- NOBOX-672aa -- "
            "Newborn-Ovary-Homeobox-75kDa-Paired-Like-Homeobox-Domain-Oocyte-Nucleus -- "
            "Transcriptional-Activator-GDF9-BMP15-ZP2-ZP3-Zona-Pellucida-Genes -- "
            "OMIM-Gene-610934-Disease-POI5-611548"
        ),
        "locus": "7q35",
        "protein_size": "672 aa / ~75 kDa",
        "inheritance": (
            "Autosomal recessive. NOBOX is an oocyte-specific transcription factor with a paired-like "
            "homeodomain. FUNCTION: NOBOX regulates the critical transition from primordial follicle "
            "(quiescent) to primary follicle (growth-activated) — without this transition, follicles "
            "cannot enter the growing pool and are eventually atretic. "
            "NOBOX TARGETS: GDF9, BMP15, ZP1, ZP2, ZP3, POU5F1/OCT4, FIGLA — NOBOX activates the entire "
            "oocyte-specific transcriptional programme. "
            "BIALLELIC LOF → Follicles arrested at primordial stage → no antral follicles → "
            "primary amenorrhoea (no oestrogen production, no puberty without HRT). "
            "FREQUENCY: One of the most frequently identified AR POI genes in non-consanguineous European "
            "populations (~5-7% of AR POI). "
            "HETEROZYGOUS: May confer slightly earlier menopause but NOT clinical POI in most carriers — "
            "variable penetrance (unlike BMP15/GDF9). "
            "Murine data: Nobox knockout female mice are fertile at birth but develop rapid follicle "
            "depletion by day 21 → no progeny beyond first litter — models human AR POI."
        ),
        "disease_category": (
            "NOBOX-related POI (POI5); AR biallelic → primary amenorrhoea or very early POI (<20y); "
            "one of the most common AR POI genes identified on gene panels; "
            "most patients present with primary amenorrhoea and elevated FSH at age 12-16y; "
            "no somatic dysmorphic features — isolated gonadal failure"
        ),
        "disease_pathway": (
            "NOBOX TRANSCRIPTIONAL NETWORK: "
            "NOBOX binds NOBOX-binding element (NBE: TAATCC) in promoters of oocyte-specific genes → "
            "activates: GDF9, BMP15 (oocyte paracrine factors), ZP1/ZP2/ZP3 (zona pellucida proteins), "
            "FIGLA (primordial follicle assembly TF), POU5F1/OCT4 (pluripotency). "
            "PRIMORDIAL FOLLICLE ACTIVATION: Primordial follicles require paracrine signals to enter "
            "growth activation; NOBOX LOF → reduced GDF9/BMP15 secretion → granulosa cells fail to "
            "receive oocyte paracrine signals → follicles remain quiescent → accelerated atresia → "
            "no antral follicle development → no oestrogen → no puberty. "
            "DOWNSTREAM CONSEQUENCES: Without oestrogen production → "
            "very high FSH (>100 IU/L in severe cases), very high LH, very low oestradiol (<30 pmol/L) → "
            "hypergonadotropic hypogonadism (OPPOSITE to HH: FSH ELEVATED, not low). "
            "KARYOTYPE: 46,XX (normal); no chromosomal abnormality."
        ),
        "pathognomonic": (
            "NOBOX / AR POI CLINICAL PEARLS: "
            "PRESENTATION: Primary amenorrhoea at age 12-16y + very high FSH (>40-100 IU/L) + "
            "very low oestradiol + small ovaries on USS + 46,XX karyotype → AR POI panel. "
            "HYPERGONADOTROPIC: FSH ELEVATED (distinguish from HH where FSH is LOW); "
            "DIAGNOSIS DDx: Turner syndrome (45,X or mosaic) vs 46,XX AR POI — karyotype is critical. "
            "NOBOX FOUNDER VARIANTS: p.Arg355His and p.Gln197* in European populations; "
            "p.Arg303Cys in East Asian populations — specific panels catch these. "
            "PUBERTY INDUCTION: Start low-dose oestrogen at age 11-12y if primary amenorrhoea; "
            "escalate slowly over 2-3 years — simulate natural puberty timing to maximise bone acquisition. "
            "FERTILITY: Essentially no own eggs — egg donation is the route to pregnancy. "
            "BONE: Very high osteoporosis risk due to absent pubertal oestrogen; "
            "DEXA at diagnosis; aggressive HRT until age 51 (higher dose than typical POI). "
            "UTERUS: Absent oestrogen → small/juvenile uterus; HRT promotes uterine growth before egg donation attempt."
        ),
        "treatment": (
            "PUBERTY INDUCTION: Low-dose transdermal oestradiol (6.25-12.5 mcg/day) from age 11-12y → "
            "escalate over 2-3 years to 100 mcg/day → add cyclical progestogen after 2 years or on "
            "first breakthrough bleeding. "
            "HRT LONG-TERM: Full-dose transdermal oestradiol + progestogen until age 51; "
            "monitor bone density annually. "
            "FERTILITY: Egg donation (own oocytes absent); uterine development with HRT before transfer; "
            "progesterone luteal support for egg donation cycle. "
            "BONE: Calcium 1200 mg/day + vitamin D 1000 IU/day; DEXA at diagnosis and every 2 years. "
            "GENETIC COUNSELLING: 25% sibling recurrence; prenatal testing available."
        ),
        "key_features": [
            "NOBOX: oocyte-specific homeobox TF regulating GDF9, BMP15, ZP proteins, FIGLA, OCT4",
            "Primordial-to-primary follicle transition BLOCKED → primary amenorrhoea",
            "AR biallelic → primary amenorrhoea + hypergonadotropic hypogonadism (FSH >40 IU/L)",
            "Most common identifiable AR POI gene in non-consanguineous European populations",
            "No somatic features: isolated gonadal failure with 46,XX karyotype",
            "Puberty induction mandatory from age 11-12y — start LOW dose, escalate slowly",
            "No own eggs: egg donation is the fertility route",
            "Very high osteoporosis risk — aggressive HRT and DEXA from diagnosis",
        ],
        "key_ddx": (
            "NOBOX vs Turner: karyotype distinguishes 46,XX (NOBOX) from 45,X or mosaic (Turner); "
            "NOBOX vs FIGLA: both AR oocyte TFs; FIGLA slightly more severe (earlier arrest); "
            "NOBOX vs autoimmune POI: no anti-ovarian antibodies; no thyroid/adrenal autoimmunity; "
            "NOBOX AR vs FMR1: FMR1 premutation (CGG repeat expansion); no Fragile X family history in NOBOX"
        ),
        "systemic_involvement": (
            "REPRODUCTIVE: Primary amenorrhoea; hypergonadotropic hypogonadism; absent puberty; "
            "SKELETAL: Severe osteoporosis risk if HRT not started at puberty-appropriate age; "
            "CARDIOVASCULAR: Premature atherosclerosis risk without long-term HRT; "
            "No other systemic involvement; no neurological features"
        ),
        "onset_age": "Primary amenorrhoea at expected puberty (age 12-16y); hypergonadotropic state",
        "surgical_urgency": "No surgical emergency; urgent puberty induction and bone protection",
        "gene_family": "NOBOX — paired-like homeodomain transcription factor; oocyte-specific",
        "morphology": "Ovaries small/streak on USS; no antral follicles; AFC = 0; uterus hypoplastic",
        "n_patients": 40,
    },
    # -- FIGLA -- Factor In the Germline Alpha (AR / primordial follicle assembly) ---------------
    {
        "gene": "FIGLA",
        "alt_name": (
            "FIGLA (FIGLA-115aa-2p13.3 / AR -- "
            "FACTOR-IN-GERMLINE-ALPHA-BHLH-TF -- "
            "PRIMORDIAL-FOLLICLE-ASSEMBLY-MASTER-REGULATOR -- "
            "ZP1-ZP3-ZP4-ZONA-PELLUCIDA-TRANSCRIPTION-FACTOR -- "
            "COMPLETE-PRIMARY-AMENORRHOEA-AR-BIALLELIC -- "
            "RAREST-IDENTIFIABLE-CAUSE-AR-POI)"
        ),
        "protein": (
            "FIGLA -- 2p13.3 AR -- FIGLA-115aa -- "
            "Factor-In-Germline-Alpha-13kDa-Basic-HLH-Transcription-Factor-Oocyte-Specific -- "
            "Dimerises-E-Proteins-TCFE2A-Activates-ZP1-ZP3-ZP4-Zona-Pellucida-Assembly -- "
            "OMIM-Gene-608697-Disease-POI6-612310"
        ),
        "locus": "2p13.3",
        "protein_size": "115 aa / ~13 kDa",
        "inheritance": (
            "Autosomal recessive. FIGLA is among the smallest known transcription factors (115 aa). "
            "It is a basic helix-loop-helix (bHLH) transcription factor expressed exclusively in oocytes. "
            "FUNCTION: FIGLA is required for primordial follicle assembly — the process by which naked "
            "oocytes in the neonatal ovary are individually enclosed by somatic (pre-granulosa) cells "
            "to form primordial follicles, establishing the ovarian reserve. "
            "WITHOUT FIGLA: Zona pellucida proteins ZP1, ZP3, ZP4 are not transcribed → zona pellucida "
            "absent → oocytes cannot recruit pre-granulosa cells → primordial follicle assembly fails → "
            "no ovarian reserve established → primary amenorrhoea. "
            "MOUSE MODEL: Figla null females develop oogonia normally (oocytes made) but primordial "
            "follicle assembly fails completely → no follicles → sterile females from birth; "
            "males normal. "
            "HUMAN: Very rare; biallelic LOF mutations → complete primary amenorrhoea with streak gonads; "
            "heterozygous: likely fertile (haploinsufficiency insufficient to cause clinical POI in most)."
        ),
        "disease_category": (
            "FIGLA-related POI (POI6); AR biallelic → complete primary amenorrhoea with streak gonads; "
            "very rare — only a few dozen cases reported in literature; "
            "diagnosis by gene panel in workup of unexplained primary amenorrhoea + 46,XX + streak gonads; "
            "no somatic features (unlike Turner); no other organ involvement"
        ),
        "disease_pathway": (
            "FIGLA PRIMORDIAL FOLLICLE ASSEMBLY: "
            "Neonatal ovary (birth to postnatal week 2-3 in humans): oocytes are in meiotic arrest as "
            "dictyotene (prophase I) oocytes, grouped in cysts derived from oogonial synchronous divisions. "
            "FIGLA expressed in oocytes → dimerises with E-proteins (TCF3/E12, TCF4/E2-2) → "
            "activates ZP1, ZP3, ZP4 promoters → zona pellucida protein synthesis → "
            "zona pellucida forms around individual oocytes → zona pellucida signals pre-granulosa "
            "cell recruitment → cyst breakdown + individual primordial follicle enclosure. "
            "FIGLA LOF → NO zona pellucida → NO cyst breakdown signal → NO primordial follicle formation → "
            "naked oocytes die in clusters → NO ovarian reserve established → "
            "streak gonads at puberty → hypergonadotropic hypogonadism. "
            "ALSO: FIGLA activates sperm receptor ZP3 — relevant for fertilisation (moot if no follicles) "
            "and anti-ZP antibody-mediated POI (iatrogenic ZP3 autoimmunity — rare)."
        ),
        "pathognomonic": (
            "FIGLA / PRIMORDIAL FOLLICLE ASSEMBLY FAILURE PEARLS: "
            "PRESENTATION: Primary amenorrhoea + streak gonads + 46,XX + no follicles on USS → "
            "gene panel (FIGLA, NOBOX, FOXL2, BMP15, MCM8). "
            "STREAK GONADS: Unlike NOBOX (small ovaries), FIGLA may cause complete streak gonads "
            "(no follicular structure at all) — most severe POI phenotype at birth. "
            "ZONA PELLUCIDA: FIGLA directly controls ZP1/ZP3/ZP4 — absent zona pellucida cannot be "
            "detected clinically (no clinical test for zona pellucida absence). "
            "RARE DIAGNOSIS: Very rare — even centres with high POI gene panel testing see <1-2 FIGLA "
            "cases per year; required for complete AR POI panel. "
            "MALE FERTILITY: Males with biallelic FIGLA LOF — unaffected (zona pellucida genes not "
            "expressed in male germline; testicular FIGLA expression absent). "
            "GONADOBLASTOMA: Streak gonads in 46,XX (unlike 46,XY streak gonads) have LOW gonadoblastoma "
            "risk — do NOT routinely remove 46,XX streak gonads (unlike 46,XY). "
            "HRT: Puberty induction mandatory; long-term HRT."
        ),
        "treatment": (
            "PUBERTY INDUCTION: Transdermal oestradiol (low-dose escalation from age 11-12y) → "
            "add progestogen after 2 years; simulate natural puberty. "
            "HRT LONG-TERM: Full-dose HRT until age 51; aggressive bone protection. "
            "FERTILITY: Egg donation only (own oocytes absent from birth in severe FIGLA LOF). "
            "BONE: High-dose calcium + vitamin D; DEXA at diagnosis; bisphosphonates if T-score <-2.5. "
            "PSYCHOLOGICAL: Infertility disclosure in adolescence — MDT with psychologist. "
            "GENETIC COUNSELLING: 25% sibling recurrence."
        ),
        "key_features": [
            "FIGLA: smallest oocyte TF (115 aa, 13 kDa); bHLH; activates ZP1/ZP3/ZP4 zona pellucida genes",
            "Primordial follicle ASSEMBLY fails — no ovarian reserve established at birth (most severe AR POI)",
            "Streak gonads + primary amenorrhoea + 46,XX + very high FSH",
            "Very rare: handful of cases reported; include on ALL AR POI panels",
            "46,XX streak gonads have LOW gonadoblastoma risk — do NOT routinely remove",
            "Males with biallelic FIGLA LOF are unaffected and fertile",
            "Puberty induction from age 11-12y; egg donation for fertility",
            "No somatic features; isolated gonadal failure",
        ],
        "key_ddx": (
            "FIGLA vs NOBOX: FIGLA more severe (streak gonads from birth); NOBOX may have some follicle remnants; "
            "FIGLA vs Turner: karyotype 46,XX; no somatic features; "
            "FIGLA vs MCM8: MCM8 is a DNA repair gene (meiotic arrest rather than assembly failure); "
            "FIGLA vs autoimmune POI: no anti-ZP antibodies consistently; no thyroid autoimmunity"
        ),
        "systemic_involvement": (
            "REPRODUCTIVE: Streak gonads; primary amenorrhoea; hypergonadotropic hypogonadism; "
            "SKELETAL: Severe osteoporosis without HRT from puberty; "
            "CARDIOVASCULAR: Premature risk without oestrogen; "
            "No other systemic involvement; no neurological or somatic dysmorphic features"
        ),
        "onset_age": "Primary amenorrhoea at expected puberty; streak gonads present from birth (detected at puberty)",
        "surgical_urgency": "No surgical emergency in 46,XX; urgent puberty induction",
        "gene_family": "FIGLA — basic HLH transcription factor; oocyte-specific primordial follicle regulator",
        "morphology": "Streak gonads on USS (no follicular structure); uterus hypoplastic",
        "n_patients": 40,
    },
    # -- MCM8 -- Minichromosome Maintenance 8 (AR DNA repair POI) --------------------------------
    {
        "gene": "MCM8",
        "alt_name": (
            "MCM8 (MCM8-840aa-20p12.3 / AR -- "
            "MINICHROMOSOME-MAINTENANCE-8-HELICASE -- "
            "MCM8-MCM9-DNA-REPAIR-COMPLEX-MEIOTIC-RECOMBINATION -- "
            "OOCYTE-DNA-REPAIR-FAILURE-MEIOTIC-ARREST -- "
            "Lynch-SYNDROME-OVERLAP-CHROMOSOMAL-INSTABILITY -- "
            "COHORT-FOUNDER-VARIANTS-JEWISH-POPULATIONS)"
        ),
        "protein": (
            "MCM8 -- 20p12.3 AR -- MCM8-840aa -- "
            "Minichromosome-Maintenance-8-Helicase-93kDa-AAA-ATPase-Hexameric-Ring-DNA-Unwinding -- "
            "MCM8-MCM9-Complex-Meiotic-DSB-Repair-Homologous-Recombination-Correction -- "
            "OMIM-Gene-608187-Disease-POI10-612885"
        ),
        "locus": "20p12.3",
        "protein_size": "840 aa / ~93 kDa",
        "inheritance": (
            "Autosomal recessive. MCM8 is an AAA+ ATPase DNA helicase. "
            "The MCM8-MCM9 complex (hexameric helicase ring) functions in: "
            "(1) Resolution of stalled replication forks in somatic cells; "
            "(2) Homologous recombination repair of double-strand breaks (DSBs) during meiosis. "
            "IN OOCYTES: Meiosis I requires programmed DSB formation (SPO11) + homologous recombination "
            "repair; MCM8-MCM9 complex unwinds DNA at DSB sites to allow RAD51/BRCA2-mediated strand "
            "invasion and repair. MCM8 LOF → DSBs not repaired → oocyte checkpoint activation → "
            "oocyte death → follicular atresia → POI. "
            "CHROMOSOMAL INSTABILITY: MCM8 also functions in mitotic DNA repair → MCM8 biallelic LOF → "
            "somatic chromosomal instability → potential CANCER RISK. "
            "LYNCH OVERLAP: MCM8 LOF associated with mismatch repair deficiency features in some reports "
            "— endometrial cancer risk surveillance recommended. "
            "FOUNDER VARIANTS: p.Lys325Glu (Ashkenazi Jewish founder) and p.Arg215Trp (Turkish) most common."
        ),
        "disease_category": (
            "MCM8-related POI (POI10); AR biallelic; oocyte meiotic DNA repair failure → POI; "
            "chromosomal instability → cancer risk (endometrial; colorectal surveillance); "
            "Ashkenazi Jewish founder variant p.Lys325Glu; "
            "clinical presentation: primary or secondary amenorrhoea depending on residual MCM8 function; "
            "distinguished by DNA repair phenotype (sensitivity to genotoxic agents)"
        ),
        "disease_pathway": (
            "MCM8-MCM9 MEIOTIC DNA REPAIR: "
            "SPO11 creates programmed DSBs in meiotic leptotene oocytes (~200 DSBs per oocyte) → "
            "DSBs signal homologous chromosome pairing → RAD51/DMC1 load onto 3' overhangs → "
            "strand invasion of homologous chromosome → MCM8-MCM9 helicase unwinds recipient DNA → "
            "D-loop extension → crossover formation → meiotic recombination complete → chiasmata → "
            "chromosome segregation. "
            "MCM8 LOF → DSBs not resolved by homologous recombination → ATM/CHK2 checkpoint activated → "
            "oocyte apoptosis via PUMA/BIM → massive oocyte loss at meiotic prophase I → "
            "severely reduced ovarian reserve. "
            "WHEN POI PRESENTS: If severe LOF → primary amenorrhoea; if partial LOF → some oocytes "
            "survive meiosis → secondary amenorrhoea in 20s-30s. "
            "SOMATIC INSTABILITY: MCM8-MCM9 also resolves replication stress at common fragile sites → "
            "LOF → somatic chromosomal rearrangements → cancer predisposition. "
            "GENOTOXIC SENSITIVITY: Cells with MCM8 LOF are hypersensitive to mitomycin C, cisplatin, "
            "UV — relevant if cancer therapy needed (avoid DNA crosslinkers)."
        ),
        "pathognomonic": (
            "MCM8 / DNA REPAIR POI CLINICAL PEARLS: "
            "CANCER SURVEILLANCE: All MCM8 biallelic patients → annual colonoscopy + endometrial sampling "
            "from age 30y (Lynch-like instability); gynae oncology co-management. "
            "GENOTOXIC AVOIDANCE: If cancer diagnosed → avoid cisplatin/mitomycin C crosslinkers; "
            "select alternative chemotherapy regimens; haematology involvement. "
            "FOUNDER TESTING: In Ashkenazi Jewish women with POI → include MCM8 p.Lys325Glu on panel. "
            "CHROMOSOMAL INSTABILITY SCREENING: Karyotype at diagnosis (micronuclei, breaks); "
            "consider chromosomal fragility assay. "
            "DNA REPAIR GENES IN POI: MCM8 is one of several meiotic DNA repair genes causing AR POI — "
            "panel should include: MCM8, MCM9, BRCA2 (Fanconi), FANCA, PALB2, ERCC4. "
            "CHEMOTHERAPY HISTORY: POI after cancer treatment → exclude treatment-induced POI before "
            "attributing to MCM8 (MCM8 carriers may have exaggerated ovarian toxicity from chemotherapy). "
            "PREGNANCY: Own eggs extremely unlikely if primary amenorrhoea; egg donation; "
            "prenatal MCM8 testing if partner is carrier (AR disease)."
        ),
        "treatment": (
            "HRT: Transdermal oestradiol + progestogen from POI diagnosis until age 51; "
            "bone DEXA at diagnosis; aggressive calcium + vitamin D. "
            "CANCER SURVEILLANCE: Annual colonoscopy from age 30y; endometrial USS + sampling from 30y; "
            "CA-125 + transvaginal USS annually; gynae oncology follow-up. "
            "FERTILITY: Own eggs feasible only if partial LOF (some AFC present); egg donation otherwise; "
            "ICSI if IVF attempted. "
            "GENOTOXIC DRUGS: Avoid platinum-based chemotherapy + mitomycin C; use alternative regimens. "
            "GENETIC COUNSELLING: 25% sibling recurrence; MCM9 testing in family (paralog, similar POI); "
            "carrier testing in Ashkenazi Jewish population (p.Lys325Glu)."
        ),
        "key_features": [
            "MCM8-MCM9 helicase complex: resolves DSBs during meiotic homologous recombination in oocytes",
            "Oocyte meiotic checkpoint failure → oocyte apoptosis → POI",
            "CANCER RISK: chromosomal instability → Lynch-like endometrial/colorectal risk (annual surveillance)",
            "GENOTOXIC SENSITIVITY: avoid cisplatin/mitomycin C crosslinkers if cancer treatment needed",
            "Ashkenazi Jewish founder: p.Lys325Glu (test specifically in AJ population)",
            "DNA repair POI panel: MCM8, MCM9, BRCA2, FANCA, PALB2, ERCC4 should be co-tested",
            "Primary or secondary amenorrhoea depending on residual MCM8 helicase activity",
            "HRT mandatory; cancer surveillance mandatory from age 30y in biallelic patients",
        ],
        "key_ddx": (
            "MCM8 vs other AR POI: MCM8 has cancer surveillance requirement (others do not); "
            "MCM8 vs BRCA2 (Fanconi/POI): BRCA2 also causes POI + cancer risk but via different repair pathway; "
            "MCM8 vs chemotherapy-induced POI: MCM8 genotype worsens chemo-induced POI risk; "
            "MCM8 vs MCM9: MCM9 (6q22.31) — paralog; causes similar AR POI; test together"
        ),
        "systemic_involvement": (
            "REPRODUCTIVE: POI (primary or secondary); oocyte loss from meiotic arrest; "
            "ONCOLOGICAL: Endometrial cancer risk; colorectal cancer risk (Lynch-like instability); "
            "HAEMATOLOGICAL: Chromosomal fragility on mitomycin C challenge; "
            "NEUROLOGICAL: None specific; "
            "SKELETAL: Osteoporosis from hypoestrogenism if HRT not given"
        ),
        "onset_age": "Primary amenorrhoea (severe LOF) or secondary amenorrhoea in 20s-30s (partial LOF)",
        "surgical_urgency": "No immediate surgical emergency; urgent cancer surveillance programme from age 30y",
        "gene_family": "MCM8 — AAA+ ATPase DNA helicase; MCM (minichromosome maintenance) family",
        "morphology": "Ovaries small to streak; very low AFC; normal karyotype 46,XX",
        "n_patients": 40,
    },
]


# ── Patient simulation ─────────────────────────────────────────────────────────────────────────────

def _make_patients(gene_entry: dict) -> list[dict]:
    rng = random.Random(SEED_BASE + POI_GENES.index(gene_entry))
    gene = gene_entry["gene"]
    patients = []
    for i in range(gene_entry["n_patients"]):
        age = rng.randint(15, 52)

        # All POI patients are female (46,XX) — except NR5A1 where some may be 46,XY DSD
        if gene == "NR5A1" and i < 5:
            sex = "M-46XY-DSD"  # 5 of 40 are 46,XY DSD presentation
        else:
            sex = "F-46XX"

        # Phenotype / POI severity using hearing_severity field (repurposed for POI type)
        if gene == "FMR1":
            if i < 12:
                severity = "FXPOI + secondary amenorrhoea age 28-35"
            elif i < 22:
                severity = "FXPOI + premature menopause age 35-42"
            elif i < 32:
                severity = "FXPOI + subfertility (IVF failure) age 30-38"
            else:
                severity = rng.choice([
                    "FXPOI + FXTAS tremor-ataxia co-occurrence (age >50)",
                    "Premutation carrier + POI + family Fragile X male",
                    "FXPOI + autoimmune thyroiditis co-association",
                ])
        elif gene == "FOXL2":
            if i < 20:
                severity = "BPES type I + POI (eyelid triad + secondary amenorrhoea)"
            elif i < 30:
                severity = "BPES type I + primary amenorrhoea (severe FOXL2 LOF)"
            else:
                severity = rng.choice([
                    "BPES type I + granulosa cell tumour surveillance (no somatic mutation)",
                    "BPES type II (eyelid anomaly only — no POI) — carrier family member",
                    "BPES type I + POI + amblyopia (ptosis repair delayed)",
                ])
        elif gene == "BMP15":
            if i < 15:
                severity = "BMP15 XLD + POI age 25-32 (heterozygous)"
            elif i < 28:
                severity = "BMP15 XLD + IVF OHSS risk + poor reserve"
            elif i < 36:
                severity = "BMP15 homozygous + severe POI primary amenorrhoea (rare)"
            else:
                severity = rng.choice([
                    "BMP15 carrier + subfertility + low AMH",
                    "BMP15 + co-occurring GDF9 variant (cumulin disruption)",
                ])
        elif gene == "GDF9":
            if i < 10:
                severity = "GDF9 AR biallelic + primary amenorrhoea (severe)"
            elif i < 30:
                severity = "GDF9 AD heterozygous + premature menopause age 32-42"
            else:
                severity = rng.choice([
                    "GDF9 AD + IVF poor fertilisation (cumulus expansion failure)",
                    "GDF9 AD + subfertility + low AMH age 28",
                    "GDF9 + dizygotic twinning family history (gain-of-function polymorphism carrier)",
                ])
        elif gene == "NR5A1":
            if i < 5:
                severity = "NR5A1 46,XY DSD (female external genitalia, bilateral streak gonads)"
            elif i < 20:
                severity = "NR5A1 46,XX POI + subclinical adrenal insufficiency (Synacthen borderline)"
            elif i < 32:
                severity = "NR5A1 46,XX POI + normal adrenal function (adrenal threshold preserved)"
            else:
                severity = rng.choice([
                    "NR5A1 46,XX POI + adrenal crisis at surgery (unrecognised AI)",
                    "NR5A1 46,XX + low inhibin B age 22 (before FSH rise)",
                    "NR5A1 46,XX + DHEA-S very low + androgen deficiency symptoms",
                ])
        elif gene == "NOBOX":
            if i < 25:
                severity = "NOBOX AR biallelic + primary amenorrhoea age 13-16"
            elif i < 36:
                severity = "NOBOX AR + streak ovaries + very high FSH (>80 IU/L)"
            else:
                severity = rng.choice([
                    "NOBOX AR + severe osteoporosis at diagnosis (HRT not given)",
                    "NOBOX AR + juvenile uterus (puberty induction delayed)",
                    "NOBOX AR + egg donation pregnancy success",
                ])
        elif gene == "FIGLA":
            if i < 28:
                severity = "FIGLA AR biallelic + streak gonads + primary amenorrhoea"
            elif i < 36:
                severity = "FIGLA AR + no follicles on USS + FSH >100 IU/L"
            else:
                severity = rng.choice([
                    "FIGLA AR + zona pellucida absent (confirmed sperm-ZP binding assay — research)",
                    "FIGLA AR + egg donation pregnancy (uterus primed with HRT)",
                    "FIGLA AR + severe osteoporosis (delayed diagnosis)",
                ])
        elif gene == "MCM8":
            if i < 15:
                severity = "MCM8 AR biallelic + primary amenorrhoea + chromosomal instability"
            elif i < 28:
                severity = "MCM8 AR + secondary amenorrhoea age 22-30 + cancer surveillance initiated"
            elif i < 36:
                severity = "MCM8 AR + endometrial sampling initiated age 30 (Lynch-like)"
            else:
                severity = rng.choice([
                    "MCM8 AR + Ashkenazi founder p.Lys325Glu + POI age 25",
                    "MCM8 AR + mitomycin C sensitivity confirmed (cisplatin avoided)",
                    "MCM8 AR + colorectal polyp detected age 32 (colonoscopy surveillance)",
                ])

        # Management
        if gene == "FMR1":
            mgmt_choices = [
                "HRT transdermal oestradiol + progestogen + fertility counselling",
                "IVF own eggs (AFC ≥5) + ICSI",
                "Egg donation (AFC depleted)",
                "Oocyte cryopreservation (premutation carrier <30y)",
                "Fertility preservation + FXTAS surveillance",
            ]
        elif gene == "FOXL2":
            mgmt_choices = [
                "Ptosis repair age 3-4y + blepharophimosis repair age 5-7y + HRT",
                "HRT + annual AFC/FSH/AMH + oocyte cryopreservation",
                "Egg donation (AFC depleted)",
                "Amblyopia patching + ptosis repair + HRT",
                "Granulosa cell tumour surveillance (USS annually)",
            ]
        elif gene == "BMP15":
            mgmt_choices = [
                "HRT + low-dose FSH IVF (OHSS prevention) + antagonist protocol",
                "Egg donation (AFC depleted)",
                "Oocyte cryopreservation (AFC 4-6 remaining)",
                "GnRH agonist trigger (avoid hCG OHSS risk)",
                "HRT + genetic counselling (50% daughters at risk)",
            ]
        elif gene == "GDF9":
            mgmt_choices = [
                "HRT + IVF-ICSI (preferred over conventional IVF — poor cumulus expansion)",
                "Egg donation (biallelic AR primary amenorrhoea)",
                "Oocyte cryopreservation + ICSI",
                "HRT transdermal + progestogen + bone DEXA",
                "Genetic counselling (AR 25% recurrence vs AD 50%)",
            ]
        elif gene == "NR5A1":
            mgmt_choices = [
                "HRT + Synacthen test + hydrocortisone sick-day rule",
                "Adrenal crisis management + HRT + DHEA",
                "IVF-ICSI + inhibin B annual monitoring",
                "Egg donation + 46,XY DSD MDT management",
                "HRT + NR5A1 gene panel + cascade testing",
            ]
        elif gene == "NOBOX":
            mgmt_choices = [
                "Puberty induction (low-dose oestradiol escalation) + HRT long-term",
                "Egg donation + uterine preparation with HRT",
                "HRT + aggressive bone protection (DEXA + bisphosphonates if T <-2.5)",
                "Puberty induction age 11-12y + add progestogen after 2 years",
                "Psychological support + MDT for primary amenorrhoea diagnosis",
            ]
        elif gene == "FIGLA":
            mgmt_choices = [
                "Puberty induction (low-dose oestradiol escalation) + HRT",
                "Egg donation (own oocytes absent from birth)",
                "HRT + bone DEXA + calcium + vitamin D",
                "Psychological MDT + adolescent fertility counselling",
                "Long-term HRT until age 51 + cardiovascular screening",
            ]
        elif gene == "MCM8":
            mgmt_choices = [
                "HRT + cancer surveillance (colonoscopy + endometrial sampling from age 30)",
                "Egg donation + avoid cisplatin/crosslinkers if cancer",
                "HRT + chromosomal instability monitoring",
                "Cancer surveillance + MCM9 co-testing (paralog)",
                "Ashkenazi founder testing + carrier cascade",
            ]
        else:
            mgmt_choices = ["HRT + HRT + specialist review"]

        mgmt = rng.choice(mgmt_choices)

        # Extra clinical events
        evs = []
        if gene == "FMR1":
            cggg = rng.randint(59, 195)
            evs.append(f"FMR1 PCR: {cggg} CGG repeats (premutation confirmed)")
            if rng.random() < 0.25:
                evs.append("AMH: <0.5 ng/mL at age 30 (severely reduced)")
            if rng.random() < 0.20:
                evs.append("TSH elevated: autoimmune thyroiditis co-association")
            if i < 5:
                evs.append("FXTAS MRI: middle cerebellar peduncle T2 hyperintensity (age >50)")
            if rng.random() < 0.20:
                evs.append("Family: affected male relative with Fragile X syndrome confirmed")
        elif gene == "FOXL2":
            evs.append("Eyelid exam: blepharophimosis + ptosis + epicanthus inversus confirmed")
            if rng.random() < 0.70:
                evs.append("Ptosis repair performed age 3-4y (timely — no amblyopia)")
            if rng.random() < 0.25:
                evs.append("Amblyopia detected (ptosis repair delayed beyond age 5y)")
            if rng.random() < 0.15:
                evs.append("USS pelvis: ovarian cyst (granulosa cell tumour surveillance — benign on biopsy)")
            evs.append(f"FSH: {rng.randint(25, 95)} IU/L (hypergonadotropic)")
        elif gene == "BMP15":
            if rng.random() < 0.30:
                evs.append("IVF cycle: OHSS risk identified — switched to GnRH agonist trigger")
            if rng.random() < 0.25:
                evs.append("AMH: <0.8 ng/mL at age 28 (earlier than expected decline)")
            evs.append(f"FSH: {rng.randint(18, 65)} IU/L")
            if rng.random() < 0.15:
                evs.append("Co-occurring GDF9 variant identified (compound oocyte paracrine defect)")
        elif gene == "GDF9":
            if rng.random() < 0.40:
                evs.append("IVF: poor cumulus expansion — ICSI performed all MII oocytes")
            if i < 10:
                evs.append("Primary amenorrhoea: karyotype 46,XX; FSH >80 IU/L; AR GDF9 biallelic")
            else:
                evs.append(f"FSH: {rng.randint(20, 65)} IU/L at age {rng.randint(28, 42)}")
            if rng.random() < 0.15:
                evs.append("Family: dizygotic twins in maternal line (GDF9 gain-of-function polymorphism screening)")
        elif gene == "NR5A1":
            if i < 5:
                evs.append("Karyotype: 46,XY; external genitalia female; laparoscopy: streak gonads")
            else:
                syn = rng.choice([
                    "Synacthen stimulation: peak cortisol 320 nmol/L (borderline — hydrocortisone prescribed)",
                    "Synacthen stimulation: peak cortisol 580 nmol/L (adequate adrenal reserve)",
                    "Synacthen stimulation: peak cortisol 200 nmol/L (subnormal — AI confirmed)",
                ])
                evs.append(syn)
            if rng.random() < 0.25:
                evs.append("Inhibin B: <20 pg/mL (fell before FSH rise — earliest marker)")
            if rng.random() < 0.20:
                evs.append("DHEA-S: <0.5 umol/L (low androgen; DHEA supplement started)")
        elif gene == "NOBOX":
            evs.append(f"FSH: {rng.randint(55, 110)} IU/L (very high — hypergonadotropic)")
            evs.append("USS pelvis: no antral follicles; ovary length <2 cm")
            if rng.random() < 0.35:
                evs.append("DEXA: T-score -2.1 (osteopenia — HRT not started promptly)")
            if rng.random() < 0.15:
                evs.append("Egg donation cycle: successful pregnancy (uterus primed with HRT 6 months)")
        elif gene == "FIGLA":
            evs.append(f"FSH: {rng.randint(65, 120)} IU/L; LH: {rng.randint(30, 80)} IU/L")
            evs.append("USS pelvis: streak gonads bilaterally; no follicular structure visible")
            if rng.random() < 0.20:
                evs.append("Egg donation: successful pregnancy after 18 months uterine HRT priming")
            if rng.random() < 0.30:
                evs.append("DEXA: T-score -2.4 (osteopenia; bisphosphonates added)")
        elif gene == "MCM8":
            evs.append("Chromosomal fragility: micronuclei observed on karyotype")
            if rng.random() < 0.35:
                evs.append("Colonoscopy age 30: tubular adenoma removed (Lynch-like surveillance)")
            if rng.random() < 0.25:
                evs.append("Endometrial biopsy: normal (year 2 surveillance)")
            if i < 6:
                evs.append("Founder: MCM8 p.Lys325Glu homozygous (Ashkenazi Jewish)")
            if rng.random() < 0.15:
                evs.append("Mitomycin C sensitivity: chromosomal breaks confirmed (cancer treatment planning)")

        patients.append({
            "id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "age": age,
            "sex": sex,
            "onset": gene_entry["onset_age"][:60],
            "hearing_severity": severity,   # repurposed: POI phenotype/severity
            "management": mgmt,
            "key_features": "; ".join(gene_entry["key_features"][:3]),
            "extra_events": "; ".join(evs) if evs else "—",
        })
    return patients


def _all_patients() -> list[dict]:
    out = []
    for g in POI_GENES:
        out.extend(_make_patients(g))
    return out


# ── API output functions ────────────────────────────────────────────────────────────────────────────

def get_overview() -> dict:
    """Aggregate summary for the /overview endpoint."""
    patients = _all_patients()
    gene_counts: dict[str, int] = {}
    phenotype_counts: dict[str, int] = {}
    mgmt_counts: dict[str, int] = {}

    for p in patients:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1
        phenotype_counts[p["hearing_severity"]] = phenotype_counts.get(p["hearing_severity"], 0) + 1
        mgmt_counts[p["management"]] = mgmt_counts.get(p["management"], 0) + 1

    gene_highlights = {
        g["gene"]: {
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"][:120] + "...",
            "disease_category": g["disease_category"][:120] + "...",
            "pathognomonic_short": g["key_features"][0],
        }
        for g in POI_GENES
    }

    return {
        "atlas": "Hereditary-Primary-Ovarian-Insufficiency-Atlas",
        "subtitle": (
            "Complete 8-Gene Primary Ovarian Insufficiency (POI) Reference "
            "(FMR1 · FOXL2 · BMP15 · GDF9 · NR5A1 · NOBOX · FIGLA · MCM8)"
        ),
        "total_patients": len(patients),
        "genes_covered": len(POI_GENES),
        "seed_range": f"{SEED_BASE}–{SEED_BASE + len(POI_GENES) - 1}",
        "patients_per_gene": gene_counts,
        "hearing_severity_distribution": phenotype_counts,
        "management_distribution": mgmt_counts,
        "gene_highlights": gene_highlights,
        "clinical_pearls": [
            "FMR1 (FXPOI): TEST FIRST in all POI — most common identifiable cause (20-28% of premutation carriers); PCR + Southern blot (WES misses CGG repeat)",
            "FOXL2 (BPES I): eyelid triad at birth (blepharophimosis+ptosis+epicanthus inversus); ptosis repair MANDATORY age 3-4y — amblyopia irreversible if delayed",
            "BMP15 (XLD POI): heterozygous females POI; hemizygous males unaffected; OHSS risk paradox — low-dose FSH + GnRH agonist trigger in IVF",
            "GDF9 (AR/AD): oocyte TGF-β; BMP15-GDF9 cumulin heterodimer 10x potency; ICSI preferred (poor cumulus expansion); test BMP15+GDF9 together",
            "NR5A1 (SF-1): 46,XX POI + adrenal insufficiency risk — Synacthen test MANDATORY at diagnosis; emergency hydrocortisone kit if borderline",
            "NOBOX (AR): primordial→primary follicle transition blocked; primary amenorrhoea; FSH >40-100 IU/L; puberty induction age 11-12y mandatory",
            "FIGLA (AR): primordial follicle ASSEMBLY failure; streak gonads from birth; rarest AR POI; 46,XX streak gonads LOW gonadoblastoma risk",
            "MCM8 (AR): meiotic DNA repair; CANCER SURVEILLANCE mandatory from age 30y (colonoscopy + endometrial); avoid cisplatin/crosslinkers",
        ],
    }


def get_breakdown() -> dict:
    """Per-gene breakdown for the /breakdown endpoint."""
    breakdown = {}
    for g in POI_GENES:
        patients = _make_patients(g)
        mgmt_breakdown: dict[str, int] = {}
        severity_breakdown: dict[str, int] = {}
        for p in patients:
            mgmt_breakdown[p["management"]] = mgmt_breakdown.get(p["management"], 0) + 1
            severity_breakdown[p["hearing_severity"]] = severity_breakdown.get(p["hearing_severity"], 0) + 1

        breakdown[g["gene"]] = {
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
            "onset_age": g["onset_age"],
            "surgical_urgency": g["surgical_urgency"],
            "gene_family": g["gene_family"],
            "morphology": g["morphology"],
            "n_patients": g["n_patients"],
            "hearing_severity_distribution": severity_breakdown,
            "management_distribution": mgmt_breakdown,
            "sample_patients": patients[:5],
        }
    return {"breakdown_by_gene": breakdown, "total_genes": len(POI_GENES)}


def get_definitions() -> dict:
    """Clinical definitions for the /definitions endpoint."""
    return {
        "genes": [g["gene"] for g in POI_GENES],
        "seed_range": f"{SEED_BASE}–{SEED_BASE + len(POI_GENES) - 1}",
        "definitions": {
            "Primary_Ovarian_Insufficiency": (
                "Primary ovarian insufficiency (POI): hypergonadotropic hypogonadism in women <40 years; "
                "FSH >25 IU/L on two occasions ≥4 weeks apart; amenorrhoea ≥4 months; "
                "affects 1% of women <40y, 0.1% <30y; ~25-30% have an identifiable genetic cause "
                "on comprehensive gene panel; CRITICAL: 5-10% spontaneous pregnancies possible — "
                "contraception needed if pregnancy not desired"
            ),
            "Hypergonadotropic_vs_Hypogonadotropic": (
                "POI (primary gonadal failure): FSH ELEVATED (>25 IU/L), LH elevated, oestradiol LOW → "
                "gonads not responding to pituitary signals = PRIMARY gonadal failure; "
                "HH (Kallmann/nIHH): FSH LOW (<2 IU/L), LH LOW, oestradiol LOW → "
                "pituitary not signalling = SECONDARY gonadal failure; "
                "CRITICAL DISTINCTION: measure FSH, LH, oestradiol BEFORE starting HRT or OCP; "
                "OCP suppresses FSH — cannot diagnose POI on OCP"
            ),
            "FMR1_Premutation_POI": (
                "FMR1 premutation (55-200 CGG): elevated FMR1 mRNA (NOT silenced unlike full mutation >200 CGG); "
                "mRNA toxic gain-of-function in granulosa cells → mitochondrial dysfunction → follicular atresia; "
                "20-28% of female premutation carriers develop POI; "
                "DIAGNOSIS: PCR for CGG repeat sizing + Southern blot for instability; WES MISSES expansions; "
                "FERTILITY PRESERVATION: oocyte cryopreservation in all premutation carriers aged 20-25y before FSH rises"
            ),
            "BPES_Eyelid_Triad": (
                "Blepharophimosis-Ptosis-Epicanthus Inversus: three eyelid anomalies visible from birth; "
                "Blepharophimosis: reduced horizontal palpebral fissure (normal 28-30mm; BPES <22mm); "
                "Ptosis: drooping upper eyelid (levator palpebrae superioris underdevelopment); "
                "Epicanthus inversus: medial canthal skin fold curves UPWARD (vs downward in Down syndrome); "
                "BPES type I = eyelid triad + POI (FOXL2 haploinsufficiency); "
                "BPES type II = eyelid triad only (no POI); "
                "PTOSIS REPAIR TIMING: before age 4y — visual axis obstruction → irreversible deprivational amblyopia"
            ),
            "Cumulin_BMP15_GDF9_Heterodimer": (
                "Cumulin = BMP15 + GDF9 heterodimer; secreted by oocyte; binds BMPRII+ALK6 AND ALK5 simultaneously; "
                "activates SMAD1/5/8 + SMAD2/3 in granulosa → dual cross-pathway → 10x more potent than "
                "either homodimer; triggers cumulus expansion (HAS2, PTGS2, TNFAIP6) → COC competence; "
                "LOF in EITHER BMP15 or GDF9 → reduced cumulin → poor IVF outcomes (poor fertilisation); "
                "ICSI preferred when either gene affected; recombinant cumulin under investigation as IVF adjunct"
            ),
            "NR5A1_Adrenal_Insufficiency_Risk": (
                "NR5A1 (SF-1) activates entire steroidogenesis cascade (StAR, CYP11A1, HSD3B2, CYP17A1, CYP19A1); "
                "ADRENAL THRESHOLD: adrenals require more NR5A1 activity than gonads — heterozygous LOF → "
                "gonadal failure (POI) but adrenal often preserved; HOWEVER: stress response impaired; "
                "SYNACTHEN TEST MANDATORY: 250 mcg ACTH IM; peak cortisol <500 nmol/L = borderline; "
                "<350 nmol/L = subnormal = adrenal insufficiency; SICK-DAY RULE: double/triple hydrocortisone "
                "for fever/illness/surgery; emergency hydrocortisone 100mg IM kit for all borderline patients"
            ),
            "MCM8_Cancer_Surveillance": (
                "MCM8-MCM9 helicase complex resolves meiotic DSBs and mitotic replication stress; "
                "biallelic LOF → chromosomal instability in somatic cells → Lynch-like cancer risk; "
                "SURVEILLANCE from age 30y: annual colonoscopy (colorectal adenoma/cancer risk); "
                "annual endometrial USS + sampling (endometrial cancer risk); CA-125 + TVUSS annually; "
                "GENOTOXIC AVOIDANCE: avoid cisplatin, mitomycin C, oxaliplatin — crosslinkers exacerbate "
                "chromosomal instability in MCM8 deficient cells; use carboplatin alternatives if possible"
            ),
            "POI_HRT_Protocol": (
                "HRT for POI (NOT postmenopausal HRT — different risk profile): "
                "OESTROGEN: Transdermal oestradiol preferred (avoids first-pass, safer thrombosis profile); "
                "dose 100-200 mcg/day or equivalent (higher than postmenopausal dose — replacing lost natural production); "
                "PROGESTOGEN: Micronised progesterone 200 mg/day (12 days/month) OR "
                "levonorgestrel IUS (Mirena) as intrauterine progestogen; "
                "DURATION: Until age 51 (natural menopause age) — not the standard '5-year limit' for symptomatic menopause HRT; "
                "BENEFITS: bone protection, cardiovascular protection, cognitive protection, vasomotor symptom control; "
                "POI HRT is REPLACEMENT not supplementation — no increased breast cancer risk vs natural menopause"
            ),
            "Puberty_Induction_POI": (
                "Primary amenorrhoea / absent puberty (NOBOX, FIGLA, severe NR5A1, MCM8): "
                "START OESTROGEN age 11-12y (not earlier — mimics natural puberty timing); "
                "DOSE ESCALATION: Start 1/8-1/4 adult dose (6.25-12.5 mcg transdermal/day); "
                "double dose every 6 months over 2-3 years; TARGET: breast development Tanner stage II-IV; "
                "ADD PROGESTOGEN: after 2 years OR at breakthrough bleeding (whichever first); "
                "BONE: Calcium + vitamin D throughout; DEXA at age 15 and every 2-3 years; "
                "UTERINE GROWTH: Progressive oestrogen causes uterine growth → egg donation feasible once "
                "endometrial stripe ≥7mm achieved"
            ),
            "Fertility_in_POI": (
                "SPONTANEOUS PREGNANCY: 5-10% chance possible (fluctuating function) — contraception needed; "
                "OWN EGGS (if AFC present): IVF with ICSI; "
                "random-start antagonist protocol; low-dose FSH; trigger with GnRH agonist (avoid OHSS — BMP15); "
                "EGG DONATION: mainstay for depleted reserves; donor-recipient synchronisation; "
                "GnRH-a long protocol for recipient; progesterone-based luteal support; "
                "PRESERVATION: oocyte/embryo cryopreservation offered before AFC depleted; "
                "FMR1 premutation carriers: offer at first visit aged 20-25y"
            ),
            "Inhibin_B_as_POI_Early_Marker": (
                "Inhibin B (granulosa cell product, FSH-regulated): falls BEFORE FSH rises in POI; "
                "NORMAL inhibin B (premenopausal): >45 pg/mL; "
                "LOW inhibin B (<20 pg/mL) with normal FSH = incipient POI — EARLIEST detectable marker; "
                "ESPECIALLY USEFUL: NR5A1 carriers (inhibin B low years before clinical POI); "
                "AMH: less fluctuation than FSH; reflects antral follicle pool; low AMH + low inhibin B "
                "= significant ovarian reserve depletion even before FSH elevation"
            ),
        },
        "emergency_protocols": [
            "NR5A1 ADRENAL CRISIS: Give hydrocortisone 100 mg IM + IV saline + glucose immediately; ECG; measure cortisol (post-treatment); call endocrine MDT",
            "FOXL2 AMBLYOPIA PREVENTION: Refer to paediatric ophthalmology URGENTLY if ptosis detected in child <4y — visual axis occlusion → irreversible amblyopia",
            "MCM8 GENOTOXIC AVOIDANCE: If MCM8 patient requires oncology treatment, flag for avoidance of cisplatin/mitomycin C crosslinkers; use alternative regimens",
            "FMR1 FXTAS EMERGENCY: Falling motor function in premutation carrier >50y → urgent neurology; FXTAS may mimic Parkinson's or MSA",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (FMR1) ===")
    bk = get_breakdown()
    print(json.dumps(bk["breakdown_by_gene"]["FMR1"], indent=2)[:2000])
    print("\n=== DEFINITIONS (first 1000 chars) ===")
    df = get_definitions()
    print(json.dumps(df, indent=2)[:1000])
