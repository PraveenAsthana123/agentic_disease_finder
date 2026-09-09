#!/usr/bin/env python3
"""Hereditary Precocious Puberty Atlas — Complete 8-Gene Reference
(MKRN3 · DLK1 · KISS1 · KISS1R · LIN28B · GNAS · LEPR · GNRH1).

MKRN3  (Makorin Ring Finger Protein 3; 507 aa; ~58 kDa; Xq27.1;
         AD — paternally imprinted (maternally silenced) LOF;
         MOST COMMON hereditary CPP — ~46% of familial CPP;
         GnRHa is curative; seed SEED_BASE+0).
DLK1   (Delta-Like Non-Canonical Notch Ligand 1; 383 aa; ~45 kDa; 14q32.2;
         AD — paternally expressed / maternal LOF causes CPP in offspring;
         2nd most common hereditary CPP; seed SEED_BASE+1).
KISS1  (Kisspeptin-1; 145 aa; ~16 kDa; 1q32.1;
         AR/AD GOF — kisspeptin ligand excess drives pulsatile GnRH;
         first GOF gene described; R73C most common; seed SEED_BASE+2).
KISS1R (GPR54 / Kisspeptin Receptor; 398 aa; ~46 kDa; 19p13.3;
         AD GOF (CPP) vs LOF (IHH) — same gene, opposite mutation classes;
         A243V GOF Syrian girl — earliest case; seed SEED_BASE+3).
LIN28B (Lin-28 Homolog B; 250 aa; ~28 kDa; 6q16.3;
         AD GOF — represses let-7 miRNA → disinhibits pubertal axis;
         GWAS-identified; seed SEED_BASE+4).
GNAS   (Guanine Nucleotide Binding Protein Alpha Subunit; 395 aa; ~45 kDa; 20q13.32;
         Somatic GOF postzygotic — McCune-Albright syndrome (MAS);
         PERIPHERAL precocious puberty — GnRHa FAILS — aromatase inhibitor Rx;
         fibrous dysplasia + café-au-lait (Coast of Maine) PATHOGNOMONIC;
         seed SEED_BASE+5).
LEPR   (Leptin Receptor; 1165 aa; ~132 kDa; 1p31.3;
         AR LOF — severe early obesity; CPP after leptin/LEPR pathway restoration;
         leptin is permissive signal for puberty onset; seed SEED_BASE+6).
GNRH1  (Gonadotropin Releasing Hormone 1; 92 aa; ~10 kDa; 8p21.2;
         AD activating mutations — GnRH hyperpulse drives precocious HPG axis;
         rare (~3-5% familial CPP); seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2494-2501).
"""

import random

SEED_BASE = 2494

CPP_GENES = [
    # -- MKRN3 -- MOST COMMON HEREDITARY CPP ------------------------------------------
    {
        "gene": "MKRN3",
        "alt_name": (
            "MKRN3 (MKRN3-507aa-Xq27.1 / AD-paternal-imprinting-LOF -- "
            "Makorin-Ring-Finger-Protein-3 -- "
            "MOST-COMMON-Hereditary-CPP-~46pct-Familial -- "
            "MATERNAL-TRANSMISSION-ONLY-Symptomatic-X-Linked-Like -- "
            "GnRHa-Curative)"
        ),
        "protein": (
            "MKRN3 -- Xq27.1 AD-paternal-imprinting -- MKRN3-507aa -- "
            "Makorin-Ring-Finger-3-58kDa-RING-ZNF-Ubiquitin-E3-Ligase-Nuclear -- "
            "Paternally-Expressed-Maternally-Silenced-Imprinted-Gene -- "
            "OMIM-Gene-603856-Disease-CPP2-615346"
        ),
        "locus": "Xq27.1",
        "protein_size": "507 aa / ~58 kDa",
        "inheritance": (
            "Autosomal dominant, paternally imprinted (maternally silenced gene). "
            "MKRN3 is located on Xq27.1 within the Prader-Willi syndrome imprinting region. "
            "It is MATERNALLY SILENCED — only the paternal allele is expressed. "
            "CONSEQUENCE: Only LOF mutations inherited from the FATHER cause CPP. "
            "A mutation inherited from the MOTHER does NOT cause CPP (maternal allele already silenced). "
            "This produces an X-linked-like inheritance pattern — daughters of affected fathers are at 50% risk; "
            "sons of affected fathers carry the mutation but the allele is silenced in their daughters "
            "(paternal → maternal transmission silences the allele). "
            "MECHANISM: MKRN3 normally suppresses GnRH neuron activity; LOF → disinhibition of GnRH "
            "pulse generator (arcuate nucleus KNDy neurons) → precocious GnRH pulsatility → LH/FSH "
            "surge → gonadal sex steroid production → early secondary sexual characteristics. "
            "PREVALENCE: ~46% of familial CPP; the single most important monogenic cause of hereditary CPP. "
            "Frameshift and nonsense mutations most common; missense (loss-of-function) also reported. "
            "IMPORTANT: Standard WES/exome panels include MKRN3 — confirm deletion/duplication "
            "analysis also performed (MLPA or chromosomal microarray for large deletions). "
            "TIMING: Breast development <8 years (girls) or testicular volume ≥4 mL <9 years (boys); "
            "bone age accelerated; final height reduced without treatment. "
            "Treat with: GnRHa (leuprolide, triptorelin) — highly effective, suppresses HPG axis."
        ),
        "disease_category": (
            "Hereditary Central Precocious Puberty type 2 (CPP2, OMIM #615346); "
            "AD paternally imprinted LOF; most common monogenic CPP; Xq27.1; "
            "affects girls more frequently (hormonal threshold lower); "
            "GnRHa treatment → normalises puberty tempo and final height"
        ),
        "disease_pathway": (
            "MKRN3 SUPPRESSION OF GnRH PULSE GENERATOR: "
            "MKRN3 expressed in arcuate nucleus KNDy neurons (Kisspeptin-Neurokinin B-Dynorphin); "
            "MKRN3 ubiquitinates and degrades activators of KISS1/TAC3 expression → "
            "suppresses kisspeptin release → suppresses GnRH pulsatility. "
            "LOF → KNDy neurons hyperactive → kisspeptin surges → "
            "GnRH pulses premature → pituitary LH/FSH surge → "
            "ovarian/testicular steroid production → early puberty. "
            "PREPUBERTAL LEVEL: MKRN3 expression HIGH in juvenile; "
            "falls just before pubertal onset (normal; removes brake on GnRH) — "
            "LOF anticipates this fall pathologically early. "
            "BONE AGE: Advanced bone age → fused epiphyses prematurely → short adult stature without Rx. "
            "GnRHa ACTION: Pituitary GnRH receptor downregulation → FSH/LH suppressed → "
            "gonadal steroids suppressed → bone age normalises → full height potential preserved."
        ),
        "pathognomonic": (
            "MKRN3 CLINICAL PEARLS: "
            "FATHER TRANSMISSION ONLY: In a girl with CPP and a family history — "
            "ask specifically: 'Did the father or paternal grandmother have early puberty?' "
            "Maternal family history of CPP does NOT increase MKRN3 risk. "
            "PEDIGREE IS DIAGNOSTIC: Multiple girls with CPP across generations through fathers → MKRN3 first. "
            "BOYS: 46XY individuals with MKRN3 LOF can also develop CPP (testicular volume ≥4 mL <9 yr); "
            "boys are less commonly affected due to higher androgen threshold. "
            "BONE AGE: X-ray left hand/wrist — advanced bone age (>2SD above chronological age) confirms CPP. "
            "BRAIN MRI: MANDATORY to exclude hypothalamic hamartoma, optic glioma, other structural causes "
            "BEFORE attributing CPP to MKRN3 alone — structural CPP is most common cause overall. "
            "GnRHa RESPONSE: LH/FSH suppression to prepubertal levels within 2 weeks confirms central axis. "
            "OMIM Gene: 603856, Disease: CPP2 — 615346."
        ),
        "treatment": (
            "GnRH AGONIST (GnRHa): Leuprolide depot 3.75 mg IM q28d (or 7.5 mg q84d); "
            "triptorelin 3.75 mg IM q28d; histrelin implant 50 mg SC annually. "
            "MONITORING: LH/FSH: suppressed to prepubertal levels (<0.5 IU/L) after 2-3 months; "
            "oestradiol <73 pmol/L (girls) or testosterone <0.7 nmol/L (boys); "
            "bone age X-ray annually; height velocity; pubertal staging (Tanner). "
            "DURATION: Until appropriate chronological age for puberty onset (~10-11 yr girls, ~11-12 yr boys). "
            "FINAL HEIGHT: GnRHa treatment preserves 4-9 cm of adult height if started ≤6 yr. "
            "BONE DENSITY: DXA at treatment start and after 2 yr — GnRHa may reduce BMD transiently; "
            "recovered once puberty resumes. "
            "FERTILITY: Outcomes excellent — ovarian reserve and fertility unaffected by GnRHa."
        ),
        "key_features": [
            "MOST COMMON hereditary CPP gene (~46% familial CPP, most important single gene)",
            "PATERNALLY IMPRINTED: only paternal LOF causes CPP — maternal mutation is silent",
            "Xq27.1 location — X-linked-like transmission pattern (father → daughters)",
            "Brain MRI MANDATORY to exclude structural causes before attributing CPP to MKRN3",
            "GnRHa (leuprolide/triptorelin/histrelin) is highly effective and curative",
            "Bone age advanced — GnRHa treatment preserves final adult height",
            "Exome panels include MKRN3 — but also confirm MLPA for large deletions",
            "Boys can be affected (CPP ≥4 mL testis <9 yr) though less commonly than girls",
        ],
        "key_ddx": (
            "MKRN3 vs DLK1: DLK1 is maternally inherited (not paternally); both cause CPP; "
            "MKRN3 vs hypothalamic hamartoma: brain MRI excludes structural cause — mandatory first; "
            "MKRN3 vs KISS1/KISS1R GOF: KISS1/KISS1R have biallelic or dominant GOF, not imprinting; "
            "MKRN3 vs premature thelarche: isolated breast development with no LH response to GnRH test; "
            "MKRN3 vs adrenal CPP: dehydroepiandrosterone (DHEAS) elevated in premature adrenarche; LH normal"
        ),
        "systemic_involvement": {
            "HPG_axis": "GnRH pulse generator disinhibited; LH/FSH prematurely elevated",
            "bone": "Advanced bone age; epiphyseal fusion risk without treatment",
            "growth": "Initially accelerated growth; reduced final height without GnRHa",
            "behaviour": "Emotional lability, mood changes associated with early sex steroid exposure",
        },
        "cascade_testing": (
            "AD paternal imprinting — cascade testing: "
            "Father of affected girl → test for MKRN3 mutation; "
            "paternal grandmother → may carry mutation (asymptomatic, maternal allele); "
            "siblings: daughters of affected father → 50% risk; "
            "sons of affected father → carry mutation but silenced in their daughters; "
            "prenatal: molecular diagnosis available; "
            "clinical surveillance of at-risk daughters: pubertal staging annually from age 6 yr"
        ),
        "emergency_protocol": (
            "RARELY an emergency. However: "
            "VAGINAL BLEEDING in a pre-school girl → CPP or precocious menarche; "
            "immediate assessment: LH/FSH, oestradiol, pelvic USS (uterine length >3.4 cm = oestrogenised); "
            "brain MRI within 48-72 hr (exclude tumour/hamartoma); "
            "PSYCHOLOGICAL CRISIS: Precocious puberty causes significant psychosocial distress; "
            "refer to paediatric psychologist alongside endocrine treatment; "
            "GnRHa START: Begin treatment within weeks of diagnosis to minimise bone age advancement."
        ),
    },

    # -- DLK1 -- 2ND MOST COMMON -- PATERNAL IMPRINTING --------------------------------
    {
        "gene": "DLK1",
        "alt_name": (
            "DLK1 (DLK1-383aa-14q32.2 / AD-maternal-LOF-causes-CPP -- "
            "Delta-Like-Non-Canonical-Notch-Ligand-1 -- "
            "Paternally-Expressed-Maternally-Silenced -- "
            "Maternal-LOF-Causes-CPP-Second-Most-Common -- "
            "GWAS-Confirmed-Puberty-Timing-Locus)"
        ),
        "protein": (
            "DLK1 -- 14q32.2 AD-paternal-imprinting -- DLK1-383aa -- "
            "DLK1-Delta-Like-Non-Canonical-Notch-Ligand-1-45kDa-EGF-Repeat-Transmembrane -- "
            "Paternally-Expressed-Maternally-Silenced-14q32-Imprinting-Cluster -- "
            "OMIM-Gene-176290-Disease-CPP -- Puberty-Timing-GWAS-2021"
        ),
        "locus": "14q32.2",
        "protein_size": "383 aa / ~45 kDa",
        "inheritance": (
            "Autosomal dominant, paternally expressed / maternally silenced. "
            "DLK1 resides within the 14q32 imprinting cluster (DLK1-MEG3 domain) alongside GTL2/MEG3. "
            "DLK1 is PATERNALLY EXPRESSED — only the paternal copy is active. "
            "LOF mutations inherited from the MOTHER cause CPP (maternal LOF → derepression of hypothalamic "
            "signalling without DLK1 suppression). "
            "Contrast with MKRN3: MKRN3 requires paternal inheritance for disease; "
            "DLK1 requires MATERNAL inheritance for disease — opposite imprinting direction. "
            "MECHANISM: DLK1 (also called Pref-1 or FA1) is a Notch pathway modulator; "
            "expressed in hypothalamic astrocytes and neurons adjacent to GnRH neurons; "
            "DLK1 suppresses astrocytic maturation signals that would otherwise activate GnRH neurons; "
            "LOF → astrocytic activation of GnRH neurons premature → CPP. "
            "FREQUENCY: ~16% of familial CPP (2nd most common after MKRN3). "
            "ADDITIONAL FEATURES: DLK1 is also critical for adipogenesis — LOF may be associated "
            "with increased adiposity (contrast with LEPR-related obesity). "
            "GWAS: Multiple common variants near DLK1-MEG3 locus are associated with earlier puberty timing "
            "in population studies — supporting DLK1 role in normal variation as well as Mendelian CPP."
        ),
        "disease_category": (
            "Hereditary Central Precocious Puberty (DLK1-related CPP); "
            "AD maternally transmitted (paternally expressed gene); 14q32.2; "
            "2nd most common monogenic CPP after MKRN3; "
            "may co-exist with increased adiposity; GnRHa treatment effective"
        ),
        "disease_pathway": (
            "DLK1 NOTCH-ASTROCYTE-GnRH PATHWAY: "
            "DLK1 expressed in hypothalamic tanycytes and astrocytes surrounding GnRH neurons. "
            "DLK1 suppresses astrocytic maturation and secretion of factors (EGF, bFGF, TGFβ1) "
            "that would directly stimulate GnRH neuron activity. "
            "LOF → astrocytic Notch signalling unopposed → "
            "premature astrocytic EGF/bFGF secretion → "
            "GnRH neuron stimulation → pulsatile GnRH premature → "
            "pituitary LH/FSH surge → ovarian oestradiol → CPP. "
            "ADIPOGENESIS PARALLEL: DLK1 (Pref-1) normally inhibits adipocyte differentiation; "
            "LOF → increased lipid accumulation potential — explains adiposity association. "
            "IMPRINTING: 14q32 deletion (paternal) → Temple syndrome (short stature, obesity, DM); "
            "14q32 deletion (maternal) → Kagami-Ogata syndrome — entirely different phenotype; "
            "isolated DLK1 LOF causes CPP specifically via maternal inheritance of LOF allele."
        ),
        "pathognomonic": (
            "DLK1 CLINICAL PEARLS: "
            "MATERNAL INHERITANCE — OPPOSITE TO MKRN3: In a girl with CPP and maternal family history "
            "of early puberty → DLK1 before MKRN3 (MKRN3 requires paternal history). "
            "IMPRINTING AXIS: If pedigree shows: mother with CPP → daughter with CPP → DLK1 most likely. "
            "GWAS VALIDATION: DLK1-MEG3 region is strongest GWAS locus for female puberty timing; "
            "confirms biological relevance beyond rare Mendelian cases. "
            "ADIPOSITY: May have higher BMI percentile than average CPP case; "
            "adiposity itself accelerates puberty (leptin pathway) — DLK1 adds Notch pathway on top. "
            "TREATMENT: GnRHa identical to MKRN3 CPP — LH/FSH suppression highly effective; "
            "monitor adiposity alongside puberty control. "
            "OMIM Gene: 176290. Associated with CPP and puberty timing GWAS loci."
        ),
        "treatment": (
            "GnRHa: Same as MKRN3 — leuprolide depot / triptorelin / histrelin implant. "
            "MONITORING: LH/FSH prepubertal suppression; oestradiol; bone age; Tanner staging. "
            "ADIPOSITY MANAGEMENT: Dietary advice; physical activity; "
            "monitor BMI percentile alongside GnRHa therapy. "
            "METABOLIC MONITORING: Consider HbA1c, fasting glucose if significant adiposity; "
            "DLK1 LOF may impair adipocyte regulation long-term. "
            "GENETIC COUNSELLING: Maternal transmission — sisters at 50% risk if mother carries mutation; "
            "paternal aunts/uncles of affected child also at risk if maternal grandfather carries."
        ),
        "key_features": [
            "2nd most common hereditary CPP gene (~16% familial CPP)",
            "MATERNALLY TRANSMITTED (paternally expressed gene) — opposite inheritance to MKRN3",
            "14q32.2 DLK1-MEG3 imprinting cluster — population GWAS locus for puberty timing",
            "May be associated with adiposity (DLK1 suppresses adipogenesis normally)",
            "GnRHa treatment effective — same protocol as MKRN3 CPP",
            "Molecular testing: include DLK1 deletion/duplication analysis (MLPA/chromosomal microarray)",
            "Pedigree with maternal CPP history → DLK1 before MKRN3",
        ],
        "key_ddx": (
            "DLK1 vs MKRN3: DLK1 maternal inheritance; MKRN3 paternal inheritance; "
            "both are imprinted but in OPPOSITE directions — pedigree is key; "
            "DLK1 vs Temple syndrome: 14q32 deletion paternal → Temple (growth failure, obesity); "
            "DLK1 isolated LOF maternal → CPP only; "
            "DLK1 vs obesity-related CPP: increased BMI accelerates puberty via leptin; "
            "DLK1 adds Notch pathway contribution independent of BMI"
        ),
        "systemic_involvement": {
            "HPG_axis": "Premature GnRH activation via astrocytic Notch disinhibition",
            "adipose": "Increased adiposity risk (DLK1/Pref-1 normally inhibits adipocyte differentiation)",
            "bone": "Advanced bone age; height preservation requires timely GnRHa",
        },
        "cascade_testing": (
            "Maternal transmission — cascade: mother → 50% of daughters risk; "
            "maternal grandmother may carry mutation; "
            "maternal aunts (sisters of mother) → 50% carriers; their daughters → at risk; "
            "males carry mutation silently (paternally expressed gene — their children not at risk via paternal); "
            "prenatal: molecular diagnosis available"
        ),
        "emergency_protocol": (
            "Same as MKRN3 — not typically an emergency. "
            "VAGINAL BLEEDING in pre-school girl with maternal CPP history → urgent evaluation; "
            "pelvic USS + LH/FSH/oestradiol + bone age + brain MRI; "
            "refer to paediatric endocrinology for GnRHa initiation."
        ),
    },

    # -- KISS1 -- GOF KISSPEPTIN LIGAND -------------------------------------------------
    {
        "gene": "KISS1",
        "alt_name": (
            "KISS1 (KISS1-145aa-1q32.1 / AR/AD-GOF -- "
            "Kisspeptin-1-Ligand -- "
            "First-GOF-Gene-Described-in-Hereditary-CPP -- "
            "R73C-Most-Common-Argentine-Family -- "
            "Extreme-Early-Onset-2yr)"
        ),
        "protein": (
            "KISS1 -- 1q32.1 AR/AD-GOF -- KISS1-145aa -- "
            "Kisspeptin-1-16kDa-Cleaved-to-Kisspeptin-54-Kisspeptin-10-Metastatin -- "
            "Arcuate-Nucleus-KNDy-Neurons-Master-GnRH-Pulse-Regulator -- "
            "OMIM-Gene-603286-Disease-CPP1-176400"
        ),
        "locus": "1q32.1",
        "protein_size": "145 aa / ~16 kDa",
        "inheritance": (
            "Autosomal recessive or autosomal dominant gain-of-function. "
            "KISS1 encodes kisspeptin-1 (also called metastin), a peptide cleaved into multiple "
            "active forms (kisspeptin-54, -14, -13, -10) that act on the kisspeptin receptor KISS1R/GPR54. "
            "GOF MUTATIONS: Increase resistance to peptidase cleavage → prolonged kisspeptin signalling → "
            "sustained GnRH neuron stimulation → premature puberty. "
            "KEY MUTATION: p.Arg73Cys (R73C) — first described in an Argentine family; "
            "creates a novel disulfide bond → protects kisspeptin-54 from enzymatic degradation; "
            "increased half-life → prolonged KISS1R activation. "
            "IMPRINTING: None — KISS1 is biallelically expressed; GOF mutations cause CPP regardless "
            "of parental origin. "
            "ONSET: Can be extremely early — as young as 2 years of age in homozygous/compound "
            "heterozygous GOF. "
            "MOLECULAR TESTING: Ensure KISS1 sequencing includes exon covering R73C (codon 73); "
            "standard exome panels should capture this. "
            "CONTRAST WITH LOF: KISS1 LOF → hypogonadotropic hypogonadism (IHH) — "
            "GOF and LOF cause completely opposite phenotypes."
        ),
        "disease_category": (
            "Hereditary Central Precocious Puberty type 1 (CPP1, OMIM #176400); "
            "KISS1 GOF; 1q32.1; AR (biallelic GOF) or AD (dominant GOF); "
            "extreme early onset (as young as 2 yr); prolonged kisspeptin signalling; "
            "GnRHa highly effective"
        ),
        "disease_pathway": (
            "KISS1 GOF → PROLONGED GNRH STIMULATION: "
            "Kisspeptin-54 cleaved to kisspeptin-10 by matrix metalloproteinases (MMP-1, neprilysin). "
            "R73C GOF: disulfide bond at position 73 → protects kisspeptin-54 from MMP cleavage → "
            "kisspeptin-54 half-life prolonged from minutes to hours → "
            "KISS1R/GPR54 sustained activation → "
            "GnRH neurons continuously stimulated → "
            "pituitary LH/FSH surges → sex steroid production → CPP. "
            "HYPOTHALAMUS: KNDy neurons (arcuate nucleus) express KISS1; "
            "AVPV neurons (anteroventral periventricular nucleus) express KISS1 in females → "
            "LH surge control; GOF in both neuron populations → precocious activation. "
            "GNRH DOWNSTREAM: GnRH → anterior pituitary GnRH-R → LH + FSH → gonad steroid production. "
            "GnRHa MECHANISM: GnRHa downregulates pituitary GnRH-R → breaks the downstream chain; "
            "kisspeptin remains elevated but receptor is downregulated → suppression achieved."
        ),
        "pathognomonic": (
            "KISS1 CLINICAL PEARLS: "
            "VERY EARLY ONSET: CPP at age 2 yr → KISS1/KISS1R GOF should be specifically tested; "
            "structural causes (hamartoma) remain most common but KISS1 GOF is the leading monogenic cause "
            "of extremely early-onset CPP (onset <4 yr). "
            "BILATERAL SYMMETRY: Breast development and bone age advanced; no virilisation (LH/FSH driven). "
            "BRAIN MRI STILL MANDATORY: Even with positive KISS1 mutation — structural cause must be excluded; "
            "hypothalamic hamartoma produces kisspeptin-like signals but is structural (not imprinted). "
            "GOF vs LOF TESTING: Request 'KISS1 sequencing with GOF/LOF interpretation'; "
            "R73C is activating; loss-of-function variants cause IHH — opposite phenotype. "
            "OMIM Gene: 603286, Disease: CPP1 — 176400."
        ),
        "treatment": (
            "GnRHa (leuprolide/triptorelin/histrelin) — highly effective; "
            "pituitary GnRH-R downregulation breaks downstream activation chain. "
            "MONITORING: LH/FSH suppressed to prepubertal; oestradiol/testosterone suppressed; "
            "bone age annually; pubertal staging. "
            "DURATION: Until appropriate chronological age for puberty. "
            "KISSPEPTIN LEVELS: May be measured but not routine for monitoring — "
            "GnRHa efficacy judged by LH/FSH/gonadal steroid suppression, not kisspeptin levels."
        ),
        "key_features": [
            "First GOF gene described in hereditary CPP (2008, Argentine family, R73C)",
            "AR/AD GOF — not imprinted; biallelic GOF can cause extreme early-onset CPP (<4 yr)",
            "R73C most common: disulfide bond protects kisspeptin-54 from MMP cleavage",
            "KISS1 GOF (CPP) vs KISS1 LOF (IHH) — same gene, opposite phenotypes",
            "Brain MRI still mandatory even with positive KISS1 mutation",
            "GnRHa highly effective — pituitary receptor downregulation breaks chain",
            "Test at age <4 yr with very early-onset CPP after excluding hamartoma/other structural",
        ],
        "key_ddx": (
            "KISS1 GOF vs KISS1R GOF: both elevate GnRH pulsatility; kisspeptin levels distinguish — "
            "KISS1 GOF: kisspeptin elevated; KISS1R GOF: kisspeptin normal/elevated but receptor hyperactive; "
            "KISS1 GOF vs hypothalamic hamartoma: MRI distinguishes — hamartoma visible; "
            "KISS1 GOF vs MKRN3/DLK1: no imprinting pattern; may appear de novo or AR"
        ),
        "systemic_involvement": {
            "HPG_axis": "Sustained kisspeptin-GPR54 → constitutive GnRH → early LH/FSH surge",
            "bone": "Severely advanced bone age in earliest-onset cases",
        },
        "cascade_testing": (
            "GOF — recessive or dominant; parental testing for R73C or other GOF allele; "
            "siblings: 25% risk (biallelic) or 50% risk (AD GOF); "
            "no imprinting — inheritance pattern is standard Mendelian"
        ),
        "emergency_protocol": (
            "CPP at age 2 yr in a girl → urgent paediatric endocrinology referral; "
            "brain MRI within 48 hr (hamartoma exclusion); "
            "LH/FSH, oestradiol, pelvic USS, bone age all same day; "
            "GnRHa start as soon as possible — bone age advances rapidly at very early onset CPP."
        ),
    },

    # -- KISS1R -- GOF RECEPTOR --------------------------------------------------------
    {
        "gene": "KISS1R",
        "alt_name": (
            "KISS1R (KISS1R-398aa-19p13.3 / AD-GOF-CPP-vs-AR-LOF-IHH -- "
            "GPR54-Kisspeptin-Receptor -- "
            "A243V-GOF-Syrian-Family-First-Case -- "
            "SAME-GENE-OPPOSITE-MUTATIONS-CPP-vs-IHH -- "
            "GnRHa-Curative)"
        ),
        "protein": (
            "KISS1R -- 19p13.3 AD-GOF -- KISS1R-398aa -- "
            "GPR54-46kDa-7-TM-GPCR-Gq-Coupled-Phospholipase-C-Activation-IP3-DAG -- "
            "KNDy-Neuron-Master-GnRH-Pulse-Switch -- "
            "OMIM-Gene-604161-Disease-CPP1-176400-and-HH7-146110"
        ),
        "locus": "19p13.3",
        "protein_size": "398 aa / ~46 kDa",
        "inheritance": (
            "Autosomal dominant gain-of-function (CPP) OR autosomal recessive loss-of-function (IHH/Kallmann). "
            "KISS1R encodes GPR54, the kisspeptin receptor — a 7-TM Gq-coupled GPCR. "
            "CRITICAL TEACHING POINT — SAME GENE, OPPOSITE PHENOTYPES: "
            "GOF mutations (activating) → constitutive receptor activation without kisspeptin → "
            "continuous GnRH stimulation → CENTRAL PRECOCIOUS PUBERTY. "
            "LOF mutations (biallelic) → no kisspeptin-mediated GnRH stimulation → "
            "absent puberty → HYPOGONADOTROPIC HYPOGONADISM (in Kallmann atlas). "
            "FIRST GOF CPP CASE: Syrian female with p.Ala243Val (A243V) — reported 2007; "
            "breast development at 1 year; bone age advanced by 3.7 years; "
            "A243V in TM6 domain → constitutive IP3/DAG signalling. "
            "GOF MECHANISM: A243V disrupts normal receptor inactivation; "
            "Gq-IP3-DAG pathway constitutively active → GnRH neurons stimulated continuously. "
            "MOLECULAR TESTING: Request KISS1R sequencing with functional interpretation; "
            "A243V is in exon 5 (TM6 region); "
            "confirm GOF vs LOF functionally if variant of uncertain significance."
        ),
        "disease_category": (
            "Hereditary Central Precocious Puberty (KISS1R GOF-CPP, OMIM #176400); "
            "OR Hypogonadotropic Hypogonadism type 7 (HH7, OMIM #146110) for LOF; "
            "AD GOF causes CPP at 1-2 yr; AR LOF causes absent puberty/IHH; "
            "same gene, opposite clinical phenotypes"
        ),
        "disease_pathway": (
            "KISS1R GOF → CONSTITUTIVE GnRH ACTIVATION: "
            "Normal: Kisspeptin-10/54 binds GPR54 TM2/3/7 → Gq → PLC activation → IP3 + DAG → "
            "Ca2+ release → GnRH neuron depolarisation → GnRH pulse. "
            "GOF A243V: TM6 conformational change → Gq constitutively active WITHOUT kisspeptin → "
            "IP3/DAG/Ca2+ → continuous GnRH firing → "
            "pituitary LH/FSH constant stimulation → sex steroids → CPP. "
            "CONTRAST WITH LOF: LOF → no kisspeptin-mediated Gq signalling → "
            "GnRH neurons never activated → IHH/Kallmann phenotype. "
            "GnRHa TREATMENT: Pituitary GnRH receptor downregulation; "
            "KISS1R GOF-CPP responds just as well as other central CPP causes."
        ),
        "pathognomonic": (
            "KISS1R CLINICAL PEARLS: "
            "SAME GENE, TWO PHENOTYPES: When ordering KISS1R testing, specify whether testing "
            "for CPP (GOF) or IHH (LOF) — functional interpretation is critical. "
            "EARLIEST POSSIBLE ONSET: A243V can cause breast development at 1 year — "
            "earliest monogenic CPP cases on record (alongside KISS1 GOF). "
            "GOF CONFIRMED: If variant uncertain, transfection assay of mutant receptor shows "
            "constitutive IP3/Ca2+ activation in absence of kisspeptin. "
            "IMPRINTING: None — KISS1R is not imprinted; standard AD or AR. "
            "DIAGNOSTIC CLUE: In CPP with no structural cause and no family history → "
            "de novo AD GOF in KISS1R/KISS1 should be tested; "
            "particularly if onset before 3 yr. "
            "OMIM Gene: 604161."
        ),
        "treatment": (
            "GnRHa (leuprolide/triptorelin) — same as all central CPP. "
            "GOF RECEPTOR IS AT GNRH NEURON LEVEL — GnRHa works at pituitary, downstream; "
            "pituitary GnRH-R downregulation suppresses LH/FSH regardless of upstream KISS1R GOF. "
            "MONITORING: Standard CPP monitoring — LH/FSH, oestradiol, bone age. "
            "DURATION: Until appropriate pubertal age."
        ),
        "key_features": [
            "Same gene as Kallmann/IHH atlas — opposite mutations cause opposite phenotype (TEACHING POINT)",
            "GOF (A243V, TM6) → constitutive GPR54 → CPP; LOF → IHH",
            "Earliest monogenic CPP cases (breast at 1 yr); de novo or AD GOF",
            "19p13.3 — 7-TM GPCR — Gq-PLC-IP3/DAG-Ca2+ pathway",
            "GnRHa treatment effective (acts downstream at pituitary, not on KISS1R)",
            "Functional assay may be needed to confirm GOF vs VUS interpretation",
        ],
        "key_ddx": (
            "KISS1R GOF (CPP) vs KISS1R LOF (IHH): opposite phenotypes — check family history carefully; "
            "KISS1R GOF vs KISS1 GOF: both cause CPP; kisspeptin levels — elevated in KISS1 GOF; "
            "KISS1R GOF vs hypothalamic hamartoma: MRI mandatory regardless of genotype; "
            "KISS1R GOF vs MKRN3/DLK1: no imprinting; can be de novo"
        ),
        "systemic_involvement": {
            "HPG_axis": "Constitutive GPR54 → continuous GnRH secretion → precocious LH/FSH surge",
        },
        "cascade_testing": (
            "AD GOF — 50% risk to offspring; de novo possible; "
            "sibling testing: clinical surveillance; "
            "LOF carriers: heterozygous IHH carriers — test for reduced fertility if clinically relevant"
        ),
        "emergency_protocol": (
            "Breast development at 1-2 yr → urgent paediatric endocrinology referral; "
            "brain MRI within 48 hr (hamartoma exclusion); "
            "KISS1R GOF testing; GnRHa start promptly."
        ),
    },

    # -- LIN28B -- RNA-BINDING / GWAS --------------------------------------------------
    {
        "gene": "LIN28B",
        "alt_name": (
            "LIN28B (LIN28B-250aa-6q16.3 / AD-GOF -- "
            "Lin-28-Homolog-B-RNA-Binding-Protein -- "
            "let-7-miRNA-Repressor -- "
            "GWAS-Strongest-Hit-Female-Puberty-Timing -- "
            "Earlier-Puberty-Onset-2yr-Advanced-Bone-Age)"
        ),
        "protein": (
            "LIN28B -- 6q16.3 AD-GOF -- LIN28B-250aa -- "
            "LIN28B-28kDa-CSD-CCHC-Zinc-Knuckle-RNA-Binding-Protein-Cytoplasmic -- "
            "Represses-let-7-miRNA-Biogenesis-Dicer-Pre-let-7-Interaction -- "
            "OMIM-Gene-611044"
        ),
        "locus": "6q16.3",
        "protein_size": "250 aa / ~28 kDa",
        "inheritance": (
            "Autosomal dominant gain-of-function. "
            "LIN28B is an RNA-binding protein with cold-shock domain (CSD) and two CCHC zinc-knuckle motifs. "
            "It functions as a master regulator of the let-7 miRNA family. "
            "let-7 miRNAs normally repress: LIN28B itself, IGF2BP, HMGA2, and TARGETS including "
            "GNRH1, KISS1R, and multiple puberty-timing genes. "
            "LIN28B GOF → blocks let-7 biogenesis → let-7 targets derepressed → "
            "KISS1/GNRH1 upregulated → earlier pubertal axis activation. "
            "GWAS: The LIN28B locus (6q16.3) is the strongest GWAS hit for female age at menarche "
            "(2009, Perry et al., Ong et al.) — common variants explain ~7.7% of heritability for "
            "puberty timing in population studies. "
            "RARE MENDELIAN FORM: Rare coding GOF mutations cause familial CPP with earlier onset "
            "than the common GWAS variant effect (GWAS variants shift puberty timing by weeks; "
            "coding GOF mutations cause clinically significant CPP weeks to years earlier). "
            "DEVELOPMENTAL ROLE: LIN28B is highly expressed in embryonic tissue; "
            "downregulated post-natally; reactivation in let-7 repressor gain-of-function → "
            "precocious pubertal programming."
        ),
        "disease_category": (
            "Hereditary Central Precocious Puberty (LIN28B GOF-CPP); "
            "AD GOF; 6q16.3; strongest GWAS locus for female puberty timing; "
            "let-7 miRNA repression → upstream puberty gene derepression; "
            "GnRHa effective"
        ),
        "disease_pathway": (
            "LIN28B → let-7 REPRESSION → PUBERTY GENE DEREPRESSION: "
            "let-7 miRNAs: repress HMGA2 (chromatin modifier), IGF2BP (growth regulator), "
            "and directly target 3'UTR of KISS1R, LIN28A, and GNRH1 transcripts. "
            "LIN28B GOF → inhibits Dicer-mediated pre-let-7 processing → mature let-7 reduced → "
            "KISS1R, GNRH1 mRNA 3'UTR protection → increased translation → "
            "enhanced kisspeptin signalling → GnRH pulse precocious. "
            "LIN28A vs LIN28B: LIN28A is more embryonic; LIN28B more relevant to puberty timing. "
            "ANIMAL MODEL: Conditional hypothalamic Lin28b overexpression in mice → CPP; "
            "confirms mechanistic link. "
            "GWAS vs MENDELIAN: Common 6q16.3 variants near LIN28B shift menarche by 1-2 months per allele; "
            "rare coding GOF shift by years → Mendelian CPP."
        ),
        "pathognomonic": (
            "LIN28B CLINICAL PEARLS: "
            "GWAS CONTEXT IMPORTANT: Many patients have the common 6q16.3 variant (not pathogenic); "
            "request LIN28B CODING SEQUENCE specifically for Mendelian CPP investigation. "
            "FAMILY HISTORY: CPP in multiple generations through either parent (no imprinting); "
            "GOF mutation in a parent with earlier-than-average puberty. "
            "LET-7 PATHWAY: LIN28B is the 'master switch' for the let-7 suppression network; "
            "understanding this network helps explain why KISS1R, GNRH1, and other genes are "
            "co-regulated in puberty timing. "
            "TREATMENT: GnRHa identical to other CPP causes — LIN28B-mediated activation is "
            "upstream of GnRH; pituitary GnRH-R downregulation breaks the chain. "
            "OMIM Gene: 611044."
        ),
        "treatment": (
            "GnRHa (leuprolide/triptorelin/histrelin) — effective. "
            "MONITORING: Standard CPP surveillance — LH/FSH, oestradiol, bone age. "
            "NOTE: LIN28B operates via miRNA pathway — no specific miRNA therapy available; "
            "GnRHa remains the standard of care. "
            "DURATION: Until appropriate pubertal age."
        ),
        "key_features": [
            "Strongest GWAS locus for female puberty timing (6q16.3 common variants, 2009)",
            "Rare coding GOF mutations cause familial CPP with years earlier onset",
            "RNA-binding protein repressing let-7 miRNA → derepresses KISS1R, GNRH1",
            "AD GOF — no imprinting; family history through either parent",
            "Animal models confirm hypothalamic Lin28b overexpression → CPP",
            "GnRHa effective — LIN28B acts upstream of GnRH/pituitary axis",
        ],
        "key_ddx": (
            "LIN28B coding GOF vs common GWAS variant: GWAS common variant shifts puberty by weeks; "
            "coding GOF causes clinical CPP by years — distinguish via rare variant analysis; "
            "LIN28B GOF vs MKRN3: MKRN3 paternal imprinting; LIN28B no imprinting; "
            "LIN28B vs idiopathic CPP: LIN28B coding variant provides molecular diagnosis"
        ),
        "systemic_involvement": {
            "HPG_axis": "let-7 repression → KISS1R/GNRH1 derepression → precocious GnRH pulsatility",
            "growth": "Advanced bone age; early height acceleration followed by short adult stature",
        },
        "cascade_testing": (
            "AD GOF — 50% risk to offspring; test symptomatic relatives; "
            "clinical surveillance: annual pubertal staging from age 6 yr in at-risk children; "
            "common 6q16.3 variants: not actionable in isolation — do not over-report"
        ),
        "emergency_protocol": (
            "CPP in girl <8 yr or boy <9 yr with positive LIN28B GOF → "
            "same pathway as all CPP: brain MRI, hormone panel, GnRHa initiation."
        ),
    },

    # -- GNAS -- McCABE-ALBRIGHT PERIPHERAL PP ----------------------------------------
    {
        "gene": "GNAS",
        "alt_name": (
            "GNAS (GNAS-395aa-20q13.32 / Somatic-GOF-Postzygotic -- "
            "Guanine-Nucleotide-Binding-Protein-Alpha-S -- "
            "McCune-Albright-Syndrome-MAS -- "
            "PERIPHERAL-Precocious-Puberty-GnRHa-FAILS -- "
            "Fibrous-Dysplasia-Cafe-au-Lait-Coast-of-Maine-PATHOGNOMONIC)"
        ),
        "protein": (
            "GNAS -- 20q13.32 Somatic-GOF-Postzygotic -- GNAS-395aa -- "
            "Gsalpha-45kDa-Stimulatory-G-Protein-alpha-Subunit-GDP-GTP-Switch-GNRH-LH-ACTH-TSH-Receptor -- "
            "OMIM-Gene-139320-Disease-MAS-174800"
        ),
        "locus": "20q13.32",
        "protein_size": "395 aa / ~45 kDa",
        "inheritance": (
            "Somatic (postzygotic) gain-of-function — NOT germline. "
            "McCune-Albright syndrome (MAS) results from a postzygotic somatic activating mutation "
            "in GNAS (most common: p.Arg201His, p.Arg201Cys). "
            "Arg201 is the GTPase arginine — essential for intrinsic GTPase activity that inactivates Gsα. "
            "GOF MUTATION: R201H or R201C → impaired GTP hydrolysis → Gsα locked in GTP-bound active state → "
            "constitutive adenylyl cyclase activation → cAMP elevated → downstream receptor pathway activation. "
            "SOMATIC MOSAICISM: MAS is caused ONLY by postzygotic mutation (germline R201H is embryonic lethal); "
            "severity depends on timing of postzygotic mutation and proportion of affected cells. "
            "CLASSIC MAS TRIAD: (1) Precocious puberty (PERIPHERAL), (2) Fibrous dysplasia, "
            "(3) Café-au-lait macules (irregular 'Coast of Maine' borders). "
            "PERIPHERAL PRECOCIOUS PUBERTY: Ovarian cysts (girls) or testicular lesions (boys) produce "
            "sex steroids AUTONOMOUSLY without LH/FSH stimulation; "
            "GnRHa is INEFFECTIVE because the sex steroid production bypasses the GnRH/pituitary axis. "
            "CENTRAL CPP SUPERIMPOSED: Long-standing GNAS-mediated sex steroid exposure can eventually "
            "trigger central CPP secondarily — then GnRHa becomes partially helpful for the CENTRAL component. "
            "DIAGNOSIS: Sequencing of affected tissue (not blood WBC) for R201H/C; "
            "blood WBC DNA may be negative due to mosaic distribution."
        ),
        "disease_category": (
            "McCune-Albright Syndrome (MAS, OMIM #174800); "
            "somatic GOF GNAS; 20q13.32; "
            "PERIPHERAL precocious puberty (ovarian cysts/testicular lesions autonomous); "
            "fibrous dysplasia + café-au-lait; GnRHa FAILS for peripheral component"
        ),
        "disease_pathway": (
            "GNAS R201H/C → CONSTITUTIVE cAMP SIGNALLING: "
            "Normal: LH binds LH receptor (Gsα-coupled) → GTP loading of Gsα → adenylyl cyclase → "
            "cAMP → PKA → StAR/steroidogenesis → sex steroids → negative feedback → GTPase inactivation. "
            "GNAS R201H/C GOF: GTPase inactivated → Gsα locked GTP-on → adenylyl cyclase ALWAYS ON → "
            "cAMP permanently elevated → StAR/CYP19A1 constitutively active → "
            "autonomous ovarian oestrogen (or testicular testosterone) → peripheral PP. "
            "FIBROUS DYSPLASIA: GNAS GOF in osteoblastic lineage → excess cAMP → "
            "impaired differentiation → fibrous tissue replaces bone; "
            "RANKIN distribution follows mosaic pattern. "
            "CAFE-AU-LAIT: Irregular 'Coast of Maine' borders (vs smooth 'Coast of California' in NF1); "
            "melanocyte GNAS GOF → excess cAMP → melanin production → large unilateral macules. "
            "ENDOCRINOPATHIES: Hyperthyroidism, acromegaly, Cushing — all via autonomous Gsα activation "
            "in thyroid/pituitary/adrenal."
        ),
        "pathognomonic": (
            "GNAS MAS CLINICAL PEARLS: "
            "CAFE-AU-LAIT BORDERS: 'Coast of Maine' (jagged/irregular) in MAS; "
            "'Coast of California' (smooth) in NF1 — do NOT confuse. "
            "GnRHa FAILS: This is the single most important clinical rule for MAS precocious puberty; "
            "GnRHa suppresses pituitary LH/FSH but the ovarian cyst produces oestrogen independently; "
            "GnRHa alone will NOT stop bone age advancement in MAS. "
            "TREATMENT: Aromatase inhibitors (letrozole, anastrozole, testolactone) + ketoconazole; "
            "letrozole preferred in girls; fulvestrant (ER antagonist) emerging. "
            "SECONDARY CPP: After years of sex steroid exposure, the hypothalamus may 'turn on' centrally; "
            "then GnRHa CAN help for the CENTRAL component layered on top of peripheral. "
            "BONE MANAGEMENT: Bisphosphonates (pamidronate) for fibrous dysplasia; "
            "orthopaedic referral for long bone deformity. "
            "MOSAIC DETECTION: Blood WBC DNA may miss R201H/C (low mosaic fraction); "
            "request affected tissue biopsy (bone, cyst wall) if blood negative. "
            "OMIM Gene: 139320, Disease: MAS — 174800."
        ),
        "treatment": (
            "PERIPHERAL CPP (ovarian cysts): Aromatase inhibitors — letrozole 2.5 mg/day (girls); "
            "testolactone (historical, less used); fulvestrant (ER antagonist, emerging). "
            "KETOCONAZOLE: CYP17/CYP11B1 inhibitor — blocks adrenal and ovarian steroidogenesis; "
            "liver toxicity monitoring required (LFTs). "
            "GnRHa: NOT effective for primary peripheral CPP — "
            "ONLY use if secondary central CPP has developed (confirmed by LH response >5 IU/L to GnRH test). "
            "FIBROUS DYSPLASIA: Pamidronate IV every 6 months; "
            "orthopaedic review; vitamin D + calcium; avoid heavy impact sports. "
            "HYPERTHYROIDISM: Methimazole if GNAS GOF causes autonomous thyroid activity. "
            "MONITORING: Pelvic USS q3-6 monthly (ovarian cysts); bone age annually; "
            "LFTs if ketoconazole; bisphosphonate response (alkaline phosphatase)."
        ),
        "key_features": [
            "McCune-Albright Syndrome — somatic postzygotic GNAS GOF (R201H or R201C most common)",
            "PERIPHERAL precocious puberty — autonomous ovarian cysts/testicular lesions",
            "GnRHa IS INEFFECTIVE for peripheral component — aromatase inhibitor is treatment",
            "Café-au-lait 'Coast of Maine' jagged borders — PATHOGNOMONIC (vs NF1 smooth borders)",
            "Fibrous dysplasia — cAMP excess in osteoblastic lineage → bone replaced by fibrous tissue",
            "Somatic mosaic — blood DNA may be negative; test affected tissue",
            "Secondary central CPP can develop after years of sex steroid exposure; GnRHa then partially helpful",
        ],
        "key_ddx": (
            "MAS peripheral CPP vs central CPP: GnRH stimulation test — "
            "peripheral PP: LH response <5 IU/L (prepubertal); central CPP: LH >5 IU/L (pubertal); "
            "MAS vs NF1: café-au-lait borders — jagged (MAS) vs smooth (NF1); "
            "MAS vs MKRN3/DLK1: MKRN3/DLK1 → central CPP, GnRHa works; MAS → peripheral CPP, GnRHa fails; "
            "MAS vs isolated ovarian cysts: MAS is mosaic systemic disease with other features"
        ),
        "systemic_involvement": {
            "ovary": "Autonomous oestrogen-producing cysts — peripheral PP mechanism",
            "bone": "Fibrous dysplasia — fracture risk; deformity; pain",
            "skin": "Coast of Maine café-au-lait macules — unilateral, large, irregular",
            "thyroid": "Autonomous nodules/hyperthyroidism in 20%",
            "pituitary": "Acromegaly if somatotroph GNAS GOF",
            "adrenal": "Cushing syndrome if adrenocortical GNAS GOF",
        },
        "cascade_testing": (
            "NOT germline — no family testing needed; "
            "MAS is somatic mosaic — children of MAS patients are NOT at increased risk; "
            "GENETIC COUNSELLING: Reassure parents MAS will NOT be passed to offspring"
        ),
        "emergency_protocol": (
            "OVARIAN TORSION: Autonomous ovarian cysts can enlarge rapidly → "
            "acute pelvic pain in girl with MAS → urgent USS → torsion = surgical emergency; "
            "FRACTURE: Fibrous dysplasia → pathological fracture; "
            "orthopaedic emergency management; bisphosphonate intensification post-fracture; "
            "ACUTE HYPERTHYROIDISM: antithyroid drug + beta-blocker if GNAS thyroid involvement."
        ),
    },

    # -- LEPR -- LEPTIN RECEPTOR / OBESITY + CPP ----------------------------------------
    {
        "gene": "LEPR",
        "alt_name": (
            "LEPR (LEPR-1165aa-1p31.3 / AR-LOF -- "
            "Leptin-Receptor-OB-R -- "
            "Severe-Early-Onset-Obesity -- "
            "CPP-After-Leptin-Pathway-Restoration -- "
            "Leptin-Permissive-Signal-for-Puberty)"
        ),
        "protein": (
            "LEPR -- 1p31.3 AR-LOF -- LEPR-1165aa -- "
            "Leptin-Receptor-132kDa-Class-I-Cytokine-Receptor-JAK2-STAT3-Signalling -- "
            "Hypothalamic-Arcuate-Nucleus-Energy-Balance-Reproduction-Master-Sensor -- "
            "OMIM-Gene-601007-Disease-Obesity-614963-Hypogonadotropic-Hypogonadism"
        ),
        "locus": "1p31.3",
        "protein_size": "1165 aa / ~132 kDa",
        "inheritance": (
            "Autosomal recessive loss-of-function (biallelic). "
            "Leptin receptor (LEPR/OB-R) mediates leptin signalling in the hypothalamus. "
            "LEPR LOF → inability to sense adiposity → hypothalamus perceives starvation → "
            "HYPERPHAGIA + severe obesity (body weight 150-200 kg by adulthood); "
            "delayed/absent puberty (leptin is permissive for GnRH axis). "
            "CPP CONNECTION: When LEPR-deficient patients receive metreleptin (leptin replacement) "
            "or if weight loss normalises adipokine signalling → rapid onset of PUBERTY/CPP — "
            "the precocious puberty in LEPR deficiency is the puberty that was arrested, "
            "triggered once the leptin signal is restored. "
            "HYPOGONADOTROPIC HYPOGONADISM: Primary phenotype of untreated LEPR deficiency; "
            "LH/FSH low; absent puberty; infertility. "
            "LEPTIN AS PUBERTY SIGNAL: Fat mass → leptin secretion → LEPR in arcuate nucleus → "
            "KISS1 upregulation → GnRH → puberty; "
            "this is why weight gain in early childhood is associated with earlier puberty in population studies. "
            "FREQUENCY: LEPR deficiency rare; Kabyle Algerian founder (p.Gln223Arg); "
            "Egyptian, Pakistani, Turkish founder mutations also described."
        ),
        "disease_category": (
            "Congenital Leptin Receptor Deficiency (OMIM #614963); "
            "AR biallelic LOF; 1p31.3; "
            "severe early-onset obesity + hypogonadotropic hypogonadism; "
            "CPP can occur as rebound when leptin pathway restored (metreleptin/bariatric); "
            "metreleptin treatment highly effective for metabolism but triggers rapid puberty onset"
        ),
        "disease_pathway": (
            "LEPR LOF → FALSE STARVATION SIGNAL → HPG AXIS SUPPRESSION: "
            "Leptin binds LEPR (long isoform, LEPRb) in arcuate nucleus → JAK2 → STAT3 → "
            "KISS1 upregulation in KNDy neurons → GnRH stimulation → puberty. "
            "LOF → JAK2-STAT3 not activated → KISS1 not upregulated → GnRH pulse suppressed → "
            "hypogonadotropic hypogonadism (primary phenotype). "
            "OBESITY PATHWAY: Leptin-LEPR signals satiety; LOF → ARC NPY/AgRP neurons unrestrained → "
            "hyperphagia → severe obesity → adipokine dysregulation. "
            "METRELEPTIN REBOUND CPP: Recombinant leptin (metreleptin) restores hypothalamic leptin signal → "
            "KISS1 surges → GnRH pulses → LH/FSH surge → rapid pubertal development → "
            "CPP occurs as consequence of treatment. "
            "GnRHa REQUIRED: When metreleptin-treated LEPR patients develop CPP → GnRHa to control tempo."
        ),
        "pathognomonic": (
            "LEPR CLINICAL PEARLS: "
            "HYPERPHAGIA IS PATHOGNOMONIC: Constant, insatiable hunger from infancy; "
            "family cannot keep food away; eating everything available; distinguishes from simple obesity. "
            "ABSENT PUBERTY: Most LEPR-deficient adolescents have NO pubertal development — "
            "hormonal panel: LH <0.5, FSH <1.0, oestradiol/testosterone prepubertal; "
            "this is hypogonadotropic hypogonadism, NOT primary gonadal failure. "
            "IMMUNE DYSFUNCTION: LEPR expressed on immune cells → recurrent respiratory infections; "
            "IL-6 and TNF-α signalling disrupted. "
            "METRELEPTIN CPP: Warn families: 'When we start leptin treatment, puberty will begin — "
            "sometimes rapidly'; GnRHa may be needed alongside metreleptin. "
            "METRELEPTIN PRESCRIBING: FDA-approved for generalised lipodystrophy; "
            "off-label for LEPR deficiency in specialist centres; cost significant. "
            "OMIM Gene: 601007, Disease: LEPR Deficiency — 614963."
        ),
        "treatment": (
            "METRELEPTIN (recombinant methionyl leptin): SC daily injection; "
            "reduces hyperphagia → weight loss; triggers puberty onset. "
            "GnRHa: When metreleptin causes rapid CPP → add leuprolide/triptorelin to control puberty tempo. "
            "BARIATRIC SURGERY: For extreme obesity if metreleptin unavailable; "
            "weight loss may also restore partial leptin signalling. "
            "SEX HORMONE REPLACEMENT: If puberty absent without metreleptin → "
            "testosterone (males) or oestrogen (females) induction from ~13 yr; "
            "continue as HRT. "
            "MONITORING: Weight, BMI, waist circumference; LH/FSH if on metreleptin; "
            "pubertal staging (if metreleptin started); bone density (obesity + hypogonadism both reduce BMD)."
        ),
        "key_features": [
            "AR biallelic LOF — severe early-onset obesity from infancy + pathognomonic hyperphagia",
            "Primary phenotype: hypogonadotropic hypogonadism (absent puberty)",
            "CPP occurs as treatment rebound when metreleptin or weight loss restores leptin signal",
            "Metreleptin (recombinant leptin) highly effective for obesity and triggers puberty",
            "GnRHa may be needed alongside metreleptin to control CPP tempo",
            "Demonstrates leptin as essential permissive signal for puberty onset",
            "Kabyle Algerian founder (p.Gln223Arg); other founders in Egyptian/Pakistani/Turkish",
        ],
        "key_ddx": (
            "LEPR LOF vs LEP LOF (leptin deficiency): both AR, both severe obesity + absent puberty; "
            "LEPR: leptin levels HIGH (receptor absent); LEP: leptin levels VERY LOW (leptin absent); "
            "LEPR vs simple obesity CPP: simple obesity → higher leptin via normal LEPR → earlier puberty "
            "(common); LEPR LOF → absent puberty (opposite — leptin signal absent); "
            "LEPR vs MC4R LOF: MC4R → obesity without gonadal involvement; leptin levels elevated in both"
        ),
        "systemic_involvement": {
            "obesity": "Severe early-onset hyperphagia-driven obesity; BMI >40 by adolescence",
            "immunity": "Recurrent respiratory infections; LEPR on immune cells",
            "HPG_axis": "Hypogonadotropic hypogonadism (primary); CPP as treatment rebound",
            "bone": "Low BMD from hypogonadism + obesity paradox",
        },
        "cascade_testing": (
            "AR — parental carriers; 25% sibling risk; "
            "founder communities (Kabyle Algeria, Pakistan, Egypt): targeted founder variant testing; "
            "newborn: no NBS for LEPR; watch for hyperphagia + rapid weight gain infancy → test early"
        ),
        "emergency_protocol": (
            "HYPOGLYCAEMIA: Rare in LEPR (not a metabolic enzyme defect); "
            "RESPIRATORY CRISIS: Recurrent chest infections in obese LEPR patient → "
            "treat infection; consider immune dysfunction; "
            "METRELEPTIN RAPID CPP: When family reports sudden pubertal changes after metreleptin start → "
            "urgent paediatric endocrinology; confirm CPP with LH/FSH; add GnRHa promptly."
        ),
    },

    # -- GNRH1 -- ACTIVATING MUTATIONS -------------------------------------------------
    {
        "gene": "GNRH1",
        "alt_name": (
            "GNRH1 (GNRH1-92aa-8p21.2 / AD-activating-GOF -- "
            "Gonadotropin-Releasing-Hormone-1 -- "
            "GnRH-Hyperpulse-Drives-Precocious-HPG-Axis -- "
            "Rare-~3-5pct-Familial-CPP -- "
            "Same-Gene-as-Isolated-IHH-LOF)"
        ),
        "protein": (
            "GNRH1 -- 8p21.2 AD-activating -- GNRH1-92aa -- "
            "GnRH-Decapeptide-pGlu-His-Trp-Ser-Tyr-Gly-Leu-Arg-Pro-Gly-NH2-10kDa-Precursor -- "
            "Hypothalamic-Magnocellular-Arcuate-Neurons-Master-Reproductive-Hormone -- "
            "OMIM-Gene-152760"
        ),
        "locus": "8p21.2",
        "protein_size": "92 aa / ~10 kDa",
        "inheritance": (
            "Autosomal dominant activating gain-of-function (CPP) OR autosomal recessive LOF (isolated IHH). "
            "GNRH1 is processed from a 92-aa precursor to the active decapeptide "
            "(pGlu-His-Trp-Ser-Tyr-Gly-Leu-Arg-Pro-Gly-NH2). "
            "GOF MUTATIONS (CPP): Activating GNRH1 variants → "
            "increased GnRH production, secretion, or receptor binding affinity → "
            "precocious GnRH pulsatility → LH/FSH surge → sex steroids → CPP. "
            "FREQUENCY: Rare cause of CPP — ~3-5% of familial CPP; "
            "GNRH1 sequencing should be included in panel for unexplained CPP. "
            "LOF CONTRAST: Biallelic GNRH1 LOF → absent GnRH secretion → "
            "isolated hypogonadotropic hypogonadism (IHH without anosmia — not Kallmann); "
            "anosmia absent because GnRH neurons develop and migrate correctly in GNRH1 LOF "
            "(contrast with ANOS1/FGFR1 Kallmann where neuron migration fails). "
            "PRECURSOR PROCESSING: GnRH precursor → signal peptide cleavage → "
            "GnRH decapeptide + GnRH-associated peptide (GAP) — "
            "GOF mutations may affect processing, secretion rate, or receptor affinity. "
            "MOLECULAR TESTING: Ensure GNRH1 is on the CPP gene panel; "
            "commonly included alongside MKRN3, DLK1, KISS1, KISS1R in hereditary CPP panels."
        ),
        "disease_category": (
            "Hereditary Central Precocious Puberty (GNRH1 GOF-CPP); "
            "AD activating mutations; 8p21.2; rare (~3-5% familial CPP); "
            "same gene causes isolated IHH (LOF) — opposite phenotype; "
            "GnRHa highly effective"
        ),
        "disease_pathway": (
            "GNRH1 GOF → HYPERPULSATILE GnRH → PRECOCIOUS HPG AXIS: "
            "Normal GnRH pulse: 1 pulse/60-90 min (luteal phase) to 1 pulse/90-120 min (follicular); "
            "prepubertally: GnRH pulse quiescent (MKRN3 and other brakes active). "
            "GNRH1 GOF activating mutation → "
            "increased GnRH synthesis/secretion rate or increased receptor affinity → "
            "GnRH pulses prematurely activated → pituitary gonadotropes stimulated → "
            "LH/FSH surge → gonadal sex steroids → CPP. "
            "GnRHa MECHANISM: Exogenous GnRH analogue (long-acting, continuous) → "
            "pituitary GnRH receptor downregulation (loss of pulsatile stimulation pattern) → "
            "LH/FSH secretion suppressed → sex steroid suppression → puberty arrested. "
            "CONTRAST WITH LOF: GNRH1 LOF → no GnRH → no LH/FSH → IHH; "
            "these patients require GnRH pump (pulsatile GnRH) for fertility."
        ),
        "pathognomonic": (
            "GNRH1 CLINICAL PEARLS: "
            "SAME GENE, OPPOSITE PHENOTYPES — AGAIN: GNRH1 GOF → CPP; GNRH1 LOF → IHH; "
            "this is the third example in this atlas (alongside KISS1 and KISS1R) of the same gene "
            "causing CPP via GOF and IHH via LOF — important for understanding HPG axis biology. "
            "PANEL TESTING: GNRH1 should be on any hereditary CPP panel; "
            "it is less commonly positive than MKRN3/DLK1 but captures ~3-5% of familial cases. "
            "SMELL NORMAL: GNRH1 LOF causes IHH WITHOUT anosmia (unlike ANOS1/FGFR1 Kallmann); "
            "if CPP patient has a sibling with IHH + NORMAL smell → test GNRH1 first. "
            "GnRHa RESPONSE: Excellent — pituitary receptor downregulation suppresses LH/FSH; "
            "GnRH hyperpulse from GOF is irrelevant once receptor is downregulated at pituitary. "
            "OMIM Gene: 152760."
        ),
        "treatment": (
            "GnRHa (leuprolide/triptorelin/histrelin) — highly effective. "
            "MONITORING: Standard CPP monitoring — LH/FSH, oestradiol, bone age. "
            "FERTILITY: GnRHa treatment does not impair future fertility; "
            "ovarian reserve and spermatogenesis preserved. "
            "DURATION: Until appropriate chronological pubertal age."
        ),
        "key_features": [
            "Rare (~3-5% familial CPP) — include on hereditary CPP panel alongside MKRN3/DLK1",
            "AD activating GOF (CPP) vs AR LOF (isolated IHH without anosmia) — same gene",
            "Third example in this atlas of same gene causing CPP (GOF) and IHH (LOF)",
            "GnRH decapeptide from 92-aa precursor; activating mutations → hyperpulsatile GnRH",
            "GnRHa pituitary receptor downregulation is highly effective",
            "GNRH1 LOF patients in family may have IHH with normal smell (not Kallmann anosmia)",
        ],
        "key_ddx": (
            "GNRH1 GOF (CPP) vs GNRH1 LOF (IHH): opposite phenotypes; sequence the gene and test function; "
            "GNRH1 vs KISS1/KISS1R GOF: all cause central CPP; GNRH1 acts further downstream; "
            "GNRH1 GOF vs MKRN3/DLK1: MKRN3/DLK1 are imprinted; GNRH1 GOF is standard AD; "
            "GNRH1 IHH vs Kallmann (ANOS1/FGFR1): GNRH1 IHH has NORMAL anosmia; Kallmann has anosmia"
        ),
        "systemic_involvement": {
            "HPG_axis": "Hyperpulsatile GnRH → premature LH/FSH surge → sex steroid production",
        },
        "cascade_testing": (
            "AD GOF — 50% offspring risk; family testing for IHH-affected relatives "
            "(may carry same allele as LOF if compound heterozygous); "
            "clinical monitoring: pubertal staging annually from age 6 yr in at-risk children"
        ),
        "emergency_protocol": (
            "CPP in girl <8 yr or boy <9 yr with positive GNRH1 GOF → "
            "brain MRI (exclude structural cause); GnRHa initiation without delay."
        ),
    },
]


def _make_cohort(gene_data: dict, seed: int, n: int = 40) -> list:
    r = random.Random(seed)
    gene = gene_data["gene"]

    pres_map = {
        "MKRN3": [
            "CPP_breast_before_8yr_paternal_family_history",
            "advanced_bone_age_MKRN3_female_father_CPP",
            "bilateral_breast_development_6yr_LH_pubertal",
            "testicular_enlargement_8yr_male_MKRN3",
            "MKRN3_frameshift_familial_CPP_three_generations",
        ],
        "DLK1": [
            "CPP_breast_7yr_maternal_grandmother_CPP",
            "advanced_bone_age_DLK1_adiposity_elevated_BMI",
            "DLK1_LOF_maternal_inheritance_early_menarche",
            "precocious_puberty_maternal_transmission_two_sisters",
            "DLK1_14q32_maternal_variant_girl_CPP",
        ],
        "KISS1": [
            "extreme_early_CPP_2yr_breast_R73C_KISS1_GOF",
            "CPP_4yr_Argentine_family_KISS1_GOF_compound_het",
            "very_early_bone_age_advanced_5yr_KISS1_mutation",
            "de_novo_KISS1_GOF_CPP_onset_under_3yr",
            "bilateral_early_thelarche_kisspeptin_elevated",
        ],
        "KISS1R": [
            "CPP_1yr_breast_A243V_KISS1R_Syrian_girl",
            "CPP_2yr_de_novo_KISS1R_GOF_TM6_mutation",
            "precocious_puberty_earliest_onset_KISS1R_GOF",
            "GPR54_constitutive_activation_CPP_LH_elevated",
            "family_CPP_AD_KISS1R_GOF_father_daughter",
        ],
        "LIN28B": [
            "familial_CPP_maternal_paternal_LIN28B_GOF",
            "LIN28B_coding_GOF_early_menarche_10yr",
            "LIN28B_let7_repression_precocious_axis_activation",
            "GWAS_locus_LIN28B_coding_variant_CPP_diagnosis",
            "two_sisters_CPP_LIN28B_GOF_family",
        ],
        "GNAS": [
            "MAS_peripheral_PP_GnRHa_failed_ovarian_cyst",
            "fibrous_dysplasia_cafe_au_lait_CPP_McCune_Albright",
            "R201H_GNAS_somatic_mosaic_unilateral_fibrous_dysplasia",
            "peripheral_precocious_puberty_aromatase_inhibitor_started",
            "MAS_acromegaly_hyperthyroidism_CPP_multiple_endocrinopathy",
        ],
        "LEPR": [
            "severe_obesity_infancy_hyperphagia_absent_puberty_LEPR",
            "metreleptin_started_rapid_CPP_onset_LEPR_deficiency",
            "Kabyle_Algerian_founder_Q223R_LEPR_severe_obesity",
            "LEPR_LOF_absent_LH_FSH_extreme_BMI_40",
            "CPP_as_rebound_after_bariatric_surgery_LEPR",
        ],
        "GNRH1": [
            "familial_CPP_GNRH1_GOF_sister_IHH_normal_smell",
            "AD_GNRH1_activating_variant_CPP_3_generations",
            "GNRH1_GOF_GnRHa_excellent_response_CPP",
            "de_novo_GNRH1_GOF_early_puberty_7yr",
            "CPP_panel_GNRH1_GOF_no_structural_cause_MRI",
        ],
    }

    mgmt_map = {
        "MKRN3": [
            "GnRHa_leuprolide_depot_3.75mg_q28d",
            "histrelin_implant_annual_bone_age_monitoring",
            "triptorelin_depot_LH_FSH_prepubertal_suppressed",
            "GnRHa_bone_age_normal_height_potential_preserved",
        ],
        "DLK1": [
            "GnRHa_leuprolide_BMI_monitoring_DLK1",
            "GnRHa_triptorelin_adiposity_dietitian_referral",
            "histrelin_implant_maternal_family_counselling",
            "GnRHa_DLK1_bone_density_DXA_monitoring",
        ],
        "KISS1": [
            "GnRHa_leuprolide_KISS1_GOF_extreme_early_CPP",
            "histrelin_implant_2yr_onset_KISS1_GOF",
            "GnRHa_triptorelin_bone_age_monitoring_KISS1",
            "GnRHa_very_early_start_KISS1_GOF_psychosocial_support",
        ],
        "KISS1R": [
            "GnRHa_leuprolide_A243V_KISS1R_CPP",
            "histrelin_implant_earliest_onset_KISS1R",
            "GnRHa_triptorelin_de_novo_KISS1R_GOF",
            "GnRHa_KISS1R_functional_assay_confirmed_GOF",
        ],
        "LIN28B": [
            "GnRHa_leuprolide_LIN28B_GOF_standard_CPP",
            "GnRHa_triptorelin_family_counselled_LIN28B",
            "histrelin_implant_LIN28B_let7_pathway_CPP",
            "GnRHa_LIN28B_coding_GOF_menarche_delayed",
        ],
        "GNAS": [
            "letrozole_aromatase_inhibitor_MAS_peripheral_PP",
            "ketoconazole_GNAS_MAS_ovarian_cysts_aromatase_inhibitor",
            "GnRHa_added_secondary_central_CPP_MAS",
            "bisphosphonate_pamidronate_fibrous_dysplasia_letrozole_CPP",
        ],
        "LEPR": [
            "metreleptin_started_GnRHa_added_CPP_rebound",
            "bariatric_surgery_weight_loss_GnRHa_CPP_LEPR",
            "sex_hormone_induction_testosterone_absent_puberty_LEPR",
            "oestrogen_induction_LEPR_deficiency_absent_puberty_female",
        ],
        "GNRH1": [
            "GnRHa_leuprolide_GNRH1_GOF_excellent_response",
            "histrelin_implant_GNRH1_GOF_CPP",
            "GnRHa_triptorelin_GNRH1_GOF_bone_age_normal",
            "GnRHa_GNRH1_sister_IHH_GnRH_pump_fertility",
        ],
    }

    age_ranges = {
        "MKRN3": (5, 8),
        "DLK1":  (5, 8),
        "KISS1": (2, 5),
        "KISS1R":(1, 4),
        "LIN28B":(6, 8),
        "GNAS":  (2, 7),
        "LEPR":  (0, 3),   # infancy onset obesity; puberty absent or delayed
        "GNRH1": (5, 8),
    }

    pres_list = pres_map.get(gene, ["hereditary_CPP_presentation"])
    mgmt_list = mgmt_map.get(gene, ["GnRHa_standard_treatment"])
    age_min, age_max = age_ranges.get(gene, (4, 8))

    patients = []
    for i in range(n):
        age_dx = r.randint(age_min, age_max)
        age_curr = age_dx + r.randint(1, 15)
        patients.append({
            "id": f"{gene}-{seed}-{i+1:02d}",
            "gene": gene,
            "age_at_diagnosis": age_dx,
            "age_current": min(age_curr, 35),
            "presentation": r.choice(pres_list),
            "management": r.choice(mgmt_list),
            "outcome": r.choice([
                "puberty_controlled_GnRHa",
                "bone_age_normalised",
                "adult_height_preserved",
                "ongoing_GnRHa_surveillance",
                "puberty_resumed_appropriate_age",
            ]),
        })
    return patients


def get_overview() -> dict:
    cohorts = []
    for i, g in enumerate(CPP_GENES):
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
        "atlas": "Hereditary-Precocious-Puberty-Atlas",
        "subtitle": (
            "Complete 8-Gene Reference — MKRN3 · DLK1 · KISS1 · KISS1R · LIN28B · GNAS · LEPR · GNRH1"
        ),
        "total_patients": total_patients,
        "genes_covered": len(CPP_GENES),
        "seed_range": f"{SEED_BASE}–{SEED_BASE + len(CPP_GENES) - 1}",
        "cohort_size_per_gene": 40,
        "domain": (
            "Hereditary Precocious Puberty — Central (GnRHa-Responsive) and Peripheral (McCune-Albright); "
            "Imprinting Disorders (MKRN3/DLK1), Kisspeptin Axis (KISS1/KISS1R), "
            "RNA Biology (LIN28B), Somatic Mosaic (GNAS/MAS), Metabolic (LEPR), Neuropeptide (GNRH1)"
        ),
        "key_diagnostic_tests": [
            "GnRH stimulation test (Buserelin/leuprolide IV): LH >5 IU/L = central CPP",
            "LH, FSH, oestradiol (girls) / testosterone (boys) — basal + stimulated",
            "Bone age X-ray (left hand/wrist Greulich-Pyle) — advanced ≥2 SD = CPP",
            "Pelvic USS: uterine length >3.4 cm = oestrogenised; ovarian follicles/cysts",
            "Brain MRI (1.5-3T with hypothalamic-pituitary protocol) — hamartoma/tumour exclusion MANDATORY",
            "MKRN3 sequencing (+ MLPA) — first-line in familial CPP with paternal history",
            "DLK1 sequencing (+ MLPA) — second-line in familial CPP with maternal history",
            "KISS1 + KISS1R sequencing — early-onset (<4 yr) or de novo CPP",
            "GNAS R201H/C tissue testing (NOT blood WBC) — if MAS features (café-au-lait, fibrous dysplasia)",
            "Leptin levels — if severe early-onset obesity + absent/delayed puberty (LEPR deficiency)",
            "Testicular/ovarian USS — peripheral precocious puberty assessment",
        ],
        "key_emergency_rules": [
            "MAS (GNAS): GnRHa DOES NOT WORK for peripheral PP — start aromatase inhibitor; GnRHa only if secondary central CPP develops",
            "GNAS/MAS ovarian cyst: can torsion → acute pelvic pain = surgical emergency",
            "KISS1/KISS1R GOF: Brain MRI mandatory even with positive mutation — hamartoma must be excluded",
            "LEPR + metreleptin: Warn family rapid CPP onset expected; have GnRHa prescription ready",
            "ALL CPP: Brain MRI BEFORE starting GnRHa — do not treat without excluding structural cause",
            "MKRN3 familial CPP: Annual puberty monitoring of daughters from age 6 yr (affected father)",
            "Bone age advanced: If >3 SD above chronological age → start GnRHa immediately",
        ],
        "cohort_breakdown": cohorts,
    }


def get_breakdown() -> dict:
    breakdown_by_gene = {}
    for i, g in enumerate(CPP_GENES):
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
    return {"breakdown_by_gene": breakdown_by_gene, "total_genes": len(CPP_GENES)}


def get_definitions() -> dict:
    return {
        "atlas_domain": "Hereditary-Precocious-Puberty-Atlas — Complete 8-Gene Central & Peripheral Reference",
        "key_definitions": {
            "Central_Precocious_Puberty_CPP": (
                "Premature activation of the hypothalamic-pituitary-gonadal (HPG) axis; "
                "breast development <8 yr (girls) or testicular volume ≥4 mL / pubic hair <9 yr (boys); "
                "LH response >5 IU/L after GnRH stimulation confirms central origin; "
                "most common cause: idiopathic (structural/organic excluded); "
                "second most common: hypothalamic hamartoma (MRI: isointense sessile lesion at tuber cinereum); "
                "monogenic hereditary causes: MKRN3, DLK1, KISS1, KISS1R, LIN28B, GNRH1; "
                "treatment: GnRH agonist (GnRHa) — pituitary receptor downregulation."
            ),
            "Peripheral_Precocious_Puberty": (
                "Sex steroid production INDEPENDENT of GnRH/pituitary stimulation; "
                "LH response to GnRH stimulation PREPUBERTAL (<5 IU/L) — pituitary is not activated; "
                "cause: autonomous steroid production (ovarian cyst, testicular tumour, McCune-Albright/GNAS, "
                "adrenal tumour, exogenous steroid exposure); "
                "GnRHa is INEFFECTIVE for peripheral PP (pituitary not the problem); "
                "treatment depends on cause: aromatase inhibitor (MAS/ovarian), ketoconazole, "
                "lesion removal if tumour."
            ),
            "GnRH_Stimulation_Test": (
                "Diagnostic gold standard for CPP; "
                "IV or SC buserelin/leuprolide 100 mcg → LH + FSH at 0, 30, 60 min; "
                "PUBERTAL RESPONSE: LH peak >5 IU/L (immunofluorometric assay); "
                "PREPUBERTAL/PERIPHERAL: LH peak <5 IU/L; "
                "LH:FSH ratio >1 in central CPP (LH predominance in puberty); "
                "oestradiol >110 pmol/L or testosterone >1.7 nmol/L supports pubertal status."
            ),
            "Paternally_Imprinted_Maternally_Silenced": (
                "Genomic imprinting: epigenetic silencing of one parental allele; "
                "PATERNALLY IMPRINTED = maternal allele active, paternal allele silenced; "
                "confusing terminology: 'imprinted' means 'silenced' (opposite to intuition); "
                "MKRN3: MATERNALLY SILENCED (so only PATERNAL mutation causes CPP); "
                "DLK1: MATERNALLY SILENCED (paternally expressed — so maternal LOF causes CPP); "
                "CLINICALLY: Ask which parent is affected / who has the mutation."
            ),
            "Hypothalamic_Hamartoma": (
                "Sessile non-neoplastic mass at tuber cinereum/mammillary bodies; "
                "isointense to grey matter on T1/T2 MRI; "
                "contains GnRH-producing neurons — intrinsic GnRH pulse from the lesion; "
                "causes central CPP; also associated with gelastic (laughing) seizures; "
                "treatment: GnRHa for CPP component; stereotactic radiosurgery/laser interstitial "
                "therapy for seizure-causing hamartoma; "
                "BRAIN MRI IS MANDATORY in all CPP before attributing to genetic cause."
            ),
            "GnRH_Agonist_Treatment": (
                "Mechanism: Continuous (non-pulsatile) GnRH receptor stimulation → "
                "receptor downregulation and desensitisation → LH/FSH suppression → "
                "sex steroid reduction → puberty arrested; "
                "agents: leuprolide acetate (Lupron) 3.75 mg IM q28d or 11.25 mg q84d; "
                "triptorelin (Decapeptyl) 3.75 mg IM q28d; histrelin implant 50 mg SC annually; "
                "monitoring: LH/FSH nadir at 60 min post-GnRHa <4 IU/L = adequate suppression; "
                "bone age annually; Tanner staging 6-monthly; DXA at start and after 2 yr."
            ),
            "McCune_Albright_Syndrome_MAS": (
                "Somatic mosaic GNAS GOF (R201H or R201C); "
                "triad: peripheral precocious puberty + fibrous dysplasia + café-au-lait macules; "
                "café-au-lait borders: irregular 'Coast of Maine' (vs smooth 'Coast of California' in NF1); "
                "fibrous dysplasia: cAMP excess in osteoblasts → fibrous replacement of bone; "
                "precocious puberty: autonomous ovarian/testicular steroid → GnRHa FAILS; "
                "endocrinopathies: hyperthyroidism (~20%), acromegaly, Cushing also possible; "
                "treatment: letrozole (aromatase inhibitor) for PP; pamidronate for fibrous dysplasia."
            ),
            "Kisspeptin_GnRH_Axis": (
                "Kisspeptin (KISS1 gene product) → KISS1R/GPR54 → Gq-IP3 → GnRH neuron activation; "
                "arcuate KNDy neurons: Kisspeptin + Neurokinin B + Dynorphin — pulse generator; "
                "AVPV neurons: kisspeptin → LH surge (females only); "
                "MKRN3 normally suppresses KISS1 expression — LOF removes brake → precocious kisspeptin; "
                "KISS1 GOF: kisspeptin itself resists cleavage → prolonged action; "
                "KISS1R GOF: constitutive receptor activation without kisspeptin; "
                "ALL GnRHa-responsive: GnRHa acts downstream at pituitary regardless of upstream cause."
            ),
            "Bone_Age_Interpretation": (
                "X-ray left hand/wrist compared to Greulich-Pyle atlas; "
                "NORMAL: bone age = chronological age ± 1 SD (~1 yr); "
                "ADVANCED: bone age >2 yr ahead of chronological age = significant (CPP concern); "
                "SEVERELY ADVANCED: bone age >3 yr ahead = immediate action; "
                "sex steroids accelerate epiphyseal ossification → premature fusion → "
                "height velocity accelerates initially then stops (final height reduced); "
                "GnRHa preserves ~4-9 cm final height if started when bone age <12 yr (girls)."
            ),
            "Aromatase_Inhibitor_in_MAS": (
                "Letrozole (2.5 mg/day) or anastrozole — competitive aromatase (CYP19A1) inhibitor; "
                "blocks oestrogen synthesis from androgen precursors in ovaries/adrenals/fat; "
                "reduces autonomous ovarian oestradiol in MAS → slows bone age advancement; "
                "does NOT fully stop MAS PP (GNAS-mediated cyst autonomy continues); "
                "combined letrozole + GnRHa when secondary central CPP develops on top of MAS; "
                "testolactone (historical, non-specific aromatase inhibitor) less used now; "
                "fulvestrant (ER antagonist) emerging in refractory MAS."
            ),
        },
        "key_drug_contraindications": [
            "MAS (GNAS): GnRHa ALONE is INEFFECTIVE for peripheral PP — must add aromatase inhibitor",
            "LEPR + metreleptin: Expect rapid CPP onset — have GnRHa prescription prepared before starting",
            "ALL CPP: Do NOT start GnRHa before brain MRI — must exclude hypothalamic hamartoma/tumour",
            "Ketoconazole (MAS): Hepatotoxic — LFTs at baseline, monthly × 3, then q3 monthly",
            "MAS fibrous dysplasia: Avoid high-impact sports — pathological fracture risk",
            "GNAS WBC DNA: Do NOT rely on blood DNA alone for MAS — mosaic variant may be absent in leukocytes",
            "MKRN3/DLK1: Do NOT withhold GnRHa due to 'normal' MRI — genetic CPP does not always have structural cause",
        ],
        "genes_in_atlas": [g["gene"] for g in CPP_GENES],
        "seeds": list(range(SEED_BASE, SEED_BASE + len(CPP_GENES))),
        "total_patients_modelled": 320,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (MKRN3) ===")
    bk = get_breakdown()
    print(json.dumps(bk["breakdown_by_gene"]["MKRN3"], indent=2)[:2000])
    print("\n=== DEFINITIONS (first 1000 chars) ===")
    df = get_definitions()
    print(json.dumps(df, indent=2)[:1000])
