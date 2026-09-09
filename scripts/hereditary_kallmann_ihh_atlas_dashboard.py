#!/usr/bin/env python3
"""Hereditary-Kallmann-IHH-Atlas — Complete 8-Gene Atlas
(ANOS1 · FGFR1 · PROKR2 · PROK2 · CHD7 · FGF8 · GNRHR · KISS1R).

ANOS1    (Anosmin-1; 680 aa; ~100 kDa; Xp22.31; XLR;
           Kallmann Syndrome Type 1 (KS1); most common X-linked KS;
           anosmia + HH + bimanual synkinesis 50% + unilateral renal agenesis 25%;
           seed SEED_BASE+0).
FGFR1    (Fibroblast Growth Factor Receptor 1; 822 aa; ~92 kDa; 8p11.23; AD;
           Kallmann Syndrome Type 2 (KS2); most common AD form;
           anosmia/hyposmia + HH + cleft palate/lip 10–15% + dental agenesis;
           seed SEED_BASE+1).
PROKR2   (Prokineticin Receptor 2; 384 aa; ~43 kDa; 20p13; AR/dig-AD;
           Kallmann Syndrome Type 3 (KS3); variable anosmia; digenic with PROK2/FGFR1;
           seed SEED_BASE+2).
PROK2    (Prokineticin 2; 81 aa; ~9 kDa; 3p13; AR/dig-AD;
           Kallmann Syndrome Type 4 (KS4); prokineticin 2 ligand; sleep disorder overlap;
           seed SEED_BASE+3).
CHD7     (Chromodomain Helicase DNA Binding Protein 7; 2997 aa; ~337 kDa; 8q12.2; AD de novo;
           CHARGE syndrome overlap — anosmia (olfactory bulb aplasia/hypoplasia) + HH;
           seed SEED_BASE+4).
FGF8     (Fibroblast Growth Factor 8; 215 aa; ~23 kDa; 10q24.32; AD;
           Kallmann Syndrome Type 6 (KS6); FGF8 ligand for FGFR1; cleft palate overlap;
           seed SEED_BASE+5).
GNRHR    (Gonadotropin-Releasing Hormone Receptor; 328 aa; ~37 kDa; 4q13.2; AR;
           Normosmic IHH (nIHH); NO anosmia; most common AR nIHH gene;
           pulsatile GnRH therapy diagnostic/therapeutic;
           seed SEED_BASE+6).
KISS1R   (Kisspeptin Receptor / GPR54; 398 aa; ~45 kDa; 19p13.3; AR;
           Normosmic IHH (nIHH); NO anosmia; kisspeptin receptor; puberty switch;
           low/absent LH pulses; reversal phenomenon possible;
           seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2454-2461).
"""

import random

SEED_BASE = 2454

IHH_GENES = [
    # -- ANOS1 -- Kallmann Syndrome Type 1 (X-Linked) -----------------------------------------------
    {
        "gene": "ANOS1",
        "alt_name": (
            "ANOS1 (ANOS1-680aa-Xp22.31 / XLR -- "
            "KS1-KALLMANN-TYPE1-MOST-COMMON-X-LINKED -- "
            "ANOSMIA-ABSOLUTE-ABSENT-OLFACTORY-BULBS-MRI -- "
            "BIMANUAL-SYNKINESIS-MIRROR-MOVEMENTS-50pct-PATHOGNOMONIC -- "
            "UNILATERAL-RENAL-AGENESIS-25pct-CRYPTORCHIDISM-MALES -- "
            "FIBRONECTIN-III-REPEATS-OLFACTORY-GNRH-NEURON-MIGRATION)"
        ),
        "protein": (
            "ANOS1 -- Xp22.31 XLR -- ANOS1-680aa -- "
            "Anosmin-1-100kDa-Fibronectin-III-Cysteine-Rich-Extracellular-Adhesion-Molecule -- "
            "GnRH-Neuron-Migration-Olfactory-Axon-Pathfinding -- "
            "OMIM-Gene-300836-Disease-KS1-308700"
        ),
        "locus": "Xp22.31",
        "protein_size": "680 aa / ~100 kDa",
        "inheritance": (
            "XLR (X-linked recessive); males predominantly affected; female carriers rarely symptomatic; "
            "ANOS1 encodes anosmin-1, an extracellular matrix glycoprotein with WAP domain + 4 fibronectin "
            "type III repeats + a C-terminal cysteine-rich domain. "
            "FUNCTION: Anosmin-1 is secreted into the extracellular matrix and acts as a guidance cue for "
            "migrating GnRH neurons from the olfactory epithelium (placode) through the cribriform plate "
            "to the hypothalamus. Also required for olfactory axon pathfinding and branching. "
            "ANOS1 LOF → GnRH neurons fail to reach hypothalamus → no GnRH secretion → pituitary "
            "not stimulated → LH/FSH undetectable → no sex steroid production → absent puberty (HH). "
            "OLFACTORY BULB: ANOS1 also required for olfactory bulb development; LOF → absent or severely "
            "hypoplastic olfactory bulbs on MRI — PATHOGNOMONIC for Kallmann syndrome. "
            "BIMANUAL SYNKINESIS: ANOS1 also expressed in corticospinal tract; LOF → abnormal pyramidal "
            "tract decussation → mirror movements (bimanual synkinesis) in ~50% — highly specific for KS1. "
            "RENAL AGENESIS: ANOS1 expressed in developing metanephros; unilateral renal agenesis in ~25%. "
            "X-LINKED: hemizygous males fully affected; heterozygous females occasionally have anosmia alone "
            "(microsmia) or isolated HH (very rare carrier expression due to skewed X-inactivation). "
            "GENOTYPE-PHENOTYPE: null alleles → complete KS1; missense can give partial phenotype."
        ),
        "disease_category": (
            "Kallmann Syndrome Type 1 (KS1) — XLR; anosmia + HH (absent puberty) + bimanual synkinesis 50%; "
            "unilateral renal agenesis 25%; absent olfactory bulbs on MRI — PATHOGNOMONIC; "
            "cryptorchidism in nearly all males; micropenis"
        ),
        "disease_pathway": (
            "ANOS1 IN GnRH NEURON MIGRATION: GnRH neurons originate in olfactory placode → migrate along "
            "vomeronasal/terminal nerve fibres through cribriform plate → hypothalamus. "
            "Anosmin-1 acts as matrix-bound guidance molecule; interacts with heparan sulfate proteoglycans "
            "(HSPGs) on cell surface; modulates FGF8/FGFR1 signalling in olfactory/GnRH pathway. "
            "ANOS1 LOF → GnRH neuron arrest in nasal cavity or cribriform plate → fail to reach "
            "hypothalamic arcuate nucleus → no pulsatile GnRH → no LH/FSH → no sex steroids → HH. "
            "OLFACTORY: anosmin-1 required for olfactory axon branching and olfactory bulb morphogenesis; "
            "absent/hypoplastic olfactory bulbs on coronal T2 MRI (mandatory imaging). "
            "SYNKINESIS: cortical spinal tract decussation requires anosmin-1 for axon targeting; "
            "ANOS1 LOF → corticospinal fibres to ipsilateral hand remain → mirror movements when one "
            "hand moves, the other mirrors involuntarily — bimanual synkinesis test: tap one finger, "
            "observe contralateral finger — mirror movement = positive. "
            "HORMONAL AXIS: low LH (<1 IU/L), low FSH, low testosterone/oestradiol; "
            "pulsatile GnRH → LH/FSH rise (GnRH-responsive pituitary) — DISTINGUISHES from pituitary LOF."
        ),
        "pathognomonic": (
            "ANOS1 / KS1 PATHOGNOMONIC FEATURES: "
            "1. ABSENT OLFACTORY BULBS ON MRI: coronal T2 MRI shows absent/severely hypoplastic olfactory bulbs "
            "in olfactory sulci — pathognomonic for Kallmann (distinguishes from normosmic IHH/GNRHR/KISS1R); "
            "2. BIMANUAL SYNKINESIS (~50%): involuntary mirror movements — tap left index finger, right finger "
            "mirrors; highly specific for KS1 (ANOS1) — rarely seen in other KS genes; "
            "3. ANOSMIA (olfactory testing): University of Pennsylvania Smell Identification Test (UPSIT) score "
            "< 18/40 = anosmia; confirms olfactory loss; "
            "4. UNILATERAL RENAL AGENESIS (~25%): renal USS mandatory — missing kidney is non-functioning; "
            "5. CRYPTORCHIDISM (males ~95%): bilateral or unilateral undescended testes; "
            "micropenis (stretched penile length <2.5 SD); absent puberty; "
            "6. HORMONAL PROFILE: LH <1 IU/L; FSH low-normal; testosterone <100 ng/dL (males); "
            "oestradiol low (females); GnRH stimulation test → LH/FSH rise (pituitary intact). "
            "X-LINKED: family history in maternal male relatives; hemizygous in males."
        ),
        "treatment": (
            "SEX STEROID INDUCTION (puberty): "
            "MALES: testosterone enanthate 50 mg IM monthly → gradually increase over 2-3 years to 250 mg; "
            "OR testosterone undecanoate (depot); target adult testosterone 400-700 ng/dL; "
            "FEMALES: ethinyl oestradiol 2 mcg/day → increase over 2 years; add progesterone at 2 years; "
            "FERTILITY (males): pulsatile GnRH pump (5-20 mcg/90 min) → spermatogenesis in 12-24 months; "
            "OR hCG 1500-2000 IU IM 3x/week + FSH 75-150 IU 3x/week; "
            "FERTILITY (females): pulsatile GnRH pump → ovulation; or FSH/LH injection; "
            "CRYPTORCHIDISM: orchidopexy before age 1 year — reduces malignancy risk; "
            "RENAL: renal USS every 1-2 years; avoid nephrotoxic drugs; "
            "REVERSAL: 10-20% of KS1 may have spontaneous reversal — withdraw treatment trial at 3-6 months; "
            "SYNKINESIS: physiotherapy; usually not disabling; "
            "MONITORING: bone density (DEXA) annually until adult bone mineral density achieved."
        ),
        "key_features": [
            "Absent olfactory bulbs on MRI — PATHOGNOMONIC for Kallmann (DDx normosmic IHH: GNRHR/KISS1R)",
            "Bimanual synkinesis (mirror movements) ~50% — highly specific for KS1/ANOS1",
            "Unilateral renal agenesis ~25%; cryptorchidism ~95% males; micropenis",
            "XLR — hemizygous males affected; carrier females rarely symptomatic",
            "GnRH-responsive pituitary (pulsatile GnRH → LH/FSH rise — DDx pituitary LOF)",
            "Anosmin-1: extracellular matrix; FN-III repeats; GnRH neuron migration cue",
            "Testosterone/oestrogen replacement for puberty; pulsatile GnRH pump for fertility",
            "Reversal phenomenon possible (~10-20%); always trial withdrawal at 3-6 months",
        ],
        "key_ddx": (
            "FGFR1 (KS2): AD, NO synkinesis, cleft palate; "
            "PROKR2/PROK2 (KS3/4): AR/digenic, variable anosmia; "
            "GNRHR (nIHH): normosmic IHH — NO anosmia, olfactory bulbs PRESENT; "
            "KISS1R (nIHH): normosmic — NO anosmia; kisspeptin pathway; "
            "Delayed puberty (constitutional): positive family history, bone age delayed, spontaneous puberty eventual; "
            "Panhypopituitarism: LH + FSH + TSH + ACTH + GH all low; MRI pituitary lesion."
        ),
        "systemic_involvement": (
            "ENDOCRINE: absent puberty; infertility; osteoporosis (sex steroid deficiency). "
            "RENAL: unilateral renal agenesis ~25% — renal USS mandatory. "
            "NEUROLOGICAL: bimanual synkinesis ~50%; rarely epilepsy; "
            "EAR: SNHL rare (DDx CHD7/CHARGE). "
            "ORAL: cleft palate rare (DDx FGFR1). "
            "MUSCULOSKELETAL: delayed bone age; osteopenia without sex steroids."
        ),
        "onset_age": "Congenital (cryptorchidism, micropenis); absent/partial puberty at expected age 11-14y",
        "surgical_urgency": "Orchidopexy before 12 months; HH treatment from age 14-16y for puberty induction",
        "gene_family": "Extracellular matrix glycoprotein; WAP domain + 4× fibronectin III repeats",
        "morphology": (
            "MRI BRAIN: absent/hypoplastic olfactory bulbs on coronal T2 (olfactory sulci); "
            "RENAL USS: absent right or left kidney; contralateral compensatory hypertrophy; "
            "TESTICULAR USS: small testes (prepubertal volume <4 mL in adults); "
            "BONE DENSITY: osteopenia on DEXA; "
            "HORMONAL: LH <1 IU/L, FSH low-normal, testosterone <100 ng/dL"
        ),
        "n_patients": 40,
    },

    # -- FGFR1 -- Kallmann Syndrome Type 2 (Most Common AD) -----------------------------------------
    {
        "gene": "FGFR1",
        "alt_name": (
            "FGFR1 (FGFR1-822aa-8p11.23 / AD -- "
            "KS2-KALLMANN-TYPE2-MOST-COMMON-AD-FORM -- "
            "ANOSMIA-HYPOSMIA-VARIABLE-OLFACTORY-BULB-HYPOPLASTIC -- "
            "CLEFT-PALATE-LIP-10-15pct-DENTAL-AGENESIS -- "
            "DIGITAL-ANOMALIES-SYNPOLYDACTYLY -- "
            "FGF8-FGFR1-SIGNALLING-AXIS-GnRH-NEURON-MIGRATION)"
        ),
        "protein": (
            "FGFR1 -- 8p11.23 AD -- FGFR1-822aa -- "
            "Fibroblast-Growth-Factor-Receptor-1-92kDa-RTK-IgI-II-III-TM-TK -- "
            "GnRH-Neuron-Migration-Olfactory-Axon-Branching -- "
            "OMIM-Gene-136350-Disease-KS2-147950"
        ),
        "locus": "8p11.23",
        "protein_size": "822 aa / ~92 kDa",
        "inheritance": (
            "AD (autosomal dominant; haploinsufficiency); FGFR1 mutations cause KS2, the most common autosomal "
            "dominant form of Kallmann syndrome. "
            "FGFR1 is a receptor tyrosine kinase (RTK) with 3 extracellular Ig-like domains (IgI, IgII, IgIII), "
            "a transmembrane domain, and an intracellular split tyrosine kinase domain. "
            "FUNCTION: FGF8 (the main ligand in KS) binds FGFR1 IgIII → receptor dimerisation → TK "
            "autophosphorylation → RAS-MAPK and PI3K-AKT signalling → GnRH neuron migration, olfactory "
            "axon growth, and olfactory bulb morphogenesis. "
            "FGFR1 LOF → FGF8 signalling lost → GnRH neuron migration fails → HH; olfactory axon "
            "branching impaired → olfactory bulb hypoplasia → hyposmia/anosmia. "
            "INCOMPLETE PENETRANCE: ~30-40% of FGFR1 carriers have partial or no phenotype — "
            "important for genetic counselling (expressivity very variable). "
            "ASSOCIATED FEATURES: cleft palate/lip (10-15%), dental agenesis, digital anomalies "
            "(preaxial polydactyly, split hand/foot), Pfeiffer syndrome craniosynostosis (GOF alleles). "
            "DIGENIC: FGFR1 mutations act synergistically with PROKR2, PROK2, FGF8, ANOS1 to cause KS "
            "(digenic/oligogenic inheritance common in FGFR1 families)."
        ),
        "disease_category": (
            "Kallmann Syndrome Type 2 (KS2) — AD; most common autosomal dominant KS; "
            "anosmia/hyposmia + HH; cleft palate/lip 10-15%; dental agenesis; digital anomalies; "
            "variable expressivity + incomplete penetrance 30-40%"
        ),
        "disease_pathway": (
            "FGF8-FGFR1 AXIS IN KS: FGF8 produced in olfactory region binds FGFR1 on migrating GnRH neurons; "
            "FGFR1 LOF → reduced FGF8 signalling → impaired GnRH neuron migration and survival → fewer "
            "GnRH neurons reach hypothalamus → reduced pulsatile GnRH → LH/FSH deficiency → HH. "
            "OLFACTORY BULB: FGF8-FGFR1 required for olfactory axon branching into bulb; FGFR1 LOF → "
            "olfactory bulb hypoplasia/aplasia → hyposmia (more often than complete anosmia vs. ANOS1). "
            "HEPARAN SULFATE: FGFR1 co-receptor is heparan sulfate proteoglycan (HSPG); anosmin-1 (ANOS1) "
            "modulates FGF8-FGFR1 via HSPG interactions — explains ANOS1/FGFR1 functional overlap. "
            "CRANIOFACIAL: FGFR1 required for neural crest migration in midline; FGFR1 LOF → cleft palate "
            "midline defects, dental agenesis; similar to FGF8 LOF (both same pathway). "
            "OLIGOGENIC: single heterozygous FGFR1 variant alone may cause incomplete KS; additional hit "
            "in PROKR2 or PROK2 tips the balance to full KS — oligogenic model explains variable expression."
        ),
        "pathognomonic": (
            "FGFR1 / KS2 CLINICAL FEATURES: "
            "1. ANOSMIA/HYPOSMIA: olfactory testing mandatory; complete anosmia or partial hyposmia "
            "(more variable than ANOS1 which is nearly always complete anosmia); "
            "2. OLFACTORY BULB: coronal MRI T2 — hypoplastic or absent olfactory bulbs; variable "
            "(less severe than ANOS1 on average); "
            "3. CLEFT PALATE/LIP (10-15%): midline developmental defect; examine mouth at diagnosis; "
            "may have been repaired in childhood; dental agenesis (missing teeth, especially upper lateral incisors); "
            "4. DIGITAL ANOMALIES: preaxial polydactyly (extra finger on thumb side); "
            "split-hand/foot malformation; camptodactyly — examine hands in all KS2 patients; "
            "5. HH: low LH, FSH, testosterone/oestradiol; absent/arrested puberty; "
            "6. INCOMPLETE PENETRANCE: affected parent may have anosmia alone, HH alone, or no phenotype; "
            "family history often negative despite AD inheritance — new mutation or non-penetrance; "
            "7. VARIABLE EXPRESSIVITY: siblings with same FGFR1 mutation may have complete KS vs. "
            "isolated anosmia vs. isolated HH vs. normal."
        ),
        "treatment": (
            "PUBERTY INDUCTION: same as ANOS1 (testosterone for males; oestrogen/progesterone for females); "
            "FERTILITY: pulsatile GnRH pump (FGFR1 mutations → GnRH-responsive pituitary as in all KS); "
            "spermatogenesis with GnRH pump in 60-70%; gonadotropins (hCG + FSH) as alternative; "
            "CLEFT: if unrepaired, refer craniofacial surgery; dental referral for agenesis; "
            "DIGITAL: occupational therapy; orthopaedic if functional impairment; "
            "BONE DENSITY: DEXA at diagnosis and annually; bisphosphonates if T-score < -2.5; "
            "GENETIC COUNSELLING: AD but incomplete penetrance ~30-40%; offer testing to relatives; "
            "cascade testing of asymptomatic relatives for anosmia (smell test) + reproductive axis; "
            "REVERSAL: ~10-20% spontaneous reversal possible — trial withdrawal."
        ),
        "key_features": [
            "Most common AD Kallmann syndrome (KS2); anosmia/hyposmia + HH",
            "Cleft palate/lip 10-15%; dental agenesis; digital anomalies (check hands)",
            "Incomplete penetrance 30-40%; variable expressivity within families",
            "FGF8 ligand → FGFR1 RTK → GnRH neuron migration (FGF8/FGFR1 axis)",
            "Olfactory bulb hypoplastic (MRI) — more variable than ANOS1",
            "Oligogenic: FGFR1 + PROKR2 or PROK2 second hit → more severe KS",
            "GnRH-responsive pituitary; pulsatile GnRH pump effective for fertility",
            "AD with variable expressivity — anosmia alone ≠ excluded from KS2 family",
        ],
        "key_ddx": (
            "ANOS1 (KS1): XLR, bimanual synkinesis ~50%, renal agenesis 25% — DDx FGFR1 which is AD; "
            "FGF8 (KS6): same pathway, smaller protein, cleft palate also; "
            "PROKR2/PROK2 (KS3/4): AR/digenic, GPCR pathway not FGF; "
            "Pfeiffer syndrome (FGFR1 GOF): craniosynostosis + broad thumb/toe — GOF ≠ LOF KS; "
            "Cleidocranial dysplasia (RUNX2): clavicle defect, no anosmia."
        ),
        "systemic_involvement": (
            "ENDOCRINE: absent puberty; infertility; osteoporosis. "
            "OROFACIAL: cleft palate/lip 10-15%; dental agenesis especially upper lateral incisors. "
            "LIMBS: digital anomalies (polydactyly, split hand/foot). "
            "NEUROLOGICAL: no synkinesis (DDx ANOS1). "
            "RENAL: renal agenesis rare (DDx ANOS1 25%)."
        ),
        "onset_age": "Absent/partial puberty at age 11-14y; cleft palate/digital anomalies congenital",
        "surgical_urgency": "Cleft palate repair in infancy; orchidopexy if cryptorchidism; HH treatment from 14-16y",
        "gene_family": "Receptor tyrosine kinase (RTK); FGF receptor subfamily; IgI-II-III-TM-TK domains",
        "morphology": (
            "MRI BRAIN: hypoplastic olfactory bulbs (coronal T2); "
            "HANDS: X-ray for digital anomalies; "
            "HORMONAL: LH <1-2 IU/L, FSH low, testosterone <150 ng/dL; "
            "SMELL TEST: UPSIT partial (hyposmia) or complete anosmia"
        ),
        "n_patients": 40,
    },

    # -- PROKR2 -- Kallmann Syndrome Type 3 (AR/Digenic) --------------------------------------------
    {
        "gene": "PROKR2",
        "alt_name": (
            "PROKR2 (PROKR2-384aa-20p13 / AR-dig -- "
            "KS3-KALLMANN-TYPE3-GPCR-PROKINETICIN-RECEPTOR-2 -- "
            "ANOSMIA-VARIABLE-HYPOSMIA-PARTIAL -- "
            "DIGENIC-PROK2-FGFR1-SECOND-HIT -- "
            "SLEEP-DISORDER-OBESITY-OVERLAP-RARE -- "
            "GnRH-NEURON-MIGRATION-OLFACTORY-BULB)"
        ),
        "protein": (
            "PROKR2 -- 20p13 AR/dig -- PROKR2-384aa -- "
            "Prokineticin-Receptor-2-43kDa-GPCR-7TM-Gq-cAMP -- "
            "GnRH-Neuron-Migration-Olfactory-Bulb-Morphogenesis -- "
            "OMIM-Gene-607123-Disease-KS3-244200"
        ),
        "locus": "20p13",
        "protein_size": "384 aa / ~43 kDa",
        "inheritance": (
            "AR (autosomal recessive) or digenic-AD (one PROKR2 + one second hit in PROK2/FGFR1/ANOS1); "
            "PROKR2 encodes prokineticin receptor 2, a 7-transmembrane (7TM) GPCR that signals via Gq "
            "(PLC-IP3-Ca2+) and Gi (cAMP reduction) pathways. "
            "PROK2 (prokineticin 2) is the primary endogenous ligand for PROKR2. "
            "FUNCTION: PROK2-PROKR2 signalling is required for GnRH neuron migration from olfactory "
            "placode to hypothalamus AND for olfactory bulb morphogenesis (olfactory bulb neurons use "
            "PROK2 as a paracrine mitogen for cell proliferation and survival). "
            "PROKR2 LOF → impaired PROK2 signalling → GnRH neuron stall → HH; olfactory bulb "
            "development impaired → anosmia (variable severity). "
            "DIGENIC INHERITANCE: many PROKR2 carriers have only one mutant allele and are heterozygous "
            "carriers; full KS phenotype requires a second hit (digenic): heterozygous PROKR2 + "
            "heterozygous PROK2 (or FGFR1, ANOS1) → disease (oligogenic model). "
            "SLEEP/OBESITY: PROKR2 expressed in hypothalamus — PROKR2 biallelic LOF can → "
            "circadian sleep disorder, obesity (rare overlapping phenotype — mouse PROKR2 KO: obesity + "
            "anosmia + HH + sleep anomalies)."
        ),
        "disease_category": (
            "Kallmann Syndrome Type 3 (KS3) — AR or digenic-AD; variable anosmia/hyposmia; "
            "HH; olfactory bulb hypoplasia (variable); digenic with PROK2, FGFR1, or ANOS1"
        ),
        "disease_pathway": (
            "PROK2-PROKR2 AXIS: PROK2 secreted in olfactory bulb → binds PROKR2 on GnRH progenitors "
            "and olfactory neurons → Gq activation → IP3 → Ca2+ release → cell migration and survival; "
            "also Gi → reduced cAMP → mitogen signalling. "
            "GnRH MIGRATION: PROKR2 expressed on migrating GnRH neurons; PROK2 gradient from olfactory "
            "bulb to nasal cavity acts as chemoattractant for GnRH neurons migrating centrally; "
            "PROKR2 LOF → neurons cannot follow PROK2 gradient → stall in nasal cavity → HH. "
            "OLFACTORY BULB MORPHOGENESIS: PROK2 is a mitogen for olfactory bulb interneuron progenitors; "
            "PROKR2 LOF → reduced OB neuron proliferation → hypoplastic OB → anosmia/hyposmia. "
            "DIGENIC MODEL: single PROKR2 heterozygous variant often insufficient alone; "
            "synergistic loss of PROK2, FGFR1, or ANOS1 on other allele causes full phenotype; "
            "explains variable expressivity within families."
        ),
        "pathognomonic": (
            "PROKR2 / KS3 CLINICAL FEATURES: "
            "1. VARIABLE ANOSMIA/HYPOSMIA: olfactory testing — complete anosmia OR partial hyposmia "
            "(more variable than ANOS1; hyposmia more common than in ANOS1-KS1); "
            "2. OLFACTORY BULB: MRI T2 — hypoplastic or absent (variable); sometimes only mildly reduced volume; "
            "3. HH: low LH/FSH/testosterone; absent puberty or partial puberty (arrested); "
            "4. SLEEP DISORDER: ask about circadian rhythm disruption; may have delayed sleep phase; "
            "5. DIGENIC HISTORY: family members may carry one PROKR2 variant with anosmia alone "
            "(no HH) → suggests second hit required for full phenotype; "
            "6. REVERSIBILITY: reversal phenomenon more common in PROKR2 (partial LOF) than complete KS1."
        ),
        "treatment": (
            "PUBERTY INDUCTION: testosterone (males) or oestrogen/progesterone (females); "
            "FERTILITY: pulsatile GnRH pump (GnRH-responsive pituitary); "
            "spermatogenesis achievable in most; "
            "REVERSAL: spontaneous reversal more likely if partial LOF; trial withdrawal at 3-6 months; "
            "SLEEP: sleep hygiene counselling; melatonin if circadian disorder; "
            "GENETIC COUNSELLING: digenic — test partner/family for PROK2/FGFR1/ANOS1 second hits; "
            "recurrence risk depends on digenic vs. AR; "
            "MONITORING: LH/FSH/testosterone annually; bone density; smell testing."
        ),
        "key_features": [
            "Kallmann Type 3: AR or digenic with PROK2/FGFR1/ANOS1 (oligogenic inheritance)",
            "Variable anosmia/hyposmia — partial (hyposmia) more common than KS1 complete anosmia",
            "PROK2-PROKR2 GPCR pathway: GnRH neuron migration + olfactory bulb morphogenesis",
            "Sleep disorder + obesity overlap (hypothalamic PROKR2 — mouse model: circadian + obesity + HH)",
            "Reversal phenomenon more common with partial LOF alleles",
            "Heterozygous PROKR2 alone often insufficient — screen for second hit",
            "GnRH-responsive pituitary: pulsatile GnRH effective for fertility",
            "Olfactory bulbs variably hypoplastic on MRI (less severe than KS1/ANOS1 on average)",
        ],
        "key_ddx": (
            "ANOS1 (KS1): XLR, complete anosmia, bimanual synkinesis 50%; "
            "PROK2 (KS4): AR/digenic, ligand for same receptor PROKR2 — allelic; "
            "FGFR1 (KS2): AD, cleft palate, digital anomalies — FGF pathway not GPCR; "
            "GNRHR: normosmic IHH — olfactory bulbs PRESENT, no anosmia; "
            "Constitutional delayed puberty: spontaneous puberty, bone age delay, family history."
        ),
        "systemic_involvement": (
            "ENDOCRINE: HH; absent puberty; infertility; osteopenia. "
            "NEUROLOGICAL: circadian sleep disturbance (hypothalamic PROKR2). "
            "METABOLIC: obesity risk (hypothalamic PROKR2 in weight regulation). "
            "OLFACTORY: variable hyposmia/anosmia; olfactory bulb hypoplasia on MRI."
        ),
        "onset_age": "Absent/partial puberty at 11-14y; anosmia/hyposmia may be noted earlier",
        "surgical_urgency": "Orchidopexy if cryptorchidism; no emergency; HH treatment from 14-16y",
        "gene_family": "G-protein coupled receptor (GPCR); rhodopsin family; 7-transmembrane; Gq/Gi signalling",
        "morphology": (
            "MRI: variably hypoplastic olfactory bulbs (less severe than ANOS1); "
            "HORMONAL: LH low (<2 IU/L), FSH low-normal, testosterone <200 ng/dL; "
            "SMELL TEST: UPSIT partial-complete anosmia"
        ),
        "n_patients": 40,
    },

    # -- PROK2 -- Kallmann Syndrome Type 4 (AR/Digenic) ---------------------------------------------
    {
        "gene": "PROK2",
        "alt_name": (
            "PROK2 (PROK2-81aa-3p13 / AR-dig -- "
            "KS4-KALLMANN-TYPE4-PROKINETICIN-2-LIGAND -- "
            "ANOSMIA-VARIABLE-OLFACTORY-BULB-HYPOPLASIA -- "
            "DIGENIC-PROKR2-FGFR1-SECOND-HIT -- "
            "SLEEP-CIRCADIAN-OBESITY-OVERLAP -- "
            "SMALLEST-PROKINETICIN-FAMILY-MEMBER-81aa)"
        ),
        "protein": (
            "PROK2 -- 3p13 AR/dig -- PROK2-81aa -- "
            "Prokineticin-2-9kDa-Cysteine-Rich-EGF-Like-Colipase-Domain -- "
            "GnRH-Neuron-Migration-Olfactory-Bulb-Mitogen-Circadian -- "
            "OMIM-Gene-607002-Disease-KS4-610628"
        ),
        "locus": "3p13",
        "protein_size": "81 aa / ~9 kDa",
        "inheritance": (
            "AR (autosomal recessive) or digenic-AD (PROK2 + PROKR2 or FGFR1/ANOS1 second hit); "
            "PROK2 is the smallest member of the prokineticin family (81 aa); it encodes prokineticin 2, "
            "the primary ligand for PROKR2. PROK2 has a conserved N-terminal sequence (AVITGA motif) "
            "essential for PROKR2 binding and activation; also contains an EGF-like and colipase-fold domain. "
            "FUNCTION: PROK2 is secreted and acts on PROKR2 to: "
            "(1) guide GnRH neurons from olfactory placode through cribriform plate (chemoattractant/survival); "
            "(2) drive olfactory bulb neuron proliferation (mitogen for OB interneuron progenitors); "
            "(3) regulate circadian rhythms (suprachiasmatic nucleus PROK2 expression peaks at dawn). "
            "PROK2 LOF → PROKR2 not activated → same pathway failure as PROKR2 — HH + variable anosmia. "
            "CIRCADIAN: PROK2 is a key circadian output molecule from SCN; PROK2 LOF → circadian period "
            "lengthening, sleep phase disruption — occasionally seen clinically. "
            "DIGENIC: heterozygous PROK2 + heterozygous PROKR2 or FGFR1 → full phenotype."
        ),
        "disease_category": (
            "Kallmann Syndrome Type 4 (KS4) — AR or digenic; variable anosmia + HH; "
            "olfactory bulb hypoplasia; circadian sleep disturbance; obesity overlap"
        ),
        "disease_pathway": (
            "PROK2 SIGNALLING AXIS: PROK2 secreted by olfactory bulb neurons → binds PROKR2 on "
            "GnRH neurons and OB progenitors → Gq (IP3-Ca2+) + Gi (cAMP) → cell migration + proliferation. "
            "PROK2 LOF → same downstream failure as PROKR2 LOF: GnRH neuron arrest → HH; "
            "OB neuron deficit → olfactory bulb hypoplasia → anosmia (variable). "
            "CIRCADIAN ROLE: PROK2 expression in suprachiasmatic nucleus (SCN) peaks at lights-on (circadian dawn); "
            "PROK2 signals SCN output to drive daytime arousal; PROK2 LOF → reduced arousal signal → "
            "delayed sleep phase / circadian disruption. "
            "SMALLEST PROKINETICIN: PROK2 (81aa) has N-terminal AVITGA motif — conserved binding epitope "
            "for PROKR2; mutations in this motif → complete LOF; missense elsewhere → partial LOF → "
            "variable expressivity. "
            "OLIGOGENIC: PROK2 heterozygous alone may → isolated anosmia; second hit → full KS."
        ),
        "pathognomonic": (
            "PROK2 / KS4 CLINICAL FEATURES: "
            "1. ANOSMIA/HYPOSMIA: variable; partial hyposmia possible; "
            "2. MRI BRAIN: hypoplastic/absent olfactory bulbs (variable severity); "
            "3. HH: low LH/FSH/sex steroids; absent or arrested puberty; "
            "4. CIRCADIAN: delayed sleep phase; difficulty waking in morning; irregular sleep timing — "
            "ask specifically; more prominent than in ANOS1/FGFR1; "
            "5. OBESITY: BMI monitoring; hypothalamic PROK2 regulates food intake; "
            "6. DIGENIC CONTEXT: family members may have isolated anosmia alone (PROK2 het) or "
            "isolated HH alone → suggests oligogenic model operating in family."
        ),
        "treatment": (
            "PUBERTY: testosterone/oestrogen as per standard KS protocol; "
            "FERTILITY: pulsatile GnRH pump or gonadotropins; "
            "CIRCADIAN: melatonin 0.5-5 mg at targeted bedtime; sleep hygiene; light therapy morning; "
            "WEIGHT: diet + exercise counselling; metabolic monitoring; "
            "REVERSAL: possible if partial LOF; trial withdrawal at 3-6 months; "
            "GENETIC COUNSELLING: test PROKR2 and FGFR1 for digenic hits; "
            "MONITORING: LH/FSH/testosterone; bone density; smell annually."
        ),
        "key_features": [
            "KS4: AR or digenic; PROK2 is smallest prokineticin (81aa); ligand for PROKR2",
            "Circadian sleep disturbance prominent (SCN PROK2) — ask about sleep timing",
            "Variable anosmia/hyposmia; obesity risk (hypothalamic PROK2)",
            "Digenic model: PROK2 het + PROKR2 het (or FGFR1) → full phenotype",
            "Same downstream pathway as PROKR2 (KS3) — allelic to PROKR2 disease functionally",
            "GnRH-responsive pituitary: pulsatile GnRH pump effective for fertility",
            "Olfactory bulbs variably hypoplastic; reversal phenomenon possible",
            "AVITGA N-terminal motif in PROK2 essential for PROKR2 binding — mutations here = complete LOF",
        ],
        "key_ddx": (
            "PROKR2 (KS3): same pathway — receptor (PROKR2) vs. ligand (PROK2); allelic disease; "
            "ANOS1 (KS1): XLR, synkinesis, renal agenesis; "
            "FGFR1 (KS2): AD, cleft palate, FGF pathway; "
            "Narcolepsy: cataplexy + sleep attacks — different mechanism (orexin/hypocretin); "
            "Prader-Willi syndrome: hypotonia + HH + hyperphagia — chromosome 15q; "
            "Constitutional delayed puberty: reversible; no anosmia."
        ),
        "systemic_involvement": (
            "ENDOCRINE: HH; absent puberty; osteopenia. "
            "NEUROLOGICAL: circadian rhythm disorder (delayed sleep phase). "
            "METABOLIC: obesity risk. "
            "OLFACTORY: variable hyposmia/anosmia."
        ),
        "onset_age": "Absent puberty 11-14y; sleep disturbance from childhood; anosmia/hyposmia variable",
        "surgical_urgency": "No emergency; HH treatment initiation from 14-16y; orchidopexy if needed",
        "gene_family": "Prokineticin family; EGF-like + colipase-fold; AVITGA N-terminal motif",
        "morphology": (
            "MRI: variably hypoplastic olfactory bulbs; "
            "HORMONAL: LH <2 IU/L, FSH low, testosterone <200 ng/dL; "
            "SMELL TEST: variable partial-complete anosmia"
        ),
        "n_patients": 40,
    },

    # -- CHD7 -- CHARGE Syndrome Overlap (Anosmia + HH) ---------------------------------------------
    {
        "gene": "CHD7",
        "alt_name": (
            "CHD7 (CHD7-2997aa-8q12.2 / AD-de-novo -- "
            "CHARGE-SYNDROME-OVERLAP-ANOSMIA-HH-COMPONENT -- "
            "OLFACTORY-BULB-APLASIA-SEMICIRCULAR-CANAL-APLASIA -- "
            "HH-IN-CHARGE-80pct-OLFACTORY-BULB-ABSENT-90pct -- "
            "COLOBOMA-HEART-CHOANAL-ATRESIA-RETARDATION-GENITAL-EAR -- "
            "CHROMODOMAIN-HELICASE-DNA-BINDING-CHROMATIN-REMODELLER)"
        ),
        "protein": (
            "CHD7 -- 8q12.2 AD de novo -- CHD7-2997aa -- "
            "Chromodomain-Helicase-DNA-Binding-Protein-7-337kDa-Chromatin-Remodeller-CHD -- "
            "GnRH-Neuron-Development-Olfactory-Placode-Craniofacial-Neural-Crest -- "
            "OMIM-Gene-608892-Disease-CHARGE-214800"
        ),
        "locus": "8q12.2",
        "protein_size": "2997 aa / ~337 kDa",
        "inheritance": (
            "AD (autosomal dominant); >90% de novo; CHD7 is a chromodomain helicase DNA-binding protein "
            "(CHD subfamily) that remodels chromatin by repositioning nucleosomes. "
            "CHD7 binds H3K4me1-marked enhancers and interacts with PBAF/SWI-SNF complex. "
            "CHD7 is required for activation of hundreds of tissue-specific enhancers during development. "
            "HH IN CHARGE: anosmia + HH occur in ~60-80% of CHARGE patients; olfactory bulb aplasia/hypoplasia "
            "is seen in ~90% of CHARGE on MRI — among the most severe olfactory bulb defects. "
            "CHD7 regulates transcription of genes required for GnRH neuron specification in olfactory "
            "placode and for olfactory bulb development. "
            "HH IN CHARGE may be central (hypothalamic — CHD7 directly) or compound (olfactory bulb "
            "aplasia → no GnRH neuron housing → HH). "
            "CHARGE MNEMONIC: C = Coloboma; H = Heart defect; A = choanal Atresia; R = Retardation; "
            "G = Genital abnormalities (HH + cryptorchidism); E = Ear anomalies (SNHL, SCCaplasia). "
            "KS-OVERLAP: ~5-10% of patients who present with 'Kallmann syndrome' and have additional "
            "features (coloboma, SCC aplasia, heart defect) are actually CHD7 mutations — CHARGE, not pure KS."
        ),
        "disease_category": (
            "CHARGE syndrome (CHD7) — AD de novo; anosmia + HH as component of CHARGE; "
            "olfactory bulb aplasia; semicircular canal aplasia CT near-pathognomonic; "
            "choanal atresia may be neonatal airway emergency"
        ),
        "disease_pathway": (
            "CHD7 IN KS/HH CONTEXT: CHD7 is required for development of olfactory placode → GnRH neuron "
            "specification; CHD7 LOF → reduced GnRH neuron number → HH. "
            "OLFACTORY BULB: CHD7 required for olfactory bulb morphogenesis enhancer activation; "
            "CHD7 LOF → olfactory bulb aplasia (most severe of all KS genes — near complete absence). "
            "SEMICIRCULAR CANAL APLASIA: CHD7 required for otocyst development; SCC aplasia = near "
            "pathognomonic for CHARGE — distinguishes from other KS genes (no SCC abnormality in KS1-6). "
            "CHROMATIN REMODELLING: CHD7 repositions nucleosomes at enhancers → gene activation; "
            "hundreds of target genes across multiple tissues → explains multisystem CHARGE phenotype. "
            "CHOANAL ATRESIA: CHD7 required for choanal plate resorption; bilateral choanal atresia → "
            "neonatal airway emergency (neonates are obligate nasal breathers). "
            "HH IN CHARGE: most CHARGE patients have hypogonadism; often overlooked clinically; "
            "screening for HH mandatory in all CHARGE patients reaching puberty."
        ),
        "pathognomonic": (
            "CHD7 / CHARGE CLINICAL FEATURES WITH HH: "
            "1. OLFACTORY BULB APLASIA (MRI ~90%): most severe of all KS/HH genes; near-absent olfactory "
            "bulbs with absent olfactory sulci; "
            "2. SEMICIRCULAR CANAL APLASIA (CT ~90%): absent posterior, lateral, and superior SCCs; "
            "profound vestibular areflexia → unable to use vestibular VOR; motor delay; "
            "NEAR-PATHOGNOMONIC for CHARGE (distinguishes from other HH genes); "
            "3. CHOANAL ATRESIA: bony or membranous; bilateral = neonatal EMERGENCY (obligate nasal breathers); "
            "4. COLOBOMA: iris coloboma or chorioretinal coloboma or optic disc coloboma; "
            "5. HH (~60-80%): cryptorchidism; micropenis; delayed/absent puberty; low LH/FSH; "
            "6. CARDIAC: congenital heart defects in ~75% (ASD, VSD, ToF, interrupted aortic arch); "
            "7. SNHL: profound bilateral (SCC aplasia); cochlear implant effective. "
            "KEY DIAGNOSTIC RULE: any patient presenting with 'Kallmann syndrome' + coloboma or SCC aplasia "
            "or choanal atresia → suspect CHARGE/CHD7 and sequence CHD7."
        ),
        "treatment": (
            "HH: same testosterone/oestrogen protocol as other KS; "
            "FERTILITY: pulsatile GnRH pump; limited data in CHARGE — intellectual disability may limit; "
            "CHOANAL ATRESIA: neonatal — McGovern nipple / oral airway; urgent ENT choanoplasty; "
            "SNHL/SCC APLASIA: cochlear implant for severe-profound SNHL; vestibular physiotherapy mandatory; "
            "COLOBOMA: ophthalmology annual; low vision support; "
            "CARDIAC: paediatric cardiology; surgical repair as indicated; "
            "CRYPTORCHIDISM: orchidopexy <12 months; "
            "INTELLECTUAL DISABILITY: developmental support; special education."
        ),
        "key_features": [
            "CHARGE overlap: anosmia + HH in ~60-80% of CHARGE; olfactory bulb aplasia ~90%",
            "SCC aplasia on CT — near-PATHOGNOMONIC for CHARGE; profound vestibular loss; motor delay",
            "Bilateral choanal atresia = neonatal airway EMERGENCY (obligate nasal breathers)",
            "Coloboma (iris/chorioretinal/optic disc); cardiac defect ~75%; SNHL",
            "CHD7 de novo AD; chromatin remodeller activating developmental enhancers",
            "HH in CHARGE often under-recognized — screen ALL CHARGE patients at puberty",
            "KS-CHARGE overlap: pure KS with coloboma/SCC aplasia → sequence CHD7",
            "2997aa — largest of all KS/HH genes; multisystem transcriptional regulator",
        ],
        "key_ddx": (
            "ANOS1 (KS1): no coloboma, no SCC aplasia, no choanal atresia, no cardiac; "
            "FGFR1 (KS2): cleft palate, digital anomalies — no SCC aplasia, no coloboma; "
            "PROKR2/PROK2: small 7TM/ligand proteins; no SCC aplasia; no choanal atresia; "
            "GNRHR/KISS1R: normosmic IHH — olfactory bulbs present, no SCC aplasia."
        ),
        "systemic_involvement": (
            "ENDOCRINE: HH ~60-80%; cryptorchidism; micropenis. "
            "EYE: coloboma (iris/retinal/optic disc). "
            "EAR: SNHL (profound) + SCC aplasia; absent VOR; motor delay. "
            "CARDIAC: CHD ~75%. "
            "AIRWAY: choanal atresia (bilateral = neonatal emergency). "
            "NEUROLOGICAL: intellectual disability (variable)."
        ),
        "onset_age": "Congenital (choanal atresia, coloboma, heart defect); HH at puberty age",
        "surgical_urgency": "Bilateral choanal atresia: neonatal airway EMERGENCY; cardiac repair neonatal",
        "gene_family": "Chromodomain helicase DNA-binding (CHD) protein family; CHD subfamily",
        "morphology": (
            "MRI BRAIN: olfactory bulb aplasia; SCC aplasia on CT temporal bone; "
            "HORMONAL: low LH/FSH/testosterone; "
            "OPHTHALMOLOGY: coloboma on fundoscopy; "
            "CARDIOLOGY: ECHO for CHD"
        ),
        "n_patients": 40,
    },

    # -- FGF8 -- Kallmann Syndrome Type 6 (AD; FGF Ligand) ------------------------------------------
    {
        "gene": "FGF8",
        "alt_name": (
            "FGF8 (FGF8-215aa-10q24.32 / AD -- "
            "KS6-KALLMANN-TYPE6-FGF8-LIGAND-FGFR1 -- "
            "ANOSMIA-VARIABLE-HH-CLEFT-PALATE-LIP -- "
            "SAME-PATHWAY-FGFR1-DIGENIC-OLIGOGENIC -- "
            "CEREBELLAR-VERMIS-HYPOPLASIA-RARE -- "
            "FGF-FAMILY-SMALLEST-KS-GENE-215aa)"
        ),
        "protein": (
            "FGF8 -- 10q24.32 AD -- FGF8-215aa -- "
            "Fibroblast-Growth-Factor-8-23kDa-Beta-Trefoil-Fold -- "
            "FGFR1-Ligand-GnRH-Neuron-Migration-Craniofacial -- "
            "OMIM-Gene-600483-Disease-KS6-612702"
        ),
        "locus": "10q24.32",
        "protein_size": "215 aa / ~23 kDa",
        "inheritance": (
            "AD (autosomal dominant; haploinsufficiency) or AR (rare, biallelic); "
            "FGF8 encodes fibroblast growth factor 8, a member of the FGF family with a conserved "
            "beta-trefoil fold structure. FGF8 binds FGFR1 (with IIIc isoform preference) and FGFR2/4. "
            "FGF8 is the primary FGF8 axis ligand in the olfactory/GnRH system (together with FGF17). "
            "FUNCTION: FGF8 produced in olfactory epithelium and nasal mesenchyme → binds FGFR1 on "
            "migrating GnRH neurons → promotes survival and migration. "
            "FGF8 also required for midline craniofacial development → FGF8 LOF → cleft palate/lip "
            "(same as FGFR1 LOF — same pathway). "
            "CEREBELLAR: FGF8 produced in isthmic organizer → midbrain-hindbrain boundary; FGF8 "
            "severe LOF → cerebellar vermis hypoplasia (rare, more AR cases). "
            "SAME PATHWAY AS FGFR1: FGF8 heterozygous LOF → same KS phenotype as FGFR1 het LOF; "
            "FGF8 + FGFR1 in same family is epistatic (same pathway — loss of ligand = loss of receptor). "
            "OLIGOGENIC: FGF8 het + FGFR1 het → more severe phenotype (synergistic pathway failure). "
            "PENETRANCE: incomplete as in FGFR1; variable expressivity."
        ),
        "disease_category": (
            "Kallmann Syndrome Type 6 (KS6) — AD; anosmia/hyposmia + HH; cleft palate/lip; "
            "cerebellar vermis hypoplasia (rare, severe alleles); same pathway as FGFR1 (KS2)"
        ),
        "disease_pathway": (
            "FGF8-FGFR1 AXIS AGAIN (KS6): same molecular pathway as KS2; "
            "FGF8 LOF → reduced FGFR1 signalling on GnRH neurons → GnRH neuron migration failure → HH; "
            "olfactory axon branching impaired → variable olfactory bulb hypoplasia → anosmia/hyposmia. "
            "CRANIOFACIAL: FGF8 secreted by facial ectoderm → drives midline palatal fusion; "
            "FGF8 LOF → cleft palate/lip — same defect as FGFR1 LOF. "
            "ISTHMIC ORGANIZER: FGF8 is the key organizer at midbrain-hindbrain boundary; "
            "haploinsufficiency rarely → cerebellar vermis hypoplasia (mainly biallelic or very hypomorphic); "
            "check MRI cerebellum in all FGF8/KS6 patients. "
            "HEPARAN SULFATE: FGF8 binds FGFR1 in heparan sulfate-stabilised complex; "
            "mutations disrupting HS-binding surface of FGF8 → partial LOF → oligogenic disease."
        ),
        "pathognomonic": (
            "FGF8 / KS6 CLINICAL FEATURES: "
            "1. ANOSMIA/HYPOSMIA: variable; olfactory testing mandatory; "
            "2. HH: low LH/FSH/sex steroids; absent/partial puberty; "
            "3. CLEFT PALATE/LIP: less common than FGFR1 but same craniofacial defect pattern; "
            "4. CEREBELLAR VERMIS HYPOPLASIA: rare; order MRI cerebellum (Dandy-Walker spectrum variants); "
            "present as ataxia, balance issues — check in FGF8 patients; "
            "5. MRI BRAIN: olfactory bulb hypoplasia; cerebellar vermis (if severe allele); "
            "6. SAME FAMILY AS FGFR1: if FGFR1 already known in family, consider FGF8 digenic hit; "
            "7. INCOMPLETE PENETRANCE: as FGFR1 — same gene, same pathway, same penetrance characteristics."
        ),
        "treatment": (
            "PUBERTY: testosterone/oestrogen standard KS protocol; "
            "FERTILITY: pulsatile GnRH pump; GnRH-responsive pituitary; "
            "CLEFT: craniofacial surgery referral; dental evaluation; "
            "CEREBELLAR: physiotherapy for ataxia/balance; no disease-modifying therapy; "
            "REVERSAL: possible (partial LOF cases); withdrawal trial at 3-6 months; "
            "GENETIC COUNSELLING: AD; incomplete penetrance; test for FGFR1 digenic."
        ),
        "key_features": [
            "KS6: FGF8 ligand for FGFR1 (same pathway as KS2); AD; incomplete penetrance",
            "Cleft palate/lip (same craniofacial defect as FGFR1/KS2 — same pathway)",
            "Cerebellar vermis hypoplasia (rare, severe alleles) — check MRI cerebellum",
            "Smallest non-prokineticin KS protein: 215aa; beta-trefoil fold; binds FGFR1 IIIc",
            "FGF8 + FGFR1 digenic → more severe phenotype (synergistic in same pathway)",
            "Variable anosmia/hyposmia; olfactory bulb hypoplastic on MRI",
            "GnRH-responsive pituitary: pulsatile GnRH for fertility",
            "Same approach as KS2/FGFR1 for all management decisions",
        ],
        "key_ddx": (
            "FGFR1 (KS2): same pathway — receptor vs. ligand; cleft palate + digital anomalies also KS2; "
            "CHD7 (CHARGE): SCC aplasia, coloboma, choanal atresia — distinguishes from FGF8; "
            "Dandy-Walker malformation (severe): isolated; no HH/anosmia; "
            "PROKR2/PROK2: GPCR pathway, not FGF; no cleft palate."
        ),
        "systemic_involvement": (
            "ENDOCRINE: HH; absent puberty. "
            "OROFACIAL: cleft palate/lip (rare). "
            "NEUROLOGICAL: cerebellar vermis hypoplasia (rare) → ataxia. "
            "OLFACTORY: variable hyposmia/anosmia."
        ),
        "onset_age": "Absent puberty 11-14y; cleft palate congenital; cerebellar signs variable onset",
        "surgical_urgency": "Cleft repair in infancy; no other emergency; HH treatment from 14-16y",
        "gene_family": "FGF (fibroblast growth factor) family; beta-trefoil fold; FGFR1 ligand",
        "morphology": (
            "MRI BRAIN: hypoplastic olfactory bulbs; check cerebellar vermis (rare hypoplasia); "
            "HORMONAL: LH <2 IU/L, FSH low, testosterone <200 ng/dL; "
            "SMELL TEST: variable anosmia/hyposmia"
        ),
        "n_patients": 40,
    },

    # -- GNRHR -- Normosmic IHH (AR; GnRH Receptor) ------------------------------------------------
    {
        "gene": "GNRHR",
        "alt_name": (
            "GNRHR (GNRHR-328aa-4q13.2 / AR -- "
            "NORMOSMIC-IHH-NO-ANOSMIA-OLFACTORY-BULBS-PRESENT -- "
            "MOST-COMMON-AR-nIHH-GENE-40pct-AR-nIHH -- "
            "GNRH-RECEPTOR-PITUITARY-GNRH-RESISTANT -- "
            "COMPOUND-HETEROZYGOUS-Q106R-R262Q-MOST-COMMON -- "
            "PULSATILE-GNRH-DIAGNOSTIC-PITUITARY-RESPONSIVE)"
        ),
        "protein": (
            "GNRHR -- 4q13.2 AR -- GNRHR-328aa -- "
            "GnRH-Receptor-37kDa-GPCR-7TM-Gq-PLC-LH-FSH-Release -- "
            "Pituitary-Gonadotroph-GnRH-Sensing-LH-FSH-Pulsatile-Response -- "
            "OMIM-Gene-138850-Disease-nIHH-146110"
        ),
        "locus": "4q13.2",
        "protein_size": "328 aa / ~37 kDa",
        "inheritance": (
            "AR (autosomal recessive; biallelic pathogenic variants required); "
            "GNRHR is the most common cause of autosomal recessive normosmic IHH, accounting for "
            "~40% of AR nIHH cases. "
            "GNRHR encodes the GnRH receptor (GPCR, 7TM) on pituitary gonadotrophs. "
            "Normal: hypothalamic GnRH released in pulses every 90-120 min → binds GNRHR → Gq activation → "
            "PLC → IP3 → Ca2+ → LH and FSH secretion. "
            "GNRHR LOF: pituitary gonadotrophs cannot respond to GnRH → no LH/FSH release → sex steroid "
            "deficiency → HH. Crucially, GnRH neurons ARE present and DO fire — the defect is at the "
            "receptor (pituitary), NOT the hypothalamus. "
            "KEY DIAGNOSTIC TEST: exogenous pulsatile GnRH pump → LH/FSH DO NOT RISE (pituitary-level block); "
            "contrasts with KS1-6 where pituitary IS intact and responds to pulsatile GnRH. "
            "PARTIAL vs COMPLETE LOF: partial LOF alleles (Q106R) → partial puberty (arrested); "
            "complete LOF (R262Q) → total absent puberty; compound het Q106R/R262Q most common combination. "
            "NORMOSMIC IHH: olfactory bulbs PRESENT on MRI; smell NORMAL — critical DDx from Kallmann."
        ),
        "disease_category": (
            "Normosmic Isolated Hypogonadotropic Hypogonadism (nIHH) — AR; NO anosmia; olfactory bulbs PRESENT; "
            "pituitary-level GnRH resistance; most common AR nIHH gene (~40% of AR nIHH); "
            "pulsatile GnRH pump → LH/FSH do NOT rise (DDx all KS where pituitary responds)"
        ),
        "disease_pathway": (
            "GNRHR IN GONADOTROPH PHYSIOLOGY: GnRH binds GNRHR → Gαq11 activation → PLCβ → IP3 + DAG → "
            "Ca2+ oscillations + PKC → vesicular exocytosis of LH and FSH. "
            "GNRHR LOF → absent/reduced Ca2+ oscillation in gonadotroph → no LH/FSH pulsatile secretion → "
            "no LH → no testosterone/oestradiol → HH. "
            "PULSE FREQUENCY SENSITIVITY: normally different GnRH pulse frequencies → LH vs FSH ratio changes "
            "(fast frequency = LH-preferring; slow = FSH-preferring); GNRHR LOF → this regulation lost. "
            "GONADOTROPH INTACT: GnRH neurons fire normally in nIHH/GNRHR → GnRH released normally → "
            "but receptor on pituitary non-functional → no response. "
            "TREATMENT IMPLICATIONS: pulsatile GnRH pump WILL NOT WORK for GNRHR LOF (no receptor); "
            "must use gonadotropin injections (hCG for LH action, FSH injection) to bypass the receptor."
        ),
        "pathognomonic": (
            "GNRHR / nIHH PATHOGNOMONIC FEATURES: "
            "1. NORMOSMIC: smell test NORMAL (UPSIT ≥34); olfactory bulbs PRESENT on coronal MRI — "
            "CRITICAL DDx from all Kallmann syndrome genes; "
            "2. PITUITARY DOES NOT RESPOND TO PULSATILE GnRH: exogenous pulsatile GnRH pump → "
            "LH/FSH fail to rise — PATHOGNOMONIC for GNRHR LOF (receptor-level block); "
            "contrasts with KS1-6 where pituitary intact and LH/FSH rise with GnRH pump; "
            "3. PARTIAL PUBERTY: Q106R compound het → partial puberty (some breast/testicular growth "
            "then arrested) → distinguish from complete nIHH; "
            "4. HORMONAL: LH low/pulsatile absent; FSH low-normal; testosterone/oestradiol low; "
            "GnRH stimulation (100mcg bolus) → blunted or absent LH/FSH response; "
            "5. GONADOTROPINS EFFECTIVE: hCG alone raises testosterone; FSH → spermatogenesis; "
            "fertility achievable with gonadotropin therapy despite GNRHR LOF."
        ),
        "treatment": (
            "PUBERTY INDUCTION: "
            "MALES: testosterone enanthate/cypionate IM or undecanoate depot — standard protocol; "
            "FEMALES: oestrogen/progesterone as standard; "
            "FERTILITY (CRITICAL DIFFERENCE FROM KS1-6): "
            "PULSATILE GnRH PUMP WILL NOT WORK — receptor defective; "
            "MALES: hCG 1500-2000 IU IM 3x/week → testosterone + partial spermatogenesis; "
            "add recombinant FSH (75-150 IU 3x/week) for spermatogenesis — takes 12-24 months; "
            "FEMALES: FSH (recombinant) 75-150 IU daily + LH (recombinant or hCG) → follicular development; "
            "REVERSAL: uncommon in GNRHR biallelic complete LOF; partial LOF (Q106R) may show reversal; "
            "MONITORING: testosterone, LH, FSH, sperm count (males); oestradiol, LH, FSH (females); "
            "BONE DENSITY: DEXA annually."
        ),
        "key_features": [
            "Normosmic IHH: olfactory bulbs PRESENT on MRI; smell NORMAL — DDx from all Kallmann genes",
            "Most common AR nIHH gene (~40% of AR nIHH); biallelic LOF required",
            "Pituitary-level block: pulsatile GnRH pump DOES NOT WORK (receptor defective) — CRITICAL",
            "Gonadotropins (hCG + FSH) effective for fertility — bypasses defective receptor",
            "Q106R/R262Q compound het most common; Q106R partial LOF → partial puberty arrested",
            "GPCR 7TM on pituitary gonadotrophs; Gq → PLC → Ca2+ → LH/FSH release",
            "LH/FSH blunted on GnRH bolus (100mcg) — confirms pituitary-level resistance",
            "Reversal uncommon in complete LOF; partial LOF (Q106R) may occasionally reverse",
        ],
        "key_ddx": (
            "ANOS1/FGFR1/KS genes: Kallmann — anosmia + absent olfactory bulbs; GNRHR = normosmic; "
            "KISS1R (nIHH): normosmic IHH — kisspeptin pathway; pituitary responds to pulsatile GnRH "
            "(KISS1R upstream of GnRH neurons, not receptor); "
            "Panhypopituitarism: LH + FSH + TSH + ACTH + GH all low; MRI pituitary lesion; "
            "Constitutional delayed puberty: bone age delay; spontaneous puberty; normal smell."
        ),
        "systemic_involvement": (
            "ENDOCRINE: HH; absent puberty; infertility; osteopenia. "
            "OLFACTORY: NORMAL (distinguishes from Kallmann). "
            "NO craniofacial anomalies; NO cardiac; NO renal anomalies."
        ),
        "onset_age": "Absent/partial puberty at 11-14y; no other congenital anomalies",
        "surgical_urgency": "No emergency; orchidopexy if cryptorchidism; HH treatment from 14-16y",
        "gene_family": "G-protein coupled receptor (GPCR); rhodopsin subfamily; 7-transmembrane; GnRH receptor",
        "morphology": (
            "MRI BRAIN: NORMAL olfactory bulbs (DDx Kallmann); "
            "HORMONAL: LH <1 IU/L, FSH <2 IU/L, testosterone <100 ng/dL; "
            "SMELL TEST: NORMAL (UPSIT ≥34/40); "
            "GnRH PUMP TRIAL: no LH/FSH rise — PATHOGNOMONIC"
        ),
        "n_patients": 40,
    },

    # -- KISS1R -- Normosmic IHH (AR; Kisspeptin Receptor) ------------------------------------------
    {
        "gene": "KISS1R",
        "alt_name": (
            "KISS1R (KISS1R-398aa-19p13.3 / AR -- "
            "GPR54-KISSPEPTIN-RECEPTOR-NORMOSMIC-IHH -- "
            "NO-ANOSMIA-OLFACTORY-BULBS-PRESENT-DDx-KALLMANN -- "
            "PUBERTY-SWITCH-KISSPEPTIN-KISS1-GNRH-PULSE-INITIATOR -- "
            "REVERSAL-PHENOMENON-POSSIBLE -- "
            "L102P-R331X-MOST-COMMON-AR-FOUNDER)"
        ),
        "protein": (
            "KISS1R -- 19p13.3 AR -- KISS1R-398aa -- "
            "Kisspeptin-Receptor-GPR54-45kDa-GPCR-7TM-Gq-GnRH-Pulse-Initiator -- "
            "Hypothalamic-KNDy-Neurons-GnRH-Pulse-Puberty-Switch -- "
            "OMIM-Gene-604161-Disease-nIHH-146110"
        ),
        "locus": "19p13.3",
        "protein_size": "398 aa / ~45 kDa",
        "inheritance": (
            "AR (autosomal recessive; biallelic pathogenic variants); "
            "KISS1R (GPR54) is a GPCR that is the receptor for kisspeptin (encoded by KISS1). "
            "KISSPEPTIN-KISS1R AXIS is the key upstream initiator of GnRH pulsatility — the 'puberty switch.' "
            "MECHANISM: Kisspeptin (Kiss1) neurons in arcuate nucleus (KNDy neurons: Kisspeptin + "
            "Neurokinin B + Dynorphin) release kisspeptin → binds KISS1R on GnRH neurons → "
            "Gq-PLC-IP3-Ca2+ → GnRH neuron depolarisation → GnRH pulse released. "
            "KISS1R LOF → GnRH neurons present but cannot receive the 'fire' signal from KNDy → "
            "GnRH pulses absent → no LH/FSH → HH. "
            "GnRH NEURONS INTACT: unlike KS1-6 where GnRH neurons fail to migrate; in KISS1R LOF "
            "GnRH neurons ARE in hypothalamus (migration completed normally) — defect is upstream signalling. "
            "PITUITARY INTACT: GnRH receptor (GNRHR) also intact — if given exogenous pulsatile GnRH, "
            "pituitary DOES respond (LH/FSH rise) — KEY DDx from GNRHR (where pituitary does NOT respond). "
            "NORMOSMIC: olfactory bulbs PRESENT; smell NORMAL — KEY DDx from Kallmann. "
            "REVERSAL: spontaneous reversal of IHH can occur — kisspeptin pathway can be 'unblocked' "
            "by increasing body weight, stress reduction, metabolic recovery."
        ),
        "disease_category": (
            "Normosmic Isolated Hypogonadotropic Hypogonadism (nIHH) — AR; NO anosmia; "
            "kisspeptin receptor defect; GnRH neurons INTACT and present; pituitary INTACT; "
            "pulsatile GnRH pump IS effective (DDx GNRHR where pump fails); reversal phenomenon possible"
        ),
        "disease_pathway": (
            "KISS1R / KNDy AXIS: KNDy neurons (arcuate nucleus) release kisspeptin10 (decapeptide C-terminus) → "
            "KISS1R on GnRH neuron dendrites → Gαq11 → PLCβ → IP3 → Ca2+ oscillation → "
            "GnRH neuron action potential → GnRH exocytosis from median eminence → pituitary portal blood → "
            "GNRHR on gonadotrophs → LH/FSH pulsatile secretion. "
            "KISS1R LOF → kisspeptin cannot initiate GnRH pulses → GnRH neurons 'silent' despite being "
            "in correct location and being architecturally normal. "
            "PUBERTY SWITCH: kisspeptin-KISS1R is the molecular switch for puberty onset; KISS1R LOF → "
            "puberty switch never flipped → absent puberty. "
            "PULSATILE GnRH BYPASSES BLOCK: exogenous pulsatile GnRH can activate GNRHR directly → "
            "LH/FSH rise → sex steroid production → fertility possible; "
            "effectively 'bypasses' the KISS1R → GNRH step. "
            "REVERSAL: endogenous kisspeptin activity recovers in some patients (metabolic, "
            "weight-related) → spontaneous puberty; withdrawal trial essential."
        ),
        "pathognomonic": (
            "KISS1R / nIHH PATHOGNOMONIC FEATURES: "
            "1. NORMOSMIC: smell test NORMAL (UPSIT ≥34); olfactory bulbs PRESENT on MRI — "
            "KEY DDx from all Kallmann syndromes; "
            "2. PULSATILE GnRH PUMP WORKS: pituitary is INTACT; exogenous pulsatile GnRH → LH/FSH RISE — "
            "DDx from GNRHR where pituitary does NOT respond; "
            "3. LH PULSE ANALYSIS: prolonged LH sampling (12-24h, every 10 min) → absent or very infrequent "
            "LH pulses (GnRH neuron silent); "
            "4. REVERSAL PHENOMENON: ~10-20% of KISS1R cases show spontaneous testosterone/oestradiol "
            "recovery → reversal of IHH; always trial withdrawal; "
            "5. FOUNDER MUTATIONS: L102P (Turkish founder); R331X (various); "
            "6. HORMONAL: LH very low (<0.5 IU/L), FSH low-normal, testosterone <100 ng/dL; "
            "7. NO ASSOCIATED FEATURES: no craniofacial, no cardiac, no renal — pure IHH."
        ),
        "treatment": (
            "PUBERTY INDUCTION: "
            "MALES: testosterone IM or transdermal — standard; "
            "FEMALES: oestrogen/progesterone — standard; "
            "FERTILITY (CRITICAL): pulsatile GnRH pump DOES WORK (pituitary + GnRH receptor intact); "
            "GnRH pump 5-20 mcg per pulse every 90 min → spermatogenesis in 60-70%; "
            "best fertility outcomes of normosmic IHH genes (because both pituitary and GnRH neuron intact); "
            "ALTERNATIVE: hCG + FSH injections; "
            "REVERSAL: trial withdrawal at 3-6 months; LH/FSH/testosterone monitoring; "
            "spontaneous puberty can occur — document carefully; "
            "MONITORING: testosterone, LH, FSH, sperm; DEXA; "
            "GENETIC COUNSELLING: AR; 25% recurrence risk."
        ),
        "key_features": [
            "Normosmic IHH: olfactory bulbs PRESENT; smell NORMAL — DDx from all Kallmann genes",
            "Puberty switch gene: kisspeptin-KISS1R initiates GnRH pulses in KNDy arcuate neurons",
            "Pulsatile GnRH pump WORKS (pituitary intact + GNRHR intact) — DDx GNRHR where pump fails",
            "Reversal phenomenon possible (~10-20%): spontaneous testosterone/oestradiol recovery",
            "L102P Turkish founder; R331X; biallelic LOF required",
            "GnRH neurons in correct hypothalamic location (migration normal) — firing defect only",
            "Best fertility outcomes with GnRH pump (full signalling chain intact downstream of KISS1R)",
            "AR; no associated craniofacial/cardiac/renal anomalies — pure IHH",
        ],
        "key_ddx": (
            "Kallmann genes (ANOS1/FGFR1/etc.): anosmia + absent olfactory bulbs — KISS1R is normosmic; "
            "GNRHR (nIHH): pituitary does NOT respond to pulsatile GnRH — KISS1R pituitary DOES respond; "
            "KISS1 (kisspeptin ligand): same pathway — LOF of ligand vs. receptor; allelic disease; "
            "LEP/LEPR (obesity-IHH): leptin deficiency causes IHH; profound obesity; no anosmia; "
            "Constitutional delayed puberty: bone age delayed; spontaneous puberty; normal smell + LH pulses."
        ),
        "systemic_involvement": (
            "ENDOCRINE: HH; absent puberty; infertility; osteopenia. "
            "OLFACTORY: NORMAL. "
            "NO craniofacial, cardiac, renal anomalies. "
            "NEUROLOGICAL: GnRH neurons structurally normal — functional silence."
        ),
        "onset_age": "Absent puberty at 11-14y; no other congenital anomalies; reversal may occur 20-30y",
        "surgical_urgency": "No emergency; orchidopexy if cryptorchidism; HH treatment from 14-16y",
        "gene_family": "G-protein coupled receptor (GPCR); kisspeptin receptor; 7TM; Gq signalling",
        "morphology": (
            "MRI BRAIN: NORMAL olfactory bulbs (DDx Kallmann); GnRH neuron histology normal (if autopsy); "
            "HORMONAL: LH <0.5 IU/L, FSH low, testosterone <100 ng/dL; "
            "LH PULSE ANALYSIS: absent/infrequent pulses over 12-24h; "
            "GnRH PUMP TRIAL: LH/FSH DO RISE — DDx GNRHR"
        ),
        "n_patients": 40,
    },
]


# ── Patient simulation ──────────────────────────────────────────────────────────────────────────────

def _make_patients(gene_entry: dict) -> list[dict]:
    rng = random.Random(SEED_BASE + IHH_GENES.index(gene_entry))
    gene = gene_entry["gene"]
    patients = []
    for i in range(gene_entry["n_patients"]):
        age = rng.randint(14, 55)

        # Sex — XLR: ANOS1 mostly males; others mixed
        if gene == "ANOS1":
            sex = "M" if i < 37 else "F"  # 37 males, 3 carrier females with mild phenotype
        elif gene == "CHD7":
            sex = rng.choice(["M", "M", "F"])  # slight male predominance in presentation
        else:
            sex = rng.choice(["M", "F"])

        # Phenotype (using hearing_severity field for IHH severity)
        if gene == "ANOS1":
            severity = rng.choice([
                "Complete IHH + anosmia (KS1)", "Complete IHH + anosmia (KS1)",
                "Complete IHH + anosmia + bimanual synkinesis",
                "Complete IHH + unilateral renal agenesis",
                "Partial puberty arrested + anosmia",
            ])
        elif gene == "FGFR1":
            # Variable expressivity
            if i < 10:
                severity = "Anosmia alone (incomplete penetrance — no HH)"
            elif i < 28:
                severity = rng.choice(["Complete IHH + anosmia (KS2)", "Partial IHH + hyposmia"])
            else:
                severity = rng.choice(["Complete IHH + cleft palate", "Complete IHH + dental agenesis",
                                       "Complete IHH + digital anomaly + anosmia"])
        elif gene == "PROKR2":
            severity = rng.choice([
                "Complete IHH + anosmia (KS3)", "Partial IHH + hyposmia",
                "Complete IHH + sleep disorder + anosmia",
                "Normosmic HH (partial penetrance)", "Partial puberty arrested + hyposmia",
            ])
        elif gene == "PROK2":
            severity = rng.choice([
                "Complete IHH + anosmia (KS4)", "Partial IHH + hyposmia + circadian disorder",
                "Complete IHH + obesity + anosmia",
                "Partial IHH + sleep phase delay", "Hyposmia alone (carrier)",
            ])
        elif gene == "CHD7":
            severity = rng.choice([
                "CHARGE + complete IHH + anosmia", "CHARGE + HH + coloboma",
                "CHARGE + HH + choanal atresia repaired",
                "CHARGE + HH + SCC aplasia + SNHL", "CHARGE + HH + cardiac defect repaired",
            ])
        elif gene == "FGF8":
            if i < 8:
                severity = "Anosmia alone (incomplete penetrance)"
            else:
                severity = rng.choice([
                    "Complete IHH + hyposmia (KS6)", "Partial IHH + hyposmia + cleft palate",
                    "Complete IHH + anosmia", "Partial puberty arrested + hyposmia",
                ])
        elif gene == "GNRHR":
            if i < 12:
                severity = "Partial puberty arrested (Q106R partial LOF)"
            else:
                severity = rng.choice([
                    "Complete nIHH normosmic (biallelic LOF)", "Complete nIHH normosmic — normal smell",
                    "Partial nIHH — some breast/testicular development arrested",
                ])
        elif gene == "KISS1R":
            if i < 6:
                severity = "Spontaneous reversal — testosterone/oestradiol recovered"
            else:
                severity = rng.choice([
                    "Complete nIHH normosmic — absent LH pulses",
                    "Partial nIHH — arrested puberty, reversal trial ongoing",
                    "Complete nIHH — GnRH pump effective",
                ])

        # Management
        if gene in ("ANOS1", "FGFR1", "PROKR2", "PROK2", "FGF8"):
            # Kallmann — GnRH-responsive pituitary
            if sex == "M":
                mgmt = rng.choice([
                    "Testosterone replacement (puberty induction)",
                    "Pulsatile GnRH pump (fertility)",
                    "hCG + rFSH (fertility)",
                    "Testosterone replacement ongoing",
                    "GnRH pump — spermatogenesis initiated",
                ])
            else:
                mgmt = rng.choice([
                    "Oestrogen/progesterone (puberty induction)",
                    "Pulsatile GnRH pump (fertility)",
                    "rFSH + rLH (fertility)",
                    "HRT ongoing",
                ])
        elif gene == "CHD7":
            mgmt = rng.choice([
                "Testosterone replacement (HH component of CHARGE)",
                "Oestrogen/progesterone (HH CHARGE)",
                "Pulsatile GnRH pump (fertility — CHARGE)", "hCG + rFSH",
                "Under developmental/intellectual disability support",
            ])
        elif gene == "GNRHR":
            # Pulsatile GnRH pump DOES NOT WORK
            if sex == "M":
                mgmt = rng.choice([
                    "Testosterone replacement (puberty)",
                    "hCG 1500 IU 3×/week (testosterone production)",
                    "hCG + rFSH (spermatogenesis)",
                    "Testosterone ongoing",
                ])
            else:
                mgmt = rng.choice([
                    "Oestrogen/progesterone (puberty)",
                    "rFSH + rLH (fertility)",
                    "HRT ongoing",
                ])
        elif gene == "KISS1R":
            if i < 6:
                mgmt = "Reversal — testosterone/oestradiol now normal, treatment withdrawn"
            elif sex == "M":
                mgmt = rng.choice([
                    "Pulsatile GnRH pump (fertility — GnRH-responsive)",
                    "Testosterone replacement",
                    "GnRH pump — spermatogenesis confirmed",
                    "hCG + rFSH (alternative)",
                ])
            else:
                mgmt = rng.choice([
                    "Pulsatile GnRH pump (fertility)",
                    "Oestrogen/progesterone",
                    "rFSH + rLH",
                ])

        # Extra events
        evs = []
        if gene == "ANOS1":
            evs.append("MRI: absent olfactory bulbs on coronal T2")
            if i < 20 and sex == "M":
                if rng.random() < 0.50:
                    evs.append("Bimanual synkinesis confirmed — mirror movements left/right hand")
            if sex == "M" and rng.random() < 0.25:
                evs.append("Renal USS: unilateral renal agenesis (right absent)")
            if sex == "M" and rng.random() < 0.95:
                evs.append("Cryptorchidism bilateral — orchidopexy completed")
            evs.append(f"UPSIT smell score: {rng.randint(4, 16)}/40 (anosmia <18)")
        elif gene == "FGFR1":
            evs.append("MRI: hypoplastic olfactory bulbs (bilateral, variable)")
            if rng.random() < 0.13:
                evs.append("Cleft palate — repaired in infancy")
            if rng.random() < 0.10:
                evs.append("Dental agenesis: missing upper lateral incisors")
            if rng.random() < 0.08:
                evs.append("Digital anomaly: preaxial polydactyly right hand")
            evs.append(f"UPSIT smell score: {rng.randint(8, 22)}/40 (anosmia/hyposmia)")
        elif gene == "PROKR2":
            evs.append("MRI: variably hypoplastic olfactory bulbs")
            if rng.random() < 0.40:
                evs.append("Sleep diary: delayed sleep phase; melatonin prescribed")
            if rng.random() < 0.25:
                evs.append("Second hit identified: PROK2 heterozygous (digenic KS3/4)")
            evs.append(f"UPSIT smell score: {rng.randint(10, 26)}/40 (variable hyposmia)")
        elif gene == "PROK2":
            evs.append("MRI: olfactory bulb hypoplasia (variable severity)")
            if rng.random() < 0.50:
                evs.append("Actigraphy: circadian period lengthened; delayed sleep onset")
            if rng.random() < 0.30:
                evs.append("BMI: 28-33 (overweight/obese range — hypothalamic PROK2)")
            if rng.random() < 0.25:
                evs.append("Digenic: PROKR2 heterozygous confirmed on panel")
            evs.append(f"UPSIT smell score: {rng.randint(10, 24)}/40")
        elif gene == "CHD7":
            if rng.random() < 0.90:
                evs.append("CT temporal bone: semicircular canal aplasia confirmed")
            if rng.random() < 0.65:
                evs.append(rng.choice([
                    "Iris coloboma bilateral", "Chorioretinal coloboma unilateral",
                    "Optic disc coloboma", "Iris coloboma + microphtalmia"
                ]))
            if rng.random() < 0.45:
                evs.append(rng.choice([
                    "Choanal atresia repaired neonatal", "Bilateral choanal atresia — neonatal emergency managed"
                ]))
            if rng.random() < 0.75:
                evs.append(rng.choice([
                    "Cardiac defect repaired (VSD)", "ToF repaired", "ASD closed"
                ]))
            evs.append("MRI: olfactory bulb aplasia (near complete)")
        elif gene == "FGF8":
            evs.append("MRI: hypoplastic olfactory bulbs (moderate)")
            if rng.random() < 0.08:
                evs.append("Cleft lip/palate — repaired in infancy")
            if rng.random() < 0.05:
                evs.append("MRI cerebellum: vermis hypoplasia (mild)")
            evs.append(f"UPSIT smell score: {rng.randint(10, 24)}/40")
        elif gene == "GNRHR":
            evs.append("MRI: NORMAL olfactory bulbs — normosmic IHH confirmed")
            evs.append(f"UPSIT smell score: {rng.randint(35, 40)}/40 (NORMAL)")
            evs.append("GnRH pump trial: LH/FSH DID NOT RISE — pituitary receptor LOF confirmed")
            if sex == "M" and rng.random() < 0.40:
                evs.append("hCG therapy: testosterone rose to normal range — pituitary bypassed")
        elif gene == "KISS1R":
            evs.append("MRI: NORMAL olfactory bulbs — normosmic IHH confirmed")
            evs.append(f"UPSIT smell score: {rng.randint(35, 40)}/40 (NORMAL)")
            if i < 6:
                evs.append("Reversal: withdrawal trial at 30y — testosterone spontaneously normal at 1 year")
            else:
                if rng.random() < 0.60:
                    evs.append("GnRH pump trial: LH/FSH rose to normal — pituitary intact DDx GNRHR")
            if rng.random() < 0.15:
                evs.append("L102P (Turkish founder) or R331X compound heterozygous confirmed")

        patients.append({
            "id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "age": age,
            "sex": sex,
            "onset": gene_entry["onset_age"][:60],
            "hearing_severity": severity,  # repurposed: IHH/anosmia phenotype
            "management": mgmt,
            "key_features": "; ".join(gene_entry["key_features"][:3]),
            "extra_events": "; ".join(evs) if evs else "—",
        })
    return patients


def _all_patients() -> list[dict]:
    out = []
    for g in IHH_GENES:
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
        for g in IHH_GENES
    }

    return {
        "atlas": "Hereditary-Kallmann-IHH-Atlas",
        "subtitle": (
            "Complete 8-Gene Kallmann Syndrome / Isolated Hypogonadotropic Hypogonadism Reference "
            "(ANOS1 · FGFR1 · PROKR2 · PROK2 · CHD7 · FGF8 · GNRHR · KISS1R)"
        ),
        "total_patients": len(patients),
        "genes_covered": len(IHH_GENES),
        "seed_range": f"{SEED_BASE}–{SEED_BASE + len(IHH_GENES) - 1}",
        "patients_per_gene": gene_counts,
        "hearing_severity_distribution": phenotype_counts,
        "management_distribution": mgmt_counts,
        "gene_highlights": gene_highlights,
        "clinical_pearls": [
            "ANOS1 (KS1): absent olfactory bulbs MRI + bimanual synkinesis 50% + unilateral renal agenesis 25%; XLR",
            "FGFR1 (KS2): most common AD KS; cleft palate 10-15%; digital anomalies; incomplete penetrance ~30-40%",
            "PROKR2 (KS3): AR/digenic; variable hyposmia; sleep disorder; PROKR2 het alone often insufficient",
            "PROK2 (KS4): AR/digenic ligand for PROKR2; circadian disorder; obesity overlap; 81aa smallest",
            "CHD7 (CHARGE): anosmia + HH in CHARGE; SCC aplasia CT near-PATHOGNOMONIC; choanal atresia neonatal EMERGENCY",
            "FGF8 (KS6): same FGF8-FGFR1 pathway as KS2; cerebellar vermis hypoplasia rare; cleft palate",
            "GNRHR (nIHH): normosmic IHH — olfactory bulbs PRESENT; pulsatile GnRH pump DOES NOT WORK (receptor LOF)",
            "KISS1R (nIHH): normosmic IHH; pulsatile GnRH pump WORKS; reversal phenomenon possible ~10-20%",
        ],
    }


def get_breakdown() -> dict:
    """Per-gene breakdown for the /breakdown endpoint."""
    breakdown = {}
    for g in IHH_GENES:
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
    return {"breakdown_by_gene": breakdown, "total_genes": len(IHH_GENES)}


def get_definitions() -> dict:
    """Clinical definitions for the /definitions endpoint."""
    return {
        "genes": [g["gene"] for g in IHH_GENES],
        "seed_range": f"{SEED_BASE}–{SEED_BASE + len(IHH_GENES) - 1}",
        "definitions": {
            "Kallmann_Syndrome": (
                "Hypogonadotropic hypogonadism (HH) combined with anosmia/hyposmia due to defective "
                "GnRH neuron migration from olfactory placode to hypothalamus; olfactory bulbs absent/hypoplastic; "
                "genes include ANOS1, FGFR1, PROKR2, PROK2, CHD7, FGF8; GnRH-responsive pituitary in all KS"
            ),
            "Normosmic_IHH": (
                "Isolated hypogonadotropic hypogonadism WITHOUT anosmia; olfactory bulbs PRESENT on MRI; "
                "smell testing NORMAL (UPSIT ≥34/40); caused by GNRHR or KISS1R (or KISS1, GNRH1, etc.); "
                "CRITICAL DDx from Kallmann syndrome — same HH phenotype but smell NORMAL"
            ),
            "Olfactory_Bulb_MRI": (
                "Coronal T2 MRI through olfactory sulci: absent or severely hypoplastic olfactory bulbs = "
                "KALLMANN SYNDROME (any KS gene); present and normal-volume olfactory bulbs = normosmic IHH; "
                "MRI olfactory bulbs mandatory in ALL patients presenting with HH + absent puberty"
            ),
            "Pulsatile_GnRH_Pump": (
                "Subcutaneous pump delivering GnRH 5-20 mcg every 90 minutes; mimics physiological "
                "hypothalamic GnRH pulsatility; WORKS in Kallmann (KS1-6) and KISS1R — pituitary intact; "
                "DOES NOT WORK in GNRHR (receptor defective); gold standard fertility treatment in KS"
            ),
            "Bimanual_Synkinesis": (
                "Involuntary mirror movements: when one hand performs voluntary movement, the contralateral hand "
                "mirrors it involuntarily; ~50% of ANOS1/KS1; due to abnormal corticospinal tract decussation; "
                "TEST: tap left index finger rhythmically — observe right finger involuntary mirroring; "
                "highly specific for KS1 (ANOS1) — rarely in other KS genes"
            ),
            "Reversal_Phenomenon": (
                "Spontaneous recovery of hypothalamic-pituitary-gonadal (HPG) axis in ~10-20% of IHH patients; "
                "testosterone/oestradiol returns to normal without treatment; "
                "ALWAYS perform withdrawal trial at 3-6 months — stop treatment, monitor LH/FSH/testosterone; "
                "more common with partial LOF alleles (PROKR2, PROK2, KISS1R, FGFR1)"
            ),
            "Oligogenic_Digenic_KS": (
                "Single heterozygous variant in one KS gene (e.g., PROKR2) may be insufficient for full KS; "
                "second hit in another KS gene (e.g., PROK2, FGFR1, ANOS1) tips balance to disease; "
                "explains variable expressivity — carriers with one variant have partial phenotype (anosmia alone); "
                "full PANEL TESTING mandatory in all KS patients (not just single gene)"
            ),
            "UPSIT_Smell_Testing": (
                "University of Pennsylvania Smell Identification Test: 40-item scratch-and-sniff standardised test; "
                "ANOSMIA: <18/40; HYPOSMIA: 18-33/40; NORMAL: ≥34/40; "
                "MANDATORY in ALL patients presenting with delayed puberty/HH to distinguish Kallmann from nIHH; "
                "Sniffin Sticks (Threshold-Discrimination-Identification) also validated"
            ),
            "HPG_Axis_Testing": (
                "Low basal LH (<1 IU/L), FSH (<2 IU/L), testosterone (<100 ng/dL males) = IHH; "
                "GnRH bolus test (100 mcg IV): LH rise <4 IU/L = severe deficiency; "
                "pulsatile GnRH pump trial (5 mcg/90 min × 7 days): LH/FSH rise = pituitary intact (KS, KISS1R); "
                "no LH/FSH rise with pump = GNRHR LOF (pituitary receptor defective)"
            ),
            "Orchidopexy_Timing": (
                "Undescended testes (cryptorchidism) in almost all males with ANOS1/KS1; "
                "TIMING: orchidopexy before 12 months (ideally 6-12 months); "
                "RATIONALE: reduces testicular malignancy risk; preserves fertility potential; "
                "do NOT wait for HH treatment — orchidopexy is independent of hormonal therapy"
            ),
            "Testosterone_Induction_Protocol": (
                "Standard male HH puberty induction: testosterone enanthate 50 mg IM/month → "
                "increase by 50 mg every 6 months over 2-3 years → adult dose 250 mg/month; "
                "monitors: mid-injection testosterone (target 400-700 ng/dL), LH suppressed (expected), "
                "haematocrit <52%; bone density DEXA at baseline and every 2 years"
            ),
            "CHARGE_HH": (
                "Hypogonadotropic hypogonadism in CHARGE syndrome (CHD7): present in ~60-80%; "
                "often under-recognised clinically (overshadowed by cardiac, airway, vision, hearing issues); "
                "SCREEN ALL CHARGE patients at puberty age: LH, FSH, testosterone/oestradiol; "
                "HH management as per other KS — testosterone/oestrogen induction; pulsatile GnRH for fertility"
            ),
            "Gonadotropin_Therapy_vs_GnRH": (
                "GNRHR patients: pulsatile GnRH pump fails (receptor LOF); use hCG 1500-2000 IU 3×/week "
                "(LH analog → testosterone) + rFSH 75-150 IU 3×/week (spermatogenesis); "
                "KS genes + KISS1R: pulsatile GnRH pump preferred for fertility (physiological stimulation); "
                "gonadotropins (hCG + rFSH) are equally effective alternative for spermatogenesis"
            ),
            "KNDy_Neurons": (
                "Kisspeptin-Neurokinin B-Dynorphin (KNDy) neurons in hypothalamic arcuate nucleus: "
                "pulse generator for GnRH release; kisspeptin binds KISS1R on GnRH neurons to trigger "
                "GnRH pulse; Neurokinin B amplifies; Dynorphin terminates pulse — 90-min cycle; "
                "KISS1R LOF → KNDy cannot trigger GnRH pulses → silent GnRH neurons → HH"
            ),
        },
        "emergency_protocols": [
            "BILATERAL CHOANAL ATRESIA (CHD7/CHARGE): McGovern nipple or oral airway immediately; neonatal choanoplasty URGENT",
            "HYPOCALCAEMIC TETANY (if GATA3 overlap): IV calcium gluconate 10% SLOW IV; ECG monitor",
            "ORCHIDOPEXY TIMING: schedule before 12 months in all males with cryptorchidism",
            "GNRHR FERTILITY: do NOT use GnRH pump — ineffective; use hCG + rFSH instead; counsel early",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (ANOS1) ===")
    bk = get_breakdown()
    print(json.dumps(bk["breakdown_by_gene"]["ANOS1"], indent=2)[:2000])
    print("\n=== DEFINITIONS (first 1000 chars) ===")
    df = get_definitions()
    print(json.dumps(df, indent=2)[:1000])
