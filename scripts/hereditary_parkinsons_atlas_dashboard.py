#!/usr/bin/env python3
"""Hereditary-Parkinson's-Disease-Atlas — Complete 8-Gene Hereditary Parkinson's Disease Atlas
LRRK2   (Leucine-Rich Repeat Kinase 2; 2527 aa; 12q12; AD;
          PARK8 — most common identifiable genetic cause of PD;
          Gly2019Ser kinase gain-of-function → Rab GTPase hyperphosphorylation → impaired vesicular trafficking;
          Ashkenazi Jews 15-20%, North African Arabs 40%, general population 1-2%;
          incomplete penetrance 25-80%; kinase inhibitor trials DNL201/DNL151 open) ·
PRKN    (Parkin RBR E3 Ubiquitin Protein Ligase; 465 aa; 6q25.2-q27; AR;
          PARK2 — most common early-onset AR PD (onset 20-50 years);
          exon rearrangements 50% of variants — standard sequencing MISSES; MLPA mandatory;
          excellent levodopa response; early dyskinesias; very slow progression;
          sleep benefit; dopamine agonist first if young-onset) ·
PINK1   (PTEN-Induced Kinase 1; 581 aa; 1p36.12; AR;
          PARK6 — second most common AR PD; PINK1-Parkin mitophagy pathway;
          similar to PRKN clinically — early onset, slow progression, excellent levodopa;
          psychiatric comorbidities frequent; MRI normal) ·
SNCA    (Alpha-Synuclein; 140 aa; 4q22.1; AD;
          PARK1/PARK4 — point mutations rare (A53T, A30P, E46K);
          duplications: mild-moderate PD; triplications: severe PD + dementia (DLB-like);
          synuclein aggregates = pathological hallmark of ALL PD;
          anti-synuclein trials prasinezumab/cinpanemab; MLPA mandatory for copy number) ·
DJ-1    (DJ-1 Protein / Parkinsonism-Associated Deglycase; 189 aa; 1p36.23; AR;
          PARK7 — rare AR early-onset PD; mild tremor-predominant; very slow progression;
          L166P most recurrent variant; oxidative stress sensor;
          good levodopa response; MRI normal — no iron accumulation) ·
GBA     (Glucocerebrosidase; 497 aa; 1q22; biallelic AR Gaucher / heterozygous AD PD risk;
          most common genetic risk factor for PD (5-15% of all PD patients);
          heterozygous = 5-8x lifetime PD risk; severe allele L444P > earlier/more cognitive;
          ambroxol Phase 2/3 trial; cognitive decline more rapid than idiopathic PD) ·
VPS35   (Vacuolar Protein Sorting-Associated Protein 35; 796 aa; 16q11.2; AD;
          PARK17 — retromer complex; D620N sole recurrent variant (~95% of VPS35-PD);
          late-onset PD; levodopa responsive; good prognosis; retromer therapeutic target) ·
ATP13A2 (ATPase Cation Transporting 13A2; 1180 aa; 1p36.13; AR;
          PARK9 / Kufor-Rakeb Syndrome — atypical parkinsonism with pyramidal signs,
          supranuclear gaze palsy, dementia; MRI iron in putamen/caudate PATHOGNOMONIC;
          lysosomal ATPase dysfunction; levodopa initially responsive then wanes)
320-patient aggregate cohort (8 × 40, seeds 1502–1509)
"""

import random

SEED_BASE = 1502

PARKINSONS_GENES = [
    # ── LRRK2 — Most Common Identifiable Genetic PD ──
    {
        "gene": "LRRK2",
        "protein": "Leucine-Rich Repeat Kinase 2 — Most Common Genetic PD, G2019S Kinase Gain-of-Function",
        "alias": (
            "LRRK2; OMIM gene 609007; PARK8 OMIM 607060; 12q12; 2527 aa; ~286 kDa; "
            "LRRK2 encodes a large multidomain GTPase-kinase (Roco family) expressed in brain, "
            "kidney, lung, and leucocytes. The protein contains LRR (leucine-rich repeat), ROC "
            "(Ras of complex proteins), COR (C-terminal of ROC), kinase, and WD40 domains. "
            "LRRK2 phosphorylates a subset of Rab GTPases (Rab8A, Rab10, Rab35) at a conserved "
            "Thr/Ser within their switch II region, controlling vesicular trafficking, lysosomal "
            "function, and ciliogenesis. Gly2019Ser (p.G2019S, c.6055G>A): located in the kinase "
            "activation loop (DYG motif); gain-of-function → 2-3x increased kinase activity → "
            "Rab GTPase hyperphosphorylation → impaired recycling → synaptic vesicle and "
            "autophagy defects. G2019S is the most common PD-causing variant globally: prevalence "
            "among PD patients — Ashkenazi Jews 15-20%, North African Arabs ~40%, Southern Europeans "
            "5-6%, general European/North American 1-2%. Incomplete penetrance: 25% by age 59, "
            "45% by age 69, 80% by age 79 — age-dependent, modified by environmental factors "
            "(smoking appears paradoxically protective in epidemiological studies). Clinical: "
            "typical levodopa-responsive PD, tremor-predominant, slow progression, benign course. "
            "Kinase inhibitor trials DNL201 and DNL151 (Denali Therapeutics) target LRRK2 kinase "
            "directly; urine phospho-Rab10 (pThr73-Rab10) serves as pharmacodynamic biomarker."
        ),
        "aa": "2527 aa",
        "kDa": "~286 kDa",
        "locus": "12q12",
        "omim_gene": 609007,
        "omim_disease": 607060,
        "inheritance": "AD — gain-of-function kinase hyperactivation; incomplete penetrance 25-80% age-dependent",
        "gene_class": (
            "LRRK2 is a Roco family GTPase-kinase that serves as a master regulator of endolysosomal "
            "vesicle dynamics. The protein cycles between cytosol and membrane (trans-Golgi network, "
            "recycling endosomes, autophagic vesicles), where it recruits and phosphorylates Rab "
            "GTPases. Rab hyperphosphorylation by LRRK2-G2019S locks Rabs in an inactive GDP-bound "
            "state, preventing their dissociation from effectors and impairing endosomal sorting. "
            "This disrupts multiple downstream pathways: autophagy-lysosome pathway (impaired "
            "mitophagy), ciliogenesis (Rab8/Rab10 control primary cilia assembly), synaptic vesicle "
            "recycling, and neurite outgrowth. Alpha-synuclein clearance is impaired downstream of "
            "Rab dysregulation, creating a feed-forward loop accelerating Lewy pathology. "
            "The G2019S variant has been estimated to account for ~5-6% of familial PD and 1-2% of "
            "sporadic PD worldwide, making it the largest single known genetic contributor to PD. "
            "LRRK2 kinase inhibitors reduce phospho-Rab10 levels and are entering Phase 2/3 trials; "
            "the challenge is off-target effects (lung, kidney where LRRK2 is highly expressed). "
            "Urine pThr73-Rab10 (normalized to total Rab10) provides a non-invasive PD biomarker "
            "and kinase inhibitor pharmacodynamic readout."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("p.G2019S Gly2019Ser kinase activation-loop — most common PD variant globally", 0.78),
            ("p.R1441G/C/H Arg1441 ROC domain — GTP-binding impairment", 0.12),
            ("p.Y1699C Tyr1699 COR domain — GTPase-kinase interface", 0.05),
            ("p.I2020T atypical kinase domain — reduced GTPase activity", 0.03),
            ("p.G2385R Asian risk variant — heterozygous risk modifier", 0.02),
        ],
        "key_alerts": [
            "LRRK2-TEST-ASHKENAZI-NORTH-AFRICAN-MANDATORY: G2019S prevalence 15-20% (Ashkenazi) and ~40% (North African Arab) PD patients — targeted testing mandatory in these populations",
            "LRRK2-LEVODOPA-EXCELLENT-RESPONSE: LRRK2-PD responds excellently to levodopa — typical PD treatment approach; no deviation from standard motor management",
            "LRRK2-SLOW-PROGRESSION-TREMOR-PREDOMINANT: Generally benign motor course; tremor-predominant phenotype; slow progression; normal life expectancy common",
            "LRRK2-KINASE-INHIBITOR-TRIALS-DNL201-DNL151-Enrol-Eligible: Denali Therapeutics kinase inhibitor trials — confirm G2019S and enrol eligible patients; LRRK2 kinase inhibition = first disease-modifying approach",
            "LRRK2-INCOMPLETE-PENETRANCE-25-80pct-Age-Dependent: Penetrance 25% at age 59, 45% at 69, 80% at 79 — counsel pre-symptomatic carriers carefully; age-dependent risk not deterministic",
            "LRRK2-URINE-BIOMARKER-Rab10-pThr73-Research-Use: Urine phospho-Rab10 (pThr73) as LRRK2 kinase activity biomarker — used in clinical trials for pharmacodynamic monitoring",
        ],
    },
    # ── PRKN — Most Common AR Early-Onset PD ──
    {
        "gene": "PRKN",
        "protein": "Parkin RBR E3 Ubiquitin Ligase — Most Common AR PD, MLPA Mandatory for Exon Rearrangements",
        "alias": (
            "PRKN (formerly PARK2); OMIM gene 602544; PARK2 OMIM 600116; 6q25.2-q27; 465 aa; ~52 kDa; "
            "PRKN encodes parkin, a RING-between-RING (RBR) E3 ubiquitin ligase that operates downstream "
            "of PINK1 in the mitophagy pathway. Parkin is recruited to depolarised mitochondria by "
            "PINK1-mediated phosphorylation of ubiquitin (Ser65) and parkin's own Ubl domain, "
            "activating its E3 ligase activity to ubiquitinate outer mitochondrial membrane proteins "
            "(VDAC1, MFN1/2, MIRO1/2) and flag damaged mitochondria for autophagic clearance. "
            "PRKN is the most common cause of early-onset autosomal recessive PD: onset typically "
            "20-50 years (range 6-72); accounts for ~50% of familial AR PD in young-onset cohorts. "
            "CRITICAL DIAGNOSTIC POINT: exon rearrangements (deletions and duplications) account "
            "for ~50% of pathogenic PRKN variants — these copy number changes are completely missed "
            "by Sanger sequencing and standard short-read WES. MLPA (multiplex ligation-dependent "
            "probe amplification) or aCGH (array comparative genomic hybridisation) is MANDATORY "
            "in any patient with young-onset AR-PD, even if WES reports no PRKN variants. "
            "Clinical: excellent levodopa response; early motor fluctuations and dyskinesias expected; "
            "very slow disease progression; normal life expectancy; sleep benefit (symptoms better "
            "on wakening); tremor often the dominant feature; marked response to exercise."
        ),
        "aa": "465 aa",
        "kDa": "~52 kDa",
        "locus": "6q25.2-q27",
        "omim_gene": 602544,
        "omim_disease": 600116,
        "inheritance": "AR — biallelic loss-of-function; exon rearrangements 50% of alleles — MLPA mandatory",
        "gene_class": (
            "Parkin is an RBR (RING1-IBR-RING2) E3 ubiquitin ligase that exists in an autoinhibited "
            "conformation in healthy cells. Upon mitochondrial depolarisation, PINK1 accumulates on "
            "the outer mitochondrial membrane (OMM) and phosphorylates both ubiquitin (pSer65-Ub) "
            "and parkin's Ubl domain (pSer65-Ubl), triggering a cascade of allosteric changes that "
            "open the RING2 domain active site and allow transfer of ubiquitin to OMM substrates. "
            "Parkin builds polyubiquitin chains on VDAC1, MIRO1/2, MFN1/2, and other OMM proteins, "
            "recruiting the autophagy adapters NDP52 and OPTN, which in turn recruit ULK1 and the "
            "autophagosome machinery to engulf and degrade the mitochondrion. Loss of parkin activity "
            "prevents mitophagy, leading to accumulation of damaged, ROS-producing mitochondria, "
            "bioenergetic failure, and dopaminergic neuron death. "
            "Monoallelic (single heterozygous) PRKN exon rearrangements are an incomplete finding — "
            "a second allele must be identified before a diagnosis of PRKN-PD can be made with certainty; "
            "digenic interactions with other PD genes (DJ-1, PINK1) have been reported in rare families."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("exon deletion/duplication — single heterozygous — second allele pending MLPA", 0.32),
            ("exon deletion — compound biallelic — diagnostic by MLPA/aCGH", 0.28),
            ("p.R275W Arg275Trp RING1 domain — catalytic impairment", 0.15),
            ("p.K161N Lys161Asn UBL domain — parkin autoinhibition disrupted", 0.10),
            ("p.C289G Cys289Gly RING1 domain — zinc-binding disruption", 0.08),
            ("p.T240M splice-site variant — altered mRNA processing", 0.07),
        ],
        "key_alerts": [
            "PRKN-MLPA-MANDATORY-Exon-Rearrangements-50pct-Sequencing-MISSES: Exon deletions/duplications = 50% of PRKN variants — standard WES/Sanger completely blind to these; always perform MLPA/aCGH",
            "PRKN-EARLY-ONSET-Test-First-In-AR-PD-Under-50: First gene to test in any patient with AR PD onset under age 50; highest diagnostic yield in young-onset cohort",
            "PRKN-EXCELLENT-LEVODOPA-RESPONSE-Early-Dyskinesias-Expected: Excellent levodopa response is characteristic; counsel early that motor fluctuations and dyskinesias are expected and manageable",
            "PRKN-SLOW-PROGRESSION-Normal-Life-Expectancy: Very slow disease progression; normal life expectancy in many patients; distinguish from atypical parkinsonism syndromes",
            "PRKN-EXERCISE-NEUROPROTECTIVE-Aerobic-High-Intensity-PARK-Trial: High-intensity aerobic exercise — PARK trial evidence; prescribe structured exercise program; may slow progression",
            "PRKN-DOPAMINE-AGONIST-First-Delay-Levodopa-If-Young: In young-onset (<50): start dopamine agonist first to delay levodopa and reduce dyskinesia risk — then add levodopa as needed",
        ],
    },
    # ── PINK1 — Second Most Common AR PD ──
    {
        "gene": "PINK1",
        "protein": "PTEN-Induced Kinase 1 — AR PD PARK6, PINK1-Parkin Mitophagy Pathway",
        "alias": (
            "PINK1; OMIM gene 608309; PARK6 OMIM 605909; 1p36.12; 581 aa; ~66 kDa; "
            "PINK1 encodes a serine/threonine kinase targeted to mitochondria via its N-terminal "
            "mitochondrial targeting sequence (MTS). In healthy mitochondria, PINK1 is imported, "
            "cleaved by the inner membrane protease PARL, and retrotranslocated to the cytosol "
            "for proteasomal degradation — maintaining very low steady-state PINK1 levels. "
            "Upon mitochondrial depolarisation (loss of membrane potential, ΔΨm), import is "
            "arrested and PINK1 accumulates on the OMM, where it autophosphorylates and then "
            "phosphorylates ubiquitin (Ser65) and parkin (Ser65 of Ubl domain) to activate "
            "mitophagy. PINK1 is the second most common cause of AR early-onset PD (after PRKN), "
            "accounting for ~4-8% of young-onset AR-PD. Clinical phenotype: closely resembles "
            "PRKN-PD — early onset (teens to 40s typically), excellent levodopa response, early "
            "motor fluctuations and dyskinesias, very slow progression, sleep benefit. "
            "Psychiatric comorbidities (anxiety, depression) occur in a significant proportion. "
            "MRI is normal — structural neuroimaging does not show specific abnormalities, "
            "distinguishing PINK1 from ATP13A2 (Kufor-Rakeb) where iron accumulation is "
            "pathognomonic. Exercise and mitochondria-targeted therapies are rationally attractive."
        ),
        "aa": "581 aa",
        "kDa": "~66 kDa",
        "locus": "1p36.12",
        "omim_gene": 608309,
        "omim_disease": 605909,
        "inheritance": "AR — biallelic loss-of-function; kinase domain variants predominate",
        "gene_class": (
            "PINK1 is the apical sensor of mitochondrial damage in the PINK1-Parkin mitophagy pathway. "
            "The protein contains an N-terminal MTS, a single transmembrane segment, and a C-terminal "
            "serine/threonine kinase domain. PINK1 substrate phosphorylation creates the 'eat-me' "
            "signal on depolarised mitochondria by generating pSer65-ubiquitin, which then "
            "allosterically activates parkin's E3 ligase activity in a feed-forward amplification loop. "
            "PINK1 loss abolishes this quality control signal: damaged mitochondria evade autophagic "
            "clearance, accumulate, and produce excess reactive oxygen species, leading to bioenergetic "
            "failure in dopaminergic neurons of the substantia nigra pars compacta. "
            "The PINK1-Parkin pathway has been extensively modelled in Drosophila (where loss of "
            "Pink1 or Parkin produces identical flight muscle degeneration phenotypes) and mouse "
            "models. Therapeutic strategies targeting this pathway include NAD+ precursors "
            "(which boost mitochondrial biogenesis), PINK1 kinase activators, and ULK1 activators "
            "to rescue autophagosome recruitment. High-intensity aerobic exercise activates PGC-1α "
            "and mitochondrial biogenesis — the mechanistic basis for exercise as a disease-modifying "
            "intervention in PINK1-PD."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("p.Q456X Gln456Ter kinase domain truncation — complete LOF", 0.22),
            ("p.W437X Trp437Ter kinase domain truncation — complete LOF", 0.18),
            ("p.G309D Gly309Asp kinase activation segment — kinase dead", 0.20),
            ("p.R246X Arg246Ter — premature truncation", 0.15),
            ("p.L347P Leu347Pro kinase domain — misfolding/instability", 0.13),
            ("p.H271Q His271Gln kinase domain — catalytic impairment", 0.12),
        ],
        "key_alerts": [
            "PINK1-MITOPHAGY-PATHWAY-PINK1-Parkin-Ubiquitin-Mitochondria-Clearance: PINK1-Parkin pathway is the master regulator of mitophagy — therapeutic target rationale; explain pathway to patients",
            "PINK1-EARLY-ONSET-AR-Similar-PRKN-Clinically: Clinical phenotype nearly identical to PRKN-PD — early onset, slow progression, excellent levodopa; distinguish by genetic testing only",
            "PINK1-LEVODOPA-EXCELLENT-Early-Motor-Fluctuations: Excellent levodopa response; early motor fluctuations and dyskinesias are expected — same management as PRKN-PD",
            "PINK1-EXERCISE-THERAPEUTIC-Aerobic-Mitochondrial-Biogenesis: Aerobic exercise stimulates PGC-1α and mitochondrial biogenesis — rationally therapeutic; prescribe structured exercise",
            "PINK1-PSYCHIATRIC-Anxiety-Depression-Screen-Proactively: Anxiety and depression occur in a significant proportion — screen proactively at every visit; treat aggressively",
            "PINK1-MRI-NORMAL-No-Structural-Change-Expected: MRI brain normal — if iron accumulation seen on MRI, reconsider diagnosis; consider ATP13A2 or other NBIA causes",
        ],
    },
    # ── SNCA — Alpha-Synuclein ──
    {
        "gene": "SNCA",
        "protein": "Alpha-Synuclein — PARK1/PARK4, Synuclein Aggregates Pathological Hallmark of All PD",
        "alias": (
            "SNCA; OMIM gene 163890; PARK1 OMIM 168601; PARK4 OMIM 605543; 4q22.1; 140 aa; ~14 kDa; "
            "SNCA encodes alpha-synuclein, a small intrinsically disordered presynaptic protein that "
            "modulates vesicle dynamics, neurotransmitter release, and mitochondrial function. "
            "Alpha-synuclein aggregates (Lewy bodies and Lewy neurites) are the pathological hallmark "
            "of ALL Parkinson's disease — whether genetic or sporadic — making SNCA products the "
            "primary target for disease-modifying therapy. "
            "Point mutations are rare: p.A53T (Greek-Italian founder, most common SNCA mutation), "
            "p.A30P (German), p.E46K (Basque), p.H50Q, p.G51D, p.A53E — all cluster in the N-terminal "
            "amphipathic helix and promote aggregation or impair autophagic clearance. "
            "GENE DOSAGE EFFECT is the critical concept: normal SNCA has 2 copies (diploid). "
            "Duplication (3 copies): mild-moderate PD with onset similar to sporadic PD (50s-60s). "
            "Triplication (4 copies): severe, early-onset PD with dementia, resembling Dementia with "
            "Lewy Bodies (DLB-like phenotype) — because excess synuclein overwhelms clearance. "
            "MLPA or digital PCR is mandatory to detect copy number changes — standard sequencing "
            "reports the sequence as normal in duplication/triplication cases. "
            "Anti-synuclein immunotherapy trials: prasinezumab (Roche/Prothena), cinpanemab (Biogen)."
        ),
        "aa": "140 aa",
        "kDa": "~14 kDa",
        "locus": "4q22.1",
        "omim_gene": 163890,
        "omim_disease": 168601,
        "inheritance": "AD — point mutations; gene duplications (mild); gene triplications (severe+dementia)",
        "gene_class": (
            "Alpha-synuclein is a 140-amino acid presynaptic protein that adopts an alpha-helical "
            "conformation when bound to membrane phospholipids and a disordered conformation in solution. "
            "Physiologically, synuclein clusters synaptic vesicles, modulates SNARE complex assembly, "
            "and facilitates dopamine neurotransmitter release. Under pathological conditions "
            "(overexpression, mutation, post-translational modifications including phospho-Ser129, "
            "oxidative stress, impaired clearance), synuclein misfolds into beta-sheet-rich oligomers "
            "and amyloid fibrils — the toxic species that propagate intercellularly in a 'prion-like' "
            "fashion, explaining the Braak staging of Lewy pathology spreading from gut/olfactory bulb "
            "to brainstem to cortex. Gene dosage explains the phenotypic spectrum: each additional "
            "copy of the SNCA locus roughly doubles protein production, overwhelming the ubiquitin-"
            "proteasome system and autophagy-lysosome pathway clearance capacity. Triplication "
            "(4 copies) → very high synuclein burden → early and severe Lewy pathology → "
            "DLB-like dementia and parkinsonism by the 4th-5th decade. "
            "Anti-synuclein immunotherapy (prasinezumab, cinpanemab) targets extracellular fibrillar "
            "and oligomeric species to prevent intercellular propagation; LRRK2 inhibitors target "
            "Rab-dependent clearance pathways that underpin synuclein degradation."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("gene duplication — 3 copies — mild-moderate PD, typical onset", 0.40),
            ("gene triplication — 4 copies — severe PD + dementia, DLB-like", 0.25),
            ("p.A53T Ala53Thr Greek-Italian founder — rare point mutation", 0.20),
            ("p.A30P Ala30Pro German founder — rare point mutation", 0.08),
            ("p.E46K Glu46Lys Basque founder — rare point mutation", 0.07),
        ],
        "key_alerts": [
            "SNCA-TRIPLICATION-Severe-Early-Dementia-DLB-Phenotype-Distinguish: Triplication (4 copies) causes severe early-onset PD with prominent dementia — distinguish from DLB; annual MoCA mandatory",
            "SNCA-A53T-Greek-Founder-Test-Greek-Italian-Families: p.A53T is a Greek-Italian founder mutation — targeted testing mandatory in families of Greek or southern Italian descent",
            "SNCA-DUPLICATION-vs-TRIPLICATION-Gene-Dosage-Phenotype-Correlation: Gene dosage determines severity — 3 copies = typical PD; 4 copies = DLB-like dementia; copy number MUST be quantified",
            "SNCA-ANTI-SYNUCLEIN-TRIALS-Prasinezumab-Cinpanemab-Enrol-Eligible: Anti-synuclein immunotherapy trials (prasinezumab, cinpanemab) — confirm SNCA and enrol eligible patients",
            "SNCA-COPY-NUMBER-MLPA-Mandatory-Standard-Sequencing-MISSES-Multiplications: Standard WES/Sanger reports sequence as normal in duplication/triplication — MLPA or digital PCR mandatory to detect copy number",
            "SNCA-DEMENTIA-SCREEN-MOCA-Annual-Triplication-Phenotype: Annual MoCA in all SNCA patients, especially triplication — early dementia detection guides care planning and trial eligibility",
        ],
    },
    # ── DJ-1/PARK7 — Rare AR Mild PD ──
    {
        "gene": "DJ1",
        "protein": "DJ-1 Parkinsonism-Associated Deglycase — PARK7 Rare AR PD, Oxidative Stress Sensor",
        "alias": (
            "DJ1 (PARK7); OMIM gene 602533; PARK7 OMIM 606324; 1p36.23; 189 aa; ~20 kDa; "
            "DJ-1 encodes a small dimeric protein of the DJ-1/ThiJ/PfpI superfamily with multiple "
            "proposed functions: glyoxalase/deglycase activity (methylglyoxal and other toxic electrophile "
            "scavenging), RNA-binding chaperone, mitochondria-associated redox sensor, and transcriptional "
            "co-activator. DJ-1 is oxidised at its critical catalytic cysteine (Cys106) by reactive "
            "oxygen species (ROS); this oxidation-activated form translocates to depolarised mitochondria "
            "where it protects against oxidative damage, complements PINK1-parkin mitophagy, and "
            "prevents cytochrome c release. DJ-1 loss leaves dopaminergic neurons vulnerable to "
            "mitochondrial oxidative stress. PARK7 is a rare cause of AR early-onset PD, accounting "
            "for approximately 1-2% of young-onset AR-PD cohorts. Clinical: typically mild, tremor-"
            "predominant, very slow progression, excellent dopaminergic response. Sleep benefit "
            "reported. p.Leu166Pro (L166P) is the most recurrent pathogenic variant globally — "
            "it disrupts the protein's homodimerisation interface, causing rapid proteasomal "
            "degradation of the misfolded monomer. Entire exon 7 deletion is the second most "
            "common pathogenic mechanism. MRI is normal — iron does not accumulate, distinguishing "
            "DJ-1-PD from ATP13A2 (Kufor-Rakeb) where T2 hypointensity in basal ganglia is "
            "pathognomonic."
        ),
        "aa": "189 aa",
        "kDa": "~20 kDa",
        "locus": "1p36.23",
        "omim_gene": 602533,
        "omim_disease": 606324,
        "inheritance": "AR — biallelic loss-of-function; L166P most recurrent; exon 7 deletion common",
        "gene_class": (
            "DJ-1 is a member of the DJ-1/ThiJ/PfpI superfamily, functioning as a redox-sensitive "
            "chaperone and deglycase that protects cells from oxidative and electrophilic stress. "
            "Structurally, DJ-1 homodimerises through a hydrophobic interface; disease mutations "
            "that disrupt this interface (particularly L166P) destabilise the protein and target "
            "it for proteasomal degradation, effectively nullifying DJ-1 activity. "
            "The Cys106 residue is the primary redox sensor: mild oxidation (Cys-SO2H) activates "
            "DJ-1 chaperone function; overoxidation (Cys-SO3H) inactivates it. "
            "DJ-1 contributes to mitochondrial quality control in parallel with the PINK1-Parkin "
            "pathway, stabilising complex I of the mitochondrial electron transport chain and "
            "preventing permeability transition pore opening under ROS challenge. "
            "Loss of DJ-1 in dopaminergic neurons increases sensitivity to MPTP, rotenone, and "
            "6-OHDA neurotoxins — mechanistic basis for the 'toxic-hit' model of sporadic PD "
            "in genetically predisposed individuals. "
            "Exercise-induced antioxidant mechanisms (SOD2, GSH, HO-1 upregulation) may compensate "
            "for loss of DJ-1 deglycase activity — providing rationale for aerobic exercise prescription."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("p.L166P Leu166Pro dimerisation domain — protein instability/degradation", 0.45),
            ("exon 7 deletion — null allele — protein absent", 0.25),
            ("p.M26I Met26Ile oxidoreductase active site — reduced deglycase activity", 0.15),
            ("p.A104T Ala104Thr moderate structural disruption", 0.10),
            ("p.D149A Asp149Ala ATP-binding site disruption", 0.05),
        ],
        "key_alerts": [
            "DJ1-RARE-AR-Mild-Tremor-Slow-Progression: PARK7 is a rare but important AR PD cause — mild tremor-predominant phenotype with very slow progression; counsel accordingly",
            "DJ1-OXIDATIVE-STRESS-Sensor-Protects-Mitochondria: DJ-1 is an oxidative stress sensor protecting mitochondria from ROS; environmental oxidant exposures may modify disease course",
            "DJ1-L166P-Most-Common-Pathogenic-Variant-Folding-Disruption: L166P disrupts homodimerisation — protein is degraded; most recurrent pathogenic variant; targeted in genetic testing panels",
            "DJ1-LEVODOPA-GOOD-RESPONSE: Good to excellent levodopa response — standard PD pharmacotherapy applies; dopamine agonist first if young-onset (<50) to delay dyskinesias",
            "DJ1-EXERCISE-Antioxidant-Neuroprotective-Aerobic: Aerobic exercise upregulates antioxidant defence (SOD2, GSH) — rationally compensates for loss of DJ-1 deglycase; prescribe structured program",
            "DJ1-MRI-NORMAL-No-Iron-Accumulation-Distinguishes-ATP13A2: MRI brain normal — absence of iron in basal ganglia is a key negative finding; iron on MRI should trigger ATP13A2/NBIA workup",
        ],
    },
    # ── GBA — Most Common PD Risk Gene ──
    {
        "gene": "GBA",
        "protein": "Glucocerebrosidase — Most Common PD Risk Gene, Heterozygous 5-8x PD Risk, Ambroxol Trial",
        "alias": (
            "GBA; OMIM gene 606463; Gaucher disease OMIM 230800; 1q22; 497 aa; ~60 kDa; "
            "GBA encodes acid beta-glucocerebrosidase (GCase), a lysosomal enzyme that cleaves "
            "glucocerebroside (glucosylceramide) into glucose and ceramide. Biallelic loss-of-function "
            "GBA variants cause Gaucher disease (most common lysosomal storage disorder) — "
            "visceral and haematological manifestations; neuronopathic forms (Types 2, 3) cause "
            "brain involvement. The landmark finding (2008-2024): GBA heterozygous carriers — "
            "who do NOT have Gaucher disease — have a 5-8x lifetime risk of developing Parkinson's "
            "disease, making GBA the most common genetic risk factor for PD. GBA variants are found "
            "in 5-15% of ALL PD patients across diverse populations. Allele severity matters: "
            "severe alleles (p.L444P) confer greater PD risk and earlier cognitive decline than "
            "mild alleles (p.N370S). p.N370S (Asn370Ser): most common GBA variant in Ashkenazi Jews "
            "and Europeans; protects against Gaucher type 3 neurological form but increases PD risk. "
            "Mechanism: reduced GCase activity → glucosylceramide and glucosylsphingosine accumulation "
            "→ lysosomal dysfunction → impaired autophagy → alpha-synuclein aggregation "
            "(reciprocal relationship: synuclein also inhibits GCase, creating a feed-forward loop). "
            "Ambroxol (pharmacological chaperone for GCase) Phase 2/3 trial in GBA-PD (AiM-PD study). "
            "Cognitive decline in GBA-PD is more rapid than in idiopathic PD; annual MoCA mandatory."
        ),
        "aa": "497 aa",
        "kDa": "~60 kDa",
        "locus": "1q22",
        "omim_gene": 606463,
        "omim_disease": 230800,
        "inheritance": "Biallelic AR causes Gaucher disease; heterozygous AD risk modifier for PD (5-8x risk)",
        "gene_class": (
            "Glucocerebrosidase (GCase) is a lysosomal acid hydrolase that degrades glucosylceramide, "
            "a glycosphingolipid abundant in cell membranes, particularly in macrophages and Schwann cells. "
            "GCase is synthesised in the ER, chaperoned by LIMP-2/SCARB2 and saposin C for delivery "
            "to lysosomes, and activated at lysosomal pH (~4.5-5.0). Reduced GCase activity disrupts "
            "sphingolipid homeostasis and lysosomal function: glucosylceramide accumulation impairs "
            "lysosomal membrane integrity; glucosylsphingosine is directly neurotoxic. "
            "The GCase-synuclein feedback loop is critical to GBA-PD pathophysiology: reduced GCase "
            "activity impairs lysosomal degradation of alpha-synuclein; accumulated synuclein in turn "
            "sequesters GCase in the ER (by binding to the GCase/LIMP-2 complex), further reducing "
            "its lysosomal delivery — a vicious cycle accelerating Lewy pathology. "
            "Ambroxol (originally a mucolytic) is a pharmacological chaperone that binds the GCase "
            "active site at neutral pH, stabilising the protein for correct ER-to-lysosome trafficking, "
            "then releases at lysosomal pH — increasing lysosomal GCase activity. "
            "This mechanism explains GBA-PD's particular sensitivity to ambroxol as a therapeutic. "
            "The AiM-PD (Ambroxol in Parkinson's Disease) trial tests ambroxol 1260 mg/day in GBA-PD."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("p.N370S Asn370Ser mild Gaucher/PD modifier — heterozygous carrier", 0.45),
            ("p.L444P Leu444Pro severe allele — greater PD risk, earlier cognitive decline", 0.28),
            ("p.E326K Glu326Lys risk variant — moderate risk modifier", 0.12),
            ("p.T369M Thr369Met risk variant — mild risk modifier", 0.08),
            ("complex alleles RecNciI — pseudo-homozygosity on WES — MLPA needed", 0.07),
        ],
        "key_alerts": [
            "GBA-MOST-COMMON-PD-RISK-GENE-Test-ALL-PD-Patients: GBA variants in 5-15% of ALL PD — GBA testing should be offered to every PD patient regardless of family history",
            "GBA-HETEROZYGOUS-5-8x-PD-Risk-NOT-Gaucher: Single GBA variant = 5-8x lifetime PD risk; does NOT cause Gaucher disease — critical counselling distinction for heterozygous carriers",
            "GBA-BIALLELIC-Gaucher-Plus-PD-Risk-Both-Diagnoses: Biallelic GBA = Gaucher disease + high PD risk; co-manage with metabolic/Gaucher specialist and movement disorder neurologist",
            "GBA-AMBROXOL-TRIAL-Chaperone-Enzyme-Restoration-Enrol-Eligible: AiM-PD ambroxol trial — confirm GBA variant and enrol eligible patients; ambroxol as pharmacological GCase chaperone",
            "GBA-COGNITIVE-DECLINE-More-Rapid-Annual-MoCA: Cognitive decline more rapid than idiopathic PD — annual MoCA mandatory; early discussion of care planning and advance directives",
            "GBA-SEVERE-ALLELE-L444P-Earlier-Onset-More-Cognitive: L444P confers greater PD risk and earlier/more severe cognitive decline than N370S — allele-specific prognosis counselling",
            "GBA-LYSOSOMAL-PATHWAY-Ceramide-Accumulation-Synuclein-Aggregation: GCase-synuclein feed-forward loop — lysosomal dysfunction accelerates synuclein aggregation; explain mechanism to patients",
        ],
    },
    # ── VPS35 — Retromer Complex ──
    {
        "gene": "VPS35",
        "protein": "Vacuolar Protein Sorting 35 — PARK17 Retromer Complex, D620N Sole Recurrent Variant",
        "alias": (
            "VPS35; OMIM gene 601501; PARK17 OMIM 614203; 16q11.2; 796 aa; ~91 kDa; "
            "VPS35 encodes the largest subunit of the retromer cargo-recognition complex (CRC), "
            "a heterotrimer of VPS35-VPS26-VPS29 that sorts endosomal cargo (e.g., mannose-6-phosphate "
            "receptors, SorL1, Wls/wntless) into recycling tubules destined for the Golgi or "
            "plasma membrane. VPS35 acts as the scaffold subunit, bridging VPS26 (cargo-adaptor) "
            "and VPS29 (regulator) and recruiting the WASH complex (actin nucleation machinery) "
            "to endosomal recycling tubules. p.Asp620Asn (D620N, c.1858G>A) is the sole recurrent "
            "pathogenic variant in VPS35-PD, accounting for ~95% of all identified VPS35-PD alleles. "
            "D620N impairs VPS35 interaction with the WASH complex component FAM21, disrupting "
            "actin-mediated tubule formation and endosomal cargo retrieval. This leads to mis-sorting "
            "of lysosomal hydrolase receptors and impaired autophagy-lysosome pathway function, "
            "promoting synuclein accumulation. Clinical: late-onset PD (mean onset 50-60 years), "
            "phenotypically similar to typical sporadic PD — levodopa responsive, tremor often "
            "prominent, relatively good prognosis. AD inheritance with high (but not 100%) penetrance. "
            "First-degree relatives should be offered genetic testing and monitoring."
        ),
        "aa": "796 aa",
        "kDa": "~91 kDa",
        "locus": "16q11.2",
        "omim_gene": 601501,
        "omim_disease": 614203,
        "inheritance": "AD — gain-of-function/dominant-negative D620N; high penetrance; late-onset",
        "gene_class": (
            "The retromer complex retrieves transmembrane cargo from early/late endosomes and routes "
            "it either retrograde to the Golgi or anterograde to the plasma membrane, preventing "
            "lysosomal degradation of recycling receptors. VPS35 serves as the central scaffold: "
            "its N-terminal HEAT repeat domain contacts VPS26 (which directly recognises the "
            "Ø-x-[L/M] sorting motif of cargo cytoplasmic tails), and its C-terminal domain "
            "interacts with VPS29, which in turn recruits TBC1D5 (Rab7-GAP) and the WASH complex "
            "via FKBP15. D620N disrupts the VPS35-FAM21 interface (FAM21 is the WASH complex "
            "tether on endosomes), impairing WASH-mediated actin polymerisation on endosomal "
            "membranes, which is required for tubular carrier formation. The net effect: "
            "mannose-6-phosphate receptors (required for lysosomal enzyme delivery), AMPA receptors, "
            "and other cargoes are mis-sorted to lysosomes and degraded — impairing lysosomal enzyme "
            "delivery and synaptic receptor recycling. Retromer defects also impair mitochondria-"
            "associated ER membrane (MAM) function. Retromer chaperone compounds (R33, R55) stabilise "
            "the VPS35-VPS26 interface — preclinical therapeutic approach."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("p.D620N Asp620Asn retromer VPS35-WASH interface — sole recurrent variant", 0.95),
            ("other missense retromer cargo-recognition interface — very rare", 0.05),
        ],
        "key_alerts": [
            "VPS35-D620N-SOLE-VARIANT-95pct-Targeted-Panel-Sufficient: D620N accounts for ~95% of VPS35-PD — targeted single-variant assay sufficient after WES is unrevealing; check for D620N specifically",
            "VPS35-LATE-ONSET-Similar-Sporadic-PD: Late-onset (50-60 years mean); clinically indistinguishable from sporadic PD — genetic testing is the only distinguisher",
            "VPS35-RETROMER-COMPLEX-Endosomal-Cargo-Retrieval-Therapeutic-Target: Retromer complex dysfunction — retromer chaperone compounds in preclinical development; mechanistically important therapeutic target",
            "VPS35-LEVODOPA-GOOD-RESPONSE: Good levodopa response — standard PD motor management applies; typical PD treatment algorithm",
            "VPS35-PENETRANCE-High-AD-Test-First-Degree-Relatives: High but incomplete penetrance — first-degree relatives should be offered genetic testing and neurological monitoring",
            "VPS35-CHMP2B-VPS26A-Co-Test-Panel: Consider panel testing VPS26A and other retromer subunit genes if VPS35-D620N negative in suspected retromer-PD family",
        ],
    },
    # ── ATP13A2 — Kufor-Rakeb Syndrome ──
    {
        "gene": "ATP13A2",
        "protein": "ATPase Cation Transporting 13A2 — PARK9 Kufor-Rakeb Syndrome, MRI Iron PATHOGNOMONIC",
        "alias": (
            "ATP13A2; OMIM gene 610513; Kufor-Rakeb Syndrome PARK9 OMIM 606693; 1p36.13; 1180 aa; ~130 kDa; "
            "ATP13A2 encodes a P5B-type ATPase lysosomal membrane cation transporter involved in "
            "vacuolar acidification and polyamine/cation export from lysosomes to the cytosol. "
            "ATP13A2 is a lysosomal proton pump: it acidifies the lysosomal lumen, maintains "
            "lysosomal membrane potential, and transports polyamines (putrescine, spermidine, spermine) "
            "from lumen to cytosol. Loss of ATP13A2 causes lysosomal alkalinisation → impaired "
            "lysosomal hydrolase activity → defective autophagy → protein and lipid accumulation "
            "→ cell death. Kufor-Rakeb Syndrome (KRS) is an atypical parkinsonism — clinically "
            "distinct from typical PRKN/PINK1/LRRK2 PD: features pyramidal tract signs (spasticity, "
            "hyperreflexia), supranuclear upgaze palsy (oculomotor abnormality), and prominent "
            "cognitive decline/dementia. PATHOGNOMONIC IMAGING: T2/T2*-weighted MRI shows hypointensity "
            "(iron deposition) in the putamen and caudate nucleus — a neurodegeneration with brain "
            "iron accumulation (NBIA)-like pattern that DOES NOT occur in PRKN, PINK1, or DJ-1 PD "
            "(where MRI is normal). Levodopa initially provides motor benefit but effect wanes as "
            "the disease progresses and pyramidal involvement becomes dominant. "
            "p.A746T (Ala746Thr) is a Jordanian founder variant, found in the original Kufor-Rakeb "
            "pedigree from the village of Kufor-Rakeb, Jordan."
        ),
        "aa": "1180 aa",
        "kDa": "~130 kDa",
        "locus": "1p36.13",
        "omim_gene": 610513,
        "omim_disease": 606693,
        "inheritance": "AR — biallelic loss-of-function; lysosomal cation ATPase absent/dysfunctional",
        "gene_class": (
            "ATP13A2 is a P5B-type ATPase (a subclass of P-type ATPases) embedded in the lysosomal "
            "membrane, with 10 transmembrane segments, a large cytoplasmic actuator (A), "
            "phosphorylation (P), and nucleotide-binding (N) domain arrangement characteristic of "
            "P-type ATPases. The protein cycles through E1/E2 conformational states driven by ATP "
            "hydrolysis to transport divalent cations and polyamines across the lysosomal membrane. "
            "Loss of ATP13A2 impairs lysosomal proton gradient maintenance, reducing pH from ~4.5 "
            "to ~5.0-5.5 — sufficient to substantially impair acid hydrolase activity (cathepsins "
            "B, D, L all require pH <5.0 for optimal activity). This lysosomal dysfunction impairs: "
            "(1) mitophagy — damaged mitochondria are not cleared; (2) proteostasis — alpha-synuclein "
            "and other aggregation-prone proteins accumulate; (3) lipid metabolism — glucocerebrosidase "
            "(GCase/GBA) activity is reduced, mirroring the GBA-PD lysosomal defect. "
            "Iron accumulation in basal ganglia (NBIA-like) results from impaired lysosomal iron "
            "handling and ferritin autophagic degradation — lysosomes normally sequester and export "
            "iron released from haem degradation. T2 hypointensity on MRI reflects paramagnetic "
            "effect of non-haem iron deposits. This distinguishes KRS from all other PRKN-pathway "
            "AR-PD genes where MRI is structurally normal."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("p.A746T Ala746Thr Jordanian founder variant — original Kufor-Rakeb pedigree", 0.38),
            ("p.T12M exon 2 Jordanian founder — loss of translation initiation", 0.22),
            ("frameshift biallelic truncation — complete protein absence", 0.20),
            ("splice-site biallelic — aberrant mRNA splicing", 0.12),
            ("large deletion — ATP13A2 null — biallelic", 0.08),
        ],
        "key_alerts": [
            "ATP13A2-MRI-IRON-PUTAMEN-CAUDATE-T2-Hypointensity-PATHOGNOMONIC: T2/T2* hypointensity in putamen and caudate = iron deposition = PATHOGNOMONIC for KRS; MRI mandatory in all suspected ATP13A2-PD",
            "ATP13A2-PYRAMIDAL-SIGNS-Distinguish-Typical-PD: Pyramidal signs (spasticity, hyperreflexia) distinguish KRS from typical PD — mandatory neurological examination for pyramidal features",
            "ATP13A2-KUFOR-RAKEB-Supranuclear-Gaze-Palsy-Diagnostic-Clue: Supranuclear upgaze palsy is a diagnostic clue in KRS; examine eye movements in young-onset atypical parkinsonism",
            "ATP13A2-LEVODOPA-Initially-Responsive-Wanes-Pyramidal-Progresses: Levodopa provides early motor benefit; effect wanes as pyramidal and cognitive features progress — counsel family on trajectory",
            "ATP13A2-LYSOSOMAL-ATPase-Vacuolar-Acidification-Impaired: Lysosomal acidification failure — impaired autophagy; explain lysosomal disease mechanism and research implications",
            "ATP13A2-DEMENTIA-Prominent-Earlier-than-Typical-PD: Dementia is a prominent and early feature — annual cognitive assessment; early care planning and neuropsychological support",
            "ATP13A2-NOT-PRKN-PINK1-MRI-Iron-Is-Key-DDx: Iron on MRI DISTINGUISHES KRS from PRKN/PINK1/DJ-1 where MRI is normal — MRI iron = first test in atypical young AR parkinsonism workup",
        ],
    },
]


def _make_cohort(gene_data: dict) -> list:
    r = random.Random(gene_data["seed"])
    gene = gene_data["gene"]
    etiologies = gene_data["etiologies"]
    pts = []

    for i in range(gene_data["n_patients"]):
        # Draw etiology
        roll = r.random()
        cumul = 0.0
        etiol = etiologies[-1][0]
        for et, prob in etiologies:
            cumul += prob
            if roll < cumul:
                etiol = et
                break

        # Sex distribution — PD generally equal; PRKN/PINK1 slight male excess in some series
        if gene in ("LRRK2", "SNCA", "VPS35"):
            sex = "M" if r.random() < 0.55 else "F"  # slight male excess AD PD
        elif gene in ("PRKN", "PINK1", "DJ1", "ATP13A2"):
            sex = "M" if r.random() < 0.55 else "F"  # slight male excess AR PD
        elif gene == "GBA":
            sex = "M" if r.random() < 0.53 else "F"
        else:
            sex = "M" if r.random() < 0.50 else "F"

        # Onset age (years)
        onset_ranges = {
            "LRRK2": (50, 70),    # typical late onset
            "PRKN": (20, 50),     # early onset
            "PINK1": (18, 50),    # early onset
            "SNCA": (35, 65),     # variable by copy number
            "DJ1": (20, 48),      # early onset
            "GBA": (45, 70),      # moderate to late onset
            "VPS35": (45, 70),    # late onset
            "ATP13A2": (12, 35),  # juvenile/young onset
        }
        lo, hi = onset_ranges[gene]
        onset_y = round(lo + r.random() * (hi - lo), 1)
        onset_m = round(onset_y * 12)
        age_current_y = round(onset_y + r.random() * 15 + 3)
        dx_delay_m = round(r.gauss(28, 20))  # months to diagnosis — PD often delayed
        if dx_delay_m < 1:
            dx_delay_m = 1

        flags = {}

        if gene == "LRRK2":
            flags["gly2019s_variant"] = r.random() < 0.78
            flags["kinase_inhibitor_trial_eligible"] = r.random() < 0.38
            flags["levodopa_response_excellent"] = r.random() < 0.88
            flags["tremor_predominant"] = r.random() < 0.72
            flags["slow_progression"] = r.random() < 0.78
            flags["cascade_tested"] = r.random() < 0.54
            flags["penetrance_counselled"] = r.random() < 0.62
            flags["urine_rab10_tested"] = r.random() < 0.18

        elif gene == "PRKN":
            flags["exon_rearrangement"] = r.random() < 0.60
            flags["mlpa_performed"] = r.random() < 0.58
            flags["levodopa_response"] = r.random() < 0.92
            flags["early_dyskinesias"] = r.random() < 0.72
            flags["tremor_dominant"] = r.random() < 0.68
            flags["exercise_enrolled"] = r.random() < 0.48
            flags["dopamine_agonist_first"] = r.random() < 0.62
            flags["onset_age_under40"] = onset_y < 40

        elif gene == "PINK1":
            flags["levodopa_response"] = r.random() < 0.90
            flags["early_dyskinesias"] = r.random() < 0.68
            flags["psychiatric_comorbidity"] = r.random() < 0.44
            flags["exercise_enrolled"] = r.random() < 0.46
            flags["sleep_benefit"] = r.random() < 0.58
            flags["mitophagy_pathway_confirmed"] = r.random() < 0.72
            flags["mri_normal"] = r.random() < 0.96

        elif gene == "SNCA":
            flags["snca_duplication"] = r.random() < 0.40
            flags["snca_triplication"] = r.random() < 0.25
            flags["point_mutation"] = not (flags["snca_duplication"] or flags["snca_triplication"])
            flags["a53t_variant"] = flags["point_mutation"] and r.random() < 0.57
            flags["dementia_developed"] = flags["snca_triplication"] and r.random() < 0.78
            flags["anti_synuclein_trial_eligible"] = r.random() < 0.42
            flags["mlpa_performed"] = r.random() < 0.54
            flags["copy_number_quantified"] = flags["mlpa_performed"] and r.random() < 0.88

        elif gene == "DJ1":
            flags["l166p_variant"] = r.random() < 0.45
            flags["levodopa_response"] = r.random() < 0.88
            flags["tremor_predominant"] = r.random() < 0.74
            flags["oxidative_stress_testing"] = r.random() < 0.22
            flags["exercise_enrolled"] = r.random() < 0.50
            flags["mri_normal"] = r.random() < 0.97
            flags["very_slow_progression"] = r.random() < 0.80

        elif gene == "GBA":
            flags["n370s_variant"] = r.random() < 0.45
            flags["l444p_variant"] = r.random() < 0.28
            flags["heterozygous_gba"] = r.random() < 0.85
            flags["biallelic_gba"] = not flags["heterozygous_gba"]
            flags["gaucher_diagnosed"] = flags["biallelic_gba"] and r.random() < 0.72
            flags["ambroxol_trial_eligible"] = r.random() < 0.40
            flags["moca_annual"] = r.random() < 0.68
            flags["cognitive_decline"] = r.random() < 0.48
            flags["lysosomal_pathway_confirmed"] = r.random() < 0.56
            flags["levodopa_response"] = r.random() < 0.86

        elif gene == "VPS35":
            flags["d620n_variant"] = r.random() < 0.95
            flags["levodopa_response"] = r.random() < 0.88
            flags["late_onset"] = onset_y >= 50
            flags["retromer_pathway"] = r.random() < 0.78
            flags["cascade_tested"] = r.random() < 0.52
            flags["penetrance_counselled"] = r.random() < 0.60
            flags["similar_sporadic_phenotype"] = r.random() < 0.86

        elif gene == "ATP13A2":
            flags["mri_iron_basal_ganglia"] = r.random() < 0.88
            flags["pyramidal_signs"] = r.random() < 0.82
            flags["supranuclear_gaze_palsy"] = r.random() < 0.62
            flags["levodopa_initial_response"] = r.random() < 0.74
            flags["lysosomal_dysfunction_confirmed"] = r.random() < 0.68
            flags["dementia_prominent"] = r.random() < 0.72
            flags["a746t_variant"] = r.random() < 0.38
            flags["jordanian_founder"] = flags["a746t_variant"]

        pts.append({
            "pid": f"{gene}-{i+1:03d}",
            "gene": gene,
            "etiology": etiol,
            "sex": sex,
            "age_onset_years": onset_y,
            "age_onset_months": onset_m,
            "age_current_years": age_current_y,
            "dx_delay_months": dx_delay_m,
            **flags,
        })
    return pts


def get_overview():
    all_patients = []
    gene_summaries = []

    for gd in PARKINSONS_GENES:
        pts = _make_cohort(gd)
        all_patients.extend(pts)

        gene_summaries.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "seed": gd["seed"],
            "etiologies": [e[0] for e in gd["etiologies"]],
            "key_alerts": gd["key_alerts"],
            "alias": gd["alias"],
            "gene_class": gd["gene_class"],
        })

    n = len(all_patients)

    def g_pts(gene):
        return [p for p in all_patients if p["gene"] == gene]

    def pct(lst, key, val=True):
        if not lst:
            return 0.0
        return round(100 * sum(1 for p in lst if p.get(key) == val) / len(lst), 1)

    lrrk2 = g_pts("LRRK2")
    prkn  = g_pts("PRKN")
    pink1 = g_pts("PINK1")
    snca  = g_pts("SNCA")
    dj1   = g_pts("DJ1")
    gba   = g_pts("GBA")
    vps35 = g_pts("VPS35")
    atp   = g_pts("ATP13A2")

    # Mean dx delay across all
    mean_delay = round(sum(p["dx_delay_months"] for p in all_patients) / n, 1)

    # Cross-gene levodopa response (genes with levodopa_response or levodopa_response_excellent)
    levodopa_responsive_pct = round(100 * sum(
        1 for p in all_patients
        if p.get("levodopa_response") or p.get("levodopa_response_excellent") or p.get("levodopa_initial_response")
    ) / n, 1)

    # MLPA performed (PRKN + SNCA)
    mlpa_genes = prkn + snca
    mlpa_performed_pct = round(100 * sum(1 for p in mlpa_genes if p.get("mlpa_performed")) / len(mlpa_genes), 1)

    # Exercise enrolled (PRKN, PINK1, DJ1)
    exercise_genes = prkn + pink1 + dj1
    exercise_enrolled_pct = round(100 * sum(1 for p in exercise_genes if p.get("exercise_enrolled")) / len(exercise_genes), 1)

    # Trial eligible (kinase inhibitor + anti-synuclein + ambroxol)
    trial_eligible_pct = round(100 * sum(
        1 for p in all_patients
        if p.get("kinase_inhibitor_trial_eligible") or p.get("anti_synuclein_trial_eligible") or p.get("ambroxol_trial_eligible")
    ) / n, 1)

    # MRI abnormal (ATP13A2 iron)
    mri_abnormal_pct = round(100 * sum(1 for p in atp if p.get("mri_iron_basal_ganglia")) / len(atp), 1)

    # Early dyskinesias (PRKN + PINK1)
    early_dys_genes = prkn + pink1
    early_dyskinesias_pct = round(100 * sum(1 for p in early_dys_genes if p.get("early_dyskinesias")) / len(early_dys_genes), 1)

    agg = {
        "total_patients": n,
        "total_genes": 8,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "mean_dx_delay_months": mean_delay,
        # Cross-gene summary stats
        "levodopa_response_pct": levodopa_responsive_pct,
        "mlpa_performed_pct": mlpa_performed_pct,
        "exercise_enrolled_pct": exercise_enrolled_pct,
        "trial_eligible_pct": trial_eligible_pct,
        "mri_iron_abnormal_pct_atp13a2": mri_abnormal_pct,
        "early_dyskinesias_pct_prkn_pink1": early_dyskinesias_pct,
        # LRRK2
        "lrrk2_g2019s_pct": pct(lrrk2, "gly2019s_variant"),
        "lrrk2_kinase_trial_eligible_pct": pct(lrrk2, "kinase_inhibitor_trial_eligible"),
        "lrrk2_levodopa_excellent_pct": pct(lrrk2, "levodopa_response_excellent"),
        "lrrk2_tremor_predominant_pct": pct(lrrk2, "tremor_predominant"),
        "lrrk2_penetrance_counselled_pct": pct(lrrk2, "penetrance_counselled"),
        "lrrk2_cascade_tested_pct": pct(lrrk2, "cascade_tested"),
        # PRKN
        "prkn_exon_rearrangement_pct": pct(prkn, "exon_rearrangement"),
        "prkn_mlpa_performed_pct": pct(prkn, "mlpa_performed"),
        "prkn_levodopa_response_pct": pct(prkn, "levodopa_response"),
        "prkn_early_dyskinesias_pct": pct(prkn, "early_dyskinesias"),
        "prkn_dopamine_agonist_first_pct": pct(prkn, "dopamine_agonist_first"),
        "prkn_exercise_enrolled_pct": pct(prkn, "exercise_enrolled"),
        # PINK1
        "pink1_levodopa_response_pct": pct(pink1, "levodopa_response"),
        "pink1_psychiatric_comorbidity_pct": pct(pink1, "psychiatric_comorbidity"),
        "pink1_exercise_enrolled_pct": pct(pink1, "exercise_enrolled"),
        "pink1_mri_normal_pct": pct(pink1, "mri_normal"),
        "pink1_sleep_benefit_pct": pct(pink1, "sleep_benefit"),
        # SNCA
        "snca_duplication_pct": pct(snca, "snca_duplication"),
        "snca_triplication_pct": pct(snca, "snca_triplication"),
        "snca_dementia_developed_pct": pct(snca, "dementia_developed"),
        "snca_mlpa_performed_pct": pct(snca, "mlpa_performed"),
        "snca_anti_synuclein_trial_pct": pct(snca, "anti_synuclein_trial_eligible"),
        # DJ1
        "dj1_l166p_pct": pct(dj1, "l166p_variant"),
        "dj1_levodopa_response_pct": pct(dj1, "levodopa_response"),
        "dj1_mri_normal_pct": pct(dj1, "mri_normal"),
        "dj1_exercise_enrolled_pct": pct(dj1, "exercise_enrolled"),
        "dj1_slow_progression_pct": pct(dj1, "very_slow_progression"),
        # GBA
        "gba_n370s_pct": pct(gba, "n370s_variant"),
        "gba_l444p_pct": pct(gba, "l444p_variant"),
        "gba_heterozygous_pct": pct(gba, "heterozygous_gba"),
        "gba_ambroxol_trial_pct": pct(gba, "ambroxol_trial_eligible"),
        "gba_moca_annual_pct": pct(gba, "moca_annual"),
        "gba_cognitive_decline_pct": pct(gba, "cognitive_decline"),
        # VPS35
        "vps35_d620n_pct": pct(vps35, "d620n_variant"),
        "vps35_levodopa_response_pct": pct(vps35, "levodopa_response"),
        "vps35_cascade_tested_pct": pct(vps35, "cascade_tested"),
        "vps35_late_onset_pct": pct(vps35, "late_onset"),
        # ATP13A2
        "atp13a2_mri_iron_pct": pct(atp, "mri_iron_basal_ganglia"),
        "atp13a2_pyramidal_signs_pct": pct(atp, "pyramidal_signs"),
        "atp13a2_supranuclear_gaze_pct": pct(atp, "supranuclear_gaze_palsy"),
        "atp13a2_dementia_pct": pct(atp, "dementia_prominent"),
        "atp13a2_a746t_pct": pct(atp, "a746t_variant"),
    }

    all_alerts = []
    for gd in PARKINSONS_GENES:
        all_alerts.extend(gd["key_alerts"])

    return {
        "title": "Hereditary-Parkinson's-Disease-Atlas — Complete 8-Gene Hereditary PD Reference",
        "subtitle": (
            "LRRK2 · PRKN · PINK1 · SNCA · DJ1 · GBA · VPS35 · ATP13A2 — "
            "320 patients (8×40, seeds 1502–1509) — Kinase GOF PD, AR Mitophagy PD, "
            "Synuclein Dosage, GBA Risk Modifier, Retromer PD, Kufor-Rakeb Syndrome"
        ),
        "genes": gene_summaries,
        "aggregate_stats": agg,
        "top_alerts": all_alerts,
    }


def get_breakdown():
    breakdown = []
    for gd in PARKINSONS_GENES:
        pts = _make_cohort(gd)
        sex_dist = {"M": sum(1 for p in pts if p["sex"] == "M"),
                    "F": sum(1 for p in pts if p["sex"] == "F")}
        mean_onset = round(sum(p["age_onset_years"] for p in pts) / len(pts), 1)
        mean_delay = round(sum(p["dx_delay_months"] for p in pts) / len(pts), 1)
        etiol_counts = {}
        for p in pts:
            etiol_counts[p["etiology"]] = etiol_counts.get(p["etiology"], 0) + 1

        breakdown.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "seed": gd["seed"],
            "mean_onset_years": mean_onset,
            "mean_dx_delay_months": mean_delay,
            "sex_distribution": sex_dist,
            "etiology_counts": etiol_counts,
            "key_alerts": gd["key_alerts"],
            "alias": gd["alias"],
            "gene_class": gd["gene_class"],
            "patients": pts,
        })
    return {"breakdown": breakdown}


def get_definitions():
    return {
        "atlas": "Hereditary-Parkinson's-Disease-Atlas — Complete 8-Gene Hereditary PD Reference",
        "genes": [gd["gene"] for gd in PARKINSONS_GENES],
        "clinical_definitions": [
            {
                "term": "LRRK2 Kinase Activation and Incomplete Penetrance",
                "definition": (
                    "Gly2019Ser is located in the DYG motif of the LRRK2 kinase activation loop — "
                    "the same structural position as the DFG motif in canonical kinases. The Gly→Ser "
                    "substitution removes a conformational constraint on the activation loop, "
                    "increasing kinase catalytic efficiency 2-3 fold over wild-type. Rab GTPases "
                    "(Rab8A Thr72, Rab10 Thr73, Rab35 Thr72) are the primary physiological substrates. "
                    "Hyperphosphorylated Rabs accumulate in their inactive GDP-bound form on "
                    "endosomal membranes, impairing vesicular trafficking, ciliogenesis, and "
                    "lysosomal enzyme delivery. Incomplete penetrance is a defining feature of "
                    "LRRK2-PD: at the population level, G2019S penetrance is estimated at 25% by "
                    "age 59, 45% by age 69, and up to 80% by age 79. This age-dependence means "
                    "pre-symptomatic carriers require age-stratified risk counselling — a 35-year-old "
                    "G2019S carrier has a fundamentally different risk profile to a 65-year-old. "
                    "Environmental modifiers are documented: epidemiological studies consistently show "
                    "that cigarette smoking reduces PD risk by ~45% in G2019S carriers (paradoxical "
                    "protective effect — proposed mechanisms include nicotine-induced LRRK2 "
                    "downregulation and monoamine oxidase-B inhibition). Urine pThr73-Rab10 "
                    "(normalised to total Rab10) serves as a non-invasive pharmacodynamic biomarker "
                    "for LRRK2 kinase inhibitor trials — a validated surrogate for brain LRRK2 activity."
                ),
            },
            {
                "term": "PRKN Exon Rearrangements — MLPA is Mandatory, Not Optional",
                "definition": (
                    "Approximately 50% of pathogenic PRKN alleles are copy number variants (CNVs) — "
                    "exon-level deletions or duplications of the PRKN gene on chromosome 6q25-27. "
                    "These structural rearrangements are completely invisible to short-read Sanger "
                    "sequencing and standard next-generation sequencing (WES, targeted gene panels): "
                    "short-read platforms cannot distinguish between one copy and two copies of an "
                    "exon, because relative read-depth normalisation can equalise coverage across "
                    "heterozygous deletions. The diagnostic consequence: a patient with early-onset "
                    "AR-PD whose WES reports 'no PRKN variants found' may have a heterozygous exon "
                    "deletion on one allele (and a point mutation on the other) that WES simply "
                    "cannot see. MLPA (Multiplex Ligation-dependent Probe Amplification) directly "
                    "quantifies copy number at each PRKN exon — it is the gold-standard CNV assay. "
                    "aCGH (array CGH) can also detect large rearrangements. Clinical protocol: "
                    "in any patient with young-onset AR-PD (onset <50 years) where WES does not "
                    "identify biallelic variants in PRKN, MLPA for PRKN must be performed before "
                    "declaring the patient 'PRKN-negative.' A monoallelic rearrangement (single "
                    "heterozygous deletion) is an incomplete finding — a second pathogenic allele "
                    "(point mutation or second CNV on the other chromosome) must be identified for "
                    "a definitive PRKN-PD diagnosis. Rare digenic cases (PRKN + PINK1 heterozygous) "
                    "have been reported and should be considered."
                ),
            },
            {
                "term": "PINK1-Parkin Mitophagy Pathway — The Common Pathway of AR PD",
                "definition": (
                    "PINK1 and Parkin operate as a sequential kinase-E3 ligase cascade to detect "
                    "and eliminate damaged mitochondria. In healthy mitochondria, PINK1 is constitutively "
                    "imported into the inner membrane matrix, cleaved by the PARL protease, and "
                    "retrotranslocated to the cytosol for rapid proteasomal degradation — maintaining "
                    "very low steady-state PINK1 levels. When mitochondrial membrane potential (ΔΨm) "
                    "collapses (due to oxidative damage, respiratory chain failure, or toxins), "
                    "PINK1 import arrests and the kinase accumulates on the outer mitochondrial "
                    "membrane (OMM), where it dimerises and autophosphorylates (pThr257). Active "
                    "PINK1 then phosphorylates ubiquitin at Ser65 (pSer65-Ub), which allosterically "
                    "activates parkin by engaging parkin's Ubl domain (pSer65-Ubl), opening the "
                    "RING2 domain active site. Activated parkin builds K48/K63 polyubiquitin chains "
                    "on OMM proteins (VDAC1, MFN1/2, MIRO1, TOM20), flagging the mitochondrion "
                    "for autophagic capture. Autophagy adapters (NDP52, OPTN, p62) recognise "
                    "polyubiquitin signals and recruit the ULK1-FIP200 autophagy initiation complex "
                    "and phagophore membrane to engulf and degrade the mitochondrion. "
                    "Therapeutic strategy: NAD+ precursors (NMN, NR) boost mitochondrial biogenesis; "
                    "ULK1 activators rescue autophagosome initiation; exercise activates PGC-1α "
                    "and mitochondrial biogenesis — the mechanistic basis for exercise in PRKN/PINK1-PD."
                ),
            },
            {
                "term": "SNCA Gene Dosage Effect — Duplication vs Triplication Phenotype",
                "definition": (
                    "The normal diploid genome has 2 functional SNCA copies. The relationship between "
                    "SNCA copy number and disease severity demonstrates a clear gene dosage effect: "
                    "2 copies (normal): no PD attributable to SNCA. "
                    "3 copies (duplication): mild-moderate PD indistinguishable from late-onset sporadic "
                    "PD; onset typically 5th-6th decade; standard levodopa response; life expectancy "
                    "minimally reduced. "
                    "4 copies (triplication): severe, early-onset (4th-5th decade) PD with prominent "
                    "dementia resembling Dementia with Lewy Bodies (DLB); cognitive decline is an "
                    "early and defining feature (not late as in sporadic PD); autonomic features; "
                    "rapid progression. The mechanistic explanation: each additional SNCA copy "
                    "roughly doubles steady-state synuclein protein, exceeding clearance capacity "
                    "of the ubiquitin-proteasome system and autophagy-lysosome pathway — driving "
                    "oligomer and fibril formation. Anti-synuclein trial eligibility: all SNCA "
                    "multiplication patients are theoretically eligible for prasinezumab/cinpanemab "
                    "trials (which target extracellular synuclein propagation). MLPA or digital PCR "
                    "is mandatory to distinguish duplication from triplication — and to detect either "
                    "when standard sequencing reports 'normal sequence.' "
                    "MoCA annually in all SNCA patients; cognitive decline in triplication warrants "
                    "early neuropsychological support and advance care planning."
                ),
            },
            {
                "term": "GBA as PD Risk Modifier — Heterozygous vs Biallelic Distinction",
                "definition": (
                    "The GBA genotype-phenotype relationship has two clinically distinct scenarios "
                    "that must never be conflated in genetic counselling: "
                    "Biallelic GBA variants (two pathogenic alleles): cause Gaucher disease — "
                    "the lysosomal storage disorder with visceral (spleen, liver), haematological, "
                    "and potentially neurological manifestations. These patients also carry a high "
                    "lifetime risk of developing PD. Management requires both a Gaucher specialist "
                    "(enzyme replacement therapy with imiglucerase, velaglucerase, or substrate "
                    "reduction therapy with eliglustat) AND a movement disorder neurologist. "
                    "Heterozygous GBA variants (one pathogenic allele): do NOT cause Gaucher disease "
                    "— this is the most critical counselling point. Carriers are clinically healthy "
                    "with respect to lysosomal storage. However, they carry a 5-8x lifetime PD risk "
                    "compared to non-carriers. The severity of the GBA allele modifies this risk: "
                    "p.L444P (severe allele) → greater PD risk, earlier onset, more rapid cognitive "
                    "decline than p.N370S (mild allele). p.N370S protects against Gaucher type 3 "
                    "(neurological Gaucher) but increases PD risk — a paradoxical allele-specific "
                    "effect explained by differential residual GCase activity levels in different "
                    "cell types. Ambroxol (pharmacological GCase chaperone): stabilises GCase at "
                    "neutral ER pH → promotes correct lysosomal trafficking → increases lysosomal "
                    "GCase activity → breaks the GCase-synuclein feed-forward loop. AiM-PD trial."
                ),
            },
            {
                "term": "VPS35 Retromer Complex — D620N as the Pathogenic Variant",
                "definition": (
                    "The retromer complex (VPS35-VPS26-VPS29 trimer) retrieves transmembrane cargo "
                    "from sorting endosomes to the trans-Golgi network or plasma membrane, preventing "
                    "lysosomal degradation of recycling receptors and lipids. VPS35 is the scaffold "
                    "subunit, directly interacting with VPS26 (which contacts cargo sorting signals) "
                    "and VPS29 (which recruits regulatory proteins). The retromer also associates "
                    "with the WASH complex (Wiskott-Aldrich syndrome protein and SCAR homolog) "
                    "through FAM21, which tethers WASH to endosomal membranes. WASH nucleates "
                    "branched actin on endosomes, forming the tubular carriers that pinch off "
                    "sorted cargo. p.D620N disrupts the VPS35-FAM21 interaction: Asp620 forms a "
                    "critical electrostatic contact with FAM21's MIM (WASH interaction motif), "
                    "and the Asn substitution eliminates this contact, impairing WASH recruitment "
                    "and actin-driven tubule formation. The net result: cargo destined for "
                    "recycling is instead degraded, lysosomal enzyme receptors (CI-M6PR) are "
                    "mis-sorted, lysosomal hydrolase delivery is impaired, and autophagic flux "
                    "is reduced. D620N accounts for ~95% of all VPS35-PD alleles identified "
                    "worldwide — targeted single-variant testing is sufficient when VPS35-PD is "
                    "clinically suspected. Retromer chaperone compounds (R33, R55) stabilise the "
                    "VPS35-VPS26 interface in cell models — a rationale for therapeutic development."
                ),
            },
            {
                "term": "ATP13A2 Kufor-Rakeb Syndrome — Iron on MRI is the DDx Key",
                "definition": (
                    "Kufor-Rakeb Syndrome (KRS) is an atypical parkinsonism caused by biallelic "
                    "ATP13A2 loss-of-function. The clinical tetrad — juvenile/young-onset parkinsonism, "
                    "pyramidal tract signs, supranuclear upgaze palsy, and cognitive decline/dementia — "
                    "distinguishes it clearly from typical PRKN/PINK1/DJ-1 early-onset PD. "
                    "The single most important distinguishing investigation is brain MRI: "
                    "in KRS, T2-weighted and T2*-weighted (gradient echo or SWI) sequences show "
                    "hypointensity (signal dropout) in the putamen and caudate nucleus — "
                    "reflecting paramagnetic non-haem iron accumulation. This neurodegeneration "
                    "with brain iron accumulation (NBIA)-like pattern is ABSENT in all other "
                    "monogenic PD genes (PRKN, PINK1, DJ-1, SNCA, LRRK2, VPS35) where MRI is "
                    "structurally normal. The iron accumulates because ATP13A2 normally exports "
                    "iron from lysosomes following autophagic degradation of iron-containing "
                    "proteins (ferritin, haemoglobin): ATP13A2 loss traps iron in lysosomes, "
                    "which then rupture, releasing iron into the cytosol where it generates "
                    "hydroxyl radicals via Fenton chemistry, causing oxidative cell death. "
                    "Levodopa provides initial motor benefit but its effect diminishes as pyramidal "
                    "and cognitive features (which are non-dopaminergic) become dominant — "
                    "a pattern of 'initial levodopa response that wanes' is characteristic and "
                    "should prompt ATP13A2 testing in any young-onset atypical parkinsonism patient."
                ),
            },
            {
                "term": "Exercise as Disease-Modifying Therapy in Genetic PD",
                "definition": (
                    "High-intensity aerobic exercise has accumulated substantial evidence as a "
                    "disease-modifying intervention in PD, with particularly strong mechanistic "
                    "rationale in PRKN, PINK1, and DJ-1 forms. The PARK trial (Schenkman 2018, "
                    "JAMA Neurology) demonstrated that high-intensity treadmill exercise (80-85% "
                    "maximal heart rate, 3 sessions/week) significantly slowed motor progression "
                    "over 6 months in early PD (MDS-UPDRS III), with superior effects to moderate "
                    "intensity. Mechanistic pathways: (1) PGC-1α activation → mitochondrial "
                    "biogenesis (directly compensates for PINK1/Parkin mitophagy deficit); "
                    "(2) BDNF upregulation → dopaminergic neurotrophic support → synaptic "
                    "plasticity preservation; (3) dopamine turnover enhancement → improved "
                    "dopaminergic transmission efficiency; (4) antioxidant enzyme induction "
                    "(SOD2, GPx, HO-1) → compensates for DJ-1 oxidative stress sensor loss; "
                    "(5) autophagy induction → clears protein aggregates in exercise-activated "
                    "neurons. The prescription: aerobic exercise at 70-85% maximum heart rate, "
                    "minimum 150 minutes per week (ideally 3× 50-minute sessions); physiotherapy-"
                    "supervised initially; Nordic walking and cycling are well-tolerated modalities. "
                    "Resistance training adds benefit for freezing of gait and postural stability. "
                    "The PACT-PD and PD-SAFE trials further support exercise across disease stages."
                ),
            },
            {
                "term": "Cascade Testing Strategy in Hereditary Parkinson's Disease",
                "definition": (
                    "Cascade testing strategy by inheritance pattern and gene: "
                    "Autosomal dominant (LRRK2, SNCA, VPS35): first-degree relatives have 50% "
                    "a priori risk of inheriting the variant. Offer predictive genetic testing "
                    "after comprehensive pre-test genetic counselling. LRRK2: explain incomplete "
                    "penetrance (25-80% by age 80) and age-dependence before testing; a positive "
                    "result is NOT a diagnosis of PD. SNCA: distinguish duplication vs triplication "
                    "by MLPA before cascade testing begins — this determines the severity message "
                    "conveyed to relatives. "
                    "Autosomal recessive (PRKN, PINK1, DJ-1, ATP13A2): parents are obligate "
                    "heterozygous carriers (healthy); siblings have 25% risk of biallelic variants "
                    "and 50% risk of being healthy carriers. Carrier testing is offered to siblings "
                    "for reproductive decision-making. Cascade testing in AR-PD families is most "
                    "impactful for prenatal/preimplantation genetic testing options. "
                    "GBA: heterozygous GBA in a PD patient → offer carrier testing to first-degree "
                    "relatives; relatives who test positive should receive penetrance counselling "
                    "(5-8x PD risk, not a diagnosis) and regular neurological monitoring from age 50. "
                    "Pre-symptomatic testing age threshold: typically not offered to minors (<18) "
                    "for adult-onset conditions without clear preventive benefit; exceptions for "
                    "ATP13A2 (juvenile onset) and PRKN/PINK1 (occasional teenage onset). "
                    "Psychological support (genetic counsellor or psychologist) should be available "
                    "at result disclosure for all pre-symptomatic testing."
                ),
            },
            {
                "term": "Anti-Synuclein and Disease-Modifying Trials — Matching Genotype to Trial",
                "definition": (
                    "The genetic PD trial landscape has reached an inflection point where genotype "
                    "determines trial eligibility — incorrect genotype-trial matching wastes trial "
                    "slots and exposes patients to inappropriate interventions. "
                    "LRRK2 kinase inhibitors (DNL201, DNL151 — Denali Therapeutics): indicate "
                    "confirmed LRRK2-PD (G2019S or other kinase-domain pathogenic variant); "
                    "pharmacodynamic monitoring uses urine pRab10; lung and kidney safety monitoring "
                    "required (LRRK2 highly expressed in these tissues). "
                    "Anti-synuclein immunotherapy (prasinezumab — Roche/Prothena; cinpanemab — Biogen): "
                    "target extracellular/aggregated synuclein; applicable to all genetic PD forms "
                    "with confirmed synuclein pathology (SNCA multiplication, GBA-PD, and potentially "
                    "LRRK2-PD); genotyped patients with biomarker-confirmed synuclein elevation "
                    "(CSF alpha-synuclein seed amplification assay positive) are priority for enrolment. "
                    "Ambroxol GCase chaperone (AiM-PD trial — 1260 mg/day): specifically for "
                    "GBA-PD (heterozygous or biallelic); GBA genotype must be confirmed and GCase "
                    "activity assay performed at baseline (primary pharmacodynamic endpoint is "
                    "GCase activity increase in CSF). "
                    "NAD+ precursor trials (NMN, NR): mechanistically relevant for PINK1/PRKN-PD "
                    "(mitochondrial biogenesis pathway); PINK1-confirmed patients are priority. "
                    "General principle: confirm genotype, confirm variant pathogenicity, assess "
                    "relevant biomarkers (urine pRab10, CSF synuclein SAA, GCase activity) "
                    "before enrolling — never assume trial eligibility without documented genotype."
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print(json.dumps(ov["aggregate_stats"], indent=2))
    print(f"\nTop alerts ({len(ov['top_alerts'])}):")
    for a in ov["top_alerts"]:
        print(f"  • {a}")
