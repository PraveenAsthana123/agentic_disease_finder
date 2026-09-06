#!/usr/bin/env python3
"""Hereditary-Epilepsy-Atlas — Complete 8-Gene Hereditary Monogenic Epilepsy Atlas
SCN1A  (Nav1.1; 1998 aa; 2q24.3; AD;
         Dravet syndrome — most common severe monogenic epilepsy; haploinsufficiency →
         inhibitory interneuron (PV+ and SST+) dysfunction → runaway excitation;
         AVOID carbamazepine/lamotrigine/phenytoin (worsen by blocking residual Nav1.1);
         valproate + clobazam first-line; stiripentol (adj), fenfluramine, cannabidiol FDA-approved;
         temperature-sensitivity trigger; SUDEP risk highest of all genetic epilepsies;
         genotype-phenotype: missense > truncating for severity at C-terminus) ·
SCN2A  (Nav1.2; 2005 aa; 2q24.3; AD;
         Gain-of-function early onset <3 months → Na channel blocker RESPONSIVE (carbamazepine, phenytoin);
         Loss-of-function late onset >3 months → Na channel blocker HARMFUL;
         Age-of-onset PREDICTS GOF vs LOF — critical pharmacogenomic decision;
         GOF: neonatal seizures, often self-limiting BFNIS or evolving DEE;
         LOF: autism spectrum/DEE; phenotype-driven precision therapy) ·
KCNQ2  (Kv7.2; 872 aa; 20q13.33; AD;
         BFNE (Benign Familial Neonatal Epilepsy) — self-limiting by 4-6 months OR KCNQ2-DEE severe neonatal;
         M-current (IKM) controller; KCNQ2-DEE GOF: carbamazepine/phenytoin HIGHLY EFFECTIVE;
         Ezogabine/retigabine (Kv7 opener) — withdrawn from market; flupirtine analogue;
         EEG: burst-suppression neonatal → improves with age; SNHL screen all patients;
         LOF mild BFNE vs GOF severe DEE — genotype predicts treatment response) ·
CDKL5  (CDKL5 Deficiency Disorder; 1030 aa; Xp22.13; XLD;
         Seizure onset <5 months (median 3 months) — refractory multifocal and spasms;
         Rett-like features (hand stereotypies, absent speech) but DISTINCT from MECP2-Rett;
         No specific ASM highly effective; ketogenic diet best non-pharmacological evidence;
         Ganaxolone (CDKL5 Foundation-funded) FDA-approved 2022 — FIRST specific therapy;
         X-linked dominant: affected females (80%), males (20%) more severe;
         mTOR pathway research ongoing; annual ophthalmology (cortical visual impairment 60%)) ·
PCDH19  (Protocadherin 19; 1148 aa; Xq22.1; XLD;
          PCDH19 Epilepsy / EFMR (Epilepsy and Intellectual Disability limited to Females);
          CELLULAR INTERFERENCE MODEL: affected females (heterozygous), UNAFFECTED carrier males (hemizygous);
          Mosaic males CAN be affected — critical DDx;
          Clustering febrile-associated seizures, prominent emotional/behavioral component;
          Clobazam for clustering; stiripentol consideration; AVOID levetiracetam (worsens behavior);
          Remission possible post-puberty in ~50%; psychiatric comorbidity high) ·
SCN8A  (Nav1.6; 1980 aa; 12q13.13; AD;
         SCN8A-DEE (EIEE13); GOF → persistent Na current → burst firing → early severe DEE;
         Phenytoin/carbamazepine HIGHLY EFFECTIVE for GOF — OPPOSITE of SCN1A;
         De novo GOF dominant: p.Asn1768Asp most recurrent pathogenic hotspot;
         Quinidine reported but less established than Na channel blockers;
         Loss-of-function: benign tremor/movement disorder — AVOID Na channel blockers (worsen);
         MRI: often normal early; cerebellar atrophy late; mortality rate ~25% by age 15) ·
KCNT1  (Slack/KCa4.1; 1257 aa; 9q34.3; AD/AR;
         EIMFS (Epilepsy of Infancy with Migrating Focal Seizures) — most severe spectrum;
         Also ADNFLE (Autosomal Dominant Nocturnal Frontal Lobe Epilepsy);
         GOF: excess Na-activated K+ (KNa1.1) current → depolarisation block → autonomic dysregulation;
         Quinidine blocks KCNT1 in vitro — clinical trials ongoing (KCNT1 quinidine response variable);
         Quinidine trial warranted for GOF KCNT1 (case-by-case, toxicity monitoring required);
         Refractory to standard ASM; ketogenic diet partial responders) ·
SLC6A1  (GAT-1 GABA Transporter; 797 aa; 3p25.3; AD;
          MAE (Myoclonic-Atonic Epilepsy / Doose syndrome) — most common SLC6A1 phenotype;
          GAT-1 haploinsufficiency → impaired synaptic GABA reuptake paradoxically → receptor downregulation;
          Valproate first-line; ketogenic diet highly effective (>50% responders);
          AVOID carbamazepine/lamotrigine/vigabatrin (worsen myoclonic-atonic);
          Also: MAE + intellectual disability + autism spectrum; ataxia during seizure cluster;
          DROP ATTACKS major injury risk — helmet mandatory)
320-patient aggregate cohort (8 × 40, seeds 1486–1493)
"""

import random

SEED_BASE = 1486

EPILEPSY_GENES = [
    # ── SCN1A — Dravet Syndrome / GEFS+ ──
    {
        "gene": "SCN1A",
        "protein": "Voltage-Gated Sodium Channel Nav1.1 — Inhibitory Interneuron Guardian",
        "alias": (
            "SCN1A; OMIM gene 182389; Dravet syndrome OMIM 607208; GEFS+ OMIM 604233; 2q24.3; 1998 aa; ~228 kDa; "
            "SCN1A encodes Nav1.1, the principal voltage-gated sodium channel of GABAergic fast-spiking "
            "parvalbumin-positive (PV+) and somatostatin-positive (SST+) inhibitory interneurons. "
            "Haploinsufficiency reduces Nav1.1 density selectively in inhibitory interneurons, "
            "impairing their high-frequency firing capacity while excitatory pyramidal neurons "
            "(which express Nav1.2/Nav1.6) remain unaffected — net result is runaway excitation. "
            "Dravet syndrome: heterozygous loss-of-function (truncations, missense C-terminus, "
            "splice-site) — onset 4–8 months with prolonged febrile hemiclonic seizure; "
            "temperature-sensitive (hot bath, fever → status epilepticus); polyphenotypic by age 1–2y. "
            "AVOID carbamazepine, lamotrigine, phenytoin — block residual Nav1.1, worsen disease. "
            "FDA-approved specific treatments: stiripentol (2018), cannabidiol (2018), fenfluramine (2020). "
            "SUDEP: highest all-cause mortality of any genetic epilepsy (10–20% lifetime risk). "
            "GEFS+ spectrum: milder missense variants → febrile seizures plus or SMEI. "
            "SCN1A testing first-line in any infant with prolonged febrile hemiclonic seizure."
        ),
        "aa": "1998 aa",
        "kDa": "~228 kDa",
        "locus": "2q24.3",
        "omim_gene": 182389,
        "omim_disease": 607208,
        "inheritance": "AD — haploinsufficiency (truncations/missense/splice) de novo ~90%",
        "gene_class": (
            "SCN1A encodes Nav1.1, the alpha-subunit of a tetrodotoxin-sensitive voltage-gated sodium "
            "channel critical for action potential generation in fast-spiking GABAergic interneurons. "
            "The protein spans four domains (DI–DIV), each containing six transmembrane segments. "
            "The S4 voltage sensor and DIV linker region are mutational hotspots. "
            "Dravet syndrome is caused by haploinsufficiency: truncating or functionally null missense "
            "variants reduce Nav1.1 protein below the threshold needed for sustained high-frequency "
            "interneuron firing. Excitatory neurons compensate via upregulation of Nav1.2/1.6, "
            "creating a profound excitation-inhibition imbalance. "
            "The precision pharmacotherapy imperative is absolute: sodium channel blockers that block "
            "all Nav isoforms (carbamazepine, lamotrigine, phenytoin) further suppress the already-depleted "
            "Nav1.1 in inhibitory interneurons, worsening seizure frequency, duration, and SUDEP risk. "
            "This is one of the few epilepsy precision medicine axioms with Level A evidence."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("Dravet — truncating variant (nonsense/frameshift/splice) — haploinsufficiency", 0.52),
            ("Dravet — missense C-terminus domain DIV — functionally null", 0.22),
            ("GEFS+ spectrum — missense DI-DIII — partial loss of function", 0.14),
            ("SCN1A large deletion/duplication (MLPA)", 0.07),
            ("SCN1A VUS / mosaicism", 0.05),
        ],
        "key_alerts": [
            "SCN1A-DRAVET: AVOID carbamazepine/lamotrigine/phenytoin — block Nav1.1 → WORSEN seizures + SUDEP risk",
            "SCN1A-SUDEP: Highest SUDEP risk of any genetic epilepsy — 10-20% lifetime; nighttime supervision",
            "SCN1A-TEMPERATURE: Fever/hot bath → status epilepticus risk; FEVER PROTOCOL: treat early, tepid sponge",
            "SCN1A-FENFLURAMINE: FDA-approved 2020 — serotonin-mediated; cardiac monitoring mandatory (echocardiogram annually)",
            "SCN1A-CANNABIDIOL: FDA-approved 2018 (Epidiolex) — CYP3A4/2C19 interaction with clobazam (↑ active metabolite)",
            "SCN1A-STIRIPENTOL: FDA-approved 2018 adjunct with valproate + clobazam — inhibits CYP2C19 (raises clobazam levels)",
            "SCN1A-STATUS: Buccal midazolam / rectal diazepam rescue mandatory for all families — >5 min seizure = emergency",
        ],
    },
    # ── SCN2A — Neonatal/Early-Onset Epilepsy GOF vs LOF ──
    {
        "gene": "SCN2A",
        "protein": "Voltage-Gated Sodium Channel Nav1.2 — Age-of-Onset Pharmacogenomic Switch",
        "alias": (
            "SCN2A; OMIM gene 182390; BFNIS OMIM 607745; SCN2A-DEE OMIM 613721; 2q24.3; 2005 aa; ~228 kDa; "
            "SCN2A encodes Nav1.2, highly expressed in excitatory pyramidal neurons and axon initial segments. "
            "Gain-of-function variants (GOF): persistent Na current → sustained depolarisation → "
            "early-onset seizures (<3 months) — typically neonatal; carbamazepine/phenytoin HIGHLY EFFECTIVE "
            "(block persistent Nav1.2 current). Loss-of-function variants (LOF): late onset (>3 months) → "
            "autism spectrum disorder, intellectual disability, LOF-DEE — Na channel blockers CONTRAINDICATED "
            "(further reduce Nav1.2 → worse). Age-of-onset is the critical clinical switch: "
            "<3 months → GOF phenotype → Na channel blocker → high response; "
            ">3 months → LOF phenotype → avoid Na channel blockers. "
            "Both self-limiting BFNIS (de novo GOF mild) and severe DEE (GOF strong or LOF) occur. "
            "Electroclinical phenotyping by onset age guides precision ASM selection."
        ),
        "aa": "2005 aa",
        "kDa": "~228 kDa",
        "locus": "2q24.3",
        "omim_gene": 182390,
        "omim_disease": 613721,
        "inheritance": "AD — de novo GOF (early onset) or LOF (late onset); some AR LOF severe DEE",
        "gene_class": (
            "SCN2A encodes Nav1.2, structurally homologous to Nav1.1 but predominantly expressed in "
            "excitatory pyramidal neurons and concentrated at the axon initial segment (AIS). "
            "During postnatal development, Nav1.2 is gradually replaced by Nav1.6 at the AIS, "
            "explaining why SCN2A GOF seizures often attenuate with age. "
            "The pharmacogenomic duality — GOF responds to, LOF is worsened by, sodium channel blockers — "
            "makes SCN2A unique among DEE genes. Functional characterisation of variants is critical: "
            "patch-clamp data showing persistent inward current → GOF → carbamazepine trial; "
            "haploinsufficiency/dominant-negative → LOF → avoid CBZ, consider Na channel-sparing ASMs. "
            "This age-dependent therapeutic decision separates SCN2A management from all other Nav epilepsies."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("GOF — de novo missense (persistent inward current) — early neonatal onset", 0.38),
            ("GOF — BFNIS self-limiting — mild persistent current — family history", 0.15),
            ("LOF — de novo truncating — late-onset DEE/autism spectrum", 0.28),
            ("LOF — de novo missense (haploinsufficiency) — ASD/ID without prominent epilepsy", 0.12),
            ("SCN2A VUS — functional testing pending", 0.07),
        ],
        "key_alerts": [
            "SCN2A-AGE-RULE: Onset <3 months → GOF → carbamazepine/phenytoin HIGHLY EFFECTIVE; onset >3 months → LOF → AVOID Na channel blockers",
            "SCN2A-GOF-CBZ: Carbamazepine first-line for GOF early-onset SCN2A — dramatic response expected in 70%",
            "SCN2A-LOF-AVOID-CBZ: LOF variants → carbamazepine/lamotrigine/phenytoin WORSEN seizures and development",
            "SCN2A-FUNCTIONAL: Always request functional characterisation for missense VUS — predicts GOF vs LOF",
            "SCN2A-ASD: LOF SCN2A is a major ASD-DEE gene — developmental regression and autism features",
        ],
    },
    # ── KCNQ2 — Neonatal Epilepsy / KCNQ2-DEE ──
    {
        "gene": "KCNQ2",
        "protein": "Voltage-Gated Potassium Channel Kv7.2 — M-Current Neonatal Seizure Controller",
        "alias": (
            "KCNQ2; OMIM gene 602235; BFNE OMIM 121200; KCNQ2-DEE OMIM 613720; 20q13.33; 872 aa; ~97 kDa; "
            "KCNQ2 encodes Kv7.2, the principal subunit of the M-current (IKM) — a slow non-inactivating K+ "
            "current that limits repetitive neuronal firing and modulates resting membrane potential. "
            "M-current is highest at AIS and nodes of Ranvier; loss reduces firing threshold globally. "
            "Benign Familial Neonatal Epilepsy (BFNE): heterozygous haploinsufficiency — self-limiting "
            "within 4–6 months; excellent neurodevelopment. KCNQ2-DEE: de novo GOF or dominant-negative "
            "LOF at critical positions → severe neonatal encephalopathy; EEG: burst-suppression; "
            "carbamazepine/phenytoin HIGHLY EFFECTIVE early — window of opportunity in neonatal period. "
            "Ezogabine (Kv7 opener) — withdrawn 2017 due to retinal pigmentation; no replacement. "
            "Key neonatal management: early CBZ/phenytoin → improves EEG and outcome. "
            "SNHL (sensorineural hearing loss) screen all KCNQ2 patients — cochlear Kv7.2 expression."
        ),
        "aa": "872 aa",
        "kDa": "~97 kDa",
        "locus": "20q13.33",
        "omim_gene": 602235,
        "omim_disease": 613720,
        "inheritance": "AD — BFNE (haploinsufficiency, family); DEE (de novo GOF/strong-LOF)",
        "gene_class": (
            "KCNQ2 encodes Kv7.2, the primary pore-forming subunit of the neuronal M-type potassium channel. "
            "The M-current (named for its muscarinic inhibition) stabilises the neuronal membrane, "
            "prevents repetitive firing, and is critical during the neonatal period when other K+ channels "
            "have not yet matured. Kv7.2 assembles with Kv7.3 (KCNQ3) as a heteromeric complex. "
            "KCNQ2-DEE mutations typically cluster at the voltage sensor domain, pore loop (G-Y-G signature), "
            "or the calmodulin-binding helix A/B — positions critical for channel gating or PIP2 sensitivity. "
            "The critical clinical insight is the narrow treatment window: carbamazepine/phenytoin given "
            "in the neonatal period (first weeks) dramatically suppress the burst-suppression pattern "
            "and may improve long-term neurodevelopmental outcome — delay in diagnosis costs developmental time."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("KCNQ2-DEE — de novo GOF/dominant-negative at voltage sensor/pore — severe neonatal EE", 0.48),
            ("BFNE — familial haploinsufficiency — self-limiting <6m", 0.30),
            ("KCNQ2-DEE — de novo truncating with dominant-negative effect", 0.12),
            ("KCNQ2 large deletion (MLPA)", 0.06),
            ("KCNQ2 VUS — benign or pathogenic uncertain", 0.04),
        ],
        "key_alerts": [
            "KCNQ2-DEE-WINDOW: Carbamazepine/phenytoin in NEONATAL period — window of opportunity → treat early, improves EEG + outcome",
            "KCNQ2-EEG: Burst-suppression neonatal EEG + clinical seizures → KCNQ2-DEE until proven otherwise → immediate CBZ trial",
            "KCNQ2-SNHL: Sensorineural hearing loss screen ALL patients — KCNQ2 expressed in cochlear hair cells",
            "KCNQ2-EZOGABINE: Kv7 opener ezogabine/retigabine WITHDRAWN (retinal pigmentation) — no replacement; CBZ/PHT preferred",
            "KCNQ2-BFNE: Familial BFNE — excellent prognosis, self-limiting; but offspring risk 50% — prenatal counselling",
        ],
    },
    # ── CDKL5 — CDKL5 Deficiency Disorder ──
    {
        "gene": "CDKL5",
        "protein": "Cyclin-Dependent Kinase-Like 5 — Synaptic Plasticity Kinase",
        "alias": (
            "CDKL5; OMIM gene 300203; CDKL5 Deficiency Disorder OMIM 300672; Xp22.13; 1030 aa; ~115 kDa; "
            "CDKL5 encodes a serine-threonine kinase with an N-terminal kinase domain and C-terminal regulatory "
            "tail. CDKL5 phosphorylates MEF2 (muscle enhancer factor 2) and MeCP2 (methyl-CpG binding protein 2), "
            "regulating synaptic plasticity, dendritic spine morphology, and mTOR pathway activity. "
            "CDD presents at 3–5 months (range 2 weeks to 5 months) with refractory multifocal seizures, "
            "then evolving to infantile spasms in many; severe hypotonia; Rett-like features by 12–18 months "
            "(hand stereotypies, absent/lost purposeful hand use, reduced eye contact); "
            "DISTINCT from MECP2-Rett: CDD has earlier seizure onset, no regression phase. "
            "Ganaxolone (CMP 10) FDA-approved Feb 2022 — first specific therapy for CDD; "
            "a neurosteroid that positively modulates GABA-A receptors (delta subunit extrasynaptic). "
            "Ketogenic diet: 50% >50% seizure reduction in controlled series. "
            "Annual ophthalmology: cortical visual impairment in 60%. X-linked dominant: "
            "males (hemizygous) typically more severely affected than females (heterozygous)."
        ),
        "aa": "1030 aa",
        "kDa": "~115 kDa",
        "locus": "Xp22.13",
        "omim_gene": 300203,
        "omim_disease": 300672,
        "inheritance": "XLD — de novo in 90% females; hemizygous males severely affected",
        "gene_class": (
            "CDKL5 belongs to the CMGC family of serine-threonine kinases. Its bipartite nuclear localisation "
            "signal allows nuclear-cytoplasmic shuttling. The N-terminal kinase domain activates via "
            "autophosphorylation and phosphorylates substrates including MeCP2 (Ser80), MAP1S, and APC. "
            "In neurons, CDKL5 concentrates at excitatory synapses and is required for activity-dependent "
            "actin remodelling in dendritic spines. Loss disrupts synaptic plasticity, reduces dendritic "
            "complexity, and impairs mTOR signalling — explaining the rational basis for mTOR-targeted trials. "
            "The disease mechanism is loss of function: both frameshift/truncating and missense variants "
            "in the kinase domain abolish catalytic activity. Phenotypic variability in females reflects "
            "X-inactivation skewing — heavily skewed X-inactivation (favouring mutant allele) predicts severe phenotype."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("CDD — truncating variant (frameshift/nonsense) kinase domain — severe", 0.44),
            ("CDD — missense kinase domain (catalytic residue) — loss of function", 0.30),
            ("CDD — splice-site variant — exon skipping kinase domain", 0.12),
            ("CDD — large deletion Xp22.13 (MLPA)", 0.09),
            ("CDD — male hemizygous truncating — severe neonatal", 0.05),
        ],
        "key_alerts": [
            "CDKL5-GANAXOLONE: FDA-approved Feb 2022 (Zafgen/Marinus) — first specific CDD therapy; neurosteroid GABA-A delta-modulator",
            "CDKL5-KD: Ketogenic diet — 50% achieve >50% seizure reduction; trial early if ASM refractory",
            "CDKL5-NOT-RETT: DISTINCT from MECP2-Rett — earlier seizure onset <5m, no regression-plateau; separate genetic test",
            "CDKL5-OPHTHO: Cortical visual impairment 60% — annual ophthalmology from diagnosis; visual training programmes",
            "CDKL5-MALES: Hemizygous males more severe — earlier diagnosis; consider CDKL5 in severe neonatal epilepsy males",
            "CDKL5-MTOR: mTOR pathway dysregulated — mTORC1 inhibitors in clinical trials; rapamycin analogue research",
        ],
    },
    # ── PCDH19 — Female-Restricted Epilepsy ──
    {
        "gene": "PCDH19",
        "protein": "Protocadherin 19 — X-linked Cellular Interference Epilepsy Mediator",
        "alias": (
            "PCDH19; OMIM gene 300460; PCDH19 Epilepsy / EFMR OMIM 300088; Xq22.1; 1148 aa; ~126 kDa; "
            "PCDH19 encodes protocadherin 19, a cell-adhesion molecule of the delta-2 protocadherin subfamily "
            "expressed in cortical neurons and hippocampus. Unique inheritance: heterozygous females AFFECTED, "
            "hemizygous males UNAFFECTED carriers — explained by the CELLULAR INTERFERENCE MODEL: "
            "mixtures of PCDH19-positive and PCDH19-negative cells in female brain (random X-inactivation) "
            "create abnormal cell-sorting and signalling at tissue interfaces; uniform-null males lack this mismatch. "
            "MOSAIC MALES: somatic mosaicism → mixture of PCDH19+ and PCDH19-null cells → AFFECTED; "
            "critical diagnostic trap — sequencing of blood may miss somatic mosaicism in males; "
            "skin biopsy + multiple tissues recommended for suspected mosaic male. "
            "Clustering febrile-associated seizures; prominent behavioral/emotional dysregulation; "
            "autism spectrum; sleep disorder. Clobazam for clustering; stiripentol consideration. "
            "AVOID levetiracetam: worsens behavioral/psychiatric comorbidity. Remission possible post-puberty ~50%."
        ),
        "aa": "1148 aa",
        "kDa": "~126 kDa",
        "locus": "Xq22.1",
        "omim_gene": 300460,
        "omim_disease": 300088,
        "inheritance": "XLD — cellular interference; affected females; unaffected carrier males (except mosaics)",
        "gene_class": (
            "PCDH19 encodes a single-pass transmembrane cadherin with an extracellular domain containing "
            "six cadherin repeats (EC1-EC6), a transmembrane domain, and a cytoplasmic tail. "
            "It mediates homophilic cell–cell adhesion in cortical neuron migration, hippocampal lamination, "
            "and inhibitory synapse formation. The cellular interference mechanism was proposed by Bhatt et al. (2014): "
            "in heterozygous females, cells expressing normal and mutant PCDH19 cannot sort normally, "
            "creating patch-like mosaicism that disrupts cortical interneuron positioning, "
            "ultimately impairing inhibitory network function. The clustering pattern of seizures "
            "(multiple seizures over 1–5 days separated by seizure-free weeks) mirrors the episodic "
            "dysregulation of these inhibitory networks. PCDH19 epilepsy exemplifies the principle that "
            "X-linked genes can cause dominant conditions in females through non-standard mechanisms."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("PCDH19 — missense EC1-EC6 cadherin domain — heterozygous female — affected", 0.45),
            ("PCDH19 — truncating (nonsense/frameshift) — heterozygous female", 0.28),
            ("PCDH19 — splice-site — exon skipping", 0.12),
            ("PCDH19 — somatic mosaic male — blood/skin confirmed", 0.08),
            ("PCDH19 — large deletion (MLPA)", 0.07),
        ],
        "key_alerts": [
            "PCDH19-FEMALE-ONLY: Heterozygous females AFFECTED; hemizygous males UNAFFECTED — unique X-linked dominant cellular interference",
            "PCDH19-MOSAIC-MALE: Somatic mosaic males CAN be affected — test skin biopsy + multiple tissues; blood sequencing alone insufficient",
            "PCDH19-AVOID-LEV: Levetiracetam WORSENS behavioral/psychiatric comorbidity in PCDH19 — avoid or substitute",
            "PCDH19-CLOBAZAM: Clobazam for clustering protocol — high-dose during cluster phase; taper between clusters",
            "PCDH19-CLUSTERING: Febrile/illness-triggered seizure clusters (multiple/day × 3–5 days) — rescue protocol mandatory for families",
            "PCDH19-PUBERTY: Remission possible in ~50% post-puberty — re-evaluate ASM need in adolescence",
        ],
    },
    # ── SCN8A — Early Severe DEE ──
    {
        "gene": "SCN8A",
        "protein": "Voltage-Gated Sodium Channel Nav1.6 — GOF Excitatory Burst Driver",
        "alias": (
            "SCN8A; OMIM gene 600702; SCN8A-DEE (EIEE13) OMIM 614558; 12q13.13; 1980 aa; ~222 kDa; "
            "SCN8A encodes Nav1.6, the dominant sodium channel isoform at axon initial segments (AIS) and "
            "nodes of Ranvier in both excitatory and inhibitory neurons. Nav1.6 is critical for sustained "
            "high-frequency axonal action potential propagation. GOF mutations → persistent sodium current → "
            "sustained depolarisation → increased burst firing in excitatory circuits → severe early DEE. "
            "Phenytoin and carbamazepine preferentially target persistent Na current — HIGHLY EFFECTIVE in "
            "SCN8A GOF (opposite pharmacology to SCN1A). Most common recurrent pathogenic hotspot: "
            "p.Asn1768Asp (within DIV S4-S5 linker — major determinant of fast inactivation). "
            "Loss-of-function SCN8A: benign tremor/movement disorder — Na channel blockers CONTRAINDICATED. "
            "Mortality: ~25% of SCN8A-DEE patients die by age 15 (SUDEP + status). "
            "MRI: normal early; cerebellar atrophy and cortical thinning on serial imaging."
        ),
        "aa": "1980 aa",
        "kDa": "~222 kDa",
        "locus": "12q13.13",
        "omim_gene": 600702,
        "omim_disease": 614558,
        "inheritance": "AD — de novo GOF dominant (~95%); rare AR LOF severe DEE",
        "gene_class": (
            "SCN8A encodes Nav1.6, expressed at the highest density of any Nav isoform at axon initial segments "
            "and nodes of Ranvier throughout the central and peripheral nervous systems. Unlike Nav1.1 "
            "(inhibitory interneurons) and Nav1.2 (developing excitatory neurons), Nav1.6 is expressed in both "
            "cell types but critically mediates resurgent sodium current — a distinctive non-inactivating "
            "current that enables high-frequency repetitive firing. GOF mutations that impair fast inactivation "
            "or increase persistent current exploit this property to drive pathological burst firing. "
            "The therapeutic implication is the converse of SCN1A: because Nav1.6 is primarily excitatory-network "
            "expressed, sodium channel blockers dampen pathological excitatory activity without impairing "
            "the (already dysfunctional) inhibitory system. This precise pharmacogenomic understanding "
            "transforms treatment outcomes for SCN8A-DEE from refractory to controlled when CBZ/PHT is applied early."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("SCN8A-DEE — de novo GOF missense — DIV S4-S5 linker (p.Asn1768Asp hotspot)", 0.35),
            ("SCN8A-DEE — de novo GOF missense — DI-DIII S4-S5 or pore regions", 0.30),
            ("SCN8A-DEE — de novo truncating with GOF dominant-negative component", 0.15),
            ("SCN8A-LOF — benign tremor/movement disorder — haploinsufficiency", 0.12),
            ("SCN8A VUS — functional characterisation needed", 0.08),
        ],
        "key_alerts": [
            "SCN8A-GOF-CBZ: Phenytoin/carbamazepine HIGHLY EFFECTIVE for GOF SCN8A-DEE — OPPOSITE of SCN1A — treat early",
            "SCN8A-LOF-AVOID-CBZ: LOF SCN8A (movement disorder) → Na channel blockers WORSEN — critical phenotype distinction",
            "SCN8A-ASND1768: p.Asn1768Asp — most common recurrent GOF hotspot; reports 30%+ of SCN8A-DEE",
            "SCN8A-MORTALITY: ~25% mortality by age 15 — SUDEP and status epilepticus; monitor closely, nighttime supervision",
            "SCN8A-FUNCTIONAL: Request functional characterisation for missense — persistent current assay → GOF confirms CBZ indication",
            "SCN8A-QUINIDINE: Quinidine (Na channel blocker with different kinetics) — case reports; monitoring for cardiac QTc required",
        ],
    },
    # ── KCNT1 — EIMFS / ADNFLE ──
    {
        "gene": "KCNT1",
        "protein": "Sodium-Activated Potassium Channel Slack (KCa4.1) — GOF Hyperactive Current Epilepsy Driver",
        "alias": (
            "KCNT1; OMIM gene 608167; EIMFS OMIM 614959; ADNFLE OMIM 615005; 9q34.3; 1257 aa; ~141 kDa; "
            "KCNT1 encodes Slack (sequence like a calcium-activated K channel), also known as KCa4.1 or "
            "K(Na)1.1, a Na+-activated K+ channel that activates during sustained neuronal firing. "
            "Under normal conditions, Slack limits repetitive firing by providing a long-lasting "
            "hyperpolarisation after high-frequency discharge. GOF mutations → constitutively overactive "
            "channel → paradoxical depolarising block (cell stuck in depolarised-inactivated state) → "
            "disrupted thalamo-cortical rhythms and autonomic regulation. "
            "EIMFS (Epilepsy of Infancy with Migrating Focal Seizures): most severe phenotype; "
            "continuously migrating multifocal ictal EEG; onset <6m; refractory to all standard ASM; "
            "severe global developmental impairment. ADNFLE: milder GOF → nocturnal frontal lobe seizures. "
            "Quinidine blocks KCNT1 current in vitro → clinical trials ongoing; "
            "response variable (30–50% cases); QTc monitoring mandatory. "
            "Quinidine trial warranted for confirmed GOF KCNT1 EIMFS."
        ),
        "aa": "1257 aa",
        "kDa": "~141 kDa",
        "locus": "9q34.3",
        "omim_gene": 608167,
        "omim_disease": 614959,
        "inheritance": "AD (EIMFS/ADNFLE de novo GOF); AR (rare severe DEE both alleles)",
        "gene_class": (
            "KCNT1 encodes Slack, a two-transmembrane-domain K+ channel of the Slo superfamily. "
            "Unlike Ca2+-activated BK channels (Slo1), Slack is gated by intracellular Na+ rising "
            "during sustained neuronal firing. The C-terminal regulatory domain contains two "
            "regulators of K+ conductance (RCK) domains that bind Na+, allosterically opening the pore. "
            "GOF mutations in RCK1 and RCK2 domains increase Na+ sensitivity or constitutively activate "
            "gating, flooding the cell with K+ efflux during firing. The paradox is that excessive K+ "
            "conductance causes depolarisation block rather than hyperpolarisation, because the Na+/K+ ATPase "
            "cannot restore resting membrane potential against massive K+ loss. "
            "Quinidine inhibits Slack by accessing the channel pore in a use-dependent manner — "
            "the same mechanism as its antiarrhythmic action on cardiac channels, requiring ECG monitoring."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("EIMFS — de novo GOF missense RCK1 domain — early severe", 0.38),
            ("EIMFS — de novo GOF missense RCK2 domain", 0.22),
            ("ADNFLE — de novo GOF missense transmembrane domain — milder nocturnal", 0.20),
            ("KCNT1 AR — biallelic LOF severe DEE", 0.12),
            ("KCNT1 VUS — RCK domain missense, functional testing pending", 0.08),
        ],
        "key_alerts": [
            "KCNT1-QUINIDINE: Quinidine trial warranted for GOF KCNT1 EIMFS — 30-50% response; MANDATORY cardiac QTc monitoring before + during",
            "KCNT1-QUINIDINE-CARDIAC: Quinidine is a class Ia antiarrhythmic — risk of QT prolongation/VT; ECG before/after dose, cardiac review",
            "KCNT1-EIMFS-REFRACTORY: EIMFS is refractory to all standard ASM — avoid polypharmacy cascade; ketogenic diet first-line non-drug",
            "KCNT1-FUNCTIONAL: Confirm GOF status by functional assay before quinidine trial — LOF phenotype does NOT benefit from quinidine",
            "KCNT1-AUTONOMIC: Autonomic dysregulation (apnea, bradycardia) in EIMFS — cardiorespiratory monitoring in first year of life",
        ],
    },
    # ── SLC6A1 — MAE / Doose Syndrome ──
    {
        "gene": "SLC6A1",
        "protein": "GABA Transporter GAT-1 — Synaptic GABA Reuptake Haploinsufficiency",
        "alias": (
            "SLC6A1; OMIM gene 137165; MAE/Doose syndrome SLC6A1 OMIM 616874; 3p25.3; 797 aa; ~80 kDa; "
            "SLC6A1 encodes GAT-1 (GABA Transporter 1), the major synaptic GABA reuptake transporter "
            "in cortex, hippocampus, and cerebellum — responsible for terminating inhibitory signalling "
            "by removing GABA from the synaptic cleft. Haploinsufficiency → reduced GAT-1 → prolonged "
            "synaptic GABA → paradoxical net inhibitory deficit via compensatory GABA-A receptor downregulation "
            "and altered tonic/phasic balance. "
            "Myoclonic-Atonic Epilepsy (MAE/Doose syndrome): myoclonic jerks + atonic drop attacks → "
            "major injury risk; ataxia during clusters; cognitive regression during active phase. "
            "Valproate first-line; ketogenic diet highly effective (>50% responders). "
            "AVOID carbamazepine, lamotrigine (worsen myoclonic-atonic); AVOID vigabatrin "
            "(increases GABA — paradoxical effect in haploinsufficiency). "
            "Helmet MANDATORY — drop attack head injury prevention. "
            "ASM should enhance tonic GABAergic transmission (valproate, clobazam, levetiracetam)."
        ),
        "aa": "797 aa",
        "kDa": "~80 kDa",
        "locus": "3p25.3",
        "omim_gene": 137165,
        "omim_disease": 616874,
        "inheritance": "AD — de novo haploinsufficiency dominant (most); inherited ~15%",
        "gene_class": (
            "SLC6A1 encodes GAT-1, a member of the solute carrier 6 (SLC6) family of Na+/Cl−-dependent "
            "neurotransmitter transporters. GAT-1 is a 12-transmembrane-domain protein that co-transports "
            "GABA with 2 Na+ and 1 Cl− per cycle, exploiting the transmembrane Na+ gradient to drive GABA "
            "uptake against its concentration gradient. It is the predominant perisynaptic GABA transporter "
            "in cortex and is expressed in both GABAergic neurons and astrocytes. "
            "GAT-1 haploinsufficiency causes spatially restricted reduction in GABA reuptake capacity. "
            "The apparently paradoxical consequence — increased synaptic GABA duration causing epilepsy — "
            "is explained by compensatory receptor downregulation: excess synaptic GABA duration leads to "
            "desensitisation of GABA-A receptors, reducing GABAergic tone in a delayed, non-linear manner. "
            "This mechanism explains why GAT-1 inhibitors (tiagabine, nipecotic acid) can provoke "
            "absence-like seizures in healthy subjects, and why treatment must rely on valproate-class "
            "agents that enhance GABA synthesis rather than reducing reuptake further."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("MAE — de novo missense (haploinsufficiency) — transmembrane domain", 0.38),
            ("MAE — de novo truncating (nonsense/frameshift)", 0.30),
            ("MAE — de novo splice-site variant", 0.12),
            ("MAE + ASD — de novo missense intracellular loop", 0.13),
            ("MAE — inherited — inherited carrier parent with GEFS+ mild", 0.07),
        ],
        "key_alerts": [
            "SLC6A1-DROP-ATTACKS: Atonic/myoclonic-atonic drop attacks → HEAD INJURY RISK — protective helmet MANDATORY",
            "SLC6A1-AVOID-CBZ-LTG: Carbamazepine/lamotrigine WORSEN myoclonic-atonic epilepsy — contraindicated in MAE",
            "SLC6A1-AVOID-VGB: Vigabatrin (irreversible GABA-T inhibitor) CONTRAINDICATED in SLC6A1 haploinsufficiency — worsens",
            "SLC6A1-VALPROATE: Valproate first-line in MAE/SLC6A1 — enhances GABA synthesis (not reuptake); titrate to clinical response",
            "SLC6A1-KD: Ketogenic diet — >50% seizure reduction in MAE; early trial if VPA partial response",
            "SLC6A1-TIAGABINE-CI: Tiagabine (GAT-1 inhibitor) CONTRAINDICATED — reduces already-impaired GAT-1 function → worse",
        ],
    },
]


def _make_cohort(gene_def):
    r = random.Random(gene_def["seed"])
    gene = gene_def["gene"]
    pts = []
    for i in range(gene_def["n_patients"]):
        # Etiology
        rnd = r.random()
        cumulative = 0.0
        etiol = gene_def["etiologies"][-1][0]
        for label, prob in gene_def["etiologies"]:
            cumulative += prob
            if rnd < cumulative:
                etiol = label
                break

        # Sex and age
        if gene in ("CDKL5", "PCDH19"):
            # X-linked dominant: mostly female affected
            sex = "F" if r.random() < 0.82 else "M"
        elif gene == "SCN1A":
            sex = "F" if r.random() < 0.51 else "M"
        else:
            sex = "F" if r.random() < 0.48 else "M"

        # Age of onset (months)
        if gene == "KCNQ2":
            onset_m = r.uniform(0, 1)  # neonatal
        elif gene == "SCN8A":
            onset_m = r.uniform(0.5, 8)
        elif gene == "SCN1A":
            onset_m = r.uniform(4, 10)
        elif gene == "CDKL5":
            onset_m = r.uniform(1, 5)
        elif gene == "SCN2A":
            onset_m = r.uniform(0, 18)  # bimodal GOF early vs LOF late
        elif gene == "KCNT1":
            onset_m = r.uniform(0.5, 6)
        elif gene == "SLC6A1":
            onset_m = r.uniform(12, 48)
        elif gene == "PCDH19":
            onset_m = r.uniform(6, 36)
        else:
            onset_m = r.uniform(3, 24)

        age_onset_y = round(onset_m / 12, 2)
        age_current_y = round(age_onset_y + r.uniform(1, 18), 1)
        dx_delay_m = round(r.uniform(0.5, onset_m + r.uniform(3, 24)), 1)

        flags = {}

        if gene == "SCN1A":
            flags["dravet_confirmed"] = r.random() < 0.88
            flags["na_channel_blocker_prescribed_erroneously"] = r.random() < 0.24
            flags["sudep_risk_counselled"] = r.random() < 0.72
            flags["fever_protocol_in_place"] = r.random() < 0.81
            flags["rescue_midazolam_prescribed"] = r.random() < 0.88
            flags["stiripentol_prescribed"] = r.random() < 0.42
            flags["cannabidiol_prescribed"] = r.random() < 0.38
            flags["fenfluramine_prescribed"] = r.random() < 0.19
            flags["valproate_prescribed"] = r.random() < 0.74
            flags["status_epilepticus_history"] = r.random() < 0.61
            flags["temperature_trigger"] = r.random() < 0.79
        elif gene == "SCN2A":
            flags["gof_phenotype"] = onset_m < 3
            flags["lof_phenotype"] = onset_m >= 3
            flags["cbz_phenytoin_prescribed"] = r.random() < 0.55
            flags["cbz_response_good"] = (flags["gof_phenotype"] and r.random() < 0.68)
            flags["cbz_worsened"] = (flags["lof_phenotype"] and r.random() < 0.31)
            flags["asd_comorbidity"] = (flags["lof_phenotype"] and r.random() < 0.52)
            flags["functional_test_done"] = r.random() < 0.41
        elif gene == "KCNQ2":
            flags["dee_phenotype"] = r.random() < 0.62
            flags["burst_suppression_eeg"] = r.random() < 0.58
            flags["cbz_phenytoin_early"] = r.random() < 0.61
            flags["eeg_improved_with_cbz"] = r.random() < 0.54
            flags["snhl_screened"] = r.random() < 0.58
            flags["snhl_confirmed"] = r.random() < 0.18
            flags["bfne_family_history"] = r.random() < 0.31
        elif gene == "CDKL5":
            flags["ganaxolone_prescribed"] = r.random() < 0.38
            flags["ketogenic_diet_tried"] = r.random() < 0.61
            flags["ketogenic_diet_response"] = (flags["ketogenic_diet_tried"] and r.random() < 0.51)
            flags["infantile_spasms"] = r.random() < 0.58
            flags["rett_like_features"] = r.random() < 0.72
            flags["cortical_visual_impairment"] = r.random() < 0.61
            flags["ophthalmology_annual"] = r.random() < 0.52
            flags["misdiagnosed_mecp2_rett"] = r.random() < 0.21
            flags["mtor_trial"] = r.random() < 0.09
        elif gene == "PCDH19":
            flags["female_heterozygous"] = sex == "F"
            flags["mosaic_male"] = (sex == "M" and r.random() < 0.08)
            flags["clustering_seizures"] = r.random() < 0.89
            flags["febrile_trigger"] = r.random() < 0.81
            flags["clobazam_cluster_protocol"] = r.random() < 0.71
            flags["lev_prescribed_worsening"] = r.random() < 0.19
            flags["psychiatric_comorbidity"] = r.random() < 0.68
            flags["autism_comorbidity"] = r.random() < 0.42
            flags["remission_post_puberty"] = r.random() < 0.28
            flags["multi_tissue_tested_male"] = (sex == "M" and r.random() < 0.42)
        elif gene == "SCN8A":
            flags["gof_phenotype"] = r.random() < 0.82
            flags["lof_movement_disorder"] = r.random() < 0.12
            flags["cbz_phenytoin_prescribed"] = r.random() < 0.58
            flags["cbz_response_good"] = (flags["gof_phenotype"] and r.random() < 0.62)
            flags["asnd1768_hotspot"] = r.random() < 0.28
            flags["status_epilepticus"] = r.random() < 0.48
            flags["cerebellar_atrophy_mri"] = r.random() < 0.38
            flags["sudep_risk_counselled"] = r.random() < 0.62
            flags["quinidine_trial"] = r.random() < 0.12
        elif gene == "KCNT1":
            flags["eimfs_phenotype"] = r.random() < 0.61
            flags["adnfle_phenotype"] = r.random() < 0.28
            flags["quinidine_trial"] = r.random() < 0.38
            flags["quinidine_qtc_monitored"] = (flags["quinidine_trial"] and r.random() < 0.81)
            flags["quinidine_response"] = (flags["quinidine_trial"] and r.random() < 0.38)
            flags["ketogenic_diet_tried"] = r.random() < 0.54
            flags["autonomic_events"] = r.random() < 0.51
            flags["functional_gof_confirmed"] = r.random() < 0.52
        elif gene == "SLC6A1":
            flags["mae_phenotype"] = r.random() < 0.82
            flags["drop_attacks"] = r.random() < 0.78
            flags["helmet_prescribed"] = r.random() < 0.72
            flags["valproate_prescribed"] = r.random() < 0.80
            flags["na_channel_blocker_prescribed_erroneously"] = r.random() < 0.21
            flags["cbz_worsened_mae"] = r.random() < 0.18
            flags["ketogenic_diet_tried"] = r.random() < 0.58
            flags["ketogenic_diet_response"] = (flags["ketogenic_diet_tried"] and r.random() < 0.52)
            flags["asd_comorbidity"] = r.random() < 0.44
            flags["tiagabine_prescribed_erroneously"] = r.random() < 0.07

        pts.append({
            "pid": f"{gene}-{i+1:03d}",
            "gene": gene,
            "etiology": etiol,
            "sex": sex,
            "age_onset_months": round(onset_m, 1),
            "age_onset_years": age_onset_y,
            "age_current_years": age_current_y,
            "dx_delay_months": dx_delay_m,
            **flags,
        })
    return pts


def get_overview():
    all_patients = []
    gene_summaries = []

    for gd in EPILEPSY_GENES:
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

    scn1a = g_pts("SCN1A")
    scn2a = g_pts("SCN2A")
    kcnq2 = g_pts("KCNQ2")
    cdkl5 = g_pts("CDKL5")
    pcdh19 = g_pts("PCDH19")
    scn8a = g_pts("SCN8A")
    kcnt1 = g_pts("KCNT1")
    slc6a1 = g_pts("SLC6A1")

    agg = {
        "total_patients": n,
        "total_genes": 8,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        # SCN1A — Dravet
        "scn1a_dravet_confirmed_pct": pct(scn1a, "dravet_confirmed"),
        "scn1a_na_blocker_erroneous_pct": pct(scn1a, "na_channel_blocker_prescribed_erroneously"),
        "scn1a_rescue_midazolam_pct": pct(scn1a, "rescue_midazolam_prescribed"),
        "scn1a_fever_protocol_pct": pct(scn1a, "fever_protocol_in_place"),
        "scn1a_sudep_counselled_pct": pct(scn1a, "sudep_risk_counselled"),
        "scn1a_stiripentol_pct": pct(scn1a, "stiripentol_prescribed"),
        "scn1a_cannabidiol_pct": pct(scn1a, "cannabidiol_prescribed"),
        "scn1a_status_history_pct": pct(scn1a, "status_epilepticus_history"),
        # SCN2A
        "scn2a_gof_pct": pct(scn2a, "gof_phenotype"),
        "scn2a_lof_pct": pct(scn2a, "lof_phenotype"),
        "scn2a_cbz_good_response_pct": pct(scn2a, "cbz_response_good"),
        "scn2a_cbz_worsened_pct": pct(scn2a, "cbz_worsened"),
        "scn2a_asd_pct": pct(scn2a, "asd_comorbidity"),
        "scn2a_functional_test_done_pct": pct(scn2a, "functional_test_done"),
        # KCNQ2
        "kcnq2_dee_pct": pct(kcnq2, "dee_phenotype"),
        "kcnq2_burst_suppression_pct": pct(kcnq2, "burst_suppression_eeg"),
        "kcnq2_cbz_early_pct": pct(kcnq2, "cbz_phenytoin_early"),
        "kcnq2_eeg_improved_cbz_pct": pct(kcnq2, "eeg_improved_with_cbz"),
        "kcnq2_snhl_screened_pct": pct(kcnq2, "snhl_screened"),
        "kcnq2_snhl_confirmed_pct": pct(kcnq2, "snhl_confirmed"),
        # CDKL5
        "cdkl5_ganaxolone_pct": pct(cdkl5, "ganaxolone_prescribed"),
        "cdkl5_kd_tried_pct": pct(cdkl5, "ketogenic_diet_tried"),
        "cdkl5_kd_response_pct": pct(cdkl5, "ketogenic_diet_response"),
        "cdkl5_cvi_pct": pct(cdkl5, "cortical_visual_impairment"),
        "cdkl5_ophtho_annual_pct": pct(cdkl5, "ophthalmology_annual"),
        "cdkl5_misdiagnosed_rett_pct": pct(cdkl5, "misdiagnosed_mecp2_rett"),
        # PCDH19
        "pcdh19_clustering_pct": pct(pcdh19, "clustering_seizures"),
        "pcdh19_febrile_trigger_pct": pct(pcdh19, "febrile_trigger"),
        "pcdh19_clobazam_protocol_pct": pct(pcdh19, "clobazam_cluster_protocol"),
        "pcdh19_lev_worsening_pct": pct(pcdh19, "lev_prescribed_worsening"),
        "pcdh19_psychiatric_pct": pct(pcdh19, "psychiatric_comorbidity"),
        "pcdh19_remission_puberty_pct": pct(pcdh19, "remission_post_puberty"),
        # SCN8A
        "scn8a_gof_pct": pct(scn8a, "gof_phenotype"),
        "scn8a_cbz_response_pct": pct(scn8a, "cbz_response_good"),
        "scn8a_asnd1768_pct": pct(scn8a, "asnd1768_hotspot"),
        "scn8a_status_pct": pct(scn8a, "status_epilepticus"),
        "scn8a_sudep_counselled_pct": pct(scn8a, "sudep_risk_counselled"),
        # KCNT1
        "kcnt1_eimfs_pct": pct(kcnt1, "eimfs_phenotype"),
        "kcnt1_quinidine_trial_pct": pct(kcnt1, "quinidine_trial"),
        "kcnt1_quinidine_qtc_pct": pct(kcnt1, "quinidine_qtc_monitored"),
        "kcnt1_quinidine_response_pct": pct(kcnt1, "quinidine_response"),
        "kcnt1_kd_tried_pct": pct(kcnt1, "ketogenic_diet_tried"),
        # SLC6A1
        "slc6a1_mae_pct": pct(slc6a1, "mae_phenotype"),
        "slc6a1_drop_attacks_pct": pct(slc6a1, "drop_attacks"),
        "slc6a1_helmet_pct": pct(slc6a1, "helmet_prescribed"),
        "slc6a1_valproate_pct": pct(slc6a1, "valproate_prescribed"),
        "slc6a1_na_blocker_erroneous_pct": pct(slc6a1, "na_channel_blocker_prescribed_erroneously"),
        "slc6a1_kd_response_pct": pct(slc6a1, "ketogenic_diet_response"),
        # Cross-gene
        "any_na_blocker_erroneous_pct": round(100 * sum(
            1 for p in all_patients if p.get("na_channel_blocker_prescribed_erroneously")
        ) / n, 1),
        "any_status_epilepticus_pct": round(100 * sum(
            1 for p in all_patients if p.get("status_epilepticus_history") or p.get("status_epilepticus")
        ) / n, 1),
        "any_kd_tried_pct": round(100 * sum(
            1 for p in all_patients if p.get("ketogenic_diet_tried")
        ) / n, 1),
    }

    top_alerts = [alert for gd in EPILEPSY_GENES for alert in gd["key_alerts"]]

    return {
        "title": "Hereditary-Epilepsy-Atlas — Complete 8-Gene Hereditary Monogenic Epilepsy Atlas",
        "subtitle": (
            "SCN1A · SCN2A · KCNQ2 · CDKL5 · PCDH19 · SCN8A · KCNT1 · SLC6A1 — "
            "320 patients (8×40, seeds 1486–1493)"
        ),
        "aggregate_stats": agg,
        "genes": gene_summaries,
        "top_alerts": top_alerts,
    }


def get_breakdown():
    breakdown = []
    for gd in EPILEPSY_GENES:
        pts = _make_cohort(gd)
        etiol_counts = {}
        for p in pts:
            etiol_counts[p["etiology"]] = etiol_counts.get(p["etiology"], 0) + 1

        sex_dist = {"M": sum(1 for p in pts if p["sex"] == "M"),
                    "F": sum(1 for p in pts if p["sex"] == "F")}
        mean_onset_m = round(sum(p["age_onset_months"] for p in pts) / len(pts), 1)
        mean_delay = round(sum(p["dx_delay_months"] for p in pts) / len(pts), 1)

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
            "mean_onset_months": mean_onset_m,
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
        "definitions": [
            {
                "term": "SCN1A Precision Pharmacotherapy — Na Channel Blocker Contraindication",
                "definition": (
                    "In Dravet syndrome (SCN1A haploinsufficiency), carbamazepine, lamotrigine, and phenytoin "
                    "are absolutely contraindicated. These agents block all Nav isoforms non-selectively, "
                    "further reducing the already-depleted Nav1.1 in inhibitory interneurons. The clinical "
                    "consequence is worsened seizure frequency, prolonged status epilepticus, and increased "
                    "SUDEP risk. This is Level A evidence (ILAE Dravet guidelines). "
                    "Correct first-line: valproate + clobazam; adjuncts: stiripentol, cannabidiol, fenfluramine. "
                    "All prescribers and emergency physicians treating Dravet patients must be aware."
                ),
            },
            {
                "term": "SCN2A Age-of-Onset GOF vs LOF Pharmacogenomic Rule",
                "definition": (
                    "SCN2A epilepsies divide into two pharmacogenomically distinct groups based solely on age of onset: "
                    "onset <3 months predicts gain-of-function (GOF) — persistent sodium current → carbamazepine/phenytoin "
                    "HIGHLY EFFECTIVE (70% response); onset >3 months predicts loss-of-function (LOF) → "
                    "carbamazepine/phenytoin HARMFUL (worsen seizures and development). "
                    "This age threshold is the critical clinical decision point. Functional assays (patch-clamp) "
                    "confirm GOF vs LOF for missense variants. The biological basis: Nav1.2 is progressively "
                    "replaced by Nav1.6 at AIS during the first year of life — GOF effects are maximal when "
                    "Nav1.2 expression peaks in the neonatal period."
                ),
            },
            {
                "term": "KCNQ2-DEE — Neonatal Treatment Window",
                "definition": (
                    "KCNQ2 developmental and epileptic encephalopathy (DEE) presents in the neonatal period "
                    "with burst-suppression EEG and refractory seizures. Early treatment with carbamazepine "
                    "or phenytoin within the first weeks of life — exploiting the brief window of maximal "
                    "Nav1.2/M-current sensitivity — dramatically suppresses the burst-suppression pattern "
                    "and may improve long-term neurodevelopmental outcome. Delayed treatment loses this window. "
                    "The paradox: KCNQ2 is a K+ channel gene, but Na channel blockers work because they reduce "
                    "the compensatory hyperexcitability that fills the void left by reduced M-current. "
                    "All neonates with burst-suppression EEG + seizures should have KCNQ2 sequencing STAT."
                ),
            },
            {
                "term": "CDKL5 Deficiency Disorder — Distinction from MECP2-Rett Syndrome",
                "definition": (
                    "CDD and classic MECP2-Rett syndrome share Rett-like features (hand stereotypies, "
                    "absent speech, autistic features) but are genetically and clinically distinct. "
                    "CDD: X-linked dominant CDKL5 kinase; seizure onset <5 months (often by 3 months); "
                    "NO regression-plateau phase; continuous refractory epilepsy. "
                    "MECP2-Rett: X-linked dominant MeCP2 transcription factor; normal development until "
                    "12–18 months → regression period → plateau; seizures onset later (after regression). "
                    "CDD requires specific CDKL5 genetic testing — MECP2 panel alone will miss CDD. "
                    "Ganaxolone (FDA-approved 2022) is specific for CDD, not MECP2-Rett."
                ),
            },
            {
                "term": "PCDH19 Cellular Interference Model — Why Males Are Unaffected",
                "definition": (
                    "PCDH19 epilepsy follows X-linked dominant inheritance with an unprecedented mechanism: "
                    "heterozygous females are affected, hemizygous males are NOT affected. "
                    "The cellular interference model (Bhatt 2014) explains this: in heterozygous females, "
                    "random X-inactivation creates a mosaic brain with adjacent PCDH19-expressing and "
                    "PCDH19-null cells. This cellular mismatch disrupts cell-sorting during cortical "
                    "neuronal migration, impairing inhibitory interneuron positioning. "
                    "Uniform-null hemizygous males have no mismatch — no cellular interference. "
                    "Critical exception: somatic mosaic males (PCDH19+ and PCDH19-null cells in same "
                    "brain) ARE affected — standard blood sequencing may miss mosaicism; "
                    "skin biopsy sequencing is recommended for males with unexplained DEE."
                ),
            },
            {
                "term": "SCN8A-DEE — Precision Na Channel Blocker Therapy (Opposite to SCN1A)",
                "definition": (
                    "SCN8A gain-of-function DEE requires sodium channel blockers (carbamazepine, phenytoin) "
                    "as first-line therapy — the pharmacological opposite of SCN1A/Dravet. "
                    "Nav1.6 (SCN8A) is expressed predominantly in excitatory neurons at axon initial segments; "
                    "GOF variants produce persistent Na current that drives pathological burst firing in "
                    "excitatory circuits. Na channel blockers suppress this persistent current, providing "
                    "60–70% seizure reduction in GOF-confirmed SCN8A-DEE. "
                    "Crucially, loss-of-function SCN8A (benign tremor/movement disorder) must NOT receive "
                    "Na channel blockers — functional characterisation is mandatory before CBZ initiation. "
                    "The p.Asn1768Asp hotspot accounts for ~30% of SCN8A-DEE and is reliably GOF."
                ),
            },
            {
                "term": "KCNT1 Quinidine Trial Protocol — Monitoring Requirements",
                "definition": (
                    "Quinidine inhibits KCNT1 (Slack) channels via pore-blocking in a use-dependent manner, "
                    "the same mechanism as its class Ia cardiac antiarrhythmic action. "
                    "Clinical protocol for KCNT1 GOF epilepsy: (1) confirm GOF by functional assay; "
                    "(2) baseline ECG — QTc must be <450 ms before initiation; (3) start at low dose "
                    "(5–8 mg/kg/day BID); (4) ECG at each dose escalation; (5) target quinidine level "
                    "8–12 μg/mL; (6) stop if QTc >500 ms or arrhythmia; (7) assess seizure response at 3 months. "
                    "Approximately 30–50% of patients show meaningful seizure reduction. "
                    "Quinidine for EIMFS: a 'precision therapy' trial, not standard of care — requires "
                    "specialist centre, informed consent, and ECG monitoring infrastructure."
                ),
            },
            {
                "term": "SLC6A1 MAE — Contraindicated Antiseizure Medications",
                "definition": (
                    "Myoclonic-atonic epilepsy (Doose syndrome) due to SLC6A1 haploinsufficiency has several "
                    "drug contraindications based on seizure type and mechanism: "
                    "(1) Carbamazepine, lamotrigine — sodium channel blockers worsen myoclonic and atonic seizures; "
                    "(2) Vigabatrin — irreversible GABA transaminase inhibitor increases synaptic GABA duration, "
                    "paradoxically worsening outcomes in GAT-1 haploinsufficiency via receptor desensitisation; "
                    "(3) Tiagabine — direct GAT-1 inhibitor reduces the already-impaired transporter function. "
                    "Correct approach: valproate (GABA synthesis enhancer) + ketogenic diet. "
                    "Drop attack helmet is mandatory from diagnosis — atonic falls cause serious head injury."
                ),
            },
            {
                "term": "SUDEP Risk in Monogenic Epilepsies — SCN1A and SCN8A",
                "definition": (
                    "Sudden Unexpected Death in Epilepsy (SUDEP) risk is highest in Dravet syndrome (SCN1A): "
                    "lifetime risk 10–20%, predominantly in the first decade. SCN8A-DEE: ~25% mortality by "
                    "age 15 (combined SUDEP + status epilepticus). Risk factors across monogenic epilepsies: "
                    "nocturnal seizures, prone sleeping position, seizure frequency ≥3/month, "
                    "generalised tonic-clonic seizures, male sex. Mitigation: nighttime supervision or "
                    "approved mattress alarm, side-lying position post-seizure, optimise seizure control "
                    "with gene-appropriate ASMs, rescue medication availability. "
                    "All monogenic epilepsy families must receive standardised SUDEP counselling at diagnosis."
                ),
            },
            {
                "term": "Ketogenic Diet in Monogenic Epilepsies — Evidence by Gene",
                "definition": (
                    "Ketogenic diet (KD) evidence in hereditary epilepsy varies by gene: "
                    "CDKL5-CDD: ~50% achieve >50% seizure reduction — strong evidence; early trial recommended; "
                    "SLC6A1-MAE: >50% responders — among best KD responses in genetic epilepsy; "
                    "KCNT1-EIMFS: partial responders (~40%) — worthwhile trial given refractory nature; "
                    "SCN1A-Dravet: 50% >50% reduction — recommended if 2+ ASMs fail; "
                    "KCNQ2-DEE: emerging evidence; worth trial in refractory cases. "
                    "KD mechanism: ketone body metabolism reduces neuronal excitability via multiple pathways "
                    "(KATP channel activation, GABA modulation, HCN channel changes, mTOR suppression). "
                    "KD should be managed by metabolic dietitian + neurologist team."
                ),
            },
            {
                "term": "Cascade Testing — Hereditary Epilepsy Families",
                "definition": (
                    "First-degree relatives of monogenic epilepsy probands should be offered: "
                    "(1) genetic counselling explaining inheritance pattern and recurrence risk; "
                    "(2) targeted testing for the family variant (X-linked: maternal relatives; AD: offspring 50%); "
                    "(3) SCN1A/SCN2A/KCNQ2: EEG monitoring in at-risk children even without overt seizures; "
                    "(4) PCDH19: test sisters of affected females; test brothers for carrier status (unaffected but transmit); "
                    "(5) CDKL5: recurrence risk 1% (de novo) unless parental mosaicism present — "
                    "parental blood + skin biopsy for gonadal mosaicism in severely affected probands; "
                    "(6) Fever and vaccine protocols should be provided to all at-risk families proactively."
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
