#!/usr/bin/env python3
"""Hereditary-Rett-Spectrum-Atlas — Complete 8-Gene Rett & Rett-Related Neurodevelopmental Atlas
MECP2  (methyl-CpG binding protein 2; 498 aa; Xq28; X-linked dominant;
         Classic Rett syndrome (RTT) — females; MECP2 Duplication (males);
         HAND STEREOTYPIES PATHOGNOMONIC (hand wringing/washing);
         Regression 6-18 months; BREATHING IRREGULARITIES PATHOGNOMONIC;
         seed SEED_BASE+0) ·
CDKL5  (cyclin-dependent kinase-like 5; 1030 aa; Xp22.13; X-linked dominant;
         CDKL5 Deficiency Disorder (CDD); EARLY-ONSET SEIZURES BEFORE 5 MONTHS PATHOGNOMONIC;
         Hand stereotypies (mouthing); never achieves independent ambulation majority;
         seed SEED_BASE+1) ·
FOXG1  (forkhead box G1; 481 aa; 14q12; AD de novo;
         FOXG1 Syndrome (Congenital Rett variant);
         NO REGRESSION — congenital onset (distinguishes from classic Rett);
         HYPERSALIVATION + DYSKINESIA PATHOGNOMONIC combination; profound ID;
         seed SEED_BASE+2) ·
MEF2C  (myocyte enhancer factor 2C; 473 aa; 5q14.3; AD de novo LOF;
         MEF2C Haploinsufficiency Syndrome (MRFMEF2C);
         STEREOTYPIC HAND MOVEMENTS + HYPERKINESIS + SEIZURES + ABSENT SPEECH;
         Myeloid leukemia susceptibility (unique feature);
         seed SEED_BASE+3) ·
WDR45  (WD repeat domain 45; 364 aa; Xp11.23; X-linked dominant de novo;
         Beta-propeller protein-associated neurodegeneration (BPAN) = NBIA5;
         BIPHASIC: childhood ID+seizures → young adult PARKINSONISM+DEMENTIA;
         IRON ACCUMULATION GLOBUS PALLIDUS + SUBSTANTIA NIGRA T2 HYPOINTENSITY PATHOGNOMONIC;
         seed SEED_BASE+4) ·
DDX3X  (DEAD-box helicase 3 X-linked; 662 aa; Xp11.3; X-linked dominant de novo;
         DDX3X Syndrome; MOST COMMON X-LINKED ID IN FEMALES;
         Mild-moderate ID; autism 50%; hypotonia; corpus callosum anomalies;
         seed SEED_BASE+5) ·
PURA   (purine-rich element binding protein A; 322 aa; 5q31.2; AD de novo LOF;
         PURA Syndrome (PURAMINE);
         NEONATAL HYPOTONIA + APNOEA + FEEDING TUBE ALMOST UNIVERSAL;
         EXCESSIVE DAYTIME SLEEPINESS PATHOGNOMONIC;
         seed SEED_BASE+6) ·
HNRNPH2 (heterogeneous nuclear ribonucleoprotein H2; 449 aa; Xq22.1; X-linked dominant de novo;
          HNRNPH2 Syndrome (Au-Kline syndrome);
          PROMINENT FOREHEAD + HYPERTELORISM + BROAD NASAL TIP PATHOGNOMONIC FACIES;
          Severe ID; absent-limited speech; brachycephaly; short stature;
          seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 3054-3061)
"""
import random

SEED_BASE = 3054

ATLAS_GENES = [
    {
        "gene": "MECP2",
        "protein": (
            "MECP2 -- Xq28 X-linked-Dominant -- 498aa -- Methyl-CpG-Binding-Protein-2-"
            "MBD-TRD-NLS-RTT-Classic-Rett-HAND-STEREOTYPIES-BREATHING-IRREGULARITIES-PATHOGNOMONIC-OMIM-300005"
        ),
        "locus": "Xq28",
        "protein_size": (
            "498 aa / 52 kDa (MeCP2; methyl-CpG binding protein 2; "
            "STRUCTURE: N-terminal domain + MBD (methyl-CpG binding domain) + TRD (transcriptional repression domain) + "
            "NLS (nuclear localisation signal) + C-terminal domain; "
            "ISOFORMS: MeCP2e1 (dominant in brain) vs MeCP2e2 (peripheral); "
            "FUNCTION: "
            "  Binds methylated CpG → recruits co-repressors (NCoR1, Sin3A, HDAC) → transcriptional silencing; "
            "  Also activates transcription of BDNF + other neuronal genes; "
            "  Critical for neuronal maturation, synaptogenesis, astrocyte function; "
            "  Loss → global transcriptional dysregulation of hundreds of genes in neurons; "
            "CLASSIC RETT SYNDROME (RTT) — FEMALES: "
            "  PREVALENCE: ~1:10,000 live female births; MOST COMMON cause of profound ID in females; "
            "  FOUR STAGES: "
            "    Stage 1 (6-18 months): Developmental stagnation — SUBTLE; often missed; "
            "    Stage 2 (1-3 years): REGRESSION — rapid loss of purposeful hand use, speech, social skills; "
            "      HAND STEREOTYPIES EMERGE (PATHOGNOMONIC): hand wringing/washing/squeezing; "
            "      BREATHING IRREGULARITIES (PATHOGNOMONIC): apnoea, hyperventilation, breath-holding; "
            "    Stage 3 (2-10 years): PLATEAU — seizures develop (80%); improvement in hand use; "
            "    Stage 4 (10+ years): LATE MOTOR DETERIORATION — scoliosis, wheelchair; "
            "  PATHOGNOMONIC COMBINATION: regression + HAND STEREOTYPIES + BREATHING IRREGULARITIES; "
            "  AUTONOMIC FEATURES: irregular heart rate (long QT); vasomotor instability (cold blue feet); "
            "  SEIZURES: 80%+ — multifocal, absence, tonic-clonic; EEG: slow background + spikes; "
            "  SCOLIOSIS: 80%+ — progresses with age; spinal surveillance mandatory; "
            "  GROWTH: microcephaly develops postnatally (not congenital); "
            "  BREATHING: hyperventilation + apnoea characteristic — SpO2 monitoring; "
            "MECP2 DUPLICATION SYNDROME (MALES): "
            "  MALES with Xq28 duplication including MECP2: "
            "  PROGRESSIVE SPASTIC QUADRIPLEGIA — early hypotonia → spasticity; "
            "  RECURRENT RESPIRATORY INFECTIONS — leading cause of death; "
            "  MILD DYSMORPHIC FEATURES; absent speech majority; "
            "  FEMALES carriers: usually asymptomatic (X-inactivation); occasional mild features; "
            "  INCIDENCE: uncommon; "
            "TREATMENT: "
            "  Trofinetide (DAYBUE) — FDA APPROVED 2023 for Rett syndrome (ALL ages); "
            "  Mechanism: IGF-1 analogue → synaptic restoration; "
            "  Seizure: valproate, levetiracetam, lamotrigine; AVOID carbamazepine (worsens autonomic); "
            "  Scoliosis: bracing early; surgery threshold lower than idiopathic; "
            "  Trofinetide: 12mg/kg/dose BD in paediatric weight bands; GI side effects; "
            "TESTING: MECP2 sequencing + MLPA (deletion/duplication 7-10%); "
        ),
        "inheritance": (
            "X-linked dominant de novo LOF (classic Rett — females; duplication — males); "
            "MECP2 Xq28; 95%+ de novo; familial rare (germline mosaicism documented); "
            "Carriers: female carriers of LOF usually unaffected (skewed X-inactivation); "
            "Male LOF hemizygous: encephalopathy lethal in early childhood (rare surviving: somatic mosaic); "
        ),
        "disease_category": "Rett-Spectrum / X-linked Neurodevelopmental",
        "key_mutations": [
            "p.Arg306Cys (common missense — Stage 3 plateau preserved)",
            "p.Arg255Ter (common nonsense — moderate-severe)",
            "p.Arg270Ter (common nonsense)",
            "p.Thr158Met (most common missense — variable)",
            "p.Arg168Ter (common nonsense — severe)",
            "p.Arg133Cys (missense — milder)",
            "Large deletions (exon 3-4) — MLPA required",
        ],
        "clinical_keys": [
            "HAND STEREOTYPIES (wringing/washing) PATHOGNOMONIC — onset Stage 2",
            "BREATHING IRREGULARITIES (apnoea/hyperventilation) PATHOGNOMONIC",
            "Regression 6-18 months — purposeful hand use + speech loss",
            "Trofinetide (DAYBUE) FDA 2023 — first approved treatment",
            "AVOID carbamazepine — worsens autonomic features",
            "Scoliosis 80%+ — spine X-ray annually from diagnosis",
            "Long QT — ECG mandatory; avoid QT-prolonging drugs",
            "Males with MECP2 LOF: encephalopathic (usually lethal); mosaic survives",
        ],
    },
    {
        "gene": "CDKL5",
        "protein": (
            "CDKL5 -- Xp22.13 X-linked-Dominant -- 1030aa -- Cyclin-Dependent-Kinase-Like-5-"
            "Serine-Threonine-Kinase-MECP2-Phosphorylation-CDD-EARLY-ONSET-SEIZURES-PATHOGNOMONIC-OMIM-300203"
        ),
        "locus": "Xp22.13",
        "protein_size": (
            "1030 aa / 115 kDa (CDKL5; cyclin-dependent kinase-like 5; "
            "STRUCTURE: N-terminal kinase domain (catalytic) + C-terminal regulatory domain; "
            "FUNCTION: "
            "  Serine/threonine kinase; phosphorylates MeCP2 at Ser80 and Ser229; "
            "  Regulates dendritic morphology, synaptogenesis; "
            "  CDKL5 localises to nucleus and synapses; "
            "  Loss → impaired synaptic plasticity, reduced dendritic spine density; "
            "CDKL5 DEFICIENCY DISORDER (CDD): "
            "  PREVALENCE: ~1:40,000–1:60,000; underdiagnosed historically; "
            "  KEY DISTINCTION FROM RTT: "
            "    CDD: SEIZURES BEFORE 5 MONTHS (PATHOGNOMONIC) — 80%+ before 5 months; "
            "    RTT: seizures onset Stage 3 (2-10 years) AFTER regression; "
            "    CDD: hand stereotypies PRESENT but predominantly HAND MOUTHING (not wringing); "
            "    CDD: NO CLASSIC REGRESSION (developmental plateau from outset — never achieves milestones); "
            "    CDD: BREATHING IRREGULARITIES ABSENT (distinguishes from RTT); "
            "  SEIZURE TYPES IN CDD: "
            "    Epileptic spasms (infantile spasms) — most common; "
            "    Focal seizures — temporal/occipital; "
            "    Tonic seizures; hypermotor seizures; "
            "    TYPICALLY DRUG-RESISTANT — multiple AED trials often needed; "
            "  MOTOR: "
            "    Majority never achieve independent ambulation; "
            "    Truncal hypotonia + peripheral hypertonicity; "
            "    Stereotyped hand movements — HAND MOUTHING (not wringing — distinguishes from RTT); "
            "  COGNITIVE: profound ID in most; "
            "  SOCIAL: eye contact maintained (more than RTT) — smiling, social responsiveness; "
            "  HAND-EYE: purposeful reaching sometimes preserved; "
            "GENDER EFFECTS: "
            "  Females predominantly affected (X-linked dominant); "
            "  Males with hemizygous CDKL5 loss: more severe, often more seizures; "
            "  Female carriers: X-inactivation affects severity; "
            "TREATMENT: "
            "  No FDA-approved specific therapy (unlike Rett — trofinetide); "
            "  Vigabatrin: reasonable first-line (spasms); "
            "  Ketogenic diet: documented benefit in CDD — RECOMMEND early; "
            "  ACTH/prednisolone: infantile spasms standard; "
            "  Cannabidiol (Epidiolex): FDA for refractory; evidence in CDD growing; "
            "  Sodium valproate + clobazam: common baseline; "
            "  AVOID: avoid phenytoin (worsens) in many; "
            "TESTING: CDKL5 sequencing; include MLPA (deletions documented); X-linked → test mother; "
        ),
        "inheritance": (
            "X-linked dominant de novo LOF (predominantly females affected); "
            "CDKL5 Xp22.13; 95%+ de novo; "
            "Hemizygous males: more severe presentation; "
            "Carrier females: occasionally mild features (skewed X-inactivation); "
        ),
        "disease_category": "CDKL5 Deficiency Disorder / X-linked Epileptic Encephalopathy",
        "key_mutations": [
            "p.Arg59Ter (common nonsense — severe, early spasms)",
            "p.Arg178Trp (missense kinase domain)",
            "p.Arg564Ter (C-terminal nonsense)",
            "p.Ala40Val (missense kinase domain)",
            "Exon deletions (MLPA)",
            "p.Pro365Arg (missense)",
            "p.Leu220Pro (missense kinase domain)",
        ],
        "clinical_keys": [
            "EARLY-ONSET SEIZURES BEFORE 5 MONTHS PATHOGNOMONIC — distinguishes from RTT",
            "HAND MOUTHING stereotypy (not wringing) — distinguishes from classic Rett",
            "NO REGRESSION — plateau from outset (differs from Rett Stage 2)",
            "Majority never walk independently",
            "Ketogenic diet — recommend early (documented seizure reduction in CDD)",
            "NO breathing irregularities — helps distinguish from MECP2-Rett",
            "Vigabatrin first-line for infantile spasms in CDD",
            "Eye contact maintained — more social than RTT",
        ],
    },
    {
        "gene": "FOXG1",
        "protein": (
            "FOXG1 -- 14q12 AD-de-novo -- 481aa -- Forkhead-Box-G1-"
            "Transcription-Factor-Telencephalon-Development-FOXG1-Syndrome-NO-REGRESSION-DYSKINESIA-HYPERSALIVATION-PATHOGNOMONIC-OMIM-164874"
        ),
        "locus": "14q12",
        "protein_size": (
            "481 aa / 54 kDa (FOXG1; forkhead box G1; "
            "STRUCTURE: forkhead domain (DNA binding) + N-terminal JARID1B-interacting domain + C-terminal domain; "
            "FUNCTION: "
            "  Master transcription factor for telencephalon development; "
            "  Regulates cortical progenitor proliferation and differentiation; "
            "  FOXG1 represses CDKN1C (p57kip2) → maintains progenitor pool; "
            "  Critical for development of cerebral cortex, hippocampus, basal ganglia; "
            "  Loss → cortical hypoplasia, simplified gyration, reduced cortical volume; "
            "FOXG1 SYNDROME (CONGENITAL RETT VARIANT): "
            "  PREVALENCE: ~1:100,000; "
            "  CONGENITAL ONSET (DISTINGUISHES FROM CLASSIC RETT): "
            "    NO NORMAL DEVELOPMENTAL PERIOD — abnormalities evident from birth/first months; "
            "    NO REGRESSION — never achieves milestones to regress from; "
            "    Classic RTT: normal first 6 months then regression; FOXG1: abnormal from birth; "
            "  PATHOGNOMONIC FEATURES: "
            "    DYSKINESIA: hyperkinetic involuntary movements (choreoathetosis, dystonia); "
            "    HYPERSALIVATION: drooling, difficulty managing secretions; "
            "    Combination: DYSKINESIA + HYPERSALIVATION + ABSENT SPEECH = FOXG1 triad; "
            "  INTELLECTUAL DISABILITY: PROFOUND — deepest of Rett spectrum; "
            "  SPEECH: ABSENT in virtually all; "
            "  MOTOR: truncal hypotonia; peripheral hypertonicity; "
            "    Ambulation: majority DO NOT walk independently; "
            "  BRAIN MRI: "
            "    Simplified gyral pattern (pachygyria/lissencephaly spectrum); "
            "    Reduced frontal lobe volume; "
            "    Hypoplasia of corpus callosum (especially genu); "
            "    Reduced myelination; "
            "    Basal ganglia hypoplasia/signal changes; "
            "  SEIZURES: 80%+ — drug-resistant; multiple types; "
            "  SOCIAL: partial social awareness (eye contact) despite profound ID; "
            "  BREATHING: LESS prominent than classic RTT; "
            "GENOTYPE-PHENOTYPE: "
            "  Intragenic LOF (frame/nonsense): typically more severe (classic Foxg1 syndrome); "
            "  14q12 chromosomal deletion: may include adjacent PRKD1/BMP4 → additional features; "
            "  Missense forkhead domain: variable — sometimes milder; "
            "TREATMENT: "
            "  No approved specific therapy; "
            "  Seizures: levetiracetam, valproate, clonazepam; ketogenic diet reasonable; "
            "  Dyskinesia: trihexyphenidyl (anticholinergic) — may help; baclofen for spasticity; "
            "  Drooling: scopolamine patch; glycopyrrolate; botulinum toxin parotid; "
            "  Feeds: NG tube / gastrostomy frequent; "
            "TESTING: FOXG1 sequencing + deletion/duplication (14q12 del panel); "
            "  14q12 chromosomal microarray for broader deletions; "
        ),
        "inheritance": (
            "Autosomal dominant de novo LOF (intragenic) or 14q12 deletion (CMA); "
            "FOXG1 14q12; virtually all de novo; "
            "Rare familial — parental mosaicism documented; "
            "No sex predilection (unlike MECP2/CDKL5); affects males + females equally; "
        ),
        "disease_category": "FOXG1 Syndrome / Congenital Rett Variant / AD NDD",
        "key_mutations": [
            "p.Trp308Ter (common nonsense)",
            "p.Ala472Val (C-terminal missense)",
            "p.Arg461Cys (forkhead domain missense)",
            "p.Gly271Asp (forkhead domain missense)",
            "14q12 chromosomal deletion (CMA)",
            "p.Arg327His (forkhead missense)",
            "p.Ser323Leu (missense)",
        ],
        "clinical_keys": [
            "NO REGRESSION — congenital onset (key difference from classic Rett)",
            "DYSKINESIA + HYPERSALIVATION PATHOGNOMONIC combination",
            "Profound ID — deepest of Rett spectrum",
            "Simplified gyral pattern on brain MRI — frontal lobe predominant",
            "Absent speech virtually universal",
            "No sex predilection — unlike MECP2/CDKL5 (both sexes equally affected)",
            "14q12 deletion on CMA vs intragenic FOXG1 variant — both cause same syndrome",
            "Drooling management: scopolamine/glycopyrrolate/botulinum toxin",
        ],
    },
    {
        "gene": "MEF2C",
        "protein": (
            "MEF2C -- 5q14.3 AD-de-novo-LOF -- 473aa -- Myocyte-Enhancer-Factor-2C-"
            "MADS-box-MEF2-domain-TF-MRFMEF2C-STEREOTYPIES-HYPERKINESIS-ABSENT-SPEECH-Myeloid-Leukemia-OMIM-600662"
        ),
        "locus": "5q14.3",
        "protein_size": (
            "473 aa / 52 kDa (MEF2C; myocyte enhancer factor 2C; "
            "STRUCTURE: MADS box + MEF2 domain (N-terminal DNA binding/dimerisation) + C-terminal transactivation domain; "
            "FUNCTION: "
            "  Transcription factor; MEF2 family (MEF2A-D); "
            "  Regulates neuronal differentiation, synaptic plasticity, dendritic development; "
            "  Critical for GABAergic interneuron development; "
            "  Also expressed in cardiac muscle, skeletal muscle, T cells, myeloid cells; "
            "  Loss → reduced inhibitory interneuron function → excitatory-inhibitory imbalance → seizures + stereotypies; "
            "MEF2C HAPLOINSUFFICIENCY (MRFMEF2C) — CLINICAL: "
            "  PREVALENCE: rare; underdiagnosed; 5q14.3 deletion most common; "
            "  CARDINAL FEATURES: "
            "    STEREOTYPIC HAND MOVEMENTS (PATHOGNOMONIC in context): "
            "      Midline hand movements — similar to Rett but distinct pattern; "
            "    HYPERKINESIS: excessive purposeless movements (distinguishes from Rett); "
            "    EPILEPSY: 80%+ — often drug-resistant; various types; infantile spasms common; "
            "    ABSENT SPEECH: non-verbal virtually universal; "
            "    PROFOUND ID: deepest in NDD spectrum; "
            "  ADDITIONAL FEATURES: "
            "    MYELOID LEUKEMIA SUSCEPTIBILITY (UNIQUE): "
            "      MEF2C is a myeloid differentiation factor; "
            "      AML documented in MEF2C haploinsufficiency — ANNUAL FBC SURVEILLANCE; "
            "    Hypotonia — truncal; "
            "    Stereotyped tongue movements; drooling; "
            "    Autistic features in majority; "
            "    Poor temperature regulation; "
            "  BRAIN MRI: "
            "    Often normal OR simplified gyral pattern; "
            "    Thin corpus callosum in some; "
            "    Cerebral volume reduction; "
            "  5q14.3 DELETION: "
            "    Most common mechanism (~50%): CMA detects; variable size; "
            "    Smaller deletions: just MEF2C; larger: RGNEF, CETN3 also included; "
            "    Intragenic MEF2C variants (~50%): sequencing required; "
            "GENOTYPE-PHENOTYPE: "
            "  Larger 5q14.3 deletions: more severe (additional gene haploinsufficiency); "
            "  Pure MEF2C LOF: core syndrome as above; "
            "TREATMENT: "
            "  No specific approved therapy; "
            "  Seizures: levetiracetam, valproate, vigabatrin (spasms); ketogenic diet; "
            "  Hyperkinesis: clonazepam may reduce; avoid stimulants; "
            "  FBC ANNUALLY — myeloid leukemia surveillance; "
            "TESTING: CMA FIRST (5q14.3 deletion ~50%); MEF2C sequencing if CMA normal; "
        ),
        "inheritance": (
            "Autosomal dominant de novo LOF (intragenic) or 5q14.3 chromosomal deletion; "
            "MEF2C 5q14.3; virtually all de novo; "
            "5q14.3 deletion: CMA detects; MEF2C single gene: sequencing; "
            "No sex predilection; "
        ),
        "disease_category": "MEF2C Haploinsufficiency / 5q14.3 Deletion Syndrome",
        "key_mutations": [
            "5q14.3 chromosomal deletion (CMA — most common ~50%)",
            "p.Arg3Ter (nonsense MADS box)",
            "p.Gln13Ter (nonsense)",
            "p.Tyr131Cys (missense MEF2 domain)",
            "p.Gln68Ter (nonsense)",
            "p.Arg187Ter (nonsense transactivation domain)",
            "Exon deletions (MLPA / CMA)",
        ],
        "clinical_keys": [
            "STEREOTYPIC HAND MOVEMENTS + HYPERKINESIS + ABSENT SPEECH triad",
            "MYELOID LEUKEMIA SUSCEPTIBILITY — annual FBC mandatory (unique feature)",
            "5q14.3 deletion by CMA (~50% mechanism) — CMA FIRST",
            "Profound ID — virtually non-verbal",
            "Drug-resistant epilepsy — multiple AED trials",
            "Ketogenic diet — evidence supports",
            "MEF2C sequencing if CMA normal",
            "Distinguish from Rett: HYPERKINESIS (not seen in RTT)",
        ],
    },
    {
        "gene": "WDR45",
        "protein": (
            "WDR45 -- Xp11.23 X-linked-Dominant-de-novo -- 364aa -- WD-Repeat-Domain-45-"
            "Beta-Propeller-Autophagy-Adaptor-BPAN-NBIA5-BIPHASIC-IRON-ACCUMULATION-GLOBUS-PALLIDUS-SUBSTANTIA-NIGRA-PATHOGNOMONIC-OMIM-300526"
        ),
        "locus": "Xp11.23",
        "protein_size": (
            "364 aa / 40 kDa (WDR45; WD repeat domain 45; "
            "STRUCTURE: 7-bladed WD40 beta-propeller structure; WIPI2 paralogous; "
            "FUNCTION: "
            "  Phosphoinositide-binding autophagy adaptor (PI3P binding); "
            "  WIPI family member — regulates autophagosome biogenesis (Stage 1-2 autophagy); "
            "  WDR45 interacts with ATG2A/B — membrane tethering; "
            "  Loss → impaired autophagy → intracellular iron accumulation in neurons; "
            "  NBIA = Neurodegeneration with Brain Iron Accumulation; "
            "BETA-PROPELLER PROTEIN-ASSOCIATED NEURODEGENERATION (BPAN = NBIA5): "
            "  PREVALENCE: ~1:100,000; MOST COMMON NBIA in paediatric series; "
            "  BIPHASIC CLINICAL COURSE (PATHOGNOMONIC): "
            "    CHILDHOOD PHASE: "
            "      Intellectual disability — mild-moderate; "
            "      Epilepsy — multiple types (infantile spasms, absence, focal, myoclonic); "
            "      Autistic features; "
            "      Non-specific global developmental delay; "
            "      BRAIN MRI CHILDHOOD: may appear normal or mild signal changes; "
            "    ADOLESCENCE / YOUNG ADULT PHASE (SECOND PHASE): "
            "      PARKINSONISM — sudden onset; tremor, bradykinesia, rigidity; "
            "      DYSTONIA — severe; may be painful; "
            "      RAPID COGNITIVE DECLINE → DEMENTIA; "
            "      BRAIN MRI: T2 HYPOINTENSITY GLOBUS PALLIDUS + SUBSTANTIA NIGRA (iron deposits) PATHOGNOMONIC; "
            "        Halo of T1 hyperintensity around hypointense globus pallidus ('double panda' not BPAN); "
            "        BPAN: T2 hypointensity GP + SN; T1 hyperintense halo periGP; "
            "  SEX PREDILECTION: "
            "    FEMALES predominantly affected (X-linked dominant); "
            "    Males with hemizygous WDR45 loss: severe encephalopathy; rare surviving cases; "
            "    Female carriers: X-inactivation determines expression; "
            "MANAGEMENT: "
            "  IRON CHELATION: "
            "    Deferiprone: crosses blood-brain barrier; FIRST-CHOICE chelator in BPAN; "
            "    Clinical trial evidence: modest stabilisation; not curative; "
            "  PARKINSONISM: levodopa — partial response; may help tremor/bradykinesia; "
            "  DYSTONIA: clonazepam; baclofen; tetrabenazine; DBS investigational; "
            "  EPILEPSY: valproate, levetiracetam — standard; "
            "  SURVEILLANCE: "
            "    Brain MRI with iron-sensitive sequences (SWI/susceptibility-weighted) every 2-3 years; "
            "    Screen for parkinsonism/dystonia in adolescence — early treatment may slow; "
            "TESTING: WDR45 sequencing; MLPA; X-linked → test mother; "
        ),
        "inheritance": (
            "X-linked dominant de novo (virtually all sporadic — de novo); "
            "WDR45 Xp11.23; males with hemizygous LOF: severe encephalopathy/lethal; "
            "Females predominantly: X-linked dominant; "
            "X-inactivation: random — determines phenotype in female carriers; "
        ),
        "disease_category": "BPAN / NBIA5 / X-linked Autophagy-Neurodegeneration",
        "key_mutations": [
            "p.Arg152Ter (common nonsense — severe)",
            "p.Arg152Gln (missense — milder parkinsonian)",
            "p.Tyr38Cys (beta-propeller missense)",
            "p.Asp92Gly (WD40 missense)",
            "p.Glu226Ter (C-terminal nonsense)",
            "Exon deletions (MLPA)",
            "Splice-site variants (exon skipping — MLPA confirms)",
        ],
        "clinical_keys": [
            "BIPHASIC: childhood ID+epilepsy → adolescent PARKINSONISM+DEMENTIA PATHOGNOMONIC",
            "T2 HYPOINTENSITY GLOBUS PALLIDUS + SUBSTANTIA NIGRA on MRI PATHOGNOMONIC (iron deposits)",
            "Deferiprone (iron chelator) — FIRST-CHOICE brain-penetrant chelator in BPAN",
            "Levodopa for parkinsonism — partial response",
            "SWI MRI sequences mandatory for iron detection",
            "Females predominantly (X-linked dominant); males hemizygous: usually lethal",
            "Screen adolescents proactively: parkinsonism onset sudden — early treatment",
            "BPAN: MOST COMMON NBIA in paediatric series",
        ],
    },
    {
        "gene": "DDX3X",
        "protein": (
            "DDX3X -- Xp11.3 X-linked-Dominant-de-novo -- 662aa -- DEAD-Box-Helicase-3-X-Linked-"
            "RNA-Helicase-mRNA-Translation-DDXX3X-Syndrome-MOST-COMMON-XL-ID-FEMALES-OMIM-300160"
        ),
        "locus": "Xp11.3",
        "protein_size": (
            "662 aa / 73 kDa (DDX3X; DEAD-box helicase 3 X-linked; "
            "STRUCTURE: N-terminal domain + DEAD (Asp-Glu-Ala-Asp) helicase core (motifs I-VI) + C-terminal domain; "
            "FUNCTION: "
            "  ATP-dependent RNA helicase; unwinds RNA secondary structures; "
            "  Regulates mRNA translation (especially IRES-mediated translation); "
            "  Involved in stress granule formation, nonsense-mediated decay; "
            "  Critical for neuronal differentiation and cortical development; "
            "  Loss → impaired mRNA translation → reduced protein synthesis in neurons; "
            "DDX3X SYNDROME: "
            "  PREVALENCE: MOST COMMON X-LINKED ID IN FEMALES (~1:40,000 females); "
            "    DDX3X may account for 1-3% of unexplained ID in girls; "
            "  CORE FEATURES: "
            "    INTELLECTUAL DISABILITY: mild-moderate (range mild to severe); "
            "    AUTISM SPECTRUM DISORDER: ~50% of affected females; "
            "    HYPOTONIA: truncal (universal); "
            "    BRAIN MRI: "
            "      Corpus callosum anomalies (30-40%): thin/hypoplastic; "
            "      Periventricular nodular heterotopia (15%); "
            "      Simplified gyral pattern (minority); "
            "    SEIZURES: 40-50%; various types; usually responsive; "
            "    SHORT STATURE: common; "
            "    FACIAL FEATURES: non-specific; "
            "  MALES: "
            "    Hemizygous DDX3X LOF: severe encephalopathy (DDX3Y partially compensates in males); "
            "    Most males with DDX3X LOF: not viable; rare hemizygous surviving = mosaic; "
            "    DDX3Y (Y-linked paralogue) provides partial compensation in normal males; "
            "  SEVERITY SPECTRUM: "
            "    Missense gain-of-function-like: often more severe (dominant negative); "
            "    LOF nonsense/frameshift: moderate severity; "
            "    Degree of X-inactivation skewing determines phenotype in females; "
            "TREATMENT: "
            "  No approved specific therapy; "
            "  Seizures: standard AEDs — usually responsive; levetiracetam first-line; "
            "  ASD: behavioural intervention (ABA); "
            "  Physiotherapy for hypotonia; "
            "  Language therapy — expressive delay common; "
            "TESTING: DDX3X sequencing; X-linked → test mother; MLPAfor deletions; "
            "  WES increasingly first-line (DDX3X often found on exome in unexplained ID females); "
        ),
        "inheritance": (
            "X-linked dominant de novo LOF (predominantly females affected); "
            "DDX3X Xp11.3; >95% de novo; "
            "Hemizygous males: severe/lethal (DDX3Y partially compensates in normal males); "
            "Carrier females: usually unaffected (skewed X-inactivation); "
        ),
        "disease_category": "DDX3X Syndrome / X-linked ID Females",
        "key_mutations": [
            "p.Arg326His (common missense helicase core — dominant negative effect)",
            "p.Arg376Cys (missense — common)",
            "p.Glu348Asp (missense)",
            "p.Arg488Ter (nonsense)",
            "p.Arg534Ter (C-terminal nonsense)",
            "p.Thr204Met (missense helicase)",
            "Exon deletions (MLPA)",
        ],
        "clinical_keys": [
            "MOST COMMON X-LINKED ID IN FEMALES (1-3% unexplained ID in girls)",
            "Corpus callosum anomalies 30-40% on brain MRI",
            "Autism 50% — ABA behavioural intervention",
            "Hypotonia universal — physiotherapy",
            "Seizures 40-50% — usually AED-responsive (unlike CDKL5/MEF2C)",
            "Missense > LOF: often more severe (dominant negative mechanism)",
            "Males hemizygous: DDX3Y partially compensates; hemizygous LOF: lethal/severe",
            "WES increasingly finds DDX3X as first diagnosis in unexplained ID females",
        ],
    },
    {
        "gene": "PURA",
        "protein": (
            "PURA -- 5q31.2 AD-de-novo-LOF -- 322aa -- Purine-Rich-Element-Binding-Protein-A-"
            "ssDNA-RNA-Binding-PUR-Domain-PURA-Syndrome-NEONATAL-HYPOTONIA-APNOEA-FEEDING-TUBE-EXCESSIVE-DAYTIME-SLEEPINESS-PATHOGNOMONIC-OMIM-600473"
        ),
        "locus": "5q31.2",
        "protein_size": (
            "322 aa / 35 kDa (PURA; purine-rich element binding protein A; "
            "STRUCTURE: three PUR domains (I, II, III) — single-strand DNA/RNA binding; homodimerises; "
            "FUNCTION: "
            "  Binds purine-rich single-stranded DNA and RNA motifs (GGGN repeats); "
            "  Regulates transcription and mRNA translation in neurons; "
            "  Critical for neuronal differentiation, axon development, dendritic mRNA transport; "
            "  PURA interacts with FMR1 (fragile X protein) — links to FMRP pathway; "
            "  Loss → impaired neuronal gene regulation → profound early-onset NDD; "
            "PURA SYNDROME (PURAMINE): "
            "  PREVALENCE: rare; ~200 cases reported; likely underdiagnosed; "
            "  NEONATAL PRESENTATION (ALMOST UNIVERSAL): "
            "    NEONATAL HYPOTONIA — severe, universal; PICU admission common; "
            "    APNOEA: respiratory episodes; often requires monitoring/CPAP; "
            "    FEEDING TUBE: NG tube or gastrostomy — majority; "
            "    Neonatal jitteriness/tremors; "
            "  PATHOGNOMONIC FEATURE: "
            "    EXCESSIVE DAYTIME SLEEPINESS (EDS): "
            "      Profound hypersomnia from infancy; "
            "      Naps multiple times daily; "
            "      Persists into childhood; "
            "      May partially improve with age; "
            "      EDS is the most distinctive clinical feature of PURA syndrome; "
            "  EPILEPSY: "
            "    60-70% — onset typically infantile spasms or myoclonic; "
            "    INFANTILE SPASMS: common onset seizure type; "
            "    Myoclonic seizures: prominent; "
            "    Lennox-Gastaut spectrum in some; "
            "  INTELLECTUAL DISABILITY: moderate-profound; "
            "  SPEECH: minimal-absent; a minority develop some words; "
            "  MOVEMENT DISORDER: "
            "    Involuntary movements — stereotypies; "
            "    Ataxic gait (those who achieve walking); "
            "    Approximately 50% achieve some independent ambulation; "
            "  BRAIN MRI: "
            "    Often normal; "
            "    Thin corpus callosum (30%); "
            "    Reduced myelination; "
            "    Occasionally simplified gyral pattern; "
            "  TEMPERATURE DYSREGULATION: autonomic instability; "
            "GENOTYPE-PHENOTYPE: "
            "  Frameshift/nonsense: typical PURA syndrome; "
            "  Missense PUR domain: variable — may be milder; "
            "TREATMENT: "
            "  No specific therapy; "
            "  SLEEPINESS: modafinil trialled (case reports); melatonin for circadian rhythm; "
            "  Epilepsy: ACTH (infantile spasms); valproate, levetiracetam baseline; "
            "  Feeding: NG → gastrostomy; "
            "  Respiratory: CPAP if apnoea significant; "
            "TESTING: PURA sequencing; MLPA for deletions; WES commonly finds PURA; "
        ),
        "inheritance": (
            "Autosomal dominant de novo LOF; "
            "PURA 5q31.2; virtually all de novo; "
            "No sex predilection; "
            "Rare familial (parental mosaicism documented); "
        ),
        "disease_category": "PURA Syndrome / AD De Novo NDD",
        "key_mutations": [
            "p.Arg159Trp (missense PUR domain II — common)",
            "p.Arg140Ter (nonsense PUR domain)",
            "p.Gln161Ter (nonsense)",
            "p.Thr206Met (missense PUR domain)",
            "p.Leu239Pro (PUR domain missense)",
            "p.Arg163Gln (missense)",
            "Exon deletions (MLPA / CMA)",
        ],
        "clinical_keys": [
            "EXCESSIVE DAYTIME SLEEPINESS PATHOGNOMONIC — profound hypersomnia from infancy",
            "NEONATAL HYPOTONIA + APNOEA + FEEDING TUBE almost universal — NICU presentation",
            "Infantile spasms common onset seizure type — ACTH first-line",
            "Approximately 50% achieve some walking (better motor than speech)",
            "Modafinil trialled for EDS (case reports only)",
            "PURA interacts with FMR1 — links to fragile X pathway",
            "NG tube → gastrostomy in majority; feeding support essential",
            "Temperature dysregulation — autonomic features",
        ],
    },
    {
        "gene": "HNRNPH2",
        "protein": (
            "HNRNPH2 -- Xq22.1 X-linked-Dominant-de-novo -- 449aa -- Heterogeneous-Nuclear-Ribonucleoprotein-H2-"
            "RNA-Splicing-mRNA-Processing-HNRNPH2-Syndrome-Au-Kline-Syndrome-PROMINENT-FOREHEAD-HYPERTELORISM-BROAD-NASAL-TIP-PATHOGNOMONIC-OMIM-300610"
        ),
        "locus": "Xq22.1",
        "protein_size": (
            "449 aa / 49 kDa (HNRNPH2; heterogeneous nuclear ribonucleoprotein H2; "
            "STRUCTURE: three RNA recognition motifs (RRM1, RRM2, RRM3) + glycine-rich domain; "
            "FUNCTION: "
            "  RNA-binding protein; hnRNP H family; "
            "  Regulates pre-mRNA splicing — binds GGG motifs in exonic/intronic splicing enhancers; "
            "  Regulates alternative splicing of neuronal transcripts; "
            "  Also involved in mRNA stability and poly-A site selection; "
            "  Loss → widespread splicing dysregulation in neurons → NDD; "
            "HNRNPH2 SYNDROME (AU-KLINE SYNDROME): "
            "  PREVALENCE: ultra-rare; ~50 cases reported; X-linked; predominantly females; "
            "  DISTINCTIVE FACIAL FEATURES (PATHOGNOMONIC COMBINATION): "
            "    PROMINENT FOREHEAD: tall broad forehead; "
            "    HYPERTELORISM: widely spaced eyes; "
            "    BROAD NASAL TIP: wide nasal tip; "
            "    Brachycephaly; "
            "    Widely spaced teeth; "
            "    Small chin/micrognathia; "
            "    Epicanthal folds; "
            "    Short palpebral fissures; "
            "  INTELLECTUAL DISABILITY: severe; "
            "  SPEECH: absent or severely limited; "
            "  SHORT STATURE: common; "
            "  SEIZURES: 50-60%; "
            "  BRAIN MRI ABNORMALITIES: "
            "    Thin/hypoplastic corpus callosum (40%); "
            "    Periventricular leukomalacia-like changes; "
            "    Cortical volume reduction; "
            "  LIMB FEATURES: "
            "    Joint hyperlaxity; "
            "    Tapering fingers; "
            "    Short 5th finger; "
            "  CARDIAC: CHD in subset (15-20%); "
            "  BEHAVIOURAL: autistic features; stereotypies; "
            "SEX DISTRIBUTION: "
            "  FEMALES predominantly affected (X-linked dominant); "
            "  Males with hemizygous HNRNPH2 LOF: more severe (limited case reports); "
            "  X-inactivation determines severity in female carriers; "
            "  HNRNPH1 (autosomal paralogue): partially compensates in males; "
            "TESTING: HNRNPH2 sequencing; X-linked → test mother; "
            "  WES often first diagnostic — HNRNPH2 not on older gene panels; "
            "TREATMENT: "
            "  No specific therapy; "
            "  Seizures: levetiracetam, valproate; "
            "  Cardiac: echo at diagnosis; "
            "  Speech-language therapy; physiotherapy; "
        ),
        "inheritance": (
            "X-linked dominant de novo LOF; "
            "HNRNPH2 Xq22.1; virtually all de novo; "
            "Females predominantly affected; males hemizygous: more severe; "
            "HNRNPH1 (autosomal paralogue) partially compensates in males; "
        ),
        "disease_category": "HNRNPH2 Syndrome / Au-Kline Syndrome / X-linked NDD",
        "key_mutations": [
            "p.Pro209Leu (common missense RRM domain — dominant negative)",
            "p.Pro238Leu (missense glycine-rich domain)",
            "p.Pro336His (glycine-rich domain missense)",
            "p.Arg206Gln (RRM missense)",
            "p.Gly294Asp (glycine-rich missense)",
            "p.Arg200Gln (RRM missense)",
            "Exon deletions (MLPA)",
        ],
        "clinical_keys": [
            "PROMINENT FOREHEAD + HYPERTELORISM + BROAD NASAL TIP PATHOGNOMONIC facies",
            "Severe ID with absent/severely limited speech",
            "Corpus callosum anomalies 40% on brain MRI",
            "Females predominantly (X-linked dominant); HNRNPH1 compensates in males",
            "Cardiac CHD 15-20% — echo at diagnosis",
            "WES often first diagnosis — not on older gene panels",
            "X-inactivation studies may help interpret severity in females",
            "Brachycephaly + short stature common",
        ],
    },
]


def _generate_patients_for_gene(gene: str, seed: int) -> list:
    """Generate 40 synthetic patients for a single gene."""
    rng = random.Random(seed)
    patients = []

    gene_params = {
        "MECP2": {
            "iq_range": (20, 55), "epilepsy_rate": 0.80, "speech_absent_rate": 0.65,
            "walk_rate": 0.60, "hand_stereo_rate": 0.95, "breath_irreg_rate": 0.90,
            "regression_rate": 0.98, "scoliosis_rate": 0.80, "dx_months": (18, 48),
            "severity_dist": (0.50, 0.40, 0.10),  # severe, moderate, mild
            "mutations": ["p.Arg306Cys", "p.Arg255Ter", "p.Arg270Ter", "p.Thr158Met",
                          "p.Arg168Ter", "p.Arg133Cys", "Large del exon3-4"],
        },
        "CDKL5": {
            "iq_range": (15, 40), "epilepsy_rate": 0.98, "speech_absent_rate": 0.80,
            "walk_rate": 0.30, "hand_stereo_rate": 0.85, "breath_irreg_rate": 0.10,
            "regression_rate": 0.05, "scoliosis_rate": 0.45, "dx_months": (4, 18),
            "severity_dist": (0.65, 0.30, 0.05),
            "mutations": ["p.Arg59Ter", "p.Arg178Trp", "p.Arg564Ter", "p.Ala40Val",
                          "Exon del", "p.Pro365Arg", "p.Leu220Pro"],
        },
        "FOXG1": {
            "iq_range": (10, 30), "epilepsy_rate": 0.80, "speech_absent_rate": 0.97,
            "walk_rate": 0.15, "hand_stereo_rate": 0.70, "breath_irreg_rate": 0.20,
            "regression_rate": 0.00, "scoliosis_rate": 0.35, "dx_months": (3, 12),
            "severity_dist": (0.80, 0.18, 0.02),
            "mutations": ["p.Trp308Ter", "p.Ala472Val", "p.Arg461Cys", "p.Gly271Asp",
                          "14q12 del", "p.Arg327His", "p.Ser323Leu"],
        },
        "MEF2C": {
            "iq_range": (10, 30), "epilepsy_rate": 0.82, "speech_absent_rate": 0.95,
            "walk_rate": 0.40, "hand_stereo_rate": 0.90, "breath_irreg_rate": 0.05,
            "regression_rate": 0.10, "scoliosis_rate": 0.30, "dx_months": (6, 24),
            "severity_dist": (0.75, 0.20, 0.05),
            "mutations": ["5q14.3 del", "p.Arg3Ter", "p.Gln13Ter", "p.Tyr131Cys",
                          "p.Gln68Ter", "p.Arg187Ter", "Exon del"],
        },
        "WDR45": {
            "iq_range": (25, 60), "epilepsy_rate": 0.85, "speech_absent_rate": 0.55,
            "walk_rate": 0.70, "hand_stereo_rate": 0.40, "breath_irreg_rate": 0.10,
            "regression_rate": 0.90, "scoliosis_rate": 0.25, "dx_months": (24, 120),
            "severity_dist": (0.35, 0.50, 0.15),
            "mutations": ["p.Arg152Ter", "p.Arg152Gln", "p.Tyr38Cys", "p.Asp92Gly",
                          "p.Glu226Ter", "Exon del", "Splice site"],
        },
        "DDX3X": {
            "iq_range": (35, 70), "epilepsy_rate": 0.45, "speech_absent_rate": 0.30,
            "walk_rate": 0.80, "hand_stereo_rate": 0.30, "breath_irreg_rate": 0.05,
            "regression_rate": 0.05, "scoliosis_rate": 0.15, "dx_months": (12, 48),
            "severity_dist": (0.20, 0.50, 0.30),
            "mutations": ["p.Arg326His", "p.Arg376Cys", "p.Glu348Asp", "p.Arg488Ter",
                          "p.Arg534Ter", "p.Thr204Met", "Exon del"],
        },
        "PURA": {
            "iq_range": (15, 45), "epilepsy_rate": 0.65, "speech_absent_rate": 0.85,
            "walk_rate": 0.50, "hand_stereo_rate": 0.50, "breath_irreg_rate": 0.70,
            "regression_rate": 0.10, "scoliosis_rate": 0.20, "dx_months": (2, 18),
            "severity_dist": (0.50, 0.40, 0.10),
            "mutations": ["p.Arg159Trp", "p.Arg140Ter", "p.Gln161Ter", "p.Thr206Met",
                          "p.Leu239Pro", "p.Arg163Gln", "Exon del"],
        },
        "HNRNPH2": {
            "iq_range": (15, 40), "epilepsy_rate": 0.55, "speech_absent_rate": 0.80,
            "walk_rate": 0.55, "hand_stereo_rate": 0.50, "breath_irreg_rate": 0.10,
            "regression_rate": 0.05, "scoliosis_rate": 0.20, "dx_months": (12, 36),
            "severity_dist": (0.55, 0.35, 0.10),
            "mutations": ["p.Pro209Leu", "p.Pro238Leu", "p.Pro336His", "p.Arg206Gln",
                          "p.Gly294Asp", "p.Arg200Gln", "Exon del"],
        },
    }

    p = gene_params.get(gene, gene_params["MECP2"])
    sev_w = p["severity_dist"]

    for i in range(40):
        iq = rng.randint(*p["iq_range"])
        severity = rng.choices(["severe", "moderate", "mild"], weights=sev_w, k=1)[0]
        sex = rng.choice(["F", "F", "F", "M"] if gene in ("MECP2", "CDKL5", "WDR45", "DDX3X", "HNRNPH2") else ["F", "M"])
        dx_mo = rng.randint(*p["dx_months"])
        mutation = rng.choice(p["mutations"])

        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "sex": sex,
            "iq_estimate": iq,
            "severity": severity,
            "age_at_diagnosis_mo": dx_mo,
            "mutation": mutation,
            "epilepsy": rng.random() < p["epilepsy_rate"],
            "speech_absent": rng.random() < p["speech_absent_rate"],
            "independent_walk": rng.random() < p["walk_rate"],
            "hand_stereotypies": rng.random() < p["hand_stereo_rate"],
            "breathing_irregular": rng.random() < p["breath_irreg_rate"],
            "regression": rng.random() < p["regression_rate"],
            "scoliosis": rng.random() < p["scoliosis_rate"],
            "autism_features": rng.random() < 0.55,
            "gastrostomy": rng.random() < (0.30 if gene not in ("FOXG1", "PURA") else 0.60),
            "corpus_callosum_anom": rng.random() < (0.15 if gene in ("MEF2C", "MECP2") else
                                                    0.35 if gene in ("DDX3X", "HNRNPH2") else
                                                    0.40 if gene == "FOXG1" else 0.25),
            "parkinsonism": rng.random() < (0.60 if gene == "WDR45" else 0.02),
            "myeloid_risk": rng.random() < (0.08 if gene == "MEF2C" else 0.00),
            "daytime_sleepiness": rng.random() < (0.90 if gene == "PURA" else 0.10),
            "iron_accumulation_mri": rng.random() < (0.75 if gene == "WDR45" else 0.02),
        })
    return patients


def generate_overview() -> dict:
    """Overview data for Hereditary-Rett-Spectrum-Atlas."""
    return {
        "atlas":          "Hereditary-Rett-Spectrum-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary Rett & Rett-Related Neurodevelopmental Atlas "
            "(MECP2-CDKL5-FOXG1-MEF2C-WDR45-DDX3X-PURA-HNRNPH2)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}–{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "MECP2":   "X-linked dominant LOF (Classic Rett females; HAND STEREOTYPIES + BREATHING IRREGULARITIES PATHOGNOMONIC; Trofinetide FDA 2023; MECP2 Dup → males progressive)",
            "CDKL5":   "X-linked dominant LOF (CDD; EARLY-ONSET SEIZURES BEFORE 5 MONTHS PATHOGNOMONIC; HAND MOUTHING not wringing; NO regression; ketogenic diet early)",
            "FOXG1":   "AD de novo LOF / 14q12 deletion (FOXG1 Syndrome; NO REGRESSION congenital onset; DYSKINESIA + HYPERSALIVATION PATHOGNOMONIC; profound ID; both sexes)",
            "MEF2C":   "AD de novo LOF / 5q14.3 deletion (STEREOTYPIES + HYPERKINESIS + ABSENT SPEECH; MYELOID LEUKEMIA SUSCEPTIBILITY unique; CMA first)",
            "WDR45":   "X-linked dominant de novo (BPAN NBIA5; BIPHASIC childhood ID+seizures → adolescent PARKINSONISM+DEMENTIA; IRON ACCUMULATION GP+SN T2 PATHOGNOMONIC; deferiprone)",
            "DDX3X":   "X-linked dominant de novo (MOST COMMON XL-ID in females; mild-moderate ID; ASD 50%; CC anomalies 30-40%; males DDX3Y compensates; usually AED-responsive)",
            "PURA":    "AD de novo LOF (EXCESSIVE DAYTIME SLEEPINESS PATHOGNOMONIC; NEONATAL HYPOTONIA + APNOEA + FEEDING TUBE universal; infantile spasms; 50% walk)",
            "HNRNPH2": "X-linked dominant de novo (PROMINENT FOREHEAD + HYPERTELORISM + BROAD NASAL TIP PATHOGNOMONIC; severe ID; absent speech; CC anomalies 40%; cardiac CHD 15-20%)",
        },
        "key_clinical_rules": [
            "MECP2 (Rett): HAND STEREOTYPIES (wringing/washing) + BREATHING IRREGULARITIES PATHOGNOMONIC; Trofinetide (DAYBUE) FDA 2023 — PRESCRIBE; AVOID carbamazepine",
            "CDKL5 (CDD): SEIZURES BEFORE 5 MONTHS PATHOGNOMONIC in X-linked NDD girl; HAND MOUTHING (not wringing) — distinguishes from Rett; ketogenic diet early",
            "FOXG1 Syndrome: NO REGRESSION — congenital onset KEY DIFFERENTIATOR from Rett; DYSKINESIA + HYPERSALIVATION pathognomonic; profound ID; both sexes equally",
            "MEF2C: MYELOID LEUKEMIA SUSCEPTIBILITY UNIQUE — ANNUAL FBC MANDATORY; 5q14.3 deletion — CMA FIRST; hyperkinesis distinguishes from Rett",
            "WDR45 (BPAN): BIPHASIC course is PATHOGNOMONIC; T2 GP+SN hypointensity on MRI PATHOGNOMONIC; deferiprone (FIRST-CHOICE chelator); screen adolescents for parkinsonism",
            "DDX3X: MOST COMMON XL-ID in females — test DDX3X in unexplained ID girls; WES often first-line; seizures usually AED-responsive (unlike CDD/MEF2C)",
            "PURA: EXCESSIVE DAYTIME SLEEPINESS PATHOGNOMONIC; neonatal PICU presentation universal; NG/gastrostomy majority; infantile spasms → ACTH",
            "HNRNPH2: Prominent forehead+hypertelorism+broad nasal tip PATHOGNOMONIC facies; cardiac echo at diagnosis (CHD 15-20%); not on older panels — WES first",
            "RTT vs CDD vs FOXG1 key differences: RTT=regression+hand wringing+breathing; CDD=seizures<5mo+hand mouthing+no regression; FOXG1=congenital+dyskinesia+no regression",
            "X-linked syndromes in this atlas (MECP2/CDKL5/WDR45/DDX3X/HNRNPH2): predominantly females; test mother for carrier status; males hemizygous usually more severe/lethal",
            "AUTOSOMAL syndromes (FOXG1/MEF2C/PURA): both sexes equally; de novo virtually all; WES increasingly first-line for all 8 genes",
            "Trofinetide (DAYBUE 2023) MECP2-Rett specific — do NOT use in CDD/FOXG1/MEF2C without specific evidence; mechanism: IGF-1 analogue synaptic restoration",
        ],
        "gene_panel_note": (
            "Rett-spectrum/Rett-related gene panel (2024): MECP2, CDKL5, FOXG1, MEF2C, WDR45, DDX3X, PURA, HNRNPH2; "
            "extended panel includes: NTNG1, IQSEC2, CASK, BCOR, MED17, EP400, HIST1H1E; "
            "Testing strategy: CMA first (5q14.3/14q12 deletions); "
            "then Rett-spectrum gene panel; WES if panel negative; "
            "Trofinetide (FDA 2023) changes MECP2 counselling — treat-as-diagnosed approach; "
            "BPAN (WDR45): SWI MRI mandatory from diagnosis for iron-sensitive sequences; "
            "DDX3X increasingly found on WES in unexplained ID girls"
        ),
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-Rett-Spectrum-Atlas."""
    genes_data = []
    for idx, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        seed = SEED_BASE + idx
        patients = _generate_patients_for_gene(gene, seed)
        n = len(patients)

        epilepsy_n      = sum(1 for p in patients if p["epilepsy"])
        speech_absent_n = sum(1 for p in patients if p["speech_absent"])
        walk_n          = sum(1 for p in patients if p["independent_walk"])
        hand_stereo_n   = sum(1 for p in patients if p["hand_stereotypies"])
        breath_irreg_n  = sum(1 for p in patients if p["breathing_irregular"])
        regression_n    = sum(1 for p in patients if p["regression"])
        scoliosis_n     = sum(1 for p in patients if p["scoliosis"])
        autism_n        = sum(1 for p in patients if p["autism_features"])
        gastrostomy_n   = sum(1 for p in patients if p["gastrostomy"])
        cc_n            = sum(1 for p in patients if p["corpus_callosum_anom"])
        parkinson_n     = sum(1 for p in patients if p["parkinsonism"])
        myeloid_n       = sum(1 for p in patients if p["myeloid_risk"])
        sleepy_n        = sum(1 for p in patients if p["daytime_sleepiness"])
        iron_mri_n      = sum(1 for p in patients if p["iron_accumulation_mri"])
        severe_n        = sum(1 for p in patients if p["severity"] == "severe")
        moderate_n      = sum(1 for p in patients if p["severity"] == "moderate")
        mild_n          = sum(1 for p in patients if p["severity"] == "mild")
        mean_iq         = round(sum(p["iq_estimate"] for p in patients) / n, 1)
        mean_age        = round(sum(p["age_at_diagnosis_mo"] for p in patients) / n, 1)
        mutations_seen  = list({p["mutation"] for p in patients})

        genes_data.append({
            "gene":                 gene,
            "locus":                gene_info["locus"],
            "n_patients":           n,
            "severe_pct":           round(severe_n / n * 100, 1),
            "moderate_pct":         round(moderate_n / n * 100, 1),
            "mild_pct":             round(mild_n / n * 100, 1),
            "mean_iq":              mean_iq,
            "epilepsy_pct":         round(epilepsy_n / n * 100, 1),
            "speech_absent_pct":    round(speech_absent_n / n * 100, 1),
            "independent_walk_pct": round(walk_n / n * 100, 1),
            "hand_stereo_pct":      round(hand_stereo_n / n * 100, 1),
            "breath_irreg_pct":     round(breath_irreg_n / n * 100, 1),
            "regression_pct":       round(regression_n / n * 100, 1),
            "scoliosis_pct":        round(scoliosis_n / n * 100, 1),
            "autism_pct":           round(autism_n / n * 100, 1),
            "gastrostomy_pct":      round(gastrostomy_n / n * 100, 1),
            "corpus_callosum_pct":  round(cc_n / n * 100, 1),
            "parkinsonism_pct":     round(parkinson_n / n * 100, 1),
            "myeloid_risk_pct":     round(myeloid_n / n * 100, 1),
            "daytime_sleepiness_pct": round(sleepy_n / n * 100, 1),
            "iron_accumulation_pct": round(iron_mri_n / n * 100, 1),
            "mean_age_dx_mo":       mean_age,
            "sample_mutations":     mutations_seen[:4],
            "protein":              gene_info["protein"],
            "inheritance":          gene_info["inheritance"][:200],
            "disease_category":     gene_info["disease_category"],
        })
    return {
        "atlas": "Hereditary-Rett-Spectrum-Atlas",
        "count": len(genes_data),
        "genes": genes_data,
    }


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-Rett-Spectrum-Atlas."""
    definitions = [
        {
            "term": "Rett-Spectrum Disorders: Classification, Staging, and Differential Diagnosis Framework",
            "genes": ["MECP2", "CDKL5", "FOXG1", "MEF2C", "WDR45", "DDX3X", "PURA", "HNRNPH2"],
            "definition": (
                "HEREDITARY RETT-SPECTRUM ATLAS — OVERVIEW AND CLASSIFICATION: "
                "CLASSIC RETT SYNDROME (RTT) vs RETT-RELATED DISORDERS: "
                "  Classic Rett (MECP2 females): DIAGNOSTIC CRITERIA = "
                "    (1) Normal first 6 months + (2) Regression of purposeful hand use + speech + "
                "    (3) HAND STEREOTYPIES + (4) Gait abnormalities — ALL 4 required for 'typical' RTT; "
                "  RETT VARIANTS (Rett-related disorders): genes in this atlas; "
                "KEY DIFFERENTIATING FEATURES (most important clinical decision): "
                "  REGRESSION present: MECP2 (classic), WDR45 (second-phase regression); "
                "  REGRESSION absent (congenital): FOXG1, CDKL5 (plateau), MEF2C, PURA; "
                "  SEIZURES FIRST: CDKL5 (before 5 months PATHOGNOMONIC); MEF2C (infantile spasms); PURA (spasms); "
                "  BREATHING IRREGULARITIES: MECP2 (PATHOGNOMONIC); absent in CDKL5/FOXG1; "
                "  HAND STEREOTYPIES: MECP2 (wringing/washing); CDKL5 (mouthing); MEF2C (midline); "
                "  HYPERKINESIS/DYSKINESIA: FOXG1 (PATHOGNOMONIC); MEF2C; absent in MECP2; "
                "  DAYTIME SLEEPINESS: PURA (PATHOGNOMONIC — profound hypersomnia); "
                "  IRON ON MRI: WDR45 (PATHOGNOMONIC — GP+SN T2 hypointensity); "
                "  FACIAL DYSMORPHISM: HNRNPH2 (prominent forehead+hypertelorism+broad nasal tip); "
                "  LEUKEMIA RISK: MEF2C (UNIQUE — myeloid surveillance mandatory); "
                "INHERITANCE PATTERN QUICK GUIDE: "
                "  X-LINKED (predominantly females): MECP2, CDKL5, WDR45, DDX3X, HNRNPH2; "
                "  AUTOSOMAL DOMINANT de novo: FOXG1, MEF2C, PURA; "
                "DIAGNOSTIC ALGORITHM: "
                "  Step 1: CMA — detects 14q12 (FOXG1) + 5q14.3 (MEF2C) deletion ~50%; "
                "  Step 2: Rett-spectrum gene panel (8 genes + extended); "
                "  Step 3: WES/WGS if panel negative; "
                "  Trofinetide (FDA 2023): MECP2-RTT specific — test MECP2 first in females with regression+stereotypies; "
                "TREATMENT SUMMARY BY GENE: "
                "  MECP2: Trofinetide FDA 2023 + standard AEDs; AVOID carbamazepine; "
                "  CDKL5: Ketogenic diet early; vigabatrin (spasms); no specific approved Rx; "
                "  FOXG1: Symptomatic — baclofen (spasticity); drooling management; no specific Rx; "
                "  MEF2C: Annual FBC (myeloid); standard AEDs; "
                "  WDR45: Deferiprone (chelation); levodopa (parkinsonism); "
                "  DDX3X/PURA/HNRNPH2: Supportive; AEDs; rehabilitation"
            ),
        },
        {
            "term": "Classic Rett Syndrome (MECP2) — Diagnostic Criteria, Trofinetide Protocol, and Surveillance",
            "genes": ["MECP2"],
            "definition": (
                "CLASSIC RETT SYNDROME — MECP2 CLINICAL PROTOCOL: "
                "FOUR STAGES (Hagberg 1985 / revised 2010): "
                "  STAGE 1 (6-18 months): Stagnation — subtle; often missed by parents; "
                "    Poor eye contact; reduced play; hypotonia subtle; "
                "  STAGE 2 (1-3 years): REGRESSION — rapid purposeful hand/speech loss; "
                "    HAND STEREOTYPIES EMERGE: hand wringing/washing/squeezing/patting — PATHOGNOMONIC; "
                "    BREATHING IRREGULARITIES: hyperventilation, apnoea, breath-holding — PATHOGNOMONIC; "
                "    Irritability, sleep disturbance; "
                "  STAGE 3 (2-10 years): PLATEAU — relative stability; "
                "    Some improvement in communication; hand stereotypies persist; "
                "    Seizures develop (80%) — may be controlled; "
                "    Some purposeful hand use returns; "
                "  STAGE 4 (post-10): LATE MOTOR DETERIORATION: "
                "    Progressive scoliosis; wheelchair; "
                "    Seizures may reduce; "
                "TROFINETIDE (DAYBUE) — FDA APPROVED March 2023: "
                "  MECHANISM: synthetic IGF-1 tripeptide analogue → restores synaptic function; "
                "  INDICATION: Rett syndrome (Mecp2-confirmed) age ≥2 years; "
                "  DOSING: weight-based twice daily oral solution; "
                "    <9 kg: 100 mg/dose BD; 9-<20 kg: 200 mg/dose BD; "
                "    20-<35 kg: 300 mg/dose BD; ≥35 kg: 400 mg/dose BD; "
                "  KEY SIDE EFFECTS: diarrhoea (80%), vomiting — manage with loperamide/ondansetron; "
                "  EVIDENCE: LAVENDER trial — significant improvement in behaviour + motor; "
                "SURVEILLANCE PROTOCOL: "
                "  ECG: ANNUALLY — long QT risk; "
                "  Spine X-ray: ANNUALLY from diagnosis (scoliosis 80%); "
                "  Cardiac: QT-prolonging drug list (avoid together with Rett); "
                "  Neurological: seizure diary; AED review; "
                "  Respiratory: SpO2 monitoring during apnoeic episodes; "
                "  Nutrition: gastrostomy if swallow fails; "
                "AED GUIDANCE: "
                "  First-line: valproate, levetiracetam, lamotrigine; "
                "  AVOID: carbamazepine (worsens autonomic instability, may worsen apnoea); "
                "  Avoid: phenytoin (long QT interaction); "
                "MECP2 DUPLICATION SYNDROME (males): "
                "  Progressive spastic quadriplegia; recurrent respiratory infections; "
                "  Annual respiratory function + infection surveillance; "
                "  IVIG: documented benefit for recurrent infections in MECP2 duplication; "
                "  Female MECP2 duplication carriers: usually asymptomatic"
            ),
        },
        {
            "term": "CDKL5 Deficiency Disorder vs Classic Rett — Key Differential and Ketogenic Diet Protocol",
            "genes": ["CDKL5", "MECP2"],
            "definition": (
                "CDKL5 DEFICIENCY DISORDER vs CLASSIC RETT — DIFFERENTIAL DIAGNOSIS: "
                "FEATURE COMPARISON TABLE: "
                "  SEIZURE ONSET: "
                "    CDD: BEFORE 5 MONTHS (PATHOGNOMONIC) — 80% before 5 months; "
                "    RTT: Stage 3 (2-10 years) — AFTER regression; "
                "  REGRESSION: "
                "    CDD: NO — developmental plateau from outset; "
                "    RTT: YES — purposeful regression Stage 2 (1-3 years); "
                "  HAND STEREOTYPIES: "
                "    CDD: HAND MOUTHING (not wringing) — hand to mouth; "
                "    RTT: HAND WRINGING/WASHING — midline hand stereotypies; "
                "  BREATHING IRREGULARITIES: "
                "    CDD: ABSENT — no apnoea/hyperventilation; "
                "    RTT: PATHOGNOMONIC — apnoea, hyperventilation; "
                "  WALKING: "
                "    CDD: Majority NEVER walk independently; "
                "    RTT: 50-70% achieve walking (lost Stage 4); "
                "  SOCIAL INTERACTION: "
                "    CDD: More preserved eye contact/smiling; "
                "    RTT: Social withdrawal during Stage 2; "
                "  EEG: "
                "    CDD: High-amplitude multifocal spikes; hypsarrhythmia; "
                "    RTT: Slow background; centrotemporal spikes; "
                "CDKL5 SEIZURE MANAGEMENT: "
                "  KETOGENIC DIET: RECOMMEND EARLY — documented 50%+ seizure reduction; "
                "    Implement by 12 months of age ideally; "
                "    4:1 or 3:1 ratio; monitor ketones; "
                "  ACTH/prednisolone: infantile spasms (standard IS protocol); "
                "  Vigabatrin: reasonable first-line for spasms; "
                "  Cannabidiol (Epidiolex): FDA approved for refractory epilepsy — growing CDD evidence; "
                "  Sodium valproate + clobazam: common combination baseline; "
                "  Fenfluramine: emerging evidence in refractory epilepsy; "
                "  AVOID: phenytoin may worsen in many CDD cases (anecdotal); "
                "TROFINETIDE IN CDD: "
                "  NOT approved for CDD (approved MECP2-Rett only); "
                "  Phase 3 trial investigating CDKL5 — results pending; "
                "GENOTYPE: "
                "  N-terminal kinase domain variants: most severe (earlier/more seizures); "
                "  C-terminal variants: milder phenotype possible; "
                "  X-inactivation studies: skewing toward CDKL5-mutant allele → more severe"
            ),
        },
        {
            "term": "FOXG1 Syndrome and MEF2C Haploinsufficiency — Congenital Rett Variants Without Regression",
            "genes": ["FOXG1", "MEF2C"],
            "definition": (
                "FOXG1 SYNDROME + MEF2C HAPLOINSUFFICIENCY — NO-REGRESSION CONGENITAL VARIANTS: "
                "FOXG1 SYNDROME (Congenital Rett variant): "
                "  KEY DISTINCTION FROM CLASSIC RTT: "
                "    NO NORMAL DEVELOPMENTAL PERIOD — abnormal from birth; "
                "    NO REGRESSION — cannot regress from milestones never reached; "
                "    This is the MOST IMPORTANT clinical differentiator; "
                "  PATHOGNOMONIC TRIAD: "
                "    DYSKINESIA: choreoathetosis + dystonia — involuntary; often severe; "
                "    HYPERSALIVATION: drooling, swallowing difficulty; "
                "    ABSENT SPEECH: virtually universal; "
                "  BRAIN MRI FINDINGS: "
                "    Simplified gyral pattern (pachygyria): predominant frontal; "
                "    Reduced frontal lobe volume; "
                "    Hypoplastic corpus callosum (genu especially); "
                "    Delayed myelination; "
                "  SEX: EQUAL — unlike MECP2/CDKL5; males+females equally affected; "
                "  TREATMENT: "
                "    Dyskinesia: trihexyphenidyl; baclofen; DBS investigational; "
                "    Drooling: scopolamine patch 0.5mg; glycopyrrolate 0.1mg/kg BD; "
                "       Botulinum toxin parotid/submandibular glands; "
                "    Feeds: NG → gastrostomy majority; "
                "    AEDs: levetiracetam, valproate, clonazepam; "
                "MEF2C HAPLOINSUFFICIENCY: "
                "  5q14.3 DELETION (~50%): CMA FIRST — detects chromosomal deletion; "
                "  UNIQUE FEATURE — MYELOID LEUKEMIA SUSCEPTIBILITY: "
                "    MEF2C is master myeloid transcription factor; "
                "    AML documented in several cases; "
                "    ANNUAL FBC WITH DIFFERENTIAL — mandatory; "
                "    Threshold to investigate cytopenias: LOW; "
                "  HYPERKINESIS: excess purposeless movements — distinguishes from Rett; "
                "  TREATMENT: "
                "    Annual FBC — leukemia surveillance; "
                "    AEDs: vigabatrin (spasms); valproate/levetiracetam; "
                "    Ketogenic diet: reasonable (epilepsy drug-resistant); "
                "FOXG1 vs MEF2C DIFFERENTIAL: "
                "  FOXG1: dyskinesia + hypersalivation + profound ID; frontal gyral simplification MRI; "
                "  MEF2C: hyperkinesis + MYELOID RISK + 5q14.3 deletion CMA; "
                "  Both: congenital onset; no regression; absent speech; profound ID"
            ),
        },
        {
            "term": "WDR45 BPAN — Biphasic Course, Iron Chelation Protocol, and NBIA Neuroimaging",
            "genes": ["WDR45"],
            "definition": (
                "BETA-PROPELLER PROTEIN-ASSOCIATED NEURODEGENERATION (BPAN) — WDR45 PROTOCOL: "
                "BIPHASIC CLINICAL COURSE (PATHOGNOMONIC FOR BPAN): "
                "  PHASE 1 — CHILDHOOD (age 0-15): "
                "    Global developmental delay — intellectual disability; "
                "    Epilepsy — various types (infantile spasms, absence, focal, myoclonic); "
                "    Autistic features; stereotypies; "
                "    BRAIN MRI: may appear NORMAL or subtle — DO NOT REASSURE; "
                "    Diagnosis often delayed until second phase develops; "
                "  PHASE 2 — ADOLESCENCE/YOUNG ADULTHOOD (age 15-25): "
                "    SUDDEN ONSET PARKINSONISM: tremor, rigidity, bradykinesia; "
                "    DYSTONIA: severe; may be painful; focal → generalized; "
                "    RAPID COGNITIVE DECLINE → FRANK DEMENTIA; "
                "    Phase 2 onset can be precipitated by stress/illness; "
                "NEUROIMAGING — PATHOGNOMONIC: "
                "  MRI T2-WEIGHTED: HYPOINTENSITY GLOBUS PALLIDUS + SUBSTANTIA NIGRA "
                "    = iron deposition PATHOGNOMONIC for NBIA; "
                "  MRI T1-WEIGHTED: T1 HYPERINTENSE HALO peri-globus-pallidus (BPAN specific); "
                "  SWI SEQUENCES: MANDATORY — best sensitivity for iron detection; "
                "  BPAN RADIOLOGICAL DDx (vs other NBIA): "
                "    PKAN (PANK2): 'eye of the tiger' GP T2 hypointensity + T2 hyperintense centre; "
                "    BPAN (WDR45): GP + SN hypointensity; T1 halo periGP; "
                "    Plan-A: SCAN EARLY — before phase 2 if suspected; "
                "IRON CHELATION — DEFERIPRONE PROTOCOL: "
                "  Deferiprone: FIRST-CHOICE brain-penetrant chelator for BPAN; "
                "  Dose: 25-33 mg/kg/day in 3 divided doses (up to 75-100 mg/kg/day); "
                "  MONITORING: FBC weekly first 6 months (agranulocytosis risk); monthly ongoing; "
                "  LIVER FUNCTION: 3-monthly; "
                "  EVIDENCE: modest MRI iron reduction + clinical stabilisation (not curative); "
                "  CLINICAL TRIAL: DeferN — completed; evidence of slowing progression; "
                "PARKINSONISM MANAGEMENT: "
                "  Levodopa/carbidopa: reasonable first-line — partial response; "
                "  Pramipexole/ropinirole: dopamine agonists; "
                "  DBS (deep brain stimulation): investigational in BPAN; "
                "  Anticholinergics: trihexyphenidyl for dystonia; "
                "SURVEILLANCE PROTOCOL: "
                "  Brain MRI with SWI: every 2-3 years from diagnosis; "
                "  Proactive review for parkinsonism in adolescence: tremor, gait change; "
                "  FBC: weekly (deferiprone first 6 months) → monthly; "
                "  Neuropsychology: annual cognitive assessment tracking"
            ),
        },
    ]
    return {
        "atlas": "Hereditary-Rett-Spectrum-Atlas",
        "count": len(definitions),
        "definitions": definitions,
    }
