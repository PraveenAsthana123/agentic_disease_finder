#!/usr/bin/env python3
"""SCA-Atlas — Complete 8-Gene Hereditary Ataxia / Spinocerebellar Ataxia Atlas
FXN    (Friedreich Ataxia; AR; GAA repeat; most common hereditary ataxia; frataxin; 9q21.11) ·
ATXN1  (SCA1; AD; CAG polyQ; 6p22.3; olivopontocerebellar atrophy; abnormal saccades) ·
ATXN2  (SCA2; AD; CAG polyQ; 12q24.12; slow saccades PATHOGNOMONIC; ALS risk ATXN2 intermediate) ·
ATXN3  (SCA3/MJD; AD; CAG polyQ; 14q32.12; most common SCA worldwide; Azorean founder) ·
CACNA1A (SCA6/EA2; AD; CAG polyQ mild/channel mutation EA2; 19p13.13; episodic+progressive) ·
ATXN7  (SCA7; AD; CAG polyQ; 3p14.1; visual loss/retinal dystrophy PATHOGNOMONIC among SCAs) ·
TBP    (SCA17; AD; CAG/CAA polyQ; 6q27; widest phenotype; Huntington DDx) ·
RFC1   (CANVAS; AR; AAGGG pentanucleotide repeat; 4p14; cerebellar+neuropathy+vestibular areflexia)
320-patient aggregate cohort (8 × 40, seeds 990–997)

Hereditary Ataxia — Key Neurological Principles:
  - HEREDITARY ATAXIAS: heterogeneous group of progressive cerebellar disorders.
    Classified by inheritance: AR (Friedreich = most common), AD (SCAs = dominant spinocerebellar),
    X-linked (rare), and mitochondrial.
  - REPEAT EXPANSION DISEASES: 6 of 8 genes in this atlas are CAG/repeat expansion diseases.
    CAG repeats (polyglutamine) — ATXN1, ATXN2, ATXN3, CACNA1A, ATXN7, TBP.
    GAA repeat (intronic; loss of frataxin expression) — FXN (Friedreich).
    AAGGG pentanucleotide repeat (intronic) — RFC1 (CANVAS).
    Pathogenesis differs: polyQ (gain-of-toxic-function), GAA (silencing/loss), AAGGG (loss/RNA toxicity).
  - ANTICIPATION: most polyQ SCAs show anticipation (earlier/more severe with each generation,
    especially when inherited from father — paternal transmission carries higher CAG expansion risk).
    Exception: SCA6 — very limited anticipation (CAG range is narrow: 20-33 repeats).
    Friedreich: no anticipation (GAA repeat alleles are unstable but often contract on transmission).
  - KEY CLINICAL DIFFERENTIATORS:
    ATXN2 (SCA2): SLOW SACCADES — pathognomonic; among all SCAs, only SCA2 reliably has this.
      Slowed saccadic eye movements on bedside exam is the single most discriminating sign.
      Also intermediate ATXN2 repeats (27-32 CAG) dramatically increase ALS risk.
    ATXN7 (SCA7): RETINAL DYSTROPHY + ATAXIA — pathognomonic combination.
      Macular degeneration causing cone-rod dystrophy precedes or coincides with ataxia.
      Visual loss BEFORE ataxia in some patients. No other polyQ SCA affects vision.
    TBP (SCA17): WIDEST PHENOTYPE — resembles Huntington's (chorea, psychiatric, dementia) +
      ataxia; most common when SCA panel returns negative and Huntington-like picture exists.
    RFC1 (CANVAS): TRIPLE COMBINATION (cerebellar ataxia + sensorimotor neuropathy + bilateral
      vestibular areflexia) — very specific; chronic cough in >60% (sensory neuropathy of vagus).
      AAGGG repeat biallelic is highly prevalent (carrier frequency ~1:80 in Europeans).
  - FRIEDREICH ATAXIA (FXN): AR; GAA trinucleotide repeat in intron 1 of FXN gene → frataxin
    deficiency → mitochondrial iron dysregulation → oxidative damage → DRG/spinal cord/heart.
    UNIQUE among hereditary ataxias: (a) AR (not AD), (b) non-neurological features dominate
    prognosis — HYPERTROPHIC CARDIOMYOPATHY causes death in 60-70% without treatment,
    (c) OMAVELOXOLONE (Skyclarys, FDA 2023) — first approved treatment for Friedreich ataxia.
    mFARS score improvement. (d) absent lower limb reflexes (dorsal column/DRG loss early).
  - TREATMENT LANDSCAPE (2026): No disease-modifying treatment for polyQ SCAs; rehabilitation
    and supportive care only. EXCEPTIONS:
    SCA6/EA2 (CACNA1A): acetazolamide for EPISODIC ATAXIA 2 component (carbonic anhydrase inhibitor).
    SCA17 (TBP): no specific treatment.
    RFC1-CANVAS: no treatment; physical rehabilitation for vestibular loss.
    FRIEDREICH: omaveloxolone (FDA Feb 2023) — Nrf2 activator; mFARS improvement ~2.1 points.

COHORT: 8 × 40 = 320 patient slots (seeds 990–997; gene-specific seeds)
"""

import random

SEED_BASE = 990

SCA_GENES = [
    # ── FXN — Friedreich Ataxia ─────────────────────────────────────────
    {
        "gene": "FXN", "protein": "Frataxin",
        "alias": "Friedreich Ataxia (FRDA; OMIM #229300); AR; most common hereditary ataxia; GAA repeat; hypertrophic cardiomyopathy; omaveloxolone FDA 2023",
        "aa": "210 aa", "kDa": "23 kDa",
        "gene_class": (
            "Mitochondrial iron-binding protein; localises to mitochondrial matrix; "
            "functions in iron-sulfur cluster (ISC) assembly and mitochondrial iron homeostasis. "
            "Frataxin deficiency → mitochondrial iron accumulation → Fenton reaction → "
            "reactive oxygen species → oxidative damage to high-energy tissues (DRG, spinocerebellar "
            "tracts, dentate nucleus, myocardium). "
            "GENETIC MECHANISM: ~96% of cases have biallelic GAA trinucleotide repeat expansion "
            "in intron 1 of FXN (normal: ≤33 GAA; carrier: 33-65; pathogenic: 66-1300). "
            "Longer GAA repeats → less frataxin → earlier onset and more severe cardiomyopathy. "
            "~4%: compound heterozygote (one GAA expansion + one point mutation/deletion). "
            "9q21.11; OMIM gene 606829."
        ),
        "ataxia_group": "Autosomal Recessive Ataxia",
        "subtype": "Friedreich Ataxia — Frataxin Deficiency",
        "locus": "9q21.11", "omim_gene": 606829, "omim_disease": 229300,
        "inheritance": "Autosomal Recessive (AR). Biallelic GAA repeat expansions in 96%. Carrier frequency ~1:100 European. Both sexes equally affected.",
        "seed_offset": 0,
        "onset_range_y": (5.0, 25.0),
        "gender": "both",
        "severity_weights": [0.15, 0.40, 0.45],
        "repeat_type": "GAA (intron 1, both alleles)",
        "anticipation": False,
        "phenotype": (
            "GAIT ATAXIA: onset mean ~15 years (range 5-25); progressive, wheelchair ~10 years "
            "after onset. ABSENT LOWER LIMB DEEP TENDON REFLEXES (DRG loss — pathognomonic early sign). "
            "POSITIVE BABINSKI / EXTENSOR PLANTAR response (UMN tract involvement). "
            "Dysarthria, dysphagia. Sensory neuropathy (vibration + proprioception loss, "
            "spinothalamic relatively spared). Scoliosis (structural; early sign). "
            "Pes cavus + pes equinovarus (foot deformity). "
            "HYPERTROPHIC CARDIOMYOPATHY (HCM): present in >80%; major cause of death (arrhythmia, "
            "heart failure). HCM onset may precede/follow neurological symptoms. "
            "Diabetes mellitus: 10-15% (pancreatic islet iron toxicity). "
            "Hearing loss: 10-20%. Visual loss: optic neuropathy (rare, later). "
            "ATYPICAL: SCA phenotype, late-onset FRDA (LOFA, onset >25y, slower)."
        ),
        "disease": (
            "Most common hereditary ataxia: prevalence ~1:50,000-1:100,000 European. "
            "GAA repeat size correlates with age-of-onset, wheelchair age, and HCM severity "
            "(allele 1 — shorter allele — determines phenotype). "
            "OMAVELOXOLONE (Skyclarys, Reata/Biogen, FDA Feb 2023): first approved disease-modifying "
            "treatment for Friedreich ataxia. Mechanism: Nrf2 pathway activator → antioxidant "
            "protection → reduced mitochondrial oxidative stress. Pivotal MOXIe trial: mFARS "
            "improvement 2.1 points vs placebo (p<0.001). Age ≥16; dose 150 mg daily. "
            "Cardiac monitoring: annual echo + Holter mandatory. ICD for sustained VT. "
            "Physiotherapy and speech therapy are essential."
        ),
        "treatment_options": [
            "Omaveloxolone (Skyclarys) 150 mg daily (FDA 2023 — ≥16y)",
            "Physiotherapy + gait aids + orthoses",
            "Cardiology follow-up + echo annually (HCM management)",
            "Scoliosis: physiotherapy; surgical if Cobb >40°",
            "Diabetes: metformin/insulin as needed",
        ],
        "outcome_options": [
            "Progressive — wheelchair ~10y; omaveloxolone partially slows",
            "Progressive — cardiomyopathy (ICD inserted) + ataxia",
            "Progressive — diabetes + severe HCM; ataxia moderate",
            "Late-onset FRDA — slower progression; independent walking >20y",
            "Severe — rapid HCM; early death from arrhythmia",
        ],
        "gaa_range": (66, 1300),
        "has_cardiac": True,
        "has_diabetes": True,
        "visual_loss": False,
        "vestibular": False,
    },

    # ── ATXN1 — SCA1 ────────────────────────────────────────────────────
    {
        "gene": "ATXN1", "protein": "Ataxin-1",
        "alias": "SCA1 (Spinocerebellar Ataxia Type 1; OMIM #164400); AD; CAG polyQ; olivopontocerebellar atrophy; abnormal saccades; 6p22.3",
        "aa": "816 aa", "kDa": "87 kDa",
        "gene_class": (
            "Nuclear protein; contains AXH domain (for protein-protein interactions and "
            "RNA binding); normally shuttles between nucleus and cytoplasm. Ataxin-1 is involved "
            "in transcriptional regulation. Expanded polyQ (>39 CAG) → intranuclear inclusion "
            "bodies → nuclear dysfunction → Purkinje cell degeneration → olivopontocerebellar atrophy. "
            "Normal: ≤35 CAG (including interruptions); pathogenic: ≥39 CAG (uninterrupted). "
            "Meiotic instability (especially paternal transmission) → anticipation. "
            "Penetrance: 100% above 45 CAG repeats; reduced (95%) 39-44 CAG. "
            "6p22.3; OMIM gene 601556."
        ),
        "ataxia_group": "Autosomal Dominant Polyglutamine SCA",
        "subtype": "SCA1 — Spinocerebellar Ataxia Type 1",
        "locus": "6p22.3", "omim_gene": 601556, "omim_disease": 164400,
        "inheritance": "Autosomal Dominant (AD). CAG repeat expansion in ATXN1. Anticipation present (paternal > maternal transmission). De novo expansion from alleles 35-38 CAG.",
        "seed_offset": 1,
        "onset_range_y": (25.0, 55.0),
        "gender": "both",
        "severity_weights": [0.20, 0.45, 0.35],
        "repeat_type": "CAG (exon 8, polyQ)",
        "anticipation": True,
        "phenotype": (
            "Onset: mean 30-40 years (range 20-65, earlier with larger repeats). "
            "GAIT AND LIMB ATAXIA (early; progressive). "
            "OCULOMOTOR ABNORMALITIES: nystagmus, hypometric saccades (fast eye movements "
            "overshoot corrections; abnormal but not as markedly slow as SCA2). "
            "DYSARTHRIA: prominent, often early. "
            "UPPER LIMB INVOLVEMENT: earlier than many other SCAs. "
            "EXTRAPYRAMIDAL FEATURES: dystonia, spasticity (especially lower limbs). "
            "AMYOTROPHY/PERIPHERAL NEUROPATHY: rare but reported. "
            "OLIVOPONTOCEREBELLAR ATROPHY: characteristic MRI pattern (cerebellar + pons + "
            "inferior olives atrophied; 'hot cross bun' sign in pons). "
            "DISEASE COURSE: mean survival 15-20 years from onset; death usually "
            "from aspiration pneumonia or respiratory failure. "
            "NO cardiac involvement (distinguished from FRDA)."
        ),
        "disease": (
            "SCA1 prevalence: ~1-2:100,000 (varies by region). "
            "CAG repeat size inversely correlates with age of onset (1 CAG repeat → ~1.5 years earlier). "
            "No disease-modifying treatment (2026). "
            "Genetic counselling: 50% inheritance risk per child. "
            "Predictive testing (presymptomatic): genetic counselling mandatory first. "
            "Gene silencing approaches in preclinical development."
        ),
        "treatment_options": [
            "Physiotherapy (gait rehabilitation, balance training)",
            "Speech therapy (dysarthria; dysphagia risk reduction)",
            "Occupational therapy + adaptive equipment",
            "Botulinum toxin (dystonia component if present)",
            "No disease-modifying treatment (2026)",
        ],
        "outcome_options": [
            "Progressive — gait aid 10y; wheelchair ~15y; dysarthria severe",
            "Progressive — early onset large repeat; rapid deterioration",
            "Moderate — mid-range repeat; slower progression",
            "Progressive — dysphagia leading to PEG 15-20y after onset",
            "Progressive — respiratory failure; ventilatory support",
        ],
        "cag_range": (39, 82),
        "has_cardiac": False,
        "has_diabetes": False,
        "visual_loss": False,
        "vestibular": False,
    },

    # ── ATXN2 — SCA2 ────────────────────────────────────────────────────
    {
        "gene": "ATXN2", "protein": "Ataxin-2",
        "alias": "SCA2 (Spinocerebellar Ataxia Type 2; OMIM #183090); AD; CAG polyQ; SLOW SACCADES PATHOGNOMONIC; ALS risk at 27-32 CAG intermediate; 12q24.12",
        "aa": "1313 aa", "kDa": "140 kDa",
        "gene_class": (
            "Large cytoplasmic RNA-binding protein; contains PAM2 motif (PABP-interacting) and "
            "LSm/LSmAD domains (RNA metabolism). Interacts with TDP-43 (ALS risk). "
            "Normal: ≤31 CAG; intermediate: 27-31 (or 31-32 — variable) — MAJOR ALS MODIFIER; "
            "pathogenic: ≥33 CAG. INTERMEDIATE ATXN2 REPEATS (27-32 CAG): dramatically increased "
            "ALS risk — found in ~5% of familial ALS and ~1% of sporadic ALS; proposed ALS modifier "
            "via TDP-43 aggregation facilitation. Antisense oligonucleotide (ASO) targeting ATXN2 "
            "in clinical trials for ALS (Ionis). 12q24.12; OMIM gene 601517."
        ),
        "ataxia_group": "Autosomal Dominant Polyglutamine SCA",
        "subtype": "SCA2 — Spinocerebellar Ataxia Type 2",
        "locus": "12q24.12", "omim_gene": 601517, "omim_disease": 183090,
        "inheritance": "Autosomal Dominant (AD). CAG repeat in ATXN2. Anticipation (especially paternal). Intermediate repeats (27-32) associated with ALS (not SCA); this group usually de novo or non-affected parents.",
        "seed_offset": 2,
        "onset_range_y": (25.0, 60.0),
        "gender": "both",
        "severity_weights": [0.20, 0.45, 0.35],
        "repeat_type": "CAG (exon 1, polyQ)",
        "anticipation": True,
        "phenotype": (
            "Onset: mean 30-40 years. "
            "GAIT ATAXIA (early, prominent). "
            "SLOW SACCADES — PATHOGNOMONIC: markedly slowed saccadic eye movements "
            "(slowest of all SCAs); latency prolonged and velocity reduced. "
            "This is the single most discriminating bedside sign for SCA2 vs other SCAs. "
            "HYPOREFLEXIA / AREFLEXIA (peripheral neuropathy component — absent or reduced DTRs "
            "distinguishes from SCA1 which has hyperreflexia). "
            "TREMOR: action tremor early. "
            "FASCICULATIONS: perioral and tongue fasciculations (bulbar motor neuron involvement). "
            "PARKINSONISM: in some patients, especially Cuban founder allele families (L-DOPA responsive). "
            "DEMENTIA: 25-30% in late stages. "
            "DYSTONIA: occasional. "
            "MRI: cerebellar + pontine atrophy; cortical atrophy in advanced stages. "
            "EARLY ONSET (<10y) with very large repeats (>60 CAG) — severe multisystem disease. "
            "ALS VARIANT: intermediate 27-32 CAG → ALS phenotype, NOT SCA."
        ),
        "disease": (
            "SCA2 prevalence: ~1-2:100,000 globally; higher in Cuba (Holguín province founder) "
            "and India (Tamil Nadu — p.Lys166Arg or repeat variants). "
            "Cuban founder mutation: large repeat alleles; parkinsonism phenotype prominent. "
            "ATXN2 ASO (IONIS-ATXN2Rx): in Phase 1/2 ALS trial (targets both ALS + SCA2). "
            "Parkinsonism subtype may respond to L-DOPA (trial warranted). "
            "No approved disease-modifying treatment for SCA2 ataxia."
        ),
        "treatment_options": [
            "Physiotherapy + speech therapy + occupational therapy",
            "L-DOPA trial (parkinsonism phenotype — Cuban/Indian families)",
            "Clonazepam (tremor; limited evidence)",
            "No SCA2-specific disease-modifying Rx (2026)",
            "ATXN2 ASO in ALS trials (not yet SCA2 approved)",
        ],
        "outcome_options": [
            "Progressive — slow saccades + ataxia; wheelchair 12-15y",
            "Progressive — parkinsonism variant; L-DOPA partial response",
            "Progressive — peripheral neuropathy + ataxia; areflexia",
            "Severe — very large repeat; bulbar failure early",
            "Moderate — late onset; slower progression",
        ],
        "cag_range": (33, 77),
        "has_cardiac": False,
        "has_diabetes": False,
        "visual_loss": False,
        "vestibular": False,
    },

    # ── ATXN3 — SCA3 / MJD ──────────────────────────────────────────────
    {
        "gene": "ATXN3", "protein": "Ataxin-3 (Josephin)",
        "alias": "SCA3/MJD (Machado-Joseph Disease; OMIM #109150); AD; CAG polyQ; MOST COMMON SCA worldwide; Azorean founder; bulging eyes + lid retraction; 14q32.12",
        "aa": "361 aa", "kDa": "42 kDa",
        "gene_class": (
            "Deubiquitinase (DUB); contains Josephin domain (papain-like cysteine protease) "
            "with two ubiquitin-interacting motifs (UIMs). Ataxin-3 edits polyubiquitin chains "
            "and is involved in proteasome-mediated protein degradation quality control. "
            "Expanded polyQ (>55 CAG) → misfolding → intranuclear/cytoplasmic inclusions → "
            "ubiquitin-proteasome system overwhelmed → widespread neuronal degeneration. "
            "Normal: ≤44 CAG; pathogenic: ≥55 CAG. 45-54: reduced penetrance / pre-mutation zone. "
            "14q32.12; OMIM gene 607047. "
            "Azorean founder mutation: most Portuguese/Azorean SCA3 trace to 14th-century mutation. "
            "Most common SCA worldwide (40% of all SCA in many registries)."
        ),
        "ataxia_group": "Autosomal Dominant Polyglutamine SCA",
        "subtype": "SCA3/MJD — Machado-Joseph Disease",
        "locus": "14q32.12", "omim_gene": 607047, "omim_disease": 109150,
        "inheritance": "Autosomal Dominant (AD). CAG repeat expansion in ATXN3. Most common SCA worldwide. Azorean/Portuguese founder effect. Anticipation (paternal > maternal).",
        "seed_offset": 3,
        "onset_range_y": (20.0, 60.0),
        "gender": "both",
        "severity_weights": [0.20, 0.45, 0.35],
        "repeat_type": "CAG (exon 10, polyQ)",
        "anticipation": True,
        "phenotype": (
            "GAIT ATAXIA (progressive). "
            "OCULOMOTOR FEATURES: nystagmus; saccadic pursuit; GAZE-EVOKED NYSTAGMUS. "
            "PATHOGNOMONIC CLINICAL SIGN: BULGING EYES + LID RETRACTION (Stellwag's sign analogue) "
            "— distinctive facial appearance in SCA3; sclera visible above iris. "
            "EYELID RETRACTION and REDUCED BLINKING (staring appearance). "
            "DYSARTHRIA (prominent, early). "
            "DYSTONIA (in younger-onset, larger CAG): hemidystonia → generalised dystonia "
            "preceding ataxia (TYPE I phenotype). "
            "PYRAMIDAL SIGNS (spasticity, hyperreflexia — TYPE II). "
            "PERIPHERAL NEUROPATHY (areflexia, amyotrophy — TYPE III — older onset, smaller repeat). "
            "PARKINSONISM (TYPE IV — older onset). "
            "RESTLESS LEGS SYNDROME: very common in SCA3 (60-80%) — important quality of life issue. "
            "No retinal/visual involvement (DDx SCA7)."
        ),
        "disease": (
            "Most common SCA worldwide (~40% of all dominant SCAs). "
            "Especially prevalent: Portugal/Azores (40% of all neurological hereditary disease), "
            "Brazil, Germany, China, Japan. "
            "CAG repeat size correlates with phenotype subtype and age of onset. "
            "4 clinical subtypes (Types I-IV) originally described but substantial overlap. "
            "RLS: treated with dopamine agonists (pramipexole, ropinirole). "
            "No approved disease-modifying treatment. ASO and gene silencing in development."
        ),
        "treatment_options": [
            "Physiotherapy + speech therapy + occupational therapy",
            "Dopamine agonists for RLS (pramipexole — very common symptom)",
            "Botulinum toxin for dystonia (Type I)",
            "Baclofen / tizanidine for spasticity (Type II)",
            "No disease-modifying Rx for SCA3 (2026)",
        ],
        "outcome_options": [
            "Progressive — gait aid 10-12y; bulging eyes; RLS managed",
            "Progressive — Type I; dystonia + ataxia; early onset",
            "Progressive — Type III; neuropathy + ataxia; areflexia",
            "Progressive — Type II; spasticity + ataxia; later onset",
            "Moderate — small repeat; late onset; slower decline",
        ],
        "cag_range": (55, 87),
        "has_cardiac": False,
        "has_diabetes": False,
        "visual_loss": False,
        "vestibular": False,
    },

    # ── CACNA1A — SCA6 / EA2 ─────────────────────────────────────────────
    {
        "gene": "CACNA1A", "protein": "Voltage-Gated Calcium Channel Subunit Alpha1-A (P/Q-type Cav2.1)",
        "alias": "SCA6/EA2 (Spinocerebellar Ataxia Type 6; Episodic Ataxia Type 2; OMIM #183086/#108500); AD; CAG polyQ SCA6; channel mutation EA2; acetazolamide-responsive EA2; 19p13.13",
        "aa": "2368 aa", "kDa": "255 kDa",
        "gene_class": (
            "P/Q-type voltage-gated calcium channel (Cav2.1); alpha1-A pore-forming subunit. "
            "Expressed ubiquitously but CRITICAL in Purkinje cells and cerebellar granule cells. "
            "THREE distinct diseases from CACNA1A variants: "
            "(1) SCA6: CAG repeat expansion (>19 CAG) in 3'UTR-like region → polyQ → Purkinje cell "
            "degeneration. Normal ≤18 CAG; pathogenic ≥20. NARROW range (20-33) — very limited "
            "anticipation. Pure cerebellar; late onset. "
            "(2) EPISODIC ATAXIA TYPE 2 (EA2): heterozygous loss-of-function point mutations/truncations "
            "→ interictal cerebellar signs + episodic attacks. ACETAZOLAMIDE dramatically effective. "
            "(3) FAMILIAL HEMIPLEGIC MIGRAINE TYPE 1 (FHM1): gain-of-function missense → FHM. "
            "19p13.13; OMIM gene 601011."
        ),
        "ataxia_group": "Autosomal Dominant Calcium Channel Ataxia (SCA6/EA2)",
        "subtype": "SCA6/EA2 — CACNA1A Calcium Channelopathy",
        "locus": "19p13.13", "omim_gene": 601011, "omim_disease": 183086,
        "inheritance": "Autosomal Dominant (AD). SCA6: CAG repeat ≥20 (narrow range, limited anticipation). EA2: heterozygous point mutations/truncations (LOF). De novo mutations account for ~5-10%.",
        "seed_offset": 4,
        "onset_range_y": (25.0, 70.0),
        "gender": "both",
        "severity_weights": [0.35, 0.45, 0.20],
        "repeat_type": "CAG (3'coding, polyQ) for SCA6; missense/truncation for EA2",
        "anticipation": False,
        "phenotype": (
            "SCA6 PHENOTYPE (CAG expansion ≥20): "
            "LATE ONSET (mean 52 years; rarely <20y). Pure cerebellar syndrome. "
            "GAIT ATAXIA; limb ataxia; dysarthria. Slower progression than SCA1/2/3. "
            "Normal lifespan in most. DOWNBEAT NYSTAGMUS (characteristic of SCA6 — upbeat/downbeat "
            "gaze-evoked nystagmus). Very limited phenotypic range. "
            "EA2 PHENOTYPE (LOF point mutations): "
            "EPISODIC ATAXIA TYPE 2: attacks of cerebellar ataxia lasting minutes to days, "
            "triggered by stress, exercise, alcohol, fever. Between attacks: downbeat nystagmus, "
            "mild interictal cerebellar signs. Progressive cerebellar atrophy over time. "
            "ACETAZOLAMIDE: highly effective for EA2 — reduces attack frequency significantly "
            "(mechanism: carbonic anhydrase inhibition → altered pH gradient → channel stabilisation). "
            "FHM1 COMPONENT: some EA2 families have hemiplegic migraine overlap. "
            "KEY DDx: SCA6 vs EA2 — episodic vs continuous ataxia; same gene, different mutation type."
        ),
        "disease": (
            "SCA6 prevalence: ~1-5:100,000 (varies). EA2 prevalence: ~1:100,000 (estimated). "
            "SCA6: very pure cerebellar; among the mildest dominant SCAs; >60 year survival common. "
            "EA2: excellent response to acetazolamide (500-1000 mg/day); quality of life good on Rx. "
            "Acetazolamide monitoring: renal stones, metabolic acidosis; potassium supplementation. "
            "4-aminopyridine (4-AP): alternative/adjunct for EA2 (potassium channel blocker; "
            "restores Purkinje cell function; 5-10 mg TID). "
            "No disease-modifying Rx for progressive SCA6 or EA2 cerebellar atrophy."
        ),
        "treatment_options": [
            "Acetazolamide (EA2: 500-1000 mg/day — HIGHLY EFFECTIVE)",
            "4-Aminopyridine (EA2: alternative/adjunct; Purkinje cell stabilizer)",
            "Physiotherapy (progressive cerebellar; SCA6)",
            "Trigger avoidance (EA2: stress, alcohol, exercise, fever)",
            "No disease-modifying Rx for progressive SCA6 atrophy",
        ],
        "outcome_options": [
            "Good — EA2; acetazolamide fully controls attacks",
            "Moderate — EA2 partially controlled; interictal signs stable",
            "Progressive — SCA6; pure cerebellar; slow decline",
            "Moderate — SCA6; late onset 60+; walking maintained",
            "Progressive — SCA6; gait aid; mild by lifespan end",
        ],
        "cag_range": (20, 33),
        "has_cardiac": False,
        "has_diabetes": False,
        "visual_loss": False,
        "vestibular": False,
    },

    # ── ATXN7 — SCA7 ─────────────────────────────────────────────────────
    {
        "gene": "ATXN7", "protein": "Ataxin-7",
        "alias": "SCA7 (Spinocerebellar Ataxia Type 7; OMIM #164500); AD; CAG polyQ; RETINAL DYSTROPHY + ATAXIA PATHOGNOMONIC — only polyQ SCA with visual loss; 3p14.1",
        "aa": "892 aa", "kDa": "97 kDa",
        "gene_class": (
            "Nuclear protein; component of the SAGA transcriptional coactivator complex "
            "(STAGA/GCN5L — histone acetyltransferase module and deubiquitinase module). "
            "Ataxin-7 is the structural anchor of the DUB module; polyQ expansion disrupts "
            "SAGA complex function → transcriptional dysregulation in Purkinje cells AND "
            "photoreceptors (cone > rod in retina). "
            "Normal: ≤27 CAG; pathogenic: ≥36 CAG. "
            "VERY STRONG ANTICIPATION in SCA7 — especially with paternal transmission; "
            "juvenile/infantile cases (>100 CAG) occur and are rapidly fatal. "
            "3p14.1; OMIM gene 607640."
        ),
        "ataxia_group": "Autosomal Dominant Polyglutamine SCA",
        "subtype": "SCA7 — Spinocerebellar Ataxia Type 7 with Retinal Dystrophy",
        "locus": "3p14.1", "omim_gene": 607640, "omim_disease": 164500,
        "inheritance": "Autosomal Dominant (AD). Strong anticipation (paternal transmission → very large expansions → juvenile/infantile onset). De novo large expansions possible.",
        "seed_offset": 5,
        "onset_range_y": (5.0, 60.0),
        "gender": "both",
        "severity_weights": [0.15, 0.40, 0.45],
        "repeat_type": "CAG (exon 3, polyQ)",
        "anticipation": True,
        "phenotype": (
            "PATHOGNOMONIC COMBINATION — the only polyQ SCA with visual involvement: "
            "RETINAL DYSTROPHY (cone-rod pattern): "
            "  - Early: macular degeneration → colour vision loss → central visual loss. "
            "  - Later: peripheral visual loss → complete blindness. "
            "  - Fundoscopy: macular pigmentary changes → bull's eye maculopathy. "
            "  - ERG: cone dysfunction precedes rod dysfunction. "
            "  - In severe (large repeat): visual loss may PRECEDE ataxia by years. "
            "CEREBELLAR ATAXIA: gait, limb, dysarthria (progressive). "
            "OCULOMOTOR ABNORMALITIES: nystagmus; gaze palsy (late). "
            "PYRAMIDAL FEATURES: spasticity (late, in some). "
            "JUVENILE/INFANTILE FORM (>100 CAG): neonatal hypotonia; rapid multi-organ failure; "
            "patent ductus arteriosus; seizures; death in infancy/childhood. "
            "ANTICIPATION IS EXTREME: parent 50 CAG → child may have 100+ CAG; parent adult onset "
            "→ child infant onset. Family history essential."
        ),
        "disease": (
            "SCA7 prevalence: ~1-5:100,000 (common in Scandinavian populations, South Africa, "
            "and some South American populations). "
            "Visual loss + ataxia: always test for SCA7 gene (ERG + ophthalmology + SCA panel). "
            "Low vision aids; ophthalmological follow-up. "
            "No approved disease-modifying treatment. "
            "Genetic counselling mandatory (extreme anticipation risk). "
            "De novo cases require fresh expansion from parental pre-mutation alleles."
        ),
        "treatment_options": [
            "Low vision aids; adaptive technology for visual loss",
            "Ophthalmology follow-up (macular monitoring)",
            "Physiotherapy + speech therapy + gait aids",
            "Genetic counselling (extreme anticipation risk to offspring)",
            "No disease-modifying Rx for SCA7 (2026)",
        ],
        "outcome_options": [
            "Progressive — visual loss + ataxia; legally blind + wheelchair 15-20y",
            "Progressive — visual loss precedes ataxia; bilateral macular degeneration",
            "Severe — juvenile onset; rapid visual + cerebellar failure",
            "Progressive — adult onset; slower decline; low vision aids",
            "Fatal — infantile/neonatal form; multi-organ failure",
        ],
        "cag_range": (36, 300),
        "has_cardiac": False,
        "has_diabetes": False,
        "visual_loss": True,
        "vestibular": False,
    },

    # ── TBP — SCA17 ──────────────────────────────────────────────────────
    {
        "gene": "TBP", "protein": "TATA-Box Binding Protein",
        "alias": "SCA17 (Spinocerebellar Ataxia Type 17; OMIM #607136); AD; CAG/CAA polyQ; WIDEST PHENOTYPE among SCAs; Huntington Disease DDx; 6q27",
        "aa": "339 aa", "kDa": "38 kDa",
        "gene_class": (
            "Universal transcription factor; component of the TFIID complex; binds TATA-box "
            "in promoters of RNA pol I, II, III genes. Ubiquitous — absolutely required for "
            "basal transcription of most eukaryotic genes. PolyQ/polyA tract in N-terminus "
            "(encoded by CAG and CAA codons — both encode glutamine). "
            "Normal: ≤40 repeats (mixed CAG/CAA); pathogenic: ≥45. 41-44: reduced penetrance. "
            "PolyQ expansion → global transcriptional dysregulation → widespread neuronal death "
            "(not limited to Purkinje cells — broader neuronal populations than most SCAs). "
            "6q27; OMIM gene 600075. "
            "HUNTINGTON DDx: SCA17 can present with chorea + dementia + psychiatric = "
            "Huntington-like phenotype → classified as Huntington Disease-Like Disorder (HDL4)."
        ),
        "ataxia_group": "Autosomal Dominant Polyglutamine SCA",
        "subtype": "SCA17 — TATA-Box Binding Protein CAG Expansion",
        "locus": "6q27", "omim_gene": 600075, "omim_disease": 607136,
        "inheritance": "Autosomal Dominant (AD). CAG/CAA repeat expansion in TBP. Moderate anticipation. De novo expansions from large normal alleles.",
        "seed_offset": 6,
        "onset_range_y": (20.0, 65.0),
        "gender": "both",
        "severity_weights": [0.20, 0.45, 0.35],
        "repeat_type": "CAG+CAA (N-terminal polyQ)",
        "anticipation": True,
        "phenotype": (
            "WIDEST PHENOTYPIC SPECTRUM of all SCAs — can mimic many diseases: "
            "CEREBELLAR ATAXIA: usually present but not always dominant feature. "
            "CHOREA AND INVOLUNTARY MOVEMENTS: prominent in many patients → HUNTINGTON PHENOCOPY. "
            "PSYCHIATRIC FEATURES: depression, psychosis, personality change, bipolar-like. "
            "DEMENTIA: frontal > other; earlier than most SCAs. "
            "SEIZURES: ~20-30% (uncommon in other SCAs). "
            "DYSTONIA: generalised in some. "
            "PYRAMIDAL SIGNS: common. "
            "PARKINSONISM: in some older-onset patients. "
            "MRI: cerebellar + cerebral cortex atrophy (more diffuse than SCA1/2/3). "
            "HDL4 PHENOTYPE: some patients have minimal ataxia and predominantly "
            "choreiform movements + dementia + psychiatric — misdiagnosed as Huntington disease "
            "until HTT repeat test returns normal → SCA17/TBP testing warranted. "
            "INVESTIGATION KEY: always include TBP repeat testing in Huntington-like "
            "phenotype with normal HTT; and in SCA panel if chorea prominent."
        ),
        "disease": (
            "SCA17 prevalence: estimated ~1-5:100,000 (underdiagnosed due to phenotypic breadth). "
            "Among the most common diagnoses when SCA1/2/3/6/7 panel is negative and phenotype "
            "includes chorea/psychiatric features. "
            "Repeat size (above 45): imprecise correlation — same repeat can cause different phenotypes. "
            "No approved disease-modifying treatment. "
            "Symptomatic: tetrabenazine for chorea; antipsychotics for psychiatric; "
            "antiepileptics for seizures (avoid valproate if liver disease; LEV preferred)."
        ),
        "treatment_options": [
            "Tetrabenazine / deutetrabenazine (chorea suppression)",
            "Antipsychotics (psychiatric features; atypical preferred)",
            "Antiepileptics — LEV (seizures; ~20-30% of patients)",
            "Physiotherapy + cognitive support",
            "No disease-modifying Rx for SCA17 (2026)",
        ],
        "outcome_options": [
            "Progressive — ataxia + dementia + chorea; Huntington phenocopy",
            "Progressive — chorea + psychiatric predominant; late ataxia",
            "Progressive — seizures + ataxia + dementia",
            "Moderate — cerebellar predominant; slower course",
            "Severe — early dementia + seizures + ataxia; rapid decline",
        ],
        "cag_range": (45, 63),
        "has_cardiac": False,
        "has_diabetes": False,
        "visual_loss": False,
        "vestibular": False,
    },

    # ── RFC1 — CANVAS ─────────────────────────────────────────────────────
    {
        "gene": "RFC1", "protein": "Replication Factor C Subunit 1",
        "alias": "CANVAS (Cerebellar Ataxia, Neuropathy, Vestibular Areflexia Syndrome; OMIM #614575); AR; AAGGG pentanucleotide repeat; most common adult-onset AR ataxia after FRDA; chronic cough; 4p14",
        "aa": "1148 aa", "kDa": "128 kDa",
        "gene_class": (
            "Replication factor C large subunit; DNA clamp loader component; involved in "
            "DNA replication. AAGGG pentanucleotide repeat expansion in intron 2 (biallelic) "
            "→ CANVAS. Normal: AAAAG (non-pathogenic); AAGGG biallelic expansion: pathogenic; "
            "mixed: AAGGG + AAAGG (or other interruptions) — may be pathogenic. "
            "AAGGG expansion → putative RNA foci/toxicity → sensory neuron + cerebellar + "
            "vestibular ganglion degeneration. Mechanism under active investigation. "
            "CARRIER FREQUENCY ~1:80 in Europeans — CANVAS is underdiagnosed. "
            "Recent recognition (~2019-2023): previously many CANVAS cases were 'idiopathic'. "
            "4p14; OMIM gene 102579."
        ),
        "ataxia_group": "Autosomal Recessive Repeat-Expansion Ataxia",
        "subtype": "CANVAS — Cerebellar Ataxia, Neuropathy, Vestibular Areflexia Syndrome",
        "locus": "4p14", "omim_gene": 102579, "omim_disease": 614575,
        "inheritance": "Autosomal Recessive (AR). Biallelic AAGGG pentanucleotide repeat expansion in intron 2. High carrier frequency (~1:80 Europeans). Diagnosed from ~2019 onwards — many prior cases misdiagnosed.",
        "seed_offset": 7,
        "onset_range_y": (45.0, 75.0),
        "gender": "both",
        "severity_weights": [0.30, 0.45, 0.25],
        "repeat_type": "AAGGG pentanucleotide (intron 2, biallelic)",
        "anticipation": False,
        "phenotype": (
            "PATHOGNOMONIC TRIPLE COMBINATION: "
            "(1) CEREBELLAR ATAXIA: progressive gait and limb ataxia; dysarthria; slow progression. "
            "(2) SENSORIMOTOR NEUROPATHY: length-dependent; absent lower limb reflexes; "
            "loss of vibration/proprioception; EMG/NCS: axonal sensorimotor neuropathy. "
            "(3) BILATERAL VESTIBULAR AREFLEXIA: complete bilateral loss of vestibular function → "
            "oscillopsia (world bounces during walking), imbalance worse in dark/eyes closed, "
            "positive head impulse test bilaterally, no caloric responses. "
            "CHRONIC COUGH: >60% of CANVAS patients have chronic dry cough (sensory neuropathy "
            "of the recurrent laryngeal branch of vagus nerve → loss of cough reflex suppression). "
            "Often misdiagnosed as idiopathic, ACE-inhibitor cough, etc. "
            "ONSET: typically 45-75 years (adult-onset; unlike FRDA childhood onset). "
            "NOT ALL 3 FEATURES REQUIRED SIMULTANEOUSLY: incomplete CANVAS "
            "(only 2 features) is recognised. "
            "SLOW PROGRESSION: compatible with decades of survival; not rapidly fatal. "
            "MRI: cerebellar cortical + dorsal column atrophy."
        ),
        "disease": (
            "Previously severely underdiagnosed as 'idiopathic late-onset cerebellar ataxia' "
            "or 'idiopathic sensorimotor neuropathy.' "
            "CANVAS now estimated to be the SECOND MOST COMMON AR hereditary ataxia after FRDA. "
            "European carrier frequency ~1:80 → disease frequency ~1:6,400. "
            "Diagnosis: repeat-primed PCR for AAGGG expansion (standard SCA gene panels miss it — "
            "requires CANVAS-specific testing). "
            "No disease-modifying treatment. "
            "Vestibular rehabilitation (helpful for imbalance). "
            "Chronic cough: may respond to neuromodulators (gabapentin, pregabalin)."
        ),
        "treatment_options": [
            "Vestibular physiotherapy (gaze stabilisation exercises)",
            "Walking aids + balance training",
            "Gabapentin/pregabalin (chronic cough — off-label)",
            "Speech therapy (dysarthria)",
            "No disease-modifying Rx for CANVAS (2026)",
        ],
        "outcome_options": [
            "Progressive — triple combination; cane then walker; cough managed",
            "Progressive — bilateral vestibular areflexia dominant; ataxia mild",
            "Progressive — neuropathy + ataxia; vestibular training helping",
            "Moderate — incomplete CANVAS (2 features); slower course",
            "Progressive — all 3 features; chronic cough + ataxia + neuropathy",
        ],
        "cag_range": None,
        "aaggg_range": (400, 2000),
        "has_cardiac": False,
        "has_diabetes": False,
        "visual_loss": False,
        "vestibular": True,
    },
]


def _make_patients(gd):
    rng = random.Random(SEED_BASE + gd["seed_offset"])
    n = 40
    pts = []
    sev_labels = ["Mild", "Moderate", "Severe"]
    sev_weights = gd.get("severity_weights", [0.25, 0.45, 0.30])
    gender_bias = gd.get("gender", "both")
    for i in range(n):
        sid = f"{gd['gene']}-{SEED_BASE + gd['seed_offset']:03d}-{i+1:03d}"
        sev = rng.choices(sev_labels, weights=sev_weights, k=1)[0]
        lo, hi = gd["onset_range_y"]
        age_onset = round(rng.uniform(lo, hi), 1)
        # Diagnosis delay
        if gd["gene"] == "RFC1":
            delay = round(rng.uniform(1.0, 8.0), 1)  # frequently misdiagnosed pre-2019
        elif gd["gene"] == "FXN":
            delay = round(rng.uniform(0.5, 4.0), 1)
        else:
            delay = round(rng.uniform(1.0, 6.0), 1)
        # Gender
        if gender_bias == "male":
            sex = "M" if rng.random() < 0.80 else "F"
        elif gender_bias == "female":
            sex = "F" if rng.random() < 0.70 else "M"
        else:
            sex = rng.choice(["M", "F"])
        # Repeat size
        if gd.get("cag_range"):
            lo_r, hi_r = gd["cag_range"]
            repeat_n = rng.randint(lo_r, hi_r)
        elif gd.get("gaa_range"):
            lo_r, hi_r = gd["gaa_range"]
            repeat_n = rng.randint(lo_r, hi_r)
        elif gd.get("aaggg_range"):
            lo_r, hi_r = gd["aaggg_range"]
            repeat_n = rng.randint(lo_r, hi_r)
        else:
            repeat_n = None
        # Cardiac (FXN only)
        has_cardiac = gd.get("has_cardiac", False) and rng.random() < 0.80
        # Visual loss (ATXN7)
        visual_loss = gd.get("visual_loss", False) and rng.random() < 0.85
        # Vestibular (RFC1)
        vestibular = gd.get("vestibular", False) and rng.random() < 0.90
        # Chronic cough (RFC1)
        chronic_cough = gd["gene"] == "RFC1" and rng.random() < 0.65
        # Anticipation (for dominant SCAs)
        paternal_transmission = gd.get("anticipation", False) and rng.random() < 0.55
        # Diabetes (FXN)
        has_diabetes = gd.get("has_diabetes", False) and rng.random() < 0.12
        # MRI atrophy pattern
        if gd["gene"] == "FXN":
            mri = "Dorsal column + dentate nucleus + spinocerebellar tract atrophy"
        elif gd["gene"] in ("ATXN1", "ATXN2", "ATXN3"):
            mri = "Olivopontocerebellar atrophy (OPCA) — cerebellar + pons + inferior olives"
        elif gd["gene"] == "CACNA1A":
            mri = "Isolated cerebellar atrophy (pure cerebellar); pons normal"
        elif gd["gene"] == "ATXN7":
            mri = "Cerebellar + brainstem atrophy; macular changes on fundoscopy"
        elif gd["gene"] == "TBP":
            mri = "Cerebellar + cerebral cortical atrophy (diffuse)"
        elif gd["gene"] == "RFC1":
            mri = "Cerebellar cortical atrophy + dorsal column signal"
        else:
            mri = "Cerebellar atrophy"
        treatment = rng.choice(gd["treatment_options"])
        outcome = rng.choice(gd["outcome_options"])
        pts.append({
            "id": sid,
            "gene": gd["gene"],
            "sex": sex,
            "age_onset_y": age_onset,
            "dx_delay_y": delay,
            "severity": sev,
            "repeat_n": repeat_n,
            "repeat_type": gd.get("repeat_type"),
            "mri_pattern": mri,
            "has_cardiac": has_cardiac,
            "visual_loss": visual_loss,
            "vestibular_areflexia": vestibular,
            "chronic_cough": chronic_cough,
            "has_diabetes": has_diabetes,
            "paternal_anticipation": paternal_transmission,
            "treatment": treatment,
            "outcome": outcome,
        })
    return pts


def get_overview():
    all_pts = []
    gene_summary = {}
    group_counts = {}
    for gd in SCA_GENES:
        pts = _make_patients(gd)
        all_pts.extend(pts)
        gene_summary[gd["gene"]] = len(pts)
        grp = gd["ataxia_group"]
        group_counts[grp] = group_counts.get(grp, 0) + len(pts)
    n = len(all_pts)
    avg_onset = round(sum(p["age_onset_y"] for p in all_pts) / n, 1)
    avg_delay = round(sum(p["dx_delay_y"] for p in all_pts) / n, 1)
    sev_dist = {"Mild": 0, "Moderate": 0, "Severe": 0}
    cardiac_n = 0
    visual_n = 0
    vestibular_n = 0
    cough_n = 0
    diabetic_n = 0
    anticipation_n = 0
    for p in all_pts:
        sev_dist[p["severity"]] += 1
        if p["has_cardiac"]:
            cardiac_n += 1
        if p["visual_loss"]:
            visual_n += 1
        if p["vestibular_areflexia"]:
            vestibular_n += 1
        if p["chronic_cough"]:
            cough_n += 1
        if p["has_diabetes"]:
            diabetic_n += 1
        if p["paternal_anticipation"]:
            anticipation_n += 1
    return {
        "title": "SCA-Atlas — Complete 8-Gene Hereditary Ataxia / Spinocerebellar Ataxia Atlas",
        "subtitle": "FXN/FRDA · ATXN1/SCA1 · ATXN2/SCA2 · ATXN3/SCA3-MJD · CACNA1A/SCA6-EA2 · ATXN7/SCA7 · TBP/SCA17 · RFC1/CANVAS",
        "genes": [gd["gene"] for gd in SCA_GENES],
        "subtypes": [gd["subtype"] for gd in SCA_GENES],
        "total_patients": n,
        "avg_onset_y": avg_onset,
        "avg_dx_delay_y": avg_delay,
        "severity_distribution": sev_dist,
        "cardiac_n": cardiac_n,
        "cardiac_pct": round(100 * cardiac_n / n, 1),
        "visual_loss_n": visual_n,
        "visual_loss_pct": round(100 * visual_n / n, 1),
        "vestibular_n": vestibular_n,
        "vestibular_pct": round(100 * vestibular_n / n, 1),
        "chronic_cough_n": cough_n,
        "chronic_cough_pct": round(100 * cough_n / n, 1),
        "ataxia_groups": group_counts,
        "gene_summary": gene_summary,
        "key_facts": [
            "FRIEDREICH ATAXIA (FXN): only AR ataxia in this atlas; GAA repeat; HCM causes most deaths; omaveloxolone FDA 2023 (first approved Rx)",
            "ATXN2 (SCA2): SLOW SACCADES pathognomonic — slowest of all SCAs; intermediate 27-32 CAG = ALS risk modifier (IONIS ASO in trial)",
            "ATXN3 (SCA3/MJD): MOST COMMON SCA worldwide (40%); bulging eyes + lid retraction; restless legs 60-80%; Azorean founder",
            "ATXN7 (SCA7): ONLY polyQ SCA with VISUAL LOSS — cone-rod retinal dystrophy + ataxia; extreme anticipation (paternal)",
            "TBP (SCA17): WIDEST PHENOTYPE — Huntington disease-like (chorea + dementia + psychiatric); test TBP when HTT negative + HD-like",
            "RFC1 (CANVAS): AR; AAGGG repeat; triple combination (cerebellar + neuropathy + bilateral vestibular areflexia); chronic cough 60%; carrier 1:80",
            "CACNA1A (SCA6/EA2): SCA6 = late onset pure cerebellar; EA2 = episodic attacks; ACETAZOLAMIDE HIGHLY EFFECTIVE for EA2",
            "ANTICIPATION: strong in SCA7 (extreme; paternal); moderate in SCA2 (ALS overlap); limited in SCA6",
            "SCA PANEL MUST INCLUDE TBP AND RFC1 — missed by older conventional SCA1/2/3/6/7 panels only",
            "FRDA ECG/ECHO: annual monitoring mandatory (HCM + arrhythmia risk; ICD for sustained VT)",
            "EA2 DDx IMPORTANT: episodic vs progressive ataxia → test CACNA1A first; acetazolamide trial diagnostic",
            "ATXN2 INTERMEDIATE (27-32 CAG): ALS modifier — always report in ALS genetic counselling",
        ],
        "critical_distinctions": {
            "SCA2 vs all": "SLOW SACCADES most discriminating bedside sign; hyporeflexia (unlike SCA1 hyperreflexia)",
            "SCA7 vs all": "RETINAL DYSTROPHY + ataxia — only polyQ SCA with visual loss; macular exam mandatory",
            "SCA17 vs Huntington": "Both: chorea + dementia + psychiatric. HTT normal → test TBP. SCA17 has ataxia more.",
            "FRDA vs SCAs": "AR (not AD); childhood onset; ABSENT lower reflexes; cardiac disease; GAA not CAG repeat",
            "CANVAS vs FRDA": "Both AR. CANVAS: adult onset; bilateral vestibular areflexia; chronic cough; AAGGG not GAA",
            "EA2 vs SCA6": "Same gene; episodic (EA2 = LOF point mutation) vs progressive (SCA6 = CAG expansion)",
        },
        "seeds_used": f"{SEED_BASE}–{SEED_BASE + len(SCA_GENES) - 1}",
    }


_gd_cache = {gd["gene"]: gd for gd in SCA_GENES}


def gd_by_gene(gene):
    return _gd_cache[gene]


def get_breakdown():
    result = []
    for gd in SCA_GENES:
        pts = _make_patients(gd)
        sev = {"Severe": 0, "Moderate": 0, "Mild": 0}
        for p in pts:
            sev[p["severity"]] += 1
        avg_delay = round(sum(p["dx_delay_y"] for p in pts) / len(pts), 1)
        avg_onset = round(sum(p["age_onset_y"] for p in pts) / len(pts), 1)
        cardiac_n = sum(1 for p in pts if p["has_cardiac"])
        visual_n = sum(1 for p in pts if p["visual_loss"])
        vestibular_n = sum(1 for p in pts if p["vestibular_areflexia"])
        cough_n = sum(1 for p in pts if p["chronic_cough"])
        treatments = {}
        for p in pts:
            treatments[p["treatment"]] = treatments.get(p["treatment"], 0) + 1
        top_tx = sorted(treatments.items(), key=lambda x: -x[1])[:3]
        outcomes = {}
        for p in pts:
            outcomes[p["outcome"]] = outcomes.get(p["outcome"], 0) + 1
        result.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "ataxia_group": gd["ataxia_group"],
            "subtype": gd["subtype"],
            "n_patients": len(pts),
            "severity_distribution": sev,
            "avg_onset_y": avg_onset,
            "avg_dx_delay_y": avg_delay,
            "cardiac_n": cardiac_n,
            "cardiac_pct": round(100 * cardiac_n / len(pts), 1),
            "visual_loss_n": visual_n,
            "visual_loss_pct": round(100 * visual_n / len(pts), 1),
            "vestibular_n": vestibular_n,
            "vestibular_pct": round(100 * vestibular_n / len(pts), 1),
            "cough_n": cough_n,
            "cough_pct": round(100 * cough_n / len(pts), 1),
            "repeat_type": gd.get("repeat_type"),
            "anticipation": gd.get("anticipation", False),
            "top_treatments": [{"tx": t, "n": c} for t, c in top_tx],
            "outcome_distribution": outcomes,
            "gene_class": gd["gene_class"],
            "phenotype": gd["phenotype"],
            "disease": gd["disease"],
        })
    return {"total_genes": len(SCA_GENES), "total_patients": sum(r["n_patients"] for r in result), "breakdown": result}


def get_definitions():
    return {
        "definitions": [
            {
                "term": "Spinocerebellar Ataxia (SCA) — Naming, Classification, and Numbering",
                "definition": (
                    "SCAs are numbered sequentially in order of gene discovery: SCA1, SCA2, SCA3, etc. "
                    "(currently >40 numbered SCAs). NOT a phenotypic progression. "
                    "Classification by inheritance: Autosomal Dominant (AD-SCAs; most), AR (FRDA, CANVAS), "
                    "X-linked (rare), mitochondrial. By molecular mechanism: "
                    "(1) Polyglutamine (polyQ/CAG repeat): SCA1, 2, 3, 6, 7, 17, DRPLA — gain of toxic function. "
                    "(2) Non-coding repeat expansions: SCA8 (CTG/CAG), SCA10 (ATTCT), SCA12 (CAG PPP2R2B), "
                    "CANVAS/RFC1 (AAGGG — AR). "
                    "(3) Conventional mutations (missense/truncation/deletion): SCA5, 11, 13, 14, 15, 19, 22, 23, 26, 28, 35, 41. "
                    "Key learning: SCA panel must include polyQ (repeat-primed PCR) AND sequencing; "
                    "CANVAS requires specific AAGGG assay; standard SCA1-7 panel misses most SCAs."
                )
            },
            {
                "term": "Anticipation in Polyglutamine SCAs — Mechanism and Paternal Bias",
                "definition": (
                    "Anticipation: earlier onset and/or more severe disease in successive generations. "
                    "Mechanism: meiotic instability of CAG repeats → repeat expansion during cell division, "
                    "especially spermatogenesis (male gametes undergo more replications → larger instability). "
                    "Paternal bias: expansion more frequent and larger via paternal than maternal transmission. "
                    "SCA7 (ATXN7): EXTREME anticipation. Paternal transmission → massive expansions (parent 50 → child 100+ CAG) → "
                    "juvenile/infantile onset from an adult-onset parent. "
                    "SCA3 (ATXN3): Moderate anticipation; paternal transmission more unstable. "
                    "SCA2 (ATXN2): Moderate anticipation; largest expansions paternally transmitted. "
                    "SCA6 (CACNA1A): Minimal anticipation — CAG range is narrow (20-33) with limited instability. "
                    "Clinical implication: offspring of SCA7 patient should have genetic counselling urgently; "
                    "prenatal/preimplantation testing options exist. Recurrence risk = 50% per child."
                )
            },
            {
                "term": "Slow Saccades in SCA2 — Bedside Examination",
                "definition": (
                    "Saccades are rapid voluntary eye movements used to change gaze direction. Normal saccadic "
                    "velocity: 300-700 degrees/second. In SCA2 (ATXN2): markedly reduced velocity "
                    "(<200 degrees/sec; often <100 deg/sec). Clinical testing: ask patient to look rapidly "
                    "left-right between two fixed targets. Slow saccades are obvious — eyes drift slowly, "
                    "may require a corrective movement to reach target. "
                    "PATHOGNOMONIC VALUE: of all the polyQ SCAs, SCA2 reliably produces the slowest saccades. "
                    "SCA1, SCA3, SCA6, SCA17: saccades may be abnormal but not as dramatically slowed. "
                    "Bedside test: hold one finger left and one finger right; ask patient to alternately fixate "
                    "rapidly. Slow drift rather than brisk flick = SCA2 likely. "
                    "EMG equivalent: infrared oculography in lab confirms and quantifies. "
                    "CLINICAL UTILITY: when you see slow saccades in a patient with hereditary ataxia → "
                    "order SCA2 (ATXN2) repeat testing first."
                )
            },
            {
                "term": "CANVAS (RFC1) — Diagnosis and Why It Was Previously Missed",
                "definition": (
                    "CANVAS = Cerebellar Ataxia, Neuropathy, Vestibular Areflexia Syndrome. "
                    "Caused by biallelic AAGGG pentanucleotide repeat expansion in intron 2 of RFC1. "
                    "WHY PREVIOUSLY MISSED (pre-2019): "
                    "(1) RFC1 was not on standard SCA panels (which use short-range PCR for CAG repeats). "
                    "(2) AAGGG repeat requires repeat-primed PCR or long-read sequencing — different assay. "
                    "(3) All three features (ataxia + neuropathy + vestibular) required to recognise the triad. "
                    "(4) Vestibular testing not always performed in ataxia clinic. "
                    "(5) Chronic cough attributed to other causes (ACEi, reflux, idiopathic). "
                    "DIAGNOSIS NOW: RFC1 AAGGG specific assay available (most neurology centres). "
                    "Who to test: late-onset cerebellar ataxia + neuropathy + bilateral vestibular failure "
                    "(no caloric responses; positive head impulse bilaterally). "
                    "Add RFC1 to ALL adult-onset ataxia workups — very high prevalence in Europeans (~1:6,400)."
                )
            },
            {
                "term": "Omaveloxolone (Skyclarys) — First Approved Treatment for Friedreich Ataxia",
                "definition": (
                    "FDA approved February 2023 (Reata Pharmaceuticals, now AstraZeneca/Biogen). "
                    "First disease-modifying treatment for Friedreich ataxia (FRDA). "
                    "Mechanism: synthetic triterpenoid; activates Nrf2 (nuclear factor erythroid 2-related "
                    "factor 2) pathway → upregulates antioxidant response element (ARE) genes → "
                    "reduced mitochondrial oxidative stress → neuroprotection. "
                    "Does NOT increase frataxin levels (not a frataxin replacement). "
                    "Pivotal trial: MOXIe Part 2 (NEJM 2022): mFARS (modified Friedreich Ataxia Rating Scale) "
                    "improved by 2.1 points vs placebo worsening of 1.4 points (p<0.001). "
                    "Indication: adults and adolescents ≥16 years with FRDA. "
                    "Dose: 150 mg orally once daily. "
                    "Monitoring: LFTs (hepatotoxicity ~15%), oedema, hypertension. "
                    "Does NOT halt progression; slows functional decline. Cardiac effects: neutral in trial. "
                    "Ongoing: frataxin upregulators (epigenetic silencing reversal), AAV gene therapy trials."
                )
            },
            {
                "term": "Acetazolamide for Episodic Ataxia Type 2 (EA2/CACNA1A) — Mechanism",
                "definition": (
                    "Acetazolamide (carbonic anhydrase inhibitor) is the treatment of choice for "
                    "Episodic Ataxia Type 2 (EA2) caused by loss-of-function CACNA1A mutations. "
                    "Typical dose: 250-1000 mg/day (titrated). "
                    "Efficacy: reduces attack frequency by 50-75% in responders (60-70% response rate). "
                    "Mechanism (not fully elucidated): inhibition of carbonic anhydrase alters neuronal pH "
                    "and CO2 levels → altered intracellular calcium dynamics → reduced cerebellar neuronal "
                    "hyperexcitability. Separate from its diuretic effect (which is the main side effect). "
                    "Alternative: 4-Aminopyridine (4-AP, 5-10 mg TID) — potassium channel blocker; "
                    "restores Purkinje cell firing regularity; preferred in acetazolamide-intolerant. "
                    "Side effects: paraesthesiae, renal stones, metabolic acidosis, fatigue. "
                    "Potassium monitoring required; potassium supplement often needed. "
                    "KEY: EA2 must be distinguished from SCA6 (same gene, different mutation type) — "
                    "EA2 responds to acetazolamide; SCA6 (progressive) does NOT."
                )
            },
            {
                "term": "SCA17 (TBP) — Huntington Disease-Like Phenotype (HDL4)",
                "definition": (
                    "SCA17 caused by CAG/CAA polyQ repeat expansion in TATA-binding protein (TBP). "
                    "TBP is a universal transcription factor — disruption causes widespread neurodegeneration "
                    "beyond pure cerebellar. "
                    "HUNTINGTON PHENOCOPY: chorea + psychiatric disease + cognitive decline WITHOUT prominent ataxia "
                    "→ may be identical to Huntington's disease on clinical grounds. "
                    "Classification: HDL4 (Huntington Disease-Like 4) when this phenotype predominates. "
                    "CLINICAL ALGORITHM for Huntington-like disease: "
                    "(1) Test HTT CAG repeat (Huntington's): if ≤35 CAG → normal. "
                    "(2) If HTT negative → test SCA17/TBP repeat. "
                    "(3) Also test: DRPLA (ATXN1-related) in Asian patients; HDL1 (PRNP), HDL2 (JPH3). "
                    "SCA17 features favouring over HD: cerebellar signs (ataxia, dysarthria), seizures (20-30%), "
                    "family history suggesting AD ataxia. "
                    "Repeat range: ≥45 CAG/CAA; normal ≤40. Gray zone 41-44: reduced penetrance. "
                    "No disease-modifying treatment. Tetrabenazine for chorea; antipsychotics for psychiatric; LEV for seizures."
                )
            },
            {
                "term": "Friedreich Ataxia Cardiomyopathy — Mechanism, Monitoring, and Treatment",
                "definition": (
                    "Hypertrophic cardiomyopathy (HCM) occurs in >80% of Friedreich ataxia (FRDA) patients "
                    "and is the leading cause of death (arrhythmia/heart failure). "
                    "MECHANISM: frataxin deficiency in cardiomyocytes → mitochondrial iron accumulation → "
                    "Fenton reaction → reactive oxygen species → cardiomyocyte fibrosis and hypertrophy. "
                    "Cardiac HCM in FRDA: typically concentric LVH; does NOT cause LV outflow tract obstruction "
                    "typically (unlike HCM from MYBPC3/MYH7 — different pathogenesis). "
                    "ECHOCARDIOGRAM: annual monitoring from diagnosis. LV wall thickness, EF, diastolic function. "
                    "ECG: PR/QRS interval; VT risk. "
                    "TREATMENT: "
                    "  - Beta-blockers: standard for HCM in FRDA (rate control, anti-arrhythmic). "
                    "  - ICD: for sustained VT or SCD risk (SCD is not uncommon in young FRDA). "
                    "  - Heart transplantation: considered for severe end-stage cardiomyopathy in young patients. "
                    "  - Omaveloxolone: cardiac effects in MOXIe trial — neutral (did not worsen but not yet "
                    "    proven to improve cardiac outcomes specifically). "
                    "DIABETES: 10-15% of FRDA develop DM (pancreatic beta-cell iron toxicity); managed as type 1/insulin-requiring."
                )
            },
            {
                "term": "Repeat-Primed PCR and Repeat Expansion Ataxia Testing",
                "definition": (
                    "Standard short-range PCR cannot amplify across large repeat expansions (>~100 repeats). "
                    "Repeat-primed PCR (RP-PCR): uses a repeat-specific primer → generates ladder pattern "
                    "on capillary electrophoresis → identifies presence of expansion (qualitative). "
                    "Limitation: does not give exact repeat length for very large expansions. "
                    "Long-range PCR: can estimate size for shorter large alleles. "
                    "Southern blot: classic method for GAA in FRDA (large GAA alleles up to 1300 repeats). "
                    "Long-read sequencing (PacBio, Nanopore): most accurate for large and complex repeats; "
                    "now used for RFC1 AAGGG (distinguishes AAGGG from AAAAG normal and AAAGG intermediate). "
                    "CLINICAL IMPLICATION: "
                    "  - Standard SCA gene panel (SCA1/2/3/6/7): covers CAG repeats only; MISSES RFC1/CANVAS. "
                    "  - FXN GAA: requires specific test (RP-PCR or Southern blot). "
                    "  - RFC1 AAGGG: specific assay required; most current panels now include it. "
                    "  - Ask your lab: 'Does your SCA panel include RFC1?' — if not, order separately. "
                    "SCA genetic test algorithm: 1) polyQ panel (SCA1/2/3/6/7/17), 2) FXN GAA, 3) RFC1, "
                    "4) extended/whole exome if still negative."
                )
            },
            {
                "term": "ATXN2 Intermediate Repeats — ALS Risk Modifier",
                "definition": (
                    "Normal ATXN2 CAG repeats: ≤31 (some labs use ≤27). "
                    "SCA2 pathogenic: ≥33 (or ≥34 by some criteria). "
                    "INTERMEDIATE / GRAY ZONE: 27-32 CAG — does NOT cause SCA2, but DRAMATICALLY "
                    "increases ALS (amyotrophic lateral sclerosis) risk. "
                    "Evidence: ATXN2 intermediate repeats (~5% of familial ALS, ~1% of sporadic ALS). "
                    "Mechanism: intermediate polyQ enhances phase separation and TDP-43 (TARDBP) aggregation "
                    "— TDP-43 proteinopathy is the hallmark pathology of most ALS. "
                    "Clinical significance: "
                    "(1) Routine ATXN2 testing now recommended in ALS genetic panels. "
                    "(2) If ALS patient has ATXN2 intermediate → family counselling (50% transmission risk, "
                    "though ALS penetrance is incomplete and complex). "
                    "(3) IONIS-ATXN2-Lrx (antisense oligonucleotide targeting ATXN2): in Phase 1/2 ALS trials "
                    "with promising early data; also investigated for SCA2 prevention. "
                    "IMPORTANT DISTINCTION: intermediate repeat carriers (27-32) do NOT develop SCA2; "
                    "ALS risk is a modifier not a deterministic cause."
                )
            },
        ]
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print("=== SCA ATLAS OVERVIEW ===")
    print(f"Genes: {ov['genes']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Avg onset: {ov['avg_onset_y']} y")
    print(f"Avg dx delay: {ov['avg_dx_delay_y']} y")
    print(f"Cardiac disease: {ov['cardiac_n']} ({ov['cardiac_pct']}%)")
    print(f"Visual loss: {ov['visual_loss_n']} ({ov['visual_loss_pct']}%)")
    print(f"Vestibular areflexia: {ov['vestibular_n']} ({ov['vestibular_pct']}%)")
    print(f"Ataxia groups: {ov['ataxia_groups']}")
    bd = get_breakdown()
    print(f"\n=== BREAKDOWN ({bd['total_genes']} genes) ===")
    for g in bd["breakdown"]:
        print(f"  {g['gene']}: {g['n_patients']} pts, onset {g['avg_onset_y']}y, delay {g['avg_dx_delay_y']}y")
    defs = get_definitions()
    print(f"\n=== DEFINITIONS: {len(defs['definitions'])} terms ===")
