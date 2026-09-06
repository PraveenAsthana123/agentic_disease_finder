"""
NHLRC1 Epilepsy — Lafora Disease Type 2 (EPM2B / Malin E3 Ubiquitin Ligase / Progressive Myoclonic Epilepsy)
===============================================================================================================
40-patient cohort · NHLRC1 (6p22.3) · Autosomal recessive LOF
NHLRC1 = NHL Repeat Containing 1 (EPM2B); encodes MALIN — a RING-type E3 ubiquitin ligase
LOF → impaired glycogen ubiquitination → abnormal polyglucosan chain accumulation
→ Lafora bodies (intracellular PAS-positive polyglucosan inclusions)
→ progressive neuronal dysfunction → myoclonic epilepsy → fatal encephalopathy

NHLRC1 PROTEIN BIOLOGY (MALIN):
NHLRC1 (6p22.3), NHL Repeat Containing 1 / EPM2B:
  - 395 amino acids, ~42 kDa
  - Domain structure:
      RING domain (aa 1-60): RING-H2 variant E3 ubiquitin ligase catalytic domain;
        recruits E2 ubiquitin-conjugating enzyme → transfers ubiquitin to substrate
      6 × NHL repeat domains (aa ~100-395): tandem NHL propeller repeats;
        substrate recognition / protein-protein interaction scaffold;
        NHL repeats bind LAFORIN (EPM2A) and ubiquitination targets
  - Obligate functional partner: LAFORIN (EPM2A) — phosphatase; binds glycogen via
      carbohydrate-binding module (CBM); removes phosphate from polyglucosan;
      forms MALIN-LAFORIN E3 complex
  - Substrates ubiquitinated by MALIN:
      Glycogen synthase (GYS1) — K48-polyubiquitin → proteasomal degradation → reduces glycogen synthesis
      Protein targeting to glycogen (PTG / PPP1R3C) — limits glycogen-bound PP1 → reduces GYS1 activation
      STBD1 (starch-binding domain protein 1) — autophagic glycogen delivery regulated
      AGL (debranching enzyme) — modulates glycogen branching indirectly
      R5/PTG (PP1 regulatory subunit 3C) — phosphatase targeting to glycogen
  - Function: MALIN-LAFORIN complex = master glycogen quality control checkpoint;
      maintains soluble, branched glycogen structure; prevents polyglucosan (long, unbranched
      insoluble chains) accumulation; LOF → GYS1 not degraded → excess glycogen synthesis
      → phosphorylated, poorly branched polyglucosan → LAFORA BODIES in neurons, muscle,
      liver, heart (neurons most vulnerable — lack glucagon-mediated glycogen mobilisation)

LAFORA DISEASE — PROGRESSIVE MYOCLONIC EPILEPSY TYPE 2:
  Onset: age 10-18 years; previously normal development (most critical diagnostic feature)
  Cardinal features:
    (1) Progressive myoclonic epilepsy — cortical myoclonus + myoclonic seizures, WORSENING over years
    (2) Visual seizures — occipital-origin; flickering phosphenes/zigzag patterns; photosensitivity
    (3) GTCS — generalised tonic-clonic seizures
    (4) Rapid, relentless cognitive decline — dementia-like, years after onset
    (5) LAFORA BODIES — PAS-positive polyglucosan inclusions in neurons, sweat glands
  Epidemiology: ~1:300,000-1,000,000; higher prevalence in Mediterranean (Spain, Italy), South Asia
    (India, Pakistan), Middle East (Turkey, Iran), North Africa; founder effects in specific populations
  Genetics: Autosomal recessive; EPM2A (laforin, OMIM *607566) — Type 1;
    NHLRC1/EPM2B (malin, OMIM *608072) — Type 2; compound heterozygous common; Type 2 may be
    slightly milder progression than Type 1 (Gomez-Abad 2005), though both invariably fatal

CONTRAINDICATIONS UNIQUE TO NHLRC1 / LAFORA DISEASE:
  1. CARBAMAZEPINE / OXCARBAZEPINE / PHENYTOIN — ABSOLUTE CI:
     Na-channel blockers acutely WORSEN myoclonic seizures in Lafora disease (mechanism: interneuron
     disinhibition → myoclonic circuit disinhibition); paradoxical myoclonic worsening in all PME;
     in Lafora disease, worsening may accelerate disease trajectory; IV fosphenytoin ABSOLUTELY
     CONTRAINDICATED for status epilepticus in Lafora disease — use IV LEV + BDZ instead
  2. LAMOTRIGINE MONOTHERAPY — HIGH RISK: LTG Na-channel blockade → myoclonic worsening; LTG also
     elevates peak glutamate release in cortical circuits → aggravates visual cortex excitability
     (Lafora visual seizures); if adjunct LTG required for focal component, combine ONLY with VPA
  3. TIAGABINE — ABSOLUTE CI: GAT-1 block → GABA-A desensitisation → NCSE; in Lafora disease,
     GABA-A receptor interneuron density progressively declines → amplified NCSE risk as disease advances
  4. VIGABATRIN — AVOID: irreversible visual field defects (retinal GABA-T) + may worsen myoclonic;
     Lafora disease has early visual cortex involvement → VGB visual toxicity particularly harmful
  5. VPA + POLG1 — ABSOLUTE CI: Lafora patients not pre-screened; biallelic POLG1 + VPA = fatal
     Alpers-Huttenlocher; POLG1 screen mandatory before VPA in ALL patients
  6. Abrupt AED withdrawal — ABSOLUTE: severe myoclonic status epilepticus triggered by abrupt
     withdrawal in PME/Lafora; taper all AEDs extremely slowly (>2% per week)

GENETICS:
  Gene:        NHLRC1 / EPM2B (6p22.3) — NHL Repeat Containing 1; encodes MALIN
  Protein:     395 aa ~42 kDa; RING-H2 E3 ubiquitin ligase; 6 × NHL repeat substrate-binding domains
  Inheritance: Autosomal recessive LOF; biallelic mutations required
  pLI:         0.14 (recessive constraint; homozygous LOF viable in non-neural tissues)
  OMIM:        *608072 (NHLRC1 gene); #254780 (Lafora disease)
  Incidence:   ~1:300,000-1,000,000; higher in consanguineous Mediterranean/South-Asian populations
  First report: Chan EM et al. 2003 (Nature Genetics) — NHLRC1 as second Lafora disease locus
  Fatal:       Universal; median survival 5-10 years from seizure onset; death ~20-30s (aspiration,
               status epilepticus, cardiorespiratory failure)
  Cure:        None; symptomatic management only; ASO/gene therapy research phase

KEY STANDARDS:
  ILAE-2022  PME classification and genetic testing guidelines
  NICE-NG217 Epilepsy management in children and adults (UK)
  Turnbull-2016 Lafora disease management consensus review
  CPIC-POLG1-2023 VPA contraindication in POLG1 carriers
  MHRA-VPPP-2021 Valproate Pregnancy Prevention Programme
  WHO-ICF-2019 Function and disability classification
  Chan-2003 NHLRC1 discovery paper
  Minassian-1998 EPM2A discovery
  Delgado-Escueta-2001 PME genetics framework
  Mukherjee-2019 Perampanel evidence in Lafora disease

KEY PHARMACOLOGICAL DISTINCTIONS:
  1. CBZ/OXC/PHT ABSOLUTE CI — PARADOXICAL MYOCLONIC WORSENING: Na-channel blockade reduces
     cortical interneuron firing → disinhibition of myoclonic circuits → catastrophic worsening;
     documented in all PMEs but particularly severe in Lafora disease (progressive interneuron loss)
  2. VPA + LEV + CLB = PME BACKBONE: valproate (GABA-T inhibition + Na-channel + HDAC) + LEV
     (SV2A modulation) + clobazam (BDZ allosteric GABA-A) provides triple-mechanism anti-myoclonic
     coverage; VPA most important (broadest spectrum); synergistic when combined
  3. PERAMPANEL AMPA RATIONALE: Lafora cortical myoclonus driven by excess corticothalamic AMPA
     receptor-mediated excitatory drive; perampanel (non-competitive AMPA antagonist) reduces
     cortical excitability; Level B evidence (Mukherjee 2019 series: 65% myoclonic reduction);
     maximum 12 mg/day; aggression monitoring in adolescents
  4. PIRACETAM ANTI-MYOCLONIC MECHANISM: positive allosteric AMPA modulator (paradox: facilitates
     synaptic AMPA but improves cortical synchrony); anti-myoclonic in high dose (20-45 g/day);
     specific for cortical myoclonus; no anticonvulsant for GTCS; combine with VPA
  5. METFORMIN EMERGING GLYCOGEN THERAPY: metformin activates AMPK → inhibits GYS1 (glycogen
     synthase) → reduces glycogen synthesis → slows Lafora body formation; demonstrated in mouse
     models (Berthier 2016); human trial phase; may slow disease progression (not seizure control)
     — disease-modifying rather than anti-seizure; dose: 500-2000 mg/day; monitor renal function
  6. RAPAMYCIN mTOR PATHWAY EMERGING: mTOR drives GYS1 activity; rapamycin → mTOR inhibition →
     reduced GYS1 → slowed polyglucosan accumulation; additive with metformin (different pathways);
     experimental/research phase; may be combined with ASO approaches in future
  7. LTG HIGH RISK — VISUAL CORTEX AGGRAVATION: LTG may worsen Lafora visual cortex seizures
     (flickering/photosensitive occipital bursts) by paradoxical cortical disinhibition at visual
     cortex; if LTG required (focal adjunct), ALWAYS combine with VPA (VPA partially counteracts
     LTG myoclonic aggravation); NEVER LTG monotherapy in Lafora disease
  8. KD MECHANISTIC RATIONALE (DUAL): (a) ketone bodies → AMPK activation → GYS1 inhibition →
     reduced polyglucosan (analogous to metformin mechanism); (b) glucose restriction → less
     glycogen substrate → fewer Lafora bodies; modified Atkins diet preferred (easier compliance
     in adolescents); may provide both symptomatic (anti-seizure) and disease-modifying benefits
  9. PHOTOSENSITIVITY MANAGEMENT — OCCIPITAL SPECIFICITY: Lafora photosensitivity is OCCIPITAL
     CORTEX-mediated (Lafora bodies in occipital neurons early); phosphene aura (flickering zigzag
     lights) = occipital seizure; VPA + perampanel reduces photoparoxysmal response; blue-light
     (450nm) filtering lenses; no stroboscopic environments; screen time management; television
     viewing distance ≥2m
  10. SKIN BIOPSY GOLD STANDARD DIAGNOSIS: Lafora bodies in eccrine sweat gland DUCTS (not cells)
      in axillary skin biopsy — PAS-positive, diastase-resistant, round inclusions 3-40 µm;
      pathognomonic for Lafora disease; should precede genetic testing for cost efficiency;
      biopsy + genetic testing together for definitive diagnosis; positive biopsy + compatible
      phenotype = begin AED without waiting for genetics in acute presentation
"""

import random
random.seed(98765)  # reproducible

# ─────────────────────────────────────────────
# PATIENT SAMPLE — 40 synthetic patients
# ─────────────────────────────────────────────
PATIENT_SAMPLE = []
_subtypes = [
    ("NHLRC1-RING-Domain-Missense", 0.40),
    ("NHLRC1-Truncating-Frameshift-Nonsense", 0.30),
    ("NHLRC1-Compound-Heterozygous", 0.20),
    ("NHLRC1-Exon-Deletion-CNV", 0.07),
    ("Phenocopy-EPM2A-Negative-Lafora-like", 0.03),
]
_sex_dist = {
    "NHLRC1-RING-Domain-Missense": 0.50,
    "NHLRC1-Truncating-Frameshift-Nonsense": 0.50,
    "NHLRC1-Compound-Heterozygous": 0.50,
    "NHLRC1-Exon-Deletion-CNV": 0.50,
    "Phenocopy-EPM2A-Negative-Lafora-like": 0.50,
}
# Onset mean/sd in months (~14y = 168m; range 10-18y)
_onsets = {
    "NHLRC1-RING-Domain-Missense": (168, 18),
    "NHLRC1-Truncating-Frameshift-Nonsense": (162, 20),
    "NHLRC1-Compound-Heterozygous": (175, 15),
    "NHLRC1-Exon-Deletion-CNV": (170, 22),
    "Phenocopy-EPM2A-Negative-Lafora-like": (180, 24),
}

idx = 0
for sub, prop in _subtypes:
    n = round(40 * prop)
    for i in range(n):
        idx += 1
        female = random.random() < _sex_dist.get(sub, 0.5)
        mu, sd = _onsets[sub]
        onset = max(120, min(216, int(random.gauss(mu, sd))))  # clamp 10-18 years
        yrs_since_onset = random.randint(1, 8)
        disease_stage = (
            "early" if yrs_since_onset <= 2
            else "middle" if yrs_since_onset <= 5
            else "late"
        )
        consanguineous = sub in ("NHLRC1-RING-Domain-Missense", "NHLRC1-Truncating-Frameshift-Nonsense") and random.random() < 0.45
        biopsy_positive = sub != "Phenocopy-EPM2A-Negative-Lafora-like"
        on_vpa = random.random() < 0.85
        on_lev = random.random() < 0.72
        on_clb = random.random() < 0.60
        on_perampanel = disease_stage != "early" and random.random() < 0.45
        on_piracetam = random.random() < 0.35
        on_kd = disease_stage == "early" and random.random() < 0.30
        on_metformin = random.random() < 0.20
        polg1_screened = on_vpa or random.random() < 0.80
        cognitive_decline = disease_stage in ("middle", "late")
        vision_impaired = disease_stage == "late" or random.random() < 0.15
        PATIENT_SAMPLE.append({
            "id": f"NHLRC1-{idx:03d}",
            "subtype": sub,
            "sex": "F" if female else "M",
            "age_onset_months": onset,
            "years_since_onset": yrs_since_onset,
            "disease_stage": disease_stage,
            "consanguineous": consanguineous,
            "biopsy_positive": biopsy_positive,
            "drug_resistant": True,  # ALL Lafora patients become DRE
            "on_vpa": on_vpa,
            "on_lev": on_lev,
            "on_clb": on_clb,
            "on_perampanel": on_perampanel,
            "on_piracetam": on_piracetam,
            "on_kd": on_kd,
            "on_metformin": on_metformin,
            "polg1_screened": polg1_screened,
            "cognitive_decline": cognitive_decline,
            "vision_impaired": vision_impaired,
        })

while len(PATIENT_SAMPLE) < 40:
    idx += 1
    yrs = random.randint(1, 6)
    PATIENT_SAMPLE.append({
        "id": f"NHLRC1-{idx:03d}",
        "subtype": "NHLRC1-RING-Domain-Missense",
        "sex": "F" if random.random() < 0.5 else "M",
        "age_onset_months": max(120, int(random.gauss(168, 18))),
        "years_since_onset": yrs,
        "disease_stage": "early" if yrs <= 2 else "middle" if yrs <= 5 else "late",
        "consanguineous": random.random() < 0.40,
        "biopsy_positive": True,
        "drug_resistant": True,
        "on_vpa": random.random() < 0.85,
        "on_lev": random.random() < 0.70,
        "on_clb": random.random() < 0.58,
        "on_perampanel": random.random() < 0.38,
        "on_piracetam": random.random() < 0.30,
        "on_kd": random.random() < 0.22,
        "on_metformin": random.random() < 0.18,
        "polg1_screened": random.random() < 0.82,
        "cognitive_decline": yrs > 2,
        "vision_impaired": yrs > 5 or random.random() < 0.12,
    })

PATIENT_SAMPLE = PATIENT_SAMPLE[:40]

# ─────────────────────────────────────────────
# ETIOLOGY CATALOG
# ─────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "NHLRC1 RING-Domain Missense (LOF-RING-catalytic-E3-activity)",
        "n": 16,
        "pct": 40,
        "variants": "Missense in RING-H2 domain (aa 1-60): C26S, C29G, H58Y, D146N, W219G (NHL-repeat junction); abolish E3 ubiquitin ligase catalytic activity or LAFORIN interaction; most pathogenic variants in Mediterranean (Spain: W219G founder) and South Asia (India/Pakistan: V85E, D146N)",
        "phenotype": "Classic Lafora disease onset 12-16 years; initially normal cognition; myoclonic + visual seizures; rapid cognitive decline 2-3 years after onset; PAS-positive skin biopsy positive in 95%+ at diagnosis",
        "mechanism": "RING domain LOF → MALIN E3 activity abolished → GYS1/PTG/R5 not ubiquitinated → proteasomal degradation fails → GYS1 constitutively active → excess glycogen synthesis → polyglucosan chains (long, unbranched) → Lafora bodies in neurons/sweat glands/heart/muscle",
        "prognosis": "Progressive invariably fatal; median survival 5-8 years from onset; cognitive trajectory faster than EPM2A/Type 1 in some series (Gomez-Abad 2005); death from aspiration/SE/cardiorespiratory failure",
    },
    {
        "category": "NHLRC1 Truncating — Frameshift / Nonsense (LOF-null-biallelic)",
        "n": 12,
        "pct": 30,
        "variants": "Frameshift (insertion/deletion): c.205insC, c.785_786delAG; Nonsense: Q104*, W219*, R311*; all produce premature stop codon → NMD → absent protein; homozygous or compound heterozygous; consanguineous families in North Africa, Turkey, Iran",
        "phenotype": "Earlier onset (10-13 years) and more rapid progression than missense; severe myoclonic storms within 2 years; visual seizures prominent; cognitive decline accelerated; Lafora bodies dense in skin biopsy",
        "mechanism": "Null alleles → complete MALIN absence → no GYS1/PTG ubiquitination → maximal glycogen synthesis dysregulation → Lafora body formation most severe",
        "prognosis": "Worst prognostic subgroup; death typically 5-8 years from onset; may reach vegetative state within 7 years; aggressive palliative planning from diagnosis",
    },
    {
        "category": "NHLRC1 Compound Heterozygous (missense + truncating biallelic)",
        "n": 8,
        "pct": 20,
        "variants": "One missense allele (partial LOF) + one truncating allele (null); e.g., D146N/c.785delAG; total MALIN activity ~5-15% residual; intermediate severity; common in non-consanguineous European and South Asian populations",
        "phenotype": "Onset 13-17 years; intermediate progression between homozygous truncating and homozygous missense; visual seizures + myoclonic + GTCS; cognitive decline moderate pace in first 3 years then accelerates",
        "mechanism": "Residual MALIN activity from missense allele (~10%) provides some GYS1/PTG ubiquitination → slower Lafora body accumulation than biallelic truncating; natural history intermediate",
        "prognosis": "Slightly longer survival (6-10 years from onset) compared to biallelic truncating; same ultimate fatal trajectory; intermediate Lafora body density on biopsy",
    },
    {
        "category": "NHLRC1 Exon Deletion / CNV",
        "n": 3,
        "pct": 7,
        "variants": "Whole-exon deletions detected by MLPA/array-CGH: exon 1-2 deletion (complete gene ablation); single-exon in-frame deletions (partial LOF if NHL repeat); CNV in ~5-7% of Lafora patients not diagnosed by point mutation sequencing",
        "phenotype": "Clinically indistinguishable from truncating LOF; requires CNV-aware testing (standard Sanger/next-gen sequencing misses exon deletions); important consideration when one pathogenic allele found and second allele 'missing'",
        "mechanism": "Exonic deletion → frameshifted or absent MALIN protein (large deletions) or partial domain loss (in-frame deletions affecting NHL repeats → substrate recognition failure without complete RING loss)",
        "prognosis": "Similar to truncating LOF (complete deletions); in-frame NHL-repeat deletions may have slightly slower progression; genetic testing strategy: MLPA/CNV analysis mandatory when only one pathogenic allele found",
    },
    {
        "category": "Phenocopy / EPM2A-Negative Lafora-like (NHLRC1-negative atypical PME)",
        "n": 1,
        "pct": 3,
        "variants": "Clinical Lafora disease phenotype; NHLRC1 sequencing negative; EPM2A (laforin) also negative; DDx: (a) deep intronic/regulatory NHLRC1 variant not captured; (b) NHLRC1/EPM2A large deletion missed; (c) atypical DRPLA; (d) neuronal ceroid lipofuscinosis (CLN); (e) ultra-rare PRDM8 PME",
        "phenotype": "Lafora-like adolescent PME; skin biopsy may be negative or equivocal; genetic testing non-diagnostic on standard panel; brain biopsy or ultra-deep RNA sequencing may be required",
        "mechanism": "Incomplete genetic diagnosis; atypical PME with polyglucosan accumulation may occur via alternative glycogen metabolism gene defects not yet catalogued",
        "prognosis": "Uncertain without genetic diagnosis; presume Lafora disease management until alternative confirmed; RNA sequencing + CNV + deep intronic analysis recommended",
    },
]

# ─────────────────────────────────────────────
# SEIZURE TYPES
# ─────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Cortical Myoclonus — Myoclonic Seizures (Stimulus-Sensitive)",
        "frequency_pct": 95,
        "eeg": "Irregular polyspike-wave (PSW) bursts 3-5 Hz; cortical correlate time-locked to myoclonic jerk (back-averaging shows cortical PSW 15-30ms before muscle jerk); occipital-parietal predominance; stimulus-triggered bursts with photic stimulation; progressive background slowing with disease",
        "semiology": "Sudden, brief (50-200ms) bilateral or multifocal muscle jerks; arms > legs; triggered by voluntary movement (action myoclonus), photic stimulation, touch, emotion; amplitude increases with disease progression; may prevent purposeful hand use (writing, eating) — major disability; morning clustering (post-arousal myoclonus); status myoclonicus precipitated by CBZ/LTG",
        "clinical_tip": "CRITICAL: myoclonic worsening = #1 clinical red flag for wrong AED; if myoclonus suddenly worsens → assume new Na-channel blocker prescribed; take FULL medication history including any ER prescriptions; myoclonic status epilepticus in Lafora = life-threatening; IV LEV + IV BDZ (NOT fosphenytoin) for status",
    },
    {
        "type": "Visual Focal Seizures — Occipital Epilepsy (Photosensitive)",
        "frequency_pct": 85,
        "eeg": "Posterior (occipital-parietal) rhythmic spike-wave or fast activity at onset; photoparoxysmal response (PPR) to intermittent photic stimulation in ~80% of Lafora patients; occipital spikes/PSW at rest; may spread anteriorly to FBTCS",
        "semiology": "Aura: flickering phosphenes, zigzag lines, coloured circles, formed visual hallucinations (animals, faces); brief (15-60 sec); may evolve to head/eye deviation, nausea, ictal vomiting; intense photosensitivity to TV/computer screens, sunlight flickering through trees; visual aura = clinical HALLMARK of Lafora — distinguish from other PMEs (Unverricht-Lundborg has no visual aura)",
        "clinical_tip": "Lafora visual aura is PATHOGNOMONIC among PMEs — presence of visual aura + myoclonus in adolescent strongly suggests Lafora disease (not idiopathic JME which lacks structured visual aura). Occipital seizure from TV/screens at onset: document screen distance, ambient lighting, frame rate. Blue-light filtering essential. Perampanel particularly effective for visual cortex seizures via AMPA blockade.",
    },
    {
        "type": "Generalised Tonic-Clonic Seizures — GTCS / FBTCS",
        "frequency_pct": 78,
        "eeg": "Generalised high-amplitude polyspike-wave pre-ictally; ictal generalised fast activity → spike-wave; post-ictal generalised suppression; often preceded by myoclonic build-up (jerk series → GTCS); may have focal (occipital) onset → FBTCS",
        "semiology": "Full convulsive seizure; often preceded by myoclonic series (crescendo) then GTCS; common at night; post-ictal confusion may be prolonged (30-60 min); may be presenting seizure before myoclonic pattern recognised; bite wound, incontinence, injury common; GTCS frequency ≤2/month initially, increasing with disease",
        "clinical_tip": "Isolated GTCS in adolescent often triggers CBZ/OXC prescription by ER — CATASTROPHIC ERROR in Lafora disease (worsens myoclonus). If adolescent presents with GTCS + visual aura history + myoclonic jerks → presume PME, DO NOT prescribe CBZ/OXC; use LEV + VPA; arrange skin biopsy + EEG urgently.",
    },
    {
        "type": "Absence-like Spells / Atypical Absence",
        "frequency_pct": 42,
        "eeg": "Irregular 2-3 Hz generalised spike-wave; NOT typical 3Hz absence pattern; may have associated myoclonic component (myoclonic absence); background theta slowing; exacerbated by CBZ/OXC/LTG",
        "semiology": "Brief staring (5-20 sec) + unresponsiveness; less discrete than typical CAE; often accompanied by subtle eyelid or perioral myoclonus; not purely generalised (occipital involvement creates visual staring); transition from absence-like → myoclonic burst common; easily confused with typical absence in early disease",
        "clinical_tip": "Presence of myoclonic jerks + occipital seizure history differentiates Lafora absence-like spells from typical childhood absence epilepsy (CAE). Ethosuximide (standard for CAE) is INSUFFICIENT for Lafora disease and does not address GTCS or myoclonus; VPA is essential. Do NOT use ESM monotherapy.",
    },
    {
        "type": "Drop Attacks / Epileptic Falls (Late Disease)",
        "frequency_pct": 35,
        "eeg": "Sudden atonic or myoclonic-atonic seizure pattern; generalised burst-suppression with disease progression; diffuse PSW → sudden posture loss; falls may be clonic-atonic sequence",
        "semiology": "Sudden falls without warning; may be myoclonic-atonic (brief jerk then drop) or pure atonic; high injury risk (head trauma, fractures); protective equipment (helmet, padding) mandatory in late disease; may present as apparent stumbling/clumsiness before frank falls; seizure frequency high in late Lafora → SUDEP risk",
        "clinical_tip": "Drop attacks in Lafora late disease = indicator of significant disease progression and SUDEP risk elevation. Falls risk assessment mandatory; home safety audit; VNS may reduce fall burden (salvage therapy when AED maximal); palliative care involvement appropriate at onset of drop attacks.",
    },
]

# ─────────────────────────────────────────────
# TRIGGERS
# ─────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Photic Stimulation — Screen / Flicker / Sunlight",
        "pct": 92,
        "mechanism": "Occipital Lafora body deposition → posterior cortex hyperexcitability → photoparoxysmal response (PPR) to 15-20 Hz flicker range; television (interlaced display 25-30 Hz), computer/phone screens, sunlight through moving trees, strobe lights, video games — all trigger occipital seizure → myoclonic burst or GTCS",
        "management": "Blue-light filtering glasses (wavelength 450nm block); TV viewing distance ≥2m; ambient room lighting during screen use; non-interlaced displays (LCD/OLED at ≥60 Hz native); avoid strobe light environments; UV-protective lenses outdoors; VPA + perampanel reduces PPR significantly; carers trained to cover eyes during exposure if seizure occurs",
    },
    {
        "trigger": "Sleep Deprivation",
        "pct": 85,
        "mechanism": "Sleep deprivation → increased cortical excitability → lowered myoclonic threshold; morning post-arousal myoclonus especially sensitive to prior night sleep quality; Lafora disease progressively impairs sleep architecture (PSW during NREM) worsening this loop",
        "management": "Strict sleep schedule (same time daily); melatonin 5-10mg for sleep maintenance; avoid late-night screen use; carers monitor for nocturnal myoclonic seizures; CLB at bedtime for nocturnal cluster prevention; VPA evening dose optimised",
    },
    {
        "trigger": "Voluntary Movement / Action Myoclonus",
        "pct": 80,
        "mechanism": "Cortical (somatosensory-motor) myoclonus: voluntary movement → cortical discharge → post-movement myoclonic jerk; action myoclonus = movement-induced cortical reflex myoclonus; most disabling trigger (prevents feeding, writing, self-care); becomes dominant symptom in middle-late disease",
        "management": "OT assessment and adaptive equipment (weighted utensils, non-slip mats); CLB + piracetam best for action myoclonus; VPA + LEV backbone; perampanel may reduce action myoclonus amplitude; assess functional impact on ADL regularly; consider nasogastric tube in late disease",
    },
    {
        "trigger": "Emotional Stress / Startling",
        "pct": 75,
        "mechanism": "Startle reflex → cortical hyperexcitability → myoclonic response; emotional arousal (fear, excitement, anger) → amygdalar activation → cortical myoclonic circuit sensitisation; Lafora patients develop anticipatory anxiety about seizures → psychological stress itself triggers seizures (vicious cycle)",
        "management": "Stress management counselling (CBT); carers trained not to startle patient; quiet low-stimulation home environment in later stages; CLB adjunct for anxiety-driven seizure clusters; family psychoeducation on trigger avoidance; avoid surprise noise (doorbells, alarms)",
    },
    {
        "trigger": "Missed AED Dose",
        "pct": 68,
        "mechanism": "Sub-therapeutic AED trough levels → rebound myoclonic excitability; VPA trough level drop particularly destabilising (short half-life if immediate-release); Lafora disease pharmacokinetics less studied — VPA chronic-release preferred; adolescent adherence challenges (school schedule, social stigma)",
        "management": "VPA-CR (controlled-release) preferred over immediate-release (steadier trough); blister-pack dispensing; school nurse medication administration record; family caregiver supervision; VPA TDM if adherence uncertain; LEV-XR for smoother trough; school seizure action plan",
    },
    {
        "trigger": "Intercurrent Illness / Fever",
        "pct": 60,
        "mechanism": "Fever → neuronal excitability increase; Lafora disease impairs brain metabolic reserve → febrile illness precipitates myoclonic storms; metabolic stress (dehydration, electrolyte imbalance) compounds AED kinetics alteration",
        "management": "Early antipyretic (paracetamol from 37.5°C threshold — lower than general recommendation); maintain AED adherence during illness (use syrup/IV formulations if vomiting); CLB rescue during febrile illness; ER plan for status myoclonicus; document Lafora diagnosis on ER care card (do NOT give CBZ/OXC/PHT)",
    },
    {
        "trigger": "AED Dose Change / Taper",
        "pct": 48,
        "mechanism": "Rate of AED level change (not just absolute level) triggers withdrawal myoclonic status; abrupt VPA withdrawal = highest risk in PME; even small reductions may provoke myoclonic storm in Lafora disease",
        "management": "Taper rate MAXIMUM 5-10% every 4 weeks (slower than general recommendations); NEVER abruptly stop any AED in Lafora disease; consultant-supervised tapering only; rescue BDZ (diazepam rectal 10mg or midazolam buccal 10mg) available at home at all times; taper during school holiday to allow monitoring",
    },
    {
        "trigger": "Exercise / Physical Exertion",
        "pct": 30,
        "mechanism": "Exercise-induced myoclonic worsening via: (a) metabolic acidosis (lactic acid → pH change → ion channel kinetics); (b) glucose consumption → temporary hypoglycaemia (Lafora glycogen metabolism impaired — neurons cannot mobilise glycogen reserves); (c) fatigue → reduced cortical inhibitory reserve",
        "management": "Gradual exercise introduction; avoid competitive sports with SUDEP risk (swimming unsupervised, cycling in traffic); water-based exercise with 1:1 supervision; glucose snack before exercise; VPA with food to avoid post-exercise level drop; exercise diary to correlate myoclonic events",
    },
]

# ─────────────────────────────────────────────
# TREATMENTS
# ─────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproate / Sodium Valproate (Epilim CR / Depakote ER) — First-Line Backbone",
        "evidence": "Level B (PME backbone; ILAE 2022 PME guideline; most evidence in Lafora disease among all AEDs; controlled-release preferred for trough stability)",
        "dose": "Adult: 1500-3000 mg/day CR (divided BID); paediatric: 30-60 mg/kg/day CR; higher end required in Lafora (hepatic metabolism accelerated in adolescence); target VPA TDM 70-120 mcg/mL (higher range than typical; needed for myoclonic control)",
        "moa": "Multiple mechanisms: (1) GABA-T inhibition → increased synaptic GABA; (2) Na-channel slow inactivation; (3) T-type Ca²⁺ channel inhibition → reduces thalamo-cortical resonance; (4) HDAC inhibition → chromatin remodelling; (5) mild mTOR pathway modulation → potential modest reduction in GYS1 activity (HDAC-mTOR crosstalk; indirect Lafora body benefit unproven in humans)",
        "efficacy": "Best anti-myoclonic agent in Lafora disease; 60-75% reduction in myoclonic seizures when therapeutic; combination with LEV + CLB achieves 80-90% myoclonic control in early disease; response diminishes with disease progression (increasing Lafora body burden)",
        "monitoring": "VPA TDM (target 70-120 mcg/mL total); POLG1 screen MANDATORY before initiation; LFTs monthly × 3 then q6m; CBC (thrombocytopenia risk); NH₃ (hyperammonaemia — check if encephalopathy); weight gain (VPA); VPPP programme for females of childbearing potential",
        "nhlrc1_note": "LAFORA-SPECIFIC DOSING: adolescent rapid hepatic VPA metabolism requires higher mg/kg doses than adults; controlled-release formulation ESSENTIAL (smoother trough — immediate-release peaks may cause somnolence, troughs trigger breakthrough myoclonus); if switching from immediate-release → CR, TDM after 1 week to confirm equivalent exposure. VPA + METFORMIN: no pharmacokinetic interaction; may provide additive disease-modifying benefit (VPA HDAC effect + metformin AMPK). POLG1 screen: ALWAYS before VPA in Lafora (not pre-screened population).",
    },
    {
        "drug": "Levetiracetam (Keppra / Keppra XR) — Anti-Myoclonic Adjunct",
        "evidence": "Level B (broad-spectrum PME adjunct; SV2A modulation; good evidence for myoclonic seizures in JME extrapolated to Lafora disease PME series)",
        "dose": "Paediatric: 20-60 mg/kg/day; adult: 1000-4000 mg/day; BID; XR (extended-release) preferred for trough stability; IV available for status; 1000-2000 mg/day starting dose with food",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulation → reduces vesicle fusion priming → decreased neurotransmitter release from hyperactive corticothalamic terminals; SV2A highly expressed in pyramidal neurons driving cortical myoclonus",
        "efficacy": "Additive to VPA for myoclonic seizures (60-70% responders in PME); reduces GTCS frequency; does NOT reduce VPA levels (minimal CYP interaction) — ideal backbone combination; less effective as monotherapy than VPA",
        "monitoring": "Behavioural side effects (LEV-induced irritability/aggression — particularly in adolescents with Lafora-associated depression/anxiety); monitor behavioural baseline; LEV-XR reduces peak-level behavioural toxicity; CBC (rare); renal function (renal excretion — reduce dose in CKD)",
        "nhlrc1_note": "PME BEHAVIOURAL RISK: Lafora patients develop progressive cognitive decline and depression; LEV may exacerbate emotional lability; LEV-XR (extended-release) significantly reduces behavioural adverse effects vs immediate-release; if LEV behavioural toxicity confirmed → switch to perampanel or piracetam as adjunct; never abruptly discontinue LEV in Lafora disease.",
    },
    {
        "drug": "Clobazam (Onfi / Frisium) — Adjunct BDZ Anti-Myoclonic",
        "evidence": "Level B (adjunct for myoclonic seizures; BDZ positive allosteric modulation; good Lafora disease tolerability relative to clonazepam; less sedation)",
        "dose": "Adult: 10-40 mg/day; paediatric: 0.1-1.0 mg/kg/day; BID (max 40 mg/day); start low (5 mg nocte) and titrate weekly; nocturnal dosing reduces daytime sedation",
        "moa": "Positive allosteric modulator of GABA-A receptor (α2/α3/α5 subunit preference vs clonazepam α1 preference) → chloride influx → neuronal hyperpolarisation; less sedating than clonazepam; effective at reducing cortical myoclonic threshold",
        "efficacy": "Anti-myoclonic: 50-65% responder rate as adjunct; good tolerability profile; less tachyphylaxis than clonazepam (CLB tolerance develops slower); effective for catamenial myoclonic worsening; CLB + VPA = synergistic anti-myoclonic for Lafora disease",
        "monitoring": "Sedation (dose-dependent); tolerance to anti-myoclonic effect (may develop 6-12 months — periodic drug holidays if possible); respiratory depression risk in combination with other CNS depressants; swallowing safety in late disease (CLB sedation + dysphagia = aspiration risk)",
        "nhlrc1_note": "CLB VS CLONAZEPAM IN LAFORA: clonazepam (CNZ) is more sedating (α1 preference → sedation + ataxia) and develops tolerance faster; CLB preferred as first-choice BDZ in Lafora disease; in late disease when dysphagia develops, CLB available as oral solution for NGT administration; nocturnal CLB dosing reduces morning myoclonic amplitude (key clinical benefit for breakfast/morning ADL).",
    },
    {
        "drug": "Perampanel (Fycompa) — AMPA Antagonist Anti-Myoclonic",
        "evidence": "Level B (emerging; Mukherjee 2019 Lafora disease series: 65% myoclonic reduction; multiple PME case series; ILAE 2022 emerging evidence for PME; FDA approved for myoclonic seizures in JME — basis for Lafora off-label use)",
        "dose": "2-12 mg nocte (start 2mg, titrate by 2mg q2 weeks to target 8-12 mg); nocturnal dosing reduces dizziness; with VPA: maximum 8 mg/day (VPA inhibits perampanel CYP3A4 metabolism → ↑ perampanel levels); lower starting dose with VPA",
        "moa": "Non-competitive AMPA (α-amino-3-hydroxy-5-methyl-4-isoxazolepropionic acid) receptor antagonist → reduces excitatory glutamatergic synaptic drive → directly opposes cortical myoclonic hyperexcitability; AMPA receptors mediate rapid excitatory corticothalamic and corticocortical transmission driving cortical myoclonus",
        "efficacy": "65% reduction in myoclonic seizure frequency (Mukherjee 2019 series n=12); superior to previous adjuncts in some Lafora cases; particularly effective for visual cortex seizures (occipital AMPA overactivity in Lafora); reduces photoparoxysmal response",
        "monitoring": "Dizziness/ataxia (dose-limiting at higher doses); aggression/irritability (AMPA blockade in limbic circuits; monitor in adolescents with Lafora-associated emotional lability); weight gain; VPA-perampanel interaction: VPA inhibits CYP3A4 → ↑perampanel AUC → limit to 8 mg/day maximum with VPA; titrate slowly",
        "nhlrc1_note": "LAFORA-SPECIFIC AMPA RATIONALE: Lafora bodies in occipital cortex create focal AMPA receptor upregulation → occipital photosensitive seizures; perampanel's occipital AMPA antagonism addresses this directly. VPA-PERAMPANEL COMBINATION: maximum perampanel 8mg/day (not standard 12mg) due to VPA CYP3A4 inhibition → perampanel exposure increased ~15-20%; monitor perampanel-related dizziness closely when adding VPA. Perampanel nocturnal dosing reduces daytime ataxia — important for already-impaired gait in Lafora patients.",
    },
    {
        "drug": "Piracetam (Nootropil) — Specific Anti-Myoclonic",
        "evidence": "Level C (Lafora disease and other PME case series; specific anti-myoclonic agent for cortical myoclonus; no RCT in Lafora disease; widespread clinical use in Europe, Canada, India for PME)",
        "dose": "Adult: 20-45 g/day (very high doses); paediatric: 160-320 mg/kg/day; TID-QID; oral or IV; dose-escalate weekly by 5g increments to maximum tolerated dose; higher doses more effective for action myoclonus",
        "moa": "High-dose piracetam facilitates AMPA receptor-mediated synaptic transmission in cortical circuits (positive allosteric AMPA modulator) → paradoxical anti-myoclonic effect by improving cortical synchrony; also platelet anti-aggregatory effect; mechanism for anti-myoclonic effect incompletely understood despite decades of use",
        "efficacy": "Anti-myoclonic specific: reduces cortical myoclonus amplitude and frequency; not effective for GTCS (must be combined with VPA); good evidence for action myoclonus (most disabling Lafora feature); response rate 50-70% as adjunct; higher doses (>30g/day) more effective",
        "monitoring": "Renal excretion — reduce dose in CKD; bleeding risk (platelet anti-aggregation — avoid NSAIDs combination); nervousness/agitation at high doses; weight gain (fluid retention); no hepatic metabolism — safe with VPA without pharmacokinetic interaction; CBC baseline",
        "nhlrc1_note": "ACTION MYOCLONUS PRIORITY: piracetam is the specific anti-myoclonic agent for Lafora action myoclonus (preventing eating, writing, self-care); should be added when action myoclonus becomes functionally disabling (usually within 2-3 years of onset); VPA + LEV + CLB + PIRACETAM = standard full PME regimen; available in many jurisdictions as IV formulation for acute myoclonic status (20g IV over 30 min).",
    },
    {
        "drug": "Zonisamide (Zonegran) — Broad-Spectrum Adjunct",
        "evidence": "Level C (adjunct PME including Lafora disease; broad-spectrum; particularly effective for myoclonic-atonic falls in late disease)",
        "dose": "100-600 mg/day; start 50-100mg nocte, titrate by 50-100mg every 2 weeks; BID; paediatric: 4-12 mg/kg/day; renal excretion (reduce in CKD)",
        "moa": "T-type Ca²⁺ channel blockade → reduces thalamo-cortical oscillations; Na-channel slow inactivation (mild — less than CBZ); carbonic anhydrase inhibition → mild acidosis → anti-myoclonic (via GABA enhancement in acidic environment); weak MAO-B inhibition",
        "efficacy": "Reduces GTCS frequency (60-70%); modest anti-myoclonic effect (40-50%); particularly useful for atonic/myoclonic-atonic seizures (drops) in late Lafora disease; synergistic with VPA + LEV",
        "monitoring": "Renal calculi (carbonic anhydrase inhibition → decreased urinary citrate → stone risk; hydration 2L/day mandatory); oligohidrosis + hyperthermia in children (thermoregulation impaired — avoid hot environments); cognitive slowing; anorexia/weight loss (useful in VPA-related weight gain); mood depression; caution in sulfonamide allergy",
        "nhlrc1_note": "LATE DISEASE DROP ATTACKS: zonisamide most useful when myoclonic-atonic falls become prominent (late Lafora); T-type Ca²⁺ blockade reduces thalamo-cortical myoclonic drive that generates atonic component; also provides modest GTCS control; RENAL STONE RISK: Lafora patients on long-term treatment should have renal US annually if on ZNS; adequate hydration essential (also reduces exercise-triggered seizures).",
    },
    {
        "drug": "Metformin (Glucophage) — Emerging Disease-Modifying Glycogen Therapy",
        "evidence": "Level C (experimental disease-modifying; Berthier 2016 mouse model data; 2023 human pilot data emerging; mechanism-based rationale strong; NOT primarily anti-seizure — targets glycogen accumulation)",
        "dose": "Adult: 500-2000 mg/day with food; paediatric: 500-1500 mg/day; start 500 mg OD → 500 mg BID after 1 week → 1000 mg BID target; extended-release preferred (GI tolerability); adjust for renal function (eGFR monitoring)",
        "moa": "Metformin activates AMPK (AMP-activated protein kinase) via inhibition of mitochondrial complex I → AMPK phosphorylates and INACTIVATES glycogen synthase (GYS1) → reduced glycogen synthesis → reduced polyglucosan substrate → SLOWER LAFORA BODY FORMATION; disease-modifying mechanism independent of AED anticonvulsant action; NOT primarily anti-seizure but may modestly reduce seizure frequency via reduced glycogen accumulation over months-years",
        "efficacy": "Mouse model: metformin + rapamycin reduced Lafora body burden 60-80% and improved neurological function; human pilot data (n=8, 2022): reduced seizure frequency 30-40% over 12 months; full RCT ongoing (ISRCTN); expected to slow disease progression rather than acutely suppress seizures",
        "monitoring": "eGFR before and q3m (hold if eGFR <30; risk of lactic acidosis); GI tolerability (nausea — use with food; CR formulation); vitamin B12 (metformin reduces B12 absorption — check annually); lactic acidosis risk if sepsis/hypoxia (hold during severe illness); iodinated contrast media: hold 24-48h before IV contrast procedures",
        "nhlrc1_note": "DISEASE-MODIFYING RATIONALE — MALIN-AMPK-GYS1 AXIS: MALIN normally ubiquitinates GYS1 for degradation; in NHLRC1 LOF, GYS1 is not degraded → constitutively active → excess polyglucosan; METFORMIN activates AMPK → AMPK phosphorylates GYS1 (Ser7) → allosteric inhibition of GYS1 → REDUCED GLYCOGEN SYNTHESIS → fewer Lafora bodies — BYPASSES the absent MALIN ubiquitination using AMPK as an alternative GYS1 inactivation pathway. This is the strongest disease-modifying rationale for any currently available drug in Lafora disease.",
    },
    {
        "drug": "Ketogenic Diet (Modified Atkins) — Anti-Seizure + Disease-Modifying",
        "evidence": "Level C (PME use including Lafora disease; multiple case reports; dual mechanism — anti-seizure + glycogen reduction; modified Atkins preferred over classical KD in adolescents for compliance)",
        "dose": "Modified Atkins: carbohydrate restriction 20-30g/day net carbs initially → 40-50g/day maintenance; fat-dominant diet; supplement with Ca²⁺, Mg²⁺, Zn²⁺, Se²⁺, B12, vitamin D, folate; classical KD 4:1 ratio if modified Atkins fails; dietitian mandatory",
        "moa": "DUAL MECHANISM: (1) ANTI-SEIZURE: BHB → adenosine A1 receptor activation → presynaptic K⁺ channel hyperpolarisation → reduced excitability; BHB also suppresses mTORC1 via AMPK → TSC2 activation; (2) DISEASE-MODIFYING: glucose restriction → reduced glycogen substrate → less polyglucosan for Lafora body formation; BHB-mediated AMPK → GYS1 inhibition (same mechanism as metformin — potentially additive if combined)",
        "efficacy": "Anti-seizure: 50-65% ≥50% seizure reduction in paediatric DRE (extrapolated); myoclonic reduction documented in JME/Lafora case series; disease-modifying: slower Lafora body accumulation (theoretical + mouse model evidence); clinical response better in early disease",
        "monitoring": "Nutritional deficiency surveillance (carnitine, selenium, zinc, vitamin D, B12); fasting lipid panel (KD elevates LDL); renal calculi (hydration + potassium citrate supplement); growth monitoring in adolescents (critical age for bone mineral density); BHB monitoring (target 2-4 mmol/L); seizure diary",
        "nhlrc1_note": "COMPLIANCE CHALLENGE IN LAFORA ADOLESCENTS: cognitive decline impairs diet self-management; family/carer must manage diet preparation; social isolation risk (school cafeteria, birthday parties); modified Atkins diet more compatible with adolescent social life than classical 4:1 KD; combined KD + METFORMIN may provide additive AMPK-mediated glycogen suppression (BHB + metformin both activate AMPK by different mechanisms — pilot combination strategy emerging in research centres).",
    },
]

# ─────────────────────────────────────────────
# CONTRAINDICATIONS
# ─────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "CARBAMAZEPINE / OXCARBAZEPINE / PHENYTOIN — Na-Channel Blockers",
        "risk_level": "ABSOLUTE",
        "mechanism": "Na-channel blockade (CBZ/OXC/PHT) → preferentially reduces fast-firing interneuron (GABAergic) discharge rate → cortical disinhibition → PARADOXICAL MYOCLONIC WORSENING; in Lafora disease, interneuron loss from progressive Lafora body deposition amplifies this disinhibition; documented cases of myoclonic status epilepticus triggered by CBZ initiation in Lafora patients; IV PHT/fosphenytoin for SE = ABSOLUTE CI → gives Na-channel block IV → acute catastrophic myoclonic worsening",
        "management": "NEVER prescribe CBZ, OXC, or PHT in any patient with Lafora disease or suspected PME with myoclonus; if ER physician prescribes CBZ for seizure → STOP immediately and reverse; for acute SE in Lafora disease: IV levetiracetam (60 mg/kg loading dose) + IV benzodiazepine (lorazepam 0.1 mg/kg); document ABSOLUTE CI in medical record, ER care card, school health record, pharmacy record",
        "nhlrc1_specific": "The most dangerous and most commonly made error in Lafora disease: a previously well adolescent presents to ER with first GTCS, CBZ or OXC prescribed without recognising underlying PME. Within days, myoclonic storms develop. If myoclonus suddenly worsens acutely → FIRST QUESTION: was any new AED prescribed (especially by ER/GP)? Take complete medication list including ER prescriptions. Lafora diagnosis card should explicitly list CBZ/OXC/PHT as ABSOLUTE CI on front face.",
    },
    {
        "drug": "LAMOTRIGINE MONOTHERAPY — High Risk of Myoclonic Worsening",
        "risk_level": "HIGH RISK",
        "mechanism": "LTG Na-channel blockade → interneuron disinhibition → myoclonic worsening (shared mechanism with CBZ/OXC but LTG effect is less severe than CBZ); additionally, LTG at visual cortex → disinhibition of occipital excitatory circuits → worsening photosensitive visual seizures in Lafora; LTG monotherapy documented to precipitate myoclonic status in JME (same PME mechanism applies to Lafora disease)",
        "management": "NEVER use LTG monotherapy in Lafora disease; if LTG required as focal adjunct (for FBTCS control), use ONLY as adjunct to VPA (VPA partially counteracts LTG myoclonic aggravation effect, and VPA doubles LTG levels requiring dose halving of LTG); maximum LTG dose with VPA 100-150 mg/day (VPA inhibits LTG glucuronidation → LTG half-life doubles → use lower LTG doses)",
        "nhlrc1_specific": "LTG is sometimes tried in PME by clinicians unfamiliar with Lafora disease (LTG effective in many generalised epilepsies). In Lafora disease specifically, LTG worsens the visual cortex component — flickering phosphene seizures increase and may evolve to status epilepticus. If LTG was previously prescribed and myoclonus or visual seizures worsened after LTG initiation — discontinue LTG carefully (taper over 4 weeks with VPA and CLB cover) and document LTG HIGH RISK in record.",
    },
    {
        "drug": "TIAGABINE (TGB) — GAT-1 Inhibitor",
        "risk_level": "ABSOLUTE",
        "mechanism": "TGB inhibits GAT-1 (GABA transporter SLC6A1) → excess perisynaptic GABA accumulation → GABA-A receptor desensitisation → paradoxical NCSE; in Lafora disease, progressive GABAergic interneuron dysfunction (Lafora bodies in interneurons) reduces the buffering capacity against GABA-A desensitisation → higher NCSE risk than general population; NCSE in a cognitively declining Lafora patient may be misattributed to disease progression",
        "management": "ABSOLUTE contraindication in Lafora disease; if TGB inadvertently given, continuous EEG monitoring mandatory (NCSE diagnosis requires EEG — cannot diagnose clinically in cognitively impaired patient); treat NCSE with IV BDZ + IV VPA; document TGB CI on all records",
        "nhlrc1_specific": "NCSE DIAGNOSTIC TRAP in Lafora disease: cognitive decline is expected and progressive; NCSE from TGB may present as apparent sudden acceleration of cognitive decline, behavioural change, or increased seizure burden — all attributable to disease rather than drug. EEG is the ONLY way to distinguish TGB-induced NCSE from disease progression. If clinical deterioration is acute and rapid → PERFORM EEG URGENTLY (even if 'expected' disease course).",
    },
    {
        "drug": "VIGABATRIN (VGB) — GABA-T Inhibitor",
        "risk_level": "HIGH RISK",
        "mechanism": "VGB → irreversible retinal GABA-T inhibition → permanent concentric visual field defects (constriction); Lafora disease already impairs VISUAL CORTEX function (Lafora bodies in occipital cortex; photosensitive visual seizures); addition of VGB visual field defect to already-compromised vision = disproportionate functional harm; additionally, VGB may worsen myoclonic seizures (excess GABA → GABA-A desensitisation at high concentrations)",
        "management": "AVOID vigabatrin in Lafora disease; if VGB somehow initiated, ERG and visual field assessment mandatory q3m; stop immediately on first ERG nasal scotoma finding; for infantile spasms (rare Lafora presentation at IS age — extremely atypical), VGB may be considered briefly with ERG monitoring, but never for treatment of Lafora-onset PME seizures",
        "nhlrc1_specific": "Visual function is a critical quality-of-life domain in Lafora disease (visual seizures, visual hallucinations, visual cortex degeneration from Lafora bodies over time). Loss of visual field from VGB in a patient already experiencing visual cortex seizures and occipital Lafora bodies is doubly harmful. Ophthalmology review of current visual status before any drug with visual toxicity potential.",
    },
    {
        "drug": "VPA + POLG1 (without screen)",
        "risk_level": "ABSOLUTE",
        "mechanism": "Biallelic POLG1 variants (mitochondrial DNA polymerase gamma) → mtDNA depletion; VPA inhibits mitochondrial beta-oxidation → precipitates Alpers-Huttenlocher syndrome (acute hepatic failure + encephalopathy + seizures → death); Lafora patients are NOT pre-screened for POLG1; VPA is first-line and essential in Lafora disease — POLG1 screen cannot be bypassed",
        "management": "POLG1 sequencing BEFORE VPA initiation (turnaround 7-14 days); bridge with LEV + CLB during POLG1 wait; if POLG1 biallelic → VPA permanently contraindicated → backbone is LEV + CLB + piracetam + perampanel + ZNS; if POLG1 result unavailable and SE requires VPA urgently → document medical necessity, lowest possible dose, plan POLG1 within 48h",
        "nhlrc1_specific": "VPA is uniquely irreplaceable as the PME backbone in Lafora disease; no single alternative drug provides equivalent anti-myoclonic efficacy. POLG1 screen must NEVER be bypassed. If biallelic POLG1 is found in a Lafora patient, the management challenge is severe — multi-drug PME regimen without VPA requires specialist epilepsy centre coordination. POLG1 carrier status (heterozygous, common) does NOT contraindicate VPA — only biallelic LOF.",
    },
    {
        "drug": "Abrupt AED Withdrawal — Any Agent",
        "risk_level": "ABSOLUTE",
        "mechanism": "Abrupt withdrawal of any AED in Lafora disease → immediate myoclonic status epilepticus risk; PME withdrawal status is extremely severe and may be fatal; Lafora disease hyperexcitable cortex has minimal inhibitory reserve — any sudden reduction in AED levels → catastrophic myoclonic storm; VPA withdrawal particularly dangerous (removes primary GABAergic + Na-channel + HDAC protection simultaneously)",
        "management": "ALL AED tapers in Lafora disease: maximum 5-10% dose reduction every 4-6 weeks (slower than any general recommendation); NEVER withdraw any AED abruptly for any reason; if adherence lapse occurs → urgent consultation; home supply of rescue BDZ (diazepam 10mg rectal or midazolam 10mg buccal) mandatory at all times; written taper schedule agreed with patient and family before any change; ER team briefed on danger of abrupt AED stop in Lafora disease",
        "nhlrc1_specific": "Caregiver training: family must be warned that stopping ANY AED even briefly (missed prescription, vomiting, no pharmacy stock) can cause life-threatening myoclonic status in Lafora disease. Emergency supply at home AND school. If patient vomiting → switch to IV formulations (IV VPA, IV LEV) urgently rather than delaying doses. Medic-Alert bracelet mandatory listing AED regime and ABSOLUTE NO CBZ/OXC/PHT.",
    },
]

# ─────────────────────────────────────────────
# MONITORING
# ─────────────────────────────────────────────
MONITORING = [
    {"item": "Skin Biopsy (Axillary) + PAS Staining — Gold Standard Diagnosis", "frequency": "Once at diagnosis (before genetic result if clinical suspicion high)", "rationale": "PAS-positive, diastase-resistant Lafora bodies in eccrine sweat gland ducts — pathognomonic; positive in >95% of confirmed Lafora; axillary site preferred (higher sweat gland density); electron microscopy shows fibrillar polyglucosan ultrastructure; positive biopsy confirms diagnosis even before genetic result"},
    {"item": "WES-TRIO / NHLRC1 + EPM2A Panel (Lafora Genetics)", "frequency": "Once at diagnosis (after or concurrent with skin biopsy)", "rationale": "Confirms NHLRC1 or EPM2A biallelic pathogenic variants; MLPA/CNV panel mandatory (exon deletions in ~7%); NHLRC1 sequencing first if consanguineous; if standard sequencing negative → deep intronic + RNA-seq; genetic result guides family cascade testing and prenatal counselling"},
    {"item": "EEG — Photoparoxysmal Response + Background Monitoring", "frequency": "At diagnosis; q6m or after seizure change; include photic stimulation protocol", "rationale": "Posterior-predominant polyspike-wave; photoparoxysmal response (80% of Lafora patients); progressive background slowing tracks disease progression; EEG for NCSE if acute cognitive deterioration; nocturnal EEG if nocturnal myoclonic storms suspected"},
    {"item": "POLG1 Screen (before VPA)", "frequency": "Once before VPA initiation (7-14 day turnaround)", "rationale": "Biallelic POLG1 + VPA = Alpers-Huttenlocher fatal hepatotoxicity; POLG1 screen mandatory regardless of Lafora diagnosis (not pre-screened); bridge LEV + CLB during wait"},
    {"item": "VPA TDM + LFTs + CBC + NH₃", "frequency": "Monthly × 3 then q3m; NH₃ if encephalopathic symptoms", "rationale": "Lafora target VPA 70-120 mcg/mL (higher than standard 50-100); LFTs for hepatotoxicity; CBC for thrombocytopenia; NH₃ for hyperammonaemia (common in high-dose VPA); encephalopathic worsening = check NH₃ BEFORE attributing to disease progression"},
    {"item": "Neuropsychological Battery (Cognitive Decline Monitoring)", "frequency": "At diagnosis; q6-12m; Bayley-III (if child), WAIS-IV, VABS-3, MoCA", "rationale": "Cognitive decline is the most disabling Lafora feature after myoclonus; serial neuropsychological assessment tracks decline rate; informs educational placement, driving, employment decisions; helps distinguish NCSE-accelerated decline from disease progression"},
    {"item": "Brain MRI 3T", "frequency": "At diagnosis; repeat if clinical deterioration or focal deficit; annual in late disease", "rationale": "Rules out alternative diagnoses (DRPLA, CLN, structural lesions); non-specific white matter changes in Lafora disease; progressive cortical atrophy in late disease (especially occipital); guides palliative care planning"},
    {"item": "Ophthalmology — Visual Fields + ERG + OCT", "frequency": "At diagnosis; q12m; q3m if on vigabatrin (avoid VGB)", "rationale": "Visual seizures + visual cortex Lafora bodies → visual function monitoring; if perampanel prescribed, assess any visual changes; rule out VGB-related visual field defects if VGB erroneously prescribed; ERG assesses retinal function; OCT tracks retinal thickness"},
    {"item": "Swallowing Assessment (Dysphagia in Late Disease)", "frequency": "When swallowing difficulty reported; annual from year 5 of disease", "rationale": "Lafora late disease → brainstem involvement + cognitive decline → dysphagia → aspiration; aspiration is common cause of death in late Lafora disease; FEES/videofluoroscopy; nasogastric tube planning; texture modification of diet; CLB liquid formulation for dysphagia patients"},
    {"item": "Eeg Monitoring for NCSE if Acute Deterioration", "frequency": "Immediately on any unexplained acute cognitive worsening or seizure burden increase", "rationale": "NCSE may be misattributed to disease progression in Lafora; triggered by TGB/erroneous AED; distinguish NCSE from disease by EEG; continuous EEG monitoring in hospitalised late-disease Lafora patients with frequent seizures"},
    {"item": "Metformin Monitoring (if prescribed)", "frequency": "eGFR before start; q3m; B12 annually; hold during illness/contrast", "rationale": "Lactic acidosis risk if eGFR impaired; B12 malabsorption; GI tolerability monitoring (use CR formulation); hold during sepsis, dehydration, IV contrast procedures"},
    {"item": "SUDEP / Safety Risk Assessment", "frequency": "Annually; after any severe seizure", "rationale": "Lafora disease = high SUDEP risk (frequent GTCS, nocturnal seizures, DRE, progressive disability); bed alarm (Emfit/Embrace2); never sleep alone; prone positioning avoidance; rescue BDZ protocol; VNS consideration for SUDEP risk reduction; nocturnal supervision plan"},
    {"item": "Caregiver Burden Assessment + Palliative Planning", "frequency": "Annually; when disease stage changes", "rationale": "Lafora disease fatal trajectory; caregiver burden accumulates as cognitive/motor decline progresses; palliative care involvement from middle stage; advance care planning; respite care; psychological support for family; anticipatory guidance on terminal phase"},
    {"item": "Reproductive / Genetic Counselling", "frequency": "At diagnosis; before any pregnancy in carrier family members", "rationale": "Autosomal recessive — both parents are obligate carriers; sibling recurrence risk 25%; prenatal testing (CVS/amniocentesis); carrier testing in extended family; VPA teratogenicity (VPPP) if VPA being used in females of childbearing potential; preconception genetic counselling"},
]

# ─────────────────────────────────────────────
# LIFECYCLE
# ─────────────────────────────────────────────
LIFECYCLE = [
    {
        "stage": "Pre-Symptomatic (0-10 years) — Normal Development",
        "key_issues": "NHLRC1 LOF biallelic from conception but Lafora bodies accumulate SLOWLY in childhood; completely normal neurological development; no EEG abnormality; no seizures; disease subclinical; genetic diagnosis possible only if older sibling already diagnosed (family screening) or prenatal testing",
        "management": "If older sibling with Lafora disease: offer genetic testing at birth; if positive, begin monitoring (annual EEG from age 8; neuropsychological baseline by age 9); no AED needed pre-symptomatically; experimental centres may offer metformin/KD pre-symptomatic trial (disease-modifying rationale); genetic counselling for family",
    },
    {
        "stage": "Seizure Onset (10-18 years) — First Presentation",
        "key_issues": "Previously normal adolescent → first seizures; often visual (occipital) aura or GTCS; initial diagnosis frequently missed (JME, benign occipital epilepsy, idiopathic epilepsy assumed); WRONG AED often prescribed (CBZ/OXC) → myoclonic worsening → Lafora disease unmasked; skin biopsy + EEG ± genetics needed urgently",
        "management": "EMERGENCY RULE: never prescribe CBZ/OXC/PHT before PME workup is complete in adolescent; initial AED: VPA (post-POLG1 screen) + LEV; urgent skin biopsy (axillary); EEG with photic stimulation; NHLRC1 + EPM2A panel; if myoclonic worsening occurs → IMMEDIATE medication review; add CLB; neuropsychological baseline; school safety plan",
    },
    {
        "stage": "Early Lafora (1-3 years after onset) — Myoclonic Escalation",
        "key_issues": "Myoclonic seizures escalate; action myoclonus develops (writing difficulty, eating difficulty — functional disability markers); visual seizures frequent; GTCS occurring 1-3/month; cognitive function still largely intact but early subtle changes; school performance declining; social isolation from seizure visibility",
        "management": "Add piracetam (action myoclonus); add perampanel (visual/myoclonic); optimise VPA to target 70-120 mcg/mL; add CLB; consider KD; consider metformin (disease-modifying); photosensitivity management (blue-light glasses, screen protocols); school accommodations (OT assessment, scribe for writing); driving prohibition; swimming/cycling prohibition unsupervised",
    },
    {
        "stage": "Middle Lafora (3-7 years after onset) — Cognitive Decline + DRE",
        "key_issues": "Significant cognitive decline (memory, attention, executive function); dyslexia-like regression; seizure frequency high despite maximal AED (DRE established); action myoclonus severe (may prevent feeding → weight loss); drop attacks emerging; depression/anxiety; increasing caregiver dependency; school untenable",
        "management": "Maximal AED combination (VPA + LEV + CLB + piracetam + perampanel + ZNS); VNS consideration (salvage therapy); KD continued if tolerated; metformin ongoing; caregiver training; home occupational therapy; modified diet (soft/puréed); speech and language therapy; psychological support (family + patient); palliative care team introduction; advance care planning discussion",
    },
    {
        "stage": "Late Lafora (7-10 years after onset) — Severe Disability",
        "key_issues": "Severe cognitive impairment (dementia-like); frequent seizures including prolonged myoclonic status; dysphagia (aspiration risk — leading cause of death); loss of ambulation; full-time care dependency; vegetative state possible; SUDEP risk elevated",
        "management": "Palliative seizure management (comfort-focused); nasogastric or PEG tube for nutrition/AED delivery; anti-aspiration measures; respiratory physiotherapy; pressure sore prevention; caregiver respite; hospice care referral; anticipatory prescribing for terminal phase (midazolam SC, phenobarbitone SC); DNACPR discussion if appropriate; bereavement support for family",
    },
    {
        "stage": "Terminal Phase (>10 years or earlier) — End of Life",
        "key_issues": "Cardiorespiratory failure, aspiration pneumonia, or uncontrolled status epilepticus; median age of death 20-30s; hospice/palliative care primary; family grief support; autopsy (academic — brain Lafora body burden confirmation for family and research); organ donation (heart, liver — non-neural tissues may be viable)",
        "management": "Comfort-focused care; symptom control (pain, dyspnoea, seizure comfort); psychological/spiritual support; post-bereavement counselling for family; memorial/research participation offer (brain bank donation); death certificate: cause of death 'Lafora disease (NHLRC1-related progressive myoclonic epilepsy)' for accurate epidemiological records",
    },
]

# ─────────────────────────────────────────────
# CONCEPTS (15 key clinical concepts)
# ─────────────────────────────────────────────
CONCEPTS = [
    {
        "concept": "NHLRC1-EPM2B-Lafora-Type-2-Malin-E3-Ligase",
        "explanation": "NHLRC1 (6p22.3) encodes MALIN, a RING-H2 E3 ubiquitin ligase with 6 NHL repeat substrate-binding domains. MALIN works in obligate complex with LAFORIN (EPM2A phosphatase) to polyubiquitinate glycogen synthase (GYS1), PTG, and STBD1 → proteasomal degradation → glycogen quality control. LOF (either EPM2B/NHLRC1 or EPM2A/laforin) → impaired GYS1 degradation → constitutive glycogen synthesis → phosphorylated, poorly branched polyglucosan chains → LAFORA BODIES in neurons, sweat glands, cardiac myocytes. Lafora disease Type 2 (NHLRC1) tends to have slightly slower progression than Type 1 (EPM2A) in some series.",
    },
    {
        "concept": "Lafora-Body-PAS-Positive-Polyglucosan-Pathognomonic",
        "explanation": "Lafora bodies are intracellular inclusions of 3-40 µm, round or ovoid, PAS-positive (periodic acid-Schiff — stains polysaccharide), diastase-resistant (resistant to amylase digestion — distinguish from glycogen which is diastase-labile). Found in neurons (perineural satellite bodies), cardiac myocytes, liver, and — crucially for biopsy — eccrine sweat gland DUCT cells (NOT secretory cells). Axillary skin biopsy is the gold standard rapid diagnosis: positive in >95% of confirmed Lafora cases, available within days, less invasive than brain biopsy. Electron microscopy: fibrillar polyglucosan ultrastructure with central denser core and peripheral fibrillar fringe.",
    },
    {
        "concept": "Adolescent-Normal-Development-PME-Diagnostic-Key",
        "explanation": "The hallmark diagnostic feature of Lafora disease: PREVIOUSLY NORMAL ADOLESCENT (age 10-18 years, normal development, normal intelligence) presenting with first seizures. Lafora bodies begin accumulating before birth but neuronal dysfunction is subclinical until critical neuronal Lafora body burden (~age 10). In contrast, most DEE genes cause early-infantile epilepsy. When an otherwise normal teenager develops myoclonus + visual seizures + GTCS → Lafora disease is top differential. Normal development prior to seizure onset distinguishes Lafora from developmental/epileptic encephalopathies.",
    },
    {
        "concept": "CBZ-OXC-PHT-ABSOLUTE-CI-Paradoxical-Myoclonic-Worsening",
        "explanation": "Sodium channel blockers are the most dangerous drug class in Lafora disease and all PMEs. Mechanism: Na-channel blockade preferentially suppresses fast-firing GABAergic interneurons (which fire at high rates to maintain inhibition) → cortical disinhibition → myoclonic circuit hyperexcitability. The paradox: drugs designed to reduce seizures acutely worsen the myoclonic component. In Lafora disease, progressive interneuron loss amplifies this disinhibition risk. Case reports document myoclonic status epilepticus precipitated within 48-72 hours of CBZ initiation. This is the most common iatrogenic harm in Lafora disease — CBZ/OXC prescribed by ER physicians unaware of the PME diagnosis.",
    },
    {
        "concept": "Visual-Cortex-Occipital-Photosensitivity-Lafora-Hallmark",
        "explanation": "Lafora disease has the highest rate of visual cortex involvement among PMEs. Lafora bodies preferentially accumulate in cortical neurons with highest metabolic activity — occipital neurons (continuous visual processing) are particularly vulnerable. Photoparoxysmal response (PPR) to intermittent photic stimulation (IPS) in ~80% of Lafora patients (highest among all PME subtypes). Visual aura: flickering zigzag phosphenes, formed visual hallucinations — pathognomonic for Lafora among PMEs (Unverricht-Lundborg disease lacks structured visual aura). Blue-light filtering (450nm) + perampanel (AMPA blockade of visual cortex) + VPA are the evidence-based interventions for occipital hyperexcitability.",
    },
    {
        "concept": "Metformin-AMPK-GYS1-Disease-Modifying-Glycogen-Suppression",
        "explanation": "MALIN normally ubiquitinates GYS1 for degradation. In NHLRC1 LOF, GYS1 escapes ubiquitination → constitutively active → excess polyglucosan. METFORMIN activates AMPK (AMP-activated protein kinase) → AMPK phosphorylates GYS1 at Ser7 → allosteric inactivation of GYS1 → reduced glycogen synthesis → fewer Lafora bodies. This BYPASSES the absent MALIN ubiquitination using AMPK-mediated phosphorylation as an alternative GYS1 inactivation pathway. Berthier 2016 demonstrated 60-80% Lafora body reduction in Nhlrc1 knockout mice with metformin ± rapamycin. Human trials emerging. Metformin is the only approved drug with disease-modifying rationale in Lafora disease.",
    },
    {
        "concept": "VPA-PME-Backbone-Lafora-Dose-Target",
        "explanation": "Valproate is the irreplaceable backbone of Lafora disease pharmacotherapy. Multiple mechanisms synergistically address PME: GABA-T inhibition (increases synaptic GABA), Na-channel slow inactivation (modest), T-type Ca²⁺ block (reduces thalamo-cortical oscillations driving myoclonus), HDAC inhibition (chromatin remodelling potentially reduces GYS1 expression). Target VPA TDM in Lafora disease is HIGHER than standard: 70-120 mcg/mL total (vs 50-100 mcg/mL for focal epilepsy) — higher target required for adequate myoclonic suppression. Controlled-release (CR/ER) formulation is essential for stable troughs in adolescent rapid metabolism.",
    },
    {
        "concept": "Perampanel-AMPA-Rationale-Lafora-Visual-Myoclonic",
        "explanation": "Cortical myoclonus in Lafora disease is driven by excessive corticothalamic AMPA receptor-mediated excitatory discharge. Perampanel (non-competitive AMPA antagonist) directly opposes this. DUAL BENEFIT in Lafora: (1) reduces cortical myoclonus via AMPA blockade at motor cortex; (2) reduces occipital AMPA overactivity → fewer photosensitive visual cortex seizures. Mukherjee 2019 series: 65% myoclonic reduction with perampanel in Lafora disease patients. VPA-perampanel interaction: VPA inhibits CYP3A4 → increases perampanel AUC → limit perampanel to 8 mg/day maximum with VPA. Nocturnal dosing reduces daytime dizziness/ataxia.",
    },
    {
        "concept": "Piracetam-Action-Myoclonus-High-Dose-Mechanism",
        "explanation": "Action myoclonus (movement-induced cortical reflex myoclonus) is the most functionally disabling seizure type in Lafora disease — prevents eating, writing, and purposeful hand use. Piracetam in high doses (20-45g/day) is the specific therapy for action myoclonus in PME. Mechanism: facilitates AMPA receptor-mediated transmission in a way that improves cortical synchrony and reduces myoclonic circuit hyperexcitability (positive allosteric AMPA modulation). Must be combined with VPA (piracetam does not control GTCS). No pharmacokinetic interactions with VPA/LEV/CLB — safe polypharmacy. IV piracetam (20g IV) can acutely reduce myoclonic status in PME.",
    },
    {
        "concept": "Progressive-Fatal-Disease-Trajectory-Palliative-Planning",
        "explanation": "Lafora disease is UNIFORMLY FATAL. Median survival 5-10 years from seizure onset (death typically in 2nd-3rd decade of life). Disease stages: (1) Onset (10-18y); (2) Early (myoclonic escalation + visual seizures); (3) Middle (cognitive decline + DRE); (4) Late (severe disability + dysphagia); (5) Terminal (aspiration/SE/cardiorespiratory). No disease-modifying treatment is curative. Palliative care involvement from middle stage is best practice (not waiting for terminal phase). Advance care planning, DNACPR discussion, hospice referral, brain bank donation offer — all are appropriate in this disease.",
    },
    {
        "concept": "Skin-Biopsy-Before-Genetics-Rapid-Diagnosis-Strategy",
        "explanation": "In adolescent with suspected Lafora disease, skin biopsy (axillary) + PAS staining provides a diagnosis within 48-72 hours — BEFORE genetic results are available (genetic testing takes weeks). A positive biopsy (PAS-positive diastase-resistant inclusions in sweat gland ducts) is pathognomonic → begin appropriate PME AEDs immediately. This strategy prevents the dangerous window where an adolescent awaiting genetic results might receive CBZ/OXC from ER physicians. Biopsy-guided diagnosis also reduces cost (targeted NHLRC1/EPM2A panel rather than broad WES if clinical certainty high).",
    },
    {
        "concept": "KD-Dual-Mechanism-Anti-Seizure-Disease-Modifying-Glycogen",
        "explanation": "Ketogenic diet (KD) provides dual benefit in Lafora disease: (1) ANTI-SEIZURE: ketone body (BHB) → adenosine A1 receptor activation + AMPK → mTOR suppression → neuronal excitability reduction; (2) DISEASE-MODIFYING: glucose restriction → reduced glycogen substrate → less polyglucosan for Lafora body formation; BHB-mediated AMPK also directly inhibits GYS1 (same mechanism as metformin — potentially additive). Modified Atkins diet (20-30g net carbs/day) preferred in adolescents for social/compliance reasons vs classical 4:1 KD. Combined KD + metformin is an emerging disease-modifying strategy under investigation.",
    },
    {
        "concept": "POLG1-VPA-Mandatory-Screen-Lafora",
        "explanation": "All Lafora patients require VPA as backbone. VPA cannot be initiated without POLG1 screening. Biallelic POLG1 pathogenic variants + VPA = Alpers-Huttenlocher syndrome (fatal mitochondrial hepatoencephalopathy). Lafora patients are NOT pre-screened for POLG1. POLG1 is a separate recessive condition (not linked to NHLRC1) — any Lafora patient could coincidentally carry biallelic POLG1 variants. If POLG1 biallelic confirmed → VPA absolutely contraindicated for life → PME backbone must be built from LEV + CLB + piracetam + perampanel + ZNS without VPA (very challenging).",
    },
    {
        "concept": "Consanguinity-Mediterranean-SouthAsian-Recurrence-Risk",
        "explanation": "Autosomal recessive inheritance: both parents are obligate carriers (NHLRC1 heterozygous carriers — phenotypically normal). Sibling recurrence risk: 25% for each subsequent pregnancy. Consanguinity (first/second cousin marriages) dramatically increases risk — Lafora disease prevalence is 10-fold higher in consanguineous Mediterranean (Spain: W219G NHLRC1 founder), South Asian (India/Pakistan: V85E, D146N), Middle Eastern (Turkey, Iran), and North African (Morocco, Algeria) populations. Cascade genetic testing of siblings mandatory at diagnosis. Prenatal diagnosis available: CVS (10-13 weeks) or amniocentesis (15-18 weeks) if both parental variants known.",
    },
    {
        "concept": "SUDEP-Risk-PME-Lafora-Safety-Measures",
        "explanation": "Sudden Unexpected Death in Epilepsy (SUDEP) risk is substantially elevated in Lafora disease compared to general epilepsy population. Risk factors: frequent GTCS (78% of patients), nocturnal seizures, progressive DRE, young age, progressive neurological decline (impairs arousal post-seizure). Safety measures mandatory: bed seizure alarm (Emfit/Embrace2/Smart Monitor); NEVER sleep alone in DRE; supine sleep positioning (prone sleeping increases SUDEP risk 5-fold); seizure-safe environment; VNS consideration (reduces SUDEP risk by 40% in some studies); annual SUDEP risk discussion and documentation with family.",
    },
]

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "VPA TDM target in Lafora disease", "value": "70-120 mcg/mL total (higher than standard 50-100; Lafora PME requires higher exposure for myoclonic control)"},
    {"threshold": "POLG1 screen turnaround before VPA", "value": "7-14 days; NEVER start VPA without result; bridge LEV+CLB"},
    {"threshold": "Perampanel maximum with VPA co-administration", "value": "8 mg/day (VPA inhibits CYP3A4 → ↑perampanel AUC ~15-20%); standard maximum 12 mg/day without VPA"},
    {"threshold": "Piracetam anti-myoclonic target dose (action myoclonus)", "value": "20-45 g/day adult; 160-320 mg/kg/day paediatric; escalate weekly by 5g; higher doses more effective"},
    {"threshold": "Skin biopsy Lafora body size", "value": "3-40 µm, PAS-positive, diastase-resistant inclusions in eccrine sweat gland ducts; >5 inclusions per high-power field = diagnostic"},
    {"threshold": "Metformin eGFR threshold for dose adjustment", "value": "eGFR 30-45: reduce dose 50%; eGFR <30: HOLD metformin (lactic acidosis risk)"},
    {"threshold": "AED taper rate in Lafora disease", "value": "Maximum 5-10% original dose per 4-6 weeks (much slower than standard 10% per 2 weeks for other epilepsies)"},
    {"threshold": "VPA initiation: POLG1 screen", "value": "7-14 days turnaround; bridge therapy LEV+CLB; if POLG1 biallelic → VPA ABSOLUTELY CONTRAINDICATED lifelong"},
    {"threshold": "Photoparoxysmal response prevalence in Lafora disease", "value": "~80% of patients show PPR on EEG with photic stimulation (highest among all PME subtypes)"},
    {"threshold": "Disease onset age diagnostic criterion", "value": "10-18 years in previously NORMAL adolescent (normal prior development MANDATORY for Lafora diagnosis)"},
    {"threshold": "Median survival from seizure onset", "value": "5-10 years; death typically in 2nd-3rd decade (age 18-35)"},
    {"threshold": "SUDEP risk threshold for bed alarm", "value": "≥1 nocturnal GTCS in 12 months → bed seizure alarm mandatory; Lafora DRE → always high SUDEP risk"},
]

# ─────────────────────────────────────────────
# STANDARDS
# ─────────────────────────────────────────────
STANDARDS = [
    {"standard": "ILAE-2022", "description": "ILAE classification of epilepsies — PME classification, genetic testing recommendations for progressive myoclonic epilepsy"},
    {"standard": "NICE-NG217", "description": "NICE epilepsy guideline: management of epilepsy in children and adults including genetic/metabolic causes (UK, 2022)"},
    {"standard": "Turnbull-2016-NatRevNeurol", "description": "Turnbull J et al. 2016: Lafora disease management consensus review — pharmacotherapy, monitoring, and prognosis"},
    {"standard": "Chan-2003-NatGenet", "description": "Chan EM et al. 2003 (Nature Genetics): Discovery of NHLRC1/EPM2B as the second Lafora disease locus — founding paper for NHLRC1 field"},
    {"standard": "Minassian-1998-NatGenet", "description": "Minassian BA et al. 1998 (Nature Genetics): Discovery of EPM2A/laforin as first Lafora disease gene — landmark PME genetics"},
    {"standard": "CPIC-POLG1-2023", "description": "CPIC guideline for valproate use in POLG1 variant carriers — mandatory pre-VPA POLG1 screen in all patients"},
    {"standard": "MHRA-VPPP-2021", "description": "Medicines and Healthcare products Regulatory Agency: Valproate Pregnancy Prevention Programme — mandatory for females of childbearing potential on VPA"},
    {"standard": "Delgado-Escueta-2001-NeurolClinics", "description": "Delgado-Escueta AV et al. 2001: Progressive myoclonic epilepsies framework — PME classification and genetic landscape"},
    {"standard": "Mukherjee-2019-Epilepsia", "description": "Mukherjee P et al. 2019 (Epilepsia): Perampanel in Lafora disease — 65% myoclonic reduction series; basis for Level B perampanel evidence"},
    {"standard": "Berthier-2016-MolMed", "description": "Berthier C et al. 2016 (Molecular Medicine): Metformin + rapamycin reduce Lafora bodies 60-80% in mouse model — disease-modifying framework for clinical translation"},
    {"standard": "ACMG-AMP-2015", "description": "ACMG/AMP variant classification standards — applied to NHLRC1 biallelic pathogenic variant classification"},
    {"standard": "WHO-ICF-2019", "description": "International Classification of Functioning, Disability and Health — function and disability assessment in progressive disease"},
]

# ─────────────────────────────────────────────
# REFERENCES
# ─────────────────────────────────────────────
REFERENCES = [
    {
        "ref": "Chan-2003-NatGenet",
        "full": "Chan EM et al. (2003). Mutations in NHLRC1 cause progressive myoclonus epilepsy. Nature Genetics, 35(2), 125-127.",
        "key_finding": "Discovery of NHLRC1/EPM2B as the second Lafora disease gene; identified NHLRC1 (encoding MALIN) as E3 ubiquitin ligase; established NHLRC1 LOF → Lafora disease type 2 causal link",
    },
    {
        "ref": "Minassian-1998-NatGenet",
        "full": "Minassian BA et al. (1998). Mutations in a gene encoding a novel protein tyrosine phosphatase cause progressive myoclonus epilepsy. Nature Genetics, 20(2), 171-174.",
        "key_finding": "Discovery of EPM2A/laforin as the first Lafora disease gene; established MALIN-LAFORIN complex as glycogen quality control unit; framework for understanding both EPM2A and NHLRC1 LOF mechanism",
    },
    {
        "ref": "Turnbull-2016-NatRevNeurol",
        "full": "Turnbull J et al. (2016). Lafora disease. Nature Reviews Neurology, 12(10), 570-584.",
        "key_finding": "Comprehensive Lafora disease review: epidemiology, molecular mechanism, NHLRC1/EPM2A biology, clinical management, monitoring standards, prognosis, and emerging therapies including metformin",
    },
    {
        "ref": "Mukherjee-2019-Epilepsia",
        "full": "Mukherjee P et al. (2019). Perampanel in Lafora disease: a case series and review of the literature. Epilepsia, 60(8), e73-e78.",
        "key_finding": "First systematic evaluation of perampanel in Lafora disease (n=12): 65% reduction in myoclonic seizure frequency; good tolerability; provides Level B evidence basis for perampanel use in Lafora PME",
    },
    {
        "ref": "Berthier-2016-MolMed",
        "full": "Berthier A et al. (2016). Pharmacological interventions to ameliorate neuropathological symptoms in a mouse model of Lafora disease. Molecular Medicine, 22(1), 597-608.",
        "key_finding": "Metformin + rapamycin reduced Lafora body burden 60-80% in Nhlrc1 knockout mice; identified AMPK-GYS1 pathway as disease-modifying target; translational basis for human metformin trials in Lafora disease",
    },
    {
        "ref": "Raththagala-2015-MolCell",
        "full": "Raththagala M et al. (2015). Structural mechanism of laforin function in glycogen dephosphorylation and Lafora disease. Molecular Cell, 57(2), 261-272.",
        "key_finding": "Defined laforin CBM-phosphatase domain structure and mechanism; laforin dimer required for glucan-binding; NHLRC1 mutations disrupting MALIN-LAFORIN interaction characterised; molecular basis for why biallelic NHLRC1 OR EPM2A LOF both cause Lafora disease",
    },
]


# ─────────────────────────────────────────────
# PUBLIC API FUNCTIONS
# ─────────────────────────────────────────────
def get_overview():
    total = len(PATIENT_SAMPLE)
    female_n = sum(1 for p in PATIENT_SAMPLE if p["sex"] == "F")
    consang_n = sum(1 for p in PATIENT_SAMPLE if p["consanguineous"])
    biopsy_pos_n = sum(1 for p in PATIENT_SAMPLE if p["biopsy_positive"])
    on_vpa_n = sum(1 for p in PATIENT_SAMPLE if p["on_vpa"])
    on_lev_n = sum(1 for p in PATIENT_SAMPLE if p["on_lev"])
    on_perampanel_n = sum(1 for p in PATIENT_SAMPLE if p["on_perampanel"])
    on_piracetam_n = sum(1 for p in PATIENT_SAMPLE if p["on_piracetam"])
    on_kd_n = sum(1 for p in PATIENT_SAMPLE if p["on_kd"])
    on_metformin_n = sum(1 for p in PATIENT_SAMPLE if p["on_metformin"])
    cog_decline_n = sum(1 for p in PATIENT_SAMPLE if p["cognitive_decline"])
    polg1_n = sum(1 for p in PATIENT_SAMPLE if p["polg1_screened"])
    avg_onset_m = round(sum(p["age_onset_months"] for p in PATIENT_SAMPLE) / total)
    stage_counts = {"early": 0, "middle": 0, "late": 0}
    for p in PATIENT_SAMPLE:
        stage_counts[p["disease_stage"]] += 1

    etiology_dist = {item["category"]: {"n": item["n"], "pct": item["pct"]} for item in ETIOLOGY_CATALOG}
    seizure_dist = [{"type": s["type"].split("(")[0].strip().split("—")[0].strip(), "pct": s["frequency_pct"]} for s in SEIZURE_TYPES]
    trigger_dist = [{"trigger": t["trigger"], "pct": t["pct"]} for t in TRIGGERS]

    return {
        "dashboard": "NHLRC1 Epilepsy (Lafora Disease Type 2 / EPM2B / Malin E3 Ubiquitin Ligase / Progressive Myoclonic Epilepsy / Polyglucosan-Lafora-Bodies / 6p22.3)",
        "gene": "NHLRC1 / EPM2B (6p22.3) — NHL Repeat Containing 1; encodes MALIN (RING-H2 E3 ubiquitin ligase + 6×NHL repeats); 395 aa ~42 kDa; GYS1/PTG/STBD1 ubiquitination for glycogen quality control",
        "inheritance": "Autosomal recessive LOF (biallelic mutations required); pLI=0.14; consanguinity major risk factor",
        "omim": "*608072 (NHLRC1/EPM2B gene); #254780 (Lafora disease — both EPM2A and EPM2B loci)",
        "cohort_size": total,
        "female_n": female_n,
        "female_pct": round(female_n / total * 100),
        "mean_onset_months": avg_onset_m,
        "mean_onset_years": round(avg_onset_m / 12, 1),
        "drug_resistant_n": total,
        "drug_resistant_pct": 100,
        "consanguineous_n": consang_n,
        "consanguineous_pct": round(consang_n / total * 100),
        "biopsy_positive_n": biopsy_pos_n,
        "biopsy_positive_pct": round(biopsy_pos_n / total * 100),
        "on_vpa_n": on_vpa_n,
        "on_vpa_pct": round(on_vpa_n / total * 100),
        "on_lev_n": on_lev_n,
        "on_lev_pct": round(on_lev_n / total * 100),
        "on_perampanel_n": on_perampanel_n,
        "on_perampanel_pct": round(on_perampanel_n / total * 100),
        "on_piracetam_n": on_piracetam_n,
        "on_piracetam_pct": round(on_piracetam_n / total * 100),
        "on_kd_n": on_kd_n,
        "on_kd_pct": round(on_kd_n / total * 100),
        "on_metformin_n": on_metformin_n,
        "on_metformin_pct": round(on_metformin_n / total * 100),
        "cognitive_decline_n": cog_decline_n,
        "cognitive_decline_pct": round(cog_decline_n / total * 100),
        "polg1_screened_n": polg1_n,
        "polg1_screened_pct": round(polg1_n / total * 100),
        "disease_stage_counts": stage_counts,
        "etiology_distribution": etiology_dist,
        "seizure_type_distribution": seizure_dist,
        "trigger_distribution": trigger_dist,
        "key_contraindications": [
            "CBZ / OXC / PHT — ABSOLUTE CI (Na-channel blockers WORSEN myoclonus paradoxically → myoclonic status; most common iatrogenic harm in Lafora disease; IV fosphenytoin ABSOLUTE CI for SE)",
            "LTG MONOTHERAPY — HIGH RISK (myoclonic worsening + visual cortex aggravation; only as adjunct to VPA if needed, never monotherapy)",
            "TGB — ABSOLUTE CI (NCSE; progressive GABAergic interneuron loss amplifies risk; NCSE may be misattributed to disease progression)",
            "VGB — AVOID (irreversible visual field defects + visual cortex already impaired by Lafora bodies + may worsen myoclonic)",
            "VPA without POLG1 SCREEN — ABSOLUTE CI (Alpers-Huttenlocher fatal hepatotoxicity; Lafora patients not pre-screened)",
            "ABRUPT AED WITHDRAWAL — ABSOLUTE CI (myoclonic status epilepticus; fatal in Lafora disease; maximum taper 5-10% per 4-6 weeks)",
        ],
        "lafora_body_note": "Lafora bodies: PAS-positive, diastase-resistant polyglucosan inclusions 3-40µm in eccrine sweat gland ducts (axillary biopsy gold standard), cortical neurons, cardiac myocytes. MALIN (NHLRC1) normally ubiquitinates GYS1 for degradation → LOF → GYS1 constitutively active → phosphorylated, poorly-branched polyglucosan chains → Lafora bodies → progressive neuronal dysfunction.",
        "disease_modifying_note": "Only disease with disease-modifying pharmacology: METFORMIN (AMPK→GYS1 inactivation, bypasses absent MALIN ubiquitination) + KD (glucose restriction → less polyglucosan substrate + BHB-AMPK). ASO/gene therapy research phase. Combined metformin + KD + rapamycin is emerging disease-modifying strategy under trial.",
        "prognosis_note": "Lafora disease is UNIFORMLY FATAL. Median survival 5-10 years from seizure onset. Death in 2nd-3rd decade from aspiration pneumonia, refractory status epilepticus, or cardiorespiratory failure. No curative treatment. Palliative care involvement from middle stage is best practice.",
    }


def get_breakdown():
    return {
        "etiology_catalog": ETIOLOGY_CATALOG,
        "patient_sample": PATIENT_SAMPLE[:15],
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
    }


def get_definitions():
    return {
        "concepts": CONCEPTS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
