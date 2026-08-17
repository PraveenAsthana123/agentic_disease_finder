"""
RORB Epilepsy — Genetic Generalised Epilepsy / RORB-GGE / DEE-RORB
(Retinoid acid Receptor-related Orphan Receptor Beta / 9q21.13)
======================================================================
40-patient cohort · RORB heterozygous LOF / de novo · Autosomal Dominant (variable penetrance)
Thalamocortical circuit transcription factor · GGE / Absence / Myoclonic-atonic / DEE spectrum

RORB BIOLOGY:
RORB (9q21.13) encodes RORβ (Retinoid acid receptor-related Orphan Receptor Beta), a
nuclear receptor transcription factor and member of the NR1 subfamily (RORα/β/γ). RORβ
lacks a cognate ligand (orphan receptor) and acts as a constitutive transcriptional activator
or repressor depending on target gene context.

RORB PROTEIN STRUCTURE:
  - DNA-binding domain (DBD, 2 zinc fingers): recognises RORE (ROR response elements) with
    core motif AGGTCA / RGGTCA — controls gene expression of >1,000 target loci.
  - Ligand-binding domain (LBD): orphan pocket; constitutively bound by cholesterol sulphate
    as partial agonist; no known clinical pharmacological ligand.
  - Hinge region: mediates corepressor / coactivator interactions (NCoR, HDAC3).
  - N-terminal A/B domain: AF-1 transactivation; contributes to DNA-binding cooperativity.

RORB NEURODEVELOPMENTAL FUNCTIONS:
  1. Thalamic relay neuron specification: RORβ is the master transcription factor for
     layer IV thalamocortical recipient neurons (excitatory granule cells) in somatosensory
     cortex and thalamic relay nuclei. LOF → failure to specify thalamic relay neurons
     → dysfunctional thalamocortical synchronisation → generalised spike-wave discharge (SWD).
  2. GABAergic interneuron maturation: RORβ regulates parvalbumin (PV) expression in
     cortical interneurons via direct RORE binding. LOF → reduced PV+ interneuron density
     → impaired feedforward inhibition → GGE phenotype (absence + myoclonic seizures).
  3. Circadian rhythm integration: RORβ is a core circadian clock component (RORα/β/γ drive
     Bmal1 transcription via RORE). LOF → disrupted circadian gating of cortical excitability
     → seizure clustering at sleep–wake transitions (consistent with JME-like diurnal pattern).
  4. Retinal photoreceptor development: RORβ required for cone photoreceptor specification
     (distinct from neuronal role). Not clinically relevant for epilepsy but explains Rorb−/−
     mouse abnormal electroretinogram and light-dark preference changes.
  5. Spindle morphogenesis (cortical): RORβ specifies layer IV neurons that form cortical
     barrels in rodents; human equivalent → sensory thalamocortical columns. LOF → structural
     thalamocortical misspecification subtly visible on 7T MRI (reduced thalamic volume, thin
     layer IV, reduced cortical barrel-equivalent columns in 40% of patients in published series).

RORB-EPILEPSY (RORB-GGE / DEE-RORB):
  OMIM #614142 (RORB: Epilepsy, childhood absence 6, susceptibility to)
  OMIM #619696 (RORB: Neurodevelopmental disorder with or without epilepsy)

  PHENOTYPE SPECTRUM (Rinaldi et al. 2022 series of 24 + published):
    I.  Mild GGE: typical absence epilepsy (CAE / JAE phenotype) + NORMAL cognition (~25%)
    II. JME-like: myoclonic + GTCS + absence + normal-borderline cognition (~30%)
    III. Myoclonic-atonic (Doose-like): drop attacks + absence + mild-moderate ID (~20%)
    IV. Encephalopathy (DEE): severe ID + drug-resistant epilepsy + regression + autism (~15%)
    V.  Non-epileptic NDD: ID + developmental delay + ASD without seizures (~10%)

  PATHOGNOMONIC CLINICAL FEATURE:
  RORB-GGE does NOT have a single pathognomonic semiology, but features that raise suspicion:
    - Generalised spike-wave at 3 Hz + myoclonic-atonic component (Doose-like pattern)
    - Sleep-wake transition clustering: peak at awakening (JME-like circadian pattern)
    - Photosensitivity + eye-closure sensitivity in ~30%
    - Variable cognitive phenotype within the SAME family (penetrance ~80%, incomplete)
    - Thalamic hypoplasia on MRI in ~35% (subtle; requires volumetric analysis)

  GENETICS:
  - Gene: RORB, chromosome 9q21.13
  - Inheritance: Autosomal Dominant; ~80% penetrance; ~60% de novo; ~40% familial
  - Variant types: missense ~50%, frameshift ~25%, nonsense ~12%, splice-site ~8%, CNV ~5%
  - LOF mechanism: haploinsufficiency (one functional allele insufficient for RORβ-dependent
    thalamocortical circuit specification during critical period P0–P14 in mice / fetal weeks 15–28)
  - Key variants: p.Ala200Thr, p.Ser216Phe (DBD: reduced RORE binding), p.Leu376Arg (LBD)
  - ClinVar: ~80 RORB variants; growing number of P/LP de novo in gnomAD constrained gene
  - Intolerance: pLI = 0.97 (high LOF intolerance, gnomAD v4) — consistent with haploinsufficiency

EEG IN RORB-GGE:
  - Interictal: generalised 3–4 Hz spike-wave (SWD); posterior > frontal; activated by drowsiness
  - Ictal absence: 3 Hz GSW for 4–30 seconds; abrupt onset/offset; eye flutter / automatisms
  - Ictal myoclonic: 3–6 Hz polyspike-wave; brief bilateral arm/shoulder jerks; arousal-triggered
  - Ictal myoclonic-atonic: polyspike → slow wave; drop attack; distinct from Lennox-Gastaut
  - Sleep EEG: NREM-activated SWD; sleep spindle morphology abnormal in ~40%
  - Photoparoxysmal response (PPR): present in ~30%; type II–III (Waltz 1992 classification)
  - Background: normal in Mild GGE; may show theta slowing in DEE variant

TREATMENT APPROACH:
  Broad-spectrum AEDs preferred (GGE pattern); Na-channel blockers (CBZ/OXC/PHT) are
  CONTRAINDICATED — they worsen absence and myoclonic seizures via use-dependent Na-channel
  blockade paradoxically increasing cortical excitability in the GGE circuit.

KEY STANDARDS:
  ILAE 2022 · NICE NG217 · Rinaldi et al. 2022 (Epilepsia — largest RORB cohort) ·
  Ricos et al. 2016 (Epileptic Disord — RORB in severe EE) ·
  Coppola et al. 2019 (Epilepsy Res — RORB GGE spectrum) ·
  CPIC HLA-B*15:02 2023 (if CBZ is wrongly tried) · MHRA VPPP 2021 · ACMG-AMP 2015 ·
  ILAE Dietary Therapies 2018 · FDA Valproate REMS · SANAD-II 2021
"""
import random

# ── Etiology catalog ─────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "RORB-de-novo-LOF (classic GGE / mild phenotype)",
        "category": "RORB-de-novo-LOF-mild",
        "pct": 28,
        "n": 11,
        "mechanism": (
            "De novo heterozygous RORB LOF variant (missense in DBD zinc-finger or frameshift). "
            "Haploinsufficiency during fetal thalamocortical development: 50% normal RORβ → "
            "insufficient specification of thalamic relay neurons + PV interneurons → "
            "thalamocortical hyperexcitability → generalised 3 Hz SWD. Penetrance ~80%. "
            "Mild phenotype: classic absence epilepsy (CAE/JAE) with NORMAL cognition; "
            "excellent response to ETH/VPA. No structural MRI lesion in most (60–70%)."
        ),
        "eeg_correlate": (
            "Generalised 3 Hz SWD · Typical absence ictal EEG · Background normal · "
            "NREM-activated SWD · PPR in ~25%"
        ),
        "semiology": (
            "Typical absence (blank stare, eye flutter, automatisms, abrupt offset); "
            "GTCS in ~50% (awakening-associated); myoclonic jerks at shoulders on awakening."
        ),
        "treatment": "ETH (absence-primary) or VPA (Level A, broad-spectrum); LEV (Level B adjunct)",
        "prognosis": "Good — 70% seizure-free on ETH or VPA; seizure remission in ~60% by adulthood",
    },
    {
        "etiology": "RORB-de-novo-GOF/dominant-negative (JME-like / myoclonic)",
        "category": "RORB-de-novo-DN",
        "pct": 25,
        "n": 10,
        "mechanism": (
            "De novo RORB missense in LBD or hinge region causing dominant-negative (DN) effect: "
            "mutant RORβ dimerises with wild-type RORβ → suppresses WT RORE transactivation "
            "below haploinsufficiency threshold. Circadian pathway more severely disrupted → "
            "abnormal sleep architecture → increased awakening-triggered SWD and myoclonic bursts. "
            "JME-like: myoclonic + GTCS + occasional absence; normal/borderline IQ."
        ),
        "eeg_correlate": (
            "3–6 Hz polyspike-wave (PSW) · Awakening-triggered PSW bursts · "
            "Occasional 3 Hz GSW · Background normal"
        ),
        "semiology": (
            "Bilateral shoulder myoclonic jerks on awakening (pathognomonic window 30 min post-arousal); "
            "GTCS (predominantly morning); absence in ~40%; photosensitivity in ~35%."
        ),
        "treatment": "VPA (Level A, first-line); LEV (Level B, alternative); CLB (Level C adjunct)",
        "prognosis": "Moderate — 55% seizure-free on VPA; lifelong AED required in ~70%; drug-resistant ~20%",
    },
    {
        "etiology": "RORB-de-novo-LOF (myoclonic-atonic / Doose-like phenotype)",
        "category": "RORB-de-novo-LOF-MAE",
        "pct": 20,
        "n": 8,
        "mechanism": (
            "De novo RORB LOF haploinsufficiency with more severe thalamic relay neuron loss: "
            "reduced thalamic volume on MRI (35%), thinner layer IV cortex. "
            "Myoclonic-atonic epilepsy (MAE/Doose-like): myoclonic burst → immediate loss of "
            "postural tone → drop attack. EEG shows brief 2–3 Hz SW + polyspike → slow wave "
            "(distinct from Lennox-Gastaut which has <2.5 Hz). Mild-moderate ID in ~60%. "
            "Ketogenic diet shows 50–60% responder rate (published MAE series)."
        ),
        "eeg_correlate": (
            "2–3 Hz polyspike-wave → slow wave (drop attack ictal EEG) · "
            "NREM-activated SWD · Photosensitivity ~30% · Background: generalised theta in ~40%"
        ),
        "semiology": (
            "Drop attacks (myoclonic-atonic): sudden fall with preserved consciousness; "
            "absence; GTCS; eyelid myoclonia; more severe phenotype than GGE mild."
        ),
        "treatment": (
            "VPA + ETH (Level B combination); KD (Level B, 50–60% responder); "
            "CLB (Level C); Perampanel (Level C drop attacks)"
        ),
        "prognosis": "Moderate-poor — 40% seizure-free; KD often required; helmet mandatory for drop attacks",
    },
    {
        "etiology": "RORB-de-novo-LOF (DEE / severe encephalopathy phenotype)",
        "category": "RORB-de-novo-DEE",
        "pct": 17,
        "n": 7,
        "mechanism": (
            "De novo RORB truncating LOF (frameshift / nonsense) causing severe haploinsufficiency "
            "affecting both DBD and LBD function. Most severe circuit disruption: "
            "severe reduction of PV+ interneurons, structural thalamic hypoplasia, thin corpus callosum. "
            "Phenotype: severe DEE — refractory epilepsy (>3 seizure types), severe-profound ID, "
            "ASD features, regression. May overlap with CDKL5/FOXG1 phenotypic spectrum."
        ),
        "eeg_correlate": (
            "Multifocal SWD + GSW · Hypsarrhythmia in infancy if IS onset · "
            "Suppression-burst rare · Background moderate-severe slowing"
        ),
        "semiology": (
            "Multiple seizure types: infantile spasms (30%), focal motor, myoclonic, "
            "tonic, GTCS, absence; severe ID + ASD features; regression possible."
        ),
        "treatment": (
            "VPA (Level B); KD (Level B); LEV (Level C); CLB (Level C); "
            "ACTH (Level B if IS); CBD-Epidiolex (Level C adjunct)"
        ),
        "prognosis": "Poor — <20% seizure-free; most require polytherapy; KD + VPA combinations used",
    },
    {
        "etiology": "Phenocopy: GABRA1 / GABRB3 / GABRG2 (GGE-phenocopy negative-RORB)",
        "category": "phenocopy-GGE",
        "pct": 10,
        "n": 4,
        "mechanism": (
            "Clinically indistinguishable GGE phenotype (absence + myoclonic + GTCS) with normal "
            "RORB sequencing. Phenocopies from GABRA1/GABRB3/GABRG2 variants (GABA-A receptor "
            "subunit genes on 5q34 — same chromosomal band, may co-migrate on SNP arrays). "
            "Also consider: CACNB4, CLCN2, EFHC1 (JME loci), NRXN1 microdeletion. "
            "Treatment identical (broad-spectrum AEDs); genetic panel required."
        ),
        "eeg_correlate": (
            "3 Hz GSW (indistinguishable from RORB-GGE) · Phenotype overlap ·"
            "PPR may be higher in GABRG2"
        ),
        "semiology": (
            "Typical absence ± myoclonic ± GTCS — identical to RORB-GGE mild. "
            "CBZ/OXC avoid regardless (GGE-broad CI). Panel testing distinguishes."
        ),
        "treatment": "Same as RORB-GGE: ETH/VPA first-line; avoid CBZ/OXC/PHT (GGE class-effect)",
        "prognosis": "Good (GABRA1/GABRG2 mild) to variable (GABRB3 DEE spectrum)",
    },
]

# ── Seizure types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Typical Absence (GGE-RORB)",
        "prevalence_pct": 80,
        "duration": "5–30 s",
        "eeg_ictal": "Abrupt-onset 3 Hz generalised spike-wave; posterior > frontal; normal EEG at end",
        "semiology": (
            "Blank stare + eye flutter (60%), perioral automatisms (40%), "
            "unresponsive; abrupt onset + abrupt offset; no post-ictal confusion. "
            "Hyperventilation (3 min) reliably provokes absence in untreated patients."
        ),
        "clinical_tip": (
            "Absence in RORB-GGE may cluster at SLEEP–WAKE transitions (circadian RORβ influence) "
            "— morning hyperventilation test is highest yield. "
            "Distinguishing RORB-mild from CAE/JAE: RORB often has coexisting myoclonic jerks."
        ),
    },
    {
        "type": "Myoclonic Jerks (bilateral, awakening-triggered)",
        "prevalence_pct": 65,
        "duration": "< 2 s; clustered in morning 30 min after arousal",
        "eeg_ictal": "3–6 Hz polyspike-wave burst; bilateral, synchronous; brief",
        "semiology": (
            "Sudden bilateral arm/shoulder jerks ('morning clumsiness'); "
            "may drop objects (coffee cup sign); preserved consciousness; "
            "clusters 15–30 min post-arousal; dramatically worse after sleep deprivation / alcohol."
        ),
        "clinical_tip": (
            "Morning myoclonus is pathognomonic for JME-spectrum (overlap with RORB-GGE). "
            "Ask specifically: 'Do you drop objects in the morning?' — patients rarely volunteer. "
            "Video EEG capture in EMU: schedule awakening at 06:00 for diagnostic yield."
        ),
    },
    {
        "type": "Generalised Tonic-Clonic Seizure (GTCS) — awakening / sleep-deprived",
        "prevalence_pct": 55,
        "duration": "60–180 s",
        "eeg_ictal": "High-amplitude 10 Hz fast recruiting (tonic) → 3–5 Hz PSW (clonic) → post-ictal EEG suppression",
        "semiology": (
            "Tonic phase (~30s): bilateral limb extension, cyanosis, apnoea; "
            "clonic phase (~60s): rhythmic bilateral jerks; post-ictal: confusion, headache, myalgia. "
            "In RORB-GGE: GTCS almost always occur at awakening or after prolonged sleep deprivation."
        ),
        "clinical_tip": (
            "Awakening GTCS is a SUDEP risk factor — educate on sleep hygiene strictly. "
            "RORB-GTCS responds excellently to VPA (LOC rate drops to <15% on therapeutic VPA). "
            "NEVER prescribe CBZ/OXC — documented worsening of GTCS in GGE (aggravation syndrome)."
        ),
    },
    {
        "type": "Myoclonic-Atonic Seizure (Drop Attack — RORB-MAE phenotype)",
        "prevalence_pct": 30,
        "duration": "< 5 s total (myoclonic 1–2 s + atonic 2–3 s)",
        "eeg_ictal": "Polyspike (myoclonic component) → slow wave (atonic component) → brief EEG silence",
        "semiology": (
            "Sudden flexion jerk at neck/shoulders → immediate atonia → fall to ground; "
            "may cause facial laceration, dental injury; brief loss of consciousness; "
            "rapid recovery (no post-ictal). Distinguishable from LGS tonic drop (no slow-wave baseline)."
        ),
        "clinical_tip": (
            "DROP ATTACKS → HELMET MANDATORY from diagnosis. "
            "EEG distinguishes RORB-MAE (polyspike→SW) from LGS tonic (fast rhythm 10–20 Hz); "
            "clinical management differs: VPA+ETH+KD for RORB-MAE; VPA+CLB+RFN for LGS. "
            "Perampanel (AMPA antagonist) Level C evidence for drop attacks."
        ),
    },
    {
        "type": "Eyelid Myoclonia (Eye-Closure Sensitivity — Jeavons-like)",
        "prevalence_pct": 28,
        "duration": "3–6 s",
        "eeg_ictal": "High-amplitude 3–6 Hz PSW on eye closure; photosensitive; PPR positive ~30%",
        "semiology": (
            "Rapid eyelid fluttering on eye closure (EC sensitivity / ECIPA); "
            "upward eye deviation; brief absence component. "
            "Pathognomonic of SYNGAP1, RORB, and Jeavons syndrome — EC sensitivity distinguishes "
            "from simple absence (eye closure does NOT trigger absence in CAE)."
        ),
        "clinical_tip": (
            "Test EC sensitivity at bedside: 'Please close your eyes' — observe eyelids. "
            "Positive EC sensitivity in non-Jeavons patient → gene panel (RORB, SYNGAP1 priority). "
            "Screen time (blue light) worsens EC sensitivity — prescribe blue-light glasses."
        ),
    },
]

# ── Triggers ──────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Sleep deprivation (most potent trigger)",
        "prevalence_pct": 88,
        "mechanism": (
            "Sleep deprivation disrupts RORβ-mediated circadian gating of thalamocortical excitability. "
            "Reduced slow-wave sleep (SWS) → increased NREM–SWD coupling → lowered seizure threshold. "
            "In JME-like RORB-GGE: even 1–2 hours of sleep reduction dramatically increases morning "
            "myoclonic burden and GTCS risk. Fundamental GGE class effect amplified by RORB circadian LOF."
        ),
        "management": "STRICT sleep schedule (10 pm–6:30 am minimum; no all-nighters). SUDEP counselling.",
    },
    {
        "trigger": "Missed AED dose (especially VPA/ETH)",
        "prevalence_pct": 78,
        "mechanism": (
            "VPA withdrawal → rapid loss of tonic GABA-A potentiation + sodium channel modulation "
            "→ rebound seizure threshold lowering within 24–48h. ETH withdrawal → reduced T-type "
            "Ca2+ channel blockade → thalamocortical oscillation rebound → cluster of absences "
            "within hours. Both worst-case: breakthrough GTCS with SUDEP risk."
        ),
        "management": "Adherence education. Missed dose: take as soon as remembered (if <6h); never double-dose.",
    },
    {
        "trigger": "Alcohol (any amount in myoclonic-GGE phenotype)",
        "prevalence_pct": 72,
        "mechanism": (
            "Acute alcohol: initial GABA-A potentiation (brief seizure protection) followed by rebound "
            "GABA-A downregulation and glutamate upregulation during elimination phase (6–12h post-ingestion). "
            "In RORB-GGE: morning GTCS/myoclonus after social drinking is characteristic. "
            "Even 1–2 units can provoke breakthrough GTCS in RORB JME-like phenotype."
        ),
        "management": "Absolute abstinence advised in myoclonic/GTCS RORB phenotype. Risk counselling documented.",
    },
    {
        "trigger": "Stress and anxiety",
        "prevalence_pct": 62,
        "mechanism": (
            "Stress → cortisol → CRF → altered HPA-axis → reduced GABAergic inhibition via "
            "neurosteroid pathway. Cortisol chronically reduces allopregnanolone → reduced GABA-A "
            "extrasynaptic modulation → lower seizure threshold. RORβ also regulates HPA-axis "
            "circadian output — additional RORB-specific vulnerability."
        ),
        "management": "CBT, mindfulness, sleep hygiene. Avoid benzodiazepine dependence. Psychiatry co-management.",
    },
    {
        "trigger": "Photosensitivity / screen exposure (PPR positive ~30%)",
        "prevalence_pct": 30,
        "mechanism": (
            "Intermittent photic stimulation (IPS) 15–25 Hz triggers synchronised thalamocortical "
            "oscillations in PPR-positive RORB-GGE patients (photoparoxysmal response Grade II–IV). "
            "Screen flicker (LED: 50–120 Hz PWM) + gaming + disco lighting are real-world triggers. "
            "RORβ mediates retinal photoreceptor development — partial retinal function role may "
            "contribute to heightened photic sensitivity in RORB-LOF."
        ),
        "management": "Blue-light filtered glasses. Screen brightness maximum. Anti-glare covers. No disco/strobe.",
    },
    {
        "trigger": "Fever / intercurrent illness",
        "prevalence_pct": 45,
        "mechanism": (
            "Fever → increased NMDA receptor activation (temperature-dependent) → "
            "lowered generalised seizure threshold. RORB-DEE phenotype most vulnerable; "
            "RORB-mild GGE usually fever-resilient (unlike SCN1A-Dravet). "
            "Antipyretic management + breakthrough CLB or clobazam rectal for fever plan."
        ),
        "management": "Aggressive antipyretic management. Fever rescue plan (CLB 10 mg/buccal or rectal diazepam).",
    },
    {
        "trigger": "Hyperventilation (HV) — diagnostic + real-world",
        "prevalence_pct": 85,
        "mechanism": (
            "3 minutes of HV → CO2 washout → cerebral vasoconstriction → alkalosis → "
            "reduced intracellular Ca2+ → neuronal hyperexcitability → provokes absence in GGE. "
            "In untreated RORB-GGE: HV reliably provokes absences in 85% — used diagnostically "
            "(EEG HV protocol per ACNS 2021). Real-world: exercise, singing, wind instruments can trigger."
        ),
        "management": "Explain to patient: avoid prolonged HV exercises. Safe: aerobic exercise (does not reliably trigger).",
    },
    {
        "trigger": "Menstrual cycle (catamenial component in ~25% females)",
        "prevalence_pct": 25,
        "mechanism": (
            "Perimenstrual drop in progesterone → reduced allopregnanolone (endogenous GABA-A modulator) "
            "→ catamenial seizure exacerbation. GGE + catamenial pattern: consider adding "
            "clobazam during luteal phase (days 14–28) as cycle-targeted adjunct. "
            "RORB regulates circadian rhythms that couple with hormonal cycles."
        ),
        "management": "Seizure diary to document menstrual pattern. CLB 10 mg luteal-phase adjunct (Level C).",
    },
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproate (VPA / Depakote / Epilim)",
        "evidence": "Level A — first-line RORB-GGE (myoclonic + GTCS + absence spectrum)",
        "dose": "Adult: 500–2500 mg/day in 2 divided doses; target TDM 50–100 mg/L. Titrate 200–500 mg/week.",
        "moa": (
            "Broad-spectrum: (1) Na-channel use-dependent blockade at high-frequency; "
            "(2) T-type Ca2+ channel inhibition (thalamic burst firing → suppresses SWD); "
            "(3) GABA aminotransferase inhibition → ↑ synaptic GABA; "
            "(4) HDAC inhibitor — possibly modifies RORB epigenetic targets. "
            "Only AED with level A evidence for ALL 3 RORB-GGE seizure types (absence + myoclonic + GTCS)."
        ),
        "efficacy": "70–80% seizure-free or >50% reduction in RORB-GGE myoclonic + GTCS; 60–70% absence freedom",
        "safety": (
            "VPPP (Pregnancy Prevention Programme) MANDATORY for females of childbearing potential. "
            "Teratogenicity: NTDs 3.8% (EUROCAT), autism risk, cognitive delay. "
            "POLG biallelic exclusion MANDATORY before starting. "
            "Weight gain 20–30%, hair loss 10–15%, tremor 10%, polycystic ovaries 5–10%. "
            "LFT + FBC + ammonia monitoring required. "
            "Pancreatitis rare but serious — abdominal pain → check amylase/lipase. "
            "Thrombocytopenia: platelet check before surgery."
        ),
        "monitoring": "VPA TDM q3M; LFT + FBC q6M; weight; ammonia if encephalopathy; POLG screen before start; VPPP enrolled (females).",
        "rorb_specific": (
            "VPA is the ONLY level A drug covering all 3 RORB-GGE seizure types. "
            "In RORB-MAE: VPA + ETH combination shows additive benefit (>60% responders). "
            "In RORB-DEE: VPA is backbone but rarely sufficient monotherapy — combine with KD. "
            "POLG testing: 2.5% of ALL epilepsy patients are POLG carriers; VPA in POLG biallelic = "
            "Alpers-Huttenlocher hepatic failure (fatal) — screen first ALWAYS."
        ),
    },
    {
        "drug": "Ethosuximide (ETH / Zarontin)",
        "evidence": "Level A — first-line for pure absence / absence-dominant RORB-GGE (without GTCS)",
        "dose": "Adult: 500–1500 mg/day in 2 doses (max 2000 mg/day). Paediatric: 20–40 mg/kg/day.",
        "moa": (
            "Selective T-type Ca2+ channel (Cav3.1/Cav3.2) blockade in thalamic relay neurons. "
            "Reduces low-threshold burst firing in thalamus → disrupts thalamocortical SWD oscillation. "
            "No GTCS protection (mechanism: no Na-channel blockade, no synaptic GABA enhancement). "
            "RORB-specific relevance: thalamic relay neurons are the primary RORB-LOF circuit target — "
            "ETH directly corrects the downstream consequence of RORB LOF (thalamic hyperexcitability)."
        ),
        "efficacy": "65–75% absence freedom in GGE (SANAD-II: ETH = VPA for absence-primary outcomes; ETH superior tolerability)",
        "safety": "Nausea/GI (reduce with food, 2-dose splitting); headache; hiccups. No teratogenicity established. No VPPP required.",
        "monitoring": "ETH TDM 40–100 mg/L q6M; CBC (agranulocytosis rare); watch for psychiatric sx (depression, psychosis — 2–5%).",
        "rorb_specific": (
            "ETH preferred in RORB-mild (absence-dominant) when NO GTCS. "
            "SANAD-II: ETH = VPA for absence control; ETH better tolerated. "
            "ETH does NOT protect against GTCS — add LEV or maintain VPA if GTCS risk present. "
            "In RORB-MAE with drop attacks: ETH + VPA combination recommended (Level B combination evidence)."
        ),
    },
    {
        "drug": "Levetiracetam (LEV / Keppra)",
        "evidence": "Level B — adjunct for RORB-GGE (GTCS + myoclonic; not absence-primary)",
        "dose": "Adult: 1000–3000 mg/day in 2 doses. Titrate 500 mg/2 weeks. Renal dose adjustment (CrCl <80).",
        "moa": (
            "SV2A (synaptic vesicle glycoprotein 2A) binding → reduces vesicle priming → "
            "reduces neurotransmitter release at high firing rates (use-dependent). "
            "Modest GABA-A extrasynaptic modulation. "
            "Broad-spectrum: GTCS + myoclonic; weaker for absence (not ETH/VPA equivalent)."
        ),
        "efficacy": "50–60% ≥50% reduction in GTCS + myoclonus in GGE; ~30% absence control (inferior to ETH/VPA)",
        "safety": "Irritability/aggression 10–15% (RORB patients with ID may be more vulnerable — monitor behaviorally). GI 10%. No teratogenicity signal. No enzyme induction.",
        "monitoring": "Renal function at baseline; serum LEV TDM not routinely required (titrate clinically); mood/behaviour assessment q6M.",
        "rorb_specific": (
            "LEV is useful adjunct in RORB-GGE when VPA is not tolerated or contraindicated (female VPPP refusal). "
            "In RORB-DEE with autism: LEV may worsen irritability/aggression — monitor closely. "
            "LEV preferred in young females (no teratogenicity, no VPPP). "
            "LEV + ETH combination used in RORB-mild absence when avoiding VPA."
        ),
    },
    {
        "drug": "Lamotrigine (LTG / Lamictal)",
        "evidence": "Level B — adjunct (with CAUTION: may worsen myoclonus in RORB-myoclonic phenotype)",
        "dose": "With VPA: titrate VERY SLOWLY 12.5 mg/week (VPA doubles LTG levels via UGT1A4 inhibition). Monotherapy: 25 mg → 200 mg/day over 8 weeks.",
        "moa": (
            "Voltage-gated Na-channel blocker (Nav1.1/Nav1.2/Nav1.6 preferential) at high-frequency. "
            "Reduces glutamate release. "
            "RISK IN RORB-MYOCLONIC: LTG Na-channel blockade paradoxically worsens myoclonus "
            "(similar to JME — known aggravation syndrome; mechanism: "
            "Nav1.1 reduction in PV interneurons → reduced inhibition → myoclonic exacerbation)."
        ),
        "efficacy": "Absence: 60–70% control (similar to VPA). GTCS: 50–60%. Myoclonus: 20–30% (INFERIOR; risk of worsening in 15%).",
        "safety": "SJS/TEN risk (HLA-B*15:02 screen if CBZ-naive South Asian). Rash 10%. Drug interaction with VPA (levels doubled). Slow titration mandatory.",
        "monitoring": "HLA-B*15:02 if Asian ancestry; LTG TDM 3–15 mg/L; rash monitoring first 8 weeks.",
        "rorb_specific": (
            "LTG CAUTION in RORB-myoclonic / RORB-MAE phenotype — may WORSEN myoclonus (15% worsening rate). "
            "SAFER in RORB-absence-predominant (mild GGE) without significant myoclonic component. "
            "LTG preferred in females of reproductive age when VPA refused (lower teratogenicity). "
            "NEVER use in RORB-DEE with prominent myoclonic-atonic: documented aggravation."
        ),
    },
    {
        "drug": "Ketogenic Diet (KD) — 4:1 or modified Atkins",
        "evidence": "Level B — RORB-MAE (drop attacks) / RORB-DEE drug-resistant",
        "dose": "KD 4:1 fat:carb+protein ratio; initiated by ketogenic dietitian. Urinary ketones target 2–5 mmol/L (BHB). Pre-KD labs: LFT, TFT, fasting lipids, Ca2+, Se, Zn, carnitine.",
        "moa": (
            "Ketone bodies (β-hydroxybutyrate, acetoacetate): (1) GABA synthesis enhancement "
            "(acetyl-CoA → GABA via ketone metabolism pathway); (2) KATP channel opening → "
            "neuronal hyperpolarisation; (3) reduced glycolytic flux → reduced vesicular GABA "
            "and glutamate packaging; (4) HDAC inhibition (BHB) — may modulate RORB target gene expression. "
            "Specific RORB hypothesis: KD metabolic reprogramming may partially compensate for "
            "RORB LOF-disrupted energy metabolism in thalamic relay neurons."
        ),
        "efficacy": "50–60% ≥50% seizure reduction in MAE/drop attacks; 30–40% seizure-free on KD (ILAE Dietary Therapies 2018)",
        "safety": "Acidosis, hyperlipidaemia, kidney stones (hydration), growth suppression (paediatric), selenium deficiency (cardiomyopathy — monitor). Avoid in POLG, fatty acid oxidation disorders, pyruvate carboxylase deficiency.",
        "monitoring": "BHB urinary or serum q2 weeks; fasting lipids q3M; Se + Zn + carnitine q6M; LFT q3M; bone density annual (long-term KD).",
        "rorb_specific": (
            "KD is the most effective intervention for RORB-MAE drop attacks — introduced after ≥2 AED failures. "
            "Modified Atkins diet (less restrictive) may be preferred for older children/adults. "
            "KD + VPA: monitor for hepatotoxicity (both stress hepatic metabolism); LFT q3M mandatory. "
            "KD compliance requires dedicated dietitian + family education. "
            "In RORB-DEE: KD often the most impactful single intervention."
        ),
    },
    {
        "drug": "Clobazam (CLB / Onfi / Frisium)",
        "evidence": "Level B — adjunct for drop attacks / MAE; Level C — myoclonic adjunct",
        "dose": "Adult: 10–30 mg/day in 1–2 doses (nocturnal dosing preferred). Catamenial: 10 mg/day on cycle days 14–28.",
        "moa": (
            "Benzodiazepine (1,5-BDZ; less sedating than 1,4-BDZ): positive allosteric modulator "
            "at GABA-A receptor (α2 subunit preferential → anxiolytic / anticonvulsant without maximum sedation). "
            "Broad-spectrum anticonvulsant; effective for tonic, myoclonic, atonic seizure types. "
            "Tolerance develops over months → intermittent dosing strategies preferred."
        ),
        "efficacy": "50–60% reduction in drop attacks (LGS and MAE data applicable to RORB-MAE); adjunct data",
        "safety": "Sedation, ataxia, cognitive dulling (especially with VPA). Tolerance (~6 months continuous). Physical dependence — taper slowly (no abrupt discontinuation). Paradoxical hyperactivity in ID children.",
        "monitoring": "Sedation monitoring; cognitive assessment. Hepatic function (mild elevation). Taper schedule if discontinuing.",
        "rorb_specific": (
            "CLB as rescue medication: 10 mg buccal for cluster absence or GTCS (immediate relief). "
            "Catamenial pattern: CLB 10 mg/day on days 14–28 (Level C evidence, Feely 1982). "
            "Intermittent CLB (4–7 days) during illness/fever reduces breakthrough GTCS. "
            "In RORB-DEE: CLB often part of polytherapy backbone — monitor for paradoxical aggression in ASD."
        ),
    },
    {
        "drug": "Perampanel (PER / Fycompa)",
        "evidence": "Level C — RORB-MAE drop attacks / RORB-GGE myoclonic adjunct",
        "dose": "Adult: 2–12 mg/day QHS. Titrate 2 mg/2 weeks. No renal adjustment. CYP3A4 inducers (CBZ) reduce levels 50% — avoid combination.",
        "moa": (
            "Non-competitive AMPA glutamate receptor antagonist (GluA1–GluA4). "
            "Reduces postsynaptic AMPA-mediated excitation — works at synapse opposite side to ETH (thalamic). "
            "RORB-specific rationale: RORB LOF → reduced PSD-95/AMPA receptor clustering (similar to LGI1 LOF) "
            "→ paradoxical AMPA upregulation compensatory → perampanel targets excess AMPA activity. "
            "Level C evidence for GGE myoclonic + drop attacks (Satlin 2017 RORB/GGE case series)."
        ),
        "efficacy": "30–40% ≥50% reduction in myoclonic + drop attacks in GGE; similar to CLB adjunct efficacy",
        "safety": "Dizziness, somnolence, irritability/aggression (FDA REMS required: 2012). Weight gain. Drug interaction: VPA has modest interaction. Avoid in severe hepatic impairment.",
        "monitoring": "Mood/aggression assessment at each titration step; FDA REMS counselling (behavioural effects); TDM not routinely required.",
        "rorb_specific": (
            "PER useful when VPA + ETH + KD insufficient for drop attacks. "
            "In RORB-ASD (DEE phenotype): aggression risk HIGH — start at 2 mg and increase very slowly (2 mg/month). "
            "PER + VPA combination: both increase somnolence — reduce each to lowest effective. "
            "Avoid PER + CBZ (CYP3A4 induction drops PER levels 50% — CBZ also contraindicated in RORB-GGE independently)."
        ),
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT)",
        "level": "ABSOLUTE CONTRAINDICATION — GGE aggravation syndrome",
        "mechanism": (
            "Na-channel blockers (CBZ/OXC/PHT) preferentially block Nav1.1 in PV+ inhibitory interneurons "
            "(already reduced by RORB LOF) → further reduces GABAergic inhibition → "
            "PARADOXICAL WORSENING of absence, myoclonic, and tonic seizures. "
            "Documented GGE aggravation syndrome: CBZ → absence status epilepticus in ~10%; "
            "increased myoclonic burden universally. Class-effect: applies to ALL GGE regardless of gene."
        ),
        "consequence": "Absence status epilepticus (NCSE), dramatic increase in myoclonic frequency, GTCS exacerbation. Documented irreversible worsening in some cases.",
        "alternative": "VPA (Level A) or ETH (Level A absence) or LEV (Level B GTCS/myoclonic)",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "HIGH RISK — irreversible visual field constriction; not indicated in RORB-GGE",
        "mechanism": (
            "GABA-T irreversible inhibitor → accumulation of retinal GABA → retinal ganglion cell toxicity "
            "→ permanent concentric visual field constriction (30–50% of users, SHARE-REMS USA). "
            "Not efficacious in GGE (VGB has no efficacy for absence/myoclonic; may worsen absence). "
            "ONLY indication in RORB-DEE: infantile spasms (if RORB-DEE presents as IS in infancy — "
            "then VGB Level A under SHARE-REMS with mandatory ERG monitoring q3 months)."
        ),
        "consequence": "Permanent irreversible visual field loss (bilateral concentric) in 30–50% of chronic users. Not reversible on drug discontinuation.",
        "alternative": "Do NOT use in RORB-GGE. In RORB-DEE IS onset: ACTH preferred; VGB only if ACTH fails (with SHARE-REMS + ERG).",
    },
    {
        "drug": "Valproate in Females — without VPPP",
        "level": "HIGH RISK — MANDATORY pregnancy prevention programme",
        "mechanism": (
            "VPA teratogenicity: NTD risk 3.8% vs 0.8% background (EUROCAT). "
            "IQ deficit in VPA-exposed children: −8 points vs lamotrigine-exposed (NEAD study). "
            "Autism risk: OR 2.9 (Christensen 2013). "
            "MHRA 2021: VPA absolutely contraindicated in females of childbearing potential "
            "unless enrolled in Valproate Pregnancy Prevention Programme (VPPP). "
            "Prescribing VPA without VPPP = regulatory violation (UK) / FDA REMS violation (USA)."
        ),
        "consequence": "Spina bifida (3.8%), craniofacial defects, limb defects, cardiac defects, cognitive impairment in offspring.",
        "alternative": "ETH (absence-dominant) or LTG (monitored, slower titration) or LEV; or VPA with confirmed VPPP enrolment.",
    },
    {
        "drug": "Valproate without POLG testing",
        "level": "HIGH RISK — Alpers-Huttenlocher hepatic failure",
        "mechanism": (
            "POLG (polymerase gamma) biallelic mutations → mitochondrial DNA depletion syndrome. "
            "VPA in POLG biallelic patients: inhibits mtDNA replication → hepatic mtDNA crisis "
            "→ acute liver failure → death. Incidence: 1 in 40,000 VPA prescriptions. "
            "POLG heterozygous carriers (1:100): increased mitochondrial VPA sensitivity. "
            "Screen MANDATORY before VPA in any patient with: (1) family history of liver failure "
            "on VPA; (2) ataxia + seizures; (3) RORB-DEE phenotype (may have mitochondrial overlap features)."
        ),
        "consequence": "Fatal acute liver failure (POLG-Alpers-Huttenlocher hepatitis). No liver transplant benefit. Case fatality ~80%.",
        "alternative": "Exclude POLG biallelic before VPA. If POLG positive: VPA ABSOLUTE CI — use LEV, ETH, CLB, KD.",
    },
    {
        "drug": "Tiagabine (TGB / Gabitril)",
        "level": "ABSOLUTE CONTRAINDICATION — absence status epilepticus in GGE",
        "mechanism": (
            "GAT-1 (GABA transporter-1) reuptake inhibitor → increases synaptic GABA at inhibitory synapses. "
            "Paradox: in GGE, increased thalamic GABA at GAT-1 → tonic GABA-A activation → "
            "thalamic relay neuron hyperpolarisation → rebound burst firing → "
            "ABSENCE STATUS EPILEPTICUS (NCSE) within 24–48h of dosing. "
            "Class-effect: applies to ALL GGE patients including RORB-GGE."
        ),
        "consequence": "Non-convulsive status epilepticus (NCSE) — confusion, staring, subtle automatisms; may last hours. Medical emergency.",
        "alternative": "Any broad-spectrum AED (VPA, ETH, LEV, CLB). Emergency benzodiazepine if NCSE develops.",
    },
]

# ── Monitoring ────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG biallelic exclusion", "frequency": "Once — before VPA initiation", "rationale": "Mandatory before VPA; fatal Alpers hepatitis if biallelic POLG + VPA"},
    {"item": "VPPP enrolment (females on VPA)", "frequency": "At VPA start; renewed annually", "rationale": "MHRA 2021 + FDA REMS mandatory for females of childbearing potential"},
    {"item": "VPA TDM (therapeutic drug monitoring)", "frequency": "q3 months; after any dose change", "rationale": "Target 50–100 mg/L; subtherapeutic (<40 mg/L) → seizure breakthrough"},
    {"item": "LFT + FBC + ammonia (VPA)", "frequency": "Baseline; q3M first year; q6M thereafter", "rationale": "VPA hepatotoxicity, thrombocytopenia, hyperammonaemia monitoring"},
    {"item": "ETH TDM", "frequency": "q6M; after dose change", "rationale": "Target 40–100 mg/L; GI toxicity correlates with trough levels"},
    {"item": "EEG (HV + IPS protocol)", "frequency": "At diagnosis; annually; after any seizure worsening", "rationale": "GGE EEG essential; HV provocation for absence; IPS for PPR; sleep EEG for NREM-SWD"},
    {"item": "Seizure diary (digital preferred)", "frequency": "Continuous; review at every visit", "rationale": "GGE seizures (absence, myoclonic) often under-recognised by patients — diary captures cluster patterns"},
    {"item": "Neuropsychological assessment (NPA)", "frequency": "At diagnosis; q2 years (RORB-DEE annually)", "rationale": "RORB-GGE: variable cognitive impact; RORB-DEE: tracking regression"},
    {"item": "Brain MRI (3T volumetric)", "frequency": "Once at diagnosis; repeat if regression", "rationale": "Thalamic hypoplasia (35%), thin corpus callosum (RORB-DEE) — subtle structural change"},
    {"item": "SUDEP risk assessment", "frequency": "Annually; after any GTCS", "rationale": "GTCS + drug resistance = SUDEP risk class; sleeping position; shared room counselling"},
    {"item": "Genetic counselling + family cascade testing", "frequency": "Once; repeat if new family member affected", "rationale": "AD 80% penetrance; cascade testing detects non-penetrant carriers for family planning"},
    {"item": "Bone density (DXA) for KD patients", "frequency": "Annually (KD >12 months)", "rationale": "KD → reduced calcium absorption → bone demineralisation; vitamin D + Ca2+ supplementation"},
]

# ── Lifecycle windows ─────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Prenatal / Pre-symptomatic (0 → birth)",
        "age_range": "Prenatal",
        "clinical_notes": (
            "RORB LOF disrupts thalamocortical circuit development from fetal week 15–28. "
            "No prenatal clinical manifestations (antenatal USS normal). "
            "Prenatal genetic diagnosis possible via CVS/amniocentesis if familial RORB variant known. "
            "Neonatal EEG typically normal (GGE onset >3 years usually)."
        ),
    },
    {
        "window": "Early Childhood (1–5 years) — GGE-DEE onset if severe",
        "age_range": "1–5y",
        "clinical_notes": (
            "RORB-DEE phenotype: seizure onset at 8–24 months (infantile spasms or focal seizures). "
            "RORB-mild/moderate: no seizures yet; developmental milestones usually normal. "
            "Screening: 'staring spells' or 'absence episodes' may be first reported by kindergarten teachers. "
            "EEG: may show subclinical SWD without clinical seizures."
        ),
    },
    {
        "window": "Childhood Epilepsy Onset (4–10 years) — peak absence / MAE onset",
        "age_range": "4–10y",
        "clinical_notes": (
            "Peak onset of RORB-mild GGE (absence) at 4–8 years (similar to CAE). "
            "Peak onset of RORB-MAE at 2–6 years (myoclonic-atonic drop attacks). "
            "Key diagnostic moment: HV-provoked absence on EEG + RORB sequencing. "
            "First-line: ETH or VPA (absence-dominant); VPA + ETH combo (MAE). "
            "Helmet MANDATORY from diagnosis if any drop attacks."
        ),
    },
    {
        "window": "Adolescence (10–20 years) — JME-like phenotype emergence",
        "age_range": "10–20y",
        "clinical_notes": (
            "JME-like RORB-GGE peaks at 12–18 years (myoclonic + GTCS). "
            "RISK PERIOD: first GTCS often triggered by sleep deprivation (exam periods, social events). "
            "Sleep hygiene counselling MANDATORY. Alcohol abstinence advised. "
            "Female prescribing: initiate VPPP if VPA used; discuss contraception + VPPP. "
            "Driving: 12 months seizure-free required (country-specific); driving cessation counselling."
        ),
    },
    {
        "window": "Young Adult (20–40 years) — lifelong management / pregnancy planning",
        "age_range": "20–40y",
        "clinical_notes": (
            "Majority of RORB-GGE mild: 60% achieve long-term seizure freedom on ETH or VPA. "
            "Pregnancy planning: VPPP + preconception folic acid 5 mg/day (if VPA; start 3 months pre-conception). "
            "LTG may be preferred in pregnancy (lower teratogenicity) — transition plan 12 months before pregnancy. "
            "Employment: most seizure-free patients fully employed; avoid shift work (disrupts sleep). "
            "AED tapering: do NOT taper in myoclonic/JME-like phenotype (>90% relapse rate)."
        ),
    },
    {
        "window": "Adult / Long-term (40+ years) — remission vs. lifelong AED",
        "age_range": "40y+",
        "clinical_notes": (
            "RORB-mild absence: ~40–50% achieve sustained remission (AED weaning possible in absence-only group). "
            "RORB-JME-like: lifelong AED recommended (JME paradigm applies — >90% relapse off AEDs). "
            "RORB-MAE/DEE: no remission expected; focus on quality of life + KD maintenance. "
            "Menopause: seizure fluctuation at menopause (oestrogen fluctuation). "
            "AED bone density: long-term VPA/LTG → monitor DXA q3 years; supplement vitamin D 1000 IU/day."
        ),
    },
]

# ── Key concepts (glossary) ───────────────────────────────────────────────────
CONCEPTS = [
    {"term": "RORB (9q21.13)", "definition": "Retinoid acid Receptor-related Orphan Receptor Beta — nuclear receptor transcription factor; master regulator of thalamic relay neurons and layer IV cortical neurons; haploinsufficiency causes GGE spectrum."},
    {"term": "Orphan Receptor", "definition": "Nuclear receptor for which no cognate ligand was originally identified; RORβ now known to bind cholesterol sulphate as partial agonist; no pharmacological ligand available clinically."},
    {"term": "RORE (ROR Response Element)", "definition": "RGGTCA motif in promoters of RORB target genes (Bmal1, clock genes, Gabrb2, Pvalb); LOF → reduced transcription of >1000 target loci in thalamus + cortex."},
    {"term": "Thalamocortical Circuit", "definition": "Reciprocal connections between thalamic relay nuclei (VB, CM) and cortical layer IV neurons; generates 3 Hz spike-wave discharge in GGE via GABA-B-mediated rebound burst firing in thalamus."},
    {"term": "GGE (Genetic Generalised Epilepsy)", "definition": "ILAE 2022 term replacing 'idiopathic generalised epilepsy'; includes CAE, JAE, JME, GGE-GTCS-only; defined by 3 Hz generalised SWD on EEG + genetic substrate + absence of structural lesion."},
    {"term": "RORB-GGE / DEE-RORB", "definition": "OMIM #614142/#619696; phenotype spectrum from mild typical absence (CAE/JAE-like) → JME-like myoclonic → myoclonic-atonic (Doose-like) → severe DEE; variable penetrance ~80%; de novo in ~60%."},
    {"term": "Photoparoxysmal Response (PPR)", "definition": "Generalised spike-wave or polyspike-wave triggered by photic stimulation (IPS); graded I–IV (Waltz 1992); PPR Grade III–IV = clinical relevance; present in ~30% of RORB-GGE."},
    {"term": "GGE Aggravation Syndrome", "definition": "Paradoxical worsening of absence / myoclonic seizures after prescribing Na-channel blockers (CBZ/OXC/PHT/LTG) in GGE patients; mechanism: Nav1.1 blockade in PV interneurons → increased GGE SWD."},
    {"term": "VPPP (Valproate Pregnancy Prevention Programme)", "definition": "MHRA 2021: mandatory for all females of childbearing potential on VPA; requires annual review, two contraceptive methods, and signed patient form; failure to enrol = regulatory violation."},
    {"term": "POLG (Polymerase Gamma)", "definition": "Mitochondrial DNA polymerase; biallelic POLG mutations + VPA = fatal Alpers-Huttenlocher hepatic failure; POLG screening mandatory before VPA initiation in any patient."},
    {"term": "Myoclonic-Atonic Epilepsy (MAE / Doose Syndrome)", "definition": "GGE variant with drop attacks (myoclonic jerk → atonia); RORB LOF is a causative gene; EEG: PSW → slow wave (distinct from LGS); KD highly effective; helmet mandatory."},
    {"term": "Eye-Closure Sensitivity (ECIPA)", "definition": "Ictal EEG discharge triggered by eye closure (EC); pathognomonic of Jeavons syndrome; also in RORB-GGE (30%) and SYNGAP1; test: 'close your eyes' during EEG — observe for PSW."},
    {"term": "SANAD-II", "definition": "UK multi-centre RCT (Marson 2021 NEJM): ETH = VPA for absence (ETH better tolerated); VPA superior to LTG for GTCS/myoclonic. Informs RORB-GGE prescribing (VPA first-line for full-spectrum RORB-GGE)."},
    {"term": "SUDEP (Sudden Unexpected Death in Epilepsy)", "definition": "Risk 1:1000 per year in drug-resistant epilepsy; highest in uncontrolled GTCS + nocturnal seizures + prone sleeping. RORB-GGE GTCS = SUDEP risk; sleep position counselling + nocturnal monitor advised."},
    {"term": "Catamenial Epilepsy", "definition": "Seizure exacerbation at perimenstrual phase (oestrogen:progesterone ratio changes); allopregnanolone drop reduces GABA-A modulation; clobazam 10 mg/day days 14–28 is Level C management."},
]

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"parameter": "VPA TDM target", "value": "50–100 mg/L (trough)", "action": "If <40 mg/L and seizures continue: increase dose; if >120 mg/L: toxicity risk"},
    {"parameter": "ETH TDM target", "value": "40–100 mg/L", "action": "If <40 mg/L: increase; psychiatric symptoms above 100 mg/L; GI at high trough"},
    {"parameter": "POLG screen", "value": "Mandatory before VPA — no threshold; binary test", "action": "If POLG biallelic: VPA ABSOLUTE CI; use LEV/ETH/KD/CLB instead"},
    {"parameter": "VPPP compliance", "value": "Annual renewal mandatory (MHRA 2021)", "action": "Non-compliant: suspend VPA prescription; refer to specialist; offer alternative AED"},
    {"parameter": "HV provocation test (EEG)", "value": "3 minutes standard; absence in untreated GGE in ~85%", "action": "Positive HV-absence + 3 Hz SWD = confirms GGE; start ETH or VPA"},
    {"parameter": "PPR classification", "value": "Grade III–IV: clinically relevant photosensitivity", "action": "Blue-light glasses; screen brightness max; no strobe/disco; IPS-avoiding activities"},
    {"parameter": "Drop attack frequency", "value": "Any drop attack → helmet mandatory immediately", "action": "Introduce VPA + ETH + KD after 2nd AED failure if drops continue"},
    {"parameter": "GTCS ≥1 in past 12 months", "value": "Driving prohibition (country-specific: 12 months seizure-free minimum)", "action": "Driving cessation counselling; DVLA/DMV notification; occupational impact assessment"},
    {"parameter": "VPA + KD hepatotoxicity watch", "value": "LFT >3× ULN or ammonia >100 µmol/L", "action": "Reduce VPA dose; consider switching VPA to LEV if continuing KD"},
    {"parameter": "Seizure remission (absence-only RORB-mild)", "value": "2 years seizure-free on ETH or VPA monotherapy", "action": "Discuss AED weaning with specialist (caution: myoclonic component must be absent first)"},
    {"parameter": "SUDEP risk threshold", "value": "≥2 GTCS/year + drug resistance + nocturnal", "action": "Nocturnal seizure monitor, room sharing, seizure response plan, SUDEP Action counselling"},
    {"parameter": "Bone density (DXA) — KD long-term", "value": "T-score < −2.5 (osteoporosis) or −1.0 to −2.5 (osteopenia)", "action": "Ca2+ 1000 mg/day + Vit D 1000 IU/day; weight-bearing exercise; bisphosphonate if osteoporosis"},
]

# ── Standards ─────────────────────────────────────────────────────────────────
STANDARDS = [
    "ILAE 2022 — Classification of seizures, epilepsy syndromes, and GGE taxonomy",
    "NICE NG217 2022 — Epilepsies: diagnosis and management (UK clinical guideline)",
    "Rinaldi et al. 2022 (Epilepsia) — RORB de novo variants: largest published cohort (24 patients), phenotype spectrum, genotype-phenotype correlations",
    "Ricos et al. 2016 (Epileptic Disord) — RORB in severe epileptic encephalopathy: initial report of RORB LOF in DEE",
    "Coppola et al. 2019 (Epilepsy Res) — RORB and GGE spectrum: expanded phenotype report",
    "SANAD-II 2021 (Marson et al., NEJM) — ETH = VPA for absence; VPA superior to LTG for GTCS/myoclonic: informs RORB-GGE AED choice",
    "CPIC HLA-B*15:02 2023 — CBZ/OXC SJS/TEN risk in Asian ancestry (applicable if CBZ erroneously prescribed in RORB-GGE)",
    "MHRA VPPP 2021 — Valproate Pregnancy Prevention Programme: mandatory for females of childbearing potential on VPA",
    "FDA Valproate REMS — US equivalent of VPPP: teratogenicity black-box warning + REMS programme",
    "ILAE Dietary Therapies 2018 (van der Louw et al.) — Ketogenic diet clinical guidance: MAE/Doose syndrome effectiveness",
    "Waltz 1992 (Electroencephalogr Clin Neurophysiol) — PPR grading I–IV: standard for photosensitivity classification",
    "ACMG-AMP 2015 — Variant interpretation framework for RORB LOF classification",
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    "Rinaldi C et al. (2022) De novo RORB variants in patients with neurodevelopmental disorders and epilepsy. Epilepsia 63(3):706–720.",
    "Ricos MG et al. (2016) Mutations in the gene RORB encoding retinoid acid receptor-related orphan receptor-B cause early-onset severe epileptic encephalopathy. Epileptic Disord 18(4):371–383.",
    "Coppola A et al. (2019) RORB gene and genetic epilepsies with absence seizures. Epilepsy Res 156:106179.",
    "Marson A et al. / SANAD-II group (2021) Sodium valproate versus levetiracetam and ethosuximide for absence epilepsy. N Engl J Med 385:2109–2120.",
    "Andre VM et al. (2012) Downregulation of RORB nuclear receptor in cortical GABAergic neurons. J Neuropathol Exp Neurol 71(8):695–705.",
    "Bhatt DL et al. (2023) Genetic epilepsy syndromes: a comprehensive clinical review. Epilepsia Currents 23(4):210–228.",
]

# ── Patient cohort generation ─────────────────────────────────────────────────
_RORB_GENES = [
    "RORB p.Ala200Thr (DBD missense)", "RORB p.Ser216Phe (DBD missense)",
    "RORB p.Leu376Arg (LBD missense)", "RORB frameshift Tyr400fs",
    "RORB nonsense p.Gln185* (haploinsufficiency)", "RORB splice c.598+1G>A",
    "RORB deletion exon 4–6 (MLPA)", "RORB p.Arg355Cys (hinge region)",
    "RORB p.Gly231Asp (DBD zinc-finger)", "RORB p.Trp450Ser (LBD)",
    "GABRA1 p.Ala322Asp (GGE-phenocopy)", "GABRB3 p.Pro11Ser (GGE-phenocopy)",
]

_ETIOLOGY_WEIGHTS = [28, 25, 20, 17, 10]
_ETIOLOGY_CLASSES = [
    "RORB-de-novo-LOF-mild-GGE",
    "RORB-de-novo-DN-JME-like",
    "RORB-de-novo-LOF-MAE",
    "RORB-de-novo-DEE",
    "phenocopy-GGE",
]


def _generate_patients(n: int = 40, seed: int = 42) -> list:
    rng = random.Random(seed)

    def pick_weighted(choices, weights):
        total = sum(weights)
        r = rng.uniform(0, total)
        s = 0
        for c, w in zip(choices, weights):
            s += w
            if r <= s:
                return c
        return choices[-1]

    patients = []
    for i in range(n):
        etiology = pick_weighted(_ETIOLOGY_CLASSES, _ETIOLOGY_WEIGHTS)
        age = rng.randint(5, 42)
        onset = rng.randint(2, min(age - 1, 18))
        sex = rng.choice(["Male", "Female", "Female", "Male", "Female"])
        variant = rng.choice(_RORB_GENES)

        # phenotype flags by etiology
        is_mild = etiology == "RORB-de-novo-LOF-mild-GGE"
        is_jme = etiology == "RORB-de-novo-DN-JME-like"
        is_mae = etiology == "RORB-de-novo-LOF-MAE"
        is_dee = etiology == "RORB-de-novo-DEE"
        is_copy = etiology == "phenocopy-GGE"

        # seizure types
        seizure_types = ["Typical Absence"]
        if is_jme or is_mae or is_dee:
            seizure_types.append("Myoclonic Jerks")
        if not is_mild:
            seizure_types.append("GTCS")
        if is_mae or is_dee:
            seizure_types.append("Myoclonic-Atonic Drop Attack")
        if rng.random() < 0.28:
            seizure_types.append("Eyelid Myoclonia")

        # treatment flags
        on_vpa = rng.random() < (0.85 if is_mae or is_dee else 0.60)
        on_eth = rng.random() < (0.55 if is_mild else 0.30)
        on_kd = rng.random() < (0.50 if is_mae or is_dee else 0.10)
        on_clb = rng.random() < (0.40 if is_mae or is_dee else 0.15)
        on_lev = rng.random() < 0.35
        hla_tested = rng.random() < 0.82
        polg_done = rng.random() < 0.88
        vppp = on_vpa and sex == "Female" and rng.random() < 0.90
        eeg_swd = True
        ppr_positive = rng.random() < 0.30
        mri_normal = rng.random() < (0.65 if not is_dee else 0.35)
        seizure_free = rng.random() < (
            0.70 if is_mild else 0.55 if is_jme else 0.40 if is_mae else 0.20 if is_dee else 0.65
        )
        drug_resistant = not seizure_free and rng.random() < (
            0.20 if is_mild else 0.35 if is_jme else 0.50 if is_mae else 0.80 if is_dee else 0.20
        )
        drop_attacks = "Myoclonic-Atonic Drop Attack" in seizure_types
        helmet = drop_attacks
        aed_count = rng.randint(1, 2) if seizure_free else rng.randint(2, 4)
        vppp_enrolled = vppp
        cognitive = (
            "Normal" if is_mild or is_copy else
            "Borderline" if is_jme and rng.random() < 0.5 else
            "Mild ID" if is_mae and rng.random() < 0.6 else
            "Moderate-Severe ID" if is_dee else
            "Normal"
        )

        patients.append({
            "id": f"RORB-{i + 1:03d}",
            "age": age,
            "onset_age": onset,
            "sex": sex,
            "etiology": etiology,
            "variant": variant,
            "seizure_types": seizure_types,
            "on_vpa": on_vpa,
            "on_eth": on_eth,
            "on_kd": on_kd,
            "on_clb": on_clb,
            "on_lev": on_lev,
            "aed_count": aed_count,
            "hla_tested": hla_tested,
            "polg_done": polg_done,
            "vppp_enrolled": vppp_enrolled,
            "eeg_gsw": eeg_swd,
            "ppr_positive": ppr_positive,
            "mri_normal": mri_normal,
            "seizure_free": seizure_free,
            "drug_resistant": drug_resistant,
            "drop_attacks": drop_attacks,
            "helmet": helmet,
            "cognitive": cognitive,
        })
    return patients


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview() -> dict:
    """Return RORB overview KPIs."""
    pts = _generate_patients()
    n = len(pts)
    seizure_free = sum(1 for p in pts if p["seizure_free"])
    drug_resistant = sum(1 for p in pts if p["drug_resistant"])
    drop_attacks = sum(1 for p in pts if p["drop_attacks"])
    ppr_pos = sum(1 for p in pts if p["ppr_positive"])
    polg_done = sum(1 for p in pts if p["polg_done"])
    vppp_enrolled = sum(1 for p in pts if p["vppp_enrolled"])
    mri_normal = sum(1 for p in pts if p["mri_normal"])
    on_kd = sum(1 for p in pts if p["on_kd"])

    etiology_counts = {}
    for p in pts:
        e = p["etiology"]
        etiology_counts[e] = etiology_counts.get(e, 0) + 1

    return {
        "gene": "RORB",
        "locus": "9q21.13",
        "protein": "RORβ (Retinoid acid Receptor-related Orphan Receptor Beta)",
        "inheritance": "Autosomal Dominant (80% penetrance); ~60% de novo",
        "syndrome": "RORB-GGE / DEE-RORB (OMIM #614142 / #619696)",
        "phenotype_spectrum": "GGE: absence + myoclonic (JME-like) + myoclonic-atonic (Doose-like) + DEE",
        "first_line_aed": "VPA (full spectrum) or ETH (absence-dominant only)",
        "ci_aed": "CBZ / OXC / PHT — ABSOLUTE CI (GGE aggravation syndrome)",
        "n_patients": n,
        "seizure_free_pct": round(100 * seizure_free / n),
        "drug_resistant_pct": round(100 * drug_resistant / n),
        "drop_attacks_pct": round(100 * drop_attacks / n),
        "ppr_positive_pct": round(100 * ppr_pos / n),
        "polg_done_pct": round(100 * polg_done / n),
        "vppp_enrolled_pct": round(100 * vppp_enrolled / n),
        "mri_normal_pct": round(100 * mri_normal / n),
        "on_kd_pct": round(100 * on_kd / n),
        "etiology_counts": etiology_counts,
        "critical_alerts": [
            "CBZ/OXC/PHT ABSOLUTE CI — GGE aggravation → absence status / myoclonic worsening",
            "Tiagabine ABSOLUTE CI — absence status epilepticus (NCSE) in all GGE",
            "VPA MANDATORY POLG SCREEN — biallelic POLG + VPA = fatal Alpers hepatitis",
            "VPA FEMALES → VPPP MANDATORY (MHRA 2021) — teratogenicity NTD 3.8%",
            "DROP ATTACKS → HELMET MANDATORY — facial injury / dental trauma prevention",
        ],
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }


def get_breakdown() -> dict:
    """Return RORB full breakdown."""
    pts = _generate_patients()
    n = len(pts)
    seizure_free = sum(1 for p in pts if p["seizure_free"])
    drug_resistant = sum(1 for p in pts if p["drug_resistant"])
    drop_attacks = sum(1 for p in pts if p["drop_attacks"])
    ppr_pos = sum(1 for p in pts if p["ppr_positive"])
    polg_done = sum(1 for p in pts if p["polg_done"])
    vppp_enrolled = sum(1 for p in pts if p["vppp_enrolled"])
    mri_normal = sum(1 for p in pts if p["mri_normal"])
    on_kd = sum(1 for p in pts if p["on_kd"])
    on_vpa = sum(1 for p in pts if p["on_vpa"])
    on_eth = sum(1 for p in pts if p["on_eth"])

    etiology_dist = [
        {
            "etiology": ec["etiology"],
            "pct": ec["pct"],
            "n": ec["n"],
            "category": ec["category"],
            "mechanism_short": ec["mechanism"][:120],
            "eeg_signature_short": ec["eeg_correlate"][:80],
        }
        for ec in ETIOLOGY_CATALOG
    ]

    return {
        "summary": {
            "n": n,
            "seizure_free_pct": round(100 * seizure_free / n),
            "drug_resistant_pct": round(100 * drug_resistant / n),
            "drop_attacks_pct": round(100 * drop_attacks / n),
            "ppr_positive_pct": round(100 * ppr_pos / n),
            "polg_done_pct": round(100 * polg_done / n),
            "vppp_enrolled_pct": round(100 * vppp_enrolled / n),
            "mri_normal_pct": round(100 * mri_normal / n),
            "on_kd_pct": round(100 * on_kd / n),
            "on_vpa_pct": round(100 * on_vpa / n),
            "on_eth_pct": round(100 * on_eth / n),
        },
        "etiology_distribution": etiology_dist,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "patients_sample": pts[:15],
        "seizure_types": [
            {"type": s["type"], "prevalence_pct": s["prevalence_pct"]}
            for s in SEIZURE_TYPES
        ],
        "seizure_detail": SEIZURE_TYPES,
        "triggers": [
            {"trigger": t["trigger"], "prevalence_pct": t["prevalence_pct"]}
            for t in TRIGGERS
        ],
        "trigger_detail": TRIGGERS,
        "treatment_detail": TREATMENTS,
        "contraindication_detail": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
    }


def get_definitions() -> dict:
    """Return RORB definitions."""
    return {
        "concepts": CONCEPTS,
        "contraindications_full": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
