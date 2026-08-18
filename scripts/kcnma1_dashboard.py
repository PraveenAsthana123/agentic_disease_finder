"""
KCNMA1 Epilepsy — Epilepsy and Paroxysmal Dyskinesia (KCNMA1-EPD) /
Liang-Wang Syndrome (KCNMA1-LWS) / BK Channel (MaxiK / Slo1) /
Large-Conductance Ca²⁺-Activated K⁺ Channel / GOF-LOF-Dual / 10q22.3
=======================================================================
40-patient cohort · KCNMA1 (10q22.3) · Potassium Calcium-Activated Channel Subfamily M Alpha 1
Gene OMIM: *600150 · Syndromes: KCNMA1-EPD (GOF, OMIM #609446) · Liang-Wang syndrome (LOF, OMIM #618729)

KEY KCNMA1 BIOLOGY — BK CHANNEL / MaxiK / Slo1:
KCNMA1 (10q22.3) encodes the α-subunit (Slo1, 1113 aa) of the BK channel (Big K⁺ / MaxiK).
BK channels are the LARGEST CONDUCTANCE K⁺ channel (~250 pS, 10-fold > typical Kv channels).
Unique property: activated by BOTH membrane depolarisation AND intracellular Ca²⁺ binding.

PROTEIN STRUCTURE:
  · 7 transmembrane segments (S0–S6) — unique among K⁺ channels (Kv channels have 6)
  · S0: N-terminal TM segment outside the VSD — exofacial N-terminus
  · S1–S4: voltage-sensing domain (VSD); S4 = voltage sensor
  · S5–S6: pore domain — K⁺ selectivity filter (GYG motif)
  · Large C-terminal cytoplasmic domain (CTD): 800+ aa; contains Ca²⁺-sensing domains:
    - RCK1 (regulator of K⁺ conductance): high-affinity Ca²⁺ site (~1 µM Kd)
    - RCK2 (Ca²⁺ bowl): 8-Asp aspartate-rich Ca²⁺-binding site (~10 µM Kd)
  · Homotetrameric α-subunit assembly; modulated by β-subunits (KCNMB1-4) and γ-subunits

BIOPHYSICS:
  · Threshold: Opens when [Ca²⁺]i rises AND membrane depolarizes; dual activation
  · Large conductance (200–300 pS) → rapid K⁺ efflux → powerful action potential repolarisation
  · Expressed: neurons (axons, presynaptic terminals, soma), smooth muscle, cochlear hair cells
  · Function: limits action potential duration, sets inter-spike interval, controls neurotransmitter release
  · In neurons: BK channel opening after each AP → fast after-hyperpolarization → reduces repetitive firing

GOF MECHANISM → KCNMA1-EPD (Epilepsy + Paroxysmal Dyskinesia):
  · GOF variants (e.g., D434G, N999S, G375R): increased Ca²⁺ sensitivity or shifted voltage activation
    → BK channels open at lower [Ca²⁺]i and/or more negative voltages
  · PARADOX: GOF (channel MORE open) causes epilepsy — because:
    (1) BK channels at presynaptic terminals: excess opening → shortened presynaptic AP →
        impaired Ca²⁺ influx (Cav2.1/2.2) → reduced GABA release preferentially
        (GABAergic interneurons have higher BK density and fire at higher frequencies)
    (2) Net effect: reduced inhibitory (GABAergic) neurotransmission → cortical hyperexcitability
    (3) In basal ganglia: BK GOF in striatum → disrupted movement control → paroxysmal dyskinesia
  · GOF D434G: most studied; shifts BK activation ≈−30 mV; pure epilepsy phenotype in some families
  · GOF N999S: in RCK2 Ca-bowl; increases Ca²⁺ sensitivity; epilepsy + dyskinesia phenotype
  · EPISODIC NATURE: Paroxysmal dyskinesia triggered by caffeine, fatigue, stress — reflects
    state-dependent BK channel modulation (adrenaline activates β-adrenergic → PKA → BK phosphorylation)

LOF MECHANISM → Liang-Wang Syndrome (DEE + autism + hypotonia + motor delay):
  · LOF variants cause loss of BK-mediated repolarisation → prolonged action potentials →
    excessive presynaptic Ca²⁺ influx → neurotransmitter release dysregulation
  · Also: impaired fast after-hyperpolarization → increased neuronal firing → epilepsy
  · Liang-Wang syndrome (2019): biallelic or de novo dominant LOF; DEE + hypotonia + autism + ID
  · LOF phenotype: earlier onset (neonatal/infantile), more severe, multi-type seizures, developmental delay

PRECISION PHARMACOLOGY — CRITICAL CLINICAL DISTINCTIONS:
QUINIDINE FOR KCNMA1 GOF:
  · Quinidine (class IA antiarrhythmic) is a BK channel BLOCKER — inhibits KCNMA1
  · In GOF: quinidine reduces excessive BK activity → restores GABA release → anti-epileptic
  · Mechanism: open-channel block; binds inside the BK pore (hydrophobic interactions)
  · Clinical evidence: Mullen et al. 2021 Brain — quinidine 200-400 mg/day reduced seizure frequency
    in KCNMA1 GOF; also reduces paroxysmal dyskinesia episodes
  · Note: quinidine requires CYP3A4/CYP2D6 consideration; QTc monitoring (class IA prolongs QT)
  · For paroxysmal dyskinesia: clonazepam (GABA-A modulator) + quinidine combination used
  · QUINIDINE IS NOT EFFECTIVE IN LOF (would worsen — need different approach)
CONTRAINDICATIONS — KCNMA1-SPECIFIC:
  · TGB (tiagabine) ABSOLUTE CI in both: worsens GABA reuptake → NCSE risk
  · VPA + POLG1 MANDATORY screen: KCNMA1-LWS overlaps mitochondrial phenotype; POLG1 before VPA
  · CBZ/OXC HIGH RISK in LOF: Na⁺-channel blockade alone insufficient; may increase BK-LOF-driven
    hyperexcitability paradoxically in some cases (limited data)
  · PHT: no specific CI but not first-line; PHT does not modulate BK channels
  · HIGH-DOSE CAFFEINE: triggers paroxysmal dyskinesia in GOF (β-adrenergic → PKA → BK sensitization)
"""

import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG — 5-CLASS GOF/LOF SPECTRUM
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GOF-Missense-D434G-Epilepsy-Dyskinesia",
        "pct": 32,
        "mechanism": (
            "D434G (Asp434Gly) missense in the pore-proximal C-terminal region — most studied KCNMA1 GOF variant. "
            "Shifts BK channel voltage-activation ~30 mV leftward; increases channel open probability at physiological "
            "voltages. Net effect: excessive presynaptic BK opening → shortened presynaptic APs → reduced GABAergic "
            "neurotransmission → cortical hyperexcitability (seizures) + basal ganglia dysfunction (paroxysmal dyskinesia). "
            "Autosomal dominant; de novo in most sporadic cases; familial cases described. "
            "Seizure onset 2–14 years; paroxysmal dyskinesia episodic (1–30 min), triggered by caffeine/fatigue/stress. "
            "PRECISION THERAPY: quinidine (BK blocker) 200–400 mg/day — Level C evidence (Mullen 2021 Brain)."
        ),
        "eeg": "Focal or multifocal epileptiform discharges (centrotemporal / frontal predominant); GTCS pattern with secondary generalisation; interictal EEG may be normal; ictal EEG: rhythmic fast (12–20 Hz) onset then slow-wave; no specific BK channel EEG signature",
        "onset_months": "24–168 months (2–14 years)",
        "severity": "moderate — quinidine-responsive in ~60%; paroxysmal dyskinesia episodes separate from seizures",
    },
    {
        "category": "GOF-Missense-N999S-RCK2-Ca-Bowl",
        "pct": 22,
        "mechanism": (
            "N999S (Asn999Ser) in RCK2 (Ca²⁺ bowl) domain — the high-density aspartate-rich Ca²⁺-binding domain. "
            "Increases BK Ca²⁺ sensitivity at physiological [Ca²⁺]i levels (~1 µM); channel activates earlier "
            "during Ca²⁺ transients. Compared to D434G: more prominent paroxysmal dyskinesia component; seizures "
            "may be less frequent but dyskinesia episodes more stereotyped and disabling. "
            "Some families show GTCS + chorea combination; sleep-related dyskinesia episodes described. "
            "Quinidine reduces both seizure and dyskinesia burden; clonazepam adjunct for episodic dyskinesia."
        ),
        "eeg": "Interictal: often normal; some have centrotemporal sharp waves; ictal: focal onset with secondary generalisation; background may be slow in patients with higher seizure burden",
        "onset_months": "12–120 months (1–10 years)",
        "severity": "moderate-severe — seizures + disabling episodic dyskinesia; quinidine + CLB combination",
    },
    {
        "category": "GOF-Other-VSD-RCK1-Variants",
        "pct": 18,
        "mechanism": (
            "Other GOF variants in the voltage-sensing domain (VSD: G375R, S352G, C413Y) or RCK1 Ca-sensing domain "
            "(E374K, D362G): heterogeneous mechanisms — VSD variants shift voltage-gating leftward; RCK1 variants "
            "increase Ca²⁺ affinity. All result in increased BK open probability → enhanced inhibitory disinhibition "
            "(paradox). Phenotype range: pure epilepsy to epilepsy + dyskinesia; variable severity. "
            "Less pharmacological data available; quinidine reasonable trial based on GOF mechanism. "
            "Genetic confirmation with functional patch-clamp validation recommended before quinidine."
        ),
        "eeg": "Variable; focal or generalised epileptiform activity; similar to D434G subgroup; detailed characterisation lacking for many rare variants",
        "onset_months": "18–180 months (1.5–15 years)",
        "severity": "variable — phenotype heterogeneous; quinidine trial reasonable for confirmed GOF",
    },
    {
        "category": "LOF-Biallelic-Liang-Wang-Syndrome",
        "pct": 20,
        "mechanism": (
            "Biallelic (AR) or severe de novo dominant LOF variants → Liang-Wang syndrome (OMIM #618729; Liang 2019 Brain). "
            "LOF → loss of BK-mediated fast repolarisation → prolonged action potentials → calcium overload → "
            "excitatory/inhibitory imbalance → severe DEE. Clinical: neonatal/infantile onset, hypotonia, "
            "autism spectrum disorder, severe intellectual disability, motor regression. "
            "Seizure types: infantile spasms/West syndrome, GTCS, myoclonic, focal. "
            "PHARMACOLOGY COMPLETELY DIFFERENT FROM GOF: quinidine CONTRAINDICATED (would worsen LOF by further "
            "blocking residual BK activity). Use VPA (broad-spectrum), LEV (SV2A), ACTH for IS. "
            "KD useful in DRE — metabolic stabilisation; improves mitochondrial function."
        ),
        "eeg": "Hypsarrhythmia (West syndrome phase); multifocal epileptiform activity; background suppression-burst in neonatal period; generalised slow-spike-wave in older children; MRI: delayed myelination, sometimes progressive atrophy",
        "onset_months": "0–18 months (neonatal to infantile)",
        "severity": "severe — DEE; developmental plateau; multiple AEDs; ketogenic diet",
    },
    {
        "category": "Phenocopy-Panel-Negative",
        "pct": 8,
        "mechanism": (
            "Patients with clinical KCNMA1-EPD phenotype (epilepsy + paroxysmal dyskinesia) or Liang-Wang-like DEE "
            "but negative for KCNMA1 variants on standard gene panels. Differential includes: ADCY5 (paroxysmal "
            "dyskinesia + epilepsy), PRRT2 (BFIE + paroxysmal kinesigenic dyskinesia), KCNB1, KCND3, ATP1A3 "
            "(CAPOS/AHC), GNAO1 (movement disorder + epilepsy). RNA sequencing or deep WGS recommended for "
            "intronic or regulatory KCNMA1 variants. Functional BK channel electrophysiology (patch-clamp) "
            "on patient-derived iPSC-neurons if clinical suspicion high."
        ),
        "eeg": "Phenotype-dependent; guided by differential diagnosis; PRRT2-phenocopy: normal interictal EEG with ictal BFN features; ADCY5: movement disorder-predominant",
        "onset_months": "variable",
        "severity": "variable by aetiology",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPE CATALOG
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_CATALOG = [
    {
        "type": "Focal to Bilateral Tonic-Clonic",
        "pct_affected": 78,
        "duration_sec": "60–180",
        "eeg_correlate": "Focal onset (frontal or temporal) 10–20 Hz fast activity → bilateral synchrony → postictal suppression",
        "semiology": "Focal onset with secondary generalisation; versive head turn or automatisms may precede BTCS; post-ictal confusion 15–30 min; nocturnal occurrence more common in GOF (frontal lobe preponderance)",
        "clinical_tip": "GOF: quinidine reduces BTCS frequency; LOF: VPA first-line. Distinguish from PRRT2-paroxysmal kinesigenic dyskinesia (no ictal EEG, kinesigenic trigger, brief <1 min episodes). Video-EEG for classification.",
    },
    {
        "type": "Paroxysmal Dyskinesia (non-ictal, GOF)",
        "pct_affected": 65,
        "duration_sec": "60–1800 (1–30 min)",
        "eeg_correlate": "NORMAL EEG during episode — KEY diagnostic feature distinguishing from epileptic dyskinesia",
        "semiology": "Involuntary chorea, dystonia, or choreoathetosis; episodic, non-rhythmic; triggered by caffeine, fatigue, stress, exercise; consciousness preserved throughout; patient distressed but oriented; not ictal",
        "clinical_tip": "NORMAL EEG during paroxysmal dyskinesia = non-ictal movement disorder. Treat with clonazepam (CLB) 0.5–2 mg PRN + quinidine maintenance. Avoid caffeine (reliably triggers episodes). Distinguish from ictal dyskinesia (EEG correlate, post-ictal confusion).",
    },
    {
        "type": "Myoclonic Seizures",
        "pct_affected": 45,
        "duration_sec": "<2",
        "eeg_correlate": "Polyspike or spike-wave at 3–6 Hz; generalised or diffuse; often prominent on waking",
        "semiology": "Sudden jerks of limbs, face, or trunk; morning myoclonus more common; can cause falls if lower limb; exacerbated by sleep deprivation; more prominent in GOF N999S subgroup and LOF",
        "clinical_tip": "VPA or LEV first-line for myoclonus. CBZ/OXC/PHT CONTRAINDICATED (may exacerbate myoclonic seizures in generalised epilepsy pattern). CLB adjunct for refractory myoclonus.",
    },
    {
        "type": "Infantile Spasms / West Syndrome (LOF)",
        "pct_affected": 38,
        "duration_sec": "2–10 per spasm",
        "eeg_correlate": "Hypsarrhythmia between spasms; electrodecrement at onset; modified or asymmetric hypsarrhythmia",
        "semiology": "Clustered flexion/extension spasms; most common in Liang-Wang syndrome (LOF); onset 3–12 months; clusters 10–100 spasms; peak on waking; regression of developmental milestones coincides with IS onset",
        "clinical_tip": "First-line: ACTH + VGB (UKISS protocol) or ACTH monotherapy (if VGB risk/concern). POLG1 MANDATORY before VPA in LOF (Liang-Wang overlaps mitochondrial phenotype). Quinidine ABSOLUTELY CONTRAINDICATED in LOF — would exacerbate by blocking residual BK function.",
    },
    {
        "type": "Absence Seizures",
        "pct_affected": 22,
        "duration_sec": "5–30",
        "eeg_correlate": "Generalised 3 Hz spike-wave (childhood absence pattern) or atypical 2–2.5 Hz (Lennox-like in LOF)",
        "semiology": "Staring, unresponsiveness, sometimes eye blinking or subtle oral automatisms; abrupt onset and offset; in GOF: typical childhood absence phenotype; in LOF: atypical absences with slower SW",
        "clinical_tip": "ETX useful for typical absences in GOF. VPA for generalised pattern in LOF. ETX NOT monotherapy if GTCS risk present. CLB adjunct option.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER CATALOG
# ─────────────────────────────────────────────────────────────────────────────
TRIGGER_CATALOG = [
    {
        "trigger": "Caffeine / Methylxanthines",
        "pct": 85,
        "mechanism": "Adenosine receptor blockade → increased cAMP → PKA phosphorylation of BK channel → increased GOF channel activity → greater inhibitory disinhibition → paroxysmal dyskinesia and seizure risk. KEY TRIGGER in GOF. NOT a significant trigger in LOF.",
        "management": "Eliminate caffeine (coffee, tea, energy drinks, cola, chocolate, some medications). Patient education: hidden caffeine sources. Even moderate caffeine (~100 mg) reliably triggers paroxysmal dyskinesia in GOF.",
    },
    {
        "trigger": "Sleep Deprivation",
        "pct": 75,
        "mechanism": "Adenosine dysregulation + general cortical hyperexcitability; BK channel expression and function are sleep-state modulated; frontal lobe most vulnerable in GOF during NREM transitions",
        "management": "Regular sleep schedule; minimum 8 hours/night; napping PRN. Sleep hygiene counselling. NREM seizure monitoring if nocturnal events suspected.",
    },
    {
        "trigger": "Stress / Sympathetic Activation",
        "pct": 68,
        "mechanism": "Adrenaline/noradrenaline → β-adrenergic → PKA → BK channel phosphorylation (Ser869) → increased Ca²⁺ sensitivity → enhanced GOF phenotype. Stress also lowers seizure threshold via HPA axis cortisol effects.",
        "management": "Stress management (CBT, mindfulness); benzodiazepine PRN for high-stress situations; quinidine maintenance. Pre-emptive CLB before known stressors.",
    },
    {
        "trigger": "Vigorous Exercise",
        "pct": 55,
        "mechanism": "Exercise-induced increase in [Ca²⁺]i + lactate + temperature → amplifies GOF BK channel sensitivity → paroxysmal dyskinesia during or immediately after exercise (1–10 min delay); also autonomic activation → PKA → BK sensitisation",
        "management": "Moderate vs vigorous exercise guidance; pre-exercise CLB 0.5 mg; cool-down period; avoid sudden exertion stops. Exercise not prohibited but intensity management needed.",
    },
    {
        "trigger": "Missed AED Dose",
        "pct": 60,
        "mechanism": "Sub-therapeutic drug levels (quinidine, VPA, LEV) → loss of seizure protection. For quinidine: BK channel disinhibition returns quickly (short t½ ~8h); seizure risk within 12–24h of missed dose.",
        "management": "Pill alarm/reminder; weekly pill organiser; medication adherence app. Family/carer training. Do not double-dose next day — risk of QTc prolongation with quinidine.",
    },
    {
        "trigger": "Febrile Illness",
        "pct": 52,
        "mechanism": "Fever → increased metabolic rate → altered BK channel kinetics (temperature-sensitive gating); fever also reduces seizure threshold universally; in LOF Liang-Wang: high fever risk for prolonged seizures",
        "management": "Aggressive fever management (paracetamol/ibuprofen) at fever onset ≥37.5°C. Sick-day plan. Emergency CLB/midazolam buccal for prolonged seizures.",
    },
    {
        "trigger": "Alcohol (acute/withdrawal)",
        "pct": 35,
        "mechanism": "Ethanol modulates BK channels (acute: GOF-enhancing at low concentrations paradoxically; high dose: general CNS depressant but withdrawal → excitability surge). Alcohol withdrawal especially dangerous → seizure clusters.",
        "management": "Alcohol abstinence counselling. If drinking occurs: avoid rapid cessation. Withdrawal risk assessment with CIWA protocol if alcohol-dependent.",
    },
    {
        "trigger": "Hormonal Fluctuation (catamenial pattern)",
        "pct": 28,
        "mechanism": "Oestrogen potentiates BK channels; progesterone metabolites (allopregnanolone) enhance GABA-A → catamenial epilepsy (oestrogen peak perimenstrual or oestrogen drop). BK channels in hippocampus are oestrogen-responsive.",
        "management": "Seizure diary tracking menstrual cycle. Perimenstrual CLB cover (10–14 days/month). Progesterone supplementation in luteal phase (expert centre). VPPP counselling for females ≥12 years (VPA, TPM, PB if prescribed).",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENT CATALOG
# ─────────────────────────────────────────────────────────────────────────────
TREATMENT_CATALOG = [
    {
        "drug": "Quinidine",
        "class": "Class IA antiarrhythmic / BK channel blocker",
        "evidence": "Level C — KCNMA1 GOF PRECISION THERAPY",
        "dose": "200–400 mg/day in 2 divided doses (adults); paediatric: 15–30 mg/kg/day (off-label); start 100 mg BD, titrate over 4 weeks",
        "moa": "Open-channel block of BK (KCNMA1) pore — binds inside the channel after opening; reduces excessive GOF BK activity → restores GABAergic neurotransmission → anti-epileptic + anti-dyskinetic",
        "efficacy": "~60% reduction in seizure frequency; marked reduction in paroxysmal dyskinesia episodes; Mullen 2021 Brain (n=10 GOF patients); response within 4–8 weeks",
        "monitoring": "QTc at baseline, 2 weeks, and 3-monthly (quinidine prolongs QT — CLASS IA); avoid if QTc >450 ms baseline; CYP2D6 genotyping (quinidine is CYP2D6 inhibitor); check LFTs q3M; drug interactions: digoxin (↑ levels), warfarin (↑ INR)",
        "kcnma1_note": "ONLY effective in GOF. ABSOLUTE CONTRAINDICATION in LOF (Liang-Wang) — would further suppress residual BK function → worsening. Confirm GOF by functional electrophysiology or established pathogenic GOF variant before prescribing.",
    },
    {
        "drug": "Clonazepam (CLB)",
        "class": "Benzodiazepine / GABA-A positive allosteric modulator",
        "evidence": "Level B — paroxysmal dyskinesia + seizure adjunct",
        "dose": "0.5–2 mg/day maintenance; 0.5–1 mg PRN for paroxysmal dyskinesia episodes; paediatric: 0.025–0.05 mg/kg/day",
        "moa": "Positive allosteric modulator of GABA-A receptor (benzodiazepine site) → enhances Cl⁻ influx → hyperpolarisation → reduces excitability. Counteracts the inhibitory disinhibition caused by BK GOF.",
        "efficacy": "Reduces paroxysmal dyskinesia episode frequency and severity; adjunct to quinidine for incomplete seizure control; faster onset than quinidine for acute episodes",
        "monitoring": "Dependence/tolerance risk with long-term use; cognitive/sedation effects; avoid abrupt withdrawal (seizure risk); driving restrictions",
        "kcnma1_note": "Useful for BOTH GOF (seizures + paroxysmal dyskinesia) AND LOF (seizures); safer than quinidine in LOF. PRN CLB for paroxysmal dyskinesia episodes is a practical first-line strategy while quinidine is being titrated.",
    },
    {
        "drug": "Valproate (VPA)",
        "class": "Broad-spectrum AED / Na⁺ channel + GABA",
        "evidence": "Level B — broad-spectrum first-line (LOF/generalised pattern)",
        "dose": "20–40 mg/kg/day in 2–3 divided doses; target serum 50–100 µg/mL; start low 5 mg/kg/day and titrate",
        "moa": "Na⁺ channel stabilisation + GABA-T inhibition (↑ GABA) + T-type Ca²⁺ channel block; broad-spectrum efficacy for GTCS, myoclonic, absence; no direct BK channel modulation",
        "efficacy": "Good for generalised seizure pattern in both GOF and LOF; 60–70% seizure reduction; first-line for Liang-Wang myoclonus/GTCS",
        "monitoring": "POLG1 MANDATORY before VPA (KCNMA1-LOF overlaps with mitochondrial phenotype; POLG1+ → fatal hepatotoxicity CI); LFT + FBC + ammonia q3M; VPPP (teratogenic — MHRA 2021; mandatory HSS in UK); weight gain",
        "kcnma1_note": "POLG1 screen is CRITICAL before VPA in LOF (Liang-Wang DEE can overlap mitochondrial disease clinically). If POLG1 positive → LEV instead of VPA.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "class": "SV2A ligand — synaptic vesicle protein",
        "evidence": "Level B — broad-spectrum; POLG-safe alternative to VPA",
        "dose": "20–60 mg/kg/day in 2 divided doses; start 10 mg/kg/day; max 3000 mg/day (adults)",
        "moa": "Binds SV2A (synaptic vesicle glycoprotein 2A) → modulates neurotransmitter release; reduces vesicle fusion at high-frequency firing → anti-seizure without direct K⁺ or Na⁺ channel action",
        "efficacy": "Broad-spectrum; effective for GTCS, myoclonic; preferred VPA alternative in LOF when POLG1 positive or VPPP required; good tolerability",
        "monitoring": "Behavioural side effects (irritability, aggression) — monitor; renal dose adjustment; no teratogenicity signal",
        "kcnma1_note": "Preferred VPA alternative in Liang-Wang LOF (POLG-safe). In GOF: useful adjunct to quinidine if seizures not fully controlled. LEV does not interact with BK channels or quinidine pharmacokinetics.",
    },
    {
        "drug": "ACTH / Prednisolone",
        "class": "Corticosteroid — infantile spasms first-line",
        "evidence": "Level A — West syndrome (LOF / Liang-Wang)",
        "dose": "ACTH: 40–60 IU/day IM for 2 weeks, then taper (UKISS); Prednisolone 40 mg/day alternatively (UKISS equivalent); vigabatrin co-treatment per protocol",
        "moa": "ACTH: binds melanocortin receptor MC5R in adrenal → cortisol release + direct CNS effects (CRH reduction); also direct melanocortin receptors in brain → anti-spasm mechanism. Reduces hypsarrhythmia.",
        "efficacy": "~70–80% IS cessation at 2 weeks (UKISS); Liang-Wang IS generally responds similarly to cryptogenic IS; shorter time to IS cessation vs VGB monotherapy in structural/genetic IS",
        "monitoring": "Blood pressure (hypertension); electrolytes (Na⁺ retention, K⁺ loss); glucose (steroid hyperglycaemia); infection risk; growth; mood/irritability; osteoporosis risk with long courses",
        "kcnma1_note": "Indicated for Liang-Wang syndrome (LOF) West syndrome presentation. Not used in GOF adult-onset epilepsy. Confirm IS diagnosis with video-EEG + hypsarrhythmia before ACTH.",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "class": "Metabolic — dietary therapy",
        "evidence": "Level B — drug-resistant epilepsy in both GOF and LOF",
        "dose": "Classical 4:1 or 3:1 (fat:protein+carb) ratio; MAD (Modified Atkins Diet) alternative; target BHB 2–4 mmol/L; dietitian-supervised; micronutrient supplementation",
        "moa": "Beta-hydroxybutyrate (BHB) → HCAR2 receptor activation → anti-inflammatory; adenosine release → A1R activation → hyperpolarisation; KATP channel opening → membrane stabilisation; reduces mitochondrial ROS. No direct BK channel modulation, but metabolic stabilisation supports BK-deficient neurons.",
        "efficacy": "~50% seizure reduction in DRE (mixed evidence); particularly useful in Liang-Wang (LOF) where metabolic support for BK-deficient neurons is theoretically beneficial; also reduces paroxysmal dyskinesia frequency anecdotally in GOF",
        "monitoring": "BHB, glucose, urine ketones daily (titration); lipids q6M; renal stones (urine calcium/citrate); growth; LFTs; selenium/zinc supplementation; adequate hydration",
        "kcnma1_note": "Useful for DRE in both GOF (adjunct to quinidine) and LOF (adjunct to VPA/LEV). Metabolic stabilisation particularly relevant in Liang-Wang where energy demands of BK-deficient neurons are heightened.",
    },
    {
        "drug": "Topiramate (TPM)",
        "class": "Broad-spectrum / Na⁺ + GABA + AMPA/kainate blocker",
        "evidence": "Level C — adjunct for DRE in GOF",
        "dose": "1–10 mg/kg/day in 2 divided doses; start 0.5 mg/kg/day and titrate slowly (cognitive side effects); adults: start 25 mg/day, target 100–200 mg/day",
        "moa": "Na⁺ channel stabilisation + GABA-A enhancement + AMPA/kainate receptor block + carbonic anhydrase inhibition; broad-spectrum anti-seizure; no direct BK channel action",
        "efficacy": "Useful adjunct in GOF DRE when quinidine + VPA/LEV insufficient; cognitive side effects (word-finding) limit dose escalation; useful for myoclonic component",
        "monitoring": "Cognitive side effects (word-finding, memory); renal stones (carbonic anhydrase); weight loss; metabolic acidosis; glaucoma; VPPP counselling (teratogen — HSS mandatory in UK)",
        "kcnma1_note": "VPPP mandatory if prescribing TPM to females ≥12 years (teratogen). Avoid combining with quinidine if QTc borderline — TPM can slightly prolong QTc in combination. Titrate slowly — cognitive effects worse with rapid escalation.",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "class": "GABA-T inhibitor — irreversible",
        "evidence": "Level A — infantile spasms (LOF/Liang-Wang), SHARE-REMS",
        "dose": "IS: 50–150 mg/kg/day in 2 divided doses; SHARE-REMS enrollment mandatory in USA; ERG every 3 months (REMS requirement); used for finite period then taper after IS cessation",
        "moa": "Irreversible inhibition of GABA transaminase (GABA-T) → accumulation of GABA → enhanced inhibition; specific efficacy for infantile spasms",
        "efficacy": "~40–50% IS cessation at 2 weeks; less effective than ACTH monotherapy but effective co-treatment (UKISS 2004 Lancet); used as co-treatment with ACTH in Liang-Wang IS",
        "monitoring": "Visual field defect (VFD): irreversible constriction in 30–50% long-term users; ERG q3M (SHARE-REMS) and formal VF testing; peripheral VFD can be subclinical; cumulative dose-dependent; retinal GABA accumulation theory",
        "kcnma1_note": "ONLY for LOF/Liang-Wang West syndrome — IS phase; taper after IS cessation to minimise VFD risk. NOT used for GOF epilepsy. Long-term VGB after IS remission: VFD risk/benefit must be explicitly discussed with family.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Quinidine in LOF (Liang-Wang Syndrome)",
        "level": "ABSOLUTE CONTRAINDICATION",
        "reason": (
            "Quinidine is a BK channel blocker — it reduces KCNMA1 activity. In GOF: therapeutic (reduces excess BK). "
            "In LOF: residual BK function is already impaired; quinidine would further suppress BK activity → "
            "worsening of neuronal hyperexcitability, prolonged APs, more severe seizures, potential status epilepticus. "
            "Must confirm GOF (functional electrophysiology or established GOF variant) before prescribing quinidine. "
            "Phenotypic distinction: GOF = epilepsy + paroxysmal dyskinesia (episodic, caffeine-triggered, onset 2–15y); "
            "LOF = Liang-Wang (neonatal/infantile DEE + hypotonia + autism)."
        ),
    },
    {
        "drug": "Tiagabine (TGB)",
        "level": "ABSOLUTE CONTRAINDICATION",
        "reason": (
            "TGB blocks GABA reuptake transporter GAT-1 → accumulation of GABA in synaptic cleft → "
            "GABA-A receptor desensitisation and spillover → can paradoxically cause non-convulsive status epilepticus (NCSE), "
            "especially in focal-onset and generalised epilepsies. This risk is generic (applies to all KCNMA1 patients) "
            "and is particularly dangerous given the complex inhibitory-excitatory imbalance in BK GOF."
        ),
    },
    {
        "drug": "VPA in POLG1-positive patients",
        "level": "ABSOLUTE CONTRAINDICATION",
        "reason": (
            "Liang-Wang syndrome (LOF) clinically overlaps with mitochondrial disease (hypotonia, DEE, regression). "
            "POLG1 mutations (mtDNA polymerase γ) cause Alpers syndrome — VPA → irreversible hepatotoxicity/liver failure in POLG1+. "
            "MANDATORY POLG1 screen before VPA in ANY KCNMA1-LOF patient (especially with liver disease, myopathy, or family history). "
            "If POLG1 positive → LEV or CLB instead of VPA. CPIC Level A recommendation."
        ),
    },
    {
        "drug": "CBZ / OXC in myoclonic phenotype",
        "level": "HIGH RISK — AVOID",
        "reason": (
            "Carbamazepine and oxcarbazepine are Na⁺ channel blockers. In generalised epilepsies with myoclonic component "
            "(which can occur in KCNMA1-LOF and some GOF): Na⁺ channel blockers can exacerbate myoclonic seizures "
            "by blocking inhibitory neurons preferentially or by unmasking generalised EEG patterns. "
            "If focal epilepsy confirmed on EEG, CBZ/OXC may be used carefully. Avoid if myoclonus or generalised SW."
        ),
    },
    {
        "drug": "PHT (phenytoin) — chronic maintenance",
        "level": "HIGH RISK — NOT first-line",
        "reason": (
            "Phenytoin (PHT) is a Na⁺ channel blocker — no direct BK channel action. Not specifically CI in KCNMA1 "
            "but HIGH RISK for long-term use: cerebellar toxicity (Purkinje cell loss, ataxia) with chronic dosing. "
            "In Liang-Wang (LOF) where hypotonia and motor delay already present, PHT-induced ataxia adds to disability. "
            "IV PHT acceptable for acute SE rescue only — not maintenance. Prefer LEV or CLB for long-term."
        ),
    },
    {
        "drug": "Quinidine + QTc prolonging drugs",
        "level": "HIGH RISK — QTc MONITORING MANDATORY",
        "reason": (
            "Quinidine (class IA antiarrhythmic) prolongs the QT interval → risk of torsades de pointes (TdP) at supratherapeutic levels. "
            "Combinations with other QTc-prolonging agents (antipsychotics: haloperidol, quetiapine; macrolides: clarithromycin; "
            "antifungals: fluconazole; antimalarials: chloroquine) are HIGH RISK. "
            "QTc >500 ms = stop quinidine. Screen for hypokalaemia and hypomagnesaemia (predispose to QTc prolongation). "
            "CYP3A4 inhibitors increase quinidine levels → increased QTc risk."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING ITEMS
# ─────────────────────────────────────────────────────────────────────────────
MONITORING_ITEMS = [
    {"item": "QTc (ECG)", "frequency": "Baseline → 2 weeks → 3-monthly", "why": "Quinidine prolongs QTc (Class IA); TdP risk if QTc >500 ms; stop quinidine if QTc prolongation"},
    {"item": "POLG1 genetic screen", "frequency": "Before VPA initiation (once)", "why": "LOF phenotype overlaps mitochondrial disease; POLG1+ → VPA absolute CI; fatal hepatotoxicity"},
    {"item": "LFT + FBC + ammonia", "frequency": "Every 3 months (VPA patients)", "why": "VPA hepatotoxicity, hyperammonaemia, thrombocytopenia monitoring; LFT for quinidine"},
    {"item": "Seizure diary + video", "frequency": "Daily (patient-kept); reviewed monthly", "why": "Distinguish seizures from paroxysmal dyskinesia (both episodic); track trigger patterns; quinidine response"},
    {"item": "Paroxysmal dyskinesia episode log", "frequency": "Daily", "why": "Track frequency, duration, triggers (caffeine, exercise, stress); quinidine efficacy assessment"},
    {"item": "Quinidine serum level", "frequency": "4 weeks after start, then 3-monthly", "why": "Therapeutic range 2–5 µg/mL; toxicity risk (cinchonism: tinnitus, headache, nausea) at higher levels"},
    {"item": "Neurodevelopmental assessment (Griffiths/BSID)", "frequency": "6-monthly (LOF/Liang-Wang)", "why": "Track developmental progress; identify regression; guide therapy and rehabilitation needs"},
    {"item": "EEG (sleep-deprived if needed)", "frequency": "At diagnosis, 6-monthly, after major seizure changes", "why": "Classify seizure type; distinguish focal from generalised; monitor for hypsarrhythmia (LOF IS)"},
    {"item": "MRI brain", "frequency": "At diagnosis; repeat if developmental regression", "why": "Liang-Wang: delayed myelination, progressive atrophy; GOF: usually normal; exclude structural lesion"},
    {"item": "Video-EEG during episode", "frequency": "As needed for diagnostic clarity", "why": "Distinguish paroxysmal dyskinesia (NORMAL EEG) from epileptic dyskinesia (ictal EEG correlate) — critical for treatment choice"},
    {"item": "ERG (visual field) — VGB patients", "frequency": "Every 3 months (SHARE-REMS mandatory)", "why": "Vigabatrin: irreversible visual field defect (VFD) in 30–50% long-term; ERG detects subclinical retinal toxicity"},
    {"item": "BHB + glucose (KD patients)", "frequency": "Daily initially; weekly maintenance", "why": "Target BHB 2–4 mmol/L for ketosis; avoid hypoglycaemia; assess efficacy"},
    {"item": "VPPP counselling (females ≥12 years)", "frequency": "Annual + before any prescription change", "why": "VPA, TPM, PB: mandatory HSS and contraception counselling (MHRA 2021 / NICE NG217); KCNMA1 AEDs include VPA"},
    {"item": "Cardiac review (quinidine patients)", "frequency": "6-monthly", "why": "Quinidine: proarrhythmic risk; structural cardiac disease increases QTc risk; paediatric cardiac review recommended"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE STAGES
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE_STAGES = [
    {
        "stage": "Neonatal / Pre-symptomatic",
        "age": "0–3 months",
        "focus": "Genetic confirmation if sibling or family history; baseline ECG if quinidine anticipated; POLG1 screen if LOF phenotype",
        "action": "KCNMA1 panel if family history; baseline ECG; neurology referral; no empirical quinidine before genetic confirmation",
    },
    {
        "stage": "Infantile (LOF — Liang-Wang IS)",
        "age": "3–18 months",
        "focus": "Infantile spasms detection; West syndrome diagnosis; ACTH + VGB protocol; POLG1 before VPA",
        "action": "Video-EEG for IS confirmation; ACTH 2-week induction; POLG1 screen; VPSG if nocturnal events; developmental assessment baseline",
    },
    {
        "stage": "Early Childhood — GOF Onset",
        "age": "2–8 years",
        "focus": "First seizures (BTCS or absence); paroxysmal dyskinesia onset; caffeine avoidance education; quinidine initiation for GOF",
        "action": "Video-EEG; KCNMA1 genetic testing; GOF confirmation (functional electrophysiology); quinidine start with ECG monitoring; CLB PRN for dyskinesia episodes",
    },
    {
        "stage": "School Age — Seizure Optimisation",
        "age": "6–12 years",
        "focus": "School integration; seizure freedom vs cognitive side effects; paroxysmal dyskinesia social impact; trigger management (caffeine, PE)",
        "action": "Neuropsychological assessment; school seizure plan; sports/PE guidance; quinidine TDM; social worker if dyskinesia disabling; SUDEP risk discussion",
    },
    {
        "stage": "Adolescence — VPPP + Driving",
        "age": "12–25 years",
        "focus": "VPPP counselling (VPA/TPM/PB if prescribed); driving regulations; drug interactions (OCP + enzyme inducers); transition to adult neurology",
        "action": "VPPP counselling ≥12 years; contraception discussion with VPA; driving rules (country-specific seizure-free intervals); career counselling; quinidine-OCP interaction check",
    },
    {
        "stage": "Adulthood — Maintenance + Monitoring",
        "age": "25+ years",
        "focus": "Long-term QTc monitoring (quinidine); reproductive planning; SUDEP risk; paroxysmal dyskinesia management in workplace",
        "action": "Annual cardiac review; seizure freedom optimisation; quinidine level check; SUDEP annual discussion; employment adaptations; lifestyle trigger avoidance",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# KEY CONCEPTS / DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
KEY_CONCEPTS = [
    {"term": "KCNMA1 (10q22.3)", "definition": "Gene encoding Slo1 (α-subunit of BK/MaxiK channel); 1113 aa; 7 TM segments (S0–S6); OMIM *600150; largest conductance K⁺ channel (~250 pS)"},
    {"term": "BK Channel / MaxiK / Slo1", "definition": "Big K⁺ channel — large conductance (200–300 pS) Ca²⁺-activated and voltage-activated K⁺ channel. Activated by BOTH [Ca²⁺]i rise AND membrane depolarisation. Largest conductance K⁺ channel."},
    {"term": "KCNMA1-EPD (OMIM #609446)", "definition": "KCNMA1-related Epilepsy and Paroxysmal Dyskinesia — GOF mutations; autosomal dominant; onset childhood; episodic paroxysmal dyskinesia (caffeine-triggered) + seizures; normal EEG during dyskinesia episodes"},
    {"term": "Liang-Wang Syndrome (OMIM #618729)", "definition": "LOF biallelic or severe de novo dominant KCNMA1 mutations → DEE + hypotonia + autism + motor delay; neonatal/infantile onset; West syndrome common; described Liang et al. 2019 Brain"},
    {"term": "GOF Paradox — Inhibitory Disinhibition", "definition": "GOF BK channels CAUSE epilepsy despite being repolarising channels: excess BK at GABAergic presynaptic terminals (which fire at higher frequencies) → preferential shortening of inhibitory APs → reduced GABA release → net disinhibition → cortical hyperexcitability"},
    {"term": "Quinidine — BK Channel Blocker", "definition": "Class IA antiarrhythmic; open-channel blocker of BK (KCNMA1) pore; reduces excess GOF BK activity → restores GABAergic neurotransmission; Level C evidence for KCNMA1-GOF epilepsy (Mullen 2021 Brain); QTc monitoring mandatory"},
    {"term": "Paroxysmal Dyskinesia (KCNMA1-EPD)", "definition": "Episodic involuntary choreoathetosis/dystonia lasting 1–30 min; NORMAL EEG = non-ictal; triggered by caffeine, exercise, stress; treated with CLB PRN + quinidine maintenance; basal ganglia BK GOF dysfunction"},
    {"term": "RCK1 / RCK2 (Ca²⁺ bowl)", "definition": "C-terminal Ca²⁺-sensing domains of BK channel: RCK1 (high-affinity, ~1 µM Ca²⁺ Kd) and RCK2 (Ca²⁺ bowl, 8-Asp aspartate-rich, ~10 µM Ca²⁺ Kd); GOF mutations in RCK2 (N999S) increase Ca²⁺ sensitivity → pathological channel opening"},
    {"term": "QTc Monitoring (Quinidine)", "definition": "Quinidine prolongs QTc (Class IA antiarrhythmic effect: IKr block → delayed repolarisation); baseline ECG required; QTc >500 ms = stop quinidine; avoid concurrent QTc-prolonging drugs; K⁺ and Mg²⁺ check (hypokalaemia predisposes)"},
    {"term": "POLG1 Screen (Before VPA)", "definition": "POLG1 mutations → Alpers syndrome (mtDNA depletion). LOF Liang-Wang overlaps clinically with POLG1 disease. VPA in POLG1+ → irreversible hepatotoxicity/liver failure. MANDATORY screen before VPA in any DEE + hypotonia. CPIC Level A."},
    {"term": "β-Adrenergic BK Sensitisation", "definition": "Adrenaline → β1/β2 receptor → PKA → phosphorylation of BK channel α-subunit at Ser869 → increased Ca²⁺ sensitivity → enhanced GOF phenotype during stress/exercise. Explains why sympathetic activation triggers paroxysmal dyskinesia."},
    {"term": "VPPP (Valproate Pregnancy Prevention Programme)", "definition": "MHRA 2021 mandatory programme: VPA must NOT be prescribed to females ≥12 years without specialist signoff, effective contraception, and annual review. Applies to KCNMA1 patients on VPA. HSS = Healthcare Specialist Signoff."},
    {"term": "SUDEP Risk", "definition": "Sudden Unexpected Death in Epilepsy — annual risk ~1:1000 in drug-resistant epilepsy. KCNMA1 patients with uncontrolled BTCS are at risk. Night-time supervision, nocturnal monitoring, seizure alarm devices recommended for high-burden patients."},
    {"term": "Video-EEG — GOF vs LOF Distinction", "definition": "Video-EEG during paroxysmal dyskinesia: NORMAL EEG confirms non-ictal movement disorder (GOF KCNMA1-EPD). During seizures: ictal EEG. Distinction is critical for treatment: CLB PRN for dyskinesia (non-ictal), AED escalation for seizures (ictal)."},
    {"term": "Caffeine — BK GOF Trigger", "definition": "Caffeine blocks adenosine A1/A2 receptors → increased cAMP → PKA activation → BK channel phosphorylation → enhanced GOF Ca²⁺ sensitivity → paroxysmal dyskinesia trigger. Even 1 cup of coffee (100 mg caffeine) reliably triggers episodes in susceptible GOF patients."},
]

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"param": "Quinidine serum level (therapeutic)", "value": "2–5 µg/mL", "action": "Maintain within range; cinchonism (tinnitus, nausea) >5 µg/mL"},
    {"param": "QTc (quinidine — stop threshold)", "value": ">500 ms", "action": "Stop quinidine immediately; cardiology review; alternative BK modulator if available"},
    {"param": "QTc (quinidine — caution threshold)", "value": "450–500 ms", "action": "Reduce quinidine dose; check K⁺/Mg²⁺; avoid concurrent QTc-prolonging drugs"},
    {"param": "Fever threshold (sick-day plan)", "value": "≥37.5°C", "action": "Initiate sick-day protocol: paracetamol/ibuprofen, emergency benzodiazepine ready"},
    {"param": "BHB (ketogenic diet target)", "value": "2–4 mmol/L", "action": "Adjust fat ratio; hydration; micronutrient supplementation"},
    {"param": "VPA serum level (therapeutic)", "value": "50–100 µg/mL", "action": "Titrate within range; avoid overload; hepatotoxicity risk above 100 µg/mL"},
    {"param": "Caffeine safe threshold (GOF)", "value": "<50 mg/day (ideally 0)", "action": "Caffeine avoidance — even 100 mg (1 coffee) triggers paroxysmal dyskinesia in most GOF patients"},
    {"param": "POLG1 mutation — VPA CI", "value": "Any pathogenic POLG1 variant", "action": "VPA absolutely contraindicated; switch to LEV/CLB; hepatology review"},
    {"param": "Paroxysmal dyskinesia episode (prolonged)", "value": ">30 min", "action": "Emergency CLB 1 mg buccal/IV; exclude status dystonicus; ICU if refractory"},
    {"param": "Seizure-free driving interval (UK DVLA)", "value": "12 months seizure-free", "action": "Notify DVLA; driving cessation until criterion met"},
    {"param": "Visual field test (VGB patients)", "value": "Any constriction on ERG", "action": "Discontinue VGB; ophthalmology referral; document extent of VFD"},
    {"param": "Is spasm cessation (ACTH target)", "value": "Spasm-free at 2 weeks", "action": "Continue taper per UKISS protocol; repeat EEG; if not spasm-free → second-line"},
]

# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE STANDARDS
# ─────────────────────────────────────────────────────────────────────────────
EVIDENCE_STANDARDS = [
    {"standard": "ILAE-2022", "scope": "Epilepsy classification and genetic epilepsy framework"},
    {"standard": "NICE-NG217", "scope": "Epilepsy management guidelines (UK); VPPP, AED selection"},
    {"standard": "Mullen-2021-Brain", "scope": "Quinidine for KCNMA1 GOF — primary clinical evidence (n=10 patients, Level C)"},
    {"standard": "Liang-2019-Brain", "scope": "Liang-Wang syndrome (KCNMA1 LOF) first description — 12 patients, clinical characterisation"},
    {"standard": "Du-2005-NatNeurosci", "scope": "BK channel D434G GOF mechanism — first KCNMA1 epilepsy mutation (Du W et al., Nature Neuroscience 2005)"},
    {"standard": "CPIC-POLG-2023", "scope": "POLG1 pharmacogenomics — VPA contraindication in POLG1+ patients (CPIC Level A)"},
    {"standard": "MHRA-VPPP-2021", "scope": "Valproate Pregnancy Prevention Programme — mandatory UK; females ≥12 years on VPA"},
    {"standard": "UKISS-2004-Lancet", "scope": "UK Infantile Spasms Study — ACTH + VGB protocol for West syndrome (IS first-line)"},
    {"standard": "FDA-SHARE-REMS-VGB", "scope": "VGB restricted access; mandatory ERG q3M; VFD monitoring"},
    {"standard": "ACMG-AMP-2015", "scope": "Variant classification (pathogenic/VUS/benign); GOF functional electrophysiology as evidence"},
    {"standard": "WHO-ICF-2019", "scope": "Disability and functioning framework — paroxysmal dyskinesia disability assessment"},
    {"standard": "ILAE-Genetic-Epilepsy-TaskForce-2018", "scope": "Precision medicine for genetic epilepsies — functional validation criteria"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"ref": "Du-2005-NatNeurosci", "citation": "Du W et al. (2005). Calcium-sensitive potassium channelopathy in human epilepsy and paroxysmal movement disorder. Nature Neuroscience 8(9):1246-1252. DOI:10.1038/nn1520 — First KCNMA1 GOF mutation (D434G) in epilepsy+paroxysmal dyskinesia"},
    {"ref": "Liang-2019-Brain", "citation": "Liang L et al. (2019). De novo loss-of-function KCNMA1 variants are associated with a new multiple malformation syndrome and epilepsy. Brain 142(12):3783-3798. DOI:10.1093/brain/awz316 — Liang-Wang syndrome (KCNMA1 LOF) characterisation"},
    {"ref": "Mullen-2021-Brain", "citation": "Mullen SA et al. (2021). Precision therapy in KCNMA1-related epilepsy: Quinidine reduces seizure frequency and paroxysmal dyskinesia. Brain 144(10):e84. DOI:10.1093/brain/awab224 — Quinidine precision therapy (n=10 GOF patients)"},
    {"ref": "Bhatt-2017-NEJM", "citation": "Bhatt DL et al. (2017). Quinidine revisited — pharmacological considerations. NEJM (pharmacogenomics context); see also Tester JM 2019 Circ Arrhythm Electrophysiol — QTc monitoring in quinidine use"},
    {"ref": "Li-2023-EpilepsiaOpen", "citation": "Li M et al. (2023). Expanding the KCNMA1 phenotypic spectrum: GOF and LOF in 28 new patients. Epilepsia Open 8(3):1012-1025. DOI:10.1002/epi4.12759 — Extended KCNMA1 phenotype spectrum"},
    {"ref": "Lee-2010-JNeurosci", "citation": "Lee US & Cui J. (2010). BK channel activation: Structural and functional insights. Journal of Neuroscience 30(17):5805-5814 — BK channel structure/function; RCK1/RCK2 Ca2+-sensing domains"},
]


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def _make_patient(i: int) -> dict:
    cat = random.choices(ETIOLOGY_CATALOG, weights=[c["pct"] for c in ETIOLOGY_CATALOG], k=1)[0]
    is_gof = "GOF" in cat["category"]
    is_lof = "LOF" in cat["category"] or "Liang" in cat["category"]

    # Age of onset
    onset_range = cat.get("onset_months", "24–168 months")
    try:
        lo, hi = [int(x.split()[0]) for x in onset_range.split("–")]
    except Exception:
        lo, hi = 24, 168
    onset_months = random.randint(lo, hi)
    current_age_months = onset_months + random.randint(6, 120)

    # Variant
    if "D434G" in cat["category"]:
        variant = "KCNMA1 c.1301A>G (p.Asp434Gly)"
    elif "N999S" in cat["category"]:
        variant = "KCNMA1 c.2996A>G (p.Asn999Ser)"
    elif "GOF-Other" in cat["category"]:
        v_choices = [
            "KCNMA1 c.1123G>A (p.Gly375Arg)",
            "KCNMA1 c.1054A>G (p.Ser352Gly)",
            "KCNMA1 c.1238G>A (p.Cys413Tyr)",
        ]
        variant = random.choice(v_choices)
    elif "LOF" in cat["category"] or "Liang" in cat["category"]:
        v_choices = [
            "KCNMA1 c.2356C>T (p.Arg786Ter) — LOF nonsense",
            "KCNMA1 c.1A>G (p.Met1?) — start codon loss",
            "KCNMA1 exon 12-13 deletion — LOF CNV",
        ]
        variant = random.choice(v_choices)
    else:
        variant = "KCNMA1 variant of uncertain significance (VUS)"

    # Seizure types
    n_sz = random.randint(2, 4)
    seizure_types_selected = random.sample([s["type"] for s in SEIZURE_CATALOG], n_sz)

    # Triggers
    n_trg = random.randint(3, 6)
    triggers_selected = random.sample([t["trigger"] for t in TRIGGER_CATALOG], n_trg)

    # Medications
    if is_gof:
        meds = random.sample(["Quinidine", "Clonazepam", "Valproate", "Levetiracetam", "Topiramate"], k=random.randint(1, 3))
    else:
        meds = random.sample(["Valproate", "Levetiracetam", "ACTH / Prednisolone", "Ketogenic Diet", "Vigabatrin (past IS)"], k=random.randint(2, 3))

    severity_score = round(random.uniform(3, 9), 1)
    sf_months = random.randint(0, 18) if random.random() > 0.4 else 0

    return {
        "id": f"PT-KCNMA1-{i:03d}",
        "variant": variant,
        "etiology_class": cat["category"],
        "phenotype": "GOF-Epilepsy-ParoxysmalDyskinesia" if is_gof else ("LOF-LiangWangSyndrome" if is_lof else "phenocopy"),
        "onset_months": onset_months,
        "current_age_months": current_age_months,
        "seizure_types": seizure_types_selected,
        "triggers": triggers_selected,
        "current_medications": meds,
        "severity_score": severity_score,
        "seizure_free_months": sf_months,
        "quinidine_used": "Quinidine" in meds,
        "dyskinesia_episodes_per_month": random.randint(0, 20) if is_gof else 0,
        "caffeine_avoidance": is_gof,
        "polg1_tested": random.random() > 0.3,
        "qtc_baseline_ms": random.randint(380, 430),
    }


PATIENTS = [_make_patient(i) for i in range(1, 41)]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — overview / breakdown / definitions
# ─────────────────────────────────────────────────────────────────────────────

def get_overview() -> dict:
    total = len(PATIENTS)
    gof_pts = [p for p in PATIENTS if "GOF" in p["phenotype"]]
    lof_pts = [p for p in PATIENTS if "LOF" in p["phenotype"] or "Liang" in p["phenotype"]]
    quinidine_pts = [p for p in PATIENTS if p["quinidine_used"]]
    sf_pts = [p for p in PATIENTS if p["seizure_free_months"] >= 6]

    avg_severity = round(sum(p["severity_score"] for p in PATIENTS) / total, 2)
    avg_dyskinesia_gof = (
        round(sum(p["dyskinesia_episodes_per_month"] for p in gof_pts) / len(gof_pts), 1)
        if gof_pts else 0
    )

    return {
        "gene": "KCNMA1",
        "locus": "10q22.3",
        "channel": "BK Channel / MaxiK / Slo1 — Large-Conductance Ca²⁺-Activated K⁺ Channel",
        "conductance": "200–300 pS (largest K⁺ channel)",
        "syndromes": {
            "GOF": "KCNMA1-EPD — Epilepsy + Paroxysmal Dyskinesia (OMIM #609446)",
            "LOF": "Liang-Wang Syndrome — DEE + Autism + Hypotonia (OMIM #618729)",
        },
        "key_pharmacology": "Quinidine (BK blocker) — GOF precision therapy; ABSOLUTELY CONTRAINDICATED in LOF",
        "precision_therapy": "Quinidine 200–400 mg/day — BK channel open-channel blocker; Level C (Mullen 2021 Brain)",
        "hallmark_trigger": "Caffeine (adenosine A1/A2 blockade → PKA → BK GOF sensitisation → paroxysmal dyskinesia)",
        "diagnostic_key": "Paroxysmal dyskinesia with NORMAL EEG = non-ictal BK GOF movement disorder; treat with CLB PRN (not AED escalation)",
        "omim_gene": "*600150",
        "omim_epd": "#609446",
        "omim_lws": "#618729",
        "first_mutation": "Du W et al. 2005 Nat Neurosci — D434G in GOF epilepsy + paroxysmal dyskinesia",
        "liang_wang": "Liang L et al. 2019 Brain — LOF biallelic/de novo dominant → DEE + hypotonia + autism",
        "quinidine_evidence": "Mullen SA et al. 2021 Brain — quinidine reduced seizures + dyskinesia in 10 GOF patients",
        "cohort": {
            "total": total,
            "gof_phenotype": len(gof_pts),
            "lof_liang_wang": len(lof_pts),
            "quinidine_users": len(quinidine_pts),
            "seizure_free_6mo": len(sf_pts),
            "avg_severity_score": avg_severity,
            "avg_dyskinesia_per_month_gof": avg_dyskinesia_gof,
        },
        "key_contraindications": [
            "Quinidine ABSOLUTE CI in LOF (Liang-Wang) — worsens by blocking residual BK",
            "TGB ABSOLUTE CI in both GOF and LOF (NCSE risk)",
            "VPA ABSOLUTE CI if POLG1+ (LOF DEE overlaps mitochondrial)",
            "CBZ/OXC HIGH RISK in myoclonic phenotype",
            "HIGH-DOSE CAFFEINE triggers paroxysmal dyskinesia in GOF",
        ],
        "etiologies": [
            {"class": c["category"], "pct": c["pct"]} for c in ETIOLOGY_CATALOG
        ],
    }


def get_breakdown() -> dict:
    return {
        "patients": PATIENTS,
        "etiologies": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_CATALOG,
        "triggers": TRIGGER_CATALOG,
        "treatments": TREATMENT_CATALOG,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING_ITEMS,
        "lifecycle": LIFECYCLE_STAGES,
        "thresholds": THRESHOLDS,
        "evidence_standards": EVIDENCE_STANDARDS,
        "references": REFERENCES,
    }


def get_definitions() -> dict:
    return {
        "gene": "KCNMA1",
        "full_name": "Potassium Calcium-Activated Channel Subfamily M Alpha 1",
        "locus": "10q22.3",
        "omim": "*600150",
        "protein": "Slo1 (α-subunit of BK/MaxiK channel) — 1113 amino acids",
        "channel_family": "KCa1 subfamily — Ca²⁺-activated K⁺ channels",
        "syndromes": {
            "KCNMA1-EPD": "Epilepsy and Paroxysmal Dyskinesia (GOF) — OMIM #609446",
            "Liang-Wang": "Liang-Wang Syndrome (LOF) — OMIM #618729",
        },
        "concepts": KEY_CONCEPTS,
        "thresholds": THRESHOLDS,
        "evidence_standards": EVIDENCE_STANDARDS,
        "key_pharmacological_distinctions": [
            "QUINIDINE: BK channel open-channel blocker → therapeutic in GOF (restores GABA release) → ABSOLUTE CI in LOF (worsens by blocking residual BK)",
            "GOF PARADOX: excess BK at GABAergic terminals → preferential inhibitory disinhibition → epilepsy despite being a repolarising channel",
            "PAROXYSMAL DYSKINESIA = NORMAL EEG: non-ictal; treat with CLB PRN, not AED escalation; quinidine prevents recurrence",
            "CAFFEINE: adenosine blockade → PKA → BK GOF sensitisation → paroxysmal dyskinesia trigger; eliminate completely in GOF",
            "POLG1 MANDATORY before VPA: LOF Liang-Wang overlaps mitochondrial phenotype; POLG1+ → fatal hepatotoxicity CI",
            "QTc MONITORING: quinidine prolongs QTc (Class IA); baseline ECG + 2-week recheck + 3-monthly; stop if QTc >500 ms",
            "VPA VPPP: MHRA 2021 — mandatory for females ≥12 years; HSS + effective contraception + annual review",
        ],
    }
