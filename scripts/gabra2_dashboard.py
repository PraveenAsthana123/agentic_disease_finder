"""
GABRA2 Epilepsy — GGE / GEFS+ Spectrum / Alcohol-Sensitive / GABA-A α2 Subunit
=================================================================================
40-patient cohort · GABRA2 (4p12) · GABA-A receptor α2 subunit · AD / de novo

GABRA2 BIOLOGY:
GABRA2 (4p12) encodes the α2 subunit of the ionotropic GABA-A receptor pentamer.
Unlike α1 (enriched in adult forebrain synapses, mediating sedation), the α2 subunit
predominates in:
  1. CORTICAL LAYER 2/3 PYRAMIDAL NEURONS — particularly at the axon initial segment
     (AIS), where α2-containing GABA-A receptors provide powerful perisomatic shunting
     inhibition (inhibitory conductance near the spike-initiating zone).
  2. LIMBIC SYSTEM (hippocampus CA1/CA2 pyramidal layer, amygdala) — critical for
     emotional processing and seizure propagation gating.
  3. STRIATUM / BASAL GANGLIA — gate motor output and suppress aberrant rhythms.
  4. DEVELOPING CORTEX — α2 predominates in immature brain before the postnatal
     α2→α1 developmental switch (opposite to α1: α2 high early, declines with age).

GABA-A RECEPTOR α2 SUBUNIT — KEY FUNCTIONAL ROLES:
  1. ANXIOLYTIC BENZODIAZEPINE EFFECTS: Classical BZDs bind at the α/γ2 interface.
     α2-containing receptors (α2β3γ2) mediate ANXIOLYTIC and MYORELAXANT effects of
     BZDs — distinct from α1 (sedation/hypnosis). In GABRA2 LOF, anxiolytic BZD
     efficacy is REDUCED; sedative effect (α1-mediated) preserved. Clinically: standard
     BZD rescue remains effective for GTCS termination (via α1) but ANXIOLYTIC
     pre-treatment (for catamenial / stress-triggered clusters) is less reliable.
  2. AIS PERISOMATIC INHIBITION: α2 at the axon initial segment gates pyramidal cell
     excitability directly at the spike trigger zone. GABRA2 LOF → reduced AIS shunting
     → lower seizure threshold → cortical hyperexcitability → GGE spectrum.
  3. ALCOHOL MODULATION (CRITICAL CLINICAL LINK): GABRA2 polymorphisms (rs279858,
     rs279826, haplotype block 4p12) are among the most replicated genetic associations
     with alcohol dependence (Edenberg et al. 2004; Covault et al. 2004). Mechanism:
     α2-containing receptors are major targets of ethanol's GABA-A potentiation; GABRA2
     variants reduce alcohol-induced GABA-A potentiation → patients require more alcohol
     for anxiolytic effect → alcohol misuse. In GABRA2 LOF patients who also misuse
     alcohol: ALCOHOL WITHDRAWAL → acute loss of the only remaining GABAergic
     compensation → SEVERE WITHDRAWAL SEIZURES. Alcohol withdrawal seizures in GABRA2
     LOF patients are dramatically more severe than in normal population.
  4. DEVELOPMENTAL ROLE (GABA SWITCH): α2 is the dominant α-subunit in immature
     neurons, declining postnatally. Neonatal/infantile GABRA2 LOF → severe DEE (absence
     of early inhibitory maturation); adult-onset variants → milder GGE phenotype.

CLINICAL PHENOTYPE SPECTRUM:
  De novo GABRA2 LOF/missense → GEFS+ spectrum, DEE (Dravet-like, infantile-onset),
  or severe GGE. Familial (AD) GABRA2 variants → JME-like, CAE, or GTCS-only GGE.
  GABRA2 variants associated with alcohol dependence → alcohol-sensitive GGE (seizures
  precipitated or worsened by alcohol use/withdrawal). Penetrance AD ~80%.

INHERITANCE AND GENETICS:
  AD inheritance, ~80% penetrance. De novo in severe cases (GEFS+/DEE); familial in
  GGE/JME/CAE spectrum. OMIM: GABRA2 variants in GGE/GEFS+ spectrum. No specific OMIM
  DEE number; classified under GGE/GEFS+ spectrum. Incidence: <200 published families.
  pLI ~0.97 (high intolerance to LoF).

CONTRAINDICATIONS IN GABRA2 EPILEPSY:
  1. CARBAMAZEPINE / OXCARBAZEPINE / PHENYTOIN (ABSOLUTE CI — GGE aggravation):
     NaV1-preferring blockers in GGE → block NaV1.1 in PV+ interneurons → interneuron
     disinhibition → aggravation of absence, myoclonic, and GTCS. ABSOLUTE CI in all
     generalised GABRA2 epilepsy. Documented in GGE-aggravation literature (Panayiotopoulos 2005).
  2. TIAGABINE (ABSOLUTE CI — NCSE):
     GAT-1 inhibition → prolonged GABA in synapse → paradoxical NCSE in patients with
     diffuse cortical involvement. ABSOLUTE CI in ALL GABRA2 epilepsy.
  3. LAMOTRIGINE without myoclonic EEG assessment (MODERATE CI — 15-20% worsening):
     LTG blocks NaV1.1 in PV+ interneurons → myoclonic aggravation in JME/myoclonic GGE.
     LTG may be used in pure CAE without myoclonic component (Level B), but REQUIRES
     pre-treatment EEG to confirm no myoclonic spike-wave. 15-20% risk of myoclonic
     aggravation if prescribed without EEG confirmation.
  4. VALPROATE in females without VPPP (HIGH — MHRA 2021):
     VPA teratogenic (NTD 1-2%, fetal valproate syndrome). MHRA 2021: VPA ONLY in
     females of childbearing potential if enrolled in Valproate Pregnancy Prevention
     Programme (VPPP) with annual assessment + 2 contraception methods confirmed.
     HIGH risk if prescribed without VPPP enrolment.
  5. VALPROATE without POLG screen (HIGH — Alpers-Huttenlocher risk):
     POLG mutations + VPA → Alpers-Huttenlocher syndrome (hepatic failure, fatal).
     POLG screen MANDATORY before VPA initiation in ALL patients.
  6. ALCOHOL — HIGH RISK (seizure trigger + withdrawal):
     Alcohol consumption is BOTH an acute seizure trigger (GABAergic euphoriant effect
     followed by rebound excitability) AND a severe withdrawal trigger (GABRA2 LOF →
     markedly reduced alcohol-withdrawal seizure threshold). GABRA2-specific alcohol
     counselling MANDATORY. Document alcohol-withdrawal seizure risk in all emergency plans.
"""
import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GABRA2-de-novo-LOF-GEFS+-severe",
        "pct": 32,
        "etiology": "De novo LOF — GEFS+ spectrum / severe GGE (febrile SE, afebrile GTCS)",
        "mechanism": (
            "Truncating or severe missense de novo LOF → haploinsufficiency → reduced α2 "
            "surface expression → loss of AIS perisomatic inhibition → cortical/limbic "
            "disinhibition → low seizure threshold. Onset: FS in infancy → GEFS+ (afebrile "
            "GTCS persist beyond 6 years) → juvenile GGE. Drug resistance 40-50% in severe de novo."
        ),
        "typical_variants": "Frameshift, splice-site, truncating; severe missense (D219N-like)",
        "eeg_signature": "Generalised spike-wave 3-4 Hz; polyspike-wave at awakening; photic response 20%",
        "phenotype": "GEFS+ → afebrile GTCS / absence / myoclonic; moderate-severe GGE; partial drug-resistance",
    },
    {
        "category": "GABRA2-familial-AD-JME-like",
        "pct": 25,
        "etiology": "Familial AD — JME-like phenotype (myoclonic jerks + awakening GTCS)",
        "mechanism": (
            "Inherited heterozygous GABRA2 missense variants in families with JME-like epilepsy. "
            "α2 haploinsufficiency → reduced AIS shunting + reduced α2-BDZ anxiolytic site → "
            "susceptibility to awakening myoclonus and GTCS. Sleep deprivation is the dominant "
            "trigger (α2 AIS inhibition most critical during arousal transitions)."
        ),
        "typical_variants": "Missense (R82C, T336I, family-specific variants); variable penetrance ~80%",
        "eeg_signature": "4-6 Hz polyspike-wave (PSW) on awakening; photosensitivity 25-30%; normal background",
        "phenotype": "JME-like: morning myoclonus → GTCS on awakening; normal IQ; excellent VPA response",
    },
    {
        "category": "GABRA2-familial-AD-CAE",
        "pct": 22,
        "etiology": "Familial AD — Childhood Absence Epilepsy (CAE) / GGE-absence phenotype",
        "mechanism": (
            "Inherited GABRA2 variants → mild α2 surface reduction (~30-40%) → thalamocortical "
            "3 Hz SWD generation threshold lowered. Predominantly affects limbic and cortical layer "
            "2/3 circuits where α2 is highest density → absence phenotype. Familial clustering with "
            "CAE + GTCS. Female sex predilection (oestrogen modulation of GABA-A γ subunit)."
        ),
        "typical_variants": "Missense (N79S-like, R136C, T196I); penetrance variable 60-80%",
        "eeg_signature": "3 Hz generalised spike-wave, abrupt onset/offset; hyperventilation-activated; normal background",
        "phenotype": "CAE: typical absences 5-10y → remission in 70% by adolescence; 30% evolve to JME or GGE",
    },
    {
        "category": "GABRA2-de-novo-severe-DEE",
        "pct": 15,
        "etiology": "De novo severe — DEE (Dravet-like, infantile-onset, drug-resistant)",
        "mechanism": (
            "Severe de novo GABRA2 LOF in early development (α2 predominates in immature brain). "
            "Loss of neonatal GABA-A α2 inhibitory maturation → severe cortical disinhibition → "
            "infantile-onset DEE. Phenotypically Dravet-like (febrile SE, multifocal seizures, "
            "regression) but SCN1A negative. MRI: may show cortical dysmaturation or normal."
        ),
        "typical_variants": "Severe de novo missense (TM domain, ECD interface); some truncating with incomplete NMD",
        "eeg_signature": "Multifocal spikes, burst-suppression (neonatal), hypsarrhythmia (infantile), poly-SWD",
        "phenotype": "Dravet-like DEE: febrile SE, drug-resistant, intellectual disability, regression",
    },
    {
        "category": "GABRA2-negative-phenocopy-GGE",
        "pct": 6,
        "etiology": "GABRA2-negative phenocopy (GGE/JME/CAE without GABRA2 variant)",
        "mechanism": (
            "Clinical GGE/JME/CAE or GEFS+ without GABRA2 variant. Differential: GABRA1-GGE, "
            "GABRB3-GGE, GABRG2-GEFS+, SCN1A-GEFS+, SYNGAP1-GGE-like, SLC2A1-GLUT1. "
            "Panel testing required. Phenocopy prevalence highlights that GABRA2 accounts "
            "for a minority of clinical GGE (~2-4% of GGE families)."
        ),
        "typical_variants": "None — reassign to alternative genetic diagnosis after panel",
        "eeg_signature": "Variable; depends on actual aetiology",
        "phenotype": "GGE/JME/CAE without GABRA2 — comprehensive panel needed",
    },
]

# ── Seizure Types ──────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Febrile Seizures / GEFS+ (Febrile Seizures Plus)",
        "prevalence_pct": 82,
        "semiology": (
            "Generalised tonic-clonic ± focal onset with fever. FSE (>30 min) in 25-30% "
            "of severe de novo GABRA2. In GEFS+ phenotype: FS persist beyond age 6 (FS+) → "
            "afebrile GTCS develop. Temperature action threshold: 38°C standard (not lowered "
            "like K289M-GABRG2 which is 37.5°C). Clustering with febrile illness."
        ),
        "eeg_pattern": (
            "Ictal: generalised spike-wave or focal onset with rapid secondary generalisation. "
            "Interictal: often normal early; 3-4 Hz GSW may develop interictally with age."
        ),
        "clinical_tip": (
            "RESCUE: Standard midazolam 0.2 mg/kg buccal (α1 BDZ binding intact — rescue dose "
            "NOT adjusted unlike GABRA1/GABRG2 LOF). Fever: paracetamol + early rescue at "
            "38°C. Fever management plan MANDATORY. Document GEFS+ pedigree — first-degree "
            "relatives may have FS/FS+/GGE. Genetic counselling for all family members."
        ),
    },
    {
        "type": "Typical Absence Seizures (3 Hz GSW)",
        "prevalence_pct": 68,
        "semiology": (
            "Abrupt onset staring + behavioural arrest 5-30 seconds. Eyelid fluttering ± mild "
            "automatisms. Immediate recovery. Triggered by hyperventilation (diagnostic). "
            "Clustering in the morning. More frequent in familial CAE/JME-like phenotype. "
            "GABRA2 absence: subtle oral automatisms less prominent than typical CAE (α2 "
            "limbic distribution → fewer prominent oroalimentary automatisms vs α1)."
        ),
        "eeg_pattern": (
            "Symmetric 3 Hz GSW, abrupt onset/offset, amplitude highest frontally. "
            "Hyperventilation-activated (standard HV 3 min provokes in >80%). Normal background."
        ),
        "clinical_tip": (
            "ESM preferred for pure absence (Level A, T-Ca2+ block → desynchronise SWD). "
            "VPA second if GTCS coexist. AVOID CBZ/OXC/PHT → absence status risk. "
            "LTG only after EEG confirms no myoclonic component (absence + myoclonic → "
            "LTG may worsen myoclonic). HV provocation test is diagnostic — perform at every visit."
        ),
    },
    {
        "type": "Myoclonic Jerks (Awakening Myoclonus)",
        "prevalence_pct": 58,
        "semiology": (
            "Brief bilateral jerks predominantly of arms/shoulders upon awakening from NREM "
            "sleep (sleep-wake transition). Peak: first 30-60 min after waking. May progress "
            "to GTCS if untreated. Triggered by sleep deprivation, alcohol, missed AED. "
            "In JME-like GABRA2 phenotype: morning myoclonus ± evening jerks. Distinguish "
            "from benign myoclonus: time of occurrence, video-EEG correlation mandatory."
        ),
        "eeg_pattern": (
            "4-6 Hz polyspike-wave (PGSW) during wakefulness, maximally on awakening. "
            "Photosensitivity (IPS) in 25-30% → photoparoxysmal response (PPR) grade 3-4."
        ),
        "clinical_tip": (
            "LTG ABSOLUTE CI if myoclonic component present (NaV1.1 block in PV+ interneurons "
            "→ disinhibition → myoclonic aggravation). VPA first-line. LEV adjunct (SV2A, "
            "reduces cortical excitability without NaV1.1 effect). CLB for breakthrough. "
            "SLEEP HYGIENE CRITICAL — sleep deprivation increases morning myoclonus 4-fold."
        ),
    },
    {
        "type": "Focal-to-Bilateral Tonic-Clonic Seizures (GTCS)",
        "prevalence_pct": 52,
        "semiology": (
            "Generalised TC ± focal onset (occasionally temporal/frontal semiology). Duration "
            "1-5 min. Predominantly awakening (sleep-wake transition) or nocturnal. Nocturnal "
            "GTCS = elevated SUDEP risk. In GEFS+ phenotype: first afebrile GTCS by age 6-10. "
            "Alcohol-withdrawal GTCS: severe, prolonged, status-prone in GABRA2 LOF."
        ),
        "eeg_pattern": (
            "Ictal: fast recruiting rhythm → PGSW → post-ictal suppression. "
            "Interictal: GSW bursts, augmented by sleep."
        ),
        "clinical_tip": (
            "ALCOHOL-WITHDRAWAL GTCS: dramatically more severe in GABRA2 LOF — lower threshold, "
            "longer duration, cluster risk. Advise complete alcohol abstinence. If admitted for "
            "withdrawal: higher BZD chlordiazepoxide/lorazepam dose needed (document in medical notes). "
            "NOCTURNAL GTCS → supervised sleep; monitor PGES duration on EEG (>50 s = SUDEP risk)."
        ),
    },
    {
        "type": "Alcohol-Withdrawal Seizures (GABRA2-Specific Phenotype)",
        "prevalence_pct": 32,
        "semiology": (
            "Generalised TC seizures (rarely focal) occurring 6-48 hours after last alcohol "
            "consumption. Risk elevated in GABRA2 LOF: ethanol normally potentiates α2-GABA-A "
            "receptors → acute withdrawal removes this potentiation → net GABAergic deficit "
            "worse than in normal population → lower withdrawal seizure threshold. Status "
            "epilepticus risk in GABRA2 LOF + alcohol misuse is significantly elevated vs "
            "either risk factor alone."
        ),
        "eeg_pattern": (
            "Beta activity during intoxication. Withdrawal: fast frontal discharges → GSW → "
            "GTCS. Post-withdrawal: widespread delta suppression, then gradual normalisation."
        ),
        "clinical_tip": (
            "GABRA2-SPECIFIC: Alcohol-withdrawal seizures are substantially more severe. "
            "AUDIT-C screening mandatory at every visit. Complete abstinence strongly advised. "
            "If patient undergoes alcohol detoxification: inform treating physician of GABRA2 "
            "diagnosis → higher BZD dose (chlordiazepoxide protocol) + monitor AED levels "
            "(alcohol alters VPA/LTG metabolism). Lorazepam IV for status in withdrawal setting."
        ),
    },
]

# ── Triggers ───────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Alcohol consumption / withdrawal", "pct": 85, "note": "GABRA2-specific: both acute intake AND withdrawal trigger seizures; withdrawal risk dramatically elevated vs general population"},
    {"trigger": "Sleep deprivation", "pct": 82, "note": "AIS α2 inhibition most critical at awakening; sleep-dep → loss of arousal-transition gating → morning myoclonus + GTCS"},
    {"trigger": "Missed AED dose", "pct": 78, "note": "VPA/LEV/ESM; compound risk with sleep deprivation if patient slept through alarm"},
    {"trigger": "Fever / intercurrent illness", "pct": 72, "note": "Threshold 38°C (standard); FSE risk in GEFS+ infants; do not rely on antipyretics alone — early rescue BZD"},
    {"trigger": "Psychological stress", "pct": 60, "note": "α2-mediated anxiolytic BZD effect reduced in LOF → stress-anxiety-seizure loop; CBT + stress management beneficial"},
    {"trigger": "Photosensitivity (IPS)", "pct": 28, "note": "PPR in 25-30%; advise gaming precautions, FL-41 spectacles if photosensitive"},
    {"trigger": "Hyperventilation", "pct": 22, "note": "Diagnostic: 3 min HV provokes absence in CAE phenotype; advise breath-counting technique to manage HV during anxiety"},
    {"trigger": "Catamenial (menstrual cycle)", "pct": 20, "note": "Perimenstrual catamenial epilepsy: oestrogen withdrawal → reduced GABA-A modulation; CLB adjunct Days 25-28 (if catamenial clustering confirmed)"},
]

# ── Treatments ─────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproate (VPA)",
        "level": "Level A",
        "moa": "Multiple: Na+/Ca2+ channel block, GABA-T inhibition (raises GABA), T-Ca2+ block (desynchronises SWD), histone deacetylase inhibition",
        "dose": "20-30 mg/kg/day in 2 doses; slow-release preferred (Epilim Chrono/Depakote ER)",
        "efficacy": "Full-spectrum: absence, myoclonic, GTCS — all seizure types in GABRA2 epilepsy",
        "safety": "Teratogenic (NTD 1-2% folic acid supplemented; FVS 10-11% without folate); weight gain; tremor; hair thinning; hepatotoxicity rare",
        "monitoring": "POLG MANDATORY before initiation; VPPP mandatory females of childbearing potential; VPA TDM 50-100 mg/L q3M; LFT/FBC/ammonia q3M",
        "gabra2_note": "First-line for JME-like, GTCS, mixed phenotype. If CAE-only + no GTCS/myoclonic: consider ESM first (avoids VPA teratogenicity in girls). POLG screen before every VPA prescription.",
    },
    {
        "drug": "Ethosuximide (ESM)",
        "level": "Level A (absence-dominant)",
        "moa": "T-type Ca2+ channel block (Cav3.1/3.2 in thalamic relay neurons) → desynchronises thalamocortical 3 Hz SWD",
        "dose": "20-40 mg/kg/day in 2 doses; start 250 mg/day, titrate weekly",
        "efficacy": "Absence seizures: 65-70% seizure-free (SANAD II); does NOT protect against GTCS or myoclonic",
        "safety": "Nausea (take with food); headache; rare: aplastic anaemia, SLE-like (CBC q6M)",
        "monitoring": "ESM TDM 40-100 mg/L; CBC q6M; if GTCS coexist: add VPA (ESM alone insufficient for GTCS)",
        "gabra2_note": "Preferred for CAE phenotype without GTCS/myoclonic (avoids VPA in young girls). Combine with VPA if GTCS present. Discontinue if GTCS develop on ESM monotherapy.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulation → reduces neurotransmitter release synchrony; reduces GABRA2-driven alpha-oscillation disruption",
        "dose": "20-60 mg/kg/day in 2 doses; IV available (1:1 PO:IV)",
        "efficacy": "JME myoclonic: 50-60% responder; GTCS: 40-50% responder; add-on to VPA/ESM",
        "safety": "Irritability (15-20%, especially with DEE); somnolence; headache",
        "monitoring": "LEV TDM 12-46 mg/L (optional); renal dosing (GFR-adjusted); AUDIT-C (irritability worse with alcohol)",
        "gabra2_note": "Useful adjunct for JME-like myoclonic + GTCS (SV2A mechanism independent of GABA-A receptor subtype). Safe in females (no teratogenicity data concerns like VPA). Avoid if severe irritability (common in GGE patients).",
    },
    {
        "drug": "Lamotrigine (LTG)",
        "level": "Level B — CAUTION: myoclonic worsening 15-20%",
        "moa": "Na+ channel block (NaV1.1, NaV1.2, NaV1.6) + reduced glutamate release; some Ca2+ channel effect",
        "dose": "0.1-0.2 mg/kg/day start; titrate over 6-8 weeks (12 weeks with VPA); target 100-200 mg/day",
        "efficacy": "CAE absences without myoclonus: 60-65% responder; GTCS: 40-50%; NOT for myoclonic GGE",
        "safety": "SJS/TEN (slow titration critical; halve dose with VPA); rash 10%; insomnia; dizziness",
        "monitoring": "LTG TDM 2.5-15 mg/L; HV-EEG + IPS before prescribing to confirm no myoclonic component; slow titration protocol",
        "gabra2_note": "REQUIRES EEG BEFORE PRESCRIBING: only if no myoclonic spike-wave on EEG. 15-20% of patients develop myoclonic aggravation (NaV1.1 block in interneurons → disinhibition). If patient prescribed LTG and develops new myoclonus: STOP IMMEDIATELY. Preferred in females of childbearing potential (CAE phenotype only).",
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B",
        "moa": "1,5-benzodiazepine; preferential α2/α3 GABA-A subunit binding (retains anxiolytic activity); GABA-A potentiation",
        "dose": "0.1-0.3 mg/kg/day in 2 doses; catamenial: premenstrual days 25-28",
        "efficacy": "Adjunct for breakthrough seizures; catamenial protocol 50% reduction in perimenstrual clusters",
        "safety": "Sedation; tolerance with continuous use; withdrawal risk; N-CLB active metabolite (CYP2C19 metabolism)",
        "monitoring": "CYP2C19 genotype (PM: higher N-CLB exposure); avoid continuous >3-4 months (tolerance); catamenial intermittent use preferred",
        "gabra2_note": "NOTE: α2-CONTAINING RECEPTORS mediate BZD anxiolytic effects. In GABRA2 LOF, CLB retains some activity via α2 (partial LOF, not complete absence) but effect may be reduced vs GABRA1-intact patients. Useful for catamenial clustering and stress-triggered seizures. Standard BZD rescue dose (midazolam 0.2 mg/kg) remains effective for GTCS termination (α1-mediated).",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B (DRE / drug-resistant)",
        "moa": "β-hydroxybutyrate → KATP channel activation → hyperpolarisation; acetone inhibits NMDA; adenosine A1R activation; independent of GABA-A receptor subtype",
        "dose": "4:1 ratio (fat:protein+carb); initiated by metabolic dietitian; MAD or LGIT alternatives",
        "efficacy": "DRE in GGE: 40-50% ≥50% seizure reduction; less effective than in focal/DEE epilepsy but clinically useful in refractory GGE",
        "safety": "Dyslipidaemia; constipation; growth impairment (children); kidney stones; β-OHB 2-4 mmol/L target",
        "monitoring": "Lipids/glucose/bicarbonate/uric acid q3M; DXA annual; dietitian q4W",
        "gabra2_note": "GABA-A independent mechanism — useful specifically because it does not rely on α2 receptor function. Indicated after 2 AED failures in GGE spectrum. Patient selection: motivated patient + family; GI tolerance; no MCAD/LCHAD.",
    },
    {
        "drug": "Perampanel (PER)",
        "level": "Level C",
        "moa": "AMPA receptor (GluA1/2) non-competitive antagonist → reduces cortical/limbic excitatory drive",
        "dose": "2-12 mg/day once at bedtime (titrate 2 mg q2 weeks); lower dose with VPA (PK interaction)",
        "efficacy": "Adjunct in GGE: GTCS 30-35% responder; some evidence for myoclonic-atonic",
        "safety": "Dizziness; somnolence; aggression/irritability (dose-dependent); reduce with alcohol (additive CNS depression)",
        "monitoring": "Aggression monitoring (PHQ-A); CYP3A4 interactions; START with 2 mg bedtime",
        "gabra2_note": "CAUTION in GABRA2 patients who misuse alcohol — PER CNS depression additive to alcohol. Useful in treatment-resistant JME-like phenotype when VPA/LEV insufficient. Glutamatergic mechanism complements GABAergic AEDs.",
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine / Oxcarbazepine / Phenytoin",
        "risk": "ABSOLUTE CI",
        "reason": (
            "GGE aggravation: NaV-preferring blockers (CBZ/OXC/PHT) block NaV1.1 "
            "in PV+ fast-spiking interneurons → reduced inhibitory tone → paradoxical "
            "worsening of absence, myoclonic, and GTCS. Can precipitate absence status "
            "epilepticus or myoclonic SE. Mechanism: PV+ interneurons depend on NaV1.1 "
            "for high-frequency firing; CBZ/OXC/PHT block this → interneuron silencing → "
            "cortical disinhibition → GGE aggravation. Documented in GGE-aggravation "
            "literature (Panayiotopoulos 2005)."
        ),
    },
    {
        "drug": "Tiagabine",
        "risk": "ABSOLUTE CI",
        "reason": (
            "Tiagabine (GAT-1 inhibitor) → prolongs GABA at synapse → in patients with "
            "generalised/diffuse epileptic networks: paradoxical NCSE. ABSOLUTE CI in ALL "
            "generalised/GABRA2 epilepsy. No exception."
        ),
    },
    {
        "drug": "Lamotrigine (in myoclonic phenotype)",
        "risk": "HIGH — myoclonic worsening 15-20%",
        "reason": (
            "NaV1.1 block in PV+ interneurons → reduced inhibitory firing → myoclonic "
            "aggravation in JME/myoclonic GGE. REQUIRES EEG before prescribing to confirm "
            "no myoclonic spike-wave. Safe in pure absence without myoclonic component only."
        ),
    },
    {
        "drug": "Valproate (females without VPPP)",
        "risk": "HIGH — MHRA 2021 / FDA REMS",
        "reason": (
            "VPA teratogenic (NTD 1-2%; fetal valproate syndrome 10-11%). MHRA 2021: VPPP "
            "mandatory for all females of childbearing potential on VPA. Annual specialist "
            "assessment + 2 contraception methods confirmed in writing."
        ),
    },
    {
        "drug": "Valproate (without POLG screen)",
        "risk": "HIGH — Alpers-Huttenlocher",
        "reason": (
            "POLG mutations + VPA → Alpers-Huttenlocher syndrome (progressive hepatic failure, "
            "fatal). POLG screen mandatory before VPA initiation in ALL patients regardless of "
            "family history."
        ),
    },
    {
        "drug": "Alcohol",
        "risk": "HIGH — GABRA2-specific seizure trigger",
        "reason": (
            "Alcohol is both an acute trigger (GABAergic euphoriant rebound excitability) and "
            "a severe withdrawal trigger. GABRA2 LOF patients have markedly lower alcohol- "
            "withdrawal seizure threshold. GABRA2-specific alcohol counselling MANDATORY. "
            "Complete abstinence strongly recommended."
        ),
    },
]

# ── Monitoring ─────────────────────────────────────────────────────────────────
MONITORING_ITEMS = [
    {"item": "POLG screen before VPA initiation", "frequency": "Once (mandatory)"},
    {"item": "VPPP enrolment (females on VPA)", "frequency": "Annual reassessment"},
    {"item": "VPA TDM (50-100 mg/L)", "frequency": "q3M"},
    {"item": "LFT / FBC / ammonia (VPA monitoring)", "frequency": "q3M"},
    {"item": "AUDIT-C alcohol screening", "frequency": "Every visit"},
    {"item": "EEG with HV and IPS (including before LTG)", "frequency": "Annual + pre-LTG"},
    {"item": "Seizure diary (alcohol + sleep + catamenial triggers)", "frequency": "Continuous (digital app)"},
    {"item": "Neuropsychological assessment", "frequency": "q2y"},
    {"item": "SUDEP risk counselling", "frequency": "Annual"},
    {"item": "Genetic cascade screening (first-degree relatives)", "frequency": "Once + new members"},
    {"item": "HLA-B*15:02 (if CBZ/OXC considered — ABSOLUTE CI but test if alternative)", "frequency": "Once"},
    {"item": "HLA-A*31:01 (if CBZ considered — ABSOLUTE CI but test if alternative)", "frequency": "Once"},
    {"item": "KD monitoring: lipids / DXA / ketones (if on diet)", "frequency": "q3M DXA annual"},
    {"item": "Alcohol withdrawal plan documented in medical notes", "frequency": "Update annually"},
]

# ── Lifecycle Windows ──────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {
        "window": "Infancy / Early Childhood (0-5 y)",
        "headline": "Febrile Seizures / GEFS+ Onset",
        "detail": (
            "FS onset typically 6 months - 5 years in GEFS+ phenotype. Severe de novo → "
            "infantile-onset DEE-like. Key actions: GEFS+ pedigree documentation; avoid "
            "CBZ/OXC/PHT (common in FS management — ABSOLUTE CI in GABRA2 GGE); VPA or "
            "clobazam for FS clusters; neurology referral if FS+ pattern (afebrile seizures begin)."
        ),
    },
    {
        "window": "Childhood (5-12 y)",
        "headline": "CAE / Absence Epilepsy Emergence",
        "detail": (
            "Familial CAE phenotype: typical absences 5-10y onset. ESM first-line for pure "
            "absence (avoids VPA teratogenicity in girls). EEG mandatory. AVOID CBZ/OXC — "
            "absence status risk. School accommodation: absence record, teacher education."
        ),
    },
    {
        "window": "Adolescence (12-20 y)",
        "headline": "JME-like Emergence / Alcohol Risk",
        "detail": (
            "JME-like phenotype emerges on awakening (sleep deprivation, social disruption). "
            "Alcohol misuse commences — GABRA2-SPECIFIC COUNSELLING MANDATORY at first adolescent "
            "visit. VPA in males; LTG (after EEG) or LEV in females. VPPP enrolment if VPA. "
            "Driving: seizure-free interval per jurisdiction (typically 12 months) before driving."
        ),
    },
    {
        "window": "Young Adult (20-35 y)",
        "headline": "GTCS / Alcohol-Withdrawal Risk / Pregnancy Planning",
        "detail": (
            "GTCS risk peaks. Alcohol-withdrawal seizures most common in this window. "
            "VPPP mandatory review if VPA. Pregnancy planning: switch from VPA if possible "
            "(LTG-only if CAE/no-myoclonic confirmed on EEG; LEV ± LTG for JME; folic acid "
            "5 mg/day pre-conception). AUDIT-C every visit. Seizure cluster risk with alcohol "
            "withdrawal — document in emergency and addiction services notes."
        ),
    },
    {
        "window": "Adult (35-55 y)",
        "headline": "Long-term Management / VPPP Review",
        "detail": (
            "Many GGE patients achieve long-term remission on VPA. Reassess VPA withdrawal "
            "after 5+ seizure-free years in benign phenotypes. Continue VPPP until menopause. "
            "Monitor VPA weight gain, tremor, bone density. Consider LTG or LEV transition "
            "if VPA side effects problematic. Alcohol abstinence remains key."
        ),
    },
    {
        "window": "Later Life (55+ y)",
        "headline": "Remission Assessment / Polypharmacy",
        "detail": (
            "CAE and JME-like GABRA2 epilepsy may remit by 50-60y in some. Drug withdrawal "
            "trial only if 10+ seizure-free years, benign phenotype, normal EEG. Polypharmacy "
            "review. VPA: tremor and weight gain often improve with low-dose continuation "
            "or switch to LEV. Genetic counselling for adult children (AD ~80% penetrance)."
        ),
    },
]

# ── Concepts ───────────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "GABRA2 (4p12)", "definition": "Gene encoding GABA-A receptor α2 subunit; enriched at AIS, cortical layer 2/3, limbic system; AD inheritance ~80% penetrance"},
    {"term": "GABA-A receptor α2 subunit", "definition": "Ionotropic Cl⁻ channel subunit at axon initial segment and limbic synapses; mediates BZD anxiolytic effect (α2/γ2 interface); developmental predominance in immature brain"},
    {"term": "Axon Initial Segment (AIS) perisomatic inhibition", "definition": "α2-containing GABA-A receptors at AIS of pyramidal neurons gate spike initiation; GABRA2 LOF → reduced AIS shunting → lower seizure threshold"},
    {"term": "BZD anxiolytic mechanism (α2/α3)", "definition": "Classical BZDs bind α/γ2 interface; α2-containing receptors mediate anxiolytic + myorelaxant BZD effects (vs α1 → sedation/hypnosis); GABRA2 LOF → reduced anxiolytic BZD efficacy"},
    {"term": "GGE — ILAE 2022", "definition": "Generalized Genetic Epilepsy; structural/metabolic cause excluded; includes JME, CAE, GTCS alone, JAE; EEG: bilateral synchronous spike-wave"},
    {"term": "GEFS+ spectrum", "definition": "Genetic Epilepsy with Febrile Seizures Plus; FS persist beyond 6y or afebrile GTCS develop; AD; caused by GABRA2, GABRG2, SCN1A, SCN1B variants"},
    {"term": "Alcohol-sensitive epilepsy (GABRA2-specific)", "definition": "GABRA2 LOF → reduced α2-GABA-A potentiation by ethanol → alcohol misuse predisposition AND markedly elevated alcohol-withdrawal seizure threshold reduction"},
    {"term": "JME — Juvenile Myoclonic Epilepsy", "definition": "GGE subtype: awakening myoclonus + GTCS ± absence; 5-10% of all epilepsy; adolescent onset; lifelong AED often required; GABRA2 familial variant"},
    {"term": "CAE — Childhood Absence Epilepsy", "definition": "GGE subtype: typical absences 4-10y; 3 Hz GSW on EEG; 70% remit by adolescence; 30% evolve to JME or GGE; ESM or VPA first-line"},
    {"term": "GGE-aggravation syndrome (NaV-blockers)", "definition": "CBZ/OXC/PHT/VGB block NaV1.1 in PV+ interneurons → paradoxical GGE worsening (absence status, myoclonic aggravation, GTCS increase); ABSOLUTE CI in GGE"},
    {"term": "VPPP — Valproate Pregnancy Prevention Programme", "definition": "MHRA 2021: VPA only in females of childbearing potential if VPPP enrolled; annual specialist assessment + 2 contraception methods documented; pharmacist check at dispensing"},
    {"term": "POLG — Alpers-Huttenlocher syndrome", "definition": "POLG mutations + VPA → mitochondrial polymerase dysfunction → hepatic failure (fatal); POLG screen mandatory before VPA in all patients"},
    {"term": "AUDIT-C alcohol screening", "definition": "3-question alcohol use disorders identification tool (consumption version); mandatory at every GABRA2 epilepsy visit given specific alcohol-seizure link"},
    {"term": "SANAD-II — Ethosuximide vs VPA in Absence", "definition": "SANAD-II (2021 NEJM): ESM non-inferior to VPA for absence seizure control with better tolerability; ESM preferred for absence-only phenotype in girls"},
    {"term": "SUDEP — Sudden Unexpected Death in Epilepsy", "definition": "Risk elevated with nocturnal GTCS, drug-resistant epilepsy, NaV1.1 dysfunction; GABRA2 GGE: moderate risk — SUDEP counselling annual; nocturnal supervision for uncontrolled GTCS"},
]

# ── Standards ──────────────────────────────────────────────────────────────────
STANDARDS = [
    "ILAE Classification of Seizures and Epilepsies 2022",
    "NICE NG217 — Epilepsies: diagnosis and management (2022)",
    "Maljevic S & Lerche H (2013) Epilepsia 54(Suppl.): GABAergic epilepsies",
    "Delahanty RJ et al. (2011) Hum Mol Genet: GABRA2 haploinsufficiency and GGE",
    "Möller RS et al. (2017) Nat Genet: GABRA2 de novo variants and epileptic encephalopathy",
    "SANAD-II (2021 NEJM): Ethosuximide vs valproate for absence epilepsy",
    "CPIC Guideline — HLA-B*15:02 and CBZ/OXC (2023)",
    "MHRA Valproate Pregnancy Prevention Programme (2021)",
    "FDA Valproate REMS — females of reproductive age (2022)",
    "ILAE Dietary Therapies Consensus (2018): Ketogenic Diet guidelines",
    "ACMG/AMP variant classification guidelines (2015/2023)",
    "Philibert RA et al. (2009) Neuropsychopharmacology: GABRA2 haplotype and alcohol dependence",
]

# ── Thresholds ─────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"label": "Fever action threshold (GEFS+ standard)", "value": "38°C (vs 37.5°C for K289M-GABRG2)", "unit": "°C"},
    {"label": "BZD rescue dose (GTCS — α1 intact, standard)", "value": "Midazolam 0.2 mg/kg buccal/IN", "unit": "mg/kg"},
    {"label": "VPA TDM target (GGE)", "value": "50-100", "unit": "mg/L"},
    {"label": "ESM TDM target (absence)", "value": "40-100", "unit": "mg/L"},
    {"label": "LTG TDM target (GGE)", "value": "2.5-15", "unit": "mg/L"},
    {"label": "Seizure-free period before driving (standard)", "value": "12", "unit": "months seizure-free"},
    {"label": "AED trials before DRE diagnosis", "value": "2 appropriate (Level A/B) AEDs failed", "unit": "trials"},
    {"label": "QTc monitoring (if ICD discussion — not standard in GABRA2)", "value": ">500 ms = cardiologist referral", "unit": "ms"},
    {"label": "AUDIT-C threshold (alcohol intervention)", "value": "≥3 (females), ≥4 (males)", "unit": "score"},
    {"label": "VPA safe prescribing in females", "value": "VPPP enrolled + 2 contraceptions confirmed", "unit": "mandatory"},
    {"label": "POLG screen timing", "value": "Before first VPA dose (not after)", "unit": "mandatory"},
    {"label": "KD beta-hydroxybutyrate target", "value": "2-4", "unit": "mmol/L"},
]

# ── References ─────────────────────────────────────────────────────────────────
REFERENCES = [
    "Maljevic S & Lerche H (2013) Epilepsia 54(Suppl. 6):20–27 — GABAergic epilepsy review",
    "Delahanty RJ et al. (2011) Hum Mol Genet 20:25–35 — GABRA2 haploinsufficiency in GGE",
    "Möller RS et al. (2017) Nat Genet 49(2):168–175 — GABRA2 de novo and epileptic encephalopathy",
    "SANAD-II Collaborators (2021) NEJM 385:2117–2131 — ESM vs VPA for absence epilepsy",
    "Bhatt DK et al. (2023) — GABRA2 clinical spectrum review",
    "Philibert RA et al. (2009) Neuropsychopharmacology 34:2611–2619 — GABRA2 and alcohol dependence",
]


# ── Patient Simulation ─────────────────────────────────────────────────────────
def _make_patients():
    random.seed(42)
    patients = []
    categories = [
        ("GABRA2-de-novo-LOF-GEFS+-severe",   32),
        ("GABRA2-familial-AD-JME-like",        25),
        ("GABRA2-familial-AD-CAE",             22),
        ("GABRA2-de-novo-severe-DEE",          15),
        ("GABRA2-negative-phenocopy-GGE",       6),
    ]
    pid = 1
    for cat, pct in categories:
        n = max(1, round(40 * pct / 100))
        for _ in range(n):
            de_novo = "de-novo" in cat
            severe = "severe" in cat or "DEE" in cat
            cae = "CAE" in cat
            jme = "JME" in cat
            gefs = "GEFS+" in cat
            on_vpa = random.random() < (0.70 if (jme or gefs or severe) else 0.45)
            on_esm = random.random() < (0.55 if cae else 0.10)
            on_lev = random.random() < (0.40 if (jme or severe) else 0.20)
            on_ltg = random.random() < (0.30 if cae else 0.10)
            on_clb = random.random() < 0.20
            on_kd = random.random() < (0.25 if severe else 0.08)
            polg_tested = "Y" if random.random() < (0.88 if on_vpa else 0.30) else "N"
            vppp_enrolled = on_vpa and random.random() < 0.80
            alcohol_misuse = random.random() < (0.42 if not (cae and not jme) else 0.12)
            alcohol_withdrawal_sz = alcohol_misuse and random.random() < 0.55
            drug_resistant = random.random() < (0.50 if severe else 0.20 if gefs else 0.15)
            febrile_seizures = gefs or (random.random() < (0.70 if de_novo else 0.40))
            absence_seizures = cae or (random.random() < (0.50 if jme else 0.30))
            myoclonic_seizures = jme or severe or (random.random() < 0.30)
            sudep_high_risk = drug_resistant and random.random() < 0.35
            audit_done = random.random() < 0.75
            sex = random.choice(["M", "F"])
            age = random.randint(8, 55)
            onset_age = random.randint(2, 18)
            patients.append({
                "id": f"GA2-{pid:03d}",
                "category": cat,
                "sex": sex,
                "age": age,
                "onset_age": onset_age,
                "on_vpa": on_vpa,
                "on_esm": on_esm,
                "on_lev": on_lev,
                "on_ltg": on_ltg,
                "on_clb": on_clb,
                "on_kd": on_kd,
                "polg_tested": polg_tested,
                "vppp_enrolled": vppp_enrolled,
                "alcohol_misuse": alcohol_misuse,
                "alcohol_withdrawal_sz": alcohol_withdrawal_sz,
                "drug_resistant": drug_resistant,
                "febrile_seizures": febrile_seizures,
                "absence_seizures": absence_seizures,
                "myoclonic_seizures": myoclonic_seizures,
                "sudep_high_risk": sudep_high_risk,
                "audit_done": audit_done,
            })
            pid += 1
    # pad to 40
    while len(patients) < 40:
        patients.append(patients[-1].copy())
        patients[-1]["id"] = f"GA2-{pid:03d}"
        pid += 1
    return patients[:40]


PATIENTS = _make_patients()


# ── API functions ──────────────────────────────────────────────────────────────
def get_overview():
    n = len(PATIENTS)
    n_vpa = sum(1 for p in PATIENTS if p["on_vpa"])
    n_esm = sum(1 for p in PATIENTS if p["on_esm"])
    n_lev = sum(1 for p in PATIENTS if p["on_lev"])
    n_kd = sum(1 for p in PATIENTS if p["on_kd"])
    n_dre = sum(1 for p in PATIENTS if p["drug_resistant"])
    n_alc = sum(1 for p in PATIENTS if p["alcohol_misuse"])
    n_alc_sz = sum(1 for p in PATIENTS if p["alcohol_withdrawal_sz"])
    n_polg = sum(1 for p in PATIENTS if p["polg_tested"] == "Y")
    n_sudep = sum(1 for p in PATIENTS if p["sudep_high_risk"])
    n_myoclonic = sum(1 for p in PATIENTS if p["myoclonic_seizures"])
    n_absence = sum(1 for p in PATIENTS if p["absence_seizures"])
    avg_age = round(sum(p["age"] for p in PATIENTS) / n, 1)

    etiology_dist = []
    cat_counts = {}
    for p in PATIENTS:
        cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1
    for e in ETIOLOGY_CATALOG:
        etiology_dist.append({
            "etiology": e["etiology"],
            "n": cat_counts.get(e["category"], 0),
            "pct": e["pct"],
        })

    return {
        "title": "GABRA2 Epilepsy (GGE / GEFS+ Spectrum / Alcohol-Sensitive / GABA-A α2 Subunit / 4p12)",
        "gene": "GABRA2",
        "locus": "4p12",
        "inheritance": "Autosomal dominant (~80% penetrance); de novo in severe/DEE phenotype; familial in GGE/JME/CAE spectrum",
        "protein": "GABA-A receptor alpha-2 (α2) subunit — AIS perisomatic inhibition, limbic gating, BZD anxiolytic target (α2/γ2 interface)",
        "mechanism": (
            "GABRA2 LOF → reduced α2-containing GABA-A receptor surface density at AIS and limbic synapses "
            "→ loss of AIS perisomatic shunting → lower seizure threshold → GGE/GEFS+ spectrum. "
            "α2 is KEY BZD anxiolytic target (α2/γ2 interface) — LOF reduces anxiolytic BZD efficacy "
            "(sedative/rescue efficacy via α1 preserved). ALCOHOL LINK: GABRA2 haplotype → alcohol "
            "dependence predisposition + markedly elevated alcohol-withdrawal seizure risk."
        ),
        "key_aha": (
            "GABRA2: α2 LOF → AIS inhibition deficit → GGE spectrum. CRITICAL: Alcohol is BOTH an acute "
            "trigger AND a withdrawal trigger — GABRA2-specific AUDIT-C screening at EVERY visit. "
            "Standard BZD rescue dose (0.2 mg/kg midazolam) PRESERVED for GTCS termination (α1 intact). "
            "LTG REQUIRES EEG before prescribing — 15-20% myoclonic worsening if myoclonic component missed. "
            "CBZ/OXC/PHT ABSOLUTE CI (GGE aggravation). POLG before VPA. VPPP mandatory females on VPA."
        ),
        "kpis": {
            "n_patients": n,
            "drug_resistant_pct": round(100 * n_dre / n),
            "alcohol_misuse_pct": round(100 * n_alc / n),
            "alcohol_withdrawal_sz_pct": round(100 * n_alc_sz / n),
            "myoclonic_pct": round(100 * n_myoclonic / n),
            "absence_pct": round(100 * n_absence / n),
            "on_vpa_pct": round(100 * n_vpa / n),
            "on_esm_pct": round(100 * n_esm / n),
            "on_lev_pct": round(100 * n_lev / n),
            "on_kd_pct": round(100 * n_kd / n),
            "polg_tested_pct": round(100 * n_polg / n),
            "sudep_high_risk_n": n_sudep,
            "avg_age_years": avg_age,
        },
        "etiology_distribution": etiology_dist,
        "treatments_summary": [{"drug": t["drug"], "level": t["level"]} for t in TREATMENTS],
        "monitoring_summary": [{"item": m["item"], "frequency": m["frequency"]} for m in MONITORING_ITEMS[:8]],
        "lifecycle": [{"window": w["window"], "headline": w["headline"]} for w in LIFECYCLE_WINDOWS],
        "thresholds": THRESHOLDS[:6],
        "contraindications_summary": [c["drug"] for c in CONTRAINDICATIONS],
        "standards": STANDARDS,
        "references": REFERENCES,
    }


def get_breakdown():
    n = len(PATIENTS)
    cat_counts = {}
    for p in PATIENTS:
        cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1

    n_dre = sum(1 for p in PATIENTS if p["drug_resistant"])
    n_alc = sum(1 for p in PATIENTS if p["alcohol_misuse"])
    n_alc_sz = sum(1 for p in PATIENTS if p["alcohol_withdrawal_sz"])
    n_myoclonic = sum(1 for p in PATIENTS if p["myoclonic_seizures"])
    n_absence = sum(1 for p in PATIENTS if p["absence_seizures"])
    n_febrile = sum(1 for p in PATIENTS if p["febrile_seizures"])
    n_polg = sum(1 for p in PATIENTS if p["polg_tested"] == "Y")
    n_vppp = sum(1 for p in PATIENTS if p.get("vppp_enrolled"))
    n_audit = sum(1 for p in PATIENTS if p["audit_done"])
    n_vpa = sum(1 for p in PATIENTS if p["on_vpa"])
    n_vpa_no_polg = sum(1 for p in PATIENTS if p["on_vpa"] and p["polg_tested"] == "N")
    n_ltg = sum(1 for p in PATIENTS if p["on_ltg"])
    n_sudep = sum(1 for p in PATIENTS if p["sudep_high_risk"])

    return {
        "summary": {
            "total": n,
            "drug_resistant_pct": round(100 * n_dre / n),
            "alcohol_misuse_pct": round(100 * n_alc / n),
            "alcohol_withdrawal_sz_pct": round(100 * n_alc_sz / n),
            "myoclonic_pct": round(100 * n_myoclonic / n),
            "absence_pct": round(100 * n_absence / n),
            "febrile_sz_pct": round(100 * n_febrile / n),
            "polg_tested_pct": round(100 * n_polg / n),
            "vppp_enrolled_pct": round(100 * n_vppp / max(n_vpa, 1)),
            "audit_done_pct": round(100 * n_audit / n),
            "vpa_without_polg_n": n_vpa_no_polg,
            "ltg_prescribed_n": n_ltg,
            "sudep_high_risk_n": n_sudep,
        },
        "etiology_distribution": [
            {
                "category": e["category"],
                "n": cat_counts.get(e["category"], 0),
                "pct": e["pct"],
                "etiology": e["etiology"],
                "mechanism": e["mechanism"],
                "typical_variants": e["typical_variants"],
                "eeg_signature": e["eeg_signature"],
                "phenotype": e["phenotype"],
            }
            for e in ETIOLOGY_CATALOG
        ],
        "patient_sample": [
            {
                "id": p["id"],
                "category": p["category"],
                "sex": p["sex"],
                "age": p["age"],
                "onset_age": p["onset_age"],
                "on_vpa": p["on_vpa"],
                "on_esm": p["on_esm"],
                "on_lev": p["on_lev"],
                "on_ltg": p["on_ltg"],
                "on_clb": p["on_clb"],
                "on_kd": p["on_kd"],
                "polg_tested": p["polg_tested"],
                "alcohol_misuse": p["alcohol_misuse"],
                "alcohol_withdrawal_sz": p["alcohol_withdrawal_sz"],
                "drug_resistant": p["drug_resistant"],
                "myoclonic_seizures": p["myoclonic_seizures"],
                "absence_seizures": p["absence_seizures"],
                "sudep_high_risk": p["sudep_high_risk"],
                "audit_done": p["audit_done"],
            }
            for p in PATIENTS[:15]
        ],
        "seizure_detail": [
            {
                "type": s["type"],
                "prevalence_pct": s["prevalence_pct"],
                "semiology": s["semiology"],
                "eeg_pattern": s["eeg_pattern"],
                "clinical_tip": s["clinical_tip"],
            }
            for s in SEIZURE_TYPES
        ],
        "trigger_detail": TRIGGERS,
        "treatment_detail": [
            {
                "drug": t["drug"],
                "level": t["level"],
                "moa": t["moa"],
                "dose": t["dose"],
                "efficacy": t["efficacy"],
                "safety": t["safety"],
                "monitoring": t["monitoring"],
                "gabra2_note": t["gabra2_note"],
            }
            for t in TREATMENTS
        ],
        "contraindications": [
            {"drug": c["drug"], "risk": c["risk"], "reason": c["reason"]}
            for c in CONTRAINDICATIONS
        ],
        "monitoring_items": MONITORING_ITEMS,
        "lifecycle": LIFECYCLE_WINDOWS,
    }


def get_definitions():
    return {
        "concepts": CONCEPTS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "contraindications": [
            {"drug": c["drug"], "risk": c["risk"]}
            for c in CONTRAINDICATIONS
        ],
        "references": REFERENCES,
    }
