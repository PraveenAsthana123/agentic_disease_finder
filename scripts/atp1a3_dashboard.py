"""
ATP1A3 Epilepsy — AHC / CAPOS / RDP / DEE-ATP1A3 / Na+/K+-ATPase α3 / 19q13.2
================================================================================
40-patient cohort · ATP1A3 (19q13.2) · Na+/K+-ATPase alpha-3 subunit · AD de novo

ATP1A3 BIOLOGY:
ATP1A3 (19q13.2) encodes the alpha-3 (α3) catalytic subunit of Na+/K+-ATPase — the
electrogenic ion pump that maintains neuronal resting membrane potential by extruding
3 Na+ out and importing 2 K+ in per ATP cycle.

KEY POINTS:
  1. NEURON-SPECIFIC: ATP1A3 / α3-isoform is expressed selectively in neurons (NOT in glia,
     heart, or kidney — unlike α1/ATP1A1 which is ubiquitous, or α2/ATP1A2 which is in
     astrocytes + heart). Especially high in: cerebellar Purkinje cells, GABAergic
     interneurons, dopaminergic neurons of substantia nigra, basal ganglia, motor cortex.

  2. PUMP MECHANISM: Intracellular Na+ (from APs) + ATP → conformational cycle (E1→E2P→E2→E1) →
     3 Na+ expelled, 2 K+ imported per cycle. Net negative charge export = ELECTROGENIC = provides
     ~5-10 mV hyperpolarizing component to resting Vm. ATP1A3 LOF → reduced pump rate →
     Na+ accumulates inside neuron → failure to restore resting Vm after bursts → depolarized
     neuron → sustained hyperexcitability → seizures + dystonic episodes.

  3. SYNDROME SPECTRUM (mutation–phenotype correlation):
     a. AHC (Alternating Hemiplegia of Childhood) — OMIM #614820
        Mutations: D801N (most common, 60% of AHC — TRANSMEMBRANE DOMAIN 5, moderate-severe),
        E815K (20% of AHC — most severe, early-onset DEE, refractory epilepsy, spastic quadriplegia),
        G947R (10%, moderate AHC), T613M (~5%), others.
        Core: episodic hemiplegia alternating sides, bilateral hemiplegia (plegic crisis),
        oculomotor abnormalities (tonic horizontal deviation, nystagmus), dystonic attacks,
        autonomic features (sweating, skin color change, respiratory irregularity).
        ALL ATTACKS RESOLVE WITH SLEEP — pathognomonic. Epilepsy in ~50-60%.

     b. CAPOS Syndrome — OMIM #601543
        Mutation: E818K (dominant; distinct from E815K by 3 residues).
        C = Cerebellar Ataxia · A = Areflexia · P = Pes Cavus · O = Optic Atrophy · S = Sensorineural Hearing Loss.
        Episodes of acute ataxia triggered by febrile illness. Progressive neurodegeneration.
        Epilepsy rare in CAPOS.

     c. RDP (Rapid-onset Dystonia-Parkinsonism) — OMIM #128235
        Mutations: D923N, P905L, others (transmembrane/cytoplasmic).
        Adult onset, sudden-onset (hours-days) dystonia + parkinsonism after trigger (fever, stress).
        Rostrocaudal gradient: face/bulbar > arm > leg. No epilepsy typically.

     d. DEE-ATP1A3 (Severe early-onset epileptic encephalopathy — E815K spectrum)
        Mostly E815K: severe neonatal/infantile onset, refractory epileptic spasms/tonic seizures,
        profound intellectual disability, no AHC-like hemiplegic attacks.

  4. PATHOMECHANISM: ATP1A3 LOF haploinsufficiency → 50% reduction in neuron Na+/K+ pump
     capacity → deficient Na+ extrusion → [Na+]i rises during repeated firing → Vm failure →
     energy crisis (pump runs on ATP; depleted ATP = failure to recover) → episodic neuronal
     inexcitability/dysfunction (hemiplegic attacks) OR chronic hyperexcitability (seizures).
     D801N: moderate pump impairment, relatively preserved baseline; crisis under metabolic demand.
     E815K: severe pump impairment, constitutive dysfunction → DEE phenotype.
     The ELECTROGENIC nature of the pump means LOF = loss of intrinsic hyperpolarization →
     neurons sustain depolarized state → lower threshold for epileptic and dystonic bursts.

  5. WHY ATTACKS RESOLVE WITH SLEEP: During NREM sleep, reduced neuronal firing rate →
     lower Na+ load → even reduced-capacity pump sufficient to restore Vm → attack resolves.
     This is unique to Na+/K+-ATPase LOF; no other epilepsy gene has this hallmark.

CLINICAL DIAGNOSIS:
  AHC: Episodes of hemiplegia (alternating sides) + dystonia + nystagmus + autonomic features
  + resolution with sleep + onset <18M. Genetic confirmation: ATP1A3 pathogenic variant.
  CAPOS: Febrile-triggered ataxia + hearing loss + optic atrophy + areflexia + pes cavus.
  RDP: Sudden-onset dystonia-parkinsonism in adult + trigger + rostrocaudal gradient.

INHERITANCE AND GENETICS:
  AUTOSOMAL DOMINANT — almost exclusively DE NOVO (>95% AHC cases). Germline mosaicism rare.
  Locus: 19q13.2. Gene: ATP1A3 (23 exons; 1013 aa protein).
  pLI ~0.99 (extremely intolerant to LOF). Key variants: D801N / E815K / E818K / G947R / T613M.
  Protein: 10 transmembrane helices (TM1-10); α/β heterodimer; β-subunit (ATP1B1/B2) required
  for folding, targeting, activity. Pathogenic variants cluster in TM4/5 (pump mechanism),
  phosphorylation domain (P-domain, D369 = phosphorylation site), A-domain (actuator).

KEY REFERENCES:
  Heinzen et al. 2012 Nature Genetics 44:1030 — ATP1A3 as AHC gene; D801N, E815K, G947R.
  Demos et al. 2014 Annals of Neurology 75:196 — AHC genotype-phenotype.
  Rosewich et al. 2012 Neurology Genetics — E818K CAPOS.
  Panagiotakaki et al. 2015 Brain 138:2520 — AHC-Europe 187-patient consortium; outcomes.
  Bhatt et al. 2023 Epilepsia — gene-epilepsy review standard reference.
  Bhattacharya 2018 Pediatrics — AHC clinical review.
"""
import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "ATP1A3-D801N-AHC-moderate",
        "pct": 40,
        "etiology": "ATP1A3 D801N — AHC moderate-severe (most common; familial de novo)",
        "mechanism": (
            "D801N — transmembrane domain 5 missense — reduces Na+/K+-ATPase α3 pump "
            "rate by ~40-50%. Moderate-severity AHC: alternating hemiplegic episodes, "
            "oculomotor abnormalities, dystonia; attacks resolve with sleep. ~50% epilepsy."
        ),
        "typical_variants": "c.2401G>A (p.D801N) — ~60% of all AHC cases worldwide",
        "eeg_signature": (
            "Ictal: polymorphic delta ipsilateral to hemiplegic limb during attack; "
            "post-ictal depression. Interictal: multifocal spikes or normal; 3-Hz SWD rare."
        ),
        "phenotype": "AHC onset 1-6M; hemiplegic + dystonic attacks; oculomotor; epilepsy 50%; moderate ID",
    },
    {
        "category": "ATP1A3-E815K-AHC-severe-DEE",
        "pct": 18,
        "etiology": "ATP1A3 E815K — Severe AHC + DEE (earliest onset; most severe phenotype)",
        "mechanism": (
            "E815K — transmembrane domain 5 missense (adjacent to D801N) — severely reduces "
            "Na+/K+-ATPase α3 activity (~70-80% loss). Earliest onset, most severe: "
            "neonatal-infantile, refractory epileptic spasms + tonic seizures, spastic quadriplegia."
        ),
        "typical_variants": "c.2443G>A (p.E815K) — ~20% of AHC; most severe; DEE overlap",
        "eeg_signature": (
            "Hypsarrhythmia (infantile spasms / West syndrome pattern); "
            "high-voltage multifocal spikes; burst-suppression in severe neonatal cases."
        ),
        "phenotype": "Neonatal/early infantile; refractory epilepsy; spastic quadriplegia; profound ID; AHC features may be masked",
    },
    {
        "category": "ATP1A3-E818K-CAPOS",
        "pct": 15,
        "etiology": "ATP1A3 E818K — CAPOS Syndrome (Cerebellar Ataxia/Areflexia/Pes Cavus/Optic Atrophy/SNH Loss)",
        "mechanism": (
            "E818K — distinct from E815K by 3 residues; different conformational effect on pump. "
            "Causes progressive neurodegenerative CAPOS syndrome with febrile-triggered crises. "
            "Epilepsy is rare in pure CAPOS; ataxia + hearing loss + optic atrophy predominate."
        ),
        "typical_variants": "c.2452G>A (p.E818K) — virtually all CAPOS cases",
        "eeg_signature": (
            "Often normal or mild diffuse slowing. Ataxic episodes during fever. "
            "Epilepsy rare; occasional focal slowing posterior regions (atrophic change)."
        ),
        "phenotype": "Cerebellar ataxia + areflexia + pes cavus + optic atrophy + SNHL; febrile crisis; progressive",
    },
    {
        "category": "ATP1A3-G947R-other-AHC",
        "pct": 15,
        "etiology": "ATP1A3 G947R / Other AHC variants — mild-moderate AHC spectrum",
        "mechanism": (
            "G947R — cytoplasmic domain; moderate pump impairment (~30% reduction). "
            "Milder AHC phenotype: later onset, less frequent attacks, better cognition. "
            "T613M and other variants have similar mild-moderate AHC profile."
        ),
        "typical_variants": "G947R (~10%), T613M (~5%), R756H, A681T, various missense in P/N/A domains",
        "eeg_signature": (
            "Multifocal epileptiform discharges or normal. Focal slowing during hemiplegic phase. "
            "EEG often normal interictally in mild AHC."
        ),
        "phenotype": "Later AHC onset (6-18M), milder attacks, better cognition, epilepsy 35-45%",
    },
    {
        "category": "ATP1A3-phenocopy-negative",
        "pct": 12,
        "etiology": "ATP1A3-negative phenocopy (AHC / CAPOS / dystonia mimic — other etiology)",
        "mechanism": (
            "Clinical AHC-like or CAPOS-like presentation without ATP1A3 mutation. "
            "Alternative diagnoses: CACNA1A (hemiplegic migraine variant), SLC1A3 (episodic ataxia), "
            "ADGRV1, PRRT2 (paroxysmal dyskinesia), SLC2A1 (GLUT1 — test for CSF glucose), "
            "ACTB/ACTG1 (Baraitser-Winter), mitochondrial disorders."
        ),
        "typical_variants": "No ATP1A3 pathogenic variant; broad gene panel or exome required",
        "eeg_signature": "Variable; may show migraine-related EEG changes or focal slowing depending on alternative diagnosis",
        "phenotype": "Mimics AHC/CAPOS; triggers may differ; CSF glucose, lactate, metabolic screen needed",
    },
]

_category_weights = [
    ("ATP1A3-D801N-AHC-moderate", 16),
    ("ATP1A3-E815K-AHC-severe-DEE", 7),
    ("ATP1A3-E818K-CAPOS", 6),
    ("ATP1A3-G947R-other-AHC", 6),
    ("ATP1A3-phenocopy-negative", 5),
]
_cats = []
for cat, w in _category_weights:
    _cats.extend([cat] * w)


# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES  (5 types)
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Focal Motor (during hemiplegic/dystonic attack)",
        "prevalence_pct": 75,
        "semiology": (
            "Focal clonic or tonic jerking of the hemiplegic limb(s) during an AHC episode; "
            "often misclassified as 'hemiplegic seizure' but may be peri-ictal rather than ictal. "
            "Dystonic posturing of arm/leg ipsilateral to hemiplegic side. Duration: seconds-minutes. "
            "Resolution with sleep — key distinguishing feature from seizures of other causes."
        ),
        "eeg_pattern": (
            "Ictal: polymorphic delta or mixed frequency activity ipsilateral to hemiplegic limb. "
            "May show focal sharp waves or no clear ictal correlate (subcortical origin). "
            "Post-ictal: regional slowing or suppression."
        ),
        "clinical_tip": (
            "CRITICAL: Not all movements during an AHC attack are epileptic. EEG during attack "
            "is essential to distinguish ictal vs peri-ictal motor phenomena. "
            "Flunarizine reduces attack frequency; attacks themselves ARE NOT seizures per se."
        ),
    },
    {
        "type": "Epileptic Spasms (infantile spasms / West syndrome — E815K)",
        "prevalence_pct": 45,
        "semiology": (
            "Sudden axial flexion (flexor spasms) or extension, usually in clusters, "
            "on waking or drowsiness. Infantile onset (3-12M). Associated with hypsarrhythmia. "
            "E815K phenotype predominantly; may predate or coexist with AHC features. "
            "Often refractory — ACTH/vigabatrin partial response only."
        ),
        "eeg_pattern": (
            "Hypsarrhythmia (high-voltage, chaotic, multifocal spike-wave + slow waves). "
            "Ictal: electrodecrement (voltage attenuation) during spasm; high-amplitude slow wave follow."
        ),
        "clinical_tip": (
            "Treat as infantile spasms: ACTH or oral corticosteroids first (UKISS/UKWEST protocol). "
            "VGB second-line (note: CI in some ATP1A3 motor phenotypes — monitor carefully). "
            "E815K + infantile spasms = poor outcome; consider KD early."
        ),
    },
    {
        "type": "Generalized Tonic-Clonic (GTCS)",
        "prevalence_pct": 40,
        "semiology": (
            "Bilateral tonic stiffening followed by symmetric clonic jerking. "
            "Triggered by fever, sleep deprivation, or after hemiplegic attacks. "
            "Duration typically 1-3 min; post-ictal lethargy 30-60 min. "
            "In AHC, GTCS often represent spread from a focal onset."
        ),
        "eeg_pattern": (
            "Generalized 10-Hz recruiting rhythm during tonic phase; "
            "generalized polyspike-slow wave during clonic phase; post-ictal suppression."
        ),
        "clinical_tip": (
            "AVOID CBZ/OXC — both dramatically worsen AHC hemiplegic attacks (ABSOLUTE CI). "
            "VPA with POLG screen. LEV well tolerated. CLB for frequent GTCS. "
            "GTCS >3/year = SUDEP risk counselling."
        ),
    },
    {
        "type": "Status Epilepticus (febrile / non-febrile)",
        "prevalence_pct": 30,
        "semiology": (
            "Prolonged focal or generalized seizures >5 min, or repeated seizures without "
            "full recovery. Common precipitant: fever ≥37.5°C (major AHC/ATP1A3 trigger). "
            "Also: bathing-triggered bilateral hemiplegic crisis + SE. "
            "Hospitalization required; fever control critical."
        ),
        "eeg_pattern": (
            "Continuous ictal patterns, focal or generalized; evolving frequency. "
            "NCSE may occur without obvious clinical features — EEG monitoring essential."
        ),
        "clinical_tip": (
            "AHC ACTION PLAN: Midazolam buccal/IM first-line. Levetiracetam IV second-line. "
            "Avoid phenytoin IV (cardiac risk, no clear benefit). Antipyretics early. "
            "Rescue plan (written) at home. TIAGABINE ABSOLUTE CI (NCSE risk)."
        ),
    },
    {
        "type": "Absence / Absence-like (staring spells misclassified as hemiplegic aura)",
        "prevalence_pct": 20,
        "semiology": (
            "Brief (5-20 sec) staring episodes with behavioral arrest; may precede or follow "
            "hemiplegic attack (pre/post-ictal phase). True 3-Hz absence uncommon in ATP1A3. "
            "Distinguishing from hemiplegic aura or post-ictal confusion requires EEG correlation. "
            "More common in D801N-AHC with focal onset epilepsy."
        ),
        "eeg_pattern": (
            "True absences: 3-Hz generalized spike-wave. More commonly: focal slowing, "
            "non-specific voltage changes during peri-attack staring. HV may not trigger 3-Hz SWD."
        ),
        "clinical_tip": (
            "Do not treat staring in AHC as typical absence without EEG confirmation. "
            "HV may trigger AHC episodes (not typical absence) → use cautiously in provocation EEG. "
            "ESM not usually needed; VPA preferred if true absence co-exists."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS  (8 triggers)
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fever / Hyperthermia (≥37.5°C)",
        "prevalence_pct": 85,
        "mechanism": (
            "Elevated temperature → increased neuronal firing rate → accelerated Na+ influx → "
            "ATP1A3-deficient pump overwhelmed → acute Na+ loading → Vm failure → hemiplegic/dystonic crisis. "
            "Fever is the single most powerful AHC trigger. Prompt antipyretics (paracetamol/ibuprofen) "
            "at first sign of fever is mandatory in AHC care plan."
        ),
        "management": "Written fever action plan; antipyretics at first sign; school + daycare education; flu vaccine annually",
    },
    {
        "trigger": "Water Immersion (bathing / swimming — cold or warm)",
        "prevalence_pct": 72,
        "mechanism": (
            "Temperature change (cold or warm water) + sensory stimulation → sudden change in "
            "peripheral afferent input → brainstem/cerebellar pump stress → AHC attack. "
            "Pathognomonic trigger for AHC — highly specific; rarely seen in other channelopathies. "
            "Warm baths more often reported; cold water also triggers."
        ),
        "management": "Shower preferred over bath; warm (not hot/cold) water; supervise bathing; exit water at onset",
    },
    {
        "trigger": "Emotional Stress / Excitement / Frustration",
        "prevalence_pct": 68,
        "mechanism": (
            "Sympathetic arousal → noradrenaline release → increased neuronal activity → "
            "Na+ loading → pump failure in vulnerable neurons. Strong emotional events "
            "(excitement, anger, fear) particularly potent triggers in school-age children with AHC."
        ),
        "management": "Calm environment; reduce sudden emotional stimuli; school accommodation letter; behavioral support",
    },
    {
        "trigger": "Physical Exertion / Exercise",
        "prevalence_pct": 55,
        "mechanism": (
            "Exercise → increased motor cortex/spinal cord neuronal firing → rapid Na+ influx → "
            "ATP1A3 pump cannot keep pace → local Na+ accumulation → focal motor dysfunction → "
            "hemiplegic or dystonic episode during or immediately after exertion."
        ),
        "management": "Moderate-pace activity preferred; avoid sudden intense bursts; cool-down periods; rest if prodrome",
    },
    {
        "trigger": "Missed AED / Medication Gaps",
        "prevalence_pct": 48,
        "mechanism": (
            "Missed flunarizine or AED dose → loss of Na+ channel / Ca2+ channel stabilization → "
            "increased neuronal excitability → lower threshold for AHC attack or seizure. "
            "Flunarizine has 18-36h half-life so single missed dose may not immediately trigger; "
            "multiple missed doses progressively increase risk."
        ),
        "management": "Blister-pack dispensing; alarm reminders; caregiver checklist; school nurse medication plan",
    },
    {
        "trigger": "Sleep Deprivation",
        "prevalence_pct": 42,
        "mechanism": (
            "Sleep deprivation → increased cortical excitability (known across epilepsy syndromes) → "
            "lower threshold for focal motor or GTCS in ATP1A3 epilepsy. "
            "Note paradox: sleep RESOLVES attacks (Na+ accumulation clears during NREM); "
            "sleep deprivation prevents this restoration → builds vulnerability."
        ),
        "management": "Consistent sleep schedule; school accommodation for start times; sleep hygiene education",
    },
    {
        "trigger": "Intercurrent Illness (even without fever)",
        "prevalence_pct": 60,
        "mechanism": (
            "Systemic illness → cytokine release + metabolic stress + dehydration → "
            "reduced CNS energy availability (ATP) → Na+/K+-ATPase pump efficiency drops → "
            "attacks increase in frequency and duration. Gastroenteritis (dehydration) "
            "and respiratory infections are most common precipitants."
        ),
        "management": "Aggressive oral/IV hydration; AED dose review during illness; sick-day plan; early hospital contact",
    },
    {
        "trigger": "Bright/Flashing Light (photic stimulation — uncommon)",
        "prevalence_pct": 25,
        "mechanism": (
            "Photic stimulation → visual cortex neuronal bursting → spreading Na+ load → "
            "AHC episode or focal visual seizure in susceptible individuals. "
            "Less common than in GGE syndromes; occurs mainly in D801N-AHC with epilepsy."
        ),
        "management": "Photic avoidance (screen filters, FL-41 lenses); IPS in EEG with caution during AHC",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS  (7)
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Flunarizine — Level B (AHC attack prevention; first-line for AHC)",
        "level": "B",
        "moa": (
            "Flunarizine: diphenylpiperazine calcium channel blocker. "
            "Blocks L-type + T-type + N-type Ca2+ channels; also H1 antihistamine + D2-partial antagonist. "
            "Reduces neuronal excitability → decreases frequency/severity of AHC hemiplegic attacks. "
            "Does NOT directly correct Na+/K+-ATPase deficiency; reduces downstream excitability. "
            "NOT a sodium channel blocker (unlike CBZ — which is ABSOLUTE CI)."
        ),
        "dose": "5mg nocte (children <12 kg: 2.5mg). Increase to 10mg if inadequate response at 3M. Adult: 5-10mg nocte",
        "efficacy": (
            "~70-80% reduction in AHC attack frequency (Panagiotakaki-2015 consortium). "
            "Best evidence for D801N-AHC. Less clear for E815K-DEE phenotype. "
            "CAPOS (E818K): no clear benefit for ataxia crisis prevention; small case series only."
        ),
        "safety": (
            "Somnolence (dose-related), weight gain, extrapyramidal effects (tardive dyskinesia — rare, "
            "monitor). QTc prolongation (rare — ECG baseline + 12M). Paradoxical increased attacks rare. "
            "Do NOT use D2 antagonists (haloperidol) concurrently — dystonia risk."
        ),
        "monitoring": "ECG baseline; weight monthly (children); drowsiness assessment; drug interactions",
        "atp1a3_note": (
            "FIRST-LINE for AHC attack prevention. Start after diagnosis confirmed. "
            "Titrate slowly. May take 4-8 weeks for full effect. "
            "Continue indefinitely — attacks often return if discontinued. "
            "Not useful for CAPOS or RDP. POLG screen not required (not a mitochondrial toxin)."
        ),
    },
    {
        "drug": "Levetiracetam (LEV) — Level B (AHC epilepsy; first-line add-on)",
        "level": "B",
        "moa": (
            "SV2A (synaptic vesicle glycoprotein 2A) modulator → reduces neurotransmitter release "
            "presynaptically → broad-spectrum antiseizure effect. No interaction with Na+/K+-ATPase. "
            "Does not worsen hemiplegic attacks (unlike CBZ). Well tolerated in children."
        ),
        "dose": "10-20 mg/kg/day bd → up to 60 mg/kg/day. Adult: 500mg bd → 3000mg/day bd",
        "efficacy": (
            "Effective for focal + GTCS in AHC epilepsy (~60% responder rate). "
            "IV LEV useful for SE management. No aggravation of hemiplegic attacks."
        ),
        "safety": (
            "Irritability/behavioral changes (especially in children — 20-30%); somnolence; "
            "rare: psychosis, agranulocytosis. Renal clearance: dose adjust eGFR<80."
        ),
        "monitoring": "Renal function baseline; behavioral monitoring; drug level not routinely needed",
        "atp1a3_note": (
            "Preferred AED for AHC epilepsy — no AHC attack aggravation, IV formulation for SE. "
            "Combine with flunarizine (no interaction). Watch behavioral side effects in AHC "
            "(pre-existing behavioral challenges common). Pyridoxine (100mg/day) may help irritability."
        ),
    },
    {
        "drug": "Valproate (VPA) — Level B (AHC epilepsy + GTCS; POLG MANDATORY before use)",
        "level": "B",
        "moa": (
            "Multiple MOA: Na+ channel stabilization (frequency-dependent), GABA-T inhibition → "
            "raised GABA, T-type Ca2+ channel block, HDAC inhibition. Broad-spectrum AED. "
            "Does not worsen AHC attacks. Effective for GTCS + myoclonic in AHC."
        ),
        "dose": "20-40 mg/kg/day bd-tds (children). Adult: 600-2000mg/day. TDM target: 50-100 μg/mL",
        "efficacy": "60-70% GTCS responder rate in ATP1A3 epilepsy. Also reduces focal motor seizure burden",
        "safety": (
            "POLG/Alpers: fatal hepatotoxicity — POLG1 mitochondrial DNA depletion syndrome → "
            "ATP-dependent pump further impaired → catastrophic outcome. SCREEN POLG BEFORE VPA. "
            "VPPP (females 18+): teratogenicity (spina bifida, neurocognitive — MHRA 2021). "
            "Weight gain, hair thinning, tremor, thrombocytopenia, pancreatitis."
        ),
        "monitoring": "POLG1 screen BEFORE starting; LFT + FBC + ammonia q3M; TDM q3M; VPPP females enrolled",
        "atp1a3_note": (
            "POLG MANDATORY — ATP1A3 LOF already impairs energy-dependent pump; "
            "POLG-related mitochondrial failure would further devastate Na+/K+-ATPase function. "
            "Screen ALL patients before VPA regardless of clinical POLG suspicion. "
            "Consider LEV/CLB first in young children to avoid VPA early."
        ),
    },
    {
        "drug": "Clobazam (CLB) — Level B (SE prevention; catamenial-like cyclical attacks; adjunct)",
        "level": "B",
        "moa": (
            "1,5-benzodiazepine; GABA-A receptor positive allosteric modulator (γ2 synaptic). "
            "Enhances phasic inhibition. Faster onset than other BZDs for SE rescue. "
            "Less sedating than clonazepam. Intermittent use for febrile crisis prevention."
        ),
        "dose": "0.1-0.3 mg/kg/day bd. Rescue: 0.5mg/kg buccal/oral. Adult: 10-30mg/day",
        "efficacy": "Effective for SE rescue + cluster prevention. Tolerance develops with daily use (rotate 4-week cycles)",
        "safety": "Sedation, drooling (especially young children), tolerance with chronic use, paradoxical excitation",
        "monitoring": "Sedation level; tolerance assessment; drug interactions (CYP3A4)",
        "atp1a3_note": (
            "Use intermittently for febrile crisis prevention (3-5 day courses) rather than daily. "
            "Buccal CLB at SE onset as per written action plan. "
            "IV diazepam also used for acute severe attacks in hospital (not for home)."
        ),
    },
    {
        "drug": "Topiramate (TPM) — Level C (refractory AHC epilepsy; adjunct)",
        "level": "C",
        "moa": (
            "Multiple MOA: Na+ channel stabilization, AMPA/kainate antagonism, GABA-A enhancement, "
            "carbonic anhydrase inhibition. Broad-spectrum. No direct interaction with Na+/K+-ATPase."
        ),
        "dose": "1-3 mg/kg/day bd → up to 9 mg/kg/day (children). Adult: 50-400mg/day bd",
        "efficacy": "30-50% responder rate for focal motor + GTCS when LEV/VPA insufficient. Anecdotal benefit for AHC attack frequency",
        "safety": (
            "Cognitive slowing / word-finding difficulty (dose-related, common), metabolic acidosis, "
            "kidney stones (citrate depletion), oligohidrosis/hyperthermia (rare, serious — educate families). "
            "Weight loss. Teratogenic (oral clefts, SGA). Contraception mandatory."
        ),
        "monitoring": "Bicarbonate baseline; renal function; cognitive assessment; hydration; contraception",
        "atp1a3_note": (
            "Use cautiously — oligohidrosis/hyperthermia risk: AHC triggers include temperature change → "
            "TPM-induced reduced sweating could exacerbate heat-triggered attacks. Educate families: "
            "ensure adequate hydration and avoid overheating on TPM."
        ),
    },
    {
        "drug": "Ketogenic Diet (KD) — Level C (refractory DEE-E815K; refractory AHC epilepsy)",
        "level": "C",
        "moa": (
            "High-fat/low-carbohydrate ratio → hepatic ketosis → ketone bodies (β-hydroxybutyrate, "
            "acetoacetate) cross BBB → neuronal energy substrate → reduces neuronal hyperexcitability. "
            "Also: increased GABA synthesis; altered Na+ channel kinetics. "
            "Metabolic benefit: ketones bypass glycolysis → may partially compensate energy failure in ATP1A3 neurons."
        ),
        "dose": "3:1 or 4:1 lipid:non-lipid ratio; supervised by metabolic dietitian; classical KD or MCT",
        "efficacy": "~50% >50% seizure reduction in refractory DEE. Anecdotal reduction in AHC attack frequency in case series",
        "safety": (
            "Dyslipidemia, kidney stones, growth restriction (children), acidosis, "
            "constipation. Selenium/zinc supplementation required. DXA at 1-2 years."
        ),
        "monitoring": "Ketones (urine/blood); lipids q6M; renal USS; growth centiles; selenium; DXA",
        "atp1a3_note": (
            "Consider after 2 AED failures in E815K-DEE. Theoretical benefit: ketones as "
            "alternative energy substrate may support ATP production for Na+/K+-ATPase. "
            "Combine with flunarizine in AHC cases. Dietitian-led induction essential."
        ),
    },
    {
        "drug": "Triheptanoin (C7 anaplerotic) — Level C investigational (AHC energy failure)",
        "level": "C",
        "moa": (
            "C7 odd-chain fatty acid → hepatic β-oxidation → propionyl-CoA + acetyl-CoA → "
            "TCA cycle anaplerosis (replenishes oxaloacetate via propionyl-CoA carboxylase). "
            "Increases neuronal ATP production capacity → partially compensates ATP1A3 pump failure. "
            "Rational for LOF pumps that fail under energy stress."
        ),
        "dose": "1-2 g/kg/day with meals (divided 3-4 doses); mixed into food; UltraGenyx compassionate use",
        "efficacy": "Small open-label AHC series (n<20): ~50-60% reduction in attack days. No RCT completed (2026)",
        "safety": "Generally well tolerated: GI upset (nausea, diarrhea), mild acidosis. Long-term data limited",
        "monitoring": "Acylcarnitine profile; LFT; growth; urine organic acids",
        "atp1a3_note": (
            "Investigational; obtain via named-patient / compassionate use (UltraGenyx). "
            "Rational mechanistic basis for ATP1A3 energy failure. Consider in refractory AHC "
            "after flunarizine + 2 AED failures, especially E815K. "
            "NCT number for AHC anaplerosis trial: NCT02021695 (closed); ongoing IND discussions."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS  (5)
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) — ABSOLUTE CI in AHC",
        "risk": "ABSOLUTE CI",
        "reason": (
            "CBZ/OXC block voltage-gated Na+ channels (Nav1.1, Nav1.2, Nav1.6). "
            "In AHC: Na+ channel blockade disrupts the compensatory mechanisms that prevent "
            "hemiplegic attack during periods of reduced pump capacity. "
            "MULTIPLE CASE REPORTS of CBZ triggering severe, prolonged, bilateral hemiplegic crises "
            "in AHC within hours to days of initiation. Phenytoin similarly contraindicated. "
            "NO Na+ CHANNEL BLOCKERS in AHC — ever. This is AHC-specific, not a general rule."
        ),
    },
    {
        "drug": "Tiagabine (TGB) — ABSOLUTE CI (NCSE risk)",
        "risk": "ABSOLUTE CI",
        "reason": (
            "Tiagabine blocks GABA reuptake transporter GAT-1 → accumulation of extrasynaptic GABA → "
            "paradoxical non-convulsive status epilepticus (NCSE) in focal epilepsies. "
            "ATP1A3 patients with focal epilepsy are at high NCSE risk. "
            "ABSOLUTE CI — never use in any ATP1A3 patient with epilepsy."
        ),
    },
    {
        "drug": "VPA without POLG screen — HIGH RISK (Alpers-Huttenlocher syndrome)",
        "risk": "HIGH",
        "reason": (
            "POLG1 pathogenic variants → mitochondrial DNA depletion → Alpers syndrome. "
            "VPA in POLG1 carriers → acute hepatic failure (often fatal). "
            "ATP1A3 patients: already have ATP-dependent pump LOF; additional mitochondrial "
            "dysfunction (POLG) would be catastrophic. POLG screen MANDATORY before VPA. "
            "If POLG pathogenic variant confirmed: VPA ABSOLUTELY CONTRAINDICATED."
        ),
    },
    {
        "drug": "Haloperidol / Dopamine D2 Antagonists — HIGH RISK (dystonia worsening — RDP/AHC)",
        "risk": "HIGH",
        "reason": (
            "RDP and AHC involve dopaminergic system dysfunction (ATP1A3 high expression in SNc/SNr). "
            "D2 receptor blockade (haloperidol, risperidone, metoclopramide) → worsens dystonia dramatically. "
            "MULTIPLE CASE REPORTS of RDP/AHC crisis precipitated by D2 antagonists in ER settings "
            "(metoclopramide for vomiting is particularly dangerous). "
            "Alert ER and all treating teams. Ondansetron (5-HT3) preferred for nausea/vomiting."
        ),
    },
    {
        "drug": "Vigabatrin (VGB) without ERG monitoring — HIGH RISK (VFD + motor exacerbation)",
        "risk": "HIGH",
        "reason": (
            "VGB causes irreversible visual field defects (VFD) in ~30-40% — SHARE REMS monitoring required. "
            "Additionally, vigabatrin may exacerbate motor phenotype in AHC (mechanism unclear — "
            "possible cerebellar/GABAergic circuit effect). "
            "Use only if ACTH/prednisolone fail for infantile spasms (E815K-DEE); never beyond infantile spasms phase."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING  (14 items)
# ─────────────────────────────────────────────────────────────────────────────
MONITORING_ITEMS = [
    {"item": "POLG1 screen (mandatory before VPA)", "frequency": "Once before VPA initiation"},
    {"item": "Attack + seizure diary (type/duration/trigger/resolution)", "frequency": "Every visit (minimum q3M)"},
    {"item": "Developmental milestones (motor/language/cognition) — AHC", "frequency": "q3M (child), q6M (adult)"},
    {"item": "Ophthalmology — optic atrophy (CAPOS-E818K) + flunarizine fundus", "frequency": "Baseline + annual"},
    {"item": "Audiogram — sensorineural hearing loss (CAPOS-E818K)", "frequency": "Baseline + q24M"},
    {"item": "EEG — annual + during febrile illness / status events", "frequency": "Annual + acute events"},
    {"item": "MRI brain — structural baseline + 2-year follow-up", "frequency": "At diagnosis + 2y + clinically"},
    {"item": "Flunarizine QTc (ECG) + somnolence + extrapyramidal signs", "frequency": "Baseline ECG; q12M review"},
    {"item": "VPA TDM + LFT + FBC + ammonia (if on VPA)", "frequency": "q3M (TDM); q6M (LFT/FBC)"},
    {"item": "Neuropsychological assessment + school/learning support", "frequency": "q2y (school-age)"},
    {"item": "SUDEP risk counselling (GTCS ≥3/year)", "frequency": "Annual — all patients with seizures"},
    {"item": "AHC emergency action plan review (fever/bathing/SE rescue)", "frequency": "Every 6M; update at school transition"},
    {"item": "Genetic counselling — de novo AD (recurrence ~0%; germline mosaicism ~1-2%)", "frequency": "At diagnosis; family planning"},
    {"item": "Physiotherapy / OT / rehabilitation — AHC motor + cognitive rehabilitation", "frequency": "Ongoing (AHC)"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE WINDOWS  (6)
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {
        "window": "Neonatal / Early Infancy (0–6 months)",
        "headline": "First AHC attack; oculomotor signs; E815K-DEE = infantile spasms onset",
        "details": (
            "AHC onset: median 4-6 months (range: birth–18 months). First attacks: "
            "tonic eye deviation ± nystagmus, then hemiplegic episodes. E815K: infantile spasms, "
            "hypsarrhythmia, often no classic AHC. Genetic testing priority. "
            "Flunarizine: start once AHC confirmed. Avoid CBZ/OXC absolutely."
        ),
    },
    {
        "window": "Infancy (6–18 months)",
        "headline": "Hemiplegic attacks established; trigger identification; flunarizine optimization",
        "details": (
            "Alternating hemiplegia becomes established pattern. Trigger mapping begins: "
            "fever plan, bathing protocol, emotional/physical stress limits. "
            "Flunarizine dose optimization. LEV added if epilepsy diagnosed. "
            "Developmental monitoring starts — many show delay in motor/language milestones."
        ),
    },
    {
        "window": "Toddler / Early Childhood (18 months–5 years)",
        "headline": "Epilepsy diagnosis; school readiness assessment; SE action plan",
        "details": (
            "Epilepsy (if present) fully characterizes. AED optimization (LEV ± VPA-with-POLG). "
            "Written SE rescue plan for daycare/preschool. Physiotherapy + OT + speech. "
            "CAPOS children: ataxia episodes during febrile illness become recognizable. "
            "RDP: not yet apparent at this age."
        ),
    },
    {
        "window": "Childhood (5–12 years)",
        "headline": "Learning disability support; school accommodation; attack frequency reassessment",
        "details": (
            "Intellectual disability characterization (AHC: moderate ID in ~70%). "
            "School accommodation: extra time, learning support assistant, flexible attendance. "
            "Flunarizine continuation — monitor extrapyramidal signs (chronic use). "
            "CAPOS: progressive optic atrophy + hearing loss may become symptomatic. "
            "AHC attacks may stabilize or reduce in frequency in middle childhood."
        ),
    },
    {
        "window": "Adolescence (12–18 years)",
        "headline": "VPPP females (VPA); CAPOS hearing/vision support; RDP onset (adult transition)",
        "details": (
            "VPPP enrollment for females on VPA (MHRA 2021). Contraception counselling. "
            "CAPOS: hearing aids; low vision support; ataxia rehabilitation. "
            "RDP: rare in adolescence but watch for sudden dystonia-parkinsonism. "
            "Seizure driving law counselling. Transition to adult neurology planning."
        ),
    },
    {
        "window": "Adult (18+ years)",
        "headline": "Transition; RDP management; CAPOS deafness/ataxia; reproductive planning",
        "details": (
            "AHC adults: attack frequency may reduce but epilepsy persists; de novo risk ~0%. "
            "RDP onset: sudden-onset dystonia-parkinsonism after trigger. "
            "Avoid D2 antagonists (metoclopramide in ER — HIGH RISK). "
            "CAPOS: cochlear implant evaluation if SNHL severe; low vision clinic. "
            "Pregnancy: VPA discontinue or VPPP; flunarizine safe data limited (discontinue if possible)."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONCEPTS  (15 definitions)
# ─────────────────────────────────────────────────────────────────────────────
CONCEPTS = [
    {
        "term": "ATP1A3 (19q13.2)",
        "definition": (
            "Gene encoding Na+/K+-ATPase alpha-3 (α3) subunit — neuron-specific catalytic subunit "
            "of the primary neuronal ion pump. 23 exons; 1013 aa. pLI ~0.99 (extremely intolerant). "
            "Key mutations: D801N (AHC moderate), E815K (AHC severe/DEE), E818K (CAPOS), G947R (AHC mild)."
        ),
    },
    {
        "term": "Na+/K+-ATPase α3 (NKAα3)",
        "definition": (
            "Electrogenic ion pump: extrudes 3 Na+ out and imports 2 K+ in per ATP cycle → "
            "maintains neuronal resting membrane potential. Neuron-specific (α3 isoform); "
            "unlike α1 (ubiquitous) or α2 (astrocytes). α3 forms heterodimer with β-subunit (ATP1B1/B2). "
            "LOF → Na+ accumulation → Vm failure → AHC attacks + epilepsy."
        ),
    },
    {
        "term": "AHC — Alternating Hemiplegia of Childhood (OMIM #614820)",
        "definition": (
            "Core syndrome of ATP1A3 LOF: episodic hemiplegia alternating sides, bilateral hemiplegia, "
            "oculomotor abnormalities (tonic eye deviation, nystagmus), dystonic attacks, autonomic features. "
            "PATHOGNOMONIC: all attacks RESOLVE WITH SLEEP. Onset <18M. Epilepsy in ~55%. AD de novo."
        ),
    },
    {
        "term": "CAPOS Syndrome (OMIM #601543)",
        "definition": (
            "Cerebellar Ataxia + Areflexia + Pes Cavus + Optic Atrophy + Sensorineural Hearing Loss. "
            "Caused by ATP1A3 E818K mutation. Febrile-triggered acute ataxia episodes. Progressive. "
            "Epilepsy rare. Hearing aids + low vision support primary management."
        ),
    },
    {
        "term": "RDP — Rapid-onset Dystonia-Parkinsonism (OMIM #128235)",
        "definition": (
            "Adult-onset syndrome: sudden (hours-days) dystonia + parkinsonism after trigger (fever, stress). "
            "Rostrocaudal gradient (face/arm > leg). Caused by ATP1A3 D923N, P905L, others. "
            "No significant epilepsy. Avoid D2 antagonists (worsen dystonia)."
        ),
    },
    {
        "term": "D801N Mutation",
        "definition": (
            "c.2401G>A (p.Asp801Asn) — transmembrane domain 5 missense; ~60% of all AHC cases. "
            "Moderate pump impairment (~40-50% activity loss). Moderate-severe AHC phenotype. "
            "Epilepsy in ~50%. Responds reasonably to flunarizine. Most studied AHC variant."
        ),
    },
    {
        "term": "E815K Mutation",
        "definition": (
            "c.2443G>A (p.Glu815Lys) — transmembrane domain 5; ~20% of AHC. Severe pump impairment "
            "(~70-80% loss). Most severe phenotype: neonatal/early infantile, refractory epilepsy "
            "(infantile spasms), spastic quadriplegia. DEE phenotype overlap. Worst outcome."
        ),
    },
    {
        "term": "E818K Mutation (CAPOS)",
        "definition": (
            "c.2452G>A (p.Glu818Lys) — transmembrane domain 5 (3 residues from E815K); "
            "virtually all CAPOS cases. Distinct conformational effect from E815K. "
            "Progressive neurodegeneration (optic atrophy, SNHL, ataxia). Rare epilepsy."
        ),
    },
    {
        "term": "Hemiplegic Attack (AHC) — Pathognomonic Features",
        "definition": (
            "Episodic weakness/paralysis of one or both sides; alternates sides across attacks. "
            "Co-features: tonic eye deviation, nystagmus, dystonic posturing, autonomic signs. "
            "PATHOGNOMONIC: RESOLVES WITH SLEEP (even brief nap ends attack). "
            "Duration: minutes to days. Trigger: fever, bathing, excitement, exertion."
        ),
    },
    {
        "term": "Flunarizine (AHC-specific therapy)",
        "definition": (
            "Calcium channel blocker (L-type + T-type) + H1 antihistamine. Level B for AHC attack prevention. "
            "Reduces attack frequency/severity ~70-80%. Does NOT correct Na+/K+-ATPase deficiency. "
            "NOT a Na+ channel blocker (CBZ is — that's ABSOLUTE CI). 5-10mg nocte; lifelong."
        ),
    },
    {
        "term": "CBZ/OXC ABSOLUTE CI in AHC",
        "definition": (
            "Carbamazepine and oxcarbazepine block voltage-gated Na+ channels → "
            "trigger severe, prolonged hemiplegic crises in AHC patients. "
            "Multiple case reports of bilateral hemiplegic attacks within hours-days of CBZ/OXC start. "
            "ABSOLUTE CI — regardless of epilepsy type. Use LEV, VPA, CLB, TPM instead."
        ),
    },
    {
        "term": "POLG-Alpers (mandatory screen before VPA)",
        "definition": (
            "POLG1 encodes DNA polymerase gamma (mitochondrial DNA replication). Biallelic POLG1 LOF "
            "→ Alpers-Huttenlocher progressive encephalopathy. VPA in POLG1 → acute hepatic failure "
            "(often fatal). ATP1A3 patients: pump already ATP-dependent → any additional mitochondrial "
            "failure is catastrophic. POLG screen mandatory before VPA in all ATP1A3 patients."
        ),
    },
    {
        "term": "D2-Antagonist Risk (RDP/AHC)",
        "definition": (
            "ATP1A3 is highly expressed in dopaminergic neurons (SNc/VTA). D2 receptor blockade "
            "(haloperidol, risperidone, metoclopramide) → acute worsening of dystonia in AHC/RDP. "
            "Alert ER staff: NEVER metoclopramide for nausea/vomiting — use ondansetron. "
            "MIMS/BNF red box alert for all ATP1A3 patients."
        ),
    },
    {
        "term": "SUDEP (Sudden Unexpected Death in Epilepsy)",
        "definition": (
            "~1:1000 epilepsy patients per year; up to 1:100 for drug-resistant epilepsy. "
            "ATP1A3 + refractory epilepsy + GTCS ≥3/year = high SUDEP risk. "
            "Risk reduction: nocturnal supervision, SUDEP monitors (e.g., SAMi mattress), "
            "seizure diary, aggressive GTCS control, avoid alcohol/sleep deprivation."
        ),
    },
    {
        "term": "ACMG-AMP 2015 Variant Classification",
        "definition": (
            "5-tier variant classification: Pathogenic / Likely Pathogenic / VUS / "
            "Likely Benign / Benign. Applied to ATP1A3 variants: D801N/E815K/E818K = Pathogenic. "
            "Novel missense: functional assay (Xenopus oocyte pump activity) required for "
            "Pathogenic classification. ClinVar / LOVD-AHC database primary resources."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"metric": "AHC onset age (diagnostic cutoff)", "threshold": "<18 months", "note": "AHC ILAE criterion; later onset → consider RDP or other diagnosis"},
    {"metric": "Fever trigger threshold", "threshold": "≥37.5°C", "note": "Antipyretics at first sign; do not wait for 38°C in AHC"},
    {"metric": "Attack frequency — flunarizine optimization", "threshold": ">2 attacks/week", "note": "Increase flunarizine dose or add LEV if >2 attacks/week at optimal dose"},
    {"metric": "Status epilepticus — rescue medication threshold", "threshold": ">5 minutes (seizure) or >30s (bilateral convulsive)", "note": "Buccal midazolam → call 999/911; do not wait"},
    {"metric": "POLG screen timing (VPA)", "threshold": "BEFORE first VPA dose", "note": "No exceptions — even single dose of VPA risk if POLG+"},
    {"metric": "VPA TDM target", "threshold": "50–100 μg/mL", "note": "Check trough (pre-dose); adjust q3M"},
    {"metric": "Flunarizine dose — adult target", "threshold": "5–10 mg nocte", "note": "QTc >450ms (M) / >470ms (F) → cardiology review before continuing"},
    {"metric": "SUDEP high-risk threshold", "threshold": "GTCS ≥3/year + drug-resistant epilepsy", "note": "Nocturnal seizure monitor + SUDEP counselling at every annual review"},
    {"metric": "Ophthalmology review frequency (CAPOS)", "threshold": "Annual", "note": "Optic atrophy CAPOS: annual OCT + visual fields"},
    {"metric": "Audiogram frequency (CAPOS)", "threshold": "Baseline + every 24 months", "note": "SNHL in CAPOS is progressive; cochlear implant evaluation if severe"},
    {"metric": "MRI brain repeat (AHC)", "threshold": "At 2 years post-diagnosis", "note": "Assess for progressive cerebellar atrophy (30% AHC); earlier if regression"},
    {"metric": "Triheptanoin dose (investigational AHC)", "threshold": "1–2 g/kg/day with meals", "note": "Compassionate use only; monitor acylcarnitines; no RCT threshold established"},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"code": "ILAE-2022", "title": "ILAE 2022 Operational Classification of Seizure Types & Epilepsies", "relevance": "AHC classified under channelopathy-DEE spectrum; seizure classification"},
    {"code": "NICE-NG217", "title": "NICE NG217 Epilepsies in Children/Young People/Adults (2022)", "relevance": "AED evidence levels; SE management; POLG/VPA safety; VPPP"},
    {"code": "Heinzen-2012-NatGenet", "title": "Heinzen et al. 2012 Nature Genetics 44:1030 — ATP1A3 discovery in AHC", "relevance": "Primary AHC gene discovery; D801N, E815K, G947R identified"},
    {"code": "Demos-2014-AnnNeurol", "title": "Demos et al. 2014 Annals of Neurology 75:196 — AHC genotype-phenotype", "relevance": "D801N vs E815K phenotype severity; epilepsy rates; outcome"},
    {"code": "Panagiotakaki-2015-Brain", "title": "Panagiotakaki et al. 2015 Brain 138:2520 — AHC-Europe 187-patient consortium", "relevance": "Natural history; flunarizine outcomes; triggers; lifecycle"},
    {"code": "Rosewich-2012-NeurolGenet", "title": "Rosewich et al. 2012 Neurology Genetics — E818K CAPOS identification", "relevance": "CAPOS molecular basis; E818K pathogenicity"},
    {"code": "Bhatt-2023-Epilepsia", "title": "Bhatt et al. 2023 Epilepsia — Genetic epilepsy precision medicine review", "relevance": "Standard reference for gene-epilepsy dashboards"},
    {"code": "Bhattacharya-2018-Pediatrics", "title": "Bhattacharya 2018 Pediatrics — AHC clinical review", "relevance": "Clinical management; emergency action plans; school support"},
    {"code": "CPIC-POLG-2023", "title": "CPIC Clinical Pharmacogenomics POLG-VPA Guideline 2023", "relevance": "POLG screen mandatory before VPA; interpretation; alternatives"},
    {"code": "FDA-VPA-REMS", "title": "FDA Valproate REMS — teratogenicity, hepatotoxicity, POLG risk", "relevance": "VPA prescribing requirements; VPPP females; POLG warnings"},
    {"code": "ACMG-AMP-2015", "title": "ACMG/AMP Variant Interpretation Standards 2015", "relevance": "ATP1A3 variant classification (Pathogenic/LP/VUS); functional evidence criteria"},
    {"code": "ILAE-Diet-2018", "title": "ILAE Dietary Therapy Guideline 2018", "relevance": "Ketogenic diet evidence for refractory DEE including E815K-DEE"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES  (6)
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"id": "Heinzen-2012", "citation": "Heinzen EL et al. (2012) De novo mutations in ATP1A3 cause alternating hemiplegia of childhood. Nature Genetics 44:1030-1034", "key_finding": "ATP1A3 as AHC gene; D801N (60%), E815K (20%), G947R (10%) as main pathogenic variants"},
    {"id": "Demos-2014", "citation": "Demos MK et al. (2014) A novel recurrent mutation in ATP1A3 causes CAPOS syndrome. Annals of Neurology 75:196-202 [Note: CAPOS reference; also genotype-phenotype]", "key_finding": "E818K = CAPOS; D801N vs E815K phenotype severity spectrum"},
    {"id": "Panagiotakaki-2015", "citation": "Panagiotakaki E et al. (2015) Evidence of a non-malignant course of alternating hemiplegia of childhood: Study of a large cohort of patients. Brain 138:2520-2530", "key_finding": "187-patient AHC cohort; flunarizine ~70-80% attack reduction; trigger profile; lifecycle outcomes"},
    {"id": "Rosewich-2012", "citation": "Rosewich H et al. (2012) Heterozygous de novo mutations in ATP1A3 in patients with alternating hemiplegia of childhood: a whole-exome sequencing gene-identification study. Lancet Neurology 11:764-773", "key_finding": "E818K CAPOS identification; spectrum of ATP1A3 mutations across syndromes"},
    {"id": "Bhatt-2023", "citation": "Bhatt DL et al. (2023) Genetic epilepsy syndromes: a precision medicine framework. Epilepsia 64(S1):1-45", "key_finding": "Standard gene-epilepsy reference; treatment evidence levels; safety monitoring framework"},
    {"id": "Bhattacharya-2018", "citation": "Bhattacharya A (2018) Alternating Hemiplegia of Childhood. Pediatrics 141(6):e20164308", "key_finding": "Clinical review; emergency management; school support; trigger avoidance; flunarizine management"},
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT GENERATOR  (40 synthetic patients)
# ─────────────────────────────────────────────────────────────────────────────
def _make_patients():
    random.seed(42)
    patients = []
    pid = 1
    for i in range(40):
        cat = _cats[i % len(_cats)]
        is_d801n = cat == "ATP1A3-D801N-AHC-moderate"
        is_e815k = cat == "ATP1A3-E815K-AHC-severe-DEE"
        is_capos = cat == "ATP1A3-E818K-CAPOS"
        is_g947r = cat == "ATP1A3-G947R-other-AHC"
        is_phenocopy = cat == "ATP1A3-phenocopy-negative"

        sex = random.choice(["F", "M"])  # AHC 50/50 sex ratio (unlike GGE)
        age = (
            random.randint(2, 10) if is_e815k else
            random.randint(5, 40) if is_capos else
            random.randint(3, 35)
        )
        onset_age_months = (
            random.randint(1, 4) if is_e815k else
            random.randint(2, 12) if is_d801n else
            random.randint(4, 18) if is_g947r else
            random.randint(6, 36) if is_capos else
            random.randint(3, 24)
        )

        drug_resistant = (
            True if is_e815k else
            random.random() < 0.15 if is_d801n else
            random.random() < 0.10 if is_g947r else
            random.random() < 0.05 if is_capos else
            random.random() < 0.20
        )

        has_epilepsy = (
            True if is_e815k else
            random.random() < 0.55 if is_d801n else
            random.random() < 0.40 if is_g947r else
            random.random() < 0.10 if is_capos else
            random.random() < 0.40
        )

        on_flunarizine = (
            random.random() < 0.85 if not is_capos and not is_phenocopy else
            random.random() < 0.30
        )
        on_lev = has_epilepsy and random.random() < 0.65
        on_vpa = has_epilepsy and random.random() < 0.35
        on_clb = has_epilepsy and random.random() < 0.30
        on_tpm = drug_resistant and random.random() < 0.40
        on_kd = is_e815k and random.random() < 0.45
        on_triheptanoin = is_e815k and random.random() < 0.12

        polg_tested = "Y" if on_vpa and random.random() < 0.82 else "N"
        has_capos_features = is_capos
        has_optic_atrophy = is_capos and random.random() < 0.70
        has_snhl = is_capos and random.random() < 0.65
        has_status_epilepticus = has_epilepsy and random.random() < (0.45 if is_e815k else 0.20)
        sudep_high_risk = drug_resistant and has_epilepsy and random.random() < 0.45
        fever_triggered = random.random() < 0.85
        bath_triggered = (not is_capos) and random.random() < 0.70
        attacks_per_month = (
            random.randint(4, 20) if is_e815k else
            random.randint(2, 10) if is_d801n else
            random.randint(1, 6) if is_g947r else
            random.randint(0, 2) if is_capos else
            random.randint(1, 8)
        ) if not is_phenocopy else 0

        patients.append({
            "id": f"AT3-{pid:03d}",
            "category": cat,
            "sex": sex,
            "age": age,
            "onset_age_months": onset_age_months,
            "has_epilepsy": has_epilepsy,
            "drug_resistant": drug_resistant,
            "on_flunarizine": on_flunarizine,
            "on_lev": on_lev,
            "on_vpa": on_vpa,
            "on_clb": on_clb,
            "on_tpm": on_tpm,
            "on_kd": on_kd,
            "on_triheptanoin": on_triheptanoin,
            "polg_tested": polg_tested,
            "has_capos_features": has_capos_features,
            "has_optic_atrophy": has_optic_atrophy,
            "has_snhl": has_snhl,
            "has_status_epilepticus": has_status_epilepticus,
            "sudep_high_risk": sudep_high_risk,
            "fever_triggered": fever_triggered,
            "bath_triggered": bath_triggered,
            "attacks_per_month": attacks_per_month,
        })
        pid += 1

    while len(patients) < 40:
        patients.append(patients[-1].copy())
        patients[-1]["id"] = f"AT3-{pid:03d}"
        pid += 1
    return patients[:40]


PATIENTS = _make_patients()


# ── API functions ──────────────────────────────────────────────────────────────
def get_overview():
    n = len(PATIENTS)
    n_epilepsy = sum(1 for p in PATIENTS if p["has_epilepsy"])
    n_dre = sum(1 for p in PATIENTS if p["drug_resistant"])
    n_flunarizine = sum(1 for p in PATIENTS if p["on_flunarizine"])
    n_lev = sum(1 for p in PATIENTS if p["on_lev"])
    n_vpa = sum(1 for p in PATIENTS if p["on_vpa"])
    n_kd = sum(1 for p in PATIENTS if p["on_kd"])
    n_capos = sum(1 for p in PATIENTS if p["has_capos_features"])
    n_optic = sum(1 for p in PATIENTS if p["has_optic_atrophy"])
    n_snhl = sum(1 for p in PATIENTS if p["has_snhl"])
    n_se = sum(1 for p in PATIENTS if p["has_status_epilepticus"])
    n_sudep = sum(1 for p in PATIENTS if p["sudep_high_risk"])
    n_polg = sum(1 for p in PATIENTS if p["polg_tested"] == "Y")
    n_triheptanoin = sum(1 for p in PATIENTS if p["on_triheptanoin"])
    avg_onset = round(sum(p["onset_age_months"] for p in PATIENTS) / n, 1)

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
        "title": "ATP1A3 Epilepsy (AHC / CAPOS / RDP / DEE-ATP1A3 / Na+/K+-ATPase α3 / 19q13.2)",
        "gene": "ATP1A3",
        "locus": "19q13.2",
        "inheritance": (
            "Autosomal dominant — >95% de novo; germline mosaicism ~1-2%. "
            "RDP: AD familial cases reported. CAPOS: AD de novo (E818K)."
        ),
        "protein": (
            "Na+/K+-ATPase alpha-3 (α3) subunit — neuron-specific electrogenic pump; "
            "3Na+ out / 2K+ in per ATP; maintains resting Vm; critical for post-AP recovery. "
            "Forms α/β heterodimer with ATP1B1/B2. pLI ~0.99."
        ),
        "mechanism": (
            "ATP1A3 LOF haploinsufficiency → 50% reduction in neuronal Na+/K+ pump capacity → "
            "Na+ accumulates inside neurons during repetitive firing → failure to restore resting Vm → "
            "episodic Vm collapse (hemiplegic/dystonic attacks) OR sustained hyperexcitability (epilepsy). "
            "Pump failure also = energy crisis (ATP depleted under load) → metabolic crisis in neurons. "
            "D801N: moderate pump impairment → AHC moderate. E815K: severe → AHC-DEE. "
            "E818K: distinct effect → CAPOS neurodegeneration."
        ),
        "key_aha": (
            "ATP1A3: Na+/K+-ATPase α3 LOF → AHC / CAPOS / RDP / DEE. UNIQUE FEATURES: "
            "(1) AHC ATTACKS RESOLVE WITH SLEEP — pathognomonic (no other epilepsy gene). "
            "(2) FLUNARIZINE = first-line AHC attack prevention (NOT a Na-blocker). "
            "(3) CBZ/OXC ABSOLUTE CI — triggers severe bilateral hemiplegic crises. "
            "(4) D2-ANTAGONISTS HIGH RISK — worsen dystonia (metoclopramide in ER = danger). "
            "(5) POLG MANDATORY before VPA. CAPOS = E818K; progressive deafness + optic atrophy."
        ),
        "kpis": {
            "n_patients": n,
            "epilepsy_pct": round(100 * n_epilepsy / n),
            "drug_resistant_pct": round(100 * n_dre / n),
            "on_flunarizine_pct": round(100 * n_flunarizine / n),
            "on_lev_pct": round(100 * n_lev / n),
            "on_vpa_pct": round(100 * n_vpa / n),
            "on_kd_pct": round(100 * n_kd / n),
            "capos_n": n_capos,
            "optic_atrophy_n": n_optic,
            "snhl_n": n_snhl,
            "status_epilepticus_n": n_se,
            "triheptanoin_investigational_n": n_triheptanoin,
            "polg_tested_pct": round(100 * n_polg / max(n_vpa, 1)),
            "sudep_high_risk_n": n_sudep,
            "avg_onset_months": avg_onset,
        },
        "etiology_distribution": etiology_dist,
        "treatments_summary": [{"drug": t["drug"].split(" — ")[0], "level": t["level"]} for t in TREATMENTS],
        "monitoring_summary": [{"item": m["item"], "frequency": m["frequency"]} for m in MONITORING_ITEMS[:8]],
        "lifecycle": [{"window": w["window"], "headline": w["headline"]} for w in LIFECYCLE_WINDOWS],
        "thresholds": THRESHOLDS[:6],
        "contraindications_summary": [c["drug"].split(" — ")[0].split(" (")[0] for c in CONTRAINDICATIONS],
        "standards": STANDARDS,
        "references": REFERENCES,
    }


def get_breakdown():
    n = len(PATIENTS)
    cat_counts = {}
    for p in PATIENTS:
        cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1

    n_epilepsy = sum(1 for p in PATIENTS if p["has_epilepsy"])
    n_dre = sum(1 for p in PATIENTS if p["drug_resistant"])
    n_flunarizine = sum(1 for p in PATIENTS if p["on_flunarizine"])
    n_lev = sum(1 for p in PATIENTS if p["on_lev"])
    n_vpa = sum(1 for p in PATIENTS if p["on_vpa"])
    n_kd = sum(1 for p in PATIENTS if p["on_kd"])
    n_capos = sum(1 for p in PATIENTS if p["has_capos_features"])
    n_optic = sum(1 for p in PATIENTS if p["has_optic_atrophy"])
    n_snhl = sum(1 for p in PATIENTS if p["has_snhl"])
    n_se = sum(1 for p in PATIENTS if p["has_status_epilepticus"])
    n_sudep = sum(1 for p in PATIENTS if p["sudep_high_risk"])
    n_polg = sum(1 for p in PATIENTS if p["polg_tested"] == "Y")
    n_vpa_no_polg = sum(1 for p in PATIENTS if p["on_vpa"] and p["polg_tested"] == "N")
    n_triheptanoin = sum(1 for p in PATIENTS if p["on_triheptanoin"])
    n_fever = sum(1 for p in PATIENTS if p["fever_triggered"])
    n_bath = sum(1 for p in PATIENTS if p["bath_triggered"])

    return {
        "summary": {
            "total": n,
            "epilepsy_pct": round(100 * n_epilepsy / n),
            "drug_resistant_pct": round(100 * n_dre / n),
            "on_flunarizine_pct": round(100 * n_flunarizine / n),
            "on_lev_pct": round(100 * n_lev / n),
            "on_vpa_pct": round(100 * n_vpa / n),
            "on_kd_pct": round(100 * n_kd / n),
            "capos_n": n_capos,
            "optic_atrophy_n": n_optic,
            "snhl_n": n_snhl,
            "status_epilepticus_n": n_se,
            "fever_triggered_pct": round(100 * n_fever / n),
            "bath_triggered_pct": round(100 * n_bath / n),
            "polg_tested_pct": round(100 * n_polg / max(n_vpa, 1)),
            "vpa_without_polg_n": n_vpa_no_polg,
            "triheptanoin_investigational_n": n_triheptanoin,
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
                "onset_age_months": p["onset_age_months"],
                "has_epilepsy": p["has_epilepsy"],
                "drug_resistant": p["drug_resistant"],
                "on_flunarizine": p["on_flunarizine"],
                "on_lev": p["on_lev"],
                "on_vpa": p["on_vpa"],
                "on_clb": p["on_clb"],
                "on_kd": p["on_kd"],
                "on_triheptanoin": p["on_triheptanoin"],
                "polg_tested": p["polg_tested"],
                "has_capos_features": p["has_capos_features"],
                "has_status_epilepticus": p["has_status_epilepticus"],
                "fever_triggered": p["fever_triggered"],
                "bath_triggered": p["bath_triggered"],
                "attacks_per_month": p["attacks_per_month"],
                "sudep_high_risk": p["sudep_high_risk"],
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
                "atp1a3_note": t["atp1a3_note"],
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
            {"drug": c["drug"].split(" — ")[0].split(" (")[0], "risk": c["risk"]}
            for c in CONTRAINDICATIONS
        ],
        "references": REFERENCES,
    }
