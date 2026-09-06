"""
CACNA1H Epilepsy — Genetic Generalised Epilepsy (GGE) / CAE / JME / GEFS+
===========================================================================
40-patient cohort · CACNA1H (16p13.3) · Cav3.2 T-type Ca²⁺ Channel · AD GOF · GGE Spectrum
Precision treatment: Ethosuximide (T-type Cav3.2 blocker, Level A CAE)

CACNA1H BIOLOGY:
CACNA1H (16p13.3) encodes Cav3.2, the dominant T-type (low-voltage-activated) calcium
channel in the thalamus, hypothalamus, cerebral cortex, and dorsal root ganglia.

KEY POINTS:
  1. Cav3.2 BIOPHYSICS — T-TYPE (TRANSIENT) CALCIUM CHANNEL:
     T-type channels are distinguished by:
       (a) Low activation threshold: activation begins at −70 to −60 mV (well below resting
           Vm of −65 mV in some neurons) — hence 'low-voltage-activated' (LVA).
       (b) Transient current (T for transient): rapid inactivation despite continued
           depolarisation (τ_inactivation ~20–30 ms at −40 mV).
       (c) Small single-channel conductance: ~8 pS for Cav3.2 (cf. 20–25 pS for L-type).
       (d) 'Window current': a persistent component near rest (where steady-state
           activation and inactivation overlap, ~−70 to −50 mV) — physiologically
           relevant for setting resting Ca²⁺ influx in thalamic relay neurons.
     CACNA1H GOF variants typically shift steady-state inactivation to MORE negative
     potentials (e.g. −10 to −20 mV shift in V1/2 inactivation) → channels available
     at more negative voltages → ENLARGED window current at rest → enhanced burst
     firing in thalamic relay neurons even without overt depolarisation.

  2. THALAMO-CORTICAL LOOP AND 3-Hz SWD:
     The thalamo-cortical loop generates 3-Hz spike-wave discharges (SWD) characteristic
     of absence epilepsy:
       Thalamic Relay (TC) neurons (VPL/VPM/VL nuclei):
         - TC neurons express Cav3.2 at high density.
         - At hyperpolarised potentials (e.g., after GABA-B IPSP from TRN): Cav3.2
           de-inactivates → on rebound depolarisation: Cav3.2 opens → low-threshold Ca²⁺
           spike (LTCS) → burst of Na⁺ action potentials → synchronized output to cortex.
         - This burst-pause oscillation at ~3 Hz, when coupled to cortical layer VI,
           produces generalized 3-Hz SWD.
       Thalamic Reticular Nucleus (TRN):
         - TRN is GABAergic and receives collaterals from TC and cortical pyramidal neurons.
         - TRN also expresses Cav3.2 (Cav3.3 dominant but Cav3.2 present).
         - TRN inhibits TC neurons via GABA-B → prolonged hyperpolarisation → de-inactivation
           of TC Cav3.2 → rebound LTCS → synchronized bursting.
       CACNA1H GOF → enhanced Cav3.2 window current in TC neurons → increased burst
       frequency → higher 3-Hz SWD probability → absence/generalized epilepsy.

  3. CACNA1H GENE AND PROTEIN STRUCTURE:
     Gene: CACNA1H (Calcium Voltage-Gated Channel Subunit Alpha1 H), chromosome 16p13.3.
     35 exons; 2353 aa (Cav3.2 α1 subunit). No obligate β-subunit (T-type channels
     are monomeric — 4-domain α1 is sufficient). Auxiliary: β-anchoring-and-regulation
     (BARP) and α2δ proteins modulate surface expression but are not obligatory.
     Domains: DI-DIV (each with S1-S6 transmembrane segments + S4 voltage sensor +
     S5-S6 pore loop). Inactivation gate: domain linker I-II intracellular.
     Key GOF hot-spots: DII-DIII linker (intracellular) · DIII S4-S5 (voltage sensor
     coupling) · DI-DII (exon 6-9 region, V187L/G188V cluster).

  4. CACNA1H EPILEPSY GENETICS:
     pLI: ~0.25 (NOT highly intolerant to LoF; T-type GOF, not haploinsufficiency).
     Inheritance: AD, reduced penetrance (~60–70%); de novo 30–40%; familial 60–70%.
     OMIM: No dedicated OMIM DEE number (CACNA1H variants in epilepsy susceptibility;
     associated: EIG6 = susceptibility to idiopathic generalised epilepsy, OMIM #612899).
     Discovery: Chen YH et al. 2003 Nature Genetics 32(1):76–79 (p.G773D, p.R788C —
     first CACNA1H variants in Chinese CAE patients); replicated globally.
     Prevalence: 5–10% of CAE probands carry CACNA1H variants (many low penetrance).
     Common disease-associated variants: p.G773D · p.R788C · p.V187L · p.A748V ·
     p.G1011S · p.C456S · p.P648L.
     Functional requirement: GOF diagnosis requires electrophysiological confirmation
     (Xenopus oocyte or HEK293) — many variants are risk alleles, not fully penetrant.

  5. CLINICAL PHENOTYPE SPECTRUM:
     (a) Childhood Absence Epilepsy (CAE): 3-Hz SWD, onset 5–10Y, typical absences
         (brief, abrupt, 5–30s, eye flutter). ETX gold standard (Level A). 60–70% remit
         by adolescence. EEG: bilateral synchronous 3-Hz SWD, activated by HV (100%).
     (b) Juvenile Absence Epilepsy (JAE): adolescence onset 10–17Y; GTCS added in 80%;
         less complete remission than CAE. VPA or ETX+LTG.
     (c) Juvenile Myoclonic Epilepsy (JME): morning myoclonus + GTCS ± absence.
         CACNA1H in ~5% of JME. VPA/LEV, lifelong.
     (d) GEFS+ spectrum: febrile seizures plus (FS+) evolving to GGE.
     (e) GTCS-Alone: CACNA1H in isolated GTCS cohorts (low penetrance variants).

  6. PRECISION TREATMENT — ETHOSUXIMIDE (ETX):
     Mechanism: ETX is a selective T-type calcium channel blocker. Primary target:
     Cav3.1 and Cav3.2 in thalamic relay neurons. ETX reduces LTCS (low-threshold
     Ca²⁺ spike) in TC neurons → reduces burst-pause oscillations → suppresses 3-Hz SWD.
     In CACNA1H GOF: ETX directly opposes the enhanced Cav3.2 window current.
     Dose: 20–40 mg/kg/day (divided bid-tid); therapeutic level 40–100 mg/L.
     Evidence: Level A (highest). SANAD trial (2007 Lancet): ETX = VPA for absence
     (non-inferior efficacy; superior tolerability). SANAD II (2021 NEJM): VPA vs ETX
     vs LEV for newly-diagnosed GGE — ETX superior for absence seizure control.
     In CAE: ETX preferred first-line (especially females — avoid VPA VPPP).
     Note: ETX does NOT prevent GTCS → if GTCS present, VPA or ETX+LTG combination.

  7. GGE-AGGRAVATING AGENTS (ABSOLUTE CI IN CACNA1H GGE):
     Sodium-channel blockers (CBZ / OXC / PHT) are ABSOLUTELY CONTRAINDICATED in GGE:
     Mechanism: NaV block in cortical pyramidal neurons spares (or blocks less) thalamic
     GABAergic TRN interneurons → net DISINHIBITION of TC neurons → ENHANCED burst
     firing → worse 3-Hz SWD → absence aggravation or absence status epilepticus.
     Clinically: CBZ/OXC convert CAE to absence status epilepticus in hours.
     SAME contraindication applies across ALL GGE syndromes: CACNA1H, CLCN2, GABRG2,
     GABRA1, GABRA2, GABRB2/B3, KCNT1 GGE, JME, CAE, JAE.

  8. ZONISAMIDE — ALTERNATIVE T-TYPE BLOCKER:
     Zonisamide (ZNS) has dual T-type (Cav3.2) + NaV blocking activity.
     In GGE/absence: ZNS reduces thalamic burst firing (T-type component) with less
     GGE aggravation than pure NaV blockers (NaV component is partial).
     Level B (ZNS for GGE in Japan Level A; outside Japan Level B–C).
     Often used as ETX alternative (ETX has no NaV activity).
     Caution: ZNS + acetazolamide → additive metabolic acidosis (monitor HCO₃⁻).

KEY REFERENCES:
  Chen YH et al. 2003 Nat Genet 32(1):76-79 — CACNA1H in childhood absence epilepsy
  SANAD 2007 Lancet 369:1000-1015 — ETX vs VPA in absence epilepsy (Level A)
  SANAD-II 2021 NEJM 385:2211-2222 — ETX superior for absence seizures
  Bhatt DL et al. 2023 Epilepsia — gene-epilepsy reference
  Khosravani H et al. 2004 J Neurosci 24(35):7481-7488 — CACNA1H GOF mechanism
  Bhattacharya A & Bhatt DL 2020 Front Neurol — channelopathy treatment framework
"""
import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GOF-CAE-Classic",
        "pct": 42,
        "etiology": "CACNA1H GOF missense — Childhood Absence Epilepsy (CAE) classic phenotype",
        "mechanism": (
            "Gain-of-function missense variants (e.g., p.G773D, p.R788C, p.A748V) shift "
            "Cav3.2 steady-state inactivation to more negative potentials (−10 to −20 mV "
            "leftward shift in V1/2 inactivation). This enlarges the window current at rest "
            "(overlap of steady-state activation and inactivation near −65 to −50 mV) → "
            "increased tonic Ca²⁺ influx in thalamic relay neurons at resting voltage → "
            "enhanced rebound burst firing after GABA-B IPSPs from TRN → elevated probability "
            "of 3-Hz burst-pause oscillation synchronisation → typical 3-Hz SWD → "
            "CAE: brief, abrupt, stereotyped absences 5–30s, eye flutter, abrupt offset."
        ),
        "typical_variants": "p.G773D · p.R788C · p.A748V · p.V187L · p.G1011S",
        "onset_age_years": 7,
        "outcome": "60–70% remit by age 12–16Y with ETX; 30% evolve to JAE/JME requiring lifelong AED; GTCS in 20–30%",
    },
    {
        "category": "GOF-JME-Overlap",
        "pct": 25,
        "etiology": "CACNA1H GOF missense — Juvenile Myoclonic Epilepsy (JME) phenotype",
        "mechanism": (
            "GOF CACNA1H variants in JME pedigrees (5% of JME cohorts in founder populations). "
            "Thalamo-cortical loop disruption via enhanced Cav3.2 → morning myoclonic jerks "
            "(SMA/pre-supplementary motor area activation in 3-Hz polyspike-wave bursts on waking), "
            "GTCS (generalised from thalamo-cortical hyper-synchrony), and ± absence. "
            "JME phenotype: onset 12–18Y; morning clustering; sleep deprivation, alcohol, "
            "missed-dose triggers. SUDEP risk in uncontrolled JME + GTCS (higher than CAE)."
        ),
        "typical_variants": "p.C456S · p.P648L · p.Q1952R · p.I1655V",
        "onset_age_years": 14,
        "outcome": "VPA or LEV preferred; 70–80% GTCS-free with optimal treatment; lifelong therapy required; myoclonic may persist",
    },
    {
        "category": "GOF-GEFS-Spectrum",
        "pct": 18,
        "etiology": "CACNA1H GOF missense — GEFS+ spectrum (febrile seizures plus evolving to GGE)",
        "mechanism": (
            "GEFS+ (Genetic Epilepsy with Febrile Seizures Plus) is a spectrum disorder "
            "with multiple causative genes (SCN1A, SCN1B, GABRG2, CACNA1H). CACNA1H GOF "
            "in GEFS+ families: enhanced thalamo-cortical T-type Ca²⁺ current lowers the "
            "febrile seizure threshold (fever-induced depolarisation activates enhanced "
            "Cav3.2 window current → LTCS in TC neurons → SWD/GTCS). Febrile seizures "
            "continue beyond age 6Y (FS+) and evolve to absence/GTCS/myoclonic. Penetrance "
            "in GEFS+ pedigrees ~60%; phenotypic variability within same family is high. "
            "Some family members: only febrile seizures; others: full JME/CAE."
        ),
        "typical_variants": "p.G1011S · p.N1659S (familial GEFS+ pedigrees)",
        "onset_age_years": 3,
        "outcome": "Variable — depends on phenotype endpoint; FS+ resolves in majority; GGE subset requires long-term AED",
    },
    {
        "category": "GOF-GTCS-Alone",
        "pct": 10,
        "etiology": "CACNA1H GOF missense — GTCS-Alone (isolated convulsive seizures)",
        "mechanism": (
            "A subset of CACNA1H GOF carriers present with isolated GTCS without recognisable "
            "absence or myoclonic history. This likely represents missed absence (never enquired), "
            "incomplete penetrance of GGE features, or variants with milder GOF effect. EEG: "
            "generalised polyspike-wave without clinical correlate on IPS or HV; 3-Hz SWD bursts. "
            "Management: VPA or LEV (ETX insufficient without absence component); driving counselling "
            "critical (GTCS risk). Often misclassified as 'idiopathic epilepsy' without genetic testing."
        ),
        "typical_variants": "Low-penetrance CACNA1H variants; p.V187L (common risk allele in some cohorts)",
        "onset_age_years": 16,
        "outcome": "60–75% GTCS-free with VPA/LEV; recurrence on AED withdrawal high; consider lifelong unless prolonged seizure-free",
    },
    {
        "category": "Phenocopy-VUS",
        "pct": 5,
        "etiology": "GGE phenocopy — CACNA1H variant of uncertain significance (VUS) or polygenic risk",
        "mechanism": (
            "CACNA1H is the most-commonly-reported GGE susceptibility gene in population cohorts, "
            "but many reported variants have low effect sizes and do not demonstrate clear GOF "
            "in functional assays. A proportion of patients with GGE phenotype carry CACNA1H VUS "
            "that represents polygenic risk (CACNA1H variant + other GGE susceptibility alleles) "
            "rather than monogenic GOF epilepsy. True monogenic CACNA1H-GOF is distinguished by "
            "functional assay (heterologous expression: V1/2 shift ≥8 mV, current increase ≥30%). "
            "Management: treat as standard GGE; do not presume ETX precision benefit without functional "
            "confirmation; genetics re-review every 18 months."
        ),
        "typical_variants": "VUS / low-frequency risk variants (MAF 0.01–0.1% in gnomAD)",
        "onset_age_years": 9,
        "outcome": "Standard GGE outcomes; monogenic CACNA1H precision (ETX add-on) withheld pending functional confirmation",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES  (5 types)
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Typical Absence",
        "pct": 80,
        "eeg": "Bilateral synchronous 3-Hz SWD; abrupt onset/offset; duration 5–30s; activated by HV (100% in untreated CAE); IPS in 35–40%",
        "semiology": "Sudden cessation of activity; blank stare; eye flutter/upward deviation; lip smacking (automatisms in 40%); ± peri-oral automatisms; abrupt return to awareness; post-ictal-FREE (distinguishes from focal impaired awareness seizures)",
        "clinical_tip": "HV MUST be performed in clinic: 3 min of voluntary hyperventilation triggers absence in >90% of untreated CAE-CACNA1H. The response to HV is diagnostic AND monitors treatment adequacy (ETX should abolish HV-induced absences).",
    },
    {
        "type": "GTCS (Generalised Tonic-Clonic)",
        "pct": 55,
        "eeg": "Generalised polyspike-wave evolving from 3-Hz → high-frequency fast activity (tonic) → rhythmic spike-wave (clonic); post-ictal slowing",
        "semiology": "Tonic phase (10–30s): loss of consciousness, axial stiffening, upward eye deviation, vocalization; clonic phase (30–60s): rhythmic limb jerking; post-ictal confusion/drowsiness 10–30min; incontinence common; tongue bite lateral not central",
        "clinical_tip": "In CACNA1H GGE, GTCS are often morning-onset (post-wake, during missed-dose period or after sleep deprivation). Driving MUST be discussed and documented: GTCS automatically prohibits driving in most jurisdictions until 12-month seizure-free period achieved.",
    },
    {
        "type": "Myoclonic Jerks",
        "pct": 35,
        "eeg": "Polyspike-wave bursts (>3 Hz); brief (<1 s) bilateral synchronous high-amplitude polyspike; consciousness preserved",
        "semiology": "Sudden, brief involuntary muscle jerks; predominantly arms/shoulders; morning clustering (30–90 min after waking); may cause drops (held objects fall); preserved awareness during jerk; may be subtle (patient dismisses as 'stumbling' or 'clumsiness'); enquire specifically",
        "clinical_tip": "Ask: 'Do you ever drop your toothbrush or coffee cup in the morning?' — the signature myoclonic inquiry for JME/CACNA1H-JME. Myoclonus is under-reported: patients often present only after GTCS. EMG-EEG correlation confirms myoclonus vs tremor.",
    },
    {
        "type": "Febrile Seizures Plus (FS+)",
        "pct": 20,
        "eeg": "Febrile context: may be normal or show focal/generalised slowing post-ictally. Afebrile EEG: 3-Hz SWD if GGE component present",
        "semiology": "FS+ = febrile seizures continuing beyond age 6Y; febrile GTCS without clear alternative cause. GEFS+ spectrum marker. May be only manifestation in carrier family members with low-penetrance CACNA1H GOF. Evolution to afebrile seizures (absence/GTCS/myoclonic) in 30–40% of FS+ CACNA1H carriers",
        "clinical_tip": "GEFS+ family history (multiple members with FS or FS+) + CAE/JME phenotype in proband = trigger genetic panel including CACNA1H, SCN1A, SCN1B, GABRG2, GABRD. Fever threshold testing is not done clinically but fever avoidance counselling is appropriate for CACNA1H GEFS+ families.",
    },
    {
        "type": "Absence Status Epilepticus (ASE)",
        "pct": 8,
        "eeg": "Prolonged continuous or nearly-continuous 3-Hz SWD (>30 min); amplitude may fluctuate; may have waxing-waning quality; no post-ictal slowing distinguishes from convulsive SE",
        "semiology": "Prolonged twilight state: patient appears confused, slowed, responding imperfectly; NOT unconscious; may walk, speak (incoherently); duration hours (if untreated). Often precipitated by: AED withdrawal, fever, menstrual phase, OR — critically — inadvertent administration of GGE-aggravating drugs (CBZ/OXC/PHT/tiagabine). Emergency EEG distinguishes ASE from non-convulsive focal SE.",
        "clinical_tip": "Benzodiazepines IV (lorazepam 0.1 mg/kg) abort most ASE rapidly. CRITICAL: if ASE precipitated by CBZ/PHT/OXC recently started → WITHDRAW the offending agent immediately AND add VPA IV. Tiagabine-induced NCSE: tiagabine must be stopped and BZD + VPA given — do NOT add tiagabine even temporarily as add-on in CACNA1H GGE.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS  (8 triggers)
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Sleep deprivation", "pct": 90,
     "note": "Most potent trigger across all GGE/CACNA1H phenotypes. Reduced sleep → reduced slow-wave sleep → impaired thalamic adenosine-mediated inhibition → Cav3.2 window current operates without adenosine suppression → lower 3-Hz SWD threshold. Prescribe sleep hygiene as AED adjunct."},
    {"trigger": "Missed AED dose", "pct": 85,
     "note": "ETX plasma half-life ~40–60h in adults, ~30h in children → missed dose → trough below therapeutic window (40 mg/L) → 3-Hz SWD re-emerges. Stress dose-alarm adherence and dispense weekly pill organizers."},
    {"trigger": "Stress", "pct": 72,
     "note": "Psychosocial stress → HPA axis → cortisol → cortical hyperexcitability → lowers SWD threshold in thalamo-cortical circuit. Exam periods and acute life events cluster with breakthrough CAE/JME. Refer to clinical psychologist for seizure stress management."},
    {"trigger": "Alcohol", "pct": 65,
     "note": "Acute alcohol → sedation (masks seizures); alcohol withdrawal → rebound GABA-A downregulation → GTCS and myoclonic surges 12–24h after heavy intake. JME-CACNA1H: Saturday-morning myoclonus/GTCS classic. AUDIT-C counselling mandatory at every visit."},
    {"trigger": "Hyperventilation (HV)", "pct": 62,
     "note": "Voluntary HV (exercise, anxiety, breath control) → CO₂ washout → respiratory alkalosis → [Ca²⁺]free rises (alkalosis reduces protein binding of Ca²⁺) → Cav3.2 window current enhanced → 3-Hz SWD threshold lowered. HV testing in clinic: 3 min HV triggers absence in >90% untreated CAE. ETX/VPA treatment abolishes HV-induced SWD → monitor as treatment adequacy marker."},
    {"trigger": "Photostimulation (IPS)", "pct": 40,
     "note": "Intermittent Photic Stimulation triggers SWD in ~35–40% of CAE/JME patients. Photo-sensitivity (PS): visual-cortex loop amplification of 3-Hz SWD. PS is NOT universal in CACNA1H — it is a marker of thalamo-occipital loop instability. ETX reduces PS in 70% of PS-positive CAE patients. Advise: flickering lights (disco, video games, sun through trees) may trigger absences."},
    {"trigger": "Fever / intercurrent illness", "pct": 35,
     "note": "Fever lowers seizure threshold (GEFS+ spectrum). Acute febrile illness → intercurrent antipyretic use → paracetamol/ibuprofen → no seizure-specific risk, but undertreatment of fever → threshold lowering. High fever (>38.5°C) → management plan including antipyretics and rescue protocol (CLB 10mg oral or midazolam buccal 0.2mg/kg for convulsive breakthrough)."},
    {"trigger": "Catamenial (perimenstrual)", "pct": 28,
     "note": "Oestrogen is pro-convulsant (enhances NMDA, reduces GABA); progesterone metabolite allopregnanolone is anti-convulsant (positive GABA-A modulator at δ-subunit). Perimenstrual: ALLO withdrawal → net excitatory shift → myoclonic surge/absence clustering. Duncan C1 pattern (days 25–3) most common in CAE/JME. CLB (clobazam) cycle-adjusted add-on (2.5–10 mg days 20–28) is first-line catamenial strategy."},
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS  (8 treatments)
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Ethosuximide (ETX)",
        "level": "Level A",
        "indication": "CAE first-line (absence-dominant); CACNA1H precision T-type Cav3.2 blocker",
        "dose": "20–40 mg/kg/day (children); start 250 mg/day (5–6Y: 125–250 mg/day), increase by 250 mg/week to therapeutic level 40–100 mg/L; TDM q6M",
        "moa": "Selective T-type (Cav3.2/Cav3.1) calcium channel blocker. Reduces LTCS (low-threshold Ca²⁺ spike) in thalamic relay neurons → reduces burst-pause 3-Hz oscillation → suppresses SWD. CACNA1H GOF: ETX directly opposes enhanced window current. Also modulates GABA-B and persistent Na⁺ currents at higher concentrations.",
        "efficacy": "CAE 50% seizure freedom rate in SANAD (2007 Lancet) vs 45% VPA (non-inferior efficacy; superior tolerability); SANAD II (2021 NEJM): ETX vs VPA vs LEV — ETX best 12-month outcome for absence. HV-induced SWD abolition = clinical proxy for ETX efficacy.",
        "safety": "GI: nausea/vomiting (take with food), abdominal pain (improve over weeks); CNS: hiccups, headache, dizziness; rare: SJS/TEN (HLA-B*1502 if Asian ancestry); blood: aplastic anaemia (rare, monitor FBC q6M); psychiatric: rare behavioral change. ADVANTAGE over VPA: no hepatotoxicity, no teratogenicity, no VPPP risk.",
        "monitoring": "TDM q6M (therapeutic 40–100 mg/L); FBC q6M; LFT not required (not hepatically processed via UGT/CYP in same way as VPA). HV in clinic every 6M as ETX efficacy marker (SWD should be suppressed at therapeutic levels).",
        "cacna1h_note": "ETX is the PRECISION drug for CACNA1H GOF: the enhanced Cav3.2 window current is directly blocked by ETX at therapeutic concentrations (40–100 mg/L). Dose escalation to therapeutic upper limit (100 mg/L) before declaring ETX failure.",
    },
    {
        "drug": "Valproate / Valproic Acid (VPA)",
        "level": "Level A",
        "indication": "Full-spectrum GGE (absence + GTCS + myoclonic); males or post-reproductive females preferred",
        "dose": "20–30 mg/kg/day; start 300 mg/day, increase by 300 mg/week; TDM therapeutic 50–100 mg/L",
        "moa": "Broad-spectrum: (1) T-type Ca²⁺ block (Cav3.1/Cav3.2 — same mechanism as ETX at higher concentrations); (2) NaV block (use-dependent); (3) GABA-T inhibition → increased synaptic GABA. Triple mechanism makes VPA the most effective full-spectrum GGE agent. In CACNA1H GGE with GTCS: VPA first-line if ETX insufficient for complete GGE spectrum.",
        "efficacy": "SANAD: VPA = ETX for absence (non-inferior); VPA superior for GTCS-containing phenotype. SANAD II: VPA and ETX comparable for absence; VPA slightly better for full GGE spectrum.",
        "safety": "VPPP (teratogenicity — MHRA 2021 black box; absolute CI in pregnancy without PREVENT oversight); hepatotoxicity (POLG mutation CI); weight gain; polycystic ovarian syndrome; hair loss; hyperammonaemia. ALWAYS: POLG1 screen before VPA; VPPP counselling with annual documentation in females 9–50Y.",
        "monitoring": "POLG1 before prescribing. VPA TDM q3M. LFT/FBC/ammonia q3M. VPPP annual form females. Fasting glucose + lipids q6M (metabolic risk). HbA1c if obesity.",
        "cacna1h_note": "VPA is preferred over ETX when GTCS are present (ETX alone does not prevent GTCS). In CACNA1H JME/GEFS+/GTCS-Alone: VPA first-line (broad spectrum vs ETX narrow spectrum). In CACNA1H CAE: ETX preferred (less toxicity, especially in females).",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B",
        "indication": "GGE (JME, GTCS, absence); preferred in females of childbearing age (vs VPA); CACNA1H GEFS+ spectrum",
        "dose": "20–40 mg/kg/day; start 250 mg bid, increase by 500 mg/week; no TDM required (therapeutic guidance: 10–40 mg/L, but not mandatory)",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulator — reduces vesicular Ca²⁺-triggered glutamate and GABA release. Also inhibits N-type (Cav2.2) channels. Does NOT directly block T-type Ca²⁺ channels (Cav3.2) — therefore less 'precision' for CACNA1H GOF vs ETX, but broad anti-seizure effect including on thalamo-cortical circuits via SV2A in TC and TRN neurons.",
        "efficacy": "Level B for absence (less effective than ETX/VPA for pure absence — SANAD II: LEV worst for absence among 3 drugs). Superior for GTCS/myoclonic component. Female-preferred: no teratogenicity, no VPPP, no CYP interaction.",
        "safety": "Behavioral: irritability/aggression in 10–15% (worst in adolescents and psychiatric comorbidity). Start low/slow (125 mg bid increasing 2-weekly). B6 supplementation (50–100 mg/day) may reduce LEV irritability. No organ toxicity. Safe in pregnancy (Neurology 2006, EURAP registry).",
        "monitoring": "No TDM mandatory. FBC at 6M (leukopenia rare). Renal function q12M (LEV renally cleared — dose reduction in CKD). Mood/behavior diary at every visit. AUDIT-C (alcohol interacts with behavioral SE).",
        "cacna1h_note": "LEV preferred over VPA in CACNA1H females of childbearing age (avoid VPPP). Not preferred for pure CACNA1H CAE (absence) — ETX superior. Best in CACNA1H JME-overlap or GTCS-dominant phenotype.",
    },
    {
        "drug": "Lamotrigine (LTG)",
        "level": "Level B — CAUTION if myoclonic component (EEG mandatory before prescribing)",
        "indication": "Absence + GTCS combination (CAE/JAE with GTCS) — avoid if myoclonic jerks present",
        "dose": "Start 25 mg/day (NO valproate); slow titration over 8 weeks to 100–200 mg/day. If VPA co-medication: start 12.5 mg/day (VPA inhibits LTG glucuronidation → double LTG levels)",
        "moa": "Use-dependent NaV block (NaV1.1, NaV1.2) → reduces repetitive firing. Also reduces glutamate release at presynaptic Na⁺ channels. Does NOT block T-type Ca²⁺ → NOT precision for CACNA1H Cav3.2 GOF. Useful for GTCS add-on to ETX.",
        "efficacy": "Level B for GGE absence. Effective for GTCS. ETX + LTG combination: additive for absence + GTCS (Level B, SANAD). LTG alone for absence: SANAD: LTG inferior to ETX and VPA for absence seizure control.",
        "safety": "SJS/TEN: HLA-B*1502 screen (South/East Asian ancestry) before prescribing — HIGH-RISK Asian patients: avoid LTG or pre-screen. Rash: 10–15% (usually benign maculopapular); CAUTION: any rash → HOLD LTG. Interaction: VPA doubles LTG levels (slow dose and halve maintenance). CRITICAL — MYOCLONIC AGGRAVATION: LTG worsens myoclonic jerks in 15–20% of JME patients (NaV1.1 block in PV+ interneurons → disinhibition → enhanced myoclonus). EEG MANDATORY before LTG prescription to confirm absence of significant polyspike-wave/myoclonic component.",
        "monitoring": "LFT/FBC at 6M. HLA-B*1502 before first prescription if Asian ancestry. EEG pre-LTG AND 4-8 weeks post-initiation (to detect myoclonic worsening). TDM not mandatory but therapeutic 2.5–15 mg/L. Monthly rash review first 3M.",
        "cacna1h_note": "LTG in CACNA1H: acceptable add-on for absence + GTCS. AVOID as monotherapy in CACNA1H JME (myoclonic worsening risk 15–20%). Use ETX (or ETX + LTG) for CACNA1H CAE/JAE. EEG mandatory before LTG in any CACNA1H patient — confirm no polyspike-wave myoclonic component.",
    },
    {
        "drug": "Zonisamide (ZNS)",
        "level": "Level B",
        "indication": "Alternative T-type blocker; GGE absence/JME; CACNA1H second-line when ETX fails or not tolerated",
        "dose": "4–12 mg/kg/day; start 50 mg/day, increase by 50 mg every 2 weeks; therapeutic 10–40 mg/L (TDM q6M)",
        "moa": "Dual mechanism: (1) T-type Ca²⁺ block (Cav3.1, Cav3.2) — same mechanism as ETX but less selective; also blocks Cav3.2 window current in thalamic relay neurons; (2) NaV block (use-dependent) — partial GGE-aggravation risk but mitigated by T-type component. Net: reduces LTCS in TC neurons. CACNA1H GOF: ZNS directly blocks enhanced Cav3.2 current (second-line ETX alternative).",
        "efficacy": "Level A in Japan (JE-approved for absence/JME). Level B outside Japan. Open-label series: ZNS comparable to ETX for absence in ETX-intolerant patients. Effective for GTCS (unlike ETX). ZNS + ETX: NOT recommended (both T-type blockers, marginal additive benefit; additive CNS SE).",
        "safety": "CNS: cognitive slowing, word-finding difficulty, drowsiness. Renal: kidney stones (carbonic anhydrase inhibition — drink ≥2L/day). Metabolic: metabolic acidosis (CA inhibition — monitor HCO₃⁻ q3M). Weight neutral. SJS rare. AVOID ZNS + acetazolamide (additive CA inhibition → metabolic acidosis). Oligohidrosis/hyperthermia in children (rare).",
        "monitoring": "TDM q6M (10–40 mg/L). Renal function q6M. Urinalysis for stones. Serum HCO₃⁻ q3M. Avoid if sulfa allergy (sulphonamide-derived). Weight monthly.",
        "cacna1h_note": "ZNS is the best ETX-alternative for CACNA1H GOF — shares T-type Cav3.2 blocking mechanism. Useful when ETX not tolerated (GI SE). Advantage over ETX: also blocks NaV → addresses GTCS component without adding a pure NaV-blocker (which would worsen absence).",
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B",
        "indication": "Catamenial seizure clustering (cycle-adjusted add-on); nocturnal rescue; adjunct refractory GGE",
        "dose": "Catamenial: 5–10 mg/day for days 20–28 of cycle. Adjunct: 5–20 mg/day (divide bid). Rescue: 10–20 mg stat oral.",
        "moa": "1,5-benzodiazepine (not 1,4-BZD) — GABA-A positive allosteric modulator. Retains activity at α2 and α3 subunit-containing GABA-A receptors (responsible for anxiolytic + anti-seizure effects) with reduced α1 tolerance compared to classical BZDs. Functional selectivity: less sedation, less tolerance than diazepam/clonazepam. Enhances GABA-A Cl⁻ conductance → membrane hyperpolarisation → reduces TC neuron burst-pause oscillation.",
        "efficacy": "Catamenial C1 pattern: CLB cycle-adjusted 5–10 mg/day significantly reduces perimenstrual seizure clustering. General GGE: adjunct Level B, widely used. LGS: Level A (FDA-approved 2011). Tolerance develops to sedation faster than anti-seizure effect (tolerance-resistant for seizures: 1,5-BZD selectivity for α2/α3).",
        "safety": "Sedation (transient). Tolerance to anxiolytic/sedative effect in weeks; anti-seizure effect more durable. Dependence: taper if >4 weeks continuous use. Respiratory depression less than diazepam. Withdrawal: taper over 4+ weeks to avoid rebound seizures. Safe in pregnancy (Lancet Neurol EURAP).",
        "monitoring": "No TDM mandatory (plasma level available if toxicity concern). Sedation VAS at each visit. Assess tolerance at 3M. Cycle diary for catamenial CLB (days 20–28 tracking). Driving — sedation warning.",
        "cacna1h_note": "CLB preferred catamenial strategy in CACNA1H GGE with perimenstrual clustering (GEFS+, CAE/JAE, JME phenotypes). Cycle-adjusted CLB (10 mg days 20–28) is well-tolerated and avoids continuous BZD tolerance. Also used as intermittent rescue around known high-risk triggers (exam periods, travel). Not a T-type Cav3.2 blocker — addresses GABA-A inhibitory deficit, not the primary CACNA1H mechanism.",
    },
    {
        "drug": "Perampanel (PER)",
        "level": "Level B",
        "indication": "GTCS-dominant CACNA1H GGE; JME add-on; refractory absence (off-label)",
        "dose": "Start 2 mg/day at bedtime (with food); increase by 2 mg every 2 weeks; target 8–12 mg/day. Max 12 mg/day for GGE.",
        "moa": "Non-competitive AMPA receptor antagonist (α-amino-3-hydroxy-5-methyl-4-isoxazole-propionic acid). Blocks post-synaptic AMPA-type ionotropic glutamate receptor → reduces fast excitatory neurotransmission → reduces cortical and thalamo-cortical excitability. Does NOT directly block Cav3.2 T-type channels. Complements ETX (different mechanism) and VPA (different mechanism) for multi-drug refractory GGE.",
        "efficacy": "FDA-approved for primary GTCS (2015) and focal seizures. Level B for JME-GTCS combination. AMPA receptor is expressed in cortical pyramidal neurons and thalamic relay neurons → reduces burst generation independently of Cav3.2. PER + ETX: Level B combination for refractory CACNA1H CAE with breakthrough GTCS.",
        "safety": "Behavioral: dizziness (30%), irritability/aggression (10–15% — similar to LEV; monitor behavioral SE); insomnia. Psychiatric: depression/suicidal ideation (monitor PHQ-9). Avoid alcohol (CNS potentiation + behavioral SE). Drug interactions: CYP3A4 inducers (carbamazepine, PHT, rifampicin) reduce PER levels by 50–60% — AVOID CBZ co-prescribing in CACNA1H GGE (independent CI: CBZ absolute CI in GGE).",
        "monitoring": "Behavioral diary at every visit. PHQ-9 at baseline and q6M. Weight q3M (appetite change). Drug interaction screen (CYP3A4 inducers). AUDIT-C (alcohol amplifies behavioral SE).",
        "cacna1h_note": "PER useful in CACNA1H GTCS-dominant or JME-overlap phenotype where ETX (insufficient for GTCS alone) + VPA fails or is not tolerated. PER + ETX: complementary mechanisms (AMPA-block + T-type block) for refractory CACNA1H GGE with mixed seizure types. CAUTION: PER interacts with CBZ/PHT (levels reduced) — but CBZ/PHT are already ABSOLUTE CI in CACNA1H GGE.",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B",
        "indication": "Refractory CACNA1H GGE uncontrolled on ≥3 AEDs; absence status epilepticus refractory",
        "dose": "Classic 4:1 (fat:protein+carb); alternatively MAD (modified Atkins, 10–20g carb/day) or LGIT (low glycaemic index). Initiation in hospital with dietitian/neurology team. Duration minimum 3 months before assessing efficacy.",
        "moa": "Ketone bodies (β-hydroxybutyrate, acetoacetate) produced during fat metabolism: (1) Open KATP channels in neurons → hyperpolarisation; (2) Reduce vesicular glutamate release; (3) Reduce thalamo-cortical excitability via multiple mechanisms. CACNA1H relevance: acidosis component of KD may slightly reduce T-type Ca²⁺ window current (proton block of Cav3.2 at H-motif). Primarily indirect anti-seizure mechanism.",
        "efficacy": "Level B for GGE/generalised absence refractory. >50% seizure reduction in 50–60% of refractory CAE/JME in small series. Absence status epilepticus: Level B add-on. Not first-line: AED-based therapy (ETX/VPA) always first.",
        "safety": "GI: constipation, hyperlipidaemia; growth retardation in children; renal stones (ensure fluid intake ≥2L/day + citrate supplements); carnitine deficiency (supplement 50mg/kg/day); selenium/zinc/B-vitamins supplementation. Long-term: bone density monitoring (DXA q2Y in KD > 2Y).",
        "monitoring": "Urine ketones daily. Fasting lipids q3M. Renal function q6M. DXA at 2Y. Amino acids/carnitine q6M. Dietitian follow-up monthly for first 6M.",
        "cacna1h_note": "KD reserved for CACNA1H GGE refractory to ETX + VPA + ≥1 add-on AED. Absence status epilepticus refractory to IV BZD + IV VPA: KD (especially MAD) may be considered as bridge. Liaise with metabolic/KD team (ILAE KD 2018 guidelines).",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS  (6)
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine / Oxcarbazepine / Phenytoin (CBZ / OXC / PHT)",
        "risk": "ABSOLUTE CI — GGE aggravation → absence status epilepticus",
        "mechanism": "Use-dependent NaV block in cortical pyramidal neurons selectively impairs inhibitory interneuron firing (PV+ interneurons, which have high baseline Na⁺ channel activity, are blocked preferentially) → net disinhibition of TC neurons → ENHANCED thalamo-cortical 3-Hz oscillation → WORSE SWD → absence status epilepticus. Onset may be within HOURS of first dose of CBZ/OXC in CACNA1H GGE. Emergency EEG is required if CBZ/OXC inadvertently started and patient presents with confusion.",
        "management": "NEVER prescribe CBZ, OXC, PHT, eslicarbazepine, or lacosamide (LCM, partial NaV) in any GGE phenotype including CACNA1H. If prescribed accidentally: WITHDRAW IMMEDIATELY. Emergency BZD (lorazepam IV 0.1 mg/kg) + VPA IV for absence status.",
    },
    {
        "drug": "Tiagabine (TGB)",
        "risk": "ABSOLUTE CI — non-convulsive status epilepticus (NCSE)",
        "mechanism": "TGB inhibits GAT-1 (GABA transporter 1) on astrocytes → excess synaptic GABA accumulation → paradoxical GABA-A depolarisation (NKCC1-dependent in dysmaturation; HCO₃⁻ efflux through GABA-A) + GABA-B receptor desensitisation → sustained excitatory state → NCSE. Reported within HOURS of TGB initiation in GGE patients. NEVER use TGB in any GGE syndrome.",
        "management": "ABSOLUTE CI. If TGB given accidentally: STOP immediately. IV lorazepam + IV VPA. Monitor with EEG for NCSE (continuous 3-Hz SWD without clinical correlate).",
    },
    {
        "drug": "Valproate — Females 9–50Y (VPPP / Pregnancy)",
        "risk": "HIGH RISK — teratogenicity (MHRA 2021 VPPP mandatory); avoid unless no alternative",
        "mechanism": "VPA causes neural tube defects (spina bifida, 1–2%), cardiac defects, PCOS, and neurodevelopmental impairment in offspring (IQ, autism, ADHD — 30–40% risk at doses >1000 mg/day). MHRA 2021 Valproate Patient & Pregnancy Prevention Programme (VPPP): annual risk acknowledgement form mandatory for ALL females 9–50Y on VPA in UK. Similar restrictions in EU (ERMA) and Canada.",
        "management": "PREFER ETX (CAE/absence-dominant) or LEV/LTG (JME/GTCS) in CACNA1H females 9–50Y. If VPA only option: VPPP form annually, two reliable contraceptive methods, folic acid 5 mg/day, specialist review. POLG1 screen before VPA (POLG LOF → Alpers syndrome → fatal liver failure with VPA).",
    },
    {
        "drug": "Valproate + POLG1 Mutation",
        "risk": "ABSOLUTE CI — Alpers-Huttenlocher syndrome (fatal hepatotoxicity)",
        "mechanism": "POLG1 encodes mitochondrial DNA polymerase γ. Biallelic POLG1 LOF → Alpers-Huttenlocher syndrome (AHS): progressive neurodegenerative disease with hepatic involvement. VPA in POLG1 AHS → inhibits mitochondrial β-oxidation → acute liver failure (ALT rises within weeks) → irreversible → liver transplant rarely successful. Fatal in 70% of POLG+VPA interactions. POLG1 status MUST be known before VPA is given.",
        "management": "SCREEN POLG1 before VPA in all patients with: developmental regression, hepatopathy, family history liver disease, myopathy, external ophthalmoplegia, ataxia. Use CPIC POLG guidelines (2023). If POLG1 biallelic LOF: ABSOLUTE CI VPA → use LEV, CLB, KD. Report POLG status clearly in medical notes.",
    },
    {
        "drug": "Lamotrigine — Myoclonic component (EEG pre-prescribing mandatory)",
        "risk": "HIGH CAUTION — myoclonic worsening in 15–20% JME/GGE with myoclonic component",
        "mechanism": "LTG blocks NaV1.1 in PV+ fast-spiking GABAergic interneurons (highest NaV1.1 expression) → interneuron firing suppressed → reduced GABAergic inhibition → cortical disinhibition → enhanced polyspike-wave discharge → MYOCLONIC WORSENING. In CACNA1H GGE: if myoclonic component present (JME phenotype), LTG may worsen jerks. Paradox: LTG helps GTCS but worsens myoclonic — requires EEG to define phenotype before prescribing.",
        "management": "EEG MANDATORY before LTG: look for significant polyspike-wave (myoclonic EEG correlate). If myoclonic component present: AVOID LTG monotherapy → prefer ETX (CAE/absence) or LEV/VPA (JME). If GTCS-only (no myoclonic): LTG may be used with EEG monitoring at 4–8 weeks. Myoclonic deterioration after LTG start: WITHDRAW LTG, switch to LEV or VPA.",
    },
    {
        "drug": "Zonisamide + Acetazolamide (or Topiramate)",
        "risk": "CAUTION — additive metabolic acidosis; avoid combination",
        "mechanism": "ZNS and AZM (and topiramate) all inhibit carbonic anhydrase (CA) isoforms. Additive CA inhibition → significantly reduced plasma bicarbonate → metabolic acidosis (HCO₃⁻ <15 mmol/L) → compensatory hyperventilation → paradoxical epilepsy risk (alkalosis-mitigating effect of HCO₃⁻ impaired) + nephrolithiasis. CACNA1H GGE: HV is a potent seizure trigger — if metabolic acidosis triggers compensatory HV → SWD risk increased.",
        "management": "MONITOR: serum bicarbonate, renal function, urinalysis if using any two of: ZNS + AZM + TPM. Hydration ≥2L/day. Avoid combining ZNS + AZM unless specifically indicated (e.g., AZM for CLCN2 GOF co-diagnosis). If metabolic acidosis develops: reduce or discontinue one CA inhibitor.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING  (14 items)
# ─────────────────────────────────────────────────────────────────────────────
MONITORING_ITEMS = [
    {"item": "POLG1 screen before VPA",
     "frequency": "Before first VPA prescription (one-time)",
     "standard": "CPIC POLG guidelines 2023",
     "rationale": "POLG LOF + VPA = fatal Alpers syndrome. SCREEN ALL patients considering VPA."},
    {"item": "VPPP annual form (females 9–50Y on VPA)",
     "frequency": "Annual",
     "standard": "MHRA VPPP 2021",
     "rationale": "Mandatory annual valproate pregnancy risk acknowledgement in UK/EU for all females of childbearing age on VPA."},
    {"item": "Ethosuximide TDM",
     "frequency": "Every 6 months",
     "standard": "ILAE 2022 therapeutic drug monitoring",
     "rationale": "Therapeutic window 40–100 mg/L. ETX TDM at steady state (take sample 12h post-dose). Adjust if <40 (seizures) or >100 (toxicity: hiccups, GI)."},
    {"item": "VPA TDM",
     "frequency": "Every 3 months",
     "standard": "ILAE 2022",
     "rationale": "Therapeutic 50–100 mg/L (total). Free VPA monitoring if suspected protein binding issues."},
    {"item": "LFT/FBC/ammonia (VPA)",
     "frequency": "Every 3 months",
     "standard": "NICE NG217",
     "rationale": "VPA hepatotoxicity monitoring. Elevated ammonia → hyperammonaemic encephalopathy (rare). ALT >3× ULN → consider dose reduction or switch."},
    {"item": "EEG at baseline with HV and IPS",
     "frequency": "At diagnosis, then annually",
     "standard": "ILAE EEG 2022 / ACNS 2021",
     "rationale": "3-Hz SWD documentation. HV (3 min): induces absence in >90% untreated CAE. IPS: photo-sensitivity in 35–40%. Repeat annually or after AED change."},
    {"item": "HV challenge in clinic as ETX efficacy marker",
     "frequency": "At 3 months post-ETX initiation, then every 6 months",
     "standard": "Clinical practice / SANAD 2007 sub-analysis",
     "rationale": "If ETX at therapeutic level, HV-induced SWD should be abolished. Persistent HV-induced SWD at therapeutic ETX level → dose increase or add VPA. This is the MOST USEFUL CLINICAL PROXY for treatment adequacy in CACNA1H CAE."},
    {"item": "EEG pre-LTG (myoclonic screen)",
     "frequency": "Before lamotrigine initiation",
     "standard": "NICE NG217 / ILAE JME 2022",
     "rationale": "LTG worsens myoclonic jerks (15–20% JME). EEG confirms: if polyspike-wave present → avoid LTG monotherapy. If pure SWD without polyspike → LTG acceptable with monitoring."},
    {"item": "Seizure diary (time-of-day, type, trigger)",
     "frequency": "Continuous patient diary; review every visit",
     "standard": "ILAE 2022",
     "rationale": "Morning clustering (myoclonic/GTCS) confirms JME-phenotype vs CAE. Trigger documentation (sleep, alcohol, stress) personalises advice. Absence count (patient may undercount — ask third-party witness)."},
    {"item": "Sleep diary",
     "frequency": "3 months post-diagnosis; review q6M or on change",
     "standard": "ILAE 2022 lifestyle",
     "rationale": "Sleep deprivation is the #1 trigger (90%). Sleep diary documents average hours and quality. Actigraphy if concerned. Prescribe minimum 8h sleep in school-age; 9h in adolescents."},
    {"item": "AUDIT-C alcohol screening",
     "frequency": "Every visit (JME/GEFS+ phenotype)",
     "standard": "NICE NG217 lifestyle",
     "rationale": "Alcohol withdrawal triggers JME/GTCS. AUDIT-C score ≥3 (women) or ≥4 (men) → alcohol harm counselling. Particularly important in CACNA1H JME (Saturday-morning GTCS pattern)."},
    {"item": "Driving counselling and documentation",
     "frequency": "At diagnosis; at every status change; at restart after 12-month seizure-free",
     "standard": "DVLA / Transport Canada / ECMT",
     "rationale": "GTCS: mandatory cessation until 12-month seizure-free. Absence seizures: driving may be permitted after 12-month seizure-free (jurisdiction-dependent). Document advice given. CACNA1H GGE: absence may impair driving even if brief — any absence = impairment during episode."},
    {"item": "SUDEP annual risk counselling",
     "frequency": "Annual",
     "standard": "NICE NG217 / SUDEP Action / ILAE 2022",
     "rationale": "SUDEP risk in CAE is LOW (no nocturnal GTCS). SUDEP risk in CACNA1H JME with uncontrolled GTCS = MODERATE (GTCS principal SUDEP risk factor). Annual SUDEP risk discussion: document in notes. Night-time safety: bed rails, shared bedroom, seizure monitor."},
    {"item": "Neuropsychological assessment (attention/memory)",
     "frequency": "At diagnosis (baseline), then q2Y or on academic concern",
     "standard": "ILAE 2022 / WHO ICF 2019",
     "rationale": "CAE causes attention deficit and working memory impairment (even during apparently 'controlled' absence — subclinical SWD on EEG impairs ongoing cognitive processing). Academic failure in 40% untreated CAE. Assessment: WISC-V/WPPSI-IV (cognitive), Conners (attention), VABS (adaptive). Educational accommodations if cognitive impact demonstrated."},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE WINDOWS  (6)
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Childhood CAE Onset (5–12Y)",
        "phase": "Diagnosis + ETX initiation",
        "priorities": [
            "EEG with HV (3 min) → document 3-Hz SWD → diagnostic confirmation",
            "ETX 20–40 mg/kg/day: titrate to therapeutic 40–100 mg/L; absence freedom within 4–6 weeks if therapeutic",
            "School accommodation letter: absence-related learning impact, EEG monitoring permission",
            "Family counselling: CACNA1H AD inheritance — 50% transmission risk to offspring; variable penetrance",
            "Swimming / bathing supervision until 6-month seizure-free",
            "Neuropsychology baseline: attention/working memory (absence affects cognition even when brief)",
        ],
        "key_risk": "Academic failure from subclinical SWD; under-recognition of absence frequency (teacher/parent education essential)",
    },
    {
        "window": "Adolescence — JME/JAE Risk (12–18Y)",
        "phase": "Phenotype evolution + lifestyle counselling",
        "priorities": [
            "Re-assess EEG: CAE may evolve to JAE (FS persist, GTCS added) or JME (myoclonus emerges 12–18Y)",
            "If morning myoclonus reported: switch to VPA or LEV (ETX alone insufficient if GTCS risk)",
            "Sleep: mandatory 8–9h/night — school / extracurricular schedule counselling",
            "Alcohol education (early): alcohol withdrawal = high GTCS risk in CACNA1H JME",
            "Driving countdown: document expected seizure-free milestone for licence application",
            "Females: begin VPPP discussion if VPA used; start contraception counselling early",
        ],
        "key_risk": "GTCS on first alcohol exposure / first night of sleep deprivation; AED adjustment lag (ETX→VPA/LEV transition delay)",
    },
    {
        "window": "Young Adult (18–25Y)",
        "phase": "Independence, university, driving, employment",
        "priorities": [
            "Driving: confirm 12-month seizure-free status (GTCS) or specialist letter for absence-only drivers",
            "University: academic accommodation; repeat neuropsychology for workplace planning",
            "Alcohol harm reduction: detailed counselling; AUDIT-C at every visit",
            "Employment: seizure-safety counselling (heights, operating machinery, water exposure)",
            "Females: fertility counselling (VPA → VPPP; ETX/LEV preferred); contraception effectiveness (LTG induces OCP metabolism → OCP failure → use non-hormonal or higher-dose OCP)",
            "Males: inform re: VPA gonadotoxicity (reversible); sperm banking if prolonged VPA and fertility planning",
        ],
        "key_risk": "AED non-adherence (lifestyle disruption at university); alcohol+sleep deprivation co-triggers at weekends/parties",
    },
    {
        "window": "Reproductive Years — Females (25–40Y)",
        "phase": "Pre-conception planning, pregnancy, post-partum",
        "priorities": [
            "Pre-conception: switch from VPA to ETX/LEV/LTG; folic acid 5 mg/day ≥3M before conception",
            "Pregnancy: ETX (evidence Category D — some risk, benefit usually outweighs); LEV preferred (safest registry data); avoid VPA",
            "EURAP/NEAD registry data: ETX vs LEV vs LTG safety profile discussion",
            "Breastfeeding: ETX enters breast milk (monitor infant ETX levels if breastfeeding); LEV safer for breastfeeding",
            "Post-partum: sleep deprivation from newborn care → high GTCS/myoclonic risk; support plan with partner",
            "CACNA1H genetic counselling: 50% risk to offspring; prenatal testing if desired",
        ],
        "key_risk": "VPA continuation into unplanned pregnancy (VPPP failures); post-partum sleep deprivation → first post-partum GTCS",
    },
    {
        "window": "Perimenopausal (40–55Y)",
        "phase": "Hormonal transition + seizure re-emergence",
        "priorities": [
            "Catamenial: CLB 5–10 mg days 20–28 may no longer be predictable if cycles irregular → switch to continuous CLB if needed",
            "HRT discussion: oestrogen-containing HRT may worsen GGE; progesterone preferred (or transdermal minimal-dose oestrogen)",
            "Bone density: VPA and PB reduce bone density; DXA scan. Calcium/vitamin D supplementation",
            "Cognitive: perimenopause affects cognition + mood; distinguish from AED cognitive SE",
            "Polypharmacy review: AED-AED and AED-medical drug interactions increase with age",
            "SUDEP: re-assess GTCS control; SUDEP risk counselling update",
        ],
        "key_risk": "Re-emergence of seizures at perimenopause after long seizure-free period; oestrogen-mediated CACNA1H GGE worsening",
    },
    {
        "window": "Senior (60Y+)",
        "phase": "Long-term management, cognitive decline screening",
        "priorities": [
            "Cognitive: distinguish AED cognitive SE from dementia; ETX is relatively well-tolerated cognitively in elderly; LEV behavioral SE increased",
            "Falls: GGE-related absences + myoclonus + AED-related ataxia → high fall risk; physiotherapy assessment",
            "Renal/hepatic function: ETX (renally cleared) and VPA (hepatically metabolized) require dose adjustment with decline",
            "Polypharmacy: AED-cardiovascular and AED-metabolic drug interactions (VPA-aspirin; ETX-CYP interactions minimal)",
            "KD not practical in seniors: adherence and nutritional status challenges",
            "SUDEP: absolute risk lower (GTCS typically well-controlled by this stage); annual counselling maintained",
        ],
        "key_risk": "Cognitive misattribution (AED SE vs dementia vs seizure); polypharmacy drug interactions causing breakthrough seizures or toxicity",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DEFINITIONS / CONCEPTS  (15)
# ─────────────────────────────────────────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "CACNA1H — 16p13.3",
        "definition": (
            "CACNA1H (Calcium Voltage-Gated Channel Subunit Alpha1 H) maps to chromosome 16p13.3. "
            "35 exons; 2353 amino acids; encodes Cav3.2 (T-type, low-voltage-activated calcium channel). "
            "pLI ~0.25 (tolerant of LOF — GOF mechanism in GGE, not haploinsufficiency). "
            "AD inheritance, reduced penetrance ~60–70%. De novo 30–40% (severe GOF variants); "
            "familial 60–70% (risk alleles + low-penetrance GOF). OMIM #607682 (gene) + #612899 (GGE susceptibility locus EIG6)."
        ),
    },
    {
        "term": "Cav3.2 — T-type (Low-Voltage-Activated) Calcium Channel",
        "definition": (
            "Cav3.2 is encoded by CACNA1H. Distinctive properties: (1) Low activation threshold "
            "(activates at −70 to −60 mV — below resting potential in thalamic relay neurons); "
            "(2) Transient current (rapid inactivation at sustained depolarisation, τ ~20–30 ms); "
            "(3) Small single-channel conductance (~8 pS); (4) Window current (persistent component "
            "where steady-state activation and inactivation overlap, −70 to −50 mV). Cav3.2 is the "
            "dominant T-type isoform in thalamic relay neurons (VPL/VPM), hypothalamus, and nociceptors."
        ),
    },
    {
        "term": "Low-Threshold Calcium Spike (LTCS)",
        "definition": (
            "The low-threshold calcium spike (LTCS, also 'Ca²⁺ spike') is the characteristic all-or-nothing "
            "event in thalamic relay neurons at hyperpolarised potentials. Triggered by Cav3.2 "
            "(T-type) de-inactivation after hyperpolarisation (e.g., GABA-B IPSP from TRN). "
            "LTCS: Cav3.2 opens → Ca²⁺ influx → depolarisation → Na⁺ action potential burst "
            "(2–5 spikes). The burst-pause oscillation (LTCS → rebound → GABA-B IPSP → LTCS) "
            "at 3 Hz is the cellular correlate of 3-Hz SWD. CACNA1H GOF → enhanced LTCS probability "
            "at rest → higher SWD frequency → GGE phenotype."
        ),
    },
    {
        "term": "Thalamo-Cortical (TC) Circuit and 3-Hz SWD",
        "definition": (
            "3-Hz spike-wave discharges arise from rhythmic thalamo-cortical synchrony: "
            "TC relay neurons (VPL/VPM) ↔ cortical layer VI pyramidal neurons ↔ TRN (GABAergic). "
            "Cycle: TC burst → TRN GABA-B IPSP → TC hyperpolarisation → Cav3.2 de-inactivation → "
            "rebound LTCS → TC burst (3 Hz). Cortical recruitment: TC → cortical layer IV → "
            "widespread SWD. CACNA1H GOF amplifies LTCS → lower threshold for synchronisation → "
            "3-Hz SWD with minimal triggering. HV abolishes CO₂ buffering of H⁺ → increased T-type "
            "channel availability → powerful 3-Hz SWD inducer."
        ),
    },
    {
        "term": "Window Current (T-Type)",
        "definition": (
            "The window current is the persistent, steady-state calcium influx through T-type channels "
            "at voltages where steady-state activation and steady-state inactivation curves overlap "
            "(approximately −70 to −50 mV for Cav3.2). In wild-type Cav3.2: window current is small "
            "at resting potential (~−65 mV). CACNA1H GOF: shifts steady-state inactivation to more "
            "negative voltages (V1/2 inactivation moves −10 to −20 mV leftward) → larger overlap "
            "region → LARGER window current at rest → tonic Ca²⁺ influx → reduced rebound LTCS "
            "threshold → more frequent TC bursting → GGE. Ethosuximide blocks this window current."
        ),
    },
    {
        "term": "CAE — Childhood Absence Epilepsy",
        "definition": (
            "CAE is an ILAE-2022 defined generalised epilepsy syndrome with onset 5–10Y, typical "
            "absence seizures (3-Hz SWD, duration 5–30s, eye flutter, abrupt return to awareness), "
            "and otherwise normal development. EEG: bilateral synchronous 3-Hz SWD, 100% activated "
            "by 3-min HV, IPS-positive in 35%. Aetiology: largely genetic (polygenic + CACNA1H, "
            "CLCN2, GABRG2, GABRA1 genes). Prognosis: 60–70% remit by mid-adolescence. CACNA1H "
            "associated in 5–10% of CAE probands. First-line: ETX (Level A)."
        ),
    },
    {
        "term": "JME — Juvenile Myoclonic Epilepsy",
        "definition": (
            "JME is the commonest GGE syndrome in adults, onset 12–18Y, characterised by triad: "
            "morning myoclonic jerks (arms/shoulders), GTCS (often triggered by missed sleep, alcohol, "
            "missed AED), and absence seizures (30%). EEG: polyspike-wave ≥3.5 Hz. IPS-positive in "
            "40%. JME is rarely considered 'benign': 90% require lifelong AED. CACNA1H GOF in ~5% "
            "of JME cohorts. Treatment: VPA (Level A, fullest spectrum) or LEV (females). AVOID "
            "LTG monotherapy (myoclonic worsening in 15–20%)."
        ),
    },
    {
        "term": "GEFS+ — Genetic Epilepsy with Febrile Seizures Plus",
        "definition": (
            "GEFS+ is a familial epilepsy syndrome defined by febrile seizures persisting beyond age 6Y "
            "(FS+) and/or evolving to afebrile generalised epilepsy (absence, GTCS, myoclonic) within "
            "a family pedigree. Multiple genes: SCN1A (50% of GEFS+ families), SCN1B, GABRG2, GABRD, "
            "CACNA1H. CACNA1H GOF: enhances thalamic burst firing and febrile seizure threshold → "
            "GEFS+ spectrum. Phenotypic variability within single GEFS+ family: FS to FS+ to CAE "
            "to JME to Dravet-like (especially SCN1A GEFS+). Management: treat the seizure type "
            "rather than the GEFS+ spectrum per se."
        ),
    },
    {
        "term": "ETX Precision — T-type Cav3.2 Blocker",
        "definition": (
            "Ethosuximide (ETX, succinimide class) is the pharmacological PRECISION drug for CACNA1H "
            "GOF: ETX selectively inhibits Cav3.1 and Cav3.2 T-type calcium channels by reducing "
            "open-probability at the voltage range of the window current (−70 to −50 mV). In CACNA1H "
            "GOF: the enhanced window current is directly opposed by ETX. SANAD (2007 Lancet): ETX = "
            "VPA for absence seizure freedom; SANAD II (2021 NEJM): ETX best 12-month outcome for "
            "pure absence. ETX does NOT block GTCS → if GTCS present, VPA or ETX + LTG required."
        ),
    },
    {
        "term": "GGE Aggravation — NaV-Blocker CI",
        "definition": (
            "Sodium-channel blockers (CBZ, OXC, PHT, lacosamide) are ABSOLUTELY CONTRAINDICATED in "
            "all GGE syndromes including CACNA1H GGE. Mechanism: NaV block in cortical neurons "
            "preferentially impairs PV+ fast-spiking GABAergic interneurons (highest NaV1.1 density) "
            "→ net disinhibition of TC neurons → ENHANCED thalamo-cortical oscillation → WORSE "
            "3-Hz SWD → absence aggravation or absence status epilepticus. Clinical onset: hours "
            "to days. Emergency risk: absence status epilepticus. SAME CI applies to all GGE genes: "
            "CACNA1H, CLCN2, GABRG2, GABRA1, GABRA2, GABRB2/B3."
        ),
    },
    {
        "term": "VPPP — Valproate Pregnancy Prevention Programme",
        "definition": (
            "UK/EU mandatory programme (MHRA 2021, updated from 2018 EU Medicines Agency restrictions). "
            "Applies to ALL females aged 9–50Y prescribed valproate. Requirements: (1) annual risk "
            "acknowledgement form (signed by patient/guardian + prescriber); (2) two reliable "
            "contraceptive methods in sexually active women; (3) documentation in patient's notes; "
            "(4) specialist confirmation that VPA is the only suitable treatment. Background: VPA "
            "causes neural tube defects (1–2%), cardiac defects, neurodevelopmental impairment "
            "(IQ 8–9 points lower, autism 3×, ADHD 4× vs untreated epilepsy). CACNA1H females: "
            "prefer ETX/LEV over VPA."
        ),
    },
    {
        "term": "POLG1 — Mitochondrial DNA Polymerase γ (VPA Absolute CI)",
        "definition": (
            "POLG1 encodes mitochondrial DNA polymerase gamma (pol-γ), responsible for mtDNA replication "
            "and repair. Biallelic POLG1 LOF variants → Alpers-Huttenlocher syndrome (AHS): progressive "
            "neuro-hepatic degeneration (cortical epilepsy, cerebellar ataxia, hepatic failure). "
            "VPA in POLG1 AHS → inhibits mitochondrial β-oxidation → acute liver failure → fatal "
            "(70% mortality). ABSOLUTE CI. Screen: pre-VPA POLG1 testing in any patient with "
            "developmental regression, hepatopathy, external ophthalmoplegia, myopathy, ataxia, "
            "or family history of AHS. CPIC POLG 2023 guidelines. CACNA1H + co-occurring POLG1 "
            "mutation: use ETX/LEV, never VPA."
        ),
    },
    {
        "term": "Hyperventilation (HV) as SWD Trigger and ETX Efficacy Marker",
        "definition": (
            "3-minute voluntary hyperventilation (HV) triggers 3-Hz SWD in >90% of untreated CAE. "
            "Mechanism: CO₂ washout → respiratory alkalosis → [H⁺] falls → shifts T-type Ca²⁺ "
            "channel H⁺-mediated block → Cav3.2 window current enhanced at rest → TC neuron "
            "burst-firing → 3-Hz SWD. Clinically: HV is both (1) a diagnostic tool (confirms "
            "GGE/CAE if SWD induced) and (2) an ETX efficacy marker: at therapeutic ETX levels "
            "(40–100 mg/L), HV-induced SWD should be ABOLISHED. Persistent HV-induced SWD after "
            "ETX → dose increase or add VPA."
        ),
    },
    {
        "term": "SUDEP — Sudden Unexpected Death in Epilepsy",
        "definition": (
            "SUDEP: sudden, unexpected, witnessed or unwitnessed, non-traumatic and non-drowning "
            "death in a person with epilepsy, with or without evidence of a seizure, excluding "
            "status epilepticus, where post-mortem does not reveal a toxicological or anatomical "
            "cause of death (ILAE Nashef 1997 definition). Risk in CACNA1H GGE: LOW for CAE "
            "(absence, no nocturnal GTCS = low SUDEP risk; SUDEP rate ~0.5/1000/year GGE vs "
            "~1–3/1000/year uncontrolled epilepsy). HIGHER in CACNA1H JME with uncontrolled "
            "nocturnal GTCS. Annual counselling, night-time safety (seizure monitor, shared bedroom), "
            "optimise GTCS control. Document SUDEP discussion in notes."
        ),
    },
    {
        "term": "ACMG-AMP 2015 — Variant Classification",
        "definition": (
            "CACNA1H variants in GGE are classified using ACMG-AMP 2015 pathogenicity criteria: "
            "Pathogenic/Likely Pathogenic (P/LP) → meets criteria for monogenic CACNA1H-GOF GGE: "
            "de novo variant + functional electrophysiology confirming GOF + consistent phenotype. "
            "VUS → variant present in case + absent/rare in gnomAD but no functional confirmation. "
            "Many CACNA1H epilepsy-reported variants are VUS or low-penetrance risk alleles. "
            "Clinical implication: ETX precision add-on (T-type) justified for confirmed P/LP GOF; "
            "withheld for VUS pending functional data. Re-class review every 18 months (gnomAD/ClinVar updates)."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"name": "ETX therapeutic level",       "value": "40–100 mg/L (plasma trough, 12h post-dose)",
     "action": "Below 40: dose increase. Above 100: GI/CNS toxicity → dose reduction"},
    {"name": "VPA therapeutic level",       "value": "50–100 mg/L (total plasma)",
     "action": "Monitor free level if albumin low or drug interaction. Free VPA 5–20 mg/L"},
    {"name": "SWD-free on HV",              "value": "3-min HV must NOT induce SWD at therapeutic ETX",
     "action": "HV-induced SWD persisting at ETX 100 mg/L → add VPA or switch to ZNS"},
    {"name": "ALT (VPA hepatotoxicity)",    "value": ">3× ULN (ALT >100 U/L)",
     "action": "Hold VPA, specialist review. POLG1 re-check. Consider switch to ETX/LEV"},
    {"name": "Serum ammonia (VPA)",         "value": ">80 µmol/L",
     "action": "Reduce VPA dose. Add L-carnitine 50 mg/kg/day. Investigate POLG, ornithine cycle"},
    {"name": "Serum HCO₃⁻ (CA inhibitors)", "value": "<18 mmol/L",
     "action": "Reduce ZNS or AZM dose. Increase hydration. Avoid combining CA inhibitors"},
    {"name": "POLG1 variants",              "value": "Biallelic pathogenic/likely pathogenic POLG1 variants",
     "action": "ABSOLUTE CI: valproate. Use ETX, LEV, CLB, KD. Neurology+metabolic co-management"},
    {"name": "ETX GI toxicity onset",       "value": "Nausea/vomiting within 2 weeks of initiation",
     "action": "Take with food. Reduce dose tempo. ZNS alternative if persistent GI intolerance"},
    {"name": "GTCS driving restriction",    "value": "≥1 GTCS in prior 12 months",
     "action": "Mandatory driving cessation until 12-month GTCS-free (DVLA/Transport Canada)"},
    {"name": "Absence seizure driving",     "value": "Jurisdiction-dependent: some permit after 12M absence-free",
     "action": "Specialist letter required. ANY seizure causing impaired awareness → driving stopped"},
    {"name": "VPPP age range",              "value": "Females 9–50Y on valproate",
     "action": "Annual VPPP form. Two contraception methods. Folic acid 5 mg/day if fertile"},
    {"name": "IPS photo-sensitivity",       "value": "3–50 Hz IPS-induced SWD (bilateral occipital/generalised)",
     "action": "Photo-sensitive patient: advise screen distance, polarised glasses, avoid disco/strobe"},
]

# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE STANDARDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"code": "ILAE-2022",             "title": "ILAE Epilepsy Syndromes Classification 2022",
     "relevance": "GGE, CAE, JME, GEFS+, GTCS-Alone syndrome definitions"},
    {"code": "NICE-NG217",            "title": "NICE Guideline NG217: Epilepsies in children, young people and adults (2022)",
     "relevance": "AED choice by syndrome; monitoring; VPPP; SUDEP counselling"},
    {"code": "Chen-2003-NatGenet",    "title": "Chen YH et al. Nat Genet 2003;32(1):76-79",
     "relevance": "Discovery of CACNA1H GOF variants (p.G773D, p.R788C) in childhood absence epilepsy"},
    {"code": "SANAD-2007-Lancet",     "title": "Marson AG et al. (SANAD trial). Lancet 2007;369:1000-1015",
     "relevance": "ETX vs VPA vs LTG for GGE/absence: ETX = VPA efficacy, superior tolerability"},
    {"code": "SANAD-II-2021-NEJM",    "title": "Marson A et al. (SANAD II trial). NEJM 2021;385:2211-2222",
     "relevance": "ETX best 12-month absence seizure-freedom rate vs VPA vs LEV"},
    {"code": "Bhatt-2023-Epilepsia",  "title": "Bhatt DL et al. Epilepsia 2023 — Gene-epilepsy reference standard",
     "relevance": "CACNA1H classification, gene-phenotype correlations, evidence levels"},
    {"code": "Khosravani-2004-JNeurosci", "title": "Khosravani H et al. J Neurosci 2004;24(35):7481-7488",
     "relevance": "CACNA1H GOF functional electrophysiology: V1/2 shift and window current enlargement"},
    {"code": "MHRA-VPPP-2021",        "title": "MHRA Valproate Patient & Pregnancy Prevention Programme 2021",
     "relevance": "Annual VPPP form, dual contraception requirement, VPA in females 9–50Y"},
    {"code": "CPIC-POLG-2023",        "title": "CPIC POLG Guideline 2023",
     "relevance": "POLG1 screening before valproate; biallelic LOF → absolute VPA CI"},
    {"code": "CPIC-HLA-B1502-2023",   "title": "CPIC HLA-B*15:02 Guideline 2023",
     "relevance": "HLA-B*1502 screening before LTG in South/East Asian ancestry"},
    {"code": "ACMG-AMP-2015",         "title": "Richards S et al. Genet Med 2015;17:405-424",
     "relevance": "CACNA1H variant pathogenicity classification (P/LP/VUS/LB/B)"},
    {"code": "ILAE-Diet-2018",         "title": "Kossoff EH et al. Epilepsia 2018;59(6):1085-1106",
     "relevance": "Ketogenic diet practice guidelines; KD for refractory GGE/absence"},
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT COHORT  (40 patients)
# ─────────────────────────────────────────────────────────────────────────────
PATIENTS = []
_etio_pool = []
for et in ETIOLOGY_CATALOG:
    n = round(et["pct"] * 40 / 100)
    _etio_pool.extend([et["category"]] * n)
while len(_etio_pool) < 40:
    _etio_pool.append(ETIOLOGY_CATALOG[0]["category"])
random.shuffle(_etio_pool)

_drugs = ["ETX", "VPA", "LEV", "LTG", "ZNS", "CLB", "PER", "KD"]
_seizure_types = ["Typical Absence", "GTCS", "Myoclonic Jerks", "Febrile Seizures+", "Absence Status"]
_female_names = ["Emma", "Lily", "Sarah", "Mia", "Chloe", "Olivia", "Ava", "Sophia", "Isabella", "Grace",
                 "Hannah", "Amelia", "Ella", "Charlotte", "Zoe", "Abigail", "Natalie", "Samantha",
                 "Victoria", "Alexandra"]
_male_names = ["Liam", "Noah", "Ethan", "Oliver", "William", "James", "Lucas", "Benjamin", "Mason",
               "Elijah", "Logan", "Jackson", "Aiden", "Carter", "Sebastian", "Owen", "Luke", "Daniel",
               "Henry", "Matthew"]

for i in range(40):
    sex = "F" if i < 22 else "M"
    name = (_female_names[i % 20] if sex == "F" else _male_names[i % 20])
    etio = _etio_pool[i]
    age = random.randint(7, 45)
    etio_obj = next(e for e in ETIOLOGY_CATALOG if e["category"] == etio)
    onset = int(etio_obj["onset_age_years"] + random.randint(-2, 3))
    onset = max(3, min(onset, age - 1))
    # seizure types
    stypes = ["Typical Absence"]
    if random.random() < 0.55:
        stypes.append("GTCS")
    if etio in ("GOF-JME-Overlap",) and random.random() < 0.7:
        stypes.append("Myoclonic Jerks")
    if etio == "GOF-GEFS-Spectrum" and random.random() < 0.6:
        stypes.append("Febrile Seizures+")
    # drugs
    if etio in ("GOF-CAE-Classic", "Phenocopy-VUS"):
        primary_drug = random.choice(["ETX", "ETX", "ETX", "VPA", "LEV"])
    elif etio == "GOF-JME-Overlap":
        primary_drug = random.choice(["VPA", "LEV", "LEV", "ETX+LEV"])
    elif etio == "GOF-GTCS-Alone":
        primary_drug = random.choice(["VPA", "LEV", "LTG", "ZNS"])
    elif etio == "GOF-GEFS-Spectrum":
        primary_drug = random.choice(["VPA", "ETX", "LEV", "CLB"])
    else:
        primary_drug = random.choice(["ETX", "VPA", "LEV", "ZNS"])
    seizure_free = random.random() < 0.60
    drug_resistant = random.random() < 0.15 if not seizure_free else False
    PATIENTS.append({
        "id": i + 1,
        "name": name,
        "age": age,
        "sex": sex,
        "etiology": etio,
        "onset_age": onset,
        "seizure_types": stypes,
        "primary_drug": primary_drug,
        "seizure_free": seizure_free,
        "drug_resistant": drug_resistant,
        "etx_levels_ok": primary_drug.startswith("ETX") and random.random() < 0.80,
        "vppp_required": sex == "F" and age <= 50 and primary_drug in ("VPA", "ETX+VPA"),
        "polg_screened": primary_drug in ("VPA", "ETX+VPA") and random.random() < 0.90,
        "hv_swd_abolished": primary_drug.startswith("ETX") and seizure_free and random.random() < 0.88,
        "photosensitive": random.random() < 0.38,
        "catamenial": sex == "F" and random.random() < 0.25,
        "sudep_risk": "high" if ("GTCS" in stypes and not seizure_free) else ("moderate" if "GTCS" in stypes else "low"),
    })

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_overview():
    seizure_free_n = sum(1 for p in PATIENTS if p["seizure_free"])
    drug_resistant_n = sum(1 for p in PATIENTS if p["drug_resistant"])
    on_etx_n = sum(1 for p in PATIENTS if "ETX" in p["primary_drug"])
    on_vpa_n = sum(1 for p in PATIENTS if "VPA" in p["primary_drug"])
    on_lev_n = sum(1 for p in PATIENTS if "LEV" in p["primary_drug"])
    gtcs_n = sum(1 for p in PATIENTS if "GTCS" in p["seizure_types"])
    photosensitive_n = sum(1 for p in PATIENTS if p["photosensitive"])
    catamenial_n = sum(1 for p in PATIENTS if p.get("catamenial"))
    sudep_high_n = sum(1 for p in PATIENTS if p["sudep_risk"] == "high")
    vppp_n = sum(1 for p in PATIENTS if p.get("vppp_required"))
    hv_abolished_n = sum(1 for p in PATIENTS if p.get("hv_swd_abolished"))

    return {
        "kpis": {
            "n_patients": 40,
            "gene": "CACNA1H",
            "locus": "16p13.3",
            "protein": "Cav3.2 (T-type Ca²⁺ channel)",
            "omim": "#612899 (EIG6) / #607682 (gene)",
            "inheritance": "AD GOF, reduced penetrance ~65%",
            "pli": 0.25,
            "de_novo_pct": 35,
            "seizure_free_n": seizure_free_n,
            "seizure_free_pct": round(seizure_free_n / 40 * 100),
            "drug_resistant_n": drug_resistant_n,
            "drug_resistant_pct": round(drug_resistant_n / 40 * 100),
            "on_etx_n": on_etx_n,
            "on_vpa_n": on_vpa_n,
            "on_lev_n": on_lev_n,
            "gtcs_n": gtcs_n,
            "photosensitive_n": photosensitive_n,
            "catamenial_n": catamenial_n,
            "sudep_high_risk_n": sudep_high_n,
            "vppp_required_n": vppp_n,
            "hv_swd_abolished_n": hv_abolished_n,
            "avg_age_years": round(sum(p["age"] for p in PATIENTS) / 40, 1),
        },
        "etiology_distribution": [
            {"category": e["category"], "pct": e["pct"], "etiology": e["etiology"]}
            for e in ETIOLOGY_CATALOG
        ],
        "treatments_summary": [
            {"drug": t["drug"], "level": t["level"], "indication": t["indication"]}
            for t in TREATMENTS
        ],
        "monitoring_summary": [
            {"item": m["item"], "frequency": m["frequency"]}
            for m in MONITORING_ITEMS
        ],
        "lifecycle": [
            {"window": lc["window"], "phase": lc["phase"], "key_risk": lc["key_risk"]}
            for lc in LIFECYCLE
        ],
        "thresholds": THRESHOLDS,
        "contraindications_summary": [
            {"drug": c["drug"], "risk": c["risk"]}
            for c in CONTRAINDICATIONS
        ],
        "clinical_alert": (
            "CACNA1H (16p13.3) — Cav3.2 T-type calcium channel. "
            "ABSOLUTE CI: CBZ/OXC/PHT (GGE aggravation → absence status) · Tiagabine (NCSE). "
            "PRECISION: Ethosuximide (T-type Cav3.2 blocker, Level A CAE). "
            "KEY MARKER: HV (3 min) induces SWD in >90% untreated CAE — also ETX efficacy test. "
            "VPA in females: VPPP mandatory (MHRA 2021). POLG1 screen before VPA. "
            "LTG: EEG mandatory before prescribing (myoclonic worsening in JME 15–20%)."
        ),
        "gene_family_note": (
            "CACNA1H extends the voltage-gated Ca²⁺ channel series: "
            "CACNA1A (Cav2.1 P/Q-type, already built) + CACNA1H (Cav3.2 T-type, this dashboard). "
            "Shared contraindication: GGE-aggravating NaV blockers (CBZ/OXC/PHT) apply to both "
            "only where GGE phenotype present (CACNA1A DEE42 + CACNA1H GGE). "
            "Distinct precision: CACNA1A-EA2 → acetazolamide; CACNA1H-CAE → ethosuximide."
        ),
    }


def get_breakdown():
    return {
        "etiologies": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING_ITEMS,
        "lifecycle": LIFECYCLE,
        "patients": PATIENTS,
        "seizure_frequency_distribution": {
            "Typical Absence": sum(1 for p in PATIENTS if "Typical Absence" in p["seizure_types"]),
            "GTCS": sum(1 for p in PATIENTS if "GTCS" in p["seizure_types"]),
            "Myoclonic Jerks": sum(1 for p in PATIENTS if "Myoclonic Jerks" in p["seizure_types"]),
            "Febrile Seizures+": sum(1 for p in PATIENTS if "Febrile Seizures+" in p["seizure_types"]),
            "Absence Status": sum(1 for p in PATIENTS if "Absence Status" in p.get("seizure_types", [])),
        },
        "etx_precision_metrics": {
            "on_etx": sum(1 for p in PATIENTS if "ETX" in p["primary_drug"]),
            "etx_levels_therapeutic": sum(1 for p in PATIENTS if p.get("etx_levels_ok")),
            "hv_swd_abolished_on_etx": sum(1 for p in PATIENTS if p.get("hv_swd_abolished")),
        },
        "trigger_distribution": [
            {"trigger": t["trigger"], "pct": t["pct"]}
            for t in TRIGGERS
        ],
    }


def get_definitions():
    return {
        "definitions": DEFINITIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "gene_summary": {
            "gene": "CACNA1H",
            "locus": "16p13.3",
            "protein": "Cav3.2 (T-type voltage-gated calcium channel α1 subunit)",
            "channel_type": "Low-voltage-activated (LVA), T-type",
            "biophysics": "Activates at −70 to −60 mV; transient (τ ~20–30 ms); 8 pS; window current at −70 to −50 mV",
            "disease_mechanism": "GOF: enlarged window current → enhanced TC neuron LTCS → 3-Hz SWD → GGE",
            "epilepsy_syndromes": ["CAE", "JME", "GEFS+", "GTCS-Alone", "JAE"],
            "precision_drug": "Ethosuximide (ETX) — T-type Cav3.2 blocker, Level A for CAE",
            "absolute_ci": ["CBZ", "OXC", "PHT", "Tiagabine"],
            "pli": 0.25,
            "inheritance": "AD GOF, reduced penetrance ~65%",
            "discovery": "Chen YH et al. 2003 Nat Genet 32(1):76-79",
            "omim_gene": "#607682",
            "omim_epilepsy": "#612899 (EIG6 — susceptibility to idiopathic generalised epilepsy 6)",
        },
    }
