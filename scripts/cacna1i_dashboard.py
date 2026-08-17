"""
CACNA1I Epilepsy — Genetic Generalised Epilepsy (GGE) / CAE / JME / GEFS+
===========================================================================
40-patient cohort · CACNA1I (22q13.1) · Cav3.3 T-type Ca²⁺ Channel · AD GOF · TRN-Dominant

KEY CACNA1I BIOLOGY — THALAMIC RETICULAR NUCLEUS (TRN) DOMINANT T-TYPE CHANNEL:
CACNA1I (22q13.1) encodes Cav3.3 (α1I), the DOMINANT T-type (low-voltage-activated, LVA)
calcium channel in Thalamic Reticular Nucleus (TRN) neurons. This is the critical
distinguishing feature from its siblings:
  · Cav3.1 (CACNA1G, 17q21.33): TC-neuron DOMINANT — drives thalamo-cortical LTCS
  · Cav3.2 (CACNA1H, 16p13.3): TC+TRN balanced expression
  · Cav3.3 (CACNA1I, 22q13.1): TRN-DOMINANT — drives TRN burst firing → GABA-B → TC LTCS

KEY POINTS:

  1. Cav3.3 BIOPHYSICS — SLOWEST T-TYPE INACTIVATION IN TRN:
     Cav3.3 is distinguished within the Cav3 subfamily by:
       (a) HIGHEST activation voltage: −65 to −55 mV (Cav3.1: −80 to −70 mV; Cav3.2: −75
           to −65 mV). Cav3.3 activates at the LEAST negative potential of the T-type
           channels — matching the typical resting/hyperpolarised range of TRN neurons
           (which sit at −70 to −60 mV after TC and cortical input withdrawal).
       (b) SLOWEST inactivation: τ_inactivation ~50–80 ms at −40 mV — compared to
           Cav3.1 (~20–25 ms) and Cav3.2 (~20–30 ms). This slow inactivation allows
           Cav3.3 to sustain a LONGER Ca²⁺ burst in TRN neurons during the rebound
           depolarisation phase → larger Ca²⁺ charge entry → more powerful GABA-B
           IPSP delivered to TC relay neurons.
       (c) LARGEST window current amplitude: the combination of highest activation
           V1/2 and slow inactivation creates the most prominent window current of the
           Cav3 subfamily at −60 to −50 mV — the physiological range of TRN resting
           membrane potential. Cav3.3 GOF window current enlargement → TRN neurons fire
           tonically even at modest depolarisation.
       (d) Monomeric 4-domain α1I subunit: 2031 aa (shorter than Cav3.1 2377 aa and
           Cav3.2 2353 aa). No obligate β-subunit (all T-type channels are monomeric).
       (e) Single-channel conductance: ~11 pS (slightly larger than Cav3.1 9 pS and
           Cav3.2 8 pS) — net result: high per-channel Ca²⁺ flux in TRN.

  2. TRN-CENTRIC MECHANISM — THE CACNA1I UNIQUE STORY:
     Thalamic Reticular Nucleus (TRN) in normal physiology:
       - TRN neurons are GABAergic interneurons forming a shell around thalamic relay nuclei.
       - They receive glutamatergic collaterals from: (a) TC→cortex axons and (b) cortico-
         thalamic axons. Both inputs activate TRN.
       - TRN fires → GABA-B post-synaptic receptors on TC neurons →  prolonged K⁺-mediated
         IPSP → TC membrane reaches −85 to −95 mV → de-inactivation of TC Cav3.1/Cav3.2
         → on rebound: LTCS (low-threshold Ca²⁺ spike) → burst of Na⁺ APs → 3-Hz rhythm.
     CACNA1I GOF in TRN (the CACNA1I-SPECIFIC effect):
       - Cav3.3 GOF (V1/2 inactivation shift −10 to −20 mV leftward) → ENLARGED window
         current at TRN resting potential (−60 to −50 mV) → TRN fires AUTONOMOUSLY with
         less TC/cortical input needed → MORE POWERFUL GABA-B IPSPs to TC neurons →
         DEEPER TC hyperpolarisation → STRONGER Cav3.1/Cav3.2 de-inactivation → LARGER
         TC LTCS → HIGHER 3-Hz SWD probability and amplitude.
       - Key insight: CACNA1I GOF does NOT directly increase TC excitability (unlike
         CACNA1G GOF). Instead, it AMPLIFIES TRN inhibition → STRONGER TC rebound.
         The net effect is the same (more 3-Hz SWD) but the mechanism is INDIRECT via
         enhanced TRN GABA-B drive — distinguishing CACNA1I from CACNA1G epilepsy.

  3. CLINICAL PHENOTYPE — GGE SPECTRUM WITH TRN SIGNATURE:
     CACNA1I GOF → enhanced TRN drive → amplified TC rebound → 3-Hz SWD → GGE spectrum.
     Phenotypes (similar to CACNA1G/CACNA1H but with subtly different EEG amplitude pattern):
       (a) Childhood Absence Epilepsy (CAE): 3-Hz SWD; onset 5–10Y; typical absences.
           TRN-dominant effect: SWD may appear HIGHER AMPLITUDE and more widely distributed
           than CACNA1G CAE (stronger TRN→TC drive synchronises more TC neurons).
       (b) Juvenile Absence Epilepsy (JAE): adolescence onset; GTCS in 70–80%.
       (c) Juvenile Myoclonic Epilepsy (JME): morning myoclonus + GTCS ± absence; ~8% of JME.
       (d) GEFS+ spectrum: febrile seizures plus (CACNA1I found in GEFS+ pedigrees).
       (e) GTCS-Alone: isolated convulsive GGE; generalised polyspike-wave on EEG.

  4. PRECISION TREATMENT — ETHOSUXIMIDE (LEVEL B in CACNA1I):
     ETX mechanism in CACNA1I (different from CACNA1G Level A):
       - ETX blocks Cav3.1 and Cav3.2 in TC neurons → REDUCES TC LTCS rebound.
       - ETX also blocks Cav3.3 in TRN neurons — but less potent (IC50 higher for Cav3.3
         vs Cav3.1/Cav3.2 at clinical concentrations of 40–100 mg/L).
       - Net effect: ETX reduces BOTH the TRN burst (Cav3.3 partial block) AND the TC
         rebound (Cav3.1/Cav3.2 block) → suppresses 3-Hz SWD.
       - Level B (not Level A) for CACNA1I: because the primary GOF target (Cav3.3 in TRN)
         is LESS ETX-sensitive than Cav3.1/Cav3.2 (the primary CACNA1G/CACNA1H targets).
       - Clinical corollary: some CACNA1I patients require VPA + ETX combination or
         VPA alone for complete seizure control.
     VPA: Level B — broad-spectrum (T-type + NaV + GABA-T) → covers full GGE spectrum.

  5. GGE-AGGRAVATING AGENTS (SAME AS ALL GGE):
     Sodium-channel blockers: CBZ / OXC / PHT — ABSOLUTE CONTRAINDICATION.
     Mechanism identical to CACNA1G/CACNA1H: NaV block spares TRN interneurons
     (TRN uses T-type preferentially, not NaV for burst) → TC disinhibition → worsened 3-Hz SWD.

KEY REFERENCES:
  Khosravani H et al. 2004 J Physiol 561(3):873-884 — Cav3.3 biophysics (TRN expression)
  Bhatt DL et al. 2023 Epilepsia 64(S3) — CACNA1I in GGE spectrum
  Bhattacharya A & Bhatt DL 2020 Front Neurol 11:534 — T-type channel DEE review
  Kim D et al. 2001 Nature 410:458 — CACNA1G foundational (T-type Cav3 context)
  Talley EM et al. 1999 J Neurosci 19(6):1895-1911 — Cav3.3 TRN-dominant expression
  ILAE 2022 Operational classification of seizure types Epilepsia 63(6)
"""
import random

random.seed(43)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GOF-CAE-Classic",
        "pct": 38,
        "etiology": "CACNA1I GOF missense — Childhood Absence Epilepsy (CAE) classic phenotype",
        "mechanism": (
            "Gain-of-function missense variants shift Cav3.3 steady-state inactivation to more "
            "negative potentials (−10 to −20 mV leftward shift in V1/2 inactivation). Enlarged "
            "window current at TRN resting potential (−60 to −50 mV) → TRN fires with less input → "
            "stronger GABA-B IPSPs to TC neurons → deeper TC hyperpolarisation → larger Cav3.1/Cav3.2 "
            "de-inactivation → higher-amplitude LTCS rebound → 3-Hz SWD. TRN-mediated amplification "
            "produces slightly higher SWD amplitude compared to pure TC-dominant (CACNA1G) epilepsy. "
            "Onset: 5–10Y; typical absences; EEG: 3-Hz SWD activated by HV (>85% yield)."
        ),
        "typical_variants": "GOF missense — DI-DII linker · DIII S4-S5 · DII voltage sensor region",
        "onset_age_years": 7,
        "outcome": "55–65% remit by adolescence with ETX±VPA; 35% evolve to JAE/JME requiring lifelong AED; GTCS in 25–35%",
    },
    {
        "category": "GOF-JME-Overlap",
        "pct": 28,
        "etiology": "CACNA1I GOF missense — Juvenile Myoclonic Epilepsy (JME) overlap phenotype",
        "mechanism": (
            "CACNA1I GOF in JME pedigrees (~8% of JME cohorts in multi-gene panel studies). Enhanced "
            "TRN Cav3.3 → stronger TC oscillation → morning polyspike-wave bursts on waking → "
            "myoclonic jerks (cortical excitability peak post-waking amplified by thalamo-cortical "
            "desynchronisation from sleep). JME onset 12–18Y; morning myoclonus + GTCS ± absence. "
            "TRN-driven oscillation may explain the pronounced thalamo-cortical synchronisation seen "
            "in JME-CACNA1I vs purely cortical JME phenotypes."
        ),
        "typical_variants": "GOF missense (DII-DIII linker, DIII S4 voltage sensor)",
        "onset_age_years": 14,
        "outcome": "VPA/LEV preferred; 65–75% GTCS-free; lifelong therapy required for JME; myoclonic may persist with sleep deprivation",
    },
    {
        "category": "GOF-GEFS-Plus",
        "pct": 20,
        "etiology": "CACNA1I GOF missense — GEFS+ spectrum (febrile seizures plus evolving to GGE)",
        "mechanism": (
            "GEFS+ (Genetic Epilepsy with Febrile Seizures Plus): CACNA1I GOF in GEFS+ pedigrees. "
            "Enhanced TRN Cav3.3 window current lowers the febrile seizure threshold: fever-induced "
            "depolarisation activates enlarged Cav3.3 window current in TRN → autonomous TRN burst → "
            "GABA-B IPSPs to TC → rebound LTCS → SWD/GTCS at lower temperature threshold. "
            "Febrile seizures continue beyond 6Y (FS+); evolve to absence/GTCS/myoclonic in 35–45%. "
            "Familial: multiple members — some only FS, others full CAE/JME."
        ),
        "typical_variants": "Low-penetrance GOF variants (familial GEFS+ pedigrees)",
        "onset_age_years": 3,
        "outcome": "Variable: FS+ resolves in 55%; GGE subset requires long-term AED; full phenotype spectrum within same family",
    },
    {
        "category": "GOF-GTCS-Alone",
        "pct": 10,
        "etiology": "CACNA1I GOF missense — GTCS-Alone (isolated GGE without recognisable absence)",
        "mechanism": (
            "Isolated GTCS without clear absence or myoclonic history in CACNA1I GOF carriers. "
            "Likely represents: (1) missed absence (brief absences never enquired), or "
            "(2) variants with higher activation-shift than inactivation-shift → TC burst without "
            "classic 3-Hz pattern (atypical polyspike-wave on EEG). Enhanced TRN→TC drive with "
            "threshold above classic 3-Hz SWD → directly recruits GTCS via TC burst overflow to "
            "motor cortex. Management: VPA or LEV; ETX alone insufficient for GTCS; driving counselling "
            "mandatory."
        ),
        "typical_variants": "Atypical GOF (activation shift dominant; less inactivation shift)",
        "onset_age_years": 15,
        "outcome": "60–70% GTCS-free with VPA/LEV; high recurrence on withdrawal; lifelong therapy often required",
    },
    {
        "category": "Phenocopy-GGE-No-CACNA1I",
        "pct": 4,
        "etiology": "GGE phenocopy — CACNA1I VUS or polygenic risk background",
        "mechanism": (
            "CACNA1I variants of uncertain significance (VUS) identified in GGE probands where "
            "polygenic background (multiple GGE susceptibility alleles) rather than monogenic Cav3.3 "
            "GOF explains the phenotype. Functional assay (heterologous expression) required to "
            "confirm GOF (V1/2 inactivation shift ≥8 mV, current density increase ≥25%). Without "
            "functional confirmation, treat as standard GGE; do not presume ETX precision benefit. "
            "Genetics re-review every 18 months as CACNA1I functional data accrues."
        ),
        "typical_variants": "VUS / low-frequency risk alleles (MAF 0.01–0.1% gnomAD)",
        "onset_age_years": 9,
        "outcome": "Standard GGE outcomes; CACNA1I precision treatment (ETX) withheld pending functional confirmation",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES  (5 types)
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Typical Absence",
        "pct": 76,
        "eeg": "Bilateral synchronous 3-Hz SWD; abrupt onset/offset; duration 5–30s; HIGH-AMPLITUDE (TRN-driven) pattern; activated by HV (>85% untreated); IPS in 30–38%",
        "semiology": "Sudden cessation of activity; blank stare; eye flutter; ± peri-oral automatisms; abrupt return to full awareness; NO post-ictal phase (distinguishes from focal impaired awareness seizures). CACNA1I distinguishing feature: SWD may appear higher amplitude and more generalised due to stronger TRN→TC synchronisation.",
        "clinical_tip": "HV 3 minutes in clinic: triggers absence in >85% of untreated CACNA1I-CAE. EEG-HV abolition = ETX/VPA treatment adequacy marker. Note: CACNA1I SWD amplitude tends slightly higher than CACNA1G — reflect stronger TRN-driven synchronisation; useful to document pre-treatment for monitoring.",
    },
    {
        "type": "GTCS (Generalised Tonic-Clonic)",
        "pct": 52,
        "eeg": "Generalised polyspike-wave evolving to fast tonic activity → rhythmic clonic spike-wave; post-ictal diffuse slowing",
        "semiology": "Tonic phase (10–30s): loss of consciousness, axial stiffening, upward eye deviation; clonic phase (30–60s): rhythmic limb jerking; post-ictal confusion 10–30 min. In CACNA1I: often morning-onset (post-waking); driving prohibition mandatory until 12-month seizure-free; document each episode date/time for medicolegal record.",
        "clinical_tip": "Morning GTCS clustering + prior absence history + GGE EEG = CACNA1I/CACNA1G/CACNA1H differential. Check for missed ETX dose or sub-therapeutic level (TDM at next visit). Driving: document risk discussion at every visit.",
    },
    {
        "type": "Myoclonic Jerks",
        "pct": 40,
        "eeg": "Polyspike-wave bursts >3 Hz; brief (<1 s) bilateral high-amplitude polyspike; consciousness preserved during jerk",
        "semiology": "Sudden brief involuntary jerks: arms/shoulders predominant; morning clustering (30–90 min post-waking); objects dropped (toothbrush, cup); preserved awareness; may be subtle (patient attributes to 'clumsiness'). Enquire specifically: 'Do you ever drop things in the morning?'",
        "clinical_tip": "EMG-EEG correlation confirms myoclonus (vs. functional movement or tremor). Morning myoclonus + 3-Hz SWD on EEG = JME phenotype spectrum. If LTG prescribed for JME-CACNA1I: myoclonic WORSENING is a warning sign — EEG mandatory before LTG to rule out polyspike-wave dominance.",
    },
    {
        "type": "Febrile Seizures Plus (FS+)",
        "pct": 35,
        "eeg": "Febrile context: often normal or mildly diffusely slow post-ictally; afebrile EEG: 3-Hz SWD if GGE component present",
        "semiology": "Febrile seizures continuing beyond age 6Y without alternative cause. GEFS+ spectrum marker: family members may have only FS while proband has full CAE/JME. Evolution to afebrile GGE in 35–45% of FS+ CACNA1I carriers. Fever counselling: prescribe rescue protocol (midazolam buccal 0.2 mg/kg or CLB oral) for fever >38.5°C.",
        "clinical_tip": "GEFS+ family history + GGE phenotype in proband: gene panel (CACNA1I, SCN1A, SCN1B, GABRG2, GABRD, CACNA1H). CACNA1I penetrance in GEFS+ families ~55% — counsel about variable expressivity within same family.",
    },
    {
        "type": "Absence Status Epilepticus (ASE)",
        "pct": 10,
        "eeg": "Prolonged continuous or near-continuous 3-Hz SWD (>30 min); amplitude fluctuates; no post-ictal slowing; may have waxing-waning quality",
        "semiology": "Prolonged twilight state: confused, slowed, imperfect responses; NOT unconscious; may walk/speak incoherently; lasts hours if untreated. Precipitants: AED withdrawal, fever, missed dose, OR — critically — inadvertent GGE-aggravating drug (CBZ/OXC/PHT/TGB).",
        "clinical_tip": "ACUTE: IV lorazepam 0.1 mg/kg → aborts most ASE. If ASE triggered by recently started CBZ/PHT/OXC: WITHDRAW immediately + add IV VPA (20–30 mg/kg load at 3–6 mg/kg/min). Emergency EEG is mandatory to confirm ASE and guide termination endpoint.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS  (8 triggers)
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Sleep deprivation", "pct": 92,
     "note": "Most potent CACNA1I trigger. Reduced slow-wave sleep → reduced thalamic adenosine (A1-receptor TRN inhibition) → CACNA1I GOF Cav3.3 operates without adenosine suppression → autonomous TRN burst → 3-Hz SWD. Sleep hygiene prescriptions mandatory: ≥8h/night; regular schedule; avoid napping shifts. CACNA1I patients often cannot tolerate night-shift work."},
    {"trigger": "Missed AED dose", "pct": 82,
     "note": "ETX half-life ~40h adults / ~30h children → trough below 40 mg/L → 3-Hz SWD re-emerges. VPA trough equally important (50–100 mg/L therapeutic). Weekly pill organisers + smartphone alarms standard of care. TDM at next visit after any missed dose cluster."},
    {"trigger": "Alcohol", "pct": 68,
     "note": "Alcohol withdrawal (12–24h after heavy intake) → GABA-A downregulation rebound → net excitatory shift → GTCS/myoclonic surge. Saturday-morning GTCS after Friday drinking = classic CACNA1I JME pattern. AUDIT-C screen at every visit; counsel specifically about alcohol-seizure risk."},
    {"trigger": "Hyperventilation (HV)", "pct": 85,
     "note": "Voluntary HV (exercise, anxiety) → CO₂ washout → respiratory alkalosis → free [Ca²⁺] rises → CACNA1I Cav3.3 window current enhanced → TRN fires more readily → 3-Hz SWD. Clinical HV test (3 min) triggers absence in >85% untreated CACNA1I-CAE. ETX/VPA should abolish HV-induced SWD — monitor as treatment adequacy marker each visit."},
    {"trigger": "Fever / intercurrent illness", "pct": 55,
     "note": "CACNA1I GEFS+ component: fever lowers TRN Cav3.3 threshold (depolarisation shifts activation into GOF window current range). Prescribe antipyretics (paracetamol/ibuprofen) with clear instructions. Rescue protocol: midazolam buccal 0.2 mg/kg or CLB 10mg oral if GTCS begins during febrile episode."},
    {"trigger": "Stress", "pct": 62,
     "note": "Psychosocial stress → cortisol → cortical excitability increase → amplifies TRN-TC loop oscillation. Exam periods and acute life events cluster with breakthrough absence/GTCS. Refer to clinical psychologist for seizure stress management. Biofeedback has early evidence for GGE trigger reduction."},
    {"trigger": "Photostimulation (IPS)", "pct": 32,
     "note": "Intermittent photic stimulation → TRN visual-cortex loop → 3-Hz SWD in 30–35% CACNA1I-GGE. Advise: flickering lights (screens, disco, sunlight through trees) may trigger absences. ETX reduces photo-sensitivity in 60–70% of PS-positive patients."},
    {"trigger": "Catamenial (perimenstrual)", "pct": 25,
     "note": "Allopregnanolone (ALLO) withdrawal at menstruation → reduced GABA-A modulation → net excitatory shift → myoclonic surge/absence clustering. Duncan C1 pattern (days 25–3 of cycle). CLB cycle-adjusted (2.5–10 mg days 20–28) is first-line catamenial strategy for CACNA1I-GGE females. Document catamenial pattern in seizure diary."},
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS  (8 treatments)
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Ethosuximide (ETX)",
        "level": "Level B",
        "indication": "CAE-dominant CACNA1I GOF (absence-predominant; less ETX-sensitive than CACNA1G/CACNA1H due to Cav3.3 TRN-dominance)",
        "dose": "20–40 mg/kg/day; start 250 mg/day (children 5–6Y: 125 mg/day), increase by 250 mg/week; TDM target 40–100 mg/L; upper-range (80–100 mg/L) before declaring failure",
        "moa": "T-type Ca²⁺ channel blocker: primary targets Cav3.1 and Cav3.2 (TC neurons) → reduces TC LTCS rebound. Also blocks Cav3.3 in TRN — but with lower potency at clinical concentrations (IC50 for Cav3.3 ~30% higher than for Cav3.1). Net result: ETX reduces both TRN burst (partial Cav3.3 block) AND TC rebound (Cav3.1/Cav3.2 block). LEVEL B for CACNA1I (vs Level A for CACNA1G/CACNA1H): TRN Cav3.3 is the PRIMARY GOF target but is LESS ETX-sensitive → ETX achieves partial but potentially incomplete suppression.",
        "efficacy": "55–65% absence freedom in CACNA1I-CAE at therapeutic levels (vs 70–75% for CACNA1G/CACNA1H at same levels). ETX + VPA combination often needed for complete control. HV abolition remains a useful ETX efficacy marker (3-min HV in clinic should not trigger SWD at therapeutic ETX).",
        "safety": "GI: nausea/vomiting (take with food), abdominal pain; CNS: hiccups, headache, dizziness (usually transient); rare: aplastic anaemia (FBC q6M); SJS/TEN rare (HLA-B*1502 Asian ancestry); psychiatric behavioral change (rare). Advantage: no hepatotoxicity, no teratogenicity vs VPA.",
        "monitoring": "TDM q6M (target 40–100 mg/L; upper range 80–100 mg/L for CACNA1I incomplete response). FBC q6M. HV clinic test every 6M as efficacy proxy.",
        "cacna1i_note": "CACNA1I-SPECIFIC: ETX is Level B (not Level A) because Cav3.3 in TRN is the primary GOF target and Cav3.3 is LESS ETX-sensitive than Cav3.1/Cav3.2. Dose to therapeutic upper range (80–100 mg/L) before adding or switching to VPA. ETX + VPA combination is frequently required for complete GGE spectrum control.",
    },
    {
        "drug": "Valproate / Valproic Acid (VPA)",
        "level": "Level B — VPPP mandatory in females; POLG1 mandatory",
        "indication": "Full-spectrum CACNA1I GGE (absence + GTCS + myoclonic); first-line if GTCS or myoclonic component present; preferred males/post-reproductive females",
        "dose": "20–30 mg/kg/day; start 300 mg/day, increase by 300 mg/week; TDM 50–100 mg/L",
        "moa": "Broad-spectrum triple mechanism: (1) T-type Ca²⁺ block (Cav3.1/Cav3.2/Cav3.3 — broader than ETX at therapeutic concentrations); (2) use-dependent NaV block; (3) GABA-T inhibition → increased synaptic GABA. VPA blocks Cav3.3 more potently than ETX at therapeutic serum concentrations — makes VPA the preferred CACNA1I agent when ETX alone is insufficient.",
        "efficacy": "70–80% full GGE control in CACNA1I absence+GTCS combination; VPA alone or ETX+VPA achieves better control than ETX alone for CACNA1I TRN-dominant phenotype.",
        "safety": "VPPP (teratogenicity: MHRA 2021 — ABSOLUTE CI in pregnancy without PREVENT programme oversight; females 9–50Y); hepatotoxicity (POLG mutation: ABSOLUTE CI); weight gain; PCOS; hair loss; hyperammonaemia. POLG1 screen MANDATORY before prescribing.",
        "monitoring": "POLG1 before prescribing. VPA TDM q3M. LFT/FBC/ammonia q3M. VPPP annual documentation females. Fasting glucose q6M.",
        "cacna1i_note": "VPA blocks Cav3.3 with higher potency than ETX at therapeutic levels — mechanistic rationale for VPA superiority in CACNA1I vs CACNA1G (where ETX alone often suffices as Level A). Consider ETX+VPA combination for CACNA1I absence-dominant phenotype; use VPA alone or with LTG for GTCS/myoclonic-dominant.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B",
        "indication": "GGE (JME/GTCS/GEFS+); preferred in CACNA1I females of childbearing age; POLG1-safe",
        "dose": "20–40 mg/kg/day; start 250 mg bid, increase by 500 mg every 2 weeks; no TDM required",
        "moa": "SV2A modulator → reduces Ca²⁺-triggered glutamate and GABA release from presynaptic terminals. Inhibits N-type (Cav2.2) indirectly. Does NOT directly block T-type (Cav3.3) — not precision for CACNA1I GOF. Broad anti-seizure effect including thalamo-cortical circuits via SV2A in TC and TRN neurons.",
        "efficacy": "Level B for CACNA1I absence (SANAD II: LEV worst for pure absence among ETX/VPA/LEV). Effective for GTCS and myoclonic component. Preferred in CACNA1I-JME females of childbearing age (no teratogenicity, no VPPP).",
        "safety": "Behavioral: irritability/aggression 10–15% (worst in adolescents and psychiatric comorbidity). B6 supplementation (pyridoxine 50–100 mg/day) may reduce behavioral SE. No organ toxicity. Safe in pregnancy.",
        "monitoring": "No TDM mandatory. FBC at 6M. Renal function q12M (renally cleared). Behavioral diary at every visit.",
        "cacna1i_note": "LEV is preferred over VPA in CACNA1I females of childbearing age. Not preferred for pure CACNA1I CAE — ETX or ETX+VPA superior for absence. Best for CACNA1I-JME-overlap or GTCS-dominant phenotype.",
    },
    {
        "drug": "Lamotrigine (LTG)",
        "level": "Level B — CAUTION if myoclonic; EEG MANDATORY before prescribing",
        "indication": "Absence + GTCS combination in CACNA1I GGE; ETX add-on for GTCS; AVOID if myoclonic jerks dominant",
        "dose": "Start 25 mg/day (no VPA co-medication); 8-week slow titration to 100–200 mg/day. If VPA co-medication: start 12.5 mg/day (VPA inhibits LTG glucuronidation → doubles LTG levels)",
        "moa": "Use-dependent NaV block (NaV1.1, NaV1.2) + glutamate release reduction. Does NOT block T-type Cav3.3 — not precision for CACNA1I. Useful as ETX add-on for GTCS component.",
        "efficacy": "Level B for absence (SANAD: LTG inferior to ETX/VPA for pure absence). Good for GTCS. ETX+LTG: additive for absence+GTCS combination in CACNA1I.",
        "safety": "SJS/TEN: HLA-B*1502 screen (South/East Asian ancestry) mandatory before prescribing. Rash 10–15% (any rash → hold LTG). VPA interaction: halve LTG dose. CRITICAL: myoclonic aggravation in 15–20% of JME (NaV1.1 block in PV+ interneurons → disinhibition). EEG MANDATORY before LTG in CACNA1I to rule out polyspike-wave dominance.",
        "monitoring": "HLA-B*1502 before prescribing in Asian ancestry. LFT at baseline. LTG levels if seizure breakthrough (though not routinely). EEG before prescribing (rule out myoclonic component).",
        "cacna1i_note": "EEG IS MANDATORY before LTG in CACNA1I-GGE: if polyspike-wave bursts or clinical myoclonus present → LTG may WORSEN myoclonic component (JME aggravation). Use LTG only if EEG shows absence/GTCS without dominant polyspike-wave and patient confirms no morning myoclonus.",
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B",
        "indication": "Adjunct for CACNA1I-GGE (nocturnal GTCS, catamenial pattern); rescue protocol",
        "dose": "5–20 mg/day (adults); catamenial: 5–10 mg days 20–28 of cycle; rescue: 10–20 mg single oral dose",
        "moa": "1,5-benzodiazepine: positive allosteric modulator at GABA-A receptors (α1, α2, α3-containing — NOT α5 subunit, unlike full BZDs). Less sedating than classical 1,4-BZDs. Reduces thalamo-cortical oscillation via enhanced GABA-A inhibition in TC and TRN neurons.",
        "efficacy": "Level B adjunct for refractory CACNA1I-GGE. Particularly effective for catamenial pattern (cycle-adjusted dosing). Nocturnal GTCS reduction in 50–60% of patients as add-on.",
        "safety": "Sedation; tolerance develops over 6–12 weeks if used daily continuously. Withdraw gradually to avoid CLB withdrawal seizures. Cognitive effects milder than 1,4-BZDs. Interactions: VPA and CLB → increased N-desmethyl CLB (active metabolite) levels.",
        "monitoring": "Reassess need for continuous CLB every 6M (tolerance). Catamenial use: monthly diary. Behavioral assessment (disinhibition in some patients).",
        "cacna1i_note": "CLB is particularly useful in CACNA1I-GEFS+ (catamenial pattern) and for nocturnal GTCS (bed-time CLB 10 mg). Cycle-adjusted dosing (5–10 mg days 20–28) is an evidence-based catamenial strategy for CACNA1I females.",
    },
    {
        "drug": "Zonisamide (ZNS)",
        "level": "Level C",
        "indication": "CACNA1I-GGE: alternative/add-on to ETX (T-type + NaV dual block); ETX-intolerant patients",
        "dose": "4–8 mg/kg/day; start 50 mg/day, increase by 50 mg every 2 weeks to 200–400 mg/day",
        "moa": "DUAL mechanism: (1) T-type Ca²⁺ block (Cav3.1/Cav3.2/Cav3.3) + (2) use-dependent NaV block. The T-type component directly targets CACNA1I GOF Cav3.3; the NaV component less relevant for GGE (cf. ETX has no NaV activity). ZNS level C for absence (Japan: Level A; outside Japan: B–C).",
        "efficacy": "Level C globally (Level A Japan). Small trials show ZNS reduces absence in 40–55% as add-on. May be preferred over pure NaV blockers (CBZ/PHT) as partial T-type activity reduces GGE aggravation risk.",
        "safety": "Metabolic acidosis (carbonic anhydrase inhibition → bicarbonate loss — monitor HCO₃⁻ q6M; avoid with acetazolamide co-medication). Nephrolithiasis (hydration 2L/day). Weight loss. Cognitive effects (memory, word-finding).",
        "monitoring": "Serum bicarbonate q6M. Renal function q12M. Urinalysis (crystals). Weight and BMI q3M.",
        "cacna1i_note": "ZNS is an alternative to ETX in CACNA1I-GGE when ETX not tolerated (GI side effects). ZNS blocks Cav3.3 with comparable potency to ETX but adds NaV component. Do NOT use ZNS as sole agent for GTCS prevention in CACNA1I-GGE.",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B",
        "indication": "Drug-resistant CACNA1I-GGE (≥2 AED failures); particularly valuable for GTCS-dominant refractory phenotype",
        "dose": "4:1 ratio (fat:protein+carbohydrate) classic KD or MAD (modified Atkins); dietitian-supervised; serum β-hydroxybutyrate target 2–5 mmol/L",
        "moa": "Multi-mechanism: (1) metabolic shift → ketosis → GABA-A upregulation; (2) BHB directly inhibits vesicular glutamate release; (3) HCN channel modulation → reduces thalamo-cortical rhythmicity (relevant for CACNA1I TRN-mediated 3-Hz oscillation). KD specifically reduces thalamo-cortical synchronisation — mechanistically relevant for CACNA1I GOF.",
        "efficacy": "Level B for GGE DRE: 50% seizure reduction in ~50% of patients; 10–15% seizure-free. CACNA1I TRN-mediated SWD: KD may reduce TRN burst synchronisation via HCN modulation.",
        "safety": "GI intolerance (nausea, constipation, reflux); hyperlipidaemia; growth restriction in children (monitor growth); acidosis; nephrolithiasis; long-term: cardiomyopathy monitoring (lipid panel q6M, ECHO q12M if dyslipidaemia).",
        "monitoring": "BHB ketones daily (urine sticks or blood meter); lipid panel q6M; renal function q6M; growth in children (height/weight q3M); ECHO q12M if dyslipidaemia.",
        "cacna1i_note": "KD is HIGH PRIORITY for drug-resistant CACNA1I-GGE — do not delay as 3rd-line after ≥2 AED failures. KD reduces thalamo-cortical oscillation (mechanistically relevant for CACNA1I TRN-dominant channelopathy). Consider KD before adding 3rd or 4th AED.",
    },
    {
        "drug": "Phenobarbital (PB)",
        "level": "Level C",
        "indication": "Acute rescue for refractory CACNA1I absence status epilepticus; 3rd-line GGE maintenance",
        "dose": "Acute SE: 20 mg/kg IV loading; maintenance 2–4 mg/kg/day oral",
        "moa": "Barbiturate: positive allosteric modulator at GABA-A α1β2γ2 → prolonged Cl⁻ channel opening → inhibition. Also moderate T-type block at higher concentrations. Central GABA enhancement reduces thalamo-cortical oscillation.",
        "efficacy": "Effective for acute ASE. Level C for chronic GGE maintenance (high sedation, cognitive effects). Used as bridge therapy in acute ASE until VPA/ETX therapeutic.",
        "safety": "Sedation, cognitive impairment, tolerance/dependence (abrupt withdrawal → seizures). Long-term: Dupuytren's, metabolic bone disease (monitor vitamin D, DEXA). CYP inducer (multiple drug interactions). Teratogenic.",
        "monitoring": "PB serum levels (therapeutic 15–40 mg/L). Cognitive assessment q6M. Vitamin D + DEXA q12M (chronic use). Liver function q6M. Avoid abrupt withdrawal.",
        "cacna1i_note": "PB is acute/bridge therapy for CACNA1I-ASE, not chronic maintenance. If used for acute ASE: transition to IV VPA load as primary treatment and wean PB. Avoid long-term PB unless all other options failed.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS  (5 entries)
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT)",
        "risk": "ABSOLUTE CONTRAINDICATION — GGE aggravation → absence status epilepticus",
        "mechanism": "NaV block in cortical pyramidal neurons (NaV1.1, NaV1.2) relatively spares TRN interneurons → NET DISINHIBITION of TC relay neurons → ENHANCED 3-Hz burst firing → WORSENED SWD → absence aggravation or absence status epilepticus. CACNA1I TRN-dominant phenotype: TRN disinhibition by NaV blockers may be particularly catastrophic given already-enhanced TRN→TC drive from Cav3.3 GOF.",
        "action": "NEVER prescribe in CACNA1I-GGE. If inadvertently prescribed and seizures worsen: STOP IMMEDIATELY, start IV lorazepam + IV VPA. Document allergy/CI in medical record.",
    },
    {
        "drug": "Tiagabine (TGB)",
        "risk": "ABSOLUTE CONTRAINDICATION — non-convulsive status epilepticus (NCSE)",
        "mechanism": "GAT-1 inhibitor → increases ambient GABA → PARADOXICAL thalamo-cortical desynchronisation followed by 3-Hz SWD burst (tonic GABA-A activation → thalamic pacemaker initiation). Can precipitate absence status epilepticus lasting hours in GGE patients.",
        "action": "NEVER prescribe in CACNA1I-GGE. If TGB-induced NCSE: STOP TGB immediately, IV lorazepam, EEG monitoring, IV VPA. No rechallenge.",
    },
    {
        "drug": "Valproate + POLG1 mutation (Alpers-Huttenlocher)",
        "risk": "ABSOLUTE CONTRAINDICATION — fatal hepatotoxicity / Alpers syndrome",
        "mechanism": "POLG encodes mitochondrial DNA polymerase γ. VPA inhibits mitochondrial β-oxidation → oxidative stress in POLG-deficient neurons → irreversible hepatocellular necrosis → fulminant liver failure. POLG mutations (p.A467T, p.W748S) cause Alpers-Huttenlocher syndrome (DEE + liver failure). VPA in POLG patients carries 10–20% risk of fatal liver failure.",
        "action": "POLG1 gene test MANDATORY before VPA in any patient with: (1) refractory early-onset epilepsy, (2) cognitive regression, (3) liver disease, (4) family history of mitochondrial disease. If POLG mutation confirmed: VPA is ABSOLUTELY CONTRAINDICATED — use LEV or CLB.",
    },
    {
        "drug": "Lamotrigine (LTG) — HIGH CAUTION if myoclonic",
        "risk": "HIGH CAUTION — myoclonic aggravation in JME/CACNA1I-JME phenotype",
        "mechanism": "LTG NaV1.1 block in PV+ fast-spiking interneurons (same cells that suppress cortical excitability) → DISINHIBITION of cortical and thalamic circuits → ENHANCED polyspike-wave burst → WORSENED myoclonic jerks. 15–20% of JME patients have myoclonic worsening with LTG. In CACNA1I-JME: TRN GOF already drives enhanced bursting → LTG-induced disinhibition further amplifies the burst.",
        "action": "EEG MANDATORY before LTG in CACNA1I: if polyspike-wave dominant or clinical morning myoclonus confirmed → DO NOT prescribe LTG. If LTG started and myoclonus worsens: WITHDRAW LTG and switch to LEV or VPA.",
    },
    {
        "drug": "Valproate — VPPP mandatory in females 9–55Y",
        "risk": "HIGH RISK — teratogenicity (MHRA 2021 Black Box); VPPP valproate pregnancy prevention programme",
        "mechanism": "VPA is the most teratogenic major AED: neural tube defects 1–2%, congenital malformations 10%, fetal valproate syndrome (cognitive impairment, autism, dysmorphia). Risks dose-dependent and especially high in first trimester.",
        "action": "VPA in females 9–55Y: VPPP (Valproate Pregnancy Prevention Programme) MANDATORY before prescribing and annually. VPPP requires: (1) GP+specialist co-prescription; (2) confirmed 2 methods of contraception; (3) annual VPPP form; (4) pregnancy test before each prescription. If pregnancy planned: transition to LEV or LTG under neurologist supervision ≥3 months before conception.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING  (14 items)
# ─────────────────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG1 genetic test before VPA", "frequency": "Once (mandatory before first VPA prescription)", "rationale": "Fatal Alpers hepatotoxicity in POLG mutation carriers. Screen all patients before VPA regardless of age."},
    {"item": "ETX therapeutic drug monitoring (TDM)", "frequency": "Every 6 months (target 40–100 mg/L; upper range 80–100 for CACNA1I)", "rationale": "CACNA1I ETX Level B: dose to therapeutic upper range (80–100 mg/L) before declaring failure. Trough level most informative."},
    {"item": "HV-SWD clinic test (ETX/VPA adequacy)", "frequency": "Every 6 months", "rationale": "3-min voluntary HV: should NOT induce absence at therapeutic ETX. HV-induced SWD = under-treated or sub-therapeutic. CACNA1I-specific: document SWD amplitude pre/post treatment (TRN-driven amplitude tends to fall with treatment)."},
    {"item": "VPA therapeutic drug monitoring", "frequency": "Every 3 months (target 50–100 mg/L)", "rationale": "VPA is the key broad-spectrum agent for CACNA1I-GGE. Maintain therapeutic range; dose adjust for breakthrough seizures or toxicity symptoms."},
    {"item": "LFT + FBC + ammonia", "frequency": "Every 3 months on VPA", "rationale": "VPA hepatotoxicity and hyperammonaemia monitoring. Any elevation > 3× ULN: consider VPA dose reduction or switch."},
    {"item": "LTG EEG before prescribing (myoclonic screen)", "frequency": "Once before LTG (mandatory); EEG at 6M if myoclonus suspected", "rationale": "LTG worsens myoclonic jerks in 15–20% of JME. EEG must confirm absence of dominant polyspike-wave/myoclonic pattern before LTG prescription in CACNA1I."},
    {"item": "EEG baseline + annual (treatment monitoring)", "frequency": "At diagnosis; every 12 months or after medication change", "rationale": "Track 3-Hz SWD burden, SWD amplitude (TRN-driven), HV-response. Document pre-treatment amplitude as CACNA1I-specific marker."},
    {"item": "Cognitive assessment (neuropsychology)", "frequency": "Every 6 months", "rationale": "Absence seizures impair attention/memory during episodes (ongoing micro-absences = school/work impact). Track with formal testing (WPPSI/WISC children; WAIS adults; BRIEF executive function)."},
    {"item": "MRI brain (baseline)", "frequency": "Once at diagnosis (structural epilepsy excluded)", "rationale": "Exclude structural cause. CACNA1I-GGE: MRI typically normal — TRN-dominant channelopathy leaves no structural mark on MRI."},
    {"item": "VPPP annual form (females 9–55Y on VPA)", "frequency": "Annually", "rationale": "MHRA 2021: mandatory VPPP documentation before each VPA prescription in females of childbearing potential. Co-signed by GP+specialist."},
    {"item": "SUDEP annual risk assessment", "frequency": "Annually", "rationale": "CACNA1I-GGE with uncontrolled GTCS + nocturnal seizures = elevated SUDEP risk. Document: nocturnal supervision plan, wearable seizure alert, rescue protocol."},
    {"item": "Photosensitivity (IPS) testing", "frequency": "At diagnosis; repeat if photosensitive triggers reported", "rationale": "IPS in 30–35% of CACNA1I-GGE. Document: maximal photosensitive frequency range; response to ETX (should reduce PS). Advise screen filters/blue-light blocking if PS-positive."},
    {"item": "Catamenial diary (females)", "frequency": "Monthly (ongoing)", "rationale": "Document seizure clustering vs menstrual cycle. Catamenial pattern = indication for cycle-adjusted CLB (days 20–28). Track response to catamenial strategy."},
    {"item": "Genetic counselling", "frequency": "At diagnosis; before pregnancy", "rationale": "AD GOF with de novo ~30–40% and familial penetrance ~55%. Counsel: risk to offspring ~50% (if parent affected); variable expressivity in family members (from FS only to full JME); prenatal/preimplantation genetic testing options available."},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE WINDOWS  (6 windows)
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {"window": "Childhood CAE (5–12Y)", "key_issues": "Typical absence onset; school performance impact; ETX initiation; HV testing; avoid GGE-aggravating AEDs. Absence awareness education for teachers (school seizure action plan)."},
    {"window": "GEFS+ Infancy (0–5Y)", "key_issues": "Febrile seizures plus (continuing beyond 6Y); fever counselling; rescue protocol (midazolam buccal); family genetic testing for GEFS+ siblings."},
    {"window": "Adolescence/JME Onset (12–25Y)", "key_issues": "JME phenotype emergence (morning myoclonus + GTCS); alcohol counselling; sleep hygiene; driving (seizure-free period required); VPA vs LEV in females (VPPP discussion); school/university transition plan."},
    {"window": "Female Reproductive Years", "key_issues": "VPPP mandatory if VPA; ETX preferred for pure CAE (no VPPP, no teratogenicity); LEV preferred for JME/GTCS in childbearing women; catamenial management (CLB cycle-adjusted); pregnancy planning ≥3 months advance neurologist consultation."},
    {"window": "Seizure-Free Goal (12M)", "key_issues": "Target: ≥12 months seizure-free before considering driving. Annual SUDEP risk assessment. If seizure-free ≥3 years on stable AED: AED tapering discussion (CACNA1I-GGE: remission in 55–65% CAE; JME rarely remits — lifelong therapy usually needed for JME)."},
    {"window": "Adult Chronic GGE (>25Y)", "key_issues": "Lifelong therapy for JME-CACNA1I; seizure diary; employment (avoid sleep deprivation, alcohol, shift work); annual review; SUDEP counselling; advance directives."},
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"name": "ETX therapeutic range (CACNA1I)", "value": "40–100 mg/L (target upper range 80–100 mg/L before declaring failure in CACNA1I)"},
    {"name": "VPA therapeutic range", "value": "50–100 mg/L (TDM q3M)"},
    {"name": "HV-SWD abolition (ETX adequacy)", "value": "3-min HV should NOT induce absence at therapeutic ETX; if SWD → sub-therapeutic (check level)"},
    {"name": "LFT/FBC toxicity threshold (VPA)", "value": "ALT/AST >3× ULN: reduce VPA dose or switch; >10× ULN: STOP VPA immediately"},
    {"name": "Seizure-free period for driving", "value": "≥12 months seizure-free (jurisdiction-dependent); document risk discussion at every visit"},
    {"name": "SUDEP high-risk threshold", "value": "Uncontrolled GTCS ≥3/year + nocturnal seizures + drug-resistant epilepsy"},
    {"name": "Catamenial trigger (CLB cycle)", "value": "≥2× seizure rate days 20–3 of cycle → CLB 5–10 mg/day days 20–28"},
    {"name": "VPPP VPA documentation (females)", "value": "Annual VPPP form mandatory; GP + specialist co-signature; 2 contraception methods confirmed"},
    {"name": "POLG1 pre-VPA screen", "value": "Mandatory: any refractory early-onset epilepsy, cognitive regression, liver disease, family history mitochondrial disease"},
    {"name": "KD ketosis target", "value": "Serum BHB 2–5 mmol/L; urine ketones 4+ daily"},
    {"name": "HCO₃⁻ monitoring on ZNS", "value": "Serum bicarbonate q6M; threshold: <18 mmol/L → dose reduction or switch"},
    {"name": "MRI structural exclusion", "value": "MRI at diagnosis (CACNA1I-GGE: typically normal; structural abnormality triggers DRE workup)"},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"name": "ILAE 2022", "description": "Operational classification of seizure types and epilepsy syndromes (Epilepsia 63[6])"},
    {"name": "NICE NG217 (2022)", "description": "Epilepsies in children, young people, and adults — guideline for AED choice in GGE"},
    {"name": "SANAD 2007 (Lancet)", "description": "ETX vs VPA in newly diagnosed absence — Level A; ETX non-inferior VPA for absence"},
    {"name": "SANAD II 2021 (NEJM)", "description": "ETX vs VPA vs LEV in GGE — ETX superior for absence; LEV worst for absence"},
    {"name": "Bhatt 2023 (Epilepsia)", "description": "Gene-epilepsy phenotype catalogue: CACNA1I in GGE spectrum; T-type subfamily comparison"},
    {"name": "Talley 1999 (J Neurosci)", "description": "Cav3.3 TRN-dominant expression — definitive expression atlas (J Neurosci 19[6]:1895-1911)"},
    {"name": "Khosravani 2004 (J Physiol)", "description": "Cav3.3 biophysics: TRN expression; slow inactivation; window current (J Physiol 561[3]:873-884)"},
    {"name": "CPIC POLG Guidelines 2023", "description": "Clinical Pharmacogenomics Implementation Consortium: POLG mutation → VPA absolute CI"},
    {"name": "MHRA VPPP 2021", "description": "UK Medicines and Healthcare Regulatory Agency: valproate pregnancy prevention programme (mandatory)"},
    {"name": "ACMG-AMP 2015", "description": "Variant classification framework: pathogenic/likely pathogenic/VUS/likely benign/benign (Genet Med 17[5]:405)"},
    {"name": "ILAE Dietary Therapies 2018", "description": "Expert consensus: KD, MAD, LGIT for drug-resistant epilepsy (Epilepsia 59[8]:1646-1659)"},
    {"name": "WHO ICF 2019", "description": "International Classification of Functioning, Disability and Health — outcomes framework for GGE"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    "Talley EM et al. J Neurosci 1999;19(6):1895-1911 — Cav3.3 TRN-dominant expression atlas",
    "Khosravani H et al. J Physiol 2004;561(3):873-884 — Cav3.3 biophysics and TRN expression",
    "Bhatt DL et al. Epilepsia 2023;64(S3) — CACNA1I in GGE; T-type Cav3 subfamily comparison",
    "Bhattacharya A & Bhatt DL. Front Neurol 2020;11:534 — T-type channelopathy treatment framework",
    "ILAE 2022 Operational classification of seizure types and epilepsy syndromes. Epilepsia 63(6)",
    "NICE NG217 (2022) — AED selection framework for GGE; ETX and VPA recommendations",
]

# ─────────────────────────────────────────────────────────────────────────────
# DEFINITIONS (15 key concepts)
# ─────────────────────────────────────────────────────────────────────────────
DEFINITIONS = [
    {"term": "CACNA1I (22q13.1)", "definition": "Calcium Voltage-Gated Channel Subunit Alpha1 I; encodes Cav3.3, the dominant T-type Ca²⁺ channel in TRN neurons; 2031 aa; 4-domain monomeric α1I subunit; AD GOF → GGE spectrum."},
    {"term": "Cav3.3 (α1I) — T-type LVA", "definition": "Third member of Cav3 (T-type) subfamily: Cav3.1/CACNA1G (TC-dominant) · Cav3.2/CACNA1H (TC+TRN) · Cav3.3/CACNA1I (TRN-dominant). LVA = Low-Voltage-Activated (activates at −65 to −55 mV). Transient current (T = transient). Monomeric (no obligate β-subunit)."},
    {"term": "TRN-Dominant Expression", "definition": "Cav3.3 is the PRIMARY T-type channel in Thalamic Reticular Nucleus (TRN) neurons. CACNA1I GOF → enhanced TRN bursting → stronger GABA-B IPSPs to TC neurons → deeper TC hyperpolarisation → larger Cav3.1/Cav3.2 de-inactivation → more powerful LTCS → 3-Hz SWD. TRN-indirect mechanism distinguishes CACNA1I from CACNA1G (TC-direct)."},
    {"term": "Slow Inactivation (Cav3.3 unique feature)", "definition": "τ_inactivation ~50–80 ms for Cav3.3 (Cav3.1: ~20–25 ms; Cav3.2: ~20–30 ms). Slowest inactivation in the Cav3 subfamily. Physiological consequence: TRN neurons sustain longer Cav3.3 Ca²⁺ bursts → more Ca²⁺ charge → more powerful GABA-B IPSP to TC. CACNA1I GOF further slows inactivation → even longer TRN bursts."},
    {"term": "Window Current (Cav3.3)", "definition": "Overlap of steady-state activation and inactivation curves at −60 to −50 mV — the TRN resting membrane potential range. LARGEST window current of Cav3 subfamily. CACNA1I GOF enlarges window current (leftward V1/2 inactivation shift) → TRN fires tonically at rest → autonomous GABA-B drive → TC rebound."},
    {"term": "LTCS (Low-Threshold Ca²⁺ Spike)", "definition": "Rebound depolarisation in TC neurons (Cav3.1/Cav3.2 de-inactivation) triggering a burst of Na⁺ action potentials. The LTCS is generated in TC neurons, not TRN. CACNA1I GOF AMPLIFIES the TC LTCS INDIRECTLY: enhanced TRN→GABA-B→deeper TC hyperpolarisation → more Cav3.1/Cav3.2 de-inactivation → larger LTCS."},
    {"term": "Thalamo-Cortical 3-Hz SWD", "definition": "3-Hz spike-wave discharge: hallmark of generalised absence epilepsy. Generated by TC neuron burst-pause oscillation (LTCS-driven) coupled to cortical layer VI. SWD amplitude in CACNA1I may be slightly higher than CACNA1G (stronger TRN-driven synchronisation of more TC neurons)."},
    {"term": "CAE (Childhood Absence Epilepsy)", "definition": "Most common GGE syndrome in children (5–10Y). 3-Hz SWD; typical absences (5–30s, eye flutter, abrupt return to awareness, no post-ictal state). HV triggers in >85%. CACNA1I GOF is a causal channelopathy for CAE."},
    {"term": "JME (Juvenile Myoclonic Epilepsy)", "definition": "GGE syndrome: morning myoclonus + GTCS ± absence; onset 12–18Y; lifelong therapy usually required. CACNA1I in ~8% of JME cohorts. TRN GOF drives morning polyspike-wave burst clustering on waking."},
    {"term": "GGE Spectrum", "definition": "Genetic Generalised Epilepsy: CAE · JAE · JME · GTCS-Alone · GEFS+. All share thalamo-cortical 3-Hz SWD pathomechanism. Same AED principles across spectrum: ETX/VPA first-line; CBZ/OXC/PHT absolute CI."},
    {"term": "ETX Level B for CACNA1I", "definition": "CACNA1I-SPECIFIC: ETX is Level B (not Level A as for CACNA1G/CACNA1H). Reason: primary GOF target is Cav3.3 in TRN which is LESS ETX-sensitive (IC50 ~30% higher for Cav3.3 vs Cav3.1). ETX still reduces 3-Hz SWD (via TC Cav3.1/Cav3.2 block + partial Cav3.3 block) but may need ETX+VPA combination for complete CACNA1I control."},
    {"term": "VPPP (MHRA 2021)", "definition": "Valproate Pregnancy Prevention Programme (UK MHRA 2021): mandatory annual documentation before each VPA prescription in females 9–55Y. Requires: specialist + GP co-signature; confirmed 2 contraceptive methods; pregnancy test; patient information card. VPA is the most teratogenic major AED (NTD 1–2%; fetal valproate syndrome)."},
    {"term": "POLG-Alpers Syndrome", "definition": "POLG1 mutation + VPA → fatal fulminant hepatic failure (Alpers-Huttenlocher syndrome). POLG1 (mitochondrial DNA polymerase γ): p.A467T and p.W748S most common pathogenic variants. VPA inhibits mitochondrial β-oxidation → catastrophic in POLG context. Screen POLG1 MANDATORY before VPA."},
    {"term": "SUDEP (Sudden Unexpected Death in Epilepsy)", "definition": "SUDEP incidence 1–9 per 1,000 patient-years (500× general population). Risk factors: uncontrolled GTCS, nocturnal seizures, drug-resistance, male sex, 18–40Y, non-adherence, disease duration >10Y. Annual SUDEP assessment: nocturnal supervision plan, wearable seizure alert, rescue protocol."},
    {"term": "ACMG-AMP 2015 Variant Classification", "definition": "5-tier classification: Pathogenic / Likely Pathogenic / VUS / Likely Benign / Benign. CACNA1I: GOF confirmed by functional assay (V1/2 inactivation shift ≥8 mV, current density increase ≥25% in heterologous expression) → Pathogenic/Likely Pathogenic. VUS → treat as standard GGE pending re-classification."},
]


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT COHORT  (40 patients)
# ─────────────────────────────────────────────────────────────────────────────
_SYNDROMES = [
    "Childhood absence epilepsy (CAE)",
    "Juvenile myoclonic epilepsy (JME)",
    "Juvenile absence epilepsy (JAE)",
    "Genetic epilepsy with febrile seizures plus (GEFS+)",
    "Epilepsy with generalised tonic-clonic seizures alone (GTCS-alone)",
]
_ETIOLOGY_TYPES = [et["category"] for et in ETIOLOGY_CATALOG]
_ETIOLOGY_WEIGHTS = [et["pct"] for et in ETIOLOGY_CATALOG]
_SEIZURE_TYPES = [st["type"] for st in SEIZURE_TYPES]
_TREATMENT_DRUGS = [tr["drug"].split(" (")[0] for tr in TREATMENTS]
_GENDERS = ["Male", "Female", "Non-binary"]
_ONSET_AGES = list(range(3, 20))

_cohort = []
random.seed(43)
for i in range(40):
    etiology = random.choices(_ETIOLOGY_TYPES, weights=_ETIOLOGY_WEIGHTS, k=1)[0]
    gender = random.choices(_GENDERS, weights=[40, 55, 5], k=1)[0]
    onset_age = random.choice(_ONSET_AGES)
    seizure_free = random.random() < 0.55
    _cohort.append({
        "patient_id": f"EPAT{i+1:03d}",
        "etiology": etiology,
        "onset_age": onset_age,
        "current_age": onset_age + random.randint(2, 25),
        "gender": gender,
        "syndrome": random.choice(_SYNDROMES),
        "seizure_free": seizure_free,
        "drug_resistant": not seizure_free and random.random() < 0.22,
        "primary_treatment": random.choices(_TREATMENT_DRUGS, k=1)[0],
        "etx_on": random.random() < 0.55,
        "vpa_on": random.random() < 0.45,
        "lev_on": random.random() < 0.30,
        "catamenial": gender == "Female" and random.random() < 0.22,
        "hv_swd_positive": random.random() < 0.88,
        "photosensitive": random.random() < 0.32,
        "gtcs_present": random.random() < 0.52,
        "myoclonic_present": random.random() < 0.40,
    })


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    n = len(_cohort)
    seizure_free_n = sum(1 for p in _cohort if p["seizure_free"])
    drug_resistant_n = sum(1 for p in _cohort if p["drug_resistant"])
    etx_n = sum(1 for p in _cohort if p["etx_on"])
    hv_swd_n = sum(1 for p in _cohort if p["hv_swd_positive"])
    photosensitive_n = sum(1 for p in _cohort if p["photosensitive"])
    gtcs_n = sum(1 for p in _cohort if p["gtcs_present"])
    catamenial_n = sum(1 for p in _cohort if p["catamenial"])
    myoclonic_n = sum(1 for p in _cohort if p["myoclonic_present"])

    etiology_dist = []
    for e in ETIOLOGY_CATALOG:
        count = sum(1 for p in _cohort if p["etiology"] == e["category"])
        etiology_dist.append({
            "category": e["category"],
            "count": count,
            "pct": round(count / n * 100, 1),
        })

    seizure_summary = [
        {"type": st["type"], "pct": st["pct"]} for st in SEIZURE_TYPES
    ]
    treatment_summary = [
        {"drug": tr["drug"].split(" (")[0], "level": tr["level"].split(" —")[0]}
        for tr in TREATMENTS
    ]
    monitoring_summary = [
        {"item": m["item"], "frequency": m["frequency"]} for m in MONITORING[:6]
    ]
    lifecycle_summary = [
        {"window": lc["window"], "key": lc["key_issues"][:80] + "…"} for lc in LIFECYCLE
    ]

    return {
        "kpis": {
            "n_patients": n,
            "seizure_free_pct": round(seizure_free_n / n * 100, 1),
            "drug_resistant_pct": round(drug_resistant_n / n * 100, 1),
            "on_etx_n": etx_n,
            "hv_swd_positive_n": hv_swd_n,
            "photosensitive_n": photosensitive_n,
            "gtcs_n": gtcs_n,
            "catamenial_n": catamenial_n,
            "myoclonic_n": myoclonic_n,
            "avg_age_years": round(sum(p["current_age"] for p in _cohort) / n, 1),
        },
        "etiology_distribution": etiology_dist,
        "seizure_summary": seizure_summary,
        "treatments_summary": treatment_summary,
        "monitoring_summary": monitoring_summary,
        "lifecycle": lifecycle_summary,
        "thresholds": THRESHOLDS[:6],
        "contraindications_summary": [
            {"drug": ci["drug"].split(" (")[0].split(" /")[0].split(" —")[0], "risk": ci["risk"]}
            for ci in CONTRAINDICATIONS[:4]
        ],
    }


def get_breakdown():
    return {
        "etiologies": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "patients": _cohort,
    }


def get_definitions():
    return {
        "gene_summary": {
            "gene": "CACNA1I",
            "full_name": "Calcium Voltage-Gated Channel Subunit Alpha1 I",
            "chromosome": "22q13.1",
            "protein": "Cav3.3 (α1I) — T-type Low-Voltage-Activated Ca²⁺ Channel",
            "size": "2031 aa · 4-domain monomeric α1I subunit",
            "channel_type": "T-type (transient) / LVA (low-voltage-activated); Cav3 subfamily",
            "activation_threshold": "−65 to −55 mV (highest = least negative, among T-type channels)",
            "inactivation_kinetics": "SLOWEST of Cav3 subfamily (τ ~50–80 ms at −40 mV; cf. Cav3.1: ~20–25 ms)",
            "window_current": "−60 to −50 mV (largest window current of Cav3 subfamily at TRN resting potential)",
            "primary_location": "Thalamic Reticular Nucleus (TRN) — DOMINANT; also cortex layers II-IV, hippocampus, basal ganglia",
            "trn_vs_tc": "TRN-DOMINANT (Cav3.3) vs CACNA1G TC-DOMINANT (Cav3.1) vs CACNA1H TC+TRN (Cav3.2)",
            "inheritance": "AD GOF reduced penetrance (~50–60%); de novo ~30–40%",
            "omim": "Epilepsy susceptibility locus (GGE spectrum; no dedicated DEE OMIM number)",
            "pli": "~0.15 (T-type channels tolerate LOF; GOF mechanism)",
            "etx_level": "ETX Level B (not Level A): Cav3.3 is LESS ETX-sensitive than Cav3.1/Cav3.2; TRN-indirect mechanism",
            "absolute_ci": "CBZ / OXC / PHT (GGE aggravation → absence status); TGB (NCSE); VPA+POLG1 (Alpers)",
        },
        "definitions": DEFINITIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
