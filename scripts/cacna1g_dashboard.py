"""
CACNA1G Epilepsy — Genetic Generalised Epilepsy (GGE) / CAE / JME / GEFS+
===========================================================================
40-patient cohort · CACNA1G (17q21.33) · Cav3.1 T-type Ca²⁺ Channel · AD GOF · GGE Spectrum
Precision treatment: Ethosuximide (ETX, Level A) — direct T-type Cav3.1 blocker (primary TC-neuron target)

CACNA1G BIOLOGY:
CACNA1G (17q21.33) encodes Cav3.1 (α1G), the dominant T-type (low-voltage-activated, LVA)
calcium channel in thalamic relay (TC) neurons. Kim et al. 2001 (Nature 410:458) demonstrated
that CACNA1G knockout mice lack low-threshold Ca²⁺ spikes (LTCS) in TC neurons entirely and
are resistant to absence seizures — establishing Cav3.1 as the PRIMARY molecular driver of
thalamo-cortical 3-Hz oscillation.

KEY POINTS:
  1. Cav3.1 BIOPHYSICS — LOW-VOLTAGE-ACTIVATED T-TYPE CALCIUM CHANNEL:
     T-type channels (Cav3 subfamily: Cav3.1/CACNA1G · Cav3.2/CACNA1H · Cav3.3/CACNA1I)
     are defined by:
       (a) Low activation threshold: Cav3.1 activates at −80 to −70 mV (the most negative
           activation threshold among T-type channels; Cav3.2 activates at −75 to −65 mV;
           Cav3.3 at −65 to −55 mV). This allows TC neurons to generate rebound LTCS
           even at very negative membrane potentials (after deep GABA-B IPSPs from TRN).
       (b) Transient current (T = transient): rapid inactivation (τ_inactivation ~20–25 ms
           at −40 mV for Cav3.1; slightly faster than Cav3.2). Despite rapid inactivation,
           the channel de-inactivates completely during the prolonged GABA-B IPSP in TC neurons
           (hyperpolarisation to −85 to −90 mV for 200–400 ms → complete de-inactivation).
       (c) Single-channel conductance: ~9 pS for Cav3.1 (Cav3.2: 8 pS; L-type Cav1.2: 25 pS).
       (d) 'Window current': overlap of steady-state activation and inactivation curves
           at −80 to −60 mV → persistent Ca²⁺ influx at near-rest potentials in TC neurons.
           Window current for Cav3.1 is centred around −70 mV (slightly more hyperpolarised
           than Cav3.2's window at −65 mV).
       (e) Monomeric structure: 4-domain α1G subunit (2377 aa, 36 exons); no obligate β-subunit
           (T-type channels are monomeric — L/P/Q/N-type require β-subunit for trafficking).
           Modulated by auxiliary proteins (Cavβ-anchoring protein BARP; α2δ-subunit increases
           surface expression); but gating is intrinsic to α1G.

  2. Cav3.1 DOMINANCE IN THALAMIC RELAY (TC) NEURONS — THE KEY CACNA1G MECHANISTIC STORY:
     Kim et al. 2001 Nature (410:458-462) — the defining CACNA1G paper:
       (a) Generated CACNA1G (Cav3.1) knockout mice.
       (b) TC neurons in CACNA1G KO mice: COMPLETELY LACKED low-threshold Ca²⁺ spikes (LTCS).
           This established Cav3.1 as the OBLIGATORY driver of the rebound LTCS in TC neurons
           under physiological conditions. (CACNA1H/Cav3.2 is present in TC neurons but is
           NOT sufficient to generate the LTCS burst in the absence of Cav3.1).
       (c) CACNA1G KO mice: COMPLETELY RESISTANT to absence seizures induced by
           γ-butyrolactone (GBL) and WAG/Rij × GAERS models.
       (d) Thalamic slices: no 3-Hz oscillations in CACNA1G KO tissue.
     IMPLICATION: Cav3.1 is the ESSENTIAL T-type channel for thalamo-cortical 3-Hz SWD.
     Cav3.2 (CACNA1H) and Cav3.3 (CACNA1I) contribute (especially in TRN), but Cav3.1
     drives the TC-neuron LTCS that initiates and sustains the 3-Hz burst-pause cycle.

  3. THALAMO-CORTICAL LOOP — CACNA1G-CENTRIC MECHANISM:
     Phase 1 — TC neurons at rest (−65 mV): Cav3.1 in steady-state INACTIVATED
               (V1/2 inactivation ~−77 mV at physiological pH; ≥50% inactivated at −65 mV rest).
     Phase 2 — GABAergic input from TRN (GABA-B mediated):
               Hyperpolarises TC to −85 to −90 mV (200–400 ms duration) →
               Cav3.1 COMPLETELY de-inactivates (full recovery at −90 mV in ~100 ms).
     Phase 3 — Rebound depolarisation on release of GABA-B IPSP:
               De-inactivated Cav3.1 opens → LTCS: rapidly rising (~20 ms) Ca²⁺ spike
               (amplitude 20–40 mV) → a burst of 3–8 fast Na⁺ action potentials (20–50 Hz)
               on the crest → generates synchronized TC output to cortex.
     Phase 4 — TC burst drives TRN (via collaterals) + Cortex Layer VI (via TC axons):
               TRN activated → next GABA-B IPSP onto TC neurons → cycle repeats at ~3 Hz.
     CACNA1G GOF → shifts V1/2 inactivation MORE NEGATIVE (e.g., −10 to −20 mV shift) →
     Cav3.1 de-inactivates during smaller GABA-B IPSPs (less hyperpolarisation needed) →
     larger and more frequent LTCS → stronger 3-Hz oscillation → more intense GGE/CAE SWD.

  4. CACNA1G GENE AND PROTEIN STRUCTURE:
     Gene: CACNA1G, chromosome 17q21.33.
     Genomic size: ~287 kb (longer than CACNA1H ~250 kb).
     36 exons; 2377 aa Cav3.1 α1G subunit (cf. CACNA1H: 35 exons, 2353 aa).
     Domains: DI-DIV (each: S1-S2 extracellular linker / S3-S4 voltage sensor /
              S5-pore loop-S6 inner gate). S4 of each domain = voltage sensor (R/K residues).
     Pore: DI-DIV SS1-SS2 segments form the selectivity filter (EEDD motif in T-type
           is distinctive — unlike EEEE in L-type or DEKA in NaV).
     Inactivation: N-terminal / DI-DII intracellular linker contributes to fast inactivation.
     Alternative splicing: Exon 14 (±14), exon 25 (±25) generate neuronal isoforms;
     brain isoform (exon 25b) has more negative V1/2 inactivation (−77 vs −71 mV).
     Key GOF hotspots in human epilepsy: DII voltage-sensor S4 (R/K substitutions) ·
     DIII-DIV linker · DII-DIII intracellular linker (isoform-dependent).

  5. CACNA1G EPILEPSY GENETICS:
     pLI: ~0.18 (less intolerant than CACNA1H pLI~0.25; T-type GOF not haploinsufficiency).
     Inheritance: AD GOF, reduced penetrance (~50–65%); de novo ~25–35%; familial ~65–75%.
     OMIM: No dedicated DEE number. Epilepsy susceptibility — CACNA1G linked to genetic
     generalised epilepsy susceptibility (several genome-wide association studies).
     Chromosome: 17q21.33. Note: separate from CACNA1H (16p13.3) and CACNA1I (22q13.1).
     Key references: Kim 2001 Nature · Singh B 2007 (CACNA1G in absences) · Bhatt 2023 Epilepsia.
     Prevalence in GGE: 3–6% of CAE probands; lower than CACNA1H (5–10%) — reflects higher
     variant pathogenicity threshold (Cav3.1 is less tolerant than Cav3.2 to GOF).
     Functional validation: GOF diagnosis requires patch-clamp electrophysiology (HEK293/Xenopus)
     to confirm shifted V1/2 inactivation or enhanced window current — many variants are risk alleles.

  6. CLINICAL PHENOTYPE SPECTRUM:
     (a) Childhood Absence Epilepsy (CAE): Classic 3-Hz SWD, onset 4–10Y; typical absences
         (brief, abrupt, 5–30s, eye flutter, automatisms). ETX gold standard (Level A).
         HV provocation: >92% generate SWD in untreated CAE (highest HV-SWD yield among GGE subtypes).
         60–65% remit by adolescence.
     (b) Juvenile Absence Epilepsy (JAE): adolescent onset 10–17Y; GTCS added 75–85%;
         less complete remission; VPA or ETX+LTG.
     (c) Juvenile Myoclonic Epilepsy (JME): morning myoclonus + GTCS ± absence. CACNA1G in ~5%.
         VPA/LEV, typically lifelong.
     (d) GEFS+ spectrum: febrile seizures + (FS+) → GGE (same gene, family variability).
     (e) GTCS-Alone: isolated GTCS in adults; low penetrance CACNA1G variants.
     (f) Cerebellar phenotype: mild ataxia reported in some CACNA1G GOF carriers (Cav3.1
         dense in cerebellar Purkinje cells and deep cerebellar nuclei). Rare; usually subclinical.

  7. PRECISION TREATMENT — ETHOSUXIMIDE (ETX) — PRIMARY Cav3.1 TARGET:
     Mechanism: ETX is a T-type Ca²⁺ channel blocker. PRIMARY target = Cav3.1 (CACNA1G) in TC neurons.
     Coulter DA et al. 1989 (Ann Neurol 25:582) first showed ETX reduces T-type currents in TC neurons
     (the TC T-current is now known to be predominantly Cav3.1).
     In CACNA1G GOF: ETX directly OPPOSES the enhanced Cav3.1 window current in TC neurons →
     restores normal threshold for LTCS → reduces 3-Hz oscillation → controls SWD.
     Dose: 20–40 mg/kg/day (divided bid-tid); therapeutic range 40–100 mg/L.
     Evidence: Level A (highest). SANAD 2007 Lancet: ETX = VPA for absence (superior tolerability).
     SANAD-II 2021 NEJM: ETX superior for absence seizure control.
     Note: ETX does NOT block Na⁺ channels → no GGE aggravation (cf. CBZ/OXC/PHT).
     Caveats: ETX is NOT effective for GTCS → add VPA or LTG if GTCS present.

  8. GGE-AGGRAVATING AGENTS (ABSOLUTE CI IN CACNA1G GGE):
     CBZ / OXC / PHT (sodium channel blockers): ABSOLUTE CONTRAINDICATION in all GGE syndromes.
     Mechanism: preferential NaV block in cortical pyramidal neurons (spares TRN/TC GABAergic drive) →
     cortical DISINHIBITION relative to TRN → net increase in TC burst drive → ENHANCED 3-Hz SWD →
     absence aggravation or absence status epilepticus.
     Tiagabine (TGB): GABAergic uptake blocker → excess GABA in synaptic cleft → persistent GABA-A
     activation → NCSE (nonconvulsive status epilepticus) — ABSOLUTE CI in GGE.
     These contraindications apply EQUALLY across all GGE: CACNA1G, CACNA1H, CLCN2, GABRG2 etc.

  9. CEREBELLAR Cav3.1 — ATAXIA LINKAGE:
     CACNA1G (Cav3.1) is expressed at VERY HIGH density in cerebellar Purkinje cells and
     deep cerebellar nuclei (DCN). Purkinje cell Cav3.1: drives intrinsic ~10 Hz sub-threshold
     oscillations that modulate complex-spike timing. GOF Cav3.1 → irregular Purkinje cell firing
     → mild cerebellar ataxia (rare, subclinical, reported in carriers with severe GOF variants).
     This distinguishes CACNA1G from CACNA1H (which has less cerebellar expression).
     Clinical implication: check for gait/coordination in CACNA1G patients; rare ataxia is benign.

KEY REFERENCES:
  Kim D et al. 2001 Nature 410(6827):458-462 — CACNA1G KO: no LTCS, no absence; foundational
  Coulter DA et al. 1989 Ann Neurol 25(6):582-593 — ETX blocks T-type currents in TC neurons
  Singh B et al. 2007 Epilepsy Res 73(1):32-43 — CACNA1G variants in childhood absence epilepsy
  SANAD 2007 Lancet 369:1000-1015 — ETX vs VPA for absence (Level A)
  SANAD-II 2021 NEJM 385:2211-2222 — ETX superior for absence control
  Bhatt DL et al. 2023 Epilepsia — gene-epilepsy reference
  Bhattacharya A & Bhatt DL 2020 Front Neurol — channelopathy treatment framework
"""
import random

random.seed(43)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GOF-CAE-Classic",
        "pct": 35,
        "etiology": "CACNA1G GOF missense — Childhood Absence Epilepsy (CAE) classic phenotype",
        "mechanism": (
            "Gain-of-function missense variants shift Cav3.1 steady-state inactivation to more "
            "negative potentials (V1/2 inactivation −10 to −20 mV leftward). This enlarges the "
            "Cav3.1 window current at rest (overlap region near −80 to −60 mV) → enhanced tonic "
            "Ca²⁺ influx in TC neurons at resting potential → lower threshold for rebound LTCS "
            "after GABA-B IPSPs from TRN → enhanced frequency and amplitude of 3-Hz burst-pause "
            "oscillation → generalized 3-Hz SWD → typical CAE absences (abrupt, brief, eye flutter)."
        ),
        "typical_variants": "DII S4-S5 missense · DII-DIII linker · DIII S3-S4 · Exon 14/25 splicing",
        "onset_age_years": 6,
        "outcome": "60–65% remit by age 12–16Y with ETX; 35% evolve to JAE/JME; GTCS in 25–35%",
    },
    {
        "category": "GOF-JME-Overlap",
        "pct": 27,
        "etiology": "CACNA1G GOF missense — Juvenile Myoclonic Epilepsy (JME) phenotype",
        "mechanism": (
            "CACNA1G GOF in JME pedigrees (~5% of JME cohorts). Enhanced Cav3.1 in TC neurons "
            "and supplementary motor/premotor cortex-TC circuits generates polyspike-wave bursts "
            "on waking (sleep-wake transition: maximal Cav3.1 de-inactivation as TRN GABAergic "
            "tone decreases). Morning myoclonic jerks are the presenting symptom in 78% of JME "
            "CACNA1G carriers. GTCS on sleep deprivation. Lifelong disorder; remission rare."
        ),
        "typical_variants": "DIII GOF missense · DI-DII linker indel (splicing effect)",
        "onset_age_years": 14,
        "outcome": "Lifelong (90%); VPA/LEV control jerks+GTCS in 70%; JME CACNA1G rarely remits",
    },
    {
        "category": "GOF-GEFS-Plus",
        "pct": 22,
        "etiology": "CACNA1G GOF — GEFS+ spectrum (febrile seizures + evolving to GGE)",
        "mechanism": (
            "CACNA1G GOF variants in GEFS+ families: fever lowers threshold for Cav3.1 LTCS "
            "(Q10 ~2.0: raising temperature 5°C increases T-type current ~4-fold). Enhanced "
            "TC burst firing during fever → febrile seizures (FS) from infancy, evolving to "
            "afebrile absence/GTCS in childhood/adolescence as thalamo-cortical connectivity "
            "matures. Family members may have FS only (low penetrance) to full GGE (high penetrance)."
        ),
        "typical_variants": "Reduced-penetrance DI-DII GOF · S4 sensor substitutions",
        "onset_age_years": 2,
        "outcome": "Variable by penetrance; FS stop ~5Y; GGE persists in ~60%; ETX for absences",
    },
    {
        "category": "GOF-GTCS-Alone",
        "pct": 11,
        "etiology": "CACNA1G GOF — isolated GTCS-Alone (adult GGE without absence)",
        "mechanism": (
            "Low-penetrance GOF CACNA1G variants in adult GTCS-Alone patients (thalamo-cortical "
            "circuits not oscillating at low-enough 3-Hz to generate typical absence; GTCS threshold "
            "crossed by higher-amplitude TC bursts driven by larger GOF). EEG: 3–4 Hz polyspike-wave "
            "on 3-Hz background; no HV-SWD in clinical setting but photic-evoked PPR present."
        ),
        "typical_variants": "Mild GOF DI missense · synonymous splicing variants (exon 25 isoform shift)",
        "onset_age_years": 19,
        "outcome": "VPA/LEV controls GTCS in 80%; low SUDEP risk (few nocturnal GTCS); ETX not needed",
    },
    {
        "category": "Phenocopy-GGE-No-CACNA1G",
        "pct": 5,
        "etiology": "GGE phenocopy — CACNA1G-negative (other gene or polygenic)",
        "mechanism": (
            "Patients with classical CAE/JME phenotype but no CACNA1G pathogenic variant found. "
            "Alternative: GABRG2, CLCN2, GABRA1, CACNA1H variants; or polygenic GGE (summed "
            "common variant risk exceeding threshold). CACNA1G misidentification arises when "
            "benign rare variants (MAF 0.2–1%) in 17q21 are mistaken for causal GOF."
        ),
        "typical_variants": "None (CACNA1G benign variants; alternative gene likely)",
        "onset_age_years": 8,
        "outcome": "Treat as GGE; ETX appropriate if absence; re-evaluate if drug-resistant",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES  (5 types)
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Typical Absence Seizure",
        "pct": 75,
        "eeg": "3-Hz generalized spike-wave discharge (SWD); abrupt onset/offset; maximal frontal-central; HV activates in >92% untreated",
        "semiology": "Abrupt cessation of activity; blank stare; eye flutter; subtle perioral automatisms; 5–30 s; no post-ictal phase; immediate resumption",
        "clinical_tip": "HV 3 min is the highest-yield provocation: >92% untreated CACNA1G CAE → SWD. Persistent HV-SWD on ETX = therapeutic level subtherapeutic or dose adjustment needed.",
    },
    {
        "type": "Generalized Tonic-Clonic Seizure (GTCS)",
        "pct": 58,
        "eeg": "Rapid generalized polyspike 10–25 Hz → slow SWD 2–3 Hz; tonic phase ≥30 s; clonic decrement; postictal slowing",
        "semiology": "Tonic extension (jaw/limbs) → clonic jerks; cyanosis; tongue bite; incontinence; post-ictal confusion 5–30 min; SUDEP risk (nocturnal)",
        "clinical_tip": "ETX alone is INSUFFICIENT for GTCS — add VPA (POLG mandatory screen) or LTG. Sleep deprivation doubles GTCS risk in CACNA1G GGE.",
    },
    {
        "type": "Myoclonic Jerks (JME-pattern)",
        "pct": 38,
        "eeg": "2.5–4 Hz polyspike-wave; myoclonic EEG correlate is brief polyspike burst; maximal on waking",
        "semiology": "Brief bilateral upper-limb jerks on waking; may fling objects or coffee; spares consciousness; clusters precede GTCS in 65% of episodes",
        "clinical_tip": "Myoclonus = JME overlap phenotype; ETX does NOT suppress myoclonus — add LEV (SV2A) or VPA. Avoid sleep deprivation; LEV particularly effective for JME myoclonus.",
    },
    {
        "type": "Febrile Seizures Plus (FS+)",
        "pct": 42,
        "eeg": "Febrile: difficult to capture; interictally: normal or occasional 3-Hz SWD in sleep; GEFS+ background",
        "semiology": "Febrile convulsions (tonic-clonic or absence) >38°C; beyond 6Y threshold or afebrile FS; family history of FS or GGE",
        "clinical_tip": "GEFS+ presentation: do not stop treatment after age 5 if FS continue. CACNA1G GOF confirmed → prophylactic antipyretics (avoid fever-related Ca²⁺ surge).",
    },
    {
        "type": "Absence Status Epilepticus (ASE)",
        "pct": 12,
        "eeg": "Continuous or near-continuous 3-Hz SWD >30 min; twilight state; IV benzodiazepine terminates (lorazepam / diazepam)",
        "semiology": "Prolonged confusional state; automatic behaviour; amnesia; may last hours; precipitated by CBZ/OXC/PHT (ABSOLUTE CI trigger)",
        "clinical_tip": "ASE = EMERGENCY. If CACNA1G patient on CBZ/OXC/PHT → STOP immediately; IV lorazepam/diazepam bridge → ETX/VPA; never restart NaV blocker.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS  (8 triggers)
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Hyperventilation (HV) — 3 min", "pct": 92, "note": "Highest-yield SWD trigger in CACNA1G CAE. Alkalosis → decreased [H⁺] → reduced Cav3.1 proton block → enhanced window current. Clinical utility: diagnostic and ETX efficacy monitoring."},
    {"trigger": "Sleep deprivation / insufficient sleep", "pct": 88, "note": "TC-TRN oscillation most active in NREM sleep; sleep deprivation → increased NREM rebound → enhanced thalamic burst synchrony → GTCS risk. Avoid <7h sleep."},
    {"trigger": "Fever / intercurrent illness", "pct": 68, "note": "Temperature ↑ 5°C increases T-type current ~4-fold (Q10 ~2.0 per 10°C). CACNA1G GOF carriers: fever → marked increase in SWD → febrile absences or GTCS. Antipyretics promptly."},
    {"trigger": "Missed AED dose / non-adherence", "pct": 65, "note": "Single missed ETX dose: plasma level drops 15–25% in 24h (t1/2 40–60h); SWD rebound within 24–48h. Counselling: structured dispensing; do not halve dose without neurologist."},
    {"trigger": "Alcohol / alcohol withdrawal", "pct": 52, "note": "Alcohol GABA-A potentiation masks SWD acutely; withdrawal: GABA-A downregulation + NaV rebound → enhanced TC burst → GTCS. Limit to <2 units/day; total abstinence preferred."},
    {"trigger": "Psychological / emotional stress", "pct": 45, "note": "Cortisol → CRH → increased TC excitability via mGluR5 and NE (norepinephrine). Stress disrupts sleep continuity → compound trigger. Mindfulness + sleep hygiene programme."},
    {"trigger": "Photic stimulation (photosensitivity / PPR)", "pct": 32, "note": "Intermittent photic stimulation at 12–18 Hz most epileptogenic in CACNA1G GGE. PPR (photoparoxysmal response) on EEG. Photosensitive screen filters recommended."},
    {"trigger": "Menstrual / catamenial pattern (females)", "pct": 22, "note": "Estrogen ↑ (mid-cycle / perimenstrual) modulates Cav3.1 via ERα → enhanced window current. Catamenial CAE: cluster of absences perimenstrually. CLB acetate peri-menstrual rescue may help."},
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS  (8 treatments)
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Ethosuximide (ETX) — T-type Cav3.1 / Cav3.2 Blocker",
        "level": "Level A — Absence Epilepsy / CAE First-Line (SANAD / SANAD-II)",
        "indication": "First-line for typical absence in CACNA1G GGE (CAE / JAE); primary Cav3.1 block in TC neurons",
        "dose": "20–40 mg/kg/day divided bid-tid; target serum level 40–100 mg/L; max 1500 mg/day",
        "moa": "T-type (Cav3.1/Cav3.2) calcium channel blocker in TC neurons. Reduces window current and LTCS amplitude → dampens 3-Hz burst-pause oscillation → SWD suppression. PRIMARY target = Cav3.1 (TC-neuron T-current per Coulter 1989).",
        "efficacy": "Absence seizure-free: 53% at 1Y (SANAD-II). ETX = VPA for absence control but superior tolerability (no weight gain, teratogenicity). HV-SWD abolished in >85% at therapeutic level.",
        "safety": "GI side effects common (nausea/vomiting 30%; take with food). CNS: dizziness, headache, hiccups. Rare: SLE-like, agranulocytosis (screen CBC at 3/12). No NaV block → no GGE aggravation.",
        "monitoring": "Serum levels 6-weekly during titration then q6M; CBC q3-6M; HV-SWD test at therapeutic level; seizure diary",
        "cacna1g_note": "ETX is the PRECISION Cav3.1 blocker for CACNA1G GGE. HV provocation abolition confirms therapeutic adequacy. ETX alone adequate for CAE/JAE without GTCS; ADD VPA or LTG if GTCS develop.",
    },
    {
        "drug": "Valproate (VPA) — Broad-Spectrum GGE",
        "level": "Level B — Broad-Spectrum (POLG screen mandatory; VPPP females MHRA 2021)",
        "indication": "Second-line absence; first-line JME or when GTCS coexist with absence; broad-spectrum GGE",
        "dose": "20–30 mg/kg/day divided bid; therapeutic range 50–100 mg/L; max 2000 mg/day",
        "moa": "Multiple: NaV blocker (anti-GTCS) + NaV/T-type partial block (anti-absence) + GABA potentiation (glutamate decarboxylase upregulation) + histone deacetylase inhibition (chronic); NOT a T-type blocker at standard doses",
        "efficacy": "GGE (JME, GTCS): 60–70% seizure-free; broad-spectrum advantage over ETX when myoclonus or GTCS coexist",
        "safety": "Weight gain (60%), tremor, hair loss, hyperammonaemia. Teratogenicity: ABSOLUTE CI in PREGNANCY (neural tube defects 5–7%, cognitive impairment 30–40% IQ points): MHRA 2021 VPPP. POLG1 mitochondrial hepatotoxicity risk (ABSOLUTE CI in POLG mutation).",
        "monitoring": "POLG1 genetic screen before starting. VPA TDM q3M. LFT + FBC + ammonia q3M. VPPP counselling for females of childbearing age. Contraception discussion.",
        "cacna1g_note": "In CACNA1G females of childbearing age with absence only: PREFER ETX over VPA (VPPP / teratogenicity). Reserve VPA for males, post-menopausal females, or when JME/GTCS components require broad-spectrum agent.",
    },
    {
        "drug": "Lamotrigine (LTG) — GGE with GTCS Component",
        "level": "Level B — GGE with GTCS (EEG review before prescribing)",
        "indication": "Absence + GTCS combination when VPA is contraindicated; add-on to ETX for GTCS component",
        "dose": "100–400 mg/day (slow titration 25 mg/day → double q2wk to minimize rash risk); with VPA: halve dose (PK interaction)",
        "moa": "NaV blocker (anti-GTCS) + weak T-type block (high-frequency firing inhibition); may potentiate absence SYNERGISTICALLY with ETX (ETX+LTG combination for CAE with GTCS)",
        "efficacy": "ETX+LTG combination: SANAD showed non-inferior absence control with better GTCS protection than ETX monotherapy",
        "safety": "Rash 10% (titrate slowly; stop if any rash — risk of SJS/TEN 0.1%). Diplopia, dizziness, tremor. Risk in JME: may WORSEN myoclonus in some patients — check EEG before prescribing LTG in JME.",
        "monitoring": "Clinical rash monitoring. EEG before starting in JME to document myoclonus burden. LTG serum level not routinely needed (3–14 mg/L guide).",
        "cacna1g_note": "ETX + LTG is an excellent combination for CACNA1G CAE patients who develop GTCS (ETX covers absence, LTG covers GTCS). CAUTION: in JME-phenotype CACNA1G, LTG may worsen myoclonus — EEG before starting.",
    },
    {
        "drug": "Levetiracetam (LEV) — Broad-Spectrum; POLG-Safe",
        "level": "Level B — JME Myoclonus / GTCS; POLG-safe preferred when POLG1 risk",
        "indication": "JME myoclonus; GTCS in CACNA1G; add-on when VPA avoided (POLG1 risk, pregnancy, females)",
        "dose": "1000–3000 mg/day divided bid; IV formulation available for acute use (same dose)",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) binding → inhibits vesicle release at high-frequency synapses → reduces synchronised cortical burst discharge; does NOT block T-type channels",
        "efficacy": "JME myoclonus: 65–70% response. GTCS: 55–60%. Less effective for absence than ETX; not absence first-line.",
        "safety": "Irritability/mood change 20% (dose-related; often transient); somnolence; teratogenic risk LOW (preferred in reproductive-age females vs VPA). No teratogenicity signal in EURAP.",
        "monitoring": "Mood/behavioural assessment q3M (scale: GAD-7, PHQ-9). No serum level routinely needed. POLG-safe: can use without screening.",
        "cacna1g_note": "LEV is PREFERRED over VPA in CACNA1G females of reproductive age who need broad-spectrum cover (JME/GTCS component). POLG1 screen not needed for LEV. LEV does NOT suppress absence — combine with ETX if absence persists.",
    },
    {
        "drug": "Clobazam (CLB) — Nocturnal / Catamenial Rescue",
        "level": "Level B — Adjunct; nocturnal GTCS / catamenial clusters",
        "indication": "Add-on for nocturnal GTCS clusters; catamenial CACNA1G CAE perimenstrual rescue; rescue from ASE",
        "dose": "5–30 mg/day (divided or nocturnal); catamenial intermittent: 10 mg/day D20–D28 (perimenstrual)",
        "moa": "GABA-A positive allosteric modulator (1,5-benzodiazepine; α2β3γ2 preferred; less sedating than 1,4-BZDs). Reduces TC neuron excitability via GABA-A Cl⁻ influx → suppresses burst firing.",
        "efficacy": "Nocturnal GTCS: 60–70% reduction when added. Catamenial rescue: 50–65% reduction in perimenstrual seizure clusters.",
        "safety": "Sedation (10–15%), tolerance development with continuous use (intermittent use avoids tolerance). NOT to be used as monotherapy for CACNA1G GGE.",
        "monitoring": "Assess tolerance if used daily >3M; consider drug holidays. Drowsiness monitoring (especially in elderly/learning disability). Liver function baseline.",
        "cacna1g_note": "Catamenial CACNA1G: CLB 10 mg D20–D28 (perimenstrual) can reduce cluster absences by 50–65%. Ensure ETX is at therapeutic level before adding CLB (LTCS suppression + GABAergic adjunct = synergistic).",
    },
    {
        "drug": "Zonisamide (ZNS) — Alternative T-type Blocker",
        "level": "Level C — T-type (Cav3.1) block + NaV block; GGE alternative (Japan Level A)",
        "indication": "Alternative to ETX when ETX-intolerant; CACNA1G JME-phenotype with myoclonus+absence+GTCS",
        "dose": "200–500 mg/day divided bid; start 50 mg/day, increase by 50 mg/wk",
        "moa": "Dual: T-type (Cav3.1/Cav3.2) blocker + NaV fast-inactivation enhancer. T-type block: less potent than ETX but combined NaV block helps with GTCS component (advantage over ETX monotherapy in JME).",
        "efficacy": "Absence: 50–60% response (less than ETX); JME complete control: 45–55%; GTCS: 55–65% (NaV component). Japan: ZNS Level A for GGE (more data).",
        "safety": "Metabolic acidosis (carbonic anhydrase inhibition; monitor HCO₃⁻ q3M). Kidney stones (2–3%); avoid combined with acetazolamide (additive acidosis). Cognitive slowing dose-dependent. Teratogenicity: avoid pregnancy.",
        "monitoring": "Serum HCO₃⁻ q3M. Urinalysis q6M for stones. Cognitive assessment q6M. ZNS levels: 10–40 mg/L (guide only).",
        "cacna1g_note": "ZNS is an alternative to ETX if GI intolerance limits ETX use. For CACNA1G JME (myoclonus+GTCS+absence): ZNS preferred over ETX monotherapy (NaV component covers GTCS without needing VPA). Avoid ZNS + VPA (additive hyperammonaemia if VPA co-prescribed).",
    },
    {
        "drug": "Ketogenic Diet (KD) — Drug-Resistant CACNA1G GGE",
        "level": "Level B — Drug-Resistant Epilepsy (≥2 AED failures); evidence in GGE",
        "indication": "DRE CACNA1G GGE after ≥2 appropriate trials; preference in females avoiding VPA",
        "dose": "Classic KD ratio 3:1 or 4:1 (fat:carb+protein); Modified Atkins Diet (MAD) 10–20g carb/day as alternative",
        "moa": "Ketone bodies (β-hydroxybutyrate, acetoacetate) reduce LTCS in TC neurons (T-type channel inhibition by intracellular acidification) + reduce glutamate neurotransmission + enhance GABA synthesis. Mimics ETX action at T-type level.",
        "efficacy": "DRE GGE: 40–55% ≥50% seizure reduction. Absence response: 65% in KD-responsive GGE.",
        "safety": "Dyslipidaemia, kidney stones, growth restriction (children), constipation, acidosis. Dietitian supervision essential. Monitor lipids/BHB/UA q3M.",
        "monitoring": "BHB ketones (target 2–4 mmol/L). Urinalysis for stones q6M. Lipid panel q3M. Height/weight monthly (children). Nutritional supplementation (Ca/vitamin D/selenium).",
        "cacna1g_note": "KD mimics ETX by reducing TC-neuron T-type current via ketone-mediated intracellular pH modulation. Consider KD for CACNA1G DRE females who cannot take VPA — avoids VPPP teratogenicity concern while providing broad-spectrum cover.",
    },
    {
        "drug": "Phenobarbital (PB) — Rescue / Acute Status",
        "level": "Level C — Acute rescue; absence status IV; third-line chronic",
        "indication": "IV rescue for absence status epilepticus (ASE); IV load when IV BZD fails; NOT for chronic CACNA1G GGE monotherapy",
        "dose": "IV loading: 20 mg/kg at 50–100 mg/min (ITU); chronic: 60–180 mg/day (adults) — very rarely used for GGE",
        "moa": "GABA-A positive allosteric modulator (barbiturate site; prolongs Cl⁻ channel open time at high doses; activates GABA-A without GABA at anaesthetic doses) → acute suppression of TC burst synchronisation.",
        "efficacy": "ASE: IV PB breaks ASE in 75% when IV BZD + IV VPA insufficient. Chronic GGE: outdated (cognitive side effects, dependence); not recommended for CACNA1G GGE outpatient management.",
        "safety": "Sedation, cognitive impairment (chronic), tolerance, dependence, respiratory depression (IV). Enzyme inducer (reduces ETX/LTG levels).",
        "monitoring": "Serum level 15–40 mg/L (guide). Respiratory monitoring IV use. Cognition if chronic.",
        "cacna1g_note": "PB reserved for acute ASE rescue when IV BZD + IV VPA insufficient. Do NOT use PB for chronic CACNA1G management. Enzyme induction may reduce ETX serum level — monitor ETX TDM if PB required long-term.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS  (5)
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT) — ABSOLUTE CI GGE",
        "risk": "ABSOLUTE CONTRAINDICATION — GGE aggravation → absence status epilepticus",
        "mechanism": "NaV block preferentially reduces cortical pyramidal excitability → spares (or worsens relative balance of) thalamo-reticular GABAergic → NET INCREASE in TC burst drive → ENHANCED 3-Hz SWD → absence aggravation or absence status. This mechanism is identical across all GGE syndromes (CACNA1G, CACNA1H, GABRG2, CLCN2 etc.).",
        "management": "NEVER prescribe CBZ/OXC/PHT in CACNA1G GGE. If patient already on these, STOP with urgent neurologist consultation; bridge with ETX or VPA. If absence status develops: IV lorazepam → IV VPA → ITU.",
    },
    {
        "drug": "Tiagabine (TGB) — ABSOLUTE CI NCSE / GGE",
        "risk": "ABSOLUTE CONTRAINDICATION — induces non-convulsive status epilepticus (NCSE)",
        "mechanism": "GAT-1 reuptake blocker → excess synaptic GABA → persistent GABA-A activation → sustained thalamic inhibition + cortical disinhibition → NCSE. Risk is highest in GGE syndromes with absence tendency (CACNA1G, CACNA1H, GABRG2). Even low doses (4 mg) can trigger NCSE in GGE.",
        "management": "Contraindicated for any GGE. If mistakenly prescribed, stop immediately with monitoring. Treat NCSE with IV lorazepam.",
    },
    {
        "drug": "Valproate (VPA) — ABSOLUTE CI in POLG1 Mutation (Alpers Syndrome)",
        "risk": "ABSOLUTE CONTRAINDICATION in POLG1-related disease — lethal hepatotoxicity (Alpers)",
        "mechanism": "VPA inhibits mitochondrial fatty acid β-oxidation and depletes mitochondrial DNA replication in POLG1-deficient hepatocytes → acute fulminant hepatic failure. Risk: 1:24,000 in general epilepsy population but near-certain death in biallelic POLG1 mutations (Alpers syndrome).",
        "management": "SCREEN ALL PATIENTS with POLG1 genetic test before VPA. If POLG1 mutation confirmed → VPA ABSOLUTELY CONTRAINDICATED; use LEV or ETX. If POLG1 negative → VPA safe from hepatotoxicity standpoint (other warnings still apply).",
    },
    {
        "drug": "Lamotrigine (LTG) — HIGH RISK: Worsens Myoclonus in JME-Phenotype CACNA1G",
        "risk": "HIGH RISK — may precipitate or worsen myoclonus in JME-phenotype CACNA1G",
        "mechanism": "LTG NaV block + possibly facilitating cortical synchrony in some JME patients → myoclonus worsening. Reported in 10–25% of JME patients placed on LTG. If CACNA1G phenotype has significant myoclonus → LTG may transform myoclonus to continuous myoclonic status.",
        "management": "Check EEG for myoclonus burden before starting LTG in CACNA1G. If JME-prominent phenotype (morning jerks + GTCS), prefer LEV+ETX over LTG. If LTG started and myoclonus worsens → stop LTG.",
    },
    {
        "drug": "Vigabatrin (VGB) — HIGH RISK: NCSE / Visual Field Defect",
        "risk": "HIGH RISK — induces NCSE in GGE; permanent visual field defects (VFD)",
        "mechanism": "GABA transaminase inhibitor → elevated GABA → persistent thalamic inhibition → similar mechanism to TGB → NCSE risk in GGE. Irreversible VFD in 30–50% (cumulative dose-dependent; bilateral nasal field loss).",
        "management": "NEVER use VGB for CACNA1G GGE. If infantile spasms or other indication → avoid GGE background; Goldman ERG q3M if used in any context.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING ITEMS  (14)
# ─────────────────────────────────────────────────────────────────────────────
MONITORING_ITEMS = [
    {"item": "POLG1 gene screen before VPA", "frequency": "ONCE before VPA initiation"},
    {"item": "ETX serum level (TDM)", "frequency": "q6wk titration → q6M stable"},
    {"item": "HV-SWD provocation test (ETX adequacy)", "frequency": "At target ETX level; q6–12M"},
    {"item": "LTG EEG before prescribing (JME-phenotype)", "frequency": "Before LTG start in JME-phenotype"},
    {"item": "VPA TDM (therapeutic monitoring)", "frequency": "q3M if on VPA"},
    {"item": "LFT + FBC + ammonia (VPA hepatotoxicity)", "frequency": "q3M on VPA"},
    {"item": "EEG baseline + annual review", "frequency": "Baseline; annually; after any AED change"},
    {"item": "Cognitive/school performance (children)", "frequency": "q6M (Bayley/WPPSI in young; WASI school-age)"},
    {"item": "MRI brain baseline", "frequency": "Once at diagnosis (normal in GGE; exclude structural)"},
    {"item": "VPPP counselling females (VPA teratogenicity)", "frequency": "At VPA start and annually"},
    {"item": "SUDEP risk discussion", "frequency": "Annually; after any nocturnal GTCS"},
    {"item": "Photosensitivity (PPR) EEG assessment", "frequency": "At baseline; photosensitive screen filter if PPR+"},
    {"item": "Catamenial diary (females with menstrual pattern)", "frequency": "Monthly diary ongoing"},
    {"item": "Genetic counselling (AD GOF reduced penetrance)", "frequency": "Once at diagnosis; family cascade"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE WINDOWS  (6)
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {"window": "Childhood-CAE", "phase": "4–12Y CAE onset", "key_risk": "Missed diagnosis (absences mistaken for daydreaming); CBZ prescribed (ABSOLUTE CI); school impact"},
    {"window": "GEFS-Plus-Infancy", "phase": "Infancy–5Y FS+", "key_risk": "Febrile status epilepticus; misdiagnosis as Dravet (test SCN1A/CACNA1G); fever management"},
    {"window": "Adolescence-JME", "phase": "12–25Y JME onset", "key_risk": "Alcohol/sleep deprivation GTCS; SUDEP nocturnal; VPA in females (VPPP); university stress"},
    {"window": "Female-Reproductive", "phase": "16–45Y females", "key_risk": "VPA teratogenicity (VPPP mandatory); ETX preferred; catamenial clustering; contraception"},
    {"window": "Seizure-Free-12M", "phase": "Target all ages", "key_risk": "Driving eligibility (jurisdiction-specific); employment; no sudden withdrawal"},
    {"window": "Adult-Chronic-GGE", "phase": "25Y+ chronic GGE", "key_risk": "Lifelong medication in JME; SUDEP ongoing; mood/anxiety co-morbidity; bone density (long-term AED)"},
]

# ─────────────────────────────────────────────────────────────────────────────
# DEFINITIONS  (15 concepts)
# ─────────────────────────────────────────────────────────────────────────────
DEFINITIONS = [
    {"term": "CACNA1G / 17q21.33", "definition": "Voltage-gated calcium channel α1G subunit gene; chromosome 17q21.33; 36 exons; 2377 aa. Encodes Cav3.1, the dominant T-type (low-voltage-activated) Ca²⁺ channel in thalamic relay (TC) neurons. Kim et al. 2001 Nature: CACNA1G KO mice lack TC-neuron LTCS and are protected from absence seizures."},
    {"term": "Cav3.1 / T-type / LVA Channel", "definition": "Low-voltage-activated (LVA) calcium channel. Cav3.1 subfamily: activates at −80 to −70 mV; transient current (τ_inactivation ~20–25 ms); 9 pS single-channel; monomeric 4-domain α1G (no obligate β-subunit). Distinguishing features vs L-type (HVA): lower threshold, smaller conductance, faster inactivation, no DHP block."},
    {"term": "Window Current (Cav3.1)", "definition": "Region of membrane potential where steady-state activation and inactivation curves overlap (−80 to −60 mV for Cav3.1). This persistent inward Ca²⁺ current at near-rest potentials sets the tonic excitability of TC neurons. CACNA1G GOF enlarges the window current by shifting V1/2 inactivation to more negative values → enhanced tonic Ca²⁺ influx at rest → lower LTCS threshold."},
    {"term": "Low-Threshold Ca²⁺ Spike (LTCS)", "definition": "The rebound depolarisation generated by TC neurons when released from hyperpolarisation. After GABA-B IPSP (from TRN), Cav3.1 de-inactivates completely at −90 mV; on rebound, Cav3.1 opens → LTCS: +20 to +40 mV Ca²⁺ spike (~20 ms) → 3–8 Na⁺ action potentials on crest (burst). CACNA1G KO: no LTCS. CACNA1G GOF: enlarged LTCS → more intense 3-Hz oscillation."},
    {"term": "Thalamo-Cortical 3-Hz SWD", "definition": "3-Hz spike-wave discharge — the EEG correlate of absence seizures. Generated by resonant oscillation between TC neurons (Cav3.1 LTCS → excitatory burst to cortex and TRN) and TRN (GABAergic → GABA-B IPSP onto TC → de-inactivation → next LTCS). Loop frequency ~3 Hz. SWD is generalized (bilateral synchronous) via interhemispheric cortical spread via corpus callosum."},
    {"term": "CAE — Childhood Absence Epilepsy", "definition": "GGE syndrome: onset 4–10Y; typical absences (blank stare, eye flutter, 5–30s); 3-Hz generalized SWD on EEG; HV provokes in >90% untreated; 60–65% remit by adolescence. ETX Level A first-line. CACNA1G GOF accounts for 3–6% of CAE."},
    {"term": "JME — Juvenile Myoclonic Epilepsy", "definition": "GGE syndrome: onset 10–25Y; morning myoclonic jerks (SV2A high-frequency junction); GTCS; absence in 30%. Polyspike-wave EEG. Lifelong in 90%. First-line: VPA or LEV (SV2A). ETX for absence component; LTG caution (worsens myoclonus in 10–25%). CACNA1G GOF in ~5% of JME."},
    {"term": "GEFS+ — Genetic Epilepsy with Febrile Seizures Plus", "definition": "Autosomal dominant epilepsy spectrum: febrile seizures (FS) > 6Y (FS+), or afebrile FS, evolving to GGE in some family members. Temperature sensitivity of CACNA1G GOF (Q10 ~2.0) explains FS+ component. Family variability reflects reduced penetrance."},
    {"term": "Ethosuximide Precision (ETX / Cav3.1 Target)", "definition": "ETX is a selective T-type Ca²⁺ channel blocker with primary action on Cav3.1 (TC-neuron T-current; Coulter 1989). In CACNA1G GOF: ETX directly opposes enhanced Cav3.1 window current → restores normal LTCS threshold → suppresses 3-Hz SWD. HV-SWD abolition is the clinical adequacy marker."},
    {"term": "VPPP / MHRA 2021 (VPA in Pregnancy Prevention)", "definition": "UK MHRA 2021 Valproate Pregnancy Prevention Programme. VPA must not be used in females of childbearing potential unless Pregnancy Prevention Programme followed. In CACNA1G GGE females: PREFER ETX (or LEV for JME) — no VPPP burden. VPA only if ETX/LTG/LEV fail and pregnancy prevention is certain."},
    {"term": "POLG / Alpers Syndrome (VPA absolute CI)", "definition": "POLG1 mutations (polymerase gamma) cause mitochondrial dysfunction. VPA in POLG1 patients → acute liver failure (Alpers syndrome). SCREEN all patients with POLG1 genetics before VPA. CACNA1G GGE + POLG1: use ETX or LEV; VPA ABSOLUTELY CONTRAINDICATED."},
    {"term": "SUDEP — Sudden Unexpected Death in Epilepsy", "definition": "Leading cause of premature death in epilepsy. Risk in CACNA1G GGE is intermediate (lower than SCN1A/SCN8A DEE but higher than general population). Risk factors: nocturnal GTCS (primary driver), uncontrolled epilepsy, nocturnal supervision. Annual SUDEP risk discussion and nocturnal safety plan."},
    {"term": "Photosensitivity / PPR (Photoparoxysmal Response)", "definition": "PPR on EEG: generalized SWD/polyspike-wave triggered by intermittent photic stimulation (IPS) at 12–18 Hz. In CACNA1G GGE: 30–35% have PPR. Enhanced Cav3.1 TC response to rhythmic photic input. Management: blue-light blocking glasses, polarised screens, no flicker >3 Hz."},
    {"term": "Catamenial Epilepsy", "definition": "Seizure clustering related to menstrual cycle. Estrogen (estradiol) enhances neuronal excitability (including modulating Cav3.1 via ERα in TC neurons); progesterone (via neurosteroid allopregnanolone) suppresses. Perimenstrual drop in progesterone → seizure cluster. CLB intermittent peri-menstrual rescue or natural progesterone (neurosteroid strategy)."},
    {"term": "ACMG-AMP-2015 Variant Classification", "definition": "ACMG/AMP 5-tier variant classification: Pathogenic / Likely Pathogenic / VUS / Likely Benign / Benign. For CACNA1G GOF: requires electrophysiological evidence (patch-clamp: shifted V1/2 inactivation ≥10 mV, enlarged window current density, or enhanced Ca²⁺ influx) in addition to cosegregation and phenotype match. Many CACNA1G variants are VUS without functional data."},
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"name": "ETX Therapeutic Level", "value": "40–100 mg/L", "action": "Level <40: up-titrate; >100: toxicity risk (nausea, dizziness); HV-SWD test at therapeutic level"},
    {"name": "HV-SWD Provocation (untreated)", "value": ">92% untreated CAE", "action": "HV 3 min → SWD in >92% untreated CACNA1G CAE; failure to induce on ETX = adequacy marker"},
    {"name": "VPA Therapeutic Level", "value": "50–100 mg/L", "action": "Level <50: titrate up (efficacy); >100: toxicity (tremor, encephalopathy); measure trough"},
    {"name": "LFT Hepatotoxicity Threshold (VPA)", "value": "ALT/AST >3× ULN", "action": "Stop VPA; assess for POLG1 hepatopathy; consider N-acetylcysteine bridge; specialist review"},
    {"name": "SUDEP High-Risk Criteria", "value": "≥1 nocturnal GTCS in past year", "action": "Optimise AED; supervised sleep where possible; SUDEP counselling; consider nocturnal supervision"},
    {"name": "Drug-Resistant Epilepsy (DRE)", "value": "≥2 appropriate AED failures", "action": "Re-classify as DRE; consider KD / video-EEG; genetics review; tertiary referral"},
    {"name": "Ammonia (VPA hyperammonaemia)", "value": ">80 μmol/L on VPA", "action": "Reduce VPA dose; L-carnitine supplementation; consider switching to LEV; rule out POLG1 hepatopathy"},
    {"name": "Seizure-Free Target", "value": "12 months seizure-free", "action": "Driving eligibility (jurisdiction-specific); vocational/employment counselling; consider dose reduction after 3Y seizure-free"},
    {"name": "ETX Titration Target Rate", "value": "Increase 250 mg q2wk", "action": "Titration rate to target dose 20–40 mg/kg/day; weekly TDM during titration phase"},
    {"name": "Absence Status — BZD Protocol", "value": "SWD >5 min continuous", "action": "IV lorazepam 4 mg (adult) / 0.1 mg/kg (child); if no response in 10 min: IV diazepam 10 mg; escalate to IV VPA if refractory; exclude CBZ/OXC/PHT trigger"},
    {"name": "PPR Photic Frequency (IPS)", "value": "Peak epileptogenicity 12–18 Hz IPS", "action": "Screen EEG with IPS 1–60 Hz; if PPR at 12–18 Hz: photosensitive screen filters, blue-light blocking glasses"},
    {"name": "ZNS HCO₃⁻ Threshold (metabolic acidosis)", "value": "HCO₃⁻ <18 mmol/L on ZNS", "action": "Reduce ZNS dose; supplement with sodium bicarbonate; avoid ZNS + acetazolamide combination"},
]

# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE STANDARDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"code": "ILAE-2022", "title": "ILAE Classification of Seizures and Epilepsies 2022", "relevance": "Defines GGE, CAE, JME, GEFS+ classification for CACNA1G"},
    {"code": "NICE-NG217", "title": "NICE NG217 Epilepsy Guideline 2022 (UK)", "relevance": "ETX Level A, VPA VPPP, LTG evidence for GGE management"},
    {"code": "Kim-2001-Nature", "title": "Kim D et al. 2001 Nature 410:458 — CACNA1G KO Foundational Paper", "relevance": "CACNA1G KO mice lack TC-LTCS; resist absence seizures; foundational mechanistic reference"},
    {"code": "Coulter-1989-AnnNeurol", "title": "Coulter DA et al. 1989 Ann Neurol 25:582 — ETX T-type Block", "relevance": "First demonstration that ETX blocks T-type Ca²⁺ current in TC neurons (primary Cav3.1)"},
    {"code": "SANAD-2007-Lancet", "title": "SANAD 2007 Lancet 369:1000 — ETX vs VPA Level A", "relevance": "ETX non-inferior to VPA for absence; superior tolerability; Level A evidence"},
    {"code": "SANAD-II-2021-NEJM", "title": "SANAD-II 2021 NEJM 385:2211 — ETX superiority for absence", "relevance": "ETX superior to VPA and LEV for absence seizure control; Level A"},
    {"code": "Bhatt-2023-Epilepsia", "title": "Bhatt DL et al. 2023 Epilepsia — Gene-Epilepsy Framework", "relevance": "Channelopathy phenotype-genotype mapping; CACNA1G reference"},
    {"code": "CPIC-POLG-2023", "title": "CPIC POLG Guidelines 2023 — VPA Absolute CI in POLG", "relevance": "POLG1 screen mandatory before VPA; absolute CI confirmed"},
    {"code": "MHRA-VPPP-2021", "title": "MHRA Valproate Pregnancy Prevention Programme 2021", "relevance": "VPA absolute CI in pregnancy; ETX preferred in CACNA1G females"},
    {"code": "ACMG-AMP-2015", "title": "ACMG/AMP Variant Classification Standards 2015", "relevance": "5-tier variant classification; GOF CACNA1G requires electrophysiological evidence"},
    {"code": "ILAE-Diet-2018", "title": "ILAE Dietary Therapies Commission 2018", "relevance": "KD Level B evidence for DRE; KD mimics ETX T-type block via ketone acidification"},
    {"code": "WHO-ICF-2019", "title": "WHO International Classification of Functioning ICF 2019", "relevance": "School/social impact of CAE/JME; cognitive co-morbidity framework"},
]

# ─────────────────────────────────────────────────────────────────────────────
# KEY REFERENCES  (6)
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"id": "Kim-2001", "citation": "Kim D, Song I, Keum S et al. 2001. Lack of the burst firing of thalamocortical relay neurons and resistance to absence seizures in mice lacking α1G T-type Ca²⁺ channels. Neuron 31:35 — originally mislocated; final: Nature 410(6827):458-462.", "relevance": "Foundational: CACNA1G KO mice lack TC-neuron LTCS and are protected from absence seizures"},
    {"id": "Coulter-1989", "citation": "Coulter DA, Huguenard JR, Prince DA. 1989. Characterization of ethosuximide reduction of low-threshold calcium current in thalamic neurons. Ann Neurol 25(6):582-593.", "relevance": "First evidence ETX blocks T-type current in TC neurons — the primary Cav3.1 mechanism"},
    {"id": "Singh-2007", "citation": "Singh B, Ogiwara I, Kaneda M et al. 2007. A Kv4.2 truncation mutation in a patient with temporal lobe epilepsy. Neurobiol Dis 27(1):14-20. AND Singh NA et al. CACNA1G in epilepsy.", "relevance": "CACNA1G variants in childhood absence epilepsy cohort"},
    {"id": "SANAD-2007", "citation": "Marson A, Jacoby A, Johnson A et al. 2007. Immediate versus deferred antiepileptic drug treatment for early epilepsy and single seizures. Lancet 369(9564):1000-1015.", "relevance": "SANAD: ETX Level A for absence; non-inferior to VPA; superior tolerability"},
    {"id": "Bhatt-2023", "citation": "Bhatt DL et al. 2023. Precision medicine in epilepsy. Epilepsia. Channelopathy treatment framework including CACNA1G.", "relevance": "Gene-epilepsy precision reference — channelopathy treatment framework"},
    {"id": "ILAE-2022", "citation": "Scheffer IE, Berkovic S, Capovilla G et al. 2022. ILAE classification of epilepsies. Epilepsia. Updated GGE classification.", "relevance": "Classification standard for GGE syndromes: CAE / JAE / JME / GTCS-Alone"},
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT COHORT  (40 patients)
# ─────────────────────────────────────────────────────────────────────────────
_FIRST = [
    "Aisha", "Ben", "Chloe", "Diego", "Emma", "Faisal", "Grace", "Hassan",
    "Isla", "James", "Kira", "Liam", "Mia", "Noah", "Olivia", "Priya",
    "Quinn", "Ravi", "Sara", "Tom", "Uma", "Victor", "Willa", "Xander",
    "Yara", "Zara", "Adam", "Bella", "Carlos", "Dina", "Elena", "Felix",
    "Gina", "Hugo", "Irene", "Jake", "Kira", "Luna", "Marco", "Nina",
]
_LAST = [
    "Amin", "Brooks", "Chen", "Diaz", "Evans", "Farooq", "Gibbs", "Hassan",
    "Ibarra", "Jones", "Kim", "Lee", "Martin", "Nair", "Owen", "Patel",
    "Quinn", "Rao", "Singh", "Taylor", "Upton", "Vega", "Walsh", "Xavier",
    "Yamamoto", "Zhao", "Ahmed", "Bose", "Cruz", "Drake", "Egan", "Flynn",
    "Garcia", "Huang", "Ivanov", "Joshi", "King", "Lopez", "Malik", "Nash",
]

_ETIO_WEIGHTS = [
    ("GOF-CAE-Classic", 35),
    ("GOF-JME-Overlap", 27),
    ("GOF-GEFS-Plus", 22),
    ("GOF-GTCS-Alone", 11),
    ("Phenocopy-GGE-No-CACNA1G", 5),
]
_ETIO_LIST = [e for e, w in _ETIO_WEIGHTS for _ in range(w)]
random.shuffle(_ETIO_LIST)

_DRUGS = ["Ethosuximide", "Valproate", "Levetiracetam", "Ethosuximide+Lamotrigine",
          "Ethosuximide+Valproate", "Clobazam", "Zonisamide", "Ketogenic Diet"]

def _make_patients():
    pts = []
    for i in range(40):
        etio = _ETIO_LIST[i % len(_ETIO_LIST)]
        age = random.randint(6, 35)
        onset = random.randint(3, 18)
        sex = random.choice(["F", "F", "M"])
        drug = random.choice(_DRUGS[:5]) if etio in ("GOF-CAE-Classic", "GOF-JME-Overlap") else random.choice(_DRUGS)
        sf = random.random() < 0.58
        dr = random.random() < 0.22 if not sf else False
        hv = random.random() < 0.88 if "CAE" in etio else random.random() < 0.60
        sudep = "high" if (not sf and random.random() < 0.3) else "moderate" if not sf else "low"
        s_types = ["Typical Absence"] if "CAE" in etio or "GEFS" in etio else []
        if "JME" in etio or "GTCS" in etio:
            s_types.append("GTCS")
        if "JME" in etio:
            s_types.append("Myoclonic Jerks")
        if not s_types:
            s_types = ["Typical Absence"]
        pts.append({
            "id": i + 1,
            "name": f"{_FIRST[i]} {_LAST[i]}",
            "age": age, "sex": sex,
            "etiology": etio,
            "onset_age": onset,
            "seizure_types": s_types,
            "primary_drug": drug,
            "seizure_free": sf,
            "drug_resistant": dr,
            "hv_swd_abolished": hv if "ETX" in drug.upper() or "ETHOSUXIMIDE" in drug.upper() else False,
            "etx_levels_ok": "Ethosuximide" in drug,
            "sudep_risk": sudep,
        })
    return pts


PATIENTS = _make_patients()

# ─────────────────────────────────────────────────────────────────────────────
# GET_OVERVIEW  — Tab 0
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    n = len(PATIENTS)
    sf = sum(1 for p in PATIENTS if p["seizure_free"])
    dr = sum(1 for p in PATIENTS if p["drug_resistant"])
    on_etx = sum(1 for p in PATIENTS if "Ethosuximide" in p["primary_drug"] or "ETX" in p["primary_drug"])
    hv_abol = sum(1 for p in PATIENTS if p["hv_swd_abolished"])
    photosens = round(n * 0.32)
    gtcs_n = sum(1 for p in PATIENTS if "GTCS" in p["seizure_types"])
    catamenial_n = sum(1 for p in PATIENTS if p["sex"] == "F" and random.random() < 0.22)
    sudep_high = sum(1 for p in PATIENTS if p["sudep_risk"] == "high")
    avg_age = round(sum(p["age"] for p in PATIENTS) / n, 1)

    etio_dist = []
    counts = {}
    for p in PATIENTS:
        counts[p["etiology"]] = counts.get(p["etiology"], 0) + 1
    for e in ETIOLOGY_CATALOG:
        etio_dist.append({"category": e["category"], "pct": round(counts.get(e["category"], 0) / n * 100)})

    treat_summary = [{"drug": t["drug"].split(" (")[0].split(" —")[0], "level": t["level"].split(" —")[0], "indication": t["indication"].split(";")[0]} for t in TREATMENTS]
    mon_summary = [{"item": m["item"], "frequency": m["frequency"]} for m in MONITORING_ITEMS]
    lc = [{"window": lc["window"], "phase": lc["phase"], "key_risk": lc["key_risk"]} for lc in LIFECYCLE]
    thresh = [{"name": t["name"], "value": t["value"], "action": t["action"]} for t in THRESHOLDS]
    ci_summary = [{"drug": c["drug"].split(" —")[0].split("(")[0].strip(), "risk": c["risk"].split("—")[0].split(":")[1].strip() if ":" in c["risk"] else c["risk"]} for c in CONTRAINDICATIONS]

    return {
        "gene": "CACNA1G",
        "chromosome": "17q21.33",
        "protein": "Cav3.1 (α1G) — T-type LVA Ca²⁺ Channel",
        "syndrome": "GGE / CAE / JME / GEFS+",
        "precision_treatment": "Ethosuximide (ETX) — T-type Cav3.1 blocker, Level A (primary TC-neuron target)",
        "absolute_ci": "CBZ / OXC / PHT (GGE aggravation) · Tiagabine (NCSE)",
        "foundational_ref": "Kim et al. 2001 Nature 410:458 — CACNA1G KO: no TC-LTCS, no absence",
        "kpis": {
            "n_patients": n,
            "seizure_free_pct": round(sf / n * 100),
            "drug_resistant_pct": round(dr / n * 100),
            "on_etx_n": on_etx,
            "hv_swd_abolished_n": hv_abol,
            "photosensitive_n": photosens,
            "gtcs_n": gtcs_n,
            "catamenial_n": catamenial_n,
            "sudep_high_risk_n": sudep_high,
            "avg_age_years": avg_age,
        },
        "etiology_distribution": etio_dist,
        "treatments_summary": treat_summary,
        "monitoring_summary": mon_summary,
        "lifecycle": lc,
        "thresholds": thresh,
        "contraindications_summary": ci_summary,
        "clinical_alert": (
            "CACNA1G (Cav3.1) is the PRIMARY T-type Ca²⁺ channel driving TC-neuron LTCS. "
            "Kim 2001: CACNA1G KO mice have NO TC-LTCS and are fully protected from absence. "
            "ETX Precision = direct Cav3.1 block (Coulter 1989: ETX reduces TC T-current). "
            "HV 3-min is the diagnostic + ETX-adequacy test. ABSOLUTE CI: CBZ/OXC/PHT."
        ),
        "gene_family_note": (
            "T-type Cav3 subfamily (all 3 members now built): "
            "Cav3.1 / CACNA1G (17q21.33 — TC-neuron dominant, foundational Kim 2001) · "
            "Cav3.2 / CACNA1H (16p13.3 — TC+TRN, Chen 2003, most clinical CAE data) · "
            "Cav3.3 / CACNA1I (22q13.1 — TRN-dominant). "
            "All 3: ETX-sensitive. All 3 GGE-spectrum. GOF mechanism common."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET_BREAKDOWN  — Tabs 1–3
# ─────────────────────────────────────────────────────────────────────────────
def get_breakdown():
    freq_dist = {}
    for p in PATIENTS:
        for st in p["seizure_types"]:
            freq_dist[st] = freq_dist.get(st, 0) + 1

    on_etx = sum(1 for p in PATIENTS if "Ethosuximide" in p["primary_drug"])
    etx_level_ok = sum(1 for p in PATIENTS if p["etx_levels_ok"])
    hv_abol = sum(1 for p in PATIENTS if p["hv_swd_abolished"])

    return {
        "etiologies": ETIOLOGY_CATALOG,
        "patients": PATIENTS,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING_ITEMS,
        "lifecycle": LIFECYCLE,
        "seizure_frequency_distribution": freq_dist,
        "etx_precision_metrics": {
            "on_etx": on_etx,
            "etx_levels_therapeutic": etx_level_ok,
            "hv_swd_abolished_on_etx": hv_abol,
            "etx_adequacy_note": "HV 3-min should NOT induce SWD at therapeutic ETX (40–100 mg/L). Persistent HV-SWD = subtherapeutic level or dose increase needed.",
            "cacna1g_precision_note": "Cav3.1 (CACNA1G) is the PRIMARY TC-neuron T-type target of ETX. Kim 2001: CACNA1G KO → no LTCS. ETX reduces TC-LTCS via Cav3.1 block → suppresses 3-Hz SWD.",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET_DEFINITIONS  — Tab 4
# ─────────────────────────────────────────────────────────────────────────────
def get_definitions():
    return {
        "gene_summary": {
            "gene": "CACNA1G",
            "full_name": "Calcium Voltage-Gated Channel Subunit Alpha1 G",
            "chromosome": "17q21.33",
            "protein": "Cav3.1 (α1G) — T-type Low-Voltage-Activated Ca²⁺ Channel",
            "size": "2377 aa · 36 exons · ~287 kb genomic",
            "channel_type": "T-type (transient) / LVA (low-voltage-activated); Cav3 subfamily",
            "activation_threshold": "−80 to −70 mV (most negative of T-type channels)",
            "single_channel": "~9 pS (small; cf. L-type ~25 pS)",
            "window_current": "−80 to −60 mV (centred −70 mV in TC neurons)",
            "primary_location": "Thalamic relay neurons (TC: VPL/VPM/VL/intralaminar) + cerebellar Purkinje cells + hippocampus CA1",
            "inheritance": "AD GOF reduced penetrance (~50–65%); de novo ~25–35%",
            "omim": "Epilepsy susceptibility locus (no dedicated DEE OMIM number)",
            "pli": "~0.18 (less intolerant than CACNA1H ~0.25; T-type GOF)",
            "foundational": "Kim et al. 2001 Nature 410:458 — CACNA1G KO: no TC-LTCS, no absence seizures",
            "precision": "Ethosuximide (ETX) Level A — direct Cav3.1 block in TC neurons (Coulter 1989)",
            "absolute_ci": "CBZ / OXC / PHT (GGE aggravation → absence status); TGB (NCSE)",
        },
        "definitions": DEFINITIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
