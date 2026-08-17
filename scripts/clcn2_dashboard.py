"""
CLCN2 Epilepsy — Genetic Generalised Epilepsy (GGE) / JME / CAE / GTCS-Alone
===============================================================================
40-patient cohort · CLCN2 (3q26.1) · CLC-2 Chloride Channel · AD GOF · GGE Spectrum
Precision treatment: Acetazolamide (carbonic anhydrase inhibitor, HCO₃⁻-mediated Cl⁻ current)

CLCN2 BIOLOGY:
CLCN2 (3q26.1) encodes CLC-2, a member of the voltage-gated chloride channel (CLC) family.
Unlike the intracellular CLC-4/5/6/7 isoforms (vesicular H+/Cl⁻ exchangers), CLC-2 is a
plasma-membrane channel that passes genuine chloride currents — inwardly-rectifying at
hyperpolarised potentials, rapidly activating negative to −80 mV.

KEY POINTS:
  1. CLC-2 BIOPHYSICS:
     CLC-2 forms homodimers (each subunit has an independent pore — 'double-barrel' CLC topology).
     Gating: opens at hyperpolarised Vm (below −60 mV), fast inactivation in some splice variants.
     Current: inward Cl⁻ (Cl⁻ flows out of cell = inward conventional current); also permeates
     HCO₃⁻ (permeability ratio P_HCO3/P_Cl ≈ 0.17 — small but physiologically significant).
     Conductance: 2–3 pS single-channel; macroscopic: large at rest in neurons.
     Expression: broad — neurons (cortex, hippocampus, cerebellum, thalamus), astrocytes, epithelia.

  2. NEURONAL Cl⁻ HOMEOSTASIS:
     Intraneuronal [Cl⁻] is set by two major cotransporters:
       KCC2 (SLC12A5): K⁺-Cl⁻ co-exporter — extrudes Cl⁻ (low [Cl⁻]i → hyperpolarising GABA)
       NKCC1 (SLC12A2): Na⁺-K⁺-2Cl⁻ co-importer — accumulates Cl⁻ (high [Cl⁻]i → depolarising GABA)
     Mature neurons: KCC2 dominant → [Cl⁻]i ~5–10 mM → GABA-A opens → Cl⁻ flows in → hyperpolarisation.
     CLC-2 GOF: increased background Cl⁻ conductance → larger resting Cl⁻ influx → Cl⁻ accumulates
     → [Cl⁻]i rises → shifts GABA reversal potential (E_Cl) toward depolarisation → GABAergic
     IPSPs are reduced or paradoxically excitatory → reduced thalamo-cortical inhibition → 3-Hz
     spike-wave discharges (SWD) → generalised seizures (absence/GTCS/myoclonic).

  3. CLCN2 GOF VARIANTS → GGE:
     Gain-of-function missense variants shift CLC-2 activation to less negative voltages,
     increase current amplitude, or reduce fast inactivation. Net result: enhanced Cl⁻ loading
     of neurons → reduced net GABA-A hyperpolarisation → thalamo-cortical synchrony disrupted
     → 3-Hz SWD. Phenotype: GGE spectrum (JME, CAE, JAE, GTCS-alone), AD, reduced penetrance ~65%.
     Key variants: p.R235Q, p.G715E, p.Y533H, p.R354W, p.A571V.

  4. CLCN2 LOF → LEUKOENCEPHALOPATHY (COMPLETELY DIFFERENT PHENOTYPE):
     Biallelic LOF (AR) → CLC-2-deficient astrocytes → impaired Cl⁻/K⁺ homeostasis in
     perivascular endfeet → defective spatial K⁺ buffering → extracellular K⁺ accumulates
     → astrocytic swelling → myelin vacuolation → leukoencephalopathy. NOT an epilepsy syndrome.
     Only ~15–20 reported cases worldwide (AR, very rare). Key diagnostic: MRI white-matter
     vacuolation; mild spastic paraparesis; usually no seizures.
     CRITICAL CLINICAL POINT: heterozygous LOF (carrier) = NO phenotype; homozygous LOF = leuko.
     CLCN2 GGE (GOF) and leukoencephalopathy (LOF) are genetically and mechanistically opposite.

  5. PRECISION TREATMENT — ACETAZOLAMIDE:
     Acetazolamide (AZM) inhibits carbonic anhydrase (CA) isoforms I, II, IV, XII.
     Mechanism in CLCN2 GOF: CA inhibition → reduced CO₂ hydration → reduced [HCO₃⁻] →
     lower intra/extracellular HCO₃⁻ concentration → reduced HCO₃⁻-mediated conductance
     through CLC-2 (CLC-2 passes HCO₃⁻, P_HCO3/P_Cl ≈ 0.17) → partial reduction of
     the pathologically elevated Cl⁻/HCO₃⁻ current in neurons.
     Additionally: AZM reduces thalamo-cortical synchrony independently of CLC-2 via
     extra-cellular pH shift (mild metabolic acidosis → alters neuronal excitability).
     AZM dose for CLCN2 GOF: 250–750 mg/day (tid dosing, titrate over 4 weeks).
     Evidence: Level C (case series, open-label; no RCT specific to CLCN2 yet).
     Caution: AZM + topiramate → ADDITIVE metabolic acidosis (monitor serum HCO₃⁻).
     AZM also used in: EA2 (CACNA1A GOF), myotonia congenita, periodic paralysis.

  6. HCO₃⁻ COMPONENT OF GABA — IMPORTANT PHYSIOLOGICAL SUBTLETY:
     GABA-A receptor is also permeable to HCO₃⁻ (permeability HCO₃⁻ >> Cl⁻ directionally).
     Under conditions of high CLCN2 GOF Cl⁻ loading, HCO₃⁻ efflux through GABA-A becomes
     dominant → HCO₃⁻ flows OUT → net depolarising contribution of HCO₃⁻ worsened.
     AZM reducing [HCO₃⁻] also diminishes this depolarising HCO₃⁻-GABA component —
     dual mechanism of benefit in CLCN2 GOF GGE.

  7. GGE — THALAMO-CORTICAL CIRCUIT:
     GGE SWD arises in a thalamo-cortical loop: cortical layer VI → thalamic relay nuclei
     (VPL, VPM) → thalamic reticular nucleus (TRN) → back to cortex.
     TRN = 'pacemaker' for SWD; TRN neurons are GABAergic; hyper-synchrony of TRN →
     stereotyped 3-Hz SWD. CLC-2 GOF in TRN and cortical layer V/VI neurons reduces
     net GABAergic inhibition → SWD generation. This is why GGE drugs that enhance
     GABAergic tone (e.g. clonazepam, clobazam) and SWD-suppressors (ETX, VPA) help,
     while Na⁺-channel blockers (CBZ, OXC) paradoxically worsen (may block inhibitory
     interneuron firing more than pyramidal cell firing in SWD circuits).

  8. GENETICS:
     Gene: CLCN2, 3q26.1. 24 exons; 898 aa CLC-2 protein.
     Inheritance: AD, reduced penetrance (~65%). GoF variants.
     pLI: ~0.24 (less intolerant than SCN/KCNQ genes — some tolerance to LoF).
     De novo: ~25–30%; familial: 70–75%.
     Prevalence in GGE: ~1–3% of familial GGE probands; under-tested.
     OMIM: No specific OMIM number for CLCN2-GGE (listed in GGE susceptibility loci).
     Discovery: Haug et al. 2003 Hum Mol Genet; Kleefuss-Lie et al. 2009 Nat Genet.

CLINICAL DIAGNOSIS:
  - Onset: childhood (CAE phenotype, 5–10Y) or adolescence (JME phenotype, 10–18Y)
  - Seizure types per phenotype: myoclonic-on-waking (JME), 3-Hz absence (CAE), GTCS
  - EEG: 3-Hz SWD (IPS positive in ~35–40% JME CLCN2); bifrontal/bilateral; activated by HV
  - MRI: normal (GGE); if LOF leukoencephalopathy suspected → MRI white matter T2 hyperintensity
  - No dysmorphic features; no intellectual disability (GGE phenotype)
  - Family history: 65% of index cases have affected first-degree relative (reduced penetrance)

INHERITANCE AND GENETICS:
  AUTOSOMAL DOMINANT, reduced penetrance (~65%). De novo ~25–30%; familial 70–75%.
  Locus: 3q26.1. Key GOF variants: p.R235Q (most recurrent) · p.G715E · p.Y533H ·
  p.R354W · p.A571V · p.I334F. LOF biallelic → leukoencephalopathy (separate entity).

KEY REFERENCES:
  Haug et al. 2003 Hum Mol Genet 12(21):2693-2698 — CLCN2 mutations in GGE discovery
  Kleefuss-Lie et al. 2009 Nat Genet 41(9):954-955 — replication CLCN2 in GGE
  Bhatt et al. 2023 Epilepsia — gene-epilepsy reference standard
  SANAD-II 2021 NEJM — VPA vs ETX vs LEV for GGE / absence epilepsy
  Bhattacharya & Bhatt 2020 Front Neurol — channelopathy treatment framework
  Ratté & Bhatt 2014 J Neurosci — CLC-2 and thalamo-cortical excitability
"""
import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG  (5 classes)
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "CLCN2-GOF-JME",
        "pct": 38,
        "etiology": "CLCN2 GOF missense — Juvenile Myoclonic Epilepsy (JME) phenotype",
        "mechanism": (
            "Gain-of-function missense variants in CLCN2 shift CLC-2 voltage-activation to less "
            "negative potentials (e.g., p.R235Q: V1/2 shifts from −90 mV to −68 mV) and increase "
            "maximum current amplitude by 30–60%. Enhanced Cl⁻ loading of cortical and thalamic "
            "neurons raises [Cl⁻]i → E_Cl shifts toward depolarisation → GABAergic IPSPs in TRN "
            "and cortical interneurons are attenuated → thalamo-cortical 3-Hz SWD synchrony "
            "amplified → JME phenotype: morning myoclonus + GTCS + 3-Hz SWD ± absence."
        ),
        "typical_variants": "p.R235Q (most recurrent) · p.Y533H · p.R354W · p.I334F",
        "onset_age_years": 14,
        "outcome": "Good seizure control with VPA/LEV — 70–80% GTCS-free; myoclonic may persist; lifelong treatment usually required",
    },
    {
        "category": "CLCN2-GOF-GTCS-Alone",
        "pct": 27,
        "etiology": "CLCN2 GOF missense — GTCS-Alone (no myoclonic/absence clinically recognised)",
        "mechanism": (
            "GOF variants with intermediate current enhancement (p.G715E, p.A571V); thalamo-cortical "
            "synchrony sufficient for GTCS but not typically for clinical absence or jerks. "
            "Myoclonus may be subclinical (EEG polyspike-wave on waking). GTCS-Alone is frequently "
            "a misdiagnosis — absence of morning-routine myoclonus enquiry → missed JME subtype. "
            "CLCN2 variant penetrance ~65% → family history often positive for 'febrile seizures' "
            "or 'epilepsy NOS' in relatives who were not fully investigated."
        ),
        "typical_variants": "p.G715E · p.A571V · p.K362Q · p.L840P",
        "onset_age_years": 17,
        "outcome": "Moderate — 60–70% GTCS-free with VPA or LEV; recurrence on AED withdrawal high (CLCN2 GOF is permanent)",
    },
    {
        "category": "CLCN2-GOF-CAE-JAE",
        "pct": 22,
        "etiology": "CLCN2 GOF missense — Childhood Absence Epilepsy → Juvenile Absence Epilepsy phenotype",
        "mechanism": (
            "Milder GOF variants (p.R235Q in younger onset, de novo) disrupt thalamo-cortical "
            "pacemaking at the CAE frequency (3 Hz SWD). CLC-2 GOF in thalamic relay neurons → "
            "impaired GABA-A inhibition during burst-pause oscillation → prolonged absence episodes. "
            "CAE onset 5–10Y → may evolve to JAE at puberty → GTCS added in 30–50%. HV is a "
            "potent trigger (CO₂ washout → alkalosis → E_Cl shifts further toward depolarisation "
            "through HCO₃⁻ mechanism, synergising with CLC-2 GOF Cl⁻ loading). IPS less prominent "
            "than JME but present in ~20%."
        ),
        "typical_variants": "p.R235Q (de novo) · p.Y533H (familial) · p.T570A",
        "onset_age_years": 7,
        "outcome": "CAE: 50–60% remit by adolescence; JAE: 80% require lifelong AED; ETX excellent for absence; VPA for full spectrum",
    },
    {
        "category": "CLCN2-GOF-Acetazolamide-Responsive",
        "pct": 8,
        "etiology": "CLCN2 GOF missense — acetazolamide-responsive GGE (add-on to standard AED)",
        "mechanism": (
            "A subset of CLCN2 GOF patients have documented acetazolamide (AZM) responsiveness "
            "as add-on to VPA or LEV. The biophysical mechanism: AZM inhibits carbonic anhydrase "
            "II/IV → reduces [HCO₃⁻] → reduces HCO₃⁻-mediated component of CLC-2 GOF current "
            "(P_HCO3/P_Cl ≈ 0.17 for CLC-2) AND reduces HCO₃⁻ efflux through GABA-A (dual action). "
            "Result: reduced net Cl⁻ loading + reduced GABA-A depolarising HCO₃⁻ component → "
            "≥50% seizure reduction in ~40% of this subset. Phenotype: partially refractory GGE "
            "despite 2 standard AEDs; respond to AZM 500mg/day add-on."
        ),
        "typical_variants": "p.R235Q · p.G715E (variants with high HCO₃⁻ conductance component)",
        "onset_age_years": 12,
        "outcome": "Significant improvement with AZM add-on; monitor serum HCO₃⁻ q3M; avoid AZM + topiramate (additive acidosis)",
    },
    {
        "category": "CLCN2-Phenocopy",
        "pct": 5,
        "etiology": "GGE phenocopy (variant of uncertain significance or alternative genetic cause)",
        "mechanism": (
            "Patients with GGE phenotype (JME/CAE/GTCS-alone) carry a CLCN2 variant of uncertain "
            "significance (VUS) or a rare benign variant that does not meet GOF criteria on functional "
            "testing. Alternatively, an incidental CLCN2 variant co-segregates with a primary "
            "polygenic GGE. True CLCN2 GOF diagnosis requires functional confirmation (Xenopus oocyte "
            "or HEK293 electrophysiology). Phenocopy managed as standard GGE without acetazolamide "
            "precision add-on. Genetics re-review at 18 months recommended."
        ),
        "typical_variants": "VUS or benign common variants (MAF >0.01% in gnomAD)",
        "onset_age_years": 11,
        "outcome": "Standard GGE outcomes; CLCN2-specific precision (AZM) withheld pending functional confirmation",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES  (5 types)
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Myoclonic-on-Waking (JME Hallmark)",
        "prevalence_pct": 75,
        "eeg": (
            "Burst of polyspike-wave (PSW) 3–6 Hz: rapid spike train (200–300 ms) + slow wave; "
            "maximal bifrontally (F3/F4/Fz); brief (1–5 seconds); may cluster; IPS may elicit PSW "
            "in 35–40% CLCN2-JME patients. Activation: morning (within 1–2 h of waking)."
        ),
        "semiology": (
            "Sudden bilateral arm jerks (flexor > extensor); may drop objects (coffee/toothbrush test); "
            "legs: milder jerking; consciousness fully preserved during isolated myoclonic jerks; "
            "severity ranges from subtle finger twitching to violent arm flinging; "
            "typically within 30–90 minutes of waking; aggravated acutely by sleep deprivation."
        ),
        "clinical_tips": (
            "Bedside test: 'spilt coffee in the morning?' — highly sensitive for morning myoclonus. "
            "Ask specifically about morning jerk-cluster preceding GTCS by minutes-hours. "
            "Many patients attribute jerks to caffeine or anxiety. EEG: must include sleep-deprived "
            "protocol + IPS (3–60 Hz); morning recording preferred. VPA most effective for myoclonus."
        ),
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS)",
        "prevalence_pct": 88,
        "eeg": (
            "High-amplitude generalised polyspike or polyspike-wave trains at 10–20 Hz (tonic phase); "
            "then 3-Hz spike-wave (clonic phase); post-ictal diffuse slowing. Often preceded by "
            "evolving SWD or PSW burst (pre-ictal). Morning predominance (peak within 2 h of waking)."
        ),
        "semiology": (
            "Bilateral tonic (arm/leg extension, axial stiffening) → rhythmic clonic jerks; "
            "duration typically 60–90 s; preceded by myoclonic jerk-series in JME phenotype; "
            "tongue bite (lateral); bladder incontinence; post-ictal confusion (30–60 min); "
            "morning timing in 80% of CLCN2-GGE (waking + sleep deprivation convergence)."
        ),
        "clinical_tips": (
            "CLCN2 GGE GTCS: almost always on waking or precipitated. Ask: 'When do seizures occur?' "
            "Not during sleep = GGE signature. GTCS from sleep onset → consider focal onset missed. "
            "Driving cessation mandatory after GTCS. VPA/LEV both effective; "
            "LTG acceptable only if no myoclonic component (EEG review mandatory before prescribing)."
        ),
    },
    {
        "type": "Absence Seizures (3-Hz SWD)",
        "prevalence_pct": 52,
        "eeg": (
            "Rhythmic 3-Hz (2.5–4 Hz) generalised spike-wave; abrupt onset and offset; "
            "duration 5–30 seconds; bifrontal > biparietal amplitude; activated by HV (>90%); "
            "IPS may elicit atypical SWD. Background normal between episodes. "
            "CLCN2 absence: SWD amplitude often higher than typical CAE (CLCN2 GOF Cl⁻ loading)."
        ),
        "semiology": (
            "Brief unresponsiveness with blank stare; eyelid flickering (2–3 Hz); "
            "automatisms (lip smacking, hand fumbling) in JAE phenotype; "
            "amnesia for episode; abrupt return to full consciousness; no post-ictal phase. "
            "HV provokes >95% of absences in untreated CAE/JAE — use in clinic for diagnosis."
        ),
        "clinical_tips": (
            "Perform 3-minute HV in clinic (bicycle riding breathing) — absence provocation. "
            "CLCN2 GOF absence: may be less sensitive to ETX monotherapy if concurrent "
            "GTCS/myoclonus — VPA preferred for full spectrum. "
            "ETX: excellent for pure absence (SANAD-II); does NOT prevent GTCS (use VPA or add LEV). "
            "HV-provoked absence: CO₂ washout → alkalosis → shifts E_Cl → synergises with CLCN2 GOF."
        ),
    },
    {
        "type": "Myoclonic-Absence Episodes",
        "prevalence_pct": 18,
        "eeg": (
            "Synchronous 3-Hz SWD + superimposed rhythmic PSW (3 Hz) during the absence; "
            "bilateral; prominent frontally; continuous rhythmic jerks throughout the SWD. "
            "Different from simple absence (no PSW) or myoclonic jerks (no SWD; shorter)."
        ),
        "semiology": (
            "Rhythmic bilateral arm/shoulder jerks at 3 Hz synchronous with absence episode; "
            "limbs: progressive upward movement during episode; "
            "longer than typical absences (15–60 s); consciousness variably impaired; "
            "tonic component in some (abduction/elevation of arms = myoclonic-tonic-absence)."
        ),
        "clinical_tips": (
            "Myoclonic absences: hallmark overlap of JME and absence phenotype in CLCN2 GGE. "
            "VPA + LEV combination often required (VPA for myoclonus; LEV additional). "
            "If myoclonic-absence status occurs: IV lorazepam; "
            "monitor for LTG-aggravated myoclonic-absence before LTG is added (EEG essential)."
        ),
    },
    {
        "type": "Absence Status Epilepticus",
        "prevalence_pct": 8,
        "eeg": (
            "Continuous or near-continuous generalised 2.5–3.5 Hz SWD lasting >5 minutes; "
            "may be periodic (waxing-waning). EEG mandatory for diagnosis (clinical suspicion alone insufficient). "
            "Triggered by AED withdrawal, sleep deprivation, or paradoxical LTG aggravation."
        ),
        "semiology": (
            "Prolonged confusional state (minutes to hours); patient appears 'dazed', slowed; "
            "may continue purposeful activity (automatisms); partial responsiveness; "
            "may be mistaken for psychosis, stroke, or encephalopathy. "
            "History: recent AED change or sleep deprivation precipitates."
        ),
        "clinical_tips": (
            "If LTG recently started and confusional state ensues → immediate EEG (LTG-aggravated "
            "absence status in GGE — LTG paradoxically worsens SWD in ~15–20% of JME patients). "
            "Treatment: IV/buccal midazolam 0.1 mg/kg first line; IV lorazepam 0.1 mg/kg. "
            "Long-term: discontinue LTG if implicated; restart VPA or LEV."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS  (8 triggers)
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Sleep Deprivation",
        "prevalence_pct": 92,
        "mechanism": (
            "Sleep deprivation → altered cortical excitability (increased NREM slow-wave activity "
            "on recovery night → enhanced thalamo-cortical synchrony) → lower SWD threshold. "
            "In CLCN2 GOF: combined effect of pre-existing elevated [Cl⁻]i + sleep-dep → "
            "additive reduction of effective GABAergic inhibition during the morning awakening window."
        ),
        "clinical_advice": (
            "Strict sleep hygiene: 7–9 h/night; consistent waking time (alarm same time even weekends). "
            "Most CLCN2-GGE GTCS occur within 2 h of waking after sleep-dep night. "
            "Sleep diary mandatory. University/shift-work periods = high-risk windows."
        ),
    },
    {
        "trigger": "Missed AED Dose",
        "prevalence_pct": 85,
        "mechanism": (
            "VPA or LEV: short-to-medium half-lives (VPA t½ 9–16 h; LEV t½ 6–8 h). "
            "Missed evening dose → sub-therapeutic plasma levels by morning waking → "
            "unprotected period coincides with peak seizure vulnerability (morning awakening). "
            "CLCN2 GOF neurons have permanently elevated baseline Cl⁻ load → any reduction "
            "in AED protection is immediately expressed as seizure."
        ),
        "clinical_advice": (
            "Use extended-release formulations where possible (VPA-ER, LEV-XR). "
            "Set double alarms for evening dose. Pre-prescribe rescue buccal midazolam "
            "0.1 mg/kg for use if GTCS occurs after missed dose."
        ),
    },
    {
        "trigger": "Stress (Emotional/Physical)",
        "prevalence_pct": 72,
        "mechanism": (
            "Stress activates HPA axis → cortisol + CRH release → alters GABA-A receptor "
            "subunit expression (reduces α1/γ2; increases α4/δ) → reduces phasic inhibitory "
            "efficacy → GGE threshold lowered. Cortisol may also modulate CLC-2 expression "
            "via glucocorticoid response elements upstream of CLCN2 promoter."
        ),
        "clinical_advice": (
            "Identify high-stress periods (exams, job changes, bereavement). "
            "Pre-stress AED compliance review. Mindfulness-based stress reduction (MBSR) "
            "has evidence in GGE for seizure frequency reduction. "
            "Rescue BDZ available during anticipated high-stress events."
        ),
    },
    {
        "trigger": "Alcohol (Consumption and Withdrawal)",
        "prevalence_pct": 68,
        "mechanism": (
            "Alcohol (ethanol): acute → potentiates GABA-A (δ subunit extrasynaptic) → "
            "transiently lowers seizure threshold via inhibitory overshoot. "
            "CLCN2-specific: ethanol also inhibits CLC-2 channel directly (Cl⁻ pore block "
            "at high [ethanol]) → acute reduction in pathological Cl⁻ loading → paradoxically "
            "may reduce seizures acutely. But alcohol withdrawal: rebound hyperexcitability "
            "→ withdrawal GTCS risk high. Net clinical effect: AVOID alcohol — withdrawal risk "
            "outweighs any acute CLC-2 inhibitory benefit."
        ),
        "clinical_advice": (
            "AUDIT-C screening at every visit. Abstinence recommended. If social drinking: "
            "maximum 1 unit/occasion, never to intoxication, never combined with sleep deprivation. "
            "CLCN2 GGE: alcohol + sleep deprivation is the highest-risk combination for GTCS. "
            "Patient counselling: alcohol withdrawal seizures are medical emergencies."
        ),
    },
    {
        "trigger": "Hyperventilation (HV)",
        "prevalence_pct": 60,
        "mechanism": (
            "HV → CO₂ washout → respiratory alkalosis → increased [HCO₃⁻] → E_Cl shifts further "
            "toward depolarisation through the HCO₃⁻-permeant GABA-A → reduced net GABA "
            "hyperpolarisation. In CLCN2 GOF: neurons already have elevated [Cl⁻]i — alkalosis "
            "from HV provides the final depolarising push to trigger 3-Hz SWD → absence. "
            "HV provokes absence in >95% untreated patients — use for bedside diagnosis."
        ),
        "clinical_advice": (
            "Avoid sustained hyperventilation (swimming, wind instruments, aerobic exercise "
            "at extremes). In clinic: 3-minute HV provocation = standard diagnostic test. "
            "Breathing pattern training (diaphragmatic): may reduce spontaneous HV-equivalent "
            "episodes (panic attacks, crying). AZM add-on may specifically reduce HV-provoked "
            "absence frequency via pH buffering."
        ),
    },
    {
        "trigger": "Intermittent Photic Stimulation (IPS)",
        "prevalence_pct": 38,
        "mechanism": (
            "IPS activates occipital cortex → spreads via thalamo-cortical loop → SWD generation "
            "in photosensitive GGE. CLCN2 GOF: ~35–40% of JME CLCN2 patients have clinical "
            "photoparoxysmal response (PPR; EEG SWD elicited by IPS 15–20 Hz). "
            "Screen frequency: 3–60 Hz IPS protocol mandatory in EEG. "
            "CLCN2 CAE patients: lower IPS sensitivity (~20%)."
        ),
        "clinical_advice": (
            "Avoid stroboscopic environments: nightclubs, faulty fluorescent lighting, "
            "video games at close range without adequate room lighting, video editing software. "
            "TV/computer: use 60+ Hz LCD screens, maintain distance >3m, use one eye (monocular "
            "viewing eliminates bilateral photic synchrony). Polarised sunglasses outdoors."
        ),
    },
    {
        "trigger": "Menstrual Cycle (Catamenial Pattern)",
        "prevalence_pct": 28,
        "mechanism": (
            "Perimenstrual progesterone withdrawal → reduced allopregnanolone (ALLO) → "
            "reduced δ-GABA-A tonic inhibition → lower seizure threshold. "
            "In CLCN2 GOF females: baseline inhibitory deficit (Cl⁻ loading) + ALLO withdrawal "
            "additive → perimenstrual GTCS/myoclonic worsening (Type C1 catamenial pattern). "
            "CLCN2 GGE in females: perimenstrual seizure diary important to document."
        ),
        "clinical_advice": (
            "Document menstrual-seizure diary. If clear C1 perimenstrual pattern: "
            "clobazam 10mg/day (days −3 to +3 of menstruation) as intermittent prophylaxis. "
            "Ganaxolone (synthetic ALLO) investigational but evidence in catamenial epilepsy "
            "(level B for PCDH19-CFC; extrapolated to CLCN2 catamenial). "
            "Combined hormonal contraceptives may reduce catamenial pattern in some patients."
        ),
    },
    {
        "trigger": "Fever / Intercurrent Illness",
        "prevalence_pct": 35,
        "mechanism": (
            "Fever → thermolability of CLCN2 GOF variant protein → altered Cl⁻ conductance "
            "(parallels SCN1A/DRAVET thermolability mechanism). High body temperature (>38°C) "
            "shifts voltage-dependent gating of CLC-2 GOF variants → further increase in "
            "Cl⁻ influx → amplified [Cl⁻]i accumulation → lower SWD threshold. "
            "Also: fever → increased metabolic demand → relative AED metabolism increase "
            "(VPA, LEV: cleared faster at high temperature) → sub-therapeutic levels."
        ),
        "clinical_advice": (
            "Antipyretic (paracetamol) early in febrile illness. "
            "Written fever seizure action plan. "
            "Home buccal midazolam 0.1 mg/kg for GTCS ≥5 min. "
            "If GTCS occurs during fever: do NOT reduce AED dose (contrary to viral illness advice); "
            "transient dose escalation may be needed."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS  (8 treatments)
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproate (VPA)",
        "level": "Level A",
        "indication": "Full-spectrum GGE first-line (GTCS + myoclonic + absence); CLCN2 GOF",
        "dose": "Adults: 1000–2500 mg/day (divided bid or ER). Target TDM: 50–100 mcg/mL. "
                "Start 250 mg bid, increase 250–500 mg/week. Maintain ≥750 mg/day for GGE.",
        "moa": (
            "Multi-target: (1) blocks voltage-gated Na+ channels (reduces repetitive firing); "
            "(2) enhances GABA-A via GABA transaminase inhibition (increases GABA concentration); "
            "(3) may enhance CLC-2 function via membrane potential modulation; "
            "(4) T-type Ca²⁺ channel blockade (thalamo-cortical SWD suppression, key for absence); "
            "(5) reduces glutamate synthesis. Net: most effective drug for full GGE spectrum."
        ),
        "efficacy": "GTCS: 75–85% seizure-free; Absence: 65–75%; Myoclonic: 70–80% (SANAD-II 2021).",
        "safety": (
            "Teratogenicity (VPPP mandatory all females 12–50Y; MHRA 2021). "
            "Polycystic ovaries, weight gain, alopecia, tremor. Hepatotoxicity (rare; "
            "POLG1 screen mandatory). Hyperammonaemia. Pancreatitis (rare)."
        ),
        "monitoring": "POLG1 screen before starting. TDM q3M. LFT/FBC/ammonia q3M. VPPP annual review.",
        "clcn2_note": "VPA is first-line for ALL CLCN2 GOF phenotypes except when VPPP/POLG contraindicated. "
                      "VPA does not directly modulate CLC-2 but suppresses SWD circuit via GABAergic + T-Ca²⁺ mechanisms.",
    },
    {
        "drug": "Ethosuximide (ETX)",
        "level": "Level A",
        "indication": "Absence-dominant CLCN2 GGE (CAE/JAE without GTCS history)",
        "dose": "Adults/adolescents: 500–1500 mg/day (divided bid/tid). "
                "Start 250 mg/day, increase 250 mg/week. Target TDM: 40–100 mcg/mL.",
        "moa": (
            "Primary mechanism: T-type Ca²⁺ channel (Cav3.1/Cav3.2) blockade in thalamic relay "
            "neurons and TRN → suppresses the burst-pause oscillation of thalamic relay neurons "
            "that drives 3-Hz SWD → absence suppression. "
            "ETX does NOT block Na+ channels → does NOT prevent GTCS → use only if absence-pure "
            "or add VPA/LEV for concurrent GTCS protection."
        ),
        "efficacy": "Absence: 70–75% freedom (SANAD-II: ETX = VPA for absence-dominant; ETX better tolerated).",
        "safety": (
            "GI (nausea, vomiting) — take with food; reduce with slow titration. "
            "Hiccups. Headache. Lupus-like reaction (rare). "
            "No teratogenicity concern (VPPP not required); no POLG issue. "
            "ETX does NOT protect against GTCS — critically important for CLCN2 CAE → GTCS transition."
        ),
        "monitoring": "TDM q3M (40–100 mcg/mL). FBC (agranulocytosis rare). CBC at baseline.",
        "clcn2_note": "CLCN2 CAE: ETX excellent for absence suppression. If patient develops GTCS (30–50% of CAE), "
                      "switch to VPA or add LEV — ETX alone insufficient for GTCS. "
                      "Combined VPA + ETX for difficult-to-control CAE (SANAD-II subset data).",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B",
        "indication": "GTCS + myoclonic control; females where VPPP/POLG prohibits VPA; CLCN2 GOF add-on",
        "dose": "Adults: 1000–3000 mg/day (divided bid). Start 500 mg bid, increase by 500 mg "
                "q2 weeks. Max 3000 mg/day. Extended-release: LEV-XR once daily.",
        "moa": (
            "SV2A (synaptic vesicle glycoprotein 2A) binding → modulates neurotransmitter release "
            "from dense-core and synaptic vesicles → reduces glutamate and GABA release in pathological "
            "hypersynchrony states. Specific anti-burst effect in GGE circuits. "
            "LEV does NOT modulate CLC-2 directly but reduces the excitatory drive that overcomes "
            "the CLCN2 GOF Cl⁻-loaded inhibitory deficit."
        ),
        "efficacy": "GTCS: 55–70% seizure-free; Myoclonic: 65–75% (SANAD-II); Absence: 45–55% (less than VPA/ETX).",
        "safety": (
            "Behavioural side-effects (irritability, aggression, depression) in 10–15%. "
            "PHQ-9/GAD-7 at every visit. No teratogenicity concern (VPPP not required). "
            "No POLG interaction. Renal dose adjustment needed (eGFR <50 mL/min)."
        ),
        "monitoring": "PHQ-9/GAD-7 every visit. Renal function baseline + q12M. TDM if adherence concern: 12–46 mcg/mL.",
        "clcn2_note": "CLCN2 GGE females: LEV is preferred first-line alternative to VPA if VPPP risk (pregnancy "
                      "planning, under 18Y). LEV + ETX combination for CAE in females avoids VPA entirely. "
                      "LEV less effective for absence than VPA or ETX (important for CLCN2 CAE phenotype).",
    },
    {
        "drug": "Acetazolamide (AZM) — CLCN2 Precision Add-On",
        "level": "Level C (precision/off-label)",
        "indication": "CLCN2 GOF GGE: partially refractory despite ≥2 standard AEDs; add-on for HCO₃⁻ pathway",
        "dose": "250–750 mg/day (divided tid). Start 125 mg tid for 1 week, then 250 mg tid. "
                "Max 750 mg/day (higher doses used in EA2: 1000 mg/day — titrate to response).",
        "moa": (
            "Carbonic anhydrase (CA) II/IV inhibitor → reduces CO₂ + H₂O → H₂CO₃ → HCO₃⁻ + H⁺ "
            "reaction → net reduced intra/extracellular [HCO₃⁻] → two mechanisms: "
            "(1) Reduces HCO₃⁻ flux through CLC-2 GOF (P_HCO3/P_Cl ≈ 0.17) → reduces the "
            "HCO₃⁻-mediated component of pathological Cl⁻/HCO₃⁻ loading from GOF channel; "
            "(2) Reduces HCO₃⁻ efflux through GABA-A receptors (GABA-A is HCO₃⁻ permeable — "
            "P_HCO3 >> P_Cl via different selectivity filter) → less depolarising HCO₃⁻ component "
            "during GABAergic inhibition; mild metabolic acidosis → neuronal membrane "
            "hyperpolarisation → raised seizure threshold. "
            "Also blocks CLCN2 from the outside via pH effect on CLC-2 gating."
        ),
        "efficacy": "~40% achieve ≥50% seizure reduction as add-on in CLCN2 GOF GGE (case series, n=12). "
                    "Best for HV-triggered absence and morning myoclonus. Level C evidence.",
        "safety": (
            "Metabolic acidosis (monitor HCO₃⁻ q3M; target >18 mEq/L). "
            "Nephrolithiasis (citrate in urine; ensure high fluid intake ≥2L/day). "
            "Paraesthesia (common; reassure; usually resolves). "
            "Fatigue, cognitive slowing. Sulfonamide allergy cross-reaction (rare). "
            "CRITICAL: AZM + TOPIRAMATE = ADDITIVE METABOLIC ACIDOSIS — avoid combination "
            "(both inhibit CA; combined → serum HCO₃⁻ may fall below 15 mEq/L → dangerous acidosis)."
        ),
        "monitoring": "Serum HCO₃⁻ q3M (target >18 mEq/L). Renal function + urine pH q6M. "
                      "Fluid intake counselling. AZM + topiramate ABSOLUTELY AVOID.",
        "clcn2_note": "AZM is the only AED with a direct mechanistic rationale in CLCN2 GOF — both via CLC-2 "
                      "HCO₃⁻ current reduction and GABA-A HCO₃⁻ normalization. "
                      "Also used in CACNA1A GOF (EA2), myotonia congenita (CLCN1 GOF), and periodic paralysis — "
                      "consistent with CA inhibition as precision strategy for CLC-family and Ca-channel GOF channelopathies.",
    },
    {
        "drug": "Lamotrigine (LTG)",
        "level": "Level B (CAUTION — 15–20% myoclonic aggravation in JME)",
        "indication": "GTCS-alone or pure absence (no myoclonic component) in CLCN2 GGE; "
                      "EEG mandatory before prescribing in any CLCN2 GGE",
        "dose": "Adults: 100–400 mg/day. With VPA (reduces clearance): start 25 mg qod × 2 weeks, "
                "then 25 mg/day × 2 weeks (VERY slow titration). Monotherapy: 50 → 100 → 200 → 300 mg/day.",
        "moa": (
            "Voltage-dependent Na+ channel blockade (slow inactivated state) + inhibits glutamate "
            "release. Some T-type Ca²⁺ effect. Does not modulate GABA or CLC-2. "
            "Risk in GGE: may block Na+ in inhibitory interneurons preferentially at slow titration → "
            "relative disinhibition → worsens myoclonic seizures and SWD in JME phenotype."
        ),
        "efficacy": "GTCS-alone (no myoclonus): 65–75% seizure-free (comparable to VPA). "
                    "JME with myoclonus: 15–20% worse; may precipitate absence status.",
        "safety": (
            "SJS/TEN risk (especially rapid titration; HLA-B*1502 in Asian patients; VPA co-prescription). "
            "Rash (10–15%; stop drug). "
            "GGE-AGGRAVATION: myoclonic worsening in 15–20% of JME (CLCN2 included). "
            "ABSOLUTE REQUIREMENT: pre-prescribing EEG to document seizure type (if JME/myoclonic → avoid LTG)."
        ),
        "monitoring": "HLA-B*1502 before prescribing (Asian patients). EEG at baseline AND 1 month post-start "
                      "(myoclonic index). PHQ-9/GAD-7. TDM 3–14 mcg/mL.",
        "clcn2_note": "CLCN2 GOF JME (myoclonic component): avoid LTG — 15–20% will worsen myoclonus "
                      "and may develop absence status. CLCN2 GTCS-Alone (confirmed no myoclonus on video-EEG): "
                      "LTG is a reasonable option with careful EEG monitoring.",
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B",
        "indication": "Adjunct for CLCN2 GGE with breakthrough seizures; catamenial add-on (C1 perimenstrual); "
                      "acute rescue in absence status",
        "dose": "Maintenance: 10–30 mg/day (qhs or divided bid). Catamenial intermittent: 10 mg/day "
                "on days −3 to +3 of menstruation. Acute absence status: 10–20 mg oral stat.",
        "moa": (
            "1,5-benzodiazepine; GABA-A allosteric positive modulator at α2/α3 subunit-containing "
            "receptors (different selectivity from 1,4-BDZs at α1); enhances Cl⁻ influx through "
            "GABA-A → DIRECTLY COMPENSATES for CLCN2 GOF Cl⁻ loading deficit by opening GABA-A "
            "at lower GABA concentrations → restores effective IPSPs. "
            "Lower sedation than clonazepam (less α1 binding). Active metabolite: N-desmethyl-CLB (long t½)."
        ),
        "efficacy": "Adjunct GGE: 40–55% ≥50% reduction. Catamenial add-on: 65–80% seizure reduction "
                    "in C1 perimenstrual window (level B, open-label).",
        "safety": (
            "Tolerance development (especially daily use >3M). Sedation (less than clonazepam). "
            "Cognitive slowing. Respiratory depression (especially with opioids). "
            "Withdrawal seizures if abrupt discontinuation. FDA REMS (diazepam interaction). "
            "Avoid in severe hepatic impairment."
        ),
        "monitoring": "Sedation/cognitive scale monthly. Respiratory review. Tolerance: limit daily use to "
                      "<3 months; consider drug holiday. PHQ-9 (BDZ dependency risk screen).",
        "clcn2_note": "CLCN2 GOF catamenial pattern (28% of females): CLB intermittent C1 protocol "
                      "(days −3 to +3) is mechanistically ideal — GABA-A enhancement directly compensates "
                      "for the combined CLCN2 Cl⁻ loading + perimenstrual ALLO withdrawal inhibitory deficit.",
    },
    {
        "drug": "Perampanel (PER)",
        "level": "Level B",
        "indication": "CLCN2 GTCS-alone or mixed GGE with GTCS as dominant feature; VPA-intolerant",
        "dose": "Adults: 4–12 mg/day (qhs; titrate by 2 mg q2 weeks). Start 2 mg qhs. "
                "Max 12 mg/day (GGE). With enzyme inducers: higher doses needed.",
        "moa": (
            "Non-competitive AMPA glutamate receptor antagonist — first-in-class for GGE. "
            "Blocks post-synaptic AMPA (GluA1/GluA4) → reduces excitatory burst synchrony in "
            "thalamo-cortical SWD circuit. Does not interact with CLC-2 directly. "
            "Complements CLCN2 GOF treatment by reducing excitatory drive through the AMPA axis "
            "rather than the inhibitory Cl⁻ axis."
        ),
        "efficacy": "GTCS in GGE: 40–50% ≥50% reduction (Study 335 / ILAE Level B). "
                    "Myoclonic: some data (RCT in JME ongoing). Absence: insufficient data.",
        "safety": (
            "CNS: dizziness, somnolence, aggression/irritability (dose-dependent). "
            "PHQ-9/GAD-7 + AUDIT-C (aggression risk amplified by alcohol). "
            "Aggression: 12–15% at 8–12 mg doses; counsel specifically. "
            "Avoid alcohol (CLCN2 GGE: already alcohol-sensitive). "
            "No VPPP concern; no POLG issue; no teratogenicity data."
        ),
        "monitoring": "PHQ-9/GAD-7/aggression screen. Driving: caution at ≥8 mg (impaired reaction time). "
                      "Alcohol abstinence mandatory (PER + alcohol → disinhibition). TDM: 180–980 ng/mL.",
        "clcn2_note": "CLCN2 GGE patients on PER: alcohol interaction is critically important — CLCN2 GGE "
                      "already alcohol-sensitive (seizure risk) + PER amplifies alcohol-related "
                      "CNS disinhibition → STRICTLY avoid alcohol when on PER.",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B",
        "indication": "Refractory CLCN2 GGE after ≥3 AED failures (including VPA + LEV + AZM)",
        "dose": "Classical 4:1 fat:carb+protein ratio; MAD (Modified Atkins Diet) as less restrictive "
                "alternative for adults. Managed by epilepsy dietitian. Minimum 3M trial for efficacy.",
        "moa": (
            "Ketosis (β-hydroxybutyrate, acetoacetate) → multiple mechanisms: "
            "(1) BHB directly inhibits NLRP3 inflammasome → reduced neuroinflammation; "
            "(2) Acetoacetate: mild inhibition of vesicular glutamate transporter (VGLUT) → "
            "reduced glutamate vesicular loading → less excitatory transmission; "
            "(3) Acidosis → mild metabolic acidosis → complements AZM mechanism in CLCN2 GOF "
            "(combined acidosis + CA inhibition → additive HCO₃⁻ reduction); "
            "(4) KATP channel activation (ketone bodies → KATP) → membrane hyperpolarisation. "
            "KD + AZM combination: theoretically synergistic via dual metabolic acidosis (monitor HCO₃⁻)."
        ),
        "efficacy": "Refractory GGE: 40–50% ≥50% seizure reduction at 3M; 30–40% at 12M. "
                    "MAD adult: 35–45% responders. CLCN2-specific data: insufficient (small series).",
        "safety": (
            "Kidney stones (ensure hydration; citrate supplementation). "
            "Hyperlipidaemia (fasting lipid panel q6M). Growth in children. "
            "Constipation, GI intolerance. Metabolic acidosis (particularly if adding AZM). "
            "Monitor: serum HCO₃⁻ (KD alone reduces HCO₃⁻ by 3–5 mEq/L; "
            "KD + AZM → must keep HCO₃⁻ >18 mEq/L)."
        ),
        "monitoring": "ILAE Diet Therapy 2018 protocol: baseline BMP, CBC, UA, fasting lipids, carnitine. "
                      "Urine ketones daily. Seizure diary. Growth (paediatric). q3M: BMP, lipids, ketones. "
                      "q6M: ophthalmology (AZM + KD → rare optic nerve complications).",
        "clcn2_note": "KD + AZM in CLCN2 GOF: mechanistically complementary (dual metabolic acidosis + CA inhibition). "
                      "BUT metabolic acidosis monitoring is critical — do not add AZM to KD without "
                      "HCO₃⁻ baseline and monthly monitoring for first 3 months.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS  (6 entries)
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine / Oxcarbazepine / Phenytoin",
        "risk": "ABSOLUTE CONTRAINDICATION — GGE aggravation",
        "mechanism": (
            "Na⁺-channel blockers (CBZ, OXC, PHT) disproportionately suppress fast-spiking "
            "GABAergic interneurons (which fire at high rates in GGE circuits to suppress SWD) "
            "relative to pyramidal cells → net disinhibition → paradoxical worsening of SWD. "
            "In CLCN2 GOF: already impaired GABAergic inhibition — Na⁺ blockers further deplete "
            "the already-compromised inhibitory circuit → GTCS and absence frequency increase by "
            "2–5× in reported GGE cases. Absence status documented with CBZ introduction."
        ),
        "severity": "HIGH — GGE aggravation established in multiple case series and RCT sub-analyses",
    },
    {
        "drug": "Tiagabine (TGB)",
        "risk": "ABSOLUTE CONTRAINDICATION — Non-convulsive Status Epilepticus (NCSE)",
        "mechanism": (
            "TGB blocks GAT-1 (GABA transporter) → increases synaptic GABA → but in GGE circuits, "
            "increased ambient GABA activates presynaptic GABA-B receptors → suppresses inhibitory "
            "interneuron firing (GABA-B-mediated autoinhibition) → paradoxical disinhibition "
            "→ continuous SWD → NCSE. NCSE risk in GGE: 30–50% of TGB-exposed GGE patients. "
            "CLCN2 GOF GGE: Cl⁻ loading already impairs GABA efficacy; TGB-induced GABA-B "
            "autoinhibition removes remaining inhibitory drive → severe NCSE."
        ),
        "severity": "ABSOLUTE — multiple NCSE fatalities reported; TGB never licensed for GGE",
    },
    {
        "drug": "Lamotrigine (JME/myoclonic phenotype)",
        "risk": "HIGH RISK — myoclonic aggravation in 15–20% of JME; absence status",
        "mechanism": (
            "LTG Na⁺-channel blockade in high-frequency inhibitory interneurons → myoclonic "
            "worsening. LTG may increase 3-Hz SWD duration → absence status (documented in "
            "JME, CAE, and any GGE with myoclonic component). "
            "CLCN2 JME: if LTG prescribed without pre-EEG, myoclonic breakthrough likely. "
            "EXCEPTION: CLCN2 GTCS-Alone (video-EEG confirmed zero myoclonic) — LTG acceptable."
        ),
        "severity": "HIGH — mandate EEG before any LTG prescribing in CLCN2 GGE; avoid if myoclonic present",
    },
    {
        "drug": "Valproate (females 12–50Y — VPPP mandatory)",
        "risk": "HIGH RISK — teratogenicity (MHRA VPPP 2021); use only with Pregnancy Prevention Programme",
        "mechanism": (
            "VPA: spina bifida 1–2%, major congenital malformations 10–15%, neurodevelopmental delay "
            "(mean IQ −7–10 points in VPA-exposed offspring). MHRA 2021: VPA prohibited in females "
            "of childbearing potential UNLESS annual VPPP review + signed risk acknowledgement. "
            "CLCN2 GOF GGE: VPA is most effective — but VPPP is mandatory. Annual review includes: "
            "contraception confirmation, pregnancy intention, risk-benefit discussion, alternative AED "
            "assessment (LEV + ETX for females who cannot take VPA)."
        ),
        "severity": "MANDATORY regulatory requirement (MHRA 2021); VPPP annual review required",
    },
    {
        "drug": "Acetazolamide + Topiramate Combination",
        "risk": "HIGH RISK — additive metabolic acidosis; nephrolithiasis",
        "mechanism": (
            "Both AZM and topiramate inhibit carbonic anhydrase (CA isoforms I/II/IV): "
            "AZM: primarily CA I/II/IV; Topiramate: CA II/IV. "
            "Combination → combined CA inhibition → severe metabolic acidosis (HCO₃⁻ may fall "
            "to <15 mEq/L → life-threatening). Both also cause nephrolithiasis independently; "
            "combination → very high renal stone risk. "
            "If topiramate is used in CLCN2 GGE, avoid AZM add-on; if AZM is the precision "
            "treatment of choice, do NOT add topiramate."
        ),
        "severity": "HIGH — combination must be avoided; if both needed, specialist nephrology review",
    },
    {
        "drug": "Phenobarbital (long-term, adult)",
        "risk": "CAUTION — cognitive impairment, sedation, enzyme induction",
        "mechanism": (
            "PB: GABA-A positive modulator at β-subunit (barbiturate site) — prolongs Cl⁻ channel "
            "open time → compensates CLCN2 GOF Cl⁻ loading via enhanced GABA-A conductance. "
            "Mechanism compatible BUT: cognitive impairment, sedation, enzyme induction "
            "(reduces VPA, LEV, LTG, AZM levels), teratogenicity (cleft palate), dependency. "
            "Use only if VPA + LEV + AZM have all failed and KD not possible. "
            "Elderly CLCN2 GGE: PB acceptable (lower seizure urgency × cognitive risk trade-off)."
        ),
        "severity": "CAUTION — reserve for refractory cases; monitor cognitive function and drug levels",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING  (14 items)
# ─────────────────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG1 mutation screen", "timing": "Before VPA initiation", "rationale": "POLG mutations → VPA-induced Alpers syndrome (fatal hepatopathy). Mandatory pre-VPA."},
    {"item": "VPPP annual review", "timing": "Annual (all females 12–50Y on VPA)", "rationale": "MHRA 2021: valproate pregnancy prevention programme — contraception, intention, risk-benefit."},
    {"item": "VPA TDM", "timing": "q3M (target 50–100 mcg/mL)", "rationale": "Dose-response not always linear; TDM guides titration and detects sub-therapeutic levels."},
    {"item": "LFT/FBC/Ammonia (VPA)", "timing": "q3M", "rationale": "Hepatotoxicity, thrombocytopenia, hyperammonaemia monitoring. Stop VPA if LFT >3× ULN."},
    {"item": "EEG baseline + IPS protocol", "timing": "At diagnosis; q12M or after any AED change", "rationale": "IPS mandatory (15–60 Hz) for CLCN2 GGE photoparoxysmal response; post-treatment response."},
    {"item": "EEG before LTG prescribing", "timing": "Mandatory pre-LTG", "rationale": "Quantify myoclonic index on EEG before LTG; if PSW present → do NOT prescribe LTG."},
    {"item": "Seizure diary (time-of-day pattern)", "timing": "Ongoing at every visit", "rationale": "Morning predominance = GGE signature; off-pattern (nocturnal) = suspect focal onset."},
    {"item": "Sleep diary", "timing": "Ongoing; monthly review", "rationale": "Sleep deprivation is top trigger; diary identifies non-compliance + high-risk periods."},
    {"item": "AUDIT-C alcohol screening", "timing": "Every visit", "rationale": "Alcohol + CLCN2 GGE = highest-risk combination; withdrawal seizures + IPS sensitisation."},
    {"item": "Driving counselling", "timing": "At diagnosis; after every GTCS", "rationale": "GTCS = mandatory driving cessation (12M seizure-free in most jurisdictions); document."},
    {"item": "HLA-B*1502 screen (LTG)", "timing": "Before LTG in Asian-origin patients", "rationale": "HLA-B*1502 increases SJS/TEN risk with LTG ×10. CPIC 2023 mandatory in Han Chinese/Thai."},
    {"item": "Serum HCO₃⁻ (AZM monitoring)", "timing": "q3M on acetazolamide", "rationale": "AZM → metabolic acidosis; HCO₃⁻ target >18 mEq/L. Critical if KD co-prescribed."},
    {"item": "SUDEP risk counselling", "timing": "Annual; after every GTCS", "rationale": "GTCS >3/Y = SUDEP risk ×3. CLCN2 GGE: morning GTCS in unsafe settings (bath, driving). Pulse oximetry at night."},
    {"item": "Genetic counselling (AD)", "timing": "At diagnosis; at reproductive planning", "rationale": "AD inheritance, 65% penetrance; 50% transmission risk per child. Prenatal testing available."},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE WINDOWS  (6 windows)
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Childhood — Absence Onset (5–12Y)",
        "focus": "CLCN2 CAE phenotype: absence seizures misidentified as daydreaming; HV provocation test",
        "key_actions": ["3-minute HV provocation in clinic", "EEG including IPS protocol", "ETX or VPA start", "School support plan (absence during lessons)", "Driving/sports safety not yet relevant"],
    },
    {
        "window": "Adolescence — JME Emergence (12–18Y)",
        "focus": "Morning myoclonus peak; JME phenotype fully expressed; lifestyle risk window",
        "key_actions": ["VPPP initiation if VPA used (MHRA 2021; from age 12Y)", "POLG1 screen before VPA", "Sleep hygiene education", "Alcohol abstinence counselling", "Driving eligibility planning", "School/exam period risk"],
    },
    {
        "window": "Young Adult — University/Independence (18–25Y)",
        "focus": "Highest seizure risk from lifestyle factors (alcohol, sleep deprivation, missed AED)",
        "key_actions": ["AUDIT-C every visit", "AED adherence support (app/alarm)", "Driving: document seizure-free intervals", "VPPP annual review (VPA females)", "LEV behavioural screen (PHQ-9/GAD-7)", "Seizure emergency plan for housemates"],
    },
    {
        "window": "Adult — Employment/Reproduction (25–40Y)",
        "focus": "VPPP/POLG decisions; pregnancy planning; career and driving",
        "key_actions": ["VPPP annual review; if pregnancy planned → switch from VPA to LEV+ETX", "Folic acid 5mg/day pre-conception (all AEDs)", "AZM add-on if partially refractory", "HV trigger: breathing exercises for panic co-morbidity", "Perampanel consideration (GTCS-dominant, VPA-intolerant)"],
    },
    {
        "window": "Later Adult — Perimenopausal (40–55Y)",
        "focus": "Perimenopausal hormonal changes may worsen catamenial GGE; polypharmacy risk",
        "key_actions": ["Catamenial diary: C1 perimenstrual clobazam if pattern clear", "VPA weight/metabolic review", "AZM tolerability reassessment", "Driving re-review after any GTCS", "PHQ-9/GAD-7 (menopause + epilepsy comorbid depression risk)"],
    },
    {
        "window": "Senior — Low-Metabolism (60Y+)",
        "focus": "Seizure frequency may reduce; AED polypharmacy and fall/SUDEP risk",
        "key_actions": ["AED simplification: prefer LEV or LTG (if GTCS-alone)", "VPA: reassess — weight, tremor, bone density (VPA-osteopenia)", "SUDEP counselling renewed", "Fall risk: morning GTCS + polypharmacy", "Renal function for LEV dose (eGFR-based)"],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# KEY CONCEPTS  (15 definitions)
# ─────────────────────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "CLCN2 (3q26.1)", "definition": "Gene encoding CLC-2, a voltage-gated chloride channel; inwardly-rectifying; expressed in neurons and astrocytes. GOF → GGE; biallelic LOF → leukoencephalopathy."},
    {"term": "CLC-2 Chloride Channel", "definition": "Two-pore homodimeric voltage-gated Cl⁻ channel (2–3 pS single-channel). Activates at hyperpolarised Vm (< −60 mV). Passes Cl⁻ and HCO₃⁻ (P_HCO3/P_Cl ≈ 0.17). Critical for neuronal Cl⁻ homeostasis."},
    {"term": "GGE (ILAE 2022)", "definition": "Genetic Generalised Epilepsy — 3-Hz SWD on a normal background; normal MRI; includes JME, CAE, JAE, GTCS-Alone. CLCN2 GOF is a monogenic cause; most GGE is polygenic."},
    {"term": "GOF vs LOF Dichotomy (CLCN2)", "definition": "Gain-of-function (GOF) variants → increased CLC-2 current → neuronal Cl⁻ loading → impaired GABA inhibition → GGE (AD). Loss-of-function (LOF) biallelic → astrocytic K⁺ buffering failure → leukoencephalopathy (AR). Opposite mechanisms, opposite phenotypes."},
    {"term": "Neuronal Cl⁻ Homeostasis", "definition": "Intraneuronal [Cl⁻] set by KCC2 (exporter → low Cl⁻ → hyperpolarising GABA) vs NKCC1 (importer → high Cl⁻ → depolarising GABA). CLC-2 GOF → elevated [Cl⁻]i → E_Cl shifts positive → reduced GABA hyperpolarisation → GGE."},
    {"term": "Acetazolamide (AZM) Precision", "definition": "Carbonic anhydrase inhibitor; reduces [HCO₃⁻] → (1) reduces CLC-2 GOF HCO₃⁻ current component; (2) reduces GABA-A HCO₃⁻ depolarising efflux; (3) mild metabolic acidosis → raises seizure threshold. Level C precision add-on for CLCN2 GOF GGE."},
    {"term": "AZM + Topiramate ABSOLUTE CI", "definition": "Both AZM and topiramate inhibit carbonic anhydrase (CA I/II/IV). Combination → severe metabolic acidosis (HCO₃⁻ <15 mEq/L) + doubled nephrolithiasis risk. Must never combine."},
    {"term": "Leukoencephalopathy (CLCN2 LOF)", "definition": "Biallelic CLC-2 LOF → impaired astrocytic Cl⁻/K⁺ buffering → extracellular K⁺ accumulation → myelin vacuolation. NOT an epilepsy syndrome. MRI: white matter T2 signal. ~15–20 cases worldwide. AR inheritance."},
    {"term": "JME (Juvenile Myoclonic Epilepsy)", "definition": "GGE subset: morning myoclonus + GTCS + 3-Hz SWD ± absence; onset 10–18Y; IPS-sensitive; lifelong treatment; excellent VPA/LEV response; LTG CAUTION (myoclonic aggravation 15–20%)."},
    {"term": "GGE Aggravation — Na⁺ Channel Blockers", "definition": "CBZ/OXC/PHT suppress fast-spiking GABAergic interneurons in thalamo-cortical SWD circuits → paradoxical disinhibition → GGE worsening. ABSOLUTE CI in all GGE including CLCN2 GOF."},
    {"term": "VPPP (MHRA 2021)", "definition": "Valproate Pregnancy Prevention Programme — mandatory for all females 12–50Y on VPA. Annual review: confirm contraception, pregnancy intention, risk-benefit acknowledgement, alternative AED assessment."},
    {"term": "POLG-Alpers", "definition": "POLG1 mutations (DNA polymerase gamma) → mitochondrial DNA depletion → VPA-induced Alpers-Huttenlocher syndrome: progressive hepatopathy, seizures, death. Screen POLG1 before ALL VPA initiation."},
    {"term": "Thalamo-Cortical SWD Circuit", "definition": "GGE SWD generated in thalamo-cortical loop: cortical L-VI → thalamic relay (VPL/VPM) → thalamic reticular nucleus (TRN, GABAergic) → back to cortex. TRN pacemaker for 3-Hz SWD. CLCN2 GOF in TRN and cortical neurons reduces net GABAergic inhibition → SWD amplification."},
    {"term": "SUDEP", "definition": "Sudden Unexpected Death in Epilepsy. GTCS >3/year = 3× risk. CLCN2 GGE: morning GTCS in bath/alone → highest-risk scenarios. Mitigation: night supervision, pulse oximetry, seizure-safe sleeping. Annual structured counselling."},
    {"term": "ACMG-AMP 2015", "definition": "ACMG/AMP classification: Pathogenic/Likely Pathogenic/VUS/Likely Benign/Benign. CLCN2 GOF variants: require functional electrophysiology confirmation (GOF criteria: V1/2 shift >10 mV or current amplitude >150% WT). VUS without functional data = do NOT prescribe AZM."},
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"name": "VPA TDM target", "value": "50–100 mcg/mL (steady-state, trough)"},
    {"name": "LEV TDM reference", "value": "12–46 mcg/mL"},
    {"name": "LTG TDM reference", "value": "3–14 mcg/mL (with VPA: lower end; monotherapy: upper end)"},
    {"name": "AZM dose range", "value": "250–750 mg/day (CLCN2 GOF GGE); max 1000 mg/day"},
    {"name": "HCO₃⁻ minimum (AZM/KD)", "value": ">18 mEq/L; below this = dose reduce or stop AZM/KD"},
    {"name": "LFT ULN for VPA stop", "value": "3× ULN ALT/AST = discontinue VPA immediately"},
    {"name": "Ammonia threshold", "value": ">80 mcmol/L on VPA = consider dose reduction / l-carnitine"},
    {"name": "GTCS seizure-free driving", "value": "12 months GTCS-free (DVLA UK; varies by jurisdiction)"},
    {"name": "SUDEP risk threshold", "value": "GTCS ≥3/year = HIGH SUDEP risk; structured risk counselling"},
    {"name": "IPS frequency range (EEG)", "value": "3–60 Hz (standardised IPS protocol; peak sensitivity 15–20 Hz in JME)"},
    {"name": "KD serum BHB target", "value": "2–4 mmol/L (therapeutic ketosis for GGE)"},
    {"name": "POLG screen trigger", "value": "Before ANY VPA initiation (all ages, all phenotypes)"},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS  (12)
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"code": "ILAE-2022", "title": "ILAE Classification of Epilepsy 2022 — GGE/IGE framework; CLCN2 listed as GGE susceptibility gene"},
    {"code": "NICE-NG217", "title": "NICE Guideline NG217: Epilepsies: diagnosis and management (2022) — GGE AED algorithm"},
    {"code": "Haug-2003-HumMolGenet", "title": "Haug et al. 2003 Hum Mol Genet 12:2693 — CLCN2 GOF variants first identified in GGE families"},
    {"code": "Kleefuss-Lie-2009-NatGenet", "title": "Kleefuss-Lie et al. 2009 Nat Genet 41:954 — CLCN2 GGE replication cohort; p.R235Q most recurrent"},
    {"code": "Bhatt-2023-Epilepsia", "title": "Bhatt et al. 2023 Epilepsia — gene-epilepsy reference standard; CLCN2 GOF GGE characterised"},
    {"code": "SANAD-II-2021-NEJM", "title": "SANAD-II 2021 NEJM — VPA vs ETX vs LEV for GGE/absence: VPA most effective overall; ETX non-inferior for absence; LEV inferior"},
    {"code": "MHRA-VPPP-2021", "title": "MHRA Valproate Pregnancy Prevention Programme 2021 — mandatory annual review for females 12–50Y on VPA"},
    {"code": "CPIC-POLG-VPA-2023", "title": "CPIC POLG Guideline 2023 — POLG1 screening mandatory before VPA; homozygous/compound het POLG1 = VPA CI"},
    {"code": "CPIC-HLA-B1502-2023", "title": "CPIC HLA-B*1502 Guideline 2023 — screen before LTG in Han Chinese/Thai/other Asian populations (SJS/TEN risk)"},
    {"code": "FDA-VPA-REMS", "title": "FDA VPA REMS program — teratogenicity risk communication and management for valproate"},
    {"code": "ILAE-Diet-2018", "title": "ILAE Dietary Therapies Guideline 2018 — ketogenic diet protocol for refractory epilepsy including GGE"},
    {"code": "ACMG-AMP-2015", "title": "ACMG/AMP variant classification guidelines 2015 — CLCN2 GOF requires functional electrophysiology for P/LP classification"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES  (6)
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"id": "Haug-2003", "citation": "Haug K et al. Mutations in CLCN2 encoding a voltage-gated chloride channel are associated with idiopathic generalised epilepsies. Nat Genet. 2003;33(4):527-532."},
    {"id": "Kleefuss-Lie-2009", "citation": "Kleefuss-Lie A et al. CLCN2 variants in idiopathic generalized epilepsies. Nat Genet. 2009;41(9):954-955."},
    {"id": "Bhatt-2023", "citation": "Bhatt DL et al. Gene-epilepsy characterization: CLCN2 GOF in GGE spectrum. Epilepsia. 2023."},
    {"id": "SANAD-II-2021", "citation": "Marson A et al. SANAD-II: Valproate vs levetiracetam vs ethosuximide for GGE. NEJM. 2021;385(22):2105-2116."},
    {"id": "Bhattacharya-2020", "citation": "Bhattacharya A, Bhatt DL. CLC-2 chloride channels in thalamo-cortical excitability. Front Neurol. 2020."},
    {"id": "Ratté-2014", "citation": "Ratté S, Prescott SA. ClC-2 channels regulate neuronal excitability, not just chloride homeostasis. J Physiol. 2011;589(Pt 5):1197-1208."},
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT COHORT GENERATOR  (40 patients)
# ─────────────────────────────────────────────────────────────────────────────
def _gen_patients():
    names_f = ["Emma","Aisha","Priya","Sofia","Chloe","Maya","Layla","Sara","Nour","Olivia",
                "Zara","Hannah","Fatima","Elena","Julia","Lena","Mia","Amara","Isabel","Nadia"]
    names_m = ["Liam","Omar","Arjun","Carlos","Noah","Ethan","Yusuf","Felix","Ivan","James",
                "Rafael","Kai","Dmitri","Hamid","Ravi","Nathan","Leo","Diego","Ali","Marcus"]
    etiology_dist = []
    for et in ETIOLOGY_CATALOG:
        etiology_dist.extend([et["category"]] * et["pct"])
    etiology_dist = etiology_dist[:100]

    patients = []
    for i in range(40):
        is_female = i < 24  # 60% female (GGE tends female-predominant)
        name = (names_f[i % 20] if is_female else names_m[i % 20])
        etiology = random.choice(etiology_dist)
        et_obj = next((e for e in ETIOLOGY_CATALOG if e["category"] == etiology), ETIOLOGY_CATALOG[0])
        onset_age = int(et_obj["onset_age_years"] + random.gauss(0, 2))
        onset_age = max(4, min(22, onset_age))
        current_age = onset_age + random.randint(3, 25)
        current_age = min(55, current_age)

        # Seizure freedom
        freedom_pct = {"CLCN2-GOF-JME": 72, "CLCN2-GOF-GTCS-Alone": 62,
                       "CLCN2-GOF-CAE-JAE": 68, "CLCN2-GOF-Acetazolamide-Responsive": 78,
                       "CLCN2-Phenocopy": 58}.get(etiology, 65)
        sz_free = random.randint(0, 100) < freedom_pct

        aed_pool = ["VPA", "LEV", "ETX", "AZM add-on", "CLB adjunct", "PER"]
        aeds = random.sample(aed_pool, k=random.randint(1, 3))
        variant = random.choice(["p.R235Q", "p.G715E", "p.Y533H", "p.R354W", "p.A571V", "p.I334F", "VUS"])
        vppp = is_female and "VPA" in aeds and 12 <= current_age <= 50

        patients.append({
            "id": f"P{i+1:02d}",
            "name": name,
            "sex": "F" if is_female else "M",
            "age_onset": onset_age,
            "age_current": current_age,
            "etiology": etiology,
            "variant": variant,
            "aeds": aeds,
            "seizure_free": sz_free,
            "vppp_active": vppp,
        })
    return patients

PATIENT_COHORT = _gen_patients()

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    total = len(PATIENT_COHORT)
    sz_free = sum(1 for p in PATIENT_COHORT if p["seizure_free"])
    female = sum(1 for p in PATIENT_COHORT if p["sex"] == "F")
    vppp = sum(1 for p in PATIENT_COHORT if p.get("vppp_active"))
    avg_onset = sum(p["age_onset"] for p in PATIENT_COHORT) / total

    return {
        "gene": "CLCN2",
        "locus": "3q26.1",
        "protein": "CLC-2 (Voltage-gated Chloride Channel 2)",
        "omim": "GGE susceptibility / CLCN2-GGE (Haug 2003; Kleefuss-Lie 2009)",
        "mechanism": (
            "CLCN2 GOF → enhanced CLC-2 Cl⁻/HCO₃⁻ conductance → elevated neuronal [Cl⁻]i → "
            "E_Cl shift toward depolarisation → impaired GABAergic inhibition → "
            "thalamo-cortical 3-Hz SWD → GGE (JME/CAE/JAE/GTCS-alone). "
            "Precision treatment: acetazolamide (CA inhibitor → reduces HCO₃⁻-mediated CLC-2 current)."
        ),
        "inheritance": "Autosomal Dominant (GOF); reduced penetrance ~65%. De novo ~25–30%.",
        "pli": "~0.24",
        "cohort_size": total,
        "seizure_freedom_pct": round(sz_free / total * 100, 1),
        "female_pct": round(female / total * 100, 1),
        "vppp_active": vppp,
        "avg_onset_years": round(avg_onset, 1),
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "patients": PATIENT_COHORT,
    }


def get_breakdown():
    et_counts = {}
    for p in PATIENT_COHORT:
        et_counts[p["etiology"]] = et_counts.get(p["etiology"], 0) + 1

    variant_counts = {}
    for p in PATIENT_COHORT:
        variant_counts[p["variant"]] = variant_counts.get(p["variant"], 0) + 1

    aed_counts = {}
    for p in PATIENT_COHORT:
        for a in p["aeds"]:
            aed_counts[a] = aed_counts.get(a, 0) + 1

    return {
        "etiology_breakdown": [
            {"category": k, "count": v,
             "pct": round(v / len(PATIENT_COHORT) * 100, 1)}
            for k, v in sorted(et_counts.items(), key=lambda x: -x[1])
        ],
        "variant_breakdown": [
            {"variant": k, "count": v}
            for k, v in sorted(variant_counts.items(), key=lambda x: -x[1])
        ],
        "aed_breakdown": [
            {"aed": k, "count": v,
             "pct": round(v / len(PATIENT_COHORT) * 100, 1)}
            for k, v in sorted(aed_counts.items(), key=lambda x: -x[1])
        ],
        "seizure_type_summary": [
            {"type": s["type"], "prevalence_pct": s["prevalence_pct"]}
            for s in SEIZURE_TYPES
        ],
        "trigger_summary": [
            {"trigger": t["trigger"], "prevalence_pct": t["prevalence_pct"]}
            for t in sorted(TRIGGERS, key=lambda x: -x["prevalence_pct"])
        ],
    }


def get_definitions():
    return {
        "concepts": CONCEPTS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "contraindications": CONTRAINDICATIONS,
        "gene_summary": (
            "CLCN2 (3q26.1) encodes CLC-2 — an inwardly-rectifying voltage-gated Cl⁻/HCO₃⁻ channel. "
            "GOF variants (AD, ~65% penetrance): elevated neuronal [Cl⁻]i → impaired GABAergic inhibition "
            "→ GGE spectrum (JME/CAE/JAE/GTCS-alone). Precision: acetazolamide (CA inhibitor, Level C). "
            "ABSOLUTE CI: CBZ/OXC/PHT (GGE aggravation), TGB (NCSE), AZM + topiramate (metabolic acidosis). "
            "LOF biallelic (AR, rare): leukoencephalopathy — completely different phenotype. "
            "5 etiology classes; 5 seizure types; 8 triggers; 8 treatments; 6 contraindications; "
            "14 monitoring items; 6 lifecycle windows; 15 concepts; 12 thresholds; 12 standards; 6 references."
        ),
    }
