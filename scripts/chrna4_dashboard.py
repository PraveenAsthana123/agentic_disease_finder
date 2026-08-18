"""
CHRNA4 Epilepsy — Autosomal Dominant Nocturnal Frontal Lobe Epilepsy (ADNFLE) /
Nicotinic Acetylcholine Receptor Alpha-4 Subunit / nAChR GOF / 20q13.33
==========================================================================
40-patient cohort · CHRNA4 (20q13.33) · Cholinergic Receptor Nicotinic Alpha-4 Subunit
Gene OMIM: *118504 · Syndrome: ADNFLE — Autosomal Dominant Nocturnal Frontal Lobe Epilepsy
OMIM #600513 · First epilepsy ion channel gene described (Steinlein et al. 1995 Nat Genet)

KEY CHRNA4 BIOLOGY — nAChR ALPHA-4 SUBUNIT / FRONTAL LOBE EPILEPSY:
CHRNA4 (20q13.33) encodes the α4 subunit of neuronal nicotinic acetylcholine receptors (nAChRs).
nAChRs are pentameric ligand-gated ion channels permeable to Na⁺, K⁺, and Ca²⁺.
The most abundant neuronal nAChR is the heteropentamer (α4)₂(β2)₃ (encoded by CHRNA4 + CHRNB2).

PROTEIN STRUCTURE:
  · α4 subunit: 627 aa, 4 transmembrane segments (M1–M4); M2 forms the channel pore
  · M2 segment lines the ion channel gate — most ADNFLE mutations cluster here
  · Signal transduction: ACh binds at α/β subunit interface → conformational change → channel opens
  · Desensitization: prolonged ACh exposure → closed, high-affinity, unresponsive (refractory) state

GOF MECHANISM IN ADNFLE:
  · ADNFLE mutations cause GAIN-OF-FUNCTION (GOF) of nAChRs:
    (1) Increased sensitivity to ACh (decreased EC₅₀) — receptors open at lower ACh concentrations
    (2) Impaired/slowed desensitization — receptors stay open longer before becoming refractory
    (3) Net effect: excessive nicotinic cholinergic transmission → frontal cortex hyperexcitability
  · Nocturnal predominance: during NREM sleep, cholinergic tone decreases and thalamo-cortical
    synchrony is high → transition from quiet to phasic NREM → brief cholinergic burst → triggers
    GOF nAChR-mediated frontal lobe seizure
  · Thalamo-frontal hypothesis: frontal layer V/VI neurons express high-density nAChR (α4β2) →
    cholinergic surges during NREM arousals → depolarise frontal networks → hypermotor ictal discharge

KEY MUTATIONS:
  · S284L (Ser284Leu) — M2 domain serine to leucine: first ADNFLE mutation described
    (Steinlein 1995 Nat Genet); accounts for ~20% of CHRNA4 ADNFLE; delays desensitization
  · 776insL (leucine insertion at position 776) — in-frame insertion near M2-M3 linker;
    described in Australian/Scottish families (Hirose 1999 Ann Neurol; Phillips 2000 Hum Mol Genet)
  · S252L, T265I, L301V — additional M2 region GOF variants
  · C-terminal and non-M2 variants: less common; variable functional impact
  · 30–40% of ADNFLE families have no CHRNA4/CHRNB2/CHRNA2 mutation (genetic heterogeneity)

NICOTINIC RECEPTOR PHARMACOLOGY — CLINICAL RELEVANCE:
  · CARBAMAZEPINE (CBZ) is FIRST-LINE for ADNFLE: 70–80% complete seizure freedom
    Mechanism: voltage-gated Na⁺ channel stabilisation → reduces frontal hyperexcitability
    NOT a direct nicotinic receptor modulator — indirect via Na⁺-channel-dependent excitability
  · NICOTINE PARADOX: CHRNA4 encodes the PRIMARY NICOTINE BINDING SITE
    Low-dose nicotine → receptor desensitization (net inhibitory effect on already-activated GOF receptors)
    Some patients report smoking reduces seizure frequency (anecdotal)
    Nicotine patch (7–14 mg/24h) — Level C anecdotal benefit (Brodtkorb 2002; Steinlein 2003)
    HIGH-DOSE nicotine (heavy smoking, chewing tobacco, high-dose NRT) → activates GOF receptors
    acutely → potential seizure precipitant before desensitization occurs
    SMOKING CESSATION abruptly → receptor upregulation → transient increased sensitivity → monitor closely
  · CARBAMAZEPINE + HLA-B*15:02: HIGH RISK in SE Asian ancestry (SJS/TEN) — MANDATORY HLA-B test
    before CBZ in patients of Han Chinese, Thai, Malay, Vietnamese, Filipino ancestry
  · OXCARBAZEPINE: CBZ alternative; similar Na⁺-channel mechanism; fewer drug interactions;
    HLA-B*15:02 risk also applies (cross-reactive)
  · LACOSAMIDE: slow-inactivation Na⁺-channel modulator; useful CBZ adjunct or alternative for DRE
  · TOPIRAMATE: broad-spectrum; useful adjunct; cognitive side effects limit use
  · LAMOTRIGINE: Na⁺-channel stabiliser but LESS EFFECTIVE than CBZ for frontal nocturnal seizures
    (fewer case series data for ADNFLE; CBZ remains preferred)
  · PHENOBARBITONE: GABA-A potentiator; older alternative; sedating; not first-line
  · CLONAZEPAM: benzodiazepine adjunct; useful for sleep-related seizure clusters

ADNFLE DIAGNOSTIC CHALLENGE:
  · MISDIAGNOSIS RATE: 30–50% of ADNFLE patients are initially misdiagnosed as:
    (1) Parasomnias — NREM parasomnias (sleepwalking, night terrors), REM sleep behavior disorder
    (2) Psychiatric — nocturnal panic attacks, psychogenic non-epileptic events, nightmares
    (3) Obstructive sleep apnoea — arousal events, respiratory events
  · DIAGNOSTIC GOLD STANDARD: Video-polysomnography (VPSG) — captures seizure during NREM sleep
  · ROUTINE EEG: Often normal even during awake state; abnormal EEG only in 50–60% during sleep EEG
  · MRI: Normal in ADNFLE (unlike structural FLE from FCD, tumour, etc.)
  · GENETICS: Panel testing (CHRNA4, CHRNB2, CHRNA2) recommended; 60–70% positive in familial ADNFLE
"""

import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG — 5-CLASS GOF SPECTRUM
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GOF-Missense-S284L-Classic-ADNFLE",
        "pct": 35,
        "mechanism": "Serine284Leucine (S284L) missense in M2 transmembrane domain — first CHRNA4 ADNFLE mutation (Steinlein 1995 Nat Genet). M2 leucine insertion alters gate hydrophobicity → slowed desensitization → prolonged nAChR opening → excessive frontal cholinergic activation during NREM sleep transitions. Full ADNFLE penetrance ~70%; interfamilial variability in seizure frequency and severity.",
        "eeg": "VPSG: electrodecrement or low-amplitude fast activity over frontal leads during NREM stage 2–3; seizure onset 10–14 Hz rhythmic fast activity F3/F4; secondary generalisation rare; interictal EEG normal in 50–60%",
        "onset_months": "84–204 months (7–17 years)",
        "severity": "moderate — CBZ responsive in 75%",
    },
    {
        "category": "GOF-Missense-Other-M2-Domain-Variants",
        "pct": 25,
        "mechanism": "Other M2 domain missense variants (S252L, T265I, L301V, C244G): all cluster in or near the pore-lining M2 segment. Functional consequences similar to S284L (GOF: reduced EC₅₀, slowed desensitization) but with variable magnitude. T265I: reduced Ca²⁺ permeability. L301V: altered rectification. Generally similar clinical ADNFLE phenotype; some with higher seizure burden.",
        "eeg": "VPSG: frontal fast activity during NREM; similar to S284L; ictal EEG may show early bifrontal involvement; post-ictal EEG suppression rare",
        "onset_months": "72–228 months (6–19 years)",
        "severity": "moderate — CBZ responsive in 65–70%",
    },
    {
        "category": "GOF-Insertion-776insL-Australian-Scottish",
        "pct": 20,
        "mechanism": "In-frame leucine insertion in the M2–M3 intracellular linker domain (776insL); described in Australian/Scottish pedigrees (Hirose 1999 Ann Neurol; Phillips 2000 Hum Mol Genet). Alters the M2–M3 loop conformation → modified gating; delayed channel closure. Clinically: more prominent hypermotor component, higher proportion with nocturnal GTCS than S284L families. CBZ effective ~70%.",
        "eeg": "VPSG: prominent fast ictal discharge; wider frontal field than S284L; secondary generalisation more common (GTCS in sleep in 30%); ambulatory EEG sometimes captures if frequent",
        "onset_months": "96–216 months (8–18 years)",
        "severity": "moderate — slightly more nocturnal GTCS than S284L",
    },
    {
        "category": "GOF-Atypical-Non-M2-or-New-Variant",
        "pct": 15,
        "mechanism": "Novel CHRNA4 variants outside the M2 domain (N-terminal, M3, C-terminal) with uncertain or indirect GOF. Includes missense variants affecting subunit assembly, membrane insertion, or post-translational processing. Penetrance may be lower (50–60%); phenotype sometimes milder or atypical (diurnal seizures, dystonic posturing rather than hypermotor). Functional confirmation needed.",
        "eeg": "VPSG: variable; some with more diurnal focal EEG changes; less predictable NREM pattern than classic M2 mutations",
        "onset_months": "108–288 months (9–24 years)",
        "severity": "mild–moderate — higher proportion CBZ partial responders",
    },
    {
        "category": "Phenocopy-ADNFLE-Panel-Negative",
        "pct": 5,
        "mechanism": "ADNFLE phenotype (hypermotor nocturnal seizures from NREM sleep, frontal EEG onset, CBZ response) but CHRNA4/CHRNB2/CHRNA2 panel negative. 30–40% of familial ADNFLE has no identified mutation. Alternative loci: KCNT1 (ADNFLE form), CRH, DEPDC5 (mTOR pathway focal epilepsy — NFLE phenocopy), structural FCD (MRI negative FCD2B).",
        "eeg": "VPSG: identical ADNFLE EEG pattern; genetic testing reflex to KCNT1, DEPDC5; high-resolution MRI (3T FCD protocol)",
        "onset_months": "84–228 months (7–19 years)",
        "severity": "variable — reflects heterogeneous aetiology",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE TYPES — 5 TYPES
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Hypermotor Nocturnal Seizures (ADNFLE Core)",
        "frequency_pct": 95,
        "eeg_correlate": "VPSG: frontal fast activity 10–14 Hz, often bilateral (F3-F4); electrodecrement at onset; evolves to rhythmic frontal discharge 5–8 Hz; scalp EEG may be obscured by motion artefact; duration 20–180 seconds; arise from NREM stage 2–3",
        "semiology": "Sudden arousal from sleep → head deviation (often contralateral to hemisphere of onset) → thrashing, cycling leg movements, complex hypermotor activity (boxing, bicycling, rocking); vocalisations (screams, moans); sometimes bizarre posturing. Post-ictal: rapid return to sleep, no confusion or full post-ictal. Patient often has no recall.",
        "clinical_tip": "VIDEO-POLYSOMNOGRAPHY is essential: hypermotor semiology often invisible on routine EEG. In ADNFLE, seizures cluster in first 2–3 hours of NREM sleep (sleep onset). CBZ at night (higher bedtime dose) is preferred strategy. Often misdiagnosed as night terrors or REM sleep behavior disorder — key distinction: ADNFLE seizures arise from NREM (EEG) and have stereotyped motor pattern.",
    },
    {
        "type": "Minor Motor Events / Paroxysmal Arousals",
        "frequency_pct": 68,
        "eeg_correlate": "Brief frontal electrodecrement or low-amplitude fast discharge; often missed on ambulatory EEG; VPSG essential to capture. Duration <30s. May precede or follow hypermotor events in clusters.",
        "semiology": "Brief (5–30 second) partial awakenings: sit up in bed, mumble, frightened expression, stereotyped arm/leg movement; sometimes associated with aura (chest tightness, fear, pressure sensation). Stereotypy is key feature — same sequence of movements each event.",
        "clinical_tip": "Minor motor events may occur up to 20–30 times per night in uncontrolled ADNFLE. Patients/families often count these as 'sleep disturbances' not seizures. Detailed sleep questionnaire (seizure diary INCLUDING minor motor events) crucial. CBZ dramatically reduces minor motor events alongside hypermotor seizures.",
    },
    {
        "type": "Nocturnal Tonic Seizures",
        "frequency_pct": 52,
        "eeg_correlate": "Paroxysmal fast activity 10–20 Hz over frontal leads; tonic period corresponds to sustained frontal discharge; may spread to parietal; EEG artefact from tonic muscular activity masks cortical signal",
        "semiology": "Bilateral tonic stiffening from sleep: extended posture (M2 posturing if bilateral frontal), no focal clonic; may be mistaken for sleep myoclonus or hypnic jerk if brief (<15 seconds). Sustained tonic >30s more obvious.",
        "clinical_tip": "Tonic seizures in ADNFLE are distinct from Lennox-Gastaut tonic: they arise from NREM, patient is arousable, no drop attack risk while lying down. If tonic seizures appear frequently → increase CBZ dose. Tonic seizures respond to CBZ/OXC better than to VPA or LTG.",
    },
    {
        "type": "Epileptic Nocturnal Wandering",
        "frequency_pct": 38,
        "eeg_correlate": "Prolonged frontal discharge (>60s) → secondary spread to temporal/parietal; ambulatory patients with complex nocturnal wandering have higher injury risk; VPSG essential for classification",
        "semiology": "Complex automatisms during sleep: getting out of bed, walking, performing complex actions (appearing purposeful but stereotyped); may last 1–5 minutes; reminiscent of non-REM parasomnia (confusional arousal, sleepwalking) but eyes often open + staring + stereotyped motor sequence. Risk of injury (falls, leaving home).",
        "clinical_tip": "Epileptic wandering is a SAFETY EMERGENCY: bed alarms, door sensors, stair gates essential. It is a misdiagnosed seizure — not sleepwalking — and responds to CBZ. Distinguish from NREM parasomnia by VPSG: ADNFLE has EEG ictal correlate; parasomnia has slow EEG (NREM 4) without ictal discharge during the event.",
    },
    {
        "type": "Occasional Diurnal Focal Seizures",
        "frequency_pct": 15,
        "eeg_correlate": "Frontal rhythmic fast activity; sometimes bilateral; interictal EEG may show frontal sharp waves; diurnal seizures more likely to be captured on routine 30-minute EEG (15–20% capture rate)",
        "semiology": "Focal aware seizures: aura (epigastric rising, fear, déjà vu, chest tightness) → brief focal motor (hand automatisms, dystonic posturing). Diurnal events are less common than nocturnal (15%) but more frequently the diagnostic trigger (patient/doctor notices daytime events).",
        "clinical_tip": "Diurnal seizures in ADNFLE suggest high epileptic burden. Escalate CBZ. If diurnal seizures persist despite optimal CBZ → add LCM or switch to OXC/CBZ combination. Diurnal focal seizures should prompt formal VPSG and genetics panel (CHRNA4/CHRNB2/CHRNA2/KCNT1/DEPDC5).",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGERS — 8 KEY TRIGGERS
# ─────────────────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Sleep Onset / NREM Stage Transitions", "pct": 92, "note": "ADNFLE seizures are exquisitely linked to NREM sleep stage 2–3 transitions; cholinergic surges during wake→NREM transitions activate GOF nAChRs. CBZ at bedtime (70% of total daily dose) reduces nocturnal ictal burden most effectively."},
    {"trigger": "Sleep Deprivation / Disruption", "pct": 78, "note": "Irregular sleep schedule increases frontal hyperexcitability; NREM homeostatic pressure increases → more NREM 3 → higher seizure risk. Strict sleep hygiene: fixed sleep-wake times; night shift work contraindicated for uncontrolled ADNFLE. Sleep diary maintained."},
    {"trigger": "Stress / Emotional Arousal", "pct": 62, "note": "Stress increases central cholinergic tone (locus coeruleus activation) → primes GOF nAChRs for ictal discharge. Pre-sleep anxiety management (CBT-I, relaxation techniques) reduces seizure frequency. Evening stress is highest-risk period."},
    {"trigger": "Intercurrent Illness / Fever", "pct": 48, "note": "Fever increases neuronal excitability globally; fever-provoked ADNFLE is distinct from febrile seizures (older age, frontal semiology, NREM onset). Pre-emptive paracetamol if fever >38°C. Ensure oral CBZ not missed during illness (IV CBZ/fosphenytoin if NPO)."},
    {"trigger": "Missed CBZ Doses", "pct": 43, "note": "CBZ is short-acting; missed dose → trough level drop within 24h → rebound seizures. Extended-release CBZ-XR (Tegretol XR) preferred over standard formulation for more stable levels. CBZ TDM essential — target 4–12 mg/L (8–12 for adequate ADNFLE control in most patients)."},
    {"trigger": "Alcohol Consumption", "pct": 35, "note": "Alcohol acutely suppresses neuronal activity (GABA-A enhancement) but rebound hyperexcitability during alcohol metabolism and the early NREM sleep after drinking → seizure risk. ADNFLE patients should avoid evening alcohol. Moderation counselling at first appointment."},
    {"trigger": "Shift Work / Jet Lag / Irregular Sleep Schedule", "pct": 30, "note": "Circadian rhythm disruption redistributes NREM stages — jet lag forces early NREM sleep at unaccustomed circadian phases → cholinergic surge at non-typical times → ADNFLE seizure risk. Plan CBZ timing relative to sleep onset (bedtime dose), not clock time."},
    {"trigger": "Abrupt Nicotine Cessation / Nicotine Surges", "pct": 22, "note": "CHRNA4 PARADOX: nAChR α4 subunit is the PRIMARY NICOTINE BINDING SITE. Abrupt smoking cessation → receptor upregulation → transient increased sensitivity to endogenous ACh → seizure risk during transition. Very heavy nicotine (≥30 cigarettes/day, high-dose NRT bolus) may acutely activate GOF receptors. Low-dose nicotine patch (7–14 mg/24h) is investigational; discuss gradual cessation if patient is a smoker. Do NOT advise abrupt cessation without medical supervision."},
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS — 8 KEY AEDs/INTERVENTIONS
# ─────────────────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Carbamazepine (CBZ) — Tegretol/Tegretol-XR (Level B, First-Line ADNFLE)",
        "level": "Level B (observational series, no CHRNA4-specific RCT; ILAE 2022 focal epilepsy first-line)",
        "dose": "Start 200 mg twice daily; titrate to 600–1200 mg/day (max 1600 mg/day) in 2–3 doses; PREFERRED: extended-release CBZ-XR for more stable trough levels and once/twice daily dosing. Bedtime-heavy dosing (e.g., 1/3 morning, 2/3 bedtime) for nocturnal seizure coverage.",
        "moa": "Voltage-gated sodium channel (VGSC) slow-inactivation stabilisation — selectively reduces high-frequency, repetitive Na⁺-channel firing; reduces burst discharge from frontal cortex. Na⁺-channel stabilisation → reduces cholinergic-triggered frontal hyperexcitability cascade in ADNFLE.",
        "efficacy": "70–80% complete seizure freedom in CHRNA4 ADNFLE with adequate dosing; best results with CBZ-XR for bedtime coverage. Partial responders benefit from dose escalation to high-therapeutic range (8–12 mg/L TDM).",
        "monitoring": "CBZ TDM (total serum): target 4–12 mg/L (8–12 mg/L for adequate nocturnal coverage); q3M steady-state; FBC q3M first year (agranulocytosis risk <0.5%); LFT q3M first year; HLA-B*15:02 testing BEFORE initiation in SE Asian ancestry (SJS/TEN risk); Na⁺ q6M (CBZ-induced SIADH — hyponatraemia); drug interactions (CBZ autoinducer — CYP3A4 inducer); VPPP for CBZ in pregnancy (carbamazepine teratogenesis — spina bifida risk lower than VPA but non-trivial).",
        "chrna4_note": "FIRST-LINE of choice for ADNFLE. CBZ-XR preferred over immediate-release for stable nocturnal troughs. Bedtime dose optimisation is critical. Monitor HLA-B*15:02 in all SE Asian patients BEFORE starting CBZ — SJS/TEN risk is highest in first 8 weeks.",
    },
    {
        "drug": "Oxcarbazepine (OXC) — Trileptal/Oxtellar XR (Level C, CBZ Alternative)",
        "level": "Level C (observational; extrapolation from focal epilepsy data; similar mechanism to CBZ)",
        "dose": "Start 150–300 mg twice daily; titrate to 600–1800 mg/day (max 2400 mg/day); 2 divided doses; XR formulation (Oxtellar XR) preferred for nocturnal coverage",
        "moa": "VGSC slow-inactivation blockade — same mechanism as CBZ via active metabolite monohydroxycarbamazepine (MHD). No CYP3A4 autoinduction (unlike CBZ → more stable levels). Fewer drug interactions than CBZ.",
        "efficacy": "65–75% seizure freedom (no direct ADNFLE RCT; extrapolated from focal epilepsy equivalence); better tolerability profile than CBZ (fewer cognitive SEs, rash in 3% vs 10% for CBZ). Preferred in patients with CBZ hypersensitivity (cross-reactivity with OXC in ~25–30% — caution).",
        "monitoring": "Na⁺ q3M (OXC-induced hyponatraemia MORE common than CBZ — 25% vs 5%); MHD TDM (target 12–30 mg/L); LFT q6M; HLA-B*15:02 test before initiation (OXC SJS/TEN risk lower than CBZ but present in HLA carriers); renal function q12M",
        "chrna4_note": "Preferred alternative to CBZ if: (1) CBZ-induced hyponatraemia problematic; (2) CBZ drug interactions; (3) better tolerability required. Hyponatraemia (Na⁺ <130 mmol/L) more common — Na⁺ monitoring essential especially in elderly.",
    },
    {
        "drug": "Lacosamide (LCM) — Vimpat (Level C, Adjunct/Alternative for CBZ-Resistant ADNFLE)",
        "level": "Level C (observational series in focal epilepsy; small ADNFLE case series; FDA/EMA approved focal epilepsy)",
        "dose": "Start 50 mg twice daily; titrate q2W by 100 mg/day; target 200–400 mg/day (max 600 mg/day); IV formulation available (1:1 dose conversion for NPO)",
        "moa": "Selective VGSC slow-inactivation enhancer — distinct from CBZ/OXC mechanism (collapsin response mediator protein 2 / CRMP2 interaction + slow-inactivation). Additive to CBZ/OXC when combined. Cardiac PR prolongation (PR >200 ms baseline → caution).",
        "efficacy": "40–55% ≥50% seizure reduction as add-on to CBZ in ADNFLE focal epilepsy; synergistic with CBZ; useful for CBZ-resistant cases; reduces GTCS in nocturnal epilepsy",
        "monitoring": "ECG before initiation and at dose increase >400 mg/day (PR prolongation); LCM TDM not routine (clinical titration); cognitive effects (dizziness, diplopia) at higher doses; LFT q12M; avoid in 2nd/3rd degree AV block",
        "chrna4_note": "Excellent adjunct to CBZ for CBZ partial responders or as monotherapy if CBZ not tolerated. PR interval monitoring is critical. Lacosamide acts on slow inactivation — complementary to CBZ fast/slow inactivation dual mechanism.",
    },
    {
        "drug": "Nicotine Patch (Transdermal) — Level C (Investigational; Anecdotal)",
        "level": "Level C (case reports; Brodtkorb 2002 Epileptic Disord; Steinlein 2003 Epilepsy Res; no RCT)",
        "dose": "7–14 mg/24h nicotine transdermal patch (lowest available dose); applied overnight; some reports of benefit with once-daily low-dose patch at bedtime. Not commercially prescribed for epilepsy — off-label, experimental.",
        "moa": "CHRNA4 DESENSITISATION HYPOTHESIS: low-dose sustained nicotine → binds GOF nAChR α4β2 → induces receptor desensitisation (inactivated, high-affinity state) → reduces available activated receptors for ACh-triggered ictal discharge during NREM sleep transitions. Paradoxically, the nicotinic receptor GOF is 'silenced' by sustained low-dose nicotine-induced desensitisation.",
        "efficacy": "Anecdotal: 3–4 case reports of partial seizure reduction with nicotine patch in CHRNA4 ADNFLE. Not replicated in controlled trial. Effect size uncertain. Some patients smoke to reduce seizures — likely same desensitisation mechanism.",
        "monitoring": "Cardiovascular monitoring (BP, HR — nicotine patch: tachycardia, hypertension); skin reaction at patch site; sleep quality (nicotine disrupts REM sleep); dependency counselling; psychiatric assessment (addiction); do not use in patients with active cardiovascular disease, arrhythmia, uncontrolled hypertension",
        "chrna4_note": "INVESTIGATIONAL ONLY. Discuss informed consent. Some patients independently report smoking reduces seizures (self-discovered desensitisation). Low-dose patch at bedtime is the proposed approach. Do not prescribe HIGH-DOSE nicotine — paradoxical nAChR activation risk. Smoking cessation in ADNFLE: taper gradually, not abruptly.",
    },
    {
        "drug": "Levetiracetam (LEV) — Keppra (Level C, Adjunct)",
        "level": "Level C (focal epilepsy adjunct; no ADNFLE-specific series; useful when CBZ not tolerated)",
        "dose": "500–3000 mg/day in 2 divided doses; titrate q2W; IV 1:1 conversion for NPO situations",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulation → reduces Ca²⁺-dependent neurotransmitter release; modulates nAChR-mediated glutamate release indirectly. Not primarily a Na⁺-channel drug → additive to CBZ via different mechanism.",
        "efficacy": "25–40% ≥50% seizure reduction as adjunct to CBZ in focal epilepsy; limited specific ADNFLE data; useful when CBZ/OXC not tolerated; behavioural side effects (irritability, aggression) in ~20%",
        "monitoring": "Renal function q12M (LEV renally cleared); behavioural monitoring (irritability, depression — more prominent in adults); TDM not routine; no hepatotoxicity",
        "chrna4_note": "Reserve for CBZ-intolerant patients or as adjunct. LEV is POLG-safe (important if VPA being considered). Behavioural side effects (irritability) may affect sleep quality and indirectly worsen nocturnal seizures — monitor.",
    },
    {
        "drug": "Topiramate (TPM) — Topamax (Level C, Adjunct for Drug-Resistant ADNFLE)",
        "level": "Level C (focal epilepsy adjunct; broad-spectrum; useful for DRE ADNFLE)",
        "dose": "Start 25 mg/day; titrate 25–50 mg/week increments; target 100–400 mg/day; 2 divided doses",
        "moa": "Multi-mechanism: (1) Na⁺-channel blockade; (2) GABA-A modulation; (3) AMPA/kainate receptor blockade; (4) carbonic anhydrase inhibition. Broad-spectrum activity useful for multiple seizure types.",
        "efficacy": "35–50% ≥50% seizure reduction in DRE focal epilepsy as adjunct; useful when CBZ + LCM insufficient",
        "monitoring": "Cognitive effects (word-finding, concentration — 'dopamax') — use slow titration; renal stones (carbonic anhydrase → hypercalciuria; ensure adequate hydration ≥2L/day); metabolic acidosis; glaucoma (acute angle-closure — rare; stop if eye pain); weight loss; serum bicarbonate q6M",
        "chrna4_note": "Third-line adjunct for DRE ADNFLE. Cognitive side effects are significant — titrate slowly (≥25 mg/2W increments). Monitor cognition formally (MOCA/cognitive screening). Not suitable for patients in demanding cognitive occupations at higher doses.",
    },
    {
        "drug": "Clobazam (CLB) — Onfi / Frisium (Level C, Adjunct for Seizure Clusters)",
        "level": "Level C (observational adjunct; benzodiazepine adjunct for focal epilepsy clusters)",
        "dose": "5–20 mg/day (adults); 0.1–0.3 mg/kg/day paediatric; once daily at bedtime preferred (long T½ ~36h for N-desmethylclobazam) — concentrations peak during NREM sleep period",
        "moa": "1,5-Benzodiazepine; GABA-A agonist (α1/α5 subunit preference); anticonvulsant + mild sedation. Bedtime dosing ensures peak effect during first 3–4h sleep (highest ADNFLE risk period).",
        "efficacy": "35–50% ≥50% seizure reduction as adjunct; tolerance develops 30–40% over 12 months (GABA-A downregulation); drug holiday protocol useful; particularly effective for seizure clustering",
        "monitoring": "Daytime sedation (active metabolite N-desmethylclobazam persists); CYP2C19 poor metabolisers accumulate N-DMC → excess sedation; respiratory function; daytime vigilance (driving); dependency counselling for long-term use",
        "chrna4_note": "Bedtime dosing strategy is a clinical advantage in ADNFLE — peak CLB levels during sleep window when ADNFLE seizures occur. Consider intermittent use (3 months on, 1 month holiday) to prevent tolerance.",
    },
    {
        "drug": "Zonisamide (ZNS) — Zonegran (Level C, Alternative Broad-Spectrum Adjunct)",
        "level": "Level C (focal epilepsy adjunct; case series; useful for CBZ partial responders)",
        "dose": "Start 50 mg/day; titrate q2W by 50 mg; target 200–400 mg/day (max 600 mg/day); once daily dosing (T½ 63h); bedtime administration preferred",
        "moa": "Multi-mechanism: Na⁺-channel slow inactivation, T-type Ca²⁺ channel blockade, carbonic anhydrase inhibition, free radical scavenging. Distinct from CBZ — useful adjunct.",
        "efficacy": "30–45% ≥50% seizure reduction as adjunct in focal epilepsy; once-daily dosing aids adherence; long T½ → stable nocturnal levels",
        "monitoring": "Renal stones (carbonic anhydrase — hydration); cognitive effects (less than TPM); oligohydrosis/hyperthermia in paediatric patients; serum bicarbonate q6M; metabolic acidosis",
        "chrna4_note": "Once-daily bedtime dosing is a practical advantage for ADNFLE — maximises nocturnal drug levels. Alternative to TPM with possibly fewer cognitive side effects at equivalent doses.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS — 6 KEY
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine / Oxcarbazepine + HLA-B*15:02 — ABSOLUTE CI in HLA-B*15:02 Carriers",
        "risk": "ABSOLUTE CI",
        "mechanism": "CBZ (and to a lesser degree OXC) causes Stevens-Johnson Syndrome (SJS) / Toxic Epidermal Necrolysis (TEN) in HLA-B*15:02 carriers. HLA-B*15:02 prevalence: Han Chinese 8–10%; Thai 8%; Malay 6%; Vietnamese 4%; Filipino 3%; South Asian 2–4%. Risk of SJS/TEN within FIRST 8 WEEKS of CBZ initiation — potentially fatal. MANDATORY: HLA-B*15:02 test BEFORE starting CBZ or OXC in patients of SE Asian ancestry. CPIC Level A guideline: if HLA-B*15:02 positive → ABSOLUTE CI for CBZ/OXC → use LCM or ZNS or phenobarbitone instead.",
        "alternative": "Lacosamide (LCM) or zonisamide (ZNS) as CBZ alternatives in HLA-B*15:02-positive ADNFLE patients. Phenobarbitone (sedating — less preferred). LTG (SJS risk lower in HLA-B*15:02 than CBZ but cross-reactivity exists; CPIC recommends caution for LTG in HLA-A*31:01 — different HLA, different gene, different ethnicity)."
    },
    {
        "drug": "Tiagabine (TGB) — ABSOLUTE CI (Non-Convulsive Status Epilepticus)",
        "risk": "ABSOLUTE CI",
        "mechanism": "TGB (GABA reuptake inhibitor, GAT-1) → non-convulsive status epilepticus (NCSE) in patients with focal epilepsy and genetic epilepsy backgrounds. ADNFLE nocturnal NCSE triggered by TGB is particularly dangerous (occurring during sleep, often unwitnessed). NEVER prescribe TGB in ADNFLE regardless of refractory status.",
        "alternative": "CLB, LEV, LCM, or ZNS for adjunctive therapy. No justification for TGB use in ADNFLE."
    },
    {
        "drug": "Valproic Acid + POLG1 Pathogenic Variant — ABSOLUTE CI (Alpers Hepatotoxicity)",
        "risk": "ABSOLUTE CI",
        "mechanism": "VPA in POLG1 pathogenic variant carriers → fatal hepatotoxicity (Alpers-Huttenlocher syndrome). Although VPA is not first-line for ADNFLE (CBZ preferred), VPA is sometimes used as adjunct for patients with co-existing generalised epilepsy features. POLG1 testing recommended before any VPA initiation. If POLG1 positive or result unavailable urgently → avoid VPA → use LEV instead.",
        "alternative": "LEV is the safe alternative if POLG1 status unknown or positive. For ADNFLE, CBZ/LCM/ZNS preferred over VPA regardless."
    },
    {
        "drug": "High-Dose Nicotine (Heavy Smoking / High-Dose NRT Bolus) — HIGH RISK",
        "risk": "HIGH RISK",
        "mechanism": "CHRNA4 PARADOX: nAChR α4 is the primary nicotine binding site. LOW-DOSE sustained nicotine (patch 7–14 mg) → receptor desensitisation (net inhibitory). HIGH-DOSE ACUTE nicotine (smoking ≥30 cigarettes/day; chewing tobacco; nicotine gum 4 mg; e-cigarette high-strength bolus) → acute activation of GOF CHRNA4 receptors BEFORE desensitisation occurs → potential seizure precipitation. ABRUPT SMOKING CESSATION → receptor upregulation → transient hypersensitivity to endogenous ACh → risk of seizure cluster during withdrawal period.",
        "alternative": "If patient smokes: do NOT advise abrupt cessation. Gradual taper with low-dose nicotine patch (7 mg). If nicotine cessation planned, overlap with increased CBZ dose during transition. Avoid high-dose nicotine products."
    },
    {
        "drug": "Phenytoin (PHT) — HIGH RISK (Suboptimal + Cognitive Toxicity)",
        "risk": "HIGH RISK",
        "mechanism": "PHT is a VGSC blocker — same mechanistic class as CBZ, but with inferior pharmacodynamic profile for ADNFLE: (1) narrow therapeutic window (TDM essential, non-linear kinetics); (2) no evidence of superior efficacy vs CBZ/OXC in ADNFLE; (3) significant cognitive toxicity (cerebellar) with chronic use; (4) cosmetic side effects (gingival hyperplasia, facial coarsening, hirsutism); (5) teratogenicity; (6) drug interactions (CYP2C9/3A4 inducer). Overall inferior benefit-risk ratio vs CBZ/OXC for ADNFLE.",
        "alternative": "CBZ/OXC/LCM preferred. IV PHT acceptable only as acute SE rescue if IV CBZ/VPA/LEV unavailable. NOT for maintenance ADNFLE management."
    },
    {
        "drug": "Lamotrigine (LTG) Monotherapy — MODERATE RISK (Inadequate Frontal Coverage)",
        "risk": "MODERATE RISK",
        "mechanism": "LTG is a VGSC blocker effective for focal and generalised epilepsy but has INSUFFICIENT EFFICACY DATA for ADNFLE specifically. Multiple case series and clinical practice suggest LTG is less effective than CBZ/OXC for nocturnal frontal lobe seizures — possibly due to differential Na⁺-channel subtype profile or frontal lobe–specific pharmacodynamics. LTG monotherapy for ADNFLE risks inadequate seizure control. However, LTG as ADJUNCT (CBZ+LTG) has some supporting data and is acceptable.",
        "alternative": "Use CBZ or OXC as first-line monotherapy for ADNFLE. LTG may be used as adjunct (CBZ+LTG) or in HLA-B*15:02-positive patients where CBZ/OXC are contraindicated."
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING — 14 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "HLA-B*15:02 genotyping before CBZ/OXC initiation (SE Asian ancestry)", "frequency": "Once — BEFORE first CBZ or OXC dose", "rationale": "SJS/TEN risk in HLA-B*15:02 carriers given CBZ/OXC; prevalence 3–10% in SE Asian populations; fatality rate SJS/TEN 10–40%. CPIC Level A: test ALL SE Asian ancestry patients before CBZ/OXC. If HLA-B*15:02 positive → do NOT use CBZ or OXC → use LCM/ZNS instead."},
    {"item": "CBZ TDM (serum total carbamazepine, trough)", "frequency": "q3M steady-state; q4W during dose changes; target 4–12 mg/L (8–12 mg/L for ADNFLE nocturnal)", "rationale": "CBZ non-linear kinetics; autoinduction (CYP3A4) reduces own levels over first 4–6 weeks; TDM essential for optimal nocturnal coverage. ADNFLE patients may need higher-range TDM (8–12 mg/L) for complete seizure freedom."},
    {"item": "CBZ FBC (full blood count — agranulocytosis monitoring)", "frequency": "Baseline; q4W for first 3 months; q3M thereafter", "rationale": "CBZ causes agranulocytosis in <0.5% (neutrophil nadir at 3–6 weeks); aplastic anaemia rare but potentially fatal; leucopaenia more common (5–10%). Baseline WBC essential. If neutrophils <1.5 × 10⁹/L → reduce dose or switch."},
    {"item": "Serum sodium (hyponatraemia — CBZ/OXC SIADH)", "frequency": "Baseline; q3M (CBZ); q6W initially (OXC — higher risk)", "rationale": "Syndrome of inappropriate ADH secretion (SIADH) causes hyponatraemia in 5–10% CBZ (20–25% OXC). Severe hyponatraemia (Na⁺ <125 mmol/L) → seizures, encephalopathy. Fluid restriction; switch to LCM if persistent. OXC more hyponatraemic — Na⁺ monitoring q6W first year."},
    {"item": "Video-polysomnography (VPSG) — diagnostic and treatment monitoring", "frequency": "At diagnosis (captures ADNFLE semiology + EEG); repeat if treatment change or diagnostic uncertainty", "rationale": "VPSG is gold standard for ADNFLE diagnosis: documents NREM stage of seizures, frontal EEG onset, hypermotor semiology. Distinguishes ADNFLE from parasomnias. Post-treatment VPSG confirms seizure freedom vs subclinical continuation."},
    {"item": "Sleep diary (seizure + sleep quality) with video", "frequency": "Daily diary; video for each seizure type; weekly review with carer", "rationale": "ADNFLE seizures are nocturnal — patient often unaware. Carer/bed-partner observation essential. Diary captures minor motor events AND hypermotor seizures. Video (phone camera pointed at bed) is essential diagnostic tool. Count frequency before and after CBZ titration."},
    {"item": "CBZ LFT (liver function — first year)", "frequency": "Baseline; q3M for first year; q12M thereafter", "rationale": "CBZ hepatotoxicity rare but documented; risk highest first 3 months. ALT/AST >3× ULN → reduce dose; >10× ULN → stop CBZ. LFT monitoring mandatory in polypharmacy."},
    {"item": "CHRNA4/CHRNB2/CHRNA2 genetic panel + family cascade testing", "frequency": "Once at diagnosis; family members at risk (50% transmission per AD affected parent)", "rationale": "ADNFLE is AD 70% penetrance — first-degree relatives at 50% transmission risk. Cascade testing identifies affected and at-risk family members. Panel includes CHRNA4, CHRNB2, CHRNA2, KCNT1, DEPDC5 (differential for NFLE-like phenotype)."},
    {"item": "Neuropsychological assessment (cognition + psychiatric comorbidity)", "frequency": "At diagnosis; q12M for children; q24M adults", "rationale": "ADNFLE often associated with psychiatric comorbidity: nocturnal panic disorder (misdiagnosis common), OCD, hyperactivity. Formal neuropsychological testing identifies cognitive phenotype. ADNFLE patients frequently have normal IQ (unlike DEE) — cognitive assessment confirms baseline and monitors CBZ effects (memory, processing speed)."},
    {"item": "CBZ drug interaction review (CYP3A4 inducer)", "frequency": "At every medication change; annually in stable patients", "rationale": "CBZ is a potent CYP3A4 inducer — reduces levels of: hormonal contraceptives (OCP — switch to non-hormonal or higher-dose OCP + VPPP counselling), warfarin, statins, immunosuppressants, antiretrovirals, other AEDs (LTG, TPM levels reduced by CBZ). Annual medication reconciliation mandatory."},
    {"item": "VPPP (CBZ/OXC Pregnancy Prevention Programme — teratogenicity counselling)", "frequency": "Annual counselling for females of childbearing potential on CBZ/OXC", "rationale": "CBZ: spina bifida risk 0.5–1% (vs 0.06% background); neural tube defects + cardiac malformations. Not as high-risk as VPA but significant. Folic acid 5 mg/day pre-conception if CBZ. VPPP counselling mandatory. OXC: teratogenicity data less complete than CBZ — similar counselling. For conception: neurologist review + genetic counselling (ADNFLE AD inheritance 50% offspring risk)."},
    {"item": "Driving and occupational safety assessment (nocturnal seizures)", "frequency": "At diagnosis; after each change in seizure control", "rationale": "DVLA (UK)/transport authority notification required for nocturnal seizures — if seizures exclusively nocturnal AND patient has not had daytime seizure for ≥1 year AND meets local licensing criteria → can drive with medical supervision in some jurisdictions. Nocturnal seizures with diurnal breakthrough → driving prohibited. Occupational risk assessment: ADNFLE patients must avoid machinery, heights, water, driving until confirmed seizure-free per local regulations."},
    {"item": "SUDEP risk assessment and nocturnal safety measures", "frequency": "At diagnosis; annually", "rationale": "SUDEP risk in ADNFLE is lower than DEE/Dravet (seizures are less severe, no hypoxia typically) but not negligible for nocturnal GTCS during sleep. Nocturnal monitoring (pulse oximeter, seizure alarm mattress), prone sleeping position avoidance, bed partner education (recovery position), emergency plan."},
    {"item": "Ambulatory EEG / prolonged EEG-video monitoring", "frequency": "At diagnosis; if seizure type change; if pharmacoresistance develops", "rationale": "Ambulatory 24–72h EEG is useful if VPSG unavailable; captures nocturnal events; 50–70% capture rate in active ADNFLE. Prolonged in-patient video-EEG monitoring essential for surgical workup if drug-resistant ADNFLE (rare: <10% truly refractory, but frontal cortex resection/neuromodulation options exist)."},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE — 6 STAGES
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "stage": "Pre-symptomatic / At-Risk (0 – 7 years)",
        "key_action": "Family with known CHRNA4 pathogenic variant: genetic cascade testing for siblings and children. Pre-symptomatic genetic counselling (AD 50% transmission). Educate family to recognise first nocturnal events. Baseline developmental assessment. No prophylactic AED treatment pre-symptomatically."
    },
    {
        "stage": "Onset / Initial Diagnosis (7–17 years — Peak ADNFLE Onset)",
        "key_action": "Video-polysomnography (VPSG) for definitive diagnosis. CHRNA4/CHRNB2/CHRNA2 genetic panel + HLA-B*15:02 before CBZ. Start CBZ-XR with bedtime-heavy dosing (2/3 dose at bedtime). CBZ TDM at steady state (4–6 weeks). Sleep diary + video. VPPP counselling for females. Driving advice (nocturnal seizures)."
    },
    {
        "stage": "Treatment Optimisation (1–3 years post-onset)",
        "key_action": "CBZ dose optimisation to 8–12 mg/L trough for adequate nocturnal coverage. If partial responder → add LCM or ZNS. Repeat VPSG to confirm seizure freedom (objective, not just subjective). Psychiatric screening (panic disorder, OCD comorbidities). Sleep hygiene reinforcement. School/occupational assessment."
    },
    {
        "stage": "Established ADNFLE (3–15 years post-onset)",
        "key_action": "Long-term CBZ maintenance (ADNFLE is lifelong in most patients). Annual FBC/LFT/Na⁺/TDM monitoring. Annual VPPP counselling for females. Driving assessment annually (nocturnal-only seizures allow driving in many jurisdictions if ≥1yr seizure-free daytime). Psychiatric comorbidity review. Family planning: genetic counselling (50% offspring risk). Nicotine management if smoker."
    },
    {
        "stage": "Drug-Resistant ADNFLE (<10%) — Surgical / Neuromodulation Evaluation",
        "key_action": "Drug-resistant ADNFLE is rare (8–12% truly refractory to CBZ + 1 adjunct). Evaluation: high-resolution 3T MRI (FCD protocol), FDG-PET, ictal SPECT, neuropsychology. Frontal cortex resection (epilepsy surgery) for structural FCD — SEEG-guided; effective in 50–65% if focus localised. VNS (Level C) for non-resectable. CHRNA4 GOF precision drug development: investigational α4β2-selective nicotinic antagonists (not yet clinical)."
    },
    {
        "stage": "Adulthood / Long-term (25+ years)",
        "key_action": "Lifelong ADNFLE management in majority. Long-term CBZ review (cumulative toxicity monitoring: Na⁺, metabolic bone disease from CYP3A4 induction reducing Vitamin D, FBC). Pregnancy planning (neurologist + geneticist joint consultation; folic acid 5 mg pre-conception; CBZ lowest effective dose; AD offspring risk 50%). SUDEP education for significant others. Annual review of ADNFLE treatment landscape (precision therapies emerging)."
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DEFINITIONS — 15 KEY CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────
DEFINITIONS = [
    {"term": "CHRNA4 (20q13.33)", "definition": "Cholinergic receptor nicotinic alpha-4 subunit gene, chromosome 20q13.33. Encodes the α4 subunit of neuronal nicotinic acetylcholine receptors (nAChRs). ADNFLE mutations are gain-of-function (GOF), predominantly in the M2 transmembrane pore-lining segment. First epilepsy ion channel gene identified in a human epilepsy syndrome (Steinlein et al. 1995 Nat Genet — S284L mutation). OMIM gene *118504."},
    {"term": "ADNFLE (Autosomal Dominant Nocturnal Frontal Lobe Epilepsy)", "definition": "Distinctive epilepsy syndrome: hypermotor seizures arising exclusively or predominantly from NREM sleep; frontal lobe EEG onset; onset childhood/adolescence (7–17 years); AD inheritance ~70% penetrance. Caused by CHRNA4 (most common), CHRNB2, CHRNA2 mutations. ADNFLE is the first epilepsy syndrome attributed to an ion channel mutation (1995). OMIM #600513."},
    {"term": "nAChR α4β2 — Neuronal Nicotinic Acetylcholine Receptor", "definition": "The predominant neuronal nAChR in the CNS is the (α4)₂(β2)₃ heteropentamer, encoded by CHRNA4 (α4) + CHRNB2 (β2). Permeable to Na⁺, K⁺, Ca²⁺. The α4/β2 interface is the ACh and nicotine binding site. Highly expressed in frontal cortex layer V/VI neurons (pyramidal cells) and thalamo-cortical relay neurons. GOF → excessive depolarisation during cholinergic surges (NREM transitions) → frontal lobe seizure."},
    {"term": "S284L Mutation (Serine284Leucine) — Founding ADNFLE Mutation", "definition": "Serine-to-leucine substitution at position 284 in the M2 pore-lining domain of CHRNA4 α4 subunit. Discovered in 1995 (Steinlein et al. Nature Genetics) — the first human epilepsy ion channel mutation identified. Functional effect: slowed desensitisation → channels remain open longer after ACh activation → GOF. Accounts for ~20% of CHRNA4 ADNFLE cases in reported series."},
    {"term": "GOF Mechanism in ADNFLE — Delayed Desensitisation", "definition": "ADNFLE CHRNA4 mutations cause GOF by: (1) increased sensitivity to ACh (reduced EC₅₀); (2) impaired desensitisation (slowed transition to high-affinity refractory state). Net result: nAChR α4β2 responds more strongly and for longer to cholinergic surges → frontal cortex pyramidal neuron hyperexcitability during NREM sleep cholinergic transitions → ictal discharge propagating as hypermotor frontal lobe seizure."},
    {"term": "Nicotine Desensitisation Paradox", "definition": "CHRNA4 encodes the primary nicotine binding site (α4β2 is the high-affinity nicotine receptor). Low-dose sustained nicotine (patch 7–14 mg) → receptor desensitisation → net inhibitory effect on GOF nAChR → anecdotal seizure reduction (Brodtkorb 2002, Steinlein 2003). HIGH-DOSE acute nicotine (heavy smoking, high-dose NRT) → acute receptor ACTIVATION before desensitisation → potential seizure trigger. Abrupt smoking cessation → receptor upregulation → transient hypersensitivity. LOW-DOSE nicotine at bedtime is investigational; HIGH-DOSE is HIGH RISK."},
    {"term": "HLA-B*15:02 and CBZ/OXC SJS/TEN", "definition": "HLA-B*15:02 is a pharmacogenomic risk allele for Stevens-Johnson Syndrome (SJS) and Toxic Epidermal Necrolysis (TEN) induced by carbamazepine (and to lesser extent oxcarbazepine). Prevalence: Han Chinese 8–10%, Thai 8%, Malay 6%, Vietnamese 4%. SJS/TEN occurs within first 8 weeks of CBZ initiation; mortality 10–40%. CPIC Level A guideline: test ALL SE Asian ancestry patients before CBZ or OXC. HLA-B*15:02-positive → ABSOLUTE CI for CBZ/OXC. Critically important in ADNFLE since CBZ is first-line."},
    {"term": "NREM Sleep and ADNFLE — Cholinergic Surge Hypothesis", "definition": "During NREM sleep (stages 1–3), the brain oscillates between quiet NREM and brief micro-arousals associated with phasic cholinergic bursts from the basal forebrain (nucleus basalis of Meynert). In ADNFLE, GOF nAChR α4β2 in frontal layer V/VI neurons responds to these cholinergic micro-bursts with excessive depolarisation → synchronised frontal discharge → hypermotor seizure. Seizures cluster in first 2–3h of nocturnal NREM (highest homeostatic NREM pressure)."},
    {"term": "Video-Polysomnography (VPSG) in ADNFLE", "definition": "Simultaneous overnight sleep study with: (1) full EEG (32-channel), (2) video monitoring, (3) EOG (eye movements — REM/NREM distinction), (4) EMG (chin/leg), (5) respiratory monitoring (airflow, SpO₂, effort), (6) ECG. Gold standard for ADNFLE diagnosis: documents NREM stage at seizure onset, frontal EEG ictal correlate, hypermotor semiology. Distinguishes ADNFLE from REM sleep behavior disorder (REM onset), NREM parasomnia (deep NREM without ictal correlate), and OSA. Essential before epilepsy surgery evaluation."},
    {"term": "ADNFLE Misdiagnosis — Parasomnia Spectrum", "definition": "30–50% of ADNFLE patients are initially misdiagnosed as: (1) NREM parasomnia (sleepwalking, night terrors — same age group, similar nocturnal events, absence of EEG ictal discharge apparent on clinical assessment); (2) REM sleep behavior disorder (RBD — REM onset, dream enactment; ADNFLE is NREM and more stereotyped); (3) Nocturnal panic attacks (psychiatric history, CBT-directed); (4) OSA arousals (CPAP trial before epilepsy considered). KEY DISTINGUISHING FEATURES of ADNFLE: stereotypy (same sequence every event), frontal EEG onset on VPSG, CBZ response, family history of same events."},
    {"term": "CBZ Autoinduction (CYP3A4) in ADNFLE", "definition": "Carbamazepine is a potent CYP3A4 inducer — it induces its own metabolism over the first 4–6 weeks of treatment (autoinduction), reducing its own plasma levels by 30–50%. This means the initial therapeutic CBZ level measured at 4 weeks will be higher than the steady-state level at 6–8 weeks. TDM must be re-checked at 6–8 weeks (after autoinduction is complete) to ensure adequate nocturnal coverage. Additionally, CBZ reduces levels of hormonal contraceptives (OCP), warfarin, LTG, TPM — medication reconciliation mandatory."},
    {"term": "POLG1 / VPA-Alpers Hepatotoxicity (applicable if VPA adjunct considered)", "definition": "Although VPA is not first-line for ADNFLE, it is sometimes considered for adjunct generalised seizure coverage. POLG1 (polymerase gamma) pathogenic variants cause Alpers-Huttenlocher syndrome. VPA in POLG1 carriers → fatal hepatotoxicity. POLG1 testing recommended before any VPA initiation in any patient. If POLG1 positive → absolute CI for VPA → use LEV instead. For ADNFLE, CBZ/LCM/ZNS preferred."},
    {"term": "Frontal Lobe Epilepsy Surgery in Drug-Resistant ADNFLE", "definition": "True drug-resistant ADNFLE (<10% of patients) may be candidates for epilepsy surgery. Prerequisites: (1) VPSG with clear frontal onset; (2) high-resolution 3T MRI (FCD protocol — subtle FCD2B may be genetic ADNFLE mimicry with structural component); (3) SEEG implantation for precise surgical boundary; (4) neuropsychological testing. Resection outcomes: 50–65% seizure freedom if structural lesion; lower if purely genetic. VNS (Level C) for non-resectable. Emerging: DBS of anterior thalamus (SANTE trial) for frontal epilepsy."},
    {"term": "ADNFLE Penetrance and Intrafamilial Variability", "definition": "ADNFLE has ~70% penetrance — 30% of CHRNA4 variant carriers are clinically unaffected despite carrying the pathogenic variant. Intrafamilial variability is marked: within the same family, some members have frequent nightly seizures, others rare events, others are carriers without clinical epilepsy. Phenotype does NOT reliably predict genotype severity. This has implications for genetic counselling: offspring of affected patient have 50% variant transmission risk and 70% chance of clinical epilepsy if they carry the variant."},
    {"term": "ADNFLE in Pregnancy — Unique Considerations", "definition": "ADNFLE presents two concurrent genetic considerations in pregnancy: (1) MATERNAL RISK: CBZ/OXC teratogenicity (spina bifida 0.5–1%; folic acid 5 mg pre-conception; neurologist + obstetric review); nocturnal seizures less dangerous than awake GTCS but nocturnal monitoring during pregnancy recommended; (2) OFFSPRING GENETIC RISK: AD 50% transmission + 70% penetrance = ~35% absolute risk of clinical ADNFLE in offspring. Genetic counselling and prenatal/preimplantation genetic testing available for known CHRNA4 pathogenic variants."},
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS — 12 ACTION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "HLA-B*15:02 result unavailable and CBZ urgently needed in SE Asian patient", "action": "Use lacosamide (LCM) as bridge while awaiting HLA-B*15:02 result; do NOT start CBZ or OXC empirically in SE Asian ancestry patients without HLA-B*15:02 clearance; result typically available 3–5 business days"},
    {"threshold": "CBZ serum trough <8 mg/L in ADNFLE with continuing nocturnal seizures", "action": "Increase CBZ dose (increments of 200 mg, preferably at bedtime); recheck TDM at 4–6 weeks (autoinduction must be complete); bedtime dose should be 2/3 of total daily dose; check compliance (sleep diary confirms missed doses)"},
    {"threshold": "Serum Na⁺ <130 mmol/L on CBZ or OXC", "action": "Hold CBZ/OXC dose; fluid restriction; consider switch to LCM (no hyponatraemia risk); re-check Na⁺ in 48h; if Na⁺ <120 → hospitalise, IV NaCl (isotonic — avoid rapid correction: osmotic demyelination risk); endocrinology review for SIADH"},
    {"threshold": "Neutrophil count <1.5 × 10⁹/L on CBZ", "action": "Reduce CBZ dose by 25%; repeat FBC in 2 weeks; if neutrophils remain <1.0 × 10⁹/L → consider switching to OXC or LCM; haematology referral; CBZ-induced agranulocytosis risk: educate patient to report sore throat, fever immediately"},
    {"threshold": "Rash within 8 weeks of CBZ/OXC initiation", "action": "STOP CBZ/OXC IMMEDIATELY; assess for SJS/TEN: skin involvement >10% BSA + mucosal involvement → emergency dermatology; refer to burns unit if TEN; never rechallenge with CBZ or OXC; switch to lacosamide; consider HLA-B*15:02 testing retrospectively"},
    {"threshold": "Breakthrough nocturnal seizures on stable CBZ 8–12 mg/L trough", "action": "Rule out: missed doses (compliance check + TDM retest); drug interactions reducing CBZ levels (CYP3A4 inducers: rifampicin, St John's Wort); sleep disruption (polysomnography); consider adding lacosamide 100–200 mg at bedtime; VPSG to confirm break-through events are truly ADNFLE seizures vs parasomnias"},
    {"threshold": "Patient discloses smoking ≥20 cigarettes/day (heavy smoker) with active ADNFLE", "action": "Do NOT advise abrupt cessation (receptor upregulation → seizure cluster); plan gradual reduction over ≥8 weeks; if smoking cessation NRT: use low-dose patch 7–14 mg/24h (NOT gum or e-cigarettes with high bolus doses); increase CBZ monitoring during cessation transition; discuss nicotine desensitisation paradox with patient"},
    {"threshold": "Female patient with ADNFLE on CBZ reporting OCP failure (breakthrough bleeding)", "action": "CBZ autoinduction → reduces OCP efficacy (estrogen/progesterone reduced by CYP3A4); switch to non-hormonal contraception (IUD preferred — unaffected by CBZ) OR use higher-dose OCP (50 mcg ethinylestradiol) PLUS additional barrier contraception; VPPP counselling; document conversation."},
    {"threshold": "ADNFLE patient requiring emergency AED administration (status epilepticus) with CBZ as maintenance", "action": "IV lorazepam 0.1 mg/kg as first-line for convulsive status; IV lacosamide 200 mg over 15 min (CBZ-compatible, different mechanism); IV sodium valproate 25 mg/kg (if POLG1 excluded); if POLG1 unknown → IV LEV 60 mg/kg; IV fosphenytoin only as last resort (inferior frontal coverage, cerebellar toxicity risk)"},
    {"threshold": "VPSG confirms persistent nocturnal seizures at adequate CBZ TDM (≥8 mg/L) after 6 months", "action": "Formally diagnose CBZ-partial-resistance; add lacosamide 200 mg at bedtime; or switch to OXC-XR (check Na⁺); consider high-resolution 3T MRI (FCD protocol) and referral to epilepsy surgery centre for SEEG evaluation; CBZ + LCM combination is most evidence-based second-line for refractory ADNFLE"},
    {"threshold": "ADNFLE patient plans pregnancy (female, CBZ-maintained)", "action": "(1) Neurologist + geneticist joint consultation pre-conception; (2) folic acid 5 mg/day for ≥3 months before conception; (3) aim lowest effective CBZ dose; (4) discuss teratogenicity (spina bifida 0.5–1%); (5) 50% offspring ADNFLE inheritance risk; (6) prenatal testing options (NIPT + amniocentesis for CHRNA4 variant); (7) enhanced ultrasound at 16–20 weeks (spinal survey); (8) document VPPP discussion"},
    {"threshold": "Newly diagnosed ADNFLE in patient of SE Asian ancestry presenting for first CBZ prescription", "action": "MANDATORY: HLA-B*15:02 test BEFORE first CBZ or OXC dose; prescribe LCM as bridge AED (50 mg twice daily, safe, no SJS/TEN risk); HLA-B*15:02 result in 3–5 days; if NEGATIVE → start CBZ-XR (bedtime heavy dose); if POSITIVE → continue LCM long-term (do not use CBZ/OXC); document HLA-B*15:02 testing in notes"},
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDARDS — 12 GUIDELINES
# ─────────────────────────────────────────────────────────────────────────────
STANDARDS = [
    {"name": "ILAE Classification of Epilepsies 2022 (Scheffer et al.)", "applies": "ADNFLE classified under genetic focal epilepsy; CHRNA4/CHRNB2/CHRNA2 genetic aetiology; frontal lobe epilepsy classification; age-related onset criteria"},
    {"name": "NICE NG217 (2022) — Epilepsies: diagnosis and management", "applies": "Focal epilepsy first-line AED recommendations; CBZ as first-choice focal epilepsy monotherapy; sleep EEG/VPSG referral criteria; specialist referral guidelines; VPPP for CBZ"},
    {"name": "Steinlein et al. 1995 Nature Genetics (Founding CHRNA4 paper)", "applies": "Discovery of S284L mutation in the first ADNFLE pedigree; established nAChR as epilepsy channelopathy target; landmark pharmacogenomic reference for CHRNA4"},
    {"name": "CPIC Guideline HLA-B*15:02 + Carbamazepine 2023 (Phillips et al.)", "applies": "CPIC Level A pharmacogenomics guideline: mandatory HLA-B*15:02 testing before CBZ/OXC in SE Asian ancestry patients; management if HLA-B*15:02 positive; alternative AEDs"},
    {"name": "ILAE Task Force on Genetic Epilepsies 2018 (Guerrini et al.)", "applies": "Genetic testing recommendations for channelopathy syndromes; ADNFLE panel approach (CHRNA4/CHRNB2/CHRNA2); penetrance and phenotypic variability in AD epilepsies"},
    {"name": "Brodtkorb & Picard 2006 Epilepsy Behav (Nicotine ADNFLE review)", "applies": "Nicotine-ADNFLE interaction: desensitisation hypothesis; clinical reports of smoking effect on seizures; nicotine patch investigational data; basis for nicotine paradox counselling"},
    {"name": "AASM Sleep Disorders Classification 2023 (ICSD-3)", "applies": "Differential diagnosis of parasomnia vs nocturnal epilepsy; VPSG diagnostic criteria; NREM parasomnia vs ADNFLE distinguishing features; obstructive sleep apnoea co-morbidity"},
    {"name": "CPIC POLG-VPA Guideline 2023", "applies": "POLG1 testing before VPA initiation; fatal hepatotoxicity risk; LEV substitution protocol — applicable if VPA adjunct considered in ADNFLE"},
    {"name": "MHRA VPPP 2021 — Valproate Pregnancy Prevention Programme", "applies": "Applicable if VPA adjunct used; mandatory annual risk form + contraception for females ≥12y on VPA; CBZ pregnancy counselling analogously required (non-MHRA but NICE NG217 mandated)"},
    {"name": "NICE NG224 (2023) — Pharmacogenomics implementation", "applies": "Broader NHS pharmacogenomics implementation including HLA testing before AED initiation; point-of-care HLA testing services; prescribing support tools"},
    {"name": "ACMG-AMP Variant Interpretation Standards 2015 (Richards et al.)", "applies": "CHRNA4 variant classification (pathogenic/likely pathogenic/VUS); GOF functional evidence classification; segregation analysis in AD families; ClinVar submission standards"},
    {"name": "WHO ICF 2019 (International Classification of Functioning)", "applies": "ADNFLE disability framework: impact of nocturnal epilepsy on sleep quality, daytime cognition, driving, occupational function, family dynamics; participation-focused care planning beyond seizure control"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES — 6 KEY
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"author": "Steinlein OK et al.", "year": 1995, "journal": "Nat Genet 11:201–203", "title": "A missense mutation in the neuronal nicotinic acetylcholine receptor alpha 4 subunit is associated with autosomal dominant nocturnal frontal lobe epilepsy", "pmid": "7550350"},
    {"author": "Hirose S et al.", "year": 1999, "journal": "Ann Neurol 45:269–272", "title": "A novel mutation of CHRNA4 responsible for autosomal dominant nocturnal frontal lobe epilepsy", "pmid": "9989630"},
    {"author": "Brodtkorb E & Picard F", "year": 2006, "journal": "Epilepsy Behav 9:550–554", "title": "Tobacco habits modulate autosomal dominant nocturnal frontal lobe epilepsy", "pmid": "17010681"},
    {"author": "Tinuper P et al.", "year": 2016, "journal": "Neurology 86:1826–1833", "title": "Definition and diagnostic criteria of sleep-related hypermotor epilepsy", "pmid": "27164709"},
    {"author": "Scheffer IE et al.", "year": 2014, "journal": "Epilepsia 55:1635–1642", "title": "Self-limited epilepsy with nocturnal frontal lobe seizures: autosomal dominant and sporadic forms", "pmid": "25082128"},
    {"author": "Phillips HA et al.", "year": 2000, "journal": "Hum Mol Genet 9:2109–2117", "title": "ADNFLE mutations increase the Ca2+ permeability of the neuronal nicotinic acetylcholine receptor", "pmid": "10958658"},
]

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC COHORT — 40 PATIENTS
# ─────────────────────────────────────────────────────────────────────────────
_etiology_pool = (
    ["GOF-Missense-S284L-Classic-ADNFLE"] * 14 +
    ["GOF-Missense-Other-M2-Domain-Variants"] * 10 +
    ["GOF-Insertion-776insL-Australian-Scottish"] * 8 +
    ["GOF-Atypical-Non-M2-or-New-Variant"] * 6 +
    ["Phenocopy-ADNFLE-Panel-Negative"] * 2
)

_aed_pool = [
    ["CBZ"], ["CBZ-XR"], ["CBZ-XR", "LCM"],
    ["CBZ", "CLB"], ["OXC", "LCM"], ["CBZ-XR", "ZNS"],
    ["LCM", "ZNS"], ["CBZ", "TPM"], ["OXC"],
    ["CBZ-XR", "LCM", "CLB"], ["CBZ", "LEV"], ["OXC", "CLB"],
    ["LCM"], ["CBZ", "ZNS"], ["CBZ-XR", "CLB"],
]

_outcomes = ["seizure-free", "≥50%-reduction", "partial-response", "DRE"]
_outcome_wts = [0.55, 0.22, 0.16, 0.07]

_sexes = ["M", "F"] * 20
random.shuffle(_sexes)

_cohort = []
for i in range(40):
    et = _etiology_pool[i]
    is_s284l = "S284L" in et
    is_776 = "776insL" in et
    is_phenocopy = "Phenocopy" in et
    age_onset = random.randint(7, 19)
    current_age = age_onset + random.randint(2, 18)
    aeds = _aed_pool[i % len(_aed_pool)]
    outcome = random.choices(_outcomes, weights=_outcome_wts)[0]
    cbz_use = any("CBZ" in a for a in aeds)
    lcm_use = "LCM" in aeds
    hypermotor = not is_phenocopy or random.random() < 0.85
    nocturnal_gtcs = (is_776 and random.random() < 0.30) or (not is_776 and random.random() < 0.15)
    misdiagnosed_parasomnia = random.random() < 0.35
    smoker = random.random() < 0.22
    hla_tested = random.random() < 0.60
    _cohort.append({
        "patient_id": f"P{i+1:03d}",
        "age_onset_years": age_onset,
        "current_age_years": current_age,
        "sex": _sexes[i],
        "etiology": et,
        "seizure_free": outcome == "seizure-free",
        "dre": outcome == "DRE",
        "outcome": outcome,
        "aeds": aeds,
        "hypermotor_seizures": hypermotor,
        "nocturnal_gtcs": nocturnal_gtcs,
        "diurnal_seizures": random.random() < 0.15,
        "misdiagnosed_parasomnia": misdiagnosed_parasomnia,
        "cbz_use": cbz_use,
        "lcm_use": lcm_use,
        "smoker": smoker,
        "hla_b1502_tested": hla_tested,
        "vpsg_confirmed": random.random() < 0.70,
        "genetic_panel_positive": not is_phenocopy,
        "family_history_adnfle": random.random() < 0.72,
        "cbz_hyponatraemia": cbz_use and random.random() < 0.08,
        "cbz_rash": cbz_use and random.random() < 0.10,
    })


# ─────────────────────────────────────────────────────────────────────────────
# API RESPONSE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_overview():
    n = len(_cohort)
    seizure_free = sum(1 for p in _cohort if p["seizure_free"])
    dre = sum(1 for p in _cohort if p["dre"])
    hypermotor = sum(1 for p in _cohort if p["hypermotor_seizures"])
    misdiag = sum(1 for p in _cohort if p["misdiagnosed_parasomnia"])
    cbz_users = sum(1 for p in _cohort if p["cbz_use"])
    vpsg_conf = sum(1 for p in _cohort if p["vpsg_confirmed"])
    genetic_pos = sum(1 for p in _cohort if p["genetic_panel_positive"])
    family_hx = sum(1 for p in _cohort if p["family_history_adnfle"])

    etiology_dist = {}
    for et_spec in ETIOLOGY_CATALOG:
        count = sum(1 for p in _cohort if p["etiology"] == et_spec["category"])
        etiology_dist[et_spec["category"]] = count

    seizure_summary = [
        {"type": s["type"], "frequency_pct": s["frequency_pct"]}
        for s in SEIZURE_TYPES
    ]
    treatment_summary = [
        {"drug": t["drug"].split(" —")[0].split(" (")[0][:45], "level": t["level"].split(" (")[0]}
        for t in TREATMENTS
    ]
    monitoring_summary = [
        {"item": m["item"].split(" (")[0][:60], "frequency": m["frequency"].split(";")[0]}
        for m in MONITORING[:8]
    ]
    lifecycle_summary = [
        {"stage": lc["stage"][:70], "key_action": lc["key_action"][:100]}
        for lc in LIFECYCLE
    ]

    return {
        "gene": "CHRNA4",
        "locus": "20q13.33",
        "protein": "Nicotinic Acetylcholine Receptor Alpha-4 Subunit (nAChR α4) — (α4)₂(β2)₃ heteropentamer",
        "channel_role": "Ligand-gated ion channel; ACh/nicotine-gated; permeable to Na⁺/K⁺/Ca²⁺; GOF → delayed desensitisation → frontal hyperexcitability in NREM sleep",
        "syndrome": "ADNFLE — Autosomal Dominant Nocturnal Frontal Lobe Epilepsy (first epilepsy channelopathy, 1995)",
        "gene_omim": "*118504 CHRNA4 gene · #600513 ADNFLE syndrome",
        "inheritance": "Autosomal Dominant (AD); ~70% penetrance; intrafamilial variability; 50% offspring transmission risk",
        "color": "#6d4c41",
        "precision_tx": "No approved precision nAChR-specific therapy. CBZ first-line (70–80% seizure freedom). Nicotine patch investigational (desensitisation paradox). HLA-B*15:02 mandatory before CBZ in SE Asian.",
        "total_patients": n,
        "seizure_free_pct": round(seizure_free / n * 100, 1),
        "dre_pct": round(dre / n * 100, 1),
        "hypermotor_count": hypermotor,
        "misdiagnosed_parasomnia_count": misdiag,
        "cbz_users_count": cbz_users,
        "vpsg_confirmed_count": vpsg_conf,
        "genetic_panel_positive_count": genetic_pos,
        "family_history_count": family_hx,
        "etiology_distribution": etiology_dist,
        "seizure_summary": seizure_summary,
        "treatments_summary": treatment_summary,
        "monitoring_summary": monitoring_summary,
        "lifecycle": lifecycle_summary,
        "thresholds": THRESHOLDS[:6],
        "contraindications_summary": [
            {"drug": ci_["drug"].split(" —")[0].split(" (")[0][:55], "risk": ci_["risk"]}
            for ci_ in CONTRAINDICATIONS[:6]
        ],
    }


def get_breakdown():
    return {
        "etiologies": ETIOLOGY_CATALOG,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "seizure_catalog": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "treatment_catalog": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "patients": _cohort,
    }


def get_definitions():
    return {
        "gene_summary": {
            "gene": "CHRNA4",
            "full_name": "Cholinergic Receptor Nicotinic Alpha 4 Subunit",
            "chromosome": "20q13.33",
            "protein": "nAChR α4 subunit — forms (α4)₂(β2)₃ heteropentamer with CHRNB2; 627 aa; 4 TM segments; M2 pore-lining (most mutations here)",
            "function": "Ligand-gated ion channel: ACh and nicotine binding → channel opening → Na⁺/K⁺/Ca²⁺ flux → neuronal depolarisation. ADNFLE GOF: slowed desensitisation → prolonged activation during NREM cholinergic surges → frontal ictal discharge.",
            "expression": "Frontal cortex layer V/VI pyramidal neurons, thalamo-cortical relay neurons, hippocampus, basal ganglia. Highest frontal density accounts for ADNFLE phenotype.",
            "animal_model": "CHRNA4 S284L knock-in mice (Teper 2007 J Neurosci): reduced duration and amplitude of nicotinic EPSPs in cortical neurons + increased frontal excitability during NREM-equivalent sleep states.",
            "inheritance": "AD; ~70% penetrance; intrafamilial variability; 30% carriers unaffected; 50% offspring risk",
            "gene_omim": "OMIM *118504 CHRNA4 gene · #600513 ADNFLE syndrome",
            "syndrome": "ADNFLE — Autosomal Dominant Nocturnal Frontal Lobe Epilepsy",
            "key_distinction": "FIRST EPILEPSY ION CHANNEL GENE (1995). Nicotinic receptor GOF → nocturnal hypermotor seizures. CBZ first-line (70–80%). HLA-B*15:02 MANDATORY before CBZ in SE Asian. NICOTINE PARADOX: low-dose patch desensitises GOF receptor; high-dose activates it. Often MISDIAGNOSED as parasomnia (30–50%).",
            "nachr_subfamily": "nAChR subfamily: α4 (CHRNA4, ADNFLE) · β2 (CHRNB2, ADNFLE) · α2 (CHRNA2, ADNFLE rare) · α7 (CHRNA7, cognitive/inflammatory) · α3β4 (autonomic ganglion) · α1β1γδ (NMJ — myasthenia gravis target)",
            "absolute_ci": "CBZ/OXC + HLA-B*15:02 (SJS/TEN fatal) · TGB (NCSE) · VPA+POLG1 (Alpers) · High-dose acute nicotine (GOF activation) · PHT maintenance (inferior FLE efficacy)",
        },
        "definitions": DEFINITIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
