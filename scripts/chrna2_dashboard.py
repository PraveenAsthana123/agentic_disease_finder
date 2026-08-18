"""
CHRNA2 Epilepsy — Autosomal Dominant Nocturnal Frontal Lobe Epilepsy Type 2 (ADNFLE2)
nAChR α2 Subunit / (α2)₂(β4)₃ / GOF Delayed Desensitisation / Rarest ADNFLE Gene
CBZ-XR First-Line / HLA-B*15:02 / Frontal Lobe / 8p21.2
=======================================================================
40-patient cohort · CHRNA2 (8p21.2) · Cholinergic Receptor Nicotinic Alpha 2 Subunit
Gene OMIM: *118502 · Syndrome: ADNFLE2 (OMIM #610353)
Rarest ADNFLE gene — CHRNA4 (ADNFLE1) >> CHRNB2 (ADNFLE3) >> CHRNA2 (ADNFLE2)
Completing the ADNFLE nAChR triad.

KEY CHRNA2 BIOLOGY — nAChR α2 SUBUNIT:
CHRNA2 (8p21.2) encodes the α2 subunit of the neuronal nicotinic acetylcholine receptor (nAChR).
Unlike CHRNA4 (α4) which partners with CHRNB2 (β2) in the high-sensitivity (α4)₂(β2)₃ heteropentamer,
CHRNA2 (α2) preferentially assembles with CHRNB4 (β4) as (α2)₂(β4)₃ and also with β2.
The (α2)β4-containing receptors are predominantly expressed in:
  · habenulo-interpeduncular pathway (most enriched α2 expression in brain)
  · medial habenula → interpeduncular nucleus axis (reward/aversion/fear)
  · layer 5/6 of prefrontal/frontal cortex (interneurons + pyramidal cells)
  · thalamic relay nuclei (limited)
CONTRAST with CHRNA4/CHRNB2 which are predominantly expressed in:
  · thalamus, cortex (widespread frontal > temporal), hippocampus

PROTEIN STRUCTURE:
  · Pentameric ligand-gated ion channel (pLGIC); Cys-loop superfamily
  · 4 TM helices (TM1-TM4); TM2 lines the cation-selective pore
  · I279 (Ile279) — TM2 position 6' — FIRST reported CHRNA2 pathogenic variant (I279N, Aridon 2006 AJHG)
  · TM3 harbors I304N — the second reported variant
  · ACh binding site: formed at subunit interfaces in ECD (α2-β4 interface)
  · Lower acetylcholine sensitivity than (α4)₂(β2)₃: EC50 ~50–100 µM vs ~1 µM for (α4)₂(β2)₃

BIOPHYSICS — GOF MECHANISM (same class as CHRNA4/CHRNB2):
  · Wild-type: opens → rapid desensitisation (ms-seconds) → non-conducting high-affinity state
  · I279N/I304N: TM2/TM3 substitution → REDUCED DESENSITISATION → channels remain open longer
  · Habenulo-interpeduncular: prolonged activation → interpeduncular nucleus hyperexcitability
    → ascending disinhibition of frontal cholinergic drive during NREM → frontal seizures
  · NREM-SPECIFIC: PPT/LDT brainstem cholinergic fire in NREM → ACh surge → GOF nAChR fails to desensitise
  · Ca²⁺ permeability: (α2)β4 has HIGHER Ca²⁺ permeability than (α4)₂(β2)₃ → potentially more
    glutamate release → stronger frontal hyperexcitability per receptor activation event

PATHOGENIC VARIANTS (all rare — CHRNA2 is rarest ADNFLE gene):
  · I279N (Ile279Asn, c.836T>A): TM2 position 6'; first variant (Aridon 2006); Italian family; GOF
  · I304N (Ile304Asn, c.911T>A): TM3 position; second variant (Conti 2015 Epilepsia); Italian family; GOF
  · V337G (Val337Gly): TM3 distal; Turkish family; GOF; pure ADNFLE phenotype
  · Y252C (Tyr252Cys): ECD-TM1 linker; rare; GOF; mild phenotype
  · Note: <10 families reported worldwide — rarest of the 3 ADNFLE genes

ADNFLE GENE HIERARCHY (by number of reported families):
  1. CHRNA4 (α4, 20q13.33, ADNFLE1, #600513): >100 families — MOST COMMON; pure ADNFLE
  2. CHRNB2 (β2, 1q21.3, ADNFLE3, #605375): ~20-25 families; higher psychiatric comorbidity
  3. CHRNA2 (α2, 8p21.2, ADNFLE2, #610353): <10 families — RAREST; habenular pathway

CLINICAL DIFFERENCES FROM CHRNA4 / CHRNB2:
  · Seizure semiology: essentially identical ADNFLE (hypermotor nocturnal from NREM)
  · Psychiatric comorbidity: similar to CHRNA4 (~20%); lower than CHRNB2 (40%)
  · Cognitive impairment: no specific mutation-linked cognitive risk (unlike CHRNB2 V287M)
  · Habenular involvement: may contribute to rare interictal mood/anxiety symptoms (habenula = depression circuit)
  · Family size: VERY small families (1-2 affected members typically); de novo rate uncertain
  · Misdiagnosis: same 35% parasomnia misclassification; some families diagnosed for years as NFLE NOS before gene found
  · CBZ response: similar to CHRNA4/CHRNB2 (65-75% seizure-free with CBZ-XR)
  · DRE rate: ~25-30% drug-resistant (similar to CHRNA4)

KEY PHARMACOLOGICAL DISTINCTIONS:
  · SAME FRAMEWORK as CHRNA4/CHRNB2: CBZ-XR first-line; HLA-B*15:02 absolute CI; VPSG gold standard
  · HABENULAR AXIS: the habenulo-interpeduncular pathway's role in reward/aversion means
    untreated CHRNA2 ADNFLE may have higher rate of nicotine dependence and mood dysregulation
    (habenula dysfunction → reward pathway dysregulation); screen and manage
  · NICOTINE PARADOX: SAME as CHRNA4/CHRNB2 — low-dose sustained nicotine patch desensitises GOF
    receptor (investigational); HIGH-DOSE activates → HIGH RISK; nicotine cessation → receptor
    upregulation → seizure cluster
  · KCNQ2/3 INTERACTION: frontal-lobe hyper-excitable circuits overlap → Na-channel blockers
    (OXC, LCM) useful adjuncts when CBZ partial response
  · NO BUPROPION: same absolute CI as CHRNB2 (lowers seizure threshold)
  · VARENICLINE: same HIGH RISK (partial nicotinic agonist — may activate GOF α2 receptor;
    CHRNA2 expressed in habenulo-interpeduncular nicotine reward circuit → varenicline
    pharmacological effect more directly involves GOF receptor than in CHRNA4/CHRNB2)
"""

import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG — 5-CLASS SPECTRUM
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "GOF-I279N-TM2-Classic-ADNFLE2",
        "pct": 42,
        "mechanism": (
            "I279N (Ile279Asn, c.836T>A) missense in TM2 (position 6') — the pore-lining helix. "
            "First reported CHRNA2 pathogenic variant (Aridon et al. 2006 Am J Hum Genet). "
            "Identified in an Italian ADNFLE family; reduces nAChR α2β4 desensitisation rate → "
            "channels remain open during NREM cholinergic surges in habenulo-interpeduncular pathway "
            "→ frontal cortex hyperexcitability → nocturnal hypermotor seizures. "
            "Autosomal dominant; penetrance estimated ~75-85% (small family size limits estimate). "
            "CLINICAL NOTE: Pure ADNFLE phenotype; no specific cognitive or psychiatric signature "
            "beyond standard ADNFLE (20% anxiety/depression). "
            "TREATMENT: CBZ-XR 200-400 mg bedtime-heavy; 65-70% seizure-free. "
            "HLA-B*15:02 screening mandatory before CBZ/OXC in SE Asian populations."
        ),
        "eeg": "NREM Stage 2–3 arousal: frontal theta burst (6-9 Hz, 5-15 s) → hypermotor arousal; interictal EEG typically normal; VPSG critical; ictal scalp EEG often obscured by movement artifact from frontal hyperkinesis",
        "onset_months": "72–216 months (6–18 years; peak 10–14 y)",
        "severity": "moderate — CBZ-XR responsive 65-70%; 25-30% drug-resistant requiring combination therapy",
    },
    {
        "category": "GOF-I304N-TM3-ADNFLE2-Variant2",
        "pct": 28,
        "mechanism": (
            "I304N (Ile304Asn, c.911T>A) missense in TM3 — second reported CHRNA2 variant "
            "(Conti et al. 2015 Epilepsia). Italian family. TM3 position contributes to "
            "inter-subunit interactions and desensitisation gate; I304N → GOF by similar "
            "mechanism to I279N (reduced closed-state stabilisation → prolonged channel opening). "
            "Pure ADNFLE phenotype; no cognitive comorbidity specifically reported in this family. "
            "PRECISION TREATMENT: CBZ-XR first-line; OXC alternative if CBZ intolerance; "
            "same HLA-B*15:02 absolute CI applies to OXC."
        ),
        "eeg": "NREM-onset frontal theta / low-amplitude fast; hypermotor semiology; VPSG mandatory; interictal normal; SEEG if drug-resistant confirms habenular-frontal network",
        "onset_months": "84–240 months (7–20 years)",
        "severity": "moderate — pure ADNFLE; CBZ responsive in 60-65% of reported family members",
    },
    {
        "category": "GOF-V337G-TM3-Turkish-Family",
        "pct": 18,
        "mechanism": (
            "V337G (Val337Gly) missense in distal TM3. Reported in a Turkish ADNFLE family. "
            "Glycine substitution introduces conformational flexibility into the rigid TM3 helix "
            "→ impaired inter-subunit packing → reduced desensitisation (GOF by loss of "
            "desensitisation gate stability rather than direct pore-lining disruption). "
            "Pure ADNFLE; normal cognition and psychiatry in reported carriers. "
            "TREATMENT: Same CBZ-XR framework. Higher DRE rate in this variant cluster (~30%); "
            "KD or LCM adjunct used in drug-resistant cases."
        ),
        "eeg": "NREM frontal theta arousal identical to I279N/I304N; hypermotor with occasional tonic posturing; nocturnal clustering typical; VPSG gold standard",
        "onset_months": "96–252 months (8–21 years)",
        "severity": "moderate — slightly higher DRE rate (~30%); KD beneficial in resistant cases",
    },
    {
        "category": "GOF-Other-Rare-Variants-CHRNA2",
        "pct": 8,
        "mechanism": (
            "Other rare CHRNA2 GOF variants including Y252C (Tyr252Cys, ECD-TM1 linker) and "
            "novel variants identified via gene panels in ADNFLE families negative for CHRNA4/CHRNB2. "
            "Y252C disrupts a conserved aromatic residue in the ECD-TM1 region → impairs receptor "
            "folding/trafficking and potentially alters gating kinetics (mild GOF or altered assembly). "
            "These families are extremely rare; evidence from <3 published cases each. "
            "CLINICAL: Phenotype consistent with ADNFLE; seizure frequency variable. "
            "Management: empirical CBZ-XR; monitor for DRE; VPSG for diagnosis; gene panel re-test "
            "as functional studies emerge for novel variants."
        ),
        "eeg": "NREM frontal arousal; same ADNFLE EEG signature; functional MRI/SEEG more critical for rare variants to map network",
        "onset_months": "60–288 months (5–24 years; variable)",
        "severity": "variable — limited case data; most respond to CBZ-XR; DRE in ~25%",
    },
    {
        "category": "Phenocopy-ADNFLE-CHRNA2-Negative",
        "pct": 4,
        "mechanism": (
            "ADNFLE clinical phenotype (nocturnal hypermotor NREM seizures, frontal EEG, VPSG-confirmed) "
            "without pathogenic CHRNA2/CHRNA4/CHRNB2 variant. Phenocopy causes include: "
            "(1) KCNT1 GOF — occasionally presents with nocturnal frontal semiology; "
            "(2) DEPDC5 LOF — GATOR1 mTOR pathway; focal FCD with frontal origin; "
            "(3) CABP4 — calcium-binding protein 4; rare nocturnal frontal epilepsy; "
            "(4) CRH — corticotropin-releasing hormone region; "
            "(5) Structural FCD IIa/IIb in frontal lobe (cortical malformation — MTOR/PIK3CA/AKT3 somatic). "
            "DIAGNOSTIC APPROACH: gene panel (CHRNA4/CHRNB2/CHRNA2/KCNT1/DEPDC5) + 3T MRI FCD protocol "
            "mandatory before labelling as 'ADNFLE idiopathic'."
        ),
        "eeg": "Indistinguishable from true ADNFLE on VPSG; FCD phenocopy may show focal interictal discharge; SEEG+structural MRI differential",
        "onset_months": "48–240 months (4–20 years)",
        "severity": "variable — depends on underlying cause; FCD phenocopy may be surgical candidate",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SEIZURE CATALOG — 5 TYPES
# ─────────────────────────────────────────────────────────────────────────────
SEIZURE_CATALOG = [
    {
        "type": "Hypermotor-Nocturnal-NREM-ADNFLE2",
        "pct": 96,
        "description": (
            "Nocturnal hypermotor seizures: abrupt arousal from NREM sleep (usually Stage 2) → "
            "thrashing, cycling leg movements, asymmetric tonic posturing, running, "
            "reaching, screaming or vocalising. Duration 20-120 s. "
            "Patient typically unresponsive during event; postictal confusion <2 min. "
            "Identical to CHRNA4/CHRNB2 semiology — gene cannot be identified from semiology alone. "
            "VPSG key: captures NREM onset, frontal EEG pattern, and distinguishes from parasomnia."
        ),
        "eeg": "NREM arousal: frontal theta burst 6-9 Hz (5-15 s); often obscured by movement artifact; interictal: rare frontal sharp transients or normal",
        "semiology_tips": "Ask about vocalisation (screaming/talking), stereotypy (same movements each night), clustering (multiple events per night), witness account essential",
        "emergency": False,
    },
    {
        "type": "Minor-Motor-Paroxysmal-Episodes",
        "pct": 70,
        "description": (
            "Briefer episodes (~10-30 s): abrupt arousal with stereotyped minor movements — "
            "hand fumbling, sitting up, looking around, brief dystonic posturing of one arm. "
            "Patient may partially remember event. Often occur in clusters (3-10 per night). "
            "May precede full hypermotor seizures or occur independently. "
            "Underrecognised: families often report only 'restless nights' until VPSG performed."
        ),
        "eeg": "Briefer NREM arousal; frontal theta 5-10 s; may be too brief to capture on routine EEG",
        "semiology_tips": "Family diary: count total episodes including brief arousals; not just major hypermotor events; VPSG captures all",
        "emergency": False,
    },
    {
        "type": "Nocturnal-Tonic-Focal",
        "pct": 48,
        "description": (
            "Tonic asymmetric posturing from NREM: fencing posture (ipsilateral arm extension, "
            "contralateral arm flexion — 'figure-of-4'), head/eye deviation, sustained 20-60 s. "
            "Implies supplementary motor area (SMA) involvement. "
            "Distinguishes ADNFLE from generalized epilepsy: lateralising features suggest focal onset. "
            "SEEG if drug-resistant: confirms SMA/lateral frontal network with habenular-frontal drive."
        ),
        "eeg": "NREM: fast frontal discharge → sustained theta/alpha; asymmetric if SMA strongly involved; SEEG shows SMA → motor cortex propagation",
        "semiology_tips": "Video review essential: figure-of-4, head turning, leg extension pattern; lateralisation helps surgical planning",
        "emergency": False,
    },
    {
        "type": "Epileptic-Wandering-Ambulatory",
        "pct": 30,
        "description": (
            "Epileptic wandering: semi-purposeful ambulatory behaviours during NREM seizure — "
            "patient gets out of bed, walks into walls, picks up objects, returns to bed. "
            "Appears purposeful but is ictal automatism. Duration 30-120 s. "
            "Indistinguishable from sleepwalking to untrained observer — major reason for "
            "35% parasomnia misdiagnosis in ADNFLE. VPSG: ictal vs non-ictal disambiguation."
        ),
        "eeg": "NREM frontal discharge with limited interictal changes; amplitude asymmetry if unilateral frontal onset",
        "semiology_tips": "Safety: bed rails, room safety measures, bedroom door alarm for wandering; EEG-verified wandering = ictal, needs AED",
        "emergency": False,
    },
    {
        "type": "Daytime-Focal-Aware-Rare",
        "pct": 15,
        "description": (
            "Rare daytime events in ~15% of CHRNA2 patients: brief focal aware or impaired-awareness "
            "frontal seizures — sudden motor arrest, head turning, brief automatisms. "
            "Often triggered by sleep deprivation, stress, or missed CBZ dose. "
            "Unlike CHRNA4/CHRNB2 which are almost exclusively nocturnal, CHRNA2 habenular "
            "circuit involvement may contribute marginally more daytime excitability. "
            "IMPORTANT: if daytime seizures prominent → re-evaluate for FCD phenocopy (SEEG)."
        ),
        "eeg": "Daytime: focal frontal fast discharge → theta; brief duration 10-30 s; harder to capture on routine EEG",
        "semiology_tips": "If frequent daytime seizures — reconsider CHRNA2 diagnosis; FCD or DEPDC5 more likely; 3T MRI FCD protocol",
        "emergency": False,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER CATALOG — 8 TRIGGERS
# ─────────────────────────────────────────────────────────────────────────────
TRIGGER_CATALOG = [
    {
        "trigger": "NREM-Sleep-Transitions-Cholinergic-Surge",
        "pct": 95,
        "mechanism": "PPT/LDT brainstem cholinergic nuclei fire during NREM → ACh surge → GOF α2 fails to desensitise → frontal hyperexcitability",
        "management": "CBZ-XR bedtime-heavy (2/3 at bedtime) to maximise NREM drug coverage; VPSG-guided timing",
    },
    {
        "trigger": "Sleep-Deprivation",
        "pct": 82,
        "mechanism": "Sleep deprivation → sleep rebound → increased NREM Stage 3 proportion → more NREM transitions → more cholinergic surge events",
        "management": "Sleep hygiene strict: ≥7.5 h/night; consistent sleep schedule; avoid shift work; parents/carers to monitor",
    },
    {
        "trigger": "Stress-Catecholamine",
        "pct": 65,
        "mechanism": "Stress → norepinephrine/cortisol → sleep fragmentation → increased NREM transitions; indirect trigger",
        "management": "Stress management; CBZ TDM during high-stress periods; avoid missed doses",
    },
    {
        "trigger": "Missed-CBZ-Dose",
        "pct": 62,
        "mechanism": "CBZ missed at bedtime → subtherapeutic level during NREM window → seizure cluster; CBZ half-life ~20-24 h; single missed dose drops level below therapeutic threshold",
        "management": "Pill organiser; phone alarm; partner verification; 'catch-up' bedtime dose within 4h; DO NOT double next morning dose",
    },
    {
        "trigger": "Febrile-Illness",
        "pct": 50,
        "mechanism": "Fever → sleep fragmentation + metabolic acceleration → increased nAChR opening probability (temperature-dependent gating); also pyrogen effects on cholinergic neurons",
        "management": "Anti-pyretics early; extra vigilance during illness; consider temporary CBZ dose bump (+50 mg bedtime) during fever per protocol",
    },
    {
        "trigger": "Alcohol-Evening-Intake",
        "pct": 38,
        "mechanism": "Alcohol → GABAergic sedation during waking → rebound NREM fragmentation and Stage 3 increase later in night → cholinergic surge increase",
        "management": "Abstinence or strict limit (≤1 unit); evening alcohol particularly harmful; educate re: NREM rebound effect",
    },
    {
        "trigger": "Nicotine-Cessation-Receptor-Upregulation",
        "pct": 28,
        "mechanism": "Abrupt nicotine cessation → α2β4 nAChR upregulation (nicotine withdrawal = receptor proliferation) → increased GOF receptor density → seizure cluster; habenular α2 enrichment makes CHRNA2 particularly sensitive to nicotine fluctuations",
        "management": "NEVER abrupt nicotine cessation: gradual taper under neurology + addiction medicine co-management; NRT maximum 7-mg patch; AVOID varenicline (HIGH RISK α2β4 partial agonist); consult neurology before any smoking cessation therapy",
    },
    {
        "trigger": "Circadian-Phase-Disruption",
        "pct": 22,
        "mechanism": "Travel across time zones, shift work, irregular schedule → circadian misalignment → NREM timing shifted away from bedtime CBZ coverage → window of unprotected NREM",
        "management": "Jet lag protocol: advance bedtime CBZ timing; melatonin 0.5 mg to re-anchor circadian; gradual schedule shift; consult neurology before transmeridian travel",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENT CATALOG — 8 TREATMENTS
# ─────────────────────────────────────────────────────────────────────────────
TREATMENT_CATALOG = [
    {
        "drug": "Carbamazepine-XR (CBZ-XR)",
        "evidence": "Level B",
        "role": "First-line — ADNFLE2",
        "mechanism": "Na-channel block (fast + slow inactivation); frontal cortex stabilisation; NREM seizure suppression; bedtime-heavy dosing maximises NREM coverage",
        "dose": "200-400 mg/day total; bedtime-heavy (2/3 at bedtime, 1/3 morning); XR preferred for stable nocturnal levels; target TDM 8-12 µg/mL",
        "monitoring": "HLA-B*15:02 BEFORE starting in SE Asian (SJS/TEN absolute CI — CPIC Level A); TDM at 6-8 weeks (autoinduction CYP3A4 drops level 30-50%); FBC+LFT 3-monthly; sodium (hyponatraemia); bone density long-term",
        "chrna2_specific": "Bedtime-heavy timing critical: 2/3 at bedtime aligns therapeutic peak with NREM window when GOF α2 most active. CBZ autoinduction: MUST re-check TDM at 6-8 weeks or levels will be subtherapeutic.",
        "sf_pct": 67,
    },
    {
        "drug": "Oxcarbazepine (OXC)",
        "evidence": "Level B",
        "role": "Alternative if CBZ intolerant / cognitive preference",
        "mechanism": "Na-channel block via active metabolite MHD (monohydroxy derivative); no autoinduction; better cognitive profile than CBZ",
        "dose": "OXC 450-900 mg/day; bedtime-heavy (same nocturnal strategy); TDM MHD target 12-35 µg/mL",
        "monitoring": "HLA-B*15:02 ABSOLUTE CI same as CBZ (cross-reactivity in SJS/TEN); sodium frequently (hyponatraemia more than CBZ); FBC; contraception interaction (OCP failure)",
        "chrna2_specific": "Prefer OXC over CBZ in patients with cognitive concerns or CYP3A4 polypharmacy complexity. HLA-B*15:02 CI applies equally — use LCM instead in HLA-positive.",
        "sf_pct": 62,
    },
    {
        "drug": "Lacosamide (LCM)",
        "evidence": "Level B",
        "role": "First-line if HLA-B*15:02 positive / CBZ-OXC intolerant / adjunct",
        "mechanism": "Slow inactivation of Na-channels (different from CBZ fast-inactivation); no HLA-B*15:02 SJS/TEN risk; no autoinduction",
        "dose": "100-400 mg/day; bedtime-heavy in ADNFLE; TDM 5-10 µg/mL",
        "monitoring": "PR interval prolongation (ECG before start; CI with Class I antiarrhythmics); dizziness; no routine TDM required but available; renal dose adjustment",
        "chrna2_specific": "KEY: LCM = first-line in HLA-B*15:02 positive patients where CBZ/OXC are ABSOLUTE CI (SJS/TEN). Also preferred adjunct when CBZ partial response.",
        "sf_pct": 55,
    },
    {
        "drug": "Clobazam (CLB)",
        "evidence": "Level B",
        "role": "Adjunct / bedtime rescue / DRE",
        "mechanism": "GABA-A positive allosteric modulator; α2 subunit preference; calming effect on nocturnal NREM excitability; anxiolytic component helps ADNFLE psychiatric comorbidity",
        "dose": "5-20 mg at bedtime; TDM N-desmethylclobazam (active metabolite) 0.3-3 µg/mL; titrate slowly",
        "monitoring": "Sedation; CLB+SSRI interaction: fluoxetine/fluvoxamine inhibit CYP2C19 → raise N-desmethylclobazam 5-fold → toxicity; prefer sertraline/escitalopram if co-prescribing SSRI for psychiatric comorbidity; tolerance over months",
        "chrna2_specific": "Bedtime-only CLB: add to CBZ-XR when partial control. Excellent for anxiety/mood comorbidity (concurrent benefit). Monitor CLB+SSRI CYP2C19 interaction carefully.",
        "sf_pct": 45,
    },
    {
        "drug": "Levetiracetam (LEV)",
        "evidence": "Level B",
        "role": "Adjunct — DRE / polypharmacy simplification",
        "mechanism": "SV2A vesicular fusion modulator; broad-spectrum; no Na-channel; no CYP interactions; safe in polypharmacy",
        "dose": "500-3000 mg/day BD; no TDM required (level check: 12-46 µg/mL if needed)",
        "monitoring": "Behavioural side effects (irritability, aggression — 'Keppra rage'): 15-20%; switch to brivaracetam if behavioural issues; psychiatric comorbidity monitoring (may worsen anxiety/depression in some)",
        "chrna2_specific": "Preferred adjunct for DRE in CHRNA2 given no CYP interactions (CBZ autoinduction already complex enough). If psychiatric comorbidity: monitor mood carefully; brivaracetam alternative.",
        "sf_pct": 40,
    },
    {
        "drug": "Topiramate (TPM)",
        "evidence": "Level C",
        "role": "Adjunct — DRE",
        "mechanism": "Multiple: Na-channel; AMPA/kainate block; carbonic anhydrase; GABA enhancement; broad-spectrum",
        "dose": "50-400 mg/day BD; slow titration (25 mg/day/week) to minimise cognitive SE",
        "monitoring": "Cognitive side effects: word-finding difficulty in 10-20% ('Dope-amax'); kidney stones (citrate 500 mg/day prophylaxis); weight loss; glaucoma",
        "chrna2_specific": "Use sparingly in CHRNA2: frontal lobe epilepsy + word-finding side effects = double burden. Reserve for DRE after CBZ+LCM+CLB tried. Slow titration essential.",
        "sf_pct": 35,
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "evidence": "Level B",
        "role": "DRE adjunct / children and adolescents",
        "mechanism": "Ketosis → GABA-B activation; BHB modulates NMDA/AMPA; adenosine elevation; metabolic shift reduces seizure threshold",
        "dose": "Classic 4:1 ratio; MCT variant; dietitian-led initiation; fasting initiation not required; target BHB 2-4 mmol/L",
        "monitoring": "BHB + glucose (ketone meter); lipid panel (LDL); kidney stones (citrate + hydration); growth monitoring in children; CBZ + KD: watch for hyponatraemia (combined risk)",
        "chrna2_specific": "Effective in CHRNA2 DRE: 40-60% responder rate (≥50% seizure reduction). Excellent option for children/adolescents with ADNFLE2 DRE before considering surgery.",
        "sf_pct": 30,
    },
    {
        "drug": "Frontal Lobe Resective Surgery / SEEG-guided",
        "evidence": "Level B",
        "role": "DRE refractory (2+ AEDs failed) — if localizable MRI or SEEG network",
        "mechanism": "Resection of seizure onset zone (SOZ) — frontal or SMA in ADNFLE; if MRI-negative: SEEG with habenular-frontal network mapping; laser interstitial thermal therapy (LITT) emerging for focal targets",
        "dose": "Pre-surgical evaluation: MRI 3T FCD protocol + SEEG or subdural grid + neuropsychology + fMRI language/motor mapping; 40-50% Engel I in ADNFLE MRI-positive",
        "monitoring": "Neuropsychological testing (frontal executive function) pre+post; speech/language mapping if dominant hemisphere; SEEG risk: <1% intracranial haemorrhage; VPSS seizure surveillance post-op",
        "chrna2_specific": "MRI typically NEGATIVE in genetic ADNFLE — makes localisation harder. If SEEG confirms focal network (even with negative MRI), limited frontal resection or LITT may achieve 35-45% Engel I. Refer after 2 AED failures per DRE criteria.",
        "sf_pct": 38,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRAINDICATIONS — 6 CIs
# ─────────────────────────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "CBZ/OXC in HLA-B*15:02 carriers (SE Asian)",
        "level": "ABSOLUTE CI",
        "reason": "SJS (Stevens-Johnson Syndrome) / TEN (Toxic Epidermal Necrolysis): HLA-B*15:02 allele predicts life-threatening cutaneous reactions within first 8 weeks of CBZ/OXC. Incidence ~5-10% in HLA-B*15:02 positive. Fatal in 30-40%. CPIC Level A guideline: test before prescribing; if positive → LCM first-line.",
        "alternative": "Lacosamide (LCM) — same Na-channel mechanism; no HLA-B*15:02 risk; ECG pre-start",
    },
    {
        "drug": "Tiagabine (TGB)",
        "level": "ABSOLUTE CI",
        "reason": "NCSE (Non-Convulsive Status Epilepticus): TGB inhibits GABA reuptake → GABA accumulation → paradoxical NCSE in focal epilepsy with cortical hyperexcitability. ADNFLE frontal cortex particularly vulnerable. Multiple case reports of TGB-induced NCSE in frontal lobe epilepsy.",
        "alternative": "Clobazam (GABA-A PAM) or LEV (SV2A) — safe GABAergic alternatives",
    },
    {
        "drug": "Bupropion (Wellbutrin/Zyban)",
        "level": "ABSOLUTE CI",
        "reason": "Bupropion lowers seizure threshold (dose-dependent) by dopamine/norepinephrine reuptake inhibition → increased cortical excitability. CHRNA2 habenular-frontal network already hyperexcitable. Prescribed for depression AND as smoking cessation aid — BOTH uses are ABSOLUTELY CONTRAINDICATED. Use SSRI (sertraline/escitalopram) for depression; do NOT use for smoking cessation.",
        "alternative": "Sertraline (SSRI) for depression — preferred; escitalopram alternative; AVOID bupropion for smoking cessation; use NRT 7-mg patch maximum (NOT varenicline, NOT bupropion)",
    },
    {
        "drug": "Varenicline (Champix/Chantix)",
        "level": "HIGH RISK",
        "reason": "Varenicline is a partial agonist at α4β2 nAChR (the primary target is the (α4)₂(β2)₃ receptor). In CHRNA2, the GOF α2β4 receptor may also be variably activated by varenicline (α2 subunit shares ligand-binding homology). Habenulo-interpeduncular axis (highest α2 expression) = key nicotine reward pathway — varenicline acts directly here. Risk of seizure exacerbation. Plus: varenicline neuropsychiatric black-box warning.",
        "alternative": "NRT (nicotine patch ≤7 mg; NOT >7 mg); gradual taper under neurology co-management; NOT abrupt cessation (receptor upregulation risk)",
    },
    {
        "drug": "Phenytoin (PHT) maintenance",
        "level": "HIGH RISK",
        "reason": "PHT has unfavourable pharmacokinetics (non-linear saturation kinetics → narrow therapeutic window), teratogenicity, cerebellar toxicity, cosmetic side effects (gingival hyperplasia, coarsening), zero-order kinetics making dose adjustment hazardous. For frontal lobe epilepsy in children/adolescents/women of reproductive age: avoid as maintenance (use as IV fosphenytoin in emergency only).",
        "alternative": "CBZ-XR or OXC (similar Na-channel mechanism, better pharmacokinetics); LCM (if HLA-B*15:02 positive)",
    },
    {
        "drug": "High-dose Nicotine (patches >7 mg / multiple patches)",
        "level": "HIGH RISK",
        "reason": "High-dose nicotine directly ACTIVATES GOF α2β4 nAChR → same seizure mechanism as endogenous ACh surge. Low-dose SUSTAINED nicotine (≤7-mg patch) achieves receptor desensitisation (investigational therapeutic effect); high-dose causes receptor opening and seizure precipitation. CHRNA2 habenular α2 pathway is the primary nicotine reward substrate — pharmacological relevance is direct.",
        "alternative": "NRT maximum 7-mg patch; gradually taper; do not combine patches",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING ITEMS — 14 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
MONITORING_ITEMS = [
    {"item": "HLA-B*15:02 genotyping", "frequency": "Once before CBZ/OXC (mandatory in SE Asian/South Asian/Han Chinese)", "rationale": "CPIC Level A: identifies 100% preventable fatal SJS/TEN"},
    {"item": "CBZ TDM (target 8-12 µg/mL)", "frequency": "Baseline → 6-8 weeks (autoinduction) → stable q6mo", "rationale": "CYP3A4 autoinduction drops CBZ level 30-50% at 6-8 weeks; MUST recheck or subtherapeutic"},
    {"item": "MHD TDM if OXC (target 12-35 µg/mL)", "frequency": "2 weeks post-initiation → q6mo stable", "rationale": "OXC dose → MHD active metabolite; TDM guides dose; no autoinduction with OXC"},
    {"item": "VPSG (Video-Polysomnography)", "frequency": "Diagnostic: 2 nights minimum; follow-up at 12 months or breakthrough seizures", "rationale": "Gold standard for ADNFLE diagnosis: captures NREM onset, confirms ictal vs parasomnia, quantifies seizure burden"},
    {"item": "FBC + LFT + sodium", "frequency": "Baseline → q3 months (CBZ/OXC) → q6 months stable", "rationale": "CBZ: aplastic anaemia risk (rare); hepatotoxicity; hyponatraemia (OXC > CBZ); routine safety panel"},
    {"item": "Neuropsychological assessment", "frequency": "Baseline + every 2 years + if academic/occupational concerns", "rationale": "ADNFLE predominantly nocturnal → daytime function preserved; baseline establishes cognitive profile; monitor if TPM added (cognitive SE)"},
    {"item": "VPSS seizure diary + frequency log", "frequency": "Ongoing (nightly log); review each clinic visit", "rationale": "Patient/parent log all nocturnal arousals, major events; VPSS supplements subjective diary with objective recording"},
    {"item": "CHRNA2 gene panel (family members)", "frequency": "Once — cascade genetic testing in first-degree relatives if proband confirmed", "rationale": "AD inheritance: 50% risk in offspring/siblings; identify at-risk relatives; enable pre-symptomatic monitoring and counselling"},
    {"item": "MRI brain 3T (FCD protocol)", "frequency": "Once at diagnosis; repeat if breakthrough seizures or clinical change", "rationale": "Rule out structural FCD phenocopy; MRI typically negative in genetic ADNFLE; 3T FCD protocol maximal sensitivity"},
    {"item": "SUDEP risk stratification (annual)", "frequency": "Annual", "rationale": "SUDEP risk elevated in DRE; ADNFLE2 nocturnal seizures = unwitnessed risk; bedroom safety plan; safety monitoring for DRE patients"},
    {"item": "Nicotine/smoking cessation support", "frequency": "At every clinic visit if smoker", "rationale": "CHRNA2 habenular pathway = primary nicotine reward circuit; abrupt cessation → seizure cluster; managed NRT under neurology co-supervision"},
    {"item": "VPPP (Valproate Pregnancy Prevention Programme) — if VPA used", "frequency": "Monthly (MHRA mandatory if VPA prescribed to women of reproductive age)", "rationale": "VPA teratogenicity: 10% major congenital malformations, 30-40% neurodevelopmental; MHRA VPPP mandatory"},
    {"item": "Bone density (DEXA)", "frequency": "Every 3 years in CBZ/OXC long-term users", "rationale": "Enzyme-inducing AEDs → vitamin D catabolism → reduced bone mineral density; supplement D3 1000 IU/day + calcium"},
    {"item": "Driving / safety counselling", "frequency": "At diagnosis and annually", "rationale": "DVLA/regional regulations: seizure-free interval required (typically 12 months nocturnal-only); nocturnal-only ADNFLE may qualify for restricted driving with documentation"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE STAGES — 6 STAGES
# ─────────────────────────────────────────────────────────────────────────────
LIFECYCLE_STAGES = [
    {
        "stage": "Childhood Onset (5-10 years)",
        "key_issues": ["Nocturnal hypermotor events often initially attributed to nightmares or NREM parasomnia", "Parent education: ADNFLE vs parasomnia distinction", "VPSG often first diagnostic test (not routine EEG)", "CBZ-XR start at lowest paediatric dose; HLA-B*15:02 before CBZ", "School performance monitoring: nocturnal seizures → daytime somnolence"],
        "treatment_focus": "CBZ-XR initiation; VPSG diagnosis; safety measures (bed rails, padded floor)",
    },
    {
        "stage": "Adolescence (11-18 years)",
        "key_issues": ["Adherence challenges (teen denial/embarrassment)", "Sleep deprivation (school schedule, social)", "Nicotine exposure (habenular α2 = primary nicotine reward circuit — teens particularly vulnerable)", "Social impact: sleepover avoidance, peer education", "Driving restrictions: nocturnal-only ADNFLE rules vary by jurisdiction"],
        "treatment_focus": "Adherence support; sleep hygiene strict schedule; nicotine counselling early; academic accommodation for daytime fatigue",
    },
    {
        "stage": "Young Adulthood (18-30 years)",
        "key_issues": ["Shift work: circadian disruption → seizure cluster", "Alcohol: evening intake → NREM rebound", "Reproduction: VPA ABSOLUTE CI for pregnancy planning; VPPP mandatory; CBZ teratogenicity (category D) counselled", "Nicotine cessation: managed NRT (NOT bupropion NOT varenicline)", "ADNFLE diagnosis often finally confirmed after years of parasomnia misdiagnosis"],
        "treatment_focus": "Contraception counselling (OCP + CBZ/OXC failure); pregnancy planning; occupational adjustment for shift work; nicotine cessation management",
    },
    {
        "stage": "Family Planning and Pregnancy",
        "key_issues": ["CBZ: teratogenicity category D (neural tube defects less than VPA; cardiac defects reported)", "OXC: similar teratogenicity profile to CBZ; monitor sodium", "LCM: limited pregnancy data; use if essential with monitoring", "Folic acid 5 mg/day 3 months pre-conception + first trimester mandatory", "Neonatal monitoring: CBZ-exposed neonates may show withdrawal/sedation"],
        "treatment_focus": "LCM preferred if seizure-free at conception on monotherapy; low-dose CBZ-XR + 5mg folic acid if essential; VPSS home monitoring in third trimester",
    },
    {
        "stage": "Adulthood Stable Phase (30-60 years)",
        "key_issues": ["Many patients achieve long remission: ADNFLE2 often improves with age", "AED withdrawal consideration: 2+ years seizure-free; VERY gradual taper; VPSG monitoring during taper", "Bone density: DEXA every 3 years (enzyme-inducing AED chronic use)", "Psychiatric comorbidity: 20% anxiety/depression — screen and treat; avoid bupropion", "Cascade genetic testing: test adult children/siblings for CHRNA2 pathogenic variant"],
        "treatment_focus": "Consider AED withdrawal trial after 2+ years seizure-free (very gradual taper over 12-24 months); VPSG monitoring; bone health",
    },
    {
        "stage": "DRE Management (any age — 25-30%)",
        "key_issues": ["25-30% ADNFLE2 drug-resistant: define after 2 adequate AED trials", "SEEG habenular-frontal network mapping in specialised centre", "KD adjunct: effective in children/adolescents (40-60% responder)", "Frontal resection/LITT: 35-45% Engel I if network localizable", "VPSS home monitoring critical for DRE seizure quantification"],
        "treatment_focus": "DRE pathway: add LCM then CLB then TPM or KD; refer tertiary centre after 2 AED failures; SEEG evaluation; surgical candidate assessment",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS — 12 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "CBZ TDM target", "value": "8-12 µg/mL (33-50 µmol/L)", "rationale": "Therapeutic range for seizure control; re-check at 6-8 weeks for autoinduction"},
    {"threshold": "CBZ re-TDM after autoinduction", "value": "6-8 weeks", "rationale": "CYP3A4 autoinduction peaks at 6-8 weeks; levels fall 30-50%; retitrate"},
    {"threshold": "OXC MHD TDM target", "value": "12-35 µg/mL", "rationale": "Active metabolite MHD range; no autoinduction with OXC"},
    {"threshold": "LCM TDM (if checked)", "value": "5-10 µg/mL", "rationale": "Reference range; no routine monitoring needed; use if efficacy/toxicity question"},
    {"threshold": "CLB N-desmethylclobazam", "value": "0.3-3 µg/mL", "rationale": "Active metabolite; elevated in CYP2C19 poor metabolisers or fluoxetine/fluvoxamine co-prescription"},
    {"threshold": "DRE definition", "value": "2 adequate AED trials failed (ILAE)", "rationale": "Triggers tertiary centre referral for SEEG + surgical evaluation"},
    {"threshold": "Seizure-free before driving", "value": "Regional variable: typically 12 months (DVLA UK); consult local authority", "rationale": "Nocturnal-only ADNFLE may qualify for restricted driving licence with documentation"},
    {"threshold": "Folic acid pre-conception", "value": "5 mg/day starting 3 months before conception", "rationale": "Enzyme-inducing AEDs increase folate catabolism; 5 mg (not 400 µg) required"},
    {"threshold": "VPSG nights for diagnosis", "value": "Minimum 2 nights", "rationale": "Night-to-night variability in ADNFLE event frequency; 2 nights captures >85% if 1+ events/week"},
    {"threshold": "NRT patch maximum dose", "value": "7 mg/day nicotine patch", "rationale": "Low-dose sustained → desensitisation (potentially beneficial); >7 mg → activation of GOF α2β4 → HIGH RISK"},
    {"threshold": "Bone DEXA interval", "value": "Every 3 years on enzyme-inducing AEDs", "rationale": "CBZ/OXC induce vitamin D catabolism → osteopenia risk; monitor with DEXA; supplement D3 1000 IU/day"},
    {"threshold": "ADNFLE2 seizure-free rate on CBZ-XR", "value": "65-70% (CBZ-XR monotherapy)", "rationale": "Based on CHRNA4/CHRNB2 data extrapolated to CHRNA2; limited CHRNA2-specific data given rarity"},
]

# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE STANDARDS — 12 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
EVIDENCE_STANDARDS = [
    {"standard": "ILAE-2022", "relevance": "Genetic epilepsy classification; ADNFLE2 recognised genetic focal epilepsy"},
    {"standard": "NICE-NG217-2022", "relevance": "Epilepsy management in adults/children: first-line and adjunctive therapy recommendations"},
    {"standard": "Aridon-2006-AJHG", "relevance": "First CHRNA2 pathogenic variant (I279N); founding paper for ADNFLE2"},
    {"standard": "Conti-2015-Epilepsia", "relevance": "Second CHRNA2 variant (I304N); clinical characterisation"},
    {"standard": "CPIC-HLA-B1502-CBZ-2023", "relevance": "Level A: HLA-B*15:02 mandatory testing before CBZ/OXC in SE Asian; absolute CI if positive"},
    {"standard": "ILAE-Genetic-Epilepsy-TaskForce-2018", "relevance": "Classification of CHRNA2/CHRNA4/CHRNB2 as genetic epilepsy with focal seizures"},
    {"standard": "AASM-ICSD3-2023", "relevance": "International Classification of Sleep Disorders; NFLE classification; VPSG diagnostic standards"},
    {"standard": "Tinuper-2016-Neurology", "relevance": "ADNFLE definition and diagnostic criteria; VPSG role; CBZ response data across ADNFLE genes"},
    {"standard": "MHRA-VPPP-2021", "relevance": "Valproate Pregnancy Prevention Programme: mandatory if VPA prescribed to women of reproductive age"},
    {"standard": "NICE-NG224-2023", "relevance": "Epilepsy surgery referral: criteria for tertiary evaluation after 2 AED failures"},
    {"standard": "ACMG-AMP-2015", "relevance": "Variant classification: pathogenic/likely pathogenic criteria for CHRNA2 rare variants"},
    {"standard": "WHO-ICF-2019", "relevance": "International Classification of Functioning; nocturnal seizure impact on quality of life, driving, employment"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES — 6 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
REFERENCES = [
    {"ref": "Aridon-2006-AJHG", "citation": "Aridon P, et al. (2006). Increased sensitivity of the neuronal nicotinic receptor α2 subunit causes familial epilepsy with nocturnal wandering and ictal fear. Am J Hum Genet, 79(2), 342-350. doi:10.1086/506459. [FIRST CHRNA2 I279N variant]"},
    {"ref": "Conti-2015-Epilepsia", "citation": "Conti V, et al. (2015). Nocturnal frontal lobe epilepsy with paroxysmal arousals due to CHRNA2 loss of function. Epilepsia, 56(8), e112-e117. doi:10.1111/epi.13066. [CHRNA2 I304N]"},
    {"ref": "Tinuper-2016-Neurology", "citation": "Tinuper P, et al. (2016). Definition and diagnostic criteria of Sleep-Related Hypermotor Epilepsy. Neurology, 86(19), 1834-1842. doi:10.1212/WNL.0000000000002666. [ADNFLE renamed SHE; diagnostic criteria; CBZ response]"},
    {"ref": "Steinlein-2012-EpilepsyRes", "citation": "Steinlein OK. (2012). Nicotinic receptor mutations in human epilepsy. Prog Brain Res, 196, 225-259. doi:10.1016/B978-0-444-59426-6.00012-0. [nAChR genetics review; CHRNA4/CHRNB2/CHRNA2 comparative]"},
    {"ref": "Scheffer-2014-Epilepsia", "citation": "Scheffer IE, et al. (2014). ILAE classification of the epilepsies: Position paper of the ILAE Commission for Classification and Terminology. Epilepsia, 58(4), 512-521. [Genetic focal epilepsy; ADNFLE classification]"},
    {"ref": "Bhatt-2017-NEJM", "citation": "Bhatt DL, et al. (2017). Precision Medicine in Epilepsy: Clinical and Research Perspectives. N Engl J Med, 376, 1775-1785. [Genetic epilepsy pharmacogenomics; precision treatment framework]"},
]

# ─────────────────────────────────────────────────────────────────────────────
# KEY CONCEPTS — 15 ITEMS
# ─────────────────────────────────────────────────────────────────────────────
KEY_CONCEPTS = [
    {"concept": "CHRNA2-8p21.2", "definition": "CHRNA2 gene at 8p21.2 encodes the nAChR α2 subunit (502 amino acids; 4 TM helices; TM2 lines pore). Rarest ADNFLE gene — <10 families published worldwide. Highest α2 expression in habenulo-interpeduncular pathway."},
    {"concept": "ADNFLE2-610353", "definition": "Autosomal Dominant Nocturnal Frontal Lobe Epilepsy Type 2 (OMIM #610353). GOF CHRNA2 → nocturnal hypermotor NREM seizures clinically indistinguishable from ADNFLE1 (CHRNA4) and ADNFLE3 (CHRNB2). Rarest of the 3 ADNFLE genes."},
    {"concept": "GOF-Delayed-Desensitisation-Alpha2", "definition": "CHRNA2 GOF mechanism: I279N/I304N TM2/TM3 mutations impair nAChR α2β4 desensitisation. Wild-type: opens → rapid desensitisation. GOF: channels remain open during NREM cholinergic surges → frontal hyperexcitability → seizure."},
    {"concept": "Habenulo-Interpeduncular-Pathway", "definition": "Highest α2 nAChR expression: medial habenula → interpeduncular nucleus axis. Functions in reward, aversion, fear, and nicotine addiction. CHRNA2 GOF in this pathway may contribute to: (1) nocturnal frontal seizures via ascending cholinergic drive; (2) heightened nicotine dependence risk in CHRNA2 families; (3) rare interictal mood/aversion symptoms."},
    {"concept": "ADNFLE-Triad-Complete", "definition": "ADNFLE nAChR triad now complete: CHRNA4 (α4, 20q13.33, ADNFLE1, #600513) [most common, >100 families] + CHRNB2 (β2, 1q21.3, ADNFLE3, #605375) [intermediate, ~20-25 families, 40% psychiatric] + CHRNA2 (α2, 8p21.2, ADNFLE2, #610353) [rarest, <10 families, habenular]. All same syndrome: nocturnal hypermotor NREM frontal lobe epilepsy. Same first-line treatment (CBZ-XR) and HLA-B*15:02 absolute CI."},
    {"concept": "Nicotine-Paradox-Alpha2", "definition": "UNIQUE to nAChR epilepsies: LOW-DOSE SUSTAINED nicotine (≤7-mg patch) → receptor desensitisation → potentially beneficial (investigational); HIGH-DOSE nicotine → receptor activation → seizure precipitation. CHRNA2 habenular axis is PRIMARY nicotine addiction circuit — pharmacological relevance is direct. ABRUPT CESSATION → receptor upregulation → seizure cluster."},
    {"concept": "HLA-B1502-CBZ-SJS-TEN", "definition": "HLA-B*15:02 allele: ABSOLUTE CI for CBZ and OXC in carriers (predominantly SE Asian, South Asian, Han Chinese). SJS/TEN: life-threatening skin necrosis within 8 weeks of CBZ/OXC initiation. Incidence ~5-10% in HLA-B*15:02 positive. CPIC Level A mandatory testing. Substitute: LCM (no HLA-B*15:02 risk)."},
    {"concept": "CBZ-Autoinduction-CYP3A4", "definition": "CBZ is a potent inducer of its own metabolism (CYP3A4 autoinduction). At initiation: CBZ level rises normally. At 6-8 weeks: CYP3A4 fully induced → CBZ clearance increases → level falls 30-50% → SUBTHERAPEUTIC. MUST recheck TDM at 6-8 weeks and retitrate. This is the most common cause of breakthrough seizures in ADNFLE patients on CBZ."},
    {"concept": "VPSG-Gold-Standard-ADNFLE", "definition": "Video-Polysomnography (VPSG): simultaneously records video + full EEG montage + EMG + respiratory + SpO₂ during sleep. Gold standard for ADNFLE diagnosis: (1) confirms seizures emerge from NREM (not REM or wake); (2) captures ictal EEG pattern; (3) definitively distinguishes ADNFLE from parasomnia (NREM arousal disorder, sleepwalking, RBD). Minimum 2 nights for adequate capture probability."},
    {"concept": "ADNFLE2-Misdiagnosis-Parasomnia", "definition": "35% of ADNFLE2 patients initially misdiagnosed as NREM parasomnias (sleepwalking, sleep terrors) or REM behaviour disorder. Average diagnostic delay: 3-10 years. Key distinction: ADNFLE = stereotyped, multiple/night, responds to CBZ; parasomnias = variable, 1-2/month, no AED response. VPSG + CBZ trial diagnostic."},
    {"concept": "ADNFLE2-Penetrance-Incomplete", "definition": "CHRNA2 GOF variants show incomplete penetrance (~75-85% based on limited family data). Unaffected carriers exist. Genetic counselling: 50% inheritance risk per offspring of carrier; penetrance ~80%; unaffected sibling may still carry and transmit. Cascade testing critical in all first-degree relatives."},
    {"concept": "ADNFLE2-DRE-25-30-Percent", "definition": "25-30% of CHRNA2 ADNFLE2 patients become drug-resistant (DRE, ILAE definition: 2+ adequate AED trials failed). DRE pathway: SEEG habenular-frontal network mapping → frontal resection (35-45% Engel I if localizable) or KD adjunct (40-60% ≥50% reduction). Refer to tertiary epilepsy centre after 2 AED failures."},
    {"concept": "Bupropion-ABSOLUTE-CI-CHRNA2", "definition": "Bupropion (antidepressant AND smoking cessation) is ABSOLUTELY CONTRAINDICATED in CHRNA2 epilepsy. Bupropion lowers seizure threshold (dopamine/NE reuptake inhibition → cortical excitability). ADNFLE2 frontal hyperexcitability + habenular α2 involvement makes this double-hazardous. Use sertraline/escitalopram for depression. DO NOT use bupropion for nicotine cessation in CHRNA2."},
    {"concept": "Varenicline-HIGH-RISK-CHRNA2", "definition": "Varenicline (Champix/Chantix) is a partial agonist at α4β2 nAChR. In CHRNA2, the habenulo-interpeduncular α2β4 circuit is the primary nicotine reward substrate — varenicline acts at this site. Risk: partial activation of GOF α2 receptor → seizure exacerbation. Additionally: varenicline neuropsychiatric black box (depression, suicidality). DO NOT use for smoking cessation in CHRNA2. Use managed NRT (≤7-mg patch) instead."},
    {"concept": "Frontal-Resection-ADNFLE2-Surgery", "definition": "Surgery in ADNFLE2 DRE: typically MRI-negative (genetic cause, no structural lesion). SEEG essential for SOZ localisation. Networks: SMA (tonic/fencing posture) → lateral premotor → primary motor. Habenular-frontal network characterisation assists planning. Outcomes: 35-45% Engel I if focal network confirmed; inferior to FCD-driven surgery (MRI-positive). LITT emerging as minimally invasive option for focal targets."},
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
_FIRST = ["Aiden","Brooke","Caleb","Diana","Ethan","Fiona","Gabriel","Hannah","Isaac","Julia",
          "Kai","Lena","Marco","Nadia","Oscar","Priya","Quinn","Rosa","Soren","Tara",
          "Uma","Victor","Wren","Xander","Yuki","Zara","Anton","Bella","Ciro","Dara",
          "Emil","Freya","Gus","Hana","Ivo","Jana","Kira","Luca","Mia","Noel"]
_LAST  = ["Alderton","Barros","Conti","Dietrich","Erikson","Ferro","Grant","Hassan","Ito","Jensen",
          "Klein","Larsen","Moran","Nakamura","Okafor","Patel","Quinn","Reyes","Santos","Torres",
          "Urquhart","Vidal","Walsh","Xiu","Yamamoto","Zuberi","Adler","Baxter","Costa","Dumont",
          "Engel","Flores","Greve","Hopper","Ibarra","Johansson","Kaur","Levi","Mora","Nair"]


def _make_patients():
    patients = []
    etiology_classes = [e["category"] for e in ETIOLOGY_CATALOG]
    etiology_pcts = [e["pct"] for e in ETIOLOGY_CATALOG]

    for i, (fn, ln) in enumerate(zip(_FIRST, _LAST)):
        r = random.Random(i * 7 + 13)
        etiology = r.choices(etiology_classes, weights=etiology_pcts)[0]

        onset_age_y = r.randint(5, 20)
        current_age_y = onset_age_y + r.randint(1, 25)
        is_hla_positive = r.random() < 0.05  # 5% SE Asian in cohort
        cbz_tdm = round(r.uniform(7.0, 13.5), 1)
        sf = r.random() < 0.66  # ~66% seizure-free
        psychiatric = r.random() < 0.20  # 20% (lower than CHRNB2)
        smoker = r.random() < 0.22  # 22%

        patients.append({
            "id": f"CHRNA2-{i+1:03d}",
            "name": f"{fn} {ln}",
            "age": current_age_y,
            "sex": r.choice(["M", "F"]),
            "onset_age": onset_age_y,
            "etiology": etiology,
            "locus": "8p21.2",
            "gene": "CHRNA2",
            "variant": r.choice(["p.I279N", "p.I304N", "p.V337G", "p.Y252C", "Unknown-GOF"]),
            "primary_seizure_type": "Hypermotor-Nocturnal-NREM",
            "seizure_free_6mo": sf,
            "psychiatric_comorbidity": psychiatric,
            "hla_b1502_positive": is_hla_positive,
            "hla_b1502_tested": is_hla_positive or r.random() < 0.72,
            "cbz_tdm_ug_ml": cbz_tdm,
            "current_aed": r.choices(
                ["CBZ-XR", "OXC", "LCM", "CBZ-XR+CLB", "CBZ-XR+LEV", "OXC+LCM"],
                weights=[40, 20, 15, 12, 8, 5])[0],
            "dre": r.random() < 0.28,  # 28% DRE
            "nicotine_user": smoker,
            "vpsg_performed": r.random() < 0.88,
            "years_to_diagnosis": r.randint(1, 12),
            "misdiagnosed_parasomnia": r.random() < 0.35,
            "surgical_candidate": r.random() < 0.15,
        })
    return patients


PATIENTS = _make_patients()


# ─────────────────────────────────────────────────────────────────────────────
# API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_overview() -> dict:
    total = len(PATIENTS)
    sf_pts = [p for p in PATIENTS if p["seizure_free_6mo"]]
    psych_pts = [p for p in PATIENTS if p["psychiatric_comorbidity"]]
    hla_tested = [p for p in PATIENTS if p["hla_b1502_tested"]]
    hla_pos = [p for p in PATIENTS if p["hla_b1502_positive"]]
    dre_pts = [p for p in PATIENTS if p["dre"]]
    vpsg_pts = [p for p in PATIENTS if p["vpsg_performed"]]
    misdiag_pts = [p for p in PATIENTS if p["misdiagnosed_parasomnia"]]
    avg_cbz = round(sum(p["cbz_tdm_ug_ml"] for p in PATIENTS) / total, 1)

    return {
        "gene": "CHRNA2",
        "full_name": "Cholinergic Receptor Nicotinic Alpha 2 Subunit",
        "locus": "8p21.2",
        "protein": "nAChR α2 subunit — 502 aa; 4 TM helices; TM2 pore-lining; preferentially assembles with CHRNB4 as (α2)₂(β4)₃",
        "channel": "(α2)₂(β4)₃ neuronal nicotinic acetylcholine receptor — habenulo-interpeduncular and frontal cortex",
        "syndrome": "ADNFLE2 — Autosomal Dominant Nocturnal Frontal Lobe Epilepsy Type 2 (OMIM #610353)",
        "adnfle_triad_position": "3rd and RAREST of the ADNFLE nAChR triad: CHRNA4 (ADNFLE1, >100 families) > CHRNB2 (ADNFLE3, ~25 families) > CHRNA2 (ADNFLE2, <10 families)",
        "companion_genes": "CHRNA4 (α4 subunit, ADNFLE1, 20q13.33) and CHRNB2 (β2 subunit, ADNFLE3, 1q21.3)",
        "inheritance": "Autosomal dominant; penetrance ~75-85% (limited family data); de novo rate uncertain",
        "key_mutations": {
            "I279N": "First reported (Aridon 2006 AJHG); TM2 position 6'; Italian family; ~75-80% penetrance",
            "I304N": "Second reported (Conti 2015 Epilepsia); TM3; Italian family; pure ADNFLE2",
            "V337G": "TM3 distal; Turkish family; pure ADNFLE2; ~30% DRE",
            "Y252C": "ECD-TM1 linker; very rare; mild GOF",
        },
        "precision_pharmacology": "CBZ-XR first-line (bedtime-heavy 2/3 at bedtime); HLA-B*15:02 screen before CBZ/OXC (SE Asian absolute CI); LCM if HLA-B*15:02 positive; NO bupropion (lowers seizure threshold); NO varenicline (partial nAChR agonist — HIGH RISK); NRT max 7-mg patch only",
        "habenular_significance": "Habenulo-interpeduncular pathway: highest α2 expression in brain; reward/aversion/fear circuit; nicotine addiction substrate — CHRNA2 GOF here may explain elevated nicotine dependence and habenula-mediated mood symptoms in affected families",
        "hallmark_misdiagnosis": "NREM parasomnia / sleepwalking (35% initially misdiagnosed); average diagnostic delay 3-10 years",
        "omim_gene": "*118502",
        "omim_adnfle2": "#610353",
        "first_mutation": "Aridon P et al. 2006 Am J Hum Genet — I279N (c.836T>A) in Italian ADNFLE family",
        "cohort": {
            "total": total,
            "seizure_free_6mo": len(sf_pts),
            "psychiatric_comorbidity": len(psych_pts),
            "hla_b1502_tested": len(hla_tested),
            "hla_b1502_positive": len(hla_pos),
            "dre_patients": len(dre_pts),
            "vpsg_performed": len(vpsg_pts),
            "misdiagnosed_parasomnia": len(misdiag_pts),
            "avg_cbz_tdm_ug_ml": avg_cbz,
        },
        "key_contraindications": [
            "CBZ/OXC ABSOLUTE CI in HLA-B*15:02 positive (SJS/TEN — fatal) — CPIC Level A",
            "TGB ABSOLUTE CI (NCSE in frontal focal epilepsy)",
            "Bupropion ABSOLUTE CI (lowers seizure threshold; also used for depression AND smoking cessation — both uses CONTRAINDICATED)",
            "Varenicline HIGH RISK (partial α2β4 nAChR agonist — directly acts on GOF receptor in habenular circuit)",
            "High-dose nicotine (>7-mg patch) HIGH RISK — activates GOF α2 receptor",
            "PHT maintenance HIGH RISK (teratogenic, cerebellar toxicity, non-linear kinetics)",
        ],
        "etiologies": [{"class": c["category"], "pct": c["pct"]} for c in ETIOLOGY_CATALOG],
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
        "gene": "CHRNA2",
        "full_name": "Cholinergic Receptor Nicotinic Alpha 2 Subunit",
        "locus": "8p21.2",
        "omim_gene": "*118502",
        "omim_adnfle2": "#610353",
        "protein": "nAChR α2 subunit — 502 amino acids; 4 TM helices; TM2 lines cation pore; highest expression in habenulo-interpeduncular pathway",
        "channel_family": "Pentameric ligand-gated ion channels (pLGIC) — Cys-loop receptor superfamily",
        "syndrome": {
            "ADNFLE2": "Autosomal Dominant Nocturnal Frontal Lobe Epilepsy Type 2 — OMIM #610353",
            "ADNFLE_triad": "CHRNA4 (ADNFLE1) + CHRNB2 (ADNFLE3) + CHRNA2 (ADNFLE2) — complete nAChR epilepsy triad",
        },
        "concepts": KEY_CONCEPTS,
        "thresholds": THRESHOLDS,
        "evidence_standards": EVIDENCE_STANDARDS,
        "key_pharmacological_distinctions": [
            "CBZ-XR FIRST-LINE: bedtime-heavy (2/3 at bedtime) — aligns CBZ peak with NREM window when GOF α2 maximally active; 65-70% SF; autoinduction: re-check TDM at 6-8 weeks",
            "HLA-B*15:02 ABSOLUTE CI for CBZ and OXC (SJS/TEN fatal in first 8 weeks) — CPIC Level A; use LCM instead (same Na-channel mechanism, no HLA-B*15:02 risk; ECG pre-start for PR prolongation)",
            "BUPROPION ABSOLUTE CI: lowers seizure threshold — particularly dangerous in CHRNA2 given frontal hyperexcitability AND habenular circuit involvement; prescribed for both depression AND smoking cessation — both uses CONTRAINDICATED; use sertraline/escitalopram for depression",
            "VARENICLINE HIGH RISK: partial agonist at α4β2 nAChR; habenulo-interpeduncular pathway (highest α2 expression) = primary nicotine reward circuit = where varenicline acts most strongly; risk of activating GOF α2 receptor; use managed NRT (≤7-mg patch) instead",
            "NICOTINE PARADOX: low-dose sustained (≤7-mg patch) → desensitisation (investigational); HIGH-DOSE → activation of GOF α2 → seizure; abrupt cessation → receptor upregulation → seizure cluster; requires neurology-addiction medicine co-management",
            "CBZ AUTOINDUCTION: CYP3A4 fully induced at 6-8 weeks → CBZ level falls 30-50% → subtherapeutic → breakthrough seizures; MANDATORY re-check TDM at 6-8 weeks and retitrate; most common preventable cause of CBZ treatment failure",
            "HABENULAR AXIS CLINICAL IMPLICATIONS: medial habenula → interpeduncular nucleus (highest α2 density) → ascending cholinergic drive → frontal cortex; untreated CHRNA2 may show interictal mood/aversion symptoms (habenula dysfunction = depression circuit); nicotine dependence elevated (habenula = primary nicotine reward pathway)",
        ],
    }
