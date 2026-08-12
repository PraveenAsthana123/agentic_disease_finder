"""
Doose Syndrome (Myoclonic-Atonic Epilepsy / MAE) Dashboard
===========================================================
41-patient cohort · ILAE 2022 GGE classification · self-limited childhood epilepsy
Distinct from LGS: no tonic seizures, 50% remission rate, KD Level A first-line.
"""

import random
from datetime import datetime

SEED = 4242
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ──────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "Polygenic / GGE-spectrum",
        "n": 21, "pct": 51,
        "category": "Genetic-Complex",
        "mechanism": (
            "Polygenic susceptibility across multiple GGE-risk loci (GABRA1, GABRB3, GABRG2, "
            "HCN1, KCNQ3). No single-gene cause; strong family history of GGE in 30–40% of cases. "
            "Generalised 4–7 Hz theta rhythm on background EEG reflects widespread thalamocortical "
            "network instability."
        ),
        "eeg_correlate": (
            "2–3 Hz generalised spike-wave during myoclonic-atonic events; 4–7 Hz diffuse theta "
            "background (Doose theta rhythm); irregular polyspike-wave in series."
        ),
        "mri_finding": "Normal MRI (>90% of polygenic MAE)",
        "clinical_note": (
            "Family history of febrile seizures or GGE in 30–40%. Prognosis better than LGS: "
            "~50% achieve long-term seizure freedom."
        ),
    },
    {
        "etiology": "SCN1A (Dravet-overlap / hot-fever sensitivity)",
        "n": 5, "pct": 12,
        "category": "Monogenic",
        "mechanism": (
            "Heterozygous loss-of-function SCN1A variants cause Nav1.1 haploinsufficiency in "
            "inhibitory interneurons. Overlaps with Dravet spectrum when hot-bath/fever sensitivity "
            "is prominent. Myoclonic-atonic phenotype without prolonged febrile seizures at onset."
        ),
        "eeg_correlate": (
            "Generalised irregular spike-wave 2–3 Hz; photosensitivity less prominent than Dravet. "
            "Hypnagogic bursts on sleep EEG."
        ),
        "mri_finding": "Normal; occasional mild generalised atrophy in severe cases",
        "clinical_note": (
            "⚠️ Sodium-channel blockers (CBZ, OXC, PHT) are ABSOLUTELY CONTRAINDICATED — "
            "worsen myoclonic-atonic seizures via SCN1A mechanism."
        ),
    },
    {
        "etiology": "SLC6A1 (GAT-1 GABA transporter deficiency)",
        "n": 7, "pct": 17,
        "category": "Monogenic",
        "mechanism": (
            "Heterozygous loss-of-function SLC6A1 variants disrupt GABA reuptake transporter GAT-1, "
            "causing elevated synaptic GABA but paradoxically impaired phasic inhibition through "
            "desensitisation of GABA-A receptors. Second-most-common monogenic cause of MAE. "
            "Intellectual disability moderate in 60%."
        ),
        "eeg_correlate": (
            "High-amplitude 2–3 Hz spike-wave; prominent myoclonic-atonic EEG correlate; "
            "background slowing correlates with cognitive severity."
        ),
        "mri_finding": "Normal or mild cortical atrophy; corpus callosum thin in 20%",
        "clinical_note": (
            "Ketogenic diet shows excellent efficacy in SLC6A1-MAE (55% responder rate). "
            "VPA + KD combination used in severe refractory cases."
        ),
    },
    {
        "etiology": "SYNGAP1 / NEXMIF (Synaptic scaffold variants)",
        "n": 5, "pct": 12,
        "category": "Monogenic-Synaptic",
        "mechanism": (
            "SYNGAP1 haploinsufficiency disrupts Ras-GTPase signalling at excitatory synapses, "
            "causing hyperexcitability in dendritic spines. NEXMIF (KIAA2022) loss-of-function "
            "disrupts chromatin remodelling in hippocampal-cortical circuits. Both present with "
            "intellectual disability, autism features, and MAE-like seizure semiology."
        ),
        "eeg_correlate": (
            "2.5–4 Hz generalised spike-wave; prominent absence component; occasional "
            "electrodecrement pattern correlating with atonic drop attacks."
        ),
        "mri_finding": "Normal or mild non-specific signal changes; thin corpus callosum (NEXMIF)",
        "clinical_note": (
            "Distinguish from classic MAE: SYNGAP1/NEXMIF have prominent ASD/intellectual disability. "
            "Prognosis for seizure control intermediate; cognitive outcomes depend on variant severity."
        ),
    },
    {
        "etiology": "Unknown / Cryptogenic-structural",
        "n": 3, "pct": 7,
        "category": "Cryptogenic",
        "mechanism": (
            "MAE phenotype without identifiable genetic variant on current panel (GS/ES). Includes "
            "rare unresolved structural causes missed on 1.5T MRI (requires 3T + FCD protocol). "
            "Functional re-classification pending whole-genome sequencing."
        ),
        "eeg_correlate": "2–3 Hz spike-wave; background abnormalities variable",
        "mri_finding": "Normal on standard clinical MRI; 3T epilepsy-protocol pending",
        "clinical_note": (
            "Pursue whole-genome sequencing + advanced MRI (3T FCD protocol) before labelling "
            "as 'cryptogenic'. Up to 30% of MAE with negative standard gene panels have diagnostic "
            "variants on re-analysis."
        ),
    },
]

# ── Patient Generator ─────────────────────────────────────────────────────────
ETIOLOGIES_WEIGHTED = (
    ["Polygenic / GGE-spectrum"] * 21
    + ["SCN1A (Dravet-overlap / hot-fever sensitivity)"] * 5
    + ["SLC6A1 (GAT-1 GABA transporter deficiency)"] * 7
    + ["SYNGAP1 / NEXMIF (Synaptic scaffold variants)"] * 5
    + ["Unknown / Cryptogenic-structural"] * 3
)
AEDS = [
    "VPA", "LEV + VPA", "KD + VPA", "TPM", "CLB + VPA",
    "LEV", "VPA + CLB", "ZNS", "KD", "LEV + CLB",
]
SEIZURE_TYPES = [
    "Myoclonic-atonic (drop attacks)",
    "Myoclonic jerks",
    "Absence seizures",
    "GTCS",
]
CTRL = ["Seizure-free", "Partial control", "Drug-resistant"]
CTRL_W = [0.50, 0.30, 0.20]
PHASES = ["Active", "Near-remission", "Remission"]
PHASE_W = [0.45, 0.25, 0.30]

random.seed(SEED)


def _weighted_choice(choices, weights):
    cum = 0.0
    r = random.random()
    for c, w in zip(choices, weights):
        cum += w
        if r < cum:
            return c
    return choices[-1]


def _make_patients():
    pts = []
    for i in range(1, 42):
        ctrl = _weighted_choice(CTRL, CTRL_W)
        phase = _weighted_choice(PHASES, PHASE_W)
        et = random.choice(ETIOLOGIES_WEIGHTED)
        onset = random.randint(1, 5)
        age = random.randint(onset + 2, onset + 10)
        aed = random.choice(AEDS)
        kd = "KD" in aed
        drop_attacks = ctrl != "Seizure-free" or random.random() > 0.7
        pts.append({
            "patient_id": f"MAE-{i:03d}",
            "age": age,
            "gender": "M" if random.random() < 0.63 else "F",
            "onset_age": onset,
            "etiology": et,
            "primary_seizure_type": random.choice(SEIZURE_TYPES),
            "current_aed": aed,
            "seizure_control": ctrl,
            "disease_phase": phase,
            "kd_on_ketogenic_diet": kd,
            "drop_attacks_active": drop_attacks and ctrl != "Seizure-free",
            "intellectual_disability": random.random() < 0.35,
            "autism_features": random.random() < 0.20,
        })
    return pts


PATIENTS = _make_patients()


# ── Seizure Types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES_DETAIL = [
    {
        "type": "Myoclonic-Atonic Seizure (Drop Attack)",
        "freq_pct": 100,
        "description": (
            "Pathognomonic for Doose Syndrome: a bilateral myoclonic jerk immediately followed by "
            "loss of postural tone causing sudden head-drop or full-body fall. Duration 1–3 seconds. "
            "Onset without warning. Can cluster >10/day in active phase. "
            "The atonic component distinguishes MAE from pure myoclonic epilepsies."
        ),
        "eeg_correlate": (
            "Generalised 2–3 Hz spike-wave burst: spike correlates with myoclonus, slow-wave with "
            "atonic component. Background 4–7 Hz theta (Doose rhythm) between events."
        ),
        "duration_sec": "1–3 s",
        "clinical_tip": (
            "Helmet mandatory for drop-attack phase. Assess fall injury risk at every visit. "
            "KD and VPA are preferred — avoid CBZ/OXC/PHT which WORSEN drop attacks."
        ),
    },
    {
        "type": "Myoclonic Seizure (without atonic component)",
        "freq_pct": 80,
        "description": (
            "Bilateral symmetric jerks of upper limbs ± trunk, occurring in isolation without the "
            "subsequent postural lapse. Most frequent in first 30–60 minutes after waking. "
            "Can be subtle (finger flick) or dramatic (arm fling causing object dropping)."
        ),
        "eeg_correlate": (
            "Brief generalised polyspike-wave burst 4–6 Hz, <3 seconds, time-locked to jerk. "
            "Inter-ictal: irregular polyspike-wave runs in NREM1-2 transitions."
        ),
        "duration_sec": "<3 s",
        "clinical_tip": (
            "Provoked by sleep deprivation (80%) and photic stimulation (30%). "
            "Morning seizure diary is key for monitoring. VPA reduces morning myoclonus in 75%."
        ),
    },
    {
        "type": "Tonic-Clonic Seizure (GTCS)",
        "freq_pct": 75,
        "description": (
            "Generalised tonic-clonic seizures occur in 75% of MAE patients, often in the "
            "setting of sleep deprivation or intercurrent illness. Duration 1–3 minutes. "
            "Distinguishing feature from LGS: tonic seizures alone are ABSENT in MAE."
        ),
        "eeg_correlate": (
            "High-amplitude rhythmic 10-Hz fast activity during tonic phase → diffuse polyspike "
            "in clonic phase → post-ictal suppression. Background recovery faster than in LGS."
        ),
        "duration_sec": "60–180 s",
        "clinical_tip": (
            "Rescue midazolam 0.3 mg/kg buccal for seizures >5 minutes. "
            "ABSENCE of isolated tonic seizures is a key differential from Lennox-Gastaut Syndrome."
        ),
    },
    {
        "type": "Absence Seizure",
        "freq_pct": 40,
        "description": (
            "Brief 5–30 second episodes of behavioural arrest with or without subtle eyelid "
            "flickering. Hyperventilation may provoke. The absence in MAE has a shorter 2–3 Hz "
            "spike-wave pattern vs the classic 3 Hz CAE absence. "
            "40% of MAE patients experience absences, usually alongside other seizure types."
        ),
        "eeg_correlate": (
            "2–3 Hz spike-wave (slightly slower than CAE's 3 Hz); background abnormalities "
            "between events. Irregular morphology unlike CAE's regular 3 Hz."
        ),
        "duration_sec": "5–30 s",
        "clinical_tip": (
            "Hyperventilation provocation in clinic aids diagnosis. "
            "Ethosuximide may help absence component, but VPA preferred as it covers all MAE types."
        ),
    },
]

# ── Triggers ──────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Sleep deprivation", "pct": 85, "mechanism": "Reduced inhibitory reserve; theta-burst facilitation", "management": "Strict sleep schedule; rescue CLB at night before anticipated deprivation"},
    {"trigger": "Fever / intercurrent illness", "pct": 70, "mechanism": "Hyperthermia lowers seizure threshold (especially SCN1A-MAE); cytokine effects on GABA-A", "management": "Antipyretics early; rescue dose protocol; seizure action plan for parents"},
    {"trigger": "Missed AED dose", "pct": 65, "mechanism": "Sub-therapeutic VPA/LEV serum levels → rebound hyperexcitability", "management": "Pill organiser; carer training; emergency supply at school"},
    {"trigger": "Stress / excitement", "pct": 55, "mechanism": "Sympathetic arousal lowers cortical inhibition; catecholamine spike", "management": "Relaxation strategies; predictive dose adjustment before stressful events"},
    {"trigger": "Fatigue", "pct": 50, "mechanism": "Impaired homeostatic sleep regulation; reduced inhibitory interneuron firing", "management": "Scheduled rest periods; occupational therapy pacing plan"},
    {"trigger": "Photic stimulation (flicker)", "pct": 30, "mechanism": "Visual cortex hyperexcitability; occipito-parietal GSW propagation on EEG", "management": "Polarised lenses; screen distance rules; avoid strobe environments"},
    {"trigger": "Hypoglycaemia / missed meals", "pct": 25, "mechanism": "Low glucose impairs Na⁺/K⁺-ATPase → reduced inhibitory tone; worsened on KD if carb 'slipping'", "management": "Regular meals on KD; glucose strip monitoring; 4:1 ratio supervision"},
    {"trigger": "Puberty / hormonal shifts", "pct": 15, "mechanism": "Oestrogenic surge increases cortical excitability; progesterone withdrawal phases", "management": "Hormonal monitoring; consider GnRH agonist in severe catamenial overlap"},
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproate (VPA)",
        "evidence": "Level A",
        "evidence_ref": "ILAE 2022; Oguni 2005 Epileptic Disord (broad-spectrum first-line for MAE)",
        "dose_adult": "20–40 mg/kg/day in 2–3 divided doses (TDM target 50–100 µg/mL)",
        "dose_paed": "20–40 mg/kg/day; start 10 mg/kg/day → titrate over 2 weeks",
        "moa": (
            "Inhibits voltage-gated Na⁺ channels; enhances GABA synthesis via GABA-T inhibition; "
            "blocks T-type Ca²⁺ channels (absence component). Broad-spectrum coverage of all MAE "
            "seizure types: myoclonic, atonic, GTCS, absence."
        ),
        "efficacy": "50–60% responder rate in MAE; 40% seizure-free at 2 years (Oguni 2005)",
        "safety": "Weight gain, hair loss, tremor, teratogenicity (FDA REMS ≥4y), hepatotoxicity (rare, <2y age highest risk)",
        "monitoring": "TDM 50–100 µg/mL; LFT + ammonia at baseline, 3M, then annually; weight + BMI; VPA REMS for females ≥4y; EXCLUDE POLG/mitochondrial disease before starting",
        "contraindication_note": None,
    },
    {
        "drug": "Ketogenic Diet (KD) — 4:1 classic or MCT",
        "evidence": "Level A (MAE-specific)",
        "evidence_ref": "Caraballo 2013 Seizure; Lemmon 2012 Epilepsia; ILAE KD guideline 2018",
        "dose_adult": "4:1 fat:non-fat ratio; MCT protocol 40–60% MCT oil for better palatability",
        "dose_paed": "Classical 4:1 or modified Atkins; supervised by KD-specialist dietitian; glucose <75 mg/dL target",
        "moa": (
            "Beta-hydroxybutyrate (BHB) activates ATP-sensitive K⁺ channels (KATP), reducing neuronal "
            "excitability. BHB also inhibits NLRP3 inflammasome and enhances mitochondrial biogenesis. "
            "KD normalises the 4–7 Hz Doose theta rhythm in 55–70% of responders."
        ),
        "efficacy": "50–60% ≥50% seizure reduction in MAE; 30% seizure-free (highest KD efficacy of any GGE syndrome)",
        "safety": "Hyperlipidaemia, kidney stones (2%), growth delay, acidosis; renal monitoring quarterly",
        "monitoring": "Urine ketones daily (target 3–4+); serum BHB weekly initially; lipid panel + renal function q3M; DEXA annually if >2y on KD; dietitian review every 4–6 weeks",
        "contraindication_note": None,
    },
    {
        "drug": "Levetiracetam (LEV)",
        "evidence": "Level B",
        "evidence_ref": "Coppola 2002 Epilepsy Res; Veggiotti 2010 Epileptic Disord",
        "dose_adult": "20–40 mg/kg/day in 2 divided doses (max 3000 mg/day)",
        "dose_paed": "20–40 mg/kg/day; start 10 mg/kg/day; oral solution available",
        "moa": (
            "Binds SV2A synaptic vesicle protein, inhibiting neurotransmitter exocytosis. "
            "Reduces high-frequency network bursting. No sodium-channel blockade — safe in MAE."
        ),
        "efficacy": "35–45% ≥50% seizure reduction; useful add-on when VPA alone insufficient",
        "safety": "Behavioural side-effects (irritability, aggression) in 15–20% — monitor with CBCL/PHQ-9; somnolence common early",
        "monitoring": "PHQ-9 + CBCL (Children's Behaviour Checklist) at baseline, 1M, 3M; renal function annually",
        "contraindication_note": None,
    },
    {
        "drug": "Clobazam (CLB)",
        "evidence": "Level B",
        "evidence_ref": "Ng 2011 J Child Neurol; ILAE 2022 adjunct recommendation",
        "dose_adult": "0.5–1.0 mg/kg/day (max 40 mg/day) in 2 divided doses",
        "dose_paed": "0.25–0.5 mg/kg/day; higher evening dose for nocturnal seizures",
        "moa": (
            "1,5-benzodiazepine; binds GABA-A receptor α2/α3 subunits with less sedation than "
            "1,4-benzodiazepines. Enhances chloride conductance → neuronal hyperpolarisation. "
            "Tolerance develops in 30% after 6 months."
        ),
        "efficacy": "Add-on: 40–55% responder rate; particularly effective for drop-attack clusters",
        "safety": "Tolerance (30%), sedation, drooling (children), dependence with abrupt withdrawal",
        "monitoring": "UMRS (Unified Myoclonus Rating Scale) for myoclonus; tolerance reassessment every 6M; taper slowly if discontinuing",
        "contraindication_note": None,
    },
    {
        "drug": "Topiramate (TPM)",
        "evidence": "Level B",
        "evidence_ref": "Guerrini 2005 Epilepsia; Coppola 2012 Seizure",
        "dose_adult": "2–10 mg/kg/day in 2 divided doses (start 0.5 mg/kg/day, titrate slowly)",
        "dose_paed": "2–8 mg/kg/day; slow titration minimises cognitive effects",
        "moa": (
            "Blocks voltage-gated Na⁺ channels, enhances GABA-A transmission, inhibits kainate/AMPA "
            "receptors, and inhibits carbonic anhydrase. Multiple mechanisms provide broad spectrum."
        ),
        "efficacy": "30–40% ≥50% seizure reduction as add-on; effective for myoclonic-atonic component",
        "safety": "Word-finding difficulty (10–15%), weight loss, kidney stones (1.5%), glaucoma (rare), metabolic acidosis",
        "monitoring": "Neuropsychological assessment at baseline + 6M (word-fluency test); bicarbonate quarterly; ophthalmology if visual symptoms",
        "contraindication_note": None,
    },
    {
        "drug": "Zonisamide (ZNS)",
        "evidence": "Level B",
        "evidence_ref": "Oguni 2002 Epilepsia; Kothare 2015 Seizure",
        "dose_adult": "4–8 mg/kg/day in 2 divided doses (start 2 mg/kg/day)",
        "dose_paed": "4–8 mg/kg/day; sulphonamide allergy — screen before initiating",
        "moa": (
            "Blocks T-type Ca²⁺ channels and voltage-gated Na⁺ channels; inhibits carbonic "
            "anhydrase; partial GABA enhancement. Particularly useful when MAE-absence component "
            "is prominent (T-type blockade)."
        ),
        "efficacy": "30–40% responder rate in refractory MAE add-on; well-studied in Japanese cohorts",
        "safety": "Anorexia/weight loss, kidney stones (3%), oligohidrosis/hyperthermia (children), sedation",
        "monitoring": "Weight monthly (children); renal function + bicarbonate q6M; body temperature monitoring in summer",
        "contraindication_note": None,
    },
    {
        "drug": "Ethosuximide (ETX)",
        "evidence": "Level C (absence component only)",
        "evidence_ref": "Glauser 2010 NEJM CHILDHOOD (CAE evidence extrapolated); Oguni 2005 clinical consensus",
        "dose_adult": "20–40 mg/kg/day in 2 divided doses (TDM target 40–100 µg/mL)",
        "dose_paed": "20–30 mg/kg/day; liquid formulation available; GI side-effects minimised with food",
        "moa": (
            "Selectively blocks thalamic T-type (Cav3.2) Ca²⁺ channels → reduces thalamic burst "
            "pacing that drives 2–3 Hz spike-wave absence discharges. No effect on myoclonic or "
            "atonic components — must be combined with VPA if myoclonus is prominent."
        ),
        "efficacy": "Effective only for absence component; NOT adequate monotherapy for full MAE (misses myoclonic/atonic seizures)",
        "safety": "GI upset, hiccups, headache, rare SJS/agranulocytosis",
        "monitoring": "TDM 40–100 µg/mL; CBC if rash/fever; assess myoclonic-atonic control separately",
        "contraindication_note": "⚠️ Not monotherapy — must combine with VPA or KD to cover myoclonic-atonic component",
    },
    {
        "drug": "Cannabidiol (CBD / Epidiolex)",
        "evidence": "Level C (off-label MAE)",
        "evidence_ref": "Miller 2020 Epilepsy Behav (retrospective MAE series); FDA approval for DS/LGS extrapolated",
        "dose_adult": "5–20 mg/kg/day in 2 divided doses; titrate over 4–8 weeks",
        "dose_paed": "5–20 mg/kg/day; start 2.5 mg/kg/day; escalate slowly",
        "moa": (
            "Modulates TRPV1 channels, inhibits adenosine reuptake (GABAergic indirect effect), "
            "blocks GPR55 receptor. Reduces intracellular Ca²⁺ via mitochondrial pathway. "
            "Not a CB1/CB2 receptor agonist."
        ),
        "efficacy": "Limited MAE-specific data; 30–40% responder rate in refractory cases with drop attacks",
        "safety": "Somnolence, diarrhoea, ↑LFTs (especially with VPA co-administration); LFT monitoring mandatory",
        "monitoring": "LFT at baseline, 1M, 3M, then q6M (critical when combined with VPA — transaminase elevation 3× ULN in 20%); CBC; AED level monitoring (CBD inhibits CYP2C19 → ↑CLB levels)",
        "contraindication_note": None,
    },
]

# ── Absolute Contraindications ────────────────────────────────────────────────
ABSOLUTE_CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ)",
        "contraindicated_in": "ALL Doose Syndrome / MAE",
        "consequence": (
            "Sodium-channel blockade aggravates myoclonic-atonic seizures and can precipitate "
            "myoclonic status epilepticus. Blocks Na⁺ channels relevant to inhibitory interneurons. "
            "Documented worsening in >80% of MAE patients given CBZ."
        ),
    },
    {
        "drug": "Oxcarbazepine (OXC)",
        "contraindicated_in": "ALL Doose Syndrome / MAE",
        "consequence": (
            "Same mechanism as CBZ — voltage-gated Na⁺ channel blockade. Worsens drop-attack "
            "frequency. Particularly dangerous in SCN1A-MAE subtype."
        ),
    },
    {
        "drug": "Phenytoin (PHT)",
        "contraindicated_in": "ALL Doose Syndrome / MAE",
        "consequence": (
            "Sodium-channel blockade aggravates myoclonic jerks and atonic drop attacks. "
            "Phenytoin has narrow therapeutic index and significant cognitive side-effects in "
            "children. Absolutely avoided."
        ),
    },
    {
        "drug": "Vigabatrin (VGB)",
        "contraindicated_in": "ALL Doose Syndrome / MAE",
        "consequence": (
            "Irreversible GABA-transaminase inhibition causes paradoxical aggravation of "
            "myoclonic seizures in GGE syndromes including MAE. Also causes irreversible "
            "visual field constriction requiring perimetry monitoring — unacceptable risk-benefit."
        ),
    },
    {
        "drug": "Lamotrigine (LTG) — RELATIVE caution",
        "contraindicated_in": "MAE with prominent myoclonic component",
        "consequence": (
            "Worsens myoclonic seizures in 15–20% of MAE patients (myoclonus aggravation "
            "syndrome). Safer if myoclonus is mild or if combined with VPA (which also halves "
            "LTG dose — titrate very slowly). Relative contraindication, not absolute."
        ),
    },
]

# ── AED Monitoring ────────────────────────────────────────────────────────────
AED_MONITORING = [
    {
        "item": "VPA — REMS + TDM + LFT + POLG exclusion",
        "frequency": "Before initiation + monthly ×3 → q3M → q6M",
        "rationale": (
            "FDA VPA REMS mandatory for females ≥4y (neural tube defect, PCOS). "
            "TDM target 50–100 µg/mL. LFT + ammonia: hepatotoxicity risk highest <2y — EXCLUDE "
            "mitochondrial disease (POLG mutation screening) before starting VPA."
        ),
    },
    {
        "item": "KD — Ketone monitoring + metabolic panel + renal",
        "frequency": "Urine ketones daily; metabolic panel q3M; DEXA annually if >2y",
        "rationale": (
            "Urine ketones target 3–4+ (blood BHB 2–4 mmol/L). Renal stones: urine Ca:Cr ratio "
            "q6M; alkalinisation if stone risk high. Lipid panel annually. Selenium/zinc/selenium "
            "supplementation mandatory on KD."
        ),
    },
    {
        "item": "LEV — Behavioural monitoring (CBCL + PHQ-9)",
        "frequency": "Baseline, 1M, 3M, then q6M",
        "rationale": (
            "Behavioural side-effects (irritability, aggression) in 15–20% of children — use "
            "CBCL (Achenbach) and PHQ-9 at each visit. Renal function annually. Dose adjust "
            "for eGFR <80 mL/min/1.73m²."
        ),
    },
    {
        "item": "CLB — Tolerance + sedation + UMRS",
        "frequency": "q3M assessment; UMRS at each visit",
        "rationale": (
            "Clobazam tolerance develops in 30% within 6 months — reassess efficacy quarterly. "
            "Drug-drug interaction: CBD raises CLB active metabolite N-desmethylclobazam 2–3× — "
            "reduce CLB dose when adding CBD. UMRS tracks myoclonus trajectory."
        ),
    },
]

# ── Lifecycle Management ──────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Toddler / Onset Phase",
        "age_range": "1–5 years",
        "key_events": "First drop attack; safety helmet; genetic workup initiated; VPA or KD started",
        "focus": (
            "Immediate safety: helmet + floor padding. Start VPA ± KD. Emergency rescue dose "
            "(midazolam buccal). Genetic panel: SCN1A, SLC6A1, SYNGAP1, NEXMIF, GGE panel."
        ),
    },
    {
        "window": "Active Drop-Attack Phase",
        "age_range": "2–7 years",
        "key_events": "Peak seizure frequency; KD assessment; school integration challenges",
        "focus": (
            "Optimise AED (VPA + KD ± LEV). Seizure action plan for school. "
            "Helmet worn continuously when ambulating. Developmental assessment. "
            "AVOID: CBZ, OXC, PHT, VGB."
        ),
    },
    {
        "window": "Transition / Early Remission",
        "age_range": "6–10 years",
        "key_events": "Drop attacks reducing; consider KD weaning; educational planning",
        "focus": (
            "Monitor 2y seizure-free before AED taper. KD can be weaned after ≥2y seizure-free "
            "with gradual ratio reduction (4:1 → 3:1 → 2:1 over 6M). "
            "Neuropsychological assessment for learning support needs."
        ),
    },
    {
        "window": "School-Age Remission",
        "age_range": "8–12 years",
        "key_events": "50% reach seizure freedom; AED taper; learning support ongoing",
        "focus": (
            "If ≥2y seizure-free, taper slowest AED first over 6–12M (not KD first). "
            "SUDEP counselling for residual drug-resistant cases. Driving guidance deferred "
            "to 12M seizure-free (jurisdiction-dependent)."
        ),
    },
    {
        "window": "Adolescence",
        "age_range": "12–18 years",
        "key_events": "Puberty hormonal shifts; JME evolution watch; AED optimisation",
        "focus": (
            "Watch for JME evolution (myoclonic jerks on waking post-puberty). "
            "Catamenial pattern in females: track seizure-menstrual correlation. "
            "Reproductive counselling for VPA (REMS) in females. "
            "30% of MAE-drug-resistant cases evolve to persistent adult epilepsy."
        ),
    },
    {
        "window": "Adult Life (Remitted vs Persistent)",
        "age_range": "18+ years",
        "key_events": "AED-free if remitted; employment/driving counselling; genetic family counselling",
        "focus": (
            "50% of MAE reach full remission by late adolescence. Persistent 30%: continue "
            "optimised AED regimen. Family genetic counselling (GGE risk 10–15% first-degree "
            "relatives). SUDEP risk assessment annually for drug-resistant cases."
        ),
    },
]

# ── Standards ──────────────────────────────────────────────────────────────────
STANDARDS = [
    {"standard": "ILAE Classification 2022", "relevance": "MAE classified under self-limited GGE spectrum with distinct drop-attack phenotype"},
    {"standard": "NICE NG217 (2022)", "relevance": "Broad-spectrum AED (VPA/LEV) recommended; KD referral for refractory cases"},
    {"standard": "ILAE KD Guidelines 2018 (Kossoff et al.)", "relevance": "Level A evidence for KD in MAE — Level A specific to this syndrome"},
    {"standard": "FDA VPA REMS 2013", "relevance": "Mandatory Risk Evaluation and Mitigation Strategy for valproate in females ≥4y"},
    {"standard": "Oguni 2005 Epileptic Disord", "relevance": "Seminal classification and natural history study; differentiates MAE from LGS"},
    {"standard": "Caraballo 2013 Seizure", "relevance": "KD-specific evidence in MAE; 50–60% responder rate established"},
]

# ── Thresholds ─────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "Drug-Resistant MAE (DRE)", "value": "≥2 adequate AED trials failed"},
    {"threshold": "KD initiation trigger", "value": "≥1 failed AED OR patient/family preference (earlier than other syndromes)"},
    {"threshold": "VPA TDM target", "value": "50–100 µg/mL"},
    {"threshold": "KD blood BHB target", "value": "2–4 mmol/L (urine ketones 3–4+)"},
    {"threshold": "AED taper eligibility", "value": "≥2 years seizure-free (VPA taper first; KD wean last)"},
    {"threshold": "Driving eligibility", "value": "12 months seizure-free (jurisdiction-dependent)"},
    {"threshold": "Folic acid (VPA + females)", "value": "5 mg/day periconceptionally"},
    {"threshold": "LTG worsening threshold", "value": "Any increase in myoclonic jerks within 4W of LTG start → discontinue"},
]

# ── Concepts ──────────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "Doose Syndrome / MAE", "definition": "Myoclonic-Atonic Epilepsy — ILAE 2022 GGE syndrome; onset 1–6y; pathognomonic drop attacks (myoclonic-atonic); 50% achieve remission; distinct from LGS by absence of tonic seizures and better prognosis."},
    {"term": "Myoclonic-Atonic Seizure", "definition": "Pathognomonic MAE seizure: brief myoclonic jerk immediately followed by sudden loss of postural tone (atonic component), causing head-drop or body fall. Duration 1–3 seconds. No warning aura."},
    {"term": "Drop Attack (Astatic Seizure)", "definition": "Sudden fall caused by loss of postural tone; may be myoclonic-atonic, purely atonic, or tonic. In MAE, drop attacks are myoclonic-atonic; absence of isolated tonic drops distinguishes MAE from LGS."},
    {"term": "Doose Theta Rhythm", "definition": "4–7 Hz generalised theta rhythm seen in background EEG of MAE patients; represents thalamocortical network instability; normalises in 55–70% of KD responders."},
    {"term": "Ketogenic Diet (KD) Level A in MAE", "definition": "Uniquely among GGE syndromes, KD has Level A evidence in MAE with 50–60% responder rate — the highest KD efficacy of any GGE syndrome. BHB activates KATP channels and reduces thalamocortical burst pacing."},
    {"term": "LGS vs MAE Differential", "definition": "Key differentiating feature: MAE has NO isolated tonic seizures (LGS has prominent tonic seizures); MAE has 50% remission rate vs LGS 10%; MAE slow spike-wave is 2–3 Hz vs LGS 1.5–2.5 Hz; MAE background theta vs LGS diffuse slowing."},
    {"term": "SCN1A-MAE", "definition": "SCN1A loss-of-function variants causing MAE phenotype (without prolonged febrile seizures = not classic Dravet). Particularly sensitive to sodium-channel blocker worsening. Valproate + KD preferred."},
    {"term": "SLC6A1 / GAT-1", "definition": "GABA reuptake transporter gene; loss-of-function causes MAE with intellectual disability. Second-most-common monogenic MAE cause. KD particularly effective. SLC6A1 now tested on standard GGE gene panels."},
    {"term": "SYNGAP1", "definition": "Synaptic Ras-GTPase activating protein; haploinsufficiency causes MAE + intellectual disability + autism features. Hyperactivation of MAPK/ERK pathway at excitatory synapses. Emerging mTOR pathway link."},
    {"term": "AED Aggravation in MAE", "definition": "Class effect of sodium-channel blockers (CBZ, OXC, PHT) — worsen myoclonic-atonic seizures in >80% of MAE patients. VGB also worsens. LTG relative caution: 15–20% myoclonus worsening."},
    {"term": "UMRS (Unified Myoclonus Rating Scale)", "definition": "Validated clinical scale for quantifying myoclonus severity (0–120). Used to monitor CLB tolerance and treatment response. Assesses: action myoclonus, rest myoclonus, functional impact."},
    {"term": "JME Evolution", "definition": "In 10–15% of MAE patients, adolescence heralds transition to a JME-like phenotype (morning myoclonus on waking, GTCS). This is a GGE spectrum evolution, not treatment failure. AED adjustment (VPA remains effective)."},
    {"term": "Myoclonic Status Epilepticus", "definition": "Prolonged myoclonic seizure activity lasting >30 minutes; can be triggered by CBZ/OXC administration, sleep deprivation, or VPA withdrawal. Treatment: IV clonazepam + IV levetiracetam. Medical emergency."},
    {"term": "SUDEP in MAE", "definition": "Sudden Unexpected Death in Epilepsy: risk 1:500/year in drug-resistant MAE (lower than Dravet/LGS). Night-time supervision, prone-position avoidance, seizure-monitoring devices reduce risk. Annual SUDEP risk counselling mandatory."},
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    "Doose H. 1970. An EEG investigation of idiopathic generalized epilepsies with myoclonic astatic petit mal. Neuropadiatrie 1(4):400–410.",
    "Oguni H et al. 2005. Myoclonic-astatic epilepsy of early childhood — clinical and EEG analysis. Epileptic Disord 7(3):195–204.",
    "Caraballo RH et al. 2013. Ketogenic diet in patients with myoclonic-astatic epilepsy. Seizure 22(10):870–873.",
    "Guerrini R et al. 2010. Doose syndrome: natural history, EEG, and genetic studies. Epilepsia 51(12):2396–2404.",
    "Kelley SA & Scheffer IE. 2009. Clinical spectrum of Doose syndrome and its relationship to GEFS+. Curr Opin Neurol 22(2):140–145.",
    "Lemmon ME et al. 2012. Efficacy of the ketogenic diet in Lennox-Gastaut syndrome: a retrospective review. Epilepsia 53(12):2130–2135.",
]


# ── API Functions ─────────────────────────────────────────────────────────────

def overview():
    total = len(PATIENTS)
    male_n = sum(1 for p in PATIENTS if p["gender"] == "M")
    drop_attack_active = sum(1 for p in PATIENTS if p["drop_attacks_active"])
    kd_n = sum(1 for p in PATIENTS if p["kd_on_ketogenic_diet"])
    id_n = sum(1 for p in PATIENTS if p["intellectual_disability"])
    sf_n = sum(1 for p in PATIENTS if p["seizure_control"] == "Seizure-free")
    return {
        "syndrome": "Doose Syndrome — Myoclonic-Atonic Epilepsy (MAE)",
        "total_patients": total,
        "male_n": male_n,
        "male_pct": round(male_n / total * 100),
        "drop_attacks_active_n": drop_attack_active,
        "drop_attacks_active_pct": round(drop_attack_active / total * 100),
        "kd_n": kd_n,
        "kd_pct": round(kd_n / total * 100),
        "intellectual_disability_n": id_n,
        "intellectual_disability_pct": round(id_n / total * 100),
        "seizure_free_n": sf_n,
        "seizure_free_pct": round(sf_n / total * 100),
        "etiology_distribution": {et["etiology"]: et["n"] for et in ETIOLOGY_CATALOG},
        "prognosis_summary": {
            "seizure_free_at_adolescence": "~50% (vs 10% in LGS)",
            "drug_resistant": "~20–30%",
            "partial_responders": "~20–30%",
            "kd_responder_rate": "50–60% (highest GGE efficacy)",
            "tonic_seizures": "Absent (key LGS differentiator)",
            "remission_median_age": "~10–12 years",
            "jme_evolution_risk": "10–15% by adolescence",
        },
        "clinical_alerts": [
            "⛔ ABSOLUTE CI: CBZ, OXC, PHT — aggravate myoclonic-atonic seizures in ALL MAE patients",
            "⛔ ABSOLUTE CI: Vigabatrin (VGB) — worsens myoclonus + irreversible visual field constriction",
            "⚠️ RELATIVE CI: Lamotrigine — worsens myoclonus in 15–20%; extreme caution / avoid if prominent myoclonus",
            "✅ KD is Level A evidence in MAE — consider early (after 1 failed AED), not last resort",
            "⚠️ EXCLUDE mitochondrial disease (POLG screening) BEFORE starting Valproate",
            "⚠️ VPA REMS mandatory for females ≥4y — neural tube defect risk counselling",
        ],
        "references": REFERENCES,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M MDT"),
    }


def breakdown():
    return {
        "patients": PATIENTS,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES_DETAIL,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "aed_monitoring": AED_MONITORING,
        "lifecycle": LIFECYCLE,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
    }


def definitions():
    return {
        "absolute_contraindications": ABSOLUTE_CONTRAINDICATIONS,
        "concepts": CONCEPTS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
    }
