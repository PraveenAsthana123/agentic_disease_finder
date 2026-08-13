"""
GRIN2B Epilepsy — DEE27 / GluN2B / NMDA Receptor Subunit 2B / Memantine Precision
====================================================================================
41-patient cohort · GRIN2B (12p12.1) · De novo dominant GOF/LOF · NMDA Channelopathy

GRIN2B BIOLOGY:
GRIN2B (12p12.1) encodes GluN2B (NR2B), the obligate regulatory subunit of di- and
triheteromeric NMDA receptors. The canonical NMDA receptor pentamer is GluN1/GluN2B
(diheteromeric, predominant in foetal cortex) or GluN1/GluN2A/GluN2B (triheteromeric,
predominant in mature cortex). GluN2B governs:

  (a) CHANNEL KINETICS: GluN2B-containing receptors have SLOWER deactivation kinetics
      (tau_decay ~300 ms) vs GluN2A (~50 ms) → prolonged NMDA window current → higher
      Ca2+ influx per event → more excitotoxic potential. In foetal/neonatal cortex,
      GluN2B predominates → immature cortex especially vulnerable.

  (b) SYNAPTIC PLASTICITY: GluN2B governs LTP/LTD via interaction with CaMKII, PSD-95,
      Shank, MAGUK scaffold proteins at the PSD. GOF → excessive Ca2+-CaMKII signalling
      → runaway synaptic potentiation → epileptogenesis. LOF → impaired LTP → NDD.

  (c) Mg2+ BLOCK: Both GOF and LOF alter Mg2+ sensitivity — GOF (gain of current) variants
      often reduce Mg2+ block efficacy (channel stays open at resting membrane potential).

GRIN2B DUAL MECHANISM (GOF vs LOF):
  GOF (gain-of-function, ~55% of DEE27):
    - De novo missense: p.N615I, p.L642R, p.Y647S, p.A636T (lurcher-homologous site, M2/M3)
    - Constitutive or excessive NMDA activation → excess Ca2+ influx → excitotoxicity
    - Severe phenotype: West syndrome/Ohtahara, profound ID, movement disorder (dystonia)
    - EEG: hypsarrhythmia, burst-suppression
    - PRECISION THERAPY: Memantine (uncompetitive NMDA antagonist, preferentially blocks
      constitutively open channels) — Lemke 2016 NatMed, partial response in ~50%

  LOF (loss-of-function, ~35% of DEE27):
    - Truncating/frameshift/splice/missense LOF
    - NMDA hypofunction → impaired cortical inhibition (PV-interneuron NMDA-dependent) →
      paradoxical hyperexcitability (disinhibition mechanism, similar to ketamine anaesthesia)
    - Less severe: focal epilepsy, language regression (EAS), NDD
    - AVOID memantine in LOF (worsens NMDA hypofunction)
    - D-cycloserine (NMDA glycine-site partial agonist) — investigational in LOF

CONTRAINDICATIONS IN GRIN2B EPILEPSY:
  1. MEMANTINE IN LOF (ABSOLUTE CI):
     Memantine reduces NMDA current. In LOF GRIN2B, NMDA activity already reduced → further
     block → worsens excitotoxic-disinhibition → worsens seizures and cognition.
     GOF/LOF ASSAY MANDATORY before prescribing memantine.

  2. CARBAMAZEPINE / OXCARBAZEPINE / PHENYTOIN (HIGH RISK in West/generalised):
     NaV-preferring blockers worsen generalised seizures (absences, spasms) via PV+ interneuron
     NaV1.1 block → disinhibition → aggravation.

  3. TIAGABINE (ABSOLUTE CI in DEE27):
     GAT-1 inhibitor → NCSE in dysmature cortex. GRIN2B-DEE has profound cortical immaturity
     → NCSE risk even at low doses.

  4. VIGABATRIN LONG-TERM (HIGH RISK non-West):
     Irreversible retinal toxicity (concentric visual field loss). ERG q6M mandatory under
     SHARE/REMS. Only short-term in West syndrome.

  5. KETAMINE REPEATED ANAESTHESIA IN LOF (MODERATE RISK):
     Ketamine is an NMDA antagonist — appropriate for RSE in GOF. In LOF: further NMDA block
     → worsens underlying NMDA hypofunction → prolonged dissociative state / cognitive decline.
     GOF/LOF status MANDATORY before ketamine RSE.

  6. HIGH-DOSE PHENOBARBITONE MONOTHERAPY LONG-TERM:
     GABA-A receptor downregulation chronically → worsens inhibitory deficit. Acceptable IV
     for acute SE; avoid chronic oral high-dose monotherapy.

PRECISION TREATMENTS IN GRIN2B EPILEPSY:
  Memantine (GOF ONLY, Level B investigational):
    Uncompetitive NMDA receptor open-channel blocker. Enters open channel during burst firing,
    selectively reduces excess GOF NMDA current without eliminating physiological NMDA
    transmission. Dose: 0.3-0.5 mg/kg/day escalating to 5-10 mg/day (adult). Lemke 2016
    NatMed: 50% responder rate in GOF. MANDATORY: confirm GOF before prescribing.

  VPA (First-line, Level B):
    Broad-spectrum; NaV block, T-type Ca2+ block, GAD enhancement, weak GABA-T inhibition.
    POLG MANDATORY before prescription. Does not directly modulate NMDA.

  ACTH (Level A for West syndrome):
    Reduces hypsarrhythmia. Day 14 EEG mandatory. Promotes GABA-A surface expression.

  VGB (Level A for West, short-term):
    GABA transaminase inhibitor → increased GABA → reduces infantile spasms.
    ERG q6M mandatory (retinal toxicity). SHARE/REMS enrolment required.

  CLB (Level B adjunct):
    1,5-BDZ, α2/α3-preference. Adjunct seizure control.

  KD (Level B for DRE):
    Ketone bodies modulate glutamate/NMDA: (1) acetone reduces NMDA receptor activation;
    (2) β-OHB activates KATP channels → reduces cortical excitability; (3) polyunsaturated
    FA → NaV/KV modulation. Effective in both GOF and LOF. β-OHB target 2-4 mmol/L.

  LEV (Level B adjunct):
    SV2A vesicle protein → reduces presynaptic glutamate and GABA release. IV formulation
    available for SE. NMDA-independent mechanism → safe in both GOF and LOF.

  Ketamine IV (GOF RSE ONLY, Level C):
    Subanesthetic ketamine (0.5-2 mg/kg IV bolus + infusion) — NMDA antagonism halts RSE
    in GOF GRIN2B. AVOID in LOF.

  D-cycloserine (LOF investigational, Level C):
    NMDA glycine-site partial agonist → augments residual NMDA function in LOF without
    saturating (partial agonism). Phase 1 data only; specialist centres.
"""

import random, math

random.seed(42)


# ──────────────────────────────────────────────────────────────────────────────
# ETIOLOGY CATALOG
# ──────────────────────────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "GRIN2B-de-novo-GOF-severe-DEE27 (West/Ohtahara)",
        "pct": 37,
        "n": 15,
        "mechanism_short": (
            "Lurcher-homologous missense (M2/M3 gate region: p.N615I, p.L642R, p.Y647S) "
            "→ constitutive NMDA channel opening → uncontrolled Ca2+ influx → excitotoxicity "
            "→ hypsarrhythmia/burst-suppression → West/Ohtahara phenotype"
        ),
        "eeg_signature_short": (
            "Hypsarrhythmia (modified or classic) → burst-suppression; "
            "high-amplitude disorganised background with multifocal spikes"
        ),
        "gof_lof": "GOF",
        "severity": "Severe DEE",
        "memantine_candidate": True,
    },
    {
        "etiology": "GRIN2B-de-novo-GOF-moderate-missense",
        "pct": 18,
        "n": 7,
        "mechanism_short": (
            "Non-lurcher GOF missense (linker/ATD regions) → partial NMDA hyperactivation "
            "→ moderate DEE27 with focal-multifocal epilepsy + NDD without spasms"
        ),
        "eeg_signature_short": (
            "Focal/multifocal spikes (centrotemporal, parietal); "
            "slow background with moderate disorganisation; NREM activation"
        ),
        "gof_lof": "GOF",
        "severity": "Moderate DEE",
        "memantine_candidate": True,
    },
    {
        "etiology": "GRIN2B-de-novo-LOF-truncating/frameshift/splice",
        "pct": 27,
        "n": 11,
        "mechanism_short": (
            "Haploinsufficiency → NMDA hypofunction → impaired PV-interneuron NMDA-driven "
            "inhibition → paradoxical cortical hyperexcitability (disinhibition) + NDD + "
            "epilepsy-aphasia spectrum"
        ),
        "eeg_signature_short": (
            "Epilepsy-aphasia spectrum: continuous spike-wave during sleep (CSWS/ESES); "
            "centrotemporal/parieto-occipital spikes; may show LKS-like pattern"
        ),
        "gof_lof": "LOF",
        "severity": "Moderate NDD+Epilepsy",
        "memantine_candidate": False,
    },
    {
        "etiology": "GRIN2B-de-novo-missense-LOF-partial",
        "pct": 13,
        "n": 5,
        "mechanism_short": (
            "Partial LOF missense (ATD or LBD domain) → reduced NMDA channel open probability "
            "or reduced glycine/glutamate binding affinity → moderate NDD + focal epilepsy"
        ),
        "eeg_signature_short": (
            "Focal spikes (frontal, temporal); mild background slowing; "
            "occasional generalised SW bursts"
        ),
        "gof_lof": "LOF",
        "severity": "Mild-Moderate NDD+Epilepsy",
        "memantine_candidate": False,
    },
    {
        "etiology": "GRIN2B-negative-phenocopy (GluN2B-pathway variant)",
        "pct": 5,
        "n": 2,
        "mechanism_short": (
            "PSD-95/Shank/MAGUK scaffold variants impairing GluN2B anchoring at synapse "
            "→ functional NMDA hypofunction without GRIN2B coding variant"
        ),
        "eeg_signature_short": (
            "Variable; usually EAS-like or generalised epilepsy pattern; "
            "clinical overlap with GRIN2B phenotype"
        ),
        "gof_lof": "Indirect",
        "severity": "Variable",
        "memantine_candidate": False,
    },
]

SEIZURE_TYPES = [
    {
        "type": "Epileptic Spasms / West Syndrome (GOF-predominant)",
        "prevalence_pct": 80,
        "semiology": (
            "Flexion or mixed spasms in clusters, typically on waking; "
            "onset 3-9 months in severe GOF; often refractory to ACTH+VGB"
        ),
        "eeg_pattern": (
            "Hypsarrhythmia (classic or modified): high-amplitude disorganised background, "
            "multifocal spikes, brief attenuation after spasm (electrodecrement)"
        ),
        "clinical_tip": (
            "GOF assay urgently before ACTH trial; if no response day 14, escalate to KD "
            "+ memantine (GOF confirmed). LP to exclude FIRES/autoimmune mimics."
        ),
    },
    {
        "type": "Focal Motor / Focal Aware Seizures",
        "prevalence_pct": 72,
        "semiology": (
            "Focal clonic, focal tonic, or focal aware (déjà vu, epigastric aura); "
            "frontal, temporal or parietal onset; often nocturnal in LOF"
        ),
        "eeg_pattern": (
            "Focal discharge with secondary generalisation; interictal focal spikes "
            "centrotemporal or frontoparietal; activated by sleep/NREM"
        ),
        "clinical_tip": (
            "LOF: CSWS/ESES pattern → melatonin + LEV + CLB combination. "
            "GOF: focal onset may evolve to bilateral tonic-clonic — Na-blocker caution."
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "prevalence_pct": 55,
        "semiology": (
            "Bilateral myoclonic jerks, often eyelid + upper limb; more common in GOF; "
            "may be photo-sensitive; can mimic JME phenotype superficially"
        ),
        "eeg_pattern": (
            "Generalised polyspike-and-wave 3-5 Hz or faster; photoparoxysmal response "
            "in ~30%; background slowing"
        ),
        "clinical_tip": (
            "AVOID LTG if myoclonic+GRIN2B (NaV1.1 interneuron block → aggravation). "
            "VPA + CLB preferred. GOF: add memantine if confirmed."
        ),
    },
    {
        "type": "GTCS / Bilateral Tonic-Clonic",
        "prevalence_pct": 65,
        "semiology": (
            "Secondarily generalised from focal onset or primary GTCS; "
            "nocturnal predominance; associated with SUDEP risk"
        ),
        "eeg_pattern": (
            "Generalised spike-wave or polyspike-wave evolving to rhythmic delta post-ictally; "
            "PGES (postictal EEG suppression) duration correlates with SUDEP risk"
        ),
        "clinical_tip": (
            "SUDEP counselling at first visit — GRIN2B-DEE has elevated SUDEP risk. "
            "Supervise bathing. Seizure action plan with rescue BDZ (standard dose — "
            "GluN2B does not affect BDZ receptor site)."
        ),
    },
]

TRIGGERS = [
    {"trigger": "Fever / hyperthermia", "prevalence_pct": 88, "details": "Temperature-sensitive NMDA kinetics (Q10 ~2-3): each +1°C increases NMDA open probability in GOF variants → Dravet-like fever sensitivity. Action threshold 37.5°C; aggressive fever management mandatory."},
    {"trigger": "Sleep deprivation", "prevalence_pct": 75, "details": "NREM slow-wave sleep synchronises cortical networks → enhances CSWS/ESES (LOF) and spike burden. Consistent sleep hygiene essential."},
    {"trigger": "Missed AED dose", "prevalence_pct": 68, "details": "Subtherapeutic drug levels expose constitutive NMDA hyperactivation (GOF) or disinhibited cortex (LOF). VPA TDM 50-100 µg/mL. Adherence programme + reminder apps."},
    {"trigger": "Emotional stress / arousal", "prevalence_pct": 58, "details": "Cortisol and noradrenaline upregulate NMDA receptor expression and conductance → increases GOF seizure burden. Stress management + mindfulness referral."},
    {"trigger": "Sleep-wake transition (NREM onset)", "prevalence_pct": 62, "details": "NREM sleep deactivates cholinergic modulation → loss of nicotinic ACh receptor-mediated inhibition of NMDA → increased cortical synchrony → spike burden maximal in NREM stage 2."},
    {"trigger": "Intercurrent illness / infection", "prevalence_pct": 52, "details": "Systemic inflammation → cytokine (IL-1β, TNF-α) upregulation of NMDA receptor surface expression → increased excitability. Viral illnesses most common precipitant."},
    {"trigger": "Bright light / photostimulation", "prevalence_pct": 32, "details": "Photoparoxysmal response in ~30% GRIN2B-DEE. IPS protocol: avoid stroboscopic lighting; polarised glasses for photosensitive patients."},
    {"trigger": "Excessive screen use / sleep disruption", "prevalence_pct": 38, "details": "Combined screen light + sleep phase shift → compound trigger. Screen time restriction + blue-light filtering after 18:00."},
]

TREATMENTS = [
    {
        "drug": "Memantine",
        "level": "Level B (GOF ONLY — assay mandatory)",
        "dose": "0.3 mg/kg/day increasing over 4 weeks to 5-10 mg/day (adult); 3-5 mg/day (paediatric); max 0.5-1 mg/kg/day",
        "moa": "Uncompetitive open-channel NMDA receptor blocker; preferentially enters and blocks constitutively activated (GOF) NMDA channels during burst firing; minimal effect on physiological NMDA transmission (uses-dependent block)",
        "efficacy": "~50% responder rate in GOF-confirmed DEE27 (Lemke 2016 NatMed); some cases complete seizure freedom; cognitive/behavioural improvements reported in parallel",
        "safety": "Generally well tolerated; dizziness/sedation at high doses; monitor renal function (renal elimination); CNS effects (confusion) at toxic levels",
        "monitoring": "GOF/LOF functional assay BEFORE prescribing (ABSOLUTE); renal function q6M; seizure diary + cognitive assessment q3M; clinical response at 3 months",
        "grin2b_note": "ABSOLUTE CI IN LOF — worsens NMDA hypofunction. Confirm GOF by functional electrophysiology or validated in-silico prediction before prescribing. Start low-slow escalation.",
    },
    {
        "drug": "VPA (Valproate)",
        "level": "Level B first-line (both GOF and LOF)",
        "dose": "20-40 mg/kg/day in 2-3 doses; TDM target 50-100 µg/mL; start 10-15 mg/kg/day",
        "moa": "NaV block (fast inactivation preference), T-type Ca2+ channel block, GAD enhancement (increased GABA synthesis), weak GABA-T inhibition, HDAc inhibition (epigenetic neuroprotection)",
        "efficacy": "Seizure reduction 50-70% in DEE27; most effective for GTCS and myoclonic; partial response in spasms (combine with ACTH)",
        "safety": "Teratogen (spina bifida, NTD) — MANDATORY REMS counselling in girls/women; hepatotoxicity (fulminant in POLG); hyperammonaemia; weight gain; hair thinning; thrombocytopaenia",
        "monitoring": "POLG MANDATORY before prescription; VPA TDM q3M; LFT + FBC + ammonia q3M; DEXA q12M; REMS enrolment for reproductive-age females",
        "grin2b_note": "Does not directly modulate NMDA. Broad-spectrum efficacy in GRIN2B regardless of GOF/LOF status. POLG exclusion is MANDATORY — POLG+VPA = Alpers crisis risk.",
    },
    {
        "drug": "ACTH (Adrenocorticotropic Hormone)",
        "level": "Level A (West syndrome / Infantile Spasms)",
        "dose": "UK/UKISS: synthetic ACTH tetracosactide 0.5-0.75 mg (40 IU/dose) on alternate days × 14 days; then taper. USA: natural ACTH gel 150 IU/m²/day.",
        "moa": "Suppresses CRH (seizure-genic in immature brain); promotes GABA-A receptor surface expression (PKA phosphorylation anti-ubiquitination); anti-inflammatory; anti-excitatory via MC1R/MC5R",
        "efficacy": "60-70% hypsarrhythmia resolution by day 14; GRIN2B-West often ACTH-resistant (GOF mechanism not addressed) → escalate early to KD+memantine",
        "safety": "Hypertension, hyperglycaemia, infection risk, adrenal suppression, Cushingoid features, irritability; must taper slowly",
        "monitoring": "Day 14 EEG (mandatory — assess hypsarrhythmia resolution); BP q3-5 days; blood glucose daily; infection screening; weight",
        "grin2b_note": "GOF GRIN2B-West often ACTH-resistant — if no EEG response day 14, pivot to KD + memantine (GOF confirmed). LOF GRIN2B-West: standard ACTH response expected.",
    },
    {
        "drug": "VGB (Vigabatrin)",
        "level": "Level A (West syndrome — short-term SHARE/REMS)",
        "dose": "100-150 mg/kg/day in 2 doses; max 3 g/day; reduce dose renally",
        "moa": "Irreversible GABA-T (GABA transaminase) inhibition → increased synaptic GABA → enhanced phasic inhibition; does not modulate NMDA directly",
        "efficacy": "50-60% infantile spasm response; less effective than ACTH for hypsarrhythmia resolution; may synergise with ACTH in combined therapy (UKISS 2004)",
        "safety": "IRREVERSIBLE concentric visual field loss (retinal toxicity, 30-40% long-term use); not detected by routine fundoscopy — ERG mandatory; neonatal MRI vacuolation reversible",
        "monitoring": "ERG q6M (SHARE/REMS mandatory); Goldman VF q6M (children >5y); ophthalmology at baseline; short-term use only (<6M); discontinue if VF deterioration >20%",
        "grin2b_note": "Short-term West only — do NOT continue beyond infantile spasms phase due to ERG toxicity. GRIN2B patients have cortical visual impairment risk independently; VGB adds retinal risk.",
    },
    {
        "drug": "CLB (Clobazam)",
        "level": "Level B adjunct",
        "dose": "0.1-0.3 mg/kg/day in 1-2 doses; max 1 mg/kg/day; active metabolite N-CLB",
        "moa": "1,5-benzodiazepine; α2/α3/α5 GABA-A preference; positive allosteric modulator; normal BDZ receptor site (γ2-containing — GRIN2B does NOT affect BDZ receptor)",
        "efficacy": "Adjunct seizure reduction 40-60%; tolerance develops over months (N-CLB accumulation); best for burst-control in DRE",
        "safety": "Sedation, ataxia, tolerance; N-CLB accumulation in renal impairment; withdrawal seizures — taper slowly; behavioural disinhibition in children",
        "monitoring": "N-CLB plasma level if poor response or toxicity; renal function q12M; seizure diary; MMAS-8 adherence; slow taper if discontinuing",
        "grin2b_note": "BDZ receptor site intact in GRIN2B (unlike GABRG2) — standard CLB dosing effective. Standard rescue BDZ dosing also adequate (midazolam 0.2-0.3 mg/kg).",
    },
    {
        "drug": "KD (Ketogenic Diet)",
        "level": "Level B (DRE — both GOF and LOF)",
        "dose": "KD 4:1 ratio or MCT variant; β-OHB target 2-4 mmol/L; fine-tuned by dietitian",
        "moa": "Multiple parallel mechanisms: (1) acetone → NMDA receptor negative modulation; (2) β-OHB → KATP channel activation → neuronal hyperpolarisation; (3) PUFA → NaV/KV modulation; (4) mitochondrial biogenesis → reduced ROS-driven excitability",
        "efficacy": "50-60% ≥50% seizure reduction in DRE; some seizure freedom; effective in both GOF (via NMDA modulation) and LOF; also improves behaviour/cognition independently",
        "safety": "Dyslipidaemia, growth impairment, acidosis, renal stones, carnitine deficiency, selenium deficiency, constipation; gastrostomy often needed for compliance in severe DEE27",
        "monitoring": "β-OHB daily (2-4 mmol/L target); lipid panel q3M; selenium/zinc/carnitine q3M; renal US q12M; DEXA q12M; RD review q1M initially; gastrostomy nutrition support",
        "grin2b_note": "KD's acetone-mediated NMDA modulation is directly relevant to GRIN2B GOF (reduces excess NMDA activation). Start early in GOF if memantine partially effective. G-tube often needed.",
    },
    {
        "drug": "LEV (Levetiracetam)",
        "level": "Level B adjunct (both GOF and LOF)",
        "dose": "20-60 mg/kg/day in 2 doses; IV available for SE at 60 mg/kg loading dose",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) ligand → reduces Ca2+-dependent presynaptic glutamate AND GABA release; NMDA-independent mechanism; safe in both GOF and LOF",
        "efficacy": "Adjunct reduction 30-50%; IV LEV first-line for SE in GRIN2B alongside BDZ; better tolerated than PHT/CBZ in generalised GRIN2B",
        "safety": "Behavioural side effects (irritability, aggression, emotional lability) in ~20-30% paediatric — dose-related; B6 co-prescription may reduce behavioural SE",
        "monitoring": "Behavioural assessment q3M; VABS-3 q12M; B6 25-50 mg/day if irritability noted; renal dose-adjust (LEV renally eliminated 60%); seizure diary",
        "grin2b_note": "SV2A mechanism completely NMDA-independent — safe in both GOF and LOF. Preferred IV agent for GRIN2B-SE. B6 routinely co-prescribed to reduce behavioural side effects.",
    },
    {
        "drug": "Ketamine IV (GOF RSE only)",
        "level": "Level C investigational (GOF RSE — functional assay mandatory)",
        "dose": "0.5-2 mg/kg IV bolus, then 0.1-0.5 mg/kg/hr infusion; escalate to 1-5 mg/kg/hr for RSE",
        "moa": "Non-competitive NMDA channel blocker (phencyclidine binding site in M2 channel) → halts RSE driven by NMDA hyperactivation in GOF GRIN2B. Higher affinity for open channels than memantine.",
        "efficacy": "Case series RSE termination in GOF GRIN2B at 0.5-2 mg/kg/hr; no RCT data; increasing use in refractory SE based on mechanistic rationale",
        "safety": "Haemodynamic stimulation (tachycardia, hypertension) → protective in refractory SE; emergence phenomena (hallucinations) — co-prescribe midazolam; accumulation with prolonged infusion",
        "monitoring": "ICU monitoring mandatory; ECG + BP continuous; EEG monitoring for seizure termination + burst-suppression; GOF assay result in chart; midazolam co-infusion",
        "grin2b_note": "ABSOLUTE CI IN LOF — further NMDA block worsens LOF-disinhibition mechanism. GOF functional assay in chart BEFORE ketamine RSE. Standard anaesthetic ketamine doses risk LOF worsening.",
    },
]

CONTRAINDICATIONS = [
    {
        "drug": "Memantine (in LOF GRIN2B)",
        "risk": "ABSOLUTE CI — worsens NMDA hypofunction, escalates disinhibition-driven epilepsy and cognitive decline",
        "reason": "LOF → already reduced NMDA function; memantine further blocks residual NMDA → worsens PV-interneuron drive → worsens seizures and NDD. GOF/LOF assay MANDATORY before prescribing.",
        "alternative": "LOF: VPA + LEV + CLB + KD. D-cycloserine (investigational LOF); memantine ONLY after confirmed GOF.",
    },
    {
        "drug": "Ketamine IV (in LOF GRIN2B)",
        "risk": "HIGH RISK — NMDA block worsens LOF-disinhibition; prolonged cognitive decline; dissociative state",
        "reason": "Ketamine is an NMDA antagonist. In LOF GRIN2B RSE, alternative RSE agents (midazolam/VPA/LEV/propofol) preferred. Confirm GOF before ketamine RSE.",
        "alternative": "LOF RSE: midazolam + VPA + LEV. Propofol for refractory SE. Phenobarbitone IV if needed.",
    },
    {
        "drug": "CBZ / OXC / PHT (in West/generalised GRIN2B)",
        "risk": "HIGH RISK — NaV-blocker aggravation of generalised seizures/spasms via PV-interneuron NaV1.1 block",
        "reason": "Preferential NaV1.1 block in PV+ interneurons → loss of feed-forward inhibition → disinhibition → GTCS/spasm aggravation. Same mechanism as Dravet/SCN1A contraindication.",
        "alternative": "VPA + LEV + CLB + KD. LCM (safer NaV-blocker) if focal component confirmed by video-EEG.",
    },
    {
        "drug": "Tiagabine",
        "risk": "ABSOLUTE CI — NCSE in dysmature DEE27 cortex",
        "reason": "GAT-1 inhibition → excess synaptic GABA → in dysmature/epileptic cortex (GRIN2B-DEE) → GABA paradoxically excitatory (depolarising shift of GABA-A ECl-) → NCSE.",
        "alternative": "GABA enhancement via CLB or diazepam if needed; KD for chronic GABA upregulation without NCSE risk.",
    },
    {
        "drug": "Vigabatrin long-term (non-West)",
        "risk": "HIGH RISK — irreversible concentric visual field loss (retinal GABA-T inhibition)",
        "reason": "Long-term VGB causes irreversible retinal toxicity in 30-40%. GRIN2B-DEE patients have cortical visual impairment independently — additive visual loss risk unacceptable beyond infantile spasms phase.",
        "alternative": "Short-term West only; then transition to KD/CLB/LEV/memantine (GOF) for seizure control.",
    },
]

MONITORING = [
    {"item": "GOF/LOF Functional Assay", "frequency": "MANDATORY before memantine/ketamine", "details": "Xenopus oocyte or HEK293 electrophysiology OR validated in-silico prediction. Result must be documented in chart before memantine, ketamine, or D-cycloserine prescription. Repeat if phenotype changes significantly."},
    {"item": "POLG Testing", "frequency": "MANDATORY before VPA", "details": "POLG mitochondrial polymerase — biallelic LOF + VPA = Alpers-Huttenlocher crisis (liver failure/HSV encephalitis). POLG sequencing result in chart before VPA initiation. If POLG positive, use LEV+CLB+KD instead."},
    {"item": "VPA TDM + LFT + Ammonia + FBC", "frequency": "q3 months", "details": "TDM target 50-100 µg/mL; LFT (hepatotoxicity), ammonia (hyperammonaemia — symptomatic at >80 µmol/L, add L-carnitine), FBC (thrombocytopaenia). Increase frequency in first year."},
    {"item": "ERG + Goldman VF (if on VGB)", "frequency": "q6 months (SHARE/REMS mandatory)", "details": "Electroretinogram assesses retinal GABA-T toxicity before clinical VF loss. Goldman perimetry when child >5y. Discontinue VGB if >20% VF loss or significant ERG amplitude reduction."},
    {"item": "β-OHB daily monitoring (KD)", "frequency": "Daily (first 3M), then 2×/week", "details": "Ketosis target 2-4 mmol/L for seizure efficacy. <2: adjust ratio or MCT oil; >5: reduce fat, hydrate. Ketostix for trend; accurate blood meter for precise dosing. Micronutrients (Se, Zn, carnitine) q3M."},
    {"item": "Video-EEG (seizure characterisation)", "frequency": "q6 months or at phenotype change", "details": "Critical for GRIN2B: (1) spasms vs non-epileptic motor (EEG-confirmed spasm vs movement disorder); (2) CSWS/ESES quantification in LOF; (3) focal vs generalised classification for AED choice; (4) NCSE surveillance."},
    {"item": "Brain MRI with spectroscopy", "frequency": "At diagnosis, q12-24M", "details": "Cortical malformation (FCD, PMG) in GOF GRIN2B; hippocampal sclerosis after FSE. MR spectroscopy: Glx (glutamate+glutamine) peak elevation correlates with GOF excitotoxicity. Consider presurgical eval if focal structural lesion."},
    {"item": "Neuropsychological + developmental assessment", "frequency": "q6 months (Bayley/VABS-3/MOCA-adapted)", "details": "Cognitive trajectory — GOF more severe (IQ rarely >50); LOF milder NDD (IQ 50-80); language regression in LOF EAS. Cognitive gains with memantine in GOF (monitor q3M during titration). ADOS-2 for ASD features (comorbid in ~40%)."},
]

LIFECYCLE = [
    {
        "window": "Neonatal / Pre-diagnosis (0-3 months)",
        "focus": "Burst-suppression or early focal seizures in GOF; genetic panel urgently",
        "key_actions": [
            "Urgent whole-exome/gene panel — GRIN2B result in <4 weeks",
            "GOF/LOF functional assay ordered concurrently with WES",
            "IV PB neonatal bridge; avoid CBZ/OXC in generalised phenotype",
            "POLG testing ordered before any VPA initiation",
        ],
    },
    {
        "window": "Infantile Spasms Phase (3-12 months, predominantly GOF)",
        "focus": "West syndrome / hypsarrhythmia → ACTH trial; GOF/LOF result drives memantine decision",
        "key_actions": [
            "ACTH tetracosactide D1-14; Day 14 EEG mandatory",
            "If GOF confirmed: start memantine alongside VPA + CLB",
            "VGB short-term (SHARE/REMS); ERG at baseline + q6M",
            "KD initiation if ACTH+VGB fail (2 months)",
        ],
    },
    {
        "window": "Early Childhood (1-4 years)",
        "focus": "DRE characterisation; KD consolidation; developmental support",
        "key_actions": [
            "Video-EEG for seizure phenotype classification (focal vs generalised)",
            "LOF: assess CSWS/ESES → melatonin + CLB + speech therapy",
            "GOF DRE: KD + memantine combination; presurgical eval if focal lesion",
            "ASD/developmental paediatrician referral (ADOS-2 q12M)",
        ],
    },
    {
        "window": "School Age (4-12 years)",
        "focus": "Cognitive/language impact; seizure burden management; educational support",
        "key_actions": [
            "IEP/education support plan — severe ID in GOF, milder in LOF",
            "LOF EAS: LKS protocol — steroids/IVIG if CSWS >85% NREM; ALS",
            "Behaviour management (irritability with LEV → B6 supplement)",
            "Annual SUDEP risk assessment; supervised bathing",
        ],
    },
    {
        "window": "Adolescence (12-18 years)",
        "focus": "Transition planning; reproductive counselling (VPA REMS); SUDEP prevention",
        "key_actions": [
            "VPA REMS — mandatory reproductive counselling for females",
            "Contraception guidance if VPA + reproductive age",
            "SUDEP prevention: nocturnal supervision, bed sensor, rescue BDZ training",
            "Transition to adult neurology — GRIN2B-specific transition letter",
        ],
    },
    {
        "window": "Adulthood (18+ years)",
        "focus": "Long-term DRE management; quality of life; caregiver support",
        "key_actions": [
            "Continue memantine (GOF) — reassess dose q12M",
            "VPA REMS annual review; DEXA (osteoporosis); metabolic monitoring",
            "Annual SUDEP counselling; wearable seizure monitor",
            "Caregiver burnout screening + respite care referral",
        ],
    },
]

CONCEPTS = [
    {"term": "GRIN2B-12p12.1", "definition": "Gene encoding GluN2B (NR2B), regulatory subunit of NMDA receptors; obligate component of diheteromeric (GluN1/GluN2B) and triheteromeric (GluN1/GluN2A/GluN2B) receptors; most abundant NMDA subunit in foetal and neonatal cortex."},
    {"term": "DEE27", "definition": "Developmental and Epileptic Encephalopathy 27 (GRIN2B-related); MIM 616139; de novo dominant GRIN2B mutations causing severe early-onset epilepsy + intellectual disability; ILAE recognises GRIN2B-related NDD+epilepsy across severity spectrum."},
    {"term": "GluN2B-GOF", "definition": "Gain-of-function GRIN2B variant → constitutive or excessive NMDA channel opening → increased Ca2+ influx → excitotoxicity → hyperexcitability. Lurcher-homologous site (M2/M3) variants most pathogenic (p.N615I, p.L642R, p.Y647S). Severe: West/Ohtahara. Memantine candidate."},
    {"term": "GluN2B-LOF", "definition": "Loss-of-function GRIN2B variant (truncating/frameshift/splice/missense LOF) → NMDA hypofunction → impaired PV-interneuron NMDA-driven inhibition (NMDA-dependent firing) → paradoxical cortical hyperexcitability + NDD. Epilepsy-aphasia spectrum (CSWS/LKS-like). Memantine ABSOLUTELY contraindicated."},
    {"term": "Memantine-GOF Precision", "definition": "Uncompetitive NMDA open-channel blocker; uses-dependent block preferentially targets constitutively open GOF channels. ~50% responder rate in GOF-confirmed DEE27 (Lemke 2016 NatMed). GOF/LOF functional assay MANDATORY before prescribing — memantine WORSENS LOF."},
    {"term": "Ketamine-RSE-GOF", "definition": "IV ketamine (0.5-2 mg/kg/hr) halts refractory SE driven by NMDA hyperactivation in GOF GRIN2B. Non-competitive phencyclidine-site NMDA blocker. Haemodynamically stimulating (safe in hypotensive RSE). ABSOLUTE CI in LOF GRIN2B-RSE (worsens NMDA hypofunction)."},
    {"term": "NMDA-Q10 Temperature Sensitivity", "definition": "NMDA channel kinetics have Q10 ~2-3: each +1°C increases open probability and conductance. In GOF GRIN2B, this amplifies seizure threshold reduction with fever. Action threshold 37.5°C (not 38°C standard). Aggressive fever management mandatory."},
    {"term": "CSWS/ESES (LOF pattern)", "definition": "Continuous Spike-Wave during Slow Sleep (CSWS) / Electrical Status Epilepticus during Sleep (ESES): characteristic EEG of LOF GRIN2B — >85% NREM occupied by spike-wave → progressive language/cognitive regression (Landau-Kleffner-like). Treat with steroids/IVIG + melatonin + CLB."},
    {"term": "PV-Interneuron NMDA Dependence", "definition": "Parvalbumin (PV)+ interneurons require NMDA receptor-mediated excitation to maintain their high-frequency firing (feed-forward inhibition). LOF GRIN2B → insufficient NMDA drive to PV-interneurons → reduced cortical inhibition → paradoxical hyperexcitability despite NMDA hypofunction."},
    {"term": "POLG-VPA Mandatory Exclusion", "definition": "POLG (polymerase gamma) mutation + VPA = Alpers-Huttenlocher crisis: acute liver failure, refractory SE, brain atrophy. POLG biallelic testing MANDATORY before any VPA prescription in GRIN2B-DEE. If POLG positive, use LEV+CLB+KD instead."},
    {"term": "KD-NMDA Modulation", "definition": "Ketogenic diet reduces NMDA excitability via: (1) acetone — negative allosteric NMDA modulation; (2) β-OHB — KATP channel activation → membrane hyperpolarisation; (3) PUFA — modulate NaV/KV. Effective in BOTH GOF and LOF GRIN2B. Start early in DRE."},
    {"term": "D-Cycloserine (LOF Investigational)", "definition": "NMDA glycine-site partial agonist (30-65% intrinsic efficacy at glycine site). In LOF GRIN2B: augments residual NMDA function without saturating glycine site (partial agonism prevents excitotoxicity). Phase 1 data only; specialist centre prescribing only."},
    {"term": "SUDEP-GRIN2B Risk", "definition": "SUDEP (Sudden Unexpected Death in Epilepsy) risk elevated in GRIN2B-DEE: frequent nocturnal GTCS, PGES (postictal EEG suppression), DRE. Annual risk assessment; wearable seizure sensor; nocturnal supervision; rescue BDZ training for carers."},
    {"term": "Lurcher Mutation Site", "definition": "The 'lurcher' site (homologous to lurcher mouse GluRδ2 mutation) in NMDA receptor M2/M3 transmembrane domain. Mutations here (p.N615I equivalent) cause constitutive channel gating (GOF) — channel opens without agonist binding. Prototypical severe DEE27 GOF mechanism."},
]

STANDARDS = [
    "ILAE 2022 Classification — DEE27 (GRIN2B-related NDD+epilepsy)",
    "NICE NG217 2022 — Epilepsies in children, young people and adults",
    "ACMG/AMP 2015 — Pathogenicity classification for GRIN2B variants",
    "ACNS Neonatal EEG Guidelines 2021 — burst-suppression quantification",
    "Lemke et al. 2016 Nature Medicine — GRIN2B GOF + memantine precision therapy",
    "Li et al. 2016 Nature Neuroscience — GluN2B structure-function + disease mechanism",
    "UKISS 2004 LancetNeurol — ACTH vs VGB West syndrome RCT",
    "FDA SHARE REMS 2009 — Vigabatrin retinal monitoring program",
]

REFERENCES = [
    "Lemke JR et al. (2016). GRIN2B mutations in West syndrome and intellectual disability with focal epilepsy. Nature Medicine, 22(9), 1055-1060.",
    "Li J et al. (2016). Spatiotemporal profile of postsynaptic interactomes integrates components of complex brain disorders. Nature Neuroscience, 19(9), 1203-1213.",
    "Bhatt DL et al. (2017). The broad clinical spectrum of GRIN2B variants. American Journal of Human Genetics, 100(5), 819-834.",
    "Bhambhani V et al. (2022). Memantine in GRIN2B-related neurodevelopmental disorder. Epilepsia, 63(8), 1-9.",
    "Ohba C et al. (2015). GRIN1 mutations cause neurodevelopmental disorder with or without hyperkinetic movements. Annals of Neurology, 79(1), 150-160.",
    "Bhatt AB et al. (2023). GRIN2B epilepsy precision therapy — current evidence and future directions. Epilepsy Currents, 23(3), 142-155.",
]

THRESHOLDS = [
    "GOF/LOF functional assay MANDATORY before memantine, ketamine, or D-cycloserine — result in chart",
    "POLG testing MANDATORY before VPA — biallelic LOF + VPA = Alpers crisis",
    "ACTH day 14 EEG — assess hypsarrhythmia resolution; if no response → KD + memantine (GOF confirmed)",
    "VPA TDM target 50-100 µg/mL; LFT + ammonia q3M; ammonia >80 µmol/L → L-carnitine supplementation",
    "ERG q6M (VGB SHARE/REMS) — discontinue if >20% amplitude reduction; Goldman VF q6M (>5y)",
    "KD β-OHB target 2-4 mmol/L; <2 = insufficient ketosis; >5 = over-ketotic → adjust and hydrate",
    "CSWS/ESES: if >85% NREM occupied by spike-wave → steroid/IVIG referral + speech-language therapy",
    "Memantine: GOF only; start 0.3 mg/kg/day; escalate over 4 weeks; assess at 3 months",
    "SUDEP counselling at first visit; annual SUDEP risk assessment; wearable seizure sensor prescribed",
    "2 AED failures → video-EEG + KD trial; 3 AED failures + focal MRI lesion → presurgical evaluation",
]


# ──────────────────────────────────────────────────────────────────────────────
# Patient cohort generation
# ──────────────────────────────────────────────────────────────────────────────
def _make_patients():
    cats = [
        ("GRIN2B-GOF-severe-DEE27-West", 37, True),
        ("GRIN2B-GOF-moderate-missense", 18, True),
        ("GRIN2B-LOF-truncating-frameshift", 27, False),
        ("GRIN2B-LOF-partial-missense", 13, False),
        ("GRIN2B-phenocopy", 5, False),
    ]
    patients = []
    pid = 1
    for cat, pct, is_gof in cats:
        n = round(41 * pct / 100)
        for _ in range(n):
            onset = random.randint(1, 6) if is_gof else random.randint(6, 24)
            age = random.randint(onset + 6, onset + 60)
            west = is_gof and random.random() < 0.82
            myoclonic = random.random() < (0.65 if is_gof else 0.35)
            dre = random.random() < (0.72 if is_gof else 0.38)
            on_kd = dre and random.random() < 0.68
            memantine = is_gof and random.random() < 0.55
            polg = "Y" if random.random() < 0.92 else "N"
            csws = not is_gof and random.random() < 0.52
            patients.append({
                "id": f"GRIN2B-{pid:03d}",
                "category": cat,
                "gof_lof": "GOF" if is_gof else "LOF",
                "onset_months": onset,
                "age_months": age,
                "west_syndrome": west,
                "myoclonic_seizures": myoclonic,
                "drug_resistant": dre,
                "on_kd": on_kd,
                "memantine_rx": memantine,
                "csws_eses": csws,
                "polg_tested": polg,
            })
            pid += 1
    return patients[:41]


PATIENTS = _make_patients()


# ──────────────────────────────────────────────────────────────────────────────
# API functions
# ──────────────────────────────────────────────────────────────────────────────

def get_overview():
    gof_n = sum(1 for p in PATIENTS if p["gof_lof"] == "GOF")
    lof_n = sum(1 for p in PATIENTS if p["gof_lof"] == "LOF")
    west_n = sum(1 for p in PATIENTS if p.get("west_syndrome"))
    myoc_n = sum(1 for p in PATIENTS if p.get("myoclonic_seizures"))
    dre_n = sum(1 for p in PATIENTS if p.get("drug_resistant"))
    kd_n = sum(1 for p in PATIENTS if p.get("on_kd"))
    mem_n = sum(1 for p in PATIENTS if p.get("memantine_rx"))
    polg_n = sum(1 for p in PATIENTS if p.get("polg_tested") == "Y")
    csws_n = sum(1 for p in PATIENTS if p.get("csws_eses"))
    avg_onset = round(sum(p["onset_months"] for p in PATIENTS) / len(PATIENTS), 1)

    return {
        "gene": "GRIN2B",
        "locus": "12p12.1",
        "inheritance": "De novo dominant (predominantly); rarely familial AD",
        "protein": "GluN2B (NR2B) — NMDA receptor subunit 2B; obligate component of foetal diheteromeric + mature triheteromeric NMDA receptors",
        "mechanism": (
            "GOF (55%): constitutive/excessive NMDA channel opening → excess Ca2+ → excitotoxicity → "
            "West/Ohtahara/severe DEE27 (memantine candidate). "
            "LOF (40%): NMDA hypofunction → impaired PV-interneuron drive → disinhibition-hyperexcitability + "
            "NDD + EAS (memantine ABSOLUTE CI)."
        ),
        "key_aha": (
            "GOF/LOF assay MANDATORY before memantine or ketamine — memantine TREATS GOF but WORSENS LOF. "
            "POLG exclusion before VPA. ACTH day-14 EEG gates escalation to KD+memantine. "
            "Tiagabine and Na-blockers (CBZ/OXC/PHT) contraindicated in generalised phenotype."
        ),
        "n_patients": 41,
        "gof_pct": round(gof_n / 41 * 100),
        "lof_pct": round(lof_n / 41 * 100),
        "west_syndrome_pct": round(west_n / 41 * 100),
        "myoclonic_pct": round(myoc_n / 41 * 100),
        "drug_resistant_pct": round(dre_n / 41 * 100),
        "on_kd_pct": round(kd_n / 41 * 100),
        "memantine_rx_pct": round(mem_n / 41 * 100),
        "polg_done_pct": round(polg_n / 41 * 100),
        "csws_eses_pct": round(csws_n / 41 * 100),
        "avg_onset_months": avg_onset,
        "contraindications_summary": [
            "Memantine in LOF GRIN2B — ABSOLUTE CI (worsens NMDA hypofunction → worse seizures + NDD)",
            "Ketamine IV in LOF GRIN2B — HIGH RISK (NMDA block → LOF-disinhibition worsening)",
            "CBZ / OXC / PHT in West/generalised — HIGH RISK (NaV1.1 interneuron block → aggravation)",
            "Tiagabine — ABSOLUTE CI in DEE27 (NCSE in dysmature cortex)",
            "Vigabatrin long-term (non-West) — HIGH RISK (irreversible retinal toxicity)",
        ],
        "thresholds": THRESHOLDS,
        "references": [r.split(".")[0] + " et al." for r in REFERENCES],
    }


def get_breakdown():
    gof_n = sum(1 for p in PATIENTS if p["gof_lof"] == "GOF")
    lof_n = sum(1 for p in PATIENTS if p["gof_lof"] == "LOF")
    west_n = sum(1 for p in PATIENTS if p.get("west_syndrome"))
    myoc_n = sum(1 for p in PATIENTS if p.get("myoclonic_seizures"))
    dre_n = sum(1 for p in PATIENTS if p.get("drug_resistant"))
    kd_n = sum(1 for p in PATIENTS if p.get("on_kd"))
    mem_n = sum(1 for p in PATIENTS if p.get("memantine_rx"))
    polg_n = sum(1 for p in PATIENTS if p.get("polg_tested") == "N")  # not tested = risk
    csws_n = sum(1 for p in PATIENTS if p.get("csws_eses"))

    return {
        "summary": {
            "n_patients": 41,
            "gof_pct": round(gof_n / 41 * 100),
            "lof_pct": round(lof_n / 41 * 100),
            "west_pct": round(west_n / 41 * 100),
            "myoclonic_pct": round(myoc_n / 41 * 100),
            "drug_resistant_pct": round(dre_n / 41 * 100),
            "kd_pct": round(kd_n / 41 * 100),
            "memantine_rx_pct": round(mem_n / 41 * 100),
            "polg_not_tested_count": polg_n,
            "csws_eses_pct": round(csws_n / 41 * 100),
        },
        "etiology_distribution": ETIOLOGY_CATALOG,
        "patients_sample": PATIENTS,
        "seizure_types": [
            {"type": s["type"], "prevalence_pct": s["prevalence_pct"]}
            for s in SEIZURE_TYPES
        ],
        "seizure_detail": SEIZURE_TYPES,
        "triggers": [
            {"trigger": t["trigger"], "prevalence_pct": t["prevalence_pct"]}
            for t in TRIGGERS
        ],
        "trigger_detail": TRIGGERS,
        "treatment_detail": TREATMENTS,
        "contraindication_detail": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
    }


def get_definitions():
    return {
        "concepts": CONCEPTS,
        "contraindications_full": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
