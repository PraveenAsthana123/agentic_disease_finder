"""
FOXG1 Syndrome — Congenital Rett Variant / FOXG1-Related Disorder / DEE
=========================================================================
41-patient cohort · FOXG1 (14q12) · X-linked dominant / de novo · Congenital onset DEE

FOXG1 BIOLOGY:
FOXG1 (Forkhead Box G1, 14q12) encodes a brain-specific transcriptional repressor and
activator critical for forebrain development. FOXG1 protein contains:
  (1) Forkhead DNA-binding domain (FHD) — binds FOXO/FKHR consensus sequences
  (2) N-terminal repressor domain — recruits Groucho/TLE co-repressors and HDAC
  (3) C-terminal transactivation domain — interacts with JARID1B/KDM5B
  (4) CREB-binding protein (CBP) interaction motif

FOXG1 is expressed from embryonic day E9 (mouse) in the telencephalon — the earliest
transcription factor marking the prospective cerebral cortex. FOXG1 governs:
  (a) Progenitor cell cycle exit timing (FOXG1 LOF → premature exit → fewer neurons)
  (b) Cortical layering (LOF shifts laminar composition toward early-born deep-layer neurons)
  (c) Interneuron migration (FOXG1 required for MGE-derived PV+ and SST+ interneuron
      tangential migration into cortex — LOF causes interneuron deficit → E/I imbalance)
  (d) Olfactory bulb/hippocampal development (FOXG1 activates IGF1/BDNF signalling)
  (e) Postsynaptic AMPA receptor trafficking (FOXG1 regulates GluA1 endocytosis via
      PREX1 → LOF reduces synaptic AMPA density → reduced excitatory transmission,
      seemingly paradoxically worsening E/I balance via interneuron deficit)

FOXG1 SYNDROME vs CLASSIC RETT:
FOXG1 Syndrome is DISTINCT from Classic Rett Syndrome (RTT, caused by MECP2 mutations):
  Classic RTT (MECP2):
    - 6-18 months normal development THEN regression
    - Girls predominantly
    - Hand stereotypies (hallmark — absent in FOXG1)
    - Autonomic instability (breathing irregularities)
    - Preserved ambulation early
  FOXG1 Syndrome (Congenital RTT):
    - CONGENITAL or early infantile onset (no normal period → no regression)
    - Equal sex ratio (X-linked dominant, de novo in most; males equally affected)
    - Hyperkinetic DYSKINESIAS (not stereotypies): continuous choreoathetosis,
      stereotyped hand-wringing is ABSENT; instead dyskinetic hand movements
    - Frontal-lobe hypoplasia on MRI (FOXG1 is a forebrain transcription factor)
    - Profound hypotonia then hyperkinesis
    - DYSOSMIA (reduced olfaction — olfactory bulb underdevelopment)
    - Severely limited ambulation (most never walk independently)

FOXG1 GENOTYPE-PHENOTYPE:
  SEVERE (LOF — truncating, whole-gene deletion):
    - Profound hypotonia, absent language, no sitting, DLB/vegetative
    - Onset seizures 3-6 months
    - Severe microcephaly (OFC -3 to -5 SD)
  MODERATE (missense — FHD domain):
    - Some head control, occasional 1-2 words then loss
    - Onset seizures 6-12 months
    - Mild-moderate microcephaly (OFC -2 to -3 SD)
  ATYPICAL / MILD (missense — outside FHD, or C-terminal):
    - Limited but present motor skills
    - Later seizure onset (up to 24 months)
    - Near-normal OFC

CONTRAINDICATIONS IN FOXG1 SYNDROME:
  1. PHENOBARBITONE AS SOLE AGENT: PB reduces cortical GABA-A activity long-term via
     receptor downregulation; in already GABA-impaired FOXG1 cortex → worsens hyperkinesia
     and may increase seizure burden paradoxically. PB may be used short-term in neonates/
     infants for acute SE, but avoid chronic monotherapy in FOXG1.
  2. CBZ/OXC: Sodium channel blockers worsen the myoclonic/generalised components seen in
     some FOXG1 patients. FOXG1 seizures are often multifocal generalised; NaV blockers
     (CBZ, OXC, PHT) have poor efficacy and risk exacerbating myoclonus.
  3. TIAGABINE: GAT-1 inhibitor → increases synaptic GABA → can induce non-convulsive
     SE in patients with diffuse cortical dysmaturation (including FOXG1). AVOID.
  4. VIGABATRIN (long-term): irreversible retinal toxicity risk; FOG1 patients have limited
     visual function already (cortical visual impairment is common in FOXG1 with frontal/
     occipital involvement). ERG mandatory if using. Usually short-term West only.
  5. KETAMINE (avoid in dyskinesia exacerbation): NMDA antagonism may worsen hyperkinetic
     dyskinesias acutely. Use midazolam/lorazepam preferentially for acute SE.

PRECISION TREATMENTS:
  - Valproate (VPA): broad-spectrum GABA-enhancer + Na-channel modulator; POLG exclusion
    MANDATORY before use (FOXG1 patients sometimes carry POLG variants on background).
  - ACTH/Prednisolone: for West syndrome component (infantile spasms) — same first-line
    indication as other DEE with West. ACTH promotes GABA-A β3/γ2 subunit surface expression.
  - KD (Ketogenic Diet): bypasses impaired inhibitory network via β-OHB/acetone/KATP
    mechanisms; Level B evidence across multiple DEE types; 40-60% responder rate in FOXG1.
  - Baclofen: for dyskinesias / hyperkinesias — GABA-B agonist → reduces thalamo-cortical
    hyperkinetic output; useful adjunct but does NOT reduce seizures.
  - Triheptanoin (investigational): odd-chain fat → anaplerotic TCA cycle substrate;
    rationale in FOXG1 for metabolic support of energy-deficient cortical neurons.
"""

import random, math
random.seed(42)

# ── Etiology catalog ──────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "FOXG1-de-novo-truncating-LOF-classic",
        "pct": 42,
        "etiology": "De novo FOXG1 truncating / LOF (classic congenital)",
        "mechanism": (
            "Frameshift, nonsense, or splice-site de novo FOXG1 variants (14q12) causing "
            "haploinsufficiency of the forkhead box G1 transcription factor. Loss of one "
            "functional FOXG1 allele → 50% reduction in forebrain FOXG1 protein during "
            "critical embryonic neurogenesis → premature progenitor cell cycle exit → "
            "reduced cortical neuron number + interneuron migration defect → E/I imbalance "
            "→ severe congenital DEE. Classic phenotype: profound hypotonia, absent language, "
            "congenital microcephaly, severe cortical visual impairment, seizures 3-6 months."
        ),
        "eeg_signature": (
            "High-amplitude multifocal spikes + slow waves; burst-suppression or theta-delta "
            "predominance; hypsarrhythmia in West phase; diffuse slowing with loss of sleep "
            "architecture; photoparoxysmal response uncommon. Background increasingly attenuated "
            "with disease progression."
        ),
    },
    {
        "category": "FOXG1-de-novo-missense-FHD-domain",
        "pct": 29,
        "etiology": "De novo FOXG1 missense — forkhead domain (FHD)",
        "mechanism": (
            "Missense variants within the forkhead DNA-binding domain (exon 1, residues ~176-272) "
            "disrupting FOXG1 DNA binding. FHD variants cause dominant-negative or LOF depending "
            "on position: variants at the recognition helix (H3, W294) abolish DNA binding; "
            "variants near the wing (W2) reduce nuclear localisation. Phenotype: moderate-severe "
            "DEE; some patients achieve limited head control; OFC -2 to -3 SD; seizures 6-12 months."
        ),
        "eeg_signature": (
            "Multifocal epileptiform discharges (frontal predominant reflecting FOXG1 forebrain "
            "expression); modified hypsarrhythmia; theta-alpha background in wakefulness; "
            "NREM slow-wave abnormalities with sleep spindle reduction."
        ),
    },
    {
        "category": "FOXG1-whole-gene-deletion-14q12",
        "pct": 15,
        "etiology": "14q12 microdeletion (FOXG1 whole-gene deletion)",
        "mechanism": (
            "Chromosomal microdeletion spanning 14q12 encompassing FOXG1 ± neighbouring genes "
            "(PRKD1, NOVA1, NOVA2 sometimes deleted in larger deletions). NOVA1/NOVA2 encode "
            "splicing factors for neurexin/GRIN2 pre-mRNA — larger deletions worsen phenotype. "
            "Array CGH or chromosomal microarray (CMA) required for detection. FISH and karyotype "
            "insufficient for small 14q12 deletions (<500 kb). Most deletions are de novo."
        ),
        "eeg_signature": (
            "Often most severe EEG pattern: burst-suppression in neonatal period evolving to "
            "high-amplitude multifocal discharges; hypsarrhythmia or modified hyps; prolonged "
            "electrographic seizures with subtle clinical correlate."
        ),
    },
    {
        "category": "FOXG1-de-novo-missense-outside-FHD",
        "pct": 9,
        "etiology": "De novo FOXG1 missense — outside forkhead domain (C-terminal / N-terminal)",
        "mechanism": (
            "Missense variants in FOXG1 N-terminal repressor domain or C-terminal transactivation "
            "domain. N-terminal variants disrupt TLE/Groucho co-repressor binding → partial LOF "
            "of transcriptional repression. C-terminal variants impair CBP interaction → reduced "
            "CREB-dependent activation. Generally milder phenotype: some motor milestones achieved; "
            "later seizure onset; near-normal OFC; better language trajectory (occasional words)."
        ),
        "eeg_signature": (
            "Focal frontal or multifocal spikes; normal or near-normal background possible early; "
            "sleep EEG shows focal slow-wave discharges. Better background preservation correlates "
            "with milder phenotype. IED density lower than classic truncating cases."
        ),
    },
    {
        "category": "FOXG1-negative-phenocopy",
        "pct": 5,
        "etiology": "Clinical FOXG1 phenocopy (FOXG1-negative)",
        "mechanism": (
            "Patients with FOXG1 syndrome clinical features (congenital DEE + dyskinesias + "
            "frontal hypoplasia + severe ID) but negative FOXG1 sequencing + CMA. Consider: "
            "CNTNAP2-CDFE (cortical dysplasia-focal epilepsy), MEF2C haploinsufficiency "
            "(5q14.3), ARX mutations (X-linked — boys predominant), WDR45 (BPAN — progressive), "
            "CASK mutations, or copy number variants at other loci. Whole-exome sequencing (WES) "
            "recommended as next step to identify alternative diagnosis."
        ),
        "eeg_signature": (
            "Similar multifocal pattern; clinical and EEG features overlap with FOXG1 syndrome; "
            "distinguishing features depend on alternative diagnosis. MEF2C: often multifocal + "
            "hyperkinesias very similar to FOXG1; ARX: often asymmetric focal onset; WDR45: "
            "progressive EEG deterioration with iron-deposition MRI changes."
        ),
    },
]

# ── Seizure types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Epileptic Spasms (West Syndrome / Infantile Spasms)",
        "prevalence_pct": 82,
        "semiology": (
            "Clusters of brief (0.5-2 s) flexion or flexion-extension spasms, often on awakening. "
            "Clusters of 5-30 spasms separated by 5-30 s inter-spasm interval. Subtle in FOXG1 "
            "due to hypotonia — may appear as head drops or truncal quiver. Onset 3-9 months."
        ),
        "eeg_pattern": (
            "Hypsarrhythmia (high-amplitude chaotic multifocal spikes + SW between spasms; "
            "modified hypsarrhythmia common in FOXG1). Each spasm: high-amplitude slow wave "
            "followed by EEG attenuation (electrodecremental response). Post-spasm suppression "
            "5-10 s."
        ),
        "clinical_tip": (
            "ACTH + vigabatrin first-line for West. FOXG1-associated West has LOWER response to "
            "ACTH (~40-50%) vs idiopathic West (70-80%) — monitor for incomplete response and "
            "escalate promptly. VGB SHARE REMS required; ERG baseline mandatory. "
            "KD as second-line if poor ACTH response."
        ),
    },
    {
        "type": "Focal Motor Seizures (Frontal-Onset)",
        "prevalence_pct": 74,
        "semiology": (
            "Hypermotor frontal lobe seizures or focal clonic activity (limb / hemiface); often "
            "nocturnal clustering. Brief (10-60 s). Secondary generalisation in ~40%. Dyskinesias "
            "may intensify peri-ictally — can be difficult to distinguish from seizure vs "
            "dyskinetic movement. Video-EEG correlation essential."
        ),
        "eeg_pattern": (
            "Frontal-maximal spike/polyspike discharge at seizure onset; bilateral spread rapid. "
            "Ictal: rhythmic theta/alpha frequency discharge evolving to fast activity. "
            "Post-ictal: focal slowing ipsilateral frontal. Sleep-wake: most frequent at "
            "sleep-wake transitions (frontal lobe hypermotor pattern)."
        ),
        "clinical_tip": (
            "VPA first-line for focal motor in FOXG1. CBZ/OXC are NOT effective and risk "
            "worsening myoclonic components. If frontal-motor nocturnal: add CLB (clobazam). "
            "Lacosamide (LCM) may have adjunct role via Nav1.7 slow-inactivation mechanism "
            "without worsening myoclonus."
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "prevalence_pct": 61,
        "semiology": (
            "Brief, shock-like involuntary muscle jerks — axial (head/trunk) or multifocal "
            "limbs. Often occur in clusters, especially morning on awakening. Can be provoked "
            "by sudden sound or tactile stimuli (stimulus-sensitive myoclonus). DISTINCT from "
            "the continuous background dyskinesias of FOXG1 (which are choreoathetoid, not "
            "shock-like)."
        ),
        "eeg_pattern": (
            "Generalised polyspike-and-wave discharge (< 3 Hz, often 2-2.5 Hz in FOXG1 vs "
            "classical 3 Hz). Polyspike correlates with myoclonic jerk on EMG. Background "
            "slowing between bursts. Photosensitivity (PPR) in ~20% of FOXG1 myoclonic cases."
        ),
        "clinical_tip": (
            "VPA + ETH (ethosuximide for generalised polyspike component) effective combination. "
            "AVOID CBZ/OXC/PHT — worsen myoclonus. LEV may help via SV2A mechanism but can "
            "paradoxically worsen irritability/behavioural dysregulation in FOXG1 (GABA deficit "
            "hypothesis). Pyridoxine trial in young infants (<18 months) if uncertain about PDE."
        ),
    },
    {
        "type": "GTCS / Tonic-Clonic Seizures",
        "prevalence_pct": 48,
        "semiology": (
            "Generalised tonic-clonic: tonic phase 10-30 s → clonic 30-60 s → post-ictal "
            "exhaustion. Often evolution from focal (frontal) with rapid bilateral spread. "
            "Nocturnal predominance. In FOXG1: GTCS may have very brief tonic phase due to "
            "hypotonia affecting tonic posturing. Distinguishing from spasms requires video-EEG."
        ),
        "eeg_pattern": (
            "Tonic phase: rapid (18-25 Hz) generalised recruiting rhythm, amplitude increment. "
            "Clonic phase: rhythmic spike-wave, decelerating frequency. Post-ictal: diffuse "
            "suppression 10-60 s."
        ),
        "clinical_tip": (
            "VPA primary. Add CLB or LEV as adjunct. Rescue medication: midazolam buccal 0.3 mg/kg "
            "or intranasal. AVOID KETAMINE for acute SE in FOXG1 dyskinetic patients — NMDA "
            "blockade may worsen dyskinesias. Prefer midazolam/lorazepam for SE. "
            "Levetiracetam IV 40-60 mg/kg for SE second-line."
        ),
    },
]

# ── Triggers ──────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Fever / Illness", "prevalence_pct": 88, "mechanism": "Fever increases neuronal excitability via temperature-dependent ion channel kinetics; FOXG1 interneuron deficit worsens thermoregulatory seizure threshold. Unlike HCN1/Dravet, FOXG1 fever threshold is ~38.5°C (standard). Aggressive fever management mandatory."},
    {"trigger": "Sleep-wake transitions", "prevalence_pct": 79, "mechanism": "Frontal hypermotor seizures cluster at NREM-wake transitions; reduced sleep spindle density in FOXG1 (reduced thalamo-cortical inhibition) fails to protect against arousal-triggered discharges."},
    {"trigger": "Missed AED dose", "prevalence_pct": 71, "mechanism": "VPA and CLB discontinuation → rapid rebound. VPA half-life 12-20h in children on polytherapy — missing one dose can drop levels 40-50% before next dose. Extended-release formulations reduce trough dipping."},
    {"trigger": "Overstimulation / Sensory overload", "prevalence_pct": 64, "mechanism": "FOXG1 cortical dysmaturation → impaired sensory gating; bright lights/loud sounds may trigger myoclonic jerks (stimulus-sensitive myoclonus) and cluster focal seizures. Dark, quiet environments during breakthrough periods."},
    {"trigger": "Sleep deprivation", "prevalence_pct": 58, "mechanism": "FOXG1 patients often have severely disturbed sleep (cortical dysmaturation, dyskinesias). Sleep deprivation worsens seizure threshold; melatonin supplementation (0.5-3 mg) helps circadian regulation."},
    {"trigger": "Constipation / GI distress", "prevalence_pct": 45, "mechanism": "VPA-induced constipation is common; constipation in FOXG1 (gut dyskinesias, reduced mobility) → altered VPA absorption kinetics → drug level fluctuations. Regular bowel regimen essential."},
    {"trigger": "Intercurrent infection (non-febrile)", "prevalence_pct": 38, "mechanism": "Metabolic stress of infection (even without fever) can lower seizure threshold in FOXG1 via cytokine-mediated neuroexcitability (IL-1β, TNF-α increase NMDA receptor conductance). Sick-day plan: do NOT reduce AEDs during illness."},
    {"trigger": "Drug interactions / CYP450", "prevalence_pct": 28, "mechanism": "VPA is a CYP2C9/UGT enzyme inhibitor; adding CLB (CYP3A4 substrate) → N-desmethyl-CLB accumulation → paradoxical overdose/sedation. Monitor CLB levels when starting VPA. VPA-CLB combination requires dose adjustment."},
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproate (VPA) / Sodium Valproate",
        "level": "Level B — first-line broad-spectrum (both seizure types)",
        "dose": "20-60 mg/kg/day PO in 2-3 divided doses; target TDM 50-100 µg/mL (total VPA); extended-release preferred to reduce troughs.",
        "moa": "Multi-modal: Na-channel inactivation, GABA transaminase inhibition (↑ synaptic GABA), T-type Ca2+ channel block (↓ thalamo-cortical oscillations), HDAC inhibition (epigenetic). Uniquely useful in FOXG1 due to broad-spectrum coverage of focal + generalised + myoclonic components.",
        "efficacy": "40-60% ≥50% seizure reduction in FOXG1-associated DEE. Best evidence for generalised + myoclonic components. Less effective for pure focal-motor if frontal-onset dominant.",
        "safety": "POLG EXCLUSION MANDATORY before VPA — FOXG1 patients may carry POLG variants; VPA-induced hepatotoxicity fatal in POLG+VPA. Check POLG sequencing and LFTs at baseline. VPA + CLB: monitor N-CLB accumulation. VPA teratogenicity: MANDATORY sodium valproate pregnancy counselling (MHRA PREVENT, UK). Neural tube defects 1-2%, cognitive teratogenicity.",
        "monitoring": "VPA TDM q3M; LFT + ammonia q3M; FBC (thrombocytopenia); weight gain; tremor; POLG before initiation.",
        "foxg1_specific": "FOXG1 patients often have feeding difficulties → VPA enteral administration via G-tube using liquid formulation (not sprinkle capsules via NGT due to absorption variability).",
    },
    {
        "drug": "ACTH / Prednisolone (for West Syndrome component)",
        "level": "Level A — West syndrome (infantile spasms) first-line",
        "dose": "ACTH: synthetic ACTH (Synacthen) 0.5-1 mg/day IM, or natural ACTH 40-80 IU/day IM × 2 weeks then taper. Prednisolone: 4 mg/kg/day PO (max 40 mg) × 14 days then taper.",
        "moa": "ACTH → adrenocortical steroid production → reduces CRH levels (CRH is proconvulsant) + promotes GABA-A β3/γ2 subunit surface expression via PKA phosphorylation → improved synaptic inhibition. Also anti-inflammatory (cytokine suppression).",
        "efficacy": "Spasm-free rate ~40-50% in FOXG1-associated West vs ~70% in cryptogenic West. FOXG1 West often requires combination ACTH+VGB or early KD escalation. UK INFANT trial: ACTH superior to VGB for spasm cessation at 14 days.",
        "safety": "Hypertension, hyperglycaemia, GI bleeding, infection risk (PCP prophylaxis not routinely required in short courses), adrenal suppression on tapering. Daily BP monitoring during ACTH course.",
        "monitoring": "BP daily; BM qid; weight daily; electrolytes weekly; EEG at day 14 to assess response; ophthalmology if >4 weeks (cataracts).",
        "foxg1_specific": "Response assessment: EEG + clinical at day 14. If incomplete: ADD vigabatrin (SHARE REMS, ERG baseline) or ESCALATE to KD. Do NOT extend ACTH beyond 4 weeks without clear EEG evidence of continued benefit.",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "Level A — West syndrome first-line (with ACTH); Level C elsewhere",
        "dose": "100-150 mg/kg/day PO in 2 divided doses for infantile spasms. Taper once spasm-free. Avoid chronic use beyond West syndrome phase due to retinal toxicity.",
        "moa": "Irreversible GABA transaminase inhibitor → ↑ synaptic GABA. Does NOT affect FOXG1 transcription directly. Effective for West syndrome in FOXG1 by enhancing inhibitory tone in an E/I-imbalanced cortex.",
        "efficacy": "Spasm cessation 40-60% in symptomatic West (including FOXG1). Often combined with ACTH (UKISS trial: ACTH+VGB superior to ACTH alone for spasm cessation speed and EEG normalisation).",
        "safety": "IRREVERSIBLE PERIPHERAL RETINAL TOXICITY — cumulative dose dependent. FDA SHARE REMS mandatory. ERG baseline + q6M. In FOXG1: cortical visual impairment already present → VGB retinal damage additive. Use for short-term West treatment only; taper once spasms controlled.",
        "monitoring": "ERG q6M (SHARE REMS protocol); visual field testing (when feasible); VGB not recommended beyond 6-12 months due to retinal risk.",
        "foxg1_specific": "FOXG1 patients have limited reliable visual function assessment (cortical visual impairment). ERG is objective and MANDATORY. Discuss retinal risk with family before initiating. Minimum effective dose, minimum duration.",
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B — adjunct for focal-motor and generalised components",
        "dose": "0.1-0.3 mg/kg/day PO in 2 divided doses; max 40 mg/day. Start low (0.05 mg/kg/day) in FOXG1 with hypotonia (respiratory risk).",
        "moa": "Selective GABA-A α2/α3 subunit positive allosteric modulator (PAM) → prefrontal/limbic GABA-A potentiation with less α1 (sedation/tolerance). Active metabolite N-desmethyl-CLB (N-CLB) 20× more potent — accumulates with VPA co-administration.",
        "efficacy": "40-60% ≥50% seizure reduction as add-on in mixed-seizure DEE. Particularly effective for focal motor and nocturnal hypermotor components.",
        "safety": "Sedation (dose-dependent); tolerance develops over months; respiratory depression risk in hypotonic FOXG1 patients — START LOW. VPA co-admin → N-CLB accumulation → over-sedation. Monitor CLB levels and N-CLB if available. Risk of withdrawal seizures on rapid taper.",
        "monitoring": "CLB + N-CLB TDM if clinical over-sedation; respiratory rate monitoring in hypotonic patients; taper slowly (10-25% per week) to avoid withdrawal.",
        "foxg1_specific": "VPA + CLB most common combination in FOXG1. Ensure N-CLB measured if sedation occurs — often overlooked cause of excessive sedation.",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B — for drug-resistant epilepsy (DRE) or West non-response",
        "dose": "Classical 4:1 or 3:1 fat:carbohydrate+protein ratio. MAD (Modified Atkins Diet) in older/more mobile FOXG1 patients. Target β-OHB 2-4 mmol/L. G-tube feeding common in FOXG1 — use KD formula (KetoCal).",
        "moa": "β-hydroxybutyrate (β-OHB) → multiple anti-seizure mechanisms: KATP channel opening (hyperpolarises neurons), direct GABA-B receptor activation, inhibition of AMPA receptor-mediated excitation, reduction of reactive oxygen species. Bypasses impaired GABAergic synaptic inhibition in FOXG1.",
        "efficacy": "40-60% ≥50% seizure reduction in FOXG1-DEE (similar to other DEE types). KD particularly effective for spasms not controlled by ACTH+VGB. Dyskinesias may also partially improve on KD (mechanism unclear; possibly β-OHB basal ganglia effect).",
        "safety": "Hypoglycaemia (especially at initiation); dyslipidaemia; growth restriction; nephrolithiasis (prophylactic K-citrate 1-2 mEq/kg/day); selenium/carnitine/zinc deficiency. G-tube placement often needed in FOXG1 for reliable enteral feeding.",
        "monitoring": "β-OHB daily (2-4 mmol/L target); lipids q3M; growth z-score q3M; selenium, carnitine, zinc, calcium q6M; DXA annually; renal USS if urinary calcium elevated.",
        "foxg1_specific": "G-tube feeding is common in FOXG1 (severe oromotor dysfunction). KetoCal 4:1 via G-tube is well-tolerated. Ensure gastrostomy button is appropriate size — FOXG1 dyskinesias can dislodge PEG tubes.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level C — adjunct (caution: behavioural side effects in FOXG1)",
        "dose": "20-60 mg/kg/day PO/IV in 2 divided doses. IV LEV 40-60 mg/kg for acute SE.",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulator → reduces synaptic vesicle exocytosis from over-active synapses. GABA-A receptor modulation (indirect). Hepatically independent.",
        "efficacy": "Modest adjunct effect in FOXG1-DEE. Useful for acute SE (IV formulation). Less effective as chronic anti-seizure monotherapy. IV LEV valuable in POLG patients (hepatically safe alternative when VPA contraindicated — though POLG exclusion done before VPA start).",
        "safety": "BEHAVIOURAL TOXICITY in FOXG1: irritability, agitation, aggression in 20-30% of children with pre-existing neurodevelopmental disorders. FOXG1 patients have limited communication — caregiver-reported irritability/distress increase may be the only signal. Consider LEV-associated behavioural worsening if irritability increases within 4 weeks of starting.",
        "monitoring": "Caregiver behavioural diary; renal function (dose adjust in renal impairment); no TDM routinely needed (level-effect correlation poor).",
        "foxg1_specific": "LEV behavioural worsening more common in ASD/ID/FOXG1. Supplement with pyridoxine 50 mg/day co-administration (may reduce LEV-associated irritability — anecdotal but low-risk). Stop if clear behavioural deterioration.",
    },
    {
        "drug": "Fenfluramine (FFA / Fintepla)",
        "level": "Level C — investigational / compassionate use in refractory FOXG1",
        "dose": "0.1-0.35 mg/kg/day PO (max 26 mg/day); FDA-approved for Dravet and LGS. Off-label FOXG1 via compassionate use / REMS program (REMS mandatory: echocardiography q6M).",
        "moa": "Serotonin releasing agent (5-HT2C agonist) → activates sigma-1 receptor → modulates voltage-gated Na+ and K+ channels; indirect GABA potentiation via serotonergic circuitry. Mechanism partially overlaps with FOXG1's interneuron deficit (5-HT2C receptors prominent on GABAergic interneurons).",
        "efficacy": "Limited FOXG1-specific data; case reports suggest 50% reduction in seizure frequency in some refractory FOXG1 patients. Clinical trials underway in FOXG1 network.",
        "safety": "CARDIAC VALVULOPATHY and PULMONARY ARTERIAL HYPERTENSION (PAH) — REMS mandatory. Echocardiogram baseline + q6M. If TR velocity >2.8 m/s or any new valvular regurgitation: STOP. Appetite suppression / weight loss — significant concern in already-malnourished FOXG1 patients. Monitor weight monthly.",
        "monitoring": "Echocardiogram q6M (REMS); weight monthly; appetite assessment; QTc ECG at baseline and after dose changes.",
        "foxg1_specific": "Weight loss from FFA is a serious concern in FOXG1 — patients often already nutritionally vulnerable. Gastrostomy feeding + dietitian involvement essential before starting FFA. Pre-treatment weight Z-score must be documented.",
    },
    {
        "drug": "Baclofen (for dyskinesias — NOT seizures)",
        "level": "Level C — symptom management (hyperkinetic dyskinesias)",
        "dose": "2.5-5 mg PO TID initially; titrate to 10-20 mg TID. Intrathecal baclofen (ITB) pump for severe refractory dyskinesias unresponsive to oral baclofen.",
        "moa": "GABA-B agonist → presynaptic inhibition of excitatory neurotransmitter release from basal ganglia circuitry → reduces thalamo-cortical hyperkinetic output. Does NOT directly reduce seizures but improves quality of life and reduces injury from dyskinesias.",
        "efficacy": "Dyskinesia reduction in 50-70% of FOXG1 patients. Oral baclofen partially effective; ITB pump more effective for severe continuous dyskinesias. ITB requires general anaesthesia for pump implantation — consider developmental stage and anaesthetic risk.",
        "safety": "Baclofen WITHDRAWAL SYNDROME: potentially life-threatening — fever, rhabdomyolysis, seizures, dyskinesia storm. Never abruptly stop (especially ITB pump failure). Sedation, hypotonia worsening, respiratory depression. Drug interactions: CNS depressants (CLB, VPA sedation) additive.",
        "monitoring": "Baclofen dose titration diary (dyskinesia severity rating); FOXG1 Dyskinesia Scale (FDS) if available; baclofen withdrawal protocol for sick days / pump changes.",
        "foxg1_specific": "Dyskinesias are one of the most disabling aspects of FOXG1 syndrome for caregivers. Baclofen is often the most impactful quality-of-life intervention. Consider multidisciplinary dyskinesia clinic (neurology + movement disorder specialist).",
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine / Oxcarbazepine / Phenytoin (NaV blockers)",
        "risk": "HIGH",
        "reason": (
            "Voltage-gated Na-channel blockers have poor efficacy in FOXG1 generalised and myoclonic "
            "seizure components and risk WORSENING myoclonus and spasms. FOXG1 seizures are "
            "predominantly generalised/multifocal — NaV blockers are effective only for focal "
            "seizures. Additionally: CBZ/OXC are CYP3A4 inducers → increase VPA metabolism → "
            "lower VPA levels → seizure breakthrough. HLA-B*15:02 mandatory before CBZ/OXC "
            "(Steven-Johnson syndrome risk, CPIC Level A recommendation)."
        ),
    },
    {
        "drug": "Phenobarbitone (PB) — chronic monotherapy",
        "risk": "MODERATE-HIGH",
        "reason": (
            "Chronic PB → GABA-A receptor downregulation (subunit composition change: α4/δ subunits "
            "replace α1 subunits → reduced BZD sensitivity, altered GABA-A kinetics). In FOXG1 where "
            "GABA-A inhibitory network is already impaired, PB long-term monotherapy may paradoxically "
            "worsen seizure threshold over months. PB acceptable SHORT-TERM: neonatal/infantile SE, "
            "acute seizure clusters when IV VPA unavailable. AVOID as maintenance monotherapy in FOXG1."
        ),
    },
    {
        "drug": "Tiagabine (TGB)",
        "risk": "HIGH — ABSOLUTE AVOID",
        "reason": (
            "GAT-1 (GABA transporter-1) inhibitor → increases ambient GABA → can PARADOXICALLY induce "
            "non-convulsive status epilepticus (NCSE) in patients with diffuse cortical dysmaturation "
            "or focal cortical dysplasia. FOXG1 involves diffuse cortical immaturity (reduced neuron "
            "number, impaired lamination) — tiagabine NCSE risk is very high. Multiple case reports "
            "of TGB-induced NCSE in DEE patients. ABSOLUTE CONTRAINDICATION."
        ),
    },
    {
        "drug": "Vigabatrin (VGB) — long-term maintenance",
        "risk": "HIGH — retinal toxicity additive to cortical visual impairment",
        "reason": (
            "VGB causes IRREVERSIBLE PERIPHERAL RETINAL TOXICITY (concentric visual field constriction, "
            "retinal nerve fibre layer thinning) in ~30-40% of patients on chronic therapy. FOXG1 "
            "patients commonly have CORTICAL VISUAL IMPAIRMENT (CVI) from frontal/occipital cortical "
            "maldevelopment — adding retinal toxicity to existing CVI causes additional visual "
            "disability. VGB acceptable SHORT-TERM for West syndrome treatment (≤6-12 months); "
            "AVOID chronic maintenance use. ERG mandatory q6M if using."
        ),
    },
]

# ── Monitoring items ──────────────────────────────────────────────────────────
MONITORING_ITEMS = [
    {"item": "POLG sequencing before VPA", "frequency": "Once at diagnosis", "rationale": "POLG mutations + VPA → fatal hepatotoxicity; MANDATORY before starting VPA. Check serum lactate and LFTs simultaneously."},
    {"item": "VPA TDM + LFT + ammonia + FBC", "frequency": "q3M", "rationale": "VPA TDM 50-100 µg/mL; LFT (transaminases) for hepatotoxicity signal; ammonia for VPA hyperammonaemia; FBC for thrombocytopenia."},
    {"item": "ERG (electroretinogram) — if on VGB", "frequency": "q6M (SHARE REMS)", "rationale": "VGB retinal toxicity monitoring; ERG is objective (circumvents communication barriers in FOXG1). Baseline before VGB, then q6M. Stop VGB if >20% amplitude reduction."},
    {"item": "Echocardiogram — if on FFA (Fintepla)", "frequency": "q6M (REMS)", "rationale": "FFA cardiac valvulopathy + PAH risk. REMS-mandated echocardiogram. Baseline before starting, q6M during therapy. Stop if TR velocity >2.8 m/s or new regurgitation."},
    {"item": "MRI brain — frontal volumetrics", "frequency": "At diagnosis + q12-24M if clinically changing", "rationale": "FOXG1 frontal hypoplasia is diagnostic clue; serial MRI monitors gliosis progression, delayed myelination. MRI biomarkers correlate with phenotype severity."},
    {"item": "Video-EEG", "frequency": "At diagnosis; at seizure-type change; q6-12M for VEEG", "rationale": "Distinguish ictal (epileptic) from dyskinetic movements (non-epileptic). FOXG1 dyskinesias can mimic seizures on routine EEG. Video correlation is essential. NREM EEG shows sleep spindle deficit (FOXG1 signature)."},
    {"item": "Developmental assessment (Bayley/VABS/FOXG1 scale)", "frequency": "q6M until age 5, then q12M", "rationale": "FOXG1-specific developmental trajectories: most patients stabilise (not regress — unlike classic Rett) but do not progress. Document gains in head control, social responsiveness, purposeful hand use."},
    {"item": "Nutritional assessment + GI review", "frequency": "q3M dietitian; q6M GI specialist if G-tube", "rationale": "FOXG1 severe oromotor dysfunction → G-tube often required (60% of patients). Monitor weight Z-score, height, head circumference. VPA and KD both affect nutritional state."},
]

# ── Lifecycle windows ─────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {
        "window": "Neonatal / Early Infantile (0-3 months)",
        "key_events": "Profound hypotonia noted at birth; abnormal MRI (frontal hypoplasia); FOXG1 sequencing + CMA; poor feeding → early G-tube planning; seizures may be subtle (tonic posturing, apnoeic episodes).",
        "priority_actions": "Molecular diagnosis ASAP (FOXG1 panel or WES); POLG before VPA; baseline MRI; ophthalmology (CVI assessment); feeding support; PB short-term for acute seizures only.",
    },
    {
        "window": "Infantile Spasms Phase (3-12 months)",
        "key_events": "Epileptic spasms (West syndrome) emerge 3-9 months; EEG shows hypsarrhythmia; ACTH + VGB first-line; VPA commenced; KD planned if poor ACTH response.",
        "priority_actions": "ACTH + VGB (SHARE REMS, ERG baseline); VPA started (POLG done); EEG at day 14; KD initiation if ACTH incomplete; ERG q6M; eye specialist for CVI.",
    },
    {
        "window": "Early Childhood — Seizure Consolidation (1-4 years)",
        "key_events": "Spasms transition to mixed focal + myoclonic pattern; dyskinesias emerge (choreoathetosis); KD ongoing; VGB weaned (retinal risk); CLB added; developmental plateau — no regression.",
        "priority_actions": "VPA + CLB ± KD combination; baclofen for dyskinesias; VGB taper; video-EEG for seizure type characterisation; developmental assessment q6M; G-tube review.",
    },
    {
        "window": "School Age (4-12 years)",
        "key_events": "Seizure pattern stabilises; DRE established in ~70%; KD long-term or transition to MAD; ITB pump considered for severe dyskinesias; augmentative communication (AAC) devices; school placement.",
        "priority_actions": "VPA + CLB + KD optimisation; ITB evaluation if dyskinesias severe; AAC assessment; SUDEP counselling (annual); FFA if DRE and poor KD response; school inclusion planning.",
    },
    {
        "window": "Adolescence (12-18 years)",
        "key_events": "Puberty may exacerbate seizures (catamenial component in girls); scoliosis risk (dyskinesias + hypotonia); VPA teratogenicity counselling in girls; transition planning; SUDEP risk annual review.",
        "priority_actions": "VPA pregnancy counselling (MHRA PREVENT); contraception planning; scoliosis monitoring; orthopaedics; transition to adult neurology; care plan update.",
    },
    {
        "window": "Adulthood (18+ years)",
        "key_events": "Lifelong care needs; seizure frequency often plateaus; dyskinesias may partially improve; SUDEP risk ongoing; advance care planning; caregiver burnout; residential care planning.",
        "priority_actions": "Annual SUDEP risk assessment; VPA + CLB ± KD continuation; ITB pump maintenance; caregiver support (burnout screening); annual multi-disciplinary review; advance care directive.",
    },
]

# ── Key concepts (definitions) ────────────────────────────────────────────────
CONCEPTS = [
    {"term": "FOXG1 / Forkhead Box G1", "definition": "Brain-specific transcription factor (14q12) critical for forebrain neurogenesis, cortical layering, and interneuron migration. LOF → FOXG1 Syndrome (congenital DEE + severe ID + dyskinesias)."},
    {"term": "FOXG1 Syndrome / Congenital Rett Variant", "definition": "DEE caused by de novo FOXG1 LOF or 14q12 deletion. DISTINCT from classic Rett (MECP2): congenital onset, no regression period, hyperkinetic dyskinesias (not stereotypies), equal sex ratio, frontal hypoplasia."},
    {"term": "Dyskinesias (FOXG1)", "definition": "Continuous choreoathetoid involuntary movements — hallmark non-epileptic feature of FOXG1. Baclofen (GABA-B agonist) reduces severity. NOT to be confused with seizures (video-EEG correlation mandatory)."},
    {"term": "Frontal Hypoplasia (MRI biomarker)", "definition": "Reduced frontal lobe volume on MRI — reflects FOXG1's critical role in forebrain development. Delayed myelination and corpus callosum hypoplasia often co-occur. Severity correlates with phenotype."},
    {"term": "Cortical Visual Impairment (CVI)", "definition": "Reduced visual function due to cortical (occipital + frontal) maldevelopment in FOXG1 — NOT primary retinal disease. Ophthalmology + visual electrophysiology (VEP) assessment required. CVI makes VGB retinal toxicity assessment harder (ERG needed)."},
    {"term": "West Syndrome in FOXG1", "definition": "Epileptic spasms + hypsarrhythmia emerging 3-9 months in FOXG1. Lower ACTH response rate (~40-50%) vs cryptogenic West. ACTH+VGB first-line; KD second-line. ERG mandatory if VGB used."},
    {"term": "POLG exclusion before VPA", "definition": "POLG variants + VPA → fatal hepatotoxicity (quadruple-hit: POLG2 inhibition + GSH depletion + carnitine depletion + mPTP opening). FOXG1 patients may carry POLG on background. POLG sequencing MANDATORY before VPA initiation."},
    {"term": "Tiagabine NCSE risk", "definition": "Tiagabine (GAT-1 inhibitor) causes non-convulsive status epilepticus (NCSE) in diffuse cortical dysmaturation including FOXG1. ABSOLUTE contraindication. Mechanism: paradoxical GABA overflow causing tonic GABA-A desensitisation."},
    {"term": "VGB Retinal Toxicity (SHARE REMS)", "definition": "Vigabatrin causes irreversible peripheral retinal toxicity (concentric VF constriction). FDA SHARE REMS: ERG q6M mandatory. In FOXG1: additive to existing CVI. Use VGB short-term only (West syndrome ≤12 months)."},
    {"term": "FHD (Forkhead Domain)", "definition": "DNA-binding domain of FOXG1 (residues ~176-272). FHD missense variants disrupt FOXO consensus binding → moderate-severe FOXG1 syndrome. Outside-FHD variants generally milder phenotype."},
    {"term": "E/I Imbalance (FOXG1)", "definition": "FOXG1 LOF → interneuron deficit (MGE-derived PV+, SST+ interneurons fail to migrate) → reduced cortical inhibition relative to excitation. Primary mechanism of FOXG1 epileptogenicity. Target for GABAergic pharmacotherapy."},
    {"term": "KetoCal / G-tube KD", "definition": "Ketogenic Diet formula (KetoCal 4:1) for enteral feeding via G-tube — essential in FOXG1 where oromotor dysfunction prevents oral KD. β-OHB target 2-4 mmol/L. Bypasses FOXG1 interneuron deficit via KATP/β-OHB mechanisms."},
    {"term": "Baclofen / ITB (Intrathecal Baclofen)", "definition": "GABA-B agonist for FOXG1 dyskinesias. Oral baclofen: 10-20 mg TID. ITB pump: for refractory dyskinesias; more effective but requires GA for implantation. Baclofen withdrawal = medical emergency (fever, rhabdomyolysis, seizure storm)."},
    {"term": "SUDEP in FOXG1", "definition": "Sudden Unexpected Death in Epilepsy — FOXG1 patients are at high SUDEP risk due to drug-resistant epilepsy + GTCS + impaired post-ictal recovery. Annual SUDEP counselling, PGES (post-ictal generalised EEG suppression) monitoring, nocturnal supervision."},
]

# ── Standards & references ────────────────────────────────────────────────────
STANDARDS = [
    "ILAE 2022 Classification of Epilepsies",
    "NICE NG217 Epilepsy Guidelines 2022 (UK)",
    "Ariani 2008 Am J Hum Genet (FOXG1 discovery)",
    "Kortüm 2011 Am J Hum Genet (FOXG1 genotype-phenotype)",
    "UKISS Trial 2004 Lancet Neurol (ACTH+VGB for West)",
    "FDA SHARE REMS Vigabatrin 2021",
    "MHRA PREVENT VPA Pregnancy Prevention Programme 2023",
    "ACMG/AMP Variant Interpretation Standards 2015",
]

THRESHOLDS = [
    "FOXG1 diagnosis: gene sequencing + CMA at congenital DEE — do not await regression (no regression in FOXG1)",
    "POLG exclusion MANDATORY before VPA — zero VPA without POLG result",
    "ACTH day 14 EEG: if spasms persist → escalate (KD or ACTH+VGB combination)",
    "VGB ERG q6M mandatory (SHARE REMS) — ERG >20% reduction → STOP VGB",
    "FFA (Fintepla) echocardiogram q6M — TR velocity >2.8 m/s → STOP FFA",
    "Tiagabine: ABSOLUTE contraindication in FOXG1 (NCSE risk)",
    "CBZ/OXC/PHT: avoid in FOXG1 (worsens myoclonus, reduces VPA levels)",
    "Baclofen withdrawal: medical emergency — NEVER abruptly stop baclofen or ITB",
    "VPA + CLB: check N-desmethyl-CLB levels if sedation occurs",
    "SUDEP counselling: annually from first GTCS — document in notes",
]

REFERENCES = [
    "Ariani F et al. 2008 Am J Hum Genet — FOXG1 discovery as Rett variant cause",
    "Kortüm F et al. 2011 Am J Hum Genet — FOXG1 genotype-phenotype correlation 45 patients",
    "Marwan F et al. 2012 Brain — FOXG1 dyskinesias and clinical characterisation",
    "Vegas N et al. 2018 Eur J Hum Genet — FOXG1 variant registry 104 patients",
    "UKISS Collaborative Group 2004 Lancet Neurol — ACTH vs VGB for West syndrome RCT",
    "Lux AL et al. 2005 Lancet — UKISS 14-month follow-up (ACTH + VGB West outcomes)",
]

# ── Synthetic patient cohort ──────────────────────────────────────────────────
ETIOL_POOL = [
    ("FOXG1-de-novo-truncating-LOF-classic", 42),
    ("FOXG1-de-novo-missense-FHD-domain", 29),
    ("FOXG1-whole-gene-deletion-14q12", 15),
    ("FOXG1-de-novo-missense-outside-FHD", 9),
    ("FOXG1-negative-phenocopy", 5),
]

def _make_patients():
    pats = []
    idx = 0
    for cat, pct in ETIOL_POOL:
        n = round(41 * pct / 100)
        for _ in range(n):
            if len(pats) >= 41:
                break
            onset_m = random.randint(1, 9) if "truncating" in cat or "deletion" in cat else random.randint(3, 18)
            age_m = random.randint(onset_m + 12, 240)
            west = random.random() < 0.82
            drug_resistant = random.random() < 0.72
            on_kd = drug_resistant and random.random() < 0.60
            on_vpa = random.random() < 0.88
            polg_tested = random.choice(["Y", "Y", "Y", "N"])
            gtube = random.random() < 0.62
            baclofen_on = random.random() < 0.55
            acth_response = random.choice(["complete", "partial", "none"]) if west else "n/a"
            vgb_used = west and random.random() < 0.70
            pats.append({
                "id": f"FXG-{idx+1:03d}",
                "category": cat,
                "onset_months": onset_m,
                "age_months": age_m,
                "west_syndrome": west,
                "drug_resistant": drug_resistant,
                "on_kd": on_kd,
                "on_vpa": on_vpa,
                "polg_tested": polg_tested,
                "gtube": gtube,
                "baclofen_on": baclofen_on,
                "acth_response": acth_response,
                "vgb_used": vgb_used,
            })
            idx += 1
    # Fill to 41 if rounding leaves short
    while len(pats) < 41:
        pats.append({
            "id": f"FXG-{idx+1:03d}",
            "category": "FOXG1-de-novo-truncating-LOF-classic",
            "onset_months": random.randint(2, 6),
            "age_months": random.randint(24, 180),
            "west_syndrome": True,
            "drug_resistant": True,
            "on_kd": True,
            "on_vpa": True,
            "polg_tested": "Y",
            "gtube": True,
            "baclofen_on": False,
            "acth_response": "partial",
            "vgb_used": True,
        })
        idx += 1
    return pats[:41]

PATIENTS = _make_patients()

# ─── Public API ───────────────────────────────────────────────────────────────

def get_overview():
    n = len(PATIENTS)
    n_west = sum(1 for p in PATIENTS if p["west_syndrome"])
    n_dre = sum(1 for p in PATIENTS if p["drug_resistant"])
    n_kd = sum(1 for p in PATIENTS if p["on_kd"])
    n_gtube = sum(1 for p in PATIENTS if p["gtube"])
    n_baclofen = sum(1 for p in PATIENTS if p["baclofen_on"])
    n_polg_done = sum(1 for p in PATIENTS if p["polg_tested"] == "Y")
    n_vpa = sum(1 for p in PATIENTS if p["on_vpa"])
    avg_onset = round(sum(p["onset_months"] for p in PATIENTS) / n, 1)

    return {
        "title": "FOXG1 Syndrome (Congenital Rett Variant / FOXG1-Related DEE)",
        "gene": "FOXG1",
        "locus": "14q12",
        "inheritance": "X-linked dominant / de novo (equal sex ratio)",
        "protein": "Forkhead Box G1 (forebrain transcription factor — progenitor pool, cortical layering, interneuron migration)",
        "mechanism": "FOXG1 haploinsufficiency → premature cortical progenitor cell cycle exit + interneuron migration defect (PV+/SST+ interneurons) → E/I imbalance → congenital DEE + dyskinesias",
        "key_aha": (
            "FOXG1 syndrome is NOT classic Rett — NO regression period (congenital onset), "
            "hyperkinetic DYSKINESIAS (not hand stereotypies), equal sex ratio. "
            "CBZ/OXC/PHT worsen myoclonus. Tiagabine = ABSOLUTE CI (NCSE risk in diffuse cortical dysmaturation). "
            "POLG MANDATORY before VPA. Baclofen (GABA-B) for dyskinesias — withdrawal = emergency."
        ),
        "n_patients": n,
        "west_syndrome_pct": round(100 * n_west / n),
        "drug_resistant_pct": round(100 * n_dre / n),
        "on_kd_pct": round(100 * n_kd / n),
        "gtube_pct": round(100 * n_gtube / n),
        "baclofen_pct": round(100 * n_baclofen / n),
        "polg_done_pct": round(100 * n_polg_done / n),
        "on_vpa_pct": round(100 * n_vpa / n),
        "avg_onset_months": avg_onset,
        "contraindications_summary": [c["drug"] for c in CONTRAINDICATIONS],
        "standards": STANDARDS,
        "thresholds": THRESHOLDS[:5],
        "references": REFERENCES,
    }


def get_breakdown():
    n = len(PATIENTS)
    etiol_counts = {}
    for p in PATIENTS:
        etiol_counts[p["category"]] = etiol_counts.get(p["category"], 0) + 1

    seizure_bars = [
        {"type": s["type"], "prevalence_pct": s["prevalence_pct"]}
        for s in SEIZURE_TYPES
    ]
    trigger_bars = [
        {"trigger": t["trigger"], "prevalence_pct": t["prevalence_pct"]}
        for t in TRIGGERS
    ]
    treatment_cards = [
        {
            "drug": t["drug"],
            "level": t["level"],
            "dose": t["dose"],
            "efficacy": t["efficacy"],
        }
        for t in TREATMENTS
    ]
    ci_list = [
        {"drug": c["drug"], "risk": c["risk"], "reason_short": c["reason"][:120]}
        for c in CONTRAINDICATIONS
    ]

    n_west = sum(1 for p in PATIENTS if p["west_syndrome"])
    n_acth_complete = sum(1 for p in PATIENTS if p["acth_response"] == "complete")
    n_acth_partial  = sum(1 for p in PATIENTS if p["acth_response"] == "partial")
    n_acth_none     = sum(1 for p in PATIENTS if p["acth_response"] == "none")
    n_vpa_without_polg = sum(1 for p in PATIENTS if p["on_vpa"] and p["polg_tested"] == "N")

    return {
        "summary": {
            "total": n,
            "west_syndrome_pct": round(100 * n_west / n),
            "drug_resistant_pct": round(100 * sum(1 for p in PATIENTS if p["drug_resistant"]) / n),
            "kd_pct": round(100 * sum(1 for p in PATIENTS if p["on_kd"]) / n),
            "gtube_pct": round(100 * sum(1 for p in PATIENTS if p["gtube"]) / n),
            "acth_complete_pct": round(100 * n_acth_complete / max(n_west, 1)),
            "acth_partial_pct": round(100 * n_acth_partial / max(n_west, 1)),
            "acth_none_pct": round(100 * n_acth_none / max(n_west, 1)),
            "vpa_without_polg": n_vpa_without_polg,
            "baclofen_pct": round(100 * sum(1 for p in PATIENTS if p["baclofen_on"]) / n),
        },
        "etiology_distribution": [
            {
                "category": e["category"],
                "n": etiol_counts.get(e["category"], 0),
                "pct": e["pct"],
                "etiology": e["etiology"],
                "mechanism_short": e["mechanism"][:150],
                "eeg_signature_short": e["eeg_signature"][:150],
            }
            for e in ETIOLOGY_CATALOG
        ],
        "seizure_types": seizure_bars,
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
        "triggers": trigger_bars,
        "trigger_detail": TRIGGERS,
        "treatments": treatment_cards,
        "treatment_detail": TREATMENTS,
        "contraindications": ci_list,
        "contraindication_detail": CONTRAINDICATIONS,
        "monitoring": MONITORING_ITEMS,
        "lifecycle": LIFECYCLE_WINDOWS,
        "patients_sample": [
            {
                "id": p["id"],
                "category": p["category"],
                "onset_months": p["onset_months"],
                "age_months": p["age_months"],
                "west_syndrome": p["west_syndrome"],
                "drug_resistant": p["drug_resistant"],
                "on_kd": p["on_kd"],
                "gtube": p["gtube"],
                "baclofen_on": p["baclofen_on"],
                "polg_tested": p["polg_tested"],
            }
            for p in PATIENTS
        ],
    }


def get_definitions():
    return {
        "gene": "FOXG1",
        "syndrome": "FOXG1 Syndrome (Congenital Rett Variant / FOXG1-Related DEE)",
        "concepts": CONCEPTS,
        "contraindications_full": CONTRAINDICATIONS,
        "monitoring_full": MONITORING_ITEMS,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
        "lifecycle": LIFECYCLE_WINDOWS,
    }
