"""
CALM1 / CALM2 / CALM3 Calmodulinopathy — DEE + Long-QT Syndrome / CPVT
(Calmodulin-Related Developmental & Epileptic Encephalopathy / Sudden Cardiac Death Risk)
==========================================================================================
40-patient cohort · CALM1 (14q32.11) / CALM2 (2p21) / CALM3 (19q13.32) · De novo GOF
DEE + Polymorphic VT + LQTS + CPVT · OMIM #616036 / #616037 / #616038

CALMODULIN BIOLOGY:
CALM1, CALM2, and CALM3 encode IDENTICAL 148-amino acid calmodulin (CaM) protein
via three different chromosomal loci — a unique feature in mammalian genetics.
CaM is the principal intracellular Ca²⁺ sensor in the CNS and myocardium.

CaM STRUCTURE & FUNCTION:
  - 4 EF-hand Ca²⁺-binding motifs (EF1-4): two per globular domain (N-lobe + C-lobe)
  - Ca²⁺ affinity: Kd ~10⁻⁶ M (physiological range); conformation shifts on Ca²⁺ binding
  - IQ-motif binding: CaM constitutively binds NaV1.5 IQ-domain, RyR2, L-VDCC (Cav1.2),
    and KCNQ2/3 — regulating their gating in a Ca²⁺-dependent manner.
  - CDI (Ca²⁺-dependent inactivation): CaM C-lobe mediates Cav1.2 CDI after Ca²⁺ influx
    → limits plateau phase of cardiac AP and neuronal burst firing.

CALMODULINOPATHY PATHOMECHANISM (de novo GOF):
  CALM missense variants (most commonly N-lobe EF2: D96V, N97I, N97S; C-lobe EF3:
  D130G, D132E, F142L, F142C) → reduced Ca²⁺-binding affinity of mutant CaM EF-hands.

  CARDIAC CONSEQUENCES:
  → Impaired CDI of Cav1.2 → prolonged L-type Ca²⁺ current (ICaL) → prolonged cardiac AP
    plateau → QTc prolongation (LQTS type 14/15/16) → Torsade de Pointes (TdP) → SCD.
  → Impaired CaM-RyR2 interaction → aberrant sarcoplasmic Ca²⁺ release → triggered
    activity → CPVT (Catecholaminergic Polymorphic VT) phenotype at normal QTc.
  → Combined LQTS + CPVT is a unique calmodulinopathy-specific lethal arrhythmia risk.

  NEUROLOGICAL CONSEQUENCES:
  → Impaired Ca²⁺-CaM signalling in NaV1.5, NMDA receptors, KCa channels → neuronal
    hyperexcitability → early-onset DEE.
  → CaM-KCNQ2/3 interaction loss → M-current failure → neonatal seizure onset
    (calmodulinopathy BFNS-like in milder CALM1/3 variants).
  → CaM-RyR1 impairment in neurons → dendritic Ca²⁺ dysregulation → DEE.

CALMODULINOPATHY CARDIAC PHENOTYPES:
  - LQTS-14 (CALM1): Mean QTc 570ms; IVF/TdP first presentation 30%
  - LQTS-15 (CALM2): Most severe; mean QTc 600ms; neonatal cardiac arrest common
  - LQTS-16 (CALM3): Intermediate; QTc 540ms; seizures + CPVT co-presentation
  - CPVT5 (CALM1/2): CPVT at near-normal QTc; exercise-induced bidirectional VT
  - Overlap syndrome: LQTS + CPVT in same patient — highest SCD risk

CALMODULINOPATHY EPILEPSY PHENOTYPES:
  - Early-onset DEE (neonatal/infantile): most with CALM2 and severe CALM1 variants
  - Seizure types: Focal motor, IS/West, multifocal clonic, GTCS
  - EEG: modified hypsarrhythmia, burst-suppression, multifocal IED
  - Drug-resistant: >75% (similar to KCNT1-EIMFS)
  - Mechanism: neuronal hyperexcitability + seizure-triggered cardiac arrhythmia
    (seizure → catecholamine surge → CPVT/TdP in sensitised myocardium → SCD)

GENETICS:
  - Genes: CALM1 (14q32.11), CALM2 (2p21), CALM3 (19q13.32)
  - All encode: IDENTICAL 148 aa calmodulin (CaM) — same protein, 3 gene loci
  - Inheritance: Autosomal Dominant; >99% de novo (familial <1%)
  - Variant types: Missense in EF-hand domains ~95%; truncating very rare (CaM essential gene)
  - pLI: CALM1=1.00, CALM2=1.00, CALM3=1.00 (among most LOF-intolerant genes)
  - Incidence: ~1:300,000–500,000 live births; ~350+ published families (all 3 genes)
  - OMIM: CALM1-LQTS14 #616036; CALM2-LQTS15 #616037; CALM3-LQTS16 #616038
  - Database: ClinVar ~120 P/LP variants across CALM1/2/3; hotspot residues well-defined

KEY CLINICAL DISTINCTIONS:
  1. Calmodulinopathy = ONLY epilepsy syndrome with mandatory cardiology co-management
  2. QTc-prolonging AEDs MUST be avoided (phenytoin → minimal QTc risk but avoid;
     carbamazepine → no QTc prolongation significant; SAFE drugs: VPA, LEV, CLB, KD)
  3. Implantable Cardioverter Defibrillator (ICD) indicated for QTc >500ms + VT history
  4. Beta-blocker (nadolol/propranolol) + flecainide → CPVT-specific treatment
  5. Seizure → catecholamine surge → can trigger VT/VF → seizure and cardiac arrest
     may be clinically inseparable → SUDDEN UNEXPECTED DEATH
  6. Mexiletine (INa blocker) + flecainide combination for LQTS14/15 with ICD

KEY STANDARDS:
  ILAE 2022 · NICE NG217 · Nyegaard et al. 2012 (Am J Hum Genet — CALM1/2 first report) ·
  Crotti et al. 2013 (Heart Rhythm — CALM3 first report) · Reed et al. 2015 (Circulation) ·
  Bhatt et al. 2023 (Neurology) · ESC LQTS Guidelines 2022 · HRS CPVT Guidelines 2019 ·
  MHRA VPPP 2021 · ILAE Dietary Therapies 2018 · ACMG-AMP 2015 ·
  CPIC POLG-VPA 2023 · EAN Neonatal SE Guidelines 2019
"""
import random

# ── Etiology catalog ──────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "CALM2 de-novo GOF (Severe Neonatal DEE + LQTS-15 — Most Severe)",
        "category": "CALM2-de-novo-GOF-neonatal-DEE-LQTS15-38%",
        "pct": 38,
        "n": 15,
        "mechanism": (
            "CALM2 variants (N97S, D130G, D132E) in EF-hand Ca²⁺-binding loops produce strongest "
            "CDI impairment of Cav1.2 → most prolonged QTc (mean 600ms) + most severe neurological "
            "phenotype among calmodulinopathies. CALM2 de novo missense → neonatal burst-suppression, "
            "IS/West syndrome, multifocal DEE. Cardiac: neonatal VT/VF + TdP prominent; ICD placement "
            "in neonatal period. Combined LQTS15 + CPVT5 overlap — highest SCD risk (~35% SCD by age 10 "
            "untreated). Intellectual disability: severe-profound; epilepsy drug-resistant 80%."
        ),
        "eeg_correlate": (
            "Burst-suppression pattern neonatal · Modified hypsarrhythmia 0-12M · "
            "Multifocal IED + GCSW · Background severe slowing · "
            "Ictal: multifocal clonic + IS patterns"
        ),
        "semiology": (
            "Neonatal seizures day 1-7 (focal clonic → multifocal); West syndrome 3-12M; "
            "severe global DD; neonatal/infantile cardiac arrest in 30% prior to epilepsy dx; "
            "QTc 580-620ms; IVF/TdP episodes; CPVT-triggered VT on exertion. "
            "SCD risk highest in this class without ICD."
        ),
        "treatment": "VPA+CLB (broad; avoid QTc-prolonging AEDs); KD (Level B, DRE); Nadolol+Flecainide+ICD (cardiac)",
        "prognosis": "Poor — drug-resistant 80%; severe ID; SCD risk 35% by age 10 without ICD; QTc monitoring lifelong",
    },
    {
        "etiology": "CALM1 de-novo GOF (DEE + LQTS-14 + CPVT5 — Combined Overlap)",
        "category": "CALM1-de-novo-GOF-DEE-LQTS14-CPVT5-30%",
        "pct": 30,
        "n": 12,
        "mechanism": (
            "CALM1 variants (D96V, N97I, F142L) — EF-hand Ca²⁺ affinity reduction → dual "
            "impairment of Cav1.2 CDI (LQTS component: mean QTc 570ms) and RyR2 regulation "
            "(CPVT component: adrenergic-triggered bidirectional VT at catecholamine surge). "
            "Neurological: early infantile DEE (onset 1-6M), West syndrome 30%, focal/multifocal 70%. "
            "Seizure-to-cardiac-arrhythmia cascade: seizure → epinephrine surge → RyR2 "
            "hypersensitivity → VT/VF → sudden death. Drug-resistant 70%; moderate-severe ID."
        ),
        "eeg_correlate": (
            "Hypsarrhythmia (West 30%) · Multifocal spikes · "
            "IED activation in NREM · "
            "Post-ictal cardiac rhythm disturbance on telemetry"
        ),
        "semiology": (
            "Infantile spasms 1-6M; focal motor 70%; GTCS; exercise-provoked CPVT "
            "(bidirectional VT, syncope); QTc 540-580ms; syncopal episodes mistaken for seizures. "
            "ICD implanted in 75% of CALM1-CPVT5 overlap. Moderate-severe ID."
        ),
        "treatment": "VPA+LEV (cardiac-safe); Nadolol (CPVT) + Flecainide (CPVT+LQTS); ICD (overlap); KD (DRE)",
        "prognosis": "Moderate-poor — drug-resistant 70%; moderate-severe ID; SCD 20% untreated; ICD dramatically improves survival",
    },
    {
        "etiology": "CALM3 de-novo GOF (Intermediate DEE + LQTS-16 / CPVT5)",
        "category": "CALM3-de-novo-GOF-DEE-LQTS16-15%",
        "pct": 15,
        "n": 6,
        "mechanism": (
            "CALM3 variants (D130G, F142C, E141K) — intermediate phenotypic severity between CALM1 "
            "and CALM2. QTc mean 540ms; CPVT component present in 60% (CALM3-CPVT5). "
            "Neurological: infantile/early childhood onset DEE; seizure types: IS (20%), focal motor "
            "(65%), GTCS (45%). CALM3 mRNA expression slightly lower than CALM1/2 in neurons → "
            "partially protective against most severe neurological impairment. "
            "Drug-resistant 60%; mild-moderate ID."
        ),
        "eeg_correlate": (
            "Multifocal IED · IS-type pattern (20%) · "
            "Generalised spike-wave bursts · "
            "Background slowing — moderate"
        ),
        "semiology": (
            "Infantile onset (1-12M); spasms + focal motor; GTCS from age 2-5; "
            "QTc 520-560ms; exercise-provoked syncope (CPVT); cardiac-neuro overlap events. "
            "Moderate cognitive disability; ICD in 50%."
        ),
        "treatment": "LEV+VPA (cardiac-safe combination); Nadolol (CPVT); ICD if QTc>500ms+VT; KD (Level B, DRE)",
        "prognosis": "Moderate — drug-resistant 60%; mild-moderate ID; SCD risk 15% without ICD; good cardiac outcomes with ICD+nadolol",
    },
    {
        "etiology": "CALM1/2/3 Mild (BFNS-like / Isolated LQTS — Lower Severity)",
        "category": "CALM-mild-BFNS-LQTS-only-10%",
        "pct": 10,
        "n": 4,
        "mechanism": (
            "Hypomorphic missense in non-critical EF-hand positions → partial CDI impairment → "
            "QTc 470-510ms (borderline-prolonged); CPVT only on extreme exercise. "
            "Neurological: BFNS-like neonatal focal seizures (self-limited by 3-6M) or no epilepsy. "
            "Cardiac monitoring mandatory even with mild QTc (incomplete penetrance of CPVT). "
            "Normal-borderline IQ. AED withdrawal after 2-3 years of seizure freedom considered."
        ),
        "eeg_correlate": (
            "Focal IED (rolandic/temporal) neonatal · "
            "Generalised normal background · "
            "Normal by 6-12 months in most"
        ),
        "semiology": (
            "Neonatal focal clonic seizures; spontaneous remission 3-6M; "
            "normal development; QTc 470-510ms; stress ECG mandated annually. "
            "ICD generally not indicated unless QTc >500ms + VT documented."
        ),
        "treatment": "PB (Level B, neonatal); CBZ (BFNS-safe, NO QTc prolongation); Observation + cardiac surveillance",
        "prognosis": "Good neurologically — self-limited seizures; normal IQ; cardiac surveillance lifelong (CPVT risk persists)",
    },
    {
        "etiology": "Phenocopy (LQTS-Non-CALM + DEE / SCN1A/KCNQ2 + acquired LQTS)",
        "category": "Phenocopy-LQTS-nonCALM-DEE-7%",
        "pct": 7,
        "n": 3,
        "mechanism": (
            "DEE (SCN1A/KCNQ2/SCN2A) + acquired LQTS from QTc-prolonging AED (phenytoin, "
            "large-dose carbamazepine at toxic levels, or co-prescribed QTc drugs). "
            "Negative CALM1/2/3 sequencing; LQTS-panel genes screened (KCNQ1, KCNH2, SCN5A). "
            "Key distinction: acquired QTc prolongation resolves with AED change; "
            "CALM GOF QTc is fixed and persistent regardless of AED."
        ),
        "eeg_correlate": (
            "Gene-specific DEE pattern (SCN1A/KCNQ2) · "
            "No CALM-specific feature"
        ),
        "semiology": (
            "DEE phenotype (Dravet/DEE2-like); QTc prolonged on ECG; "
            "CALM1/2/3 negative; acquired LQTS on drug review; "
            "cardiac recovery after AED optimisation."
        ),
        "treatment": "Gene-specific AED (VPA/LEV/CLB for SCN1A Dravet; CBZ/OXC for KCNQ2); withdraw QTc-prolonging drug",
        "prognosis": "Variable — depends on underlying gene; acquired LQTS resolves; CALM excluded → standard DEE management",
    },
]

# ── Patient cohort ─────────────────────────────────────────────────────────────
random.seed(42)
_CALM_GENES = ["CALM1", "CALM1", "CALM1", "CALM2", "CALM2", "CALM2", "CALM2", "CALM3", "CALM3", "CALM3"]
_CALM_VARIANTS = {
    "CALM1": ["D96V", "N97I", "F142L", "F142C", "D130G"],
    "CALM2": ["N97S", "D130G", "D132E", "F141L", "N97I"],
    "CALM3": ["D130G", "F142C", "E141K", "D96V", "N97S"],
}
_OUTCOMES = ["Drug-resistant", "Drug-resistant", "Drug-resistant", "Partial-response", "Partial-response", "Seizure-free"]
_PHENOTYPES = ["DEE+LQTS", "DEE+LQTS+CPVT", "DEE+CPVT", "Mild-BFNS+LQTS", "DEE+LQTS"]
_ICD = [True, True, True, True, False, False, False, True, True, True]

PATIENTS = []
for i in range(40):
    gene = random.choice(_CALM_GENES)
    var = random.choice(_CALM_VARIANTS[gene])
    qtc = random.randint(460, 620)
    icd = qtc > 500 or random.random() < 0.3
    onset_m = random.randint(0, 18)
    outcome = random.choice(_OUTCOMES)
    phenotype = random.choice(_PHENOTYPES)
    PATIENTS.append({
        "patient_id": f"CALM-{i+1:03d}",
        "gene": gene,
        "variant": f"p.{var}",
        "inheritance": "de novo",
        "onset_months": onset_m,
        "phenotype": phenotype,
        "qtc_ms": qtc,
        "icd_implanted": icd,
        "outcome": outcome,
        "drug_resistant": outcome == "Drug-resistant",
        "current_aeds": random.choice(["VPA+LEV", "VPA+CLB", "LEV+CLB", "VPA+LEV+KD", "KD+CLB"]),
        "cardiac_rx": "Nadolol+Flecainide+ICD" if icd else ("Nadolol" if qtc > 480 else "Surveillance"),
    })

# ── Seizure types ──────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Infantile Spasms / West Syndrome",
        "prevalence_pct": 55,
        "eeg_pattern": "Hypsarrhythmia (modified) · High-voltage chaotic background · Burst-suppression pre-West",
        "semiology": (
            "Clustered flexion/extension spasms 3-12M; "
            "CALM2 and severe CALM1: hypsarrhythmia + developmental plateau; "
            "ACTH response 20-30% (lower than typical West); "
            "vigabatrin partial response; KD most effective in calmodulinopathy-IS."
        ),
        "clinical_tip": (
            "MANDATORY: ECG during spasm cluster (seizure-triggered catecholamine surge → VT). "
            "Inpatient cardiac telemetry for IS treatment initiation."
        ),
    },
    {
        "type": "Focal Motor Seizures (Multifocal Clonic)",
        "prevalence_pct": 78,
        "eeg_pattern": "Multifocal IED (frontal/temporal/occipital) · Ictal: focal rhythmic theta evolving → clonic",
        "semiology": (
            "Neonatal onset: focal clonic (arm/face/leg); migrating pattern (KCNT1-like in CALM2 severe); "
            "post-ictal contralateral weakness; prolonged focal SE common (30%). "
            "QTc monitoring during prolonged focal seizures — cardiac risk window."
        ),
        "clinical_tip": (
            "Any prolonged focal seizure in calmodulinopathy → cardiac monitoring mandatory. "
            "Seizure-induced catecholamine → CPVT trigger window."
        ),
    },
    {
        "type": "GTCS (Generalised Tonic-Clonic Seizures)",
        "prevalence_pct": 48,
        "eeg_pattern": "Bilateral recruiting rhythm → tonic → clonic · Post-ictal EEG suppression · QT interval measured post-ictally",
        "semiology": (
            "Childhood/adolescent GTCS (following neonatal/infantile onset); "
            "GTCS = highest-risk event in calmodulinopathy (catecholamine surge peaks); "
            "post-ictal VT/VF reported; resuscitation required in 8%; "
            "nocturnal GTCS + SUDEP risk synergise with LQTS — extreme SCD risk."
        ),
        "clinical_tip": (
            "GTCS in calmodulinopathy = cardiac emergency co-management. "
            "Post-ictal ECG mandatory. ICD evaluation after first convulsion in LQTS+DEE."
        ),
    },
    {
        "type": "Absence-like / NCSE Episodes",
        "prevalence_pct": 35,
        "eeg_pattern": "Generalised 2.5-3.5 Hz spike-wave (atypical) · May represent ictal QT prolongation on ECG simultaneously",
        "semiology": (
            "Staring + unresponsiveness misinterpreted as absence; "
            "may be non-convulsive SE or cardiac syncope (VT → cerebral hypoperfusion); "
            "CRITICAL DISTINCTION: NCSE (EEG active) vs. cardiac syncope (normal EEG, abnormal ECG); "
            "simultaneous EEG-ECG monitoring required for diagnosis."
        ),
        "clinical_tip": (
            "Never treat calmodulinopathy staring spells with AED alone without ECG. "
            "Cardiac syncope from CPVT/TdP mimics absence perfectly."
        ),
    },
    {
        "type": "Tonic Seizures / Myoclonic Episodes",
        "prevalence_pct": 28,
        "eeg_pattern": "Tonic: fast low-amplitude EEG recruiting rhythm · Myoclonic: bilateral polyspike burst",
        "semiology": (
            "Tonic seizures: nocturnal, LGS-like in older children with severe CALM2 DEE; "
            "myoclonic: jerks not pathognomonic (less prominent than SCN1A-Dravet); "
            "tonic + LGS pattern → KD-first approach."
        ),
        "clinical_tip": (
            "Tonic seizures with cardiac telemetry: QTc measured pre/during/post tonic. "
            "KD initiation for tonic + LGS in calmodulinopathy: monitor ketosis carefully."
        ),
    },
]

# ── Triggers ──────────────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Fever / Intercurrent Illness", "prevalence_pct": 88,
     "cardiac_risk": "HIGH — fever elevates heart rate → shortens diastolic RyR2 Ca²⁺ recovery → CPVT window",
     "mechanism": "Hyperthermia + catecholamine → synergistic CPVT/TdP risk; febrile seizure → SCD cascade",
     "management": "Aggressive antipyresis mandatory; cardiac monitoring during febrile illness; hospital admission for >38.5°C"},
    {"trigger": "Exercise / Physical Exertion", "prevalence_pct": 72,
     "cardiac_risk": "VERY HIGH — adrenergic surge → CPVT trigger (classic CALM-CPVT5 trigger)",
     "mechanism": "Catecholamine surge on exercise → RyR2 DADs → bidirectional VT → VF → SCD",
     "management": "Supervised exercise only; beta-blocker mandatory before any sport; ICD indispensable for CPVT"},
    {"trigger": "Missed AED / Medication Non-Adherence", "prevalence_pct": 75,
     "cardiac_risk": "HIGH — breakthrough seizure → catecholamine → VT cascade",
     "mechanism": "Seizure breakthrough → adrenergic surge → CALM-sensitised RyR2 → cardiac arrhythmia",
     "management": "Electronic medication reminders; caregiver double-check; rescue AED protocol"},
    {"trigger": "Emotional Stress / Fright / Startles", "prevalence_pct": 65,
     "cardiac_risk": "HIGH — adrenergic stress → CPVT trigger (identical to CALM-CPVT5)",
     "mechanism": "CNS-mediated sympathetic activation → cardiac and neurological simultaneous hyperexcitability",
     "management": "Stress management; beta-blocker coverage; avoid startling stimuli in cardiac-at-risk periods"},
    {"trigger": "Sleep-Wake Transitions", "prevalence_pct": 60,
     "cardiac_risk": "MODERATE — autonomic transition + arousal catecholamine",
     "mechanism": "NREM-REM transition → heart rate surge → CPVT window; seizures cluster at awakening",
     "management": "Overnight telemetry for new/breakthrough events; SUDEP pillow (prone-to-supine alarms)"},
    {"trigger": "QTc-Prolonging Co-medications", "prevalence_pct": 55,
     "cardiac_risk": "VERY HIGH — drug-induced QTc further prolongation on already prolonged baseline",
     "mechanism": "Macrolide antibiotics/antifungals/metoclopramide + CALM-LQTS → extreme QTc → VF",
     "management": "MANDATORY drug interaction check before ANY co-prescription; CredibleMeds database consultation"},
    {"trigger": "Electrolyte Disturbance (Hypokalaemia/Hypocalcaemia)", "prevalence_pct": 45,
     "cardiac_risk": "HIGH — electrolyte derangement additively prolongs QTc",
     "mechanism": "K⁺ ↓ → delayed cardiac repolarisation → TdP risk compounded with CALM-LQTS",
     "management": "KD patients: monitor K⁺/Ca²⁺/Mg²⁺ (KD causes electrolyte shifts); IV electrolyte replacement during illness"},
    {"trigger": "Sudden Loud Noise / Acoustic Startle", "prevalence_pct": 35,
     "cardiac_risk": "MODERATE-HIGH — startle-induced cardiac arrest reported in CALM calmodulinopathy",
     "mechanism": "Auditory startle → vagal reflex + sympathetic surge → CPVT trigger",
     "management": "Acoustic startle triggers documented in notes; family/school educated on sudden-noise CPVT risk"},
]

# ── Treatments ─────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproate (VPA)",
        "evidence": "Level B (Neurological) — FIRST-LINE for DEE component (no QTc prolongation)",
        "dose": "30–60 mg/kg/day (neonatal 20–40); TDM target 75–120 mg/L",
        "moa": "Na⁺-channel modulation + GABA-T inhibition + histone deacetylase inhibition; NO Cav1.2/RyR2 effect → cardiac-safe",
        "efficacy": "40-50% ≥50% seizure reduction in calmodulinopathy DEE; superior to PHT/CBZ in broad-spectrum",
        "safety": "POLG screen MANDATORY before VPA; hepatotoxicity neonatal (<3y); VPPP females ≥12y; hyperammonaemia",
        "monitoring": "VPA TDM q3M; LFT + FBC + ammonia q3M; POLG pre-VPA; VPPP females annually",
        "calm_note": "PREFERRED for calmodulinopathy DEE — no QTc prolongation, no Cav1.2 effect; safer cardiac profile than phenytoin",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "evidence": "Level B (Neurological) — FIRST-LINE adjunct; cardiac-safe",
        "dose": "20–60 mg/kg/day; neonatal 10–40 mg/kg/day; IV loading 40–60 mg/kg for SE",
        "moa": "SV2A (synaptic vesicle protein 2A) modulator → reduces vesicular glutamate + GABA release; no cardiac ion channel effect",
        "efficacy": "30-40% responder in DEE; best for focal motor + GTCS component; add-on to VPA",
        "safety": "Irritability/aggression (DEE patients 25%); rare SJS; NO QTc effect; renal dose-adjusted",
        "monitoring": "Behavioural monitoring; renal function; no TDM required routinely",
        "calm_note": "PREFERRED cardiac-safe AED for calmodulinopathy; no ion channel cardiac interaction; IV formulation useful in SE",
    },
    {
        "drug": "Clobazam (CLB)",
        "evidence": "Level B — adjunct for refractory DEE; cardiac-safe",
        "dose": "0.1–0.3 mg/kg/day (divided bid); max 1–1.5 mg/kg/day neonatal-infantile",
        "moa": "GABA-A BZD site agonist; reduces network excitability; no cardiac ion channel effect",
        "efficacy": "35-45% responder (add-on to VPA+LEV); tolerance develops in 30% at 6-12M",
        "safety": "Sedation; tolerance; neonatal respiratory depression; NO QTc effect",
        "monitoring": "Sedation scale; respiratory rate neonatal; tolerance assessment q6M",
        "calm_note": "Safe in calmodulinopathy — no cardiac interaction; useful for focal SE rescue",
    },
    {
        "drug": "Ketogenic Diet (KD 4:1)",
        "evidence": "Level B — most effective for refractory calmodulinopathy-IS and DEE",
        "dose": "4:1 ratio (fat:carbohydrate+protein); BHB target 2–4 mmol/L; RD-supervised",
        "moa": "Ketone bodies substitute glucose in seizure network; KATP channel opening; reduces glutamate; no QTc effect",
        "efficacy": "40-55% ≥50% seizure reduction in calmodulinopathy; most effective after ≥2 AED failures; IS responder 35%",
        "safety": "Dyslipidaemia; renal calculi; growth retardation; electrolyte shifts (K⁺↓ Mg²⁺↓ Ca²⁺↓ → CARDIAC monitoring!)",
        "monitoring": "Ketones daily; lipids q3M; electrolytes q3M (K⁺/Ca²⁺/Mg²⁺ — CALM cardiac safety); growth q6M; DEXA annual",
        "calm_note": "CRITICAL KD-cardiac interaction: KD causes hypokalaemia/hypocalcaemia → compoundsQTc in CALM-LQTS. Electrolyte correction MANDATORY.",
    },
    {
        "drug": "Nadolol (Beta-Blocker — CARDIAC FIRST-LINE)",
        "evidence": "Level A (Cardiac) — CPVT5 and LQTS14/15/16 first-line cardiac drug",
        "dose": "1–2 mg/kg/day (nadolol preferred over propranolol: longer half-life, more consistent CPVT suppression)",
        "moa": "Beta-1/Beta-2 adrenergic blockade → blocks catecholamine-triggered RyR2 Ca²⁺ release → suppresses CPVT; QTc-neutral",
        "efficacy": "65-75% CPVT episode-free; 40% QTc shortening in LQTS-component; reduces exercise-induced VT",
        "safety": "Bradycardia; bronchospasm (caution asthma); hypoglycaemia in neonates; neonatal dosing challenging",
        "monitoring": "Resting HR >50/min (neonatal); BP; exercise stress test annually (CPVT response); 24h Holter",
        "calm_note": "MANDATORY for all CALM1/2/3 with CPVT5 phenotype; also initiated in LQTS-only as SCD prevention; no CNS AED interaction",
    },
    {
        "drug": "Flecainide (Na-Channel Blocker — CARDIAC ADJUNCT)",
        "evidence": "Level B (Cardiac) — CPVT5 adjunct when nadolol insufficient",
        "dose": "100–200mg/day adults; 2–4 mg/kg/day paediatric; TDM target 0.2–1.0 mg/L",
        "moa": "INa blockade reduces DAD amplitude at RyR2-hyperactive sites → suppresses triggered CPVT arrhythmia",
        "efficacy": "Additional 30-40% CPVT suppression on top of beta-blocker; used in nadolol-insufficient CALM-CPVT5",
        "safety": "CAUTION: flecainide is pro-arrhythmic in structural heart disease (NOT INDICATED without ICD backup in severe LQTS)",
        "monitoring": "ECG QRS widening (<25% increase); TDM; ICD interrogation quarterly",
        "calm_note": "ALWAYS used WITH ICD in calmodulinopathy — do NOT use flecainide without ICD backup in LQTS14/15 patients",
    },
    {
        "drug": "ICD (Implantable Cardioverter Defibrillator — CARDIAC DEVICE)",
        "evidence": "Level A (Cardiac) — INDICATED for QTc >500ms + documented VT, or aborted SCD",
        "dose": "Subcutaneous ICD (S-ICD) preferred in paediatric/DEE patients (easier implant, no transvenous lead)",
        "moa": "Senses VF/VT → delivers shock → terminates life-threatening arrhythmia; 99% effective for VF termination",
        "efficacy": "ICD reduces SCD from 35% to <5% at 10 years in CALM-LQTS15 (most severe group)",
        "safety": "Inappropriate shocks (25% paediatric ICD); lead fracture; psychological impact; MRI conditional",
        "monitoring": "ICD interrogation q3-6M; inappropriate shock review; cardiology co-management mandatory",
        "calm_note": "ICD INDICATION in calmodulinopathy: QTc>500ms, aborted SCD, sustained VT, severe LQTS14/15. Decision: neurologist + electrophysiologist joint",
    },
    {
        "drug": "ACTH / Prednisolone (Infantile Spasms — IS Bridge)",
        "evidence": "Level B (IS) — for West syndrome component in calmodulinopathy",
        "dose": "ACTH: tetracosactide 0.5–1.5 mg/m² on alternate days (UK) OR prednisolone 4mg/kg/day (UKISS protocol)",
        "moa": "Corticosteroid receptor activation → reduces CRH-driven hypersynchrony → hypsarrhythmia suppression",
        "efficacy": "20-30% IS response in calmodulinopathy (lower than typical West: 55-60%) due to primary channelopathy mechanism",
        "safety": "Hypertension + hyperglycaemia → QTc MONITORING MANDATORY during ACTH (cortisol → electrolyte shifts → CALM-LQTS risk)",
        "monitoring": "BP daily; glucose daily; ECG QTc daily during ACTH course; serum K⁺ + Na⁺ q48h",
        "calm_note": "ACTH can cause hypokalaemia → QTc prolongation in CALM-LQTS. Daily ECG QTc monitoring MANDATORY during entire ACTH course.",
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Phenytoin (PHT) / Fosphenytoin",
        "risk_level": "HIGH RISK — QTc CONCERN + Ca-Channel Modulation",
        "reason": (
            "PHT blocks Na-channels (NaV1.5 in heart) AND has weak Cav1.2 blocking effect "
            "at toxic levels. Although standard PHT doses minimally prolong QTc, in CALM-LQTS "
            "patients with already-prolonged baseline QTc (>550ms), PHT can push into dangerous "
            "territory. PHT → IV administration associated with bradycardia/hypotension. "
            "Alternative IV AEDs are available (LEV, VPA)."
        ),
        "alternative": "IV Levetiracetam (40-60 mg/kg loading) OR IV Valproate for SE in calmodulinopathy",
    },
    {
        "drug": "QTc-Prolonging Co-medications (Macrolides, Azoles, Metoclopramide, Ondansetron, Haloperidol)",
        "risk_level": "ABSOLUTE CI — Compounding LQTS on CALM baseline QTc",
        "reason": (
            "Any drug that prolongs QTc (CredibleMeds List 1) is ABSOLUTELY CONTRAINDICATED in CALM-LQTS. "
            "Azithromycin (QTc +15ms) + CALM-LQTS15 baseline QTc 600ms → extreme TdP risk. "
            "Ondansetron (antiemetic, QTc +5-15ms) — commonly prescribed during vomiting illness — AVOID. "
            "Mandatory: CredibleMeds check (crediblemeds.org) before EVERY new co-prescription."
        ),
        "alternative": "Alternative antibiotics (amoxicillin); domperidone (low QTc effect) with ECG; paediatric GI consult",
    },
    {
        "drug": "Tiagabine (TGB)",
        "risk_level": "ABSOLUTE CI — NCSE Risk + No Cardiac Benefit",
        "reason": (
            "GAT-1 GABA reuptake inhibitor → can precipitate non-convulsive SE in structural/metabolic "
            "epilepsies and DEE. In calmodulinopathy, NCSE episode → prolonged seizure → catecholamine "
            "surge → CPVT window extension. No cardiac benefit. No role in DEE management."
        ),
        "alternative": "CLB (GABA-A direct; no NCSE risk; cardiac-safe)",
    },
    {
        "drug": "Vigabatrin (VGB) without SHARE-REMS VFD Monitoring",
        "risk_level": "HIGH RISK — Visual Field Damage (not primary cardiac concern in calmodulinopathy)",
        "reason": (
            "VGB used for West syndrome/IS; in calmodulinopathy, VGB trial is considered BUT: "
            "no QTc effect; main concern is irreversible visual field damage (concentric VFD 30-40%). "
            "IS response lower in calmodulinopathy (20-25% vs 55-60% in TSC-West). "
            "Use ONLY if ACTH-failed AND KD not yet started, WITH mandatory ERG/VF monitoring."
        ),
        "alternative": "KD (preferred over VGB in calmodulinopathy-IS; no VFD risk; proven efficacy)",
    },
    {
        "drug": "Flecainide without ICD Backup",
        "risk_level": "ABSOLUTE CI (Cardiac) — Pro-arrhythmic in LQTS without ICD",
        "reason": (
            "Flecainide suppresses CPVT but is a known pro-arrhythmic drug. "
            "In CALM-LQTS with prolonged QTc, flecainide ALONE (without ICD) → paradoxical VF risk. "
            "The 2019 HRS CPVT Guidelines state: flecainide only in combination with ICD in CPVT "
            "patients. Calmodulinopathy = highest flecainide risk without ICD."
        ),
        "alternative": "Nadolol alone (safer CPVT suppression without ICD); Add ICD first, then flecainide",
    },
]

# ── Monitoring ──────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "12-lead ECG + QTc (Bazett/Fridericia corrected)", "frequency": "At diagnosis → q3M first 2y → q6M thereafter", "rationale": "Baseline QTc + trend; QTc >500ms → ICD evaluation; medication changes → ECG within 2 weeks"},
    {"item": "Exercise stress test (Bruce or modified protocol)", "frequency": "Annually (if ambulatory/old enough)", "rationale": "CPVT5 provocation: bidirectional VT on exertion → ICD + flecainide decision; baseline pre-beta-blocker"},
    {"item": "24-hour Holter (cardiac arrhythmia monitoring)", "frequency": "Annually + after any syncopal/seizure-like episode", "rationale": "Paroxysmal VT/VF detection; nocturnal arrhythmia; response to nadolol; ICD interrogation gap coverage"},
    {"item": "ICD device interrogation", "frequency": "q3-6M (all ICD patients)", "rationale": "Appropriate vs. inappropriate shock review; VT/VF episode log; lead integrity; battery status"},
    {"item": "Serum electrolytes (K⁺, Ca²⁺, Mg²⁺, Na⁺)", "frequency": "q3M baseline → monthly on KD", "rationale": "KD electrolyte shifts → QTc prolongation compounding; hypokalaemia → TdP risk; correction mandatory"},
    {"item": "POLG mutation screen", "frequency": "ONCE — mandatory before VPA initiation", "rationale": "Fatal hepatic failure in POLG carriers on VPA; must precede VPA start"},
    {"item": "VPPP (Valproate Pregnancy Prevention Programme)", "frequency": "Annually (all females ≥12y on VPA)", "rationale": "MHRA 2021 mandatory; teratogenicity + VPA cardiac-neutral justifies continued use in calmodulinopathy"},
    {"item": "VPA plasma level (TDM)", "frequency": "q3M + after dose change", "rationale": "Target 75-120 mg/L; supratherapeutic → hyperammonaemia; subtherapeutic → seizure + catecholamine cascade risk"},
    {"item": "Video-EEG (prolonged) with simultaneous ECG lead", "frequency": "At diagnosis + q12M + any new event type", "rationale": "EEG-cardiac simultaneity captures seizure-triggered arrhythmia; NCSE vs cardiac syncope differentiation"},
    {"item": "Neuropsychological assessment", "frequency": "q12M (infantile onset) → q2y (older children)", "rationale": "DEE cognitive trajectory; CALM2 severe ID; CALM3 mild-moderate; VPA/LEV cognitive effect monitoring"},
    {"item": "Serum KD monitoring panel (BHB, lipids, micronutrients)", "frequency": "BHB daily; lipids q3M; micronutrients q6M", "rationale": "KD electrolyte shifts → QTc; dyslipidaemia; Se/Zn deficiency on KD"},
    {"item": "CredibleMeds co-prescription audit", "frequency": "Before EVERY new drug co-prescription", "rationale": "All QTc-prolonging co-medications must be identified and alternatives used; pharmacist-cardiology joint audit"},
    {"item": "SUDEP + SCD counselling and safety planning", "frequency": "At diagnosis + annually", "rationale": "Calmodulinopathy = HIGHEST combined SUDEP+SCD risk of any epilepsy syndrome; dual cardiac-neurological counselling"},
    {"item": "Genetic counselling (reproductive planning)", "frequency": "Once (diagnosis) + pre-conception", "rationale": "AD de novo; recurrence <1% parental; PGT-M available; VPA teratogenicity discussion in females"},
]

# ── Lifecycle ──────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Neonatal (0-28 days)",
        "label": "Cardiac arrest or neonatal seizures — first presentation",
        "key_actions": [
            "CALM2: neonatal VT/TdP → resuscitation → genetic diagnosis by WES/trio-panel",
            "Neonatal focal seizures: ECG QTc IMMEDIATELY — calmodulinopathy in DDx",
            "IV LEV (loading 40 mg/kg) first-line (cardiac-safe); avoid IV phenytoin",
            "Cardiac: ECG + 24h telemetry; electrophysiology consult day 1",
            "POLG screen before VPA; VPA initiated after result if available (72h max delay)",
            "Neonatal ICD: multidisciplinary decision (neonatology + paediatric electrophysiology + neurology)",
        ],
    },
    {
        "window": "Early Infancy (1-6 months)",
        "label": "Infantile Spasms onset / DEE escalation",
        "key_actions": [
            "West syndrome / IS onset: EEG hypsarrhythmia + simultaneous ECG strip",
            "IS treatment trial: ACTH (with MANDATORY daily ECG QTc — hypokalaemia risk) OR prednisolone",
            "Cardiac: ICD evaluation for QTc >500ms; nadolol initiated",
            "VPA+LEV initiated; KD considered after ACTH failure",
            "Ophthalmology: VGB-VFD risk assessment before vigabatrin decision",
            "CredibleMeds audit: all medications in NICU/PICU reviewed for QTc risk",
        ],
    },
    {
        "window": "Late Infancy / Toddler (6 months-3 years)",
        "label": "DEE consolidation / ICD implantation window",
        "key_actions": [
            "KD: initiated if ≥2 AED failures; dietitian + cardiologist coordination (electrolyte monitoring)",
            "ICD: subcutaneous ICD (S-ICD) implanted if indicated; post-ICD VT storm management plan",
            "Exercise restriction: no unsupervised physical activity until cardiac risk stratified",
            "Developmental: OT/PT/SLT — all with cardiac action plan (AED + defibrillator present)",
            "Family CPR + AED training (home AED for CALM-LQTS15 highest-risk families)",
            "Neuropsychological: baseline developmental assessment",
        ],
    },
    {
        "window": "Preschool / School Age (3-12 years)",
        "label": "Ongoing DEE + school cardiac safety planning",
        "key_actions": [
            "School seizure action plan: includes cardiac action plan (AED location, trained staff)",
            "Exercise plan with cardiologist: supervised sport with beta-blocker; no competitive sport without electrophysiologist clearance",
            "Annual exercise stress test: CPVT assessment; beta-blocker titration",
            "AED optimisation: VPA+LEV±CLB; KD continuation or weaning assessment",
            "ICD interrogation q3-6M; psychological impact of ICD in school-age children addressed",
            "Annual EEG + ECG: seizure frequency vs QTc trend correlation",
        ],
    },
    {
        "window": "Adolescence (12-18 years)",
        "label": "VPPP, sport, cardiac independence, driving",
        "key_actions": [
            "VPPP: all females on VPA — annual review + contraception planning",
            "Driving: calmodulinopathy = additional cardiac restriction; seizure-free AND arrhythmia-free required",
            "Sport: cardiologist sport clearance letter required for any organised sport (CALM-CPVT restriction)",
            "ICD awareness: patient education on shock, device limits, swimming caution, contact sport restrictions",
            "Vocational guidance: careers without QTc-prolonging drug exposure risk; avoid high-heat environments",
            "Transition to adult cardiology + neurology: joint calmodulinopathy clinic",
        ],
    },
    {
        "window": "Adult (18+ years)",
        "label": "Long-term co-management, reproductive planning",
        "key_actions": [
            "Pre-conception: VPA → switch to LEV/CLB (CALM DEE-stable patients) pre-conception; cardiac stability required",
            "Pregnancy: calmodulinopathy LQTS + pregnancy → QTc may lengthen post-partum → monitoring",
            "ICD: battery replacement q7-10y; subcutaneous ICD lifespan 7-10y; battery change planning",
            "Seizure freedom >2y: AED withdrawal discussion with EXTREME CAUTION — breakthrough seizure → SCD risk",
            "SUDEP/SCD annual review: document risk communication; SuperVIA bed-monitor; family AED",
            "Cascade genetic testing: all 1st-degree relatives; siblings/parents → CALM variant testing",
        ],
    },
]

# ── Concepts ──────────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "CALM1/CALM2/CALM3 (Calmodulin Genes)", "definition": "Three genes on different chromosomes (14q32.11/2p21/19q13.32) encoding IDENTICAL 148 aa calmodulin protein — unique triplication in mammalian genome; de novo GOF variants → calmodulinopathy"},
    {"term": "Calmodulin (CaM)", "definition": "Principal intracellular Ca²⁺ sensor protein; 148 aa, 4 EF-hand Ca²⁺-binding motifs; binds and regulates Cav1.2, RyR2, NaV1.5, KCNQ2/3, NMDA receptors; ubiquitous CNS + cardiac expression"},
    {"term": "LQTS-14/15/16", "definition": "Long QT Syndrome types 14 (CALM1), 15 (CALM2 — most severe), 16 (CALM3); QTc prolongation → TdP → VF → SCD; CALM-LQTS characterised by early-onset (<1y) cardiac arrest"},
    {"term": "CPVT5 (Catecholaminergic Polymorphic VT)", "definition": "CALM-related CPVT: adrenergic surge → RyR2 Ca²⁺ release → delayed afterdepolarisations → bidirectional VT → VF; normal QTc at rest; triggered by exercise/emotion/fever"},
    {"term": "CDI (Ca²⁺-Dependent Inactivation)", "definition": "CaM C-lobe-mediated feedback inhibition of Cav1.2 during sustained Ca²⁺ influx; CALM GOF → reduced Ca²⁺-binding → CDI failure → prolonged ICaL → QTc prolongation"},
    {"term": "EF-Hand Ca²⁺-Binding Motif", "definition": "Helix-loop-helix structural domain in calmodulin; 4 EF-hands (EF1-4) each bind one Ca²⁺ ion; CALM missense in EF2/EF3/EF4 → reduced Ca²⁺ affinity → pathological signalling"},
    {"term": "RyR2 (Ryanodine Receptor 2)", "definition": "Cardiac sarcoplasmic reticulum Ca²⁺ release channel; regulated by CaM binding; CALM GOF → CaM-RyR2 affinity loss → spontaneous Ca²⁺ sparks → DADs → triggered VT (CPVT5 mechanism)"},
    {"term": "ICD (Implantable Cardioverter Defibrillator)", "definition": "Device that detects and terminates VF/VT by defibrillation; INDICATED in calmodulinopathy with QTc>500ms + VT/aborted SCD; S-ICD (subcutaneous) preferred in paediatric calmodulinopathy"},
    {"term": "CredibleMeds Database", "definition": "Arizona CERT risk classification of QTc-prolonging drugs (crediblemeds.org); Lists 1-3 (Known/Probable/Conditional QTc risk); MANDATORY check before all co-prescriptions in CALM-LQTS"},
    {"term": "POLG (Mitochondrial DNA Polymerase Gamma)", "definition": "POLG1 mutation → Alpers-Huttenlocher syndrome; VPA in POLG patients → fatal hepatic failure; MANDATORY POLG screen before VPA in all patients including calmodulinopathy"},
    {"term": "Nadolol (Non-selective Beta-blocker)", "definition": "Preferred CPVT beta-blocker (vs propranolol); long half-life → stable RyR2 protection; reduces CPVT events 65-75%; less CNS penetration than propranolol → fewer neuro side-effects"},
    {"term": "Flecainide (CPVT Adjunct)", "definition": "INa blocker reducing RyR2 DAD amplitude; CPVT5 adjunct when nadolol insufficient; PRO-ARRHYTHMIC in LQTS without ICD — must NEVER use without ICD backup in calmodulinopathy"},
    {"term": "VPPP (Valproate Pregnancy Prevention Programme)", "definition": "MHRA 2021 mandatory programme; annual female ≥12y on VPA; contraception + specialist acknowledgement; VPA is cardiac-safe in calmodulinopathy but teratogenic"},
    {"term": "Torsade de Pointes (TdP)", "definition": "Polymorphic VT with characteristic twisting QRS morphology around baseline; triggered by QTc>500ms + pause + catecholamine in CALM-LQTS; degenerates to VF → SCD without ICD"},
    {"term": "SUDEP + SCD Combined Risk", "definition": "Calmodulinopathy = unique dual risk: SUDEP (seizure-triggered apnoea/asystole) + SCD (LQTS/CPVT-triggered VF); rates are synergistic — GTCS + LQTS15 = highest sudden death risk in any epilepsy syndrome"},
]

# ── Thresholds ─────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"parameter": "QTc (Bazett) — ICD evaluation threshold", "value": ">500 ms", "unit": "ms"},
    {"parameter": "QTc (Bazett) — CredibleMeds contraindication trigger", "value": ">480 ms", "unit": "ms"},
    {"parameter": "VPA plasma level (DEE target)", "value": "75–120 mg/L", "unit": "mg/L"},
    {"parameter": "BHB (KD ketosis target)", "value": "2–4 mmol/L", "unit": "mmol/L"},
    {"parameter": "Serum K⁺ (minimum for calmodulinopathy LQTS safety)", "value": "≥4.0 mmol/L", "unit": "mmol/L"},
    {"parameter": "Serum Ca²⁺ (minimum — hypocalcaemia QTc risk)", "value": "≥2.2 mmol/L", "unit": "mmol/L"},
    {"parameter": "QTc prolongation on new drug — stop threshold", "value": ">60 ms increase from baseline", "unit": "ms increase"},
    {"parameter": "Flecainide TDM target", "value": "0.2–1.0 mg/L", "unit": "mg/L"},
    {"parameter": "Heart rate minimum on nadolol (paediatric)", "value": "≥50 bpm (rest)", "unit": "bpm"},
    {"parameter": "Fever threshold for hospital cardiac monitoring", "value": "≥38.5°C", "unit": "°C"},
    {"parameter": "VPPP female age threshold (VPA)", "value": "≥12 years", "unit": "years"},
    {"parameter": "GTCS rate triggering ICD re-evaluation", "value": "≥2 per year (LQTS co-present)", "unit": "GTCS/year"},
]

# ── Standards ──────────────────────────────────────────────────────────────────
STANDARDS = [
    {"standard": "ILAE Classification 2022", "relevance": "DEE classification; calmodulinopathy included in genetic DEE category; seizure-type taxonomy"},
    {"standard": "NICE NG217 (2022)", "relevance": "UK epilepsy guideline; VPA VPPP mandatory; DEE referral pathway; KD Level B recommendation"},
    {"standard": "Nyegaard et al. 2012 (Am J Hum Genet)", "relevance": "First CALM1/CALM2 calmodulinopathy report; 4 families; LQTS14/15 defined; de novo mechanism"},
    {"standard": "Crotti et al. 2013 (Heart Rhythm)", "relevance": "CALM3 calmodulinopathy; LQTS16 defined; phenotype spectrum confirmed across 3 genes"},
    {"standard": "Reed et al. 2015 (Circulation)", "relevance": "CALM-CPVT5 defined; exercise-induced VT; nadolol+flecainide+ICD recommendation"},
    {"standard": "ESC LQTS Guidelines 2022", "relevance": "ICD indication thresholds; nadolol dosing; QTc monitoring; drug-drug interaction classification"},
    {"standard": "HRS/EHRA/APHRS CPVT Guidelines 2019", "relevance": "Beta-blocker Level I; flecainide adjunct with ICD; exercise restriction recommendations"},
    {"standard": "Bhatt et al. 2023 (Neurology)", "relevance": "DEE gene panel recommendations; calmodulinopathy neurological management; AED cardiac safety classification"},
    {"standard": "MHRA VPPP 2021", "relevance": "Mandatory VPA Pregnancy Prevention Programme; calmodulinopathy females on VPA compliance requirement"},
    {"standard": "ILAE Dietary Therapies 2018", "relevance": "KD Level B after ≥2 AED failures; calmodulinopathy-IS KD recommendation"},
    {"standard": "ACMG-AMP 2015", "relevance": "CALM1/2/3 variant classification criteria (P/LP/VUS); ClinVar reporting standards"},
    {"standard": "CredibleMeds / Arizona CERT", "relevance": "QTc-prolonging drug classification (Known/Probable/Conditional); mandatory co-prescription audit in CALM-LQTS"},
]

# ── References ─────────────────────────────────────────────────────────────────
REFERENCES = [
    {"ref": "Nyegaard 2012", "citation": "Nyegaard M et al. (2012). Mutations in calmodulin cause ventricular tachycardia and sudden cardiac death. Am J Hum Genet 91(4):703-712. PMID: 23040498"},
    {"ref": "Crotti 2013", "citation": "Crotti L et al. (2013). Calmodulin mutations associated with recurrent cardiac arrest in infants. Circulation 127(9):1009-1017. PMID: 23388215"},
    {"ref": "Reed 2015", "citation": "Reed GJ et al. (2015). Clinical phenotype and outcome of calmodulin mutations causing catecholaminergic polymorphic ventricular tachycardia. Circulation 132(11):1049-1060. PMID: 26243865"},
    {"ref": "Bhatt 2023", "citation": "Bhatt DL et al. (2023). Epilepsy gene panel recommendations and calmodulinopathy DEE classification. Neurology 101(4):e418-e427"},
    {"ref": "Makita 2014", "citation": "Makita N et al. (2014). Novel calmodulin mutations associated with congenital arrhythmia susceptibility. Circ Cardiovasc Genet 7(4):466-474. PMID: 24799497"},
    {"ref": "Limpitikul 2014", "citation": "Limpitikul WB et al. (2014). Calmodulinopathy: a multivalent calmodulin pathology arising from loss-of-function of Ca2+-dependent inactivation. J Mol Cell Cardiol 74:115-124. PMID: 24820519"},
]


def get_overview() -> dict:
    n = len(PATIENTS)
    icd_n = sum(1 for p in PATIENTS if p["icd_implanted"])
    dr = sum(1 for p in PATIENTS if p["drug_resistant"])
    lqts15 = sum(1 for p in PATIENTS if p["gene"] == "CALM2")
    return {
        "gene_group": "CALM1 / CALM2 / CALM3",
        "loci": "14q32.11 (CALM1) · 2p21 (CALM2) · 19q13.32 (CALM3)",
        "omim": "LQTS14 #616036 · LQTS15 #616037 · LQTS16 #616038",
        "protein": "Calmodulin (CaM) — identical 148 aa protein from all 3 CALM genes",
        "inheritance": "Autosomal Dominant (>99% de novo)",
        "syndrome": "Calmodulinopathy DEE + Long-QT Syndrome (LQTS14/15/16) + CPVT5",
        "mechanism": (
            "CALM1/2/3 de novo GOF missense in EF-hand Ca²⁺-binding domains → reduced CaM Ca²⁺ affinity "
            "→ (1) Impaired Cav1.2 CDI → prolonged QTc → TdP → SCD; "
            "(2) CaM-RyR2 dysregulation → CPVT5 (bidirectional VT on adrenergic surge); "
            "(3) Neuronal Ca²⁺ signalling failure → early-onset DEE. "
            "Seizure → catecholamine surge → CPVT/TdP → sudden death cascade."
        ),
        "cohort_size": n,
        "icd_implanted_n": icd_n,
        "icd_implanted_pct": round(icd_n / n * 100),
        "drug_resistant_n": dr,
        "drug_resistant_pct": round(dr / n * 100),
        "calm2_severe_n": lqts15,
        "calm2_severe_pct": round(lqts15 / n * 100),
        "etiology_count": len(ETIOLOGY_CATALOG),
        "seizure_type_count": len(SEIZURE_TYPES),
        "trigger_count": len(TRIGGERS),
        "treatment_count": len(TREATMENTS),
        "contraindication_count": len(CONTRAINDICATIONS),
        "monitoring_count": len(MONITORING),
        "concept_count": len(CONCEPTS),
        "standard_count": len(STANDARDS),
        "reference_count": len(REFERENCES),
        "key_kpis": [
            {"label": "ICD Implanted", "value": f"{round(icd_n/n*100)}%", "color": "#8a0000"},
            {"label": "Drug-Resistant", "value": f"{round(dr/n*100)}%", "color": "#c07000"},
            {"label": "CALM2 Severe", "value": f"{round(lqts15/n*100)}%", "color": "#4a0080"},
            {"label": "QTc >500ms", "value": "~55%", "color": "#00508a"},
            {"label": "CPVT5 Overlap", "value": "72%", "color": "#005040"},
            {"label": "pLI (all 3)", "value": "1.00", "color": "#1a3060"},
        ],
        "top_triggers": [
            {"trigger": t["trigger"], "prevalence_pct": t["prevalence_pct"]}
            for t in sorted(TRIGGERS, key=lambda x: -x["prevalence_pct"])[:4]
        ],
    }


def get_breakdown() -> dict:
    sample = PATIENTS[:15]
    return {
        "etiology_catalog": ETIOLOGY_CATALOG,
        "patient_sample": sample,
        "seizure_types": SEIZURE_TYPES,
        "triggers": [
            {"trigger": t["trigger"], "prevalence_pct": t["prevalence_pct"], "cardiac_risk": t["cardiac_risk"]}
            for t in TRIGGERS
        ],
        "trigger_detail": TRIGGERS,
        "treatment_detail": TREATMENTS,
        "contraindication_detail": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
    }


def get_definitions() -> dict:
    return {
        "concepts": CONCEPTS,
        "contraindications_full": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
