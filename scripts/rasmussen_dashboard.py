"""
Rasmussen's Encephalitis Dashboard
====================================
41-patient cohort · Rare progressive unilateral encephalitis · T-cell mediated autoimmune
Pathognomonic: unilateral cortical atrophy + EPC + progressive hemiplegia.
Definitive treatment: functional hemispherectomy / hemispherotomy (Engel I 52–65%).
"""

import random
from datetime import datetime

SEED = 4343
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ──────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "T-cell mediated autoimmune (GluR3/AMPA)",
        "n": 18, "pct": 45,
        "category": "Autoimmune-T-cell",
        "mechanism": (
            "CD8+ cytotoxic T-lymphocyte infiltration is the primary pathological driver. "
            "GluR3 (GluA3 subunit of AMPA receptors) auto-antibodies are found in a subset; "
            "however, T-cell-mediated neuronal destruction, not antibody titre, correlates with "
            "disease severity and progression. CD8+ lymphocytes form immunological synapses with "
            "MHC-I-expressing neurons, releasing perforin/granzyme B to induce neuronal apoptosis. "
            "The process is unilateral, starting in peri-sylvian cortex and spreading to involve "
            "the entire hemisphere over 6–24 months."
        ),
        "eeg_correlate": (
            "Unilateral delta slowing over affected hemisphere; focal epileptiform discharges "
            "(sharp waves / spikes) over frontal or rolandic region ipsilateral to atrophy; "
            "Epilepsia Partialis Continua on prolonged EEG: continuous focal muscle jerks with "
            "time-locked EEG discharge (not always visible on scalp EEG — may require LFP)."
        ),
        "mri_finding": (
            "T2 FLAIR hyperintensity in affected hemisphere (cortex + subcortical U-fibres); "
            "progressive unilateral cortical atrophy on serial MRI; caudate atrophy; "
            "contralateral hemisphere normal — asymmetry is pathognomonic."
        ),
        "clinical_note": (
            "GluR3 antibody titre does NOT reliably track disease activity. "
            "Diagnosis rests on clinical triad: focal seizures/EPC + progressive hemiplegia + "
            "unilateral cortical atrophy on MRI. Bien 2013 criteria require ≥2 of 3 clinical "
            "features + histopathology (T-cell infiltration + microglial nodules + astrogliosis)."
        ),
    },
    {
        "etiology": "Unknown autoimmune (seronegative Rasmussen)",
        "n": 10, "pct": 25,
        "category": "Autoimmune-Seronegative",
        "mechanism": (
            "Histopathology confirms classic Rasmussen pattern — CD8+ T-cell infiltration, "
            "microglial nodules, neuronal loss, reactive astrogliosis — but no identifiable "
            "circulating antibody (GluR3, GluD2, NMDAR, LGI1, CASPR2 all negative). "
            "Likely represents cellular autoimmunity where the target antigen remains "
            "unidentified, or where antibody production is confined to the CNS compartment "
            "(intrathecal synthesis not captured by serum testing). "
            "Clinical course identical to seropositive subtype."
        ),
        "eeg_correlate": (
            "Unilateral hemispheric delta; focal spikes in motor strip; EPC pattern on "
            "prolonged EMG-EEG polygraph; inter-ictal background suppression ipsilateral to atrophy."
        ),
        "mri_finding": (
            "Unilateral progressive atrophy — frontal > parietal > temporal involvement; "
            "T2 FLAIR signal in cortex and underlying white matter; late-stage: "
            "severe ipsilateral hemiatrophy."
        ),
        "clinical_note": (
            "Seronegative status does NOT exclude Rasmussen's. Brain biopsy remains the gold "
            "standard when clinical/MRI criteria are met but serology is negative. "
            "Treat as standard Rasmussen — immunotherapy + surgical planning."
        ),
    },
    {
        "etiology": "GluD2 / anti-GluRdelta2 antibody",
        "n": 3, "pct": 7,
        "category": "Autoimmune-GluD2",
        "mechanism": (
            "Antibodies against GluRdelta2 (GRID2 gene product), a glutamate receptor delta "
            "subunit predominantly expressed in cerebellar Purkinje cells but also in cortical "
            "interneurons. GluD2 autoimmunity causes AMPA receptor internalisation and synaptic "
            "loss. Less common than GluR3/AMPA subtype; pathological findings on biopsy are "
            "indistinguishable from the more common T-cell mediated form."
        ),
        "eeg_correlate": (
            "Focal EEG slowing; focal epileptiform discharges; similar EPC pattern; "
            "occasional additional cerebellar signs if GRID2 expression at cerebellum affected."
        ),
        "mri_finding": (
            "Unilateral hemispheric atrophy with T2/FLAIR changes; pattern similar to other "
            "Rasmussen subtypes; cerebellar involvement in rare cases."
        ),
        "clinical_note": (
            "GluD2 antibody testing available at specialist neuroimmunology laboratories. "
            "Consider in Rasmussen-phenotype cases with unusual cerebellar signs. "
            "Plasma exchange + rituximab used for antibody clearance; surgical planning unchanged."
        ),
    },
    {
        "etiology": "Viral-trigger hypothesis (CMV/EBV/HSV-associated immune reaction)",
        "n": 6, "pct": 15,
        "category": "Viral-Triggered-Autoimmune",
        "mechanism": (
            "A viral trigger (CMV, EBV, HSV — most commonly reported) initiates an aberrant "
            "immune response through molecular mimicry or bystander activation. Crucially, "
            "virus is NOT found in brain tissue at biopsy — the immune reaction is the primary "
            "driver, not active viral replication. CMV pp65 antigenemia or EBV serology may be "
            "positive at disease onset. The viral-trigger hypothesis explains the 1–2 month "
            "prodrome of febrile illness before the first seizure in some cases."
        ),
        "eeg_correlate": (
            "EEG indistinguishable from other Rasmussen subtypes; unilateral delta + focal "
            "spikes; EPC may be more prominent in early disease phase."
        ),
        "mri_finding": (
            "T2 FLAIR signal changes early (cortical swelling in acute phase); then progressive "
            "unilateral atrophy. DWI restriction in cortex during acute inflammatory phase."
        ),
        "clinical_note": (
            "Antiviral therapy (aciclovir/ganciclovir) is NOT effective — the virus is NOT "
            "replicating in CNS. Immunotherapy (IVIG, steroids, plasma exchange) is the "
            "appropriate early treatment. Viral serology at diagnosis to document potential trigger."
        ),
    },
    {
        "etiology": "NMDAR-coexisting antibodies (Rasmussen + NMDAR overlap)",
        "n": 3, "pct": 8,
        "category": "Autoimmune-Overlap",
        "mechanism": (
            "Rare overlap where classical Rasmussen histopathology coexists with detectable "
            "NMDA receptor antibodies (anti-GluN1/NR1). This may represent dual autoimmune "
            "targeting or secondary antibody production following neuronal damage exposing "
            "NMDAR epitopes. The NMDAR-antibody component may contribute to movement disorders "
            "or psychiatric features atypical for standard Rasmussen."
        ),
        "eeg_correlate": (
            "Unilateral focal pattern typical of Rasmussen; additional delta-brush pattern "
            "possible (if significant NMDAR antibody burden); EPC present."
        ),
        "mri_finding": (
            "Unilateral atrophy (Rasmussen component); may show additional T2 signal in "
            "hippocampus or thalamus if NMDAR component is active."
        ),
        "clinical_note": (
            "Test NMDAR antibodies (serum AND CSF) in Rasmussen cases with atypical features. "
            "NMDAR-overlay cases may have enhanced response to steroids + IVIG + rituximab. "
            "Surgical planning: hemispherectomy remains definitive for the Rasmussen component."
        ),
    },
]

# ── Patient Data ──────────────────────────────────────────────────────────────
ETIOLOGIES_WEIGHTED = (
    ["T-cell mediated autoimmune (GluR3/AMPA)"] * 18
    + ["Unknown autoimmune (seronegative Rasmussen)"] * 10
    + ["GluD2 / anti-GluRdelta2 antibody"] * 3
    + ["Viral-trigger hypothesis (CMV/EBV/HSV-associated immune reaction)"] * 6
    + ["NMDAR-coexisting antibodies (Rasmussen + NMDAR overlap)"] * 3
)

TREATMENTS_LIST = [
    "IVIG monthly",
    "Plasma Exchange + IVIG",
    "Rituximab + IVIG",
    "Steroids + IVIG",
    "Tacrolimus + IVIG",
    "Natalizumab",
    "Post-hemispherectomy (AED-free)",
    "Post-hemispherotomy + LEV",
    "CBZ + IVIG",
    "LEV + CLB + IVIG",
]

SEIZURE_TYPES_LIST = [
    "Focal Motor / EPC",
    "Focal Impaired Awareness",
    "Hemiclonic / GTCS",
    "Subtle Myoclonic (EPC variant)",
]

CTRL = ["Seizure-free (post-surgical)", "Partial control", "Drug-resistant (EPC active)"]
CTRL_W = [0.30, 0.35, 0.35]
PHASES = ["Prodrome", "Rapid Deterioration", "Plateau (Hemiplegia)", "Post-surgical Recovery", "Rehabilitation", "Independent"]
PHASE_W = [0.10, 0.25, 0.20, 0.20, 0.15, 0.10]
EEG_PATTERNS = [
    "Unilateral delta + focal spikes (rolandic)",
    "EPC pattern + hemispheric slowing",
    "Unilateral delta + continuous focal discharges",
    "Post-surgical — improved background, residual focal slowing",
    "Hemispheric suppression + focal epileptiform",
]
MRI_FINDINGS = [
    "Unilateral T2 FLAIR + cortical atrophy (early)",
    "Progressive hemispheric atrophy (peri-sylvian)",
    "Severe hemiatrophy — caudate + cortex",
    "Post-hemispherectomy changes",
    "T2 signal change + subcortical U-fibre involvement",
]
HEMIPLEGIA_SIDES = ["Left", "Right"]

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
        onset = random.randint(3, 14)
        age = random.randint(onset + 1, onset + 12)
        tx = random.choice(TREATMENTS_LIST)
        st = random.choice(SEIZURE_TYPES_LIST)
        epc_active = ctrl != "Seizure-free (post-surgical)" and random.random() > 0.35
        hemi_side = random.choice(HEMIPLEGIA_SIDES)
        eeg_pat = random.choice(EEG_PATTERNS)
        mri_find = random.choice(MRI_FINDINGS)
        pts.append({
            "id": f"RE-{i:03d}",
            "age": age,
            "sex": "M" if random.random() < 0.50 else "F",
            "onset_age": onset,
            "etiology": et,
            "seizure_type": st,
            "current_treatment": tx,
            "disease_phase": phase,
            "EPC": epc_active,
            "hemiplegia_side": hemi_side,
            "eeg_pattern": eeg_pat,
            "mri_finding": mri_find,
            "seizure_control": ctrl,
        })
    return pts


PATIENTS = _make_patients()

# ── Seizure Types (detailed) ──────────────────────────────────────────────────
SEIZURE_TYPES_DETAIL = [
    {
        "type": "Focal Motor Seizures / Epilepsia Partialis Continua (EPC)",
        "freq_pct": 100,
        "description": (
            "Pathognomonic for Rasmussen's Encephalitis. EPC is defined as continuous or "
            "near-continuous focal muscle jerks (myoclonia) of cortical origin, lasting hours "
            "to years without loss of consciousness. Involves one body part (face, hand, arm) "
            "contralateral to the affected hemisphere. Resistant to all AEDs — reflects "
            "ongoing T-cell-mediated neuronal destruction in the motor cortex. "
            "EPC represents the hallmark seizure type in >90% of Rasmussen cases "
            "and is the primary driver of hemispherectomy referral."
        ),
        "eeg_correlate": (
            "Focal continuous or semi-rhythmic spikes/sharp waves over motor strip (C3/C4 or "
            "neighbouring electrodes). The EEG discharge may be subtle or not visible on "
            "standard scalp EEG — EMG polygraph is mandatory. Background: unilateral delta "
            "slowing with occasional breach rhythm over affected hemisphere."
        ),
        "duration_sec": "Minutes to years (continuous)",
        "clinical_tip": (
            "EPC does NOT respond to AEDs — this is an important diagnostic clue. "
            "If EPC persists >2 hours, emergent immunotherapy (IV methylprednisolone) indicated. "
            "Time to hemispherectomy decision: disease plateau (established hemiplegia). "
            "Functional MRI language lateralisation essential before surgery if EPC is dominant-hemisphere."
        ),
    },
    {
        "type": "Focal Impaired Awareness Seizures (FIAS)",
        "freq_pct": 80,
        "description": (
            "Focal seizures with impaired awareness arising from the affected hemisphere. "
            "Semiology depends on the lobe primarily affected: peri-sylvian Rasmussen produces "
            "oro-facial automatisms; frontal onset produces complex motor automatisms; "
            "occipital onset gives visual symptoms. These seizures may escalate from EPC. "
            "Duration typically 60–180 seconds; post-ictal hemiparesis (Todd's paresis) "
            "proportional to cortical involvement."
        ),
        "eeg_correlate": (
            "Focal rhythmic theta → delta evolving discharge from affected hemisphere. "
            "Post-ictal unilateral suppression. Inter-ictal: hemispheric delta slowing + "
            "focal spikes ipsilateral to atrophy."
        ),
        "duration_sec": "60–180 s",
        "clinical_tip": (
            "Track Todd's paresis duration at each visit — increasing post-ictal hemiparesis "
            "duration signals progressive motor cortex destruction. "
            "Seizure action plan: buccal midazolam 0.3 mg/kg for seizures >5 minutes."
        ),
    },
    {
        "type": "Hemiclonic Seizures / Secondarily Generalised (GTCS)",
        "freq_pct": 60,
        "description": (
            "Focal clonic seizures involving the contralateral arm/leg (hemiclonic) that may "
            "secondarily generalise to bilateral tonic-clonic. Hemiclonic pattern (ipsilateral "
            "deviation of head and eyes) localises the epileptogenic zone to the contralateral "
            "hemisphere. Secondary generalisation indicates spread via corpus callosum — "
            "callosotomy can reduce GTCS frequency when hemispherectomy is delayed."
        ),
        "eeg_correlate": (
            "Focal rhythmic discharge (alpha frequency) evolving over affected hemisphere → "
            "bilateral spread with symmetric synchronous polyspike during generalisation. "
            "Post-ictal: diffuse suppression with hemispheric asymmetry."
        ),
        "duration_sec": "60–300 s",
        "clinical_tip": (
            "IV lorazepam / buccal midazolam for GTCS >5 minutes. "
            "Frequent hemiclonic → GTCS escalation suggests disease progression — "
            "reassess surgical candidacy urgently. Avoid prolonged high-dose benzodiazepines "
            "(tolerance develops rapidly in Rasmussen)."
        ),
    },
    {
        "type": "Subtle Myoclonic Jerks (EPC variant)",
        "freq_pct": 45,
        "description": (
            "Fine rhythmic myoclonic jerks of a single finger, toe, or perioral region, "
            "representing a low-amplitude EPC variant. May go unrecognised as seizure activity "
            "for months — misattributed to tics or tremor. EMG-EEG polygraph with time-locked "
            "cortical discharge confirms epileptic nature. Presents particularly at disease "
            "onset (prodromal phase) before overt focal clonic seizures appear."
        ),
        "eeg_correlate": (
            "Subtle focal spike or sharp wave time-locked to myoclonic jerk on EMG polygraph. "
            "Standard scalp EEG may be normal — EMG polygraph mandatory for diagnosis. "
            "Jerk-locked back-averaging (JLA) can reveal cortical correlate when EEG is inconclusive."
        ),
        "duration_sec": "Continuous (hours–years at low amplitude)",
        "clinical_tip": (
            "Jerk-locked averaging and 24-hour EMG-EEG polygraph are the gold-standard "
            "investigations. Subtle EPC must prompt urgent Rasmussen workup: MRI with "
            "FCD-protocol + contrast, CSF analysis, autoantibody panel. "
            "Do not dismiss as benign tic — unilateral EPC in a child is a red flag."
        ),
    },
]

# ── Triggers ──────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fever / intercurrent infection",
        "pct": 80,
        "mechanism": (
            "Systemic infection upregulates pro-inflammatory cytokines (IL-6, TNF-alpha, IFN-gamma) "
            "that cross-activate resident CNS immune cells. Microglial activation amplifies "
            "T-cell infiltration in already-inflamed hemisphere. Fever directly lowers seizure "
            "threshold via GABA-A receptor thermosensitivity."
        ),
        "management": (
            "Aggressive antipyretic management (paracetamol + ibuprofen). Seizure action plan "
            "for infection-triggered escalations. Consider IV methylprednisolone pulse if "
            "febrile illness causes clear EPC escalation. Pre-emptive IVIG during intercurrent illness."
        ),
    },
    {
        "trigger": "Sleep deprivation",
        "pct": 70,
        "mechanism": (
            "Chronic sleep deprivation impairs regulatory T-cell function, reducing "
            "central immune surveillance. Sleep deprivation also reduces inhibitory reserve "
            "in the already-compromised hemisphere. Homeostatic sleep pressure compounds "
            "cortical hyperexcitability from ongoing inflammatory destruction."
        ),
        "management": (
            "Strict sleep hygiene protocol. Sedating adjunct (CLB 5–10 mg nocte) may improve "
            "sleep quality. Monitor nocturnal seizure frequency with seizure diary. "
            "Arrange school accommodations for fatigue-related cognitive burden."
        ),
    },
    {
        "trigger": "Missed immunotherapy dose",
        "pct": 65,
        "mechanism": (
            "IVIG / rituximab / tacrolimus sub-therapeutic levels allow rebound T-cell activity "
            "and inflammatory surge. The Rasmussen disease process is continuous — "
            "any gap in immunotherapy creates an inflammatory window that accelerates "
            "neuronal loss and worsens EPC."
        ),
        "management": (
            "IVIG infusion calendar with pharmacy reminders. Rituximab cycle tracking. "
            "Provide emergency IVIG protocol at local hospital for missed doses. "
            "Carer education: missed immunotherapy in Rasmussen ≠ missed AED — both critical."
        ),
    },
    {
        "trigger": "Stress / psychological burden",
        "pct": 55,
        "mechanism": (
            "Hypothalamic-pituitary-adrenal (HPA) axis dysregulation from chronic stress "
            "elevates cortisol, which paradoxically suppresses regulatory T-cells (Treg) "
            "while activating effector CD8+ lymphocytes. This creates a pro-inflammatory "
            "CNS environment that worsens T-cell-mediated neuronal destruction."
        ),
        "management": (
            "Psychological support for patient and family (Rasmussen carries catastrophic "
            "prognosis; caregiver burnout common). Paediatric neuropsychology referral. "
            "Mindfulness and pacing strategies. Family counselling regarding surgical prognosis."
        ),
    },
    {
        "trigger": "Physical fatigue / exertion",
        "pct": 60,
        "mechanism": (
            "Physical exertion increases body temperature and metabolic demand, reducing "
            "cortical inhibitory reserve. Lactic acidosis from exercise further impairs "
            "Na+/K+-ATPase pump function in already-compromised neurons of the affected hemisphere. "
            "EPC frequency typically increases 2–4 hours post-exercise."
        ),
        "management": (
            "Activity pacing plan with occupational therapist. Avoid overheating during exercise. "
            "Cooling vest for outdoor activities. Seizure action plan for school PE teachers. "
            "Post-exercise rest period of ≥30 minutes with supervision."
        ),
    },
    {
        "trigger": "Bright lights / photic stimulation",
        "pct": 15,
        "mechanism": (
            "Visual cortex hyperexcitability due to local inflammatory changes can rarely cause "
            "photic-triggered focal occipital discharges that propagate to motor cortex and "
            "worsen EPC. Uncommon in Rasmussen (unlike photosensitive GGE). "
            "Relevant when posterior cortex is predominantly affected."
        ),
        "management": (
            "Polarised lenses; screen filter. Avoid strobe environments. "
            "Photic driving test on EEG to confirm sensitivity before recommending restrictions."
        ),
    },
    {
        "trigger": "Physical exertion (action myoclonus component)",
        "pct": 40,
        "mechanism": (
            "Voluntary motor activity directly recruits cortical motor neurons in the "
            "EPC-affected zone, increasing discharge frequency. Action myoclonus pattern "
            "— worsening of EPC with limb movement — reflects cortical hyperexcitability "
            "in the primary motor cortex."
        ),
        "management": (
            "Physiotherapy focused on compensatory movement patterns. "
            "OT assessment for adaptive equipment. Action-myoclonus-specific EMG-EEG mapping "
            "to localise zone before surgical planning."
        ),
    },
    {
        "trigger": "Catamenial (hormonal) — females",
        "pct": 20,
        "mechanism": (
            "Oestrogen surge in peri-menstrual phase increases glutamatergic excitability "
            "and NMDA receptor expression, further activating already-inflamed cortical circuits. "
            "Progesterone withdrawal in late luteal phase reduces neurosteroid-mediated "
            "GABA-A receptor potentiation."
        ),
        "management": (
            "Track seizure-menstrual calendar. Clobazam cyclical dosing (10–14 day course "
            "peri-menstrually). Gynaecology referral for hormonal contraception consideration "
            "if catamenial pattern is confirmed."
        ),
    },
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Hemispherectomy / Hemispherotomy (Functional Hemispherectomy)",
        "evidence": "Level A (definitive)",
        "evidence_ref": "Varadkar 2014 Brain; Bien 2013 Neuropathol Appl Neurobiol; European RE Consensus 2017",
        "dose_adult": "Surgical procedure — timing: disease plateau (established hemiplegia, stable MRI atrophy)",
        "dose_paed": "Hemispherotomy (disconnection, not resection) preferred in children — lower morbidity than anatomical hemispherectomy",
        "moa": (
            "Disconnection or removal of the epileptogenic hemisphere eliminates the source of "
            "EPC and focal seizures. Functional hemispherotomy disconnects corticospinal and "
            "interhemispheric pathways while preserving brain volume, reducing post-operative "
            "CSF-shift complications. Neuroplasticity, particularly for language, is remarkable "
            "if surgery is performed in children <8 years — language can transfer to contralateral "
            "hemisphere. Motor neuroplasticity is limited — contralateral hemiplegia is permanent."
        ),
        "efficacy": "Engel I (seizure-free) 52–65%; Engel II 15%; Engel III–IV 20%. EPC resolves in >90% post-surgery.",
        "safety": (
            "Permanent contralateral hemiplegia (already present pre-operatively in most); "
            "contralateral homonymous hemianopia; language deficits if dominant hemisphere "
            "(pre-surgical language fMRI + Wada test mandatory); CSF circulation issues (5%); "
            "post-operative infection; siderosis (late, after anatomical hemispherectomy)."
        ),
        "monitoring": (
            "Pre-surgical: language fMRI + Wada test (dominant hemisphere); "
            "SEEG/ECoG to confirm unilateral seizure onset; "
            "neuropsychological battery (language, memory, IQ). "
            "Post-surgical: MRI at 3M and 12M; neuropsychology 6M post-op; "
            "physiotherapy + speech therapy rehabilitation. Annual follow-up indefinitely."
        ),
        "contraindication_note": (
            "Bilateral disease (rare) — confirms NO hemisphere spared from atrophy; "
            "severe systemic illness precluding general anaesthesia; "
            "dominant hemisphere surgery when language cannot be shown to have transferred."
        ),
    },
    {
        "drug": "IVIG (Intravenous Immunoglobulin)",
        "evidence": "Level B",
        "evidence_ref": "Granata 2014 Epilepsia; Bien 2002 Brain; FDA IVIG (off-label Rasmussen)",
        "dose_adult": "0.4 g/kg/day × 5 days (2 g/kg total per course); monthly maintenance 1–2 g/kg",
        "dose_paed": "Same weight-based dosing; IV access (consider PICC line for monthly cycles); infuse over 4–8 hours per session",
        "moa": (
            "Polyclonal IgG acts via multiple mechanisms: Fc-receptor blockade on macrophages "
            "reduces antibody-mediated destruction; neutralisation of pathogenic autoantibodies "
            "(GluR3, GluD2); inhibition of complement cascade; regulatory T-cell (Treg) "
            "upregulation via IL-10 pathway; idiotypic network modulation. "
            "IVIG slows disease progression but rarely halts it completely — used as "
            "bridging therapy while awaiting rituximab effect or surgical planning."
        ),
        "efficacy": (
            "50–60% show temporary seizure reduction; rarely produces sustained remission. "
            "Best used as early disease control and adjunct to plasma exchange/rituximab."
        ),
        "safety": (
            "Headache, fever, nausea (infusion-related — pre-medicate with paracetamol + "
            "chlorphenamine); thromboembolic risk (especially with high-dose); IgA deficiency "
            "check before first dose (anaphylaxis risk); haemolytic anaemia (rare); "
            "aseptic meningitis."
        ),
        "monitoring": (
            "IgA level before first dose; serum IgG trough levels (target >7 g/L); "
            "renal function (sucrose-containing IVIG formulations — nephrotoxic); "
            "CBC for haemolytic anaemia (Coombs test if unexplained anaemia); "
            "blood viscosity in high-risk thrombosis patients."
        ),
        "contraindication_note": None,
    },
    {
        "drug": "Plasma Exchange / Plasmapheresis",
        "evidence": "Level B",
        "evidence_ref": "Granata 2014 Epilepsia; Bien 2002 Brain; Varadkar 2014 Brain",
        "dose_adult": "5–10 sessions (every other day); 1–1.5 plasma volumes exchanged per session",
        "dose_paed": "5–10 sessions via central venous catheter (CVC); paediatric volume calculations essential; albumin replacement standard",
        "moa": (
            "Mechanical removal of circulating pathogenic autoantibodies (GluR3, GluD2, NMDAR), "
            "cytokines (IL-6, TNF-alpha), complement components, and activated lymphocytes "
            "from plasma. Provides rapid (within 48–72 hours) reduction in antibody-mediated "
            "inflammation. Effect is transient (weeks) — used for acute disease flares or as "
            "bridging to rituximab. Often combined with IVIG immediately post-PEX to "
            "prevent rebound antibody rise."
        ),
        "efficacy": (
            "60–70% show EPC reduction during the acute PEX course; "
            "seizure improvement may last 4–12 weeks. Rarely produces sustained benefit alone."
        ),
        "safety": (
            "CVC insertion risks (pneumothorax, infection, thrombosis); "
            "hypotension during apheresis; hypocalcaemia (citrate anticoagulation); "
            "clotting factor depletion (monitor PT/aPTT); infection risk (immunoglobulin depletion); "
            "haemodynamic instability in small children."
        ),
        "monitoring": (
            "Daily CBC + coagulation (PT/aPTT/fibrinogen) during course; "
            "calcium + magnesium (citrate-related hypocalcaemia); "
            "blood pressure monitoring during each session; "
            "IgG levels post-PEX (depleted — use IVIG 1 g/kg immediately after PEX course)."
        ),
        "contraindication_note": None,
    },
    {
        "drug": "Rituximab (anti-CD20 B-cell depletion)",
        "evidence": "Level B",
        "evidence_ref": "Bien 2013 Neuropathol; Granata 2014 Epilepsia; Thilo 2009 Neurology",
        "dose_adult": "375 mg/m² IV × 4 weekly cycles (induction); re-treatment every 6 months based on CD19/CD20 monitoring",
        "dose_paed": "375 mg/m² per dose × 4 weekly cycles; pre-medicate with methylprednisolone 100 mg IV + paracetamol + chlorphenamine",
        "moa": (
            "Anti-CD20 monoclonal antibody depletes B-lymphocytes via ADCC, CDC, and apoptosis. "
            "In Rasmussen (primarily T-cell disease), rituximab's benefit may be indirect: "
            "B-cells serve as antigen-presenting cells for CD8+ T-cells — depleting B-cells "
            "disrupts the T-cell activation cascade. "
            "Additionally, rituximab reduces GluR3/GluD2 antibody production by plasma cell "
            "precursors. CD19/CD20 depletion is maintained for 6–9 months after a 4-cycle course."
        ),
        "efficacy": (
            "40–55% show sustained seizure reduction (6+ months). "
            "Best evidence for slowing MRI progression when given early in disease course. "
            "Rarely produces complete disease remission."
        ),
        "safety": (
            "Infusion reactions (anaphylaxis, bronchospasm — pre-medicate mandatory); "
            "prolonged immunosuppression (B-cell depletion 6–9 months); "
            "progressive multifocal leukoencephalopathy (PML — rare but fatal, monitor JC virus antibody); "
            "hepatitis B reactivation (screen HBsAg + anti-HBc BEFORE first dose); "
            "late-onset neutropenia; hypogammaglobulinaemia."
        ),
        "monitoring": (
            "Hepatitis B screen (HBsAg, anti-HBc, anti-HBs) BEFORE first dose — mandatory; "
            "JC virus antibody index (PML risk stratification); "
            "CD19/CD20 counts at 1M, 3M, 6M post-infusion; "
            "IgG levels quarterly (risk of secondary hypogammaglobulinaemia); "
            "chest X-ray + TB screen (Quantiferon) before initiating; "
            "CBC + differential monthly for first 6M."
        ),
        "contraindication_note": None,
    },
    {
        "drug": "Tacrolimus (Calcineurin Inhibitor)",
        "evidence": "Level C (compassionate use)",
        "evidence_ref": "Bien 2002 Brain (calcineurin inhibitor series); Granata 2014 Epilepsia (case reports)",
        "dose_adult": "0.05–0.15 mg/kg/day in 2 divided doses; TDM target trough 5–10 ng/mL",
        "dose_paed": "0.15–0.2 mg/kg/day (higher weight-based dose in children); TDM mandatory; oral suspension available",
        "moa": (
            "Binds FKBP12 intracellularly → complex inhibits calcineurin → prevents "
            "dephosphorylation of NFAT → blocks IL-2 transcription → suppresses T-cell "
            "activation and proliferation. Directly targets the CD8+ T-cell arm of "
            "Rasmussen pathogenesis. Unlike rituximab (B-cell depleter), tacrolimus "
            "directly suppresses the cytotoxic T-cell response."
        ),
        "efficacy": "Case series suggest 30–40% seizure reduction; limited by nephrotoxicity at effective immunosuppressive doses.",
        "safety": (
            "Nephrotoxicity (dose-dependent; monitor eGFR); hypertension; neurotoxicity "
            "(tremor, headache, encephalopathy at high levels); opportunistic infections; "
            "PTLD (post-transplant lymphoproliferative disorder — rare); "
            "hyperglycaemia; hypomagnesaemia."
        ),
        "monitoring": (
            "TDM trough level 5–10 ng/mL (weekly until stable, then monthly); "
            "renal function (eGFR, creatinine) monthly; "
            "blood pressure weekly (first month) then at each visit; "
            "magnesium + glucose monthly; "
            "full blood count and LFT quarterly."
        ),
        "contraindication_note": "Avoid with strong CYP3A4 inhibitors (e.g., fluconazole — raises tacrolimus levels 3-5x)",
    },
    {
        "drug": "Methylprednisolone / Prednisolone (Corticosteroids)",
        "evidence": "Level B",
        "evidence_ref": "Granata 2014 Epilepsia; Bien 2002 Brain; European RE Consensus 2017",
        "dose_adult": "IV methylprednisolone 20–30 mg/kg/day × 3 days (pulse) → oral prednisolone 1–2 mg/kg/day tapering over 6–8 weeks",
        "dose_paed": "IV methylprednisolone 20–30 mg/kg/day × 3 days (max 1 g/day); then oral prednisolone 1 mg/kg/day; taper over 3–6 months with weekly 10% reductions",
        "moa": (
            "Broad immunosuppression via glucocorticoid receptor: suppresses pro-inflammatory "
            "cytokines (IL-1β, IL-6, TNF-alpha, IFN-gamma); reduces BBB permeability "
            "(reducing T-cell CNS entry); promotes regulatory T-cell function; "
            "direct anti-oedema effect (reduces vasogenic oedema in acutely inflamed cortex). "
            "IV pulse provides rapid anti-inflammatory effect; oral taper maintains suppression."
        ),
        "efficacy": "50–65% show acute seizure reduction during IV pulse; effect typically lasts 4–12 weeks before rebound.",
        "safety": (
            "Short-term: hyperglycaemia, hypertension, Cushingoid features, mood disturbance, "
            "peptic ulcer (PPI co-prescription mandatory), immunosuppression; "
            "long-term: adrenal suppression, osteoporosis (DEXA + calcium/D3 supplementation), "
            "growth impairment in children, cataracts, avascular necrosis."
        ),
        "monitoring": (
            "Blood glucose (BGL) daily during IV pulse; blood pressure at each infusion; "
            "bone mineral density (DEXA) at 6M if prolonged course; "
            "calcium + vitamin D3 supplementation mandatory on long-term steroids; "
            "morning cortisol before each taper step to check HPA axis recovery; "
            "ophthalmology annually (cataracts); growth chart in children."
        ),
        "contraindication_note": None,
    },
    {
        "drug": "Natalizumab (anti-VLA-4 / anti-α4-integrin)",
        "evidence": "Level C (off-label)",
        "evidence_ref": "Bien 2013 case series; Granata 2014 Epilepsia (off-label Rasmussen report)",
        "dose_adult": "300 mg IV every 4 weeks (standard MS dosing, used off-label in Rasmussen)",
        "dose_paed": "Same dose (300 mg IV q4W); weight-adjusted consideration in children <40 kg (limited data)",
        "moa": (
            "Humanised monoclonal antibody against VLA-4 (α4β1 integrin) on T-lymphocyte surface. "
            "Blocks interaction with VCAM-1 on brain endothelium, preventing T-lymphocyte "
            "transmigration across the blood-brain barrier. Directly addresses the primary "
            "pathological mechanism of Rasmussen: CD8+ T-cell CNS entry. "
            "Rapid onset of action — T-cell CNS counts decrease within 2–4 weeks. "
            "Most targeted available immunotherapy for T-cell-mediated RE."
        ),
        "efficacy": "Case series: 40–50% EPC reduction in early disease; minimal data in late-stage disease.",
        "safety": (
            "Progressive multifocal leukoencephalopathy (PML) due to JC virus reactivation — "
            "ABSOLUTE risk monitoring required (JC antibody index; risk stratification mandatory); "
            "infusion reactions; hepatotoxicity (rare); immunosuppression; "
            "immune reconstitution inflammatory syndrome (IRIS) on discontinuation."
        ),
        "monitoring": (
            "JC virus antibody index BEFORE initiation and every 6 months — MANDATORY; "
            "if JC index >1.5, risk-benefit discussion required; "
            "MRI brain every 3–6 months to detect PML early (asymmetric non-enhancing T2 lesions); "
            "LFT at 3M and 6M; CBC quarterly; "
            "neurological review before each infusion."
        ),
        "contraindication_note": "JC antibody index >2.0 = HIGH PML risk — weigh against disease severity before prescribing",
    },
    {
        "drug": "AEDs (CBZ / OXC / LEV / CLB) — Symptomatic only",
        "evidence": "Symptomatic (no disease-modifying effect)",
        "evidence_ref": "European RE Consensus 2017; Varadkar 2014 Brain (AED review in RE)",
        "dose_adult": "CBZ 200–1200 mg/day; OXC 300–2400 mg/day; LEV 500–3000 mg/day; CLB 5–40 mg/day",
        "dose_paed": "CBZ 5–20 mg/kg/day; LEV 20–60 mg/kg/day; CLB 0.5–1 mg/kg/day",
        "moa": (
            "Standard AEDs (Na+-channel blockers, GABAergic, SV2A-binding) provide partial "
            "symptomatic seizure control but do NOT alter the underlying autoimmune "
            "neuroinflammatory process. EPC is characteristically AED-resistant. "
            "LEV and CLB are preferred adjuncts (fewer drug interactions, less cognitive burden). "
            "CBZ and OXC may provide modest focal seizure control but are not effective for EPC."
        ),
        "efficacy": (
            "Partial focal seizure control (40–60% reduction) in some patients; "
            "EPC is uniformly AED-resistant — this is a diagnostic feature of RE. "
            "AEDs are used to reduce FIAS/GTCS frequency while immunotherapy/surgical workup proceeds."
        ),
        "safety": "Standard AED-specific profiles apply — see individual AED datasheets",
        "monitoring": (
            "Standard AED monitoring (TDM where applicable, LFT, CBC, electrolytes). "
            "Regular reassessment — if AED polypharmacy with no EPC improvement, "
            "escalate to immunotherapy and surgical planning rather than adding more AEDs."
        ),
        "contraindication_note": (
            "AEDs ALONE do NOT halt disease progression in Rasmussen's Encephalitis — "
            "immunotherapy + surgical referral must run in parallel."
        ),
    },
]

# ── Monitoring Protocol ────────────────────────────────────────────────────────
MONITORING = [
    {
        "item": "MRI Brain (serial) — T2 FLAIR unilateral atrophy tracking",
        "frequency": "Every 3–6 months during active disease; annually post-surgical",
        "rationale": (
            "Serial MRI is the primary disease-progression tool in Rasmussen. "
            "T2 FLAIR hyperintensity and progressive unilateral cortical atrophy track "
            "disease activity. Caudate atrophy is a sensitive early marker. "
            "Stabilisation of atrophy = disease plateau → surgical timing window. "
            "Post-hemispherectomy: confirm surgical completeness + exclude PML (if on natalizumab)."
        ),
    },
    {
        "item": "EEG (standard + 24h EMG-EEG polygraph for EPC quantification)",
        "frequency": "Standard EEG every 3–6 months; EMG-EEG polygraph at diagnosis and with disease change",
        "rationale": (
            "Unilateral delta slowing is the background signature of Rasmussen. "
            "EMG-EEG polygraph is mandatory to: (1) confirm EPC (time-lock EEG spike to EMG jerk); "
            "(2) quantify EPC frequency (jerks/hour); (3) demonstrate AED resistance. "
            "Intra-operative ECoG guides hemispherotomy completeness. "
            "Post-surgical EEG confirms no residual epileptiform discharge from disconnected hemisphere."
        ),
    },
    {
        "item": "Neuropsychological Assessment",
        "frequency": "Annually (minimum); 6-monthly in pre-surgical phase",
        "rationale": (
            "Rasmussen causes progressive cognitive decline proportional to hemispheric destruction. "
            "Annual neuropsychological battery (language, memory, IQ, executive function) tracks "
            "disease burden and informs surgical urgency. In dominant-hemisphere Rasmussen, "
            "language transfer to the contralateral hemisphere is assessed by fMRI + Wada test "
            "— critical for surgical candidacy. Post-surgical neuropsychology at 6M and 12M "
            "to guide educational support."
        ),
    },
    {
        "item": "Motor / Language Function Assessment (physiotherapy + SLT)",
        "frequency": "Every 3–6 months during active disease; post-surgical every 3 months for 2 years",
        "rationale": (
            "Progressive hemiplegia and language deterioration are the defining morbidities "
            "of Rasmussen. Motor assessment quantifies hemiplegia severity (MACS, Melbourne scale). "
            "Speech-language therapy (SLT) monitors language function and facilitates "
            "transfer assessment pre-surgically. Post-surgical intensive physiotherapy + "
            "SLT is essential to maximise neuroplasticity-based recovery."
        ),
    },
]

# ── Lifecycle Management ──────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Prodrome — First Focal Seizures / EPC (1–6 months)",
        "age_range": "Variable onset; typically 2–12 years",
        "key_events": "First EPC or focal motor seizure; brain MRI ± normal or early subtle signal change; autoantibody panel; urgent specialist referral",
        "focus": (
            "Diagnosis: Bien 2013 criteria (clinical + MRI + histopathology). "
            "Initiate IVIG immediately while diagnosis confirmed. "
            "Brain biopsy if seronegative + high clinical suspicion. "
            "EMG-EEG polygraph to document EPC. "
            "First MRI with FCD-protocol + contrast. "
            "Autoantibody panel: GluR3, GluD2, NMDAR, LGI1, CASPR2, GAD65 (serum + CSF)."
        ),
    },
    {
        "window": "Rapid Deterioration Phase (6–18 months)",
        "age_range": "Peak disease activity phase",
        "key_events": "EPC escalation; progressive hemiplegia onset; cognitive decline; MRI progression; surgical workup begins",
        "focus": (
            "Aggressive immunotherapy: IVIG monthly + plasma exchange for flares + rituximab (4 cycles). "
            "Seizure action plan for school. Safety adaptations for hemiplegia. "
            "Begin surgical workup: language fMRI + Wada test (if dominant hemisphere). "
            "Refer to epilepsy surgery centre. Neuropsychology 6-monthly. "
            "Family counselling: realistic prognosis, hemispherectomy discussion."
        ),
    },
    {
        "window": "Plateau Phase — Hemiplegia Established",
        "age_range": "Typically 18–36 months from onset",
        "key_events": "Hemiplegia maximal; MRI atrophy stable (3 consecutive MRIs unchanged); surgical candidacy confirmed",
        "focus": (
            "Surgical timing: hemispherotomy at disease plateau (once hemiplegia is established, "
            "surgical hemiplegia adds no additional functional deficit). "
            "Complete pre-surgical workup: scalp video-EEG, SEEG if required, FDG-PET. "
            "Confirm unilateral seizure onset. Language fMRI result review. "
            "Anaesthetic fitness assessment. Consent process with family."
        ),
    },
    {
        "window": "Post-Surgical Recovery (3–12 months)",
        "age_range": "Post-hemispherectomy",
        "key_events": "EPC resolution; GTCS cessation; intensive rehabilitation begins; AED wean if seizure-free",
        "focus": (
            "Intensive inpatient rehabilitation: physiotherapy (compensatory movement), "
            "occupational therapy (ADL adaptation), speech therapy (language re-acquisition). "
            "MRI at 3M and 12M to confirm surgical completeness. "
            "EEG at 3M: confirm no epileptiform discharge. "
            "AED taper considered at 12M if seizure-free (Engel I). "
            "Neuropsychology at 6M."
        ),
    },
    {
        "window": "School Rehabilitation Phase (1–5 years post-surgery)",
        "age_range": "School age",
        "key_events": "Educational reintegration; compensatory hand function development; language consolidation (if language-transfer hemisphere)",
        "focus": (
            "School placement with educational support plan (EHCP / IEP). "
            "Physiotherapy: compensatory upper-limb strategies, gait training. "
            "Neuroplasticity window: intensive language therapy if dominant-hemisphere surgery. "
            "Annual neuropsychology. Driving ban: 12M minimum seizure-free (then DVLA assessment). "
            "Psychosocial support for body-image / disability adaptation."
        ),
    },
    {
        "window": "Adult Independence",
        "age_range": "16+ years",
        "key_events": "Vocational training; driving assessment; independent living; ongoing physiotherapy",
        "focus": (
            "Long-term follow-up: annual neurology review (seizure recurrence rare but possible). "
            "Employment support: majority of Engel I post-hemispherectomy patients achieve "
            "sheltered or supported employment. Driving assessment at 12M seizure-free. "
            "Annual MRI for those on natalizumab (PML surveillance). "
            "Family genetic counselling: Rasmussen is not heritable — reassure siblings."
        ),
    },
]

# ── Standards ──────────────────────────────────────────────────────────────────
STANDARDS = [
    {
        "standard": "ILAE Classification 2022",
        "relevance": "Rasmussen's classified as structural-autoimmune epilepsy; EPC recognised as epilepsy variant",
    },
    {
        "standard": "NICE NG217 (2022)",
        "relevance": "Referral to specialist epilepsy surgery centre recommended for Rasmussen; immunotherapy pathway endorsed",
    },
    {
        "standard": "Bien 2013 — Rasmussen Encephalitis Diagnostic Criteria",
        "relevance": "3-part criteria: (1) clinical (focal seizures/EPC + hemiplegia + cognitive decline); (2) MRI (unilateral atrophy/signal); (3) histopathology (T-cell infiltration + microglial nodules). ≥2/3 required.",
    },
    {
        "standard": "Varadkar 2014 Brain — Rasmussen Treatment Consensus",
        "relevance": "Hemispherectomy/hemispherotomy defined as definitive treatment; Engel I 52–65%; immunotherapy as bridging therapy",
    },
    {
        "standard": "European Rasmussen Encephalitis Consensus 2017",
        "relevance": "IVIG + plasma exchange + rituximab immunotherapy ladder; surgical timing at disease plateau; language fMRI mandatory pre-surgically",
    },
    {
        "standard": "FDA IVIG (off-label Rasmussen's use)",
        "relevance": "IVIG used off-label; dosing protocol 0.4 g/kg/day × 5 days per course endorsed by European consensus",
    },
]

# ── Thresholds ─────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "EPC emergency threshold", "value": ">2 hours continuous EPC → emergent IV methylprednisolone"},
    {"threshold": "Surgical timing (hemispherectomy)", "value": "Disease plateau: hemiplegia established + 3 consecutive stable MRIs"},
    {"threshold": "Post-hemispherectomy Engel I target", "value": "≥52% (institutional benchmark); international literature 52–65%"},
    {"threshold": "IVIG dosing", "value": "0.4 g/kg/day × 5 days induction; 1–2 g/kg monthly maintenance"},
    {"threshold": "Rituximab cycle", "value": "375 mg/m² × 4 weekly doses; re-treat at 6M based on CD19/CD20 levels"},
    {"threshold": "Driving ban", "value": "Minimum 12 months seizure-free (jurisdiction-dependent; DVLA assessment required)"},
    {"threshold": "MRI follow-up interval", "value": "Every 3–6 months during active disease; annually post-surgical"},
    {"threshold": "Cognitive testing frequency", "value": "Every 6 months during active disease; annually post-surgical"},
]

# ── Concepts ──────────────────────────────────────────────────────────────────
CONCEPTS = [
    {
        "term": "Rasmussen's Encephalitis (RE)",
        "definition": (
            "Rare, progressive, unilateral inflammatory brain disease characterised by focal "
            "seizures/EPC, progressive hemiplegia, and cognitive decline. Pathologically driven "
            "by CD8+ T-lymphocyte infiltration causing neuronal destruction confined to one "
            "hemisphere. First described by Rasmussen in 1958. No cure — hemispherectomy is "
            "definitive treatment."
        ),
    },
    {
        "term": "Epilepsia Partialis Continua (EPC)",
        "definition": (
            "Continuous or nearly continuous focal myoclonic jerks of cortical origin affecting "
            "one body part, lasting hours to years, without loss of consciousness. Pathognomonic "
            "feature of Rasmussen's Encephalitis when associated with progressive hemiplegia. "
            "Absolutely AED-resistant — this resistance is itself a diagnostic feature."
        ),
    },
    {
        "term": "Unilateral Cortical Atrophy",
        "definition": (
            "Progressive loss of cortical volume confined to one hemisphere on serial MRI, "
            "beginning in peri-sylvian regions and spreading to involve the entire hemisphere. "
            "Caudate atrophy is an early and sensitive marker. T2 FLAIR hyperintensity precedes "
            "volume loss. Contralateral hemisphere is normal — asymmetry is pathognomonic for RE."
        ),
    },
    {
        "term": "GluR3 / AMPA Receptor Autoimmunity",
        "definition": (
            "GluR3 (GluA3 subunit of AMPA-type glutamate receptors) autoantibodies were the first "
            "identified in Rasmussen (Rogers 1994). However, antibody titre does NOT correlate "
            "with disease severity — CD8+ T-cell infiltration is the primary pathological driver. "
            "GluR3 antibodies are absent in many confirmed Rasmussen cases."
        ),
    },
    {
        "term": "T-lymphocyte Infiltration (CD8+ Cytotoxic T-cells)",
        "definition": (
            "The primary pathological mechanism of Rasmussen's: CD8+ cytotoxic T-lymphocytes "
            "form immunological synapses with MHC-I-expressing neurons and release perforin/granzyme B, "
            "causing neuronal apoptosis. This T-cell-mediated neuronal destruction is restricted "
            "to one hemisphere and is the definitive histopathological finding (along with microglial "
            "nodules and astrogliosis)."
        ),
    },
    {
        "term": "Functional Hemispherectomy",
        "definition": (
            "Surgical disconnection of the epileptogenic hemisphere from the rest of the brain "
            "without complete anatomical removal. Corticospinal tracts and corpus callosum are "
            "divided. Preferred over anatomical hemispherectomy in children due to lower risk of "
            "late CSF complications and cerebral siderosis. Eliminates seizure source in "
            "Rasmussen while preserving brain volume."
        ),
    },
    {
        "term": "Hemispherotomy",
        "definition": (
            "A specific functional hemispherectomy technique (Villemure or modified technique) "
            "that achieves hemispheric disconnection through a minimal cortical incision, "
            "disconnecting the white matter tracts, corpus callosum, and internal capsule. "
            "Increasingly preferred over classical hemispherectomy for lower morbidity and "
            "equivalent seizure outcomes (Engel I 52–65%)."
        ),
    },
    {
        "term": "IVIG (Intravenous Immunoglobulin)",
        "definition": (
            "Polyclonal IgG preparation used as immunomodulatory therapy in Rasmussen. "
            "Mechanisms include Fc-receptor blockade, autoantibody neutralisation, complement "
            "inhibition, and Treg upregulation. Provides temporary disease control (4–12 weeks). "
            "Level B evidence. First-line early immunotherapy in Rasmussen while surgical "
            "planning proceeds."
        ),
    },
    {
        "term": "Plasma Exchange (Plasmapheresis)",
        "definition": (
            "Mechanical removal of plasma containing pathogenic antibodies, cytokines, and "
            "complement components. Provides rapid (48–72 hour) reduction in circulating "
            "inflammatory mediators. Used for acute EPC flares and as bridging to rituximab. "
            "5–10 sessions standard course. Always followed by IVIG to prevent rebound antibody surge."
        ),
    },
    {
        "term": "Rituximab",
        "definition": (
            "Anti-CD20 monoclonal antibody depleting B-lymphocytes. In Rasmussen (primary T-cell disease), "
            "rituximab acts indirectly by removing B-cells that serve as antigen-presenting cells "
            "for CD8+ T-cells. Also reduces GluR3/GluD2 antibody production. "
            "375 mg/m² × 4 weekly doses. Level B evidence. Best evidence for slowing MRI progression."
        ),
    },
    {
        "term": "Engel Classification",
        "definition": (
            "Surgical outcome classification: Engel I = seizure-free; Engel II = worthwhile improvement; "
            "Engel III = worthwhile improvement (not seizure-free); Engel IV = no worthwhile improvement. "
            "In Rasmussen hemispherectomy, Engel I achieved in 52–65%. EPC resolves in >90% of all surgical cases."
        ),
    },
    {
        "term": "Neuroplasticity / Language Transfer",
        "definition": (
            "In young children (<8 years) undergoing dominant-hemisphere hemispherectomy, "
            "language function can transfer to the contralateral (healthy) hemisphere. "
            "This neuroplasticity window is age-dependent — earlier surgery maximises transfer. "
            "Language fMRI + Wada test assess pre-surgical transfer extent and predict post-surgical "
            "language outcome."
        ),
    },
    {
        "term": "Progressive Hemiplegia",
        "definition": (
            "The defining motor morbidity of Rasmussen's Encephalitis: progressive contralateral "
            "hemiparesis advancing to complete hemiplegia over 6–24 months, caused by progressive "
            "motor cortex destruction. By disease plateau, hemiplegia is complete and irreversible. "
            "This is why hemispherectomy can be performed at plateau without additional motor loss."
        ),
    },
    {
        "term": "SUDEP in Rasmussen's Encephalitis",
        "definition": (
            "Sudden Unexpected Death in Epilepsy: elevated risk in Rasmussen due to high-frequency "
            "uncontrolled seizures (EPC, nocturnal GTCS) and progressive cortical dysfunction. "
            "Risk estimated 1:250/year in active, drug-resistant Rasmussen. Night-time seizure "
            "monitoring, prone-position avoidance, and prompt surgical referral are risk-reduction strategies."
        ),
    },
]

# ── Contraindications ──────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "AEDs as sole treatment (monotherapy without immunotherapy)",
        "contraindicated_in": "All Rasmussen's Encephalitis",
        "consequence": (
            "AEDs provide partial symptomatic seizure control only and do NOT halt the "
            "underlying T-cell-mediated neuroinflammatory process. Relying on AEDs alone "
            "allows unopposed hemispheric destruction. EPC is AED-resistant by definition. "
            "Immunotherapy + surgical planning must be initiated in parallel with AED symptomatic management."
        ),
    },
    {
        "drug": "Antiviral therapy (aciclovir / ganciclovir) as primary treatment",
        "contraindicated_in": "All Rasmussen's Encephalitis (viral-trigger hypothesis cases)",
        "consequence": (
            "Active viral replication is NOT found in Rasmussen brain tissue at biopsy. "
            "Treating Rasmussen with antivirals alone delays appropriate immunotherapy and "
            "surgical referral, causing preventable hemispheric destruction. "
            "Antivirals are only appropriate if concurrent HSV encephalitis is confirmed on CSF PCR."
        ),
    },
    {
        "drug": "Natalizumab in JC antibody index >2.0 without risk-benefit counselling",
        "contraindicated_in": "High JC antibody index (>2.0) Rasmussen patients",
        "consequence": (
            "Progressive multifocal leukoencephalopathy (PML) risk increases substantially "
            "at JC index >2.0, especially after 24 months of natalizumab. PML causes "
            "irreversible neurological disability or death. JC antibody index must be checked "
            "before initiation and every 6 months throughout treatment."
        ),
    },
    {
        "drug": "Rituximab without hepatitis B screening",
        "contraindicated_in": "All Rasmussen patients before first rituximab dose",
        "consequence": (
            "Rituximab causes B-cell depletion and profound immunosuppression. "
            "In hepatitis B core-antibody-positive (anti-HBc+) patients, rituximab triggers "
            "HBV reactivation — a potentially fatal hepatitis. HBsAg + anti-HBc + anti-HBs "
            "MUST be tested before rituximab initiation; prophylactic antiviral (entecavir) "
            "required for anti-HBc+ patients."
        ),
    },
    {
        "drug": "Hemispherectomy in bilateral disease",
        "contraindicated_in": "Rasmussen's with bilateral hemispheric involvement",
        "consequence": (
            "Rare bilateral Rasmussen (<5%) — confirmed by bilateral MRI atrophy and bilateral "
            "seizure onset on SEEG. Hemispherectomy is contraindicated as there is no 'normal' "
            "hemisphere to sustain function post-surgery. Palliative immunotherapy + AEDs only."
        ),
    },
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    "Rasmussen T et al. 1958. Focal seizures due to chronic localized encephalitis. Neurology 8:435–445.",
    "Rogers SW et al. 1994. Autoantibodies to glutamate receptor GluR3 in Rasmussen's encephalitis. Science 265:648–651.",
    "Bien CG et al. 2002. Rasmussen encephalitis: evidence of a T-cell-mediated immune reaction. Ann Neurol 51:311–318.",
    "Varadkar S et al. 2014. Rasmussen's encephalitis: clinical features, pathobiology, and treatment advances. Lancet Neurol 13:195–205.",
    "Granata T et al. 2014. Brain MR findings and surgical outcome in Rasmussen encephalitis. Epilepsia 55:1228–1237.",
    "Guan Y et al. 2022. Long-term outcome after hemispherectomy in Rasmussen encephalitis: a multicenter study. Epilepsia 63:1234–1245.",
]


# ── API Functions ─────────────────────────────────────────────────────────────

def overview():
    total = len(PATIENTS)
    male_n = sum(1 for p in PATIENTS if p["sex"] == "M")
    epc_active = sum(1 for p in PATIENTS if p["EPC"])
    hemi_n = sum(1 for p in PATIENTS if "hemispherectomy" in p["current_treatment"].lower()
                 or "hemispherotomy" in p["current_treatment"].lower())
    sf_n = sum(1 for p in PATIENTS if p["seizure_control"] == "Seizure-free (post-surgical)")
    dr_n = sum(1 for p in PATIENTS if p["seizure_control"] == "Drug-resistant (EPC active)")
    return {
        "syndrome": "Rasmussen's Encephalitis",
        "icd10": "G04.81",
        "total_patients": total,
        "male_n": male_n,
        "male_pct": round(male_n / total * 100),
        "epc_active_n": epc_active,
        "epc_active_pct": round(epc_active / total * 100),
        "post_surgical_n": hemi_n,
        "post_surgical_pct": round(hemi_n / total * 100),
        "seizure_free_n": sf_n,
        "seizure_free_pct": round(sf_n / total * 100),
        "drug_resistant_n": dr_n,
        "drug_resistant_pct": round(dr_n / total * 100),
        "etiology_distribution": {et["etiology"]: et["n"] for et in ETIOLOGY_CATALOG},
        "prognosis_summary": {
            "engel_I_post_hemispherectomy": "52–65% seizure-free",
            "epc_resolution_post_surgical": ">90% resolve",
            "progressive_hemiplegia": "100% by disease plateau",
            "cognitive_decline": "Proportional to hemispheric destruction",
            "language_transfer_probability": "High if surgery <8 years (neuroplasticity window)",
            "disease_course": "Progressive (no spontaneous remission without surgery)",
            "sudep_risk": "~1:250/year in active drug-resistant RE",
        },
        "clinical_alerts": [
            "⛔ AEDs ALONE do NOT halt disease progression — immunotherapy + surgical referral mandatory",
            "⛔ EPC is AED-resistant by definition — persistent EPC is a diagnostic and surgical trigger",
            "⚠️ Natalizumab: check JC antibody index before initiation and every 6 months (PML risk)",
            "⚠️ Rituximab: hepatitis B screen (HBsAg + anti-HBc) MANDATORY before first dose",
            "✅ Hemispherectomy/hemispherotomy: definitive treatment — Engel I 52–65%; refer early",
            "⚠️ Dominant hemisphere RE: language fMRI + Wada test BEFORE surgical decision",
            "⚠️ EPC >2 hours: emergent IV methylprednisolone 20–30 mg/kg/day × 3 days",
        ],
        "key_thresholds": {
            "epc_emergency": ">2h continuous EPC → emergent immunotherapy",
            "surgical_timing": "Disease plateau (stable hemiplegia + 3 stable MRIs)",
            "ivig_dose": "0.4 g/kg/day × 5 days induction",
            "rituximab_dose": "375 mg/m² × 4 weekly cycles",
            "mri_followup": "Every 3–6 months active disease",
            "driving_ban": "12 months seizure-free minimum",
        },
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
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
    }


def definitions():
    return {
        "concepts": CONCEPTS,
        "contraindications": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
    }
