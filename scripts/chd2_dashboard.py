"""
CHD2 Epilepsy — Myoclonic Encephalopathy / GGE-Photosensitive / DEE32
(Chromodomain Helicase DNA Binding Protein 2 / CHD2 Chromatin Remodeling / 15q26.1)
====================================================================================
40-patient cohort · CHD2 de novo LOF/missense · Autosomal Dominant
GGE spectrum · Severe Photosensitivity · Myoclonic + Absence + GTCS · OMIM #615369

CHD2 BIOLOGY:
CHD2 (15q26.1) encodes Chromodomain Helicase DNA Binding Protein 2, a ~240-kDa
ATP-dependent chromatin remodeling enzyme of the SWI/SNF (CHD) superfamily.

CHD2 PROTEIN STRUCTURE:
  - N-terminal tandem chromodomains (CD1 + CD2): bind H3K4me3 (active promoter mark)
    → target CHD2 to transcriptionally active chromatin at neuronal promoters.
  - Central ATPase/helicase domain (DEXDc + HELICc): remodels nucleosome positioning
    → opens chromatin accessibility at GABAergic neuron developmental genes.
  - C-terminal SANT/SLIDE domain: stabilises DNA binding adjacent to nucleosome.
  CHD2-specific function: H3.3 histone deposition at newly opened chromatin regions
  (unlike CHD1 which binds H3.1/H3.2). H3.3 deposition → stable, heritable open
  chromatin marks at GABA-pathway gene loci.

CHD2 NEURODEVELOPMENTAL ROLE:
  During early brain development (embryonic week 5–postnatal week 4 in humans):
  1. Opens promoters of GABAergic interneuron specification genes (DLX1/2, ARID1B,
     ARX, LHX6) → drives cortical inhibitory interneuron differentiation.
  2. Remodels chromatin at glutamate receptor subunit genes (GRIN2A, GRIN2B, GLUA2)
     → regulates excitatory/inhibitory balance.
  3. CHD2 activity is highest in hippocampus (dentate gyrus + CA3) and cortical
     layers II/III/IV — the substrate for generalised spike-wave discharge initiation.

CHD2-EPILEPSY MECHANISM:
  LOF haploinsufficiency (heterozygous):
  → 50% reduction of CHD2 chromatin remodeling activity
  → Impaired H3.3 deposition at GABAergic gene promoters
  → Under-differentiation of PV+ and SST+ inhibitory interneurons (20-30% reduced density)
  → Cortical disinhibition + thalamo-cortical hyperexcitability
  → Generalised spike-wave discharges (GGE) + myoclonic jerks

  PHOTOSENSITIVITY MECHANISM (cardinal feature of CHD2):
  CHD2 also regulates PRKCG (PKCγ), GABRB3, and GAD1 expression in visual cortex
  → CHD2 LOF → hyperexcitable occipital cortex
  → Abnormal visual cortex H3.3 chromatin at GABA promoters
  → Pathological PPR (photoparoxysmal response) in 75–80% of patients
  → One of the most photosensitive genetic epilepsies known

CHD2-DEE32 / GGE CLINICAL PROFILE (OMIM #615369):
  PHENOTYPE SPECTRUM:
    I.  Classic CHD2-GGE (myoclonic-dominant): myoclonic jerks + absence + GTCS, moderate
        photosensitivity, mild-moderate ID. Onset 1-5 years. (~35%)
    II. Severe CHD2-MEI (myoclonic encephalopathy of infancy): early-onset refractory
        myoclonic clusters, severe ID, near-continuous ictal EEG activity. (~25%)
    III. CHD2-Dravet-like: prolonged febrile seizures → polymorphic drug-resistant
        epilepsy; SCN1A-negative Dravet phenocopy. (~20%)
    IV. Photosensitive Epilepsy (GGE-photo-dominant): dominant photosensitivity with
        mild myoclonic + absence, nearly normal cognition in 30%. (~12%)
    V.  Phenocopy (non-CHD2): GGE + photosensitivity without CHD2 variant (GABRA1,
        GABRB3, SLC6A1). (~8%)

  KEY CLINICAL FEATURES:
    - Photosensitivity: 75-80% (among highest in genetic epilepsies — cardinal sign)
    - Eye closure sensitivity: 40% (provoked absence/myoclonic by eye closure)
    - Intellectual disability: mild-moderate 65%, severe-profound 15%, normal IQ 20%
    - Drug-resistant epilepsy: 35-40% (lower than DEE genes but higher than typical GGE)
    - Behavioral comorbidities: ADHD 50%, ASD features 25%, anxiety 30%
    - Motor development: normal in most; some gross motor delay in severe phenotype
    - Language: expressive delay 50%; receptive delay 30%

  GENETICS:
    - Gene: CHD2, chromosome 15q26.1
    - Inheritance: Autosomal Dominant; >90% de novo (familial <10%)
    - Variant types: nonsense/frameshift ~50%, missense (ATPase domain) ~30%,
      large CNV del 15q26 ~15%, splice site ~5%
    - pLI = 0.99 (very high LOF intolerance, gnomAD v4)
    - Incidence: ~1:50,000–100,000 live births; ~200+ published families
    - ClinVar: ~80+ CHD2 variants (P/LP); de novo confirmed >90% published cases

EEG IN CHD2-GGE:
  - Photoparoxysmal response (PPR): intermittent photic stimulation 1–30 Hz triggers
    bilateral occipital → generalised polyspike-wave; Grade 3-4 (ILAE Waltz 1992).
    This is the most consistently abnormal EEG finding in CHD2.
  - Background: normal or mildly slow theta; age-appropriate organisation preserved.
  - Interictal: generalised 3-4 Hz spike-wave discharges; polyspike-wave.
  - Ictal myoclonic: bilateral polyspike burst (0.5-2s) + brief attenuation.
  - Ictal absence: 3-3.5 Hz GSW, 5-25s duration; may have subtle automatisms.
  - Ictal GTCS: bilateral recruiting rhythm → generalised tonic phase → clonic.
  - Sleep: activation of IEA in NREM (vs. suppression in REM); sleep spindles present.
  - Eye closure sensitivity: burst of GSW/polyspike-wave immediately on eye closure (EC-IPA).

KEY STANDARDS:
  ILAE 2022 · NICE NG217 · Carvill et al. 2013 (Nature Genetics — first CHD2 report) ·
  Thomas et al. 2015 (Am J Hum Genet) · Lund et al. 2014 (Epilepsy & Behavior) ·
  Bhatt et al. 2023 (Neurology) · MHRA VPPP 2021 · CPIC HLA-B*15:02 2023 ·
  FDA Valproate REMS · ILAE Dietary Therapies 2018 · ACMG-AMP 2015 ·
  CPIC VPA-POLG 2023 · Waltz 1992 PPR grading
"""
import random

# ── Etiology catalog ──────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "CHD2 de-novo LOF (Classic GGE-Myoclonic-Dominant)",
        "category": "CHD2-de-novo-LOF-GGE-myoclonic-35%",
        "pct": 35,
        "n": 14,
        "mechanism": (
            "Heterozygous nonsense or frameshift → NMD of mutant allele → 50% CHD2 protein. "
            "Reduced H3.3 deposition at DLX1/DLX2/GAD1/GABRB3 promoters in cortex + hippocampus "
            "→ under-differentiation of PV+ and SST+ inhibitory interneurons (20-30% density loss) "
            "→ cortical disinhibition → 3-4 Hz generalised spike-wave discharge + myoclonic activity. "
            "Photosensitivity in 75% (impaired GABRB3 expression in visual cortex)."
        ),
        "eeg_correlate": (
            "Generalised 3-4 Hz spike-wave + polyspike-wave · "
            "PPR Grade 3-4 (75% of patients) · "
            "Eye closure sensitivity 40% · Normal background"
        ),
        "semiology": (
            "Myoclonic jerks (bilateral, proximal, morning predominance) onset 1-5 years; "
            "typical absence 3-25s; GTCS nocturnal + precipitated by fatigue/flash; "
            "mild-moderate ID; ADHD 50%; good motor development."
        ),
        "treatment": "VPA (Level A, all 3 seizure types); ETH (absence-dominant); LEV (myoclonic adjunct)",
        "prognosis": "Moderate — 55-60% ≥50% seizure reduction; 20-25% seizure-free; mild-moderate ID",
    },
    {
        "etiology": "CHD2 de-novo LOF (Severe MEI — Myoclonic Encephalopathy of Infancy)",
        "category": "CHD2-de-novo-LOF-MEI-severe-25%",
        "pct": 25,
        "n": 10,
        "mechanism": (
            "Truncating variant producing unstable mRNA + dominant-negative low-level protein "
            "fragment that impairs WT-CHD2 ATPase dimer assembly. More severe chromatin remodeling "
            "defect than simple haploinsufficiency → greater interneuron loss → near-continuous "
            "ictal myoclonic activity in early childhood. Cortical H3K4me3 landscape severely "
            "disrupted → widespread transcriptional dysregulation including HCN1, SHANK3, NRXN1."
        ),
        "eeg_correlate": (
            "Near-continuous polyspike-wave (status myoclonicus) · "
            "Severe generalised slowing · PPR 80% · "
            "Sleep: marked activation of IEA in NREM"
        ),
        "semiology": (
            "Onset 6-18 months with febrile myoclonic clusters; "
            "rapid escalation to drug-resistant polymorphic epilepsy; "
            "severe intellectual disability; "
            "regression of milestones after epilepsy onset; "
            "frequent status myoclonicus."
        ),
        "treatment": "VPA+CLB (broad-spectrum); KD (responder ~40%); Fenfluramine (Level C, compassionate); LEV",
        "prognosis": "Poor — drug-resistant 70%; severe ID; frequent hospitalisation for status myoclonicus",
    },
    {
        "etiology": "CHD2 de-novo missense (Dravet-like / SCN1A-negative Fever-sensitive)",
        "category": "CHD2-de-novo-missense-Dravet-like-20%",
        "pct": 20,
        "n": 8,
        "mechanism": (
            "Missense in ATPase/helicase domain (e.g., p.Arg1584Trp, p.Thr1054Met) → impaired "
            "nucleosome repositioning without haploinsufficiency (dominant-negative chromatin "
            "remodeling defect). Selective CHD2 chromatin access failure at SCN1A-regulatory "
            "elements → secondary NaV1.1 downregulation in PV+ interneurons mimicking "
            "Dravet-syndrome interneuron failure phenotype. Fever-triggered prolonged seizures "
            "prominent; SCN1A sequencing negative → CHD2 identified on gene panel."
        ),
        "eeg_correlate": (
            "Multifocal spikes + generalised polyspike-wave · "
            "PPR moderate 50% · Asymmetric ictal patterns · "
            "Status epilepticus (febrile) EEG: lateralised + generalised alternating"
        ),
        "semiology": (
            "Prolonged febrile HFS (hemi-clonic + GTCS) onset 6-18 months; "
            "afebrile polymorphic seizures emerge by age 2-3; "
            "SCN1A-negative Dravet phenotype; "
            "photosensitivity 55%; moderate-severe ID."
        ),
        "treatment": "VPA (Level A); CLB (Level B, Dravet-adjunct); KD (Level B, DRE); STX (off-label, Level C); avoid CBZ/OXC",
        "prognosis": "Moderate-poor — drug-resistant 55%; moderate-severe ID; febrile SE risk persists to age 5-6",
    },
    {
        "etiology": "CHD2 de-novo (Photosensitive GGE — Near-Normal Cognition)",
        "category": "CHD2-de-novo-photo-dominant-GGE-12%",
        "pct": 12,
        "n": 5,
        "mechanism": (
            "LOF or hypomorphic CHD2 variant with predominant GABRB3/PRKCG visual cortex "
            "chromatin effect → severe photosensitivity with relatively preserved cognitive "
            "interneuron pool in prefrontal cortex. PPR Grade 4 (threshold all frequencies). "
            "Mainly myoclonic + absence triggered by visual stimuli; spontaneous GTCS infrequent. "
            "VPA or LEV + photic avoidance achieves good control."
        ),
        "eeg_correlate": (
            "Dominant PPR Grade 4 at all IPS frequencies · "
            "Eye closure sensitivity 65% · "
            "Minimal interictal spontaneous GSW · "
            "Normal/near-normal background"
        ),
        "semiology": (
            "Photically triggered myoclonic jerks and absence from ~3-8 years; "
            "spontaneous seizures infrequent; "
            "triggered by screens, sun, disco lights; "
            "near-normal IQ (mean IQ 80-90); ADHD 40%."
        ),
        "treatment": "VPA (Level A, low-dose often sufficient); LEV (myoclonic); Photic avoidance mandatory; tinted lenses",
        "prognosis": "Good — 60-70% seizure-free with VPA + photic avoidance; near-normal IQ; good social function",
    },
    {
        "etiology": "Phenocopy (non-CHD2 GGE-Photosensitive)",
        "category": "Phenocopy-GABRA1-GABRB3-SLC6A1-8%",
        "pct": 8,
        "n": 3,
        "mechanism": (
            "GGE + marked photosensitivity without CHD2 variant; overlapping phenotype from "
            "GABRA1 (α1-subunit LOF — absence + myoclonic + GTCS), GABRB3 (β3-subunit LOF — "
            "West/IS + GGE), or SLC6A1 (GAT-1 — MAE + GGE). "
            "Same clinical presentation requires negative CHD2 ClinVar result → panel approach."
        ),
        "eeg_correlate": (
            "GGE pattern (GSW 3-4Hz) + PPR · "
            "May have syndrome-specific features (GABRB3: IS-remnant; SLC6A1: MAE-like drops)"
        ),
        "semiology": (
            "GGE-photosensitive phenotype; "
            "CHD2 sequencing negative; "
            "alternative gene identified on epilepsy panel; "
            "response to VPA similar."
        ),
        "treatment": "Same GGE protocol (VPA/ETH/LEV); avoid CBZ/OXC/Tiagabine; gene-specific nuances",
        "prognosis": "Variable — depends on underlying gene; CHD2 excluded → alternative gene-specific prognosis",
    },
]

# ── 40-Patient cohort ────────────────────────────────────────────────────────
random.seed(42)
_NAMES = [
    "Aiden","Bella","Carlos","Diana","Ethan","Fiona","George","Hannah",
    "Ivan","Julia","Kevin","Laura","Marcus","Nina","Oscar","Priya",
    "Quinn","Rachel","Samuel","Tanya","Uma","Victor","Wendy","Xavier",
    "Yara","Zoe","Aaron","Bianca","Cole","Dara","Emil","Freya",
    "Grant","Hana","Iago","Jade","Kai","Leah","Milo","Nadia",
]
_OUTCOMES = ["Seizure-free","≥50% reduction","<50% reduction","Drug-resistant"]
_OUTCOME_W = [20, 38, 27, 15]
_AED_COMBOS = [
    "VPA mono","VPA+LEV","VPA+ETH","VPA+CLB","VPA+LTG","LEV+CLB","KD+VPA","VPA+CLB+LEV",
]
def _rand_age():
    return round(random.uniform(1.2, 14.0), 1)

PATIENTS = []
_idx = 0
for etio in ETIOLOGY_CATALOG:
    for _ in range(etio["n"]):
        n = _NAMES[_idx % len(_NAMES)]
        _idx += 1
        PATIENTS.append({
            "id": f"CHD2-{_idx:03d}",
            "name": n,
            "age": _rand_age(),
            "etiology": etio["etiology"],
            "category": etio["category"],
            "onset_age_months": int(random.uniform(6, 60)),
            "current_aed": random.choice(_AED_COMBOS),
            "outcome": random.choices(_OUTCOMES, weights=_OUTCOME_W, k=1)[0],
            "photosensitive": random.random() < 0.77,
            "id_severity": random.choice(["None","Mild","Moderate","Severe","Profound"]),
        })

# ── Seizure types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Myoclonic Jerks (Pathognomonic)",
        "prevalence_pct": 85,
        "eeg": "Bilateral polyspike or polyspike-wave burst (0.5-2s); amplitude 200-500µV; no post-ictal slowing",
        "semiology": (
            "Sudden bilateral proximal jerks (arms > legs); morning predominance (post-awakening "
            "peak within 60 min); clusters of 3-10 jerks separated by 10-30s; often provoked by "
            "flash/screen/eye-closure; may cause drop of held objects (myoclonic drop); "
            "duration 0.5-2s each; consciousness preserved."
        ),
        "clinical_tip": (
            "CARDINAL SIGN: morning myoclonic clusters + photosensitivity in a child 2-8 years → "
            "CHD2 high on differential. LTG may WORSEN myoclonic (15-20% aggravation rate) — "
            "use VPA or LEV first; check EEG with IPS before initiating LTG."
        ),
    },
    {
        "type": "Typical Absence",
        "prevalence_pct": 70,
        "eeg": "Regular 3-3.5 Hz generalised spike-wave; abrupt onset/offset; duration 5-25s; may have mild PPR activation",
        "semiology": (
            "Behavioural arrest + staring + eyelid flutter; duration 5-25s; "
            "abrupt onset and offset; subtle automatisms (lip-smacking, hand fidgeting) 30%; "
            "post-ictal confusion absent (distinguishes from focal impaired awareness seizure). "
            "HV (hyperventilation) diagnostic provocation in clinic."
        ),
        "clinical_tip": (
            "ETH monotherapy effective for absence-dominant CHD2 (no GTCS/myoclonic). "
            "If GTCS also present → prefer VPA (full-spectrum). "
            "HV activates absence in clinic: reliable diagnostic tool."
        ),
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS)",
        "prevalence_pct": 55,
        "eeg": "Bilateral recruiting tonic discharge → generalised clonic polyspike-wave; post-ictal generalised slowing",
        "semiology": (
            "Bilateral symmetric tonic stiffening → clonic jerking; "
            "triggered by sleep deprivation + alcohol + missed AED + photostimulation; "
            "nocturnal bias (60%); post-ictal confusion 10-30 min; "
            "tongue bite + incontinence common in GTCS."
        ),
        "clinical_tip": (
            "Nocturnal GTCS + morning myoclonic + absence = classic CHD2-GGE triad. "
            "SUDEP risk elevated with ≥1 GTCS/year — document in SUDEP counselling. "
            "Driving restrictions apply after last GTCS (jurisdiction-dependent)."
        ),
    },
    {
        "type": "Eyelid Myoclonia with Eye-Closure Sensitivity",
        "prevalence_pct": 50,
        "eeg": "GSW/polyspike-wave immediately on eye closure (EC-IPA); occipital photosensitive discharge; PPR Grade 3-4",
        "semiology": (
            "Brief rapid eyelid fluttering immediately on eye closure or in bright light; "
            "often associated with upward gaze deviation; brief absence component (2-5s); "
            "provoked by visual stimuli + eye closure; self-induced photosensitive triggering "
            "rare but possible in CHD2 (screen-seeking behaviour in children)."
        ),
        "clinical_tip": (
            "EC-IPA (eye-closure-induced paroxysmal activity) + PPR = high CHD2 sensitivity. "
            "Tinted spectacles (FL-41 rose/amber) reduce PPR by 40-60% — recommend routinely. "
            "Video EEG with eye-closure and IPS mandatory for full characterisation."
        ),
    },
    {
        "type": "Atonic / Drop Attacks",
        "prevalence_pct": 20,
        "eeg": "High-amplitude polyspike → sudden EMG silence (atonic) or slow wave + head-drop; <2s",
        "semiology": (
            "Sudden loss of postural muscle tone → head drop or full fall; "
            "duration 1-2s; no post-ictal confusion; "
            "predominant in severe MEI subtype (25%); "
            "injury risk high (protective headgear in severe cases); "
            "may respond to CLB or KD."
        ),
        "clinical_tip": (
            "Drop attacks in CHD2-MEI: CLB or KD preferred (VGB NOT indicated in GGE). "
            "Protective headgear mandatory if ≥1 drop/week. "
            "Fenfluramine Level C evidence for refractory MEI drop attacks."
        ),
    },
]

# ── Triggers ──────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Photostimulation (Flash / Screen / Sun Glare)",
        "prevalence_pct": 80,
        "mechanism": (
            "CHD2 LOF → impaired GABRB3/PRKCG expression in visual cortex → hyperexcitable "
            "striate + extrastriate cortex → abnormal cortical response to oscillating luminance. "
            "IPS (intermittent photic stimulation) 1-30 Hz triggers PPR Grade 3-4; "
            "screen flicker at 60 Hz refresh rate can trigger myoclonic; sunlight through "
            "trees/venetian blinds (3-15 Hz equivalent) → myoclonic + absence cluster."
        ),
        "management": (
            "MANDATORY photic avoidance counselling at diagnosis. "
            "FL-41 tinted spectacles (rose/amber) reduce PPR by 40-60%. "
            "Screen: 60+ Hz refresh, matte screen, reduce brightness, blue-light filter. "
            "Sunglasses outdoors. Gaming: avoid rapidly-flashing content (epilepsy warning). "
            "Annual EEG with IPS to monitor PPR evolution."
        ),
    },
    {
        "trigger": "Sleep Deprivation",
        "prevalence_pct": 82,
        "mechanism": (
            "Sleep deprivation lowers seizure threshold by reducing GABA-B receptor-mediated "
            "inhibition and increasing cortical excitability via adenosine clearance. "
            "CHD2-GGE patients particularly vulnerable: already reduced interneuron reserve → "
            "any further disinhibition crosses threshold. Morning post-awakening peak of "
            "myoclonic + GTCS directly related to sleep-wake transition excitability surge."
        ),
        "management": (
            "Regular sleep schedule mandatory (±30 min). "
            "Minimum 8-9h/night in school-age children; 7-8h in adolescents. "
            "Naps permissible but not sleep replacement. "
            "Sleep hygiene counselling at each clinic visit. "
            "Document sleep quality in seizure diary."
        ),
    },
    {
        "trigger": "Missed AED Dose",
        "prevalence_pct": 75,
        "mechanism": (
            "Sub-therapeutic VPA/LEV plasma levels remove residual inhibitory augmentation → "
            "net excitatory surge. VPA half-life 9-16h (ER formulation 12-17h) → single missed "
            "dose causes trough-to-zero in 24-36h. CHD2-GGE: breakthrough GTCS and prolonged "
            "absence status both documented after single missed dose."
        ),
        "management": (
            "Electronic pill reminders (app-based). "
            "VPA ER (extended-release) reduces peak-trough variability. "
            "If dose missed <4h: take immediately. "
            "If >4h and next dose due: skip missed dose, take next on schedule. "
            "Never double-dose (VPA toxicity: tremor, hyperammonaemia). "
            "Document breakthrough seizures after missed doses in diary."
        ),
    },
    {
        "trigger": "Alcohol",
        "prevalence_pct": 65,
        "mechanism": (
            "Acute alcohol → enhances GABA-A (inhibitory, sedative); "
            "alcohol WITHDRAWAL (3-12h after last drink) → rebound glutamate surge, "
            "GABA-A downregulation → seizure threshold lowered. "
            "CHD2-GGE: both acute (photosensitivity + myoclonic at party/nightclub) "
            "and withdrawal GTCS documented. Adolescent and adult patients at particular risk."
        ),
        "management": (
            "Complete avoidance recommended in drug-resistant CHD2. "
            "If drinking: maximum 1 unit/day; avoid binge. "
            "Avoid nightclubs/discos (combined alcohol + strobe lights = maximal risk). "
            "Counselling at adolescent transition (14-16 years)."
        ),
    },
    {
        "trigger": "Stress / Anxiety",
        "prevalence_pct": 55,
        "mechanism": (
            "Cortisol/CRH → downregulate GABA-A receptor surface expression (γ2 subunit "
            "endocytosis) → reduced inhibitory tone. HPA axis hyperactivation + amygdala "
            "sensitisation → limbic→cortical excitability spread. "
            "CHD2 ADHD comorbidity (50%) may amplify stress-seizure coupling via dopamine "
            "dysregulation in prefrontal inhibitory circuits."
        ),
        "management": (
            "Mindfulness-based stress reduction (MBSR). "
            "CBT for ADHD/anxiety comorbidities. "
            "Regular exercise (30 min/day; avoid hyperthermia). "
            "Seizure diary: log stress as potential trigger. "
            "Referral: neuropsychologist for ADHD assessment + management."
        ),
    },
    {
        "trigger": "Fever / Intercurrent Illness",
        "prevalence_pct": 45,
        "mechanism": (
            "Fever >38°C: sodium channel inactivation kinetics accelerated → higher neuronal "
            "firing rates; also GABA-A desensitisation by cytokine release (IL-1β, TNF-α). "
            "CHD2-Dravet-like subtype (20%): febrile GTCS are the initial presentation and "
            "remain a major risk throughout childhood; fever management critical."
        ),
        "management": (
            "Early antipyretic use (paracetamol/ibuprofen) at temperature ≥37.5°C. "
            "Rescue AED (midazolam buccal/nasal) prescribed at diagnosis for febrile seizures. "
            "Avoid overheating: tepid bath, cool environment. "
            "Flu vaccine annually (fever prevention > fever risk of vaccine). "
            "CHD2-Dravet-like: emergency letter documenting heat sensitivity."
        ),
    },
    {
        "trigger": "Hyperventilation",
        "prevalence_pct": 40,
        "mechanism": (
            "HV → hypocapnia → cerebral vasoconstriction → cortical hypoxia → alkalosis → "
            "reduced ionised Ca²⁺ → increased neuronal excitability → generalised 3-Hz SW "
            "and absence. Used diagnostically (3 min HV in EEG lab). "
            "Spontaneous HV during anxiety/exercise → breakthrough absence clusters."
        ),
        "management": (
            "HV used diagnostically in EEG lab (3 min provocation — safe, supervised). "
            "Spontaneous: coach breathing relaxation techniques. "
            "Vigorous aerobic exercise (→HV equivalent) managed with paced breathing. "
            "If exercise-induced absence common: dose VPA before anticipated exercise."
        ),
    },
    {
        "trigger": "Catamenial (Perimenstrual) Exacerbation",
        "prevalence_pct": 25,
        "mechanism": (
            "Pre-menstrual oestrogen:progesterone ratio rise → relative oestrogen excess "
            "at luteal-follicular transition → GABA-A receptor modulation by "
            "allopregnanolone drop → decreased neurosteroid inhibitory tone. "
            "CHD2-GGE females: catamenial GTCS clustering Days 24-28 of cycle documented. "
            "Contraception interactions: OCP may alter VPA metabolism (reduce levels 20%)."
        ),
        "management": (
            "Catamenial seizure diary (calendar-based). "
            "CLB pulse dosing (days 22-28) shown effective in catamenial GGE. "
            "Neurosteroid therapy (progesterone supplementation, ILAE Level C). "
            "OCP: avoid enzyme-inducing AEDs (not typically used in CHD2-GGE). "
            "Check VPA levels if OCP introduced (consider VPA-level monitoring q3M)."
        ),
    },
]

# ── Treatments ─────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproate (VPA) — Sodium Valproate / Valproic Acid",
        "evidence": "Level A",
        "indication": "All 3 seizure types (myoclonic + absence + GTCS) — broad-spectrum first-line for CHD2-GGE",
        "dose": (
            "Children: 20-40 mg/kg/day in 2-3 divided doses (ER: twice daily). "
            "Adults: 500-2000 mg/day. Target level 50-100 mg/L (myoclonic) / 75-120 mg/L (GTCS). "
            "Start low: 5 mg/kg/day; titrate weekly by 5 mg/kg increments."
        ),
        "moa": (
            "Sodium channel inactivation (reduces repetitive firing) + GABA transaminase inhibition "
            "(increases GABA) + HCN channel modulation (reduces Ih-mediated burst firing in "
            "thalamocortical relay neurons → reduces 3-Hz spike-wave generation)."
        ),
        "efficacy": "60-70% achieve ≥50% seizure reduction; 25-30% CHD2-GGE seizure-free",
        "safety": (
            "Teratogenicity (NTDs, ASD, DDI) — MANDATORY VPPP (Valproate Pregnancy Prevention "
            "Programme, MHRA 2021) for all females ≥12 years. "
            "Pancreatitis (rare, acute). Hyperammonaemic encephalopathy. "
            "Thrombocytopenia. Weight gain. Tremor dose-dependent. PCOS risk long-term females. "
            "Hepatotoxicity rare (1:37,000 all-ages; POLG mutation: 1:200 — MANDATORY POLG screen)."
        ),
        "monitoring": "VPA TDM q3M; LFT + ammonia + FBC q3M; VPPP annual review; weight BMI; POLG pre-treatment",
        "chd2_note": (
            "First-line for CHD2-GGE (all subtypes). "
            "CHD2-MEI: VPA + CLB combination often required. "
            "CHD2-photo-dominant: low-dose VPA (20-25 mg/kg/day) often sufficient + photic avoidance."
        ),
    },
    {
        "drug": "Ethosuximide (ETH)",
        "evidence": "Level A",
        "indication": "CHD2-absence-dominant (≥60% absence seizures; myoclonic/GTCS infrequent)",
        "dose": (
            "Children 3-6y: 15 mg/kg/day; Children >6y/adults: 500-1500 mg/day. "
            "Usual: 250 mg twice daily, titrate to 750-1500 mg/day over 4-8 weeks. "
            "Target level 40-100 mg/L."
        ),
        "moa": (
            "T-type Ca²⁺ channel blocker (Cav3.1/Cav3.2 in thalamocortical relay neurons) "
            "→ reduces low-threshold burst firing → disrupts thalamo-cortical 3-Hz oscillation "
            "generating generalised spike-wave. Has NO effect on GTCS or myoclonic (no Na-channel action)."
        ),
        "efficacy": "60-65% absence-free in absence-dominant GGE (SANAD II comparison VPA vs ETH vs LTG)",
        "safety": (
            "GI intolerance (nausea, vomiting) — take with food; slow titration. "
            "Headache. Lethargy. Blood dyscrasia (agranulocytosis — very rare; FBC q6M). "
            "No major drug interactions. NOT teratogenic (limited data). "
            "SLE-like reaction rare. DOES NOT cover GTCS — add VPA or LEV if GTCS emerges."
        ),
        "monitoring": "ETH level q6M; FBC q6M; LFT annually",
        "chd2_note": (
            "Use only when absence is the dominant seizure type AND GTCS/myoclonic absent or very infrequent. "
            "If GTCS emerge on ETH → switch or add VPA. "
            "Preferred in CHD2-photo-dominant subtype with minimal GTCS risk."
        ),
    },
    {
        "drug": "Levetiracetam (LEV)",
        "evidence": "Level B",
        "indication": "CHD2-myoclonic adjunct + GTCS adjunct; monotherapy if VPA intolerated",
        "dose": (
            "Children: 10-60 mg/kg/day in 2 divided doses (max 3000 mg/day). "
            "Adults: 500-3000 mg/day twice daily. "
            "Start 250 mg twice daily; titrate weekly by 250-500 mg increments."
        ),
        "moa": (
            "SV2A (synaptic vesicle protein 2A) binding → reduces vesicular glutamate release "
            "+ modulates high-voltage Ca²⁺ channels and GABA-A receptor trafficking. "
            "Anti-myoclonic mechanism likely via SV2A in hippocampal/cortical mossy fibres."
        ),
        "efficacy": "40-55% ≥50% reduction in CHD2-myoclonic; 35% reduction in GTCS as adjunct",
        "safety": (
            "Behavioural effects (irritability, aggression) in 20-30% — higher in CHD2-ADHD comorbidity. "
            "Somnolence. Depression (monitor mood). "
            "No hepatotoxicity. No teratogenicity signal (EURAP data — but data limited). "
            "No drug interactions (renal excretion unchanged). "
            "BRIVIACT (brivaracetam) = analogue with fewer behavioural effects if LEV poorly tolerated."
        ),
        "monitoring": "Behavioural checklist at each visit; renal function q12M; PHQ-9/GAD-7 adults",
        "chd2_note": (
            "CHD2 + ADHD (50%): monitor behavioural side effects carefully with LEV. "
            "Consider BRIVIACT (brivaracetam) as alternative if irritability limits LEV dose. "
            "LEV preferred over LTG for myoclonic component."
        ),
    },
    {
        "drug": "Lamotrigine (LTG) — CAUTION: Myoclonic Worsening",
        "evidence": "Level B — CAUTION",
        "indication": "CHD2-absence-dominant (GTCS without myoclonic) IF VPA/ETH intolerated; AVOID in myoclonic-dominant",
        "dose": (
            "Children (no VPA): 0.3 mg/kg/day titrating to 5-15 mg/kg/day over 12 weeks (slow!). "
            "Children (with VPA): 0.15 mg/kg/day titrating to 1-5 mg/kg/day (VPA doubles LTG levels). "
            "Adults: 100-400 mg/day; with VPA: 50-200 mg/day."
        ),
        "moa": (
            "Voltage-gated Na-channel (fast inactivation) + Ca-channel blockade → "
            "reduces repetitive neuronal firing. Active against GTCS and absence. "
            "DOES NOT enhance GABA → limited effect on myoclonic (may WORSEN by suppressing tonic "
            "inhibition without enhancing phasic GABA-A)."
        ),
        "efficacy": "Good for absence + GTCS (45-55%); UNRELIABLE/WORSENING for myoclonic (worsen 15-20%)",
        "safety": (
            "SJS/TEN risk (HLA-B*15:02 — MANDATORY screening in SE/EA Asian patients). "
            "Rash (7-10% — slow titration reduces risk). "
            "No behavioural side effects. Safe in pregnancy (best-evidenced AED). "
            "Drug interaction with VPA: VPA halves LTG clearance → double LTG levels → rash/toxicity."
        ),
        "monitoring": "HLA-B*15:02 before prescribing; titrate SLOWLY; rash protocol; LTG level optional",
        "chd2_note": (
            "CHD2 WARNING: LTG may WORSEN myoclonic jerks (15-20% aggravation rate, higher than JME). "
            "Perform EEG with IPS BEFORE starting LTG to confirm myoclonic absence. "
            "If myoclonic present → prefer LEV over LTG. "
            "Safe and effective for absence-dominant CHD2 without significant myoclonic component."
        ),
    },
    {
        "drug": "Clobazam (CLB)",
        "evidence": "Level B",
        "indication": "CHD2-MEI (drop attacks + severe myoclonic clusters); adjunct therapy; catamenial pulse dosing",
        "dose": (
            "Children: 0.1-0.3 mg/kg/day (max 1 mg/kg/day) in 1-2 doses. "
            "Adults: 10-40 mg/day (max 80 mg in DRE). "
            "Start: 5-10 mg/day; titrate weekly."
        ),
        "moa": (
            "GABA-A positive allosteric modulator (benzodiazepine site, BZD-1/BZD-2). "
            "High selectivity for α2/α3-containing GABA-A receptors (vs. α1 — less sedation). "
            "Reduces myoclonic, tonic, and atonic seizures. Catamenial: pulse dosing days 22-28."
        ),
        "efficacy": "50-60% ≥50% reduction in drop attacks/severe myoclonic as adjunct; tolerance develops over months",
        "safety": (
            "Tolerance (months-years). Sedation. Ataxia. Disinhibition in children. "
            "Dependence with chronic high-dose. Respiratory depression at high doses "
            "(not a concern at typical AED doses unless combined with opioids). "
            "Withdrawal: taper slowly (reduce by 10%/2 weeks minimum)."
        ),
        "monitoring": "CLB level optional; sedation/cognition assessment q3M; taper plan documented",
        "chd2_note": (
            "CHD2-MEI: VPA+CLB is the standard combination backbone. "
            "Catamenial CHD2: CLB pulse dosing (5-10 mg/day days 22-28) reduces perimenstrual clustering. "
            "Tolerance management: drug holiday (2-4 week break) may restore sensitivity."
        ),
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "evidence": "Level B",
        "indication": "CHD2-MEI (refractory drop attacks + myoclonic); CHD2-Dravet-like (DRE); after 2 AED failures",
        "dose": (
            "Standard 4:1 (fat:protein+carbohydrate) or MCT oil diet. "
            "Target urine ketones 4-8 mmol/L (Ketostix 2+/3+). "
            "Dietitian-supervised; 3-month trial minimum before assessment of efficacy."
        ),
        "moa": (
            "β-hydroxybutyrate (βHB) → adenosine kinase inhibition → adenosine↑ → A1R "
            "activation → reduces neuronal excitability. Also: KATP channel opening, "
            "mTORC1 suppression, direct GABA-A potentiation, mitochondrial biogenesis, "
            "and anti-inflammatory (NF-κB inhibition). CHD2 hypothesis: KD may partially "
            "compensate for reduced GABAergic interneuron function via βHB GABA augmentation."
        ),
        "efficacy": "~40-50% CHD2 responders (≥50% seizure reduction); 10-15% seizure-free",
        "safety": (
            "Dyslipidaemia (LDL rise 30-50%); renal stones 3-5%; growth suppression long-term; "
            "GI intolerance (constipation, reflux); selenium/carnitine deficiency. "
            "Annual DXA bone density in long-term KD. Contraindicated: fatty acid oxidation disorders. "
            "Drug interactions: VPA + KD → hepatotoxicity risk (monitor LFT + ammonia closely)."
        ),
        "monitoring": "Ketones daily; lipid panel q3M; LFT q3M (esp. with VPA); DXA annually; micronutrients q6M",
        "chd2_note": (
            "CHD2-MEI: KD after 2 AED failures is Level B evidence (ILAE Dietary Therapies 2018). "
            "CHD2-photo: photic avoidance + VPA usually sufficient — KD rarely needed. "
            "VPA+KD: monitor LFT + ammonia monthly for first 6 months (hepatotoxicity risk). "
            "CHD2-Dravet-like: KD often started earlier (similar to Dravet management)."
        ),
    },
    {
        "drug": "Phenobarbital / Bromides (PB/KBr)",
        "evidence": "Level C",
        "indication": "Refractory CHD2-MEI with dominant photosensitive myoclonic and drop attacks; salvage only",
        "dose": (
            "PB: Children 3-5 mg/kg/day; Adults 60-180 mg/day (monitor level 15-40 mg/L). "
            "KBr: 30-40 mg/kg/day (particularly effective for photosensitive GGE in Germany/Japan data). "
            "Monitor PB level + liver function."
        ),
        "moa": (
            "PB: GABA-A positive allosteric modulator (barbiturate site → Cl⁻ channel opening duration). "
            "KBr: chloride loading → hyperpolarises neuronal membrane → raises seizure threshold; "
            "also reduces PPR (direct anti-photosensitive effect — MOA poorly understood)."
        ),
        "efficacy": "PB: 30-40% reduction in refractory CHD2; KBr: 40-60% PPR reduction (anti-photosensitive specialist use)",
        "safety": (
            "PB: sedation, cognitive impairment, DRESS (rare), bone density loss long-term, "
            "teratogenic (Category D), enzyme induction (reduces many drug levels). "
            "KBr: acneiform rash (bromoderma), sedation, nausea; not available in all countries."
        ),
        "monitoring": "PB level q3M; LFT; cognition/behaviour; bone density long-term",
        "chd2_note": (
            "Salvage therapy only in refractory CHD2-MEI with dominant photosensitive seizures. "
            "KBr used in Germany/Japan for photo-GGE — significant anti-photosensitive property. "
            "PB enzyme induction reduces VPA/LEV/CLB levels — monitor therapeutic drug levels. "
            "Fenfluramine (Level C, compassionate) being evaluated for CHD2-MEI in US/EU centres."
        ),
    },
]

# ── Contraindications ──────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine / Oxcarbazepine / Phenytoin (CBZ/OXC/PHT)",
        "risk_level": "ABSOLUTE CONTRAINDICATION",
        "reason": (
            "GGE aggravation: Na-channel blockers without GABAergic enhancement → "
            "paradoxical worsening of myoclonic jerks + absence + GTCS in generalised epilepsies. "
            "Mechanism: selective suppression of cortical surround inhibition without blocking "
            "thalamic T-type Ca²⁺ channels → net increase in 3-Hz spike-wave oscillation. "
            "Documented: CBZ/OXC → precipitate absence status epilepticus and myoclonic status "
            "in CHD2/GGE patients. PHT: same mechanism + long-term cerebellar atrophy."
        ),
        "alternative": "VPA (all seizure types) or ETH (absence-only) or LEV (myoclonic/GTCS)",
    },
    {
        "drug": "Tiagabine (TGB)",
        "risk_level": "ABSOLUTE CONTRAINDICATION",
        "reason": (
            "Non-convulsive status epilepticus (NCSE) induction: GAT-1 (GABA transporter) "
            "inhibition → excessive phasic GABA spillover → paradoxical tonic depolarisation "
            "block in cortical interneurons → absence status / NCSE in GGE patients. "
            "Multiple case reports of tiagabine-induced NCSE in GGE (all subtypes). "
            "Extreme risk in photosensitive CHD2-GGE."
        ),
        "alternative": "CLB (adjunct GABAergic) or VPA (safer GABA-transaminase inhibitor)",
    },
    {
        "drug": "Valproate — Females of Child-Bearing Age Without VPPP",
        "risk_level": "HIGH RISK — MANDATORY SAFEGUARD",
        "reason": (
            "VPA teratogenicity: NTD risk 10× (spina bifida), ASD 4×, cognitive impairment "
            "(mean IQ reduction 6-9 points), PCOS, DDI. "
            "MHRA 2021 VPPP (Valproate Pregnancy Prevention Programme) mandatory in UK for ALL "
            "females ≥12 years on VPA: annual contraception review + specialist acknowledgement form. "
            "In CHD2-GGE (VPA is Level A first-line): VPPP compliance = non-negotiable."
        ),
        "alternative": "LEV (pregnancy-safest GGE AED per EURAP); LTG (if no significant myoclonic); contraception discussion",
    },
    {
        "drug": "Valproate — Without POLG Screening",
        "risk_level": "HIGH RISK — MANDATORY SCREEN",
        "reason": (
            "POLG1 (mitochondrial DNA polymerase gamma) mutations → Alpers-Huttenlocher syndrome. "
            "VPA in POLG patients: hepatic failure (fatal, 1:200 risk in POLG carriers) via "
            "VPA β-oxidation inhibition + mtDNA depletion synergy. "
            "CHD2-GGE: if POLG mutation present alongside CHD2 (rare but documented de novo), "
            "VPA is absolutely contraindicated → use LEV+CLB instead."
        ),
        "alternative": "Screen POLG BEFORE VPA. If POLG+: LEV+CLB+KD regimen (VPA-free)",
    },
    {
        "drug": "Lamotrigine Monotherapy — Without Myoclonic EEG Assessment",
        "risk_level": "MODERATE RISK — EEG GATE REQUIRED",
        "reason": (
            "LTG may worsen myoclonic jerks in CHD2-GGE (15-20% aggravation rate). "
            "If myoclonic present on EEG (polyspike-wave) → LTG may increase myoclonic "
            "frequency and severity. "
            "Required: EEG with IPS before LTG initiation to confirm myoclonic absence. "
            "If PPR + polyspike-wave confirmed → avoid LTG; prefer LEV."
        ),
        "alternative": "LEV (safe for myoclonic); VPA if tolerated; EEG-gated LTG initiation protocol",
    },
]

# ── Monitoring ──────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG mutation screen", "frequency": "ONCE — mandatory before VPA initiation", "rationale": "Fatal hepatic failure risk in POLG mutation carriers"},
    {"item": "VPPP annual review", "frequency": "Annually (all females ≥12y on VPA)", "rationale": "MHRA 2021 mandatory contraception + risk acknowledgement"},
    {"item": "VPA TDM (plasma level)", "frequency": "q3M (or after dose change)", "rationale": "Target 50-100 mg/L myoclonic; 75-120 mg/L GTCS control"},
    {"item": "LFT + ammonia + FBC", "frequency": "q3M (VPA)", "rationale": "Hepatotoxicity screening + hyperammonaemia detection"},
    {"item": "EEG with IPS (photic stimulation)", "frequency": "Annually + before LTG initiation", "rationale": "PPR monitoring; myoclonic assessment; LTG-gate (EEG-gated prescribing)"},
    {"item": "Seizure diary (digital)", "frequency": "Continuous (reviewed each clinic)", "rationale": "Document triggers (photostimulation, sleep-dep, alcohol), catamenial pattern"},
    {"item": "HLA-B*15:02 screen", "frequency": "ONCE — before CBZ/OXC/LTG (SE/EA Asian ancestry)", "rationale": "SJS/TEN risk reduction"},
    {"item": "Neuropsychological assessment", "frequency": "q2 years", "rationale": "ID severity, ADHD, ASD features, academic function"},
    {"item": "Photic avoidance counselling", "frequency": "At diagnosis + annually", "rationale": "FL-41 spectacles, screen settings, outdoor behaviour — reduces PPR seizures"},
    {"item": "SUDEP annual risk discussion", "frequency": "Annually", "rationale": "≥1 GTCS/year → elevated SUDEP risk; document in notes"},
    {"item": "Driving regulations", "frequency": "At diagnosis (adolescent+) + each GTCS", "rationale": "Jurisdiction-specific: 12 months seizure-free (UK) / 1 year (Canada/Australia)"},
    {"item": "Genetic counselling (cascade)", "frequency": "Once (at diagnosis) + pre-conception", "rationale": "AD de novo; recurrence <1% parental; pre-conception VPA planning"},
    {"item": "DXA bone density (KD patients)", "frequency": "Annually (on KD)", "rationale": "Long-term KD: bone density loss risk"},
    {"item": "Lipid panel + ketones (KD patients)", "frequency": "Lipids q3M; ketones daily (KD)", "rationale": "KD dyslipidaemia + metabolic monitoring"},
]

# ── Lifecycle ──────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Neonatal (0-1 month)",
        "label": "Asymptomatic / Genetic suspicion",
        "key_actions": [
            "CHD2 identified on family panel (sibling/parent with CHD2)",
            "No intervention; developmental monitoring initiated",
            "Genetic counselling for parents",
            "EEG: normal at this stage",
        ],
    },
    {
        "window": "Infancy (1-12 months)",
        "label": "MEI onset / Febrile seizures",
        "key_actions": [
            "CHD2-MEI: myoclonic cluster onset 6-12 months",
            "CHD2-Dravet-like: first prolonged febrile HS (6-18 months)",
            "VPA initiated after POLG screening; VPPP documented if female",
            "Video EEG: myoclonic + IPS provocation protocol",
            "Developmental assessment: motor + language milestones",
        ],
    },
    {
        "window": "Early Childhood (1-5 years)",
        "label": "GGE emergence / Photosensitivity peak",
        "key_actions": [
            "Classic CHD2-GGE onset: myoclonic + absence + GTCS",
            "Photosensitivity maximal (IPS PPR peak age 4-8 years)",
            "EEG with IPS: annual monitoring; LTG-gate if LTG considered",
            "Photic avoidance counselling + FL-41 spectacles",
            "VPA ± ETH (absence) ± LEV (myoclonic adjunct)",
            "ADHD first identified — neuropsychological referral",
        ],
    },
    {
        "window": "Childhood (5-12 years)",
        "label": "School-age epilepsy management",
        "key_actions": [
            "Seizure diary: photostimulation triggers (classroom, gaming, sports)",
            "School seizure action plan: rescue AED + photo-avoidance policy",
            "AED optimisation: VPA dose adjusted for weight; TDM quarterly",
            "Neuropsychological: ADHD/ASD formal assessment + IEP if needed",
            "ASD/ID comorbidity management: occupational therapy, speech therapy",
            "Annual EEG with IPS: document PPR trend (may improve in adolescence)",
        ],
    },
    {
        "window": "Adolescence (12-18 years)",
        "label": "VPPP, alcohol, driving counselling",
        "key_actions": [
            "VPPP: annual female counselling (contraception + VPA risk acknowledgement form)",
            "Alcohol counselling: complete avoidance or 1 unit maximum",
            "Driving: country-specific rules (1 year seizure-free for most); document at first GTCS",
            "Screen-avoidance: gaming, nightclubs (strobe), social media reels",
            "ADHD medications: stimulants generally safe in CHD2-GGE (avoid excitatory risk; monitor)",
            "Mental health: PHQ-9/GAD-7; ADHD + depression co-occur in 30-40%",
        ],
    },
    {
        "window": "Adult (18+ years)",
        "label": "Employment, pregnancy planning, long-term management",
        "key_actions": [
            "Pre-conception counselling: VPA → alternative AED in planning pregnancy (LEV/LTG)",
            "Employment: photosensitivity risk assessment for screen-intensive work",
            "Seizure freedom ≥2 years: AED withdrawal discussion (risk of relapse 30%)",
            "Annual SUDEP review: ≥1 GTCS/year → document + SuperVIA bed/SUDEP-GP",
            "Long-term VPA: DXA bone density; PCOS screening in females",
            "Cognitive monitoring: IQ stable in most GGE-photo; progressive decline in MEI",
        ],
    },
]

# ── Concepts ───────────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "CHD2 (15q26.1)", "definition": "Chromodomain Helicase DNA Binding Protein 2; 240-kDa chromatin remodeling enzyme; SWI/SNF superfamily; H3.3 histone deposition; regulates GABAergic interneuron developmental promoters"},
    {"term": "GGE (Genetic Generalised Epilepsy)", "definition": "ILAE 2022 classification: absence + myoclonic + GTCS on generalised EEG background; thalamocortical network epilepsy; autosomal dominant in CHD2"},
    {"term": "MEI (Myoclonic Encephalopathy of Infancy)", "definition": "CHD2-specific severe early-onset phenotype (6-18 months); near-continuous myoclonic + status myoclonicus; severe ID; drug-resistant"},
    {"term": "PPR (Photoparoxysmal Response)", "definition": "EEG: bilateral generalised spike-wave triggered by IPS; CHD2 has PPR in 75-80%; Waltz 1992 grading (Grade 1-4); cardinal CHD2 EEG finding"},
    {"term": "EC-IPA (Eye Closure-Induced Paroxysmal Activity)", "definition": "Generalised spike-wave burst immediately on eye closure; high CHD2 specificity; mechanism: visual cortex hyperexcitability with CHD2 LOF"},
    {"term": "H3.3 Histone Deposition", "definition": "CHD2 deposits H3.3 (replication-independent histone variant) at newly opened chromatin → stable active chromatin marks; CHD2 LOF → failed H3.3 deposition → closed GABAergic gene promoters"},
    {"term": "FL-41 Spectacles", "definition": "Rose/amber-tinted lenses (FL-41 filter); reduces PPR by 40-60% in photosensitive epilepsies; prescribed routinely in CHD2 at diagnosis; OTC and prescription available"},
    {"term": "VPPP (Valproate Pregnancy Prevention Programme)", "definition": "MHRA 2021 mandatory programme: annual contraception review + specialist acknowledgement form for all females ≥12 years on VPA; CHD2-GGE key compliance requirement"},
    {"term": "POLG (Mitochondrial DNA Polymerase Gamma)", "definition": "POLG1 mutation → Alpers-Huttenlocher syndrome; VPA in POLG patients → fatal hepatic failure; mandatory POLG screen BEFORE VPA initiation"},
    {"term": "GGE Aggravation Syndrome", "definition": "Paradoxical worsening of absence + myoclonic + GTCS by Na-channel blockers (CBZ/OXC/PHT); mechanism: cortical surround inhibition suppression without thalamic T-Ca block"},
    {"term": "Catamenial Epilepsy", "definition": "Perimenstrual seizure clustering (luteal-follicular transition); CHD2: CLB pulse dosing days 22-28 is first-line; allopregnanolone progesterone therapy Level C"},
    {"term": "SV2A (Synaptic Vesicle Protein 2A)", "definition": "LEV/brivaracetam binding target; reduces vesicular glutamate release; anti-myoclonic mechanism in GGE; safer than LTG for myoclonic component"},
    {"term": "SUDEP (Sudden Unexpected Death in Epilepsy)", "definition": "Risk elevated with ≥1 GTCS/year; CHD2-GGE SUDEP rate ~1:1000 patient-years; document annual SUDEP counselling; SuperVIA bed-monitor discussed"},
    {"term": "ETH (Ethosuximide)", "definition": "T-type Ca²⁺ channel blocker (Cav3.1/3.2 in thalamic relay neurons); Level A absence-only GGE; NO anti-myoclonic or anti-GTCS effect; monitor FBC"},
    {"term": "HLA-B*15:02", "definition": "HLA allele in SE/EA Asian populations; SJS/TEN risk with CBZ/OXC/PHT/LTG; mandatory screen before prescribing; CBZ absolutely contraindicated in CHD2-GGE regardless of HLA"},
]

# ── Thresholds ─────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"parameter": "VPA plasma level (myoclonic control)", "value": "50–100 mg/L", "unit": "mg/L"},
    {"parameter": "VPA plasma level (GTCS control)", "value": "75–120 mg/L", "unit": "mg/L"},
    {"parameter": "ETH plasma level", "value": "40–100 mg/L", "unit": "mg/L"},
    {"parameter": "CLB plasma level", "value": "30–300 ng/mL", "unit": "ng/mL"},
    {"parameter": "PPR grade (Waltz 1992) for photic avoidance protocol", "value": "Grade ≥3", "unit": "Grade 1-4"},
    {"parameter": "Ammonia (VPA toxicity threshold)", "value": ">100 µmol/L", "unit": "µmol/L"},
    {"parameter": "ALT/AST (VPA hepatotoxicity alert)", "value": ">3× ULN", "unit": "× ULN"},
    {"parameter": "Urine ketones (KD target)", "value": "4–8 mmol/L (Ketostix 2+/3+)", "unit": "mmol/L"},
    {"parameter": "LDL (KD dyslipidaemia alert)", "value": ">4.0 mmol/L", "unit": "mmol/L"},
    {"parameter": "Seizure-free period (driving eligibility)", "value": "12 months", "unit": "months"},
    {"parameter": "VPPP female age threshold", "value": "≥12 years", "unit": "years"},
    {"parameter": "GTCS rate (SUDEP counselling trigger)", "value": "≥1 per year", "unit": "GTCS/year"},
]

# ── Standards ──────────────────────────────────────────────────────────────────
STANDARDS = [
    {"standard": "ILAE Classification 2022", "relevance": "GGE classification; CHD2 listed in GGE gene panel; seizure-type taxonomy"},
    {"standard": "NICE NG217 (2022)", "relevance": "UK epilepsy management guideline; VPA VPPP mandatory; referral pathways"},
    {"standard": "Carvill et al. 2013 (Nat Genet)", "relevance": "First CHD2 epilepsy report; 9 patients; de novo LOF; GGE + photosensitivity established"},
    {"standard": "Thomas et al. 2015 (Am J Hum Genet)", "relevance": "Extended CHD2 cohort (93 patients); phenotype spectrum; MEI subtype defined"},
    {"standard": "Lund et al. 2014 (Epilepsy Behav)", "relevance": "CHD2 photosensitivity characterisation; PPR grade + prevalence; FL-41 recommendations"},
    {"standard": "Bhatt et al. 2023 (Neurology)", "relevance": "GGE gene panel recommendations; CHD2 included in Tier 1 GGE panel"},
    {"standard": "MHRA VPPP 2021", "relevance": "Mandatory Valproate Pregnancy Prevention Programme (UK); CHD2-GGE primary compliance target"},
    {"standard": "CPIC HLA-B*15:02 2023", "relevance": "Mandatory HLA-B*15:02 screen before LTG/CBZ/OXC in SE/EA Asian ancestry"},
    {"standard": "FDA Valproate REMS", "relevance": "US equivalent of VPPP; risk communication materials for VPA in females"},
    {"standard": "ILAE Dietary Therapies 2018", "relevance": "KD Level B recommendation after 2 AED failures in GGE-refractory"},
    {"standard": "ACMG-AMP 2015", "relevance": "Variant classification criteria (P/LP/VUS) for CHD2 ClinVar reporting"},
    {"standard": "Waltz 1992 PPR Grading", "relevance": "PPR Grade 1-4 classification for IPS EEG reporting; Grade ≥3 → photic avoidance protocol"},
]

# ── References ─────────────────────────────────────────────────────────────────
REFERENCES = [
    {"ref": "Carvill 2013", "citation": "Carvill GL et al. (2013). Targeted resequencing in epileptic encephalopathies identifies de novo mutations in CHD2 and SYNGAP1. Nat Genet 45(8):825-830. PMID: 23708187"},
    {"ref": "Thomas 2015", "citation": "Thomas RH et al. (2015). CHD2 myoclonic encephalopathy is frequently associated with self-induced seizures. Neurology 84(9):951-958. PMID: 25653293"},
    {"ref": "Lund 2014", "citation": "Lund C et al. (2014). CHD2 epilepsy with photosensitivity and myoclonic encephalopathy: phenotype spectrum from 24 patients. Epilepsy Behav 39:112-116. PMID: 24954795"},
    {"ref": "Bhatt 2023", "citation": "Bhatt DL et al. (2023). Epilepsy gene panel recommendations and CHD2 GGE classification. Neurology 101(4):e418-e427"},
    {"ref": "Marguet 2022", "citation": "Marguet F et al. (2022). CHD2 encephalopathy in infancy: clinico-EEG-genetic study of 30 patients and literature review. Epilepsia 63(3):652-665. PMID: 35072247"},
    {"ref": "Suls 2013", "citation": "Suls A et al. (2013). De novo loss-of-function mutations in CHD2 cause a fever-sensitive myoclonic epileptic encephalopathy sharing features with Dravet syndrome. Am J Hum Genet 93(5):967-975. PMID: 24207120"},
]

# ── Overview KPI builder ───────────────────────────────────────────────────────
def get_overview() -> dict:
    n = len(PATIENTS)
    photo_n = sum(1 for p in PATIENTS if p["photosensitive"])
    seizure_free = sum(1 for p in PATIENTS if p["outcome"] == "Seizure-free")
    dr = sum(1 for p in PATIENTS if p["outcome"] == "Drug-resistant")
    return {
        "gene": "CHD2",
        "locus": "15q26.1",
        "omim": "#615369",
        "protein": "Chromodomain Helicase DNA Binding Protein 2 (CHD2)",
        "inheritance": "Autosomal Dominant (>90% de novo)",
        "syndrome": "GGE-Photosensitive / Myoclonic Encephalopathy of Infancy (MEI) / DEE32",
        "mechanism": (
            "CHD2 LOF haploinsufficiency → reduced H3.3 deposition at GABAergic gene promoters "
            "(DLX1/DLX2/GAD1/GABRB3) → under-differentiation of PV+/SST+ interneurons → cortical "
            "disinhibition → GGE. Visual cortex GABRB3/PRKCG chromatin defect → severe "
            "photosensitivity (PPR 75-80%) — one of the most photosensitive genetic epilepsies."
        ),
        "cohort_size": n,
        "photosensitive_n": photo_n,
        "photosensitive_pct": round(photo_n / n * 100),
        "seizure_free_n": seizure_free,
        "seizure_free_pct": round(seizure_free / n * 100),
        "drug_resistant_n": dr,
        "drug_resistant_pct": round(dr / n * 100),
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
            {"label": "Photosensitive", "value": f"{round(photo_n/n*100)}%", "color": "#c77d00"},
            {"label": "Seizure-free", "value": f"{round(seizure_free/n*100)}%", "color": "#1a6b1a"},
            {"label": "Drug-resistant", "value": f"{round(dr/n*100)}%", "color": "#8a1c1c"},
            {"label": "GTCS risk", "value": "55%", "color": "#1a4080"},
            {"label": "ADHD comorbid", "value": "50%", "color": "#5a1080"},
            {"label": "pLI", "value": "0.99", "color": "#2a6080"},
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
            {"trigger": t["trigger"], "prevalence_pct": t["prevalence_pct"]}
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
