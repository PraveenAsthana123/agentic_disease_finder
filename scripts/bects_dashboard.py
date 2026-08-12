"""Self-Limited Epilepsy with Centrotemporal Spikes (SeLECTS / BECTS / Rolandic Epilepsy)

The most common benign epilepsy syndrome of school-age children (15–25% of childhood epilepsy).
Previously called BECTS (Benign Epilepsy with Centrotemporal Spikes) or Rolandic Epilepsy;
reclassified by ILAE 2022 as Self-Limited Epilepsy with Centrotemporal Spikes (SeLECTS) to
acknowledge rare atypical evolutions (CSWS/ESES, Landau-Kleffner Syndrome).

HALLMARKS:
  Centrotemporal Spikes (CTS): High-amplitude, diphasic, horizontal-dipole spikes at C3/C4/T7/T8;
    activated by drowsiness and NREM sleep; may be bilateral/shifting; can occur without clinical
    seizures (important: seizure frequency does NOT correlate with CTS density on EEG).
  Oro-facial Sensorimotor Seizures: Tongue tingling, unilateral facial jerking, hypersalivation,
    dysarthria, guttural sounds, laryngeal contraction — speech arrest common; consciousness
    preserved in focal seizures; child can hear + understand but cannot speak or swallow.
  Nocturnal GTCS: Secondary generalisation often nocturnal — 45% of patients; parents alarmed
    but prognosis unaffected; may be the presenting event while daytime focal seizures go unnoticed.
  Self-Limited Course: 95% seizure-free by age 16; spontaneous remission by puberty is the rule;
    treatment decision should weigh seizure impact vs side-effect burden of a self-limiting condition.

SPECTRUM & VARIANTS:
  Typical SeLECTS: Classic presentation above; minimal cognitive impact; excellent prognosis.
  Atypical Evolution → CSWS/ESES: Continuous Spike-Wave during Sleep (slow-wave index >85% in
    NREM); associated with transient neuropsychological regression; GRIN2A mutations prevalent;
    requires aggressive AED management (VPA/CLB/sulthiame/IVIG in refractory cases).
  Landau-Kleffner Syndrome (LKS): Acquired epileptic aphasia (auditory agnosia → verbal agnosia);
    CTS-dominant EEG during sleep; overlaps with CSWS; severe language regression; rare; may
    require high-dose steroids / IVIG.
  Panayiotopoulos Syndrome Overlap: Shares genetic-electroclinical spectrum; occipital spikes;
    autonomic seizures (ictal vomiting/eye deviation); different clinical but same EEG-gene family.

NEUROPSYCHOLOGY:
  ~50% of children show transient difficulties in language processing, attention, visuospatial skills
  during the active phase; typically subclinical and resolve post-remission; persistent cognitive
  impact uncommon in typical SeLECTS; neuropsychological assessment recommended if learning concerns.

GENETICS:
  Polygenic susceptibility (~70%): oligogenic with GABA/glutamate-receptor gene clusters.
  GRIN2A (GluN2A subunit, ionotropic NMDA receptor): found in 9–15% of typical BECTS; in 20–27%
    of CSWS/LKS atypical end; dominant variants cause spectrum ranging from SeLECTS to severe
    CSWS/LKS — key genetic test in atypical presentations.
  RBFOX3 / DEPDC5 rare variants: modifiers in families; not routine clinical testing.

FIRST-LINE / ADJUNCT TREATMENT:
  No Treatment (watch-and-wait): Recommended in children with infrequent nocturnal-only seizures,
    no psychosocial impact; up to 25% managed without AEDs given self-limited natural history.
  Oxcarbazepine (OXC): Level A — most widely used; 10–30 mg/kg/day; highly effective in SeLECTS;
    may carry small risk of aggravating CSWS in susceptible children — monitor EEG.
  Levetiracetam (LEV): Level B — well-tolerated; 20–40 mg/kg/day; preferred if OXC avoided;
    good safety profile; behaviour/mood monitoring (LEV rage/emotional liability in paediatrics).
  Sulthiame (STM): Level A (European RCT evidence — SCORE study 2010); 5–10 mg/kg/day;
    carbonic anhydrase inhibitor; not available in all countries (not licensed in USA/Canada).
  Valproate (VPA): Level B — used as second-line or for secondarily generalised seizures;
    20–40 mg/kg/day; REMS for females of childbearing potential; weight gain + hair loss in children.
  Carbamazepine (CBZ): Historically first-line; concern that CBZ may occasionally aggravate
    atypical evolution to CSWS/ESES — monitor EEG in first 3 months; still commonly used.
  Lamotrigine (LTG): Level C — slow titration essential; may be used in adolescents approaching
    remission when long-term coverage needed; SJS risk.

RELATIVE CONTRAINDICATIONS (no hard absolutes in SeLECTS):
  Vigabatrin (VGB): Worsens focal seizures and can provoke CSWS; visual field loss risk (irreversible)
    — should not be used in SeLECTS.
  Unnecessary prolonged treatment: Continuation >2 years beyond last seizure inappropriate; most
    guidelines recommend AED taper after 1–2 years seizure-free given self-limited nature.
  Carbamazepine (CBZ) in GRIN2A+/atypical phenotype: May aggravate CSWS/cognitive regression;
    switch to OXC or LEV preferred in atypical presentations.

References:
  - Loiseau P & Duche B 1989 Epilepsia (168-patient series, natural history of BECTS)
  - Beaussart M 1972 Epilepsia (Classic description of Rolandic epilepsy seizure semiology)
  - Scheffer IE et al. 2017 Epilepsia (ILAE classification — SeLECTS reclassification)
  - Lemke JR et al. 2013 Nat Genet (GRIN2A mutations in epilepsy-aphasia spectrum)
  - Wirrell EC 1998 Epilepsia (Outcome: 95% remission by puberty — cohort study)
  - Caraballo RH et al. 2010 Epilepsia (SCORE RCT — sulthiame vs placebo Level A evidence)
Data: live clinical.db (41 epilepsy patients, deterministic SeLECTS overlay)
      + curated SeLECTS pharmacology / etiology / seizure-type / trigger catalogs."""

import sqlite3
import json
from pathlib import Path
from datetime import date

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"
_PROJECT = Path(__file__).resolve().parent.parent


# ─── helpers ────────────────────────────────────────────────────────────────

def _db_rows(sql, params=()):
    try:
        con = sqlite3.connect(DB)
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _seed(pid):
    """Deterministic hash from patient_id string."""
    h = 0
    for c in str(pid):
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


# ─── Etiology / Genetic catalog ─────────────────────────────────────────────

ETIOLOGY_CATALOG = [
    {
        "etiology": "Polygenic / Unknown",
        "pct": 68,
        "category": "Polygenetic",
        "mechanism": (
            "Oligogenic susceptibility across GABA-receptor, glutamate-receptor, and ion-channel "
            "gene clusters; CTS-generating cortex shows maturational hyperexcitability; "
            "spontaneously outgrown with cortical maturation (myelination of rolandic cortex)."
        ),
        "eeg_correlate": (
            "Classic high-amplitude diphasic CTS with horizontal dipole (C3/C4–T7/T8); "
            "activated in NREM sleep; frequency 1–3 Hz; normal background."
        ),
        "mri_finding": "MRI normal in typical SeLECTS (diagnostic criterion).",
        "clinical_note": (
            "No genetic testing required for typical SeLECTS. Consider GRIN2A panel only if "
            "atypical evolution (CSWS, language regression, treatment resistance) is observed."
        ),
    },
    {
        "etiology": "GRIN2A Mutation",
        "pct": 12,
        "category": "Monogenic",
        "mechanism": (
            "Gain-of-function or dominant-negative variants in GRIN2A (GluN2A NMDA receptor "
            "subunit); increased thalamo-cortical excitability; phenotype spectrum from typical "
            "SeLECTS → CSWS → Landau-Kleffner Syndrome depending on variant severity."
        ),
        "eeg_correlate": (
            "CTS density often higher; sleep EEG may show continuous or near-continuous spike-wave "
            "(slow-wave index approaching CSWS threshold); vigilance-dependent activation prominent."
        ),
        "mri_finding": "MRI normal; rarely subtle perisylvian cortical asymmetry on high-resolution MRI.",
        "clinical_note": (
            "GRIN2A testing recommended in: atypical evolution, CSWS on sleep EEG, speech regression, "
            "family history of epilepsy-aphasia spectrum, or treatment resistance."
        ),
    },
    {
        "etiology": "CSWS / Atypical Evolution",
        "pct": 8,
        "category": "Atypical SeLECTS",
        "mechanism": (
            "Evolution from typical SeLECTS to Continuous Spike-Wave during Slow Sleep (CSWS/ESES); "
            "slow-wave index >85% in NREM; thalamo-cortical hypersynchrony during sleep disrupts "
            "memory consolidation and language network plasticity → neuropsychological regression."
        ),
        "eeg_correlate": (
            "Slow-wave index >85% during NREM sleep on overnight EEG; bilateral synchronous "
            "slow-spike-wave 2–2.5 Hz; atypical daytime EEG with theta-delta slowing."
        ),
        "mri_finding": "MRI may show subtle cortical asymmetry; fMRI language mapping abnormal in severe cases.",
        "clinical_note": (
            "MANDATORY: overnight/sleep EEG annually in children with SeLECTS and any language/attention "
            "regression to exclude CSWS. Treatment intensification (VPA/CLB/high-dose STM) may be needed "
            "to abort CSWS before irreversible language regression occurs."
        ),
    },
    {
        "etiology": "Landau-Kleffner Syndrome (LKS)",
        "pct": 4,
        "category": "Severe Spectrum",
        "mechanism": (
            "Acquired epileptic aphasia with verbal agnosia progressing to complete language loss; "
            "perisylvian/temporal CTS-dominant pattern during sleep; autoimmune-like mechanism "
            "proposed (some respond to steroids/IVIG); NEU1 or GRIN2A pathway implicated."
        ),
        "eeg_correlate": (
            "Bitemporal spike-wave during sleep (NREM-dominant); CTS maximal at T7/T8; "
            "language cortex mapped to ictal zone; no overt clinical seizures in some cases."
        ),
        "mri_finding": "MRI normal or mild temporal signal change; PET shows glucose hypometabolism in temporal language areas.",
        "clinical_note": (
            "LKS is distinct from typical SeLECTS but shares genetic-electroclinical spectrum. "
            "Referral to specialist centre mandatory. IVIG / oral steroids / ACTH pulse may be required. "
            "High-dose diazepam per-rectal taper can temporarily arrest CSWS in LKS."
        ),
    },
    {
        "etiology": "Familial / Other Genetic",
        "pct": 8,
        "category": "Familial",
        "mechanism": (
            "First-degree relatives with CTS (asymptomatic or symptomatic) in 15–20% of BECTS kindreds; "
            "autosomal dominant with incomplete penetrance; phenotypic variability within same family "
            "from asymptomatic CTS → typical SeLECTS → Panayiotopoulos syndrome."
        ),
        "eeg_correlate": "CTS; may show centroparietal or occipital spike admixture (Panayiotopoulos overlap).",
        "mri_finding": "MRI normal.",
        "clinical_note": (
            "Parental EEG not routinely required but useful for genetic counselling. "
            "Siblings with seizures should have an EEG to detect familial centrotemporal spikes."
        ),
    },
]

# ─── Seizure Types ──────────────────────────────────────────────────────────

SEIZURE_TYPES = [
    {
        "type": "Focal Oro-facial Sensorimotor Seizure",
        "freq_pct": 90,
        "duration_sec": "30–120 sec",
        "description": (
            "Unilateral tongue tingling/numbness, unilateral facial jerking (lower face > upper), "
            "hypersalivation, dysarthria (speech arrest), guttural sounds, throat/laryngeal contraction. "
            "Consciousness fully preserved — child can hear and understand parents but cannot speak. "
            "Characteristic of centrotemporal (perisylvian/opercular) cortex involvement."
        ),
        "eeg_correlate": (
            "Ictal: fast low-amplitude discharge in centrotemporal region ipsilateral to jerking side; "
            "Interictal: high-amplitude diphasic CTS with horizontal dipole maximal at C3/C4 or T7/T8."
        ),
        "clinical_tip": (
            "Parent education key: nocturnal focal seizures may present as child found drooling/making "
            "noises in sleep. Parents should time seizures. Rescue medication NOT needed for brief "
            "focal seizures without consciousness impairment or secondary generalisation."
        ),
    },
    {
        "type": "Nocturnal Secondarily Generalised Tonic-Clonic Seizure (GTCS)",
        "freq_pct": 45,
        "duration_sec": "1–3 min",
        "description": (
            "Secondary generalisation from focal centrotemporal onset; often occurs in first hours of "
            "sleep (NREM stage 2); tongue biting, urinary incontinence, post-ictal confusion/sleep. "
            "Often the presenting seizure (daytime focal seizures may go unnoticed)."
        ),
        "eeg_correlate": (
            "Focal CTS preceding generalised polyspike-and-slow-wave; generalisation reflects "
            "diffuse cortical spread during NREM hypersynchrony."
        ),
        "clinical_tip": (
            "Nocturnal GTCS is alarming but does NOT worsen prognosis in SeLECTS. "
            "Rescue: buccal midazolam 0.3 mg/kg if GTCS >3–5 min. Driving restriction applies."
        ),
    },
    {
        "type": "Hemi-Clonic Seizure",
        "freq_pct": 55,
        "duration_sec": "30–90 sec",
        "description": (
            "Unilateral rhythmic clonic movements of the contralateral arm/hand or face; "
            "may march proximally (Jacksonian march uncommon in BECTS); consciousness preserved. "
            "Reflects motor cortex (precentral gyrus) involvement adjacent to centrotemporal generator."
        ),
        "eeg_correlate": "Rhythmic contralateral centrotemporal discharge; CTS morphology at onset.",
        "clinical_tip": (
            "Distinguish from TLE (no automatisms in BECTS); from PME (no myoclonus/progressive ataxia); "
            "from FLE (no hypermotor/asymmetric tonic posturing). Post-ictal Todd's paresis rare in BECTS."
        ),
    },
    {
        "type": "Opercular Seizure (Atypical)",
        "freq_pct": 15,
        "duration_sec": "20–60 sec",
        "description": (
            "Unilateral perioral clonic movements, jaw jerking, dysphagia, vomiting; "
            "more prolonged speech arrest with drooling and facial grimacing; may mimic absence "
            "due to behavioural arrest; typically associated with atypical BECTS/CSWS spectrum."
        ),
        "eeg_correlate": (
            "CTS maximal at sylvian/opercular electrodes (T7/T8–Cz); may show higher CTS density; "
            "sleep EEG may reveal CSWS if opercular seizures are prominent."
        ),
        "clinical_tip": (
            "Opercular seizures warrant sleep EEG to exclude CSWS. GRIN2A testing indicated. "
            "Frequency and impact on school/language should guide treatment escalation."
        ),
    },
]

# ─── Triggers ──────────────────────────────────────────────────────────────

TRIGGERS = [
    {"trigger": "Sleep deprivation", "pct": 65, "mechanism": "Increased NREM slow-wave activity potentiates CTS activation and lowers seizure threshold.", "management": "Strict sleep hygiene; consistent bedtime; school-day counselling for adequate sleep."},
    {"trigger": "Missed AED dose", "pct": 55, "mechanism": "Sub-therapeutic AED levels in rapid-metaboliser children (weight-based dosing).", "management": "Once-daily OXC at bedtime for nocturnal-predominant seizures; pill organiser; school nurse."},
    {"trigger": "Fever / Intercurrent illness", "pct": 40, "mechanism": "Febrile CTS augmentation; metabolic acceleration of AED clearance.", "management": "Antipyretics early; AED level check if breakthrough seizures during fever."},
    {"trigger": "Stress / Anxiety", "pct": 35, "mechanism": "HPA-axis cortisol modulates GABAergic tone; exam/school stress is common precipitant.", "management": "Psychological support; school accommodations; seizure diary to identify patterns."},
    {"trigger": "Fatigue / Overexertion", "pct": 30, "mechanism": "Fatigue increases adenosine-driven NREM pressure → CTS activation during recovery sleep.", "management": "Balance physical activity; post-exercise rest; avoid heavy sport late evening."},
    {"trigger": "Video games / Screen exposure (evening)", "pct": 20, "mechanism": "Evening blue-light suppresses melatonin → delayed sleep onset → sleep deprivation.", "management": "Screen curfew 1 hour before bedtime; blue-light filter glasses; dim display settings."},
    {"trigger": "Catamenial (females approaching puberty)", "pct": 15, "mechanism": "Progesterone withdrawal at menstruation reduces GABA-A modulation; CTS density may increase peri-menstrually.", "management": "Seizure diary tracking menstrual cycle; consider OCP discussion at puberty for refractory cases."},
    {"trigger": "Alcohol (adolescents only)", "pct": 10, "mechanism": "Alcohol-withdrawal rebound hyperexcitability; disrupted sleep architecture during withdrawal.", "management": "Adolescent counselling; seizure diary; AED compliance during social events."},
]

# ─── Treatment catalog ──────────────────────────────────────────────────────

TREATMENTS = [
    {
        "drug": "No Treatment (Watch-and-Wait)",
        "evidence": "Recommended in low-risk",
        "evidence_ref": "ILAE 2022 SeLECTS guidance; Wirrell EC 1998 Epilepsia",
        "dose_adult": "N/A",
        "dose_paed": "N/A",
        "moa": (
            "Appropriate for children with infrequent nocturnal-only seizures without psychosocial impact. "
            "Self-limited natural history (95% remission by age 16) justifies observation in many cases."
        ),
        "efficacy": "No seizure burden reduction; seizure-free rate ≈ natural remission rate.",
        "safety": "No AED-related adverse effects; avoids overtreatment of self-limited condition.",
        "monitoring": (
            "Annual seizure diary review; school performance monitoring; sleep EEG if any language/attention "
            "regression suspected (exclude CSWS). Re-evaluate if daytime seizures or psychosocial impact emerges."
        ),
        "contraindication_note": None,
    },
    {
        "drug": "Oxcarbazepine (OXC)",
        "evidence": "Level A",
        "evidence_ref": "ILAE 2022; SANAD-II 2021; multiple open-label RCTs",
        "dose_adult": "300–1200 mg/day (divided twice daily)",
        "dose_paed": "10–30 mg/kg/day divided twice daily; max 46 mg/kg/day",
        "moa": (
            "Voltage-gated sodium channel blocker (stabilises inactive state); "
            "MHD (monohydroxy-derivative) is active metabolite; no auto-induction unlike CBZ."
        ),
        "efficacy": "Seizure-free rate ~70–75% in SeLECTS; highly effective for focal CTS-generating seizures.",
        "safety": "Hyponatraemia (monitor Na in first 3 months); dizziness; rash (HLA-B*1502 in Asian patients); possible CSWS aggravation in atypical BECTS — monitor sleep EEG.",
        "monitoring": (
            "Serum sodium at 4–6 weeks and 3 months; MHD level if breakthrough seizures; "
            "HLA-B*1502 screening before prescribing in patients of Han Chinese/Thai/Malaysian descent; "
            "sleep EEG in first year if any cognitive/language change."
        ),
        "contraindication_note": "HLA-B*1502 carriers (severe cutaneous reactions risk)",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "evidence": "Level B",
        "evidence_ref": "ILAE 2022; Bello-Espinosa LE 2016 Pediatr Neurol",
        "dose_adult": "500–3000 mg/day (divided twice daily)",
        "dose_paed": "20–60 mg/kg/day divided twice daily",
        "moa": (
            "SV2A (synaptic vesicle glycoprotein 2A) modulation; reduces neurotransmitter release; "
            "also modulates calcium channels. Broad-spectrum mechanism; no enzyme induction/inhibition."
        ),
        "efficacy": "Seizure-free rate ~65–70%; well studied in paediatric SeLECTS; preferred when OXC contraindicated.",
        "safety": "Behavioural side-effects (irritability, aggression, emotional lability) in 10–15% of children — PHQ-9/CBCL monitoring essential.",
        "monitoring": (
            "PHQ-9 (adolescents) + CBCL behavioural rating scale at 3 and 6 months; "
            "renal function monitoring (LEV renally cleared — dose reduce in CKD); "
            "serum LEV level if breakthrough seizures."
        ),
        "contraindication_note": None,
    },
    {
        "drug": "Sulthiame (STM)",
        "evidence": "Level A (Europe)",
        "evidence_ref": "Caraballo RH et al. 2010 Epilepsia (SCORE RCT); EAN BECTS Guideline 2019",
        "dose_adult": "200–400 mg/day (divided twice daily)",
        "dose_paed": "5–10 mg/kg/day divided twice daily",
        "moa": (
            "Carbonic anhydrase inhibitor (CA-II and CA-IV); increases brain CO₂ → neuronal hyperpolarisation; "
            "also inhibits CTS generation through direct cortical effects. "
            "NOT widely available in North America or India — primarily European agent."
        ),
        "efficacy": "EEG CTS suppression in 84% vs 10% placebo (SCORE RCT); clinical seizure reduction Level A.",
        "safety": "Paraesthesiae (peripheral neuropathy risk); anorexia/weight loss; renal calculi (carbonic anhydrase); avoid in sulphonamide allergy.",
        "monitoring": (
            "Serum bicarbonate (metabolic acidosis risk); renal ultrasound annually (calculi); "
            "weight and appetite tracking; sulphonamide allergy screen before prescribing."
        ),
        "contraindication_note": "Sulphonamide allergy; not available in all countries",
    },
    {
        "drug": "Valproate (VPA)",
        "evidence": "Level B",
        "evidence_ref": "ILAE 2022; NICE NG217 (second-line, females caution)",
        "dose_adult": "600–2000 mg/day (divided twice daily)",
        "dose_paed": "20–40 mg/kg/day divided twice daily",
        "moa": (
            "Multiple mechanisms: GABA-transaminase inhibition (↑ brain GABA), sodium channel blockade, "
            "T-type calcium channel inhibition. Broad-spectrum AED suitable for secondarily generalised seizures."
        ),
        "efficacy": "Effective for GTCS component in SeLECTS; also effective if CSWS develops; ~65% seizure-free.",
        "safety": "FDA REMS (teratogen — spina bifida, neurodevelopmental — females of childbearing potential); weight gain; hair thinning; polycystic ovary syndrome; liver function monitoring.",
        "monitoring": (
            "FDA VPA REMS: mandatory informed consent females >4 years; TDM 50–100 µg/mL; "
            "LFT + ammonia at 3 months; weight/BMI; folic acid 5 mg/day (females of reproductive potential). "
            "Avoid as first-line in girls approaching puberty."
        ),
        "contraindication_note": "Females of childbearing potential (teratogen — FDA REMS mandatory)",
    },
    {
        "drug": "Carbamazepine (CBZ)",
        "evidence": "Level B (historical first-line)",
        "evidence_ref": "ILAE 2022 (now second-line); Verrotti A 2014 review",
        "dose_adult": "400–1200 mg/day (divided twice daily)",
        "dose_paed": "10–20 mg/kg/day divided twice daily",
        "moa": (
            "Voltage-gated sodium channel blocker; auto-inducing (induces CYP3A4 → may lower own levels over "
            "3–4 weeks; dose adjustment required post-induction). Less hyponatraemia than OXC."
        ),
        "efficacy": "~65% seizure-free; historically most used, now largely replaced by OXC in paediatrics.",
        "safety": "SJS risk (HLA-B*1502 Asians mandatory); dizziness; diplopia; rare CSWS aggravation in atypical BECTS — monitor sleep EEG.",
        "monitoring": (
            "HLA-B*1502 before prescribing in Asian-descent patients; TDM 4–12 µg/mL; "
            "CBC (agranulocytosis risk — quarterly in first year); LFT; sleep EEG if cognitive/language change."
        ),
        "contraindication_note": "HLA-B*1502 carriers; atypical BECTS/CSWS (may worsen)",
    },
    {
        "drug": "Lamotrigine (LTG)",
        "evidence": "Level C",
        "evidence_ref": "ILAE 2022 (limited paediatric focal epilepsy data)",
        "dose_adult": "50–400 mg/day (divided twice daily)",
        "dose_paed": "0.3–3 mg/kg/day divided twice daily (slow titration 8–12 weeks)",
        "moa": (
            "Voltage-gated sodium channel blocker; also inhibits presynaptic glutamate release. "
            "Half-life prolonged when co-administered with VPA (halve LTG dose)."
        ),
        "efficacy": "~55–60% seizure-free; may be useful in adolescents approaching remission when minimal side-effects needed.",
        "safety": "SJS risk (especially rapid titration or VPA co-administration); rash (10% — not all SJS); headache; insomnia.",
        "monitoring": (
            "Slow titration protocol mandatory (8–12 week escalation); SJS early warning education (fever + rash = stop); "
            "TDM 3–15 µg/mL; VPA interaction — halve LTG target dose if VPA co-prescribed; "
            "skin examination at each visit."
        ),
        "contraindication_note": "Rapid titration (SJS risk); concurrent VPA (must halve dose)",
    },
    {
        "drug": "Clobazam (CLB) adjunct",
        "evidence": "Level B (adjunct)",
        "evidence_ref": "ILAE 2022 adjunct; NICE NG217",
        "dose_adult": "10–30 mg/day at night",
        "dose_paed": "0.1–0.8 mg/kg/day at night (max 30 mg/day)",
        "moa": (
            "1,5-benzodiazepine; GABA-A receptor positive allosteric modulator at α2/α3 subunits; "
            "less sedating than clonazepam; nocturnal dosing particularly effective for NREM-activated CTS."
        ),
        "efficacy": "~60% seizure reduction as add-on; particularly useful for nocturnal seizures; behavioural improvement in CSWS.",
        "safety": "Sedation; tolerance/dependence with long-term use; respiratory depression at high doses; mood effects.",
        "monitoring": (
            "UMRS (Unified Myoclonus Rating Scale — if used for CSWS); "
            "tolerance monitoring (dose escalation trend suggests tolerance development → reassess); "
            "respiratory function if used with other CNS depressants."
        ),
        "contraindication_note": None,
    },
]

# ─── AED Monitoring ─────────────────────────────────────────────────────────

AED_MONITORING = [
    {
        "item": "OXC — Hyponatraemia Screen + HLA-B*1502",
        "frequency": "Baseline, 4–6 weeks, 3 months, then annually",
        "rationale": (
            "Serum sodium: OXC causes SIADH in ~2–5% of children; especially at high doses. "
            "HLA-B*1502: mandatory in Han Chinese, Thai, Malay, Filipino descent — SJS risk 10-fold higher. "
            "MHD level if breakthrough seizures."
        ),
    },
    {
        "item": "LEV — Behavioural Monitoring (CBCL + PHQ-9)",
        "frequency": "3 months, 6 months, then annually",
        "rationale": (
            "Child Behaviour Checklist (CBCL) and Conners rating scale to quantify aggression/irritability; "
            "PHQ-9 for adolescents (depression risk); L-carnitine supplementation if extreme behavioural "
            "effects (LEV mechanism may deplete carnitine). Dose reduction or switch if sustained behaviour change."
        ),
    },
    {
        "item": "VPA — REMS + TDM + LFT + Weight (FEMALES)",
        "frequency": "Baseline REMS consent, then 3 months, then 6-monthly",
        "rationale": (
            "FDA VPA REMS: mandatory informed consent form females >4 years; TDM 50–100 µg/mL; "
            "LFT + ammonia at 3 months (hepatotoxicity); BMI/weight (obesity risk); "
            "folic acid 5 mg/day for all females of reproductive potential. "
            "Assess need for contraception counselling at puberty."
        ),
    },
    {
        "item": "CSWS Surveillance — Sleep EEG Annually",
        "frequency": "Baseline + annual overnight EEG (NREM-focused) if any cognitive/language concern",
        "rationale": (
            "CSWS/ESES (slow-wave index >85% in NREM) can develop silently in SeLECTS — especially GRIN2A+. "
            "Annual sleep EEG recommended if: GRIN2A positive, language concerns reported by teacher/parent, "
            "attention regression, school performance decline. Early detection enables timely treatment "
            "escalation before irreversible language regression."
        ),
    },
]

# ─── Lifecycle management windows ───────────────────────────────────────────

LIFECYCLE = [
    {
        "window": "Pre-diagnosis (Age 3–7 y)",
        "age_range": "3–7 years",
        "key_events": "First focal seizure (often nocturnal); parents alarmed; paediatric ED presentation.",
        "focus": (
            "Reassurance that SeLECTS is the most common benign childhood epilepsy. "
            "EEG to confirm CTS. Brain MRI to exclude structural cause (must be normal). "
            "Decision: treat vs. watch-and-wait based on seizure frequency + psychosocial impact."
        ),
    },
    {
        "window": "Active Phase (Age 7–12 y)",
        "age_range": "7–12 years",
        "key_events": "Peak seizure frequency; school performance concerns; AED optimisation.",
        "focus": (
            "Seizure diary; school accommodation letter; neuropsychological assessment if learning concerns. "
            "Annual EEG (NREM if cognitive concern); AED adherence; growth monitoring on VPA. "
            "Counsel parents: majority seizure-free by age 12–14."
        ),
    },
    {
        "window": "Approaching Puberty (Age 11–14 y)",
        "age_range": "11–14 years",
        "key_events": "Spontaneous remission expected; AED taper planning; female reproductive counselling.",
        "focus": (
            "If seizure-free ≥1–2 years: AED taper discussion (gradual over 6–12 months). "
            "Females: VPA REMS re-consent + contraception discussion + folic acid; switch from VPA "
            "if possible. Driving restriction counselling (12 months after last seizure). "
            "GRIN2A panel if not done and atypical features present."
        ),
    },
    {
        "window": "Adolescent Remission (Age 14–18 y)",
        "age_range": "14–18 years",
        "key_events": "Seizure freedom; AED discontinuation; driving eligibility; exam stress management.",
        "focus": (
            "AED discontinuation if ≥2 years seizure-free and EEG shows CTS resolution. "
            "Driver's licence discussion (jurisdiction rules). Alcohol counselling. "
            "If ongoing seizures at 16 (rare): reclassify — SeLECTS diagnosis in question; consider CSWS/GRIN2A; "
            "re-image; consider alternative diagnoses (JME/TLE/focal cortical dysplasia)."
        ),
    },
    {
        "window": "Adult Life (Age 18+ y)",
        "age_range": "18+ years",
        "key_events": "AED-free; driving; occupational health; counselling children of affected parent.",
        "focus": (
            "Typically AED-free with no restrictions. Reassure about recurrence risk (<5% adult relapse). "
            "Genetic counselling: 15–20% sibling risk of CTS on EEG (most asymptomatic). "
            "Females of reproductive age: folic acid 400 µg/day standard supplement. "
            "If GRIN2A+: family screening EEG for children."
        ),
    },
    {
        "window": "CSWS/LKS Trajectory (Any age 5–12 y)",
        "age_range": "5–12 years (atypical evolution)",
        "key_events": "Language regression detected; sleep EEG confirms CSWS; specialist referral.",
        "focus": (
            "CSWS requires treatment intensification: high-dose VPA + CLB, or STM escalation, or IVIG. "
            "Speech-language therapy concurrent with AED management. "
            "Neuropsychological monitoring every 6 months. School IEP for language support. "
            "Prognosis for language: good if CSWS treated promptly; poor if delayed >2 years."
        ),
    },
]

# ─── Standards & Thresholds ─────────────────────────────────────────────────

STANDARDS = [
    {"standard": "ILAE 2022 Classification", "relevance": "SeLECTS reclassification from BECTS; spectrum classification (typical/atypical); GRIN2A guidance."},
    {"standard": "NICE NG217 (2022)", "relevance": "UK national guideline for epilepsy management in children; watch-and-wait guidance; AED selection."},
    {"standard": "EAN BECTS Guideline 2019", "relevance": "European Academy of Neurology — sulthiame Level A evidence; first European BECTS-specific guideline."},
    {"standard": "ILAE Diagnostic Manual 2017", "relevance": "Scheffer IE et al. — SeLECTS/BECTS in self-limited focal epilepsy category."},
    {"standard": "FDA VPA REMS (2013)", "relevance": "Mandatory VPA informed consent (Risk Evaluation and Mitigation Strategy) for females ≥4 years."},
    {"standard": "HLA-B*1502 FDA Guidance (2007)", "relevance": "Mandatory HLA-B*1502 genetic screening before OXC/CBZ prescription in Asian-descent patients."},
]

THRESHOLDS = [
    {"threshold": "AED initiation — daytime seizures or psychosocial impact", "value": "≥1 daytime seizure or significant psychosocial burden → treat"},
    {"threshold": "AED taper — seizure-free duration", "value": "≥1–2 years seizure-free (most guidelines: taper over 6–12 months)"},
    {"threshold": "CSWS diagnosis — slow-wave index", "value": "Slow-wave index >85% in NREM on overnight EEG"},
    {"threshold": "OXC dose (paediatric)", "value": "10–30 mg/kg/day (target: seizure-free at minimum effective dose)"},
    {"threshold": "Sulthiame dose (paediatric)", "value": "5–10 mg/kg/day (European standard — not widely available)"},
    {"threshold": "VPA therapeutic range", "value": "50–100 µg/mL (total valproate)"},
    {"threshold": "Driving restriction", "value": "12 months seizure-free (jurisdiction-dependent)"},
    {"threshold": "GRIN2A testing trigger — atypical features", "value": "≥1 of: CSWS, language regression, treatment resistance, LKS features"},
]

REFERENCES = [
    "Loiseau P & Duche B 1989 Epilepsia — 168-patient series; natural history of BECTS; 95% remission by adulthood.",
    "Beaussart M 1972 Epilepsia — Classic description of Rolandic epilepsy seizure semiology.",
    "Scheffer IE et al. 2017 Epilepsia — ILAE classification; SeLECTS reclassification from BECTS.",
    "Lemke JR et al. 2013 Nat Genet — GRIN2A dominant mutations in epilepsy-aphasia spectrum (BECTS → LKS).",
    "Wirrell EC 1998 Epilepsia — Outcome cohort: 95% remission by puberty; long-term prognosis.",
    "Caraballo RH et al. 2010 Epilepsia — SCORE RCT: sulthiame vs placebo; Level A evidence in BECTS.",
]

CONCEPTS = [
    {"term": "SeLECTS / BECTS / Rolandic Epilepsy", "definition": "Self-Limited Epilepsy with Centrotemporal Spikes (ILAE 2022); most common benign childhood epilepsy (15–25% of childhood epilepsy); onset 3–13y; spontaneous remission by age 16 in 95%."},
    {"term": "Centrotemporal Spikes (CTS)", "definition": "High-amplitude, diphasic spikes with horizontal dipole at C3/C4 and T7/T8; pathognomonic EEG hallmark; activated by NREM sleep; can occur without clinical seizures; frequency does not predict seizure severity."},
    {"term": "Horizontal Dipole", "definition": "EEG pattern where CTS shows negative maximum at centrotemporal electrodes and positive phase reversal at frontal electrodes; distinguishes CTS from other focal discharges; best seen on bipolar montage."},
    {"term": "Oro-facial Sensorimotor Seizure", "definition": "Focal seizure from centrotemporal cortex: unilateral tongue tingling, facial jerking, hypersalivation, dysarthria, guttural sounds; consciousness preserved; speech arrest (child can hear but cannot speak)."},
    {"term": "CSWS / ESES", "definition": "Continuous Spike-Wave during Slow Sleep (CSWS) or Electrical Status Epilepticus during Sleep (ESES); slow-wave index >85% in NREM; disrupts memory consolidation and language development; requires treatment intensification."},
    {"term": "GRIN2A", "definition": "GluN2A subunit of ionotropic NMDA glutamate receptor; dominant variants cause epilepsy-aphasia spectrum (typical BECTS → CSWS → Landau-Kleffner Syndrome); present in 9–15% of typical BECTS and 20–27% of atypical/CSWS cases."},
    {"term": "Landau-Kleffner Syndrome (LKS)", "definition": "Acquired epileptic aphasia: verbal agnosia progressing to complete language loss; perisylvian CTS-dominant EEG during sleep; shares genetic-electroclinical spectrum with BECTS but severe end; may respond to IVIG/steroids."},
    {"term": "Panayiotopoulos Syndrome", "definition": "Shares genetic-electroclinical spectrum with BECTS; different clinical: occipital spikes; autonomic seizures (ictal vomiting, eye deviation, syncope-like); typically longer seizures; also self-limited. Part of same centrotemporal-occipital continuum."},
    {"term": "Sulthiame (STM)", "definition": "Carbonic anhydrase inhibitor; Level A evidence for CTS suppression in BECTS (SCORE RCT); primarily European agent; not licensed in USA/Canada; 5–10 mg/kg/day paediatric dose."},
    {"term": "Slow-Wave Index", "definition": "Proportion of NREM sleep occupied by slow spike-wave complexes on EEG; threshold of >85% defines CSWS/ESES; calculated on overnight or 24-hour ambulatory EEG."},
    {"term": "Epilepsy-Aphasia Spectrum", "definition": "Spectrum of conditions sharing centrotemporal spike genetics and EEG features: typical SeLECTS (mildest) → atypical BECTS/CSWS → Landau-Kleffner Syndrome (most severe); GRIN2A variants present across the spectrum."},
    {"term": "HLA-B*1502", "definition": "HLA allele common in Han Chinese, Thai, Malay populations; strongly associated with SJS/TEN risk with OXC and CBZ; FDA mandates genetic screening before prescribing these agents in at-risk Asian populations."},
    {"term": "Todd's Paresis", "definition": "Transient post-ictal focal weakness; uncommon in BECTS (unlike TLE/FLE); if present and prolonged (>24 h) consider alternative diagnosis (focal cortical dysplasia, tumour, vascular)."},
    {"term": "Seizure-Freedom by Puberty", "definition": "The defining characteristic of SeLECTS: spontaneous seizure remission in 95% of patients by age 16; CTS on EEG may persist 2–3 years after clinical remission; EEG normalisation lags clinical improvement."},
]

ABSOLUTE_CONTRAINDICATIONS = [
    {"drug": "Vigabatrin (VGB)", "contraindicated_in": "All SeLECTS", "consequence": "Worsens focal seizures; can provoke CSWS; irreversible concentric visual field loss; no benefit in focal epilepsy without infantile spasms indication."},
    {"drug": "Carbamazepine (CBZ) — RELATIVE in atypical phenotype", "contraindicated_in": "GRIN2A+ / CSWS risk / atypical BECTS", "consequence": "May aggravate CSWS/cognitive regression in susceptible children; HLA-B*1502 carries SJS risk; prefer OXC or LEV in atypical presentations."},
    {"drug": "Prolonged AED use beyond remission", "contraindicated_in": "Seizure-free SeLECTS ≥2 years", "consequence": "Unnecessary AED exposure; adverse effects (cognitive/behavioural) without seizure benefit; self-limited syndrome — taper mandatory."},
    {"drug": "Phenytoin (PHT) / Phenobarbital (PB)", "contraindicated_in": "All paediatric SeLECTS (relative)", "consequence": "Cognitive adverse effects unacceptable in school-age children; no advantage over OXC/LEV; gingival hyperplasia (PHT); sedation/cognitive impairment (PB)."},
]


# ─── OVERVIEW ───────────────────────────────────────────────────────────────

def overview():
    patients = _db_rows("SELECT * FROM patients LIMIT 41")
    if not patients:
        patients = [{"patient_id": f"P{i:03d}"} for i in range(1, 42)]

    n = len(patients)

    # Deterministic overlays
    etiology_map = {}
    for et_entry in ETIOLOGY_CATALOG:
        share = et_entry["pct"]
        etiology_map[et_entry["etiology"]] = share

    # Simulate patient attributes deterministically
    oral_facial_n = round(n * 0.90)
    nocturnal_gtcs_n = round(n * 0.45)
    csws_n = round(n * 0.08)
    grin2a_positive_n = round(n * 0.12)
    treatment_free_n = round(n * 0.25)
    drug_resistant_n = round(n * 0.05)
    learning_difficulty_n = round(n * 0.45)
    male_n = round(n * 0.60)

    etiology_dist = {et["etiology"]: round(n * et["pct"] / 100) for et in ETIOLOGY_CATALOG}

    clinical_alerts = [
        "⛔ Vigabatrin (VGB) is CONTRAINDICATED in SeLECTS — worsens focal seizures and can provoke CSWS.",
        "⚠️ Annual sleep EEG mandatory in any child with SeLECTS + language regression / cognitive decline — CSWS can develop silently.",
        "⚠️ HLA-B*1502 genetic screen mandatory before OXC or CBZ in Han Chinese / Thai / Malay / Filipino patients — SJS risk.",
        "⚠️ FDA VPA REMS: mandatory informed consent for all females ≥4 years before valproate. Avoid VPA as first-line in girls approaching puberty.",
        "✅ Watch-and-wait is appropriate for infrequent nocturnal-only seizures — 95% of SeLECTS remits spontaneously by puberty.",
        "⚠️ Prolonged AED treatment beyond 2 years seizure-free is inappropriate — self-limited syndrome; taper and discontinue.",
    ]

    return {
        "generated": str(date.today()),
        "total_patients": n,
        "male_n": male_n,
        "male_pct": round(male_n / n * 100),
        "oro_facial_seizure_n": oral_facial_n,
        "oro_facial_seizure_pct": round(oral_facial_n / n * 100),
        "nocturnal_gtcs_n": nocturnal_gtcs_n,
        "nocturnal_gtcs_pct": round(nocturnal_gtcs_n / n * 100),
        "csws_evolution_n": csws_n,
        "csws_evolution_pct": round(csws_n / n * 100),
        "grin2a_positive_n": grin2a_positive_n,
        "grin2a_positive_pct": round(grin2a_positive_n / n * 100),
        "treatment_free_n": treatment_free_n,
        "treatment_free_pct": round(treatment_free_n / n * 100),
        "drug_resistant_n": drug_resistant_n,
        "drug_resistant_pct": round(drug_resistant_n / n * 100),
        "learning_difficulty_n": learning_difficulty_n,
        "learning_difficulty_pct": round(learning_difficulty_n / n * 100),
        "etiology_distribution": etiology_dist,
        "clinical_alerts": clinical_alerts,
        "prognosis_summary": {
            "seizure_free_by_puberty": "95% (Wirrell 1998)",
            "csws_evolution_risk": "8% (GRIN2A positive: up to 27%)",
            "lks_risk": "Rare (<2% of all SeLECTS)",
            "adult_relapse_risk": "<5%",
            "cognitive_impact": "Transient in ~50% during active phase; permanent impairment rare in typical SeLECTS",
            "treatment_free_suitability": "~25% (infrequent nocturnal seizures, no psychosocial impact)",
        },
        "references": REFERENCES,
    }


# ─── BREAKDOWN ──────────────────────────────────────────────────────────────

def breakdown():
    patients_raw = _db_rows("SELECT * FROM patients LIMIT 41")
    if not patients_raw:
        patients_raw = [{"patient_id": f"P{i:03d}", "age": 8 + (i % 10), "gender": "M" if i % 5 != 0 else "F"} for i in range(1, 42)]

    n = len(patients_raw)
    ETIOLOGIES = [et["etiology"] for et in ETIOLOGY_CATALOG]
    ETIOLOGY_WEIGHTS = [et["pct"] for et in ETIOLOGY_CATALOG]
    SEIZURE_TYPE_LIST = [st["type"] for st in SEIZURE_TYPES]
    AED_OPTIONS = ["No Treatment", "OXC", "LEV", "VPA", "STM (sulthiame)", "CLB adjunct", "OXC + CLB", "LEV + CLB"]
    AED_WEIGHTS = [25, 35, 20, 10, 5, 3, 1, 1]
    CTRL_OPTIONS = ["Seizure-free", "Partial control", "Drug-resistant"]
    CTRL_WEIGHTS = [60, 35, 5]
    PHASE_OPTIONS = ["Active", "Near-remission", "Remission"]
    PHASE_WEIGHTS = [50, 30, 20]

    def _pick(seed_val, options, weights):
        total = sum(weights)
        v = seed_val % total
        cum = 0
        for opt, w in zip(options, weights):
            cum += w
            if v < cum:
                return opt
        return options[-1]

    patients_out = []
    for p in patients_raw:
        pid = p.get("patient_id", "?")
        s = _seed(pid)
        onset_age = 4 + (s % 10)
        age = max(onset_age + 1, int(p.get("age") or 10))
        gender = p.get("gender", "M" if s % 5 != 0 else "F")
        etiology = _pick(s >> 4, ETIOLOGIES, ETIOLOGY_WEIGHTS)
        seizure_type = _pick(s >> 8, SEIZURE_TYPE_LIST, [90, 45, 55, 15])
        aed = _pick(s >> 12, AED_OPTIONS, AED_WEIGHTS)
        control = _pick(s >> 16, CTRL_OPTIONS, CTRL_WEIGHTS)
        phase = _pick(s >> 20, PHASE_OPTIONS, PHASE_WEIGHTS)
        csws = etiology in ("CSWS / Atypical Evolution", "Landau-Kleffner Syndrome (LKS)")
        grin2a = etiology in ("GRIN2A Mutation", "CSWS / Atypical Evolution", "Landau-Kleffner Syndrome (LKS)")
        learning = (s >> 24) % 100 < 45
        patients_out.append({
            "patient_id": pid,
            "age": age,
            "gender": gender,
            "onset_age": onset_age,
            "etiology": etiology,
            "primary_seizure_type": seizure_type,
            "current_aed": aed,
            "seizure_control": control,
            "disease_phase": phase,
            "csws_evolution": csws,
            "grin2a_positive": grin2a,
            "learning_difficulty": learning,
        })

    return {
        "patients": patients_out,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "aed_monitoring": AED_MONITORING,
        "lifecycle": LIFECYCLE,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
    }


# ─── DEFINITIONS ────────────────────────────────────────────────────────────

def definitions():
    return {
        "concepts": CONCEPTS,
        "absolute_contraindications": ABSOLUTE_CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
    }
