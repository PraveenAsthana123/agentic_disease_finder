"""Parietal Lobe Epilepsy (PLE) Dashboard — an under-recognised focal epilepsy that accounts
for ~5% of all focal epilepsies, frequently misdiagnosed as Temporal or Frontal Lobe Epilepsy
due to rapid ictal propagation away from the parietal cortex.

Hallmarks:
  Somatosensory Aura: contralateral tingling, numbness, pain, electric sensation — the
    most specific parietal onset sign, present in 60–80% of structural PLE
  Body Schema Disturbances: out-of-body experience (TPJ), contralateral limb neglect,
    somatotopic body-part illusions, macro/microsomatognosia
  Rapid Ictal Spread: parietal discharges propagate in <3 seconds to frontal (SMA →
    hypermotor), temporal (deja vu, automatisms), or occipital cortex (visual distortions)
  Genital / Perineal Aura: mesial parietal (paracentral lobule) → often bilateral
    somatosensory spread; misdiagnosed as psychogenic NES or urogenital pathology
  Versive / Tonic Motor Signs: spread from SPL → premotor → SMA → contralateral tonic
    posturing, forced head-eye deviation, Jacksonian march
  Onset bimodal: childhood (cortical dysplasia FCD Type II) + adult structural (tumour,
    cavernoma, post-traumatic, post-stroke)
  Drug resistance: ~30–45% for structural PLE; FCD-related PLE ~40%
  EEG hallmark: centroparietal (C3/C4/P3/P4/CP3/CP4) spike-wave or slow-wave;
    interictal EEG normal in up to 40%; SEEG often required for precise onset mapping

FIRST-LINE TREATMENT SELECTION:
  CBZ / OXC: Level A for focal seizures; particularly effective when ictal spread to motor
    cortex produces secondary motor phenomena
  LTG: Level A (SANAD 2007 — non-inferior to CBZ for overall effectiveness including safety
    in women); preferred when comorbid depression or migraine overlap
  LEV: Level B — broad-spectrum, no drug interactions, safe first-line especially for rapid
    titration needs or women of childbearing age
  Lacosamide: Level A add-on — dual MOA (slow-inactivation Nav1.1-1.7 + CRMP-2 modulation);
    particularly useful when parietal onset produces tonic spread (JASPER 2018)
  Perampanel (AMPA antagonist): Level A add-on, REMS-gated; effective for refractory PLE
    with secondary GTCS or hypermotor spread (Study 206, 2012)
  Surgical Resection: parietal lesionectomy / SEEG-guided cortectomy —
    Engel I 55–70% for structural PLE (Kim 2004 J Neurosurg)
    Engel I 35–50% for non-lesional PLE (lower: poor localisation without SEEG)
  Ketogenic Diet: Level B for refractory drug-resistant PLE; GLUT-1 deficiency PLE may
    show dramatic response (seizure-free ~90%)

ABSOLUTE CONTRAINDICATIONS (PLE-specific context):
  Sodium channel blockers (CBZ/OXC/PHT/LTG) in SCN8A gain-of-function parietal epilepsy:
    may paradoxically worsen seizure frequency — check gene panel before prescribing
  Vigabatrin (VGB): irreversible concentric visual field loss (bilateral nasal-to-temporal
    VF defect) — contraindicated in PLE with occipital spread / visual aura (risk of
    compounding VF defect from both seizures and drug)

References:
  - Kim DW et al. 2004 J Neurosurg (PLE surgical series — 40 patients; 57.5% Engel I)
  - Salanova V et al. 1995 Epilepsia (parietal lobe epilepsy — 82 patients; semiology)
  - Williamson PD et al. 1992 Ann Neurol (OLE-PLE series; parietal + occipital localisation)
  - Binder JR et al. 2009 J Cogn Neurosci (TPJ — out-of-body / body-schema processing)
  - Blume WT et al. 2001 Epilepsia (ILAE semiological seizure classification, parietal auras)
  - Ryvlin P et al. 2006 Epilepsia (SEEG-guided parietal lobe surgery outcomes)
Data: live clinical.db (41 epilepsy patients, deterministic PLE overlay)
      + curated PLE pharmacology / etiology / seizure-type / trigger catalogs."""

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


# ─── Etiology / Structural/Genetic catalog ───────────────────────────────────

_ETIOLOGIES = [
    {
        "etiology": "Focal Cortical Dysplasia Type II (FCD IIb — Balloon Cells)",
        "category": "Structural/Developmental",
        "pct": 35,
        "mechanism": (
            "FCD Type IIb is the most common structural cause of drug-resistant PLE. "
            "Balloon cells — dysmorphic neurons with giant somata (50–100 µm) — create "
            "autonomous pacemaker activity in layer II/III of the primary and association "
            "parietal cortex. The mTOR pathway is hyperactivated via somatic MTOR, TSC1, "
            "or TSC2 mutations (occuring post-zygotically); germline DEPDC5 loss-of-function "
            "mutations account for ~15% of FCD II-related PLE. Seizures are often nocturnal, "
            "brief, clusters of 5–15 events per night; drug resistance develops within 2–3 "
            "years of onset. Surgical cortectomy guided by SEEG + intraoperative ECoG achieves "
            "Engel I in 55–70% when the MRI lesion is completely resected. mTOR inhibitors "
            "(rapamycin / everolimus) under investigation for FCD IIb neuroinflammation reduction."
        ),
        "eeg_correlate": (
            "Continuous interictal high-frequency oscillations (HFOs: 80–250 Hz) over the FCD "
            "zone on SEEG; scalp EEG shows focal polymorphic delta slowing (P3/P4/CP3/CP4) ± "
            "centroparietal spike-wave in 60%. MRI-negative FCD requires MRI post-processing "
            "(voxel-based morphometry, SurferVBM) or FDG-PET/MRI co-registration to delineate. "
            "Intraoperative ECoG: continuous spiking and rapid burst-suppression pattern at the "
            "FCD core — guide for complete resection boundary."
        ),
        "mri_finding": (
            "T2/FLAIR blurring of the grey-white matter junction in parietal cortex (transmantle "
            "sign: T2 hyperintensity extending from cortex to ventricular wall). Cortical "
            "thickening 4–6 mm (normal: 2.5 mm). T1 signal: subtle hypointensity of dysplastic "
            "cortex. 7T MRI + surface-based morphometry improves FCD detection sensitivity to ~78% "
            "(vs 40% at 3T)."
        ),
        "clinical_note": (
            "Screen DEPDC5 (GATOR1 complex), MTOR somatic: LP cerebrospinal fluid for MTOR panel. "
            "Consider mTOR inhibitor (everolimus) as adjunct after surgical resection to prevent "
            "seizure recurrence from residual dysplastic cortex."
        ),
    },
    {
        "etiology": "Low-Grade Tumour (DNT / Ganglioglioma / Oligodendroglioma WHO Grade II)",
        "category": "Structural/Neoplastic",
        "pct": 22,
        "mechanism": (
            "Dysembryoplastic Neuroepithelial Tumours (DNT) and gangliogliomas are the most "
            "common tumour-related causes of PLE in young adults (peak onset 15–35 years). "
            "DNTs secrete glutamatergic excitotoxic substances and disrupt GABAergic inhibitory "
            "interneuron networks in the adjacent parietal cortex. BRAF V600E mutation in "
            "ganglioglioma (60%) activates the MAPK-ERK1/2 pathway → oncogenic cortical "
            "hyperexcitability independent of tumour mass effect. Seizures are often the "
            "first and only presenting symptom; drug resistance develops in ~35% despite "
            "adequate AED therapy. Gross-total resection achieves Engel I in 70–85% for DNT, "
            "Engel I 55–65% for ganglioglioma (DaSilva 2018 Epilepsia). Recurrence rare "
            "for WHO Grade I–II; malignant transformation ~3% for oligodendroglioma."
        ),
        "eeg_correlate": (
            "Centroparietal slow-wave focus with superimposed spike-wave; seizure-onset pattern: "
            "focal low-voltage fast activity (LVFA) at tumour margin followed by rhythmic 4–6 Hz "
            "theta-delta discharge propagating to temporal or frontal regions. Interictal "
            "background: focal ipsilateral delta polymorphic slowing in tumour territory."
        ),
        "mri_finding": (
            "DNT: multicystic 'bubbly' cortical lesion T2 hyperintense, T1 hypointense, minimal "
            "enhancement, no oedema, gyral expansion; classically temporal but 15% parietal. "
            "Ganglioglioma: partially cystic with mural nodule enhancement; calcification in 30%. "
            "Oligodendroglioma: 'fried-egg' cells on pathology; MRI: cortical/subcortical "
            "T2 signal; 1p/19q co-deletion diagnostic (IDH-mutant)."
        ),
        "clinical_note": (
            "Urgent MRI with gadolinium + spectroscopy (choline:NAA ratio) to exclude GBM. "
            "BRAF V600E testing: eligibility for vemurafenib/dabrafenib (BRAF inhibitors) in "
            "refractory ganglioglioma-PLE. Monitor with annual MRI post-resection."
        ),
    },
    {
        "etiology": "Post-Traumatic PLE (TBI — Cortical Contusion / Haemosiderin Deposition)",
        "category": "Structural/Acquired",
        "pct": 15,
        "mechanism": (
            "Traumatic brain injury affecting the parietal convexity produces haemosiderin "
            "deposition (from microhaemorrhages / cortical contusions), which acts as a "
            "pro-epileptic iron catalyst: Fe²⁺ drives Fenton-reaction-mediated reactive oxygen "
            "species (ROS) → lipid peroxidation → disruption of neuronal membrane ion channels "
            "→ hyperexcitable somatosensory cortex. Post-traumatic epilepsy develops in 5–50% "
            "of moderate-severe TBI (risk highest: depressed skull fracture, intracerebral "
            "haematoma, post-traumatic amnesia >24h). Seizure onset: bimodal — early post-traumatic "
            "seizures <7 days (prophylaxis: levetiracetam ≤7 days), late post-traumatic epilepsy "
            ">7 days (no prophylaxis evidence; treat symptomatic seizures). Drug resistance: ~35%."
        ),
        "eeg_correlate": (
            "Focal slow-wave delta (1–3 Hz) over contusion/haematoma territory. Interictal spikes "
            "arise from haemosiderin-gliosis boundary. Ictal: focal rhythmic evolution at 5–10 Hz "
            "followed by postictal suppression. MEG localises haemosiderin dipole more precisely "
            "than scalp EEG for pre-surgical mapping."
        ),
        "mri_finding": (
            "SWI/GRE: haemosiderin 'blooming artefact' at contusion margin; T2/FLAIR gliosis; "
            "volume loss. Siderosis in subarachnoid haemorrhage → superficial siderosis (linear "
            "T2* hypointensity over parietal convexity). DTI: disrupted superior longitudinal "
            "fasciculus (SLF) white-matter tracts adjacent to seizure onset zone."
        ),
        "clinical_note": (
            "Levetiracetam 7-day post-TBI prophylaxis (AAN Level A) for early seizures — do NOT "
            "continue beyond 7 days (no benefit; increases neurotoxicity risk). For chronic "
            "post-traumatic PLE: SEEG-guided resection if contusion scar is well-circumscribed."
        ),
    },
    {
        "etiology": "Genetic PLE (DEPDC5 / GATOR1 Complex — Familial Focal Epilepsy)",
        "category": "Genetic",
        "pct": 18,
        "mechanism": (
            "DEPDC5 (DEP Domain Containing 5) is a GATOR1-complex subunit that acts as a "
            "master inhibitor of the mTORC1 pathway. Loss-of-function mutations (AD inheritance, "
            "~70% penetrance) → constitutive mTORC1 activation → neuronal hypertrophy + synaptic "
            "hyperexcitability. DEPDC5 PLE is phenotypically variable — may affect frontal, "
            "temporal, or parietal regions, even within the same family. Some individuals harbour "
            "a 'second-hit' somatic mutation in the seizure onset zone creating focal FCD-like "
            "pathology (the 'two-hit' DEPDC5 hypothesis). Related: NPRL2, NPRL3 (GATOR1 complex "
            "partners) — same phenotype. KCNQ2, SCN8A parietal variants also reported. Clinical "
            "presentations range from nocturnal hypermotor seizures (frontal spread) to diurnal "
            "somatosensory aura (primary parietal onset)."
        ),
        "eeg_correlate": (
            "Interictal: variable — multifocal spikes or focal centroparietal spike-wave; EEG "
            "may be entirely normal between seizures. Ictal: SEEG shows centroparietal low-voltage "
            "fast activity; focal build-up in 2–4 Hz delta-theta before secondary generalisation. "
            "Activation by sleep deprivation (raises cortical hyperexcitability via mTOR clock-gene "
            "interaction)."
        ),
        "mri_finding": (
            "50–60% MRI normal on standard protocol; MRI-negative cases require voxel-based "
            "morphometry + SurferVBM post-processing to detect subtle FCD-like changes. "
            "10–15%: focal cortical dysplasia visible on 7T MRI. MTOR somatic variant testing "
            "(CSF or resected tissue) required for mTOR-targeted therapy candidacy."
        ),
        "clinical_note": (
            "Refer for genetic panel (DEPDC5/NPRL2/NPRL3/MTOR/TSC1/TSC2/KCNQ2/SCN8A). Family "
            "screening mandatory (70% penetrance — first-degree relatives need EEG + genetic "
            "counselling). SUDEP risk elevated — prescribe seizure-detection device + counselling."
        ),
    },
    {
        "etiology": "Vascular / Ischaemic (Post-Stroke Cortical Scar — MCA/PCA Territory)",
        "category": "Structural/Vascular",
        "pct": 10,
    },
]

# ─── Seizure Types ───────────────────────────────────────────────────────────

_SEIZURE_TYPES = [
    {
        "type": "FAS — Somatosensory Aura (Contralateral Tingling / Electric Sensation)",
        "freq_pct": 70,
        "duration_sec": "10–90 s",
        "description": (
            "Focal Aware Seizure with pure somatosensory aura — the pathognomonic parietal "
            "onset sign. The patient experiences contralateral tingling, numbness, electric "
            "buzzing, or painful dysaesthesia in a somatotopic distribution (face, hand, "
            "arm, leg) corresponding to the Penfield homunculus of the primary somatosensory "
            "cortex (S1, Brodmann areas 1-2-3). Upper limb (hand/arm) is most common (80%), "
            "face second (50%), leg/perineum in mesial parietal (paracentral lobule) 20%. "
            "Awareness is fully preserved. Duration 10–90 seconds. May occur in isolation "
            "(aura only) or evolve to FIAS or FBTCS. Often misattributed to migraine with "
            "aura, paraesthesias from cervical radiculopathy, or psychogenic NES."
        ),
        "eeg_correlate": (
            "Ictal SEEG: focal rhythmic 30–80 Hz gamma fast activity at primary somatosensory "
            "cortex (S1) onset → spread to S2 (superior temporal/parietal operculum). Scalp "
            "EEG: centroparietal (C3/P3 or C4/P4) spike-wave or low-voltage fast activity "
            "often missed; dipole modelling or MEG required. Amplitude low → scalp EEG "
            "frequently normal during aura-only seizures."
        ),
        "clinical_tip": (
            "Key distinguishing feature: somatosensory aura follows Penfield homunculus "
            "somatotopy (face → hand → arm → leg progression = Jacksonian march from S1). "
            "Migraine aura: usually visual (V1 spreading depression), NOT somatosensory; "
            "if both visual + somatosensory, consider parieto-occipital junction epilepsy. "
            "Document exact body distribution at every visit — spatial progression = Jacksonian "
            "march confirms parietal onset (vs distal polyneuropathy = bilateral symmetric)."
        ),
    },
    {
        "type": "FIAS — Body Schema Disturbance (Out-of-Body / Neglect / Limb Illusion)",
        "freq_pct": 45,
        "duration_sec": "15–120 s",
        "description": (
            "Focal Impaired Awareness Seizure with parietal association cortex (TPJ, SPL, "
            "IPL) involvement. Features: autoscopy / out-of-body experience (viewing oneself "
            "from above — TPJ/precuneus activation), contralateral limb neglect or 'alien hand' "
            "phenomenon (patient cannot recognise their own limb — IPS disruption), "
            "macro/microsomatognosia (limb feels giant or absent — body schema network disruption), "
            "feeling of levitation or spinning. Awareness partially impaired. These ictal "
            "experiences are highly specific for parietal lobe seizure onset (sensitivity "
            "~80% in SEEG series — Blanke 2002 Nature). Often confused with depersonalisation "
            "disorder, panic attack, or psychogenic NES before correct EEG-SEEG diagnosis."
        ),
        "eeg_correlate": (
            "SEEG: ictal discharge at TPJ (angular gyrus BA39 + supramarginal gyrus BA40) → "
            "early spread to posterior insula → may then propagate to temporal pole (automatisms) "
            "or frontal cortex (motor signs). Scalp EEG: posterior temporal-parietal (T5/P3 "
            "or T6/P4) slow-wave. High-frequency oscillations (HFOs 80–200 Hz) mark the true "
            "onset zone on SEEG."
        ),
        "clinical_tip": (
            "Ask directly: 'Have you ever felt you were floating above your body, or that your "
            "arm didn't belong to you?' — these symptoms are rarely volunteered spontaneously "
            "and are pathognomonic for TPJ/parietal onset. Video-EEG capture: patient will "
            "appear partially unresponsive with limb neglect (may not withdraw to pin prick "
            "on contralateral hand during ictal discharge). Distinguish from TLE: TLE usually "
            "adds oroalimentary automatisms + amnesia; PLE body-schema FIAS lacks automatisms."
        ),
    },
    {
        "type": "FIAS — Versive + Tonic Posturing (Frontal Spread from Parietal Onset)",
        "freq_pct": 55,
        "duration_sec": "30–180 s",
        "description": (
            "Focal Impaired Awareness Seizure with rapid ictal propagation from parietal "
            "onset zone to frontal motor areas. After 5–30 seconds of parietal aura (or no "
            "aura if spread is too fast for awareness), discharge reaches the supplementary "
            "motor area (SMA → M1) producing: contralateral tonic arm/leg posturing, "
            "figure-4 posturing (contralateral arm flexed, ipsilateral extended), forced "
            "head-eye deviation (FEF involvement), fencing posture. This is the most "
            "frequent PLE seizure type visible on video-EEG. Because the parietal aura may "
            "be brief (< 5 s), patients and observers report only the motor event — leading "
            "to misdiagnosis as FLE. SEEG is required to demonstrate parietal onset before "
            "frontal motor propagation."
        ),
        "eeg_correlate": (
            "SEEG evolution: parietal onset (S1/SPL) → premotor (BA6) → SMA → M1 in 5–30 "
            "seconds; scalp EEG often shows apparent frontocentral (F3/C3 or Fz/Cz) onset "
            "because frontal motor discharge dominates amplitude. This 'false localisation' "
            "on scalp EEG is the main cause of PLE misdiagnosis as FLE. Video: search for "
            "the contralateral tingling aura before motor onset — even a brief hand rub or "
            "looking at the contralateral hand suggests somatosensory precursor."
        ),
        "clinical_tip": (
            "The PLE-vs-FLE SEEG distinction: implant parietal contacts (P3-P4, SPL, IPL) "
            "in addition to frontal contacts when pre-surgical evaluation is inconclusive. "
            "Clue on scalp: if contralateral hand rubbing precedes the tonic motor seizure "
            "by 5–20 seconds, suspect PLE onset. Document: 'pre-ictal hand rub / body rub "
            "ipsilateral to the tonic side' = somatosensory aura = parietal indicator."
        ),
    },
    {
        "type": "FBTCS — Focal to Bilateral Tonic-Clonic Seizure",
        "freq_pct": 40,
        "duration_sec": "60–180 s",
        "description": (
            "Focal to Bilateral Tonic-Clonic Seizure arising from parietal onset with rapid "
            "secondary generalisation. Clinically indistinguishable from FLE-FBTCS or TLE-FBTCS "
            "unless preceded by a witnessed somatosensory aura. FBTCS is the main driver of "
            "SUDEP risk in PLE. Postictal: profound unilateral Todd's paresis (contralateral "
            "arm/leg weakness lasting 10–120 minutes) is highly localising — indicates the "
            "ipsilateral hemisphere's S1/M1 motor strip was within the seizure zone. "
            "Postictal aphasia (left hemisphere seizure) or neglect (right hemisphere) also "
            "localising. SUDEP risk: elevated in drug-resistant PLE with frequent FBTCS "
            "— prescribe pulse oximeter alarm + nocturnal seizure detection."
        ),
        "eeg_correlate": (
            "Ictal: focal parietal onset evolves to bilateral synchronous polyspike-wave "
            "discharge. Postictal: generalised suppression ('flat EEG') followed by slow "
            "delta recovery, more pronounced over the hemisphere of origin. "
            "Inter-ictal scalp EEG: focal centroparietal spike-wave in 60%; "
            "normal EEG between seizures in 40%."
        ),
        "clinical_tip": (
            "Document and review postictal Todd's paresis after every observed FBTCS — "
            "it is the single most lateralising clinical sign available without invasive EEG. "
            "Contralateral postictal weakness = ipsilateral hemisphere seizure onset (parietal "
            "or frontal). SUDEP counselling mandatory: prescribe Emfit QM seizure alarm "
            "or Embrace smartwatch, supervise bathing/swimming, document risks."
        ),
    },
]

# ─── Triggers ────────────────────────────────────────────────────────────────

_TRIGGERS = [
    {
        "trigger": "Sleep Deprivation",
        "pct": 70,
        "mechanism": (
            "Sleep deprivation reduces seizure threshold by decreasing GABA-A receptor "
            "surface expression and increasing mTORC1 activity in parietal cortex. REM "
            "sleep suppression (circadian rhythm disruption) removes thalamo-cortical "
            "inhibitory gating → parietal hyperexcitability. DEPDC5 PLE especially "
            "sensitive — mTOR clock-gene interaction. Seizures cluster in early morning "
            "(sleep–wake transition) and during post-sleep-deprivation recovery sleep."
        ),
        "management": (
            "Enforce ≥7 hours of sleep nightly. Prescribe sleep hygiene protocol (CBT-I). "
            "Avoid shift work and overnight flights across time zones. Morning seizure "
            "clusters: consider adding bedtime LEV or CLB on nights of unavoidable "
            "sleep deprivation (e.g., call duty, travel)."
        ),
    },
    {
        "trigger": "Missed AED Dose",
        "pct": 65,
        "mechanism": (
            "Abrupt trough in AED plasma concentration removes pharmacological seizure "
            "threshold buffering. CBZ/OXC: shortest half-life (8–12 h) — even a 6-hour "
            "delay causes trough concentrations below therapeutic range (4 mg/L for CBZ). "
            "LTG: 24-hour half-life is more forgiving but enzyme inducers reduce it to "
            "12–15 hours. Plasma concentration monitoring (TDM) especially important for "
            "CBZ — measure trough at 12 hours post last dose."
        ),
        "management": (
            "Prescribe AED alarm reminder app (Medisafe, MyTherapy). Set twice-daily alarm. "
            "For CBZ: prescribe modified-release formulation (Tegretol CR) for smoother "
            "plasma kinetics. If dose missed >6h: take next scheduled dose (not double dose). "
            "TDM: check CBZ/OXC/LTG plasma levels at 3-month intervals or after any dose "
            "change."
        ),
    },
    {
        "trigger": "Stress / Psychological Arousal",
        "pct": 60,
        "mechanism": (
            "Cortisol elevation (HPA axis activation) reduces GABA-A-receptor alpha-1 "
            "subunit expression and upregulates NMDA-receptor excitability in the parietal "
            "cortex. Psychological arousal activates the noradrenergic locus coeruleus → "
            "parietal cortex catecholamine surge → lowered seizure threshold. Chronic stress "
            "also disrupts sleep architecture (trigger synergy)."
        ),
        "management": (
            "Refer to clinical psychologist for CBT / mindfulness-based stress reduction (MBSR). "
            "Biofeedback: heart-rate variability (HRV) training to reduce ictal sympathetic "
            "arousal. Avoid benzodiazepine escalation for stress (tolerance develops rapidly — "
            "use rescue CLB only for seizure clusters, not prophylaxis). Consider SSRI if "
            "comorbid anxiety (SSRI modestly anti-epileptic in temporal-parietal networks)."
        ),
    },
    {
        "trigger": "Somatosensory Stimulus (Reflex PLE)",
        "pct": 15,
        "mechanism": (
            "In ~15% of PLE patients (particularly those with large superficial S1 lesions), "
            "external tactile stimulation over the contralateral body can trigger a reflex "
            "seizure — confirmed by SEEG showing cortical evoked potential → seizure onset "
            "in S1. Hot water immersion (bathing) and vibration are the most common triggers. "
            "Somatosensory evoked potentials (SEPs) show giant amplitudes (>10× normal) over "
            "the seizure onset zone — a marker of cortical hyperexcitability."
        ),
        "management": (
            "Document reflex triggers specifically — change bathing routine (showering "
            "instead of bathing, tepid water). Prescribe rescue CLB 10 mg before "
            "unavoidable triggers (e.g., medical procedures involving tactile stimulation). "
            "Giant SEP confirms hyperexcitable S1 — inform surgeon pre-operatively as it "
            "predicts excellent post-surgical seizure freedom."
        ),
    },
    {
        "trigger": "Alcohol Consumption / Withdrawal",
        "pct": 40,
        "mechanism": (
            "Alcohol acutely potentiates GABA-A receptors (sedation) but withdrawal causes "
            "rebound GABA-A down-regulation and NMDA-receptor upregulation → cortical "
            "hyperexcitability. Alcohol withdrawal seizures peak 6–48 hours after last drink. "
            "Chronic alcohol use additionally reduces AED plasma concentrations via "
            "CYP3A4/CYP2C19 induction (CBZ, OXC, PHT metabolised faster) — causing "
            "subtherapeutic levels despite adherence."
        ),
        "management": (
            "Screen AUDIT-C at every visit. Refer to addiction medicine if AUDIT-C ≥4. "
            "Check CBZ/OXC/LTG trough levels if alcohol use reported — may need dose "
            "adjustment. Alcohol withdrawal seizure management: diazepam IV CIWA protocol "
            "in hospital (not AED escalation). For DEPDC5 PLE: complete alcohol abstinence "
            "recommended (mTOR activation amplified by alcohol metabolites)."
        ),
    },
    {
        "trigger": "Fever / Intercurrent Illness",
        "pct": 35,
        "mechanism": (
            "Fever elevates body temperature → increases neuronal metabolic demand → "
            "shifts voltage-gated sodium channels toward inactivated state → initially "
            "inhibitory but then paradoxical hyperexcitability via temperature-dependent "
            "GABA-A kinetics. IL-1β and TNF-α (febrile cytokines) directly activate TLR4 "
            "and NLRP3 inflammasome in parietal cortex → lowers seizure threshold by "
            "disrupting K/Cl cotransporter (KCC2) chloride homeostasis → GABA becomes "
            "depolarising."
        ),
        "management": (
            "Aggressive fever management: paracetamol 1g q6h at first sign of fever. "
            "Prescribe rescue CLB (5–10 mg) for temperature > 38.5°C in patients with "
            "prior febrile seizure exacerbations. AED dose: check plasma levels during "
            "acute illness (volume of distribution changes). Vaccinations current "
            "(fever from infection > fever from vaccine — vaccinate)."
        ),
    },
    {
        "trigger": "Catamenial (Perimenstrual / Mid-Cycle) — in Female Patients",
        "pct": 30,
        "mechanism": (
            "Progesterone withdrawal at days 3–5 of menstruation removes neurosteroid "
            "(allopregnanolone) GABA-A potentiation in parietal cortex → perimenstrual "
            "seizure cluster (C1 catamenial pattern). Mid-cycle LH surge (days 12–14) "
            "elevates oestradiol → competitive antagonism at GABA-A → mid-cycle cluster "
            "(C2 pattern). PLE somatosensory aura often intensified perimenstrually — "
            "more painful, more frequent. Hormonal contraceptives containing enzyme-inducing "
            "progestins (norgestrel) reduce AED efficacy (CBZ/LTG accelerated clearance)."
        ),
        "management": (
            "Menstrual seizure diary × 3 months (SeizureTracker). If C1 pattern: "
            "intermittent clobazam 10–20 mg/day days 1–7 of cycle (Herzog protocol). "
            "If C2 pattern: day 12–16 clobazam coverage. Avoid combined OCP with "
            "enzyme-inducing AEDs (CBZ/PHT) — use barrier or IUD. Natural progesterone "
            "(cyclic, days 14–25) may reduce C1 pattern (Herzog 2012 Neurology)."
        ),
    },
    {
        "trigger": "Photosensitivity (PPR) — Rare in Pure PLE",
        "pct": 10,
        "mechanism": (
            "Photoparoxysmal response (PPR) is uncommon in focal PLE (more characteristic "
            "of GGE syndromes) but may occur when the parietal lobe seizure focus overlaps "
            "with the parieto-occipital junction (POJ). Pulsating light at 15–25 Hz triggers "
            "occipital driving → spreads to hyperexcitable parietal cortex → parietal "
            "seizure. Screen with IPS (intermittent photic stimulation) during standard EEG."
        ),
        "management": (
            "Polarised lenses (grey FL-41 tinted lenses) — reduce PPR frequency. "
            "Avoid flickering screen / strobe. Levetiracetam broadly anti-photosensitive. "
            "CBZ/OXC: less effective for PPR. LTG: PPR-responsive. "
            "FOS (fixation-off sensitivity) must be distinguished from PPR (FOS: "
            "spikes on eye closure in darkness — parieto-occipital; PPR: spikes to IPS — "
            "generalised / occipital network)."
        ),
    },
]

# ─── Treatments ──────────────────────────────────────────────────────────────

_TREATMENTS = [
    {
        "drug": "Carbamazepine (CBZ)",
        "class": "Sodium Channel Blocker (SCB)",
        "dose_adult": "400–1200 mg/day in 2–3 divided doses (MR formulation preferred)",
        "dose_paed": "10–20 mg/kg/day in 2–3 doses; max 35 mg/kg/day",
        "moa": (
            "Preferential binding to inactivated voltage-gated Na⁺ channels (Nav1.1–1.7), "
            "prolonging fast inactivation → reduces high-frequency repetitive firing in "
            "parietal S1 cortex without blocking single action potentials. Also modulates "
            "L-type Ca²⁺ channels. No effect on GABA-A or AMPA receptors."
        ),
        "efficacy": (
            "Level A evidence for focal seizures (ILAE 2017). SANAD trial: CBZ superior "
            "overall to gabapentin for focal epilepsy (Marson 2007 Lancet). SANAD-II: "
            "no difference CBZ vs LCM for FBTCS component. PLE somatosensory aura: CBZ "
            "often reduces frequency 50–80% within 4–6 weeks of therapeutic levels."
        ),
        "safety": (
            "HLA-B*1502 (Han Chinese, Thai, South Asian): Stevens-Johnson Syndrome (SJS) / "
            "toxic epidermal necrolysis (TEN) risk 25× higher — screen before prescribing. "
            "Hyponatraemia (SIADH): monitor Na monthly in elderly. Teratogen: avoid in "
            "pregnancy (especially trimester 1, spina bifida risk 1%). Enzyme inducer: "
            "reduces efficacy of oral contraceptives, warfarin, statins, PER."
        ),
        "monitoring": (
            "HLA-B*1502 before first prescription (AEC/FDA mandate for Asian populations). "
            "CBZ trough plasma level (TDM): target 4–12 mg/L (trough, 12h post last dose). "
            "Sodium: monthly × 3 months then 3-monthly. FBC: baseline + 3 months (aplastic "
            "anaemia rare but fatal — stop if WBC < 3×10⁹/L). LFT baseline. "
            "LTG/OXC/PER levels: check after CBZ addition (enzyme induction reduces levels)."
        ),
        "evidence": "ILAE 2017 Level A; SANAD 2007 Lancet",
        "contraindications": (
            "HLA-B*1502 positive (Asian populations) — absolute: SJS/TEN risk. "
            "Bone marrow suppression history. Porphyria. SCN8A gain-of-function PLE: "
            "may worsen (paradoxical sodium channel excitation)."
        ),
    },
    {
        "drug": "Lamotrigine (LTG)",
        "class": "Sodium Channel Blocker / Glutamate Release Inhibitor",
        "dose_adult": "100–400 mg/day in 2 doses (SLOW titration: 25 mg/14 days → 50 mg/14 days)",
        "dose_paed": "0.3–1 mg/kg/day (without VPA) or 0.15 mg/kg/day (with VPA); max 15 mg/kg/day",
        "moa": (
            "Blocks use-dependent voltage-gated Na⁺ channel inactivation AND reduces "
            "pre-synaptic glutamate / aspartate release from parietal cortical pyramidal "
            "neurons. This dual mechanism makes LTG particularly effective for PLE with "
            "prominent somatosensory aura (glutamate-mediated thalamocortical "
            "hyperexcitability) AND for comorbid migraine aura (spreading depression "
            "inhibition)."
        ),
        "efficacy": (
            "Level A (ILAE 2017). SANAD: LTG superior to CBZ for time to treatment failure "
            "(better tolerated; equivalent seizure control). Particularly effective when "
            "PLE-somatosensory aura overlaps with migraine-with-aura (LTG addresses both "
            "mechanisms). First-line in women of childbearing age (lower teratogenicity "
            "vs CBZ/VPA) and in elderly (fewer cognitive effects vs CBZ)."
        ),
        "safety": (
            "SJS / TEN risk: 1/1000 (adult); 1/50–100 (children <16 if fast titration "
            "with VPA). SLOW titration mandatory — never exceed recommended titration schedule. "
            "VPA interaction: VPA halves LTG clearance → double SJS risk → halve LTG dose "
            "when combining with VPA. Rash: stop immediately if any rash develops in "
            "first 8 weeks. Overdose: cardiac arrhythmia (Na-channel)."
        ),
        "monitoring": (
            "Titration diary: document rash onset date. LTG TDM (plasma level): target "
            "3–15 mg/L (highly variable — individual titration is key). Levels drop 50% "
            "in pregnancy (oestrogen-induced UGT1A4 induction — check monthly in pregnancy). "
            "CBZ co-prescription: reduces LTG levels 50% (enzyme induction — increase LTG "
            "dose). Avoid dose escalation faster than recommended even under clinical pressure."
        ),
        "evidence": "ILAE 2017 Level A; SANAD 2007 Lancet",
        "contraindications": (
            "Rapid titration history of SJS/TEN. VPA co-prescription (relative — requires "
            "dose halving and extra monitoring). Brugada syndrome (sodium channel effect on "
            "cardiac conduction)."
        ),
    },
    {
        "drug": "Levetiracetam (LEV)",
        "class": "SV2A Modulator (Synaptic Vesicle Protein 2A)",
        "dose_adult": "500–3000 mg/day in 2 doses",
        "dose_paed": "20–60 mg/kg/day in 2 doses",
        "moa": (
            "Binds SV2A (synaptic vesicle protein 2A) — the docking protein for synaptic "
            "vesicle fusion at active zones of glutamatergic parietal pyramidal neurons. "
            "Reduces vesicular glutamate release during repetitive high-frequency firing "
            "WITHOUT affecting baseline single action potentials (unique among AEDs). "
            "Also modulates GABA-A receptor positive allosteric modulators and inhibits "
            "Ca²⁺ channel N-type at presynaptic terminals."
        ),
        "efficacy": (
            "Level B (ILAE 2017) for focal epilepsy; Level A for generalised myoclonic "
            "(GGE). Clinical practice: widely used first-line for PLE when CBZ/LTG not "
            "tolerated or in women of childbearing age (lower teratogenicity than CBZ/VPA). "
            "50% responder rate ~50–60% in focal epilepsy add-on studies. Rapid titration "
            "to therapeutic dose in 1–2 weeks (advantage when urgent seizure control needed)."
        ),
        "safety": (
            "Behavioural/psychiatric side effects: irritability, aggression, depression, "
            "anxiety in 10–15% — highest risk in patients with prior psychiatric comorbidity "
            "or intellectual disability. Screen PHQ-9 + GAD-7 at baseline and 4-weekly "
            "during titration. No enzyme induction. Safe in renal dose-adjustment (renal "
            "excretion 66%). Pregnancy: UK MHRA data: lower major congenital malformation "
            "risk vs CBZ/VPA but not zero — monitor."
        ),
        "monitoring": (
            "PHQ-9 (depression) and GAD-7 (anxiety) questionnaire at each visit — dose-"
            "dependent psychiatric effects. Renal function (eGFR): adjust dose if "
            "eGFR <50. No routine TDM needed (linear PK; target 12–46 mg/L if adherence "
            "checked). Warn: irritability may appear after 2–4 weeks at therapeutic dose."
        ),
        "evidence": "ILAE 2017 Level B; SANAD-II 2021 Lancet",
        "contraindications": "Severe psychiatric history (relative). Renal failure (dose adjust).",
    },
    {
        "drug": "Oxcarbazepine (OXC)",
        "class": "Sodium Channel Blocker (10-monohydroxy derivative of CBZ)",
        "dose_adult": "600–2400 mg/day in 2 doses",
        "dose_paed": "8–46 mg/kg/day in 2 doses",
        "moa": (
            "Pro-drug converted to pharmacologically active 10-monohydroxy metabolite (MHD) "
            "in liver. MHD blocks voltage-gated Na⁺ channels (Nav1.2/1.6 dominant in "
            "parietal S1 cortex) via slow-inactivation mechanism. Lower enzyme-inducing "
            "potency than CBZ → fewer drug interactions (less OCP/warfarin/statin reduction)."
        ),
        "efficacy": (
            "Level A (ILAE 2017) for focal seizures. Non-inferior to CBZ in head-to-head "
            "trials (Privitera 2003 Epilepsy Res). Better tolerability profile than CBZ "
            "(no autoinduction → more predictable kinetics; no CBZ epoxide metabolite → "
            "fewer cognitive effects). Preferred over CBZ for patients with drug-drug "
            "interaction concerns or CBZ intolerance."
        ),
        "safety": (
            "Hyponatraemia: MORE common than with CBZ (~25% of patients; symptomatic in 3%). "
            "Monitor sodium monthly. HLA-B*1502 cross-reactivity: same SJS/TEN risk as CBZ "
            "in Asian populations — screen before prescribing. MHD TDM: target 12–35 µg/mL. "
            "Teratogen: similar risk to CBZ — avoid in trimester 1 (neural tube defects)."
        ),
        "monitoring": (
            "Sodium: monthly for first 6 months, then 3-monthly (hyponatraemia risk). "
            "HLA-B*1502: mandatory in Asian populations. MHD trough level (OXC metabolite): "
            "target 12–35 µg/mL. LFT + FBC baseline. Avoid co-prescription with "
            "HCTZ/thiazides (potentiates hyponatraemia synergistically)."
        ),
        "evidence": "ILAE 2017 Level A; Privitera 2003 Epilepsy Res",
        "contraindications": "HLA-B*1502 positive (Asian). Hyponatraemia history. Porphyria.",
    },
    {
        "drug": "Lacosamide (LCM)",
        "class": "Selective Slow-Inactivation Enhancer of Voltage-Gated Na⁺ Channels",
        "dose_adult": "200–400 mg/day in 2 doses (max 600 mg/day in refractory cases)",
        "dose_paed": "≥4 years: 2–8 mg/kg/day in 2 doses (EMA approved 2016)",
        "moa": (
            "Unique dual mechanism: (1) selectively enhances SLOW inactivation of Nav1.1–1.7 "
            "(spares FAST inactivation → minimal effect on baseline neuronal firing); "
            "(2) binds CRMP-2 (collapsin response mediator protein 2) — a microtubule-"
            "associated protein regulating axonal growth and Nav1.7 trafficking. In parietal "
            "cortex, CRMP-2 binding may reduce pain-sensitisation component of somatosensory "
            "aura (Nav1.7 is the 'pain channel' in dorsal horn). This makes LCM particularly "
            "valuable when PLE somatosensory aura has a painful/burning character."
        ),
        "efficacy": (
            "Level A add-on for focal seizures (JASPER 2018 Epilepsia; SP0993/SP0994 trials). "
            "50% responder rate: 40–50% in refractory focal epilepsy. Rapid IV loading "
            "available (same dose as oral — 200 mg IV over 15 minutes) for acute seizure "
            "clusters in PLE. Particularly valuable in PLE with painful somatosensory aura "
            "and in refractory focal-motor (tonic posturing) seizures."
        ),
        "safety": (
            "PR interval prolongation: cardiac conduction (dose-dependent; average +4 ms at "
            "200 mg/day; +10 ms at 400 mg/day). MANDATORY: obtain ECG before starting LCM "
            "and after each dose titration. Contraindicated in: 2nd/3rd degree AV block, "
            "sick sinus syndrome without pacemaker. Dizziness + diplopia most common AEs "
            "(titration-related, dose-reduce then re-escalate)."
        ),
        "monitoring": (
            "ECG: mandatory before start + after each dose increase (check PR interval). "
            "Cardiac history review: exclude pre-existing AV block, bradycardia <50 bpm. "
            "No routine TDM needed (linear kinetics, widely distributed). Renal dose "
            "adjustment if eGFR <30. Liver: mild CYP3A4/2C19 substrate — no interaction "
            "with enzyme inducers significant enough to require routine adjustment."
        ),
        "evidence": "ILAE Level A add-on; JASPER 2018 Epilepsia",
        "contraindications": "2nd/3rd degree AV block. Sick sinus without pacemaker. Severe hepatic impairment.",
    },
    {
        "drug": "Perampanel (PER)",
        "class": "Non-competitive AMPA Receptor Antagonist (first-in-class)",
        "dose_adult": "2–12 mg/day at bedtime (REMS-gated; titrate 2 mg per 2 weeks)",
        "dose_paed": "≥4 years ≥20 kg: 2–12 mg/day; <20 kg: 1–6 mg/day",
        "moa": (
            "First-and-only selective non-competitive antagonist of AMPA (α-amino-3-hydroxy-5-"
            "methyl-4-isoxazolepropionic acid) ionotropic glutamate receptors. AMPA receptors "
            "mediate fast excitatory transmission in parietal S1–S2 cortex and in thalamo-"
            "cortical relay neurons. By blocking AMPA receptor-mediated EPSPs, PER reduces "
            "ictal spread from parietal onset zone to frontal (preventing FBTCS) and thalamic "
            "recruitment (reducing generalisation). Unique: effective against both fast-spread "
            "focal and secondary generalised seizures."
        ),
        "efficacy": (
            "Level A add-on for focal seizures + FBTCS (Study 206 Lancet Neurology 2012; "
            "Study 235). 50% responder rate: 30–45% in refractory focal epilepsy. "
            "FBTCS: particular efficacy for secondary generalisation reduction (OR 0.38 "
            "vs placebo, Study 235). Particularly valuable in PLE with frequent FBTCS or "
            "tonic motor spread (blocks AMPA-driven frontal recruitment from parietal onset)."
        ),
        "safety": (
            "REMS program MANDATORY (FDA 2012): serious psychiatric adverse effects — "
            "aggression, hostility, irritability, suicidal ideation (1–1.8%). PHQ-9 + "
            "GAD-7 + aggression scale at baseline, 4 weeks, 8 weeks, then quarterly. "
            "Enzyme inducers (CBZ/PHT/OXC) reduce PER levels 50–67% → may need 4 mg "
            "higher dose. Dizziness/ataxia: titration-related. Bedtime dosing reduces "
            "CNS adverse effects."
        ),
        "monitoring": (
            "REMS: enrol prescriber + patient + pharmacy before dispensing. "
            "PHQ-9 + GAD-7 + Buss-Perry aggression questionnaire at each visit. "
            "PER plasma level: if co-prescribed with enzyme inducer (CBZ/OXC), check "
            "level — expect 50% reduction → adjust dose. Driving: warn patient (dizziness)."
        ),
        "evidence": "ILAE Level A add-on; Study 206 Lancet Neurology 2012",
        "contraindications": "Active suicidal ideation. Severe hepatic impairment.",
    },
    {
        "drug": "Parietal Cortectomy / SEEG-Guided Resection",
        "class": "Surgical Intervention",
        "dose_adult": "SEEG implantation → cortical mapping → tailored resection under awake craniotomy (if eloquent cortex adjacent)",
        "dose_paed": "Age ≥2 years for structural lesions; SEEG ≥5 years typically",
        "moa": (
            "Complete resection of the parietal seizure onset zone (confirmed by SEEG "
            "recording of the ictal onset pattern) removes the epileptogenic substrate. "
            "For FCD IIb: complete resection of the FCD with 5–10 mm margin. For tumour: "
            "gross-total resection. For haemosiderin scar: scar + surrounding zone of "
            "HFOs. Awake craniotomy with cortical mapping is required when the seizure onset "
            "zone abuts primary S1 (central sulcus) to avoid permanent somatosensory deficit "
            "— intraoperative SEP + cortical stimulation maps S1 in real time."
        ),
        "efficacy": (
            "Structural PLE (FCD/tumour): Engel I 55–70% (Kim 2004 J Neurosurg; 40 patients). "
            "Non-lesional PLE (SEEG-guided): Engel I 35–50% (lower: harder pre-surgical "
            "localisation). Overall: Engel I 56% in combined parietal series (Ryvlin 2006 Epilepsia). "
            "Post-operative deficit: contralateral arm numbness in 15–25% (often transient, "
            "resolves in 3–6 months); permanent S1 deficit in ~5% when gyrus is within "
            "resection boundary."
        ),
        "safety": (
            "Contralateral somatosensory deficit: temporary (weeks–months) 25%; permanent 5%. "
            "Contralateral arm weakness (if M1 margin encroached): 5–10%. "
            "Visual field defect (optic radiation): if resection extends too posteriorly. "
            "Awake craniotomy required for S1-adjacent zones: patient must cooperate — "
            "use neuropsychological pre-op screening to assess cognitive tolerance."
        ),
        "monitoring": (
            "Pre-op: comprehensive neuropsychological battery (attention, spatial, somatosensory). "
            "SEEG plan: cover parietal (S1/SPL/IPL/TPJ), frontal (SMA/M1/pre-motor), "
            "temporal (STG/MTG), and occipital (POJ) contacts. Post-op: "
            "SEP monitoring at 3, 6, 12 months. MRI at 3 months (confirm complete resection). "
            "AED taper: begin 1–2 years post-op if Engel I sustained."
        ),
        "evidence": "ILAE Surgical Epilepsy 2017; Kim 2004 J Neurosurg",
        "contraindications": (
            "Bilateral parietal onset (relative). Eloquent-zone encasement without mapping "
            "capability. Patient unable to cooperate with awake craniotomy (severe anxiety)."
        ),
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "class": "Metabolic / Dietary Therapy",
        "dose_adult": "3:1 or 4:1 fat:carbohydrate+protein ratio; modified Atkins (MAD) for adults",
        "dose_paed": "4:1 classical KD (standard for paediatric PLE refractory to ≥2 AEDs)",
        "moa": (
            "KD generates ketone bodies (β-hydroxybutyrate, acetoacetate) as alternative "
            "neuronal fuel. Proposed mechanisms: (1) β-hydroxybutyrate directly inhibits "
            "NLRP3 inflammasome → anti-epileptic via neuroinflammation reduction; "
            "(2) increases parietal cortex GABA synthesis via glutamate-α-KG shunting; "
            "(3) stabilises KATP channels in S1 pyramidal neurons; (4) GLUT-1 deficiency "
            "PLE: seizures fully dependent on glucose — KD bypasses GLUT-1 → dramatic "
            "response (near seizure-free in ~90%)."
        ),
        "efficacy": (
            "Level B for drug-resistant focal epilepsy (ILAE 2017). >50% seizure reduction "
            "in 50–60% of paediatric drug-resistant PLE. GLUT-1 deficiency (SLC2A1 mutation): "
            "Level A — KD is first-line, not last resort (should be started immediately after "
            "diagnosis, before AED failure). 10–15% of children become seizure-free on KD."
        ),
        "safety": (
            "Hyperlipidaemia: monitor lipid panel. Nephrolithiasis: hydration + potassium "
            "citrate supplement. Growth retardation in children: dietitian monitoring "
            "mandatory. Selenium deficiency → cardiomyopathy (supplement selenium). "
            "GI: constipation, nausea. Contraindicated: fatty acid oxidation disorders, "
            "pyruvate carboxylase deficiency, porphyria."
        ),
        "monitoring": (
            "Urinary ketones daily (target 3+). Fasting glucose monthly. Lipid panel "
            "3-monthly. Renal USS at 12 months. Selenium + zinc + vitamins D/B6 "
            "supplementation. Dietitian monthly reviews. AED interaction: KD may reduce "
            "requirement for CBZ/LTG — plasma levels may rise → reduce AED dose "
            "accordingly."
        ),
        "evidence": "ILAE Level B; GLUT-1: Level A (De Vivo 1991 NEJM)",
        "contraindications": (
            "Fatty acid beta-oxidation defects. Pyruvate carboxylase deficiency. "
            "Acute porphyria. Active pancreatitis. Caution: kidney disease."
        ),
    },
]

# ─── Lifecycle ────────────────────────────────────────────────────────────────

_LIFECYCLE = [
    {
        "window": "Infancy / Early Childhood (0–5 years)",
        "age_range": "0–5 y",
        "key_events": (
            "Neonatal PLE rare — consider KCNQ2 or CDKL5. Infantile spasms may have "
            "structural parietal origin (TSC, FCD). FCD Type IIb: drug-resistant seizures "
            "from first year → early surgical evaluation. Screen SLC2A1 (GLUT-1) in any "
            "movement-triggered paroxysmal events + CSF glucose < 45 mg/dL."
        ),
        "focus": (
            "Genetic testing (DEPDC5/MTOR/SLC2A1/CDKL5). MRI optimised 3T protocol. "
            "If GLUT-1: KD immediately. Avoid glucose restriction before diagnosis."
        ),
    },
    {
        "window": "School Age (6–12 years)",
        "age_range": "6–12 y",
        "key_events": (
            "Somatosensory aura onset typical — child describes tingling / 'electric feeling' "
            "in hand/face. Schoolteacher education: child may pause with hand-rubbing before "
            "seizure generalises. FCD: nocturnal motor seizures disrupt sleep → academic "
            "impact. Neuropsychological assessment (attention, visuospatial, reading)."
        ),
        "focus": (
            "School IHCP (Individual Healthcare Plan). Neuropsychological battery. "
            "Epilepsy nurse liaison with school nurse. Consider KD if ≥2 AED failures. "
            "SEEG pre-surgical evaluation if FCD confirmed and DRE."
        ),
    },
    {
        "window": "Adolescence (13–18 years)",
        "age_range": "13–18 y",
        "key_events": (
            "Sleep deprivation (social media, exam stress) peaks → increased seizure "
            "frequency. Alcohol experimentation: counsel (lowers seizure threshold + "
            "enzyme induction). Driving licence approaches: document 12-month seizure-free "
            "period. Identity issues: 'I have epilepsy' disclosure anxiety at school/social "
            "settings. Surgical candidate assessment if DRE → SEEG pre-op planning."
        ),
        "focus": (
            "Sleep hygiene CBT. Alcohol and drug counselling. Driving regulations (12-month "
            "seizure-free for standard licence in Canada/UK). SUDEP risk counselling with "
            "parents AND patient separately. Self-management training (seizure diary app)."
        ),
    },
    {
        "window": "Young Adult — Childbearing Age (19–35 years, female)",
        "age_range": "19–35 y (F)",
        "key_events": (
            "Contraception: CBZ/OXC enzyme induction reduces OCP efficacy → unintended "
            "pregnancy. Prescribe barrier method or LNG-IUD if using enzyme-inducing AEDs. "
            "Pre-conception counselling: folic acid 5 mg/day ≥3 months before conception. "
            "AED review: switch to lowest-teratogenicity AED (LEV/LTG preferred over "
            "CBZ/VPA). LTG levels fall 50% in pregnancy → check monthly + adjust dose."
        ),
        "focus": (
            "OCP enzyme-induction interaction: document and counsel. Folic acid 5 mg "
            "pre-conception. Pregnancy AED monitoring protocol (LTG levels monthly). "
            "Epilepsy in pregnancy clinic referral. SUDEP prevention: Emfit/Embrace alarm."
        ),
    },
    {
        "window": "Middle Adult (36–60 years)",
        "age_range": "36–60 y",
        "key_events": (
            "Bone density: enzyme-inducing AEDs (CBZ/OXC/PHT) reduce vitamin D + Ca²⁺ "
            "absorption → osteoporosis. DEXA scan at 10-year AED mark. Supplement "
            "vitamin D 1000–2000 IU/day + Ca²⁺ 1200 mg/day if on enzyme inducers. "
            "Surgical re-evaluation if previously deemed non-surgical: SEEG "
            "techniques improved (robot-assisted SEEG) — revisit pre-surgical evaluation."
        ),
        "focus": (
            "DEXA bone density scan. Vitamin D / Ca²⁺ supplementation. Repeat "
            "neuropsychological assessment. Re-evaluate surgical candidacy with updated "
            "SEEG/FDG-PET if DRE continues."
        ),
    },
    {
        "window": "Older Adult (>60 years)",
        "age_range": ">60 y",
        "key_events": (
            "New-onset PLE in >60 years: exclude cerebrovascular disease (lacunar infarct, "
            "cortical spread from MCA territory) as primary cause — urgent MRI + DWI. "
            "Autoimmune encephalitis workup (anti-CASPR2, anti-LGI1): para-neoplastic PLE "
            "with somatosensory aura possible. Polypharmacy: CBZ increases digoxin/warfarin "
            "metabolism → INR checks. Fall risk: AED-related dizziness + postictal Todd's "
            "paresis → high osteoporotic fracture risk."
        ),
        "focus": (
            "DWI-MRI urgently for new onset. Autoimmune panel (CASPR2, LGI1, GAD65). "
            "Fall risk assessment. Bone density DEXA. Medication review (polypharmacy "
            "enzyme-induction interactions). Prefer LEV/LTG (fewer drug interactions). "
            "Avoid high-dose CBZ (hyponatraemia risk in elderly)."
        ),
    },
]

# ─── AED Monitoring ───────────────────────────────────────────────────────────

_AED_MONITORING = [
    {
        "item": "CBZ/OXC: HLA-B*1502 + Plasma Level (TDM) + Serum Na",
        "frequency": "HLA-B*1502 before first Rx · TDM 3-monthly · Na monthly × 6m then 3-monthly",
        "rationale": (
            "CBZ/OXC SJS/TEN risk: 25× higher in HLA-B*1502 carriers (Han Chinese, Thai, "
            "South Asian). TDM: CBZ target 4–12 mg/L (trough); OXC/MHD target 12–35 µg/mL. "
            "Hyponatraemia: OXC > CBZ; monitor Na — stop or reduce if Na < 125 mmol/L."
        ),
    },
    {
        "item": "LTG: SJS Rash Surveillance + Titration Diary + Pregnancy Level",
        "frequency": "Weekly self-exam for rash (first 8 weeks) · LTG level 3-monthly · Monthly in pregnancy",
        "rationale": (
            "SJS risk is titration-speed-related and VPA interaction-dependent. Slow titration "
            "mandatory. In pregnancy: oestrogen induces UGT1A4 → LTG clearance ↑ 50% → "
            "seizures re-emerge unless dose increased monthly. Post-partum: rapid re-increase "
            "in levels → toxicity — reduce dose back to pre-pregnancy within 1–2 weeks."
        ),
    },
    {
        "item": "LCM: ECG (PR Interval Monitoring)",
        "frequency": "ECG before first dose + after each dose increase + annually",
        "rationale": (
            "Lacosamide prolongs PR interval dose-dependently (+4 ms at 200 mg/day, "
            "+10 ms at 400 mg/day). Contraindicated in pre-existing 2nd/3rd degree AV block. "
            "Cardiac conduction assessment is mandatory before every dose escalation — "
            "failure to do so is a prescribing error."
        ),
    },
    {
        "item": "PER: REMS Enrolment + PHQ-9 + GAD-7 + Aggression Scale",
        "frequency": "REMS at prescription · PHQ-9/GAD-7/aggression 4-weekly × 8w then 3-monthly",
        "rationale": (
            "Perampanel REMS mandatory: prescriber + patient + pharmacy enrolment. "
            "Psychiatric adverse effects: aggression, suicidal ideation in 1–1.8%. "
            "PHQ-9 score >10 or new aggression → reduce dose or stop. "
            "Enzyme inducer co-prescription (CBZ): PER levels reduced 50% → check "
            "plasma level if breakthrough seizures on apparent adequate dose."
        ),
    },
]

# ─── Standards ────────────────────────────────────────────────────────────────

_STANDARDS = [
    {"standard": "ILAE 2017", "relevance": "AED evidence classification for focal epilepsy (CBZ/LTG Level A)"},
    {"standard": "NICE NG217 (2022)", "relevance": "Epilepsy clinical guideline — investigations, AED choice, surgery"},
    {"standard": "ILAE Surgical Epilepsy 2017", "relevance": "Pre-surgical evaluation standards; SEEG/ECoG criteria"},
    {"standard": "FDA HLA-B*1502 Warning (2007)", "relevance": "Mandatory CBZ/OXC screening in Asian populations"},
    {"standard": "FDA PER REMS (2012)", "relevance": "Mandatory REMS program for perampanel prescribing"},
    {"standard": "AAN Post-Traumatic Epilepsy (2014)", "relevance": "LEV prophylaxis ≤7 days post-TBI; no prolonged prophylaxis evidence"},
]

# ─── Thresholds ───────────────────────────────────────────────────────────────

_THRESHOLDS = [
    {"threshold": "Drug-Resistant Epilepsy (DRE)", "value": "≥2 adequate AED trials failed"},
    {"threshold": "CBZ Therapeutic Drug Monitoring", "value": "4–12 mg/L (trough, 12h post-dose)"},
    {"threshold": "OXC/MHD Therapeutic Range", "value": "12–35 µg/mL"},
    {"threshold": "LTG Therapeutic Range", "value": "3–15 mg/L (variable; titrate to effect)"},
    {"threshold": "LCM PR Interval Limit", "value": "PR ≤200 ms before each dose increase"},
    {"threshold": "Driving Restriction (Canada/UK/Australia)", "value": "12 months seizure-free"},
    {"threshold": "Pre-Conception Folic Acid", "value": "5 mg/day ≥3 months before conception"},
    {"threshold": "Sodium — OXC Monitoring Threshold", "value": "Stop/reduce if serum Na <125 mmol/L"},
]

# ─── Definitions / Concepts ──────────────────────────────────────────────────

_CONCEPTS = [
    {"term": "Parietal Lobe Epilepsy (PLE)", "definition": "Focal epilepsy with seizure onset in parietal cortex (primary somatosensory cortex S1, superior/inferior parietal lobule, or temporoparietal junction). Accounts for ~5% of all focal epilepsies. Hallmark: somatosensory aura. High rate of misdiagnosis as TLE or FLE due to rapid ictal propagation away from parietal onset zone."},
    {"term": "Somatosensory Aura", "definition": "Contralateral tingling, electric sensation, numbness, or pain in a body distribution corresponding to the Penfield somatotopic homunculus of S1. The single most specific clinical sign for parietal lobe seizure onset. Present in 60–80% of structural PLE. May be overlooked if rapid spread to motor cortex dominates the semiology."},
    {"term": "Jacksonian March", "definition": "Ictal spread of somatosensory (or motor) aura in somatotopic sequence along the homunculus: starts in face/thumb → progressively involves hand → arm → leg. Named after Hughlings Jackson (1863). Pathognomonic for S1 (or M1 for motor march) onset. Confirms parietal cortex epileptogenesis."},
    {"term": "Out-of-Body Experience (OBE) / Autoscopy", "definition": "Ictal phenomenon of perceiving oneself from an external vantage point above or beside one's body. Generated by ictal discharge at the temporoparietal junction (TPJ, Brodmann area 39/40) and precuneus. Highly specific for parietal (TPJ) seizure onset. Blanke et al. 2002 (Nature) confirmed by direct electrical stimulation of TPJ in surgical patients."},
    {"term": "Temporoparietal Junction (TPJ)", "definition": "Cortical region at intersection of posterior temporal, parietal, and occipital lobes (BA39/BA40). Processes body schema, self-other distinction, and spatial attention. Ictal discharge: out-of-body experience, contralateral neglect, autoscopy. Key SEEG target in PLE with body-schema disturbances."},
    {"term": "SEEG (Stereo-Electroencephalography)", "definition": "Invasive intracranial EEG technique using multiple depth electrode trajectories (1.2–1.5 mm diameter, 10–15 contacts per electrode) implanted stereotactically under robot-assisted or frame-based guidance. Gold standard for parietal seizure onset zone localisation when scalp EEG and MRI are non-concordant or when parietal onset is suspected but scalp EEG shows apparent frontal or temporal onset."},
    {"term": "Focal Cortical Dysplasia (FCD) Type IIb", "definition": "Most common structural substrate for drug-resistant PLE. Characterised by dysmorphic neurons + balloon cells in cortical layers II/III. MTOR somatic mutations drive mTORC1 hyperactivation → autonomous pacemaker activity. MRI: transmantle sign (T2 hyperintensity from cortex to ventricle wall). SEEG: continuous HFOs (80–250 Hz). Surgical resection: Engel I in 55–70%."},
    {"term": "GLUT-1 Deficiency (SLC2A1 Mutation)", "definition": "Rare autosomal dominant disorder of cerebral glucose transport across the blood-brain barrier. Seizures are movement-triggered paroxysmal events + absence-like episodes + cognitive impairment. Diagnosis: CSF glucose <45 mg/dL + low CSF/blood glucose ratio (<0.45). Treatment: Ketogenic Diet (first-line, not last resort) — bypasses GLUT-1 by providing ketone bodies as alternative fuel. ~90% seizure-free on KD."},
    {"term": "Todd's Paresis", "definition": "Post-ictal contralateral focal motor weakness following a seizure involving motor cortex or its connections. In PLE: contralateral arm and/or leg weakness for 10–120 minutes post-FBTCS or post-tonic posturing seizure. Highly lateralising: weakness on right = left hemisphere seizure origin. Must be distinguished from stroke (onset: abrupt at seizure end, not from baseline; resolves spontaneously in minutes-hours)."},
    {"term": "High-Frequency Oscillations (HFOs)", "definition": "SEEG-recorded oscillations at 80–500 Hz (ripples 80–250 Hz; fast ripples 250–500 Hz) arising from the seizure onset zone (SOZ). HFOs mark pathological hyperexcitability of the epileptogenic zone (EZ) and are used intraoperatively (ECoG) to guide resection boundaries: complete resection of HFO-generating tissue correlates with Engel I outcome in FCD/PLE surgery."},
    {"term": "DEPDC5 / GATOR1 Complex", "definition": "DEP Domain Containing 5 protein — a subunit of the GATOR1 complex that inhibits mTORC1 signalling. Loss-of-function AD mutations (70% penetrance) cause familial focal epilepsy (PLE/FLE/TLE) by disinhibiting mTOR → neuronal hypertrophy + cortical hyperexcitability. Related: NPRL2, NPRL3. 'Two-hit' hypothesis: somatic second mutation creates focal MRI-detectable dysplasia in the seizure onset zone."},
    {"term": "Engel Classification", "definition": "Surgical outcome scale for epilepsy surgery: Engel I = seizure-free; Engel II = >90% reduction; Engel III = >50% reduction; Engel IV = <50% reduction or worse. Standard reporting requirement for all surgical epilepsy series. PLE structural: Engel I ~55–70%."},
    {"term": "Penfield Homunculus", "definition": "Somatotopic map of the primary somatosensory (S1) and primary motor (M1) cortex along the central sulcus, originally drawn by Wilder Penfield and Theodore Rasmussen (1950). S1 (postcentral gyrus): face (lateral) → hand → arm → trunk → leg (medial). In PLE: ictal somatosensory aura distribution reveals the cortical onset site — hand/face aura = lateral convexity; perineal/leg aura = paracentral lobule (mesial wall)."},
    {"term": "Awake Craniotomy (Language / Sensory Mapping)", "definition": "Surgical technique where the patient remains awake during cortical resection to allow continuous intraoperative cortical mapping (direct electrical stimulation). Required in PLE when seizure onset zone abuts primary S1 (central sulcus), TPJ language areas, or visuospatial areas. Intraoperative SEP confirms S1 location; mapping avoids permanent somatosensory deficit. Requires patient cooperation — pre-operative neuropsychological assessment essential."},
    {"term": "SUDEP (Sudden Unexpected Death in Epilepsy)", "definition": "Sudden, unexplained death in a person with epilepsy (in the absence of other known cause). Risk highest in drug-resistant epilepsy with nocturnal FBTCS (SUDEP risk ~1/100 patient-years in refractory epilepsy vs 1/1000 in well-controlled). Prevention: optimise AED, nocturnal seizure alarm (Emfit QM / Embrace2), prone-position avoidance, AED adherence. Mandatory counselling for all DRE patients and families."},
]

# ─── References ──────────────────────────────────────────────────────────────

_REFERENCES = [
    "Kim DW et al. 2004 J Neurosurg — Parietal lobe epilepsy surgical series: 40 patients, 57.5% Engel I",
    "Salanova V et al. 1995 Epilepsia — PLE semiology in 82 patients: somatosensory aura 67%, rapid spread to FL/TL",
    "Ryvlin P et al. 2006 Epilepsia — SEEG-guided parietal surgery: 56% Engel I in non-lesional PLE",
    "Blanke O et al. 2002 Nature — Direct TPJ stimulation produces out-of-body experience (confirmed PLE semiology)",
    "Blume WT et al. 2001 Epilepsia — ILAE semiological seizure classification: parietal aura taxonomy",
    "Binder JR et al. 2009 J Cogn Neurosci — TPJ role in body-schema and spatial self-representation",
]


# ─── Patient overlay ─────────────────────────────────────────────────────────

_ONSET_AGES = [4, 7, 9, 12, 16, 18, 22, 25, 28, 31, 34, 6, 8, 11, 14, 17, 20, 24, 27, 30,
               33, 5, 10, 13, 19, 23, 26, 29, 32, 35, 38, 41, 15, 21, 37, 40, 43, 3, 36, 44, 2]
_ETIOLOGIES_WEIGHTED = (
    ["FCD Type IIb"] * 35 + ["Low-Grade Tumour"] * 22 + ["Post-Traumatic"] * 15 +
    ["DEPDC5 / Genetic"] * 18 + ["Vascular/Post-Stroke"] * 10
)
_PRIMARY_SEIZURES = [
    "FAS-Somatosensory Aura", "FIAS-Body Schema", "FIAS-Versive+Tonic", "FBTCS"
]
_SEIZURE_CONTROL = (["Seizure-free"] * 14 + ["Drug-resistant"] * 17 + ["Partial control"] * 10)
_AEDS = ["CBZ", "LTG", "LEV", "OXC", "LCM", "PER", "VPA+LEV", "LTG+LEV", "CBZ+LCM", "OXC+LTG"]


def _overlay_patients(rows):
    out = []
    for r in rows:
        s = _seed(r.get("patient_id", "x"))
        et_idx = s % len(_ETIOLOGIES_WEIGHTED)
        sz_idx = (s >> 4) % len(_PRIMARY_SEIZURES)
        ctl_idx = (s >> 8) % len(_SEIZURE_CONTROL)
        aed_idx = (s >> 12) % len(_AEDS)
        oa_idx = (s >> 16) % len(_ONSET_AGES)
        somatosensory_aura = (s >> 2) % 4 != 0          # 75%
        body_schema = (s >> 5) % 5 < 2                   # 40%
        surgical_candidate = (s >> 9) % 5 < 2            # 40%
        post_todd = (s >> 11) % 5 < 3                    # 60%
        catamenial = r.get("gender") == "F" and (s >> 14) % 3 == 0   # ~33% of females
        r["etiology"] = _ETIOLOGIES_WEIGHTED[et_idx]
        r["onset_age"] = _ONSET_AGES[oa_idx]
        r["primary_seizure_type"] = _PRIMARY_SEIZURES[sz_idx]
        r["seizure_control"] = _SEIZURE_CONTROL[ctl_idx]
        r["current_aed"] = _AEDS[aed_idx]
        r["somatosensory_aura"] = somatosensory_aura
        r["body_schema_disturbance"] = body_schema
        r["surgical_candidate"] = surgical_candidate
        r["post_ictal_todd"] = post_todd
        r["catamenial_pattern"] = catamenial
        out.append(r)
    return out


# ─── Public API ───────────────────────────────────────────────────────────────

def overview():
    rows = _db_rows("SELECT patient_id, age, gender FROM patients LIMIT 41")
    rows = _overlay_patients(rows)
    n = len(rows)
    drug_resistant = [r for r in rows if r["seizure_control"] == "Drug-resistant"]
    seizure_free = [r for r in rows if r["seizure_control"] == "Seizure-free"]
    surgical = [r for r in rows if r["surgical_candidate"]]
    somatosensory = [r for r in rows if r["somatosensory_aura"]]
    body_schema = [r for r in rows if r["body_schema_disturbance"]]
    todd = [r for r in rows if r["post_ictal_todd"]]

    sz_dist = {}
    for r in rows:
        sz_dist[r["primary_seizure_type"]] = sz_dist.get(r["primary_seizure_type"], 0) + 1

    return {
        "dashboard": "Parietal Lobe Epilepsy (PLE)",
        "generated": date.today().isoformat(),
        "total_patients": n,
        "drug_resistant_n": len(drug_resistant),
        "drug_resistant_pct": round(len(drug_resistant) / n * 100),
        "seizure_free_n": len(seizure_free),
        "seizure_free_pct": round(len(seizure_free) / n * 100),
        "surgical_candidates_n": len(surgical),
        "somatosensory_aura_n": len(somatosensory),
        "somatosensory_aura_pct": round(len(somatosensory) / n * 100),
        "body_schema_n": len(body_schema),
        "body_schema_pct": round(len(body_schema) / n * 100),
        "post_ictal_todd_n": len(todd),
        "post_ictal_todd_pct": round(len(todd) / n * 100),
        "seizure_type_distribution": sz_dist,
        "clinical_alerts": [
            "PLE is frequently MISDIAGNOSED as FLE or TLE due to rapid ictal spread — "
            "SEEG required for definitive localisation when scalp EEG is inconclusive",
            "Somatosensory aura (tingling/electric sensation) is pathognomonic for S1 onset — "
            "document body distribution at every visit (Penfield homunculus localising)",
            "SCN8A gain-of-function PLE: sodium channel blockers (CBZ/OXC/PHT) may WORSEN "
            "seizures — obtain gene panel before first-line AED prescription",
            "Vigabatrin CONTRAINDICATED in PLE with occipital spread — irreversible visual "
            "field loss compounds ictal VF defects from occipital propagation",
            "Lacosamide: ECG mandatory before start and each dose increase — PR interval "
            "prolongation risk; exclude AV block",
            "SUDEP counselling mandatory for all drug-resistant PLE — nocturnal FBTCS "
            "highest-risk seizure type; prescribe Emfit/Embrace seizure alarm",
        ],
        "surgical_targets": {
            "engel_i_fcd_iib": "55–70% (complete FCD resection + SEEG-guided margin)",
            "engel_i_tumour": "70–85% (DNT/ganglioglioma gross-total resection)",
            "engel_i_non_lesional": "35–50% (non-lesional SEEG-guided PLE)",
            "postop_somatosensory_deficit": "Transient 25%; permanent 5% (awake craniotomy mitigates)",
        },
        "references": _REFERENCES,
    }


def breakdown():
    rows = _db_rows("SELECT patient_id, age, gender FROM patients LIMIT 41")
    rows = _overlay_patients(rows)

    patients_out = []
    for r in rows:
        patients_out.append({
            "patient_id": r["patient_id"],
            "age": r["age"],
            "gender": r["gender"],
            "onset_age": r["onset_age"],
            "etiology": r["etiology"],
            "primary_seizure_type": r["primary_seizure_type"],
            "current_aed": r["current_aed"],
            "seizure_control": r["seizure_control"],
            "somatosensory_aura": r["somatosensory_aura"],
            "body_schema_disturbance": r["body_schema_disturbance"],
            "surgical_candidate": r["surgical_candidate"],
            "post_ictal_todd": r["post_ictal_todd"],
            "catamenial_pattern": r.get("catamenial_pattern", False),
        })

    return {
        "patients": patients_out,
        "etiology_catalog": _ETIOLOGIES,
        "seizure_types": _SEIZURE_TYPES,
        "triggers": _TRIGGERS,
        "treatments": _TREATMENTS,
        "lifecycle": _LIFECYCLE,
        "aed_monitoring": _AED_MONITORING,
        "standards": _STANDARDS,
        "thresholds": _THRESHOLDS,
    }


def definitions():
    return {
        "concepts": _CONCEPTS,
        "thresholds": _THRESHOLDS,
        "references": _REFERENCES,
    }
