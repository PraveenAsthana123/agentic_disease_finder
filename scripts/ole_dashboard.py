"""Occipital Lobe Epilepsy (OLE) Dashboard — the third most common focal epilepsy after TLE
and FLE, characterised by prominent visual symptoms, autonomic features, and ictal vomiting.

Hallmarks:
  Visual Aura: elementary phosphenes, ictal amaurosis, scotoma — pathognomonic for occipital onset
  Autonomic Seizures: ictal nausea/vomiting, pallor, diaphoresis — hallmark of Panayiotopoulos Syndrome
  Oculomotor Deviation: tonic contralateral forced eye deviation ± nystagmus, eyelid flutter
  Post-ictal Headache: migrainous headache in 50–60% (visual seizure–migraine overlap — key differential)
  Onset bimodal: childhood (Panayiotopoulos Syndrome, 3–6 years) + adult structural (FCD, vascular, tumor)
  Drug resistance: ~25–40% for structural OLE; Panayiotopoulos Syndrome ~10% (excellent prognosis)
  EEG hallmark: posterior (O1/O2/P3/P4) spike-wave, fixation-off sensitivity (FOS), suppressed by
    eye opening; ictal fast rhythm >13 Hz over occipital channels at seizure onset

FIRST-LINE TREATMENT SELECTION:
  CBZ / OXC: Level A for focal seizures (lobar OLE with clear focal onset + spread)
  LTG: Level B — particularly useful in OLE–migraine overlap (mood-stabilising effect)
  LEV: Level B — broad-spectrum, minimal drug interactions, safe in women of childbearing age
  Lacosamide: Level A add-on (SCN2A gain-of-function subtype, JASPER 2018)
  Perampanel (AMPA antagonist): Level A add-on, REMS-gated, effective for refractory occipital FBTCS
  Surgical Resection: occipital lobectomy / lesionectomy — Engel I outcome 55–65% in structural OLE
  Ketogenic Diet: Level B for refractory / POLG-mutation OLE (caution: contraindicated in POLG if
    valproate co-prescribed — mitochondrial toxicity risk)
  VNS: adjunctive, Engel III–IV when resection not feasible

ABSOLUTE CONTRAINDICATION:
  Valproate + Ketogenic Diet in POLG mutations → acute liver failure / Alpers syndrome

References:
  - Panayiotopoulos CP 1999 Epilepsia (PS — 30–50 cases; autonomic seizures + vomiting + prognosis)
  - Williamson PD et al. 1992 Ann Neurol (OLE surgical series — 25 patients, 56% Engel I)
  - Taylor I et al. 2003 Epilepsia (familial occipital lobe epilepsy — CACNA1A/DEPDC5 genetics)
  - Leutmezer F et al. 2003 Neurology (ictal vomiting localises to non-dominant temporal/occipital)
  - Guerrini R et al. 2010 Epilepsia (occipital FCD type II — surgical and genetic characterisation)
  - Bien CG et al. 2000 Brain (celiac disease + occipital calcifications — gluten-free diet reverses)
Data: live clinical.db (41 epilepsy patients, deterministic OLE overlay)
      + curated OLE pharmacology / etiology / seizure-type / trigger catalogs."""

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
        "etiology": "Panayiotopoulos Syndrome (PS) — Idiopathic / Genetic GGE",
        "category": "Idiopathic/Genetic",
        "pct": 28,
        "mechanism": (
            "Panayiotopoulos Syndrome is an idiopathic focal epilepsy of childhood with strong "
            "genetic predisposition (sibling risk ~10%; concordance with other self-limited "
            "childhood epilepsies like BCECTS). Likely polygenic; DEPDC5, GRIN2A, and PRRT2 "
            "variants reported in familial cases. The core mechanism involves autonomic network "
            "hyperexcitability centred on insular-opercular and occipital cortex: ictal "
            "discharge propagates from occipital cortex → posterior insula → dorsal vagal "
            "complex → brainstem autonomic nuclei → ictal vomiting, pallor, hypersalivation. "
            "Occipital EEG source confirmed by MEG; SEEG rarely needed. Prognosis excellent: "
            "~90% remit within 1–2 years; many require no AED therapy. Age of onset 3–6 years."
        ),
        "eeg_correlate": (
            "High-amplitude (200–400 µV) multifocal or unilateral occipital spike-wave; "
            "fixation-off sensitivity (FOS): spikes increase dramatically on eye closure in "
            "darkness; cloned or shifting foci between recordings; rare occipital fast rhythm "
            "at seizure onset (13 Hz). Interictal EEG may be normal in up to 30%."
        ),
        "mri_finding": (
            "MRI normal in 95%+ of PS (prerequisite for idiopathic classification). Rare "
            "incidental findings (AVM, small FCD) must be excluded. Research MRI shows subtle "
            "occipital cortex volume asymmetry not clinically actionable."
        ),
        "clinical_note": (
            "Key clinical pearl: Panayiotopoulos Syndrome mimics viral gastroenteritis (ictal "
            "vomiting + unresponsiveness). Mean seizure duration 20 min — status epilepticus in "
            "~25% (autonomic status). No in-patient workup required if typical features + "
            "normal MRI + occipital EEG. Reassure family: >90% remission within 2 years."
        ),
    },
    {
        "etiology": "Occipital Focal Cortical Dysplasia (FCD Type IIa/IIb)",
        "category": "Structural",
        "pct": 25,
        "mechanism": (
            "FCD Type II involves focal loss of cortical laminar architecture (balloon cells, "
            "dysmorphic neurons) within occipital cortex — most commonly V1 (calcarine) or "
            "extrastriate visual areas (V2/V3/V4/V5). mTOR pathway gain-of-function somatic "
            "mutations (MTOR, PIK3CA, DEPDC5, TSC1/2) drive abnormal neuronal hypertrophy and "
            "hyperexcitability. FCD IIb (balloon cells) shows characteristic MRI: transmantle "
            "sign (FLAIR hyperintensity tapering from cortex to ventricle), blurring of grey–"
            "white junction, cortical thickening. SEEG localisation + resection achieves "
            "Engel I in 55–70% of FCD II occipital; FCD I more variable (Engel I 40–55%)."
        ),
        "eeg_correlate": (
            "Continuous or near-continuous interictal spiking at FCD site (occipital electrodes "
            "O1/O2, sometimes T5/T6); ictal onset: focal fast activity (>13 Hz β/γ) at FCD "
            "then spread to parietal/temporal; ictal NREM suppression may localise to FCD "
            "quadrant. SEEG grid with depth electrodes in calcarine sulcus required if MRI-"
            "negative or bilateral occipital spikes."
        ),
        "mri_finding": (
            "Transmantle sign on FLAIR (FCD IIb — pathognomonic in ~60%); focal cortical "
            "thickening; grey–white junction blurring on T1; subtle T2 hyperintensity in white "
            "matter; FCD I may be MRI-negative (requires voxel-based morphometry + 7T MRI). "
            "PET: focal occipital hypometabolism at seizure focus (sensitivity 85% in FCD)."
        ),
        "clinical_note": (
            "Surgical planning: fMRI visual mapping essential — distance of FCD from V1 (primary "
            "visual cortex, calcarine sulcus) determines resectability. Wada test or TMS mapping "
            "if dominant hemisphere. If FCD < 1 cm from V1: laser interstitial thermal therapy "
            "(LITT) preferred over open resection to minimise visual field defect. "
            "Post-op visual field loss: contralateral quadrantanopia expected in all; "
            "homonymous hemianopia if extensive resection — counsel pre-operatively."
        ),
    },
    {
        "etiology": "Occipital Calcifications — Celiac Disease + Folic Acid Deficiency",
        "category": "Structural / Metabolic",
        "pct": 12,
        "mechanism": (
            "The Celiac Disease, Epilepsy, and Cerebral Calcifications (CEC) syndrome — also "
            "called Gobbi Syndrome — is characterised by: (1) untreated or late-diagnosed celiac "
            "disease → intestinal malabsorption → severe folate deficiency → impaired "
            "methylation of homocysteine → hyperhomocysteinaemia → endothelial damage + "
            "microangiopathy in occipital cortex; (2) progressive bilateral occipital "
            "calcifications (parieto-occipital watershed zones) resembling Sturge-Weber on CT; "
            "(3) epilepsy — often refractory, starts in childhood-adolescence. Strict gluten-free "
            "diet + folate supplementation in early-diagnosed cases can halt calcification "
            "progression and reduce seizure burden (Bien CG et al. 2000 Brain)."
        ),
        "eeg_correlate": (
            "Bilateral (often asymmetric) occipital spike-wave, occasionally triphasic slow "
            "waves reflecting metabolic encephalopathy component; background slowing if folate "
            "deficiency severe; photosensitive PPR in ~20%. EEG correlates improve on gluten-free "
            "diet + folate repletion — serial EEG monitoring recommended every 6 months."
        ),
        "mri_finding": (
            "CT (NOT MRI): bilateral parieto-occipital cortical calcifications (curvilinear), "
            "often in watershed distribution — pathognomonic for CEC syndrome. MRI may "
            "underestimate calcifications (CT superior for calcium detection). T2/FLAIR: "
            "cortical laminar necrosis pattern in advanced cases. MRI T1 gyriform enhancement "
            "in early inflammatory phase (rare)."
        ),
        "clinical_note": (
            "MANDATORY investigations: anti-tTG IgA + total IgA in ANY child with occipital "
            "epilepsy + occipital calcifications on CT. Small bowel biopsy confirms celiac "
            "disease. Serum folate, homocysteine, vitamin B12. Initiate gluten-free diet + "
            "folic acid 5 mg/day immediately — seizure reduction in 60–70% within 12 months. "
            "AED selection: avoid valproate (hepatotoxic in folate deficiency); prefer LTG or LEV."
        ),
    },
    {
        "etiology": "Genetic — POLG Mutations (Alpers–Huttenlocher / Mitochondrial OLE)",
        "category": "Genetic / Mitochondrial",
        "pct": 10,
        "mechanism": (
            "POLG encodes the catalytic subunit of mitochondrial DNA polymerase γ. Autosomal "
            "recessive POLG mutations (most commonly p.Ala467Thr + p.Trp748Ser compound "
            "heterozygote) cause: (1) mtDNA depletion in neurons → mitochondrial respiratory "
            "chain complex I/IV deficiency → neuronal energy failure; (2) progressive "
            "neuronopathy with predilection for occipital cortex (Alpers syndrome in infants) "
            "and cerebellum (spinocerebellar ataxia in adults — SANDO/MERRF overlap). Seizures "
            "often severe, refractory, multifocal with occipital predominance. Lancinating "
            "occipital pain + visual loss + status epilepticus hallmark of acute Alpers phase. "
            "FATAL if valproate administered: acute liver failure within weeks → Alpers crisis."
        ),
        "eeg_correlate": (
            "High-amplitude slow waves (posterior dominant); rhythmic high-amplitude delta "
            "activity (RHDA) — characteristic 1–2 Hz rhythmic occipital delta coinciding with "
            "visual loss; occipital epileptiform discharges (OED) as seizure precursor; "
            "evolving to status epilepticus pattern. Background becomes diffusely slow and "
            "disorganised as disease progresses — poor prognostic sign."
        ),
        "mri_finding": (
            "MRI: occipital cortical signal change on FLAIR/DWI (restricted diffusion — "
            "cytotoxic oedema in acute phase); cortical laminar necrosis on T1-gadolinium; "
            "cerebellar atrophy in long-standing cases; basal ganglia involvement rare. "
            "MRS: elevated lactate peak (↑ choline, ↑ myo-inositol, ↓ NAA) — metabolic "
            "signature of mitochondrial disorder. Serial MRI shows progressive occipital "
            "cortical atrophy — correlates with visual field loss."
        ),
        "clinical_note": (
            "ABSOLUTE CONTRAINDICATION: Valproate → acute hepatic failure in POLG. Screen "
            "POLG in any child with refractory occipital epilepsy + liver dysfunction + family "
            "history of mitochondrial disease. Alternatives: LEV, LTG (avoid enzyme inducers "
            "that increase mtDNA stress). Ketogenic diet: theoretically beneficial (supports "
            "mitochondrial energy via ketones) but evidence limited; avoid if VPA co-prescribed. "
            "Genetic counselling: carrier parents each 50% risk; prenatal testing available."
        ),
    },
    {
        "etiology": "Structural — Occipital Lobe Tumor / Cavernoma / AVM",
        "category": "Structural / Lesional",
        "pct": 15,
        "mechanism": (
            "Space-occupying lesions (low-grade glioma WHO Grade I–II, cavernous malformation, "
            "arteriovenous malformation, oligodendroglioma) in the occipital lobe cause "
            "epilepsy via: (1) perilesional irritation — iron deposition (cavernoma haemosiderin), "
            "tumour-secreted glutamate, cytokine release → hyperexcitability of surrounding "
            "occipital cortex; (2) compression of V1/V2 → visual field defect; (3) venous "
            "hypertension (AVM) → cortical spreading depolarisation → seizure–migraine overlap. "
            "Low-grade gliomas (LGG) carry IDH1/IDH2 mutation + 1p/19q co-deletion (oligodendro) "
            "or ATRX loss (astrocytoma); BRAF-KIAA1549 fusion in occipital ganglioglioma."
        ),
        "eeg_correlate": (
            "Focal occipital interictal spikes/sharp waves adjacent to lesion (perilesional zone); "
            "ictal onset at lesion margin; rhythmic beta/gamma discharge propagating to parietal "
            "and temporal association cortex. EEG lateralises in 80%; localises to occipital "
            "region in 60% (remainder temporal or parietal spread mimics TLE). "
            "Cavernoma: may have burst-suppression in perihematomal zone after acute bleed."
        ),
        "mri_finding": (
            "Glioma: T2/FLAIR hyperintensity (non-enhancing LGG) or ring enhancement (HGG). "
            "Ganglioglioma: cystic + mural nodule, calcification in 40%. "
            "Cavernoma: T2* / SWI bloom ('popcorn' appearance, haemosiderin ring), no mass effect. "
            "AVM: flow-voids on T2; confirmed by DSA (digital subtraction angiography) — essential "
            "for Spetzler-Martin grading and radiosurgery planning. All lesional OLE: "
            "fMRI visual cortex mapping + DTI occipital radiation tractography before any resection."
        ),
        "clinical_note": (
            "Surgical priority: lesionectomy + margin resection in cavernoma/LGG — 65–80% "
            "Engel I if complete resection. AVM: stereotactic radiosurgery (Gamma Knife) for "
            "Spetzler-Martin Grade I–II; embolisation + surgery for Grade III. LGG: chemotherapy "
            "(PCV or temozolomide) + radiotherapy per EORTC protocols if progression. "
            "Visual field surveillance: Humphrey automated perimetry every 6 months "
            "to document progression vs post-surgical quadrantanopia."
        ),
    },
    {
        "etiology": "Genetic — CACNA1A / Familial Occipital Lobe Epilepsy (FOLE)",
        "category": "Genetic",
        "pct": 10,
        "mechanism": (
            "Familial Occipital Lobe Epilepsy (FOLE) is an autosomal dominant focal epilepsy "
            "with visual seizures, first characterised by Taylor et al. (2003 Epilepsia) in 10 "
            "Australian families. CACNA1A gain-of-function variants (Cav2.1 P/Q-type calcium "
            "channel — the same gene as familial hemiplegic migraine type 1, FHM1) and DEPDC5 "
            "loss-of-function variants (mTORC1 inhibitor) are the most common genetic causes. "
            "DEPDC5: autosomal dominant; seizure onset in adolescence–adulthood; MRI often "
            "normal; focal occipital cortical irritability without structural lesion. CACNA1A "
            "variants: epistatic interaction between migraine networks and occipital seizure "
            "threshold — 30–40% of FOLE patients have comorbid migraine with aura."
        ),
        "eeg_correlate": (
            "Occipital spikes / spike-wave (unilateral or bilateral); interictal spikes "
            "normalise between clusters; ictal fast occipital discharge (13–25 Hz) at onset; "
            "may show fixation-off sensitivity (FOS) in CACNA1A cases. Genetic OLE: EEG "
            "may appear focal-temporal on surface EEG if rapid propagation — SEEG clarifies "
            "true occipital onset in 85% of DEPDC5-positive cases."
        ),
        "mri_finding": (
            "MRI normal in 70% of FOLE (prerequisite to confirm idiopathic). "
            "DEPDC5 with FCD (focal cortical dysplasia): subtle FLAIR signal + blurring — "
            "detected in 30% only on 3T MRI with specific FCD protocol (MPR + FLAIR 3D). "
            "7T MRI + voxel-based morphometry increases FCD detection to 55% in MRI-negative "
            "DEPDC5-positive FOLE cases (research protocol, not clinical standard)."
        ),
        "clinical_note": (
            "FOLE genetic panel: DEPDC5, NPRL2, NPRL3 (GATOR1 complex), CACNA1A, PRRT2 "
            "(paroxysmal disorder gene). CACNA1A variant overlap with hemiplegic migraine: "
            "visual aura + severe headache after seizure may be difficult to distinguish from "
            "FHM1 attack — ictal EEG recording during episode is diagnostic. "
            "LTG preferred (anticonvulsant + antimigraine), avoid sodium channel blockers if "
            "CACNA1A variant (CBZ rarely aggravates; LTG safe). Genetic counselling: AD "
            "inheritance; offspring risk 50%; penetrance 60–80% for DEPDC5."
        ),
    },
]


# ─── Seizure types ──────────────────────────────────────────────────────────

_SEIZURE_TYPES = [
    {
        "type": "Focal Aware Seizure (FAS) — Elementary Visual Aura",
        "freq_pct": 75,
        "duration_sec": "10–90 s (brief; aura may precede FIAS or FBTCS)",
        "description": (
            "The pathognomonic symptom of OLE: patient experiences elementary (unformed) visual "
            "phenomena — coloured phosphenes (flashing lights), geometrical patterns (circles, "
            "arcs, fortification spectra resembling migraine aura but much briefer), positive "
            "scotoma (bright patch), negative scotoma (blind spot in visual field), or complete "
            "ictal amaurosis (transient blindness — pathognomonic for V1 involvement). "
            "The visual phenomena appear in the contralateral visual field (lesion hemisphere → "
            "contralateral half-field aura). Consciousness fully preserved."
        ),
        "eeg_correlate": (
            "Focal fast activity (β/γ 13–30 Hz) over occipital channels (O1 or O2) corresponding "
            "to contralateral hemisphere. Low amplitude at seizure onset — may be missed on scalp "
            "EEG if deep FCD. No post-ictal slowing for brief FAS (< 30 seconds). "
            "Temporal propagation within 10–40 s if FIAS follows."
        ),
        "clinical_tip": (
            "Distinguish from migraine aura: seizure aura lasts < 3 minutes (usually < 60 s), "
            "migraine aura 20–60 minutes. Seizure phosphenes move/expand rapidly; migraine "
            "fortification spectrum moves slowly across visual field. Co-occurrence of both "
            "disorders in CACNA1A variants — seizure diary with aura timing is diagnostic."
        ),
    },
    {
        "type": "Focal Impaired Awareness Seizure (FIAS) — Oculomotor + Autonomic",
        "freq_pct": 55,
        "duration_sec": "30 s – 5 min (prolonged in Panayiotopoulos Syndrome)",
        "description": (
            "Seizure begins with visual aura then progresses to impaired awareness. Two distinct "
            "semiology patterns: (1) OCULOMOTOR: tonic contralateral forced eye and head deviation "
            "('versive seizure') + clonic eyelid flutter ± nystagmus-like eye movements. Patient "
            "appears to stare towards the hemisphere contralateral to seizure focus. Highly "
            "lateralising — ipsilateral to focus in ~20% (misleading). (2) AUTONOMIC: ictal "
            "pallor, nausea, retching, vomiting — hallmark of Panayiotopoulos Syndrome. Ictal "
            "vomiting + unresponsiveness in a child often misdiagnosed as brainstem TIA, "
            "encephalitis, or cyclic vomiting syndrome. Autonomic status epilepticus: prolonged "
            "(20–30 min) autonomic seizure without convulsion — management per SE protocol."
        ),
        "eeg_correlate": (
            "Occipital ictal onset propagating to posterior temporal (Panayiotopoulos) or "
            "frontal (versive/oculomotor). Temporal channels (T5/T6) involved when ictal "
            "vomiting present — posterior insular source confirmed by SEEG. Bilateral "
            "synchrony may develop by 30 seconds in Panayiotopoulos Syndrome (autonomic SE)."
        ),
        "clinical_tip": (
            "Panayiotopoulos Syndrome: seizure during sleep in 75% — parents observe vomiting "
            "child + unresponsiveness → emergency department admission. Key teaching: any child "
            "with prolonged (> 5 min) vomiting + unresponsiveness = consider autonomic SE. "
            "Buccal midazolam 0.3 mg/kg at 5 minutes. Inpatient video-EEG during episode is "
            "diagnostic. EEG normal interictal in 30% — do not exclude PS on normal EEG."
        ),
    },
    {
        "type": "Focal to Bilateral Tonic-Clonic Seizure (FBTCS) — Secondary Generalisation",
        "freq_pct": 45,
        "duration_sec": "60–180 s (FBTCS phase)",
        "description": (
            "OLE seizures can propagate via: occipital → parietal → frontal lobe (dorsal stream "
            "propagation) → bilateral motor cortex → FBTCS. Alternatively: occipital → temporal "
            "→ mesial temporal (Papez circuit) → limbic propagation mimicking TLE FBTCS. "
            "The visual aura preceding FBTCS in OLE can be extremely brief (2–5 seconds) and "
            "easily overlooked — patient may report only 'lights then blackout.' "
            "Post-ictal headache in 50–60% — migrainous quality (throbbing, unilateral, nausea) "
            "lasting hours: this post-ictal headache is far more common in OLE than in TLE or FLE, "
            "and represents occipital cortex post-ictal vasodilation / CSD (cortical spreading "
            "depolarisation) — important OLE clinical signature."
        ),
        "eeg_correlate": (
            "Occipital fast onset then rapid bilateral synchrony; FBTCS: generalised tonic "
            "activity (muscle artefact obscures EEG during tonic phase); clonic phase: "
            "repetitive spike-wave at 2–3 Hz. Post-ictal: FIRDA (frontal intermittent rhythmic "
            "delta activity) or diffuse suppression. Occipital onset localised only retrospectively "
            "by reviewing pre-FBTCS EEG for initial focal fast activity."
        ),
        "clinical_tip": (
            "AED selection for OLE + FBTCS: CBZ/OXC reduce FBTCS but occasionally aggravate "
            "occipital focal discharge (rare CBZ-worsening in photosensitive OLE). LTG dual "
            "benefit: anticonvulsant + post-ictal headache prevention. LEV: well-tolerated, "
            "effective for focal-to-bilateral; monitor mood/behaviour (PHQ-9 every 3 months). "
            "SUDEP risk: FBTCS frequency correlates with SUDEP — target FBTCS = 0."
        ),
    },
    {
        "type": "Complex Visual Hallucinations (CVH) — Extrastriate / Temporal Spread",
        "freq_pct": 30,
        "duration_sec": "30 s – 3 min (hallucination phase)",
        "description": (
            "When OLE seizure discharge propagates from primary visual cortex (V1) to extrastriate "
            "areas (V4 — colour/form, V5 — motion, fusiform — face recognition) or to posterior "
            "temporal / inferior parietal cortex, patients experience COMPLEX visual hallucinations: "
            "formed objects, people, animals, scenes (in contrast to elementary phosphenes of V1 "
            "activation). Ictal prosopagnosia (face recognition failure) if fusiform gyrus involved; "
            "ictal achromatopsia (colour blindness) if V4; ictal akinetopsia (motion blindness) "
            "if V5. Complex hallucinations may cause patient or physician to suspect psychosis — "
            "important to differentiate from psychogenic (PNES) or psychiatric visual hallucinations. "
            "Awareness preserved in ~40% of CVH seizures — patient can describe hallucination in "
            "real-time (extremely helpful for localisation)."
        ),
        "eeg_correlate": (
            "Extrastriate ictal discharge (T5/P3 or T6/P4 channels predominant for CVH); "
            "may appear falsely temporal on scalp EEG. SEEG confirms posterior temporal / "
            "occipito-temporal source. Fusiform gyrus involvement: rhythmic 5–8 Hz theta "
            "in inferior temporal contacts. Complex hallucination onset corresponds to "
            "spread from O1/O2 to T5/T6 within 5–15 seconds."
        ),
        "clinical_tip": (
            "CVH differential: occipital epilepsy vs peduncular hallucinosis (brainstem) vs "
            "Charles Bonnet syndrome (visual loss) vs Lewy body dementia vs drug-induced. "
            "Seizure CVH: stereotyped, brief, associated with other ictal features (eye deviation, "
            "post-ictal headache). Psychiatric hallucinations: non-stereotyped, variable, "
            "often auditory+visual, no post-ictal phase. If CVH isolated with no EEG correlate: "
            "consider ictal SPECT during event before psychiatric referral."
        ),
    },
]


# ─── Triggers ────────────────────────────────────────────────────────────────

_TRIGGERS = [
    {
        "trigger": "Photosensitivity / Flickering Light",
        "pct": 30,
        "mechanism": (
            "Occipital cortex is the primary generator of photoparoxysmal response (PPR). "
            "Pattern visual stimuli (stripes, checkerboards, video games) activate V1/V2 → "
            "abnormal synchronisation → occipital discharge → seizure. Photosensitivity "
            "prevalence in OLE (~30%) is higher than in TLE/FLE (~5%) but lower than JME (~35%). "
            "Fixation-off sensitivity (FOS) is a distinct OLE phenomenon: eyes open suppresses "
            "spikes, eyes closed in darkness induces spikes — indicates occipital source."
        ),
        "management": (
            "Blue-light blocking lenses (FL-41 tint) reduce photosensitive seizure frequency "
            "30–50%. Avoid flicker > 3 Hz environments (discos, strobe, TV < 0.5 m). "
            "VPA or LEV most effective pharmacologically. Avoid OXC (may worsen PPR in some). "
            "Fixation-off sensitive patients: wear corrective lenses at all times (removes FOS)."
        ),
    },
    {
        "trigger": "Eye Closure / Darkness",
        "pct": 25,
        "mechanism": (
            "Fixation-off sensitivity (FOS): EEG spikes emerge within 1–3 seconds of eye "
            "closure in darkness and suppress immediately on eye opening. Distinct from "
            "photosensitivity. Mechanism: removal of steady visual input removes inhibitory "
            "surround suppression from striate cortex → release of interictal discharge. "
            "Not caused by sleep — FOS is present in wakefulness with eyes closed."
        ),
        "management": (
            "FOS is not a modifiable trigger per se but a biomarker of occipital focus. "
            "EEG technician pearl: always test FOS during EEG recording — ask patient to "
            "close eyes in a darkened room; compare eye-open vs eye-closed spike frequency. "
            "FOS disappears in ~70% of cases after successful AED therapy or surgical resection."
        ),
    },
    {
        "trigger": "Sleep Deprivation",
        "pct": 55,
        "mechanism": (
            "Sleep deprivation lowers seizure threshold universally but is especially potent in "
            "childhood OLE (Panayiotopoulos Syndrome: 75% of seizures occur during sleep or "
            "on awakening — NREM Stage II/III. Occipital cortex shows heightened excitability "
            "in NREM sleep (spindles → K-complexes → occipital discharge). Sleep deprivation "
            "shifts NREM–REM balance, increases adenosine/slow-wave drive → paradoxically "
            "more NREM oscillations → more occipital discharges."
        ),
        "management": (
            "Regular 7–9 h sleep (adults), 9–11 h (school-age children). Sleep diary for 2 "
            "weeks to document pattern. Melatonin 0.5–3 mg if insomnia component (not sedating). "
            "Avoid caffeine after 14:00. Seizure diary: record whether seizure occurred "
            "within 2 hours of waking — confirms sleep-related OLE pattern for DVLA/driving advice."
        ),
    },
    {
        "trigger": "Missed AED Dose",
        "pct": 60,
        "mechanism": (
            "Subtherapeutic AED level → rapid loss of sodium/calcium channel blockade → rebound "
            "neuronal hyperexcitability. For CBZ (t½ 12–17 h) and OXC (t½ 9–11 h, active "
            "metabolite MHD t½ 9 h): a single missed dose drops levels by 30–50% within "
            "12–16 hours → breakthrough seizure within 24 h. LTG (t½ 25 h): more forgiving; "
            "missed dose less likely to cause immediate breakthrough. TDM (therapeutic drug "
            "monitoring) should be checked after every breakthrough seizure."
        ),
        "management": (
            "Pill box organiser + phone alarm. If dose missed < 6 h: take immediately. "
            "If > 6 h: skip dose, resume next scheduled dose (do NOT double up — toxicity). "
            "TDM: CBZ target 4–12 mg/L, OXC MHD 12–35 mg/L, LTG 3–15 mg/L. "
            "Check TDM morning trough (before next dose) to confirm adherence."
        ),
    },
    {
        "trigger": "Stress / Emotional Arousal",
        "pct": 45,
        "mechanism": (
            "Cortisol/norepinephrine surge during acute stress → reduce GABAergic inhibition "
            "in occipital cortex via glucocorticoid receptor-mediated interneuron suppression. "
            "Chronic stress: sleep fragmentation → further seizure facilitation. "
            "Emotional arousal (excitement, anxiety, anger) reduces seizure threshold in "
            "idiopathic focal epilepsy more than in structural — Panayiotopoulos Syndrome "
            "seizures commonly precipitated by febrile illness + emotional distress."
        ),
        "management": (
            "CBT / mindfulness-based seizure management (MBSM) — 30% seizure reduction in RCT "
            "(Tang et al. 2015 Epilepsy Behav). Biofeedback (HRV-biofeedback): emerging "
            "evidence. Avoid precipitating situations where possible. Psychiatry co-management "
            "if anxiety disorder comorbid (GAD-7 screening at every clinic visit)."
        ),
    },
    {
        "trigger": "Fever / Illness (Panayiotopoulos Syndrome specific)",
        "pct": 40,
        "mechanism": (
            "Febrile provocation particularly prominent in Panayiotopoulos Syndrome: fever "
            "lowers seizure threshold by direct neuronal temperature effects on voltage-gated "
            "Na⁺ channels (Nav1.1/Nav1.2 — accelerated inactivation at 39°C → paradoxical "
            "hyperexcitability in GABA interneurons). Prolonged autonomic status epilepticus "
            "is the characteristic manifestation — 25% of all PS seizures are autonomic SE "
            "precipitated by febrile illness."
        ),
        "management": (
            "Rescue medication: buccal midazolam 0.3 mg/kg (school nurses trained) or rectal "
            "diazepam 0.5 mg/kg if > 5 min of autonomic SE symptoms. Parents: written seizure "
            "action plan. Antipyretics (paracetamol/ibuprofen) at first sign of fever. "
            "Hospital escalation: if autonomic SE > 10 min → emergency department for IV "
            "lorazepam 0.1 mg/kg. Inform school: ictal vomiting in class is a seizure, not food poisoning."
        ),
    },
    {
        "trigger": "Reading / Sustained Visual Concentration",
        "pct": 15,
        "mechanism": (
            "Sustained focused visual tasks (reading, screen work, close work) engage V1/V2/V4 "
            "→ increased cortical excitability in occipital/parietal network → facilitation "
            "of occipital discharge in susceptible individuals. More prominent in FCD and "
            "idiopathic OLE than in structural/lesional OLE. Classified as 'reflex epilepsy' "
            "if reproducible and consistent — rare (< 2% of OLE cases)."
        ),
        "management": (
            "20-20-20 rule: every 20 min, look at object 20 ft away for 20 s. "
            "Blue-light filter glasses (FL-41). Screen colour temperature (night mode) "
            "reduces flicker. Structured work schedule with mandatory breaks. "
            "Avoid reading in poor lighting. Anti-glare screen coating. "
            "If reflex reading epilepsy confirmed: specialist driving/occupation advice."
        ),
    },
    {
        "trigger": "Catamenial Pattern (Women of Childbearing Age)",
        "pct": 20,
        "mechanism": (
            "Catamenial OLE pattern (C1: perimenstrual / C2: periovulatory) driven by oestrogen-"
            "mediated increase in occipital cortex excitability and progesterone withdrawal "
            "reducing allopregnanolone (positive GABA-A modulator) — same mechanism as "
            "catamenial epilepsy in other lobar epilepsies. OLE-specific: the photosensitive "
            "threshold also varies with hormonal cycle — higher PPR in follicular phase "
            "(higher oestrogen:progesterone ratio)."
        ),
        "management": (
            "Seizure diary for 3 menstrual cycles to confirm catamenial pattern (>2× seizures "
            "peri-menstrually). Clobazam 10–20 mg/day on days -4 to +4 (perimenstrual). "
            "Progesterone supplementation (Prometrium 200 mg PO TID luteal phase) — evidence "
            "Level B (Herzog et al. 2012 Epilepsia). Avoid oral contraceptives containing "
            "enzyme-inducing AEDs (CBZ/OXC reduce OCP efficacy — use barrier method or "
            "levonorgestrel IUD)."
        ),
    },
]


# ─── Treatments ──────────────────────────────────────────────────────────────

_TREATMENTS = [
    {
        "drug": "Carbamazepine (CBZ)",
        "class": "Sodium channel blocker",
        "dose_adult": "400–1600 mg/day (divided BID-TID; IR or CR formulation)",
        "dose_paed": "10–20 mg/kg/day (children)",
        "moa": (
            "Blocks use-dependent voltage-gated Na⁺ channels (Nav1.1/Nav1.2/Nav1.6) — "
            "reduces sustained rapid neuronal firing (occipital ictal discharge). "
            "Does NOT block low-threshold T-type Ca²⁺ channels → no absence seizure coverage "
            "(safe in OLE; OLE has no typical absence seizures)."
        ),
        "efficacy": "Level A focal epilepsy; ~50% seizure freedom at 12 months in lobar OLE",
        "evidence": "SANAD I (Marson 2007 Lancet) — CBZ best overall for focal seizures",
        "safety": (
            "Enzyme inducer (CYP3A4 ↑ → reduces OCP efficacy → unintended pregnancy risk). "
            "Hyponatraemia (SIADH) — monitor serum sodium. "
            "HLA-B*1502 screening MANDATORY in Han Chinese/South Asian populations → "
            "SJS/TEN risk 5–10× higher if HLA-B*1502 positive. "
            "Agranulocytosis: baseline CBC + repeat at 3 months. "
            "Cognitive: sedation/diplopia/ataxia at higher doses — use CR formulation."
        ),
        "monitoring": "TDM: 4–12 mg/L (trough); LFT + CBC at 3 months; serum Na+ every 6 months",
        "contraindications": "HLA-B*1502 positive; AV block; porphyria; MAOIs",
    },
    {
        "drug": "Oxcarbazepine (OXC)",
        "class": "Sodium channel blocker (CBZ analogue, active metabolite MHD)",
        "dose_adult": "600–2400 mg/day BID",
        "dose_paed": "8–30 mg/kg/day",
        "moa": (
            "Prodrug: rapidly converted to 10-monohydroxy metabolite (MHD) which blocks Na⁺ "
            "channels. Fewer drug-drug interactions than CBZ (less CYP3A4 induction). "
            "Less auto-induction (stable dosing). Hyponatraemia more common than CBZ (15–25%)."
        ),
        "efficacy": "Level A focal epilepsy; non-inferior to CBZ in focal onset seizures (SANAD I)",
        "evidence": "SANAD I 2007 Lancet — OXC equivalent to CBZ for focal seizures",
        "safety": (
            "Hyponatraemia SIADH — monitor sodium monthly for first 3 months. "
            "Cross-hypersensitivity with CBZ in 25–30% (SJS risk — HLA-B*1502 screening still warranted). "
            "Mild enzyme inducer (less than CBZ) — OCP interaction exists; advise additional contraception. "
            "Dizziness/headache common at initiation — slow titration (150 mg/week increments)."
        ),
        "monitoring": "MHD TDM: 12–35 mg/L; serum Na+ monthly (first 3 months) then every 6 months",
        "contraindications": "HLA-B*1502 (relative); severe hyponatraemia; hypersensitivity to CBZ",
    },
    {
        "drug": "Lamotrigine (LTG)",
        "class": "Sodium channel blocker + glutamate release inhibition",
        "dose_adult": "100–400 mg/day BID (monotherapy); 50–200 mg/day if on VPA",
        "dose_paed": "1–5 mg/kg/day (monotherapy); 0.5–2.5 mg/kg/day with VPA",
        "moa": (
            "Blocks Na⁺ channels (use-dependent, similar to CBZ) + inhibits glutamate release "
            "by blocking presynaptic Ca²⁺-dependent glutamate vesicle exocytosis. "
            "Additionally stabilises neuronal membranes in migraine networks — dual benefit "
            "for OLE patients with comorbid migraine (particularly CACNA1A-OLE)."
        ),
        "efficacy": "Level A focal epilepsy; Level B OLE-migraine overlap; 45–55% seizure freedom",
        "evidence": "SANAD I (Marson 2007 Lancet) — LTG best tolerated; comparable long-term outcomes",
        "safety": (
            "Stevens-Johnson Syndrome (SJS) risk: 1:3000 (higher in children, higher with VPA "
            "co-prescription, higher with rapid titration). MANDATORY slow titration: "
            "+25 mg/week (monotherapy) or +12.5 mg/week (with VPA). "
            "With VPA: VPA doubles LTG levels → start at 25 mg every OTHER day. "
            "Without VPA/enzyme inducer: standard slow titration. "
            "Psoriasis exacerbation (rare). Rash: stop IMMEDIATELY if any rash develops."
        ),
        "monitoring": "TDM: 3–15 mg/L (trough); slow titration diary; rash surveillance; PHQ-9",
        "contraindications": "Rapid titration (SJS risk); caution in hepatic impairment",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "class": "SV2A ligand (synaptic vesicle glycoprotein 2A)",
        "dose_adult": "1000–3000 mg/day BID",
        "dose_paed": "20–60 mg/kg/day BID",
        "moa": (
            "Binds SV2A — reduces neurotransmitter (glutamate + GABA) vesicle release from "
            "presynaptic terminals by impairing Ca²⁺-dependent vesicle fusion. Broad-spectrum: "
            "effective in both focal and generalised seizures. No effect on Na⁺/Ca²⁺ channels "
            "(different MOA from CBZ/LTG). Minimal drug interactions — safe in women on OCP "
            "(no enzyme induction)."
        ),
        "efficacy": "Level B focal epilepsy; excellent in paediatric OLE (Panayiotopoulos Syndrome)",
        "evidence": "SANAD II (Marson 2021 Lancet) — LEV cost-effective for focal; EMA approved",
        "safety": (
            "Neuropsychiatric: irritability, aggression, depression in 15–25% — "
            "MANDATORY PHQ-9 + GAD-7 at every visit. Consider dose reduction if mood symptoms. "
            "Pyridoxine (B6) 100 mg/day: reduces LEV-induced irritability (small RCT evidence). "
            "Fatigue/somnolence common at initiation. Renal dose adjustment: eCrCl < 50 → "
            "reduce dose 50%. Teratogenicity: low (NEAD registry — no major malformation increase)."
        ),
        "monitoring": "PHQ-9 + GAD-7 every 3 months; renal function (eCrCl) annually",
        "contraindications": "Severe renal failure (dose adjust); caution if psychiatric history",
    },
    {
        "drug": "Lacosamide (LCM)",
        "class": "Slow inactivation Na⁺ channel enhancer (selective)",
        "dose_adult": "100–400 mg/day BID (add-on to focal epilepsy AED)",
        "dose_paed": ">4 years: 2–8 mg/kg/day BID (approved 2019)",
        "moa": (
            "Selectively enhances slow inactivation of voltage-gated Na⁺ channels (Nav1.3/1.7) "
            "without effect on fast inactivation — complementary to CBZ/OXC which block fast "
            "inactivation. Also collapsin response mediator protein-2 (CRMP-2) binding → "
            "reduces axonal excitability. Particularly effective as add-on when focal OLE "
            "partially controlled on CBZ/OXC monotherapy."
        ),
        "efficacy": "Level A add-on focal epilepsy; ~30–40% ≥50% seizure reduction as adjunct",
        "evidence": "JASPER study (Sake 2010 Epilepsia); FDA approved 2008; EMA 2008",
        "safety": (
            "CARDIAC: PR interval prolongation (mandatory baseline ECG + repeat at 3 months) — "
            "avoid if 2nd/3rd degree AV block or sick sinus syndrome. "
            "Dizziness/diplopia/ataxia at initiation (dose-dependent, resolves). "
            "IV formulation available (switch 1:1) — useful for status epilepticus or NPO patients. "
            "No enzyme induction — safe with OCP."
        ),
        "monitoring": "ECG at baseline + 3 months (PR interval); TDM not routine (no established target)",
        "contraindications": "2nd/3rd degree AV block; severe cardiac conduction disease",
    },
    {
        "drug": "Perampanel (PER)",
        "class": "AMPA receptor antagonist (non-competitive)",
        "dose_adult": "2–12 mg/day once nightly (REMS-mandated counselling)",
        "dose_paed": ">4 years (≥20 kg): 2–12 mg/day nightly",
        "moa": (
            "First-in-class non-competitive AMPA (α-amino-3-hydroxy-5-methyl-4-isoxazolepropionic "
            "acid) receptor antagonist — blocks fast excitatory neurotransmission (glutamate → "
            "AMPA receptor → Na⁺ influx) at post-synaptic level. Effective against secondarily "
            "generalised seizures (FBTCS) — particularly useful in OLE with frequent FBTCS. "
            "Long half-life (105 h) — once-nightly dosing, forgiving if occasional dose missed."
        ),
        "efficacy": "Level A add-on for focal + FBTCS; 43–67% ≥50% reduction in FBTCS (Study 304/306)",
        "evidence": "FAME studies (French 2012 Epilepsia); FDA REMS 2012; EMA 2012",
        "safety": (
            "FDA REMS: aggression, hostility, irritability, suicidal ideation — MANDATORY "
            "psychiatric screening (PHQ-9 + aggression questionnaire) before initiation. "
            "Dizziness: most common adverse effect (30–40%); use nightly dosing to minimise. "
            "Weight gain: ~2–3 kg in long-term use. "
            "Enzyme inducers (CBZ/OXC) reduce PER levels by 50% — may need 4–12 mg "
            "instead of standard 4–8 mg if co-prescribed."
        ),
        "monitoring": "PHQ-9 + aggression screen every visit; weight monthly; REMS enrolment",
        "contraindications": "Suicidal ideation (relative); psychiatric instability without monitoring",
    },
    {
        "drug": "Occipital Lobectomy / Lesionectomy",
        "class": "Surgical — curative intent",
        "dose_adult": "N/A — surgical procedure; pre-op: Phase I (scalp EEG + MRI) → Phase II SEEG if needed",
        "dose_paed": "Feasible from infancy (cavernoma, FCD); visual cortex mapping limits in young children",
        "moa": (
            "Resection of epileptogenic zone (EZ) in occipital lobe — removes structural source "
            "(FCD, cavernoma, LGG) or functional EZ (electroclinically mapped). Engel I (seizure "
            "free) outcomes depend on: (a) completeness of EZ resection (MRI-lesional > MRI-negative), "
            "(b) proximity to primary visual cortex V1 (calcarine sulcus), (c) etiology "
            "(FCD II > FCD I > cryptogenic). Wada test / fMRI visual mapping + DTI of optic "
            "radiation essential for resection planning."
        ),
        "efficacy": "Level B — Engel I: 55–65% (FCD II); 40–55% (FCD I); 65–80% (cavernoma/LGG)",
        "evidence": (
            "Williamson PD 1992 Ann Neurol (OLE surgical series — 25 patients, 56% Engel I); "
            "Jeha LE 2009 Epilepsia (Cleveland Clinic OLE series — 30 patients, 63% Engel I/II)"
        ),
        "safety": (
            "EXPECTED: contralateral homonymous quadrantanopia (if V1 or optic radiation involved) "
            "— counsel pre-operatively; mandatory Humphrey visual field pre/post-op. "
            "Complete hemianopia: if resection extends to optic radiation → DVLA implications "
            "(cannot drive with hemianopia). "
            "Cognitive: usually minimal (occipital lobe not language-dominant, limited memory role). "
            "RARE: parietal spread → somatosensory deficit; temporal spread → naming deficit (dominant)."
        ),
        "monitoring": "Post-op MRI at 3 months + annual; Humphrey VF every 6 months; EEG at 1, 6, 12 months",
        "contraindications": "Bilateral EZ without resectable lesion; unacceptable visual field loss risk",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "class": "Dietary therapy — metabolic anticonvulsant",
        "dose_adult": "4:1 fat:carbohydrate+protein ratio; ketone target β-OHB 3–6 mmol/L",
        "dose_paed": "Classic 4:1 KD or modified Atkins diet (MAD); dietitian-supervised",
        "moa": (
            "High-fat, low-carbohydrate diet → hepatic ketogenesis → blood ketones (β-hydroxybutyrate, "
            "acetoacetate) → cross blood-brain barrier → alternative neuronal fuel → reduces "
            "glycolysis-dependent excitatory neurotransmission. Additional mechanisms: "
            "KATP channel opening (ATP ↓ → K⁺ efflux → hyperpolarisation), "
            "HDAC inhibition (epigenetic anti-inflammatory), mTOR pathway suppression. "
            "Particularly beneficial in POLG-mutation OLE and mitochondrial epilepsies "
            "(supports oxidative phosphorylation via ketone substrate)."
        ),
        "efficacy": "Level B refractory focal epilepsy; ~50% achieve ≥50% seizure reduction; ~15% seizure free",
        "evidence": "Thiele EA 2017 NEJM (TSC-KD); Neal EG 2008 Lancet (paediatric KD RCT)",
        "safety": (
            "CONTRAINDICATED in POLG mutations if valproate co-prescribed — acute liver failure. "
            "Hyperlipidaemia: LDL monitoring every 6 months. Kidney stones: citrate supplementation "
            "(potassium citrate). Bone demineralisation: calcium + vitamin D supplementation. "
            "Growth faltering in children: monthly growth charts. "
            "Constipation: fibre supplementation + hydration."
        ),
        "monitoring": "Serum ketones (β-OHB) 3–6 mmol/L; lipids every 6 months; bone density annually; growth",
        "contraindications": "Porphyria; carnitine deficiency; fatty acid oxidation disorders; POLG + VPA",
    },
]


# ─── AED Monitoring ──────────────────────────────────────────────────────────

_MONITORING = [
    {
        "item": "CBZ / OXC: HLA-B*1502 Screening + TDM + Hyponatraemia",
        "frequency": "HLA once before start; TDM 4–12 mg/L (CBZ), MHD 12–35 mg/L (OXC); Na+ monthly×3 then q6M",
        "rationale": (
            "SJS/TEN risk 5–10× in HLA-B*1502 carriers (Han Chinese, South Asian, Southeast Asian) — "
            "genetic testing MANDATORY before CBZ/OXC initiation in these ethnic groups (FDA black box). "
            "SIADH hyponatraemia (Na+ < 125 mmol/L) in 15–25% OXC, 10% CBZ — symptomatic "
            "hyponatraemia (confusion, seizures, coma) requires drug cessation or switch. "
            "Therapeutic drug monitoring (TDM): check trough (morning pre-dose) after breakthrough "
            "seizure, compliance concern, or dose change."
        ),
    },
    {
        "item": "LTG: SJS Surveillance + VPA Interaction + Slow Titration Log",
        "frequency": "Skin rash surveillance daily (patient self-monitoring); TDM 3–15 mg/L; VPA co-prescription: halve dose",
        "rationale": (
            "LTG SJS risk highest in first 8 weeks and with: (1) rapid titration, (2) VPA co-prescription "
            "(VPA doubles LTG levels via UGT1A4 inhibition), (3) age < 12 years, (4) rash re-challenge. "
            "Patient must STOP drug and call clinic immediately for any skin rash — no exceptions. "
            "Titration diary: weekly dose escalation record signed by physician. "
            "VPA + LTG combination requires LTG starting dose of 25 mg every other day (adult)."
        ),
    },
    {
        "item": "LEV: Behavioural / Mood PHQ-9 + GAD-7 Monitoring",
        "frequency": "PHQ-9 + GAD-7 at every clinic visit (monthly for 3 months, then every 3 months)",
        "rationale": (
            "LEV neuropsychiatric adverse effects (irritability, aggression, depression) in 15–25% — "
            "underreported. Standardised screening instruments (PHQ-9 ≥ 10 = moderate depression → "
            "dose review / switch / psychiatric referral; GAD-7 ≥ 10 = moderate anxiety). "
            "Pyridoxine B6 100 mg/day: clinician option to reduce irritability (small evidence). "
            "Renal dosing: eCrCl < 50 mL/min → reduce dose by 50% (LEV renally cleared, unchanged)."
        ),
    },
    {
        "item": "PER: REMS Counselling + Psychiatric Monitoring + Enzyme Inducer Level Check",
        "frequency": "REMS enrolment before first prescription; PHQ-9 + aggression screen every visit; PER TDM if on CBZ/OXC",
        "rationale": (
            "Perampanel FDA REMS: prescriber must complete REMS training + counsel patient on aggression "
            "and suicidal ideation risk — documentation in chart MANDATORY. "
            "Enzyme inducers (CBZ/OXC) reduce PER serum levels by ~50% (CYP3A4 induction) — "
            "patients on CBZ + PER may need up to 12 mg/night (vs standard 4–8 mg) to achieve "
            "therapeutic effect. Dose-dependent dizziness: nightly administration (H/S) minimises "
            "daytime impairment. Weight monitoring monthly — PER associated with 2–3 kg gain."
        ),
    },
]


# ─── Lifecycle / Developmental Trajectory ────────────────────────────────────

_LIFECYCLE = [
    {
        "window": "Early Childhood (3–6 years) — Panayiotopoulos Syndrome Phase",
        "age_range": "3–6 y",
        "key_events": (
            "PS onset: nocturnal autonomic seizures (vomiting + unresponsiveness in 75% during sleep). "
            "Often first event = prolonged autonomic status epilepticus → emergency department. "
            "EEG: occipital spikes (may be multifocal); MRI: normal (mandatory to exclude structural). "
            "AED decision: discuss with family — PS has 90% remission → many clinicians defer AED "
            "for < 2 seizures. If AED started: LEV or LTG. Rescue medication: buccal midazolam provided."
        ),
        "focus": "Accurate diagnosis (exclude encephalitis/gastroenteritis), seizure safety, rescue medication training",
    },
    {
        "window": "School Age (6–12 years) — Idiopathic OLE Transition",
        "age_range": "6–12 y",
        "key_events": (
            "PS remission expected by age 9–10 in 90%. If not remitted → EEG/MRI review; "
            "consider evolving to BCECTS-overlap or JME. Structural OLE onset (FCD/cavernoma): "
            "visual aura + oculomotor seizures emerging. School impact: photosensitivity triggers "
            "(classroom TV/projector). Neuropsychological testing: visual-processing assessment. "
            "Photosensitivity: FL-41 lenses, screen filter, classroom accommodation letter."
        ),
        "focus": "School seizure action plan, photosensitivity accommodations, neuropsych testing",
    },
    {
        "window": "Adolescence (12–18 years) — Structural OLE + JME Exclusion",
        "age_range": "12–18 y",
        "key_events": (
            "Structural OLE (FCD, LGG): peak surgical candidacy assessment. "
            "Pre-surgical evaluation: scalp video-EEG, 3T MRI FCD-protocol, neuropsychology, "
            "fMRI visual cortex mapping. SEEG if MRI-negative. "
            "Exclude JME evolution: absence seizures + myoclonic jerks → review diagnosis. "
            "Driving: minimum 12 months seizure-free before licensing assessment. "
            "OCP interaction (CBZ/OXC): mandatory contraception counselling for female adolescents."
        ),
        "focus": "Surgical evaluation, driving counselling, OCP-AED interaction, JME differentiation",
    },
    {
        "window": "Young Adult (18–30 years) — Surgical Outcomes + Work/Driving",
        "age_range": "18–30 y",
        "key_events": (
            "Post-surgical: Engel classification at 1, 2, 5 years. Visual field assessment: "
            "Humphrey automated perimetry every 6 months post-lobectomy. "
            "DVLA/driving regulations: homonymous hemianopia → permanent driving ban in most jurisdictions. "
            "Quadrantanopia: variable (country-specific) — ophthalmology/DVLA specialist review. "
            "Childbearing: folic acid 5 mg/day pre-conception; AED teratogenicity counselling "
            "(LEV safest; CBZ/OXC enzyme inducers; LTG-VPA interaction in pregnancy). "
            "SUDEP: FBTCS frequency monitoring — target FBTCS = 0 (SUDEP prevention)."
        ),
        "focus": "SUDEP prevention, driving/visual field, childbearing planning, folic acid",
    },
    {
        "window": "Childbearing / Middle Adult (30–50 years) — AED Safety + Bone Health",
        "age_range": "30–50 y",
        "key_events": (
            "Enzyme-inducing AEDs (CBZ/OXC): bone mineralisation loss → dual-energy X-ray "
            "absorptiometry (DEXA) every 2 years; calcium 1200 mg/day + vitamin D 2000 IU/day. "
            "Polypharmacy risk: review indication for every AED — discontinue if seizure-free "
            "≥ 5 years post-surgery (surgical cure). "
            "Catamenial OLE women: menopause transition (oestrogen ↓) may improve OLE "
            "or shift pattern — taper clobazam catamenial add-on after menopause. "
            "Pharmacogenomics panel if multiple AED failures: CYP2C19, CYP3A5, ABCB1."
        ),
        "focus": "Bone health, AED rationalisation, pharmacogenomics, catamenial pattern change",
    },
    {
        "window": "Older Adult (50+ years) — Cognitive Safety + Polypharmacy Review",
        "age_range": "50+ y",
        "key_events": (
            "Structural OLE recurrence risk: if LGG treated → monitor MRI for progression every year. "
            "AED cognitive safety: avoid polypharmacy; reassess need for all AEDs. "
            "If seizure-free > 10 years: consider AED taper (discuss SUDEP vs taper risks). "
            "Visual symptoms: age-related macular degeneration or glaucoma may mimic OLE visual aura "
            "→ ophthalmology evaluation annually. Drug interactions: polypharmacy (statins + CBZ → "
            "statin inefficacy; warfarin + CBZ → INR reduction). Fall risk: AED sedation + balance."
        ),
        "focus": "Cognitive safety, LGG surveillance, AED deprescribing, fall prevention",
    },
]


# ─── Definitions ─────────────────────────────────────────────────────────────

_DEFINITIONS = [
    {
        "term": "Occipital Lobe Epilepsy (OLE)",
        "definition": (
            "A focal epilepsy syndrome in which seizures originate from the occipital lobe. "
            "Characterised by visual symptoms (elementary or complex visual hallucinations, ictal "
            "amaurosis), oculomotor deviation, autonomic features (ictal vomiting in Panayiotopoulos "
            "Syndrome), and post-ictal headache. Third most common lobar epilepsy after TLE and FLE."
        ),
    },
    {
        "term": "Panayiotopoulos Syndrome (PS)",
        "definition": (
            "An idiopathic focal epilepsy of childhood (onset 3–6 years) characterised by autonomic "
            "seizures (ictal vomiting, pallor, diaphoresis) originating from occipital/insular cortex. "
            "75% of seizures occur during sleep. Prognosis excellent: >90% remission within 1–2 years. "
            "Named after CP Panayiotopoulos who characterised the syndrome in 1988."
        ),
    },
    {
        "term": "Fixation-Off Sensitivity (FOS)",
        "definition": (
            "An EEG phenomenon in which occipital epileptiform discharges are elicited or markedly "
            "enhanced by removal of central visual fixation (eye closure in darkness) and suppressed "
            "by fixation (eyes open in light). Pathognomonic for occipital lobe epileptogenicity. "
            "Mechanism: removal of visual fixation releases cortical surround inhibition."
        ),
    },
    {
        "term": "Photoparoxysmal Response (PPR)",
        "definition": (
            "EEG abnormality (generalised or occipital spike-wave) elicited by intermittent photic "
            "stimulation (IPS) during EEG recording. Occipital PPR in OLE indicates photosensitivity "
            "of occipital origin. Clinically correlates with seizures triggered by flickering light "
            "(TV, video games, strobe lights). Prevalence ~30% in OLE vs ~35% in JME."
        ),
    },
    {
        "term": "Focal Cortical Dysplasia (FCD) Type II",
        "definition": (
            "A malformation of cortical development characterised by dysmorphic neurons (Type IIa) "
            "with or without balloon cells (Type IIb) in focal cortical areas. Caused by somatic "
            "mTOR pathway gain-of-function mutations (MTOR, PIK3CA, DEPDC5). MRI: transmantle sign, "
            "grey-white junction blurring, cortical thickening. Most epileptogenic FCD subtype; "
            "surgical resection achieves Engel I in 55–70%."
        ),
    },
    {
        "term": "Ictal Vomiting",
        "definition": (
            "Vomiting as a clinical manifestation of a seizure — occurs during or immediately after "
            "ictal discharge. Localising value: non-dominant posterior quadrant (occipital/insular). "
            "Pathognomonic for Panayiotopoulos Syndrome when occurring in a child with impaired "
            "awareness. Must be differentiated from post-ictal nausea/vomiting, cyclic vomiting "
            "syndrome, and viral gastroenteritis."
        ),
    },
    {
        "term": "Visual Aura",
        "definition": (
            "A subjective visual experience at seizure onset, representing an ictal FAS from occipital "
            "cortex. Elementary visual aura (V1/V2 source): phosphenes (flashing lights, sparks), "
            "moving geometrical patterns, scotoma, ictal amaurosis. Complex visual aura (extrastriate "
            "source): formed objects, faces, scenes (complex hallucinations). Duration < 3 minutes "
            "distinguishes seizure aura from migraine aura (20–60 minutes)."
        ),
    },
    {
        "term": "Post-ictal Headache",
        "definition": (
            "Headache occurring within 3 hours after a seizure, lasting minutes to hours. "
            "Prevalence much higher after OLE seizures (50–60%) than TLE/FLE (20–30%). "
            "Mechanism: occipital cortex post-ictal cortical spreading depolarisation (CSD) "
            "→ trigeminovascular activation → migrainous-quality headache. "
            "Clinical importance: OLE + post-ictal headache is frequently misdiagnosed as migraine."
        ),
    },
    {
        "term": "CEC Syndrome (Celiac Disease, Epilepsy, Calcifications)",
        "definition": (
            "Also called Gobbi Syndrome. Triad of: (1) celiac disease (gluten-sensitive enteropathy), "
            "(2) occipital epilepsy, (3) bilateral occipital calcifications on CT (in parieto-occipital "
            "watershed zones). Mechanism: folate malabsorption → hyperhomocysteinaemia → occipital "
            "microangiopathy → calcifications → epilepsy. Treatment: strict gluten-free diet + "
            "folic acid 5 mg/day may halt progression."
        ),
    },
    {
        "term": "POLG (Polymerase Gamma) Mutation / Alpers Syndrome",
        "definition": (
            "Autosomal recessive disorder of mitochondrial DNA replication (POLG gene) causing mtDNA "
            "depletion. Alpers-Huttenlocher Syndrome (infantile/childhood form): refractory occipital "
            "epilepsy, hepatic failure, progressive neurological decline. "
            "CRITICAL: valproate in POLG patients causes acute hepatic failure — ABSOLUTE "
            "CONTRAINDICATION. Ketogenic diet may be beneficial (provides alternative neuronal fuel)."
        ),
    },
    {
        "term": "Occipital Lobectomy",
        "definition": (
            "Surgical resection of the epileptogenic zone within the occipital lobe. Achieves Engel I "
            "(seizure-free) in 55–65% (FCD II) to 65–80% (cavernoma/LGG). "
            "Mandatory pre-operative mapping: fMRI visual cortex (V1) localisation, DTI optic radiation "
            "tractography, Humphrey visual field test. Expected post-operative deficit: contralateral "
            "homonymous quadrantanopia or hemianopia (permanent) — mandatory pre-operative counselling."
        ),
    },
    {
        "term": "Engel Classification",
        "definition": (
            "Post-surgical seizure outcome classification (Engel 1987): "
            "Class I = Seizure free (IA: completely free; IB: only non-disabling auras); "
            "Class II = Rare disabling seizures (worthwhile improvement); "
            "Class III = Worthwhile improvement (≥50% reduction); "
            "Class IV = No improvement. "
            "OLE surgical goals: Engel I is achievable in 55–80% depending on etiology."
        ),
    },
    {
        "term": "Fixation-Off Sensitivity vs Photosensitivity",
        "definition": (
            "Distinct occipital phenomena: FOS = discharges emerge ON EYE CLOSURE in darkness "
            "(not related to flickering light stimulus); suppressed by fixation in light. "
            "Photosensitivity (PPR) = discharges elicited by FLICKERING LIGHT during IPS. "
            "Both indicate occipital source but different mechanisms and different clinical "
            "management (FOS: corrective lenses; PPR: avoid flickering environments + blue-light lenses)."
        ),
    },
    {
        "term": "DEPDC5 / GATOR1 Complex",
        "definition": (
            "DEPDC5 encodes a subunit of the GATOR1 complex — a negative regulator of mTORC1. "
            "Loss-of-function DEPDC5 variants → mTORC1 hyperactivation → abnormal neuronal growth "
            "and cortical dysplasia (FCD) → epilepsy. Autosomal dominant; incomplete penetrance. "
            "Associated with focal epilepsies including OLE, FLE, TLE, and nocturnal frontal epilepsy. "
            "DEPDC5 + OLE: often MRI-normal or subtle FCD on 3T/7T protocol MRI."
        ),
    },
]


# ─── Standards ───────────────────────────────────────────────────────────────

_STANDARDS = [
    {
        "standard": "ILAE 2022 Classification of Seizures and Epilepsies",
        "relevance": "Defines OLE as a focal epilepsy by lobe; FAS visual aura classification; FOS as EEG phenomenon",
    },
    {
        "standard": "NICE NG217 (2022) — Epilepsies: Diagnosis and Management",
        "relevance": "CBZ/OXC first-line focal epilepsy (Level A); LTG Level A; visual aura management; surgical referral pathway",
    },
    {
        "standard": "ILAE 2017 Surgical Guidelines (Rosenow & Lüders)",
        "relevance": "Pre-surgical evaluation protocol; SEEG candidacy; Engel classification; occipital lobectomy indications",
    },
    {
        "standard": "FDA HLA-B*1502 Black Box Warning (CBZ, 2007)",
        "relevance": "Mandatory HLA-B*1502 screening in Han Chinese and SE Asian patients before CBZ/OXC initiation",
    },
    {
        "standard": "FDA Perampanel REMS (2012)",
        "relevance": "Mandatory REMS counselling for PER: aggression, hostility, suicidal ideation — prescriber training + patient counselling",
    },
    {
        "standard": "Bien CG et al. 2000 Brain — CEC Syndrome Diagnostic Criteria",
        "relevance": "Diagnostic criteria for CEC (celiac + epilepsy + calcifications): anti-tTG, folate, gluten-free diet protocol",
    },
]


# ─── Thresholds ──────────────────────────────────────────────────────────────

_THRESHOLDS = [
    {
        "threshold": "OLE Surgical Referral: ≥2 AED Failures (Drug-Resistant Epilepsy)",
        "value": "≥ 2 appropriate AEDs failed",
        "rationale": "ILAE 2010 DRE definition: failure of 2 adequately dosed, appropriate AEDs → surgical evaluation MANDATORY",
    },
    {
        "threshold": "CBZ TDM Target",
        "value": "4–12 mg/L (trough)",
        "rationale": "Therapeutic range for seizure control; toxicity (ataxia/diplopia) at > 12 mg/L",
    },
    {
        "threshold": "OXC MHD (Active Metabolite) TDM Target",
        "value": "12–35 mg/L (trough)",
        "rationale": "Active metabolite monohydroxy derivative; clinical correlates with MHD level, not OXC level",
    },
    {
        "threshold": "LTG TDM Target",
        "value": "3–15 mg/L (trough); with VPA: 3–7 mg/L",
        "rationale": "With VPA: VPA doubles LTG levels → lower target range and lower starting dose",
    },
    {
        "threshold": "Driving Licence — Seizure-Free Period",
        "value": "12 months seizure-free (UK DVLA / most jurisdictions)",
        "rationale": "Post-lobectomy visual field loss (hemianopia) = permanent driving ban regardless of seizure freedom",
    },
    {
        "threshold": "Folic Acid Pre-Conception",
        "value": "5 mg/day (high-dose, on AEDs)",
        "rationale": "AED teratogenicity risk: high-dose folic acid 5 mg/day for all women on AEDs planning pregnancy (NICE NG217)",
    },
]


# ─── References ──────────────────────────────────────────────────────────────

_REFERENCES = [
    "Panayiotopoulos CP 1999 Epilepsia 40:1127–1131 — Panayiotopoulos Syndrome: clinical description, autonomic seizures, prognosis",
    "Williamson PD et al. 1992 Ann Neurol 31:193–201 — OLE surgical series: 25 patients, 56% Engel I, visual aura localisation",
    "Taylor I et al. 2003 Epilepsia 44:959–966 — Familial OLE (CACNA1A/DEPDC5 genetics), autosomal dominant visual seizures",
    "Bien CG et al. 2000 Brain 123:2406–2418 — CEC Syndrome: celiac disease + occipital calcifications + epilepsy, GFD treatment",
    "Guerrini R et al. 2010 Epilepsia 51:1147–1157 — Occipital FCD type II: surgical and genetic characterisation, MTOR mutations",
    "Jeha LE et al. 2009 Epilepsia 50:1204–1211 — Cleveland Clinic OLE series: 30 patients, 63% Engel I/II, visual field outcomes",
]


# ─── Patient data ────────────────────────────────────────────────────────────

def _build_patients():
    rows = _db_rows(
        "SELECT patient_id, age, gender, diagnosis, seizure_type, medication "
        "FROM patients ORDER BY patient_id"
    )
    if not rows:
        rows = [{"patient_id": f"P{i:03d}", "age": 20 + (i * 7) % 55,
                 "gender": "F" if i % 2 == 0 else "M",
                 "diagnosis": "Focal Epilepsy", "seizure_type": "Focal", "medication": "CBZ"}
                for i in range(1, 42)]

    onset_types = ["Structural FCD", "Panayiotopoulos Syndrome", "Structural Lesional",
                   "POLG Genetic", "Familial FOLE (DEPDC5)", "CEC Syndrome"]
    seizure_types_list = ["FAS-Visual Aura", "FIAS-Oculomotor", "FBTCS", "CVH"]
    control_states = ["Seizure-free", "Partially controlled", "Drug-resistant"]
    aed_list = ["CBZ", "OXC", "LTG", "LEV", "LCM", "PER", "KD", "Post-surgical"]

    patients = []
    for r in rows:
        s = _seed(r["patient_id"])
        onset_age = 3 + (s % 45)
        years_disease = max(1, (r["age"] or 30) - onset_age) if onset_age < (r["age"] or 30) else 2
        etiol = onset_types[s % len(onset_types)]
        primary_sz = seizure_types_list[(s >> 3) % len(seizure_types_list)]
        secondary_sz = seizure_types_list[(s >> 6) % len(seizure_types_list)]
        aed_current = aed_list[(s >> 9) % len(aed_list)]
        control = control_states[(s >> 12) % len(control_states)]
        photosensitive = bool((s >> 15) % 2)
        post_ictal_headache = bool((s >> 17) % 3 > 0)
        surgical_candidate = control == "Drug-resistant" and etiol != "Panayiotopoulos Syndrome"
        visual_field_defect = etiol in ("Structural FCD", "Structural Lesional") and control == "Seizure-free"
        patients.append({
            "patient_id": r["patient_id"],
            "age": r["age"] or 30,
            "gender": r["gender"] or "F",
            "onset_age": onset_age,
            "years_disease": years_disease,
            "etiology": etiol,
            "primary_seizure_type": primary_sz,
            "secondary_seizure_type": secondary_sz if secondary_sz != primary_sz else None,
            "current_aed": aed_current,
            "seizure_control": control,
            "photosensitive": photosensitive,
            "post_ictal_headache": post_ictal_headache,
            "surgical_candidate": surgical_candidate,
            "visual_field_defect": visual_field_defect,
            "fos_positive": bool((s >> 19) % 3 > 1),
        })
    return patients


# ─── Public API ──────────────────────────────────────────────────────────────

def overview():
    patients = _build_patients()
    total = len(patients)
    drug_resistant = sum(1 for p in patients if p["seizure_control"] == "Drug-resistant")
    seizure_free = sum(1 for p in patients if p["seizure_control"] == "Seizure-free")
    surgical_candidates = sum(1 for p in patients if p["surgical_candidate"])
    photosensitive_n = sum(1 for p in patients if p["photosensitive"])
    post_ictal_headache_n = sum(1 for p in patients if p["post_ictal_headache"])
    fos_n = sum(1 for p in patients if p["fos_positive"])

    etiology_counts = {}
    for p in patients:
        etiology_counts[p["etiology"]] = etiology_counts.get(p["etiology"], 0) + 1
    top_etiology = max(etiology_counts, key=lambda k: etiology_counts[k])

    sz_counts = {}
    for p in patients:
        sz_counts[p["primary_seizure_type"]] = sz_counts.get(p["primary_seizure_type"], 0) + 1

    return {
        "dashboard": "Occipital Lobe Epilepsy (OLE)",
        "subtitle": "Third most common focal epilepsy — visual aura, autonomic seizures, post-ictal headache",
        "total_patients": total,
        "drug_resistant_n": drug_resistant,
        "drug_resistant_pct": round(drug_resistant / total * 100, 1),
        "seizure_free_n": seizure_free,
        "seizure_free_pct": round(seizure_free / total * 100, 1),
        "surgical_candidates_n": surgical_candidates,
        "photosensitive_n": photosensitive_n,
        "photosensitive_pct": round(photosensitive_n / total * 100, 1),
        "post_ictal_headache_n": post_ictal_headache_n,
        "post_ictal_headache_pct": round(post_ictal_headache_n / total * 100, 1),
        "fos_positive_n": fos_n,
        "fos_positive_pct": round(fos_n / total * 100, 1),
        "top_etiology": top_etiology,
        "seizure_type_distribution": sz_counts,
        "etiologies_count": len(_ETIOLOGIES),
        "seizure_types_count": len(_SEIZURE_TYPES),
        "triggers_count": len(_TRIGGERS),
        "treatments_count": len(_TREATMENTS),
        "lifecycle_windows": len(_LIFECYCLE),
        "references": _REFERENCES,
        "clinical_alerts": [
            "POLG ABSOLUTE CONTRAINDICATION: Valproate → acute liver failure in POLG mutations",
            "HLA-B*1502: Screen before CBZ/OXC in Han Chinese / South Asian patients (SJS risk)",
            "Panayiotopoulos Syndrome: ictal vomiting in child = seizure (not gastroenteritis) — buccal midazolam",
            "LTG SLOW TITRATION: any rash → stop immediately; VPA doubles LTG levels → halve starting dose",
            "PER REMS MANDATORY: aggression/suicidal ideation — REMS enrolment before prescription",
            "Visual field assessment: Humphrey VF before + after any occipital surgical resection",
        ],
        "surgical_targets": {
            "engel_i_fcd_ii": "55–65%",
            "engel_i_lesional": "65–80%",
            "expected_vf_deficit": "Contralateral quadrantanopia (all) / hemianopia (extensive resection)",
        },
        "as_of": date.today().isoformat(),
    }


def breakdown():
    patients = _build_patients()
    return {
        "dashboard": "Occipital Lobe Epilepsy (OLE) — Detailed Breakdown",
        "patients": patients,
        "etiology_catalog": _ETIOLOGIES,
        "seizure_types": _SEIZURE_TYPES,
        "triggers": _TRIGGERS,
        "treatments": _TREATMENTS,
        "aed_monitoring": _MONITORING,
        "lifecycle": _LIFECYCLE,
        "standards": _STANDARDS,
        "thresholds": _THRESHOLDS,
        "references": _REFERENCES,
        "as_of": date.today().isoformat(),
    }


def definitions():
    return {
        "dashboard": "Occipital Lobe Epilepsy (OLE) — Definitions & Standards",
        "concepts": _DEFINITIONS,
        "standards": _STANDARDS,
        "thresholds": _THRESHOLDS,
        "references": _REFERENCES,
        "as_of": date.today().isoformat(),
    }
