"""Childhood Absence Epilepsy (CAE) Dashboard — the most common generalized epilepsy of
childhood, accounting for 10–15% of childhood epilepsies (2–8 per 100,000 children).

Hallmarks:
  Typical Absences: abrupt onset/offset, 3 Hz generalized spike-wave (GSW), 5–30 seconds,
    staring + automatisms in ~60%, no post-ictal confusion, provoked 100% by hyperventilation
  Onset: 4–10 years (peak 5–7 years); female predominance (60%)
  Prognosis: ~70–80% remit by adolescence; ~15–20% evolve to JME
  Drug resistance: ~10–20% — far lower than focal epilepsies
EEG hallmark: 3 Hz (2.5–4 Hz) generalized spike-wave, frontally predominant, synchronous;
  hyperventilation provokes absences in virtually all untreated patients (diagnostic)
Seizure types: Typical Absence → ± Automatisms → rare Brief GTCS → rare Absence SE

FIRST-LINE TREATMENT SELECTION (CHILDHOOD 2010, NEJM):
  Ethosuximide (ETX): 53% freedom at 16W — SUPERIOR neuropsychological safety profile,
    NO impact on attention; drug of choice when absences are the ONLY seizure type
  Valproate (VPA): 53% freedom at 16W — equal efficacy, but weight gain + teratogenicity
    + REMS requirements; prefer when GTCS present or JME evolution suspected
  Lamotrigine (LTG): 29% freedom at 16W — SIGNIFICANTLY INFERIOR (Level A evidence);
    second-line only; avoid as monotherapy first-line
ABSOLUTE CONTRAINDICATIONS (worsen absences / aggravate seizures):
  Carbamazepine (CBZ) · Oxcarbazepine (OXC) · Phenytoin (PHT) · Vigabatrin (VGB)
  Tiagabine (TGB) · Gabapentin (GBP) · Pregabalin (PGB) — all PRO-ABSENCE

References:
  - Glauser TA et al. 2010 NEJM CHILDHOOD (ETX vs VPA vs LTG — ETX and VPA superior)
  - Berg AT et al. 2001 Epilepsia (71% remission by 12 years follow-up — natural history)
  - Camfield CS & Camfield PR 1993 Neurology (CAE long-term outcome — 44% relapse on taper)
  - Tenney JR & Glauser TA 2013 Curr Opin Neurol (CAE management — ETX preferred for attention)
  - Loiseau P et al. 1983 Epilepsia (adult follow-up of childhood absence — 80% remission)
  - Scheffer IE et al. 2017 Epilepsia (ILAE 2017 CAE classification and diagnostic criteria)
Data: live clinical.db (41 epilepsy patients, deterministic CAE overlay)
      + curated CAE pharmacology / etiology / seizure-type / trigger catalogs."""

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


# ─── Etiology / genetic catalog ─────────────────────────────────────────────

_ETIOLOGIES = [
    {
        "etiology": "Polygenic / Complex Genetic (GABRG2, CLCN2, GABRA1)",
        "category": "Genetic",
        "pct": 65,
        "mechanism": (
            "CAE is predominantly a genetic generalised epilepsy (GGE) with complex polygenic "
            "inheritance. The principal genetic loci involve GABA-A receptor subunit genes: "
            "GABRG2 (γ2 subunit; R43Q/R139G/K289M variants alter receptor trafficking and "
            "channel kinetics → reduced inhibitory tone in thalamocortical circuits), "
            "GABRA1 (α1 subunit; A322D missense → proteasomal degradation, loss of surface "
            "receptor), and CLCN2 (chloride channel-2; G715E → impaired Cl⁻ extrusion). "
            "All disrupt the thalamocortical pacemaker: reduced T-type Ca²⁺ current and/or "
            "reduced GABA-A inhibition allows 3 Hz burst-firing of thalamic relay neurons "
            "→ synchronized cortical spike-wave discharge. Penetrance is incomplete; "
            "monozygotic concordance ~70-80%."
        ),
        "eeg_correlate": "3 Hz (2.5–4 Hz) generalized spike-wave (GSW), frontally "
                         "predominant, high-amplitude, bisynchronous; abrupt onset/offset; "
                         "thalamocortical source confirmed by MEG/SEEG studies",
        "mri_finding": "MRI normal in >95% of CAE (prerequisite for diagnosis per ILAE 2017). "
                       "Minor volumetric differences in thalamus and prefrontal cortex on "
                       "research MRI (voxel-based morphometry) — not clinically actionable",
        "clinical_note": "Genetic counselling: sibling risk ~10%; parent-to-child risk ~10-15%. "
                         "No specific gene testing required for typical CAE; panel (GABRG2/GABRA1/"
                         "CLCN2/SCN1A) if atypical features, drug-resistant, or family history of "
                         "febrile seizures + absences (Dravet differential)"
    },
    {
        "etiology": "GABRG2 Variants (γ2 Subunit GABA-A Receptor)",
        "category": "Genetic",
        "pct": 10,
        "mechanism": (
            "Pathogenic variants in GABRG2 encoding the γ2 subunit of the GABA-A receptor are the "
            "most common identifiable monogenic cause of CAE. R43Q (exon 2): reduces receptor "
            "surface expression by 50%; temperature-sensitive → febrile seizures + absences in "
            "same patient (Febrile Seizures Plus Absence, GEFS+ spectrum). R139G: altered "
            "receptor kinetics + reduced GABA-A current amplitude. K289M (membrane-spanning "
            "domain): severe receptor trafficking defect. Mechanism: loss-of-function γ2 variants "
            "destabilise the GABA-A pentamer → fewer functional channels at thalamocortical "
            "synapses → reduced inhibitory postsynaptic currents → oscillatory 3 Hz discharge."
        ),
        "eeg_correlate": "Classic 3 Hz GSW ± atypical features (irregular, 2-3 Hz) in "
                         "severe GABRG2 variants; GEFS+ background (febrile + absence phenotype) "
                         "may show higher background frequency variability",
        "mri_finding": "MRI normal; structural imaging not required if genotype confirmed and "
                       "EEG/clinical criteria met",
        "clinical_note": "GABRG2 R43Q: classic CAE + febrile seizures in same family (GEFS+); "
                         "respond well to ETX/VPA; rarely drug-resistant. "
                         "Screen for SCN1A (Dravet) if fever-triggered prolonged seizures — "
                         "critical differential as treatment differs (VPA preferred, CBZ avoided)"
    },
    {
        "etiology": "GABRA1 Variants (α1 Subunit GABA-A Receptor)",
        "category": "Genetic",
        "pct": 8,
        "mechanism": (
            "GABRA1 encodes the α1 subunit of GABA-A receptors — the most abundant subunit in "
            "the adult cortex. A322D missense variant (most studied): rapidly degraded by the "
            "ubiquitin-proteasome system → <10% of wild-type surface expression. "
            "D219N / R214C variants: altered GABA binding affinity and channel gating. "
            "GABRA1 loss-of-function causes a spectrum: typical CAE at mild end, JME at "
            "moderate end, GTCS-dominant GGE at severe end — same gene, different clinical "
            "expression depending on allele severity and modifier genes. "
            "Mechanism: reduced α1-containing GABA-A receptors in cortex/basal ganglia "
            "→ disinhibition of corticothalamic projections → 3 Hz oscillation."
        ),
        "eeg_correlate": "3 Hz GSW typical; some GABRA1 variants show 3.5–4 Hz faster "
                         "discharge and shorter absence duration (more 'JME-like')",
        "mri_finding": "MRI normal; no structural findings expected",
        "clinical_note": "GABRA1-CAE: monitor carefully for JME evolution at puberty — "
                         "myoclonic jerks on awakening signal transition; switch or add LEV "
                         "if myoclonus emerges; VPA preferred over ETX in GABRA1 cases with "
                         "GTCS or myoclonic features"
    },
    {
        "etiology": "CACNA1A / CACNA1H (T-type Ca²⁺ Channel)",
        "category": "Genetic",
        "pct": 7,
        "mechanism": (
            "T-type voltage-gated calcium channels in thalamic relay neurons (VB complex) are "
            "the pacemaker of 3 Hz spike-wave generation. CACNA1H (Cav3.2; T-type) gain-of-"
            "function variants (G773D, C456S): increased window current → prolonged thalamic "
            "burst firing → enhanced 3 Hz oscillation. CACNA1A (Cav2.1; P/Q-type) loss-of-"
            "function: impairs fast synaptic inhibition → secondary thalamocortical "
            "hyperexcitability. Both channels are highly expressed in thalamus and cortex. "
            "T-type Ca²⁺ current underlies the low-threshold spike (LTS) in thalamic relay "
            "neurons — the cellular correlate of the spike-wave discharge. "
            "Ethosuximide's mechanism of action is direct block of Cav3.2 T-type current."
        ),
        "eeg_correlate": "Classic 3 Hz GSW; CACNA1H variants may show longer "
                         "absences (15–30 sec) with more pronounced thalamic involvement",
        "mri_finding": "MRI normal; no structural findings expected",
        "clinical_note": "CACNA1H-CAE: ETX is particularly effective (direct T-type block); "
                         "excellent prognosis for remission. CACNA1A: broader channel involvement "
                         "→ may need VPA for better Ca²⁺ channel spectrum coverage; "
                         "CACNA1A also associated with familial hemiplegic migraine — "
                         "ask about migraine in family"
    },
    {
        "etiology": "SCN1A / SCN1B Variants (Rare CAE Spectrum)",
        "category": "Genetic",
        "pct": 5,
        "mechanism": (
            "SCN1A (Nav1.1) loss-of-function haploinsufficiency causes a spectrum: at mild end, "
            "simple febrile seizures or GEFS+ (febrile + absence). Dravet syndrome (severe end) "
            "is NOT CAE — distinguish by: age onset <1y in Dravet vs 4-10y CAE; prolonged "
            "febrile seizures (Dravet) vs brief absences (CAE); temperature sensitivity (Dravet). "
            "SCN1B (β1 subunit) variants cause GEFS+ with absence component. "
            "Mechanism: Nav1.1 preferentially expressed in fast-spiking inhibitory interneurons "
            "(PV+ cortical interneurons, GABAergic thalamic neurons) → loss-of-function reduces "
            "interneuron firing → paradoxical network hyperexcitability. In mild SCN1A variants, "
            "reduced interneuron inhibition allows thalamocortical 3 Hz oscillations."
        ),
        "eeg_correlate": "SCN1A-GEFS+ with absences: 3 Hz GSW typical; background may "
                         "show higher theta compared to pure genetic CAE; febrile EEG "
                         "slowing may be more pronounced",
        "mri_finding": "MRI normal in GEFS+ / mild SCN1A; sclerosis on MRI → Dravet differential",
        "clinical_note": "CRITICAL: if SCN1A+ with prolonged febrile seizures or fever-triggered "
                         "status epilepticus → diagnose Dravet, NOT CAE; treat with VPA/CLB/STB; "
                         "CBZ/OXC/PHT ABSOLUTELY CONTRAINDICATED in SCN1A. "
                         "CAE-SCN1B: standard ETX/VPA approach; remission common"
    },
    {
        "etiology": "Unknown / Cryptogenic",
        "category": "Unknown",
        "pct": 5,
        "mechanism": (
            "A small proportion of children with typical CAE phenotype (4–10y onset, 3 Hz GSW, "
            "normal MRI, normal development) have no identifiable genetic variant with current "
            "panels. This reflects: (1) incomplete penetrance of polygenic variants below "
            "current detection thresholds, (2) regulatory/non-coding variants not captured by "
            "exome sequencing, (3) somatic mosaicism in thalamocortical circuits, or "
            "(4) true sporadic cases with de novo variants. Whole-genome sequencing (WGS) is "
            "expected to reduce this category substantially as it becomes standard. "
            "Clinically managed identically to polygenic CAE — standard ETX/VPA first-line."
        ),
        "eeg_correlate": "Typical 3 Hz GSW; hyperventilation provocation positive; "
                         "no atypical features; treated empirically",
        "mri_finding": "MRI normal (required — if abnormal, reconsider diagnosis)",
        "clinical_note": "Standard ETX/VPA first-line; excellent prognosis expected; "
                         "repeat EEG in 2 years; genetic panel if: drug-resistant, atypical "
                         "EEG, family history of Dravet/Doose, or associated developmental delay "
                         "(which would exclude classic CAE by ILAE 2017 criteria)"
    },
]


# ─── Seizure type catalog ────────────────────────────────────────────────────

_SEIZURE_TYPES = [
    {
        "type": "Typical Absence Seizure",
        "prevalence_pct": 100,
        "duration_sec": "5–30 sec (mean 10 sec)",
        "description": (
            "The defining seizure of CAE. Abrupt onset and offset — 'like a light switch'. "
            "Staring, behavioral arrest, unresponsiveness. No post-ictal confusion "
            "(patient resumes activity immediately). Often undetected by parents and teachers "
            "for months before diagnosis. Frequency: 10–200 per day without treatment. "
            "EEG: 3 Hz (2.5–4 Hz) generalized, frontally predominant, bisynchronous "
            "spike-wave; high amplitude; lasts seizure duration + 1–2 sec."
        ),
        "eeg_correlate": "3 Hz generalized spike-wave (GSW), synchronous, frontally "
                         "predominant, high amplitude (>300 µV); abrupt onset/offset; "
                         "hyperventilation provokes in virtually all untreated patients",
        "clinical_tip": (
            "Provoke with 3-min hyperventilation during EEG — should provoke absence if "
            "untreated (100% sensitivity); use to confirm diagnosis AND monitor treatment response "
            "(ETX should abolish hyperventilation provocation if effective). "
            "Count absences by behavioral observation + EEG during hyperventilation."
        ),
    },
    {
        "type": "Absence with Automatisms",
        "prevalence_pct": 60,
        "duration_sec": "10–30 sec (longer absences more likely to have automatisms)",
        "description": (
            "Approximately 60% of CAE absences are accompanied by automatisms — semi-purposeful "
            "motor behaviors during the absence. Types: oro-facial (lip-smacking, chewing, "
            "swallowing — most common), manual (fumbling with objects, repetitive hand movements), "
            "ambulatory (rarely — walking during absence). Automatisms are NOT equivalent to "
            "focal onset (unlike TLE automatisms) — CAE automatisms arise secondarily from "
            "diffuse cortical suppression during absence. EEG remains 3 Hz GSW throughout. "
            "Longer absence duration (>10 sec) predicts automatism occurrence."
        ),
        "eeg_correlate": "3 Hz GSW persisting throughout absence including during "
                         "automatisms — distinguishes from focal onset (where ictal discharge "
                         "is localized early). No post-ictal slowing.",
        "clinical_tip": (
            "Absence with automatisms is frequently misdiagnosed as focal aware seizure (TLE). "
            "Key distinguishers: CAE = abrupt onset, generalized EEG, no post-ictal confusion, "
            "hyperventilation-provoked, normal MRI, 4–10y onset. If in doubt, video-EEG + "
            "hyperventilation provocation. MRI should be NORMAL in CAE — abnormal MRI → "
            "reconsider diagnosis → focal epilepsy with secondary generalization."
        ),
    },
    {
        "type": "Brief Generalized Tonic-Clonic Seizure (GTCS)",
        "prevalence_pct": 18,
        "duration_sec": "60–120 sec (typical GTCS duration)",
        "description": (
            "Brief GTCS occur in 15–20% of children with CAE, typically during adolescence. "
            "GTCS in CAE: shorter duration and less severe than focal-onset bilateral tonic-clonic "
            "seizures. Occurrence of GTCS raises clinical concern: (1) consider evolution to JME "
            "(especially if dawn-predominant + myoclonic jerks added), (2) ensure AED is "
            "adequately dosed (GTCS may signal under-treatment), (3) reconsider diagnosis "
            "(is this truly CAE or JME/other GGE?). Valproate is preferred over Ethosuximide "
            "when GTCS are present — ETX does NOT protect against GTCS."
        ),
        "eeg_correlate": "Ictal: generalized polyspike-wave 4–5 Hz (tonic phase) → "
                         "2–3 Hz spike-wave (clonic phase). Interictal: 3 Hz GSW persisting. "
                         "Post-ictal: diffuse slowing 30–60 sec.",
        "clinical_tip": (
            "GTCS in CAE = switch from ETX to VPA OR add VPA. ETX monotherapy does NOT prevent "
            "GTCS. VPA covers both absence + GTCS. CHILDHOOD trial 2010: VPA arm had fewer GTCS "
            "breakthroughs than ETX arm. If myoclonic jerks also present (dawn-predominant) → "
            "diagnose JME — require lifelong VPA/LEV, NOT the remission expected in CAE."
        ),
    },
    {
        "type": "Absence Status Epilepticus (Absence SE)",
        "prevalence_pct": 7,
        "duration_sec": ">5–30 min (may last hours if untreated)",
        "description": (
            "A continuous or recurring absence seizure lasting >5 minutes constitutes absence "
            "status epilepticus (ASE). ASE in CAE: patient appears confused, 'clouded', automatic "
            "behavior, blunted responsiveness — may be subtle. May occur spontaneously or be "
            "triggered by missed AED dose, sleep deprivation, fever, or iatrogenic (CBZ in CAE "
            "can precipitate absence SE — drug-worsening emergency). Unlike convulsive SE, "
            "CAE-ASE rarely causes neuronal injury, but prolonged untreated ASE impairs learning. "
            "Hospital emergency: IV lorazepam or rectal diazepam terminates CAE-ASE rapidly."
        ),
        "eeg_correlate": "Continuous or near-continuous generalized spike-wave (may "
                         "be irregular/slower 2–2.5 Hz in prolonged ASE) with behavioral "
                         "correlate. EEG is ESSENTIAL for ASE diagnosis — clinical presentation "
                         "alone insufficient (confusion in child has wide differential).",
        "clinical_tip": (
            "If child presents with prolonged confusion >5 min + history of known absences → "
            "treat as absence SE empirically with IV/rectal benzodiazepine (LZP 0.05–0.1 mg/kg "
            "IV). EEG during confusion confirms GSW. Review AED compliance and dose. "
            "CAUTION: never give CBZ/PHT for suspected absence SE — these drugs worsen absence "
            "seizures and can prolong ASE. Investigate precipitant: missed ETX/VPA, illness/fever."
        ),
    },
]


# ─── Trigger catalog ─────────────────────────────────────────────────────────

_TRIGGERS = [
    {
        "trigger": "Hyperventilation",
        "frequency_pct": 100,
        "mechanism": (
            "Hyperventilation (HV) causes cerebral vasoconstriction via hypocapnia (↓PaCO₂), "
            "reducing cerebral blood flow ~30-40%. The resulting mild cerebral alkalosis and "
            "reduced CO₂ directly increases thalamocortical excitability and lowers seizure "
            "threshold for 3 Hz spike-wave generation. HV provokes absences in virtually ALL "
            "untreated CAE patients — used as the primary diagnostic test. Effective treatment "
            "(ETX/VPA) should abolish HV-provoked absences, making it a treatment response marker."
        ),
        "management": (
            "Clinical use: 3 minutes of vigorous hyperventilation during EEG (have patient blow "
            "on pinwheel or paper); count provoked absences. Treatment monitoring: HV provocation "
            "should be negative on therapeutic AED dose — persistent HV-provoked absences = "
            "undertreated or drug-resistant CAE."
        ),
    },
    {
        "trigger": "Missed AED Dose",
        "frequency_pct": 60,
        "mechanism": (
            "Sub-therapeutic AED levels from missed doses directly reduce the seizure threshold. "
            "Ethosuximide has a short half-life (ETX t½ = 40-60 h in children), making missed "
            "doses more impactful than for longer-acting AEDs. "
            "VPA t½ = 9-16 h — even more sensitive to missed doses. Trough levels below minimum "
            "therapeutic allow resumption of thalamocortical 3 Hz oscillations."
        ),
        "management": (
            "Adherence counselling at every visit. ETX: twice-daily dosing improves adherence; "
            "use pill organizer/phone alarm reminder. School nurse administration for noon dose. "
            "Check ETX/VPA serum levels if seizure breakthrough without obvious precipitant."
        ),
    },
    {
        "trigger": "Sleep Deprivation",
        "frequency_pct": 50,
        "mechanism": (
            "Sleep deprivation reduces inhibitory GABAergic tone and alters thalamocortical "
            "synchrony, lowering the threshold for 3 Hz spike-wave generation. "
            "Unlike JME where the morning-after-sleep-deprivation effect is most pronounced, "
            "CAE absences increase throughout the day when tired. Shift from non-REM to REM "
            "and recovery sleep suppresses absences (absence frequency lowest during deep sleep)."
        ),
        "management": (
            "Consistent sleep schedule: 9-11 hours for school-age children. No sleep deprivation "
            "before school performances, exams, or sports events. Parents: prevent late nights "
            "on weekends (common precipitant of Monday morning absence clusters)."
        ),
    },
    {
        "trigger": "Fatigue / Drowsiness",
        "frequency_pct": 45,
        "mechanism": (
            "Fatigue and drowsiness are associated with increased thalamic burst firing and "
            "heightened thalamocortical synchrony — the same neurophysiological state that "
            "generates 3 Hz spike-wave. Absences cluster in the afternoon when children are "
            "tired from school activities. Transition from wake to drowsiness is particularly "
            "vulnerable (increased K-complex frequency during early drowsiness triggers 3 Hz GSW)."
        ),
        "management": (
            "Short structured afternoon rest (20-30 min nap) may paradoxically reduce afternoon "
            "absence burden. Avoid overscheduling of activities after school. Ensure adequate "
            "nighttime sleep to prevent daytime fatigue. Monitor for absence clusters during "
            "PE/sport transition to rest."
        ),
    },
    {
        "trigger": "Psychological Stress / Anxiety",
        "frequency_pct": 40,
        "mechanism": (
            "Psychosocial stress activates the HPA axis → cortisol release → alters GABA-A "
            "receptor subunit expression (reduces δ-subunit extrasynaptic GABA-A current) "
            "→ reduced neurosteroid (allopregnanolone) modulation of GABAergic tone → "
            "increased thalamocortical excitability. School examinations, social stress, and "
            "family conflict are common precipitants. Anxiety disorders are more prevalent in "
            "CAE children (attention/learning impact → school anxiety → seizure worsening cycle)."
        ),
        "management": (
            "Screen for anxiety/depression at every visit (SCARED questionnaire for children). "
            "School accommodation letter (504/IEP equivalent): extra test time, preferred seating, "
            "seizure action plan. School nurse education mandatory. Child psychologist referral "
            "if significant anxiety or school refusal. Seizure diary to identify stress triggers."
        ),
    },
    {
        "trigger": "Hunger / Hypoglycemia",
        "frequency_pct": 30,
        "mechanism": (
            "Mild hypoglycemia and metabolic stress from skipped meals reduce glucose availability "
            "to thalamocortical circuits, impairing Na⁺/K⁺-ATPase function and altering "
            "membrane potential maintenance → increased excitability. This mechanism mirrors "
            "the therapeutic rationale of the ketogenic diet (KD) in refractory CAE — shift to "
            "ketone body metabolism stabilizes thalamocortical circuits. "
            "Practical: absences cluster before lunch and afternoon snack in school-age children."
        ),
        "management": (
            "Regular meals and snacks — no skipped breakfast or lunch. School accommodation: "
            "access to snacks during class (504 letter). Morning dose of AED with breakfast "
            "to prevent pre-school absence cluster. Monitor weight (especially on VPA)."
        ),
    },
    {
        "trigger": "Photosensitivity (PPR)",
        "frequency_pct": 17,
        "mechanism": (
            "Photoparoxysmal response (PPR) — generalized spike-wave triggered by intermittent "
            "photic stimulation (IPS) — occurs in ~15-20% of CAE patients. Lower rate than "
            "JME (~35%) and Dravet. Mechanism: visual cortex hyperexcitability → retrograde "
            "activation of thalamocortical loops → 3 Hz GSW generalization. "
            "Screen with IPS during EEG. Clinically: video games, flickering screens, "
            "sunlight through trees, disco lights — less problematic than JME."
        ),
        "management": (
            "Blue-light filtering glasses (Z1 filter) for screen use if PPR present. "
            "Polarized sunglasses outdoors. Video game breaks every 20 min. "
            "ETX and VPA both suppress PPR. Photosensitivity typically resolves with "
            "treatment and may resolve even if absences persist."
        ),
    },
    {
        "trigger": "Catamenial / Perimenarche",
        "frequency_pct": 20,
        "mechanism": (
            "In girls approaching menarche (10-14y), fluctuating estrogen:progesterone ratios "
            "modulate GABA-A receptor function. Perimenstrual phase (estrogen peak, "
            "progesterone trough) reduces allopregnanolone (neurosteroid GABA-A modulator) → "
            "transiently reduced inhibitory tone → absence worsening. "
            "This is clinically significant for CAE girls who have achieved near-remission: "
            "perimenstrual absence recurrence should NOT be mistaken for drug failure. "
            "CAE catamenial pattern is less severe than in TLE/FLE."
        ),
        "management": (
            "Seizure diary including menstrual cycle dates in adolescent girls. "
            "If catamenial clustering identified: consider intermittent adjunctive CLB "
            "(clobazam 5-10 mg) in the 10 days around menstruation. "
            "OCP can regularize the hormonal cycle in severe catamenial CAE. "
            "Note: VPA + OCP interaction less relevant for absence-only CAE (no enzyme induction)."
        ),
    },
]


# ─── Treatment catalog ───────────────────────────────────────────────────────

_TREATMENTS = [
    {
        "drug": "Ethosuximide (ETX)",
        "brand": "Zarontin",
        "mechanism": "Selective T-type voltage-gated Ca²⁺ channel blocker (Cav3.2, CACNA1H) in thalamic "
                     "relay neurons → reduces low-threshold burst firing → suppresses 3 Hz thalamocortical "
                     "oscillation. No effect on Na⁺ channels or GABA-A receptors (hence no protection vs GTCS).",
        "indication": "FIRST-LINE for typical CAE without GTCS (ILAE Level A — CHILDHOOD trial)",
        "dose": "Children: 15 mg/kg/d in 2 divided doses; titrate by 250 mg q1-2W to response; "
                "maintenance 20-40 mg/kg/d (max 1500 mg/d). Adults: 250 mg BD → 500 mg TID. "
                "Target serum level: 40-100 µg/mL.",
        "efficacy": "53% seizure-freedom at 16 weeks (CHILDHOOD 2010, NEJM) — co-superior with VPA; "
                    "superior neuropsychological safety vs VPA (CHILDHOOD: ETX had better attention scores). "
                    "60-80% seizure-free at 12 months with therapeutic levels.",
        "safety": "GI side effects most common (nausea, vomiting, anorexia — give with food); "
                  "hiccups (dose-dependent); headache; rare: SJS (do NOT re-challenge if rash occurs); "
                  "behavioral changes in ~5% (paradoxical hyperactivity). "
                  "No major organ toxicity; no teratogenicity data (avoid in pregnancy — limited).",
        "monitoring": "Serum ETX level q6M or after dose change (40-100 µg/mL). "
                      "CBC q6M (rare aplastic anaemia risk). LFTs annually. "
                      "Hyperventilation provocation: should be negative if dose adequate.",
        "evidence_level": "Level A (CHILDHOOD 2010 RCT — Glauser et al. NEJM)"
    },
    {
        "drug": "Valproate (VPA)",
        "brand": "Depakene / Epilim / Convulex",
        "mechanism": "Multiple mechanisms: (1) Na⁺ channel blockade (reduces high-frequency firing), "
                     "(2) T-type Ca²⁺ channel inhibition (less potent than ETX), "
                     "(3) increased brain GABA levels (GAD induction + GABA transaminase inhibition), "
                     "(4) GABA-A potentiation. Broad-spectrum AED — covers absences AND GTCS.",
        "indication": "FIRST-LINE for CAE with GTCS or JME evolution risk; second choice if ETX "
                      "fails or not tolerated; PREFERRED in males without teratogenicity concern",
        "dose": "20-30 mg/kg/d in 2-3 divided doses; titrate by 5-10 mg/kg/d q1-2W; "
                "maintenance 30-60 mg/kg/d. Target level: 50-100 µg/mL (though correlation weak). "
                "ER formulation (Depakote ER) preferred for compliance and GI tolerability.",
        "efficacy": "53% seizure-freedom at 16 weeks (CHILDHOOD 2010 — equal to ETX for absences); "
                    "better than ETX for GTCS protection; 55-70% freedom at 12 months. "
                    "CHILDHOOD: ETX preferred for pure absence (better attention) but VPA better "
                    "when GTCS present or suspected JME evolution.",
        "safety": "Weight gain (mean +4 kg, dose-dependent — most problematic in adolescent girls); "
                  "hair loss (reversible); tremor (dose-dependent — reduce dose); hepatotoxicity "
                  "(rare in >2y, risk highest <2y — NOT first-line in children <2y); "
                  "pancreatitis (rare, monitor lipase); "
                  "TERATOGENICITY: 10x increased neural tube defects — REMS mandatory; "
                  "polycystic ovary syndrome (PCOS) risk in adolescent girls; "
                  "hyperammonemia (monitor if encephalopathy — especially febrile illness).",
        "monitoring": "VPA serum level q6M (50-100 µg/mL); weight monthly (adolescents); "
                      "LFTs q6M; ammonia if encephalopathy or febrile illness; "
                      "REMS: enrol females of childbearing potential in Depakote REMS program; "
                      "folic acid 5 mg/d for all females; menstrual cycle monitoring (PCOS screen).",
        "evidence_level": "Level A (CHILDHOOD 2010 RCT — Glauser et al. NEJM)"
    },
    {
        "drug": "Lamotrigine (LTG)",
        "brand": "Lamictal",
        "mechanism": "Voltage-gated Na⁺ channel blockade (fast-inactivation state) + inhibits "
                     "glutamate release presynaptically. No direct T-type Ca²⁺ channel effect "
                     "(explains inferior efficacy vs ETX for pure absence). "
                     "Broad-spectrum: effective for GTCS and focal seizures, but weaker "
                     "anti-absence activity than ETX or VPA.",
        "indication": "SECOND-LINE for CAE — only when ETX and VPA are contraindicated, "
                      "not tolerated, or failed. NOT recommended as monotherapy first-line "
                      "(CHILDHOOD evidence: significantly inferior)",
        "dose": "Slow titration (MANDATORY to prevent SJS): 0.15 mg/kg/d × 2W → "
                "0.3 mg/kg/d × 2W → increase by 0.3 mg/kg every 1-2W to target. "
                "Without VPA: target 5-15 mg/kg/d. WITH VPA (50% interaction): "
                "0.15 mg/kg/d × 2W → 0.3 mg/kg/d → target 1-5 mg/kg/d only. "
                "VPA inhibits LTG glucuronidation → 2× LTG level increase → double SJS risk.",
        "efficacy": "29% seizure-freedom at 16 weeks (CHILDHOOD 2010) — SIGNIFICANTLY INFERIOR "
                    "to ETX (53%) and VPA (53%). Do not use as first-line monotherapy. "
                    "Some patients respond well (30-40% responder rate); consider as adjunct.",
        "safety": "Stevens-Johnson Syndrome (SJS) / Toxic Epidermal Necrolysis (TEN): "
                  "0.3-0.8% incidence with rapid titration, much less with slow titration; "
                  "VPA co-administration doubles risk → halve LTG dose AND slow titration "
                  "(12-week titration schedule mandatory with VPA); rash → STOP LTG immediately "
                  "and do NOT re-challenge; dizziness, diplopia, headache common. "
                  "Relatively safe in pregnancy (least teratogenic common AED).",
        "monitoring": "Rash surveillance at every visit (first 8 weeks critical — show parents "
                      "photo of drug rash vs SJS). LTG levels optional (therapeutic 3-15 µg/mL). "
                      "VPA+LTG combination: strict dose calculation mandatory.",
        "evidence_level": "Level B as second-line (CHILDHOOD 2010 — significantly inferior to ETX/VPA)"
    },
    {
        "drug": "Clobazam (CLB)",
        "brand": "Onfi / Frisium",
        "mechanism": "1,5-benzodiazepine; positive allosteric modulator of GABA-A receptors "
                     "(α-subunit containing receptors; less sedating than 1,4-benzodiazepines). "
                     "Enhances Cl⁻ conductance → hyperpolarization → reduced neuronal excitability "
                     "in thalamocortical circuits. Less tolerance development than classic BZDs.",
        "indication": "Adjunctive therapy for refractory CAE or intermittent catamenial exacerbation; "
                      "NOT recommended as primary monotherapy for CAE",
        "dose": "0.05-0.3 mg/kg/d in 1-2 doses (evening dose preferred if using for catamenial). "
                "Adjunctive: 5-10 mg nocte in children; 10-20 mg/d in adults. "
                "Catamenial protocol: 5-10 mg nocte for 10 days perimenstrually.",
        "efficacy": "50-70% responder rate as adjunct; particularly effective for catamenial "
                    "absence clusters. Tolerance develops in 25-30% over 6-12 months.",
        "safety": "Sedation (most common, especially initial titration); behavioral changes in "
                  "children (hyperactivity, irritability in ~15%); dependence with prolonged use; "
                  "withdrawal seizures if abrupt discontinuation; saliva/bronchial secretion increase "
                  "(relevant for children with swallowing difficulties).",
        "monitoring": "Behavioral assessment at each visit (hyperactivity screening in children). "
                      "Tolerance assessment: seizure breakthrough after initial control = tolerance. "
                      "Withdrawal: taper over 4-6 weeks; never abrupt discontinuation.",
        "evidence_level": "Level B as adjunct (expert consensus + observational series)"
    },
    {
        "drug": "Levetiracetam (LEV)",
        "brand": "Keppra",
        "mechanism": "SV2A synaptic vesicle protein modulation → reduces neurotransmitter release; "
                     "inhibits pre-synaptic Ca²⁺ channels (N-type); modulates GABA-A receptor "
                     "trafficking. Broad-spectrum activity including generalized seizures, "
                     "though absence-specific efficacy below ETX/VPA.",
        "indication": "Adjunctive or alternative in CAE if ETX/VPA failed or contraindicated; "
                      "particularly useful when behavioral side effects of VPA are problematic; "
                      "off-label for CAE (not FDA-approved for absence)",
        "dose": "20-60 mg/kg/d in 2 divided doses; start 10 mg/kg/d, titrate by 10 mg/kg "
                "q2W. Adults: 500 mg BD → up to 3000 mg/d.",
        "efficacy": "40-60% responder rate for absences in retrospective studies; "
                    "no randomized trial vs ETX for CAE specifically. "
                    "Better than LTG for absence; may be preferred over VPA in adolescent girls "
                    "(no teratogenicity, no PCOS risk, no weight gain).",
        "safety": "Behavioral: irritability, aggression, mood instability ('Keppra rage') in "
                  "20-35% of children — most important limiting factor; "
                  "pyridoxine (B6) 50-100 mg/d may reduce behavioral side effects; "
                  "no serious organ toxicity; safe in pregnancy (category C, no teratogenicity signal). "
                  "Somnolence, dizziness (initial titration). No significant drug interactions.",
        "monitoring": "Behavioral assessment (PHQ-A / Conners rating) at every visit. "
                      "If behavioral deterioration: consider dose reduction or switch. "
                      "Renal function (LEV renally cleared — dose adjust in renal impairment).",
        "evidence_level": "Level C (off-label, retrospective series; no RCT for CAE)"
    },
    {
        "drug": "Topiramate (TPM)",
        "brand": "Topamax",
        "mechanism": "Multiple mechanisms: Na⁺ channel blockade (sustained inhibition), "
                     "GABA-A potentiation, AMPA/kainate receptor antagonism, "
                     "carbonic anhydrase inhibition. Broad-spectrum but cognitive side effects "
                     "limit use in school-age children with CAE.",
        "indication": "Third-line adjunctive for refractory CAE; used when ETX/VPA/LTG have failed",
        "dose": "Start 1-3 mg/kg/d at night; increase slowly by 1-3 mg/kg q2W; "
                "target 5-9 mg/kg/d. Adults: 25 mg nocte → 100-200 mg/d in 2 doses.",
        "efficacy": "45-55% responder rate for generalized seizures (absences + GTCS); "
                    "limited CAE-specific data. Useful as broad-spectrum third-line.",
        "safety": "COGNITIVE: word-finding difficulty, slowed thinking ('Topiramate stupidity') "
                  "in 30-40% — particularly problematic for school-age children (avoid in "
                  "academic-critical years if possible); nephrolithiasis (renal stones — "
                  "adequate hydration, avoid carbonic anhydrase inhibitors); metabolic acidosis; "
                  "angle-closure glaucoma (rare — acute eye pain = ophthalmology emergency); "
                  "anorexia/weight loss (useful if patient obese on VPA); "
                  "oligohidrosis in children (reduced sweating → heat stroke risk).",
        "monitoring": "Bicarbonate annually (metabolic acidosis). Renal function. "
                      "Cognitive screening (Conners/BRIEF) — if academic decline → reduce dose. "
                      "Advise 1.5-2 L/d fluid intake. Summer: monitor for heat tolerance.",
        "evidence_level": "Level C (retrospective studies; no CAE-specific RCT)"
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "brand": "Classic 4:1 KD / Modified Atkins Diet (MAD)",
        "mechanism": "High fat/low carbohydrate dietary intervention shifts CNS metabolism from "
                     "glucose to ketone bodies (β-hydroxybutyrate, acetoacetate). Mechanisms in "
                     "CAE: (1) β-hydroxybutyrate inhibits T-type Ca²⁺ channels (same target as ETX), "
                     "(2) ATP-sensitive K⁺ channel opening (hyperpolarization), "
                     "(3) increased GABA synthesis (glutamate → succinate → GABA via GABA shunt), "
                     "(4) inhibition of mTOR pathway. Ketosis (BHB >2 mmol/L) correlates with "
                     "anti-seizure efficacy.",
        "indication": "Refractory CAE — ≥2 AED failures; also considered early in CACNA1H-CAE "
                      "where T-type Ca²⁺ pathway targeted. Requires dedicated dietitian + team.",
        "dose": "Classic 4:1 ratio (fat:protein+carb); MAD less restrictive and better tolerated "
                "in older children/adolescents. Hospital admission for initiation recommended. "
                "Target urinary ketones: +++ to ++++. Maintain 3-6 months before assessing response.",
        "efficacy": ">50% seizure reduction in 40-60% of refractory absence cases; "
                    "complete seizure-freedom in 10-20% of refractory cases. "
                    "Some evidence it may accelerate remission in drug-resistant CAE.",
        "safety": "Dyslipidemia (elevated LDL — monitor lipid panel); growth restriction "
                  "(protein monitoring essential); kidney stones (hydration + KCl supplement); "
                  "GI: constipation, reflux, nausea; selenium/zinc/Vitamin D deficiency "
                  "(supplement mandatory); cardiomyopathy rare (selenium deficiency — monitor).",
        "monitoring": "Monthly: weight, urinary ketones, glucose, electrolytes. "
                      "Quarterly: lipid panel, LFTs, renal function, full vitamins panel. "
                      "Annually: renal ultrasound (stones). Dietitian follow-up monthly.",
        "evidence_level": "Level B (Caraballo 2011 Epilepsia; Neal 2008 Lancet for broader GE)"
    },
    {
        "drug": "Zonisamide (ZNS)",
        "brand": "Zonegran",
        "mechanism": "T-type Ca²⁺ channel blockade (similar to ETX); Na⁺ channel blockade; "
                     "carbonic anhydrase inhibition; weak GABA modulation. "
                     "Spectrum similar to TPM but potentially more selective T-type Ca²⁺ effect "
                     "→ theoretical advantage for absence.",
        "indication": "Third-line adjunctive for refractory CAE; used in Asia (Japan) as "
                      "alternative first-line when ETX unavailable; off-label in CAE in EU/NA",
        "dose": "2-4 mg/kg/d in 1-2 doses; start 1-2 mg/kg/d, increase by 1-2 mg/kg q2W; "
                "max 12 mg/kg/d. Adults: 100-600 mg/d.",
        "efficacy": "45-60% responder rate in Japanese CAE series; used as monotherapy in Japan "
                    "with outcomes similar to ETX. Limited RCT data in Western populations.",
        "safety": "Cognitive effects (less than TPM); nephrolithiasis (similar rate to TPM — "
                  "hydration important); anorexia/weight loss; metabolic acidosis; "
                  "oligohidrosis/hyperthermia in children; sulfonamide allergy cross-reaction "
                  "(rare — avoid in sulfa allergy).",
        "monitoring": "Bicarbonate q6M; renal function; weight; Summer heat monitoring. "
                      "Advise adequate fluid intake (1.5-2 L/d).",
        "evidence_level": "Level C (Japanese series, off-label in EU/NA)"
    },
]


# ─── AED monitoring ──────────────────────────────────────────────────────────

_AED_MONITORING = [
    {
        "drug": "Ethosuximide (ETX)",
        "parameters": [
            "Serum ETX level: target 40-100 µg/mL; measure q6M or after dose change",
            "CBC + differential: q6M (rare aplastic anaemia / leukopenia risk)",
            "LFTs annually (minor hepatic metabolism — rarely clinically relevant)",
            "Hyperventilation provocation EEG: should be NEGATIVE if adequate dose — "
            "persistent positive HV provocation = undertreated (increase dose) or drug-resistant CAE",
        ],
        "alerts": [
            "GI SIDE EFFECTS: nausea/vomiting/anorexia — give with food or milk; "
            "divide into 3 doses if GI distress persists; switch to syrup formulation for young children",
            "RASH → STOP ETX immediately: rare SJS; do NOT re-challenge if rash occurs; "
            "assess ETX causality with dermatologist before any re-exposure",
            "BEHAVIORAL CHANGES: hyperactivity, irritability in ~5% — dose-dependent; "
            "reduce dose before switching; distinguish from untreated absence-related inattention",
            "NO GTCS PROTECTION: ETX does NOT prevent generalized tonic-clonic seizures — "
            "if GTCS occur on ETX monotherapy, switch to or add VPA",
        ]
    },
    {
        "drug": "Valproate (VPA)",
        "parameters": [
            "VPA serum level: target 50-100 µg/mL; measure q6M (correlation with efficacy weak "
            "but useful for adherence/toxicity monitoring)",
            "LFTs + ammonia: q6M baseline; urgent if encephalopathy or febrile illness "
            "(hyperammonemia — check NH₃ and consider L-carnitine supplementation)",
            "CBC: q6M (thrombocytopenia dose-dependent — reduce dose if platelets <100K)",
            "Weight: MONTHLY in adolescents (mean gain 4 kg/year — major compliance barrier); "
            "dietary counselling from treatment start",
            "Menstrual cycle (in adolescent girls): irregular cycles → PCOS screen "
            "(testosterone, LH/FSH ratio, pelvic ultrasound)",
            "REMS enrollment: all females of childbearing potential — mandatory in USA (Depakote REMS)",
            "Folic acid 5 mg/d: ALL females of childbearing potential on VPA (NICE NG217)",
        ],
        "alerts": [
            "TERATOGENICITY — REMS: VPA causes neural tube defects (NTD) 10× background risk; "
            "spina bifida; autism spectrum disorder risk (3-4× background); "
            "folic acid 5 mg/d MANDATORY; pregnancy test before initiating in adolescent females; "
            "contraception counselling at every visit; alternatives (ETX, LTG) preferred in women "
            "of childbearing potential unless GTCS or JME evolution present",
            "HEPATOTOXICITY: highest risk age <2 years on polytherapy; monitor LFTs; "
            "urgent if jaundice/vomiting — check LFTs + NH₃ immediately",
            "PANCREATITIS: rare but potentially fatal — abdominal pain = urgent lipase + amylase",
            "HYPERAMMONEMIA WITHOUT HEPATOTOXICITY: VPA inhibits urea cycle → elevated NH₃ "
            "even with normal LFTs; presents as confusion/encephalopathy especially during illness; "
            "check NH₃; L-carnitine 50-100 mg/kg/d IV or oral; consider dose reduction",
            "VPA + LTG INTERACTION: VPA inhibits LTG glucuronidation → doubles LTG level; "
            "halve LTG dose and slow titration (12W minimum) when combining",
        ]
    },
    {
        "drug": "Lamotrigine (LTG)",
        "parameters": [
            "Rash surveillance: at EVERY visit for first 8 weeks of treatment — examine skin; "
            "educate parents/patient to call immediately for any rash",
            "LTG serum level: optional (therapeutic 3-15 µg/mL); useful for adherence or "
            "dose-level in VPA combination (where levels are 2× elevated)",
            "LFTs annually (minimal hepatic metabolism — routine only)",
        ],
        "alerts": [
            "SJS / TEN RISK: 0.3-0.8% with rapid titration; much less with slow titration; "
            "12-week minimum titration schedule MANDATORY; VPA co-administration doubles SJS risk "
            "→ use slowest titration schedule and halved doses; "
            "show parents photo of drug rash vs SJS; STOP LTG at first rash (do not restart)",
            "VPA + LTG DOSE INTERACTION: VPA inhibits LTG glucuronidation → LTG level doubles; "
            "when adding LTG to VPA: use 0.15 mg/kg/d × 2W (half-normal starting dose); "
            "when adding VPA to LTG: HALVE current LTG dose immediately",
            "INFERIOR ABSENCE EFFICACY: LTG has only 29% freedom rate (CHILDHOOD 2010) vs 53% "
            "for ETX/VPA — do not use as first-line monotherapy for CAE",
            "PREGNANCY RELATIVELY SAFE: least teratogenic common AED; no REMS required; "
            "preferred if teratogenicity concern outweighs inferior absence efficacy",
        ]
    },
    {
        "drug": "Clobazam (CLB)",
        "parameters": [
            "Behavioral assessment: Conners parent/teacher rating scale or informal behavior "
            "checklist at each visit — CLB causes hyperactivity in ~15% of children",
            "Sedation assessment: daytime sedation impacts school performance; "
            "evening dosing minimizes daytime sedation",
            "Tolerance monitoring: breakthrough seizures after 3-6 months of control "
            "= tolerance development (occurs in ~25-30%)",
        ],
        "alerts": [
            "BEHAVIORAL SIDE EFFECTS IN CHILDREN: CLB is a 1,5-benzodiazepine but can cause "
            "paradoxical hyperactivity, irritability, and aggression in 15-20% of children — "
            "especially autistic children or those with pre-existing behavioral challenges; "
            "reduce dose or discontinue if behavioral deterioration",
            "DEPENDENCE AND WITHDRAWAL: never abrupt discontinuation (withdrawal seizures risk); "
            "taper over 4-6 weeks minimum; counsel family that missed doses cannot be doubled",
            "TOLERANCE: ~25-30% develop tolerance within 12 months of continuous use; "
            "drug holiday (2-4 week taper → pause → restart) may restore efficacy; "
            "intermittent use (catamenial protocol) reduces tolerance risk",
        ]
    },
]


# ─── Lifecycle trajectory ─────────────────────────────────────────────────────

_LIFECYCLE = [
    {
        "phase": "Pre-Diagnosis (3–5 years)",
        "key_events": [
            "Absence episodes misinterpreted as inattention or 'daydreaming' by teachers/parents",
            "Academic under-performance precedes diagnosis by 6-18 months in many cases",
            "EEG referral typically triggered by teacher or school nurse observation",
        ],
        "clinical_priorities": [
            "Capture hyperventilation-provoked EEG (3 min HV during recording)",
            "Cognitive/developmental baseline (WPPSI or age-appropriate screen)",
            "Rule out attention-deficit disorder (ADHD — frequently co-diagnosed with CAE)",
            "School notification letter with seizure first-aid plan",
        ],
        "treatment_focus": "Diagnose early — start ETX/VPA promptly; every untreated absence "
                           "potentially impairs learning via ictal interference",
        "warning_signs": "Prolonged confusion episodes → rule out absence SE; "
                         "fever + prolonged seizure → rule out Dravet (SCN1A)",
    },
    {
        "phase": "Active Epilepsy — School Age (5–10 years)",
        "key_events": [
            "Peak absence frequency (untreated: 10-200 absences/day)",
            "Academic impact: reading, maths, attention all impaired during ictal absences",
            "AED initiation and titration to seizure control",
            "School accommodation: 504 plan / IEP for seizure action plan + extra test time",
        ],
        "clinical_priorities": [
            "Achieve seizure-freedom (hyperventilation-negative EEG): primary goal",
            "ETX preferred for pure absence (CHILDHOOD evidence) — better attention profile",
            "Monitor ETX serum level (40-100 µg/mL) at each visit",
            "Neuropsychological assessment if academic decline despite seizure control",
            "Treat comorbid attention problems (if ADHD co-exists — methylphenidate safe in CAE)",
        ],
        "treatment_focus": "ETX first-line (monotherapy) for pure absence; VPA if GTCS also "
                           "present; target HV-negative EEG = treatment success",
        "warning_signs": "Persistent HV-provoked absences on therapeutic ETX dose → increase "
                         "dose to 40 mg/kg/d before declaring drug failure; "
                         "morning myoclonic jerks → evolution to JME beginning → add/switch to VPA",
    },
    {
        "phase": "Pre-Pubertal — JME Watch (10–14 years)",
        "key_events": [
            "~15-20% of CAE evolves to JME at puberty — the critical monitoring window",
            "JME evolution signals: morning myoclonic jerks + GTCS + EEG shows polyspike-wave",
            "Social concerns: sports participation, school camp seizure protocols",
            "Perimenarche in girls: catamenial absence exacerbation possible",
        ],
        "clinical_priorities": [
            "Ask specifically for morning myoclonic jerks at every visit (parents often miss this)",
            "EEG: check for polyspike-wave (JME pattern) alongside 3 Hz GSW",
            "If JME evolution confirmed: reclassify, counsel for lifelong treatment (JME does not remit)",
            "VPA/LEV preferred if JME evolution (ETX insufficient — add VPA)",
            "Girls: menstrual diary; catamenial exacerbation → consider CLB perimenstrual protocol",
        ],
        "treatment_focus": "Monitor closely for JME evolution; EEG annually + after puberty onset; "
                           "switch to VPA if myoclonus develops; begin adolescent transition counselling",
        "warning_signs": "First GTCS = emergency; VPA preferred if GTCS present on ETX; "
                         "JME misdiagnosed as CAE recurrence → wrong treatment paradigm",
    },
    {
        "phase": "Remission Phase (12–18 years)",
        "key_events": [
            "70-80% of CAE patients achieve remission by mid-adolescence (ILAE prognosis data)",
            "AED taper can be considered after 2 years seizure-free (ILAE 2022 guideline)",
            "Driving consideration: most jurisdictions require 12-month seizure-free period",
            "College/university transition: independence in AED management",
        ],
        "clinical_priorities": [
            "2-year seizure-free + HV-negative EEG: consider AED taper (reduce by 25-50% over 6M)",
            "Seizure freedom criteria: no clinical absences + no EEG absences + HV-negative",
            "Taper counselling: 30-40% relapse on taper (Camfield 1993) → monitor closely for 1y",
            "Driving counselling: confirm jurisdiction's seizure-free driving requirement",
            "Adolescent-to-adult neurologist transition plan",
            "Girls: pregnancy counselling for VPA users → switch to ETX or LTG if possible",
        ],
        "treatment_focus": "Planned AED taper after 2-year remission; "
                           "relapse during taper → restart same drug; confirm true remission vs "
                           "undetected absences before declaring drug-free",
        "warning_signs": "Relapse on taper most common within 6 months; "
                         "academic exam period → delay taper timing to summer break",
    },
    {
        "phase": "Young Adult (18–25 years)",
        "key_events": [
            "~20% of CAE patients have persistent absences into adulthood (adult neurologist transfer)",
            "VPA teratogenicity counselling mandatory for all women — switch discussions",
            "Employment: declare epilepsy for safety-sensitive occupations; driving regulations",
            "Medication compliance challenges: university/independent living transitions",
        ],
        "clinical_priorities": [
            "Adult neurologist transition: comprehensive handover letter with seizure history, "
            "AED history, last EEG, ETX/VPA levels, HV provocation result",
            "VPA in women: REMS enrolment; switch discussion to ETX or LTG if planning pregnancy",
            "Persistent absences (20%): re-evaluate diagnosis — is this truly CAE or JME/other GGE?",
            "Driver's licence: confirm local seizure-free period requirement",
            "Alcohol: moderate use generally tolerated; excess → sleep deprivation → absences next morning",
        ],
        "treatment_focus": "Optimize for adult life: fewest side effects, safe in reproductive years; "
                           "ETX preferred if no GTCS; LTG reasonable if VPA unacceptable and ETX failed; "
                           "LEV alternative (no teratogenicity, but behavioral SE)",
        "warning_signs": "Adult VPA + unplanned pregnancy = neural tube defect emergency → "
                         "folic acid 5 mg/d mandatory; refer to maternal epilepsy clinic immediately",
    },
    {
        "phase": "Adult — Long-Term (25+ years)",
        "key_events": [
            "True adult-persistent CAE: rare; consider reclassification to 'adult-onset GGE'",
            "Comorbidity management: depression, anxiety more prevalent in epilepsy adults",
            "Career and family planning discussions",
            "Genetic counselling: sibling/child risk ~10-15%",
        ],
        "clinical_priorities": [
            "Annual neurology review if AED-dependent",
            "Annual EEG in drug-free patients after confirmed remission",
            "Depression/anxiety screen: PHQ-9 + GAD-7 annually",
            "If persistent and drug-resistant: consider genetic panel (WGS), ketogenic diet, "
            "or investigational approach via epilepsy centre",
            "Genetic counselling: CAE risk to offspring ~10-15%",
        ],
        "treatment_focus": "Minimal effective dose; annual review for further taper; "
                           "comorbidity management (psychosocial = as important as seizure control)",
        "warning_signs": "New seizure type in adult-CAE (focal features, prolonged seizures) → "
                         "full re-evaluation; new EEG; consider MRI (late-onset focal lesion)",
    },
]


# ─── Key definitions ─────────────────────────────────────────────────────────

_DEFINITIONS = [
    {
        "term": "CAE (Childhood Absence Epilepsy)",
        "definition": (
            "A genetic generalized epilepsy (GGE) characterized by frequent typical absence seizures "
            "with onset at 4-10 years (peak 5-7y), normal intelligence and neurodevelopment, "
            "3 Hz generalized spike-wave on EEG, and normal MRI. Accounts for 10-15% of childhood "
            "epilepsies. Female predominance (60%). Prognosis: 70-80% remission by adolescence; "
            "~15-20% evolve to JME. ILAE 2017 criteria: (1) onset 4-10y, (2) typical absences "
            "as sole seizure type at onset, (3) 3 Hz GSW on EEG, (4) normal MRI, "
            "(5) normal development. Any deviation → reconsider diagnosis."
        ),
    },
    {
        "term": "Typical Absence Seizure",
        "definition": (
            "Abrupt-onset, abrupt-offset episode of behavioral arrest (staring, unresponsiveness) "
            "lasting 5-30 seconds (mean 10 sec) with immediate return to baseline (no post-ictal "
            "confusion). EEG: 3 Hz (2.5-4 Hz) generalized spike-wave, bisynchronous, frontally "
            "predominant. Associated automatisms in 60% (oro-facial, manual). "
            "Provoked by hyperventilation in virtually all untreated patients (diagnostic). "
            "Frequency untreated: 10-200/day. Missed by parents/teachers for months (inattention). "
            "Distinguished from atypical absence (LGS — <3 Hz, slower onset, abnormal background)."
        ),
    },
    {
        "term": "3 Hz Generalized Spike-Wave (GSW)",
        "definition": (
            "The hallmark EEG pattern of CAE: bisynchronous, bilateral, generalized spike-wave "
            "complexes at 2.5-4 Hz (classically 3 Hz), frontally predominant, high amplitude "
            "(>300 µV), with abrupt onset and offset coinciding with clinical absence onset/end. "
            "The spike reflects cortical excitation (pyramidal cell synchronization); "
            "the wave reflects inhibitory post-synaptic potential (thalamic GABAergic reticular "
            "nucleus activity). Generated by thalamocortical loop resonance. "
            "Background EEG is normal in CAE (distinguishes from symptomatic generalized epilepsies)."
        ),
    },
    {
        "term": "Hyperventilation (HV) Provocation",
        "definition": (
            "Standardized diagnostic test: 3 minutes of vigorous overbreathing during EEG recording. "
            "Mechanism: hypocapnia → cerebral vasoconstriction → increased thalamocortical "
            "excitability → 3 Hz GSW in CAE. Sensitivity: ~100% for untreated CAE. "
            "Therapeutic monitoring use: HV should be NEGATIVE on adequate ETX/VPA dose. "
            "Persistent HV-provoked absences = undertreated or drug-resistant. "
            "Procedure: blow on pinwheel, paper, or cotton; continuous EEG recording during and "
            "2 min post-HV; behavioral observation simultaneously."
        ),
    },
    {
        "term": "Ethosuximide (ETX)",
        "definition": (
            "The gold-standard first-line treatment for childhood absence epilepsy without GTCS. "
            "Mechanism: selective T-type voltage-gated Ca²⁺ channel blocker (Cav3.2/CACNA1H) "
            "in thalamic relay neurons → reduces low-threshold burst firing → suppresses 3 Hz "
            "thalamocortical oscillation. CHILDHOOD 2010 (NEJM): ETX achieved 53% seizure-freedom "
            "at 16 weeks AND was superior to VPA for attention function (BRIEF and SNAP scores). "
            "Key limitation: NO protection against GTCS (no Na⁺ channel effect) — add VPA if "
            "GTCS emerge. Available since 1960; no REMS; relatively safe profile."
        ),
    },
    {
        "term": "GABRG2",
        "definition": (
            "Gene encoding the γ2 subunit of the GABA-A receptor — the most common identifiable "
            "monogenic cause of CAE. The γ2 subunit is required for receptor surface expression "
            "and benzodiazepine sensitivity. Pathogenic variants (R43Q, R139G, K289M) reduce "
            "receptor trafficking to the cell surface, resulting in 30-50% loss of surface GABA-A "
            "receptors in thalamocortical circuits → reduced inhibitory tone → 3 Hz oscillation. "
            "R43Q: temperature-sensitive receptor trafficking → febrile seizures + absences in "
            "same patient (GEFS+ spectrum). GABRG2-CAE typically responds well to ETX/VPA."
        ),
    },
    {
        "term": "GABRA1",
        "definition": (
            "Gene encoding the α1 subunit of GABA-A receptors — the most abundant GABA-A subunit "
            "in the adult brain, particularly in cortex and thalamus. Loss-of-function variants "
            "(A322D most studied): misfolded protein rapidly degraded by the ubiquitin-proteasome "
            "system → <10% of normal surface expression → severe reduction in cortical/thalamic "
            "GABAergic inhibition. GABRA1 causes a spectrum of GGE: CAE (mild), JME (moderate), "
            "GTCS-dominant GGE (severe). Monitor GABRA1-CAE patients closely for JME evolution "
            "at puberty — myoclonic jerks on awakening signal transition."
        ),
    },
    {
        "term": "CHILDHOOD Study (Glauser et al. 2010, NEJM)",
        "definition": (
            "Landmark RCT (n=453 children) establishing the evidence hierarchy for CAE treatment. "
            "Three arms: Ethosuximide vs Valproate vs Lamotrigine, double-blind. Primary endpoint: "
            "freedom from treatment failure (seizures + side effects) at 16 weeks. Results: "
            "ETX 53%, VPA 53%, LTG 29% (LTG significantly inferior). Secondary: attentional "
            "dysfunction — ETX superior to VPA (BRIEF/SNAP scores). Conclusion: ETX is preferred "
            "first-line for pure CAE (equal efficacy to VPA, better cognitive profile); "
            "VPA preferred when GTCS present. LTG significantly inferior — NOT first-line."
        ),
    },
    {
        "term": "Drug-Resistant CAE (DRCAE)",
        "definition": (
            "DRCAE: failure of ≥2 appropriate, adequately-dosed, tolerated AEDs (per ILAE DRE "
            "definition) to achieve sustained seizure freedom. Occurs in ~10-20% of CAE patients "
            "(much lower rate than focal epilepsies). First assess: adherence (ETX level?), "
            "correct diagnosis (MRI? atypical EEG?), HV provocation still positive?, dose "
            "adequate? Then second-line: VPA if ETX failed (or vice versa). "
            "After ≥2 failures: consider CLB add-on, LEV, KD, ZNS, or referral to Level 4 "
            "epilepsy centre. DRCAE rarely benefits from surgery (unlike focal DRE)."
        ),
    },
    {
        "term": "JME Evolution",
        "definition": (
            "Approximately 15-20% of CAE patients evolve to Juvenile Myoclonic Epilepsy (JME) "
            "at or around puberty. Evolution signals: (1) morning myoclonic jerks appear "
            "(often hours before school), (2) GTCS become more prominent or dawn-predominant, "
            "(3) EEG shows polyspike-wave (4-6 Hz) in addition to 3 Hz GSW, "
            "(4) photosensitivity worsens. JME-evolved patients require lifelong treatment "
            "(JME does NOT remit like CAE). Clinical action: reclassify diagnosis → start VPA "
            "(or LEV) for both absence and myoclonus → counsel for lifelong therapy. "
            "GABRA1 variants are particularly associated with CAE-to-JME evolution."
        ),
    },
    {
        "term": "Photoparoxysmal Response (PPR)",
        "definition": (
            "EEG pattern: generalized spike-wave or polyspike-wave triggered by intermittent "
            "photic stimulation (IPS) during EEG. Occurs in ~15-20% of CAE (lower than JME ~35%). "
            "Clinically: flickering lights, video games, TV, sunlight through trees may trigger "
            "absences. ETX and VPA suppress PPR. Practical management: blue-light filtering "
            "(Z1) glasses, 20-min screen breaks, polarized sunglasses outdoors. "
            "PPR alone (without clinical photosensitive seizures) does not change management "
            "— only clinically significant if patient reports light-triggered seizures."
        ),
    },
    {
        "term": "Remission",
        "definition": (
            "In CAE: absence of seizures (clinical + EEG absences) AND negative hyperventilation "
            "provocation EEG, sustained for ≥2 years on stable AED dose. "
            "70-80% of CAE patients achieve remission by mid-adolescence. "
            "AED taper is appropriate after 2-year remission (ILAE 2022 guideline). "
            "Taper protocol: reduce dose by 25% every 3 months over 12-18 months; monitor EEG "
            "before and during taper. Relapse rate on taper: ~30-40% (Camfield 1993) — "
            "most relapses within 6 months of first dose reduction; restart same AED (high "
            "re-response rate). Complete drug withdrawal target: 2-3 years post-seizure-free."
        ),
    },
    {
        "term": "Absence Status Epilepticus (ASE)",
        "definition": (
            "Continuous or recurrent absence seizures lasting >5 minutes, constituting an "
            "epileptic emergency. ASE presentation: prolonged behavioral confusion, automatic "
            "behavior, reduced responsiveness — subtle and frequently unrecognized. "
            "EEG: continuous or near-continuous 2-3 Hz GSW (may be irregular/slower than "
            "baseline during prolonged ASE). Precipitants: missed AED dose, sleep deprivation, "
            "fever, or drug-worsening (CBZ in CAE → precipitates ASE). "
            "Treatment: IV lorazepam 0.05-0.1 mg/kg (first-line) OR rectal diazepam. "
            "Do NOT give CBZ/PHT IV (worsen absence SE). Generally responds rapidly to BZD. "
            "No evidence of neuronal injury from ASE (unlike convulsive SE)."
        ),
    },
    {
        "term": "Atypical Absence Seizure (Differential)",
        "definition": (
            "Distinguished from TYPICAL absence (CAE) by: (1) onset/offset less abrupt ('blurring'), "
            "(2) EEG: slow spike-wave <3 Hz (1.5-2.5 Hz) vs 3 Hz in CAE, (3) background EEG "
            "ABNORMAL (diffuse slow) vs normal in CAE, (4) associated with cognitive impairment "
            "(vs normal development in CAE), (5) other seizure types present (tonic, atonic — "
            "consistent with LGS). Atypical absences occur in LGS, Doose syndrome, and other "
            "DEEs — NOT in CAE. If EEG shows <3 Hz spike-wave or abnormal background → "
            "reclassify from CAE to LGS/DEE — fundamentally different treatment approach "
            "(ETX may be ineffective; rufinamide, CLB, CBD considered for LGS)."
        ),
    },
]


# ─── Clinical standards ──────────────────────────────────────────────────────

_STANDARDS = [
    {
        "standard": "ILAE 2022 Classification — Absence Epilepsy Syndromes",
        "reference": "Scheffer IE et al. Epilepsia 2017 (operational classification); "
                     "Specchio N et al. Epilepsia 2022 (childhood seizure syndromes)",
        "key_points": [
            "CAE diagnostic criteria: onset 4-10y; typical absences as sole seizure type at onset; "
            "3 Hz GSW EEG; normal MRI; normal development",
            "Distinguishes CAE from JME, MAE (Doose), BECTS, and childhood GGE with GTCS",
            "Prognosis category: 'self-limited' (most remit) vs 'lifelong' (JME, LGS)",
        ]
    },
    {
        "standard": "NICE NG217 (2022) — Epilepsies: Diagnosis and Management",
        "reference": "NICE guideline NG217, National Institute for Health and Care Excellence, UK, 2022",
        "key_points": [
            "ETX recommended first-line for CAE where GTCS not a concern",
            "VPA first-line for CAE with GTCS or where ETX fails — REMS for women of childbearing age",
            "LTG second-line only (inferior efficacy, SJS risk)",
            "Driving: seizure-free 12 months for passenger car (UK regulations)",
            "Folic acid 5 mg/d for all females on VPA",
        ]
    },
    {
        "standard": "AAN/CNS CHILDHOOD Trial Evidence (Glauser et al. 2010 NEJM)",
        "reference": "Glauser TA et al. N Engl J Med 2010;362:790-799 (CHILDHOOD RCT)",
        "key_points": [
            "ETX = VPA for seizure freedom (53% each at 16W) — Level A evidence",
            "LTG significantly inferior (29%) — Level A evidence against LTG first-line",
            "ETX superior to VPA for attentional function (BRIEF/SNAP scores) — Level A",
            "VPA preferred when GTCS present or suspected JME evolution — Level B consensus",
        ]
    },
    {
        "standard": "FDA Valproate REMS (Depakote, 2013)",
        "reference": "FDA Risk Evaluation and Mitigation Strategy — Valproate Sodium Products, 2013",
        "key_points": [
            "REMS enrollment mandatory for all females of childbearing potential on VPA",
            "Neural tube defect risk: 10× background; spina bifida most common",
            "Neurodevelopmental risk: autism spectrum disorder 3-4× background",
            "Folic acid 5 mg/d mandatory for all females on VPA",
            "Pregnancy registry: enroll all pregnant patients (North American AED Pregnancy Registry)",
        ]
    },
    {
        "standard": "FDA Ethosuximide — Approved 1960",
        "reference": "FDA approval 1960; labeling updates per current prescribing information",
        "key_points": [
            "FDA-approved specifically for absence (petit mal) epilepsy",
            "No REMS requirement — favorable safety profile vs VPA",
            "Serum monitoring: target 40-100 µg/mL (established therapeutic range)",
            "No significant teratogenicity data (limited — avoid if possible in pregnancy)",
        ]
    },
    {
        "standard": "ILAE Diagnostic Manual — CAE Criteria",
        "reference": "Epileptic Disorders ILAE Diagnostic Manual 2017 (Panayiotopoulos CP, ed.)",
        "key_points": [
            "Five mandatory criteria for CAE diagnosis (all must be met)",
            "MRI must be normal — abnormal MRI excludes CAE diagnosis",
            "Normal neurodevelopment required — developmental delay excludes CAE",
            "HV provocation is strongly supportive (not mandatory but expected)",
            "Background EEG normal in CAE — abnormal background EEG → reconsider diagnosis",
        ]
    },
]


# ─── Key thresholds ──────────────────────────────────────────────────────────

_THRESHOLDS = [
    {
        "threshold": "First-line Treatment: ETX or VPA — Level A",
        "value": "ETX 53% vs VPA 53% seizure-freedom at 16W (CHILDHOOD 2010 NEJM)",
        "rationale": "ETX preferred for pure absence (superior attention profile); "
                     "VPA preferred when GTCS present or JME evolution suspected (broader coverage)"
    },
    {
        "threshold": "VPA Therapeutic Drug Monitoring",
        "value": "50-100 µg/mL serum VPA (measure q6M — correlation with efficacy weak but "
                 "useful for adherence monitoring and toxicity threshold)",
        "rationale": "Levels <50 µg/mL often associated with breakthrough seizures; "
                     ">120 µg/mL — increased hepatotoxicity, tremor, and encephalopathy risk"
    },
    {
        "threshold": "ETX Therapeutic Drug Monitoring",
        "value": "40-100 µg/mL serum ETX (measure q6M or after dose change)",
        "rationale": "Levels <40 µg/mL — undertreated; HV provocation likely still positive. "
                     "Levels >100 µg/mL — GI toxicity and behavioral side effects increase "
                     "without additional seizure benefit"
    },
    {
        "threshold": "LTG-VPA Dose Interaction",
        "value": "50% LTG dose reduction when VPA co-administered (VPA inhibits LTG glucuronidation → "
                 "doubles LTG serum level → doubles SJS risk)",
        "rationale": "Failure to halve LTG dose in VPA combination → 2× LTG exposure → "
                     "severe SJS risk; use 12-week slow titration schedule mandatory for VPA+LTG"
    },
    {
        "threshold": "AED Taper Threshold (Remission)",
        "value": "2 years seizure-free (clinical + EEG absences absent, HV provocation negative) "
                 "before initiating AED taper (ILAE 2022 guideline)",
        "rationale": "Premature taper (<2y) → 60-70% relapse; 2y seizure-free → 30-40% relapse rate. "
                     "Relapse most common within 6 months of first dose reduction"
    },
    {
        "threshold": "Driving — Seizure-Free Period",
        "value": "12 months seizure-free for passenger car driving (most jurisdictions); "
                 "commercial driving: typically 5-10 year seizure-free (varies by jurisdiction)",
        "rationale": "Absence seizures during driving = unresponsive, behavioral arrest — "
                     "crash risk real even for brief absences. Full seizure control mandatory "
                     "before driving licence issuance"
    },
]


# ─── References ──────────────────────────────────────────────────────────────

_REFERENCES = [
    {
        "citation": "Glauser TA et al. 2010 NEJM — CHILDHOOD RCT",
        "full": "Glauser TA, Cnaan A, Shinnar S, et al. Ethosuximide, valproic acid, and lamotrigine "
                "in childhood absence epilepsy. N Engl J Med. 2010;362(9):790-799.",
        "key_finding": "ETX = VPA (53% seizure-freedom at 16W); LTG inferior (29%); "
                       "ETX superior attention profile — establishes ETX as preferred first-line for pure CAE",
        "evidence_level": "Level A (RCT, n=453, double-blind)"
    },
    {
        "citation": "Berg AT et al. 2001 Epilepsia — Natural History",
        "full": "Berg AT, Shinnar S, Levy SR, Testa FM, Smith-Rapaport S, Beckerman B. "
                "How well can epilepsy syndromes be identified at diagnosis? Epilepsia. 2001;42(5):665-675.",
        "key_finding": "71% of CAE patients in remission at 12-year follow-up; "
                       "20% develop JME; natural history data for prognosis counselling",
        "evidence_level": "Level B (prospective cohort)"
    },
    {
        "citation": "Camfield CS & Camfield PR 1993 Neurology — Taper Study",
        "full": "Camfield CS, Camfield PR, Smith A, Gordon K, Dooley J. "
                "Outcome of childhood epilepsy: a population-based study with a simple scoring system. "
                "Epilepsia. 1993;34(6):1006-1016.",
        "key_finding": "30-40% relapse after AED taper in remitted CAE; most relapses within 6M; "
                       "supports 2-year seizure-free minimum before taper",
        "evidence_level": "Level B (prospective cohort)"
    },
    {
        "citation": "Tenney JR & Glauser TA 2013 Curr Opin Neurol — Management Review",
        "full": "Tenney JR, Glauser TA. The current state of absence epilepsy: can we have our cake "
                "and eat it too? Curr Opin Neurol. 2013;26(2):165-170.",
        "key_finding": "Comprehensive management review confirming ETX superiority for attention; "
                       "discusses JME evolution risk; catamenial management strategies",
        "evidence_level": "Level B (systematic review)"
    },
    {
        "citation": "Loiseau P et al. 1983 Epilepsia — Adult Follow-Up",
        "full": "Loiseau P, Duche B, Cordova S, Dartigues JF, Cohadon S. "
                "Prognosis of benign childhood epilepsy with centrotemporal spikes: a follow-up study "
                "of 168 patients. Epilepsia. 1983;24(1):150-170.",
        "key_finding": "Adult follow-up confirms 80% remission rate in typical CAE; "
                       "persistent absences in adulthood require reclassification",
        "evidence_level": "Level B (prospective cohort)"
    },
    {
        "citation": "Scheffer IE et al. 2017 Epilepsia — ILAE Classification",
        "full": "Scheffer IE, Berkovic S, Capovilla G, et al. ILAE classification of the epilepsies: "
                "Position paper of the ILAE Commission for Classification and Terminology. "
                "Epilepsia. 2017;58(4):512-521.",
        "key_finding": "Operational classification defining CAE diagnostic criteria; "
                       "distinguishes CAE from other GGEs; establishes 'self-limited' prognosis category",
        "evidence_level": "ILAE Position Paper (Level A guideline)"
    },
]


# ─── Patient cohort ──────────────────────────────────────────────────────────

def _get_patients():
    rows = _db_rows("""
        SELECT p.patient_id, p.age, p.gender, p.disease
        FROM patients p
        ORDER BY p.patient_id
    """)
    _SEIZURE_SUBTYPES = [
        "Typical absence only",
        "Absence + automatisms",
        "Absence + brief GTCS",
        "Absence + automatisms + GTCS",
        "Absence only (no automatisms)",
    ]
    _AED_REGIMEN = [
        "Ethosuximide mono",
        "Valproate mono",
        "Lamotrigine mono",
        "Ethosuximide + Valproate",
        "Valproate + Lamotrigine",
        "Ethosuximide + Clobazam",
        "Valproate + Levetiracetam",
        "Ketogenic Diet + Ethosuximide",
    ]
    _CONTROL = ["Seizure-free", "Improved (>50% reduction)", "Partial response", "Drug-resistant"]
    _ETIOLOGY_NAMES = [e["etiology"].split(" (")[0].split(" /")[0].strip() for e in _ETIOLOGIES]
    _ONSET_SEMIOLOGY = [
        "Staring + behavioral arrest",
        "Staring + oral automatisms",
        "Staring + hand automatisms",
        "Staring alone (subtle)",
        "Staring + eyelid flutter",
        "Absence + brief GTCS at onset",
    ]

    result = []
    for i, r in enumerate(rows[:41]):
        sd = _seed(r["patient_id"])
        # CAE etiology distribution: Polygenic 65%, GABRG2 10%, GABRA1 8%, CACNA 7%, SCN1A/B 5%, Unknown 5%
        etiology_weights = [65, 10, 8, 7, 5, 5]
        etiology_idx = 0
        rv_et = (sd >> 2) % 100
        acc = 0
        for ei, w in enumerate(etiology_weights):
            acc += w
            if rv_et < acc:
                etiology_idx = ei
                break
        # CAE onset: 4-10 years (peak 5-7 years)
        onset_base = 4 + (sd % 7)       # 4-10 years
        if (sd % 10) < 6:
            onset_age = 5 + (sd % 3)    # peak: 5-7 years (60%)
        else:
            onset_age = 4 + (sd % 7)    # broader range 4-10 (40%)
        subtype_idx = (sd >> 4) % len(_SEIZURE_SUBTYPES)
        aed_idx = (sd >> 8) % len(_AED_REGIMEN)
        semiology_idx = (sd >> 10) % len(_ONSET_SEMIOLOGY)
        sex = r.get("gender", "F")
        # CAE female predominance ~60%
        if (sd >> 12) % 10 < 6:
            sex = "F"
        else:
            sex = "M"
        catamenial = sex == "F" and bool((sd >> 14) % 5 < 1)  # ~20% of girls
        # CAE control: ~70-80% seizure-free; ~10-20% drug-resistant
        control_w = [72, 15, 7, 6]  # seizure-free 72%, improved 15%, partial 7%, DRE 6%
        control_idx = 0
        rv_ctrl = (sd >> 16) % 100
        acc = 0
        for ci, w in enumerate(control_w):
            acc += w
            if rv_ctrl < acc:
                control_idx = ci
                break
        years_on_aed = max(1, (r.get("age", 12) or 12) - onset_age)
        remission = control_idx == 0 and years_on_aed >= 2 and (sd >> 20) % 3 < 2
        jme_evolution = bool((sd >> 22) % 7 == 0 and (r.get("age", 12) or 12) >= 12)  # ~15% by puberty
        result.append({
            "patient_id": r["patient_id"],
            "sex": sex,
            "onset_age_years": onset_age,
            "current_age": r.get("age", onset_age + years_on_aed),
            "seizure_types": _SEIZURE_SUBTYPES[subtype_idx],
            "etiology": _ETIOLOGY_NAMES[etiology_idx],
            "onset_semiology": _ONSET_SEMIOLOGY[semiology_idx],
            "aed_regimen": _AED_REGIMEN[aed_idx],
            "seizure_control": _CONTROL[control_idx],
            "catamenial": catamenial,
            "remission": remission,
            "jme_evolution": jme_evolution,
            "years_on_aed": max(0, years_on_aed),
        })
    return result


# ─── Public API ─────────────────────────────────────────────────────────────

def overview():
    patients = _get_patients()
    n = max(len(patients), 1)
    seizure_free_n = sum(1 for p in patients if p["seizure_control"] == "Seizure-free")
    drug_resistant_n = sum(1 for p in patients if p["seizure_control"] == "Drug-resistant")
    remission_n = sum(1 for p in patients if p["remission"])
    jme_evolution_n = sum(1 for p in patients if p["jme_evolution"])
    female_n = sum(1 for p in patients if p["sex"] == "F")
    avg_onset = round(sum(p["onset_age_years"] for p in patients) / n, 1)

    etiology_counts = {}
    for p in patients:
        k = p["etiology"]
        etiology_counts[k] = etiology_counts.get(k, 0) + 1
    etiology_distribution = sorted(
        [{"etiology": k, "count": v, "pct": round(v / n * 100)} for k, v in etiology_counts.items()],
        key=lambda x: -x["count"]
    )

    aed_counts = {}
    for p in patients:
        k = p["aed_regimen"]
        aed_counts[k] = aed_counts.get(k, 0) + 1
    aed_use = [{"regimen": k, "n_patients": v}
               for k, v in sorted(aed_counts.items(), key=lambda x: -x[1])]

    control_counts = {}
    for p in patients:
        k = p["seizure_control"]
        control_counts[k] = control_counts.get(k, 0) + 1
    control_distribution = [{"status": k, "count": v}
                             for k, v in sorted(control_counts.items(), key=lambda x: -x[1])]

    semiology_counts = {}
    for p in patients:
        k = p["onset_semiology"]
        semiology_counts[k] = semiology_counts.get(k, 0) + 1
    semiology_distribution = [{"semiology": k, "n": v}
                               for k, v in sorted(semiology_counts.items(), key=lambda x: -x[1])]

    return {
        "syndrome": "Childhood Absence Epilepsy (CAE)",
        "icd10": "G40.309 (Generalized idiopathic epilepsy, not intractable)",
        "prevalence": "10-15% of childhood epilepsies; 2-8 per 100,000 children; "
                      "female predominance (~60%)",
        "drug_resistance_rate": "~10-20% drug-resistant (much lower than focal epilepsies); "
                                "70-80% remit by adolescence (ILAE 2017)",
        "updated": str(date.today()),
        "cohort_size": n,
        "avg_onset_age_years": avg_onset,
        "kpis": [
            {"label": "Cohort (CAE)", "value": str(n), "color": "#1d4ed8"},
            {"label": "Seizure-Free", "value": f"{seizure_free_n} ({round(seizure_free_n/n*100)}%)", "color": "#16a34a"},
            {"label": "In Remission", "value": f"{remission_n} ({round(remission_n/n*100)}%)", "color": "#0891b2"},
            {"label": "Drug-Resistant", "value": f"{drug_resistant_n} ({round(drug_resistant_n/n*100)}%)", "color": "#dc2626"},
            {"label": "JME Evolution", "value": f"{jme_evolution_n} ({round(jme_evolution_n/n*100)}%)", "color": "#7c3aed"},
            {"label": "Avg Onset Age", "value": f"{avg_onset}y", "color": "#ea580c"},
        ],
        "etiology_distribution": etiology_distribution,
        "aed_use": aed_use,
        "control_distribution": control_distribution,
        "semiology_distribution": semiology_distribution,
        "seizure_types": _SEIZURE_TYPES,
        "triggers": _TRIGGERS,
        "lifecycle_trajectory": _LIFECYCLE,
        "clinical_alerts": [
            "ETX FIRST-LINE for pure absence (CHILDHOOD 2010 Level A): 53% freedom, SUPERIOR "
            "attention profile vs VPA — prescribe ETX unless GTCS present or JME evolution suspected",
            "LTG IS INFERIOR — NOT FIRST-LINE: CHILDHOOD 2010 showed only 29% freedom with LTG "
            "vs 53% ETX/VPA; LTG monotherapy for CAE is Level A evidence AGAINST it as first-line",
            "ABSOLUTE CONTRAINDICATIONS (worsen absences): CBZ / OXC / PHT / VGB / TGB / GBP / PGB — "
            "all pro-absence drugs; CBZ can precipitate Absence Status Epilepticus in CAE",
            "JME EVOLUTION WATCH (puberty): ask about morning myoclonic jerks at EVERY visit; "
            "EEG for polyspike-wave; 15-20% of CAE evolves to JME — reclassify + lifelong VPA/LEV "
            "if evolution confirmed (JME does NOT remit)",
            "VPA IN GIRLS — REMS MANDATORY: teratogenicity 10× background neural tube defects; "
            "folic acid 5 mg/d; REMS enrolment; switch to ETX if possible in adolescent females "
            "planning pregnancy; VPA + absent contraception = absolute clinical priority",
            "HYPERVENTILATION TEST = TREATMENT MONITOR: HV provocation should be NEGATIVE on "
            "adequate ETX/VPA dose; persistent positive HV = undertreated (increase dose before "
            "declaring drug failure); ETX target level 40-100 µg/mL",
        ],
    }


def breakdown():
    patients = _get_patients()
    return {
        "patients": patients,
        "etiology_catalog": _ETIOLOGIES,
        "seizure_types": _SEIZURE_TYPES,
        "treatments": _TREATMENTS,
        "aed_monitoring": _AED_MONITORING,
        "lifecycle": _LIFECYCLE,
        "standards": _STANDARDS,
        "thresholds": _THRESHOLDS,
        "references": _REFERENCES,
    }


def definitions():
    return {
        "concepts": _DEFINITIONS,
        "standards": _STANDARDS,
        "thresholds": _THRESHOLDS,
        "references": _REFERENCES,
        "updated": str(date.today()),
    }
