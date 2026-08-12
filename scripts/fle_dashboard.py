"""Frontal Lobe Epilepsy (FLE) Dashboard — the second most common focal epilepsy,
accounting for ~20-30% of focal epilepsies (after TLE).

Hallmarks:
  Hypermotor Seizures: explosive onset, bilateral asymmetric limb movements, vocalization,
    nocturnal predilection — the characteristic FLE seizure
  SMA Seizures (Supplementary Motor Area): tonic posturing of contralateral arm, speech arrest,
    Jacksonian march; EEG: contralateral frontal discharge
  ADNFLE (Autosomal Dominant Nocturnal FLE): CHRNA4/CHRNB2 variants; adolescent onset
EEG hallmark: frontal fast activity, low-voltage fast activity (LVFA), contralateral frontal
  discharge; scalp EEG often poorly localising in FLE (deep frontal sources)
Seizure types: Hypermotor → FAS (SMA) → FIAS (frontal automatisms) → FBTCS (rapid bilateral)
Drug resistance: ~30-40% — similar to TLE
Surgery: frontal lobectomy / SEEG-guided resection — 50-60% Engel Class I (lower than MTLE-HS)

CAUTION — Enzyme-inducing AEDs (CBZ, OXC):
  FIRST-LINE for FLE (unlike JME/Dravet where they are contraindicated)
  Reduce oral contraceptive efficacy → use non-hormonal or high-dose OCP (same as TLE)
  No specific AED contraindications unique to FLE — contrast with JME (CBZ CONTRAINDICATED)
  Perampanel (PER): AMPA antagonist — REMS monitoring required (aggression/psychiatric)
  Lacosamide (LCM): cardiac PR interval — baseline ECG mandatory

References:
  - Jeha LE et al. 2007 Epilepsia (frontal lobe resection: 56% seizure-free)
  - Englot DJ et al. 2011 J Neurosurg (predictors of frontal lobe surgery outcome)
  - Scheffer IE et al. 1995 Lancet (ADNFLE — autosomal dominant nocturnal FLE description)
  - Baulac S et al. 2015 Nat Genet (DEPDC5 mTOR pathway FLE)
  - Wirrell EC et al. 2022 Epilepsia (ILAE FLE syndrome classification)
  - Bernhardt BC et al. 2016 Nat Rev Neurol (FCD and cortical malformations)
Data: live clinical.db (41 epilepsy patients, deterministic FLE overlay)
      + curated FLE pharmacology / etiology / seizure-type / trigger catalogs."""

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


# ─── Etiology / structural-genetic catalog ───────────────────────────────────

_ETIOLOGIES = [
    {
        "etiology": "Focal Cortical Dysplasia (FCD)",
        "category": "Structural",
        "pct": 35,
        "mechanism": (
            "Malformation of cortical development affecting the frontal lobe — the most common "
            "structural cause of FLE. FCD Type IIb: balloon cells + dysmorphic neurons with "
            "somatic mTOR pathway mutations (MTOR, TSC1/TSC2, DEPDC5). FCD Type I: columnar/"
            "laminar disorganisation of prefrontal and premotor cortex. Hyperexcitable dysplastic "
            "cortex generates continuous or high-frequency interictal discharges. "
            "MRI may be subtle (blurring of grey-white junction, transmantle sign) or negative in ~30%."
        ),
        "mri_finding": "Blurring of grey-white junction, cortical thickening, transmantle sign (FCD IIb); "
                       "T2/FLAIR signal in frontal sulci; MRI-negative in 30% — 7T MRI or post-processing required",
        "surgical_outcome": "Engel Class I: 50-65% with complete FCD resection; lower if MRI-negative; "
                            "SEEG essential when scalp EEG and MRI discordant",
        "clinical_note": "SEEG mandatory when MRI-negative FLE; mTOR inhibitors (everolimus) adjunctive in "
                         "TSC-related FCD; genetic panel (DEPDC5, MTOR, TSC1/TSC2) in all surgical candidates"
    },
    {
        "etiology": "Tumor (Low-Grade Glioma, DNET, Ganglioglioma)",
        "category": "Structural",
        "pct": 18,
        "mechanism": (
            "Low-grade neoplasms in the frontal lobe — the second most common structural FLE cause. "
            "DNETs (dysembryoplastic neuroepithelial tumors) and gangliogliomas are cortical tumors "
            "arising in frontal/perisylvian cortex. Epileptogenicity arises from peritumoral cortex "
            "rather than the tumor itself in most cases. BRAF V600E in 50% of gangliogliomas; "
            "FGFR1 in DNET. Slow-growing — months to years before epilepsy diagnosis."
        ),
        "mri_finding": "Cortical/subcortical lesion in frontal lobe; often involves premotor or prefrontal "
                       "cortex; ring enhancement absent (low-grade); cystic change common in ganglioglioma",
        "surgical_outcome": "Engel Class I: 75-90% with complete lesionectomy ± cortectomy; best surgical "
                            "prognosis of all FLE etiologies if lesion is discrete and resectable",
        "clinical_note": "Excellent surgical prognosis; IDH/BRAF mutation status guides oncologic management; "
                         "yearly MRI surveillance; frontal language mapping required if near Broca/premotor areas"
    },
    {
        "etiology": "Post-Traumatic",
        "category": "Structural",
        "pct": 12,
        "mechanism": (
            "Traumatic brain injury (TBI) — particularly contusion, haematoma, and diffuse axonal injury "
            "involving frontal lobes — is a common acquired cause of FLE. Frontal lobes are vulnerable to "
            "coup-contrecoup injury. Mechanism: neuroinflammation, axonal injury, gliosis, iron deposition "
            "from haemosiderin → chronic epileptogenic focus. Latency: months to years post-injury "
            "(post-traumatic epilepsy latent period). Risk correlates with injury severity and frontal location."
        ),
        "mri_finding": "Frontal gliosis, encephalomalacia, haemosiderin deposition (T2*/SWI); "
                       "cortical contusion scars; porencephalic cyst in severe cases",
        "surgical_outcome": "Engel Class I: 45-65% with resection of gliotic scar; outcome depends on "
                            "extent of epileptogenic zone beyond visible lesion; SEEG often required",
        "clinical_note": "Document TBI history — especially military TBI, contact sport concussions, MVA. "
                         "Phenytoin/LEV as acute post-TBI prophylaxis (7 days); does NOT prevent epilepsy "
                         "development — counsel accordingly"
    },
    {
        "etiology": "Genetic/Unknown (DEPDC5, KCNT1, ADNFLE-CHRNA4/CHRNB2)",
        "category": "Genetic",
        "pct": 20,
        "mechanism": (
            "Heterogeneous genetic causes of FLE. DEPDC5 (mTOR pathway repressor — loss-of-function): "
            "most common familial focal epilepsy gene; causes FLE with or without cortical dysplasia; "
            "variable expressivity. KCNT1 (potassium channel gain-of-function): severe nocturnal FLE, "
            "often drug-resistant; quinidine (off-label) partially effective. "
            "ADNFLE (Autosomal Dominant Nocturnal FLE): CHRNA4 / CHRNB2 (nicotinic acetylcholine receptor "
            "subunits) — hypermotor nocturnal seizures; autosomal dominant; misdiagnosed as parasomnias."
        ),
        "mri_finding": "ADNFLE: typically MRI-normal (genetic channelopathy); "
                       "DEPDC5: subtle FCD in 30-40% of cases; KCNT1: often MRI-normal",
        "surgical_outcome": "ADNFLE (CHRNA4/CHRNB2): responds well to CBZ; surgery rarely needed. "
                            "DEPDC5 + FCD: surgical if MRI lesion identified (Engel I ~50%). "
                            "KCNT1: medical management; surgery limited evidence",
        "clinical_note": "Family history of nocturnal frontal lobe seizures → ADNFLE gene panel "
                         "(CHRNA4, CHRNB2, CHRNA2); DEPDC5 gene panel in all familial focal epilepsy; "
                         "mTOR pathway: consider everolimus in DEPDC5-FCD (off-label)"
    },
    {
        "etiology": "Post-Infectious/Vascular (Stroke, Cavernoma)",
        "category": "Structural",
        "pct": 10,
        "mechanism": (
            "Frontal lobe stroke (ischaemic or haemorrhagic) causes gliosis and cortical reorganisation "
            "→ post-stroke epilepsy (5-15% of stroke patients). Cavernous malformations (cavernomas) "
            "with haemosiderin deposits are epileptogenic; frontal cavernomas cause FLE. "
            "Post-infectious: herpes simplex encephalitis (less common frontal than temporal), "
            "bacterial meningitis-related frontal cortical injury, autoimmune encephalitis "
            "(anti-GABA-B, anti-CASPR2 affecting frontal lobe)."
        ),
        "mri_finding": "Stroke: frontal DWI restriction (acute) → T2/FLAIR gliosis (chronic). "
                       "Cavernoma: popcorn lesion on T2/SWI; haemosiderin ring. "
                       "Post-encephalitis: frontal T2 signal; meningeal enhancement in acute phase",
        "surgical_outcome": "Cavernoma resection: Engel Class I 70-80% if complete removal. "
                            "Post-stroke: Engel Class I 40-55%; limited by widespread injury. "
                            "Post-infectious: variable depending on extent of cortical injury",
        "clinical_note": "New-onset FLE after stroke → AED (LEV or LTG preferred; avoid CBZ in acute stroke). "
                         "Cavernoma: surgical resection if accessible and drug-resistant; radiosurgery alternative. "
                         "Annual MRI surveillance for cavernoma growth"
    },
    {
        "etiology": "Cryptogenic",
        "category": "Unknown/Cryptogenic",
        "pct": 5,
        "mechanism": (
            "No identifiable structural or genetic cause on standard 3T MRI with epilepsy protocol "
            "and current genetic panel. Presumed microscopic FCD, subtle dysplasia, or heterotopia "
            "below MRI resolution. FDG-PET and SPECT add localising value. "
            "MRI-negative FLE is the most challenging surgical case — requires SEEG for candidacy. "
            "May harbour somatic mTOR mutations detectable only on resected tissue."
        ),
        "mri_finding": "Normal 3T MRI (epilepsy protocol); FDG-PET may show frontal hypometabolism; "
                       "MEG for frontal source localisation; SPECT hyperperfusion (ictal)",
        "surgical_outcome": "Engel Class I: 30-45% with tailored resection guided by SEEG + FDG-PET concordance; "
                            "lowest of all FLE etiologies; outcome improves with multimodal concordance",
        "clinical_note": "Refer to Level 4 epilepsy centre; consider ultra-high-field 7T MRI; "
                         "FDG-PET + SEEG multimodal workup; SEEG must cover entire frontal lobe "
                         "systematically including mesial surfaces (SMA, cingulate)"
    },
]


# ─── Seizure type catalog ────────────────────────────────────────────────────

_SEIZURE_TYPES = [
    {
        "type": "Hypermotor Seizure",
        "prevalence_pct": 70,
        "description": (
            "The characteristic FLE seizure. Explosive, abrupt onset with bilateral asymmetric "
            "limb movements (thrashing, cycling, kicking), axial involvement (rocking, pelvic thrusting), "
            "vocalization (screaming, grunting), and often preserved or only partially impaired awareness. "
            "Nocturnal predilection (arising from NREM sleep). Short duration: 10–60 seconds. "
            "Multiple seizures per night common. Clusters frequent. Often misdiagnosed as "
            "parasomnias (REM sleep behavior disorder, night terrors) — video-EEG is essential."
        ),
        "eeg": "Frontal fast activity (gamma range, >30 Hz) or low-voltage fast activity (LVFA) at onset; "
               "scalp EEG often poorly localising or obscured by movement artifact; "
               "rhythmic frontal theta/delta may follow; ictal discharge may not be visible on scalp EEG "
               "(deep cingulate/SMA source); SEEG required for precise localisation",
        "clinical_tip": (
            "Key differentiator from parasomnias: hypermotor seizures are shorter (<2 min), "
            "stereotyped, may recur multiple times per night, patient may have rapid return to sleep. "
            "Night terrors: screaming + confusion + amnesia but less stereotyped motor pattern. "
            "Document video from caregivers/bed partner. Video-EEG with sleep recording is gold standard."
        ),
        "triggers": "Sleep deprivation, missed AED dose, stress, alcohol",
        "first_line_aed": "Carbamazepine / Oxcarbazepine (ADNFLE responds particularly well to CBZ)"
    },
    {
        "type": "Focal Aware Seizure (FAS) — SMA/Premotor",
        "prevalence_pct": 55,
        "description": (
            "Supplementary Motor Area (SMA) seizures: tonic posturing of contralateral arm "
            "(fencing posture — arm extended, head deviated toward raised arm), bilateral tonic posturing, "
            "speech arrest (ictal aphasia) with preserved comprehension, and Jacksonian march "
            "(motor spread from distal to proximal limb — primary motor cortex involvement). "
            "Awareness fully preserved during SMA seizures. Sudden offset with immediate recovery. "
            "Primary motor cortex FAS: clonic jerking of contralateral face/hand (Jacksonian march)."
        ),
        "eeg": "Contralateral frontal discharge (midline or parasagittal); "
               "often bilateral synchrony from SMA given bihemispheric projections; "
               "high-amplitude spike-wave or rhythmic beta activity; "
               "scalp EEG may miss mesial frontal (SMA/cingulate) source",
        "clinical_tip": (
            "SMA seizure semiology is LATERALISING: contralateral arm elevation with ipsilateral arm flexion "
            "(fencing posture) → seizure onset contralateral to raised arm. "
            "Jacksonian march: documents primary motor involvement — maps eloquent cortex. "
            "Speech arrest: if patient cannot speak but understands, seizure involves dominant frontal operculum/SMA. "
            "Document exact posture with video for presurgical mapping."
        ),
        "triggers": "Sleep deprivation, missed AED dose, fatigue, stress",
        "first_line_aed": "Carbamazepine / Lamotrigine / Levetiracetam"
    },
    {
        "type": "Focal Impaired Awareness Seizure (FIAS) — Frontal Automatisms",
        "prevalence_pct": 40,
        "description": (
            "Frontal automatisms differ from temporal automatisms: rocking, cycling leg movements, "
            "bipedal automatisms, bimanual activity, and sexual automatisms. "
            "Duration shorter than TLE-FIAS (typically 10–60 seconds vs 1–3 minutes in TLE). "
            "Key FLE feature: NO postictal confusion (or very brief — <30 seconds) — patient "
            "returns rapidly to normal. Frontal lobe does not modulate consciousness as deeply as "
            "hippocampo-temporal circuits. Vocalisation common (grunting, repetitive words)."
        ),
        "eeg": "Bilateral frontal discharge; may show bifrontal spike-wave or frontal rhythmic activity; "
               "ictal EEG often bilateral even with unilateral frontal onset (bihemispheric frontal connections); "
               "post-ictal EEG: rapid return to normal (no post-ictal delta — distinguishes from TLE-FIAS)",
        "clinical_tip": (
            "Absent or very brief postictal confusion is a KEY CLINICAL SIGN distinguishing FLE-FIAS "
            "from TLE-FIAS. In TLE, postictal confusion typically lasts 2–30 minutes. "
            "Frontal automatisms (cycling, rocking) are more proximal/axial than temporal automatisms "
            "(lip smacking, hand fumbling). Preserved or rapid return of awareness post-ictally = FLE."
        ),
        "triggers": "Stress, sleep deprivation, alcohol, missed AED dose, fatigue",
        "first_line_aed": "Carbamazepine / Oxcarbazepine / Levetiracetam"
    },
    {
        "type": "Focal to Bilateral Tonic-Clonic (FBTCS)",
        "prevalence_pct": 50,
        "description": (
            "Focal frontal onset spreading to bilateral cortical involvement → generalised tonic-clonic. "
            "Rapid secondary generalisation (faster than TLE — frontal lobes have direct commissural "
            "connections via corpus callosum). Often preceded by hypermotor or SMA seizure for seconds. "
            "Highest SUDEP risk event type. Postictal: 15–60 min confusion, headache, muscle soreness, "
            "Todd's paresis (contralateral hemiparesis 5–10%). Nocturnal FBTCS from hypermotor onset "
            "is particularly dangerous (unwitnessed, prone position)."
        ),
        "eeg": "Focal frontal ictal onset → rapid bilateral spread via corpus callosum; "
               "bilateral synchrony; postictal generalised suppression (PGES) after FBTCS; "
               "postictal delta over frontal region; PGES duration correlates with SUDEP risk",
        "clinical_tip": (
            "Frontal FBTCS generalise FASTER than temporal FBTCS — shorter warning time. "
            "Nocturnal FBTCS from hypermotor onset is highest SUDEP risk configuration. "
            "PGES (postictal generalised EEG suppression) >50 seconds is a biomarker of SUDEP risk. "
            "Any breakthrough FBTCS mandates urgent AED review and SUDEP counselling."
        ),
        "triggers": "Highest risk: sleep deprivation + missed AED dose (frontal lobe more sensitive to sleep "
                    "deprivation than temporal); alcohol withdrawal; fever",
        "first_line_aed": "Carbamazepine / Levetiracetam (add-on for FBTCS breakthrough); optimize primary AED first"
    },
]


# ─── Trigger catalog ─────────────────────────────────────────────────────────

_TRIGGERS = [
    {
        "trigger": "Sleep Deprivation",
        "frequency_pct": 75,
        "mechanism": (
            "Frontal lobe is particularly sensitive to sleep deprivation — more so than temporal lobe. "
            "Sleep loss reduces GABAergic inhibition of prefrontal and supplementary motor cortex. "
            "NREM sleep is the preferred state for hypermotor FLE seizures — sleep deprivation disrupts "
            "NREM architecture → rebound NREM increases frontal excitability. "
            "FLE seizures cluster in NREM Stage 2 and N3."
        ),
        "mitigation": (
            "Minimum 7-9 hours nightly sleep; consistent sleep/wake times critical for ADNFLE; "
            "CBT-I for insomnia; avoid shift work and night work; sleep diary; "
            "consider melatonin if sleep initiation difficulty on AEDs"
        )
    },
    {
        "trigger": "Missed AED Doses",
        "frequency_pct": 65,
        "mechanism": (
            "Sub-therapeutic AED plasma level → seizure threshold drops; "
            "rebound hyperexcitability of frontal cortex within 12-24 hours of missed dose. "
            "CBZ/OXC: enzyme auto-induction means plasma levels fluctuate more — missed doses "
            "have amplified impact. Nocturnal seizures may occur when evening dose is forgotten."
        ),
        "mitigation": (
            "Smart pill dispenser; phone alarm; weekly blister pack; 30-day refill reminders; "
            "pharmacist medication reconciliation; bedside dose for nocturnal coverage; "
            "XR (extended-release) formulations for CBZ/OXC to smooth plasma level variability"
        )
    },
    {
        "trigger": "Psychological Stress",
        "frequency_pct": 60,
        "mechanism": (
            "HPA axis activation → cortisol surge → CRH directly reduces frontal cortex seizure threshold. "
            "Prefrontal cortex is a major CRH target; stress-induced neuroinflammation in frontal circuits. "
            "Acute stress also disrupts sleep → compounding sleep deprivation trigger effect."
        ),
        "mitigation": (
            "Mindfulness-based stress reduction (MBSR); CBT referral for anxiety/depression; "
            "biofeedback; regular aerobic exercise (neuroprotective); social support network; "
            "identify occupational stressors — occupational therapy referral if needed"
        )
    },
    {
        "trigger": "Fatigue",
        "frequency_pct": 50,
        "mechanism": (
            "Physical and mental fatigue alters frontal cortical excitability — prefrontal cortex "
            "is particularly metabolically demanding. Fatigue-induced impairment of inhibitory "
            "control circuits in DLPFC and ACC → reduced GABAergic tone. "
            "Closely related to sleep deprivation but operates via metabolic pathway independently."
        ),
        "mitigation": (
            "Pacing activities; planned rest periods; avoid overexertion; "
            "occupational therapy for energy conservation strategies; "
            "iron/B12 check if fatigue disproportionate; thyroid function screen"
        )
    },
    {
        "trigger": "Alcohol Use",
        "frequency_pct": 45,
        "mechanism": (
            "Acute alcohol: GABAergic potentiation (transient protective effect). "
            "Withdrawal rebound (12-48h post-binge): NMDA upregulation + GABA downregulation → "
            "frontal cortex hyperexcitability. Alcohol also disrupts sleep architecture "
            "(suppresses NREM Stage 3 → rebound NREM increases frontal excitability). "
            "CBZ/OXC: alcohol-drug interaction may alter enzyme induction."
        ),
        "mitigation": (
            "Abstinence recommended; if drinking: limit to 1-2 standard units/occasion; "
            "avoid binge drinking; never combine with sedating AEDs; "
            "refer to addiction medicine if alcohol use disorder present"
        )
    },
    {
        "trigger": "Fever / Systemic Illness",
        "frequency_pct": 35,
        "mechanism": (
            "Fever lowers seizure threshold (Na+ channel kinetics shifted at elevated temperature); "
            "dehydration raises AED concentration variability; "
            "intercurrent illness alters AED absorption and distribution. "
            "Frontal lobe FCD is particularly sensitive to metabolic perturbation."
        ),
        "mitigation": (
            "Aggressive antipyresis (paracetamol/ibuprofen) at first fever sign; "
            "rescue benzodiazepine protocol (buccal midazolam / rectal diazepam); "
            "adequate hydration; sick-day AED instructions; "
            "written emergency action plan for carers"
        )
    },
    {
        "trigger": "Hormonal / Catamenial",
        "frequency_pct": 20,
        "mechanism": (
            "Catamenial epilepsy affects ~30% of women with focal epilepsy including FLE. "
            "Oestrogen-progesterone ratio changes at luteal-follicular transition → "
            "oestrogen excitatory (frontal lobe sensitisation); "
            "progesterone withdrawal reduces GABA-A neurosteroid modulation. "
            "CBZ/OXC enzyme induction reduces progesterone levels — may worsen catamenial pattern."
        ),
        "mitigation": (
            "Seizure diary to confirm catamenial pattern; "
            "clobazam 10 mg/day perimenstrual (days -3 to +3); "
            "progesterone supplementation in refractory cases; gynaecology liaison; "
            "note: CBZ/OXC reduce OCP efficacy — non-hormonal contraception essential"
        )
    },
    {
        "trigger": "Physical Exertion",
        "frequency_pct": 15,
        "mechanism": (
            "Hyperventilation (respiratory alkalosis) lowers CO₂ → cerebral vasoconstriction → "
            "relative ischaemia in epileptogenic frontal zone; "
            "electrolyte derangement with heavy sweating alters AED distribution. "
            "Less common than sleep deprivation or missed dose as FLE trigger."
        ),
        "mitigation": (
            "Encourage supervised moderate exercise (reduces SUDEP risk overall — do not discourage); "
            "avoid maximal hyperventilatory sports acutely; "
            "hydration and electrolyte replacement; "
            "warm-up to prevent abrupt alkalosis"
        )
    },
]


# ─── Treatment catalog ───────────────────────────────────────────────────────

_TREATMENTS = [
    {
        "drug": "Carbamazepine (CBZ / Tegretol)",
        "type": "AED — First-line focal FLE (ILAE Level A)",
        "dose": "Start 200 mg BD; titrate by 200 mg/week; maintenance 400-1600 mg/day in 2-3 divided doses; "
                "XR formulation preferred for smoother levels and nocturnal coverage",
        "moa": "Use-dependent Na+ channel blockade (Nav1.1, Nav1.2) → reduces high-frequency neuronal firing; "
               "also blocks L-type Ca²⁺ channels at higher doses; CYP3A4 auto-induction",
        "efficacy": "Level A ILAE focal epilepsy; 50-60% seizure-free in new-onset FLE; "
                    "particularly effective in ADNFLE (CHRNA4/CHRNB2) — first-line of choice; "
                    "reference standard for focal AED comparisons",
        "safety": "Hyponatremia (SIADH — 5-10%); SJS/TEN (HLA-B*1502 screen in Asian patients mandatory); "
                  "enzyme inducer (CYP3A4/2C19) — reduces OCP efficacy; agranulocytosis (rare); "
                  "NOTE: CBZ is FIRST-LINE in FLE — contrast with JME/Dravet where it is CONTRAINDICATED",
        "fda_status": "FDA-approved 1968; NICE first-line focal epilepsy; ILAE Level A evidence; "
                      "European ADNFLE Consensus 2015: CBZ preferred AED",
        "evidence_level": "Level A (ILAE 2006/2017); class I-II RCTs; ADNFLE Consensus 2015"
    },
    {
        "drug": "Lamotrigine (LTG / Lamictal)",
        "type": "AED — First-line focal, preferred in women of childbearing age",
        "dose": "SLOW titration mandatory: 25 mg/day × 2 weeks → 50 mg/day × 2 weeks → "
                "increase by 50 mg/2 weeks; maintenance 100-400 mg/day; "
                "halve dose if adding valproate; triple if on enzyme inducers (CBZ/OXC)",
        "moa": "Voltage-gated Na+ channel blockade; reduces glutamate release; "
               "some Ca²⁺ channel activity; anti-kindling properties",
        "efficacy": "Level A ILAE focal epilepsy; 45-55% responder rate; "
                    "comparable to CBZ (SANAD 2007); preferred in women (better teratogenicity profile); "
                    "good for FAS/FIAS in FLE; less effective for hypermotor seizures than CBZ",
        "safety": "SJS/TEN (1:1000 — SLOW titration mandatory); "
                  "LTG-VPA pharmacokinetic interaction (VPA doubles LTG level — halve LTG); "
                  "maculopapular rash 10%; behavioural activation in children; "
                  "insomnia (may worsen nocturnal FLE if taken at night — consider morning dosing)",
        "fda_status": "FDA-approved adjunctive 1994, monotherapy 1998; NICE first-line; SANAD trial",
        "evidence_level": "Level A (ILAE 2017); SANAD trial (Marson 2007)"
    },
    {
        "drug": "Levetiracetam (LEV / Keppra)",
        "type": "AED — Broad-spectrum, favoured for add-on",
        "dose": "Start 500 mg BD; increase by 500 mg/week; maintenance 1000-3000 mg/day; "
                "renal dose adjustment (CrCl <80 mL/min); IV form available",
        "moa": "SV2A (synaptic vesicle protein 2A) ligand → reduces vesicular neurotransmitter release; "
               "modulates GABA-A receptor conformation; reduces high-voltage Ca²⁺ current; "
               "anti-kindling properties in frontal lobe models",
        "efficacy": "Level B ILAE; 40-50% ≥50% responder rate as add-on in drug-resistant FLE; "
                    "30-40% seizure-free in new-onset focal; broad-spectrum; "
                    "no enzyme induction advantage over CBZ for FLE specifically",
        "safety": "Behavioural/mood side effects (irritability, aggression, depression — 15%); "
                  "frontal lobe dysfunction may amplify irritability/aggression symptoms from LEV; "
                  "pre-existing psychiatric disorder = relative risk; "
                  "minimal enzyme induction; safe in pregnancy (relative)",
        "fda_status": "FDA-approved adjunctive focal 1999; IV form 2006; NICE Level A",
        "evidence_level": "Level B (ILAE 2017); multiple RCTs"
    },
    {
        "drug": "Oxcarbazepine (OXC / Trileptal)",
        "type": "AED — First-line focal (ILAE Level A)",
        "dose": "Start 300 mg BD; increase by 300 mg/week; maintenance 600-2400 mg/day; "
                "XR/ESL preferred for tolerability and adherence; "
                "once-daily ESL (eslicarbazepine acetate) for nocturnal FLE convenience",
        "moa": "Active metabolite MHD (monohydroxy derivative): Na+ channel blockade similar to CBZ "
               "but less auto-induction; some Ca²⁺ channel blockade",
        "efficacy": "Level A ILAE; 50-60% responder rate; comparable to CBZ with better tolerability; "
                    "effective in ADNFLE (CBZ/OXC equally effective for CHRNA4 variants); "
                    "less enzyme induction than CBZ → fewer drug interactions",
        "safety": "Hyponatremia (10% — monitor Na+ especially elderly and with diuretics); "
                  "mild enzyme inducer; 10% OCP interaction (non-hormonal contraception required); "
                  "SJS/TEN less common than CBZ; HLA-B*1502 cross-caution; "
                  "10% cross-allergy with CBZ (if prior CBZ rash, caution with OXC)",
        "fda_status": "FDA-approved adjunctive 2000, monotherapy 2017; NICE first-line focal",
        "evidence_level": "Level A (ILAE 2006/2017)"
    },
    {
        "drug": "Lacosamide (LCM / Vimpat)",
        "type": "AED — Adjunctive focal (ILAE Level A add-on)",
        "dose": "Start 50 mg BD; increase by 100 mg/week; maintenance 100-400 mg/day (200-400 mg/day target); "
                "IV form available for acute management; max 400 mg/day",
        "moa": "Enhances SLOW inactivation of voltage-gated Na+ channels (unique mechanism — distinct from "
               "CBZ/OXC which target fast inactivation); CRMP-2 binding; "
               "complementary mechanism makes LCM rational add-on to CBZ/OXC",
        "efficacy": "Level A add-on; 40-50% ≥50% responder rate as add-on in drug-resistant FLE; "
                    "effective for frontal lobe FAS/FBTCS; "
                    "particularly useful as add-on to Na-channel AED via complementary mechanism (slow vs fast inactivation)",
        "safety": "PR interval prolongation (pre-treatment ECG MANDATORY; avoid in 2°/3° AV block); "
                  "dizziness, diplopia, ataxia (dose-related); "
                  "avoid in severe cardiac conduction disease; "
                  "additive PR effect with other Na-channel AEDs (CBZ/OXC/LTG)",
        "fda_status": "FDA Schedule V; FDA adjunctive focal 2008, monotherapy 2014; "
                      "ESC cardiac monitoring recommendation",
        "evidence_level": "Level A add-on (ILAE 2017); BENMILD RCTs"
    },
    {
        "drug": "Perampanel (PER / Fycompa)",
        "type": "AED — Adjunctive focal (ILAE Level A add-on); REMS required",
        "dose": "Start 2 mg nocte; increase by 2 mg/2 weeks; maintenance 4-12 mg nocte; "
                "nocturnal dosing preferred (sedating + ideal for nocturnal FLE); "
                "enzyme inducers (CBZ) halve PER levels → may need higher doses (up to 12 mg)",
        "moa": "Non-competitive AMPA glutamate receptor antagonist — unique mechanism; "
               "reduces glutamate-mediated excitatory neurotransmission in frontal cortex; "
               "no sodium channel activity; complements CBZ/LTG/LCM",
        "efficacy": "Level A add-on; 35-40% ≥50% responder rate in drug-resistant focal epilepsy; "
                    "particularly effective for FBTCS (reduces FBTCS frequency ≥50% in 40% of patients); "
                    "nocturnal dosing aligns with hypermotor FLE nocturnal seizure pattern",
        "safety": "Aggression, irritability, psychiatric side effects (10-20%) — FDA BLACK BOX WARNING; "
                  "REMS program: PHQ-9 + aggression scale at each visit; "
                  "frontal lobe dysfunction may amplify PER behavioural toxicity; "
                  "dizziness, somnolence; falls risk; enzyme inducers (CBZ) reduce PER levels significantly",
        "fda_status": "FDA-approved adjunctive focal 2012; REMS program mandatory; "
                      "FDA also approved for primary generalised FBTCS (2015)",
        "evidence_level": "Level A add-on (ILAE 2017); FYCOMPA Phase III RCTs (Study 304/305/306)"
    },
    {
        "drug": "Frontal Lobectomy / SEEG-Guided Resection",
        "type": "Surgical — Curative for drug-resistant FLE",
        "dose": (
            "Frontal lobectomy: resect epileptogenic zone with 1 cm margin (non-eloquent frontal). "
            "SEEG-guided tailored resection: defined by SEEG seizure onset zone + propagation zone. "
            "Pre-surgical workup: video-EEG (prolonged), MRI epilepsy protocol, FDG-PET, "
            "neuropsychology, SEEG (mandatory for most FLE — scalp EEG poorly localising), "
            "fMRI/TMS language + motor mapping if near eloquent cortex."
        ),
        "moa": (
            "Removes the primary epileptogenic zone in frontal lobe. "
            "FLE surgery is more complex than TLE surgery: "
            "frontal lobe poorly localised on scalp EEG; eloquent cortex (motor, SMA, Broca) at risk; "
            "SEEG essential for 3D mapping; multiple epileptogenic foci possible (FCD often multifocal)."
        ),
        "efficacy": (
            "50-60% Engel Class I at 2 years (Jeha 2007, Englot 2011) — "
            "lower than MTLE-HS (60-70%) due to: scalp EEG poor localisation, "
            "proximity to eloquent cortex limiting resection extent, FCD often diffuse. "
            "Best outcomes: lesional FLE (tumor/cavernoma: 75-90%), FCD with complete resection (60-65%); "
            "worst: MRI-negative cryptogenic FLE (30-45%)."
        ),
        "safety": (
            "Motor deficit risk (if near primary motor cortex — careful mapping required); "
            "aphasia risk (dominant frontal — Broca/SMA involvement); "
            "executive function/working memory changes (prefrontal resection); "
            "infection/haemorrhage <1%; "
            "psychiatric deterioration (pre-op psychiatric assessment mandatory); "
            "personality change in large frontal resections"
        ),
        "fda_status": "ILAE Level A surgical evidence (2017); "
                      "AAN Practice Guideline Level A (2003) — applies to focal resective surgery broadly; "
                      "SEEG: endorsed by ILAE and major epilepsy surgery centres",
        "evidence_level": "Level A (ILAE Surgical Guidelines 2017); Jeha 2007 Epilepsia; Englot 2011 J Neurosurg"
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "type": "Non-pharmacological — Adjunctive",
        "dose": "Classical 4:1 fat:protein+carb ratio; MCT diet (more palatable); "
                "LGIT (low glycaemic index); minimum 3-month trial; dietitian-supervised; "
                "mTOR inhibitor hypothesis: particularly rational for FCD-related FLE",
        "moa": "Ketone bodies (β-hydroxybutyrate) → GABA synthesis upregulation + glutamate downregulation; "
               "mTOR pathway inhibition (particularly relevant in FCD Type II / DEPDC5-related FLE); "
               "mitochondrial biogenesis; direct KATP channel opening; "
               "reduces frontal cortex excitability via metabolic shift",
        "efficacy": "Level B; 40-50% ≥50% seizure reduction in drug-resistant FLE; "
                    "better evidence in paediatric FLE with FCD; "
                    "mTOR inhibition rationale makes KD particularly relevant for DEPDC5/FCD FLE; "
                    "adjunctive — combine with AEDs, not replace",
        "safety": "Hyperlipidaemia, nephrolithiasis (citrate supplement), constipation, "
                  "growth retardation in children; acidosis; "
                  "contraindicated in fatty acid oxidation disorders; "
                  "carnitine supplementation; lipid monitoring every 3 months",
        "fda_status": "No FDA approval as drug; ILAE recommended adjunctive (2009 consensus); "
                      "pediatric epilepsy guidelines Level B",
        "evidence_level": "Level B (ILAE 2009 consensus); multiple RCTs in paediatric DRE"
    },
]


# ─── AED monitoring ──────────────────────────────────────────────────────────

_AED_MONITORING = [
    {
        "drug": "Carbamazepine (CBZ)",
        "category": "ENZYME INDUCER + HLA SCREEN + TDM",
        "risk": (
            "Hyponatremia (SIADH — 5-10%); agranulocytosis/aplastic anaemia (1:10,000-500,000); "
            "SJS/TEN (HLA-B*1502 in Han Chinese/SE Asian — mandatory screen before prescribing); "
            "enzyme induction (CYP3A4/2C19) → OCP failure, drug interactions; "
            "NOTE for FLE: CBZ is FIRST-LINE (not contraindicated as in JME/Dravet)"
        ),
        "monitoring": (
            "Baseline + 6-monthly: CBC, LFTs, serum Na+; "
            "TDM target 4-12 mg/L (check at steady state and after dose change); "
            "HLA-B*1502 before initiation in Han Chinese/SE Asian populations; "
            "ECG if cardiac history or combining with LCM"
        ),
        "mitigation": (
            "Slow titration (200 mg/week); XR formulation (smoother levels, better nocturnal coverage); "
            "check Na+ if nausea/confusion develops; "
            "non-hormonal contraception MANDATORY (enzyme induction reduces OCP efficacy); "
            "HLA screen before prescribing in at-risk populations; "
            "written alert card for patient regarding OCP interaction"
        )
    },
    {
        "drug": "Lamotrigine (LTG)",
        "category": "SJS/TEN RISK — SLOW TITRATION MANDATORY",
        "risk": (
            "SJS/TEN (Stevens-Johnson/Toxic Epidermal Necrolysis): 1:1000 adult; higher in children; "
            "risk increases with rapid titration and co-prescribing valproate (VPA doubles LTG level); "
            "insomnia as side effect — may worsen nocturnal FLE if dosed in evening (consider AM dosing)"
        ),
        "monitoring": (
            "Rash monitoring at every visit (first 8 weeks highest risk); "
            "TDM 3-15 μg/mL (optional but useful in pregnancy and enzyme-inducer combinations); "
            "pregnancy: LTG clearance increases ~300% → dose escalation required throughout gestation; "
            "postpartum: rapid LTG level rise → toxicity risk (reduce dose promptly post-delivery)"
        ),
        "mitigation": (
            "MANDATORY slow titration (25 mg × 2wk → 50 mg × 2wk → +50 mg/2 wk); "
            "halve LTG dose when adding valproate; "
            "triple LTG if on enzyme-inducing AEDs (CBZ/OXC); "
            "written rash action plan for patient; "
            "immediate discontinuation if muco-cutaneous rash; "
            "consider morning dosing if nocturnal insomnia worsening FLE"
        )
    },
    {
        "drug": "Lacosamide (LCM)",
        "category": "CARDIAC MONITORING — PR INTERVAL MANDATORY",
        "risk": (
            "PR interval prolongation (dose-dependent; risk increases with cardiac comorbidity); "
            "2°/3° AV block risk in pre-existing conduction disease; "
            "additive PR effect when combined with other Na+ channel blockers (CBZ, OXC, LTG); "
            "dizziness/diplopia (dose-related — titrate slowly)"
        ),
        "monitoring": (
            "PRE-TREATMENT ECG — MANDATORY before initiating LCM; "
            "ECG at target dose; "
            "repeat ECG if PR >200 ms or symptoms (syncope, palpitations); "
            "caution with Na-channel AED polytherapy (CBZ + LCM + OXC); "
            "CONTRAINDICATED in 2°/3° AV block or sick sinus syndrome without pacemaker"
        ),
        "mitigation": (
            "Slow IV infusion (>15 min if IV route); "
            "check co-medications for PR-prolonging drugs (antiarrhythmics, beta-blockers); "
            "cardiology clearance if PR >200 ms at baseline; "
            "reduce dose at first signs of diplopia/ataxia; "
            "IV to oral switch as soon as feasible"
        )
    },
    {
        "drug": "Perampanel (PER)",
        "category": "PSYCHIATRIC REMS — AGGRESSION/PSYCHIATRIC MONITORING",
        "risk": (
            "Aggression, hostility, irritability, psychiatric side effects (10-20%) — "
            "FDA BLACK BOX WARNING issued 2013; "
            "frontal lobe executive dysfunction in FLE may amplify PER behavioural toxicity; "
            "suicidality (all AEDs — FDA class warning); "
            "enzyme inducers (CBZ/OXC/PHT) reduce PER levels by ~50-70% → may need higher doses"
        ),
        "monitoring": (
            "PHQ-9 (depression) + standardised aggression scale at EVERY visit; "
            "caregiver/family interview mandatory — patients may lack insight into behavioural change; "
            "REMS program enrollment and documentation; "
            "consider dose reduction at first behavioural signal; "
            "drug levels not routinely available but interaction with CBZ is clinically important"
        ),
        "mitigation": (
            "Start low (2 mg nocte); increase slowly (2 mg/2 weeks); "
            "psychoeducation for patient AND family/caregivers about aggression risk; "
            "psychiatric co-management if prior mood/personality disorder; "
            "nocturnal dosing minimises daytime sedation + aligns with hypermotor FLE pattern; "
            "if aggression emerges: reduce dose first, then discontinue if unresponsive; "
            "account for CBZ/OXC enzyme induction — may need 8-12 mg instead of 4-8 mg"
        )
    },
]


# ─── Lifecycle trajectory ────────────────────────────────────────────────────

_LIFECYCLE = [
    {
        "age_window": "Childhood (0-12 years)",
        "typical_profile": (
            "FCD is the most common etiology presenting in childhood — peak FCD diagnosis age 2-8 years. "
            "Nocturnal hypermotor seizures frequently misdiagnosed as parasomnias (night terrors, "
            "REM sleep behavior disorder) — video-EEG during sleep essential. "
            "Developmental and cognitive impact from frequent nocturnal seizures + frontal dysfunction."
        ),
        "fle_indicators": (
            "Nocturnal clustering of stereotyped motor episodes (screaming + thrashing); "
            "developmental regression or plateau; school difficulties (executive function/attention); "
            "MRI with epilepsy protocol; EEG during sleep recording; "
            "paediatric neurologist referral within 6 weeks of first seizure"
        ),
        "action": (
            "Epilepsy protocol MRI (3T + epilepsy sequences); "
            "prolonged video-EEG with sleep recording; "
            "neuropsychological baseline (executive function, attention, memory); "
            "early paediatric epilepsy centre referral; "
            "CBZ or OXC first-line for childhood FLE; "
            "surgical evaluation if drug-resistant ≥2 AEDs — FCD: best surgical window in childhood "
            "(neuroplasticity for post-surgical recovery)"
        )
    },
    {
        "age_window": "Adolescent (12-18 years)",
        "typical_profile": (
            "ADNFLE (Autosomal Dominant Nocturnal FLE) peaks in adolescence: "
            "CHRNA4/CHRNB2/CHRNA2 variants. Hypermotor nocturnal seizures. "
            "Family history often present. Academic impact from nocturnal seizures and sleep deprivation. "
            "Risk of misdiagnosis as epileptic psychosis or psychiatric disorder (frontal lobe behavioural features)."
        ),
        "fle_indicators": (
            "Family history of nocturnal seizures + adolescent onset + hypermotor semiology → ADNFLE; "
            "academic decline (frontal executive dysfunction from seizures + AED cognitive effects); "
            "peer stigma; driving ineligibility approaching (prepare at 16+); "
            "mental health screening (depression prevalence elevated in FLE)"
        ),
        "action": (
            "ADNFLE gene panel (CHRNA4, CHRNB2, CHRNA2, DEPDC5) if family history; "
            "first-line AED: CBZ (particularly effective in ADNFLE); "
            "sleep hygiene counselling (critical for adolescents); "
            "psychiatric screening (PHQ-A for adolescents); "
            "driving legislation counselling (begin age 16+ in anticipation of licensure); "
            "neuropsychological assessment for academic support planning; "
            "surgical discussion if 2 AED failures (FCD in adolescence — good neuroplasticity)"
        )
    },
    {
        "age_window": "Young Adult (18-30 years)",
        "typical_profile": (
            "Highest impact years: employment, driving, relationships. "
            "~30-40% drug-resistant at this stage — surgical evaluation window is optimal. "
            "SEEG candidacy evaluation for MRI-negative FLE. "
            "Driving cessation impacts independence and employment significantly."
        ),
        "fle_indicators": (
            "Drug-resistant (≥2 AED failures) → urgent surgical referral; "
            "FBTCS breakthrough → SUDEP risk; "
            "frontal executive dysfunction affecting employment/relationships; "
            "cognitive decline on neuropsychological testing (working memory, executive function)"
        ),
        "action": (
            "SURGICAL EVALUATION (Level 4 centre) if ≥2 AED failures: "
            "video-EEG + MRI (3T epilepsy protocol) + FDG-PET + neuropsychology + SEEG; "
            "fMRI/TMS language + motor mapping before frontal surgery; "
            "SUDEP counselling — young adults with uncontrolled FBTCS are highest risk group; "
            "driving: 12 months seizure-free required (Canada/UK); counsel on employment implications; "
            "vocational rehabilitation if executive dysfunction impairs work performance"
        )
    },
    {
        "age_window": "Childbearing (25-40 years)",
        "typical_profile": (
            "Enzyme-inducing AEDs (CBZ/OXC) reduce OCP efficacy — unplanned pregnancies at risk. "
            "LTG preferred if seizure-free (better teratogenicity profile, no enzyme induction). "
            "Catamenial FLE in ~20% of women. "
            "Nocturnal hypermotor seizures during pregnancy — positioning and supervision concerns."
        ),
        "fle_indicators": (
            "On enzyme-inducing AED (CBZ/OXC) — OCP counselling MANDATORY; "
            "catamenial pattern (seizure diary); "
            "planning pregnancy — AED review required; "
            "postpartum sleep deprivation → breakthrough nocturnal FLE risk"
        ),
        "action": (
            "Pre-conception counselling: ENZYME INDUCERS (CBZ/OXC) reduce OCP efficacy — "
            "non-hormonal contraception or high-dose OCP MANDATORY; "
            "consider switch to LTG (non-enzyme-inducer, better teratogenicity) if seizure-free; "
            "folic acid 5 mg/day pre-conception and first trimester; "
            "obstetric neurology co-management; "
            "postpartum: plan for sleep support to prevent nocturnal FLE relapse; "
            "breastfeeding: CBZ/OXC/LTG generally compatible"
        )
    },
    {
        "age_window": "Middle Adult (40-60 years)",
        "typical_profile": (
            "Long-term AED effects: bone density loss from enzyme-inducing AEDs (CBZ/OXC/PHT); "
            "mood monitoring for LEV/PER (frontal lobe dysfunction may amplify behavioural toxicity). "
            "Cumulative executive function changes from FLE + AEDs. "
            "Re-evaluate surgical candidacy — never too late for frontal resection if newly lesional."
        ),
        "fle_indicators": (
            "Bone density concern (on CBZ/OXC long-term); "
            "mood/personality changes on LEV or PER — frontal amplification; "
            "new MRI lesion found (cavernoma, tumor) — re-evaluate for surgery; "
            "AED polypharmacy rationalization needed; "
            "occupational impact of frontal executive dysfunction"
        ),
        "action": (
            "DEXA scan if on enzyme-inducing AEDs >5 years (osteoporosis risk); "
            "Vitamin D + Ca²⁺ supplementation if on CBZ/OXC/PHT; "
            "neuropsychological reassessment (executive function, working memory); "
            "AED rationalisation (minimise polypharmacy); "
            "re-evaluate surgical candidacy if newly lesional; "
            "PHQ-9 + aggression monitoring if on LEV/PER; "
            "occupational therapy if executive dysfunction impairs work"
        )
    },
    {
        "age_window": "Older Adult (60+ years)",
        "typical_profile": (
            "New-onset FLE in elderly: consider structural (frontal tumor, stroke, cavernoma) and "
            "autoimmune (anti-CASPR2, anti-GABA-B affecting frontal lobe). "
            "Falls risk from tonic FLE seizures. Polypharmacy. "
            "Pharmacokinetic changes: reduced renal clearance (LEV dose), reduced hepatic metabolism; "
            "CBZ hyponatremia risk increases with diuretics."
        ),
        "fle_indicators": (
            "New-onset frontal lobe seizures → MRI urgently (exclude tumor/vascular); "
            "CBZ/OXC hyponatremia risk with concurrent diuretics; "
            "falls from tonic FLE seizures — hip fracture risk; "
            "cognitive decline multifactorial (epilepsy vs AED vs dementia); "
            "polypharmacy AED interaction risk (CBZ enzyme induction)"
        ),
        "action": (
            "Brain MRI with gadolinium (exclude frontal glioma, metastasis, vascular); "
            "autoimmune encephalitis panel if subacute onset (anti-CASPR2, anti-GABA-B, anti-NMDAR); "
            "avoid sedating AEDs (benzodiazepines, barbiturates, high-dose CBZ); "
            "prefer LTG (lower enzyme induction) or LEV (renal dose adjust) in elderly; "
            "fall prevention: helmet assessment, environment modification, physiotherapy; "
            "osteoporosis screening (DEXA) if on long-term enzyme-inducing AED; "
            "cardiology liaison if adding LCM (PR monitoring essential in elderly cardiac patients)"
        )
    },
]


# ─── Key definitions ─────────────────────────────────────────────────────────

_DEFINITIONS = [
    {
        "term": "FLE",
        "definition": (
            "Frontal Lobe Epilepsy — focal epilepsy with seizure onset in the frontal lobe; "
            "second most common focal epilepsy (~20-30% of focal epilepsies); "
            "hallmark: hypermotor seizures, nocturnal predilection, brief duration, rapid recovery"
        )
    },
    {
        "term": "ADNFLE",
        "definition": (
            "Autosomal Dominant Nocturnal Frontal Lobe Epilepsy — genetic FLE caused by CHRNA4, "
            "CHRNB2, or CHRNA2 nicotinic acetylcholine receptor subunit mutations; "
            "autosomal dominant; hypermotor nocturnal seizures from NREM sleep; "
            "responds well to CBZ; frequently misdiagnosed as parasomnias"
        )
    },
    {
        "term": "Hypermotor Seizure",
        "definition": (
            "The characteristic FLE seizure type: explosive onset, bilateral asymmetric limb movements "
            "(thrashing, cycling, kicking), axial involvement, vocalization, often preserved awareness; "
            "nocturnal predilection (NREM sleep); duration 10-60 seconds; rapid offset"
        )
    },
    {
        "term": "FCD (Focal Cortical Dysplasia)",
        "definition": (
            "Malformation of cortical development — most common structural cause of FLE (35%); "
            "FCD Type IIb: balloon cells + dysmorphic neurons + mTOR pathway somatic mutations; "
            "MRI may be subtle (transmantle sign, grey-white blurring) or negative in 30%; "
            "SEEG essential for surgical planning"
        )
    },
    {
        "term": "SMA Seizure (Supplementary Motor Area)",
        "definition": (
            "Focal aware seizure arising from supplementary motor area: tonic posturing of "
            "contralateral arm (fencing posture), bilateral tonic posturing, speech arrest with "
            "preserved comprehension; sudden offset with immediate full recovery; "
            "awareness fully preserved; EEG: contralateral frontal or bilateral discharge"
        )
    },
    {
        "term": "Jacksonian March",
        "definition": (
            "Progressive spread of focal clonic motor activity from distal to proximal limb "
            "(e.g., finger → hand → arm → face); reflects spread of ictal discharge along "
            "primary motor cortex somatotopic map (homunculus); "
            "localising sign for seizure origin in primary motor cortex contralateral to march"
        )
    },
    {
        "term": "SEEG (Stereo-EEG)",
        "definition": (
            "Intracranial depth electrode implantation for 3D mapping of seizure onset zone; "
            "essential in FLE (scalp EEG poorly localising for deep frontal/mesial sources); "
            "covers cingulate, SMA, orbital frontal cortex that scalp EEG misses; "
            "guides tailored SEEG-based frontal resection; endorsed by ILAE"
        )
    },
    {
        "term": "Frontal Lobectomy",
        "definition": (
            "Surgical resection of frontal lobe epileptogenic zone; "
            "50-60% Engel Class I (lower than MTLE-HS due to EEG localisation challenges and "
            "eloquent cortex proximity); pre-surgical mapping mandatory: fMRI/TMS language/motor; "
            "best outcomes: lesional FLE (tumor 75-90%), FCD with complete resection (60-65%)"
        )
    },
    {
        "term": "DEPDC5",
        "definition": (
            "DEP Domain Containing 5 — mTOR pathway repressor gene; "
            "most common cause of familial focal epilepsy including FLE; "
            "loss-of-function mutations → mTOR pathway activation → cortical dysplasia; "
            "variable expressivity; accounts for 10-12% of familial focal epilepsy"
        )
    },
    {
        "term": "KCNT1",
        "definition": (
            "Potassium channel gene — gain-of-function mutations cause severe nocturnal FLE; "
            "often drug-resistant; quinidine (K-channel blocker) partially effective off-label; "
            "associated with ADNFLE-like nocturnal hypermotor seizures and intellectual disability in severe cases"
        )
    },
    {
        "term": "Engel Classification",
        "definition": (
            "Surgical outcome scale: Class I = seizure-free (worthwhile improvement); "
            "Class II = rare seizures; Class III = worthwhile improvement; "
            "Class IV = no worthwhile improvement; "
            "FLE surgical benchmark: 50-60% Engel Class I vs 60-70% for MTLE-HS (Jeha 2007)"
        )
    },
    {
        "term": "mTOR Pathway",
        "definition": (
            "Mechanistic Target of Rapamycin — intracellular kinase regulating cell growth "
            "and proliferation; dysregulated in FCD (DEPDC5, MTOR, TSC1/TSC2 mutations) → "
            "cortical malformations and epileptogenesis; mTOR inhibitors (everolimus/rapamycin) "
            "reduce seizure frequency in TSC and FCD (off-label FLE)"
        )
    },
    {
        "term": "Low-Voltage Fast Activity (LVFA)",
        "definition": (
            "EEG hallmark of FLE ictal onset: low-amplitude, high-frequency (gamma range, >30 Hz) "
            "activity at seizure onset on SEEG; often not visible on scalp EEG (deep frontal sources); "
            "indicates seizure zone in dysplastic or neoplastic frontal cortex; "
            "best biomarker of FLE seizure onset zone"
        )
    },
    {
        "term": "SUDEP",
        "definition": (
            "Sudden Unexpected Death in Epilepsy; risk 1:1000/year in focal epilepsy including FLE; "
            "highest risk: uncontrolled FBTCS, nocturnal seizures (hypermotor FLE → FBTCS), prone position; "
            "risk reduced by seizure freedom + supervised sleep; "
            "PGES (postictal generalised EEG suppression >50 seconds) is biomarker of SUDEP risk"
        )
    },
]


# ─── Clinical standards ──────────────────────────────────────────────────────

_STANDARDS = [
    {
        "name": "ILAE Classification 2022",
        "body": "ILAE",
        "scope": (
            "Seizure type (FAS/FIAS/FBTCS/Hypermotor) and epilepsy syndrome classification; "
            "FLE defined as focal epilepsy with frontal lobe onset; "
            "updated syndrome taxonomy including ADNFLE"
        )
    },
    {
        "name": "NICE NG217 (2022)",
        "body": "NICE UK",
        "scope": (
            "First-line: CBZ or LTG for focal epilepsy including FLE; "
            "refer to Level 4 epilepsy centre after 2 AED failures; "
            "video-EEG with sleep recording for suspected nocturnal FLE; "
            "SEEG recommended when scalp EEG non-localising"
        )
    },
    {
        "name": "ILAE 2006 AED Monotherapy Evidence (Level A/B)",
        "body": "ILAE",
        "scope": (
            "CBZ and OXC: Level A evidence for focal epilepsy monotherapy; "
            "LTG: Level A evidence (SANAD trial); "
            "LEV: Level B; defines evidence hierarchy for AED selection in FLE"
        )
    },
    {
        "name": "FDA Perampanel REMS (2012)",
        "body": "FDA",
        "scope": (
            "Risk Evaluation and Mitigation Strategy for perampanel (Fycompa); "
            "BLACK BOX WARNING: serious psychiatric/behavioural events including aggression; "
            "REMS enrollment required; PHQ-9 + aggression scale at each visit; "
            "particularly relevant in FLE where frontal dysfunction may amplify PER toxicity"
        )
    },
    {
        "name": "ILAE Surgical Guidelines 2017",
        "body": "ILAE",
        "scope": (
            "Level A evidence for resective surgery in drug-resistant focal epilepsy; "
            "SEEG endorsed for invasive monitoring; "
            "pre-surgical workup standards: MRI + video-EEG + neuropsychology + FDG-PET; "
            "FLE: SEEG recommended given poor scalp EEG localisation"
        )
    },
    {
        "name": "European ADNFLE Consensus 2015",
        "body": "European Epilepsy Society",
        "scope": (
            "Diagnostic criteria for ADNFLE (hypermotor nocturnal seizures + family history + "
            "CHRNA4/CHRNB2/CHRNA2 mutations); "
            "CBZ as preferred first-line AED for ADNFLE; "
            "genetic testing panel recommendations; "
            "distinguishing ADNFLE from parasomnias"
        )
    },
]


# ─── Key thresholds ──────────────────────────────────────────────────────────

_THRESHOLDS = [
    {
        "parameter": "DRE referral threshold",
        "threshold": "≥2",
        "unit": "AED failures",
        "significance": (
            "Mandatory surgical evaluation referral after ≥2 adequate AED trials fail "
            "(ILAE 2010 DRE definition); applies equally to FLE as to TLE"
        )
    },
    {
        "parameter": "CBZ therapeutic range",
        "threshold": "4–12",
        "unit": "mg/L",
        "significance": (
            "Serum TDM target range; >12 mg/L: toxicity (diplopia, ataxia, hyponatremia); "
            "check at steady state; XR formulation reduces peak-trough variability"
        )
    },
    {
        "parameter": "LTG-VPA interaction",
        "threshold": "50%",
        "unit": "LTG dose reduction",
        "significance": (
            "Halve LTG maintenance dose when adding valproate "
            "(VPA inhibits LTG glucuronidation → LTG level doubles → SJS/TEN risk)"
        )
    },
    {
        "parameter": "FLE surgical outcome (Engel Class I)",
        "threshold": "50–60",
        "unit": "% Engel Class I",
        "significance": (
            "Expected seizure-free rate post-frontal lobectomy; lower than MTLE-HS (60-70%); "
            "lesional FLE (tumor): 75-90%; FCD complete resection: 60-65%; "
            "MRI-negative cryptogenic FLE: 30-45% (Jeha 2007, Englot 2011)"
        )
    },
    {
        "parameter": "Driving seizure-free period",
        "threshold": "12",
        "unit": "months",
        "significance": (
            "UK/Canada: 12 months seizure-free for private vehicle licence; "
            "USA: 3-6 months (state-variable); "
            "nocturnal-only FLE: jurisdiction-specific — some allow driving if seizures exclusively nocturnal"
        )
    },
    {
        "parameter": "Folic acid (women of childbearing age on enzyme-inducing AEDs)",
        "threshold": "5",
        "unit": "mg/day",
        "significance": (
            "High-dose folic acid for women on enzyme-inducing AEDs (CBZ/OXC/PHT/PB); "
            "start pre-conception; reduce NTD (neural tube defect) risk; "
            "continue throughout first trimester minimum"
        )
    },
]


# ─── References ──────────────────────────────────────────────────────────────

_REFERENCES = [
    {
        "citation": "Jeha LE et al. 2007 Epilepsia (Frontal Lobe Resection Outcomes)",
        "key_finding": (
            "Frontal lobe resection outcomes: 56% seizure-free (Engel Class I) at mean 3.5-year follow-up; "
            "lesional FLE: 62% Engel I; MRI-negative: 36% Engel I; "
            "established the lower surgical benchmark for FLE vs TLE"
        )
    },
    {
        "citation": "Englot DJ et al. 2011 J Neurosurg (Predictors of Frontal Lobe Surgery Outcome)",
        "key_finding": (
            "Meta-analysis: 45-60% Engel Class I for frontal lobectomy overall; "
            "predictors of good outcome: complete lesion resection, MRI-positive, "
            "concordant SEEG/scalp EEG; negative MRI = independent predictor of poor outcome"
        )
    },
    {
        "citation": "Scheffer IE et al. 1995 Lancet (ADNFLE Description)",
        "key_finding": (
            "First description of Autosomal Dominant Nocturnal FLE (ADNFLE) as a distinct syndrome; "
            "hypermotor nocturnal seizures from NREM sleep; autosomal dominant inheritance; "
            "CHRNA4 mutation identified; responds well to CBZ"
        )
    },
    {
        "citation": "Baulac S et al. 2015 Nat Genet (DEPDC5 mTOR Pathway FLE)",
        "key_finding": (
            "DEPDC5 loss-of-function mutations identified as most common cause of familial focal epilepsy; "
            "mTOR pathway dysregulation → cortical dysplasia in FLE; "
            "variable expressivity; mTOR inhibitor rationale in DEPDC5-FCD"
        )
    },
    {
        "citation": "Wirrell EC et al. 2022 Epilepsia (ILAE FLE Syndrome Classification)",
        "key_finding": (
            "Updated ILAE operational classification for FLE syndromes; "
            "formal recognition of hypermotor seizure as FLE hallmark seizure type; "
            "ADNFLE: classified as genetic epilepsy with known pathogenic variants; "
            "FLE seizure type taxonomy aligned with 2017 ILAE classification framework"
        )
    },
    {
        "citation": "Bernhardt BC et al. 2016 Nat Rev Neurol (FCD and Cortical Malformations)",
        "key_finding": (
            "Comprehensive review of FCD pathology, genetics, and neuroimaging; "
            "FCD Type IIb: mTOR pathway somatic mutations (MTOR, TSC1/TSC2, DEPDC5); "
            "transmantle sign on MRI; surgical outcome correlates with FCD type and completeness of resection; "
            "7T MRI and post-processing for MRI-negative FCD detection"
        )
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
        "Hypermotor only (nocturnal)",
        "Hypermotor + FBTCS",
        "FAS (SMA) + Hypermotor",
        "FAS + FIAS + FBTCS",
        "FIAS + FBTCS",
    ]
    _AED_REGIMEN = [
        "Carbamazepine mono",
        "Oxcarbazepine mono",
        "Lamotrigine mono",
        "Levetiracetam mono",
        "Carbamazepine + Lamotrigine",
        "Oxcarbazepine + Levetiracetam",
        "Lacosamide + Carbamazepine",
        "Perampanel + Oxcarbazepine",
    ]
    _CONTROL = ["Seizure-free", "Improved (>50% reduction)", "Partial response", "Drug-resistant"]
    _ETIOLOGY_NAMES = [e["etiology"].split(" (")[0].split(" —")[0].strip() for e in _ETIOLOGIES]
    _ONSET_SEMIOLOGY = [
        "Hypermotor (nocturnal)",
        "SMA tonic posturing",
        "Frontal automatisms",
        "Jacksonian march",
        "Explosive bilateral motor",
        "Vocalization + thrashing",
    ]

    result = []
    for i, r in enumerate(rows[:41]):
        sd = _seed(r["patient_id"])
        # FLE etiology distribution: FCD 35%, Tumor 18%, Post-traumatic 12%,
        # Genetic/Unknown 20%, Post-infectious/vascular 10%, Cryptogenic 5%
        etiology_weights = [35, 18, 12, 20, 10, 5]
        etiology_idx = 0
        rv_et = (sd >> 2) % 100
        acc = 0
        for ei, w in enumerate(etiology_weights):
            acc += w
            if rv_et < acc:
                etiology_idx = ei
                break
        # FLE onset: bimodal — childhood (0-12) and young adult (18-35); peak at 8 and 25
        if (sd % 10) < 4:
            onset_age = 4 + (sd % 9)    # childhood: 4-12 years
        elif (sd % 10) < 7:
            onset_age = 12 + (sd % 14)  # adolescent/young adult: 12-25 years
        else:
            onset_age = 25 + (sd % 16)  # young to middle adult: 25-40 years
        subtype_idx = (sd >> 4) % len(_SEIZURE_SUBTYPES)
        aed_idx = (sd >> 8) % len(_AED_REGIMEN)
        semiology_idx = (sd >> 10) % len(_ONSET_SEMIOLOGY)
        catamenial = r.get("gender", "M") == "F" and bool((sd >> 14) % 5 == 0)  # ~20% of women
        # FLE control: ~30-40% drug-resistant
        control_w = [40, 28, 17, 15]
        control_idx = 0
        rv_ctrl = (sd >> 16) % 100
        acc = 0
        for ci, w in enumerate(control_w):
            acc += w
            if rv_ctrl < acc:
                control_idx = ci
                break
        years_on_aed = max(1, (r.get("age", 30) or 30) - onset_age)
        nocturnal = bool((sd >> 18) % 3 < 2)  # ~67% nocturnal predilection
        post_surgical = (control_idx == 0 and etiology_idx in [0, 1] and (sd >> 20) % 4 == 0)
        result.append({
            "patient_id": r["patient_id"],
            "sex": r.get("gender", "Unknown"),
            "onset_age_years": onset_age,
            "seizure_types": _SEIZURE_SUBTYPES[subtype_idx],
            "etiology": _ETIOLOGY_NAMES[etiology_idx],
            "onset_semiology": _ONSET_SEMIOLOGY[semiology_idx],
            "aed_regimen": _AED_REGIMEN[aed_idx],
            "seizure_control": _CONTROL[control_idx],
            "nocturnal_predilection": nocturnal,
            "catamenial": catamenial,
            "post_surgical": post_surgical,
            "years_on_aed": max(0, years_on_aed),
        })
    return result


# ─── Public API ─────────────────────────────────────────────────────────────

def overview():
    patients = _get_patients()
    n = max(len(patients), 1)
    seizure_free_n = sum(1 for p in patients if p["seizure_control"] == "Seizure-free")
    drug_resistant_n = sum(1 for p in patients if p["seizure_control"] == "Drug-resistant")
    post_surgical_n = sum(1 for p in patients if p["post_surgical"])
    nocturnal_n = sum(1 for p in patients if p["nocturnal_predilection"])
    catamenial_n = sum(1 for p in patients if p["catamenial"])
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
        "syndrome": "Frontal Lobe Epilepsy (FLE)",
        "icd10": "G40.109 / G40.119",
        "prevalence": "~20-30% of focal epilepsies; second most common focal epilepsy after TLE",
        "drug_resistance_rate": "~30-40% drug-resistant; similar rate to TLE",
        "updated": str(date.today()),
        "cohort_size": n,
        "avg_onset_age_years": avg_onset,
        "kpis": [
            {"label": "Cohort (FLE)", "value": str(n), "color": "#1d4ed8"},
            {"label": "Seizure-Free", "value": f"{seizure_free_n} ({round(seizure_free_n/n*100)}%)", "color": "#16a34a"},
            {"label": "Drug-Resistant", "value": f"{drug_resistant_n} ({round(drug_resistant_n/n*100)}%)", "color": "#dc2626"},
            {"label": "Post-Surgical", "value": f"{post_surgical_n} ({round(post_surgical_n/n*100)}%)", "color": "#7c3aed"},
            {"label": "Nocturnal Predilection", "value": f"{nocturnal_n} ({round(nocturnal_n/n*100)}%)", "color": "#0891b2"},
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
            "ENZYME INDUCERS (CBZ / OXC): FIRST-LINE for FLE (unlike JME/Dravet where CONTRAINDICATED); "
            "however, reduce OCP efficacy → non-hormonal or high-dose OCP mandatory for all women on CBZ/OXC",
            "NOCTURNAL FLE MISDIAGNOSIS: hypermotor seizures frequently misdiagnosed as parasomnias "
            "(night terrors, REM sleep behavior disorder) — video-EEG during NREM sleep is gold standard; "
            "document with home video; refer to sleep epilepsy unit",
            "SURGICAL REFERRAL: ≥2 AED failures = drug-resistant FLE → Level 4 epilepsy centre; "
            "SEEG mandatory (scalp EEG poorly localising for deep frontal sources); "
            "50-60% Engel Class I — lower than TLE but still significant benefit",
            "PERAMPANEL REMS: aggression/psychiatric BLACK BOX — PHQ-9 + aggression scale at every visit; "
            "frontal lobe dysfunction in FLE may amplify PER behavioural toxicity; "
            "caregiver interview mandatory; reduce dose at first behavioural signal",
            "LACOSAMIDE CARDIAC: PRE-TREATMENT ECG mandatory — PR interval prolongation; "
            "avoid in 2°/3° AV block; additive PR effect with CBZ/OXC/LTG polytherapy",
            "ADNFLE (GENETIC FLE): family history + adolescent onset + nocturnal hypermotor seizures → "
            "CHRNA4/CHRNB2/DEPDC5 gene panel urgently; CBZ is preferred first-line; "
            "misdiagnosis as psychiatric disorder is common — semiology video is diagnostic",
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
