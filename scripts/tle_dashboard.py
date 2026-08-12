"""Temporal Lobe Epilepsy (TLE) Dashboard — the most common form of focal (partial) epilepsy,
accounting for ~60% of focal epilepsies and ~30% of all epilepsies.

Hallmarks:
  Mesial TLE (MTLE): hippocampal sclerosis; déjà vu / epigastric aura → oral automatisms
  Lateral/Neocortical TLE: auditory aura (LGI1/ADTLE), visuo-spatial, sensory aura
EEG hallmark: temporal spike/sharp-wave, rhythmic theta in ictal phase (anterior-temporal > mid-temporal)
Seizure types: FAS (aura) → FIAS (automatisms) → FBTCS (generalised spread)
Drug resistance: ~30–40% — highest of any common epilepsy syndrome
Surgery: temporal lobectomy — 60–70% Engel Class I (seizure-free) in MTLE-HS (Wiebe 2001)

CAUTION — Enzyme-inducing AEDs (CBZ, OXC, PHT, PB):
  Reduce oral contraceptive efficacy → use non-hormonal or high-dose OCP
  Increase metabolism of many co-medications (warfarin, statins, immunosuppressants)
  Valproate: NOT first-line for focal epilepsy; teratogenic; use only if broad-spectrum needed

References:
  - Wiebe et al. 2001 NEJM (surgery vs medical therapy: 58% vs 8% seizure-free)
  - Engel et al. 2012 NEJM ERSET trial (early surgery superior)
  - Helmstaedter et al. 2003 Lancet (memory outcomes in TLE)
  - Jobst & Cascino 2015 JAMA Neurol (resective epilepsy surgery review)
  - Ryvlin & Rheims 2018 Lancet Neurol (SUDEP risk in TLE)
  - Scheffer et al. 2022 Epilepsia (ILAE classification)
Data: live clinical.db (41 epilepsy patients, deterministic TLE overlay)
      + curated TLE pharmacology / etiology / seizure-type / trigger catalogs."""

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
        "etiology": "Hippocampal Sclerosis (HS) / MTLE-HS",
        "category": "Structural",
        "pct": 50,
        "mechanism": (
            "Neuronal loss and gliosis predominantly in CA1 and CA3 subfields of the hippocampus; "
            "associated with prolonged febrile seizures in childhood (30–50% of MTLE-HS). "
            "mTOR pathway dysregulation, mossy fibre sprouting, and GABAergic reorganisation → "
            "recurrent synchronised hippocampal discharge."
        ),
        "mri_finding": "T2/FLAIR hippocampal hyperintensity + unilateral atrophy; volume loss on volumetry",
        "surgical_outcome": "Engel Class I: 60–70% at 2 years post-anterior temporal lobectomy (Wiebe 2001)",
        "clinical_note": "Most surgically remediable epilepsy; SEG/wada test pre-op if bilateral or ambiguous MRI"
    },
    {
        "etiology": "Low-Grade Brain Tumor (DNET, Ganglioglioma, Oligodendroglioma)",
        "category": "Structural",
        "pct": 15,
        "mechanism": (
            "Dysembryoplastic neuroepithelial tumors (DNET) and gangliogliomas are cortical tumors "
            "arising in the temporal lobe. Epileptogenicity from peritumoral cortex. "
            "BRAF V600E in 50% ganglioglioma; FGFR1 in DNET. Slow-growing — years before diagnosis."
        ),
        "mri_finding": "Cortical/subcortical lesion, often temporal pole or mesio-temporal; ring enhancement absent (low-grade)",
        "surgical_outcome": "Engel Class I: 80–90% with complete resection (lesionectomy ± cortectomy)",
        "clinical_note": "Excellent surgical prognosis; IDH/BRAF mutation status guides oncologic management; yearly MRI surveillance"
    },
    {
        "etiology": "Focal Cortical Dysplasia (FCD Type I / II)",
        "category": "Structural",
        "pct": 10,
        "mechanism": (
            "Malformation of cortical development. FCD Type IIb: balloon cells + dysmorphic neurons; "
            "mTOR pathway somatic mutations (MTOR, TSC1/TSC2, DEPDC5). "
            "FCD Type I: columnar/laminar disorganisation. Hyperexcitable dysplastic cortex → "
            "continuous or high-frequency interictal discharges; MRI may be subtle or negative."
        ),
        "mri_finding": "Blurring of grey-white junction, cortical thickening, transmantle sign (FCD IIb); MRI-negative in 30%",
        "surgical_outcome": "Engel Class I: 40–60% (lower than HS due to diffuse involvement and MRI-negativity)",
        "clinical_note": "SEEG mandatory when MRI-negative; mTOR inhibitor (everolimus) adjunctive in TSC-related FCD"
    },
    {
        "etiology": "Post-Encephalitic / Post-Infectious",
        "category": "Structural",
        "pct": 10,
        "mechanism": (
            "Herpes simplex encephalitis (HSE) most common viral cause; direct neuronal injury + "
            "inflammatory cascade → mesio-temporal scarring. Also: limbic encephalitis (anti-LGI1, "
            "anti-NMDAR), CMV, EBV in immunocompromised. "
            "Post-bacterial meningitis cortical injury less focal."
        ),
        "mri_finding": "Bilateral or unilateral mesio-temporal T2 signal (acute); chronic: atrophy + gliosis; anti-LGI1: basal ganglia FDG-PET hypermetabolism",
        "surgical_outcome": "Variable; bilateral involvement limits resective surgery; immunotherapy (IVIG/steroids/rituximab) first if autoimmune",
        "clinical_note": "Exclude autoimmune encephalitis in new-onset TLE: anti-LGI1, anti-CASPR2, anti-NMDAR, anti-GABA-B serology mandatory"
    },
    {
        "etiology": "Genetic — LGI1 / DEPDC5 / ADTLE",
        "category": "Genetic",
        "pct": 8,
        "mechanism": (
            "LGI1 (leucine-rich glioma-inactivated 1) GOF/autoimmune overlap — Autosomal Dominant Lateral TLE (ADTLE): "
            "auditory aura, focal aware seizures. DEPDC5 (mTOR pathway repressor) LOF → TLE with cortical dysplasia. "
            "Other: GRIN2A (Landau-Kleffner/CSWS spectrum), KCNQ2 overlap."
        ),
        "mri_finding": "Often MRI-normal (ADTLE); DEPDC5: subtle FCD; autoimmune LGI1: mesio-temporal T2 signal",
        "surgical_outcome": "LGI1 GOF: respond to AEDs; autoimmune LGI1: immunotherapy. DEPDC5-FCD: surgical if lesion identified",
        "clinical_note": "Family history of TLE → gene panel. ADTLE: auditory phenomena + family history = pathognomonic"
    },
    {
        "etiology": "Cryptogenic / MRI-Negative TLE",
        "category": "Unknown/Cryptogenic",
        "pct": 7,
        "mechanism": (
            "No identifiable structural lesion on 3T MRI with epilepsy protocol; presumed microscopic "
            "dysplasia, subtle HS, or heterotopia below MRI resolution. "
            "FDG-PET and SPECT add localising value. SEEG required for surgical candidacy."
        ),
        "mri_finding": "Normal 3T MRI (epilepsy protocol); FDG-PET may show temporal hypometabolism; MEG for source localisation",
        "surgical_outcome": "Engel Class I: 30–50% with tailored resection guided by SEEG + FDG-PET concordance",
        "clinical_note": "Refer to Level 4 epilepsy centre; consider ultra-high-field 7T MRI; FDG-PET + SEEG multimodal workup"
    },
]


# ─── Seizure type catalog ────────────────────────────────────────────────────

_SEIZURE_TYPES = [
    {
        "type": "Focal Aware Seizure (FAS) / Aura",
        "prevalence_pct": 80,
        "description": (
            "Consciousness fully preserved. Most common aura: déjà vu / jamais vu (65%), "
            "epigastric rising sensation (60%), fear/anxiety (35%), olfactory (10%), "
            "auditory hallucination (ADTLE/LGI1). Duration: seconds. Localising value is HIGH."
        ),
        "eeg": "Temporal spike-wave or focal theta (anterior > mid-temporal); may be subtle; "
               "scalp EEG negative in ~50% of mesio-temporal FAS",
        "clinical_tip": (
            "FAS = reliable lateralising sign: epigastric rising + déjà vu → ipsilateral MTLE; "
            "auditory aura → lateral TLE (superior temporal gyrus). Document aura semiology at each visit."
        ),
        "triggers": "Sleep deprivation, stress, missed dose, fever",
        "first_line_aed": "Carbamazepine / Lamotrigine"
    },
    {
        "type": "Focal Impaired Awareness Seizure (FIAS)",
        "prevalence_pct": 90,
        "description": (
            "Impaired consciousness + automatisms: oro-alimentary (lip smacking, chewing, swallowing — 65%), "
            "gestural (hand fumbling — 50%), verbal (repetitive speech/humming — 20%), "
            "ambulatory automatisms (30%). Postictal confusion 2–30 min. Ictal vocalisation: high-pitched cry at onset."
        ),
        "eeg": "Rhythmic 5–8 Hz temporal theta; focal temporal spike burst at onset; "
               "ictal spread to ipsilateral hemisphere precedes impaired awareness",
        "clinical_tip": (
            "Oro-alimentary automatisms with ipsilateral hand fumbling → MTLE ipsilateral. "
            "Contralateral dystonic posturing = strong lateralising sign (ipsilateral TLE). "
            "Postictal aphasia → dominant (left) hemisphere."
        ),
        "triggers": "Missed AED dose, fever, alcohol, sleep deprivation",
        "first_line_aed": "Carbamazepine / Lamotrigine / Levetiracetam"
    },
    {
        "type": "Focal to Bilateral Tonic-Clonic Seizure (FBTCS)",
        "prevalence_pct": 50,
        "description": (
            "Focal onset spreading to bilateral cortical involvement → generalised tonic-clonic. "
            "Highest SUDEP risk event type. Often preceded by FAS/FIAS. "
            "Postictal: 30–120 min confusion, headache, muscle soreness, Todd's paresis (5%)."
        ),
        "eeg": "Focal temporal ictal onset → rapid bilateral spread; postictal generalised suppression; "
               "postictal delta slowing over temporal region",
        "clinical_tip": (
            "FBTCS frequency is the primary driver of SUDEP risk. "
            "Any breakthrough FBTCS mandates AED review. "
            "Supervise patient; clear area; time seizure; recovery position post-ictally."
        ),
        "triggers": "Highest risk: sleep deprivation + alcohol + missed AED dose (triple trigger)",
        "first_line_aed": "Carbamazepine / Levetiracetam (add-on for FBTCS breakthrough)"
    },
    {
        "type": "Subclinical / Electrographic Seizure",
        "prevalence_pct": 30,
        "description": (
            "Ictal EEG discharge without overt clinical manifestation (subtle behavioral arrest, "
            "blank stare, mild automatisms). Detected on prolonged cEEG or ambulatory EEG. "
            "More common in MTLE-HS. Associated with cumulative cognitive injury."
        ),
        "eeg": "Temporal ictal rhythm (>10s) without clinical correlate; ictal DC shift on DC-coupled EEG; "
               "high-frequency oscillations (HFO 80–200 Hz) at seizure onset zone",
        "clinical_tip": (
            "Subclinical seizures contribute to cognitive decline in TLE — treat aggressively. "
            "Consider 72h ambulatory EEG if clinical seizure count doesn't match cognitive decline. "
            "HFO biomarkers guide SEEG resection zone mapping."
        ),
        "triggers": "Often unprovoked; worse with sleep disruption",
        "first_line_aed": "Optimise AED regimen; consider surgical evaluation"
    },
]


# ─── Trigger catalog ─────────────────────────────────────────────────────────

_TRIGGERS = [
    {
        "trigger": "Missed AED Doses",
        "frequency_pct": 70,
        "mechanism": "Sub-therapeutic AED plasma level → seizure threshold drops; rebound hyperexcitability within 12–24h of missed dose",
        "mitigation": "Smart pill dispenser; phone alarm; weekly blister pack; 30-day refill reminders; pharmacist medication reconciliation"
    },
    {
        "trigger": "Sleep Deprivation",
        "frequency_pct": 68,
        "mechanism": "Sleep loss reduces GABAergic inhibition and lowers cortical arousal threshold; REM sleep rebound increases temporal lobe excitability",
        "mitigation": "Minimum 7–9 hours nightly sleep; consistent wake time; CBT-I for insomnia; avoid night shifts; sleep diary"
    },
    {
        "trigger": "Psychological Stress",
        "frequency_pct": 65,
        "mechanism": "HPA axis activation → cortisol surge → CRH reduces seizure threshold; limbic system hyperreactivity; stress-induced neuroinflammation",
        "mitigation": "Mindfulness-based stress reduction (MBSR); CBT referral; biofeedback; regular aerobic exercise; social support network"
    },
    {
        "trigger": "Fever / Systemic Illness",
        "frequency_pct": 55,
        "mechanism": "Fever lowers seizure threshold (Na+ channel kinetics shifted); dehydration raises AED concentration variability; intercurrent illness alters AED absorption",
        "mitigation": "Aggressive antipyresis (paracetamol/ibuprofen) at first fever sign; rescue benzodiazepine protocol; adequate hydration; sick-day AED instructions"
    },
    {
        "trigger": "Alcohol Use",
        "frequency_pct": 45,
        "mechanism": "Acute alcohol: GABAergic potentiation (protective acutely); withdrawal rebound: NMDA upregulation + GABA downregulation → seizure threshold ↓ in 12–48h post-binge",
        "mitigation": "Abstinence recommended; if drinking: limit to 1–2 standard units/occasion; avoid binge drinking; never combine with enzyme-inhibiting AEDs"
    },
    {
        "trigger": "Catamenial (Perimenstrual)",
        "frequency_pct": 30,
        "mechanism": "Oestrogen-progesterone ratio changes at luteal-follicular transition → oestrogen excitatory (temporal lobe sensitisation); progesterone withdrawal reduces GABA-A neurosteroid modulation",
        "mitigation": "Seizure diary to confirm catamenial pattern; clobazam 10 mg/day perimenstrual (days -3 to +3); progesterone supplementation in refractory cases; gynaecology liaison"
    },
    {
        "trigger": "Physical Exertion",
        "frequency_pct": 25,
        "mechanism": "Hyperventilation (respiratory alkalosis) lowers CO₂ → cerebral vasoconstriction → relative ischaemia in epileptogenic zone; electrolyte derangement with heavy sweating",
        "mitigation": "Encourage supervised moderate exercise (reduces SUDEP risk overall); avoid hyper-ventilatory sports acutely; hydration and electrolyte replacement"
    },
    {
        "trigger": "Photosensitivity (rare in TLE)",
        "frequency_pct": 10,
        "mechanism": "Occipital visual cortex → temporal propagation in lateral TLE; rare in MTLE-HS. Screen with intermittent photic stimulation (IPS) at EEG if suspected.",
        "mitigation": "Polarised lenses; screen flicker reduction; avoid strobe lighting. Note: photosensitivity is rare in TLE — confirm EEG-PPR before restricting"
    },
]


# ─── Treatment catalog ───────────────────────────────────────────────────────

_TREATMENTS = [
    {
        "drug": "Carbamazepine (CBZ / Tegretol)",
        "type": "AED — First-line focal",
        "dose": "Start 200 mg BD; titrate by 200 mg/week; maintenance 400–1600 mg/day in 2–3 divided doses; XR formulation preferred",
        "moa": "Use-dependent Na+ channel blockade (Nav1.1, Nav1.2) → reduces high-frequency neuronal firing; also blocks L-type Ca²⁺ channels at higher doses",
        "efficacy": "Level A ILAE focal epilepsy; 50–60% seizure-free in new-onset TLE; reference standard for focal AED comparisons",
        "safety": "Hyponatremia (SIADH — 5–10%); SJS/TEN (HLA-B*1502 screen in Asian patients mandatory); enzyme inducer (CYP3A4/2C19); reduces OCP efficacy; agranulocytosis (rare)",
        "fda_status": "FDA-approved 1968; NICE first-line focal epilepsy; ILAE Level A evidence",
        "evidence_level": "Level A (ILAE 2017); class I–II RCTs"
    },
    {
        "drug": "Lamotrigine (LTG / Lamictal)",
        "type": "AED — First-line focal, preferred in women",
        "dose": "SLOW titration mandatory: 25 mg/day × 2 weeks → 50 mg/day × 2 weeks → increase by 50 mg/2 weeks; maintenance 100–400 mg/day; halve dose if adding VPA",
        "moa": "Voltage-gated Na+ channel blockade; reduces glutamate release; some Ca²⁺ channel activity; mechanism of slow titration: SJS prevention",
        "efficacy": "Comparable to CBZ for focal epilepsy (SANAD 2007); superior tolerability; preferred in elderly and women of childbearing age",
        "safety": "SJS/TEN (1:1000 — SLOW titration mandatory); LTG-VPA pharmacokinetic interaction (VPA doubles LTG level — halve LTG); maculopapular rash 10%; behavioural activation in children",
        "fda_status": "FDA-approved adjunctive 1994, monotherapy 1998; NICE first-line; SANAD trial equivalent to CBZ",
        "evidence_level": "Level A (ILAE 2017); SANAD trial (Marson 2007)"
    },
    {
        "drug": "Levetiracetam (LEV / Keppra)",
        "type": "AED — Broad-spectrum, favoured for add-on",
        "dose": "Start 500 mg BD; increase by 500 mg/week; maintenance 1000–3000 mg/day; renal dose adjustment (CrCl <80 mL/min)",
        "moa": "SV2A (synaptic vesicle protein 2A) ligand → reduces vesicular neurotransmitter release; modulates GABA-A receptor conformation; reduces high-voltage Ca²⁺ current",
        "efficacy": "40–50% ≥50% responder rate as add-on in DRE; 30–40% seizure-free in new-onset focal epilepsy; broad-spectrum (also effective generalised)",
        "safety": "Behavioural/mood side effects (irritability, aggression, depression — 15%); pre-existing psychiatric disorder = relative risk; minimal enzyme induction; safe in pregnancy (relative)",
        "fda_status": "FDA-approved adjunctive focal 1999; IV form 2006; NICE Level A; preferred if psychiatric monitoring available",
        "evidence_level": "Level B (ILAE 2017); multiple RCTs"
    },
    {
        "drug": "Oxcarbazepine (OXC / Trileptal)",
        "type": "AED — First-line focal",
        "dose": "Start 300 mg BD; increase by 300 mg/week; maintenance 600–2400 mg/day; XR/ESL preferred for tolerability",
        "moa": "Active metabolite MHD (monohydroxy derivative): Na+ channel blockade similar to CBZ but less auto-induction; some Ca²⁺ channel blockade",
        "efficacy": "Comparable to CBZ (ESLICAB trial); better tolerability (less enzyme induction); 10% cross-allergy with CBZ (caution in prior CBZ rash)",
        "safety": "Hyponatremia (10% — monitor Na+ especially elderly); mild enzyme inducer; 10% OCP interaction; SJS/TEN less common than CBZ; HLA-B*1502 cross-caution",
        "fda_status": "FDA-approved adjunctive 2000, monotherapy 2017; NICE first-line focal",
        "evidence_level": "Level A (ILAE 2017)"
    },
    {
        "drug": "Lacosamide (LCM / Vimpat)",
        "type": "AED — Adjunctive focal (ILAE Level A add-on)",
        "dose": "Start 50 mg BD; increase by 100 mg/week; maintenance 100–400 mg/day; IV form available for status epilepticus",
        "moa": "Enhances slow inactivation of voltage-gated Na+ channels (unique mechanism vs CBZ/OXC which target fast inactivation); CRMP-2 binding",
        "efficacy": "~30–40% ≥50% responder rate as add-on; superior to placebo in 3 Phase III trials; IV form for SE",
        "safety": "PR interval prolongation (pre-treatment ECG mandatory; avoid in 2°/3° AV block); dizziness, diplopia, ataxia (dose-related); avoid in severe cardiac conduction disease",
        "fda_status": "FDA Schedule V (CRMP-2 mechanism); FDA adjunctive focal 2008, monotherapy 2014; ESC cardiac monitoring recommendation",
        "evidence_level": "Level A add-on (ILAE 2017); BENMILD/BENMILD-2 RCTs"
    },
    {
        "drug": "Eslicarbazepine Acetate (ESL / Aptiom)",
        "type": "AED — Adjunctive / once-daily focal",
        "dose": "Start 400 mg once daily; increase to 800 mg after 1–2 weeks; maintenance 400–1200 mg once daily",
        "moa": "Pro-drug → active MHD (same as OXC); slower Na+ channel inactivation; lower drug interactions vs OXC; once-daily dosing = improved adherence",
        "efficacy": "30–35% ≥50% responder rate as add-on; comparable to OXC with better tolerability and adherence profile",
        "safety": "Hyponatremia (less than OXC); SJS/TEN rare; 10% cross-allergy with CBZ/OXC; dizziness, somnolence; once daily preferred for adherence",
        "fda_status": "FDA adjunctive focal 2013, monotherapy 2015; EMA approved 2009 (Zebinix)",
        "evidence_level": "Level A add-on (ILAE 2017)"
    },
    {
        "drug": "Temporal Lobectomy (Anterior / Selective Amygdalohippocampectomy)",
        "type": "Surgical — Curative first-line for MTLE-HS",
        "dose": (
            "Standard anterior temporal lobectomy (ATL): resect 4–6 cm temporal pole (non-dominant) / 3–4 cm (dominant). "
            "SAH: selective amygdalohippocampectomy spares neocortex → better neurocognitive outcomes."
        ),
        "moa": (
            "Removes the primary epileptogenic zone (hippocampus + amygdala + parahippocampal gyrus in MTLE). "
            "Pre-surgical workup: video-EEG, MRI-HS, FDG-PET, neuropsychology, WADA test / fMRI lateralisation, "
            "SEEG if MRI-negative or bilateral."
        ),
        "efficacy": (
            "MTLE-HS: Engel Class I 60–70% at 2y (Wiebe 2001 NEJM). Early surgery (ERSET trial): "
            "73% seizure-free at 2y vs 0% medical therapy. SAH comparable to ATL for seizure freedom."
        ),
        "safety": (
            "Memory decline (dominant ATL: verbal memory −10–20% in 30% patients); "
            "quadrantanopia (Meyer loop); infection/haemorrhage <1%; "
            "psychiatric deterioration risk (pre-op psychiatric assessment mandatory)"
        ),
        "fda_status": "ILAE Level A surgical evidence (2017); AAN Practice Guideline Level A (2003); long-term cost-effective vs medical management (>5 years)",
        "evidence_level": "Level A (ILAE Surgical Guidelines 2017); Wiebe 2001 NEJM RCT; Engel 2012 NEJM ERSET"
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "type": "Non-pharmacological — Adjunctive",
        "dose": "Classical 4:1 fat:protein+carb ratio; MCT diet (more palatable); LGIT (low glycaemic index); minimum 3-month trial; dietitian-supervised",
        "moa": "Ketone bodies (β-hydroxybutyrate) → GABA synthesis upregulation + glutamate downregulation; mTOR pathway inhibition; mitochondrial biogenesis; direct KATP channel opening",
        "efficacy": "20–30% ≥50% reduction in drug-resistant TLE; better evidence in children; adjunctive role; GLUT1 deficiency: first-line (reverses glucose transport defect)",
        "safety": "Hyperlipidaemia, nephrolithiasis (citrate supplement), constipation, growth retardation in children; acidosis; contraindicated in fatty acid oxidation disorders; carnitine supplementation",
        "fda_status": "No FDA approval as drug; ILAE recommended adjunctive (2009 consensus); pediatric epilepsy guidelines Level B",
        "evidence_level": "Level B (ILAE 2009 consensus); multiple RCTs in paediatric DRE"
    },
]


# ─── AED monitoring ──────────────────────────────────────────────────────────

_AED_MONITORING = [
    {
        "drug": "Carbamazepine (CBZ)",
        "category": "ENZYME INDUCER + HYPONATREMIA",
        "risk": "Hyponatremia (SIADH — 5–10%); agranulocytosis/aplastic anaemia (1:10,000–500,000); SJS/TEN (HLA-B*1502); enzyme induction → OCP failure, drug interactions",
        "monitoring": "Baseline + 6-monthly: CBC, LFTs, serum Na; TDM target 4–12 μg/mL; HLA-B*1502 before initiation in Han Chinese/SE Asian; ECG if cardiac history",
        "mitigation": "Slow titration (200 mg/week); use XR formulation (smoother levels); check Na if nausea/confusion; non-hormonal contraception (enzyme induction); HLA screen before prescribing in at-risk populations"
    },
    {
        "drug": "Lamotrigine (LTG)",
        "category": "SJS/TEN RISK — SLOW TITRATION MANDATORY",
        "risk": "SJS/TEN (Stevens-Johnson/Toxic Epidermal Necrolysis): 1:1000 adult; higher in children; risk ↑ with rapid titration and co-prescribing valproate (VPA doubles LTG level)",
        "monitoring": "Rash monitoring at every visit (first 8 weeks highest risk); TDM 3–15 μg/mL; pregnancy: LTG clearance increases 300% → dose escalation required throughout gestation; postpartum: rapid LTG level rise (toxicity risk)",
        "mitigation": "MANDATORY slow titration (see dose schedule); halve LTG dose when adding valproate; triple LTG dose if on enzyme-inducing AEDs (CBZ/PHT/PB); written rash action plan for patient; immediate discontinuation if muco-cutaneous rash"
    },
    {
        "drug": "Levetiracetam (LEV)",
        "category": "BEHAVIORAL MONITORING",
        "risk": "Irritability, aggression, depression, anxiety (15% of patients); psychiatric history = independent risk factor; suicidality (all AEDs — FDA class warning); renal clearance dependent → toxicity in CKD",
        "monitoring": "Mood/behaviour at every visit (PHQ-9 + GAD-7 screening); caregiver interview; CrCl at baseline and annually; TDM 12–46 μg/mL (optional — correlates poorly with effect)",
        "mitigation": "Start low (500 mg BD); psychoeducation for patient and family; add pyridoxine (B6) 50–100 mg/day (some evidence reduces LEV irritability); psychiatric co-management if prior mood disorder; dose reduction if severe behavioural change"
    },
    {
        "drug": "Lacosamide (LCM)",
        "category": "CARDIAC MONITORING — PR INTERVAL",
        "risk": "PR interval prolongation (dose-dependent); risk of 2°/3° AV block in pre-existing conduction disease; additive effect with other Na+ channel blockers (CBZ/OXC/LTG); dizziness/diplopia (dose-related)",
        "monitoring": "Pre-treatment ECG (MANDATORY); ECG at target dose; caution with sodium channel AED polytherapy; avoid in 2°/3° AV block, sick sinus syndrome",
        "mitigation": "Slow IV infusion (>15 min); check co-medications for PR-prolonging drugs; cardiology clearance if PR >200 ms; use oral form if IV not required; reduce dose at first signs of diplopia/ataxia"
    },
]


# ─── Lifecycle trajectory ────────────────────────────────────────────────────

_LIFECYCLE = [
    {
        "age_window": "Childhood (0–12 years)",
        "typical_profile": "First febrile seizure age 6–18 months in 30–50% of MTLE-HS; school-age onset of FAS/FIAS; MRI with epilepsy protocol; neuropsychology baseline",
        "tle_indicators": "Febrile status epilepticus (>30 min) → MTLE risk; temporal lobe tumour (DNET/ganglioglioma); behavioral/learning concerns from subclinical seizures",
        "action": "Epilepsy protocol MRI + prolonged EEG; IQ/memory baseline; early paediatric epilepsy centre referral; avoid polypharmacy; surgical evaluation if drug-resistant ≥2 AEDs"
    },
    {
        "age_window": "Adolescent (12–18 years)",
        "typical_profile": "MTLE-HS often presents first with FAS (aura) → FIAS evolution; academic decline; peer stigma; driving ineligibility; mental health (depression prevalence 30%)",
        "tle_indicators": "Classic aura onset (déjà vu, epigastric rising) → escalating to FIAS; first FBTCS; EEG temporal focus confirmed; memory testing (verbal and visual)",
        "action": "First-line AED trial (CBZ or LTG); sleep hygiene counselling; psychiatric screening (PHQ-A); driving legislation counselling; genetic testing if family history; surgical discussion if 2 AED failures"
    },
    {
        "age_window": "Young Adult (18–35 years)",
        "typical_profile": "Highest impact years: employment, driving, relationships; ~30% drug-resistant at this stage; surgical evaluation window is optimal (better neuroplasticity); mood/anxiety comorbidity peaks",
        "tle_indicators": "Drug-resistant (≥2 AED failures) → surgical referral urgently; FBTCS breakthrough → SUDEP risk; cognitive decline on neuropsychological testing",
        "action": "SURGICAL EVALUATION (Level 4 centre) if ≥2 AED failures; video-EEG + MRI + neuropsychology + WADA/fMRI; SUDEP counselling; depression/anxiety treatment (SSRIs safe with most AEDs); vocational rehabilitation"
    },
    {
        "age_window": "Childbearing / Women (18–45 years)",
        "typical_profile": "AED teratogenicity planning; enzyme-inducing AEDs reduce OCP efficacy; LTG levels rise 300% in pregnancy; catamenial TLE (30% of women); postpartum seizure risk with sleep deprivation",
        "tle_indicators": "Catamenial pattern (seizure diary); unplanned pregnancy on teratogenic AED; postpartum breakthrough seizure with sleep deprivation; memory concerns (dominant TLE)",
        "action": "Pre-conception counselling: avoid valproate/CBZ (teratogenic); prefer LTG/LEV; folic acid 5 mg/day; non-hormonal contraception with enzyme inducers; LTG TDM throughout pregnancy; obstetric neurology co-management; breastfeeding: most AEDs compatible"
    },
    {
        "age_window": "Middle Adult (35–60 years)",
        "typical_profile": "Cumulative cognitive decline from TLE + AEDs; verbal memory most affected (dominant MTLE); potential late-onset surgery candidate; comorbid depression/anxiety well-established; career impact",
        "tle_indicators": "Accelerating memory complaints; breakthrough seizures with stress/ageing; review AED polypharmacy (rationalise); cognitive neuropsychology reassessment",
        "action": "Neuropsychological reassessment (verbal/visual memory, executive function); AED rationalisation (minimise polypharmacy); re-evaluate surgical candidacy (never too late for MTLE-HS); mood disorder management; occupational therapy if cognitive decline"
    },
    {
        "age_window": "Older Adult (60+ years)",
        "typical_profile": "New-onset TLE in elderly: consider structural (tumour, vascular) and autoimmune (anti-LGI1 peaks 50–70 years). Falls risk with sedating AEDs. Polypharmacy. Cognitive decline multifactorial.",
        "tle_indicators": "New-onset temporal lobe seizures (new MRI/CSF; autoimmune encephalitis screen); CBZ/OXC hyponatremia risk ↑ with diuretics; anti-LGI1 limbic encephalitis (faciobrachial dystonic seizures, FBDS)",
        "action": "Anti-LGI1/CASPR2/NMDAR serology; brain MRI with gadolinium (tumour/vascular); avoid sedating AEDs (benzodiazepines, barbiturates); prefer LTG or LEV (renal dose); fall prevention; osteoporosis screening (enzyme-inducing AEDs deplete Vit D/Ca²⁺)"
    },
]


# ─── Key definitions ─────────────────────────────────────────────────────────

_DEFINITIONS = [
    {"term": "TLE", "definition": "Temporal Lobe Epilepsy — focal epilepsy with seizure onset in temporal lobe structures; ~60% of focal epilepsies; most common drug-resistant epilepsy syndrome"},
    {"term": "MTLE", "definition": "Mesial Temporal Lobe Epilepsy — TLE with hippocampal/amygdala focus; associated with hippocampal sclerosis (HS); classic: déjà vu aura → oral automatisms → postictal confusion"},
    {"term": "Hippocampal Sclerosis (HS)", "definition": "Neuronal loss and reactive gliosis in hippocampal subfields (CA1 > CA3 > CA4); MRI: T2 signal + atrophy; most common cause of drug-resistant TLE"},
    {"term": "FAS (Focal Aware Seizure)", "definition": "Formerly 'simple partial seizure'; consciousness fully preserved; corresponds to the clinical aura; high localising value for epileptogenic zone"},
    {"term": "FIAS (Focal Impaired Awareness Seizure)", "definition": "Formerly 'complex partial seizure'; impaired consciousness + automatisms (oro-alimentary/gestural); bilateral temporal involvement → impaired awareness"},
    {"term": "FBTCS (Focal to Bilateral Tonic-Clonic)", "definition": "Formerly 'secondarily generalised seizure'; focal temporal onset → bilateral tonic-clonic spread; highest SUDEP risk seizure type"},
    {"term": "Aura", "definition": "Subjective ictal experience during FAS; déjà vu, epigastric rising, fear, olfactory are most common in TLE; lateralising value: ipsilateral to seizure onset"},
    {"term": "Automatism", "definition": "Involuntary, purposeless, stereotyped motor activity during FIAS; oro-alimentary (lip smacking, chewing) and gestural (hand fumbling) most common in TLE"},
    {"term": "Engel Classification", "definition": "Surgical outcome scale: Class I = seizure-free (worthwhile improvement); Class II = rare seizures; Class III = worthwhile improvement; Class IV = no worthwhile improvement"},
    {"term": "ADTLE", "definition": "Autosomal Dominant Lateral TLE; LGI1 gene (GOF mutation); auditory aura (buzzing/voices) + family history; responds well to AEDs; excellent prognosis"},
    {"term": "Drug-Resistant Epilepsy (DRE)", "definition": "Failure of ≥2 adequate, appropriate, tolerated AED schedules (ILAE 2010 definition); ~30% of TLE; surgical evaluation mandatory at this threshold"},
    {"term": "SEEG (Stereo-EEG)", "definition": "Intracranial depth electrode implantation for 3D mapping of seizure onset zone; used when MRI-negative or bilateral; guides tailored surgical resection"},
    {"term": "SUDEP", "definition": "Sudden Unexpected Death in Epilepsy; risk 1:1000/year in TLE; highest risk: uncontrolled FBTCS, nocturnal seizures, prone position; risk reduced by seizure freedom + supervised sleep"},
    {"term": "HFO (High-Frequency Oscillations)", "definition": "EEG oscillations 80–500 Hz; ripples (80–250 Hz) and fast ripples (250–500 Hz); biomarker of epileptogenic zone; predicts surgical outcome when resected"},
]


# ─── Clinical standards ──────────────────────────────────────────────────────

_STANDARDS = [
    {"name": "ILAE Classification 2022", "body": "ILAE", "scope": "Seizure type (FAS/FIAS/FBTCS) and epilepsy syndrome classification; TLE defined as focal epilepsy with temporal lobe onset"},
    {"name": "NICE NG217 (2022)", "body": "NICE UK", "scope": "First-line: CBZ or LTG for focal epilepsy; refer to Level 4 epilepsy centre after 2 AED failures; anti-LGI1 workup in new-onset limbic encephalitis"},
    {"name": "AAN Surgical Guideline (2003)", "body": "AAN / AES", "scope": "Level A: ATL superior to continued medical therapy for MTLE-HS; refer for presurgical evaluation after 2 AED failures"},
    {"name": "ILAE Surgical Outcome Classification (2017)", "body": "ILAE", "scope": "Engel Class I–IV for reporting surgical seizure outcomes; minimum 2-year follow-up required for Class I outcome claim"},
    {"name": "ILAE Drug-Resistant Epilepsy Definition (2010)", "body": "ILAE", "scope": "DRE = failure of ≥2 adequate, appropriately chosen AED schedules; triggers mandatory surgical evaluation referral"},
    {"name": "FDA Lacosamide Labelling (2008)", "body": "FDA", "scope": "LCM: Schedule V controlled substance; pre-treatment ECG for PR monitoring; IV dose: max 300 mg/day in non-oral patients; avoid in 2°/3° AV block"},
]


# ─── Key thresholds ──────────────────────────────────────────────────────────

_THRESHOLDS = [
    {"parameter": "DRE referral threshold", "threshold": "≥2", "unit": "AED failures", "significance": "Mandatory surgical evaluation referral after ≥2 adequate AED trials fail (ILAE 2010 DRE definition)"},
    {"parameter": "CBZ therapeutic range", "threshold": "4–12", "unit": "μg/mL", "significance": "Serum TDM target range; >12 μg/mL: toxicity (diplopia, ataxia, hyponatremia); check at steady state"},
    {"parameter": "LTG-VPA interaction", "threshold": "50%", "unit": "LTG dose reduction", "significance": "Halve LTG maintenance dose when adding valproate (VPA inhibits LTG glucuronidation → LTG level doubles → SJS risk)"},
    {"parameter": "MTLE-HS surgical outcome", "threshold": "60–70", "unit": "% Engel Class I", "significance": "Expected seizure-free rate at 2y post-ATL for MTLE-HS; ERSET trial: 73% early surgery vs 0% medical therapy"},
    {"parameter": "Driving seizure-free period", "threshold": "12", "unit": "months", "significance": "UK/Canada: 12 months seizure-free for private vehicle; USA: 3–6 months (state-variable); FBTCS = highest risk to driving ability"},
    {"parameter": "Folic acid (women of childbearing age)", "threshold": "5", "unit": "mg/day", "significance": "High-dose folic acid for women on enzyme-inducing AEDs (CBZ/OXC/PHT/PB/TPM); start pre-conception; reduce NTD risk"},
]


# ─── References ──────────────────────────────────────────────────────────────

_REFERENCES = [
    {
        "citation": "Wiebe et al. 2001 NEJM (Surgery for TLE RCT)",
        "key_finding": "First RCT: 58% seizure-free with ATL vs 8% medical therapy at 1 year; surgery significantly superior — Level A evidence for temporal lobectomy"
    },
    {
        "citation": "Engel et al. 2012 NEJM (ERSET Trial)",
        "key_finding": "Early surgical intervention (after 2 AED failures): 73% seizure-free at 2y vs 0% continued medical therapy; supports early referral for surgical evaluation"
    },
    {
        "citation": "Helmstaedter et al. 2003 Lancet",
        "key_finding": "Memory outcomes in TLE: surgical patients showed initial verbal memory decline then stabilisation; medical therapy = progressive memory decline — surgery better long-term"
    },
    {
        "citation": "Jobst & Cascino 2015 JAMA Neurol",
        "key_finding": "Systematic review of resective epilepsy surgery: 66% Engel I for TLE overall; SAH comparable to ATL for seizure freedom with better neurocognitive profile"
    },
    {
        "citation": "Ryvlin & Rheims 2018 Lancet Neurol",
        "key_finding": "SUDEP risk in TLE: uncontrolled FBTCS = primary risk factor; nocturnal seizures + prone position = highest-risk configuration; seizure freedom reduces SUDEP risk 27-fold"
    },
    {
        "citation": "Scheffer et al. 2022 Epilepsia (ILAE Classification)",
        "key_finding": "Updated ILAE operational classification: FAS/FIAS/FBTCS terminology adopted; TLE classified as 'focal epilepsy with or without known etiology' (structural/genetic/unknown)"
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
        "FAS only (aura)",
        "FAS + FIAS",
        "FIAS only",
        "FAS + FIAS + FBTCS",
        "FBTCS + subclinical",
    ]
    _AED_REGIMEN = [
        "Carbamazepine mono",
        "Lamotrigine mono",
        "Levetiracetam mono",
        "Carbamazepine + Lamotrigine",
        "Lamotrigine + Levetiracetam",
        "Oxcarbazepine + Levetiracetam",
        "Lacosamide + Lamotrigine",
    ]
    _CONTROL = ["Seizure-free", "Improved (>50% reduction)", "Partial response", "Drug-resistant"]
    _ETIOLOGY_NAMES = [e["etiology"].split(" (")[0].split(" /")[0] for e in _ETIOLOGIES]
    _AURA_TYPES = ["Déjà vu", "Epigastric rising", "Fear/anxiety", "Olfactory", "Auditory (ADTLE)", "None"]

    result = []
    for i, r in enumerate(rows[:41]):
        sd = _seed(r["patient_id"])
        etiology_idx = sd % len(_ETIOLOGIES)
        # Bias toward HS (~50%): if sd%10 < 5, use HS (index 0)
        if (sd % 10) < 5:
            etiology_idx = 0
        onset_age = 15 + (sd % 35)  # 15–50 years (wide TLE onset range)
        subtype_idx = (sd >> 4) % len(_SEIZURE_SUBTYPES)
        aed_idx = (sd >> 8) % len(_AED_REGIMEN)
        aura_idx = (sd >> 10) % len(_AURA_TYPES)
        catamenial = r.get("gender", "M") == "F" and bool((sd >> 14) % 3 == 0)
        control_w = [45, 25, 15, 15]  # TLE: ~30% drug-resistant
        control_idx = 0
        rv = (sd >> 16) % 100
        acc = 0
        for ci, w in enumerate(control_w):
            acc += w
            if rv < acc:
                control_idx = ci
                break
        years_on_aed = max(1, (r.get("age", 35) or 35) - onset_age)
        post_surgical = (control_idx == 0 and etiology_idx == 0 and (sd >> 20) % 3 == 0)
        result.append({
            "patient_id": r["patient_id"],
            "sex": r.get("gender", "Unknown"),
            "onset_age_years": onset_age,
            "seizure_types": _SEIZURE_SUBTYPES[subtype_idx],
            "etiology": _ETIOLOGY_NAMES[etiology_idx],
            "aura": _AURA_TYPES[aura_idx],
            "aed_regimen": _AED_REGIMEN[aed_idx],
            "seizure_control": _CONTROL[control_idx],
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
    aed_use = [{"regimen": k, "n_patients": v} for k, v in sorted(aed_counts.items(), key=lambda x: -x[1])]

    control_counts = {}
    for p in patients:
        k = p["seizure_control"]
        control_counts[k] = control_counts.get(k, 0) + 1
    control_distribution = [{"status": k, "count": v} for k, v in sorted(control_counts.items(), key=lambda x: -x[1])]

    aura_counts = {}
    for p in patients:
        k = p["aura"]
        aura_counts[k] = aura_counts.get(k, 0) + 1
    aura_distribution = [{"aura": k, "n": v} for k, v in sorted(aura_counts.items(), key=lambda x: -x[1])]

    return {
        "syndrome": "Temporal Lobe Epilepsy (TLE)",
        "icd10": "G40.209 / G40.219",
        "prevalence": "~60% of focal epilepsies; ~30% of all epilepsies; ~6 per 1000 population",
        "drug_resistance_rate": "~30–40% drug-resistant — highest of any common epilepsy syndrome",
        "updated": str(date.today()),
        "cohort_size": n,
        "avg_onset_age_years": avg_onset,
        "kpis": [
            {"label": "Cohort (TLE)", "value": str(n), "color": "#1d4ed8"},
            {"label": "Seizure-Free", "value": f"{seizure_free_n} ({round(seizure_free_n/n*100)}%)", "color": "#16a34a"},
            {"label": "Drug-Resistant", "value": f"{drug_resistant_n} ({round(drug_resistant_n/n*100)}%)", "color": "#dc2626"},
            {"label": "Post-Surgical", "value": f"{post_surgical_n} ({round(post_surgical_n/n*100)}%)", "color": "#7c3aed"},
            {"label": "Catamenial (F)", "value": f"{catamenial_n}", "color": "#db2777"},
            {"label": "Avg Onset Age", "value": f"{avg_onset}y", "color": "#0891b2"},
        ],
        "etiology_distribution": etiology_distribution,
        "aed_use": aed_use,
        "control_distribution": control_distribution,
        "aura_distribution": aura_distribution,
        "seizure_types": _SEIZURE_TYPES,
        "triggers": _TRIGGERS,
        "clinical_alerts": [
            "ENZYME INDUCERS (CBZ / OXC / PHT / PB): reduce oral contraceptive (OCP) efficacy — prescribe non-hormonal or high-dose OCP; counsel all women of childbearing age",
            "SURGICAL REFERRAL MANDATORY: ≥2 AED failures = drug-resistant epilepsy (DRE) → immediate Level 4 epilepsy centre referral for presurgical evaluation (ILAE 2010 / AAN 2003)",
            "SUDEP RISK: uncontrolled FBTCS = highest SUDEP risk; nocturnal unsupervised seizures highest risk configuration — counsel patient and caregivers; supervised sleep; prone position avoidance",
            "LTG TITRATION: SLOW titration mandatory (25 mg/day × 2 weeks → gradual increase) — rapid titration → SJS/TEN; HALVE LTG dose when adding valproate (VPA doubles LTG plasma level)",
            "LACOSAMIDE CARDIAC: PRE-TREATMENT ECG mandatory — lacosamide prolongs PR interval; contraindicated in 2°/3° AV block; avoid combination with other Na+ channel AEDs without cardiac monitoring",
            "ANTI-LGI1/AUTOIMMUNE: New-onset TLE in adults (esp. 50–70y) with faciobrachial dystonic seizures (FBDS) → anti-LGI1/CASPR2/NMDAR panel urgently — immunotherapy within days improves outcome",
        ],
    }


def breakdown():
    patients = _get_patients()
    return {
        "patients": patients,
        "etiologies": _ETIOLOGIES,
        "seizure_types": _SEIZURE_TYPES,
        "triggers": _TRIGGERS,
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
